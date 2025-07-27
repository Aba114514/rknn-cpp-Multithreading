#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <string>
#include <vector>
#include <iostream>
#include <stdlib.h> // For getenv()

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "Yolo11.hpp"
#include "rknnPool.hpp"

// 定义输出模式
enum class OutputMode {
    WINDOW_DISPLAY, // 在桌面窗口中显示 (xvimagesink)
    KMS_DISPLAY,    // 全屏硬件显示 (kmssink)
    RTP_STREAM      // RTP推流
};

/**
 * @brief 检查程序是否在图形化桌面环境中运行
 * @return 如果检测到桌面环境 (X11 or Wayland) 则返回 true, 否则返回 false
 */
bool isDesktopEnvironmentAvailable() {
    // 检查 X11 的 DISPLAY 环境变量
    const char* display = getenv("DISPLAY");
    if (display != nullptr && display[0] != '\0') {
        return true;
    }
    // 检查 Wayland 的 WAYLAND_DISPLAY 环境变量
    const char* wayland_display = getenv("WAYLAND_DISPLAY");
    if (wayland_display != nullptr && wayland_display[0] != '\0') {
        return true;
    }
    return false;
}

int main(int argc, char **argv)
{
    // --- 参数解析 ---
    if (argc < 3) {
        printf("Usage: %s <rknn model> <video_path | camera_id> [options]\n", argv[0]);
        printf("Options:\n");
        printf("  --stream rtp://<ip>:<port>   Enable RTP streaming mode.\n");
        printf("Display mode is detected automatically:\n");
        printf("  - In a desktop environment, a window will be used (xvimagesink).\n");
        printf("  - In a TTY (console), fullscreen hardware display will be used (kmssink).\n");
        return -1;
    }

    char *model_name = argv[1];
    char *video_name = argv[2];
    // 默认显示模式，后续会自动判断
    OutputMode output_mode = OutputMode::WINDOW_DISPLAY;
    std::string rtp_url;
    bool is_streaming = false;

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--stream" && (i + 1) < argc) {
            is_streaming = true;
            rtp_url = argv[i + 1];
            i++;
        }
    }

    // --- 关键修改：自动检测显示模式 ---
    if (is_streaming) {
        output_mode = OutputMode::RTP_STREAM;
    } else {
        if (isDesktopEnvironmentAvailable()) {
            output_mode = OutputMode::WINDOW_DISPLAY;
            printf("Desktop environment detected. Using windowed display (xvimagesink).\n");
        } else {
            output_mode = OutputMode::KMS_DISPLAY;
            printf("No desktop environment detected. Using fullscreen hardware display (kmssink).\n");
        }
    }

    // --- 初始化模型线程池 ---
    int threadNum = 3;
    rknnPool<Yolo11, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    // --- 初始化视频捕捉 ---
    cv::VideoCapture capture;
    std::string video_source = video_name;

    if (video_source.length() == 1 && isdigit(video_source[0])) {
        // 在 v4l2src 后添加 queue 以增强稳定性
        std::string gst_pipeline = "v4l2src device=/dev/video0 ! queue ! image/jpeg,width=1920,height=1080 ! mppjpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink";
        printf("Using GStreamer pipeline for camera: %s\n", gst_pipeline.c_str());
        capture.open(gst_pipeline, cv::CAP_GSTREAMER);
    } else {
        // 本地文件硬件加速输入管线
        std::string gst_pipeline = "filesrc location=" + video_source + " ! qtdemux ! h264parse ! mppvideodec ! videoconvert ! video/x-raw,format=BGR ! appsink";
        printf("Using GStreamer pipeline for video file: %s\n", gst_pipeline.c_str());
        capture.open(gst_pipeline, cv::CAP_GSTREAMER);
    }

    if (!capture.isOpened()) {
        fprintf(stderr, "Error: Could not open video source: %s\n", video_name);
        return -1;
    }

    // --- 动态获取视频流的实际尺寸 ---
    const int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    const int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps_ref = 30.0;

    if (frame_width == 0 || frame_height == 0) {
        fprintf(stderr, "Error: Frame width or height is 0. The source may be invalid or the pipeline failed.\n");
        return -1;
    }
    printf("Successfully opened source with resolution: %dx%d\n", frame_width, frame_height);

    // --- 根据自动检测的模式和动态尺寸初始化输出 ---
    cv::VideoWriter video_writer;
    std::string gst_output_pipeline;

    switch (output_mode) {
        case OutputMode::WINDOW_DISPLAY:
            gst_output_pipeline = "appsrc ! videoconvert ! queue ! xvimagesink sync=false";
            break;

        case OutputMode::KMS_DISPLAY:
            gst_output_pipeline = "appsrc ! videoconvert ! kmssink";
            printf("Hint: KMS mode requires running with sudo.\n");
            break;

        case OutputMode::RTP_STREAM:
            std::string host = rtp_url.substr(rtp_url.find("://") + 3, rtp_url.find(":", rtp_url.find("://") + 3) - (rtp_url.find("://") + 3));
            std::string port = rtp_url.substr(rtp_url.find(":", rtp_url.find("://") + 3) + 1);
            long long bps_value = 6000000; // 6 Mbps
            gst_output_pipeline = "appsrc ! videoconvert ! mpph264enc bps=" + std::to_string(bps_value) + " ! h264parse ! rtph264pay config-interval=1 ! udpsink host=" + host + " port=" + port;
            break;
    }

    printf("GStreamer Output Pipeline: %s\n", gst_output_pipeline.c_str());
    video_writer.open(gst_output_pipeline, cv::CAP_GSTREAMER, 0, fps_ref, cv::Size(frame_width, frame_height), true);
    if (!video_writer.isOpened()) {
        fprintf(stderr, "Error: Could not open VideoWriter for the selected output mode.\n");
        return -1;
    }

    // --- 主循环 ---
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int frames = 0;
    auto beforeTime = startTime;

    while (true)
    {
        cv::Mat img;
        if (!capture.read(img)) {
            printf("End of video stream.\n");
            break;
        }

        if (testPool.put(img) != 0)
            break;

        if (frames >= threadNum && testPool.get(img) != 0)
            break;

        video_writer.write(img);

        frames++;

        if (frames % 120 == 0) {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("Average FPS over 120 frames:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }

    // 清理剩余的帧
    while(true)
    {
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;
        video_writer.write(img);
        frames++;
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    printf("\n--- Final Stats ---\n");
    printf("Total frames processed: %d\n", frames);
    printf("Total time: %lld ms\n", endTime - startTime);
    if (endTime > startTime) {
        printf("Overall Average FPS: %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);
    }

    capture.release();
    video_writer.release();

    return 0;
}

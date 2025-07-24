#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <string>
#include <vector>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp" // For cv::VideoWriter
#include "Yolo11.hpp"
#include "rknnPool.hpp"

// 定义输出模式
enum class OutputMode {
    DISPLAY, // 本地显示
    RTP_STREAM // RTP推流
};

int main(int argc, char **argv)
{
    // --- 参数解析 ---
    if (argc < 3) {
        printf("Usage: %s <rknn model> <video_path | camera_id> [--stream rtp://<ip>:<port>]\n", argv[0]);
        return -1;
    }

    char *model_name = argv[1];
    char *video_name = argv[2];
    OutputMode output_mode = OutputMode::DISPLAY;
    std::string rtp_url;

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--stream" && (i + 1) < argc) {
            output_mode = OutputMode::RTP_STREAM;
            rtp_url = argv[i + 1];
            i++; // 跳过URL参数
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
        std::string gst_pipeline = "libcamerasrc ! image/jpeg,width=1920,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink";
        printf("Using GStreamer pipeline for camera: %s\n", gst_pipeline.c_str());
        capture.open(gst_pipeline, cv::CAP_GSTREAMER);
    } else {
        printf("Opening video file: %s\n", video_source.c_str());
        capture.open(video_source);
    }

    if (!capture.isOpened()) {
        fprintf(stderr, "Error: Could not open video source: %s\n", video_name);
        return -1;
    }

    // --- 根据模式初始化输出 (显示窗口或推流) ---
    cv::VideoWriter video_writer;

    if (output_mode == OutputMode::DISPLAY) {
        cv::namedWindow("Camera FPS", cv::WINDOW_AUTOSIZE);
        printf("Mode: Local Display\n");
    } else {
        // 从RTP URL中解析IP和端口
        std::string host;
        int port = 0;
        size_t pos1 = rtp_url.find("://");
        if (pos1 != std::string::npos) {
            size_t pos2 = rtp_url.find(":", pos1 + 3);
            if (pos2 != std::string::npos) {
                host = rtp_url.substr(pos1 + 3, pos2 - (pos1 + 3));
                port = std::stoi(rtp_url.substr(pos2 + 1));
            }
        }

        if (host.empty() || port == 0) {
            fprintf(stderr, "Invalid RTP URL format. Use rtp://<ip>:<port>\n");
            return -1;
        }

        // 获取视频属性
        int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = capture.get(cv::CAP_PROP_FPS);

        // GStreamer推流管线
        std::string gst_rtp_pipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=" + host + " port=" + std::to_string(port);

        printf("Mode: RTP Stream\n");
        printf("Streaming to: %s\n", rtp_url.c_str());
        printf("GStreamer Pipeline: %s\n", gst_rtp_pipeline.c_str());

        video_writer.open(gst_rtp_pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(frame_width, frame_height), true);
        if (!video_writer.isOpened()) {
            fprintf(stderr, "Error: Could not open VideoWriter for RTP streaming.\n");
            fprintf(stderr, "Hint: Please check if your OpenCV was built with GStreamer support and GStreamer plugins (especially x264enc) are installed.\n");
            return -1;
        }
    }

    // --- 主循环 ---
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int frames = 0;
    auto beforeTime = startTime;

    while (capture.isOpened())
    {
        cv::Mat img;
        if (!capture.read(img))
            break;

        if (testPool.put(img) != 0)
            break;

        if (frames >= threadNum && testPool.get(img) != 0)
            break;

        // --- 根据模式输出 ---
        if (output_mode == OutputMode::DISPLAY) {
            cv::imshow("Camera FPS", img);
            if (cv::waitKey(1) == 'q')
                break;
        } else {
            video_writer.write(img);
        }

        frames++;

        if (frames % 120 == 0) {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("Average FPS over 120 frames:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }

    // --- 清理剩余帧 ---
    while (true)
    {
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;

        if (output_mode == OutputMode::DISPLAY) {
            cv::imshow("Camera FPS", img);
            if (cv::waitKey(1) == 'q')
                break;
        } else {
            video_writer.write(img);
        }
        frames++;
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Total frames: %d\n", frames);
    printf("Total time: %lld ms\n", endTime - startTime);
    printf("Overall Average FPS:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    // 释放资源
    capture.release();
    if (output_mode == OutputMode::DISPLAY) {
        cv::destroyAllWindows();
    } else {
        video_writer.release();
    }

    return 0;
}
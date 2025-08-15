// फाइल का नाम: src/main.cc 

#include <stdio.h> 
#include <memory> 
#include <sys/time.h> 
#include <string> 
#include <vector> 
#include <iostream> 
#include <queue> 
#include <stdlib.h> // For getenv() 
#include <ctype.h>  // For isdigit() 

#include "opencv2/core/core.hpp" 
#include "opencv2/videoio.hpp" 
#include "opencv2/highgui.hpp" 
#include "opencv2/imgproc.hpp" // For cvtColor 

#include "Yolo11.hpp" 
#include "rknnPool.hpp" 
#include "Visualizer.hpp" 
#include "postprocess.h" 

// 定义输出模式 
enum class OutputMode { 
    WINDOW_DISPLAY, 
    KMS_DISPLAY, 
    RTP_STREAM 
}; 

/** * @brief 检查程序是否在图形化桌面环境中运行 
 * @return 如果检测到桌面环境 (X11 or Wayland) 则返回 true, 否则返回 false 
 */ 
bool isDesktopEnvironmentAvailable() { 
    const char* display = getenv("DISPLAY"); 
    if (display != nullptr && display[0] != '\0') { 
        return true; 
    } 
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
        printf("  --stream rtp://<ip>:<port>    Enable RTP streaming mode.\n"); 
        printf("Display mode is detected automatically.\n"); 
        return -1; 
    } 

    char *model_name = argv[1]; 
    char *video_name = argv[2]; 
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

    if (is_streaming) { 
        output_mode = OutputMode::RTP_STREAM; 
    } else { 
        if (isDesktopEnvironmentAvailable()) { 
            output_mode = OutputMode::WINDOW_DISPLAY; 
            printf("Desktop environment detected. Using windowed display.\n"); 
        } else { 
            output_mode = OutputMode::KMS_DISPLAY; 
            printf("No desktop environment detected. Using fullscreen hardware display (kmssink).\n"); 
        } 
    } 

    // --- 初始化模型线程池 --- 
    int threadNum = 3; 
    rknnPool<Yolo11, cv::Mat, object_detect_result_list> infer_pool(model_name, threadNum); 
    if (infer_pool.init() != 0) 
    { 
        printf("rknnPool init fail!\n"); 
        return -1; 
    } 

    init_post_process(); 

    // --- 初始化视频捕捉 --- 
    cv::VideoCapture capture; 
    std::string video_source = video_name; 

    if (video_source.length() == 1 && isdigit(video_source[0])) { 
        std::string gst_pipeline = "v4l2src device=/dev/video" + video_source + " ! queue ! image/jpeg,width=1920,height=1080 ! mppjpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink"; 
        printf("Using GStreamer pipeline for camera: %s\n", gst_pipeline.c_str()); 
        capture.open(gst_pipeline, cv::CAP_GSTREAMER); 
    } else { 
        std::string gst_pipeline = "filesrc location=" + video_source + " ! qtdemux ! h264parse ! mppvideodec ! videoconvert ! video/x-raw,format=BGR ! appsink"; 
        printf("Using GStreamer pipeline for video file: %s\n", gst_pipeline.c_str()); 
        capture.open(gst_pipeline, cv::CAP_GSTREAMER); 
    } 

    if (!capture.isOpened()) { 
        fprintf(stderr, "Error: Could not open video source: %s\n", video_name); 
        return -1; 
    } 

    const int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH); 
    const int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT); 
    const double fps_ref = capture.get(cv::CAP_PROP_FPS) > 0 ? capture.get(cv::CAP_PROP_FPS) : 30.0; 

    if (frame_width == 0 || frame_height == 0) { 
        fprintf(stderr, "Error: Frame width or height is 0.\n"); 
        return -1; 
    } 
    // printf("Successfully opened source with resolution: %dx%d @ %f FPS\n", frame_width, frame_height, fps_ref); 

    // --- 初始化视频输出 --- 
    cv::VideoWriter video_writer; 
    std::string gst_output_pipeline; 

    switch (output_mode) { 
        case OutputMode::WINDOW_DISPLAY: 
            gst_output_pipeline = "appsrc ! videoconvert ! queue ! xvimagesink sync=false"; 
            break; 
        case OutputMode::KMS_DISPLAY: 
            gst_output_pipeline = "appsrc ! videoconvert ! kmssink"; 
            printf("Hint: KMS mode may require running with sudo.\n"); 
            break; 
        case OutputMode::RTP_STREAM: 

            /* std::string host = rtp_url.substr(rtp_url.find("://") + 3, rtp_url.find(":", rtp_url.find("://") + 3) - (rtp_url.find("://") + 3)); 
            std::string port = rtp_url.substr(rtp_url.find(":", rtp_url.find("://") + 3) + 1); 
            long long bps_value = 6000000; 
            gst_output_pipeline = "appsrc ! videoconvert ! mpph264enc bps=" + std::to_string(bps_value) + " ! h264parse ! rtph264pay config-interval=1 ! udpsink host=" + host + " port=" + port; 
            */ 
            long long bps_value = 6000000; 
            std::string stream_url = rtp_url; 
            // rtp_url 变量将直接包含完整的推流地址，例如 "rtsp://192.168.x.x:8554/yolocam" 
            // 我们使用 rtspclientsink 将流推送到这个地址 
            gst_output_pipeline = "appsrc ! videoconvert ! mpph264enc bps=" + std::to_string(bps_value) + " ! h264parse ! rtspclientsink location=" + stream_url; 
            break; 
    } 

    printf("GStreamer Output Pipeline: %s\n", gst_output_pipeline.c_str()); 
    video_writer.open(gst_output_pipeline, cv::CAP_GSTREAMER, 0, fps_ref, cv::Size(frame_width, frame_height), true); 
    if (!video_writer.isOpened()) { 
        fprintf(stderr, "Error: Could not open VideoWriter.\n"); 
        return -1; 
    } 

    // --- 主循环 --- 
    std::queue<cv::Mat> frame_queue; 
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

        infer_pool.put(img); 
        frame_queue.push(img); 

        if (frame_queue.size() > threadNum) 
        { 
            object_detect_result_list results; 
            if (infer_pool.get(results) == 0) { 
                cv::Mat original_frame = frame_queue.front(); 
                frame_queue.pop(); 

                Visualizer::draw(original_frame, results); 

                // 如果是推流模式，mpph264enc 需要RGB格式，此处进行转换 
                if (output_mode == OutputMode::RTP_STREAM) { 
                    cv::cvtColor(original_frame, original_frame, cv::COLOR_BGR2RGB); 
                } 

                video_writer.write(original_frame); 
                frames++; 
            } 
        } 

        if (frames > 0 && frames % 120 == 0) { 
            gettimeofday(&time, nullptr); 
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000; 
            printf("Average FPS over 120 frames:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0); 
            beforeTime = currentTime; 
        } 
    } 

    // 清理流水线中剩余的帧 
    while(!frame_queue.empty()) 
    { 
        object_detect_result_list results; 
        if (infer_pool.get(results) == 0) { 
            cv::Mat original_frame = frame_queue.front(); 
            frame_queue.pop(); 
            Visualizer::draw(original_frame, results); 

            // 同样，如果是推流模式，需要转换颜色 
            if (output_mode == OutputMode::RTP_STREAM) { 
                cv::cvtColor(original_frame, original_frame, cv::COLOR_BGR2RGB); 
            } 

            video_writer.write(original_frame); 
            frames++; 
        } else { 
            break; 
        } 
    } 

    gettimeofday(&time, nullptr); 
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000; 
    printf("\n--- Final Stats ---\n"); 
    printf("Total frames processed: %d\n", frames); 
    if (endTime > startTime) { 
      printf("Total time: %lld ms\n", endTime - startTime); 
      printf("Overall Average FPS: %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0); 
    } 

    capture.release(); 
    video_writer.release(); 
    deinit_post_process(); 

    return 0; 
}

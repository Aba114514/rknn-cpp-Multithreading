#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <string> // 包含 <string> 头文件

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Yolo11.hpp" 
#include "rknnPool.hpp"

int main(int argc, char **argv)
{
    char *model_name = NULL;
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <video_path | camera_id> \n", argv[0]);
        return -1;
    }
    model_name = (char *)argv[1];
    char *vedio_name = argv[2];

    int threadNum = 3;
    rknnPool<Yolo11, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    cv::namedWindow("Camera FPS");
    cv::VideoCapture capture;

    // ==================== 修改开始 ====================
    std::string video_source = vedio_name;

    // 判断输入是摄像头ID还是视频文件路径
    // 如果是单个数字字符，则认为是摄像头ID，并使用GStreamer管线
    if (video_source.length() == 1 && isdigit(video_source[0])) 
    {
        // GStreamer管线，使用libcamera作为后端捕获MJPG格式
        // 您可以根据需要修改分辨率和帧率
        std::string gst_pipeline = "libcamerasrc ! image/jpeg,width=1920,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink";
        
        printf("Using GStreamer pipeline: %s\n", gst_pipeline.c_str());
        
        // 使用cv::CAP_GSTREAMER标志打开管线
        capture.open(gst_pipeline, cv::CAP_GSTREAMER);
    }
    else 
    {
        // 如果是文件路径，则直接打开视频文件
        printf("Opening video file: %s\n", video_source.c_str());
        capture.open(video_source);
    }

    // 检查视频源是否成功打开
    if (!capture.isOpened()) {
        fprintf(stderr, "Error: Could not open video source: %s\n", vedio_name);
        if (video_source.length() == 1 && isdigit(video_source[0])) {
            fprintf(stderr, "Hint: Please check if the camera is connected and if your OpenCV was built with GStreamer support.\n");
        }
        return -1;
    }
    // ==================== 修改结束 ====================


    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    int frames = 0;
    auto beforeTime = startTime;
    while (capture.isOpened())
    {
        cv::Mat img;
        if (capture.read(img) == false)
            break; 
        
        if (testPool.put(img) != 0)
            break;

        if (frames >= threadNum && testPool.get(img) != 0)
            break; 

        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q') 
            break;
        frames++;

        if (frames % 120 == 0)
        {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }

    while (true)
    {
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;
        
        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q')
            break;
        frames++;
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}

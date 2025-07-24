#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Yolo11.hpp"       // <-- 修改
#include "rknnPool.hpp"

int main(int argc, char **argv)
{
    char *model_name = NULL;
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <video_path/camera_id> \n", argv[0]);
        return -1;
    }
    model_name = (char *)argv[1];
    char *vedio_name = argv[2];

    int threadNum = 3;
    // 使用新的Yolo11类
    rknnPool<Yolo11, cv::Mat, cv::Mat> testPool(model_name, threadNum); // <-- 修改
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    // 创建一个名为 "Camera FPS" 的窗口用于显示图像
    cv::namedWindow("Camera FPS");
    // 创建OpenCV视频捕捉对象
    cv::VideoCapture capture;
    // 判断视频源参数是一个字符（通常是摄像头ID）还是字符串（视频文件路径）
    if (strlen(vedio_name) == 1)
        // 如果是单个字符，则将其转换为整数作为摄像头ID打开
        capture.open((int)(vedio_name[0] - '0'));
    else
        // 否则，直接作为文件路径打开
        capture.open(vedio_name);

    // +++ START of ADDED CODE +++
    // 当输入源是摄像头时，尝试设置摄像头的分辨率为1080p (1920x1080)
    // 这个操作只对摄像头有效，对视频文件无效
    // if (strlen(vedio_name) == 1) {/
    //     capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    //     capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    // }
    // +++ END of ADDED CODE +++

    // 获取当前时间，用于计算总运行时间和FPS
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000; // 转换为毫秒

    int frames = 0; // 帧计数器
    auto beforeTime = startTime; // 用于计算周期性FPS的时间戳
    // 主循环，当视频成功打开时持续执行
    while (capture.isOpened())
    {
        cv::Mat img;
        // 从视频源读取一帧图像
        if (capture.read(img) == false)
            break; // 如果读取失败（视频结束或出错），则退出循环
        
        // 将读取到的帧异步提交到rknn线程池进行处理
        if (testPool.put(img) != 0)
            break; // 如果提交失败，退出循环

        // 当提交的帧数达到或超过线程数时，开始尝试从线程池获取处理完的结果
        // 这是为了保证处理流水线中有足够的任务，避免主线程过早等待
        if (frames >= threadNum && testPool.get(img) != 0)
            break; // 如果获取结果失败，退出循环

        // 在窗口中显示处理后的图像（带有检测框）
        cv::imshow("Camera FPS", img);
        // 等待1毫秒，并检查是否有按键。如果按下 'q' 键，则退出循环
        if (cv::waitKey(1) == 'q') 
            break;
        frames++; // 帧计数加一

        // 每处理120帧，计算并打印一次这期间的平均帧率
        if (frames % 120 == 0)
        {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime; // 更新时间戳以便下次计算
        }
    }

    // 清空rknn线程池中剩余的已处理帧
    // 当视频源结束后，此循环确保所有已提交的帧都被取回并显示
    while (true)
    {
        cv::Mat img;
        // 尝试从线程池获取结果
        if (testPool.get(img) != 0)
            break; // 如果获取失败（表示线程池已空），则退出循环
        
        cv::imshow("Camera FPS", img); // 显示剩余的帧
        if (cv::waitKey(1) == 'q') // 同样允许按 'q' 键提前退出
            break;
        frames++; // 总帧数继续计数
    }

    // 获取程序结束时间
    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    // 计算并打印整个运行过程的平均帧率
    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0; // 程序正常结束
}
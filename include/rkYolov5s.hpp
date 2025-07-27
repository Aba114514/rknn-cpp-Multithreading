#ifndef RKYOLOV5S_H // 防止头文件被重复包含的宏保护
#define RKYOLOV5S_H

#include "rknn_api.h" // 引入Rockchip神经网络（RKNN）C语言API的核心头文件

#include "opencv2/core/core.hpp" // 引入OpenCV核心库，主要为了使用 cv::Mat 数据结构

// 声明一个静态辅助函数，用于打印RKNN张量（tensor）的属性，方便调试
static void dump_tensor_attr(rknn_tensor_attr *attr);
// 声明一个静态辅助函数，用于从文件指针中加载指定偏移和大小的二进制数据
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
// 声明一个静态辅助函数，用于从文件中加载整个RKNN模型，并返回模型大小
static unsigned char *load_model(const char *filename, int *model_size);
// 声明一个静态辅助函数，用于将浮点数数组保存到文件中，方便调试和分析模型输出
static int saveFloat(const char *file_name, float *output, int element_size);

// 定义 rkYolov5s 类，封装了YOLOv5s模型在RKNN上的所有操作
class rkYolov5s
{
private:
    int ret;                    // 用于存储RKNN API调用的返回值，便于错误检查
    std::mutex mtx;             // 互斥锁，用于保证在多线程环境下该类实例的线程安全
    std::string model_path;     // 存储.rknn模型文件的路径
    unsigned char *model_data;  // 指向内存中加载的模型数据的指针

    rknn_context ctx;           // RKNN推理上下文句柄，是与NPU交互的核心
    rknn_input_output_num io_num; // 存储模型输入输出张量的数量
    rknn_tensor_attr *input_attrs;  // 指向模型输入张量属性数组的指针
    rknn_tensor_attr *output_attrs; // 指向模型输出张量属性数组的指针
    rknn_input inputs[1];       // 用于设置模型输入的数组（YOLOv5s通常只有一个输入）

    int channel, width, height; // 模型输入张量的维度（通道数、宽度、高度）
    int img_width, img_height;  // 原始输入图像的宽度和高度，用于后续坐标还原

    float nms_threshold;        // 非极大值抑制（NMS）的阈值，用于合并重叠的检测框
    float box_conf_threshold;   // 检测框置信度阈值，用于过滤掉低置信度的检测结果

public:
    // 构造函数，接收模型文件路径
    rkYolov5s(const std::string &model_path);

    /**
     * @brief 初始化函数
     * @param ctx_in [in] 传入的父上下文句柄，用于实现零拷贝
     * @param isChild [in] 标志位，指示当前是否作为子上下文进行初始化（用于多线程优化）
     * @return int 0表示成功，其他值表示失败
     */
    int init(rknn_context *ctx_in, bool isChild);

    // 获取内部RKNN上下文句柄的指针，主要由线程池管理类调用
    rknn_context *get_pctx();

    /**
     * @brief 执行模型推理
     * @param ori_img [in] 待检测的原始OpenCV图像
     * @return cv::Mat 返回绘制了检测结果的图像
     */
    cv::Mat infer(cv::Mat &ori_img);

    // 析构函数，用于释放资源，如销毁RKNN上下文、释放内存等
    ~rkYolov5s();
};

#endif // RKYOLOV5S_H
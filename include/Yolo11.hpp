// फाइल का नाम: include/Yolo11.hpp

#ifndef YOLO11_HPP
#define YOLO11_HPP

#include "rknn_api.h"
#include "common_types.h"
#include "opencv2/core/core.hpp"
#include <mutex>
#include <string>
#include <vector>

class Yolo11
{
private:
    rknn_context rknn_ctx;
    std::mutex mtx;
    std::string model_path;

    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;

    int model_width;
    int model_height;
    int model_channel;
    bool is_quant;

    // 内部辅助函数，用于加载模型文件
    static unsigned char *load_model(const char *filename, int *model_size);

public:
    Yolo11(const std::string &model_path);
    ~Yolo11();

    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();

    object_detect_result_list infer(const cv::Mat &orig_img);
};

#endif // YOLO11_HPP
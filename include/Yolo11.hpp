#ifndef YOLO11_HPP
#define YOLO11_HPP

#include "rknn_api.h"
#include "postprocess.h" // 使用新的postprocess.h
#include "preprocess.h"  // 预处理可以复用
#include "opencv2/core/core.hpp"
#include <mutex>

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

public:
    // 公共getter方法，供postprocess函数访问
    int get_model_width() const { return model_width; }
    int get_model_height() const { return model_height; }
    int get_io_num_n_output() const { return io_num.n_output; }
    bool get_is_quant() const { return is_quant; }
    rknn_tensor_attr* get_output_attrs() const { return output_attrs; }

public:
    Yolo11(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild); // 保持与rknnPool兼容的init接口
    rknn_context *get_pctx();
    cv::Mat infer(cv::Mat &ori_img);
    ~Yolo11();
};

#endif // YOLO11_HPP
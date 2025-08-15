// 文件: include/Yolo11.hpp

#ifndef YOLO11_HPP
#define YOLO11_HPP

#include "rknn_api.h"
#include "common_types.h"
#include "opencv2/core/core.hpp"
#include <mutex>
#include <string>
#include <vector>

#include "dma_alloc.h" // <--- 包含dma_alloc.h

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

    // --- 新增：用于零拷贝的DMA缓冲区成员 ---
    int resized_img_dma_fd;
    void* resized_img_dma_va;
    size_t resized_img_dma_size;

    static unsigned char *load_model(const char *filename, int *model_size);

public:
    Yolo11(const std::string &model_path);
    ~Yolo11();

    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();

    object_detect_result_list infer(const cv::Mat &orig_img);
};

#endif // YOLO11_HPP

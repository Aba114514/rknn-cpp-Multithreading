// फाइल का नाम: src/Yolo11.cc

#include "Yolo11.hpp"
#include "postprocess.h"
#include "preprocess.h"
#include "coreNum.hpp"
#include <vector>
#include <cstdio>
#include <cstdlib>

// 内部辅助函数，用于从磁盘加载模型文件
unsigned char* Yolo11::load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(size);
    if (data == nullptr) {
        fclose(fp);
        printf("malloc model buffer fail!\n");
        return nullptr;
    }
    fread(data, 1, size, fp);
    fclose(fp);
    *model_size = size;
    return data;
}

// 构造函数：初始化成员变量
Yolo11::Yolo11(const std::string &path)
    : model_path(path), rknn_ctx(0), input_attrs(nullptr), output_attrs(nullptr)
{
}

// 析构函数：释放所有资源
Yolo11::~Yolo11()
{
    if (rknn_ctx != 0) {
        rknn_destroy(rknn_ctx);
    }
    if (input_attrs) {
        free(input_attrs);
        input_attrs = nullptr;
    }
    if (output_attrs) {
        free(output_attrs);
        output_attrs = nullptr;
    }
}

// 获取rknn_context的指针，供rknnPool初始化时使用
rknn_context* Yolo11::get_pctx() {
    return &rknn_ctx;
}

// 初始化函数：加载并配置RKNN模型
int Yolo11::init(rknn_context *ctx_in, bool isChild)
{
    int ret;
    int model_len = 0;
    unsigned char *model = load_model(model_path.c_str(), &model_len);
    if (model == nullptr) { return -1; }

    if (isChild) {
        ret = rknn_dup_context(ctx_in, &rknn_ctx);
    } else {
        ret = rknn_init(&rknn_ctx, model, model_len, 0, nullptr);
    }
    free(model);

    if (ret < 0) {
        printf("rknn_init or rknn_dup_context fail! ret=%d\n", ret);
        return -1;
    }

    // 绑定NPU核心
    rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
    int core_id = get_core_num();
    if (core_id == 0) core_mask = RKNN_NPU_CORE_0;
    else if (core_id == 1) core_mask = RKNN_NPU_CORE_1;
    else if (core_id == 2) core_mask = RKNN_NPU_CORE_2;

    ret = rknn_set_core_mask(rknn_ctx, core_mask);
    if (ret < 0) {
        printf("rknn_set_core_mask error ret=%d\n", ret);
        return -1;
    }

    // 查询模型的输入输出信息
    ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return -1;

    input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    if (!input_attrs || !output_attrs) {
        printf("malloc for input/output attrs fail!\n");
        return -1;
    }

    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) return -1;
    }

    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) return -1;
    }

    // 获取模型输入的维度信息
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    } else {
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }

    // ** 关键修正：使用正确的逻辑来判断模型是否为量化模型 **
    is_quant = (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8);

    return 0;
}

// 推理函数：执行预处理、NPU推理和后处理
object_detect_result_list Yolo11::infer(const cv::Mat &orig_img)
{
    // 锁定此实例，确保线程安全
    std::lock_guard<std::mutex> instance_lock(mtx);

    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(od_results));
    int ret;

    // 1. 预处理: 使用RGA硬件将输入图像缩放并转换为RGB格式
    cv::Mat resized_img(model_height, model_width, CV_8UC3);
    ret = resize_rga(orig_img, resized_img);
    if (ret != 0) {
        printf("Pre-process (resize_rga) failed.\n");
        return od_results; // 返回空结果
    }

    // 2. NPU推理: 设置输入并运行模型
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = model_width * model_height * model_channel;
    inputs[0].buf = resized_img.data;

    ret = rknn_inputs_set(rknn_ctx, io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        return od_results;
    }

    ret = rknn_run(rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return od_results;
    }

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 0; // 后处理需要原始INT8数据进行反量化
    }
    ret = rknn_outputs_get(rknn_ctx, io_num.n_output, outputs, nullptr);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        // 即使获取失败，也要尝试释放，防止内存泄漏
        rknn_outputs_release(rknn_ctx, io_num.n_output, outputs);
        return od_results;
    }

    // 3. 后处理: 解码NPU输出，得到结构化检测结果
    BOX_RECT letter_box;
    // 因为是直接resize，所以计算宽高缩放比用于坐标还原
    letter_box.scale_w = (float)orig_img.cols / model_width;
    letter_box.scale_h = (float)orig_img.rows / model_height;

    post_process(outputs, io_num.n_output, output_attrs, is_quant,
                 model_width, model_height, &letter_box, &od_results);

    // 释放rknn_outputs_get分配的内存
    rknn_outputs_release(rknn_ctx, io_num.n_output, outputs);

    // 返回结构化数据
    return od_results;
}
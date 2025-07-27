#include "Yolo11.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>

#include "rga.h"
#include "im2d.h"
#include "coreNum.hpp"

// 定义一个全局互斥锁，用于保护所有对RGA硬件的并发访问
static std::mutex rga_mutex;

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(size);
    fread(data, 1, size, fp);
    fclose(fp);
    *model_size = size;
    return data;
}

Yolo11::Yolo11(const std::string &path) : model_path(path) {
    init_post_process();
}

Yolo11::~Yolo11()
{
    if (rknn_ctx != 0) {
        rknn_destroy(rknn_ctx);
    }
    if (input_attrs) free(input_attrs);
    if (output_attrs) free(output_attrs);
    deinit_post_process();
}

rknn_context* Yolo11::get_pctx() {
    return &rknn_ctx;
}

int Yolo11::init(rknn_context *ctx_in, bool isChild)
{
    int ret;
    int model_len = 0;
    unsigned char *model = load_model(model_path.c_str(), &model_len);
    if (model == NULL) { return -1; }

    if (isChild) {
        ret = rknn_dup_context(ctx_in, &rknn_ctx);
    } else {
        ret = rknn_init(&rknn_ctx, model, model_len, 0, NULL);
    }
    free(model);

    if (ret < 0) {
        printf("rknn_init or rknn_dup_context fail! ret=%d\n", ret);
        return -1;
    }

    // NPU核心绑定逻辑
    rknn_core_mask core_mask;
    int core_id = get_core_num();
    switch (core_id)
    {
        case 0:
            core_mask = RKNN_NPU_CORE_0;
            printf("Instance init on NPU Core 0\n");
            break;
        case 1:
            core_mask = RKNN_NPU_CORE_1;
            printf("Instance init on NPU Core 1\n");
            break;
        case 2:
            core_mask = RKNN_NPU_CORE_2;
            printf("Instance init on NPU Core 2\n");
            break;
        default:
            core_mask = RKNN_NPU_CORE_AUTO;
            printf("Instance init on NPU Core AUTO\n");
            break;
    }
    ret = rknn_set_core_mask(rknn_ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_set_core_mask error ret=%d\n", ret);
        return -1;
    }

    ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return -1;

    input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));

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

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    } else {
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }

    is_quant = (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8);

    return 0;
}

cv::Mat Yolo11::infer(cv::Mat &orig_img)
{
    std::lock_guard<std::mutex> instance_lock(mtx);
    int ret;

    cv::Mat resized_img(model_height, model_width, CV_8UC3);

    {
        // --- 预处理：使用全局锁保护 RGA 硬件访问 ---
        std::lock_guard<std::mutex> rga_lock(rga_mutex);

        // 使用基于 size 的 importbuffer API, 解决 >4G 内存访问问题
        size_t src_size = orig_img.total() * orig_img.elemSize();
        size_t dst_size = resized_img.total() * resized_img.elemSize();

        rga_buffer_handle_t src_handle = importbuffer_virtualaddr(orig_img.data, src_size);
        rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(resized_img.data, dst_size);

        IM_STATUS rga_status = IM_STATUS_FAILED;

        if (src_handle && dst_handle) {
            rga_buffer_t src_rga = wrapbuffer_handle(src_handle, orig_img.cols, orig_img.rows, RK_FORMAT_BGR_888);
            rga_buffer_t dst_rga = wrapbuffer_handle(dst_handle, resized_img.cols, resized_img.rows, RK_FORMAT_RGB_888);

            // 使用RGA硬件加速完成 BGR->RGB 转换和图像缩放
            rga_status = imresize(src_rga, dst_rga);
        } else {
            fprintf(stderr, "RGA importbuffer failed for imresize. src_handle=%d, dst_handle=%d\n", src_handle, dst_handle);
        }

        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);

        if (rga_status != IM_STATUS_SUCCESS) {
            fprintf(stderr, "RGA imresize failed with status %d: %s\n", rga_status, imStrError(rga_status));
            return orig_img;
        }
    }

    // --- 模型推理 ---
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = model_width * model_height * model_channel;
    inputs[0].buf = resized_img.data;

    ret = rknn_inputs_set(rknn_ctx, io_num.n_input, inputs);
    if (ret < 0) return orig_img;

    ret = rknn_run(rknn_ctx, nullptr);
    if (ret < 0) return orig_img;

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 0;
    }
    ret = rknn_outputs_get(rknn_ctx, io_num.n_output, outputs, NULL);
    if (ret < 0) return orig_img;

    // --- 后处理 ---
    object_detect_result_list od_results;
    float scale_w = (float)orig_img.cols / model_width;
    float scale_h = (float)orig_img.rows / model_height;
    BOX_RECT letter_box;
    letter_box.scale_w = scale_w;
    letter_box.scale_h = scale_h;
    post_process(this, outputs, &letter_box, BOX_THRESH, NMS_THRESH, &od_results);

    // --- 关键修改：根据您的最终决策，使用稳定可靠的OpenCV CPU函数进行绘制 ---
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result *det_result = &(od_results.results[i]);
        char text[256];
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        // 使用 OpenCV 的 rectangle 函数绘制检测框
        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 3);
        // 使用 OpenCV 的 putText 函数绘制文本
        putText(orig_img, text, cv::Point(x1, y1 > 12 ? y1 - 12 : y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    rknn_outputs_release(rknn_ctx, io_num.n_output, outputs);
    return orig_img;
}

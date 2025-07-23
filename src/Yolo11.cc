#include "Yolo11.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> // for std::min

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
    std::lock_guard<std::mutex> lock(mtx);
    int ret;

    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

    cv::Mat resized_img(model_height, model_width, CV_8UC3);
    rga_buffer_t src_rga, dst_rga;
    memset(&src_rga, 0, sizeof(src_rga));
    memset(&dst_rga, 0, sizeof(dst_rga));

    ret = resize_rga(src_rga, dst_rga, img, resized_img, cv::Size(model_width, model_height));
    if (ret != 0) {
        fprintf(stderr, "resize with rga error\n");
        return orig_img;
    }

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

    object_detect_result_list od_results;

    // ** 关键修复：计算独立的宽高缩放比例 **
    float scale_w = (float)orig_img.cols / model_width;
    float scale_h = (float)orig_img.rows / model_height;

    // 创建一个临时的BOX_RECT来传递缩放比例
    BOX_RECT letter_box;
    // 将两个比例都存入，虽然有点冗余，但可以避免修改post_process的函数签名
    letter_box.scale_w = scale_w;
    letter_box.scale_h = scale_h;

    post_process(this, outputs, &letter_box, BOX_THRESH, NMS_THRESH, &od_results);

    // 绘制结果
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result *det_result = &(od_results.results[i]);
        char text[256];
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        putText(orig_img, text, cv::Point(x1, y1 > 10 ? y1 - 10 : y1 + 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    rknn_outputs_release(rknn_ctx, io_num.n_output, outputs);
    return orig_img;
}
#include <stdio.h>
#include <mutex>
#include "rknn_api.h"

// 引入预处理和后处理的头文件
#include "postprocess.h"
#include "preprocess.h"

// 引入OpenCV相关库
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// 引入自定义的NPU核心绑定工具和类定义头文件
#include "coreNum.hpp"
#include "rkYolov5s.hpp"

// 静态辅助函数：打印张量属性（已注释，用于调试）
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    // 将张量的维度信息格式化为字符串
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    // 打印张量的所有属性，如索引、名称、维度、大小、格式、类型、量化参数等
    // printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
    //        "type=%s, qnt_type=%s, "
    //        "zp=%d, scale=%f\n",
    //        attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
    //        attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
    //        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// 静态辅助函数：从文件指针的指定偏移处加载指定大小的数据
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    // 将文件指针移动到指定偏移
    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    // 分配内存以存储数据
    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    // 从文件读取数据到内存
    ret = fread(data, 1, sz, fp);
    return data;
}

// 静态辅助函数：从文件加载整个模型
static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    // 以二进制只读方式打开模型文件
    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    // 获取文件大小
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    // 加载整个文件内容
    data = load_data(fp, 0, size);

    fclose(fp);

    // 通过指针返回模型大小
    *model_size = size;
    return data;
}

// 静态辅助函数：将浮点数数组保存到文件（用于调试）
static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

// rkYolov5s 类的构造函数
rkYolov5s::rkYolov5s(const std::string &model_path)
{
    this->model_path = model_path;
    nms_threshold = NMS_THRESH;      // 从宏定义初始化 NMS 阈值
    box_conf_threshold = BOX_THRESH; // 从宏定义初始化置信度阈值
}

// 初始化函数
int rkYolov5s::init(rknn_context *ctx_in, bool share_weight)
{
    printf("Loading model...\n");
    int model_data_size = 0;
    // 加载模型文件到内存
    model_data = load_model(model_path.c_str(), &model_data_size);

    // 根据 share_weight 参数决定是创建新上下文还是复制上下文
    if (share_weight == true)
        // 在多线程中，复制现有上下文以共享模型权重，节省内存
        ret = rknn_dup_context(ctx_in, &ctx);
    else
        // 创建一个全新的RKNN上下文
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置此上下文需要绑定的NPU核心
    rknn_core_mask core_mask;
    switch (get_core_num()) // get_core_num() 是一个外部函数，用于决定使用哪个核心
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    // 查询并打印SDK和驱动版本信息
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // 获取模型的输入输出张量数量
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 查询并设置输入张量的属性
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i])); // 打印属性（调试用）
    }

    // 查询并设置输出张量的属性
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i])); // 打印属性（调试用）
    }

    // 根据输入张量的格式（NCHW或NHWC）确定模型的输入维度
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    // 初始化用于设置输入的结构体
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8; // 数据类型
    inputs[0].size = width * height * channel; // 数据总大小
    inputs[0].fmt = RKNN_TENSOR_NHWC; // 告诉RKNN，我们将以NHWC格式提供数据
    inputs[0].pass_through = 0; // 非直通模式，需要SDK进行预处理

    return 0;
}

// 获取RKNN上下文句柄的指针
rknn_context *rkYolov5s::get_pctx()
{
    return &ctx;
}

// 执行推理
cv::Mat rkYolov5s::infer(cv::Mat &orig_img)
{
    // 使用 lock_guard 保证此函数在多线程环境下的线程安全
    std::lock_guard<std::mutex> lock(mtx);
    cv::Mat img;
    // 将OpenCV的BGR图像转换为RGB
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    img_width = img.cols;
    img_height = img.rows;

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    // 目标尺寸为模型输入尺寸
    cv::Size target_size(width, height);
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
    // 计算缩放比例
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;

    // 如果原始图像尺寸与模型输入尺寸不符，则进行缩放
    if (img_width != width || img_height != height)
    {
        // 优先使用RGA硬件进行高效缩放
        rga_buffer_t src;
        rga_buffer_t dst;
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
        ret = resize_rga(src, dst, img, resized_img, target_size);
        if (ret != 0)
        {
            fprintf(stderr, "resize with rga error\n");
        }
        /*********
        // 如果不使用RGA，可以使用OpenCV的letterbox方法进行缩放和填充
        float min_scale = std::min(scale_w, scale_h);
        scale_w = min_scale;
        scale_h = min_scale;
        letterbox(img, resized_img, pads, min_scale, target_size);
        *********/
        // 将缩放后的图像数据指针设置给输入结构体
        inputs[0].buf = resized_img.data;
    }
    else
    {
        // 如果尺寸相同，直接使用原始图像数据
        inputs[0].buf = img.data;
    }

    // 将输入数据设置到RKNN上下文
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    // 准备接收输出的结构体
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0; // 我们希望得到量化后的int8输出，而不是浮点数
    }

    // 执行模型推理
    ret = rknn_run(ctx, NULL);
    // 获取推理结果
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    // 对模型的原始输出进行后处理
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    // 收集每个输出层的量化参数（scale和zero point）
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    // 调用后处理函数，解码、NMS等操作都在这里完成
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                 box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    // 绘制检测框
    char text[256];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        // 打印物体信息到控制台（调试用）
        // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
        //        det_result->box.right, det_result->box.bottom, det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        // 在原始图像上绘制矩形框
        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
        // 在框的上方绘制类别和置信度文本
        putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    // 释放本次推理的输出缓存，必须调用！
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    // 返回绘制好检测框的原始图像
    return orig_img;
}

// rkYolov5s 类的析构函数
rkYolov5s::~rkYolov5s()
{
    // 释放后处理中可能申请的资源
    deinitPostProcess();

    // 销毁RKNN上下文
    ret = rknn_destroy(ctx);

    // 释放加载的模型数据内存
    if (model_data)
        free(model_data);

    // 释放输入输出属性数组的内存
    if (input_attrs)
        free(input_attrs);
    if (output_attrs)
        free(output_attrs);
}
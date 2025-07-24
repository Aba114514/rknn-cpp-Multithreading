// 版权和许可证信息
// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// ... (Apache License 2.0)

#include <stdio.h>
#include "im2d.h"     // Rockchip RGA 2D图像操作库的头文件
#include "rga.h"      // Rockchip RGA 核心库的头文件
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"  // 可能是定义了 BOX_RECT 结构体的头文件

/**
 * @brief 对图像进行 letterbox 处理，保持纵横比缩放并填充至目标尺寸
 * @param image [in] 输入的原始OpenCV图像
 * @param padded_image [out] 经过letterbox处理后的输出图像
 * @param pads [out] 记录上下左右填充像素数的结构体
 * @param scale [in] 统一的缩放比例
 * @param target_size [in] 最终的目标尺寸
 * @param pad_color [in] 用于填充边框的颜色
 */
void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, const float scale, const cv::Size &target_size, const cv::Scalar &pad_color)
{
    // 1. 根据指定的缩放比例调整图像大小
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(), scale, scale);

    // 2. 计算需要填充的宽度和高度
    int pad_width = target_size.width - resized_image.cols;
    int pad_height = target_size.height - resized_image.rows;

    // 3. 计算上下左右四个方向的填充量，使其基本居中
    pads.left = pad_width / 2;
    pads.right = pad_width - pads.left;
    pads.top = pad_height / 2;
    pads.bottom = pad_height - pads.top;

    // 4. 在图像周围添加指定颜色的边框（填充）
    cv::copyMakeBorder(resized_image, padded_image, pads.top, pads.bottom, pads.left, pads.right, cv::BORDER_CONSTANT, pad_color);
}

/**
 * @brief 使用RGA硬件加速来缩放图像
 * @param src [out] 包装好的RGA源缓冲
 * @param dst [out] 包装好的RGA目标缓冲
 * @param image [in] 输入的原始OpenCV图像 (必须是RGB格式)
 * @param resized_image [in/out] 用于存放缩放结果的OpenCV图像
 * @param target_size [in] 目标尺寸
 * @return int 0表示成功，-1表示失败
 */
int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size)
{
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    size_t img_width = image.cols;
    size_t img_height = image.rows;
    // RGA通常处理RGB/BGR格式，检查输入图像类型是否正确
    if (image.type() != CV_8UC3)
    {
        printf("source image type is %d!\n", image.type());
        return -1;
    }
    size_t target_width = target_size.width;
    size_t target_height = target_size.height;

    // 1. 将源图像的内存地址包装成RGA buffer，这是一个零拷贝操作
    src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_RGB_888);
    // 2. 将目标图像的内存地址包装成RGA buffer
    dst = wrapbuffer_virtualaddr((void *)resized_image.data, target_width, target_height, RK_FORMAT_RGB_888);

    // 3. 检查RGA操作的参数是否有效
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        fprintf(stderr, "rga check error! %s", imStrError((IM_STATUS)ret));
        return -1;
    }

    // 4. 启动RGA硬件执行图像缩放
    IM_STATUS STATUS = imresize(src, dst);
    return 0;
}
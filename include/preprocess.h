#ifndef _RKNN_YOLOV5_DEMO_PREPROCESS_H_ // 防止头文件被重复包含的宏保护
#define _RKNN_YOLOV5_DEMO_PREPROCESS_H_

#include <stdio.h>
#include "im2d.h"      // Rockchip RGA 2D图像操作库头文件
#include "rga.h"       // Rockchip RGA 硬件加速库头文件
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h" // 引入后处理头文件，主要是为了使用其中定义的BOX_RECT结构体

/**
 * @brief 使用letterbox方式对图像进行缩放和填充，以保持其原始纵横比。
 * @param image         [in] 输入的原始cv::Mat图像。
 * @param padded_image  [out] 经过处理后的输出cv::Mat图像。
 * @param pads          [out] 记录填充边框尺寸的结构体。
 * @param scale         [in] 缩放因子。
 * @param target_size   [in] 最终的目标尺寸。
 * @param pad_color     [in] 用于填充边框的颜色，默认为灰色(128, 128, 128)。
 */
void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, const float scale, const cv::Size &target_size, const cv::Scalar &pad_color = cv::Scalar(128, 128, 128));

/**
 * @brief 使用Rockchip RGA硬件加速单元来调整图像尺寸。
 * @param src           [in] RGA源缓冲区的句柄。
 * @param dst           [in] RGA目标缓冲区的句柄。
 * @param image         [in] 输入的原始cv::Mat图像。
 * @param resized_image [out] 经过缩放后的输出cv::Mat图像。
 * @param target_size   [in] 最终的目标尺寸。
 * @return int 0表示成功，其他值表示失败。
 */
int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size);

#endif //_RKNN_YOLOV5_DEMO_PREPROCESS_H_
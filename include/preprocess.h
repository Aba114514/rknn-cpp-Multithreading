// 文件: include/preprocess.h

#ifndef _RKNN_YOLO_DEMO_PREPROCESS_H_
#define _RKNN_YOLO_DEMO_PREPROCESS_H_

#include "opencv2/core/core.hpp"
#include "common_types.h"
#include <mutex>

extern std::mutex rga_mutex;

/**
 * @brief 使用RGA硬件将图像缩放并输出到指定的DMA缓冲区
 * @param src_img   [in] 输入的原始BGR格式cv::Mat图像
 * @param dst_fd    [in] 目标DMA缓冲区的fd (file descriptor)
 * @param dst_w     [in] 目标宽度
 * @param dst_h     [in] 目标高度
 * @return int 0表示成功，其他值表示失败
 */
int resize_rga(const cv::Mat &src_img, int dst_fd, int dst_w, int dst_h);

#endif //_RKNN_YOLO_DEMO_PREPROCESS_H_
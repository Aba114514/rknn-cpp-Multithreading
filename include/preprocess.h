// फाइल का नाम: include/preprocess.h

#ifndef _RKNN_YOLO_DEMO_PREPROCESS_H_
#define _RKNN_YOLO_DEMO_PREPROCESS_H_

#include "opencv2/core/core.hpp"
#include "common_types.h" // 包含新的类型头文件
#include <mutex>

// 定义一个全局互斥锁，用于保护所有对RGA硬件的并发访问
extern std::mutex rga_mutex;

/**
 * @brief 使用Rockchip RGA硬件加速单元来调整图像尺寸并进行颜色空间转换 (BGR->RGB)
 * @param src_img       [in] 输入的原始BGR格式cv::Mat图像。
 * @param dst_img       [out] 经过处理后的输出RGB格式cv::Mat图像。
 * @return int 0表示成功，其他值表示失败。
 */
int resize_rga(const cv::Mat &src_img, cv::Mat &dst_img);

#endif //_RKNN_YOLO_DEMO_PREPROCESS_H_
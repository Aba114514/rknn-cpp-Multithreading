// फाइल का नाम: include/Visualizer.hpp

#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "opencv2/core/core.hpp"
#include "common_types.h"

namespace Visualizer {
    /**
     * @brief 在图像上绘制检测结果
     * @param img [in/out] 需要被绘制的OpenCV图像
     * @param results [in] 结构化的检测结果列表
     */
    void draw(cv::Mat& img, const object_detect_result_list& results);
}

#endif // VISUALIZER_HPP
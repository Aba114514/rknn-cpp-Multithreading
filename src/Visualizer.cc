//src/Visualizer.cc

#include "Visualizer.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h" // 需要用coco_cls_to_name

namespace Visualizer {

    void draw(cv::Mat& img, const object_detect_result_list& results)
    {
        for (int i = 0; i < results.count; i++) {
            const object_detect_result *det_result = &(results.results[i]);
            char text[256];
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            // 使用 OpenCV 的 rectangle 函数绘制检测框
            rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 3);
            // 使用 OpenCV 的 putText 函数绘制文本
            putText(img, text, cv::Point(x1, y1 > 20 ? y1 - 10 : y1 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
    }

} // namespace Visualizer

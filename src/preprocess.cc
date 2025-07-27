// फाइल का नाम: src/preprocess.cc

#include "preprocess.h"
#include "rga.h"
#include "im2d.h"
#include <cstdio>

// 在此源文件中定义全局互斥锁
std::mutex rga_mutex;

// 统一的、权威的RGA预处理实现
int resize_rga(const cv::Mat &src_img, cv::Mat &dst_img)
{
    // 使用全局锁保护 RGA 硬件访问
    std::lock_guard<std::mutex> rga_lock(rga_mutex);

    if(src_img.empty() || dst_img.empty()) {
        fprintf(stderr, "Error: Input or output matrix is empty in resize_rga.\n");
        return -1;
    }

    // 使用基于 size 的 importbuffer API, 解决 >4G 内存访问问题
    size_t src_size = src_img.total() * src_img.elemSize();
    size_t dst_size = dst_img.total() * dst_img.elemSize();

    rga_buffer_handle_t src_handle = importbuffer_virtualaddr(src_img.data, src_size);
    rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(dst_img.data, dst_size);

    if (!src_handle || !dst_handle) {
        fprintf(stderr, "RGA importbuffer failed. src_handle=%p, dst_handle=%p\n", src_handle, dst_handle);
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        return -1;
    }

    IM_STATUS rga_status = IM_STATUS_FAILED;

    rga_buffer_t src_rga = wrapbuffer_handle(src_handle, src_img.cols, src_img.rows, RK_FORMAT_BGR_888);
    rga_buffer_t dst_rga = wrapbuffer_handle(dst_handle, dst_img.cols, dst_img.rows, RK_FORMAT_RGB_888);

    // 使用RGA硬件加速完成 BGR->RGB 转换和图像缩放
    rga_status = imresize(src_rga, dst_rga);

    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);

    if (rga_status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "RGA imresize failed with status %d: %s\n", rga_status, imStrError(rga_status));
        return -1;
    }

    return 0;
}
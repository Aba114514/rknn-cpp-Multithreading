// 文件: src/preprocess.cc

#include "preprocess.h"
#include "rga.h"
#include "im2d.h"
#include <cstdio>
#include <mutex> // 头文件可保留，以防未来在别处需要

// 全局互斥锁已移除，以允许多个线程并行调用RGA硬件。
// std::mutex rga_mutex;

int resize_rga(const cv::Mat &src_img, int dst_fd, int dst_w, int dst_h)
{
    // 下一行中的锁已被移除，这是之前版本的主要性能瓶颈。
    // std::lock_guard<std::mutex> rga_lock(rga_mutex);

    if (src_img.empty()) {
        fprintf(stderr, "Error: Input matrix is empty in resize_rga.\n");
        return -1;
    }

    rga_buffer_handle_t src_handle = importbuffer_virtualaddr(src_img.data, src_img.total() * src_img.elemSize());
    rga_buffer_handle_t dst_handle = importbuffer_fd(dst_fd, dst_w * dst_h * 3);

    if (!src_handle || !dst_handle) {
        // =================================================================
        //            == 关键修正：使用 %x 打印整数句柄，消除警告 ==
        // =================================================================
        fprintf(stderr, "RGA importbuffer failed. src_handle=0x%x, dst_handle=0x%x\n", src_handle, dst_handle);
        // =================================================================

        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        return -1;
    }

    rga_buffer_t src_rga = wrapbuffer_handle(src_handle, src_img.cols, src_img.rows, RK_FORMAT_BGR_888);
    rga_buffer_t dst_rga = wrapbuffer_handle(dst_handle, dst_w, dst_h, RK_FORMAT_RGB_888);

    IM_STATUS rga_status = imresize(src_rga, dst_rga);

    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);

    if (rga_status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "RGA imresize failed with status %d: %s\n", rga_status, imStrError(rga_status));
        return -1;
    }

    return 0;
}
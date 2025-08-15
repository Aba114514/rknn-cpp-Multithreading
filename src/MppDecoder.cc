// src/MppDecoder.cc

#include "MppDecoder.hpp"
#include <iostream>
#include <unistd.h>
#include <string.h>

MppDecoder::MppDecoder() : file_in(nullptr), ctx(nullptr), mpi(nullptr), frame_group(nullptr) {
    buffer.resize(1024 * 1024); // 1MB buffer for reading packet
}

MppDecoder::~MppDecoder() {
    if (frame_group) {
        mpp_buffer_group_put(frame_group);
        frame_group = nullptr;
    }
    if (ctx) {
        mpp_destroy(ctx);
        ctx = nullptr;
    }
    if (file_in) {
        fclose(file_in);
        file_in = nullptr;
    }
}

int MppDecoder::init(const char* video_path, MppCodingType type) {
    file_in = fopen(video_path, "rb");
    if (!file_in) {
        std::cerr << "Error: Could not open video file: " << video_path << std::endl;
        return -1;
    }

    RK_U32 need_split = 1;
    MPP_RET ret = mpp_create(&ctx, &mpi);
    if (ret != MPP_OK) {
        std::cerr << "Error: mpp_create failed. ret=" << ret << std::endl;
        return -1;
    }

    ret = mpi->control(ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);
    if (ret != MPP_OK) {
        std::cerr << "Error: set parser split mode failed. ret=" << ret << std::endl;
        return -1;
    }

    ret = mpp_init(ctx, MPP_CTX_DEC, type);
    if (ret != MPP_OK) {
        std::cerr << "Error: mpp_init failed. ret=" << ret << std::endl;
        return -1;
    }

    // 配置为直接输出DMA FD，这是实现零拷贝的核心
    MppDecCfg cfg = nullptr;
    mpp_dec_cfg_init(&cfg);
    mpp_dec_cfg_set_u32(cfg, "base:fast_parse", 1);
    ret = mpi->control(ctx, MPP_DEC_SET_CFG, cfg);
    mpp_dec_cfg_deinit(cfg);
    if (ret != MPP_OK) {
        std::cerr << "Error: failed to set fast parse config" << std::endl;
    }

    std::cout << "MppDecoder initialized successfully for " << video_path << std::endl;
    return 0;
}

int MppDecoder::get_frame(DmaFrame &dma_frame) {
    MppPacket packet = nullptr;
    mpp_packet_init(&packet, buffer.data(), buffer.size());

    MppFrame frame = nullptr;
    MPP_RET ret = MPP_OK;

    // 循环直到成功解码出一帧
    while (1) {
        // 尝试从解码器获取一帧
        ret = mpi->decode_get_frame(ctx, &frame);
        if (ret == MPP_OK && frame) {
            if (mpp_frame_get_info_change(frame)) {
                // 分辨率等信息发生变化
                frame_width = mpp_frame_get_width(frame);
                frame_height = mpp_frame_get_height(frame);
                printf("Decoder got info change: %dx%d\n", frame_width, frame_height);

                // 让MPP自己管理帧缓冲区
                ret = mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, NULL);
                if (ret != MPP_OK) {
                    std::cerr << "Error: set ext buf group failed" << std::endl;
                    return -1;
                }

                mpp_frame_deinit(&frame);
                frame = nullptr;
                continue; // 继续解码
            }

            // 成功获取到一帧数据
            MppBuffer buffer = mpp_frame_get_buffer(frame);
            if (buffer) {
                dma_frame.fd = mpp_buffer_get_fd(buffer);
                dma_frame.width = mpp_frame_get_width(frame);
                dma_frame.height = mpp_frame_get_height(frame);
                dma_frame.format = mpp_frame_get_fmt(frame);
                dma_frame.va = mpp_buffer_get_ptr(buffer); // 获取虚拟地址
                dma_frame.frame = frame; // 传递句柄，用于后续释放
                return 0; // 成功返回
            } else {
                 mpp_frame_deinit(&frame);
                 frame = nullptr;
            }

        } else if (feof(file_in)) {
            // 文件结束
            return -1;
        }

        // 如果解码器需要更多数据，则从文件读取
        if (ret == MPP_ERR_DEC_NEED_MORE) {
            size_t read_size = fread(buffer.data(), 1, buffer.size(), file_in);
            if (read_size > 0) {
                mpp_packet_set_pos(packet, buffer.data());
                mpp_packet_set_length(packet, read_size);
                ret = mpi->decode_put_packet(ctx, packet);
                if (ret != MPP_OK) {
                    std::cerr << "Error: decode_put_packet failed" << std::endl;
                }
            } else if (feof(file_in)) {
                // 文件已读完，发送EOS packet
                mpp_packet_set_eos(packet);
                mpi->decode_put_packet(ctx, packet);
                printf("Sent EOS packet to decoder.\n");
            }
        }
    }
    mpp_packet_deinit(&packet);
    return -1;
}

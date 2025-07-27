// फाइल का नाम: include/common_types.h

#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

// 共享的检测框结构体，增加了独立的宽高缩放比例
// 供预处理计算和后处理坐标还原时使用
typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
    float scale_w = 1.0; // 宽度缩放比例
    float scale_h = 1.0; // 高度缩放比例
} BOX_RECT;

// 单个检测结果
typedef struct {
    BOX_RECT box;
    float prop;
    int cls_id;
} object_detect_result;

// 一帧图像的检测结果列表
typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

#endif // COMMON_TYPES_H
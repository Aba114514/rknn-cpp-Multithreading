#ifndef _RKNN_YOLO11_DEMO_POSTPROCESS_H_
#define _RKNN_YOLO11_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

class Yolo11;

// ** 关键修复：修改BOX_RECT以携带独立的宽高缩放比例 **
typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
    float scale_w; // 宽度缩放比例
    float scale_h; // 高度缩放比例
} BOX_RECT;

typedef struct {
    BOX_RECT box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
int post_process(Yolo11 *model_instance, rknn_output *outputs, BOX_RECT *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);

#endif //_RKNN_YOLO11_DEMO_POSTPROCESS_H_
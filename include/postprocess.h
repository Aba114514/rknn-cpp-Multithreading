// फाइल का नाम: include/postprocess.h

#ifndef _RKNN_YOLO_DEMO_POSTPROCESS_H_
#define _RKNN_YOLO_DEMO_POSTPROCESS_H_

#include "rknn_api.h"
#include "common_types.h" // 包含新的类型头文件

int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);

/**
 * @brief 对RKNN模型的输出进行后处理
 * @param outputs           [in] RKNN推理的原始输出数组
 * @param n_output          [in] 输出数组的长度
 * @param output_attrs      [in] RKNN输出张量的属性数组
 * @param is_quant          [in] 模型是否是量化模型
 * @param model_input_w     [in] 模型的输入宽度
 * @param model_input_h     [in] 模型的输入高度
 * @param letter_box        [in] 坐标变换信息
 * @param od_results        [out] 解码后的结构化检测结果
 * @return int 0表示成功
 */
int post_process(rknn_output *outputs, int n_output, rknn_tensor_attr* output_attrs, bool is_quant,
                 int model_input_w, int model_input_h,
                 BOX_RECT *letter_box, object_detect_result_list *od_results);

#endif //_RKNN_YOLO_DEMO_POSTPROCESS_H_
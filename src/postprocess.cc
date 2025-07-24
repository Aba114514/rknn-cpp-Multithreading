// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// ... (版权信息省略) ...

#include "Yolo11.hpp"
#include "postprocess.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <set>
#include <vector>

#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char *labels[OBJ_CLASS_NUM];

// ... (readLine, readLines, loadLabelName, CalculateOverlap, nms, quick_sort_indice_inverse 等辅助函数保持不变，直接从你已有的文件中复制过来即可) ...
// (为了简洁，这里省略这些未改变的辅助函数)

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
static char *readLine(FILE *fp, char *buffer, int *len){int ch;int i = 0;size_t buff_len = 0;buffer = (char *)malloc(buff_len + 1);if (!buffer)return NULL;while ((ch = fgetc(fp)) != '\n' && ch != EOF){buff_len++;void *tmp = realloc(buffer, buff_len + 1);if (tmp == NULL){free(buffer);return NULL;}buffer = (char *)tmp;buffer[i] = (char)ch;i++;}buffer[i] = '\0';*len = buff_len;if (ch == EOF && (i == 0 || ferror(fp))){free(buffer);return NULL;}return buffer;}
static int readLines(const char *fileName, char *lines[], int max_line){FILE *file = fopen(fileName, "r");char *s;int i = 0;int n = 0;if (file == NULL){printf("Open %s fail!\n", fileName);return -1;}while ((s = readLine(file, s, &n)) != NULL){lines[i++] = s;if (i >= max_line)break;}fclose(file);return i;}
static int loadLabelName(const char *locationFilename, char *label[]){printf("load lable %s\n", locationFilename);readLines(locationFilename, label, OBJ_CLASS_NUM);return 0;}
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,float ymax1){float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);float i = w * h;float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;return u <= 0.f ? 0.f : (i / u);}
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,int filterId, float threshold){for (int i = 0; i < validCount; ++i){int n = order[i];if (n == -1 || classIds[n] != filterId){continue;}for (int j = i + 1; j < validCount; ++j){int m = order[j];if (m == -1 || classIds[m] != filterId){continue;}float xmin0 = outputLocations[n * 4 + 0];float ymin0 = outputLocations[n * 4 + 1];float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];float xmin1 = outputLocations[m * 4 + 0];float ymin1 = outputLocations[m * 4 + 1];float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);if (iou > threshold){order[j] = -1;}}}return 0;}
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices){float key;int key_index;int low = left;int high = right;if (left < right){key_index = indices[left];key = input[left];while (low < high){while (low < high && input[high] <= key){high--;}input[low] = input[high];indices[low] = indices[high];while (low < high && input[low] >= key){low++;}input[high] = input[low];indices[high] = indices[low];}input[low] = key;indices[low] = key_index;quick_sort_indice_inverse(input, left, low - 1, indices);quick_sort_indice_inverse(input, low + 1, right, indices);}return low;}
inline static int32_t __clip(float val, float min, float max){float f = val <= min ? min : (val >= max ? max : val);return f;}
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale){float dst_val = (f32 / scale) + zp;int8_t res = (int8_t)__clip(dst_val, -128, 127);return res;}
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
static void compute_dfl(float* tensor, int dfl_len, float* box){for (int b=0; b<4; b++){float exp_t[dfl_len];float exp_sum=0;float acc_sum=0;for (int i=0; i< dfl_len; i++){exp_t[i] = exp(tensor[i+b*dfl_len]);exp_sum += exp_t[i];}for (int i=0; i< dfl_len; i++){acc_sum += exp_t[i]/exp_sum *i;}box[b] = acc_sum;}}

// 关键修复：修正了 process_i8 函数中的大括号不匹配问题
static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            } // ** 修复点：这里缺少了一个 '}' **

            int8_t max_score = -score_zp;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            if (max_score > score_thres_i8) {
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}


int post_process(Yolo11 *model_instance, rknn_output *outputs, BOX_RECT *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;

    int model_in_w = model_instance->get_model_width();
    int model_in_h = model_instance->get_model_height();
    int n_output = model_instance->get_io_num_n_output();
    bool is_quant = model_instance->get_is_quant();
    rknn_tensor_attr* output_attrs = model_instance->get_output_attrs();

    memset(od_results, 0, sizeof(object_detect_result_list));

    int dfl_len = output_attrs[0].dims[1] / 4;
    int output_per_branch = n_output / 3;

    for (int i = 0; i < 3; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = outputs[i*output_per_branch + 2].buf;
            score_sum_zp = output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;

        grid_h = output_attrs[box_idx].dims[2];
        grid_w = output_attrs[box_idx].dims[3];
        stride = model_in_h / grid_h;

        if (is_quant)
        {
            validCount += process_i8((int8_t *)outputs[box_idx].buf, output_attrs[box_idx].zp, output_attrs[box_idx].scale,
                                     (int8_t *)outputs[score_idx].buf, output_attrs[score_idx].zp, output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, conf_threshold);
        }
    }

    if (validCount <= 0)
    {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        // ** 核心修改点：使用独立的 scale_w 和 scale_h 进行坐标还原 **
        float scale_w = letter_box->scale_w;
        float scale_h = letter_box->scale_h;

        // 检测框在模型输入尺寸(e.g. 640x640)上的坐标
        float box_x = filterBoxes[n * 4 + 0];
        float box_y = filterBoxes[n * 4 + 1];
        float box_w = filterBoxes[n * 4 + 2];
        float box_h = filterBoxes[n * 4 + 3];

        // 还原到原始图像尺寸上的坐标
        float x1 = box_x * scale_w;
        float y1 = box_y * scale_h;
        float x2 = (box_x + box_w) * scale_w;
        float y2 = (box_y + box_h) * scale_h;

        int id = classId[n];
        float obj_conf = objProbs[i];

        // 使用原始图像的宽高进行clamp
        int raw_w = model_instance->get_model_width() * scale_w;
        int raw_h = model_instance->get_model_height() * scale_h;

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, raw_w));
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, raw_h));
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, raw_w));
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, raw_h));
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

char *coco_cls_to_name(int cls_id)
{
    if (cls_id >= OBJ_CLASS_NUM) { return (char*)"null"; }
    if (labels[cls_id]) { return labels[cls_id]; }
    return (char*)"null";
}

void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
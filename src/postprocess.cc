// फाइल का नाम: src/postprocess.cc

#include "postprocess.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <set>
#include <vector>
#include <algorithm> // <<-- 优化点#3：为 std::sort 和 std::unique 引入头文件

#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char *labels[OBJ_CLASS_NUM];

// ... (readLine, readLines, loadLabelName, CalculateOverlap, nms 等辅助函数保持不变) ...
// ====================================================================================
//               == 优化点#3：手写的快速排序函数可以被移除 ==
// ====================================================================================
// static int quick_sort_indice_inverse(...) { ... }
// ====================================================================================
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
static char *readLine(FILE *fp, char *buffer, int *len){int ch;int i = 0;size_t buff_len = 0;buffer = (char *)malloc(buff_len + 1);if (!buffer)return NULL;while ((ch = fgetc(fp)) != '\n' && ch != EOF){buff_len++;void *tmp = realloc(buffer, buff_len + 1);if (tmp == NULL){free(buffer);return NULL;}buffer = (char *)tmp;buffer[i] = (char)ch;i++;}buffer[i] = '\0';*len = buff_len;if (ch == EOF && (i == 0 || ferror(fp))){free(buffer);return NULL;}return buffer;}
static int readLines(const char *fileName, char *lines[], int max_line){FILE *file = fopen(fileName, "r");char *s;int i = 0;int n = 0;if (file == NULL){printf("Open %s fail!\n", fileName);return -1;}while ((s = readLine(file, s, &n)) != NULL){lines[i++] = s;if (i >= max_line)break;}fclose(file);return i;}
static int loadLabelName(const char *locationFilename, char *label[]){printf("load lable %s\n", locationFilename);readLines(locationFilename, label, OBJ_CLASS_NUM);return 0;}
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,float ymax1){float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);float i = w * h;float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;return u <= 0.f ? 0.f : (i / u);}
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> &classIds, std::vector<int> &order,int filterId, float threshold){for (int i = 0; i < validCount; ++i){int n = order[i];if (n == -1 || classIds[n] != filterId){continue;}for (int j = i + 1; j < validCount; ++j){int m = order[j];if (m == -1 || classIds[m] != filterId){continue;}float xmin0 = outputLocations[n * 4 + 0];float ymin0 = outputLocations[n * 4 + 1];float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];float xmin1 = outputLocations[m * 4 + 0];float ymin1 = outputLocations[m * 4 + 1];float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);if (iou > threshold){order[j] = -1;}}}return 0;}
inline static int32_t __clip(float val, float min, float max){float f = val <= min ? min : (val >= max ? max : val);return f;}
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale){float dst_val = (f32 / scale) + zp;int8_t res = (int8_t)__clip(dst_val, -128, 127);return res;}
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
static void compute_dfl(float* tensor, int dfl_len, float* box){for (int b=0; b<4; b++){float exp_t[dfl_len];float exp_sum=0;float acc_sum=0;for (int i=0; i< dfl_len; i++){exp_t[i] = exp(tensor[i+b*dfl_len]);exp_sum += exp_t[i];}for (int i=0; i< dfl_len; i++){acc_sum += exp_t[i]/exp_sum *i;}box[b] = acc_sum;}}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    // ... 函数内部逻辑保持不变 ...
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = 0;
    if (score_sum_tensor != nullptr) {
        score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);
    }

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            int8_t max_score = -128;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            if (max_class_id != -1) {
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


int post_process(rknn_output *outputs, int n_output, rknn_tensor_attr* output_attrs, bool is_quant,
                 int model_in_w, int model_in_h,
                 BOX_RECT *letter_box, object_detect_result_list *od_results)
{
    // 优化点#2：为vector预分配内存
    int output_per_branch = n_output / 3;
    int max_possible_boxes = 0;
    for (int i = 0; i < 3; i++) {
        int box_idx = i * output_per_branch;
        max_possible_boxes += output_attrs[box_idx].dims[2] * output_attrs[box_idx].dims[3];
    }
    std::vector<float> filterBoxes;
    filterBoxes.reserve(max_possible_boxes * 4);
    std::vector<float> objProbs;
    objProbs.reserve(max_possible_boxes);
    std::vector<int> classId;
    classId.reserve(max_possible_boxes);

    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;

    memset(od_results, 0, sizeof(object_detect_result_list));

    int dfl_len = output_attrs[0].dims[1] / 4;

    for (int i = 0; i < 3; i++)
    {
        // ... 此处 for 循环内部逻辑保持不变 ...
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
                                     filterBoxes, objProbs, classId, BOX_THRESH);
        }
    }

    if (validCount <= 0)
    {
        return 0;
    }

    std::vector<int> indexArray;
    indexArray.reserve(validCount); // 也可以为indexArray预分配内存
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }

    // ====================================================================================
    //      == 优化点#3.1: 使用 std::sort 替代手写的 quick_sort_indice_inverse ==
    // ====================================================================================
    std::sort(indexArray.begin(), indexArray.end(),
              [&objProbs](int a, int b) {
                  return objProbs[a] > objProbs[b];
              });
    // ====================================================================================


    // ====================================================================================
    //      == 优化点#3.2: 使用 vector+sort+unique 替代 std::set 获取唯一类别 ==
    // ====================================================================================
    if (classId.empty()) { // 安全检查
        return 0;
    }
    std::vector<int> unique_classes = classId;
    std::sort(unique_classes.begin(), unique_classes.end());
    unique_classes.erase(std::unique(unique_classes.begin(), unique_classes.end()), unique_classes.end());

    for (auto c : unique_classes)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, NMS_THRESH);
    }
    // ====================================================================================

    int last_count = 0;
    od_results->count = 0;

    for (int i = 0; i < validCount; ++i)
    {
        // ... 此处 for 循环内部逻辑保持不变 ...
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float scale_w = letter_box->scale_w;
        float scale_h = letter_box->scale_h;

        float box_x = filterBoxes[n * 4 + 0];
        float box_y = filterBoxes[n * 4 + 1];
        float box_w = filterBoxes[n * 4 + 2];
        float box_h = filterBoxes[n * 4 + 3];

        float x1 = box_x * scale_w;
        float y1 = box_y * scale_h;
        float x2 = (box_x + box_w) * scale_w;
        float y2 = (box_y + box_h) * scale_h;

        int id = classId[n];
        float obj_conf = objProbs[i]; // 注意：这里objProbs的顺序没变，但indexArray的顺序变了，所以要用indexArray[i]来索引

        // 计算原始图像的宽高用于clamp
        int raw_w = model_in_w * scale_w;
        int raw_h = model_in_h * scale_h;

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, raw_w));
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, raw_h));
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, raw_w));
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, raw_h));
        od_results->results[last_count].prop = objProbs[n]; // 修正：应该使用 objProbs[n] 获取正确置信度
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}


int init_post_process()
{
    return loadLabelName(LABEL_NALE_TXT_PATH, labels);
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

char *coco_cls_to_name(int cls_id)
{
    if (cls_id >= OBJ_CLASS_NUM || cls_id < 0) { return (char*)"null"; }
    if (labels[cls_id]) { return labels[cls_id]; }
    return (char*)"null";
}
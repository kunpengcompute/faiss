/*
test API list:
1. krl_store_LUT8Handle
2. krl_build_LUT8Handle_fromfile
3. krl_store_distanceHandle
4. krl_build_distanceHandle_fromfile
*/

#include "tools.h"
extern "C" {
#include "krl.h"
#include "krl_internal.h"
}
#include <string>

int compare_DistanceHandle(const KRLDistanceHandle *kdh1, const KRLDistanceHandle *kdh2, std::string &output_str)
{
    output_str = "Error list:";
    int same = 1;
    if (kdh1->data_bits != kdh2->data_bits) {
        same = 0;
        output_str += " data_bits,";
    }
    if (kdh1->full_data_bits != kdh2->full_data_bits) {
        same = 0;
        output_str += " full_data_bits,";
    }
    if (kdh1->M != kdh2->M) {
        same = 0;
        output_str += " M,";
    }
    if (kdh1->blocksize != kdh2->blocksize) {
        same = 0;
        output_str += " blocksize,";
    }
    if (kdh1->d != kdh2->d) {
        same = 0;
        output_str += " d,";
    }
    if (kdh1->ny != kdh2->ny) {
        same = 0;
        output_str += " ny,";
    }
    if (kdh1->ceil_ny != kdh2->ceil_ny) {
        same = 0;
        output_str += " ceil_ny,";
    }
    if (kdh1->quanted_scale != kdh2->quanted_scale) {
        same = 0;
        output_str += " quanted_scale,";
    }
    if (kdh1->quanted_bias != kdh2->quanted_bias) {
        same = 0;
        output_str += " quanted_bias,";
    }
    if (kdh1->metric_type != kdh2->metric_type) {
        same = 0;
        output_str += " metric_type,";
    }
    if (kdh1->quanted_bytes != kdh2->quanted_bytes) {
        same = 0;
        output_str += " quanted_bytes,";
    } else {
        for (int i = 0; i < kdh1->quanted_bytes; ++i) {
            if (kdh1->quanted_codes[i] != kdh2->quanted_codes[i]) {
                same = 0;
                output_str += " quanted_codes,";
                break;
            }
        }
    }
    if (kdh1->transposed_bytes != kdh2->transposed_bytes) {
        same = 0;
        output_str += " transposed_bytes,";
    } else {
        for (int i = 0; i < kdh1->transposed_bytes / sizeof(float); ++i) {
            if (kdh1->transposed_codes[i] != kdh2->transposed_codes[i]) {
                same = 0;
                output_str += " transposed_codes,";
                break;
            }
        }
    }
    if (same == 1) {
        output_str = "Test pass!\n";
        return 1;
    }
    output_str.replace(output_str.end() - 2, output_str.end(), ',', '\n');
    return 0;
}

int compare_LUT8Handle(const KRLLUT8bHandle *klh1, const KRLLUT8bHandle *klh2, std::string &output_str)
{
    output_str = "Error list:";
    int same = 1;
    if (klh1->capacity != klh2->capacity) {
        same = 0;
        output_str += " capacity,";
    } else if (klh1->capacity > 0) {
        if (klh1->use_idx != klh2->use_idx) {
            same = 0;
            output_str += " use_idx,";
        } else if (klh1->use_idx) {
            for (int i = 0; i < klh1->capacity; ++i) {
                if (klh1->idx_buffer[i] != klh2->idx_buffer[i]) {
                    same = 0;
                    output_str += " idx_buffer,";
                    break;
                }
            }
        }
        for (int i = 0; i < klh1->capacity; ++i) {
            if (klh1->distance_buffer[i] != klh2->distance_buffer[i]) {
                same = 0;
                output_str += " distance_buffer,";
                break;
            }
        }
    }
    if (same == 1) {
        output_str = "Test pass!\n";
        return 1;
    }
    output_str.replace(output_str.end() - 2, output_str.end(), ',', '\n');

    return 0;
}

void check_LUT8_handle(size_t capacity)
{
    KRLLUT8bHandle *gt_klh = NULL, *file_klh = NULL;
    krl_create_LUT8b_handle(&gt_klh, 1, capacity);
    size_t *g_idx = krl_get_idx_pointer(gt_klh);
    float *g_dis = krl_get_dist_pointer(gt_klh);
    init_vector(capacity, g_idx, (int)0, (int)capacity);
    init_vector(capacity, g_dis, (int)0, (int)32768);
    FILE *LUT_file = fopen("./indices/LUT8_handle.index", "w");
    krl_store_LUT8Handle(LUT_file, gt_klh);
    fclose(LUT_file);
    LUT_file = fopen("./indices/LUT8_handle.index", "r");
    krl_build_LUT8Handle_fromfile(LUT_file, &file_klh);
    fclose(LUT_file);
    std::string s;
    compare_LUT8Handle(gt_klh, file_klh, s);
    printf("%s", s.c_str());
    krl_clean_LUT8b_handle(&gt_klh);
    krl_clean_LUT8b_handle(&file_klh);
}

void check_DistanceHandle_handle(size_t codes_num, size_t dim)
{
    KRLDistanceHandle *gt_kdh = NULL, *file_kdh = NULL;
    float *x = new float[codes_num * dim];
    init_vector(codes_num * dim, x, (int)0, (int)32768);
    krl_create_reorder_handle(&gt_kdh, 1, 3, codes_num, dim, 0, (const uint8_t *)x, codes_num * dim * 4);
    FILE *Distance_file = fopen("./indices/DistanceHandle.index", "w");
    krl_store_distanceHandle(Distance_file, gt_kdh);
    fclose(Distance_file);
    Distance_file = fopen("./indices/DistanceHandle.index", "r");
    krl_build_distanceHandle_fromfile(Distance_file, &file_kdh);
    fclose(Distance_file);
    std::string s;
    compare_DistanceHandle(gt_kdh, file_kdh, s);
    printf("%s", s.c_str());
    krl_clean_distance_handle(&gt_kdh);
    krl_clean_distance_handle(&file_kdh);
}

int main()
{
    printf("begin test read store func\n");
    check_LUT8_handle(60000);
    check_DistanceHandle_handle(60000, 784);
    return 0;
}
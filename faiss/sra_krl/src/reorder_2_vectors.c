/*
   Copyright 2025 Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"
#include <stdio.h>

#define Obtain_L2_threshold(dis) (dis * 1.0625)
#define Obtain_IP_threshold(dis) (dis > 0 ? dis * 0.9375 : dis * 1.0625)

/*
 * @brief Find the first index with a negative value in the label array.
 * @param n The size of the label array.
 * @param label Pointer to the label array.
 * @return The index of the first negative value.
 */
static inline idx_t get_first_inf(idx_t n, idx_t *label)
{
    idx_t left = -1, right = n - 1;
    while (left < right - 1) {
        idx_t mid = (left + right) >> 1;
        if (label[mid] < 0) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return right;
}

/*
 * @brief Create a continuous index array starting from a given value.
 * @param idx Pointer to the destination index array.
 * @param ny The number of elements to create.
 * @param begin The starting value for the index array.
 */
static inline void create_continuous_idx(int64_t *idx, int64_t ny, int64_t begin)
{
    const int64_t last = begin + ny;
    for (; begin < last; ++begin, ++idx) {
        *idx = begin;
    }
}

/*
 * @brief Reorder vectors based on distance calculations.
 * @param kdh Pointer to the distance handle containing configuration and data.
 * @param base_k The number of base vectors.
 * @param base_dis Pointer to the base distance array.
 * @param base_idx Pointer to the base index array.
 * @param query_vector Pointer to the query vector.
 * @param k The number of nearest neighbors to find.
 * @param dis Pointer to the distance array for results.
 * @param idx Pointer to the index array for results.
 * @param query_vector_size Length of query_vector.
 */
int krl_reorder_2_vector(const KRLDistanceHandle *kdh, int64_t base_k, float *base_dis, int64_t *base_idx,
    const float *query_vector, int64_t k, float *dis, int64_t *idx, size_t query_vector_size)
{
    const size_t use_parm = kdh->blocksize;
    const size_t compute_bits = kdh->data_bits;
    const int metric_type = kdh->metric_type;
    const size_t d = kdh->d;
    const uint8_t *quanted_index = kdh->quanted_codes;
    const float *base_vector = kdh->transposed_codes;
    const size_t codes_num = kdh->ny;
    if (unlikely(base_idx[base_k - 1] < 0)) {
        base_k = get_first_inf(base_k, base_idx);
        if (base_k <= k) {
            if (metric_type == METRIC_L2) {
                krl_reorder_2_heaps_asce(base_k, idx, dis, base_k, base_idx, base_dis);
                for (int i = base_k; i < k; ++i) {
                    idx[i] = -1;
                    dis[i] = INFINITY;
                }
            } else {
                krl_reorder_2_heaps_desc(base_k, idx, dis, base_k, base_idx, base_dis);
                for (int i = base_k; i < k; ++i) {
                    idx[i] = -1;
                    dis[i] = -INFINITY;
                }
            }
            return SUCCESS;
        }
    }
    if (compute_bits == 8) {
        if (metric_type == METRIC_L2) {
            uint8_t *quant_x = (uint8_t *)malloc(d * sizeof(uint8_t));
            if (quant_x == NULL) {
                printf("Error: FAILALLOC in krl_reorder_2_vector\n");
                return FAILALLOC;
            }
            if (use_parm != 0) {
                const size_t full_data_bits = kdh->full_data_bits;
                const float qscale = kdh->quanted_scale;
                const float qbias = kdh->quanted_bias;
                quant_u8_with_parm(query_vector, d, quant_x, qscale, qbias);
                /* first rough calculation */
                krl_L2sqr_by_idx_u8f32(base_dis, quant_x, (const uint8_t *)quanted_index, base_idx, d, base_k);
                if (full_data_bits == 32) {
                    /* Obtains recalculation threshold */
                    krl_obtion_topk_heap_asce(k, dis, base_k, base_dis);
                    /* Amplification threshold */
                    const float target = Obtain_L2_threshold(dis[0]); /* 1/16 */
                    const idx_t k_base2 = Adapt_reorder_asce(base_dis, base_idx, base_k, target);
                    krl_L2sqr_by_idx(base_dis, query_vector, base_vector, base_idx, d, k_base2);
                    krl_reorder_2_heaps_asce(k, idx, dis, k_base2, base_idx, base_dis);
                } else {
                    krl_reorder_2_heaps_asce(k, idx, dis, base_k, base_idx, base_dis);
                }
            } else {
                quant_u8(query_vector, d, quant_x);
                krl_L2sqr_by_idx_u8f32(base_dis, quant_x, (const uint8_t *)quanted_index, base_idx, d, base_k);
                krl_reorder_2_heaps_asce(k, idx, dis, base_k, base_idx, base_dis);
            }
            free(quant_x);
        } else {
            int8_t *quant_x = (int8_t *)malloc(d * sizeof(int8_t));
            if (quant_x == NULL) {
                printf("Error: FAILALLOC in krl_reorder_2_vector\n");
                return FAILALLOC;
            }
            if (use_parm != 0) {
                const size_t full_data_bits = kdh->full_data_bits;
                const float qscale = kdh->quanted_scale;
                quant_s8_with_parm(query_vector, d, quant_x, qscale);
                /* first rough calculation */
                krl_inner_product_by_idx_s8f32(
                    base_dis, quant_x, (const int8_t *)quanted_index, base_idx, d, base_k);
                if (full_data_bits == 32) {
                    /* Obtains recalculation threshold */
                    krl_obtion_topk_heap_desc(k, dis, base_k, base_dis);
                    /* Amplification threshold */
                    const float target = Obtain_IP_threshold(dis[0]); /* 1/32 */
                    const idx_t k_base2 = Adapt_reorder_desc(base_dis, base_idx, base_k, target);
                    /* Fine-grained computing */
                    krl_inner_product_by_idx(base_dis, query_vector, base_vector, base_idx, d, k_base2);
                    krl_reorder_2_heaps_desc(k, idx, dis, k_base2, base_idx, base_dis);
                } else {
                    krl_reorder_2_heaps_desc(k, idx, dis, base_k, base_idx, base_dis);
                }
            } else {
                quant_s8(query_vector, d, quant_x);
                krl_inner_product_by_idx_s8f32(
                    base_dis, quant_x, (const int8_t *)quanted_index, base_idx, d, base_k);
                krl_reorder_2_heaps_desc(k, idx, dis, base_k, base_idx, base_dis);
            }
            free(quant_x);
        }
    } else if (compute_bits == 16) {
        float16_t *quant_x = (float16_t *)malloc(d * sizeof(float16_t));
        if (quant_x == NULL) {
            printf("Error: FAILALLOC in krl_reorder_2_vector\n");
            return FAILALLOC;
        }
        quant_f16(query_vector, d, quant_x);
        if (metric_type == METRIC_L2) {
            krl_L2sqr_by_idx_f16f32(
                base_dis, (const uint16_t *)quant_x, (const uint16_t *)quanted_index, base_idx, d, base_k);
            krl_reorder_2_heaps_asce(k, idx, dis, base_k, base_idx, base_dis);
        } else {
            krl_inner_product_by_idx_f16f32(
                base_dis, (const uint16_t *)quant_x, (const uint16_t *)quanted_index, base_idx, d, base_k);
            krl_reorder_2_heaps_desc(k, idx, dis, base_k, base_idx, base_dis);
        }
        free(quant_x);
    } else if (metric_type == METRIC_L2) {
        krl_L2sqr_by_idx(base_dis, query_vector, (const float *)base_vector, base_idx, d, base_k);
        krl_reorder_2_heaps_asce(k, idx, dis, base_k, base_idx, base_dis);
    } else {
        krl_inner_product_by_idx(base_dis, query_vector, (const float *)base_vector, base_idx, d, base_k);
        krl_reorder_2_heaps_desc(k, idx, dis, base_k, base_idx, base_dis);
    }
    return SUCCESS;
}

/*
 * @brief Reorder vectors based on distance calculations for continuous data.
 * @param kdh Pointer to the distance handle containing configuration and data.
 * @param base_k The number of base vectors.
 * @param begin_id The starting index for continuous processing.
 * @param query_vector Pointer to the query vector.
 * @param k The number of nearest neighbors to find.
 * @param dis Pointer to the distance array for results.
 * @param idx Pointer to the index array for results.
 * @param query_vector_size Length of query_vector.
 */
int krl_reorder_2_vector_continuous(const KRLDistanceHandle *kdh, int64_t base_k, int64_t begin_id,
    const float *query_vector, int64_t k, float *dis, int64_t *idx, size_t query_vector_size)
{
    const size_t use_parm = kdh->blocksize;
    const size_t compute_bits = kdh->data_bits;
    const int metric_type = kdh->metric_type;
    const size_t d = kdh->d;
    const float *base_vector = kdh->transposed_codes;
    const uint8_t *quanted_index = kdh->quanted_codes + begin_id * d;
    const size_t codes_num = kdh->ny;
    float *base_dis = (float *)malloc(base_k * sizeof(float));
    if (base_dis == NULL) {
        printf("Error: FAILALLOC in krl_reorder_2_vector_continuous\n");
        return FAILALLOC;
    }
    int64_t *base_idx = (int64_t *)malloc(base_k * sizeof(int64_t));
    if (base_idx == NULL) {
        printf("Error: FAILALLOC in krl_reorder_2_vector_continuous\n");
        free(base_dis);
        return FAILALLOC;
    }
    create_continuous_idx(base_idx, base_k, begin_id);
    if (compute_bits == 8) {
        if (metric_type == METRIC_L2) {
            uint8_t *quant_x = (uint8_t *)malloc(d * sizeof(uint8_t));
            if (quant_x == NULL) {
                printf("Error: FAILALLOC in krl_reorder_2_vector_continuous\n");
                free(base_dis);
                free(base_idx);
                return FAILALLOC;
            }
            if (use_parm != 0) {
                quant_u8_with_parm(query_vector, d, quant_x, kdh->quanted_scale, kdh->quanted_bias);
                /* First rough calculation */
                krl_L2sqr_ny_u8f32(base_dis, quant_x, (const uint8_t *)quanted_index, base_k, d);
                /* Obtains recalculation threshold */
                krl_obtion_topk_heap_asce(k, dis, base_k, base_dis);
                const float target = Obtain_L2_threshold(dis[0]); /* 1/16 */
                idx_t k_base2 = Adapt_reorder_asce(base_dis, base_idx, base_k, target);
                /* Fine-grained computing */
                krl_L2sqr_by_idx(base_dis, query_vector, base_vector, base_idx, d, k_base2);
                /* Secondary reflow */
                krl_reorder_2_heaps_asce(k, idx, dis, k_base2, base_idx, base_dis);
            } else {
                quant_u8(query_vector, d, quant_x);
                krl_L2sqr_ny_u8f32(base_dis, quant_x, (const uint8_t *)quanted_index, base_k, d);
                krl_reorder_2_heaps_asce(k, idx, dis, base_k, base_idx, base_dis);
            }
            free(quant_x);
        } else {
            int8_t *quant_x = (int8_t *)malloc(d * sizeof(int8_t));
            if (quant_x == NULL) {
                printf("Error: FAILALLOC in krl_reorder_2_vector_continuous\n");
                free(base_dis);
                free(base_idx);
                return FAILALLOC;
            }
            if (use_parm != 0) {
                quant_s8_with_parm(query_vector, d, quant_x, kdh->quanted_scale);
                krl_inner_product_ny_s8f32(base_dis, quant_x, (const int8_t *)quanted_index, base_k, d);
                krl_obtion_topk_heap_desc(k, dis, base_k, base_dis);
                const float target = Obtain_IP_threshold(dis[0]); /* 1/32 */
                idx_t k_base2 = Adapt_reorder_desc(base_dis, base_idx, base_k, target);
                krl_inner_product_by_idx(base_dis, query_vector, base_vector, base_idx, d, k_base2);
                krl_reorder_2_heaps_desc(k, idx, dis, k_base2, base_idx, base_dis);
            } else {
                quant_s8(query_vector, d, quant_x);
                krl_inner_product_ny_s8f32(base_dis, quant_x, (const int8_t *)quanted_index, base_k, d);
                krl_reorder_2_heaps_desc(k, idx, dis, base_k, base_idx, base_dis);
            }
            free(quant_x);
        }
    } else if (compute_bits == 16) {
        float16_t *quant_x = (float16_t *)malloc(d * sizeof(float16_t));
        if (quant_x == NULL) {
            printf("Error: FAILALLOC in krl_reorder_2_vector_continuous\n");
            free(base_dis);
            free(base_idx);
            return FAILALLOC;
        }
        quant_f16(query_vector, d, quant_x);
        if (metric_type == METRIC_L2) {
            krl_L2sqr_by_idx_f16f32(base_dis,
                (const uint16_t *)quant_x,
                (const uint16_t *)(kdh->quanted_codes),
                base_idx,
                d,
                base_k);
            krl_reorder_2_heaps_asce(k, idx, dis, base_k, base_idx, base_dis);
        } else {
            krl_inner_product_by_idx_f16f32(base_dis,
                (const uint16_t *)quant_x,
                (const uint16_t *)(kdh->quanted_codes),
                base_idx,
                d,
                base_k);
            krl_reorder_2_heaps_desc(k, idx, dis, base_k, base_idx, base_dis);
        }
        free(quant_x);
    } else if (metric_type == METRIC_L2) {
        krl_L2sqr_ny(base_dis, query_vector, (const float *)base_vector + begin_id * d, base_k, d);
        krl_reorder_2_heaps_asce(k, idx, dis, base_k, base_idx, base_dis);
    } else {
        krl_inner_product_ny(base_dis, query_vector, (const float *)base_vector + begin_id * d, base_k, d);
        krl_reorder_2_heaps_desc(k, idx, dis, base_k, base_idx, base_dis);
    }
    free(base_idx);
    free(base_dis);
    return SUCCESS;
}

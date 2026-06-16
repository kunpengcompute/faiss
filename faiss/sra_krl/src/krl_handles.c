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
#include "safe_memory.h"
#include <stdio.h>

/*
 * @brief Reset the distance handle to initial state.
 * @param kdh Pointer to the distance handle to be reset.
 */
static void reset_distance_handle(KRLDistanceHandle **kdh)
{
    if (kdh == NULL || (*kdh) != NULL) {
        printf("Error: INVALPOINTER in reset_distance_handle\n");
        return;
    }
    (*kdh)->data_bits = 0;
    (*kdh)->full_data_bits = 0;
    (*kdh)->M = 0;
    (*kdh)->blocksize = 0;
    (*kdh)->d = 0;
    (*kdh)->ny = 0;
    (*kdh)->ceil_ny = 0;
    (*kdh)->quanted_scale = 1;
    (*kdh)->quanted_bias = 0;
    (*kdh)->metric_type = 0;
    if ((*kdh)->quanted_bytes > 0 && (*kdh)->quanted_codes) {
        free((*kdh)->quanted_codes);
    }
    if ((*kdh)->transposed_bytes > 0 && (*kdh)->transposed_codes) {
        free((*kdh)->transposed_codes);
    }
    (*kdh)->quanted_bytes = 0;
    (*kdh)->transposed_bytes = 0;
    (*kdh)->quanted_codes = NULL;
    (*kdh)->transposed_codes = NULL;
}

/*
 * @brief Initialize the parameters of the distance handle.
 * @param kdh Pointer to the distance handle to be initialized.
 * @param quant_bits Quantization bits for the handle.
 * @param blocksize Block size for matrix operations.
 * @param codes_num Number of code vectors.
 * @param dim Dimension of the vectors.
 * @param num_base Number of bases.
 * @param metric_type Metric type for distance calculation.
 * @return SUCCESS if initialization is successful, otherwise error code.
 */
static inline int init_handle_param(KRLDistanceHandle **kdh, size_t quant_bits, size_t blocksize, size_t codes_num,
    size_t dim, size_t num_base, int metric_type)
{
    (*kdh) = (KRLDistanceHandle *)malloc(sizeof(KRLDistanceHandle));
    if ((*kdh) == NULL) {
        printf("Error: FAILALLOC in init_handle_param\n");
        return FAILALLOC;
    }
    (*kdh)->data_bits = quant_bits;
    (*kdh)->blocksize = blocksize;
    (*kdh)->ny = codes_num;
    (*kdh)->d = dim;
    (*kdh)->M = num_base;
    (*kdh)->metric_type = metric_type;
    return SUCCESS;
}

/*
 * @brief Create a distance handle for vector distance calculations.
 * @param kdh Pointer to the distance handle to be created.
 * @param accu_level Accuracy level for quantization (0-3).
 * @param blocksize Block size for matrix operations (16, 32, or 64).
 * @param codes_num Number of code vectors.
 * @param dim Dimension of the vectors.
 * @param num_base Number of bases.
 * @param metric_type Metric type for distance calculation.
 * @param codes Pointer to the input code vectors.
 * @param codes_size Length of codes.
 * @return SUCCESS if handle creation is successful, otherwise error code.
 */
int krl_create_distance_handle(KRLDistanceHandle **kdh, size_t accu_level, size_t blocksize, size_t codes_num,
    size_t dim, size_t num_base, int metric_type, const uint8_t *codes, size_t codes_size)
{
    if (kdh == NULL || (*kdh) != NULL) {
        printf("Error: INVALPOINTER in krl_create_distance_handle\n");
        return INVALPOINTER;
    }
    if (accu_level < 1 || accu_level > 3 || num_base < 1 || num_base > 65535 ||
        (blocksize != 16 && blocksize != 32 && blocksize != 64) || codes_num < 1 || codes_num > 1ULL << 30 || dim < 1 ||
        dim > 65535 || (metric_type != 0 && metric_type != 1) || codes == NULL ||
        codes_size < num_base * codes_num * dim * 4) {
        printf("Error: INVALPARAM in krl_create_distance_handle\n");
        return INVALPARAM;
    }
    const size_t quant_bits = 4 << accu_level;
    /* Initializes the handle and its parameters. */
    int singal = init_handle_param(kdh, quant_bits, blocksize, codes_num, dim, num_base, metric_type);
    if (singal != SUCCESS) {
        return singal;
    }
    /* The value of ny must be a multiple of the blocksize. */
    (*kdh)->ceil_ny = (codes_num + blocksize - 1) & (-blocksize);
    /* Full-precision matrix-vector multiplication, transpose required */
    if (quant_bits == 32) {
        (*kdh)->quanted_bytes = 0;
        (*kdh)->quanted_codes = NULL;
        (*kdh)->transposed_bytes = num_base * (*kdh)->ceil_ny * dim * sizeof(float);
        (*kdh)->transposed_codes =
            (float *)aligned_alloc(KRL_DEFAULT_ALIGNED, num_base * (*kdh)->ceil_ny * dim * sizeof(float));
        if ((*kdh)->transposed_codes == NULL) {
            krl_clean_distance_handle(kdh);
            printf("Error: FAILALLOC in krl_create_distance_handle\n");
            return FAILALLOC;
        }
        for (size_t m = 0; m < num_base; m++) {
            int ret = krl_matrix_block_transpose((const uint32_t *)(codes + sizeof(float) * m * codes_num * dim),
                codes_num,
                dim,
                blocksize,
                (uint32_t *)((*kdh)->transposed_codes + m * (*kdh)->ceil_ny * dim),
                (*kdh)->ceil_ny * dim * sizeof(uint32_t));
            if (ret != 0) {
                krl_clean_distance_handle(kdh);
                printf("Error: UNSAFEMEM in krl_create_distance_handle\n");
                return UNSAFEMEM;
            }
        }
        /* Quantization matrix-vector multiplication, which needs to be quantized and does not need to be transposed. */
    } else {
        const size_t codes_length = num_base * codes_num * dim;
        (*kdh)->transposed_bytes = 0;
        (*kdh)->transposed_codes = NULL;
        (*kdh)->quanted_bytes = codes_length * (quant_bits >> 3);
        (*kdh)->quanted_codes = (uint8_t *)aligned_alloc(KRL_DEFAULT_ALIGNED, codes_length * (quant_bits >> 3));
        if ((*kdh)->quanted_codes == NULL) {
            krl_clean_distance_handle(kdh);
            printf("Error: FAILALLOC in krl_create_distance_handle\n");
            return FAILALLOC;
        }
        if (quant_bits == 8) {
            (*kdh)->blocksize = 0;
            (*kdh)->quanted_scale = 1;
            (*kdh)->quanted_bias = 0;
            quant_sq8((idx_t)codes_length,
                (const float *)codes,
                (uint8_t *)(*kdh)->quanted_codes,
                metric_type,
                (*kdh)->blocksize,
                (*kdh)->quanted_scale,
                (*kdh)->quanted_bias);
        } else if (quant_bits == 16) {
            (*kdh)->blocksize = 0;
            (*kdh)->quanted_scale = 1;
            (*kdh)->quanted_bias = 0;
            quant_f16((const float *)codes, (idx_t)codes_length, (float16_t *)(*kdh)->quanted_codes);
        }
    }
    return SUCCESS;
}

/*
 * @brief Create a reorder handle for vector distance calculations.
 * @param kdh Pointer to the reorder handle to be created.
 * @param accu_level Accuracy level for quantization (0-3).
 * @param full_accu_level Full accuracy level for high-precision calculations.
 * @param codes_num Number of code vectors.
 * @param dim Dimension of the vectors.
 * @param metric_type Metric type for distance calculation.
 * @param codes Pointer to the input code vectors.
 * @param codes_size Length of codes.
 * @return SUCCESS if handle creation is successful, otherwise error code.
 */
int krl_create_reorder_handle(KRLDistanceHandle **kdh, size_t accu_level, size_t full_accu_level, size_t codes_num,
    size_t dim, int metric_type, const uint8_t *codes, size_t codes_size)
{
    if (kdh == NULL || (*kdh) != NULL) {
        printf("Error: INVALPOINTER in krl_create_reorder_handle\n");
        return INVALPOINTER;
    }
    if (accu_level < 1 || accu_level > 3 || full_accu_level < accu_level || full_accu_level > 3 || codes_num < 1 ||
        codes_num > 1ULL << 30 || dim < 1 || (metric_type != 0 && metric_type != 1) || codes == NULL ||
        codes_size != codes_num * dim * 4) {
        printf("Error: INVALPARAM in krl_create_reorder_handle\n");
        return INVALPARAM;
    }
    const size_t full_data_bits = 4 << full_accu_level;
    const size_t quant_bits = 4 << accu_level;
    /* Initializes the handle and its parameters. */
    int singal = init_handle_param(kdh, quant_bits, 0, codes_num, dim, 0, metric_type);
    if (singal != SUCCESS) {
        return singal;
    }
    const idx_t codes_length = codes_num * dim;
    /* The maximum precision is 8 bits, 16 bits, or 32 bits, which is not lower than the quantization precision. */
    (*kdh)->full_data_bits = full_data_bits;
    if (quant_bits <= full_data_bits) {
        /* quant_bits is less than 32 bits and needs to be quantized. */
        if (quant_bits < 32) {
            (*kdh)->quanted_bytes = codes_num * dim * (quant_bits >> 3);
            (*kdh)->quanted_codes = (uint8_t *)aligned_alloc(KRL_DEFAULT_ALIGNED, codes_num * dim * (quant_bits >> 3));
            if ((*kdh)->quanted_codes == NULL) {
                krl_clean_distance_handle(kdh);
                printf("Error: FAILALLOC in krl_create_reorder_handle\n");
                return FAILALLOC;
            }
            /* Quantization */
            if (quant_bits == 8) {
                (*kdh)->blocksize = compute_quant_parm(codes_length,
                    (const float *)codes,
                    metric_type,
                    256,
                    &((*kdh)->quanted_scale),
                    &((*kdh)->quanted_bias));
                quant_sq8((idx_t)codes_length,
                    (const float *)codes,
                    (uint8_t *)(*kdh)->quanted_codes,
                    metric_type,
                    (*kdh)->blocksize,
                    (*kdh)->quanted_scale,
                    (*kdh)->quanted_bias);
            } else if (quant_bits == 16) {
                (*kdh)->blocksize = 0; /* No parms is required. */
                (*kdh)->quanted_scale = 1;
                (*kdh)->quanted_bias = 0;
                quant_f16((const float *)codes, (idx_t)codes_length, (float16_t *)(*kdh)->quanted_codes);
            }
        } else {
            (*kdh)->quanted_bytes = 0;
            (*kdh)->quanted_codes = NULL;
        }
        /* High-precision rearrangement is required for 8-bit quantization, and 32-bit data needs to be saved for
         * full-precision calculation. */
        if ((quant_bits == 32) || (quant_bits == 8 && 32 == full_data_bits)) {
            /* Allocates memory for high-precision transposed_codes. */
            (*kdh)->transposed_bytes = codes_length * (full_data_bits >> 3);
            (*kdh)->transposed_codes =
                (float *)aligned_alloc(KRL_DEFAULT_ALIGNED, codes_length * (full_data_bits >> 3));
            if ((*kdh)->transposed_codes == NULL) {
                krl_clean_distance_handle(kdh);
                printf("Error: FAILALLOC in krl_create_reorder_handle\n");
                return FAILALLOC;
            }
            if (full_data_bits == 16) {
                quant_f16((const float *)codes, codes_length, (float16_t *)(*kdh)->transposed_codes);
            } else if (full_data_bits == 32) {
                int ret = SafeMemory::CheckAndMemcpy(
                    (*kdh)->transposed_codes, codes_length * sizeof(float), codes, codes_length * sizeof(float));
                if (ret != 0) {
                    krl_clean_distance_handle(kdh);
                    printf("Error: UNSAFEMEM in krl_create_reorder_handle\n");
                    return UNSAFEMEM;
                }
            }
        } else {
            (*kdh)->transposed_bytes = 0;
            (*kdh)->transposed_codes = NULL;
        }
    } else {
        krl_clean_distance_handle(kdh);
        printf("Error: INVALPARAM in krl_create_reorder_handle\n");
        return INVALPARAM;
    }
    return SUCCESS;
}

/*
 * @brief Clean up and release the distance handle.
 * @param kdh Pointer to the distance handle to be cleaned.
 */
void krl_clean_distance_handle(KRLDistanceHandle **kdh)
{
    if (kdh == NULL) {
        return;
    }
    if (*kdh) {
        if ((*kdh)->transposed_codes && (*kdh)->transposed_bytes > 0) {
            free((*kdh)->transposed_codes);
            (*kdh)->transposed_codes = NULL;
        }
        if ((*kdh)->quanted_codes && (*kdh)->quanted_bytes > 0) {
            free((*kdh)->quanted_codes);
            (*kdh)->quanted_codes = NULL;
        }
        free(*kdh);
        (*kdh) = NULL;
    }
}

/*
 * @brief Create a lookup table (LUT) handle for 8-bit operations.
 * @param klh Pointer to the LUT handle to be created.
 * @param use_idx Whether to use index buffer (1 for yes, 0 for no).
 * @param capacity Capacity of the LUT.
 * @return SUCCESS if handle creation is successful, otherwise error code.
 */
int krl_create_LUT8b_handle(KRLLUT8bHandle **klh, int use_idx, size_t capacity)
{
    if (klh == NULL || (*klh) != NULL) {
        printf("Error: INVALPOINTER in krl_create_LUT8b_handle\n");
        return INVALPOINTER;
    }
    if (use_idx < 0 || use_idx > 1 || capacity < 1) {
        printf("Error: INVALPARAM in krl_create_LUT8b_handle\n");
        return INVALPARAM;
    }
    (*klh) = (KRLLUT8bHandle *)malloc(sizeof(KRLLUT8bHandle));
    if ((*klh) == NULL) {
        printf("Error: FAILALLOC in krl_create_LUT8b_handle\n");
        return FAILALLOC;
    }
    (*klh)->use_idx = use_idx;
    (*klh)->capacity = (capacity + 15) & (-16);
    if (use_idx == 1) {
        (*klh)->idx_buffer = (size_t *)aligned_alloc(KRL_DEFAULT_ALIGNED, capacity * sizeof(size_t));
        if ((*klh)->idx_buffer == NULL) {
            krl_clean_LUT8b_handle(klh);
            printf("Error: FAILALLOC in krl_create_LUT8b_handle\n");
            return FAILALLOC;
        }
    } else {
        (*klh)->idx_buffer = NULL;
    }
    (*klh)->distance_buffer = (float *)aligned_alloc(KRL_DEFAULT_ALIGNED, capacity * sizeof(float));
    if ((*klh)->distance_buffer == NULL) {
        krl_clean_LUT8b_handle(klh);
        printf("Error: FAILALLOC in krl_create_LUT8b_handle\n");
        return FAILALLOC;
    }
    return SUCCESS;
}

/*
 * @brief Clean up and release the LUT8b handle.
 * @param klh Pointer to the LUT8b handle to be cleaned.
 */
void krl_clean_LUT8b_handle(KRLLUT8bHandle **klh)
{
    if (klh == NULL) {
        return;
    }
    if ((*klh)->capacity > 0) {
        if ((*klh)->use_idx > 0 && (*klh)->idx_buffer) {
            free((*klh)->idx_buffer);
            (*klh)->idx_buffer = NULL;
        }
        if ((*klh)->distance_buffer) {
            free((*klh)->distance_buffer);
            (*klh)->distance_buffer = NULL;
        }
        free(*klh);
        (*klh) = NULL;
    }
}

/*
 * @brief Get the index buffer pointer from the LUT8b handle.
 * @param klh Pointer to the LUT8b handle.
 * @return Pointer to the index buffer.
 */
size_t *krl_get_idx_pointer(const KRLLUT8bHandle *klh)
{
    if (klh == NULL) {
        printf("Error: INVALPOINTER in krl_get_idx_pointer\n");
        return NULL;
    } else {
        return klh->idx_buffer;
    }
}

/*
 * @brief Get the distance buffer pointer from the LUT8b handle.
 * @param klh Pointer to the LUT8b handle.
 * @return Pointer to the distance buffer.
 */
float *krl_get_dist_pointer(const KRLLUT8bHandle *klh)
{
    if (klh == NULL) {
        printf("Error: INVALPOINTER in krl_get_dist_pointer\n");
        return NULL;
    } else {
        return klh->distance_buffer;
    }
}

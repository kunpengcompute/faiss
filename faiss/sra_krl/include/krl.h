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

#ifndef KRL_H
#define KRL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define KRL_API_PUBLIC __attribute__((visibility("default")))

#ifdef __aarch64__
typedef __fp16 float16_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief Handle for distance computation.
 */
typedef struct KRLBatchDistanceHandle KRLDistanceHandle;

/*
 * @brief Create a distance computation handle.
 * @param kdh Pointer to the distance handle.
 * @param accu_level Accuracy level, 1, 2, or 3.
 * @param blocksize Block size for computation, 16, 32, or 64.
 * @param codes_num Number of base vectors.
 * @param dim Dimension of vectors.
 * @param num_base Number of base vectors.
 * @param metric_type Distance measure type, 0 for inner product, 1 for L2.
 * @param codes Base vector data.
 * @param codes_size Length of codes.
 * @return int 0 on success, non-zero on failure.
 */
KRL_API_PUBLIC int krl_create_distance_handle(KRLDistanceHandle **kdh, size_t accu_level, size_t blocksize,
    size_t codes_num, size_t dim, size_t num_base, int metric_type, const uint8_t *codes, size_t codes_size);

/*
 * @brief Create a distance computation handle with additional accuracy levels.
 * @param kdh Pointer to the distance handle.
 * @param accu_level Accuracy level for initial computation, 1, 2, or 3.
 * @param full_accu_level Accuracy level for final computation, 1, 2, or 3.
 * @param codes_num Number of base vectors.
 * @param dim Dimension of vectors.
 * @param metric_type Distance measure type, 0 for inner product, 1 for L2.
 * @param codes Base vector data.
 * @param codes_size Length of codes.
 * @return int 0 on success, non-zero on failure.
 */
KRL_API_PUBLIC int krl_create_reorder_handle(KRLDistanceHandle **kdh, size_t accu_level, size_t full_accu_level,
    size_t codes_num, size_t dim, int metric_type, const uint8_t *codes, size_t codes_size);

/*
 * @brief Clean up and release the distance computation handle.
 * @param kdh Pointer to the distance handle.
 */
KRL_API_PUBLIC void krl_clean_distance_handle(KRLDistanceHandle **kdh);

/*
 * @brief Handle for 8-bit lookup table operations.
 */
typedef struct KRLLookupTable8bitHandle KRLLUT8bHandle;

/*
 * @brief Create an 8-bit lookup table handle.
 * @param klh Pointer to the lookup table handle.
 * @param use_idx Whether to use index buffer.
 * @param capacity Capacity of the lookup table.
 * @return int 0 on success, non-zero on failure.
 */
KRL_API_PUBLIC int krl_create_LUT8b_handle(KRLLUT8bHandle **klh, int use_idx, size_t capacity);

/*
 * @brief Clean up and release the 8-bit lookup table handle.
 * @param klh Pointer to the lookup table handle.
 */
KRL_API_PUBLIC void krl_clean_LUT8b_handle(KRLLUT8bHandle **klh);

/*
 * @brief Get the index pointer from the lookup table handle.
 * @param klh Pointer to the lookup table handle.
 * @return size_t* Pointer to the index buffer.
 */
KRL_API_PUBLIC size_t *krl_get_idx_pointer(const KRLLUT8bHandle *klh);

/*
 * @brief Get the distance pointer from the lookup table handle.
 * @param klh Pointer to the lookup table handle.
 * @return float* Pointer to the distance buffer.
 */
KRL_API_PUBLIC float *krl_get_dist_pointer(const KRLLUT8bHandle *klh);

/* -------------------------------------- 1 to 1 distance compute -------------------------------------- */

/*
 * @brief Compute L2 square distance between two vectors.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param d Dimension of vectors.
 * @param dis Stores the computed L2 square result (float).
 */
KRL_API_PUBLIC int krl_L2sqr(const float *x, const float *__restrict y, const size_t d, float *dis);

/*
 * @brief Compute L2 square distance between two 16-bit floating point vectors.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param d Dimension of vectors.
 * @param dis Stores the computed L2 square result (float).
 */
KRL_API_PUBLIC int krl_L2sqr_f16f32(
    const uint16_t *x, const uint16_t *__restrict y, size_t d, float *dis);

/*
 * @brief Compute L2 square distance between two 8-bit integer vectors.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param d Dimension of vectors.
 * @param dis Stores the computed L2 square result (uint32_t).
 */
KRL_API_PUBLIC int krl_L2sqr_u8u32(
    const uint8_t *x, const uint8_t *__restrict y, size_t d, uint32_t *dis);

/*
 * @brief Compute inner product distance between two vectors.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param d Dimension of vectors.
 * @param dis Stores the inner product result (float).
 */
KRL_API_PUBLIC int krl_ipdis(const float *x, const float *__restrict y, const size_t d, float *dis);

/*
 * @brief Compute negative inner product distance between two 16-bit floating point vectors.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param d Dimension of vectors.
 * @param dis Stores the inner product result (float).
 */
KRL_API_PUBLIC int krl_negative_ipdis_f16f32(
    const uint16_t *x, const uint16_t *__restrict y, const size_t d, float *dis);

/* -------------------------------------- Sparse distance calculation -------------------------------------- */

/*
 * @brief Compute L2 square distance between vectors using indices.
 * @param dis Output distance array.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param ids Indices of vectors.
 * @param d Dimension of vectors.
 * @param ny Number of vectors.
 */
KRL_API_PUBLIC int krl_L2sqr_by_idx(
    float *dis, const float *x, const float *y, const int64_t *ids, size_t d, size_t ny);

/*
 * @brief Compute L2 square distance between 16-bit floating point vectors using indices.
 * @param dis Output distance array.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param ids Indices of vectors.
 * @param d Dimension of vectors.
 * @param ny Number of vectors.
 */
KRL_API_PUBLIC int krl_L2sqr_by_idx_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, const int64_t *ids, size_t d, size_t ny);

/*
 * @brief Compute L2 square distance between 8-bit integer vectors using indices.
 * @param dis Output distance array.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param ids Indices of vectors.
 * @param d Dimension of vectors.
 * @param ny Number of vectors.
 */
KRL_API_PUBLIC int krl_L2sqr_by_idx_u8f32(
    float *dis, const uint8_t *x, const uint8_t *y, const int64_t *ids, size_t d, size_t ny);

/*
 * @brief Compute negative inner product distance between 16-bit floating point vectors using indices.
 * @param dis Output distance array.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param ids Indices of vectors.
 * @param d Dimension of vectors.
 * @param ny Number of vectors.
 */
KRL_API_PUBLIC int krl_negative_inner_product_by_idx_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, const int64_t *ids, size_t d, size_t ny);

/*
 * @brief Compute inner product distance between 8-bit integer vectors using indices.
 * @param dis Output distance array.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param ids Indices of vectors.
 * @param d Dimension of vectors.
 * @param ny Number of vectors.
 */
KRL_API_PUBLIC int krl_inner_product_by_idx_s8f32(
    float *dis, const int8_t *x, const int8_t *y, const int64_t *ids, size_t d, size_t ny);

/*
 * @brief Compute inner product distance between 16-bit floating point vectors using indices.
 * @param dis Output distance array.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param ids Indices of vectors.
 * @param d Dimension of vectors.
 * @param ny Number of vectors.
 */
KRL_API_PUBLIC int krl_inner_product_by_idx_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, const int64_t *ids, size_t d, size_t ny);

/*
 * @brief Compute inner product distance between vectors using indices.
 * @param dis Output distance array.
 * @param x Pointer to the first vector.
 * @param y Pointer to the second vector.
 * @param ids Indices of vectors.
 * @param d Dimension of vectors.
 * @param ny Number of vectors.
 */
KRL_API_PUBLIC int krl_inner_product_by_idx(
    float *dis, const float *x, const float *y, const int64_t *ids, size_t d, size_t ny);

/* -------------------------------------- dense distance calculation -------------------------------------- */

/*
 * @brief Compute L2 square distance between multiple vectors.
 * @param dis Output distance array.
 * @param x Pointer to the first set of vectors.
 * @param y Pointer to the second set of vectors.
 * @param ny Number of vectors.
 * @param d Dimension of vectors.
 */
KRL_API_PUBLIC int krl_L2sqr_ny(float *dis, const float *x, const float *y, size_t ny, size_t d);

/*
 * @brief Compute L2 square distance between multiple 16-bit floating point vectors.
 * @param dis Output distance array.
 * @param x Pointer to the first set of vectors.
 * @param y Pointer to the second set of vectors.
 * @param ny Number of vectors.
 * @param d Dimension of vectors.
 */
KRL_API_PUBLIC int krl_L2sqr_ny_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d);

/*
 * @brief Compute L2 square distance between multiple 8-bit integer vectors.
 * @param dis Output distance array.
 * @param x Pointer to the first set of vectors.
 * @param y Pointer to the second set of vectors.
 * @param ny Number of vectors.
 * @param d Dimension of vectors.
 */
KRL_API_PUBLIC int krl_L2sqr_ny_u8f32(
    float *dis, const uint8_t *x, const uint8_t *y, size_t ny, size_t d);

/*
 * @brief Compute L2 square distance using a distance handle.
 * @param kdh Pointer to the distance handle.
 * @param dis Output distance array.
 * @param x Pointer to the query vector.
 * @param dis_size Length of dis.
 * @param x_size Length of x.
 */
KRL_API_PUBLIC int krl_L2sqr_ny_with_handle(
    const KRLDistanceHandle *kdh, float *dis, const float *x, size_t dis_size, size_t x_size);

/*
 * @brief Compute inner product distance between multiple vectors.
 * @param dis Output distance array.
 * @param x Pointer to the first set of vectors.
 * @param y Pointer to the second set of vectors.
 * @param ny Number of vectors.
 * @param d Dimension of vectors.
 */
KRL_API_PUBLIC int krl_inner_product_ny(
    float *dis, const float *x, const float *y, size_t ny, size_t d);

/*
 * @brief Compute inner product distance between multiple 16-bit floating point vectors.
 * @param dis Output distance array.
 * @param x Pointer to the first set of vectors.
 * @param y Pointer to the second set of vectors.
 * @param ny Number of vectors.
 * @param d Dimension of vectors.
 */
KRL_API_PUBLIC int krl_inner_product_ny_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d);

/*
 * @brief Compute inner product distance between multiple 8-bit integer vectors.
 * @param dis Output distance array.
 * @param x Pointer to the first set of vectors.
 * @param y Pointer to the second set of vectors.
 * @param ny Number of vectors.
 * @param d Dimension of vectors.
 */
KRL_API_PUBLIC int krl_inner_product_ny_s8f32(
    float *dis, const int8_t *x, const int8_t *y, size_t ny, size_t d);

/*
 * @brief Compute inner product distance using a distance handle.
 * @param kdh Pointer to the distance handle.
 * @param dis Output distance array.
 * @param x Pointer to the query vector.
 * @param dis_size Length of dis.
 * @param x_size Length of x.
 */
KRL_API_PUBLIC int krl_inner_product_ny_with_handle(
    const KRLDistanceHandle *kdh, float *dis, const float *x, size_t dis_size, size_t x_size);

/* -------------------------------------- 8-bits table lookup -------------------------------------- */

/*
 * @brief Lookup table function for 8-bit codes.
 * @param nsq Number of subquantizers.
 * @param ncode Number of codes.
 * @param codes Input codes.
 * @param sim_table Similarity table.
 * @param dis Output distance array.
 * @param dis0 Initial distance value.
 * @param codes_size Length of codes.
 * @param sim_table_size Length of sim_table.
 */
KRL_API_PUBLIC int krl_table_lookup_8b_f32(size_t nsq, size_t ncode, const uint8_t *codes, const float *sim_table,
    float *dis, float dis0, size_t codes_size, size_t sim_table_size);

/*
 * @brief Lookup table function for 8-bit codes with indices.
 * @param nsq Number of subquantizers.
 * @param ncode Number of codes.
 * @param codes Input codes.
 * @param sim_table Similarity table.
 * @param dis Output distance array.
 * @param dis0 Initial distance value.
 * @param idx Indices of codes.
 * @param codes_size Length of codes.
 * @param sim_table_size Length of sim_table.
 */
KRL_API_PUBLIC int krl_table_lookup_8b_f32_by_idx(size_t nsq, size_t ncode, const uint8_t *codes,
    const float *sim_table, float *dis, float dis0, const size_t *idx, size_t codes_size, size_t sim_table_size);

/* -------------------------------------- 4-bits table lookup -------------------------------------- */

/*
 * @brief Fast table lookup function for 4-bit codes (batched).
 * @param nq Number of queries.
 * @param nsq Number of subquantizers.
 * @param codes Input codes.
 * @param LUT Precomputed distances.
 * @param dis Output distances.
 * @param threshold Filter threshold.
 * @param lt_mask Filter result mask.
 * @param keep_min Whether to keep minimum values.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param threshold_size Length of threshold.
 * @param lt_mask_size Length of lt_mask.
 */
KRL_API_PUBLIC int krl_fast_table_lookup_step(int nq, int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    const uint16_t *threshold, uint32_t *lt_mask, int keep_min, size_t codes_size, size_t LUT_size, size_t threshold_size,
    size_t lt_mask_size);

/*
 * @brief Perform fast L2 table lookup and filtering operations for single query with batch size 32.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=32).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 32.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 1.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param dis_size Length of dis.
 * @param lt_mask_size Length of lt_mask.
 */
KRL_API_PUBLIC int krl_L2_table_lookup_fast_scan_bs32(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

KRL_API_PUBLIC int krl_L2_table_lookup_fast_scan_bs32_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

/*
 * @brief Perform fast IP table lookup and filtering operations for single query with batch size 32.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=32).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 32.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 1.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param dis_size Length of dis.
 * @param lt_mask_size Length of lt_mask.
 */
KRL_API_PUBLIC int krl_IP_table_lookup_fast_scan_bs32(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

KRL_API_PUBLIC int krl_IP_table_lookup_fast_scan_bs32_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

/*
 * @brief Fast table lookup function for 4-bit codes (single query, batch size 64).
 * @param nsq Number of subquantizers.
 * @param codes Input codes.
 * @param LUT Precomputed distances.
 * @param dis Output distances.
 * @param threshold Filter threshold.
 * @param lt_mask Filter result mask.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
KRL_API_PUBLIC int krl_L2_table_lookup_fast_scan_bs64(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

KRL_API_PUBLIC int krl_L2_table_lookup_fast_scan_bs64_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

/*
 * @brief Fast table lookup function for 4-bit codes (single query, batch size 64, inner product).
 * @param nsq Number of subquantizers.
 * @param codes Input codes.
 * @param LUT Precomputed distances.
 * @param dis Output distances.
 * @param threshold Filter threshold.
 * @param lt_mask Filter result mask.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
KRL_API_PUBLIC int krl_IP_table_lookup_fast_scan_bs64(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

KRL_API_PUBLIC int krl_IP_table_lookup_fast_scan_bs64_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

/*
 * @brief Fast table lookup function for 4-bit codes (single query, batch size 96).
 * @param nsq Number of subquantizers.
 * @param codes Input codes.
 * @param LUT Precomputed distances.
 * @param dis Output distances.
 * @param threshold Filter threshold.
 * @param lt_mask Filter result mask.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
KRL_API_PUBLIC int krl_L2_table_lookup_fast_scan_bs96(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dise,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

KRL_API_PUBLIC int krl_L2_table_lookup_fast_scan_bs96_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

/*
 * @brief Fast table lookup function for 4-bit codes (single query, batch size 96, inner product).
 * @param nsq Number of subquantizers.
 * @param codes Input codes.
 * @param LUT Precomputed distances.
 * @param dis Output distances.
 * @param threshold Filter threshold.
 * @param lt_mask Filter result mask.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
KRL_API_PUBLIC int krl_IP_table_lookup_fast_scan_bs96(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

KRL_API_PUBLIC int krl_IP_table_lookup_fast_scan_bs96_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size);

/*
 * @brief Pack 4-bit codes into blocks.
 * @param codes Input codes.
 * @param ncode Total number of codes.
 * @param nsq Number of subquantizers.
 * @param blocks Output packed blocks.
 * @param batchsize Number of base vectors per batch.
 * @param dim_cross Whether to arrange dimensions in cross mode.
 * @param codes_size Length of codes.
 * @param blocks_size Length of blocks.
 */
KRL_API_PUBLIC int krl_pack_codes_4b(const uint8_t *codes, size_t ncode, size_t nsq, uint8_t *blocks, size_t batchsize,
    int dim_cross, size_t codes_size, size_t blocks_size);

/*
 * @brief Repack 4-bit codes with a different batch size.
 *
 * This function repacks 4-bit quantization codes from a previous
 * batch layout into a new batch layout. For IVFPQ (dim_cross == 0),
 * repacking is done directly. For PQFS, codes are first unpacked
 * into a temporary buffer and then packed again with the new
 * batch size.
 *
 * @param codes Pointer to the source packed codes.
 * @param ncode Total number of vectors.
 * @param nsq Number of subquantizers.
 * @param blocks Pointer to the destination packed blocks.
 * @param batchsize Number of blocks in the destination layout.
 * @param prev_batchsize Batch size used in the source layout.
 * @param after_batchsize Batch size to use in the destination layout.
 * @param dim_cross Dimension cross flag (0 for IVFPQ, non-zero for PQFS).
 */
KRL_API_PUBLIC int krl_repack_codes_4b(const uint8_t* codes, size_t ncode, size_t nsq, uint8_t* blocks, 
    size_t batchsize, size_t prev_batchsize, size_t after_batchsize, int dim_cross);

/* -------------------------------------- reorder function -------------------------------------- */

/*
 * @brief Reorder two vectors based on distance.
 * @param kdh Pointer to the distance handle.
 * @param base_k Number of base vectors obtained in the first phase.
 * @param base_dis Distance array from the first phase.
 * @param base_idx Index array from the first phase.
 * @param query_vector Query vector.
 * @param k Number of final output base vectors.
 * @param dis Final distance array.
 * @param idx Final index array.
 * @param query_vector_size Length of query_vector.
 */
KRL_API_PUBLIC int krl_reorder_2_vector(const KRLDistanceHandle *kdh, int64_t base_k, float *base_dis,
    int64_t *base_idx, const float *query_vector, int64_t k, float *dis, int64_t *idx, size_t query_vector_size);

/*
 * @brief Reorder two vectors with continuous indices.
 * @param kdh Pointer to the distance handle.
 * @param base_k Number of base vectors obtained in the first phase.
 * @param begin_id Starting index of base vectors.
 * @param query_vector Query vector.
 * @param k Number of final output base vectors.
 * @param dis Final distance array.
 * @param idx Final index array.
 * @param query_vector_size Length of query_vector.
 */
KRL_API_PUBLIC int krl_reorder_2_vector_continuous(const KRLDistanceHandle *kdh, int64_t base_k, int64_t begin_id,
    const float *query_vector, int64_t k, float *dis, int64_t *idx, size_t query_vector_size);

KRL_API_PUBLIC int krl_reorder_2_vector_continuous_f16(const KRLDistanceHandle *kdh, int64_t base_k, int64_t begin_id,
    const float16_t *query_vector, int64_t k, float *dis, int64_t *idx, size_t query_vector_size);

/* -------------------------------------- handle IO function -------------------------------------- */

/*
 * @brief Store the distance handle to a file.
 * @param f File pointer.
 * @param kdh Pointer to the distance handle.
 */
KRL_API_PUBLIC int krl_store_distanceHandle(FILE *f, const KRLDistanceHandle *kdh);

/*
 * @brief Build the distance handle from a file.
 * @param f File pointer.
 * @param kdh Pointer to the distance handle.
 * @return int 0 on success, non-zero on failure.
 */
KRL_API_PUBLIC int krl_build_distanceHandle_fromfile(FILE *f, KRLDistanceHandle **kdh);

#ifdef __cplusplus
}
#endif

#endif  // KRL_H
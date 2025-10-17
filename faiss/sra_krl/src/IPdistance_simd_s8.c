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

/*
 * @brief Compute the inner product between two int8_t vectors.
 * @param x Pointer to the first vector (int8_t).
 * @param y Pointer to the second vector (int8_t).
 * @param d Dimension of the vectors.
 * @return int32_t The computed inner product result.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
int32_t krl_inner_product_s8s32(const int8_t *x, const int8_t *__restrict y, const size_t d)
{
    size_t i;
    int32_t res;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 64;
    int32x4_t res1 = vdupq_n_s32(0);
    int32x4_t res2 = vdupq_n_s32(0);
    int32x4_t res3 = vdupq_n_s32(0);
    int32x4_t res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 64 */
    for (i = 0; i + double_round <= d; i += double_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t x8_1 = vld1q_s8(x + i + 16);
        const int8x16_t x8_2 = vld1q_s8(x + i + 32);
        const int8x16_t x8_3 = vld1q_s8(x + i + 48);

        const int8x16_t y8_0 = vld1q_s8(y + i);
        const int8x16_t y8_1 = vld1q_s8(y + i + 16);
        const int8x16_t y8_2 = vld1q_s8(y + i + 32);
        const int8x16_t y8_3 = vld1q_s8(y + i + 48);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res2 = vdotq_s32(res2, x8_1, y8_1);
        res3 = vdotq_s32(res3, x8_2, y8_2);
        res4 = vdotq_s32(res4, x8_3, y8_3);
    }

    /* Process remaining vectors in batches of 16 */
    for (; i + single_round <= d; i += single_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t y8_0 = vld1q_s8(y + i);
        res1 = vdotq_s32(res1, x8_0, y8_0);
    }

    /* Sum the results */
    res1 = vaddq_s32(res1, res2);
    res3 = vaddq_s32(res3, res4);
    res1 = vaddq_s32(res1, res3);
    res = vaddvq_s32(res1);

    /* Handle remaining elements */
    for (; i < d; i++) {
        res += (int32_t)(x[i]) * (int32_t)(y[i]);
    }

    return res;
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for two vectors with indices.
 * @param x Pointer to the query vector (int8_t).
 * @param y0 Pointer to the first database vector (int8_t).
 * @param y1 Pointer to the second database vector (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch2_s8s32(
    const int8_t *x, const int8_t *__restrict y0, const int8_t *__restrict y1, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    int32x4_t res1 = vdupq_n_s32(0);
    int32x4_t res2 = vdupq_n_s32(0);
    int32x4_t res3 = vdupq_n_s32(0);
    int32x4_t res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 32 */
    for (i = 0; i + double_round <= d; i += double_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t x8_1 = vld1q_s8(x + i + 16);

        const int8x16_t y8_0 = vld1q_s8(y0 + i);
        const int8x16_t y8_1 = vld1q_s8(y0 + i + 16);
        const int8x16_t y8_2 = vld1q_s8(y1 + i);
        const int8x16_t y8_3 = vld1q_s8(y1 + i + 16);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res2 = vdotq_s32(res2, x8_1, y8_1);
        res3 = vdotq_s32(res3, x8_0, y8_2);
        res4 = vdotq_s32(res4, x8_1, y8_3);
    }

    /* Process remaining vectors in batches of 16 */
    for (; i + single_round <= d; i += single_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t y8_0 = vld1q_s8(y0 + i);
        const int8x16_t y8_1 = vld1q_s8(y1 + i);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res3 = vdotq_s32(res3, x8_0, y8_1);
    }

    /* Sum the results */
    res1 = vaddq_s32(res1, res2);
    res3 = vaddq_s32(res3, res4);

    dis[0] = (float)vaddvq_s32(res1);
    dis[1] = (float)vaddvq_s32(res3);

    /* Handle remaining elements */
    for (; i < d; i++) {
        dis[0] += (float)(x[i]) * (float)(y0[i]);
        dis[1] += (float)(x[i]) * (float)(y1[i]);
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for four vectors with indices.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch4_s8s32(const int8_t *x, const int8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 16 */
    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        int8x16_t neon_base1 = vld1q_s8(y[0] + i);
        int8x16_t neon_base2 = vld1q_s8(y[1] + i);
        int8x16_t neon_base3 = vld1q_s8(y[2] + i);
        int8x16_t neon_base4 = vld1q_s8(y[3] + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);

    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (float)(x[i]) * (float)*(y[0] + i);
        float d1 = (float)(x[i]) * (float)*(y[1] + i);
        float d2 = (float)(x[i]) * (float)*(y[2] + i);
        float d3 = (float)(x[i]) * (float)*(y[3] + i);
        for (i++; i < d; ++i) {
            d0 += (float)(x[i]) * (float)*(y[0] + i);
            d1 += (float)(x[i]) * (float)*(y[1] + i);
            d2 += (float)(x[i]) * (float)*(y[2] + i);
            d3 += (float)(x[i]) * (float)*(y[3] + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for eight vectors with indices.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch8_s8s32(const int8_t *x, const int8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);

    /* Process vectors in batches of 16 */
    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        int8x16_t neon_base1 = vld1q_s8(y[0] + i);
        int8x16_t neon_base2 = vld1q_s8(y[1] + i);
        int8x16_t neon_base3 = vld1q_s8(y[2] + i);
        int8x16_t neon_base4 = vld1q_s8(y[3] + i);
        int8x16_t neon_base5 = vld1q_s8(y[4] + i);
        int8x16_t neon_base6 = vld1q_s8(y[5] + i);
        int8x16_t neon_base7 = vld1q_s8(y[6] + i);
        int8x16_t neon_base8 = vld1q_s8(y[7] + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
        neon_res5 = vdotq_s32(neon_res5, neon_query, neon_base5);
        neon_res6 = vdotq_s32(neon_res6, neon_query, neon_base6);
        neon_res7 = vdotq_s32(neon_res7, neon_query, neon_base7);
        neon_res8 = vdotq_s32(neon_res8, neon_query, neon_base8);
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res5 = vpaddq_s32(neon_res5, neon_res6);
    neon_res7 = vpaddq_s32(neon_res7, neon_res8);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);
    neon_res5 = vpaddq_s32(neon_res5, neon_res7);

    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
    vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (float)(x[i]) * (float)*(y[0] + i);
        float d1 = (float)(x[i]) * (float)*(y[1] + i);
        float d2 = (float)(x[i]) * (float)*(y[2] + i);
        float d3 = (float)(x[i]) * (float)*(y[3] + i);
        float d4 = (float)(x[i]) * (float)*(y[4] + i);
        float d5 = (float)(x[i]) * (float)*(y[5] + i);
        float d6 = (float)(x[i]) * (float)*(y[6] + i);
        float d7 = (float)(x[i]) * (float)*(y[7] + i);
        for (i++; i < d; ++i) {
            d0 += (float)(x[i]) * (float)*(y[0] + i);
            d1 += (float)(x[i]) * (float)*(y[1] + i);
            d2 += (float)(x[i]) * (float)*(y[2] + i);
            d3 += (float)(x[i]) * (float)*(y[3] + i);
            d4 += (float)(x[i]) * (float)*(y[4] + i);
            d5 += (float)(x[i]) * (float)*(y[5] + i);
            d6 += (float)(x[i]) * (float)*(y[6] + i);
            d7 += (float)(x[i]) * (float)*(y[7] + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for sixteen vectors with indices and prefetch optimization.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_prefetch_batch16_s8s32(
    const int8_t *x, const int8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;   /* 128 / 8 */
    constexpr size_t prefetch_round = 64; /* 4 * single_round */

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);
    int32x4_t neon_res9 = vdupq_n_s32(0);
    int32x4_t neon_res10 = vdupq_n_s32(0);
    int32x4_t neon_res11 = vdupq_n_s32(0);
    int32x4_t neon_res12 = vdupq_n_s32(0);
    int32x4_t neon_res13 = vdupq_n_s32(0);
    int32x4_t neon_res14 = vdupq_n_s32(0);
    int32x4_t neon_res15 = vdupq_n_s32(0);
    int32x4_t neon_res16 = vdupq_n_s32(0);

    /* Process vectors with prefetch optimization */
    if (d >= prefetch_round) {
        for (i = 0; i < d - prefetch_round; i += prefetch_round) {
            /* Prefetch data to improve cache utilization */
            prefetch_L1(x + i + prefetch_round);
            for (size_t idx = 0; idx < 16; ++idx) {
                prefetch_Lx(y[idx] + i + prefetch_round);
            }

            /* Process vectors in batches of 16 */
            for (size_t j = 0; j < prefetch_round; j += single_round) {
                int8x16_t neon_query = vld1q_s8(x + i + j);
                int8x16_t neon_base1 = vld1q_s8(y[0] + i + j);
                int8x16_t neon_base2 = vld1q_s8(y[1] + i + j);
                int8x16_t neon_base3 = vld1q_s8(y[2] + i + j);
                int8x16_t neon_base4 = vld1q_s8(y[3] + i + j);
                int8x16_t neon_base5 = vld1q_s8(y[4] + i + j);
                int8x16_t neon_base6 = vld1q_s8(y[5] + i + j);
                int8x16_t neon_base7 = vld1q_s8(y[6] + i + j);
                int8x16_t neon_base8 = vld1q_s8(y[7] + i + j);

                neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
                neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
                neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
                neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
                neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
                neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
                neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
                neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

                neon_base1 = vld1q_s8(y[8] + i + j);
                neon_base2 = vld1q_s8(y[9] + i + j);
                neon_base3 = vld1q_s8(y[10] + i + j);
                neon_base4 = vld1q_s8(y[11] + i + j);
                neon_base5 = vld1q_s8(y[12] + i + j);
                neon_base6 = vld1q_s8(y[13] + i + j);
                neon_base7 = vld1q_s8(y[14] + i + j);
                neon_base8 = vld1q_s8(y[15] + i + j);

                neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
                neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
                neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
                neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
                neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
                neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
                neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
                neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
            }
        }

        /* Process remaining vectors in batches of 16 */
        for (; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y[0] + i);
            int8x16_t neon_base2 = vld1q_s8(y[1] + i);
            int8x16_t neon_base3 = vld1q_s8(y[2] + i);
            int8x16_t neon_base4 = vld1q_s8(y[3] + i);
            int8x16_t neon_base5 = vld1q_s8(y[4] + i);
            int8x16_t neon_base6 = vld1q_s8(y[5] + i);
            int8x16_t neon_base7 = vld1q_s8(y[6] + i);
            int8x16_t neon_base8 = vld1q_s8(y[7] + i);

            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
            neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_s8(y[8] + i);
            neon_base2 = vld1q_s8(y[9] + i);
            neon_base3 = vld1q_s8(y[10] + i);
            neon_base4 = vld1q_s8(y[11] + i);
            neon_base5 = vld1q_s8(y[12] + i);
            neon_base6 = vld1q_s8(y[13] + i);
            neon_base7 = vld1q_s8(y[14] + i);
            neon_base8 = vld1q_s8(y[15] + i);

            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
            neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
        }
    } else {
        /* Process vectors without prefetch optimization */
        for (i = 0; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y[0] + i);
            int8x16_t neon_base2 = vld1q_s8(y[1] + i);
            int8x16_t neon_base3 = vld1q_s8(y[2] + i);
            int8x16_t neon_base4 = vld1q_s8(y[3] + i);
            int8x16_t neon_base5 = vld1q_s8(y[4] + i);
            int8x16_t neon_base6 = vld1q_s8(y[5] + i);
            int8x16_t neon_base7 = vld1q_s8(y[6] + i);
            int8x16_t neon_base8 = vld1q_s8(y[7] + i);

            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
            neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_s8(y[8] + i);
            neon_base2 = vld1q_s8(y[9] + i);
            neon_base3 = vld1q_s8(y[10] + i);
            neon_base4 = vld1q_s8(y[11] + i);
            neon_base5 = vld1q_s8(y[12] + i);
            neon_base6 = vld1q_s8(y[13] + i);
            neon_base7 = vld1q_s8(y[14] + i);
            neon_base8 = vld1q_s8(y[15] + i);

            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
            neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
        }
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res5 = vpaddq_s32(neon_res5, neon_res6);
    neon_res7 = vpaddq_s32(neon_res7, neon_res8);
    neon_res9 = vpaddq_s32(neon_res9, neon_res10);
    neon_res11 = vpaddq_s32(neon_res11, neon_res12);
    neon_res13 = vpaddq_s32(neon_res13, neon_res14);
    neon_res15 = vpaddq_s32(neon_res15, neon_res16);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);
    neon_res5 = vpaddq_s32(neon_res5, neon_res7);
    neon_res9 = vpaddq_s32(neon_res9, neon_res11);
    neon_res13 = vpaddq_s32(neon_res13, neon_res15);

    /* Store the results */
    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
    vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));
    vst1q_f32(dis + 8, vcvtq_f32_s32(neon_res9));
    vst1q_f32(dis + 12, vcvtq_f32_s32(neon_res13));

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (float)(x[i]) * (float)*(y[0] + i);
        float d1 = (float)(x[i]) * (float)*(y[1] + i);
        float d2 = (float)(x[i]) * (float)*(y[2] + i);
        float d3 = (float)(x[i]) * (float)*(y[3] + i);
        float d4 = (float)(x[i]) * (float)*(y[4] + i);
        float d5 = (float)(x[i]) * (float)*(y[5] + i);
        float d6 = (float)(x[i]) * (float)*(y[6] + i);
        float d7 = (float)(x[i]) * (float)*(y[7] + i);
        float d8 = (float)(x[i]) * (float)*(y[8] + i);
        float d9 = (float)(x[i]) * (float)*(y[9] + i);
        float d10 = (float)(x[i]) * (float)*(y[10] + i);
        float d11 = (float)(x[i]) * (float)*(y[11] + i);
        float d12 = (float)(x[i]) * (float)*(y[12] + i);
        float d13 = (float)(x[i]) * (float)*(y[13] + i);
        float d14 = (float)(x[i]) * (float)*(y[14] + i);
        float d15 = (float)(x[i]) * (float)*(y[15] + i);

        for (i++; i < d; ++i) {
            d0 += (float)(x[i]) * (float)*(y[0] + i);
            d1 += (float)(x[i]) * (float)*(y[1] + i);
            d2 += (float)(x[i]) * (float)*(y[2] + i);
            d3 += (float)(x[i]) * (float)*(y[3] + i);
            d4 += (float)(x[i]) * (float)*(y[4] + i);
            d5 += (float)(x[i]) * (float)*(y[5] + i);
            d6 += (float)(x[i]) * (float)*(y[6] + i);
            d7 += (float)(x[i]) * (float)*(y[7] + i);
            d8 += (float)(x[i]) * (float)*(y[8] + i);
            d9 += (float)(x[i]) * (float)*(y[9] + i);
            d10 += (float)(x[i]) * (float)*(y[10] + i);
            d11 += (float)(x[i]) * (float)*(y[11] + i);
            d12 += (float)(x[i]) * (float)*(y[12] + i);
            d13 += (float)(x[i]) * (float)*(y[13] + i);
            d14 += (float)(x[i]) * (float)*(y[14] + i);
            d15 += (float)(x[i]) * (float)*(y[15] + i);
        }

        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for twenty-four vectors with indices and prefetch optimization.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_prefetch_batch24_s8s32(
    const int8_t *x, const int8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;   /* 128 / 8 */
    constexpr size_t prefetch_round = 64; /* 4 * single_round */

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);
    int32x4_t neon_res9 = vdupq_n_s32(0);
    int32x4_t neon_res10 = vdupq_n_s32(0);
    int32x4_t neon_res11 = vdupq_n_s32(0);
    int32x4_t neon_res12 = vdupq_n_s32(0);
    int32x4_t neon_res13 = vdupq_n_s32(0);
    int32x4_t neon_res14 = vdupq_n_s32(0);
    int32x4_t neon_res15 = vdupq_n_s32(0);
    int32x4_t neon_res16 = vdupq_n_s32(0);
    int32x4_t neon_res17 = vdupq_n_s32(0);
    int32x4_t neon_res18 = vdupq_n_s32(0);
    int32x4_t neon_res19 = vdupq_n_s32(0);
    int32x4_t neon_res20 = vdupq_n_s32(0);
    int32x4_t neon_res21 = vdupq_n_s32(0);
    int32x4_t neon_res22 = vdupq_n_s32(0);
    int32x4_t neon_res23 = vdupq_n_s32(0);
    int32x4_t neon_res24 = vdupq_n_s32(0);

    if (d >= prefetch_round) {
        for (i = 0; i < d - prefetch_round; i += prefetch_round) {
            prefetch_L1(x + i + prefetch_round);
            prefetch_Lx(y[0] + i + prefetch_round);
            prefetch_Lx(y[1] + i + prefetch_round);
            prefetch_Lx(y[2] + i + prefetch_round);
            prefetch_Lx(y[3] + i + prefetch_round);
            prefetch_Lx(y[4] + i + prefetch_round);
            prefetch_Lx(y[5] + i + prefetch_round);
            prefetch_Lx(y[6] + i + prefetch_round);
            prefetch_Lx(y[7] + i + prefetch_round);
            prefetch_Lx(y[8] + i + prefetch_round);
            prefetch_Lx(y[9] + i + prefetch_round);
            prefetch_Lx(y[10] + i + prefetch_round);
            prefetch_Lx(y[11] + i + prefetch_round);
            prefetch_Lx(y[12] + i + prefetch_round);
            prefetch_Lx(y[13] + i + prefetch_round);
            prefetch_Lx(y[14] + i + prefetch_round);
            prefetch_Lx(y[15] + i + prefetch_round);
            prefetch_Lx(y[16] + i + prefetch_round);
            prefetch_Lx(y[17] + i + prefetch_round);
            prefetch_Lx(y[18] + i + prefetch_round);
            prefetch_Lx(y[19] + i + prefetch_round);
            prefetch_Lx(y[20] + i + prefetch_round);
            prefetch_Lx(y[21] + i + prefetch_round);
            prefetch_Lx(y[22] + i + prefetch_round);
            prefetch_Lx(y[23] + i + prefetch_round);
            for (size_t j = 0; j < prefetch_round; j += single_round) {
                int8x16_t neon_query = vld1q_s8(x + i + j);
                int8x16_t neon_base1 = vld1q_s8(y[0] + i + j);
                int8x16_t neon_base2 = vld1q_s8(y[1] + i + j);
                int8x16_t neon_base3 = vld1q_s8(y[2] + i + j);
                int8x16_t neon_base4 = vld1q_s8(y[3] + i + j);
                neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
                neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
                neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
                neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);

                neon_base1 = vld1q_s8(y[4] + i + j);
                neon_base2 = vld1q_s8(y[5] + i + j);
                neon_base3 = vld1q_s8(y[6] + i + j);
                neon_base4 = vld1q_s8(y[7] + i + j);
                neon_res5 = vdotq_s32(neon_res5, neon_base1, neon_query);
                neon_res6 = vdotq_s32(neon_res6, neon_base2, neon_query);
                neon_res7 = vdotq_s32(neon_res7, neon_base3, neon_query);
                neon_res8 = vdotq_s32(neon_res8, neon_base4, neon_query);

                neon_base1 = vld1q_s8(y[8] + i + j);
                neon_base2 = vld1q_s8(y[9] + i + j);
                neon_base3 = vld1q_s8(y[10] + i + j);
                neon_base4 = vld1q_s8(y[11] + i + j);
                neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
                neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
                neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
                neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);

                neon_base1 = vld1q_s8(y[12] + i + j);
                neon_base2 = vld1q_s8(y[13] + i + j);
                neon_base3 = vld1q_s8(y[14] + i + j);
                neon_base4 = vld1q_s8(y[15] + i + j);
                neon_res13 = vdotq_s32(neon_res13, neon_base1, neon_query);
                neon_res14 = vdotq_s32(neon_res14, neon_base2, neon_query);
                neon_res15 = vdotq_s32(neon_res15, neon_base3, neon_query);
                neon_res16 = vdotq_s32(neon_res16, neon_base4, neon_query);

                neon_base1 = vld1q_s8(y[16] + i + j);
                neon_base2 = vld1q_s8(y[17] + i + j);
                neon_base3 = vld1q_s8(y[18] + i + j);
                neon_base4 = vld1q_s8(y[19] + i + j);
                neon_res17 = vdotq_s32(neon_res17, neon_base1, neon_query);
                neon_res18 = vdotq_s32(neon_res18, neon_base2, neon_query);
                neon_res19 = vdotq_s32(neon_res19, neon_base3, neon_query);
                neon_res20 = vdotq_s32(neon_res20, neon_base4, neon_query);

                neon_base1 = vld1q_s8(y[20] + i + j);
                neon_base2 = vld1q_s8(y[21] + i + j);
                neon_base3 = vld1q_s8(y[22] + i + j);
                neon_base4 = vld1q_s8(y[23] + i + j);
                neon_res21 = vdotq_s32(neon_res21, neon_base1, neon_query);
                neon_res22 = vdotq_s32(neon_res22, neon_base2, neon_query);
                neon_res23 = vdotq_s32(neon_res23, neon_base3, neon_query);
                neon_res24 = vdotq_s32(neon_res24, neon_base4, neon_query);
            }
        }
        for (; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y[0] + i);
            int8x16_t neon_base2 = vld1q_s8(y[1] + i);
            int8x16_t neon_base3 = vld1q_s8(y[2] + i);
            int8x16_t neon_base4 = vld1q_s8(y[3] + i);
            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[4] + i);
            neon_base2 = vld1q_s8(y[5] + i);
            neon_base3 = vld1q_s8(y[6] + i);
            neon_base4 = vld1q_s8(y[7] + i);
            neon_res5 = vdotq_s32(neon_res5, neon_base1, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base2, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base3, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[8] + i);
            neon_base2 = vld1q_s8(y[9] + i);
            neon_base3 = vld1q_s8(y[10] + i);
            neon_base4 = vld1q_s8(y[11] + i);
            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[12] + i);
            neon_base2 = vld1q_s8(y[13] + i);
            neon_base3 = vld1q_s8(y[14] + i);
            neon_base4 = vld1q_s8(y[15] + i);
            neon_res13 = vdotq_s32(neon_res13, neon_base1, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base2, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base3, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[16] + i);
            neon_base2 = vld1q_s8(y[17] + i);
            neon_base3 = vld1q_s8(y[18] + i);
            neon_base4 = vld1q_s8(y[19] + i);
            neon_res17 = vdotq_s32(neon_res17, neon_base1, neon_query);
            neon_res18 = vdotq_s32(neon_res18, neon_base2, neon_query);
            neon_res19 = vdotq_s32(neon_res19, neon_base3, neon_query);
            neon_res20 = vdotq_s32(neon_res20, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[20] + i);
            neon_base2 = vld1q_s8(y[21] + i);
            neon_base3 = vld1q_s8(y[22] + i);
            neon_base4 = vld1q_s8(y[23] + i);
            neon_res21 = vdotq_s32(neon_res21, neon_base1, neon_query);
            neon_res22 = vdotq_s32(neon_res22, neon_base2, neon_query);
            neon_res23 = vdotq_s32(neon_res23, neon_base3, neon_query);
            neon_res24 = vdotq_s32(neon_res24, neon_base4, neon_query);
        }
        neon_res1 = vpaddq_s32(neon_res1, neon_res2);
        neon_res3 = vpaddq_s32(neon_res3, neon_res4);
        neon_res5 = vpaddq_s32(neon_res5, neon_res6);
        neon_res7 = vpaddq_s32(neon_res7, neon_res8);
        neon_res9 = vpaddq_s32(neon_res9, neon_res10);
        neon_res11 = vpaddq_s32(neon_res11, neon_res12);
        neon_res13 = vpaddq_s32(neon_res13, neon_res14);
        neon_res15 = vpaddq_s32(neon_res15, neon_res16);
        neon_res17 = vpaddq_s32(neon_res17, neon_res18);
        neon_res19 = vpaddq_s32(neon_res19, neon_res20);
        neon_res21 = vpaddq_s32(neon_res21, neon_res22);
        neon_res23 = vpaddq_s32(neon_res23, neon_res24);
        neon_res1 = vpaddq_s32(neon_res1, neon_res3);
        neon_res5 = vpaddq_s32(neon_res5, neon_res7);
        neon_res9 = vpaddq_s32(neon_res9, neon_res11);
        neon_res13 = vpaddq_s32(neon_res13, neon_res15);
        neon_res17 = vpaddq_s32(neon_res17, neon_res19);
        neon_res21 = vpaddq_s32(neon_res21, neon_res23);

        vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_s32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_s32(neon_res13));
        vst1q_f32(dis + 16, vcvtq_f32_s32(neon_res17));
        vst1q_f32(dis + 20, vcvtq_f32_s32(neon_res21));
    } else {
        for (i = 0; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y[0] + i);
            int8x16_t neon_base2 = vld1q_s8(y[1] + i);
            int8x16_t neon_base3 = vld1q_s8(y[2] + i);
            int8x16_t neon_base4 = vld1q_s8(y[3] + i);
            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[4] + i);
            neon_base2 = vld1q_s8(y[5] + i);
            neon_base3 = vld1q_s8(y[6] + i);
            neon_base4 = vld1q_s8(y[7] + i);
            neon_res5 = vdotq_s32(neon_res5, neon_base1, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base2, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base3, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[8] + i);
            neon_base2 = vld1q_s8(y[9] + i);
            neon_base3 = vld1q_s8(y[10] + i);
            neon_base4 = vld1q_s8(y[11] + i);
            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[12] + i);
            neon_base2 = vld1q_s8(y[13] + i);
            neon_base3 = vld1q_s8(y[14] + i);
            neon_base4 = vld1q_s8(y[15] + i);
            neon_res13 = vdotq_s32(neon_res13, neon_base1, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base2, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base3, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[16] + i);
            neon_base2 = vld1q_s8(y[17] + i);
            neon_base3 = vld1q_s8(y[18] + i);
            neon_base4 = vld1q_s8(y[19] + i);
            neon_res17 = vdotq_s32(neon_res17, neon_base1, neon_query);
            neon_res18 = vdotq_s32(neon_res18, neon_base2, neon_query);
            neon_res19 = vdotq_s32(neon_res19, neon_base3, neon_query);
            neon_res20 = vdotq_s32(neon_res20, neon_base4, neon_query);

            neon_base1 = vld1q_s8(y[20] + i);
            neon_base2 = vld1q_s8(y[21] + i);
            neon_base3 = vld1q_s8(y[22] + i);
            neon_base4 = vld1q_s8(y[23] + i);
            neon_res21 = vdotq_s32(neon_res21, neon_base1, neon_query);
            neon_res22 = vdotq_s32(neon_res22, neon_base2, neon_query);
            neon_res23 = vdotq_s32(neon_res23, neon_base3, neon_query);
            neon_res24 = vdotq_s32(neon_res24, neon_base4, neon_query);
        }
        neon_res1 = vpaddq_s32(neon_res1, neon_res2);
        neon_res3 = vpaddq_s32(neon_res3, neon_res4);
        neon_res5 = vpaddq_s32(neon_res5, neon_res6);
        neon_res7 = vpaddq_s32(neon_res7, neon_res8);
        neon_res9 = vpaddq_s32(neon_res9, neon_res10);
        neon_res11 = vpaddq_s32(neon_res11, neon_res12);
        neon_res13 = vpaddq_s32(neon_res13, neon_res14);
        neon_res15 = vpaddq_s32(neon_res15, neon_res16);
        neon_res17 = vpaddq_s32(neon_res17, neon_res18);
        neon_res19 = vpaddq_s32(neon_res19, neon_res20);
        neon_res21 = vpaddq_s32(neon_res21, neon_res22);
        neon_res23 = vpaddq_s32(neon_res23, neon_res24);
        neon_res1 = vpaddq_s32(neon_res1, neon_res3);
        neon_res5 = vpaddq_s32(neon_res5, neon_res7);
        neon_res9 = vpaddq_s32(neon_res9, neon_res11);
        neon_res13 = vpaddq_s32(neon_res13, neon_res15);
        neon_res17 = vpaddq_s32(neon_res17, neon_res19);
        neon_res21 = vpaddq_s32(neon_res21, neon_res23);

        vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_s32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_s32(neon_res13));
        vst1q_f32(dis + 16, vcvtq_f32_s32(neon_res17));
        vst1q_f32(dis + 20, vcvtq_f32_s32(neon_res21));
    }
    if (i < d) {
        float d0 = (float)(x[i]) * (float)*(y[0] + i);
        float d1 = (float)(x[i]) * (float)*(y[1] + i);
        float d2 = (float)(x[i]) * (float)*(y[2] + i);
        float d3 = (float)(x[i]) * (float)*(y[3] + i);
        float d4 = (float)(x[i]) * (float)*(y[4] + i);
        float d5 = (float)(x[i]) * (float)*(y[5] + i);
        float d6 = (float)(x[i]) * (float)*(y[6] + i);
        float d7 = (float)(x[i]) * (float)*(y[7] + i);
        float d8 = (float)(x[i]) * (float)*(y[8] + i);
        float d9 = (float)(x[i]) * (float)*(y[9] + i);
        float d10 = (float)(x[i]) * (float)*(y[10] + i);
        float d11 = (float)(x[i]) * (float)*(y[11] + i);
        float d12 = (float)(x[i]) * (float)*(y[12] + i);
        float d13 = (float)(x[i]) * (float)*(y[13] + i);
        float d14 = (float)(x[i]) * (float)*(y[14] + i);
        float d15 = (float)(x[i]) * (float)*(y[15] + i);
        float d16 = (float)(x[i]) * (float)*(y[16] + i);
        float d17 = (float)(x[i]) * (float)*(y[17] + i);
        float d18 = (float)(x[i]) * (float)*(y[18] + i);
        float d19 = (float)(x[i]) * (float)*(y[19] + i);
        float d20 = (float)(x[i]) * (float)*(y[20] + i);
        float d21 = (float)(x[i]) * (float)*(y[21] + i);
        float d22 = (float)(x[i]) * (float)*(y[22] + i);
        float d23 = (float)(x[i]) * (float)*(y[23] + i);
        for (i++; i < d; ++i) {
            d0 += (float)(x[i]) * (float)*(y[0] + i);
            d1 += (float)(x[i]) * (float)*(y[1] + i);
            d2 += (float)(x[i]) * (float)*(y[2] + i);
            d3 += (float)(x[i]) * (float)*(y[3] + i);
            d4 += (float)(x[i]) * (float)*(y[4] + i);
            d5 += (float)(x[i]) * (float)*(y[5] + i);
            d6 += (float)(x[i]) * (float)*(y[6] + i);
            d7 += (float)(x[i]) * (float)*(y[7] + i);
            d8 += (float)(x[i]) * (float)*(y[8] + i);
            d9 += (float)(x[i]) * (float)*(y[9] + i);
            d10 += (float)(x[i]) * (float)*(y[10] + i);
            d11 += (float)(x[i]) * (float)*(y[11] + i);
            d12 += (float)(x[i]) * (float)*(y[12] + i);
            d13 += (float)(x[i]) * (float)*(y[13] + i);
            d14 += (float)(x[i]) * (float)*(y[14] + i);
            d15 += (float)(x[i]) * (float)*(y[15] + i);
            d16 += (float)(x[i]) * (float)*(y[16] + i);
            d17 += (float)(x[i]) * (float)*(y[17] + i);
            d18 += (float)(x[i]) * (float)*(y[18] + i);
            d19 += (float)(x[i]) * (float)*(y[19] + i);
            d20 += (float)(x[i]) * (float)*(y[20] + i);
            d21 += (float)(x[i]) * (float)*(y[21] + i);
            d22 += (float)(x[i]) * (float)*(y[22] + i);
            d23 += (float)(x[i]) * (float)*(y[23] + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
        dis[16] += d16;
        dis[17] += d17;
        dis[18] += d18;
        dis[19] += d19;
        dis[20] += d20;
        dis[21] += d21;
        dis[22] += d22;
        dis[23] += d23;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for two vectors with 8-bit integer precision and 32-bit integer results.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (int32_t).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch2_s8s32(const int8_t *x, const int8_t *__restrict y, const size_t d, int32_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    int32x4_t res1 = vdupq_n_s32(0);
    int32x4_t res2 = vdupq_n_s32(0);
    int32x4_t res3 = vdupq_n_s32(0);
    int32x4_t res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 32 */
    for (i = 0; i + double_round <= d; i += double_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t x8_1 = vld1q_s8(x + i + 16);

        const int8x16_t y8_0 = vld1q_s8(y + i);
        const int8x16_t y8_1 = vld1q_s8(y + i + 16);
        const int8x16_t y8_2 = vld1q_s8(y + d + i);
        const int8x16_t y8_3 = vld1q_s8(y + d + i + 16);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res2 = vdotq_s32(res2, x8_1, y8_1);
        res3 = vdotq_s32(res3, x8_0, y8_2);
        res4 = vdotq_s32(res4, x8_1, y8_3);
    }

    /* Process remaining vectors in batches of 16 */
    for (; i + single_round <= d; i += single_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t y8_0 = vld1q_s8(y + i);
        const int8x16_t y8_1 = vld1q_s8(y + d + i);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res3 = vdotq_s32(res3, x8_0, y8_1);
    }

    /* Sum the results */
    res1 = vaddq_s32(res1, res2);
    res3 = vaddq_s32(res3, res4);

    /* Store the results */
    dis[0] = (int32_t)vaddvq_s32(res1);
    dis[1] = (int32_t)vaddvq_s32(res3);

    /* Handle remaining elements */
    if (i < d) {
        dis[0] += (int32_t)(x[i]) * (int32_t)(y[i]);
        dis[1] += (int32_t)(x[i]) * (int32_t)(y[d + i]);
        for (i++; i < d; ++i) {
            dis[0] += (int32_t)(x[i]) * (int32_t)(y[i]);
            dis[1] += (int32_t)(x[i]) * (int32_t)(y[d + i]);
        }
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for two vectors with 8-bit integer precision and 32-bit floating-point results.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch2_s8f32(const int8_t *x, const int8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    int32x4_t res1 = vdupq_n_s32(0);
    int32x4_t res2 = vdupq_n_s32(0);
    int32x4_t res3 = vdupq_n_s32(0);
    int32x4_t res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 32 */
    for (i = 0; i + double_round <= d; i += double_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t x8_1 = vld1q_s8(x + i + 16);

        const int8x16_t y8_0 = vld1q_s8(y + i);
        const int8x16_t y8_1 = vld1q_s8(y + i + 16);
        const int8x16_t y8_2 = vld1q_s8(y + d + i);
        const int8x16_t y8_3 = vld1q_s8(y + d + i + 16);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res2 = vdotq_s32(res2, x8_1, y8_1);
        res3 = vdotq_s32(res3, x8_0, y8_2);
        res4 = vdotq_s32(res4, x8_1, y8_3);
    }

    /* Process remaining vectors in batches of 16 */
    for (; i + single_round <= d; i += single_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t y8_0 = vld1q_s8(y + i);
        const int8x16_t y8_1 = vld1q_s8(y + d + i);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res3 = vdotq_s32(res3, x8_0, y8_1);
    }

    /* Sum the results */
    res1 = vaddq_s32(res1, res2);
    res3 = vaddq_s32(res3, res4);

    /* Store the results */
    dis[0] = (float)vaddvq_s32(res1);
    dis[1] = (float)vaddvq_s32(res3);

    /* Handle remaining elements */
    if (i < d) {
        dis[0] += (int32_t)(x[i]) * (int32_t)(y[i]);
        dis[1] += (int32_t)(x[i]) * (int32_t)(y[d + i]);
        for (i++; i < d; ++i) {
            dis[0] += (int32_t)(x[i]) * (int32_t)(y[i]);
            dis[1] += (int32_t)(x[i]) * (int32_t)(y[d + i]);
        }
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for four vectors with 8-bit integer precision and 32-bit integer results.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (int32_t).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch4_s8s32(const int8_t *x, const int8_t *__restrict y, const size_t d, int32_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 16 */
    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        int8x16_t neon_base1 = vld1q_s8(y + i);
        int8x16_t neon_base2 = vld1q_s8(y + d + i);
        int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
        int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);

    /* Store the results */
    vst1q_s32(dis, neon_res1);

    /* Handle remaining elements */
    if (i < d) {
        int32_t d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        int32_t d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        int32_t d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        int32_t d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for four vectors with 8-bit integer precision and 32-bit floating-point results.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch4_s8f32(const int8_t *x, const int8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 16 */
    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        int8x16_t neon_base1 = vld1q_s8(y + i);
        int8x16_t neon_base2 = vld1q_s8(y + d + i);
        int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
        int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);

    /* Store the results */
    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        float d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        float d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        float d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for eight vectors with 8-bit integer precision and 32-bit integer results.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (int32_t).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch8_s8s32(const int8_t *x, const int8_t *__restrict y, const size_t d, int32_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);

    /* Process vectors in batches of 16 */
    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        int8x16_t neon_base1 = vld1q_s8(y + i);
        int8x16_t neon_base2 = vld1q_s8(y + d + i);
        int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
        int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
        int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
        int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
        int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
        int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
        neon_res5 = vdotq_s32(neon_res5, neon_query, neon_base5);
        neon_res6 = vdotq_s32(neon_res6, neon_query, neon_base6);
        neon_res7 = vdotq_s32(neon_res7, neon_query, neon_base7);
        neon_res8 = vdotq_s32(neon_res8, neon_query, neon_base8);
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res5 = vpaddq_s32(neon_res5, neon_res6);
    neon_res7 = vpaddq_s32(neon_res7, neon_res8);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);
    neon_res5 = vpaddq_s32(neon_res5, neon_res7);

    /* Store the results */
    vst1q_s32(dis, neon_res1);
    vst1q_s32(dis + 4, neon_res5);

    /* Handle remaining elements */
    if (i < d) {
        int32_t d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        int32_t d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        int32_t d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        int32_t d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        int32_t d4 = (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
        int32_t d5 = (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
        int32_t d6 = (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
        int32_t d7 = (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
            d4 += (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
            d5 += (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
            d6 += (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
            d7 += (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for eight vectors with 8-bit integer precision and 32-bit floating-point results.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch8_s8f32(const int8_t *x, const int8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);

    /* Process vectors in batches of 16 */
    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        int8x16_t neon_base1 = vld1q_s8(y + i);
        int8x16_t neon_base2 = vld1q_s8(y + d + i);
        int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
        int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
        int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
        int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
        int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
        int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
        neon_res5 = vdotq_s32(neon_res5, neon_query, neon_base5);
        neon_res6 = vdotq_s32(neon_res6, neon_query, neon_base6);
        neon_res7 = vdotq_s32(neon_res7, neon_query, neon_base7);
        neon_res8 = vdotq_s32(neon_res8, neon_query, neon_base8);
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res5 = vpaddq_s32(neon_res5, neon_res6);
    neon_res7 = vpaddq_s32(neon_res7, neon_res8);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);
    neon_res5 = vpaddq_s32(neon_res5, neon_res7);

    /* Store the results */
    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
    vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        float d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        float d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        float d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        float d4 = (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
        float d5 = (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
        float d6 = (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
        float d7 = (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
            d4 += (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
            d5 += (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
            d6 += (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
            d7 += (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for 16 vectors with 8-bit integer precision and 32-bit integer results.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (int32_t).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_prefetch_batch16_s8s32(
    const int8_t *x, const int8_t *__restrict y, const size_t d, int32_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16;   /* 128 / 8 */
    constexpr size_t prefetch_round = 64; /* 4 * single_round */

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);
    int32x4_t neon_res9 = vdupq_n_s32(0);
    int32x4_t neon_res10 = vdupq_n_s32(0);
    int32x4_t neon_res11 = vdupq_n_s32(0);
    int32x4_t neon_res12 = vdupq_n_s32(0);
    int32x4_t neon_res13 = vdupq_n_s32(0);
    int32x4_t neon_res14 = vdupq_n_s32(0);
    int32x4_t neon_res15 = vdupq_n_s32(0);
    int32x4_t neon_res16 = vdupq_n_s32(0);

    /* Process vectors with prefetching */
    if (d >= prefetch_round) {
        for (i = 0; i < d - prefetch_round; i += prefetch_round) {
            /* Prefetch data to L1 cache */
            prefetch_L1(x + i + prefetch_round);
            prefetch_Lx(y + i + prefetch_round);
            prefetch_Lx(y + d + i + prefetch_round);
            prefetch_Lx(y + 2 * d + i + prefetch_round);
            prefetch_Lx(y + 3 * d + i + prefetch_round);
            prefetch_Lx(y + 4 * d + i + prefetch_round);
            prefetch_Lx(y + 5 * d + i + prefetch_round);
            prefetch_Lx(y + 6 * d + i + prefetch_round);
            prefetch_Lx(y + 7 * d + i + prefetch_round);
            prefetch_Lx(y + 8 * d + i + prefetch_round);
            prefetch_Lx(y + 9 * d + i + prefetch_round);
            prefetch_Lx(y + 10 * d + i + prefetch_round);
            prefetch_Lx(y + 11 * d + i + prefetch_round);
            prefetch_Lx(y + 12 * d + i + prefetch_round);
            prefetch_Lx(y + 13 * d + i + prefetch_round);
            prefetch_Lx(y + 14 * d + i + prefetch_round);
            prefetch_Lx(y + 15 * d + i + prefetch_round);

            /* Process data in single_round chunks */
            for (size_t j = 0; j < prefetch_round; j += single_round) {
                int8x16_t neon_query = vld1q_s8(x + i + j);
                int8x16_t neon_base1 = vld1q_s8(y + i + j);
                int8x16_t neon_base2 = vld1q_s8(y + d + i + j);
                int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i + j);
                int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i + j);
                int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i + j);
                int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i + j);
                int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i + j);
                int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i + j);

                neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
                neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
                neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
                neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
                neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
                neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
                neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
                neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

                neon_base1 = vld1q_s8(y + 8 * d + i + j);
                neon_base2 = vld1q_s8(y + 9 * d + i + j);
                neon_base3 = vld1q_s8(y + 10 * d + i + j);
                neon_base4 = vld1q_s8(y + 11 * d + i + j);
                neon_base5 = vld1q_s8(y + 12 * d + i + j);
                neon_base6 = vld1q_s8(y + 13 * d + i + j);
                neon_base7 = vld1q_s8(y + 14 * d + i + j);
                neon_base8 = vld1q_s8(y + 15 * d + i + j);

                neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
                neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
                neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
                neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
                neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
                neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
                neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
                neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
            }
        }

        /* Process remaining data */
        for (; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y + i);
            int8x16_t neon_base2 = vld1q_s8(y + d + i);
            int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
            int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
            int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
            int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
            int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
            int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
            neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_s8(y + 8 * d + i);
            neon_base2 = vld1q_s8(y + 9 * d + i);
            neon_base3 = vld1q_s8(y + 10 * d + i);
            neon_base4 = vld1q_s8(y + 11 * d + i);
            neon_base5 = vld1q_s8(y + 12 * d + i);
            neon_base6 = vld1q_s8(y + 13 * d + i);
            neon_base7 = vld1q_s8(y + 14 * d + i);
            neon_base8 = vld1q_s8(y + 15 * d + i);

            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
            neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
        }
        /* Sum the results */
        neon_res1 = vpaddq_s32(neon_res1, neon_res2);
        neon_res3 = vpaddq_s32(neon_res3, neon_res4);
        neon_res5 = vpaddq_s32(neon_res5, neon_res6);
        neon_res7 = vpaddq_s32(neon_res7, neon_res8);
        neon_res9 = vpaddq_s32(neon_res9, neon_res10);
        neon_res11 = vpaddq_s32(neon_res11, neon_res12);
        neon_res13 = vpaddq_s32(neon_res13, neon_res14);
        neon_res15 = vpaddq_s32(neon_res15, neon_res16);

        neon_res1 = vpaddq_s32(neon_res1, neon_res3);
        neon_res5 = vpaddq_s32(neon_res5, neon_res7);
        neon_res9 = vpaddq_s32(neon_res9, neon_res11);
        neon_res13 = vpaddq_s32(neon_res13, neon_res15);

        /* Store the results */
        vst1q_s32(dis, neon_res1);
        vst1q_s32(dis + 4, neon_res5);
        vst1q_s32(dis + 8, neon_res9);
        vst1q_s32(dis + 12, neon_res13);
    } else {
        /* Process without prefetching */
        for (i = 0; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y + i);
            int8x16_t neon_base2 = vld1q_s8(y + d + i);
            int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
            int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
            int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
            int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
            int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
            int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
            neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_s8(y + 8 * d + i);
            neon_base2 = vld1q_s8(y + 9 * d + i);
            neon_base3 = vld1q_s8(y + 10 * d + i);
            neon_base4 = vld1q_s8(y + 11 * d + i);
            neon_base5 = vld1q_s8(y + 12 * d + i);
            neon_base6 = vld1q_s8(y + 13 * d + i);
            neon_base7 = vld1q_s8(y + 14 * d + i);
            neon_base8 = vld1q_s8(y + 15 * d + i);

            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
            neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
        }
        /* Sum the results */
        neon_res1 = vpaddq_s32(neon_res1, neon_res2);
        neon_res3 = vpaddq_s32(neon_res3, neon_res4);
        neon_res5 = vpaddq_s32(neon_res5, neon_res6);
        neon_res7 = vpaddq_s32(neon_res7, neon_res8);
        neon_res9 = vpaddq_s32(neon_res9, neon_res10);
        neon_res11 = vpaddq_s32(neon_res11, neon_res12);
        neon_res13 = vpaddq_s32(neon_res13, neon_res14);
        neon_res15 = vpaddq_s32(neon_res15, neon_res16);

        neon_res1 = vpaddq_s32(neon_res1, neon_res3);
        neon_res5 = vpaddq_s32(neon_res5, neon_res7);
        neon_res9 = vpaddq_s32(neon_res9, neon_res11);
        neon_res13 = vpaddq_s32(neon_res13, neon_res15);

        /* Store the results */
        vst1q_s32(dis, neon_res1);
        vst1q_s32(dis + 4, neon_res5);
        vst1q_s32(dis + 8, neon_res9);
        vst1q_s32(dis + 12, neon_res13);
    }
    /* Handle remaining elements */
    if (i < d) {
        int32_t d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        int32_t d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        int32_t d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        int32_t d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        int32_t d4 = (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
        int32_t d5 = (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
        int32_t d6 = (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
        int32_t d7 = (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        int32_t d8 = (int32_t)(x[i]) * (int32_t) * (y + 8 * d + i);
        int32_t d9 = (int32_t)(x[i]) * (int32_t) * (y + 9 * d + i);
        int32_t d10 = (int32_t)(x[i]) * (int32_t) * (y + 10 * d + i);
        int32_t d11 = (int32_t)(x[i]) * (int32_t) * (y + 11 * d + i);
        int32_t d12 = (int32_t)(x[i]) * (int32_t) * (y + 12 * d + i);
        int32_t d13 = (int32_t)(x[i]) * (int32_t) * (y + 13 * d + i);
        int32_t d14 = (int32_t)(x[i]) * (int32_t) * (y + 14 * d + i);
        int32_t d15 = (int32_t)(x[i]) * (int32_t) * (y + 15 * d + i);

        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
            d4 += (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
            d5 += (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
            d6 += (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
            d7 += (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
            d8 += (int32_t)(x[i]) * (int32_t) * (y + 8 * d + i);
            d9 += (int32_t)(x[i]) * (int32_t) * (y + 9 * d + i);
            d10 += (int32_t)(x[i]) * (int32_t) * (y + 10 * d + i);
            d11 += (int32_t)(x[i]) * (int32_t) * (y + 11 * d + i);
            d12 += (int32_t)(x[i]) * (int32_t) * (y + 12 * d + i);
            d13 += (int32_t)(x[i]) * (int32_t) * (y + 13 * d + i);
            d14 += (int32_t)(x[i]) * (int32_t) * (y + 14 * d + i);
            d15 += (int32_t)(x[i]) * (int32_t) * (y + 15 * d + i);
        }

        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for 16 vectors with 8-bit integer precision and 32-bit float results.
 * @param x Pointer to the query vector (int8_t).
 * @param y Array of pointers to the database vectors (int8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_prefetch_batch16_s8f32(
    const int8_t *x, const int8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;   /* 128 / 8 */
    constexpr size_t prefetch_round = 64; /* 4 * single_round */

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);
    int32x4_t neon_res9 = vdupq_n_s32(0);
    int32x4_t neon_res10 = vdupq_n_s32(0);
    int32x4_t neon_res11 = vdupq_n_s32(0);
    int32x4_t neon_res12 = vdupq_n_s32(0);
    int32x4_t neon_res13 = vdupq_n_s32(0);
    int32x4_t neon_res14 = vdupq_n_s32(0);
    int32x4_t neon_res15 = vdupq_n_s32(0);
    int32x4_t neon_res16 = vdupq_n_s32(0);

    /* Process vectors with prefetching */
    if (d >= prefetch_round) {
        for (i = 0; i < d - prefetch_round; i += prefetch_round) {
            /* Prefetch data to L1 cache */
            prefetch_L1(x + i + prefetch_round);
            prefetch_Lx(y + i + prefetch_round);
            prefetch_Lx(y + d + i + prefetch_round);
            prefetch_Lx(y + 2 * d + i + prefetch_round);
            prefetch_Lx(y + 3 * d + i + prefetch_round);
            prefetch_Lx(y + 4 * d + i + prefetch_round);
            prefetch_Lx(y + 5 * d + i + prefetch_round);
            prefetch_Lx(y + 6 * d + i + prefetch_round);
            prefetch_Lx(y + 7 * d + i + prefetch_round);
            prefetch_Lx(y + 8 * d + i + prefetch_round);
            prefetch_Lx(y + 9 * d + i + prefetch_round);
            prefetch_Lx(y + 10 * d + i + prefetch_round);
            prefetch_Lx(y + 11 * d + i + prefetch_round);
            prefetch_Lx(y + 12 * d + i + prefetch_round);
            prefetch_Lx(y + 13 * d + i + prefetch_round);
            prefetch_Lx(y + 14 * d + i + prefetch_round);
            prefetch_Lx(y + 15 * d + i + prefetch_round);

            /* Process data in single_round chunks */
            for (size_t j = 0; j < prefetch_round; j += single_round) {
                int8x16_t neon_query = vld1q_s8(x + i + j);
                int8x16_t neon_base1 = vld1q_s8(y + i + j);
                int8x16_t neon_base2 = vld1q_s8(y + d + i + j);
                int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i + j);
                int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i + j);
                int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i + j);
                int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i + j);
                int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i + j);
                int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i + j);

                neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
                neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
                neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
                neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
                neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
                neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
                neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
                neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

                neon_base1 = vld1q_s8(y + 8 * d + i + j);
                neon_base2 = vld1q_s8(y + 9 * d + i + j);
                neon_base3 = vld1q_s8(y + 10 * d + i + j);
                neon_base4 = vld1q_s8(y + 11 * d + i + j);
                neon_base5 = vld1q_s8(y + 12 * d + i + j);
                neon_base6 = vld1q_s8(y + 13 * d + i + j);
                neon_base7 = vld1q_s8(y + 14 * d + i + j);
                neon_base8 = vld1q_s8(y + 15 * d + i + j);

                neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
                neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
                neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
                neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
                neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
                neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
                neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
                neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
            }
        }

        /* Process remaining data */
        for (; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y + i);
            int8x16_t neon_base2 = vld1q_s8(y + d + i);
            int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
            int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
            int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
            int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
            int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
            int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
            neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_s8(y + 8 * d + i);
            neon_base2 = vld1q_s8(y + 9 * d + i);
            neon_base3 = vld1q_s8(y + 10 * d + i);
            neon_base4 = vld1q_s8(y + 11 * d + i);
            neon_base5 = vld1q_s8(y + 12 * d + i);
            neon_base6 = vld1q_s8(y + 13 * d + i);
            neon_base7 = vld1q_s8(y + 14 * d + i);
            neon_base8 = vld1q_s8(y + 15 * d + i);

            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
            neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
        }
        neon_res1 = vpaddq_s32(neon_res1, neon_res2);
        neon_res3 = vpaddq_s32(neon_res3, neon_res4);
        neon_res5 = vpaddq_s32(neon_res5, neon_res6);
        neon_res7 = vpaddq_s32(neon_res7, neon_res8);
        neon_res9 = vpaddq_s32(neon_res9, neon_res10);
        neon_res11 = vpaddq_s32(neon_res11, neon_res12);
        neon_res13 = vpaddq_s32(neon_res13, neon_res14);
        neon_res15 = vpaddq_s32(neon_res15, neon_res16);
        neon_res1 = vpaddq_s32(neon_res1, neon_res3);
        neon_res5 = vpaddq_s32(neon_res5, neon_res7);
        neon_res9 = vpaddq_s32(neon_res9, neon_res11);
        neon_res13 = vpaddq_s32(neon_res13, neon_res15);

        vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_s32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_s32(neon_res13));
    } else {
        /* Process without prefetching */
        for (i = 0; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y + i);
            int8x16_t neon_base2 = vld1q_s8(y + d + i);
            int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
            int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
            int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
            int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
            int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
            int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
            neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_s8(y + 8 * d + i);
            neon_base2 = vld1q_s8(y + 9 * d + i);
            neon_base3 = vld1q_s8(y + 10 * d + i);
            neon_base4 = vld1q_s8(y + 11 * d + i);
            neon_base5 = vld1q_s8(y + 12 * d + i);
            neon_base6 = vld1q_s8(y + 13 * d + i);
            neon_base7 = vld1q_s8(y + 14 * d + i);
            neon_base8 = vld1q_s8(y + 15 * d + i);

            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
            neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
        }
        /* Sum the results */
        neon_res1 = vpaddq_s32(neon_res1, neon_res2);
        neon_res3 = vpaddq_s32(neon_res3, neon_res4);
        neon_res5 = vpaddq_s32(neon_res5, neon_res6);
        neon_res7 = vpaddq_s32(neon_res7, neon_res8);
        neon_res9 = vpaddq_s32(neon_res9, neon_res10);
        neon_res11 = vpaddq_s32(neon_res11, neon_res12);
        neon_res13 = vpaddq_s32(neon_res13, neon_res14);
        neon_res15 = vpaddq_s32(neon_res15, neon_res16);
        neon_res1 = vpaddq_s32(neon_res1, neon_res3);
        neon_res5 = vpaddq_s32(neon_res5, neon_res7);
        neon_res9 = vpaddq_s32(neon_res9, neon_res11);
        neon_res13 = vpaddq_s32(neon_res13, neon_res15);

        /* Store the results */
        vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_s32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_s32(neon_res13));
    }

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        float d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        float d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        float d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        float d4 = (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
        float d5 = (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
        float d6 = (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
        float d7 = (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        float d8 = (int32_t)(x[i]) * (int32_t) * (y + 8 * d + i);
        float d9 = (int32_t)(x[i]) * (int32_t) * (y + 9 * d + i);
        float d10 = (int32_t)(x[i]) * (int32_t) * (y + 10 * d + i);
        float d11 = (int32_t)(x[i]) * (int32_t) * (y + 11 * d + i);
        float d12 = (int32_t)(x[i]) * (int32_t) * (y + 12 * d + i);
        float d13 = (int32_t)(x[i]) * (int32_t) * (y + 13 * d + i);
        float d14 = (int32_t)(x[i]) * (int32_t) * (y + 14 * d + i);
        float d15 = (int32_t)(x[i]) * (int32_t) * (y + 15 * d + i);
        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
            d4 += (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
            d5 += (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
            d6 += (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
            d7 += (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
            d8 += (int32_t)(x[i]) * (int32_t) * (y + 8 * d + i);
            d9 += (int32_t)(x[i]) * (int32_t) * (y + 9 * d + i);
            d10 += (int32_t)(x[i]) * (int32_t) * (y + 10 * d + i);
            d11 += (int32_t)(x[i]) * (int32_t) * (y + 11 * d + i);
            d12 += (int32_t)(x[i]) * (int32_t) * (y + 12 * d + i);
            d13 += (int32_t)(x[i]) * (int32_t) * (y + 13 * d + i);
            d14 += (int32_t)(x[i]) * (int32_t) * (y + 14 * d + i);
            d15 += (int32_t)(x[i]) * (int32_t) * (y + 15 * d + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute inner products for a batch of vectors based on given indices.
 * @param dis Output array to store the results (float).
 * @param x Pointer to the query vector (int8_t).
 * @param y Pointer to the database vectors (int8_t).
 * @param ids Array of indices specifying which vectors to use from y.
 * @param d Dimension of the vectors.
 * @param ny Number of vectors to process.
 * @param dis_size Length of dis.
 */
int krl_inner_product_by_idx_s8f32(
    float *dis, const int8_t *x, const int8_t *y, const int64_t *ids, size_t d, size_t ny, size_t dis_size)
{
    size_t i = 0;
    const int8_t *__restrict listy[24];

    /* Process vectors in batches of 24 */
    for (; i + 24 <= ny; i += 24) {
        /* Load 24 vectors from y based on ids */
        listy[0] = (const int8_t *)(y + *(ids + i) * d);
        listy[1] = (const int8_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const int8_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const int8_t *)(y + *(ids + i + 3) * d);
        listy[4] = (const int8_t *)(y + *(ids + i + 4) * d);
        listy[5] = (const int8_t *)(y + *(ids + i + 5) * d);
        listy[6] = (const int8_t *)(y + *(ids + i + 6) * d);
        listy[7] = (const int8_t *)(y + *(ids + i + 7) * d);
        listy[8] = (const int8_t *)(y + *(ids + i + 8) * d);
        listy[9] = (const int8_t *)(y + *(ids + i + 9) * d);
        listy[10] = (const int8_t *)(y + *(ids + i + 10) * d);
        listy[11] = (const int8_t *)(y + *(ids + i + 11) * d);
        listy[12] = (const int8_t *)(y + *(ids + i + 12) * d);
        listy[13] = (const int8_t *)(y + *(ids + i + 13) * d);
        listy[14] = (const int8_t *)(y + *(ids + i + 14) * d);
        listy[15] = (const int8_t *)(y + *(ids + i + 15) * d);
        listy[16] = (const int8_t *)(y + *(ids + i + 16) * d);
        listy[17] = (const int8_t *)(y + *(ids + i + 17) * d);
        listy[18] = (const int8_t *)(y + *(ids + i + 18) * d);
        listy[19] = (const int8_t *)(y + *(ids + i + 19) * d);
        listy[20] = (const int8_t *)(y + *(ids + i + 20) * d);
        listy[21] = (const int8_t *)(y + *(ids + i + 21) * d);
        listy[22] = (const int8_t *)(y + *(ids + i + 22) * d);
        listy[23] = (const int8_t *)(y + *(ids + i + 23) * d);

        /* Compute inner products for the batch of 24 vectors */
        krl_inner_product_idx_prefetch_batch24_s8s32(x, listy, d, dis + i);
    }

    /* Handle remaining vectors */
    if (i + 16 <= ny) {
        /* Load 16 vectors from y based on ids */
        listy[0] = (const int8_t *)(y + *(ids + i) * d);
        listy[1] = (const int8_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const int8_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const int8_t *)(y + *(ids + i + 3) * d);
        listy[4] = (const int8_t *)(y + *(ids + i + 4) * d);
        listy[5] = (const int8_t *)(y + *(ids + i + 5) * d);
        listy[6] = (const int8_t *)(y + *(ids + i + 6) * d);
        listy[7] = (const int8_t *)(y + *(ids + i + 7) * d);
        listy[8] = (const int8_t *)(y + *(ids + i + 8) * d);
        listy[9] = (const int8_t *)(y + *(ids + i + 9) * d);
        listy[10] = (const int8_t *)(y + *(ids + i + 10) * d);
        listy[11] = (const int8_t *)(y + *(ids + i + 11) * d);
        listy[12] = (const int8_t *)(y + *(ids + i + 12) * d);
        listy[13] = (const int8_t *)(y + *(ids + i + 13) * d);
        listy[14] = (const int8_t *)(y + *(ids + i + 14) * d);
        listy[15] = (const int8_t *)(y + *(ids + i + 15) * d);

        /* Compute inner products for the batch of 16 vectors */
        krl_inner_product_idx_prefetch_batch16_s8s32(x, listy, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        /* Load 8 vectors from y based on ids */
        listy[0] = (const int8_t *)(y + *(ids + i) * d);
        listy[1] = (const int8_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const int8_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const int8_t *)(y + *(ids + i + 3) * d);
        listy[4] = (const int8_t *)(y + *(ids + i + 4) * d);
        listy[5] = (const int8_t *)(y + *(ids + i + 5) * d);
        listy[6] = (const int8_t *)(y + *(ids + i + 6) * d);
        listy[7] = (const int8_t *)(y + *(ids + i + 7) * d);

        /* Compute inner products for the batch of 8 vectors */
        krl_inner_product_idx_batch8_s8s32(x, listy, d, dis + i);
        i += 8;
    }

    /* Handle remaining vectors */
    if (ny & 4) {
        /* Load 4 vectors from y based on ids */
        listy[0] = (const int8_t *)(y + *(ids + i) * d);
        listy[1] = (const int8_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const int8_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const int8_t *)(y + *(ids + i + 3) * d);

        /* Compute inner products for the batch of 4 vectors */
        krl_inner_product_idx_batch4_s8s32(x, listy, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        /* Load 2 vectors from y based on ids */
        const int8_t *y0 = y + *(ids + i) * d;
        const int8_t *y1 = y + *(ids + i + 1) * d;

        /* Compute inner products for the batch of 2 vectors */
        krl_inner_product_idx_batch2_s8s32(x, y0, y1, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        /* Compute inner product for the remaining single vector */
        dis[i] = (float)krl_inner_product_s8s32(x, y + d * ids[i], d);
    }
    return SUCCESS;
}

/*
 * @brief Compute inner products for a batch of vectors.
 * @param dis Output array to store the results (int32_t).
 * @param x Pointer to the query vector (int8_t).
 * @param y Pointer to the database vectors (int8_t).
 * @param ny Number of vectors to process.
 * @param d Dimension of the vectors.
 */
void krl_inner_product_ny_s8s32(int32_t *dis, const int8_t *x, const int8_t *y, size_t ny, size_t d)
{
    size_t i = 0;

    /* Process vectors in batches of 16 */
    for (; i + 16 <= ny; i += 16) {
        krl_inner_product_prefetch_batch16_s8s32(x, y + i * d, d, dis + i);
    }

    /* Handle remaining vectors */
    if (ny & 8) {
        krl_inner_product_batch8_s8s32(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_inner_product_batch4_s8s32(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_inner_product_batch2_s8s32(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        dis[i] = (int32_t)krl_inner_product_s8s32(x, y + i * d, d);
    }
}

/*
 * @brief Compute inner products for a batch of vectors with float results.
 * @param dis Output array to store the results (float).
 * @param x Pointer to the query vector (int8_t).
 * @param y Pointer to the database vectors (int8_t).
 * @param ny Number of vectors to process.
 * @param d Dimension of the vectors.
 * @param dis_size Length of dis.
 */
int krl_inner_product_ny_s8f32(float *dis, const int8_t *x, const int8_t *y, size_t ny, size_t d, size_t dis_size)
{
    size_t i = 0;

    /* Process vectors in batches of 16 */
    for (; i + 16 <= ny; i += 16) {
        krl_inner_product_prefetch_batch16_s8f32(x, y + i * d, d, dis + i);
    }

    /* Handle remaining vectors */
    if (ny & 8) {
        krl_inner_product_batch8_s8f32(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_inner_product_batch4_s8f32(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_inner_product_batch2_s8f32(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        dis[i] = (float)krl_inner_product_s8s32(x, y + i * d, d);
    }
    return SUCCESS;
}
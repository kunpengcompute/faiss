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

/*
 * @brief Compute the inner product of two float16 vectors using NEON instructions
 * @param x Pointer to the first float16 vector
 * @param y Pointer to the second float16 vector (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @return The computed inner product as a float16 value
 */
KRL_IMPRECISE_FUNCTION_BEGIN
float16_t krl_inner_product_f16f16(const float16_t *x, const float16_t *__restrict y, const size_t d)
{
    size_t i;
    float16_t res;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 32;
    float16x8_t res1 = vdupq_n_f16(0.0f);
    float16x8_t res2 = vdupq_n_f16(0.0f);
    float16x8_t res3 = vdupq_n_f16(0.0f);
    float16x8_t res4 = vdupq_n_f16(0.0f);

    /* Compute inner product using NEON vector operations */
    for (i = 0; i + double_round <= d; i += double_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);
        float16x8_t x8_2 = vld1q_f16(x + i + 16);
        float16x8_t x8_3 = vld1q_f16(x + i + 24);

        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + i + 8);
        float16x8_t y8_2 = vld1q_f16(y + i + 16);
        float16x8_t y8_3 = vld1q_f16(y + i + 24);

        res1 = vfmaq_f16(res1, x8_0, y8_0);
        res2 = vfmaq_f16(res2, x8_1, y8_1);
        res3 = vfmaq_f16(res3, x8_2, y8_2);
        res4 = vfmaq_f16(res4, x8_3, y8_3);
    }

    /* Handle remaining elements with single-round processing */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);
        res1 = vfmaq_f16(res1, x8_0, y8_0);
    }

    /* Accumulate results from all vector registers */
    res1 = vaddq_f16(res1, res2);
    res3 = vaddq_f16(res3, res4);
    res1 = vaddq_f16(res1, res3);

    /* Reduce the vector result to a scalar value */
    /* 8 -> 4 */
    res1 = vpaddq_f16(res1, res1);
    /* 4 -> 2 */
    res1 = vpaddq_f16(res1, res1);
    /* 2 -> 1 */
    res1 = vpaddq_f16(res1, res1);
    res = vgetq_lane_f16(res1, 0);

    /* Handle any remaining elements */
    for (; i < d; i++) {
        res += (float16_t)(x[i] * y[i]);
    }

    return res;
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of a float16 vector with two batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y0 Pointer to the first batch of float16 vectors (with restrict qualifier for better optimization)
 * @param y1 Pointer to the second batch of float16 vectors (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch2_f16f16(
    const float16_t *x, const float16_t *__restrict y0, const float16_t *__restrict y1, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 16;
    float16x8_t res1 = vdupq_n_f16(0.0f);
    float16x8_t res2 = vdupq_n_f16(0.0f);
    float16x8_t res3 = vdupq_n_f16(0.0f);
    float16x8_t res4 = vdupq_n_f16(0.0f);

    /* Compute inner product using NEON vector operations */
    for (i = 0; i + double_round <= d; i += double_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);

        float16x8_t y8_0 = vld1q_f16(y0 + i);
        float16x8_t y8_1 = vld1q_f16(y0 + i + 8);
        float16x8_t y8_2 = vld1q_f16(y1 + i);
        float16x8_t y8_3 = vld1q_f16(y1 + i + 8);

        res1 = vfmaq_f16(res1, x8_0, y8_0);
        res2 = vfmaq_f16(res2, x8_1, y8_1);
        res3 = vfmaq_f16(res3, x8_0, y8_2);
        res4 = vfmaq_f16(res4, x8_1, y8_3);
    }

    /* Handle remaining elements with single-round processing */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y0 + i);
        float16x8_t y8_1 = vld1q_f16(y1 + i);

        res1 = vfmaq_f16(res1, x8_0, y8_0);
        res3 = vfmaq_f16(res3, x8_0, y8_1);
    }

    /* Accumulate results from all vector registers */
    res1 = vpaddq_f16(res1, res2);
    res3 = vpaddq_f16(res3, res4);

    /* Reduce the vector result to scalar values */
    /* 8 -> 4 */
    res1 = vpaddq_f16(res1, res3);
    /* 4 -> 2 */
    res1 = vpaddq_f16(res1, res1);
    /* 2 -> 1 */
    res1 = vpaddq_f16(res1, res1);

    dis[0] = vgetq_lane_f16(res1, 0);
    dis[1] = vgetq_lane_f16(res1, 1);

    /* Handle any remaining elements */
    for (; i < d; i++) {
        dis[0] += (float16_t)(x[i] * y0[i]);
        dis[1] += (float16_t)(x[i] * y1[i]);
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of a float16 vector with four batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of pointers to the four batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch4_f16f16(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8;

    float16x8_t neon_res1 = vdupq_n_f16(0.0f);
    float16x8_t neon_res2 = vdupq_n_f16(0.0f);
    float16x8_t neon_res3 = vdupq_n_f16(0.0f);
    float16x8_t neon_res4 = vdupq_n_f16(0.0f);

    /* Compute inner product using NEON vector operations */
    for (i = 0; i + single_round <= d; i += single_round) {
        float16x8_t neon_query = vld1q_f16(x + i);
        float16x8_t neon_base1 = vld1q_f16(y[0] + i);
        float16x8_t neon_base2 = vld1q_f16(y[1] + i);
        float16x8_t neon_base3 = vld1q_f16(y[2] + i);
        float16x8_t neon_base4 = vld1q_f16(y[3] + i);

        neon_res1 = vfmaq_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmaq_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmaq_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmaq_f16(neon_res4, neon_query, neon_base4);
    }

    /* Accumulate results from all vector registers */
    neon_res1 = vpaddq_f16(neon_res1, neon_res2);
    neon_res3 = vpaddq_f16(neon_res3, neon_res4);
    neon_res1 = vpaddq_f16(neon_res1, neon_res3);
    neon_res1 = vpaddq_f16(neon_res1, neon_res1);

    /* Store the results */
    vst1_f16(dis, vget_low_f16(neon_res1));

    /* Handle any remaining elements */
    if (i < d) {
        float16_t d0 = x[i] * *(y[0] + i);
        float16_t d1 = x[i] * *(y[1] + i);
        float16_t d2 = x[i] * *(y[2] + i);
        float16_t d3 = x[i] * *(y[3] + i);
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y[0] + i);
            d1 += x[i] * *(y[1] + i);
            d2 += x[i] * *(y[2] + i);
            d3 += x[i] * *(y[3] + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of a float16 vector with eight batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of pointers to the eight batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch8_f16f16(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8;

    float16x8_t neon_res1 = vdupq_n_f16(0.0f);
    float16x8_t neon_res2 = vdupq_n_f16(0.0f);
    float16x8_t neon_res3 = vdupq_n_f16(0.0f);
    float16x8_t neon_res4 = vdupq_n_f16(0.0f);
    float16x8_t neon_res5 = vdupq_n_f16(0.0f);
    float16x8_t neon_res6 = vdupq_n_f16(0.0f);
    float16x8_t neon_res7 = vdupq_n_f16(0.0f);
    float16x8_t neon_res8 = vdupq_n_f16(0.0f);

    /* Compute inner product using NEON vector operations */
    for (i = 0; i + single_round <= d; i += single_round) {
        float16x8_t neon_query = vld1q_f16(x + i);
        float16x8_t neon_base1 = vld1q_f16(y[0] + i);
        float16x8_t neon_base2 = vld1q_f16(y[1] + i);
        float16x8_t neon_base3 = vld1q_f16(y[2] + i);
        float16x8_t neon_base4 = vld1q_f16(y[3] + i);
        float16x8_t neon_base5 = vld1q_f16(y[4] + i);
        float16x8_t neon_base6 = vld1q_f16(y[5] + i);
        float16x8_t neon_base7 = vld1q_f16(y[6] + i);
        float16x8_t neon_base8 = vld1q_f16(y[7] + i);

        neon_res1 = vfmaq_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmaq_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmaq_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmaq_f16(neon_res4, neon_query, neon_base4);
        neon_res5 = vfmaq_f16(neon_res5, neon_query, neon_base5);
        neon_res6 = vfmaq_f16(neon_res6, neon_query, neon_base6);
        neon_res7 = vfmaq_f16(neon_res7, neon_query, neon_base7);
        neon_res8 = vfmaq_f16(neon_res8, neon_query, neon_base8);
    }

    /* Accumulate results from all vector registers */
    neon_res1 = vpaddq_f16(neon_res1, neon_res2);
    neon_res3 = vpaddq_f16(neon_res3, neon_res4);
    neon_res5 = vpaddq_f16(neon_res5, neon_res6);
    neon_res7 = vpaddq_f16(neon_res7, neon_res8);
    neon_res1 = vpaddq_f16(neon_res1, neon_res3);
    neon_res5 = vpaddq_f16(neon_res5, neon_res7);
    neon_res1 = vpaddq_f16(neon_res1, neon_res5);

    /* Store the results */
    vst1q_f16(dis, neon_res1);

    /* Handle any remaining elements */
    if (i < d) {
        float16_t d0 = x[i] * *(y[0] + i);
        float16_t d1 = x[i] * *(y[1] + i);
        float16_t d2 = x[i] * *(y[2] + i);
        float16_t d3 = x[i] * *(y[3] + i);
        float16_t d4 = x[i] * *(y[4] + i);
        float16_t d5 = x[i] * *(y[5] + i);
        float16_t d6 = x[i] * *(y[6] + i);
        float16_t d7 = x[i] * *(y[7] + i);
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y[0] + i);
            d1 += x[i] * *(y[1] + i);
            d2 += x[i] * *(y[2] + i);
            d3 += x[i] * *(y[3] + i);
            d4 += x[i] * *(y[4] + i);
            d5 += x[i] * *(y[5] + i);
            d6 += x[i] * *(y[6] + i);
            d7 += x[i] * *(y[7] + i);
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
 * @brief Compute the inner product of a float16 vector with 16 batches of float16 vectors using NEON instructions with
 * prefetch optimization
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of pointers to the 16 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_prefetch_batch16_f16f16(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */

    float16x8_t neon_res1 = vdupq_n_f16(0.0f);
    float16x8_t neon_res2 = vdupq_n_f16(0.0f);
    float16x8_t neon_res3 = vdupq_n_f16(0.0f);
    float16x8_t neon_res4 = vdupq_n_f16(0.0f);
    float16x8_t neon_res5 = vdupq_n_f16(0.0f);
    float16x8_t neon_res6 = vdupq_n_f16(0.0f);
    float16x8_t neon_res7 = vdupq_n_f16(0.0f);
    float16x8_t neon_res8 = vdupq_n_f16(0.0f);
    float16x8_t neon_res9 = vdupq_n_f16(0.0f);
    float16x8_t neon_res10 = vdupq_n_f16(0.0f);
    float16x8_t neon_res11 = vdupq_n_f16(0.0f);
    float16x8_t neon_res12 = vdupq_n_f16(0.0f);
    float16x8_t neon_res13 = vdupq_n_f16(0.0f);
    float16x8_t neon_res14 = vdupq_n_f16(0.0f);
    float16x8_t neon_res15 = vdupq_n_f16(0.0f);
    float16x8_t neon_res16 = vdupq_n_f16(0.0f);

    /* Main computation loop with prefetch optimization */
    if (d >= multi_round) {
        for (i = 0; i < d - multi_round; i += multi_round) {
            /* Prefetch data for better memory access */
            prefetch_L1(x + i + multi_round);
            prefetch_Lx(y[0] + i + multi_round);
            prefetch_Lx(y[1] + i + multi_round);
            prefetch_Lx(y[2] + i + multi_round);
            prefetch_Lx(y[3] + i + multi_round);
            prefetch_Lx(y[4] + i + multi_round);
            prefetch_Lx(y[5] + i + multi_round);
            prefetch_Lx(y[6] + i + multi_round);
            prefetch_Lx(y[7] + i + multi_round);
            prefetch_Lx(y[8] + i + multi_round);
            prefetch_Lx(y[9] + i + multi_round);
            prefetch_Lx(y[10] + i + multi_round);
            prefetch_Lx(y[11] + i + multi_round);
            prefetch_Lx(y[12] + i + multi_round);
            prefetch_Lx(y[13] + i + multi_round);
            prefetch_Lx(y[14] + i + multi_round);
            prefetch_Lx(y[15] + i + multi_round);

            /* Process data in single_round chunks */
            for (size_t j = 0; j < multi_round; j += single_round) {
                const float16x8_t neon_query = vld1q_f16(x + i + j);
                float16x8_t neon_base1 = vld1q_f16(y[0] + i + j);
                float16x8_t neon_base2 = vld1q_f16(y[1] + i + j);
                float16x8_t neon_base3 = vld1q_f16(y[2] + i + j);
                float16x8_t neon_base4 = vld1q_f16(y[3] + i + j);
                float16x8_t neon_base5 = vld1q_f16(y[4] + i + j);
                float16x8_t neon_base6 = vld1q_f16(y[5] + i + j);
                float16x8_t neon_base7 = vld1q_f16(y[6] + i + j);
                float16x8_t neon_base8 = vld1q_f16(y[7] + i + j);

                neon_res1 = vfmaq_f16(neon_res1, neon_query, neon_base1);
                neon_res2 = vfmaq_f16(neon_res2, neon_query, neon_base2);
                neon_res3 = vfmaq_f16(neon_res3, neon_query, neon_base3);
                neon_res4 = vfmaq_f16(neon_res4, neon_query, neon_base4);
                neon_res5 = vfmaq_f16(neon_res5, neon_query, neon_base5);
                neon_res6 = vfmaq_f16(neon_res6, neon_query, neon_base6);
                neon_res7 = vfmaq_f16(neon_res7, neon_query, neon_base7);
                neon_res8 = vfmaq_f16(neon_res8, neon_query, neon_base8);

                neon_base1 = vld1q_f16(y[8] + i + j);
                neon_base2 = vld1q_f16(y[9] + i + j);
                neon_base3 = vld1q_f16(y[10] + i + j);
                neon_base4 = vld1q_f16(y[11] + i + j);
                neon_base5 = vld1q_f16(y[12] + i + j);
                neon_base6 = vld1q_f16(y[13] + i + j);
                neon_base7 = vld1q_f16(y[14] + i + j);
                neon_base8 = vld1q_f16(y[15] + i + j);

                neon_res9 = vfmaq_f16(neon_res9, neon_query, neon_base1);
                neon_res10 = vfmaq_f16(neon_res10, neon_query, neon_base2);
                neon_res11 = vfmaq_f16(neon_res11, neon_query, neon_base3);
                neon_res12 = vfmaq_f16(neon_res12, neon_query, neon_base4);
                neon_res13 = vfmaq_f16(neon_res13, neon_query, neon_base5);
                neon_res14 = vfmaq_f16(neon_res14, neon_query, neon_base6);
                neon_res15 = vfmaq_f16(neon_res15, neon_query, neon_base7);
                neon_res16 = vfmaq_f16(neon_res16, neon_query, neon_base8);
            }
        }

        /* Handle remaining elements after multi_round processing */
        for (; i + single_round <= d; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            float16x8_t neon_base5 = vld1q_f16(y[4] + i);
            float16x8_t neon_base6 = vld1q_f16(y[5] + i);
            float16x8_t neon_base7 = vld1q_f16(y[6] + i);
            float16x8_t neon_base8 = vld1q_f16(y[7] + i);

            neon_res1 = vfmaq_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmaq_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmaq_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmaq_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmaq_f16(neon_res8, neon_query, neon_base8);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base5 = vld1q_f16(y[12] + i);
            neon_base6 = vld1q_f16(y[13] + i);
            neon_base7 = vld1q_f16(y[14] + i);
            neon_base8 = vld1q_f16(y[15] + i);

            neon_res9 = vfmaq_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmaq_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmaq_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmaq_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmaq_f16(neon_res16, neon_query, neon_base8);
        }
    } else {
        /* Handle cases where d < multi_round */
        for (i = 0; i + single_round <= d; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            float16x8_t neon_base5 = vld1q_f16(y[4] + i);
            float16x8_t neon_base6 = vld1q_f16(y[5] + i);
            float16x8_t neon_base7 = vld1q_f16(y[6] + i);
            float16x8_t neon_base8 = vld1q_f16(y[7] + i);

            neon_res1 = vfmaq_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmaq_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmaq_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmaq_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmaq_f16(neon_res8, neon_query, neon_base8);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base5 = vld1q_f16(y[12] + i);
            neon_base6 = vld1q_f16(y[13] + i);
            neon_base7 = vld1q_f16(y[14] + i);
            neon_base8 = vld1q_f16(y[15] + i);

            neon_res9 = vfmaq_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmaq_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmaq_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmaq_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmaq_f16(neon_res16, neon_query, neon_base8);
        }
    }

    /* Accumulate results from all vector registers */
    neon_res1 = vpaddq_f16(neon_res1, neon_res2);
    neon_res3 = vpaddq_f16(neon_res3, neon_res4);
    neon_res5 = vpaddq_f16(neon_res5, neon_res6);
    neon_res7 = vpaddq_f16(neon_res7, neon_res8);
    neon_res9 = vpaddq_f16(neon_res9, neon_res10);
    neon_res11 = vpaddq_f16(neon_res11, neon_res12);
    neon_res13 = vpaddq_f16(neon_res13, neon_res14);
    neon_res15 = vpaddq_f16(neon_res15, neon_res16);

    neon_res1 = vpaddq_f16(neon_res1, neon_res3);
    neon_res5 = vpaddq_f16(neon_res5, neon_res7);
    neon_res9 = vpaddq_f16(neon_res9, neon_res11);
    neon_res13 = vpaddq_f16(neon_res13, neon_res15);

    neon_res1 = vpaddq_f16(neon_res1, neon_res5);
    neon_res9 = vpaddq_f16(neon_res9, neon_res13);

    /* Store the results */
    vst1q_f16(dis, neon_res1);
    vst1q_f16(dis + 8, neon_res9);

    /* Handle any remaining elements */
    if (i < d) {
        float d0 = x[i] * *(y[0] + i);
        float d1 = x[i] * *(y[1] + i);
        float d2 = x[i] * *(y[2] + i);
        float d3 = x[i] * *(y[3] + i);
        float d4 = x[i] * *(y[4] + i);
        float d5 = x[i] * *(y[5] + i);
        float d6 = x[i] * *(y[6] + i);
        float d7 = x[i] * *(y[7] + i);
        float d8 = x[i] * *(y[8] + i);
        float d9 = x[i] * *(y[9] + i);
        float d10 = x[i] * *(y[10] + i);
        float d11 = x[i] * *(y[11] + i);
        float d12 = x[i] * *(y[12] + i);
        float d13 = x[i] * *(y[13] + i);
        float d14 = x[i] * *(y[14] + i);
        float d15 = x[i] * *(y[15] + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y[0] + i);
            d1 += x[i] * *(y[1] + i);
            d2 += x[i] * *(y[2] + i);
            d3 += x[i] * *(y[3] + i);
            d4 += x[i] * *(y[4] + i);
            d5 += x[i] * *(y[5] + i);
            d6 += x[i] * *(y[6] + i);
            d7 += x[i] * *(y[7] + i);
            d8 += x[i] * *(y[8] + i);
            d9 += x[i] * *(y[9] + i);
            d10 += x[i] * *(y[10] + i);
            d11 += x[i] * *(y[11] + i);
            d12 += x[i] * *(y[12] + i);
            d13 += x[i] * *(y[13] + i);
            d14 += x[i] * *(y[14] + i);
            d15 += x[i] * *(y[15] + i);
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
 * @brief Compute the inner product of a float16 vector with 24 batches of float16 vectors using NEON instructions with
 * prefetch optimization
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of pointers to the 24 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_prefetch_batch24_f16f16(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */

    float16x8_t neon_res1 = vdupq_n_f16(0.0f);
    float16x8_t neon_res2 = vdupq_n_f16(0.0f);
    float16x8_t neon_res3 = vdupq_n_f16(0.0f);
    float16x8_t neon_res4 = vdupq_n_f16(0.0f);
    float16x8_t neon_res5 = vdupq_n_f16(0.0f);
    float16x8_t neon_res6 = vdupq_n_f16(0.0f);
    float16x8_t neon_res7 = vdupq_n_f16(0.0f);
    float16x8_t neon_res8 = vdupq_n_f16(0.0f);
    float16x8_t neon_res9 = vdupq_n_f16(0.0f);
    float16x8_t neon_res10 = vdupq_n_f16(0.0f);
    float16x8_t neon_res11 = vdupq_n_f16(0.0f);
    float16x8_t neon_res12 = vdupq_n_f16(0.0f);
    float16x8_t neon_res13 = vdupq_n_f16(0.0f);
    float16x8_t neon_res14 = vdupq_n_f16(0.0f);
    float16x8_t neon_res15 = vdupq_n_f16(0.0f);
    float16x8_t neon_res16 = vdupq_n_f16(0.0f);
    float16x8_t neon_res17 = vdupq_n_f16(0.0f);
    float16x8_t neon_res18 = vdupq_n_f16(0.0f);
    float16x8_t neon_res19 = vdupq_n_f16(0.0f);
    float16x8_t neon_res20 = vdupq_n_f16(0.0f);
    float16x8_t neon_res21 = vdupq_n_f16(0.0f);
    float16x8_t neon_res22 = vdupq_n_f16(0.0f);
    float16x8_t neon_res23 = vdupq_n_f16(0.0f);
    float16x8_t neon_res24 = vdupq_n_f16(0.0f);

    /* Main computation loop with prefetch optimization */
    if (d >= multi_round) {
        for (i = 0; i < d - multi_round; i += multi_round) {
            const size_t next_i = i + multi_round;
            /* Prefetch data for better memory access */
            prefetch_L1(x + next_i);
            prefetch_Lx(y[0] + next_i);
            prefetch_Lx(y[1] + next_i);
            prefetch_Lx(y[2] + next_i);
            prefetch_Lx(y[3] + next_i);
            prefetch_Lx(y[4] + next_i);
            prefetch_Lx(y[5] + next_i);
            prefetch_Lx(y[6] + next_i);
            prefetch_Lx(y[7] + next_i);
            prefetch_Lx(y[8] + next_i);
            prefetch_Lx(y[9] + next_i);
            prefetch_Lx(y[10] + next_i);
            prefetch_Lx(y[11] + next_i);
            prefetch_Lx(y[12] + next_i);
            prefetch_Lx(y[13] + next_i);
            prefetch_Lx(y[14] + next_i);
            prefetch_Lx(y[15] + next_i);
            prefetch_Lx(y[16] + next_i);
            prefetch_Lx(y[17] + next_i);
            prefetch_Lx(y[18] + next_i);
            prefetch_Lx(y[19] + next_i);
            prefetch_Lx(y[20] + next_i);
            prefetch_Lx(y[21] + next_i);
            prefetch_Lx(y[22] + next_i);
            prefetch_Lx(y[23] + next_i);

            /* Process data in single_round chunks */
            for (size_t j = i; j < next_i; j += single_round) {
                const float16x8_t neon_query = vld1q_f16(x + j);
                float16x8_t neon_base1 = vld1q_f16(y[0] + j);
                float16x8_t neon_base2 = vld1q_f16(y[1] + j);
                float16x8_t neon_base3 = vld1q_f16(y[2] + j);
                float16x8_t neon_base4 = vld1q_f16(y[3] + j);

                neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_query);
                neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_query);
                neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_query);
                neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y[4] + j);
                neon_base2 = vld1q_f16(y[5] + j);
                neon_base3 = vld1q_f16(y[6] + j);
                neon_base4 = vld1q_f16(y[7] + j);

                neon_res5 = vfmaq_f16(neon_res5, neon_base1, neon_query);
                neon_res6 = vfmaq_f16(neon_res6, neon_base2, neon_query);
                neon_res7 = vfmaq_f16(neon_res7, neon_base3, neon_query);
                neon_res8 = vfmaq_f16(neon_res8, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y[8] + j);
                neon_base2 = vld1q_f16(y[9] + j);
                neon_base3 = vld1q_f16(y[10] + j);
                neon_base4 = vld1q_f16(y[11] + j);

                neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_query);
                neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_query);
                neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_query);
                neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y[12] + j);
                neon_base2 = vld1q_f16(y[13] + j);
                neon_base3 = vld1q_f16(y[14] + j);
                neon_base4 = vld1q_f16(y[15] + j);

                neon_res13 = vfmaq_f16(neon_res13, neon_base1, neon_query);
                neon_res14 = vfmaq_f16(neon_res14, neon_base2, neon_query);
                neon_res15 = vfmaq_f16(neon_res15, neon_base3, neon_query);
                neon_res16 = vfmaq_f16(neon_res16, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y[16] + j);
                neon_base2 = vld1q_f16(y[17] + j);
                neon_base3 = vld1q_f16(y[18] + j);
                neon_base4 = vld1q_f16(y[19] + j);

                neon_res17 = vfmaq_f16(neon_res17, neon_base1, neon_query);
                neon_res18 = vfmaq_f16(neon_res18, neon_base2, neon_query);
                neon_res19 = vfmaq_f16(neon_res19, neon_base3, neon_query);
                neon_res20 = vfmaq_f16(neon_res20, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y[20] + j);
                neon_base2 = vld1q_f16(y[21] + j);
                neon_base3 = vld1q_f16(y[22] + j);
                neon_base4 = vld1q_f16(y[23] + j);

                neon_res21 = vfmaq_f16(neon_res21, neon_base1, neon_query);
                neon_res22 = vfmaq_f16(neon_res22, neon_base2, neon_query);
                neon_res23 = vfmaq_f16(neon_res23, neon_base3, neon_query);
                neon_res24 = vfmaq_f16(neon_res24, neon_base4, neon_query);
            }
        }

        /* Handle remaining elements after multi_round processing */
        for (; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);

            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);

            neon_res5 = vfmaq_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmaq_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmaq_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmaq_f16(neon_res8, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);

            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);

            neon_res13 = vfmaq_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmaq_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmaq_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmaq_f16(neon_res16, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);

            neon_res17 = vfmaq_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmaq_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmaq_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmaq_f16(neon_res20, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);

            neon_res21 = vfmaq_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmaq_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmaq_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmaq_f16(neon_res24, neon_base4, neon_query);
        }
    } else if (d >= single_round) {
        /* Handle cases where multi_round > d >= single_round */
        for (i = 0; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);

            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);

            neon_res5 = vfmaq_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmaq_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmaq_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmaq_f16(neon_res8, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);

            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);

            neon_res13 = vfmaq_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmaq_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmaq_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmaq_f16(neon_res16, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);

            neon_res17 = vfmaq_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmaq_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmaq_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmaq_f16(neon_res20, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);

            neon_res21 = vfmaq_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmaq_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmaq_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmaq_f16(neon_res24, neon_base4, neon_query);
        }
    } else {
        /* Handle cases where d < single_round */
        i = 0;
    }

    /* Accumulate results from all vector registers */
    neon_res1 = vpaddq_f16(neon_res1, neon_res2);
    neon_res3 = vpaddq_f16(neon_res3, neon_res4);
    neon_res5 = vpaddq_f16(neon_res5, neon_res6);
    neon_res7 = vpaddq_f16(neon_res7, neon_res8);
    neon_res9 = vpaddq_f16(neon_res9, neon_res10);
    neon_res11 = vpaddq_f16(neon_res11, neon_res12);
    neon_res13 = vpaddq_f16(neon_res13, neon_res14);
    neon_res15 = vpaddq_f16(neon_res15, neon_res16);
    neon_res17 = vpaddq_f16(neon_res17, neon_res18);
    neon_res19 = vpaddq_f16(neon_res19, neon_res20);
    neon_res21 = vpaddq_f16(neon_res21, neon_res22);
    neon_res23 = vpaddq_f16(neon_res23, neon_res24);

    neon_res1 = vpaddq_f16(neon_res1, neon_res3);
    neon_res5 = vpaddq_f16(neon_res5, neon_res7);
    neon_res9 = vpaddq_f16(neon_res9, neon_res11);
    neon_res13 = vpaddq_f16(neon_res13, neon_res15);
    neon_res17 = vpaddq_f16(neon_res17, neon_res19);
    neon_res21 = vpaddq_f16(neon_res21, neon_res23);

    neon_res1 = vpaddq_f16(neon_res1, neon_res5);
    neon_res9 = vpaddq_f16(neon_res9, neon_res13);
    neon_res17 = vpaddq_f16(neon_res17, neon_res21);

    /* Store the results */
    vst1q_f16(dis, neon_res1);
    vst1q_f16(dis + 8, neon_res9);
    vst1q_f16(dis + 16, neon_res17);

    /* Handle any remaining elements */
    if (i < d) {
        float16_t d0 = x[i] * *(y[0] + i);
        float16_t d1 = x[i] * *(y[1] + i);
        float16_t d2 = x[i] * *(y[2] + i);
        float16_t d3 = x[i] * *(y[3] + i);
        float16_t d4 = x[i] * *(y[4] + i);
        float16_t d5 = x[i] * *(y[5] + i);
        float16_t d6 = x[i] * *(y[6] + i);
        float16_t d7 = x[i] * *(y[7] + i);
        float16_t d8 = x[i] * *(y[8] + i);
        float16_t d9 = x[i] * *(y[9] + i);
        float16_t d10 = x[i] * *(y[10] + i);
        float16_t d11 = x[i] * *(y[11] + i);
        float16_t d12 = x[i] * *(y[12] + i);
        float16_t d13 = x[i] * *(y[13] + i);
        float16_t d14 = x[i] * *(y[14] + i);
        float16_t d15 = x[i] * *(y[15] + i);
        float16_t d16 = x[i] * *(y[16] + i);
        float16_t d17 = x[i] * *(y[17] + i);
        float16_t d18 = x[i] * *(y[18] + i);
        float16_t d19 = x[i] * *(y[19] + i);
        float16_t d20 = x[i] * *(y[20] + i);
        float16_t d21 = x[i] * *(y[21] + i);
        float16_t d22 = x[i] * *(y[22] + i);
        float16_t d23 = x[i] * *(y[23] + i);
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y[0] + i);
            d1 += x[i] * *(y[1] + i);
            d2 += x[i] * *(y[2] + i);
            d3 += x[i] * *(y[3] + i);
            d4 += x[i] * *(y[4] + i);
            d5 += x[i] * *(y[5] + i);
            d6 += x[i] * *(y[6] + i);
            d7 += x[i] * *(y[7] + i);
            d8 += x[i] * *(y[8] + i);
            d9 += x[i] * *(y[9] + i);
            d10 += x[i] * *(y[10] + i);
            d11 += x[i] * *(y[11] + i);
            d12 += x[i] * *(y[12] + i);
            d13 += x[i] * *(y[13] + i);
            d14 += x[i] * *(y[14] + i);
            d15 += x[i] * *(y[15] + i);
            d16 += x[i] * *(y[16] + i);
            d17 += x[i] * *(y[17] + i);
            d18 += x[i] * *(y[18] + i);
            d19 += x[i] * *(y[19] + i);
            d20 += x[i] * *(y[20] + i);
            d21 += x[i] * *(y[21] + i);
            d22 += x[i] * *(y[22] + i);
            d23 += x[i] * *(y[23] + i);
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
 * @brief Compute the inner product of a float16 vector with 2 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 2 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch2_f16f16(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t double_round = 16;

    if (likely(d >= double_round)) {
        float16x8_t x_0 = vld1q_f16(x);
        float16x8_t x_1 = vld1q_f16(x + 8);

        float16x8_t y0_0 = vld1q_f16(y);
        float16x8_t y0_1 = vld1q_f16(y + 8);
        float16x8_t y1_0 = vld1q_f16(y + d);
        float16x8_t y1_1 = vld1q_f16(y + d + 8);

        float16x8_t d0_0 = vmulq_f16(x_0, y0_0);
        float16x8_t d0_1 = vmulq_f16(x_1, y0_1);
        float16x8_t d1_0 = vmulq_f16(x_0, y1_0);
        float16x8_t d1_1 = vmulq_f16(x_1, y1_1);

        /* Main computation loop */
        for (i = double_round; i <= d - double_round; i += double_round) {
            x_0 = vld1q_f16(x + i);
            y0_0 = vld1q_f16(y + i);
            y1_0 = vld1q_f16(y + d + i);
            d0_0 = vfmaq_f16(d0_0, x_0, y0_0);
            d1_0 = vfmaq_f16(d1_0, x_0, y1_0);

            x_1 = vld1q_f16(x + i + 8);
            y0_1 = vld1q_f16(y + i + 8);
            y1_1 = vld1q_f16(y + d + i + 8);
            d0_1 = vfmaq_f16(d0_1, x_1, y0_1);
            d1_1 = vfmaq_f16(d1_1, x_1, y1_1);
        }

        /* Accumulate results */
        d0_0 = vaddq_f16(d0_0, d0_1);
        d1_0 = vaddq_f16(d1_0, d1_1);

        /* Horizontal addition to reduce the results */
        d0_0 = vpaddq_f16(d0_0, d1_0); /* 8 elements -> 4 elements */
        d0_0 = vpaddq_f16(d0_0, d0_0); /* 4 elements -> 2 elements */
        d0_0 = vpaddq_f16(d0_0, d0_0); /* 2 elements -> 1 element */

        /* Store the results */
        dis[0] = vgetq_lane_f16(d0_0, 0);
        dis[1] = vgetq_lane_f16(d0_0, 1);
    } else {
        /* Handle cases where d < double_round */
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    /* Handle remaining elements */
    for (; i < d; i++) {
        const float16_t tmp0 = x[i] * *(y + i);
        const float16_t tmp1 = x[i] * *(y + d + i);
        dis[0] += tmp0;
        dis[1] += tmp1;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of a float16 vector with 4 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 4 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch4_f16f16(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */

    if (likely(d >= single_round)) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);

        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_query);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_query);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_query);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_query);

        /* Main computation loop */
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);

            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_query);
        }

        /* Accumulate results */
        neon_res1 = vpaddq_f16(neon_res1, neon_res2);
        neon_res3 = vpaddq_f16(neon_res3, neon_res4);
        neon_res1 = vpaddq_f16(neon_res1, neon_res3);
        neon_res1 = vpaddq_f16(neon_res1, neon_res1);

        /* Store the results */
        vst1_f16(dis, vget_low_f16(neon_res1));
    } else {
        /* Handle cases where d < single_round */
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }

    /* Handle remaining elements */
    if (i < d) {
        float16_t d0 = x[i] * *(y + i);
        float16_t d1 = x[i] * *(y + d + i);
        float16_t d2 = x[i] * *(y + 2 * d + i);
        float16_t d3 = x[i] * *(y + 3 * d + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
        }

        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of a float16 vector with 8 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 8 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch8_f16f16(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
{
    size_t i;
    const size_t single_round = 8; /* 128 / 16 */

    if (d >= single_round) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        float16x8_t neon_base5 = vld1q_f16(y + 4 * d);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d);

        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_query);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_query);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_query);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_query);
        float16x8_t neon_res5 = vmulq_f16(neon_base5, neon_query);
        float16x8_t neon_res6 = vmulq_f16(neon_base6, neon_query);
        float16x8_t neon_res7 = vmulq_f16(neon_base7, neon_query);
        float16x8_t neon_res8 = vmulq_f16(neon_base8, neon_query);

        /* Main computation loop */
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_query);

            neon_base5 = vld1q_f16(y + 4 * d + i);
            neon_base6 = vld1q_f16(y + 5 * d + i);
            neon_base7 = vld1q_f16(y + 6 * d + i);
            neon_base8 = vld1q_f16(y + 7 * d + i);
            neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_query);
            neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_query);
            neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_query);
            neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_query);
        }

        /* Accumulate results */
        neon_res1 = vpaddq_f16(neon_res1, neon_res2);
        neon_res3 = vpaddq_f16(neon_res3, neon_res4);
        neon_res5 = vpaddq_f16(neon_res5, neon_res6);
        neon_res7 = vpaddq_f16(neon_res7, neon_res8);
        neon_res1 = vpaddq_f16(neon_res1, neon_res3);
        neon_res5 = vpaddq_f16(neon_res5, neon_res7);
        neon_res1 = vpaddq_f16(neon_res1, neon_res5);

        /* Store the results */
        vst1q_f16(dis, neon_res1);
    } else {
        /* Handle cases where d < single_round */
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }

    /* Handle remaining elements */
    if (i < d) {
        float16_t d0 = x[i] * *(y + i);
        float16_t d1 = x[i] * *(y + d + i);
        float16_t d2 = x[i] * *(y + 2 * d + i);
        float16_t d3 = x[i] * *(y + 3 * d + i);
        float16_t d4 = x[i] * *(y + 4 * d + i);
        float16_t d5 = x[i] * *(y + 5 * d + i);
        float16_t d6 = x[i] * *(y + 6 * d + i);
        float16_t d7 = x[i] * *(y + 7 * d + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
            d4 += x[i] * *(y + 4 * d + i);
            d5 += x[i] * *(y + 5 * d + i);
            d6 += x[i] * *(y + 6 * d + i);
            d7 += x[i] * *(y + 7 * d + i);
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
 * @brief Compute the inner product of a float16 vector with 16 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 16 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch16_f16f16(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */

    if (likely(d >= single_round)) {
        float16x8_t neon_query = vld1q_f16(x);

        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_query);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_query);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_query);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_query);

        float16x8_t neon_base5 = vld1q_f16(y + 4 * d);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d);
        float16x8_t neon_res5 = vmulq_f16(neon_base5, neon_query);
        float16x8_t neon_res6 = vmulq_f16(neon_base6, neon_query);
        float16x8_t neon_res7 = vmulq_f16(neon_base7, neon_query);
        float16x8_t neon_res8 = vmulq_f16(neon_base8, neon_query);

        neon_base1 = vld1q_f16(y + 8 * d);
        neon_base2 = vld1q_f16(y + 9 * d);
        neon_base3 = vld1q_f16(y + 10 * d);
        neon_base4 = vld1q_f16(y + 11 * d);
        float16x8_t neon_res9 = vmulq_f16(neon_base1, neon_query);
        float16x8_t neon_res10 = vmulq_f16(neon_base2, neon_query);
        float16x8_t neon_res11 = vmulq_f16(neon_base3, neon_query);
        float16x8_t neon_res12 = vmulq_f16(neon_base4, neon_query);

        neon_base5 = vld1q_f16(y + 12 * d);
        neon_base6 = vld1q_f16(y + 13 * d);
        neon_base7 = vld1q_f16(y + 14 * d);
        neon_base8 = vld1q_f16(y + 15 * d);
        float16x8_t neon_res13 = vmulq_f16(neon_base5, neon_query);
        float16x8_t neon_res14 = vmulq_f16(neon_base6, neon_query);
        float16x8_t neon_res15 = vmulq_f16(neon_base7, neon_query);
        float16x8_t neon_res16 = vmulq_f16(neon_base8, neon_query);

        /* Main computation loop */
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_query);

            neon_base5 = vld1q_f16(y + 4 * d + i);
            neon_base6 = vld1q_f16(y + 5 * d + i);
            neon_base7 = vld1q_f16(y + 6 * d + i);
            neon_base8 = vld1q_f16(y + 7 * d + i);
            neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_query);
            neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_query);
            neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_query);
            neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_query);

            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);
            neon_res13 = vfmaq_f16(neon_res13, neon_base5, neon_query);
            neon_res14 = vfmaq_f16(neon_res14, neon_base6, neon_query);
            neon_res15 = vfmaq_f16(neon_res15, neon_base7, neon_query);
            neon_res16 = vfmaq_f16(neon_res16, neon_base8, neon_query);
        }

        /* Accumulate results */
        neon_res1 = vpaddq_f16(neon_res1, neon_res2);
        neon_res3 = vpaddq_f16(neon_res3, neon_res4);
        neon_res5 = vpaddq_f16(neon_res5, neon_res6);
        neon_res7 = vpaddq_f16(neon_res7, neon_res8);
        neon_res9 = vpaddq_f16(neon_res9, neon_res10);
        neon_res11 = vpaddq_f16(neon_res11, neon_res12);
        neon_res13 = vpaddq_f16(neon_res13, neon_res14);
        neon_res15 = vpaddq_f16(neon_res15, neon_res16);

        neon_res1 = vpaddq_f16(neon_res1, neon_res3);
        neon_res5 = vpaddq_f16(neon_res5, neon_res7);
        neon_res9 = vpaddq_f16(neon_res9, neon_res11);
        neon_res13 = vpaddq_f16(neon_res13, neon_res15);

        neon_res1 = vpaddq_f16(neon_res1, neon_res5);
        neon_res9 = vpaddq_f16(neon_res9, neon_res13);

        /* Store the results */
        vst1q_f16(dis, neon_res1);
        vst1q_f16(dis + 8, neon_res9);
    } else {
        /* Handle cases where d < single_round */
        for (int i = 0; i < 16; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }

    /* Handle remaining elements */
    if (i < d) {
        float16_t d0 = x[i] * *(y + i);
        float16_t d1 = x[i] * *(y + d + i);
        float16_t d2 = x[i] * *(y + 2 * d + i);
        float16_t d3 = x[i] * *(y + 3 * d + i);
        float16_t d4 = x[i] * *(y + 4 * d + i);
        float16_t d5 = x[i] * *(y + 5 * d + i);
        float16_t d6 = x[i] * *(y + 6 * d + i);
        float16_t d7 = x[i] * *(y + 7 * d + i);
        float16_t d8 = x[i] * *(y + 8 * d + i);
        float16_t d9 = x[i] * *(y + 9 * d + i);
        float16_t d10 = x[i] * *(y + 10 * d + i);
        float16_t d11 = x[i] * *(y + 11 * d + i);
        float16_t d12 = x[i] * *(y + 12 * d + i);
        float16_t d13 = x[i] * *(y + 13 * d + i);
        float16_t d14 = x[i] * *(y + 14 * d + i);
        float16_t d15 = x[i] * *(y + 15 * d + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
            d4 += x[i] * *(y + 4 * d + i);
            d5 += x[i] * *(y + 5 * d + i);
            d6 += x[i] * *(y + 6 * d + i);
            d7 += x[i] * *(y + 7 * d + i);
            d8 += x[i] * *(y + 8 * d + i);
            d9 += x[i] * *(y + 9 * d + i);
            d10 += x[i] * *(y + 10 * d + i);
            d11 += x[i] * *(y + 11 * d + i);
            d12 += x[i] * *(y + 12 * d + i);
            d13 += x[i] * *(y + 13 * d + i);
            d14 += x[i] * *(y + 14 * d + i);
            d15 += x[i] * *(y + 15 * d + i);
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
 * @brief Compute the inner product of a float16 vector with multiple float16 vectors based on given indices
 * @param udis Pointer to the array storing the computed inner products
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of float16 vectors
 * @param ids Pointer to the array of indices specifying which y vectors to use
 * @param d The dimension of the vectors
 * @param ny The number of y vectors to process
 */
void krl_inner_product_by_idx_f16f16(
    uint16_t *udis, const uint16_t *x, const uint16_t *y, const int64_t *ids, size_t d, size_t ny)
{
    size_t i = 0;
    const float16_t *__restrict listy[24];
    float16_t *dis = (float16_t *)udis;

    /* Process in batches of 24 */
    for (; i + 24 <= ny; i += 24) {
        prefetch_L1(x); /* Prefetch x into L1 cache */
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        prefetch_Lx(listy[0]); /* Prefetch y vector 0 into Lx cache */
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        prefetch_Lx(listy[1]); /* Prefetch y vector 1 into Lx cache */
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        prefetch_Lx(listy[2]); /* Prefetch y vector 2 into Lx cache */
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        prefetch_Lx(listy[3]); /* Prefetch y vector 3 into Lx cache */
        listy[4] = (const float16_t *)(y + *(ids + i + 4) * d);
        prefetch_Lx(listy[4]); /* Prefetch y vector 4 into Lx cache */
        listy[5] = (const float16_t *)(y + *(ids + i + 5) * d);
        prefetch_Lx(listy[5]); /* Prefetch y vector 5 into Lx cache */
        listy[6] = (const float16_t *)(y + *(ids + i + 6) * d);
        prefetch_Lx(listy[6]); /* Prefetch y vector 6 into Lx cache */
        listy[7] = (const float16_t *)(y + *(ids + i + 7) * d);
        prefetch_Lx(listy[7]); /* Prefetch y vector 7 into Lx cache */
        listy[8] = (const float16_t *)(y + *(ids + i + 8) * d);
        prefetch_Lx(listy[8]); /* Prefetch y vector 8 into Lx cache */
        listy[9] = (const float16_t *)(y + *(ids + i + 9) * d);
        prefetch_Lx(listy[9]); /* Prefetch y vector 9 into Lx cache */
        listy[10] = (const float16_t *)(y + *(ids + i + 10) * d);
        prefetch_Lx(listy[10]); /* Prefetch y vector 10 into Lx cache */
        listy[11] = (const float16_t *)(y + *(ids + i + 11) * d);
        prefetch_Lx(listy[11]); /* Prefetch y vector 11 into Lx cache */
        listy[12] = (const float16_t *)(y + *(ids + i + 12) * d);
        prefetch_Lx(listy[12]); /* Prefetch y vector 12 into Lx cache */
        listy[13] = (const float16_t *)(y + *(ids + i + 13) * d);
        prefetch_Lx(listy[13]); /* Prefetch y vector 13 into Lx cache */
        listy[14] = (const float16_t *)(y + *(ids + i + 14) * d);
        prefetch_Lx(listy[14]); /* Prefetch y vector 14 into Lx cache */
        listy[15] = (const float16_t *)(y + *(ids + i + 15) * d);
        prefetch_Lx(listy[15]); /* Prefetch y vector 15 into Lx cache */
        listy[16] = (const float16_t *)(y + *(ids + i + 16) * d);
        prefetch_Lx(listy[16]); /* Prefetch y vector 16 into Lx cache */
        listy[17] = (const float16_t *)(y + *(ids + i + 17) * d);
        prefetch_Lx(listy[17]); /* Prefetch y vector 17 into Lx cache */
        listy[18] = (const float16_t *)(y + *(ids + i + 18) * d);
        prefetch_Lx(listy[18]); /* Prefetch y vector 18 into Lx cache */
        listy[19] = (const float16_t *)(y + *(ids + i + 19) * d);
        prefetch_Lx(listy[19]); /* Prefetch y vector 19 into Lx cache */
        listy[20] = (const float16_t *)(y + *(ids + i + 20) * d);
        prefetch_Lx(listy[20]); /* Prefetch y vector 20 into Lx cache */
        listy[21] = (const float16_t *)(y + *(ids + i + 21) * d);
        prefetch_Lx(listy[21]); /* Prefetch y vector 21 into Lx cache */
        listy[22] = (const float16_t *)(y + *(ids + i + 22) * d);
        prefetch_Lx(listy[22]); /* Prefetch y vector 22 into Lx cache */
        listy[23] = (const float16_t *)(y + *(ids + i + 23) * d);
        prefetch_Lx(listy[23]); /* Prefetch y vector 23 into Lx cache */

        krl_inner_product_idx_prefetch_batch24_f16f16((const float16_t *)x, listy, d, dis + i);
    }

    /* Handle remaining elements */
    if (i + 16 <= ny) {
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        listy[4] = (const float16_t *)(y + *(ids + i + 4) * d);
        listy[5] = (const float16_t *)(y + *(ids + i + 5) * d);
        listy[6] = (const float16_t *)(y + *(ids + i + 6) * d);
        listy[7] = (const float16_t *)(y + *(ids + i + 7) * d);
        listy[8] = (const float16_t *)(y + *(ids + i + 8) * d);
        listy[9] = (const float16_t *)(y + *(ids + i + 9) * d);
        listy[10] = (const float16_t *)(y + *(ids + i + 10) * d);
        listy[11] = (const float16_t *)(y + *(ids + i + 11) * d);
        listy[12] = (const float16_t *)(y + *(ids + i + 12) * d);
        listy[13] = (const float16_t *)(y + *(ids + i + 13) * d);
        listy[14] = (const float16_t *)(y + *(ids + i + 14) * d);
        listy[15] = (const float16_t *)(y + *(ids + i + 15) * d);

        krl_inner_product_idx_prefetch_batch16_f16f16((const float16_t *)x, listy, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        listy[4] = (const float16_t *)(y + *(ids + i + 4) * d);
        listy[5] = (const float16_t *)(y + *(ids + i + 5) * d);
        listy[6] = (const float16_t *)(y + *(ids + i + 6) * d);
        listy[7] = (const float16_t *)(y + *(ids + i + 7) * d);

        krl_inner_product_idx_batch8_f16f16((const float16_t *)x, listy, d, dis + i);
        i += 8;
    }

    /* Handle remaining elements */
    if (ny & 4) {
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);

        krl_inner_product_idx_batch4_f16f16((const float16_t *)x, listy, d, dis + i);
        i += 4;
    }

    /* Handle remaining elements */
    if (ny & 2) {
        const float16_t *y0 = (const float16_t *)(y + *(ids + i) * d);
        const float16_t *y1 = (const float16_t *)(y + *(ids + i + 1) * d);

        krl_inner_product_idx_batch2_f16f16((const float16_t *)x, y0, y1, d, dis + i);
        i += 2;
    }

    /* Handle remaining elements */
    if (ny & 1) {
        dis[i] = (float16_t)krl_inner_product_f16f16((const float16_t *)x, (const float16_t *)y + d * ids[i], d);
    }
}

/*
 * @brief Compute the inner product of a float16 vector with multiple float16 vectors in batches
 * @param udis Pointer to the array storing the computed inner products
 * @param ux Pointer to the input uint16_t vector
 * @param uy Pointer to the array of uint16_t vectors
 * @param d The dimension of the vectors
 * @param ny The number of y vectors to process
 */
void krl_inner_product_ny_f16f16(uint16_t *udis, const uint16_t *ux, const uint16_t *uy, size_t ny, size_t d)
{
    float16_t *dis = (float16_t *)udis;
    const float16_t *x = (const float16_t *)ux;
    const float16_t *y = (const float16_t *)uy;
    size_t i = 0;

    /* Process in batches of 16 */
    for (; i + 16 <= ny; i += 16) {
        krl_inner_product_batch16_f16f16((const float16_t *)x, y + d * i, d, dis + i);
    }

    /* Handle remaining elements */
    if (ny & 8) {
        krl_inner_product_batch8_f16f16((const float16_t *)x, y + d * i, d, dis + i);
        i += 8;
    }

    /* Handle remaining elements */
    if (ny & 4) {
        krl_inner_product_batch4_f16f16((const float16_t *)x, y + d * i, d, dis + i);
        i += 4;
    }

    /* Handle remaining elements */
    if (ny & 2) {
        krl_inner_product_batch2_f16f16((const float16_t *)x, y + d * i, d, dis + i);
        i += 2;
    }

    /* Handle remaining elements */
    if (ny & 1) {
        dis[i] = (float16_t)krl_inner_product_f16f16((const float16_t *)x, (const float16_t *)y + d * i, d);
    }
}
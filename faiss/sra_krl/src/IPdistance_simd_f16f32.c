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
 * @brief Compute the inner product of two float16 vectors and return the result as float32
 * @param u16_x Pointer to the first float16 vector
 * @param u16_y Pointer to the second float16 vector (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @return The computed inner product as float32
 */
KRL_IMPRECISE_FUNCTION_BEGIN
float krl_inner_product_f16f32(const uint16_t *u16_x, const uint16_t *__restrict u16_y, const size_t d)
{
    const float16_t *x = (const float16_t *)u16_x;
    const float16_t *y = (const float16_t *)u16_y;
    size_t i;
    float res;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 32;

    /* Initialize result registers */
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);

    /* Main computation loop with double rounds */
    for (i = 0; i + double_round <= d; i += double_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);
        float16x8_t x8_2 = vld1q_f16(x + i + 16);
        float16x8_t x8_3 = vld1q_f16(x + i + 24);

        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + i + 8);
        float16x8_t y8_2 = vld1q_f16(y + i + 16);
        float16x8_t y8_3 = vld1q_f16(y + i + 24);

        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_low_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_low_f16(res3, x8_2, y8_2);
        res4 = vfmlalq_low_f16(res4, x8_3, y8_3);

        res1 = vfmlalq_high_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_high_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_high_f16(res3, x8_2, y8_2);
        res4 = vfmlalq_high_f16(res4, x8_3, y8_3);
    }

    /* Handle remaining elements with single rounds */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);

        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res3 = vfmlalq_high_f16(res3, x8_0, y8_0);
    }

    /* Accumulate results */
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    res1 = vaddq_f32(res1, res3);
    res = vaddvq_f32(res1);

    /* Handle remaining elements */
    for (; i < d; i++) {
        res += (float)(x[i] * y[i]);
    }

    return res;
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of a float16 vector with two other float16 vectors and store the results in a float
 * array
 * @param x Pointer to the input float16 vector
 * @param y0 Pointer to the first float16 vector (with restrict qualifier for better optimization)
 * @param y1 Pointer to the second float16 vector (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products (size must be at least 2)
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch2_f16f32(
    const float16_t *x, const float16_t *__restrict y0, const float16_t *__restrict y1, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;  /* Number of elements processed per single round */
    constexpr size_t double_round = 16; /* Number of elements processed per double round */

    /* Initialize result registers */
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);

    /* Main computation loop with double rounds */
    for (i = 0; i + double_round <= d; i += double_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);

        float16x8_t y8_0 = vld1q_f16(y0 + i);
        float16x8_t y8_1 = vld1q_f16(y0 + i + 8);
        float16x8_t y8_2 = vld1q_f16(y1 + i);
        float16x8_t y8_3 = vld1q_f16(y1 + i + 8);

        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_low_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_low_f16(res3, x8_0, y8_2);
        res4 = vfmlalq_low_f16(res4, x8_1, y8_3);

        res1 = vfmlalq_high_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_high_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_high_f16(res3, x8_0, y8_2);
        res4 = vfmlalq_high_f16(res4, x8_1, y8_3);
    }

    /* Handle remaining elements with single rounds */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y0 + i);
        float16x8_t y8_1 = vld1q_f16(y1 + i);

        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res3 = vfmlalq_low_f16(res3, x8_0, y8_1);
        res2 = vfmlalq_high_f16(res2, x8_0, y8_0);
        res4 = vfmlalq_high_f16(res4, x8_0, y8_1);
    }

    /* Accumulate results */
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    dis[0] = vaddvq_f32(res1);
    dis[1] = vaddvq_f32(res3);

    /* Handle remaining elements */
    for (; i < d; i++) {
        dis[0] += (float)(x[i] * y0[i]);
        dis[1] += (float)(x[i] * y1[i]);
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of a float16 vector with four other float16 vectors and store the results in a float
 * array
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of four float16 vectors (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products (size must be at least 4)
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch4_f16f32(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* Number of elements processed per single round */

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);

    /* Main computation loop */
    for (i = 0; i + single_round <= d; i += single_round) {
        /* Load query and base vectors */
        float16x8_t neon_query = vld1q_f16(x + i);
        float16x8_t neon_base1 = vld1q_f16(y[0] + i);
        float16x8_t neon_base2 = vld1q_f16(y[1] + i);
        float16x8_t neon_base3 = vld1q_f16(y[2] + i);
        float16x8_t neon_base4 = vld1q_f16(y[3] + i);

        /* Perform vectorized Fused Multiply-Add operations */
        neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
    }

    /* Store accumulated results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (float)(x[i] * *(y[0] + i));
        float d1 = (float)(x[i] * *(y[1] + i));
        float d2 = (float)(x[i] * *(y[2] + i));
        float d3 = (float)(x[i] * *(y[3] + i));
        for (i++; i < d; ++i) {
            d0 += (float)(x[i] * *(y[0] + i));
            d1 += (float)(x[i] * *(y[1] + i));
            d2 += (float)(x[i] * *(y[2] + i));
            d3 += (float)(x[i] * *(y[3] + i));
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of a float16 vector with eight other float16 vectors and store the results in a
 * float array
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of eight float16 vectors (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products (size must be at least 8)
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_batch8_f16f32(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* Number of elements processed per single round */

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);

    /* Main computation loop */
    for (i = 0; i + single_round <= d; i += single_round) {
        /* Load query and base vectors */
        float16x8_t neon_query = vld1q_f16(x + i);
        float16x8_t neon_base1 = vld1q_f16(y[0] + i);
        float16x8_t neon_base2 = vld1q_f16(y[1] + i);
        float16x8_t neon_base3 = vld1q_f16(y[2] + i);
        float16x8_t neon_base4 = vld1q_f16(y[3] + i);
        float16x8_t neon_base5 = vld1q_f16(y[4] + i);
        float16x8_t neon_base6 = vld1q_f16(y[5] + i);
        float16x8_t neon_base7 = vld1q_f16(y[6] + i);
        float16x8_t neon_base8 = vld1q_f16(y[7] + i);

        /* Perform vectorized Fused Multiply-Add operations */
        neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);
    }

    /* Store accumulated results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);
    dis[4] = vaddvq_f32(neon_res5);
    dis[5] = vaddvq_f32(neon_res6);
    dis[6] = vaddvq_f32(neon_res7);
    dis[7] = vaddvq_f32(neon_res8);

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (float)(x[i] * *(y[0] + i));
        float d1 = (float)(x[i] * *(y[1] + i));
        float d2 = (float)(x[i] * *(y[2] + i));
        float d3 = (float)(x[i] * *(y[3] + i));
        float d4 = (float)(x[i] * *(y[4] + i));
        float d5 = (float)(x[i] * *(y[5] + i));
        float d6 = (float)(x[i] * *(y[6] + i));
        float d7 = (float)(x[i] * *(y[7] + i));
        for (i++; i < d; ++i) {
            d0 += (float)(x[i] * *(y[0] + i));
            d1 += (float)(x[i] * *(y[1] + i));
            d2 += (float)(x[i] * *(y[2] + i));
            d3 += (float)(x[i] * *(y[3] + i));
            d4 += (float)(x[i] * *(y[4] + i));
            d5 += (float)(x[i] * *(y[5] + i));
            d6 += (float)(x[i] * *(y[6] + i));
            d7 += (float)(x[i] * *(y[7] + i));
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
 * @brief Compute the inner product of a float16 vector with sixteen other float16 vectors and store the results in a
 * float array
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of sixteen float16 vectors (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed inner products (size must be at least 16)
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_prefetch_batch16_f16f32(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* Number of elements processed per single round */
    constexpr size_t multi_round = 32; /* Number of elements processed per multi-round (4 * single_round) */

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);
    float32x4_t neon_res9 = vdupq_n_f32(0.0f);
    float32x4_t neon_res10 = vdupq_n_f32(0.0f);
    float32x4_t neon_res11 = vdupq_n_f32(0.0f);
    float32x4_t neon_res12 = vdupq_n_f32(0.0f);
    float32x4_t neon_res13 = vdupq_n_f32(0.0f);
    float32x4_t neon_res14 = vdupq_n_f32(0.0f);
    float32x4_t neon_res15 = vdupq_n_f32(0.0f);
    float32x4_t neon_res16 = vdupq_n_f32(0.0f);

    /* Main computation loop with multi-round processing */
    if (d >= multi_round) {
        for (i = 0; i < d - multi_round; i += multi_round) {
            /* Prefetch data for better cache utilization */
            prefetch_L1(x + i + multi_round);
            for (size_t k = 0; k < 16; ++k) {
                prefetch_Lx(y[k] + i + multi_round);
            }

            /* Process data in single_round chunks */
            for (size_t j = 0; j < multi_round; j += single_round) {
                /* Load query vector */
                const float16x8_t neon_query = vld1q_f16(x + i + j);

                /* Load base vectors for first 8 results */
                float16x8_t neon_base1 = vld1q_f16(y[0] + i + j);
                float16x8_t neon_base2 = vld1q_f16(y[1] + i + j);
                float16x8_t neon_base3 = vld1q_f16(y[2] + i + j);
                float16x8_t neon_base4 = vld1q_f16(y[3] + i + j);
                float16x8_t neon_base5 = vld1q_f16(y[4] + i + j);
                float16x8_t neon_base6 = vld1q_f16(y[5] + i + j);
                float16x8_t neon_base7 = vld1q_f16(y[6] + i + j);
                float16x8_t neon_base8 = vld1q_f16(y[7] + i + j);

                /* Perform vectorized Fused Multiply-Add operations for first 8 results */
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

                neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

                /* Load base vectors for next 8 results */
                neon_base1 = vld1q_f16(y[8] + i + j);
                neon_base2 = vld1q_f16(y[9] + i + j);
                neon_base3 = vld1q_f16(y[10] + i + j);
                neon_base4 = vld1q_f16(y[11] + i + j);
                neon_base5 = vld1q_f16(y[12] + i + j);
                neon_base6 = vld1q_f16(y[13] + i + j);
                neon_base7 = vld1q_f16(y[14] + i + j);
                neon_base8 = vld1q_f16(y[15] + i + j);

                /* Perform vectorized Fused Multiply-Add operations for next 8 results */
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

                neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
            }
        }

        /* Handle remaining elements with single_round processing */
        for (; i + single_round <= d; i += single_round) {
            /* Load query vector */
            const float16x8_t neon_query = vld1q_f16(x + i);

            /* Load base vectors for first 8 results */
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            float16x8_t neon_base5 = vld1q_f16(y[4] + i);
            float16x8_t neon_base6 = vld1q_f16(y[5] + i);
            float16x8_t neon_base7 = vld1q_f16(y[6] + i);
            float16x8_t neon_base8 = vld1q_f16(y[7] + i);

            /* Perform vectorized Fused Multiply-Add operations for first 8 results */
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

            /* Load base vectors for next 8 results */
            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base5 = vld1q_f16(y[12] + i);
            neon_base6 = vld1q_f16(y[13] + i);
            neon_base7 = vld1q_f16(y[14] + i);
            neon_base8 = vld1q_f16(y[15] + i);

            /* Perform vectorized Fused Multiply-Add operations for next 8 results */
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
        }
    } else {
        /* Handle cases where d < multi_round */
        for (i = 0; i + single_round <= d; i += single_round) {
            /* Load query vector */
            const float16x8_t neon_query = vld1q_f16(x + i);

            /* Load base vectors for first 8 results */
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            float16x8_t neon_base5 = vld1q_f16(y[4] + i);
            float16x8_t neon_base6 = vld1q_f16(y[5] + i);
            float16x8_t neon_base7 = vld1q_f16(y[6] + i);
            float16x8_t neon_base8 = vld1q_f16(y[7] + i);

            /* Perform vectorized Fused Multiply-Add operations for first 8 results */
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

            /* Load base vectors for next 8 results */
            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base5 = vld1q_f16(y[12] + i);
            neon_base6 = vld1q_f16(y[13] + i);
            neon_base7 = vld1q_f16(y[14] + i);
            neon_base8 = vld1q_f16(y[15] + i);

            /* Perform vectorized Fused Multiply-Add operations for next 8 results */
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
        }
    }

    /* Store accumulated results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);
    dis[4] = vaddvq_f32(neon_res5);
    dis[5] = vaddvq_f32(neon_res6);
    dis[6] = vaddvq_f32(neon_res7);
    dis[7] = vaddvq_f32(neon_res8);
    dis[8] = vaddvq_f32(neon_res9);
    dis[9] = vaddvq_f32(neon_res10);
    dis[10] = vaddvq_f32(neon_res11);
    dis[11] = vaddvq_f32(neon_res12);
    dis[12] = vaddvq_f32(neon_res13);
    dis[13] = vaddvq_f32(neon_res14);
    dis[14] = vaddvq_f32(neon_res15);
    dis[15] = vaddvq_f32(neon_res16);

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (float)(x[i] * *(y[0] + i));
        float d1 = (float)(x[i] * *(y[1] + i));
        float d2 = (float)(x[i] * *(y[2] + i));
        float d3 = (float)(x[i] * *(y[3] + i));
        float d4 = (float)(x[i] * *(y[4] + i));
        float d5 = (float)(x[i] * *(y[5] + i));
        float d6 = (float)(x[i] * *(y[6] + i));
        float d7 = (float)(x[i] * *(y[7] + i));
        float d8 = (float)(x[i] * *(y[8] + i));
        float d9 = (float)(x[i] * *(y[9] + i));
        float d10 = (float)(x[i] * *(y[10] + i));
        float d11 = (float)(x[i] * *(y[11] + i));
        float d12 = (float)(x[i] * *(y[12] + i));
        float d13 = (float)(x[i] * *(y[13] + i));
        float d14 = (float)(x[i] * *(y[14] + i));
        float d15 = (float)(x[i] * *(y[15] + i));

        for (i++; i < d; ++i) {
            d0 += (float)(x[i] * *(y[0] + i));
            d1 += (float)(x[i] * *(y[1] + i));
            d2 += (float)(x[i] * *(y[2] + i));
            d3 += (float)(x[i] * *(y[3] + i));
            d4 += (float)(x[i] * *(y[4] + i));
            d5 += (float)(x[i] * *(y[5] + i));
            d6 += (float)(x[i] * *(y[6] + i));
            d7 += (float)(x[i] * *(y[7] + i));
            d8 += (float)(x[i] * *(y[8] + i));
            d9 += (float)(x[i] * *(y[9] + i));
            d10 += (float)(x[i] * *(y[10] + i));
            d11 += (float)(x[i] * *(y[11] + i));
            d12 += (float)(x[i] * *(y[12] + i));
            d13 += (float)(x[i] * *(y[13] + i));
            d14 += (float)(x[i] * *(y[14] + i));
            d15 += (float)(x[i] * *(y[15] + i));
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
 * @brief Computes the inner product of multiple vectors using NEON instructions with prefetching.
 *        This function handles 24 vectors in batches, using float16 precision for input and float32 for output.
 * @param x Pointer to the input vector (float16).
 * @param y Array of pointers to the base vectors (float16).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array storing the inner products (float32).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_idx_prefetch_batch24_f16f32(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* Number of elements processed per single_round (128-bit NEON register) */
    constexpr size_t multi_round = 32; /* Number of elements processed per multi_round (4 * single_round) */

    /* Initialize NEON registers for accumulating results */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);
    float32x4_t neon_res9 = vdupq_n_f32(0.0f);
    float32x4_t neon_res10 = vdupq_n_f32(0.0f);
    float32x4_t neon_res11 = vdupq_n_f32(0.0f);
    float32x4_t neon_res12 = vdupq_n_f32(0.0f);
    float32x4_t neon_res13 = vdupq_n_f32(0.0f);
    float32x4_t neon_res14 = vdupq_n_f32(0.0f);
    float32x4_t neon_res15 = vdupq_n_f32(0.0f);
    float32x4_t neon_res16 = vdupq_n_f32(0.0f);
    float32x4_t neon_res17 = vdupq_n_f32(0.0f);
    float32x4_t neon_res18 = vdupq_n_f32(0.0f);
    float32x4_t neon_res19 = vdupq_n_f32(0.0f);
    float32x4_t neon_res20 = vdupq_n_f32(0.0f);
    float32x4_t neon_res21 = vdupq_n_f32(0.0f);
    float32x4_t neon_res22 = vdupq_n_f32(0.0f);
    float32x4_t neon_res23 = vdupq_n_f32(0.0f);
    float32x4_t neon_res24 = vdupq_n_f32(0.0f);

    if (d >= multi_round) {
        for (i = 0; i < d - multi_round; i += multi_round) {
            const size_t next_i = i + multi_round;
            /* Prefetch data for next iteration to improve cache utilization */
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

            for (size_t j = i; j < next_i; j += single_round) {
                /* Load query vector */
                const float16x8_t neon_query = vld1q_f16(x + j);

                /* Process base vectors 0-3 */
                float16x8_t neon_base1 = vld1q_f16(y[0] + j);
                float16x8_t neon_base2 = vld1q_f16(y[1] + j);
                float16x8_t neon_base3 = vld1q_f16(y[2] + j);
                float16x8_t neon_base4 = vld1q_f16(y[3] + j);
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_query);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_query);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_query);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_query);
                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_query);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_query);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_query);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_query);

                /* Process base vectors 4-7 */
                neon_base1 = vld1q_f16(y[4] + j);
                neon_base2 = vld1q_f16(y[5] + j);
                neon_base3 = vld1q_f16(y[6] + j);
                neon_base4 = vld1q_f16(y[7] + j);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_query);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_query);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_query);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_query);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_query);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_query);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_query);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_query);

                /* Process base vectors 8-11 */
                neon_base1 = vld1q_f16(y[8] + j);
                neon_base2 = vld1q_f16(y[9] + j);
                neon_base3 = vld1q_f16(y[10] + j);
                neon_base4 = vld1q_f16(y[11] + j);
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_query);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_query);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_query);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_query);
                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_query);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_query);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_query);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_query);

                /* Process base vectors 12-15 */
                neon_base1 = vld1q_f16(y[12] + j);
                neon_base2 = vld1q_f16(y[13] + j);
                neon_base3 = vld1q_f16(y[14] + j);
                neon_base4 = vld1q_f16(y[15] + j);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_query);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_query);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_query);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_query);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_query);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_query);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_query);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_query);

                /* Process base vectors 16-19 */
                neon_base1 = vld1q_f16(y[16] + j);
                neon_base2 = vld1q_f16(y[17] + j);
                neon_base3 = vld1q_f16(y[18] + j);
                neon_base4 = vld1q_f16(y[19] + j);
                neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_query);
                neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_query);
                neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_query);
                neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_query);
                neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_query);
                neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_query);
                neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_query);
                neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_query);

                /* Process base vectors 20-23 */
                neon_base1 = vld1q_f16(y[20] + j);
                neon_base2 = vld1q_f16(y[21] + j);
                neon_base3 = vld1q_f16(y[22] + j);
                neon_base4 = vld1q_f16(y[23] + j);
                neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_query);
                neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_query);
                neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_query);
                neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_query);
                neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_query);
                neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_query);
                neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_query);
                neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_query);
            }
        }

        for (; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);

            /* Process base vectors 0-3 */
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_query);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_query);

            /* Process base vectors 4-7 */
            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_query);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_query);

            /* Process base vectors 8-11 */
            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_query);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_query);

            /* Process base vectors 12-15 */
            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_query);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_query);

            /* Process base vectors 16-19 */
            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_query);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_query);

            /* Process base vectors 20-23 */
            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_query);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_query);
        }

        /* Store the accumulated results into the output array */
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        dis[8] = vaddvq_f32(neon_res9);
        dis[9] = vaddvq_f32(neon_res10);
        dis[10] = vaddvq_f32(neon_res11);
        dis[11] = vaddvq_f32(neon_res12);
        dis[12] = vaddvq_f32(neon_res13);
        dis[13] = vaddvq_f32(neon_res14);
        dis[14] = vaddvq_f32(neon_res15);
        dis[15] = vaddvq_f32(neon_res16);
        dis[16] = vaddvq_f32(neon_res17);
        dis[17] = vaddvq_f32(neon_res18);
        dis[18] = vaddvq_f32(neon_res19);
        dis[19] = vaddvq_f32(neon_res20);
        dis[20] = vaddvq_f32(neon_res21);
        dis[21] = vaddvq_f32(neon_res22);
        dis[22] = vaddvq_f32(neon_res23);
        dis[23] = vaddvq_f32(neon_res24);
    } else if (d >= single_round) {
        for (i = 0; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);

            /* Process base vectors 0-3 */
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_query);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_query);

            /* Process base vectors 4-7 */
            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_query);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_query);

            /* Process base vectors 8-11 */
            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_query);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_query);

            /* Process base vectors 12-15 */
            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_query);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_query);

            /* Process base vectors 16-19 */
            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_query);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_query);

            /* Process base vectors 20-23 */
            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_query);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_query);
        }

        /* Store the accumulated results into the output array */
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        dis[8] = vaddvq_f32(neon_res9);
        dis[9] = vaddvq_f32(neon_res10);
        dis[10] = vaddvq_f32(neon_res11);
        dis[11] = vaddvq_f32(neon_res12);
        dis[12] = vaddvq_f32(neon_res13);
        dis[13] = vaddvq_f32(neon_res14);
        dis[14] = vaddvq_f32(neon_res15);
        dis[15] = vaddvq_f32(neon_res16);
        dis[16] = vaddvq_f32(neon_res17);
        dis[17] = vaddvq_f32(neon_res18);
        dis[18] = vaddvq_f32(neon_res19);
        dis[19] = vaddvq_f32(neon_res20);
        dis[20] = vaddvq_f32(neon_res21);
        dis[21] = vaddvq_f32(neon_res22);
        dis[22] = vaddvq_f32(neon_res23);
        dis[23] = vaddvq_f32(neon_res24);
    } else {
        for (int i = 0; i < 24; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
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
        float d16 = x[i] * *(y[16] + i);
        float d17 = x[i] * *(y[17] + i);
        float d18 = x[i] * *(y[18] + i);
        float d19 = x[i] * *(y[19] + i);
        float d20 = x[i] * *(y[20] + i);
        float d21 = x[i] * *(y[21] + i);
        float d22 = x[i] * *(y[22] + i);
        float d23 = x[i] * *(y[23] + i);
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
 * @brief Compute the inner product of two batches of half-precision floating-point vectors.
 * @param x Pointer to the first input vector (half-precision float).
 * @param y Pointer to the second input vector (half-precision float), which is split into two parts.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch2_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;  /* Number of elements processed in a single-round loop */
    constexpr size_t double_round = 16; /* Number of elements processed in a double-round loop */

    /* Initialize result registers */
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);

    /* Process data in double-round chunks */
    for (i = 0; i + double_round <= d; i += double_round) {
        /* Load data for the first part */
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);

        /* Load data for the second part */
        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + i + 8);
        float16x8_t y8_2 = vld1q_f16(y + d + i);
        float16x8_t y8_3 = vld1q_f16(y + d + i + 8);

        /* Perform vectorized multiplication and accumulation */
        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_low_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_low_f16(res3, x8_0, y8_2);
        res4 = vfmlalq_low_f16(res4, x8_1, y8_3);

        res1 = vfmlalq_high_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_high_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_high_f16(res3, x8_0, y8_2);
        res4 = vfmlalq_high_f16(res4, x8_1, y8_3);
    }

    /* Process remaining data in single-round chunks */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + d + i);

        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res3 = vfmlalq_low_f16(res3, x8_0, y8_1);
        res2 = vfmlalq_high_f16(res2, x8_0, y8_0);
        res4 = vfmlalq_high_f16(res4, x8_0, y8_1);
    }

    /* Combine results from the four registers */
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);

    /* Store the final results */
    dis[0] = vaddvq_f32(res1);
    dis[1] = vaddvq_f32(res3);

    /* Handle any remaining elements */
    for (; i < d; i++) {
        dis[0] += (float)(x[i] * y[i]);
        dis[1] += (float)(x[i] * y[i + d]);
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of four batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains four batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch4_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    const size_t single_round = 8; /* Number of elements processed in a single-round loop */

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);

    /* Process data in single-round chunks */
    for (i = 0; i + single_round <= d; i += single_round) {
        /* Load data for the query and four bases */
        float16x8_t neon_query = vld1q_f16(x + i);
        float16x8_t neon_base1 = vld1q_f16(y + i);
        float16x8_t neon_base2 = vld1q_f16(y + d + i);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);

        /* Perform vectorized multiplication and accumulation */
        neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
    }

    /* Store the final results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);

    /* Handle any remaining elements */
    if (i < d) {
        /* Initialize partial sums */
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);

        /* Accumulate remaining elements */
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
        }

        /* Add partial sums to the final results */
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of eight batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains eight batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch8_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* Number of elements processed in a single-round loop */

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);

    /* Process data in single-round chunks */
    for (i = 0; i + single_round <= d; i += single_round) {
        /* Load data for the query and eight bases */
        float16x8_t neon_query = vld1q_f16(x + i);
        float16x8_t neon_base1 = vld1q_f16(y + i);
        float16x8_t neon_base2 = vld1q_f16(y + d + i);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
        float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

        /* Perform vectorized multiplication and accumulation */
        neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);
    }

    /* Store the final results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);
    dis[4] = vaddvq_f32(neon_res5);
    dis[5] = vaddvq_f32(neon_res6);
    dis[6] = vaddvq_f32(neon_res7);
    dis[7] = vaddvq_f32(neon_res8);

    /* Handle any remaining elements */
    if (i < d) {
        /* Initialize partial sums */
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);
        float d4 = x[i] * *(y + 4 * d + i);
        float d5 = x[i] * *(y + 5 * d + i);
        float d6 = x[i] * *(y + 6 * d + i);
        float d7 = x[i] * *(y + 7 * d + i);

        /* Accumulate remaining elements */
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

        /* Add partial sums to the final results */
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
 * @brief Compute the inner product of sixteen batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains sixteen batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_prefetch_batch16_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t multi_round = 32;

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);
    float32x4_t neon_res9 = vdupq_n_f32(0.0f);
    float32x4_t neon_res10 = vdupq_n_f32(0.0f);
    float32x4_t neon_res11 = vdupq_n_f32(0.0f);
    float32x4_t neon_res12 = vdupq_n_f32(0.0f);
    float32x4_t neon_res13 = vdupq_n_f32(0.0f);
    float32x4_t neon_res14 = vdupq_n_f32(0.0f);
    float32x4_t neon_res15 = vdupq_n_f32(0.0f);
    float32x4_t neon_res16 = vdupq_n_f32(0.0f);

    if (d >= multi_round) {
        /* Process data in multi-round chunks with prefetching */
        for (i = 0; i < d - multi_round; i += multi_round) {
            /* Prefetch data for the next multi_round */
            prefetch_L1(x + i + multi_round);
            prefetch_Lx(y + i + multi_round);
            prefetch_Lx(y + d + i + multi_round);
            prefetch_Lx(y + 2 * d + i + multi_round);
            prefetch_Lx(y + 3 * d + i + multi_round);
            prefetch_Lx(y + 4 * d + i + multi_round);
            prefetch_Lx(y + 5 * d + i + multi_round);
            prefetch_Lx(y + 6 * d + i + multi_round);
            prefetch_Lx(y + 7 * d + i + multi_round);
            prefetch_Lx(y + 8 * d + i + multi_round);
            prefetch_Lx(y + 9 * d + i + multi_round);
            prefetch_Lx(y + 10 * d + i + multi_round);
            prefetch_Lx(y + 11 * d + i + multi_round);
            prefetch_Lx(y + 12 * d + i + multi_round);
            prefetch_Lx(y + 13 * d + i + multi_round);
            prefetch_Lx(y + 14 * d + i + multi_round);
            prefetch_Lx(y + 15 * d + i + multi_round);

            /* Process data in single_round chunks */
            for (size_t j = 0; j < multi_round; j += single_round) {
                /* Load data for the query and eight bases */
                const float16x8_t neon_query = vld1q_f16(x + i + j);
                float16x8_t neon_base1 = vld1q_f16(y + i + j);
                float16x8_t neon_base2 = vld1q_f16(y + d + i + j);
                float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i + j);
                float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i + j);
                float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i + j);
                float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i + j);
                float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i + j);
                float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i + j);

                /* Perform vectorized multiplication and accumulation */
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

                neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

                /* Load data for the next eight bases */
                neon_base1 = vld1q_f16(y + 8 * d + i + j);
                neon_base2 = vld1q_f16(y + 9 * d + i + j);
                neon_base3 = vld1q_f16(y + 10 * d + i + j);
                neon_base4 = vld1q_f16(y + 11 * d + i + j);
                neon_base5 = vld1q_f16(y + 12 * d + i + j);
                neon_base6 = vld1q_f16(y + 13 * d + i + j);
                neon_base7 = vld1q_f16(y + 14 * d + i + j);
                neon_base8 = vld1q_f16(y + 15 * d + i + j);

                /* Perform vectorized multiplication and accumulation */
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

                neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
            }
        }

        /* Process remaining data in single_round chunks */
        for (; i + single_round <= d; i += single_round) {
            /* Load data for the query and eight bases */
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
            float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
            float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
            float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

            /* Perform vectorized multiplication and accumulation */
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

            /* Load data for the next eight bases */
            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);

            /* Perform vectorized multiplication and accumulation */
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
        }
    } else {
        /* Process data in single_round chunks without prefetching */
        for (i = 0; i + single_round <= d; i += single_round) {
            /* Load data for the query and eight bases */
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
            float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
            float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
            float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

            /* Perform vectorized multiplication and accumulation */
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

            /* Load data for the next eight bases */
            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);

            /* Perform vectorized multiplication and accumulation */
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
        }
    }

    /* Store the final results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);
    dis[4] = vaddvq_f32(neon_res5);
    dis[5] = vaddvq_f32(neon_res6);
    dis[6] = vaddvq_f32(neon_res7);
    dis[7] = vaddvq_f32(neon_res8);
    dis[8] = vaddvq_f32(neon_res9);
    dis[9] = vaddvq_f32(neon_res10);
    dis[10] = vaddvq_f32(neon_res11);
    dis[11] = vaddvq_f32(neon_res12);
    dis[12] = vaddvq_f32(neon_res13);
    dis[13] = vaddvq_f32(neon_res14);
    dis[14] = vaddvq_f32(neon_res15);
    dis[15] = vaddvq_f32(neon_res16);

    /* Handle any remaining elements */
    if (i < d) {
        /* Initialize partial sums */
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);
        float d4 = x[i] * *(y + 4 * d + i);
        float d5 = x[i] * *(y + 5 * d + i);
        float d6 = x[i] * *(y + 6 * d + i);
        float d7 = x[i] * *(y + 7 * d + i);
        float d8 = x[i] * *(y + 8 * d + i);
        float d9 = x[i] * *(y + 9 * d + i);
        float d10 = x[i] * *(y + 10 * d + i);
        float d11 = x[i] * *(y + 11 * d + i);
        float d12 = x[i] * *(y + 12 * d + i);
        float d13 = x[i] * *(y + 13 * d + i);
        float d14 = x[i] * *(y + 14 * d + i);
        float d15 = x[i] * *(y + 15 * d + i);

        /* Accumulate remaining elements */
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

        /* Add partial sums to the final results */
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
 * @brief Compute the inner product of sixteen batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains sixteen batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_prefetch_batch24_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* Number of elements processed per round (128-bit NEON vector) */
    constexpr size_t multi_round = 32; /* Number of elements processed per multi-round (4 NEON vectors) */

    /* Initialize NEON registers for accumulating results */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);
    float32x4_t neon_res9 = vdupq_n_f32(0.0f);
    float32x4_t neon_res10 = vdupq_n_f32(0.0f);
    float32x4_t neon_res11 = vdupq_n_f32(0.0f);
    float32x4_t neon_res12 = vdupq_n_f32(0.0f);
    float32x4_t neon_res13 = vdupq_n_f32(0.0f);
    float32x4_t neon_res14 = vdupq_n_f32(0.0f);
    float32x4_t neon_res15 = vdupq_n_f32(0.0f);
    float32x4_t neon_res16 = vdupq_n_f32(0.0f);
    float32x4_t neon_res17 = vdupq_n_f32(0.0f);
    float32x4_t neon_res18 = vdupq_n_f32(0.0f);
    float32x4_t neon_res19 = vdupq_n_f32(0.0f);
    float32x4_t neon_res20 = vdupq_n_f32(0.0f);
    float32x4_t neon_res21 = vdupq_n_f32(0.0f);
    float32x4_t neon_res22 = vdupq_n_f32(0.0f);
    float32x4_t neon_res23 = vdupq_n_f32(0.0f);
    float32x4_t neon_res24 = vdupq_n_f32(0.0f);

    if (d >= multi_round) {
        for (i = 0; i < d - multi_round; i += multi_round) {
            const size_t next_i = i + multi_round;
            /* Prefetch data to improve cache performance */
            prefetch_L1(x + next_i);
            prefetch_Lx(y + next_i);
            prefetch_Lx(y + d + next_i);
            prefetch_Lx(y + 2 * d + next_i);
            prefetch_Lx(y + 3 * d + next_i);
            prefetch_Lx(y + 4 * d + next_i);
            prefetch_Lx(y + 5 * d + next_i);
            prefetch_Lx(y + 6 * d + next_i);
            prefetch_Lx(y + 7 * d + next_i);
            prefetch_Lx(y + 8 * d + next_i);
            prefetch_Lx(y + 9 * d + next_i);
            prefetch_Lx(y + 10 * d + next_i);
            prefetch_Lx(y + 11 * d + next_i);
            prefetch_Lx(y + 12 * d + next_i);
            prefetch_Lx(y + 13 * d + next_i);
            prefetch_Lx(y + 14 * d + next_i);
            prefetch_Lx(y + 15 * d + next_i);
            prefetch_Lx(y + 16 * d + next_i);
            prefetch_Lx(y + 17 * d + next_i);
            prefetch_Lx(y + 18 * d + next_i);
            prefetch_Lx(y + 19 * d + next_i);
            prefetch_Lx(y + 20 * d + next_i);
            prefetch_Lx(y + 21 * d + next_i);
            prefetch_Lx(y + 22 * d + next_i);
            prefetch_Lx(y + 23 * d + next_i);

            for (size_t j = i; j < next_i; j += single_round) {
                /* Load query vector */
                const float16x8_t neon_query = vld1q_f16(x + j);
                /* Load base vectors and compute inner products */
                float16x8_t neon_base1 = vld1q_f16(y + j);
                float16x8_t neon_base2 = vld1q_f16(y + d + j);
                float16x8_t neon_base3 = vld1q_f16(y + 2 * d + j);
                float16x8_t neon_base4 = vld1q_f16(y + 3 * d + j);
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_query);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_query);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_query);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_query);
                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_query);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_query);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_query);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 4 * d + j);
                neon_base2 = vld1q_f16(y + 5 * d + j);
                neon_base3 = vld1q_f16(y + 6 * d + j);
                neon_base4 = vld1q_f16(y + 7 * d + j);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_query);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_query);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_query);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_query);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_query);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_query);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_query);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 8 * d + j);
                neon_base2 = vld1q_f16(y + 9 * d + j);
                neon_base3 = vld1q_f16(y + 10 * d + j);
                neon_base4 = vld1q_f16(y + 11 * d + j);
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_query);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_query);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_query);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_query);
                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_query);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_query);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_query);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 12 * d + j);
                neon_base2 = vld1q_f16(y + 13 * d + j);
                neon_base3 = vld1q_f16(y + 14 * d + j);
                neon_base4 = vld1q_f16(y + 15 * d + j);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_query);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_query);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_query);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_query);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_query);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_query);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_query);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 16 * d + j);
                neon_base2 = vld1q_f16(y + 17 * d + j);
                neon_base3 = vld1q_f16(y + 18 * d + j);
                neon_base4 = vld1q_f16(y + 19 * d + j);
                neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_query);
                neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_query);
                neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_query);
                neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_query);
                neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_query);
                neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_query);
                neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_query);
                neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 20 * d + j);
                neon_base2 = vld1q_f16(y + 21 * d + j);
                neon_base3 = vld1q_f16(y + 22 * d + j);
                neon_base4 = vld1q_f16(y + 23 * d + j);
                neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_query);
                neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_query);
                neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_query);
                neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_query);
                neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_query);
                neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_query);
                neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_query);
                neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_query);
            }
        }

        for (; i <= d - single_round; i += single_round) {
            /* Load query vector */
            const float16x8_t neon_query = vld1q_f16(x + i);
            /* Load base vectors and compute inner products */
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_query);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 4 * d + i);
            neon_base2 = vld1q_f16(y + 5 * d + i);
            neon_base3 = vld1q_f16(y + 6 * d + i);
            neon_base4 = vld1q_f16(y + 7 * d + i);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_query);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_query);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 12 * d + i);
            neon_base2 = vld1q_f16(y + 13 * d + i);
            neon_base3 = vld1q_f16(y + 14 * d + i);
            neon_base4 = vld1q_f16(y + 15 * d + i);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_query);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 16 * d + i);
            neon_base2 = vld1q_f16(y + 17 * d + i);
            neon_base3 = vld1q_f16(y + 18 * d + i);
            neon_base4 = vld1q_f16(y + 19 * d + i);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_query);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 20 * d + i);
            neon_base2 = vld1q_f16(y + 21 * d + i);
            neon_base3 = vld1q_f16(y + 22 * d + i);
            neon_base4 = vld1q_f16(y + 23 * d + i);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_query);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_query);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        dis[8] = vaddvq_f32(neon_res9);
        dis[9] = vaddvq_f32(neon_res10);
        dis[10] = vaddvq_f32(neon_res11);
        dis[11] = vaddvq_f32(neon_res12);
        dis[12] = vaddvq_f32(neon_res13);
        dis[13] = vaddvq_f32(neon_res14);
        dis[14] = vaddvq_f32(neon_res15);
        dis[15] = vaddvq_f32(neon_res16);
        dis[16] = vaddvq_f32(neon_res17);
        dis[17] = vaddvq_f32(neon_res18);
        dis[18] = vaddvq_f32(neon_res19);
        dis[19] = vaddvq_f32(neon_res20);
        dis[20] = vaddvq_f32(neon_res21);
        dis[21] = vaddvq_f32(neon_res22);
        dis[22] = vaddvq_f32(neon_res23);
        dis[23] = vaddvq_f32(neon_res24);
    } else {
        for (int i = 0; i < 24; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);
        float d4 = x[i] * *(y + 4 * d + i);
        float d5 = x[i] * *(y + 5 * d + i);
        float d6 = x[i] * *(y + 6 * d + i);
        float d7 = x[i] * *(y + 7 * d + i);
        float d8 = x[i] * *(y + 8 * d + i);
        float d9 = x[i] * *(y + 9 * d + i);
        float d10 = x[i] * *(y + 10 * d + i);
        float d11 = x[i] * *(y + 11 * d + i);
        float d12 = x[i] * *(y + 12 * d + i);
        float d13 = x[i] * *(y + 13 * d + i);
        float d14 = x[i] * *(y + 14 * d + i);
        float d15 = x[i] * *(y + 15 * d + i);
        float d16 = x[i] * *(y + 16 * d + i);
        float d17 = x[i] * *(y + 17 * d + i);
        float d18 = x[i] * *(y + 18 * d + i);
        float d19 = x[i] * *(y + 19 * d + i);
        float d20 = x[i] * *(y + 20 * d + i);
        float d21 = x[i] * *(y + 21 * d + i);
        float d22 = x[i] * *(y + 22 * d + i);
        float d23 = x[i] * *(y + 23 * d + i);
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
            d16 += x[i] * *(y + 16 * d + i);
            d17 += x[i] * *(y + 17 * d + i);
            d18 += x[i] * *(y + 18 * d + i);
            d19 += x[i] * *(y + 19 * d + i);
            d20 += x[i] * *(y + 20 * d + i);
            d21 += x[i] * *(y + 21 * d + i);
            d22 += x[i] * *(y + 22 * d + i);
            d23 += x[i] * *(y + 23 * d + i);
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
 * @brief Compute inner product between query vector and multiple vectors specified by indices.
 * @param dis Pointer to the output array for storing the results (float).
 * @param x Pointer to the query vector (uint16_t).
 * @param y Pointer to the database vectors (uint16_t).
 * @param ids Pointer to the indices of the database vectors to be used.
 * @param d Dimension of the vectors.
 * @param ny Number of vectors to compute inner products with.
 * @param dis_size Length of dis.
 */
int krl_inner_product_by_idx_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, const int64_t *ids, size_t d, size_t ny, size_t dis_size)
{
    size_t i = 0;
    const float16_t *__restrict listy[24];

    /* Process vectors in batches of 24 */
    for (; i + 24 <= ny; i += 24) {
        /* Prefetch query vector */
        prefetch_L1(x);
        /* Load and prefetch 24 database vectors */
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        prefetch_Lx(listy[0]);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        prefetch_Lx(listy[1]);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        prefetch_Lx(listy[2]);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        prefetch_Lx(listy[3]);
        listy[4] = (const float16_t *)(y + *(ids + i + 4) * d);
        prefetch_Lx(listy[4]);
        listy[5] = (const float16_t *)(y + *(ids + i + 5) * d);
        prefetch_Lx(listy[5]);
        listy[6] = (const float16_t *)(y + *(ids + i + 6) * d);
        prefetch_Lx(listy[6]);
        listy[7] = (const float16_t *)(y + *(ids + i + 7) * d);
        prefetch_Lx(listy[7]);
        listy[8] = (const float16_t *)(y + *(ids + i + 8) * d);
        prefetch_Lx(listy[8]);
        listy[9] = (const float16_t *)(y + *(ids + i + 9) * d);
        prefetch_Lx(listy[9]);
        listy[10] = (const float16_t *)(y + *(ids + i + 10) * d);
        prefetch_Lx(listy[10]);
        listy[11] = (const float16_t *)(y + *(ids + i + 11) * d);
        prefetch_Lx(listy[11]);
        listy[12] = (const float16_t *)(y + *(ids + i + 12) * d);
        prefetch_Lx(listy[12]);
        listy[13] = (const float16_t *)(y + *(ids + i + 13) * d);
        prefetch_Lx(listy[13]);
        listy[14] = (const float16_t *)(y + *(ids + i + 14) * d);
        prefetch_Lx(listy[14]);
        listy[15] = (const float16_t *)(y + *(ids + i + 15) * d);
        prefetch_Lx(listy[15]);
        listy[16] = (const float16_t *)(y + *(ids + i + 16) * d);
        prefetch_Lx(listy[16]);
        listy[17] = (const float16_t *)(y + *(ids + i + 17) * d);
        prefetch_Lx(listy[17]);
        listy[18] = (const float16_t *)(y + *(ids + i + 18) * d);
        prefetch_Lx(listy[18]);
        listy[19] = (const float16_t *)(y + *(ids + i + 19) * d);
        prefetch_Lx(listy[19]);
        listy[20] = (const float16_t *)(y + *(ids + i + 20) * d);
        prefetch_Lx(listy[20]);
        listy[21] = (const float16_t *)(y + *(ids + i + 21) * d);
        prefetch_Lx(listy[21]);
        listy[22] = (const float16_t *)(y + *(ids + i + 22) * d);
        prefetch_Lx(listy[22]);
        listy[23] = (const float16_t *)(y + *(ids + i + 23) * d);
        prefetch_Lx(listy[23]);
        /* Compute inner products for 24 vectors */
        krl_inner_product_idx_prefetch_batch24_f16f32((const float16_t *)x, listy, d, dis + i);
    }

    /* Handle remaining vectors in batches of 16 */
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
        krl_inner_product_idx_prefetch_batch16_f16f32((const float16_t *)x, listy, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        /* Handle remaining vectors in batches of 8 */
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        listy[4] = (const float16_t *)(y + *(ids + i + 4) * d);
        listy[5] = (const float16_t *)(y + *(ids + i + 5) * d);
        listy[6] = (const float16_t *)(y + *(ids + i + 6) * d);
        listy[7] = (const float16_t *)(y + *(ids + i + 7) * d);
        krl_inner_product_idx_batch8_f16f32((const float16_t *)x, listy, d, dis + i);
        i += 8;
    }

    /* Handle remaining vectors in batches of 4 */
    if (ny & 4) {
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        krl_inner_product_idx_batch4_f16f32((const float16_t *)x, listy, d, dis + i);
        i += 4;
    }

    /* Handle remaining vectors in batches of 2 */
    if (ny & 2) {
        const float16_t *y0 = (const float16_t *)(y + *(ids + i) * d);
        const float16_t *y1 = (const float16_t *)(y + *(ids + i + 1) * d);
        krl_inner_product_idx_batch2_f16f32((const float16_t *)x, y0, y1, d, dis + i);
        i += 2;
    }

    /* Handle the last remaining vector */
    if (ny & 1) {
        dis[i] = krl_inner_product_f16f32(x, y + d * ids[i], d);
    }
    return SUCCESS;
}

/*
 * @brief Compute inner product between query vector and multiple vectors in the database.
 * @param dis Pointer to the output array for storing the results (float).
 * @param x Pointer to the query vector (uint16_t).
 * @param y Pointer to the database vectors (uint16_t).
 * @param ny Number of vectors to compute inner products with.
 * @param d Dimension of the vectors.
 * @param dis_size Length of dis.
 */
int krl_inner_product_ny_f16f32(float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d, size_t dis_size)
{
    size_t i = 0;

    /* Process vectors in batches of 24 */
    for (; i + 24 <= ny; i += 24) {
        /* Prefetch query vector */
        prefetch_L1(x);
        /* Prefetch database vectors */
        prefetch_Lx(y + i * d);
        /* Compute inner products for 24 vectors */
        krl_inner_product_prefetch_batch24_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
    }

    /* Handle remaining vectors in batches of 16 */
    if (i + 16 <= ny) {
        /* Prefetch query vector */
        prefetch_L1(x);
        /* Prefetch database vectors */
        prefetch_Lx(y + i * d);
        /* Compute inner products for 16 vectors */
        krl_inner_product_prefetch_batch16_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        /* Handle remaining vectors in batches of 8 */
        krl_inner_product_batch8_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 8;
    }

    /* Handle remaining vectors in batches of 4 */
    if (ny & 4) {
        krl_inner_product_batch4_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 4;
    }

    /* Handle remaining vectors in batches of 2 */
    if (ny & 2) {
        krl_inner_product_batch2_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 2;
    }

    /* Handle the last remaining vector */
    if (ny & 1) {
        dis[i] = krl_inner_product_f16f32(x, y + i * d, d);
    }
    return SUCCESS;
}
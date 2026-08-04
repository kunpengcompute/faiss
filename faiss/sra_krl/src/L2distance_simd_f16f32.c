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
 * @brief Compute the L2 square of two float16 vectors and return the result as float32
 * @param u16_x Pointer to the first float16 vector
 * @param u16_y Pointer to the second float16 vector (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Stores the computed L2 square result (float).
 * @param dis_size Length of dis.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
int krl_L2sqr_f16f32(
    const uint16_t *u16_x, const uint16_t *__restrict u16_y, const size_t d, float *dis)
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

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        float16x8_t d8_1 = vsubq_f16(x8_1, y8_1);
        float16x8_t d8_2 = vsubq_f16(x8_2, y8_2);
        float16x8_t d8_3 = vsubq_f16(x8_3, y8_3);

        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_low_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_low_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_low_f16(res4, d8_3, d8_3);

        res1 = vfmlalq_high_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_high_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_high_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_high_f16(res4, d8_3, d8_3);
    }

    /* Handle remaining elements with single rounds */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res3 = vfmlalq_high_f16(res3, d8_0, d8_0);
    }
    /* Accumulate results */
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    res1 = vaddq_f32(res1, res3);
    res = vaddvq_f32(res1);
    /* Handle remaining elements */
    for (; i < d; i++) {
        const float16_t tmp = x[i] - y[i];
        res += (float)(tmp * tmp);
    }
    *dis = res;
    return SUCCESS;
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the L2 square of a float16 vector with two other float16 vectors and store the results in a float
 * array
 * @param x Pointer to the input float16 vector
 * @param y0 Pointer to the first float16 vector (with restrict qualifier for better optimization)
 * @param y1 Pointer to the second float16 vector (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares (size must be at least 2)
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_batch2_f16f32(
    const float16_t *x, const float16_t *__restrict y0, const float16_t *__restrict y1, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;  /* Number of elements processed per single round */
    constexpr size_t double_round = 16; /* Number of elements processed per double round */
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);
    for (i = 0; i + double_round <= d; i += double_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);

        float16x8_t y8_0 = vld1q_f16(y0 + i);
        float16x8_t y8_1 = vld1q_f16(y0 + i + 8);
        float16x8_t y8_2 = vld1q_f16(y1 + i);
        float16x8_t y8_3 = vld1q_f16(y1 + i + 8);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        float16x8_t d8_1 = vsubq_f16(x8_1, y8_1);
        float16x8_t d8_2 = vsubq_f16(x8_0, y8_2);
        float16x8_t d8_3 = vsubq_f16(x8_1, y8_3);

        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_low_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_low_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_low_f16(res4, d8_3, d8_3);

        res1 = vfmlalq_high_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_high_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_high_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_high_f16(res4, d8_3, d8_3);
    }
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y0 + i);
        float16x8_t y8_1 = vld1q_f16(y1 + i);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        float16x8_t d8_1 = vsubq_f16(x8_0, y8_1);
        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res3 = vfmlalq_low_f16(res3, d8_1, d8_1);
        res2 = vfmlalq_high_f16(res2, d8_0, d8_0);
        res4 = vfmlalq_high_f16(res4, d8_1, d8_1);
    }
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    dis[0] = vaddvq_f32(res1);
    dis[1] = vaddvq_f32(res3);
    for (; i < d; i++) {
        const float16_t tmp0 = x[i] - y0[i];
        const float16_t tmp1 = x[i] - y1[i];
        dis[0] += (float)(tmp0 * tmp0);
        dis[1] += (float)(tmp1 * tmp1);
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the L2 square of a float16 vector with four other float16 vectors and store the results in a float
 * array
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of four float16 vectors (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares (size must be at least 4)
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_batch4_f16f32(const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;  /* 128 / 16 */
    constexpr size_t double_round = 16; /* 2 * single_round */

    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);

    if (likely(d >= double_round)) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y[0]);
        float16x8_t neon_base2 = vld1q_f16(y[1]);
        float16x8_t neon_base3 = vld1q_f16(y[2]);
        float16x8_t neon_base4 = vld1q_f16(y[3]);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_query = vld1q_f16(x + single_round);
        neon_base1 = vld1q_f16(y[0] + single_round);
        neon_base2 = vld1q_f16(y[1] + single_round);
        neon_base3 = vld1q_f16(y[2] + single_round);
        neon_base4 = vld1q_f16(y[3] + single_round);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        for (i = double_round; i <= d - single_round; i += single_round) {
            neon_diff1 = vsubq_f16(neon_base1, neon_query);
            neon_diff2 = vsubq_f16(neon_base2, neon_query);
            neon_diff3 = vsubq_f16(neon_base3, neon_query);
            neon_diff4 = vsubq_f16(neon_base4, neon_query);

            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y[0] + i);
            neon_base2 = vld1q_f16(y[1] + i);
            neon_base3 = vld1q_f16(y[2] + i);
            neon_base4 = vld1q_f16(y[3] + i);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        }
        neon_diff1 = vsubq_f16(neon_base1, neon_query);
        neon_diff2 = vsubq_f16(neon_base2, neon_query);
        neon_diff3 = vsubq_f16(neon_base3, neon_query);
        neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
    } else if (d >= single_round) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y[0]);
        float16x8_t neon_base2 = vld1q_f16(y[1]);
        float16x8_t neon_base3 = vld1q_f16(y[2]);
        float16x8_t neon_base4 = vld1q_f16(y[3]);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        i = single_round;
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y[0] + i);
        float16_t q1 = x[i] - *(y[1] + i);
        float16_t q2 = x[i] - *(y[2] + i);
        float16_t q3 = x[i] - *(y[3] + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y[0] + i);
            q1 = x[i] - *(y[1] + i);
            q2 = x[i] - *(y[2] + i);
            q3 = x[i] - *(y[3] + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the L2 square of a float16 vector with eight other float16 vectors and store the results in a float
 * array
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of eight float16 vectors (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares (size must be at least 8)
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_batch8_f16f32(const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 16;

    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);

    if (likely(d >= double_round)) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y[0]);
        float16x8_t neon_base2 = vld1q_f16(y[1]);
        float16x8_t neon_base3 = vld1q_f16(y[2]);
        float16x8_t neon_base4 = vld1q_f16(y[3]);
        float16x8_t neon_base5 = vld1q_f16(y[4]);
        float16x8_t neon_base6 = vld1q_f16(y[5]);
        float16x8_t neon_base7 = vld1q_f16(y[6]);
        float16x8_t neon_base8 = vld1q_f16(y[7]);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_diff5 = vsubq_f16(neon_base5, neon_query);
        float16x8_t neon_diff6 = vsubq_f16(neon_base6, neon_query);
        float16x8_t neon_diff7 = vsubq_f16(neon_base7, neon_query);
        float16x8_t neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_query = vld1q_f16(x + single_round);
        neon_base1 = vld1q_f16(y[0] + single_round);
        neon_base2 = vld1q_f16(y[1] + single_round);
        neon_base3 = vld1q_f16(y[2] + single_round);
        neon_base4 = vld1q_f16(y[3] + single_round);
        neon_base5 = vld1q_f16(y[4] + single_round);
        neon_base6 = vld1q_f16(y[5] + single_round);
        neon_base7 = vld1q_f16(y[6] + single_round);
        neon_base8 = vld1q_f16(y[7] + single_round);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);

        for (i = double_round; i <= d - single_round; i += single_round) {
            neon_diff1 = vsubq_f16(neon_base1, neon_query);
            neon_diff2 = vsubq_f16(neon_base2, neon_query);
            neon_diff3 = vsubq_f16(neon_base3, neon_query);
            neon_diff4 = vsubq_f16(neon_base4, neon_query);
            neon_diff5 = vsubq_f16(neon_base5, neon_query);
            neon_diff6 = vsubq_f16(neon_base6, neon_query);
            neon_diff7 = vsubq_f16(neon_base7, neon_query);
            neon_diff8 = vsubq_f16(neon_base8, neon_query);

            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y[0] + i);
            neon_base2 = vld1q_f16(y[1] + i);
            neon_base3 = vld1q_f16(y[2] + i);
            neon_base4 = vld1q_f16(y[3] + i);
            neon_base5 = vld1q_f16(y[4] + i);
            neon_base6 = vld1q_f16(y[5] + i);
            neon_base7 = vld1q_f16(y[6] + i);
            neon_base8 = vld1q_f16(y[7] + i);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        }
        neon_diff1 = vsubq_f16(neon_base1, neon_query);
        neon_diff2 = vsubq_f16(neon_base2, neon_query);
        neon_diff3 = vsubq_f16(neon_base3, neon_query);
        neon_diff4 = vsubq_f16(neon_base4, neon_query);
        neon_diff5 = vsubq_f16(neon_base5, neon_query);
        neon_diff6 = vsubq_f16(neon_base6, neon_query);
        neon_diff7 = vsubq_f16(neon_base7, neon_query);
        neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else if (d >= single_round) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y[0]);
        float16x8_t neon_base2 = vld1q_f16(y[1]);
        float16x8_t neon_base3 = vld1q_f16(y[2]);
        float16x8_t neon_base4 = vld1q_f16(y[3]);
        float16x8_t neon_base5 = vld1q_f16(y[4]);
        float16x8_t neon_base6 = vld1q_f16(y[5]);
        float16x8_t neon_base7 = vld1q_f16(y[6]);
        float16x8_t neon_base8 = vld1q_f16(y[7]);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_diff5 = vsubq_f16(neon_base5, neon_query);
        float16x8_t neon_diff6 = vsubq_f16(neon_base6, neon_query);
        float16x8_t neon_diff7 = vsubq_f16(neon_base7, neon_query);
        float16x8_t neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        i = single_round;
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y[0] + i);
        float16_t q1 = x[i] - *(y[1] + i);
        float16_t q2 = x[i] - *(y[2] + i);
        float16_t q3 = x[i] - *(y[3] + i);
        float16_t q4 = x[i] - *(y[4] + i);
        float16_t q5 = x[i] - *(y[5] + i);
        float16_t q6 = x[i] - *(y[6] + i);
        float16_t q7 = x[i] - *(y[7] + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y[0] + i);
            q1 = x[i] - *(y[1] + i);
            q2 = x[i] - *(y[2] + i);
            q3 = x[i] - *(y[3] + i);
            q4 = x[i] - *(y[4] + i);
            q5 = x[i] - *(y[5] + i);
            q6 = x[i] - *(y[6] + i);
            q7 = x[i] - *(y[7] + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
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
 * @brief Compute the L2 square of a float16 vector with sixteen other float16 vectors and store the results in a float
 * array
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of sixteen float16 vectors (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares (size must be at least 16)
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_prefetch_batch16_f16f32(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */

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
        for (i = 0; i < d - multi_round; i += multi_round) {
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

                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_base5 = vsubq_f16(neon_base5, neon_query);
                neon_base6 = vsubq_f16(neon_base6, neon_query);
                neon_base7 = vsubq_f16(neon_base7, neon_query);
                neon_base8 = vsubq_f16(neon_base8, neon_query);

                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

                neon_base1 = vld1q_f16(y[8] + i + j);
                neon_base2 = vld1q_f16(y[9] + i + j);
                neon_base3 = vld1q_f16(y[10] + i + j);
                neon_base4 = vld1q_f16(y[11] + i + j);
                neon_base5 = vld1q_f16(y[12] + i + j);
                neon_base6 = vld1q_f16(y[13] + i + j);
                neon_base7 = vld1q_f16(y[14] + i + j);
                neon_base8 = vld1q_f16(y[15] + i + j);

                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_base5 = vsubq_f16(neon_base5, neon_query);
                neon_base6 = vsubq_f16(neon_base6, neon_query);
                neon_base7 = vsubq_f16(neon_base7, neon_query);
                neon_base8 = vsubq_f16(neon_base8, neon_query);

                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
            }
        }
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

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base5 = vld1q_f16(y[12] + i);
            neon_base6 = vld1q_f16(y[13] + i);
            neon_base7 = vld1q_f16(y[14] + i);
            neon_base8 = vld1q_f16(y[15] + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
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
    } else {
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

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base5 = vld1q_f16(y[12] + i);
            neon_base6 = vld1q_f16(y[13] + i);
            neon_base7 = vld1q_f16(y[14] + i);
            neon_base8 = vld1q_f16(y[15] + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
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
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y[0] + i);
        float16_t q1 = x[i] - *(y[1] + i);
        float16_t q2 = x[i] - *(y[2] + i);
        float16_t q3 = x[i] - *(y[3] + i);
        float16_t q4 = x[i] - *(y[4] + i);
        float16_t q5 = x[i] - *(y[5] + i);
        float16_t q6 = x[i] - *(y[6] + i);
        float16_t q7 = x[i] - *(y[7] + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        q0 = x[i] - *(y[8] + i);
        q1 = x[i] - *(y[9] + i);
        q2 = x[i] - *(y[10] + i);
        q3 = x[i] - *(y[11] + i);
        q4 = x[i] - *(y[12] + i);
        q5 = x[i] - *(y[13] + i);
        q6 = x[i] - *(y[14] + i);
        q7 = x[i] - *(y[15] + i);
        float d8 = q0 * q0;
        float d9 = q1 * q1;
        float d10 = q2 * q2;
        float d11 = q3 * q3;
        float d12 = q4 * q4;
        float d13 = q5 * q5;
        float d14 = q6 * q6;
        float d15 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y[0] + i);
            q1 = x[i] - *(y[1] + i);
            q2 = x[i] - *(y[2] + i);
            q3 = x[i] - *(y[3] + i);
            q4 = x[i] - *(y[4] + i);
            q5 = x[i] - *(y[5] + i);
            q6 = x[i] - *(y[6] + i);
            q7 = x[i] - *(y[7] + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            q0 = x[i] - *(y[8] + i);
            q1 = x[i] - *(y[9] + i);
            q2 = x[i] - *(y[10] + i);
            q3 = x[i] - *(y[11] + i);
            q4 = x[i] - *(y[12] + i);
            q5 = x[i] - *(y[13] + i);
            q6 = x[i] - *(y[14] + i);
            q7 = x[i] - *(y[15] + i);
            d8 += q0 * q0;
            d9 += q1 * q1;
            d10 += q2 * q2;
            d11 += q3 * q3;
            d12 += q4 * q4;
            d13 += q5 * q5;
            d14 += q6 * q6;
            d15 += q7 * q7;
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
 * @brief Computes the L2 square of multiple vectors using NEON instructions with prefetching.
 *        This function handles 24 vectors in batches, using float16 precision for input and float32 for output.
 * @param x Pointer to the input vector (float16).
 * @param y Array of pointers to the base vectors (float16).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array storing the L2 squares (float32).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_prefetch_batch24_f16f32(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */

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
                const float16x8_t neon_query = vld1q_f16(x + j);
                float16x8_t neon_base1 = vld1q_f16(y[0] + j);
                float16x8_t neon_base2 = vld1q_f16(y[1] + j);
                float16x8_t neon_base3 = vld1q_f16(y[2] + j);
                float16x8_t neon_base4 = vld1q_f16(y[3] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[4] + j);
                neon_base2 = vld1q_f16(y[5] + j);
                neon_base3 = vld1q_f16(y[6] + j);
                neon_base4 = vld1q_f16(y[7] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[8] + j);
                neon_base2 = vld1q_f16(y[9] + j);
                neon_base3 = vld1q_f16(y[10] + j);
                neon_base4 = vld1q_f16(y[11] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[12] + j);
                neon_base2 = vld1q_f16(y[13] + j);
                neon_base3 = vld1q_f16(y[14] + j);
                neon_base4 = vld1q_f16(y[15] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[16] + j);
                neon_base2 = vld1q_f16(y[17] + j);
                neon_base3 = vld1q_f16(y[18] + j);
                neon_base4 = vld1q_f16(y[19] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
                neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
                neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
                neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
                neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
                neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
                neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
                neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[20] + j);
                neon_base2 = vld1q_f16(y[21] + j);
                neon_base3 = vld1q_f16(y[22] + j);
                neon_base4 = vld1q_f16(y[23] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
                neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
                neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
                neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
                neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
                neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
                neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
                neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
            }
        }
        for (; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
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
    } else if (d >= single_round) {
        for (i = 0; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
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
        float q0 = x[i] - *(y[0] + i);
        float q1 = x[i] - *(y[1] + i);
        float q2 = x[i] - *(y[2] + i);
        float q3 = x[i] - *(y[3] + i);
        float q4 = x[i] - *(y[4] + i);
        float q5 = x[i] - *(y[5] + i);
        float q6 = x[i] - *(y[6] + i);
        float q7 = x[i] - *(y[7] + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        q0 = x[i] - *(y[8] + i);
        q1 = x[i] - *(y[9] + i);
        q2 = x[i] - *(y[10] + i);
        q3 = x[i] - *(y[11] + i);
        q4 = x[i] - *(y[12] + i);
        q5 = x[i] - *(y[13] + i);
        q6 = x[i] - *(y[14] + i);
        q7 = x[i] - *(y[15] + i);
        float d8 = q0 * q0;
        float d9 = q1 * q1;
        float d10 = q2 * q2;
        float d11 = q3 * q3;
        float d12 = q4 * q4;
        float d13 = q5 * q5;
        float d14 = q6 * q6;
        float d15 = q7 * q7;
        q0 = x[i] - *(y[16] + i);
        q1 = x[i] - *(y[17] + i);
        q2 = x[i] - *(y[18] + i);
        q3 = x[i] - *(y[19] + i);
        q4 = x[i] - *(y[20] + i);
        q5 = x[i] - *(y[21] + i);
        q6 = x[i] - *(y[22] + i);
        q7 = x[i] - *(y[23] + i);
        float d16 = q0 * q0;
        float d17 = q1 * q1;
        float d18 = q2 * q2;
        float d19 = q3 * q3;
        float d20 = q4 * q4;
        float d21 = q5 * q5;
        float d22 = q6 * q6;
        float d23 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y[0] + i);
            q1 = x[i] - *(y[1] + i);
            q2 = x[i] - *(y[2] + i);
            q3 = x[i] - *(y[3] + i);
            q4 = x[i] - *(y[4] + i);
            q5 = x[i] - *(y[5] + i);
            q6 = x[i] - *(y[6] + i);
            q7 = x[i] - *(y[7] + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            q0 = x[i] - *(y[8] + i);
            q1 = x[i] - *(y[9] + i);
            q2 = x[i] - *(y[10] + i);
            q3 = x[i] - *(y[11] + i);
            q4 = x[i] - *(y[12] + i);
            q5 = x[i] - *(y[13] + i);
            q6 = x[i] - *(y[14] + i);
            q7 = x[i] - *(y[15] + i);
            d8 += q0 * q0;
            d9 += q1 * q1;
            d10 += q2 * q2;
            d11 += q3 * q3;
            d12 += q4 * q4;
            d13 += q5 * q5;
            d14 += q6 * q6;
            d15 += q7 * q7;
            q0 = x[i] - *(y[16] + i);
            q1 = x[i] - *(y[17] + i);
            q2 = x[i] - *(y[18] + i);
            q3 = x[i] - *(y[19] + i);
            q4 = x[i] - *(y[20] + i);
            q5 = x[i] - *(y[21] + i);
            q6 = x[i] - *(y[22] + i);
            q7 = x[i] - *(y[23] + i);
            d16 += q0 * q0;
            d17 += q1 * q1;
            d18 += q2 * q2;
            d19 += q3 * q3;
            d20 += q4 * q4;
            d21 += q5 * q5;
            d22 += q6 * q6;
            d23 += q7 * q7;
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
 * @brief Compute the L2 square of two batches of half-precision floating-point vectors.
 * @param x Pointer to the first input vector (half-precision float).
 * @param y Pointer to the second input vector (half-precision float), which is split into two parts.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch2_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 16;
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);
    for (i = 0; i + double_round <= d; i += double_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);

        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + i + 8);
        float16x8_t y8_2 = vld1q_f16(y + d + i);
        float16x8_t y8_3 = vld1q_f16(y + d + i + 8);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        float16x8_t d8_1 = vsubq_f16(x8_1, y8_1);
        float16x8_t d8_2 = vsubq_f16(x8_0, y8_2);
        float16x8_t d8_3 = vsubq_f16(x8_1, y8_3);

        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_low_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_low_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_low_f16(res4, d8_3, d8_3);

        res1 = vfmlalq_high_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_high_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_high_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_high_f16(res4, d8_3, d8_3);
    }
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + d + i);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        float16x8_t d8_1 = vsubq_f16(x8_0, y8_1);
        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res3 = vfmlalq_low_f16(res3, d8_1, d8_1);
        res2 = vfmlalq_high_f16(res2, d8_0, d8_0);
        res4 = vfmlalq_high_f16(res4, d8_1, d8_1);
    }
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    dis[0] = vaddvq_f32(res1);
    dis[1] = vaddvq_f32(res3);
    for (; i < d; i++) {
        const float16_t tmp0 = x[i] - y[i];
        const float16_t tmp1 = x[i] - y[i + d];
        dis[0] += (float)(tmp0 * tmp0);
        dis[1] += (float)(tmp1 * tmp1);
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the L2 square of four batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains four batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch4_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 16;

    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    if (likely(d >= double_round)) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_query = vld1q_f16(x + single_round);
        neon_base1 = vld1q_f16(y + single_round);
        neon_base2 = vld1q_f16(y + d + single_round);
        neon_base3 = vld1q_f16(y + 2 * d + single_round);
        neon_base4 = vld1q_f16(y + 3 * d + single_round);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        for (i = double_round; i <= d - single_round; i += single_round) {
            neon_diff1 = vsubq_f16(neon_base1, neon_query);
            neon_diff2 = vsubq_f16(neon_base2, neon_query);
            neon_diff3 = vsubq_f16(neon_base3, neon_query);
            neon_diff4 = vsubq_f16(neon_base4, neon_query);

            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        }
        neon_diff1 = vsubq_f16(neon_base1, neon_query);
        neon_diff2 = vsubq_f16(neon_base2, neon_query);
        neon_diff3 = vsubq_f16(neon_base3, neon_query);
        neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
    } else if (d >= single_round) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        i = single_round;
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y + i);
        float16_t q1 = x[i] - *(y + d + i);
        float16_t q2 = x[i] - *(y + 2 * d + i);
        float16_t q3 = x[i] - *(y + 3 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the L2 square of eight batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains eight batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch8_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 16;

    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);

    if (likely(d >= double_round)) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        float16x8_t neon_base5 = vld1q_f16(y + 4 * d);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_diff5 = vsubq_f16(neon_base5, neon_query);
        float16x8_t neon_diff6 = vsubq_f16(neon_base6, neon_query);
        float16x8_t neon_diff7 = vsubq_f16(neon_base7, neon_query);
        float16x8_t neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_query = vld1q_f16(x + single_round);
        neon_base1 = vld1q_f16(y + single_round);
        neon_base2 = vld1q_f16(y + d + single_round);
        neon_base3 = vld1q_f16(y + 2 * d + single_round);
        neon_base4 = vld1q_f16(y + 3 * d + single_round);
        neon_base5 = vld1q_f16(y + 4 * d + single_round);
        neon_base6 = vld1q_f16(y + 5 * d + single_round);
        neon_base7 = vld1q_f16(y + 6 * d + single_round);
        neon_base8 = vld1q_f16(y + 7 * d + single_round);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);

        for (i = double_round; i <= d - single_round; i += single_round) {
            neon_diff1 = vsubq_f16(neon_base1, neon_query);
            neon_diff2 = vsubq_f16(neon_base2, neon_query);
            neon_diff3 = vsubq_f16(neon_base3, neon_query);
            neon_diff4 = vsubq_f16(neon_base4, neon_query);
            neon_diff5 = vsubq_f16(neon_base5, neon_query);
            neon_diff6 = vsubq_f16(neon_base6, neon_query);
            neon_diff7 = vsubq_f16(neon_base7, neon_query);
            neon_diff8 = vsubq_f16(neon_base8, neon_query);

            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base5 = vld1q_f16(y + 4 * d + i);
            neon_base6 = vld1q_f16(y + 5 * d + i);
            neon_base7 = vld1q_f16(y + 6 * d + i);
            neon_base8 = vld1q_f16(y + 7 * d + i);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        }
        neon_diff1 = vsubq_f16(neon_base1, neon_query);
        neon_diff2 = vsubq_f16(neon_base2, neon_query);
        neon_diff3 = vsubq_f16(neon_base3, neon_query);
        neon_diff4 = vsubq_f16(neon_base4, neon_query);
        neon_diff5 = vsubq_f16(neon_base5, neon_query);
        neon_diff6 = vsubq_f16(neon_base6, neon_query);
        neon_diff7 = vsubq_f16(neon_base7, neon_query);
        neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else if (d >= single_round) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        float16x8_t neon_base5 = vld1q_f16(y + 4 * d);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_diff5 = vsubq_f16(neon_base5, neon_query);
        float16x8_t neon_diff6 = vsubq_f16(neon_base6, neon_query);
        float16x8_t neon_diff7 = vsubq_f16(neon_base7, neon_query);
        float16x8_t neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        i = single_round;
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y + i);
        float16_t q1 = x[i] - *(y + d + i);
        float16_t q2 = x[i] - *(y + 2 * d + i);
        float16_t q3 = x[i] - *(y + 3 * d + i);
        float16_t q4 = x[i] - *(y + 4 * d + i);
        float16_t q5 = x[i] - *(y + 5 * d + i);
        float16_t q6 = x[i] - *(y + 6 * d + i);
        float16_t q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
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
 * @brief Compute the L2 square of sixteen batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains sixteen batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch16_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */

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
        for (i = 0; i < d - multi_round; i += multi_round) {
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
            for (size_t j = 0; j < multi_round; j += single_round) {
                const float16x8_t neon_query = vld1q_f16(x + i + j);
                float16x8_t neon_base1 = vld1q_f16(y + i + j);
                float16x8_t neon_base2 = vld1q_f16(y + d + i + j);
                float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i + j);
                float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i + j);
                float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i + j);
                float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i + j);
                float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i + j);
                float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i + j);

                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_base5 = vsubq_f16(neon_base5, neon_query);
                neon_base6 = vsubq_f16(neon_base6, neon_query);
                neon_base7 = vsubq_f16(neon_base7, neon_query);
                neon_base8 = vsubq_f16(neon_base8, neon_query);

                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

                neon_base1 = vld1q_f16(y + 8 * d + i + j);
                neon_base2 = vld1q_f16(y + 9 * d + i + j);
                neon_base3 = vld1q_f16(y + 10 * d + i + j);
                neon_base4 = vld1q_f16(y + 11 * d + i + j);
                neon_base5 = vld1q_f16(y + 12 * d + i + j);
                neon_base6 = vld1q_f16(y + 13 * d + i + j);
                neon_base7 = vld1q_f16(y + 14 * d + i + j);
                neon_base8 = vld1q_f16(y + 15 * d + i + j);

                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_base5 = vsubq_f16(neon_base5, neon_query);
                neon_base6 = vsubq_f16(neon_base6, neon_query);
                neon_base7 = vsubq_f16(neon_base7, neon_query);
                neon_base8 = vsubq_f16(neon_base8, neon_query);

                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
            }
        }
        for (; i + single_round <= d; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
            float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
            float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
            float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
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
    } else {
        for (i = 0; i + single_round <= d; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
            float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
            float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
            float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
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
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y + i);
        float16_t q1 = x[i] - *(y + d + i);
        float16_t q2 = x[i] - *(y + 2 * d + i);
        float16_t q3 = x[i] - *(y + 3 * d + i);
        float16_t q4 = x[i] - *(y + 4 * d + i);
        float16_t q5 = x[i] - *(y + 5 * d + i);
        float16_t q6 = x[i] - *(y + 6 * d + i);
        float16_t q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        q0 = x[i] - *(y + 8 * d + i);
        q1 = x[i] - *(y + 9 * d + i);
        q2 = x[i] - *(y + 10 * d + i);
        q3 = x[i] - *(y + 11 * d + i);
        q4 = x[i] - *(y + 12 * d + i);
        q5 = x[i] - *(y + 13 * d + i);
        q6 = x[i] - *(y + 14 * d + i);
        q7 = x[i] - *(y + 15 * d + i);
        float d8 = q0 * q0;
        float d9 = q1 * q1;
        float d10 = q2 * q2;
        float d11 = q3 * q3;
        float d12 = q4 * q4;
        float d13 = q5 * q5;
        float d14 = q6 * q6;
        float d15 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            q0 = x[i] - *(y + 8 * d + i);
            q1 = x[i] - *(y + 9 * d + i);
            q2 = x[i] - *(y + 10 * d + i);
            q3 = x[i] - *(y + 11 * d + i);
            q4 = x[i] - *(y + 12 * d + i);
            q5 = x[i] - *(y + 13 * d + i);
            q6 = x[i] - *(y + 14 * d + i);
            q7 = x[i] - *(y + 15 * d + i);
            d8 += q0 * q0;
            d9 += q1 * q1;
            d10 += q2 * q2;
            d11 += q3 * q3;
            d12 += q4 * q4;
            d13 += q5 * q5;
            d14 += q6 * q6;
            d15 += q7 * q7;
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
 * @brief Compute the L2 square of sixteen batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains sixteen batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch24_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */

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
                const float16x8_t neon_query = vld1q_f16(x + j);
                float16x8_t neon_base1 = vld1q_f16(y + j);
                float16x8_t neon_base2 = vld1q_f16(y + d + j);
                float16x8_t neon_base3 = vld1q_f16(y + 2 * d + j);
                float16x8_t neon_base4 = vld1q_f16(y + 3 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 4 * d + j);
                neon_base2 = vld1q_f16(y + 5 * d + j);
                neon_base3 = vld1q_f16(y + 6 * d + j);
                neon_base4 = vld1q_f16(y + 7 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 8 * d + j);
                neon_base2 = vld1q_f16(y + 9 * d + j);
                neon_base3 = vld1q_f16(y + 10 * d + j);
                neon_base4 = vld1q_f16(y + 11 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 12 * d + j);
                neon_base2 = vld1q_f16(y + 13 * d + j);
                neon_base3 = vld1q_f16(y + 14 * d + j);
                neon_base4 = vld1q_f16(y + 15 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 16 * d + j);
                neon_base2 = vld1q_f16(y + 17 * d + j);
                neon_base3 = vld1q_f16(y + 18 * d + j);
                neon_base4 = vld1q_f16(y + 19 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
                neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
                neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
                neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
                neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
                neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
                neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
                neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 20 * d + j);
                neon_base2 = vld1q_f16(y + 21 * d + j);
                neon_base3 = vld1q_f16(y + 22 * d + j);
                neon_base4 = vld1q_f16(y + 23 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
                neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
                neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
                neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
                neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
                neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
                neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
                neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
            }
        }
        for (; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 4 * d + i);
            neon_base2 = vld1q_f16(y + 5 * d + i);
            neon_base3 = vld1q_f16(y + 6 * d + i);
            neon_base4 = vld1q_f16(y + 7 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 12 * d + i);
            neon_base2 = vld1q_f16(y + 13 * d + i);
            neon_base3 = vld1q_f16(y + 14 * d + i);
            neon_base4 = vld1q_f16(y + 15 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 16 * d + i);
            neon_base2 = vld1q_f16(y + 17 * d + i);
            neon_base3 = vld1q_f16(y + 18 * d + i);
            neon_base4 = vld1q_f16(y + 19 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 20 * d + i);
            neon_base2 = vld1q_f16(y + 21 * d + i);
            neon_base3 = vld1q_f16(y + 22 * d + i);
            neon_base4 = vld1q_f16(y + 23 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
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
    } else if (d >= single_round) {
        for (i = 0; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 4 * d + i);
            neon_base2 = vld1q_f16(y + 5 * d + i);
            neon_base3 = vld1q_f16(y + 6 * d + i);
            neon_base4 = vld1q_f16(y + 7 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 12 * d + i);
            neon_base2 = vld1q_f16(y + 13 * d + i);
            neon_base3 = vld1q_f16(y + 14 * d + i);
            neon_base4 = vld1q_f16(y + 15 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 16 * d + i);
            neon_base2 = vld1q_f16(y + 17 * d + i);
            neon_base3 = vld1q_f16(y + 18 * d + i);
            neon_base4 = vld1q_f16(y + 19 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 20 * d + i);
            neon_base2 = vld1q_f16(y + 21 * d + i);
            neon_base3 = vld1q_f16(y + 22 * d + i);
            neon_base4 = vld1q_f16(y + 23 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
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
        memset(dis, 0, sizeof(float) * 24);
        i = 0;
    }
    if (i < d) {
        float q0 = x[i] - *(y + i);
        float q1 = x[i] - *(y + d + i);
        float q2 = x[i] - *(y + 2 * d + i);
        float q3 = x[i] - *(y + 3 * d + i);
        float q4 = x[i] - *(y + 4 * d + i);
        float q5 = x[i] - *(y + 5 * d + i);
        float q6 = x[i] - *(y + 6 * d + i);
        float q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        q0 = x[i] - *(y + 8 * d + i);
        q1 = x[i] - *(y + 9 * d + i);
        q2 = x[i] - *(y + 10 * d + i);
        q3 = x[i] - *(y + 11 * d + i);
        q4 = x[i] - *(y + 12 * d + i);
        q5 = x[i] - *(y + 13 * d + i);
        q6 = x[i] - *(y + 14 * d + i);
        q7 = x[i] - *(y + 15 * d + i);
        float d8 = q0 * q0;
        float d9 = q1 * q1;
        float d10 = q2 * q2;
        float d11 = q3 * q3;
        float d12 = q4 * q4;
        float d13 = q5 * q5;
        float d14 = q6 * q6;
        float d15 = q7 * q7;
        q0 = x[i] - *(y + 16 * d + i);
        q1 = x[i] - *(y + 17 * d + i);
        q2 = x[i] - *(y + 18 * d + i);
        q3 = x[i] - *(y + 19 * d + i);
        q4 = x[i] - *(y + 20 * d + i);
        q5 = x[i] - *(y + 21 * d + i);
        q6 = x[i] - *(y + 22 * d + i);
        q7 = x[i] - *(y + 23 * d + i);
        float d16 = q0 * q0;
        float d17 = q1 * q1;
        float d18 = q2 * q2;
        float d19 = q3 * q3;
        float d20 = q4 * q4;
        float d21 = q5 * q5;
        float d22 = q6 * q6;
        float d23 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            q0 = x[i] - *(y + 8 * d + i);
            q1 = x[i] - *(y + 9 * d + i);
            q2 = x[i] - *(y + 10 * d + i);
            q3 = x[i] - *(y + 11 * d + i);
            q4 = x[i] - *(y + 12 * d + i);
            q5 = x[i] - *(y + 13 * d + i);
            q6 = x[i] - *(y + 14 * d + i);
            q7 = x[i] - *(y + 15 * d + i);
            d8 += q0 * q0;
            d9 += q1 * q1;
            d10 += q2 * q2;
            d11 += q3 * q3;
            d12 += q4 * q4;
            d13 += q5 * q5;
            d14 += q6 * q6;
            d15 += q7 * q7;
            q0 = x[i] - *(y + 16 * d + i);
            q1 = x[i] - *(y + 17 * d + i);
            q2 = x[i] - *(y + 18 * d + i);
            q3 = x[i] - *(y + 19 * d + i);
            q4 = x[i] - *(y + 20 * d + i);
            q5 = x[i] - *(y + 21 * d + i);
            q6 = x[i] - *(y + 22 * d + i);
            q7 = x[i] - *(y + 23 * d + i);
            d16 += q0 * q0;
            d17 += q1 * q1;
            d18 += q2 * q2;
            d19 += q3 * q3;
            d20 += q4 * q4;
            d21 += q5 * q5;
            d22 += q6 * q6;
            d23 += q7 * q7;
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

/*
 * @brief Compute L2 square between query vector and multiple vectors specified by indices.
 * @param dis Pointer to the output array for storing the results (float).
 * @param x Pointer to the query vector (uint16_t).
 * @param y Pointer to the database vectors (uint16_t).
 * @param ids Pointer to the indices of the database vectors to be used.
 * @param d Dimension of the vectors.
 * @param ny Number of vectors to compute L2 squares with.
 * @param dis_size Length of dis.
 */
int krl_L2sqr_by_idx_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, const int64_t *ids, size_t d, size_t ny)
{
    size_t i = 0;
    const float16_t *__restrict listy[24];

    for (; i + 24 <= ny; i += 24) {
        prefetch_L1(x);
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
        krl_L2sqr_idx_prefetch_batch24_f16f32((const float16_t *)x, listy, d, dis + i);
    }
    if (i + 16 <= ny) {
        prefetch_L1(x);
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
        krl_L2sqr_idx_prefetch_batch16_f16f32((const float16_t *)x, listy, d, dis + i);
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
        krl_L2sqr_idx_batch8_f16f32((const float16_t *)x, listy, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        krl_L2sqr_idx_batch4_f16f32((const float16_t *)x, listy, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        const float16_t *y0 = (const float16_t *)(y + *(ids + i) * d);
        const float16_t *y1 = (const float16_t *)(y + *(ids + i + 1) * d);
        krl_L2sqr_idx_batch2_f16f32((const float16_t *)x, y0, y1, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        krl_L2sqr_f16f32(x, y + d * ids[i], d, &dis[i]);
    }
    return SUCCESS;
}

/*
 * @brief Compute L2 square between query vector and multiple vectors in the database.
 * @param dis Pointer to the output array for storing the results (float).
 * @param x Pointer to the query vector (uint16_t).
 * @param y Pointer to the database vectors (uint16_t).
 * @param ny Number of vectors to compute L2 squares with.
 * @param d Dimension of the vectors.
 */
int krl_L2sqr_ny_f16f32(float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d)
{
    size_t i = 0;

    for (; i + 24 <= ny; i += 24) {
        prefetch_L1(x);
        prefetch_Lx(y + i * d);
        krl_L2sqr_batch24_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
    }
    if (i + 16 <= ny) {
        prefetch_L1(x);
        prefetch_Lx(y + i * d);
        krl_L2sqr_batch16_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        krl_L2sqr_batch8_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_L2sqr_batch4_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_L2sqr_batch2_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        krl_L2sqr_f16f32(x, y + i * d, d, &dis[i]);
    }
    return SUCCESS;
}
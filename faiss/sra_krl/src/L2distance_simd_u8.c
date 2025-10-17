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
 * @brief Compute the L2 square between two int8_t vectors.
 * @param x Pointer to the first vector (uint8_t).
 * @param y Pointer to the second vector (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Stores the computed L2 square result (uint32_t).
 * @param dis_size Length of dis.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
int krl_L2sqr_u8u32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, uint32_t *dis, size_t dis_size)
{
    size_t i;
    uint32_t res;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 64;

    uint32x4_t res1 = vdupq_n_u32(0);
    uint32x4_t res2 = vdupq_n_u32(0);
    uint32x4_t res3 = vdupq_n_u32(0);
    uint32x4_t res4 = vdupq_n_u32(0);
    for (i = 0; i + double_round <= d; i += double_round) {
        const uint8x16_t x8_0 = vld1q_u8(x + i);
        const uint8x16_t x8_1 = vld1q_u8(x + i + 16);
        const uint8x16_t x8_2 = vld1q_u8(x + i + 32);
        const uint8x16_t x8_3 = vld1q_u8(x + i + 48);

        const uint8x16_t y8_0 = vld1q_u8(y + i);
        const uint8x16_t y8_1 = vld1q_u8(y + i + 16);
        const uint8x16_t y8_2 = vld1q_u8(y + i + 32);
        const uint8x16_t y8_3 = vld1q_u8(y + i + 48);

        const uint8x16_t d8_0 = vabdq_u8(x8_0, y8_0);
        const uint8x16_t d8_1 = vabdq_u8(x8_1, y8_1);
        const uint8x16_t d8_2 = vabdq_u8(x8_2, y8_2);
        const uint8x16_t d8_3 = vabdq_u8(x8_3, y8_3);

        res1 = vdotq_u32(res1, d8_0, d8_0);
        res2 = vdotq_u32(res2, d8_1, d8_1);
        res3 = vdotq_u32(res3, d8_2, d8_2);
        res4 = vdotq_u32(res4, d8_3, d8_3);
    }
    for (; i + single_round <= d; i += single_round) {
        const uint8x16_t x8_0 = vld1q_u8(x + i);
        const uint8x16_t y8_0 = vld1q_u8(y + i);

        const uint8x16_t d8_0 = vabdq_u8(x8_0, y8_0);
        res1 = vdotq_u32(res1, d8_0, d8_0);
    }
    res1 = vaddq_u32(res1, res2);
    res3 = vaddq_u32(res3, res4);
    res1 = vaddq_u32(res1, res3);
    res = vaddvq_u32(res1);
    for (; i < d; i++) {
        const int32_t tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    *dis = res;
    return SUCCESS;
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute L2 squares for two vectors with indices.
 * @param x Pointer to the query vector (uint8_t).
 * @param y0 Pointer to the first database vector (uint8_t).
 * @param y1 Pointer to the second database vector (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_batch2_u8f32(
    const uint8_t *x, const uint8_t *__restrict y0, const uint8_t *__restrict y1, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    uint32x4_t res1 = vdupq_n_u32(0);
    uint32x4_t res2 = vdupq_n_u32(0);
    uint32x4_t res3 = vdupq_n_u32(0);
    uint32x4_t res4 = vdupq_n_u32(0);
    for (i = 0; i + double_round <= d; i += double_round) {
        const uint8x16_t x8_0 = vld1q_u8(x + i);
        const uint8x16_t x8_1 = vld1q_u8(x + i + 16);

        const uint8x16_t y8_0 = vld1q_u8(y0 + i);
        const uint8x16_t y8_1 = vld1q_u8(y0 + i + 16);
        const uint8x16_t y8_2 = vld1q_u8(y1 + i);
        const uint8x16_t y8_3 = vld1q_u8(y1 + i + 16);

        const uint8x16_t d8_0 = vabdq_u8(x8_0, y8_0);
        const uint8x16_t d8_1 = vabdq_u8(x8_1, y8_1);
        const uint8x16_t d8_2 = vabdq_u8(x8_0, y8_2);
        const uint8x16_t d8_3 = vabdq_u8(x8_1, y8_3);

        res1 = vdotq_u32(res1, d8_0, d8_0);
        res2 = vdotq_u32(res2, d8_1, d8_1);
        res3 = vdotq_u32(res3, d8_2, d8_2);
        res4 = vdotq_u32(res4, d8_3, d8_3);
    }
    for (; i + single_round <= d; i += single_round) {
        const uint8x16_t x8_0 = vld1q_u8(x + i);
        const uint8x16_t y8_0 = vld1q_u8(y0 + i);
        const uint8x16_t y8_1 = vld1q_u8(y1 + i);

        const uint8x16_t d8_0 = vabdq_u8(x8_0, y8_0);
        const uint8x16_t d8_1 = vabdq_u8(x8_0, y8_1);
        res1 = vdotq_u32(res1, d8_0, d8_0);
        res3 = vdotq_u32(res3, d8_1, d8_1);
    }
    res1 = vaddq_u32(res1, res2);
    res3 = vaddq_u32(res3, res4);
    dis[0] = (float)vaddvq_u32(res1);
    dis[1] = (float)vaddvq_u32(res3);
    for (; i < d; i++) {
        const float tmp0 = x[i] - y0[i];
        const float tmp1 = x[i] - y1[i];
        dis[0] += tmp0 * tmp0;
        dis[1] += tmp1 * tmp1;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute L2 squares for four vectors with indices.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_batch4_u8f32(const uint8_t *x, const uint8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;

    uint32x4_t neon_res1 = vdupq_n_u32(0);
    uint32x4_t neon_res2 = vdupq_n_u32(0);
    uint32x4_t neon_res3 = vdupq_n_u32(0);
    uint32x4_t neon_res4 = vdupq_n_u32(0);

    if (d >= double_round) {
        uint8x16_t neon_query = vld1q_u8(x);
        uint8x16_t neon_base1 = vld1q_u8(y[0]);
        uint8x16_t neon_base2 = vld1q_u8(y[1]);
        uint8x16_t neon_base3 = vld1q_u8(y[2]);
        uint8x16_t neon_base4 = vld1q_u8(y[3]);

        uint8x16_t neon_diff1 = vabdq_u8(neon_base1, neon_query);
        uint8x16_t neon_diff2 = vabdq_u8(neon_base2, neon_query);
        uint8x16_t neon_diff3 = vabdq_u8(neon_base3, neon_query);
        uint8x16_t neon_diff4 = vabdq_u8(neon_base4, neon_query);

        neon_query = vld1q_u8(x + single_round);
        neon_base1 = vld1q_u8(y[0] + single_round);
        neon_base2 = vld1q_u8(y[1] + single_round);
        neon_base3 = vld1q_u8(y[2] + single_round);
        neon_base4 = vld1q_u8(y[3] + single_round);

        neon_res1 = vdotq_u32(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vdotq_u32(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vdotq_u32(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vdotq_u32(neon_res4, neon_diff4, neon_diff4);

        for (i = double_round; i <= d - single_round; i += single_round) {
            neon_diff1 = vabdq_u8(neon_base1, neon_query);
            neon_diff2 = vabdq_u8(neon_base2, neon_query);
            neon_diff3 = vabdq_u8(neon_base3, neon_query);
            neon_diff4 = vabdq_u8(neon_base4, neon_query);

            neon_query = vld1q_u8(x + i);
            neon_base1 = vld1q_u8(y[0] + i);
            neon_base2 = vld1q_u8(y[1] + i);
            neon_base3 = vld1q_u8(y[2] + i);
            neon_base4 = vld1q_u8(y[3] + i);

            neon_res1 = vdotq_u32(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vdotq_u32(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vdotq_u32(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vdotq_u32(neon_res4, neon_diff4, neon_diff4);
        }
        neon_diff1 = vabdq_u8(neon_base1, neon_query);
        neon_diff2 = vabdq_u8(neon_base2, neon_query);
        neon_diff3 = vabdq_u8(neon_base3, neon_query);
        neon_diff4 = vabdq_u8(neon_base4, neon_query);

        neon_res1 = vdotq_u32(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vdotq_u32(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vdotq_u32(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vdotq_u32(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vpaddq_u32(neon_res1, neon_res2);
        neon_res3 = vpaddq_u32(neon_res3, neon_res4);
        neon_res1 = vpaddq_u32(neon_res1, neon_res3);
        vst1q_f32(dis, vcvtq_f32_u32(neon_res1));
    } else if (d >= single_round) {
        uint8x16_t neon_query = vld1q_u8(x);
        uint8x16_t neon_base1 = vld1q_u8(y[0]);
        uint8x16_t neon_base2 = vld1q_u8(y[1]);
        uint8x16_t neon_base3 = vld1q_u8(y[2]);
        uint8x16_t neon_base4 = vld1q_u8(y[3]);

        uint8x16_t neon_diff1 = vabdq_u8(neon_base1, neon_query);
        uint8x16_t neon_diff2 = vabdq_u8(neon_base2, neon_query);
        uint8x16_t neon_diff3 = vabdq_u8(neon_base3, neon_query);
        uint8x16_t neon_diff4 = vabdq_u8(neon_base4, neon_query);

        neon_res1 = vdotq_u32(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vdotq_u32(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vdotq_u32(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vdotq_u32(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vpaddq_u32(neon_res1, neon_res2);
        neon_res3 = vpaddq_u32(neon_res3, neon_res4);
        neon_res1 = vpaddq_u32(neon_res1, neon_res3);

        vst1q_f32(dis, vcvtq_f32_u32(neon_res1));
        i = single_round;
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float q0 = x[i] - *(y[0] + i);
        float q1 = x[i] - *(y[1] + i);
        float q2 = x[i] - *(y[2] + i);
        float q3 = x[i] - *(y[3] + i);
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
 * @brief Compute L2 squares for eight vectors with indices.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_batch8_u8f32(const uint8_t *x, const uint8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;

    uint32x4_t neon_res1 = vdupq_n_u32(0);
    uint32x4_t neon_res2 = vdupq_n_u32(0);
    uint32x4_t neon_res3 = vdupq_n_u32(0);
    uint32x4_t neon_res4 = vdupq_n_u32(0);
    uint32x4_t neon_res5 = vdupq_n_u32(0);
    uint32x4_t neon_res6 = vdupq_n_u32(0);
    uint32x4_t neon_res7 = vdupq_n_u32(0);
    uint32x4_t neon_res8 = vdupq_n_u32(0);
    if (d >= double_round) {
        uint8x16_t neon_query = vld1q_u8(x);
        uint8x16_t neon_base1 = vld1q_u8(y[0]);
        uint8x16_t neon_base2 = vld1q_u8(y[1]);
        uint8x16_t neon_base3 = vld1q_u8(y[2]);
        uint8x16_t neon_base4 = vld1q_u8(y[3]);
        uint8x16_t neon_base5 = vld1q_u8(y[4]);
        uint8x16_t neon_base6 = vld1q_u8(y[5]);
        uint8x16_t neon_base7 = vld1q_u8(y[6]);
        uint8x16_t neon_base8 = vld1q_u8(y[7]);

        uint8x16_t neon_diff1 = vabdq_u8(neon_base1, neon_query);
        uint8x16_t neon_diff2 = vabdq_u8(neon_base2, neon_query);
        uint8x16_t neon_diff3 = vabdq_u8(neon_base3, neon_query);
        uint8x16_t neon_diff4 = vabdq_u8(neon_base4, neon_query);
        uint8x16_t neon_diff5 = vabdq_u8(neon_base5, neon_query);
        uint8x16_t neon_diff6 = vabdq_u8(neon_base6, neon_query);
        uint8x16_t neon_diff7 = vabdq_u8(neon_base7, neon_query);
        uint8x16_t neon_diff8 = vabdq_u8(neon_base8, neon_query);

        neon_query = vld1q_u8(x + single_round);
        neon_base1 = vld1q_u8(y[0] + single_round);
        neon_base2 = vld1q_u8(y[1] + single_round);
        neon_base3 = vld1q_u8(y[2] + single_round);
        neon_base4 = vld1q_u8(y[3] + single_round);
        neon_base5 = vld1q_u8(y[4] + single_round);
        neon_base6 = vld1q_u8(y[5] + single_round);
        neon_base7 = vld1q_u8(y[6] + single_round);
        neon_base8 = vld1q_u8(y[7] + single_round);

        neon_res1 = vdotq_u32(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vdotq_u32(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vdotq_u32(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vdotq_u32(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vdotq_u32(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vdotq_u32(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vdotq_u32(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vdotq_u32(neon_res8, neon_diff8, neon_diff8);

        for (i = double_round; i <= d - single_round; i += single_round) {
            neon_diff1 = vabdq_u8(neon_base1, neon_query);
            neon_diff2 = vabdq_u8(neon_base2, neon_query);
            neon_diff3 = vabdq_u8(neon_base3, neon_query);
            neon_diff4 = vabdq_u8(neon_base4, neon_query);
            neon_diff5 = vabdq_u8(neon_base5, neon_query);
            neon_diff6 = vabdq_u8(neon_base6, neon_query);
            neon_diff7 = vabdq_u8(neon_base7, neon_query);
            neon_diff8 = vabdq_u8(neon_base8, neon_query);

            neon_query = vld1q_u8(x + i);
            neon_base1 = vld1q_u8(y[0] + i);
            neon_base2 = vld1q_u8(y[1] + i);
            neon_base3 = vld1q_u8(y[2] + i);
            neon_base4 = vld1q_u8(y[3] + i);
            neon_base5 = vld1q_u8(y[4] + i);
            neon_base6 = vld1q_u8(y[5] + i);
            neon_base7 = vld1q_u8(y[6] + i);
            neon_base8 = vld1q_u8(y[7] + i);

            neon_res1 = vdotq_u32(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vdotq_u32(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vdotq_u32(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vdotq_u32(neon_res4, neon_diff4, neon_diff4);
            neon_res5 = vdotq_u32(neon_res5, neon_diff5, neon_diff5);
            neon_res6 = vdotq_u32(neon_res6, neon_diff6, neon_diff6);
            neon_res7 = vdotq_u32(neon_res7, neon_diff7, neon_diff7);
            neon_res8 = vdotq_u32(neon_res8, neon_diff8, neon_diff8);
        }
        neon_diff1 = vabdq_u8(neon_base1, neon_query);
        neon_diff2 = vabdq_u8(neon_base2, neon_query);
        neon_diff3 = vabdq_u8(neon_base3, neon_query);
        neon_diff4 = vabdq_u8(neon_base4, neon_query);
        neon_diff5 = vabdq_u8(neon_base5, neon_query);
        neon_diff6 = vabdq_u8(neon_base6, neon_query);
        neon_diff7 = vabdq_u8(neon_base7, neon_query);
        neon_diff8 = vabdq_u8(neon_base8, neon_query);

        neon_res1 = vdotq_u32(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vdotq_u32(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vdotq_u32(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vdotq_u32(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vdotq_u32(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vdotq_u32(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vdotq_u32(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vdotq_u32(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vpaddq_u32(neon_res1, neon_res2);
        neon_res3 = vpaddq_u32(neon_res3, neon_res4);
        neon_res5 = vpaddq_u32(neon_res5, neon_res6);
        neon_res7 = vpaddq_u32(neon_res7, neon_res8);
        neon_res1 = vpaddq_u32(neon_res1, neon_res3);
        neon_res5 = vpaddq_u32(neon_res5, neon_res7);

        vst1q_f32(dis, vcvtq_f32_u32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res5));
    } else if (d >= single_round) {
        uint8x16_t neon_query = vld1q_u8(x);
        uint8x16_t neon_base1 = vld1q_u8(y[0]);
        uint8x16_t neon_base2 = vld1q_u8(y[1]);
        uint8x16_t neon_base3 = vld1q_u8(y[2]);
        uint8x16_t neon_base4 = vld1q_u8(y[3]);
        uint8x16_t neon_base5 = vld1q_u8(y[4]);
        uint8x16_t neon_base6 = vld1q_u8(y[5]);
        uint8x16_t neon_base7 = vld1q_u8(y[6]);
        uint8x16_t neon_base8 = vld1q_u8(y[7]);

        uint8x16_t neon_diff1 = vabdq_u8(neon_base1, neon_query);
        uint8x16_t neon_diff2 = vabdq_u8(neon_base2, neon_query);
        uint8x16_t neon_diff3 = vabdq_u8(neon_base3, neon_query);
        uint8x16_t neon_diff4 = vabdq_u8(neon_base4, neon_query);
        uint8x16_t neon_diff5 = vabdq_u8(neon_base5, neon_query);
        uint8x16_t neon_diff6 = vabdq_u8(neon_base6, neon_query);
        uint8x16_t neon_diff7 = vabdq_u8(neon_base7, neon_query);
        uint8x16_t neon_diff8 = vabdq_u8(neon_base8, neon_query);

        neon_res1 = vdotq_u32(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vdotq_u32(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vdotq_u32(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vdotq_u32(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vdotq_u32(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vdotq_u32(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vdotq_u32(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vdotq_u32(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vpaddq_u32(neon_res1, neon_res2);
        neon_res3 = vpaddq_u32(neon_res3, neon_res4);
        neon_res5 = vpaddq_u32(neon_res5, neon_res6);
        neon_res7 = vpaddq_u32(neon_res7, neon_res8);
        neon_res1 = vpaddq_u32(neon_res1, neon_res3);
        neon_res5 = vpaddq_u32(neon_res5, neon_res7);

        vst1q_f32(dis, vcvtq_f32_u32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res5));
        i = single_round;
    } else {
        for (int i = 0; i < 8; i++) {
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
 * @brief Compute L2 squares for sixteen vectors with indices and prefetch optimization.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_prefetch_batch16_u8f32(
    const uint8_t *x, const uint8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;   /* 128 / 8 */
    constexpr size_t prefetch_round = 64; /* 4 * single_round */

    uint32x4_t neon_res1 = vdupq_n_u32(0);
    uint32x4_t neon_res2 = vdupq_n_u32(0);
    uint32x4_t neon_res3 = vdupq_n_u32(0);
    uint32x4_t neon_res4 = vdupq_n_u32(0);
    uint32x4_t neon_res5 = vdupq_n_u32(0);
    uint32x4_t neon_res6 = vdupq_n_u32(0);
    uint32x4_t neon_res7 = vdupq_n_u32(0);
    uint32x4_t neon_res8 = vdupq_n_u32(0);
    uint32x4_t neon_res9 = vdupq_n_u32(0);
    uint32x4_t neon_res10 = vdupq_n_u32(0);
    uint32x4_t neon_res11 = vdupq_n_u32(0);
    uint32x4_t neon_res12 = vdupq_n_u32(0);
    uint32x4_t neon_res13 = vdupq_n_u32(0);
    uint32x4_t neon_res14 = vdupq_n_u32(0);
    uint32x4_t neon_res15 = vdupq_n_u32(0);
    uint32x4_t neon_res16 = vdupq_n_u32(0);

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
            for (size_t j = 0; j < prefetch_round; j += single_round) {
                const uint8x16_t neon_query = vld1q_u8(x + i + j);
                uint8x16_t neon_base1 = vld1q_u8(y[0] + i + j);
                uint8x16_t neon_base2 = vld1q_u8(y[1] + i + j);
                uint8x16_t neon_base3 = vld1q_u8(y[2] + i + j);
                uint8x16_t neon_base4 = vld1q_u8(y[3] + i + j);
                uint8x16_t neon_base5 = vld1q_u8(y[4] + i + j);
                uint8x16_t neon_base6 = vld1q_u8(y[5] + i + j);
                uint8x16_t neon_base7 = vld1q_u8(y[6] + i + j);
                uint8x16_t neon_base8 = vld1q_u8(y[7] + i + j);

                neon_base1 = vabdq_u8(neon_base1, neon_query);
                neon_base2 = vabdq_u8(neon_base2, neon_query);
                neon_base3 = vabdq_u8(neon_base3, neon_query);
                neon_base4 = vabdq_u8(neon_base4, neon_query);
                neon_base5 = vabdq_u8(neon_base5, neon_query);
                neon_base6 = vabdq_u8(neon_base6, neon_query);
                neon_base7 = vabdq_u8(neon_base7, neon_query);
                neon_base8 = vabdq_u8(neon_base8, neon_query);

                neon_res1 = vdotq_u32(neon_res1, neon_base1, neon_base1);
                neon_res2 = vdotq_u32(neon_res2, neon_base2, neon_base2);
                neon_res3 = vdotq_u32(neon_res3, neon_base3, neon_base3);
                neon_res4 = vdotq_u32(neon_res4, neon_base4, neon_base4);
                neon_res5 = vdotq_u32(neon_res5, neon_base5, neon_base5);
                neon_res6 = vdotq_u32(neon_res6, neon_base6, neon_base6);
                neon_res7 = vdotq_u32(neon_res7, neon_base7, neon_base7);
                neon_res8 = vdotq_u32(neon_res8, neon_base8, neon_base8);

                neon_base1 = vld1q_u8(y[8] + i + j);
                neon_base2 = vld1q_u8(y[9] + i + j);
                neon_base3 = vld1q_u8(y[10] + i + j);
                neon_base4 = vld1q_u8(y[11] + i + j);
                neon_base5 = vld1q_u8(y[12] + i + j);
                neon_base6 = vld1q_u8(y[13] + i + j);
                neon_base7 = vld1q_u8(y[14] + i + j);
                neon_base8 = vld1q_u8(y[15] + i + j);

                neon_base1 = vabdq_u8(neon_base1, neon_query);
                neon_base2 = vabdq_u8(neon_base2, neon_query);
                neon_base3 = vabdq_u8(neon_base3, neon_query);
                neon_base4 = vabdq_u8(neon_base4, neon_query);
                neon_base5 = vabdq_u8(neon_base5, neon_query);
                neon_base6 = vabdq_u8(neon_base6, neon_query);
                neon_base7 = vabdq_u8(neon_base7, neon_query);
                neon_base8 = vabdq_u8(neon_base8, neon_query);

                neon_res9 = vdotq_u32(neon_res9, neon_base1, neon_base1);
                neon_res10 = vdotq_u32(neon_res10, neon_base2, neon_base2);
                neon_res11 = vdotq_u32(neon_res11, neon_base3, neon_base3);
                neon_res12 = vdotq_u32(neon_res12, neon_base4, neon_base4);
                neon_res13 = vdotq_u32(neon_res13, neon_base5, neon_base5);
                neon_res14 = vdotq_u32(neon_res14, neon_base6, neon_base6);
                neon_res15 = vdotq_u32(neon_res15, neon_base7, neon_base7);
                neon_res16 = vdotq_u32(neon_res16, neon_base8, neon_base8);
            }
        }
        for (; i + single_round <= d; i += single_round) {
            const uint8x16_t neon_query = vld1q_u8(x + i);
            uint8x16_t neon_base1 = vld1q_u8(y[0] + i);
            uint8x16_t neon_base2 = vld1q_u8(y[1] + i);
            uint8x16_t neon_base3 = vld1q_u8(y[2] + i);
            uint8x16_t neon_base4 = vld1q_u8(y[3] + i);
            uint8x16_t neon_base5 = vld1q_u8(y[4] + i);
            uint8x16_t neon_base6 = vld1q_u8(y[5] + i);
            uint8x16_t neon_base7 = vld1q_u8(y[6] + i);
            uint8x16_t neon_base8 = vld1q_u8(y[7] + i);

            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_base5 = vabdq_u8(neon_base5, neon_query);
            neon_base6 = vabdq_u8(neon_base6, neon_query);
            neon_base7 = vabdq_u8(neon_base7, neon_query);
            neon_base8 = vabdq_u8(neon_base8, neon_query);

            neon_res1 = vdotq_u32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vdotq_u32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vdotq_u32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vdotq_u32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vdotq_u32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vdotq_u32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vdotq_u32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vdotq_u32(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_u8(y[8] + i);
            neon_base2 = vld1q_u8(y[9] + i);
            neon_base3 = vld1q_u8(y[10] + i);
            neon_base4 = vld1q_u8(y[11] + i);
            neon_base5 = vld1q_u8(y[12] + i);
            neon_base6 = vld1q_u8(y[13] + i);
            neon_base7 = vld1q_u8(y[14] + i);
            neon_base8 = vld1q_u8(y[15] + i);

            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_base5 = vabdq_u8(neon_base5, neon_query);
            neon_base6 = vabdq_u8(neon_base6, neon_query);
            neon_base7 = vabdq_u8(neon_base7, neon_query);
            neon_base8 = vabdq_u8(neon_base8, neon_query);

            neon_res9 = vdotq_u32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vdotq_u32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vdotq_u32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vdotq_u32(neon_res12, neon_base4, neon_base4);
            neon_res13 = vdotq_u32(neon_res13, neon_base5, neon_base5);
            neon_res14 = vdotq_u32(neon_res14, neon_base6, neon_base6);
            neon_res15 = vdotq_u32(neon_res15, neon_base7, neon_base7);
            neon_res16 = vdotq_u32(neon_res16, neon_base8, neon_base8);
        }
        neon_res1 = vpaddq_u32(neon_res1, neon_res2);
        neon_res3 = vpaddq_u32(neon_res3, neon_res4);
        neon_res5 = vpaddq_u32(neon_res5, neon_res6);
        neon_res7 = vpaddq_u32(neon_res7, neon_res8);
        neon_res9 = vpaddq_u32(neon_res9, neon_res10);
        neon_res11 = vpaddq_u32(neon_res11, neon_res12);
        neon_res13 = vpaddq_u32(neon_res13, neon_res14);
        neon_res15 = vpaddq_u32(neon_res15, neon_res16);
        neon_res1 = vpaddq_u32(neon_res1, neon_res3);
        neon_res5 = vpaddq_u32(neon_res5, neon_res7);
        neon_res9 = vpaddq_u32(neon_res9, neon_res11);
        neon_res13 = vpaddq_u32(neon_res13, neon_res15);

        vst1q_f32(dis, vcvtq_f32_u32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_u32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_u32(neon_res13));
    } else {
        for (i = 0; i + single_round <= d; i += single_round) {
            const uint8x16_t neon_query = vld1q_u8(x + i);
            uint8x16_t neon_base1 = vld1q_u8(y[0] + i);
            uint8x16_t neon_base2 = vld1q_u8(y[1] + i);
            uint8x16_t neon_base3 = vld1q_u8(y[2] + i);
            uint8x16_t neon_base4 = vld1q_u8(y[3] + i);
            uint8x16_t neon_base5 = vld1q_u8(y[4] + i);
            uint8x16_t neon_base6 = vld1q_u8(y[5] + i);
            uint8x16_t neon_base7 = vld1q_u8(y[6] + i);
            uint8x16_t neon_base8 = vld1q_u8(y[7] + i);

            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_base5 = vabdq_u8(neon_base5, neon_query);
            neon_base6 = vabdq_u8(neon_base6, neon_query);
            neon_base7 = vabdq_u8(neon_base7, neon_query);
            neon_base8 = vabdq_u8(neon_base8, neon_query);

            neon_res1 = vdotq_u32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vdotq_u32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vdotq_u32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vdotq_u32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vdotq_u32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vdotq_u32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vdotq_u32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vdotq_u32(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_u8(y[8] + i);
            neon_base2 = vld1q_u8(y[9] + i);
            neon_base3 = vld1q_u8(y[10] + i);
            neon_base4 = vld1q_u8(y[11] + i);
            neon_base5 = vld1q_u8(y[12] + i);
            neon_base6 = vld1q_u8(y[13] + i);
            neon_base7 = vld1q_u8(y[14] + i);
            neon_base8 = vld1q_u8(y[15] + i);

            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_base5 = vabdq_u8(neon_base5, neon_query);
            neon_base6 = vabdq_u8(neon_base6, neon_query);
            neon_base7 = vabdq_u8(neon_base7, neon_query);
            neon_base8 = vabdq_u8(neon_base8, neon_query);

            neon_res9 = vdotq_u32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vdotq_u32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vdotq_u32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vdotq_u32(neon_res12, neon_base4, neon_base4);
            neon_res13 = vdotq_u32(neon_res13, neon_base5, neon_base5);
            neon_res14 = vdotq_u32(neon_res14, neon_base6, neon_base6);
            neon_res15 = vdotq_u32(neon_res15, neon_base7, neon_base7);
            neon_res16 = vdotq_u32(neon_res16, neon_base8, neon_base8);
        }
        neon_res1 = vpaddq_u32(neon_res1, neon_res2);
        neon_res3 = vpaddq_u32(neon_res3, neon_res4);
        neon_res5 = vpaddq_u32(neon_res5, neon_res6);
        neon_res7 = vpaddq_u32(neon_res7, neon_res8);
        neon_res9 = vpaddq_u32(neon_res9, neon_res10);
        neon_res11 = vpaddq_u32(neon_res11, neon_res12);
        neon_res13 = vpaddq_u32(neon_res13, neon_res14);
        neon_res15 = vpaddq_u32(neon_res15, neon_res16);
        neon_res1 = vpaddq_u32(neon_res1, neon_res3);
        neon_res5 = vpaddq_u32(neon_res5, neon_res7);
        neon_res9 = vpaddq_u32(neon_res9, neon_res11);
        neon_res13 = vpaddq_u32(neon_res13, neon_res15);

        vst1q_f32(dis, vcvtq_f32_u32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_u32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_u32(neon_res13));
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
 * @brief Compute L2 squares for twenty-four vectors with indices and prefetch optimization.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_prefetch_batch24_u8f32(
    const uint8_t *x, const uint8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    const size_t single_round = 16;   /* 128 / 8 */
    const size_t prefetch_round = 64; /* 4 * single_round */

    uint32x4_t neon_res1 = vdupq_n_u32(0);
    uint32x4_t neon_res2 = vdupq_n_u32(0);
    uint32x4_t neon_res3 = vdupq_n_u32(0);
    uint32x4_t neon_res4 = vdupq_n_u32(0);
    uint32x4_t neon_res5 = vdupq_n_u32(0);
    uint32x4_t neon_res6 = vdupq_n_u32(0);
    uint32x4_t neon_res7 = vdupq_n_u32(0);
    uint32x4_t neon_res8 = vdupq_n_u32(0);
    uint32x4_t neon_res9 = vdupq_n_u32(0);
    uint32x4_t neon_res10 = vdupq_n_u32(0);
    uint32x4_t neon_res11 = vdupq_n_u32(0);
    uint32x4_t neon_res12 = vdupq_n_u32(0);
    uint32x4_t neon_res13 = vdupq_n_u32(0);
    uint32x4_t neon_res14 = vdupq_n_u32(0);
    uint32x4_t neon_res15 = vdupq_n_u32(0);
    uint32x4_t neon_res16 = vdupq_n_u32(0);
    uint32x4_t neon_res17 = vdupq_n_u32(0);
    uint32x4_t neon_res18 = vdupq_n_u32(0);
    uint32x4_t neon_res19 = vdupq_n_u32(0);
    uint32x4_t neon_res20 = vdupq_n_u32(0);
    uint32x4_t neon_res21 = vdupq_n_u32(0);
    uint32x4_t neon_res22 = vdupq_n_u32(0);
    uint32x4_t neon_res23 = vdupq_n_u32(0);
    uint32x4_t neon_res24 = vdupq_n_u32(0);

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
                const uint8x16_t neon_query = vld1q_u8(x + i + j);
                uint8x16_t neon_base1 = vld1q_u8(y[0] + i + j);
                uint8x16_t neon_base2 = vld1q_u8(y[1] + i + j);
                uint8x16_t neon_base3 = vld1q_u8(y[2] + i + j);
                uint8x16_t neon_base4 = vld1q_u8(y[3] + i + j);
                neon_base1 = vabdq_u8(neon_base1, neon_query);
                neon_base2 = vabdq_u8(neon_base2, neon_query);
                neon_base3 = vabdq_u8(neon_base3, neon_query);
                neon_base4 = vabdq_u8(neon_base4, neon_query);
                neon_res1 = vdotq_u32(neon_res1, neon_base1, neon_base1);
                neon_res2 = vdotq_u32(neon_res2, neon_base2, neon_base2);
                neon_res3 = vdotq_u32(neon_res3, neon_base3, neon_base3);
                neon_res4 = vdotq_u32(neon_res4, neon_base4, neon_base4);

                neon_base1 = vld1q_u8(y[4] + i + j);
                neon_base2 = vld1q_u8(y[5] + i + j);
                neon_base3 = vld1q_u8(y[6] + i + j);
                neon_base4 = vld1q_u8(y[7] + i + j);
                neon_base1 = vabdq_u8(neon_base1, neon_query);
                neon_base2 = vabdq_u8(neon_base2, neon_query);
                neon_base3 = vabdq_u8(neon_base3, neon_query);
                neon_base4 = vabdq_u8(neon_base4, neon_query);
                neon_res5 = vdotq_u32(neon_res5, neon_base1, neon_base1);
                neon_res6 = vdotq_u32(neon_res6, neon_base2, neon_base2);
                neon_res7 = vdotq_u32(neon_res7, neon_base3, neon_base3);
                neon_res8 = vdotq_u32(neon_res8, neon_base4, neon_base4);

                neon_base1 = vld1q_u8(y[8] + i + j);
                neon_base2 = vld1q_u8(y[9] + i + j);
                neon_base3 = vld1q_u8(y[10] + i + j);
                neon_base4 = vld1q_u8(y[11] + i + j);
                neon_base1 = vabdq_u8(neon_base1, neon_query);
                neon_base2 = vabdq_u8(neon_base2, neon_query);
                neon_base3 = vabdq_u8(neon_base3, neon_query);
                neon_base4 = vabdq_u8(neon_base4, neon_query);
                neon_res9 = vdotq_u32(neon_res9, neon_base1, neon_base1);
                neon_res10 = vdotq_u32(neon_res10, neon_base2, neon_base2);
                neon_res11 = vdotq_u32(neon_res11, neon_base3, neon_base3);
                neon_res12 = vdotq_u32(neon_res12, neon_base4, neon_base4);

                neon_base1 = vld1q_u8(y[12] + i + j);
                neon_base2 = vld1q_u8(y[13] + i + j);
                neon_base3 = vld1q_u8(y[14] + i + j);
                neon_base4 = vld1q_u8(y[15] + i + j);
                neon_base1 = vabdq_u8(neon_base1, neon_query);
                neon_base2 = vabdq_u8(neon_base2, neon_query);
                neon_base3 = vabdq_u8(neon_base3, neon_query);
                neon_base4 = vabdq_u8(neon_base4, neon_query);
                neon_res13 = vdotq_u32(neon_res13, neon_base1, neon_base1);
                neon_res14 = vdotq_u32(neon_res14, neon_base2, neon_base2);
                neon_res15 = vdotq_u32(neon_res15, neon_base3, neon_base3);
                neon_res16 = vdotq_u32(neon_res16, neon_base4, neon_base4);

                neon_base1 = vld1q_u8(y[16] + i + j);
                neon_base2 = vld1q_u8(y[17] + i + j);
                neon_base3 = vld1q_u8(y[18] + i + j);
                neon_base4 = vld1q_u8(y[19] + i + j);
                neon_base1 = vabdq_u8(neon_base1, neon_query);
                neon_base2 = vabdq_u8(neon_base2, neon_query);
                neon_base3 = vabdq_u8(neon_base3, neon_query);
                neon_base4 = vabdq_u8(neon_base4, neon_query);
                neon_res17 = vdotq_u32(neon_res17, neon_base1, neon_base1);
                neon_res18 = vdotq_u32(neon_res18, neon_base2, neon_base2);
                neon_res19 = vdotq_u32(neon_res19, neon_base3, neon_base3);
                neon_res20 = vdotq_u32(neon_res20, neon_base4, neon_base4);

                neon_base1 = vld1q_u8(y[20] + i + j);
                neon_base2 = vld1q_u8(y[21] + i + j);
                neon_base3 = vld1q_u8(y[22] + i + j);
                neon_base4 = vld1q_u8(y[23] + i + j);
                neon_base1 = vabdq_u8(neon_base1, neon_query);
                neon_base2 = vabdq_u8(neon_base2, neon_query);
                neon_base3 = vabdq_u8(neon_base3, neon_query);
                neon_base4 = vabdq_u8(neon_base4, neon_query);
                neon_res21 = vdotq_u32(neon_res21, neon_base1, neon_base1);
                neon_res22 = vdotq_u32(neon_res22, neon_base2, neon_base2);
                neon_res23 = vdotq_u32(neon_res23, neon_base3, neon_base3);
                neon_res24 = vdotq_u32(neon_res24, neon_base4, neon_base4);
            }
        }
        for (; i + single_round <= d; i += single_round) {
            const uint8x16_t neon_query = vld1q_u8(x + i);
            uint8x16_t neon_base1 = vld1q_u8(y[0] + i);
            uint8x16_t neon_base2 = vld1q_u8(y[1] + i);
            uint8x16_t neon_base3 = vld1q_u8(y[2] + i);
            uint8x16_t neon_base4 = vld1q_u8(y[3] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res1 = vdotq_u32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vdotq_u32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vdotq_u32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vdotq_u32(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[4] + i);
            neon_base2 = vld1q_u8(y[5] + i);
            neon_base3 = vld1q_u8(y[6] + i);
            neon_base4 = vld1q_u8(y[7] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res5 = vdotq_u32(neon_res5, neon_base1, neon_base1);
            neon_res6 = vdotq_u32(neon_res6, neon_base2, neon_base2);
            neon_res7 = vdotq_u32(neon_res7, neon_base3, neon_base3);
            neon_res8 = vdotq_u32(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[8] + i);
            neon_base2 = vld1q_u8(y[9] + i);
            neon_base3 = vld1q_u8(y[10] + i);
            neon_base4 = vld1q_u8(y[11] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res9 = vdotq_u32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vdotq_u32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vdotq_u32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vdotq_u32(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[12] + i);
            neon_base2 = vld1q_u8(y[13] + i);
            neon_base3 = vld1q_u8(y[14] + i);
            neon_base4 = vld1q_u8(y[15] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res13 = vdotq_u32(neon_res13, neon_base1, neon_base1);
            neon_res14 = vdotq_u32(neon_res14, neon_base2, neon_base2);
            neon_res15 = vdotq_u32(neon_res15, neon_base3, neon_base3);
            neon_res16 = vdotq_u32(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[16] + i);
            neon_base2 = vld1q_u8(y[17] + i);
            neon_base3 = vld1q_u8(y[18] + i);
            neon_base4 = vld1q_u8(y[19] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res17 = vdotq_u32(neon_res17, neon_base1, neon_base1);
            neon_res18 = vdotq_u32(neon_res18, neon_base2, neon_base2);
            neon_res19 = vdotq_u32(neon_res19, neon_base3, neon_base3);
            neon_res20 = vdotq_u32(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[20] + i);
            neon_base2 = vld1q_u8(y[21] + i);
            neon_base3 = vld1q_u8(y[22] + i);
            neon_base4 = vld1q_u8(y[23] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res21 = vdotq_u32(neon_res21, neon_base1, neon_base1);
            neon_res22 = vdotq_u32(neon_res22, neon_base2, neon_base2);
            neon_res23 = vdotq_u32(neon_res23, neon_base3, neon_base3);
            neon_res24 = vdotq_u32(neon_res24, neon_base4, neon_base4);
        }
        neon_res1 = vpaddq_u32(neon_res1, neon_res2);
        neon_res3 = vpaddq_u32(neon_res3, neon_res4);
        neon_res5 = vpaddq_u32(neon_res5, neon_res6);
        neon_res7 = vpaddq_u32(neon_res7, neon_res8);
        neon_res9 = vpaddq_u32(neon_res9, neon_res10);
        neon_res11 = vpaddq_u32(neon_res11, neon_res12);
        neon_res13 = vpaddq_u32(neon_res13, neon_res14);
        neon_res15 = vpaddq_u32(neon_res15, neon_res16);
        neon_res17 = vpaddq_u32(neon_res17, neon_res18);
        neon_res19 = vpaddq_u32(neon_res19, neon_res20);
        neon_res21 = vpaddq_u32(neon_res21, neon_res22);
        neon_res23 = vpaddq_u32(neon_res23, neon_res24);
        neon_res1 = vpaddq_u32(neon_res1, neon_res3);
        neon_res5 = vpaddq_u32(neon_res5, neon_res7);
        neon_res9 = vpaddq_u32(neon_res9, neon_res11);
        neon_res13 = vpaddq_u32(neon_res13, neon_res15);
        neon_res17 = vpaddq_u32(neon_res17, neon_res19);
        neon_res21 = vpaddq_u32(neon_res21, neon_res23);

        vst1q_f32(dis, vcvtq_f32_u32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_u32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_u32(neon_res13));
        vst1q_f32(dis + 16, vcvtq_f32_u32(neon_res17));
        vst1q_f32(dis + 20, vcvtq_f32_u32(neon_res21));
    } else {
        for (i = 0; i + single_round <= d; i += single_round) {
            const uint8x16_t neon_query = vld1q_u8(x + i);
            uint8x16_t neon_base1 = vld1q_u8(y[0] + i);
            uint8x16_t neon_base2 = vld1q_u8(y[1] + i);
            uint8x16_t neon_base3 = vld1q_u8(y[2] + i);
            uint8x16_t neon_base4 = vld1q_u8(y[3] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res1 = vdotq_u32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vdotq_u32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vdotq_u32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vdotq_u32(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[4] + i);
            neon_base2 = vld1q_u8(y[5] + i);
            neon_base3 = vld1q_u8(y[6] + i);
            neon_base4 = vld1q_u8(y[7] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res5 = vdotq_u32(neon_res5, neon_base1, neon_base1);
            neon_res6 = vdotq_u32(neon_res6, neon_base2, neon_base2);
            neon_res7 = vdotq_u32(neon_res7, neon_base3, neon_base3);
            neon_res8 = vdotq_u32(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[8] + i);
            neon_base2 = vld1q_u8(y[9] + i);
            neon_base3 = vld1q_u8(y[10] + i);
            neon_base4 = vld1q_u8(y[11] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res9 = vdotq_u32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vdotq_u32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vdotq_u32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vdotq_u32(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[12] + i);
            neon_base2 = vld1q_u8(y[13] + i);
            neon_base3 = vld1q_u8(y[14] + i);
            neon_base4 = vld1q_u8(y[15] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res13 = vdotq_u32(neon_res13, neon_base1, neon_base1);
            neon_res14 = vdotq_u32(neon_res14, neon_base2, neon_base2);
            neon_res15 = vdotq_u32(neon_res15, neon_base3, neon_base3);
            neon_res16 = vdotq_u32(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[16] + i);
            neon_base2 = vld1q_u8(y[17] + i);
            neon_base3 = vld1q_u8(y[18] + i);
            neon_base4 = vld1q_u8(y[19] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res17 = vdotq_u32(neon_res17, neon_base1, neon_base1);
            neon_res18 = vdotq_u32(neon_res18, neon_base2, neon_base2);
            neon_res19 = vdotq_u32(neon_res19, neon_base3, neon_base3);
            neon_res20 = vdotq_u32(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_u8(y[20] + i);
            neon_base2 = vld1q_u8(y[21] + i);
            neon_base3 = vld1q_u8(y[22] + i);
            neon_base4 = vld1q_u8(y[23] + i);
            neon_base1 = vabdq_u8(neon_base1, neon_query);
            neon_base2 = vabdq_u8(neon_base2, neon_query);
            neon_base3 = vabdq_u8(neon_base3, neon_query);
            neon_base4 = vabdq_u8(neon_base4, neon_query);
            neon_res21 = vdotq_u32(neon_res21, neon_base1, neon_base1);
            neon_res22 = vdotq_u32(neon_res22, neon_base2, neon_base2);
            neon_res23 = vdotq_u32(neon_res23, neon_base3, neon_base3);
            neon_res24 = vdotq_u32(neon_res24, neon_base4, neon_base4);
        }
        neon_res1 = vpaddq_u32(neon_res1, neon_res2);
        neon_res3 = vpaddq_u32(neon_res3, neon_res4);
        neon_res5 = vpaddq_u32(neon_res5, neon_res6);
        neon_res7 = vpaddq_u32(neon_res7, neon_res8);
        neon_res9 = vpaddq_u32(neon_res9, neon_res10);
        neon_res11 = vpaddq_u32(neon_res11, neon_res12);
        neon_res13 = vpaddq_u32(neon_res13, neon_res14);
        neon_res15 = vpaddq_u32(neon_res15, neon_res16);
        neon_res17 = vpaddq_u32(neon_res17, neon_res18);
        neon_res19 = vpaddq_u32(neon_res19, neon_res20);
        neon_res21 = vpaddq_u32(neon_res21, neon_res22);
        neon_res23 = vpaddq_u32(neon_res23, neon_res24);
        neon_res1 = vpaddq_u32(neon_res1, neon_res3);
        neon_res5 = vpaddq_u32(neon_res5, neon_res7);
        neon_res9 = vpaddq_u32(neon_res9, neon_res11);
        neon_res13 = vpaddq_u32(neon_res13, neon_res15);
        neon_res17 = vpaddq_u32(neon_res17, neon_res19);
        neon_res21 = vpaddq_u32(neon_res21, neon_res23);

        vst1q_f32(dis, vcvtq_f32_u32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_u32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_u32(neon_res13));
        vst1q_f32(dis + 16, vcvtq_f32_u32(neon_res17));
        vst1q_f32(dis + 20, vcvtq_f32_u32(neon_res21));
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
 * @brief Compute L2 squares for two vectors with 8-bit integer precision and 32-bit integer results.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (uint32_t).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch2_u8u32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, uint32_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;

    if (likely(d >= single_round)) {
        uint32x4_t res00 = vdupq_n_u32(0);
        uint32x4_t res01 = vdupq_n_u32(0);
        uint32x4_t res10 = vdupq_n_u32(0);
        uint32x4_t res11 = vdupq_n_u32(0);
        for (i = 0; i + double_round <= d; i += double_round) {
            const uint8x16_t x0 = vld1q_u8(x + i);
            const uint8x16_t x1 = vld1q_u8(x + i + 16);
            const uint8x16_t y00 = vld1q_u8(y + i);
            const uint8x16_t y01 = vld1q_u8(y + i + 16);
            const uint8x16_t y10 = vld1q_u8(y + i + d);
            const uint8x16_t y11 = vld1q_u8(y + i + d + 16);

            const uint8x16_t d00 = vabdq_u8(x0, y00);
            res00 = vdotq_u32(res00, d00, d00);
            const uint8x16_t d01 = vabdq_u8(x1, y01);
            res01 = vdotq_u32(res01, d01, d01);
            const uint8x16_t d10 = vabdq_u8(x0, y10);
            res10 = vdotq_u32(res10, d10, d10);
            const uint8x16_t d11 = vabdq_u8(x1, y11);
            res11 = vdotq_u32(res11, d11, d11);
        }
        for (; i <= d - single_round; i += single_round) {
            const uint8x16_t x0 = vld1q_u8(x + i);
            const uint8x16_t y00 = vld1q_u8(y + i);
            const uint8x16_t y10 = vld1q_u8(y + i + d);

            const uint8x16_t d00 = vabdq_u8(x0, y00);
            res00 = vdotq_u32(res00, d00, d00);
            const uint8x16_t d10 = vabdq_u8(x0, y10);
            res10 = vdotq_u32(res10, d10, d10);
        }

        res00 = vaddq_u32(res00, res01);
        res10 = vaddq_u32(res10, res11);
        dis[0] = vaddvq_u32(res00);
        dis[1] = vaddvq_u32(res10);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const int32_t tmp0 = x[i] - *(y + i);
        const int32_t tmp1 = x[i] - *(y + d + i);
        dis[0] += tmp0 * tmp0;
        dis[1] += tmp1 * tmp1;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute L2 squares for two vectors with 8-bit integer precision and 32-bit floating-point results.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch2_u8f32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;

    if (likely(d >= single_round)) {
        uint32x4_t res00 = vdupq_n_u32(0);
        uint32x4_t res01 = vdupq_n_u32(0);
        uint32x4_t res10 = vdupq_n_u32(0);
        uint32x4_t res11 = vdupq_n_u32(0);
        for (i = 0; i + double_round <= d; i += double_round) {
            const uint8x16_t x0 = vld1q_u8(x + i);
            const uint8x16_t x1 = vld1q_u8(x + i + 16);
            const uint8x16_t y00 = vld1q_u8(y + i);
            const uint8x16_t y01 = vld1q_u8(y + i + 16);
            const uint8x16_t y10 = vld1q_u8(y + i + d);
            const uint8x16_t y11 = vld1q_u8(y + i + d + 16);

            const uint8x16_t d00 = vabdq_u8(x0, y00);
            res00 = vdotq_u32(res00, d00, d00);
            const uint8x16_t d01 = vabdq_u8(x1, y01);
            res01 = vdotq_u32(res01, d01, d01);
            const uint8x16_t d10 = vabdq_u8(x0, y10);
            res10 = vdotq_u32(res10, d10, d10);
            const uint8x16_t d11 = vabdq_u8(x1, y11);
            res11 = vdotq_u32(res11, d11, d11);
        }
        for (; i <= d - single_round; i += single_round) {
            const uint8x16_t x0 = vld1q_u8(x + i);
            const uint8x16_t y00 = vld1q_u8(y + i);
            const uint8x16_t y10 = vld1q_u8(y + i + d);

            const uint8x16_t d00 = vabdq_u8(x0, y00);
            res00 = vdotq_u32(res00, d00, d00);
            const uint8x16_t d10 = vabdq_u8(x0, y10);
            res10 = vdotq_u32(res10, d10, d10);
        }
        res00 = vaddq_u32(res00, res01);
        res10 = vaddq_u32(res10, res11);
        dis[0] = (float)vaddvq_u32(res00);
        dis[1] = (float)vaddvq_u32(res10);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp0 = x[i] - *(y + i);
        const float tmp1 = x[i] - *(y + d + i);
        dis[0] += tmp0 * tmp0;
        dis[1] += tmp1 * tmp1;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute L2 squares for four vectors with 8-bit integer precision and 32-bit integer results.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (uint32_t).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch4_u8u32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, uint32_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    uint8x16_t neon_query;
    uint8x16_t neon_base[4];
    uint8x16_t neon_diff[4];
    uint32x4_t neon_res[4];
    if (likely(d >= single_round)) {
        neon_query = vld1q_u8(x);
        neon_base[0] = vld1q_u8(y);
        neon_base[1] = vld1q_u8(y + d);
        neon_base[2] = vld1q_u8(y + 2 * d);
        neon_base[3] = vld1q_u8(y + 3 * d);

        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);

        neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
        neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
        neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
        neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
        if (d < double_round) {
            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            i = single_round;
        } else {
            neon_query = vld1q_u8(x + single_round);
            neon_base[0] = vld1q_u8(y + single_round);
            neon_base[1] = vld1q_u8(y + d + single_round);
            neon_base[2] = vld1q_u8(y + 2 * d + single_round);
            neon_base[3] = vld1q_u8(y + 3 * d + single_round);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            for (i = double_round; i <= d - single_round; i += single_round) {
                neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
                neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
                neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
                neon_diff[3] = vabdq_u8(neon_base[3], neon_query);

                neon_query = vld1q_u8(x + i);
                neon_base[0] = vld1q_u8(y + i);
                neon_base[1] = vld1q_u8(y + d + i);
                neon_base[2] = vld1q_u8(y + 2 * d + i);
                neon_base[3] = vld1q_u8(y + 3 * d + i);

                neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
                neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
                neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
                neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            }
            neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
            neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
            neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
            neon_diff[3] = vabdq_u8(neon_base[3], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
        }
        dis[0] = vaddvq_u32(neon_res[0]);
        dis[1] = vaddvq_u32(neon_res[1]);
        dis[2] = vaddvq_u32(neon_res[2]);
        dis[3] = vaddvq_u32(neon_res[3]);
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
        uint32_t d0 = q0 * q0;
        uint32_t d1 = q1 * q1;
        uint32_t d2 = q2 * q2;
        uint32_t d3 = q3 * q3;
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
 * @brief Compute L2 squares for four vectors with 8-bit integer precision and 32-bit floating-point results.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch4_u8f32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    uint8x16_t neon_query;
    uint8x16_t neon_base[4];
    uint8x16_t neon_diff[4];
    uint32x4_t neon_res[4];
    if (likely(d >= single_round)) {
        neon_query = vld1q_u8(x);
        neon_base[0] = vld1q_u8(y);
        neon_base[1] = vld1q_u8(y + d);
        neon_base[2] = vld1q_u8(y + 2 * d);
        neon_base[3] = vld1q_u8(y + 3 * d);

        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);

        neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
        neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
        neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
        neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
        if (d < double_round) {
            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            i = single_round;
        } else {
            neon_query = vld1q_u8(x + single_round);
            neon_base[0] = vld1q_u8(y + single_round);
            neon_base[1] = vld1q_u8(y + d + single_round);
            neon_base[2] = vld1q_u8(y + 2 * d + single_round);
            neon_base[3] = vld1q_u8(y + 3 * d + single_round);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            for (i = double_round; i <= d - single_round; i += single_round) {
                neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
                neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
                neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
                neon_diff[3] = vabdq_u8(neon_base[3], neon_query);

                neon_query = vld1q_u8(x + i);
                neon_base[0] = vld1q_u8(y + i);
                neon_base[1] = vld1q_u8(y + d + i);
                neon_base[2] = vld1q_u8(y + 2 * d + i);
                neon_base[3] = vld1q_u8(y + 3 * d + i);

                neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
                neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
                neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
                neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            }
            neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
            neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
            neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
            neon_diff[3] = vabdq_u8(neon_base[3], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
        }
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[1]);
        neon_res[2] = vpaddq_u32(neon_res[2], neon_res[3]);
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[2]);
        vst1q_f32(dis, vcvtq_f32_u32(neon_res[0]));
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
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
 * @brief Compute L2 squares for eight vectors with 8-bit integer precision and 32-bit integer results.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (uint32_t).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch8_u8u32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, uint32_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    uint8x16_t neon_query;
    uint8x16_t neon_base[8];
    uint8x16_t neon_diff[8];
    uint32x4_t neon_res[8];
    if (likely(d >= single_round)) {
        neon_query = vld1q_u8(x);
        neon_base[0] = vld1q_u8(y);
        neon_base[1] = vld1q_u8(y + d);
        neon_base[2] = vld1q_u8(y + 2 * d);
        neon_base[3] = vld1q_u8(y + 3 * d);
        neon_base[4] = vld1q_u8(y + 4 * d);
        neon_base[5] = vld1q_u8(y + 5 * d);
        neon_base[6] = vld1q_u8(y + 6 * d);
        neon_base[7] = vld1q_u8(y + 7 * d);

        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);
        neon_res[4] = vdupq_n_u32(0);
        neon_res[5] = vdupq_n_u32(0);
        neon_res[6] = vdupq_n_u32(0);
        neon_res[7] = vdupq_n_u32(0);

        neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
        neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
        neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
        neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
        neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
        neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
        neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
        neon_diff[7] = vabdq_u8(neon_base[7], neon_query);
        if (d < double_round) {
            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            i = single_round;
        } else {
            neon_query = vld1q_u8(x + single_round);
            neon_base[0] = vld1q_u8(y + single_round);
            neon_base[1] = vld1q_u8(y + d + single_round);
            neon_base[2] = vld1q_u8(y + 2 * d + single_round);
            neon_base[3] = vld1q_u8(y + 3 * d + single_round);
            neon_base[4] = vld1q_u8(y + 4 * d + single_round);
            neon_base[5] = vld1q_u8(y + 5 * d + single_round);
            neon_base[6] = vld1q_u8(y + 6 * d + single_round);
            neon_base[7] = vld1q_u8(y + 7 * d + single_round);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            for (i = double_round; i <= d - single_round; i += single_round) {
                neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
                neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
                neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
                neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
                neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
                neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
                neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
                neon_diff[7] = vabdq_u8(neon_base[7], neon_query);

                neon_query = vld1q_u8(x + i);
                neon_base[0] = vld1q_u8(y + i);
                neon_base[1] = vld1q_u8(y + d + i);
                neon_base[2] = vld1q_u8(y + 2 * d + i);
                neon_base[3] = vld1q_u8(y + 3 * d + i);
                neon_base[4] = vld1q_u8(y + 4 * d + i);
                neon_base[5] = vld1q_u8(y + 5 * d + i);
                neon_base[6] = vld1q_u8(y + 6 * d + i);
                neon_base[7] = vld1q_u8(y + 7 * d + i);

                neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
                neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
                neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
                neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
                neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
                neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
                neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
                neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            }
            neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
            neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
            neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
            neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
            neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
            neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
            neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
            neon_diff[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
        }
        dis[0] = vaddvq_u32(neon_res[0]);
        dis[1] = vaddvq_u32(neon_res[1]);
        dis[2] = vaddvq_u32(neon_res[2]);
        dis[3] = vaddvq_u32(neon_res[3]);
        dis[4] = vaddvq_u32(neon_res[4]);
        dis[5] = vaddvq_u32(neon_res[5]);
        dis[6] = vaddvq_u32(neon_res[6]);
        dis[7] = vaddvq_u32(neon_res[7]);
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
        int32_t q4 = x[i] - *(y + 4 * d + i);
        int32_t q5 = x[i] - *(y + 5 * d + i);
        int32_t q6 = x[i] - *(y + 6 * d + i);
        int32_t q7 = x[i] - *(y + 7 * d + i);
        uint32_t d0 = q0 * q0;
        uint32_t d1 = q1 * q1;
        uint32_t d2 = q2 * q2;
        uint32_t d3 = q3 * q3;
        uint32_t d4 = q4 * q4;
        uint32_t d5 = q5 * q5;
        uint32_t d6 = q6 * q6;
        uint32_t d7 = q7 * q7;
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
 * @brief Compute L2 squares for eight vectors with 8-bit integer precision and 32-bit floating-point results.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch8_u8f32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    uint8x16_t neon_query;
    uint8x16_t neon_base[8];
    uint8x16_t neon_diff[8];
    uint32x4_t neon_res[8];
    if (likely(d >= single_round)) {
        neon_query = vld1q_u8(x);
        neon_base[0] = vld1q_u8(y);
        neon_base[1] = vld1q_u8(y + d);
        neon_base[2] = vld1q_u8(y + 2 * d);
        neon_base[3] = vld1q_u8(y + 3 * d);
        neon_base[4] = vld1q_u8(y + 4 * d);
        neon_base[5] = vld1q_u8(y + 5 * d);
        neon_base[6] = vld1q_u8(y + 6 * d);
        neon_base[7] = vld1q_u8(y + 7 * d);

        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);
        neon_res[4] = vdupq_n_u32(0);
        neon_res[5] = vdupq_n_u32(0);
        neon_res[6] = vdupq_n_u32(0);
        neon_res[7] = vdupq_n_u32(0);

        neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
        neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
        neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
        neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
        neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
        neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
        neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
        neon_diff[7] = vabdq_u8(neon_base[7], neon_query);
        if (d < double_round) {
            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            i = single_round;
        } else {
            neon_query = vld1q_u8(x + single_round);
            neon_base[0] = vld1q_u8(y + single_round);
            neon_base[1] = vld1q_u8(y + d + single_round);
            neon_base[2] = vld1q_u8(y + 2 * d + single_round);
            neon_base[3] = vld1q_u8(y + 3 * d + single_round);
            neon_base[4] = vld1q_u8(y + 4 * d + single_round);
            neon_base[5] = vld1q_u8(y + 5 * d + single_round);
            neon_base[6] = vld1q_u8(y + 6 * d + single_round);
            neon_base[7] = vld1q_u8(y + 7 * d + single_round);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            for (i = double_round; i <= d - single_round; i += single_round) {
                neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
                neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
                neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
                neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
                neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
                neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
                neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
                neon_diff[7] = vabdq_u8(neon_base[7], neon_query);

                neon_query = vld1q_u8(x + i);
                neon_base[0] = vld1q_u8(y + i);
                neon_base[1] = vld1q_u8(y + d + i);
                neon_base[2] = vld1q_u8(y + 2 * d + i);
                neon_base[3] = vld1q_u8(y + 3 * d + i);
                neon_base[4] = vld1q_u8(y + 4 * d + i);
                neon_base[5] = vld1q_u8(y + 5 * d + i);
                neon_base[6] = vld1q_u8(y + 6 * d + i);
                neon_base[7] = vld1q_u8(y + 7 * d + i);

                neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
                neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
                neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
                neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
                neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
                neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
                neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
                neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            }
            neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
            neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
            neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
            neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
            neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
            neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
            neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
            neon_diff[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
        }
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[1]);
        neon_res[2] = vpaddq_u32(neon_res[2], neon_res[3]);
        neon_res[4] = vpaddq_u32(neon_res[4], neon_res[5]);
        neon_res[6] = vpaddq_u32(neon_res[6], neon_res[7]);
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[2]);
        neon_res[4] = vpaddq_u32(neon_res[4], neon_res[6]);
        vst1q_f32(dis, vcvtq_f32_u32(neon_res[0]));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res[4]));
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
        int32_t q4 = x[i] - *(y + 4 * d + i);
        int32_t q5 = x[i] - *(y + 5 * d + i);
        int32_t q6 = x[i] - *(y + 6 * d + i);
        int32_t q7 = x[i] - *(y + 7 * d + i);
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
 * @brief Compute L2 squares for 16 vectors with 8-bit integer precision and 32-bit integer results.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (uint32_t).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch16_u8u32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, uint32_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16; /* 128 / 8 */
    uint8x16_t neon_query;
    uint8x16_t neon_base[8];
    uint32x4_t neon_res[16];
    if (likely(d >= single_round)) {
        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);
        neon_res[4] = vdupq_n_u32(0);
        neon_res[5] = vdupq_n_u32(0);
        neon_res[6] = vdupq_n_u32(0);
        neon_res[7] = vdupq_n_u32(0);
        neon_res[8] = vdupq_n_u32(0);
        neon_res[9] = vdupq_n_u32(0);
        neon_res[10] = vdupq_n_u32(0);
        neon_res[11] = vdupq_n_u32(0);
        neon_res[12] = vdupq_n_u32(0);
        neon_res[13] = vdupq_n_u32(0);
        neon_res[14] = vdupq_n_u32(0);
        neon_res[15] = vdupq_n_u32(0);
        for (i = 0; i <= d - single_round; i += single_round) {
            neon_query = vld1q_u8(x + i);
            neon_base[0] = vld1q_u8(y + i);
            neon_base[1] = vld1q_u8(y + d + i);
            neon_base[2] = vld1q_u8(y + 2 * d + i);
            neon_base[3] = vld1q_u8(y + 3 * d + i);
            neon_base[4] = vld1q_u8(y + 4 * d + i);
            neon_base[5] = vld1q_u8(y + 5 * d + i);
            neon_base[6] = vld1q_u8(y + 6 * d + i);
            neon_base[7] = vld1q_u8(y + 7 * d + i);

            neon_base[0] = vabdq_u8(neon_base[0], neon_query);
            neon_base[1] = vabdq_u8(neon_base[1], neon_query);
            neon_base[2] = vabdq_u8(neon_base[2], neon_query);
            neon_base[3] = vabdq_u8(neon_base[3], neon_query);
            neon_base[4] = vabdq_u8(neon_base[4], neon_query);
            neon_base[5] = vabdq_u8(neon_base[5], neon_query);
            neon_base[6] = vabdq_u8(neon_base[6], neon_query);
            neon_base[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_base[0], neon_base[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_base[1], neon_base[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_base[2], neon_base[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_base[3], neon_base[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_base[4], neon_base[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_base[5], neon_base[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_base[6], neon_base[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_base[7], neon_base[7]);

            neon_base[0] = vld1q_u8(y + 8 * d + i);
            neon_base[1] = vld1q_u8(y + 9 * d + i);
            neon_base[2] = vld1q_u8(y + 10 * d + i);
            neon_base[3] = vld1q_u8(y + 11 * d + i);
            neon_base[4] = vld1q_u8(y + 12 * d + i);
            neon_base[5] = vld1q_u8(y + 13 * d + i);
            neon_base[6] = vld1q_u8(y + 14 * d + i);
            neon_base[7] = vld1q_u8(y + 15 * d + i);

            neon_base[0] = vabdq_u8(neon_base[0], neon_query);
            neon_base[1] = vabdq_u8(neon_base[1], neon_query);
            neon_base[2] = vabdq_u8(neon_base[2], neon_query);
            neon_base[3] = vabdq_u8(neon_base[3], neon_query);
            neon_base[4] = vabdq_u8(neon_base[4], neon_query);
            neon_base[5] = vabdq_u8(neon_base[5], neon_query);
            neon_base[6] = vabdq_u8(neon_base[6], neon_query);
            neon_base[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[8] = vdotq_u32(neon_res[8], neon_base[0], neon_base[0]);
            neon_res[9] = vdotq_u32(neon_res[9], neon_base[1], neon_base[1]);
            neon_res[10] = vdotq_u32(neon_res[10], neon_base[2], neon_base[2]);
            neon_res[11] = vdotq_u32(neon_res[11], neon_base[3], neon_base[3]);
            neon_res[12] = vdotq_u32(neon_res[12], neon_base[4], neon_base[4]);
            neon_res[13] = vdotq_u32(neon_res[13], neon_base[5], neon_base[5]);
            neon_res[14] = vdotq_u32(neon_res[14], neon_base[6], neon_base[6]);
            neon_res[15] = vdotq_u32(neon_res[15], neon_base[7], neon_base[7]);
        }
        dis[0] = vaddvq_u32(neon_res[0]);
        dis[1] = vaddvq_u32(neon_res[1]);
        dis[2] = vaddvq_u32(neon_res[2]);
        dis[3] = vaddvq_u32(neon_res[3]);
        dis[4] = vaddvq_u32(neon_res[4]);
        dis[5] = vaddvq_u32(neon_res[5]);
        dis[6] = vaddvq_u32(neon_res[6]);
        dis[7] = vaddvq_u32(neon_res[7]);
        dis[8] = vaddvq_u32(neon_res[8]);
        dis[9] = vaddvq_u32(neon_res[9]);
        dis[10] = vaddvq_u32(neon_res[10]);
        dis[11] = vaddvq_u32(neon_res[11]);
        dis[12] = vaddvq_u32(neon_res[12]);
        dis[13] = vaddvq_u32(neon_res[13]);
        dis[14] = vaddvq_u32(neon_res[14]);
        dis[15] = vaddvq_u32(neon_res[15]);
    } else {
        for (int i = 0; i < 16; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
        int32_t q4 = x[i] - *(y + 4 * d + i);
        int32_t q5 = x[i] - *(y + 5 * d + i);
        int32_t q6 = x[i] - *(y + 6 * d + i);
        int32_t q7 = x[i] - *(y + 7 * d + i);
        uint32_t d0 = q0 * q0;
        uint32_t d1 = q1 * q1;
        uint32_t d2 = q2 * q2;
        uint32_t d3 = q3 * q3;
        uint32_t d4 = q4 * q4;
        uint32_t d5 = q5 * q5;
        uint32_t d6 = q6 * q6;
        uint32_t d7 = q7 * q7;
        q0 = x[i] - *(y + 8 * d + i);
        q1 = x[i] - *(y + 9 * d + i);
        q2 = x[i] - *(y + 10 * d + i);
        q3 = x[i] - *(y + 11 * d + i);
        q4 = x[i] - *(y + 12 * d + i);
        q5 = x[i] - *(y + 13 * d + i);
        q6 = x[i] - *(y + 14 * d + i);
        q7 = x[i] - *(y + 15 * d + i);
        uint32_t d8 = q0 * q0;
        uint32_t d9 = q1 * q1;
        uint32_t d10 = q2 * q2;
        uint32_t d11 = q3 * q3;
        uint32_t d12 = q4 * q4;
        uint32_t d13 = q5 * q5;
        uint32_t d14 = q6 * q6;
        uint32_t d15 = q7 * q7;
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
 * @brief Compute L2 squares for 16 vectors with 8-bit integer precision and 32-bit float results.
 * @param x Pointer to the query vector (uint8_t).
 * @param y Array of pointers to the database vectors (uint8_t).
 * @param d Dimension of the vectors.
 * @param dis Pointer to the output array for storing the results (float).
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch16_u8f32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16; /* 128 / 8 */
    uint8x16_t neon_query;
    uint8x16_t neon_base[8];
    uint32x4_t neon_res[16];
    if (likely(d >= single_round)) {
        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);
        neon_res[4] = vdupq_n_u32(0);
        neon_res[5] = vdupq_n_u32(0);
        neon_res[6] = vdupq_n_u32(0);
        neon_res[7] = vdupq_n_u32(0);
        neon_res[8] = vdupq_n_u32(0);
        neon_res[9] = vdupq_n_u32(0);
        neon_res[10] = vdupq_n_u32(0);
        neon_res[11] = vdupq_n_u32(0);
        neon_res[12] = vdupq_n_u32(0);
        neon_res[13] = vdupq_n_u32(0);
        neon_res[14] = vdupq_n_u32(0);
        neon_res[15] = vdupq_n_u32(0);
        for (i = 0; i <= d - single_round; i += single_round) {
            neon_query = vld1q_u8(x + i);
            neon_base[0] = vld1q_u8(y + i);
            neon_base[1] = vld1q_u8(y + d + i);
            neon_base[2] = vld1q_u8(y + 2 * d + i);
            neon_base[3] = vld1q_u8(y + 3 * d + i);
            neon_base[4] = vld1q_u8(y + 4 * d + i);
            neon_base[5] = vld1q_u8(y + 5 * d + i);
            neon_base[6] = vld1q_u8(y + 6 * d + i);
            neon_base[7] = vld1q_u8(y + 7 * d + i);

            neon_base[0] = vabdq_u8(neon_base[0], neon_query);
            neon_base[1] = vabdq_u8(neon_base[1], neon_query);
            neon_base[2] = vabdq_u8(neon_base[2], neon_query);
            neon_base[3] = vabdq_u8(neon_base[3], neon_query);
            neon_base[4] = vabdq_u8(neon_base[4], neon_query);
            neon_base[5] = vabdq_u8(neon_base[5], neon_query);
            neon_base[6] = vabdq_u8(neon_base[6], neon_query);
            neon_base[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_base[0], neon_base[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_base[1], neon_base[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_base[2], neon_base[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_base[3], neon_base[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_base[4], neon_base[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_base[5], neon_base[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_base[6], neon_base[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_base[7], neon_base[7]);

            neon_base[0] = vld1q_u8(y + 8 * d + i);
            neon_base[1] = vld1q_u8(y + 9 * d + i);
            neon_base[2] = vld1q_u8(y + 10 * d + i);
            neon_base[3] = vld1q_u8(y + 11 * d + i);
            neon_base[4] = vld1q_u8(y + 12 * d + i);
            neon_base[5] = vld1q_u8(y + 13 * d + i);
            neon_base[6] = vld1q_u8(y + 14 * d + i);
            neon_base[7] = vld1q_u8(y + 15 * d + i);

            neon_base[0] = vabdq_u8(neon_base[0], neon_query);
            neon_base[1] = vabdq_u8(neon_base[1], neon_query);
            neon_base[2] = vabdq_u8(neon_base[2], neon_query);
            neon_base[3] = vabdq_u8(neon_base[3], neon_query);
            neon_base[4] = vabdq_u8(neon_base[4], neon_query);
            neon_base[5] = vabdq_u8(neon_base[5], neon_query);
            neon_base[6] = vabdq_u8(neon_base[6], neon_query);
            neon_base[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[8] = vdotq_u32(neon_res[8], neon_base[0], neon_base[0]);
            neon_res[9] = vdotq_u32(neon_res[9], neon_base[1], neon_base[1]);
            neon_res[10] = vdotq_u32(neon_res[10], neon_base[2], neon_base[2]);
            neon_res[11] = vdotq_u32(neon_res[11], neon_base[3], neon_base[3]);
            neon_res[12] = vdotq_u32(neon_res[12], neon_base[4], neon_base[4]);
            neon_res[13] = vdotq_u32(neon_res[13], neon_base[5], neon_base[5]);
            neon_res[14] = vdotq_u32(neon_res[14], neon_base[6], neon_base[6]);
            neon_res[15] = vdotq_u32(neon_res[15], neon_base[7], neon_base[7]);
        }
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[1]);
        neon_res[2] = vpaddq_u32(neon_res[2], neon_res[3]);
        neon_res[4] = vpaddq_u32(neon_res[4], neon_res[5]);
        neon_res[6] = vpaddq_u32(neon_res[6], neon_res[7]);
        neon_res[8] = vpaddq_u32(neon_res[8], neon_res[9]);
        neon_res[10] = vpaddq_u32(neon_res[10], neon_res[11]);
        neon_res[12] = vpaddq_u32(neon_res[12], neon_res[13]);
        neon_res[14] = vpaddq_u32(neon_res[14], neon_res[15]);
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[2]);
        neon_res[4] = vpaddq_u32(neon_res[4], neon_res[6]);
        neon_res[8] = vpaddq_u32(neon_res[8], neon_res[10]);
        neon_res[12] = vpaddq_u32(neon_res[12], neon_res[14]);
        vst1q_f32(dis, vcvtq_f32_u32(neon_res[0]));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res[4]));
        vst1q_f32(dis + 8, vcvtq_f32_u32(neon_res[8]));
        vst1q_f32(dis + 12, vcvtq_f32_u32(neon_res[12]));
    } else {
        for (int i = 0; i < 16; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
        int32_t q4 = x[i] - *(y + 4 * d + i);
        int32_t q5 = x[i] - *(y + 5 * d + i);
        int32_t q6 = x[i] - *(y + 6 * d + i);
        int32_t q7 = x[i] - *(y + 7 * d + i);
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
 * @brief Compute L2 squares for a batch of vectors based on given indices.
 * @param dis Output array to store the results (float).
 * @param x Pointer to the query vector (uint8_t).
 * @param y Pointer to the database vectors (uint8_t).
 * @param ids Array of indices specifying which vectors to use from y.
 * @param d Dimension of the vectors.
 * @param ny Number of vectors to process.
 * @param dis_size Length of dis.
 */
int krl_L2sqr_by_idx_u8f32(
    float *dis, const uint8_t *x, const uint8_t *y, const int64_t *ids, size_t d, size_t ny, size_t dis_size)
{
    size_t i = 0;
    const uint8_t *__restrict listy[24];

    for (; i + 24 <= ny; i += 24) {
        prefetch_L1(x);
        listy[0] = (const uint8_t *)(y + *(ids + i) * d);
        prefetch_Lx(listy[0]);
        listy[1] = (const uint8_t *)(y + *(ids + i + 1) * d);
        prefetch_Lx(listy[1]);
        listy[2] = (const uint8_t *)(y + *(ids + i + 2) * d);
        prefetch_Lx(listy[2]);
        listy[3] = (const uint8_t *)(y + *(ids + i + 3) * d);
        prefetch_Lx(listy[3]);
        listy[4] = (const uint8_t *)(y + *(ids + i + 4) * d);
        prefetch_Lx(listy[4]);
        listy[5] = (const uint8_t *)(y + *(ids + i + 5) * d);
        prefetch_Lx(listy[5]);
        listy[6] = (const uint8_t *)(y + *(ids + i + 6) * d);
        prefetch_Lx(listy[6]);
        listy[7] = (const uint8_t *)(y + *(ids + i + 7) * d);
        prefetch_Lx(listy[7]);
        listy[8] = (const uint8_t *)(y + *(ids + i + 8) * d);
        prefetch_Lx(listy[8]);
        listy[9] = (const uint8_t *)(y + *(ids + i + 9) * d);
        prefetch_Lx(listy[9]);
        listy[10] = (const uint8_t *)(y + *(ids + i + 10) * d);
        prefetch_Lx(listy[10]);
        listy[11] = (const uint8_t *)(y + *(ids + i + 11) * d);
        prefetch_Lx(listy[11]);
        listy[12] = (const uint8_t *)(y + *(ids + i + 12) * d);
        prefetch_Lx(listy[12]);
        listy[13] = (const uint8_t *)(y + *(ids + i + 13) * d);
        prefetch_Lx(listy[13]);
        listy[14] = (const uint8_t *)(y + *(ids + i + 14) * d);
        prefetch_Lx(listy[14]);
        listy[15] = (const uint8_t *)(y + *(ids + i + 15) * d);
        prefetch_Lx(listy[15]);
        listy[16] = (const uint8_t *)(y + *(ids + i + 16) * d);
        prefetch_Lx(listy[16]);
        listy[17] = (const uint8_t *)(y + *(ids + i + 17) * d);
        prefetch_Lx(listy[17]);
        listy[18] = (const uint8_t *)(y + *(ids + i + 18) * d);
        prefetch_Lx(listy[18]);
        listy[19] = (const uint8_t *)(y + *(ids + i + 19) * d);
        prefetch_Lx(listy[19]);
        listy[20] = (const uint8_t *)(y + *(ids + i + 20) * d);
        prefetch_Lx(listy[20]);
        listy[21] = (const uint8_t *)(y + *(ids + i + 21) * d);
        prefetch_Lx(listy[21]);
        listy[22] = (const uint8_t *)(y + *(ids + i + 22) * d);
        prefetch_Lx(listy[22]);
        listy[23] = (const uint8_t *)(y + *(ids + i + 23) * d);
        prefetch_Lx(listy[23]);
        krl_L2sqr_idx_prefetch_batch24_u8f32(x, listy, d, dis + i);
    }
    if (i + 16 <= ny) {
        prefetch_L1(x);
        listy[0] = (const uint8_t *)(y + *(ids + i) * d);
        prefetch_Lx(listy[0]);
        listy[1] = (const uint8_t *)(y + *(ids + i + 1) * d);
        prefetch_Lx(listy[1]);
        listy[2] = (const uint8_t *)(y + *(ids + i + 2) * d);
        prefetch_Lx(listy[2]);
        listy[3] = (const uint8_t *)(y + *(ids + i + 3) * d);
        prefetch_Lx(listy[3]);
        listy[4] = (const uint8_t *)(y + *(ids + i + 4) * d);
        prefetch_Lx(listy[4]);
        listy[5] = (const uint8_t *)(y + *(ids + i + 5) * d);
        prefetch_Lx(listy[5]);
        listy[6] = (const uint8_t *)(y + *(ids + i + 6) * d);
        prefetch_Lx(listy[6]);
        listy[7] = (const uint8_t *)(y + *(ids + i + 7) * d);
        prefetch_Lx(listy[7]);
        listy[8] = (const uint8_t *)(y + *(ids + i + 8) * d);
        prefetch_Lx(listy[8]);
        listy[9] = (const uint8_t *)(y + *(ids + i + 9) * d);
        prefetch_Lx(listy[9]);
        listy[10] = (const uint8_t *)(y + *(ids + i + 10) * d);
        prefetch_Lx(listy[10]);
        listy[11] = (const uint8_t *)(y + *(ids + i + 11) * d);
        prefetch_Lx(listy[11]);
        listy[12] = (const uint8_t *)(y + *(ids + i + 12) * d);
        prefetch_Lx(listy[12]);
        listy[13] = (const uint8_t *)(y + *(ids + i + 13) * d);
        prefetch_Lx(listy[13]);
        listy[14] = (const uint8_t *)(y + *(ids + i + 14) * d);
        prefetch_Lx(listy[14]);
        listy[15] = (const uint8_t *)(y + *(ids + i + 15) * d);
        prefetch_Lx(listy[15]);
        krl_L2sqr_idx_prefetch_batch16_u8f32(x, listy, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        listy[0] = (const uint8_t *)(y + *(ids + i) * d);
        listy[1] = (const uint8_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const uint8_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const uint8_t *)(y + *(ids + i + 3) * d);
        listy[4] = (const uint8_t *)(y + *(ids + i + 4) * d);
        listy[5] = (const uint8_t *)(y + *(ids + i + 5) * d);
        listy[6] = (const uint8_t *)(y + *(ids + i + 6) * d);
        listy[7] = (const uint8_t *)(y + *(ids + i + 7) * d);
        krl_L2sqr_idx_batch8_u8f32(x, listy, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        listy[0] = (const uint8_t *)(y + *(ids + i) * d);
        listy[1] = (const uint8_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const uint8_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const uint8_t *)(y + *(ids + i + 3) * d);
        krl_L2sqr_idx_batch4_u8f32(x, listy, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        const uint8_t *y0 = y + *(ids + i) * d;
        const uint8_t *y1 = y + *(ids + i + 1) * d;
        krl_L2sqr_idx_batch2_u8f32(x, y0, y1, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        uint32_t tmp;
        krl_L2sqr_u8u32(x, y + d * ids[i], d, &tmp, 1);
        dis[i] = (float)tmp;
    }
    return SUCCESS;
}

/*
 * @brief Compute L2 squares for a batch of vectors.
 * @param dis Output array to store the results (uint32_t).
 * @param x Pointer to the query vector (uint8_t).
 * @param y Pointer to the database vectors (uint8_t).
 * @param ny Number of vectors to process.
 * @param d Dimension of the vectors.
 */
void krl_L2sqr_ny_u8u32(uint32_t *dis, const uint8_t *x, const uint8_t *y, size_t ny, size_t d)
{
    size_t i = 0;
    for (; i + 16 <= ny; i += 16) {
        krl_L2sqr_batch16_u8u32(x, y + i * d, d, dis + i);
    }
    if (ny & 8) {
        krl_L2sqr_batch8_u8u32(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_L2sqr_batch4_u8u32(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_L2sqr_batch2_u8u32(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        krl_L2sqr_u8u32(x, y + i * d, d, &dis[i], 1);
    }
}

/*
 * @brief Compute L2 squares for a batch of vectors with float results.
 * @param dis Output array to store the results (float).
 * @param x Pointer to the query vector (uint8_t).
 * @param y Pointer to the database vectors (uint8_t).
 * @param ny Number of vectors to process.
 * @param d Dimension of the vectors.
 * @param dis_size Length of dis.
 */
int krl_L2sqr_ny_u8f32(float *dis, const uint8_t *x, const uint8_t *y, size_t ny, size_t d, size_t dis_size)
{
    size_t i = 0;

    for (; i + 16 <= ny; i += 16) {
        krl_L2sqr_batch16_u8f32(x, y + i * d, d, dis + i);
    }
    if (ny & 8) {
        krl_L2sqr_batch8_u8f32(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_L2sqr_batch4_u8f32(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_L2sqr_batch2_u8f32(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        uint32_t tmp;
        krl_L2sqr_u8u32(x, y + i * d, d, &tmp, 1);
        dis[i] = (float)tmp;
    }
    return SUCCESS;
}
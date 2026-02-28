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
 * @brief Compute the negative inner product distance between a int8 vector and two int8 vectors in batch mode.
 * @param x Pointer to the input int8 vector.
 * @param y0 Pointer to the first input int8 vector.
 * @param y1 Pointer to the second input int8 vector.
 * @param d Length of the vectors.
 * @param dis Pointer to the output array storing the computed distances.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_negative_ipdis_idx_batch2_s8f32(
    const int8_t *x, const int8_t *__restrict y0, const int8_t *__restrict y1, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    int32x4_t res1 = vdupq_n_s32(0);
    int32x4_t res2 = vdupq_n_s32(0);
    int32x4_t res3 = vdupq_n_s32(0);
    int32x4_t res4 = vdupq_n_s32(0);
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
    for (; i + single_round <= d; i += single_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t y8_0 = vld1q_s8(y0 + i);
        const int8x16_t y8_1 = vld1q_s8(y1 + i);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res3 = vdotq_s32(res3, x8_0, y8_1);
    }
    res1 = vaddq_s32(res1, res2);
    res3 = vaddq_s32(res3, res4);
    res1 = vnegq_s32(res1);
    res3 = vnegq_s32(res3);
    dis[0] = (float)vaddvq_s32(res1);
    dis[1] = (float)vaddvq_s32(res3);
    for (; i < d; i++) {
        dis[0] -= (float)(x[i]) * (float)(y0[i]);
        dis[1] -= (float)(x[i]) * (float)(y1[i]);
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the negative inner product distance between a int8 vector and four int8 vectors in batch mode.
 * @param x Pointer to the input int8 vector.
 * @param y Array of pointers to the input int8 vectors.
 * @param d Length of the vectors.
 * @param dis Pointer to the output array storing the computed distances.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_negative_ipdis_idx_batch4_s8f32(
    const int8_t *x, const int8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16; /* 128 / 8 */

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);

    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        neon_query = vnegq_s8(neon_query);
        int8x16_t neon_base1 = vld1q_s8(y[0] + i);
        int8x16_t neon_base2 = vld1q_s8(y[1] + i);
        int8x16_t neon_base3 = vld1q_s8(y[2] + i);
        int8x16_t neon_base4 = vld1q_s8(y[3] + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
    }
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);

    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
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
        dis[0] -= d0;
        dis[1] -= d1;
        dis[2] -= d2;
        dis[3] -= d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the negative inner product distance between a int8 vector and eight int8 vectors in batch mode.
 * @param x Pointer to the input int8 vector.
 * @param y Array of pointers to the input int8 vectors.
 * @param d Length of the vectors.
 * @param dis Pointer to the output array storing the computed distances.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_negative_ipdis_idx_batch8_s8f32(
    const int8_t *x, const int8_t *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16; /* 128 / 8 */

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);

    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        neon_query = vnegq_s8(neon_query);
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
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res5 = vpaddq_s32(neon_res5, neon_res6);
    neon_res7 = vpaddq_s32(neon_res7, neon_res8);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);
    neon_res5 = vpaddq_s32(neon_res5, neon_res7);

    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
    vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));
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
        dis[0] -= d0;
        dis[1] -= d1;
        dis[2] -= d2;
        dis[3] -= d3;
        dis[4] -= d4;
        dis[5] -= d5;
        dis[6] -= d6;
        dis[7] -= d7;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the negative inner product distance between a int8 vector and sixteen int8 vectors in batch mode.
 * @param x Pointer to the input int8 vector.
 * @param y Array of pointers to the input int8 vectors.
 * @param d Length of the vectors.
 * @param dis Pointer to the output array storing the computed distances.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_negative_ipdis_idx_prefetch_batch16_s8f32(
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
                int8x16_t neon_query = vld1q_s8(x + i + j);
                neon_query = vnegq_s8(neon_query);
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
        for (; i <= d - single_round; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            neon_query = vnegq_s8(neon_query);
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
        for (i = 0; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            neon_query = vnegq_s8(neon_query);
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
        dis[0] -= d0;
        dis[1] -= d1;
        dis[2] -= d2;
        dis[3] -= d3;
        dis[4] -= d4;
        dis[5] -= d5;
        dis[6] -= d6;
        dis[7] -= d7;
        dis[8] -= d8;
        dis[9] -= d9;
        dis[10] -= d10;
        dis[11] -= d11;
        dis[12] -= d12;
        dis[13] -= d13;
        dis[14] -= d14;
        dis[15] -= d15;
    }
}
KRL_IMPRECISE_FUNCTION_END

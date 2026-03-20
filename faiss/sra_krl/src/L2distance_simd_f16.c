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
 * @brief Compute the L2 square of two float16 vectors using NEON instructions
 * @param x Pointer to the first float16 vector
 * @param y Pointer to the second float16 vector (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @return The computed L2 square as a float16 value
 */
KRL_IMPRECISE_FUNCTION_BEGIN
float16_t krl_L2sqr_f16f16(const float16_t *x, const float16_t *__restrict y, const size_t d)
{
    constexpr size_t single_round = 8;
    constexpr size_t multi_round = 32;
    size_t i;
    float16_t res;
    if (d >= multi_round) {
        prefetch_Lx(x + multi_round);
        prefetch_Lx(y + multi_round);
        float16x8_t x8_0 = vld1q_f16(x);
        float16x8_t x8_1 = vld1q_f16(x + 8);
        float16x8_t x8_2 = vld1q_f16(x + 16);
        float16x8_t x8_3 = vld1q_f16(x + 24);

        float16x8_t y8_0 = vld1q_f16(y);
        float16x8_t y8_1 = vld1q_f16(y + 8);
        float16x8_t y8_2 = vld1q_f16(y + 16);
        float16x8_t y8_3 = vld1q_f16(y + 24);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        d8_0 = vmulq_f16(d8_0, d8_0);
        float16x8_t d8_1 = vsubq_f16(x8_1, y8_1);
        d8_1 = vmulq_f16(d8_1, d8_1);
        float16x8_t d8_2 = vsubq_f16(x8_2, y8_2);
        d8_2 = vmulq_f16(d8_2, d8_2);
        float16x8_t d8_3 = vsubq_f16(x8_3, y8_3);
        d8_3 = vmulq_f16(d8_3, d8_3);

        /* Compute L2 square using NEON vector operations */
        for (i = multi_round; i <= d - multi_round; i += multi_round) {
            prefetch_Lx(x + i + multi_round);
            prefetch_Lx(y + i + multi_round);
            x8_0 = vld1q_f16(x + i);
            y8_0 = vld1q_f16(y + i);
            const float16x8_t q8_0 = vsubq_f16(x8_0, y8_0);
            d8_0 = vfmaq_f16(d8_0, q8_0, q8_0);

            x8_1 = vld1q_f16(x + i + 8);
            y8_1 = vld1q_f16(y + i + 8);
            const float16x8_t q8_1 = vsubq_f16(x8_1, y8_1);
            d8_1 = vfmaq_f16(d8_1, q8_1, q8_1);

            x8_2 = vld1q_f16(x + i + 16);
            y8_2 = vld1q_f16(y + i + 16);
            const float16x8_t q8_2 = vsubq_f16(x8_2, y8_2);
            d8_2 = vfmaq_f16(d8_2, q8_2, q8_2);

            x8_3 = vld1q_f16(x + i + 24);
            y8_3 = vld1q_f16(y + i + 24);
            const float16x8_t q8_3 = vsubq_f16(x8_3, y8_3);
            d8_3 = vfmaq_f16(d8_3, q8_3, q8_3);
        }

        for (; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f16(x + i);
            y8_0 = vld1q_f16(y + i);
            const float16x8_t q8_0 = vsubq_f16(x8_0, y8_0);
            d8_0 = vfmaq_f16(d8_0, q8_0, q8_0);
        }

        d8_0 = vaddq_f16(d8_0, d8_1);
        d8_2 = vaddq_f16(d8_2, d8_3);
        d8_0 = vaddq_f16(d8_0, d8_2);
        /* Reduce the vector result to a scalar value */
        /* 8 -> 4 */
        d8_0 = vpaddq_f16(d8_0, d8_0);
        /* 4 -> 2 */
        d8_0 = vpaddq_f16(d8_0, d8_0);
        /* 2 -> 1 */
        d8_0 = vpaddq_f16(d8_0, d8_0);
        res = vgetq_lane_f16(d8_0, 0);
    } else {
        res = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float16_t tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the L2 square of a float16 vector with two batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y0 Pointer to the first batch of float16 vectors (with restrict qualifier for better optimization)
 * @param y1 Pointer to the second batch of float16 vectors (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_batch2_f16f16(
    const float16_t *x, const float16_t *__restrict y0, const float16_t *__restrict y1, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    if (likely(d >= single_round)) {
        float16x8_t x_0 = vld1q_f16(x);
        float16x8_t x_1 = vld1q_f16(x + 8);

        float16x8_t y0_0 = vld1q_f16(y0);
        float16x8_t y0_1 = vld1q_f16(y0 + 8);
        float16x8_t y1_0 = vld1q_f16(y1);
        float16x8_t y1_1 = vld1q_f16(y1 + 8);

        float16x8_t d0_0 = vsubq_f16(x_0, y0_0);
        d0_0 = vmulq_f16(d0_0, d0_0);
        float16x8_t d0_1 = vsubq_f16(x_1, y0_1);
        d0_1 = vmulq_f16(d0_1, d0_1);
        float16x8_t d1_0 = vsubq_f16(x_0, y1_0);
        d1_0 = vmulq_f16(d1_0, d1_0);
        float16x8_t d1_1 = vsubq_f16(x_1, y1_1);
        d1_1 = vmulq_f16(d1_1, d1_1);

        for (i = single_round; i <= d - single_round; i += single_round) {
            x_0 = vld1q_f16(x + i);
            y0_0 = vld1q_f16(y0 + i);
            y1_0 = vld1q_f16(y1 + i);
            const float16x8_t q0_0 = vsubq_f16(x_0, y0_0);
            const float16x8_t q1_0 = vsubq_f16(x_0, y1_0);
            d0_0 = vfmaq_f16(d0_0, q0_0, q0_0);
            d1_0 = vfmaq_f16(d1_0, q1_0, q1_0);

            x_1 = vld1q_f16(x + i + 8);
            y0_1 = vld1q_f16(y0 + i + 8);
            y1_1 = vld1q_f16(y1 + i + 8);
            const float16x8_t q0_1 = vsubq_f16(x_1, y0_1);
            const float16x8_t q1_1 = vsubq_f16(x_1, y1_1);
            d0_1 = vfmaq_f16(d0_1, q0_1, q0_1);
            d1_1 = vfmaq_f16(d1_1, q1_1, q1_1);
        }
        d0_0 = vaddq_f16(d0_0, d0_1);
        d1_0 = vaddq_f16(d1_0, d1_1);
        /* Reduce the vector result to scalar values */
        /* 8 -> 4 */
        d0_0 = vpaddq_f16(d0_0, d1_0);
        /* 4 -> 2 */
        d0_0 = vpaddq_f16(d0_0, d0_0);
        /* 2 -> 1 */
        d0_0 = vpaddq_f16(d0_0, d0_0);
        dis[0] = vgetq_lane_f16(d0_0, 0);
        dis[1] = vgetq_lane_f16(d0_0, 1);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float16_t tmp0 = x[i] - y0[i];
        const float16_t tmp1 = x[i] - y1[i];
        dis[0] += tmp0 * tmp0;
        dis[1] += tmp1 * tmp1;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the L2 square of a float16 vector with four batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of pointers to the four batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_batch4_f16f16(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float16_t *dis)
{
    constexpr size_t single_round = 8;
    size_t i;
    if (likely(d >= single_round)) {
        float16x8_t b = vld1q_f16(x);

        float16x8_t q0 = vld1q_f16(y[0]);
        float16x8_t q1 = vld1q_f16(y[1]);
        float16x8_t q2 = vld1q_f16(y[2]);
        float16x8_t q3 = vld1q_f16(y[3]);

        q0 = vsubq_f16(q0, b);
        q1 = vsubq_f16(q1, b);
        q2 = vsubq_f16(q2, b);
        q3 = vsubq_f16(q3, b);

        float16x8_t res0 = vmulq_f16(q0, q0);
        float16x8_t res1 = vmulq_f16(q1, q1);
        float16x8_t res2 = vmulq_f16(q2, q2);
        float16x8_t res3 = vmulq_f16(q3, q3);

        for (i = single_round; i <= d - single_round; i += single_round) {
            b = vld1q_f16(x + i);

            q0 = vld1q_f16(y[0] + i);
            q1 = vld1q_f16(y[1] + i);
            q2 = vld1q_f16(y[2] + i);
            q3 = vld1q_f16(y[3] + i);

            q0 = vsubq_f16(q0, b);
            q1 = vsubq_f16(q1, b);
            q2 = vsubq_f16(q2, b);
            q3 = vsubq_f16(q3, b);

            res0 = vfmaq_f16(res0, q0, q0);
            res1 = vfmaq_f16(res1, q1, q1);
            res2 = vfmaq_f16(res2, q2, q2);
            res3 = vfmaq_f16(res3, q3, q3);
        }
        res0 = vpaddq_f16(res0, res1);
        res2 = vpaddq_f16(res2, res3);
        res0 = vpaddq_f16(res0, res2);
        res0 = vpaddq_f16(res0, res0);
        vst1_f16(dis, vget_low_f16(res0));
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (d > i) {
        float16_t q0 = x[i] - *(y[0] + i);
        float16_t q1 = x[i] - *(y[1] + i);
        float16_t q2 = x[i] - *(y[2] + i);
        float16_t q3 = x[i] - *(y[3] + i);
        float16_t d0 = q0 * q0;
        float16_t d1 = q1 * q1;
        float16_t d2 = q2 * q2;
        float16_t d3 = q3 * q3;
        for (i++; i < d; ++i) {
            float16_t q0 = x[i] - *(y[0] + i);
            float16_t q1 = x[i] - *(y[1] + i);
            float16_t q2 = x[i] - *(y[2] + i);
            float16_t q3 = x[i] - *(y[3] + i);
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
 * @brief Compute the L2 square of a float16 vector with eight batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of pointers to the eight batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_prefetch_batch8_f16f16(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */
    if (d >= multi_round) {
        float16x8_t neon_res1 = vdupq_n_f16(0.0);
        float16x8_t neon_res2 = vdupq_n_f16(0.0);
        float16x8_t neon_res3 = vdupq_n_f16(0.0);
        float16x8_t neon_res4 = vdupq_n_f16(0.0);
        float16x8_t neon_res5 = vdupq_n_f16(0.0);
        float16x8_t neon_res6 = vdupq_n_f16(0.0);
        float16x8_t neon_res7 = vdupq_n_f16(0.0);
        float16x8_t neon_res8 = vdupq_n_f16(0.0);
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

                neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);
                neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_base5);
                neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_base6);
                neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_base7);
                neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_base8);
            }
        }
        for (; i <= d - single_round; i += single_round) {
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

            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_base8);
        }
        neon_res1 = vpaddq_f16(neon_res1, neon_res2);
        neon_res3 = vpaddq_f16(neon_res3, neon_res4);
        neon_res5 = vpaddq_f16(neon_res5, neon_res6);
        neon_res7 = vpaddq_f16(neon_res7, neon_res8);
        neon_res1 = vpaddq_f16(neon_res1, neon_res3);
        neon_res5 = vpaddq_f16(neon_res5, neon_res7);
        neon_res1 = vpaddq_f16(neon_res1, neon_res5);
        vst1q_f16(dis, neon_res1);
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

        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        neon_base5 = vsubq_f16(neon_base5, neon_query);
        neon_base6 = vsubq_f16(neon_base6, neon_query);
        neon_base7 = vsubq_f16(neon_base7, neon_query);
        neon_base8 = vsubq_f16(neon_base8, neon_query);

        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_base4);
        float16x8_t neon_res5 = vmulq_f16(neon_base5, neon_base5);
        float16x8_t neon_res6 = vmulq_f16(neon_base6, neon_base6);
        float16x8_t neon_res7 = vmulq_f16(neon_base7, neon_base7);
        float16x8_t neon_res8 = vmulq_f16(neon_base8, neon_base8);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y[0] + i);
            neon_base2 = vld1q_f16(y[1] + i);
            neon_base3 = vld1q_f16(y[2] + i);
            neon_base4 = vld1q_f16(y[3] + i);
            neon_base5 = vld1q_f16(y[4] + i);
            neon_base6 = vld1q_f16(y[5] + i);
            neon_base7 = vld1q_f16(y[6] + i);
            neon_base8 = vld1q_f16(y[7] + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_base8);
        }
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
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    /* Handle any remaining elements */
    if (i < d) {
        float16_t q0 = x[i] - *(y[0] + i);
        float16_t q1 = x[i] - *(y[1] + i);
        float16_t q2 = x[i] - *(y[2] + i);
        float16_t q3 = x[i] - *(y[3] + i);
        float16_t q4 = x[i] - *(y[4] + i);
        float16_t q5 = x[i] - *(y[5] + i);
        float16_t q6 = x[i] - *(y[6] + i);
        float16_t q7 = x[i] - *(y[7] + i);
        float16_t d0 = q0 * q0;
        float16_t d1 = q1 * q1;
        float16_t d2 = q2 * q2;
        float16_t d3 = q3 * q3;
        float16_t d4 = q4 * q4;
        float16_t d5 = q5 * q5;
        float16_t d6 = q6 * q6;
        float16_t d7 = q7 * q7;
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
 * @brief Compute the L2 square of a float16 vector with 16 batches of float16 vectors using NEON instructions with
 * prefetch optimization
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of pointers to the 16 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_prefetch_batch16_f16f16(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */
    if (d >= multi_round) {
        float16x8_t neon_res1 = vdupq_n_f16(0.0);
        float16x8_t neon_res2 = vdupq_n_f16(0.0);
        float16x8_t neon_res3 = vdupq_n_f16(0.0);
        float16x8_t neon_res4 = vdupq_n_f16(0.0);
        float16x8_t neon_res5 = vdupq_n_f16(0.0);
        float16x8_t neon_res6 = vdupq_n_f16(0.0);
        float16x8_t neon_res7 = vdupq_n_f16(0.0);
        float16x8_t neon_res8 = vdupq_n_f16(0.0);
        float16x8_t neon_res9 = vdupq_n_f16(0.0);
        float16x8_t neon_res10 = vdupq_n_f16(0.0);
        float16x8_t neon_res11 = vdupq_n_f16(0.0);
        float16x8_t neon_res12 = vdupq_n_f16(0.0);
        float16x8_t neon_res13 = vdupq_n_f16(0.0);
        float16x8_t neon_res14 = vdupq_n_f16(0.0);
        float16x8_t neon_res15 = vdupq_n_f16(0.0);
        float16x8_t neon_res16 = vdupq_n_f16(0.0);
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

                neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);
                neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_base5);
                neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_base6);
                neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_base7);
                neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_base8);

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

                neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);
                neon_res13 = vfmaq_f16(neon_res13, neon_base5, neon_base5);
                neon_res14 = vfmaq_f16(neon_res14, neon_base6, neon_base6);
                neon_res15 = vfmaq_f16(neon_res15, neon_base7, neon_base7);
                neon_res16 = vfmaq_f16(neon_res16, neon_base8, neon_base8);
            }
        }
        for (; i <= d - single_round; i += single_round) {
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

            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_base8);

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

            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmaq_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmaq_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmaq_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmaq_f16(neon_res16, neon_base8, neon_base8);
        }
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
        vst1q_f16(dis, neon_res1);
        vst1q_f16(dis + 8, neon_res9);
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

        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        neon_base5 = vsubq_f16(neon_base5, neon_query);
        neon_base6 = vsubq_f16(neon_base6, neon_query);
        neon_base7 = vsubq_f16(neon_base7, neon_query);
        neon_base8 = vsubq_f16(neon_base8, neon_query);

        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_base4);
        float16x8_t neon_res5 = vmulq_f16(neon_base5, neon_base5);
        float16x8_t neon_res6 = vmulq_f16(neon_base6, neon_base6);
        float16x8_t neon_res7 = vmulq_f16(neon_base7, neon_base7);
        float16x8_t neon_res8 = vmulq_f16(neon_base8, neon_base8);

        neon_base1 = vld1q_f16(y[8]);
        neon_base2 = vld1q_f16(y[9]);
        neon_base3 = vld1q_f16(y[10]);
        neon_base4 = vld1q_f16(y[11]);
        neon_base5 = vld1q_f16(y[12]);
        neon_base6 = vld1q_f16(y[13]);
        neon_base7 = vld1q_f16(y[14]);
        neon_base8 = vld1q_f16(y[15]);

        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        neon_base5 = vsubq_f16(neon_base5, neon_query);
        neon_base6 = vsubq_f16(neon_base6, neon_query);
        neon_base7 = vsubq_f16(neon_base7, neon_query);
        neon_base8 = vsubq_f16(neon_base8, neon_query);

        float16x8_t neon_res9 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res10 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res11 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res12 = vmulq_f16(neon_base4, neon_base4);
        float16x8_t neon_res13 = vmulq_f16(neon_base5, neon_base5);
        float16x8_t neon_res14 = vmulq_f16(neon_base6, neon_base6);
        float16x8_t neon_res15 = vmulq_f16(neon_base7, neon_base7);
        float16x8_t neon_res16 = vmulq_f16(neon_base8, neon_base8);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y[0] + i);
            neon_base2 = vld1q_f16(y[1] + i);
            neon_base3 = vld1q_f16(y[2] + i);
            neon_base4 = vld1q_f16(y[3] + i);
            neon_base5 = vld1q_f16(y[4] + i);
            neon_base6 = vld1q_f16(y[5] + i);
            neon_base7 = vld1q_f16(y[6] + i);
            neon_base8 = vld1q_f16(y[7] + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_base8);

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

            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmaq_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmaq_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmaq_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmaq_f16(neon_res16, neon_base8, neon_base8);
        }
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
        vst1q_f16(dis, neon_res1);
        vst1q_f16(dis + 8, neon_res9);
    } else {
        for (int i = 0; i < 16; i++) {
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
        float16_t d0 = q0 * q0;
        float16_t d1 = q1 * q1;
        float16_t d2 = q2 * q2;
        float16_t d3 = q3 * q3;
        float16_t d4 = q4 * q4;
        float16_t d5 = q5 * q5;
        float16_t d6 = q6 * q6;
        float16_t d7 = q7 * q7;
        q0 = x[i] - *(y[8] + i);
        q1 = x[i] - *(y[9] + i);
        q2 = x[i] - *(y[10] + i);
        q3 = x[i] - *(y[11] + i);
        q4 = x[i] - *(y[12] + i);
        q5 = x[i] - *(y[13] + i);
        q6 = x[i] - *(y[14] + i);
        q7 = x[i] - *(y[15] + i);
        float16_t d8 = q0 * q0;
        float16_t d9 = q1 * q1;
        float16_t d10 = q2 * q2;
        float16_t d11 = q3 * q3;
        float16_t d12 = q4 * q4;
        float16_t d13 = q5 * q5;
        float16_t d14 = q6 * q6;
        float16_t d15 = q7 * q7;
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
 * @brief Compute the L2 square of a float16 vector with 24 batches of float16 vectors using NEON instructions with
 * prefetch optimization
 * @param x Pointer to the input float16 vector
 * @param y Pointer to an array of pointers to the 24 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_idx_prefetch_batch24_f16f16(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */
    if (d >= multi_round) {
        prefetch_L1(x + multi_round);
        prefetch_Lx(y[0] + multi_round);
        prefetch_Lx(y[1] + multi_round);
        prefetch_Lx(y[2] + multi_round);
        prefetch_Lx(y[3] + multi_round);
        prefetch_Lx(y[4] + multi_round);
        prefetch_Lx(y[5] + multi_round);
        prefetch_Lx(y[6] + multi_round);
        prefetch_Lx(y[7] + multi_round);
        prefetch_Lx(y[8] + multi_round);
        prefetch_Lx(y[9] + multi_round);
        prefetch_Lx(y[10] + multi_round);
        prefetch_Lx(y[11] + multi_round);
        prefetch_Lx(y[12] + multi_round);
        prefetch_Lx(y[13] + multi_round);
        prefetch_Lx(y[14] + multi_round);
        prefetch_Lx(y[15] + multi_round);
        prefetch_Lx(y[16] + multi_round);
        prefetch_Lx(y[17] + multi_round);
        prefetch_Lx(y[18] + multi_round);
        prefetch_Lx(y[19] + multi_round);
        prefetch_Lx(y[20] + multi_round);
        prefetch_Lx(y[21] + multi_round);
        prefetch_Lx(y[22] + multi_round);
        prefetch_Lx(y[23] + multi_round);
        float16x8_t neon_res1, neon_res2, neon_res3, neon_res4;
        float16x8_t neon_res5, neon_res6, neon_res7, neon_res8;
        float16x8_t neon_res9, neon_res10, neon_res11, neon_res12;
        float16x8_t neon_res13, neon_res14, neon_res15, neon_res16;
        float16x8_t neon_res17, neon_res18, neon_res19, neon_res20;
        float16x8_t neon_res21, neon_res22, neon_res23, neon_res24;
        {
            const float16x8_t neon_query = vld1q_f16(x);
            float16x8_t neon_base1 = vld1q_f16(y[0]);
            float16x8_t neon_base2 = vld1q_f16(y[1]);
            float16x8_t neon_base3 = vld1q_f16(y[2]);
            float16x8_t neon_base4 = vld1q_f16(y[3]);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vmulq_f16(neon_base1, neon_base1);
            neon_res2 = vmulq_f16(neon_base2, neon_base2);
            neon_res3 = vmulq_f16(neon_base3, neon_base3);
            neon_res4 = vmulq_f16(neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[4]);
            neon_base2 = vld1q_f16(y[5]);
            neon_base3 = vld1q_f16(y[6]);
            neon_base4 = vld1q_f16(y[7]);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vmulq_f16(neon_base1, neon_base1);
            neon_res6 = vmulq_f16(neon_base2, neon_base2);
            neon_res7 = vmulq_f16(neon_base3, neon_base3);
            neon_res8 = vmulq_f16(neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[8]);
            neon_base2 = vld1q_f16(y[9]);
            neon_base3 = vld1q_f16(y[10]);
            neon_base4 = vld1q_f16(y[11]);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vmulq_f16(neon_base1, neon_base1);
            neon_res10 = vmulq_f16(neon_base2, neon_base2);
            neon_res11 = vmulq_f16(neon_base3, neon_base3);
            neon_res12 = vmulq_f16(neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[12]);
            neon_base2 = vld1q_f16(y[13]);
            neon_base3 = vld1q_f16(y[14]);
            neon_base4 = vld1q_f16(y[15]);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vmulq_f16(neon_base1, neon_base1);
            neon_res14 = vmulq_f16(neon_base2, neon_base2);
            neon_res15 = vmulq_f16(neon_base3, neon_base3);
            neon_res16 = vmulq_f16(neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[16]);
            neon_base2 = vld1q_f16(y[17]);
            neon_base3 = vld1q_f16(y[18]);
            neon_base4 = vld1q_f16(y[19]);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vmulq_f16(neon_base1, neon_base1);
            neon_res18 = vmulq_f16(neon_base2, neon_base2);
            neon_res19 = vmulq_f16(neon_base3, neon_base3);
            neon_res20 = vmulq_f16(neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[20]);
            neon_base2 = vld1q_f16(y[21]);
            neon_base3 = vld1q_f16(y[22]);
            neon_base4 = vld1q_f16(y[23]);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vmulq_f16(neon_base1, neon_base1);
            neon_res22 = vmulq_f16(neon_base2, neon_base2);
            neon_res23 = vmulq_f16(neon_base3, neon_base3);
            neon_res24 = vmulq_f16(neon_base4, neon_base4);
        }
        for (i = single_round; i < multi_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y[0] + i);
            float16x8_t neon_base2 = vld1q_f16(y[1] + i);
            float16x8_t neon_base3 = vld1q_f16(y[2] + i);
            float16x8_t neon_base4 = vld1q_f16(y[3] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmaq_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmaq_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmaq_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmaq_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmaq_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmaq_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmaq_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmaq_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmaq_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmaq_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmaq_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmaq_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmaq_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmaq_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmaq_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmaq_f16(neon_res24, neon_base4, neon_base4);
        }
        for (; i < d - multi_round; i += multi_round) {
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
            prefetch_Lx(y[16] + i + multi_round);
            prefetch_Lx(y[17] + i + multi_round);
            prefetch_Lx(y[18] + i + multi_round);
            prefetch_Lx(y[19] + i + multi_round);
            prefetch_Lx(y[20] + i + multi_round);
            prefetch_Lx(y[21] + i + multi_round);
            prefetch_Lx(y[22] + i + multi_round);
            prefetch_Lx(y[23] + i + multi_round);
            for (size_t j = i; j < i + multi_round; j += single_round) {
                const float16x8_t neon_query = vld1q_f16(x + j);
                float16x8_t neon_base1 = vld1q_f16(y[0] + j);
                float16x8_t neon_base2 = vld1q_f16(y[1] + j);
                float16x8_t neon_base3 = vld1q_f16(y[2] + j);
                float16x8_t neon_base4 = vld1q_f16(y[3] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[4] + j);
                neon_base2 = vld1q_f16(y[5] + j);
                neon_base3 = vld1q_f16(y[6] + j);
                neon_base4 = vld1q_f16(y[7] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res5 = vfmaq_f16(neon_res5, neon_base1, neon_base1);
                neon_res6 = vfmaq_f16(neon_res6, neon_base2, neon_base2);
                neon_res7 = vfmaq_f16(neon_res7, neon_base3, neon_base3);
                neon_res8 = vfmaq_f16(neon_res8, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[8] + j);
                neon_base2 = vld1q_f16(y[9] + j);
                neon_base3 = vld1q_f16(y[10] + j);
                neon_base4 = vld1q_f16(y[11] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[12] + j);
                neon_base2 = vld1q_f16(y[13] + j);
                neon_base3 = vld1q_f16(y[14] + j);
                neon_base4 = vld1q_f16(y[15] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res13 = vfmaq_f16(neon_res13, neon_base1, neon_base1);
                neon_res14 = vfmaq_f16(neon_res14, neon_base2, neon_base2);
                neon_res15 = vfmaq_f16(neon_res15, neon_base3, neon_base3);
                neon_res16 = vfmaq_f16(neon_res16, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[16] + j);
                neon_base2 = vld1q_f16(y[17] + j);
                neon_base3 = vld1q_f16(y[18] + j);
                neon_base4 = vld1q_f16(y[19] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res17 = vfmaq_f16(neon_res17, neon_base1, neon_base1);
                neon_res18 = vfmaq_f16(neon_res18, neon_base2, neon_base2);
                neon_res19 = vfmaq_f16(neon_res19, neon_base3, neon_base3);
                neon_res20 = vfmaq_f16(neon_res20, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y[20] + j);
                neon_base2 = vld1q_f16(y[21] + j);
                neon_base3 = vld1q_f16(y[22] + j);
                neon_base4 = vld1q_f16(y[23] + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res21 = vfmaq_f16(neon_res21, neon_base1, neon_base1);
                neon_res22 = vfmaq_f16(neon_res22, neon_base2, neon_base2);
                neon_res23 = vfmaq_f16(neon_res23, neon_base3, neon_base3);
                neon_res24 = vfmaq_f16(neon_res24, neon_base4, neon_base4);
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
            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmaq_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmaq_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmaq_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmaq_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmaq_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmaq_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmaq_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmaq_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmaq_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmaq_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmaq_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmaq_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmaq_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmaq_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmaq_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmaq_f16(neon_res24, neon_base4, neon_base4);
        }
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
        vst1q_f16(dis, neon_res1);
        vst1q_f16(dis + 8, neon_res9);
        vst1q_f16(dis + 16, neon_res17);
    } else if (d >= single_round) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y[0]);
        float16x8_t neon_base2 = vld1q_f16(y[1]);
        float16x8_t neon_base3 = vld1q_f16(y[2]);
        float16x8_t neon_base4 = vld1q_f16(y[3]);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y[4]);
        neon_base2 = vld1q_f16(y[5]);
        neon_base3 = vld1q_f16(y[6]);
        neon_base4 = vld1q_f16(y[7]);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res5 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res6 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res7 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res8 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y[8]);
        neon_base2 = vld1q_f16(y[9]);
        neon_base3 = vld1q_f16(y[10]);
        neon_base4 = vld1q_f16(y[11]);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res9 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res10 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res11 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res12 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y[12]);
        neon_base2 = vld1q_f16(y[13]);
        neon_base3 = vld1q_f16(y[14]);
        neon_base4 = vld1q_f16(y[15]);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res13 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res14 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res15 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res16 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y[16]);
        neon_base2 = vld1q_f16(y[17]);
        neon_base3 = vld1q_f16(y[18]);
        neon_base4 = vld1q_f16(y[19]);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res17 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res18 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res19 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res20 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y[20]);
        neon_base2 = vld1q_f16(y[21]);
        neon_base3 = vld1q_f16(y[22]);
        neon_base4 = vld1q_f16(y[23]);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res21 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res22 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res23 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res24 = vmulq_f16(neon_base4, neon_base4);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y[0] + i);
            neon_base2 = vld1q_f16(y[1] + i);
            neon_base3 = vld1q_f16(y[2] + i);
            neon_base4 = vld1q_f16(y[3] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[4] + i);
            neon_base2 = vld1q_f16(y[5] + i);
            neon_base3 = vld1q_f16(y[6] + i);
            neon_base4 = vld1q_f16(y[7] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmaq_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmaq_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmaq_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmaq_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[8] + i);
            neon_base2 = vld1q_f16(y[9] + i);
            neon_base3 = vld1q_f16(y[10] + i);
            neon_base4 = vld1q_f16(y[11] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[12] + i);
            neon_base2 = vld1q_f16(y[13] + i);
            neon_base3 = vld1q_f16(y[14] + i);
            neon_base4 = vld1q_f16(y[15] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmaq_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmaq_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmaq_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmaq_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[16] + i);
            neon_base2 = vld1q_f16(y[17] + i);
            neon_base3 = vld1q_f16(y[18] + i);
            neon_base4 = vld1q_f16(y[19] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmaq_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmaq_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmaq_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmaq_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y[20] + i);
            neon_base2 = vld1q_f16(y[21] + i);
            neon_base3 = vld1q_f16(y[22] + i);
            neon_base4 = vld1q_f16(y[23] + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmaq_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmaq_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmaq_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmaq_f16(neon_res24, neon_base4, neon_base4);
        }
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
        vst1q_f16(dis, neon_res1);
        vst1q_f16(dis + 8, neon_res9);
        vst1q_f16(dis + 16, neon_res17);
    } else {
        for (int i = 0; i < 24; i++) {
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
        float16_t d0 = q0 * q0;
        float16_t d1 = q1 * q1;
        float16_t d2 = q2 * q2;
        float16_t d3 = q3 * q3;
        float16_t d4 = q4 * q4;
        float16_t d5 = q5 * q5;
        float16_t d6 = q6 * q6;
        float16_t d7 = q7 * q7;
        q0 = x[i] - *(y[8] + i);
        q1 = x[i] - *(y[9] + i);
        q2 = x[i] - *(y[10] + i);
        q3 = x[i] - *(y[11] + i);
        q4 = x[i] - *(y[12] + i);
        q5 = x[i] - *(y[13] + i);
        q6 = x[i] - *(y[14] + i);
        q7 = x[i] - *(y[15] + i);
        float16_t d8 = q0 * q0;
        float16_t d9 = q1 * q1;
        float16_t d10 = q2 * q2;
        float16_t d11 = q3 * q3;
        float16_t d12 = q4 * q4;
        float16_t d13 = q5 * q5;
        float16_t d14 = q6 * q6;
        float16_t d15 = q7 * q7;
        q0 = x[i] - *(y[16] + i);
        q1 = x[i] - *(y[17] + i);
        q2 = x[i] - *(y[18] + i);
        q3 = x[i] - *(y[19] + i);
        q4 = x[i] - *(y[20] + i);
        q5 = x[i] - *(y[21] + i);
        q6 = x[i] - *(y[22] + i);
        q7 = x[i] - *(y[23] + i);
        float16_t d16 = q0 * q0;
        float16_t d17 = q1 * q1;
        float16_t d18 = q2 * q2;
        float16_t d19 = q3 * q3;
        float16_t d20 = q4 * q4;
        float16_t d21 = q5 * q5;
        float16_t d22 = q6 * q6;
        float16_t d23 = q7 * q7;
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
 * @brief Compute the L2 square of a float16 vector with 2 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 2 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch2_f16f16(const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
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

        float16x8_t d0_0 = vsubq_f16(x_0, y0_0);
        d0_0 = vmulq_f16(d0_0, d0_0);
        float16x8_t d0_1 = vsubq_f16(x_1, y0_1);
        d0_1 = vmulq_f16(d0_1, d0_1);
        float16x8_t d1_0 = vsubq_f16(x_0, y1_0);
        d1_0 = vmulq_f16(d1_0, d1_0);
        float16x8_t d1_1 = vsubq_f16(x_1, y1_1);
        d1_1 = vmulq_f16(d1_1, d1_1);

        for (i = double_round; i <= d - double_round; i += double_round) {
            x_0 = vld1q_f16(x + i);
            y0_0 = vld1q_f16(y + i);
            y1_0 = vld1q_f16(y + d + i);
            const float16x8_t q0_0 = vsubq_f16(x_0, y0_0);
            const float16x8_t q1_0 = vsubq_f16(x_0, y1_0);
            d0_0 = vfmaq_f16(d0_0, q0_0, q0_0);
            d1_0 = vfmaq_f16(d1_0, q1_0, q1_0);

            x_1 = vld1q_f16(x + i + 8);
            y0_1 = vld1q_f16(y + i + 8);
            y1_1 = vld1q_f16(y + d + i + 8);
            const float16x8_t q0_1 = vsubq_f16(x_1, y0_1);
            const float16x8_t q1_1 = vsubq_f16(x_1, y1_1);
            d0_1 = vfmaq_f16(d0_1, q0_1, q0_1);
            d1_1 = vfmaq_f16(d1_1, q1_1, q1_1);
        }
        d0_0 = vaddq_f16(d0_0, d0_1);
        d1_0 = vaddq_f16(d1_0, d1_1);
        /* 8 -> 4 */
        d0_0 = vpaddq_f16(d0_0, d1_0);
        /* 4 -> 2 */
        d0_0 = vpaddq_f16(d0_0, d0_0);
        /* 2 -> 1 */
        d0_0 = vpaddq_f16(d0_0, d0_0);
        dis[0] = vgetq_lane_f16(d0_0, 0);
        dis[1] = vgetq_lane_f16(d0_0, 1);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float16_t tmp0 = x[i] - *(y + i);
        const float16_t tmp1 = x[i] - *(y + d + i);
        dis[0] += tmp0 * tmp0;
        dis[1] += tmp1 * tmp1;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the L2 square of a float16 vector with 4 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 4 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch4_f16f16(const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
{
    constexpr size_t single_round = 8;
    size_t i;
    if (likely(d >= single_round)) {
        float16x8_t b = vld1q_f16(x);

        float16x8_t q0 = vld1q_f16(y);
        float16x8_t q1 = vld1q_f16(y + d);
        float16x8_t q2 = vld1q_f16(y + 2 * d);
        float16x8_t q3 = vld1q_f16(y + 3 * d);

        q0 = vsubq_f16(q0, b);
        q1 = vsubq_f16(q1, b);
        q2 = vsubq_f16(q2, b);
        q3 = vsubq_f16(q3, b);

        float16x8_t res0 = vmulq_f16(q0, q0);
        float16x8_t res1 = vmulq_f16(q1, q1);
        float16x8_t res2 = vmulq_f16(q2, q2);
        float16x8_t res3 = vmulq_f16(q3, q3);

        for (i = single_round; i <= d - single_round; i += single_round) {
            b = vld1q_f16(x + i);

            q0 = vld1q_f16(y + i);
            q1 = vld1q_f16(y + d + i);
            q2 = vld1q_f16(y + 2 * d + i);
            q3 = vld1q_f16(y + 3 * d + i);

            q0 = vsubq_f16(q0, b);
            q1 = vsubq_f16(q1, b);
            q2 = vsubq_f16(q2, b);
            q3 = vsubq_f16(q3, b);

            res0 = vfmaq_f16(res0, q0, q0);
            res1 = vfmaq_f16(res1, q1, q1);
            res2 = vfmaq_f16(res2, q2, q2);
            res3 = vfmaq_f16(res3, q3, q3);
        }
        res0 = vpaddq_f16(res0, res1);
        res2 = vpaddq_f16(res2, res3);
        res0 = vpaddq_f16(res0, res2);
        res0 = vpaddq_f16(res0, res0);
        vst1_f16(dis, vget_low_f16(res0));
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (d > i) {
        float16_t q0 = x[i] - *(y + i);
        float16_t q1 = x[i] - *(y + d + i);
        float16_t q2 = x[i] - *(y + 2 * d + i);
        float16_t q3 = x[i] - *(y + 3 * d + i);
        float16_t d0 = q0 * q0;
        float16_t d1 = q1 * q1;
        float16_t d2 = q2 * q2;
        float16_t d3 = q3 * q3;
        for (i++; i < d; ++i) {
            float16_t q0 = x[i] - *(y + i);
            float16_t q1 = x[i] - *(y + d + i);
            float16_t q2 = x[i] - *(y + 2 * d + i);
            float16_t q3 = x[i] - *(y + 3 * d + i);
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
 * @brief Compute the L2 square of a float16 vector with 8 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 8 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 squares
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch8_f16f16(const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    if (likely(d >= single_round)) {
        float16x8_t neon_query = vld1q_f16(x);

        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        float16x8_t neon_base5 = vld1q_f16(y + 4 * d);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d);

        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        neon_base5 = vsubq_f16(neon_base5, neon_query);
        neon_base6 = vsubq_f16(neon_base6, neon_query);
        neon_base7 = vsubq_f16(neon_base7, neon_query);
        neon_base8 = vsubq_f16(neon_base8, neon_query);

        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_base4);
        float16x8_t neon_res5 = vmulq_f16(neon_base5, neon_base5);
        float16x8_t neon_res6 = vmulq_f16(neon_base6, neon_base6);
        float16x8_t neon_res7 = vmulq_f16(neon_base7, neon_base7);
        float16x8_t neon_res8 = vmulq_f16(neon_base8, neon_base8);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);

            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base5 = vld1q_f16(y + 4 * d + i);
            neon_base6 = vld1q_f16(y + 5 * d + i);
            neon_base7 = vld1q_f16(y + 6 * d + i);
            neon_base8 = vld1q_f16(y + 7 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_base8);
        }
        neon_res1 = vpaddq_f16(neon_res1, neon_res2);
        neon_res3 = vpaddq_f16(neon_res3, neon_res4);
        neon_res5 = vpaddq_f16(neon_res5, neon_res6);
        neon_res7 = vpaddq_f16(neon_res7, neon_res8);
        neon_res1 = vpaddq_f16(neon_res1, neon_res3);
        neon_res5 = vpaddq_f16(neon_res5, neon_res7);
        neon_res1 = vpaddq_f16(neon_res1, neon_res5);
        vst1q_f16(dis, neon_res1);
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
        float16_t d0 = q0 * q0;
        float16_t d1 = q1 * q1;
        float16_t d2 = q2 * q2;
        float16_t d3 = q3 * q3;
        float16_t d4 = q4 * q4;
        float16_t d5 = q5 * q5;
        float16_t d6 = q6 * q6;
        float16_t d7 = q7 * q7;
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
 * @brief Compute the L2 square of a float16 vector with 16 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 16 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 square s
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch16_f16f16(const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    if (likely(d >= single_round)) {
        float16x8_t neon_query = vld1q_f16(x);

        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_base4);

        float16x8_t neon_base5 = vld1q_f16(y + 4 * d);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d);
        neon_base5 = vsubq_f16(neon_base5, neon_query);
        neon_base6 = vsubq_f16(neon_base6, neon_query);
        neon_base7 = vsubq_f16(neon_base7, neon_query);
        neon_base8 = vsubq_f16(neon_base8, neon_query);
        float16x8_t neon_res5 = vmulq_f16(neon_base5, neon_base5);
        float16x8_t neon_res6 = vmulq_f16(neon_base6, neon_base6);
        float16x8_t neon_res7 = vmulq_f16(neon_base7, neon_base7);
        float16x8_t neon_res8 = vmulq_f16(neon_base8, neon_base8);

        neon_base1 = vld1q_f16(y + 8 * d);
        neon_base2 = vld1q_f16(y + 9 * d);
        neon_base3 = vld1q_f16(y + 10 * d);
        neon_base4 = vld1q_f16(y + 11 * d);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res9 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res10 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res11 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res12 = vmulq_f16(neon_base4, neon_base4);

        neon_base5 = vld1q_f16(y + 12 * d);
        neon_base6 = vld1q_f16(y + 13 * d);
        neon_base7 = vld1q_f16(y + 14 * d);
        neon_base8 = vld1q_f16(y + 15 * d);
        neon_base5 = vsubq_f16(neon_base5, neon_query);
        neon_base6 = vsubq_f16(neon_base6, neon_query);
        neon_base7 = vsubq_f16(neon_base7, neon_query);
        neon_base8 = vsubq_f16(neon_base8, neon_query);
        float16x8_t neon_res13 = vmulq_f16(neon_base5, neon_base5);
        float16x8_t neon_res14 = vmulq_f16(neon_base6, neon_base6);
        float16x8_t neon_res15 = vmulq_f16(neon_base7, neon_base7);
        float16x8_t neon_res16 = vmulq_f16(neon_base8, neon_base8);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);

            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);

            neon_base5 = vld1q_f16(y + 4 * d + i);
            neon_base6 = vld1q_f16(y + 5 * d + i);
            neon_base7 = vld1q_f16(y + 6 * d + i);
            neon_base8 = vld1q_f16(y + 7 * d + i);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);
            neon_res5 = vfmaq_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmaq_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmaq_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmaq_f16(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);

            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);
            neon_res13 = vfmaq_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmaq_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmaq_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmaq_f16(neon_res16, neon_base8, neon_base8);
        }
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
        vst1q_f16(dis, neon_res1);
        vst1q_f16(dis + 8, neon_res9);
    } else {
        for (int i = 0; i < 16; i++) {
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
        float16_t d0 = q0 * q0;
        float16_t d1 = q1 * q1;
        float16_t d2 = q2 * q2;
        float16_t d3 = q3 * q3;
        float16_t d4 = q4 * q4;
        float16_t d5 = q5 * q5;
        float16_t d6 = q6 * q6;
        float16_t d7 = q7 * q7;
        q0 = x[i] - *(y + 8 * d + i);
        q1 = x[i] - *(y + 9 * d + i);
        q2 = x[i] - *(y + 10 * d + i);
        q3 = x[i] - *(y + 11 * d + i);
        q4 = x[i] - *(y + 12 * d + i);
        q5 = x[i] - *(y + 13 * d + i);
        q6 = x[i] - *(y + 14 * d + i);
        q7 = x[i] - *(y + 15 * d + i);
        float16_t d8 = q0 * q0;
        float16_t d9 = q1 * q1;
        float16_t d10 = q2 * q2;
        float16_t d11 = q3 * q3;
        float16_t d12 = q4 * q4;
        float16_t d13 = q5 * q5;
        float16_t d14 = q6 * q6;
        float16_t d15 = q7 * q7;
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
 * @brief Compute the L2 square of a float16 vector with 24 batches of float16 vectors using NEON instructions
 * @param x Pointer to the input float16 vector
 * @param y Pointer to the array of pointers to the 16 batches of float16 vectors (with restrict qualifier for better
 * optimization)
 * @param d The dimension of the vectors
 * @param dis Pointer to the array storing the computed L2 square s
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch24_f16f16(const float16_t *x, const float16_t *__restrict y, const size_t d, float16_t *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    if (likely(d >= single_round)) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res1 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res2 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res3 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res4 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y + 4 * d);
        neon_base2 = vld1q_f16(y + 5 * d);
        neon_base3 = vld1q_f16(y + 6 * d);
        neon_base4 = vld1q_f16(y + 7 * d);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res5 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res6 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res7 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res8 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y + 8 * d);
        neon_base2 = vld1q_f16(y + 9 * d);
        neon_base3 = vld1q_f16(y + 10 * d);
        neon_base4 = vld1q_f16(y + 11 * d);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res9 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res10 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res11 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res12 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y + 12 * d);
        neon_base2 = vld1q_f16(y + 13 * d);
        neon_base3 = vld1q_f16(y + 14 * d);
        neon_base4 = vld1q_f16(y + 15 * d);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res13 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res14 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res15 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res16 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y + 16 * d);
        neon_base2 = vld1q_f16(y + 17 * d);
        neon_base3 = vld1q_f16(y + 18 * d);
        neon_base4 = vld1q_f16(y + 19 * d);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res17 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res18 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res19 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res20 = vmulq_f16(neon_base4, neon_base4);

        neon_base1 = vld1q_f16(y + 20 * d);
        neon_base2 = vld1q_f16(y + 21 * d);
        neon_base3 = vld1q_f16(y + 22 * d);
        neon_base4 = vld1q_f16(y + 23 * d);
        neon_base1 = vsubq_f16(neon_base1, neon_query);
        neon_base2 = vsubq_f16(neon_base2, neon_query);
        neon_base3 = vsubq_f16(neon_base3, neon_query);
        neon_base4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_res21 = vmulq_f16(neon_base1, neon_base1);
        float16x8_t neon_res22 = vmulq_f16(neon_base2, neon_base2);
        float16x8_t neon_res23 = vmulq_f16(neon_base3, neon_base3);
        float16x8_t neon_res24 = vmulq_f16(neon_base4, neon_base4);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmaq_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmaq_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmaq_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmaq_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 4 * d + i);
            neon_base2 = vld1q_f16(y + 5 * d + i);
            neon_base3 = vld1q_f16(y + 6 * d + i);
            neon_base4 = vld1q_f16(y + 7 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmaq_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmaq_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmaq_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmaq_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmaq_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmaq_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmaq_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmaq_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 12 * d + i);
            neon_base2 = vld1q_f16(y + 13 * d + i);
            neon_base3 = vld1q_f16(y + 14 * d + i);
            neon_base4 = vld1q_f16(y + 15 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmaq_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmaq_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmaq_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmaq_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 16 * d + i);
            neon_base2 = vld1q_f16(y + 17 * d + i);
            neon_base3 = vld1q_f16(y + 18 * d + i);
            neon_base4 = vld1q_f16(y + 19 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmaq_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmaq_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmaq_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmaq_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 20 * d + i);
            neon_base2 = vld1q_f16(y + 21 * d + i);
            neon_base3 = vld1q_f16(y + 22 * d + i);
            neon_base4 = vld1q_f16(y + 23 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmaq_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmaq_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmaq_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmaq_f16(neon_res24, neon_base4, neon_base4);
        }
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
        vst1q_f16(dis, neon_res1);
        vst1q_f16(dis + 8, neon_res9);
        vst1q_f16(dis + 16, neon_res17);
    } else {
        for (int i = 0; i < 24; i++) {
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
        float16_t d0 = q0 * q0;
        float16_t d1 = q1 * q1;
        float16_t d2 = q2 * q2;
        float16_t d3 = q3 * q3;
        float16_t d4 = q4 * q4;
        float16_t d5 = q5 * q5;
        float16_t d6 = q6 * q6;
        float16_t d7 = q7 * q7;
        q0 = x[i] - *(y + 8 * d + i);
        q1 = x[i] - *(y + 9 * d + i);
        q2 = x[i] - *(y + 10 * d + i);
        q3 = x[i] - *(y + 11 * d + i);
        q4 = x[i] - *(y + 12 * d + i);
        q5 = x[i] - *(y + 13 * d + i);
        q6 = x[i] - *(y + 14 * d + i);
        q7 = x[i] - *(y + 15 * d + i);
        float16_t d8 = q0 * q0;
        float16_t d9 = q1 * q1;
        float16_t d10 = q2 * q2;
        float16_t d11 = q3 * q3;
        float16_t d12 = q4 * q4;
        float16_t d13 = q5 * q5;
        float16_t d14 = q6 * q6;
        float16_t d15 = q7 * q7;
        q0 = x[i] - *(y + 16 * d + i);
        q1 = x[i] - *(y + 17 * d + i);
        q2 = x[i] - *(y + 18 * d + i);
        q3 = x[i] - *(y + 19 * d + i);
        q4 = x[i] - *(y + 20 * d + i);
        q5 = x[i] - *(y + 21 * d + i);
        q6 = x[i] - *(y + 22 * d + i);
        q7 = x[i] - *(y + 23 * d + i);
        float16_t d16 = q0 * q0;
        float16_t d17 = q1 * q1;
        float16_t d18 = q2 * q2;
        float16_t d19 = q3 * q3;
        float16_t d20 = q4 * q4;
        float16_t d21 = q5 * q5;
        float16_t d22 = q6 * q6;
        float16_t d23 = q7 * q7;
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
 * @brief Compute the L2 square of a float16 vector with multiple float16 vectors based on given indices
 * @param udis Pointer to the array storing the computed L2 squares
 * @param ux Pointer to the input float16 vector
 * @param uy Pointer to the array of float16 vectors
 * @param ids Pointer to the array of indices specifying which y vectors to use
 * @param d The dimension of the vectors
 * @param ny The number of y vectors to process
 */
void krl_L2sqr_by_idx_f16f16(uint16_t *udis, const uint16_t *ux, const uint16_t *uy,
    const int64_t *ids, /* ids of y vecs */
    size_t d, size_t ny)
{
    float16_t *dis = (float16_t *)udis;
    const float16_t *x = (const float16_t *)ux;
    const float16_t *y = (const float16_t *)uy;
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
        krl_L2sqr_idx_prefetch_batch24_f16f16(x, listy, d, dis + i);
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
        krl_L2sqr_idx_prefetch_batch16_f16f16(x, listy, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
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
        krl_L2sqr_idx_prefetch_batch8_f16f16(x, listy, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        krl_L2sqr_idx_batch4_f16f16(x, listy, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        const float16_t *y0 = y + *(ids + i) * d;
        const float16_t *y1 = y + *(ids + i + 1) * d;
        krl_L2sqr_idx_batch2_f16f16(x, y0, y1, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        dis[i] = krl_L2sqr_f16f16(x, y + d * ids[i], d);
    }
}

/*
 * @brief Compute the L2 square of a float16 vector with multiple float16 vectors in batches
 * @param udis Pointer to the array storing the computed L2 squares
 * @param ux Pointer to the input float16 vector
 * @param uy Pointer to the array of float16 vectors
 * @param d The dimension of the vectors
 * @param ny The number of y vectors to process
 */
void krl_L2sqr_ny_f16f16(uint16_t *udis, const uint16_t *ux, const uint16_t *uy, size_t ny, size_t d)
{
    float16_t *dis = (float16_t *)udis;
    const float16_t *x = (const float16_t *)ux;
    const float16_t *y = (const float16_t *)uy;
    size_t i = 0;
    for (; i + 24 <= ny; i += 24) {
        krl_L2sqr_batch24_f16f16(x, y + i * d, d, dis + i);
    }
    if (i + 16 <= ny) {
        krl_L2sqr_batch16_f16f16(x, y + i * d, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        krl_L2sqr_batch8_f16f16(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_L2sqr_batch4_f16f16(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_L2sqr_batch2_f16f16(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        dis[i] = krl_L2sqr_f16f16(x, y + i * d, d);
    }
}
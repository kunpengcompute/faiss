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
#include <arm_sve.h>

/*
 * @brief Compute the L2 square of two float16 vectors and return the result as float32
 * @param u16_x Pointer to the first float16 vector
 * @param u16_y Pointer to the second float16 vector (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @param dis Stores the computed L2 square result (float).
 * @param dis_size Length of dis.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
int krl_L2sqr_f16f32_sve2(
    const uint16_t *u16_x, const uint16_t *__restrict u16_y, size_t d, float *dis)
{
    const float16_t *x = (const float16_t *)u16_x;
    const float16_t *y = (const float16_t *)u16_y;

    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();

    /* 4 fp32 accumulators for deep pipelining */
    svfloat32_t acc0 = svdup_n_f32(0.0f);
    svfloat32_t acc1 = svdup_n_f32(0.0f);
    svfloat32_t acc2 = svdup_n_f32(0.0f);
    svfloat32_t acc3 = svdup_n_f32(0.0f);

    uint64_t step16 = svcnth(); /* 16 for 256-bit SVE */

    size_t i = 0;

    /* Main loop: process 4 脳 step16 = 64 fp16 elements per iteration
     * 4-way unroll for maximum instruction-level parallelism */
    for (; i + 4 * step16 <= d; i += 4 * step16) {
        svfloat16_t xv0 = svld1_f16(pg16, x + i);
        svfloat16_t yv0 = svld1_f16(pg16, y + i);
        svfloat16_t xv1 = svld1_f16(pg16, x + i + step16);
        svfloat16_t yv1 = svld1_f16(pg16, y + i + step16);
        svfloat16_t xv2 = svld1_f16(pg16, x + i + 2 * step16);
        svfloat16_t yv2 = svld1_f16(pg16, y + i + 2 * step16);
        svfloat16_t xv3 = svld1_f16(pg16, x + i + 3 * step16);
        svfloat16_t yv3 = svld1_f16(pg16, y + i + 3 * step16);

        svfloat16_t diff0 = svsub_f16_x(pg16, xv0, yv0);
        svfloat16_t diff1 = svsub_f16_x(pg16, xv1, yv1);
        svfloat16_t diff2 = svsub_f16_x(pg16, xv2, yv2);
        svfloat16_t diff3 = svsub_f16_x(pg16, xv3, yv3);

        /*
         * SVE2 FMLALB/FMLALT: fp16 multiply-add long bottom/top → fp32
         *   fmlalb processes even-indexed fp16 elements
         *   fmlalt processes odd-indexed fp16 elements
         *   Together they cover all 16 fp16 elements in 2 instructions.
         */
        acc0 = svmlalb_f32(acc0, diff0, diff0);
        acc0 = svmlalt_f32(acc0, diff0, diff0);

        acc1 = svmlalb_f32(acc1, diff1, diff1);
        acc1 = svmlalt_f32(acc1, diff1, diff1);

        acc2 = svmlalb_f32(acc2, diff2, diff2);
        acc2 = svmlalt_f32(acc2, diff2, diff2);

        acc3 = svmlalb_f32(acc3, diff3, diff3);
        acc3 = svmlalt_f32(acc3, diff3, diff3);
    }

    /* Handle remaining 16-element chunks */
    for (; i + step16 <= d; i += step16) {
        svfloat16_t xv = svld1_f16(pg16, x + i);
        svfloat16_t yv = svld1_f16(pg16, y + i);
        svfloat16_t diff = svsub_f16_x(pg16, xv, yv);

        acc0 = svmlalb_f32(acc0, diff, diff);
        acc0 = svmlalt_f32(acc0, diff, diff);
    }

    /* Handle remaining 8-element chunks */
    uint64_t step8 = svcntw(); /* 8 for 256-bit */
    for (; i + step8 <= d; i += step8) {
        /* Load only bottom half worth of fp16 data */
        svbool_t pg_half = svwhilelt_b16((uint64_t)0, (uint64_t)step8);
        svfloat16_t xv = svld1_f16(pg_half, x + i);
        svfloat16_t yv = svld1_f16(pg_half, y + i);
        svfloat16_t diff = svsub_f16_x(pg_half, xv, yv);

        /* Match 16-element path: svmlalb + svmlalt cover all fp16 lanes. */
        acc0 = svmlalb_f32(acc0, diff, diff);
        acc0 = svmlalt_f32(acc0, diff, diff);
    }

    /* Horizontal reduction */
    acc0 = svadd_f32_x(pg32, acc0, acc1);
    acc2 = svadd_f32_x(pg32, acc2, acc3);
    acc0 = svadd_f32_x(pg32, acc0, acc2);

    float res = svaddv_f32(pg32, acc0);

    /* Scalar tail */
    for (; i < d; i++) {
        float16_t tmp = x[i] - y[i];
        res += (float)(tmp * tmp);
    }

    *dis = res;
    return SUCCESS;
}
KRL_IMPRECISE_FUNCTION_END
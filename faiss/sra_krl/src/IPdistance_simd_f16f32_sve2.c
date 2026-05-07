/*
   Copyright 2026 Huawei Technologies Co., Ltd.

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
 * @brief Compute the inner product of two float16 vectors and return the result as float32
 * @param u16_x Pointer to the first float16 vector
 * @param u16_y Pointer to the second float16 vector (with restrict qualifier for better optimization)
 * @param d The dimension of the vectors
 * @return The computed inner product as float32
 */
KRL_IMPRECISE_FUNCTION_BEGIN
float krl_inner_product_f16f32_sve2(const uint16_t *u16_x, const uint16_t *__restrict u16_y, const size_t d)
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

    /* Main loop: process 4 × step16 = 64 fp16 elements per iteration
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

        /*
         * SVE2 FMLALB/FMLALT: fp16 multiply-add long bottom/top → fp32
         *   fmlalb processes even-indexed fp16 elements
         *   fmlalt processes odd-indexed fp16 elements
         *   Together they cover all 16 fp16 elements in 2 instructions.
         */
        acc0 = svmlalb_f32(acc0, xv0, yv0);
        acc0 = svmlalt_f32(acc0, xv0, yv0);

        acc1 = svmlalb_f32(acc1, xv1, yv1);
        acc1 = svmlalt_f32(acc1, xv1, yv1);

        acc2 = svmlalb_f32(acc2, xv2, yv2);
        acc2 = svmlalt_f32(acc2, xv2, yv2);

        acc3 = svmlalb_f32(acc3, xv3, yv3);
        acc3 = svmlalt_f32(acc3, xv3, yv3);
    }

    /* Handle remaining 16-element chunks */
    for (; i + step16 <= d; i += step16) {
        svfloat16_t xv = svld1_f16(pg16, x + i);
        svfloat16_t yv = svld1_f16(pg16, y + i);

        acc0 = svmlalb_f32(acc0, xv, yv);
        acc0 = svmlalt_f32(acc0, xv, yv);
    }

    /* Handle remaining 8-element chunks */
    uint64_t step8 = svcntw(); /* 8 for 256-bit */
    for (; i + step8 <= d; i += step8) {
        /* Load only bottom half worth of fp16 data */
        svbool_t pg_half = svwhilelt_b16((uint64_t)0, (uint64_t)step8);
        svfloat16_t xv = svld1_f16(pg_half, x + i);
        svfloat16_t yv = svld1_f16(pg_half, y + i);

        /* Match 16-element path: svmlalb + svmlalt cover all fp16 lanes. */
        acc0 = svmlalb_f32(acc0, xv, yv);
        acc0 = svmlalt_f32(acc0, xv, yv);
    }

    /* Horizontal reduction */
    acc0 = svadd_f32_x(pg32, acc0, acc1);
    acc2 = svadd_f32_x(pg32, acc2, acc3);
    acc0 = svadd_f32_x(pg32, acc0, acc2);

    float res = svaddv_f32(pg32, acc0);

    /* Scalar tail */
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
static void krl_inner_product_idx_batch2_f16f32_sve2(
    const float16_t *x, const float16_t *__restrict y0, const float16_t *__restrict y1, const size_t d, float *dis)
{
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();

    /* Two outputs: 4 fp32 accumulators each (y0 and y1) */
    svfloat32_t acc0_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y0 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y1 = svdup_n_f32(0.0f);

    uint64_t step16 = svcnth();
    size_t i = 0;

    /* Main loop: 4 × step16 fp16 per iteration */
    for (; i + 4 * step16 <= d; i += 4 * step16) {
        svfloat16_t xv0 = svld1_f16(pg16, x + i);
        svfloat16_t xv1 = svld1_f16(pg16, x + i + step16);
        svfloat16_t xv2 = svld1_f16(pg16, x + i + 2 * step16);
        svfloat16_t xv3 = svld1_f16(pg16, x + i + 3 * step16);

        svfloat16_t y0v0 = svld1_f16(pg16, y0 + i);
        svfloat16_t y0v1 = svld1_f16(pg16, y0 + i + step16);
        svfloat16_t y0v2 = svld1_f16(pg16, y0 + i + 2 * step16);
        svfloat16_t y0v3 = svld1_f16(pg16, y0 + i + 3 * step16);

        svfloat16_t y1v0 = svld1_f16(pg16, y1 + i);
        svfloat16_t y1v1 = svld1_f16(pg16, y1 + i + step16);
        svfloat16_t y1v2 = svld1_f16(pg16, y1 + i + 2 * step16);
        svfloat16_t y1v3 = svld1_f16(pg16, y1 + i + 3 * step16);

        /* svmlalb + svmlalt for y0 (x·y0) */
        acc0_y0 = svmlalb_f32(acc0_y0, xv0, y0v0);
        acc0_y0 = svmlalt_f32(acc0_y0, xv0, y0v0);
        acc1_y0 = svmlalb_f32(acc1_y0, xv1, y0v1);
        acc1_y0 = svmlalt_f32(acc1_y0, xv1, y0v1);
        acc2_y0 = svmlalb_f32(acc2_y0, xv2, y0v2);
        acc2_y0 = svmlalt_f32(acc2_y0, xv2, y0v2);
        acc3_y0 = svmlalb_f32(acc3_y0, xv3, y0v3);
        acc3_y0 = svmlalt_f32(acc3_y0, xv3, y0v3);

        /* svmlalb + svmlalt for y1 (x·y1) */
        acc0_y1 = svmlalb_f32(acc0_y1, xv0, y1v0);
        acc0_y1 = svmlalt_f32(acc0_y1, xv0, y1v0);
        acc1_y1 = svmlalb_f32(acc1_y1, xv1, y1v1);
        acc1_y1 = svmlalt_f32(acc1_y1, xv1, y1v1);
        acc2_y1 = svmlalb_f32(acc2_y1, xv2, y1v2);
        acc2_y1 = svmlalt_f32(acc2_y1, xv2, y1v2);
        acc3_y1 = svmlalb_f32(acc3_y1, xv3, y1v3);
        acc3_y1 = svmlalt_f32(acc3_y1, xv3, y1v3);
    }

    /* Remaining 16-element chunks */
    for (; i + step16 <= d; i += step16) {
        svfloat16_t xv = svld1_f16(pg16, x + i);
        svfloat16_t y0v = svld1_f16(pg16, y0 + i);
        svfloat16_t y1v = svld1_f16(pg16, y1 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
    }

    /* Remaining 8-element chunks (step8) */
    uint64_t step8 = svcntw();
    for (; i + step8 <= d; i += step8) {
        svbool_t pg_half = svwhilelt_b16((uint64_t)0, (uint64_t)step8);
        svfloat16_t xv = svld1_f16(pg_half, x + i);
        svfloat16_t y0v = svld1_f16(pg_half, y0 + i);
        svfloat16_t y1v = svld1_f16(pg_half, y1 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
    }

    /* Horizontal reduction for y0 */
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc1_y0);
    acc2_y0 = svadd_f32_x(pg32, acc2_y0, acc3_y0);
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc2_y0);
    float res_y0 = svaddv_f32(pg32, acc0_y0);

    /* Horizontal reduction for y1 */
    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc1_y1);
    acc2_y1 = svadd_f32_x(pg32, acc2_y1, acc3_y1);
    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc2_y1);
    float res_y1 = svaddv_f32(pg32, acc0_y1);

    /* Scalar tail */
    for (; i < d; i++) {
        res_y0 += (float)(x[i] * y0[i]);
        res_y1 += (float)(x[i] * y1[i]);
    }

    dis[0] = res_y0;
    dis[1] = res_y1;
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
static void krl_inner_product_idx_batch4_f16f32_sve2(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    const float16_t *y0 = y[0];
    const float16_t *y1 = y[1];
    const float16_t *y2 = y[2];
    const float16_t *y3 = y[3];

    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();

    /* Four outputs: 4 fp32 accumulators each (y0..y3) */
    svfloat32_t acc0_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y0 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y1 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y2 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y3 = svdup_n_f32(0.0f);

    uint64_t step16 = svcnth();
    size_t i = 0;

    /* Main loop: 4 × step16 fp16 per iteration */
    for (; i + 4 * step16 <= d; i += 4 * step16) {
        svfloat16_t xv0 = svld1_f16(pg16, x + i);
        svfloat16_t xv1 = svld1_f16(pg16, x + i + step16);
        svfloat16_t xv2 = svld1_f16(pg16, x + i + 2 * step16);
        svfloat16_t xv3 = svld1_f16(pg16, x + i + 3 * step16);

        svfloat16_t y0v0 = svld1_f16(pg16, y0 + i);
        svfloat16_t y0v1 = svld1_f16(pg16, y0 + i + step16);
        svfloat16_t y0v2 = svld1_f16(pg16, y0 + i + 2 * step16);
        svfloat16_t y0v3 = svld1_f16(pg16, y0 + i + 3 * step16);

        svfloat16_t y1v0 = svld1_f16(pg16, y1 + i);
        svfloat16_t y1v1 = svld1_f16(pg16, y1 + i + step16);
        svfloat16_t y1v2 = svld1_f16(pg16, y1 + i + 2 * step16);
        svfloat16_t y1v3 = svld1_f16(pg16, y1 + i + 3 * step16);

        svfloat16_t y2v0 = svld1_f16(pg16, y2 + i);
        svfloat16_t y2v1 = svld1_f16(pg16, y2 + i + step16);
        svfloat16_t y2v2 = svld1_f16(pg16, y2 + i + 2 * step16);
        svfloat16_t y2v3 = svld1_f16(pg16, y2 + i + 3 * step16);

        svfloat16_t y3v0 = svld1_f16(pg16, y3 + i);
        svfloat16_t y3v1 = svld1_f16(pg16, y3 + i + step16);
        svfloat16_t y3v2 = svld1_f16(pg16, y3 + i + 2 * step16);
        svfloat16_t y3v3 = svld1_f16(pg16, y3 + i + 3 * step16);

        /* svmlalb + svmlalt for y0 (x·y0) */
        acc0_y0 = svmlalb_f32(acc0_y0, xv0, y0v0);
        acc0_y0 = svmlalt_f32(acc0_y0, xv0, y0v0);
        acc1_y0 = svmlalb_f32(acc1_y0, xv1, y0v1);
        acc1_y0 = svmlalt_f32(acc1_y0, xv1, y0v1);
        acc2_y0 = svmlalb_f32(acc2_y0, xv2, y0v2);
        acc2_y0 = svmlalt_f32(acc2_y0, xv2, y0v2);
        acc3_y0 = svmlalb_f32(acc3_y0, xv3, y0v3);
        acc3_y0 = svmlalt_f32(acc3_y0, xv3, y0v3);

        /* svmlalb + svmlalt for y1 (x·y1) */
        acc0_y1 = svmlalb_f32(acc0_y1, xv0, y1v0);
        acc0_y1 = svmlalt_f32(acc0_y1, xv0, y1v0);
        acc1_y1 = svmlalb_f32(acc1_y1, xv1, y1v1);
        acc1_y1 = svmlalt_f32(acc1_y1, xv1, y1v1);
        acc2_y1 = svmlalb_f32(acc2_y1, xv2, y1v2);
        acc2_y1 = svmlalt_f32(acc2_y1, xv2, y1v2);
        acc3_y1 = svmlalb_f32(acc3_y1, xv3, y1v3);
        acc3_y1 = svmlalt_f32(acc3_y1, xv3, y1v3);

        /* svmlalb + svmlalt for y2 (x·y2) */
        acc0_y2 = svmlalb_f32(acc0_y2, xv0, y2v0);
        acc0_y2 = svmlalt_f32(acc0_y2, xv0, y2v0);
        acc1_y2 = svmlalb_f32(acc1_y2, xv1, y2v1);
        acc1_y2 = svmlalt_f32(acc1_y2, xv1, y2v1);
        acc2_y2 = svmlalb_f32(acc2_y2, xv2, y2v2);
        acc2_y2 = svmlalt_f32(acc2_y2, xv2, y2v2);
        acc3_y2 = svmlalb_f32(acc3_y2, xv3, y2v3);
        acc3_y2 = svmlalt_f32(acc3_y2, xv3, y2v3);

        /* svmlalb + svmlalt for y3 (x·y3) */
        acc0_y3 = svmlalb_f32(acc0_y3, xv0, y3v0);
        acc0_y3 = svmlalt_f32(acc0_y3, xv0, y3v0);
        acc1_y3 = svmlalb_f32(acc1_y3, xv1, y3v1);
        acc1_y3 = svmlalt_f32(acc1_y3, xv1, y3v1);
        acc2_y3 = svmlalb_f32(acc2_y3, xv2, y3v2);
        acc2_y3 = svmlalt_f32(acc2_y3, xv2, y3v2);
        acc3_y3 = svmlalb_f32(acc3_y3, xv3, y3v3);
        acc3_y3 = svmlalt_f32(acc3_y3, xv3, y3v3);
    }

    /* Remaining 16-element chunks */
    for (; i + step16 <= d; i += step16) {
        svfloat16_t xv = svld1_f16(pg16, x + i);
        svfloat16_t y0v = svld1_f16(pg16, y0 + i);
        svfloat16_t y1v = svld1_f16(pg16, y1 + i);
        svfloat16_t y2v = svld1_f16(pg16, y2 + i);
        svfloat16_t y3v = svld1_f16(pg16, y3 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
        acc0_y2 = svmlalb_f32(acc0_y2, xv, y2v);
        acc0_y2 = svmlalt_f32(acc0_y2, xv, y2v);
        acc0_y3 = svmlalb_f32(acc0_y3, xv, y3v);
        acc0_y3 = svmlalt_f32(acc0_y3, xv, y3v);
    }

    /* Remaining 8-element chunks (step8) */
    uint64_t step8 = svcntw();
    for (; i + step8 <= d; i += step8) {
        svbool_t pg_half = svwhilelt_b16((uint64_t)0, (uint64_t)step8);
        svfloat16_t xv = svld1_f16(pg_half, x + i);
        svfloat16_t y0v = svld1_f16(pg_half, y0 + i);
        svfloat16_t y1v = svld1_f16(pg_half, y1 + i);
        svfloat16_t y2v = svld1_f16(pg_half, y2 + i);
        svfloat16_t y3v = svld1_f16(pg_half, y3 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
        acc0_y2 = svmlalb_f32(acc0_y2, xv, y2v);
        acc0_y2 = svmlalt_f32(acc0_y2, xv, y2v);
        acc0_y3 = svmlalb_f32(acc0_y3, xv, y3v);
        acc0_y3 = svmlalt_f32(acc0_y3, xv, y3v);
    }

    /* Horizontal reduction for y0..y3 */
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc1_y0);
    acc2_y0 = svadd_f32_x(pg32, acc2_y0, acc3_y0);
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc2_y0);
    float res_y0 = svaddv_f32(pg32, acc0_y0);

    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc1_y1);
    acc2_y1 = svadd_f32_x(pg32, acc2_y1, acc3_y1);
    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc2_y1);
    float res_y1 = svaddv_f32(pg32, acc0_y1);

    acc0_y2 = svadd_f32_x(pg32, acc0_y2, acc1_y2);
    acc2_y2 = svadd_f32_x(pg32, acc2_y2, acc3_y2);
    acc0_y2 = svadd_f32_x(pg32, acc0_y2, acc2_y2);
    float res_y2 = svaddv_f32(pg32, acc0_y2);

    acc0_y3 = svadd_f32_x(pg32, acc0_y3, acc1_y3);
    acc2_y3 = svadd_f32_x(pg32, acc2_y3, acc3_y3);
    acc0_y3 = svadd_f32_x(pg32, acc0_y3, acc2_y3);
    float res_y3 = svaddv_f32(pg32, acc0_y3);

    /* Scalar tail */
    for (; i < d; i++) {
        res_y0 += (float)(x[i] * y0[i]);
        res_y1 += (float)(x[i] * y1[i]);
        res_y2 += (float)(x[i] * y2[i]);
        res_y3 += (float)(x[i] * y3[i]);
    }

    dis[0] = res_y0;
    dis[1] = res_y1;
    dis[2] = res_y2;
    dis[3] = res_y3;
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
static void krl_inner_product_idx_batch8_f16f32_sve2(
    const float16_t *x, const float16_t *__restrict *y, const size_t d, float *dis)
{
    const float16_t *y0 = y[0];
    const float16_t *y1 = y[1];
    const float16_t *y2 = y[2];
    const float16_t *y3 = y[3];
    const float16_t *y4 = y[4];
    const float16_t *y5 = y[5];
    const float16_t *y6 = y[6];
    const float16_t *y7 = y[7];

    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();

    /* Eight outputs: 4 fp32 accumulators each (y0..y7) */
    svfloat32_t acc0_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y0 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y1 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y2 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y3 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y4 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y4 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y4 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y4 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y5 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y5 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y5 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y5 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y6 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y6 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y6 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y6 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y7 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y7 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y7 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y7 = svdup_n_f32(0.0f);

    uint64_t step16 = svcnth();
    size_t i = 0;

    /* Main loop: 4 × step16 fp16 per iteration */
    for (; i + 4 * step16 <= d; i += 4 * step16) {
        svfloat16_t xv0 = svld1_f16(pg16, x + i);
        svfloat16_t xv1 = svld1_f16(pg16, x + i + step16);
        svfloat16_t xv2 = svld1_f16(pg16, x + i + 2 * step16);
        svfloat16_t xv3 = svld1_f16(pg16, x + i + 3 * step16);

        svfloat16_t y0v0 = svld1_f16(pg16, y0 + i);
        svfloat16_t y0v1 = svld1_f16(pg16, y0 + i + step16);
        svfloat16_t y0v2 = svld1_f16(pg16, y0 + i + 2 * step16);
        svfloat16_t y0v3 = svld1_f16(pg16, y0 + i + 3 * step16);

        svfloat16_t y1v0 = svld1_f16(pg16, y1 + i);
        svfloat16_t y1v1 = svld1_f16(pg16, y1 + i + step16);
        svfloat16_t y1v2 = svld1_f16(pg16, y1 + i + 2 * step16);
        svfloat16_t y1v3 = svld1_f16(pg16, y1 + i + 3 * step16);

        svfloat16_t y2v0 = svld1_f16(pg16, y2 + i);
        svfloat16_t y2v1 = svld1_f16(pg16, y2 + i + step16);
        svfloat16_t y2v2 = svld1_f16(pg16, y2 + i + 2 * step16);
        svfloat16_t y2v3 = svld1_f16(pg16, y2 + i + 3 * step16);

        svfloat16_t y3v0 = svld1_f16(pg16, y3 + i);
        svfloat16_t y3v1 = svld1_f16(pg16, y3 + i + step16);
        svfloat16_t y3v2 = svld1_f16(pg16, y3 + i + 2 * step16);
        svfloat16_t y3v3 = svld1_f16(pg16, y3 + i + 3 * step16);

        svfloat16_t y4v0 = svld1_f16(pg16, y4 + i);
        svfloat16_t y4v1 = svld1_f16(pg16, y4 + i + step16);
        svfloat16_t y4v2 = svld1_f16(pg16, y4 + i + 2 * step16);
        svfloat16_t y4v3 = svld1_f16(pg16, y4 + i + 3 * step16);

        svfloat16_t y5v0 = svld1_f16(pg16, y5 + i);
        svfloat16_t y5v1 = svld1_f16(pg16, y5 + i + step16);
        svfloat16_t y5v2 = svld1_f16(pg16, y5 + i + 2 * step16);
        svfloat16_t y5v3 = svld1_f16(pg16, y5 + i + 3 * step16);

        svfloat16_t y6v0 = svld1_f16(pg16, y6 + i);
        svfloat16_t y6v1 = svld1_f16(pg16, y6 + i + step16);
        svfloat16_t y6v2 = svld1_f16(pg16, y6 + i + 2 * step16);
        svfloat16_t y6v3 = svld1_f16(pg16, y6 + i + 3 * step16);

        svfloat16_t y7v0 = svld1_f16(pg16, y7 + i);
        svfloat16_t y7v1 = svld1_f16(pg16, y7 + i + step16);
        svfloat16_t y7v2 = svld1_f16(pg16, y7 + i + 2 * step16);
        svfloat16_t y7v3 = svld1_f16(pg16, y7 + i + 3 * step16);

        /* svmlalb + svmlalt for y0..y7 */
        acc0_y0 = svmlalb_f32(acc0_y0, xv0, y0v0);
        acc0_y0 = svmlalt_f32(acc0_y0, xv0, y0v0);
        acc1_y0 = svmlalb_f32(acc1_y0, xv1, y0v1);
        acc1_y0 = svmlalt_f32(acc1_y0, xv1, y0v1);
        acc2_y0 = svmlalb_f32(acc2_y0, xv2, y0v2);
        acc2_y0 = svmlalt_f32(acc2_y0, xv2, y0v2);
        acc3_y0 = svmlalb_f32(acc3_y0, xv3, y0v3);
        acc3_y0 = svmlalt_f32(acc3_y0, xv3, y0v3);

        acc0_y1 = svmlalb_f32(acc0_y1, xv0, y1v0);
        acc0_y1 = svmlalt_f32(acc0_y1, xv0, y1v0);
        acc1_y1 = svmlalb_f32(acc1_y1, xv1, y1v1);
        acc1_y1 = svmlalt_f32(acc1_y1, xv1, y1v1);
        acc2_y1 = svmlalb_f32(acc2_y1, xv2, y1v2);
        acc2_y1 = svmlalt_f32(acc2_y1, xv2, y1v2);
        acc3_y1 = svmlalb_f32(acc3_y1, xv3, y1v3);
        acc3_y1 = svmlalt_f32(acc3_y1, xv3, y1v3);

        acc0_y2 = svmlalb_f32(acc0_y2, xv0, y2v0);
        acc0_y2 = svmlalt_f32(acc0_y2, xv0, y2v0);
        acc1_y2 = svmlalb_f32(acc1_y2, xv1, y2v1);
        acc1_y2 = svmlalt_f32(acc1_y2, xv1, y2v1);
        acc2_y2 = svmlalb_f32(acc2_y2, xv2, y2v2);
        acc2_y2 = svmlalt_f32(acc2_y2, xv2, y2v2);
        acc3_y2 = svmlalb_f32(acc3_y2, xv3, y2v3);
        acc3_y2 = svmlalt_f32(acc3_y2, xv3, y2v3);

        acc0_y3 = svmlalb_f32(acc0_y3, xv0, y3v0);
        acc0_y3 = svmlalt_f32(acc0_y3, xv0, y3v0);
        acc1_y3 = svmlalb_f32(acc1_y3, xv1, y3v1);
        acc1_y3 = svmlalt_f32(acc1_y3, xv1, y3v1);
        acc2_y3 = svmlalb_f32(acc2_y3, xv2, y3v2);
        acc2_y3 = svmlalt_f32(acc2_y3, xv2, y3v2);
        acc3_y3 = svmlalb_f32(acc3_y3, xv3, y3v3);
        acc3_y3 = svmlalt_f32(acc3_y3, xv3, y3v3);

        acc0_y4 = svmlalb_f32(acc0_y4, xv0, y4v0);
        acc0_y4 = svmlalt_f32(acc0_y4, xv0, y4v0);
        acc1_y4 = svmlalb_f32(acc1_y4, xv1, y4v1);
        acc1_y4 = svmlalt_f32(acc1_y4, xv1, y4v1);
        acc2_y4 = svmlalb_f32(acc2_y4, xv2, y4v2);
        acc2_y4 = svmlalt_f32(acc2_y4, xv2, y4v2);
        acc3_y4 = svmlalb_f32(acc3_y4, xv3, y4v3);
        acc3_y4 = svmlalt_f32(acc3_y4, xv3, y4v3);

        acc0_y5 = svmlalb_f32(acc0_y5, xv0, y5v0);
        acc0_y5 = svmlalt_f32(acc0_y5, xv0, y5v0);
        acc1_y5 = svmlalb_f32(acc1_y5, xv1, y5v1);
        acc1_y5 = svmlalt_f32(acc1_y5, xv1, y5v1);
        acc2_y5 = svmlalb_f32(acc2_y5, xv2, y5v2);
        acc2_y5 = svmlalt_f32(acc2_y5, xv2, y5v2);
        acc3_y5 = svmlalb_f32(acc3_y5, xv3, y5v3);
        acc3_y5 = svmlalt_f32(acc3_y5, xv3, y5v3);

        acc0_y6 = svmlalb_f32(acc0_y6, xv0, y6v0);
        acc0_y6 = svmlalt_f32(acc0_y6, xv0, y6v0);
        acc1_y6 = svmlalb_f32(acc1_y6, xv1, y6v1);
        acc1_y6 = svmlalt_f32(acc1_y6, xv1, y6v1);
        acc2_y6 = svmlalb_f32(acc2_y6, xv2, y6v2);
        acc2_y6 = svmlalt_f32(acc2_y6, xv2, y6v2);
        acc3_y6 = svmlalb_f32(acc3_y6, xv3, y6v3);
        acc3_y6 = svmlalt_f32(acc3_y6, xv3, y6v3);

        acc0_y7 = svmlalb_f32(acc0_y7, xv0, y7v0);
        acc0_y7 = svmlalt_f32(acc0_y7, xv0, y7v0);
        acc1_y7 = svmlalb_f32(acc1_y7, xv1, y7v1);
        acc1_y7 = svmlalt_f32(acc1_y7, xv1, y7v1);
        acc2_y7 = svmlalb_f32(acc2_y7, xv2, y7v2);
        acc2_y7 = svmlalt_f32(acc2_y7, xv2, y7v2);
        acc3_y7 = svmlalb_f32(acc3_y7, xv3, y7v3);
        acc3_y7 = svmlalt_f32(acc3_y7, xv3, y7v3);
    }

    /* Remaining 16-element chunks */
    for (; i + step16 <= d; i += step16) {
        svfloat16_t xv = svld1_f16(pg16, x + i);
        svfloat16_t y0v = svld1_f16(pg16, y0 + i);
        svfloat16_t y1v = svld1_f16(pg16, y1 + i);
        svfloat16_t y2v = svld1_f16(pg16, y2 + i);
        svfloat16_t y3v = svld1_f16(pg16, y3 + i);
        svfloat16_t y4v = svld1_f16(pg16, y4 + i);
        svfloat16_t y5v = svld1_f16(pg16, y5 + i);
        svfloat16_t y6v = svld1_f16(pg16, y6 + i);
        svfloat16_t y7v = svld1_f16(pg16, y7 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
        acc0_y2 = svmlalb_f32(acc0_y2, xv, y2v);
        acc0_y2 = svmlalt_f32(acc0_y2, xv, y2v);
        acc0_y3 = svmlalb_f32(acc0_y3, xv, y3v);
        acc0_y3 = svmlalt_f32(acc0_y3, xv, y3v);
        acc0_y4 = svmlalb_f32(acc0_y4, xv, y4v);
        acc0_y4 = svmlalt_f32(acc0_y4, xv, y4v);
        acc0_y5 = svmlalb_f32(acc0_y5, xv, y5v);
        acc0_y5 = svmlalt_f32(acc0_y5, xv, y5v);
        acc0_y6 = svmlalb_f32(acc0_y6, xv, y6v);
        acc0_y6 = svmlalt_f32(acc0_y6, xv, y6v);
        acc0_y7 = svmlalb_f32(acc0_y7, xv, y7v);
        acc0_y7 = svmlalt_f32(acc0_y7, xv, y7v);
    }

    /* Remaining 8-element chunks (step8) */
    uint64_t step8 = svcntw();
    for (; i + step8 <= d; i += step8) {
        svbool_t pg_half = svwhilelt_b16((uint64_t)0, (uint64_t)step8);
        svfloat16_t xv = svld1_f16(pg_half, x + i);
        svfloat16_t y0v = svld1_f16(pg_half, y0 + i);
        svfloat16_t y1v = svld1_f16(pg_half, y1 + i);
        svfloat16_t y2v = svld1_f16(pg_half, y2 + i);
        svfloat16_t y3v = svld1_f16(pg_half, y3 + i);
        svfloat16_t y4v = svld1_f16(pg_half, y4 + i);
        svfloat16_t y5v = svld1_f16(pg_half, y5 + i);
        svfloat16_t y6v = svld1_f16(pg_half, y6 + i);
        svfloat16_t y7v = svld1_f16(pg_half, y7 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
        acc0_y2 = svmlalb_f32(acc0_y2, xv, y2v);
        acc0_y2 = svmlalt_f32(acc0_y2, xv, y2v);
        acc0_y3 = svmlalb_f32(acc0_y3, xv, y3v);
        acc0_y3 = svmlalt_f32(acc0_y3, xv, y3v);
        acc0_y4 = svmlalb_f32(acc0_y4, xv, y4v);
        acc0_y4 = svmlalt_f32(acc0_y4, xv, y4v);
        acc0_y5 = svmlalb_f32(acc0_y5, xv, y5v);
        acc0_y5 = svmlalt_f32(acc0_y5, xv, y5v);
        acc0_y6 = svmlalb_f32(acc0_y6, xv, y6v);
        acc0_y6 = svmlalt_f32(acc0_y6, xv, y6v);
        acc0_y7 = svmlalb_f32(acc0_y7, xv, y7v);
        acc0_y7 = svmlalt_f32(acc0_y7, xv, y7v);
    }

    /* Horizontal reduction for y0..y7 */
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc1_y0);
    acc2_y0 = svadd_f32_x(pg32, acc2_y0, acc3_y0);
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc2_y0);
    float res_y0 = svaddv_f32(pg32, acc0_y0);

    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc1_y1);
    acc2_y1 = svadd_f32_x(pg32, acc2_y1, acc3_y1);
    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc2_y1);
    float res_y1 = svaddv_f32(pg32, acc0_y1);

    acc0_y2 = svadd_f32_x(pg32, acc0_y2, acc1_y2);
    acc2_y2 = svadd_f32_x(pg32, acc2_y2, acc3_y2);
    acc0_y2 = svadd_f32_x(pg32, acc0_y2, acc2_y2);
    float res_y2 = svaddv_f32(pg32, acc0_y2);

    acc0_y3 = svadd_f32_x(pg32, acc0_y3, acc1_y3);
    acc2_y3 = svadd_f32_x(pg32, acc2_y3, acc3_y3);
    acc0_y3 = svadd_f32_x(pg32, acc0_y3, acc2_y3);
    float res_y3 = svaddv_f32(pg32, acc0_y3);

    acc0_y4 = svadd_f32_x(pg32, acc0_y4, acc1_y4);
    acc2_y4 = svadd_f32_x(pg32, acc2_y4, acc3_y4);
    acc0_y4 = svadd_f32_x(pg32, acc0_y4, acc2_y4);
    float res_y4 = svaddv_f32(pg32, acc0_y4);

    acc0_y5 = svadd_f32_x(pg32, acc0_y5, acc1_y5);
    acc2_y5 = svadd_f32_x(pg32, acc2_y5, acc3_y5);
    acc0_y5 = svadd_f32_x(pg32, acc0_y5, acc2_y5);
    float res_y5 = svaddv_f32(pg32, acc0_y5);

    acc0_y6 = svadd_f32_x(pg32, acc0_y6, acc1_y6);
    acc2_y6 = svadd_f32_x(pg32, acc2_y6, acc3_y6);
    acc0_y6 = svadd_f32_x(pg32, acc0_y6, acc2_y6);
    float res_y6 = svaddv_f32(pg32, acc0_y6);

    acc0_y7 = svadd_f32_x(pg32, acc0_y7, acc1_y7);
    acc2_y7 = svadd_f32_x(pg32, acc2_y7, acc3_y7);
    acc0_y7 = svadd_f32_x(pg32, acc0_y7, acc2_y7);
    float res_y7 = svaddv_f32(pg32, acc0_y7);

    /* Scalar tail */
    for (; i < d; i++) {
        res_y0 += (float)(x[i] * y0[i]);
        res_y1 += (float)(x[i] * y1[i]);
        res_y2 += (float)(x[i] * y2[i]);
        res_y3 += (float)(x[i] * y3[i]);
        res_y4 += (float)(x[i] * y4[i]);
        res_y5 += (float)(x[i] * y5[i]);
        res_y6 += (float)(x[i] * y6[i]);
        res_y7 += (float)(x[i] * y7[i]);
    }

    dis[0] = res_y0;
    dis[1] = res_y1;
    dis[2] = res_y2;
    dis[3] = res_y3;
    dis[4] = res_y4;
    dis[5] = res_y5;
    dis[6] = res_y6;
    dis[7] = res_y7;
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
static void krl_inner_product_batch2_f16f32_sve2(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    const float16_t *y0 = y;
    const float16_t *y1 = y + d;

    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();

    svfloat32_t acc0_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y0 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y1 = svdup_n_f32(0.0f);

    uint64_t step16 = svcnth();
    size_t i = 0;

    for (; i + 4 * step16 <= d; i += 4 * step16) {
        svfloat16_t xv0 = svld1_f16(pg16, x + i);
        svfloat16_t xv1 = svld1_f16(pg16, x + i + step16);
        svfloat16_t xv2 = svld1_f16(pg16, x + i + 2 * step16);
        svfloat16_t xv3 = svld1_f16(pg16, x + i + 3 * step16);

        svfloat16_t y0v0 = svld1_f16(pg16, y0 + i);
        svfloat16_t y0v1 = svld1_f16(pg16, y0 + i + step16);
        svfloat16_t y0v2 = svld1_f16(pg16, y0 + i + 2 * step16);
        svfloat16_t y0v3 = svld1_f16(pg16, y0 + i + 3 * step16);

        svfloat16_t y1v0 = svld1_f16(pg16, y1 + i);
        svfloat16_t y1v1 = svld1_f16(pg16, y1 + i + step16);
        svfloat16_t y1v2 = svld1_f16(pg16, y1 + i + 2 * step16);
        svfloat16_t y1v3 = svld1_f16(pg16, y1 + i + 3 * step16);

        acc0_y0 = svmlalb_f32(acc0_y0, xv0, y0v0);
        acc0_y0 = svmlalt_f32(acc0_y0, xv0, y0v0);
        acc1_y0 = svmlalb_f32(acc1_y0, xv1, y0v1);
        acc1_y0 = svmlalt_f32(acc1_y0, xv1, y0v1);
        acc2_y0 = svmlalb_f32(acc2_y0, xv2, y0v2);
        acc2_y0 = svmlalt_f32(acc2_y0, xv2, y0v2);
        acc3_y0 = svmlalb_f32(acc3_y0, xv3, y0v3);
        acc3_y0 = svmlalt_f32(acc3_y0, xv3, y0v3);

        acc0_y1 = svmlalb_f32(acc0_y1, xv0, y1v0);
        acc0_y1 = svmlalt_f32(acc0_y1, xv0, y1v0);
        acc1_y1 = svmlalb_f32(acc1_y1, xv1, y1v1);
        acc1_y1 = svmlalt_f32(acc1_y1, xv1, y1v1);
        acc2_y1 = svmlalb_f32(acc2_y1, xv2, y1v2);
        acc2_y1 = svmlalt_f32(acc2_y1, xv2, y1v2);
        acc3_y1 = svmlalb_f32(acc3_y1, xv3, y1v3);
        acc3_y1 = svmlalt_f32(acc3_y1, xv3, y1v3);
    }

    for (; i + step16 <= d; i += step16) {
        svfloat16_t xv = svld1_f16(pg16, x + i);
        svfloat16_t y0v = svld1_f16(pg16, y0 + i);
        svfloat16_t y1v = svld1_f16(pg16, y1 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
    }

    uint64_t step8 = svcntw();
    for (; i + step8 <= d; i += step8) {
        svbool_t pg_half = svwhilelt_b16((uint64_t)0, (uint64_t)step8);
        svfloat16_t xv = svld1_f16(pg_half, x + i);
        svfloat16_t y0v = svld1_f16(pg_half, y0 + i);
        svfloat16_t y1v = svld1_f16(pg_half, y1 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
    }

    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc1_y0);
    acc2_y0 = svadd_f32_x(pg32, acc2_y0, acc3_y0);
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc2_y0);
    float res_y0 = svaddv_f32(pg32, acc0_y0);

    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc1_y1);
    acc2_y1 = svadd_f32_x(pg32, acc2_y1, acc3_y1);
    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc2_y1);
    float res_y1 = svaddv_f32(pg32, acc0_y1);

    for (; i < d; i++) {
        res_y0 += (float)(x[i] * y0[i]);
        res_y1 += (float)(x[i] * y1[i]);
    }

    dis[0] = res_y0;
    dis[1] = res_y1;
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
static void krl_inner_product_batch4_f16f32_sve2(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    const float16_t *y0 = y;
    const float16_t *y1 = y + d;
    const float16_t *y2 = y + 2 * d;
    const float16_t *y3 = y + 3 * d;

    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();

    svfloat32_t acc0_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y0 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y1 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y2 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y3 = svdup_n_f32(0.0f);

    uint64_t step16 = svcnth();
    size_t i = 0;

    for (; i + 4 * step16 <= d; i += 4 * step16) {
        svfloat16_t xv0 = svld1_f16(pg16, x + i);
        svfloat16_t xv1 = svld1_f16(pg16, x + i + step16);
        svfloat16_t xv2 = svld1_f16(pg16, x + i + 2 * step16);
        svfloat16_t xv3 = svld1_f16(pg16, x + i + 3 * step16);

        svfloat16_t y0v0 = svld1_f16(pg16, y0 + i);
        svfloat16_t y0v1 = svld1_f16(pg16, y0 + i + step16);
        svfloat16_t y0v2 = svld1_f16(pg16, y0 + i + 2 * step16);
        svfloat16_t y0v3 = svld1_f16(pg16, y0 + i + 3 * step16);

        svfloat16_t y1v0 = svld1_f16(pg16, y1 + i);
        svfloat16_t y1v1 = svld1_f16(pg16, y1 + i + step16);
        svfloat16_t y1v2 = svld1_f16(pg16, y1 + i + 2 * step16);
        svfloat16_t y1v3 = svld1_f16(pg16, y1 + i + 3 * step16);

        svfloat16_t y2v0 = svld1_f16(pg16, y2 + i);
        svfloat16_t y2v1 = svld1_f16(pg16, y2 + i + step16);
        svfloat16_t y2v2 = svld1_f16(pg16, y2 + i + 2 * step16);
        svfloat16_t y2v3 = svld1_f16(pg16, y2 + i + 3 * step16);

        svfloat16_t y3v0 = svld1_f16(pg16, y3 + i);
        svfloat16_t y3v1 = svld1_f16(pg16, y3 + i + step16);
        svfloat16_t y3v2 = svld1_f16(pg16, y3 + i + 2 * step16);
        svfloat16_t y3v3 = svld1_f16(pg16, y3 + i + 3 * step16);

        acc0_y0 = svmlalb_f32(acc0_y0, xv0, y0v0);
        acc0_y0 = svmlalt_f32(acc0_y0, xv0, y0v0);
        acc1_y0 = svmlalb_f32(acc1_y0, xv1, y0v1);
        acc1_y0 = svmlalt_f32(acc1_y0, xv1, y0v1);
        acc2_y0 = svmlalb_f32(acc2_y0, xv2, y0v2);
        acc2_y0 = svmlalt_f32(acc2_y0, xv2, y0v2);
        acc3_y0 = svmlalb_f32(acc3_y0, xv3, y0v3);
        acc3_y0 = svmlalt_f32(acc3_y0, xv3, y0v3);

        acc0_y1 = svmlalb_f32(acc0_y1, xv0, y1v0);
        acc0_y1 = svmlalt_f32(acc0_y1, xv0, y1v0);
        acc1_y1 = svmlalb_f32(acc1_y1, xv1, y1v1);
        acc1_y1 = svmlalt_f32(acc1_y1, xv1, y1v1);
        acc2_y1 = svmlalb_f32(acc2_y1, xv2, y1v2);
        acc2_y1 = svmlalt_f32(acc2_y1, xv2, y1v2);
        acc3_y1 = svmlalb_f32(acc3_y1, xv3, y1v3);
        acc3_y1 = svmlalt_f32(acc3_y1, xv3, y1v3);

        acc0_y2 = svmlalb_f32(acc0_y2, xv0, y2v0);
        acc0_y2 = svmlalt_f32(acc0_y2, xv0, y2v0);
        acc1_y2 = svmlalb_f32(acc1_y2, xv1, y2v1);
        acc1_y2 = svmlalt_f32(acc1_y2, xv1, y2v1);
        acc2_y2 = svmlalb_f32(acc2_y2, xv2, y2v2);
        acc2_y2 = svmlalt_f32(acc2_y2, xv2, y2v2);
        acc3_y2 = svmlalb_f32(acc3_y2, xv3, y2v3);
        acc3_y2 = svmlalt_f32(acc3_y2, xv3, y2v3);

        acc0_y3 = svmlalb_f32(acc0_y3, xv0, y3v0);
        acc0_y3 = svmlalt_f32(acc0_y3, xv0, y3v0);
        acc1_y3 = svmlalb_f32(acc1_y3, xv1, y3v1);
        acc1_y3 = svmlalt_f32(acc1_y3, xv1, y3v1);
        acc2_y3 = svmlalb_f32(acc2_y3, xv2, y3v2);
        acc2_y3 = svmlalt_f32(acc2_y3, xv2, y3v2);
        acc3_y3 = svmlalb_f32(acc3_y3, xv3, y3v3);
        acc3_y3 = svmlalt_f32(acc3_y3, xv3, y3v3);
    }

    for (; i + step16 <= d; i += step16) {
        svfloat16_t xv = svld1_f16(pg16, x + i);
        svfloat16_t y0v = svld1_f16(pg16, y0 + i);
        svfloat16_t y1v = svld1_f16(pg16, y1 + i);
        svfloat16_t y2v = svld1_f16(pg16, y2 + i);
        svfloat16_t y3v = svld1_f16(pg16, y3 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
        acc0_y2 = svmlalb_f32(acc0_y2, xv, y2v);
        acc0_y2 = svmlalt_f32(acc0_y2, xv, y2v);
        acc0_y3 = svmlalb_f32(acc0_y3, xv, y3v);
        acc0_y3 = svmlalt_f32(acc0_y3, xv, y3v);
    }

    uint64_t step8 = svcntw();
    for (; i + step8 <= d; i += step8) {
        svbool_t pg_half = svwhilelt_b16((uint64_t)0, (uint64_t)step8);
        svfloat16_t xv = svld1_f16(pg_half, x + i);
        svfloat16_t y0v = svld1_f16(pg_half, y0 + i);
        svfloat16_t y1v = svld1_f16(pg_half, y1 + i);
        svfloat16_t y2v = svld1_f16(pg_half, y2 + i);
        svfloat16_t y3v = svld1_f16(pg_half, y3 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
        acc0_y2 = svmlalb_f32(acc0_y2, xv, y2v);
        acc0_y2 = svmlalt_f32(acc0_y2, xv, y2v);
        acc0_y3 = svmlalb_f32(acc0_y3, xv, y3v);
        acc0_y3 = svmlalt_f32(acc0_y3, xv, y3v);
    }

    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc1_y0);
    acc2_y0 = svadd_f32_x(pg32, acc2_y0, acc3_y0);
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc2_y0);
    float res_y0 = svaddv_f32(pg32, acc0_y0);

    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc1_y1);
    acc2_y1 = svadd_f32_x(pg32, acc2_y1, acc3_y1);
    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc2_y1);
    float res_y1 = svaddv_f32(pg32, acc0_y1);

    acc0_y2 = svadd_f32_x(pg32, acc0_y2, acc1_y2);
    acc2_y2 = svadd_f32_x(pg32, acc2_y2, acc3_y2);
    acc0_y2 = svadd_f32_x(pg32, acc0_y2, acc2_y2);
    float res_y2 = svaddv_f32(pg32, acc0_y2);

    acc0_y3 = svadd_f32_x(pg32, acc0_y3, acc1_y3);
    acc2_y3 = svadd_f32_x(pg32, acc2_y3, acc3_y3);
    acc0_y3 = svadd_f32_x(pg32, acc0_y3, acc2_y3);
    float res_y3 = svaddv_f32(pg32, acc0_y3);

    for (; i < d; i++) {
        res_y0 += (float)(x[i] * y0[i]);
        res_y1 += (float)(x[i] * y1[i]);
        res_y2 += (float)(x[i] * y2[i]);
        res_y3 += (float)(x[i] * y3[i]);
    }

    dis[0] = res_y0;
    dis[1] = res_y1;
    dis[2] = res_y2;
    dis[3] = res_y3;
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
void krl_inner_product_batch8_f16f32_sve2(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    const float16_t *y0 = y;
    const float16_t *y1 = y + d;
    const float16_t *y2 = y + 2 * d;
    const float16_t *y3 = y + 3 * d;
    const float16_t *y4 = y + 4 * d;
    const float16_t *y5 = y + 5 * d;
    const float16_t *y6 = y + 6 * d;
    const float16_t *y7 = y + 7 * d;

    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();

    svfloat32_t acc0_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y0 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y0 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y1 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y1 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y2 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y2 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y3 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y3 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y4 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y4 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y4 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y4 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y5 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y5 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y5 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y5 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y6 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y6 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y6 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y6 = svdup_n_f32(0.0f);

    svfloat32_t acc0_y7 = svdup_n_f32(0.0f);
    svfloat32_t acc1_y7 = svdup_n_f32(0.0f);
    svfloat32_t acc2_y7 = svdup_n_f32(0.0f);
    svfloat32_t acc3_y7 = svdup_n_f32(0.0f);

    uint64_t step16 = svcnth();
    size_t i = 0;

    for (; i + 4 * step16 <= d; i += 4 * step16) {
        svfloat16_t xv0 = svld1_f16(pg16, x + i);
        svfloat16_t xv1 = svld1_f16(pg16, x + i + step16);
        svfloat16_t xv2 = svld1_f16(pg16, x + i + 2 * step16);
        svfloat16_t xv3 = svld1_f16(pg16, x + i + 3 * step16);

        svfloat16_t y0v0 = svld1_f16(pg16, y0 + i);
        svfloat16_t y0v1 = svld1_f16(pg16, y0 + i + step16);
        svfloat16_t y0v2 = svld1_f16(pg16, y0 + i + 2 * step16);
        svfloat16_t y0v3 = svld1_f16(pg16, y0 + i + 3 * step16);

        svfloat16_t y1v0 = svld1_f16(pg16, y1 + i);
        svfloat16_t y1v1 = svld1_f16(pg16, y1 + i + step16);
        svfloat16_t y1v2 = svld1_f16(pg16, y1 + i + 2 * step16);
        svfloat16_t y1v3 = svld1_f16(pg16, y1 + i + 3 * step16);

        svfloat16_t y2v0 = svld1_f16(pg16, y2 + i);
        svfloat16_t y2v1 = svld1_f16(pg16, y2 + i + step16);
        svfloat16_t y2v2 = svld1_f16(pg16, y2 + i + 2 * step16);
        svfloat16_t y2v3 = svld1_f16(pg16, y2 + i + 3 * step16);

        svfloat16_t y3v0 = svld1_f16(pg16, y3 + i);
        svfloat16_t y3v1 = svld1_f16(pg16, y3 + i + step16);
        svfloat16_t y3v2 = svld1_f16(pg16, y3 + i + 2 * step16);
        svfloat16_t y3v3 = svld1_f16(pg16, y3 + i + 3 * step16);

        svfloat16_t y4v0 = svld1_f16(pg16, y4 + i);
        svfloat16_t y4v1 = svld1_f16(pg16, y4 + i + step16);
        svfloat16_t y4v2 = svld1_f16(pg16, y4 + i + 2 * step16);
        svfloat16_t y4v3 = svld1_f16(pg16, y4 + i + 3 * step16);

        svfloat16_t y5v0 = svld1_f16(pg16, y5 + i);
        svfloat16_t y5v1 = svld1_f16(pg16, y5 + i + step16);
        svfloat16_t y5v2 = svld1_f16(pg16, y5 + i + 2 * step16);
        svfloat16_t y5v3 = svld1_f16(pg16, y5 + i + 3 * step16);

        svfloat16_t y6v0 = svld1_f16(pg16, y6 + i);
        svfloat16_t y6v1 = svld1_f16(pg16, y6 + i + step16);
        svfloat16_t y6v2 = svld1_f16(pg16, y6 + i + 2 * step16);
        svfloat16_t y6v3 = svld1_f16(pg16, y6 + i + 3 * step16);

        svfloat16_t y7v0 = svld1_f16(pg16, y7 + i);
        svfloat16_t y7v1 = svld1_f16(pg16, y7 + i + step16);
        svfloat16_t y7v2 = svld1_f16(pg16, y7 + i + 2 * step16);
        svfloat16_t y7v3 = svld1_f16(pg16, y7 + i + 3 * step16);

        acc0_y0 = svmlalb_f32(acc0_y0, xv0, y0v0);
        acc0_y0 = svmlalt_f32(acc0_y0, xv0, y0v0);
        acc1_y0 = svmlalb_f32(acc1_y0, xv1, y0v1);
        acc1_y0 = svmlalt_f32(acc1_y0, xv1, y0v1);
        acc2_y0 = svmlalb_f32(acc2_y0, xv2, y0v2);
        acc2_y0 = svmlalt_f32(acc2_y0, xv2, y0v2);
        acc3_y0 = svmlalb_f32(acc3_y0, xv3, y0v3);
        acc3_y0 = svmlalt_f32(acc3_y0, xv3, y0v3);

        acc0_y1 = svmlalb_f32(acc0_y1, xv0, y1v0);
        acc0_y1 = svmlalt_f32(acc0_y1, xv0, y1v0);
        acc1_y1 = svmlalb_f32(acc1_y1, xv1, y1v1);
        acc1_y1 = svmlalt_f32(acc1_y1, xv1, y1v1);
        acc2_y1 = svmlalb_f32(acc2_y1, xv2, y1v2);
        acc2_y1 = svmlalt_f32(acc2_y1, xv2, y1v2);
        acc3_y1 = svmlalb_f32(acc3_y1, xv3, y1v3);
        acc3_y1 = svmlalt_f32(acc3_y1, xv3, y1v3);

        acc0_y2 = svmlalb_f32(acc0_y2, xv0, y2v0);
        acc0_y2 = svmlalt_f32(acc0_y2, xv0, y2v0);
        acc1_y2 = svmlalb_f32(acc1_y2, xv1, y2v1);
        acc1_y2 = svmlalt_f32(acc1_y2, xv1, y2v1);
        acc2_y2 = svmlalb_f32(acc2_y2, xv2, y2v2);
        acc2_y2 = svmlalt_f32(acc2_y2, xv2, y2v2);
        acc3_y2 = svmlalb_f32(acc3_y2, xv3, y2v3);
        acc3_y2 = svmlalt_f32(acc3_y2, xv3, y2v3);

        acc0_y3 = svmlalb_f32(acc0_y3, xv0, y3v0);
        acc0_y3 = svmlalt_f32(acc0_y3, xv0, y3v0);
        acc1_y3 = svmlalb_f32(acc1_y3, xv1, y3v1);
        acc1_y3 = svmlalt_f32(acc1_y3, xv1, y3v1);
        acc2_y3 = svmlalb_f32(acc2_y3, xv2, y3v2);
        acc2_y3 = svmlalt_f32(acc2_y3, xv2, y3v2);
        acc3_y3 = svmlalb_f32(acc3_y3, xv3, y3v3);
        acc3_y3 = svmlalt_f32(acc3_y3, xv3, y3v3);

        acc0_y4 = svmlalb_f32(acc0_y4, xv0, y4v0);
        acc0_y4 = svmlalt_f32(acc0_y4, xv0, y4v0);
        acc1_y4 = svmlalb_f32(acc1_y4, xv1, y4v1);
        acc1_y4 = svmlalt_f32(acc1_y4, xv1, y4v1);
        acc2_y4 = svmlalb_f32(acc2_y4, xv2, y4v2);
        acc2_y4 = svmlalt_f32(acc2_y4, xv2, y4v2);
        acc3_y4 = svmlalb_f32(acc3_y4, xv3, y4v3);
        acc3_y4 = svmlalt_f32(acc3_y4, xv3, y4v3);

        acc0_y5 = svmlalb_f32(acc0_y5, xv0, y5v0);
        acc0_y5 = svmlalt_f32(acc0_y5, xv0, y5v0);
        acc1_y5 = svmlalb_f32(acc1_y5, xv1, y5v1);
        acc1_y5 = svmlalt_f32(acc1_y5, xv1, y5v1);
        acc2_y5 = svmlalb_f32(acc2_y5, xv2, y5v2);
        acc2_y5 = svmlalt_f32(acc2_y5, xv2, y5v2);
        acc3_y5 = svmlalb_f32(acc3_y5, xv3, y5v3);
        acc3_y5 = svmlalt_f32(acc3_y5, xv3, y5v3);

        acc0_y6 = svmlalb_f32(acc0_y6, xv0, y6v0);
        acc0_y6 = svmlalt_f32(acc0_y6, xv0, y6v0);
        acc1_y6 = svmlalb_f32(acc1_y6, xv1, y6v1);
        acc1_y6 = svmlalt_f32(acc1_y6, xv1, y6v1);
        acc2_y6 = svmlalb_f32(acc2_y6, xv2, y6v2);
        acc2_y6 = svmlalt_f32(acc2_y6, xv2, y6v2);
        acc3_y6 = svmlalb_f32(acc3_y6, xv3, y6v3);
        acc3_y6 = svmlalt_f32(acc3_y6, xv3, y6v3);

        acc0_y7 = svmlalb_f32(acc0_y7, xv0, y7v0);
        acc0_y7 = svmlalt_f32(acc0_y7, xv0, y7v0);
        acc1_y7 = svmlalb_f32(acc1_y7, xv1, y7v1);
        acc1_y7 = svmlalt_f32(acc1_y7, xv1, y7v1);
        acc2_y7 = svmlalb_f32(acc2_y7, xv2, y7v2);
        acc2_y7 = svmlalt_f32(acc2_y7, xv2, y7v2);
        acc3_y7 = svmlalb_f32(acc3_y7, xv3, y7v3);
        acc3_y7 = svmlalt_f32(acc3_y7, xv3, y7v3);
    }

    for (; i + step16 <= d; i += step16) {
        svfloat16_t xv = svld1_f16(pg16, x + i);
        svfloat16_t y0v = svld1_f16(pg16, y0 + i);
        svfloat16_t y1v = svld1_f16(pg16, y1 + i);
        svfloat16_t y2v = svld1_f16(pg16, y2 + i);
        svfloat16_t y3v = svld1_f16(pg16, y3 + i);
        svfloat16_t y4v = svld1_f16(pg16, y4 + i);
        svfloat16_t y5v = svld1_f16(pg16, y5 + i);
        svfloat16_t y6v = svld1_f16(pg16, y6 + i);
        svfloat16_t y7v = svld1_f16(pg16, y7 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
        acc0_y2 = svmlalb_f32(acc0_y2, xv, y2v);
        acc0_y2 = svmlalt_f32(acc0_y2, xv, y2v);
        acc0_y3 = svmlalb_f32(acc0_y3, xv, y3v);
        acc0_y3 = svmlalt_f32(acc0_y3, xv, y3v);
        acc0_y4 = svmlalb_f32(acc0_y4, xv, y4v);
        acc0_y4 = svmlalt_f32(acc0_y4, xv, y4v);
        acc0_y5 = svmlalb_f32(acc0_y5, xv, y5v);
        acc0_y5 = svmlalt_f32(acc0_y5, xv, y5v);
        acc0_y6 = svmlalb_f32(acc0_y6, xv, y6v);
        acc0_y6 = svmlalt_f32(acc0_y6, xv, y6v);
        acc0_y7 = svmlalb_f32(acc0_y7, xv, y7v);
        acc0_y7 = svmlalt_f32(acc0_y7, xv, y7v);
    }

    uint64_t step8 = svcntw();
    for (; i + step8 <= d; i += step8) {
        svbool_t pg_half = svwhilelt_b16((uint64_t)0, (uint64_t)step8);
        svfloat16_t xv = svld1_f16(pg_half, x + i);
        svfloat16_t y0v = svld1_f16(pg_half, y0 + i);
        svfloat16_t y1v = svld1_f16(pg_half, y1 + i);
        svfloat16_t y2v = svld1_f16(pg_half, y2 + i);
        svfloat16_t y3v = svld1_f16(pg_half, y3 + i);
        svfloat16_t y4v = svld1_f16(pg_half, y4 + i);
        svfloat16_t y5v = svld1_f16(pg_half, y5 + i);
        svfloat16_t y6v = svld1_f16(pg_half, y6 + i);
        svfloat16_t y7v = svld1_f16(pg_half, y7 + i);

        acc0_y0 = svmlalb_f32(acc0_y0, xv, y0v);
        acc0_y0 = svmlalt_f32(acc0_y0, xv, y0v);
        acc0_y1 = svmlalb_f32(acc0_y1, xv, y1v);
        acc0_y1 = svmlalt_f32(acc0_y1, xv, y1v);
        acc0_y2 = svmlalb_f32(acc0_y2, xv, y2v);
        acc0_y2 = svmlalt_f32(acc0_y2, xv, y2v);
        acc0_y3 = svmlalb_f32(acc0_y3, xv, y3v);
        acc0_y3 = svmlalt_f32(acc0_y3, xv, y3v);
        acc0_y4 = svmlalb_f32(acc0_y4, xv, y4v);
        acc0_y4 = svmlalt_f32(acc0_y4, xv, y4v);
        acc0_y5 = svmlalb_f32(acc0_y5, xv, y5v);
        acc0_y5 = svmlalt_f32(acc0_y5, xv, y5v);
        acc0_y6 = svmlalb_f32(acc0_y6, xv, y6v);
        acc0_y6 = svmlalt_f32(acc0_y6, xv, y6v);
        acc0_y7 = svmlalb_f32(acc0_y7, xv, y7v);
        acc0_y7 = svmlalt_f32(acc0_y7, xv, y7v);
    }

    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc1_y0);
    acc2_y0 = svadd_f32_x(pg32, acc2_y0, acc3_y0);
    acc0_y0 = svadd_f32_x(pg32, acc0_y0, acc2_y0);
    float res_y0 = svaddv_f32(pg32, acc0_y0);

    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc1_y1);
    acc2_y1 = svadd_f32_x(pg32, acc2_y1, acc3_y1);
    acc0_y1 = svadd_f32_x(pg32, acc0_y1, acc2_y1);
    float res_y1 = svaddv_f32(pg32, acc0_y1);

    acc0_y2 = svadd_f32_x(pg32, acc0_y2, acc1_y2);
    acc2_y2 = svadd_f32_x(pg32, acc2_y2, acc3_y2);
    acc0_y2 = svadd_f32_x(pg32, acc0_y2, acc2_y2);
    float res_y2 = svaddv_f32(pg32, acc0_y2);

    acc0_y3 = svadd_f32_x(pg32, acc0_y3, acc1_y3);
    acc2_y3 = svadd_f32_x(pg32, acc2_y3, acc3_y3);
    acc0_y3 = svadd_f32_x(pg32, acc0_y3, acc2_y3);
    float res_y3 = svaddv_f32(pg32, acc0_y3);

    acc0_y4 = svadd_f32_x(pg32, acc0_y4, acc1_y4);
    acc2_y4 = svadd_f32_x(pg32, acc2_y4, acc3_y4);
    acc0_y4 = svadd_f32_x(pg32, acc0_y4, acc2_y4);
    float res_y4 = svaddv_f32(pg32, acc0_y4);

    acc0_y5 = svadd_f32_x(pg32, acc0_y5, acc1_y5);
    acc2_y5 = svadd_f32_x(pg32, acc2_y5, acc3_y5);
    acc0_y5 = svadd_f32_x(pg32, acc0_y5, acc2_y5);
    float res_y5 = svaddv_f32(pg32, acc0_y5);

    acc0_y6 = svadd_f32_x(pg32, acc0_y6, acc1_y6);
    acc2_y6 = svadd_f32_x(pg32, acc2_y6, acc3_y6);
    acc0_y6 = svadd_f32_x(pg32, acc0_y6, acc2_y6);
    float res_y6 = svaddv_f32(pg32, acc0_y6);

    acc0_y7 = svadd_f32_x(pg32, acc0_y7, acc1_y7);
    acc2_y7 = svadd_f32_x(pg32, acc2_y7, acc3_y7);
    acc0_y7 = svadd_f32_x(pg32, acc0_y7, acc2_y7);
    float res_y7 = svaddv_f32(pg32, acc0_y7);

    for (; i < d; i++) {
        res_y0 += (float)(x[i] * y0[i]);
        res_y1 += (float)(x[i] * y1[i]);
        res_y2 += (float)(x[i] * y2[i]);
        res_y3 += (float)(x[i] * y3[i]);
        res_y4 += (float)(x[i] * y4[i]);
        res_y5 += (float)(x[i] * y5[i]);
        res_y6 += (float)(x[i] * y6[i]);
        res_y7 += (float)(x[i] * y7[i]);
    }

    dis[0] = res_y0;
    dis[1] = res_y1;
    dis[2] = res_y2;
    dis[3] = res_y3;
    dis[4] = res_y4;
    dis[5] = res_y5;
    dis[6] = res_y6;
    dis[7] = res_y7;
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
 */
int krl_inner_product_by_idx_f16f32_sve2(
    float *dis, const uint16_t *x, const uint16_t *y, const int64_t *ids, size_t d, size_t ny)
{
    size_t i = 0;
    const float16_t *__restrict listy[8];

    /* Process vectors in batches of 8 */
    for (; i + 8 <= ny; i += 8) {
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        listy[4] = (const float16_t *)(y + *(ids + i + 4) * d);
        listy[5] = (const float16_t *)(y + *(ids + i + 5) * d);
        listy[6] = (const float16_t *)(y + *(ids + i + 6) * d);
        listy[7] = (const float16_t *)(y + *(ids + i + 7) * d);
        krl_inner_product_idx_batch8_f16f32_sve2((const float16_t *)x, listy, d, dis + i);
    }

    /* Handle remaining vectors in batches of 4 */
    if (ny & 4) {
        listy[0] = (const float16_t *)(y + *(ids + i) * d);
        listy[1] = (const float16_t *)(y + *(ids + i + 1) * d);
        listy[2] = (const float16_t *)(y + *(ids + i + 2) * d);
        listy[3] = (const float16_t *)(y + *(ids + i + 3) * d);
        krl_inner_product_idx_batch4_f16f32_sve2((const float16_t *)x, listy, d, dis + i);
        i += 4;
    }

    /* Handle remaining vectors in batches of 2 */
    if (ny & 2) {
        const float16_t *y0 = (const float16_t *)(y + *(ids + i) * d);
        const float16_t *y1 = (const float16_t *)(y + *(ids + i + 1) * d);
        krl_inner_product_idx_batch2_f16f32_sve2((const float16_t *)x, y0, y1, d, dis + i);
        i += 2;
    }

    /* Handle the last remaining vector */
    if (ny & 1) {
        dis[i] = krl_inner_product_f16f32_sve2(x, y + d * ids[i], d);
    }
    return SUCCESS;
}

/*
 * @brief Compute inner product between query vector and multiple vectors in the datab
 ase.
 * @param dis Pointer to the output array for storing the results (float).
 * @param x Pointer to the query vector (uint16_t).
 * @param y Pointer to the database vectors (uint16_t).
 * @param ny Number of vectors to compute inner products with.
 * @param d Dimension of the vectors.
 */
int krl_inner_product_ny_f16f32_sve2(float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d)
{
    size_t i = 0;

    /* Process vectors in batches of 8 */
    for (; i + 8 <= ny; i += 8) {
        krl_inner_product_batch8_f16f32_sve2((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
    }

    /* Handle remaining vectors in batches of 4 */
    if (ny & 4) {
        krl_inner_product_batch4_f16f32_sve2((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 4;
    }

    /* Handle remaining vectors in batches of 2 */
    if (ny & 2) {
        krl_inner_product_batch2_f16f32_sve2((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 2;
    }

    /* Handle the last remaining vector */
    if (ny & 1) {
        dis[i] = krl_inner_product_f16f32_sve2(x, y + i * d, d);
    }
    return SUCCESS;
}

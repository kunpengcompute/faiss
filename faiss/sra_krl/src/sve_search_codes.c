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
#include <arm_sve.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * @brief Compute distance for 4-bit codes using SIMD instructions.
 * @param M Number of subquantizers.
 * @param sim_table Precomputed distance table, layout (M, ksub).
 * @param codes Pointer to the 4-bit codes.
 * @param result Pointer to store the computed distances.
 * @param dis Initial distance for IVF.
 */
static void krl_distance_codes_simd_4b_kernel(
    const size_t M, const float16_t *sim_table, const uint8_t *codes, float *result, float16_t dis)
{
    svbool_t all_true = svdup_n_b16(true);

    svfloat16_t res0 = svdup_n_f16(dis);
    svfloat16_t res1 = svdup_n_f16(dis);
    svfloat16_t res2 = svdup_n_f16(dis);
    svfloat16_t res3 = svdup_n_f16(dis);

    svuint16_t cc0 = svld1ub_u16(all_true, codes);
    svuint16_t cc1 = svld1ub_u16(all_true, codes + 16);
    codes += 32;
    svfloat16_t table = svld1_f16(all_true, sim_table);
    sim_table += 16;
    for (size_t m = 1; m < M; m++) {
        const svuint16_t c0 = svand_n_u16_m(all_true, cc0, 0x0F);
        const svuint16_t c1 = svlsr_n_u16_m(all_true, cc0, 4);
        const svuint16_t c2 = svand_n_u16_m(all_true, cc1, 0x0F);
        const svuint16_t c3 = svlsr_n_u16_m(all_true, cc1, 4);
        cc0 = svld1ub_u16(all_true, codes);
        cc1 = svld1ub_u16(all_true, codes + 16);
        codes += 32;

        /* tbl */
        svfloat16_t dict0, dict1, dict2, dict3;
        asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
                     : [res0] "=w"(dict0)
                     : [ftable] "w"(table), [index0] "w"(c0)
                     :);
        asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
                     : [res1] "=w"(dict1)
                     : [ftable] "w"(table), [index1] "w"(c1)
                     :);
        asm volatile("tbl %[res2].h, %[ftable].h, %[index2].h"
                     : [res2] "=w"(dict2)
                     : [ftable] "w"(table), [index2] "w"(c2)
                     :);
        asm volatile("tbl %[res3].h, %[ftable].h, %[index3].h"
                     : [res3] "=w"(dict3)
                     : [ftable] "w"(table), [index3] "w"(c3)
                     :);

        table = svld1_f16(all_true, sim_table);
        sim_table += 16;

        /* add */
        res0 = svadd_f16_m(all_true, res0, dict0);
        res1 = svadd_f16_m(all_true, res1, dict1);
        res2 = svadd_f16_m(all_true, res2, dict2);
        res3 = svadd_f16_m(all_true, res3, dict3);
    }
    {
        const svuint16_t c0 = svand_n_u16_m(all_true, cc0, 0x0F);
        const svuint16_t c1 = svlsr_n_u16_m(all_true, cc0, 4);
        const svuint16_t c2 = svand_n_u16_m(all_true, cc1, 0x0F);
        const svuint16_t c3 = svlsr_n_u16_m(all_true, cc1, 4);

        /* tbl */
        svfloat16_t dict0, dict1, dict2, dict3;
        asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
                     : [res0] "=w"(dict0)
                     : [ftable] "w"(table), [index0] "w"(c0)
                     :);
        asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
                     : [res1] "=w"(dict1)
                     : [ftable] "w"(table), [index1] "w"(c1)
                     :);
        asm volatile("tbl %[res2].h, %[ftable].h, %[index2].h"
                     : [res2] "=w"(dict2)
                     : [ftable] "w"(table), [index2] "w"(c2)
                     :);
        asm volatile("tbl %[res3].h, %[ftable].h, %[index3].h"
                     : [res3] "=w"(dict3)
                     : [ftable] "w"(table), [index3] "w"(c3)
                     :);

        /* add */
        res0 = svadd_f16_m(all_true, res0, dict0);
        res1 = svadd_f16_m(all_true, res1, dict1);
        res2 = svadd_f16_m(all_true, res2, dict2);
        res3 = svadd_f16_m(all_true, res3, dict3);
    }

    svfloat16_t res4 = svtrn2_f16(res0, res0);
    svfloat16_t res5 = svtrn2_f16(res1, res1);
    svfloat16_t res6 = svtrn2_f16(res2, res2);
    svfloat16_t res7 = svtrn2_f16(res3, res3);

    svst1_f32(all_true, result, svcvt_f32_f16_x(all_true, res0));
    svst1_f32(all_true, result + 8, svcvt_f32_f16_x(all_true, res4));
    svst1_f32(all_true, result + 16, svcvt_f32_f16_x(all_true, res1));
    svst1_f32(all_true, result + 24, svcvt_f32_f16_x(all_true, res5));
    svst1_f32(all_true, result + 32, svcvt_f32_f16_x(all_true, res2));
    svst1_f32(all_true, result + 40, svcvt_f32_f16_x(all_true, res6));
    svst1_f32(all_true, result + 48, svcvt_f32_f16_x(all_true, res3));
    svst1_f32(all_true, result + 56, svcvt_f32_f16_x(all_true, res7));
}

/*
 * @brief Perform table lookup for 4-bit codes with 16-bit floating-point distances.
 * @param nsq Number of subquantizers.
 * @param ncode Number of codes.
 * @param codes Pointer to the 4-bit codes.
 * @param LUT Precomputed distance table, layout (M, ksub).
 * @param dis Pointer to store the computed distances.
 * @param dis0 Initial distance for IVF.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param dis_size Length of dis.
 */
int krl_table_lookup_4b_f16(const size_t nsq, const size_t ncode, const uint8_t *codes, const uint16_t *LUT, float *dis,
    uint16_t dis0, size_t codes_size, size_t LUT_size, size_t dis_size)
{
    size_t code_size = ((nsq + 1) >> 1);
    const float16_t f16_dis0 = *((const float16_t *)(&dis0));
    for (size_t j = 0; j < ncode; j += 64) {
        krl_distance_codes_simd_4b_kernel(nsq, (const float16_t *)LUT, codes + j * code_size, dis + j, f16_dis0);
    }
    return SUCCESS;
}
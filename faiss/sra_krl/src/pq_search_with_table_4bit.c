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
#include "safe_memory.h"
#include <stdio.h>
#include <math.h>

#define FILTER_OFFSET                                              \
    {                                                              \
        0x101, 0x202, 0x404, 0x808, 0x1010, 0x2020, 0x4040, 0x8080 \
    }

#if defined(__GNUC__) && !defined(__clang__)
#define TBL2_LOOKUP_ASM(result, dict, indices) do { \
    uint8x16_t _tmp_result; \
    asm volatile( \
        "mov v16.16B, %[d0].16B\n\t" \
        "mov v17.16B, %[d1].16B\n\t" \
        "tbl %[res].16B, {v16.16B, v17.16B}, %[idx].16B" \
        : [res] "=w"(_tmp_result) \
        : [d0] "w"((dict).val[0]), [d1] "w"((dict).val[1]), [idx] "w"(indices) \
        : "v16", "v17" \
    ); \
    result = _tmp_result; \
} while(0)
#else
#define TBL2_LOOKUP_ASM(result, dict, indices) do { \
    asm volatile( \
        "tbl %[res].16B, { %[d0].16B, %[d1].16B }, %[idx].16B" \
        : [res] "=w"(result) \
        : [d0] "w"((dict).val[0]), [d1] "w"((dict).val[1]), [idx] "w"(indices) \
    ); \
} while(0)
#endif

/*
 * @brief Accumulate lookup table (LUT) results and apply filtering based on thresholds.
 * @param nq Number of queries.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=32).
 * @param LUT Pointer to the precomputed distances array, layout (nq, nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances.
 * @param threshold Pointer to the filter threshold array.
 * @param lt_mask Pointer to the array storing filter results.
 * @param keep_min Filter comparison rule (0 for keep minimum, 1 for keep maximum).
 */
static void krl_lut_accumulate_filter(int nq, int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *distance,
    const uint16_t *threshold, uint32_t *lt_mask, int keep_min)
{
    const int NQB = nq << 2;
    uint8x16_t mask0 = vld1q_u8(codes);
    uint8x16_t mask1 = vld1q_u8(codes + 16);
    uint16x8_t result[16];
    codes += 32;
    const uint8x16_t mask16 = vreinterpretq_u8_u16(vdupq_n_u16(0x1000));
    const uint8x16_t maskone = vreinterpretq_u8_u16(vdupq_n_u16(0x0100));

    /* first loop */
    if (likely(nsq > 2)) {
        __builtin_prefetch(codes + 768, 0, 0);
        const uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
        const uint8x16_t mask1_1 = vsriq_n_u8(mask16, mask0, 4);
        const uint8x16_t mask0_2 = vsliq_n_u8(mask1, maskone, 4);
        const uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
        /* preload */
        mask0 = vld1q_u8(codes);
        mask1 = vld1q_u8(codes + 16);
        codes += 32;

        for (int q = 0; q < NQB; q += 4) {
            const uint8x16x2_t dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            uint8x16_t res0, res1, res0_2, res1_2;
            TBL2_LOOKUP_ASM(res0, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(res1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(res0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(res1_2, dictCombine, mask1_2);

            result[q] = vpaddlq_u8(res0);
            result[q + 1] = vpaddlq_u8(res1);
            result[q + 2] = vpaddlq_u8(res0_2);
            result[q + 3] = vpaddlq_u8(res1_2);
        }
    } else {
        result[0] = vdupq_n_u16(0);
        result[1] = vdupq_n_u16(0);
        result[2] = vdupq_n_u16(0);
        result[3] = vdupq_n_u16(0);
        result[4] = vdupq_n_u16(0);
        result[5] = vdupq_n_u16(0);
        result[6] = vdupq_n_u16(0);
        result[7] = vdupq_n_u16(0);
        result[8] = vdupq_n_u16(0);
        result[9] = vdupq_n_u16(0);
        result[10] = vdupq_n_u16(0);
        result[11] = vdupq_n_u16(0);
        result[12] = vdupq_n_u16(0);
        result[13] = vdupq_n_u16(0);
        result[14] = vdupq_n_u16(0);
        result[15] = vdupq_n_u16(0);
    }
    /* main loop*/
    for (int sq = 2; sq < nsq - 2; sq += 2) {
        __builtin_prefetch(codes + 768, 0, 0);
        const uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
        const uint8x16_t mask1_1 = vsriq_n_u8(mask16, mask0, 4);
        const uint8x16_t mask0_2 = vsliq_n_u8(mask1, maskone, 4);
        const uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
        /* preload */
        mask0 = vld1q_u8(codes);
        mask1 = vld1q_u8(codes + 16);
        codes += 32;

        for (int q = 0; q < NQB; q += 4) {
            const uint8x16x2_t dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            uint8x16_t res0, res1, res0_2, res1_2;
            TBL2_LOOKUP_ASM(res0, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(res1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(res0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(res1_2, dictCombine, mask1_2);

            result[q] = vpadalq_u8(result[q], res0);
            result[q + 1] = vpadalq_u8(result[q + 1], res1);
            result[q + 2] = vpadalq_u8(result[q + 2], res0_2);
            result[q + 3] = vpadalq_u8(result[q + 3], res1_2);
        }
    }

    /* last loop without preload and prefetch */
    {
        const uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
        const uint8x16_t mask1_1 = vsriq_n_u8(mask16, mask0, 4);
        const uint8x16_t mask0_2 = vsliq_n_u8(mask1, maskone, 4);
        const uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);

        for (int q = 0; q < NQB; q += 4) {
            const uint8x16x2_t dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            uint8x16_t res0, res1, res0_2, res1_2;
            TBL2_LOOKUP_ASM(res0, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(res1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(res0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(res1_2, dictCombine, mask1_2);

            result[q] = vpadalq_u8(result[q], res0);
            result[q + 1] = vpadalq_u8(result[q + 1], res1);
            result[q + 2] = vpadalq_u8(result[q + 2], res0_2);
            result[q + 3] = vpadalq_u8(result[q + 3], res1_2);
        }
    }

    constexpr uint16x8_t offset = FILTER_OFFSET;
    uint16x8_t cmp0, cmp1, cmp2, cmp3;
    for (int i = 0; i < nq; ++i) {
        const uint16x8_t threshold_simd = vdupq_n_u16(threshold[i]);
        if (keep_min == 0) {
            cmp0 = vcgtq_u16(result[4 * i], threshold_simd);
            cmp1 = vcgtq_u16(result[4 * i + 1], threshold_simd);
            cmp2 = vcgtq_u16(result[4 * i + 2], threshold_simd);
            cmp3 = vcgtq_u16(result[4 * i + 3], threshold_simd);
        } else {
            cmp0 = vcltq_u16(result[4 * i], threshold_simd);
            cmp1 = vcltq_u16(result[4 * i + 1], threshold_simd);
            cmp2 = vcltq_u16(result[4 * i + 2], threshold_simd);
            cmp3 = vcltq_u16(result[4 * i + 3], threshold_simd);
        }
        cmp0 = vsliq_n_u16(cmp0, cmp1, 8);
        cmp2 = vsliq_n_u16(cmp2, cmp3, 8);
        cmp0 = vandq_u16(cmp0, offset);
        cmp2 = vandq_u16(cmp2, offset);
        uint32x4_t push_mask0 = vpaddlq_u16(cmp0);
        uint32x4_t push_mask2 = vpaddlq_u16(cmp2);
        push_mask0 = vsliq_n_u32(push_mask0, push_mask2, 16);
        lt_mask[i] = vaddvq_u32(push_mask0);
        vst1q_u16(distance + 32 * i, result[4 * i]);
        vst1q_u16(distance + 32 * i + 8, result[4 * i + 1]);
        vst1q_u16(distance + 32 * i + 16, result[4 * i + 2]);
        vst1q_u16(distance + 32 * i + 24, result[4 * i + 3]);
    }
}

/*
 * @brief Perform fast table lookup and filtering operations using vectorized instructions.
 * @param nq Number of queries.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=32).
 * @param LUT Pointer to the precomputed distances array, layout (nq, nsq, ksub=16).
 * @param dis Pointer to the array storing computed distances.
 * @param threshold Pointer to the filter threshold array.
 * @param lt_mask Pointer to the array storing filter results.
 * @param keep_min Filter comparison rule (0 for keep minimum, 1 for keep maximum).
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param threshold_size Length of threshold.
 * @param lt_mask_size Length of lt_mask.
 */
int krl_fast_table_lookup_step(int nq, int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    const uint16_t *threshold, uint32_t *lt_mask, int keep_min, size_t codes_size, size_t LUT_size,
    size_t threshold_size, size_t lt_mask_size)
{
    int q = 0;
    const int left_q = nq & 3;
    for (; q <= nq - 4; q += 4) {
        krl_lut_accumulate_filter(
            4, nsq, codes, LUT + q * nsq * 16, dis + q * 32, threshold + q, lt_mask + q, keep_min);
    }
    if (left_q) {
        krl_lut_accumulate_filter(
            left_q, nsq, codes, LUT + q * nsq * 16, dis + q * 32, threshold + q, lt_mask + q, keep_min);
    }
    return SUCCESS;
}

/*
 * @brief Perform fast table lookup and filtering operations for single query with batch size 32.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=32).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 32.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 1.
 */
template <int keep_min = 0>
static inline void krl_table_lookup_fast_scan_bs32(
    int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *distance, uint16_t threshold, uint32_t *lt_mask)
{
    uint16x8_t result[4];
    {
        uint8x16_t mask0 = vld1q_u8(codes);
        uint8x16_t mask1 = vld1q_u8(codes + 16);
        codes += 32;
        const uint8x16_t mask16 = vreinterpretq_u8_u16(vdupq_n_u16(0x1000));
        const uint8x16_t maskone = vreinterpretq_u8_u16(vdupq_n_u16(0x0100));
        uint8x16x2_t dictCombine = vld1q_u8_x2(LUT);
        LUT += 32;

        /* first loop */
        if (likely(nsq > 2)) {
            uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            mask0 = vld1q_u8(codes);
            uint8x16_t mask1_1 = vsliq_n_u8(mask1, maskone, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
            mask1 = vld1q_u8(codes + 16);
            codes += 32;

            TBL2_LOOKUP_ASM(mask0_1, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1_1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            result[0] = vpaddlq_u8(mask0_1);
            result[1] = vpaddlq_u8(mask0_2);
            result[2] = vpaddlq_u8(mask1_1);
            result[3] = vpaddlq_u8(mask1_2);
        } else {
            result[0] = vdupq_n_u16(0);
            result[1] = vdupq_n_u16(0);
            result[2] = vdupq_n_u16(0);
            result[3] = vdupq_n_u16(0);
        }

        /* main loop */
        for (int sq = 2; sq < nsq - 2; sq += 2) {
            uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            mask0 = vld1q_u8(codes);
            uint8x16_t mask1_1 = vsliq_n_u8(mask1, maskone, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
            mask1 = vld1q_u8(codes + 16);
            codes += 32;

            TBL2_LOOKUP_ASM(mask0_1, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1_1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            result[0] = vpadalq_u8(result[0], mask0_1);
            result[1] = vpadalq_u8(result[1], mask0_2);
            result[2] = vpadalq_u8(result[2], mask1_1);
            result[3] = vpadalq_u8(result[3], mask1_2);
        }

        /* last loop without preload */
        {
            uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            uint8x16_t mask1_1 = vsliq_n_u8(mask1, maskone, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);

            TBL2_LOOKUP_ASM(mask0_1, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1_1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            result[0] = vpadalq_u8(result[0], mask0_1);
            result[1] = vpadalq_u8(result[1], mask0_2);
            result[2] = vpadalq_u8(result[2], mask1_1);
            result[3] = vpadalq_u8(result[3], mask1_2);
        }
    }

    if constexpr (keep_min == 0) {
        const uint16x8_t threshold_simd = vdupq_n_u16(threshold);
        constexpr uint16x8_t offset = FILTER_OFFSET;

        uint16x8_t cmp0_0 = vcgtq_u16(result[0], threshold_simd);
        uint16x8_t cmp0_1 = vcgtq_u16(result[1], threshold_simd);
        uint16x8_t cmp0_2 = vcgtq_u16(result[2], threshold_simd);
        uint16x8_t cmp0_3 = vcgtq_u16(result[3], threshold_simd);

        cmp0_0 = vsliq_n_u16(cmp0_0, cmp0_1, 8);
        cmp0_2 = vsliq_n_u16(cmp0_2, cmp0_3, 8);
        cmp0_0 = vandq_u16(cmp0_0, offset);
        cmp0_2 = vandq_u16(cmp0_2, offset);

        lt_mask[0] = vaddvq_u16(cmp0_0) | (((uint32_t)vaddvq_u16(cmp0_2)) << 16);
        if (lt_mask[0]) {
            vst1q_u16(distance, result[0]);
            vst1q_u16(distance + 8, result[1]);
            vst1q_u16(distance + 16, result[2]);
            vst1q_u16(distance + 24, result[3]);
        }
    } else {
        const uint16x8_t threshold_simd = vdupq_n_u16(threshold);
        constexpr uint16x8_t offset = FILTER_OFFSET;

        uint16x8_t cmp0_0 = vcltq_u16(result[0], threshold_simd);
        uint16x8_t cmp0_1 = vcltq_u16(result[1], threshold_simd);
        uint16x8_t cmp0_2 = vcltq_u16(result[2], threshold_simd);
        uint16x8_t cmp0_3 = vcltq_u16(result[3], threshold_simd);

        cmp0_0 = vsliq_n_u16(cmp0_0, cmp0_1, 8);
        cmp0_2 = vsliq_n_u16(cmp0_2, cmp0_3, 8);
        cmp0_0 = vandq_u16(cmp0_0, offset);
        cmp0_2 = vandq_u16(cmp0_2, offset);

        lt_mask[0] = vaddvq_u16(cmp0_0) | (((uint32_t)vaddvq_u16(cmp0_2)) << 16);
        if (lt_mask[0]) {
            vst1q_u16(distance, result[0]);
            vst1q_u16(distance + 8, result[1]);
            vst1q_u16(distance + 16, result[2]);
            vst1q_u16(distance + 24, result[3]);
        }
    }
}

/*
 * @brief Perform fast table lookup and filtering operations for single query with batch size 64.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=96).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 96.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 3.
 */
template <int keep_min = 0>
static inline void krl_table_lookup_fast_scan_bs64(
    int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *distance, uint16_t threshold, uint32_t *lt_mask)
{
    uint16x8_t result[8];
    {
        uint8x16_t mask0 = vld1q_u8(codes);
        uint8x16_t mask1 = vld1q_u8(codes + 16);
        uint8x16_t mask2 = vld1q_u8(codes + 32);
        uint8x16_t mask3 = vld1q_u8(codes + 48);
        codes += 64;
        const uint8x16_t mask16 = vreinterpretq_u8_u16(vdupq_n_u16(0x1000));
        const uint8x16_t maskone = vreinterpretq_u8_u16(vdupq_n_u16(0x0100));
        uint8x16x2_t dictCombine = vld1q_u8_x2(LUT);
        LUT += 32;
        /* first loop */
        if (likely(nsq > 2)) {
            uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            mask0 = vld1q_u8(codes);
            uint8x16_t mask1_1 = vsliq_n_u8(mask1, maskone, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
            mask1 = vld1q_u8(codes + 16);
            uint8x16_t mask2_1 = vsliq_n_u8(mask2, maskone, 4);
            uint8x16_t mask2_2 = vsriq_n_u8(mask16, mask2, 4);
            mask2 = vld1q_u8(codes + 32);
            uint8x16_t mask3_1 = vsliq_n_u8(mask3, maskone, 4);
            uint8x16_t mask3_2 = vsriq_n_u8(mask16, mask3, 4);
            mask3 = vld1q_u8(codes + 48);
            /* preload */
            codes += 64;

            TBL2_LOOKUP_ASM(mask0_1, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1_1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            TBL2_LOOKUP_ASM(mask2_1, dictCombine, mask2_1);
            TBL2_LOOKUP_ASM(mask2_2, dictCombine, mask2_2);
            TBL2_LOOKUP_ASM(mask3_1, dictCombine, mask3_1);
            TBL2_LOOKUP_ASM(mask3_2, dictCombine, mask3_2);

            dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            result[0] = vpaddlq_u8(mask0_1);
            result[1] = vpaddlq_u8(mask0_2);
            result[2] = vpaddlq_u8(mask1_1);
            result[3] = vpaddlq_u8(mask1_2);
            result[4] = vpaddlq_u8(mask2_1);
            result[5] = vpaddlq_u8(mask2_2);
            result[6] = vpaddlq_u8(mask3_1);
            result[7] = vpaddlq_u8(mask3_2);
        } else {
            result[0] = vdupq_n_u16(0);
            result[1] = vdupq_n_u16(0);
            result[2] = vdupq_n_u16(0);
            result[3] = vdupq_n_u16(0);
            result[4] = vdupq_n_u16(0);
            result[5] = vdupq_n_u16(0);
            result[6] = vdupq_n_u16(0);
            result[7] = vdupq_n_u16(0);
        }
        /* main loop */
        for (int sq = 2; sq < nsq - 2; sq += 2) {
            uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            mask0 = vld1q_u8(codes);
            uint8x16_t mask1_1 = vsliq_n_u8(mask1, maskone, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
            mask1 = vld1q_u8(codes + 16);
            uint8x16_t mask2_1 = vsliq_n_u8(mask2, maskone, 4);
            uint8x16_t mask2_2 = vsriq_n_u8(mask16, mask2, 4);
            mask2 = vld1q_u8(codes + 32);
            uint8x16_t mask3_1 = vsliq_n_u8(mask3, maskone, 4);
            uint8x16_t mask3_2 = vsriq_n_u8(mask16, mask3, 4);
            mask3 = vld1q_u8(codes + 48);

            /* preload */
            codes += 64;

            TBL2_LOOKUP_ASM(mask0_1, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1_1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            TBL2_LOOKUP_ASM(mask2_1, dictCombine, mask2_1);
            TBL2_LOOKUP_ASM(mask2_2, dictCombine, mask2_2);
            TBL2_LOOKUP_ASM(mask3_1, dictCombine, mask3_1);
            TBL2_LOOKUP_ASM(mask3_2, dictCombine, mask3_2);

            dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            result[0] = vpadalq_u8(result[0], mask0_1);
            result[1] = vpadalq_u8(result[1], mask0_2);
            result[2] = vpadalq_u8(result[2], mask1_1);
            result[3] = vpadalq_u8(result[3], mask1_2);
            result[4] = vpadalq_u8(result[4], mask2_1);
            result[5] = vpadalq_u8(result[5], mask2_2);
            result[6] = vpadalq_u8(result[6], mask3_1);
            result[7] = vpadalq_u8(result[7], mask3_2);
        }
        /* last loop without preload and prefetch */
        {
            uint8x16_t mask0_1 = vsliq_n_u8(mask0, maskone, 4);
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            uint8x16_t mask1_1 = vsliq_n_u8(mask1, maskone, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
            uint8x16_t mask2_1 = vsliq_n_u8(mask2, maskone, 4);
            uint8x16_t mask2_2 = vsriq_n_u8(mask16, mask2, 4);
            uint8x16_t mask3_1 = vsliq_n_u8(mask3, maskone, 4);
            uint8x16_t mask3_2 = vsriq_n_u8(mask16, mask3, 4);

            TBL2_LOOKUP_ASM(mask0_1, dictCombine, mask0_1);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1_1, dictCombine, mask1_1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            TBL2_LOOKUP_ASM(mask2_1, dictCombine, mask2_1);
            TBL2_LOOKUP_ASM(mask2_2, dictCombine, mask2_2);
            TBL2_LOOKUP_ASM(mask3_1, dictCombine, mask3_1);
            TBL2_LOOKUP_ASM(mask3_2, dictCombine, mask3_2);

            result[0] = vpadalq_u8(result[0], mask0_1);
            result[1] = vpadalq_u8(result[1], mask0_2);
            result[2] = vpadalq_u8(result[2], mask1_1);
            result[3] = vpadalq_u8(result[3], mask1_2);
            result[4] = vpadalq_u8(result[4], mask2_1);
            result[5] = vpadalq_u8(result[5], mask2_2);
            result[6] = vpadalq_u8(result[6], mask3_1);
            result[7] = vpadalq_u8(result[7], mask3_2);
        }
    }
    if constexpr (keep_min == 0) {
        const uint16x8_t threshold_simd = vdupq_n_u16(threshold);
        constexpr uint16x8_t offset = FILTER_OFFSET;

        uint16x8_t cmp0_0 = vcgtq_u16(result[0], threshold_simd);
        uint16x8_t cmp0_1 = vcgtq_u16(result[1], threshold_simd);
        uint16x8_t cmp0_2 = vcgtq_u16(result[2], threshold_simd);
        uint16x8_t cmp0_3 = vcgtq_u16(result[3], threshold_simd);
        uint16x8_t cmp1_0 = vcgtq_u16(result[4], threshold_simd);
        uint16x8_t cmp1_1 = vcgtq_u16(result[5], threshold_simd);
        uint16x8_t cmp1_2 = vcgtq_u16(result[6], threshold_simd);
        uint16x8_t cmp1_3 = vcgtq_u16(result[7], threshold_simd);

        cmp0_0 = vsliq_n_u16(cmp0_0, cmp0_1, 8);
        cmp0_2 = vsliq_n_u16(cmp0_2, cmp0_3, 8);
        cmp1_0 = vsliq_n_u16(cmp1_0, cmp1_1, 8);
        cmp1_2 = vsliq_n_u16(cmp1_2, cmp1_3, 8);
        cmp0_0 = vandq_u16(cmp0_0, offset);
        cmp0_2 = vandq_u16(cmp0_2, offset);
        cmp1_0 = vandq_u16(cmp1_0, offset);
        cmp1_2 = vandq_u16(cmp1_2, offset);

        lt_mask[0] = vaddvq_u16(cmp0_0) | (((uint32_t)vaddvq_u16(cmp0_2)) << 16);
        lt_mask[1] = vaddvq_u16(cmp1_0) | (((uint32_t)vaddvq_u16(cmp1_2)) << 16);
        if (lt_mask[0]) {
            vst1q_u16(distance, result[0]);
            vst1q_u16(distance + 8, result[1]);
            vst1q_u16(distance + 16, result[2]);
            vst1q_u16(distance + 24, result[3]);
        }
        if (lt_mask[1]) {
            vst1q_u16(distance + 32, result[4]);
            vst1q_u16(distance + 40, result[5]);
            vst1q_u16(distance + 48, result[6]);
            vst1q_u16(distance + 56, result[7]);
        }
    } else {
        const uint16x8_t threshold_simd = vdupq_n_u16(threshold);
        constexpr uint16x8_t offset = FILTER_OFFSET;

        uint16x8_t cmp0_0 = vcltq_u16(result[0], threshold_simd);
        uint16x8_t cmp0_1 = vcltq_u16(result[1], threshold_simd);
        uint16x8_t cmp0_2 = vcltq_u16(result[2], threshold_simd);
        uint16x8_t cmp0_3 = vcltq_u16(result[3], threshold_simd);
        uint16x8_t cmp1_0 = vcltq_u16(result[4], threshold_simd);
        uint16x8_t cmp1_1 = vcltq_u16(result[5], threshold_simd);
        uint16x8_t cmp1_2 = vcltq_u16(result[6], threshold_simd);
        uint16x8_t cmp1_3 = vcltq_u16(result[7], threshold_simd);

        cmp0_0 = vsliq_n_u16(cmp0_0, cmp0_1, 8);
        cmp0_2 = vsliq_n_u16(cmp0_2, cmp0_3, 8);
        cmp1_0 = vsliq_n_u16(cmp1_0, cmp1_1, 8);
        cmp1_2 = vsliq_n_u16(cmp1_2, cmp1_3, 8);
        cmp0_0 = vandq_u16(cmp0_0, offset);
        cmp0_2 = vandq_u16(cmp0_2, offset);
        cmp1_0 = vandq_u16(cmp1_0, offset);
        cmp1_2 = vandq_u16(cmp1_2, offset);

        lt_mask[0] = vaddvq_u16(cmp0_0) | (((uint32_t)vaddvq_u16(cmp0_2)) << 16);
        lt_mask[1] = vaddvq_u16(cmp1_0) | (((uint32_t)vaddvq_u16(cmp1_2)) << 16);
        if (lt_mask[0]) {
            vst1q_u16(distance, result[0]);
            vst1q_u16(distance + 8, result[1]);
            vst1q_u16(distance + 16, result[2]);
            vst1q_u16(distance + 24, result[3]);
        }
        if (lt_mask[1]) {
            vst1q_u16(distance + 32, result[4]);
            vst1q_u16(distance + 40, result[5]);
            vst1q_u16(distance + 48, result[6]);
            vst1q_u16(distance + 56, result[7]);
        }
    }
}

/*
 * @brief Perform fast table lookup and filtering operations for single query with batch size 96.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=96).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 96.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 3.
 */
template <int keep_min = 0>
static inline void krl_table_lookup_fast_scan_bs96(
    int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *distance, uint16_t threshold, uint32_t *lt_mask)
{
    uint16x8_t result[12];
    {
        uint8x16_t mask0 = vld1q_u8(codes);
        uint8x16_t mask1 = vld1q_u8(codes + 16);
        uint8x16_t mask2 = vld1q_u8(codes + 32);
        uint8x16_t mask3 = vld1q_u8(codes + 48);
        uint8x16_t mask4 = vld1q_u8(codes + 64);
        uint8x16_t mask5 = vld1q_u8(codes + 80);
        codes += 96;
        uint8x16x2_t dictCombine = vld1q_u8_x2(LUT);
        LUT += 32;
        const uint8x16_t mask16 = vreinterpretq_u8_u16(vdupq_n_u16(0x1000));
        const uint8x16_t maskone = vreinterpretq_u8_u16(vdupq_n_u16(0x0100));
        /* first loop */
        if (likely(nsq > 2)) {
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
            uint8x16_t mask2_2 = vsriq_n_u8(mask16, mask2, 4);
            uint8x16_t mask3_2 = vsriq_n_u8(mask16, mask3, 4);
            uint8x16_t mask4_2 = vsriq_n_u8(mask16, mask4, 4);
            uint8x16_t mask5_2 = vsriq_n_u8(mask16, mask5, 4);

            mask0 = vsliq_n_u8(mask0, maskone, 4);
            mask1 = vsliq_n_u8(mask1, maskone, 4);
            mask2 = vsliq_n_u8(mask2, maskone, 4);
            mask3 = vsliq_n_u8(mask3, maskone, 4);
            mask4 = vsliq_n_u8(mask4, maskone, 4);
            mask5 = vsliq_n_u8(mask5, maskone, 4);

            TBL2_LOOKUP_ASM(mask0, dictCombine, mask0);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1, dictCombine, mask1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            TBL2_LOOKUP_ASM(mask2, dictCombine, mask2);
            TBL2_LOOKUP_ASM(mask2_2, dictCombine, mask2_2);
            TBL2_LOOKUP_ASM(mask3, dictCombine, mask3);
            TBL2_LOOKUP_ASM(mask3_2, dictCombine, mask3_2);

            TBL2_LOOKUP_ASM(mask4, dictCombine, mask4);
            TBL2_LOOKUP_ASM(mask4_2, dictCombine, mask4_2);
            TBL2_LOOKUP_ASM(mask5, dictCombine, mask5);
            TBL2_LOOKUP_ASM(mask5_2, dictCombine, mask5_2);

            dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            result[0] = vpaddlq_u8(mask0);
            result[2] = vpaddlq_u8(mask1);
            result[4] = vpaddlq_u8(mask2);
            result[6] = vpaddlq_u8(mask3);
            result[8] = vpaddlq_u8(mask4);
            result[10] = vpaddlq_u8(mask5);

            mask0 = vld1q_u8(codes);
            mask1 = vld1q_u8(codes + 16);
            mask2 = vld1q_u8(codes + 32);
            mask3 = vld1q_u8(codes + 48);
            mask4 = vld1q_u8(codes + 64);
            mask5 = vld1q_u8(codes + 80);
            codes += 96;

            result[1] = vpaddlq_u8(mask0_2);
            result[3] = vpaddlq_u8(mask1_2);
            result[5] = vpaddlq_u8(mask2_2);
            result[7] = vpaddlq_u8(mask3_2);
            result[9] = vpaddlq_u8(mask4_2);
            result[11] = vpaddlq_u8(mask5_2);
        } else {
            result[0] = vdupq_n_u16(0);
            result[1] = vdupq_n_u16(0);
            result[2] = vdupq_n_u16(0);
            result[3] = vdupq_n_u16(0);
            result[4] = vdupq_n_u16(0);
            result[5] = vdupq_n_u16(0);
            result[6] = vdupq_n_u16(0);
            result[7] = vdupq_n_u16(0);
            result[8] = vdupq_n_u16(0);
            result[9] = vdupq_n_u16(0);
            result[10] = vdupq_n_u16(0);
            result[11] = vdupq_n_u16(0);
        }
        /* main loop */
        for (int sq = 2; sq < nsq - 2; sq += 2) {
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
            uint8x16_t mask2_2 = vsriq_n_u8(mask16, mask2, 4);
            uint8x16_t mask3_2 = vsriq_n_u8(mask16, mask3, 4);
            uint8x16_t mask4_2 = vsriq_n_u8(mask16, mask4, 4);
            uint8x16_t mask5_2 = vsriq_n_u8(mask16, mask5, 4);

            mask0 = vsliq_n_u8(mask0, maskone, 4);
            mask1 = vsliq_n_u8(mask1, maskone, 4);
            mask2 = vsliq_n_u8(mask2, maskone, 4);
            mask3 = vsliq_n_u8(mask3, maskone, 4);
            mask4 = vsliq_n_u8(mask4, maskone, 4);
            mask5 = vsliq_n_u8(mask5, maskone, 4);

            TBL2_LOOKUP_ASM(mask0, dictCombine, mask0);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1, dictCombine, mask1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            TBL2_LOOKUP_ASM(mask2, dictCombine, mask2);
            TBL2_LOOKUP_ASM(mask2_2, dictCombine, mask2_2);
            TBL2_LOOKUP_ASM(mask3, dictCombine, mask3);
            TBL2_LOOKUP_ASM(mask3_2, dictCombine, mask3_2);

            TBL2_LOOKUP_ASM(mask4, dictCombine, mask4);
            TBL2_LOOKUP_ASM(mask4_2, dictCombine, mask4_2);
            TBL2_LOOKUP_ASM(mask5, dictCombine, mask5);
            TBL2_LOOKUP_ASM(mask5_2, dictCombine, mask5_2);

            dictCombine = vld1q_u8_x2(LUT);
            LUT += 32;

            result[0] = vpadalq_u8(result[0], mask0);
            result[2] = vpadalq_u8(result[2], mask1);
            result[4] = vpadalq_u8(result[4], mask2);
            result[6] = vpadalq_u8(result[6], mask3);
            result[8] = vpadalq_u8(result[8], mask4);
            result[10] = vpadalq_u8(result[10], mask5);

            mask0 = vld1q_u8(codes);
            mask1 = vld1q_u8(codes + 16);
            mask2 = vld1q_u8(codes + 32);
            mask3 = vld1q_u8(codes + 48);
            mask4 = vld1q_u8(codes + 64);
            mask5 = vld1q_u8(codes + 80);
            codes += 96;

            result[1] = vpadalq_u8(result[1], mask0_2);
            result[3] = vpadalq_u8(result[3], mask1_2);
            result[5] = vpadalq_u8(result[5], mask2_2);
            result[7] = vpadalq_u8(result[7], mask3_2);
            result[9] = vpadalq_u8(result[9], mask4_2);
            result[11] = vpadalq_u8(result[11], mask5_2);
        }
        /* last loop without preload and prefetch */
        {
            uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
            uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
            uint8x16_t mask2_2 = vsriq_n_u8(mask16, mask2, 4);
            uint8x16_t mask3_2 = vsriq_n_u8(mask16, mask3, 4);
            uint8x16_t mask4_2 = vsriq_n_u8(mask16, mask4, 4);
            uint8x16_t mask5_2 = vsriq_n_u8(mask16, mask5, 4);

            mask0 = vsliq_n_u8(mask0, maskone, 4);
            mask1 = vsliq_n_u8(mask1, maskone, 4);
            mask2 = vsliq_n_u8(mask2, maskone, 4);
            mask3 = vsliq_n_u8(mask3, maskone, 4);
            mask4 = vsliq_n_u8(mask4, maskone, 4);
            mask5 = vsliq_n_u8(mask5, maskone, 4);

            TBL2_LOOKUP_ASM(mask0, dictCombine, mask0);
            TBL2_LOOKUP_ASM(mask0_2, dictCombine, mask0_2);
            TBL2_LOOKUP_ASM(mask1, dictCombine, mask1);
            TBL2_LOOKUP_ASM(mask1_2, dictCombine, mask1_2);

            TBL2_LOOKUP_ASM(mask2, dictCombine, mask2);
            TBL2_LOOKUP_ASM(mask2_2, dictCombine, mask2_2);
            TBL2_LOOKUP_ASM(mask3, dictCombine, mask3);
            TBL2_LOOKUP_ASM(mask3_2, dictCombine, mask3_2);

            TBL2_LOOKUP_ASM(mask4, dictCombine, mask4);
            TBL2_LOOKUP_ASM(mask4_2, dictCombine, mask4_2);
            TBL2_LOOKUP_ASM(mask5, dictCombine, mask5);
            TBL2_LOOKUP_ASM(mask5_2, dictCombine, mask5_2);

            result[0] = vpadalq_u8(result[0], mask0);
            result[1] = vpadalq_u8(result[1], mask0_2);
            result[2] = vpadalq_u8(result[2], mask1);
            result[3] = vpadalq_u8(result[3], mask1_2);
            result[4] = vpadalq_u8(result[4], mask2);
            result[5] = vpadalq_u8(result[5], mask2_2);
            result[6] = vpadalq_u8(result[6], mask3);
            result[7] = vpadalq_u8(result[7], mask3_2);
            result[8] = vpadalq_u8(result[8], mask4);
            result[9] = vpadalq_u8(result[9], mask4_2);
            result[10] = vpadalq_u8(result[10], mask5);
            result[11] = vpadalq_u8(result[11], mask5_2);
        }
    }
    if constexpr (keep_min == 0) {
        const uint16x8_t threshold_simd = vdupq_n_u16(threshold);
        constexpr uint16x8_t offset = FILTER_OFFSET;
        uint16x8_t cmp0_0 = vcgtq_u16(result[0], threshold_simd);
        uint16x8_t cmp0_1 = vcgtq_u16(result[1], threshold_simd);
        uint16x8_t cmp0_2 = vcgtq_u16(result[2], threshold_simd);
        uint16x8_t cmp0_3 = vcgtq_u16(result[3], threshold_simd);
        uint16x8_t cmp1_0 = vcgtq_u16(result[4], threshold_simd);
        uint16x8_t cmp1_1 = vcgtq_u16(result[5], threshold_simd);
        uint16x8_t cmp1_2 = vcgtq_u16(result[6], threshold_simd);
        uint16x8_t cmp1_3 = vcgtq_u16(result[7], threshold_simd);
        uint16x8_t cmp2_0 = vcgtq_u16(result[8], threshold_simd);
        uint16x8_t cmp2_1 = vcgtq_u16(result[9], threshold_simd);
        uint16x8_t cmp2_2 = vcgtq_u16(result[10], threshold_simd);
        uint16x8_t cmp2_3 = vcgtq_u16(result[11], threshold_simd);

        cmp0_0 = vsliq_n_u16(cmp0_0, cmp0_1, 8);
        cmp0_2 = vsliq_n_u16(cmp0_2, cmp0_3, 8);
        cmp1_0 = vsliq_n_u16(cmp1_0, cmp1_1, 8);
        cmp1_2 = vsliq_n_u16(cmp1_2, cmp1_3, 8);
        cmp2_0 = vsliq_n_u16(cmp2_0, cmp2_1, 8);
        cmp2_2 = vsliq_n_u16(cmp2_2, cmp2_3, 8);

        cmp0_0 = vandq_u16(cmp0_0, offset);
        cmp0_2 = vandq_u16(cmp0_2, offset);
        cmp1_0 = vandq_u16(cmp1_0, offset);
        cmp1_2 = vandq_u16(cmp1_2, offset);
        cmp2_0 = vandq_u16(cmp2_0, offset);
        cmp2_2 = vandq_u16(cmp2_2, offset);
        lt_mask[0] = vaddvq_u16(cmp0_0) | (((uint32_t)vaddvq_u16(cmp0_2)) << 16);
        lt_mask[1] = vaddvq_u16(cmp1_0) | (((uint32_t)vaddvq_u16(cmp1_2)) << 16);
        lt_mask[2] = vaddvq_u16(cmp2_0) | (((uint32_t)vaddvq_u16(cmp2_2)) << 16);

        if (lt_mask[0]) {
            vst1q_u16(distance, result[0]);
            vst1q_u16(distance + 8, result[1]);
            vst1q_u16(distance + 16, result[2]);
            vst1q_u16(distance + 24, result[3]);
        }
        if (lt_mask[1]) {
            vst1q_u16(distance + 32, result[4]);
            vst1q_u16(distance + 40, result[5]);
            vst1q_u16(distance + 48, result[6]);
            vst1q_u16(distance + 56, result[7]);
        }
        if (lt_mask[2]) {
            vst1q_u16(distance + 64, result[8]);
            vst1q_u16(distance + 72, result[9]);
            vst1q_u16(distance + 80, result[10]);
            vst1q_u16(distance + 88, result[11]);
        }
    } else {
        const uint16x8_t threshold_simd = vdupq_n_u16(threshold);
        constexpr uint16x8_t offset = FILTER_OFFSET;
        uint16x8_t cmp0_0 = vcltq_u16(result[0], threshold_simd);
        uint16x8_t cmp0_1 = vcltq_u16(result[1], threshold_simd);
        uint16x8_t cmp0_2 = vcltq_u16(result[2], threshold_simd);
        uint16x8_t cmp0_3 = vcltq_u16(result[3], threshold_simd);
        uint16x8_t cmp1_0 = vcltq_u16(result[4], threshold_simd);
        uint16x8_t cmp1_1 = vcltq_u16(result[5], threshold_simd);
        uint16x8_t cmp1_2 = vcltq_u16(result[6], threshold_simd);
        uint16x8_t cmp1_3 = vcltq_u16(result[7], threshold_simd);
        uint16x8_t cmp2_0 = vcltq_u16(result[8], threshold_simd);
        uint16x8_t cmp2_1 = vcltq_u16(result[9], threshold_simd);
        uint16x8_t cmp2_2 = vcltq_u16(result[10], threshold_simd);
        uint16x8_t cmp2_3 = vcltq_u16(result[11], threshold_simd);

        cmp0_0 = vsliq_n_u16(cmp0_0, cmp0_1, 8);
        cmp0_2 = vsliq_n_u16(cmp0_2, cmp0_3, 8);
        cmp1_0 = vsliq_n_u16(cmp1_0, cmp1_1, 8);
        cmp1_2 = vsliq_n_u16(cmp1_2, cmp1_3, 8);
        cmp2_0 = vsliq_n_u16(cmp2_0, cmp2_1, 8);
        cmp2_2 = vsliq_n_u16(cmp2_2, cmp2_3, 8);

        cmp0_0 = vandq_u16(cmp0_0, offset);
        cmp0_2 = vandq_u16(cmp0_2, offset);
        cmp1_0 = vandq_u16(cmp1_0, offset);
        cmp1_2 = vandq_u16(cmp1_2, offset);
        cmp2_0 = vandq_u16(cmp2_0, offset);
        cmp2_2 = vandq_u16(cmp2_2, offset);
        lt_mask[0] = vaddvq_u16(cmp0_0) | (((uint32_t)vaddvq_u16(cmp0_2)) << 16);
        lt_mask[1] = vaddvq_u16(cmp1_0) | (((uint32_t)vaddvq_u16(cmp1_2)) << 16);
        lt_mask[2] = vaddvq_u16(cmp2_0) | (((uint32_t)vaddvq_u16(cmp2_2)) << 16);

        if (lt_mask[0]) {
            vst1q_u16(distance, result[0]);
            vst1q_u16(distance + 8, result[1]);
            vst1q_u16(distance + 16, result[2]);
            vst1q_u16(distance + 24, result[3]);
        }
        if (lt_mask[1]) {
            vst1q_u16(distance + 32, result[4]);
            vst1q_u16(distance + 40, result[5]);
            vst1q_u16(distance + 48, result[6]);
            vst1q_u16(distance + 56, result[7]);
        }
        if (lt_mask[2]) {
            vst1q_u16(distance + 64, result[8]);
            vst1q_u16(distance + 72, result[9]);
            vst1q_u16(distance + 80, result[10]);
            vst1q_u16(distance + 88, result[11]);
        }
    }
}

/*
 * @brief Perform fast L2 table lookup and filtering operations for single query with batch size 32.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=32).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 32.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 1.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
int krl_L2_table_lookup_fast_scan_bs32(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs32<1>(nsq, codes, LUT, dis, threshold, lt_mask);
    return SUCCESS;
}

/*
 * @brief Perform fast IP table lookup and filtering operations for single query with batch size 32.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=32).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 32.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 1.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
int krl_IP_table_lookup_fast_scan_bs32(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs32<0>(nsq, codes, LUT, dis, threshold, lt_mask);
    return SUCCESS;
}

/*
 * @brief Perform fast L2 table lookup and filtering operations for single query with batch size 64.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=96).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 96.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 3.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
int krl_L2_table_lookup_fast_scan_bs64(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs64<1>(nsq, codes, LUT, dis, threshold, lt_mask);
    return SUCCESS;
}

/*
 * @brief Perform fast IP table lookup and filtering operations for single query with batch size 64.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=96).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param dis Pointer to the array storing computed distances, length 96.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 3.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
int krl_IP_table_lookup_fast_scan_bs64(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs64<0>(nsq, codes, LUT, dis, threshold, lt_mask);
    return SUCCESS;
}

/*
 * @brief Perform fast L2 table lookup and filtering operations for single query with batch size 96.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=96).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param dis Pointer to the array storing computed distances, length 96.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 3.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
int krl_L2_table_lookup_fast_scan_bs96(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs96<1>(nsq, codes, LUT, dis, threshold, lt_mask);
    return SUCCESS;
}

/*
 * @brief Perform fast IP table lookup and filtering operations for single query with batch size 96.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=96).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param dis Pointer to the array storing computed distances, length 96.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 3.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
int krl_IP_table_lookup_fast_scan_bs96(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs96<0>(nsq, codes, LUT, dis, threshold, lt_mask);
    return SUCCESS;
}

/*
 * @brief Extract a column from a matrix and store it in the destination array.
 * @param src Pointer to the source matrix.
 * @param m Number of rows in the source matrix.
 * @param n Number of columns in the source matrix.
 * @param i Row index of the column to extract.
 * @param j Column index of the column to extract.
 * @param dest Pointer to the destination array to store the extracted column.
 * @param length Length of the destination array.
 */
static void krl_get_matrix_column(
    const uint8_t *src, size_t m, size_t n, int64_t i, int64_t j, uint8_t *dest, int length)
{
    for (int64_t k = 0; k < length; k++) {
        if (k + i >= 0 && k + i < m) {
            dest[k] = src[(k + i) * n + j];
        } else {
            dest[k] = 0;
        }
    }
}

/*
 * @brief Pack PQFS codes into blocks.
 * @param codes Pointer to the source codes array.
 * @param ntotal Total number of codes.
 * @param M Number of subquantizers.
 * @param blocks Pointer to the destination blocks array.
 * @param batchsize Batch size for packing.
 * @param blocks_size Length of blocks.
 */
static int krl_pqfs_pack_codes(
    const uint8_t *codes, size_t ntotal, size_t M, uint8_t *blocks, size_t batchsize, size_t blocks_size)
{
    const int nsq = M + M % 2;
    const size_t block1 = ((ntotal + batchsize - 1) / batchsize);
    int ret = SafeMemory::CheckAndMemset(blocks, blocks_size, 0, block1 * batchsize * nsq / 2);
    for (size_t b = 0; b < block1; b++) {
        uint8_t *codes2 = blocks + b * batchsize * nsq / 2;
        const int64_t i_base = b * batchsize;
        for (int sq = 0; sq < nsq; sq += 2) {
            uint8_t c[batchsize], c0[batchsize], c1[batchsize];
            krl_get_matrix_column(codes, ntotal, nsq / 2, i_base, sq / 2, c, batchsize);
            for (int j = 0; j < batchsize; j++) {
                c0[j] = c[j] & 15; /* base vector dim0 */
                c1[j] = c[j] >> 4; /* base vector dim1 */
            }
            for (int j = 0; j < batchsize; j += 16) {
                for (int k = 0; k < 8; ++k) {                      /* vector id */
                    uint8_t d0 = c0[k + j] | (c0[k + j + 8] << 4); /* dim0: 0,8 1,9 2,10 ... */
                    uint8_t d1 = c1[k + j] | (c1[k + j + 8] << 4); /* dim1: 0,8 1,9 2,10 ... */
                    codes2[j + (k << 1)] |= d0;
                    codes2[j + (k << 1) + 1] |= d1; /* dim0，dim1，dim0，dim1，... */
                }
            }
            codes2 += batchsize;
        }
    }
    return SUCCESS;
}

/*
 * @brief Pack IVFPQ codes into blocks with 4-bit precision.
 * @param codes Pointer to the source codes array.
 * @param ntotal Total number of codes.
 * @param nsq Number of subquantizers.
 * @param blocks Pointer to the destination blocks array.
 * @param blocksize Block size for packing.
 */
static int krl_IVFPQ_code_packer_4b(
    const uint8_t *codes, size_t ntotal, size_t nsq, uint8_t *blocks, size_t blocksize, size_t blocks_size)
{
    if (ntotal == 0) {
        return INVALPARAM;
    }
    const size_t nb = ((ntotal + blocksize - 1) & (-blocksize));
    size_t ceil_nsq = nsq + nsq % 2;
    int ret = SafeMemory::CheckAndMemset(blocks, blocks_size, 0, nb * ceil_nsq / 2);
    const uint8_t convertdict[32] = {
		0,
        8,
        1,
        9,
        2,
        10,
        3,
        11,
        4,
        12,
        5,
        13,
        6,
        14,
        7,
        15,
        32,
        40,
        33,
        41,
        34,
        42,
        35,
        43,
        36,
        44,
        37,
        45,
        38,
        46,
        39,
        47
		};
    const uint8_t offset = 16;
    uint8_t *codes2 = blocks;
    uint8_t c[blocksize], c0[blocksize], c1[blocksize];
    for (size_t i0 = 0; i0 < nb; i0 += blocksize) {
        for (size_t sq = 0; sq < ceil_nsq; sq += 2) {
            krl_get_matrix_column(codes, ntotal, ceil_nsq / 2, i0, sq / 2, c, blocksize);
            for (int j = 0; j < blocksize; j++) {
                c0[j] = c[j] & 15; /* dim : 0, bid: 0,1,2,3,... */
                c1[j] = c[j] >> 4; /* dim : 1, bid: 0,1,2,3,... */
            }
            for (int k = 0; k < blocksize; k += 64) {
                for (int j = 0; j < 32; j++) {
                    uint8_t d0, d1;
                    uint8_t base_id = convertdict[j] + k;
                    uint8_t base_offset = j + (k >> 1);
                    d0 = c0[base_id] | (c0[base_id + offset] << 4);
                    d1 = c1[base_id] | (c1[base_id + offset] << 4);
                    codes2[base_offset] = d0;
                    codes2[base_offset + (blocksize >> 1)] = d1;
                }
            }
            codes2 += blocksize;
        }
    }
    return SUCCESS;
}

/*
 * @brief Pack codes into blocks with 4-bit precision.
 * @param codes Pointer to the source codes array.
 * @param ncode Total number of codes.
 * @param nsq Number of subquantizers.
 * @param blocks Pointer to the destination blocks array.
 * @param batchsize Batch size for packing.
 * @param dim_cross Dimension cross flag (0 for IVFPQ, non-zero for PQFS).
 * @param codes_size Length of codes.
 * @param blocks_size Length of blocks.
 */
int krl_pack_codes_4b(const uint8_t *codes, size_t ncode, size_t nsq, uint8_t *blocks, size_t batchsize, int dim_cross,
    size_t codes_size, size_t blocks_size)
{
    int ret = 0;
    if (dim_cross == 0) {
        ret = krl_IVFPQ_code_packer_4b(codes, ncode, nsq, blocks, batchsize, blocks_size);
    } else {
        ret = krl_pqfs_pack_codes(codes, ncode, nsq, blocks, batchsize, blocks_size);
    }
    return ret;
}

/*
 * @brief Unpack 4-bit PQFS packed codes into plain code layout.
 *
 * This function unpacks 4-bit packed codes produced by PQFS packing.
 * Each byte contains two 4-bit subquantizer codes. The unpacking
 * performs a fixed permutation and reorganizes codes back into
 * subquantizer-major layout.
 *
 * @param codes Pointer to the source packed codes.
 * @param ncodes Total number of vectors to unpack.
 * @param M Number of subquantizers.
 * @param blocks Pointer to the destination unpacked code buffer.
 * @param block_codes Offset (in vectors) for writing unpacked codes.
 * @param batchsize Batch size used during packing.
 */
int krl_pqfs_unpack_codes(
    const uint8_t *codes,
    size_t ncodes,
    size_t M,
    uint8_t *blocks,
    size_t block_codes,
    size_t batchsize) {
    const uint8_t perm0[16] = {
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const int nsq = M + M % 2;
    const int half_nsq = nsq >> 1;
    const int blocks_offset = block_codes * half_nsq;

    size_t nb = ncodes / batchsize;
    const size_t codes_tail = ncodes % batchsize;
    for (size_t i0 = 0; i0 < nb; i0++) {
        const uint8_t *codes1 = codes + i0 * batchsize * half_nsq;
        uint8_t *codes2 = blocks + blocks_offset + i0 * batchsize * half_nsq;
        uint8_t c0[32], c1[32];
        for (int sq = 0; sq < half_nsq; sq++) {
            for (size_t i = 0; i < batchsize; i += 32) {
                for (int j = 0; j < 32; j++) {
                    c0[j] = codes1[sq * batchsize + i + j] & 15;
                    c1[j] = codes1[sq * batchsize + i + j] >> 4;
                }
                for (int j = 0; j < 16; j++) {
                    uint8_t d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                    uint8_t d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                    codes2[sq + (i + j) * half_nsq] = d0;
                    codes2[sq + (i + j + 16) * half_nsq] = d1;
                }
            }
        }
    }
    if (codes_tail > 0) {
        const uint8_t *codes1 = codes + nb * batchsize * half_nsq;
        uint8_t *codes2 = blocks + blocks_offset + nb * batchsize * half_nsq;
        uint8_t c0[32], c1[32];
        const int tiny_block = codes_tail & (-32);
        const int tiny_block_tail = codes_tail % 32;
        for (int sq = 0; sq < half_nsq; sq++) {
            for (size_t i = 0; i < tiny_block; i += 32) {
                for (int j = 0; j < 32; j++) {
                    c0[j] = codes1[sq * batchsize + i + j] & 15;
                    c1[j] = codes1[sq * batchsize + i + j] >> 4;
                }
                for (int j = 0; j < 16; j++) {
                    uint8_t d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                    uint8_t d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                    codes2[sq + (i + j) * half_nsq] = d0;
                    codes2[sq + (i + j + 16) * half_nsq] = d1;
                }
            }
            {
                for (int j = 0; j < 32; j++) {
                    c0[j] = codes1[sq * batchsize + tiny_block + j] & 15;
                    c1[j] = codes1[sq * batchsize + tiny_block + j] >> 4;
                }
                for (int j = 0; j < 16; j++) {
                    if (j + 16 < tiny_block_tail) {
                        uint8_t d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                        uint8_t d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                        codes2[sq + (tiny_block + j) * half_nsq] = d0;
                        codes2[sq + (tiny_block + j + 16) * half_nsq] = d1;
                    } else if (j < tiny_block_tail) {
                        uint8_t d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                        codes2[sq + (tiny_block + j) * half_nsq] = d0;
                    }
                }
            }
        }
    }
    return SUCCESS;
}

/*
 * @brief Repack 4-bit codes with a different batch size.
 *
 * This function repacks 4-bit quantization codes from a previous
 * batch layout into a new batch layout. For IVFPQ (dim_cross == 0),
 * repacking is done directly. For PQFS, codes are first unpacked
 * into a temporary buffer and then packed again with the new
 * batch size.
 *
 * @param codes Pointer to the source packed codes.
 * @param ncode Total number of vectors.
 * @param nsq Number of subquantizers.
 * @param blocks Pointer to the destination packed blocks.
 * @param batchsize Number of blocks in the destination layout.
 * @param prev_batchsize Batch size used in the source layout.
 * @param after_batchsize Batch size to use in the destination layout.
 * @param dim_cross Dimension cross flag (0 for IVFPQ, non-zero for PQFS).
 */
int krl_repack_codes_4b(
    const uint8_t *codes,
    size_t ncode,
    size_t nsq,
    uint8_t *blocks,
    size_t batchsize,
    size_t prev_batchsize,
    size_t after_batchsize,
    int dim_cross) {
    const size_t half_nsq = (nsq + 1) / 2;
    int ret = 0;
    if (dim_cross == 0) {
        size_t blocks_size = batchsize * half_nsq;
        ret = krl_IVFPQ_code_packer_4b(codes, ncode, nsq, blocks, after_batchsize, blocks_size);
    } else {
        const size_t prev_ntotal = (ncode + prev_batchsize - 1) / prev_batchsize * prev_batchsize;
        const size_t after_ntotal = (ncode + after_batchsize - 1) / after_batchsize * after_batchsize;
        size_t blocks_size = after_ntotal * half_nsq;

        uint8_t *tmp_buffer = (uint8_t *)malloc(prev_ntotal * half_nsq);
        ret = krl_pqfs_unpack_codes(codes, ncode, nsq, tmp_buffer, 0, prev_batchsize);
        ret = krl_pqfs_pack_codes(tmp_buffer, ncode, nsq, blocks, after_batchsize, blocks_size);
        free(tmp_buffer);
    }
    return ret;
}
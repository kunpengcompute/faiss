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
#include "safe_memory.h"
#include <stdio.h>
#include <math.h>
#include <arm_sve.h>

static const uint16_t FILTER_OFFSET_DATA[16] __attribute__((aligned(64))) = {
    0x0001, 0x0002, 0x0004, 0x0008,
    0x0010, 0x0020, 0x0040, 0x0080,
    0x0100, 0x0200, 0x0400, 0x0800,
    0x1000, 0x2000, 0x4000, 0x8000
};

/*
 * @brief store low and high halves of an SVE u16 vector separately.
 * @param lo_dst Pointer to the array storing the low half of the vector.
 * @param hi_dst Pointer to the array storing the high half of the vector.
 * @param v SVE u16 vector.
 */
static inline void store_halves_u16_sve2(uint16_t *lo_dst, uint16_t *hi_dst, svuint16_t v)
{
    svbool_t pg_lo = svwhilelt_b16((uint32_t)0, (uint32_t)8);
    svst1_u16(pg_lo, lo_dst, v);
    svuint16_t v_hi = svext_u16(v, v, 8);
    svst1_u16(pg_lo, hi_dst, v_hi);
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
static inline void krl_table_lookup_fast_scan_bs32_sve2(
    int nsq, const uint8_t *codes, const uint8_t *lut, uint16_t *distance, uint16_t threshold, uint32_t *lt_mask)
{
    const svbool_t pg8 = svptrue_b8();
    const svbool_t pg16 = svptrue_b16();

    svuint16_t res0_lo = svdup_n_u16(0);
    svuint16_t res0_hi = svdup_n_u16(0);

    svuint8_t code0 = svld1_u8(pg8, codes);
    codes += 32;

    /* SVE2 sli/sri constants */
    const svuint8_t const_01 = svreinterpret_u8_u16(svdup_n_u16(0x0100));
    const svuint8_t const_10 = svreinterpret_u8_u16(svdup_n_u16(0x1000));

    svuint8_t dict = svld1_u8(pg8, lut);
    lut += 32;

    /* First iteration with software pipeline start */
    if (__builtin_expect(nsq > 2, 1)) {
        svuint8_t idx0_lo = svsli_n_u8(code0, const_01, 4);
        svuint8_t idx0_hi = svsri_n_u8(const_10, code0, 4);
        code0 = svld1_u8(pg8, codes);
        codes += 32;

        idx0_lo = svtbl_u8(dict, idx0_lo);
        idx0_hi = svtbl_u8(dict, idx0_hi);

        dict = svld1_u8(pg8, lut);
        lut += 32;

        /* SVE2: pairwise add-accumulate */
        res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
        res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);

        /* Main loop */
        for (int sq = 4; sq < nsq; sq += 2) {
            idx0_lo = svsli_n_u8(code0, const_01, 4);
            idx0_hi = svsri_n_u8(const_10, code0, 4);
            code0 = svld1_u8(pg8, codes);
            codes += 32;

            idx0_lo = svtbl_u8(dict, idx0_lo);
            idx0_hi = svtbl_u8(dict, idx0_hi);

            dict = svld1_u8(pg8, lut);
            lut += 32;

            res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
            res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);
        }
    }

    /* Last iteration without preload */
    {
        svuint8_t idx0_lo = svsli_n_u8(code0, const_01, 4);
        svuint8_t idx0_hi = svsri_n_u8(const_10, code0, 4);

        idx0_lo = svtbl_u8(dict, idx0_lo);
        idx0_hi = svtbl_u8(dict, idx0_hi);

        res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
        res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);
    }

    /* Threshold comparison - SVE2 vectorized */
    {
        /* Pre-store for comparison path; final visibility follows NEON's lt_mask-gated behavior. */
        store_halves_u16_sve2(distance, distance + 16, res0_lo);
        store_halves_u16_sve2(distance + 8, distance + 24, res0_hi);

        /* Reload distances as SVE vectors in NEON order */
        svuint16_t d0 = svld1_u16(pg16, distance);      /* d[0..15] */
        svuint16_t d1 = svld1_u16(pg16, distance + 16); /* d[16..31] */

        svuint16_t thresh_vec = svdup_n_u16(threshold);
        svuint16_t bit_offsets = svld1_u16(pg16, FILTER_OFFSET_DATA);
        svuint16_t zero = svdup_n_u16(0);

        svbool_t cmp0, cmp1;
        if constexpr (keep_min == 0) {
            // IP mode: keep if distance > threshold (svcmpgt)
            cmp0 = svcmpgt_u16(pg16, d0, thresh_vec);
            cmp1 = svcmpgt_u16(pg16, d1, thresh_vec);
        } else {
            // L2 mode: keep if distance < threshold (svcmplt)
            cmp0 = svcmplt_u16(pg16, d0, thresh_vec);
            cmp1 = svcmplt_u16(pg16, d1, thresh_vec);
        }

        uint64_t bits0 = svaddv_u16(pg16, svsel_u16(cmp0, bit_offsets, zero));
        uint64_t bits1 = svaddv_u16(pg16, svsel_u16(cmp1, bit_offsets, zero));

        lt_mask[0] = (uint32_t)bits0 | ((uint32_t)bits1 << 16);
    }
}

/*
 * @brief Perform fast table lookup and filtering operations for single query with batch size 64.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=64).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param distance Pointer to the array storing computed distances, length 64.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 2.
 */
template <int keep_min = 0>
static inline void krl_table_lookup_fast_scan_bs64_sve2(
    int nsq, const uint8_t *codes, const uint8_t *lut, uint16_t *distance, uint16_t threshold, uint32_t *lt_mask)
{
    const svbool_t pg8 = svptrue_b8();
    const svbool_t pg16 = svptrue_b16();

    svuint16_t res0_lo = svdup_n_u16(0);
    svuint16_t res0_hi = svdup_n_u16(0);
    svuint16_t res1_lo = svdup_n_u16(0);
    svuint16_t res1_hi = svdup_n_u16(0);

    svuint8_t code0 = svld1_u8(pg8, codes);
    svuint8_t code1 = svld1_u8(pg8, codes + 32);
    codes += 64;

    /*
     * SVE2 sli/sri constants (same as NEON):
     *   const_01 = reinterpret_u8(u16(0x0100)) = {0x00, 0x01, 0x00, 0x01, ...}
     *   const_10 = reinterpret_u8(u16(0x1000)) = {0x00, 0x10, 0x00, 0x10, ...}
     */
    const svuint8_t const_01 = svreinterpret_u8_u16(svdup_n_u16(0x0100));
    const svuint8_t const_10 = svreinterpret_u8_u16(svdup_n_u16(0x1000));

    svuint8_t dict = svld1_u8(pg8, lut);
    lut += 32;

    /* First iteration with software pipeline start */
    if (__builtin_expect(nsq > 2, 1)) {
        /* SVE2: sli(code, const_01, 4) = keep low 4 bits of code, insert const_01<<4 into high bits */
        svuint8_t idx0_lo = svsli_n_u8(code0, const_01, 4);
        /* SVE2: sri(const_10, code, 4) = keep high 4 bits of const_10, insert code>>4 into low bits */
        svuint8_t idx0_hi = svsri_n_u8(const_10, code0, 4);
        code0 = svld1_u8(pg8, codes);

        svuint8_t idx1_lo = svsli_n_u8(code1, const_01, 4);
        svuint8_t idx1_hi = svsri_n_u8(const_10, code1, 4);
        code1 = svld1_u8(pg8, codes + 32);
        codes += 64;

        idx0_lo = svtbl_u8(dict, idx0_lo);
        idx0_hi = svtbl_u8(dict, idx0_hi);
        idx1_lo = svtbl_u8(dict, idx1_lo);
        idx1_hi = svtbl_u8(dict, idx1_hi);

        dict = svld1_u8(pg8, lut);
        lut += 32;

        /* SVE2: pairwise add-accumulate */
        res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
        res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);
        res1_lo = svadalp_u16_x(pg16, res1_lo, idx1_lo);
        res1_hi = svadalp_u16_x(pg16, res1_hi, idx1_hi);

        /* Main loop */
        for (int sq = 4; sq < nsq; sq += 2) {
            idx0_lo = svsli_n_u8(code0, const_01, 4);
            idx0_hi = svsri_n_u8(const_10, code0, 4);
            code0 = svld1_u8(pg8, codes);

            idx1_lo = svsli_n_u8(code1, const_01, 4);
            idx1_hi = svsri_n_u8(const_10, code1, 4);
            code1 = svld1_u8(pg8, codes + 32);
            codes += 64;

            idx0_lo = svtbl_u8(dict, idx0_lo);
            idx0_hi = svtbl_u8(dict, idx0_hi);
            idx1_lo = svtbl_u8(dict, idx1_lo);
            idx1_hi = svtbl_u8(dict, idx1_hi);

            dict = svld1_u8(pg8, lut);
            lut += 32;

            res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
            res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);
            res1_lo = svadalp_u16_x(pg16, res1_lo, idx1_lo);
            res1_hi = svadalp_u16_x(pg16, res1_hi, idx1_hi);
        }
    }

    /* Last iteration without preload */
    {
        svuint8_t idx0_lo = svsli_n_u8(code0, const_01, 4);
        svuint8_t idx0_hi = svsri_n_u8(const_10, code0, 4);
        svuint8_t idx1_lo = svsli_n_u8(code1, const_01, 4);
        svuint8_t idx1_hi = svsri_n_u8(const_10, code1, 4);

        idx0_lo = svtbl_u8(dict, idx0_lo);
        idx0_hi = svtbl_u8(dict, idx0_hi);
        idx1_lo = svtbl_u8(dict, idx1_lo);
        idx1_hi = svtbl_u8(dict, idx1_hi);

        res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
        res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);
        res1_lo = svadalp_u16_x(pg16, res1_lo, idx1_lo);
        res1_hi = svadalp_u16_x(pg16, res1_hi, idx1_hi);
    }

    /* Threshold comparison - SVE2 vectorized */
    {
        /* Pre-store for comparison path; final visibility follows NEON's lt_mask-gated behavior. */
        store_halves_u16_sve2(distance, distance + 16, res0_lo);
        store_halves_u16_sve2(distance + 8, distance + 24, res0_hi);
        store_halves_u16_sve2(distance + 32, distance + 48, res1_lo);
        store_halves_u16_sve2(distance + 40, distance + 56, res1_hi);

        /* Reload distances as SVE vectors in NEON order */
        svuint16_t d0 = svld1_u16(pg16, distance);      /* d[0..15] */
        svuint16_t d1 = svld1_u16(pg16, distance + 16); /* d[16..31] */
        svuint16_t d2 = svld1_u16(pg16, distance + 32); /* d[32..47] */
        svuint16_t d3 = svld1_u16(pg16, distance + 48); /* d[48..63] */

        svuint16_t thresh_vec = svdup_n_u16(threshold);
        svuint16_t bit_offsets = svld1_u16(pg16, FILTER_OFFSET_DATA);
        svuint16_t zero = svdup_n_u16(0);

        svbool_t cmp0, cmp1, cmp2, cmp3;
        if constexpr (keep_min == 0) {
            // IP mode: keep if distance > threshold (svcmpgt)
            cmp0 = svcmpgt_u16(pg16, d0, thresh_vec);
            cmp1 = svcmpgt_u16(pg16, d1, thresh_vec);
            cmp2 = svcmpgt_u16(pg16, d2, thresh_vec);
            cmp3 = svcmpgt_u16(pg16, d3, thresh_vec);
        } else {
            // L2 mode: keep if distance < threshold (svcmplt)
            cmp0 = svcmplt_u16(pg16, d0, thresh_vec);
            cmp1 = svcmplt_u16(pg16, d1, thresh_vec);
            cmp2 = svcmplt_u16(pg16, d2, thresh_vec);
            cmp3 = svcmplt_u16(pg16, d3, thresh_vec);
        }

        uint64_t bits0 = svaddv_u16(pg16, svsel_u16(cmp0, bit_offsets, zero));
        uint64_t bits1 = svaddv_u16(pg16, svsel_u16(cmp1, bit_offsets, zero));
        uint64_t bits2 = svaddv_u16(pg16, svsel_u16(cmp2, bit_offsets, zero));
        uint64_t bits3 = svaddv_u16(pg16, svsel_u16(cmp3, bit_offsets, zero));

        lt_mask[0] = (uint32_t)bits0 | ((uint32_t)bits1 << 16);
        lt_mask[1] = (uint32_t)bits2 | ((uint32_t)bits3 << 16);
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
static inline void krl_table_lookup_fast_scan_bs96_sve2(
    int nsq, const uint8_t *codes, const uint8_t *lut, uint16_t *distance, uint16_t threshold, uint32_t *lt_mask)
{
    const svbool_t pg8 = svptrue_b8();
    const svbool_t pg16 = svptrue_b16();

    svuint16_t res0_lo = svdup_n_u16(0);
    svuint16_t res0_hi = svdup_n_u16(0);
    svuint16_t res1_lo = svdup_n_u16(0);
    svuint16_t res1_hi = svdup_n_u16(0);
    svuint16_t res2_lo = svdup_n_u16(0);
    svuint16_t res2_hi = svdup_n_u16(0);

    svuint8_t code0 = svld1_u8(pg8, codes);
    svuint8_t code1 = svld1_u8(pg8, codes + 32);
    svuint8_t code2 = svld1_u8(pg8, codes + 64);
    codes += 96;

    /* SVE2 sli/sri constants */
    const svuint8_t const_01 = svreinterpret_u8_u16(svdup_n_u16(0x0100));
    const svuint8_t const_10 = svreinterpret_u8_u16(svdup_n_u16(0x1000));

    svuint8_t dict = svld1_u8(pg8, lut);
    lut += 32;

    /* First iteration with software pipeline start */
    if (__builtin_expect(nsq > 2, 1)) {
        svuint8_t idx0_lo = svsli_n_u8(code0, const_01, 4);
        svuint8_t idx0_hi = svsri_n_u8(const_10, code0, 4);
        code0 = svld1_u8(pg8, codes);

        svuint8_t idx1_lo = svsli_n_u8(code1, const_01, 4);
        svuint8_t idx1_hi = svsri_n_u8(const_10, code1, 4);
        code1 = svld1_u8(pg8, codes + 32);

        svuint8_t idx2_lo = svsli_n_u8(code2, const_01, 4);
        svuint8_t idx2_hi = svsri_n_u8(const_10, code2, 4);
        code2 = svld1_u8(pg8, codes + 64);
        codes += 96;

        idx0_lo = svtbl_u8(dict, idx0_lo);
        idx0_hi = svtbl_u8(dict, idx0_hi);
        idx1_lo = svtbl_u8(dict, idx1_lo);
        idx1_hi = svtbl_u8(dict, idx1_hi);
        idx2_lo = svtbl_u8(dict, idx2_lo);
        idx2_hi = svtbl_u8(dict, idx2_hi);

        dict = svld1_u8(pg8, lut);
        lut += 32;

        /* SVE2: pairwise add-accumulate */
        res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
        res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);
        res1_lo = svadalp_u16_x(pg16, res1_lo, idx1_lo);
        res1_hi = svadalp_u16_x(pg16, res1_hi, idx1_hi);
        res2_lo = svadalp_u16_x(pg16, res2_lo, idx2_lo);
        res2_hi = svadalp_u16_x(pg16, res2_hi, idx2_hi);

        /* Main loop */
        for (int sq = 4; sq < nsq; sq += 2) {
            idx0_lo = svsli_n_u8(code0, const_01, 4);
            idx0_hi = svsri_n_u8(const_10, code0, 4);
            code0 = svld1_u8(pg8, codes);

            idx1_lo = svsli_n_u8(code1, const_01, 4);
            idx1_hi = svsri_n_u8(const_10, code1, 4);
            code1 = svld1_u8(pg8, codes + 32);

            idx2_lo = svsli_n_u8(code2, const_01, 4);
            idx2_hi = svsri_n_u8(const_10, code2, 4);
            code2 = svld1_u8(pg8, codes + 64);
            codes += 96;

            idx0_lo = svtbl_u8(dict, idx0_lo);
            idx0_hi = svtbl_u8(dict, idx0_hi);
            idx1_lo = svtbl_u8(dict, idx1_lo);
            idx1_hi = svtbl_u8(dict, idx1_hi);
            idx2_lo = svtbl_u8(dict, idx2_lo);
            idx2_hi = svtbl_u8(dict, idx2_hi);

            dict = svld1_u8(pg8, lut);
            lut += 32;

            res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
            res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);
            res1_lo = svadalp_u16_x(pg16, res1_lo, idx1_lo);
            res1_hi = svadalp_u16_x(pg16, res1_hi, idx1_hi);
            res2_lo = svadalp_u16_x(pg16, res2_lo, idx2_lo);
            res2_hi = svadalp_u16_x(pg16, res2_hi, idx2_hi);
        }
    }

    /* Last iteration without preload */
    {
        svuint8_t idx0_lo = svsli_n_u8(code0, const_01, 4);
        svuint8_t idx0_hi = svsri_n_u8(const_10, code0, 4);
        svuint8_t idx1_lo = svsli_n_u8(code1, const_01, 4);
        svuint8_t idx1_hi = svsri_n_u8(const_10, code1, 4);
        svuint8_t idx2_lo = svsli_n_u8(code2, const_01, 4);
        svuint8_t idx2_hi = svsri_n_u8(const_10, code2, 4);

        idx0_lo = svtbl_u8(dict, idx0_lo);
        idx0_hi = svtbl_u8(dict, idx0_hi);
        idx1_lo = svtbl_u8(dict, idx1_lo);
        idx1_hi = svtbl_u8(dict, idx1_hi);
        idx2_lo = svtbl_u8(dict, idx2_lo);
        idx2_hi = svtbl_u8(dict, idx2_hi);

        res0_lo = svadalp_u16_x(pg16, res0_lo, idx0_lo);
        res0_hi = svadalp_u16_x(pg16, res0_hi, idx0_hi);
        res1_lo = svadalp_u16_x(pg16, res1_lo, idx1_lo);
        res1_hi = svadalp_u16_x(pg16, res1_hi, idx1_hi);
        res2_lo = svadalp_u16_x(pg16, res2_lo, idx2_lo);
        res2_hi = svadalp_u16_x(pg16, res2_hi, idx2_hi);
    }

    /* Threshold comparison - SVE2 vectorized */
    {
        /* Pre-store for comparison path; final visibility follows NEON's lt_mask-gated behavior. */
        store_halves_u16_sve2(distance, distance + 16, res0_lo);
        store_halves_u16_sve2(distance + 8, distance + 24, res0_hi);
        store_halves_u16_sve2(distance + 32, distance + 48, res1_lo);
        store_halves_u16_sve2(distance + 40, distance + 56, res1_hi);
        store_halves_u16_sve2(distance + 64, distance + 80, res2_lo);
        store_halves_u16_sve2(distance + 72, distance + 88, res2_hi);

        /* Reload distances as SVE vectors in NEON order */
        svuint16_t d0 = svld1_u16(pg16, distance);      /* d[0..15] */
        svuint16_t d1 = svld1_u16(pg16, distance + 16); /* d[16..31] */
        svuint16_t d2 = svld1_u16(pg16, distance + 32); /* d[32..47] */
        svuint16_t d3 = svld1_u16(pg16, distance + 48); /* d[48..63] */
        svuint16_t d4 = svld1_u16(pg16, distance + 64); /* d[64..79] */
        svuint16_t d5 = svld1_u16(pg16, distance + 80); /* d[80..95] */

        svuint16_t thresh_vec = svdup_n_u16(threshold);
        svuint16_t bit_offsets = svld1_u16(pg16, FILTER_OFFSET_DATA);
        svuint16_t zero = svdup_n_u16(0);

        svbool_t cmp0, cmp1, cmp2, cmp3, cmp4, cmp5;
        if constexpr (keep_min == 0) {
            // IP mode: keep if distance > threshold (svcmpgt)
            cmp0 = svcmpgt_u16(pg16, d0, thresh_vec);
            cmp1 = svcmpgt_u16(pg16, d1, thresh_vec);
            cmp2 = svcmpgt_u16(pg16, d2, thresh_vec);
            cmp3 = svcmpgt_u16(pg16, d3, thresh_vec);
            cmp4 = svcmpgt_u16(pg16, d4, thresh_vec);
            cmp5 = svcmpgt_u16(pg16, d5, thresh_vec);
        } else {
            // L2 mode: keep if distance < threshold (svcmplt)
            cmp0 = svcmplt_u16(pg16, d0, thresh_vec);
            cmp1 = svcmplt_u16(pg16, d1, thresh_vec);
            cmp2 = svcmplt_u16(pg16, d2, thresh_vec);
            cmp3 = svcmplt_u16(pg16, d3, thresh_vec);
            cmp4 = svcmplt_u16(pg16, d4, thresh_vec);
            cmp5 = svcmplt_u16(pg16, d5, thresh_vec);
        }

        uint64_t bits0 = svaddv_u16(pg16, svsel_u16(cmp0, bit_offsets, zero));
        uint64_t bits1 = svaddv_u16(pg16, svsel_u16(cmp1, bit_offsets, zero));
        uint64_t bits2 = svaddv_u16(pg16, svsel_u16(cmp2, bit_offsets, zero));
        uint64_t bits3 = svaddv_u16(pg16, svsel_u16(cmp3, bit_offsets, zero));
        uint64_t bits4 = svaddv_u16(pg16, svsel_u16(cmp4, bit_offsets, zero));
        uint64_t bits5 = svaddv_u16(pg16, svsel_u16(cmp5, bit_offsets, zero));

        lt_mask[0] = (uint32_t)bits0 | ((uint32_t)bits1 << 16);
        lt_mask[1] = (uint32_t)bits2 | ((uint32_t)bits3 << 16);
        lt_mask[2] = (uint32_t)bits4 | ((uint32_t)bits5 << 16);
    }
}

/*
 * @brief Perform fast IP table lookup and filtering operations for single query with batch size 32.
 * @param nsq Number of subquantizers.
 * @param codes Pointer to the codes array, layout (nsq, batch=32).
 * @param LUT Pointer to the precomputed distances array, layout (nsq, ksub=16).
 * @param dis Pointer to the array storing computed distances, length 32.
 * @param threshold Filter threshold value.
 * @param lt_mask Pointer to the array storing filter results, length 1.
 * @param codes_size Length of codes.
 * @param LUT_size Length of LUT.
 * @param lt_mask_size Length of lt_mask.
 */
int krl_IP_table_lookup_fast_scan_bs32_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs32_sve2<0>(nsq, codes, LUT, dis, threshold, lt_mask);
    return SUCCESS;
}

/*
 * @brief Perform fast L2 table lookup and filtering operations for single query with batch size 32`.
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
int krl_L2_table_lookup_fast_scan_bs32_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs32_sve2<1>(nsq, codes, LUT, dis, threshold, lt_mask);
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
int krl_IP_table_lookup_fast_scan_bs64_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs64_sve2<0>(nsq, codes, LUT, dis, threshold, lt_mask);
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
int krl_L2_table_lookup_fast_scan_bs64_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs64_sve2<1>(nsq, codes, LUT, dis, threshold, lt_mask);
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
int krl_IP_table_lookup_fast_scan_bs96_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs96_sve2<0>(nsq, codes, LUT, dis, threshold, lt_mask);
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
int krl_L2_table_lookup_fast_scan_bs96_sve2(int nsq, const uint8_t *codes, const uint8_t *LUT, uint16_t *dis,
    uint16_t threshold, uint32_t *lt_mask, size_t codes_size, size_t LUT_size, size_t lt_mask_size)
{
    krl_table_lookup_fast_scan_bs96_sve2<1>(nsq, codes, LUT, dis, threshold, lt_mask);
    return SUCCESS;
}


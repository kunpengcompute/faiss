/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstddef>
#include <arm_neon.h>
#include <faiss/utils/arm/asm/distances_simd.h>

namespace faiss {

#if defined(__GNUC__) && !defined(__clang__)

__attribute__((always_inline)) static inline void prfm_r(const void *p) noexcept
{
    asm volatile("prfm pldl1keep, [%0, #320]\n" ::"r"(p));
}

__attribute__((always_inline)) static inline void prfm_w(const void *p) noexcept
{
    asm volatile("prfm pstl1keep, [%0]\n" ::"r"(p));
}

static inline void ldp_q_f32(const float *&p, float32x4_t &a, float32x4_t &b)
{
    float32x4x2_t t = vld1q_f32_x2(p);
    p += 8;
    a = t.val[0];
    b = t.val[1];
}

__attribute__((always_inline)) static inline void l2sqr_step8(const float *&p, const float32x4_t x0,
                                                              const float32x4_t x1, float32x4_t &acc) noexcept
{
    float32x4_t y0;
    float32x4_t y1;
    ldp_q_f32(p, y0, y1);
    const float32x4_t d0 = vsubq_f32(y0, x0);
    const float32x4_t d1 = vsubq_f32(y1, x1);
    acc = vfmaq_f32(acc, d0, d0);
    acc = vfmaq_f32(acc, d1, d1);
}

static void L2sqrBatch24(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    const float *px = x;

    const float *p0 = y + 0 * d;
    const float *p1 = y + 1 * d;
    const float *p2 = y + 2 * d;
    const float *p3 = y + 3 * d;
    const float *p4 = y + 4 * d;
    const float *p5 = y + 5 * d;
    const float *p6 = y + 6 * d;
    const float *p7 = y + 7 * d;
    const float *p8 = y + 8 * d;
    const float *p9 = y + 9 * d;
    const float *p10 = y + 10 * d;
    const float *p11 = y + 11 * d;
    const float *p12 = y + 12 * d;
    const float *p13 = y + 13 * d;
    const float *p14 = y + 14 * d;
    const float *p15 = y + 15 * d;
    const float *p16 = y + 16 * d;
    const float *p17 = y + 17 * d;
    const float *p18 = y + 18 * d;
    const float *p19 = y + 19 * d;
    const float *p20 = y + 20 * d;
    const float *p21 = y + 21 * d;
    const float *p22 = y + 22 * d;
    const float *p23 = y + 23 * d;

    float32x4_t a0 = vdupq_n_f32(0);
    float32x4_t a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0);
    float32x4_t a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0);
    float32x4_t a5 = vdupq_n_f32(0);
    float32x4_t a6 = vdupq_n_f32(0);
    float32x4_t a7 = vdupq_n_f32(0);
    float32x4_t a8 = vdupq_n_f32(0);
    float32x4_t a9 = vdupq_n_f32(0);
    float32x4_t a10 = vdupq_n_f32(0);
    float32x4_t a11 = vdupq_n_f32(0);
    float32x4_t a12 = vdupq_n_f32(0);
    float32x4_t a13 = vdupq_n_f32(0);
    float32x4_t a14 = vdupq_n_f32(0);
    float32x4_t a15 = vdupq_n_f32(0);
    float32x4_t a16 = vdupq_n_f32(0);
    float32x4_t a17 = vdupq_n_f32(0);
    float32x4_t a18 = vdupq_n_f32(0);
    float32x4_t a19 = vdupq_n_f32(0);
    float32x4_t a20 = vdupq_n_f32(0);
    float32x4_t a21 = vdupq_n_f32(0);
    float32x4_t a22 = vdupq_n_f32(0);
    float32x4_t a23 = vdupq_n_f32(0);

    size_t i = 0;
    size_t single_round = 8;

    for (; i + single_round <= d; i += single_round) {
        prfm_r(px);

        float32x4_t x0;
        float32x4_t x1;
        ldp_q_f32(px, x0, x1);

        l2sqr_step8(p0, x0, x1, a0);
        l2sqr_step8(p1, x0, x1, a1);
        l2sqr_step8(p2, x0, x1, a2);
        l2sqr_step8(p3, x0, x1, a3);
        l2sqr_step8(p4, x0, x1, a4);
        l2sqr_step8(p5, x0, x1, a5);
        l2sqr_step8(p6, x0, x1, a6);
        l2sqr_step8(p7, x0, x1, a7);
        l2sqr_step8(p8, x0, x1, a8);
        l2sqr_step8(p9, x0, x1, a9);
        l2sqr_step8(p10, x0, x1, a10);
        l2sqr_step8(p11, x0, x1, a11);
        l2sqr_step8(p12, x0, x1, a12);
        l2sqr_step8(p13, x0, x1, a13);
        l2sqr_step8(p14, x0, x1, a14);
        l2sqr_step8(p15, x0, x1, a15);
        l2sqr_step8(p16, x0, x1, a16);
        l2sqr_step8(p17, x0, x1, a17);
        l2sqr_step8(p18, x0, x1, a18);
        l2sqr_step8(p19, x0, x1, a19);
        l2sqr_step8(p20, x0, x1, a20);
        l2sqr_step8(p21, x0, x1, a21);
        l2sqr_step8(p22, x0, x1, a22);
        l2sqr_step8(p23, x0, x1, a23);

        prfm_w(dis + 0);
        prfm_w(dis + 8);
        prfm_w(dis + 16);
    }

    float d0 = vaddvq_f32(a0);
    float d1 = vaddvq_f32(a1);
    float d2 = vaddvq_f32(a2);
    float d3 = vaddvq_f32(a3);
    float d4 = vaddvq_f32(a4);
    float d5 = vaddvq_f32(a5);
    float d6 = vaddvq_f32(a6);
    float d7 = vaddvq_f32(a7);
    float d8 = vaddvq_f32(a8);
    float d9 = vaddvq_f32(a9);
    float d10 = vaddvq_f32(a10);
    float d11 = vaddvq_f32(a11);
    float d12 = vaddvq_f32(a12);
    float d13 = vaddvq_f32(a13);
    float d14 = vaddvq_f32(a14);
    float d15 = vaddvq_f32(a15);
    float d16 = vaddvq_f32(a16);
    float d17 = vaddvq_f32(a17);
    float d18 = vaddvq_f32(a18);
    float d19 = vaddvq_f32(a19);
    float d20 = vaddvq_f32(a20);
    float d21 = vaddvq_f32(a21);
    float d22 = vaddvq_f32(a22);
    float d23 = vaddvq_f32(a23);

    float q;
    for (; i < d; ++i) {
        float xv = *px++;
        q = *p0++;
        d0 += (q - xv) * (q - xv);
        q = *p1++;
        d1 += (q - xv) * (q - xv);
        q = *p2++;
        d2 += (q - xv) * (q - xv);
        q = *p3++;
        d3 += (q - xv) * (q - xv);
        q = *p4++;
        d4 += (q - xv) * (q - xv);
        q = *p5++;
        d5 += (q - xv) * (q - xv);
        q = *p6++;
        d6 += (q - xv) * (q - xv);
        q = *p7++;
        d7 += (q - xv) * (q - xv);
        q = *p8++;
        d8 += (q - xv) * (q - xv);
        q = *p9++;
        d9 += (q - xv) * (q - xv);
        q = *p10++;
        d10 += (q - xv) * (q - xv);
        q = *p11++;
        d11 += (q - xv) * (q - xv);
        q = *p12++;
        d12 += (q - xv) * (q - xv);
        q = *p13++;
        d13 += (q - xv) * (q - xv);
        q = *p14++;
        d14 += (q - xv) * (q - xv);
        q = *p15++;
        d15 += (q - xv) * (q - xv);
        q = *p16++;
        d16 += (q - xv) * (q - xv);
        q = *p17++;
        d17 += (q - xv) * (q - xv);
        q = *p18++;
        d18 += (q - xv) * (q - xv);
        q = *p19++;
        d19 += (q - xv) * (q - xv);
        q = *p20++;
        d20 += (q - xv) * (q - xv);
        q = *p21++;
        d21 += (q - xv) * (q - xv);
        q = *p22++;
        d22 += (q - xv) * (q - xv);
        q = *p23++;
        d23 += (q - xv) * (q - xv);
    }

    dis[0] = d0;
    dis[1] = d1;
    dis[2] = d2;
    dis[3] = d3;
    dis[4] = d4;
    dis[5] = d5;
    dis[6] = d6;
    dis[7] = d7;
    dis[8] = d8;
    dis[9] = d9;
    dis[10] = d10;
    dis[11] = d11;
    dis[12] = d12;
    dis[13] = d13;
    dis[14] = d14;
    dis[15] = d15;
    dis[16] = d16;
    dis[17] = d17;
    dis[18] = d18;
    dis[19] = d19;
    dis[20] = d20;
    dis[21] = d21;
    dis[22] = d22;
    dis[23] = d23;
}

static void L2sqrBatch16(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    const float *px = x;

    const float *p0 = y + 0 * d;
    const float *p1 = y + 1 * d;
    const float *p2 = y + 2 * d;
    const float *p3 = y + 3 * d;
    const float *p4 = y + 4 * d;
    const float *p5 = y + 5 * d;
    const float *p6 = y + 6 * d;
    const float *p7 = y + 7 * d;
    const float *p8 = y + 8 * d;
    const float *p9 = y + 9 * d;
    const float *p10 = y + 10 * d;
    const float *p11 = y + 11 * d;
    const float *p12 = y + 12 * d;
    const float *p13 = y + 13 * d;
    const float *p14 = y + 14 * d;
    const float *p15 = y + 15 * d;

    float32x4_t a0 = vdupq_n_f32(0);
    float32x4_t a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0);
    float32x4_t a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0);
    float32x4_t a5 = vdupq_n_f32(0);
    float32x4_t a6 = vdupq_n_f32(0);
    float32x4_t a7 = vdupq_n_f32(0);
    float32x4_t a8 = vdupq_n_f32(0);
    float32x4_t a9 = vdupq_n_f32(0);
    float32x4_t a10 = vdupq_n_f32(0);
    float32x4_t a11 = vdupq_n_f32(0);
    float32x4_t a12 = vdupq_n_f32(0);
    float32x4_t a13 = vdupq_n_f32(0);
    float32x4_t a14 = vdupq_n_f32(0);
    float32x4_t a15 = vdupq_n_f32(0);

    size_t i = 0;
    size_t single_round = 8;

    for (; i + single_round <= d; i += single_round) {
        prfm_r(px);

        float32x4_t x0;
        float32x4_t x1;
        ldp_q_f32(px, x0, x1);

        l2sqr_step8(p0, x0, x1, a0);
        l2sqr_step8(p1, x0, x1, a1);
        l2sqr_step8(p2, x0, x1, a2);
        l2sqr_step8(p3, x0, x1, a3);
        l2sqr_step8(p4, x0, x1, a4);
        l2sqr_step8(p5, x0, x1, a5);
        l2sqr_step8(p6, x0, x1, a6);
        l2sqr_step8(p7, x0, x1, a7);
        l2sqr_step8(p8, x0, x1, a8);
        l2sqr_step8(p9, x0, x1, a9);
        l2sqr_step8(p10, x0, x1, a10);
        l2sqr_step8(p11, x0, x1, a11);
        l2sqr_step8(p12, x0, x1, a12);
        l2sqr_step8(p13, x0, x1, a13);
        l2sqr_step8(p14, x0, x1, a14);
        l2sqr_step8(p15, x0, x1, a15);

        prfm_w(dis + 0);
        prfm_w(dis + 8);
    }

    float d0 = vaddvq_f32(a0);
    float d1 = vaddvq_f32(a1);
    float d2 = vaddvq_f32(a2);
    float d3 = vaddvq_f32(a3);
    float d4 = vaddvq_f32(a4);
    float d5 = vaddvq_f32(a5);
    float d6 = vaddvq_f32(a6);
    float d7 = vaddvq_f32(a7);
    float d8 = vaddvq_f32(a8);
    float d9 = vaddvq_f32(a9);
    float d10 = vaddvq_f32(a10);
    float d11 = vaddvq_f32(a11);
    float d12 = vaddvq_f32(a12);
    float d13 = vaddvq_f32(a13);
    float d14 = vaddvq_f32(a14);
    float d15 = vaddvq_f32(a15);

    for (; i < d; ++i) {
        float xv = *px++;
        float q;
        q = *p0++;
        d0 += (q - xv) * (q - xv);
        q = *p1++;
        d1 += (q - xv) * (q - xv);
        q = *p2++;
        d2 += (q - xv) * (q - xv);
        q = *p3++;
        d3 += (q - xv) * (q - xv);
        q = *p4++;
        d4 += (q - xv) * (q - xv);
        q = *p5++;
        d5 += (q - xv) * (q - xv);
        q = *p6++;
        d6 += (q - xv) * (q - xv);
        q = *p7++;
        d7 += (q - xv) * (q - xv);
        q = *p8++;
        d8 += (q - xv) * (q - xv);
        q = *p9++;
        d9 += (q - xv) * (q - xv);
        q = *p10++;
        d10 += (q - xv) * (q - xv);
        q = *p11++;
        d11 += (q - xv) * (q - xv);
        q = *p12++;
        d12 += (q - xv) * (q - xv);
        q = *p13++;
        d13 += (q - xv) * (q - xv);
        q = *p14++;
        d14 += (q - xv) * (q - xv);
        q = *p15++;
        d15 += (q - xv) * (q - xv);
    }

    dis[0] = d0;
    dis[1] = d1;
    dis[2] = d2;
    dis[3] = d3;
    dis[4] = d4;
    dis[5] = d5;
    dis[6] = d6;
    dis[7] = d7;
    dis[8] = d8;
    dis[9] = d9;
    dis[10] = d10;
    dis[11] = d11;
    dis[12] = d12;
    dis[13] = d13;
    dis[14] = d14;
    dis[15] = d15;
}

static void L2sqrBatch8(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    const float *px = x;

    const float *p0 = y + 0 * d;
    const float *p1 = y + 1 * d;
    const float *p2 = y + 2 * d;
    const float *p3 = y + 3 * d;
    const float *p4 = y + 4 * d;
    const float *p5 = y + 5 * d;
    const float *p6 = y + 6 * d;
    const float *p7 = y + 7 * d;

    float32x4_t a0 = vdupq_n_f32(0);
    float32x4_t a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0);
    float32x4_t a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0);
    float32x4_t a5 = vdupq_n_f32(0);
    float32x4_t a6 = vdupq_n_f32(0);
    float32x4_t a7 = vdupq_n_f32(0);

    size_t i = 0;
    size_t single_round = 8;

    for (; i + single_round <= d; i += single_round) {
        prfm_r(px);
        float32x4_t x0;
        float32x4_t x1;
        ldp_q_f32(px, x0, x1);

        l2sqr_step8(p0, x0, x1, a0);
        l2sqr_step8(p1, x0, x1, a1);
        l2sqr_step8(p2, x0, x1, a2);
        l2sqr_step8(p3, x0, x1, a3);
        l2sqr_step8(p4, x0, x1, a4);
        l2sqr_step8(p5, x0, x1, a5);
        l2sqr_step8(p6, x0, x1, a6);
        l2sqr_step8(p7, x0, x1, a7);

        prfm_w(dis + 0);
    }

    float d0 = vaddvq_f32(a0);
    float d1 = vaddvq_f32(a1);
    float d2 = vaddvq_f32(a2);
    float d3 = vaddvq_f32(a3);
    float d4 = vaddvq_f32(a4);
    float d5 = vaddvq_f32(a5);
    float d6 = vaddvq_f32(a6);
    float d7 = vaddvq_f32(a7);

    for (; i < d; ++i) {
        float xv = *px++;
        float q;
        q = *p0++;
        d0 += (q - xv) * (q - xv);
        q = *p1++;
        d1 += (q - xv) * (q - xv);
        q = *p2++;
        d2 += (q - xv) * (q - xv);
        q = *p3++;
        d3 += (q - xv) * (q - xv);
        q = *p4++;
        d4 += (q - xv) * (q - xv);
        q = *p5++;
        d5 += (q - xv) * (q - xv);
        q = *p6++;
        d6 += (q - xv) * (q - xv);
        q = *p7++;
        d7 += (q - xv) * (q - xv);
    }

    dis[0] = d0;
    dis[1] = d1;
    dis[2] = d2;
    dis[3] = d3;
    dis[4] = d4;
    dis[5] = d5;
    dis[6] = d6;
    dis[7] = d7;
}

static void L2sqrBatch4(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    const float *px = x;

    const float *p0 = y + 0 * d;
    const float *p1 = y + 1 * d;
    const float *p2 = y + 2 * d;
    const float *p3 = y + 3 * d;

    float32x4_t a0 = vdupq_n_f32(0);
    float32x4_t a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0);
    float32x4_t a3 = vdupq_n_f32(0);

    size_t i = 0;
    size_t single_round = 8;

    for (; i + single_round <= d; i += single_round) {
        prfm_r(px);
        float32x4_t x0;
        float32x4_t x1;
        ldp_q_f32(px, x0, x1);

        l2sqr_step8(p0, x0, x1, a0);
        l2sqr_step8(p1, x0, x1, a1);
        l2sqr_step8(p2, x0, x1, a2);
        l2sqr_step8(p3, x0, x1, a3);

        prfm_w(dis + 0);
    }

    float d0 = vaddvq_f32(a0);
    float d1 = vaddvq_f32(a1);
    float d2 = vaddvq_f32(a2);
    float d3 = vaddvq_f32(a3);

    for (; i < d; ++i) {
        float xv = *px++;
        float q;
        q = *p0++;
        d0 += (q - xv) * (q - xv);
        q = *p1++;
        d1 += (q - xv) * (q - xv);
        q = *p2++;
        d2 += (q - xv) * (q - xv);
        q = *p3++;
        d3 += (q - xv) * (q - xv);
    }

    dis[0] = d0;
    dis[1] = d1;
    dis[2] = d2;
    dis[3] = d3;
}

static void L2sqrBatch2(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    const float *px = x;

    const float *p0 = y + 0 * d;
    const float *p1 = y + 1 * d;

    float32x4_t a0 = vdupq_n_f32(0);
    float32x4_t a1 = vdupq_n_f32(0);

    size_t i = 0;
    size_t single_round = 8;

    for (; i + single_round <= d; i += single_round) {
        prfm_r(px);
        float32x4_t x0;
        float32x4_t x1;
        ldp_q_f32(px, x0, x1);

        l2sqr_step8(p0, x0, x1, a0);
        l2sqr_step8(p1, x0, x1, a1);

        prfm_w(dis + 0);
    }

    float d0 = vaddvq_f32(a0);
    float d1 = vaddvq_f32(a1);

    for (; i < d; ++i) {
        float xv = *px++;
        float q;
        q = *p0++;
        d0 += (q - xv) * (q - xv);
        q = *p1++;
        d1 += (q - xv) * (q - xv);
    }

    dis[0] = d0;
    dis[1] = d1;
}
#elif defined(__clang__)
static void L2sqrBatch24(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    if (LIKELY(d >= single_round)) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 4 * d);
        neon_base2 = vld1q_f32(y + 5 * d);
        neon_base3 = vld1q_f32(y + 6 * d);
        neon_base4 = vld1q_f32(y + 7 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res6 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res7 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res8 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 8 * d);
        neon_base2 = vld1q_f32(y + 9 * d);
        neon_base3 = vld1q_f32(y + 10 * d);
        neon_base4 = vld1q_f32(y + 11 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 12 * d);
        neon_base2 = vld1q_f32(y + 13 * d);
        neon_base3 = vld1q_f32(y + 14 * d);
        neon_base4 = vld1q_f32(y + 15 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res13 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res14 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res15 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res16 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 16 * d);
        neon_base2 = vld1q_f32(y + 17 * d);
        neon_base3 = vld1q_f32(y + 18 * d);
        neon_base4 = vld1q_f32(y + 19 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res17 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res18 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res19 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res20 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y + 20 * d);
        neon_base2 = vld1q_f32(y + 21 * d);
        neon_base3 = vld1q_f32(y + 22 * d);
        neon_base4 = vld1q_f32(y + 23 * d);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res21 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res22 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res23 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res24 = vmulq_f32(neon_base4, neon_base4);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 4 * d + i);
            neon_base2 = vld1q_f32(y + 5 * d + i);
            neon_base3 = vld1q_f32(y + 6 * d + i);
            neon_base4 = vld1q_f32(y + 7 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base1, neon_base1);
            neon_res6 = vmlaq_f32(neon_res6, neon_base2, neon_base2);
            neon_res7 = vmlaq_f32(neon_res7, neon_base3, neon_base3);
            neon_res8 = vmlaq_f32(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 8 * d + i);
            neon_base2 = vld1q_f32(y + 9 * d + i);
            neon_base3 = vld1q_f32(y + 10 * d + i);
            neon_base4 = vld1q_f32(y + 11 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 12 * d + i);
            neon_base2 = vld1q_f32(y + 13 * d + i);
            neon_base3 = vld1q_f32(y + 14 * d + i);
            neon_base4 = vld1q_f32(y + 15 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base1, neon_base1);
            neon_res14 = vmlaq_f32(neon_res14, neon_base2, neon_base2);
            neon_res15 = vmlaq_f32(neon_res15, neon_base3, neon_base3);
            neon_res16 = vmlaq_f32(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 16 * d + i);
            neon_base2 = vld1q_f32(y + 17 * d + i);
            neon_base3 = vld1q_f32(y + 18 * d + i);
            neon_base4 = vld1q_f32(y + 19 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res17 = vmlaq_f32(neon_res17, neon_base1, neon_base1);
            neon_res18 = vmlaq_f32(neon_res18, neon_base2, neon_base2);
            neon_res19 = vmlaq_f32(neon_res19, neon_base3, neon_base3);
            neon_res20 = vmlaq_f32(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y + 20 * d + i);
            neon_base2 = vld1q_f32(y + 21 * d + i);
            neon_base3 = vld1q_f32(y + 22 * d + i);
            neon_base4 = vld1q_f32(y + 23 * d + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res21 = vmlaq_f32(neon_res21, neon_base1, neon_base1);
            neon_res22 = vmlaq_f32(neon_res22, neon_base2, neon_base2);
            neon_res23 = vmlaq_f32(neon_res23, neon_base3, neon_base3);
            neon_res24 = vmlaq_f32(neon_res24, neon_base4, neon_base4);
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

static void L2sqrBatch16(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    if (LIKELY(d >= single_round)) {
        float32x4_t neon_query = vld1q_f32(x);

        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        float32x4_t neon_base5 = vld1q_f32(y + 4 * d);
        float32x4_t neon_base6 = vld1q_f32(y + 5 * d);
        float32x4_t neon_base7 = vld1q_f32(y + 6 * d);
        float32x4_t neon_base8 = vld1q_f32(y + 7 * d);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_base8);

        neon_base1 = vld1q_f32(y + 8 * d);
        neon_base2 = vld1q_f32(y + 9 * d);
        neon_base3 = vld1q_f32(y + 10 * d);
        neon_base4 = vld1q_f32(y + 11 * d);
        neon_base5 = vld1q_f32(y + 12 * d);
        neon_base6 = vld1q_f32(y + 13 * d);
        neon_base7 = vld1q_f32(y + 14 * d);
        neon_base8 = vld1q_f32(y + 15 * d);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res13 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res14 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res15 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res16 = vmulq_f32(neon_base8, neon_base8);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base5 = vld1q_f32(y + 4 * d + i);
            neon_base6 = vld1q_f32(y + 5 * d + i);
            neon_base7 = vld1q_f32(y + 6 * d + i);
            neon_base8 = vld1q_f32(y + 7 * d + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f32(y + 8 * d + i);
            neon_base2 = vld1q_f32(y + 9 * d + i);
            neon_base3 = vld1q_f32(y + 10 * d + i);
            neon_base4 = vld1q_f32(y + 11 * d + i);
            neon_base5 = vld1q_f32(y + 12 * d + i);
            neon_base6 = vld1q_f32(y + 13 * d + i);
            neon_base7 = vld1q_f32(y + 14 * d + i);
            neon_base8 = vld1q_f32(y + 15 * d + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_base5);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_base6);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_base7);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_base8);
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
        for (int i = 0; i < 16; i++) {
            dis[i] = 0.0f;
        }
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

static void L2sqrBatch8(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    if (LIKELY(d >= single_round)) {
        float32x4_t neon_query = vld1q_f32(x);

        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        float32x4_t neon_base5 = vld1q_f32(y + 4 * d);
        float32x4_t neon_base6 = vld1q_f32(y + 5 * d);
        float32x4_t neon_base7 = vld1q_f32(y + 6 * d);
        float32x4_t neon_base8 = vld1q_f32(y + 7 * d);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_base8);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);

            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base5 = vld1q_f32(y + 4 * d + i);
            neon_base6 = vld1q_f32(y + 5 * d + i);
            neon_base7 = vld1q_f32(y + 6 * d + i);
            neon_base8 = vld1q_f32(y + 7 * d + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
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

static void L2sqrBatch4(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    constexpr size_t single_round = 4;
    size_t i;
    if (LIKELY(d >= single_round)) {
        float32x4_t b = vld1q_f32(x);

        float32x4_t q0 = vld1q_f32(y);
        float32x4_t q1 = vld1q_f32(y + d);
        float32x4_t q2 = vld1q_f32(y + 2 * d);
        float32x4_t q3 = vld1q_f32(y + 3 * d);

        q0 = vsubq_f32(q0, b);
        q1 = vsubq_f32(q1, b);
        q2 = vsubq_f32(q2, b);
        q3 = vsubq_f32(q3, b);

        float32x4_t res0 = vmulq_f32(q0, q0);
        float32x4_t res1 = vmulq_f32(q1, q1);
        float32x4_t res2 = vmulq_f32(q2, q2);
        float32x4_t res3 = vmulq_f32(q3, q3);

        for (i = single_round; i <= d - single_round; i += single_round) {
            b = vld1q_f32(x + i);

            q0 = vld1q_f32(y + i);
            q1 = vld1q_f32(y + d + i);
            q2 = vld1q_f32(y + 2 * d + i);
            q3 = vld1q_f32(y + 3 * d + i);

            q0 = vsubq_f32(q0, b);
            q1 = vsubq_f32(q1, b);
            q2 = vsubq_f32(q2, b);
            q3 = vsubq_f32(q3, b);

            res0 = vmlaq_f32(res0, q0, q0);
            res1 = vmlaq_f32(res1, q1, q1);
            res2 = vmlaq_f32(res2, q2, q2);
            res3 = vmlaq_f32(res3, q3, q3);
        }
        dis[0] = vaddvq_f32(res0);
        dis[1] = vaddvq_f32(res1);
        dis[2] = vaddvq_f32(res2);
        dis[3] = vaddvq_f32(res3);
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (d > i) {
        float q0 = x[i] - *(y + i);
        float q1 = x[i] - *(y + d + i);
        float q2 = x[i] - *(y + 2 * d + i);
        float q3 = x[i] - *(y + 3 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        for (i++; i < d; ++i) {
            float q0 = x[i] - *(y + i);
            float q1 = x[i] - *(y + d + i);
            float q2 = x[i] - *(y + 2 * d + i);
            float q3 = x[i] - *(y + 3 * d + i);
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

static void L2sqrBatch2(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;

    if (LIKELY(d >= single_round)) {
        float32x4_t x_0 = vld1q_f32(x);
        float32x4_t x_1 = vld1q_f32(x + 4);

        float32x4_t y0_0 = vld1q_f32(y);
        float32x4_t y0_1 = vld1q_f32(y + 4);
        float32x4_t y1_0 = vld1q_f32(y + d);
        float32x4_t y1_1 = vld1q_f32(y + d + 4);

        float32x4_t d0_0 = vsubq_f32(x_0, y0_0);
        d0_0 = vmulq_f32(d0_0, d0_0);
        float32x4_t d0_1 = vsubq_f32(x_1, y0_1);
        d0_1 = vmulq_f32(d0_1, d0_1);
        float32x4_t d1_0 = vsubq_f32(x_0, y1_0);
        d1_0 = vmulq_f32(d1_0, d1_0);
        float32x4_t d1_1 = vsubq_f32(x_1, y1_1);
        d1_1 = vmulq_f32(d1_1, d1_1);

        for (i = single_round; i <= d - single_round; i += single_round) {
            x_0 = vld1q_f32(x + i);
            y0_0 = vld1q_f32(y + i);
            y1_0 = vld1q_f32(y + d + i);
            const float32x4_t q0_0 = vsubq_f32(x_0, y0_0);
            const float32x4_t q1_0 = vsubq_f32(x_0, y1_0);
            d0_0 = vmlaq_f32(d0_0, q0_0, q0_0);
            d1_0 = vmlaq_f32(d1_0, q1_0, q1_0);

            x_1 = vld1q_f32(x + i + 4);
            y0_1 = vld1q_f32(y + i + 4);
            y1_1 = vld1q_f32(y + d + i + 4);
            const float32x4_t q0_1 = vsubq_f32(x_1, y0_1);
            const float32x4_t q1_1 = vsubq_f32(x_1, y1_1);
            d0_1 = vmlaq_f32(d0_1, q0_1, q0_1);
            d1_1 = vmlaq_f32(d1_1, q1_1, q1_1);
        }

        d0_0 = vaddq_f32(d0_0, d0_1);
        d1_0 = vaddq_f32(d1_0, d1_1);
        dis[0] = vaddvq_f32(d0_0);
        dis[1] = vaddvq_f32(d1_0);
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
#endif

void L2sqrSingle(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 16;
    size_t i;
    float res;

    if (LIKELY(d >= multi_round)) {
        prefetch_Lx(x + multi_round);
        prefetch_Lx(y + multi_round);
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t x8_1 = vld1q_f32(x + 4);
        float32x4_t x8_2 = vld1q_f32(x + 8);
        float32x4_t x8_3 = vld1q_f32(x + 12);

        float32x4_t y8_0 = vld1q_f32(y);
        float32x4_t y8_1 = vld1q_f32(y + 4);
        float32x4_t y8_2 = vld1q_f32(y + 8);
        float32x4_t y8_3 = vld1q_f32(y + 12);

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        float32x4_t d8_1 = vsubq_f32(x8_1, y8_1);
        d8_1 = vmulq_f32(d8_1, d8_1);
        float32x4_t d8_2 = vsubq_f32(x8_2, y8_2);
        d8_2 = vmulq_f32(d8_2, d8_2);
        float32x4_t d8_3 = vsubq_f32(x8_3, y8_3);
        d8_3 = vmulq_f32(d8_3, d8_3);

        for (i = multi_round; i <= d - multi_round; i += multi_round) {
            prefetch_Lx(x + i + multi_round);
            prefetch_Lx(y + i + multi_round);
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);

            x8_1 = vld1q_f32(x + i + 4);
            y8_1 = vld1q_f32(y + i + 4);
            const float32x4_t q8_1 = vsubq_f32(x8_1, y8_1);
            d8_1 = vmlaq_f32(d8_1, q8_1, q8_1);

            x8_2 = vld1q_f32(x + i + 8);
            y8_2 = vld1q_f32(y + i + 8);
            const float32x4_t q8_2 = vsubq_f32(x8_2, y8_2);
            d8_2 = vmlaq_f32(d8_2, q8_2, q8_2);

            x8_3 = vld1q_f32(x + i + 12);
            y8_3 = vld1q_f32(y + i + 12);
            const float32x4_t q8_3 = vsubq_f32(x8_3, y8_3);
            d8_3 = vmlaq_f32(d8_3, q8_3, q8_3);
        }

        for (; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);
        }

        d8_0 = vaddq_f32(d8_0, d8_1);
        d8_2 = vaddq_f32(d8_2, d8_3);
        d8_0 = vaddq_f32(d8_0, d8_2);
        res = vaddvq_f32(d8_0);
    } else if (d >= single_round) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t y8_0 = vld1q_f32(y);

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        for (i = single_round; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);
        }
        res = vaddvq_f32(d8_0);
    } else {
        res = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    *dis = res;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((always_inline)) static inline void ip_step8(float32x4_t &acc, const float *ybase, size_t i,
                                                           const float32x4_t x0, const float32x4_t x1, float32x4_t &a,
                                                           float32x4_t &b) noexcept
{
    const float *p = ybase + i;
    ldp_q_f32(p, a, b);
    acc = vfmaq_f32(acc, a, x0);
    acc = vfmaq_f32(acc, b, x1);
}

static void IPBatch16(const float *__restrict x, const float *__restrict y, const size_t d, float *__restrict dis)
{
    constexpr size_t single_round = 8;

    {
        const float *y0 = y + 0 * d;
        const float *y1 = y + 1 * d;
        const float *y2 = y + 2 * d;
        const float *y3 = y + 3 * d;
        const float *y4 = y + 4 * d;
        const float *y5 = y + 5 * d;
        const float *y6 = y + 6 * d;
        const float *y7 = y + 7 * d;

        float32x4_t acc0 = vdupq_n_f32(0);
        float32x4_t acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0);
        float32x4_t acc3 = vdupq_n_f32(0);
        float32x4_t acc4 = vdupq_n_f32(0);
        float32x4_t acc5 = vdupq_n_f32(0);
        float32x4_t acc6 = vdupq_n_f32(0);
        float32x4_t acc7v = vdupq_n_f32(0);

        size_t i = 0;
        for (; i + single_round <= d; i += single_round) {
            float32x4_t x0;
            float32x4_t x1;
            const float *px = x + i;
            ldp_q_f32(px, x0, x1);

            float32x4_t a;
            float32x4_t b;
            ip_step8(acc0, y0, i, x0, x1, a, b);
            ip_step8(acc1, y1, i, x0, x1, a, b);
            ip_step8(acc2, y2, i, x0, x1, a, b);
            ip_step8(acc3, y3, i, x0, x1, a, b);
            ip_step8(acc4, y4, i, x0, x1, a, b);
            ip_step8(acc5, y5, i, x0, x1, a, b);
            ip_step8(acc6, y6, i, x0, x1, a, b);
            ip_step8(acc7v, y7, i, x0, x1, a, b);
        }

        float t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7t = 0;
        for (; i < d; ++i) {
            float xv = x[i];
            t0 += xv * y0[i];
            t1 += xv * y1[i];
            t2 += xv * y2[i];
            t3 += xv * y3[i];
            t4 += xv * y4[i];
            t5 += xv * y5[i];
            t6 += xv * y6[i];
            t7t += xv * y7[i];
        }

        dis[0] = vaddvq_f32(acc0) + t0;
        dis[1] = vaddvq_f32(acc1) + t1;
        dis[2] = vaddvq_f32(acc2) + t2;
        dis[3] = vaddvq_f32(acc3) + t3;
        dis[4] = vaddvq_f32(acc4) + t4;
        dis[5] = vaddvq_f32(acc5) + t5;
        dis[6] = vaddvq_f32(acc6) + t6;
        dis[7] = vaddvq_f32(acc7v) + t7t;
    }

    {
        const float *y8 = y + 8 * d;
        const float *y9 = y + 9 * d;
        const float *y10 = y + 10 * d;
        const float *y11 = y + 11 * d;
        const float *y12 = y + 12 * d;
        const float *y13 = y + 13 * d;
        const float *y14 = y + 14 * d;
        const float *y15 = y + 15 * d;

        float32x4_t acc8 = vdupq_n_f32(0);
        float32x4_t acc9 = vdupq_n_f32(0);
        float32x4_t acc10 = vdupq_n_f32(0);
        float32x4_t acc11 = vdupq_n_f32(0);
        float32x4_t acc12 = vdupq_n_f32(0);
        float32x4_t acc13 = vdupq_n_f32(0);
        float32x4_t acc14 = vdupq_n_f32(0);
        float32x4_t acc15v = vdupq_n_f32(0);

        size_t i = 0;
        for (; i + single_round <= d; i += single_round) {
            float32x4_t x0;
            float32x4_t x1;
            const float *px = x + i;
            ldp_q_f32(px, x0, x1);

            float32x4_t a;
            float32x4_t b;
            ip_step8(acc8, y8, i, x0, x1, a, b);
            ip_step8(acc9, y9, i, x0, x1, a, b);
            ip_step8(acc10, y10, i, x0, x1, a, b);
            ip_step8(acc11, y11, i, x0, x1, a, b);
            ip_step8(acc12, y12, i, x0, x1, a, b);
            ip_step8(acc13, y13, i, x0, x1, a, b);
            ip_step8(acc14, y14, i, x0, x1, a, b);
            ip_step8(acc15v, y15, i, x0, x1, a, b);
        }

        float t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15t = 0;
        for (; i < d; ++i) {
            float xv = x[i];
            t8 += xv * y8[i];
            t9 += xv * y9[i];
            t10 += xv * y10[i];
            t11 += xv * y11[i];
            t12 += xv * y12[i];
            t13 += xv * y13[i];
            t14 += xv * y14[i];
            t15t += xv * y15[i];
        }

        dis[8] = vaddvq_f32(acc8) + t8;
        dis[9] = vaddvq_f32(acc9) + t9;
        dis[10] = vaddvq_f32(acc10) + t10;
        dis[11] = vaddvq_f32(acc11) + t11;
        dis[12] = vaddvq_f32(acc12) + t12;
        dis[13] = vaddvq_f32(acc13) + t13;
        dis[14] = vaddvq_f32(acc14) + t14;
        dis[15] = vaddvq_f32(acc15v) + t15t;
    }
}

static void IPBatch8(const float *__restrict x, const float *__restrict y, const size_t d, float *__restrict dis)
{
    constexpr size_t single_round = 8;

    const float *y0 = y + 0 * d;
    const float *y1 = y + 1 * d;
    const float *y2 = y + 2 * d;
    const float *y3 = y + 3 * d;
    const float *y4 = y + 4 * d;
    const float *y5 = y + 5 * d;
    const float *y6 = y + 6 * d;
    const float *y7 = y + 7 * d;

    size_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0);
    float32x4_t acc3 = vdupq_n_f32(0);
    float32x4_t acc4 = vdupq_n_f32(0);
    float32x4_t acc5 = vdupq_n_f32(0);
    float32x4_t acc6 = vdupq_n_f32(0);
    float32x4_t acc7v = vdupq_n_f32(0);

    if (LIKELY(d >= single_round)) {
        for (; i + single_round <= d; i += single_round) {
            float32x4_t x0;
            float32x4_t x1;
            const float *px = x + i;
            ldp_q_f32(px, x0, x1);

            float32x4_t a;
            float32x4_t b;
            ip_step8(acc0, y0, i, x0, x1, a, b);
            ip_step8(acc1, y1, i, x0, x1, a, b);
            ip_step8(acc2, y2, i, x0, x1, a, b);
            ip_step8(acc3, y3, i, x0, x1, a, b);
            ip_step8(acc4, y4, i, x0, x1, a, b);
            ip_step8(acc5, y5, i, x0, x1, a, b);
            ip_step8(acc6, y6, i, x0, x1, a, b);
            ip_step8(acc7v, y7, i, x0, x1, a, b);
        }

        if (i < d) {
            float t0 = 0;
            float t1 = 0;
            float t2 = 0;
            float t3 = 0;
            float t4 = 0;
            float t5 = 0;
            float t6 = 0;
            float t7t = 0;
            for (; i < d; ++i) {
                float xv = x[i];
                t0 += xv * y0[i];
                t1 += xv * y1[i];
                t2 += xv * y2[i];
                t3 += xv * y3[i];
                t4 += xv * y4[i];
                t5 += xv * y5[i];
                t6 += xv * y6[i];
                t7t += xv * y7[i];
            }
            dis[0] = vaddvq_f32(acc0) + t0;
            dis[1] = vaddvq_f32(acc1) + t1;
            dis[2] = vaddvq_f32(acc2) + t2;
            dis[3] = vaddvq_f32(acc3) + t3;
            dis[4] = vaddvq_f32(acc4) + t4;
            dis[5] = vaddvq_f32(acc5) + t5;
            dis[6] = vaddvq_f32(acc6) + t6;
            dis[7] = vaddvq_f32(acc7v) + t7t;
        } else {
            dis[0] = vaddvq_f32(acc0);
            dis[1] = vaddvq_f32(acc1);
            dis[2] = vaddvq_f32(acc2);
            dis[3] = vaddvq_f32(acc3);
            dis[4] = vaddvq_f32(acc4);
            dis[5] = vaddvq_f32(acc5);
            dis[6] = vaddvq_f32(acc6);
            dis[7] = vaddvq_f32(acc7v);
        }
    } else {
        float t0 = 0;
        float t1 = 0;
        float t2 = 0;
        float t3 = 0;
        float t4 = 0;
        float t5 = 0;
        float t6 = 0;
        float t7t = 0;
        for (; i < d; ++i) {
            float xv = x[i];
            t0 += xv * y0[i];
            t1 += xv * y1[i];
            t2 += xv * y2[i];
            t3 += xv * y3[i];
            t4 += xv * y4[i];
            t5 += xv * y5[i];
            t6 += xv * y6[i];
            t7t += xv * y7[i];
        }
        dis[0] = t0;
        dis[1] = t1;
        dis[2] = t2;
        dis[3] = t3;
        dis[4] = t4;
        dis[5] = t5;
        dis[6] = t6;
        dis[7] = t7t;
    }
}

static void IPBatch4(const float *__restrict x, const float *__restrict y, const size_t d, float *__restrict dis)
{
    constexpr size_t single_round = 8;

    const float *y0 = y + 0 * d;
    const float *y1 = y + 1 * d;
    const float *y2 = y + 2 * d;
    const float *y3 = y + 3 * d;

    size_t i = 0;

    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0);
    float32x4_t acc3 = vdupq_n_f32(0);

    if (LIKELY(d >= single_round)) {
        for (; i + single_round <= d; i += single_round) {
            float32x4_t x0;
            float32x4_t x1;
            const float *px = x + i;
            ldp_q_f32(px, x0, x1);

            float32x4_t a;
            float32x4_t b;

            ip_step8(acc0, y0, i, x0, x1, a, b);
            ip_step8(acc1, y1, i, x0, x1, a, b);
            ip_step8(acc2, y2, i, x0, x1, a, b);
            ip_step8(acc3, y3, i, x0, x1, a, b);
        }

        if (i < d) {
            float t0 = 0;
            float t1 = 0;
            float t2 = 0;
            float t3 = 0;

            for (; i < d; ++i) {
                float xv = x[i];
                t0 += xv * y0[i];
                t1 += xv * y1[i];
                t2 += xv * y2[i];
                t3 += xv * y3[i];
            }

            dis[0] = vaddvq_f32(acc0) + t0;
            dis[1] = vaddvq_f32(acc1) + t1;
            dis[2] = vaddvq_f32(acc2) + t2;
            dis[3] = vaddvq_f32(acc3) + t3;
        } else {
            dis[0] = vaddvq_f32(acc0);
            dis[1] = vaddvq_f32(acc1);
            dis[2] = vaddvq_f32(acc2);
            dis[3] = vaddvq_f32(acc3);
        }
    } else {
        float t0 = 0;
        float t1 = 0;
        float t2 = 0;
        float t3 = 0;

        for (; i < d; ++i) {
            float xv = x[i];
            t0 += xv * y0[i];
            t1 += xv * y1[i];
            t2 += xv * y2[i];
            t3 += xv * y3[i];
        }

        dis[0] = t0;
        dis[1] = t1;
        dis[2] = t2;
        dis[3] = t3;
    }
}

static void IPBatch2(const float *__restrict x, const float *__restrict y, const size_t d, float *__restrict dis)
{
    constexpr size_t single_round = 8;

    const float *y0 = y + 0 * d;
    const float *y1 = y + 1 * d;

    size_t i = 0;

    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);

    if (LIKELY(d >= single_round)) {
        for (; i + single_round <= d; i += single_round) {
            float32x4_t x0;
            float32x4_t x1;
            const float *px = x + i;
            ldp_q_f32(px, x0, x1);

            float32x4_t a;
            float32x4_t b;

            ip_step8(acc0, y0, i, x0, x1, a, b);
            ip_step8(acc1, y1, i, x0, x1, a, b);
        }

        float t0 = vaddvq_f32(acc0);
        float t1 = vaddvq_f32(acc1);

        for (; i < d; ++i) {
            float xv = x[i];
            t0 += xv * y0[i];
            t1 += xv * y1[i];
        }

        dis[0] = t0;
        dis[1] = t1;
    } else {
        float t0 = 0;
        float t1 = 0;

        for (; i < d; ++i) {
            float xv = x[i];
            t0 += xv * y0[i];
            t1 += xv * y1[i];
        }

        dis[0] = t0;
        dis[1] = t1;
    }
}
#elif defined(__clang__)
static void IPBatch16(const float *__restrict x, const float *__restrict y, const size_t d, float *__restrict dis)
{
    size_t i;
    constexpr size_t single_round = 4;

    if (LIKELY(d >= single_round)) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        float32x4_t neon_base5 = vld1q_f32(y + 4 * d);
        float32x4_t neon_base6 = vld1q_f32(y + 5 * d);
        float32x4_t neon_base7 = vld1q_f32(y + 6 * d);
        float32x4_t neon_base8 = vld1q_f32(y + 7 * d);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_query);

        neon_base1 = vld1q_f32(y + 8 * d);
        neon_base2 = vld1q_f32(y + 9 * d);
        neon_base3 = vld1q_f32(y + 10 * d);
        neon_base4 = vld1q_f32(y + 11 * d);
        neon_base5 = vld1q_f32(y + 12 * d);
        neon_base6 = vld1q_f32(y + 13 * d);
        neon_base7 = vld1q_f32(y + 14 * d);
        neon_base8 = vld1q_f32(y + 15 * d);

        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res13 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res14 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res15 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res16 = vmulq_f32(neon_base8, neon_query);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base5 = vld1q_f32(y + 4 * d + i);
            neon_base6 = vld1q_f32(y + 5 * d + i);
            neon_base7 = vld1q_f32(y + 6 * d + i);
            neon_base8 = vld1q_f32(y + 7 * d + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_f32(y + 8 * d + i);
            neon_base2 = vld1q_f32(y + 9 * d + i);
            neon_base3 = vld1q_f32(y + 10 * d + i);
            neon_base4 = vld1q_f32(y + 11 * d + i);
            neon_base5 = vld1q_f32(y + 12 * d + i);
            neon_base6 = vld1q_f32(y + 13 * d + i);
            neon_base7 = vld1q_f32(y + 14 * d + i);
            neon_base8 = vld1q_f32(y + 15 * d + i);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_query);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_query);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_query);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_query);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_query);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_query);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_query);
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
        for (int i = 0; i < 16; i++) {
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

static void IPBatch8(const float *__restrict x, const float *__restrict y, const size_t d, float *__restrict dis)
{
    size_t i;
    constexpr size_t single_round = 4;

    if (LIKELY(d >= single_round)) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);
        float32x4_t neon_base5 = vld1q_f32(y + 4 * d);
        float32x4_t neon_base6 = vld1q_f32(y + 5 * d);
        float32x4_t neon_base7 = vld1q_f32(y + 6 * d);
        float32x4_t neon_base8 = vld1q_f32(y + 7 * d);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_query);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);
            neon_base5 = vld1q_f32(y + 4 * d + i);
            neon_base6 = vld1q_f32(y + 5 * d + i);
            neon_base7 = vld1q_f32(y + 6 * d + i);
            neon_base8 = vld1q_f32(y + 7 * d + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);
        }

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else {
        for (int i = 0; i < 8; i++) {
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

static void IPBatch4(const float *__restrict x, const float *__restrict y, const size_t d, float *__restrict dis)
{
    size_t i;
    constexpr size_t single_round = 4;

    if (LIKELY(d >= single_round)) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y);
        float32x4_t neon_base2 = vld1q_f32(y + d);
        float32x4_t neon_base3 = vld1q_f32(y + 2 * d);
        float32x4_t neon_base4 = vld1q_f32(y + 3 * d);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y + i);
            neon_base2 = vld1q_f32(y + d + i);
            neon_base3 = vld1q_f32(y + 2 * d + i);
            neon_base4 = vld1q_f32(y + 3 * d + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
        }

        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}

static void IPBatch2(const float *__restrict x, const float *__restrict y, const size_t d, float *__restrict dis)
{
    size_t i;
    constexpr size_t single_round = 8;

    if (LIKELY(d >= single_round)) {
        float32x4_t x_0 = vld1q_f32(x);
        float32x4_t x_1 = vld1q_f32(x + 4);

        float32x4_t y0_0 = vld1q_f32(y);
        float32x4_t y0_1 = vld1q_f32(y + 4);
        float32x4_t y1_0 = vld1q_f32(y + d);
        float32x4_t y1_1 = vld1q_f32(y + d + 4);

        float32x4_t d0_0 = vmulq_f32(x_0, y0_0);
        float32x4_t d0_1 = vmulq_f32(x_1, y0_1);
        float32x4_t d1_0 = vmulq_f32(x_0, y1_0);
        float32x4_t d1_1 = vmulq_f32(x_1, y1_1);

        for (i = single_round; i <= d - single_round; i += single_round) {
            x_0 = vld1q_f32(x + i);
            y0_0 = vld1q_f32(y + i);
            y1_0 = vld1q_f32(y + d + i);
            d0_0 = vmlaq_f32(d0_0, x_0, y0_0);
            d1_0 = vmlaq_f32(d1_0, x_0, y1_0);

            x_1 = vld1q_f32(x + i + 4);
            y0_1 = vld1q_f32(y + i + 4);
            y1_1 = vld1q_f32(y + d + i + 4);
            d0_1 = vmlaq_f32(d0_1, x_1, y0_1);
            d1_1 = vmlaq_f32(d1_1, x_1, y1_1);
        }

        d0_0 = vaddq_f32(d0_0, d0_1);
        d1_0 = vaddq_f32(d1_0, d1_1);
        dis[0] = vaddvq_f32(d0_0);
        dis[1] = vaddvq_f32(d1_0);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp0 = x[i] * *(y + i);
        const float tmp1 = x[i] * *(y + d + i);
        dis[0] += tmp0;
        dis[1] += tmp1;
    }
}
#endif

void IPSingle(const float *x, const float *__restrict y, const size_t d, float *dis)
{
    size_t i;
    float res;
    constexpr size_t single_round = 16;

    if (LIKELY(d >= single_round)) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t x8_1 = vld1q_f32(x + 4);
        float32x4_t x8_2 = vld1q_f32(x + 8);
        float32x4_t x8_3 = vld1q_f32(x + 12);

        float32x4_t y8_0 = vld1q_f32(y);
        float32x4_t y8_1 = vld1q_f32(y + 4);
        float32x4_t y8_2 = vld1q_f32(y + 8);
        float32x4_t y8_3 = vld1q_f32(y + 12);

        float32x4_t d8_0 = vmulq_f32(x8_0, y8_0);
        float32x4_t d8_1 = vmulq_f32(x8_1, y8_1);
        float32x4_t d8_2 = vmulq_f32(x8_2, y8_2);
        float32x4_t d8_3 = vmulq_f32(x8_3, y8_3);

        for (i = single_round; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            d8_0 = vmlaq_f32(d8_0, x8_0, y8_0);

            x8_1 = vld1q_f32(x + i + 4);
            y8_1 = vld1q_f32(y + i + 4);
            d8_1 = vmlaq_f32(d8_1, x8_1, y8_1);

            x8_2 = vld1q_f32(x + i + 8);
            y8_2 = vld1q_f32(y + i + 8);
            d8_2 = vmlaq_f32(d8_2, x8_2, y8_2);

            x8_3 = vld1q_f32(x + i + 12);
            y8_3 = vld1q_f32(y + i + 12);
            d8_3 = vmlaq_f32(d8_3, x8_3, y8_3);
        }

        d8_0 = vaddq_f32(d8_0, d8_1);
        d8_2 = vaddq_f32(d8_2, d8_3);
        d8_0 = vaddq_f32(d8_0, d8_2);
        res = vaddvq_f32(d8_0);
    } else {
        i = 0;
        res = 0;
    }

    for (; i < d; i++) {
        const float tmp = x[i] * y[i];
        res += tmp;
    }
    *dis = res;
}

void L2sqrNy(float *dis, const float *x, const float *y, const size_t ny, const size_t d)
{
    size_t i = 0;

    for (; i + 24 <= ny; i += 24) {
        L2sqrBatch24(x, y + i * d, d, dis + i);
    }

    if (i + 16 <= ny) {
        L2sqrBatch16(x, y + i * d, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        L2sqrBatch8(x, y + i * d, d, dis + i);
        i += 8;
    }

    const size_t rem = ny - i;

    if (rem & 4) {
        L2sqrBatch4(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (rem & 2) {
        L2sqrBatch2(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (rem & 1) {
        L2sqrSingle(x, y + i * d, d, dis + i);
    }
}

void IPNy(float *dis, const float *x, const float *y, const size_t ny, const size_t d)
{
    size_t i = 0;

    for (; i + 16 <= ny; i += 16) {
        IPBatch16(x, y + i * d, d, dis + i);
    }

    const size_t rem = ny - i;

    if (rem & 8) {
        IPBatch8(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (rem & 4) {
        IPBatch4(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (rem & 2) {
        IPBatch2(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (rem & 1) {
        IPSingle(x, y + i * d, d, dis + i);
    }
}

#if defined(__GNUC__) && !defined(__clang__)
static void L2sqrBatch24ByIdx(const float *x, const float *__restrict *y, const size_t d, float *dis)
{
    const float *px = x;

    const float *p0 = y[0];
    const float *p1 = y[1];
    const float *p2 = y[2];
    const float *p3 = y[3];
    const float *p4 = y[4];
    const float *p5 = y[5];
    const float *p6 = y[6];
    const float *p7 = y[7];
    const float *p8 = y[8];
    const float *p9 = y[9];
    const float *p10 = y[10];
    const float *p11 = y[11];
    const float *p12 = y[12];
    const float *p13 = y[13];
    const float *p14 = y[14];
    const float *p15 = y[15];
    const float *p16 = y[16];
    const float *p17 = y[17];
    const float *p18 = y[18];
    const float *p19 = y[19];
    const float *p20 = y[20];
    const float *p21 = y[21];
    const float *p22 = y[22];
    const float *p23 = y[23];

    float32x4_t a0 = vdupq_n_f32(0);
    float32x4_t a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0);
    float32x4_t a3 = vdupq_n_f32(0);
    float32x4_t a4 = vdupq_n_f32(0);
    float32x4_t a5 = vdupq_n_f32(0);
    float32x4_t a6 = vdupq_n_f32(0);
    float32x4_t a7 = vdupq_n_f32(0);
    float32x4_t a8 = vdupq_n_f32(0);
    float32x4_t a9 = vdupq_n_f32(0);
    float32x4_t a10 = vdupq_n_f32(0);
    float32x4_t a11 = vdupq_n_f32(0);
    float32x4_t a12 = vdupq_n_f32(0);
    float32x4_t a13 = vdupq_n_f32(0);
    float32x4_t a14 = vdupq_n_f32(0);
    float32x4_t a15 = vdupq_n_f32(0);
    float32x4_t a16 = vdupq_n_f32(0);
    float32x4_t a17 = vdupq_n_f32(0);
    float32x4_t a18 = vdupq_n_f32(0);
    float32x4_t a19 = vdupq_n_f32(0);
    float32x4_t a20 = vdupq_n_f32(0);
    float32x4_t a21 = vdupq_n_f32(0);
    float32x4_t a22 = vdupq_n_f32(0);
    float32x4_t a23 = vdupq_n_f32(0);

    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        prfm_r(px + 32);

        prfm_r(p0 + 32);
        prfm_r(p1 + 32);
        prfm_r(p2 + 32);
        prfm_r(p3 + 32);
        prfm_r(p4 + 32);
        prfm_r(p5 + 32);
        prfm_r(p6 + 32);
        prfm_r(p7 + 32);
        prfm_r(p8 + 32);
        prfm_r(p9 + 32);
        prfm_r(p10 + 32);
        prfm_r(p11 + 32);
        prfm_r(p12 + 32);
        prfm_r(p13 + 32);
        prfm_r(p14 + 32);
        prfm_r(p15 + 32);
        prfm_r(p16 + 32);
        prfm_r(p17 + 32);
        prfm_r(p18 + 32);
        prfm_r(p19 + 32);
        prfm_r(p20 + 32);
        prfm_r(p21 + 32);
        prfm_r(p22 + 32);
        prfm_r(p23 + 32);

        float32x4_t x0;
        float32x4_t x1;
        ldp_q_f32(px, x0, x1);

        l2sqr_step8(p0, x0, x1, a0);
        l2sqr_step8(p1, x0, x1, a1);
        l2sqr_step8(p2, x0, x1, a2);
        l2sqr_step8(p3, x0, x1, a3);
        l2sqr_step8(p4, x0, x1, a4);
        l2sqr_step8(p5, x0, x1, a5);
        l2sqr_step8(p6, x0, x1, a6);
        l2sqr_step8(p7, x0, x1, a7);
        l2sqr_step8(p8, x0, x1, a8);
        l2sqr_step8(p9, x0, x1, a9);
        l2sqr_step8(p10, x0, x1, a10);
        l2sqr_step8(p11, x0, x1, a11);
        l2sqr_step8(p12, x0, x1, a12);
        l2sqr_step8(p13, x0, x1, a13);
        l2sqr_step8(p14, x0, x1, a14);
        l2sqr_step8(p15, x0, x1, a15);
        l2sqr_step8(p16, x0, x1, a16);
        l2sqr_step8(p17, x0, x1, a17);
        l2sqr_step8(p18, x0, x1, a18);
        l2sqr_step8(p19, x0, x1, a19);
        l2sqr_step8(p20, x0, x1, a20);
        l2sqr_step8(p21, x0, x1, a21);
        l2sqr_step8(p22, x0, x1, a22);
        l2sqr_step8(p23, x0, x1, a23);

        prfm_w(dis + 0);
        prfm_w(dis + 8);
        prfm_w(dis + 16);
    }

    float d0 = vaddvq_f32(a0);
    float d1 = vaddvq_f32(a1);
    float d2 = vaddvq_f32(a2);
    float d3 = vaddvq_f32(a3);
    float d4 = vaddvq_f32(a4);
    float d5 = vaddvq_f32(a5);
    float d6 = vaddvq_f32(a6);
    float d7 = vaddvq_f32(a7);
    float d8 = vaddvq_f32(a8);
    float d9 = vaddvq_f32(a9);
    float d10 = vaddvq_f32(a10);
    float d11 = vaddvq_f32(a11);
    float d12 = vaddvq_f32(a12);
    float d13 = vaddvq_f32(a13);
    float d14 = vaddvq_f32(a14);
    float d15 = vaddvq_f32(a15);
    float d16 = vaddvq_f32(a16);
    float d17 = vaddvq_f32(a17);
    float d18 = vaddvq_f32(a18);
    float d19 = vaddvq_f32(a19);
    float d20 = vaddvq_f32(a20);
    float d21 = vaddvq_f32(a21);
    float d22 = vaddvq_f32(a22);
    float d23 = vaddvq_f32(a23);

    for (; i < d; ++i) {
        float xv = *px++;
        float q, diff;

        q = *p0++;
        diff = q - xv;
        d0 += diff * diff;
        q = *p1++;
        diff = q - xv;
        d1 += diff * diff;
        q = *p2++;
        diff = q - xv;
        d2 += diff * diff;
        q = *p3++;
        diff = q - xv;
        d3 += diff * diff;
        q = *p4++;
        diff = q - xv;
        d4 += diff * diff;
        q = *p5++;
        diff = q - xv;
        d5 += diff * diff;
        q = *p6++;
        diff = q - xv;
        d6 += diff * diff;
        q = *p7++;
        diff = q - xv;
        d7 += diff * diff;
        q = *p8++;
        diff = q - xv;
        d8 += diff * diff;
        q = *p9++;
        diff = q - xv;
        d9 += diff * diff;
        q = *p10++;
        diff = q - xv;
        d10 += diff * diff;
        q = *p11++;
        diff = q - xv;
        d11 += diff * diff;
        q = *p12++;
        diff = q - xv;
        d12 += diff * diff;
        q = *p13++;
        diff = q - xv;
        d13 += diff * diff;
        q = *p14++;
        diff = q - xv;
        d14 += diff * diff;
        q = *p15++;
        diff = q - xv;
        d15 += diff * diff;
        q = *p16++;
        diff = q - xv;
        d16 += diff * diff;
        q = *p17++;
        diff = q - xv;
        d17 += diff * diff;
        q = *p18++;
        diff = q - xv;
        d18 += diff * diff;
        q = *p19++;
        diff = q - xv;
        d19 += diff * diff;
        q = *p20++;
        diff = q - xv;
        d20 += diff * diff;
        q = *p21++;
        diff = q - xv;
        d21 += diff * diff;
        q = *p22++;
        diff = q - xv;
        d22 += diff * diff;
        q = *p23++;
        diff = q - xv;
        d23 += diff * diff;
    }

    dis[0] = d0;
    dis[1] = d1;
    dis[2] = d2;
    dis[3] = d3;
    dis[4] = d4;
    dis[5] = d5;
    dis[6] = d6;
    dis[7] = d7;
    dis[8] = d8;
    dis[9] = d9;
    dis[10] = d10;
    dis[11] = d11;
    dis[12] = d12;
    dis[13] = d13;
    dis[14] = d14;
    dis[15] = d15;
    dis[16] = d16;
    dis[17] = d17;
    dis[18] = d18;
    dis[19] = d19;
    dis[20] = d20;
    dis[21] = d21;
    dis[22] = d22;
    dis[23] = d23;
}
#elif defined(__clang__)
static void L2sqrBatch24ByIdx(const float *x, const float *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 16;
    if (LIKELY(d >= multi_round)) {
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
        float32x4_t neon_res1;
        float32x4_t neon_res2;
        float32x4_t neon_res3;
        float32x4_t neon_res4;
        float32x4_t neon_res5;
        float32x4_t neon_res6;
        float32x4_t neon_res7;
        float32x4_t neon_res8;
        float32x4_t neon_res9;
        float32x4_t neon_res10;
        float32x4_t neon_res11;
        float32x4_t neon_res12;
        float32x4_t neon_res13;
        float32x4_t neon_res14;
        float32x4_t neon_res15;
        float32x4_t neon_res16;
        float32x4_t neon_res17;
        float32x4_t neon_res18;
        float32x4_t neon_res19;
        float32x4_t neon_res20;
        float32x4_t neon_res21;
        float32x4_t neon_res22;
        float32x4_t neon_res23;
        float32x4_t neon_res24;
        {
            const float32x4_t neon_query = vld1q_f32(x);
            float32x4_t neon_base1 = vld1q_f32(y[0]);
            float32x4_t neon_base2 = vld1q_f32(y[1]);
            float32x4_t neon_base3 = vld1q_f32(y[2]);
            float32x4_t neon_base4 = vld1q_f32(y[3]);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res1 = vmulq_f32(neon_base1, neon_base1);
            neon_res2 = vmulq_f32(neon_base2, neon_base2);
            neon_res3 = vmulq_f32(neon_base3, neon_base3);
            neon_res4 = vmulq_f32(neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[4]);
            neon_base2 = vld1q_f32(y[5]);
            neon_base3 = vld1q_f32(y[6]);
            neon_base4 = vld1q_f32(y[7]);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res5 = vmulq_f32(neon_base1, neon_base1);
            neon_res6 = vmulq_f32(neon_base2, neon_base2);
            neon_res7 = vmulq_f32(neon_base3, neon_base3);
            neon_res8 = vmulq_f32(neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[8]);
            neon_base2 = vld1q_f32(y[9]);
            neon_base3 = vld1q_f32(y[10]);
            neon_base4 = vld1q_f32(y[11]);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res9 = vmulq_f32(neon_base1, neon_base1);
            neon_res10 = vmulq_f32(neon_base2, neon_base2);
            neon_res11 = vmulq_f32(neon_base3, neon_base3);
            neon_res12 = vmulq_f32(neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[12]);
            neon_base2 = vld1q_f32(y[13]);
            neon_base3 = vld1q_f32(y[14]);
            neon_base4 = vld1q_f32(y[15]);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res13 = vmulq_f32(neon_base1, neon_base1);
            neon_res14 = vmulq_f32(neon_base2, neon_base2);
            neon_res15 = vmulq_f32(neon_base3, neon_base3);
            neon_res16 = vmulq_f32(neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[16]);
            neon_base2 = vld1q_f32(y[17]);
            neon_base3 = vld1q_f32(y[18]);
            neon_base4 = vld1q_f32(y[19]);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res17 = vmulq_f32(neon_base1, neon_base1);
            neon_res18 = vmulq_f32(neon_base2, neon_base2);
            neon_res19 = vmulq_f32(neon_base3, neon_base3);
            neon_res20 = vmulq_f32(neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[20]);
            neon_base2 = vld1q_f32(y[21]);
            neon_base3 = vld1q_f32(y[22]);
            neon_base4 = vld1q_f32(y[23]);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res21 = vmulq_f32(neon_base1, neon_base1);
            neon_res22 = vmulq_f32(neon_base2, neon_base2);
            neon_res23 = vmulq_f32(neon_base3, neon_base3);
            neon_res24 = vmulq_f32(neon_base4, neon_base4);
        }
        for (i = single_round; i < multi_round; i += single_round) {
            const float32x4_t neon_query = vld1q_f32(x + i);
            float32x4_t neon_base1 = vld1q_f32(y[0] + i);
            float32x4_t neon_base2 = vld1q_f32(y[1] + i);
            float32x4_t neon_base3 = vld1q_f32(y[2] + i);
            float32x4_t neon_base4 = vld1q_f32(y[3] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[4] + i);
            neon_base2 = vld1q_f32(y[5] + i);
            neon_base3 = vld1q_f32(y[6] + i);
            neon_base4 = vld1q_f32(y[7] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base1, neon_base1);
            neon_res6 = vmlaq_f32(neon_res6, neon_base2, neon_base2);
            neon_res7 = vmlaq_f32(neon_res7, neon_base3, neon_base3);
            neon_res8 = vmlaq_f32(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[8] + i);
            neon_base2 = vld1q_f32(y[9] + i);
            neon_base3 = vld1q_f32(y[10] + i);
            neon_base4 = vld1q_f32(y[11] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[12] + i);
            neon_base2 = vld1q_f32(y[13] + i);
            neon_base3 = vld1q_f32(y[14] + i);
            neon_base4 = vld1q_f32(y[15] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base1, neon_base1);
            neon_res14 = vmlaq_f32(neon_res14, neon_base2, neon_base2);
            neon_res15 = vmlaq_f32(neon_res15, neon_base3, neon_base3);
            neon_res16 = vmlaq_f32(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[16] + i);
            neon_base2 = vld1q_f32(y[17] + i);
            neon_base3 = vld1q_f32(y[18] + i);
            neon_base4 = vld1q_f32(y[19] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res17 = vmlaq_f32(neon_res17, neon_base1, neon_base1);
            neon_res18 = vmlaq_f32(neon_res18, neon_base2, neon_base2);
            neon_res19 = vmlaq_f32(neon_res19, neon_base3, neon_base3);
            neon_res20 = vmlaq_f32(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[20] + i);
            neon_base2 = vld1q_f32(y[21] + i);
            neon_base3 = vld1q_f32(y[22] + i);
            neon_base4 = vld1q_f32(y[23] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res21 = vmlaq_f32(neon_res21, neon_base1, neon_base1);
            neon_res22 = vmlaq_f32(neon_res22, neon_base2, neon_base2);
            neon_res23 = vmlaq_f32(neon_res23, neon_base3, neon_base3);
            neon_res24 = vmlaq_f32(neon_res24, neon_base4, neon_base4);
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
                const float32x4_t neon_query = vld1q_f32(x + j);
                float32x4_t neon_base1 = vld1q_f32(y[0] + j);
                float32x4_t neon_base2 = vld1q_f32(y[1] + j);
                float32x4_t neon_base3 = vld1q_f32(y[2] + j);
                float32x4_t neon_base4 = vld1q_f32(y[3] + j);
                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
                neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
                neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
                neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);

                neon_base1 = vld1q_f32(y[4] + j);
                neon_base2 = vld1q_f32(y[5] + j);
                neon_base3 = vld1q_f32(y[6] + j);
                neon_base4 = vld1q_f32(y[7] + j);
                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_res5 = vmlaq_f32(neon_res5, neon_base1, neon_base1);
                neon_res6 = vmlaq_f32(neon_res6, neon_base2, neon_base2);
                neon_res7 = vmlaq_f32(neon_res7, neon_base3, neon_base3);
                neon_res8 = vmlaq_f32(neon_res8, neon_base4, neon_base4);

                neon_base1 = vld1q_f32(y[8] + j);
                neon_base2 = vld1q_f32(y[9] + j);
                neon_base3 = vld1q_f32(y[10] + j);
                neon_base4 = vld1q_f32(y[11] + j);
                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
                neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
                neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
                neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);

                neon_base1 = vld1q_f32(y[12] + j);
                neon_base2 = vld1q_f32(y[13] + j);
                neon_base3 = vld1q_f32(y[14] + j);
                neon_base4 = vld1q_f32(y[15] + j);
                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_res13 = vmlaq_f32(neon_res13, neon_base1, neon_base1);
                neon_res14 = vmlaq_f32(neon_res14, neon_base2, neon_base2);
                neon_res15 = vmlaq_f32(neon_res15, neon_base3, neon_base3);
                neon_res16 = vmlaq_f32(neon_res16, neon_base4, neon_base4);

                neon_base1 = vld1q_f32(y[16] + j);
                neon_base2 = vld1q_f32(y[17] + j);
                neon_base3 = vld1q_f32(y[18] + j);
                neon_base4 = vld1q_f32(y[19] + j);
                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_res17 = vmlaq_f32(neon_res17, neon_base1, neon_base1);
                neon_res18 = vmlaq_f32(neon_res18, neon_base2, neon_base2);
                neon_res19 = vmlaq_f32(neon_res19, neon_base3, neon_base3);
                neon_res20 = vmlaq_f32(neon_res20, neon_base4, neon_base4);

                neon_base1 = vld1q_f32(y[20] + j);
                neon_base2 = vld1q_f32(y[21] + j);
                neon_base3 = vld1q_f32(y[22] + j);
                neon_base4 = vld1q_f32(y[23] + j);
                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_res21 = vmlaq_f32(neon_res21, neon_base1, neon_base1);
                neon_res22 = vmlaq_f32(neon_res22, neon_base2, neon_base2);
                neon_res23 = vmlaq_f32(neon_res23, neon_base3, neon_base3);
                neon_res24 = vmlaq_f32(neon_res24, neon_base4, neon_base4);
            }
        }
        for (; i <= d - single_round; i += single_round) {
            const float32x4_t neon_query = vld1q_f32(x + i);
            float32x4_t neon_base1 = vld1q_f32(y[0] + i);
            float32x4_t neon_base2 = vld1q_f32(y[1] + i);
            float32x4_t neon_base3 = vld1q_f32(y[2] + i);
            float32x4_t neon_base4 = vld1q_f32(y[3] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[4] + i);
            neon_base2 = vld1q_f32(y[5] + i);
            neon_base3 = vld1q_f32(y[6] + i);
            neon_base4 = vld1q_f32(y[7] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base1, neon_base1);
            neon_res6 = vmlaq_f32(neon_res6, neon_base2, neon_base2);
            neon_res7 = vmlaq_f32(neon_res7, neon_base3, neon_base3);
            neon_res8 = vmlaq_f32(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[8] + i);
            neon_base2 = vld1q_f32(y[9] + i);
            neon_base3 = vld1q_f32(y[10] + i);
            neon_base4 = vld1q_f32(y[11] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[12] + i);
            neon_base2 = vld1q_f32(y[13] + i);
            neon_base3 = vld1q_f32(y[14] + i);
            neon_base4 = vld1q_f32(y[15] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base1, neon_base1);
            neon_res14 = vmlaq_f32(neon_res14, neon_base2, neon_base2);
            neon_res15 = vmlaq_f32(neon_res15, neon_base3, neon_base3);
            neon_res16 = vmlaq_f32(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[16] + i);
            neon_base2 = vld1q_f32(y[17] + i);
            neon_base3 = vld1q_f32(y[18] + i);
            neon_base4 = vld1q_f32(y[19] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res17 = vmlaq_f32(neon_res17, neon_base1, neon_base1);
            neon_res18 = vmlaq_f32(neon_res18, neon_base2, neon_base2);
            neon_res19 = vmlaq_f32(neon_res19, neon_base3, neon_base3);
            neon_res20 = vmlaq_f32(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[20] + i);
            neon_base2 = vld1q_f32(y[21] + i);
            neon_base3 = vld1q_f32(y[22] + i);
            neon_base4 = vld1q_f32(y[23] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res21 = vmlaq_f32(neon_res21, neon_base1, neon_base1);
            neon_res22 = vmlaq_f32(neon_res22, neon_base2, neon_base2);
            neon_res23 = vmlaq_f32(neon_res23, neon_base3, neon_base3);
            neon_res24 = vmlaq_f32(neon_res24, neon_base4, neon_base4);
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
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y[0]);
        float32x4_t neon_base2 = vld1q_f32(y[1]);
        float32x4_t neon_base3 = vld1q_f32(y[2]);
        float32x4_t neon_base4 = vld1q_f32(y[3]);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y[4]);
        neon_base2 = vld1q_f32(y[5]);
        neon_base3 = vld1q_f32(y[6]);
        neon_base4 = vld1q_f32(y[7]);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res6 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res7 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res8 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y[8]);
        neon_base2 = vld1q_f32(y[9]);
        neon_base3 = vld1q_f32(y[10]);
        neon_base4 = vld1q_f32(y[11]);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y[12]);
        neon_base2 = vld1q_f32(y[13]);
        neon_base3 = vld1q_f32(y[14]);
        neon_base4 = vld1q_f32(y[15]);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res13 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res14 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res15 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res16 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y[16]);
        neon_base2 = vld1q_f32(y[17]);
        neon_base3 = vld1q_f32(y[18]);
        neon_base4 = vld1q_f32(y[19]);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res17 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res18 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res19 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res20 = vmulq_f32(neon_base4, neon_base4);

        neon_base1 = vld1q_f32(y[20]);
        neon_base2 = vld1q_f32(y[21]);
        neon_base3 = vld1q_f32(y[22]);
        neon_base4 = vld1q_f32(y[23]);
        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        float32x4_t neon_res21 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res22 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res23 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res24 = vmulq_f32(neon_base4, neon_base4);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y[0] + i);
            neon_base2 = vld1q_f32(y[1] + i);
            neon_base3 = vld1q_f32(y[2] + i);
            neon_base4 = vld1q_f32(y[3] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[4] + i);
            neon_base2 = vld1q_f32(y[5] + i);
            neon_base3 = vld1q_f32(y[6] + i);
            neon_base4 = vld1q_f32(y[7] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base1, neon_base1);
            neon_res6 = vmlaq_f32(neon_res6, neon_base2, neon_base2);
            neon_res7 = vmlaq_f32(neon_res7, neon_base3, neon_base3);
            neon_res8 = vmlaq_f32(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[8] + i);
            neon_base2 = vld1q_f32(y[9] + i);
            neon_base3 = vld1q_f32(y[10] + i);
            neon_base4 = vld1q_f32(y[11] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[12] + i);
            neon_base2 = vld1q_f32(y[13] + i);
            neon_base3 = vld1q_f32(y[14] + i);
            neon_base4 = vld1q_f32(y[15] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base1, neon_base1);
            neon_res14 = vmlaq_f32(neon_res14, neon_base2, neon_base2);
            neon_res15 = vmlaq_f32(neon_res15, neon_base3, neon_base3);
            neon_res16 = vmlaq_f32(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[16] + i);
            neon_base2 = vld1q_f32(y[17] + i);
            neon_base3 = vld1q_f32(y[18] + i);
            neon_base4 = vld1q_f32(y[19] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res17 = vmlaq_f32(neon_res17, neon_base1, neon_base1);
            neon_res18 = vmlaq_f32(neon_res18, neon_base2, neon_base2);
            neon_res19 = vmlaq_f32(neon_res19, neon_base3, neon_base3);
            neon_res20 = vmlaq_f32(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f32(y[20] + i);
            neon_base2 = vld1q_f32(y[21] + i);
            neon_base3 = vld1q_f32(y[22] + i);
            neon_base4 = vld1q_f32(y[23] + i);
            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_res21 = vmlaq_f32(neon_res21, neon_base1, neon_base1);
            neon_res22 = vmlaq_f32(neon_res22, neon_base2, neon_base2);
            neon_res23 = vmlaq_f32(neon_res23, neon_base3, neon_base3);
            neon_res24 = vmlaq_f32(neon_res24, neon_base4, neon_base4);
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
#endif

static void L2sqrBatch16ByIdx(const float *x, const float *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 16;
    if (LIKELY(d >= multi_round)) {
        float32x4_t neon_res1 = vdupq_n_f32(0.0);
        float32x4_t neon_res2 = vdupq_n_f32(0.0);
        float32x4_t neon_res3 = vdupq_n_f32(0.0);
        float32x4_t neon_res4 = vdupq_n_f32(0.0);
        float32x4_t neon_res5 = vdupq_n_f32(0.0);
        float32x4_t neon_res6 = vdupq_n_f32(0.0);
        float32x4_t neon_res7 = vdupq_n_f32(0.0);
        float32x4_t neon_res8 = vdupq_n_f32(0.0);
        float32x4_t neon_res9 = vdupq_n_f32(0.0);
        float32x4_t neon_res10 = vdupq_n_f32(0.0);
        float32x4_t neon_res11 = vdupq_n_f32(0.0);
        float32x4_t neon_res12 = vdupq_n_f32(0.0);
        float32x4_t neon_res13 = vdupq_n_f32(0.0);
        float32x4_t neon_res14 = vdupq_n_f32(0.0);
        float32x4_t neon_res15 = vdupq_n_f32(0.0);
        float32x4_t neon_res16 = vdupq_n_f32(0.0);
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
                const float32x4_t neon_query = vld1q_f32(x + i + j);
                float32x4_t neon_base1 = vld1q_f32(y[0] + i + j);
                float32x4_t neon_base2 = vld1q_f32(y[1] + i + j);
                float32x4_t neon_base3 = vld1q_f32(y[2] + i + j);
                float32x4_t neon_base4 = vld1q_f32(y[3] + i + j);
                float32x4_t neon_base5 = vld1q_f32(y[4] + i + j);
                float32x4_t neon_base6 = vld1q_f32(y[5] + i + j);
                float32x4_t neon_base7 = vld1q_f32(y[6] + i + j);
                float32x4_t neon_base8 = vld1q_f32(y[7] + i + j);

                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_base5 = vsubq_f32(neon_base5, neon_query);
                neon_base6 = vsubq_f32(neon_base6, neon_query);
                neon_base7 = vsubq_f32(neon_base7, neon_query);
                neon_base8 = vsubq_f32(neon_base8, neon_query);

                neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
                neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
                neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
                neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
                neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
                neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
                neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
                neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);

                neon_base1 = vld1q_f32(y[8] + i + j);
                neon_base2 = vld1q_f32(y[9] + i + j);
                neon_base3 = vld1q_f32(y[10] + i + j);
                neon_base4 = vld1q_f32(y[11] + i + j);
                neon_base5 = vld1q_f32(y[12] + i + j);
                neon_base6 = vld1q_f32(y[13] + i + j);
                neon_base7 = vld1q_f32(y[14] + i + j);
                neon_base8 = vld1q_f32(y[15] + i + j);

                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_base5 = vsubq_f32(neon_base5, neon_query);
                neon_base6 = vsubq_f32(neon_base6, neon_query);
                neon_base7 = vsubq_f32(neon_base7, neon_query);
                neon_base8 = vsubq_f32(neon_base8, neon_query);

                neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
                neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
                neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
                neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);
                neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_base5);
                neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_base6);
                neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_base7);
                neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_base8);
            }
        }
        for (; i <= d - single_round; i += single_round) {
            const float32x4_t neon_query = vld1q_f32(x + i);
            float32x4_t neon_base1 = vld1q_f32(y[0] + i);
            float32x4_t neon_base2 = vld1q_f32(y[1] + i);
            float32x4_t neon_base3 = vld1q_f32(y[2] + i);
            float32x4_t neon_base4 = vld1q_f32(y[3] + i);
            float32x4_t neon_base5 = vld1q_f32(y[4] + i);
            float32x4_t neon_base6 = vld1q_f32(y[5] + i);
            float32x4_t neon_base7 = vld1q_f32(y[6] + i);
            float32x4_t neon_base8 = vld1q_f32(y[7] + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f32(y[8] + i);
            neon_base2 = vld1q_f32(y[9] + i);
            neon_base3 = vld1q_f32(y[10] + i);
            neon_base4 = vld1q_f32(y[11] + i);
            neon_base5 = vld1q_f32(y[12] + i);
            neon_base6 = vld1q_f32(y[13] + i);
            neon_base7 = vld1q_f32(y[14] + i);
            neon_base8 = vld1q_f32(y[15] + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_base5);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_base6);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_base7);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_base8);
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
    } else if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);

        float32x4_t neon_base1 = vld1q_f32(y[0]);
        float32x4_t neon_base2 = vld1q_f32(y[1]);
        float32x4_t neon_base3 = vld1q_f32(y[2]);
        float32x4_t neon_base4 = vld1q_f32(y[3]);
        float32x4_t neon_base5 = vld1q_f32(y[4]);
        float32x4_t neon_base6 = vld1q_f32(y[5]);
        float32x4_t neon_base7 = vld1q_f32(y[6]);
        float32x4_t neon_base8 = vld1q_f32(y[7]);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_base8);

        neon_base1 = vld1q_f32(y[8]);
        neon_base2 = vld1q_f32(y[9]);
        neon_base3 = vld1q_f32(y[10]);
        neon_base4 = vld1q_f32(y[11]);
        neon_base5 = vld1q_f32(y[12]);
        neon_base6 = vld1q_f32(y[13]);
        neon_base7 = vld1q_f32(y[14]);
        neon_base8 = vld1q_f32(y[15]);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res13 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res14 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res15 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res16 = vmulq_f32(neon_base8, neon_base8);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y[0] + i);
            neon_base2 = vld1q_f32(y[1] + i);
            neon_base3 = vld1q_f32(y[2] + i);
            neon_base4 = vld1q_f32(y[3] + i);
            neon_base5 = vld1q_f32(y[4] + i);
            neon_base6 = vld1q_f32(y[5] + i);
            neon_base7 = vld1q_f32(y[6] + i);
            neon_base8 = vld1q_f32(y[7] + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f32(y[8] + i);
            neon_base2 = vld1q_f32(y[9] + i);
            neon_base3 = vld1q_f32(y[10] + i);
            neon_base4 = vld1q_f32(y[11] + i);
            neon_base5 = vld1q_f32(y[12] + i);
            neon_base6 = vld1q_f32(y[13] + i);
            neon_base7 = vld1q_f32(y[14] + i);
            neon_base8 = vld1q_f32(y[15] + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_base1);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_base2);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_base3);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_base4);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_base5);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_base6);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_base7);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_base8);
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
        for (int i = 0; i < 16; i++) {
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

static void L2sqrBatch8ByIdx(const float *x, const float *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 16;
    if (LIKELY(d >= multi_round)) {
        float32x4_t neon_res1 = vdupq_n_f32(0.0);
        float32x4_t neon_res2 = vdupq_n_f32(0.0);
        float32x4_t neon_res3 = vdupq_n_f32(0.0);
        float32x4_t neon_res4 = vdupq_n_f32(0.0);
        float32x4_t neon_res5 = vdupq_n_f32(0.0);
        float32x4_t neon_res6 = vdupq_n_f32(0.0);
        float32x4_t neon_res7 = vdupq_n_f32(0.0);
        float32x4_t neon_res8 = vdupq_n_f32(0.0);
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
                const float32x4_t neon_query = vld1q_f32(x + i + j);
                float32x4_t neon_base1 = vld1q_f32(y[0] + i + j);
                float32x4_t neon_base2 = vld1q_f32(y[1] + i + j);
                float32x4_t neon_base3 = vld1q_f32(y[2] + i + j);
                float32x4_t neon_base4 = vld1q_f32(y[3] + i + j);
                float32x4_t neon_base5 = vld1q_f32(y[4] + i + j);
                float32x4_t neon_base6 = vld1q_f32(y[5] + i + j);
                float32x4_t neon_base7 = vld1q_f32(y[6] + i + j);
                float32x4_t neon_base8 = vld1q_f32(y[7] + i + j);

                neon_base1 = vsubq_f32(neon_base1, neon_query);
                neon_base2 = vsubq_f32(neon_base2, neon_query);
                neon_base3 = vsubq_f32(neon_base3, neon_query);
                neon_base4 = vsubq_f32(neon_base4, neon_query);
                neon_base5 = vsubq_f32(neon_base5, neon_query);
                neon_base6 = vsubq_f32(neon_base6, neon_query);
                neon_base7 = vsubq_f32(neon_base7, neon_query);
                neon_base8 = vsubq_f32(neon_base8, neon_query);

                neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
                neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
                neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
                neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
                neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
                neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
                neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
                neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);
            }
        }
        for (; i <= d - single_round; i += single_round) {
            const float32x4_t neon_query = vld1q_f32(x + i);
            float32x4_t neon_base1 = vld1q_f32(y[0] + i);
            float32x4_t neon_base2 = vld1q_f32(y[1] + i);
            float32x4_t neon_base3 = vld1q_f32(y[2] + i);
            float32x4_t neon_base4 = vld1q_f32(y[3] + i);
            float32x4_t neon_base5 = vld1q_f32(y[4] + i);
            float32x4_t neon_base6 = vld1q_f32(y[5] + i);
            float32x4_t neon_base7 = vld1q_f32(y[6] + i);
            float32x4_t neon_base8 = vld1q_f32(y[7] + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);

        float32x4_t neon_base1 = vld1q_f32(y[0]);
        float32x4_t neon_base2 = vld1q_f32(y[1]);
        float32x4_t neon_base3 = vld1q_f32(y[2]);
        float32x4_t neon_base4 = vld1q_f32(y[3]);
        float32x4_t neon_base5 = vld1q_f32(y[4]);
        float32x4_t neon_base6 = vld1q_f32(y[5]);
        float32x4_t neon_base7 = vld1q_f32(y[6]);
        float32x4_t neon_base8 = vld1q_f32(y[7]);

        neon_base1 = vsubq_f32(neon_base1, neon_query);
        neon_base2 = vsubq_f32(neon_base2, neon_query);
        neon_base3 = vsubq_f32(neon_base3, neon_query);
        neon_base4 = vsubq_f32(neon_base4, neon_query);
        neon_base5 = vsubq_f32(neon_base5, neon_query);
        neon_base6 = vsubq_f32(neon_base6, neon_query);
        neon_base7 = vsubq_f32(neon_base7, neon_query);
        neon_base8 = vsubq_f32(neon_base8, neon_query);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_base1);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_base2);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_base3);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_base4);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_base5);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_base6);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_base7);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_base8);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y[0] + i);
            neon_base2 = vld1q_f32(y[1] + i);
            neon_base3 = vld1q_f32(y[2] + i);
            neon_base4 = vld1q_f32(y[3] + i);
            neon_base5 = vld1q_f32(y[4] + i);
            neon_base6 = vld1q_f32(y[5] + i);
            neon_base7 = vld1q_f32(y[6] + i);
            neon_base8 = vld1q_f32(y[7] + i);

            neon_base1 = vsubq_f32(neon_base1, neon_query);
            neon_base2 = vsubq_f32(neon_base2, neon_query);
            neon_base3 = vsubq_f32(neon_base3, neon_query);
            neon_base4 = vsubq_f32(neon_base4, neon_query);
            neon_base5 = vsubq_f32(neon_base5, neon_query);
            neon_base6 = vsubq_f32(neon_base6, neon_query);
            neon_base7 = vsubq_f32(neon_base7, neon_query);
            neon_base8 = vsubq_f32(neon_base8, neon_query);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_base1);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_base2);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_base3);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_base4);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_base5);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_base6);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_base7);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_base8);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
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

static void L2sqrBatch4ByIdx(const float *x, const float *__restrict *y, const size_t d, float *dis)
{
    constexpr size_t single_round = 4;
    size_t i;
    if (LIKELY(d >= single_round)) {
        float32x4_t b = vld1q_f32(x);

        float32x4_t q0 = vld1q_f32(y[0]);
        float32x4_t q1 = vld1q_f32(y[1]);
        float32x4_t q2 = vld1q_f32(y[2]);
        float32x4_t q3 = vld1q_f32(y[3]);

        q0 = vsubq_f32(q0, b);
        q1 = vsubq_f32(q1, b);
        q2 = vsubq_f32(q2, b);
        q3 = vsubq_f32(q3, b);

        float32x4_t res0 = vmulq_f32(q0, q0);
        float32x4_t res1 = vmulq_f32(q1, q1);
        float32x4_t res2 = vmulq_f32(q2, q2);
        float32x4_t res3 = vmulq_f32(q3, q3);

        for (i = single_round; i <= d - single_round; i += single_round) {
            b = vld1q_f32(x + i);

            q0 = vld1q_f32(y[0] + i);
            q1 = vld1q_f32(y[1] + i);
            q2 = vld1q_f32(y[2] + i);
            q3 = vld1q_f32(y[3] + i);

            q0 = vsubq_f32(q0, b);
            q1 = vsubq_f32(q1, b);
            q2 = vsubq_f32(q2, b);
            q3 = vsubq_f32(q3, b);

            res0 = vmlaq_f32(res0, q0, q0);
            res1 = vmlaq_f32(res1, q1, q1);
            res2 = vmlaq_f32(res2, q2, q2);
            res3 = vmlaq_f32(res3, q3, q3);
        }
        dis[0] = vaddvq_f32(res0);
        dis[1] = vaddvq_f32(res1);
        dis[2] = vaddvq_f32(res2);
        dis[3] = vaddvq_f32(res3);
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (d > i) {
        float q0 = x[i] - *(y[0] + i);
        float q1 = x[i] - *(y[1] + i);
        float q2 = x[i] - *(y[2] + i);
        float q3 = x[i] - *(y[3] + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        for (i++; i < d; ++i) {
            float q0 = x[i] - *(y[0] + i);
            float q1 = x[i] - *(y[1] + i);
            float q2 = x[i] - *(y[2] + i);
            float q3 = x[i] - *(y[3] + i);
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

static void L2sqrBatch2ByIdx(const float *x, const float *__restrict y0, const float *__restrict y1, const size_t d,
                             float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 8;

    if (LIKELY(d >= multi_round)) {
        float32x4_t x_0 = vld1q_f32(x);
        float32x4_t x_1 = vld1q_f32(x + 4);

        float32x4_t y0_0 = vld1q_f32(y0);
        float32x4_t y0_1 = vld1q_f32(y0 + 4);
        float32x4_t y1_0 = vld1q_f32(y1);
        float32x4_t y1_1 = vld1q_f32(y1 + 4);

        float32x4_t d0_0 = vsubq_f32(x_0, y0_0);
        d0_0 = vmulq_f32(d0_0, d0_0);
        float32x4_t d0_1 = vsubq_f32(x_1, y0_1);
        d0_1 = vmulq_f32(d0_1, d0_1);
        float32x4_t d1_0 = vsubq_f32(x_0, y1_0);
        d1_0 = vmulq_f32(d1_0, d1_0);
        float32x4_t d1_1 = vsubq_f32(x_1, y1_1);
        d1_1 = vmulq_f32(d1_1, d1_1);

        for (i = multi_round; i <= d - multi_round; i += multi_round) {
            x_0 = vld1q_f32(x + i);
            y0_0 = vld1q_f32(y0 + i);
            y1_0 = vld1q_f32(y1 + i);
            const float32x4_t q0_0 = vsubq_f32(x_0, y0_0);
            const float32x4_t q1_0 = vsubq_f32(x_0, y1_0);
            d0_0 = vmlaq_f32(d0_0, q0_0, q0_0);
            d1_0 = vmlaq_f32(d1_0, q1_0, q1_0);

            x_1 = vld1q_f32(x + i + 4);
            y0_1 = vld1q_f32(y0 + i + 4);
            y1_1 = vld1q_f32(y1 + i + 4);
            const float32x4_t q0_1 = vsubq_f32(x_1, y0_1);
            const float32x4_t q1_1 = vsubq_f32(x_1, y1_1);
            d0_1 = vmlaq_f32(d0_1, q0_1, q0_1);
            d1_1 = vmlaq_f32(d1_1, q1_1, q1_1);
        }

        for (; i <= d - single_round; i += single_round) {
            x_0 = vld1q_f32(x + i);
            y0_0 = vld1q_f32(y0 + i);
            y1_0 = vld1q_f32(y1 + i);
            const float32x4_t q0_0 = vsubq_f32(x_0, y0_0);
            const float32x4_t q1_0 = vsubq_f32(x_0, y1_0);
            d0_0 = vmlaq_f32(d0_0, q0_0, q0_0);
            d1_0 = vmlaq_f32(d1_0, q1_0, q1_0);
        }

        d0_0 = vaddq_f32(d0_0, d0_1);
        d1_0 = vaddq_f32(d1_0, d1_1);
        dis[0] = vaddvq_f32(d0_0);
        dis[1] = vaddvq_f32(d1_0);
    } else if (d >= single_round) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t y8_0 = vld1q_f32(y0);
        float32x4_t y8_1 = vld1q_f32(y1);

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        float32x4_t d8_1 = vsubq_f32(x8_0, y8_1);
        d8_1 = vmulq_f32(d8_1, d8_1);
        for (i = single_round; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x);
            y8_0 = vld1q_f32(y0);
            y8_1 = vld1q_f32(y1);

            float32x4_t q0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q0, q0);
            float32x4_t q1 = vsubq_f32(x8_0, y8_1);
            d8_1 = vmlaq_f32(d8_1, q1, q1);
        }
        dis[0] = vaddvq_f32(d8_0);
        dis[1] = vaddvq_f32(d8_1);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp0 = x[i] - y0[i];
        const float tmp1 = x[i] - y1[i];
        dis[0] += tmp0 * tmp0;
        dis[1] += tmp1 * tmp1;
    }
}

void L2sqrNyByIdx(float *dis, const float *x, const float *y, const int64_t *ids, size_t d, size_t ny)
{
    size_t i = 0;
    const float *__restrict listy[24];

    for (; i + 24 <= ny; i += 24) {
        prefetch_L1(x);
        listy[0] = (const float *)(y + *(ids + i) * d);
        prefetch_Lx(listy[0]);
        listy[1] = (const float *)(y + *(ids + i + 1) * d);
        prefetch_Lx(listy[1]);
        listy[2] = (const float *)(y + *(ids + i + 2) * d);
        prefetch_Lx(listy[2]);
        listy[3] = (const float *)(y + *(ids + i + 3) * d);
        prefetch_Lx(listy[3]);
        listy[4] = (const float *)(y + *(ids + i + 4) * d);
        prefetch_Lx(listy[4]);
        listy[5] = (const float *)(y + *(ids + i + 5) * d);
        prefetch_Lx(listy[5]);
        listy[6] = (const float *)(y + *(ids + i + 6) * d);
        prefetch_Lx(listy[6]);
        listy[7] = (const float *)(y + *(ids + i + 7) * d);
        prefetch_Lx(listy[7]);
        listy[8] = (const float *)(y + *(ids + i + 8) * d);
        prefetch_Lx(listy[8]);
        listy[9] = (const float *)(y + *(ids + i + 9) * d);
        prefetch_Lx(listy[9]);
        listy[10] = (const float *)(y + *(ids + i + 10) * d);
        prefetch_Lx(listy[10]);
        listy[11] = (const float *)(y + *(ids + i + 11) * d);
        prefetch_Lx(listy[11]);
        listy[12] = (const float *)(y + *(ids + i + 12) * d);
        prefetch_Lx(listy[12]);
        listy[13] = (const float *)(y + *(ids + i + 13) * d);
        prefetch_Lx(listy[13]);
        listy[14] = (const float *)(y + *(ids + i + 14) * d);
        prefetch_Lx(listy[14]);
        listy[15] = (const float *)(y + *(ids + i + 15) * d);
        prefetch_Lx(listy[15]);
        listy[16] = (const float *)(y + *(ids + i + 16) * d);
        prefetch_Lx(listy[16]);
        listy[17] = (const float *)(y + *(ids + i + 17) * d);
        prefetch_Lx(listy[17]);
        listy[18] = (const float *)(y + *(ids + i + 18) * d);
        prefetch_Lx(listy[18]);
        listy[19] = (const float *)(y + *(ids + i + 19) * d);
        prefetch_Lx(listy[19]);
        listy[20] = (const float *)(y + *(ids + i + 20) * d);
        prefetch_Lx(listy[20]);
        listy[21] = (const float *)(y + *(ids + i + 21) * d);
        prefetch_Lx(listy[21]);
        listy[22] = (const float *)(y + *(ids + i + 22) * d);
        prefetch_Lx(listy[22]);
        listy[23] = (const float *)(y + *(ids + i + 23) * d);
        prefetch_Lx(listy[23]);
        L2sqrBatch24ByIdx(x, listy, d, dis + i);
    }
    if (i + 16 <= ny) {
        prefetch_L1(x);
        listy[0] = (const float *)(y + *(ids + i) * d);
        prefetch_Lx(listy[0]);
        listy[1] = (const float *)(y + *(ids + i + 1) * d);
        prefetch_Lx(listy[1]);
        listy[2] = (const float *)(y + *(ids + i + 2) * d);
        prefetch_Lx(listy[2]);
        listy[3] = (const float *)(y + *(ids + i + 3) * d);
        prefetch_Lx(listy[3]);
        listy[4] = (const float *)(y + *(ids + i + 4) * d);
        prefetch_Lx(listy[4]);
        listy[5] = (const float *)(y + *(ids + i + 5) * d);
        prefetch_Lx(listy[5]);
        listy[6] = (const float *)(y + *(ids + i + 6) * d);
        prefetch_Lx(listy[6]);
        listy[7] = (const float *)(y + *(ids + i + 7) * d);
        prefetch_Lx(listy[7]);
        listy[8] = (const float *)(y + *(ids + i + 8) * d);
        prefetch_Lx(listy[8]);
        listy[9] = (const float *)(y + *(ids + i + 9) * d);
        prefetch_Lx(listy[9]);
        listy[10] = (const float *)(y + *(ids + i + 10) * d);
        prefetch_Lx(listy[10]);
        listy[11] = (const float *)(y + *(ids + i + 11) * d);
        prefetch_Lx(listy[11]);
        listy[12] = (const float *)(y + *(ids + i + 12) * d);
        prefetch_Lx(listy[12]);
        listy[13] = (const float *)(y + *(ids + i + 13) * d);
        prefetch_Lx(listy[13]);
        listy[14] = (const float *)(y + *(ids + i + 14) * d);
        prefetch_Lx(listy[14]);
        listy[15] = (const float *)(y + *(ids + i + 15) * d);
        prefetch_Lx(listy[15]);
        L2sqrBatch16ByIdx(x, listy, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        prefetch_L1(x);
        listy[0] = (const float *)(y + *(ids + i) * d);
        prefetch_Lx(listy[0]);
        listy[1] = (const float *)(y + *(ids + i + 1) * d);
        prefetch_Lx(listy[1]);
        listy[2] = (const float *)(y + *(ids + i + 2) * d);
        prefetch_Lx(listy[2]);
        listy[3] = (const float *)(y + *(ids + i + 3) * d);
        prefetch_Lx(listy[3]);
        listy[4] = (const float *)(y + *(ids + i + 4) * d);
        prefetch_Lx(listy[4]);
        listy[5] = (const float *)(y + *(ids + i + 5) * d);
        prefetch_Lx(listy[5]);
        listy[6] = (const float *)(y + *(ids + i + 6) * d);
        prefetch_Lx(listy[6]);
        listy[7] = (const float *)(y + *(ids + i + 7) * d);
        prefetch_Lx(listy[7]);
        L2sqrBatch8ByIdx(x, listy, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        listy[0] = (const float *)(y + *(ids + i) * d);
        listy[1] = (const float *)(y + *(ids + i + 1) * d);
        listy[2] = (const float *)(y + *(ids + i + 2) * d);
        listy[3] = (const float *)(y + *(ids + i + 3) * d);
        L2sqrBatch4ByIdx(x, listy, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        const float *y0 = y + *(ids + i) * d;
        const float *y1 = y + *(ids + i + 1) * d;
        L2sqrBatch2ByIdx(x, y0, y1, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        L2sqrSingle(x, y + d * ids[i], d, &dis[i]);
    }
}

static void IPBatch16ByIdx(const float *x, const float *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 32;
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
        float32x4_t neon_res1, neon_res2, neon_res3, neon_res4;
        float32x4_t neon_res5, neon_res6, neon_res7, neon_res8;
        float32x4_t neon_res9, neon_res10, neon_res11, neon_res12;
        float32x4_t neon_res13, neon_res14, neon_res15, neon_res16;
        {
            const float32x4_t neon_query = vld1q_f32(x);
            float32x4_t neon_base1 = vld1q_f32(y[0]);
            float32x4_t neon_base2 = vld1q_f32(y[1]);
            float32x4_t neon_base3 = vld1q_f32(y[2]);
            float32x4_t neon_base4 = vld1q_f32(y[3]);
            float32x4_t neon_base5 = vld1q_f32(y[4]);
            float32x4_t neon_base6 = vld1q_f32(y[5]);
            float32x4_t neon_base7 = vld1q_f32(y[6]);
            float32x4_t neon_base8 = vld1q_f32(y[7]);

            neon_res1 = vmulq_f32(neon_base1, neon_query);
            neon_res2 = vmulq_f32(neon_base2, neon_query);
            neon_res3 = vmulq_f32(neon_base3, neon_query);
            neon_res4 = vmulq_f32(neon_base4, neon_query);
            neon_res5 = vmulq_f32(neon_base5, neon_query);
            neon_res6 = vmulq_f32(neon_base6, neon_query);
            neon_res7 = vmulq_f32(neon_base7, neon_query);
            neon_res8 = vmulq_f32(neon_base8, neon_query);

            neon_base1 = vld1q_f32(y[8]);
            neon_base2 = vld1q_f32(y[9]);
            neon_base3 = vld1q_f32(y[10]);
            neon_base4 = vld1q_f32(y[11]);
            neon_base5 = vld1q_f32(y[12]);
            neon_base6 = vld1q_f32(y[13]);
            neon_base7 = vld1q_f32(y[14]);
            neon_base8 = vld1q_f32(y[15]);

            neon_res9 = vmulq_f32(neon_base1, neon_query);
            neon_res10 = vmulq_f32(neon_base2, neon_query);
            neon_res11 = vmulq_f32(neon_base3, neon_query);
            neon_res12 = vmulq_f32(neon_base4, neon_query);
            neon_res13 = vmulq_f32(neon_base5, neon_query);
            neon_res14 = vmulq_f32(neon_base6, neon_query);
            neon_res15 = vmulq_f32(neon_base7, neon_query);
            neon_res16 = vmulq_f32(neon_base8, neon_query);
        }
        for (i = single_round; i < multi_round; i += single_round) {
            const float32x4_t neon_query = vld1q_f32(x + i);
            float32x4_t neon_base1 = vld1q_f32(y[0] + i);
            float32x4_t neon_base2 = vld1q_f32(y[1] + i);
            float32x4_t neon_base3 = vld1q_f32(y[2] + i);
            float32x4_t neon_base4 = vld1q_f32(y[3] + i);
            float32x4_t neon_base5 = vld1q_f32(y[4] + i);
            float32x4_t neon_base6 = vld1q_f32(y[5] + i);
            float32x4_t neon_base7 = vld1q_f32(y[6] + i);
            float32x4_t neon_base8 = vld1q_f32(y[7] + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_f32(y[8] + i);
            neon_base2 = vld1q_f32(y[9] + i);
            neon_base3 = vld1q_f32(y[10] + i);
            neon_base4 = vld1q_f32(y[11] + i);
            neon_base5 = vld1q_f32(y[12] + i);
            neon_base6 = vld1q_f32(y[13] + i);
            neon_base7 = vld1q_f32(y[14] + i);
            neon_base8 = vld1q_f32(y[15] + i);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_query);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_query);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_query);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_query);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_query);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_query);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_query);
        }
        for (; i < d - multi_round; i += multi_round) {
            prefetch_L1(x + multi_round + i);
            prefetch_Lx(y[0] + multi_round + i);
            prefetch_Lx(y[1] + multi_round + i);
            prefetch_Lx(y[2] + multi_round + i);
            prefetch_Lx(y[3] + multi_round + i);
            prefetch_Lx(y[4] + multi_round + i);
            prefetch_Lx(y[5] + multi_round + i);
            prefetch_Lx(y[6] + multi_round + i);
            prefetch_Lx(y[7] + multi_round + i);
            prefetch_Lx(y[8] + multi_round + i);
            prefetch_Lx(y[9] + multi_round + i);
            prefetch_Lx(y[10] + multi_round + i);
            prefetch_Lx(y[11] + multi_round + i);
            prefetch_Lx(y[12] + multi_round + i);
            prefetch_Lx(y[13] + multi_round + i);
            prefetch_Lx(y[14] + multi_round + i);
            prefetch_Lx(y[15] + multi_round + i);
            for (size_t j = 0; j < multi_round; j += single_round) {
                const float32x4_t neon_query = vld1q_f32(x + i + j);
                float32x4_t neon_base1 = vld1q_f32(y[0] + i + j);
                float32x4_t neon_base2 = vld1q_f32(y[1] + i + j);
                float32x4_t neon_base3 = vld1q_f32(y[2] + i + j);
                float32x4_t neon_base4 = vld1q_f32(y[3] + i + j);
                float32x4_t neon_base5 = vld1q_f32(y[4] + i + j);
                float32x4_t neon_base6 = vld1q_f32(y[5] + i + j);
                float32x4_t neon_base7 = vld1q_f32(y[6] + i + j);
                float32x4_t neon_base8 = vld1q_f32(y[7] + i + j);

                neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
                neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
                neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
                neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
                neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
                neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
                neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
                neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);

                neon_base1 = vld1q_f32(y[8] + i + j);
                neon_base2 = vld1q_f32(y[9] + i + j);
                neon_base3 = vld1q_f32(y[10] + i + j);
                neon_base4 = vld1q_f32(y[11] + i + j);
                neon_base5 = vld1q_f32(y[12] + i + j);
                neon_base6 = vld1q_f32(y[13] + i + j);
                neon_base7 = vld1q_f32(y[14] + i + j);
                neon_base8 = vld1q_f32(y[15] + i + j);

                neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_query);
                neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_query);
                neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_query);
                neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_query);
                neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_query);
                neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_query);
                neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_query);
                neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_query);
            }
        }
        for (; i <= d - single_round; i += single_round) {
            const float32x4_t neon_query = vld1q_f32(x + i);
            float32x4_t neon_base1 = vld1q_f32(y[0] + i);
            float32x4_t neon_base2 = vld1q_f32(y[1] + i);
            float32x4_t neon_base3 = vld1q_f32(y[2] + i);
            float32x4_t neon_base4 = vld1q_f32(y[3] + i);
            float32x4_t neon_base5 = vld1q_f32(y[4] + i);
            float32x4_t neon_base6 = vld1q_f32(y[5] + i);
            float32x4_t neon_base7 = vld1q_f32(y[6] + i);
            float32x4_t neon_base8 = vld1q_f32(y[7] + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_f32(y[8] + i);
            neon_base2 = vld1q_f32(y[9] + i);
            neon_base3 = vld1q_f32(y[10] + i);
            neon_base4 = vld1q_f32(y[11] + i);
            neon_base5 = vld1q_f32(y[12] + i);
            neon_base6 = vld1q_f32(y[13] + i);
            neon_base7 = vld1q_f32(y[14] + i);
            neon_base8 = vld1q_f32(y[15] + i);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_query);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_query);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_query);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_query);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_query);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_query);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_query);
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
    } else if (d >= single_round) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y[0]);
        float32x4_t neon_base2 = vld1q_f32(y[1]);
        float32x4_t neon_base3 = vld1q_f32(y[2]);
        float32x4_t neon_base4 = vld1q_f32(y[3]);
        float32x4_t neon_base5 = vld1q_f32(y[4]);
        float32x4_t neon_base6 = vld1q_f32(y[5]);
        float32x4_t neon_base7 = vld1q_f32(y[6]);
        float32x4_t neon_base8 = vld1q_f32(y[7]);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_query);

        neon_base1 = vld1q_f32(y[8]);
        neon_base2 = vld1q_f32(y[9]);
        neon_base3 = vld1q_f32(y[10]);
        neon_base4 = vld1q_f32(y[11]);
        neon_base5 = vld1q_f32(y[12]);
        neon_base6 = vld1q_f32(y[13]);
        neon_base7 = vld1q_f32(y[14]);
        neon_base8 = vld1q_f32(y[15]);

        float32x4_t neon_res9 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res10 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res11 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res12 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res13 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res14 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res15 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res16 = vmulq_f32(neon_base8, neon_query);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y[0] + i);
            neon_base2 = vld1q_f32(y[1] + i);
            neon_base3 = vld1q_f32(y[2] + i);
            neon_base4 = vld1q_f32(y[3] + i);
            neon_base5 = vld1q_f32(y[4] + i);
            neon_base6 = vld1q_f32(y[5] + i);
            neon_base7 = vld1q_f32(y[6] + i);
            neon_base8 = vld1q_f32(y[7] + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_f32(y[8] + i);
            neon_base2 = vld1q_f32(y[9] + i);
            neon_base3 = vld1q_f32(y[10] + i);
            neon_base4 = vld1q_f32(y[11] + i);
            neon_base5 = vld1q_f32(y[12] + i);
            neon_base6 = vld1q_f32(y[13] + i);
            neon_base7 = vld1q_f32(y[14] + i);
            neon_base8 = vld1q_f32(y[15] + i);

            neon_res9 = vmlaq_f32(neon_res9, neon_base1, neon_query);
            neon_res10 = vmlaq_f32(neon_res10, neon_base2, neon_query);
            neon_res11 = vmlaq_f32(neon_res11, neon_base3, neon_query);
            neon_res12 = vmlaq_f32(neon_res12, neon_base4, neon_query);
            neon_res13 = vmlaq_f32(neon_res13, neon_base5, neon_query);
            neon_res14 = vmlaq_f32(neon_res14, neon_base6, neon_query);
            neon_res15 = vmlaq_f32(neon_res15, neon_base7, neon_query);
            neon_res16 = vmlaq_f32(neon_res16, neon_base8, neon_query);
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
        for (int i = 0; i < 16; i++) {
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

static void IPBatch8ByIdx(const float *x, const float *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    if (LIKELY(d >= single_round)) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y[0]);
        float32x4_t neon_base2 = vld1q_f32(y[1]);
        float32x4_t neon_base3 = vld1q_f32(y[2]);
        float32x4_t neon_base4 = vld1q_f32(y[3]);
        float32x4_t neon_base5 = vld1q_f32(y[4]);
        float32x4_t neon_base6 = vld1q_f32(y[5]);
        float32x4_t neon_base7 = vld1q_f32(y[6]);
        float32x4_t neon_base8 = vld1q_f32(y[7]);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);
        float32x4_t neon_res5 = vmulq_f32(neon_base5, neon_query);
        float32x4_t neon_res6 = vmulq_f32(neon_base6, neon_query);
        float32x4_t neon_res7 = vmulq_f32(neon_base7, neon_query);
        float32x4_t neon_res8 = vmulq_f32(neon_base8, neon_query);
        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y[0] + i);
            neon_base2 = vld1q_f32(y[1] + i);
            neon_base3 = vld1q_f32(y[2] + i);
            neon_base4 = vld1q_f32(y[3] + i);
            neon_base5 = vld1q_f32(y[4] + i);
            neon_base6 = vld1q_f32(y[5] + i);
            neon_base7 = vld1q_f32(y[6] + i);
            neon_base8 = vld1q_f32(y[7] + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
            neon_res5 = vmlaq_f32(neon_res5, neon_base5, neon_query);
            neon_res6 = vmlaq_f32(neon_res6, neon_base6, neon_query);
            neon_res7 = vmlaq_f32(neon_res7, neon_base7, neon_query);
            neon_res8 = vmlaq_f32(neon_res8, neon_base8, neon_query);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else {
        for (int i = 0; i < 8; i++) {
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
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y[0] + i);
            d1 += x[i] * *(y[1] + i);
            d2 += x[i] * *(y[2] + i);
            d3 += x[i] * *(y[3] + i);
            d4 += x[i] * *(y[4] + i);
            d5 += x[i] * *(y[5] + i);
            d6 += x[i] * *(y[6] + i);
            d7 += x[i] * *(y[7] + i);
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

static void IPBatch4ByIdx(const float *x, const float *__restrict *y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 4;
    if (LIKELY(d >= single_round)) {
        float32x4_t neon_query = vld1q_f32(x);
        float32x4_t neon_base1 = vld1q_f32(y[0]);
        float32x4_t neon_base2 = vld1q_f32(y[1]);
        float32x4_t neon_base3 = vld1q_f32(y[2]);
        float32x4_t neon_base4 = vld1q_f32(y[3]);

        float32x4_t neon_res1 = vmulq_f32(neon_base1, neon_query);
        float32x4_t neon_res2 = vmulq_f32(neon_base2, neon_query);
        float32x4_t neon_res3 = vmulq_f32(neon_base3, neon_query);
        float32x4_t neon_res4 = vmulq_f32(neon_base4, neon_query);

        for (i = single_round; i <= d - single_round; i += single_round) {
            neon_query = vld1q_f32(x + i);
            neon_base1 = vld1q_f32(y[0] + i);
            neon_base2 = vld1q_f32(y[1] + i);
            neon_base3 = vld1q_f32(y[2] + i);
            neon_base4 = vld1q_f32(y[3] + i);

            neon_res1 = vmlaq_f32(neon_res1, neon_base1, neon_query);
            neon_res2 = vmlaq_f32(neon_res2, neon_base2, neon_query);
            neon_res3 = vmlaq_f32(neon_res3, neon_base3, neon_query);
            neon_res4 = vmlaq_f32(neon_res4, neon_base4, neon_query);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float d0 = x[i] * *(y[0] + i);
        float d1 = x[i] * *(y[1] + i);
        float d2 = x[i] * *(y[2] + i);
        float d3 = x[i] * *(y[3] + i);

        for (i++; i < d; ++i) {
            d0 += x[i] * *(y[0] + i);
            d1 += x[i] * *(y[1] + i);
            d2 += x[i] * *(y[2] + i);
            d3 += x[i] * *(y[3] + i);
        }

        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}

static void IPBatch2ByIdx(const float *x, const float *__restrict y0, const float *__restrict y1, const size_t d,
                          float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;

    if (LIKELY(d >= single_round)) {
        float32x4_t x_0 = vld1q_f32(x);
        float32x4_t x_1 = vld1q_f32(x + 4);

        float32x4_t y0_0 = vld1q_f32(y0);
        float32x4_t y0_1 = vld1q_f32(y0 + 4);
        float32x4_t y1_0 = vld1q_f32(y1);
        float32x4_t y1_1 = vld1q_f32(y1 + 4);

        float32x4_t d0_0 = vmulq_f32(x_0, y0_0);
        float32x4_t d0_1 = vmulq_f32(x_1, y0_1);
        float32x4_t d1_0 = vmulq_f32(x_0, y1_0);
        float32x4_t d1_1 = vmulq_f32(x_1, y1_1);
        for (i = single_round; i <= d - single_round; i += single_round) {
            x_0 = vld1q_f32(x + i);
            y0_0 = vld1q_f32(y0 + i);
            y1_0 = vld1q_f32(y1 + i);
            d0_0 = vmlaq_f32(d0_0, x_0, y0_0);
            d1_0 = vmlaq_f32(d1_0, x_0, y1_0);

            x_1 = vld1q_f32(x + i + 4);
            y0_1 = vld1q_f32(y0 + i + 4);
            y1_1 = vld1q_f32(y1 + i + 4);
            d0_1 = vmlaq_f32(d0_1, x_1, y0_1);
            d1_1 = vmlaq_f32(d1_1, x_1, y1_1);
        }

        d0_0 = vaddq_f32(d0_0, d0_1);
        d1_0 = vaddq_f32(d1_0, d1_1);
        dis[0] = vaddvq_f32(d0_0);
        dis[1] = vaddvq_f32(d1_0);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp0 = x[i] * y0[i];
        const float tmp1 = x[i] * y1[i];
        dis[0] += tmp0;
        dis[1] += tmp1;
    }
}

void IPNyByIdx(float *dis, const float *x, const float *y, const int64_t *ids, size_t d, size_t ny)
{
    size_t i = 0;
    const float *__restrict listy[16];

    for (; i + 16 <= ny; i += 16) {
        prefetch_L1(x);
        listy[0] = (const float *)(y + *(ids + i) * d);
        prefetch_Lx(listy[0]);
        listy[1] = (const float *)(y + *(ids + i + 1) * d);
        prefetch_Lx(listy[1]);
        listy[2] = (const float *)(y + *(ids + i + 2) * d);
        prefetch_Lx(listy[2]);
        listy[3] = (const float *)(y + *(ids + i + 3) * d);
        prefetch_Lx(listy[3]);
        listy[4] = (const float *)(y + *(ids + i + 4) * d);
        prefetch_Lx(listy[4]);
        listy[5] = (const float *)(y + *(ids + i + 5) * d);
        prefetch_Lx(listy[5]);
        listy[6] = (const float *)(y + *(ids + i + 6) * d);
        prefetch_Lx(listy[6]);
        listy[7] = (const float *)(y + *(ids + i + 7) * d);
        prefetch_Lx(listy[7]);
        listy[8] = (const float *)(y + *(ids + i + 8) * d);
        prefetch_Lx(listy[8]);
        listy[9] = (const float *)(y + *(ids + i + 9) * d);
        prefetch_Lx(listy[9]);
        listy[10] = (const float *)(y + *(ids + i + 10) * d);
        prefetch_Lx(listy[10]);
        listy[11] = (const float *)(y + *(ids + i + 11) * d);
        prefetch_Lx(listy[11]);
        listy[12] = (const float *)(y + *(ids + i + 12) * d);
        prefetch_Lx(listy[12]);
        listy[13] = (const float *)(y + *(ids + i + 13) * d);
        prefetch_Lx(listy[13]);
        listy[14] = (const float *)(y + *(ids + i + 14) * d);
        prefetch_Lx(listy[14]);
        listy[15] = (const float *)(y + *(ids + i + 15) * d);
        prefetch_Lx(listy[15]);
        IPBatch16ByIdx(x, listy, d, dis + i);
    }
    if (ny & 8) {
        listy[0] = (const float *)(y + *(ids + i) * d);
        listy[1] = (const float *)(y + *(ids + i + 1) * d);
        listy[2] = (const float *)(y + *(ids + i + 2) * d);
        listy[3] = (const float *)(y + *(ids + i + 3) * d);
        listy[4] = (const float *)(y + *(ids + i + 4) * d);
        listy[5] = (const float *)(y + *(ids + i + 5) * d);
        listy[6] = (const float *)(y + *(ids + i + 6) * d);
        listy[7] = (const float *)(y + *(ids + i + 7) * d);
        IPBatch8ByIdx(x, listy, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        listy[0] = (const float *)(y + *(ids + i) * d);
        listy[1] = (const float *)(y + *(ids + i + 1) * d);
        listy[2] = (const float *)(y + *(ids + i + 2) * d);
        listy[3] = (const float *)(y + *(ids + i + 3) * d);
        IPBatch4ByIdx(x, listy, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        const float *y0 = y + *(ids + i) * d;
        const float *y1 = y + *(ids + i + 1) * d;
        IPBatch2ByIdx(x, y0, y1, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        IPSingle(x, y + d * ids[i], d, &dis[i]);
    }
}

template <int asc = 0>
static inline int DataCompare(float a, float b)
{
    if constexpr (asc == 0) {
        return (a < b);
    } else {
        return (a > b);
    }
}

template <int asc = 0>
static inline void TopkPushHeapAsc(int64_t k, float *bh_val, int64_t *bh_ids, float val, int64_t id)
{
    bh_val--;
    bh_ids--;
    int64_t i = k, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (DataCompare<asc>(val, bh_val[i_father])) {
            bh_val[i] = bh_val[i_father];
            bh_ids[i] = bh_ids[i_father];
            i = i_father;
        } else {
            break;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = id;
}

template <int asc = 0>
static inline void TopkReplaceTopHeapAsc(int64_t k, float *bh_val, int64_t *bh_ids, float val, int64_t id)
{
    bh_val--;
    bh_ids--;
    int64_t i = 1;
    int64_t i1;
    int64_t i2;
    while (1) {
        i1 = i << 1;
        if (i1 > k) {
            break;
        }
        if ((i1 == k) || DataCompare<asc>(bh_val[i1], bh_val[i1 + 1])) {
            i2 = i1;
        } else {
            i2 = i1 + 1;
        }
        if (DataCompare<asc>(bh_val[i2], val)) {
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        } else {
            break;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = id;
}

template <int asc = 0>
static inline void TopkInitHeapAsc(int64_t k, float *bh_val, int64_t *bh_ids, const float *x, const int64_t *ids)
{
    for (int64_t i = 0; i < k; ++i) {
        TopkPushHeapAsc<asc>(i + 1, bh_val, bh_ids, x[i], ids[i]);
    }
}

template <int asc = 0>
static inline void TopkUpdateHeapAsc(int64_t k, float *bh_val, int64_t *bh_ids, const float *x, const int64_t *ids,
                                     int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        if (DataCompare<asc>(bh_val[0], x[i])) {
            TopkReplaceTopHeapAsc<asc>(k, bh_val, bh_ids, x[i], ids[i]);
        }
    }
}

template <int asc = 0>
static inline void TopkFinalizeAsc(int64_t k, float *bh_val, int64_t *bh_ids)
{
    int64_t i = 0;
    int64_t ii = 0;
    for (; i < k - 1; i++) {
        float val = bh_val[0];
        int64_t id = bh_ids[0];

        TopkReplaceTopHeapAsc<asc>(k - i - 1, bh_val, bh_ids, bh_val[k - i - 1], bh_ids[k - i - 1]);
        bh_val[k - i - 1] = val;
        bh_ids[k - i - 1] = id;
    }
}

void SelectL2Topk(int64_t k, int64_t *labels, float *distances, int64_t k_base, const int64_t *base_labels,
                  const float *base_distances)
{
    TopkInitHeapAsc<1>(k, distances, labels, base_distances, base_labels);
    if (k_base != k) {
        TopkUpdateHeapAsc<1>(k, distances, labels, base_distances + k, base_labels + k, k_base - k);
    }
    TopkFinalizeAsc<1>(k, distances, labels);
}

void SelectIPTopk(int64_t k, int64_t *labels, float *distances, int64_t k_base, const int64_t *base_labels,
                  const float *base_distances)
{
    TopkInitHeapAsc<0>(k, distances, labels, base_distances, base_labels);
    if (k_base != k) {
        TopkUpdateHeapAsc<0>(k, distances, labels, base_distances + k, base_labels + k, k_base - k);
    }
    TopkFinalizeAsc<0>(k, distances, labels);
}

}  // namespace faiss
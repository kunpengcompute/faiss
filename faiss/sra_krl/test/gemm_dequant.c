#include <arm_sve.h>
#include <string.h>
#include "tools.h"
inline bfloat16_t* dequantize_gt_columnmajor(int kb, int nb, const uint8_t* __restrict B, const uint8_t* __restrict zero_point, const float16_t* __restrict scales, bfloat16_t* __restrict buffer) {
    const uint8_t* pB = (const uint8_t*)B;
    const uint8_t* zp = (const uint8_t*)zero_point;
    const float16_t* sc = (const float16_t*)scales;
    for(int i = 0; 2 * i < nb; ++i) {
        float low_zp = (float)(zp[i] & 0x0F);
        float high_zp = (float)(zp[i] >> 4);
        float low_sc = (float)sc[2 * i];
        float high_sc = (float)sc[2 * i + 1];
        for(int j = 0; j < kb; ++j) {
            float low_pB = (float)(pB[i * kb + j] & 0x0F);
            float high_pB = (float)(pB[i * kb + j] >> 4);
            buffer[2 * i * kb + j] = (bfloat16_t)((low_pB - low_zp) * low_sc);
            buffer[2 * i * kb + j + kb] = (bfloat16_t)((high_pB - high_zp) * high_sc);
        }
    }
    return buffer;
}


inline bfloat16_t *dequantizeBlock_packed_withscale(int nb, const uint8_t* __restrict B,
    const uint8_t* __restrict zero_point, const float16_t* __restrict scales, bfloat16_t* __restrict buffer)
{
    bfloat16_t *Bbf16 = buffer;
    const uint8_t *pB = (const uint8_t*)B;
    const svbool_t pg = svptrue_b8();
    constexpr int half_unroll = 8; // = 256 / 16 / 2
    constexpr int n_unroll = 16;   // = 256 / 16
    // Initialize the table
    const svfloat32_t tableinit_f32_even = svcvt_f32_u32_x(pg, svindex_u32(0,2));
    const svfloat32_t tableinit_f32_oodd = svcvt_f32_u32_x(pg, svindex_u32(1,2));
    for (int k = 0; 2 * k < nb; k += half_unroll) {
        const svuint32_t zp_u32 = svld1ub_u32(pg, (const uint8_t*)(zero_point + k));
        svfloat32_t zp_f32_oodd = svcvt_f32_u32_x(pg, svlsr_m(pg, zp_u32, 4));
        svfloat32_t zp_f32_even = svcvt_f32_u32_x(pg, svand_m(pg, zp_u32, 0xF));
        svfloat32_t sc_f32_even = svreinterpret_f32(svld1_f16(pg, (const float16_t*)(scales + k * 2)));
        svfloat32_t sc_f32_oodd = svreinterpret_f32(svtrn2_f16(svreinterpret_f16(sc_f32_even), svreinterpret_f16(sc_f32_even)));
        sc_f32_even = svcvt_f32_f16_x(pg, svreinterpret_f16(sc_f32_even));
        sc_f32_oodd = svcvt_f32_f16_x(pg, svreinterpret_f16(sc_f32_oodd));
        zp_f32_oodd = svmsb_n_f32_x(pg, zp_f32_oodd, sc_f32_oodd, 0.f);
        zp_f32_even = svmsb_n_f32_x(pg, zp_f32_even, sc_f32_even, 0.f);
        for(int n = 0; n < half_unroll; ++n) {
            svfloat32_t table_f32_oodd_0 = svdup_lane_f32(zp_f32_even, n);
            svfloat32_t table_f32_oodd_1 = svdup_lane_f32(zp_f32_oodd, n);
            svfloat32_t sc_f32_0 = svdup_lane_f32(sc_f32_even, n);
            svfloat32_t sc_f32_1 = svdup_lane_f32(sc_f32_oodd, n);

            svfloat32_t table_f32_even_0 = svmla_f32_x(pg, table_f32_oodd_0, tableinit_f32_even, sc_f32_0);
            table_f32_oodd_0 = svmla_f32_x(pg, table_f32_oodd_0, tableinit_f32_oodd, sc_f32_0);
            svfloat32_t table_f32_even_1 = svmla_f32_x(pg, table_f32_oodd_1, tableinit_f32_even, sc_f32_1);
            table_f32_oodd_1 = svmla_f32_x(pg, table_f32_oodd_1, tableinit_f32_oodd, sc_f32_1);
            const svuint16_t table_bf16_all_0 = svtrn2_u16(svreinterpret_u16(table_f32_even_0), svreinterpret_u16(table_f32_oodd_0));
            const svuint16_t table_bf16_all_1 = svtrn2_u16(svreinterpret_u16(table_f32_even_1), svreinterpret_u16(table_f32_oodd_1));
        // kb == 64
            svuint16_t b_u16_0 = svld1ub_u16(pg, pB + (k + n) * 64);
            svuint16_t b_u16_1 = svld1ub_u16(pg, pB + (k + n) * 64 + 16);
            svuint16_t b_u16_2 = svld1ub_u16(pg, pB + (k + n) * 64 + 32);
            svuint16_t b_u16_3 = svld1ub_u16(pg, pB + (k + n) * 64 + 48);
            svuint16_t b_u16_0_even = svand_m(pg, b_u16_0, 0xF);
            svuint16_t b_u16_0_oodd = svlsr_m(pg, b_u16_0, 4);
            svuint16_t b_u16_1_even = svand_m(pg, b_u16_1, 0xF);
            svuint16_t b_u16_1_oodd = svlsr_m(pg, b_u16_1, 4);
            svuint16_t b_u16_2_even = svand_m(pg, b_u16_2, 0xF);
            svuint16_t b_u16_2_oodd = svlsr_m(pg, b_u16_2, 4);
            svuint16_t b_u16_3_even = svand_m(pg, b_u16_3, 0xF);
            svuint16_t b_u16_3_oodd = svlsr_m(pg, b_u16_3, 4);
            asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
            : [res1] "=w"(b_u16_0_even) : [ftable] "w"(table_bf16_all_0), [index1] "w"(b_u16_0_even):);
            asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
            : [res0] "=w"(b_u16_0_oodd) : [ftable] "w"(table_bf16_all_1), [index0] "w"(b_u16_0_oodd):);
            asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
            : [res1] "=w"(b_u16_1_even) : [ftable] "w"(table_bf16_all_0), [index1] "w"(b_u16_1_even):);
            asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
            : [res0] "=w"(b_u16_1_oodd) : [ftable] "w"(table_bf16_all_1), [index0] "w"(b_u16_1_oodd):);
            asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
            : [res1] "=w"(b_u16_2_even) : [ftable] "w"(table_bf16_all_0), [index1] "w"(b_u16_2_even):);
            asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
            : [res0] "=w"(b_u16_2_oodd) : [ftable] "w"(table_bf16_all_1), [index0] "w"(b_u16_2_oodd):);
            asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
            : [res1] "=w"(b_u16_3_even) : [ftable] "w"(table_bf16_all_0), [index1] "w"(b_u16_3_even):);
            asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
            : [res0] "=w"(b_u16_3_oodd) : [ftable] "w"(table_bf16_all_1), [index0] "w"(b_u16_3_oodd):);
            svst1_bf16(pg, Bbf16 + (k + n) * 128,       svreinterpret_bf16(b_u16_0_even));
            svst1_bf16(pg, Bbf16 + (k + n) * 128 + 16,  svreinterpret_bf16(b_u16_1_even));
            svst1_bf16(pg, Bbf16 + (k + n) * 128 + 32,  svreinterpret_bf16(b_u16_2_even));
            svst1_bf16(pg, Bbf16 + (k + n) * 128 + 48,  svreinterpret_bf16(b_u16_3_even));
            svst1_bf16(pg, Bbf16 + (k + n) * 128 + 64,  svreinterpret_bf16(b_u16_0_oodd));
            svst1_bf16(pg, Bbf16 + (k + n) * 128 + 80,  svreinterpret_bf16(b_u16_1_oodd));
            svst1_bf16(pg, Bbf16 + (k + n) * 128 + 96,  svreinterpret_bf16(b_u16_2_oodd));
            svst1_bf16(pg, Bbf16 + (k + n) * 128 + 108, svreinterpret_bf16(b_u16_3_oodd));
        }
    }

    return Bbf16;
}

inline bfloat16_t *dequantizeBlock_packed(int nb, const uint8_t* __restrict B,
    const uint8_t* __restrict zero_point, bfloat16_t* __restrict buffer)
{
    bfloat16_t *Bbf16 = buffer;
    const uint8_t *pB = (const uint8_t*)B;
    const svbool_t pg = svptrue_b8();
    constexpr int half_unroll = 8; // = 256 / 16 / 2
    constexpr int n_unroll = 16;   // = 256 / 16
    // Initialize the table
    const svfloat32_t tableinit_f32_even = svcvt_f32_u32_x(pg, svindex_u32(0,2));
    const svfloat32_t tableinit_f32_oodd = svcvt_f32_u32_x(pg, svindex_u32(1,2));
    for (int k = 0; 2 * k < nb; k += half_unroll) {
        const svuint32_t zp_u32 = svld1ub_u32(pg, (const uint8_t*)(zero_point + k));
        svfloat32_t zp_f32_oodd = svcvt_f32_u32_x(pg, svlsr_m(pg, zp_u32, 4));
        svfloat32_t zp_f32_even = svcvt_f32_u32_x(pg, svand_m(pg, zp_u32, 0xF));
        zp_f32_oodd = svneg_f32_x(pg, zp_f32_oodd);
        zp_f32_even = svneg_f32_x(pg, zp_f32_even);
        for(int n = 0; n < half_unroll; ++n) {
            svfloat32_t table_f32_oodd_0 = svdup_lane_f32(zp_f32_even, n);
            svfloat32_t table_f32_oodd_1 = svdup_lane_f32(zp_f32_oodd, n);

            svfloat32_t table_f32_even_0 = svadd_f32_x(pg, table_f32_oodd_0, tableinit_f32_even);
            table_f32_oodd_0 = svadd_f32_x(pg, table_f32_oodd_0, tableinit_f32_oodd);
            svfloat32_t table_f32_even_1 = svadd_f32_x(pg, table_f32_oodd_1, tableinit_f32_even);
            table_f32_oodd_1 = svadd_f32_x(pg, table_f32_oodd_1, tableinit_f32_oodd);
            const svuint16_t table_bf16_all_0 = svtrn2_u16(svreinterpret_u16(table_f32_even_0), svreinterpret_u16(table_f32_oodd_0));
            const svuint16_t table_bf16_all_1 = svtrn2_u16(svreinterpret_u16(table_f32_even_1), svreinterpret_u16(table_f32_oodd_1));
        // kb == 64
            svuint16_t b_u16_0 = svld1ub_u16(pg, pB + (k + n) * 64);
            svuint16_t b_u16_1 = svld1ub_u16(pg, pB + (k + n) * 64 + 16);
            svuint16_t b_u16_2 = svld1ub_u16(pg, pB + (k + n) * 64 + 32);
            svuint16_t b_u16_3 = svld1ub_u16(pg, pB + (k + n) * 64 + 48);
            svuint16_t b_u16_0_even = svand_m(pg, b_u16_0, 0xF);
            svuint16_t b_u16_0_oodd = svlsr_m(pg, b_u16_0, 4);
            svuint16_t b_u16_1_even = svand_m(pg, b_u16_1, 0xF);
            svuint16_t b_u16_1_oodd = svlsr_m(pg, b_u16_1, 4);
            svuint16_t b_u16_2_even = svand_m(pg, b_u16_2, 0xF);
            svuint16_t b_u16_2_oodd = svlsr_m(pg, b_u16_2, 4);
            svuint16_t b_u16_3_even = svand_m(pg, b_u16_3, 0xF);
            svuint16_t b_u16_3_oodd = svlsr_m(pg, b_u16_3, 4);
            asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
            : [res1] "=w"(b_u16_0_even) : [ftable] "w"(table_bf16_all_0), [index1] "w"(b_u16_0_even):);
            asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
            : [res0] "=w"(b_u16_0_oodd) : [ftable] "w"(table_bf16_all_1), [index0] "w"(b_u16_0_oodd):);
            asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
            : [res1] "=w"(b_u16_1_even) : [ftable] "w"(table_bf16_all_0), [index1] "w"(b_u16_1_even):);
            asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
            : [res0] "=w"(b_u16_1_oodd) : [ftable] "w"(table_bf16_all_1), [index0] "w"(b_u16_1_oodd):);
            asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
            : [res1] "=w"(b_u16_2_even) : [ftable] "w"(table_bf16_all_0), [index1] "w"(b_u16_2_even):);
            asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
            : [res0] "=w"(b_u16_2_oodd) : [ftable] "w"(table_bf16_all_1), [index0] "w"(b_u16_2_oodd):);
            asm volatile("tbl %[res1].h, %[ftable].h, %[index1].h"
            : [res1] "=w"(b_u16_3_even) : [ftable] "w"(table_bf16_all_0), [index1] "w"(b_u16_3_even):);
            asm volatile("tbl %[res0].h, %[ftable].h, %[index0].h"
            : [res0] "=w"(b_u16_3_oodd) : [ftable] "w"(table_bf16_all_1), [index0] "w"(b_u16_3_oodd):);
            svst1_bf16(pg, Bbf16 + (n + k) * 32,       svreinterpret_bf16(b_u16_0_even));
            svst1_bf16(pg, Bbf16 + (n + k) * 32 + 16,  svreinterpret_bf16(b_u16_0_oodd));
            svst1_bf16(pg, Bbf16 + (n + k) * 32 + nb * 16,  svreinterpret_bf16(b_u16_1_even));
            svst1_bf16(pg, Bbf16 + (n + k) * 32 + nb * 16 + 16,  svreinterpret_bf16(b_u16_1_oodd));
            svst1_bf16(pg, Bbf16 + (n + k) * 32 + nb * 32,  svreinterpret_bf16(b_u16_2_even));
            svst1_bf16(pg, Bbf16 + (n + k) * 32 + nb * 32 + 16,  svreinterpret_bf16(b_u16_2_oodd));
            svst1_bf16(pg, Bbf16 + (n + k) * 32 + nb * 48,  svreinterpret_bf16(b_u16_3_even));
            svst1_bf16(pg, Bbf16 + (n + k) * 32 + nb * 48 + 16, svreinterpret_bf16(b_u16_3_oodd));
        }
    }
    return Bbf16;
}

bool test_dequant(int kb, int nb){
    assert(kb == 64);
    assert(nb % 2);
    float16_t* scale = new float16_t[nb];
    uint8_t* zero_point = new uint8_t[nb / 2];
    uint8_t* B = new uint8_t[kb * nb / 2];
    bfloat16_t* gt = new bfloat16_t[kb * nb];
    bfloat16_t* simd_test = new bfloat16_t[kb * nb];

    init_vector(nb, scale, 0, 16);
    init_vector(nb / 2, zero_point, 0, 255);
    init_vector(kb * nb / 2, B, 0, 255);

    dequantizeBlock_packed_withscale(nb, B, zero_point, scale, simd_test);

    dequantize_gt_columnmajor(kb, nb, B, zero_point, scale, gt);

    bool pass = chechResult(kb * nb, simd_test, gt, 1e-6);

    delete[] gt, simd_test;
    delete[] scale, zero_point, B;
    return pass;
}

int main(int argc, char *argv[])
{
    bool pass = true;
    pass &= test_dequant(64, 16);
    std::cout << "check: " << (pass ? "pass" : "FAILED") << '\n';
    return 0;
}

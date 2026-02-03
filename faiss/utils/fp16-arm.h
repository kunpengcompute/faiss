/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Modifications:
 * - 2026 BoostKit: Added functions for converting between FP16 and FP32.
 */

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arm_neon.h>
#include <cstdint>
#include <iostream>

namespace faiss {

inline uint16_t encode_fp16(float x) {
    float32x4_t fx4 = vdupq_n_f32(x);
    float16x4_t f16x4 = vcvt_f16_f32(fx4);
    uint16x4_t ui16x4 = vreinterpret_u16_f16(f16x4);
    return vduph_lane_u16(ui16x4, 3);
}

inline float decode_fp16(uint16_t x) {
    uint16x4_t ui16x4 = vdup_n_u16(x);
    float16x4_t f16x4 = vreinterpret_f16_u16(ui16x4);
    float32x4_t fx4 = vcvt_f32_f16(f16x4);
    return vdups_laneq_f32(fx4, 3);
}

inline void convert_fp16_to_fp32(const float16_t* src, int64_t d, float* out) {
    if (src == nullptr || out == nullptr || d <= 0) {
        std::cerr << "Error: Invalid arguments to convert_fp16_to_fp32 - "
                  << "src: " << (src ? "valid" : "null") << ", "
                  << "out: " << (out ? "valid" : "null") << ", "
                  << "d: " << d << std::endl;
        return;
    }
    int64_t l = 0;    
    constexpr int64_t multi_loop = 32;
    constexpr int64_t double_loop = 16;
    constexpr int64_t single_loop = 8;

    for(; l + multi_loop <= d; l += multi_loop) {
        float16x4_t h0 = vld1_f16(src + l);
        float16x4_t h1 = vld1_f16(src + l + 4);
        float16x4_t h2 = vld1_f16(src + l + 8);
        float16x4_t h3 = vld1_f16(src + l + 12);
        float16x4_t h4 = vld1_f16(src + l + 16);
        float16x4_t h5 = vld1_f16(src + l + 20);
        float16x4_t h6 = vld1_f16(src + l + 24);
        float16x4_t h7 = vld1_f16(src + l + 28);

        float32x4_t f0 = vcvt_f32_f16(h0);
        float32x4_t f1 = vcvt_f32_f16(h1);
        float32x4_t f2 = vcvt_f32_f16(h2);
        float32x4_t f3 = vcvt_f32_f16(h3);
        float32x4_t f4 = vcvt_f32_f16(h4);
        float32x4_t f5 = vcvt_f32_f16(h5);
        float32x4_t f6 = vcvt_f32_f16(h6);
        float32x4_t f7 = vcvt_f32_f16(h7);

        vst1q_f32(out + l, f0);
        vst1q_f32(out + l + 4, f1);
        vst1q_f32(out + l + 8, f2);
        vst1q_f32(out + l + 12, f3);
        vst1q_f32(out + l + 16, f4);
        vst1q_f32(out + l + 20, f5);
        vst1q_f32(out + l + 24, f6);
        vst1q_f32(out + l + 28, f7);
    }
    if((d - l) >= double_loop) {
        float16x4_t h0 = vld1_f16(src + l);
        float16x4_t h1 = vld1_f16(src + l + 4);
        float16x4_t h2 = vld1_f16(src + l + 8);
        float16x4_t h3 = vld1_f16(src + l + 12);

        float32x4_t f0 = vcvt_f32_f16(h0);
        float32x4_t f1 = vcvt_f32_f16(h1);
        float32x4_t f2 = vcvt_f32_f16(h2);
        float32x4_t f3 = vcvt_f32_f16(h3);

        vst1q_f32(out + l, f0);
        vst1q_f32(out + l + 4, f1);
        vst1q_f32(out + l + 8, f2);
        vst1q_f32(out + l + 12, f3);
                                                                                                                                        
        l += 16;
    }
    if((d - l) >= single_loop) {
        float16x4_t h0 = vld1_f16(src + l);
        float16x4_t h1 = vld1_f16(src + l + 4);

        float32x4_t f0 = vcvt_f32_f16(h0);
        float32x4_t f1 = vcvt_f32_f16(h1);

        vst1q_f32(out + l, f0);
        vst1q_f32(out + l + 4, f1);
                                                                                            
        l += 8;
    }
    for (; l < d; ++l) {
        out[l] = (float)(src[l]);
    }
}

inline void convert_fp32_to_fp16(const float* src, int64_t d, float16_t* out) {
    if (src == nullptr || out == nullptr || d <= 0) {
        std::cerr << "Error: Invalid arguments to convert_fp32_to_fp16 - "
                  << "src: " << (src ? "valid" : "null") << ", "
                  << "out: " << (out ? "valid" : "null") << ", "
                  << "d: " << d << std::endl;
        return;
    }
    int64_t l = 0;    
    constexpr int64_t multi_loop = 32;
    constexpr int64_t double_loop = 16;
    constexpr int64_t single_loop = 8;

    for(; l + multi_loop <= d; l += multi_loop) {
        float32x4_t f0 = vld1q_f32(src + l);
        float32x4_t f1 = vld1q_f32(src + l + 4);
        float32x4_t f2 = vld1q_f32(src + l + 8);
        float32x4_t f3 = vld1q_f32(src + l + 12);
        float32x4_t f4 = vld1q_f32(src + l + 16);
        float32x4_t f5 = vld1q_f32(src + l + 20);
        float32x4_t f6 = vld1q_f32(src + l + 24);
        float32x4_t f7 = vld1q_f32(src + l + 28);

        float16x4_t h0 = vcvt_f16_f32(f0);
        float16x4_t h1 = vcvt_f16_f32(f1);
        float16x4_t h2 = vcvt_f16_f32(f2);
        float16x4_t h3 = vcvt_f16_f32(f3);
        float16x4_t h4 = vcvt_f16_f32(f4);
        float16x4_t h5 = vcvt_f16_f32(f5);
        float16x4_t h6 = vcvt_f16_f32(f6);
        float16x4_t h7 = vcvt_f16_f32(f7);

        float16x8_t c0 = vcombine_f16(h0, h1);
        float16x8_t c1 = vcombine_f16(h2, h3);
        float16x8_t c2 = vcombine_f16(h4, h5);
        float16x8_t c3 = vcombine_f16(h6, h7);

        vst1q_f16(out + l, c0);
        vst1q_f16(out + l + 8, c1);
        vst1q_f16(out + l + 16, c2);
        vst1q_f16(out + l + 24, c3);
    }

    if((d - l) >= double_loop) {
        float32x4_t f0 = vld1q_f32(src + l);
        float32x4_t f1 = vld1q_f32(src + l + 4);
        float32x4_t f2 = vld1q_f32(src + l + 8);
        float32x4_t f3 = vld1q_f32(src + l + 12);

        float16x4_t h0 = vcvt_f16_f32(f0);
        float16x4_t h1 = vcvt_f16_f32(f1);
        float16x4_t h2 = vcvt_f16_f32(f2);
        float16x4_t h3 = vcvt_f16_f32(f3);

        float16x8_t c0 = vcombine_f16(h0, h1);
        float16x8_t c1 = vcombine_f16(h2, h3);

        vst1q_f16(out + l, c0);
        vst1q_f16(out + l + 8, c1);

        l += double_loop;
    }

    if((d - l) >= single_loop) {
        float32x4_t f0 = vld1q_f32(src + l);
        float32x4_t f1 = vld1q_f32(src + l + 4);

        float16x4_t h0 = vcvt_f16_f32(f0);
        float16x4_t h1 = vcvt_f16_f32(f1);

        float16x8_t c = vcombine_f16(h0, h1);
        vst1q_f16(out + l, c);

        l += single_loop;
    }

    for (; l < d; ++l) {
        float32x4_t tmp = vld1q_dup_f32(src + l);
        float16_t res = vget_lane_f16(vcvt_f16_f32(tmp), 0);
        out[l] = res;
    }
}

} // namespace faiss

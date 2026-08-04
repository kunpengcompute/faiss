/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_simd.h>

#ifdef COMPILE_SIMD_ARM_NEON

#ifdef KRL
#include <arm_neon.h>
#endif

namespace faiss::rabitq {

template <>
uint64_t bitwise_and_dot_product<SIMDLevel::ARM_NEON>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
#ifdef KRL
    uint64_t sum = 0;
    size_t offset = 0;
    // 16-byte NEON blocks
    for (; offset + 16 <= size; offset += 16) {
        const uint8x16_t yv = vld1q_u8(data + offset);
        for (size_t j = 0; j < qb; j++) {
            const uint8x16_t qv =
                    vld1q_u8(query + j * size + offset);
            const uint8x16_t andv = vandq_u8(qv, yv);
            sum += vaddlvq_u8(vcntq_u8(andv)) << j;
        }
    }
    // 8-byte NEON blocks
    for (; offset + 8 <= size; offset += 8) {
        uint8x8_t yv = vld1_u8(data + offset);
        for (size_t j = 0; j < qb; j++) {
            uint8x8_t qv = vld1_u8(query + j * size + offset);
            uint8x8_t andv = vand_u8(qv, yv);
            sum += vaddlv_u8(vcnt_u8(andv)) << j;
        }
    }
    // scalar tail (≤7 bytes)
    for (; offset < size; ++offset) {
        const uint8_t yv = data[offset];
        for (size_t j = 0; j < qb; j++) {
            const uint8_t qv = query[j * size + offset];
            sum += static_cast<uint64_t>(
                           __builtin_popcount(qv & yv))
                    << j;
        }
    }
    return sum;
#else
    return bitwise_and_dot_product<SIMDLevel::NONE>(query, data, size, qb);
#endif
}

template <>
uint64_t bitwise_xor_dot_product<SIMDLevel::ARM_NEON>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
#ifdef KRL
    uint64_t sum = 0;
    size_t offset = 0;
    for (; offset + 16 <= size; offset += 16) {
        const uint8x16_t yv = vld1q_u8(data + offset);
        for (size_t j = 0; j < qb; j++) {
            const uint8x16_t qv =
                    vld1q_u8(query + j * size + offset);
            sum += vaddlvq_u8(vcntq_u8(veorq_u8(qv, yv)))
                    << j;
        }
    }
    for (; offset + 8 <= size; offset += 8) {
        uint8x8_t yv = vld1_u8(data + offset);
        for (size_t j = 0; j < qb; j++) {
            uint8x8_t qv = vld1_u8(query + j * size + offset);
            sum += vaddlv_u8(vcnt_u8(veor_u8(qv, yv))) << j;
        }
    }
    for (; offset < size; ++offset) {
        const uint8_t yv = data[offset];
        for (size_t j = 0; j < qb; j++) {
            const uint8_t qv = query[j * size + offset];
            sum += static_cast<uint64_t>(
                           __builtin_popcount(qv ^ yv))
                    << j;
        }
    }
    return sum;
#else
    return bitwise_xor_dot_product<SIMDLevel::NONE>(query, data, size, qb);
#endif
}

template <>
uint64_t popcount<SIMDLevel::ARM_NEON>(const uint8_t* data, size_t size) {
#ifdef KRL
    uint64_t sum = 0;
    size_t offset = 0;
    for (; offset + 16 <= size; offset += 16) {
        const uint8x16_t yv = vld1q_u8(data + offset);
        sum += vaddlvq_u8(vcntq_u8(yv));
    }
    for (; offset + 8 <= size; offset += 8) {
        uint8x8_t yv = vld1_u8(data + offset);
        sum += vaddlv_u8(vcnt_u8(yv));
    }
    for (; offset < size; ++offset) {
        sum += __builtin_popcount(data[offset]);
    }
    return sum;
#else
    return popcount<SIMDLevel::NONE>(data, size);
#endif
}

} // namespace faiss::rabitq

namespace faiss::rabitq::multibit {

template <>
float compute_inner_product<SIMDLevel::ARM_NEON>(
        const uint8_t* __restrict sign_bits,
        const uint8_t* __restrict ex_code,
        const float* __restrict rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    return compute_inner_product<SIMDLevel::NONE>(
            sign_bits, ex_code, rotated_q, d, ex_bits, cb);
}

} // namespace faiss::rabitq::multibit

#endif // COMPILE_SIMD_ARM_NEON

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef DISTANCES_SIMD_H
#define DISTANCES_SIMD_H

#pragma once

namespace faiss {
inline void prefetch_L1(const void *address)
{
    __builtin_prefetch(address, 0, 3);
}
inline void prefetch_L2(const void *address)
{
    __builtin_prefetch(address, 0, 2);
}
inline void prefetch_L3(const void *address)
{
    __builtin_prefetch(address, 0, 1);
}
inline void prefetch_Lx(const void *address)
{
    __builtin_prefetch(address, 0, 0);
}

template <class T>
static inline bool LIKELY(T &&x) noexcept
{
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_expect(!!x, 1);
#else
    return !!x;
#endif
}

template <class T>
static inline bool UNLIKELY(T &&x) noexcept
{
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_expect(!!x, 0);
#else
    return !!x;
#endif
}

typedef struct LookupTable8bitHandle {
    size_t capacity;
    float *distance_buffer;
} LUT8bHandle;

void L2sqrNy(float *dis, const float *x, const float *y, const size_t ny, const size_t d);
void IPNy(float *dis, const float *x, const float *y, const size_t ny, const size_t d);
void L2sqrSingle(const float *x, const float *__restrict y, const size_t d, float *dis);
void IPSingle(const float *x, const float *__restrict y, const size_t d, float *dis);
void L2sqrNyByIdx(float *dis, const float *x, const float *y, const int64_t *ids, size_t d, size_t ny);
void IPNyByIdx(float *dis, const float *x, const float *y, const int64_t *ids, size_t d, size_t ny);
void SelectL2Topk(int64_t k, int64_t *labels, float *distances, int64_t k_base, const int64_t *base_labels,
                  const float *base_distances);
void SelectIPTopk(int64_t k, int64_t *labels, float *distances, int64_t k_base, const int64_t *base_labels,
                  const float *base_distances);
}
#endif

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

#pragma once

#include <stddef.h>
#include <stdint.h>

typedef int64_t idx_t;

template <int asce = 0>
inline int data_compare(float a, float b) {
    if constexpr (asce == 0) {
        return (a < b);
    } else {
        return (a > b);
    }
}

template <int asce = 0>
inline void krl_2heaps_push(
        idx_t k, float* bh_val, idx_t* bh_ids, float val, idx_t id) {
    bh_val--;
    bh_ids--;
    idx_t i = k, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (data_compare<asce>(val, bh_val[i_father])) {
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

template <int asce = 0>
inline void krl_2heaps_heapify(
        idx_t k,
        float* bh_val,
        idx_t* bh_ids,
        const float* x,
        const idx_t* ids) {
    for (idx_t i = 0; i < k; ++i) {
        krl_2heaps_push<asce>(i + 1, bh_val, bh_ids, x[i], ids[i]);
    }
}

template <int asce = 0>
inline void krl_2heaps_replace_top(
        idx_t k, float* bh_val, idx_t* bh_ids, float val, idx_t id) {
    bh_val--;
    bh_ids--;
    idx_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        if (i1 > k) {
            break;
        }
        if ((i1 == k) || data_compare<asce>(bh_val[i1], bh_val[i1 + 1])) {
            i2 = i1;
        } else {
            i2 = i1 + 1;
        }
        if (data_compare<asce>(bh_val[i2], val)) {
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

template <int asce = 0>
inline void krl_2heaps_reorder(idx_t k, float* bh_val, idx_t* bh_ids) {
    idx_t i = 0;
    for (; i < k - 1; i++) {
        float val = bh_val[0];
        idx_t id = bh_ids[0];
        krl_2heaps_replace_top<asce>(
                k - i - 1, bh_val, bh_ids, bh_val[k - i - 1],
                bh_ids[k - i - 1]);
        bh_val[k - i - 1] = val;
        bh_ids[k - i - 1] = id;
    }
}

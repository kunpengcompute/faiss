/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Modifications:
 * - 2026 BoostKit: Supports the FP16 function.
 */

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef INDEX_FLAT_H
#define INDEX_FLAT_H

#include <vector>

#include <faiss/IndexFlatCodes.h>
#ifdef __aarch64__
#include <faiss/sra_krl/include/safe_memory.h>
#endif

namespace faiss {

/** Index that stores the full vectors and performs exhaustive search */
struct IndexFlat : IndexFlatCodes {
    explicit IndexFlat(
            idx_t d, ///< dimensionality of the input vectors
            MetricType metric = METRIC_L2);

#ifdef __aarch64__
    IndexFlat(
            idx_t d,
            NumericType ntype,
            MetricType metric = METRIC_L2);
#endif

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /** compute distance with a subset of vectors
     *
     * @param x       query vectors, size n * d
     * @param labels  indices of the vectors that should be compared
     *                for each query vector, size n * k
     * @param distances
     *                corresponding output distances, size n * k
     */
    void compute_distance_subset(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            const idx_t* labels) const;

    // get pointer to the floating point data
    float* get_xb() {
        return (float*)codes.data();
    }
    const float* get_xb() const {
        return (const float*)codes.data();
    }

    IndexFlat() {}

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /* The stanadlone codec interface (just memcopies in this case) */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /* added FP16 function interfaces */
#ifdef __aarch64__
    void search(
            idx_t n,
            const float16_t* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search_ex(
            idx_t n,
            const void* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            NumericType numeric_type,
            const SearchParameters* params = nullptr) const override {
        if (numeric_type == NumericType::Float16) {
            search(n, static_cast<const float16_t*>(x), k, distances, labels, params);
        } else {
            FAISS_THROW_MSG("IndexFlat::search: unsupported numeric type");
        }
    }

    void range_search(
            idx_t n,
            const float16_t* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void range_search_ex(
            idx_t n,
            const void* x,
            float radius,
            RangeSearchResult* result,
            NumericType numeric_type,
            const SearchParameters* params = nullptr) const override {
        if (numeric_type == NumericType::Float16) {
            range_search(n, static_cast<const float16_t*>(x), radius, result, params);
        } else {
            FAISS_THROW_MSG("IndexFlat::range_search: unsupported numeric type");
        }
    }

    void reconstruct(idx_t key, float16_t* recons) const override;

    void reconstruct_ex(idx_t key, void* recons, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            reconstruct(key, static_cast<float16_t*>(recons));
        } else {
            FAISS_THROW_MSG("IndexFlat::reconstruct: unsupported numeric type");
        }
    }

    void sa_encode(idx_t n, const float16_t* x, uint8_t* bytes) const override;

    void sa_encode_ex(idx_t n, const void* x, uint8_t* bytes, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            sa_encode(n, static_cast<const float16_t*>(x), bytes);
        } else {
            FAISS_THROW_MSG("IndexFlat::sa_encode: unsupported numeric type");
        }
    }

    void sa_decode(idx_t n, const uint8_t* bytes, float16_t* x) const override;

    void sa_decode_ex(idx_t n, const uint8_t* bytes, void* x, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            sa_decode(n, bytes, static_cast<float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexFlat::sa_decode: unsupported numeric type");
        }
    }

    template<typename T>
    T* get_xb() {
        return get_xb_impl();
    }

    template<typename T>
    const T* get_xb() const {
        return get_xb_const_impl();
    }

protected:

    float16_t* get_xb_impl() {
        return (float16_t*)codes.data();
    }
    const float16_t* get_xb_const_impl() const {
        return (const float16_t*)codes.data();
    }
#endif
};

struct IndexFlatIP : IndexFlat {
    explicit IndexFlatIP(idx_t d) : IndexFlat(d, METRIC_INNER_PRODUCT) {}
#ifdef __aarch64__
    IndexFlatIP(idx_t d, NumericType ntype) : IndexFlat(d, ntype, METRIC_INNER_PRODUCT) {}
#endif
    IndexFlatIP() {}
};

struct IndexFlatL2 : IndexFlat {
    // Special cache for L2 norms.
    // If this cache is set, then get_distance_computer() returns
    // a special version that computes the distance using dot products
    // and l2 norms.
    std::vector<float> cached_l2norms;

    /**
     * @param d dimensionality of the input vectors
     */
    explicit IndexFlatL2(idx_t d) : IndexFlat(d, METRIC_L2) {}
#ifdef __aarch64__
    IndexFlatL2(idx_t d, NumericType ntype) : IndexFlat(d, ntype, METRIC_L2) {}
#endif
    IndexFlatL2() {}

    // override for l2 norms cache.
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    // compute L2 norms
    void sync_l2norms();
#ifdef __aarch64__
    void sync_l2norms(NumericType ntype);
#endif
    // clear L2 norms
    void clear_l2norms();
};

/// optimized version for 1D "vectors".
struct IndexFlat1D : IndexFlatL2 {
    bool continuous_update = true; ///< is the permutation updated continuously?

    std::vector<idx_t> perm; ///< sorted database indices

    explicit IndexFlat1D(bool continuous_update = true);

    /// if not continuous_update, call this between the last add and
    /// the first search
    void update_permutation();

    void add(idx_t n, const float* x) override;

    void reset() override;

    /// Warn: the distances returned are L1 not L2
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss

#endif

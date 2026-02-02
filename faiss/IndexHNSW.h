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

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>
#ifdef __aarch64__
#include <faiss/utils/fp16-arm.h>
#endif

namespace faiss {

struct IndexHNSW;

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSW : Index {
    typedef HNSW::storage_idx_t storage_idx_t;

    // the link strcuture
    HNSW hnsw;

    // the sequential storage
    bool own_fields = false;
    Index* storage = nullptr;
#ifdef KRL
    // add for quanting
    int quant_bits = 32;
    float quant_scale = 1.0;
    void requant(float scale);

    // add for reordering
    bool apply_reorder = true;
    size_t perm_size = 0;
    faiss::idx_t* perm = nullptr;
#endif
    explicit IndexHNSW(int d = 0, int M = 32, MetricType metric = METRIC_L2);
    explicit IndexHNSW(Index* storage, int M = 32);
#ifdef __aarch64__
    IndexHNSW(int d, int M, NumericType ntype, MetricType metric);
    IndexHNSW(Index* storage, NumericType ntype, int M = 32);
#endif

    ~IndexHNSW() override;

    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
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

    void reset() override;

    void shrink_level_0_neighbors(int size);

    /** Perform search only on level 0, given the starting points for
     * each vertex.
     *
     * @param search_type 1:perform one search per nprobe, 2: enqueue
     *                    all entry points
     */
    void search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    /// alternative graph building
    void init_level_0_from_knngraph(int k, const float* D, const idx_t* I);

    /// alternative graph building
    void init_level_0_from_entry_points(
            int npt,
            const storage_idx_t* points,
            const storage_idx_t* nearests);

    // reorder links from nearest to farthest
    void reorder_links();

    void link_singletons();

    void permute_entries(const idx_t* perm);

    /* added FP16 function interfaces */
#ifdef __aarch64__
    void add(idx_t n, const float16_t* x) override;

    void add_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            add(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexHNSW::add: unsupported numeric type");
        }
    }

    void train(idx_t n, const float16_t* x) override;

    void train_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            train(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexHNSW::train: unsupported numeric type");
        }
    }

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
            FAISS_THROW_MSG("IndexHNSW::search: unsupported numeric type");
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
            FAISS_THROW_MSG("IndexHNSW::range_search: unsupported numeric type");
        }
    }

    void reconstruct(idx_t key, float16_t* recons) const override;

    void reconstruct_ex(idx_t key, void* recons, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            reconstruct(key, static_cast<float16_t*>(recons));
        } else {
            FAISS_THROW_MSG("IndexHNSW::reconstruct: unsupported numeric type");
        }
    }

    void search_level_0(
            idx_t n,
            const float16_t* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    void search_level_0_ex(
            idx_t n,
            const void* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            NumericType numeric_type,
            int nprobe = 1,
            int search_type = 1) const {
        if (numeric_type == NumericType::Float16) {
            search_level_0(n, static_cast<const float16_t*>(x), k, nearest, nearest_d, distances, labels, nprobe, search_type);
        } else {
            FAISS_THROW_MSG("IndexHNSW::search_level_0: unsupported numeric type");
        }
    }
#endif
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWFlat : IndexHNSW {
    IndexHNSWFlat();
    IndexHNSWFlat(int d, int M, MetricType metric = METRIC_L2);
#ifdef __aarch64__
    IndexHNSWFlat(int d, int M, NumericType ntype = NumericType::Float16, MetricType metric = METRIC_L2);
#endif
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWPQ : IndexHNSW {
    IndexHNSWPQ();
    IndexHNSWPQ(int d, int pq_m, int M, int pq_nbits = 8);
    void train(idx_t n, const float* x) override;

#ifdef __aarch64__
    /* explicit FP32 function interface */
    void add(idx_t n, const float* x) override;

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

    void search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;
    
    /* added FP16 function interfaces */
    void train(idx_t n, const float16_t* x) override;

    void train_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            train(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexHNSWPQ::train: unsupported numeric type");
        }
    }

    void add(idx_t n, const float16_t* x) override;

    void add_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            add(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexHNSWPQ::add: unsupported numeric type");
        }
    }
    
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
            FAISS_THROW_MSG("IndexHNSWPQ::search: unsupported numeric type");
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
            FAISS_THROW_MSG("IndexHNSWPQ::range_search: unsupported numeric type");
        }
    }

    void reconstruct(idx_t key, float16_t* recons) const override;

    void reconstruct_ex(idx_t key, void* recons, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            reconstruct(key, static_cast<float16_t*>(recons));
        } else {
            FAISS_THROW_MSG("IndexHNSWPQ::reconstruct: unsupported numeric type");
        }
    }

    void search_level_0(
            idx_t n,
            const float16_t* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    void search_level_0_ex(
            idx_t n,
            const void* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            NumericType numeric_type,
            int nprobe = 1,
            int search_type = 1) const {
        if (numeric_type == NumericType::Float16) {
            search_level_0(n, static_cast<const float16_t*>(x), k, nearest, nearest_d, distances, labels, nprobe, search_type);
        } else {
            FAISS_THROW_MSG("IndexHNSWPQ::search_level_0: unsupported numeric type");
        }
    }
#endif
};

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWSQ : IndexHNSW {
    IndexHNSWSQ();
    IndexHNSWSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M,
            MetricType metric = METRIC_L2);
#ifdef __aarch64__
    /* explicit FP32 function interface */
    void add(idx_t n, const float* x) override;

    void train(idx_t n, const float* x) override;

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

    void search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    /* added FP16 function interfaces */
    void add(idx_t n, const float16_t* x) override;

    void add_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            add(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexHNSWSQ::add: unsupported numeric type");
        }
    }

    void train(idx_t n, const float16_t* x) override;

    void train_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            train(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexHNSWSQ::train: unsupported numeric type");
        }
    }

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
            FAISS_THROW_MSG("IndexHNSWSQ::search: unsupported numeric type");
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
            FAISS_THROW_MSG("IndexHNSWSQ::range_search: unsupported numeric type");
        }
    }

    void reconstruct(idx_t key, float16_t* recons) const override;

    void reconstruct_ex(idx_t key, void* recons, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            reconstruct(key, static_cast<float16_t*>(recons));
        } else {
            FAISS_THROW_MSG("IndexHNSWSQ::reconstruct: unsupported numeric type");
        }
    }

    void search_level_0(
            idx_t n,
            const float16_t* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    void search_level_0_ex(
            idx_t n,
            const void* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            NumericType numeric_type,
            int nprobe = 1,
            int search_type = 1) const {
        if (numeric_type == NumericType::Float16) {
            search_level_0(n, static_cast<const float16_t*>(x), k, nearest, nearest_d, distances, labels, nprobe, search_type);
        } else {
            FAISS_THROW_MSG("IndexHNSWSQ::search_level_0: unsupported numeric type");
        }
    }
#endif
};

/** 2-level code structure with fast random access
 */
struct IndexHNSW2Level : IndexHNSW {
    IndexHNSW2Level();
    IndexHNSW2Level(Index* quantizer, size_t nlist, int m_pq, int M);

    void flip_to_ivf();

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

#ifdef __aarch64__
    /* explicit FP32 function interface */
    void add(idx_t n, const float* x) override;

    void train(idx_t n, const float* x) override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    /* added FP16 function interfaces */
    void add(idx_t n, const float16_t* x) override;

    void add_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            add(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexHNSW2Level::add: unsupported numeric type");
        }
    }

    void train(idx_t n, const float16_t* x) override;

    void train_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            train(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexHNSW2Level::train: unsupported numeric type");
        }
    }

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
            FAISS_THROW_MSG("IndexHNSW2Level::search: unsupported numeric type");
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
            FAISS_THROW_MSG("IndexHNSW2Level::range_search: unsupported numeric type");
        }
    }

    void reconstruct(idx_t key, float16_t* recons) const override;

    void reconstruct_ex(idx_t key, void* recons, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            reconstruct(key, static_cast<float16_t*>(recons));
        } else {
            FAISS_THROW_MSG("IndexHNSW2Level::reconstruct: unsupported numeric type");
        }
    }

    void search_level_0(
            idx_t n,
            const float16_t* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    void search_level_0_ex(
            idx_t n,
            const void* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            NumericType numeric_type,
            int nprobe = 1,
            int search_type = 1) const {
        if (numeric_type == NumericType::Float16) {
            search_level_0(n, static_cast<const float16_t*>(x), k, nearest, nearest_d, distances, labels, nprobe, search_type);
        } else {
            FAISS_THROW_MSG("IndexHNSW2Level::search_level_0: unsupported numeric type");
        }
    }
#endif
};

} // namespace faiss

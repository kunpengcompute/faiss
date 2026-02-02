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

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/sorting.h>
#include <faiss/utils/utils.h>
#include <cstring>
#ifdef __aarch64__
#include <stdexcept>
#endif
#if defined(OPTI_IVFPQ)
#include <iostream>
#include <arm_neon.h>
#include <faiss/utils/arm/asm/distances_simd.h>
#elif defined(KRL)
#include <iostream>
#include <arm_neon.h>
extern "C" {
#include <faiss/sra_krl/include/krl.h>
}
#endif

namespace faiss {

#ifdef __aarch64__
namespace {
    size_t calculate_code_size(idx_t d, NumericType ntype) {
        switch (ntype) {
            case NumericType::Float32:
                return sizeof(float) * d;
            case NumericType::Float16:
                return sizeof(float16_t) * d;
            default:
                throw std::invalid_argument("Unsupported numeric type");
        }
    }
}
#endif

IndexFlat::IndexFlat(idx_t d, MetricType metric)
        : IndexFlatCodes(sizeof(float) * d, d, metric) {}

#ifdef __aarch64__
IndexFlat::IndexFlat(idx_t d, NumericType ntype, MetricType metric)
        : IndexFlatCodes(calculate_code_size(d, ntype), d, ntype, metric) {}
#endif

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;
#if defined(OPTI_IVFPQ)
    if (!sel &&
        (metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT) &&
		__builtin_expect(k <= ntotal, 1)) {

        const float* xb = get_xb();

        std::vector<int64_t> base_idx((size_t)ntotal);
        for (int64_t j = 0; j < (int64_t)ntotal; ++j) {
            base_idx[(size_t)j] = j;
        }

#pragma omp parallel if (n > 1)
        {
            std::vector<float> base_dis((size_t)ntotal);

#pragma omp for
            for (idx_t i = 0; i < n; ++i) {
                const float* q = x + (size_t)i * d;
                float* dist_i = distances + (size_t)i * k;
                idx_t* lab_i = labels + (size_t)i * k;

                if (metric_type == METRIC_L2) {
                    L2sqrNy(base_dis.data(), q, xb, (size_t)ntotal, (size_t)d);
                    SelectL2Topk((int64_t)k, (int64_t*)lab_i, dist_i,
                                (int64_t)ntotal, base_idx.data(), base_dis.data());
                } else {
                    IPNy(base_dis.data(), q, xb, (size_t)ntotal, (size_t)d);
                    SelectIPTopk((int64_t)k, (int64_t*)lab_i, dist_i,
                                 (int64_t)ntotal, base_idx.data(), base_dis.data());
                }
            }
        }
        return;
    }
#elif defined(KRL)
    if(use_handle && 4 * k < ntotal && !sel) {
        #pragma omp parallel for if (n > 1)
        for (int i = 0; i < n; ++i) {
            krl_reorder_2_vector_continuous(kdh, ntotal, 0, x + i * d, k, distances + i * k, labels + i * k, d);
        }
        return;
    }
#endif
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps
    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res, sel);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, sel);
    } else if (is_similarity_metric(metric_type)) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_extra_metrics(
                x, get_xb(), d, n, ntotal, metric_type, metric_arg, &res);
    } else {
        FAISS_THROW_IF_NOT(!sel);
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_extra_metrics(
                x, get_xb(), d, n, ntotal, metric_type, metric_arg, &res);
    }
}

void IndexFlat::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

namespace {

struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
#ifdef KRL
        return fvec_L2sqr(reinterpret_cast<const float*>(codes + j * code_size), 
            reinterpret_cast<const float*>(codes + i * code_size), d);
#else
        return fvec_L2sqr(b + j * d, b + i * d, d);
#endif
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

#ifdef KRL
    void distances_multi_codes(const int64_t* idx, float* dis, int ny) override {
        ndis += ny;
        krl_L2sqr_by_idx(dis, q, reinterpret_cast<const float*>(codes), idx, d, ny, ny);
    }
#endif
};

#ifdef __aarch64__
struct FlatL2DisFP16 : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float16_t* q;
    const float16_t* b;
    size_t ndis;

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr_f16(q, (float16_t*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
#ifdef KRL
        return fvec_L2sqr_f16(reinterpret_cast<const float16_t*>(codes + j * code_size), 
            reinterpret_cast<const float16_t*>(codes + i * code_size), d);
#else
        return fvec_L2sqr_f16(b + j * d, b + i * d, d);
#endif
    }

    explicit FlatL2DisFP16(const IndexFlat& storage, const float16_t* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
                      d(storage.d),
                      nb(storage.ntotal),
                      q(q),
                      b(storage.get_xb<float16_t>()),
                      ndis(0) {}

    void set_query(const float16_t* x) {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float16_t* __restrict y0 =
                reinterpret_cast<const float16_t*>(codes + idx0 * code_size);
        const float16_t* __restrict y1 =
                reinterpret_cast<const float16_t*>(codes + idx1 * code_size);
        const float16_t* __restrict y2 =
                reinterpret_cast<const float16_t*>(codes + idx2 * code_size);
        const float16_t* __restrict y3 =
                reinterpret_cast<const float16_t*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_L2sqr_batch_4_f16(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

#ifdef KRL
    void distances_multi_codes(const int64_t* idx, float* dis, int ny) override {
        ndis += ny;
        krl_L2sqr_by_idx_f16f32(dis, reinterpret_cast<const uint16_t*>(q), reinterpret_cast<const uint16_t*>(codes), idx, d, ny, ny);
    }
#endif
};
#endif

struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float symmetric_dis(idx_t i, idx_t j) final override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product(q, (const float*)code, d);
    }

    explicit FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

#ifdef KRL
    void set_base(const float* x) override {
        std::cerr << "TypeError, struct FlatIPDis can't use set_base func!\n" << std::endl;
    }
#endif

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

#ifdef KRL
    void distances_multi_codes(const int64_t* idx, float* dis, int ny) override {
        ndis += ny;
        krl_inner_product_by_idx(dis, q, reinterpret_cast<const float*>(codes), idx, d, ny, ny);
    }
#endif
};

#ifdef __aarch64__
struct FlatIPDisFP16 : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float16_t* q;
    const float16_t* b;
    size_t ndis;

    float symmetric_dis(idx_t i, idx_t j) final override {
        return fvec_inner_product_f16(b + j * d, b + i * d, d);
    }

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product_f16(q, (const float16_t*)code, d);
    }

    explicit FlatIPDisFP16(const IndexFlat& storage, const float16_t* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
                      d(storage.d),
                      nb(storage.ntotal),
                      q(q),
                      b(storage.get_xb<float16_t>()),
                      ndis(0) {}

    void set_query(const float16_t* x) {
        q = x;
    }

#ifdef KRL
    void set_base(const float16_t* x) {
        std::cerr << "TypeError, struct FlatIPDisFP16 can't use set_base func!\n" << std::endl;
    }
#endif

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float16_t* __restrict y0 =
                reinterpret_cast<const float16_t*>(codes + idx0 * code_size);
        const float16_t* __restrict y1 =
                reinterpret_cast<const float16_t*>(codes + idx1 * code_size);
        const float16_t* __restrict y2 =
                reinterpret_cast<const float16_t*>(codes + idx2 * code_size);
        const float16_t* __restrict y3 =
                reinterpret_cast<const float16_t*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4_f16(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

#ifdef KRL
    void distances_multi_codes(const int64_t* idx, float* dis, int ny) override {
        ndis += ny;
        krl_inner_product_by_idx_f16f32(dis, reinterpret_cast<const uint16_t*>(q), reinterpret_cast<const uint16_t*>(codes), idx, d, ny, ny);
    }
#endif
};
#endif

} // namespace

FlatCodesDistanceComputer* IndexFlat::get_FlatCodesDistanceComputer() const {
    if (numeric_type == NumericType::Float16) {
#ifdef __aarch64__
        if (metric_type == METRIC_L2) {
            return new FlatL2DisFP16(*this);
        } else if (metric_type == METRIC_INNER_PRODUCT) {
            return new FlatIPDisFP16(*this);
        } else {
            throw std::invalid_argument("Unsupported metric type");
        }
#else
        throw std::invalid_argument("IndexFlat::get_FlatCodesDistanceComputer: unsupported numeric type");
#endif
    } else {
        if (metric_type == METRIC_L2) {
            return new FlatL2Dis(*this);
        } else if (metric_type == METRIC_INNER_PRODUCT) {
            return new FlatIPDis(*this);
        } else {
            return get_extra_distance_computer(
                    d, metric_type, metric_arg, ntotal, get_xb());
        }
    }

}

void IndexFlat::reconstruct(idx_t key, float* recons) const {
    memcpy(recons, &(codes[key * code_size]), code_size);
}

void IndexFlat::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n > 0) {
#ifdef KRL
        for (size_t i = 0; i < n; ++i) {
            memcpy(bytes + i * code_size, x + i * d, sizeof(float) * d);
        }
#else
        memcpy(bytes, x, sizeof(float) * d * n);
#endif
    }
}

void IndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n > 0) {
#ifdef KRL
        for (size_t i = 0; i < n; ++i) {
            memcpy(x + i * d, bytes + i * code_size, sizeof(float) * d);
        }
#else
        memcpy(x, bytes, sizeof(float) * d * n);
#endif
    }
}

/***************************************************
 * IndexFlatL2
 ***************************************************/

namespace {
struct FlatL2WithNormsDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    const float* l2norms;
    float query_l2norm;

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float operator()(const idx_t i) final override {
        const float* __restrict y =
                reinterpret_cast<const float*>(codes + i * code_size);

        prefetch_L2(l2norms + i);
        const float dp0 = fvec_inner_product(q, y, d);
        return query_l2norm + l2norms[i] - 2 * dp0;
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        const float* __restrict yi =
                reinterpret_cast<const float*>(codes + i * code_size);
        const float* __restrict yj =
                reinterpret_cast<const float*>(codes + j * code_size);

        prefetch_L2(l2norms + i);
        prefetch_L2(l2norms + j);
        const float dp0 = fvec_inner_product(yi, yj, d);
        return l2norms[i] + l2norms[j] - 2 * dp0;
    }

    explicit FlatL2WithNormsDis(
            const IndexFlatL2& storage,
            const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0),
              l2norms(storage.cached_l2norms.data()),
              query_l2norm(0) {}

    void set_query(const float* x) override {
        q = x;
        query_l2norm = fvec_norm_L2sqr(q, d);
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        prefetch_L2(l2norms + idx0);
        prefetch_L2(l2norms + idx1);
        prefetch_L2(l2norms + idx2);
        prefetch_L2(l2norms + idx3);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = query_l2norm + l2norms[idx0] - 2 * dp0;
        dis1 = query_l2norm + l2norms[idx1] - 2 * dp1;
        dis2 = query_l2norm + l2norms[idx2] - 2 * dp2;
        dis3 = query_l2norm + l2norms[idx3] - 2 * dp3;
    }
};

#ifdef __aarch64__
struct FlatL2WithNormsDisFP16 : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float16_t* q;
    const float16_t* b;
    size_t ndis;

    const float* l2norms;
    float query_l2norm;

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_L2sqr_f16(q, (float16_t*)code, d);
    }

    float operator()(const idx_t i) final override {
        const float16_t* __restrict y =
                reinterpret_cast<const float16_t*>(codes + i * code_size);

        prefetch_L2(l2norms + i);
        const float dp0 = fvec_inner_product_f16(q, y, d);
        return query_l2norm + l2norms[i] - 2 * dp0;
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        const float16_t* __restrict yi =
                reinterpret_cast<const float16_t*>(codes + i * code_size);
        const float16_t* __restrict yj =
                reinterpret_cast<const float16_t*>(codes + j * code_size);

        prefetch_L2(l2norms + i);
        prefetch_L2(l2norms + j);
        const float dp0 = fvec_inner_product_f16(yi, yj, d);
        return l2norms[i] + l2norms[j] - 2 * dp0;
    }

    explicit FlatL2WithNormsDisFP16(
            const IndexFlatL2& storage,
            const float16_t* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
                      d(storage.d),
                      nb(storage.ntotal),
                      q(q),
                      b(storage.get_xb<float16_t>()),
                      ndis(0),
                      l2norms(storage.cached_l2norms.data()),
                      query_l2norm(0) {}

    void set_query(const float16_t* x) {
        q = x;
        query_l2norm = fvec_norm_L2sqr_f16(q, d);
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float16_t* __restrict y0 =
                reinterpret_cast<const float16_t*>(codes + idx0 * code_size);
        const float16_t* __restrict y1 =
                reinterpret_cast<const float16_t*>(codes + idx1 * code_size);
        const float16_t* __restrict y2 =
                reinterpret_cast<const float16_t*>(codes + idx2 * code_size);
        const float16_t* __restrict y3 =
                reinterpret_cast<const float16_t*>(codes + idx3 * code_size);

        prefetch_L2(l2norms + idx0);
        prefetch_L2(l2norms + idx1);
        prefetch_L2(l2norms + idx2);
        prefetch_L2(l2norms + idx3);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4_f16(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = query_l2norm + l2norms[idx0] - 2 * dp0;
        dis1 = query_l2norm + l2norms[idx1] - 2 * dp1;
        dis2 = query_l2norm + l2norms[idx2] - 2 * dp2;
        dis3 = query_l2norm + l2norms[idx3] - 2 * dp3;
    }
};
#endif

} // namespace

void IndexFlatL2::sync_l2norms() {
    cached_l2norms.resize(ntotal);
    fvec_norms_L2sqr(
            cached_l2norms.data(),
            reinterpret_cast<const float*>(codes.data()),
            d,
            ntotal);
}

#ifdef __aarch64__
void IndexFlatL2::sync_l2norms(NumericType ntype) {
    cached_l2norms.resize(ntotal);
    if (ntype == NumericType::Float16) {
        fvec_norms_L2sqr_f16(
                cached_l2norms.data(),
                reinterpret_cast<const float16_t*>(codes.data()),
                d,
                ntotal);
    } else {
        fvec_norms_L2sqr(
                cached_l2norms.data(),
                reinterpret_cast<const float*>(codes.data()),
                d,
                ntotal);
    }

}
#endif

void IndexFlatL2::clear_l2norms() {
    cached_l2norms.clear();
    cached_l2norms.shrink_to_fit();
}

FlatCodesDistanceComputer* IndexFlatL2::get_FlatCodesDistanceComputer() const {
#ifdef __aarch64__
    if (numeric_type == NumericType::Float16 && metric_type == METRIC_L2) {
        if (!cached_l2norms.empty()) {
            return new FlatL2WithNormsDisFP16(*this);
        }
    }
#endif
    if (metric_type == METRIC_L2) {
        if (!cached_l2norms.empty()) {
            return new FlatL2WithNormsDis(*this);
        }
    }

    return IndexFlat::get_FlatCodesDistanceComputer();
}

/***************************************************
 * IndexFlat1D
 ***************************************************/

IndexFlat1D::IndexFlat1D(bool continuous_update)
        : IndexFlatL2(1), continuous_update(continuous_update) {}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation() {
    perm.resize(ntotal);
    if (ntotal < 1000000) {
        fvec_argsort(ntotal, get_xb(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel(ntotal, get_xb(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add(idx_t n, const float* x) {
    IndexFlatL2::add(n, x);
    if (continuous_update)
        update_permutation();
}

void IndexFlat1D::reset() {
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            perm.size() == ntotal, "Call update_permutation before search");
    const float* xb = get_xb();

#pragma omp parallel for if (n > 10000)
    for (idx_t i = 0; i < n; i++) {
        float q = x[i]; // query
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (ntotal == 0) {
            for (idx_t j = 0; j < k; j++) {
                I[j] = -1;
                D[j] = HUGE_VAL;
            }
            goto done;
        }

        if (xb[perm[i0]] > q) {
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q)
                i0 = imed;
            else
                i1 = imed;
        }

        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--;
                wp++;
                if (i0 < 0) {
                    goto finish_right;
                }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++;
                wp++;
                if (i1 >= ntotal) {
                    goto finish_left;
                }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
    done:;
    }
}

/* added FP16 function interfaces */
#ifdef __aarch64__
void IndexFlat::search(
            idx_t n,
            const float16_t* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;
#ifdef KRL
    if(use_handle && 4 * k < ntotal && !sel) {
        #pragma omp parallel for if (n > 1)
        for (int i = 0; i < n; ++i) {
            krl_reorder_2_vector_continuous_f16(kdh, ntotal, 0, x + i * d, k, distances + i * k, labels + i * k, d);
        }
            return;
    }
#endif
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps
    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product_f16(x, get_xb<float16_t>(), d, n, ntotal, &res, sel);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr_f16(x, get_xb<float16_t>(), d, n, ntotal, &res, nullptr, sel);
    } else {
        FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::range_search(
        idx_t n,
        const float16_t* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product_f16(
                    x, get_xb<float16_t>(), d, n, ntotal, radius, result, sel);
            break;
        case METRIC_L2:
            range_search_L2sqr_f16(x, get_xb<float16_t>(), d, n, ntotal, radius, result, sel);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::reconstruct(idx_t key, float16_t* recons) const {
    int ret = SafeMemory::CheckAndMemcpy(recons, code_size, &(codes[key * code_size]), code_size);
}

void IndexFlat::sa_encode(idx_t n, const float16_t* x, uint8_t* bytes) const {
    if (n > 0) {
        for (size_t i = 0; i < n; ++i) {
            int ret = SafeMemory::CheckAndMemcpy(bytes + i * code_size, sizeof(float16_t) * d, x + i * d, sizeof(float16_t) * d);
        }
    }
}

void IndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float16_t* x) const {
    if (n > 0) {
        for (size_t i = 0; i < n; ++i) {
            int ret = SafeMemory::CheckAndMemcpy(x + i * d, sizeof(float16_t) * d, bytes + i * code_size, sizeof(float16_t) * d);
        }
    }
}
#endif
} // namespace faiss

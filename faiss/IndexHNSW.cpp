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

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <queue>
#include <unordered_set>

#include <sys/stat.h>
#include <sys/types.h>
#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

extern "C" {
#ifdef KRL
#include <arm_neon.h>
#include <faiss/sra_krl/include/krl.h>
#endif

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;

HNSWStats hnsw_stats;

#ifdef KRL
void graphBFSPerm(faiss::IndexHNSW *index, faiss::idx_t* perm)
{
    size_t permInd = 0;
    const auto hnsw = index->hnsw;
    const size_t nV = hnsw.levels.size();
    std::vector<bool> visited(nV, false);

    std::queue<faiss::idx_t> BFSQueue;
    for (size_t v = 0; v < nV; ++v) {
        if (hnsw.levels[v] > 1) {
            BFSQueue.push(v);
            perm[permInd++] = v;
            visited[v] = true;
        }
    }

    while (!BFSQueue.empty()) {
        const auto v = BFSQueue.front();
        BFSQueue.pop();

        size_t begin, end;
        hnsw.neighbor_range(v, 0, &begin, &end);
        for (size_t v = begin; v < end; ++v) {
            const auto vNeighbor = hnsw.neighbors[v];
            if (vNeighbor < 0) {
                break;
            }
            if (!visited[vNeighbor]) {
                BFSQueue.push(vNeighbor);
                perm[permInd++] = vNeighbor;
                visited[vNeighbor] = true;
            }   
        }
    }

    for(int i = 0; i < nV; ++i) {
        if(visited[i] == false) {
            perm[permInd++] = i;
        }
    }
}
#endif

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

#ifdef __aarch64__
    void set_query(const float16_t* x) override {
        basedis->set_query(x);
    }
#endif

#ifdef KRL
	void set_base(const float* x) override {}
#ifdef __aarch64__
	void set_base(const float16_t* x) override {}
#endif
#endif
    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        basedis->distances_batch_4(
                idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
        dis0 = -dis0;
        dis1 = -dis1;
        dis2 = -dis2;
        dis3 = -dis3;
    }

#ifdef KRL
    void distances_multi_codes(const int64_t* idx, float* dis, int ny) {
        basedis->distances_multi_codes(idx, dis, ny);
        for(int i = 0; i < ny; ++i) {
            dis[i] = -dis[i];
        }
    }
#endif
    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() {
        delete basedis;
    }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

template<typename T>
void hnsw_add_vertices(
        IndexHNSW& index_hnsw,
        size_t n0,
        size_t n,
        const T* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_hnsw.d;
    HNSW& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hnsw.d * hnsw.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_hnsw.storage));
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M, MetricType metric)
        : Index(d, metric), hnsw(M) {}

IndexHNSW::IndexHNSW(Index* storage, int M)
        : Index(storage->d, storage->metric_type), hnsw(M), storage(storage) {}

#ifdef __aarch64__
IndexHNSW::IndexHNSW(int d, int M, NumericType ntype, MetricType metric)
        : Index(d, ntype, metric), hnsw(M) {}

IndexHNSW::IndexHNSW(Index* storage, NumericType ntype, int M)
        : Index(storage->d, ntype, storage->metric_type), hnsw(M), storage(storage) {}
#endif

IndexHNSW::~IndexHNSW() {
#ifdef KRL
    if (own_fields) {
        delete storage;
        own_fields = false;
    }
    if (apply_reorder && perm != nullptr) {
        delete[] perm;
        perm = nullptr;
    }
#else
    if (own_fields) {
        delete storage;
    }
#endif
}

void IndexHNSW::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}
#ifdef KRL
struct FlatL2DisSQ8 : DistanceComputer {
    const uint8_t* q8;
    const uint8_t* b8;
    size_t d;

    explicit FlatL2DisSQ8(size_t dim) : d(dim) {};

    void set_query(const float* x) final override{
        q8 = (uint8_t*)x;
    };

    void set_base(const float* x) final override {
        b8 = (uint8_t*)x;
    }

    void set_query(const float16_t* x) final override{
        q8 = (uint8_t*)x;
    };

    void set_base(const float16_t* x) final override {
        b8 = (uint8_t*)x;
    }

    float operator()(idx_t i) override {
		uint32_t ret;
		krl_L2sqr_u8u32(q8, b8 + i * d, d, &ret);
        return (float)ret;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
		uint32_t ret;
		krl_L2sqr_u8u32(q8 + i * d, q8 + j * d, d, &ret);
        return (float)ret;
    }

    // compute with krl
    void distances_multi_codes(const int64_t* idx, float* dis, int ny) final override {
        krl_L2sqr_by_idx_u8f32(dis, q8, b8, idx, d, ny);
    }

    virtual ~FlatL2DisSQ8() { };
};

struct FlatL2DisSQ16 : DistanceComputer {
    const float16_t* q16;
    const float16_t* b16;
    size_t d;

    explicit FlatL2DisSQ16(size_t dim) : d(dim){};

    void set_query(const float* x) final override {
        q16 = (float16_t*)x;
    }

    void set_base(const float* x) final override {
        b16 = (float16_t*)x;
    }

    void set_query(const float16_t* x) final override {
        q16 = x;
    }

    void set_base(const float16_t* x) final override {
        b16 = x;
    }

    float operator()(idx_t i) override {
		float ret;
#ifdef USE_SVE2
		krl_L2sqr_f16f32_sve2((const uint16_t*)q16, (const uint16_t*)b16 + i * d, d, &ret);
#else
		krl_L2sqr_f16f32((const uint16_t*)q16, (const uint16_t*)b16 + i * d, d, &ret);
#endif
        return ret;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
		float ret;
#ifdef USE_SVE2
		krl_L2sqr_f16f32_sve2((const uint16_t*)b16 + i * d, (const uint16_t*)b16 + j * d, d, &ret);
#else
		krl_L2sqr_f16f32((const uint16_t*)b16 + i * d, (const uint16_t*)b16 + j * d, d, &ret);
#endif
        return ret;
    }

    // compute with krl
    void distances_multi_codes(const int64_t* idx, float* dis, int ny) final override {
#ifdef USE_SVE2
        krl_L2sqr_by_idx_f16f32_sve2(dis, (const uint16_t*)q16, (const uint16_t*)b16, idx, d, ny);
#else
        krl_L2sqr_by_idx_f16f32(dis, (const uint16_t*)q16, (const uint16_t*)b16, idx, d, ny);
#endif
    }

    virtual ~FlatL2DisSQ16() {};
};

struct FlatIPDisSQ16 : DistanceComputer {
    const float16_t* q16;
    const float16_t* b16;
    size_t d;

    explicit FlatIPDisSQ16(size_t dim) : d(dim){ };

    void set_query(const float* x) final override {
        q16 = (float16_t*)x;
    }

    void set_base(const float* x) final override {
        b16 = (float16_t*)x;
    }

    float operator()(idx_t i) override {
		float ret;
		krl_negative_ipdis_f16f32((const uint16_t*)q16, (const uint16_t*)b16 + i * d, d, &ret);
        return ret;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
		float ret;
		krl_negative_ipdis_f16f32((const uint16_t*)b16 + i * d, (const uint16_t*)b16 + j * d, d, &ret);
        return ret;
    }

    // compute with krl
    void distances_multi_codes(const int64_t* idx, float* dis, int ny) final override {
        krl_negative_inner_product_by_idx_f16f32(dis, (const uint16_t*)q16, (const uint16_t*)b16, idx, d, ny);
    }

    virtual ~FlatIPDisSQ16() { };
};


void quant_f16_noscale(const float* src, idx_t d, float16_t* out) {
    idx_t l = 0;
    constexpr idx_t single_loop = 4;
    constexpr idx_t multi_loop = 16;
    for(; l + multi_loop <= d; l += multi_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);
        const float16x4_t neon_c1 = vcvt_f16_f32(neon_a1);
        const float16x4_t neon_c2 = vcvt_f16_f32(neon_a2);
        const float16x4_t neon_c3 = vcvt_f16_f32(neon_a3);
        const float16x4_t neon_c4 = vcvt_f16_f32(neon_a4);
        vst1_f16(out + l, neon_c1);
        vst1_f16(out + l + 4, neon_c2);
        vst1_f16(out + l + 8, neon_c3);
        vst1_f16(out + l + 12, neon_c4);
    } 
    for(; l + single_loop <= d; l += single_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        const float16x4_t neon_c1 = vcvt_f16_f32(neon_a1);
        vst1_f16(out + l, neon_c1);
    }
    for(; l < d; ++l) {
        out[l] = (float16_t)(src[l]);
    }
}

void quant_u8_noscale(const float* src, idx_t d, uint8_t* out) {
    idx_t l = 0;    
    constexpr idx_t multi_loop = 32;
    constexpr idx_t double_loop = 16;
    constexpr idx_t single_loop = 8;
    for(; l + multi_loop <= d; l += multi_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);
        float32x4_t neon_a5 = vld1q_f32(src + l + 16);
        float32x4_t neon_a6 = vld1q_f32(src + l + 20);
        float32x4_t neon_a7 = vld1q_f32(src + l + 24);
        float32x4_t neon_a8 = vld1q_f32(src + l + 28);

        const uint32x4_t neon_b1 = vcvtnq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtnq_u32_f32(neon_a2);
        const uint32x4_t neon_b3 = vcvtnq_u32_f32(neon_a3);
        const uint32x4_t neon_b4 = vcvtnq_u32_f32(neon_a4);
        const uint32x4_t neon_b5 = vcvtnq_u32_f32(neon_a5);
        const uint32x4_t neon_b6 = vcvtnq_u32_f32(neon_a6);
        const uint32x4_t neon_b7 = vcvtnq_u32_f32(neon_a7);
        const uint32x4_t neon_b8 = vcvtnq_u32_f32(neon_a8);

        const uint8x16_t neon_c1 = vpaddq_u8(vreinterpretq_u8_u32(neon_b1), vreinterpretq_u8_u32(neon_b2));
        const uint8x16_t neon_c2 = vpaddq_u8(vreinterpretq_u8_u32(neon_b3), vreinterpretq_u8_u32(neon_b4));
        const uint8x16_t neon_c3 = vpaddq_u8(vreinterpretq_u8_u32(neon_b5), vreinterpretq_u8_u32(neon_b6));
        const uint8x16_t neon_c4 = vpaddq_u8(vreinterpretq_u8_u32(neon_b7), vreinterpretq_u8_u32(neon_b8));
        
        const uint8x16_t neon_d1 = vpaddq_u8(neon_c1, neon_c2);
        const uint8x16_t neon_d2 = vpaddq_u8(neon_c3, neon_c4);

        vst1q_u8(out + l, neon_d1);
        vst1q_u8(out + l + 16, neon_d2);
    } 
    if(d & double_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);

        const uint32x4_t neon_b1 = vcvtnq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtnq_u32_f32(neon_a2);
        const uint32x4_t neon_b3 = vcvtnq_u32_f32(neon_a3);
        const uint32x4_t neon_b4 = vcvtnq_u32_f32(neon_a4);

        const uint8x16_t neon_c1 = vpaddq_u8(vreinterpretq_u8_u32(neon_b1), vreinterpretq_u8_u32(neon_b2));
        const uint8x16_t neon_c2 = vpaddq_u8(vreinterpretq_u8_u32(neon_b3), vreinterpretq_u8_u32(neon_b4));

        const uint8x16_t neon_d1 = vpaddq_u8(neon_c1, neon_c2);

        vst1q_u8(out + l, neon_d1);
        l += double_loop;
    }
    if(d & single_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);

        const uint32x4_t neon_b1 = vcvtnq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtnq_u32_f32(neon_a2);

        const uint16x4_t neon_c1 = vmovn_u32(neon_b1);
        const uint16x4_t neon_c2 = vmovn_u32(neon_b2);

        const uint8x8_t neon_d1 = vpadd_u8(vreinterpret_u8_u16(neon_c1), vreinterpret_u8_u16(neon_c2));

        vst1_u8(out + l, neon_d1);
        l += single_loop;
    }
    for(; l < d; ++l) {
        out[l] = (uint8_t)(src[l] + 0.5);
    }
}

void quant_u8_noscale_f16(const float16_t* src, idx_t d, uint8_t* out) {
    idx_t l = 0;    
    constexpr idx_t multi_loop = 32;
    constexpr idx_t double_loop = 16;
    constexpr idx_t single_loop = 8;

    for(; l + multi_loop <= d; l += multi_loop) {
        float16x8_t neon_a1 = vld1q_f16(src + l);
        float16x8_t neon_a2 = vld1q_f16(src + l + 8);
        float16x8_t neon_a3 = vld1q_f16(src + l + 16);
        float16x8_t neon_a4 = vld1q_f16(src + l + 24);

        float32x4_t neon_b1_l = vcvt_f32_f16(vget_low_f16(neon_a1));
        float32x4_t neon_b1_h = vcvt_f32_f16(vget_high_f16(neon_a1));
        float32x4_t neon_b2_l = vcvt_f32_f16(vget_low_f16(neon_a2));
        float32x4_t neon_b2_h = vcvt_f32_f16(vget_high_f16(neon_a2));
        float32x4_t neon_b3_l = vcvt_f32_f16(vget_low_f16(neon_a3));
        float32x4_t neon_b3_h = vcvt_f32_f16(vget_high_f16(neon_a3));
        float32x4_t neon_b4_l = vcvt_f32_f16(vget_low_f16(neon_a4));
        float32x4_t neon_b4_h = vcvt_f32_f16(vget_high_f16(neon_a4));

        uint32x4_t neon_c1_l = vcvtnq_u32_f32(neon_b1_l);
        uint32x4_t neon_c1_h = vcvtnq_u32_f32(neon_b1_h);
        uint32x4_t neon_c2_l = vcvtnq_u32_f32(neon_b2_l);
        uint32x4_t neon_c2_h = vcvtnq_u32_f32(neon_b2_h);
        uint32x4_t neon_c3_l = vcvtnq_u32_f32(neon_b3_l);
        uint32x4_t neon_c3_h = vcvtnq_u32_f32(neon_b3_h);
        uint32x4_t neon_c4_l = vcvtnq_u32_f32(neon_b4_l);
        uint32x4_t neon_c4_h = vcvtnq_u32_f32(neon_b4_h);

        uint16x4_t neon_d1_l = vqmovn_u32(neon_c1_l);
        uint16x4_t neon_d1_h = vqmovn_u32(neon_c1_h);
        uint16x4_t neon_d2_l = vqmovn_u32(neon_c2_l);
        uint16x4_t neon_d2_h = vqmovn_u32(neon_c2_h);
        uint16x4_t neon_d3_l = vqmovn_u32(neon_c3_l);
        uint16x4_t neon_d3_h = vqmovn_u32(neon_c3_h);
        uint16x4_t neon_d4_l = vqmovn_u32(neon_c4_l);
        uint16x4_t neon_d4_h = vqmovn_u32(neon_c4_h);

        uint16x8_t neon_e1 = vcombine_u16(neon_d1_l, neon_d1_h);
        uint16x8_t neon_e2 = vcombine_u16(neon_d2_l, neon_d2_h);
        uint16x8_t neon_e3 = vcombine_u16(neon_d3_l, neon_d3_h);
        uint16x8_t neon_e4 = vcombine_u16(neon_d4_l, neon_d4_h);

        uint8x8_t neon_f1 = vqmovn_u16(neon_e1);
        uint8x8_t neon_f2 = vqmovn_u16(neon_e2);
        uint8x8_t neon_f3 = vqmovn_u16(neon_e3);
        uint8x8_t neon_f4 = vqmovn_u16(neon_e4);

        vst1_u8(out + l, neon_f1);
        vst1_u8(out + l + 8, neon_f2);
        vst1_u8(out + l + 16, neon_f3);
        vst1_u8(out + l + 24, neon_f4);
    }

    if((d - l) >= double_loop) {
        float16x8_t neon_a1 = vld1q_f16(src + l);
        float16x8_t neon_a2 = vld1q_f16(src + l + 8);

        float32x4_t neon_b1_l = vcvt_f32_f16(vget_low_f16(neon_a1));
        float32x4_t neon_b1_h = vcvt_f32_f16(vget_high_f16(neon_a1));
        float32x4_t neon_b2_l = vcvt_f32_f16(vget_low_f16(neon_a2));
        float32x4_t neon_b2_h = vcvt_f32_f16(vget_high_f16(neon_a2));

        uint32x4_t neon_c1_l = vcvtnq_u32_f32(neon_b1_l);
        uint32x4_t neon_c1_h = vcvtnq_u32_f32(neon_b1_h);
        uint32x4_t neon_c2_l = vcvtnq_u32_f32(neon_b2_l);
        uint32x4_t neon_c2_h = vcvtnq_u32_f32(neon_b2_h);

        uint16x4_t neon_d1_l = vqmovn_u32(neon_c1_l);
        uint16x4_t neon_d1_h = vqmovn_u32(neon_c1_h);
        uint16x4_t neon_d2_l = vqmovn_u32(neon_c2_l);
        uint16x4_t neon_d2_h = vqmovn_u32(neon_c2_h);

        uint16x8_t neon_e1 = vcombine_u16(neon_d1_l, neon_d1_h);
        uint16x8_t neon_e2 = vcombine_u16(neon_d2_l, neon_d2_h);

        uint8x8_t neon_f1 = vqmovn_u16(neon_e1);
        uint8x8_t neon_f2 = vqmovn_u16(neon_e2);

        vst1_u8(out + l, neon_f1);
        vst1_u8(out + l + 8, neon_f2);
        l += 16;
    }

    if((d - l) >= single_loop) {
        float16x8_t neon_a = vld1q_f16(src + l);

        float32x4_t neon_b_lo = vcvt_f32_f16(vget_low_f16(neon_a));
        float32x4_t neon_b_hi = vcvt_f32_f16(vget_high_f16(neon_a));

        uint32x4_t neon_c_lo = vcvtnq_u32_f32(neon_b_lo);
        uint32x4_t neon_c_hi = vcvtnq_u32_f32(neon_b_hi);

        uint16x4_t neon_d_lo = vqmovn_u32(neon_c_lo);
        uint16x4_t neon_d_hi = vqmovn_u32(neon_c_hi);

        uint16x8_t neon_e = vcombine_u16(neon_d_lo, neon_d_hi);
        uint8x8_t neon_f = vqmovn_u16(neon_e);

        vst1_u8(out + l, neon_f);
        l += 8;
    }

    for (; l < d; ++l) {
        out[l] = (uint8_t)((float)(src[l]) + 0.5f);
    }
}

#endif

namespace {
template <typename T, class BlockResultHandler>
void hnsw_search(
        const IndexHNSW* index,
        idx_t n,
        const T* x,
        BlockResultHandler& bres,
        const SearchParameters* params_in) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    const SearchParametersHNSW* params = nullptr;
    const HNSW& hnsw = index->hnsw;

    int efSearch = hnsw.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSW*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#ifdef KRL
#pragma omp parallel
        {
            VisitedTable vt(index->ntotal);
            typename BlockResultHandler::SingleResultHandler res(bres);
            std::unique_ptr<DistanceComputer> dis;
            if(index->quant_bits == 16) {
                auto f16_x = std::make_unique<float16_t[]>(index->d);
                if(index->metric_type == METRIC_L2) {
                    dis = std::make_unique<FlatL2DisSQ16>(index->d);
                } else {
                    dis = std::make_unique<FlatIPDisSQ16>(index->d);
                }
                if constexpr (std::is_same_v<T, float16_t>) {
                    dis->set_base((float16_t*)index->storage->get_codes_pointer());
                } else {
                    dis->set_base((float*)index->storage->get_codes_pointer());
                }

                for (idx_t i = i0; i < i1; i++) {
                    res.begin(i);
                    if constexpr (std::is_same_v<T, float16_t>) {
                        dis->set_query(x + i * index->d);
                    } else {
                        quant_f16_noscale(x + i * index->d, index->d, f16_x.get());
                        dis->set_query((float*)f16_x.get());
                    }
                    hnsw.search(*dis, res, vt, params);
                    res.end();
                }
            } else if (index->quant_bits == 8) {
                dis = std::make_unique<FlatL2DisSQ8>(index->d);
                
                auto u8_x = std::make_unique<uint8_t[]>(index->d);
            
                if constexpr (std::is_same_v<T, float16_t>) {
                    dis->set_base((float16_t*)index->storage->get_codes_pointer());
                    for (idx_t i = i0; i < i1; i++) {
                        quant_u8_noscale_f16(x + i * index->d, index->d, u8_x.get());
                        res.begin(i);
                        dis->set_query((float16_t*)u8_x.get());
                        hnsw.search(*dis, res, vt, params);
                        res.end();
                    }
                } else {
                    dis->set_base((float*)index->storage->get_codes_pointer());
                    for (idx_t i = i0; i < i1; i++) {
                        quant_u8_noscale(x + i * index->d, index->d, u8_x.get());
                        res.begin(i);
                        dis->set_query((float*)u8_x.get());
                        hnsw.search(*dis, res, vt, params);
                        res.end();
                    }
                }
            } else {
                dis.reset(storage_distance_computer(index->storage));
            
                for (idx_t i = i0; i < i1; i++) {
                    res.begin(i);
                    dis->set_query(x + i * index->d);
                    hnsw.search(*dis, res, vt, params);
                    res.end();
                }
            }
        }
        InterruptCallback::check();
        
    }
#else
#pragma omp parallel
        {
            VisitedTable vt(index->ntotal);
            typename BlockResultHandler::SingleResultHandler res(bres);

            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index->storage));

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                dis->set_query(x + i * index->d);

                HNSWStats stats = hnsw.search(*dis, res, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                nreorder += stats.nreorder;
                res.end();
            }
        }
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, n3, ndis, nreorder});
#endif
}

} // anonymous namespace

void IndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    hnsw_search<float>(this, n, x, bres, params_in);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
#ifdef KRL
    if (apply_reorder && perm != nullptr) {
        std::vector<idx_t> labels_permuted(n * k);
        for (size_t i = 0; i < n * k; ++i) {
            labels_permuted[i] = perm[labels[i]];
        }
        std::copy_n(labels_permuted.cbegin(), n * k, labels);
    }
#endif
}

void IndexHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HNSW::C>;
    RH bres(result, radius);

    hnsw_search<float>(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices<float>(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);

#ifdef KRL
    if(quant_bits == 16) {
        storage->quant_entries_f16(storage->get_codes_pointer(), n * d, quant_scale);
    } else if(quant_bits == 8) {
        storage->quant_entries_u8(storage->get_codes_pointer(), n * d, quant_scale);
    }
#endif
}

void IndexHNSW::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHNSW::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexHNSW::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0)
                    break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HNSW::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hnsw.neighbors[j] = -1;
            }
        }
    }
}

void IndexHNSW::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    storage_idx_t ntotal = hnsw.levels.size();

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> qdis(
                storage_distance_computer(storage));
        HNSWStats search_stats;
        VisitedTable vt(ntotal);
        RH::SingleResultHandler res(bres);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            qdis->set_query(x + i * d);

            hnsw.search_level_0(
                    *qdis.get(),
                    res,
                    nprobe,
                    nearest + i * nprobe,
                    nearest_d + i * nprobe,
                    search_type,
                    search_stats,
                    vt);
            res.end();
            vt.advance();
        }
#pragma omp critical
        { hnsw_stats.combine(search_stats); }
    }
}

void IndexHNSW::init_level_0_from_knngraph(
        int k,
        const float* D,
        const idx_t* I) {
    int dest_size = hnsw.nb_neighbors(0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer* qdis = storage_distance_computer(storage);
        std::vector<float> vec(d);
        storage->reconstruct(i, vec.data());
        qdis->set_query(vec.data());

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i)
                continue;
            if (v1 < 0)
                break;
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        HNSW::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            else
                hnsw.neighbors[j] = -1;
        }
    }
}

void IndexHNSW::init_level_0_from_entry_points(
        int n,
        const storage_idx_t* points,
        const storage_idx_t* nearests) {
    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

#pragma omp parallel
    {
        VisitedTable vt(ntotal);

        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));
        std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct(pt_id, vec.data());
            dis->set_query(vec.data());

            hnsw.add_links_starting_from(
                    *dis, pt_id, nearest, (*dis)(nearest), 0, locks.data(), vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for (int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);
}

void IndexHNSW::reorder_links() {
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances(M);
        std::vector<size_t> order(M);
        std::vector<storage_idx_t> tmp(M);
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (storage_idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp[j - begin] = nj;
            }

            fvec_argsort(end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }
    }
}

void IndexHNSW::link_singletons() {
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0)
                seen[ni] = true;
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1)
                n_sing_l1++;
        }
    }

    printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
           n_sing,
           ntotal,
           n_sing_l1);

    std::vector<float> recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {
        FAISS_ASSERT(!"not implemented");
    }
}

void IndexHNSW::permute_entries(const idx_t* perm) {
    auto flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_storage, "don't know how to permute this index");
    flat_storage->permute_entries(perm);
    hnsw.permute_entries(perm);
}

/* added FP16 function interfaces */
#ifdef __aarch64__
void IndexHNSW::add(idx_t n, const float16_t* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices<float16_t>(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
#ifdef KRL
    if(quant_bits == 16) {
        storage->quant_entries_f16(storage->get_codes_pointer(), n * d, quant_scale);
    } else if(quant_bits == 8) {
        storage->quant_entries_u8(storage->get_codes_pointer(), n * d, quant_scale);
    }
#endif
}

void IndexHNSW::train(idx_t n, const float16_t* x) {
    FAISS_THROW_IF_NOT_MSG(
        storage,
        "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexHNSW::search(
        idx_t n,
        const float16_t* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    hnsw_search<float16_t>(this, n, x, bres, params_in);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
#ifdef KRL
    // *** add for reordering ***
    if (apply_reorder && perm != nullptr) {
        std::vector<idx_t> labels_permuted(n * k);
        for (size_t i = 0; i < n * k; ++i) {
            labels_permuted[i] = perm[labels[i]];
        }
        std::copy_n(labels_permuted.cbegin(), n * k, labels);
    }
#endif
}

void IndexHNSW::range_search(
        idx_t n,
        const float16_t* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HNSW::C>;
    RH bres(result, radius);

    hnsw_search<float16_t>(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexHNSW::reconstruct(idx_t key, float16_t* recons) const {
    storage->reconstruct(key, recons);
}

void IndexHNSW::search_level_0(
        idx_t n,
        const float16_t* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    storage_idx_t ntotal = hnsw.levels.size();

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> qdis(
                storage_distance_computer(storage));
        HNSWStats search_stats;
        VisitedTable vt(ntotal);
        RH::SingleResultHandler res(bres);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            qdis->set_query(x + i * d);

            hnsw.search_level_0(
                    *qdis.get(),
                    res,
                    nprobe,
                    nearest + i * nprobe,
                    nearest_d + i * nprobe,
                    search_type,
                    search_stats,
                    vt);
            res.end();
            vt.advance();
        }
#pragma omp critical
        { hnsw_stats.combine(search_stats); }
    }
}
#endif

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() {
    is_trained = true;
}

IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2) ? new IndexFlatL2(d)
                                        : new IndexFlat(d, metric),
                  M) {
    own_fields = true;
    is_trained = true;
}

#ifdef __aarch64__
IndexHNSWFlat::IndexHNSWFlat(int d, int M, NumericType ntype, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2) ? new IndexFlatL2(d, ntype)
                                        : new IndexFlat(d, ntype, metric),
                  ntype,
                  M) {
    own_fields = true;
    is_trained = true;
}
#endif

/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/

IndexHNSWPQ::IndexHNSWPQ() = default;

IndexHNSWPQ::IndexHNSWPQ(int d, int pq_m, int M, int pq_nbits)
        : IndexHNSW(new IndexPQ(d, pq_m, pq_nbits), M) {
    own_fields = true;
    is_trained = false;
}

void IndexHNSWPQ::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/* explicit FP32 function interface */
#ifdef __aarch64__
void IndexHNSWPQ::add(idx_t n, const float* x) {
    IndexHNSW::add(n, x);
}

void IndexHNSWPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IndexHNSW::search(n, x, k, distances, labels, params);
}

void IndexHNSWPQ::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IndexHNSW::range_search(n, x, radius, result, params);
}

void IndexHNSWPQ::reconstruct(idx_t key, float* recons) const {
    IndexHNSW::reconstruct(key, recons);
}

void IndexHNSWPQ::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    IndexHNSW::search_level_0(n, x, k, nearest, nearest_d, distances, labels, nprobe, search_type);
}

/* added FP16 function interfaces */
void IndexHNSWPQ::add(idx_t n, const float16_t* x) {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::add(n, x_float.data());
}

void IndexHNSWPQ::train(idx_t n, const float16_t* x) {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::train(n, x_float.data());
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

void IndexHNSWPQ::search(
        idx_t n,
        const float16_t* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::search(n, x_float.data(), k, distances, labels, params);
}

void IndexHNSWPQ::range_search(
        idx_t n,
        const float16_t* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::range_search(n, x_float.data(), radius, result, params);
}

void IndexHNSWPQ::reconstruct(idx_t key, float16_t* recons) const {
    std::vector<float> recons_float(key * d);
    convert_fp16_to_fp32(recons, key * d, recons_float.data());
    IndexHNSW::reconstruct(key, recons_float.data());
}

void IndexHNSWPQ::search_level_0(
        idx_t n,
        const float16_t* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::search_level_0(n, x_float.data(), k, nearest, nearest_d, distances, labels, nprobe, search_type);
}
#endif

/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/

IndexHNSWSQ::IndexHNSWSQ(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexHNSW(new IndexScalarQuantizer(d, qtype, metric), M) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexHNSWSQ::IndexHNSWSQ() = default;

/* explicit FP32 function interface */
#ifdef __aarch64__
void IndexHNSWSQ::add(idx_t n, const float* x) {
    IndexHNSW::add(n, x);
}

void IndexHNSWSQ::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
}

void IndexHNSWSQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IndexHNSW::search(n, x, k, distances, labels, params);
}

void IndexHNSWSQ::range_search(
        idx_t n,
         const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IndexHNSW::range_search(n, x, radius, result, params);
}

void IndexHNSWSQ::reconstruct(idx_t key, float* recons) const {
    IndexHNSW::reconstruct(key, recons);
}

void IndexHNSWSQ::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    IndexHNSW::search_level_0(n, x, k, nearest, nearest_d, distances, labels, nprobe, search_type);
}

/* added FP16 function interfaces */
void IndexHNSWSQ::add(idx_t n, const float16_t* x) {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::add(n, x_float.data());
}

void IndexHNSWSQ::train(idx_t n, const float16_t* x) {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::train(n, x_float.data());
}

void IndexHNSWSQ::search(
        idx_t n,
        const float16_t* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::search(n, x_float.data(), k, distances, labels, params);
}

void IndexHNSWSQ::range_search(
        idx_t n,
         const float16_t* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::range_search(n, x_float.data(), radius, result, params);
}

void IndexHNSWSQ::reconstruct(idx_t key, float16_t* recons) const {
    std::vector<float> recons_float(key * d);
    convert_fp16_to_fp32(recons, key * d, recons_float.data());
    IndexHNSW::reconstruct(key, recons_float.data());
}

void IndexHNSWSQ::search_level_0(
        idx_t n,
        const float16_t* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::search_level_0(n, x_float.data(), k, nearest, nearest_d, distances, labels, nprobe, search_type);
}
#endif

/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/

IndexHNSW2Level::IndexHNSW2Level(
        Index* quantizer,
        size_t nlist,
        int m_pq,
        int M)
        : IndexHNSW(new Index2Layer(quantizer, nlist, m_pq), M) {
    own_fields = true;
    is_trained = false;
}

IndexHNSW2Level::IndexHNSW2Level() = default;

namespace {

// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0) {
    int nres = nres_in;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        FAISS_ASSERT(v1 >= 0);
        vt.visited[v1] = vt.visno + 1;
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.visited[v1] == vt.visno + 1) {
                // nothing to do
            } else {
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.visited[v1] < vt.visno) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
                vt.visited[v1] = vt.visno + 1;
            }
        }

        nstep++;
        if (nstep > hnsw.efSearch) {
            break;
        }
    }

    stats.n1++;
    if (candidates.size() == 0)
        stats.n2++;

    return nres;
}

} // namespace

void IndexHNSW2Level::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search(n, x, k, distances, labels);

    } else { // "mixed" search
        size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;

        const IndexIVFPQ* index_ivfpq =
                dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        std::unique_ptr<idx_t[]> coarse_assign(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        index_ivfpq->quantizer->search(
                n, x, nprobe, coarse_dis.get(), coarse_assign.get());

        index_ivfpq->search_preassigned(
                n,
                x,
                k,
                coarse_assign.get(),
                coarse_dis.get(),
                distances,
                labels,
                false);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));

            int candidates_size = hnsw.upper_beam;
            MinimaxHeap candidates(candidates_size);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
            for (idx_t i = 0; i < n; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0)
                        break;
                    size_t list_length = index_ivfpq->get_list_size(key);
                    const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set(ids[jj]);
                    }
                }

                candidates.clear();

                for (int j = 0; j < hnsw.upper_beam && j < k; j++) {
                    if (idxi[j] < 0)
                        break;
                    candidates.push(idxi[j], simi[j]);
                }

                // reorder from sorted to heap
                maxheap_heapify(k, simi, idxi, simi, idxi, k);

                HNSWStats search_stats;
                search_from_candidates_2(
                        hnsw,
                        *dis,
                        k,
                        idxi,
                        simi,
                        candidates,
                        vt,
                        search_stats,
                        0,
                        k);
                n1 += search_stats.n1;
                n2 += search_stats.n2;
                n3 += search_stats.n3;
                ndis += search_stats.ndis;
                nreorder += search_stats.nreorder;

                vt.advance();
                vt.advance();

                maxheap_reorder(k, simi, idxi);
            }
        }

        hnsw_stats.combine({n1, n2, n3, ndis, nreorder});
    }
}

void IndexHNSW2Level::flip_to_ivf() {
    Index2Layer* storage2l = dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT(storage2l);

    IndexIVFPQ* index_ivfpq = new IndexIVFPQ(
            storage2l->q1.quantizer,
            d,
            storage2l->q1.nlist,
            storage2l->pq.M,
            8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map(true);

    storage = index_ivfpq;
    delete storage2l;
}

/* explicit FP32 function interface */
#ifdef __aarch64__
void IndexHNSW2Level::add(idx_t n, const float* x) {
    IndexHNSW::add(n, x);
}

void IndexHNSW2Level::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
}

void IndexHNSW2Level::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IndexHNSW::range_search(n, x, radius, result, params);
}

void IndexHNSW2Level::reconstruct(idx_t key, float* recons) const {
    IndexHNSW::reconstruct(key, recons);
}

void IndexHNSW2Level::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    IndexHNSW::search_level_0(n, x, k, nearest, nearest_d, distances, labels, nprobe, search_type);
}

/* added FP16 function interfaces */
void IndexHNSW2Level::add(idx_t n, const float16_t* x) {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::add(n, x_float.data());
}

void IndexHNSW2Level::train(idx_t n, const float16_t* x) {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::train(n, x_float.data());
}

void IndexHNSW2Level::search(
        idx_t n,
        const float16_t* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search(n, x_float.data(), k, distances, labels);

    } else { // "mixed" search
        size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;

        const IndexIVFPQ* index_ivfpq =
                dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        auto coarse_assign = std::make_unique<idx_t[]>(n * nprobe);
        auto coarse_dis = std::make_unique<float[]>(n * nprobe);

        index_ivfpq->quantizer->search(
                n, x_float.data(), nprobe, coarse_dis.get(), coarse_assign.get());

        index_ivfpq->search_preassigned(
                n,
                x_float.data(),
                k,
                coarse_assign.get(),
                coarse_dis.get(),
                distances,
                labels,
                false);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));

            int candidates_size = hnsw.upper_beam;
            MinimaxHeap candidates(candidates_size);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
            for (idx_t i = 0; i < n; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x_float.data() + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0)
                        break;
                    size_t list_length = index_ivfpq->get_list_size(key);
                    const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set(ids[jj]);
                    }
                }

                candidates.clear();

                for (int j = 0; j < hnsw.upper_beam && j < k; j++) {
                    if (idxi[j] < 0)
                        break;
                    candidates.push(idxi[j], simi[j]);
                }

                // reorder from sorted to heap
                maxheap_heapify(k, simi, idxi, simi, idxi, k);

                HNSWStats search_stats;
                search_from_candidates_2(
                        hnsw,
                        *dis,
                        k,
                        idxi,
                        simi,
                        candidates,
                        vt,
                        search_stats,
                        0,
                        k);
                n1 += search_stats.n1;
                n2 += search_stats.n2;
                n3 += search_stats.n3;
                ndis += search_stats.ndis;
                nreorder += search_stats.nreorder;

                vt.advance();
                vt.advance();

                maxheap_reorder(k, simi, idxi);
            }
        }

        hnsw_stats.combine({n1, n2, n3, ndis, nreorder});
    }
}

void IndexHNSW2Level::range_search(
        idx_t n,
        const float16_t* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::range_search(n, x_float.data(), radius, result, params);
}

void IndexHNSW2Level::reconstruct(idx_t key, float16_t* recons) const {
    std::vector<float> recons_float(key * d);
    convert_fp16_to_fp32(recons, key * d, recons_float.data());
    IndexHNSW::reconstruct(key, recons_float.data());
}

void IndexHNSW2Level::search_level_0(
        idx_t n,
        const float16_t* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    std::vector<float> x_float(n * d);
    convert_fp16_to_fp32(x, n * d, x_float.data());
    IndexHNSW::search_level_0(n, x_float.data(), k, nearest, nearest_d, distances, labels, nprobe, search_type);
}
#endif
} // namespace faiss

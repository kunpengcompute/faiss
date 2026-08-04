/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQFastScan.h>
#include <faiss/IndexRefine.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#ifdef KRL
#include <omp.h>
#endif

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace faiss;

namespace {

// ---------------------------------------------------------------------------
// Helper: fill a vector with random floats in [0,1]
// ---------------------------------------------------------------------------
void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = dist(rng);
    }
}

// Helper: build a small IVFRaBitQFastScan, train, add vectors, run search,
// and verify that all results are valid.
void run_ivf_rabitq_search(
        int d,
        int nb,
        int nq,
        int k,
        int nlist,
        MetricType metric,
        int nb_bits,
        float nprobe_frac = 0.5f) {
    int nprobe = std::max(1, (int)(nlist * nprobe_frac));
    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, (uint32_t)(d * 100 + nb + (int)metric));

    // Build index
    std::unique_ptr<Index> quantizer(
            new IndexFlat(d, metric));
    IndexIVFRaBitQFastScan index(
            quantizer.get(), d, nlist, metric, 32, true, (uint8_t)nb_bits);
    index.nprobe = nprobe;

    index.train(nb, xb.data());
    index.add(nb, xb.data());

    std::vector<float> xq((size_t)nq * (size_t)d);
    fill_random(xq, (uint32_t)(d * 200 + nb + nq));

    std::vector<float> D((size_t)nq * (size_t)k);
    std::vector<idx_t> I((size_t)nq * (size_t)k);

    index.search(nq, xq.data(), k, D.data(), I.data());

    for (size_t i = 0; i < (size_t)nq * (size_t)k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) {
            EXPECT_LT(I[i], nb);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: run IndexRefineFlat wrapping IVFRaBitQFastScan
// ---------------------------------------------------------------------------
void run_ivf_rabitq_refine(
        int d,
        int nb,
        int nq,
        int k,
        int nlist,
        MetricType metric,
        int nb_bits,
        float k_factor,
        float nprobe_frac = 0.5f) {
    int nprobe = std::max(1, (int)(nlist * nprobe_frac));
    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, (uint32_t)(d * 300 + nb + (int)metric + (int)(k_factor * 10)));

    // Build base index
    std::unique_ptr<Index> quantizer(
            new IndexFlat(d, metric));
    auto base = std::make_unique<IndexIVFRaBitQFastScan>(
            quantizer.get(), d, nlist, metric, 32, true, (uint8_t)nb_bits);
    base->nprobe = nprobe;

    base->train(nb, xb.data());
    base->add(nb, xb.data());

    // Wrap in IndexRefineFlat
    IndexRefineFlat index(base.get(), xb.data());
    index.k_factor = k_factor;

    std::vector<float> xq((size_t)nq * (size_t)d);
    fill_random(xq, (uint32_t)(d * 400 + nb + nq));

    std::vector<float> D((size_t)nq * (size_t)k);
    std::vector<idx_t> I((size_t)nq * (size_t)k);

    index.search(nq, xq.data(), k, D.data(), I.data());

    for (size_t i = 0; i < (size_t)nq * (size_t)k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) {
            EXPECT_LT(I[i], nb);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: exhaustive search via IndexFlat (covers KRL distance functions)
// ---------------------------------------------------------------------------
void run_exhaustive_flat(
        int d,
        int nb,
        int nq,
        int k,
        MetricType metric) {
    IndexFlat index(d, metric);

    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, (uint32_t)(d * 500 + nb + (int)metric));
    index.add(nb, xb.data());

    std::vector<float> xq((size_t)nq * (size_t)d);
    fill_random(xq, (uint32_t)(d * 600 + nb + nq));

    std::vector<float> D((size_t)nq * (size_t)k);
    std::vector<idx_t> I((size_t)nq * (size_t)k);

    index.search(nq, xq.data(), k, D.data(), I.data());

    for (size_t i = 0; i < (size_t)nq * (size_t)k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) {
            EXPECT_LT(I[i], nb);
        }
    }
}

} // anonymous namespace

// ===========================================================================
// Section 1: IndexIVFRaBitQFastScan core search paths
//   Covers:
//     - write_subset_sum_lut (KRL NEON)
//     - compute_residual_LUT (KRL NEON uint8→float)
//     - compute_LUT_uint8 (KRL NEON min/max + NEON quantization)
//     - IVFRaBitQHeapHandler constructor (KRL krl_2heaps_heapify)
//     - IVFRaBitQHeapHandler::handle (KRL NEON 4-way + krl_2heaps_replace_top)
//     - IVFRaBitQHeapHandler::end (KRL krl_2heaps_reorder)
//     - search_implem_10 / 12 (KRL probe_map, KRL dispatch)
//     - dispatching.h accumulate_loop (KRL krl_fast_table_lookup_step)
// ===========================================================================

// --- 1a: k <= 20, nq = 1, L2, 1-bit ---
// Triggers: impl 10 (per-query, heap), non-centered path
TEST(KRLTest, IVFRaBitQ_1bit_L2_k10_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

// --- 1b: k <= 20, nq = 1, IP, 1-bit ---
TEST(KRLTest, IVFRaBitQ_1bit_IP_k10_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1);
}

// --- 1c: k > 20, nq = 1, L2, 1-bit (reservoir path) ---
// Triggers: impl 11 (per-query, reservoir)
TEST(KRLTest, IVFRaBitQ_1bit_L2_k25_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/25,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

// --- 1d: k > 20, nq = 1, IP, 1-bit (reservoir path) ---
TEST(KRLTest, IVFRaBitQ_1bit_IP_k25_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/25,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1);
}

// --- 1e: k <= 20, nq > 1, L2, 1-bit (QBS batch, impl 12) ---
// Triggers: impl 12 (KRL reserve + KRL regroup/pack buffers + KRL krl_fast_table_lookup_step)
TEST(KRLTest, IVFRaBitQ_1bit_L2_k10_nq5) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/5, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

// --- 1f: k <= 20, nq > 1, IP, 1-bit (QBS batch, impl 12) ---
TEST(KRLTest, IVFRaBitQ_1bit_IP_k10_nq5) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/5, /*k=*/10,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1);
}

// --- 1g: k > 20, nq > 1, L2, 1-bit (QBS batch, impl 13, reservoir) ---
TEST(KRLTest, IVFRaBitQ_1bit_L2_k25_nq5) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/5, /*k=*/25,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

// --- 1h: k > 20, nq > 1, IP, 1-bit (QBS batch, impl 13, reservoir) ---
TEST(KRLTest, IVFRaBitQ_1bit_IP_k25_nq5) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/5, /*k=*/25,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1);
}

// ===========================================================================
// Section 2: Multi-bit RaBitQ (nb_bits=2, nb_bits=4)
//   Covers:
//     - KRL multibit replace_top in IVFRaBitQHeapHandler::handle()
//     - compute_full_multibit_distance path
// ===========================================================================

// --- 2a: 2-bit, k <= 20, nq = 1, L2 ---
TEST(KRLTest, IVFRaBitQ_2bit_L2_k10_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/2);
}

// --- 2b: 2-bit, k <= 20, nq = 1, IP ---
TEST(KRLTest, IVFRaBitQ_2bit_IP_k10_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/2);
}

// --- 2c: 2-bit, k > 20, nq = 1, L2 (reservoir) ---
TEST(KRLTest, IVFRaBitQ_2bit_L2_k25_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/25,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/2);
}

// --- 2d: 2-bit, k <= 20, nq > 1, L2 (QBS batch, impl 12) ---
TEST(KRLTest, IVFRaBitQ_2bit_L2_k10_nq5) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/5, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/2);
}

// --- 2e: 4-bit, k <= 20, nq = 1, L2 ---
TEST(KRLTest, IVFRaBitQ_4bit_L2_k10_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/4);
}

// --- 2f: 4-bit, k <= 20, nq > 1, L2 (QBS batch, impl 12) ---
TEST(KRLTest, IVFRaBitQ_4bit_L2_k10_nq5) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/5, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/4);
}

// ===========================================================================
// Section 3: k=1 (special single-result path)
//   Covers:
//     - impl 10/12 with Top1 handler
//     - KRL path in dispatching.h (SingleResultHandler)
// ===========================================================================

TEST(KRLTest, IVFRaBitQ_1bit_L2_k1_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/1,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_1bit_L2_k1_nq5) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/5, /*k=*/1,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_1bit_IP_k1_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/1,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1);
}

// ===========================================================================
// Section 4: Edge-case dimensions (d not multiple of 4)
//   Covers:
//     - compute_residual_LUT scalar tail (ds+3 >= d_sz)
//     - write_subset_sum_lut partial dimension
//     - IVFRaBitQHeapHandler::handle scalar tail (j+3 >= max_positions)
// ===========================================================================

TEST(KRLTest, IVFRaBitQ_DimNonAligned7) {
    run_ivf_rabitq_search(/*d=*/7, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_DimNonAligned9) {
    run_ivf_rabitq_search(/*d=*/9, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_DimNonAligned15) {
    run_ivf_rabitq_search(/*d=*/15, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_DimNonAligned17) {
    run_ivf_rabitq_search(/*d=*/17, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_DimNonAligned31) {
    run_ivf_rabitq_search(/*d=*/31, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_DimNonAligned33) {
    run_ivf_rabitq_search(/*d=*/33, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

// ===========================================================================
// Section 5: IndexRefineFlat wrapping IVFRaBitQFastScan
//   Covers:
//     - IndexRefineFlat::add (KRL krl_create_reorder_handle)
//     - IndexRefineFlat::reset (KRL krl_clean_distance_handle)
//     - IndexRefineFlat::search (KRL krl_reorder_2_vector)
//     - IndexRefineFlat::~IndexRefineFlat (KRL cleanup)
// ===========================================================================

// --- 5a: L2, various k_factor values ---
TEST(KRLTest, IndexRefineFlat_L2_kfactor1) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1, /*k_factor=*/1.0f);
}

TEST(KRLTest, IndexRefineFlat_L2_kfactor2) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1, /*k_factor=*/2.0f);
}

TEST(KRLTest, IndexRefineFlat_L2_kfactor4) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1, /*k_factor=*/4.0f);
}

TEST(KRLTest, IndexRefineFlat_L2_kfactor1_k2) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/1, /*k=*/2,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1, /*k_factor=*/2.0f);
}

// --- 5b: IP, various k_factor ---
TEST(KRLTest, IndexRefineFlat_IP_kfactor2) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1, /*k_factor=*/2.0f);
}

// --- 5c: L2, nq > 1, k_factor=2 (covers OMP path in krl_reorder_2_vector) ---
TEST(KRLTest, IndexRefineFlat_L2_kfactor2_nq5) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/5, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1, /*k_factor=*/2.0f);
}

// --- 5d: IP, nq > 1 ---
TEST(KRLTest, IndexRefineFlat_IP_kfactor2_nq5) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/5, /*k=*/10,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1, /*k_factor=*/2.0f);
}

// --- 5e: Non-aligned dimensions ---
TEST(KRLTest, IndexRefineFlat_L2_kfactor2_d15) {
    run_ivf_rabitq_refine(/*d=*/15, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1, /*k_factor=*/2.0f);
}

TEST(KRLTest, IndexRefineFlat_L2_kfactor2_d33) {
    run_ivf_rabitq_refine(/*d=*/33, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1, /*k_factor=*/2.0f);
}

// ===========================================================================
// Section 6: IndexFlat exhaustive search (KRL distance functions)
//   Covers:
//     - distances.cpp krl_ipdis() (exhaustive_inner_product_seq)
//     - distances.cpp krl_L2sqr() (exhaustive_L2sqr_seq)
// ===========================================================================

TEST(KRLTest, Exhaustive_Flat_L2_k10) {
    run_exhaustive_flat(/*d=*/32, /*nb=*/200, /*nq=*/5, /*k=*/10, METRIC_L2);
}

TEST(KRLTest, Exhaustive_Flat_IP_k10) {
    run_exhaustive_flat(/*d=*/32, /*nb=*/200, /*nq=*/5, /*k=*/10, METRIC_INNER_PRODUCT);
}

TEST(KRLTest, Exhaustive_Flat_L2_k1) {
    run_exhaustive_flat(/*d=*/32, /*nb=*/200, /*nq=*/5, /*k=*/1, METRIC_L2);
}

TEST(KRLTest, Exhaustive_Flat_IP_k1) {
    run_exhaustive_flat(/*d=*/32, /*nb=*/200, /*nq=*/5, /*k=*/1, METRIC_INNER_PRODUCT);
}

// Non-aligned dims for exhaustive
TEST(KRLTest, Exhaustive_Flat_L2_d9) {
    run_exhaustive_flat(/*d=*/9, /*nb=*/200, /*nq=*/5, /*k=*/10, METRIC_L2);
}

TEST(KRLTest, Exhaustive_Flat_L2_d15) {
    run_exhaustive_flat(/*d=*/15, /*nb=*/200, /*nq=*/5, /*k=*/10, METRIC_L2);
}

// ===========================================================================
// Section 7: IndexRefineFlat.add() called multiple times (covers
//            krl_create_reorder_handle(kdh != nullptr) re-creation path)
// ===========================================================================

TEST(KRLTest, IndexRefineFlat_MultiAdd) {
    int d = 32;
    int nb = 300;
    int nq = 1;
    int k = 10;
    int nlist = 16;
    MetricType metric = METRIC_L2;

    std::vector<float> xb1((size_t)nb * (size_t)d);
    std::vector<float> xb2((size_t)nb * (size_t)d);
    fill_random(xb1, 1001);
    fill_random(xb2, 1002);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, metric));
    auto base = std::make_unique<IndexIVFRaBitQFastScan>(
            quantizer.get(), d, nlist, metric, 32, true, 1);
    base->nprobe = 8;
    base->train(nb, xb1.data());

    IndexRefineFlat index(base.get(), xb1.data());
    index.k_factor = 2.0f;

    // First add
    index.add(nb, xb1.data());

    // Second add (triggers krl_create_reorder_handle with kdh != nullptr)
    index.add(nb, xb2.data());

    std::vector<float> xq((size_t)nq * (size_t)d);
    fill_random(xq, 2001);

    std::vector<float> D((size_t)nq * (size_t)k);
    std::vector<idx_t> I((size_t)nq * (size_t)k);

    index.search(nq, xq.data(), k, D.data(), I.data());

    for (size_t i = 0; i < (size_t)nq * (size_t)k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) {
            EXPECT_LT(I[i], 2 * nb); // 2 adds
        }
    }
}

// ===========================================================================
// Section 8: IndexRefineFlat.reset() (covers KRL krl_clean_distance_handle)
// ===========================================================================

TEST(KRLTest, IndexRefineFlat_Reset) {
    int d = 32;
    int nb = 300;
    int nlist = 16;
    MetricType metric = METRIC_L2;

    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, 3001);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, metric));
    auto base = std::make_unique<IndexIVFRaBitQFastScan>(
            quantizer.get(), d, nlist, metric, 32, true, 1);
    base->nprobe = 8;
    base->train(nb, xb.data());

    IndexRefineFlat index(base.get(), xb.data());
    index.k_factor = 2.0f;
    index.add(nb, xb.data());

    // Reset should clean up KRL handle
    index.reset();

    // After reset, ntotal should be 0
    EXPECT_EQ(index.ntotal, 0);
}

// ===========================================================================
// Section 9: InvertedListScanner path (get_InvertedListScanner)
//   Covers:
//     - IVFRaBitQFastScanScanner constructor / set_list / scan_codes
//     - KRL paths in rabitq_result_handler via scanner
// ===========================================================================

TEST(KRLTest, IVFRaBitQ_InvertedListScanner_L2) {
    int d = 64;
    int nb = 500;
    int nq = 1;
    int k = 10;
    int nlist = 16;
    MetricType metric = METRIC_L2;

    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, 4001);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, metric));
    IndexIVFRaBitQFastScan index(
            quantizer.get(), d, nlist, metric, 32, true, 1);
    index.nprobe = 8;
    index.train(nb, xb.data());
    index.add(nb, xb.data());

    std::vector<float> xq((size_t)nq * (size_t)d);
    fill_random(xq, 4002);

    // Get scanner and exercise distance_to_code + scan_codes
    std::unique_ptr<InvertedListScanner> scanner(
            index.get_InvertedListScanner(false, nullptr, nullptr));
    ASSERT_NE(scanner, nullptr);

    scanner->set_query(xq.data());

    // Get candidate lists
    std::vector<float> coarse_dis(nlist);
    std::vector<idx_t> coarse_ids(nlist);
    index.quantizer->search(1, xq.data(), nlist, coarse_dis.data(), coarse_ids.data());

    scanner->set_list(coarse_ids[0], coarse_dis[0]);

    // Verify distance_to_code works
    std::vector<uint8_t> code(index.code_size, 0);
    float dist = scanner->distance_to_code(code.data());
    EXPECT_TRUE(std::isfinite(dist));
}

// ===========================================================================
// Section 10: Factory-created indexes (covers index_read.cpp KRL paths)
// ===========================================================================

TEST(KRLTest, Factory_IVFRaBitQFastScan_L2) {
    int d = 64;
    int nb = 500;
    int nq = 3;
    int k = 10;

    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, 5001);

    std::unique_ptr<Index> index(index_factory(d, "IVF16,RaBitQfs", METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    auto* ivf = dynamic_cast<IndexIVF*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->nprobe = 8;

    std::vector<float> xq((size_t)nq * (size_t)d);
    fill_random(xq, 5002);

    std::vector<float> D((size_t)nq * (size_t)k);
    std::vector<idx_t> I((size_t)nq * (size_t)k);

    index->search(nq, xq.data(), k, D.data(), I.data());

    for (size_t i = 0; i < (size_t)nq * (size_t)k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) {
            EXPECT_LT(I[i], nb);
        }
    }
}

TEST(KRLTest, Factory_IVFRaBitQFastScan_IP) {
    int d = 64;
    int nb = 500;
    int nq = 3;
    int k = 10;

    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, 6001);

    std::unique_ptr<Index> index(index_factory(d, "IVF16,RaBitQfs", METRIC_INNER_PRODUCT));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    auto* ivf = dynamic_cast<IndexIVF*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->nprobe = 8;

    std::vector<float> xq((size_t)nq * (size_t)d);
    fill_random(xq, 6002);

    std::vector<float> D((size_t)nq * (size_t)k);
    std::vector<idx_t> I((size_t)nq * (size_t)k);

    index->search(nq, xq.data(), k, D.data(), I.data());

    for (size_t i = 0; i < (size_t)nq * (size_t)k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
    }
}

// ===========================================================================
// Section 11: Higher dimension (d=128) to cover more code paths
// ===========================================================================

TEST(KRLTest, IVFRaBitQ_Dim128_L2_k10_nq1) {
    run_ivf_rabitq_search(/*d=*/128, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_Dim128_IP_k10_nq1) {
    run_ivf_rabitq_search(/*d=*/128, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_Dim128_L2_k10_nq5) {
    run_ivf_rabitq_search(/*d=*/128, /*nb=*/2000, /*nq=*/5, /*k=*/10,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

// ===========================================================================
// Section 12: Many queries (nq large) to cover QBS batch accumulate_loop_qbs
// ===========================================================================

TEST(KRLTest, IVFRaBitQ_1bit_L2_k10_nq20) {
    run_ivf_rabitq_search(/*d=*/32, /*nb=*/2000, /*nq=*/20, /*k=*/10,
                          /*nlist=*/8, METRIC_L2, /*nb_bits=*/1);
}

// ===========================================================================
// Section 13: IndexRefineFlat with IP, large k_factor (cover krl_reorder
//            with IP metric)
// ===========================================================================

TEST(KRLTest, IndexRefineFlat_IP_kfactor4) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1, /*k_factor=*/4.0f);
}

TEST(KRLTest, IndexRefineFlat_L2_kfactor1_k1) {
    run_ivf_rabitq_refine(/*d=*/32, /*nb=*/2000, /*nq=*/1, /*k=*/1,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1, /*k_factor=*/1.0f);
}

// ===========================================================================
// Section 14: IO save/load — triggers index_read.cpp KRL repack + KRL init
// ===========================================================================

TEST(KRLTest, IO_RaBitQFS_Roundtrip_Search) {
    int d = 64, nb = 500, nq = 3, k = 10, nlist = 16;
    std::vector<float> xb(nb * d), xq(nq * d);
    fill_random(xb, 7001); fill_random(xq, 7002);

    // build index
    std::unique_ptr<Index> quantizer(new IndexFlat(d, METRIC_L2));
    IndexIVFRaBitQFastScan idx(quantizer.get(), d, nlist, METRIC_L2, 32, true, 1);
    idx.nprobe = 8;
    idx.train(nb, xb.data());
    idx.add(nb, xb.data());

    // roundtrip via temp file
    const char* fname = "/tmp/krl_test_io_rabitq.faiss";
    write_index(&idx, fname);
    std::unique_ptr<Index> idx2(read_index(fname));
    ASSERT_NE(idx2.get(), nullptr);
    remove(fname);

    auto* ivf2 = dynamic_cast<IndexIVF*>(idx2.get());
    ASSERT_NE(ivf2, nullptr);
    ivf2->nprobe = 8;

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    idx2->search(nq, xq.data(), k, D.data(), I.data());
    for (size_t i = 0; i < (size_t)nq * k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) EXPECT_LT(I[i], nb);
    }
}

TEST(KRLTest, IO_RaBitQFS_2bit_Roundtrip) {
    int d = 64, nb = 500, nlist = 16;
    std::vector<float> xb(nb * d);
    fill_random(xb, 7003);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, METRIC_L2));
    IndexIVFRaBitQFastScan idx(quantizer.get(), d, nlist, METRIC_L2, 32, true, 2);
    idx.nprobe = 8;
    idx.train(nb, xb.data());
    idx.add(nb, xb.data());

    const char* fname = "/tmp/krl_test_io_rabitq2.faiss";
    write_index(&idx, fname);
    std::unique_ptr<Index> idx2(read_index(fname));
    ASSERT_NE(idx2.get(), nullptr);
    remove(fname);

    std::vector<float> xq(3 * d);
    fill_random(xq, 7004);
    auto* ivf2 = dynamic_cast<IndexIVF*>(idx2.get());
    ASSERT_NE(ivf2, nullptr);
    ivf2->nprobe = 8;
    std::vector<float> D(3 * 10);
    std::vector<idx_t> I(3 * 10);
    idx2->search(3, xq.data(), 10, D.data(), I.data());
    for (size_t i = 0; i < 30; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) EXPECT_LT(I[i], nb);
    }
}

// ===========================================================================
// Section 15: IndexRefineFlat IO roundtrip + accu_level variations
// ===========================================================================

TEST(KRLTest, IO_IndexRefineFlat_Roundtrip) {
    int d = 32, nb = 300, nlist = 16, nq = 1, k = 10;
    std::vector<float> xb(nb * d), xq(nq * d);
    fill_random(xb, 8001); fill_random(xq, 8002);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, METRIC_L2));
    auto base = std::make_unique<IndexIVFRaBitQFastScan>(
        quantizer.get(), d, nlist, METRIC_L2, 32, true, 1);
    base->nprobe = 8;
    base->train(nb, xb.data());

    IndexRefineFlat idx(base.get(), xb.data());
    idx.k_factor = 2.0f;
    idx.add(nb, xb.data());

    const char* fname = "/tmp/krl_test_io_refine.faiss";
    write_index(&idx, fname);
    std::unique_ptr<Index> idx2(read_index(fname));
    ASSERT_NE(idx2.get(), nullptr);
    remove(fname);

    auto* rf2 = dynamic_cast<IndexRefineFlat*>(idx2.get());
    ASSERT_NE(rf2, nullptr);
    rf2->k_factor = 2.0f;
    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    idx2->search(nq, xq.data(), k, D.data(), I.data());
    for (size_t i = 0; i < (size_t)nq * k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) EXPECT_LT(I[i], nb);
    }
}

TEST(KRLTest, IndexRefineFlat_AccuLevel2_L2) {
    int d = 32, nb = 300, nlist = 16, nq = 1, k = 10;
    std::vector<float> xb(nb * d), xq(nq * d);
    fill_random(xb, 9001); fill_random(xq, 9002);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, METRIC_L2));
    auto base = std::make_unique<IndexIVFRaBitQFastScan>(
        quantizer.get(), d, nlist, METRIC_L2, 32, true, 1);
    base->nprobe = 8;
    base->train(nb, xb.data());

    IndexRefineFlat idx(base.get(), xb.data());
    idx.k_factor = 2.0f;
#ifdef KRL
    idx.accu_level = 2;  // trigger f16f32 quantization path
    idx.full_level = 3;
#endif
    idx.add(nb, xb.data());

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    idx.search(nq, xq.data(), k, D.data(), I.data());
    for (size_t i = 0; i < (size_t)nq * k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) EXPECT_LT(I[i], nb);
    }
}

TEST(KRLTest, IndexRefineFlat_AccuLevel3_L2) {
    int d = 32, nb = 300, nlist = 16, nq = 1, k = 10;
    std::vector<float> xb(nb * d), xq(nq * d);
    fill_random(xb, 10001); fill_random(xq, 10002);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, METRIC_L2));
    auto base = std::make_unique<IndexIVFRaBitQFastScan>(
        quantizer.get(), d, nlist, METRIC_L2, 32, true, 1);
    base->nprobe = 8;
    base->train(nb, xb.data());

    IndexRefineFlat idx(base.get(), xb.data());
    idx.k_factor = 2.0f;
#ifdef KRL
    idx.accu_level = 3;  // trigger f32 transpose path
    idx.full_level = 3;
#endif
    idx.add(nb, xb.data());

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    idx.search(nq, xq.data(), k, D.data(), I.data());
    for (size_t i = 0; i < (size_t)nq * k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) EXPECT_LT(I[i], nb);
    }
}

TEST(KRLTest, IndexRefineFlat_AccuLevel2_IP) {
    int d = 32, nb = 300, nlist = 16, nq = 1, k = 10;
    std::vector<float> xb(nb * d), xq(nq * d);
    fill_random(xb, 11001); fill_random(xq, 11002);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, METRIC_INNER_PRODUCT));
    auto base = std::make_unique<IndexIVFRaBitQFastScan>(
        quantizer.get(), d, nlist, METRIC_INNER_PRODUCT, 32, true, 1);
    base->nprobe = 8;
    base->train(nb, xb.data());

    IndexRefineFlat idx(base.get(), xb.data());
    idx.k_factor = 2.0f;
#ifdef KRL
    idx.accu_level = 2;
    idx.full_level = 3;
#endif
    idx.add(nb, xb.data());

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    idx.search(nq, xq.data(), k, D.data(), I.data());
    for (size_t i = 0; i < (size_t)nq * k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) EXPECT_LT(I[i], nb);
    }
}

// ===========================================================================
// Section 16: More nb_bits/d/qb combos for IndexIVFRaBitQFastScan KRL NEON
// ===========================================================================

TEST(KRLTest, IVFRaBitQ_4bit_IP_k10_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/10,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/4);
}

TEST(KRLTest, IVFRaBitQ_4bit_L2_k25_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/25,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/4);
}

TEST(KRLTest, IVFRaBitQ_2bit_L2_k25_nq5) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/5, /*k=*/25,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/2);
}

TEST(KRLTest, IVFRaBitQ_1bit_L2_k3_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/3,
                          /*nlist=*/16, METRIC_L2, /*nb_bits=*/1);
}

TEST(KRLTest, IVFRaBitQ_1bit_IP_k3_nq1) {
    run_ivf_rabitq_search(/*d=*/64, /*nb=*/2000, /*nq=*/1, /*k=*/3,
                          /*nlist=*/16, METRIC_INNER_PRODUCT, /*nb_bits=*/1);
}

// ===========================================================================
// Section 17: OMP multi-thread KRL paths (IndexIVFFastScan.cpp #1377,#1490,#1508)
// ===========================================================================

TEST(KRLTest, IO_RaBitQFS_OMP_Search) {
    int d = 64, nb = 800, nq = 8, k = 10, nlist = 16;
    std::vector<float> xb(nb * d), xq(nq * d);
    fill_random(xb, 12001); fill_random(xq, 12002);

    std::unique_ptr<Index> quantizer(new IndexFlat(d, METRIC_L2));
    IndexIVFRaBitQFastScan idx(quantizer.get(), d, nlist, METRIC_L2, 32, true, 1);
    idx.nprobe = 4;
    idx.train(nb, xb.data());
    idx.add(nb, xb.data());

    const char* fname = "/tmp/krl_test_io_omp.faiss";
    write_index(&idx, fname);
    std::unique_ptr<Index> idx2(read_index(fname));
    ASSERT_NE(idx2.get(), nullptr);
    remove(fname);

    auto* ivf2 = dynamic_cast<IndexIVF*>(idx2.get());
    ASSERT_NE(ivf2, nullptr);
    ivf2->nprobe = 4;

#ifdef KRL
    // Force OMP multi-thread to hit KRL hoisting paths
    int prev_threads = omp_get_max_threads();
    omp_set_num_threads(2);
#endif

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    idx2->search(nq, xq.data(), k, D.data(), I.data());

#ifdef KRL
    omp_set_num_threads(prev_threads);
#endif

    for (size_t i = 0; i < (size_t)nq * k; i++) {
        EXPECT_TRUE(std::isfinite(D[i]));
        if (I[i] >= 0) EXPECT_LT(I[i], nb);
    }
}
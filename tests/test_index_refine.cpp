/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <random>
#include <vector>
#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexRefine.h>

static void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < v.size(); ++i) v[i] = dist(rng);
}

static void run_one_case_L2(int d, faiss::idx_t nb, faiss::idx_t k, float k_factor, uint32_t seed) {
#ifndef __aarch64__
    (void)d; (void)nb; (void)k; (void)k_factor; (void)seed;
    GTEST_SKIP();
#else
    faiss::IndexFlatL2 base(d);

    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, seed + 1);
    base.add(nb, xb.data());

    faiss::IndexRefineFlat index(&base, base.get_xb());
    index.k_factor = k_factor;

    std::vector<float> xq((size_t)1 * (size_t)d);
    fill_random(xq, seed + 2);

    std::vector<faiss::idx_t> I((size_t)k);
    std::vector<float> D((size_t)k);

    index.search(1, xq.data(), k, D.data(), I.data());

    for (faiss::idx_t i = 0; i < k; ++i) {
        EXPECT_TRUE(std::isfinite(D[(size_t)i]));
        EXPECT_GE(I[(size_t)i], 0);
        EXPECT_LT(I[(size_t)i], nb);
    }
#endif
}

static void run_one_case_IP(int d, faiss::idx_t nb, faiss::idx_t k, float k_factor, uint32_t seed) {
#ifndef __aarch64__
    (void)d; (void)nb; (void)k; (void)k_factor; (void)seed;
    GTEST_SKIP();
#else
    faiss::IndexFlatIP base(d);

    std::vector<float> xb((size_t)nb * (size_t)d);
    fill_random(xb, seed + 11);
    base.add(nb, xb.data());

    faiss::IndexRefineFlat index(&base, base.get_xb());
    index.k_factor = k_factor;

    std::vector<float> xq((size_t)1 * (size_t)d);
    fill_random(xq, seed + 12);

    std::vector<faiss::idx_t> I((size_t)k);
    std::vector<float> D((size_t)k);

    index.search(1, xq.data(), k, D.data(), I.data());

    for (faiss::idx_t i = 0; i < k; ++i) {
        EXPECT_TRUE(std::isfinite(D[(size_t)i]));
        EXPECT_GE(I[(size_t)i], 0);
        EXPECT_LT(I[(size_t)i], nb);
    }
#endif
}

TEST(IndexRefineFlat, cover_l2_all_ny_splits_and_d_tails) {
#ifndef __aarch64__
    GTEST_SKIP();
#else
    const faiss::idx_t nb = 1000;

    run_one_case_L2(7,  nb, 1, 2.0f, 100);   // d<8: tail
    run_one_case_L2(8,  nb, 1, 2.0f, 101);   // d==8: i+8 loop
    run_one_case_L2(9,  nb, 1, 2.0f, 102);   // d==8+1: loop + tail
    run_one_case_L2(15, nb, 1, 2.0f, 103);   // tail=7
    run_one_case_L2(16, nb, 1, 2.0f, 104);   // loop*2, no tail
    run_one_case_L2(17, nb, 1, 2.0f, 105);   // loop*2 + tail=1
    run_one_case_L2(31, nb, 1, 2.0f, 106);
    run_one_case_L2(32, nb, 1, 2.0f, 107);
    run_one_case_L2(33, nb, 1, 2.0f, 108);

    run_one_case_L2(32, nb, 1, 25.0f, 200);  // k_base=25 => 24 + 1
    run_one_case_L2(32, nb, 1, 26.0f, 201);  // k_base=26 => 24 + 2
    run_one_case_L2(32, nb, 1, 28.0f, 202);  // k_base=28 => 24 + 4
    run_one_case_L2(32, nb, 1, 32.0f, 203);  // k_base=32 => 24 + 8
    run_one_case_L2(32, nb, 1, 40.0f, 204);  // k_base=40 => 24 + 16
    run_one_case_L2(32, nb, 1, 48.0f, 205);  // k_base=48 => 24 + 24

    run_one_case_L2(32, nb, 2, 12.0f, 300);  // k=2, k_base=24 => 24
    run_one_case_L2(32, nb, 2, 9.0f,  301);  // k=2, k_base=18 => 16 + 2
    run_one_case_L2(32, nb, 2, 5.0f,  302);  // k=2, k_base=10 => 8 + 2
#endif
}

TEST(IndexRefineFlat, cover_ip_all_ny_splits_and_d_tails) {
#ifndef __aarch64__
    GTEST_SKIP();
#else
    const faiss::idx_t nb = 1000;

    run_one_case_IP(7,  nb, 1, 2.0f, 400);
    run_one_case_IP(8,  nb, 1, 2.0f, 401);
    run_one_case_IP(9,  nb, 1, 2.0f, 402);
    run_one_case_IP(15, nb, 1, 2.0f, 403);
    run_one_case_IP(16, nb, 1, 2.0f, 404);
    run_one_case_IP(17, nb, 1, 2.0f, 405);

    run_one_case_IP(32, nb, 1, 17.0f, 500);  // k_base=17 => 16 + 1
    run_one_case_IP(32, nb, 1, 18.0f, 501);  // k_base=18 => 16 + 2
    run_one_case_IP(32, nb, 1, 20.0f, 502);  // k_base=20 => 16 + 4
    run_one_case_IP(32, nb, 1, 24.0f, 503);  // k_base=24 => 16 + 8
    run_one_case_IP(32, nb, 1, 32.0f, 504);  // k_base=32 => 16 + 16

    run_one_case_IP(32, nb, 2, 8.5f,  600);  // k=2, k_base=17 => 16 + 1
    run_one_case_IP(32, nb, 2, 9.0f,  601);  // k=2, k_base=18 => 16 + 2
    run_one_case_IP(32, nb, 2, 10.0f, 602);  // k=2, k_base=20 => 16 + 4
#endif
}
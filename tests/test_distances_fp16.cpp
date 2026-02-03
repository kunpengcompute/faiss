/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <random>
#include "common_test_functions.h"

class TestDistances : public ::testing::Test {
protected:
    void SetUp() override
    {
        d = 128;
        nb = 200;
        nq = 20;
        k = 10;
        nx = 100;
        ny = 200;
        nsubset = 30;
        radius = 10.0f;

        // 生成测试数据
        xb_fp32.resize(d * nb);
        xq_fp32.resize(d * nq);

        faiss::float_rand(xb_fp32.data(), d * nb, 1234);
        faiss::float_rand(xq_fp32.data(), d * nq, 5678);

        // 转换为FP16
        xb_fp16.resize(d * nb);
        xq_fp16.resize(d * nq);

        faiss::convert_fp32_to_fp16(xb_fp32.data(), d * nb, xb_fp16.data());
        faiss::convert_fp32_to_fp16(xq_fp32.data(), d * nq, xq_fp16.data());

        x0_fp32.resize(d);
        y0_fp32.resize(d);
        y1_fp32.resize(d);
        y2_fp32.resize(d);
        y3_fp32.resize(d);

        faiss::float_rand(x0_fp32.data(), d, 1234);
        faiss::float_rand(y0_fp32.data(), d, 1240);
        faiss::float_rand(y1_fp32.data(), d, 1245);
        faiss::float_rand(y2_fp32.data(), d, 1250);
        faiss::float_rand(y3_fp32.data(), d, 1255);

        x0_fp16.resize(d);
        y0_fp16.resize(d);
        y1_fp16.resize(d);
        y2_fp16.resize(d);
        y3_fp16.resize(d);

        faiss::convert_fp32_to_fp16(x0_fp32.data(), d, x0_fp16.data());
        faiss::convert_fp32_to_fp16(y0_fp32.data(), d, y0_fp16.data());
        faiss::convert_fp32_to_fp16(y1_fp32.data(), d, y1_fp16.data());
        faiss::convert_fp32_to_fp16(y2_fp32.data(), d, y2_fp16.data());
        faiss::convert_fp32_to_fp16(y3_fp32.data(), d, y3_fp16.data());

        nr_fp16.resize(d * nx);
        nr_fp32.resize(d * nx);

        x_fp32.resize(d * nx);
        y_fp32.resize(d * ny);
        x_fp16.resize(d * nx);
        y_fp16.resize(d * ny);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < d * nx; i++) {
            x_fp32[i] = dist(gen);
        }

        for (size_t i = 0; i < d * ny; i++) {
            y_fp32[i] = dist(gen);
        }

        faiss::convert_fp32_to_fp16(x_fp32.data(), d * nx, x_fp16.data());
        faiss::convert_fp32_to_fp16(y_fp32.data(), d * ny, y_fp16.data());

        d_fp16.resize(nx * ny);
        d_fp32.resize(nx * ny);
        ids.resize(nx * ny);

        std::uniform_int_distribution<int64_t> id_dist(-1, 199);
        for (size_t i = 0; i < nx * ny; i++) {
            ids[i] = id_dist(gen);
        }

        dist_fp16.resize(nx * k);
        dist_fp32.resize(nx * k);
        labels_fp16.resize(nx * k);
        labels_fp32.resize(nx * k);

        std::fill(dist_fp16.begin(), dist_fp16.end(), 0.0f);
        std::fill(dist_fp32.begin(), dist_fp32.end(), 0.0f);
        std::fill(labels_fp16.begin(), labels_fp16.end(), -1);
        std::fill(labels_fp32.begin(), labels_fp32.end(), -1);

        src_fp16.resize(d);
        out_fp32.resize(d);
        ref_fp32.resize(d);

        std::uniform_real_distribution<float> dist_convert(-10.0f, 10.0f);
        for (int64_t i = 0; i < d; i++) {
            float val = dist_convert(gen);
            ref_fp32[i] = val;
        }
        faiss::convert_fp32_to_fp16(ref_fp32.data(), d, src_fp16.data());
    }

    void TearDown() override
    {}

    int d;
    int nb;
    int nq;
    int k;
    int nx, ny;
    int nsubset;
    int radius;

    std::vector<float16_t> xb_fp16, xq_fp16;
    std::vector<float> xb_fp32, xq_fp32;
    std::vector<float16_t> x0_fp16, y0_fp16, y1_fp16, y2_fp16, y3_fp16, x_fp16, y_fp16;
    std::vector<float> x0_fp32, y0_fp32, y1_fp32, y2_fp32, y3_fp32, nr_fp16, nr_fp32, x_fp32, y_fp32, d_fp16, d_fp32;
    std::vector<int64_t> ids, labels_fp16, labels_fp32;
    std::vector<float> dist_fp16, dist_fp32;
    std::vector<float16_t> src_fp16;
    std::vector<float> out_fp32, ref_fp32;
};

TEST_F(TestDistances, fvec_L2sqr_f16)
{
    const size_t num_tests = 100;
    size_t passed_tests = 0;

    for (size_t test_idx = 0; test_idx < num_tests; test_idx++) {
        size_t idx_b = test_idx % nb;
        size_t idx_q = test_idx % nq;

        const float16_t *x_fp16 = &xb_fp16[idx_b * d];
        const float16_t *y_fp16 = &xq_fp16[idx_q * d];
        const float *x_fp32 = &xb_fp32[idx_b * d];
        const float *y_fp32 = &xq_fp32[idx_q * d];

        float dist_fp16 = faiss::fvec_L2sqr_f16(x_fp16, y_fp16, d);

        float dist_fp32 = faiss::fvec_L2sqr(x_fp32, y_fp32, d);

        float abs_error = std::abs(dist_fp16 - dist_fp32);
        float rel_error = abs_error / (std::abs(dist_fp32) + 1e-9f);

        EXPECT_NEAR(dist_fp16, dist_fp32, 1e-1f)
            << "Test " << test_idx << ": FP16 distance = " << dist_fp16 << ", FP32 distance = " << dist_fp32
            << ", absolute error = " << abs_error;

        EXPECT_LT(rel_error, 0.01f) << "Test " << test_idx << ": Relative error too large: " << rel_error;

        if (abs_error < 1e-1f && rel_error < 0.01f) {
            passed_tests++;
        }
    }

    EXPECT_GE(passed_tests, 100) << "Only " << passed_tests << "/" << num_tests << " tests passed";
}

TEST_F(TestDistances, fvec_inner_product_f16)
{
    const size_t num_tests = 100;
    size_t passed_tests = 0;

    for (size_t test_idx = 0; test_idx < num_tests; test_idx++) {
        size_t idx_b = test_idx % nb;
        size_t idx_q = test_idx % nq;

        const float16_t *x_fp16 = &xb_fp16[idx_b * d];
        const float16_t *y_fp16 = &xq_fp16[idx_q * d];
        const float *x_fp32 = &xb_fp32[idx_b * d];
        const float *y_fp32 = &xq_fp32[idx_q * d];

        float dist_fp16 = faiss::fvec_inner_product_f16(x_fp16, y_fp16, d);

        float dist_fp32 = faiss::fvec_inner_product(x_fp32, y_fp32, d);

        float abs_error = std::abs(dist_fp16 - dist_fp32);
        float rel_error = abs_error / (std::abs(dist_fp32) + 1e-9f);

        EXPECT_NEAR(dist_fp16, dist_fp32, 1e-1f)
            << "Test " << test_idx << ": FP16 distance = " << dist_fp16 << ", FP32 distance = " << dist_fp32
            << ", absolute error = " << abs_error;

        EXPECT_LT(rel_error, 0.01f) << "Test " << test_idx << ": Relative error too large: " << rel_error;

        if (abs_error < 1e-1f && rel_error < 0.01f) {
            passed_tests++;
        }
    }

    EXPECT_GE(passed_tests, 100) << "Only " << passed_tests << "/" << num_tests << " tests passed";
}

TEST_F(TestDistances, fvec_L2sqr_batch_4_f16)
{
    float dis0_fp16, dis1_fp16, dis2_fp16, dis3_fp16;
    float dis0_fp32, dis1_fp32, dis2_fp32, dis3_fp32;

    faiss::fvec_L2sqr_batch_4_f16(x0_fp16.data(),
        y0_fp16.data(),
        y1_fp16.data(),
        y2_fp16.data(),
        y3_fp16.data(),
        d,
        dis0_fp16,
        dis1_fp16,
        dis2_fp16,
        dis3_fp16);

    faiss::fvec_L2sqr_batch_4(x0_fp32.data(),
        y0_fp32.data(),
        y1_fp32.data(),
        y2_fp32.data(),
        y3_fp32.data(),
        d,
        dis0_fp32,
        dis1_fp32,
        dis2_fp32,
        dis3_fp32);

    const float tolerance = 1e-1f;
    EXPECT_NEAR(dis0_fp16, dis0_fp32, tolerance) << "dis0 mismatch: " << dis0_fp16 << " vs " << dis0_fp32;
    EXPECT_NEAR(dis1_fp16, dis1_fp32, tolerance) << "dis1 mismatch: " << dis1_fp16 << " vs " << dis1_fp32;
    EXPECT_NEAR(dis2_fp16, dis2_fp32, tolerance) << "dis2 mismatch: " << dis2_fp16 << " vs " << dis2_fp32;
    EXPECT_NEAR(dis3_fp16, dis3_fp32, tolerance) << "dis3 mismatch: " << dis3_fp16 << " vs " << dis3_fp32;
}

TEST_F(TestDistances, fvec_inner_product_batch_4_f16)
{
    float dis0_fp16, dis1_fp16, dis2_fp16, dis3_fp16;
    float dis0_fp32, dis1_fp32, dis2_fp32, dis3_fp32;

    faiss::fvec_inner_product_batch_4_f16(x0_fp16.data(),
        y0_fp16.data(),
        y1_fp16.data(),
        y2_fp16.data(),
        y3_fp16.data(),
        d,
        dis0_fp16,
        dis1_fp16,
        dis2_fp16,
        dis3_fp16);

    faiss::fvec_inner_product_batch_4(x0_fp32.data(),
        y0_fp32.data(),
        y1_fp32.data(),
        y2_fp32.data(),
        y3_fp32.data(),
        d,
        dis0_fp32,
        dis1_fp32,
        dis2_fp32,
        dis3_fp32);

    const float tolerance = 1e-1f;
    EXPECT_NEAR(dis0_fp16, dis0_fp32, tolerance) << "dis0 mismatch: " << dis0_fp16 << " vs " << dis0_fp32;
    EXPECT_NEAR(dis1_fp16, dis1_fp32, tolerance) << "dis1 mismatch: " << dis1_fp16 << " vs " << dis1_fp32;
    EXPECT_NEAR(dis2_fp16, dis2_fp32, tolerance) << "dis2 mismatch: " << dis2_fp16 << " vs " << dis2_fp32;
    EXPECT_NEAR(dis3_fp16, dis3_fp32, tolerance) << "dis3 mismatch: " << dis3_fp16 << " vs " << dis3_fp32;
}

TEST_F(TestDistances, fvec_norm_L2sqr_f16)
{
    const size_t num_tests = 100;
    size_t passed_tests = 0;

    for (size_t test_idx = 0; test_idx < num_tests; test_idx++) {
        size_t idx_b = test_idx % nb;

        const float16_t *x_fp16 = &xb_fp16[idx_b * d];
        const float *x_fp32 = &xb_fp32[idx_b * d];

        float dist_fp16 = faiss::fvec_norm_L2sqr_f16(x_fp16, d);

        float dist_fp32 = faiss::fvec_norm_L2sqr(x_fp32, d);

        float abs_error = std::abs(dist_fp16 - dist_fp32);
        float rel_error = abs_error / (std::abs(dist_fp32) + 1e-9f);

        EXPECT_NEAR(dist_fp16, dist_fp32, 1e-1f)
            << "Test " << test_idx << ": FP16 distance = " << dist_fp16 << ", FP32 distance = " << dist_fp32
            << ", absolute error = " << abs_error;

        EXPECT_LT(rel_error, 0.01f) << "Test " << test_idx << ": Relative error too large: " << rel_error;

        if (abs_error < 1e-1f && rel_error < 0.01f) {
            passed_tests++;
        }
    }

    EXPECT_GE(passed_tests, 100) << "Only " << passed_tests << "/" << num_tests << " tests passed";
}

TEST_F(TestDistances, fvec_norms_L2sqr_f16)
{
    faiss::fvec_norms_L2sqr_f16(nr_fp16.data(), xb_fp16.data(), d, nx);
    faiss::fvec_norms_L2sqr(nr_fp32.data(), xb_fp32.data(), d, nx);

    const float tolerance = 1e-1f;
    for (size_t i = 0; i < nx; i++) {
        EXPECT_NEAR(nr_fp16[i], nr_fp32[i], tolerance) << "fvec_norms_L2sqr_f16 consistency test failed at index " << i;
    }
}

TEST_F(TestDistances, fvec_L2sqr_by_idx_f16)
{
    faiss::fvec_L2sqr_by_idx_f16(d_fp16.data(), x_fp16.data(), y_fp16.data(), ids.data(), d, nx, ny);
    faiss::fvec_L2sqr_by_idx(d_fp32.data(), x_fp32.data(), y_fp32.data(), ids.data(), d, nx, ny);

    const float tolerance = 1e-1f;
    size_t total_tests = nx * ny;
    size_t passed_tests = 0;
    size_t invalid_id_tests = 0;

    for (size_t j = 0; j < nx; j++) {
        for (size_t i = 0; i < ny; i++) {
            size_t idx = j * ny + i;
            int64_t current_id = ids[idx];

            if (current_id < 0) {
                invalid_id_tests++;
                EXPECT_TRUE(d_fp16[idx] == INFINITY) << "FP16 invalid ID test failed at (" << j << ", " << i << ")";
                EXPECT_TRUE(d_fp32[idx] == INFINITY) << "FP32 invalid ID test failed at (" << j << ", " << i << ")";

                if (d_fp16[idx] == INFINITY && d_fp32[idx] == INFINITY) {
                    passed_tests++;
                }
            } else {
                float diff = std::abs(d_fp16[idx] - d_fp32[idx]);
                float rel_error = diff / (std::abs(d_fp32[idx]) + 1e-9f);

                EXPECT_NEAR(d_fp16[idx], d_fp32[idx], tolerance)
                    << "Consistency test failed at (" << j << ", " << i << "), ID: " << current_id
                    << ", FP16: " << d_fp16[idx] << ", FP32: " << d_fp32[idx] << ", diff: " << diff;

                if (diff < tolerance) {
                    passed_tests++;
                }
            }
        }
    }
    double consistency_rate = static_cast<double>(passed_tests) / total_tests;
    std::cout << "Consistency test results: " << passed_tests << "/" << total_tests << " passed ("
              << consistency_rate * 100 << "%)" << std::endl;
    std::cout << "Invalid ID tests: " << invalid_id_tests << std::endl;

    EXPECT_GE(consistency_rate, 1) << "FP16-FP32 consistency rate too low: " << consistency_rate * 100 << "%";
}

TEST_F(TestDistances, fvec_inner_products_by_idx_f16)
{
    faiss::fvec_inner_products_by_idx_f16(d_fp16.data(), x_fp16.data(), y_fp16.data(), ids.data(), d, nx, ny);
    faiss::fvec_inner_products_by_idx(d_fp32.data(), x_fp32.data(), y_fp32.data(), ids.data(), d, nx, ny);

    const float tolerance = 1e-1f;
    size_t total_tests = nx * ny;
    size_t passed_tests = 0;
    size_t invalid_id_tests = 0;

    for (size_t j = 0; j < nx; j++) {
        for (size_t i = 0; i < ny; i++) {
            size_t idx = j * ny + i;
            int64_t current_id = ids[idx];

            if (current_id < 0) {
                invalid_id_tests++;
                EXPECT_TRUE(d_fp16[idx] == -INFINITY) << "FP16 invalid ID test failed at (" << j << ", " << i << ")";
                EXPECT_TRUE(d_fp32[idx] == -INFINITY) << "FP32 invalid ID test failed at (" << j << ", " << i << ")";

                if (d_fp16[idx] == -INFINITY && d_fp32[idx] == -INFINITY) {
                    passed_tests++;
                }
            } else {
                float diff = std::abs(d_fp16[idx] - d_fp32[idx]);
                float rel_error = diff / (std::abs(d_fp32[idx]) + 1e-9f);

                EXPECT_NEAR(d_fp16[idx], d_fp32[idx], tolerance)
                    << "Consistency test failed at (" << j << ", " << i << "), ID: " << current_id
                    << ", FP16: " << d_fp16[idx] << ", FP32: " << d_fp32[idx] << ", diff: " << diff;

                if (diff < tolerance) {
                    passed_tests++;
                }
            }
        }
    }
    double consistency_rate = static_cast<double>(passed_tests) / total_tests;
    std::cout << "Consistency test results: " << passed_tests << "/" << total_tests << " passed ("
              << consistency_rate * 100 << "%)" << std::endl;
    std::cout << "Invalid ID tests: " << invalid_id_tests << std::endl;

    EXPECT_GE(consistency_rate, 1) << "FP16-FP32 consistency rate too low: " << consistency_rate * 100 << "%";
}

TEST_F(TestDistances, knn_L2sqr_f16)
{
    faiss::knn_L2sqr_f16(x_fp16.data(), y_fp16.data(), d, nx, ny, k, dist_fp16.data(), labels_fp16.data());
    faiss::knn_L2sqr(x_fp32.data(), y_fp32.data(), d, nx, ny, k, dist_fp32.data(), labels_fp32.data());

    const float tolerance = 1e-1f;
    size_t total_comparisons = 0;
    size_t consistent_comparisons = 0;

    for (size_t query_idx = 0; query_idx < nx; query_idx++) {
        size_t base_idx = query_idx * k;

        EXPECT_TRUE(validate_ordering_consistency_l2(
            dist_fp16.data(), dist_fp32.data(), labels_fp16.data(), labels_fp32.data(), k, query_idx))
            << "Ordering consistency test failed for query " << query_idx;

        for (size_t neighbor_idx = 0; neighbor_idx < k; neighbor_idx++) {
            size_t idx = base_idx + neighbor_idx;

            if (labels_fp16[idx] >= 0 && labels_fp32[idx] >= 0) {
                total_comparisons++;

                bool id_match = (labels_fp16[idx] == labels_fp32[idx]);
                bool id_in_topk = false;

                for (size_t i = 0; i < k; i++) {
                    if (labels_fp16[idx] == labels_fp32[base_idx + i]) {
                        id_in_topk = true;
                        break;
                    }
                }

                if (id_match) {
                    float diff = std::abs(dist_fp16[idx] - dist_fp32[idx]);
                    if (diff < tolerance) {
                        consistent_comparisons++;
                    }

                    EXPECT_NEAR(dist_fp16[idx], dist_fp32[idx], tolerance)
                        << "Value consistency test failed for query " << query_idx << ", neighbor " << neighbor_idx
                        << ", ID: " << labels_fp16[idx];
                } else if (id_in_topk) {
                    consistent_comparisons++;
                }
            }
        }
    }

    double consistency_rate = static_cast<double>(consistent_comparisons) / total_comparisons;
    std::cout << "KNN consistency test results: " << consistent_comparisons << "/" << total_comparisons << " passed ("
              << consistency_rate * 100 << "%)" << std::endl;

    EXPECT_GE(consistency_rate, 0.90) << "FP16-FP32 KNN consistency rate too low: " << consistency_rate * 100 << "%";
}

TEST_F(TestDistances, knn_inner_product_f16)
{
    faiss::knn_inner_product_f16(
        x_fp16.data(), y_fp16.data(), d, nx, ny, k, dist_fp16.data(), labels_fp16.data(), nullptr);
    faiss::knn_inner_product(x_fp32.data(), y_fp32.data(), d, nx, ny, k, dist_fp32.data(), labels_fp32.data(), nullptr);

    const float tolerance = 1e-1f;
    size_t total_comparisons = 0;
    size_t consistent_comparisons = 0;

    for (size_t query_idx = 0; query_idx < nx; query_idx++) {
        size_t base_idx = query_idx * k;

        EXPECT_TRUE(validate_ordering_consistency_ip(
            dist_fp16.data(), dist_fp32.data(), labels_fp16.data(), labels_fp32.data(), k, query_idx))
            << "Ordering consistency test failed for query " << query_idx;

        for (size_t neighbor_idx = 0; neighbor_idx < k; neighbor_idx++) {
            size_t idx = base_idx + neighbor_idx;

            if (labels_fp16[idx] >= 0 && labels_fp32[idx] >= 0) {
                total_comparisons++;

                bool id_match = (labels_fp16[idx] == labels_fp32[idx]);
                bool id_in_topk = false;

                for (size_t i = 0; i < k; i++) {
                    if (labels_fp16[idx] == labels_fp32[base_idx + i]) {
                        id_in_topk = true;
                        break;
                    }
                }

                if (id_match) {
                    float diff = std::abs(dist_fp16[idx] - dist_fp32[idx]);
                    if (diff < tolerance) {
                        consistent_comparisons++;
                    }

                    EXPECT_NEAR(dist_fp16[idx], dist_fp32[idx], tolerance)
                        << "Value consistency test failed for query " << query_idx << ", neighbor " << neighbor_idx
                        << ", ID: " << labels_fp16[idx];
                } else if (id_in_topk) {
                    consistent_comparisons++;
                }
            }
        }
    }

    double consistency_rate = static_cast<double>(consistent_comparisons) / total_comparisons;
    std::cout << "KNN consistency test results: " << consistent_comparisons << "/" << total_comparisons << " passed ("
              << consistency_rate * 100 << "%)" << std::endl;

    EXPECT_GE(consistency_rate, 0.90) << "FP16-FP32 KNN consistency rate too low: " << consistency_rate * 100 << "%";
}

TEST_F(TestDistances, knn_L2sqr_by_idx_f16)
{
    faiss::knn_L2sqr_by_idx_f16(
        x_fp16.data(), y_fp16.data(), ids.data(), d, nx, ny, nsubset, k, dist_fp16.data(), labels_fp16.data(), -1);

    faiss::knn_L2sqr_by_idx(
        x_fp32.data(), y_fp32.data(), ids.data(), d, nx, ny, nsubset, k, dist_fp32.data(), labels_fp32.data(), -1);

    const float tolerance = 1e-1f;
    size_t total_comparisons = 0;
    size_t consistent_comparisons = 0;

    for (size_t query_idx = 0; query_idx < nx; query_idx++) {
        size_t base_idx = query_idx * k;

        for (size_t neighbor_idx = 0; neighbor_idx < k; neighbor_idx++) {
            size_t idx = base_idx + neighbor_idx;

            if (labels_fp16[idx] >= 0 && labels_fp32[idx] >= 0) {
                total_comparisons++;

                bool id_match = (labels_fp16[idx] == labels_fp32[idx]);
                bool id_in_topk = false;

                for (size_t i = 0; i < k; i++) {
                    if (labels_fp16[idx] == labels_fp32[base_idx + i]) {
                        id_in_topk = true;
                        break;
                    }
                }

                if (id_match) {
                    float diff = std::abs(dist_fp16[idx] - dist_fp32[idx]);
                    if (diff < tolerance) {
                        consistent_comparisons++;
                    }

                    EXPECT_NEAR(dist_fp16[idx], dist_fp32[idx], tolerance)
                        << "Value consistency test failed for query " << query_idx << ", neighbor " << neighbor_idx
                        << ", ID: " << labels_fp16[idx];
                } else if (id_in_topk) {
                    consistent_comparisons++;
                }
            }
        }
    }

    double consistency_rate = static_cast<double>(consistent_comparisons) / total_comparisons;
    std::cout << "KNN by ID consistency test results: " << consistent_comparisons << "/" << total_comparisons
              << " passed (" << consistency_rate * 100 << "%)" << std::endl;

    EXPECT_GE(consistency_rate, 0.90) << "FP16-FP32 KNN by ID consistency rate too low: " << consistency_rate * 100
                                      << "%";
}

TEST_F(TestDistances, knn_inner_products_by_idx_f16)
{
    faiss::knn_inner_products_by_idx_f16(
        x_fp16.data(), y_fp16.data(), ids.data(), d, nx, ny, nsubset, k, dist_fp16.data(), labels_fp16.data(), -1);

    faiss::knn_inner_products_by_idx(
        x_fp32.data(), y_fp32.data(), ids.data(), d, nx, ny, nsubset, k, dist_fp32.data(), labels_fp32.data(), -1);

    const float tolerance = 1e-1f;
    size_t total_comparisons = 0;
    size_t consistent_comparisons = 0;

    for (size_t query_idx = 0; query_idx < nx; query_idx++) {
        size_t base_idx = query_idx * k;

        for (size_t neighbor_idx = 0; neighbor_idx < k; neighbor_idx++) {
            size_t idx = base_idx + neighbor_idx;

            if (labels_fp16[idx] >= 0 && labels_fp32[idx] >= 0) {
                total_comparisons++;

                bool id_match = (labels_fp16[idx] == labels_fp32[idx]);
                bool id_in_topk = false;

                for (size_t i = 0; i < k; i++) {
                    if (labels_fp16[idx] == labels_fp32[base_idx + i]) {
                        id_in_topk = true;
                        break;
                    }
                }

                if (id_match) {
                    float diff = std::abs(dist_fp16[idx] - dist_fp32[idx]);
                    if (diff < tolerance) {
                        consistent_comparisons++;
                    }

                    EXPECT_NEAR(dist_fp16[idx], dist_fp32[idx], tolerance)
                        << "Value consistency test failed for query " << query_idx << ", neighbor " << neighbor_idx
                        << ", ID: " << labels_fp16[idx];
                } else if (id_in_topk) {

                    consistent_comparisons++;
                }
            }
        }
    }

    double consistency_rate = static_cast<double>(consistent_comparisons) / total_comparisons;
    std::cout << "KNN by ID consistency test results: " << consistent_comparisons << "/" << total_comparisons
              << " passed (" << consistency_rate * 100 << "%)" << std::endl;

    EXPECT_GE(consistency_rate, 0.90) << "FP16-FP32 KNN by ID consistency rate too low: " << consistency_rate * 100
                                      << "%";
}

TEST_F(TestDistances, range_search_L2sqr_f16)
{
    faiss::RangeSearchResult *res_fp16 = new faiss::RangeSearchResult(nx);
    faiss::RangeSearchResult *res_fp32 = new faiss::RangeSearchResult(nx);

    faiss::range_search_L2sqr_f16(x_fp16.data(), y_fp16.data(), d, nx, ny, radius, res_fp16, nullptr);
    faiss::range_search_L2sqr(x_fp32.data(), y_fp32.data(), d, nx, ny, radius, res_fp32, nullptr);

    validate_results_consistency(res_fp16, res_fp32, nx, radius);
    delete res_fp16;
    delete res_fp32;
}

TEST_F(TestDistances, range_search_inner_product_f16)
{
    faiss::RangeSearchResult *res_fp16 = new faiss::RangeSearchResult(nx);
    faiss::RangeSearchResult *res_fp32 = new faiss::RangeSearchResult(nx);
    normalize_fp16(x_fp16, nx, d);
    normalize_fp16(y_fp16, ny, d);
    normalize_fp32(x_fp32, nx, d);
    normalize_fp32(y_fp32, ny, d);

    faiss::range_search_inner_product_f16(x_fp16.data(), y_fp16.data(), d, nx, ny, radius, res_fp16, nullptr);
    faiss::range_search_inner_product(x_fp32.data(), y_fp32.data(), d, nx, ny, radius, res_fp32, nullptr);

    validate_results_consistency(res_fp16, res_fp32, nx, radius);
    delete res_fp16;
    delete res_fp32;
}

TEST_F(TestDistances, convert_fp16_to_fp32)
{
    faiss::convert_fp16_to_fp32(src_fp16.data(), d, out_fp32.data());

    for (int64_t i = 0; i < d; i++) {
        EXPECT_NEAR(out_fp32[i], ref_fp32[i], 1e-1f) << "Conversion error at index " << i << ": FP16=" << src_fp16[i]
                                                     << ", got " << out_fp32[i] << ", expected " << ref_fp32[i];
    }
}
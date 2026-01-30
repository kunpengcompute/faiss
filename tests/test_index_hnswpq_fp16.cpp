#include "common_test_functions.h"

class TestIndexHNSWPQ : public ::testing::Test {
protected:
    void SetUp() override
    {
        d = 128;
        nb = 20000;
        nq = 20;
        k = 10;
        M = 64;
        efc = 500;
        efs = 200;
        pq_m = 64;
        pq_nbits = 8;
        ntype = faiss::NumericType::Float16;

        // 生成测试数据
        xb_fp32.resize(d * nb);
        xq_fp32.resize(d * nq);

        faiss::float_rand(xb_fp32.data(), d * nb, 1234);
        faiss::float_rand(xq_fp32.data(), d * nq, 5678);

        // 转换为FP16
        xb_fp16.resize(d * nb);
        xq_fp16.resize(d * nq);
        for (size_t i = 0; i < xb_fp32.size(); i++) {
            xb_fp16[i] = static_cast<float16_t>(xb_fp32[i]);
        }
        for (size_t i = 0; i < xq_fp32.size(); i++) {
            xq_fp16[i] = static_cast<float16_t>(xq_fp32[i]);
        }

        // 创建索引
        index_l2_fp16 = new faiss::IndexHNSWPQ(d, pq_m, M, pq_nbits);
        index_l2_fp32 = new faiss::IndexHNSWPQ(d, pq_m, M, pq_nbits);

        // 设置构建和搜索参数
        hnsw_params.efSearch = efs;
    }

    void init_index()
    {
        index_l2_fp16->hnsw.efConstruction = efc;
        index_l2_fp32->hnsw.efConstruction = efc;

        index_l2_fp16->train_ex(nb, xb_fp16.data(), ntype);
        index_l2_fp32->train(nb, xb_fp32.data());

        index_l2_fp16->add_ex(nb, xb_fp16.data(), ntype);
        index_l2_fp32->add(nb, xb_fp32.data());
    }

    void TearDown() override
    {
        delete index_l2_fp16;
        delete index_l2_fp32;
    }

    int d;
    int nb;
    int nq;
    int k;
    int M;
    int efc;
    int efs;
    int pq_m, pq_nbits;
    faiss::SearchParametersHNSW hnsw_params;
    faiss::NumericType ntype;
    std::vector<float16_t> xb_fp16, xq_fp16;
    std::vector<float> xb_fp32, xq_fp32;
    faiss::IndexHNSWPQ *index_l2_fp16;
    faiss::IndexHNSWPQ *index_l2_fp32;
};

// add
TEST_F(TestIndexHNSWPQ, add)
{
    index_l2_fp16->train_ex(nb, xb_fp16.data(), ntype);
    index_l2_fp16->add_ex(nb, xb_fp16.data(), ntype);
    EXPECT_EQ(index_l2_fp16->ntotal, nb);
    EXPECT_GT(index_l2_fp16->hnsw.max_level, 0);
}

TEST_F(TestIndexHNSWPQ, add_ex)
{
    index_l2_fp32->train(nb, xb_fp32.data());
    EXPECT_THROW(index_l2_fp32->add_ex(nb, xb_fp32.data(), faiss::NumericType::Float32), faiss::FaissException);
}

// search
TEST_F(TestIndexHNSWPQ, search)
{
    init_index();

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);

    index_l2_fp16->search_ex(nq, xq_fp16.data(), k, distances_fp16.data(), labels_fp16.data(), ntype, &hnsw_params);
    index_l2_fp32->search(nq, xq_fp32.data(), k, distances_fp32.data(), labels_fp32.data(), &hnsw_params);

    compareLabelsDetailed(labels_fp16, labels_fp32, nq, k);
}

TEST_F(TestIndexHNSWPQ, search_ex)
{
    init_index();

    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);

    EXPECT_THROW(index_l2_fp32->search_ex(nq,
                     xq_fp32.data(),
                     k,
                     distances_fp32.data(),
                     labels_fp32.data(),
                     faiss::NumericType::Float32,
                     &hnsw_params),
        faiss::FaissException);
}

// range_search
TEST_F(TestIndexHNSWPQ, range_search)
{
    init_index();

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);
    float radius = 100.0f;
    faiss::RangeSearchResult result_fp32(nq);
    faiss::RangeSearchResult result_fp16(nq);

    index_l2_fp16->range_search_ex(nq, xq_fp16.data(), radius, &result_fp16, ntype, &hnsw_params);
    index_l2_fp32->range_search(nq, xq_fp32.data(), radius, &result_fp32, &hnsw_params);

    PrintRangeSearchComparison(index_l2_fp32, index_l2_fp16, nq, xq_fp32.data(), xq_fp16.data(), radius);
}

TEST_F(TestIndexHNSWPQ, range_search_ex)
{
    init_index();

    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);
    float radius = 100.0f;
    faiss::RangeSearchResult result_fp32(nq);

    EXPECT_THROW(index_l2_fp32->range_search_ex(nq, xq_fp32.data(), radius, &result_fp32, faiss::NumericType::Float32),
        faiss::FaissException);
}

// reconstruct
TEST_F(TestIndexHNSWPQ, reconstruct)
{
    std::vector<faiss::idx_t> test_ids = {1, 2};
    init_index();

    for (int i = 0; i < nb * d; i++) {
        float16_t fp16_val = (float16_t)xb_fp32[i];
        EXPECT_NEAR((float)xb_fp16[i], (float)fp16_val, 1e-3f) << "输入数据在位置 " << i << " 处不一致";
    }

    for (faiss::idx_t test_id : test_ids) {
        testReconstructionConsistency(index_l2_fp16, index_l2_fp32, xb_fp16.data(), xb_fp32.data(), test_id, d, 1e-1f);
    }
}

TEST_F(TestIndexHNSWPQ, reconstruct_ex)
{
    init_index();

    EXPECT_THROW(std::vector<float> reconstructed_fp32(d);
                 index_l2_fp32->reconstruct_ex(1, reconstructed_fp32.data(), faiss::NumericType::Float32),
                 faiss::FaissException);
}

// search_level_0
TEST_F(TestIndexHNSWPQ, search_level_0)
{
    init_index();

    int nprobe = 10;

    std::vector<float> init_distances_fp32(nq * nprobe);
    std::vector<faiss::idx_t> init_labels_fp32(nq * nprobe);

    index_l2_fp32->search(
        nq, xq_fp32.data(), nprobe, init_distances_fp32.data(), init_labels_fp32.data(), &hnsw_params);

    std::vector<float> nearest_d(nq * nprobe);
    std::vector<faiss::HNSW::storage_idx_t> nearest(nq * nprobe);

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < nprobe; j++) {
            nearest[i * nprobe + j] = init_labels_fp32[i * nprobe + j];
            nearest_d[i * nprobe + j] = init_distances_fp32[i * nprobe + j];
        }
    }

    std::vector<float> dist_fp32(nq * k), dist_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k), labels_fp16(nq * k);

    index_l2_fp32->search_level_0(
        nq, xq_fp32.data(), k, nearest.data(), nearest_d.data(), dist_fp32.data(), labels_fp32.data(), nprobe, 0);

    std::vector<float> init_distances_fp16(nq * nprobe);
    std::vector<faiss::idx_t> init_labels_fp16(nq * nprobe);
    index_l2_fp16->search_ex(
        nq, xq_fp16.data(), nprobe, init_distances_fp16.data(), init_labels_fp16.data(), ntype, &hnsw_params);

    std::vector<float> nearest_d_fp16(nq * nprobe);
    std::vector<faiss::HNSW::storage_idx_t> nearest_fp16(nq * nprobe);
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < nprobe; j++) {
            nearest_fp16[i * nprobe + j] = init_labels_fp16[i * nprobe + j];
            nearest_d_fp16[i * nprobe + j] = init_distances_fp16[i * nprobe + j];
        }
    }

    index_l2_fp16->search_level_0_ex(nq,
        xq_fp16.data(),
        k,
        nearest_fp16.data(),
        nearest_d_fp16.data(),
        dist_fp16.data(),
        labels_fp16.data(),
        ntype,
        nprobe,
        0);

    int matching_queries = 0;
    for (int i = 0; i < nq; i++) {
        bool query_matches = true;
        for (int j = 0; j < k; j++) {
            if (labels_fp32[i * k + j] != labels_fp16[i * k + j]) {
                query_matches = false;
                break;
            }
            float diff = std::abs(dist_fp32[i * k + j] - dist_fp16[i * k + j]);
            EXPECT_LT(diff, 1e-2f) << "距离差异过大 at query=" << i;
        }
        if (query_matches)
            matching_queries++;
    }

    float match_rate = static_cast<float>(matching_queries) / nq;
    EXPECT_GT(match_rate, 0.8f) << "FP16/FP32标签匹配率过低: " << match_rate;
}

TEST_F(TestIndexHNSWPQ, search_level_0_ex)
{
    init_index();

    int nprobe = 10;

    std::vector<float> init_distances_fp32(nq * nprobe);
    std::vector<faiss::idx_t> init_labels_fp32(nq * nprobe);

    index_l2_fp32->search(
        nq, xq_fp32.data(), nprobe, init_distances_fp32.data(), init_labels_fp32.data(), &hnsw_params);

    std::vector<float> nearest_d(nq * nprobe);
    std::vector<faiss::HNSW::storage_idx_t> nearest(nq * nprobe);

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < nprobe; j++) {
            nearest[i * nprobe + j] = init_labels_fp32[i * nprobe + j];
            nearest_d[i * nprobe + j] = init_distances_fp32[i * nprobe + j];
        }
    }

    std::vector<float> dist_fp32(nq * k), dist_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k), labels_fp16(nq * k);

    EXPECT_THROW(index_l2_fp32->search_level_0_ex(nq,
                     xq_fp32.data(),
                     k,
                     nearest.data(),
                     nearest_d.data(),
                     dist_fp32.data(),
                     labels_fp32.data(),
                     faiss::NumericType::Float32,
                     nprobe,
                     0),
        faiss::FaissException);
}

// train
TEST_F(TestIndexHNSWPQ, train)
{
    index_l2_fp32->train(nb, xb_fp32.data());
    index_l2_fp16->train_ex(nb, xb_fp16.data(), ntype);

    EXPECT_TRUE(isIndexTrained(index_l2_fp32));
    EXPECT_TRUE(isIndexTrained(index_l2_fp16));

    index_l2_fp32->add(nb, xb_fp32.data());
    index_l2_fp16->add_ex(nb, xb_fp16.data(), ntype);

    std::vector<float> dist_fp32(nq * k), dist_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k), labels_fp16(nq * k);

    index_l2_fp32->search(nq, xq_fp32.data(), k, dist_fp32.data(), labels_fp32.data());
    index_l2_fp16->search_ex(nq, xq_fp16.data(), k, dist_fp16.data(), labels_fp16.data(), ntype);

    compareLabelsDetailed(labels_fp16, labels_fp32, nq, k);
}

TEST_F(TestIndexHNSWPQ, train_ex)
{
    EXPECT_THROW(index_l2_fp32->train_ex(nb, xb_fp32.data(), faiss::NumericType::Float32), faiss::FaissException);
}
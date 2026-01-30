#include "common_test_functions.h"

class TestIndexHNSW : public ::testing::Test {
protected:
    void SetUp() override
    {
        d = 128;
        nb = 3025;
        nq = 10;
        k = 10;
        M = 128;
        efc = 500;
        efs = 200;
        ntype = faiss::NumericType::Float16;

        // 生成测试数据
        xb_fp32.resize(d * nb);
        xq_fp32.resize(d * nq);

        faiss::float_rand(xb_fp32.data(), d * nb, 42);
        faiss::float_rand(xq_fp32.data(), d * nq, 43);

        // 转换为FP16
        xb_fp16.resize(d * nb);
        xq_fp16.resize(d * nq);

        faiss::convert_fp32_to_fp16(xb_fp32.data(), d * nb, xb_fp16.data());
        faiss::convert_fp32_to_fp16(xq_fp32.data(), d * nq, xq_fp16.data());

        // 创建索引
        index_l2_fp16 = new faiss::IndexHNSWFlat(d, M, ntype, faiss::METRIC_L2);
        index_ip_fp16 = new faiss::IndexHNSWFlat(d, M, ntype, faiss::METRIC_INNER_PRODUCT);
        index_l2_fp32 = new faiss::IndexHNSWFlat(d, M, faiss::METRIC_L2);
        index_ip_fp32 = new faiss::IndexHNSWFlat(d, M, faiss::METRIC_INNER_PRODUCT);

        // 设置构建和搜索参数
        hnsw_params.efSearch = efs;
    }

    void init_index()
    {
        index_l2_fp16->hnsw.efConstruction = efc;
        index_l2_fp32->hnsw.efConstruction = efc;
        index_ip_fp16->hnsw.efConstruction = efc;
        index_ip_fp32->hnsw.efConstruction = efc;

        index_l2_fp16->add_ex(nb, xb_fp16.data(), ntype);
        index_l2_fp32->add(nb, xb_fp32.data());
        index_ip_fp16->add_ex(nb, xb_fp16.data(), ntype);
        index_ip_fp32->add(nb, xb_fp32.data());
    }

    void TearDown() override
    {
        delete index_l2_fp16;
        delete index_ip_fp16;
        delete index_l2_fp32;
        delete index_ip_fp32;
    }

    int d;
    int nb;
    int nq;
    int k;
    int M;
    int efc;
    int efs;
    faiss::NumericType ntype;
    faiss::SearchParametersHNSW hnsw_params;
    std::vector<float16_t> xb_fp16, xq_fp16;
    std::vector<float> xb_fp32, xq_fp32;
    faiss::IndexHNSWFlat *index_l2_fp16;
    faiss::IndexHNSWFlat *index_ip_fp16;
    faiss::IndexHNSWFlat *index_l2_fp32;
    faiss::IndexHNSWFlat *index_ip_fp32;
};

// add
TEST_F(TestIndexHNSW, add)
{
    index_l2_fp16->add_ex(nb, xb_fp16.data(), ntype);
    index_l2_fp32->add(nb, xb_fp32.data());
    EXPECT_EQ(index_l2_fp16->ntotal, nb);
    EXPECT_GT(index_l2_fp16->hnsw.max_level, 0);
    EXPECT_EQ(index_l2_fp16->hnsw.max_level, index_l2_fp32->hnsw.max_level);
    for (int i = 0; i < nb; i++) {
        auto neighbors_fp16 = get_neighbors(index_l2_fp16->hnsw, i, 0);
        auto neighbors_fp32 = get_neighbors(index_l2_fp32->hnsw, i, 0);

        EXPECT_EQ(neighbors_fp16.size(), neighbors_fp32.size());
    }

    EXPECT_EQ(index_l2_fp16->hnsw.entry_point, index_l2_fp32->hnsw.entry_point);

    for (int i = 0; i < nb; i++) {
        EXPECT_EQ(index_l2_fp16->hnsw.levels[i], index_l2_fp32->hnsw.levels[i]);
    }

    EXPECT_EQ(index_l2_fp16->hnsw.cum_nneighbor_per_level, index_l2_fp32->hnsw.cum_nneighbor_per_level);

    index_ip_fp16->add_ex(nb, xb_fp16.data(), ntype);
    index_ip_fp32->add(nb, xb_fp32.data());
    EXPECT_EQ(index_ip_fp16->ntotal, nb);
    EXPECT_GT(index_ip_fp16->hnsw.max_level, 0);
    EXPECT_EQ(index_ip_fp16->hnsw.max_level, index_ip_fp32->hnsw.max_level);
}

TEST_F(TestIndexHNSW, add_ex)
{
    EXPECT_THROW(index_l2_fp32->add_ex(nb, xb_fp32.data(), faiss::NumericType::Float32), faiss::FaissException);
}

// search
TEST_F(TestIndexHNSW, search_l2)
{
    init_index();

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);

    index_l2_fp16->search_ex(nq, xq_fp16.data(), k, distances_fp16.data(), labels_fp16.data(), ntype, &hnsw_params);
    index_l2_fp32->search(nq, xq_fp32.data(), k, distances_fp32.data(), labels_fp32.data(), &hnsw_params);

    float recall = compute_recall(labels_fp16, labels_fp32, nq, k);
    EXPECT_GE(recall, 0.6f) << "Recall too low: " << recall;
}

TEST_F(TestIndexHNSW, search_ip)
{
    init_index();

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);
    normalize_fp16(xq_fp16, nq, d);
    normalize_fp32(xq_fp32, nq, d);

    index_ip_fp16->search_ex(nq, xq_fp16.data(), k, distances_fp16.data(), labels_fp16.data(), ntype, &hnsw_params);
    index_ip_fp32->search(nq, xq_fp32.data(), k, distances_fp32.data(), labels_fp32.data(), &hnsw_params);

    float recall = compute_recall(labels_fp16, labels_fp32, nq, k);
    EXPECT_GE(recall, 0.6f) << "Recall too low: " << recall;
}

TEST_F(TestIndexHNSW, search_ex)
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

    EXPECT_THROW(index_ip_fp32->search_ex(nq,
                     xq_fp32.data(),
                     k,
                     distances_fp32.data(),
                     labels_fp32.data(),
                     faiss::NumericType::Float32,
                     &hnsw_params),
        faiss::FaissException);
}

// range_search
TEST_F(TestIndexHNSW, range_search)
{
    init_index();

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);
    float radius = 100.0f;
    faiss::RangeSearchResult result_fp32(nq);
    faiss::RangeSearchResult result_fp16(nq);

    PrintRangeSearchComparison(index_l2_fp32, index_l2_fp16, nq, xq_fp32.data(), xq_fp16.data(), radius);
}

TEST_F(TestIndexHNSW, range_search_ex)
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
TEST_F(TestIndexHNSW, reconstruct)
{
    std::vector<faiss::idx_t> test_ids = {1, 2};

    init_index();

    for (int i = 0; i < nb * d; i++) {
        float16_t fp16_val = (float16_t)xb_fp32[i];
        EXPECT_NEAR((float)xb_fp16[i], (float)fp16_val, 1e-3f) << "输入数据在位置 " << i << " 处不一致";
    }

    for (faiss::idx_t test_id : test_ids) {
        testReconstructionConsistency(index_l2_fp16, index_l2_fp32, xb_fp16.data(), xb_fp32.data(), test_id, d, 1e-2f);
    }
}

TEST_F(TestIndexHNSW, reconstruct_ex)
{
    init_index();

    EXPECT_THROW(std::vector<float> reconstructed_fp32(d);
                 index_l2_fp32->reconstruct_ex(1, reconstructed_fp32.data(), faiss::NumericType::Float32),
                 faiss::FaissException);
}

// search_level_0
TEST_F(TestIndexHNSW, search_level_0)
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

TEST_F(TestIndexHNSW, search_level_0_ex)
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
TEST_F(TestIndexHNSW, train)
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

    float recall = compute_recall(labels_fp16, labels_fp32, nq, k);
    EXPECT_GE(recall, 0.6f) << "Recall too low: " << recall;
}

TEST_F(TestIndexHNSW, train_ex)
{
    EXPECT_THROW(index_l2_fp32->train_ex(nb, xb_fp32.data(), faiss::NumericType::Float32), faiss::FaissException);
}

#ifdef KRL
TEST_F(TestIndexHNSW, quant_f16)
{
    index_l2_fp16->quant_bits = 16;
    index_l2_fp16->add_ex(nb, xb_fp16.data(), ntype);
    index_l2_fp32->add(nb, xb_fp32.data());

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);

    index_l2_fp16->search_ex(nq, xq_fp16.data(), k, distances_fp16.data(), labels_fp16.data(), ntype);
    index_l2_fp32->search(nq, xq_fp32.data(), k, distances_fp32.data(), labels_fp32.data());

    compareLabelsDetailed(labels_fp16, labels_fp32, nq, k);
}

TEST_F(TestIndexHNSW, quant_u8)
{
    index_l2_fp16->quant_bits = 8;
    index_l2_fp16->add_ex(nb, xb_fp16.data(), ntype);
    index_l2_fp32->add(nb, xb_fp32.data());

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);

    index_l2_fp16->search_ex(nq, xq_fp16.data(), k, distances_fp16.data(), labels_fp16.data(), ntype);
    index_l2_fp32->search(nq, xq_fp32.data(), k, distances_fp32.data(), labels_fp32.data());

    compareLabelsDetailed(labels_fp16, labels_fp32, nq, k);
}
#endif
#include "common_test_functions.h"

class TestIndexFlat : public ::testing::Test {
protected:
    void SetUp() override
    {
        d = 128;
        nb = 3000;
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
        index_l2_fp16 = new faiss::IndexFlat(d, ntype, faiss::METRIC_L2);
        index_ip_fp16 = new faiss::IndexFlat(d, ntype, faiss::METRIC_INNER_PRODUCT);
        index_l2_fp32 = new faiss::IndexFlat(d, faiss::METRIC_L2);
        index_ip_fp32 = new faiss::IndexFlat(d, faiss::METRIC_INNER_PRODUCT);
    }

    void init_index()
    {
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
    std::vector<float16_t> xb_fp16, xq_fp16;
    std::vector<float> xb_fp32, xq_fp32;
    faiss::IndexFlat *index_l2_fp16;
    faiss::IndexFlat *index_ip_fp16;
    faiss::IndexFlat *index_l2_fp32;
    faiss::IndexFlat *index_ip_fp32;
};

// add
TEST_F(TestIndexFlat, add)
{
    index_l2_fp16->add_ex(nb, xb_fp16.data(), ntype);
    EXPECT_EQ(index_l2_fp16->ntotal, nb);
}

TEST_F(TestIndexFlat, add_ex)
{
    EXPECT_THROW(index_l2_fp32->add_ex(nb, xb_fp32.data(), faiss::NumericType::Float32), faiss::FaissException);
}

// search
TEST_F(TestIndexFlat, search_l2)
{
    init_index();

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);

    index_l2_fp16->search_ex(nq, xq_fp16.data(), k, distances_fp16.data(), labels_fp16.data(), ntype);
    index_l2_fp32->search(nq, xq_fp32.data(), k, distances_fp32.data(), labels_fp32.data());

    compareLabelsDetailed(labels_fp16, labels_fp32, nq, k);
}

TEST_F(TestIndexFlat, search_ip)
{
    init_index();

    std::vector<float> distances_fp16(nq * k);
    std::vector<float> distances_fp32(nq * k);
    std::vector<faiss::idx_t> labels_fp16(nq * k);
    std::vector<faiss::idx_t> labels_fp32(nq * k);

    index_ip_fp16->search_ex(nq, xq_fp16.data(), k, distances_fp16.data(), labels_fp16.data(), ntype, &hnsw_params);
    index_ip_fp32->search(nq, xq_fp32.data(), k, distances_fp32.data(), labels_fp32.data(), &hnsw_params);

    compareLabelsDetailed(labels_fp16, labels_fp32, nq, k);
}

TEST_F(TestIndexFlat, search_ex)
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
TEST_F(TestIndexFlat, range_search)
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

TEST_F(TestIndexFlat, range_search_ex)
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
TEST_F(TestIndexFlat, reconstruct)
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

TEST_F(TestIndexFlat, reconstruct_ex)
{
    init_index();

    EXPECT_THROW(std::vector<float> reconstructed_fp32(d);
                 index_l2_fp32->reconstruct_ex(1, reconstructed_fp32.data(), faiss::NumericType::Float32),
                 faiss::FaissException);
}

// train
TEST_F(TestIndexFlat, train)
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

TEST_F(TestIndexFlat, train_ex)
{
    EXPECT_THROW(index_l2_fp32->train_ex(nb, xb_fp32.data(), faiss::NumericType::Float32), faiss::FaissException);
}

TEST_F(TestIndexFlat, ntype)
{
    EXPECT_NO_THROW({ IndexFlat index_fp32(d, faiss::NumericType::Float32, faiss::METRIC_L2); });
    EXPECT_NO_THROW({ IndexFlat index_fp16(d, faiss::NumericType::Float16, faiss::METRIC_L2); });
    EXPECT_THROW({ IndexFlat index(d, 10, faiss::METRIC_L2); },
        std::invalid_argument
    );
}

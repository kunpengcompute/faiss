#include "common_test_functions.h"

bool isIndexTrained(faiss::Index *index)
{
    return index->is_trained;
}

faiss::idx_t getIndexTotal(faiss::Index *index)
{
    return index->ntotal;
}

std::vector<faiss::HNSW::storage_idx_t> get_neighbors(const faiss::HNSW &hnsw, int node_id, int level)
{
    std::vector<faiss::HNSW::storage_idx_t> result;

    // 检查节点是否存在且具有指定层级
    if (node_id < 0 || node_id >= hnsw.levels.size() || level > hnsw.levels[node_id]) {
        return result;
    }

    // 使用neighbor_range方法获取邻居范围
    size_t begin, end;
    hnsw.neighbor_range(node_id, level, &begin, &end);

    // 提取邻居索引
    for (size_t i = begin; i < end; i++) {
        if (i < hnsw.neighbors.size()) {
            result.push_back(hnsw.neighbors[i]);
        }
    }

    return result;
}

int get_connection_count(const faiss::HNSW &hnsw, int node_id, int level)
{
    if (node_id < 0 || node_id >= hnsw.levels.size() || level > hnsw.levels[node_id]) {
        return 0;
    }
    size_t begin, end;
    hnsw.neighbor_range(node_id, level, &begin, &end);
    return static_cast<int>(end - begin);
}

void compareDistances(
    const std::vector<float> &dist_fp16, const std::vector<float> &dist_fp32, int nq, float absolute_tolerance)
{
    ASSERT_EQ(dist_fp16.size(), dist_fp32.size())
        << "FP16和FP32距离数组大小不一致: " << dist_fp16.size() << " vs " << dist_fp32.size();

    for (size_t i = 0; i < dist_fp16.size(); i++) {
        EXPECT_NEAR(dist_fp16[i], dist_fp32[i], absolute_tolerance)
            << "距离比较失败 at index " << i << ": FP16=" << dist_fp16[i] << ", FP32=" << dist_fp32[i]
            << ", 绝对误差=" << std::abs(dist_fp16[i] - dist_fp32[i]) << ", 相对误差="
            << (std::abs(dist_fp16[i] - dist_fp32[i]) / std::max(std::abs(dist_fp16[i]), std::abs(dist_fp32[i]))) * 100
            << "%";
    }

    std::cout << "距离比较统计: 总比较数=" << dist_fp16.size() << ", 容忍度=" << absolute_tolerance << std::endl;
}

void compareLabelsDetailed(
    const std::vector<faiss::idx_t> &labels_fp16, const std::vector<faiss::idx_t> &labels_fp32, int nq, int k)
{
    ASSERT_EQ(labels_fp16.size(), labels_fp32.size());

    int different_queries = 0;

    for (int i = 0; i < nq; i++) {
        std::vector<faiss::idx_t> set16(labels_fp16.begin() + i * k, labels_fp16.begin() + (i + 1) * k);
        std::vector<faiss::idx_t> set32(labels_fp32.begin() + i * k, labels_fp32.begin() + (i + 1) * k);

        std::vector<faiss::idx_t> sorted16 = set16;
        std::vector<faiss::idx_t> sorted32 = set32;
        std::sort(sorted16.begin(), sorted16.end());
        std::sort(sorted32.begin(), sorted32.end());

        if (sorted16 != sorted32) {
            different_queries++;
            std::cout << "查询 " << i << " 标签不匹配:" << std::endl;
            std::cout << "  FP16标签: ";
            for (auto label : sorted16)
                std::cout << label << " ";
            std::cout << std::endl << "  FP32标签: ";
            for (auto label : sorted32)
                std::cout << label << " ";
            std::cout << std::endl;
        }
    }

    std::cout << "标签比较统计: 总查询数=" << nq << ", 不匹配查询数=" << different_queries
              << ", 匹配率=" << (100.0 * (nq - different_queries) / nq) << "%" << std::endl;
}

void CompareRangeSearchResults(
    const faiss::RangeSearchResult *result_fp32, const faiss::RangeSearchResult *result_fp16, float tolerance)
{

    // 1. 检查查询数量是否一致
    ASSERT_EQ(result_fp32->nq, result_fp16->nq);

    int nq = result_fp32->nq;
    int total_matches_fp32 = result_fp32->lims[nq];
    int total_matches_fp16 = result_fp16->lims[nq];

    std::cout << "总匹配数 - FP32: " << total_matches_fp32 << ", FP16: " << total_matches_fp16 << std::endl;

    // 2. 逐个查询比较
    for (int i = 0; i < nq; i++) {
        int start_fp32 = result_fp32->lims[i];
        int end_fp32 = result_fp32->lims[i + 1];
        int start_fp16 = result_fp16->lims[i];
        int end_fp16 = result_fp16->lims[i + 1];

        int count_fp32 = end_fp32 - start_fp32;
        int count_fp16 = end_fp16 - start_fp16;

        // 比较每个查询的结果数量
        EXPECT_NEAR(count_fp32, count_fp16, std::max(1, count_fp32 / 10)) << "查询 " << i << " 的结果数量差异过大";

        // 创建集合以便比较标签
        std::unordered_set<faiss::idx_t> labels_fp32_set(
            result_fp32->labels + start_fp32, result_fp32->labels + end_fp32);
        std::unordered_set<faiss::idx_t> labels_fp16_set(
            result_fp16->labels + start_fp16, result_fp16->labels + end_fp16);

        // 3. 计算Jaccard相似度（交并比）
        std::vector<faiss::idx_t> intersection;
        std::set_intersection(labels_fp32_set.begin(),
            labels_fp32_set.end(),
            labels_fp16_set.begin(),
            labels_fp16_set.end(),
            std::back_inserter(intersection));

        float jaccard_similarity =
            intersection.size() * 1.0f /
            std::max(labels_fp32_set.size() + labels_fp16_set.size() - intersection.size(), size_t(1));

        EXPECT_GT(jaccard_similarity, 0.9f)  // 期望相似度大于90%
            << "查询 " << i << " 的标签集合相似度过低: " << jaccard_similarity;

        // 4. 比较共同标签对应的距离值
        for (auto label : intersection) {
            // 在FP32结果中查找距离
            float dist_fp32 = 0;
            for (int j = start_fp32; j < end_fp32; j++) {
                if (result_fp32->labels[j] == label) {
                    dist_fp32 = result_fp32->distances[j];
                    break;
                }
            }

            // 在FP16结果中查找距离
            float dist_fp16 = 0;
            for (int j = start_fp16; j < end_fp16; j++) {
                if (result_fp16->labels[j] == label) {
                    dist_fp16 = result_fp16->distances[j];
                    break;
                }
            }

            // 使用EXPECT_NEAR比较距离值
            EXPECT_NEAR(dist_fp32, dist_fp16, tolerance) << "标签 " << label << " 的距离值差异过大";
        }
    }
}

float compareReconstruction(const float16_t *fp16_data, const float *fp32_data, int dim)
{
    float total_error = 0.0f;
    for (int i = 0; i < dim; i++) {
        float fp16_val = (float)fp16_data[i];
        float fp32_val = fp32_data[i];
        float diff = fp16_val - fp32_val;
        total_error += diff * diff;
    }
    return std::sqrt(total_error);
}

void testReconstructionConsistency(faiss::Index *index_fp16, faiss::Index *index_fp32, const float16_t *fp16_data,
    const float *fp32_data, int test_id, int d, float tolerance)
{
    std::vector<float16_t> reconstructed_fp16(d);
    index_fp16->reconstruct_ex(test_id, reconstructed_fp16.data(), faiss::NumericType::Float16);

    std::vector<float> reconstructed_fp32(d);
    index_fp32->reconstruct(test_id, reconstructed_fp32.data());

    const float16_t *original_fp16 = &fp16_data[test_id * d];
    const float *original_fp32 = &fp32_data[test_id * d];

    float input_consistency_error = compareReconstruction(original_fp16, original_fp32, d);
    EXPECT_LT(input_consistency_error, tolerance * d)
        << "输入数据不一致 for id=" << test_id << ", error=" << input_consistency_error;

    float reconstruction_consistency_error =
        compareReconstruction(reconstructed_fp16.data(), reconstructed_fp32.data(), d);

    EXPECT_LT(reconstruction_consistency_error, tolerance * d)
        << "FP16和FP32重建结果不一致 for id=" << test_id << ", error=" << reconstruction_consistency_error;

    std::cout << "测试ID: " << test_id << " | 输入一致性误差: " << input_consistency_error
              << " | 重建一致性误差: " << reconstruction_consistency_error << std::endl;
}

void PrintRangeSearchComparison(faiss::Index *index_fp32, faiss::Index *index_fp16, int nq, const float *xq_fp32,
    const float16_t *xq_fp16, float radius)
{
    faiss::RangeSearchResult res_fp32(nq);
    faiss::RangeSearchResult res_fp16(nq);

    index_fp32->range_search(nq, xq_fp32, radius, &res_fp32);
    index_fp16->range_search_ex(nq, xq_fp16, radius, &res_fp16, faiss::NumericType::Float16);

    for (int i = 0; i < nq; i++) {
        faiss::idx_t start_fp32 = res_fp32.lims[i];
        faiss::idx_t end_fp32 = res_fp32.lims[i + 1];
        faiss::idx_t start_fp16 = res_fp16.lims[i];
        faiss::idx_t end_fp16 = res_fp16.lims[i + 1];

        const faiss::idx_t *labels_fp32 = res_fp32.labels + start_fp32;
        const faiss::idx_t *labels_fp16 = res_fp16.labels + start_fp16;
        const float *distances_fp32 = res_fp32.distances + start_fp32;
        const float *distances_fp16 = res_fp16.distances + start_fp16;

        int n_fp32 = end_fp32 - start_fp32;
        int n_fp16 = end_fp16 - start_fp16;

        std::unordered_set<faiss::idx_t> set_fp32(labels_fp32, labels_fp32 + n_fp32);
        std::unordered_set<faiss::idx_t> set_fp16(labels_fp16, labels_fp16 + n_fp16);

        std::vector<faiss::idx_t> common_labels;
        std::vector<faiss::idx_t> only_in_fp32;
        std::vector<faiss::idx_t> only_in_fp16;

        for (auto label : set_fp32) {
            if (set_fp16.count(label)) {
                common_labels.push_back(label);
            } else {
                only_in_fp32.push_back(label);
            }
        }
        for (auto label : set_fp16) {
            if (!set_fp32.count(label)) {
                only_in_fp16.push_back(label);
            }
        }

        if (!common_labels.empty()) {
            int fail_count = 0;
            for (auto target_label : common_labels) {
                float dist_fp32 = 0;
                for (int k = 0; k < n_fp32; k++) {
                    if (labels_fp32[k] == target_label) {
                        dist_fp32 = distances_fp32[k];
                        break;
                    }
                }
                float dist_fp16 = 0;
                for (int k = 0; k < n_fp16; k++) {
                    if (labels_fp16[k] == target_label) {
                        dist_fp16 = distances_fp16[k];
                        break;
                    }
                }
                float abs_diff = std::abs(dist_fp32 - dist_fp16);
                float rel_diff = (dist_fp32 != 0) ? (abs_diff / std::abs(dist_fp32)) * 100 : 0;
                if (rel_diff > 25) {
                    fail_count++;
                }
            }
            float ans = fail_count / common_labels.size();
            EXPECT_LE(ans, 0.1);
        }
    }
}

float compute_recall(
    const std::vector<faiss::idx_t> &results_fp16, const std::vector<faiss::idx_t> &results_fp32, int nq, int k)
{
    int correct = 0;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            if (results_fp16[i * k + j] == results_fp32[i * k + j]) {
                correct++;
                break;
            }
        }
    }
    return static_cast<float>(correct) / nq;
}

void normalize_fp16(std::vector<float16_t> &vectors, int n, int d)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float16_t norm = 0.0f;
        for (int j = 0; j < d; j++) {
            norm += vectors[i * d + j] * vectors[i * d + j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (int j = 0; j < d; j++) {
                vectors[i * d + j] /= norm;
            }
        }
    }
}

void normalize_fp32(std::vector<float> &vectors, int n, int d)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float norm = 0.0f;
        for (int j = 0; j < d; j++) {
            norm += vectors[i * d + j] * vectors[i * d + j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (int j = 0; j < d; j++) {
                vectors[i * d + j] /= norm;
            }
        }
    }
}

bool validate_ordering_consistency_ip(const float *dist_fp16, const float *dist_fp32, const int64_t *labels_fp16,
    const int64_t *labels_fp32, int k, size_t query_idx)
{
    size_t base_idx = query_idx * k;

    for (size_t i = 1; i < k; i++) {
        if (labels_fp16[base_idx + i] >= 0) {
            EXPECT_LE(dist_fp16[base_idx + i], dist_fp16[base_idx + i - 1])
                << "Array is not sorted in descending order at index " << i;
        }
    }

    for (size_t i = 1; i < k; i++) {
        if (labels_fp32[base_idx + i] >= 0) {
            EXPECT_LE(dist_fp32[base_idx + i], dist_fp32[base_idx + i - 1])
                << "Array is not sorted in descending order at index " << i;
        }
    }

    return true;
}

bool validate_ordering_consistency_l2(const float *dist_fp16, const float *dist_fp32, const int64_t *labels_fp16,
    const int64_t *labels_fp32, int k, size_t query_idx)
{
    size_t base_idx = query_idx * k;

    for (size_t i = 1; i < k; i++) {
        if (labels_fp16[base_idx + i] >= 0) {
            EXPECT_GE(dist_fp16[base_idx + i], dist_fp16[base_idx + i - 1])
                << "Array is not sorted in ascending order at index " << i;
        }
    }

    for (size_t i = 1; i < k; i++) {
        if (labels_fp32[base_idx + i] >= 0) {
            EXPECT_GE(dist_fp32[base_idx + i], dist_fp32[base_idx + i - 1])
                << "Array is not sorted in ascending order at index " << i;
        }
    }

    return true;
}

void validate_results_consistency(faiss::RangeSearchResult *res_fp16, faiss::RangeSearchResult *res_fp32, int nx, int radius)
{
    const float tolerance = 1e-1f;

    for (size_t i = 0; i < nx; i++) {
        size_t count_fp16 = res_fp16->lims[i + 1] - res_fp16->lims[i];
        size_t count_fp32 = res_fp32->lims[i + 1] - res_fp32->lims[i];

        EXPECT_NEAR(count_fp16, count_fp32, std::max(count_fp32 * 0.1f, 5.0f))
            << "Result count mismatch for query " << i;

        if (count_fp16 > 0 && count_fp32 > 0) {
            size_t start_fp16 = res_fp16->lims[i];
            size_t start_fp32 = res_fp32->lims[i];

            for (size_t j = 0; j < std::min(count_fp16, count_fp32); j++) {
                float dist_fp16 = res_fp16->distances[start_fp16 + j];
                float dist_fp32 = res_fp32->distances[start_fp32 + j];

                EXPECT_GE(dist_fp16, 0.0f) << "FP16 distance should be non-negative";
                EXPECT_GE(dist_fp32, 0.0f) << "FP32 distance should be non-negative";
                EXPECT_LE(dist_fp16, radius * 1.1f) << "FP16 distance should be within reasonable range";
                EXPECT_LE(dist_fp32, radius * 1.1f) << "FP32 distance should be within reasonable range";

                if (dist_fp32 > 1e-6f) {
                    float rel_error = std::abs(dist_fp16 - dist_fp32) / dist_fp32;
                    EXPECT_LT(rel_error, 0.1f) << "Relative error too large for query " << i << ", result " << j;
                }
            }
        }
    }
}
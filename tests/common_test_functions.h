#ifndef COMMON_TEST_FUNCTIONS_H
#define COMMON_TEST_FUNCTIONS_H

#include <gtest/gtest.h>
#include <faiss/IndexHNSW.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>

// 辅助函数：检查索引训练状态
bool isIndexTrained(faiss::Index *index);

// 辅助函数：获取索引中的向量数量
faiss::idx_t getIndexTotal(faiss::Index *index);

// 辅助方法：获取指定节点在指定层的邻居列表
std::vector<faiss::HNSW::storage_idx_t> get_neighbors(const faiss::HNSW &hnsw, int node_id, int level);

// 辅助方法：获取节点的连接数量
int get_connection_count(const faiss::HNSW &hnsw, int node_id, int level);

// 比较两个距离数组是否在允许误差范围内一致
void compareDistances(
    const std::vector<float> &dist_fp16, const std::vector<float> &dist_fp32, int nq, float absolute_tolerance = 1e-2f);

// 比较标签
void compareLabelsDetailed(
    const std::vector<faiss::idx_t> &labels_fp16, const std::vector<faiss::idx_t> &labels_fp32, int nq, int k);

// 比较范围查找结果
void CompareRangeSearchResults(
    const faiss::RangeSearchResult *result_fp32, const faiss::RangeSearchResult *result_fp16, float tolerance = 1e-2f);

// 比较FP16和FP32重建向量的差异
float compareReconstruction(const float16_t *fp16_data, const float *fp32_data, int dim);

// 测试FP16和FP32重建结果的一致性
void testReconstructionConsistency(faiss::Index *index_fp16, faiss::Index *index_fp32, const float16_t *fp16_data,
    const float *fp32_data, int test_id, int d, float tolerance = 1e-2f);

void PrintRangeSearchComparison(faiss::Index *index_fp32, faiss::Index *index_fp16, int nq, const float *xq_fp32,
    const float16_t *xq_fp16, float radius);

float compute_recall(const std::vector<faiss::idx_t> &results_fp16, const std::vector<faiss::idx_t> &results_fp32, int nq, int k);

void normalize_fp16(std::vector<float16_t> &vectors, int n, int d);

void normalize_fp32(std::vector<float> &vectors, int n, int d);

bool validate_ordering_consistency_ip(const float *dist_fp16, const float *dist_fp32, const int64_t *labels_fp16,
    const int64_t *labels_fp32, int k, size_t query_idx);

bool validate_ordering_consistency_l2(const float *dist_fp16, const float *dist_fp32, const int64_t *labels_fp16,
    const int64_t *labels_fp32, int k, size_t query_idx);

void validate_results_consistency(faiss::RangeSearchResult *res_fp16, faiss::RangeSearchResult *res_fp32, int nx, int radius);
#endif  // COMMON_TEST_FUNCTIONS_H
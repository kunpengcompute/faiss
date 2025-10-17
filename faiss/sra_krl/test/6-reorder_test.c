/*
test API list:
1. krl_create_reorder_handle
2. krl_clean_distance_handle
3. krl_reorder_2_vector(accu_level = 1,2,3)
4. krl_reorder_2_vector_continuous(accu_level = 1,2,3)
*/

// #include "tools.h"
// #include <vector>
// #include <algorithm>
// #include <functional>
// #include <cmath>
// extern "C" {
// #include "krl.h"
// }

// void normalize(size_t n, float* vec, float mean, float var) {
//     if (n == 0){
//         return;
//     }
//     double average = 0.0;
//     for (size_t i = 0; i < n; ++i) {
//         average += vec[i];
//     }
//     average /= n;
//     double variance = 0.0;
//     for (size_t i = 0; i < n; ++i) {
//         variance += (vec[i] - average) * (vec[i] - average);
//     }
//     variance /= n;
//     double std_dev = std::sqrt(variance);
//     double std_var = std::sqrt(var);

//     if (std_dev == 0) {
//         #pragma omp parallel for
//         for (size_t i = 0; i < n; ++i) {
//             vec[i] = mean;
//         }
//         return;
//     }

//     #pragma omp parallel for
//     for (size_t i = 0; i < n; ++i) {
//         vec[i] = (vec[i] - average) / std_dev * std_var + mean;
//     }
// }

// void build_database(size_t nx, size_t ny, size_t dim, float* query, float* base, int value_range = 65536, float mean
// = 0.0f, float var = 1.0f) {
//     if(value_range > 0) {
//         init_vector(ny * dim, base,  0, value_range);
//         init_vector(nx * dim, query, 0, value_range);
//     } else {
//         init_vector(ny * dim, base,  value_range, -value_range);
//         init_vector(nx * dim, query, value_range, -value_range);
//     }
//     normalize(ny * dim, base,  mean, var);
//     normalize(nx * dim, query, mean, var);
// }

// template<bool keep_max = false>
// double gt_func_test(int64_t* I, const float* base, const float* query, const int64_t* ids, size_t d, size_t nb,
// size_t nq, int nloops) {
//     float* dis = new float[nb * nq];
//     memset(dis, 0, sizeof(float) * nb * nq);
//     double totaltime = 0;
//     struct timespec startTime, endTime;
//     for(int nloop = 0; nloop < nloops ; ++nloop) {
//         memcpy(I, ids, sizeof(int64_t) * nb * nq);
//         if constexpr (keep_max) {
//             clock_gettime(CLOCK_MONOTONIC, &startTime);
//             for (size_t j = 0; j < nq; ++j) {
//                 for (size_t i = 0; i < nb; ++i) {
//                     dis[i + j * nb] = Ipdistance<float, float>(query + j * d, base + ids[i] * d, d);
//                 }
//                 std::sort(I + j * nb, I + j * nb + nb, [&](int a, int b){
//                     return dis[a + j * nb] > dis[b + j * nb];
//                 });
//             }
//             clock_gettime(CLOCK_MONOTONIC, &endTime);
//         } else {
//             clock_gettime(CLOCK_MONOTONIC, &startTime);
//             for (size_t j = 0; j < nq; ++j) {
//                 for (size_t i = 0; i < nb; ++i) {
//                     dis[i + j * nb] = L2distance<float, float>(query + j * d, base + ids[i] * d, d);
//                 }
//                 std::sort(I + j * nb, I + j * nb + nb, [&](int a, int b){
//                     return dis[a + j * nb] < dis[b + j * nb];
//                 });
//             }
//             clock_gettime(CLOCK_MONOTONIC, &endTime);
//         }
//         totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
//     }
//     delete[] dis;
//     return totaltime / nloops;
// }

// double simd_sparse_func_test(const KRLDistanceHandle* kdh, int64_t* I, const int64_t* ids, const float* query, size_t
// d, size_t nb, size_t nq, int k, int nloops) {
//     int64_t* input_ids = new int64_t[nb * nq];
//     float* dis = new float[nb * nq];
//     float* D = new float[k * nq];
//     memset(dis, 0, sizeof(float) * nb * nq);
//     memset(D, 0, sizeof(float) * k * nq);

//     double totaltime = 0;
//     struct timespec startTime, endTime;
//     for(int nloop = 0; nloop < nloops ; ++nloop) {
//         memcpy(input_ids, ids, nb * nq * sizeof(int64_t));
//         clock_gettime(CLOCK_MONOTONIC, &startTime);
//         for (size_t j = 0; j < nq; ++j) {
//             krl_reorder_2_vector(kdh, nb, dis + j * nb, input_ids + j * nb, query + j * d, k, D + j * k, I + j * k);
//         }
//         clock_gettime(CLOCK_MONOTONIC, &endTime);
//         totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
//     }
//     delete[] dis;
//     delete[] D;
//     delete[] input_ids;
//     return totaltime / nloops;
// }

// double simd_dense_func_test(const KRLDistanceHandle* kdh, int64_t* I, const float* query, size_t d, size_t nb, size_t
// nq, int k, int nloops) {
//     float* D = new float[k * nq];
//     memset(D, 0, sizeof(float) * k * nq);

//     double totaltime = 0;
//     struct timespec startTime, endTime;
//     for(int nloop = 0; nloop < nloops ; ++nloop) {
//         clock_gettime(CLOCK_MONOTONIC, &startTime);
//         for (size_t j = 0; j < nq; ++j) {
//             krl_reorder_2_vector_continuous(kdh, nb, 0, query + j * d, k, D + j * k, I + j * k);
//         }
//         clock_gettime(CLOCK_MONOTONIC, &endTime);
//         totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
//     }
//     delete[] D;
//     return totaltime / nloops;
// }

// double check_recall(const int64_t* gt, const int64_t* test, size_t nb, size_t nq, size_t k) {
//     int n_k = 0;
//     for (int iq = 0; iq < nq; ++iq) {
//         for (int ib = 0; ib < k; ++ib) {
//             for (int jb = 0; jb < k; ++jb) {
//                 if (gt[iq * nb + ib] == test[iq * k + jb]) {
//                     n_k++;
//                     break;
//                 }
//             }
//         }
//     }
//     double recall = 1.0 * n_k / nq / k;
//     return recall;
// }

// void print_result_table(bool verbose, const char* s, double time, size_t ny, size_t k, const int64_t* simd_test =
// nullptr, const int64_t* gt = nullptr) {
//     if (!verbose) {
//         return;
//     }
//     printf("%21s cost: ", s);
//     if (ny > 0 && k > 0) {
//         print_time(time, verbose);
//         double recall = check_recall(gt, simd_test, ny, 1, k);
//         printf(", Recall: %.4f\n", recall);
//     } else {
//         print_time(time, verbose, "\n");
//     }
// }

// void test_reorder(size_t ny, size_t dim, size_t k, size_t nloops, int value_range) {
//     assert(ny >= k);
//     float* base  = new float[dim * ny];
//     float* query = new float[dim];
//     int64_t* idx = new int64_t[ny];

//     build_database(1, ny, dim, query, base, value_range);

//     for (size_t i = 0; i < ny; ++i) {
//         idx[i] = i;
//     }

//     const size_t gt_nloops = std::max((size_t)1, nloops / 10);

//     const int nthr = omp_get_max_threads();
//     #pragma omp parallel num_threads(nthr)
//     {
//         int64_t* gt_test = new int64_t[ny];
//         int64_t* simd_test = new int64_t[k];
//         KRLDistanceHandle *kdh_L2_f32, *kdh_L2_f16, *kdh_L2_u8;
//         KRLDistanceHandle *kdh_IP_f32, *kdh_IP_f16, *kdh_IP_s8;
//         const bool verbose = (omp_get_thread_num() == 0);

//         krl_create_reorder_handle(&kdh_L2_f32, 3, 3, ny, dim, 1, (const uint8_t*)base);
//         krl_create_reorder_handle(&kdh_L2_f16, 2, 3, ny, dim, 1, (const uint8_t*)base);
//         krl_create_reorder_handle(&kdh_L2_u8 , 1, 3, ny, dim, 1, (const uint8_t*)base);
//         krl_create_reorder_handle(&kdh_IP_f32, 3, 3, ny, dim, 0, (const uint8_t*)base);
//         krl_create_reorder_handle(&kdh_IP_f16, 2, 3, ny, dim, 0, (const uint8_t*)base);
//         krl_create_reorder_handle(&kdh_IP_s8 , 1, 3, ny, dim, 0, (const uint8_t*)base);

//         double gt_time, simd_time;
//         // L2 test
//         {
//             // gt
//             gt_time = gt_func_test<false>(gt_test, base, query, idx, dim, ny, 1, gt_nloops);
//             print_result_table(verbose, "L2_flat_f32", gt_time, 0, 0);
//             simd_time = simd_sparse_func_test(kdh_L2_f32, simd_test, idx, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_L2_sparse(f32)", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_sparse_func_test(kdh_L2_f16, simd_test, idx, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_L2_sparse(f16)", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_sparse_func_test(kdh_L2_u8, simd_test, idx, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_L2_sparse(u8 )", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_dense_func_test(kdh_L2_f32, simd_test, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_L2_dense(f32)", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_dense_func_test(kdh_L2_f16, simd_test, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_L2_dense(f16)", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_dense_func_test(kdh_L2_u8, simd_test, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_L2_dense(u8 )", simd_time, ny, k, simd_test, gt_test);
//         }
//         // IP test
//         {
//             gt_time = gt_func_test<true>(gt_test, base, query, idx, dim, ny, 1, gt_nloops);
//             print_result_table(verbose, "IP_flat_f32", gt_time, 0, 0);
//             simd_time = simd_sparse_func_test(kdh_IP_f32, simd_test, idx, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_IP_sparse(f32)", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_sparse_func_test(kdh_IP_f16, simd_test, idx, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_IP_sparse(f16)", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_sparse_func_test(kdh_IP_s8, simd_test, idx, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_IP_sparse(s8 )", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_dense_func_test(kdh_IP_f32, simd_test, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_IP_dense(f32)", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_dense_func_test(kdh_IP_f16, simd_test, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_IP_dense(f16)", simd_time, ny, k, simd_test, gt_test);
//             simd_time = simd_dense_func_test(kdh_IP_s8, simd_test, query, dim, ny, 1, k, nloops);
//             print_result_table(verbose, "krl_IP_dense(s8 )", simd_time, ny, k, simd_test, gt_test);
//         }
//         krl_clean_distance_handle(&kdh_L2_f32);
//         krl_clean_distance_handle(&kdh_L2_f16);
//         krl_clean_distance_handle(&kdh_L2_u8);
//         krl_clean_distance_handle(&kdh_IP_f32);
//         krl_clean_distance_handle(&kdh_IP_f16);
//         krl_clean_distance_handle(&kdh_IP_s8);
//         delete[] gt_test;
//         delete[] simd_test;
//     }
//     delete[] base;
//     delete[] query;
//     delete[] idx;
// }

// int main() {
//     // fashion
//     test_reorder(60000, 784, 10, 1, 65536);
//     // test_reorder(100000, 128, 10, 1, 2);
//     return 0;
// }
#include "tools.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
extern "C" {
#include "krl.h"
}

void normalize(size_t n, float *vec, float mean, float var)
{
    if (n == 0) {
        return;
    }
    double average = 0.0;
    for (size_t i = 0; i < n; ++i) {
        average += vec[i];
    }
    average /= n;
    double variance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        variance += (vec[i] - average) * (vec[i] - average);
    }
    variance /= n;
    double std_dev = std::sqrt(variance);
    double std_var = std::sqrt(var);

    if (std_dev == 0) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            vec[i] = mean;
        }
        return;
    }

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        vec[i] = (vec[i] - average) / std_dev * std_var + mean;
    }
}

void build_database(size_t nx, size_t ny, size_t dim, float *query, float *base, int value_range = 65536,
    float mean = 0.0f, float var = 1.0f)
{
    if (value_range > 0) {
        init_vector(ny * dim, base, 0, value_range);
        init_vector(nx * dim, query, 0, value_range);
    } else {
        init_vector(ny * dim, base, value_range, -value_range);
        init_vector(nx * dim, query, value_range, -value_range);
    }
    normalize(ny * dim, base, mean, var);
    normalize(nx * dim, query, mean, var);
}

template <bool keep_max = false>
double gt_func_test(
    int64_t *I, const float *base, const float *query, const int64_t *ids, size_t d, size_t nb, size_t nq, int nloops)
{
    float *dis = new float[nb * nq];
    memset(dis, 0, sizeof(float) * nb * nq);
    double totaltime = 0;
    struct timespec startTime, endTime;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        memcpy(I, ids, sizeof(int64_t) * nb * nq);
        if constexpr (keep_max) {
            clock_gettime(CLOCK_MONOTONIC, &startTime);
            for (size_t j = 0; j < nq; ++j) {
                for (size_t i = 0; i < nb; ++i) {
                    dis[i + j * nb] = Ipdistance<float, float>(query + j * d, base + ids[i] * d, d);
                }
                std::sort(I + j * nb, I + j * nb + nb, [&](int a, int b) { return dis[a + j * nb] > dis[b + j * nb]; });
            }
            clock_gettime(CLOCK_MONOTONIC, &endTime);
        } else {
            clock_gettime(CLOCK_MONOTONIC, &startTime);
            for (size_t j = 0; j < nq; ++j) {
                for (size_t i = 0; i < nb; ++i) {
                    dis[i + j * nb] = L2distance<float, float>(query + j * d, base + ids[i] * d, d);
                }
                std::sort(I + j * nb, I + j * nb + nb, [&](int a, int b) { return dis[a + j * nb] < dis[b + j * nb]; });
            }
            clock_gettime(CLOCK_MONOTONIC, &endTime);
        }
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    delete[] dis;
    return totaltime / nloops;
}

double simd_sparse_func_test(const KRLDistanceHandle *kdh, int64_t *I, const int64_t *ids, const float *query, size_t d,
    size_t nb, size_t nq, int k, int nloops)
{
    int64_t *input_ids = new int64_t[nb * nq];
    float *dis = new float[nb * nq];
    float *D = new float[k * nq];
    memset(dis, 0, sizeof(float) * nb * nq);
    memset(D, 0, sizeof(float) * k * nq);

    double totaltime = 0;
    struct timespec startTime, endTime;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        memcpy(input_ids, ids, nb * nq * sizeof(int64_t));
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for (size_t j = 0; j < nq; ++j) {
            krl_reorder_2_vector(kdh, nb, dis + j * nb, input_ids + j * nb, query + j * d, k, D + j * k, I + j * k, d);
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    delete[] dis;
    delete[] D;
    delete[] input_ids;
    return totaltime / nloops;
}

double simd_dense_func_test(
    const KRLDistanceHandle *kdh, int64_t *I, const float *query, size_t d, size_t nb, size_t nq, int k, int nloops)
{
    float *D = new float[k * nq];
    memset(D, 0, sizeof(float) * k * nq);

    double totaltime = 0;
    struct timespec startTime, endTime;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for (size_t j = 0; j < nq; ++j) {
            krl_reorder_2_vector_continuous(kdh, nb, 0, query + j * d, k, D + j * k, I + j * k, d);
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    delete[] D;
    return totaltime / nloops;
}

double check_recall(const int64_t *gt, const int64_t *test, size_t nb, size_t nq, size_t k)
{
    int n_k = 0;
    for (int iq = 0; iq < nq; ++iq) {
        for (int ib = 0; ib < k; ++ib) {
            for (int jb = 0; jb < k; ++jb) {
                if (gt[iq * nb + ib] == test[iq * k + jb]) {
                    n_k++;
                    break;
                }
            }
        }
    }
    double recall = 1.0 * n_k / nq / k;
    return recall;
}

void print_result_table(bool verbose, const char *s, double time, size_t ny, size_t k,
    const int64_t *simd_test = nullptr, const int64_t *gt = nullptr)
{
    if (!verbose) {
        return;
    }
    printf("%21s cost: ", s);
    if (ny > 0 && k > 0) {
        print_time(time, verbose);
        double recall = check_recall(gt, simd_test, ny, 1, k);
        printf(", Recall: %.4f\n", recall);
    } else {
        print_time(time, verbose, "\n");
    }
}

void test_reorder(size_t ny, size_t dim, size_t k, size_t nloops, int value_range)
{
    assert(ny >= k);
    float *base = new float[dim * ny];
    float *query = new float[dim];
    int64_t *idx = new int64_t[ny];

    build_database(1, ny, dim, query, base, value_range);

    for (size_t i = 0; i < ny; ++i) {
        idx[i] = i;
    }

    const size_t gt_nloops = std::max((size_t)1, nloops / 10);

    const int nthr = omp_get_max_threads();
#pragma omp parallel num_threads(nthr)
    {
        int64_t *gt_test = new int64_t[ny];
        int64_t *simd_test = new int64_t[k];
        KRLDistanceHandle *kdh_L2_f32 = NULL, *kdh_L2_f16 = NULL, *kdh_L2_u8 = NULL;
        KRLDistanceHandle *kdh_IP_f32 = NULL, *kdh_IP_f16 = NULL, *kdh_IP_s8 = NULL;
        const bool verbose = (omp_get_thread_num() == 0);

        krl_create_reorder_handle(&kdh_L2_f32, 3, 3, ny, dim, 1, (const uint8_t *)base, dim * ny * 4);
        krl_create_reorder_handle(&kdh_L2_f16, 2, 3, ny, dim, 1, (const uint8_t *)base, dim * ny * 4);
        krl_create_reorder_handle(&kdh_L2_u8, 1, 3, ny, dim, 1, (const uint8_t *)base, dim * ny * 4);
        krl_create_reorder_handle(&kdh_IP_f32, 3, 3, ny, dim, 0, (const uint8_t *)base, dim * ny * 4);
        krl_create_reorder_handle(&kdh_IP_f16, 2, 3, ny, dim, 0, (const uint8_t *)base, dim * ny * 4);
        krl_create_reorder_handle(&kdh_IP_s8, 1, 3, ny, dim, 0, (const uint8_t *)base, dim * ny * 4);

        double gt_time, simd_time;
        {
            gt_time = gt_func_test<false>(gt_test, base, query, idx, dim, ny, 1, gt_nloops);
            print_result_table(verbose, "L2_flat_f32", gt_time, 0, 0);
            simd_time = simd_sparse_func_test(kdh_L2_f32, simd_test, idx, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_L2_sparse(f32)", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_sparse_func_test(kdh_L2_f16, simd_test, idx, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_L2_sparse(f16)", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_sparse_func_test(kdh_L2_u8, simd_test, idx, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_L2_sparse(u8 )", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_dense_func_test(kdh_L2_f32, simd_test, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_L2_dense(f32)", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_dense_func_test(kdh_L2_f16, simd_test, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_L2_dense(f16)", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_dense_func_test(kdh_L2_u8, simd_test, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_L2_dense(u8 )", simd_time, ny, k, simd_test, gt_test);
        }
        {
            gt_time = gt_func_test<true>(gt_test, base, query, idx, dim, ny, 1, gt_nloops);
            print_result_table(verbose, "IP_flat_f32", gt_time, 0, 0);
            simd_time = simd_sparse_func_test(kdh_IP_f32, simd_test, idx, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_IP_sparse(f32)", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_sparse_func_test(kdh_IP_f16, simd_test, idx, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_IP_sparse(f16)", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_sparse_func_test(kdh_IP_s8, simd_test, idx, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_IP_sparse(s8 )", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_dense_func_test(kdh_IP_f32, simd_test, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_IP_dense(f32)", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_dense_func_test(kdh_IP_f16, simd_test, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_IP_dense(f16)", simd_time, ny, k, simd_test, gt_test);
            simd_time = simd_dense_func_test(kdh_IP_s8, simd_test, query, dim, ny, 1, k, nloops);
            print_result_table(verbose, "krl_IP_dense(s8 )", simd_time, ny, k, simd_test, gt_test);
        }
        krl_clean_distance_handle(&kdh_L2_f32);
        krl_clean_distance_handle(&kdh_L2_f16);
        krl_clean_distance_handle(&kdh_L2_u8);
        krl_clean_distance_handle(&kdh_IP_f32);
        krl_clean_distance_handle(&kdh_IP_f16);
        krl_clean_distance_handle(&kdh_IP_s8);
        delete[] gt_test;
        delete[] simd_test;
    }
    delete[] base;
    delete[] query;
    delete[] idx;
}

int main()
{
    test_reorder(60000, 784, 10, 1, 65536);
    return 0;
}
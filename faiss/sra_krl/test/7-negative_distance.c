/*
test API list:
1. krl_negative_inner_product_by_idx_f16f32
2. krl_negative_ipdis_by_idx_s8f32
*/
#include "tools.h"
#include <math.h>
#include <vector>
extern "C" {
#include "krl.h"
#include "krl_internal.h"
}

template <typename T = float>
void print_result_table(bool verbose, const char *s, double time, double fops, size_t ny, const T *simd_test = nullptr,
    const T *gt = nullptr, double allowance = 0)
{
    if (!verbose) {
        return;
    }
    printf("%34s cost: ", s);
    print_time(time, verbose);
    if (ny > 0) {
        chechResult(ny, simd_test, gt, allowance, ",", verbose);
    } else {
        printf("\n");
    }
}

template <typename TA, typename TB>
double gt_func_test(TB *dis, const TA *base, const TA *query, const int64_t *ids, size_t d, size_t nb, int nloops,
    TB (*func)(const TA *, const TA *, size_t))
{
    double totaltime = 0;
    struct timespec startTime, endTime;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for (size_t i = 0; i < nb; i++) {
            dis[i] = -func(query, base + ids[i] * d, d);
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return totaltime / nloops;
}

template <typename TA, typename TB>
double simd_func_test_f16(
    TB *dis, const TA *base, const TA *query, const int64_t *ids, size_t d, size_t nb, size_t dis_size, int nloops)
{
    double totaltime = 0;
    struct timespec startTime, endTime;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        krl_negative_inner_product_by_idx_f16f32(dis, query, base, ids, d, nb, dis_size);
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return totaltime / nloops;
}

template <typename TA, typename TB>
double simd_func_test_s8(TB *dis, const TA *base, const TA *query, const int64_t *ids, size_t d, size_t nb, int nloops)
{
    double totaltime = 0;
    struct timespec startTime, endTime;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        krl_negative_inner_product_by_idx_s8f32(dis, query, base, ids, d, nb);
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return totaltime / nloops;
}

void fvec_negative_sparse_distance_test(size_t d, size_t ny, int nloops, size_t chosen_y)
{
    assert(nloops > 0);
    const int k = omp_get_max_threads();

    float *y = new float[d * ny];
    float *x = new float[d];
    float16_t *f16_y = new float16_t[d * ny];
    float16_t *f16_x = new float16_t[d];
    int8_t *s8_y = new int8_t[d * ny];
    int8_t *s8_x = new int8_t[d];

    int64_t *ids = new int64_t[chosen_y];

    init_vector(d * ny, y, 0, 128);
    init_vector(d, x, 0, 128);
    init_vector(chosen_y, ids, (size_t)0, ny);
    for (int i = 0; i < d; ++i) {
        f16_x[i] = (float16_t)x[i];
        s8_x[i] = (int8_t)(x[i] + 0.5);
    }
#pragma omp parallel for num_threads(k)
    for (int i = 0; i < d * ny; ++i) {
        f16_y[i] = (float16_t)y[i];
        s8_y[i] = (int8_t)(y[i] + 0.5);
    }

    const double IP_fops = 2 * chosen_y * d;

#pragma omp parallel for num_threads(k)
    for (int i = 0; i < k; ++i) {
        float *simd_test = new float[chosen_y];
        float *gt = new float[chosen_y];
        memset(simd_test, 0, chosen_y * sizeof(float));
        memset(gt, 0, chosen_y * sizeof(float));
        const bool verbose = (omp_get_thread_num() == 0);

        // IP fp16 dis
        {
            double gt_time = gt_func_test<float16_t>(gt, f16_y, f16_x, ids, d, chosen_y, nloops, Ipdistance);
            print_result_table(verbose, "Ipdistance_f16", gt_time, IP_fops, 0);
            double simd_time = simd_func_test_f16<uint16_t>(
                simd_test, (const uint16_t *)f16_y, (const uint16_t *)f16_x, ids, d, chosen_y, ny, nloops);
            print_result_table(
                verbose, "krl_negative_inner_product_by_idx_f16f32", simd_time, IP_fops, chosen_y, simd_test, gt, 1e-6);
        }

        // IP s8 dis
        {
            double gt_time = gt_func_test<int8_t>(gt, s8_y, s8_x, ids, d, chosen_y, nloops, Ipdistance);
            print_result_table(verbose, "Ipdistance_u8", gt_time, IP_fops, 0);
            double simd_time = simd_func_test_s8<int8_t>(simd_test, s8_y, s8_x, ids, d, chosen_y, nloops);
            print_result_table(
                verbose, "krl_negative_inner_product_by_idx_s8f32 ", simd_time, IP_fops, chosen_y, simd_test, gt, 1e-6);
        }

        delete[] simd_test;
        delete[] gt;
    }

    delete[] x;
    delete[] y;
    delete[] f16_y;
    delete[] f16_x;
    delete[] s8_y;
    delete[] s8_x;
    delete[] ids;
}

int main()
{
    std::vector<size_t> v_dim = {1, 2, 3, 7, 15, 63, 127, 512, 1023, 1024, 0};
    size_t ii = 0;
    size_t dim = v_dim[ii];
    constexpr size_t ny1 = 100000;
    constexpr size_t test_y1 = 8207;
    constexpr size_t test_y2 = 4215;
    while (dim > 0) {
        constexpr size_t nloops = 10;

        printf("dim:%d, ny:%d, chosen_y:%d\n", dim, ny1, test_y2);
        fvec_negative_sparse_distance_test(dim, ny1, nloops, test_y2);

        printf("dim:%d, ny:%d, chosen_y:%d\n", dim, ny1, test_y1);
        fvec_negative_sparse_distance_test(dim, ny1, nloops, test_y1);

        ii++;
        dim = v_dim[ii];
    }
    return 0;
}
/*
test API list:
1. krl_L2sqr_ny
2. krl_L2sqr_ny_with_handle(16,32,64)
3. krl_L2sqr_ny_f16f32
4. krl_L2sqr_ny_u8f32
5. krl_inner_product_ny
6. krl_inner_product_ny_with_handle(16,32,64)
7. krl_inner_product_ny_f16f32
8. krl_inner_product_ny_s8f32
*/

#include "tools.h"
#include <math.h>
#include <vector>
extern "C" {
#include "krl.h"
#include "krl_internal.h"
int sgemv_(const char *trans, long *m, long *n, float *alpha, const float *a, long *lda, const float *x, long *incx,
    float *beta, float *y, long *incy);
}

template <typename T = float>
void print_result_table(bool verbose, const char *s, double time, double fops, size_t ny, const T *simd_test = nullptr,
    const T *gt = nullptr, double allowance = 0)
{
    if (!verbose) {
        return;
    }
    printf("%39s cost: ", s);
    print_time(time, verbose);
    if (ny > 0) {
        chechResult(ny, simd_test, gt, allowance, ",", verbose);
    } else {
        printf("\n");
    }
}

template <typename T>
float vector_square(const T *base, size_t d)
{
    float res = 0;
    for (size_t i = 0; i < d; ++i) {
        res += (float)base[i] * (float)base[i];
    }
    return res;
}

template <typename Tin, typename Tout>
double gt_func_test(Tout *dis, const Tin *base, const Tin *query, size_t d, size_t nb, size_t nq, int nloops,
    Tout (*func)(const Tin *, const Tin *, size_t))
{
    double totaltime = 0;
    struct timespec startTime, endTime;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for (size_t j = 0; j < nq; ++j) {
            const Tin *y2 = base + j * nb * d;
            for (size_t i = 0; i < nb; ++i) {
                dis[i + j * nb] = func(query + j * d, y2 + i * d, d);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return totaltime / nloops;
}

template <bool is_L2_metric>
double sgemv_func_test(float *dis, const float *base, const float *query, const float *base_square, size_t d, size_t nb,
    size_t nq, int nloops)
{
    double totaltime = 0;
    struct timespec startTime, endTime;
    float query_square = 0;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        if constexpr (is_L2_metric) {
            clock_gettime(CLOCK_MONOTONIC, &startTime);
            for (size_t j = 0; j < nq; ++j) {
                query_square = vector_square(query + j * d, d);
                long di = d;
                long nyi = nb;
                float two = 2.0, zero = 0.0;
                long onei = 1;
                sgemv_("T", &di, &nyi, &two, base + j * nb * d, &di, query + j * d, &onei, &zero, dis + j * nb, &onei);
                for (size_t i = 0; i < nb; ++i) {
                    dis[i + j * nb] = query_square + base_square[i + j * nb] - dis[i + j * nb];
                }
            }
            clock_gettime(CLOCK_MONOTONIC, &endTime);
        } else {
            clock_gettime(CLOCK_MONOTONIC, &startTime);
            long di = d;
            long nyi = nb;
            float one = 1.0, zero = 0.0;
            long onei = 1;
            for (size_t j = 0; j < nq; ++j) {
                sgemv_("T", &di, &nyi, &one, base + j * nb * d, &di, query + j * d, &onei, &zero, dis + j * nb, &onei);
            }
            clock_gettime(CLOCK_MONOTONIC, &endTime);
        }
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return totaltime / nloops;
}

template <typename TA, typename TB>
double simd_func_test_new(TB *dis, const TA *base, const TA *query, size_t d, size_t nb, size_t nq, int nloops,
    int (*func)(TB *, const TA *, const TA *, size_t, size_t, size_t))
{
    double totaltime = 0;
    struct timespec startTime, endTime;
    memset(dis, 0, sizeof(TB) * nb * nq);
    for (int nloop = 0; nloop < nloops; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for (size_t j = 0; j < nq; ++j) {
            const TA *y2 = base + j * nb * d;
            func(dis + j * nb, query + j * d, y2, nb, d, nb);
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return totaltime / nloops;
}

template <typename TA, typename TB>
double simd_func_test_old(TB *dis, const TA *base, const TA *query, size_t d, size_t nb, size_t nq, int nloops,
    void (*func)(TB *, const TA *, const TA *, size_t, size_t))
{
    double totaltime = 0;
    struct timespec startTime, endTime;
    memset(dis, 0, sizeof(TB) * nb * nq);
    for (int nloop = 0; nloop < nloops; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for (size_t j = 0; j < nq; ++j) {
            const TA *y2 = base + j * nb * d;
            func(dis + j * nb, query + j * d, y2, nb, d);
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return totaltime / nloops;
}

double handle_func_test(const KRLDistanceHandle *kdh, float *dis, const float *query, size_t dis_size, size_t x_size,
    int nloops, int (*func)(const KRLDistanceHandle *, float *, const float *, size_t, size_t))
{
    double totaltime = 0;
    struct timespec startTime, endTime;
    for (int nloop = 0; nloop < nloops; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        func(kdh, dis, query, dis_size, x_size);
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return totaltime / nloops;
}

void fvec_continuous_distance_test(size_t d, size_t ny, size_t nx, int nloops)
{
    assert(nloops > 0);
    constexpr size_t blocksize = 64;
    const size_t round_ny = ((ny + blocksize - 1) & (-blocksize));
    printf("ny: %d -> %d\n", ny, round_ny);
    float *y = new float[nx * ny * d];
    float *x = new float[nx * d];
    float16_t *f16_y = new float16_t[nx * ny * d];
    float16_t *f16_x = new float16_t[nx * d];
    uint8_t *u8_y = new uint8_t[nx * ny * d];
    uint8_t *u8_x = new uint8_t[nx * d];

    float *square_y = new float[nx * ny];

    KRLDistanceHandle *handle16 = NULL, *handle32 = NULL, *handle64 = NULL;
    KRLDistanceHandle *L2_handle_f16 = NULL, *L2_handle_U8 = NULL;
    KRLDistanceHandle *IP_handle_f16 = NULL, *IP_handle_S8 = NULL;

    init_vector(nx * ny * d, y, 0, 128);
    init_vector(nx * d, x, 0, 128);

    // 计算codes_size
    size_t codes_size_float = nx * ny * d * 4;
    size_t codes_size_f16 = nx * ny * d * 4;
    size_t codes_size_u8 = nx * ny * d * 4;

    krl_create_distance_handle(&handle16, 3, 16, ny, d, nx, 0, (const uint8_t *)y, codes_size_float);
    krl_create_distance_handle(&handle32, 3, 32, ny, d, nx, 0, (const uint8_t *)y, codes_size_float);
    krl_create_distance_handle(&handle64, 3, 64, ny, d, nx, 0, (const uint8_t *)y, codes_size_float);

    krl_create_distance_handle(&L2_handle_f16, 2, 16, ny, d, nx, 1, (const uint8_t *)y, codes_size_f16);
    krl_create_distance_handle(&L2_handle_U8, 1, 16, ny, d, nx, 1, (const uint8_t *)y, codes_size_u8);
    krl_create_distance_handle(&IP_handle_f16, 2, 16, ny, d, nx, 0, (const uint8_t *)y, codes_size_f16);
    krl_create_distance_handle(&IP_handle_S8, 1, 16, ny, d, nx, 0, (const uint8_t *)y, codes_size_u8);

    const int k = omp_get_max_threads();
#pragma omp parallel for num_threads(k)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < d; ++j) {
            f16_x[i * d + j] = (float16_t)(x[i * d + j]);
            u8_x[i * d + j] = (uint8_t)(x[i * d + j] + 0.5);
        }
    }
#pragma omp parallel for num_threads(k)
    for (size_t i = 0; i < nx * ny * d; ++i) {
        f16_y[i] = (float16_t)(y[i]);
        u8_y[i] = (uint8_t)(y[i] + 0.5);
    }
#pragma omp parallel for num_threads(k)
    for (size_t i = 0; i < nx * ny; ++i) {
        square_y[i] = vector_square(y + i * d, d);
    }
    const double L2_fops = 4 * nx * ny * d;  // ABS - * +
    const double IP_fops = 2 * nx * ny * d;  // * +
#pragma omp parallel for num_threads(k)
    for (int i = 0; i < k; ++i) {
        float *simd_test = new float[nx * ny];
        float *gt = new float[nx * ny];
        float *sgemv_test = new float[nx * ny];
        memset(simd_test, 0, nx * ny * sizeof(float));
        memset(gt, 0, nx * ny * sizeof(float));
        memset(sgemv_test, 0, nx * ny * sizeof(float));
        const bool verbose = (omp_get_thread_num() == 0);

        // L2 f32
        {
            double gt_time = gt_func_test(gt, y, x, d, ny, nx, nloops, L2distance);
            print_result_table(verbose, "L2distance_f32", gt_time, L2_fops, 0);
            double simd_time = simd_func_test_new(simd_test, y, x, d, ny, nx, nloops, krl_L2sqr_ny);
            print_result_table(verbose, "L2distance_f32", simd_time, L2_fops, nx * ny, simd_test, gt, 1e-6);

            simd_time = handle_func_test(handle16, simd_test, x, nx * ny, nx * d, nloops, krl_L2sqr_ny_with_handle);
            print_result_table(
                verbose, "krl_L2sqr_ny_with_handle(16)", simd_time, L2_fops, nx * ny, simd_test, gt, 1e-6);
            simd_time = handle_func_test(handle32, simd_test, x, nx * ny, nx * d, nloops, krl_L2sqr_ny_with_handle);
            print_result_table(
                verbose, "krl_L2sqr_ny_with_handle(32)", simd_time, L2_fops, nx * ny, simd_test, gt, 1e-6);
            simd_time = handle_func_test(handle64, simd_test, x, nx * ny, nx * d, nloops, krl_L2sqr_ny_with_handle);
            print_result_table(
                verbose, "krl_L2sqr_ny_with_handle(64)", simd_time, L2_fops, nx * ny, simd_test, gt, 1e-6);

            double sgemv_time = sgemv_func_test<true>(sgemv_test, y, x, square_y, d, ny, nx, nloops);
            print_result_table(verbose, "L2_sgemv", sgemv_time, L2_fops, nx * ny, sgemv_test, gt, 1e-4);
        }
        // L2 f16
        {
            double gt_time = gt_func_test(gt, f16_y, f16_x, d, ny, nx, nloops, L2distance);
            print_result_table(verbose, "L2distance_f16", gt_time, L2_fops, 0);
            double simd_time = simd_func_test_new(
                simd_test, (const uint16_t *)f16_y, (const uint16_t *)f16_x, d, ny, nx, nloops, krl_L2sqr_ny_f16f32);
            print_result_table(verbose, "krl_L2sqr_ny_f16f32", simd_time, L2_fops, nx * ny, simd_test, gt, 1e-6);
            simd_time = handle_func_test(L2_handle_f16, simd_test, x, nx * ny, nx * d, nloops,
            krl_L2sqr_ny_with_handle); print_result_table(verbose, "krl_L2sqr_ny_with_handle(f16)", simd_time,
            L2_fops, nx * ny, simd_test, gt, 1e-6);
        }
        // L2 u8
        {
            double gt_time = gt_func_test(gt, u8_y, u8_x, d, ny, nx, nloops, L2distance);
            print_result_table(verbose, "L2distance_u8", gt_time, L2_fops, 0);
            double simd_time = simd_func_test_new(simd_test, u8_y, u8_x, d, ny, nx, nloops, krl_L2sqr_ny_u8f32);
            print_result_table(verbose, "krl_L2sqr_ny_u8f32", simd_time, L2_fops, nx * ny, simd_test, gt, 1e-6);
            simd_time = handle_func_test(L2_handle_U8, simd_test, x, nx * ny, nx * d, nloops,
            krl_L2sqr_ny_with_handle); print_result_table(verbose, "krl_L2sqr_ny_with_handle(U8)", simd_time,
            L2_fops, nx * ny, simd_test, gt, 1e-6);
            uint32_t *simd_u32 = (uint32_t *)simd_test;
            simd_time = simd_func_test_old((uint32_t *)simd_u32,
                (const uint8_t *)u8_y,
                (const uint8_t *)u8_x,
                d,
                ny,
                nx,
                nloops,
                krl_L2sqr_ny_u8u32);
            for (int i = 0; i < nx * ny; ++i) {
                simd_test[i] = (float)simd_u32[i];
            }
            print_result_table(verbose, "krl_L2sqr_ny_u8u32", simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);
        }
        // IP GT f32
        {
            double gt_time = gt_func_test(gt, y, x, d, ny, nx, nloops, Ipdistance);
            print_result_table(verbose, "Ipdistance_f32", gt_time, IP_fops, 0);
            double simd_time = simd_func_test_new(simd_test, y, x, d, ny, nx, nloops, krl_inner_product_ny);
            print_result_table(verbose, "krl_inner_product_ny", simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);

            simd_time =
                handle_func_test(handle16, simd_test, x, nx * ny, nx * d, nloops, krl_inner_product_ny_with_handle);
            print_result_table(
                verbose, "krl_inner_product_ny_with_handle(16)", simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);
            simd_time =
                handle_func_test(handle32, simd_test, x, nx * ny, nx * d, nloops, krl_inner_product_ny_with_handle);
            print_result_table(
                verbose, "krl_inner_product_ny_with_handle(32)", simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);
            simd_time =
                handle_func_test(handle64, simd_test, x, nx * ny, nx * d, nloops, krl_inner_product_ny_with_handle);
            print_result_table(
                verbose, "krl_inner_product_ny_with_handle(64)", simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);

            double sgemv_time = sgemv_func_test<false>(sgemv_test, y, x, square_y, d, ny, nx, nloops);
            print_result_table(verbose, "ip_sgemv", sgemv_time, IP_fops, nx * ny, sgemv_test, gt, 1e-4);
        }
        // IP f16
        {
            double gt_time = gt_func_test(gt, f16_y, f16_x, d, ny, nx, nloops, Ipdistance);
            print_result_table(verbose, "Ipdistance_f16", gt_time, IP_fops, 0);
            double simd_time = simd_func_test_new(simd_test,
                (const uint16_t *)f16_y,
                (const uint16_t *)f16_x,
                d,
                ny,
                nx,
                nloops,
                krl_inner_product_ny_f16f32);
            print_result_table(
                verbose, "krl_inner_product_ny_f16f32", simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);
            simd_time = handle_func_test(IP_handle_f16, simd_test, x, nx * ny, nx * d, nloops,
            krl_inner_product_ny_with_handle); print_result_table(verbose, "krl_inner_product_ny_with_handle(f16)",
            simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);
        }
        // IP s8
        {
            double gt_time =
                gt_func_test(gt, (const int8_t *)u8_y, (const int8_t *)u8_x, d, ny, nx, nloops, Ipdistance);
            print_result_table(verbose, "Ipdistance_s8", gt_time, IP_fops, 0);
            double simd_time = simd_func_test_new(
                simd_test, (const int8_t *)u8_y, (const int8_t *)u8_x, d, ny, nx, nloops, krl_inner_product_ny_s8f32);
            print_result_table(verbose, "krl_inner_product_ny_s8f32", simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);
            simd_time = handle_func_test(IP_handle_S8, simd_test, x, nx * ny, nx * d, nloops,
            krl_inner_product_ny_with_handle); print_result_table(verbose, "krl_inner_product_ny_with_handle(S8)",
            simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);
            int32_t *simd_s32 = (int32_t *)simd_test;
            simd_time = simd_func_test_old((int32_t *)simd_s32,
                (const int8_t *)u8_y,
                (const int8_t *)u8_x,
                d,
                ny,
                nx,
                nloops,
                krl_inner_product_ny_s8s32);
            for (int i = 0; i < nx * ny; ++i) {
                simd_test[i] = (float)simd_s32[i];
            }
            print_result_table(verbose, "krl_inner_product_ny_s8s32", simd_time, IP_fops, nx * ny, simd_test, gt, 1e-6);
        }
        delete[] gt;
        delete[] simd_test;
        delete[] sgemv_test;
    }
    krl_clean_distance_handle(&handle16);
    krl_clean_distance_handle(&handle32);
    krl_clean_distance_handle(&handle64);

    krl_clean_distance_handle(&L2_handle_f16);
    krl_clean_distance_handle(&L2_handle_U8);
    krl_clean_distance_handle(&IP_handle_f16);
    krl_clean_distance_handle(&IP_handle_S8);

    delete[] square_y;
    delete[] x;
    delete[] y;
    delete[] f16_x;
    delete[] f16_y;
    delete[] u8_x;
    delete[] u8_y;
}

void fvec_continuous_distance_test_f16(size_t d, size_t ny, size_t nx, int nloops)
{
    assert(nloops > 0);
    const int k = omp_get_max_threads();

    float16_t *f16_y = new float16_t[d * ny * nx];
    float16_t *f16_x = new float16_t[d * nx];

    init_vector(d * ny * nx, f16_y, 0, 128);
    init_vector(d * nx, f16_x, 0, 128);

    // L2: 3X; IP: 2X
    const double L2_fops = 4 * nx * ny * d;
    const double IP_fops = 2 * nx * ny * d;
// gt test
#pragma omp parallel for num_threads(k)
    for (int i = 0; i < k; ++i) {
        float16_t *simd_test = new float16_t[nx * ny];
        float16_t *gt = new float16_t[nx * ny];
        // float* gt2 = new float[nx * ny];
        memset(simd_test, 0, nx * ny * sizeof(float16_t));
        memset(gt, 0, nx * ny * sizeof(float16_t));
        const bool verbose = (omp_get_thread_num() == 0);
        // L2 fp16 dis
        {
            // gt_func_test<float16_t, float>(gt2, f16_y, f16_x, d, ny, nx, nloops, L2distance);
            // printf("%.4f\n",gt2[0]);
            double gt_time = gt_func_test<float16_t, float16_t>(gt, f16_y, f16_x, d, ny, nx, nloops, L2distance);
            print_result_table(verbose, "L2distance_f16", gt_time, L2_fops, 0);
            double simd_time = simd_func_test_old<uint16_t, uint16_t>((uint16_t *)simd_test,
                (const uint16_t *)f16_y,
                (const uint16_t *)f16_x,
                d,
                ny,
                nx,
                nloops,
                krl_L2sqr_ny_f16f16);
            print_result_table(verbose, "krl_L2sqr_by_idx_f16f16", simd_time, L2_fops, nx * ny, simd_test, gt, 4e-3);
        }

        // IP fp16 dis
        {
            // gt_func_test<float16_t, float>(gt2, f16_y, f16_x, d, ny, nx, nloops, Ipdistance);
            // printf("%.4f\n",gt2[0]);
            double gt_time = gt_func_test<float16_t, float16_t>(gt, f16_y, f16_x, d, ny, nx, nloops, Ipdistance);
            print_result_table(verbose, "Ipdistance_f16", gt_time, IP_fops, 0);
            double simd_time = simd_func_test_old<uint16_t, uint16_t>((uint16_t*)simd_test, (const uint16_t*)f16_y,
            (const uint16_t*)f16_x, d, ny, nx, nloops, krl_inner_product_ny_f16f16); print_result_table(verbose,
            "krl_inner_product_by_idx_f16f16", simd_time, IP_fops, nx * ny, simd_test, gt, 4e-3);
        }

        delete[] simd_test;
        delete[] gt;
    }
    delete[] f16_y;
    delete[] f16_x;
}

int main()
{
    std::vector<size_t> v_dim = {8, 5, 16, 49, 30, 127, 0, 1, 2, 2, 8, 6, 127, 0, 4, 2, 4, 14, 12, 127, 0};
    std::vector<size_t> v_nx = {12, 20, 8, 16, 32, 15, 0, 96, 50, 64, 98, 160, 15, 0, 24, 50, 32, 56, 80, 15, 0};
    std::vector<size_t> v_ny = {255, 15, 23, 0};
    size_t j = 0;
    size_t ii = 0;
    size_t dim = v_dim[ii];  // 256;
    size_t nx = v_nx[ii];    // 64;
    size_t ny = v_ny[j];

    while (dim >= 0 && ny > 0) {
        if (dim > 0) {
            printf("dim:%d, nx:%d, ny:%d\n", dim, nx, ny);
            fvec_continuous_distance_test(dim, ny, nx, 10);
            fvec_continuous_distance_test_f16(dim, ny, nx, 10);
            ii++;
            dim = v_dim[ii];
            nx = v_nx[ii];
        } else {
            j++;
            ii++;
            ny = v_ny[j];
            dim = v_dim[ii];
            nx = v_nx[ii];
        }
    }
    return 0;
}
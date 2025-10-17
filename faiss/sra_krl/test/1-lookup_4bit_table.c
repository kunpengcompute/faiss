/*
test API list:
1. krl_L2_table_lookup_fast_scan_bs64
2. krl_IP_table_lookup_fast_scan_bs64
3. krl_L2_table_lookup_fast_scan_bs96
4. krl_IP_table_lookup_fast_scan_bs96
5. krl_pack_codes_4b
6. krl_table_lookup_4b_f16
*/

#include "tools.h"
#include "math.h"

extern "C" {
#include "krl.h"
}

#define force_register_countinuity(var) asm("" : "+w"(var))

inline void lookuptable_single(
    int nsq, const uint8_t *codes, const uint8_t *tab, uint16_t *dis, uint16_t threshold, uint32_t *lt_mask)
{
    (*dis) = 0;
    int half_nsq = nsq >> 1;
    for (int q = 0; q < half_nsq; q++) {
        (*dis) += tab[codes[q] & 0x0F];
        tab += 16;
        (*dis) += tab[codes[q] >> 4];
        tab += 16;
    }
}

template <bool keep_max = false>
inline uint32_t get_lt_mask(const uint16_t *dis, uint16_t threshold, int num)
{
    uint32_t lt_mask = 0;
    if constexpr (keep_max) {
        for (int i = 0; i < num; ++i) {
            if (dis[i] > threshold) {
                lt_mask |= ((uint32_t)1) << i;
            }
        }
    } else {
        for (int i = 0; i < num; ++i) {
            if (dis[i] < threshold) {
                lt_mask |= ((uint32_t)1) << i;
            }
        }
    }
    return lt_mask;
}

void print_result_table(bool verbose, const char *s, double time, size_t ny, const uint16_t *simd_distance = nullptr,
    const uint16_t *gt_distance = nullptr, const uint32_t *simd_mask = nullptr, const uint32_t *gt_mask = nullptr)
{
    if (!verbose) {
        return;
    }
    printf("%25s cost: ", s);
    if (ny > 0) {
        size_t n_mask = (ny + 31) / 32;
        print_time(time, verbose);
        size_t lut = chechResult(ny, simd_distance, gt_distance, 0, "", false);
        size_t mask = chechResult(n_mask, simd_mask, gt_mask, 0, "", false);
        if (lut == 0) {
            printf(" lookuptable Test pass!");
        } else {
            printf(" lookuptable Test failed! %zu out of %zu wrong.\n", lut, ny);
        }
        if (mask == 0) {
            printf(" lt_mask Test pass!\n");
        } else {
            printf(" lt_mask Test failed! %zu out of %zu wrong.\n", mask, n_mask);
        }
    } else {
        print_time(time, verbose, "\n");
    }
}

template <int offset, bool keep_max = false>
double u8_lookuptable_test(const uint8_t *codes, const uint8_t *lut, int nb, int nsq, uint16_t *distance, int nloops,
    int (*func)(
        int, const uint8_t *, const uint8_t *, uint16_t *, uint16_t, uint32_t *, size_t, size_t, size_t, size_t))
{
    constexpr uint16_t threshold = keep_max ? 0 : 65535;
    uint32_t lt_mask[4];
    memset(lt_mask, 0, 4 * sizeof(uint32_t));
    memset(distance, 0, nb * sizeof(uint16_t));
    double timeElapsed = 0;
    struct timespec startTime, endTime;

    // 计算size参数
    size_t codes_size = (nsq * offset / 2) * sizeof(uint8_t);
    size_t LUT_size = 16 * nsq * sizeof(uint8_t);
    size_t dis_size = offset * sizeof(uint16_t);
    size_t lt_mask_size = (offset / 32) * sizeof(uint32_t);

    for (int loop = 0; loop < nloops; ++loop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for (int i = 0; i < nb; i += offset) {
            func(nsq,
                codes + nsq * i / 2,
                lut,
                distance + i,
                threshold,
                lt_mask,
                codes_size,
                LUT_size,
                dis_size,
                lt_mask_size);
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        timeElapsed += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return timeElapsed / nloops;
}

template <int offset>
double u8_simd_lt_mask_test(const uint8_t *codes, const uint8_t *lut, int nb, int nsq, uint16_t *distance,
    uint16_t threshold, uint32_t *lt_mask, int nloops,
    int (*func)(
        int, const uint8_t *, const uint8_t *, uint16_t *, uint16_t, uint32_t *, size_t, size_t, size_t, size_t))
{
    memset(lt_mask, 0, (nb / 32) * sizeof(uint32_t));
    memset(distance, 0, nb * sizeof(uint16_t));
    constexpr int mask_offset = offset / 32;
    double timeElapsed = 0;
    struct timespec startTime, endTime;

    // 计算size参数
    size_t codes_size = (nsq * offset / 2) * sizeof(uint8_t);
    size_t LUT_size = 16 * nsq * sizeof(uint8_t);
    size_t dis_size = offset * sizeof(uint16_t);
    size_t lt_mask_size = (offset / 32) * sizeof(uint32_t);

    for (int loop = 0; loop < nloops; ++loop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        uint32_t *mask = lt_mask;
        for (int i = 0; i < nb; i += offset) {
            func(nsq,
                codes + nsq * i / 2,
                lut,
                distance + i,
                threshold,
                mask,
                codes_size,
                LUT_size,
                dis_size,
                lt_mask_size);
            mask += mask_offset;
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        timeElapsed += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return timeElapsed / nloops;
}

template <bool keep_max = false>
double u8_gt_lt_mask_test(int nb, const uint16_t *distance, uint16_t threshold, uint32_t *lt_mask, int nloops)
{
    const int n_mask = nb / 32;
    const int left = nb & 31;
    memset(lt_mask, 0, (n_mask + (left != 0)) * sizeof(uint32_t));
    double timeElapsed = 0;
    struct timespec startTime, endTime;
    for (int loop = 0; loop < nloops; ++loop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for (int i = 0; i < n_mask; i++) {
            lt_mask[i] = get_lt_mask<keep_max>(distance + i * 32, threshold, 32);
        }
        if (left) {
            lt_mask[n_mask] = get_lt_mask<keep_max>(distance + n_mask * 32, threshold, left);
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        timeElapsed += 1e9 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec);
    }
    return timeElapsed / nloops;
}

void lut_test(int dim, int nloop, int nb, uint16_t threshold)
{
    uint8_t *tab = new uint8_t[16 * dim];
    uint8_t *codes = new uint8_t[dim * nb / 2];

    init_vector(16 * dim, tab, 0, 256);
    init_vector(dim * nb / 2, codes, 0, 16);

    uint8_t *packed_codes96 = new uint8_t[dim * nb / 2];
    uint8_t *packed_codes64 = new uint8_t[dim * nb / 2];

    krl_pack_codes_4b(codes, nb, dim, packed_codes96, 96, 1, nb * dim / 2, ceil(nb / 96) * 96 * dim);
    krl_pack_codes_4b(codes, nb, dim, packed_codes64, 64, 1, nb * dim / 2, ceil(nb / 64) * 64 * dim);

    int nn = omp_get_max_threads();
#pragma omp parallel for num_threads(nn)
    for (int i = 0; i < nn; ++i) {
        uint16_t *distance = new uint16_t[nb];
        uint16_t *tmp_dis = new uint16_t[nb];
        uint16_t *gt = new uint16_t[nb];
        uint32_t *lt_mask = new uint32_t[(nb + 31) / 32];
        uint32_t *gt_mask = new uint32_t[(nb + 31) / 32];
        bool verbose = (omp_get_thread_num() == 0);
        double time_gt, time_simd;
        // L2 distance
        {
            time_gt = u8_lookuptable_test<1, false>(codes,
                tab,
                nb,
                dim,
                gt,
                nloop / 10,
                [](int nsq,
                    const uint8_t *codes,
                    const uint8_t *LUT,
                    uint16_t *dis,
                    uint16_t threshold,
                    uint32_t *lt_mask,
                    size_t,
                    size_t,
                    size_t,
                    size_t) {
                    lookuptable_single(nsq, codes, LUT, dis, threshold, lt_mask);
                    return 0;
                });
            u8_gt_lt_mask_test<false>(nb, gt, threshold, gt_mask, 1);
            print_result_table(verbose, "gt_L2_lookuptable_gt", time_gt, 0);

            time_simd = u8_lookuptable_test<64, false>(
                packed_codes64, tab, nb, dim, distance, nloop, krl_L2_table_lookup_fast_scan_bs64);
            u8_simd_lt_mask_test<64>(
                packed_codes64, tab, nb, dim, tmp_dis, threshold, lt_mask, 1, krl_L2_table_lookup_fast_scan_bs64);
            print_result_table(verbose, "krl_L2_lookuptable(64)", time_simd, nb, distance, gt, lt_mask, gt_mask);

            time_simd = u8_lookuptable_test<96, false>(
                packed_codes96, tab, nb, dim, distance, nloop, krl_L2_table_lookup_fast_scan_bs96);
            u8_simd_lt_mask_test<96>(
                packed_codes96, tab, nb, dim, tmp_dis, threshold, lt_mask, 1, krl_L2_table_lookup_fast_scan_bs96);
            print_result_table(verbose, "krl_L2_lookuptable(96)", time_simd, nb, distance, gt, lt_mask, gt_mask);
        }
        // IP distance
        {
            time_gt = u8_lookuptable_test<1, true>(codes,
                tab,
                nb,
                dim,
                gt,
                nloop / 10,
                [](int nsq,
                    const uint8_t *codes,
                    const uint8_t *LUT,
                    uint16_t *dis,
                    uint16_t threshold,
                    uint32_t *lt_mask,
                    size_t,
                    size_t,
                    size_t,
                    size_t) {
                    lookuptable_single(nsq, codes, LUT, dis, threshold, lt_mask);
                    return 0;
                });
            u8_gt_lt_mask_test<true>(nb, gt, threshold, gt_mask, 1);
            print_result_table(verbose, "gt_IP_lookuptable_gt", time_gt, 0);

            time_simd = u8_lookuptable_test<64, true>(
                packed_codes64, tab, nb, dim, distance, nloop, krl_IP_table_lookup_fast_scan_bs64);
            u8_simd_lt_mask_test<64>(
                packed_codes64, tab, nb, dim, tmp_dis, threshold, lt_mask, 1, krl_IP_table_lookup_fast_scan_bs64);
            print_result_table(verbose, "krl_IP_lookuptable(64)", time_simd, nb, distance, gt, lt_mask, gt_mask);

            time_simd = u8_lookuptable_test<96, true>(
                packed_codes96, tab, nb, dim, distance, nloop, krl_IP_table_lookup_fast_scan_bs96);
            u8_simd_lt_mask_test<96>(
                packed_codes96, tab, nb, dim, tmp_dis, threshold, lt_mask, 1, krl_IP_table_lookup_fast_scan_bs96);
            print_result_table(verbose, "krl_IP_lookuptable(96)", time_simd, nb, distance, gt, lt_mask, gt_mask);
        }

        delete[] distance;
        delete[] tmp_dis;
        delete[] gt;
        delete[] lt_mask;
        delete[] gt_mask;
    }

    delete[] tab;
    delete[] codes;
    delete[] packed_codes96;
    delete[] packed_codes64;
}

void lookuptable_4bf16(
    size_t nsq, size_t ncode, const uint8_t *codes, const uint16_t *sim_table, float *result, uint16_t dis0)
{
    size_t half_nsq = nsq >> 1;
    for (size_t i = 0; i < ncode; ++i) {
        const float16_t *sim_table_f16 = (const float16_t *)sim_table;
        float16_t res = *((const float16_t *)&dis0);
        for (size_t q = 0; q < half_nsq; ++q) {
            res += sim_table_f16[codes[q] & 0x0F];
            sim_table_f16 += 16;
            res += sim_table_f16[codes[q] >> 4];
            sim_table_f16 += 16;
        }
        result[i] = (float)res;
        codes += half_nsq;
    }
}

double fp16_func_test(const float16_t *table, const uint8_t *codes, float16_t dis0, size_t d, size_t ncode, float *dis,
    int nloops,
    int (*func)(size_t, size_t, const uint8_t *, const uint16_t *, float *, uint16_t, size_t, size_t, size_t))
{
    double totaltime = 0;
    struct timespec startTime, endTime;

    size_t codes_size = (d / 2) * ncode * sizeof(uint8_t);
    size_t LUT_size = 16 * d * sizeof(uint16_t);
    size_t dis_size = ncode * sizeof(float);
    uint16_t dis0_uint = *reinterpret_cast<const uint16_t *>(&dis0);

    clock_gettime(CLOCK_MONOTONIC, &startTime);
    for (int loop = 0; loop < nloops; ++loop) {
        func(
            d, ncode, codes, reinterpret_cast<const uint16_t *>(table), dis, dis0_uint, codes_size, LUT_size, dis_size);
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    totaltime = (endTime.tv_sec - startTime.tv_sec) + 1e-9 * (endTime.tv_nsec - startTime.tv_nsec);
    return totaltime / nloops * 1000;
}

void lut_test_4b_f16(int dim, int nloops, int nb)
{
    float16_t *tab = new float16_t[16 * dim];
    uint8_t *codes = new uint8_t[dim * nb / 2];
    uint8_t *packed_codes = new uint8_t[dim * nb / 2];
    float16_t dis0 = 123.456;

    init_vector(16 * dim, tab, 0, 256);
    init_vector(dim * nb / 2, codes, 0, 256);

    krl_pack_codes_4b(codes, nb, dim, packed_codes, 64, 0, nb * dim / 2, ceil(nb / 64) * 64 * dim);

    int nn = omp_get_max_threads();
#pragma omp parallel for num_threads(nn)
    for (int i = 0; i < nn; ++i) {
        float *distance = new float[nb];
        float *gt = new float[nb];
        bool verbose = (omp_get_thread_num() == 0);

        auto gt_func = [](size_t nsq,
                           size_t ncode,
                           const uint8_t *codes,
                           const uint16_t *LUT,
                           float *dis,
                           uint16_t dis0,
                           size_t,
                           size_t,
                           size_t) {
            lookuptable_4bf16(nsq, ncode, codes, LUT, dis, dis0);
            return 0;
        };

        fp16_func_test(tab, codes, dis0, dim, nb, gt, 1, gt_func);
        fp16_func_test(tab, packed_codes, dis0, dim, nb, distance, 5, krl_table_lookup_4b_f16);

        double time_gt = fp16_func_test(tab, codes, dis0, dim, nb, gt, nloops / 10, gt_func);
        double time = fp16_func_test(tab, packed_codes, dis0, dim, nb, distance, nloops, krl_table_lookup_4b_f16);

        chechResult(nb, distance, gt, 0, "", verbose);
        if (verbose) {
            printf("Test gt_table_lookup_4b_f16 Cost %.4f ms\n", time_gt);
            printf("Test krl_table_lookup_4b_f16 Cost %.4f ms\n", time);
        }
        delete[] distance;
        delete[] gt;
    }

    delete[] tab;
    delete[] codes;
    delete[] packed_codes;
}

int main()
{
    constexpr int dim = 256;
    constexpr int nloop = 10;
    constexpr int nb = 7680;
    constexpr int nq = 1;
    lut_test(dim, nloop, nb * nq, dim * 64);
    lut_test_4b_f16(dim, nloop, nb * nq);
}
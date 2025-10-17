#include "tools.h"
#include <math.h>
#include <vector>
extern "C" {
#include "krl.h"
}

// gt funcs
static inline void Ipdistance(const uint8_t* base, const uint8_t* query, size_t dim, uint32_t* dis) {
    const size_t d = (dim >> 7);
    memset(dis, 0, sizeof(uint32_t) * 8);
    for (size_t i = 0; i < d; ++i) {
        uint8x16_t neon_base   = vld1q_u8(base + i * 16); // += 16
        uint8x16_t neon_query0 = vld1q_u8(query + i * 16); // += 64
        uint8x16_t neon_query1 = vld1q_u8(query + i * 16 + dim);
        uint8x16_t neon_query2 = vld1q_u8(query + i * 16 + dim * 2);
        uint8x16_t neon_query3 = vld1q_u8(query + i * 16 + dim * 3);
        uint8x16_t neon_query4 = vld1q_u8(query + i * 16 + dim * 4); // += 64
        uint8x16_t neon_query5 = vld1q_u8(query + i * 16 + dim * 5);
        uint8x16_t neon_query6 = vld1q_u8(query + i * 16 + dim * 6);
        uint8x16_t neon_query7 = vld1q_u8(query + i * 16 + dim * 7);
        neon_query0 = vandq_u8(neon_query0, neon_base);
        neon_query1 = vandq_u8(neon_query1, neon_base);
        neon_query2 = vandq_u8(neon_query2, neon_base);
        neon_query3 = vandq_u8(neon_query3, neon_base);
        neon_query4 = vandq_u8(neon_query4, neon_base);
        neon_query5 = vandq_u8(neon_query5, neon_base);
        neon_query6 = vandq_u8(neon_query6, neon_base);
        neon_query7 = vandq_u8(neon_query7, neon_base);
        neon_query0 = vcntq_u8(neon_query0);
        neon_query1 = vcntq_u8(neon_query1);
        neon_query2 = vcntq_u8(neon_query2);
        neon_query3 = vcntq_u8(neon_query3);
        neon_query4 = vcntq_u8(neon_query4);
        neon_query5 = vcntq_u8(neon_query5);
        neon_query6 = vcntq_u8(neon_query6);
        neon_query7 = vcntq_u8(neon_query7);
        dis[0] = dis[0] + vaddvq_u8(neon_query0);
        dis[1] = dis[1] + vaddvq_u8(neon_query1);
        dis[2] = dis[2] + vaddvq_u8(neon_query2);
        dis[3] = dis[3] + vaddvq_u8(neon_query3);
        dis[4] = dis[4] + vaddvq_u8(neon_query4);
        dis[5] = dis[5] + vaddvq_u8(neon_query5);
        dis[6] = dis[6] + vaddvq_u8(neon_query6);
        dis[7] = dis[7] + vaddvq_u8(neon_query7);
    }
}

static inline void L2distance(const uint8_t* base, const uint8_t* query, size_t dim, uint32_t* dis) {
    const size_t d = (dim >> 7);
    memset(dis, 0, sizeof(uint32_t) * 8);
    for (size_t i = 0; i < d; ++i) {
        uint8x16_t neon_base   = vld1q_u8(base + i * 16); // += 16
        uint8x16_t neon_query0 = vld1q_u8(query + i * 16); // += 64
        uint8x16_t neon_query1 = vld1q_u8(query + i * 16 + dim);
        uint8x16_t neon_query2 = vld1q_u8(query + i * 16 + dim * 2);
        uint8x16_t neon_query3 = vld1q_u8(query + i * 16 + dim * 3);
        uint8x16_t neon_query4 = vld1q_u8(query + i * 16 + dim * 4); // += 64
        uint8x16_t neon_query5 = vld1q_u8(query + i * 16 + dim * 5);
        uint8x16_t neon_query6 = vld1q_u8(query + i * 16 + dim * 6);
        uint8x16_t neon_query7 = vld1q_u8(query + i * 16 + dim * 7);
        neon_query0 = veorq_u8(neon_query0, neon_base);
        neon_query1 = veorq_u8(neon_query1, neon_base);
        neon_query2 = veorq_u8(neon_query2, neon_base);
        neon_query3 = veorq_u8(neon_query3, neon_base);
        neon_query4 = veorq_u8(neon_query4, neon_base);
        neon_query5 = veorq_u8(neon_query5, neon_base);
        neon_query6 = veorq_u8(neon_query6, neon_base);
        neon_query7 = veorq_u8(neon_query7, neon_base);
        neon_query0 = vcntq_u8(neon_query0);
        neon_query1 = vcntq_u8(neon_query1);
        neon_query2 = vcntq_u8(neon_query2);
        neon_query3 = vcntq_u8(neon_query3);
        neon_query4 = vcntq_u8(neon_query4);
        neon_query5 = vcntq_u8(neon_query5);
        neon_query6 = vcntq_u8(neon_query6);
        neon_query7 = vcntq_u8(neon_query7);
        dis[0] = dis[0] + vaddvq_u8(neon_query0);
        dis[1] = dis[1] + vaddvq_u8(neon_query1);
        dis[2] = dis[2] + vaddvq_u8(neon_query2);
        dis[3] = dis[3] + vaddvq_u8(neon_query3);
        dis[4] = dis[4] + vaddvq_u8(neon_query4);
        dis[5] = dis[5] + vaddvq_u8(neon_query5);
        dis[6] = dis[6] + vaddvq_u8(neon_query6);
        dis[7] = dis[7] + vaddvq_u8(neon_query7);
    }
}

static inline void prove_distance(const uint8_t* base, const uint8_t* query, size_t dim, uint32_t* dis) {
    const size_t d = (dim >> 7);
    memset(dis, 0, sizeof(uint32_t) * 8);
    for (size_t i = 0; i < d; ++i) {
        uint8x16_t neon_base   = vld1q_u8(base + i * 16); // += 16
        uint8x16_t neon_query0 = vld1q_u8(query + i * 16); // += 64
        uint8x16_t neon_query1 = vld1q_u8(query + i * 16 + dim);
        uint8x16_t neon_query2 = vld1q_u8(query + i * 16 + dim * 2);
        uint8x16_t neon_query3 = vld1q_u8(query + i * 16 + dim * 3);
        uint8x16_t neon_query4 = vld1q_u8(query + i * 16 + dim * 4); // += 64
        uint8x16_t neon_query5 = vld1q_u8(query + i * 16 + dim * 5);
        uint8x16_t neon_query6 = vld1q_u8(query + i * 16 + dim * 6);
        uint8x16_t neon_query7 = vld1q_u8(query + i * 16 + dim * 7);
        dis[0] = dis[0] + vaddvq_u8(neon_query0);
        dis[1] = dis[1] + vaddvq_u8(neon_query1);
        dis[2] = dis[2] + vaddvq_u8(neon_query2);
        dis[3] = dis[3] + vaddvq_u8(neon_query3);
        dis[4] = dis[4] + vaddvq_u8(neon_query4);
        dis[5] = dis[5] + vaddvq_u8(neon_query5);
        dis[6] = dis[6] + vaddvq_u8(neon_query6);
        dis[7] = dis[7] + vaddvq_u8(neon_query7);
    }
}

double distancefunc_test(const uint8_t* query, const uint8_t* base, size_t query_num, size_t d, uint32_t* dis, int nloops, void (*func)(const uint8_t*, const uint8_t*, size_t, uint32_t*)) {
    double totaltime = 0;
    struct timespec startTime, endTime;
    const size_t u8dim = d / 8;
    for(size_t i = 0; i < query_num; i += 8) {
        func(base, query + i * u8dim, d, dis + i);
    }
    for(size_t i = 0; i < query_num; i += 8) {
        func(base, query + i * u8dim, d, dis + i);
    }
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    for(int loop = 0; loop < nloops; ++loop) {
        for(size_t i = 0; i < query_num; i += 8) {
            func(base, query + i * u8dim, d, dis + i);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    totaltime = (endTime.tv_sec - startTime.tv_sec) + 1e-9 * (endTime.tv_nsec - startTime.tv_nsec);
    return totaltime / nloops;
}

void fvec_continuous_distance_test(size_t d, size_t ny, int nloops) {
    assert(nloops > 0);
    assert(d % 128 == 0 && d > 0);
    assert(ny % 8 == 0 && ny > 0);
    uint8_t* u8_x = new uint8_t[d / 8];
    uint8_t* u8_y = new uint8_t[ny * d / 8];

    init_vector(d / 8, u8_x, 0, 256);
    init_vector(ny * d / 8, u8_y, 0, 256);

    const int k = omp_get_max_threads();
    #pragma omp parallel for num_threads(k)
    for(int i = 0; i < k; ++i) {
        uint32_t* simd_test = new uint32_t[ny];
        uint32_t* gt = new uint32_t[ny];

        const bool verbose = (omp_get_thread_num() == 0);
        // gt test 
        double time_ip = distancefunc_test(u8_y, u8_x, ny, d, gt, nloops, Ipdistance);
        if(verbose) {
            printf("time_ip: %lf us\n", time_ip * 1000 * 1000);
        }
        double time_l2 = distancefunc_test(u8_y, u8_x, ny, d, gt, nloops, L2distance);
        if(verbose) {
            printf("time_l2: %lf us\n", time_l2 * 1000 * 1000);
        }
        double time_prove = distancefunc_test(u8_y, u8_x, ny, d, simd_test, nloops, prove_distance);
        if(verbose) {
            printf("time_prove: %lf us\n", time_prove * 1000 * 1000);
        }
        delete[] gt;
        delete[] simd_test;
    }
    delete[] u8_x;
    delete[] u8_y;
}

int main() {
    fvec_continuous_distance_test(256, 512, 1000000);
    return 0;
}
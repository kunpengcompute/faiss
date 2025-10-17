/*
test API list:
1. krl_L2_table_lookup_fast_scan_bs64
2. krl_IP_table_lookup_fast_scan_bs64
3. krl_L2_table_lookup_fast_scan_bs96
4. krl_IP_table_lookup_fast_scan_bs96
5. krl_pack_codes_4b
6. krl_table_lookup_4b_f16

todo list:
7. krl_fast_table_lookup_step
*/

#include "tools.h"

extern "C" {
#include "krl.h"
}

#define force_register_countinuity(var) asm("":"+w"(var))

inline void int4_kernel_accumulate_block_BB3(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* distance){
    uint8x16_t mask0 = vld1q_u8(codes);         // codes:6
    uint8x16_t mask1 = vld1q_u8(codes + 16);
    uint8x16_t mask2 = vld1q_u8(codes + 32);
    uint8x16_t mask3 = vld1q_u8(codes + 48);
    uint8x16_t mask4 = vld1q_u8(codes + 64);
    uint8x16_t mask5 = vld1q_u8(codes + 80);
    codes += 96;
    uint8x16x2_t dictCombine = vld1q_u8_x2(LUT);  // lut:2
    force_register_countinuity(dictCombine.val[0]);
    force_register_countinuity(dictCombine.val[1]);
    LUT += 32;
    uint16x8x4_t accu[3];                       // result:12
    // first loop omit initialization
   {
        accu[0].val[0] = vdupq_n_u16(0);
        accu[0].val[1] = vdupq_n_u16(0);
        accu[0].val[2] = vdupq_n_u16(0);
        accu[0].val[3] = vdupq_n_u16(0);
        accu[1].val[0] = vdupq_n_u16(0);
        accu[1].val[1] = vdupq_n_u16(0);
        accu[1].val[2] = vdupq_n_u16(0);
        accu[1].val[3] = vdupq_n_u16(0);
        accu[2].val[0] = vdupq_n_u16(0);
        accu[2].val[1] = vdupq_n_u16(0);
        accu[2].val[2] = vdupq_n_u16(0);
        accu[2].val[3] = vdupq_n_u16(0);
    }
    // main loop
    for (int sq = 0; sq < nsq - 2; sq += 2) { 
        // __builtin_prefetch(codes + 768, 0, 0);
        uint8x16_t mask0_1, mask1_1, mask2_1, mask3_1, mask4_1, mask5_1;
        uint8x16_t mask0_2, mask1_2, mask2_2, mask3_2, mask4_2, mask5_2;
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask0_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask0)
            : );
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask0_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask0)
            : );
        accu[0].val[0] = vpadalq_u8(accu[0].val[0], mask0_1);
        accu[0].val[1] = vpadalq_u8(accu[0].val[1], mask0_2);
        mask0 = vld1q_u8(codes);
        asm volatile(
            "tbl %[res2].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask2].16B"
            : [res2] "=w"(mask1_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask2] "w"(mask1)
            : );
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask1_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask1)
            : );
        accu[0].val[2] = vpadalq_u8(accu[0].val[2], mask1_1);
        accu[0].val[3] = vpadalq_u8(accu[0].val[3], mask1_2);
        mask1 = vld1q_u8(codes + 16);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask2_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask2)
            :);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask2_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask2)
            : );
        accu[1].val[0] = vpadalq_u8(accu[1].val[0], mask2_1);
        accu[1].val[1] = vpadalq_u8(accu[1].val[1], mask2_2);
        mask2 = vld1q_u8(codes + 32);
        asm volatile(
            "tbl %[res2].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask2].16B"
            : [res2] "=w"(mask3_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask2] "w"(mask3)
            :);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask3_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask3)
            : );
        accu[1].val[2] = vpadalq_u8(accu[1].val[2], mask3_1);
        accu[1].val[3] = vpadalq_u8(accu[1].val[3], mask3_2);
        mask3 = vld1q_u8(codes + 48);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask4_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask4)
            :);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask4_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask4)
            : );
        accu[2].val[0] = vpadalq_u8(accu[2].val[0], mask4_1);
        accu[2].val[1] = vpadalq_u8(accu[2].val[1], mask4_2);
        mask4 = vld1q_u8(codes + 64);
        asm volatile(
            "tbl %[res2].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask2].16B"
            : [res2] "=w"(mask5_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask2] "w"(mask5)
            :);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask5_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask5)
            : );
        accu[2].val[2] = vpadalq_u8(accu[2].val[2], mask5_1);
        accu[2].val[3] = vpadalq_u8(accu[2].val[3], mask5_2);
        mask5 = vld1q_u8(codes + 80);
        codes += 96;
        dictCombine = vld1q_u8_x2(LUT);
        LUT += 32;
    }
    // last loop without preload and prefetch
    {
        uint8x16_t mask0_1, mask1_1, mask2_1, mask3_1, mask4_1, mask5_1;
        uint8x16_t mask0_2, mask1_2, mask2_2, mask3_2, mask4_2, mask5_2;
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask0_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask0)
            : );
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask0_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask0)
            : );
        asm volatile(
            "tbl %[res2].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask2].16B"
            : [res2] "=w"(mask1_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask2] "w"(mask1)
            : );
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask1_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask1)
            : );
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask2_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask2)
            :);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask2_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask2)
            : );
        asm volatile(
            "tbl %[res2].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask2].16B"
            : [res2] "=w"(mask3_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask2] "w"(mask3)
            :);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask3_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask3)
            : );
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask4_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask4)
            :);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask4_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask4)
            : );
        asm volatile(
            "tbl %[res2].16B, { %[dictCombine1].16B - %[dictCombine2].16B }, %[mask2].16B"
            : [res2] "=w"(mask5_1)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask2] "w"(mask5)
            :);
        asm volatile(
            "tbl %[res0].16B, { %[dictCombine2].16B }, %[mask0].16B"
            : [res0] "=w"(mask5_2)
            : [dictCombine1] "w"(dictCombine.val[0]), [dictCombine2] "w"(dictCombine.val[1]), [mask0] "w"(mask5)
            : );

        accu[0].val[0] = vpadalq_u8(accu[0].val[0], mask0_1);
        accu[0].val[1] = vpadalq_u8(accu[0].val[1], mask0_2);
        accu[0].val[2] = vpadalq_u8(accu[0].val[2], mask1_1);
        accu[0].val[3] = vpadalq_u8(accu[0].val[3], mask1_2);
        accu[1].val[0] = vpadalq_u8(accu[1].val[0], mask2_1);
        accu[1].val[1] = vpadalq_u8(accu[1].val[1], mask2_2);
        accu[1].val[2] = vpadalq_u8(accu[1].val[2], mask3_1);
        accu[1].val[3] = vpadalq_u8(accu[1].val[3], mask3_2);
        accu[2].val[0] = vpadalq_u8(accu[2].val[0], mask4_1);
        accu[2].val[1] = vpadalq_u8(accu[2].val[1], mask4_2);
        accu[2].val[2] = vpadalq_u8(accu[2].val[2], mask5_1);
        accu[2].val[3] = vpadalq_u8(accu[2].val[3], mask5_2);
    }
    
    vst1q_u16_x4(distance,accu[0]);
    vst1q_u16_x4(distance + 32,accu[1]);
    vst1q_u16_x4(distance + 64,accu[2]);
}

template<int offset>
double u8_func_test(const uint8_t* codes, const uint8_t* lut, int nb,
                int nsq, uint16_t* distance, int nloops,
                void (*func)(int, const uint8_t*, const uint8_t*, uint16_t*)) {
    memset(distance, 0, nb * sizeof(uint16_t));
    double timeElapsed = 0;
    struct timespec startTime, endTime;
    for(int loop = 0; loop < nloops; ++loop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        for(int i = 0; i < nb; i += offset) {
            func(nsq, codes + nsq * i / 2, lut, distance + i);
        }
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        timeElapsed += (endTime.tv_sec - startTime.tv_sec) + 1e-9 * (endTime.tv_nsec - startTime.tv_nsec);
    }
    return timeElapsed / nloops * 1000.;
}

static inline void prove_distance(const uint8_t* base, const uint8_t* query, size_t dim, uint16_t* dis) {
    const size_t d = 4 * (dim >> 7);
    memset(dis, 0, sizeof(uint16_t) * 8);
    for(size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < d; ++i) {
            uint8x16_t neon_base0  = vld1q_u8(base + i * 16); // += 16
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
}

double distancefunc_test(const uint8_t* query, const uint8_t* base, size_t query_num, size_t d, uint16_t* dis, int nloops, void (*func)(const uint8_t*, const uint8_t*, size_t, uint16_t*)) {
    double totaltime = 0;
    struct timespec startTime, endTime;
    const size_t u8dim = 4 * d / 8;
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
    return 1000. * totaltime / nloops;
}

void lut_test(int dim, int nloop, int nb) {
    uint8_t* tab = new uint8_t[16 * dim];
    uint8_t* codes = new uint8_t[dim * nb / 2];

    init_vector(16 * dim, tab, 0, 255);
    init_vector(dim * nb / 2, codes, 0, 16);

    int nn = omp_get_max_threads();
    #pragma omp parallel for num_threads(nn)
    for(int i = 0; i < nn; ++i) {
        uint16_t* distance = new uint16_t[nb];
        bool verbose = (omp_get_thread_num() == 0);
    
        double time_96_u4 = u8_func_test<96>(codes, tab, nb, dim, distance, nloop, int4_kernel_accumulate_block_BB3);
        if(verbose)
        printf("int4_kernel_accumulate_block_BB3 cost: %.4f ms\n", time_96_u4);

        double time_distance = distancefunc_test(codes, tab, (size_t)nb, (size_t)dim, distance, nloop, prove_distance);
        if(verbose)
        printf("prove_distance cost: %.4f ms\n", time_distance);

        delete[] distance;
    }

    delete[] tab;
    delete[] codes;
}

int main() {
    constexpr int dim = 128;
    constexpr int nloop = 10000;
    constexpr int nb = 7680;
    constexpr int nq = 1;
    lut_test(dim, nloop, nb * nq);
}
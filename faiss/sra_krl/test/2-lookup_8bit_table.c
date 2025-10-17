/*
test API list:
1. krl_table_lookup_8b_f32
2. krl_table_lookup_8b_f32_by_idx
3. krl_table_lookup_8b_f32_with_handle
*/

#include "tools.h"
extern "C" {
#include "krl.h"
}

template<typename TA, typename TR>
inline TR distance_single_code_generic(
        const size_t M,
        const TA* sim_table,
        const uint8_t* code) {
    constexpr size_t ksub = 256;

    const TA* tab = sim_table;
    TR result = 0;

    for (size_t m = 0; m < M; m++) {
        result += tab[*code];
        code++;
        tab += ksub;
    }

    return result;
}

template<typename TA, typename TR>
double GT_test(size_t d, const TA* lut, const uint8_t* codes, size_t ncode, TR* distance, TR dis0, int nloops, size_t* idx = nullptr) {
    double totaltime = 0;
    struct timespec startTime, endTime;
    for(int nloop = 0; nloop < nloops ; ++nloop) {
        if (idx != nullptr) {
            clock_gettime(CLOCK_MONOTONIC, &startTime);
            for(int i = 0; i < ncode; ++i) {
                distance[i] = dis0 + distance_single_code_generic<TA,TR>(d, lut, codes + idx[i] * d);
            }
            clock_gettime(CLOCK_MONOTONIC, &endTime);
        } else {
            clock_gettime(CLOCK_MONOTONIC, &startTime);
            for(int i = 0; i < ncode; ++i) {
                distance[i] = dis0 + distance_single_code_generic<TA,TR>(d, lut, codes + i * d);
            }
            clock_gettime(CLOCK_MONOTONIC, &endTime);
        }
        totaltime += (endTime.tv_sec - startTime.tv_sec) + 1e-9 * (endTime.tv_nsec - startTime.tv_nsec);
    }
    return 1000. * totaltime / nloops;
}

double simd_func_test(KRLLUT8bHandle* klh, size_t nsq, const float* lut, const uint8_t* codes, size_t ncode, float dis0, int nloops, 
                      int (*func)(KRLLUT8bHandle*, size_t, size_t, const uint8_t*, const float*, float, size_t, size_t)) {
    double totaltime = 0;
    struct timespec startTime, endTime;
    
    size_t codes_size = nsq * ncode * sizeof(uint8_t);
    size_t sim_table_size = nsq * 256 * sizeof(float);
    
    for(int nloop = 0; nloop < nloops ; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        func(klh, nsq, ncode, codes, lut, dis0, codes_size, sim_table_size);
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += (endTime.tv_sec - startTime.tv_sec) + 1e-9 * (endTime.tv_nsec - startTime.tv_nsec);
    }
    return 1000. * totaltime / nloops;
}

double direct_func_test(size_t nsq, const float* lut, const uint8_t* codes, size_t ncode, float dis0, int nloops, 
                        int (*func)(size_t, size_t, const uint8_t*, const float*, float*, float, size_t, size_t, size_t)) {
    double totaltime = 0;
    struct timespec startTime, endTime;
    
    size_t codes_size = nsq * ncode * sizeof(uint8_t);
    size_t sim_table_size = nsq * 256 * sizeof(float);
    size_t dis_size = ncode * sizeof(float);
    
    float* distance = new float[ncode];
    
    for(int nloop = 0; nloop < nloops ; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        func(nsq, ncode, codes, lut, distance, dis0, codes_size, sim_table_size, dis_size);
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += (endTime.tv_sec - startTime.tv_sec) + 1e-9 * (endTime.tv_nsec - startTime.tv_nsec);
    }
    
    delete[] distance;
    return 1000. * totaltime / nloops;
}

double idx_func_test(size_t nsq, const float* lut, const uint8_t* codes, size_t ncode, float dis0, const size_t* idx, int nloops, 
                     int (*func)(size_t, size_t, const uint8_t*, const float*, float*, float, const size_t*, size_t, size_t, size_t)) {
    double totaltime = 0;
    struct timespec startTime, endTime;
    
    size_t codes_size = nsq * ncode * sizeof(uint8_t);
    size_t sim_table_size = nsq * 256 * sizeof(float);
    size_t dis_size = ncode * sizeof(float);
    
    float* distance = new float[ncode];
    
    for(int nloop = 0; nloop < nloops ; ++nloop) {
        clock_gettime(CLOCK_MONOTONIC, &startTime);
        func(nsq, ncode, codes, lut, distance, dis0, idx, codes_size, sim_table_size, dis_size);
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        totaltime += (endTime.tv_sec - startTime.tv_sec) + 1e-9 * (endTime.tv_nsec - startTime.tv_nsec);
    }
    
    delete[] distance;
    return 1000. * totaltime / nloops;
}

// test funcs
void lookup_table_8bits_test(size_t d, size_t ny, float dis0, int nloops) {
    float* lut = new float[d * 256];
    uint8_t* codes = new uint8_t[d * ny]; 
    init_vector(d * 256, lut, 0, (int)(64000 / d));
    init_vector(d * ny, codes, 0, 256);
	fflush(stdout);
    int nn = omp_get_max_threads();
    #pragma omp parallel for num_threads(nn)
    for(int i = 0; i < nn; ++i) {
        bool verbose = (omp_get_thread_num() == 0);
        KRLLUT8bHandle* klh_idx = NULL, *klh = NULL;
        krl_create_LUT8b_handle(&klh_idx, 1, ny);
        krl_create_LUT8b_handle(&klh, 0, ny);
		fflush(stdout);
        size_t* idx = krl_get_idx_pointer(klh_idx);
        float* gt = new float[ny];
        size_t ncode = 0;
        for(int i = 0; i < ny / 2; ++i) {
            idx[i] = (i * 2);
            ncode++;
        }
        
        double gt_idx_time = GT_test(d, lut, codes, ncode, gt, dis0, nloops, idx);
        if(verbose)
            printf("gt with idx cost: %.4f ms\n", gt_idx_time);
            
        double simd_idx_time = idx_func_test(d, lut, codes, ncode, dis0, idx, nloops, krl_table_lookup_8b_f32_by_idx);
        if(verbose)
            printf("scan_list_with_table_by_idx_test cost: %.4f ms\n", simd_idx_time);
            
        float* simd_res = new float[ncode];
        krl_table_lookup_8b_f32_by_idx(d, ncode, codes, lut, simd_res, dis0, idx, 
                                      d * ncode * sizeof(uint8_t),
                                      d * 256 * sizeof(float),
                                      ncode * sizeof(float));
        chechResult(ncode, simd_res, gt, 1e-6, "scan_list_with_table_by_idx_test,", verbose);
        delete[] simd_res;
        
        double handle_idx_time = simd_func_test(klh_idx, d, lut, codes, ncode, dis0, nloops, krl_table_lookup_8b_f32_with_handle);
        if(verbose)
            printf("handle with idx cost: %.4f ms\n", handle_idx_time);
            
        float* handle_res = krl_get_dist_pointer(klh_idx);
        chechResult(ncode, handle_res, gt, 1e-6, "handle_with_table_by_idx_test,", verbose);

        double gt_time = GT_test(d, lut, codes, ny, gt, dis0, nloops);
        if(verbose)
            printf("gt cost: %.4f ms\n", gt_time);
            
        double direct_time = direct_func_test(d, lut, codes, ny, dis0, nloops, krl_table_lookup_8b_f32);
        if(verbose)
            printf("direct scan cost: %.4f ms\n", direct_time);
            
        float* direct_res = new float[ny];
        krl_table_lookup_8b_f32(d, ny, codes, lut, direct_res, dis0, 
                               d * ny * sizeof(uint8_t),
                               d * 256 * sizeof(float),
                               ny * sizeof(float));
        chechResult(ny, direct_res, gt, 1e-6, "direct_scan_test,", verbose);
        delete[] direct_res;
        
        double handle_time = simd_func_test(klh, d, lut, codes, ny, dis0, nloops, krl_table_lookup_8b_f32_with_handle);
        if(verbose)
            printf("handle scan cost: %.4f ms\n", handle_time);
            
        float* handle_direct_res = krl_get_dist_pointer(klh);
        chechResult(ny, handle_direct_res, gt, 1e-6, "handle_scan_test,", verbose);
        
        krl_clean_LUT8b_handle(&klh_idx);
        krl_clean_LUT8b_handle(&klh);
        delete[] gt;
    }
    delete[] lut;
    delete[] codes;
}

int main() {
    printf("\nbegin test lookup table 8bits!\n");
    std::vector<size_t> v_dim = {256,64,20,16,8,4,0};
    constexpr size_t ny = 8191;
    int i = 0;
    size_t dim = v_dim[i];
    while(dim >= 2){
        size_t nloops = 10;
        printf("dim:%d, ny:%d, nloops:%d\n", dim, ny, nloops);
        lookup_table_8bits_test(dim, ny, 3.1415, nloops);
        dim = v_dim[++i];
    }
    return 0;
}
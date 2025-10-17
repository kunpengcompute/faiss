#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <stdio.h>
#include <time.h>
#include <cstring>
#include <vector>
#include <cstdint>
#include <iostream>
#include <omp.h>

template<typename T, typename TA>
void init_vector(int n, T* x, TA rmin, TA rmax) {
    TA range = rmax - rmin;
    for(int i = 0; i < n;++i) {
        x[i] = (rand() % range) + rmin;
    }
}

template<typename T>
void normalized_vector(int n, T* x) {
    T sum = 0;
    for(int i = 0;i < n; i++) {
        sum += x[i];
    }
    T average = sum / n;
    for(int i = 0; i < n; i++) {
        x[i] /= average;
    }
}

template<typename T,typename TB>
int chechResult(const int nb, const T* result, const TB* gt, float tolerances = 0.0F, const char* s = nullptr, bool verbose = true, int max_print_num = 128) {
    if(s && verbose) {
        printf("%s ",s);
    }
    bool all_true = true;
    int i = 0;
    for(; i < nb; ++i) {
        if(std::abs((float)(result[i]) - (float)(gt[i])) > std::abs((float)gt[i]) * tolerances) {
            all_true = false;
            break;
        }
    }
    if(all_true) {
        if(verbose) {
            printf("Test pass!\n");
        }
        return 0;
    }
    int k = 0;
    if(verbose) {
        printf("Error lists:\n%3s %8s %8s %8s\n","id","gt","res","diff");
    }
    for(; i < nb; ++i) {
        if(std::abs((float)(result[i]) - (float)(gt[i])) > std::abs((float)gt[i]) * tolerances) {
            if(k < max_print_num) {
                const float diff = (float)(result[i]) - (float)(gt[i]);
                if(verbose) {
                    printf("%3d %8.4f %8.4f %8.4f\n", i + 1, (float)gt[i], (float)result[i], diff);
                }
            }
            k++;
        }
    }
    if(k > max_print_num && verbose) {
        printf("...\n");
    }
    if(verbose) {
        printf("total check number: %d\n", nb);
        printf("total error number: %d\n", k);
    }
    return k;
}

void print_time(double time, bool verbose, const char* s = nullptr) {
    if(!verbose) {
        return;
    }
    if (time > 999.9) {
        time /= 1000.0;
        if (time > 999.9) {
            time /= 1000.0;
            if (time > 999.9) {
                time /= 1000.0;
                printf("%6.2f s ", time);
            } else {
                 printf("%6.2f ms", time);
            }
        } else {
            printf("%6.2f us", time);
        }
    } else {
        printf("%6.2f ns", time);
    }
    if(s) {
        printf("%s", s);
    }
}

template<typename Tin, typename Tout>
Tout Ipdistance(const Tin* base, const Tin* query, size_t d) {
    Tout res = 0;
    for (size_t i = 0; i < d; ++i) {
        res += (Tout)base[i] * (Tout)query[i];
    }
    return res;
}

template<typename Tin, typename Tout>
Tout L2distance(const Tin* base, const Tin* query, size_t d) {
    Tout res = 0;
    for (size_t i = 0; i < d; i++) {
        const Tout tmp = (Tout)base[i] - (Tout)query[i];
        res += tmp * tmp;
    }
    return res;
}
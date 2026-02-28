/*
   Copyright 2025 Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#include "krl_internal.h"
#include "platform_macros.h"
#include <stdio.h>

/* ----------------------------------------  store helper  ---------------------------------------- */

/*
 * @brief Store a single element to file.
 * @param f File pointer.
 * @param bytes Size of the element in bytes.
 * @param element Pointer to the element to store.
 * @return size_t Number of elements stored (always 1).
 */
static inline size_t store_element(FILE *f, size_t bytes, const uint8_t *element)
{
    return fwrite(element, bytes, 1, f);
}

/*
 * @brief Store a list of elements to file.
 * @param f File pointer.
 * @param n Number of elements to store.
 * @param bytes Size of each element in bytes.
 * @param vec Pointer to the list of elements.
 * @return size_t Number of elements stored.
 */
static inline size_t store_list(FILE *f, size_t n, size_t bytes, const uint8_t *vec)
{
    return fwrite(vec, bytes, n, f);
}

/* ----------------------------------------  read  helper  ---------------------------------------- */

/*
 * @brief Read a single element from file.
 * @param f File pointer.
 * @param bytes Size of the element in bytes.
 * @param element Pointer to store the read element.
 */
static inline size_t read_element(FILE *f, size_t bytes, uint8_t *element)
{
    return fread(element, bytes, 1, f);
}

/*
 * @brief Read a list of elements from file.
 * @param f File pointer.
 * @param n Number of elements to read.
 * @param bytes Size of each element in bytes.
 * @param vec Pointer to store the read elements.
 */
static inline size_t read_list(FILE *f, size_t n, size_t bytes, uint8_t *vec)
{
    if (n < 1 && n >= ((uint64_t)(1) << 40)) {
        return 0;
    }
    return fread(vec, bytes, n, f);
}

/* ----------------------------------------  store handle  ---------------------------------------- */

/*
 * @brief Store distance handle to file.
 * @param f File pointer.
 * @param kdh Pointer to the distance handle.
 */
int krl_store_distanceHandle(FILE *f, const KRLDistanceHandle *kdh)
{
    if (f == NULL || kdh == NULL) {
        printf("Error: INVALPOINTER in krl_store_distanceHandle\n");
        return INVALPOINTER;
    }
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->data_bits)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->full_data_bits)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->M)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->blocksize)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->d)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->ny)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->ceil_ny)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(float), (const uint8_t *)&(kdh->quanted_scale)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(float), (const uint8_t *)&(kdh->quanted_bias)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(int), (const uint8_t *)&(kdh->metric_type)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->quanted_bytes)) != 1)
        return FAILIO;
    if (store_element(f, sizeof(size_t), (const uint8_t *)&(kdh->transposed_bytes)) != 1)
        return FAILIO;
    if (kdh->quanted_bytes > 0) {
        if (store_list(f, kdh->quanted_bytes, sizeof(uint8_t), (const uint8_t *)kdh->quanted_codes) !=
            kdh->quanted_bytes)
            return FAILIO;
    }
    if (kdh->transposed_bytes > 0) {
        if (store_list(f, kdh->transposed_bytes, sizeof(uint8_t), (const uint8_t *)kdh->transposed_codes) !=
            kdh->transposed_bytes)
            return FAILIO;
    }
    return SUCCESS;
}

/* ----------------------------------------  read handle  ---------------------------------------- */

/*
 * @brief Build distance handle from file.
 * @param f File pointer.
 * @param kdh Pointer to the distance handle pointer.
 * @return int Status code (SUCCESS, INVALPOINTER, FAILALLOC).
 */
int krl_build_distanceHandle_fromfile(FILE *f, KRLDistanceHandle **kdh)
{
    if (f == NULL || kdh == NULL) {
        printf("Error: INVALPOINTER in krl_build_distanceHandle_fromfile\n");
        return INVALPOINTER;
    }
    (*kdh) = (KRLDistanceHandle *)malloc(sizeof(KRLDistanceHandle));
    if ((*kdh) == NULL) {
        printf("Error: FAILALLOC in krl_build_distanceHandle_fromfile\n");
        return FAILALLOC;
    }
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->data_bits)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->full_data_bits)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->M)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->blocksize)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->d)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->ny)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->ceil_ny)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(float), (uint8_t *)&((*kdh)->quanted_scale)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(float), (uint8_t *)&((*kdh)->quanted_bias)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(int), (uint8_t *)&((*kdh)->metric_type)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->quanted_bytes)) != 1)
        goto FAIL;
    if (read_element(f, sizeof(size_t), (uint8_t *)&((*kdh)->transposed_bytes)) != 1)
        goto FAIL;
    if ((*kdh)->quanted_bytes > 0) {
        (*kdh)->quanted_codes = (uint8_t *)aligned_alloc(KRL_DEFAULT_ALIGNED, (*kdh)->quanted_bytes);
        if ((*kdh)->quanted_codes == NULL) {
            krl_clean_distance_handle(kdh);
            printf("Error: FAILALLOC in krl_build_distanceHandle_fromfile\n");
            return FAILALLOC;
        }
        if (read_list(f, (*kdh)->quanted_bytes, sizeof(uint8_t), (uint8_t *)(*kdh)->quanted_codes) !=
            (*kdh)->quanted_bytes)
            goto FAIL;
    } else {
        (*kdh)->quanted_codes = NULL;
    }
    if ((*kdh)->transposed_bytes > 0) {
        (*kdh)->transposed_codes = (float *)aligned_alloc(KRL_DEFAULT_ALIGNED, (*kdh)->transposed_bytes);
        if ((*kdh)->transposed_codes == NULL) {
            krl_clean_distance_handle(kdh);
            printf("Error: FAILALLOC in krl_build_distanceHandle_fromfile\n");
            return FAILALLOC;
        }
        if (read_list(f, (*kdh)->transposed_bytes, sizeof(uint8_t), (uint8_t *)(*kdh)->transposed_codes) !=
            (*kdh)->transposed_bytes)
            goto FAIL;
    } else {
        (*kdh)->transposed_codes = NULL;
    }
    return SUCCESS;

FAIL:
    krl_clean_distance_handle(kdh);
    printf("Error: FAILIO in krl_build_distanceHandle_fromfile\n");
    return FAILIO;
}

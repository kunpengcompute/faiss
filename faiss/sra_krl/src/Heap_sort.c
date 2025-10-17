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
#include "krl.h"
#include "platform_macros.h"

/* -------------------------------------- reorder 1 heaps -------------------------------------- */
/*
 * @brief Compare the value of two elements
 * @param a The first element.
 * @param b The second element.
 * @param asce If the first element is larger, return (a > b); otherwise, return (a < b).
 */
template <int asce = 0>
inline static int data_compare(float a, float b)
{
    if constexpr (asce == 0) {
        return (a < b);
    } else {
        return (a > b);
    }
}

/*
 * @brief Replace the top element of the heap and update the heap.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param val New value to replace the top element.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_heap_replace_top(idx_t k, float *bh_val, float val)
{
    /* Use 1-based indexing for easier node->child translation */
    bh_val--;
    idx_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        if (i1 > k) {
            break;
        }
        if ((i1 == k) || data_compare<asce>(bh_val[i1], bh_val[i1 + 1])) {
            if (data_compare<asce>(bh_val[i1], val)) {
                bh_val[i] = bh_val[i1];
                i = i1;
            } else {
                break;
            }
        } else {
            i2 = i1 + 1;
            if (data_compare<asce>(bh_val[i2], val)) {
                bh_val[i] = bh_val[i2];
                i = i2;
            } else {
                break;
            }
        }
    }
    bh_val[i] = val;
}

/*
 * @brief Push a value into a descending heap.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param val Value to be pushed into the heap.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_heap_push(idx_t k, float *bh_val, float val)
{
    /* Use 1-based indexing for easier node->child translation */
    bh_val--;
    idx_t i = k, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (data_compare<asce>(val, bh_val[i_father])) {
            bh_val[i] = bh_val[i_father];
            i = i_father;
        } else {
            /* the heap structure is ok */
            break;
        }
    }
    bh_val[i] = val;
}

/*
 * @brief Heapify an array into a descending heap.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param x Pointer to the input array.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_heap_heapify(idx_t k, float *bh_val, const float *x)
{
    for (idx_t i = 0; i < k; ++i) {
        krl_heap_push<asce>(i + 1, bh_val, x[i]);
    }
}

/*
 * @brief Add multiple values to a descending heap and replace the top if necessary.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param x Pointer to the input array.
 * @param n Number of elements to add.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_heap_addn(idx_t k, float *bh_val, const float *x, idx_t n)
{
    for (idx_t i = 0; i < n; i++) {
        /* New data is larger, replace old data and update the heap */
        if (data_compare<asce>(bh_val[0], x[i])) {
            krl_heap_replace_top<asce>(k, bh_val, x[i]);
        }
    }
}

/*
 * @brief Reorder elements in a descending heap by moving the smallest element to the end.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_heap_reorder(idx_t k, float *bh_val)
{
    idx_t i = 0, ii = 0;
    for (; i < k - 1; i++) {
        /* top element should be put at the end of the list */
        float val = bh_val[0];

        /* boundary case: we will over-ride this value if not a true element */
        krl_heap_replace_top<asce>(k - i - 1, bh_val, bh_val[k - i - 1]);
        bh_val[k - i - 1] = val;
    }
}

/*
 * @brief Obtain top-k elements using a descending heap for inner product.
 * @param k Number of top elements to obtain.
 * @param distances Pointer to the distance array.
 * @param k_base Base size of the distance array.
 * @param base_distances Pointer to the base distance array.
 */
void krl_obtion_topk_heap_desc(idx_t k, float *distances, idx_t k_base, const float *base_distances)
{
    /* Build a descending heap */
    krl_heap_heapify<0>(k, distances, base_distances);
    /* Add remaining elements to the heap */
    if (k_base != k) {
        krl_heap_addn<0>(k, distances, base_distances + k, k_base - k);
    }
}

/*
 * @brief Obtain top-k elements using an ascending heap for L2 distance.
 * @param k Number of top elements to obtain.
 * @param distances Pointer to the distance array.
 * @param k_base Base size of the distance array.
 * @param base_distances Pointer to the base distance array.
 */
void krl_obtion_topk_heap_asce(idx_t k, float *distances, idx_t k_base, const float *base_distances)
{
    /* Build an ascending heap */
    krl_heap_heapify<1>(k, distances, base_distances);
    /* Add remaining elements to the heap */
    if (k_base != k) {
        krl_heap_addn<1>(k, distances, base_distances + k, k_base - k);
    }
}

/* -------------------------------------- reorder 2 heaps -------------------------------------- */

/*
 * @brief Push a value and its corresponding ID into a descending heap.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param bh_ids Pointer to the ID array.
 * @param val Value to be pushed into the heap.
 * @param id ID corresponding to the value.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_2heaps_push(idx_t k, float *bh_val, idx_t *bh_ids, float val, idx_t id)
{
    /* Use 1-based indexing for easier node->child translation */
    bh_val--;
    bh_ids--;
    idx_t i = k, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (data_compare<asce>(val, bh_val[i_father])) {
            bh_val[i] = bh_val[i_father];
            bh_ids[i] = bh_ids[i_father];
            i = i_father;
        } else {
            /* the heap structure is ok */
            break;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = id;
}

/*
 * @brief Heapify arrays into a descending heap with corresponding IDs.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param bh_ids Pointer to the ID array.
 * @param x Pointer to the input array.
 * @param ids Pointer to the input ID array.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_2heaps_heapify(idx_t k, float *bh_val, idx_t *bh_ids, const float *x, const idx_t *ids)
{
    for (idx_t i = 0; i < k; ++i) {
        krl_2heaps_push<asce>(i + 1, bh_val, bh_ids, x[i], ids[i]);
    }
}

/*
 * @brief Replace the top element of a descending heap and update the heap with corresponding IDs.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param bh_ids Pointer to the ID array.
 * @param val New value to replace the top element.
 * @param id New ID corresponding to the value.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_2heaps_replace_top(idx_t k, float *bh_val, idx_t *bh_ids, float val, idx_t id)
{
    /* Use 1-based indexing for easier node->child translation */
    bh_val--;
    bh_ids--;
    idx_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        if (i1 > k) {
            break;
        }
        if ((i1 == k) || data_compare<asce>(bh_val[i1], bh_val[i1 + 1])) {
            i2 = i1;
        } else {
            i2 = i1 + 1;
        }
        if (data_compare<asce>(bh_val[i2], val)) {
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        } else {
            break;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = id;
}

/*
 * @brief Add multiple values and their corresponding IDs to a descending heap and replace the top if necessary.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param bh_ids Pointer to the ID array.
 * @param x Pointer to the input array.
 * @param ids Pointer to the input ID array.
 * @param n Number of elements to add.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_2heaps_addn(idx_t k, float *bh_val, idx_t *bh_ids, const float *x, const idx_t *ids, idx_t n)
{
    for (idx_t i = 0; i < n; i++) {
        /* New data is larger, replace old data and update the heap */
        if (data_compare<asce>(bh_val[0], x[i])) {
            krl_2heaps_replace_top<asce>(k, bh_val, bh_ids, x[i], ids[i]);
        }
    }
}

/*
 * @brief Reorder elements in a descending heap by moving the smallest element to the end with corresponding IDs.
 * @param k Size of the heap.
 * @param bh_val Pointer to the heap array.
 * @param bh_ids Pointer to the ID array.
 * @param asce whether a ascending heap.
 */
template <int asce = 0>
inline static void krl_2heaps_reorder(idx_t k, float *bh_val, idx_t *bh_ids)
{
    idx_t i = 0, ii = 0;
    for (; i < k - 1; i++) {
        /* top element should be put at the end of the list */
        float val = bh_val[0];
        idx_t id = bh_ids[0];

        /* boundary case: we will over-ride this value if not a true element */
        krl_2heaps_replace_top<asce>(k - i - 1, bh_val, bh_ids, bh_val[k - i - 1], bh_ids[k - i - 1]);
        bh_val[k - i - 1] = val;
        bh_ids[k - i - 1] = id;
    }
}

/*
 * @brief Reorder elements using two heaps for inner product.
 * @param k Number of top elements to obtain.
 * @param labels Pointer to the label array.
 * @param distances Pointer to the distance array.
 * @param k_base Base size of the distance array.
 * @param base_labels Pointer to the base label array.
 * @param base_distances Pointer to the base distance array.
 */
void krl_reorder_2_heaps_desc(
    idx_t k, idx_t *labels, float *distances, idx_t k_base, const idx_t *base_labels, const float *base_distances)
{
    /* Build a descending heap */
    krl_2heaps_heapify<0>(k, distances, labels, base_distances, base_labels);
    /* Add remaining elements to the heap */
    if (k_base != k) {
        krl_2heaps_addn<0>(k, distances, labels, base_distances + k, base_labels + k, k_base - k);
    }
    /* Move the smallest element to the end */
    krl_2heaps_reorder<0>(k, distances, labels);
}

/*
 * @brief Reorder elements using two heaps for L2 distance.
 * @param k Number of top elements to obtain.
 * @param labels Pointer to the label array.
 * @param distances Pointer to the distance array.
 * @param k_base Base size of the distance array.
 * @param base_labels Pointer to the base label array.
 * @param base_distances Pointer to the base distance array.
 */
void krl_reorder_2_heaps_asce(
    idx_t k, idx_t *labels, float *distances, idx_t k_base, const idx_t *base_labels, const float *base_distances)
{
    /* Build an ascending heap */
    krl_2heaps_heapify<1>(k, distances, labels, base_distances, base_labels);
    /* Add remaining elements to the heap */
    if (k_base != k) {
        krl_2heaps_addn<1>(k, distances, labels, base_distances + k, base_labels + k, k_base - k);
    }
    /* Move the largest element to the end */
    krl_2heaps_reorder<1>(k, distances, labels);
}

/* -------------------------------------- quick sort 2 heaps -------------------------------------- */

/*
 * @brief Swap two elements and their corresponding IDs.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @param a2 Pointer to the first ID.
 * @param b2 Pointer to the second ID.
 */
inline static void swap_2_vector(float *a, float *b, idx_t *a2, idx_t *b2)
{
    float c = *a;
    idx_t c2 = *a2;
    *a = *b;
    *a2 = *b2;
    *b = c;
    *b2 = c2;
}

/*
 * @brief Adapt and reorder elements in ascending order.
 * @param dis Pointer to the distance array.
 * @param label Pointer to the label array.
 * @param n Number of elements.
 * @param target Target value for reordering.
 * @return idx_t Index of the target value.
 */
idx_t Adapt_reorder_asce(float *dis, idx_t *label, idx_t n, float target)
{
    idx_t left = 0, right = n - 1;
    while (left < right) {
        while (dis[left] <= target && left < right) {
            left++;
        }
        while (dis[right] > target && left < right) {
            right--;
        }
        if (left < right) {
            swap_2_vector(&dis[left], &dis[right], &label[left], &label[right]);
            left++;
            right--;
        }
    }
    return dis[left] > target ? left : left + 1;
}

/*
 * @brief Adapt and reorder elements in descending order.
 * @param dis Pointer to the distance array.
 * @param label Pointer to the label array.
 * @param n Number of elements.
 * @param target Target value for reordering.
 * @return idx_t Index of the target value.
 */
idx_t Adapt_reorder_desc(float *dis, idx_t *label, idx_t n, float target)
{
    idx_t left = 0, right = n - 1;
    while (left < right) {
        while (dis[left] >= target && left < right) {
            left++;
        }
        while (dis[right] < target && left < right) {
            right--;
        }
        if (left < right) {
            swap_2_vector(&dis[left], &dis[right], &label[left], &label[right]);
            left++;
            right--;
        }
    }
    return dis[left] < target ? left : left + 1;
}
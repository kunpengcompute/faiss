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

#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"
#include "safe_memory.h"
#include <stdio.h>

/*
 * @brief Matrix block transpose kernel function for 4x4 block optimization.
 * @param src Pointer to the input matrix (uint32_t).
 * @param dim Dimension of the matrix.
 * @param blocksize Block size for transpose operations.
 * @param block Pointer to the output block matrix (uint32_t).
 */
static void krl_matrix_block_transpose_kernel(const uint32_t *src, size_t dim, size_t blocksize, uint32_t *block)
{
    uint32x4_t matrix[16];
    uint64x2_t tmp[4];
    for (size_t i = 0; i < blocksize; i += 16) {
        for (size_t j = 0; j < 16; j += 4) {
            matrix[j] = vld1q_u32(src + (i + j) * dim);
            matrix[j + 1] = vld1q_u32(src + (i + j + 1) * dim);
            matrix[j + 2] = vld1q_u32(src + (i + j + 2) * dim);
            matrix[j + 3] = vld1q_u32(src + (i + j + 3) * dim);
            tmp[0] = vreinterpretq_u64_u32(vtrn1q_u32(matrix[j], matrix[j + 1]));
            tmp[1] = vreinterpretq_u64_u32(vtrn2q_u32(matrix[j], matrix[j + 1]));
            tmp[2] = vreinterpretq_u64_u32(vtrn1q_u32(matrix[j + 2], matrix[j + 3]));
            tmp[3] = vreinterpretq_u64_u32(vtrn2q_u32(matrix[j + 2], matrix[j + 3]));
            matrix[j] = vreinterpretq_u32_u64(vtrn1q_u64(tmp[0], tmp[2]));
            matrix[j + 1] = vreinterpretq_u32_u64(vtrn1q_u64(tmp[1], tmp[3]));
            matrix[j + 2] = vreinterpretq_u32_u64(vtrn2q_u64(tmp[0], tmp[2]));
            matrix[j + 3] = vreinterpretq_u32_u64(vtrn2q_u64(tmp[1], tmp[3]));
        }
        vst1q_u32(block + i, matrix[0]);
        vst1q_u32(block + i + 4, matrix[4]);
        vst1q_u32(block + i + 8, matrix[8]);
        vst1q_u32(block + i + 12, matrix[12]);
        vst1q_u32(block + i + blocksize, matrix[1]);
        vst1q_u32(block + i + blocksize + 4, matrix[5]);
        vst1q_u32(block + i + blocksize + 8, matrix[9]);
        vst1q_u32(block + i + blocksize + 12, matrix[13]);
        vst1q_u32(block + i + 2 * blocksize, matrix[2]);
        vst1q_u32(block + i + 2 * blocksize + 4, matrix[6]);
        vst1q_u32(block + i + 2 * blocksize + 8, matrix[10]);
        vst1q_u32(block + i + 2 * blocksize + 12, matrix[14]);
        vst1q_u32(block + i + 3 * blocksize, matrix[3]);
        vst1q_u32(block + i + 3 * blocksize + 4, matrix[7]);
        vst1q_u32(block + i + 3 * blocksize + 8, matrix[11]);
        vst1q_u32(block + i + 3 * blocksize + 12, matrix[15]);
    }
}

/*
 * @brief Matrix block transpose function for general matrix transposition.
 * @param src Pointer to the input matrix (uint32_t).
 * @param ny Number of rows in the input matrix.
 * @param dim Number of columns in the input matrix.
 * @param blocksize Block size for transpose operations.
 * @param block Pointer to the output block matrix (uint32_t). Output Matrix size (dim * ceil(ny / blocksize) *
 * blocksize).
 */
int krl_matrix_block_transpose(
    const uint32_t *src, size_t ny, size_t dim, size_t blocksize, uint32_t *block, size_t block_size)
{
    size_t i = 0;
    size_t bid = 0;
    for (; i + blocksize <= ny; i += blocksize) {
        size_t d = 0;
        for (; d + 4 <= dim; d += 4) {
            krl_matrix_block_transpose_kernel(src + i * dim + d, dim, blocksize, block + bid);
            bid += 4 * blocksize;
        }
        for (; d < dim; ++d) {
            for (size_t j = 0; j < blocksize; ++j) {
                block[bid + j] = src[(i + j) * dim + d];
            }
            bid += blocksize;
        }
    }
    if (i < ny) {
        const size_t left = ny - i;
        for (size_t d = 0; d < dim; ++d) {
            for (size_t j = 0; j < left; ++j) {
                block[bid + j] = src[(i + j) * dim + d];
            }
            size_t remaining_bytes = block_size - (bid + left) * sizeof(uint32_t);
            int ret = SafeMemory::CheckAndMemset(
                block + bid + left, remaining_bytes, 0, (blocksize - left) * sizeof(uint32_t));
            bid += blocksize;
        }
    }
    return SUCCESS;
}
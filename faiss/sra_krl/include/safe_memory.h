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

#pragma once
#include <cstring>
#include <iostream>
#include <algorithm>

namespace SafeMemory {

template <typename D, typename S>
int CheckAndMemcpy(D *dest, size_t destBufferSize, const S *src, size_t srcBufferSize)
{
    if (srcBufferSize > destBufferSize) {
        std::cerr << "Memcpy failed: destBufferSize[" << destBufferSize << "] should be >= srcBufferSize["
                  << srcBufferSize << "].\n";
        return -1;
    }
    if (dest == nullptr || src == nullptr) {
        std::cerr << "Memcpy failed: null pointer detected\n";
        return -1;
    }
    memcpy(dest, src, srcBufferSize);
    return 0;
}

template <typename D>
int CheckAndMemset(D *dest, size_t destBufferSize, int memsetValue, size_t setSize)
{
    if (setSize > destBufferSize) {
        std::cerr << "Memset failed: destBufferSize[" << destBufferSize << "] should be >= setSize[" << setSize
                  << "].\n";
        return -1;
    }
    if (dest == nullptr) {
        std::cerr << "Memset failed: null pointer detected\n";
        return -1;
    }
    memset(dest, memsetValue, setSize);
    return 0;
}

}  // namespace SafeMemory
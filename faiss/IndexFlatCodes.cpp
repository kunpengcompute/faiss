/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlatCodes.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, MetricType metric)
        : Index(d, metric), code_size(code_size) {}

#ifdef __aarch64__
uint8_t* IndexFlatCodes::get_codes_pointer() {
     return codes.data(); 
}
#endif
IndexFlatCodes::IndexFlatCodes() : code_size(0) {}

void IndexFlatCodes::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }
    codes.resize((ntotal + n) * code_size);
    sa_encode(n, x, codes.data() + (ntotal * code_size));
    ntotal += n;
}

void IndexFlatCodes::reset() {
    codes.clear();
    ntotal = 0;
}

size_t IndexFlatCodes::sa_code_size() const {
    return code_size;
}

size_t IndexFlatCodes::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove(&codes[code_size * j],
                        &codes[code_size * i],
                        code_size);
            }
            j++;
        }
    }
    size_t nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        codes.resize(ntotal * code_size);
    }
    return nremove;
}

void IndexFlatCodes::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    sa_decode(ni, codes.data() + i0 * code_size, recons);
}

void IndexFlatCodes::reconstruct(idx_t key, float* recons) const {
    reconstruct_n(key, 1, recons);
}

FlatCodesDistanceComputer* IndexFlatCodes::get_FlatCodesDistanceComputer()
        const {
    FAISS_THROW_MSG("not implemented");
}

void IndexFlatCodes::check_compatible_for_merge(const Index& otherIndex) const {
    // minimal sanity checks
    const IndexFlatCodes* other =
            dynamic_cast<const IndexFlatCodes*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(*other),
            "can only merge indexes of the same type");
}

void IndexFlatCodes::merge_from(Index& otherIndex, idx_t add_id) {
    FAISS_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in FlatCodes index");
    check_compatible_for_merge(otherIndex);
    IndexFlatCodes* other = static_cast<IndexFlatCodes*>(&otherIndex);
    codes.resize((ntotal + other->ntotal) * code_size);
    memcpy(codes.data() + (ntotal * code_size),
           other->codes.data(),
           other->ntotal * code_size);
    ntotal += other->ntotal;
    other->reset();
}

CodePacker* IndexFlatCodes::get_CodePacker() const {
    return new CodePackerFlat(code_size);
}

void IndexFlatCodes::permute_entries(const idx_t* perm) {
#ifdef __aarch64__
    std::vector<uint8_t, AlignedAllocator<uint8_t>> new_codes(codes.size());
#else
    std::vector<uint8_t> new_codes(codes.size());
#endif
    for (idx_t i = 0; i < ntotal; i++) {
        memcpy(new_codes.data() + i * code_size,
               codes.data() + perm[i] * code_size,
               code_size);
    }
    std::swap(codes, new_codes);
}
#ifdef __aarch64__
#include <arm_neon.h>
void IndexFlatCodes::dequant_entries_f32(const uint8_t* entries, idx_t num_entries, int quant_bit) {
    std::vector<uint8_t, AlignedAllocator<uint8_t>> new_codes(sizeof(float) * num_entries);
    float* c = (float*)new_codes.data();
    if(quant_bit == 16) {
        float16_t* e = (float16_t*)(entries);
        for (size_t i = 0; i < num_entries; ++i) {
            c[i] = (float)e[i];
        }
    } else if(quant_bit == 8) {
        uint8_t* e = (uint8_t*)(entries);
        for (size_t i = 0; i < num_entries; ++i) {
            c[i] = (float)e[i];
        }
    }
    std::swap(codes, new_codes);
    ntotal = num_entries;
}

void IndexFlatCodes::quant_entries_f16(const uint8_t* entries, idx_t num_entries, float scale) {
    std::vector<uint8_t, AlignedAllocator<uint8_t>> new_codes(sizeof(float16_t) * num_entries);
    float16_t* c = (float16_t*)(new_codes.data());
    float* e = (float*)entries;
    for (size_t i = 0; i < num_entries; ++i) {
        c[i] = (float16_t)(scale * e[i]);
    }
    std::swap(codes, new_codes);
    ntotal = num_entries;
}
void IndexFlatCodes::quant_entries_u8(const uint8_t* entries, idx_t num_entries, float scale) {
    std::vector<uint8_t, AlignedAllocator<uint8_t>> new_codes(sizeof(uint8_t) * num_entries);
    uint8_t* c = (uint8_t*)(new_codes.data());
    float* e = (float*)entries;
    for (size_t i = 0; i < num_entries; ++i) {
        c[i] = (uint8_t)(scale * e[i] + 0.5);
    }
    std::swap(codes, new_codes);
    ntotal = num_entries;
}
#endif
} // namespace faiss

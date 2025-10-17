/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <vector>

#ifdef __aarch64__
#include <faiss/sra_krl/include/krl.h>
#endif

namespace faiss {

struct CodePacker;

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatCodes : Index {
    size_t code_size;

    /// encoded dataset, size ntotal * code_size
#ifdef __aarch64__
    std::vector<uint8_t, AlignedAllocator<uint8_t>> codes;
    uint8_t* get_codes_pointer() override;
#else
    std::vector<uint8_t> codes;
#endif
    IndexFlatCodes();

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);

    /// default add uses sa_encode
    void add(idx_t n, const float* x) override;

    void reset() override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    size_t sa_code_size() const override;

    /** remove some ids. NB that because of the structure of the
     * index, the semantics of this operation are
     * different from the usual ones: the new ids are shifted */
    size_t remove_ids(const IDSelector& sel) override;

    /** a FlatCodesDistanceComputer offers a distance_to_code method */
    virtual FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

    DistanceComputer* get_distance_computer() const override {
        return get_FlatCodesDistanceComputer();
    }

    // returns a new instance of a CodePacker
    CodePacker* get_CodePacker() const;

    void check_compatible_for_merge(const Index& otherIndex) const override;

    virtual void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    // permute_entries. perm of size ntotal maps new to old positions
    void permute_entries(const idx_t* perm);
#ifdef __aarch64__
    bool use_handle = false;
    KRLDistanceHandle* kdh = nullptr;
    void train(idx_t n, const float* x) override {
        if (ntotal > 0 && n == -1 && !use_handle) {
            use_handle = true;
            krl_create_reorder_handle(
                &kdh, 1, 3, ntotal, d, metric_type, (const uint8_t *)codes.data(), ntotal * d * 4);
        }
    }
    ~IndexFlatCodes() {
        if(use_handle) {
            use_handle = false;
            krl_clean_distance_handle(&kdh);
        }
    }
    void dequant_entries_f32(const uint8_t* entries, idx_t num_entries, int quant_bit) override;
    void quant_entries_f16(const uint8_t* entries, idx_t num_entries, float scale)   override;
    void quant_entries_u8(const uint8_t* entries, idx_t num_entries, float scale)    override;
#endif
};

} // namespace faiss

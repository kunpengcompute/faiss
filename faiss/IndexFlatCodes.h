/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Modifications:
 * - 2026 BoostKit: Supports the FP16 function.
 */

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
#ifdef KRL
#include <faiss/sra_krl/include/krl.h>
#endif

#ifdef __aarch64__
typedef __fp16 float16_t;
#endif

namespace faiss {

struct CodePacker;

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatCodes : Index {
    size_t code_size;

    /// encoded dataset, size ntotal * code_size
#if defined(KRL) || defined(OPTI_IVFPQ)
    std::vector<uint8_t, AlignedAllocator<uint8_t>> codes;
#if defined(KRL) && !defined(OPTI_IVFPQ)
    uint8_t* get_codes_pointer() override;
#endif
#else
    std::vector<uint8_t> codes;
#endif
    IndexFlatCodes();

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);

#ifdef KRL
    IndexFlatCodes(size_t code_size, idx_t d, NumericType ntype, MetricType metric = METRIC_L2);
#endif

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

#ifdef KRL
    bool use_handle = false;
    KRLDistanceHandle* kdh = nullptr;

    IndexFlatCodes(const IndexFlatCodes& other)
	        : Index(other),
			  code_size(other.code_size),
			  codes(other.codes),
			  use_handle(false),
			  kdh(nullptr) {}

	IndexFlatCodes& operator=(const IndexFlatCodes& other) {
		if (this != &other) {
			Index::operator=(other);
			code_size = other.code_size;
			codes = other.codes;
			if (kdh) {
				krl_clean_distance_handle(&kdh);
				kdh = nullptr;
			}
			use_handle = false;
		}
		return *this;
	}

    void train(idx_t n, const float* x) override {
        if (ntotal > 0 && n == -1 && !kdh) {
            krl_create_reorder_handle(
                &kdh, 1, 3, ntotal, d, metric_type, (const uint8_t *)codes.data(), ntotal * d * sizeof(float));
        	use_handle = (kdh != nullptr);
		}
    }

    ~IndexFlatCodes() {
        if(kdh) {
            krl_clean_distance_handle(&kdh);
			kdh = nullptr;
			use_handle = false;
        }
    }
    void dequant_entries_f32(const uint8_t* entries, idx_t num_entries, int quant_bit) override;
    void quant_entries_f16(const uint8_t* entries, idx_t num_entries, float scale) override;
    void quant_entries_u8(const uint8_t* entries, idx_t num_entries, float scale) override;

    /* added FP16 function interfaces */
    void add(idx_t n, const float16_t* x) override;

    void add_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            add(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexFlatCodes::add: unsupported numeric type");
        }
    }

    void reconstruct_n(idx_t i0, idx_t ni, float16_t* recons) const override;

    void reconstruct_n_ex(idx_t i0, idx_t ni, void* recons, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            reconstruct_n(i0, ni, static_cast<float16_t*>(recons));
        } else {
            FAISS_THROW_MSG("IndexFlatCodes::reconstruct_n: unsupported numeric type");
        }
    }

    void reconstruct(idx_t key, float16_t* recons) const override;

    void reconstruct_ex(idx_t key, void* recons, NumericType numeric_type) const override {
        if (numeric_type == NumericType::Float16) {
            reconstruct(key, static_cast<float16_t*>(recons));
        } else {
            FAISS_THROW_MSG("IndexFlatCodes::reconstruct: unsupported numeric type");
        }
    }

    void train(idx_t n, const float16_t* x) override {
        if (ntotal > 0 && n == -1 && !kdh) {
            krl_create_reorder_handle(
                &kdh, 1, 3, ntotal, d, metric_type, (const uint8_t *)codes.data(), ntotal * d * sizeof(float16_t));
        	use_handle = (kdh != nullptr);
		}
    }

    void train_ex(idx_t n, const void* x, NumericType numeric_type) override {
        if (numeric_type == NumericType::Float16) {
            train(n, static_cast<const float16_t*>(x));
        } else {
            FAISS_THROW_MSG("IndexFlatCodes::train: unsupported numeric type");
        }
    }
#endif
};

} // namespace faiss

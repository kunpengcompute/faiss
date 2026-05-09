/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Auxiliary index structures, that are used in indexes but that can
// be forward-declared

#ifndef FAISS_AUX_INDEX_STRUCTURES_H
#define FAISS_AUX_INDEX_STRUCTURES_H

#include <unordered_set>
#include <stdint.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/** The objective is to have a simple result structure while
 *  minimizing the number of mem copies in the result. The method
 *  do_allocation can be overloaded to allocate the result tables in
 *  the matrix type of a scripting language like Lua or Python. */
struct RangeSearchResult {
    size_t nq;    ///< nb of queries
    size_t* lims; ///< size (nq + 1)

    idx_t* labels;    ///< result for query i is labels[lims[i]:lims[i+1]]
    float* distances; ///< corresponding distances (not sorted)

    size_t buffer_size; ///< size of the result buffers used

    /// lims must be allocated on input to range_search.
    explicit RangeSearchResult(size_t nq, bool alloc_lims = true);

    /// called when lims contains the nb of elements result entries
    /// for each query
    virtual void do_allocation();

    virtual ~RangeSearchResult();
};

/****************************************************************
 * Result structures for range search.
 *
 * The main constraint here is that we want to support parallel
 * queries from different threads in various ways: 1 thread per query,
 * several threads per query. We store the actual results in blocks of
 * fixed size rather than exponentially increasing memory. At the end,
 * we copy the block content to a linear result array.
 *****************************************************************/

/** List of temporary buffers used to store results before they are
 *  copied to the RangeSearchResult object. */
struct BufferList {
    // buffer sizes in # entries
    size_t buffer_size;

    struct Buffer {
        idx_t* ids;
        float* dis;
    };

    std::vector<Buffer> buffers;
    size_t wp; ///< write pointer in the last buffer.

    explicit BufferList(size_t buffer_size);

    ~BufferList();

    /// create a new buffer
    void append_buffer();

    /// add one result, possibly appending a new buffer if needed
    void add(idx_t id, float dis);

    /// copy elemnts ofs:ofs+n-1 seen as linear data in the buffers to
    /// tables dest_ids, dest_dis
    void copy_range(size_t ofs, size_t n, idx_t* dest_ids, float* dest_dis);
};

struct RangeSearchPartialResult;

/// result structure for a single query
struct RangeQueryResult {
    idx_t qno;   //< id of the query
    size_t nres; //< nb of results for this query
    RangeSearchPartialResult* pres;

    /// called by search function to report a new result
    void add(float dis, idx_t id);
};

/// the entries in the buffers are split per query
struct RangeSearchPartialResult : BufferList {
    RangeSearchResult* res;

    /// eventually the result will be stored in res_in
    explicit RangeSearchPartialResult(RangeSearchResult* res_in);

    /// query ids + nb of results per query.
    std::vector<RangeQueryResult> queries;

    /// begin a new result
    RangeQueryResult& new_result(idx_t qno);

    /*****************************************
     * functions used at the end of the search to merge the result
     * lists */
    void finalize();

    /// called by range_search before do_allocation
    void set_lims();

    /// called by range_search after do_allocation
    void copy_result(bool incremental = false);

    /// merge a set of PartialResult's into one RangeSearchResult
    /// on ouptut the partialresults are empty!
    static void merge(
            std::vector<RangeSearchPartialResult*>& partial_results,
            bool do_delete = true);
};

/***********************************************************
 * Interrupt callback
 ***********************************************************/

struct FAISS_API InterruptCallback {
    virtual bool want_interrupt() = 0;
    virtual ~InterruptCallback() {}

    // lock that protects concurrent calls to is_interrupted
    static std::mutex lock;

    static std::unique_ptr<InterruptCallback> instance;

    static void clear_instance();

    /** check if:
     * - an interrupt callback is set
     * - the callback returns true
     * if this is the case, then throw an exception. Should not be called
     * from multiple threads.
     */
    static void check();

    /// same as check() but return true if is interrupted instead of
    /// throwing. Can be called from multiple threads.
    static bool is_interrupted();

    /** assuming each iteration takes a certain number of flops, what
     * is a reasonable interval to check for interrupts?
     */
    static size_t get_period_hint(size_t flops);
};

/// set implementation optimized for fast access.
struct VisitedTable {
    std::vector<uint8_t> visited;
    std::unordered_set<int> visited_set;
    uint8_t visno;

    // Use hashset when ntotal >= this threshold.
    // Vector: O(1) get/set, O(ntotal) reset. Hashset: O(1) reset, slower get/set.
    // At ntotal=10M the memset cost dominates on x86; hashset wins by ~3-4x.
    // WARNING: On ARM this causes 10% regression due to cache miss overhead.
    static const int hashset_threshold = 500000;

    explicit VisitedTable(int size)
            : visno(size >= hashset_threshold ? 0 : 1) {
        if (visno != 0) {
            visited.resize(size, 0);
        } else {
            // Pre-allocate hashset capacity to avoid rehash during queries
            visited_set.reserve(1024);
        }
    }

    /// set flag #no to true; returns true if this changed it (was unvisited)
    bool set(int no) {
        if (visno == 0) {
            return visited_set.insert(no).second;
        } else if (visited[no] == visno) {
            return false;
        } else {
            visited[no] = visno;
            return true;
        }
    }

    /// get flag #no
    bool get(int no) const {
        if (visno == 0) {
            return visited_set.count(no) != 0;
        } else {
            return visited[no] == visno;
        }
    }

    /// reset all flags to false
    void advance() {
        if (visno == 0) {
            visited_set.clear();
        } else if (visno < 254) {
            ++visno;
        } else {
            memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
            visno = 1;
        }
    }
};

} // namespace faiss

#endif

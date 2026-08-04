#!/usr/bin/env bash
set -uo pipefail
# KRL incremental coverage test suite: build → run → capture → report.
# Usage:  sh capture_krl_coverage.sh [--compile true|false]
# Optional: KRL_DEPS_DIR=/path/to/deps KRL_FETCH_DEPS=true|false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cov"
FAISS_LIBDIR="${BUILD_DIR}/faiss"
DEPS_DIR="${KRL_DEPS_DIR:-${BUILD_DIR}/_deps}"
FETCH_DEPS="${KRL_FETCH_DEPS:-true}"

INFO_COV="${ROOT_DIR}/coverage_krl.info"
INFO_FILTERED="${ROOT_DIR}/coverage_krl_filtered.info"
REPORT_DIR="${ROOT_DIR}/coverage_krl_report"

LCOV_RC="--rc branch_coverage=1"
LCOV_IGNORE="--ignore-errors deprecated,inconsistent,mismatch,corrupt,range,missing"

DO_COMPILE="${1:-false}"
if [ "$DO_COMPILE" = "--compile" ]; then
    DO_COMPILE="${2:-false}"
fi

# ==================== compile ====================
if [ "$DO_COMPILE" = "true" ]; then

    # ensure offline dependencies exist
    GTEST_SRC="${DEPS_DIR}/googletest-src"
    BENCH_SRC="${DEPS_DIR}/googlebenchmark-src"

    mkdir -p "${DEPS_DIR}" || exit 1

    if [ ! -f "${GTEST_SRC}/CMakeLists.txt" ]; then
        if [ "${FETCH_DEPS}" != "true" ]; then
            echo "missing googletest source: ${GTEST_SRC}"
            echo "run once with network and KRL_FETCH_DEPS=true, or pre-populate KRL_DEPS_DIR"
            exit 1
        fi
        echo "=== Cloning googletest (one-time) ==="
        git clone --depth 1 https://github.com/google/googletest.git "${GTEST_SRC}" || exit 1
        (cd "${GTEST_SRC}" && git fetch --depth 1 origin 58d77fa8070e8cec2dc1ed015d66b454c8d78850 && git checkout 58d77fa8070e8cec2dc1ed015d66b454c8d78850) || exit 1
    fi

    if [ ! -f "${BENCH_SRC}/CMakeLists.txt" ]; then
        if [ "${FETCH_DEPS}" != "true" ]; then
            echo "missing googlebenchmark source: ${BENCH_SRC}"
            echo "run once with network and KRL_FETCH_DEPS=true, or pre-populate KRL_DEPS_DIR"
            exit 1
        fi
        echo "=== Cloning googlebenchmark (one-time) ==="
        git clone --depth 1 https://github.com/google/benchmark.git "${BENCH_SRC}" || exit 1
    fi

    echo "=== Configuring (KRL=ON + coverage) ==="
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Debug -DFAISS_ENABLE_PYTHON=OFF \
        -DBLAS_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so \
        -DFAISS_OPT_LEVEL=generic -DKRL=ON \
        -DCMAKE_C_FLAGS="--coverage -O0 -g -fno-omit-frame-pointer" \
        -DCMAKE_CXX_FLAGS="--coverage -O0 -g -fno-omit-frame-pointer" \
        -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
        -DCMAKE_SHARED_LINKER_FLAGS="--coverage" \
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="${GTEST_SRC}" \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK="${BENCH_SRC}" || exit 1

    echo "=== Building faiss_test ==="
    cmake --build "${BUILD_DIR}" --target faiss_test -j"$(nproc)" || exit 1
fi

# ==================== run ====================
echo "=== Running KRL incremental coverage tests ==="
echo ""
lcov ${LCOV_RC} ${LCOV_IGNORE} --zerocounters --directory "${BUILD_DIR}" >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="${FAISS_LIBDIR}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
${BUILD_DIR}/tests/faiss_test --gtest_filter='KRLTest.*' 2>&1
TEST_EXIT=$?
echo ""

# ==================== capture ====================
echo "=== Capturing coverage ==="
lcov --capture --directory "${BUILD_DIR}" --output-file "${INFO_COV}" \
    ${LCOV_RC} ${LCOV_IGNORE} || true

# ==================== filter ====================
echo "=== Filtering KRL-relevant C++ files ==="
lcov --extract "${INFO_COV}" \
    "*/IndexIVFRaBitQFastScan.cpp" \
    "*/IndexIVFRaBitQFastScan.h" \
    "*/IndexIVFFastScan.cpp" \
    "*/IndexRefine.cpp" \
    "*/IndexRefine.h" \
    "*/dispatching.h" \
    "*/rabitq_result_handler.h" \
    "*/distances.cpp" \
    "*/index_read.cpp" \
    "*/fast_scan.cpp" \
    --output-file "${INFO_FILTERED}" ${LCOV_RC} ${LCOV_IGNORE} || true

# ==================== HTML ====================
echo "=== Generating HTML report ==="
rm -rf "${REPORT_DIR}"
genhtml "${INFO_FILTERED}" --output-directory "${REPORT_DIR}" \
    ${LCOV_RC} ${LCOV_IGNORE} || true

# ==================== summary ====================
echo ""
echo "============================================================"
echo "  HTML report : ${REPORT_DIR}/index.html"
echo "============================================================"
echo ""
lcov --summary "${INFO_FILTERED}" ${LCOV_RC} ${LCOV_IGNORE} 2>&1 || true
echo ""
echo "--- KRL-macro-only coverage (lines inside #ifdef KRL blocks) ---"
if [ -f "${ROOT_DIR}/tests/_krl_line_cov.py" ]; then
  python3 "${ROOT_DIR}/tests/_krl_line_cov.py" "${INFO_FILTERED}" "${ROOT_DIR}" 2>&1 || true
else
  echo "(skipped: tests/_krl_line_cov.py not found)"
fi
echo ""
echo "============================================================"
echo "Test exit code: ${TEST_EXIT}"
echo "============================================================"

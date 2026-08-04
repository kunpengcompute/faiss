#!/usr/bin/env bash
set -uo pipefail
# Run original faiss unit tests under KRL=ON (exclude KRLTest incremental tests).
# Usage:  sh run_original_ut_krl.sh [--compile true|false]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cov"
FAISS_LIBDIR="${BUILD_DIR}/faiss"

DO_COMPILE="${1:-false}"
if [ "$DO_COMPILE" = "--compile" ]; then
    DO_COMPILE="${2:-false}"
fi

# ==================== compile ====================
if [ "$DO_COMPILE" = "true" ]; then

    # ensure offline dependencies exist
    DEPS_DIR="${BUILD_DIR}/_deps"
    GTEST_SRC="${DEPS_DIR}/googletest-src"
    BENCH_SRC="${DEPS_DIR}/googlebenchmark-src"

    if [ ! -d "${GTEST_SRC}" ]; then
        echo "=== Cloning googletest (one-time) ==="
        git clone --depth 1 https://github.com/google/googletest.git "${GTEST_SRC}" || exit 1
        (cd "${GTEST_SRC}" && git fetch --depth 1 origin 58d77fa8070e8cec2dc1ed015d66b454c8d78850 && git checkout 58d77fa8070e8cec2dc1ed015d66b454c8d78850) || exit 1
    fi

    if [ ! -d "${BENCH_SRC}" ]; then
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
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="${GTEST_SRC}" \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK="${BENCH_SRC}" || exit 1

    echo "=== Building faiss_test ==="
    cmake --build "${BUILD_DIR}" --target faiss_test -j"$(nproc)" || exit 1
fi

# ==================== run ====================
echo "============================================================"
echo "  Running original UT under KRL=ON (excluding KRLTest.*)"
echo "  Binary: ${BUILD_DIR}/tests/faiss_test"
echo "============================================================"
echo ""

export LD_LIBRARY_PATH="${FAISS_LIBDIR}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# --gtest_filter=-KRLTest.*  excludes the incremental KRL coverage suite
${BUILD_DIR}/tests/faiss_test --gtest_filter='-KRLTest.*' 2>&1
EXIT=$?

echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "  ALL ORIGINAL UT PASSED (KRL=ON)"
else
    echo "  SOME TESTS FAILED (exit code: $EXIT)"
fi
echo "============================================================"
exit $EXIT

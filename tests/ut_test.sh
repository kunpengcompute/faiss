#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLANG_C="${CLANG_C:-/opt/llvm-16.0.6/bin/clang}"
CLANG_CXX="${CLANG_CXX:-/opt/llvm-16.0.6/bin/clang++}"
LLVM_COV="${LLVM_COV:-/opt/llvm-16.0.6/bin/llvm-cov}"

BLAS_LIBRARIES="${BLAS_LIBRARIES:-/opt/OpenBLAS/lib/libopenblas.so}"

GOMP_SO="${GOMP_SO:-/usr/lib/gcc/aarch64-linux-gnu/12/libgomp.so}"

BUILD_GCC="${ROOT_DIR}/build_cov_gcc"
BUILD_CLANG="${ROOT_DIR}/build_cov_clang"

INFO_GCC="${ROOT_DIR}/coverage_gcc.info"
INFO_CLANG="${ROOT_DIR}/coverage_clang.info"
INFO_MERGED="${ROOT_DIR}/coverage_merged.info"

REPORT_DIR="${ROOT_DIR}/coverage_report"

COV_CFLAGS='--coverage -O0 -g -fno-omit-frame-pointer'
COV_LDFLAGS='--coverage'

COMMON_CMAKE_OPTS=(
  -DFAISS_ENABLE_GPU=OFF
  -DBUILD_TESTING=ON
  -DBUILD_SHARED_LIBS=ON
  -DCMAKE_BUILD_TYPE=Debug
  -DFAISS_ENABLE_PYTHON=OFF
  -DBLAS_LIBRARIES="${BLAS_LIBRARIES}"
  -DFAISS_OPT_LEVEL=generic
  -DOPTI_IVFPQ=on
)

LLVM_GCOV_SH="${SCRIPT_DIR}/llvm-gcov.sh"
cat > "${LLVM_GCOV_SH}" <<EOF
#!/usr/bin/env sh
exec "${LLVM_COV}" gcov "\$@"
EOF
chmod +x "${LLVM_GCOV_SH}"

clean_all() {
  rm -rf "${BUILD_GCC}" "${BUILD_CLANG}" "${REPORT_DIR}"
  rm -f "${INFO_GCC}" "${INFO_CLANG}" "${INFO_MERGED}"
}

configure_gcc() {
  cmake -S "${ROOT_DIR}" -B "${BUILD_GCC}" \
    "${COMMON_CMAKE_OPTS[@]}" \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_FLAGS="${COV_CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${COV_CFLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${COV_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COV_LDFLAGS}"
}

configure_clang() {
  local omp_flag="-fopenmp"
  local omp_lib="/opt/llvm-16.0.6/lib/libomp.so"

  if [[ ! -f "${omp_lib}" ]]; then
    echo "ERROR: ${omp_lib} not found. Please install: sudo apt-get install -y libomp-16-dev"
    exit 1
  fi

  local omp_dir
  omp_dir="$(dirname "${omp_lib}")"

  local cflags="${COV_CFLAGS} ${omp_flag}"
  local ldflags="${COV_LDFLAGS} ${omp_flag} -Wl,-rpath,${omp_dir}"

  CC="${CLANG_C}" CXX="${CLANG_CXX}" cmake -S "${ROOT_DIR}" -B "${BUILD_CLANG}" \
    "${COMMON_CMAKE_OPTS[@]}" \
    -DCMAKE_C_FLAGS="${cflags}" \
    -DCMAKE_CXX_FLAGS="${cflags}" \
    -DCMAKE_EXE_LINKER_FLAGS="${ldflags}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${ldflags}" \
    -DOpenMP_C_FLAGS="${omp_flag}" \
    -DOpenMP_CXX_FLAGS="${omp_flag}" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY="${omp_lib}"
}

build_and_test() {
  local build_dir="$1"

  cmake --build "${build_dir}" -j

  lcov --zerocounters --directory "${build_dir}" >/dev/null 2>&1 || true

  local faiss_libdir="${build_dir}/faiss"
  if [[ ! -d "${faiss_libdir}" ]]; then
    echo "ERROR: faiss lib dir not found: ${faiss_libdir}"
    exit 1
  fi

  ( export LD_LIBRARY_PATH="${faiss_libdir}:${LD_LIBRARY_PATH:-}"
    ctest --test-dir "${build_dir}/tests" --output-on-failure -j1
  )
}


capture_faiss_only_gcc() {
  lcov --capture --directory "${BUILD_GCC}" --output-file "${INFO_GCC}" --rc lcov_branch_coverage=1
  lcov --extract "${INFO_GCC}" "*/faiss/*" --output-file "${INFO_GCC}" --rc lcov_branch_coverage=1
}

capture_faiss_only_clang() {
  lcov --capture --directory "${BUILD_CLANG}" --output-file "${INFO_CLANG}" \
    --gcov-tool "${LLVM_GCOV_SH}" \
    --rc lcov_branch_coverage=1
  lcov --extract "${INFO_CLANG}" "*/faiss/*" --output-file "${INFO_CLANG}" --rc lcov_branch_coverage=1
}

merge_and_genhtml() {
  lcov --add-tracefile "${INFO_GCC}" --add-tracefile "${INFO_CLANG}" \
    --output-file "${INFO_MERGED}" --rc lcov_branch_coverage=1

  rm -rf "${REPORT_DIR}"
  genhtml "${INFO_MERGED}" --output-directory "${REPORT_DIR}" --rc lcov_branch_coverage=1
}

echo "[1/8] clean"
clean_all

echo "[2/8] configure gcc"
configure_gcc

echo "[3/8] build & test gcc"
build_and_test "${BUILD_GCC}"

echo "[4/8] capture gcc (faiss/ only)"
capture_faiss_only_gcc

echo "[5/8] configure clang"
configure_clang

echo "[6/8] build & test clang"
build_and_test "${BUILD_CLANG}"

echo "[7/8] capture clang (faiss/ only)"
capture_faiss_only_clang

echo "[8/8] merge + html"
merge_and_genhtml

echo
echo "DONE: ${REPORT_DIR}/index.html"
echo "Tracefiles:"
echo "  ${INFO_GCC}"
echo "  ${INFO_CLANG}"
echo "  ${INFO_MERGED}"

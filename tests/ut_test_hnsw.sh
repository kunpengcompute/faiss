#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLANG_C="${CLANG_C:-/opt/llvm-16.0.6/bin/clang}"
CLANG_CXX="${CLANG_CXX:-/opt/llvm-16.0.6/bin/clang++}"
LLVM_COV="${LLVM_COV:-/opt/llvm-16.0.6/bin/llvm-cov}"

BLAS_LIBRARIES="${BLAS_LIBRARIES:-/opt/OpenBLAS/lib/libopenblas.so}"

BUILD_GCC="${ROOT_DIR}/build_cov_gcc"
BUILD_CLANG="${ROOT_DIR}/build_cov_clang"

COV_CFLAGS='-O0 -fno-omit-frame-pointer -march=armv8.2-a+fp16fml+dotprod+sve+rcpc'

COMMON_CMAKE_OPTS=(
  -DFAISS_ENABLE_GPU=OFF
  -DBUILD_TESTING=ON
  -DBUILD_SHARED_LIBS=ON
  -DCMAKE_BUILD_TYPE=Debug
  -DFAISS_ENABLE_PYTHON=OFF
  -DBLAS_LIBRARIES="${BLAS_LIBRARIES}"
  -DFAISS_OPT_LEVEL=generic
  -DOPTI_IVFPQ=OFF
  -DKRL=ON
  -DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST
)

export QEMU_AARCH64="${QEMU_AARCH64:-/opt/qemu-sve2/bin/qemu-aarch64}"
export QEMU_SYSROOT="${QEMU_SYSROOT:-/}"
export QEMU_CPU="${QEMU_CPU:-max}"

export FORCE_QEMU_REGEX="${FORCE_QEMU_REGEX:-}"

WRAPPER="${SCRIPT_DIR}/run-native-or-qemu.sh"
cat > "${WRAPPER}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exe="$1"; shift

QEMU_AARCH64="${QEMU_AARCH64:-/opt/qemu-sve2/bin/qemu-aarch64}"
QEMU_SYSROOT="${QEMU_SYSROOT:-/}"
QEMU_CPU="${QEMU_CPU:-max}"
FORCE_QEMU_REGEX="${FORCE_QEMU_REGEX:-}"

run_qemu() {
  exec "${QEMU_AARCH64}" -cpu "${QEMU_CPU}" -L "${QEMU_SYSROOT}" "${exe}" "$@"
}

cmdline="${exe} $*"
if [[ -n "${FORCE_QEMU_REGEX}" ]] && echo "${cmdline}" | grep -Eq "${FORCE_QEMU_REGEX}"; then
  run_qemu "$@"
fi

set +e
"${exe}" "$@"
rc=$?
set -e

if [[ "${rc}" -eq 132 ]]; then
  echo "[wrapper] native SIGILL -> rerun with qemu: ${cmdline}" >&2
  run_qemu "$@"
fi

exit "${rc}"
EOF
chmod +x "${WRAPPER}"

TOOLCHAIN_SMART="${SCRIPT_DIR}/toolchain-smart.cmake"

EMULATOR_LIST="${WRAPPER}"
EMULATOR_LIST="${EMULATOR_LIST//;/\\;}"

cat > "${TOOLCHAIN_SMART}" <<EOF
set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_SYSTEM_PROCESSOR aarch64-sve2)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_CROSSCOMPILING_EMULATOR ${EMULATOR_LIST})

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)
EOF

LLVM_GCOV_SH="${SCRIPT_DIR}/llvm-gcov.sh"
cat > "${LLVM_GCOV_SH}" <<EOF
exec "${LLVM_COV}" gcov "\$@"
EOF
chmod +x "${LLVM_GCOV_SH}"

clean_all() {
  rm -rf "${BUILD_GCC}" "${BUILD_CLANG}"
}

configure_gcc() {
  local omp_flag="-fopenmp"

  local gomp_lib
  gomp_lib="$(gcc -print-file-name=libgomp.so.1)"
  if [[ -z "${gomp_lib}" || "${gomp_lib}" == "libgomp.so.1" || ! -f "${gomp_lib}" ]]; then
    gomp_lib="/usr/lib/aarch64-linux-gnu/libgomp.so.1"
  fi
  if [[ ! -f "${gomp_lib}" ]]; then
    echo "ERROR: libgomp.so.1 not found. Please install it:"
    echo "  sudo apt-get update && sudo apt-get install -y libgomp1"
    exit 1
  fi
  
  local cflags="${COV_CFLAGS} ${omp_flag}"
  local ldflags="${omp_flag}"

  cmake -S "${ROOT_DIR}" -B "${BUILD_GCC}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_SMART}" \
    "${COMMON_CMAKE_OPTS[@]}" \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_FLAGS="${cflags}" \
    -DCMAKE_CXX_FLAGS="${cflags}" \
    -DCMAKE_EXE_LINKER_FLAGS="${ldflags}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${ldflags}" \
    -DOpenMP_C_FLAGS="${omp_flag}" \
    -DOpenMP_CXX_FLAGS="${omp_flag}" \
    -DOpenMP_C_LIB_NAMES="gomp" \
    -DOpenMP_CXX_LIB_NAMES="gomp" \
    -DOpenMP_gomp_LIBRARY="${gomp_lib}"
}

configure_clang() {
  local omp_flag="-fopenmp"
  local omp_lib="/opt/llvm-16.0.6/lib/libomp.so"

  if [[ ! -f "${omp_lib}" ]]; then
    echo "ERROR: ${omp_lib} not found."
    echo "Hint: please install LLVM OpenMP runtime (libomp)."
    exit 1
  fi

  local omp_dir
  omp_dir="$(dirname "${omp_lib}")"

  local cflags="${COV_CFLAGS} ${omp_flag}"
  local ldflags="${omp_flag} -Wl,-rpath,${omp_dir}"

  CC="${CLANG_C}" CXX="${CLANG_CXX}" cmake -S "${ROOT_DIR}" -B "${BUILD_CLANG}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_SMART}" \
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

  cmake --build "${build_dir}" -j"$(nproc)"

  local faiss_libdir="${build_dir}/faiss"
  if [[ ! -d "${faiss_libdir}" ]]; then
    echo "ERROR: faiss lib dir not found: ${faiss_libdir}"
    exit 1
  fi

  local CTEST_JOBS="${CTEST_JOBS:-$(nproc)}"

  (
    export LD_LIBRARY_PATH="${faiss_libdir}:${LD_LIBRARY_PATH:-}"
    ctest --test-dir "${build_dir}/tests" --output-on-failure -j"${CTEST_JOBS}"
  )
}

echo "[1/8] clean"
clean_all

echo "[2/8] configure gcc"
configure_gcc

echo "[3/8] build & test gcc"
build_and_test "${BUILD_GCC}"

# echo "[5/8] configure clang"
# configure_clang

# echo "[6/8] build & test clang"
# build_and_test "${BUILD_CLANG}"

echo
echo "DONE"
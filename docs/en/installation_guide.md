# Installation Guide

## Verified Environments

To use Faiss smoothly and securely, ensure that your environment is one of the verified environments.

Table 1 Verified environments for Faiss<a id="verified-environments-for-faiss "></a>

<a name="table4692134313211"></a>
<table><thead align="left"><tr id="row1169294312212"><th class="cellrowborder" valign="top" width="21.8%" id="mcps1.2.6.1.1"><p id="p12692144313211"><a name="p12692144313211"></a><a name="p12692144313211"></a>OS</p>
</th>
<th class="cellrowborder" valign="top" width="19.91%" id="mcps1.2.6.1.2"><p id="p06926438214"><a name="p06926438214"></a><a name="p06926438214"></a>CPU</p>
</th>
<th class="cellrowborder" valign="top" width="13.700000000000001%" id="mcps1.2.6.1.3"><p id="p269284310216"><a name="p269284310216"></a><a name="p269284310216"></a>Memory</p>
</th>
<th class="cellrowborder" valign="top" width="17.34%" id="mcps1.2.6.1.4"><p id="p196922434215"><a name="p196922434215"></a><a name="p196922434215"></a>Compiler</p>
</th>
<th class="cellrowborder" valign="top" width="30%" id="mcps1.2.6.1.5"><p id="p1769219435210"><a name="p1769219435210"></a><a name="p1769219435210"></a>Remarks</p>
</th>
</tr>
</thead>
<tbody><tr id="row624713534398"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.6.1.1 "><p id="p418818120409"><a name="p418818120409"></a><a name="p418818120409"></a>openEuler 22.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="19.91%" headers="mcps1.2.6.1.2 "><p id="p468512464216"><a name="p468512464216"></a><a name="p468512464216"></a>New Kunpeng 920 processor model</p>
</td>
<td class="cellrowborder" valign="top" width="13.700000000000001%" headers="mcps1.2.6.1.3 "><p id="p56685294911"><a name="p56685294911"></a><a name="p56685294911"></a>16 × 32 GB</p>
</td>
<td class="cellrowborder" valign="top" width="17.34%" headers="mcps1.2.6.1.4 "><p id="p1118821124015"><a name="p1118821124015"></a><a name="p1118821124015"></a>GCC 12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="27.250000000000004%" headers="mcps1.2.6.1.5 "><p id="p18389155712168"><a name="p18389155712168"></a><a name="p18389155712168"></a>CMake&gt;=3.22.0</p>
</td>
</tr>
<tr id="row8219349894"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.6.1.1 "><p id="p1321915499910"><a name="p1321915499910"></a><a name="p1321915499910"></a>Debian 12</p>
</td>
<td class="cellrowborder" valign="top" width="19.91%" headers="mcps1.2.6.1.2 "><p id="p152199497916"><a name="p152199497916"></a><a name="p152199497916"></a>New Kunpeng 920 processor model</p>
</td>
<td class="cellrowborder" valign="top" width="13.700000000000001%" headers="mcps1.2.6.1.3 "><p id="p1121911491393"><a name="p1121911491393"></a><a name="p1121911491393"></a>16 × 32 GB</p>
</td>
<td class="cellrowborder" valign="top" width="17.34%" headers="mcps1.2.6.1.4 "><p id="p172198491297"><a name="p172198491297"></a><a name="p172198491297"></a>GCC 12.2.0/LLVM 16.0.6</p>
</td>
<td class="cellrowborder" valign="top" width="27.250000000000004%" headers="mcps1.2.6.1.5 "><p id="p19219164917914"><a name="p19219164917914"></a><a name="p19219164917914"></a>CMake&gt;=3.25.1</p>
</td>
</tr>
<tr id="row159615141350"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.6.1.1 "><p id="p179611141352"><a name="p179611141352"></a><a name="p179611141352"></a>openEuler 24.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="19.91%" headers="mcps1.2.6.1.2 "><p id="p8961314756"><a name="p8961314756"></a><a name="p8961314756"></a>Kunpeng 950 processor</p>
</td>
<td class="cellrowborder" valign="top" width="13.700000000000001%" headers="mcps1.2.6.1.3 "><p id="p15961191418510"><a name="p15961191418510"></a><a name="p15961191418510"></a>24 × 64 GB</p>
</td>
<td class="cellrowborder" valign="top" width="17.34%" headers="mcps1.2.6.1.4 "><p id="p496115148515"><a name="p496115148515"></a><a name="p496115148515"></a>GCC 12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="27.250000000000004%" headers="mcps1.2.6.1.5 "><p id="p109611114654"><a name="p109611114654"></a><a name="p109611114654"></a>CMake&gt;=3.22.0</p>
</td>
</tr>
<tr id="row1837942531312"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.6.1.1 "><p id="p1438015251131"><a name="p1438015251131"></a><a name="p1438015251131"></a>Debian 12</p>
</td>
<td class="cellrowborder" valign="top" width="19.91%" headers="mcps1.2.6.1.2 "><p id="p578363871318"><a name="p578363871318"></a><a name="p578363871318"></a>Kunpeng 950 processor</p>
</td>
<td class="cellrowborder" valign="top" width="13.700000000000001%" headers="mcps1.2.6.1.3 "><p id="p127833389131"><a name="p127833389131"></a><a name="p127833389131"></a>24 × 64 GB</p>
</td>
<td class="cellrowborder" valign="top" width="17.34%" headers="mcps1.2.6.1.4 "><p id="p821894241316"><a name="p821894241316"></a><a name="p821894241316"></a>GCC 12.2.0/LLVM 16.0.6</p>
</td>
<td class="cellrowborder" valign="top" width="27.250000000000004%" headers="mcps1.2.6.1.5 "><p id="p1021810427131"><a name="p1021810427131"></a><a name="p1021810427131"></a>CMake&gt;=3.25.1</p>
</td>
</tr>
</tbody>
</table>

## Compilation and Installation

Obtain the Faiss open-source code from GitCode, install the necessary dependencies and libraries, and install the patch optimized for the Kunpeng platform. Then, recompile Faiss to apply the optimized features, reduce the computing latency, and improve the computing efficiency.

1. Obtain the Faiss open-source code. The tag is `v1.8.0`. Assume that the code is stored in `/path/to/faiss`.

    ```bash
    git clone --branch v1.8.0 --single-branch https://github.com/facebookresearch/faiss.git
    ```

2. Obtain the patch file optimized for Kunpeng. The tag is `v1.8.0-2603`. Assume that the patch file is stored in `/path/to/faiss-patch`.

    ```bash
    git clone --branch v1.8.0-2603 https://gitcode.com/boostkit/faiss.git faiss-patch
    ```

    >![note](public_sys-resources/icon-note.gif) **NOTE:**
    >The following explains the patch files optimized for Kunpeng. Select a patch file as required.
    >- `0001-faiss_1.8.0-optimize-neq.patch`: non-equivalence optimization patch. It delivers optimal performance and ensures precision, but does not guarantee that the values or sequence of top K results are completely consistent with the original version.
    >- `0002-faiss_1.8.0-optimize-eqv.patch`: equivalence optimization patch. It ensures that the values and sequence of top K results are completely consistent with the original version.

3. Install Make, CMake, and GCC. The GCC 12 installation procedure applies to openEuler 22.03 LTS SP3. openEuler 24.03 LTS SP3 comes with GCC 12 pre-installed, so you only need to install Make and CMake.

    ```bash
    yum install make cmake gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ gcc-toolset-12-libstdc++-static gcc-toolset-12-gcc-gfortran
    export PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
    export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64/:$LD_LIBRARY_PATH
    ```

4. Faiss depends on the math library. Download the open-source OpenBLAS source code from the [GitHub repository](https://github.com/OpenMathLib/OpenBLAS.git) using the `v0.3.29` tag. Save the file to a path accessible to the compiler, such as `/path/to/OpenBLAS-0.3.29`.

    ```bash
    git clone --branch v0.3.29 --single-branch https://github.com/OpenMathLib/OpenBLAS.git
    ```

5. <a id="li880635723510"></a>Compile the source code to obtain the `libopenblas.so` file.

    ```bash
    cd /path/to/OpenBLAS-0.3.29/OpenBLAS
    make
    make install
    ```

    >![note](public_sys-resources/icon-note.gif) **NOTE:**
    >You can run the `make install PREFIX=/path/to/openblas/install` command to specify the installation path `/path/to/openblas/install`. The default installation path is `/opt/OpenBLAS`.

6. Install the patch file `0001-faiss\_1.8.0-optimize-neq.patch` or `0002-faiss\_1.8.0-optimize-eqv.patch`.

    ```bash
    cd /path/to/faiss
    patch -p1 < /path/to/faiss-patch/0001-faiss_1.8.0-optimize-neq.patch
    # patch -p1 < /path/to/faiss-patch/0002-faiss_1.8.0-optimize-eqv.patch
    ```

    The full directory structure of Faiss after applying the patches is as follows:

    ```text
    faiss/
    ├─ benchs/                                     // Benchmark
    ├─ c_api/                                      // C API wrapper
    ├─ cmake/                                      // CMake configuration module
    ├─ conda/                                      // Conda build script
    ├─ contrib/                                    // Python contribution module
    ├─ demos/                                      // Demonstration program
    ├─ faiss/
    │   ├─ CMakeLists.txt                          // Build configuration
    │   ├─ Index.h                                 // Abstract base class with a unified interface
    │   ├─ IndexFlat.cpp                           // Implementation of exhaustive search
    │   ├─ IndexFlatCodes.h                        // Base class for flat code storage (used for PQ, SQ, etc.)
    │   ├─ IndexFlatCodes.cpp                      // Implementation of the base class for flat code storage (used for PQ, SQ, etc.)
    │   ├─ IndexFastScan.h                         // General interface for 4-bit PQ/AQ fast scanning
    │   ├─ IndexFastScan.cpp                       // General implementation of 4-bit PQ/AQ fast scanning
    │   ├─ IndexIVF.h                              // IVF base class interface
    │   ├─ IndexIVF.cpp                            // IVF base class + specific implementation
    │   ├─ IndexIVFFlat.cpp                        // IVFFlat implementation
    │   ├─ IndexIVFPQ.cpp                          // IVFPQ implementation
    │   ├─ IndexIVFFastScan.h                      // IVFPQFastScan API
    │   ├─ IndexIVFFastScan.cpp                    // IVFPQFastScan (CPU) implementation
    │   ├─ IndexHNSW.h                             // HNSW index API
    │   ├─ IndexHNSW.cpp                           // HNSW index implementation
    │   ├─ IndexRefine.h                           // Base+refinement composite index API
    │   ├─ IndexRefine.cpp                         // Implementation of the base+refinement composite index
    │   ├─ impl/
    │   │   ├─ DistanceComputer.h                  // Abstract interface for distance computation
    │   │   ├─ ProductQuantizer.h                  // Product quantizer interface
    │   │   ├─ ProductQuantizer.cpp               // Product quantizer implementation
    │   │   ├─ pq4_fast_scan.h                     // API for 4-bit PQ fast scanning
    │   │   ├─ pq4_fast_scan_search_1.cpp          // Single-query implementation of 4-bit PQ fast scanning
    │   │   ├─ pq4_fast_scan_search_qbs.cpp        // Single-query implementation of 4-bit PQ fast scanning
    │   │   ├─ HNSW.cpp                            // Implementation of HNSW graph structure
    │   │   ├─ index_read.cpp                      // Implementation of index deserialization
    │   │   └─ simd_result_handlers.h              // SIMD result handlers
    │   ├─ invlists/
    │   │   ├─ InvertedLists.h                     // Abstract interface for inverted lists
    │   │   └─ InvertedLists.cpp                   // Implementation of inverted lists
    │   ├─ utils/
    │   │   └─ distances_simd.cpp                  // SIMD implementations for L2, Inner Product (IP), L1, and Linf distances
    │   ├─ sra_krl/
    │   │   ├─ include/
    │   │   │   ├─ krl.h                           // Declarations of unified public APIs
    │   │   │   ├─ krl_internal.h                  // Internal structures, macros, and auxiliary SIMD implementations
    │   │   │   ├─ platform_macros.h               // Error codes, metric constants, and platform macros
    │   │   │   └─ safe_memory.h                   // Safe memory operations
    │   │   └─ src/
    │   │       ├─ Heap_sort.c                     // Top-K heap construction and dual-heap reordering implementation
    │   │       ├─ IPdistance_simd.c               // SIMD implementation of single-precision vector inner product (batches of 2/4/8/16)
    │   │       ├─ IPdistance_simd_f16.c           // Implementation of float16 IP distance computation
    │   │       ├─ IPdistance_simd_f16f32.c        // Implementation of float16 IP distance computation with float output
    │   │       ├─ IPdistance_simd_s8.c            // Implementation of int8 IP distance computation with int32/float output
    │   │       ├─ L2distance_simd.c               // Implementation of float L2 distance computation (batches of 2/4/8/16/24)
    │   │       ├─ L2distance_simd_f16.c           // Implementation of float16 L2 distance computation
    │   │       ├─ L2distance_simd_f16f32.c        // Implementation of float16 L2 distance computation with float output
    │   │       ├─ L2distance_simd_u8.c            // Implementation of uint8 L2 distance computation with uint32/float output
    │   │       ├─ matrix_block_transpose.c        // / 4x4 block transposition kernel
    │   │       ├─ MinMax_quant.c                  // Quantization (fp16/u8/s8)
    │   │       ├─ NegaIPdistance_simd_f16f32.c    // Implementation of float16 negative IP distance computation with float output
    │   │       ├─ NegaIPdistance_simd_s8.c        // Implementation of int8 negative IP distance computation with int32/float output
    │   │       ├─ handle_IO.c                     // Handle serialization and deserialization (file I/O)
    │   │       ├─ krl_handles.c                   // Handle creation, initialization, cleanup, and pointer access
    │   │       ├─ pq_search_with_table_4bit.c     // 4-bit table lookup
    │   │       ├─ pq_search_with_table_8bit.c     // 8-bit table lookup
    │   │       ├─ reorder_2_vectors.c             // Sparse/contiguous reordering
    │   │       └─ sve_search_codes.c              // 4-bit fp16 table lookup using SVE
    │   ├─ cppcontrib/                             // C++ contribution module
    │   ├─ gpu/                                    // GPU subsystem
    │   └─ python/                                 // Python binding
    ├─ misc/                                       // Miscellaneous tests
    ├─ misc/                                       // Unit test
    ├─ tutorial/                                   // Tutorials and examples
    ├─ CMakeLists.txt                              // Top-level build configuration
    ├─ CHANGELOG.md
    ├─ CODE_OF_CONDUCT.md
    ├─ CONTRIBUTING.md
    ├─ INSTALL.md
    ├─ LICENSE
    └─ README_en.md
    ```

7. Compile the Faiss code to obtain `libfaiss.so`. Note: You need to enable the Kunpeng optimization macro to improve performance.

    ```bash
    cd /path/to/faiss
    cmake -B build . \
      -DFAISS_ENABLE_GPU=OFF \
      -DBUILD_TESTING=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DFAISS_OPT_LEVEL=generic \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DMKL_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so
    make -C build -j faiss
    make -C build install
    ```

   - If you choose to use the non-equivalence optimization patch `0001-faiss_1.8.0-optimize-neq.patch`, you can enable either of the following macros to improve performance (the two macros are mutually exclusive):
     - `-DKRL=ON`: non-equivalence optimization for HNSW, IVFPQ, IVFPQFS, PQFS, and IVFFLAT. It delivers optimal performance and ensures precision, but does not guarantee that the values or sequence of top K results are completely consistent with the original version.
     - `-DOPTI_IVFPQ=ON`: specific optimization for IVFPQ. Its performance on IVFPQ indexes outperforms that of the KRL macro, and it ensures that the values and sequence of top K results are completely consistent with the original version.
   - If you choose to use the equivalence optimization patch `0002-faiss\_1.8.0-optimize-eqv.patch`, you can enable the following macro to improve performance:
     - `-DKRL=ON`: non-equivalence optimization for HNSW, IVFPQ, IVFPQFS, PQFS, and IVFFLAT. It ensures that the values and sequence of top K results are completely consistent with the original version.
    >
    >**Note:**
    >
    >- During compilation, you can add the compilation option `-DCMAKE_INSTALL_PREFIX=/path/to/faiss/install` to specify the installation path `/path/to/faiss/install`. The default installation path is `/usr/local`.
    >- The compilation option `-DMKL_LIBRARIES` must be set to the installation path of OpenBLAS in step [5](#li880635723510).
    >- If the message "CMake 3.23.1 or higher is required.  You are running version 3.22.0" is displayed, modify line 21 in the `/path/to/faiss/CMakeLists.txt` file by changing `cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)` to `cmake_minimum_required(VERSION 3.22.0 FATAL_ERROR)`.

## Compatibility Verification

This section describes how to verify the compatibility of the open-source Faiss with the Kunpeng platform. The example uses the `sift-128-euclidean.hdf5` dataset, Faiss-supported algorithm (HNSW), and 32 threads.

**Obtaining the Dataset and Test Program <a name="section5124167418"></a>**

1. Obtain the [test program](https://atomgit.com/openeuler/sra_test.git). The branch is <code>v2.0.0</code>. Assume that the program runs at the <code>/path/to/sra_test</code> directory. The full directory structure is as follows:

    ```text
    ├── configs                                                   // Stores configuration files for the algorithm and dataset.
          └── hnsw
                └── hnsw_sift-128-euclidean.config 
    ├── include                                                   // Stores header files of the test framework.
          └── algo                                                // Algorithm index definitions
          └── core                                                // Header files for data processing and test result processing
          └── framework                                           // Header files related to the test framework
    ├── src                                                       // Stores source files of the test framework.
          └── algo                                                // Algorithm adaptation layers
          └── bench                                               // Centralized test file
          └── core                                                // Files for data processing and test result processing
          └── registry                                            // Algorithm factory registry
    ├── Makefile                                                  // Script for compiling the program.
    └── test.sh                                                 // Script for running the test
    ├── test_muti-numas.sh                                        // Script for running the parallel test
    ├── data                                                      // Directory for storing datasets (You need to manually create and store datasets.)
          └── sift-128-euclidean.hdf5
    ├── indexes
          └── hnsw                                                // Stores manually built index.
                └── sift.faiss                                    // Built index, which is generated when the executable file hnsw_test runs and the save_or_load parameter in the dataset configuration file is set to save.
    └── hnsw_test                                                // Executable file generated after compilation
    ```

2. Obtain the dataset and save it to `/path/to/sra_test/data`.

    ```bash
    cd /path/to/sra_test/data
    wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 --no-check-certificate
    ```

**Verifying the Compatibility with Open-Source Faiss<a name="section41250624115"></a>**

1. Install the dependencies.

    ```bash
    yum install hdf5 hdf5-devel numactl numactl-devel
    ```

2. Compile and install Faiss. Note that for open-source Faiss compatibility verification, you do not need to enable macros related to Kunpeng optimization.
3. Build the executable file. Enter the Faiss installation path and the paths to other required dependencies as prompted by the command line.

    ```bash
    make hnsw_test
    ```

    >![note](public_sys-resources/icon-note.gif) **NOTE:**
    >During the test, select the appropriate compilation instruction for each algorithm.
    >- HNSW: `make hnsw_test`
    >- PQFS: `make pqfs_test`
    >- IVFPQ: `make ivfpq_test`
    >- IVFPQFS: `make ivfpqfs_test`
    >- IVFFLAT: `make ivfflat_test`

4. For the first run, ensure that `save_or_load` in the `hnsw_sift-128-euclidean.config` file is set to `save`. In subsequent runs, you can change it to `load` to use the built graph index or retriever for querying.
5. Run the executable file. Add the OpenBLAS and Faiss dynamic library paths to the environment variable.

    ```bash
    numactl -C 0-31 -m 0 ./hnsw_test hnsw sift-128-euclidean
    ```

The command output is as follows:

<img src="figures/faiss-installation_guide.jpg" alt="faiss-installation_guide-command output" width="800"/>

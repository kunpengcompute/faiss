# 项目介绍<a name="ZH-CN_TOPIC_0000002442478428"></a>

本仓库用于存放针对鲲鹏平台进行优化的补丁，补丁文件可应用于由Facebook研发的Faiss开源代码。

# 环境部署<a name="ZH-CN_TOPIC_0000002442489280"></a>

## 已验证环境<a name="ZH-CN_TOPIC_0000002476009237"></a>

<a name="table11553123194615"></a>
<table><thead align="left"><tr id="row1955333144612"><th class="cellrowborder" valign="top" width="20%" id="mcps1.1.6.1.1"><p id="p145531431174615"><a name="p145531431174615"></a><a name="p145531431174615"></a>操作系统</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.6.1.2"><p id="p455313316464"><a name="p455313316464"></a><a name="p455313316464"></a>CPU类型</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.6.1.3"><p id="p2055318318466"><a name="p2055318318466"></a><a name="p2055318318466"></a>内存</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.6.1.4"><p id="p25531231134616"><a name="p25531231134616"></a><a name="p25531231134616"></a>编译器</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.6.1.5"><p id="p12553153118462"><a name="p12553153118462"></a><a name="p12553153118462"></a>其他依赖</p>
</th>
</tr>
</thead>
<tbody><tr id="row13553153110469"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.1 "><p id="p631524754610"><a name="p631524754610"></a><a name="p631524754610"></a>openEuler 24.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.2 "><p id="p18315184774614"><a name="p18315184774614"></a><a name="p18315184774614"></a>鲲鹏920 7592C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.3 "><p id="p83151347184614"><a name="p83151347184614"></a><a name="p83151347184614"></a>24 * 64G</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="20%" headers="mcps1.1.6.1.4 "><p id="p831564711465"><a name="p831564711465"></a><a name="p831564711465"></a>GCC 12.3.1</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="20%" headers="mcps1.1.6.1.5 "><p id="p13315124713463"><a name="p13315124713463"></a><a name="p13315124713463"></a>CMake&gt;=3.22.0</p>
</td>
</tr>
<tr id="row47712054131620"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.3 "><p id="p631524754611"><a name="p631524754611"></a><a name="p631524754611"></a>openEuler 22.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.6 "><p id="p18315184774615"><a name="p18315184774615"></a><a name="p18315184774615"></a>鲲鹏920 7282C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.3 "><p id="p83151347184615"><a name="p83151347184615"></a><a name="p83151347184615"></a>16 * 32G</p>
</td>
</tr>
</tbody>
</table>

# 快速上手<a name="ZH-CN_TOPIC_0000002442649136"></a>


## Faiss编译<a name="ZH-CN_TOPIC_0000002476009241"></a>

1.  获取开源Faiss代码，假设代码存放于“/path/to/faiss-1.8.0“。

    ```
    git clone --branch v1.8.0 --single-branch https://github.com/facebookresearch/faiss.git
    ```

2.  安装Make、CMake、GCC。

    ```
    yum install make cmake gcc-toolset-12-gcc*
    export PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
    export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64/:$LD_LIBRARY_PATH
    ```

3.  Faiss依赖数学库，从Github仓下载开源OpenBLAS源代码，标签为v0.3.29。保存在编译机器可访问的路径中，假设位于“/path/to/OpenBLAS-0.3.29“。

    ```
    git clone --branch v0.3.29 --single-branch https://github.com/OpenMathLib/OpenBLAS.git
    cd OpenBLAS
    make
    make install
    # 可通过make install PREFIX=/path/to/openblas/install设置/path/to/openblas/install以指定安装路径，默认安装路径为/opt/OpenBLAS。
    ```

4.  安装补丁文件，以0001-faiss_1.8.0-optimize-neq.patch为例。

    ```
    cd /path/to/faiss-1.8.0/faiss
    patch -p1 < 0001-faiss_1.8.0-optimize-neq.patch
    ```

5.  编译Faiss代码获取libfaiss.so。
    GCC编译：

    ```
    cd /path/to/faiss-1.8.0/faiss
    cmake -B build . \
      -DFAISS_ENABLE_GPU=OFF \
      -DBUILD_TESTING=OFF \
      -DOPTI_IVFPQ=OFF \
      -DKRL=0N \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DFAISS_OPT_LEVEL=generic \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DMKL_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so
    make -C build -j faiss
    make -C build install
    ```
    LLVM编译：
    ```
    cd /path/to/faiss-1.8.0/faiss
    cmake -B build . \
      -DFAISS_ENABLE_GPU=OFF \
      -DBUILD_TESTING=OFF \
      -DOPTI_IVFPQ=OFF \
      -DKRL=0N \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DCMAKE_INSTALL_PREFIX=/path/to/faiss/install-llvm-gomp \
      -DMKL_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so \
      -DOpenMP_C_FLAGS="-fopenmp=libgomp" \
      -DOpenMP_CXX_FLAGS="-fopenmp=libgomp" \
      -DOpenMP_C_LIB_NAMES="gomp" \
      -DOpenMP_CXX_LIB_NAMES="gomp" \
      -DOpenMP_gomp_LIBRARY=/usr/lib/gcc/aarch64-linux-gnu/12/libgomp.so
    cmake --build build --parallel
    cmake --install build
    ```
    >![](public_sys-resources/icon-note.gif) **说明：**
    >-   可通过在编译时添加编译选项-DCMAKE_INSTALL_PREFIX=/path/to/faiss/install设置/path/to/faiss/install以指定安装路径，默认安装路径为/usr/local。
    >-   编译选项-DMKL_LIBRARIES需指定为OpenBLAS的实际安装路径。
    >-   编译选项-DKRL和-DOPTI_IVFPQ用于指定是否开启KRL和OPTI_IVFPQ优化选项，二者不可同时开启。
    >-   0002-faiss_1.8.0-optimize-eqv.patch不包含OPTI_IVFPQ优化选项，仅支持KRL。


## 测试示例<a name="ZH-CN_TOPIC_0000002475969077"></a>

下方使用示例以使用sift-128-euclidean.hdf5数据集，Faiss\(HNSW\)算法，线程数32为例。

1.  获取测试程序。

    ```
    git clone https://gitcode.com/openeuler/sra_test.git
    ```

2.  创建data文件夹，获取测试数据。

    ```
    cd /path/to/sra_test
    mkdir data && cd data
    wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 --no-check-certificate
    ```

    完整的目录结构应如下所示：

    ```
    ├── configs                                                   // 存放对应算法、对应数据集配置文件
          └── hnsw
                └── hnsw_sift-128-euclidean.config
    ├── include                                                   // 存放测试框架对应的头文件
    ├── src                                                       // 存放测试框架对应的源文件
    ├── Makefile                                                  // 编译脚本文件
    ├── scripts
          └── build.sh                                            // 脚本文件
    ├── test.sh                                                   // 测试脚本
    ├── data                                                      // 存放数据集
          └── sift-128-euclidean.hdf5
    ├── indexes
          └── hnsw                                                // 存放构建好的索引，需手动创建。
                └── sift.faiss                                    // 构建好的索引，运行可执行文件hnsw_test后（对应数据集配置文件“save_or_load”为save）时生成
    └── hnsw_test                                                 // 编译后生成的可执行文件
    ```

3.  安装相关依赖。

    ```
    yum install hdf5 hdf5-devel numactl numactl-devel
    ```

4.  编译运行程序。请根据实际安装Faiss的路径，修改Makefile中FAISSROOT项。

    ```
    make hnsw_test
    # 目前可通过安装patch的形式对Faiss原生的HNSW、PQFS、IVFPQ、IVFPQFS、IVFFLAT等算法进行加速，测试时选择不同的编译指令：
    # HNSW：make hnsw_test
    # HNSW-FP16: make hnsw_fp16_test
    # PQFS：make pqfs_test
    # IVFPQ：make ivfpq_test
    # IVFPQFS：make ivfpqfs_test
	# IVFFLAT: make ivfflat_test
    ```

5.  执行测试。可根据需求调整test.sh中的数据集和绑核。

    ```
    sh test.sh hnsw
    ```
    >![](public_sys-resources/icon-note.gif) **说明：**
    >-   若是第一次执行，确保hnsw_sift-128-euclidean.config文件夹中的"save_or_load"为"save"；后续执行可改为"load"，使用构件好的图索引查询。
    >-   首次编译时，根据命令行提示交互输入对应算法动态库路径与头文件路径。
    >-   脚本将在build文件夹下自动保存对应算法所需动态库与头文件路径，后续编译无需命令行交互输入路径，可以直接修改build文件夹下config_faiss_aarch64.sh中的对应配置，再运行make hnsw_test即可。

# 贡献指南<a name="ZH-CN_TOPIC_0000002442489288"></a>

如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交issues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。

# 许可证书<a name="ZH-CN_TOPIC_0000002476009245"></a>

采用 MIT License 许可证授权，支持修改代码和再开源。

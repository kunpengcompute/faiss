# 项目介绍<a name="ZH-CN_TOPIC_0000002442478428"></a>

KRL（Kunpeng Retrieval Library）是华为提供的基于鲲鹏平台优化的用于加速向量检索的算子库。 KRL通过针对鲲鹏处理器的指令集架构与内存访问机制进行底层优化，使用低精度量化+高精度重排等方法提升算子性能，可以有效提升召回算法的计算效率与吞吐量，同时保证算法精度，尤其适用于高并发召回场景。目前已经提供了Faiss中HNSW、PQFS、IVFPQ、IVFPQFS等算法的加速样例。 KRL适用于鲲鹏920 7282C处理器，支持NEON指令（128位宽）和SVE指令（256位宽）。

KRL核心特点：

-   鲲鹏亲和优化：针对硬件参数调整鲲鹏SIMD指令选取与排布，减少总指令时延。
-   寄存器cache优化：通过将频繁使用的数据放置在寄存器cache中降低数据读取时延，提高访存速度。
-   内存排布优化：通过对乘积量化索引进行重排，使用索引时更加连续；对底库向量进行重排，降低cache miss率，提高访存速度。
-   多阶段重排优化：通过判断底库向量是否接近阈值，为不同的底库向量设计不同的距离计算方法，以提升计算速度。

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
<tbody><tr id="row13553153110469"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.1 "><p id="p631524754610"><a name="p631524754610"></a><a name="p631524754610"></a>openEuler 22.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.2 "><p id="p18315184774614"><a name="p18315184774614"></a><a name="p18315184774614"></a>鲲鹏920 7282C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.3 "><p id="p83151347184614"><a name="p83151347184614"></a><a name="p83151347184614"></a>16 * 32G</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.4 "><p id="p831564711465"><a name="p831564711465"></a><a name="p831564711465"></a>GCC 12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.6.1.5 "><p id="p13315124713463"><a name="p13315124713463"></a><a name="p13315124713463"></a>CMake&gt;=3.22.0</p>
</td>
</tr>
</tbody>
</table>

## 源码编译<a name="ZH-CN_TOPIC_0000002475969069"></a>

1.  安装环境依赖。

    ```
    yum install cmake make
    ```

2.  安装GCC 12.3.1。

    ```
    yum install gcc-toolset-12-gcc*
    export PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
    export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64/:$LD_LIBRARY_PATH
    ```

3.  执行编译。 假设仓库源代码存放于“/path/to/krl“，通过源码编译得到libkrl.so。生成的so位于“/path/to/krl/out/lib“目录下。

    ```
    cd /path/to/krl
    sh build_lib.sh
    ```

# 快速上手<a name="ZH-CN_TOPIC_0000002442649136"></a>





## UT测试（可选）<a name="ZH-CN_TOPIC_0000002442489284"></a>

执行下方指令，用于看护编译及运行环境是否正常，算子功能是否正确。

```
cd /path/to/krl/test
sh run_test.sh
```

## Faiss使能KRL<a name="ZH-CN_TOPIC_0000002476009241"></a>

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

4.  安装补丁文件0001-faiss-1.8.0-add-krl.patch。

    ```
    cd /path/to/faiss-1.8.0/faiss
    patch -p1 < 0001-faiss-1.8.0-add-krl.patch
    ```

5.  编译Faiss代码获取libfaiss.so。

    ```
    export KRL_PATH=/path/to/KRL/out
    cd /path/to/faiss-1.8.0
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
    # 环境变量KRL_PATH需指定为libkrl.so的实际存放路径。
    # 可通过在编译时添加编译选项-DCMAKE_INSTALL_PREFIX=/path/to/faiss/install设置/path/to/faiss/install以指定安装路径，默认安装路径为/usr/local。
    # 编译选项-DMKL_LIBRARIES需指定为OpenBLAS的实际安装路径。
    ```

## 测试示例<a name="ZH-CN_TOPIC_0000002475969077"></a>

下方使用示例以使用sift-128-euclidean.hdf5数据集，Faiss\(HNSW\)算法，线程数32为例。

1.  获取数据集。

    ```
    wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 --no-check-certificate
    ```

2.  获取测试程序。

    ```
    git clone --branch v1.3.0 --single-branch https://gitee.com/openeuler/sra_test.git
    ```

    完整的目录结构应如下所示：

    ```
    ├── configs                                                   // 存放对应算法、对应数据集配置文件
          └── hnsw
                └── hnsw_sift-128-euclidean.config
    ├── include                                                   // 存放测试框架对应的头文件
    ├── src                                                       // 存放测试框架对应的源文件
    ├── Makefile                                                  // 编译脚本文件
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
    export KRL_PATH=/path/to/KRL/out
    make hnsw_test
    # 环境变量KRL_PATH需指定为libkrl.so的实际存放路径。
    # 目前可通过使能KRL算子的形式对Faiss原生的HNSW、PQFS、IVFPQ、IVFPQFS等算法进行加速，测试时选择不同的编译指令：
    # HNSW：make hnsw
    # PQFS：make pqfs
    # IVFPQ：make ivfpq
    # IVFPQFS：make ivfpqfs
    ```

5.  执行测试。

    ```
    numactl -C 0-31 -m 0 ./hnsw_test hnsw sift-128-euclidean
    ```

## API参考<a name="ZH-CN_TOPIC_0000002442649140"></a>

<a name="table6260559191515"></a>
<table><thead align="left"><tr id="row1926165914150"><th class="cellrowborder" valign="top" width="13.91139113911391%" id="mcps1.1.4.1.1"><p id="p6261259191513"><a name="p6261259191513"></a><a name="p6261259191513"></a>接口类型</p>
</th>
<th class="cellrowborder" valign="top" width="27.622762276227625%" id="mcps1.1.4.1.2"><p id="p192611459131510"><a name="p192611459131510"></a><a name="p192611459131510"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="58.465846584658465%" id="mcps1.1.4.1.3"><p id="p62611159161515"><a name="p62611159161515"></a><a name="p62611159161515"></a>接口作用</p>
</th>
</tr>
</thead>
<tbody><tr id="row9261155941516"><td class="cellrowborder" rowspan="7" valign="top" width="13.91139113911391%" headers="mcps1.1.4.1.1 "><p id="p1526145901517"><a name="p1526145901517"></a><a name="p1526145901517"></a>Handle类接口</p>
<p id="p11261155914157"><a name="p11261155914157"></a><a name="p11261155914157"></a></p>
<p id="p14261125912157"><a name="p14261125912157"></a><a name="p14261125912157"></a></p>
<p id="p1226185918157"><a name="p1226185918157"></a><a name="p1226185918157"></a></p>
<p id="p226185981520"><a name="p226185981520"></a><a name="p226185981520"></a></p>
<p id="p142615597151"><a name="p142615597151"></a><a name="p142615597151"></a></p>
<p id="p6261259141515"><a name="p6261259141515"></a><a name="p6261259141515"></a></p>
</td>
<td class="cellrowborder" valign="top" width="27.622762276227625%" headers="mcps1.1.4.1.2 "><p id="p21793162165"><a name="p21793162165"></a><a name="p21793162165"></a>krl_create_distance_handle</p>
</td>
<td class="cellrowborder" valign="top" width="58.465846584658465%" headers="mcps1.1.4.1.3 "><p id="p51791516181614"><a name="p51791516181614"></a><a name="p51791516181614"></a>初始化构建一个KRLDistanceHandle实例，稠密距离计算时使用。</p>
</td>
</tr>
<tr id="row226145912155"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p171808168169"><a name="p171808168169"></a><a name="p171808168169"></a>krl_create_reorder_handle</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p8180141617160"><a name="p8180141617160"></a><a name="p8180141617160"></a>初始化构建一个KRLDistanceHandle实例，重排计算时使用。</p>
</td>
</tr>
<tr id="row162615593151"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p91804160168"><a name="p91804160168"></a><a name="p91804160168"></a>krl_clean_distance_handle</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p10180131651619"><a name="p10180131651619"></a><a name="p10180131651619"></a>析构KRLDistanceHandle实例，释放内存空间。</p>
</td>
</tr>
<tr id="row1126145991515"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p518013164165"><a name="p518013164165"></a><a name="p518013164165"></a>krl_create_LUT8b_handle</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p101801616131612"><a name="p101801616131612"></a><a name="p101801616131612"></a>初始化构建一个KRLLUT8bHandle实例，8bit查表累和时使用。</p>
</td>
</tr>
<tr id="row6261559151516"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p81800166165"><a name="p81800166165"></a><a name="p81800166165"></a>krl_clean_LUT8b_handle</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p151803169164"><a name="p151803169164"></a><a name="p151803169164"></a>析构KRLLUT8bHandle实例，释放内存空间。</p>
</td>
</tr>
<tr id="row4261145921517"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p171801216111610"><a name="p171801216111610"></a><a name="p171801216111610"></a>krl_get_idx_pointer</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p16180121617169"><a name="p16180121617169"></a><a name="p16180121617169"></a>获取KRLLUT8bHandle实例中存储的需要计算的底库向量ID。</p>
</td>
</tr>
<tr id="row6261155916159"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p19180416121619"><a name="p19180416121619"></a><a name="p19180416121619"></a>krl_get_dist_pointer</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p19180111671617"><a name="p19180111671617"></a><a name="p19180111671617"></a>获取KRLLUT8bHandle实例中存储的距离数组首地址，在调用查表累和计算算子之前为随机数，调用之后为计算得到的距离。</p>
</td>
</tr>
<tr id="row92613596156"><td class="cellrowborder" rowspan="21" valign="top" width="13.91139113911391%" headers="mcps1.1.4.1.1 "><p id="p526165911153"><a name="p526165911153"></a><a name="p526165911153"></a>距离计算接口</p>
<p id="p18261155914157"><a name="p18261155914157"></a><a name="p18261155914157"></a></p>
<p id="p1026115915153"><a name="p1026115915153"></a><a name="p1026115915153"></a></p>
<p id="p3261135931516"><a name="p3261135931516"></a><a name="p3261135931516"></a></p>
<p id="p1726165919155"><a name="p1726165919155"></a><a name="p1726165919155"></a></p>
<p id="p17262759131510"><a name="p17262759131510"></a><a name="p17262759131510"></a></p>
<p id="p17262125918157"><a name="p17262125918157"></a><a name="p17262125918157"></a></p>
<p id="p626295913156"><a name="p626295913156"></a><a name="p626295913156"></a></p>
<p id="p16262205915152"><a name="p16262205915152"></a><a name="p16262205915152"></a></p>
<p id="p16262125951516"><a name="p16262125951516"></a><a name="p16262125951516"></a></p>
<p id="p2262115914154"><a name="p2262115914154"></a><a name="p2262115914154"></a></p>
<p id="p326235919159"><a name="p326235919159"></a><a name="p326235919159"></a></p>
<p id="p11262459141516"><a name="p11262459141516"></a><a name="p11262459141516"></a></p>
<p id="p162622596159"><a name="p162622596159"></a><a name="p162622596159"></a></p>
<p id="p152621159191515"><a name="p152621159191515"></a><a name="p152621159191515"></a></p>
<p id="p3262159131519"><a name="p3262159131519"></a><a name="p3262159131519"></a></p>
<p id="p19262259121518"><a name="p19262259121518"></a><a name="p19262259121518"></a></p>
<p id="p826217598159"><a name="p826217598159"></a><a name="p826217598159"></a></p>
<p id="p1226255917155"><a name="p1226255917155"></a><a name="p1226255917155"></a></p>
<p id="p82620591154"><a name="p82620591154"></a><a name="p82620591154"></a></p>
<p id="p7262105916152"><a name="p7262105916152"></a><a name="p7262105916152"></a></p>
</td>
<td class="cellrowborder" valign="top" width="27.622762276227625%" headers="mcps1.1.4.1.2 "><p id="p2043816511162"><a name="p2043816511162"></a><a name="p2043816511162"></a>krl_L2sqr</p>
</td>
<td class="cellrowborder" valign="top" width="58.465846584658465%" headers="mcps1.1.4.1.3 "><p id="p1438175181620"><a name="p1438175181620"></a><a name="p1438175181620"></a>进行数据类型为float的一对一欧氏距离计算。</p>
</td>
</tr>
<tr id="row8261559191515"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p1443835114163"><a name="p1443835114163"></a><a name="p1443835114163"></a>krl_L2sqr_f16f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p94381551191617"><a name="p94381551191617"></a><a name="p94381551191617"></a>进行数据类型为fp16的一对一欧氏距离计算。</p>
</td>
</tr>
<tr id="row19261205913150"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p84381251191611"><a name="p84381251191611"></a><a name="p84381251191611"></a>krl_L2sqr_u8u32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p94381551191617_1"><a name="p94381551191617_1"></a><a name="p94381551191617_1"></a>进行数据类型为uint8的一对一欧氏距离计算。</p>
</td>
</tr>
<tr id="row122611159131518"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p543875191617"><a name="p543875191617"></a><a name="p543875191617"></a>krl_ipdis</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p143817518168"><a name="p143817518168"></a><a name="p143817518168"></a>进行数据类型为float的一对一内积距离计算。</p>
</td>
</tr>
<tr id="row112611859161519"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p17438165191618"><a name="p17438165191618"></a><a name="p17438165191618"></a>krl_negative_ipdis_f16f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p154381751151614"><a name="p154381751151614"></a><a name="p154381751151614"></a>进行数据类型为fp16的一对一内积距离计算。</p>
</td>
</tr>
<tr id="row626219595152"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p1543855181616"><a name="p1543855181616"></a><a name="p1543855181616"></a>krl_negative_ipdis_s8s32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p5438145120167"><a name="p5438145120167"></a><a name="p5438145120167"></a>进行数据类型为int8的一对一内积距离计算。</p>
</td>
</tr>
<tr id="row1726217597152"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p6438175121612"><a name="p6438175121612"></a><a name="p6438175121612"></a>krl_L2sqr_by_idx</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p104383515167"><a name="p104383515167"></a><a name="p104383515167"></a>进行数据类型为float的一对多欧氏距离计算。</p>
</td>
</tr>
<tr id="row326265917159"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p1743845121616"><a name="p1743845121616"></a><a name="p1743845121616"></a>krl_L2sqr_by_idx_f16f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p154381951171610"><a name="p154381951171610"></a><a name="p154381951171610"></a>进行数据类型为fp16的一对多欧氏距离计算。</p>
</td>
</tr>
<tr id="row15262125910159"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p44380512165"><a name="p44380512165"></a><a name="p44380512165"></a>krl_L2sqr_by_idx_u8f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p7438205110164"><a name="p7438205110164"></a><a name="p7438205110164"></a>进行数据类型为uint8的一对多欧氏距离计算。</p>
</td>
</tr>
<tr id="row132621359111511"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p12438195111167"><a name="p12438195111167"></a><a name="p12438195111167"></a>krl_inner_product_by_idx</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p2438165116163"><a name="p2438165116163"></a><a name="p2438165116163"></a>进行数据类型为float的一对多内积距离计算。</p>
</td>
</tr>
<tr id="row6262859191518"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p12439155161619"><a name="p12439155161619"></a><a name="p12439155161619"></a>krl_inner_product_by_idx_f16f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p7439195111169"><a name="p7439195111169"></a><a name="p7439195111169"></a>进行数据类型为fp16的一对多内积距离计算。</p>
</td>
</tr>
<tr id="row1026225901514"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p34399515166"><a name="p34399515166"></a><a name="p34399515166"></a>krl_negative_inner_product_by_idx_f16f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p943985131615"><a name="p943985131615"></a><a name="p943985131615"></a>进行数据类型为fp16的一对多内积距离计算，对结果取反。</p>
</td>
</tr>
<tr id="row3262115951510"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p20439145112161"><a name="p20439145112161"></a><a name="p20439145112161"></a>krl_inner_product_by_idx_s8f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p14391051171614"><a name="p14391051171614"></a><a name="p14391051171614"></a>进行数据类型为int8的一对多内积距离计算。</p>
</td>
</tr>
<tr id="row152627594154"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p743955111168"><a name="p743955111168"></a><a name="p743955111168"></a>krl_L2sqr_ny</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p043915191610"><a name="p043915191610"></a><a name="p043915191610"></a>进行数据类型为float的一对多欧氏距离计算。</p>
</td>
</tr>
<tr id="row9262185914155"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p0439351161617"><a name="p0439351161617"></a><a name="p0439351161617"></a>krl_L2sqr_ny_f16f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p1743935111617"><a name="p1743935111617"></a><a name="p1743935111617"></a>进行数据类型为fp16的一对多欧氏距离计算。</p>
</td>
</tr>
<tr id="row16262155941519"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p114391251121615"><a name="p114391251121615"></a><a name="p114391251121615"></a>krl_L2sqr_ny_u8f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p243916516166"><a name="p243916516166"></a><a name="p243916516166"></a>进行数据类型为uint8的一对多欧氏距离计算。</p>
</td>
</tr>
<tr id="row6262859171511"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p4439951121611"><a name="p4439951121611"></a><a name="p4439951121611"></a>krl_L2sqr_ny_with_handle</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p4439351141619"><a name="p4439351141619"></a><a name="p4439351141619"></a>进行数据类型为float的一对多欧氏距离计算，底库向量与维度存储于Handle中。</p>
</td>
</tr>
<tr id="row82621659141511"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p1343914517165"><a name="p1343914517165"></a><a name="p1343914517165"></a>krl_inner_product_ny</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p143975171620"><a name="p143975171620"></a><a name="p143975171620"></a>进行数据类型为float的一对多内积距离计算。</p>
</td>
</tr>
<tr id="row62621259161520"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p143935151610"><a name="p143935151610"></a><a name="p143935151610"></a>krl_inner_product_ny_f16f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p19439125113167"><a name="p19439125113167"></a><a name="p19439125113167"></a>进行数据类型为fp16的一对多内积距离计算。</p>
</td>
</tr>
<tr id="row826245919154"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p64391151201615"><a name="p64391151201615"></a><a name="p64391151201615"></a>krl_inner_product_ny_s8f32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p343965131619"><a name="p343965131619"></a><a name="p343965131619"></a>进行数据类型为int8的一对多内积距离计算。</p>
</td>
</tr>
<tr id="row1726295913159"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p6439125181612"><a name="p6439125181612"></a><a name="p6439125181612"></a>krl_inner_product_ny_with_handle</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p14439145116166"><a name="p14439145116166"></a><a name="p14439145116166"></a>进行数据类型为float的一对多内积距离计算，底库向量与维度存储于Handle中。</p>
</td>
</tr>
<tr id="row1526355910156"><td class="cellrowborder" rowspan="10" valign="top" width="13.91139113911391%" headers="mcps1.1.4.1.1 "><p id="p1326385911152"><a name="p1326385911152"></a><a name="p1326385911152"></a>查表累和接口</p>
<p id="p9263159141520"><a name="p9263159141520"></a><a name="p9263159141520"></a></p>
<p id="p1426313592152"><a name="p1426313592152"></a><a name="p1426313592152"></a></p>
<p id="p326335914151"><a name="p326335914151"></a><a name="p326335914151"></a></p>
<p id="p17263145941514"><a name="p17263145941514"></a><a name="p17263145941514"></a></p>
<p id="p026311598154"><a name="p026311598154"></a><a name="p026311598154"></a></p>
<p id="p13263195921510"><a name="p13263195921510"></a><a name="p13263195921510"></a></p>
<p id="p426325914156"><a name="p426325914156"></a><a name="p426325914156"></a></p>
<p id="p4263125981510"><a name="p4263125981510"></a><a name="p4263125981510"></a></p>
<p id="p20263115931511"><a name="p20263115931511"></a><a name="p20263115931511"></a></p>
</td>
<td class="cellrowborder" valign="top" width="27.622762276227625%" headers="mcps1.1.4.1.2 "><p id="p2165821161710"><a name="p2165821161710"></a><a name="p2165821161710"></a>krl_table_lookup_8b_f32</p>
</td>
<td class="cellrowborder" valign="top" width="58.465846584658465%" headers="mcps1.1.4.1.3 "><p id="p916522121712"><a name="p916522121712"></a><a name="p916522121712"></a>使用8bit索引在float类型表项中查询距离并累和，将累和结果加上dis0后存入distance。</p>
</td>
</tr>
<tr id="row726365961512"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p161659216171"><a name="p161659216171"></a><a name="p161659216171"></a>krl_table_lookup_8b_f32_by_idx</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p9165102101718"><a name="p9165102101718"></a><a name="p9165102101718"></a>使用8bit索引在float类型表项中查询距离并累和，将累和结果加上dis0后存入distance。ID在idx数组中出现的底库向量结果才会参与计算。</p>
</td>
</tr>
<tr id="row2263115916154"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p1816592116178"><a name="p1816592116178"></a><a name="p1816592116178"></a>krl_table_lookup_8b_f32_with_handle</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p17165102119179"><a name="p17165102119179"></a><a name="p17165102119179"></a>使用8bit索引在float类型表项中查询距离并累和，将累和结果加上dis0后存入distance。idx数组与distance数组被包含在KRLLUT8bHandle实例中，ID在idx数组中出现的底库向量结果才会参与计算。</p>
</td>
</tr>
<tr id="row126319593151"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p7165921161710"><a name="p7165921161710"></a><a name="p7165921161710"></a>krl_fast_table_lookup_step</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p1416542118179"><a name="p1416542118179"></a><a name="p1416542118179"></a>批量处理数据类型为float的查询向量的4bit查表累和过滤压缩算子。算子支持计算至多16个查询向量与32个底库向量间的距离。在距离计算完成后，将其与阈值进行比较，满足比较条件的底库向量的lt_mask对应位置将会设置为1，反之为0。</p>
</td>
</tr>
<tr id="row17263359191517"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p1016513217178"><a name="p1016513217178"></a><a name="p1016513217178"></a>krl_L2_table_lookup_fast_scan_bs64</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p20165102112179"><a name="p20165102112179"></a><a name="p20165102112179"></a>单独处理数据类型为float的查询向量的4bit查表累和过滤压缩算子。计算1个查询向量与64个底库向量间的欧氏距离。根据距离类型决定比较规则，将距离值满足比较规则的底库向量的lt_mask设置为1，反之为0。</p>
</td>
</tr>
<tr id="row1126315594152"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p21653219172"><a name="p21653219172"></a><a name="p21653219172"></a>krl_IP_table_lookup_fast_scan_bs64</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p1165142113176"><a name="p1165142113176"></a><a name="p1165142113176"></a>单独处理数据类型为float的查询向量的4bit查表累和过滤压缩算子。计算1个查询向量与64个底库向量间的内积距离。根据距离类型决定比较规则，将距离值满足比较规则的底库向量的lt_mask设置为1，反之为0。</p>
</td>
</tr>
<tr id="row15263859191518"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p016592115179"><a name="p016592115179"></a><a name="p016592115179"></a>krl_L2_table_lookup_fast_scan_bs96</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p151652213178"><a name="p151652213178"></a><a name="p151652213178"></a>单独处理数据类型为float的查询向量的4bit查表累和过滤压缩算子。计算1个查询向量与96个底库向量间的欧氏距离。根据距离类型决定比较规则，将距离值满足比较规则的底库向量的lt_mask设置为1，反之为0。</p>
</td>
</tr>
<tr id="row1326315971510"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p1416511219178"><a name="p1416511219178"></a><a name="p1416511219178"></a>krl_IP_table_lookup_fast_scan_bs96</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p1216614211179"><a name="p1216614211179"></a><a name="p1216614211179"></a>单独处理数据类型为float的查询向量的4bit查表累和过滤压缩算子。计算1个查询向量与96个底库向量间的内积距离。根据距离类型决定比较规则，将距离值满足比较规则的底库向量的lt_mask设置为1，反之为0。</p>
</td>
</tr>
<tr id="row14263165901510"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p16166821121719"><a name="p16166821121719"></a><a name="p16166821121719"></a>krl_table_lookup_4b_f16</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p12166721101719"><a name="p12166721101719"></a><a name="p12166721101719"></a>单独处理数据类型为fp16的查询向量的4bit查表累和过滤压缩算子。计算1个查询向量与多个底库向量间的内积距离，距离的初始值为dis_f16。此接口不会进行过滤压缩（与阈值进行比较）。</p>
</td>
</tr>
<tr id="row926335961515"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p216662112174"><a name="p216662112174"></a><a name="p216662112174"></a>krl_pack_codes_4b</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p181668217177"><a name="p181668217177"></a><a name="p181668217177"></a>单独处理数据类型为fp16的查询向量的4bit查表累和过滤压缩算子。计算1个查询向量与多个底库向量间的内积距离，距离的初始值为dis_f16。此接口不会进行过滤压缩（与阈值进行比较）。</p>
</td>
</tr>
<tr id="row32637591154"><td class="cellrowborder" rowspan="2" valign="top" width="13.91139113911391%" headers="mcps1.1.4.1.1 "><p id="p726315590155"><a name="p726315590155"></a><a name="p726315590155"></a>重排接口</p>
<p id="p12631059101519"><a name="p12631059101519"></a><a name="p12631059101519"></a></p>
</td>
<td class="cellrowborder" valign="top" width="27.622762276227625%" headers="mcps1.1.4.1.2 "><p id="p193628556177"><a name="p193628556177"></a><a name="p193628556177"></a>krl_reorder_2_vector</p>
</td>
<td class="cellrowborder" valign="top" width="58.465846584658465%" headers="mcps1.1.4.1.3 "><p id="p1236216553178"><a name="p1236216553178"></a><a name="p1236216553178"></a>计算1个查询向量与多个不连续底库向量间的高精度距离并排序。</p>
</td>
</tr>
<tr id="row5263159101516"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p17362555131714"><a name="p17362555131714"></a><a name="p17362555131714"></a>krl_reorder_2_vector_continuous</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p336311553175"><a name="p336311553175"></a><a name="p336311553175"></a>计算1个查询向量与多个连续底库向量间的高精度距离并排序。</p>
</td>
</tr>
<tr id="row11263185918154"><td class="cellrowborder" rowspan="4" valign="top" width="13.91139113911391%" headers="mcps1.1.4.1.1 "><p id="p8263759121518"><a name="p8263759121518"></a><a name="p8263759121518"></a>保存/加载接口</p>
<p id="p02631559101510"><a name="p02631559101510"></a><a name="p02631559101510"></a></p>
<p id="p12263115911511"><a name="p12263115911511"></a><a name="p12263115911511"></a></p>
<p id="p19264175921518"><a name="p19264175921518"></a><a name="p19264175921518"></a></p>
</td>
<td class="cellrowborder" valign="top" width="27.622762276227625%" headers="mcps1.1.4.1.2 "><p id="p1322109191810"><a name="p1322109191810"></a><a name="p1322109191810"></a>krl_store_LUT8Handle</p>
</td>
<td class="cellrowborder" valign="top" width="58.465846584658465%" headers="mcps1.1.4.1.3 "><p id="p10221491180"><a name="p10221491180"></a><a name="p10221491180"></a>krl_store_LUT8Handle</p>
</td>
</tr>
<tr id="row8263159101515"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p112259171813"><a name="p112259171813"></a><a name="p112259171813"></a>krl_build_LUT8Handle_fromfile</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p1122597186"><a name="p1122597186"></a><a name="p1122597186"></a>krl_build_LUT8Handle_fromfile</p>
</td>
</tr>
<tr id="row1426319595150"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p142311921810"><a name="p142311921810"></a><a name="p142311921810"></a>krl_store_distanceHandle</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p142349151819"><a name="p142349151819"></a><a name="p142349151819"></a>krl_store_distanceHandle</p>
</td>
</tr>
<tr id="row112643590157"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="p16238981819"><a name="p16238981819"></a><a name="p16238981819"></a>krl_build_distanceHandle_fromfile</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="p1423109161812"><a name="p1423109161812"></a><a name="p1423109161812"></a>krl_build_distanceHandle_fromfile</p>
</td>
</tr>
</tbody>
</table>

# 贡献指南<a name="ZH-CN_TOPIC_0000002442489288"></a>

如果使用过程中有任何问题，或者需要反馈特性需求和bug报告，可以提交issues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。

# 许可证书<a name="ZH-CN_TOPIC_0000002476009245"></a>

KRL采用 Apache 2.0 License 许可证授权，支持修改代码和再开源。


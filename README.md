# Faiss介绍<a name="ZH-CN_TOPIC_0000002518446558"></a>

## 最新消息<a name="ZH-CN_TOPIC_0000002550046405"></a>

- \[2026.03.30\]：Faiss提供全量优化补丁与等价优化补丁。其中，全量优化补丁针对IVFPQ算法进一步优化，新增支持HNSW FP16接口。
- \[2025.12.30\]：Faiss发布于Gitcode平台，实现IVFFLAT、IVFPQ、IVFPQFS、PQFS、HNSW优化。

## 项目介绍<a name="ZH-CN_TOPIC_0000002549926405"></a>

Faiss是由facebook开发的用于高效相似搜索和密集向量聚类的算法库，其核心采用C++编写，并为Python/numpy提供完整封装接口。Faiss提供IVFFlat、IVFPQ、HNSW、IVFPQFS、PQFS等索引方式。鲲鹏优化基于开源Faiss代码做侵入式修改，保持原有接口。

HNSW是Faiss提供的一种近似最近邻（ANN）图检索算法，开源Faiss\(HNSW\)接口支持FP32数据类型。为优化计算效率与内存占用，对原生faiss进行适配改造，增加FP16接口，使其在鲲鹏ARM架构下同样支持基于FP16的高效召回计算。

## 目录结构<a name="ZH-CN_TOPIC_0000002549926407"></a>

代码仓目录结构如下：

```text
faiss/
├─ 0001-faiss_1.8.0-optimize-neq.patch         // 全量优化补丁
├─ 0002-faiss_1.8.0-optimize-eqv.patch         // 等价优化补丁
└── docs
   ├── public_sys-resources
   ├── api_reference.md                        // API参考
   ├── feature_introduction.md                 // 特性介绍
   ├── best_practices.md                       // 最佳实现
   ├── installation_guide.md                   // 安装指南
   ├── quick_start.md                          // 快速上手
   ├── release_notes.md                        // 版本说明书
   └── LICENSE
```

使用补丁后Faiss完整的目录结构如下所示：

```text
faiss/
├─ benchs/                                     // 基准测试
├─ c_api/                                      // C语言API封装
├─ cmake/                                      // CMake配置模块
├─ conda/                                      // Conda构建脚本
├─ contrib/                                    // Python贡献模块
├─ demos/                                      // 示例程序
├─ faiss/
│   ├─ CMakeLists.txt                          // 构建配置
│   ├─ Index.h                                 // 抽象基类，统一接口
│   ├─ IndexFlat.cpp                           // 暴力搜索实现
│   ├─ IndexFlatCodes.h                        // 统一码存储基类（用于PQ、SQ 等）
│   ├─ IndexFlatCodes.cpp                      // 统一码存储基类实现
│   ├─ IndexFastScan.h                         // 4‑bit PQ/AQ快速扫描通用接口
│   ├─ IndexFastScan.cpp                       // 4‑bit PQ/AQ快速扫描通用实现
│   ├─ IndexIVF.h                              // IVF基类接口
│   ├─ IndexIVF.cpp                            // IVF基类+具体实现
│   ├─ IndexIVFFlat.cpp                        // IVFFlat具体实现
│   ├─ IndexIVFPQ.cpp                          // IVFPQ实现
│   ├─ IndexIVFFastScan.h                      // IVFPQFastScan接口
│   ├─ IndexIVFFastScan.cpp                    // IVFPQFastScan（CPU）实现
│   ├─ IndexHNSW.h                             // HNSW索引接口
│   ├─ IndexHNSW.cpp                           // HNSW索引实现
│   ├─ IndexRefine.h                           // 基准+细化组合索引接口
│   ├─ IndexRefine.cpp                         // 基准+细化组合索引实现
│   ├─ impl/
│   │   ├─ DistanceComputer.h                  // 距离计算抽象接口
│   │   ├─ ProductQuantizer.h                  // 乘积量化器接口
│   │   ├─ ProductQuantizer.cpp                // 乘积量化器实现
│   │   ├─ pq4_fast_scan.h                     // 4‑bit PQ快速扫描接口
│   │   ├─ pq4_fast_scan_search_1.cpp          // 4‑bit PQ快速扫描单查询实现
│   │   ├─ pq4_fast_scan_search_qbs.cpp        // 4‑bit PQ快速扫描批量查询实现
│   │   ├─ HNSW.cpp                            // HNSW图结构实现
│   │   ├─ index_read.cpp                      // 索引反序列化实现
│   │   └─ simd_result_handlers.h              // SIMD结果处理器
│   ├─ invlists/
│   │   ├─ InvertedLists.h                     // 倒排列表抽象接口
│   │   └─ InvertedLists.cpp                   // 倒排列表实现
│   ├─ utils/
│   │   └─ distances_simd.cpp                  // SIMD L2/IP/L1/Linf实现
│   ├─ sra_krl/
│   │   ├─ include/
│   │   │   ├─ krl.h                           // 对外统一API声明
│   │   │   ├─ krl_internal.h                  // 内部结构体、宏、SIMD辅助实现
│   │   │   ├─ platform_macros.h               // 错误码、度量常量、平台宏
│   │   │   └─ safe_memory.h                   // 安全内存操作
│   │   └─ src/
│   │       ├─ Heap_sort.c                     // Top‑K堆构建、双堆重排实现
│   │       ├─ IPdistance_simd.c               // 单精度向量内积SIMD实现（batch 2/4/8/16）
│   │       ├─ IPdistance_simd_f16.c           // float16 IP距离计算实现
│   │       ├─ IPdistance_simd_f16f32.c        // float16 IP距离计算实现（float输出）
│   │       ├─ IPdistance_simd_s8.c            // int8 IP距离计算实现（int32/float输出）
│   │       ├─ L2distance_simd.c               // float L2距离计算实现（batch 2/4/8/16/24）
│   │       ├─ L2distance_simd_f16.c           // float16 L2距离计算实现
│   │       ├─ L2distance_simd_f16f32.c        // float16 L2距离计算实现（float输出）
│   │       ├─ L2distance_simd_u8.c            // uint8 L2距离计算实现（uint32/float输出）
│   │       ├─ matrix_block_transpose.c        // 4×4块转置kernel
│   │       ├─ MinMax_quant.c                  // 量化（fp16/u8/s8）
│   │       ├─ NegaIPdistance_simd_f16f32.c    // float16 IP距离计算实现（取反，float输出）
│   │       ├─ NegaIPdistance_simd_s8.c        // int8 IP距离计算实现（取反，int32/float输出）
│   │       ├─ handle_IO.c                     // 句柄序列化/反序列化（文件I/O）
│   │       ├─ krl_handles.c                   // 句柄创建、初始化、清理、指针访问
│   │       ├─ pq_search_with_table_4bit.c     // 4‑bit查表
│   │       ├─ pq_search_with_table_8bit.c     // 8‑bit查表
│   │       ├─ reorder_2_vectors.c             // 稀疏/连续重排
│   │       └─ sve_search_codes.c              // 4‑bit fp16查表（sve）
│   ├─ cppcontrib/                             // C++贡献模块
│   ├─ gpu/                                    // GPU子系统
│   └─ python/                                 // Python绑定
├─ misc/                                       // 杂项测试
├─ tests/                                      // 单元测试
├─ tutorial/                                   // 教程示例
├─ CMakeLists.txt                              // 顶层构建配置
├─ CHANGELOG.md
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
├─ INSTALL.md
├─ LICENSE
└─ README.md
```

## 版本说明<a name="ZH-CN_TOPIC_0000002549926409"></a>

关于Faiss的版本更新情况请参见[《Faiss版本说明书》](./docs/release_notes.md)。

## 学习文档<a name="ZH-CN_TOPIC_0000002518286646"></a>

<a name="table1191773710200"></a>
<table><thead align="left"><tr id="row1291816372202"><th class="cellrowborder" valign="top" width="9.780978097809781%" id="mcps1.1.4.1.1"><p id="p291823714205"><a name="p291823714205"></a><a name="p291823714205"></a>学习资源类别</p>
</th>
<th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.2"><p id="p13918183762016"><a name="p13918183762016"></a><a name="p13918183762016"></a>学习资源名称</p>
</th>
<th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.3"><p id="p89181437152019"><a name="p89181437152019"></a><a name="p89181437152019"></a>学习资源简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row179181137112015"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p1918123710208"><a name="p1918123710208"></a><a name="p1918123710208"></a>文档</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p2091893722011"><a name="p2091893722011"></a><a name="p2091893722011"></a><a href="./docs/release_notes.md">版本说明书</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p491893752010"><a name="p491893752010"></a><a name="p491893752010"></a>提供Faiss每个发布版本的基础信息和特性更新信息。</p>
</td>
</tr>
<tr id="row939116371143"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p1039163711413"><a name="p1039163711413"></a><a name="p1039163711413"></a>文档</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p03913372046"><a name="p03913372046"></a><a name="p03913372046"></a><a href="./docs/quick_start.md">快速入门</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p1139217371746"><a name="p1139217371746"></a><a name="p1139217371746"></a>提供Faiss快速入门指导。</p>
</td>
</tr>
<tr id="row2918153732017"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p598512211214"><a name="p598512211214"></a><a name="p598512211214"></a>文档</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172020"><a name="p17918337172020"></a><a name="p17918337172020"></a><a href="./docs/installation_guide.md">安装指南</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742018"><a name="p15918183742018"></a><a name="p15918183742018"></a>提供Faiss编译安装方法指导。</p>
</td>
</tr>
<tr id="row2918153732018"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p598512211215"><a name="p598512211215"></a><a name="p598512211215"></a>文档</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172021"><a name="p17918337172021"></a><a name="p17918337172021"></a><a href="./docs/api_reference.md">API参考</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742019"><a name="p15918183742019"></a><a name="p15918183742019"></a>提供Faiss新增API接口定义、接口说明。</p>
</td>
</tr>
<tr id="row2918153732019"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p598512211216"><a name="p598512211216"></a><a name="p598512211216"></a>文档</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172022"><a name="p17918337172022"></a><a name="p17918337172022"></a><a href="./docs/best_practices.md">最佳实践</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742020"><a name="p15918183742020"></a><a name="p15918183742020"></a>提供Faiss使用的实践案例。</p>
</td>
</tr>
<tr id="row2918153732020"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p598512211217"><a name="p598512211217"></a><a name="p598512211217"></a>文档</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172023"><a name="p17918337172023"></a><a name="p17918337172023"></a><a href="./docs/feature_introduction.md">特性介绍</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742021"><a name="p15918183742021"></a><a name="p15918183742021"></a>提供Faiss架构介绍及优化说明。</p>
</td>
</tr>
</tbody>
</table>

## 免责声明<a name="ZH-CN_TOPIC_0000002518286638"></a>

此代码仓计划参与Faiss开源组件，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## License<a name="ZH-CN_TOPIC_0000002550046411"></a>

Faiss采用MIT License许可证授权，支持修改代码和再开源，具体请参见[LICENSE](./LICENSE)文件。

本项目的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

## 贡献声明<a name="ZH-CN_TOPIC_0000002550046409"></a>

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 致谢<a name="ZH-CN_TOPIC_0000002522434246"></a>

Faiss由华为公司的下列部门联合贡献：

- 鲲鹏计算Boostkit开发部

感谢来自社区的每一个PR，欢迎贡献Faiss！

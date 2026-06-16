# Faiss介绍

## 最新消息

- \[2026.06.30\]：优化VisitedTable访问标记，采用类hashset的generation-based标记替代全量memset重置，将VisitedTable更新从O(N)优化至O(1)。4bit查表算子增加SVE2实现。
- \[2026.03.30\]：Faiss提供全量优化补丁与等价优化补丁。其中，全量优化补丁针对IVFPQ算法进一步优化，新增支持HNSW FP16接口。
- \[2025.12.30\]：Faiss发布于Gitcode平台，实现IVFFLAT、IVFPQ、IVFPQFS、PQFS、HNSW优化。

## 项目介绍

Faiss是由facebook开发的用于高效相似搜索和密集向量聚类的算法库，其核心采用C++编写，并为Python/numpy提供完整封装接口。Faiss提供IVFFlat、IVFPQ、HNSW、IVFPQFS、PQFS等索引方式。鲲鹏优化基于开源Faiss代码做侵入式修改，保持原有接口。

HNSW是Faiss提供的一种近似最近邻（ANN）图检索算法，开源Faiss\(HNSW\)接口支持FP32数据类型。为优化计算效率与内存占用，对原生faiss进行适配改造，增加FP16接口，使其在鲲鹏ARM架构下同样支持基于FP16的高效召回计算。

## 目录结构

代码仓目录结构如下：

```text
faiss/
├─ 0001-faiss_1.8.0-optimize-neq.patch         // 全量优化补丁
├─ 0002-faiss_1.8.0-optimize-eqv.patch         // 等价优化补丁
└── docs
   ├── LICENSE
   └── zh
      ├── api_reference.md                        // API参考
      ├── feature_introduction.md                 // 特性介绍
      ├── best_practices.md                       // 最佳实践
      ├── installation_guide.md                   // 安装指南
      ├── quick_start.md                          // 快速入门
      └── release_notes.md                        // 版本说明书
```

## 版本说明

关于Faiss的版本更新情况请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

<a name="table1191773710200"></a>
<table><thead align="left"><tr id="row1291816372202"><th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.2"><p id="p13918183762016"><a name="p13918183762016"></a><a name="p13918183762016"></a>学习资源名称</p>
</th>
<th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.3"><p id="p89181437152019"><a name="p89181437152019"></a><a name="p89181437152019"></a>学习资源简介</p>
</th>
</tr>
</thead>
<tbody><tr id="row2918153732020"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172023"><a name="p17918337172023"></a><a name="p17918337172023"></a><a href="./docs/zh/feature_introduction.md">特性介绍</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742021"><a name="p15918183742021"></a><a name="p15918183742021"></a>提供Faiss架构介绍及优化说明。</p>
</td>
</tr>
<tr id="row179181137112015"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p2091893722011"><a name="p2091893722011"></a><a name="p2091893722011"></a><a href="./docs/zh/release_notes.md">版本说明书</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p491893752010"><a name="p491893752010"></a><a name="p491893752010"></a>提供Faiss每个发布版本的基础信息和特性更新信息。</p>
</td>
</tr>
<tr id="row939116371143"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p03913372046"><a name="p03913372046"></a><a name="p03913372046"></a><a href="./docs/zh/quick_start.md">快速入门</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p1139217371746"><a name="p1139217371746"></a><a name="p1139217371746"></a>提供Faiss快速入门指导。</p>
</td>
</tr>
<tr id="row2918153732017"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172020"><a name="p17918337172020"></a><a name="p17918337172020"></a><a href="./docs/zh/installation_guide.md">安装指南</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742018"><a name="p15918183742018"></a><a name="p15918183742018"></a>提供Faiss编译安装方法指导。</p>
</td>
</tr>
<tr id="row2918153732018"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172021"><a name="p17918337172021"></a><a name="p17918337172021"></a><a href="./docs/zh/api_reference.md">API参考</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742019"><a name="p15918183742019"></a><a name="p15918183742019"></a>提供Faiss新增API接口定义、接口说明。</p>
</td>
</tr>
<tr id="row2918153732019"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172022"><a name="p17918337172022"></a><a name="p17918337172022"></a><a href="./docs/zh/best_practices.md">最佳实践</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742020"><a name="p15918183742020"></a><a name="p15918183742020"></a>提供Faiss使用的实践案例。</p>
</td>
</tr>
</tbody>
</table>

## 免责声明

此代码仓计划参与Faiss开源组件，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## License

Faiss采用MIT License许可证授权，支持修改代码和再开源，具体请参见[LICENSE](./LICENSE)文件。

本项目的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 致谢

Faiss由华为公司的下列部门联合贡献：

- 鲲鹏计算Boostkit开发部

感谢来自社区的每一个PR，欢迎贡献Faiss！

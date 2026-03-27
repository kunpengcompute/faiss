# 特性介绍

## 架构介绍

本节介绍Faiss主要算法的逻辑结构与所包含的模块含义及作用。
Faiss系统总体分为接口层、索引工厂层和底层依赖三部分：核心以C++实现，通过Python wrapper提供上层调用。索引工厂层包含算法抽象和基础算法两个子层，算法抽象层封装了多种索引类型（HNSW、IVF、PQ及其组合），基础算法层提供底层数据结构和计算方法。

核心索引能力：

- HNSW通过多层可导航小世界图实现高效近似最近邻搜索
- IVF通过k-means聚类划分向量空间，缩小搜索范围
- PQ/SQ提供向量压缩能力，降低内存占用
- FastScan利用SIMD指令加速距离计算
- Refine提供粗排后精排的两阶段检索能力

在保持Faiss原生接口不变的情况下，增加扩展接口调用FP16功能实现。
Faiss逻辑架构与功能模块如下所示。
**图 1** Faiss逻辑结构图<a name="fig289735134415"></a><a id="Faiss逻辑结构图"></a>

<img src="figures/Faiss逻辑结构图.jpg" alt="Faiss逻辑结构图" width="600"/>


<a name="table1440914563559"></a>
<table><thead align="left"><tr id="row10409145645519"><th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.1"><p id="p1949441225612"><a name="p1949441225612"></a><a name="p1949441225612"></a>模块名称</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.2"><p id="p249461210569"><a name="p249461210569"></a><a name="p249461210569"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row940925645518"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p194946128569"><a name="p194946128569"></a><a name="p194946128569"></a>C++ Interface</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p19494131211564"><a name="p19494131211564"></a><a name="p19494131211564"></a>Faiss算法的外部接口，Python wrapper通过此层调用底层实现</p>
</td>
</tr>
<tr id="row04102561556"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p13494612185612"><a name="p13494612185612"></a><a name="p13494612185612"></a>IndexHNSW</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p1494141275615"><a name="p1494141275615"></a><a name="p1494141275615"></a>基于多层小世界图的索引实现，查询时从顶层快速定位逐层下探，兼顾搜索效率与召回率</p>
</td>
</tr>
<tr id="row194101156165515"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p1549421213561"><a name="p1549421213561"></a><a name="p1549421213561"></a>IndexIVFFlat</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p184945129564"><a name="p184945129564"></a><a name="p184945129564"></a>倒排文件索引+Flat存储，先通过IVF聚类缩小候选范围，再对候选向量做精确距离计算</p>
</td>
</tr>
<tr id="row12410185620555"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p13494171225615"><a name="p13494171225615"></a><a name="p13494171225615"></a>IndexIVFPQ</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p7494171275610"><a name="p7494171275610"></a><a name="p7494171275610"></a>倒排文件索引+乘积量化，对聚类后的残差向量进行PQ压缩，适合超大规模数据集</p>
</td>
</tr>
<tr id="row1410115614559"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p1494181210560"><a name="p1494181210560"></a><a name="p1494181210560"></a>IndexPQFastScan / IndexIVFPQFastScan</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p849418121569"><a name="p849418121569"></a><a name="p849418121569"></a>PQ索引的SIMD加速版本，使用低比特 PQ+查找表（LUT）实现快速距离计算</p>
</td>
</tr>
<tr id="row2410105685518"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p12494171225617"><a name="p12494171225617"></a><a name="p12494171225617"></a>IndexRefineFlat / PreTransformIndex</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p0495112175614"><a name="p0495112175614"></a><a name="p0495112175614"></a>两阶段检索：第一阶段用粗糙索引快速筛选候选集，第二阶段用Flat精确距离重排序；PreTransform支持PCA/OPQ预处理变换</p>
</td>
</tr>
<tr id="row19410125616551"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p15495412135611"><a name="p15495412135611"></a><a name="p15495412135611"></a>HNSW graph (multi-layer)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p24951912185616"><a name="p24951912185616"></a><a name="p24951912185616"></a>HNSW多层图结构的具体实现，上层稀疏用于快速跳跃，下层稠密用于精确搜索</p>
</td>
</tr>
<tr id="row1790516495615"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p24951012125614"><a name="p24951012125614"></a><a name="p24951012125614"></a>IVF (nlist/nprobe, k-means)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p4495181214561"><a name="p4495181214561"></a><a name="p4495181214561"></a>倒排文件基础算法，通过k-means将向量空间划分为nlist个cell，查询时搜索 nprobe个最近cell</p>
</td>
</tr>
<tr id="row859311016564"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p1749512126567"><a name="p1749512126567"></a><a name="p1749512126567"></a>ProductQuantizer / OPQ / SQ</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p134953124568"><a name="p134953124568"></a><a name="p134953124568"></a>向量压缩技术：PQ将向量分段量化，OPQ是旋转优化的PQ，SQ是标量量化</p>
</td>
</tr>
<tr id="row1321914912563"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p849581295616"><a name="p849581295616"></a><a name="p849581295616"></a>Measures (L2 / IP)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p19495512135614"><a name="p19495512135614"></a><a name="p19495512135614"></a>提供L2欧氏距离和IP内积两种距离度量的具体实现</p>
</td>
</tr>
<tr id="row17208117155612"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p2495141285611"><a name="p2495141285611"></a><a name="p2495141285611"></a>List Scanner (Flat / ADC / FastScan)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p16495412105614"><a name="p16495412105614"></a><a name="p16495412105614"></a>列表扫描器：Flat做精确计算，ADC用于PQ非对称距离计算，FastScan用SIMD指令加速</p>
</td>
</tr>
<tr id="row864515315561"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p2495191220565"><a name="p2495191220565"></a><a name="p2495191220565"></a>Transforms (PCA/OPQ/normalize)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p1149541213566"><a name="p1149541213566"></a><a name="p1149541213566"></a>向量预处理变换，包括PCA降维、OPQ旋转优化、向量归一化等</p>
</td>
</tr>
</tbody>
</table>

## 优化说明

### 全量优化

| 优化点 | 说明|
|--------|-----------|
| 查表累和优化 | 查表累和算子是倒排索引与全量检索的热点算子，为计算瓶颈。距离扩展累加使用额外寄存器使指令展开度下降，同时引入额外计算。对内存数据进行重新排布，充分利用256位宽寄存器，减少临时寄存器开销，提升指令展开度，消除额外计算（数位扩展）。通过减少16个寄存器使用使流水线利用率提升，降低计算时延。|
| 向量过滤压缩 | 过滤压缩在计算bitmap时中间步骤多，为计算瓶颈。中间数据多为无效数据，平均寄存器位宽利用率不足50%。利用sve谓词与寄存器256位宽特性，省去中间步骤。|
| 距离计算优化 | 在距离计算中，对查询向量的访问会重复多次，访存时延成为系统瓶颈。同时计算1个查询与多个底库向量间距离，减少查询向量的读取时延。|
| 渐进式重排 | PQ阶段：是一种向量压缩的手段，可以减少内存的使用。但是4bit量化会导致精度有所下降，在PQ粗筛后需进行重排以找回损失的精度。重排时先用SQ8进行快速距离计算，然后使用FP32对最近的Top-N个距离进行精确计算，以提高计算效率。|
| 内存数据重排 | 距离计算中，对底库向量的访存不连续。Cache miss导致访存成为系统瓶颈。通过调整数据排布使访存流式进行，配合预取降低cache miss，降低访存时延。|


### IVFPQ优化
| 优化点 | 说明|
|--------|-----------|
| 查表累和优化 | 查表累和算子是倒排索引与全量检索的热点算子，为计算瓶颈。距离扩展累加使用额外寄存器使指令展开度下降，同时引入额外计算。对内存数据进行重新排布，充分利用256位宽寄存器，减少临时寄存器开销，提升指令展开度，消除额外计算。通过手写汇编实现多code并行与多subquantizer批量展开，进一步降低循环与地址更新成本；针对PQ table的随机访问特性，手工做指令调度与软件预取，最大化内存级并行，降低计算时延。|
| 向量乘加优化 | 向量乘加是距离计算中的核心算子。在满足数据对齐条件时启用优化路径：增大单次处理数据量，优化load/compute/store流水编排缩短指令依赖链。采用非临时写入策略降低cache污染，提高写带宽效率。|


### HNSW FP16支持

| 优化点 | 说明|
|--------|-----------|
| 支持HNSW FP16接口 | FP16数据类型将每个向量分量的存储空间从FP32的4字节减少到2字节。这使得在构建HNSW索引时，所需的内存空间大幅降低，从而允许在有限的硬件资源下处理更大规模的向量数据集。同时，针对鲲鹏ARM架构，对FP16的距离计算算法进行优化，提升了图检索过程中的节点距离比较效率，减少内存占用。|
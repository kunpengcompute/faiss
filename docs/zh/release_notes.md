# 版本说明书

## 版本配套说明

### 产品版本信息

<a name="table62675726"></a>
<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p11044137"><a name="p11044137"></a><a name="p11044137"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p1597721693713"><a name="p1597721693713"></a><a name="p1597721693713"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p56669300"><a name="p56669300"></a><a name="p56669300"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p11923034"><a name="p11923034"></a><a name="p11923034"></a><span id="text189831542174711"><a name="text189831542174711"></a><a name="text189831542174711"></a>v1.0.0</span></p>
</td>
</tr>
<tr id="row1930811171892"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p2030912172097"><a name="p2030912172097"></a><a name="p2030912172097"></a>软件名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p1730912179911"><a name="p1730912179911"></a><a name="p1730912179911"></a>Faiss</p>
</td>
</tr>
</tbody>
</table>

### 与操作系统、编译器和CPU配套说明

**表 1** Faiss已验证环境<a id="Faiss已验证环境"></a>

<a name="table4692134313211"></a>
<table><thead align="left"><tr id="row1169294312212"><th class="cellrowborder" valign="top" width="21.8%" id="mcps1.2.6.1.1"><p id="p12692144313211"><a name="p12692144313211"></a><a name="p12692144313211"></a>操作系统</p>
</th>
<th class="cellrowborder" valign="top" width="19.91%" id="mcps1.2.6.1.2"><p id="p06926438214"><a name="p06926438214"></a><a name="p06926438214"></a>CPU类型</p>
</th>
<th class="cellrowborder" valign="top" width="13.700000000000001%" id="mcps1.2.6.1.3"><p id="p269284310216"><a name="p269284310216"></a><a name="p269284310216"></a>内存</p>
</th>
<th class="cellrowborder" valign="top" width="17.34%" id="mcps1.2.6.1.4"><p id="p196922434215"><a name="p196922434215"></a><a name="p196922434215"></a>编译器</p>
</th>
<th class="cellrowborder" valign="top" width="27.250000000000004%" id="mcps1.2.6.1.5"><p id="p1769219435210"><a name="p1769219435210"></a><a name="p1769219435210"></a>其他</p>
</th>
</tr>
</thead>
<tbody><tr id="row624713534398"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.6.1.1 "><p id="p418818120409"><a name="p418818120409"></a><a name="p418818120409"></a>openEuler 22.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="19.91%" headers="mcps1.2.6.1.2 "><p id="p468512464216"><a name="p468512464216"></a><a name="p468512464216"></a>鲲鹏920 7282C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="13.700000000000001%" headers="mcps1.2.6.1.3 "><p id="p56685294911"><a name="p56685294911"></a><a name="p56685294911"></a>16*32GB</p>
</td>
<td class="cellrowborder" valign="top" width="17.34%" headers="mcps1.2.6.1.4 "><p id="p1118821124015"><a name="p1118821124015"></a><a name="p1118821124015"></a>GCC 12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="27.250000000000004%" headers="mcps1.2.6.1.5 "><p id="p18389155712168"><a name="p18389155712168"></a><a name="p18389155712168"></a>CMake&gt;=3.22.0</p>
</td>
</tr>
<td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.6.1.1 "><p id="p1321915499910"><a name="p1321915499910"></a><a name="p1321915499910"></a>Debian 12</p>
</td>
<td class="cellrowborder" valign="top" width="19.91%" headers="mcps1.2.6.1.2 "><p id="p152199497916"><a name="p152199497916"></a><a name="p152199497916"></a>鲲鹏920 7282C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="13.700000000000001%" headers="mcps1.2.6.1.3 "><p id="p1121911491393"><a name="p1121911491393"></a><a name="p1121911491393"></a>16*32GB</p>
</td>
<td class="cellrowborder" valign="top" width="17.34%" headers="mcps1.2.6.1.4 "><p id="p172198491297"><a name="p172198491297"></a><a name="p172198491297"></a>GCC 12.2.0 / LLVM 16.0.6</p>
</td>
<td class="cellrowborder" valign="top" width="27.250000000000004%" headers="mcps1.2.6.1.5 "><p id="p19219164917914"><a name="p19219164917914"></a><a name="p19219164917914"></a>CMake&gt;=3.25.1</p>
</td>
<tr id="row159615141350"><td class="cellrowborder" valign="top" width="20.86%" headers="mcps1.2.6.1.1 "><p id="p179611141352"><a name="p179611141352"></a><a name="p179611141352"></a>openEuler 24.03 LTS SP3</p>
</td>
<td class="cellrowborder" valign="top" width="20.150000000000002%" headers="mcps1.2.6.1.2 "><p id="p8961314756"><a name="p8961314756"></a><a name="p8961314756"></a>鲲鹏950 7592C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.6.1.3 "><p id="p15961191418510"><a name="p15961191418510"></a><a name="p15961191418510"></a>24*64GB</p>
</td>
<td class="cellrowborder" valign="top" width="17.549999999999997%" headers="mcps1.2.6.1.4 "><p id="p496115148515"><a name="p496115148515"></a><a name="p496115148515"></a>GCC 12.3.1</p>
</td>
<td class="cellrowborder" valign="top" width="27.58%" headers="mcps1.2.6.1.5 "><p id="p109611114654"><a name="p109611114654"></a><a name="p109611114654"></a>CMake&gt;=3.22.0</p>
</td>
</tr>
<tr id="row1837942531312"><td class="cellrowborder" valign="top" width="20.86%" headers="mcps1.2.6.1.1 "><p id="p1438015251131"><a name="p1438015251131"></a><a name="p1438015251131"></a>Debian 12</p>
</td>
<td class="cellrowborder" valign="top" width="20.150000000000002%" headers="mcps1.2.6.1.2 "><p id="p578363871318"><a name="p578363871318"></a><a name="p578363871318"></a>鲲鹏950 7592C处理器</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.6.1.3 "><p id="p127833389131"><a name="p127833389131"></a><a name="p127833389131"></a>24*64GB</p>
</td>
<td class="cellrowborder" valign="top" width="17.549999999999997%" headers="mcps1.2.6.1.4 "><p id="p821894241316"><a name="p821894241316"></a><a name="p821894241316"></a>GCC 12.2.0 / LLVM 16.0.6</p>
</td>
<td class="cellrowborder" valign="top" width="27.58%" headers="mcps1.2.6.1.5 "><p id="p1021810427131"><a name="p1021810427131"></a><a name="p1021810427131"></a>CMake&gt;=3.25.1</p>
</td>
</tr>
</tbody>
</table>

## 版本使用注意事项

### 使用注意事项

请参见[《Faiss 安装指南》](./installation_guide.md)。

## v1.0.0

### 更新说明

**新增特性<a name="section11862975"></a>**

提供全量优化补丁与等价优化补丁。其中，全量优化补丁针对IVFPQ算法进一步优化，新增支持HNSW FP16接口。补丁发布至Gitcode上，代码分支版本号为**v1.0.0**。

**修改特性<a name="section16450949161512"></a>**

无

**删除特性<a name="section9218125814159"></a>**

无

### 已解决的问题

无

### 遗留问题

无

## V25.3.0

### 更新说明

**新增特性<a name="section11862975"></a>**

新增Faiss子库，代码开源发布，需要通过源代码编译后使用。补丁发布至Gitcode上，代码分支版本号为**v1.8.0-2512**。

**修改特性<a name="section16450949161512"></a>**

无

**删除特性<a name="section9218125814159"></a>**

无

### 已解决的问题

无

### 遗留问题

无

## 版本配套文档

### v1.0.0版本配套文档

<a name="table1191773710200"></a>
<table><thead align="left"><tr id="row1291816372202"><th class="cellrowborder" valign="top" width="45.019999999999996%" id="mcps1.1.4.1.1"><p id="p291823714205"><a name="p291823714205"></a><a name="p291823714205"></a>文档名称</p>
</th>
<th class="cellrowborder" valign="top" width="38.019999999999996%" id="mcps1.1.4.1.2"><p id="p13918183762016"><a name="p13918183762016"></a><a name="p13918183762016"></a>内容简介</p>
</th>
<th class="cellrowborder" valign="top" width="16.96%" id="mcps1.1.4.1.3"><p id="p89181437152019"><a name="p89181437152019"></a><a name="p89181437152019"></a>交付形式</p>
</th>
</tr>
</thead>
<tbody><tr id="row179181137112015"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p1918123710208"><a name="p1918123710208"></a><a name="p1918123710208"></a><a href="./release_notes.md">《Faiss 版本说明书》</a></p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p491893752010"><a name="p491893752010"></a><a name="p491893752010"></a>本文档提供Faiss的版本发布信息。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p491893752011"><a name="p491893752011"></a><a name="p491893752011"></a>开源仓</p>
</td>
</tr>
<tr id="row939116371143"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p1039163711413"><a name="p1039163711413"></a><a name="p1039163711413"></a><a href="./quick_start.md">《Faiss 快速入门》</a></p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p1139217371746"><a name="p1139217371746"></a><a name="p1139217371746"></a>本文档提供Faiss的快速上手指导。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p1139217371747"><a name="p1139217371747"></a><a name="p1139217371747"></a>开源仓</p>
</td>
</tr>
<tr id="row2918153732017"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p598512211214"><a name="p598512211214"></a><a name="p598512211214"></a><a href="./installation_guide.md">《Faiss 安装指南》</a></p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p15918183742018"><a name="p15918183742018"></a><a name="p15918183742018"></a>本文档提供Faiss编译安装指导。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p15918183742028"><a name="p15918183742028"></a><a name="p15918183742028"></a>开源仓</p>
</td>
</tr>
<tr id="row2918153732018"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p598512211215"><a name="p598512211215"></a><a name="p598512211215"></a><a href="./api_reference.md">《Faiss API参考》</a></p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p15918183742019"><a name="p15918183742019"></a><a name="p15918183742019"></a>本文档提供Faiss新增API接口定义、接口说明。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p15918183742029"><a name="p15918183742029"></a><a name="p15918183742029"></a>开源仓</p>
</td>
</tr>
<tr id="row2918153732019"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p598512211216"><a name="p598512211216"></a><a name="p598512211216"></a><a href="./best_practices.md">《Faiss 最佳实践》</a></p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p15918183742020"><a name="p15918183742020"></a><a name="p15918183742020"></a>本文档提供Faiss使用的实践案例。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p15918183742030"><a name="p15918183742030"></a><a name="p15918183742030"></a>开源仓</p>
</td>
</tr>
<tr id="row2918153732020"><td class="cellrowborder" valign="top" width="45.019999999999996%" headers="mcps1.1.4.1.1 "><p id="p598512211217"><a name="p598512211217"></a><a name="p598512211217"></a><a href="./feature_introduction.md">《Faiss 特性介绍》</a></p>
</td>
<td class="cellrowborder" valign="top" width="38.019999999999996%" headers="mcps1.1.4.1.2 "><p id="p15918183742021"><a name="p15918183742021"></a><a name="p15918183742021"></a>本文档提供Faiss架构介绍及优化说明。</p>
</td>
<td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.3 "><p id="p15918183742031"><a name="p15918183742031"></a><a name="p15918183742031"></a>开源仓</p>
</td>
</tr>
</tbody>
</table>

### 获取文档的方法

您可以通过访问[开源仓](https://gitcode.com/boostkit/faiss)浏览和获取相关文档。

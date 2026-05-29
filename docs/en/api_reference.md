# API Reference

## API List

This section describes the APIs that support FP16 in Faiss. Currently, only C++ FP16 APIs are provided, and open-source Python APIs remain unmodified with no new APIs added. While fully maintaining compatibility with the open-source C++ APIs of version 1.8.0, support for the FP16 data type is added through extended APIs based on version 1.13.2.

The extended APIs are named by appending the `_ex` suffix to the native API names. Each extended API can automatically invoke the corresponding implementation based on the data type of the input parameter. Currently, the extended APIs support only the FP16 data type. Any future requirements for other data types can also be supported and invoked through the extended APIs.

The following uses the `add` function in `IndexHNSW.h` as an example:

<a name="table99541454172315"></a>
<table><thead align="left"><tr id="row6954165419237"><th class="cellrowborder" valign="top" width="21.02%" id="mcps1.1.3.1.1"><p id="p5954454172319"><a name="p5954454172319"></a><a name="p5954454172319"></a>API Description</p>
</th>
<th class="cellrowborder" valign="top" width="78.97999999999999%" id="mcps1.1.3.1.2"><p id="p1195485422316"><a name="p1195485422316"></a><a name="p1195485422316"></a>API Definition</p>
</th>
</tr>
</thead>
<tbody><tr id="row4954554132316"><td class="cellrowborder" valign="top" width="21.02%" headers="mcps1.1.3.1.1 "><p id="p18954165411237"><a name="p18954165411237"></a><a name="p18954165411237"></a>Open-source API</p>
</td>
<td class="cellrowborder" valign="top" width="78.97999999999999%" headers="mcps1.1.3.1.2 "><p id="p6954165419231"><a name="p6954165419231"></a><a name="p6954165419231"></a>void add(idx_t n, const float* x)</p>
</td>
</tr>
<tr id="row395420540232"><td class="cellrowborder" valign="top" width="21.02%" headers="mcps1.1.3.1.1 "><p id="p199541454142318"><a name="p199541454142318"></a><a name="p199541454142318"></a>FP16 API</p>
</td>
<td class="cellrowborder" valign="top" width="78.97999999999999%" headers="mcps1.1.3.1.2 "><p id="p15954154162315"><a name="p15954154162315"></a><a name="p15954154162315"></a>void add(idx_t n, const float16_t* x)</p>
</td>
</tr>
<tr id="row2954165482314"><td class="cellrowborder" valign="top" width="21.02%" headers="mcps1.1.3.1.1 "><p id="p1995414543235"><a name="p1995414543235"></a><a name="p1995414543235"></a>Extended API</p>
</td>
<td class="cellrowborder" valign="top" width="78.97999999999999%" headers="mcps1.1.3.1.2 "><p id="p16955195492319"><a name="p16955195492319"></a><a name="p16955195492319"></a>void add_ex(idx_t n, const void* x, NumericType numeric_type)</p>
</td>
</tr>
</tbody>
</table>

The extended API serves as an external API. When `add_ex` is called externally with `numeric_type` set to the FP16 data type, it internally invokes the newly added FP16 `add` API to execute the specific functionality.

>![note](public_sys-resources/icon-note.gif) **NOTE:**
>`HNSWFlat` provides FP16 APIs and full FP16 functional implementation.
>Currently, `HNSWPQ`, `HNSWSQ`, and `HNSW2Level` provide only the FP16 APIs. The input FP16 data is internally converted to the FP32 data type before invoking the native APIs.

[**Table 1** FP16 external APIs provided by Faiss-HNSW](#table162731837132517) describes the FP16 external APIs provided by Faiss-HNSW.

<a id="table162731837132517"></a>
**Table 1** FP16 external APIs provided by Faiss-HNSW

<table><thead align="left"><tr id="row12274537102510"><th class="cellrowborder" valign="top" width="38.800000000000004%" id="mcps1.2.3.1.1"><p id="p18274173702516"><a name="p18274173702516"></a><a name="p18274173702516"></a> API</p>
</th>
<th class="cellrowborder" valign="top" width="61.199999999999996%" id="mcps1.2.3.1.2"><p id="p1427453792511"><a name="p1427453792511"></a><a name="p1427453792511"></a>Functions</p>
</th>
</tr>
</thead>
<tbody><tr id="row1027453714256"><td class="cellrowborder" valign="top" width="38.800000000000004%" headers="mcps1.2.3.1.1 "><p id="p123181916268"><a name="p123181916268"></a><a name="p123181916268"></a>train_ex</p>
</td>
<td class="cellrowborder" valign="top" width="61.199999999999996%" headers="mcps1.2.3.1.2 "><p id="p14316195266"><a name="p14316195266"></a><a name="p14316195266"></a>Train a group of representative vectors.</p>
</td>
</tr>
<tr id="row4274113711251"><td class="cellrowborder" valign="top" width="38.800000000000004%" headers="mcps1.2.3.1.1 "><p id="p10311719202615"><a name="p10311719202615"></a><a name="p10311719202615"></a>add_ex</p>
</td>
<td class="cellrowborder" valign="top" width="61.199999999999996%" headers="mcps1.2.3.1.2 "><p id="p4311194268"><a name="p4311194268"></a><a name="p4311194268"></a>Add vectors to the index.</p>
</td>
</tr>
<tr id="row11274113716251"><td class="cellrowborder" valign="top" width="38.800000000000004%" headers="mcps1.2.3.1.1 "><p id="p1931919142619"><a name="p1931919142619"></a><a name="p1931919142619"></a>search_ex</p>
</td>
<td class="cellrowborder" valign="top" width="61.199999999999996%" headers="mcps1.2.3.1.2 "><p id="p1931191915268"><a name="p1931191915268"></a><a name="p1931191915268"></a>Search vectors against the index and return the nearest vectors.</p>
</td>
</tr>
<tr id="row1727483719253"><td class="cellrowborder" valign="top" width="38.800000000000004%" headers="mcps1.2.3.1.1 "><p id="p203181912612"><a name="p203181912612"></a><a name="p203181912612"></a>range_search_ex</p>
</td>
<td class="cellrowborder" valign="top" width="61.199999999999996%" headers="mcps1.2.3.1.2 "><p id="p63116191264"><a name="p63116191264"></a><a name="p63116191264"></a>Return all vectors within a specified radius.</p>
</td>
</tr>
<tr id="row8274113711254"><td class="cellrowborder" valign="top" width="38.800000000000004%" headers="mcps1.2.3.1.1 "><p id="p1731111902617"><a name="p1731111902617"></a><a name="p1731111902617"></a>reconstruct_ex</p>
</td>
<td class="cellrowborder" valign="top" width="61.199999999999996%" headers="mcps1.2.3.1.2 "><p id="p113112193260"><a name="p113112193260"></a><a name="p113112193260"></a>Reconstruct stored vectors.</p>
</td>
</tr>
<tr id="row1274537112519"><td class="cellrowborder" valign="top" width="38.800000000000004%" headers="mcps1.2.3.1.1 "><p id="p2031719122613"><a name="p2031719122613"></a><a name="p2031719122613"></a>search_level_0_ex</p>
</td>
<td class="cellrowborder" valign="top" width="61.199999999999996%" headers="mcps1.2.3.1.2 "><p id="p231519172620"><a name="p231519172620"></a><a name="p231519172620"></a>Execute batch approximate nearest neighbor (ANN) search within layer 0 of the HNSW index.</p>
</td>
</tr>
</tbody>
</table>

## IndexHNSW

**API Definition<a name="section55611644173112"></a>**

IndexHNSW\(int d, int M, NumericType ntype, MetricType metric\);

IndexHNSW\(Index\* storage, NumericType ntype, int M = 32\);

**Function<a name="section14731398326"></a>**

A constructor newly added for IndexHNSW.

**Parameters<a name="section93608182320"></a>**

<a name="table1463464643311"></a>
<table><thead align="left"><tr id="row16634946173317"><th class="cellrowborder" valign="top" width="14.92%" id="mcps1.1.5.1.1"><p id="p4787454193314"><a name="p4787454193314"></a><a name="p4787454193314"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.97%" id="mcps1.1.5.1.2"><p id="p3787454133320"><a name="p3787454133320"></a><a name="p3787454133320"></a>Type</p>
</th>
<th class="cellrowborder" valign="top" width="28.38%" id="mcps1.1.5.1.3"><p id="p7787165415336"><a name="p7787165415336"></a><a name="p7787165415336"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="38.73%" id="mcps1.1.5.1.4"><p id="p078717549335"><a name="p078717549335"></a><a name="p078717549335"></a>Value Range</p>
</th>
</tr>
</thead>
<tbody><tr id="row1963424620332"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p6787165493318"><a name="p6787165493318"></a><a name="p6787165493318"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p147871554113315"><a name="p147871554113315"></a><a name="p147871554113315"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p6787105412338"><a name="p6787105412338"></a><a name="p6787105412338"></a>Vector dimension</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p378705483318"><a name="p378705483318"></a><a name="p378705483318"></a>Positive integer</p>
</td>
</tr>
<tr id="row1863444613337"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p14787354173316"><a name="p14787354173316"></a><a name="p14787354173316"></a>M</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p167876547339"><a name="p167876547339"></a><a name="p167876547339"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p678775443311"><a name="p678775443311"></a><a name="p678775443311"></a>Maximum number of connections per node</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p19787654133320"><a name="p19787654133320"></a><a name="p19787654133320"></a>Positive integer greater than 1</p>
</td>
</tr>
<tr id="row20634134603314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p978720544334"><a name="p978720544334"></a><a name="p978720544334"></a>ntype</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p4787454113315"><a name="p4787454113315"></a><a name="p4787454113315"></a>NumericType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p9787054203313"><a name="p9787054203313"></a><a name="p9787054203313"></a>Vector data type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p187871954183315"><a name="p187871954183315"></a><a name="p187871954183315"></a><code>NumericType::Float16</code> and <code>NumericType::Float32</code></p>
</td>
</tr>
<tr id="row126351746153315"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p8787165419332"><a name="p8787165419332"></a><a name="p8787165419332"></a>metric</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p5787135411336"><a name="p5787135411336"></a><a name="p5787135411336"></a>MetricType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p187871854193319"><a name="p187871854193319"></a><a name="p187871854193319"></a>Distance computation type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p17787254153311"><a name="p17787254153311"></a><a name="p17787254153311"></a><code>METRIC_L2</code> and <code>METRIC_INNER_PRODUCT</code></p>
</td>
</tr>
<tr id="row1163564613314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1787165413314"><a name="p1787165413314"></a><a name="p1787165413314"></a>storage</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p178765493317"><a name="p178765493317"></a><a name="p178765493317"></a>Index*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p67878548335"><a name="p67878548335"></a><a name="p67878548335"></a>Storage index</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p07872054143318"><a name="p07872054143318"></a><a name="p07872054143318"></a>Supports indexes of types <code>IndexHNSWFlat</code>, <code>IndexHNSWPQ</code>, <code>IndexHNSWSQ</code>, and <code>IndexHNSW2Level</code>.</p>
</td>
</tr>
</tbody>
</table>

**Example<a name="section14731398327"></a>**

```cpp
// Create an FP16 IndexHNSW index with a dimension of 128, a maximum of 32 connections per node, and L2 distance.
int d = 128;
int M = 32;
faiss::IndexHNSW index(d, M, faiss::NumericType::Float16, faiss::METRIC_L2);

// Alternatively, create an IndexHNSW instance using an existing storage index.
faiss::IndexHNSWFlat* storage = new faiss::IndexHNSWFlat(d, M, faiss::NumericType::Float16);
faiss::IndexHNSW index_with_storage(storage, faiss::NumericType::Float16);
```

## IndexHNSWFlat

**API Definition<a name="section55611644173112"></a>**

IndexHNSWFlat\(int d, int M, NumericType ntype = NumericType::Float32, MetricType metric = METRIC\_L2\);

**Function<a name="section14731398326"></a>**

A constructor newly added for IndexHNSWFlat.

**Parameters<a name="section538651253420"></a>**

<a name="table1463464643311"></a>
<table><thead align="left"><tr id="row16634946173317"><th class="cellrowborder" valign="top" width="14.92%" id="mcps1.1.5.1.1"><p id="p4787454193314"><a name="p4787454193314"></a><a name="p4787454193314"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.97%" id="mcps1.1.5.1.2"><p id="p3787454133320"><a name="p3787454133320"></a><a name="p3787454133320"></a>Type</p>
</th>
<th class="cellrowborder" valign="top" width="28.38%" id="mcps1.1.5.1.3"><p id="p7787165415336"><a name="p7787165415336"></a><a name="p7787165415336"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="38.73%" id="mcps1.1.5.1.4"><p id="p078717549335"><a name="p078717549335"></a><a name="p078717549335"></a>Value Range</p>
</th>
</tr>
</thead>
<tbody><tr id="row1963424620332"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p75129553513"><a name="p75129553513"></a><a name="p75129553513"></a>d</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p1951217518359"><a name="p1951217518359"></a><a name="p1951217518359"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p1451205143515"><a name="p1451205143515"></a><a name="p1451205143515"></a>Vector dimension</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p10512758359"><a name="p10512758359"></a><a name="p10512758359"></a>Positive integer</p>
</td>
</tr>
<tr id="row1863444613337"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p75123513353"><a name="p75123513353"></a><a name="p75123513353"></a>M</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p5512175133514"><a name="p5512175133514"></a><a name="p5512175133514"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p151218511352"><a name="p151218511352"></a><a name="p151218511352"></a>Maximum number of connections per node</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p051216513355"><a name="p051216513355"></a><a name="p051216513355"></a>Positive integer greater than 1</p>
</td>
</tr>
<tr id="row20634134603314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1751255183514"><a name="p1751255183514"></a><a name="p1751255183514"></a>ntype</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p1851255163512"><a name="p1851255163512"></a><a name="p1851255163512"></a>NumericType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p2512053351"><a name="p2512053351"></a><a name="p2512053351"></a>Vector data type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p10512458359"><a name="p10512458359"></a><a name="p10512458359"></a><code>NumericType::Float16</code> and <code>NumericType::Float32</code></p>
</td>
</tr>
<tr id="row126351746153315"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1451211510358"><a name="p1451211510358"></a><a name="p1451211510358"></a>metric</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p13512252352"><a name="p13512252352"></a><a name="p13512252352"></a>MetricType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p1551245163519"><a name="p1551245163519"></a><a name="p1551245163519"></a>Distance computation type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p75120543512"><a name="p75120543512"></a><a name="p75120543512"></a><code>METRIC_L2</code> and <code>METRIC_INNER_PRODUCT</code></p>
</td>
</tr>
</tbody>
</table>

**Example<a name="section14731398328"></a>**

```cpp
// Create an FP16 IndexHNSWFlat index with a dimension of 128, a maximum of 32 connections per node, and inner product distance.
faiss::IndexHNSWFlat index_fp16(d, M, faiss::NumericType::Float16, faiss::METRIC_INNER_PRODUCT);
```

## train_ex

**API Definition<a name="section55611644173112"></a>**

void train\_ex\(idx\_t n, const void\* x, NumericType numeric\_type\);

**Function<a name="section14731398326"></a>**

Trains a group of representative vectors.

**Parameters<a name="section1659216150343"></a>**

<a name="table1463464643311"></a>
<table><thead align="left"><tr id="row16634946173317"><th class="cellrowborder" valign="top" width="14.92%" id="mcps1.1.5.1.1"><p id="p154492037203514"><a name="p154492037203514"></a><a name="p154492037203514"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.97%" id="mcps1.1.5.1.2"><p id="p10449123783510"><a name="p10449123783510"></a><a name="p10449123783510"></a>Type</p>
</th>
<th class="cellrowborder" valign="top" width="28.38%" id="mcps1.1.5.1.3"><p id="p4449837143514"><a name="p4449837143514"></a><a name="p4449837143514"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="38.73%" id="mcps1.1.5.1.4"><p id="p544913375359"><a name="p544913375359"></a><a name="p544913375359"></a> Value Range</p>
</th>
</tr>
</thead>
<tbody><tr id="row1963424620332"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p10449173717351"><a name="p10449173717351"></a><a name="p10449173717351"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p9449153719353"><a name="p9449153719353"></a><a name="p9449153719353"></a>idx_t</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p9449153733510"><a name="p9449153733510"></a><a name="p9449153733510"></a>Number of vectors for training</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p14493378353"><a name="p14493378353"></a><a name="p14493378353"></a>Positive integer </p>
</td>
</tr>
<tr id="row1863444613337"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1144913711359"><a name="p1144913711359"></a><a name="p1144913711359"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p84491237183511"><a name="p84491237183511"></a><a name="p84491237183511"></a>const void*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p1744933718351"><a name="p1744933718351"></a><a name="p1744933718351"></a>Vectors for training with a size of <code>n</code> × <code>d</code>, where <code>d</code> indicates the dimension of a single vector.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p19449163773517"><a name="p19449163773517"></a><a name="p19449163773517"></a>None</p>
</td>
</tr>
<tr id="row20634134603314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p944913719359"><a name="p944913719359"></a><a name="p944913719359"></a>numeric_type</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p24491737133517"><a name="p24491737133517"></a><a name="p24491737133517"></a>NumericType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p7449143763519"><a name="p7449143763519"></a><a name="p7449143763519"></a>Vector data type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p17449163753513"><a name="p17449163753513"></a><a name="p17449163753513"></a><code>NumericType::Float16</code> and <code>NumericType::Float32</code></p>
</td>
</tr>
</tbody>
</table>

**Example<a name="section14731398329"></a>**

```cpp
// Assume that an IndexHNSWFlat index of the FP16 type has been created.
int d = 128;
int M = 32;
faiss::IndexHNSWFlat index(d, M, faiss::NumericType::Float16);

// Create training data (1,000 128-dimensional FP16 vectors).
int n_train = 1000;
std::vector<faiss::float16_t> train_data(n_train * d);
// Populate the training data...

// Call the train_ex API for training.
index.train_ex(n_train, train_data.data(), faiss::NumericType::Float16);
```

## add_ex

**API Definition<a name="section55611644173112"></a>**

void add\_ex\(idx\_t n, const void\* x, NumericType numeric\_type\);

**Function<a name="section14731398326"></a>**

Adds vectors to an index.

**Parameters<a name="section162221119163417"></a>**

<a name="table1463464643311"></a>
<table><thead align="left"><tr id="row16634946173317"><th class="cellrowborder" valign="top" width="14.92%" id="mcps1.1.5.1.1"><p id="p167971503617"><a name="p167971503617"></a><a name="p167971503617"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.97%" id="mcps1.1.5.1.2"><p id="p97978517363"><a name="p97978517363"></a><a name="p97978517363"></a>Type</p>
</th>
<th class="cellrowborder" valign="top" width="28.38%" id="mcps1.1.5.1.3"><p id="p12797659369"><a name="p12797659369"></a><a name="p12797659369"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="38.73%" id="mcps1.1.5.1.4"><p id="p12797050368"><a name="p12797050368"></a><a name="p12797050368"></a>Value Range</p>
</th>
</tr>
</thead>
<tbody><tr id="row1963424620332"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p5797253365"><a name="p5797253365"></a><a name="p5797253365"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p127972583610"><a name="p127972583610"></a><a name="p127972583610"></a>idx_t</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p0797175193620"><a name="p0797175193620"></a><a name="p0797175193620"></a>Number of vectors</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p147972583612"><a name="p147972583612"></a><a name="p147972583612"></a>Positive integer</p>
</td>
</tr>
<tr id="row1863444613337"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p137976543613"><a name="p137976543613"></a><a name="p137976543613"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p3797115113618"><a name="p3797115113618"></a><a name="p3797115113618"></a>const void*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p20798259364"><a name="p20798259364"></a><a name="p20798259364"></a>Vectors, with a size pf <code>n</code> x <code>d</code>, where <code>d</code> indicates the dimension of a single vector.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p107981593619"><a name="p107981593619"></a><a name="p107981593619"></a>None</p>
</td>
</tr>
<tr id="row20634134603314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p19798851362"><a name="p19798851362"></a><a name="p19798851362"></a>numeric_type</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p0798175173612"><a name="p0798175173612"></a><a name="p0798175173612"></a>NumericType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p1379812514363"><a name="p1379812514363"></a><a name="p1379812514363"></a>Vector data type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p8798165133619"><a name="p8798165133619"></a><a name="p8798165133619"></a><code>NumericType::Float16</code> and <code>NumericType::Float32</code></p>
</td>
</tr>
</tbody>
</table>

**Example<a name="section14731398330"></a>**

```cpp
// Assume that an IndexHNSWFlat index of the FP16 type has been created and trained.
int d = 128;
int M = 32;
faiss::IndexHNSWFlat index(d, M, faiss::NumericType::Float16);

// Create data to be added (500 128-dimensional FP16 vectors).
int n_add = 500;
std::vector<faiss::float16_t> add_data(n_add * d);
// Populate the data to be added...

// Call the add_ex API to add vectors to the index.
index.add_ex(n_add, add_data.data(), faiss::NumericType::Float16);
```

## search_ex

**API Definition<a name="section55611644173112"></a>**

void search\_ex\(idx\_t n, const void\* x, idx\_t k, float\* distances, idx\_t\* labels, NumericType numeric\_type, const SearchParameters\* params = nullptr\);

**Function<a name="section14731398326"></a>**

Searches <code>n</code> <code>d</code>-dimensional vectors against the index and returns a maximum of k vectors for each. If the number of query results for each is less than <code>k</code>, the result arrays are padded with <code>-1</code>.

**Parameters<a name="section176018212346"></a>**

<a name="table1463464643311"></a>
<table><thead align="left"><tr id="row16634946173317"><th class="cellrowborder" valign="top" width="14.92%" id="mcps1.1.5.1.1"><p id="p12238154313619"><a name="p12238154313619"></a><a name="p12238154313619"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.97%" id="mcps1.1.5.1.2"><p id="p623811430361"><a name="p623811430361"></a><a name="p623811430361"></a>Parameter Type</p>
</th>
<th class="cellrowborder" valign="top" width="28.38%" id="mcps1.1.5.1.3"><p id="p32381743173613"><a name="p32381743173613"></a><a name="p32381743173613"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="38.73%" id="mcps1.1.5.1.4"><p id="p1423834353618"><a name="p1423834353618"></a><a name="p1423834353618"></a>Value Range</p>
</th>
</tr>
</thead>
<tbody><tr id="row1963424620332"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p12381343193618"><a name="p12381343193618"></a><a name="p12381343193618"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p9238124323615"><a name="p9238124323615"></a><a name="p9238124323615"></a>idx_t</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p72385438361"><a name="p72385438361"></a><a name="p72385438361"></a>Number of vectors</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p10238184312368"><a name="p10238184312368"></a><a name="p10238184312368"></a>Positive integer</p>
</td>
</tr>
<tr id="row1863444613337"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p123824311361"><a name="p123824311361"></a><a name="p123824311361"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p223816432360"><a name="p223816432360"></a><a name="p223816432360"></a>const void*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p1123894313361"><a name="p1123894313361"></a><a name="p1123894313361"></a>Vectors, with a size pf <code>n</code> x <code>d</code>, where <code>d</code> indicates the dimension of a single vector.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p123815437363"><a name="p123815437363"></a><a name="p123815437363"></a>None</p>
</td>
</tr>
<tr id="row20634134603314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p6238143133618"><a name="p6238143133618"></a><a name="p6238143133618"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p72381431369"><a name="p72381431369"></a><a name="p72381431369"></a>idx_t</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p162381343203619"><a name="p162381343203619"></a><a name="p162381343203619"></a>Number of returned vectors</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p22381343143612"><a name="p22381343143612"></a><a name="p22381343143612"></a>Positive integer</p>
</td>
</tr>
<tr id="row126351746153315"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p223813438363"><a name="p223813438363"></a><a name="p223813438363"></a>distances</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p02381435366"><a name="p02381435366"></a><a name="p02381435366"></a>float*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p9238154353612"><a name="p9238154353612"></a><a name="p9238154353612"></a>Output parameter, which returns the distance values of the vectors. The size is <code>n</code> × <code>k</code></p>.
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p32383430363"><a name="p32383430363"></a><a name="p32383430363"></a>None</p>
</td>
</tr>
<tr id="row13339204083610"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p102381943133615"><a name="p102381943133615"></a><a name="p102381943133615"></a>labels</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p6238174315368"><a name="p6238174315368"></a><a name="p6238174315368"></a>idx_t*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p122381143123612"><a name="p122381143123612"></a><a name="p122381143123612"></a>Output parameter, which returns the labels of the vectors. The size is <code>n</code> × <code>k</code></p>.
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p18238184323614"><a name="p18238184323614"></a><a name="p18238184323614"></a>None</p>
</td>
</tr>
<tr id="row1163564613314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1323994353612"><a name="p1323994353612"></a><a name="p1323994353612"></a>numeric_type</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p18239114333610"><a name="p18239114333610"></a><a name="p18239114333610"></a>NumericType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p123911435360"><a name="p123911435360"></a><a name="p123911435360"></a>Vector data type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p20239164317362"><a name="p20239164317362"></a><a name="p20239164317362"></a><code>NumericType::Float16</code> and <code>NumericType::Float32</code></p>
</td>
</tr>
<tr id="row18824738123612"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1023912434365"><a name="p1023912434365"></a><a name="p1023912434365"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p182392043163620"><a name="p182392043163620"></a><a name="p182392043163620"></a>const SearchParameters*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p72392043163619"><a name="p72392043163619"></a><a name="p72392043163619"></a>Search parameter</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p10239124312360"><a name="p10239124312360"></a><a name="p10239124312360"></a>None</p>
</td>
</tr>
</tbody>
</table>

**Example<a name="section14731398331"></a>**

```cpp
// Assume that an IndexHNSWFlat index of the FP16 type has been created, trained, and added for vectors.
int d = 128;
int M = 32;
faiss::IndexHNSWFlat index(d, M, faiss::NumericType::Float16);

// Create query data (10 128-dimensional FP16 vectors).
int n_query = 10;
std::vector<faiss::float16_t> query_data(n_query * d);
// Populate the query data...

// Set the number of returned results.
int k = 5;

// Prepare the memory for the output results.
std::vector<float> distances(n_query * k);
std::vector<faiss::idx_t> labels(n_query * k);

// Call the search_ex API for query.
index.search_ex(n_query, query_data.data(), k, distances.data(), labels.data(), faiss::NumericType::Float16);

// Process the query results.
for (int i = 0; i < n_query; i++) {
    printf("Query %d:\n", i);
    for (int j = 0; j < k; j++) {
        printf("  Neighbor %d: label=%lld, distance=%.4f\n", j, labels[i * k + j], distances[i * k + j]);
    }
}
```

## range_search_ex

**API Definition<a name="section55611644173112"></a>**

void range\_search\_ex\(idx\_t n, const void\* x, float radius, RangeSearchResult\* result, NumericType numeric\_type, const SearchParameters\* params = nullptr\);

**Function<a name="section14731398326"></a>**

Returns all vectors within a specified radius.

**Parameters<a name="section64561523133412"></a>**

<a name="table1463464643311"></a>
<table><thead align="left"><tr id="row16634946173317"><th class="cellrowborder" valign="top" width="14.92%" id="mcps1.1.5.1.1"><p id="p1087492163713"><a name="p1087492163713"></a><a name="p1087492163713"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.97%" id="mcps1.1.5.1.2"><p id="p0874921153718"><a name="p0874921153718"></a><a name="p0874921153718"></a>Parameter Type</p>
</th>
<th class="cellrowborder" valign="top" width="28.38%" id="mcps1.1.5.1.3"><p id="p19874021173713"><a name="p19874021173713"></a><a name="p19874021173713"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="38.73%" id="mcps1.1.5.1.4"><p id="p19874021153717"><a name="p19874021153717"></a><a name="p19874021153717"></a>Value Range</p>
</th>
</tr>
</thead>
<tbody><tr id="row1963424620332"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p128741221143715"><a name="p128741221143715"></a><a name="p128741221143715"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p108741521173712"><a name="p108741521173712"></a><a name="p108741521173712"></a>idx_t</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p19874172116376"><a name="p19874172116376"></a><a name="p19874172116376"></a>Number of vectors</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p1887413213377"><a name="p1887413213377"></a><a name="p1887413213377"></a>Positive integer</p>
</td>
</tr>
<tr id="row1863444613337"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p14874172115373"><a name="p14874172115373"></a><a name="p14874172115373"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p1887412153714"><a name="p1887412153714"></a><a name="p1887412153714"></a>const void*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p12874112153719"><a name="p12874112153719"></a><a name="p12874112153719"></a>Vectors, with a size pf <code>n</code> x <code>d</code>, where <code>d</code> indicates the dimension of a single vector.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p88741221173710"><a name="p88741221173710"></a><a name="p88741221173710"></a>None</p>
</td>
</tr>
<tr id="row20634134603314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p487422123710"><a name="p487422123710"></a><a name="p487422123710"></a>radius</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p287422115377"><a name="p287422115377"></a><a name="p287422115377"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p1187410216378"><a name="p1187410216378"></a><a name="p1187410216378"></a>Search radius</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p1287402133716"><a name="p1287402133716"></a><a name="p1287402133716"></a>Positive floating point number</p>
</td>
</tr>
<tr id="row126351746153315"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p208747216374"><a name="p208747216374"></a><a name="p208747216374"></a>result</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p6874112133710"><a name="p6874112133710"></a><a name="p6874112133710"></a>RangeSearchResult*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p687410219373"><a name="p687410219373"></a><a name="p687410219373"></a>Output parameter, which returns the search results.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p158743215375"><a name="p158743215375"></a><a name="p158743215375"></a>None</p>
</td>
</tr>
<tr id="row1163564613314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1587452111372"><a name="p1587452111372"></a><a name="p1587452111372"></a>numeric_type</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p38749218379"><a name="p38749218379"></a><a name="p38749218379"></a>NumericType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p13874921193713"><a name="p13874921193713"></a><a name="p13874921193713"></a>Vector data type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p19874421113720"><a name="p19874421113720"></a><a name="p19874421113720"></a><code>NumericType::Float16</code> and <code>NumericType::Float32</code></p>
</td>
</tr>
<tr id="row6551517133719"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p16874421143713"><a name="p16874421143713"></a><a name="p16874421143713"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p1987442143715"><a name="p1987442143715"></a><a name="p1987442143715"></a>const SearchParameters*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p58746212374"><a name="p58746212374"></a><a name="p58746212374"></a>Search parameter</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p787542133712"><a name="p787542133712"></a><a name="p787542133712"></a>None</p>
</td>
</tr>
</tbody>
</table>

**Example<a name="section14731398332"></a>**

```cpp
// Assume that an IndexHNSWFlat index of the FP16 type has been created, trained, and added for vectors.
int d = 128;
int M = 32;
faiss::IndexHNSWFlat index(d, M, faiss::NumericType::Float16);

// Create query data (5 128-dimensional FP16 vectors).
int n_query = 5;
std::vector<faiss::float16_t> query_data(n_query * d);
// Populate the query data...

// Set the search radius.
float radius = 10.0f;

// Create a RangeSearchResult object to store the results.
faiss::RangeSearchResult result(n_query);

// Call the range_search_ex API to perform range search.
index.range_search_ex(n_query, query_data.data(), radius, &result, faiss::NumericType::Float16);

// Process the query results.
for (int i = 0; i < n_query; i++) {
    // Obtain the offset and number of each query result.
    size_t begin = result.lims[i];
    size_t end = result.lims[i + 1];
    printf("Query %d found %lld results:\n", i, end - begin);
    
    for (size_t j = begin; j < end; j++) {
        printf("  Neighbor: label=%lld, distance=%.4f\n", result.labels[j], result.distances[j]);
    }
}
```

## reconstruct_ex

**API Definition<a name="section55611644173112"></a>**

void reconstruct\_ex\(idx\_t key, void\* recons, NumericType numeric\_type\);

**Function<a name="section14731398326"></a>**

Reconstructs the stored vector.

**Parameters<a name="section848811257346"></a>**

<a name="table1463464643311"></a>
<table><thead align="left"><tr id="row16634946173317"><th class="cellrowborder" valign="top" width="14.92%" id="mcps1.1.5.1.1"><p id="p733311482370"><a name="p733311482370"></a><a name="p733311482370"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.97%" id="mcps1.1.5.1.2"><p id="p23338489376"><a name="p23338489376"></a><a name="p23338489376"></a>Parameter Type</p>
</th>
<th class="cellrowborder" valign="top" width="28.34%" id="mcps1.1.5.1.3"><p id="p1333317481377"><a name="p1333317481377"></a><a name="p1333317481377"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="38.769999999999996%" id="mcps1.1.5.1.4"><p id="p10333114813714"><a name="p10333114813714"></a><a name="p10333114813714"></a>Value Range</p>
</th>
</tr>
</thead>
<tbody><tr id="row1963424620332"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p14333164813370"><a name="p14333164813370"></a><a name="p14333164813370"></a>key</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p1533344833715"><a name="p1533344833715"></a><a name="p1533344833715"></a>idx_t</p>
</td>
<td class="cellrowborder" valign="top" width="28.34%" headers="mcps1.1.5.1.3 "><p id="p53331748123718"><a name="p53331748123718"></a><a name="p53331748123718"></a>ID of the vector to be reconstructed</p>
</td>
<td class="cellrowborder" valign="top" width="38.769999999999996%" headers="mcps1.1.5.1.4 "><p id="p733319480371"><a name="p733319480371"></a><a name="p733319480371"></a>Positive integer</p>
</td>
</tr>
<tr id="row1863444613337"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1333310483378"><a name="p1333310483378"></a><a name="p1333310483378"></a>recons</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p103331485371"><a name="p103331485371"></a><a name="p103331485371"></a>void*</p>
</td>
<td class="cellrowborder" valign="top" width="28.34%" headers="mcps1.1.5.1.3 "><p id="p1333324812376"><a name="p1333324812376"></a><a name="p1333324812376"></a>Reconstructed vector</p>
</td>
<td class="cellrowborder" valign="top" width="38.769999999999996%" headers="mcps1.1.5.1.4 "><p id="p10333114817370"><a name="p10333114817370"></a><a name="p10333114817370"></a>None</p>
</td>
</tr>
<tr id="row20634134603314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p17333748153713"><a name="p17333748153713"></a><a name="p17333748153713"></a>numeric_type</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p14333948173718"><a name="p14333948173718"></a><a name="p14333948173718"></a>NumericType</p>
</td>
<td class="cellrowborder" valign="top" width="28.34%" headers="mcps1.1.5.1.3 "><p id="p143331548163712"><a name="p143331548163712"></a><a name="p143331548163712"></a>Vector data type</p>
</td>
<td class="cellrowborder" valign="top" width="38.769999999999996%" headers="mcps1.1.5.1.4 "><p id="p153331348193717"><a name="p153331348193717"></a><a name="p153331348193717"></a><code>NumericType::Float16</code> and <code>NumericType::Float32</code></p>
</td>
</tr>
</tbody>
</table>

**Example<a name="section14731398333"></a>**

```cpp
// Assume that an IndexHNSWFlat index of the FP16 type has been created, trained, and added for vectors.
int d = 128;
int M = 32;
faiss::IndexHNSWFlat index(d, M, faiss::NumericType::Float16);

// Assume that the index already contains vectors, and vector with ID 10 needs to be reconstructed.
faiss::idx_t key = 10;

// Prepare the memory for storing the reconstructed vector (FP16 type).
std::vector<faiss::float16_t> recons(d);

// Call the reconstruct_ex API to reconstruct the vector.
index.reconstruct_ex(key, recons.data(), faiss::NumericType::Float16);

// Print the first few elements of the reconstructed vector.
printf("Reconstructed vector for key %lld:\n", key);
for (int i = 0; i < std::min(5, d); i++) {
    // Convert float16_t to float for printing.
    float value = (float)recons[i];
    printf("  %d: %.4f\n", i, value);
}
```

## search_level_0_ex

**API Definition<a name="section55611644173112"></a>**

void search\_level\_0\_ex\(idx\_t n, const void\* x, idx\_t k, const storage\_idx\_t\* nearest, const float\* nearest\_d, float\* distances, idx\_t\* labels, NumericType numeric\_type, int nprobe = 1, int search\_type = 1\);

**Function<a name="section14731398326"></a>**

Performs ANN search within layer 0 of the HNSW index.

**Parameters<a name="section15773112719345"></a>**

<a name="table1463464643311"></a>
<table><thead align="left"><tr id="row16634946173317"><th class="cellrowborder" valign="top" width="14.92%" id="mcps1.1.5.1.1"><p id="p1522017486381"><a name="p1522017486381"></a><a name="p1522017486381"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.97%" id="mcps1.1.5.1.2"><p id="p20220104833815"><a name="p20220104833815"></a><a name="p20220104833815"></a>Type</p>
</th>
<th class="cellrowborder" valign="top" width="28.38%" id="mcps1.1.5.1.3"><p id="p6220154893814"><a name="p6220154893814"></a><a name="p6220154893814"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="38.73%" id="mcps1.1.5.1.4"><p id="p322012486383"><a name="p322012486383"></a><a name="p322012486383"></a>Value Range</p>
</th>
</tr>
</thead>
<tbody><tr id="row1963424620332"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p822011486387"><a name="p822011486387"></a><a name="p822011486387"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p142201048183814"><a name="p142201048183814"></a><a name="p142201048183814"></a>idx_t</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p122074818384"><a name="p122074818384"></a><a name="p122074818384"></a>Number of query vectors</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p9220134873810"><a name="p9220134873810"></a><a name="p9220134873810"></a>Positive integer</p>
</td>
</tr>
<tr id="row1863444613337"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p7220548143820"><a name="p7220548143820"></a><a name="p7220548143820"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p42201348113816"><a name="p42201348113816"></a><a name="p42201348113816"></a>const void*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p152201448193813"><a name="p152201448193813"></a><a name="p152201448193813"></a>Vectors, with a size pf <code>n</code> x <code>d</code>, where <code>d</code> indicates the dimension of a single vector.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p132207481383"><a name="p132207481383"></a><a name="p132207481383"></a>None</p>
</td>
</tr>
<tr id="row20634134603314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1922014486380"><a name="p1922014486380"></a><a name="p1922014486380"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p4220648193814"><a name="p4220648193814"></a><a name="p4220648193814"></a>idx_t</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p172201348183816"><a name="p172201348183816"></a><a name="p172201348183816"></a>Number of returned vectors</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p12207481389"><a name="p12207481389"></a><a name="p12207481389"></a>Positive integer</p>
</td>
</tr>
<tr id="row126351746153315"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p3220448133820"><a name="p3220448133820"></a><a name="p3220448133820"></a>nearest</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p11220848153813"><a name="p11220848153813"></a><a name="p11220848153813"></a>const storage_idx_t*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p822034893817"><a name="p822034893817"></a><a name="p822034893817"></a>Pre-computed nearest neighbor index array, providing the initial entry points for the search.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p132201548113813"><a name="p132201548113813"></a><a name="p132201548113813"></a>None</p>
</td>
</tr>
<tr id="row1163564613314"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p15220194853818"><a name="p15220194853818"></a><a name="p15220194853818"></a>nearest_d</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p1622064873815"><a name="p1622064873815"></a><a name="p1622064873815"></a>const float*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p10220194853812"><a name="p10220194853812"></a><a name="p10220194853812"></a>Pre-computed nearest neighbor distance array corresponding to <code>nearest</code> to avoid redundant computations.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p192206482388"><a name="p192206482388"></a><a name="p192206482388"></a>None</p>
</td>
</tr>
<tr id="row798032413820"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p14220204843813"><a name="p14220204843813"></a><a name="p14220204843813"></a>distances</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p7220194813811"><a name="p7220194813811"></a><a name="p7220194813811"></a>float*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p182203487382"><a name="p182203487382"></a><a name="p182203487382"></a>Output parameter, which returns the distance values of the vectors. The size is <code>n</code> × <code>k</code></p>.
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p1022094813812"><a name="p1022094813812"></a><a name="p1022094813812"></a>None</p>
</td>
</tr>
<tr id="row8530173010388"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p19220164883811"><a name="p19220164883811"></a><a name="p19220164883811"></a>labels</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p12220048133815"><a name="p12220048133815"></a><a name="p12220048133815"></a>idx_t*</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p6221184883816"><a name="p6221184883816"></a><a name="p6221184883816"></a>Output parameter, which returns the labels of the vectors. The size is <code>n</code> × <code>k</code></p>.
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p1922114488384"><a name="p1922114488384"></a><a name="p1922114488384"></a>None</p>
</td>
</tr>
<tr id="row77797351381"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p172211148123811"><a name="p172211148123811"></a><a name="p172211148123811"></a>numeric_type</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p2221124820385"><a name="p2221124820385"></a><a name="p2221124820385"></a>NumericType</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p102211948123815"><a name="p102211948123815"></a><a name="p102211948123815"></a>Vector data type</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p122211448193813"><a name="p122211448193813"></a><a name="p122211448193813"></a><code>NumericType::Float16</code> and <code>NumericType::Float32</code></p>
</td>
</tr>
<tr id="row840793719388"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p322184812387"><a name="p322184812387"></a><a name="p322184812387"></a>nprobe</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p192218481385"><a name="p192218481385"></a><a name="p192218481385"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p822118484386"><a name="p822118484386"></a><a name="p822118484386"></a>Controls the search scope by specifying the number of candidate nodes to explore on each layer.</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p82215484386"><a name="p82215484386"></a><a name="p82215484386"></a>Positive integer</p>
</td>
</tr>
<tr id="row171409395382"><td class="cellrowborder" valign="top" width="14.92%" headers="mcps1.1.5.1.1 "><p id="p1622112481386"><a name="p1622112481386"></a><a name="p1622112481386"></a>search_type</p>
</td>
<td class="cellrowborder" valign="top" width="17.97%" headers="mcps1.1.5.1.2 "><p id="p152211248123818"><a name="p152211248123818"></a><a name="p152211248123818"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="28.38%" headers="mcps1.1.5.1.3 "><p id="p12221194883815"><a name="p12221194883815"></a><a name="p12221194883815"></a>Search strategy identifier</p>
</td>
<td class="cellrowborder" valign="top" width="38.73%" headers="mcps1.1.5.1.4 "><p id="p1622115480387"><a name="p1622115480387"></a><a name="p1622115480387"></a><code>1</code> and <code>2</code></p>
</td>
</tr>
</tbody>
</table>

**Example<a name="section14731398334"></a>**

```cpp
// Assume that an IndexHNSWFlat index of the FP16 type has been created, trained, and added for vectors.
int d = 128;
int M = 32;
faiss::IndexHNSWFlat index(d, M, faiss::NumericType::Float16);

// Create query data (3 128-dimensional FP16 vectors).
int n_query = 3;
std::vector<faiss::float16_t> query_data(n_query * d);
// Populate the query data...

// Set the number of returned results.
int k = 4;

// Prepare the pre-computed nearest neighbor information (simplified here; in practice, this should be pre-computed).
std::vector<faiss::storage_idx_t> nearest(n_query);
std::vector<float> nearest_d(n_query);
// Populate the pre-computed nearest neighbor indices and distances...
for (int i = 0; i < n_query; i++) {
    nearest[i] = 0; // Assume that the initial nearest neighbor for each query is index 0.
    nearest_d[i] = 0.0f;
}

// Prepare the memory for the output results.
std::vector<float> distances(n_query * k);
std::vector<faiss::idx_t> labels(n_query * k);

// Call the search_level_0_ex API to perform search within layer 0.
int nprobe = 2;
int search_type = 1;
index.search_level_0_ex(n_query, query_data.data(), k, nearest.data(), nearest_d.data(), 
                       distances.data(), labels.data(), faiss::NumericType::Float16, 
                       nprobe, search_type);

// Process the query results.
for (int i = 0; i < n_query; i++) {
    printf("Query %d (level 0 search):\n", i);
    for (int j = 0; j < k; j++) {
        printf("  Neighbor %d: label=%lld, distance=%.4f\n", j, labels[i * k + j], distances[i * k + j]);
    }
}
```

[def]: #table162731837132517

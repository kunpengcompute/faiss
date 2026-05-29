# Feature Introduction

## Architecture

This document describes the logical structure of the main Faiss algorithms, as well as the definitions and functions of the modules.
The Faiss system consists of three major layers: the API layer, the index factory layer, and underlying dependencies. Built with a C++ core, it provides upper-tier invocations via a Python wrapper. The index factory layer includes two sub-layers: algorithm abstraction, which encapsulates index types such as HNSW, IVF, PQ, and their combinations, and basic algorithms, which provide the underlying data structures and computing methods.

Core indexing capabilities:

- HNSW implements efficient approximate nearest neighbor (ANN) search via multi-layer navigable small-world graphs.
- IVF divides the vector space using K-means clustering to narrow down the search scope.
- PQ/SQ provides vector compression capabilities to significantly reduce memory footprint.
- FastScan leverages SIMD instructions to accelerate distance computations.
- Refine enables two-phase retrieval capabilities for coarse-grained ranking and fine-grained reranking.

While keeping the native Faiss APIs completely unchanged, extended APIs are introduced to enable FP16 acceleration.
The following figure shows the logical architecture and functional modules of Faiss.

**Figure 1** Faiss logical architecture<a name="fig289735134415"></a><a id="faiss logical architecture"></a>

<img src="figures/faiss-logical-architecture.jpg" alt="faiss logical architecture" width="600"/>

<a name="table1440914563559"></a>
<table><thead align="left"><tr id="row10409145645519"><th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.1"><p id="p1949441225612"><a name="p1949441225612"></a><a name="p1949441225612"></a>Module</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.2"><p id="p249461210569"><a name="p249461210569"></a><a name="p249461210569"></a>Function</p>
</th>
</tr>
</thead>
<tbody><tr id="row940925645518"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p194946128569"><a name="p194946128569"></a><a name="p194946128569"></a>C++ interface</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p19494131211564"><a name="p19494131211564"></a><a name="p19494131211564"></a>External interface for Faiss algorithms, through which the Python wrapper invokes underlying implementations.</p>
</td>
</tr>
<tr id="row04102561556"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p13494612185612"><a name="p13494612185612"></a><a name="p13494612185612"></a>IndexHNSW</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p1494141275615"><a name="p1494141275615"></a><a name="p1494141275615"></a>Index implementation based on multi-layer navigable small-world graphs. It routes down rapidly layer by layer from the top during queries to balance search efficiency and recall rate.</p>
</td>
</tr>
<tr id="row194101156165515"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p1549421213561"><a name="p1549421213561"></a><a name="p1549421213561"></a>IndexIVFFlat</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p184945129564"><a name="p184945129564"></a><a name="p184945129564"></a>Inverted file index + Flat storage. It narrows down the candidate scope through IVF clustering first, and then performs exact distance computations on candidate vectors.</p>
</td>
</tr>
<tr id="row12410185620555"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p13494171225615"><a name="p13494171225615"></a><a name="p13494171225615"></a>IndexIVFPQ</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p7494171275610"><a name="p7494171275610"></a><a name="p7494171275610"></a>Inverted file index + Product Quantization (PQ). It performs PQ compression on residual vectors after clustering, suitable for ultra-large-scale datasets.</p>
</td>
</tr>
<tr id="row1410115614559"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p1494181210560"><a name="p1494181210560"></a><a name="p1494181210560"></a>IndexPQFastScan / IndexIVFPQFastScan</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p849418121569"><a name="p849418121569"></a><a name="p849418121569"></a>SIMD-accelerated versions of PQ indexing. It implements fast distance computation using low-bit PQ and look-up table (LUT).</p>
</td>
</tr>
<tr id="row2410105685518"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p12494171225617"><a name="p12494171225617"></a><a name="p12494171225617"></a>IndexRefineFlat / PreTransformIndex</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p0495112175614"><a name="p0495112175614"></a><a name="p0495112175614"></a>Two-phase retrieval: the first phase uses a coarse index to quickly filter candidate sets, and the second phase uses Flat exact distance for re-ranking. PreTransform supports PCA/OPQ preprocessing and transformation.</p>
</td>
</tr>
<tr id="row19410125616551"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p15495412135611"><a name="p15495412135611"></a><a name="p15495412135611"></a>HNSW graph (multi-layer)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p24951912185616"><a name="p24951912185616"></a><a name="p24951912185616"></a>Implementation of the HNSW multi-layer graph structure. The upper layers are sparse for fast routing, and the lower layers are dense for precise search.</p>
</td>
</tr>
<tr id="row1790516495615"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p24951012125614"><a name="p24951012125614"></a><a name="p24951012125614"></a>IVF (nlist/nprobe, k-means)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p4495181214561"><a name="p4495181214561"></a><a name="p4495181214561"></a>Basic algorithm of inverted files. It divides the vector space into <code>nlist</code> cells via K-means, and searches the <code>nprobe</code> nearest cells during queries.</p>
</td>
</tr>
<tr id="row859311016564"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p1749512126567"><a name="p1749512126567"></a><a name="p1749512126567"></a>ProductQuantizer / OPQ / SQ</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p134953124568"><a name="p134953124568"></a><a name="p134953124568"></a>Vector compression techniques: PQ quantizes vectors by segment, OPQ is rotationally optimized PQ, and SQ is scalar quantization.</p>
</td>
</tr>
<tr id="row1321914912563"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p849581295616"><a name="p849581295616"></a><a name="p849581295616"></a>Measures (L2 / IP)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p19495512135614"><a name="p19495512135614"></a><a name="p19495512135614"></a>Provides implementations for two distance metrics: L2 Euclidean distance and Inner Product (IP) distance.</p>
</td>
</tr>
<tr id="row17208117155612"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p2495141285611"><a name="p2495141285611"></a><a name="p2495141285611"></a>List Scanner (Flat / ADC / FastScan)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p16495412105614"><a name="p16495412105614"></a><a name="p16495412105614"></a> List scanner: Flat is used for exact computation, ADC is used for PQ asymmetric distance computation, and FastScan uses SIMD instructions for acceleration.</p>
</td>
</tr>
<tr id="row864515315561"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p2495191220565"><a name="p2495191220565"></a><a name="p2495191220565"></a>Transforms (PCA/OPQ/normalize)</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p1149541213566"><a name="p1149541213566"></a><a name="p1149541213566"></a>Vector preprocessing transformations, including PCA dimensionality reduction, OPQ, and vector normalization.</p>
</td>
</tr>
</tbody>
</table>

## Optimization Description

### Non-Equivalence Optimization

| Optimization| Description|
|--------|-----------|
| LUT-based accumulation| The LUT accumulation operator is a critical hotspot operator in inverted index and exhaustive scan, often causing computational bottlenecks. Distance accumulation with sign extension requires additional registers, which reduces the degree of instruction unrolling and introduces redundant computational overhead. To address this, in-memory data layout is reordered to fully utilize the 256-bit wide registers. This approach minimizes temporary register overhead, increases the degree of instruction unrolling, and eliminates redundant computations (bit-width extension). By reducing the use of 16 registers, the pipeline utilization is improved, and the computation latency is reduced.|
| Vector filtering and compression| The filtering and compression process involves numerous intermediate steps when calculating bitmaps, creating a bottleneck. A large portion of the intermediate data is invalid, leading to less than 50% average utilization of register bit-width. This optimization leverages SVE predicates and the 256-bit register width feature to bypass intermediate steps.|
| Distance computation optimization| In distance computation, the query vector is repeatedly accessed, making the memory access latency a system bottleneck. By computing the distances between one query and multiple base vectors simultaneously, the read latency of the query vector is significantly reduced.|
| Progressive reranking| PQ phase: This is a vector compression method that can reduce memory usage. However, 4-bit quantization leads to a certain decline in precision. After PQ coarse filtering, reranking is required to recover the lost precision. During reranking, SQ8 is used for fast distance computation first, and then FP32 is used to accurately compute the top N closest distances, improving computation efficiency.|
| Memory data rearrangement| In distance computation, the memory access to base vectors is non-contiguous. Cache miss causes memory access to become a system bottleneck. The data layout is adjusted to enable streaming memory access, cooperating with prefetching to reduce cache misses and memory access latency.|

### IVFPQ Optimization

| Optimization| Description|
|--------|-----------|
| LUT-based accumulation| The LUT accumulation operator is a critical hotspot operator in inverted index and exhaustive scan, often causing computational bottlenecks. Distance accumulation with sign extension requires additional registers, which reduces the degree of instruction unrolling and introduces redundant computational overhead. To address this, in-memory data layout is reordered to fully utilize the 256-bit wide registers. This approach minimizes temporary register overhead, increases the degree of instruction unrolling, and eliminates redundant computations. Hand-coded assembly is leveraged to implement multi-code parallelism and batch unrolling of multiple subquantizers, further reducing loop and address update overhead. For the random access feature of the PQ table, manual instruction scheduling and software prefetching are performed to maximize memory-level parallelism and reduce calculation latency.|
| Vector multiply-accumulate optimization| Vector multiply-accumulate is the core operator for distance computation. When data alignment conditions are met, the optimization path is enabled: increasing the data volume processed per iteration, and optimizing load/compute/store pipeline orchestration to shorten the instruction dependency chain. A non-temporal write policy is adopted to reduce cache pollution and improve write bandwidth efficiency.|

### HNSW FP16 Support

| Optimization| Description|
|--------|-----------|
| HNSW FP16 interface supported| The FP16 data type reduces the storage space of each vector component from 4 bytes in FP32 to 2 bytes. This significantly reduces the memory space required for building HNSW indexes, allowing for the processing of larger vector datasets under limited hardware resources. In addition, optimized for the Kunpeng Arm architecture, the FP16 distance computation algorithm improves the efficiency of node distance comparison during graph retrieval, reducing memory footprint.|

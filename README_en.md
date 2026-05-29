# Introduction to Faiss

## Lastest Updates

- [2026.03.30]: Faiss provides a non-equivalence optimization patch and an equivalence optimization patch. The non-equivalence optimization patch further optimizes the IVFPQ algorithm and supports the HNSW FP16 interface.
- [2025.12.30]: Faiss was released on the Gitcode platform, optimizing IVFFLAT, IVFPQ, IVFPQFS, PQFS, and HNSW.

## Overview

Faiss is an algorithm library developed by Facebook for efficient similarity search and clustering of dense vectors. It is written in C++ and provides complete bindings for Python and NumPy. Faiss supports multiple indexing methods such as IVFFlat, IVFPQ, HNSW, IVFPQFS, and PQFS. Kunpeng optimization is based on intrusive modifications to the open-source Faiss code while retaining the original interfaces.

HNSW is an approximate nearest neighbor (ANN) graph retrieval algorithm provided by Faiss. The open-source Faiss (HNSW) interface supports the FP32 data type. To optimize the computing efficiency and memory usage, the original Faiss has been adapted and reconstructed to add an FP16 interface, enabling efficient computation for FP16-based retrieval under the Kunpeng Arm architecture

## Directory Structure

The repository directory structure is as follows:

```text
faiss/
├─ 0001-faiss_1.8.0-optimize-neq.patch         // Non-equivalence optimization patch
├─ 0002-faiss_1.8.0-optimize-eqv.patch         // Equivalence optimization patch
└── docs
   ├── LICENSE
   └── en
      ├── api_reference.md                        // API reference
      ├── feature_introduction.md                 // Feature introduction
      ├── best_practices.md                       // Best practices
      ├── installation_guide.md                   // Installation guide
      ├── quick_start.md                          // Quick start
      └── release_notes.md                        // Release notes
```

## Release Notes

For details about the version updates of Faiss, see [Release Notes](./docs/en/release_notes.md).

## Documents

<a name="table1191773710200"></a>
<table><thead align="left"><tr id="row1291816372202"><th class="cellrowborder" valign="top" width="9.780978097809781%" id="mcps1.1.4.1.1"><p id="p291823714205"><a name="p291823714205"></a><a name="p291823714205"></a>Resource Type</p>
</th>
<th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.2"><p id="p13918183762016"><a name="p13918183762016"></a><a name="p13918183762016"></a>Resource Name</p>
</th>
Introduction to <th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.3"><p id="p89181437152019"><a name="p89181437152019"></a><a name="p89181437152019"></a>Resource Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row2918153732020"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p598512211217"><a name="p598512211217"></a><a name="p598512211217"></a>Document</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172023"><a name="p17918337172023"></a><a name="p17918337172023"></a><a href="./docs/en/feature_introduction.md"> Feature Introduction</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742021"><a name="p15918183742021"></a><a name="p15918183742021"></a>Describes the Faiss architecture and optimizations.</p>
</td>
</tr>
<tr id="row179181137112015"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p1918123710208"><a name="p1918123710208"></a><a name="p1918123710208"></a>Document</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p2091893722011"><a name="p2091893722011"></a><a name="p2091893722011"></a><a href="./docs/en/release_notes.md">Release Notes</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p491893752010"><a name="p491893752010"></a><a name="p491893752010"></a>Provides basic information and feature updates of each Faiss release.</p>
</td>
</tr>
<tr id="row939116371143"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p1039163711413"><a name="p1039163711413"></a><a name="p1039163711413"></a>Document</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p03913372046"><a name="p03913372046"></a><a name="p03913372046"></a><a href="./docs/en/quick_start.md">Quick Start</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p1139217371746"><a name="p1139217371746"></a><a name="p1139217371746"></a>Provides guidance for getting started with Faiss.</p>
</td>
</tr>
<tr id="row2918153732017"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p598512211214"><a name="p598512211214"></a><a name="p598512211214"></a>Document</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172020"><a name="p17918337172020"></a><a name="p17918337172020"></a><a href="./docs/en/installation_guide.md">Installation Guide</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742018"><a name="p15918183742018"></a><a name="p15918183742018"></a>Provides guidance for compiling and installing Faiss.</p>
</td>
</tr>
<tr id="row2918153732018"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p598512211215"><a name="p598512211215"></a><a name="p598512211215"></a>Document</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172021"><a name="p17918337172021"></a><a name="p17918337172021"></a><a href="./docs/en/api_reference.md">API Reference</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742019"><a name="p15918183742019"></a><a name="p15918183742019"></a>Provides the definitions and descriptions of new Faiss APIs.</p>
</td>
</tr>
<tr id="row2918153732019"><td class="cellrowborder" valign="top" width="9.780978097809781%" headers="mcps1.1.4.1.1 "><p id="p598512211216"><a name="p598512211216"></a><a name="p598512211216"></a>Document</p>
</td>
<td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172022"><a name="p17918337172022"></a><a name="p17918337172022"></a><a href="./docs/en/best_practices.md">Best Practices</a></p>
</td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742020"><a name="p15918183742020"></a><a name="p15918183742020"></a>Provides practical cases of Faiss.</p>
</td>
</tr>
</tbody>
</table>

## Disclaimer

This code repository contributes to the Faiss open-source components. It strictly adheres to the coding style and methods, as well as security design of the native open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

Faiss is licensed under the MIT License, which allows modification and redistribution of derivative works as open source. For details, see [LICENSE](./docs/LICENSE).

The documents of this project are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit [issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md). For details, see the [contribution guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Acknowledgments

Faiss is jointly developed by the following Huawei department:

- Kunpeng Computing BoostKit Development Dept

Thank you to everyone in the community for your PRs. We warmly welcome contributions to Faiss!

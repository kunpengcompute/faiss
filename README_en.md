# Introduction to Faiss

<!-- md-trans-meta sourceCommit=5ffaf4a8395f3e273eb84b9b8cafc6aa1edf1293 translatedAt=2026-08-06T08:58:34.403Z pushedAt=2026-08-07T01:07:09.483Z -->

## Latest Updates

- [September 30, 2026]: A RabitQ index optimization patch based on Faiss v1.14.3 is provided.

- [June 30, 2026]: The VisitedTable access flags are optimized by replacing the full memset reset with a generation-based flag similar to a hashset, reducing VisitedTable updates from O(N) to O(1). The 4-bit lookup operator is implemented using SVE2..

- [2026.03.30]: Faiss provides a non-equivalence optimization patch and an equivalence optimization patch. The non-equivalence optimization patch further optimizes the IVFPQ algorithm and supports the HNSW FP16 interface.

- [2025.12.30]: Faiss was released on the Gitcode platform, optimizing IVFFLAT, IVFPQ, IVFPQFS, PQFS, and HNSW.

## Project Introduction

Faiss is an algorithm library developed by Facebook for efficient similarity search and clustering of dense vectors. It is written in C++ and provides complete bindings for Python and NumPy. Faiss supports multiple indexing methods such as IVFFlat, IVFPQ, HNSW, IVFPQFS, and PQFS. Kunpeng optimization is based on intrusive modifications to the open-source Faiss code while retaining the original interfaces.

HNSW is an approximate nearest neighbor (ANN) graph retrieval algorithm provided by Faiss. The open-source Faiss (HNSW) interface supports the FP32 data type. To optimize the computing efficiency and memory usage, the original Faiss has been adapted and reconstructed to add an FP16 interface, enabling efficient computation for FP16-based retrieval under the Kunpeng Arm architecture.

## Directory Structure

The repository directory structure is as follows:

```text
faiss/
├─ 0001-faiss_1.8.0-optimize-neq.patch         // Non-equivalence optimization patch based on Faiss v1.8.0.
├─ 0002-faiss_1.8.0-optimize-eqv.patch         // Equivalence optimization patch based on Faiss v1.8.0
├─ 0001-faiss_1.14.3-optimize-rabitq.patch     // RabitQ index optimization patch based on Faiss v1.14.3
├─ README.md                                   // Project introduction
└── docs
   ├── LICENSE
   └── zh
      ├── api_reference.md                        // API Reference
      ├── feature_introduction.md                 // Feature Introduction
      ├── best_practices.md                       // Best Practices
      ├── installation_guide.md                   // Installation Guide
      ├── quick_start.md                          // Quick Start
      └── release_notes.md                        // Release Notes
```

## Version Description

For details about the version updates of Faiss, see [*Release Notes*](./docs/en/release_notes.md).

## Documents

<a name="table1191773710200"></a>
<table><thead align="left"><tr id="row1291816372202"><th class="cellrowborder" valign="top" width="17.64176417641764%" id="mcps1.1.4.1.2"><p id="p13918183762016"><a name="p13918183762016"></a><a name="p13918183762016"></a>Resource Name</p></th>
<th class="cellrowborder" valign="top" width="72.57725772577258%" id="mcps1.1.4.1.3"><p id="p89181437152019"><a name="p89181437152019"></a><a name="p89181437152019"></a>Resource Description</p></th>
</tr>
</thead>
<tbody><tr id="row2918153732020"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172023"><a name="p17918337172023"></a><a name="p17918337172023"></a><a href="./docs/en/feature_introduction.md">Feature Introduction</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742021"><a name="p15918183742021"></a><a name="p15918183742021"></a>Provides an introduction to the Faiss architecture and optimization notes.</p></td>
</tr>
<tr id="row179181137112015"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p2091893722011"><a name="p2091893722011"></a><a name="p2091893722011"></a><a href="./docs/en/release_notes.md">Release Notes</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p491893752010"><a name="p491893752010"></a><a name="p491893752010"></a>Provides basic information and feature update details for each Faiss release.</p></td>
</tr>
<tr id="row939116371143"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p03913372046"><a name="p03913372046"></a><a name="p03913372046"></a><a href="./docs/en/quick_start.md">Quick Start</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p1139217371746"><a name="p1139217371746"></a><a name="p1139217371746"></a>Provides guidance on getting started with Faiss.</p></td>
</tr>
<tr id="row2918153732017"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172020"><a name="p17918337172020"></a><a name="p17918337172020"></a><a href="./docs/en/installation_guide.md">Installation Guide</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742018"><a name="p15918183742018"></a><a name="p15918183742018"></a>Provides guidance on Faiss compilation and installation methods.</p></td>
</tr>
<tr id="row2918153732018"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172021"><a name="p17918337172021"></a><a name="p17918337172021"></a><a href="./docs/en/api_reference.md">API Reference</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742019"><a name="p15918183742019"></a><a name="p15918183742019"></a>Provides definitions and descriptions of newly added Faiss API interfaces.</p></td>
</tr>
<tr id="row2918153732019"><td class="cellrowborder" valign="top" width="17.64176417641764%" headers="mcps1.1.4.1.2 "><p id="p17918337172022"><a name="p17918337172022"></a><a name="p17918337172022"></a><a href="./docs/en/best_practices.md">Best Practices</a></p></td>
<td class="cellrowborder" valign="top" width="72.57725772577258%" headers="mcps1.1.4.1.3 "><p id="p15918183742020"><a name="p15918183742020"></a><a name="p15918183742020"></a>Provides practical use cases for Faiss.</p></td>
</tr>
</tbody>
</table>

## Disclaimer

This code repository contributes to the Faiss open-source components. It strictly adheres to the coding style and methods, as well as security design of the native open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

Faiss is licensed under the MIT License, which allows modification and redistribution of derivative works as open source. For details, see [LICENSE](LICENSE).

The documents of this project are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit [issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md). For details, see [Contributing Guide](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Acknowledgments

Faiss is jointly developed by the following Huawei department:

- Kunpeng Computing BoostKit Development Dept

Thank you to everyone in the community for your PRs. We warmly welcome contributions to Faiss!

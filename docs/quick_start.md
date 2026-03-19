# 快速入门<a name="ZH-CN_TOPIC_0000002552674671"></a>

本节提供快速上手Faiss核心功能的简易指导。

## 核心概念<a name="ZH-CN_TOPIC_0000002552616795"></a>

Faiss的核心是索引（Index），即专门用于存储向量并高效执行相似性搜索的结构：

- 向量：原生Faiss支持float32类型的向量。
- 索引类型：入门优先用IndexFlatL2（暴力搜索，无近似，结果精确，适合学习），后续可尝试IndexIVFFlat（近似搜索，速度更快）。
- 核心流程：创建索引 → 向索引添加向量 → 执行相似性搜索

## 快速入门示例<a name="ZH-CN_TOPIC_0000002552616795"></a>

以下是覆盖Faiss核心使用场景的完整代码，包含注释和结果解析：

```cpp
#include <faiss/IndexFlat.h>
#include <iostream>
#include <vector>
#include <random>

int main() {
    // 1. 准备数据
    int d = 128;
    int nb = 10000;
    int nq = 5;

    std::vector<float> db_vectors(nb * d);
    std::vector<float> query_vectors(nq * d);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : db_vectors) v = dist(rng);
    for (auto& v : query_vectors) v = dist(rng);

    // 2. 创建索引（IndexFlatL2：基于L2距离的暴力搜索）
    faiss::IndexFlatL2 index(d);

    // 3. 添加向量到索引
    index.add(nb, db_vectors.data());
    std::cout << "索引中向量数量: " << index.ntotal << std::endl;

    // 4. 执行搜索
    int k = 4;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> indices(nq * k);

    index.search(nq, query_vectors.data(), k, distances.data(), indices.data());

    // 5. 结果解析
    std::cout << "最近邻索引和对应距离:" << std::endl;
    for (int i = 0; i < nq; i++) {
        std::cout << "查询 " << i << ": ";
        for (int j = 0; j < k; j++) {
            std::cout << indices[i * k + j] << "(" << distances[i * k + j] << ") ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

预期结果如下：

<img src="figures/faiss-quick_start.jpg" alt="faiss-quick_start" width="600"/>

## 进阶扩展<a name="ZH-CN_TOPIC_0000002552616795"></a>

近似搜索索引（IndexIVFFlat）：适合大数据集，核心是先聚类再搜索。

```cpp
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <iostream>
#include <vector>
#include <random>

int main() {
    int d = 128;
    int nb = 10000;
    int nq = 5;
    int nlist = 100;

    std::vector<float> db_vectors(nb * d);
    std::vector<float> query_vectors(nq * d);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : db_vectors) v = dist(rng);
    for (auto& v : query_vectors) v = dist(rng);

    // 创建量化器和IVF索引
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(d);
    faiss::IndexIVFFlat ivf_index(quantizer, d, nlist);

    ivf_index.train(nb, db_vectors.data());
    ivf_index.add(nb, db_vectors.data());

    // 执行搜索
    int k = 4;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> indices(nq * k);
    ivf_index.search(nq, query_vectors.data(), k, distances.data(), indices.data());

    // 结果解析
    std::cout << "最近邻索引和对应距离:" << std::endl;
    for (int i = 0; i < nq; i++) {
        std::cout << "查询 " << i << ": ";
        for (int j = 0; j < k; j++) {
            std::cout << indices[i * k + j] << "(" << distances[i * k + j] << ") ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

预期结果如下：

<img src="figures/faiss-advanced.jpg" alt="faiss-advanced" width="600"/>

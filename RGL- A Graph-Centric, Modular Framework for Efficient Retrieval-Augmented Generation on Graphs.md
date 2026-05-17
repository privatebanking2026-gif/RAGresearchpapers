# RGL- A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs

### --- Page 0001 ---

```markdown
# RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs

**Yuan Li**  
National University of Singapore  
Singapore  
li.yuan@nus.edu  

**Jun Hu**  
National University of Singapore  
Singapore  
jun.hu@nus.edu.sg  

**Zemin Liu**  
Zhejiang University  
China  
liu.zemin@zju.edu.cn  

**Bryan Hooi**  
National University of Singapore  
Singapore  
bhooi@comp.nus.edu.sg  

**Jiaxin Jiang**  
National University of Singapore  
Singapore  
jxjiang@nus.edu.sg  

**Bingsheng He**  
National University of Singapore  
Singapore  
hebs@nus.edu.sg  

## Abstract
Recent advances in graph learning have paved the way for innovative retrieval-augmented generation (RAG) systems that leverage the inherent relational structures in graph data. However, many existing approaches suffer from rigid, fixed settings and significant engineering overhead, limiting their adaptability and scalability. Additionally, the RAG community has largely overlooked the decades of research in the graph database community regarding the efficient retrieval of interesting substructures on large-scale graphs. In this work, we introduce the RAG-on-Graphs Library (RGL), a modular framework that seamlessly integrates the complete RAG pipeline—from efficient graph indexing and dynamic node retrieval to subgraph construction, tokenization, and final generation—into a unified system. RGL addresses key challenges by supporting a variety of graph formats and integrating optimized implementations for essential components, achieving speedups of up to 143x compared to conventional methods. Moreover, its flexible utilities, such as dynamic node filtering, allow for rapid extraction of pertinent subgraphs while reducing token consumption. Our extensive evaluations demonstrate that RGL not only accelerates the prototyping process but also enhances the performance and applicability of graph-based RAG systems across a range of tasks.

**Keywords**  
Graph Neural Networks, Retrieval-Augmented Generation

## 1 Introduction
Recent advances in graph learning have witnessed an explosion of methods aimed at enhancing various facets of retrieval-augmented generation (RAG) on graphs [3, 9, 14]. Given a query, RAG retrieves relevant samples (context) from existing data and generates responses based on the retrieved information. Retrieval-augmented generation on graphs (RoG) extends RAG by leveraging graph structures to retrieve contextual information more effectively. Various applications on graphs, such as question answering, node classification, and recommendation—which contain rich structural data (e.g., user-item interactions [7], particle networks [6], and more [18]), can potentially benefit from RoG techniques [1, 5].

### General RAG-on-Graphs Pipeline
Given a graph—such as a social network or an E-commerce graph—we illustrate a typical RAG-on-Graph pipeline in Figure 1. The process begins with 1) **Indexing**, where nodes are organized for efficient access. Next, 2) **Node Retrieval** selects relevant nodes based on connectivity or attributes, after which 3) **Graph Retrieval** constructs subgraphs to capture local structures. These subgraphs are then converted into a sequential format during 4) **Tokenization**, rendering them compatible with state-of-the-art language models for the final 5) **Generation** stage. This pipeline underscores more advanced integration of graph data into RAG workflows.

Although the potential of RAG-on-Graphs is significant, its practical implementation remains challenging. First, many recent models are developed under fixed settings, limiting their adaptability to new datasets or the integration of novel components. For instance, GraphRAG [1] and LightRAG [3] assume textual input for constructing knowledge graphs, which restricts their support for customized graphs—such as social networks or E-commerce graphs—and consequently limits their flexibility. Second, the requirement to implement each stage from scratch not only increases the implementation burden but also diverts researchers from focusing on methodological variations. Finally, naive implementations of these stages can lead to efficiency pitfalls, particularly during the graph retrieval phase. This stage typically becomes a bottleneck, especially for 

![The pipeline of RAG-on-Graphs](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
![Time consumption of different graph retrieval implementations. The learning time involves forward and backward propagation operations, while the retrieval time is introduced by RAG-on-Graph operations.](assets/page_0002_img_1.png)

![An overview of the RGL toolkit.](assets/page_0002_img_2.png)

## 2 RGL Overview
RGL is a modular toolkit designed to streamline the development of RAG techniques on graph data graphs. As illustrated in Figure 3, RGL is composed of four primary components—Runtime, Kernel, API, and Applications—each providing specialized functionalities for efficient and flexible RAG-on-Graphs workflows.

### 2.1 RGL Kernel
The RGL Kernel provides fundamental components that handle graph data, retrieval, and generation processes. These components are carefully optimized to support various RAG scenarios, including indexing, high-performance retrieval, and batch processing.

#### 2.1.1 Graph Data Structure
RGL provides an intuitive Python interface for constructing and manipulating graph structures. Researchers can effortlessly build RGL graph objects using native Python objects. In addition, RGL ensures seamless conversions to and from popular frameworks such as DGL [17] and PyTorch Geometric (PyG) [2], allowing users to leverage the rich datasets available in these libraries.

#### 2.1.2 Node Retrieval
To facilitate semantic-level graph querying, RGL provides indexing and vector search utilities. Graph nodes and edges can be embedded into semantic vectors, enabling similarity-based retrieval that goes beyond simple keyword or ID lookups.

#### 2.1.3 Graph Retrieval
RGL implements a suite of efficient graph retrieval algorithms that leverage Python's ease-of-use features—empowered by extensive libraries like PyTorch and DGL—can yield efficient implementations, all connected via python bindings. This hybrid approach enables computationally intensive tasks—such as shortest-path computations, neighbor expansions, and subgraph extractions—to be offloaded to optimized C++ routines, resulting in performance improvements that significantly surpass those of Python-based libraries like NetworkX [4]. By batching operations, RGL reduces function call overhead and increases throughput, making it well-suited for large-scale graph processing tasks.

#### 2.1.4 Generation Interface
This interface bridges the gap between retrieved subgraphs and downstream language models. It handles tokenization, prompt construction, and generation calls.

### 2.2 RGL Runtime
The RGL Runtime manages resource allocation, caching, and parallelization across kernel components, abstracting distributed execution and memory management to ensure scalability and performance. It also integrates with popular graph learning frameworks.
```

### --- Page 0003 ---

```markdown
## 2.3 RGL API

RGL offers a dual-API approach—an Object Oriented Programming (OOP) API and a Functional API—to accommodate a wide range of development styles and use cases:

### 2.3.1 OOP API

The OOP API provides class-level interfaces for constructing, training, and deploying RGL workflows. These classes encapsulate data structures, retrieval logic, and generation calls.

### 2.3.2 Functional API

For more fine-grained control, the Functional API exposes key operations (e.g., subgraph extraction, embedding, tokenization) as composable functions. This design is especially useful for advanced scenarios, such as meta-learning or dynamic parameterization, where developers may need to inject custom logic at various stages of the pipeline.

### 2.3.3 Dataset Manager & Utilities

RGL includes utilities for handling various graph formats, loading and preprocessing data, and managing node or edge attributes. This streamlines the process of experimenting with new datasets and graph structures, reducing boilerplate code and speeding up development.

## 2.4 RGL Applications

Built on top of the kernel, runtime, and APIs, RGL Applications serve as end-to-end solutions for common tasks in graph-based RAG. Along with the open-source library, we provide the following demonstrative examples:

- **Completion**: Enhance data completion tasks using RGL by retrieving graph contexts to infer missing data more effectively. The framework's advanced analytics capabilities provide comprehensive insights into graph structure, which aids in accurate prediction and completion of incomplete datasets, thus bolstering model accuracy.

- **Summarization**: Utilize RGL for graph-based content summarization by employing subgraph extraction and generation models. RGL's efficient graph algorithms allow for fast identification of pivotal subgraph components, enabling thorough summarization strategies to organically generate concise summaries.

- **Graph Q&A**: Implement node- and graph-level question answering using RAG-on-Graphs by integrating the RGL framework. The RGL framework facilitates the extraction of relevant graph information for real-time question answering, supporting both intrinsic graph queries and comprehensive node-centric inquiries.

Developers can customize these applications or create entirely new ones by combining the RGL kernel modules with API interfaces. The result is a powerful, extensible toolkit that simplifies the entire lifecycle of RAG-on-Graphs applications.

## 3 Empirical Evaluation

In this section, we present an empirical evaluation across two challenging tasks: modality completion and abstract generation. By simulating realistic scenarios—ranging from sparse modality data in recommendation settings to prompt-driven text generation—we empirically show the efficacy of RAG-on-Graphs on various graph learning tasks.

### 3.1 Efficiency

Figure 4 reports the time consumption under different numbers of queries, averaged over all queries using a query windowed retrieval process for a certain node. We compare the standard graph library NetworkX [4] with algorithms implemented in RGL. The time consumption is separated into two components: 1) the learning time, which is an additional cost for a given dataset and typically involves the forward and backward computations; and 2) the retrieval process, which is an additional stage that augments the learning process with the retrieved contexts for the query nodes.

NetworkX suffers from steep retrieval costs. Its retrieval time grows dramatically with the number of queries. For example, on OGBN-Arxiv the baseline Steiner graph takes more than 11 hours to process 10,000 queries, rendering it infeasible for large-scale scenarios.

RGL offers efficient large-scale retrieval. RGL consistently exhibits short retrieval times, incurring only a minor additional overhead compared with the learning time. Specifically, RGL completes the same 10,000 queries on OGBN-Arxiv in under 5 minutes, indicating a drastic improvement compared with baselines.

These experiments confirm that NetworkX becomes prohibitively expensive beyond a few hundred queries, whereas RGL scales more gracefully. Given these observations, we adopt RGL for all subsequent performance evaluations, as its total runtime remains manageable for large-scale graph retrieval tasks.

![Time consumptions (s) vs. query counts across graph retrieval methods and datasets. The light colors denote the original training time, while the dark colors mark the additional time associated with graph retrieval.](assets/page_0003_img_1.png)
```

### --- Page 0004 ---

```markdown
| Method         | Baby      | Sports     |
|----------------|-----------|------------|
|                | R@20      | N@20       | R@20      | N@20       |
| Fill0          | 0.0902    | 0.0393     | 0.0972    | 0.0434     |
| NeighMean [16] | 0.0890    | 0.0393     | 0.0997    | 0.0445     |
| PPR [16]       | 0.0766    | 0.0395     | 0.0977    | 0.0439     |
| Diffusion [13] | 0.0476    | 0.0325     | 0.0860    | 0.0384     |
| kNN            | 0.0924    | 0.0405     | 0.0993    | 0.0446     |
| kNN-Neigh      | 0.0902    | 0.0393     | 0.0987    | 0.0444     |
| RGL-Steiner    | 0.0936    | 0.0405     | 0.1004    | 0.0449     |
| RGL-Dense      | 0.0932    | 0.0405     | 0.1005    | 0.0448     |
| RGL-BFS        | 0.0928    | 0.0405     | 0.1003    | 0.0450     |

| Method         | OGBN-Arxiv to Arxiv2025 |            |            |
|----------------|--------------------------|------------|------------|
|                | ROUGE-1                  | ROUGE-2    | ROUGE-L    |
| SelfNode       | 0.3791                   | 0.0754     | 0.1775     |
| kNN            | 0.3814                   | 0.0758     | 0.1796     |
| RGL-Steiner    | 0.3831                   | 0.0771     | 0.1790     |
| RGL-Dense      | 0.3789                   | 0.0720     | 0.1790     |
| RGL-BFS        | 0.3815                   | 0.0763     | 0.1801     |

### 3.2 Performance

#### 3.2.1 Modality Completion
In this section, we evaluate the performance of multi-modality completion on two challenging multimodal recommendation datasets. The goal is to recover missing modality-specific features, which is essential for enhancing downstream recommendation tasks when data is sparse or incomplete.

**Dataset Specification.** We evaluate our approach on two bipartite graphs with multimodal data. The Baby dataset comprises 15,145 users and 7,050 items, resulting in 16,792 recorded interactions. In contrast, the Sports dataset includes 35,595 users and 18,357 items with 296,337 interactions.

**Baselines.** Our experimental setup employs several completion methods. Baseline techniques include Fill0, NeighMean [16], PPR [16], Diffusion [13], kNN, and kNN-Neigh. In addition, we propose three variants of the RGL method based on different subgraph construction strategies: RGL-Steiner, RGL-Dense, and RGL-BFS.

**Evaluation.** We use the public data splits including training, validation, and test sets, following prior works [8, 20]. We simulate the missing modality scenarios by randomly masking a subset of features during training. We follow prior work [13] to set the missing rate to 40%, underscoring the importance of effective modality completion with sparse modality data. The recommendation performance is measured using Recall at 20 (R@20) and Normalized Discounted Cumulative Gain (N@20). We repeat all experiments 5 times on a V100-32GB GPU and report the mean scores.

**Results and Analysis.** Table 1 summarizes the performance of our method under varying missing rates and completion strategies. Our approach consistently outperforms all baselines across the datasets. In particular, the RGL-based subgraph construction methods (BFS, Dense, Steiner) yield the best performance in both recall and NDCG scores on all datasets. These findings validate the effectiveness of leveraging RAG-based graph techniques for multi-modality completion in sparse data scenarios.

#### 3.2.2 Abstract Generation
In this section, we compare abstract generation approaches across context construction methods and language models, demonstrating the effectiveness of RGL.

**Dataset Statistics.** For abstract generation, we leverage a large-scale citation network extracted from OGBN-Arxiv, which comprises 169,343 nodes, 1,157,799 edges, 128-dimensional features, and 40 classes. These real-world data points demand task-specific, synthesizing concise, informative abstracts from complex graph structures.

**Baselines and Prompted Contexts.** In addition to our proposed RGL variants (RGL-Steiner, RGL-Dense, and RGL-BFS), we consider two baselines: SelfNode and kNN. Furthermore, we evaluate the generation quality with two different large language models (LLMs): GPT-40-mini and DeepSeek-V3.

**Evaluation.** We inspect a zero-shot transfer scenario—OGBN-Arxiv to Arxiv2025—out to after the LLM knowledge cutoff dates (October 1, 2023 for GPT-40-mini, and July 1, 2024 for DeepSeek-V3) to avoid knowledge leakage. We evaluate ROUGE-1, ROUGE-2, and ROUGE-L [15] as our primary evaluation metrics, which respectively quantify the overlap of unigrams, bigrams, and longest common subsequences between generated and reference abstracts. These metrics provide insights into content fidelity at different levels of granularity.

**Results and Analysis.** Table 2 summarizes the performance of our methods on the OGBN-Arxiv to Arxiv2025 task. The key findings are as follows:
- When utilizing the GPT-40-mini model, RGL-Steiner achieves the highest ROUGE-1 and ROUGE-2 scores, whereas RGL-BFS leads in ROUGE-L. In contrast, with the DeepSeek-V3 model, RGL-Dense attains the top ROUGE-1 score, and RGL-BFS continues to excel in ROUGE-L.
- These results demonstrate that our RGL framework effectively leverages both graph structure and contextual cues, thereby producing abstracts that are both coherent and highly representative of the source content.
- The variance in performance across different graph traversal strategies (Steiner, BFS, Dense) with varied modeling techniques (GPT-40-mini, DeepSeek-V3) suggests that the...
```

### --- Page 0005 ---

```markdown
# RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs

RGL framework’s adaptability is crucial for optimizing abstract generation tasks.

## 4 Conclusions

In this paper, we introduced the RAG-on-Graphs Library (RGL), a modular and highly adaptable toolkit designed to streamline the integration of graph data into retrieval-augmented generation systems. Our experimental results, spanning modality completion and abstract generation tasks, convincingly demonstrate that RoG enhances graph learning performance. Specifically, it delivers notable speedups in graph retrieval processes and markedly improves the quality of the generated content. By integrating optimized graph processing techniques, providing flexible APIs, and ensuring seamless interfacing with state-of-the-art language models, RGL establishes a solid platform for advancing research in RoG applications.

Looking ahead, several avenues for future work can be identified. Expanding the library to include a broader range of examples can facilitate better understanding and implementation. Furthermore, efforts to enhance user-friendliness will make RGL more accessible to a wider audience. Large-scale testing is necessary to further validate the robustness and scalability of the library. Additionally, exploring integration with other graph databases could provide insightful synergies, thereby extending RGL’s applicability in diverse graph environments.

---

## References

[1] Zhang, Y., Wu, Y., Newman, C., Cheng, J., Housman, A., Bradley, A., Chao, A. Apurva Mody, Steven Tritt, and Jonathan Larson. 2024. From local to global: A graph retrieval approach to user-level summarization. arXiv preprint arXiv:2401.1630.

[2] Hamilton, W. L., and Eric J. L. Lensen. 2019. Graph representation learning with PyTorch Geometric. arXiv preprint arXiv:1903.02428.

[3] Zhuang, L., Zhang, Y., Wan, Y., Zhang, T., and Chao Huang. 2020. LeHARCG: Simple and Fast Retrieval-Augmented Generation. arXiv preprint arXiv:2401.0579.

[4] Liu, H., Yang, Y., and Sandal, A. 2021. Exploring network structure, dynamics, and functions using NetworkX. Technical Report. Los Alamos National Laboratory. LA-UR-21-0001, United States.

[5] Xie, S., and Y. Zhang. 2021. G-Transformer: Retrieval-augmented generation for contextual graph understanding and question answering. Advances in Neural Information Processing Systems 37 (2023): 127867-127902.

[6] Liu, H., Yang, Y., and Riggins, E. 2024. Efficient Heterogeneous Graph Learning via Random Projection. IEEE Trans. Knowl. Data Eng. 36, 12 (2024), 1095–1107. DOI: https://doi.org/10.1109/TKDE.2024.3194659.

[7] Jin, H., Bryon, B., Brigham, E., and Yim, W. 2024. Modality-Independent Graph Neural Networks with Global Transformers for Multimodal Recommendation. arXiv:2412.1194. https://arxiv.org/abs/2412.1194.

[8] Jin, H., Bryon, B., Brigham, E., and Yim, W. 2024. Modality-Independent Graph Neural Networks with Global Transformers for Multimodal Recommendation. arXiv preprint arXiv:2412.1194.

[9] Yutong, H., Zhizhan, L., Zeyu, Z., Feng, C., Chen, Ling, and Lian Zhao. 2024. Graph retrieval-augmented generation. arXiv preprint arXiv:2405.1656.

[10] Zhang, B., Byron, C., Xuan, H., Huang, Jianing, and Sourav S. Bhownick. 2023. A distributed system for keyword search on massive graphs. IEEE Transactions on Knowledge and Data Engineering 36, 5 (2023), 1795–1799.

[11] Jiang, F., Yao, C., Jianing, K., and Sourav S. Bhownick. 2019. A generic framework for indexing keyword search on massive graphs. IEEE Transactions on Knowledge and Data Engineering 31, 6 (2019), 2322–2336.

[12] Zhang, J., Wang, H., Byron, C., Jianing, K., Sourav S. Bhownick, and Lyu. 2021. How to design an efficient framework for keyword search on public-private networks. In 2021 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 457–465.

[13] Liu, H., Su, W., Wang, Q., Zhang, Shu, Yu, and Fang Chen. [n.d.]. Generating with Fairness: A Modality-Disturbed Counterfactual Framework for Incomplete Multimodal Recommendations. In THE WEB CONFERENCE 2025.
```



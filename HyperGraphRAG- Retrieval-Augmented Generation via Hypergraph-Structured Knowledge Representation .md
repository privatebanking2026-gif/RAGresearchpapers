# HyperGraphRAG- Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation 

### --- Page 0001 ---

```markdown
# HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation

**Haoran Luo\textsuperscript{1,2}, Haihong E\textsuperscript{1}, Guanting Chen\textsuperscript{1}, Yandan Zheng\textsuperscript{2}, Xiaobao Wu\textsuperscript{3}, Yikai Guo\textsuperscript{3}, Qika Lin\textsuperscript{1}, Yu Feng\textsuperscript{5}, Zemin Kuang\textsuperscript{6}, Meina Song\textsuperscript{1}, Yifan Zhu\textsuperscript{1}, Luu Anh Tuan\textsuperscript{2}**  
1 Beijing University of Posts and Telecommunications  
2 Nanyang Technological University  
3 Beijing Institute of Computer Technology and Application  
4 National University of Singapore  
5 China Mobile Research Institute  
6 Beijing Anzhen Hospital, Capital Medical University  
haoran.luo@ieee.org, haihong@bupt.edu.cn, anhtuan.luu@ntu.edu.sg

## Abstract

Standard Retrieval-Augmented Generation (RAG) relies on chunk-based retrieval, whereas GraphRAG advances this approach by graph-based knowledge representation. However, existing graph-based RAG approaches are constrained by binary relations, as each edge in an ordinary graph connects only two entities, limiting their ability to represent the n-ary relations ($n \geq 2$) in real-world knowledge. In this work, we propose HyperGraphRAG, the first hypergraph-based RAG method that represents n-ary relational facts via hyperedges. HyperGraphRAG consists of a comprehensive pipeline, including knowledge hypergraph construction, retrieval, and generation. Experiments across medicine, agriculture, computer science, and engineering demonstrate that HyperGraphRAG outperforms both standard RAG and previous graph-based RAG methods in answer accuracy, retrieval efficiency, and generation quality. Our data and code are publicly available\footnote{Corresponding author. \url{https://github.com/LRLAB/HyperGraphRAG}}.

## 1 Introduction

Retrieval-Augmented Generation (RAG) \cite{10, 6} has advanced knowledge-intensive tasks by integrating knowledge retrieval with large language models (LLMs) \cite{17, 28}, thereby enhancing factual awareness and generation accuracy. Standard RAG typically relies on chunk-based retrieval, segmenting documents into fixed-length text chunks retrieved via dense vector similarity, which overlooks the relationships between entities. Recently, GraphRAG \cite{2} has emerged as a promising direction that structures knowledge as a graph to capture inter-entity relations, with the potential to improve retrieval efficiency and knowledge-driven generation \cite{18}.

However, since each edge in an ordinary graph connects only two entities, existing graph-based RAG approaches \cite{2, 7, 1, 8} are all restricted to binary relations, making them insufficient for modeling the n-ary relations among more than two entities that are widespread in real-world domain knowledge \cite{25}. For example, in the medical domain, as illustrated in Figure 2, representing...

![An illustration of HyperGraphRAG.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
![Comparison of knowledge representation: standard RAG uses chunks as units, GraphRAG captures binary relations with graphs, and HyperGraphRAG models n-ary relations with hyperedges.](assets/page_0002_img_1.png)

2 Related Work

Graph-based RAG. GraphRAG [2] is the first graph-based RAG method that improves LLM generation via graph-based retrieval. Based on GraphRAG, several methods [26, 22, 11, 23] focus on building graph-based RAG for different applications. LightRAG [7] enhances efficiency via graph indexing and updates. PathRAG [1] and HippoRAG [2] refine retrieval with path pruning and Personalized PageRank. However, all rely on binary relations, limiting knowledge expressiveness. In this work, we propose HyperGraphRAG, the first graph-based RAG method via hypergraph-structured knowledge representation. We compare several existing methods with HyperGraphRAG in Table 1.

Hypergraph Representation. Hypergraph-structured knowledge representation aims to overcome ordinary graph's limitations in modeling n-ary relations [15]. Early methods [25, 27, 2, 21] employ various embedding techniques to represent n-ary relational entities. Later methods [3, 24, 14] utilize GNN or attention to enhance embedding. However, existing methods mainly focus on link prediction, while hypergraphs also show potential for enhancing knowledge representation in graph-based RAG.
```

### --- Page 0003 ---

```markdown
| Method            | Knowledge Construction                                                                 | Knowledge Retrieval                                                                 |
|-------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| NaiveGeneration    | $K = \{c\}^*_{c \in C}$, where $c$ is a chunk.                                      | $K^*_r = \emptyset$                                                                  |
| StandardRAG       | $K = \{c\}^*_{c \in C}$, where $c$ is a chunk.                                      | $K^*_r = K_{top}, \{c \in K | sim(h_q, h_c)\}$                                     |
| GraphRAG [2]      | $K = \{S = (V, E) | g \in Community(C)\}$, where $S$ is the community summary set. | $K^*_r = \text{Det}(S \leq S) | U_{Krank}$, where entities & relations are retrieved with chunks. |
| LinkRAG [7]       | $K = G = (V, E)$, where $G$ is the same as LightRAG's.                              | $K^*_r = \text{Prune}(p \in P | q)$, where relational paths are retrieved via pruning. |
| PathRAG [8]       | $K = G = (V \cup M, E)$, where $V$ & $M$ are phrase & passage nodes.               | $K^*_r = \text{PageRank}(m \in M | p)$, where passages are retrieved via Personalized PageRank. |
| HyperGraphRAG (ours) | $K = G_{H} = (V, E_{H})$, where $G_H$ is structured as a hypergraph.              | $K^*_r = \text{Retrieve}(e \in V | j \in E_{H}) | U_{Krank}$, where n-ary relational facts are retrieved with chunks. |

## 3 Preliminaries

**Definition 1: RAG.** Given a question $q$ and domain knowledge $K$, standard RAG first selects relevant document fragments from $K$ based on $q$ and then generates an answer $y$ based on $q$ and $d$. The probability model is formulated as:

$$
P(y|q) = \sum_{d \in K} P(y|g, d)P(d|q, K). \tag{1}
$$

**Definition 2: Graph-based RAG.** Graph-based RAG optimizes retrieval by representing knowledge as a graph structure $G = (V, E)$, where $V$ is the set of entities and $E$ is the set of relationships between entities. $G$ consists of facts represented as $F = (e, e') \in E$, where $e$ is the relational entity and $e'$ is the entity set connected to $e$. Given a question, the retrieval process is defined as:

$$
P(y|q) = \sum_{F \in G} P(y|g, F)P(F|q, G). \tag{2}
$$

**Definition 3: Hypergraph.** A hypergraph $G_H = (V, E_H)$ is a generalized graph, where $V$ is the entity set, $E_H$ is the hyperedge set, and each hyperedge $e_H \in E_H$ connects 2 or more entities: 

$$
V_{e_H} = (v_1, v_2, \ldots, v_n), \quad n \geq 2. \tag{3}
$$

Unlike ordinary graphs, where relationships are binary $V_e = (v_h, v_t)$, hypergraphs model n-ary relational facts $F_n = (e_H, V_{e_H}) \in G_H$.

## 4 Method: HyperGraphRAG

In this section, we introduce the proposed HyperGraphRAG, as shown in Figure 3, including knowledge hypergraph construction, hypergraph retrieval strategy, and hypergraph-guided generation.

### 4.1 Knowledge Hypergraph Construction

To represent and store knowledge, we propose a knowledge hypergraph construction method that includes n-ary relational extraction, bipartite hypergraph storage, and vector representation storage.

**N-ary Relation Extraction.** To construct the knowledge hypergraph $G_H$, our first step is to extract multiple n-ary relational facts $F_n$ from natural language documents $d \in K$. Unlike traditional hyper-relations [21], events [13], or other n-ary relation models [15], in the era of LLMs, to preserve richer and more diverse n-ary relations among entities, we propose a new n-ary relation representation $F_n = (e_H, V_{e_H})$, utilizing natural language descriptions, instead of structured relations, to represent hyperedges $e_H$ among multiple entities $V_{e_H}$, as follows:

(a) **Hyperedge:** Given an input text $d$, it is parsed into several independent knowledge fragments, each treated as a hyperedge: $E_H^d = \{e_1, e_2, \ldots, e_k\}$. Each hyperedge $e_i = (e_{i}^{text}, score_{i})$ consists of two parts: a natural language description $e_{i}^{text}$, and a confidence score $score_{i} \in (0, 10]$ indicating the association degree between $e_i$ and $d$.
```

### --- Page 0004 ---

```markdown
![An overview of HyperGraphRAG, which constructs a knowledge hypergraph from domain knowledge, retrieves n-ary facts based on user questions, and generates knowledgeable responses.](assets/page_0004_img_1.png)

(b) **Entity**: For each hyperedge $e_i$, entity recognition is performed to extract all contained entities: $V_{e_i} = \{v_1, v_2, \ldots, v_n\}$, where $v_j = (name_j^{(e)}, explain_j^{(e)}, score_j^{(e)})$ consists of four parts: entity name $name_j^{(e)} \subseteq text^{(e)}$, type $type_j^{(e)}$, explanation $explain_j^{(e)}$, and confidence score $score_j^{(e)} \in (0, 100]$ indicating the extraction certainty.

Following this hypergraph-structured knowledge representation, we design an n-ary relation extraction prompt $p_{ext}$ detailed in Appendix A.1, to enable the LLM to perform end-to-end knowledge fragment segmentation and entity recognition, thereby forming the n-ary relational fact set $F_n^{e_i}$:

$$
F_n^{e_i} = \{f_1, f_2, \ldots, f_k\} \sim \pi(F_n^{text}, d), \tag{4}
$$

where each extracted n-ary relational fact $f_k = (e_i, V_{e_i})$ contains information about the corresponding hyperedge $e_i$ and its associated entity set $V_{e_i}$. We convert all documents $d \in$ hyperedges and entities using n-ary relation extraction, forming a complete knowledge hypergraph $G_H$.

**Proposition 1.** Hypergraph-structured knowledge representation is more comprehensive than binary.

**Proof.** We provide experimental results in Section 5.4 and proofs in Appendix B.1.

**Bipartite Hypergraph Storage.** After n-ary relation extraction, we store the constructed knowledge hypergraph $G_H$ in a graph database to support an efficient query. We adopt an ordinary graph database represented as a bipartite graph structure $G_B = (V_B, E_B) = \Phi(G_H)$, to store the knowledge hypergraph $G_H = (V, E_H)$ where $\Phi$ is a transformation function defined as:

$$
\Phi: V_B = V \cup E_H, \quad E_B = \{(e_{H}, v) | e_{H} \in E_H, v \in V_B\}, \tag{5}
$$

where $V_B$ is the set of nodes in $G_B$, formed by merging the entity set $V$ and the hyperedge set $E_H$ from $G_H$. The edge set $E_B$ captures the connections between each hyperedge $e_H \in E_H$ and its associated entities $v \in V_{e_H}$.

Based on $G_B$, we can efficiently query all entities associated with a hyperedge $e_H$ or query all hyperedges linked to a specific entity $v$, thereby benefiting the optimized query efficiency of an ordinary graph database, as well as preserving the complete hypergraph-structured knowledge representation.

Moreover, $G_B$ allows incremental updates through dynamically expansion: $G_B' = G_B \cup \Phi(G_H')$. Here, $G_H'$ represents newly added hypergraph information. The transformation of hyperedges and entities into the bipartite graph storage format enables seamless updates to the graph database.

**Proposition 2.** A bipartite graph can losslessly preserve and query a knowledge hypergraph.

**Proof.** We provide proofs in Appendix B.2.
```

### --- Page 0005 ---

```markdown
## Vector Representation Storage

To support efficient semantic retrieval, we embed hyperedges $e_H \in E_H$ and entities $v \in V$ using the same embedding model $f$, ensuring that the vector representation of hyperedges and entities is in the same vector space as questions. Let $\psi$ be the vector function, then the vector representation storage for the knowledge hypergraph $G_H$ is defined as:

$$
\Psi(G_H) = (E_H, E_V), 
$$

where $E_H$ is the vector base of hyperedges and $E_V$ is the vector base of entities:

$$
\psi : E_H = \{h_e | e_H \in E_H\}, \quad E_V = \{h_v | v \in V\}, 
$$

where each hyperedge $e_H$ and entity $v$ in $G_H$ is embedded into their vector representations: $h_{e_H}$ and $h_v = f(v)$, respectively.

### 4.2 Hypergraph Retrieval Strategy

After constructing and storing the hypergraph $G_H$, we design an efficient retrieval strategy to match user questions with relevant hyperedges and entities.

#### Entity Retrieval

First, we extract key entities from the question $q$ to facilitate subsequent matching. We design an entity extraction prompt $p_{ext}$ (detailed in Appendix A.2, along with the LLM $\pi$ to extract the entity set $V_q$:

$$
V_q \sim \pi(V_{p_{ext}}(q)). 
$$

After extracting entities, we retrieve the most relevant entities from the entity set $V$ of the knowledge hypergraph $G_H$. We define the entity retrieval function $R_V$, which retrieves the most relevant entities from $E_V$ using cosine similarity:

$$
R_V(q) = \argmax_{v \in E_V} \left( \text{sim}(h_V, h_e) \odot \text{score}_{\text{rev}} \right) > \tau_r,
$$

where $h_V = f(V_q)$ is the concatenated text vector representation of the extracted entity set $V_q$, $h_e \in E_H$ is the vector representation of entity $e_H$, $\text{sim}(\cdot, \cdot)$ denotes the similarity function, $\odot$ represents element-wise multiplication between similarity and entity relevance score $\text{score}_{\text{rev}}$ determining the final ranking score, $\tau_r$ is the threshold for the entity retrieval score, and $k_V$ limits the number of retrieved hyperedges.

#### Hyperedge Retrieval

Moreover, to expand the retrieval scope and capture complete n-ary relations within the hyperedge set $E_H$ of the knowledge hypergraph $G_H$, we define the hyperedge retrieval function $R_H$, which retrieves a set of hyperedges related to $q$:

$$
R_H(q) = \argmax_{e \in E_H} \left( \text{sim}(h_q, h_e) \odot \text{score}_{H} \right) > \tau_H,
$$

where $h_q = f(q)$ is the text vector representation of $q$, $h_e \in E_H$ is the vector representation of the hyperedge $e_H$, $\text{score}_{H}$ determines the final ranking score, $\tau_H$ is the threshold for the hyperedge retrieval score, and $k_H$ limits the number of retrieved hyperedges.

### 4.3 Hypergraph-Guided Generation

To fully utilize the structured knowledge in the hypergraph, we propose a Hypergraph-Guided Generation mechanism, which consists of hypergraph knowledge fusion and generation augmentation.

#### Hypergraph Knowledge Fusion

The primary goal of hypergraph knowledge fusion is to expand and reorganize the retrieved n-ary relational knowledge to form a comprehensive knowledge input. Since $q$ may only match partial entities or hyperedges, we further expand the retrieval scope. To obtain a complete set of n-ary relational facts, we design a bidirectional expansion strategy that includes expanding hyperedges from retrieved entities and expanding entities from retrieved hyperedges.

First, given the entity set retrieved from $q$, denoted as $R_V(q) = \{v_1, v_2, \ldots, v_k\}$, we retrieve all hyperedges in the knowledge hypergraph $G_H$ that connect these entities:

$$
\overline{F}_H = \bigcup_{e \in R_V(q)} \{(e_{i}, V_{e_i}) | e_i \in V_{e_i}, e_H \in E_H\}. 
$$

Next, we expand the set of entities connected to the retrieved hyperedges $R_H(q) = \{e_1, e_2, \ldots, e_k\}$:

$$
\overline{H}_H = \bigcup_{e_i \in R_H(q)} \{(e_{i}, V_{e_i}) | V_e \subseteq V\}. 
$$
```

### --- Page 0006 ---

```markdown
Finally, we merge the expanded hyperedge set $F^v$ with the expanded entity set $F^{h}$ to form a complete retrieved n-ary relational fact $K^H = F^v \cup F^{h}$. This set contains all necessary n-ary relational knowledge for reasoning and generation, ensuring a comprehensive input for the LLM.

## Generation Augmentation

Following hypergraph knowledge fusion, we augment the generation strategy to improve the accuracy and readability of the responses. We adopt a hybrid RAG fusion mechanism, combining hypergraph knowledge $K^H$ with retrieved chunk-based text fragments $K_{chunk}$ to form the final knowledge input. We define the final knowledge input $K^*$ as:

$$
K^* = K^H \cup K_{chunk},
$$

where $K_{chunk}$ consists of chunk-based text fragments retrieved using traditional RAG.

Finally, we use a retrieval-augmented generation prompt $p_{gen}$, detailed in Appendix A.3, that combines hypergraph knowledge $K^*$ and the user question $q$ as input to LLM to generate final response $y^*$:

$$
y^* \sim \mathcal{N}(y|p_{gen}, K^*, q).
$$

### Proposition 3

Retrieving knowledge on a knowledge hypergraph improves retrieval efficiency compared to methods based on ordinary binary graphs, leading to gains in generation quality.

**Proof.** We provide experimental results in Sections 5.5 and 5.6 and proofs in Appendix B.

# 5 Experiments

This section presents the experimental setup, main results, and analysis. We answer the following research questions (RQs): RQ1: Does HyperGraphRAG outperform other methods? RQ2: Does the main component of HyperGraphRAG work? RQ3: How effective is the knowledge hypergraph constructed by HyperGraphRAG across various domains? RQ4: Could the hypergraph retrieval strategy improve retrieval efficiency? RQ5: How effective is the generation quality of HyperGraphRAG? RQ6: How are the time and cost of HyperGraphRAG in construction and generation phases?

## 5.1 Experimental Setup

### Datasets

To evaluate the performance of HyperGraphRAG across multiple domains, we select four knowledge contexts from UltraDomain [7], as used in LightRAG [7]: Agriculture, Computer Science (CS), Legal, and a mixed domain (Mix). In addition, we include the latest international hypervision guidelines [16] as the foundational knowledge for the Medicine domain. For each of the five domains, we sample knowledge fragments one, two, and three ways to construct questions with ground-truth answers verified by human annotators. We then categorize the questions into Binary Source and N-ary Source, based on whether the sampled knowledge of the question contains facts among $n$ entities ($n > 2$). More details can be found in Appendix D.

### Baselines

We compare HyperGraphRAG against six publicly available baseline methods: Naive-Generation [17], which directly generates responses using LLM; Standard RAG [6], a traditional chunk-based RAG approach; GraphRAG [2], LightRAG [7], PathRAG [1], and HyperBGRAG [8], which are the four selected available graph-based RAG methods described in Table 1. To ensure fairness, we use the same generation prompt, which can be found in Appendix E.

### Evaluation Metrics

We evaluate the answer accuracy, retrieval efficiency, and generation quality of HyperGraphRAG and its baselines using 3 key metrics: F1, Retrieval Similarity (R-S), and Generation Evaluation (G-E). F1 measures word-level similarity between the generated answer and the ground-truth answer, following FlashRAG [9]. R-S assesses the semantic similarity between retrieved knowledge and the ground-truth knowledge used to construct the question, in line with RAGAS [3]. G-E, inspired by HelloBench [20], is a metric that uses LLM-as-a-judge to evaluate generation quality in 7 dimensions and reports the average score. Details are provided in Appendix E.

### Implementation Details

We use OpenAI's GPT-4.0-m for extraction and generation, and text-embedding-davinci-003 for vector. During retrieval, we set the following parameters: entity retrieval $k_{E} = 60$, $r = 50$; hyperedge retrieval $k_{H} = 60$, $t = 5$; and chunk retrieval $k_{C} = 5$, $t = 0$. All experiments were conducted on a server with an 80-core CPU and 512GB RAM.

## 5.2 Main Results (RQ1)

To evaluate the effectiveness of HyperGraphRAG, we compare its performance with various baselines across multiple domains. The results are shown in Table 2.
```

### --- Page 0007 ---

```markdown
| Method            | Binary Source |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |
|-------------------|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|                   | R-S           | G-E   | F1    | R-S   | G-E   | F1    | R-S   | G-E   | F1    | R-S   | G-E   | F1    | R-S   | G-E   | F1    | R-S   | G-E   | F1    | R-S   | G-E   | F1    | R-S   | G-E   | F1    | R-S   | G-E   | F1    | R-S   | G-E   | F1    |
|-------------------|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| NaiveGeneration    | 12.63        | 0.00  | 44.70 | 17.11 | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 |
| StandardRAG       | 26.71        | 0.68  | 52.41 | 26.31 | 42.90 | 57.68 | 28.37 | 49.44 | 17.19 | 52.91 | 25.77 | 47.57 | 28.37 | 49.44 | 17.19 | 52.91 | 25.77 | 47.57 | 28.37 | 49.44 | 17.19 | 52.91 | 25.77 | 47.57 | 28.37 | 49.44 | 17.19 | 52.91 | 25.77 | 47.57 |
| GraphRAG          | 17.63        | 5.84  | 56.19 | 20.67 | 40.92 | 52.94 | 23.73 | 57.65 | 53.71 | 30.14 | 52.64 | 26.26 | 42.62 | 25.41 | 51.12 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 |
| LightRAG          | 12.16        | 52.48 | 41.75 | 17.42 | 20.52 | 41.86 | 51.32 | 43.65 | 43.54 | 62.98 | 32.42 | 56.40 | 33.66 | 47.08 | 33.66 | 47.08 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 |
| PathRAG           | 14.72        | 53.20 | 43.56 | 17.21 | 42.31 | 21.49 | 25.82 | 32.43 | 36.45 | 56.42 | 29.38 | 32.43 | 36.45 | 56.42 | 29.38 | 32.43 | 36.45 | 56.42 | 29.38 | 32.43 | 36.45 | 56.42 | 29.38 | 32.43 | 36.45 | 56.42 | 29.38 | 32.43 | 36.45 | 56.42 |
| HippoRAG          | 21.27        | 57.10 | 51.82 | 16.28 | 16.94 | 21.05 | 29.70 | 12.30 | 14.67 | 21.70 | 13.14 | 16.37 | 21.70 | 13.14 | 16.37 | 21.70 | 13.14 | 16.37 | 21.70 | 13.14 | 16.37 | 21.70 | 13.14 | 16.37 | 21.70 | 13.14 | 16.37 | 21.70 | 13.14 | 16.37 |
| HyperGraphRAG     | 36.45        | 69.91 | 60.65 | 34.60 | 61.97 | 59.99 | 34.16 | 69.04 | 57.54 | 44.62 | 60.37 | 63.51 | 51.51 | 67.34 | 68.76 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 | 12.91 | 52.00 | 18.00 | 0.00  | 18.00 | 79.29 |

Overall Comparison Across Domains. HyperGraphRAG consistently outperforms all baselines across F1, R-S, and G-E metrics. Compared to StandardRAG, it achieves gains of +7.1, +4.62 (R-S), and +6.39 (G-E). Interestingly, existing graph-based RAG baselines often underperform StandardRAG, as their reliance on binary relational graphs causes knowledge fragmentation, sparsified retrieval, and incomplete context reconstruction during generation.

Comparison Across Source Types. HyperGraphRAG maintains strong gains under both Binary and N-ary settings. For Binary Source, it improves F1, R-S, and G-E by +8.6, +8.8, and +4.4; for N-ary source, the improvements are +5.3, +6.4, and +2.9, confirming its robustness.

### 5.3 Ablation Study (RQ2)

As shown in Figure 4, we conducted an ablation study in the Medicine domain by removing entity retrieval (w/o ER), hyperedge retrieval (w/o HR), and their combination (w/o ER & HR). We also remove chunk retrieval fusion (w/o CR), and all modules (w/o ER & HR & CR):

**Impact of Entity Retrieval (ER).** ER is critical for precise retrieval by anchoring key concepts. Without ER, F1 falls from 35.4 to 29.8, underscoring its importance in selecting relevant entities for accurate generation.

**Impact of Hyperedge Retrieval (HR).** HR captures n-ary, multi-entity facts necessary for complex reasoning. Removing HR drops F1 from 35.4 to 26.4, highlighting its unique role beyond mere entity retrieval.

**Impact of Chunk Retrieval Fusion (CR).** CR enhances retrieval by integrating unstructured text with hypergraph data. Excluding CR reduces F1 from 35.4 to 29.2, demonstrating that the fusion leads to more complete and fluent generation.

![Results of the ablation study](assets/page_0007_img_1.png)
```

### --- Page 0008 ---

```markdown
5.4 Analysis of Hypergraph-structured Knowledge Representation (RQ3)

As shown in Figure 5, we assess HyperGraphRAG's knowledge representation across 5 domains:

![Medicine HyperGraph](assets/page_0008_img_1.png) ![Agriculture HyperGraph](assets/page_0008_img_2.png) ![CS HyperGraph](assets/page_0008_img_3.png)  
![Legal HyperGraph](assets/page_0008_img_4.png) ![Mix HyperGraph](assets/page_0008_img_5.png) ![Statistics of Construction](assets/page_0008_img_6.png)  

Figure 5: (a-e) Visualizations of knowledge hypergraphs constructed in 5 domains. (f) Statistical comparison highlights HyperGraphRAG's richer expressiveness over GraphRAG and LightRAG.

### Visualization of Knowledge Structures
As shown in Figure 5(a)-5(e), unlike previous graph-based RAG methods, which only model binary relations, HyperGraphRAG connects multiple entities via hyperedges, forming a more interconnected and expressive network.

### Statistical Analysis
As shown in Figure 5(f), HyperGraphRAG surpasses GraphRAG and LightRAG in all domains. For instance, in CS, it constructs 26,902 hyperedges, whereas GraphRAG has 930 communities and LightRAG 5,632 relations, showing a stronger capacity for capturing knowledge.

5.5 Analysis of Hypergraph Retrieval Efficiency (RQ4)

As shown in Figure 6, to evaluate retrieval efficiency, we conducted two experiments: (a) examining how HyperGraphRAG's retrieval efficiency and token length scales with different top-k values and (b) comparing its F1 scores with other methods under varying retrieval length limits:

![Impact of Top-k on Retrieval Efficiency & Token Length](assets/page_0008_img_7.png) ![F1 Comparison under Limited Lengths](assets/page_0008_img_8.png)  

Figure 6: Experimental results in the Medicine domain analyzing hypergraph retrieval efficiency.

### Impact of Retrieved Hyperedge Quantity
As shown in Figure 6(a), increasing the top-k hyperedges improves F1, R-S, and G-E, along with the rise in token count. Performance saturates around k = 60, indicating that HyperGraphRAG achieves strong retrieval quality with limited input.

### Performance under Constrained Retrieval Length
As illustrated in Figure 6(b), HyperGraphRAG outperforms all binary graph-based RAG methods even under retrieval length limits, demonstrating the efficiency of n-ary representations and highlighting the semantic loss inherent in binary structures.
```

### --- Page 0009 ---

```markdown
## 5.6 Analysis of Hypergraph-Guided Generation Quality (RQ5)

As shown in Figure 7, we evaluate the quality of the generation in seven dimensions:

### Best Overall Generation Quality
HyperGraphRAG achieves the highest Overall score (61.5), significantly outperforming all baseline methods, indicating the comprehensive advantage in hypergraph-guided generation.

### Lead on Key Dimensions
HyperGraphRAG achieves notable improvements in Correctness (64.8), Relevance (66.0), and Factuality (64.2), outperforming both standard RAG and binary graph-based methods. These gains indicate its strong capacity to produce accurate, context-aware, and knowledge-grounded responses.

### Balanced Performance
Although the Diversity score (47.0) is relatively lower than other dimensions, HyperGraphRAG still exceeds all baselines, indicating that it maintains a balanced dimension-wise performance, effectively combining content richness with structural consistency for stable and high-quality generation.

![Generation Equality Evaluations](assets/page_0009_img_1.png)

## 5.7 Analysis of Time and Cost in Construction and Generation Phases (RQ6)

As shown in Table 3, to evaluate the efficiency and cost of HyperGraphRAG, we compare different methods in terms of knowledge construction and generation. We assess time consumption per 1k tokens (TpiKT), cost per token (CpiKT), time per query (TPQ), and cost per query (CPIKQ).

### Time & Cost in Construction Phase
HyperGraphRAG demonstrates efficient knowledge construction with a time cost of 3.084 seconds per 1k tokens (TpiKT) and a monetary cost of $0.0063 per token (CpiKT). This places it between the faster HippoRAG (2.675s, $0.0056) and slower GraphRAG (9.272s, $0.0085). While its cost is slightly higher than GraphRAG, HyperGraphRAG achieves a better balance between speed, expressiveness, and structure, offering a more compact yet rich representation of n-ary relational knowledge.

### Time & Cost in Generation Phase
During the generation phase, HyperGraphRAG requires 0.256 seconds per query (TPQ) and incurs a cost of $3.184 per 1k queries (CPIKQ). This is moderately higher than StandardRAG ($0.147, $1.016) but significantly lower than PathRAG ($3.496) and LightRAG ($0.359, $3.359). Compared to GraphRAG (0.221s, $1.360), HyperGraphRAG slightly increases time and cost but compensates with better retrieval quality and generation outcomes. The results suggest that HyperGraphRAG achieves a favorable trade-off between generation efficiency and output quality, suitable for real-world knowledge-intensive applications.

| Method        | TPiKT  | Construction | Generation |
|---------------|--------|--------------|------------|
| NaiveGeneration| 0 s   | 0 s          | 0.131 s    | 0.0059    |
| StandardRAG  | 0 s    | 0 s          | 0.147 s    | 1.016 s   |
| GraphRAG     | 9.272 s| 0.0085 s     | 2.221 s    | 1.363 s   |
| LightRAG     | 5.168 s| 0.0011 s     | 0.359 s    | 3.359 s   |
| PathRAG      | 5.168 s| 0.0081 s     | 0.436 s    | 3.496 s   |
| HippoRAG     | 2.675 s| 0.0065 s     | 0.306 s    | 3.483 s   |
| HyperGraphRAG| 3.084 s| 0.0063 s     | 0.256 s    | 3.184 s   |

## 6 Conclusion

In this work, we present HyperGraphRAG, a retrieval-augmented generation framework that models knowledge as hypergraphs to capture n-ary relational structures. By introducing novel methods for knowledge representation, retrieval, and generation, HyperGraphRAG addresses limitations of binary graph-based RAG methods. Experimental results across diverse domains demonstrate consistent improvements in answer accuracy, retrieval relevance, and generation quality, confirming the effectiveness and generalizability of hypergraph-guided retrieval and generation.
```

### --- Page 0010 ---

```markdown
# Acknowledgments

This work is supported by the National Natural Science Foundation of China (Grant No. 62473271, Grant No. 62176026, and Grant No. 62406036) and the Engineering Research Center of Information Networks, Ministry of Education, China.

# References

[1] Boyu Chen, Zirui Guo, Zidan Yang, Yulou Chen, Junze Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang. Pathrag: Pruning graph-based retrieval augmented generation with relational paths, 2025.

[2] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rap approach to query-focused summarization, 2024.

[3] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. RAGAs: Automated evaluation of retrieval augmented generation. In Nikolaos Aletras and Orphée De Clercq, editors, Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 150–158, St. Julians, Malta, March 2024. Association for Computational Linguistics.

[4] Mikhail Galkin, Priyansh Trivedi, Gaurav Maheshwari, Ricardo Usbeck, and Jens Lehmann. Message passing for hyper-relational knowledge graphs. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7346–7359, Online, November 2020. Association for Computational Linguistics.

[5] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey, 2024.

[6] Zirui Guo, Lianghao Xie, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation, 2024.

[7] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: Non-parametric continual learning for large language models, 2025.

[8] Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang, and Zhicheng Dou. Flashrag: A modular toolkit for efficient retrieval-augmented generation research. CoRR, abs/2405.13576, 2024.

[9] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 9459–9474. Curran Associates, Inc., 2020.

[10] Lei Liang, Mengshu Sun, Zhenke Gu, Zhongshu Zhu, Zhouyu Jiang, Ling Zhong, Yuan Qu, Peilong Zhao, Zhongpu Bo, Jin Yang, Huadong Xiong, Lin Yuan, Jun Xu, Zao Yang, Zhiqiang Zhang, Wen Zhang, Huajun Chen, Wenguang Chen, and Jun Zhou. Kag: Boosting llms in professional domains via knowledge augmented generation, 2024.

[11] Yu Liu, Quanning Yao, and Yong Li. Generalizing tensor decomposition for n-ary relational knowledge bases. In Proceedings of The Web Conference 2020, WWW ’20, page 1104–1114, New York, NY, USA, 2020. Association for Computing Machinery.

[12] Yaojie Lu, Hongyu Lin, Jin Xu, Xianpei Han, Jialong Tang, Annan Li, Le Sun, Meng Liao, and Shaoyi Chen. Text2Event: Controllable sequence-to-structure generation for end-to-end event extraction. In Cheng Zong, Fei Xia, Weijie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2795–2806, Online, August 2021. Association for Computational Linguistics.
```

### --- Page 0011 ---

```markdown
| Reference                                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------|
| [14] Haoran Luo, Haihong E, Yuhao Yang, Yikai Guo, Mingzhi Sun, Tianyu Yao, Zichen Tang, Kaiyang Wan, Meina Song, and Wei Lin. IHAE: Hierarchical attention for hyper-relational knowledge graphs in global and local level. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8095–8107, Toronto, Canada, July 2023. Association for Computational Linguistics. |
| [15] Haoran Luo, Haihong E, Yuhao Yang, Tianyu Yao, Yikai Guo, Zichen Tang, Wentai Zhang, Kaiyang Wan, Shiyao Peng, Meina Song, Wei Lin, Yifan Zhu, and Luu Anh Tuan. Text2nkg: Fine-grained n-ary relation extraction for n-ary relational knowledge graph construction, 2024. |
| [16] John William McEvoy, Cian P McCarthy, Rosa Maria Bruno, Sofie Borrows, Michelle D Canavan, Claudio Ceconi, Ruxandra Maria Christodorescu, Stella S Daskalopoulou, Charles J Ferro, Eva Gerdts, et al. 2024 esc guidelines for the management of elevated blood pressure and hypertension. Giornale italiano di cardiologia (2006), 25(11):1e–107e, 2024. |
| [17] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Julia Altenschmidt, Sam Altman, et al. Gpt-4 technical report, 2024. |
| [18] Shirui Pan, Linhao Luo, Yufei Wang, Chen He, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering, 36(7):3580–3599, 2024. |
| [19] Hongjin Qian, Peitian Zhang, Zheng Liu, KeLong Mao, and Zhicheng Dou. Memorandum: Moving towards next-gen ra via memory-inspired knowledge discovery, 2024. |
| [20] Haoran Que, Feiyu Duan, Liqun He, Yutao Mou, Wangchun Zhou, Jiaheng Liu, Wenge Rong, Zekun Moore Wang, Jian Yang, Ge Zhang, Junran Peng, Zhaoxiang Zhang, Songyang Zhang, and Kai Chen. Hellobench: Evaluating next generation capabilities of large language models, 2024. |
| [21] Paolo Rosso, Ding Liu, and Philippe-Claude Maurex. Beyond triplets: Hyper-relational knowledge graph embedding for link prediction. In Proceedings of The Web Conference 2020, WWW ’20, page 1885–1896, New York, NY, USA, 2020. Association for Computing Machinery. |
| [22] Kartik Sharma, Peeyush Kumar, and Yunyin Li. Og-rag: Ontology-grounded retrieval-augmented generation for large language models, 2024. |
| [23] Jinyu Wang, Jingjing Fu, Rui Wang, Lei Song, and Jiang Bian. Pike-rag: specialized knowledge and rationale augmented generation, 2025. |
| [24] Quan Wang, Haifeng Wang, Yajuan Lyu, and Yong Zhu. Link prediction on n-ary relational facts: A graph-based approach. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 396–407, Online, August 2021. Association for Computational Linguistics. |
| [25] Jianfeng Wen, Jianxin Li, Yongyi Mao, Shini Chen, and Richong Zhang. On the representation and embedding of knowledge bases beyond binary relations. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI’16, page 1300–1307. AAAI Press, 2016. |
| [26] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, and Vicente Grau. Medical graph rag: Towards safe medical large language model via graph retrieval-augmented generation, 2024. |
| [27] Richong Zhang, Junpeng Li, Jiajie Mei, and Yongyi Mao. Scalable instance reconstruction in knowledge bases via relatedness artificial embedding. In Proceedings of the 2018 World Wide Web Conference, WWW ’18, page 1185–1194, Republic and Canton of Geneva, CH, 2018. International World Wide Web Conferences Steering Committee. |
| [28] Wayne Xin Zhao, Kun Zhou, Junyi Li, et al. A survey of large language models, 2024. |
| [29] Dengyong Zhou, Jiayuan Huang, and Bernhard Schölkopf. Learning with hypergraphs: Clustering, classification, and embedding. In B. Schölkopf, J. Platt, and T. Hoffman, editors, Advances in Neural Information Processing Systems, volume 19. MIT Press, 2006. |
```

### --- Page 0012 ---

```markdown
# Appendix

## A Prompts Used in HyperGraphRAG

### A.1 N-ary Relation Extraction Prompt

As shown in Figure 8, this prompt is designed for extracting structured n-ary relational facts from raw text. It guides LLM to segment the input into coherent knowledge fragments, assign a completeness score to each, and identify entities with their names, types, descriptions, and importance scores.

- **Goal:** 
  Given a text document that is potentially relevant to this activity and a list of entry types, identify all entities of those types from the text and relationships among the entities.
  Use `{language}` as output language.

- **Steps:**
  1. Divide the text into several complete knowledge segments. For each knowledge segment, extract the following information:
     - **completeness_score:** A score from 0 to 10 indicating the completeness of the knowledge segment.
     - **Format each knowledge segment as `{hypertext}({input_delimiter}knowledge_segment{output_delimiter}completeness_score)`.

  2. Identify each entity in each knowledge segment. For each identified entity, extract the following information:
     - **entry:** Name of the entity, use same language as input text. If English, capitalized the name.
     - **entry_type:** Type of the entity.
     - **entry_description:** Comprehensive description of the entity's attributes and activities.
     - **importance_score:** A score from 0 to 10 indicating the importance of the entity in the text.
  
  3. Return output in `{language}` as a single list of all the entities and relationships identified in steps 1 and 2. Use `"{record_delimiter}"` as the list delimiter.

**Output:**

![Prompt for n-ary relation extraction $p_{ext}$ in Equation 4.](assets/page_0012_img_1.png)

### A.2 Entity Extraction Prompt

As shown in Figure 9, this prompt is used to extract key entities from a user query. LLM is instructed to return all identified entities in JSON format, ensuring the output is concise, human-readable, and aligned with the language of the input query. This facilitates entity-level retrieval in the hypergraph.

- **Role:** 
  You are a helpful assistant tasked with identifying entities in the user's query.

- **Goal:** 
  Given the query, list all entities.

- **Instructions:**
  - Output the keywords in JSON format.

**Examples:**
```
######################
(examples)
######################
```

**Real Data:**
```
######################
Query: 
The Output should be human text, not unicode characters. Keep the same language as `Query`.
```

**Output:**

![Prompt for entity extraction $p_{a\_ext}$ in Equation 7.](assets/page_0012_img_2.png)

### A.3 Retrieval-Augmented Generation Prompt

To ensure a fair comparison across RAG baselines, we adopt a unified Chain-of-Thought (CoT)-based generation prompt $p_{gen}$ in Equation 12 for all methods. We present this prompt together with the designed evaluation approach in Appendix E.
```

### --- Page 0013 ---

```markdown
# B  Proof

## B.1 Proof of Proposition 1

**Proposition 1.** Hypergraph-structured knowledge representation is more comprehensive than binary.

**Proof.** Given a universe of entities $V$, an n-ary fact with $n \geq 3$ is denoted as $F = \{v_1, \ldots, v_n\} \subset V$. For hypergraph representation, we represent it with a single hyperedge:

$$
e_H = F, \quad G_H = (V, E_H), \quad e_H \in E_H, \tag{13}
$$

so the representation function $\phi_H : F \mapsto e_H$ is naturally injective. For binary graph representation, we connect every pair of entities that co-occur in a fact. For any collection of facts $S \subset P(V)$, define the representation function:

$$
\phi_B(S) = (V_B, E_B) = \bigcup_{F \in S} F, \quad E_B = \{(u, w) | u \neq w, \exists S \in S : \{u, w\} \subset F\}, \tag{14}
$$

where $E_B$ consists of the binary edges activated by $S$ within the complete graph $K_{|V|}$. Each $E_B$ is a subset of some clique.

Let the random variable $X$ range over all possible fact sets $S$, with Shannon entropy:

$$
H(X) = -\sum_S p(S) \log_2 p(S), \tag{15}
$$

measuring the total information to be represented. For hypergraph representation, since $\phi_H$ is injective and each fact can be uniquely recovered,

$$
H(X | \phi_H(X)) = 0. \tag{16}
$$

For binary representation, consider any three distinct entities $a, b, c \in V$, and define

$$
S_1 = \{a, b\}, \quad S_2 = \{a, c\}, \quad S_3 = \{b, c\}, \tag{17}
$$

Clearly, $S_1 \neq S_2$, but

$$
\phi_B(S_1) = \phi_B(S_2) = \phi_B(S_3) = \{(a, b), (a, c), (b, c)\} = g, \tag{18}
$$

since both activate the same set of binary edges. Thus,

$$
\left| \phi_B^{-1}(\phi_B(S)) \right| \geq 2, \quad \Rightarrow \quad 0 < \frac{p(S)}{p(g)} < 1, \tag{19}
$$

then, we can get

$$
H(X | \phi_B(X)) = \sum_y p(y)H(X | \phi_B(X) = y) \geq p(g)H(X | \phi_B(X) = g) > 0, \tag{20}
$$

where information is inevitably lost in binary representation.

More generally, as long as there exists at least one n-ary fact (n ≥ 3) in the knowledge base, we can always construct a pair of distinct fact sets that activate the same binary edges through a merge-split transformation. Hence,

$$
H(X | \phi_B(X)) > 0, \quad I(X; \phi_B(X)) = H(X) - H(X | \phi_B(X)) < H(X), \tag{21}
$$

which proves that binary representation is lossy. In contrast, hypergraph representation satisfies $H(X | \phi_H(X)) = 0$, so the mutual information reaches its upper bound $H(X)$ and all information is preserved. In the special case where no n-ary facts with $n \geq 3$ exist, i.e., all facts are binary, then

$$
\left| \phi_B^{-1}(g_B) \right| = 1, \quad H(X | \phi_B(X)) = 0, \tag{22}
$$

so binary representation becomes ineffective and equivalent to hypergraph, with no information loss.

In conclusion, as long as the knowledge base contains at least one fact of three or higher, hypergraph-structured representation preserves more information with lossless representation, whereas binary representation inevitably loses information. Therefore, hypergraph representation is more comprehensive than binary in the information-theoretic sense.
```

### --- Page 0014 ---

```markdown
## B.2 Proof of Proposition 2

**Proposition 2.** A bipartite graph can losslessly preserve and query a knowledge hypergraph.

**Proof.** Let the knowledge hypergraph be denoted as $G_H = (V, E_H)$, $E_H \subset V \times |E_H| \geq 2$. Each hyperedge is abstracted as a new node, and combined with the set of entity nodes to form a vertex set $V_B = V \cup E_H$, with edges defined as $E_B = \{ (e_H, v) | e_H \in E_H, v \in e_H \}$, resulting in the incidence bipartite graph $\Phi(G_H) = G_B = (V_B, E_B)$.

Ordering the vertices such that entities come first and hyperedges second, $G_H$ can be represented by the binary incidence matrix

$$
M \in \{0, 1\}^{|V| \times |E|}, \quad M_{e_H, v} = 1 \iff v \in e_H, \tag{23}
$$

and the adjacency matrix of $G_B$ becomes

$$
A_{G_B} = \begin{pmatrix}
0 & M \\
M^T & 0
\end{pmatrix} \tag{24}
$$

where $M$ uniquely determines $A_{G_B}$, and conversely, $M$ can be recovered from the top-right block of $A_{G_B}$. Therefore, there exists an inverse mapping:

$$
\Phi^{-1}: G \to G_H, \quad \Phi^{-1}(V_B, E_B) = (V, \{ N_G(e_H) | e_H \in E_H \}) \tag{25}
$$

where 

$$
N_G(e_H) = \{ v \in V | (e_H, v) \in E_B \}. \tag{26}
$$

Clearly,

$$
\Phi^{-1} \circ \Phi = id_{G_H}, \quad \Phi \circ \Phi^{-1} = id_{G_B}, \tag{27}
$$

which means that $\Phi$ is a bijection and the encoding is lossless.

The query equivalence can also be derived directly via matrix operations and path counting: the set of hyperedges containing a vertex corresponds to the support of the $v$-th row of $M$, and the bipartite graph this is equivalent to the neighborhood $N_G(v)$, given by the right block of $e_H^T A_{G_B} = (0, e_H)$. Likewise, the entity set of a hyperedge $e_H$ is the support of the $e_H$-th column of $M$, which matches the left block of $e_H A_{G_B}$. To determine whether two entities $u, v$ co-occur in some hyperedge, it suffices to check whether

$$
(M M^T)_{uw} = (A_{G_B})_{uw} \neq 0, \tag{28}
$$

since $(A_{G_B})_{uw}$ counts all 2-step paths from $u$ through a hyperedge node to $v$. For a given subset of entities $S \subset V$, hyperedges that contain all of them can be found by summing the corresponding rows $\sum_{v \in S} M_{*, v}$, and selecting columns where the sum equals $|S|$; in the bipartite graph, this corresponds to the intersection

$$
\bigcap_{v \in S} N_G(v). \tag{29}
$$

All operations run in time $O(|E_B|)$, which matches the complexity of equivalent queries over $G_H$.

In conclusion, the bijection $\Phi$ guarantees full structural reversibility, while adjacency and path-based reasoning preserve the semantics of all queries involving entity-hyperedge membership. Therefore, a bipartite graph can losslessly preserve and query a knowledge hypergraph. 

## B.3 Proof of Proposition 3

**Proposition 3.** Retrieving knowledge on a knowledge hypergraph improves retrieval efficiency compared to methods based on ordinary binary graphs, leading to gains in generation quality.

**Proof.** Let the ground-truth knowledge set required for a query $q$ be modeled as a discrete random variable $X \in \mathcal{P}(V)$, with probability measure $\mu$ defined over the measurable space $(\mathcal{P}(V), \mathcal{B})$ for any $n$-ary fact $F = \{v_1, \ldots, v_n\}$ with $n \geq 3$, we define two encoders:

$$
\phi_H: F \mapsto e_H = F, \quad \phi_B: F \mapsto \{(v_i, v_j) | 1 \leq i < j \leq n\}. \tag{30}
$$

Let the encoded knowledge sets be random variables $Y_H = \phi_H(X)$ and $Y_B = \phi_B(X)$. Since $\phi_H$ is injective, the conditional entropy is zero:

$$
H(X | Y_H) = 0, \tag{31}
$$

and hence $I(X; Y_H) = H(X)$.
```

### --- Page 0015 ---

```markdown
However, when $\mu(|F| > 3) > 0$, the encoder $\varphi_B$ becomes non-injective. There exist $x_1 \neq x_2$ such that $Y_B(x_1) = Y_B(x_2)$, leading to:

$$
H(X | Y_B) = E_{Y} \left[ - \sum_{x \in \varphi^{-1}(Y_B)} \mu(x | Y_B) \log_2 \mu(x | Y_B) \right] > 0, \tag{32}
$$

$$
I(X;Y_B) = H(X) - H(X | Y_B) < H(X). \tag{33}
$$

To study encoding efficiency, consider encoding $Y^* (\ast \in \{H, B\})$ using an optimal prefix code. Let the expected code length be $L_* = E[\ell(Y^*)]$. According to Shannon’s source coding theorem:

$$
L_* \in [H(Y), H(Y) + 1]. \tag{34}
$$

Define the information efficiency density (information per bit) as:

$$
\eta_* = \frac{I(X;Y_*)}{L_*}. \tag{35}
$$

This metric quantifies the amount of effective information transmitted per bit. Since $I(X;Y_B) < H(X) \text{ and } H(Y_B) \ge H(Y_H)$ (the pairwise representation introduces a larger outcome space), we have:

$$
\eta_H = \eta_B = \frac{H(X)}{L_H} - \frac{H(X) - \delta}{L_B}, \quad \delta > 0, \quad L_B - L_H \ge 0, \tag{36}
$$

which is strictly positive when $\delta > 0$. This shows that the hypergraph representation transmits more effective information per bit. Let the maximum retrievable context budget for a language model be $L$, and define the coverage function:

$$
C(L) = P(T(X;Y_* \le L) = \mu(\{x | \eta_* \cdot \mathbb{E}(Y(x)) \le L\}), \tag{37}
$$

the chain rule yields:

$$
\frac{d}{dL} C_H(L) = \frac{d}{d\theta} \left( \frac{\partial \ell}{\partial \eta} \cdot \frac{d \ell}{d \sigma} \right) \ge \frac{d}{dL} C_B(L), \tag{38}
$$

which implies $C_H(L) \ge C_B(L)$ with strict inequality on intervals where $\mu(|F| \ge 3) > 0$. Let generation quality $E$ (e.g., G-Es score) be a differentiable function $E = g(I(X;Y), \mathcal{N})$, where $\mathcal{N}$ denotes the noise introduced by irrelevant or redundant edges, and satisfies:

$$
\frac{\partial g}{\partial Y} > 0, \quad \frac{\partial g}{\partial N} < 0. \tag{40}
$$

Here, noise $\mathcal{N}$ is defined as the set of edges retrieved under budget $L$ that are irrelevant to the ground-truth $X^*$. Under the same bit budget, higher $\eta_H$ implies fewer edges per bit, thus:

$$
E[|\mathcal{N}|] \le E[\mathcal{B}]. \tag{41}
$$

Treating $L$ as an independent variable, we apply the chain rule:

$$
\frac{d}{dL} [E_H(L) - E_B(L)] = \frac{\partial}{\partial \theta_l} \left( \frac{d}{dL} I(X;Y_H) - \frac{d}{dL} I(X;Y_B) \right) + \frac{\partial g}{\partial N} \left( \frac{d}{dL} \mathcal{N} - \frac{d}{dL} \mathcal{B} \right), \tag{41}
$$

where $\theta_l$ is an intermediate state between the two systems. From Equation 38 and Equation 40, we know: (1) The first term is strictly positive if high-arity facts exist; (2) The second term is always non-positive, as higher information density leads to lower redundancy. Therefore, the total derivative is strictly positive. Integrating over $[0, L]$, we obtain:

$$
E_H(L) - E_B(L) = \int_0^L dE(\beta) - E_B(\beta) d\beta > 0, \quad \text{unless } \mu(|F| \ge 3) = 0. \tag{42}
$$

Equation 42 formally proves that if there exists at least one fact with arity $\ge 3$ in the knowledge base, then under any fixed retrieval budget $L$, the generation quality under hypergraph encoding strictly exceeds that of the binary encoding. In the degenerate case where all facts are binary, both encodings reduce to the same mapping, and the conclusion naturally becomes an equality.
```

### --- Page 0016 ---

```markdown
# C  HyperGraphRAG Algorithm Detail

## Hypergraph Construction

To provide a clear overview of our system pipeline, we present the detailed procedures of HyperGraphRAG in the form of pseudocode. As shown in Algorithm 1, we first construct a knowledge hypergraph from raw documents via LLM-based extraction of n-ary relational facts. Each extracted fact forms a hyperedge connecting multiple entities, and the resulting hypergraph is stored in a bipartite structure for efficient indexing and retrieval. We further compute dense embeddings for all entities and hyperedges to support semantic retrieval.

### Algorithm 1: Hypergraph Construction
```
Require: Document collection $D$  
Ensure: Knowledge hypergraph $G_H = (V, E_H)$  
1: Initialize entity set $V \gets \emptyset$, hyperedge set $E_H \gets \emptyset$  
2: for each document $d \in D$ do  
3: Extract n-ary facts: $F_d = \{(e_i, v_e) \}_{i=1}^{n(d)}$  
4: $V \gets V \cup \bigcup_{i=1}^{n(d)} e_i$  
5: $E_H \gets E_H \cup \{e_i\}_{i=1}^{n(d)}$  
6: end for  
7: Store $(V, E_H)$ as bipartite graph $G_B = \Phi(G_H)$  
8: Compute embeddings: $E_V = \{f(v) | v \in V\}$, $E_{H} = \{f(e) | e \in E_H\}$  
9: return $G_H = (V, E_H)$  
```

## Complexity Analysis

Given a corpus of $D$ documents, assume each document contains at most $r$ relational facts, and each fact involves up to $m$ entities. The LLM-based extraction step has complexity $O(D)$ under the assumption of constant-time per document prompt. Constructing the hypergraph involves inserting up to $O(D \cdot r)$ hyperedges and $O(D \cdot r \cdot n)$ entities (with $n$ being the total construction time of $O(D \cdot r \cdot n)$). Embedding all nodes and hyperedges requires $O(|V| + |E_H|)$ and can be locally parallelizable.

## Hypergraph Retrieval and Generation

Once the hypergraph is constructed, the generation process begins with a query input, as detailed in Algorithm 2. We first extract relevant entities from the query and perform top-k similarity search to retrieve both entity and hyperedge candidates. We then perform bidirectional neighborhood expansion over the hypergraph to assemble a knowledge set, which may optionally be combined with chunk-level retrieval. Finally, we format the retrieved knowledge into a prompt and generate an answer using a large language model. This modular pipeline ensures efficient, expressive, and accurate generation grounded in structured knowledge.

### Algorithm 2: Hypergraph Retrieval and Generation
```
Require: Query $q$, knowledge hypergraph $G_H = (V, E_H)$  
Ensure: Final answer $y^*$  
1: Extract query entities: $V_q \sim n(q)$  
2: Retrieve top-k entities: $V_r \gets \text{TopKSim}(V_q, E_V)$  
3: Retrieve top-k hyperedges: $E_r \gets \text{TopKSim}(V_r, E_H)$  
4: Expand neighbors: $F_V = U_{v \in V_r} \text{Nbr}(v)$, $F_E = U_{e \in E_r} \text{Nbr}(e)$  
5: Assemble retrieved knowledge: $K_H = F_V \cup F_E$  
6: Retrieve additional chunks (optional): $K_{chunk} = \text{RETRIEVECHUNKS}(q)$  
7: Combine all knowledge: $K^* = K_H \cup K_{chunk}$  
8: Generate answer: $y^* \sim \mathcal{Q}(K^*)$  
9: return $y^*  
```

## Complexity Analysis

Given a query $q$, entity and hyperedge retrieval involves computing top-k similarity against all entity and hyperedge embeddings. With $|V|$ entities and $|E_H|$ hyperedges, this results in $O(|V| + |E_H|)$ embedding comparisons. The neighborhood expansion step is bounded by the degree of retrieved nodes, i.e., $O(k \cdot d)$ where $d$ is average node degree. Finally, generation is treated as a black-box LLM inference, typically $O(L)$ where $L$ is the prompt length.

In summary, HyperGraphRAG achieves efficient inference with precomputed indices, and its overall retrieval-generation time is dominated by vector similarity lookup and prompt generation, both of which scale linearly with hypergraph size and are highly parallelizable in practice.
```

### --- Page 0017 ---

```markdown
# D Dataset Construction

## D.1 Knowledge Domains

The dataset used for HyperGraphRAG evaluation covers five domains, with data sourced as follows:

- **Medicine**: Derived from the latest international hypertension guidelines [16], covering medical diagnosis, treatment plans, and clinical indicators.
- **Agriculture**: Extracted from the UltraDomain dataset [19], including knowledge on agricultural production, crop management, and pest control.
- **Computer Science (CS)**: Sourced from the UltraDomain dataset, encompassing computer architecture, algorithms, and machine learning.
- **Legal**: Based on the UltraDomain dataset, covering legal provisions, judicial precedents, and regulatory interpretations.
- **Mix**: A combination of multiple domains to assess the model’s generalization ability across interdisciplinary tasks.

## D.2 Question Sampling Strategies

To construct a fair and comprehensive evaluation benchmark, we design a uniform sampling strategy for both binary and n-ary sources. Specifically, for each domain, we sample a total of 512 questions, consisting of:

- **Binary Source (256 samples)**: 128 facts are selected via 1-hop traversal, 64 facts via 2-hop traversal, 64 facts via 3-hop traversal. These facts are composed of binary relations (i.e., pairwise connections) and are used to build the binary knowledge source.

- **N-ary Source (256 samples)**: 128 facts are sampled via 1-hop traversal, 64 facts via 2-hop traversal, 64 facts via 3-hop traversal. These facts involve multi-entity ($n \geq 3$) relational structures and are used to construct the n-ary knowledge source.

For each sampled fact, we prompt GPT to generate a corresponding question and its golden answer. All generated question-answer pairs are manually verified to ensure accuracy, relevance, and diversity. This process is repeated independently for every domain to ensure consistent scale and structure across evaluation sets. All datasets undergo manual review to ensure the accuracy of annotated answers and the fairness of model evaluation.

# E Evaluation Details

## Unified Generation Prompt

To ensure a fair comparison across all baselines, we adopt a unified generation prompt for all methods, as shown in Figure 10. Specifically, we insert the knowledge retrieved by each method into a fixed prompt template that guides the model to first perform reasoning within a `<think>` block and then provide the final answer within an `<answer>` block, preserving benefits of zero-shot CoT reasoning while maintaining consistency across different retrieval strategies.

```
![The unified prompt for generation $p_{gen}$ in Equation 12.](assets/page_0017_img_1.png)
```

### --- Page 0018 ---

```markdown
We evaluate model performance using three complementary metrics that assess different aspects of retrieval-augmented generation: factual alignment, retrieval quality, and generation fluency.

(i) F1 Score. Following FlashRAG [9], we compute the word-level F1 score between each generated answer and its ground-truth reference, and then average over all questions. This metric captures reflects factual alignment with the expected answer.

$$
F1 = \frac{1}{N} \sum_{i=1}^{N} \frac{2 \cdot P_i \cdot R_i}{P_i + R_i}, \quad P_i = \frac{|Pred_i \cap GT_i|}{|Pred_i|}, \quad R_i = \frac{|Pred_i \cap GT_i|}{|GT_i|} \quad (43)
$$

where $Pred_i$ and $GT_i$ denote the set of words in the predicted and ground-truth answers for the $i$-th question, and $N$ is the total number of evaluated questions.

(ii) Retrieval Similarity (R-S). Inspired by RAGAS [3], R-S quantifies the semantic similarity between the retrieved knowledge and the ground-truth knowledge used to construct the question. For each question, we concatenate all retrieved knowledge into a single string $k_{retr}$ and all golden knowledge into $k_{gold}$, then compute the cosine similarity between their embeddings. The final R-S score is the average similarity across the dataset:

$$
R-S = \frac{1}{N} \sum_{i=1}^{N} \cos(f(k_{retr}), f(k_{gold})) \quad (44)
$$

where $f(\cdot)$ is the embedding function (e.g., SimCSE), and $N$ is the total number of questions.

(iii) Generation Evaluation (G-E). Adapted from HelloBench [20], G-E uses GPT-40-mini as an LLM judge to evaluate generation quality along seven dimensions: Correctness, Relevance, Factuality, Comprehensiveness, Knowledgability, Logical Coherence, and Diversity. For each question, we compute the average of the seven dimension scores, then combine it with the question’s F1 score by taking their mean. The final G-E score is obtained by averaging this combined score:

$$
G-E = \frac{1}{N} \sum_{i=1}^{N} \text{mean} \left( \frac{1}{7} \sum_{d=1}^{7} s_{i,d} ; F_{i} \right) \quad (45)
$$

where $s_{i,d}$ denotes the score for dimension $d$ on question $i$, $F_{i}$ is the word-level F1 score for the $i$-th question, and $N$ is the total number of evaluated questions. This formulation encourages alignment between LLM-judged quality and factual correctness.

G-E Prompt. Figure 11 and Figure 12 show our generation evaluation prompts. Figure 11 presents the unified prompt used to score each dimension on a 0–10 scale, while Figure 12 provides the detailed scoring rubric for all seven dimensions, ensuring consistency and fairness across evaluations.

---

**Role**:  
You are a helpful assistant evaluating the “(***X***)” of a generated response.

**Question**:  
—Question—  
—Golden Answer—  
{(answers)}

—Evaluation Goal—  
Evaluate “(***X***)” using a “0–10 integer scale”.

**Input**:  
```
your_score: <num> (an integer from 0 to 10)  
<your_score_explanation>  
Explain why you gave this score.  
```

—Generation to be Evaluated—  
{(generation)}

![Prompt for G-E](assets/page_0018_img_1.png)

Figure 11: Prompt for G-E.
```

### --- Page 0019 ---

```markdown
{
  "comprehensiveness": {
    "comprehensiveness": 
      "Whether the thinking considers all important aspects and is thorough.",
    "Scoring Guide (0–10)": 
      "- 10: Extremely thorough, covering all relevant angles and considerations with depth.\n- 8–9: Covers most key aspects clearly and thoughtfully, only minor omissions.\n- 6–7: Covers some important aspects, but lacks depth or overlooks notable areas.\n- 4–5: Touches on a few relevant points, but overall lacks substance or completeness.\n- 1–3: Sparse or shallow treatment of the topic; misses key aspects.\n- 0: No comprehensiveness at all; completely superficial or irrelevant."
  },
  "knowledgeability": {
    "knowledgeability": 
      "Whether the thinking is rich in insightful, domain-relevant knowledge.",
    "Scoring Guide (0–10)": 
      "- 10: Demonstrates exceptional depth of insight with strong domain-specific knowledge.\n- 8–9: Shows detailed knowledge with good insights; mostly accurate and relevant.\n- 6–7: Displays some understanding, but lacks depth or has notable gaps.\n- 4–5: Limited knowledge shown; understanding is basic or somewhat flawed.\n- 1–3: Poor grasp of relevant knowledge; superficial or mostly incorrect.\n- 0: No evidence of meaningful knowledge."
  },
  "correctness": {
    "correctness": 
      "Whether the reasoning and answer are logically and factually correct.",
    "Scoring Guide (0–10)": 
      "- 10: Fully accurate and logically sound; no flaws in reasoning or facts.\n- 8–9: Mostly correct with minor inaccuracies or small logical gaps.\n- 6–7: Partially correct; some key flaws or inconsistencies present.\n- 4–5: Noticeable incorrect reasoning or factual errors throughout.\n- 1–3: Largely incorrect, misleading, or illogical.\n- 0: Entirely wrong or nonsensical."
  },
  "relevance": {
    "relevance": 
      "Whether the reasoning and answer are highly relevant and helpful to the question.",
    "Scoring Guide (0–10)": 
      "- 10: Fully focused on the question; highly relevant and helpful.\n- 8–9: Generally relevant, but includes distractions or less helpful parts.\n- 6–7: Mildly relevant; much of the response is off-topic or unhelpful.\n- 1–5: Barely related to the question or largely unhelpful.\n- 0: Entirely irrelevant."
  },
  "diversity": {
    "diversity": 
      "Diversity.",
    "Scoring Guide (0–10)": 
      "- 10: Exceptionally rich and original; demonstrates multiple fresh and thought-provoking ideas.\n- 8–9: Contains a few novel angles of interesting perspectives.\n- 6–7: Some variety, but generally safe or conventional.\n- 4–5: Mostly standard thinking; minimal diversity.\n- 1–3: Very predictable or monotonous.\n- 0: No diversity or originality at all."
  },
  "logical_coherence": {
    "logical coherence": 
      "Logical coherence.",
    "Scoring Guide (0–10)": 
      "- 10: Highly logical, clear, and easy to follow throughout.\n- 8–9: Well-structured with minor lapses in flow or clarity.\n- 6–7: Some structure and logic, but a few confusing or weakly connected parts.\n- 4–5: Often disorganized or unclear; logic is hard to follow.\n- 1–3: Poorly structured and incoherent.\n- 0: Entirely illogical or unreadable."
  },
  "factuality": {
    "factuality": 
      "Factuality.",
    "Scoring Guide (0–10)": 
      "- 10: All facts are accurate and verifiable.\n- 8–9: Mostly accurate; only minor factual issues.\n- 6–7: Contains some factual inaccuracies or unverified claims.\n- 4–5: Several significant factual errors.\n- 1–3: Mostly false or misleading.\n- 0: Completely fabricated or factually wrong throughout."
  }
}
```

![Figure 12: Seven Evaluation Dimensions for Generation Quality.](assets/page_0019_img_1.png)

### --- Page 0020 ---

```markdown
# F  Baseline Details

We compare HyperGraphRAG against six representative baselines that cover retrieval-free, chunk-based, and binary graph-based RAG paradigms:

- **NaiveGeneration** is a retrieval-free baseline where the LLM directly answers questions without any external knowledge input. This serves as a lower bound for retrieval-augmented generation.

- **StandardRAG** follows the original RAG design, retrieving top-$k$ text chunks from a flat corpus using dense vector similarity and feeding them into the generator.

- **GraphRAG** [2] constructs a binary relational graph and retrieves community-level summaries linked to query-relevant entities. It uses entity overlap to detect relevant subgraphs.

- **LightRAG** [7] enhances retrieval efficiency by using graph indexing and lightweight entity-relation matching over the binary graph, and then combines results with chunk-level retrieval.

- **PathRAG** [11] improves graph-based retrieval by selecting paths through the graph that are semantically relevant to the query, using path pruning strategies to reduce redundancy.

- **HippoRAG** [8] introduces a high-precision multi-hop retrieval mechanism over binary graphs, using Personalized PageRank to select passage-level nodes connected to the query.

To ensure fairness, all baselines use the same generation prompt (Figure 10) and are evaluated under identical conditions, with retrieved knowledge constrained to equivalent token budgets. Each method’s construction and retrieval mechanism is summarized in Table 1.

# G  Hyperparameter Settings

For all methods, we adopt a unified set of hyperparameters for all models across both the main evaluation in Table 2 and the time/cost experiments in Table 3 to ensure fair and consistent comparison. For chunk-based methods (e.g., StandardRAG), we retrieve the top-$5$ chunks using dense similarity. For graph-based methods, including GraphRAG, LightRAG, PathRAG, and HippoRAG2, we retrieve the top-$60$ relevant elements according to their respective retrieval strategies. HyperGraphRAG performs dual top-$60$ retrieval over entities and hyperedges, followed by neighborhood expansion. All methods are run using $16$ parallel cores and the same generation model (GPT-4-mini) with temperature $1.0$ and a maximum generation length of $32k$ tokens. Table 4 summarizes the detailed hyperparameter configurations used throughout our experiments.

| Method          | Retrieval Type | Top-$k$ Units | Parallel Cores | Generation Model |
|------------------|----------------|----------------|----------------|------------------|
| NaiveGeneration   | None           | 16             | 16             | GPT-4-mini       |
| StandardRAG      | Chunk          | 5 chunks       | 16             | GPT-4-mini       |
| GraphRAG         | Entity $\rightarrow$ Community | 60 | 16 | GPT-4-mini       |
| LightRAG         | Entity $\rightarrow$ Relation  | 60 | 16 | GPT-4-mini       |
| PathRAG          | Graph Path     | 60             | 16             | GPT-4-mini       |
| HippoRAG         | PageRank over Graph | 60       | 16             | GPT-4-mini       |
| HyperGraphRAG (ours) | Entity $\&$ Hyperedge | 60 | 16 | GPT-4-mini       |

# H  Case Study

To better understand how different methods perform in complex, knowledge-intensive scenarios, we present a case study on the question: “What type of renal denervation has been shown to demonstrate BP-lowering efficacy over 24 hours in several randomized, sham-controlled trials, particularly for resistant hypertension?” As shown in Table 5, this question requires synthesizing clinical evidence and distinguishing between treatment modalities (e.g., ultrasound vs. catheter-based techniques).

HyperGraphRAG provides the most comprehensive and accurate response, clearly identifying ultrasound renal denervation as the answer, and justifying it with domain-relevant reasoning and structured knowledge. It achieves the highest scores across all dimensions, including Correctness (10.00), Factuality (9.50), and Knowledgeability (10.00), resulting in an overall G-score of 92.86.
```

### --- Page 0021 ---

```markdown
| Query            | What type of real deconvolution has been done to demonstrate BP-lowering efficacy over 24 hours in several randomized, sham-controlled trials, particularly for resistant hypertension? |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Golden Answer** | **NaïveGeneration**                                                                                                                                              |
|                  | <div>Virtual deconvolution is produced to show resistant hypertension by disrupting the nerves near the arteries, which is believed to stabilize the regulation of blood pressure. Several techniques can be utilized for real deconvolution, including radiofrequency, ultrasound, and chemical methods. In the context of this, the method that has been reliably tested and demonstrated BP-lowering efficacy over a 24-hour period is highly applicable and clinically relevant. This approach has proven useful in various interventions. The radiofrequency method works among the sites as a technique to demonstrate BP-lowering efficacy over 24 hours in several randomized, sham-controlled trials, particularly for resistant hypertension.</div> |
| **Evaluation Score** | 16.0 | 16.0 | 48.0 | 48.0 | 5.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| **StandardRAG**  | <div>It mentions that several devices have been investigated for treating resistant hypertension, but specifically highlights certain methods that have been demonstrated to show BP-lowering efficacy. This includes radiofrequency and ultrasound methods that have been demonstrated to lower BP over 24 hours and a description of hypertension management. These methods have shown to be effective in treating resistant hypertension.</div> |
| **Evaluation Score** | 20.0 | 65.1 | 51.4 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| **GraphRAG**     | <div>This question refers to several methods that have been tested in randomized, sham-controlled trials demonstrating BP-lowering efficacy. Randomized clinical trials have shown the efficacy of these methods in producing desired clinical results, particularly in treating resistant hypertension.</div> |
| **Evaluation Score** | 16.0 | 57.3 | 48.0 | 48.0 | 48.0 | 48.0 | 48.0 | 48.0 | 48.0 | 48.0 |
| **LightRAG**     | <div>This question refers to a method specifically focusing on its therapeutic efficacy in blood pressure lowering. Real deconvolution in a randomized trial shows efficacy over 24 hours in resistant hypertension, and methodology has been established to demonstrate BP-lowering efficacy over 24 hours in several randomized, sham-controlled trials, particularly for resistant hypertension.</div> |
| **Evaluation Score** | 20.0 | 29.9 | 32.0 | 53.4 | 53.4 | 53.4 | 48.0 | 48.0 | 48.0 | 48.0 |
| **PathRAG**      | <div>This question refers to a method specifically focusing on its therapeutic efficacy in blood pressure lowering. Real deconvolution in a randomized trial shows efficacy over 24 hours in resistant hypertension, and methodology has been established to demonstrate BP-lowering efficacy over 24 hours in several randomized, sham-controlled trials, particularly for resistant hypertension.</div> |
| **Evaluation Score** | 16.0 | 64.4 | 45.7 | 48.0 | 51.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| **HippoRAG**     | <div>They provided text mentions that catheter-based renal denervation has been studied significantly in treating resistant hypertension, with a focus on the associated procedural risks and benefits. The methods have shown to be effective in producing desired clinical results, particularly in treating resistant hypertension.</div> |
| **Evaluation Score** | 19.5 | 66.5 | 52.5 | 54.8 | 54.8 | 54.8 | 66.8 | 66.8 | 54.8 | 54.8 |
| **HyperGraphRAG** | <div>Baseline methods like NaïveGeneration and StandardRAG produce factually plausible but imprecise or overly generic answers (e.g., mentioning radiofrequency denervation instead), leading to lower scores, particularly in Correctness and Factuality. Graph-based baselines such as GraphRAG, LightRAG, and PathRAG improve coherence but still fall short in domain specificity. Even the best-performing baseline, HippoRAG, fails to precisely isolate the correct answer, with reduced clarity and comprehensiveness compared to HyperGraphRAG. This case highlights the strength of HyperGraphRAG in integrating multi-entity clinical knowledge through hyperedges, enabling more precise, interpretable, and fact-grounded responses in real-world expert-level tasks.</div> |
| **Evaluation Score** | 100.0 | 70.7 | 90.0 | 90.0 | 100.0 | 75.0 | 100.0 | 75.0 | 90.0 | 90.0 |
```

### --- Page 0022 ---

# I  Limitations and Future Work

## I.1  Multimodal HyperGraphRAG

While our current framework focuses on textual knowledge, real-world information often spans multiple modalities, including images, tables, and structured metadata. A promising direction is to extend HyperGraphRAG to the multimodal setting by constructing hypergraphs that integrate both textual and non-textual entities (e.g., medical images, diagrams, or structured EHR fields). This would allow the model to reason over complex multimodal relationships, such as “image + report + diagnosis” or “chart + claim + textual guideline,” and enable broader deployment in domains like medicine, science, and law. Future work will explore how to encode, align, and retrieve multimodal hyperedges effectively, while maintaining the structural advantages of hypergraph representations.

## I.2  HyperGraphRAG with Reinforcement Learning

Another important extension lies in incorporating reinforcement learning (RL) to guide both retrieval and generation. In our current setup, retrieval is driven by fixed similarity metrics, which may not fully capture downstream utility. By formulating hypergraph-based retrieval as a sequential decision-making process, we can apply RL to optimize entity and hyperedge selection policies based on long-term generation rewards—such as factuality, coherence, or user feedback. This would allow HyperGraphRAG to dynamically adapt retrieval strategies to different tasks and domains, leading to more efficient and effective use of structured knowledge.

## I.3  Federated HyperGraphRAG for Privacy-Preserving Retrieval

Many real-world applications involve sensitive or distributed data that cannot be centralized due to privacy constraints. To address this, we propose to integrate HyperGraphRAG with federated learning techniques, allowing hypergraph construction, retrieval, and generation to occur across decentralized data silos. Each local client can construct its own partial hypergraph and share anonymized or encrypted embeddings, preserving privacy while contributing to global retrieval. This federated HyperGraphRAG would be particularly beneficial in domains like healthcare or finance, where data sharing is restricted but collective knowledge is crucial for robust decision-making.

## I.4  Toward a Foundation Model for HyperGraph-based Retrieval

As large language models continue to scale and generalize across domains, a natural extension is to explore the development of a foundation model for HyperGraphRAG. Rather than constructing and retrieving from hypergraphs on a per-task or per-domain basis, we envision a pretrained hypergraph reasoning model that jointly learns representations of entities, relations, and higher-order hyperedges across diverse corpora. This model would encode structural, semantic, and contextual signals in a unified way, and could be adapted to new domains via lightweight fine-tuning. Such a foundation model could also enable transfer learning across knowledge-intensive tasks, reducing the need for domain-specific engineering and improving the sample efficiency of retrieval and generation pipelines. Building this requires scalable hypergraph pretraining objectives, efficient storage formats, and robust generalization strategies, which we leave as future work.

## I.5  Scaling to Harder Tasks and Broader Applications

Finally, we plan to evaluate HyperGraphRAG on more challenging tasks and diverse real-world applications. This includes settings that require deeper compositional reasoning, such as multi-hop question answering, legal argument generation, or complex scientific synthesis. Additionally, we aim to apply HyperGraphRAG to broader domains beyond the current benchmarks, including policy analysis, education, and open-domain dialogue. These tasks will test the framework’s ability to generalize across domains, handle larger and more diverse knowledge bases, and maintain high-quality generation under increasingly demanding conditions.


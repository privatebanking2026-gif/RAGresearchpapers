# ArXiv 2504.11544

### --- Page 0001 ---

```markdown
# NodeRAG:  
## Structuring Graph-based RAG with Heterogeneous Nodes

**Tianyang Xu**¹, **Haojie Zheng**², **Chengze Li**¹, **Haoxiang Chen**¹  
**Yixin Liu**³, **Ruoxi Chen**⁴, **Lichao Sun**⁵  
¹ Columbia University, ² University of Pennsylvania, ³ Lehigh University  
tx2240@columbia.edu, haojiez@seas.upenn.edu  

---

### Abstract

Retrieval-augmented generation (RAG) empowers large language models to access external and private corpus, enabling factually consistent responses in specific domains. By exploiting the inherent structure of the corpus, graph-based RAG methods further enrich this process by building a knowledge graph index and leveraging the structural nature of graphs. However, current graph-based RAG approaches seldom prioritize the design of graph structures. Inadequately designing graph not only impedes the seamless integration of diverse graph algorithms but also result in workflow inconsistencies and degraded performance. To further unleash the potential of graph for RAG, we propose NodeRAG, a graph-centric framework introducing heterogeneous graph structures that enable the seamless and holistic integration of graph-based methodologies into the RAG workflow. By aligning closely with the capabilities of LLMs, this framework ensures a fully cohesive and efficient end-to-end process. Through extensive experiments, we demonstrate that NodeRAG exhibits performance advantages over previous methods, including GraphRAG and LightRAG, not only in indexing time, query time, and storage efficiency but also in delivering superior question-answering performance on multi-hop benchmarks and open-ended head-to-head evaluations with minimal retrieval tokens. Our GitHub repository can be seen at this link.

---

### 1 Introduction

Retrieval-augmented generation (RAG) has emerged as a solution to the challenges posed by the rapid evolution of real-world knowledge domains (Fan et al., 2024), coupling large language models (LLMs) with an external retrieval mechanism to ensure the generation of factually consistent and contextually relevant information (Tomoy et al., 2024; Shrestha et al., 2024; Liu et al., 2024). Despite recent progress, current RAG methods face notable shortcomings in handling multi-hop reasoning (Luo et al., 2023; Wang et al., 2024b) and summary-level queries (Han et al., 2024a; Wen et al., 2023) due to their insufficient utilization of data structures and lack of high-level understanding of the text corpus. Graph-based RAG methods (Tian et al., 2024; Park et al., 2023) have been proposed to enhance retrieval and question-answering performance, specifically addressing the two main challenges faced by traditional RAG approaches. Leveraging LLMs to decompose raw data into graph structures (Jiménez Gutiérrez et al., 2024; He et al., 2024) for utilizing structural information, as well as employing LLMs for summary-based enhancements (Edge et al., 2024; Guo et al., 2024) to derive insights beyond the original text, has gradually become mainstream approaches.

However, previous Graph-based RAG works (Trajano et al., 2023; Jiménez Gutiérrez et al., 2024) have rarely considered the critical role of graph structures, i.e., what forms of graph better support RAG. Among existing approaches, knowledge graphs (Samartin, 2024; Wang et al., 2024b) extract triples, with the graph containing only structural information, yet retrieval context remains confined to text chunks, which often lack semantic coherence and include unrelated information. While current methods attempt to incorporate more information into the graph and extract deeper insights, they suffer from inefficiencies and inconsistencies due to inadequately designed structures. For instance, as illustrated in Figure 1, GraphRAG (Edge et al., 2024) adopts a tightly coupled entity-event homogeneous structure, hindering the integration of original context and summary information in the graph. This results in inconsistencies in retrieval methods (separating local and global retrieval) and leads to coarse-grained retrieval, where retrieving an entity indiscriminately includes all.
```
![Detailed description of the chart](assets/page_0001_img_1.png)

### --- Page 0002 ---

```markdown
![Comparisons between NodeRAG and other RAG systems. NaïveRAG retrieving fragmented text chunks, leads to redundant information. HippoRAG introduces knowledge graphs but lacks high-level summarization. GraphRAG retrieves community summaries but may still produce coarse-grained information. LightRAG incorporates one-hop neighbors but retrieves redundant nodes. In contrast, NodeRAG utilizes multiple node types, including high-level elements, semantic units, and relationships, enabling more precise, hierarchical retrieval while reducing irrelevant information.](assets/page_0002_img_1.png)

To address these limitations, we propose NodeRAG, which is built around a well-designed Heterogeneous Graph, comprehensively considering the entire process of graph indexing and searching, enabling fine-grained retrieval. The heterograph adheres to the principle of unfolding and flattening, decomposing different types of information to construct a heterogeneous fully nodelized graph where nodes serve distinct functions and roles. This means that entities, relationships, original text chunks, independently decomposed events from text chunks, and summaries extracted by LLMs are all represented as nodes within the graph. The heterograph not only encapsulates information from the original corpus but also extends beyond it, incorporating enriched insights such as key node attributes, and high-level discoveries. Each node in heterograph consists of unstructured content, while preserving structural connections between nodes, striking a balance between structural integrity and flexibility. As illustrated in Figure 1, for a multi-hop question, NodeRAG can retrieve a semantically coherent, independent event (semantic unit) and high-level discoveries (high-level elements) related to key entities such as Harry, Neville, and the three-headed dog using graph algorithms, providing explainable and fine-grained retrievals as well as high-level understanding.

The key contributions of our work can be summarized in three main aspects.

(1) **Better Graph Structure for RAG**: The graph structure serves as the foundation for graph-based RAG where significance has been overlooked. Our work emphasizes its importance and introduces a graph structure that better supports RAG.

(2) **Fine-grained and Explainable Retrieval**: The heterograph enables fine-grained and functionality distinct nodes, allowing graph algorithms to effectively and reasonably identify key multi-hop nodes. This leads to more relevant retrieval with minimal retrieval context, enhancing both precision and interoperability.

(3) **Unified-Level Information Retrieval**: Decomposed information from documents and extracted insights from LLMs are not treated as separate layer but are instead unified as nodes within the heterograph. This integration allows for a cohesive framework capable of handling information across different levels.

In addition, extensive experiments demonstrate that NodeRAG not only outperforms previous graph-based RAG methods on multi-hop tasks but also exhibits superior performance in open-ended head-to-head evaluations. With minimal retrieval tokens, it achieves highly precise retrieval while also demonstrating system-level efficiency advantages, including improvements in indexing time, query time, and storage efficiency, as shown in appendix A.
```

### --- Page 0003 ---

```markdown
## 2 NodeRAG

The NodeRAG pipeline is built on a foundational graph structure defined as the heterograph, which will be introduced in Section 2.1. The workflow is divided into two primary stages, graph indexing and graph searching. Graph indexing comprises three components, graph decomposition, graph augmentation, and graph enrichment, which are discussed in Sections 2.2, 2.3, and 2.4, respectively. This stage integrates various types of nodes and edges into the heterograph by leveraging LLMs and graph algorithms. The subsequent stage, graph searching, is detailed in Section 2.5 and combines the structural advantages of the heterograph with graph algorithms to efficiently retrieve relevant information. Moreover, the fundamental concepts and implementation details of the graph algorithms used in the pipeline are provided in Appendix C, while the prompting instructions for LLMs can be found in Appendix E for reference.

### 2.1 Heterograph

The concept of the heterograph embodies the principle of comprehensive unfolding and flattening of information into a fully normalized structure. This structure achieves its granularity through the integration of seven hetero node types: entity (N), relationship (R), semantic unit (S), attribute (A), high-level elements (H), high-level overview (O), and text (T). Each node type is tailored to represent specific roles and characteristics of the information, enabling a fine-grained and functional decomposition of data. Mathematically, the heterograph is defined as:

$$
\mathcal{G} = (V, E, \Psi),
$$

where $\mathcal{G}$ is the heterograph, $V$ represents the set of nodes, $E$ is the set of edges, and $\Psi : V \to \mathcal{Y}$ is a mapping function that assigns each node $v \in V$ to a specific type. The set of node types corresponds to the seven predefined types:

Types = {N, R, S, A, H, O, T}.

For any node $v$, $\psi(v)$ defines its type, with each node type performing a distinct and well-defined function, as detailed in subsequent sections and appendix C. For each $e \in E$, the default weight of $e$ is set to 1, representing a basic connection between two nodes. Furthermore, we define $V_{types}$ as the subset of nodes corresponding to a subset set types $\subseteq \mathcal{Y}$, formally expressed as:

$$
V_{types} = \{v \in V | \Psi(v) \in types\}.
$$

For instance, $V_{N,R,S}$ represents the subset containing only entity, relationship, and semantic unit nodes. $V_{T,S,A,H}$ contain rich informational content and are classified as retrievable nodes. In contrast, $V_{N,O}$, which represent names or titles, act solely as critical linkage and entry points within the graph but are not directly retrievable. For example, $V_H$ provides detailed context for high-level concepts, while $V_O$ represents the corresponding title and keywords but does not contribute directly to the retrieved content. Additionally, $V_R$ is a nodalized edge, acting as connector nodes and secondary retrievable nodes, contributing to the retrieval context but not serving as graph entry points.

### 2.2 Graph Decomposition

First, we define a null heterograph $G^0$. The initial step involves employing a LLM to decompose text chunks from the source corpus into three primary node types: semantic units (S), entities (N), and relationships (R). These nodes are then interconnected to construct the initial heterograph. This process can be formalized as:

$$
G^1 = G^0 \cup \{v_e, e_{a}, e_{r} \in E | v \in \{S, N, R\}\},
$$

where $e$ represents the connecting edges between semantic units and entity nodes, as well as between relationship nodes and their corresponding source and target entities. For instance, if “Hinton was awarded the Nobel Prize for inventing backpropagation” serves as $v \in V_S$ derived from a text chunk, then Hinton, Nobel Prize, and backpropagation represent $v \in V_N$ nodes, with $e_d$ denoting their connections to $v \in V_S$. An example of $v \in V_R$ would be “Hinton received Nobel Prize”, where $e_r$ represents the edge connecting the source node Hinton to the target node Nobel Prize.

#### Semantic unit (S)

The semantic unit acts as a local summary, representing an independent event unit in a paraphrased form. It serves as the core node for graph augmentation and improving search quality. Since the division of text chunks is not based on semantics, unrelated or unassociated content may coexist within a single chunk. This text noise increases entropy, leading to degraded quality when using text chunks for graph augmentation or searching due to their coarse granularity and irrelevant information.

#### Entity (N) and Relationship (R)

Entities (N) are nodes that exclusively represent entity names.
```

### --- Page 0004 ---

```markdown
![Main indexing workflow of NodeRAG, illustrating the step-by-step construction of the heterograph, including graph decomposition, graph augmentation, and graph enrichment.](assets/page_0004_img_1.png)

while relationships ($R$) are also transformed into nodes that connect source and target entities. These entities and relationships are directly connected to semantic units ($S$), as $v \in V_S$ serves as the smallest, contextually meaningful representation of events within text chunks. This connection ensures that entities and relationships remain decoupled from specific events, allowing them to function independently while still being anchored to relevant contexts. Such a design prevents redundant information and enables a flexible graph structure.

### 2.3 Graph Augmentation

The heterograph $G^1$ provides a foundational low-level structure. However, it lacks high-level organization and contextual insights. To further augment the graph, we implement two primary methods: node importance-based augmentation and community detection-based aggregation, which respectively capture the perspectives of individual node significance and structural cohesion within the graph.

#### Node Importance Based Augmentation

We prioritize the selection of structurally significant and functionally pivotal entities. These key entities, along with their associated semantic units and relationships, are processed through LLMs to generate attribute summaries. This approach mirrors human reading behavior, where all relevant content associated with a critical entity is reviewed before synthesizing its attributes. The summarization specifically focuses on the important entities identified within the corpus, rather than processing all entities, ensuring both precision and efficiency. The selection of important entities, $N^*$, is guided by two complementary metrics: $K$-core decomposition (Seidman, 1983; Kong et al., 2019) and betweenness centrality (Brandes, 2001). $K$-core identifies nodes in densely connected subgraphs that are critical to graph cohesion, while betweenness centrality highlights nodes that act as bridges for information flow. These metrics are denoted as $K(G^1)$ and $B(G^1)$, where $K(\cdot)$ and $B(\cdot)$ represent the selected entity nodes from the graph. The final set of important entities is defined as:

$$
N^* = K(G^1) \cup B(G^1).
$$

Entity attributes are constructed directly from relationships and semantic units, bypassing raw texts to avoid redundancy. Each generated attribute node is added to the graph and connected to its corresponding entity node via the edge $e_a$. This update to the graph is represented as:

$$
G^2 = G^1 \cup \{v \in V, e_a \in E | v(e) \in \{A\}\}.
$$

#### Community Detection Based Aggregation

We first apply the Leiden algorithm (Traag et al., 2019) to $G^2$ to perform community detection, segment-
```

### --- Page 0005 ---

```markdown
| **Section** | **Content** |
|-------------|-------------|
| **Graph Construction** | The process of generating the graph into closely related substructures, denoted as communities. Each node $v \in G^2$ is assigned to a specific community $C_n$, where $C_n$ represents the n-th community identified by the algorithm. Within each community $C_n$, an LLM is utilized to analyze the aggregated content, extracting high-level elements ($H$) that encapsulate the core information of the community, such as summaries, sentiment analysis, and other significant insights. For each generated high-level element node $v \in V_H$, it is essential to establish meaningful connections $e_h$ with relevant nodes from $G^2$ to preserve the graph’s structural coherence. To accomplish this, we propose semantic matching within community algorithm. This algorithm identifies the most semantically related nodes within the same community $C_n$ for each high-level element node. To achieve this, K-means clustering (MacQueen et al., 1967) is applied to the embeddings of $v \in V_{S,A,H}$. The number of clusters $K$ is determined as $K = \sqrt{|V_{S,A,H}|}$, where $|V_{S,A,H}|$ represents the total number of nodes labeled $S, A$, or $H$. An edge $e(h, v')$ exists between $v \in V_{S,A,H}$ and $v' \in V_H$ if both $v$ and $v'$ belong to the same semantic cluster $S_k$ and the same community $C_n$. Additionally, the LLM can extract a keyword title for each high-level element, referred to the high-level overview ($O$), which is used for dual search as elaborated in Section 2.5. Each $v \in V_H$ and $v \in V_0$ will have a corresponding connection $e_o$. The updated graph $G^3$ incorporates high-level elements ($H$) and their corresponding connections ($e_h, e_o$). It is defined as: $$G^3 = G^2 \cup \{v \in V, e_h, e_o \in E | \psi(v) = \{H, O\}\}.$$ |
| **Graph Enrichment** | In the previous process of generating the heterograph, $G^3$ already contains a wealth of information. However, certain unique and additional details can still further enrich the heterograph, enabling it to not only preserve the entirety of the original text’s information but also gain enhanced features and insights that go far beyond the source material. **Text Insertion**: As mentioned earlier, text chunks are not directly incorporated into $G$ during graph augmentation due to their semantic incoherent nature. However, original text chunks hold significant value as they contain detailed information, which is often lost during the LLM transformation process. Therefore, it is essential to ensure that the original information remains searchable within the graph. $$G^4 = G^3 \cup \{v | \psi(v) = T\},$$ where $e_s$ denotes the edges connecting text chunks to their relevant semantic units. |
| **Embedding** | As mentioned in Section 2.1, $v \in V_{T,A,S,H}$ contains rich informational context where vector similarity is highly effective. Conversely, $v \in V_{N,O}$, which includes names and titles represented as words or phrases, is less suitable for vector similarity methods. To address this limitation, we developed a dual search mechanism. During the embedding process, we selectively embed only a subset of the graph’s data, specifically $v \in V_{T,A,S,H}$. This targeted embedding step is crucial for reducing storage overhead while preserving efficient search capabilities. |
| **HNSW Semantic Edges** | The Hierarchical Navigable Small World (HNSW) algorithm (Malkov and Yashunin, 2018) is an approximate nearest neighbor search method that organizes data into a multi-layer graph structure to efficiently retrieve semantically similar nodes. It represents the data as a layered graph $H = \{E_0, L_1, \ldots, L_m\}$, where $L_0$ is the base layer containing the densest semantic similarity connections, and higher layers ($L_i, i > 0$) are sparsely connected to facilitate coarse-grained navigation. $H$ is built iteratively. When a new node is added, it is inserted into a random level and all layers below it, connecting to similar neighbors based on cosine similarity. Higher layers remain sparse with long-range connections, while $L_0$ focuses on dense local relationships. The search starts at the sparsely connected top layer, and progressively descends to $L_0$. In our work, the base layer $L_0$ of the HNSW graph, which encodes semantic relations between nodes, is integrated with the heterograph $G$. The updated graph, denoted as $G^5$, is expressed as: $$G^5 = G^4 \cup L_0.$$ The inclusion of $L_0$ enhances the heterograph’s search capabilities by incorporating semantic dense proximity edges, augmenting its structural information in the graph. When an edge already exists in $G^4$, adding the corresponding edge from $G$ increases its weight by 1, reinforcing frequently occurring connections. |
| **Graph Searching** | We first apply a dual search mechanism to identify entry points within the heterograph. Subsequently, |
```

### --- Page 0006 ---

```markdown
![This figure focuses on the querying process, where entry points are extracted from the original query, followed by searching for related nodes that need to be retrieved in the heterograph.](assets/page_0006_img_1.png)

A shallow Personalized PageRank (PPR) algorithm is employed to extract cross nodes. The combination of entry point nodes and cross nodes is then filtered to produce the final retrieval.

## Dual Search
Dual search combines exact matching on title nodes and vector similarity search on rich information nodes to identify entry points in the heterograph G. Given a query, the LLM extracts entities $N^q$ and embeds the query into vector (q). The entry points are defined as:

$$
V_{entry} = \{ v \in V \ | \ \Phi(v, N^q) \},
$$

where the condition function $\Phi(v, N^q)$ is defined as:

$$
\Phi(v, N^q) = \{ v \in V_{N,O} \land M(v, v), \ v \in V_{S,A,H} \land R(q, v, k) \}.
$$

Here, the exact matching function $M(v, v^*)$ returns true if a node matches one of the extracted entities by word level string matching. Additionally, the similarity-ranking function $R(q, v, k)$ returns true if a node ranks among the top-k most similar to q based on the HNSW algorithm. By leveraging the non-retrievability of $v \in V_{N,O}$, they serve exclusively as entry points to the graph without contributing directly to the retrievable content. Only nodes identified through the shallow PPR as closely related to all entry points are included in the retrieval results as cross nodes. This ensures that the effects of noisy or ambiguous queries, which may lead to errors in exact matching, do not directly impact the retrieval process. Any indirect effects are further minimized by the graph algorithm, enhancing the robustness of the retrieval system.

## Shallow PPR
Personalized PageRank (PPR) identifies relevant nodes in the heterograph G by simulating a biased random walk starting from a set of entry points. In our approach, we use shallow PPR, limiting the number of iterations t to ensure that relevance remains localized to the neighborhoods of the entry points. This early stop strategy prevents excessive diffusion to distant or irrelevant parts of the graph, focusing instead on multi-hop nodes near the entry points. Let P be the normalized adjacency matrix of G, where $P_{ij}$ represents the transition probability from node i to node j. The PPR process starts with a personalization vector $p \in \mathbb{R}^{|V|}$, where $p_i = 1/|V_{entry}|$ if $v_i \in V_{entry}$, and $p_i = 0$ otherwise. The PPR score vector $\pi^{(t)}$ after t iterations is computed iteratively as:

$$
\pi^{(t)} = \alpha p + (1 - \alpha) P^T \pi^{(t-1)},
$$

where $\alpha \in (0, 1)$ is the teleport probability that balances restarting at entry points and propagating through the graph. After t iterations, the top-k nodes with the highest PPR scores for each type are selected as cross nodes, denoted as $V_{cross}$. In our default setting, we use $\alpha = 0.5$ and $t = 2$ to achieve a balance between exploration and convergence.

## Filter Retrieval Nodes
Finally, the retrieval nodes are filtered from the union of entry nodes and cross nodes to include only retrievable nodes of $v \in V_{T,A,S,H,R}$, $v \in V_{N,O}$, which contain only keywords without informational content, are excluded from the retrieval context. The final set of retrieval nodes is therefore defined as:

$$
V_{retrieval} = \{ v \in V_{entry} \cup V_{cross} \ | \ \psi(v) \in \{ T, S, A, H, R \} \}.
$$

## 3 Evaluation
We evaluate NodeRAG's performance across three different multiphon benchmarks, HotpotQA (Yang et al., 2018), MusiQue (Trivedi et al., 2022b), MultiHop-RAG (Tang and Yang, 2024), and an open-ended head to head evaluation RAG-QA Arena (Han et al., 2024b) across six domains. And we compare our method against several strong and widely used RAG methods as baseline models, including NaiveRAG (Lewis et al., 2020), HyDe.
```

### --- Page 0007 ---

```markdown
# Part 1: General comparisons

| Methods         | HotpotQA | Acc (%) | M2 | Win (M1) | Win (M2) | Domain | MuSiQue | Acc (%) | M2 | Win (M1) | Win (M2) | MultiHop | Arena-Trivia | Acc (%) | M2 | Win (M1) | Win (M2) | Arena-Science | Acc (%) | M2 | Win (M1) | Win (M2) | Arena-Recreation | Acc (%) | M2 | Win (M1) | Win (M2) | Arena-Lifestyle | Acc (%) | M2 | Win (M1) | Win (M2) | Arena-FQA | Acc (%) | M2 | Win (M1) | Win (M2) |
|-----------------|----------|---------|----|----------|----------|--------|---------|---------|----|----------|----------|----------|--------------|---------|----|----------|----------|----------------|---------|----|----------|----------|------------------|---------|----|----------|----------|------------------|---------|----|----------|----------|-----------|---------|----|----------|----------|
| NaiveRAG        | 37.05    | 90.1    | 9.8| 0.63     | 0.63     | 0.63   | 0.52    | 90.1    | 9.4| 0.79     | 0.79     | 0.51     | 0.52         | 0.70    | 91.1| 0.92     | 0.92     | 0.77          | 9.3     | 0.77| 0.79     | 0.79     | 0.82             | 9.2     | 0.82| 0.79     | 0.79     | 0.82          | 9.2     | 0.82| 0.79     | 0.79     | 0.82      | 9.2     | 0.82| 0.79     | 0.79     |
| HyDE            | 70.00    | 31.34   | 9.5| 0.79     | 0.79     | 0.79   | 0.79    | 90.1    | 9.4| 0.79     | 0.79     | 0.51     | 0.52         | 0.70    | 91.1| 0.92     | 0.92     | 0.77          | 9.3     | 0.77| 0.79     | 0.79     | 0.82             | 9.2     | 0.82| 0.79     | 0.79     | 0.82          | 9.2     | 0.82| 0.79     | 0.79     | 0.82      | 9.2     | 0.82| 0.79     | 0.79     |
| LightRAG        | 79.00    | 71.06   | 30.6| 7.9     | 0.74     | 0.74   | 0.74    | 90.1    | 9.4| 0.79     | 0.79     | 0.51     | 0.52         | 0.70    | 91.1| 0.92     | 0.92     | 0.77          | 9.3     | 0.77| 0.79     | 0.79     | 0.82             | 9.2     | 0.82| 0.79     | 0.79     | 0.82          | 9.2     | 0.82| 0.79     | 0.79     | 0.82      | 9.2     | 0.82| 0.79     | 0.79     |
| GraphRAG        | 80.00    | 41.76   | 6.06| 0.53     | 0.74     | 0.74   | 0.74    | 90.1    | 9.4| 0.79     | 0.79     | 0.51     | 0.52         | 0.70    | 91.1| 0.92     | 0.92     | 0.77          | 9.3     | 0.77| 0.79     | 0.79     | 0.82             | 9.2     | 0.82| 0.79     | 0.79     | 0.82          | 9.2     | 0.82| 0.79     | 0.79     | 0.82      | 9.2     | 0.82| 0.79     | 0.79     |
| NodeRAG         | 89.95    | 54.26   | 9.58| 0.57     | 0.61     | 0.61   | 0.61    | 90.1    | 9.4| 0.79     | 0.79     | 0.51     | 0.52         | 0.70    | 91.1| 0.92     | 0.92     | 0.77          | 9.3     | 0.77| 0.79     | 0.79     | 0.82             | 9.2     | 0.82| 0.79     | 0.79     | 0.82          | 9.2     | 0.82| 0.79     | 0.79     | 0.82      | 9.2     | 0.82| 0.79     | 0.79     |

# Part 1: Pairwise Comparisons

| Domain | M1 vs M2 | Win (M1) | Win (M2) | Domain | M1 vs M2 | Win (M1) | Win (M2) |
|--------|----------|----------|----------|--------|----------|----------|----------|
| FIQ    | NaiveRAG vs GraphRAG | 0.63     | 0.21     | Recreation | NaiveRAG vs LightRAG | 0.63     | 0.31     |
| NodeRAG vs LightRAG | 0.63     | 0.14     | NodeRAG vs GraphRAG | 0.63     | 0.14     |
| LightRAG vs NodeRAG | 0.63     | 0.14     | Writing | GraphRAG vs LightRAG | 0.63     | 0.14     |
| Lifestyle | NaiveRAG vs GraphRAG | 0.63     | 0.14     | Science | GraphRAG vs LightRAG | 0.63     | 0.14     |
| Tech | GraphRAG vs NodeRAG | 0.63     | 0.14     | | | | |

(Gao et al., 2022a; GraphRAG (Edge et al., 2024); LightRAG (Guo et al., 2024). The details of these datasets and baseline models are introduced in Appendix B.

## 3.1 Metrics

### General Comparison
In the first part, we evaluate NaiveRAG, HyDE, LightRAG, GraphRAG, and NodeRAG across four benchmark datasets. For HotpotQA and MuSiQue benchmarks, we assess accuracy (Acc) to measure effectiveness and the average number of retrieved tokens (#Token) to evaluate efficiency. For the MultiHop-RA benchmark, we adopt its original evaluation metric, Score (Sco), while still using #Token to gauge retrieval efficiency. Lastly, for the RAG-QA Arena benchmark, we continue to track #Token for efficiency and employ a win and tie ratio (W+T) against responses as a measure of performance across different methods.

### Pairwise Comparison
In this part, the evaluation focuses exclusively on the RAG-QA Arena benchmark, covering six domains: FIQ, Recreation, Writing, Lifestyle, Science, and Technology. We conduct comprehensive pairwise comparisons among all method combinations and calculate the corresponding win and tie rates for each matchup, thereby identifying the better RAG system.

## 3.2 Implementation details
By default, all these RAG methods are implemented with GPT 4.0-mini, and the temperature is set to 0 across the entire evaluation. Meanwhile, we identify a potential unfairness in the current evaluation setup, evident in several key areas. Notably, the baselines vary in their choice of prompts used to synthesis the final response based on retrieved information. Therefore, we standardized response prompts for every method. Our initiative to standardize these settings also benefits other methods like GraphRAG, improving their performance compared to their default setting, underscoring the broader value of establishing fair and consistent evaluation standards.
```

### --- Page 0008 ---

```markdown
## 3.3 Results

### General Comparison
As shown in Part I of Table 1, NodeRAG consistently outperforms competing methods on HotpotQA, MuSiQue, and MultiHopRAG, demonstrating the highest accuracy while retrieving noticeably fewer tokens. For example, for MuSiQue, NodeRAG attains an accuracy of 46.29%, surpassing GraphRAG (41.71%) and LightRAG (36.00%). In HotpotQA, while NodeRAG achieves a slightly higher accuracy (89.50% vs. 89.00% for GraphRAG), it does so with only 5k retrieved tokens, which is 1.6k fewer than GraphRAG. In the RAG-QA Arena benchmark, graph-enhanced RAG systems exhibit a clear advantage over traditional approaches. Notably, NodeRAG achieves the highest win and the ratio in each of the five domains while keeping retrieval costs minimal. For example, it attains a ratio of 94.9%, notably surpassing GraphRAG's 86.3% and LightRAG's 81.7% in the Lifestyle domain, and does so with less than half the retrieved tokens compared to the other models. It can also be noted that graph-enhanced RAG systems generally retrieve fewer tokens than traditional RAG across all benchmarks. These results confirm NodeRAG's remarkable effectiveness and efficiency, demonstrating that our heterograph can significantly boost RAG performance across diverse tasks.

### Pairwise Comparison
Across all the six domains, NodeRAG consistently achieves higher win ratios against GraphRAG, LightRAG, NaiveRAG, and HyDE, demonstrating notable dominance, for instance, in the Lifestyle domain, NodeRAG achieves 0.640 win rate against GraphRAG, 0.623 against LightRAG, 0.800 against NaiveRAG and 0.526 against HyDE. GraphRAG, LightRAG, NaiveRAG, and HyDE show scattered successes, such as LightRAG edging out NaiveRAG (0.649 vs. 0.246) in Recreation, GraphRAG beats LightRAG (0.361 vs. 0.296) in Science, yet their overall win rates remain lower when compared to NodeRAG. Notably, these trends persist across other domains like Writing, Recreation, Science, and Tech, further underscoring NodeRAG's leading position, followed by LightRAG and GraphRAG, showing the superiority of our method.

In general, NodeRAG not only achieves the highest accuracy rate and the lowest retrieval token count in general benchmarks but also outperforms all other baselines in performance evaluation comparisons. This unparalleled performance in both accuracy and computational efficiency makes NodeRAG the optimal choice for a wide range of RAG tasks, from research applications to deployments in resource-constrained environments.

## 4 Ablation experiments
![Ablation analysis on PPR iterations](assets/page_0008_img_1.png)

We conducted ablation experiments on the MuSiQue dataset, adhering to the same settings and evaluation metrics described earlier. We specifically examined the impact of four key submodules: shallow PPR, cross-node interactions, HNSW semantic edges, and dual search.

We first investigated the variation in PPR iterations and examined whether shallow PPR offers advantages. PPR, with a few iterations, performs better than deep PPR because it highlights important nodes that are closer to the entry points. Moreover, early stopping reduces unnecessary computational overhead, leading to improved retrieval efficiency.

Moreover, we evaluate the performance of applying the top-k vector similarity method to all node data in the graph. Although increasing the retrieval context, its performance remains lower than the basic version. This confirms the necessity of cross-nodes in our method, as they help identify important multi-hop nodes. Second, performing vector similarity solely on node data consistently outperforms the naive RAG approach of similarity in text chunks, demonstrating the advantages brought by graph-based data augmentation.

In addition, without integration of accurate search in dual search, accuracy drops to 44.57%, and the token count increases to 9.7k. This is because losing entity and high-level overview nodes as entry points causes nodes with long texts, such as text nodes, to have higher weights after shallow PPR. Since vector similarity entry nodes are more frequently connected to $T$ nodes, while accurate entry nodes are more connected to $S$, $A$, and $H$.
```

### --- Page 0009 ---

```markdown
| Method            | Accuracy | Time (s) | Tokens (k) |
|-------------------|----------|----------|------------|
| NodeRAG (Ours)    | 46.29%   | 4.05     | 5.96       |
| wo HNSW           | 41.71%   | 4.92     | 6.78       |
| wo Dual Search    | 44.57%   | 4.72     | 9.70       |
| wo Cross Node     |          |          |            |
| Top-k = 10        | 41.71%   | 4.15     | 4.27       |
| Top-k = 20        | 43.43%   | 4.70     | 7.89       |
| Top-k = 30        | 42.29%   | 4.80     | 11.62      |

## 5 Related Works

### Retrieval-augmented generation
Retrieval-Augmented Generation (RAG) systems (Gupta et al., 2024) enhance the performance of large language models (LLM) by retrieving relevant information from external documents, grounding responses in domain-specific knowledge. Traditional RAG approaches (Zhao et al., 2024) embed user queries and entries from a knowledge base into a shared vector space and then compare query vectors to knowledge base vectors to retrieve the top-$K$ most similar contexts based on cosine similarity or similar variants (Fan et al., 2024; Lewis et al., 2022). While effective, naive RAG methods face several limitations, prompting various enhancements in subsequent works. JPR (Min et al., 2021) improves multi-answer retrieval by refining passage selection, while IR-CoT (Trivedi et al., 2022a) integrates chain-of-thought reasoning for multi-hop question answering. Similarly, Tree of Clarifications (Kim et al., 2023) constructs a tree-based disambiguation structure to resolve ambiguous queries. HyDE (Gao et al., 2022b) also enhances the performance of dense retrieval by generating hypothetical documents. Other works examine how different document types influence RAG effectiveness and LLM performance (Hsia et al., 2024). Despite these advancements, traditional RAG systems still face significant challenges. The context window limitations (Cheng et al., 2024; Su et al., 2024) of LLMs constrain their ability to process extensive external documents holistically (Jiang et al., 2024b). RAG has been applied to various domain-specific knowledge bases, such as BioRAG and MedicalRAG (Wang et al., 2022a; Wu et al., 2024; Jiang et al., 2024a). RAG also struggles with corpus-wide understanding tasks, like query-focused abstractive summarization, which require synthesizing knowledge across large datasets.

### RAG over Hierarchical Index
To overcome the limitations of traditional RAG, advanced systems integrate hierarchical indexing to incorporate document summaries and enhance retrieval performance. Dense Hierarchical Retrieval (DHR) (Liu et al., 2021) improves passage representations by combining macroscopic document semantics with microscopic passage details. Expanding on this, Hybrid Hierarchical Retrieval (HHR) (Arivazhagan et al., 2023) fuses sparse and dense retrieval techniques for both document- and passage-level retrieval, achieving greater precision. Other methods leverage hierarchical data structures to facilitate complex document summarization. For instance, RAPTOR (Sarthi et al., 2024) employs tree-based structures to integrate knowledge across lengthy documents, synthesizing information at various levels of abstraction. Graph-based RAGs (Trajanskos et al., 2023; Zhang et al., 2024) extend this by constructing knowledge graphs (KGs) (Chen et al., 2020) at the indexing stage and applying graph algorithms during querying (Haveliwala et al., 2003). Notable examples include HiPoRAG (Jiménez Gutiérrez et al., 2024) and KAPING (Baek et al., 2023), which refine knowledge organization and retrieval efficiency. Similarly, GraphRAG (Edge et al., 2024) introduces graph-based text indexing using LLMs and generates community-based summaries (Blondet et al., 2008; Traag et al., 2019), inspiring subsequent works such as LightRAG (Guo et al., 2024), which integrates both high- and low-level information while optimizing indexing costs. While these approaches effectively leverage hierarchical data structures, they do not fully exploit the synergy between LLMs and graph-based methods. Our proposed framework addresses these gaps by refining graph structures and incorporating advanced graph algorithms, leading to superior retrieval accuracy and efficiency.

![Ablation study of NodeRAG components](assets/page_0009_img_1.png)
```

### --- Page 0010 ---

```markdown
# 6 Conclusion and Discussion

In this paper, we introduce NodeRAG, a novel framework designed to enhance RAG performance by optimizing graph structures in indexing for more effective and fine-grained retrieval. NodeRAG constructs a well-defined heterograph with functionally distinct nodes, balancing fine-grained understanding with a global perspective of the knowledge corpus. Experimental results demonstrate that NodeRAG outperforms existing methods across multi-hop reasoning benchmarks and open-ended retrieval tasks. As the saying goes, “A strong foundation supports a higher structure”. In the realm of graph-based RAG, the graph structure serves as this very foundation. The introduction of NodeRAG underscores the critical role of graph structures, encouraging a renewed emphasis on their design and optimization.

## References

| Author(s) | Title | Year | Source |
|-----------|-------|------|--------|
| Manoj Guhan Arivazhagan, Lan Liu, Peng Qi, Xinchi Chen, William Yang Wang, and Zhiheng Huang. | Hybrid hierarchical retrieval for open-domain question answering. | 2023 | In Findings of the Association for Computational Linguistics: ACL 2023, pages 10680–10689. |
| Jinhoon Baek, Alham Fikri Aji, and Amir Safar. | Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. | 2023 | arXiv preprint arXiv:2306.04136. |
| Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. | Fast unfolding of communities in large networks. | 2008 | Journal of statistical mechanics: theory and experiment, 2008(10):P10008. |
| Ulrik Brandes. | A faster algorithm for betweenness centrality. | 2001 | Journal of mathematical sociology, 25(2):163–177. |
| Zhe Chen, Yuehan Wang, Bin Zhao, Jing Cheng, Xin Zhao, and Zongtao Duan. | Knowledge graph completion: A review. | 2020 | IEEE Access, 8:192435–192456. |
| Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan. | Lift yourself up: Retrieval-augmented text generation with self-improvement. | 2024 | Advances in Neural Information Processing Systems. |
| Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. | From local to global: A graph rag approach to query-focused summarization. | 2023 | arXiv preprint 2404.16150. |
| Wengji Fan, Yujian Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. | A survey on rag meeting lms: Towards retrieval-augmented large language models. | 2024 | In International Conference on Knowledge Discovery and Data Mining (KDD), pages 6491–6501. |
| Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. | Precise zero-shot dense retrieval without relevance labels. | 2022a | arXiv preprint arXiv:2212.10496. |
| Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. | Precise zero-shot dense retrieval without relevance labels. | 2022b | arXiv preprint arXiv:2212.10496. |
| Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. | Lightrag: Simple and fast retrieval-augmented generation. | 2024 | arXiv preprint 2410.05779. |
| Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. | A comprehensive survey of retrieval-augmented generation (rag): Evolution, current landscape and future directions. | 2024 | arXiv preprint 2410.12837. |
| Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Manthesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al. | Retrieval-augmented generation with graphs (graphra). | 2024a | arXiv preprint arXiv:2501.00309. |
| Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan Wang, Lan Liu, William Yang Wang, Bonan Min, and Vittorio Castelli. | Rag-a-arena: Evaluating domain robustness for long-form retrieval augmented question answering. | 2024b | arXiv preprint arXiv:2407.13998. |
| Taher Haveliwala, Sepandar Kamvar, and Glen Jeh. | An analytical comparison of approaches to personalizing pagerank. | 2003 | Technical report, Stanford. |
| Xiaochin He, Yujin Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yan LeCun, Xavier Bresson, and Bryan Hooi. | G-retriever: Retrieval-augmented generation for textual graph understanding and questioning answering. | 2024 | arXiv preprint arXiv:2402.07630. |
| Jennifer Hsia, Afreen Shaikh, Zhiruo Wang, and Graham Neubig. | Ragged: Towards informed design of retrieval augmented generation systems. | 2023 | arXiv preprint arXiv:2403.09040. |
| Xinke Jiang, Yue Fang, Rihong Qiu, Haoyu Zhang, Yongxin Xu, Hao Chen, Wentao Zhang, Ruizhe Zhang, Yuchen Fang, Yu Chu, et al. | Turing-complete rag’s case study on medical lms systems. | 2024a | arXiv preprint arXiv:2408.099. |
| Ziyan Jiang, Xueguang Ma, and Wenhui Chen. | Longrag: Enhancing retrieval-augmented generation with long-context lms. | 2024b | arXiv preprint arXiv:2406.15319. |
```

### --- Page 0011 ---

```markdown
| Author(s)                                                                 | Title                                                                                                   |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. | Hippogra: Neurobiologically inspired long-term memory for large language models. arXiv preprint arXiv:2405.14831. |
| Gangwoo Kim, Sungdong Kim, Byeonguk Jeon, Joon-suk Park, and Jaewoo Kang. 2023. | Tree of clarifications: Answering ambiguous questions with retrieval-augmented large language models. arXiv preprint arXiv:2310.14696. |
| Yi-Xiu Kong, Gui-Yuan Shi, Rui-Jie Wu, and Yi-Cheng Zhang. 2019.          | K-core: Theories and applications. Physics Reports, 832:1–32.                                         |
| Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. | Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33:9459–9474. |
| Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michèle Bevilacqua, Fabio Petroni, and Percy Liang. 2024. | Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173. |
| Ye Liu, Kazuma Hashimoto, Yingbo Zhu, Semih Yavuz, Caiming Xiong, and Philip S Yu. 2021. | Dense hierarchical retrieval for open-domain question answering. arXiv preprint arXiv:2110.15439.       |
| Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shiriun Pan. 2023.      | Reasoning on graphs: Faithful and interpretable large language model reasoning. arXiv preprint arXiv:2310.11061. |
| James MacQueen et al. 1967.                                              | Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability, volume 1, pages 281–297. Oakland, CA, USA. |
| Yu A Malkov and Dmitry A Yashunin. 2018.                                 | Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4):824–836. |
| Sewon Min, Kenton Lee, Ming-Wei Chang, Kristina Toutanova, and Hannah Hajishirzi. 2021. | Joint passage ranking for diverse multi-answer retrieval. arXiv preprint arXiv:2104.08445.             |
| Jinyoung Park, Ameen Patel, Omar Zia Khan, Hyunwoo J Kim, and Joo-Kyung Kim. 2023. | Graphed reasoning for multi-hop question answering in large language models. arXiv preprint arXiv:2311.09762. |
| Diego Sanmartin. 2024.                                                  | Kg-rag: Bridging the gap between knowledge and creativity. arXiv preprint arXiv:2405.12033.           |
| Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Mann. 2024. | Raptor: Recursive attractive processing for tree-organized retrieval. arXiv preprint arXiv:2401.18059. |
| Stephen B Seidman. 1983.                                                | Network structure and minimum degree. Social networks, 5(3):269–287.                                   |
| Robik Shrestha, Yang Zou, Qiuy Chen, Zhiheng Li, Yusheng Xie, and Siqi Deng. 2024. | Fairrag: Fair human generation via fair retrieval augmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1196–1205. |
| Weihang Su, Yichen Tang, Qingyao Ai, Zhijhing Wu, and Yiqun Liu. 2024.  | Dragin: Dynamic retrieval-augmented generation based on the real-time information needs of large language models. arXiv preprint arXiv:2403.10081. |
| Yixuan Tang and Yi Yang. 2024.                                          | Multihop-rag: Benchmarking retrieval-augmented generation for multihop queries. arXiv preprint arXiv:2404.15391. |
| Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Liu, Fang Wang, Nitesh V Chawla, and Pan Xu. 2024. | Graph neural prompting with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19800–19808. |
| SM Tomnov, SM Zaman, Vinija Jain, Anku Rani, Vipul Rawate, Aman Chadha, and Amitava Das. 2024. | A comprehensive survey of hallucination mitigation techniques in large language models. arXiv preprint arXiv:2401.03113. |
| Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. 2019.              | From louvain to leiden: guaranteeing well-connected communities. Scientific reports, 9(1):1–12.        |
| Milena Trajanoska, Riste Stojanov, and Dimitra Tzaranos. 2023.          | Enhancing knowledge graph construction using large language models. arXiv preprint arXiv:2305.04676.  |
| Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022a. | Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. arXiv preprint arXiv:2212.10509. |
| Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022b. | Musique: Multi-hop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554. |
| Chengrui Wang, Qingqing Long, Meng Xiao, Xunxin Cai, Chengjun Wu, Zhen Meng, Xuexi Wang, and Yuanchun Zhao. 2024a. | Biorag: A rag-llm framework for biological question reasoning. arXiv preprint arXiv:2408.01107. |
```

### --- Page 0012 ---

```markdown
Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Dern. 2024b. Knowledge graph prompting for multi-document question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19206–19214.

Yilin Wen, Zifeng Wang, and Jimeng Sun. 2023. Mindmap: Knowledge graph prompting sparks graph of thoughts in large language models. arXiv preprint arXiv:2308.09729.

Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, and Vicente Grau. 2024. Medical graph rap: Towards safe medical large language model via graph retrieval-augmented generation. arXiv preprint arXiv:2408.04187.

Zhiling Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600.

Yuzhe Zhang, Yipeng Zhang, Yidong Gan, Lina Yao, and Chen Wang. 2024. Causal graph discovery with retrieval-augmented generation based large language models. arXiv preprint arXiv:2402.15301.

Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunting Geng, Fangcheng Fu, Ling Yang, Wentao Zhang, and Bin Cui. 2024. Retrieval-augmented generation for ai-generated content: A survey. arXiv preprint arXiv:2402.19473.
```

### --- Page 0013 ---

```markdown
# A Comparison of RAG System Performance

| Datasets     | Corpus Size | Index Time | Storage Usage | Query Time | Average Retrieval Tokens |
|--------------|-------------|------------|---------------|------------|--------------------------|
| HotpotQA     | 1.93M       | 69min      | 2.1m          | 227MB      | 461MB         | 2.66  | 22.65s | 5.53s |  968.63 |  81029 |  1773.67 |  9079.40 |
| MultiHop     | 1.84M       | 76min      | 25m           | 252MB      | 2.49  | 22.65s | 6.36s |  1661.71 |  11770 |  7435.63 |  20709.99 |
| Arena-Flag   | 1.65M       | 45min      | 2.1m          | 121MB      | 171MB | 5.95  | 23.96s | 13.35s |  3689.45 |  71564 |  71564.72 |  3381.72 |
| Arena-Lifestyle | 1.64M   | 39min      | 18m           | 138MB      | 127MB | 7.34  | 21.10s | 10.81s |  9696.69 |  39669 |  39669.69 |  3485.35 |
| Arena-Recreation | 0.95M   | 32min      | 18m           | 171MB      | 171MB | 5.10  | 23.10s | 6.95s |  6895.95 |  56136 |  56136.56 |  348.13 |
| Arena-Train  | 1.72M       | 54min      | 14m           | 133MB      | 270MB | 7.35  | 26.36s | 8.47s |  6755.84 |  67490 |  6722.83 |  5821.78 |
| Arena-Writing | 1.82M      | 51min      | 15m           | 151MB      | 157MB | 5.66  | 10.70s | 5.46s |  4777.37 |  87754 |  6964.57 |  3373.14 |

The table presents the system performance of mainstream graph-based RAG methods and our proposed approach. Compared to previous work, our method demonstrates superior performance across multiple datasets and in open-ended head-to-head evaluations, while also achieving better system-level efficiency. All evaluations in the table were conducted using the default indexing settings of each RAG method, with the query settings and the prompt details provided in Appendix B.2. Notably, our method demonstrates a significant advantage in indexing time, which is crucial for practical deployment. This advantage is attributed to the construction process of our Hetero Graph, which not only creates a more fine-grained and semantically meaningful graph structure but also carefully considers the algorithmic complexity of the retrieval process.

NodeRAG also exhibits relatively better storage efficiency. Although the total number of nodes in our expanded graph is significantly larger than in previous graph structures, the combination of selective embedding and dual search effectively reduces the number of embedded nodes, leading to a more efficient storage strategy. Moreover, our unified information retrieval approach results in reduced query time. While the GraphRAG local search (Graph-I) relies purely on vector similarity—similar to our "without cross-node" setting mentioned in Section 4—and achieves faster search speeds, its global mode (Graph-G) experiences significantly higher query times, exceeding 20 seconds with a concurrency of 16. This is due to its reliance on LLM-based traversal of all community information, leading to a substantial number of retrieval tokens. Given the considerable time and computational overhead associated with Graph-G queries, we conducted a full evaluation only on the MuSiQue dataset. For other datasets, query time and retrieval token statistics were estimated based on a sample of 20 selected queries. Further details on the ablation study of GraphRAG can be found in the Appendix B.4.

In contrast, our method leverages the heterograph and graph algorithms to achieve unified information retrieval, effectively capturing meaningful information needs across multiple levels within a single framework while maintaining efficient query speed. Finally, the nodes within the heterograph are connected in a fine-grained structure, ensuring that more relevant text is retrieved with relatively fewer retrieval tokens.

## B Experiment details

### B.1 Datasets

We evaluate Node RAG's performance across four different benchmarks: HotpotQA, MuSiQue, MultiHopRAG and RAG-QA Arena. However, the original question formats of HotpotQA and MuSiQue required selecting the most relevant passages from multiple documents, incorporating multi-hop reasoning details. This setup no longer aligns with mainstream RAG methods, as modern approaches perform indexing over an entire corpus and subsequently retrieve information from the indexed data. To adapt to this paradigm, we concatenate all passages into a unified corpus, transforming the task into retrieving multi-hop relevant information from the entire corpus. This modification makes the task more challenging compared to the previous methods.
```

### --- Page 0014 ---

```markdown
# Evaluation Metrics and Datasets

The evaluation metrics for HotpotQA and MuSiQue are divided into two aspects: the quality of the retrieved documents and the accuracy of the final answer, measured by metrics such as F1 score. However, current RAG methods retrieve not only text chunks but also more flexible forms of information, making it difficult to assess retrieval quality using traditional top-k document evaluation. Moreover, metrics like F1 score have become less effective in evaluating answers generated by modern generative models. Therefore, we adopt the LLM-as-a-Judge approach, leveraging LLMs to assess the final accuracy of the generated answers. The MultiHop and RAG-QA Arena dataset settings provide a strong evaluation framework for current RAG methods. Therefore, we follow the original benchmark’s proposed testing methodology and evaluation metrics. Further details regarding the benchmark settings are described below.

## HotpotQA

HotpotQA is a multi-hop question-answering dataset where each question requires combining information from multiple documents to find the correct answer. It encourages deeper reasoning by providing supporting facts—specific sentences from the texts that lead to the solution. Questions range widely across domains and often involve bridging or comparison to ensure more complex, multi-step reasoning. This makes HotpotQA a critical benchmark for evaluating advanced reading comprehension models. We sampled 200 questions from the final dataset for evaluation.

## MuSiQue

MuSiQue is also a multi-hop question-answering dataset that challenges models to combine information across multiple documents in a structured, step-by-step manner. Each question is designed to require several reasoning steps, ensuring that simple “shortcut” approaches do not suffice. As a result, MuSiQue serves as a rigorous test of advanced reading comprehension, demanding that systems accurately connect disparate pieces of evidence to arrive at correct answers. We also sample 175 questions for the evaluation.

## MultiHop-RAG

MultiHop-RAG is a multi-hop question-answering dataset that includes four distinct question types: comparison query, null query, inference query, and temporal query. From this dataset, we curated 375 questions to evaluate our approach. Each query in MultiHop requires synthesizing information from multiple sources, testing a model’s ability to perform bridging inferences, handle temporal relationships, and make higher-order logical connections. This diversity in question types provides a rigorous benchmark for assessing whether RAG methods can integrate scattered pieces of evidence.

## RAG-QA Arena

RAG-QA Arena is a new evaluation framework designed to assess the quality of retrieval-augmented generation (RAG) systems on long-form question answering. It builds on Long-form RobustQA (LFRQA), a dataset of 26K queries across seven domains including writing, tech, science, recreation and lifestyle. Each LFRQA entry features a coherent, human-written answer grounded in multiple documents. RAG-QA Arena leverages LLMs as evaluators, directly comparing a system’s generated answer with the ‘gold’ long-form answer from LFRQA. Experimental results show that these model-based comparisons correlate highly with human judgments, making it a challenging yet reliable benchmark for testing both cross-domain robustness and the ability to produce integrated, long-form responses.

## B.2 Baselines

We compare NodeRAG against several strong and widely used RAG methods. By default, all these RAG methods implement their indexing process using GPT-40-mini. However, we identify a potential unfairness in the current evaluation setup, particularly in several key areas. To ensure the correctness and validity of the evaluation data, it is crucial to standardize both the final answer response prompt and the model temperature settings. Using different response prompts or varying temperature settings for answer generation introduces inconsistencies, as a higher temperature setting may yield responses that receive a better LLM preference score compared to those generated with a lower temperature. A critical point to consider is that, as RAG methods, the primary focus of evaluation should be the quality of the retrieved context rather than the final generated answer. Therefore, to ensure that final accuracy metrics accurately reflect the quality of the retrieved context, the final answer generation process and model settings should remain consistent across all methods. Hence, we set the temperature to 0 across the entire evaluation and standardized response prompts for every method. The unified prompt is illustrated.
```
![Detailed description of the chart](assets/page_0014_img_1.png)

### --- Page 0015 ---

```markdown
In appendix E. Our initiative to standardize these settings also benefits other methods, such as GraphRAG, improving their performance compared to their default settings. This underscores the broader value of establishing fair and consistent evaluation standards. Additionally, traditional evaluation methods such as top-$k$ retrieval comparison have become increasingly difficult to apply uniformly, as retrieval is no longer restricted to isolated text chunks. To address this challenge, we propose a new evaluation standard that leverages retrieval tokens as an efficiency metric. This approach ensures that retrieval methods achieve better effectiveness while utilizing fewer retrieval tokens, promoting a more efficient and fair comparison framework. Current methods can only control the number of retrieval tokens through hyperparameter tuning. Although precise control over the exact number of tokens is not possible, we consider maintaining the average number of retrieval tokens within the range of 5K to 10K to be a reasonable and fair comparison criterion. Below, we provide a detailed introduction to each method along with its specific settings for reference.

### Naive RAG
This method serves as a standard baseline among all existing RAG systems. It first divided input document into several text chunks and encoded them into a vector space utilizing text embeddings. Then retrieve related text chunks based on similarity of query representations. The number of retrieval tokens can be adjusted through the top-$k$ parameter.

### HyDE
HyDE serves as an improved method over traditional RAG systems. It first generates "hypothetical" texts that capture the essence of a query. It then uses this generated text to retrieve relevant documents from a large corpus, employing vector similarity in an embedding space. This method modifies the input query at the frontend without altering the text chunks or their embeddings. Therefore, we can still use the top-$k$ parameter to control the number of retrieval tokens.

### GraphRAG
This approach starts by segmenting the input text into chunks and extracting the entities and relationships within them, forming a graph structure. This graph is then divided into multiple communities at different levels. At query time, GraphRAG identifies the relevant entities from the question and synthesizes answers by referencing these corresponding community summaries. Compared to traditional RAG methods, GraphRAG provides a more structured and high-level understanding of the entire document. Through our experiments, we observed that under the default settings, the number of queries in GraphRAG's local mode resulted in a higher retrieval token count than the naive retrieval approach. To ensure a fair comparison, we proportionally reduced its parameters and standardized its prompt to match our unified prompt. The ablation study in Appendix B.4 demonstrates that after these adjustments, GraphRAG's accuracy improved, further validating the fairness of our evaluation methodology. Additionally, we analyzed both the local and global modes of GraphRAG. Our findings indicate that the global mode introduces significant additional overhead in terms of time and computational cost while providing only marginal improvements compared to the local mode. This result is further supported by our ablation study, which shows that the local mode achieves better efficiency and effectiveness.

### LightRAG
LightRAG is an improved approach based on GraphRAG, designed to minimize computational overhead while enhancing the comprehensiveness of retrieved information through dual-level retrieval. This leads to more efficient retrieval and a better balance between effectiveness and speed compared to GraphRAG. Similar to GraphRAG, the default settings of LightRAG result in a higher retrieval token count than the Naïve approach. To ensure a fair comparison, we proportionally adjusted its hyperparameters to maintain the number of retrieval tokens within the range of 5K to 10K.

## B.3 NodeRAG Graph Statistics
The table 4 presents the number of each type of node in the indexed graph for each dataset, including entity (N), relationship (R), semantic unit (S), attribute (A), high-level elements (H), high-level overview (O), and text (T). These counts are detailed in the type statistics section. Additionally, the graph statistics provide information on the total number of nodes, the number of non-HNSW edges, HNSW edges, and
```
![Detailed description of the chart](assets/page_0015_img_1.png)
```

### --- Page 0016 ---

```markdown
| Datasets         | Corpus Tokens | Type Statistics | Graph Statistics                          |
|------------------|---------------|-----------------|------------------------------------------|
|                  | T         | S   | N   | R   | A   | O   | H   | Nodes | Non-HNSW Edge | HNSW Edge | Edge     |
|------------------|-----------|-----|-----|-----|-----|-----|-----|-------|----------------|-----------|----------|
| HotpotQA         | 1.93M     | 15905 | 85863 | 5678 | 644 | 449 | 479 | 176203 | 283543        | 487731   | 759172   |
| MuSiQue          | 1.84M     | 1907  | 18714 | 49904 | 6194 | 705 | 570 | 199292 | 316029        | 583126   | 888896   |
| MultiHop-RAG     | 1.41M     | 1532  | 10968 | 43164 | 29628 | 625 | 289 | 90144 | 171410        | 201319   | 367165   |
| Arena-FAQ        | 1.65M     | 1821  | 9027  | 23472 | 27428 | 610 | 1714 | 73405 | 143916        | 154109   | 295165   |
| Arena-Lifestyle   | 1.64M     | 1794  | 9400  | 34694 | 72958 | 518 | 2221 | 84861 | 149225        | 174461   | 318750   |
| Arena-Recreation  | 0.93M     | 1003  | 5342  | 26368 | 16913 | 1196 | 1969 | 1960 | 54160         | 93228    | 117915   |
| Arena-Science     | 1.43M     | 1583  | 8310  | 32232 | 23092 | 2515 | 2515 | 70425 | 127179        | 149424   | 276963   |
| Arena-Tech        | 1.72M     | 1910  | 18037 | 37774 | 29651 | 2633 | 2633 | 85885 | 165790        | 193159   | 353033   |
| Arena-Writing     | 1.82M     | 1973  | 1103  | 4223  | 29338 | 4435 | 4435 | 94259 | 149852        | 298565   | 442397   |

Table 4: Comprehensive dataset statistics, detailing corpus size, type statistics (T, S, N, R, A, O, H), and graph statistics. The graph statistics include the number of document compilation nodes, HNSW semantic edges, and total edges. Each value represents a key metric relevant to graph-based document processing and retrieval.

---

B.4 Graph RAG Ablation

| Method         | Accuracy | Avg. Processing Time | Avg. Tokens |
|----------------|----------|----------------------|-------------|
| GraphRAG (default) | 37.14%   | 4.82s                | 10.4k       |
| Graph-L        | 41.71%   | 2.94s                | 6.6k        |
| Graph-G        | 33.14%   | 22.65s               | 1.11M       |

Table 5: Performance Comparison of GraphRAG Variants. Default is the default setting. Local and global represent the local and global modes under unified prompt and hyperparameter settings.

The default setting of GraphRAG, along with its own prompting mechanism, is not standardized for evaluation, as both the number of retrieval tokens and the choice of prompts significantly impact performance. Hence, we introduce a unified prompt and adjust the hyperparameters of GraphRAG to ensure a fair comparison within a specific range. As shown in Table B.4, GraphRAG with our unified prompt achieves higher performance, demonstrating that the original prompting strategy is not optimal for this task. This further ensures fairness in comparison, as performance is influenced solely by the quality of the retrieved context. Moreover, the global mode of GraphRAG requires significantly longer processing time and incurs higher computational costs due to the LLM analyzing all community summaries, leading to increased complexity and resource consumption. Additionally, for multi-hop question answering, this approach results in degraded performance. Therefore, we conducted an exploratory ablation study only on the MuSiQue dataset, while for other datasets, we estimated query time and retrieval token statistics based on sampled queries.
```

### --- Page 0017 ---

```markdown
# C  Algorithm details

## C.1  Terminology

| Abbr. | Full Name          | Description                                                                                                                                                       | Function                                   | Example                                                                                                                                                                                                                     |
|-------|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| T     | Text               | Full-text chunks from the original source. It contains rich detailed information, although it integrates a large amount of unrelated semantic information.      | Retrievable; Entry points from vector similarity | "Hinton was awarded the Nobel Prize in 2023 for his groundbreaking contributions to artificial intelligence, particularly in deep learning. His pioneering work on backpropagation laid the foundation for modern neural networks, influencing both academia and industry. The recognition came amid increasing discussions on the ethical implications of AI, with Hinton himself advocating for responsible AI development and regulation." |
| S     | Semantic Unit      | Local summaries that are independent and meaningful events summarized from text chunks. They serve as a middle layer between text chunks and entities, acting as the basic units for graph augmentation and semantic analysis. | Retrievable; Entry points from vector similarity | "Hinton was awarded the Nobel Prize for inventing backpropagation."                                                                                                                                                       |
| A     | Attribute          | Attributes of key entities, derived from relationships and semantic units around important entities.                                                              | Retrievable; Entry points from vector similarity | "Geoffrey Hinton, often referred to as the 'Godfather of Deep Learning,' is a pioneer in the field of artificial intelligence. In 2024, he was awarded the Nobel Prize for his contributions to AI and deep learning."         |
| H     | High-Level Element  | Insights summarizing graph communities. Encapsulates core information or any high-level ideas from a community.                                                | Retrievable; Entry points from vector similarity | "Due to the increasing importance of AI, the Nobel Prize is awarded to scholars who have made tremendous contributions to the field of AI."                                                                                 |
| O     | High-Level Overview | Titles or keywords summarizing high-level elements.                                                                                                             | Non-Retrievable; Entry points from accurate search | "AI significance"                                                                                                                                                                                                          |
| R     | Relationship       | Connections between entities represented as nodes. Acts as connector nodes and secondary retrievable node.                                                       | Retrievable; Non-Entry points              | "Hinton received the Nobel Prize."                                                                                                                                                                                        |
| N     | Entity             | Named entities such as people, places, or concepts.                                                                                                            | Non-Retrievable; Entry points from accurate search | "Hinton," "Nobel Prize"                                                                                                                                                                                                    |

Table 6: Node Types in the heterograph

## C.2  K-core & Betweenness centrality

In this subsection, we present the methodology for identifying important entities and generating their attribute summaries, ensuring alignment with the mathematical framework established in the main text.

The selection of important entities, denoted as $N^*$, is based on two fundamental structural graph metrics: $K$-core decomposition and betweenness centrality. These metrics collectively ensure that the selected
```

### --- Page 0018 ---

```markdown
nodes are not only structurally integral but also play a pivotal role in facilitating information flow.

The K-core decomposition, denoted as $K(G^1)$, identifies nodes within densely connected subgraphs, ensuring that selected entities contribute significantly to the structural cohesion of the graph. Meanwhile, betweenness centrality, denoted as $B(G^1)$, highlights nodes that serve as critical intermediaries between different regions of the graph, capturing entities essential for information dissemination.

The process of identifying important entities follows the steps outlined in Algorithm 1.

## Algorithm 1 Identification of Important Entities

**Input:** Graph $G^1 = (V, E)$  
**Output:** Important entity set $N^*$  

**Step 1: Compute K-core decomposition**  
Compute the core threshold:

$$
k_{\text{default}} = \left[ |\log(|V|)| \times \left( \frac{\sum_{v \in V} \text{deg}(v)}{|V|} \right)^{1/2} \right]
$$

Extract the $K$-core subgraph:

$$
K(G^1) = \{v \in V \ | \ \text{deg}_{G^1}(v) \geq k_{\text{default}} \}
$$

**Step 2: Compute betweenness centrality**  
for each $v \in V$ do  
Approximate betweenness centrality using shortest-path sampling:

$$
b(v) = \text{betweenness\_centrality}(G^1, k = 10)
$$

end for  

Compute the average betweenness centrality:

$$
\bar{b} = \frac{\sum_{v \in V} b(v)}{|V|}
$$

Compute the scale factor:

$$
\text{scale} = |\log_{10}(|V|)|
$$

**Step 3: Select important nodes**  
for each $v \in V$ do  
if $b(v) > \bar{b} \times \text{scale}$ then  
Add $v$ to $B(G^1)$  
end if  
end for  

Compute the final set of important entities:

$$
N^* = K(G^1) \cup B(G^1)
$$

**Return** $N^*$

### C.3 Semantic Matching within Community

To establish meaningful semantic relationships among high-level element nodes, we propose the Semantic Matching within Community algorithm. This algorithm ensures that entities with strong semantic similarities are connected within their respective communities. The motivation behind this approach is
```


### --- Page 0019 ---

```markdown
# Algorithm 2 Semantic Matching within Community

**Input:** Graph $G = (V, E)$, node embeddings $\Phi(V)$, community partition $\{C_n\}$  
**Output:** Semantic edges $\hat{E}$  

**Step 1:** Select high-level element nodes  
Extract nodes with labels $S, A$, or $H$:

$$
V_{S,A,H} = \{v \in V \ | \ \psi(v) \in \{S, A, H\}\}
$$

**Step 2:** Apply K-means clustering to node embeddings  
Set number of clusters:

$$
K = \sqrt{|V_{S,A,H}|}
$$

Perform K-means clustering on $V_{S,A,H}$, obtaining clusters $\{S_k\}$  

**Step 3:** Establish semantic edges within communities  
for each community $C_n$ do  
&nbsp;&nbsp;for each cluster $S_k$ do  
&nbsp;&nbsp;&nbsp;&nbsp;Identify nodes within the community and cluster:

$$
V_{C_n,S_k} = V_{S,A,H} \cap C_n \cap S_k
$$

&nbsp;&nbsp;&nbsp;&nbsp;for each pair $(v, v')$ where $v \in \{S, A\}, v' \in H$ do  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add semantic edge:  
$$
e_h(v, v') \in \hat{E}
$$  
&nbsp;&nbsp;&nbsp;&nbsp;end for  
&nbsp;&nbsp;end for  
end for  
Return $\hat{E}$  

to organically integrate $H$ nodes into the graph structure by establishing connections with semantically related nodes within the same community. Formally, the process is summarized in Algorithm 2.

The algorithm begins by identifying nodes that belong to three specific categories: structure nodes ($S$), attribute nodes ($A$), and high-level nodes ($H$). These nodes are collectively defined as:

$$
V_{S,A,H} = \{v \in V \ | \ \psi(v) \in \{S, A, H\}\}
$$

Since these nodes exhibit inherent semantic relationships, we cluster them based on their embeddings, which capture their contextual meaning. To partition the nodes into semantically similar groups, we apply the K-means clustering algorithm (MacQueen et al., 1967) to the embedding representations of $V_{S,A,H}$.

which balances computational efficiency and granularity. This clustering process results in a partitioning of nodes into $K$ semantic clusters, denoted as $S_k$, where each cluster contains nodes with closely related semantic representations.

After clustering, the algorithm establishes edges between semantically related nodes within the same community. Communities are predefined structural subgroups in the graph, denoted as $C_n$, ensuring that local relationships are preserved. For each community-cluster pair, semantic edges are introduced between nodes in $V_{S,A}$ and nodes in $V_H$. Specifically, for any node pair $(v, v')$, where $v \in V_{S,A}$ and $v' \in V_H$, an edge $e_h(v, v')$ is established if both nodes belong to the same community and the same semantic cluster.
```

### --- Page 0020 ---

```markdown
By integrating semantic matching within community constraints, this algorithm enhances the structural integrity of the graph while maintaining computational feasibility. The choice of K-means clustering efficiently groups nodes with similar semantic properties, while the enforcement of community constraints ensures that edges are only formed between nodes that naturally belong to the same substructure. Consequently, the proposed method balances semantic consistency and graph locality, making it well-suited for applications requiring structured knowledge representation and retrieval.

## C.4 Dual Search

To efficiently locate relevant entry points within the Hetero Graph $G$, we propose the Dual Search algorithm, which integrates exact matching on structured nodes and vector similarity search on rich information nodes. This hybrid approach ensures a balance between precision and recall by leveraging both symbolic and dense representations. The core idea is to utilize exact string matching for well-structured nodes while employing approximate nearest neighbor search for nodes containing rich contextual information. By doing so, the algorithm improves both retrieval accuracy and robustness to query variations.

Given a query, a LLM extracts a set of relevant entities, denoted as $N^q$, and embeds the query into a vector representation $q$. Entry points in the graph are then determined by:

$$
V_{\text{entry}} = \{ v \in V \ | \ \Phi(v, N^q, q) \},
$$

where the condition $\Phi(v, N^q, q)$ determines whether a node qualifies as an entry point:

$$
\Phi(v, N^q, q) = \left\{ \begin{array}{ll}
v \in V_{N,O} \land M(N^q, v), & \\
v \in V_{S,A,H} \land R(q, v, k), &
\end{array} \right.
$$

Here, the exact matching function $M(N^q, v)$ returns true if node $v$ matches one of the extracted entities in $N^q$. This ensures that titles or named nodes such $V_{N,O}$ are retrieved deterministically. Meanwhile, the similarity-ranking function $R(q, v, k)$ applies HNSW, selecting the top-$k$ most similar nodes to $q$ in vector space. This is particularly beneficial for context-rich nodes for $v \in V_{S,A,H}$, which may not contain exact query terms but remain semantically relevant.

One of the major strengths of Dual Search is its ability to mitigate errors from noisy or ambiguous queries. Since $N$ and $O$ nodes serve exclusively as entry points and are not directly retrieved, any errors in exact matching do not propagate into the retrieval results. Furthermore, Personalized PageRank (PPR) is applied to refine retrieval by selecting highly weighted cross nodes—nodes that are both structurally significant and semantically aligned with multiple entry points. This graph-based expansion helps compensate for potential errors in entity extraction and ensures robustness against query variations. Another advantage of this approach is its scalability. By employing HNSW, the algorithm avoids the computational overhead of exhaustive similarity searches, making it efficient even on large-scale graphs.
```


### --- Page 0021 ---

```markdown
# D Case Study

## Question1
**Why isn’t there public transport on Christmas day in the UK?**

### NodeRAG
The absence of public transport on Christmas Day in the UK can be attributed to a combination of historical, cultural, and economic factors. Historically, public transport services were more prevalent on Christmas Day before World War II, as many people worked on that day and required transportation. However, over the years, societal norms have shifted, leading to a significant reduction in demand for public transport services on Christmas Day. In contemporary society, Christmas is widely recognized as a time for family gatherings and celebrations, which diminishes the need for public transport. Many people choose to spend the day at home or with relatives, resulting in a lower demand for transit services. Consequently, transport companies have adapted to this change by suspending services on Christmas Day, allowing transit workers to enjoy the holiday with their families as well. Additionally, the operational costs associated with running public transport services on Christmas Day can be significant. With minimal passenger demand, the financial viability of providing such services becomes questionable. Railways and other transport providers often use this time for maintenance and engineering work, taking advantage of the reduced disruption to carry out necessary improvements.

### GraphRAG
Public transport in the UK is significantly limited on Christmas Day due to a combination of historical, cultural, and operational factors. Traditionally, Christmas Day is a public holiday in the UK, and many businesses, including public transport services, close to allow employees to spend time with their families and celebrate the holiday. This practice is rooted in the cultural significance of Christmas, which emphasizes family gatherings and community celebrations. The limited operation of public transport on Christmas Day is also influenced by the low demand for services. Historically, fewer people travel on this day compared to other holidays or weekends, as many individuals prefer to stay home or visit family. As a result, transport providers often find it economically unfeasible to run full services when the number of passengers is expected to be minimal. In recent years, some transport services have begun to offer limited services on Christmas Day, particularly in major cities like London, where there may be a demand for travel to events or gatherings. However, these services are typically not as extensive as those available on other public holidays or regular days. Overall, the combination of cultural practices, low demand, and operational considerations contributes to the lack of public transport on Christmas Day in the UK.

### LightRAG
In most parts of the world, official or religious holidays are taken very seriously. On such days, employers typically have to pay higher rates to staff who are willing to work, but that effort often doesn’t pay off due to low customer turnout. Families tend to arrange their travel plans by Christmas Eve to avoid the need for transportation on Christmas Day. Christmas avoids unnecessary travel because it’s typically a day filled with home activities, and it allows workers in various roles to have a day off. The lack of public demand and the staff’s resistance contributed to the lack of services on Christmas Day. Christmas Day is also an opportune time to perform engineering work on railways with minimal disruption. The railroad industry is capable of completing extensive projects during this period, thanks to thorough planning and pre-staging.

### LLM Decision
ORDER: {“1”: “GraphRAG”, “2”: “NodeRAG”}  
Both answers provide a comprehensive explanation, but answer 2 adds historical context about public transport before World War II and mentions the use of the holiday for maintenance work, offering slightly more depth.  
ORDER: {“1”: “LightRAG”, “2”: “NodeRAG”}  
Both answers provide a comprehensive explanation, but answer 2 adds historical context about the change in public transport services since World War II, which enriches the explanation. Therefore, answer 2 is slightly better.

| Table 7: Sample Table with Multi-line Text |
|---------------------------------------------|
| 21                                          |
```

### --- Page 0022 ---

```markdown
| Question2 | What are rabbit hides used for in Minecraft? |
|-----------|------------------------------------------------|
| NodeRAG   | In Minecraft, rabbit hides serve a couple of useful purposes, particularly in crafting and resource management. One of the primary uses of rabbit hides is to craft leather. Players can combine four rabbit hides in a 2x2 crafting grid to create one piece of leather. This is especially beneficial in Survival Multiplayer (SMP) scenarios where finding other animals for leather can be challenging. Additionally, rabbits can drop a unique item known as the “rabbit’s foot”, which is an important ingredient for brewing potions. The rabbit’s foot is specifically used in the crafting of the Potion of Leaping, which grants players the ability to jump higher for a limited time. Thus, rabbit hides and their associated drops contribute significantly to crafting and potion-making in the game, enhancing the overall gameplay experience. |
| GraphRAG  | In Minecraft, rabbit hides are primarily used to craft leather. Players can combine four rabbit hides in a 2x2 crafting grid to create one piece of leather. This crafting option is particularly useful for players who may not have access to cows, which are the traditional source of leather in the game. Additionally, rabbit hides can be beneficial in multiplayer servers that utilize entity clears, as finding animals can be challenging. By using rabbit hides to craft leather, players have an alternative method to obtain this valuable resource, which is essential for crafting various items, including armor and books. |
| LightRAG  | In Minecraft, Rabbit Hides are primarily used as a crafting resource. They can be combined with other materials to create leather, which is an essential item in the game. Leather is used for crafting various items, including armor (like leather armor), books, item frames, and more. The process of obtaining Rabbit Hides involves hunting rabbits, which can be found in various biomes. Once collected, players can use the hides in their crafting recipes to enhance their gameplay experience through resource management and item creation. This highlights the importance of Rabbit Hides within the broader crafting mechanics of Minecraft. |
| LLM Decision | ORDER: {“1”: “TGRAG”, “2”: “GraphRAG”} Both answers are truthful, but answer 1 provides more detailed information about the uses of rabbit hides, including the additional use of rabbit’s foot for potion-making. Therefore, answer 1 is better. <br> ORDER: {“1”: “TGRAG”, “2”: “LightRAG”} Both answers are truthful, but answer 1 provides more detailed information about the uses of rabbit hides, including the crafting of leather and the rabbit’s foot for potions. Therefore, answer 1 is better. |

| Table 8: Sample Table with Multi-line Text | 22 |
```

### --- Page 0023 ---

```markdown
# E Overview Prompting in NodeRAG

## Response Prompt

### ---Role---
You are a thorough assistant responding to questions based on retrieved information.

### ---Goal---
1. Provide a clear and accurate response. Carefully review and verify the retrieved data, and integrate any relevant necessary knowledge to comprehensively address the user's question.
2. Do not fabricate information. If you are unsure of the answer, just say so.
3. Do not include details not supported by the provided evidence.

### ---Target response length and format---
Multiple Paragraphs

### ---Retrieved Context---
{info}

### ---Query---
{query}

---

## Community Summary Prompt

### --Goal--
Generate a concise summary of the given entity, capturing its essential attributes and important relevant relationships.

### --Requirement--
1. The summary should read like a character sketch in a novel or a product description, providing an engaging yet precise overview.
2. Ensure the output only includes the summary of the entity without any additional explanations or metadata.
3. The length must not exceed 2000 words but can be shorter if the input material is limited.
4. Focus on distilling the most important insights with a smooth narrative flow, highlighting the entity’s core traits and meaningful connections.

### --Input--
- Entity: {entity}
- Related Semantic Units: {semantic_units}
- Related Relationships: {relationships}
```

### --- Page 0024 ---

```markdown
| **--Goal--**                                                                                     | **--Goal--**                                                                                     |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Please break down the following query into a single list.                                       | You will be given a string containing tuples representing relationships between entities. Your task is to reconstruct each relationship in the following format:  |
| **--Requirement--**                                                                              | **--Requirement--**                                                                              |
| 1. Each item in the list should either be a main entity (such as a key noun or object).        | 1. The format of these relationships is incorrect and needs to be reconstructed.                |
| 2. If you have high confidence about the user’s intent or domain knowledge, you may also include closely related terms. | 2. The correct format should be: 'ENTITY_A,RELATION_TYPE,ENTITY_B', where each tuple contains three elements: two entities and a relationship type. |
| 3. If uncertain, please only extract entities and semantic chunks directly from the query. Please try to reduce the number of common nouns in the list. Ensure all elements are organized within one unified list. | 3. Please ensure the output follows this structure, accurately mapping the entities and relationships provided. |
| **--Input--**                                                                                    | **--Input--**                                                                                    |
| Query: {query}                                                                                  | Incorrect relationships tuple string: {relationship}                                             |
| **Question Decompose Prompt**                                                                    | **Relationship Reconstruction Prompt**                                                           |
```

### --- Page 0025 ---

```markdown
# Text Decomposition Prompt Part 1

## --Goal--

Given a text, segment it into multiple semantic units, each containing detailed descriptions of specific events or activities.

Perform the following tasks:

## --Steps--

1. Provide a summary for each semantic unit while retaining all crucial details relevant to the original context.

2. Extract all entities directly from the original text of each semantic unit, not from the paraphrased. Format each entity name in UPPERCASE. You should extract all entities including times, locations, people, organizations and all kinds of entities.

3. From the entities extracted in Step 2, list all relationships within the semantic unit and the corresponding original context in the form of string separated by comma : `"ENTITY_A, RELATION_TYPE, ENTITY_B"`. The RELATION_TYPE could be a descriptive sentence, while the entities involved in the relationship must come from the entity names extracted in Step 2. Please make sure the string contains three elements representing two entities and the relationship type.

## --Requirement--

1. Temporal Entities: Represent time entities based on the available details without filling in missing parts. Use specific formats based on what parts of the date or time are mentioned in the text.

2. Each semantic unit should be represented as a dictionary containing three keys: `semantic_unit` (a paraphrased summary of each semantic unit), `entities` (a list of entities extracted directly from the original text of each semantic unit, formatted in UPPERCASE), and `relationships` (a list of extracted relationship strings that contain three elements, where the relationship type is a descriptive sentence). All these dictionaries should be stored in a list to facilitate management and access.

## --Example--

**Text:**
In September 2024, Dr. EMILY ROBERTS traveled to PARIS to attend the INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY. During her visit, she explored partnerships with several EUROPEAN COMPANIES and presented her latest research on solar panel efficiency improvements. Meanwhile, on the other side of the world, her colleague, Dr. JOHN MILLER, was conducting fieldwork in the AMAZON RAINFOREST. He documented several new species and observed the effects of deforestation on the local wildlife. Both scholars' work is essential in their respective fields and contributes significantly to environmental conservation efforts.
```

### --- Page 0026 ---

```markdown
| Output |
|--------|
| [ |
| { |
| semantic\_unit: In September 2024, Dr. Emily Roberts attended the International Conference on Renewable Energy in Paris, where she presented her research on solar panel efficiency improvements and explored partnerships with European companies., |
| entities: ["DR. EMILY ROBERTS", "2024-09", "PARIS", "INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY", "EUROPEAN COMPANIES", "SOLAR PANEL EFFICIENCY"], |
| relationships: [ |
| "DR. EMILY ROBERTS, attended, INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY", |
| "DR. EMILY ROBERTS, explored partnerships with, EUROPEAN COMPANIES", |
| "DR. EMILY ROBERTS, presented research on, SOLAR PANEL EFFICIENCY" |
| ] |
| }, |
| { |
| semantic\_unit: Dr. John Miller conducted fieldwork in the Amazon Rainforest, documenting several new species and observing the effects of deforestation on local wildlife.; |
| entities: ["DR. JOHN MILLER", "AMAZON RAINFOREST", "NEW SPECIES", "DEFORESTATION", "LOCAL WILDLIFE"], |
| relationships: [ |
| "DR. JOHN MILLER, conducted fieldwork in, AMAZON RAINFOREST", |
| "DR. JOHN MILLER, documented, NEW SPECIES", |
| "DR. JOHN MILLER, observed the effects of, DEFORESTATION on LOCAL WILDLIFE" |
| ] |
| }, |
| { |
| semantic\_unit: "The work of both Dr. Emily Roberts and Dr. John Miller is crucial in their respective fields and contributes significantly to environmental conservation efforts.", |
| entities: ["DR. EMILY ROBERTS", "DR. JOHN MILLER", "ENVIRONMENTAL CONSERVATION"], |
| relationships: [ |
| "DR. EMILY ROBERTS, contributes to, ENVIRONMENTAL CONSERVATION", |
| "DR. JOHN MILLER, contributes to, ENVIRONMENTAL CONSERVATION" |
| ] |
| } |
| ] |
| --Real Input-- |
| Text:{text} |
| Text Decomposition Prompt Part 2 |
```


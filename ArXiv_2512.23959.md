# ArXiv 2512.23959

### --- Page 0001 ---

```markdown
# Improving Multi-step RAG with Hypergraph-based Memory for Long-Context Complex Relational Modeling

**Chulum Zhou\*, Chunkang Zhang\*, Guoxin Yu, Fandong Meng\dagger, Jie Zhou, Wai Lam\dagger, Mo Yu\ddagger**  
The Chinese University of Hong Kong\textsuperscript{1}, WeChat AI\textsuperscript{2}  
\{clzhou,wlam\}@se.cuhk.edu.hk, zkang5051@gmail.com, moyumyu@global.tencent.com

---

## ABSTRACT

Multi-step retrieval-augmented generation (RAG) has become a widely adopted strategy for enhancing large language models (LLMs) on tasks that demand global comprehension and intensive reasoning. Many RAG systems incorporate a working memory module to consolidate retrieved information. However, existing memory designs function primarily as passive storage that accumulates isolated facts through deduction. This static nature overlooks the crucial high-order correlations among primitive facts, the compositions of which can often provide stronger guidance for subsequent steps. Therefore, their representational strength and impact on multi-step reasoning and knowledge evolution are limited, resulting in fragmented reasoning and weak global sense-making capacity in extended contexts. We introduce HGEMM, a hypergraph-based memory mechanism that extends the concept of memory beyond simple storage into a dynamic, expressive structure for complex reasoning and global understanding. In our approach, memory is represented as a hypergraph whose hyperedges correspond to distinct memory units, enabling the progressive formation of higher-order interactions within memory. This mechanism connects facts and thoughts around the focal problem, evolving into an integrated and situated knowledge structure that provides strong propositions for deeper reasoning in subsequent steps. We evaluate HGEMM on several challenging datasets designed for global sense-making. Extensive experiments and in-depth analyses show that our method consistently improves multi-step RAG and substantially outperforms strong baseline systems across diverse tasks.¹

---

## 1 INTRODUCTION

Single-step retrieval-augmented generation (RAG) often proves insufficient for resolving complex queries within long contexts (Trivedi et al., 2023; Shao et al., 2023; Cheng et al., 2025), motivating the shift toward multi-step RAG methods that iteratively interleave retrieval with reasoning. To effectively capture dependencies across steps and condense the lengthy processing history, many approaches incorporate working memory mechanisms inspired by human cognition (Lee et al., 2024; Zhong et al., 2024). However, current memory-enhanced multi-step RAG methods still face challenges in complex relational modeling, especially for resolving global sense-making tasks over long contexts.

During multi-step RAG execution, a straightforward implementation of working memory mechanisms is to let a large language model (LLM) summarize the interaction history into a plaintext description of current problem-solving state. This strategy has been widely adopted since early studies (Liu et al., 2023; Trivedi et al., 2023) as well as in commercial systems (Jones, 2025; Shen & Yang, 2025). Nonetheless, such unstructured memory mechanisms cannot be manipulated with sufficient flexibility.

---

\*Equal contribution.  
\dagger Co-corresponding authors.  
\ddagger We release our code at [https://github.com/Encycelom/HGEM](https://github.com/Encycelom/HGEM)
```

### --- Page 0002 ---

```markdown
# Preprint

## 1 INTRODUCTION

Memory mechanisms in multi-step RAG systems have incorporated reflections to integrate available information for subsequent decisions. This reflects a simple form of memory. With the development of structured indexing for RAG, working memory also borrows this idea. Prevalent studies (Li et al., 2023; 2025a; Shen & Yang, 2025; Chikhara et al., 2025; Xu et al., 2025) examine behavior, such as task decomposing, execution tracking, and result verification, to manage task context more effectively, representing a step toward explicit working memory for complex multi-agent coordination. This also matured in chain-of-thought (CoT) and multi-round RAG, where working memory is represented as iteratively updated records of reasoning steps or retrieved evidence. For example, IRCOT (Trivedi et al., 2023) and ComoRAG (Wang et al., 2025) employ a dynamic memory workspace to iteratively consolidate past knowledge or steps and incorporate new evidence, supporting scalable and iterative reasoning across multiple steps.

Some studies take a step further to adopt a graph-structured working memory to enhance multi-step RAG (Liu et al., 2024; Li et al., 2025a). ERA-CoT (Liu et al., 2024) aids LLMs in understanding context through a series of pre-defined reasoning subsets performing entity-relationship analysis.

## 2 RELATED WORK

### 2.1 WORKING MEMORY MECHANISMS FOR MULTI-STEP RAG

Accuracy across steps often loses the ability to back-trace references to retrieved texts. Consequently, recent research has shifted toward structured or semi-structured working memory, typically with predefined schemas such as relational tables (Lu et al., 2023), knowledge graphs (Oguz et al., 2022; Xu et al., 2025), or event-centric bullet points (Wang et al., 2025).

However, existing memory mechanisms often treat memory as static storage that continually accumulates meaningful but primitive facts. This view overlooks the evolving nature of human working memory, which incrementally incorporates higher-order correlations from previously memorized content. This capacity is particularly crucial for resolving global sense-making tasks that involve complex relational modeling over long contexts. In such scenarios, the required knowledge for tackling a query is often composed of complex structures that extend beyond predefined schemas, and reasoning over long lists of primitive facts is both inefficient and prone to confusion with mixed or irrelevant information. Current memory mechanisms in multi-step RAG systems lack these abilities, preventing memory from effectively guiding LLMs' interaction with external data sources. These limitations highlight the need for a working memory with stronger representational capacity.

In this paper, we propose a hypergraph-based memory mechanism (HGMem) for multi-step RAG systems, which enables memory to evolve into more expressive structures that support complex relational modeling to enhance LLMs' understanding over long contexts. Hypergraphs, as a generalization of graphs, are particularly well-suited for this purpose (Feng et al., 2019). In our design, memory is structured as a hypergraph composed of hyperedges, each treated as a distinct memory point that represents a specific perspective of the memorized information. Initially, these memory points encode low-order primitive facts. As the LLM interacts with external environments, higher-order correlations among memory points gradually emerge and are progressively integrated into the memory through update, insertion, and merging operations. At each step before response generation, the LLM can structure the memory and generate subqueries, enabling adaptive memory-based evidence retrieval for both focused local investigation and broad global exploration.

This rich and structured memory facilitates broader contextual awareness and reasoning in real-world applications by offering several advantages. First, it maintains an integrated body of knowledge and the local problem by synthesizing primitive evidence and intermediate thoughts, typically using both predefined schemas and providing a global perspective over the evidence. Second, it offers structured and accurate guidance for the LLM's sustained interactions in two ways: (1) enabling subsequent reasoning to start from representational propositions rather than from a long list of disparate primitive facts; and (2) leveraging the topological structure of hypergraph to guide subquery generation and evidence retrieval in a more accurate manner.

We conduct extensive experiments on several challenging tasks involving global sense-making questions within long contexts. The results show that our HGMem achieves significant improvements over competitive RAG baselines, confirming the advantages.
```

### --- Page 0003 ---

```markdown
KnowTrace (Li et al., 2025a) equips LLMs with a graph-based working memory to trace relevant knowledge through multi-step RAG execution. However, the working memories of these graph-enhanced work do not effectively support modeling high-order correlations among multiple entities/relationships as each edge in their graphs can intrinsically describe at most binary relationships. By contrast, due to the high-order nature of hypergraph structure, our HGMEM naturally enables its working memory to evolve into more expressive forms capable of flexibly modeling high-order n-ary (n ≥ 2) relations. This advantage helps to fully unleash the reasoning capability of LLMs for multi-step RAG, especially crucial for resolving global sense-making questions that require complex reasoning and deep understanding over long contexts.

## 2.2 RAG WITH STRUCTURED KNOWLEDGE INDEX

There is a long line of work that studies managing extended corpora through structured knowledge indexing to enhance RAG. Though different from our focus on working memory mechanism, these work can be viewed as building structured (and static) long-term memory before actually tackling user queries, thus are relevant. Specifically, tree-structured methods, such as RAPTOR (Sarthi et al., 2024), T-RAG (Fatehila et al., 2024), and TreeRAG (Tao et al., 2025), organize text chunks or entity hierarchies, enabling multi-level or bidirectional retrieval to enhance context integration. Another line of research focuses on building graph-structured index to flexibly represent knowledge for enhancing RAG systems (Xu et al., 2022; Edge et al., 2024; Guo et al., 2024; Li et al., 2025b). For example, GraphRAG (Edge et al., 2024) and LightRAG (Guo et al., 2024) build entity graphs and community-level summaries, or leverage graph-enhanced indexing for dual-level retrieval, leading to improvements in graph reasoning, retrieval efficiency, and response diversity. CAM (Li et al., 2025b) proposes a constructivist agentic memory that flexibly assimilates and accommodates input texts within a hierarchical graph. Hyper-RAG (Feng et al., 2025), HypergraphRAG (Luo et al., 2025) and ProRAG (Wang, 2025) adopt hypergraph to build structured knowledge index and design retrieval/search algorithms for query resolution. In addition, there are a range of other memory mechanisms, essentially structured knowledge index, that simulate long contexts or dialog histories as prior memory to improve RAG systems. According to the form of memory representation, they can be basically classified as contextual memory (Chen et al., 2023; Gutierrez et al., 2024; Lee et al., 2024; Li et al., 2024b; Gutiérrez et al., 2025) and parametric memory (Qian et al., 2025).

However, these existing studies merely leverage their structured index (or memory) as static storage, which are typically constructed during an offline indexing stage before actually responding to user queries.

## 3 METHODOLOGY

We introduce HGMEM, the hypergraph-based memory mechanism designed to facilitate better contextual awareness and reasoning in multi-step RAG settings with structured data sources, especially for long-context tasks that require complex global sense-making.

### 3.1 PROBLEM FORMULATION

In this work, we consider the kind of tasks for LLMs to resolve a query based on a given document. Besides the plain texts, we assume that the document has been preprocessed into a graph through an offline graph-building stage, where entities and relationships are extracted from the document passage. Formally, let us denote the documents as $D$ segmented into a set of small manageable text chunks $\{d_1, d_2, ..., d_p\}$, and the derived graph as $\mathcal{G}$ composed of nodes $\mathcal{V}_g$ and edges $\mathcal{E}_g$ corresponding to the extracted entities and relationships, respectively. Each node $v \in \mathcal{V}_g$ or edge $e \in \mathcal{E}_g$ is associated with the source text chunks in which its embodied entity/relationship appears, which is recorded during the offline graph construction. Meanwhile, the nodes, edges, and text chunks are embedded into high-dimensional vectors for vector-based retrieval. For resolving the query, LLMs have access to both the document and its derived graph as structured data sources.
```

### --- Page 0004 ---

```markdown
![RAG System at the t-th Interaction Step](assets/page_0004_img_1.png)

## 3.2 MULTI-STEP RAG SYSTEM WITH MEMORY

When dealing with tasks requiring a comprehensive understanding, especially over the long context, RAG systems usually resort to multi-step approaches with an underlying memory mechanism, where retrieval operations are interleaved with intermediate reasoning to support broader contextual awareness.

Given a target query $\hat{q}$, the LLM interactively interacts with $D$ and $G$ while managing a memory $M$ to store relevant information for ultimately resolving $\hat{q}$. During each interaction step $t$, the LLM judges whether the content of the current memory has been sufficient with respect to the target query. If the memory is deemed sufficient, it immediately produces a response. Otherwise, it analyzes current memory and generates several subqueries $Q(t)$ that aim at fetching more information from the external environment to enrich the memory. The prompts for generating subqueries are given in Appendix E.

Let $R_V(Q)$ define the entity retrieval operation fetching the most relevant nodes to a query set $Q$ from a candidate node set $V$ using vector-based matching:

$$
R_V(Q) = \bigcup_{q \in Q} \text{argmax}_{v \in V} \text{sim}(h_q, h_v),
$$

where $n_v$ is the number of retrieved entities per query $q$, $h$ is the vector representation of $q$, $h_v$ is the vector representation of $v$, and $\text{sim}(\cdot, \cdot)$ is the cosine similarity function.

As illustrated in Figure 1 (i), at the $t$-th step, if the LLM proceeds to generate subqueries $Q(t)$ based on current memory $M^{(t-1)}$ maintained until the previous step, it retrieves a set of the most relevant entities $V_{Q(t)} = R_V(Q(t))$ from $V_G$. Then, via graph-based indexing, the relationships and text chunks associated with the entities in $V_{Q(t)}$ are also obtained, represented as $E(V_{Q(t)})$ and $D(V_{Q(t)})$, respectively. Subsequently, the LLM analyzes and consolidates this retrieved information into the memory.
```

### --- Page 0005 ---

```markdown
memory, evolving memory into $M^{(t+1)}$, which can be formalized as

$$
M^{(t+1)} = \text{LLM}(M^{(t)}, V^{(t)}, E(V^{(t)}), D(V^{(t)})).
$$

Note that, at the initial step ($t = 0$), we treat the target query $q$ as a special subquery belonging to $Q^{(0)}$, i.e., $Q^{(0)} = \{q\}$. Further details about the memory storage, subquery generation and the dynamics of memory evolving will be elaborated in Section 3.3, Section 3.4 and Section 3.5, respectively.

### 3.3 HYPERGRAPH-BASED MEMORY STORAGE

When the LLM interacts with the document $D$ and the graph $G$, it continuously consolidates relevant information into the memory storage $\mathcal{M}$, which is modeled as a hypergraph:

$$
\mathcal{M} = (V_{\mathcal{M}}, E_{\mathcal{M}}),
$$

where $V_{\mathcal{M}} = \{v_1, v_2, \ldots\}$ is the vertex set and $E_{\mathcal{M}} = \{e_1, e_2, \ldots\}$ is the hyperedge set. It should be noted that the vertices in $V_{\mathcal{M}}$ are actually equivalent to those nodes in $V_G$, both embodying identified entities. Particularly, $V_{\mathcal{M}}$ is a subset of $V_G$. In our implementation, we ensure that each vertex $v_i \in V_{\mathcal{M}}$ must also exist in $G$. Formally, every vertex $v_i \in V_{\mathcal{M}}$ is represented as

$$
v_i = (S_{v_i}^{emb}, D_{v_i}),
$$

where $S_{v_i}^{emb}$ stands for the information of its embodied entity, including name and description, and $D_{v_i}$ denotes the set of text chunks associated with this vertex $v_i$. Similarly, every hyperedge $e_j \in E_{\mathcal{M}}$ is represented as

$$
\bar{e}_j = (S_{e_j}^{emb}, V_{e_j}),
$$

where $S_{e_j}^{emb}$ characterizes the description of the embodied relationship and $V_{e_j}$ is the set of involved memory points, each of which corresponds to a certain aspect of the information stored in current memory, as shown in Figure 1 (ii). Unlike those binary edges $E_G$ that connect at most two nodes in the external graph, a hyperedge can connect an arbitrary number (two or more) of vertices. In this way, our hypergraph-based memory is capable of flexibly modeling high-order correlation among multiple vertices ($n \geq 2$). As a result, the whole memory as a hypergraph can effectively support complex relational modeling, ensuring expressiveness to enhance LLMs' reasoning.

### 3.4 ADAPTIVE MEMORY-BASED EVIDENCE RETRIEVAL

As described in Section 3.2, at each step of our RAG workflow, with respect to the target query, the LLM determines whether to immediately produce a response or proceed to acquire more information from the external documents $D$ and graph $G$. If current memory $M^{(t)} = (V^{(t)}, E^{(t)})$ is deemed insufficient, the LLM first analyzes $M^{(t)}$ and generates several subqueries $Q^{(t)}$ indicating what to further explore. Specifically, we design an adaptive memory-based evidence retrieval strategy for either local investigation or global exploration with $Q^{(t)}$:

(i) **Local Investigation:** When the LLM plans to more deeply investigate some specific memory points, its generated subqueries are utilized to trigger local evidence retrieval over $G$. Concretely, suppose a $q \in Q^{(t)}$ especially targets inspecting $\bar{e}_j \in E_{\mathcal{M}}$, the nodes corresponding to the vertices $V_{e_j}$, subordinate to $\bar{e}_j$ are used as anchor nodes on $G$. Thereafter, using the operation defined by Equation 1, entity retrieval is conducted within the neighborhood of these anchors, which is formalized as

$$
V_q = R(N(V_{e_j}))(q),
$$

$$
N(V_{e_j}) = \bigcup_{v \in M^{(t)}(v)} N_G(v),
$$

where $N_{M^{(t)}(v)}$ represents the neighboring vertices of $v$ over $M^{(t)}$ and $N_G(v)$ represents the neighboring nodes of $v$ over $G$.
```

### --- Page 0006 ---

```markdown
![An illustration of memory evolving dynamics. Each point is equivalent to a hyperedge in the hypergraph. $M^{(t)}$ evolves into $M^{(t+1)}$ through update, insertion and merging operations.](assets/page_0006_img_1.png)

(ii) Global Exploration: When there are unexplored aspects transcending the scope of current memory, the LLM resorts to generating subqueries for exploring broader information from the external documents and graph, not pertinent to any existing memory point. For a query $q \in Q(t)$, the process of entity retrieval can be written as

$$
V_q = R_{C(M^{(t)})}(q),
$$

$$
C(M^{(t)}) = V_g - V_{M^{(t)}},
$$

where $C(M^{(t)})$ represents the available scope comprised of all nodes except those already existing in the current memory.

Then, as in Section 3.2, the associated relationships $E(V_q)$ and text chunks $D(V_q)$ are obtained via graph-based indexing. Finally, following Equation 2, the LLM evolves its current memory $M^{(t)}$ into $M^{(t+1)}$. Under such a strategy, the RAG system is able to adaptively combine both local investigation and global exploration for more flexible information retrieval during interaction with external sources.

### 3.5 DYNAMIC OF MEMORY EVOLVING

Once a set of subqueries have been generated at the $t$-th step, following Equation 2, the LLM analyzes the retrieved information and consolidates useful content into the current memory $M^{(t)}$, resulting in the evolved memory $M^{(t+1)}$. As shown in Figure 1 (ii), on the basis of hypergraph-based memory storage, the dynamic of memory evolving in our proposed HGEMM involves three types of operations:

- **Update.** According to the retrieved information, if there are certain existing memory points whose descriptions should be modified, the update operation will revise the descriptions of corresponding hyperedges without changing their subordinate entities.
  
- **Insertion.** The insertion operation should be evoked when some content of the retrieved information is suitable to be inserted as additional memory points into the current memory, which creates new hyperedges in the hypergraph.
  
- **Merging.** After insertion and update, the LLM inspects current memory and selectively merges existing memory points that are more suitable to constitute a single semantically/logically cohesive unit. With respect to the target query $\hat{q}$, suppose the memory points $e_i^{(rel)}$, $v_e$ are to be merged into a high-order memory point $\hat{e}^{(rel)} = (e_i^{(rel)}, v_e)$, its description and subordinate vertices are acquired as

$$
e^{(rel)}_{\hat{k}} \leftarrow \text{LLM}(S^{(rel)}_i, S^{(rel)}_e, \hat{q})
$$

$$
v_{\hat{k}} = V_2 \cup V_2^{\prime}.
$$

Then, the newly merged memory point is added into the hyperedge set $E(M^{(t)})$ of the current memory $M^{(t)}$. This merging operation over the hypergraph-based memory builds higher-order correlations among multiple existing memory points, facilitating the resolution of queries that require complex relational modeling with disparate facts.

In this way, besides continuously accumulating primitive facts during the LLM's interactions with external data sources, the memory also gradually evolves into more sophisticated forms, capturing higher-order correlations for complex relational modeling. Figure 2 gives a concrete example illustrating the dynamics of memory evolving.
```

### --- Page 0007 ---

```markdown
## 3.6 MEMORY-ENHANCED RESPONSE GENERATION

When the LLM exceeds its maximum interaction steps or the content in current memory $M^{(t)} = (v^{(t)}_M, e^{(t)}_M)$ has been deemed sufficient, a response is immediately produced according to the information stored in current memory. Concretely, besides the descriptions of all memory points (i.e. $E^{(t)}_M$), the text chunks associated with all the entities $v^{(t)}_M$ in current memory are also provided to the LLM for producing the final response.

## 4 EXPERIMENTAL SETTINGS

### 4.1 DATASETS

We choose generative sense-making question answering (QA) (Edge et al., 2024; Guo et al., 2024) and long narrative understanding (Li et al., 2022a; Xu et al., 2022b; Yu et al., 2025; Kociský et al., 2018; Karpinska et al., 2014; Yen et al., 2025; Zhou et al., 2025) as our evaluation tasks. For generative sense-making QA, similar to the setups used in previous works (Edge et al., 2024; Guo et al., 2024), we retain a portion of long documents with more than 100k tokens from Longbench V2 (Bai et al., 2025). From each retained document, we use GPT-4 to generate several global sense-making queries that satisfy the following requirements: 1) The queries should target the overall understanding of the whole provided documents, instead of only concentrating on several specific phrases or sentence pieces. 2) The queries should require high-level understandings and global reasoning over disparate evidence scattered across the whole paragraph. For long narrative understanding, we use three public benchmarks including NarrativeQA (Kociský et al., 2018), NoCha (Karpinska et al., 2024) and Prelude (Yu et al., 2025). Both tasks require global comprehension and complex sense-making over narrative evidence across long documents. Details about the datasets used in our experiments are given in Appendix A.

### 4.2 IMPLEMENTATION DETAILS

**Offline Graph Construction.** For all the datasets used in our experiments, we first segment every document into text chunks of 200 tokens with 50 overlapping tokens between adjacent chunks. Then, GPT-4 is utilized to preprocess each of the chunked documents into a graph using the open-sourced tool provided by LightRAG (Guo et al., 2024). After building the graph, we adopt bge-m3 (Chen et al., 2022) as the embedding model to convert all the entities, relationships and text chunks into vector representations managed by nano vector database.

**System Development and Configuration.** Our RAG system is comprised of the backbone LLMs and the hypergraph-based memory. We choose GPT-4 and Qwen2.5-32B-Instruct as the representatives of advanced closed-source and open-source LLMs, respectively. During experiments, GPT-4 is accessed through the official API while Qwen2.5-32B-Instruct is locally deployed with VLLM (Kwon et al., 2023). For the configuration of LLM inference, we set the temperature to 0.8 and the maximum number of output tokens to 2,048. As for the hypergraph-based memory, we employ the hypergraph-db package to maintain and manage the hypergraph at runtime. The vector representations of the nodes, hyperedges and associated text chunks in the hypergraph are also generated by bge-m3 embedding model.

### 4.3 BASELINES AND EVALUATION METRICS

In our experiments, we compare our proposed HGEMM to two types of baseline methods, i.e. traditional RAG and multi-step RAG, which utilize plain texts and/or graph-structured data sources. Among these methods, DeepRAG (Guan et al., 2025) and ComoRAG (Wang et al., 2025) are equipped with a working memory, while the others are not. The details of these comparison methods can be found in Appendix B. To ensure fair comparison, all baselines operate on a similar number of retrieved passages. In the case of single-step RAG, this means retrieving the same average number of text chunks as our HGEMM. For multi-step RAG methods, we approximate comparability by constraining them to rewrite the same maximum number of subqueries and perform the same maximum number of steps, while requiring retrieval of the same average number of chunks per step.
```


### --- Page 0008 ---

```markdown
# Preprint

## Table 1: The overall experimental results on four benchmarks. The second column “Working Memory” distinguishes whether the corresponding method is equipped with a working memory that enhances LLMs during RAG execution. The best scores in each dataset are bolded. HGEM consistently outperforms other comparison methods across all datasets.

| Type                | Working Memory | Method      | Longchen | Comprehensiveness | Diversity | Ace (%) | % Acc (%) | Ace (%) |
|---------------------|----------------|-------------|----------|-------------------|-----------|---------|-----------|---------|
| GPT-4o              |                | NaiveRAG   | 61.62    | 64.20             | 52.00     | 67.64   | 60.00     |         |
|                     |                | GraphRAG   | 60.39    | 64.02             | 53.00     | 70.63   | 59.26     |         |
|                     |                | LightRAG   | 61.55    | 63.37             | 44.00     | 71.43   | 61.48     |         |
|                     |                | HippoRAG v2| 58.92    | 61.27             | 34.00     | 72.22   | 54.81     |         |
|                     |                | DeepRAG    | 63.62    | 68.95             | 45.00     | 67.46   | 56.30     |         |
|                     |                | ComorRAG   | 62.18    | 65.82             | 54.00     | 34.37   |         |         |
|                     |                | HGEM       | 68.75    | 69.74             | 55.00     | 73.81   | 62.96     |         |
| Qwen2.5-32B-Instruct|                | NaiveRAG   | 61.41    | 62.25             | 37.00     | 69.67   | 52.59     |         |
|                     |                | GraphRAG   | 60.78    | 62.16             | 44.00     | 62.70   | 50.37     |         |
|                     |                | LightRAG   | 60.82    | 62.73             | 40.00     | 59.82   | 60.74     |         |
|                     |                | HippoRAG v2| 56.66    | 60.80             | 33.00     | 62.25   | 51.85     |         |
|                     |                | DeepRAG    | 61.45    | 63.56             | 44.00     | 66.40   | 51.11     |         |
|                     |                | ComorRAG   | 60.74    | 61.28             | 44.00     | 57.60   | 53.37     |         |
|                     |                | Ours       | 64.18    | 66.51             | 51.00     | 70.63   | 62.22     |         |

For generative sense-making QA, we adopt the following two metrics to assess the qualities of model responses: 1) Comprehensiveness measures how well the model responds comprehensively to queries and addresses all aspects and necessary details with respect to the target query. 2) Diversity indicates how rich and diverse the responses are in providing various perspectives and insights into the query. The grading criteria gives scores ranging from 0 to 100 based on a two-step scoring scheme, as detailed in Appendix F.

For long narrative understanding, including NarrativeQA, Nocha, and Prelude, we uniformly use prediction accuracy (Acc) as the reported metric. Specifically, for NarrativeQA, prior studies (Bualuan et al., 2022; Wang et al., 2024; Zhou et al., 2025) have shown that conventional token-level metrics such as Exact Match and F1 score usually fail to reflect actual semantic equivalence between hypothesis and reference answer, especially for abstractive answers. Therefore, we also apply GPT-4 to judge whether the LLM’s prediction fully entails the reference answer, producing a binary True/False decision.

## 5 RESULTS AND ANALYSIS

### 5.1 OVERALL RESULTS

Table 1 reports the overall results across all evaluation tasks. Our HGEM consistently outperforms both single-step and multi-step RAG baselines on every dataset. Importantly, our HGEM with Qwen2.5-32B-Instruct matches or even outperforms baselines powered by the stronger GPT-4, underscoring its value in resource-efficient scenarios.

The baselines exhibit mixed performance patterns reflecting their respective representational strengths. For instance, HippoRAG v2 relies on knowledge triples, which provide strong fact representation but limited coverage of events and plots. As a result, it performs well on Nocha but falls behind NaiveRAG on NarrativeQA. In contrast, GraphRAG and LightRAG excel at building global representations but are weaker at capturing fine-grained details, leading them to outperform other baselines on Prelude and NarrativeQA. The two multi-step RAG methods, which mainly employ working memory to iteratively generate subqueries in a chaining fashion, struggle with sense-making questions, where integrating higher-order relationships is essential.

In comparison, our HGEM provides strong compositional representations that span from facts to plots, equipping LLM reasoning with high-order correlations and integrated evidence. This enables it to meet the diverse requirements posed by the evaluation tasks.
```

### --- Page 0009 ---

```markdown
![Prediction accuracies at different steps using Qwen2.5-32B-Instruct on long narrative understanding datasets](assets/page_0009_img_1.png)

| Ablation Type       | Method     | Longbench | Comprehensiveness | Diversity | Acc (%) | NarrativeQA | Acc (%) | Nocha | Acc (%) | Prelude | Acc (%) |
|---------------------|------------|-----------|-------------------|-----------|---------|-------------|---------|-------|---------|---------|---------|
| Retrieval Strategy   | HGMEM      |           | 64.18             | 66.51     | 51.00   | 70.63       | 62.22   |       |         |         |         |
|                     | w/. GE Only|           | 59.28             | 61.67     | 47.00   | 68.25       | 59.26   |       |         |         |         |
|                     | w/. LI Only|           | 61.38             | 63.82     | 43.00   | 63.49       | 60.00   |       |         |         |         |
| Memory Evolution     | HGMEM      |           | 64.18             | 66.51     | 51.00   | 70.63       | 62.22   |       |         |         |         |
|                     | w/o. Update|           | 62.48             | 64.92     | 50.00   | 68.25       | 60.00   |       |         |         |         |
|                     | w/o. Merging|          | 61.76             | 61.80     | 43.00   | 61.11       | 57.78   |       |         |         |         |

5.2 PERFORMANCE AT DIFFERENT STEPS

During the execution of our multi-step RAG system, the memory progressively evolves and guides the LLM to proceed with retrieval and reasoning. To inspect the effects of memory evolving over multiple interaction steps, we force the LLM to generate responses at every step for a total of six turns, even if it originally decides to terminate the iteration earlier. Figure 3 presents the performances at different steps using Qwen2.5-32B-Instruct on long narrative understanding tasks. Note that $t=0$ represents the initial step when the target query $q$ is used for retrieval. We can observe that our HGEM achieves its best performance at $t=3$, mostly outperforming NaiveRAG and LightRAG baselines across steps. More steps bring no further improvements at a higher cost.

5.3 ABLATION STUDIES

Evidence Retrieval Strategy. When the LLM determines to acquire more information from $D$ and our HGEM adopts an adaptive memory-based evidence retrieval strategy for either focused local investigation or broad global exploration (Section 3.4). To investigate the effects of such strategy, in Table 2, we compare our strategy to the variants that involve only Local Investigation or Global Exploration, represented as “w/. LI Only” and “w/. GE Only”, respectively. The results show that “w/. LI Only” and “w/. GE Only” significantly underperform the adaptive strategy across all datasets, demonstrating the effectiveness and necessity of adaptively combining the two modes of evidence retrieval.

Effects of Update and Merging Operations. The memory evolving in our HGEM involves update, insertion and merging operations, where merging is especially critical for building higher-order correlations from primitive facts. Because insertion is indispensable, we just carry out ablation experiments on all datasets using Qwen2.5-32B-Instruct to assess the effects of update and merging operations, as shown in Table 2. Compared to the “HGEM”, removing either operation leads to a performance drop, while removing merging (“w/o. Merging”) causes a substantially larger degradation than removing update (“w/o. Update”). It reflects the effectiveness of both operations, especially highlighting the importance of high-order correlations built through merging operations.
```

### --- Page 0010 ---

```markdown
| Query Type   | Method         | NarrativeQA | Avg.-Nₗ | Acc (%) | Nocha       | Avg.-Nₗ | Acc (%) |
|--------------|----------------|-------------|---------|---------|-------------|---------|---------|
| Primitive    | HGEM           |             | 3.35    | 70.00   |             | 3.85    | 55.00   |
|              | w.o. Merging   |             | 3.32    | 70.00   |             | 3.73    | 60.00   |
| Sense-making | HGEM           |             | 7.07    | 40.00   |             | 5.25    | 60.00   |
|              | w.o. Merging   |             | 4.10    | 30.00   |             | 3.74    | 55.00   |

5.4 DISSECTING QUERY RESOLVING: PRIMITIVE VS. SENSE-MAKING

To better understand how our proposed HGEM brings improvement to the evaluation tasks, we conduct a targeted analysis across different query types. Specifically, we randomly sample 40 queries from each long narrative understanding dataset used in our experiments, yielding a total of 120 queries. These are then manually categorized into two representative types:

- **Primitive Query**: Queries that primarily require locating directly associated chunks, which can often be resolved with local evidence and focus on straightforward factual information.

- **Sense-making Query**: Queries that require deeper comprehension by connecting and integrating multiple pieces of evidence, emphasizing the construction of higher-order relationships and interpretation beyond surface retrieval.

We compare both prediction accuracy and the average number of entities per hyperedge ($Avg.-Nₗ$) in memory for generating final responses. The latter serves as a qualitative indicator of relationship complexity. Table 3 shows that on sense-making queries, our full “HGEM” achieves higher accuracy but considerably larger $Avg.-Nₗ$. In contrast, for primitive queries, “HGEM” yields comparable or slightly lower accuracy relative to “HGEM w/o. Merging”. This is likely because the full model still tends to associate additional pieces of relevant evidence (as indicated by the slightly higher $Avg.-Nₗ$), even though the primitive evidence alone is sufficient to answer straightforward queries, resulting in redundancy.

Notably, the $Avg.-Nₗ$ on sense-making queries consistently exceeds that on primitive queries, especially when merging is applied. Taken together, these results indicate that HGEM improves contextual understanding by constructing high-order correlations for complex relational reasoning, rather than relying on shallow accumulation of surface facts.

6 CONCLUSION

In this work, we propose HGEM, the hypergraph-based memory mechanism that aims at improving multi-step RAG by enabling the evolving of memory into more sophisticated forms for complex relational modeling. In HGEM, the memory is structured as a hypergraph composed of a set of hyperedges as separate memory points. HGEM allows the memory to progressively establish higher-order correlations among previously accumulated primitive facts during the execution of multi-step RAG systems, guiding LLMs to organize and connect thoughts for a focal problem. Extensive experiments and in-depth analysis validate the effectiveness of our method over strong RAG baselines on challenging datasets featuring global sense-making questions over long context.

7 REPRODUCIBILITY STATEMENT

To ensure reproducibility, we introduce the usage and statistics of our used datasets in Section 4.1 and Appendix A. We also give the implementation details about the offline graph construction, system deployment and configuration in Section 4.2. Appendix D gives the prompts for updating, inserting and merging memory points for memory evolving during multi-step RAG execution. Appendix E describes the procedures for scoring model responses in the generative sense-making QA task.
```


### --- Page 0011 ---

```markdown
# References

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazhen Xu, Lei Hou, Yuyao Dong, Jie Tang, and Juanzi Li. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In *Proceedings of Association for Computational Linguistics*, pp. 3639–3664, 2025.

Jannis Bullain, Christian Buck, Wojciech Gajewski, Benjamin Börschinger, and Tal Schuster. Tomayo, tomahot. beyond token-level answer equivalence for question answering evaluation. CoRR, abs/2202.07654, 2022.

Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. Walking down the memory lane: Beyond context limit through interactive reading. CoRR, abs/2310.05029, 2023.

Jianyi Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. BGE m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. CoRR, abs/2402.03216, 2024.

Mingyue Cheng, Yucong Liu, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, Daoyu Wang, and Enhong Chen. A survey on knowledge-oriented retrieval-augmented generation. CoRR, abs/2503.10677, 2025.

Prateek Chikhara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready AI agents with scalable long-term memory. CoRR, abs/2504.19143, 2025.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph RAG approach to query-focused summarization. CoRR, abs/2404.16130, 2024.

Masoomeh Fatehika, Ji Kim Lucas, and Sanjay Chawla. T-RAG: lessons from the LLM trenches. CoRR, abs/2402.07483, 2024.

Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. Hypergraph neural networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pp. 3558–3565, 2019.

Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shihui Ying, Shaoyi Du, Han Hu, and Yue Gao. Hyper-rag: Combating LLM hallucinations using hypergraph-driven retrieval-augmented generation. CoRR, abs/2504.08758, 2025.

Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and Jie Zhou. Deeprag: Thinking to retrieve step by step for large language models. CoRR, abs/2502.01142, 2025.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation. CoRR, abs/2410.05779, 2024.

Bernal Jiménez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. In *Proceedings of Neural Information Processing Systems*, 2024.

Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From RAG to memory: Non-parametric continual learning for large language models. CoRR, abs/2502.14802, 2025.

Nicola Jones. Openai’s deep research tool: is it useful for scientists? *Nature*, 2025.

Mareza Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, and Mohit Iyer. One thousand and one pairs: A "novel" challenge for long-context language models. In *Proceedings of EMNLP*, pp. 1704–1708, 2024.

Tomáš Kociský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. The narrativeqa reading comprehension challenge. *Transactions of the Association for Computational Linguistics*, 6:317–328, 2018.
```

### --- Page 0012 ---

```markdown
| Author(s)                                                                 | Title                                                                                                         | Source                                                                                                   |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica | Efficient memory management for large language model serving with pagedattention.                             | In Proceedings of the Symposium on Operating Systems Principles, pp. 611–626, 2023.                     |
| Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John F. Canny, and Ian Fischer | A human-inspired reading agent with gist memory of very long contexts.                                       | In Proceedings of International Conference on Machine Learning, 2024.                                   |
| Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem | CAMEL: communicative agents for "mind" exploration of large language model society.                          | In Proceedings of Neural Information Processing Systems, 2023.                                          |
| Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang                   | Loogle: Can long-context language models understand long contexts?                                           | In Proceedings of Association for Computational Linguistics, pp. 16304–16333, 2022a.                    |
| Rui Li, Quanyu Dai, Zeyu Zhang, Xu Chen, Zhenhua Dong, and Ji-Rong Wen   | Knowtrace: Bootstrapping iterative retrieval-augmented generation with structured knowledge tracing.         | CoRR, abs/2505.2025, 2022a.                                                                             |
| Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu, Yangguang Li, Wanli Ouyang, Wenbo Su, and Bo Zheng | Grapherhead: Building graph-based agent to enhance long-context abilities of large language models.          | In Findings of Empirical Methods in Natural Language Processing, pp. 12758–12786, 2024b.               |
| Yanming Liu, Xinyue Peng, Tianyu Du, Jianwei Yin, Weihao Liu, and Xuhong Zhang | Era-cot: Improving chain-of-thought through entity relationship analysis.                                     | In Proceedings of Association for Computational Linguistics, pp. 8780–8794, 2022.                      |
| Junru Liu, Siyu An, Mingbao Lin, Gabrielle Pergola, Yulan He, Di Yin, X ing Sun, and Yunsheng Wu | MemoChat: Tuning llms to use memos for consistent long-range open-domain conversation.                       | CoRR, abs/2308.08239, 2023.                                                                             |
| Haorun Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Ze-min Kuang, Meina Song, Yifan Zhu, and Luu Anh Tuan | Hypergraph: Retrieval-augmented generation with hypergraph-structured knowledge representation.              | CoRR, abs/2503.21322, 2025.                                                                             |
| Barlas Oğuz, Xilun Chen, Vladimir Karpukhin, Stan Pesthreltev, Dmytro Okhonko, Michael Seppälä, Shlomo Gupta, Yashar Mehdad, and Scott Yih | Unik-qa: Unified representations of structured and unstructured knowledge for open-domain question answering. | In Findings of the Association for Computational Linguistics: NAACL, pp. 1535–1546, 2022.               |
| Hongjin Qian, Zheng Liu, Peitan Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Zhang | Memora: Boosting long context processing with global memory-enhanced retrieval augmentation.                  | In Proceedings of WWW 2025, pp. 2366–2377, 2025.                                                       |
| Parth Sarathi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Mann | RAPTOR: recursive abstractive processing for tree-organized retrieval.                                       | In Proceedings of International Conference on Learning Representations, 2024.                           |
| Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen | Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy.              | In Findings of EMNLP, pp. 9248–9274. Association for Computational Linguistics, 2023.                   |
| Minjie Shen and Qikai Yang                                               | From mind to machine: The rise of manus AI as a fully autonomous digital agent.                             | CoRR, abs/2505.02024, 2025.                                                                             |
```

### --- Page 0013 ---

```markdown
| Authors                                                                 | Title                                                                                                      | Source                                                                                      |
|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Wenyu Tao, Xiaofen Xing, Yirong Chen, Linyi Huang, and Xiangmin Xu.    | Treerag: Unleashing the power of hierarchical storage for enhanced knowledge retrieval in long documents.  | In Findings of the Association for Computational Linguistics, pp. 356–371, 2025.          |
| Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. | Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.       | In Proceedings of the Association for Computational Linguistics, pp. 10014–10037, 2023.  |
| Jingjin Wang.                                                           | Proprag: Guiding retrieval with beam search over proposition paths.                                       | CoRR, abs/2504.18070, 2025.                                                                |
| Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang, Mo Yu, Jie Zhou, Jin Xu, and Liyan Xu. | Comorag: A cognitive-inspired memory-organized RAG for stateful long narrative reasoning.                  | CoRR, abs/2508.10149, 2025.                                                                |
| Yang Wang, Alberto Garcia Hernandez, Roman Kyslyi, and Nicholas Kersing. | Evaluating quality of answers for retrieval-augmented generation: A strong LLM is all you need.           | CoRR, abs/2406.18064, 2025.                                                                |
| Liyan Xu, Jianan Li, Mo Yu, and Jie Zhou.                              | Fine-grained modeling of narrative context: A coherence perspective via retrospective questions.           | In Proceedings of the Association for Computational Linguistics, pp. 5822–5838, 2024a.    |
| Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. | A-MEM: agentic memory for LLM agents.                                                                      | CoRR, abs/2502.12110, 2025.                                                                |
| Zhe Xu, Jiasheng Ye, Xiangyang Liu, Tianxing Sun, Xiaoran Liu, Qipeng Guo, Linlin Li, Qun Liu, Xuanjing Huang, and Xipeng Qiu. | Detectiveqa: Evaluating long-context reasoning on detective novels.                                       | CoRR, abs/2409.02465, 2024b.                                                                |
| Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. | React: Synergizing reasoning and acting in language models.                                               | In Proceedings of International Conference on Learning Representations, 2023.              |
| Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblatt, and Danqi Chen. | HELMET: how to evaluate long-context models effectively and thoroughly.                                   | In The Thirteenth International Conference on Learning Representations, 2025.             |
| Mo Yu, Tsz Ting Chung, Chulun Zhou, Tong Li, Rui Lu, Jiangnan Li, Liyan Xu, Haoshu Lu, Ning Zhang, Jing Li, and Jie Zhou. | PRELUDE: A benchmark designed to require global comprehension and reasoning over long contexts.           | CoRR, abs/2508.09848, 2025.                                                                |
| Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang.        | Memorybank: Enhancing large language models with long-term memory.                                       | In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 19724–19731, 2024.  |
| Chulun Zhou, Qiujing Wang, Mo Yu, Xiaoqian Yue, Rui Lu, Jiangnan Li, Yifan Zhou, Shunchi Zhang, Jie Zhou, and Wai Lam. | The essence of contextual understanding in theory of mind: A study on question answering with story characters. | In Proceedings of the Association for Computational Linguistics, pp. 22612–22631, 2025.  |
```

### --- Page 0014 ---

```markdown
| Longbench (Financial) | Longbench (Governmental) | Longbench (Legal) | NarrativeQA | NoCha | Prelude |
|-----------------------|--------------------------|--------------------|-------------|-------|---------|
| #Documents            | 20                       | 22                 | 7           | 10    | 4       | 5       |
| Avg. #Tokens          | 266k                     | 256k               | 194k        | 218k  | 139k   | 280k    |
| #Queries              | 100                     | 98                  | 55          | 100   | 126    | 135     |

## A DATASET STATISTICS

### Generative Sense-making QA
We retain a portion of long documents with more than 100k tokens from Longbench V2 (Bai et al., 2025), which was originally comprised of six major task categories designed to assess the ability of LLMs to handle long-context problems. In our experiments, we select three domains of documents from the category of single-document QA, including Financial, Governmental and Legal.

### Long Narrative Understanding
We use the following public benchmarks:

- **NarrativeQA** (Kociský et al., 2018): It is one of the most widely used benchmarks for story question answering. Because of its question construction strategy over high-level book summaries, the task places greater emphasis on synthesis and inference beyond local texts. In contrast, many other existing long-context QA tasks can often be solved with only local evidence, as shown by studies in (Yu et al., 2025). For evaluation, we randomly sample 10 long books exceeding 100k tokens, together with their associated queries, from the complete benchmark.

- **NoCha** (Karpinska et al., 2024): The task involves discriminating minimally different pairs of true and false claims about English fictional books. Although the format may appear different from sense-making questions, NoCha is explicitly designed to require reconstructing a global understanding of the book in relation to the focal statement. Since the official test set is hidden, we conduct experiments using only the publicly released subset.

- **Prelude** (Yu et al., 2025): This benchmark assesses LLMs’ global comprehension and deep reasoning by requiring them to determine whether a character’s proposal story is consistent with the original book. Most instances of this task demand integrating multiple pieces of evidence or even forming a holistic impression of the character’s storyline. In our experiments, we use all English books included in Prelude for evaluation.

Table 4 gives the detailed statistics about the data used in our experiments, including the number of documents, average tokens per document and the total number of queries. Generative sense-making QA tasks involves documents from Longbench V2 benchmark in Financial, Governmental and Legal domains. Long narrative understanding task uses NarrativeQA, NoCha and Prelude benchmarks.

## B COMPARISON BASELINES

In our experiments, we compare our methods to traditional RAG and Multi-step RAG methods. Traditional RAG includes:

- **NaiveRAG** just uses the target query to retrieve a set of text chunks from the document for dealing with queries.

- **GraphRAG** (Edge et al., 2024) constructs knowledge graph from plain-text documents and build a hierarchy of communities of closely related entities before using an LLM to make responses.

- **LightRAG** (Guo et al., 2024) also builds a graph structure and employs a dual-level retrieval strategy from both low-level and high-level evidence discovery.

- **HippoRAG v2** (Gutiérrez et al., 2025) creates a knowledge graph and adopts the Personalized PageRank algorithm with dense-sparse integration of passages into the graph search process for resolving queries.
```

### --- Page 0015 ---

```markdown
# Preprint

You are an intelligent assistant responsible for resolving the [Main Query] through analyzing supportive information from external knowledge sources and making necessary treatments. Your current mission involves enabling the memory points describing what you have learned with respect to the [Main Query].

At present, in order to ultimately resolve the [Main Query], one or several subsidiary behaviors have been inserted, as shown in [Figure Subsampled]. Correspondingly, the memory points originally created and revised for the [Main Query] are shown in the next table.

Given the identified need, your task is to extract useful information concerning the updating of the [Main Query] and to merge memory points while avoiding the existing memory points.

1. **Input**: A language to be used for the retrieval of desired knowledge from the [desired] web linked with the [Main Query]. Then reorganize your [Memory] using one or more of the following methods:
   - **Update Existing Memory Points**: The update operation should be used when some aspects of the identified information are suitable to be retained by your memory or as multiple additional memory points to consider during your next query.
   - **Insert New Memory Points**: The insertion operation should be used when new aspects of the identified information are suitable to be added to your memory.

2. **Output**: A prompt to be used for the retrieval of desired and updated memory points using the “Example of Anticipated Output Format”:
   - If the memory points contain information that needs to be merged, you must provide the merged memory points in [desired Memory Format].
   - When finished, output completion delimiters.

3. **Memory**:
   ```plaintext
   [Memory]:
   [desired]
   [inserted]
   [updated]
   ```

4. **The called useful information that does not exclude the query-specific knowledge or potentially brings insights to better deal with the [Main Query]**:
   - An auxiliary context, which could provide additional insights and arguments to enhance the knowledge of the [Main Query], should be included.
   - A memory point can include multiple associated objects that describe how to deal with relationships among multiple interconnected entities.

5. **A memory point for updating existing memory points is as follows**:
   - If necessary, you can introduce new terms linked to the [Main Query] explicitly.

---

![The prompt for updating and inserting memory points during memory evolving in HG-MEM.](assets/page_0015_img_1.png)

---

![The prompt for merging memory points during memory evolving in HGMEM.](assets/page_0015_img_2.png)

---
```

### --- Page 0016 ---

```markdown
# Preprint

## Table 5: Statistics of the cost of online multi-step RAG execution in our HGMem and other baselines with working memory. Avg-Token is the average count of tokens processed by LLMs per question, while Avg-Time stands for the average inference latency per question.

| Method    | NarrativeQA | Nocha     | Prelude   |
|-----------|-------------|-----------|-----------|
|           | Avg-Token   | Avg-Time  | Avg-Token | Avg-Time  |
| HGMem     | 4346.43     | 15.84     | 5252.73   | 18.76     |
| w/o_Merging | 4154.02   | 14.84     | 4750.32   | 16.97     |
| DeepRAG   | 3904.18     | 13.94     | 4724.07   | 16.87     |
| ComoRAG   | 5083.26     | 18.15     | 5503.98   | 19.66     |

Multi-step RAG includes:

- **DeepRAG** (Guan et al., 2025) conducts multi-step reasoning as a Markov Decision Process by iteratively decomposing queries.
- **ComoRAG** (Wang et al., 2025) undergoes multi-step interactions with external data sources with a dynamic memory workspace, iteratively generating probing queries and integrating the retrieved evidence into a global memory pool.

## C COMPARISON

We conduct a cost comparison between our HGMem and other baselines with working memory in terms of token consumption and inference latency. Note that the cost of online multi-step RAG execution is the real concern for fair comparison because the offline graph construction is just for building query-agnostic indexing structure. With this focus, we measure the average token consumption and inference latency of HGMem, ComoRAG and DeepRAG in Table 5. From the statistics, we can observe that the cost of our HGMem is basically of the same level with those of DeepRAG and ComoRAG while consistently achieving better performance. We can also see that the merging operation, which is the core operation for forming high-order correlation in our HGMem, introduces minor computational overhead.

## D PROMPTS FOR MEMORY EVOLVING

Section 3.5 describes the dynamics of memory evolving in HGMem, which consists of update, insertion and merging operations. The prompts for these three types of operations are given in Figure 4 and Figure 5.

## E PROMPTS FOR SUBQUERY GENERATION

During our multi-step RAG execution, the LLM needs to generate subqueries for acquiring information from external data sources. First, it raises relevant concerns that either target specific memory points or aim at probing useful information outside current memory. Then, the LLM generates corresponding subqueries according to the raised concerns. The prompts for raising concerns and generating subqueries are given in Figure 6 and Figure 7, respectively.

## F EVALUATION PROMPTS FOR GENERATIVE SENSE-MAKING QA

For the evaluation of generative sense-making QA, we leverage GPT-4 as an evaluator to assess the quality of model responses. Given the target query and the source paragraph from which the query originated, the GPT-4 evaluator first indicates the level of comprehensiveness/diversity and then gives a final score within the value range of the corresponding level. Detailed prompts for such LMas-a-Judge evaluation. Figure 8 and Figure 9 give the prompts for scoring the comprehensiveness and diversity, respectively.
```

### --- Page 0017 ---

```markdown
# Preprint

You are an intelligent assistant responsible for dealing with the [Main Query] by making appropriate operations as specified. 

With respect to the [Main Query], you have consolidated some memory points in your [Memory] (describing what you have already known regarding the [Main Query]). Each memory point can be a specific aspect related to the Main Query, providing necessary details or insights from its perspective.

## Goal:
Your task is to analyze the [Main Query] and [Memory] to determine whether current [Memory] has been sufficient to comprehensively resolve the [Main Query]. If not, you should indicate what you want to further investigate.

### Procedures:
1. Make a rigorous judgment following the logic branches below.
   - Case 1: If the [Memory] has been sufficient to completely resolve the [Main Query], output <loops in [Census]>.
   - Case 2: If the [Memory] is not sufficient, determine which situation should be attributed to each of the following subsections.
   - Case 3: There are incomplete aspects that brought up the counter [Memory] (i.e., not related to any of the existing memory points).

#### Step 2: **Example of Anticipated Output Format**
Specifically, if your judgment is implemented using corresponding case index (1, 2.1 or 2.2):
- This example will cover the case, and an explicit check of each aspect defined by current [Memory] to better resolve the [Main Query] 
  (where case 2.2 is given to input _concern_ concerns, each of which targets a specific memory point. For each concern, specify the index of its corresponding memory).

```markdown
[Implementation]
1. Example of Anticipated Output Format for Case 2.1: 
   ######################
   [Main Query]: 
   [Concern]: 
   [Subquery_1]; 
   [Subquery_2]; 
   [Subquery_3]; 
   ######################
```

2. Example of Anticipated Output Format for Case 2.2:
   ######################
   [Main Query]: 
   [Memory]: 
   [Concern]: 
   [Previous Subqueries]; 
   [History_subqueries]; 
   ######################

#### [Main Query]:
[Memory]:
[Concern]:
[Previous Subqueries]:
[History_subqueries]:

### Output:
```

![The prompt for raising concerns either targeting at specific memory points or probing useful information outside the current memory.](assets/page_0017_img_1.png)

You are an assistant responsible for dealing with the [Main Query]. Although you have had some relevant information in your [Memory], your current [Memory] is still not sufficient to comprehensively resolve the [Main Query] due to the concern given in [Concern]. Therefore, you need to generate a subquery that aims at either retrieving more evidences or investigating unexplored aspects in [Subquery] to better deal with the [Main Query] ultimately.

[Previous Subqueries] records a series of previous subqueries that have been raised before.

```markdown
#################### Anticipated Output Format ####################
[Subquery]: xxx
######################
[Main Query]: [query]
[Memory]: 
[Concern]: 
[Previous Subqueries]: 
[History_subqueries]:
######################
```

### Output:
```
```

![The prompt for generating subqueries based on previously raised concerns.](assets/page_0017_img_2.png)

Figure 6. The prompt for raising concerns either targeting at specific memory points or probing useful information outside the current memory.

Figure 7. The prompt for generating subqueries based on previously raised concerns.
```

### --- Page 0018 ---

```markdown
Given a [Paragraph] and a [Question], you will evaluate the quality of the [Response] in terms of Comprehensiveness.

![The prompt for evaluating the comprehensiveness of a model response.](assets/page_0018_img_1.png)

### Evaluation Criteria

Comprehensiveness measures whether the [Response] comprehensively covers all key aspects in the [Paragraph] with respect to the [Question].

| Level | Score Range | Description |
|-------|-------------|-------------|
| 1     | 0-10       | The response is extremely one-sided, ignoring key parts or important aspects of the question. |
| 2     | 11-20      | The response has some content but misses many important aspects of the question and is not comprehensive enough. |
| 3     | 21-40      | The response is moderately comprehensive, covering some aspects of the question, but has some omissions. |
| 4     | 41-60      | The response is somewhat comprehensive, covering most aspects of the question, with few discussions. |
| 5     | 61-80      | The response is extremely comprehensive, covering almost all aspects of the question on omissions, causing the reader to gain a complete and enriching understanding. |

Evaluate the [Response] using the criteria above, give a level of comprehensiveness in [Level] based on the description of the indicator, and then give a score [Score] based on the corresponding value range, and finally explain in [Explanation].

**Note that:**
1. You should refer to the [Paragraph] and avoid misinterpreting any content of [Paragraph] as part of the [Response].
2. You should consider the [Question] very specific details. When the response mentions an aspect without providing very specific details, you should consider it the whole scope of the response.
3. You should not consider any other content outside the [Paragraph], as long as the context is correct, do not consider the extra elements outside for giving final evaluation.
4. Use the following format: Anticipated Output Format- give your evaluation results in [Your Evaluation].
5. [Score]: A range from 0 to 10. This should be a single number satisfying the range constraint of the corresponding [Level], not a range.
6. [Explanation]: xxx
7. [Your Evaluation]:

---

Given a [Paragraph] and a [Question], you will evaluate the quality of the [Response] in terms of Diversity.

![The prompt for evaluating the diversity of a model response.](assets/page_0018_img_2.png)

### Evaluation Criteria

Diversity measures how varied and rich the response is in offering different perspectives and insights related to the question.

| Level | Score Range | Description |
|-------|-------------|-------------|
| 1     | 0-20       | The response is extremely narrow and repetitive, revealing only a single perspective or might overlook exploring alternative viewpoints. |
| 2     | 21-40      | The response defines a few different perspectives but remains largely superficial. It may touch on alternative viewpoints but does not elaborate or provide substantial insights. |
| 3     | 41-60      | The response mentions several perspectives with moderate depth. It begins to integrate different viewpoints and provides more important pieces of local beyond exploration. |
| 4     | 61-80      | The response is rich in perspectives and insights. It basically outlines multiple viewpoints and provides substantial evidence as examples to support each one. |
| 5     | 81-100     | The response is exceptionally varied and rich in perspectives and insights. It offers a comprehensive exploration of the question, identifying the multiple depth and originality. |

Evaluate the [Response] using the criteria listed above, give a level of diversity in [Level] based on the description of the indicator, and then give a score [Score] based on the corresponding value range, and finally explain in [Explanation].

**Note that:**
1. You should refer to the [Paragraph] and avoid misinterpreting any content of [Paragraph] as part of the [Response].
2. You should consider the [Question] very specific details. When the response mentions an aspect without providing very specific details, you should consider it the whole scope of the response.
3. You should not consider any other content outside the [Paragraph], as long as the context is correct, do not consider the extra elements outside for giving final evaluation.
4. Use the following format: Anticipated Output Format- give your evaluation results in [Your Evaluation].
5. [Score]: A range from 0 to 10. This should be a single number satisfying the range constraint of the corresponding [Level], not a range.
6. [Explanation]: xxx
7. [Your Evaluation]:

---

### G CASE STUDY

As shown in Table 6, we present two representative cases highlighting HGMem's distinct reasoning advantages over LightRAG from the perspective of forming high-order correlations and the strategy of adaptive memory-based evidence retrieval during memory evolving.

The first case is from NarrativeQA, where the question requires inferring the underlying cause of Xo-dar's disalignment—a relation not explicitly stated in the original text. LightRAG just makes incorrect surface-level predictions based on the retrieved content. While DeepRAG stores the knowledge in the memory, it does not form high-order correlation and fails to predict correctly. 
```

### --- Page 0019 ---

```
# Preprint

HGMem progressively evolves its memory and establishes high-order correlations from primitive evidences accumulated from past interactions, uncovering that Xodar’s punishment originates from his defeat by Carter.

The second case is from Nocha, where the query mixes factual and misleading details. The LLM raises a subquery about the source of the name ‘White Sands’. Using the strategy of local investigation, it particularly conducts in-depth inspection about the related memory point (Point 1) in current memory and verifies that there is no clear evidence showing the name was given by Anne. However, LightRAG mistakenly recognizes that the name ‘White Sands’ was given by Anne and DeepRAG doesn’t qualify the correctness of ‘White Sands’.

Together, these examples show that HGMem enables a deeper and more accurate contextual understanding beyond superficial text retrieval.

## H A TOY EXAMPLE

To illustrate the core workflow of our method, we present a toy example in Figure 10. Given the query “Why is Xodar given to Carter as a slave?”, the LLM first retrieves relevant evidence, converting it into a structured representation (corresponding to Point 0 in the figure). It then generates sub-queries based on current memory to retrieve missing reasoning elements. In the subsequent iteration, newly retrieved evidence is integrated into the memory storage through update, insertion and merging operations, yielding a unified representation that includes high-order memory points capturing complex relationships beyond surface content in original data sources. Finally, the LLM leverages its evolved memory to produce an answer to the target query. This example illustrates how the memory evolves during the multi-step RAG execution to iteratively refine its understanding and support complex relational modeling.
```
![Detailed description of Figure 10](assets/page_0019_img_1.png)

### --- Page 0020 ---

```markdown
# Table 6: Illustrative Cases on NarrativeQA and Nocha, where red and blue stand for the relevant answer and its corresponding source, respectively

| Source          | NarrativeQA                                                                 | Nocha                                                                                     |
|-----------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Question**    | Why is Xdair given to Carter as a slave?                                   | Answer TRUE if the statement is true, otherwise answer FALSE. <br> **Statement:** Anne assigns romantic names to her surroundings, like "Lake of Shining Waters" and "White Sands." <br> **[Notice:]** Based on the context provided, is the statement TRUE or FALSE? |
| **Golden Answer** | As his punishment for being defeated by Carter previously                   | **False**                                                                                 |
| **Relevant Chunks:** | 1. Xdair was brought from the adjoining room, less afraid of him, the Hell-bent boy he despised. <br> 2. "And should you are afraid of the boy from Hell?" <br> 3. "You have despised you have bought him from me?" <br> 4. "You have degraded your rank above the lowest. No longer to you a Dancer, but remember a slave of 2..." | 1. "O, I do like that name, either. I shall call it—let me—" <br> 2. "I do like Lake of Shining Waters. Yes, I think that is the right name for it. I always have liked it." <br> 3. "What is it one that matters?" <br> 4. Matilda murmured, "It is as nice as it sounds." <br> 5. "And when you said 'slave' it isn't a picture in my mind, as quick as that!" <br> 6. "And White Sands is pretty a name, but I don't like it as well as Lake." |
| **LightBG.Re-** | **Prediction:** <br> Xdair is given to Carter as a slave due to his degradation by Issa after falling from grace. <br> He serves below the level and commands him there as a slave of Carter, who previously conquered Xdair in the prison on the Isle of Shadow. | **Prediction:** <br> [Answer]: TRUE <br> [Explanation]: The statement is true based on the information provided in the table. In Relationship, there is a reason that indicates Anne of Green Gables gives the romantic name "Lake of Shining Waters" to Barry's Pond, as follows: <br> "Source Name: Anne of Green Gables" <br> "White Sands" suggests a preference for romantic names. |
| **DepRAG.Re-**  | **Generated Subquery:** <br> 1. What does it mean to influence Issa's decision to devote Xdair from a slave? <br> 2. What specific orders or instructions did Issa give to ensure Xdair is treated as Carter's slave? | **Generated Subquery:** <br> 1. Anne gives the romantic name "Lake of Shining Waters" to Barry's Pond, including her imaginative point of view. <br> 2. Anne lists Issa's Lake picturesque and romantic, appreciating its beauty and its name. |
| **HG.Mea.Re-**  | **Generated Subquery:** <br> 1. What specific incident led Xdair was condemned a disgrace to the Immortal Race, leading directly to his punishment by Issa? <br> 2. What was the critical factor in the decision to give Xdair to him as a slave? | **Generated Subquery:** <br> 1. There is an incident in "Anne of Green Gables" that suggests there are romantic names to a place known as White Sands? <br> 2. Was the actual name of White Sands in the book "Anne of Green Gables"? <br> 3. ... |
| **Original Memory:** | **Point 1:** <br> **Descriptor:** Xdair, Issa, Cowardly injustice <br> Xdair is punished by Issa as a form of cowardly injustice for the disgrace he has brought upon the Immortal Race. <br> **Point 2:** <br> **Descriptor:** Xdair, Issa, Degradation <br> Degradation has degraded Xdair to a rank below the lowest. Xdair is no longer a Dancer, but forever a slave of Issa, and must carry for the lower ones that serve in the gardens of Issa. | **Memory After Evolving (Subquery 1):** <br> **Point 1 (Subquery):** <br> **Descriptor:** Dinah, Barry's Pond, Lover's Lane, Gilberts, Anne, White Sands, and others. <br> **Subquery:** Anne has called Barry's Pond the "Lake of Shining Waters." However, there is no specific mention of the romantic name to "White Sands." |
| **Prediction:** | Xdair is given to Carter as a slave due to his disgraceful debt to John Carter, with brotherly shame due to the Immortal Race. Issa, the goddess, punishes Xdair for his cowardice, degrading him to a slave, and he is subsequently treated as such by Carter. | **Prediction:** <br> [Answer]: FALSE <br> [Explanation]: According to the provided information, Anne does give a romantic name to Barry's Pond, referring to it as the "Lake of Shining Waters." However, there is no reference in the text that states that Anne assigns a romantic name to "White Sands," instead, White Sands is associated with Gilbert and an educational opportunity for Anne, and in this narrative, the statement is false because it incorrectly includes "White Sands" as one of the locations Anne romanticizes. |
```

### --- Page 0021 ---

```markdown
# Preprint

## Query: Why is Xodar given to Carter as a slave?

| **Iteration 0** |
|------------------|
| **Memory $M_0$:** None |
| - Retrieve Information |
| - Memory Evolving |
| - Generate Subquery |

---

| **Iteration 1** |
|------------------|
| **Memory $M_1$:** |
| - **Point I** |
|   - **Subordinate Entities:** Xodar, Issus, Cowardly Injustice |
|   - **Description:** Xodar is punished by Issus as a form of cowardly injustice for the disgrace he has brought upon the Immortal Race. |
| - **Point II** |
|   - **Subordinate Entities:** Xodar, Issus, Degradation |
|   - **Description:** Issus degrades Xodar to a rank below the lowest, Xodar is no longer a Dator, but forever a slave of slaves, to fetch and carry for the lower orders that serve in the gardens of Issus. |
| - **Point III** |
|   - **Subordinate Entities:** Nobel Blacks, Xodar, Thurid |
|   - **Description:** The Noble Blacks exhibit hostility towards Xodar, reviling and cursing him for his fall from favor. Thurid is suggested as a challenge for Xodar, demonstrating the social dynamics and conflicts within the group. This reflects the broader theme of societal judgment and persecution against those who have lost status. |
| - Retrieve Information |
| - Memory Evolving |

---

| **Iteration 2** |
|------------------|
| **Memory $M_2$:** |
| - **Point I** |
|   - **Subordinate Entities:** Xodar, Issus, Cowardly Injustice, Degradation |
|   - **Description:** Xodar's degradation into a slave is a result of his disgraceful defeat by John Carter, which brought shame to the Immortal Race, leading Issus to punish him as a form of cowardly injustice. This reflects the societal judgment and loss of honor within the Immortal Race's culture. |
| - **Point III** |
|   - **Subordinate Entities:** Nobel Blacks, Xodar, Thurid |
|   - **Description:** The Noble Blacks exhibit hostility towards Xodar, reviling and cursing him for his fall from favor. Thurid is suggested as a challenge for Xodar, demonstrating the social dynamics and conflicts within the group. This reflects the broader theme of societal judgment and persecution against those who have lost status. |
| - Generate Response |

---

**Response:** Xodar is given to John Carter as a slave by Issus to degrade him further after his defeat and humiliation. Issus wants to punish Xodar by making him serve the very person who conquered him, marking a drastic change in Xodar's status from a member of the First Born to a slave.

**Reference Answer:** As his punishment for being defeated by Carter previously.

---

![A toy example of HGMem workflow from the NarrativeQA dataset of GPT-4o](assets/page_0021_img_1.png)
```


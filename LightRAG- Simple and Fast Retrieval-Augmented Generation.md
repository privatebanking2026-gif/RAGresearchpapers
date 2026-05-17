# LightRAG- Simple and Fast Retrieval-Augmented Generation

### --- Page 0001 ---

```markdown
# LIGHTRAG: SIMPLE AND FAST RETRIEVAL-AUGMENTED GENERATION

Zirui Guo¹, Lianghao Xia², Yanhua Yu¹*, Tu Ao¹, Chao Huang²  
Beijing University of Posts and Telecommunications¹  
University of Hong Kong²  
zrgui001@hku.hk aka_xia@foxmail.com chaohuang75@gmail.com

## ABSTRACT

Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user needs. However, existing RAG systems have significant limitations, including reliance on flat data representations and inadequate contextual awareness, which can lead to fragmented answers that fail to capture complex inter-dependencies. To address these challenges, we propose LightRAG, which incorporates graph structures into text indexing and retrieval processes. This innovative framework employs a dual-level retrieval system that enhances comprehensive information retrieval from both low-level and high-level knowledge discovery. Additionally, the integration of graph structures with vector representations facilitates efficient retrieval of related entities and their relationships, significantly improving response times while maintaining contextual relevance. This capability is further enhanced by an incremental update algorithm that ensures the timely integration of new data, allowing the system to remain effective and responsive in rapidly changing data environments. Extensive experimental validation demonstrates considerable improvements in retrieval accuracy and efficiency compared to existing approaches. We have made our LightRAG open-source and available at the link: [https://github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)

## 1 INTRODUCTION

Retrieval-Augmented Generation (RAG) systems have been developed to enhance large language models (LLMs) by integrating external knowledge sources Sudhi et al. (2024); Es et al. (2024); Salem & Zaimi (2024). This innovative integration allows LLMs to generate more accurate and contextually relevant responses, significantly improving their utility in real-world applications. By adapting to specific domain knowledge Tu et al. (2024), RAG systems ensure that the information provided is not only pertinent but also tailored to the user’s needs. Furthermore, they offer access to up-to-date information Zhao et al. (2024), which is crucial in rapidly evolving fields. Chunking plays a vital role in facilitating the retrieval-augmented generation process Lyu et al. (2024). By breaking down a large external text corpus into smaller, more manageable segments, chunking significantly enhances the accuracy of information retrieval. This approach allows for more targeted similarity searches, ensuring that the retrieved content is directly relevant to user queries.

However, RAG systems have key limitations that hinder their performance. First, many methods rely on flat data representations, restricting their ability to understand and retrieve information based on intricate relationships between entities. Second, these systems often lack the contextual awareness needed to maintain coherence across various entities and their interrelations, resulting in responses that may not fully address user queries. For example, consider a user asking, “How does the rise of electric vehicles influence urban air quality and public transportation infrastructure?” Existing RAG methods might retrieve separate documents on electric vehicles, air pollution, and public transportation challenges but struggle to synthesize this information into a cohesive response. They may fail to explain how the adoption of electric vehicles can improve air quality, which in turn
```
![Detailed description of the chart](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
## Page 2

could affect public transportation planning. As a result, the user may receive a fragmented answer that does not adequately capture the complex inter-dependencies among these topics.

To address these limitations, we propose incorporating graph structures into text indexing and relevant information retrieval. Graphs are particularly effective at representing the interdependencies among different entities Rampášek et al. (2022), which enables a more nuanced understanding of relationships. The integration of graph-based knowledge structures facilitates the synthesis of information from multiple sources into coherent and contextually rich responses. Despite these advantages, developing a fast and scalable graph-empowered RAG system that efficiently handles varying query volumes is crucial. In this work, we achieve an effective and efficient RAG system by addressing three challenges: 
1. **Comprehensive Information Retrieval.** Ensuring comprehensive information retrieval that captures the full context of inter-dependent entities from all documents; 
2. **Enhanced Retrieval Efficiency.** Improving retrieval efficiency over the graph-based knowledge structures to significantly reduce response times; 
3. **Rapid Adaptation to New Data.** Enabling quick adaptation to new data updates, ensuring the system remains relevant in dynamic environments.

In response to the outlined challenges, we propose LightRAG, a model that seamlessly integrates a graph-based text indexing paradigm with a dual-level retrieval framework. This innovative approach enhances the system’s capacity to capture complex inter-dependencies among entities, resulting in more coherent and contextually rich responses. LightRAG employs efficient dual-level retrieval strategies: low-level retrieval, which focuses on precise information about specific entities and their relationships, and high-level retrieval, which encompasses broader topics and themes. By combining both detailed and conceptual retrieval, LightRAG effectively accommodates a diverse range of queries, ensuring that users receive relevant and comprehensive responses tailored to their specific needs. Additionally, by integrating graph structures with vector representations, our framework facilitates efficient retrieval of related entities and relies on enhancing the comprehensiveness of results through relevant structural information from the constructed knowledge graph.

- **General Aspect.** We emphasize the importance of developing a graph-empowered RAG system to overcome the limitations of existing methods. By integrating graph structures into text indexing, we can effectively represent complex interdependencies among entities, fostering a nuanced understanding of relationships and enabling coherent, contextually rich responses.

- **Methodologies.** To enable an efficient and adaptive RAG system, we propose LightRAG, which integrates a dual-level retrieval paradigm with graph-enhanced text indexing. This approach captures both low-level and high-level information for comprehensive, cost-effective retrieval. By eliminating the need to rebuild the entire index, LightRAG reduces computational costs and accelerates adaptation, while its incremental update algorithm ensures timely integration of new data, maintaining effectiveness in dynamic environments.

- **Experimental Findings.** Extensive experiments were conducted to evaluate the effectiveness of LightRAG in comparison to existing RAG models. These assessments focused on several key dimensions, including retrieval accuracy, model ablation, response efficiency, and adaptability to new information. The results demonstrated significant improvements over baseline methods.

## 2 RETRIEVAL-AUGMENTED GENERATION

Retrieval-Augmented Generation (RAG) integrates user queries with a collection of pertinent documents sourced from an external knowledge database, incorporating two essential elements: the Retrieval Component and the Generation Component. 
1. The retrieval component is responsible for fetching relevant documents or information from the external knowledge database. It identifies and retrieves the most pertinent data based on the input query. 
2. After the retrieval process, the generation component takes the retrieved information and generates coherent, contextually relevant responses. It leverages the capabilities of the language model to produce meaningful outputs. Formally, this RAG framework, denoted as $\mathcal{M}$, can be defined as follows:

$$
\mathcal{M} = \left( \mathcal{G}, \mathcal{R} = \left( \varphi, \psi \right) \right), \quad \mathcal{M}(q; \mathbf{D}) = q \left( \varphi; \mathbf{D} \right), \quad \hat{\mathbf{D}} = \varphi(\mathbf{D}) \tag{1}
$$

In this framework, $\mathcal{G}$ and $\mathcal{R}$ represent the generation module and the retrieval module, respectively, while $q$ denotes the input query and $\mathbf{D}$ refers to the external database. The retrieval module $\mathcal{R}$
```


### --- Page 0003 ---

```markdown
![Overall architecture of the proposed LightRAG framework](assets/page_0003_img_1.png)

includes two key functionalities: i) Data Indexer $\varphi(\cdot)$: which involves building a specific data structure $\mathcal{D}$ based on the external database $\mathcal{D}$. ii) Data Retriever $\psi(\cdot)$: The relevant documents are obtained by comparing the query against the indexed data, also denoted as “relevant documents”. By leveraging the information retrieved through $\psi(\cdot)$ along with the initial query $q$, the generative model $G(\cdot)$ efficiently produces high-quality, contextually relevant responses.

In this work, we target several key points essential for an efficient and effective Retrieval-Augmented Generation (RAG) system which are elaborated below:

- **Comprehensive Information Retrieval**: The indexing function $\varphi(\cdot)$ must be adept at extracting global information, as this is crucial for enhancing the model's ability to answer queries effectively.

- **Efficient and Low-Cost Retrieval**: The indexed data structure $\mathcal{D}$ must enable rapid and cost-efficient retrieval to effectively handle a high volume of queries.

- **Fast Adaptation to Data Changes**: The ability to swiftly and efficiently adjust the data structure to incorporate new information from the external knowledge base is crucial for ensuring that the system remains current and relevant in an ever-changing information landscape.

## 3 The LightRAG Architecture

### 3.1 Graph-Based Text Indexing

Graph-Enhanced Entity and Relationship Extraction. Our LightRAG enhances the retrieval system by segmenting documents into smaller, more manageable pieces. This strategy allows for quick identification and access to relevant information without analyzing entire documents. We leverage LLMs to identify and extract various entities (e.g., names, dates, locations, and events) along with the relationships between them. The information collected through this process will be used to create a comprehensive knowledge graph that highlights the connections and insights across the entire collection of documents. We formally represent this graph generation module as follows:

$$
\mathcal{D} = \left( \mathcal{V}, \mathcal{E} \right) = \text{Dedup} \circ \text{Prof}(\mathcal{V}, \mathcal{E}), \quad \mathcal{V} \in \bigcup_{i} \text{Recog}(D_i) \tag{2}
$$

where $\mathcal{D}$ represents the resulting knowledge graph. To generate this data, we apply three main processing steps to the raw text documents $D_i$. These steps utilize a LLM for text analysis and processing. Details about the prompt templates and specific settings for this part can be found in Appendix 7.3.2. The functions used in our graph-based text indexing paradigm are described as:

- **Extracting Entities and Relationships.** $\mathcal{R}(\cdot)$: This function prompts a LLM to identify entities (nodes) and their relationships (edges) within the text data. For instance, it can extract entities like "Cardiologists" and "Heart Disease," and relationships such as "Cardiologists diagnose Heart Disease" from the text: "Cardiologists assess symptoms to identify potential heart issues." To improve efficiency, the raw text $D$ is segmented into multiple chunks $D_i$.

- **LLM Profiling for Key-Value Pair Generation.** $\mathcal{P}(\cdot)$: We employ a LLM-enhanced profiling function, $\mathcal{P}(\cdot)$, to generate a text key-value pair $(K, V)$ for each entity node in $\mathcal{V}$ and relation edge in $\mathcal{E}$. Each index key is a word or short phrase that enables efficient retrieval, while the corresponding value is the next paragraph summarizing relevant snippets from external data to aid in text generation. Entities use their names as the sole index key, whereas relations may have multiple index keys derived from LLM enhancements that include global themes from connected entities.

- **Deduplication to Optimize Graph Operations.** $\mathcal{D}(\cdot)$: Finally, we implement a deduplication function, $\mathcal{D}(\cdot)$, that identifies and merges identical entities and relations from different segments of
```

### --- Page 0004 ---

```markdown
# Page 4

The raw text $D_t$. This process effectively reduces the overhead associated with graph operations on $\hat{D}$ by minimizing the graph’s size, leading to more efficient data processing.

Our LightRAG offers two advantages through its graph-based text indexing paradigm. First, **Comprehensive Information Understanding**. The constructed graph structures enable the extraction of global information from multi-hop subgraphs, greatly enhancing LightRAG’s ability to handle complex queries that span multiple document chunks. Second, **Enhanced Retrieval Performance**. The key-value data structures derived from the graph are optimized for rapid and precise retrieval. This provides a superior alternative to less accurate embedding matching methods (Gao et al., 2023) and inefficient chunk traversal techniques (Edge et al., 2024) commonly used in existing approaches.

## Fast Adaptation to Incremental Knowledge Base

To efficiently adapt to evolving data changes while ensuring accurate and relevant responses, our LightRAG incrementally updates the knowledge base without the need for complete reprocessing of the entire external database. For a new document $D'$, the incremental update algorithm processes it using the same graph-based indexing steps $\varphi$ before, resulting in $\hat{D} = (\hat{V}, \hat{E})$. Subsequently, LightRAG combines the new graph data with the original by taking the union of the nodes sets $\hat{V}$ and $V$, as well as the edge sets $\hat{E}$ and $E$.

Two key objectives guide our approach to fast adaptation for the incremental knowledge base: **Seamless Integration of New Data**. By applying a consistent methodology to new information, the incremental update module allows the LightRAG to integrate new external databases without disrupting the existing graph structure. This approach preserves the integrity of established connections, ensuring that historical data remains accessible while enriching the graph without conflicts or redundancies. **Reducing Computational Overhead**. By eliminating the need to rebuild the entire index graph, this method reduces computational overhead and facilitates the rapid assimilation of new data. Consequently, LightRAG maintains system accuracy, provides current information, and conserves resources, ensuring users receive timely updates and enhancing the overall RAG effectiveness.

### 3.2 Dual-level Retrieval Paradigm

To retrieve relevant information from both specific document chunks and their graph dependencies, our LightRAG proposes generating query keys at both detailed and abstract levels.

- **Specific Queries**. These queries are detail-oriented and typically reference specific entities within the graph, requiring precise retrieval of information associated with particular nodes or edges. For example, a specific query might be, “Who wrote *Pride and Prejudice*?”

- **Abstract Queries**. In contrast, abstract queries are more conceptual, encompassing broader topics, summaries, or overarching themes that are not directly tied to specific entities. An example of an abstract query is, “How does artificial intelligence influence modern education?”

To accommodate diverse query types, the LightRAG employs two distinct retrieval strategies within the dual-level retrieval paradigm. This ensures that both specific and abstract inquiries are addressed effectively, allowing the system to deliver relevant responses tailored to user needs.

- **Low-Level Retrieval**. This level is primarily focused on retrieving specific entities along with their associated attributes or relationships. Queries at this level are detail-oriented and aim to extract precise information about particular nodes or edges within the graph.

- **High-Level Retrieval**. This level addresses broader topics and overarching themes. Queries at this level aggregate information across multiple related entities and relationships, providing insights into higher-level concepts and summaries rather than specific details.

## Integrating Graph and Vectors for Efficient Retrieval

By combining graph structures with vector representations, the model gains a deeper insight into the interrelationships among entities. This synergy enhances the retrieval algorithm to effectively utilize both local and global keywords, streamlining the search process and improving the relevance of results.

1. **Query Keyword Extraction**. For a given query $q$, the retrieval algorithm of LightRAG begins by extracting both local query keywords $k^{(l)}$ and global query keywords $k^{(g)}$.

2. **Keyword Matching**. The algorithm uses an efficient vector database to match local query keywords with candidate entities and global query keywords with relations linked to global keys.
```

### --- Page 0005 ---

```markdown
### 3.3 RETRIEVAL-AUGMENTED ANSWER GENERATION

- (iii) Incorporating High-Order Relatedness. To enhance the query with higher-order relatedness, LightRAG further gathers neighboring nodes within the local subgraphs of the retrieved graph elements. This process involves the set $\{v_i | v_i \in V \land (v_i \in N_v \lor v_i \in N_e)\}$, where $N_v$ and $N_e$ represent the one-hop neighboring nodes of the retrieved nodes $v$ and edges $e$, respectively.

This dual-level retrieval paradigm not only facilitates efficient retrieval of related entities and relations through keyword matching, but also enhances the comprehensiveness of results by integrating relevant structural information from the constructed knowledge graph.

### 3.4 COMPLEXITY ANALYSIS OF THE LIGHTRAG FRAMEWORK

In this section, we analyze the complexity of our proposed LightRAG framework, which can be divided into two main parts. The first part is the graph-based Index phase. During this phase, we use the large language model (LLM) to extract entities and relationships from each chunk of text. As a result, the LLM needs to be called $total\_uses\_chunk$ times. Importantly, there is no additional overhead involved in this process, making our approach highly efficient in managing large datasets.

The second part of the process involves the graph-based retrieval phase. For each query, we first utilize the large language model (LLM) to generate relevant keywords. Similar to current Retrieval-Augmented Generation (RAG) systems Gao et al. (2022; 2022); Chan et al. (2024), our retrieval mechanism relies on vector-based search. However, instead of retrieving chunks in conventional RAG, we concentrate on retrieving entities and relationships. This approach markedly reduces retrieval overhead compared to the community-based traversal method used in GraphRAG.

### 4 EVALUATION

We conduct empirical evaluations on benchmark data to assess the effectiveness of the proposed LightRAG framework by addressing the following research questions: 
- (RQ1): How does LightRAG compare to existing RAG baselines in terms of generation performance? 
- (RQ2): How do dual-level retrieval and graph-based indexing enhance the generation quality of LightRAG? 
- (RQ3): What specific advantages does LightRAG demonstrate through case examples in various scenarios? 
- (RQ4): What are the costs associated with LightRAG, as well as its adaptability to data changes?

#### 4.1 EXPERIMENTAL SETTINGS

**Evaluation Datasets.** To conduct a comprehensive analysis of LightRAG, we selected four datasets from the UltraDomain benchmark (Qian et al., 2024). The UltraDomain data is sourced from 428 college textbooks and encompasses 18 distinct domains, including agriculture, social sciences, and humanities. From these, we chose the Agriculture, CS, Legal, and Mix datasets. Each dataset contains between 600,000 and 5,000,000 tokens, with detailed information provided in Table 1. Below is a specific introduction to the four domains utilized in our experiments:

- **Agriculture:** This domain focuses on agricultural practices, covering a range of topics including beekeeping, hive management, crop production, and disease prevention.
- **CS:** This domain focuses on computer science and encompasses key areas of data science and software engineering. It particularly highlights machine learning and big data processing, featuring content on recommendation systems, classification algorithms, and real-time analytics using Spark.
```

### --- Page 0006 ---

```markdown
# Legal
This domain centers on corporate legal practices, addressing corporate restructuring, legal agreements, regulatory compliance, and governance, with a focus on the legal and financial sectors.

# Mixed
This domain presents a rich variety of literary, biographical, and philosophical texts, spanning a broad spectrum of disciplines, including cultural, historical, and philosophical studies.

## Question Generation
To evaluate the effectiveness of RAG systems for high-level sensemaking tasks, we consolidate all text content from each dataset as context and adopt the generation method outlined in Edge et al. (2024). Specifically, we instruct an LLM to generate five RAG users, along with five tasks for each user. Each generated user is accompanied by a textual description detailing their expertise and traits that motivate their question-raising activities. Each user task is also described, emphasizing one of the user’s potential intentions when interacting with RAG systems. For each user-task combination, the LLM generates five questions that require an understanding of the entire corpus. In total, this process results in 125 questions for each dataset.

## Baselines
LightRAG is compared against the following state-of-the-art methods across all datasets:

- **Naïve RAG** (Gao et al., 2023): This model serves as a standard baseline in existing RAG systems. It segments raw texts into chunks and stores them in a vector database using text embeddings. For queries, Naïve RAG generates vectorized representations to directly retrieve text chunks based on the highest similarity in their representations, ensuring efficient and straightforward matching.

- **RO-RAG** (Chan et al., 2024): This approach leverages the LLM to decompose the input query into multiple sub-queries. These sub-queries are designed to enhance search accuracy by utilizing explicit techniques such as rewriting, decomposition, and disambiguation.

- **HyDE** (Gao et al., 2022): This method utilizes the LLM to generate a hypothetical document based on the input query. This generated document is then employed to retrieve relevant text chunks, which are subsequently used to formulate the final answer.

- **GraphRAG** (Edge et al., 2024): This is a graph-enhanced RAG system that utilizes LLM to extract entities and relationships from the text, representing them as nodes and edges. It generates corresponding descriptions for these elements, aggregates nodes into communities, and produces a community report to capture global information. When handling high-level queries, GraphRAG retrieves more comprehensive information by traversing these communities.

## Implementation and Evaluation Details
In our experiments, we utilize the nano vector database for vector data management and access. For all LLM-based operations in LightRAG, we default to using GPT-4-mini. To ensure consistency, the chunk size is set to 100 across all datasets. Additionally, the learning parameter is fixed at 1 for both GraphRAG and LightRAG.

Defining ground truth for many RAG queries, particularly those involving complex high-level semantics, poses significant challenges. To address this, we build on existing work (Edge et al., 2024) and adopt an LLM-based multi-dimensional comparison method. We employ a robust LLM, specifically GPT-4-turbo, to rank each baseline against our LightRAG. The evaluation prompt we used is detailed in Appendix 7.3.4. In total, we utilize four evaluation dimensions, including:

i) **Comprehensiveness**: How thoroughly does the answer address all aspects and details of the question?  
ii) **Diversity**: How varied and rich is the answer in offering different perspectives and insights related to the question?  
iii) **Empowerment**: How effectively does the answer enable the reader to understand the topic and make informed judgments?  
iv) **Overall**: This dimension assesses the cumulative performance across the three preceding criteria to identify the best overall answer.

The LLM directly compares two answers for each dimension and selects the superior response for each criterion. After identifying the winning answer for the three dimensions, the LLM combines the results to determine the overall better answer. To ensure a fair evaluation and mitigate the potential bias that could arise from the order in which the answers are presented in the prompt, we alternate the placement of each answer. We calculate with rates accordingly, ultimately leading to the final results.

## 4.2 COMPARISON OF LIGHTRAG WITH EXISTING RAG METHODS (RQ1)
We compare LightRAG against each baseline across various evaluation dimensions and datasets. The results are presented in Table 1. Based on these findings, we draw the following conclusions:
```

### --- Page 0007 ---

```markdown
| Table 1: Win rates (%) of baselines vs. LightRAG across four datasets and four evaluation dimensions. |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Agriculture     | CS             | Legal          | Mix            |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|                 | NaïveRAG       | LightRAG       | NaïveRAG       | LightRAG       | NaïveRAG       | LightRAG       | NaïveRAG       | LightRAG       | NaïveRAG       | LightRAG       | NaïveRAG       | LightRAG       | NaïveRAG       | LightRAG       |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Comprehensiveness | 32.46         | 67.66          | 38.94          | 61.68          | 16.46          | 83.86          | 38.86          | 61.68          | 32.46          | 67.66          | 38.94          | 61.68          | 16.46          | 83.86          |
| Diversity       | 23.42         | 67.46          | 38.06          | 61.92          | 16.46          | 83.86          | 42.86          | 57.26          | 23.42          | 67.46          | 38.06          | 61.92          | 16.46          | 83.86          |
| Empowerment     | 32.46         | 67.66          | 38.86          | 61.92          | 15.28          | 84.84          | 40.76          | 60.86          | 32.46          | 67.66          | 38.86          | 61.92          | 15.28          | 84.84          |
| Overall         | 32.46         | 67.66          | 38.86          | 61.92          | 15.28          | 84.84          | 40.76          | 60.86          | 32.46          | 67.66          | 38.86          | 61.92          | 15.28          | 84.84          |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| RQ-RAG          | LightRAG       | RQ-RAG         | LightRAG       | RQ-RAG         | LightRAG       | RQ-RAG         | LightRAG       | RQ-RAG         | LightRAG       | RQ-RAG         | LightRAG       | RQ-RAG         | LightRAG       | RQ-RAG         |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Comprehensiveness | 31.66         | 58.46          | 38.56          | 61.26          | 15.92          | 84.84          | 39.76          | 60.86          | 31.66          | 58.46          | 38.56          | 61.26          | 15.92          | 84.84          |
| Diversity       | 29.26         | 70.86          | 32.96          | 63.86          | 11.66          | 88.84          | 30.86          | 60.86          | 29.26          | 70.86          | 32.96          | 63.86          | 11.66          | 88.84          |
| Empowerment     | 31.66         | 65.46          | 36.36          | 63.16          | 15.28          | 84.86          | 42.86          | 57.76          | 31.66          | 65.46          | 36.36          | 63.16          | 15.28          | 84.86          |
| Overall         | 32.26         | 67.66          | 38.96          | 61.92          | 15.28          | 84.84          | 40.76          | 60.86          | 32.26          | 67.66          | 38.96          | 61.92          | 15.28          | 84.84          |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|

## The Superiority of Graph-enhanced RAG Systems in Large-Scale Corpora

When handling large token counts and complex queries that require a thorough understanding of the dataset’s context, graph-based RAG systems like LightRAG and GraphRAG consistently outperform purely chunk-based retrieval methods such as NaïveRAG, HyDE, and RQ-RAG. This performance gap becomes particularly pronounced as the dataset size increases. For instance, in the largest dataset (legal), the dominance of LightRAG is evident. This trend underscores the advantages of graph-enhanced RAG systems in capturing complex semantic dependencies within large-scale corpora, facilitating a more comprehensive understanding of knowledge and leading to improved generalization performance.

### Enhancing Response Diversity with LightRAG

Compared to various baselines, LightRAG demonstrates a significant advantage in the Diversity metric, particularly within the larger Legal dataset. Its consistent lead in this area underscores LightRAG’s effectiveness in generating a wider range of responses, especially in scenarios where diverse content is essential. We attribute this advantage to LightRAG’s dual-level retrieval paradigm, which facilitates comprehensive information retrieval from both low-level and high-level dimensions. This approach effectively leverages graph-based text indexing to consistently capture the full context in response to queries.

### LightRAG’s Superiority over GraphRAG

While both LightRAG and GraphRAG use graph-based retrieval mechanisms, LightRAG consistently outperforms GraphRAG, particularly in larger datasets with complex language contexts. In the Agriculture, CS, and Legal datasets—each containing millions of tokens—LightRAG shows a clear advantage, significantly surpassing GraphRAG and highlighting its strength in comprehensive information understanding within diverse environments. 

### Enhanced Response Variety

By integrating low-level retrieval of specific entities with high-level retrieval of broader topics, LightRAG boosts response diversity. This dual-level mechanism effectively addresses both detailed and abstract queries, ensuring a thorough grasp of information.

### Complex Query Handling

This approach is especially valuable in scenarios requiring diverse perspectives. By accessing both specific details and overarching themes, LightRAG adeptly responds to complex queries involving interconnected topics, providing contextually relevant answers.

### 4.3 ABLATION STUDIES (RQ2)

We also conduct ablation studies to evaluate the impact of our dual-level retrieval paradigm and the effectiveness of our graph-based text indexing in LightRAG. The results are presented in Table 2.

#### Effectiveness of Dual-level Retrieval Paradigm

We begin by analyzing the effects of low-level and high-level retrieval paradigms. We compare two ablated models—each omitting one module—against LightRAG across four datasets. Here are our key observations for the different variants:
```

### --- Page 0008 ---

```markdown
| Table 2: Performance of ablated versions of LightRAG, using NaïveRAG as reference. |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Agriculture     | CS             | Legal          | Mix            |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|                | NaïveRAG      | LightRAG       | NaïveRAG      | LightRAG       | NaïveRAG      | LightRAG       | NaïveRAG      | LightRAG       | NaïveRAG      | LightRAG       | NaïveRAG      | LightRAG       |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Comprehensiveness | 32.46        | 67.46          | 38.4          | 61.6           | 16.4          | 83.66          | 38.86          | 38.56          | 61.6           | 67.6           | 32.46          | 67.46          |
| Diversity       | 23.46         | 57.46          | 38.0          | 60.92          | 15.6          | 83.66          | 38.86          | 38.56          | 61.6           | 67.6           | 32.46          | 67.46          |
| Empowerment     | 22.94         | 67.96          | 38.8          | 61.22          | 16.46         | 83.66          | 42.98          | 57.27          | 32.46          | 67.46          | 32.46          | 67.46          |
| Overall         | 32.46         | 67.46          | 38.8          | 61.6           | 15.2          | 84.4           | 40.0           | 59.6           | 32.46          | 67.46          | 32.46          | 67.46          |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|                | NaïveRAG -High | LightRAG -High | NaïveRAG -High | LightRAG -High | NaïveRAG -High | LightRAG -High | NaïveRAG -High | LightRAG -High | NaïveRAG -High | LightRAG -High | NaïveRAG -High | LightRAG -High |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Comprehensiveness | 36.06       | 64.0           | 43.26         | 56.8           | 19.2          | 80.8           | 36.06          | 64.0           | 36.06          | 64.0           | 36.06          | 64.0           |
| Diversity       | 34.88         | 65.78          | 49.66         | 54.46          | 16.6          | 83.66          | 33.25          | 64.5           | 34.88          | 65.78          | 34.88          | 65.78          |
| Empowerment     | 35.0          | 64.0           | 43.66         | 56.42          | 18.2          | 81.2           | 35.36          | 64.8           | 35.0           | 64.0           | 35.0           | 64.0           |
| Overall         | 34.56         | 65.25          | 43.66         | 56.42          | 18.2          | 81.2           | 35.36          | 64.8           | 35.0           | 64.0           | 35.0           | 64.0           |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|                | NaïveRAG -Low  | LightRAG -Low  | NaïveRAG -Low  | LightRAG -Low  | NaïveRAG -Low  | LightRAG -Low  | NaïveRAG -Low  | LightRAG -Low  | NaïveRAG -Low  | LightRAG -Low  | NaïveRAG -Low  | LightRAG -Low  |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Comprehensiveness | 24.48       | 75.25          | 39.20         | 60.8           | 16.4          | 83.66          | 44.48          | 55.6           | 25.0           | 74.48          | 25.0           | 74.48          |
| Diversity       | 24.64         | 73.76          | 38.4          | 55.92          | 14.2          | 83.66          | 42.98          | 55.6           | 25.0           | 74.48          | 25.0           | 74.48          |
| Empowerment     | 32.09         | 68.46          | 39.26         | 58.26          | 15.76         | 82.56          | 37.56          | 55.6           | 25.0           | 74.48          | 25.0           | 74.48          |
| Overall         | 25.66         | 74.48          | 39.20         | 60.8           | 16.4          | 83.66          | 44.48          | 55.6           | 25.0           | 74.48          | 25.0           | 74.48          |

- **Low-level-only Retrieval**: The -High variant removes high-order retrieval, leading to a significant performance decrease across nearly all datasets and metrics. This drop is mainly due to the loss of comprehensive retrieval, which focuses excessively on entities and their immediate neighbors.

- **High-level-only Retrieval**: The -Low variant prioritizes exploring a broader range of relationships while leveraging entity-entity relationships rather than focusing on specific entities. This approach offers a significant advantage in comprehensiveness, allowing it to gather more extensive and varied information.

- **Hybrid Mode**: The hybrid mode, or the full version of LightRAG, combines the strengths of both low-level and high-level retrieval methods. It retrieves a broader set of relationships while simultaneously conducting an in-depth exploration of specific entities.

### 4.4 Case Study (RQ3)
To provide a clear comparison between baseline methods and our LightRAG, we present specific case examples in Table 3, which includes responses to a machine learning question from both the competitive baseline, GraphRAG, and our LightRAG framework. In this instance, LightRAG outperforms in all evaluation dimensions assessed by the LLM judge, including comprehensiveness, diversity, empowerment, and overall quality. Our key observations are as follows:

1. **Comprehensiveness**: Notably, LightRAG covers a broader range of machine learning metrics, showcasing its comprehensiveness and ability to effectively discover relevant information.
```

### --- Page 0009 ---

```markdown
# Table 3: Case Study: Comparison Between LightRAG and the Baseline Method GraphRAG.

| **GraphRAG** | **Key Metrics for Evaluating Movie Recommendation Systems:** Evaluating the effectiveness of movie recommendation systems is crucial for ensuring user satisfaction and improving the overall performance of these systems. Several metrics are widely recognized for their ability to provide insights into how well a recommendation system performs. Below are some of the most informative metrics: [1] Precision, [2] Recall, [3] F1 Score, [4] Mean Average Precision (MAP), [5] Root Mean Squared Error (RMSE), [6] User Satisfaction Metrics [...] Conclusion: In summary, metrics such as precision, recall, F1 score, MAP, and user satisfaction are critical for evaluating the performance of movie recommendation systems. Their insights can help developers and researchers refine their recommendations to enhance user experiences. By focusing on these metrics, stakeholders can ensure that their recommendation systems effectively meet user preferences and expectations. |

| **LightRAG** | **Key Metrics for Evaluating Movie Recommendation Systems:** Evaluating movie recommendation systems involves multiple metrics that assess the effectiveness and accuracy of the recommendations provided to users. Below are some of the most informative metrics commonly used in this context, derived from data and concepts surrounding recommendation systems: [1] Mean Average Precision & MAP, [2] Precision and Recall [...] [3] Root Mean Squared Error (RMSE) and Mean Squared Error (MSE) [...] 4. Area Under the Curve (AUC) [...] [5] F-Measure [...] [6] User Engagement Metrics [...] Conclusion: A comprehensive evaluation of movie recommendation systems typically involves a range of metrics that capture both the accuracy of the recommendations and user satisfaction. By effectively measuring these aspects, designers can optimize recommendation engines to deliver more personalized and relevant experiences. |

| **LTL Decision** | **Explanation:** "Answer 2 provides a broader array of metrics including MAP, AUC, and user engagement metrics alongside precision, recall, RMSE, and F-Measure. This gives a fuller view of how recommendation systems can be evaluated. Answer 1, while detailed, is more focused on traditional metrics and lacks the inclusion of engagement metrics which are critical for understanding user interaction." |

**Diversity:** "Winner": Answer 2 (LightRAG).

**Explanation:** "Answer 2 not only covers a wider range of metrics but also includes nuanced explanations of how some metrics interrelate and differ from one another, like the inclusion of both RMSE and MSE, as well as the details behind AUC. In contrast, Answer 1 sticks primarily to standard metrics without exploration of relationships."

**Engagement:** "Winner": Answer 2 (LightRAG).

**Explanation:** "Answer 2 provides a richer exploration of the topic, including insights into user predictions and understanding the implications of engagement metrics. It enables readers to make more informed decisions and understand the intricacies of user engagement metrics. Answer 1 is more straightforward but lacks the depth of insight regarding why metrics matter."

---

## 4.5 Model Cost and Adaptability Analysis (RQ4)

We compare the cost of our LightRAG with that of the top-performing baseline, GraphRAG, from two key perspectives. First, we examine the number of tokens and API calls during the indexing and retrieval processes. Second, we analyze these metrics in relation to handling data changes in dynamic environments. The results of this evaluation on the legal dataset are presented in Table 2. In this context, $Text_{rep}$ represents the token overhead for entity and relationship extraction, $C_{max}$ denotes the maximum number of tokens allowed per API call, and $C_{extract}$ indicates the number of API calls required for extraction.

In the retrieval phase, GraphRAG generates 1,399 communities, with 610 level-2 communities actively utilized for retrieval in this experiment. Each community report averages 1,000 tokens, resulting in a total token consumption of 610,000 tokens (610 communities × 1,000 tokens per community). Additionally, GraphRAG's requirement to traverse each community individually leads to hundreds of API calls, significantly increasing retrieval overhead. In contrast, LightRAG optimizes this process by using fewer than 100 tokens for keyword generation and retrieval, requiring only a single API call for the entire process. This efficiency is achieved through our retrieval mechanism.

![Comparison of Cost in Terms of Tokens and API Calls for GraphRAG and LightRAG on the Legal Dataset.](assets/page_0009_img_1.png)
```

### --- Page 0010 ---

```markdown
# 5 RELATED WORK

## 5.1 RETRIEVAL-AUGMENTED GENERATION WITH LLMS

Retrieval-Augmented Generation (RAG) systems enhance LLM inputs by retrieving relevant information from external sources, grounding responses in factual, domain-specific knowledge Ram et al. (2023); Fan et al. (2024). Current RAG approaches Gao et al. (2022; 2023); Chan et al. (2024); Yu et al. (2024) typically embed queries in a vector space to find the nearest context vectors. However, many of these methods rely on fragmented text chunks and only retrieve the top-k contexts, limiting their ability to capture comprehensive global information needed for effective responses.

Recent studies Edge et al. (2024) have explored using graph structures for knowledge representation, two key limitations persist. First, these approaches often lack the capability for dynamic updates and expansions of the knowledge graph, making it difficult to incorporate new information efficiently. In contrast, our proposed model, LightRAG, addresses this challenge by enabling RAG systems to quickly adapt to new information, ensuring the model's timeliness and accuracy. Additionally, existing methods rely on brine-force searches for each generated community, which are inefficient for large-scale queries. Our LightRAG framework overcomes this limitation by facilitating rapid retrieval of relevant information from the graph through our proposed dual-level retrieval paradigm, significantly enhancing both retrieval efficiency and response speed.

## 5.2 LARGE LANGUAGE MODEL FOR GRAPHS

Graphs are a powerful framework for representing complex relationships and find applications in numerous fields. As Large Language Models (LLMs) continue to evolve, researchers have increasingly focused on enhancing their capability to interpret graph-structured data. This body of work can be divided into three primary categories: i) GNNs as Prefix where Graph Neural Networks (GNNs) are utilized as the initial processing layer for graph data, generating structure-aware tokens that LLMs can use during inference. Notable examples include GraphGPT Tang et al. (2024) and LaGa Chen et al. (2024). ii) LLMs as Prefix involves LLMs processing graph data enriched with textual information to produce node embeddings or labels, ultimately refining the training process for GNNs, as demonstrated in systems like GALM Xie et al. (2023) and OFA Liu et al. (2024). iii) LLMs-Graphs Integration focuses on achieving a seamless interaction between LLMs and graph data, employing techniques such as fusion training and GNN alignment, and developing LLM-based agents capable of engaging with graph information directly Li et al. (2023); Brannon et al. (2023).

# 6 CONCLUSION

This work introduces an advancement in Retrieval-Augmented Generation (RAG) through the integration of a graph-based indexing approach that enhances both efficiency and comprehension in information retrieval. LightRAG utilizes a comprehensive knowledge graph to facilitate rapid and relevant document retrieval, enabling a deeper understanding of complex queries. Its dual-level retrieval paradigm allows for the extraction of both specific and abstract information, catering to diverse user needs. Furthermore, LightRAG's seamless incremental update capability ensures that the system remains current and responsive to new information, thereby maintaining its effectiveness over time. Overall, LightRAG excels in both efficiency and effectiveness, significantly improving the speed and quality of information retrieval and generation while reducing costs for LLM inference.
```


### --- Page 0011 ---

```markdown
# REFERENCES

William Brannon, Suaysh Fulay, Hang Jiang, Wonjune Kang, Brandon Roy, Jad Kabbara, and Deb Roy. Congrat: Self-supervised contrastive pretraining for joint graph and text embeddings. arXiv preprint arXiv:2305.14321, 2023.

Chi-Min Chan, Chupu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, and Jie Fu. Rq-rag: Learning to refine queries for retrieval augmented generation. arXiv preprint arXiv:2404.00610, 2024.

Runjin Chen, Tong Zhao, AJAY KUMAR JAISWAL, Neil Shah, and Zhangyang Wang. Llaq: Large language and graph assistant. In International Conference on Machine Learning (ICML), 2024.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rap approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.

Shahul Eshtin James, Luis Espinosa Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval augmented generation. In International Conference of the European Chapter of the Association for Computational Linguistics (EACL), pp. 150–158, 2024.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In International Conference on Knowledge Discovery and Data Mining (KDD), pp. 6491–6501, 2024.

Liuyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval without relevance labels. arXiv preprint arXiv:2212.10496, 2022.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiwei Sun, and Haifeng Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

Yichuan Li, Kaize Ding, and Kyumil Lee. Grenade: Graph-centric language model for self-supervised representation learning on text-attributed graphs. In International Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2745–2757, 2023.

Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, and Muhan Zhang. One for all: Towards training one graph model for all classification tasks. In International Conference on Learning Representations (ICLR), 2024.

Yuanjie Liu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu, Tong Xu, and Qinghong Chen. Crud-rag: A comprehensive chinese benchmark for retrieval-augmented generation of large language models. arXiv preprint arXiv:2401.17043, 2024.

Hongxin Qian, Peitian Zhang, Zheng Liu, Kelong Mao, and Zhicheng Dou. Memorg: Moving towards next-gen rag via memory-inspired knowledge discovery, 2024. arXiv.org/abs/2409.05591.

Ori Ram, Yoav Levine, Itay Dalmiedgos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics (TACL), 11:1316–1331, 2023.

Ladislav Rampáček, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. International Conference on Neural Information Processing Systems (NeurIPS), 35:14501–14515, 2022.

Alireza Salehi and Hamed Zamani. Evaluating retrieval quality in retrieval-augmented generation. In ACM International Conference on Research and Development in Information Retrieval (SIGIR), pp. 2395–2400, 2024.

Viju Sudhi, Sinchana Ramakant Bhat, Max Rudat, and Roman Teucher. Rag-ex: A generic framework for explaining retrieval augmented generation. In ACM International Conference on Research and Development in Information Retrieval (SIGIR), pp. 2776–2780, 2024.
```

### --- Page 0012 ---

```markdown
| Authors                                                                 | Publication Details                                                                 |
|-------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. | Graph instruction tuning for large language models. In ACM International Conference on Research and Development in Information Retrieval (SIGIR), pp. 491–500, 2024. |
| Shangqing Tu, Yanchun Wang, Jifan Yu, Yuyang Xie, Yaran Shi, Xiaozhi Wang, Jing Zhang, Lei Hou, and Juanzi Li. | R-eval: A unified toolkit for evaluating domain knowledge of retrieval augmented large language models. In International Conference on Knowledge Discovery and Data Mining (KDD), pp. 5813–5824, 2024. |
| Han Xie, Da Zheng, Jun Ma, Houyu Zhang, Vassilis N Ioannidis, Xiang Song, Qing Ping, Sheng Wang, Carl Yang, Yi Xu, et al. | Graph-aware language model pre-training on a large graph corpus can help multiple graph applications. In International Conference on Knowledge Discovery and Data Mining (KDD), pp. 5270–5281, 2023. |
| Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and Bryan Catanzaro. | Rankarg: Unifying context ranking with retrieval-augmented generation in limbs. arXiv preprint arXiv:2407.02485, 2024. |
| Penghao Zhao, Hailin Zhang, Qinhun Yu, Zhengren Wang, Yunting Geng, Fangcheng Fu, Ling Yang, Wentao Zhang, and Bin Cui. | Retrieval-augmented generation for ai-generated content: A survey. arXiv preprint arXiv:2402.19473, 2024. |
```

### --- Page 0013 ---

```markdown
# 7 APPENDIX

In this section, we elaborate on the methodologies and experimental settings used in the LightRAG framework. It describes the specific steps for extracting entities and relationships from documents, detailing how large language models (LLMs) are utilized for this purpose. The section also specifies the prompt templates and configurations used in LLM operations, ensuring clarity in the experimental setup. Additionally, it outlines the evaluation criteria and dimensions used to assess the performance of LightRAG against baselines from various dimensions.

## 7.1 EXPERIMENTAL DATA DETAILS

| Statistics      | Agriculture | CS  | Legal | Mix  |
|------------------|-------------|-----|-------|------|
| Total Documents   | 12          | 10  | 94    | 61   |
| Total Tokens      | 2,017,886   | 2,306,535 | 5,081,069 | 619,009 |

Table 4 presents statistical information for four datasets: Agriculture, CS, Legal, and Mix. The Agriculture dataset consists of 12 documents totaling 2,017,886 tokens, while the CS dataset contains 10 documents with 2,306,535 tokens. The Legal dataset is the largest, comprising 94 documents and 5,081,069 tokens. Lastly, the Mix dataset includes 61 documents with a total of 619,009 tokens.

## 7.2 CASE EXAMPLE OF RETRIEVAL-AUGMENTED GENERATION IN LIGHTRAG.

**Query:** What metrics are most informative for evaluating movie recommendation systems?

**High level keywords:** ["Metrics", "Movie recommendation system", "Evaluation methods"]

**Low level keywords:** ["Accuracy", "Precision", "Recall", "F1 score", "User satisfaction", "Diversity", "Coverage"]

### Entities

- **Performance Metrics:** Performance Metrics are measures used to evaluate how well the Decision Tree model performs, including RMSE among others.
- **Precision-Recall Curve:** The precision-recall curve is used to visualize the trade-off between precision and recall for different thresholds in a model.
- **Mean Squared Error (MSE):** Mean Squared Error is calculated based on the total variance in the dataset, showing the relationship between explained and unexplained variance.
- **Actuals vs. Predictions:** Actuals and predictions are compared against predicted models to assess the accuracy of the movie recommendation models.

### Sources

... These metrics are popular for model evaluation and selection purposes. These movie DBs are sorted in order of the estimated ratings we do for MSE and RMSE. We continue to carry out index-based evaluation metrics using ML’s Ranking class. Then, for each case, we take the list of all rated movie items and pass it to our Rank function. In a manner similar to how we computed MSE, we then sort these APIs using a reduced index and divide by the number of users (that is, count of the table’s RDD/rank), we compute the accuracy for the decision tree. First, we will compute the MSE and RMSE metrics using Regression/Act...

![A retrieval and generation example.](assets/page_0013_img_1.png)

In Figure 3, we illustrate the retrieve-and-generate process. When presented with the query, “What metrics are most informative for evaluating movie recommendation systems?”, the LLM first extracts both low-level and high-level keywords. These keywords guide the dual-level retrieval process on the...
```

### --- Page 0014 ---

```markdown
# 7.3 OVERVIEW OF THE PROMPTS USED IN LIGHTRAG

## 7.3.1 PROMPTS FOR GRAPH GENERATION

- **Goal:**  
  Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and relationships among the identified entities.

- **Steps:**  
  1. Identify all entities. For each identified entity, extract the following information:  
     - **entity_name:** Name of the entity, capitalized.  
     - **entity_type:** One of the following types (organization, person, geo, event).  
     - **entity_description:** Comprehensive description of the entity's attributes and activities.  
     - **entity_id:** An identifier for each entity (e.g., `entity_name=entity_type`).

  2. For each entity identified in step 1, identify all pairs of (source_entity, target_entity) that are "clearly related" to each other.  
     For each pair, extract the following information:  
     - **source_entity_name:** Name of the source entity, as identified in step 1.  
     - **target_entity_name:** Name of the target entity, as identified in step 1.  
     - **relationship_description:** A concise explanation as to why the source entity and target entity are related.  
     - **relationship_keywords:** One or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details.  
     - **relationship_strength:** A score (1-5) indicating the strength of the relationship.

  3. List high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.  
     Format the output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use "###" as the list delimiter.

- **When finished, output `<COMPLETE>`**

- **Real Data:**  
  - Entity types: `<entity_types>`  
  - Text input: `<text_input>`

![Graph Construct Prompt](assets/page_0014_img_1.png)

---

## 7.3.2 PROMPTS FOR QUERY GENERATION

Given the following description of a dataset: (total_description)  
Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

- Output the results in the following structure:  
  - **Task 1:**  
    - **Task description:** (Task description 1)  
    - **Question 1:** (Question 1)  
    - **Question 2:** (Question 2)  
    - **Question 3:** (Question 3)  
    - **Question 4:** (Question 4)  
    - **Question 5:** (Question 5)  

  - **Task 2:**  
    - **Task description:** (Task description 2)  
    - **Question 1:** (Question 1)  
    - **Question 2:** (Question 2)  
    - **Question 3:** (Question 3)  
    - **Question 4:** (Question 4)  
    - **Question 5:** (Question 5)  

  - **User 1:**  
    - **User description:** (User 1 description)  
    - **User 2:**  
    - **User description:** (User 2 description)  
    - **User 3:**  
    - **User description:** (User 3 description)  

![Prompts for Query Generation](assets/page_0014_img_2.png)
```

### --- Page 0015 ---

```markdown
# 7.3.3 PROMPTS FOR KEYWORD EXTRACTION

- **Role**:  
  You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

- **Goal**:  
  Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

- **Examples**:  
  **Example 1**:  
  Input: "How does international trade influence global economic stability?"  
  Output:  
  ```json
  {
    "high_level_keywords": ["International trade", "Global economic stability"],
    "low_level_keywords": ["Trade agreements", "Tariffs", "Economic impact"]
  }
  ```

  **Example 2**:  
  Input: "What are the environmental consequences of deforestation?"  
  Output:  
  ```json
  {
    "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
    "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
  }
  ```

- **Real Data**:  
  Query: (query)

![Prompts for Keyword Extraction](assets/page_0015_img_1.png)

---

# 7.3.4 PROMPTS FOR RAG EVALUATION

- **Role**:  
  You are an expert tasked with evaluating two answers to the same question based on four criteria: Comprehensiveness, Diversity, and Empowerment.

- **Goal**:  
  You will evaluate two answers to the same question based on four criteria: Comprehensiveness, Diversity, and Empowerment.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?  
- **Diversity**: How well does this answer provide different perspectives and insights on the question?  
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

- **Output your evaluation in the following JSON format**:  
  ```json
  {
    "Comprehensiveness": {
      "Winner": "Answer 1 or Answer 2",
      "Explanation": "Provide explanation here"
    },
    "Diversity": {
      "Winner": "Answer 1 or Answer 2",
      "Explanation": "Provide explanation here"
    },
    "Empowerment": {
      "Winner": "Answer 1 or Answer 2",
      "Explanation": "Provide explanation here"
    },
    "Overall Winner": {
      "Winner": "Answer 1 or Answer 2",
      "Explanation": "Summarize why this answer is the overall winner based on these criteria"
    }
  }
  ```

![Prompts for RAG Evaluation](assets/page_0015_img_2.png)
```

### --- Page 0016 ---

```markdown
# Table 5: Case Study: Comparison Between LightRAG and the Baseline NaiveRAG.

| Query | How do indigenous perspectives on ownership and collaboration influence corporate mergers in countries like Canada and Australia? |
|-------|----------------------------------------------------------------------------------------------------------------------------------|
| NaiveRAG | Indigenous perspectives on ownership and collaboration play a significant role in shaping corporate mergers in countries like Canada and Australia, particularly where resources and land rights intersect with Indigenous sovereignty and cultural values. These perspectives emphasize community engagement, respect for traditional land use, and a collaborative approach to resource management, influencing corporate strategies and outcomes. ... Both nations illustrate that prioritizing Indigenous perspectives leads to more sustainable and equitable outcomes. Acknowledging Indigenous rights and fostering collaborative relationships can reduce conflicts and enhance corporate reputations. Ultimately, integrating Indigenous perspectives is a pathway to better business practices and community relations. |
| LightRAG | Indigenous Perspectives on Ownership and Collaboration in Corporate Mergers: Indigenous perspectives significantly influence corporate mergers in countries like Canada and Australia. These views emphasize communal rights to land and resources, spiritual connections to the environment, and the importance of collaboration over competition. Cultural Significance of Land Ownership [...] The Role of Collaboration [...] Legal and Regulatory Frameworks [...] Conclusion: Indigenous perspectives shape corporate mergers by emphasizing communal ownership and collaborative partnerships. Corporations must adopt respectful, inclusive practices that honor Indigenous cultural values and rights, paving the way for sustainable, mutually beneficial outcomes. |

| LLM Decision | Completeness: "Winner": "Answer 2 (LightRAG)", "Explanation": "Answer 2 provides a thorough exploration of the indigenous perspectives by discussing cultural significance, collaboration, and legal frameworks, including specific examples and detailed insights. In contrast, Answer 1, while informative, lacks the same depth in analyzing the various dimensions of indigenous perspectives on ownership and collaboration." |
| Diversity: "Winner": "Answer 2 (LightRAG)", "Explanation": "Answer 2 presents a wider array of perspectives, focusing on the communal aspect of land ownership, spiritual connections, and practical examples of collaboration. It covers more ground by contrasting Indigenous views with Western notions, whereas Answer 1 primarily focuses on corporate strategies and outcomes with limited perspective." |
| Empowerment: "Winner": "Answer 2 (LightRAG)", "Explanation": "Answer 2 equips the reader with nuanced understanding and actionable insights by highlighting the importance of collaboration and the legal frameworks involved. It empowers corporations by illustrating how adopting an inclusive approach can facilitate better outcomes. Answer 1, while practical, does not emphasize the moral or ethical implications as strongly." |
| Overall Winner: "Winner": "Answer 2 (LightRAG)", "Explanation": "Answer 2 excels overall due to its comprehensive exploration, diversity of perspectives, and empowerment of the reader with actionable insights about indigenous perspectives and collaboration in corporate mergers. Although Answer 1 is more direct, the depth and breadth of Answer 2 make it the stronger response." |

LLM must identify which answer performs better and provide a rationale for its choice. Ultimately, an overall winner is determined based on performance across all three dimensions, accompanied by a detailed summary that justifies the decision. The evaluation is structured in JSON format, ensuring clarity and consistency, and facilitating a systematic comparison between the two answers.

## 7.4 CASE STUDY: COMPARISON BETWEEN LIGHTRAG AND THE BASELINE NAIVERAG.

To further illustrate LightRAG's superiority over baseline models in terms of comprehensiveness, empowerment, and diversity, we present a case study comparing LightRAG and NaiveRAG in mergers. This study addresses a question regarding indigenous perspectives in the context of corporate mergers. Notably, LightRAG offers a more in-depth exploration of key themes related to indigenous perspectives, such as cultural significance, collaboration, and legal frameworks, supported by specific and illustrative examples. In contrast, while NaiveRAG provides informative responses, it lacks the depth needed to thoroughly examine the various dimensions of indigenous ownership and collaboration. The dual-level retrieval process employed by LightRAG enables a more comprehensive investigation of specific entities and their interrelationships, facilitating extensive searches that effectively capture overarching themes and complexities within the topic.
```


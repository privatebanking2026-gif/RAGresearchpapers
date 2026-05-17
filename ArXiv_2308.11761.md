# ArXiv 2308.11761

### --- Page 0001 ---

```markdown
# KnowledGPT: Enhancing Large Language Models with Retrieval and Storage Access on Knowledge Bases

**Xintao Wang, Qianwen Yang, Yongting Qiu, Jiaqing Liang, Qianyu He, Zhouhong Gu, Yanghua Xiao, Wei Wang**  
Fudan University  
{xtwang21, qywang22, 222120420256, qyhe21@m.fudan.edu.cn, zhg20, shawhy, weiwang1@m.fudan.edu.cn, l.j.q.light@gmail.com}

## Abstract
Large language models (LLMs) have demonstrated impressive impact in the field of natural language processing, but they still struggle with several issues regarding, such as completeness, timeliness, faithfulness and adaptability. While recent efforts have focuses on connecting LLMs with external knowledge sources, the integration of knowledge bases (KBs) remains understudied and faces several challenges. In this paper, we introduce KnowledGPT, a comprehensive framework to bridge LLMs with various knowledge bases, facilitating both the retrieval and storage of knowledge. The retrieval process employs the program of thought prompting, which generates search language for KBs in code format with pre-defined functions for KB operations. Besides retrieval, KnowledGPT offers the capability to store knowledge in a personalized KB, catering to individual user demands. With extensive experiments, we show that by integrating LLMs with KBs, KnowledGPT properly answers a broader range of questions requiring world knowledge compared with vanilla LLMs, utilizing both knowledge existing in widely-known KBs and extracted into personalized KBs.

## 1 Introduction
Large Language Models (LLMs) (OpenAI, 2023) (Chiang et al., 2023) (Taori et al., 2023) have achieved substantial impact across a variety of natural language processing (NLP) tasks like translation (Wang et al., 2023), summarization (Zhang et al., 2023) and question answering (van Sonsbeek et al., 2023), alongside with all kinds of requests from real-world users. Their remarkable capabilities stem from the ever-increasing number of parameters and training data, which expands in correlation with their massive knowledge and emergent abilities like chain-of-thought reasoning (Kojima et al., 2022) and in-context learning (Brown et al., 2020).

![Comparison between retrieval results from document corpus and knowledge bases. In this case, no document retrieved from the corpus provides enough knowledge for the query, while the knowledge base sufficient related knowledge can be retrieved.](assets/page_0001_img_1.png)

However, LLMs still struggle with factual knowledge considering issues like completeness, timeliness, faithfulness and adaptability (OpenAI, 2023) (Kandpal et al., 2023). Firstly, LLMs demonstrate limitations in terms of timely updates and expertise in specific domains. Secondly, these models can generate unfaithful or “hallucinated” knowledge, posing both reliability and ethical concerns. Thirdly, due to constraints like costs and accessibility, LLMs can hardly incorporate new knowledge via continued training, which hampers the ability to tailor these models to accommodate specific knowledge demands. Therefore, these knowledge demands encourage comprehensive research efforts towards integrating LLMs with external sources of knowledge.

Towards this issue, some recent efforts have been made to enable LLMs to access plug-and-play knowledge sources like knowledge bases (KBs) (Modarressi et al., 2023), search engines (Schick et al., 2023), document memories (Liu, 2022), and databases (Hu et al., 2023) to provide LLMs with world knowledge, generally via LLM-generated API calls. In this paper, we focus on knowledge bases (KBs), a unique form of knowledge source featuring entity-centric knowl-
```

### --- Page 0002 ---

```markdown
# KnowledgeGPT: A Comprehensive Framework for LLMs and Knowledge Bases

On one hand, various KBs have been constructed for their practical effectiveness for applications, and the conciseness, expressiveness, interpretability and visibility of their representation. On the other hand, previous approaches have largely focused on document corpus, which reveals several deficiency when applied to KGs, as shown in Fig 1. Therefore, connecting LLMs with KBs is of significant importance, yet still remains underexplored.

Recently, several works have attempted to connect LLMs to KBs. Toolformer (Schick et al., 2023) queries Wikipedia for descriptions of interested entities to answer related questions. Graph-Toolformer (Zhang, 2023) and ToolkenGPT (Hao et al., 2023) make LLMs reason over knowledge graphs like Freebase. RET-LLM (Modarressi et al., 2023) builds personalized KG memory with relational triples extracted from past conversations for future use, in parallel with practical efforts of KG Index in LangChain (Chase, 2022) and Llama Index (Liu, 2022).

However, there are still many challenges in this direction, as shown in Figure 2. Firstly, the process by which LLMs navigate through knowledge bases for intricate and varied questions remains a problem, especially for multi-hop questions which requires information across multiple and nested entries in KBs. Secondly, aligning entities and relations in knowledge bases with their text mentions is a challenging task, as they need to map to a wide spectrum of natural language expressions and account for severe ambiguity in the knowledge bases. Thirdly, while the triple-based representation in KGs is neat and interpretable, it only covers limited information compared with natural language, which suggests the need for new representation form in KBs for LLMs.

In this paper, we propose a comprehensive framework, KnowledGPT, to connect LLMs to various knowledge bases effectively, with an improved capability in dealing with complex questions, ambiguity and knowledge representation. KnowledGPT implements a unified accessor interface for operations on different KBs, including widely-used public KBs and personalized KB memories. KnowledGPT accesses entity-oriented knowledge, including both entity descriptions and relational triples. For a given query, KnowledGPT searches KBs with three steps: search code generation, search execution, and answer generation. Inspired by (Chen et al., 2022), KnowledGPT adopts program of thoughts (PoT) prompting, interacting with KBs by generating Python code which delegates searching steps and executing it. This code encapsulates functions for assessing KBs such as entity-linking. Afterwards, KnowledGPT integrates the retrieved knowledge to generate the response. If KnowledGPT judges that the question does not necessitate knowledge from KBs, or if the retrieved knowledge is inadequate or absent, the question will be directly answered by the LLM. Besides, KnowledGPT can also extract knowledge from unstructured texts represented in various forms to enrich the personalized KB.

Overall, our contributions are summarized as follows:

1. We propose KnowledGPT, a comprehensive framework to enable LLMs to retrieve knowledge from knowledge bases. It significantly advances the collaboration between LLMs and KBs towards vital practical challenges like complex searching and ambiguity.

2. We propose the use of personalized knowledge bases as symbolic memory for LLMs, encapsulating entity-oriented knowledge in three forms of representations. This expands the scope of knowledge in symbolic memories compared with triples-only KBs.

3. We demonstrate the efficacy of our proposed methods with experiments. The results underscore the utility and potential of using KBs as symbolic memory for LLMs.

## 2 Related Works

### External Knowledge and Memory for LLMs

Large language models (LLMs), such as GPT-4 (OpenAI, 2023) and LLaMA (Touvron et al., 2023), have demonstrated impressive performance across various applications. However, they still struggle with knowledge considering completeness, timeliness, faithfulness and adaptability. Hence, many recent efforts have been devoted to equipping LLMs with external knowledge. Intent-augmented language models (Komeili et al., 2022) (Lazaridou et al., 2022), as well as New Bing and ChatGPT "Browse with Bing" plugin, allow LLMs to access up-to-date information with search engines or web browsers. Retrieval-augmented methods like REALM (Guu et al., 2020)...
```


### --- Page 0003 ---

```markdown
![Comparison between KnowledGPT and previous methods towards several challenges in bridging LLMs with KBs](assets/page_0003_img_1.png)

| Complex Question | Ambiguation | Knowledge Representations |
|------------------|-------------|---------------------------|
| Please recommend three novels written by the author of 'Lord of the Mysteries'. | Who directed the film Titanic? | U.S. Bancorp is an American bank holding company based in Minneapolis, Minnesota... |
| **Previous Methods like RET-LM** | | **Only triples** |
| RETRIEVE(`LotM`, `novels by the author of`, ``) tackles entity ambiguity with one step only. | embedding similarity | Extract triples: `(U.S. Bancorp`, `based in`, `Minnesota`) A Represent a limited portion of knowledge |
| **KnowledGPT** | | + entity-specific information |
| multi-hop searching & processing via code | | 'U.S. Bancorp': 'U.S. Bancorp, a bank holding company based in Minneapolis, Minnesota ...' |
| CODE: | | 'U.S. Bancorp': 'scope of business', 'The company provides banking, investment, mortgage, trust, and payment services products to individuals ...' |
| author: `RETRIEVE(`LotM`, `written by`, `author`)` | | |
| novels = `RETRIEVE(novels, `mi`, `len`, `novels`)` | | |

2020, RAG (Lewis et al., 2020), augment LLMs with document corpus, which have also been increasingly adopted by recent popular LLMs like ChatGPT as the memory unit (Liu, 2022; Chase, 2022). ChatDB (Hu et al., 2023) augments LLMs with databases as symbolic memory.

**Knowledge Bases for LLMs** Some recent works have studied to augment LLMs with knowledge from external KBs or use KBs as symbolic memories, usually by making LLMs generate API calls for KB operations. Toolformer (Schick et al., 2023) trains LLMs to search Wikipedia for texts of entities. Graph-Toolformer (Zhang, 2023) empowers LLMs to reason over knowledge graphs. However, it skips the entity linking step, so it requires entity id like im/053yx as input, instead of their names. ToolkenGPT (Hao et al., 2023) keeps the LLMs frozen and trains tool embeddings for relations in KBs to support relational queries. RET-LLM (Modarressi et al., 2023), similar to the KGC memory of LangChain (Chase, 2022) and Llama-Index (Liu, 2022), extracts relational triples from user inputs and store them in a symbolic KG memory. Compared with previous efforts, KnowledGPT supports various knowledge representations and both public and private KBs, as shown in Table 1.

**Knowledge-based Question Answering** (KBQA) is a search for answer entities or relations given natural language queries specific to certain KGs. Existing KBQA systems are mainly based on semantic parsing (Berant et al., 2013) or information extraction (Yao and Van Durme, 2014), where language models are increasingly involved. Semantic parsing methods (Yu et al., 2023; Cao et al., 2022; Zhang et al., 2022b; Abdelaziz et al., 2021; Lai et al., 2016) leverage semantic parser to convert natural language queries into intermediate logic forms such as SPARQL (Prud'hommeaux, 2011), which are executed on KBs to obtain the answers. However, the generated logic forms are usually non-executable, thus failing to arrive at the correct answer (Sun et al., 2020). Pangu (Gu et al., 2022) trains a language model discriminator to evaluate probability of candidate plans. Information extraction methods usually combines retrieval and reasoning (Zhang et al., 2022a; Shi et al., 2021; Sun et al., 2019; Jiang et al., 2023; Baek et al., 2023). These methods show effectiveness in handling single-hop retrieval. However, they encounter challenges with multi-hop retrieval concerning storage and computation costs, where the number of relations expands exponentially with each added hop.

KnowledGPT differs from KBQA methods in two aspects. First, many KBQA methods are designed for special queries about relational triples in KGs, while KnowledGPT augments LLMs to respond to various user queries with knowledge in various forms from KBs. Second, KBQA methods are typically trained on specific datasets and KGs, whereas KnowledGPT requires no training and can easily accommodate different LLMs and KBs.
```

### --- Page 0004 ---

```markdown
| Method        | Ent | Desc | Triples | Ent Aspect Info | Multi-Hop Search | Zero-Shot | Ent Linking | External KBs | Private KB |
|---------------|-----|------|---------|------------------|------------------|-----------|-------------|---------------|------------|
| Toolform      | ✓   | X    | ✓       | ✓                | X                | ✓         | X           | X             | X          |
| ToolkenGPT    | X   | X    | X       | X                | X                | X         | ✓           | X             | X          |
| Graph-Toolformer | X | ✓    | X       | X                | X                | X         | X           | X             | X          |
| RET-LLM       | X   | X    | X       | X                | X                | ✓         | X           | X             | X          |
| LangChain KG Memory | X | ✓ | ✓       | X                | ✓                | ✓         | X           | X             | ✓          |
| Llama-Index KG Index | X | ✓ | X     | X                | X                | ✓         | X           | ✓             | ✓          |
| KnowledGPT    | ✓   | ✓    | ✓       | ✓                | ✓                | ✓         | ✓           | ✓             | ✓          |

Table 1: Comparison between KnowledGPT and existing KB-augmented methods. Ent, Rel, and Desc are abbreviations for entity, relation and description, respectively.

## 3 Methods

In this section, we introduce KnowledGPT, a comprehensive framework to integrate LLMs with KBs. We first provide the definition of two tasks of KnowledGPT, knowledge retrieval and knowledge storage (Sec 3.1). Then, we elaborate the details in the retrieval (Sec 3.2) and storage (Sec 3.3) process of KnowledGPT.

### 3.1 Task Definition

KnowledGPT supplements LLMs with external knowledge from various knowledge bases (KBs), including a personalized KB (PKB) as an writable symbolic memory. Given a user input in natural language, KnowledGPT undertakes two primary tasks, namely knowledge retrieval and knowledge storage. In the knowledge retrieval task, the model searches through provided KBs to retrieve relevant knowledge to answer the user query. In the knowledge storage task, the model extracts knowledge from the user input and inserts it into the PKB.

### 3.2 Knowledge Retrieval

KnowledGPT follows a three-step process to answer user queries with knowledge from KBs, as shown in Fig 3. First, it generates a piece of search code as a logical form for query-specific KB access. Then, the search code is executed to retrieve related knowledge. Finally, KnowledGPT reads the retrieved knowledge and answers the query.

We utilize the program of thought (PoT) prompting approach (Chen et al., 2022), which adopts Python code as the search language generated by LLMs. In this paper, we use GPT-4 (OpenAI, 2023) as the LLM. The code is encapsulated in a search function, as is shown in the yellow part of Fig 3, which includes built-in Python functions and three custom KB functions designed to facilitate the interaction of LLMs with KBs:

1. `get_entity_info`, which accepts an entity as input and returns its encyclopedic description.

2. `find_entity_or_value`, which accepts a query consisting of an entity and a relation as input, and outputs a list of the corresponding entity or value.

3. `find_relationship`, which accepts two entities as input, and returns a list of their relationship.

Specially, each entity or relation is represented as a list of candidate aliases, rather than a single name, to effectively handle synonyms. Besides the outputs stated above, these KB functions also return a message which logs the function call and result. Then, the overall output of the search function is obtained by concatenating the messages from individual KB function calls. The prompt is shown in 7.

The search function is then executed to retrieve the expected knowledge from KBs. The code would be decorated before execution, e.g. with a try-except statement and KB-specific accessor object, as is elaborated in Sec 3.2.1. The search function is executed for each KB respectively in parallel, and their results are concatenated.

Finally, the retrieved knowledge is provided to LLMs, and LLMs are tasked with responding to the user’s query, supported by the retrieved knowledge. The prompt is shown in Sec 7. LLMs will ignore the retrieved information and address the user query independently in scenarios where LLMs judge the question does not require external knowledge or the retrieved knowledge is not enough for the query.

### 3.2.1 Code Implementation

Next, we introduce the implementation of the KB functions to execute the generated code. We implement the functions at two levels: a unified level and a KB-specific level.

Functions at the unified level provide a unified interface for operations over different KBs. These include the three KB functions (`get_entity_info`, `find_entity_or_value`, `find_relationship`) generated
```

### --- Page 0005 ---

```markdown
| Query: What are the titles of the poet writing  | Entity Linking  |
|--------------------------------------------------|------------------|
| Quiet Night Thoughts [Ling ID: 9]?               | Who is the lyricist |
|                                                  | target entity: [Bai Ju] |
|                                                  | Relation: [Writer] |

### Original Answer from GPT-4:
The poet of Quiet Night Thoughts is Bai Ju from the Tang Dynasty. He has multiple titles, including 'The Immortal Poet', 'The Wandering Poet', and 'The Wine Immortal'. His poetic talent was outstanding, his works are widely circulated, and he has been revered as 'the God in Poetry' by later generations.

### Code for Knowledge Retrieval:
```python
search(msg = 'search', 
       msg = msg, 
       entity_aliases = ['author', 'creator', 'writer'])
if author:
    titles, msg = find_entity_or_value(entity_aliases = author, 
                                        relation_aliases = ['title', 'also known as', 'appellation'])
```

### Retrieved Knowledge from KBs:
FROM ONDB:
```
find_entity_or_value(entity_aliases = ['Quiet Night Thoughts'], 
                     relation_aliases = ['Author', 'creator', 'writer']) -> [ 'Quiet Night Thoughts' ]
find_entity_or_value(entity_aliases = ['Bai Ju', 'The Order of Heaven’s Mandate'], 
                     relation_aliases = ['Bai Ju', 'The Order of Heaven’s Mandate', 'Relation_aliases = [title, Blue Lotus Hermit, Exiled Immortal.']
```

### Answer with Retrieved Knowledge:
The poet Bai Ju, who wrote 'Quiet Night Thoughts', has the titles 'Blue Lotus Hermit' and 'Exiled Immortal'.

![Detailed process of KnowledGPT's retrieval steps for the given example.](assets/page_0005_img_1.png)

### Figure 4: The detailed process of KnowledGPT's entity linking steps for the given example.
The company Apple.
Our entity_linking function comprises three steps, as depicted in Fig 4. First, we invoke the KB-specific _entity_linking function to obtain candidate entities. It basically takes as input the query and entity aliases, and utilizes the entity linking API (with both entity names and the context) and searching API (with only entity names) provided by the corresponding KB. Second, we call the _get_entity_info function (introduced in Sec. 3.2.3) to gather information about the candidates. Each piece of entity information will be truncated to a maximum length. Finally, we provide LLMs with the function input (including the query, aliases of entity and relation) and the candidate entities along with their information, and make LLMs determine the most appropriate entity.

### 3.2.3 Get Entity Information
The get_entity_info function retrieves information about a specific entity. It first employs the entity_linking function to link entity aliases to an entity in the KB. Subsequently, it invokes the KB-specific _get_entity_info function, which returns information of the given entity in KB, including its entity description and triple information. The _get_entity_triples function is called to collect its triples. The KB-specific _get_entity_info function is nested in the entity_linking function, which makes it an integral part of all KB functions at the unified level.

### 3.2.4 Find Entity or Value
Given a query composed of an entity and a relation, the find_entity_or_value function is designed to retrieve the corresponding entity or attribute value. This function undergoes several steps, as is presented in Algorithm 1. Still, it starts by invoking the entity_linking function to associate entity aliases with a corresponding entity in the KB. Then, it calls the internal find_entity_or_value function, which includes a KB-specific _get_entity_triples.
```

### --- Page 0006 ---

```markdown
# Algorithm 1: find_entity_or_value

**Input:** query $q$, alias list of entity $\mathcal{E}$, alias list of relation $R$.  
**Output:** a list of target entities or attribute values $T$.

1. Function EMBSIM(str_r, list[str_r]/R):
2. \quad $s \gets -1$
3. \quad $r \gets \text{embedding}(r)$
4. \quad for $r_i \in R$ do
5. \quad \quad $r_i \gets \text{embedding}(r_i)$
6. \quad \quad if $c(r_i) > s$ then
7. \quad \quad \quad $s \gets c(r_i)$
8. \quad \quad return $s$
9. $e \gets \text{entity\_linking}(\mathcal{E})$
10. if $e == \text{NULL}$ then
11. \quad return $\text{NULL}$
12. $r \gets \text{NULL}$
13. $s_k \gets -1$
14. for triple $t$ triples do
15. \quad $r_i \gets \text{triple.rel}$
16. \quad $s_i \gets \text{EMBSIM}(r_i, R)$
17. \quad if $s_i > s$ then
18. \quad \quad $s_r \gets s_i$
19. \quad \quad $r \gets r_i$
20. if $r == \text{NULL}$ then
21. \quad return $\text{NULL}$
22. $triples_r \gets \text{triples with relation } r$
23. $R \gets \text{target entities or attribute values in } triples_r$
24. return $R$

### 3.3 Knowledge Storage

While public KBs provide abundant world knowledge, they are still unable to cover all knowledge that users are interested in. To meet users’ personal knowledge demands, KnowledGPT introduces a personalized KB (PKB) that acts as LLMs’ symbolic memory, granting users the capability to store and access specialized knowledge. The PKB is populated by knowledge extracted from user-provided documents. When users intend to add knowledge into the PKB we prompt LLMs to extract knowledge from their provided documents, with the prompt shown in Sec A.

We consider knowledge represented in three forms, including entity description, relational triples, and entity-aspect information, as is shown in Fig 2, which is different from RET-LLM (Modarressi et al., 2023) or the KG-Index of LangChain (Chase, 2022) and Llama Index (Liu, 2022) that extract only triples, While entity description and relational triples have been widely adopted in knowledge bases like Wikipedia and Wikidata, they only represent a limited portion of knowledge, as is shown by experiments in Sec 4.4. For example, when we want to know the experience of Socrates as a soldier, most content in Socrates’ wikipedia page would be hardly helpful, and it can also hardly be represented as a triple. Therefore, we propose an additional knowledge representation, termed as entity-aspect information, for symbolic memory of LLMs. It is a variation of triple where the object is a long piece of text which describes and can be retrieved by an entity and an aspect. For instance, a record might be indexed by (“Socrates”, “Military Service”) and correspond to the description “Socrates served as a Greek hoplite or heavy infantryman...”. Knowledge represented in this form will also be retrievable by the get_entity_or_value function.

Given the utility of PKBs in comparison to public KBs, we consider a different strategy for entity linking on PKBs. The difference is mainly three-fold. First, we define the entity searching API for PKB based on exact match and embedding similarity. Embedding similarity aids in recognizing widely-known entity aliases, such as Chanele Scott Calica and Shystie. Second, during extraction,
```

### --- Page 0007 ---

```markdown
an extracted entity mention is not aligned to entities existing in the PKB. Therefore, an entity may be extracted as distinct mentions in different documents. Hence, for entity linking, KnowledGPT returns multiple matched entities. Third, an entity would be extracted with an aliases list, which would be provided to LLMs for entity linking.

For the get_entity_or_value function, since a relation can also be extracted as different expressions, we opt to retrieve relations with similarity score over a threshold, instead of a top-scored relation.

## 4 Experiments

In this section, we experiment KnowledGPT on various settings, including manually crafted diverse queries on popular KBs (Sec 4.2), knowledge-based question answering (Sec 4.3), and personalized KBs as memory for LLMs (Sec 4.4).

### 4.1 Experimental Setups

**Knowledge Bases.** KnowledGPT can access various KBs with its unified interface for KB operation. In this paper, we primarily consider the following KBs:

1. **Wikipedia** and **Wikidata**. Wikipedia provides rich encyclopedic information about world entities, maintained by global volunteers. Wikidata is a knowledge graph complementing Wikipedia, which structures and organizes this encyclopedic knowledge in relational triples.

2. **CN-DBPedia** (Xu et al., 2017) is a large-scale, constantly-updated Chinese KB extracted from various sources including Chinese Wikipedia and Baidu Baike. CN-DBPedia contains both entity descriptions like Wikipedia and relational triples like Wikidata.

3. **Personalized KB** is designed as a writable symbolic memory for LLMs. It stores upcoming knowledge extracted from user inputs.

4. **The NLPCC 2016 KBAQ Knowledge Base** (Duan, 2016), which is widely adopted to evaluate models in terms of knowledge-based question answering tasks. It contains 43 million triples. This KB is only used in Sec 4.3.

| LLM   | Direct Answer | Code | Entity Linking | KnowledGPT Answer |
|-------|---------------|------|----------------|--------------------|
| GPT-4 | 41/1          | 9/1  | 22/2           | 11/1               |
| ChatGPT | 41/1        | 8/1  | 13/19          | 4/11               |

Table 2: Number of successful answers on the selected samples. The results are evaluated by human annotators. For code generation, it is considered successful if the code is supposed to be able to retrieve all the necessary knowledge related.

To evaluate KnowledGPT in terms of diversified queries of real users requiring external knowledge, we craft 11 questions about knowledge from CN-DBPedia. These questions span a variety of types, including single-hop and multi-hop relational queries, relation prediction, diversified instructions, mixed queries, and value comparison. These questions concern both popular entities and long-tail entities. The detailed examples are shown in Sec B. We experiment on KnowledGPT with GPT-4 and ChatGPT as the base LLMs, and measure the rate of successful generation in terms of direct answers without KBs, code generation, entity linking, and KnowledGPT answers.

The results are shown in Table 2. For detailed generations, please refer to Sec B. We observe that: (1) GPT-4 and ChatGPT themselves are proficient at addressing queries about well-known entities, but they also hallucinate frequently about the unpopular entities. (2) KnowledGPT with GPT-4 excellently accomplishes tasks like code generation and

![Detailed description of the chart](assets/page_0007_img_1.png)
```

### --- Page 0008 ---

```markdown
![Case studies of KnowledGPT's entity linking process.](assets/page_0008_img_1.png)

## 4.3 Knowledge-Based Question Answering

We evaluate KnowledGPT on zero-shot knowledge-based question answering (KBQA). KBQA is an extensively researched domain that seeks to answer natural language questions specific to certain relational triples in KBs. For example, to answer the question “Who is the author of the Republic?”, a KBQA model is expected to retrieve the triple (the Republic, written by, Plato) and answer Plato.

Given the expenses of invoking OpenAI APIs, we have compiled two compact datasets, namely NLPC-C100 for single-hop queries and NLPC-MH-59 for multi-hop queries. NLPC-C-100 is com...
```

### --- Page 0009 ---

```markdown
3. SPE (Lai et al., 2016). SPE extracts subject-predicate pairs from simple questions using embedding similarity. It obtained the first place in the contest of NLPC 2016 KBQA task.

We report averaged F1 on NLPC-100 and NLPC-MH-59. Averaged F1 is a widely adopted metric for KBQA, designed for tasks with multiple golden answers and predictions. However, since in our dataset there is only one answer and one prediction for each sample, so the averaged F1 actually is equivalent to accuracy.

The results are shown in Table 3, from which we have the following observation: (1) For single-hop queries, KnowledGPT significantly outperforms retrieval methods via BM25 and embedding similarity, which shows the effectiveness of retrieval from symbolic Kbs compared with document corpus, for questions related to knowledge in Kbs. (2) Zero-shot KnowledGPT outperforms the SPE method trained on the full training set of NLPC-2016 KBQA dataset (0.92 vs 0.85), which shows the strong zero-shot performance of KnowledGPT. (3) For multi-hop queries, KnowledGPT also achieves excellent performance, while the performance of retrieval methods based on BM25 and embedding similarity drops significantly.

| Dataset        | BM25  | Embedding Similarity | SPE  | KnowledGPT |
|----------------|-------|----------------------|------|------------|
| NLPC-100       | 0.71  | 0.31                 | 0.83 | 0.92       |
| NLPC-MH-59     | 0.44  | 0.19                 | 0.93 | 0.93       |

Table 3: Averaged F1 of different methods on NLPC-100 and NLPC-MH-59.

4.4 KB as Memory  
We conduct experiments to study the effectiveness of KnowledGPT when paired with a modifiable personalized KB. KnowledGPT is tasked with building the PKB by extracting knowledge from provided documents, and study whether KnowledGPT answers corresponding questions properly with the PKB. Fig 6 shows an example where text about Socrates is provided. KnowledGPT extracts knowledge from the text, and answers related questions by retrieving the extracted knowledge.

We further apply KnowledGPT to HotpotQA (Yang et al., 2018), a multi-hop question answering dataset with provided documents. We select 25 questions from HotpotQA eval (distractor), including 5 comparison questions like “Which is a shrub, Mimosa or Cryptocurrency?” and 20 bridge questions like “When was Erik Watts’ father
```

### --- Page 0010 ---

```markdown
We further investigate the knowledge extraction coverage of KnowledGPT on 100 documents from HotpotQA (Yang et al., 2018), considering various LLMs and diverse combination of applied knowledge representations. To quantify the coverage, we employ the word recall rate, calculated as

$$
\frac{|W_{\text{extracted}}| \cdot |W_{\text{doc}}|}{|W_{\text{doc}}|},
$$

where $| \cdot |$ indicates the cardinality of a set. $W_{\text{extracted}}$ and $W_{\text{doc}}$ denote the set of words in the extracted knowledge and the document respectively, after preprocessing including the removal of stop words and lemmatization, utilizing the NLTK toolkit (Bird and Klein, 2009).

The results are shown in Table 5, from which we have the following observations: (1) When restricting knowledge representation solely to triples, the extraction coverage stands at 0.53, indicating that only a limited portion of knowledge can be represented as triples. Therefore, a PKB supporting triples alone falls short of adequately encompassing the knowledge provided by real users. (2) With additional knowledge representations, i.e., entity description and entity-aspect information, we observe a marked improvement in knowledge extraction coverage, suggesting that incorporating entity description and entity-aspect information enables KnowledGPT to populate the PKB with a broader spectrum of knowledge. (3) ChatGPT and GPT-4 achieve similar proficiency for knowledge extraction. GPT-4 outperforms ChatGPT only when entity-aspect info is included, which is probably attributed to GPT-4’s enhanced capability at following complex instructions.

![An example showing knowledge extraction and retrieval of KnowledGPT on the personalized KB.](assets/page_0010_img_1.png)

### 5 Limitations

While KnowledGPT enables LLMs to effectively perform KB operations on external knowledge bases, there remain several limitations in its current form. First, the retrieval process entails a single round of code generation and execution for efficiency concerns. However, a multi-round mecha...

| Comparison | Bridge | All |
|------------|--------|-----|
| KnowledGPT | 5/5    | 15/20 | 20/25 |

Table 4: Number of successful answers on 25 questions selected from HotPotQA. The results are evaluated by human annotators.

| $ \text{triples only} $ | $ \text{entity desc} $ | $ \text{++ entity aspect info} $ |
|--------------------------|------------------------|----------------------------------|
| ChatGPT                  | 0.53                   | 0.66                             | 0.81 |
| GPT-4                    | 0.53                   | 0.62                             | 0.86 |

Table 5: Knowledge extraction coverage of KnowledGPT on 100 documents from HotpotQA with different LLMs and various combination of applied knowledge representations.
```

### --- Page 0011 ---

```markdown
# KnowledgeGPT: A Comprehensive Framework for Integrating LLMs with External Knowledge Bases

## 6 Conclusion

In this paper, we introduce KnowledGPT, a comprehensive framework to integrate LLMs with external knowledge bases, facilitating LLMs’ retrieval and storage on KBs. For retrieval, KnowledGPT adopts “program of thought” prompting, which retrieves knowledge via code generation and execution. For storage, KnowledGPT extracts various forms of knowledge from user provided texts, and populate the personalized KB with the extracted knowledge. KnowledGPT tackles several challenges inherent in integrating LLMs with KBs, including complex question answering, ambiguitiy in entity linking, and limited forms of knowledge representations. We show with extensive experiments that KnowledGPT effectively provides LLMs with the capability to operate on external KBs.

## References

| Author(s) | Year | Title |
|-----------|------|-------|
| Ibrahim Abdelaziz, Srinivas Ravishankar, Pavan Kapanipathi, Salim Rukos, and Alexander Gray. | 2021 | A semantic parsing and reasoning-based approach to knowledge base question answering. Proceedings of the AAAI Conference on Artificial Intelligence, 35(18):15985–15987. |
| Jinhon Baek, Alham Fikri Aji, and Amir Saffari. | 2023 | Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. arXiv preprint arXiv:2306.04136. |
| Jonathan Berant, Andrew K. Chou, Roy Frostig, and Percy Liang. | 2013 | Semantic parsing on freebase from question-answer pairs. In Conference on Empirical Methods in Natural Language Processing. |
| Edward Loper Bird, Steven and Evan Klein. | 2009 | Natural language processing with python. |
| Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafula Dhariwal, Arvind Neelakantan, Pranav Sharma, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. | 2020 | Language models are few-shot learners. CoRR, abs/2005.14165. |
| Shulin Cao, Jiaxin Shi, Zijun Yao, Lin Yu, Jifan Yu, Lei Hou, Juanzi Li, Zhiyuan Liu, and Jinghui Xiao. | 2022 | Program transfer for answering complex questions over knowledge bases. |
| Harrison Chase. | 2022 | LangChain. |
| Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. | 2022 | Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588. |
| Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Leon Stoica, and Eric P. Xing. | 2023 | Vicuna: An open-source chatbot impressing gpt-4 with 90%+ chatgpt quality. |
| Nan Duan. | 2016 | Overview of the nlpcc-2016 shared task: Open domain chinese question answering. In Natural Language Understanding and Intelligent Applications, pages 942–948. Springer International Publishing. |
| Yu Gu, Xiang Deng, and Yu Su. | 2022 | Don’t generate, discriminate: A proposal for grounding language models to real-world environments. arXiv preprint arXiv:2212.09736. |
| Kelvin Gu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. | 2020 | Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR. |
```


### --- Page 0012 ---

```markdown
| Author(s)                                                                 | Title                                                                                                   | Source                                                                                     |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu.                      | Toolknapp: Augmenting frozen language models with massive visual tool embeddings.                        | arXiv preprint arXiv:2305.11554.                                                          |
| Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo Zhao, and Hang Zhao. | Augmenting LLMs with databases as their symbolic memory.                                                | arXiv preprint arXiv:2306.03901.                                                          |
| Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen.                 | Unikgg: Unified retrieval and reasoning for solving multi-hop question answering over knowledge graph. | arXiv preprint.                                                                            |
| Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. | Large language models struggle to learn long-tail knowledge.                                            | In International Conference on Machine Learning, pages 15696–15707. PMLR.                |
| Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. | Large language models are zero-shot reasoners.                                                          | Advances in neural information processing systems, 35:22199–22213.                        |
| Mojtaba Komeili, Kurt Shuster, and Jason Weston.                         | Internet-augmented dialogue generation.                                                                 | In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8460–8478, Dublin, Ireland. Association for Computational Linguistics. |
| Yuxuan Lai, Yang Lin, Jiahao Chen, Yansong Feng, and Dongyan Zhao.      | Open domain question answering system based on knowledge base.                                          | In Natural Language Understanding and Intelligent Applications: 5th CCF Conference on Natural Language Processing and Chinese Computing, NLPCC 2016, and 24th International Conference on Computer Processing of Oriental Languages, ICOOL 2016, Kunming, China, December 2–6, 2016, Proceedings 24, pages 723–733. Springer. |
| Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. | Internet-augmented language models through few-shot prompting for open-domain question answering.       | arXiv preprint arXiv:2203.05115.                                                          |
| Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpuinkin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. | Retrieval-augmented generation for knowledge-intensive NLP tasks.                                      | Advances in Neural Information Processing Systems, 33:9459–9474.                          |
| Chen Liang, Jonathan Berant, Quc Liu, Kenneth D. Forbus, and Ni Lao.    | Neural symbolic machines: Learning semantic parsers on freebase with weak supervision.                  | arXiv preprint arXiv:1611.00020.                                                          |
| Jerry Liu.                                                               | Llamalnkd.                                                                                             |                                                                                            |
| Xu Ming.                                                                 | text2vec: A tool for text to vector.                                                                    |                                                                                            |
| Ali Modarressi, Ayyoub Imani, Mohsen Fayyaz, and Hinrich Schütze.       | Rest-llm: Towards a general read-write memory for large language models.                               | arXiv preprint arXiv:2305.14322.                                                          |
| OpenAI.                                                                  | Gpt-4 technical report.                                                                                 |                                                                                            |
| Eric Prud'hommeaux.                                                      | Sparql query language for rdf.                                                                          |                                                                                            |
| Stephen Robertson, S. Walker, S. Jones, M. M. Hancock-Beaulieu, and M. Gattord. | Okapi at trec-3.                                                                                      | In Overview of the Third Text REtrieval Conference (TREC-3), pages 109–126. Gaithersburg, MD: NIST. |
| Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. | Toolformer: Language models can teach themselves to use tools.                                         | arXiv preprint arXiv:2302.04761.                                                          |
| Jiaxin Shi, Shulin Cao, Lei Hou, Juanzi Li, and Hang Zhang.             | Transfernet: An effective and transparent framework for multi-hop question answering over relation graph. |                                                                                            |
| Haitain Sun, Tania Bedard-Weiss, and William M. Cohen.                  | Pollnet: Open domain question answering with iterative retrieval over knowledge bases and text.         |                                                                                            |
| Yawei Sun, Lingling Zhang, Gong Cheng, and Yuzhong Qu.                  | Sparq: Skeleton-based semantic parsing for complex questions over knowledge bases.                      | arXiv preprint arXiv:2303.05977.                                                          |
| Rohan Taori, Ishaan Gulrajani, Tiziana Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. | Alpacaa: A strong, replicable instruction-following model.                                            | Stanford Center for Research on Foundation Models. https://crfm.stanford.edu/2023/03/31/alpacaa.html, 3(6):7. |
| Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Édouard Grave, and Guillaume Lample. | Llama: Open and efficient foundation language models.                                                  | arXiv preprint arXiv:2302.13971.                                                          |
| Yan ton Sonsbeek, Mohammad Mahdi Derakhshani, Ivona Najdenkoska, Cees GM Snoek, and Marcel Worring. | Open-ended medical visual question answering through prefix tuning of language models.                  | arXiv preprint arXiv:2303.05977.                                                          |
| Longyue Wang, Chenyang Liu, Tianbo Ji, Zhirui Zhang, Dian Yu, Shuming Shi, and Zhaopeng Tu. | Document-level machine translation with large language models.                                          | arXiv preprint arXiv:2302.02210.                                                          |
| Yue Wang and Richong Zhang.                                             | A dynamic programming-based approach to knowledge-based question answering.                             | Journal of Zhengzhou University (Science Edition), 51(4):37–42.                          |
```

### --- Page 0013 ---

```markdown
Bo Xu, Yong Xu, Jiaqing Liang, Chenhao Xie, Bin Liang, Wanyun Cui, and Yanghua Xiao. 2017. Cndbpedia: A never-ending chinese knowledge extraction system. In *International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems*, pages 428–438. Springer.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380. Brussels, Belgium. Association for Computational Linguistics.

Xuchen Yao and Benjamin Van Durme. 2014. Information extraction over structured data: Question answering with freebase. In *Proceedings of the 52nd annual meeting of the association for computational linguistics (volume 1: long papers)*, pages 956–966.

Donghan Yu, Sheng Zhang, Patrick Ng, Henghui Zhu, Alexander Hanbo Li, Jun Wang, Yiqun Hu, William Wang, Zhiguo Wang, and Bing Xiang. 2023. Defrad: Joint decoding of answers and logical forms for question answering over knowledge bases.

Jiawei Zhang. 2023. Graph-toolformer: To empower LLMs with graph reasoning ability via prompt augmented by chatgpt. arXiv preprint arXiv:2304.11116.

Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, and Hong Chen. 2022a. Subgraph retrieval enhanced model for multi-hop knowledge base question answering. In *Proceedings of the 60th Annual Meeting of Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics.

Minhao Zhang, Ruoyu Zhang, Yanzeng Li, and Lei Zou. 2022b. Crake: Causal-enhanced table-filler for question answering over large scale knowledge base.

Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori B Hashimoto. 2023. Benchmarking large language models for news summarization. arXiv preprint arXiv:2301.13848.

## A Prompts

The prompts are shown in Fig 7, Fig 8, Fig 9, Fig 10. These prompts primarily consist of task introductions, requirements, the format of input and output, and in-context examples. We request LLMs to output in JSON format, and in the same language as the query. The prompts emphasize the LLMs should finish special tasks, instead of directly answering the input questions. They are mostly written in English, with some in-context examples in Chinese to better support queries in Chinese. To ease understanding, these examples are translated into English in these figures.

**Prompt for search code generation** first asks LLMs to judge whether the query need knowledge from external KBs. If so, it continues to generate the search code. It pre-defines three functions that LLMs can generate to access KBs. It asks LLMs to generate a list of entity aliases or relation synonyms as input to these functions.

**Prompt for entity linking** directs LLMs to select from multiple candidate entities for the input query and target entity, provided with information of the candidate entities. While LLMs have the option to output [NONE] if none seems related, they are advised to use this option cautiously.

**Prompt for question answering** first guides LLMs to assess if the provided knowledge adequately supports answering this question. If so, it asks LLMs to answer the query with the retrieved knowledge. Otherwise, we let LLMs answer this question independently.

**Prompt for knowledge extraction** prompts LLMs to extract knowledge from the provided text. For long texts, LLMs tend to overlook many pieces of information during the extraction process. We find that emphasizing “Do not miss any knowledge points” largely improves the knowledge extraction coverage. Additionally, the prompt encourages LLMs to present knowledge in the form of relational triples and entity-aspect information when possible, instead of entity description.

## B Detailed Results of Queries on Popular KBs

In this section, we elaborate the detailed results of the manually crafted examples mentioned in Sec 4.2. The examples are mainly in Chinese, including their queries, retrieval results, and answers. We color incorrectly generated content that in red, including false facts or illogical code. We also color unhelpful answers in brown, where GPT-4 and ChatGPT admit their ignorance. In the two cases about value comparison, GPT-4 attempts to conduct value comparison in the code, which seems logical. However, it actually performs string comparison, which makes the results unreliable without improved value comparing functions that can take into account units and unit conversion.
```

### --- Page 0014 ---

```markdown
# Prompt for Search Code Generation:

You are an awesome knowledge retrieval assistant agent that helps to RETRIEVE related knowledge about user queries with python codes to access external knowledge sources. Your python codes should implement a search function using exclusively built-in python functions and the provided functions listed below.

## ===PROVIDED FUNCTIONS===

1. `get_entity(entity)`: Obtain encyclopedic information about an entity from external sources, which is used to answer general queries like "Who is Steve Jobs?", "Ang-entity_aliases": a list of the entity's aliases, e.g., [American], [United States], [U.S.] for the entity "American". Return two strings: `result` and `message`. `result` is the encyclopedic information about the entity if retrieved. Otherwise, `message` states this function call and its result.
2. `find_entity_or_value(entity, value)`: Access knowledge graphs to answer factual queries like "Who is the founder of Microsoft?". `args`: "entity", "aliases": a list of the entity's aliases, `relation_aliases`: a list of the relation's aliases. Return two variables, `result` and `message`. `result` is a list of any terms or attribute value to this query if retrieved. Note otherwise, `message` is a string that states this function call and its result.
3. `find_relationship`: Access knowledge graphs to predict the relationship between two entities, where the input query is like "What's the relationship between Steve Jobs and Apple Inc?". Args: "entity", "aliases": a list of entity's aliases, "entity2", "aliases": a list of entity2's aliases. Return two strings, `result` and `message`: `result` is the relationship between entity1 and entity2 if retrieved, None otherwise. `message` states this function call and its result.

## ===REQUIREMENTS===

1. **[IMPORTANT]** Always remember that your task is to retrieve related knowledge instead of answering the queries directly. Never try to directly answer your user in any form. Do not include your answer in your generated output.
2. Exclusively use built-in python functions and the provided functions.
3. To better retrieve the intended knowledge, you should make necessary paraphrase and list several candidate aliases for entities and relations when calling the provided functions, sorted by the frequency of the alias. E.g., "Where is Donald Trump born" should be paraphrased as `find_entity_or_value("Donald Trump", "President Trump", "[place of birth]", "is born in")`. Avoid entity aliases that may refer to other entities, such as "Trump" or "Donald Trump".
4. When using `find_entity_or_value`, make sure the relation is a clear relation. Avoid vague and broad relation aliases like "information". Otherwise, use exact entity info instead. For example, for the question "Who is related to the Battle of Waterloo?", you should use `get_entity`, `find_entity_or_value`, relation_aliases = ["the Battle of Waterloo"] instead of `find_entity_or_value`, relation_aliases = ["related to"] since it's too vague to be searched.
5. The input can be in both English and Chinese. If the input language is not English, make sure the args of `get_entity`, `find_entity_or_value` and `find_relationship` is in the input language.
6. The queries may need multiple or nested searching. Use smart python codes to deal with them. Note that `find_entity_or_value` will return a list of results.

7. This step by step. Firstly, you should determine whether the user input is a query that "need knowledge". If not, simply generate "no" and stop. Otherwise, generate "yes", and go through the following steps: First, Come up with a "thought" about how to find the knowledge related to the query step by step. Make sure your "thought" covers all the entities mentioned in the input. Then, implement your "thought" into "code", which is a python function with return. After that, make an "introspection" whether your "code" is problematic, including whether it can solve the query, can be executed, and whether it contradicts the requirements (especially whether it sticks to the RETRIEVE task or mistakenly tries to answer the question). Make sure "thought" and "introspection" are also in the same language as the query. Finally, set "ok" as "yes" if no problem exists, and "no" if your "introspection" shows there is any problem.

8. For every call of `get_entity`, `find_entity_or_value` and `find_relationship`, the return `message` after recording into a string named `messages`, which is the return value of `search()`.
9. Add necessary explanation to the `messages` variable after running certain built-in python codes, such as `messages += f"{top_teacher} is the teacher with most citations"`.
10. When the user query contains constraints like "first", "highest" or mathematical operations like "average", "sum", handle them with built-in functions.

11. Observe JSON format.

## ===OUTPUT FORMAT===

```json
{
  "need_knowledge": "yes or no. If no, stop generating the following.",
  "thought": "<Your thought here. Think how to find the answer to the query step by step. List possible aliases of entities and relations.>",
  "code": "def search():\\n\\tmessages = \"\"\\t<your code here. Implement your thought.>\\n",
  "introspection": "<Your introspection here.>",
  "ok": "yes or no"
}
```

## ===EXAMPLES===

1. Input: "Who are you?"
   Output:
   ```json
   {
     "need_knowledge": "no",
     "thought": "",
     "code": "",
     "introspection": "",
     "ok": "yes"
   }
   ```

2. Input: "Who proposed the theory of evolution?"
   Output:
   ```json
   {
     "need_knowledge": "yes",
     "thought": "The question is asking who proposed the theory of evolution. I need to search for the proponent of the theory of evolution. The possible expressions for the 'proponent' relationship include 'proposed', 'proponent', and 'discovered'.",
     "code": "def search():\\n\\tmessages = \"\"\\tmrg = find_entity_or_value(entity_aliases = ['theory of evolution'], relation_aliases = ['propose', 'proponent', 'discover'])\\tmessages += mrg + messages\\n\\treturn messages\\n",
     "introspection": "The generated code meets the requirements.",
     "ok": "yes"
   }
   ```

3. Input: "What is one of the stars of 'The Newcomers' known for?"
   Output:
   ```json
   {
     "need_knowledge": "yes",
     "thought": "To answer this question, firstly we need to find the stars of 'The Newcomers'. The relation can be paraphrased as 'star in', 'act in' or 'cast in'. Then, we should select one of them. Finally, we should retrieve its encyclopedic information to know what he or she is known for. We should not treat 'known for' as a relation because it is too vague.",
     "code": "def search():\\n\\tmessages = \"\"\\tstars_mrg = find_entity_or_value(entity_aliases = ['The Newcomers'], relation_aliases = ['star in', 'act in'])\\tstar = random.choice(stars_mrg)\\tinfo_msg = get_entity_info(entity_aliases = [star])\\tmessages += info_msg\\treturn messages\\n",
     "introspection": "The generated code is executable and matches user input. It adheres to the requirements. It finishes the retrieve task instead of answering the question directly.",
     "ok": "yes"
   }
   ```

![Prompt for search code generation. The second example is in Chinese and translated into English.](assets/page_0014_img_1.png)
```

### --- Page 0015 ---

```markdown
![Prompt for entity linking. The second example is in Chinese and translated into English.](assets/page_0015_img_1.png)

| **Prompt for Entity Linking:**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| You are an awesome knowledge graph accessing agent. There are many entities with similar names that exist in knowledge graphs which cause ambiguity, such as the fruit 'apple' and the company 'Apple'. Given the user input, and the interested entity mention in it, you are provided with some candidate entities and their information. Now, your task is to consider carefully which of the candidate entities matches the entity mention in user input.                                                                 |
| **NOTICE**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 1. The user input and entity information can be in English or Chinese.                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2. If all the candidate entities are irrelevant to the input entity mention, reply [None]. However, you should not reply [None] simply because the provided entity information do not directly answer the question. If the entity mention clearly matches a candidate entity, you should definitely return it.                                                                                                                                                     |
| 3. If multiple candidates are possible, just choose one that you think is most possible.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 4. When the user input is an assumption or question, do not think entities are irrelevant to the input simply because their information cannot cover the assumption or question. Use your imagination how the assumption or question can be related with the candidate entities.                                                                                                                                                                                                 |
| 5. **IMPORTANT:** Always remember that your task is to select the correct entity instead of answering the questions. Never try to directly answer user input in any form. Always reply [ENT 1], [ENT 2], ... [ENT n] or [NONE].                                                                                                                                                                                                                                                                                     |
| 6. Try your best to ensure the entity you choose is equivalent to the input target entity. They should belong to the same type. For example, if the input target entity is William "William Shakespeare", you shouldn't choose "William Shakespeare's plays" as your answer, since the former is a person while the latter are works.                                                                                                                                                |
| **INPUT FORMAT**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| You are provided with the [USER INPUT], [THE TARGET ENTITY] of the target entity mention, and several candidate entities like [ENT 1], [ENT 2], ... [ENT n].                                                                                                                                                                                                                                                                                                                                                       |
| **OUTPUT FORMAT**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| In order to find the correct entity, you should think step by step, and output in join format. First, you should generate your 'thought' considering the user input, the target entity mention, and all the candidate entities. Do not directly answer user query in your thought. Then, output your 'choice', which is the entity that matches the entity mention like [ENT 1], [ENT 2], ... [ENT n] or [NONE] if there is none.                                                                                          |
| **EXAMPLES**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Input:** [USER INPUT]: Which Nobel laureate in Literature is best known for 'Blowing in the Wind'? <br> [TARGET ENTITY]: Blowing in the Wind <br> [ENT 1]: Blowing in the Wind (04298603): album by Lou Donaldson. <br> [ENT 2]: Blowing in the Wind (Q15219932): television series. <br> [ENT 3]: Blowing in the Wind (Q95044759): scientific article published in Nature. <br> **Output:** { <br> "thought": "None of the album, tv series and scientific article seems related with Nobel Prize in Literature.", <br> "choice": "[None]" <br> } |
| 2.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Input:** [USER INPUT]: Please introduce the academic achievements of Liang Jiajing. <br> [TARGET ENTITY]: Liang Jiajing <br> [ENT 1]: Liang Jiajing: Liang Jiajing, also known as Lu Yuan, is a member of the Chinese Communist Party. He was born after the 1960s and has a university education. He is a specially appointed writer for "Chinese Writers" magazine and "Chinese Reportage Literature" magazine. Attributes: Category -> Cultural figure, Author -> The Loyal Life of a Criminal Police Captain. <br> [ENT 2]: Liang Jiajing (Scholar at Fudan University): Liang Jiajing graduated from the School of Computer Science at Fudan University and holds a Ph.D. degree. He is a well-known scholar in the field of knowledge graph and natural language processing. Attributes: Graduated from -> Fudan University. <br> **Output:** { <br> "thought": "The user wants to know about the academic achievements of Liang Jiajing, so here Liang Jiajing refers to a scholar, matching with [ENT 2].", <br> "choice": "[ENT 2]" <br> } |
```

### --- Page 0016 ---

```markdown
# Prompt for Question Answering:

You are an helpful and knowledgeable AI assistant. The user has issued a query, and you are provided with some related knowledge. Now, you need to think step by step to answer the user input with the related knowledge.

## REQUIREMENTS
1. You should think step by step. First, think carefully whether you can answer this query without the provided knowledge. Second, consider how to use the related knowledge to answer the query. Then, tell me whether this query can be answered with your own knowledge and the provided knowledge. If so, answer this question. However, if the query involves a command or an assumption, you should always regard it as answerable.
2. When you are thinking, you can use and cite the provided knowledge. However, when you are generating the answer, you should pretend that you came up with the knowledge yourself, so you should not say things like "according to the provided knowledge from ... " in the "answer" part.
3. The user query and provided knowledge can be in both Chinese and English. Generate your "thought" and "answer" in the same language as the input.
4. Response in JSON format, use double quotes.

### INPUT FORMAT
```json
{
  "query": "<the user query that you need to answer>",
  "knowledge": "<the background knowledge that you are provided with>"
}
```

### OUTPUT FORMAT
```json
{
  "thought": "<Your thought here. Think step by step as required>",
  "answerable": "<'yes' or 'no'. Whether you can answer this question with your knowledge and the provided knowledge. If the query involves a command or an assumption, say 'yes'>",
  "answer": "<Your answer here, if the query is answerable>"
}
```

### EXAMPLES

**Input 1:**
```json
{
  "query": "What is the motto of the school where Xia Mingyou graduated?",
  "knowledge": "FROM CNDBpedia|find_entity_by_value|entity_aliases=['Xia Mingyou'],relation_aliases=['graduated from','school'] -> Xia Mingyou, school: Fudan University|find_entity_by_value|entity_aliases=['Fudan University'],relation_aliases=['motto'] -> Fudan University, motto: Rich in Knowledge and Tenacious of Purpose; Inquiring with Earnestness and Reflecting with Self-practice"
}
```
**Output 1:**
```json
{
  "thought": "Based on the background knowledge from CNDBpedia, Xia Mingyou graduated from Fudan University, and the motto of Fudan University is 'Rich in Knowledge and Tenacious of Purpose; Inquiring with Earnestness and Reflecting with Self-practice'. This question can be answered based on the provided knowledge.",
  "answerable": "yes",
  "answer": "Rich in Knowledge and Tenacious of Purpose; Inquiring with Earnestness and Reflecting with Self-practice"
}
```

**Input 2:**
```json
{
  "query": "What is Liang Jiaging's weapon?",
  "knowledge": "FROM CNDBpedia|Liang Jiaging: Liang Jiaging, also known as Lu Yuan. A member of the Chinese Communist Party, born after the 1960s, with a university education. Specially appointed writer for 'Chinese Writers' magazine and 'Chinese Reportage Literature' magazine. Attributes: Author -> The Loyal Life of a Criminal Police Captain."
}
```
**Output 2:**
```json
{
  "thought": "According to the knowledge provided by CNDBpedia, Liang Jiaging is an author. The provided knowledge does not mention anything about Liang Jiaging's weapon, and authors generally do not have weapons. The question cannot be answered based on the provided knowledge or my knowledge.",
  "answerable": "no"
}
```

![Prompt for question answering. The examples are in Chinese and translated into English.](assets/page_0016_img_1.png)
```

### --- Page 0017 ---

```markdown
# Prompt for Knowledge Extraction:

You are an awesome information extraction agent and your task is to extract important pieces of information (knowledge) in a structured form from unstructured data, which will be stored to build a knowledge base memory.

You should extract the following three types of knowledge:

1. **Entity description**:
   Format: `"<entity description in the encyclopedia style>"`
   You should extract entity description, which is the most critical information about an entity, as concise as possible, and in the encyclopedia style. For example, `("Elon Reeve Musk", "Elon Reeve Musk (born June 28, 1971) is a business magnate and investor...")`

2. **Relational triple**:
   Format: `["<subject>", "<predicate>", "<object>"]`
   You should extract relational triples (a.k.a. factual triples), a common type of knowledge widely used to build knowledge graphs. The `<subject>` must be an entity in natural language, e.g., "United States". The predicate is an attribute or a relationship, e.g., "population" or "located in". Attributes normally refer to the possible attributes, features, characteristics, and parameters an entity. Relationships connect two entities and characterize their relationships. The `<object>` is an attribute value or another entity, such as "over 333 million" in "[United States]", "population", "over 333 million", or "North America" in "[United States]", "located in", "North America", respectively.

3. **Entity-Aspect-Content**:
   Format: `{"entity": "<aspect>", "content": "<question>"}`
   You should also extract textual information that describes an entity in a certain aspect. This differs from relational triple in that the content describes an entity in a certain aspect, besides creating a question about the aspect of the entity. For example, you should extract `{"Elon Reeve Musk": "biography", "Musk was born in Pretoria, South Africa, and briefly attended the University of Pretoria before moving to Canada at age 18, acquiring citizenship through his Canadian-born mother."}` to describe "Elon Reeve Musk" in terms of his "biography". The generated question can be "What is Elon Reeve Musk's biography?".

## REQUIREMENTS
1. **(IMPORTANT)** Always remember that your task is to extract information from a given corpus. Never try to answer a query or a command. More importantly, only extract information from the text, and never generate things not mentioned in the text.
2. There can be overlapped information among the three types of extracted knowledge. The same piece of information can be appropriately organized into two or more formats. For example, the corpus "Elon Reeve Musk is born on June 28, 1971" can be organized into an entity encyclopedic description like `("Elon Reeve Musk", "Elon Reeve Musk is born on June 28, 1971")`, and meanwhile, the corpus can also be organized into a relational triple like `("Elon Reeve Musk", "born on", "June 28, 1971")`.
3. The extracted entity description should be the most critical information about the entity, in the encyclopedia style.
4. When extracting knowledge in the relational triple format of `["<subject>", "<predicate>", "<object>"]`, make sure the predicate is a clear attribute of relation. Avoid broad predicates like "information", "related to". Instead, put the information into the entity description format if it's a piece of encyclopedia-like information about the entity, or put the information into the entity-aspect-content format if you can summarize a specific aspect for the piece of information.
5. The input can be in both English and Chinese. If most of the input is written in English, make sure the output is in English. If most of the input is written in Chinese, make sure the output is in Chinese.
6. Think step by step. Firstly, you need to understand what the given corpus is mainly about and then decide on the core entities you plan to extract from this corpus. Next, iterate through each entity, coming up with a "thought" about how to extract information about this core entity and accordingly generate the generated "thought", extracting "entity description", "relational triple", "entity aspect content" for each entity in turn. Specifically, the "thought" field should give details on how to fill in relevant information about the core entity from the corpus, and how to choose the appropriate formats to organize these pieces of information according to their characteristics. Make sure your "thought" covers all the relevant entities mentioned in the input.
7. The entity names should be concise, yet accurately represent the entity.
8. Extract as many relational triples as possible from input text.
9. Response is in JSON format.

## OUTPUT FORMAT
The output format is a nested JSON object. The top-level json object has two fields named "thought" and "knowledge" respectively. The content of the "thought" is a string; the content of the "knowledge" is another json object whose keys are core entities. You can extract one or more core entities from the corpus, e.g., core entity 1, core entity 2, etc. The value of the core entity 1 is another json object, whose keys are "entity", "description", "relational_triple", "entity_aspect_content". The value of "entity_description" is a piece of encyclopedia text. The value of "relational_triple" is a list of relational triples whose subject-or-object is the core entity 1. The value of "entity_aspect_content" is a list of `{"entity": "<aspect>", "content": "<question>"} ` questions whose entity is the core entity 1.

### Format:
```json
{
  "thought": "Your thought here. Determine the language of the input text and output in that language. Understand what the given corpus is mainly about and then decide on the core entities you plan to extract from this corpus. For each core entity, you should generate a thought about how to extract information about this core entity and choose the appropriate formats to organize these pieces of information according to their characteristics.",
  "knowledge": {
    "<core entity 1>": {
      "entity": "<entity description>",
      "relational_triple": ["<core entity 1>", "<predicate 1>", "<object 1>"],
      "entity_aspect_content": [{"entity": "<core entity 1>", "aspect": "<aspect 1>", "content": "<question 1>"}]
    },
    "<core entity 2>": {
      "entity": "<entity description>",
      "relational_triple": ["<core entity 2>", "<predicate 2>", "<object 2>"],
      "entity_aspect_content": [{"entity": "<core entity 2>", "aspect": "<aspect 2>", "content": "<question 2>"}]
    }
  }
}
```

## EXAMPLES
1. 
Input: "The Orenda OE600 is a 680 hp-class liquid-cooled B-cylinder V-block ... "
Output:
```json
{
  "thought": "The Orenda OE600 is a 680 hp-class liquid-cooled B-cylinder V-block aircraft engine designed by ...",
  "knowledge": {
    "Orenda OE600": {
      "entity": "The Orenda OE600 is a 680 hp-class liquid-cooled B-cylinder V-block aircraft engine intended to re-introduce ...",
      "relational_triple": ["Orenda OE600", "type", "680 hp-class liquid-cooled B-cylinder V-block aircraft engine"],
      "entity_aspect_content": [{"entity": "Orenda OE600", "aspect": "design purpose", "content": "What is the design purpose of Orenda OE600?"}]
    }
  }
}
```
```
![Prompt for knowledge extraction](assets/page_0017_img_1.png)
```

### --- Page 0018 ---

```markdown
| Input: What is the registered capital of Dong Wu Securities?                                                                 | Retrieved Knowledge                                                                                                                                                                                                 | Answer                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Model         | Search Code:                                                                                                         |                                                                                                                                                            |                                                                                              |
|               | GP14:                                                                                                                | messages = ""<br>capital_msg = find_entity_or_value(entity_aliases = ['Dong Wu Securities'], relation_aliases = ['Registered Capital', 'Capital'])<br>messages += msg<br>return messages                     |                                                                                              |
|               | ChatGPT:                                                                                                             | messages = ""<br>capital_msg = find_entity_or_value(entity_aliases = ['Dong Wu Securities'], relation_aliases = ['Registered Capital', 'Capital'])<br>messages += msg<br>return messages                     | [FROM CNDBpedia|find_entity_or_value|entity_aliases = ['Dong Wu Securities'], relation_aliases = ['Registered Capital', 'Capital'] -> Dongwu Securities, Registered Capital: 1.5 billion Yuan] |
| Input: Which historical hero is Saber?                                                                                     | Retrieved Knowledge                                                                                                                                                                                                 | Answer                                                                                       |
| Model         | Search Code:                                                                                                         |                                                                                                                                                            |                                                                                              |
|               | GP14:                                                                                                                | messages = ""<br>historical_figure_msg = find_entity_or_value(entity_aliases = ['Saber'], relation_aliases = ['historical character', 'historical archetype'])<br>messages += msg<br>return messages         |                                                                                              |
|               | ChatGPT:                                                                                                             | messages = ""<br>hero_msg = get_entity_info(entity_aliases = ['Saber'], relation_aliases = ['Historical heroes', 'Historical figures'])<br>messages += msg<br>return messages                               | [FROM CNDBpedia|find_entity_or_value|entity_aliases = ['Saber'], relation_aliases = ['Historical character', 'Historical archetype'] -> Artoria Pendragon. Saber is a character from the Japanese anime 'Fate/stay night', Born: Ancient Britannia] |
| Input: Please write a resume for Prof. Sun Maosong that has a clear format and layout.                                     | Retrieved Knowledge                                                                                                                                                                                                 | Answer                                                                                       |
| Model         | Search Code:                                                                                                         |                                                                                                                                                            |                                                                                              |
|               | GP14:                                                                                                                | messages = ""<br>info_msg = get_entity_info(entity_aliases = ['Sun Maosong', 'Professor Sun Maosong'])<br>messages += msg<br>return messages                                                                |                                                                                              |
|               | ChatGPT:                                                                                                             | messages = ""<br>info_msg = get_entity_info(entity_aliases = ['Prof. Sun Maosong'])<br>messages += msg<br>return messages                                                                                   | [FROM CNDBpedia|get_entity_info|entity_aliases = ['Prof. Sun Maosong'] -> Sun Maosong (Secretary of the Party Committee of the Department of Computer Science, Tsinghua University); Sun Maosong, Professor, Doctoral Supervisor, was the head of the Department of Computer Science and Technology of Tsinghua University...] |
| Table 6: The detailed results of single-hop relational queries and a diversified instruction on CN-DBPedia.                |                                                                                                                                                                                                                    |                                                                                              |
```

### --- Page 0019 ---

```markdown
| Input: What is the main ingredient in 'Ants on a Tree'? Who is the voice of Saber? | Retrieved Knowledge | Answer |
| --- | --- | --- |
| Model Search Code |  |  |
| GPT-4 |  |  |
| messages = "" |  |  |
| ingredients, msg = find_entity_or_value(entity_aliases = ['Ants on a Tree'], relation_aliases = ['main ingredients', 'toppings']) |  |  |
| messages += msg |  |  |
| voice_actor, msg = find_entity_or_value(entity_aliases = ['Saber'], relation_aliases = ['dub', 'voice-actor']) |  |  |
| messages += msg |  |  |
| return messages |  |  |
| ChatGPT search: |  |  |
| messages = "" |  |  |
| main_ingredient, msg = find_entity_or_value(entity_aliases = ['Ants on a Tree'], relation_aliases = ['main ingredient', 'main ingredient']) |  |  |
| messages += msg |  |  |
| voice_actor, msg = find_entity_or_value(entity_aliases = ['Saber'], relation_aliases = ['who the voice is', 'voice actor']) |  |  |
| messages += msg |  |  |
| return messages |  |  |
|  | FROM CNDBpedia[ind_entity_or_value], relation_aliases = ['Ants on a Tree'], relation_aliases = ['Main Ingredients'] | W/o KB |
|  |  | The main ingredient in Ants on a Tree is stir-fried vermicelli with minced pork, and Saber is voiced by Ayako Kawasumi. |
|  | FROM CNDBpedia[ind_entity_or_value], relation_aliases = ['Main Ingredients'] | W/o KB |
|  |  | Ants on the Tree is a traditional Sichuan dish whose main ingredients are minced pork and vermicelli. Regarding the voice of Saber, Saber is a character in the Fate/stay night series, and the voice actors for Saber vary depending on the production and version. Some of the more well-known voice actors are Ayako Kawasumi and Maya Sakamoto. |

| Input: What are the titles of the poet writing 'Quiet Night Thoughts' (Jing Ye Si)? | Retrieved Knowledge | Answer |
| --- | --- | --- |
| Model Search Code |  |  |
| GPT-4 |  |  |
| messages = "" |  |  |
| msg = find_entity_or_value(entity_aliases = ['Quiet Night Thoughts'], relation_aliases = ['author', 'creator', 'writer']) |  |  |
| messages += msg |  |  |
| titles, msg = find_entity_or_value(entity_aliases = ['author'], relation_aliases = ['title', 'also known as', 'appellation']) |  |  |
| messages += msg |  |  |
| return messages |  |  |
| ChatGPT search: |  |  |
| messages = "" |  |  |
| poet, msg = find_entity_or_value(entity_aliases = ['Quiet Night Thoughts'], relation_aliases = ['author of "Quiet Night Thoughts"', 'creator of "Quiet Night Thoughts"']) |  |  |
| messages += msg |  |  |
| titles, msg = find_entity_or_value(entity_aliases = ['title'], relation_aliases = ['title']) |  |  |
| messages += msg |  |  |
| return messages |  |  |
|  | FROM CNDBpedia[ind_entity_or_value], relation_aliases = ['Quiet Night Thoughts'], relation_aliases = ['author of "Quiet Night Thoughts"', 'creator of "Quiet Night Thoughts"'] | W/o KB |
|  |  | The poet of "Quiet Night Thoughts" is Li Bai. |
|  | FROM CNDBpedia[ind_entity_or_value], relation_aliases = ['title'], relation_aliases = ['title'] | W/o KB |
|  |  | Li Bai is the one who orders the destiny. |

| Input: Do you know when Albert II's father was born? | Retrieved Knowledge | Answer |
| --- | --- | --- |
| Model Search Code |  |  |
| GPT-4 |  |  |
| messages = "" |  |  |
| father, msg = find_entity_or_value(entity_aliases = ['Albert II'], relation_aliases = ['father', 'father is', 'dad']) |  |  |
| messages += msg |  |  |
| birth_date, msg = find_entity_or_value(entity_aliases = ['father'], relation_aliases = ['birth_date', 'born on']) |  |  |
| messages += msg |  |  |
| return messages |  |  |
| ChatGPT search: |  |  |
| messages = "" |  |  |
| father, msg = find_entity_or_value(entity_aliases = ['Albert II'], relation_aliases = ['who is the father', 'father of']) |  |  |
| messages += msg |  |  |
|  |  | None |
|  |  | W/o KB |
|  |  | Albert II's father was Albert I, who was born on April 8, 1875. |
|  |  | Albert II's father was Albert I, who was born on August 26, 1819. |

| Table 7: The detailed results of mixed queries and multi-hop relational queries on CN-DBPedia. |
```

### --- Page 0020 ---

```markdown
| Model  | Search Code                                                                 | Retrieved Knowledge                                                                                                                                                                                                                     | Answer                                                                                                                                                       |
|--------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPT-4  | def search(): ""<br>  first_winners, msg = find_entity_or_value(entity_aliases = ['Nobel Prize'], relation_aliases = ['first winner', 'first recipient'])<br>  messages = msg<br>  for winner in first_winners:<br>    award, msg = find_entity_or_value(entity_aliases = [winner], relation_aliases = ['awarded', 'award'])<br>    messages += msg<br>  return messages | [FROM CNDBpedia[find_entity_or_value], relation_aliases = ['Nobel Prize', relation_aliases = ['first winner', 'first recipient']] -> Nobel Prize, first winner: Wilhelm Conrad Röntgen, Jacobus Henricus van 't Hoff, Emil Adolf von Behring, Sully Prudhomme, Henry Dunant, Frédéric Passy[find_entity_or_value], relation_aliases = [Wilhelm Conrad Röntgen], relation_aliases = ['awarded', 'award'] -> Wilhelm Conrad Röntgen, Occupation: Physicist, ... [find_entity_or_value], relation_aliases = [Frédéric Passy], relation_aliases = ['awarded', 'award'] -> Frédéric Passy, Major Achievement: First recipient of the Nobel Peace Prize. | The inaugural winners of the Nobel Prize included Wilhelm Conrad Röntgen, Jacobus Henricus van 't Hoff, Emil Adolf von Behring, Sully Prudhomme, Henry Dunant, and Frédéric Passy. They respectively received the Nobel Prize in Physics, Chemistry, Medicine, Literature, and Peace. |
| ChatGPT | def search(): ""<br>  first_winners, msg = find_entity_or_value(entity_aliases = ['Nobel Prize'], relation_aliases = ['first winner', 'first recipient'])<br>  messages = msg<br>  awards = []<br>  for winner in first_winners:<br>    award, msg = find_entity_or_value(entity_aliases = [winner], relation_aliases = ['award'])<br>    messages += msg<br>    awards.append(award)<br>  messages += 'first winners: {}'.format(first_winners), 'respectively received awards: {}'.format(awards)<br>  return messages | [FROM CNDBpedia[find_entity_or_value], relation_aliases = ['Nobel Prize', relation_aliases = ['first winner', 'first recipient']] -> Nobel Prize, first winner: Wilhelm Conrad Röntgen, Jacobus Henricus van 't Hoff, Emil Adolf von Behring, Sully Prudhomme, Henry Dunant, Frédéric Passy[find_entity_or_value], relation_aliases = [Wilhelm Conrad Röntgen], relation_aliases = ['award'] -> Wilhelm Conrad Röntgen, Occupation: Physicist, ... | The inaugural winners of the Nobel Prize were announced in 1901. Here are the inaugural Nobel Prize winners, broken down by their respective categories: 1. Physics: Wilhelm Conrad Röntgen. 2. Nobel Prize in Chemistry: Jacobus Henricus van 't Hoff. 3. Nobel Prize in Literature: Sully Prudhomme. 4. ... Nobel Prize in Peace: Henry Dunant, who ... Frédéric Passy. |
| GPT-4  | def search(): ""<br>  lyricist, msg = find_entity_or_value(entity_aliases = ['Li Bai'], relation_aliases = ['lyrics writing', 'lyricist'])<br>  messages = msg<br>  composer, msg = find_entity_or_value(entity_aliases = ['Li Bai'], relation_aliases = ['composer', 'composer'])<br>  messages += msg<br>  arranger, msg = find_entity_or_value(entity_aliases = ['Li Bai'], relation_aliases = ['arranger', 'arranger'])<br>  messages += msg<br>  if lyricist == composer or arranger:<br>    messages += 'The lyricist, composer, and arranger of "Li Bai" are the same person.'<br>  else:<br>    messages += 'The lyricist, composer, and arranger of "Li Bai" are not the same person.'<br>  return messages | [FROM CNDBpedia[find_entity_or_value], relation_aliases = ['Li Bai'], relation_aliases = ['lyrics writing', 'lyricist'] -> Li Bai as song sung by Li Bai, lyricist: Li Ronghao, lyricist: Li Ronghao[find_entity_or_value], relation_aliases = ['Li Bai'], relation_aliases = ['arranger'] -> Li Bai as song sung by Li Ronghao, arrange: Li Ronghao[find_entity_or_value], relation_aliases = ['Li Bai'], relation_aliases = ['composer'] -> Li Bai as song sung by Li Ronghao, arrange: Li Ronghao. | Yes, the song Li Bai has its lyrics, composition, and arrangement all done by the same person, Li Ronghao. |
| ChatGPT | def search(): ""<br>  lyricist, msg = find_entity_or_value(entity_aliases = ['Li Bai'], relation_aliases = ['lyrics writing'])<br>  messages = msg<br>  composer, msg = find_entity_or_value(entity_aliases = ['Li Bai'], relation_aliases = ['composer'])<br>  messages += msg<br>  arranger, msg = find_entity_or_value(entity_aliases = ['Li Bai'], relation_aliases = ['arranger'])<br>  messages += msg<br>  if lyricist == composer and len(composer) == 1 and len(arranger) == 1:<br>    messages += 'The lyrics are the same person'<br>  else:<br>    messages += 'Not the same person'<br>  return messages | None | W/KB<br> W/o KB<br> The lyrics, composition, and arrangement of Li Bai were all done by the same person, the composer Li Ronghao. |
```


### --- Page 0021 ---

```markdown
| Input | What is the relationship between Ronald Lee and Li Bai? |
|-------|--------------------------------------------------------|
| Model | Search Code: ""                                        |
| GPT-4 | messages = ""                                         |
|       | relationship, msg = find_relationship(entity1_aliases = ["Li Ronghao"], entity2_aliases = ["Li Bai"]) |
|       | messages += msg                                       |
|       | return messages                                       |
| ChatGPT | der search(): ""                                    |
|       | messages = ""                                         |
|       | relationship, msg = find_relationship(entity1_aliases = ["Li Ronghao"], entity2_aliases = ["Li Bai"]) |
|       | messages += msg                                       |
|       | return messages                                       |
| Retrieved Knowledge | [FROM CNDBpedia[find_relationship] | 
|       | entity1_aliases = ["Li Ronghao"], entity2_aliases = ["Li Bai"] } -> Li Ronghao, Representative Work: Li Bai |
| Answer | W/ KB                                                |
|       | Li Ronghao created a song titled 'Li Bai' which is about his relationship with Li Bai. |
|       | Wo KB                                                |
|       | Li Ronghao and Li Bai are not directly related. Li Ronghao is a famous singer and music producer in modern China. |
|       | Wo KB                                                |
|       | Li Ronghao and Li Bai are the same person.          |

| Input | Who is taller, Yao Ming or Saber?                    |
|-------|--------------------------------------------------------|
| Model | Search Code: ""                                        |
| GPT-4 | messages = ""                                         |
|       | yao_height, msg = find_entity_or_value(entity_aliases = ["Yao Ming"], relation_aliases = ["height"]) |
|       | messages += msg                                       |
|       | if yao_height[0] > saber_height[0]:                 |
|       | messages += "Yao Ming is taller than Saber."         |
|       | else:                                                |
|       | messages += "Saber is taller than Yao Ming."         |
|       | return messages                                       |
| ChatGPT | der search(): ""                                    |
|       | messages = ""                                         |
|       | yao_height, msg = find_entity_or_value(entity_aliases = ["Yao Ming"], relation_aliases = ["height"]) |
|       | messages += msg                                       |
|       | return messages                                       |
| Retrieved Knowledge | [FROM CNDBpedia[find_entity_or_value] | 
|       | entity_aliases = ["Yao Ming"], relation_aliases = ["height"] } -> Yao Ming (Chairman of the Asian Basketball Association), height: 226 cm |
| Answer | W/ KB                                                |
|       | Yao Ming is taller than Saber.                       |
|       | Wo KB                                                |
|       | Yao Ming's height is 226 cm, and Saber (a fictional character from the Fate series) is 1.54 meters tall. So, Yao Ming is taller than Saber. |

| Input | Which one has a larger registered capital, Dongwu Securities or Xingye Securities? |
|-------|--------------------------------------------------------|
| Model | Search Code: ""                                        |
| GPT-4 | messages = ""                                         |
|       | ew_capital, msg = find_entity_or_value(entity_aliases = ["Dongwu Securities"], relation_aliases = ["registered capital"]) |
|       | messages += msg                                       |
|       | xingye_capital, msg = find_entity_or_value(entity_aliases = ["Xingye Securities"], relation_aliases = ["registered capital"]) |
|       | messages += msg                                       |
|       | if ew_capital and xingye_capital:                    |
|       | if ew_capital[0] > xingye_capital[0]:               |
|       | messages += "Dongwu Securities has a larger registered capital." |
|       | else:                                                |
|       | messages += "Dongwu Securities and Xingye Securities have the same registered capital." |
| ChatGPT | der search(): ""                                    |
|       | messages = ""                                         |
|       | ew_capital, msg = find_entity_or_value(entity_aliases = ["Dongwu Securities"], relation_aliases = ["registered capital"]) |
|       | messages += msg                                       |
|       | xingye_capital, msg = find_entity_or_value(entity_aliases = ["Xingye Securities"], relation_aliases = ["registered capital"]) |
|       | messages += msg                                       |
|       | return messages                                       |
| Retrieved Knowledge | [FROM CNDBpedia[find_entity_or_value] | 
|       | entity_aliases = ["Dongwu Securities"], relation_aliases = ["registered capital"] } -> Dongwu Securities, registered capital: 1.5 billion yuan |
| Answer | W/ KB                                                |
|       | Xingye Securities has a larger registered capital.    |
|       | Wo KB                                                |
|       | I'm sorry, I can't provide real-time registered capital information, as the data may change over time. It is recommended that you directly visit the official websites of these two companies or contact their customer service to get the most accurate information. |
|       | W/ KB                                                |
|       | Dongwu Securities has a larger registered capital. According to public data, the registered capital of Dongwu Securities is RMB 1.5 billion, while the registered capital of Xingye Securities is RMB 10 billion. |
```



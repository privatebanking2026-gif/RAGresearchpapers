# ArXiv 2312.10997

### --- Page 0001 ---

```markdown
# Retrieval-Augmented Generation for Large Language Models: A Survey

Yunfan Gao\(^a\), Yun Xiong\(^b\), Xinyu Gao\(^c\), Kangxiang Jia\(^d\), Jinliu Pan\(^e\), Yuxi Bi\(^f\), Yi Dai\(^g\), Jiawei Sun\(^h\), Meng Wang\(^i\), and Haofen Wang\(^a\)

\(^a\) Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University  
\(^b\) Shanghai Key Laboratory of Data Science, School of Computer Science, Fudan University  
\(^c\) College of Design and Innovation, Tongji University  

---

**Abstract**—Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases. This enhances the accuracy and credibility of the generation, particularly for knowledge-intensive tasks, and allows for continuous knowledge updates and integration of domain-specific information. RAG synergistically reimpresses LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases. This comprehensive review paper offers a detailed examination of the progression of RAG paradigms, encompassing the Native RAG, the Advanced RAG, and the Modular RAG. It meticulously scrutinizes the tripartite foundation of RAG generation, particularly focusing on retrieval, the generation, and augmentation techniques. The paper highlights the state-of-the-art technologies embedded in each of these critical components, providing a profound insight into the advancements in RAG systems. Furthermore, this paper introduces up-to-date evaluation framework and benchmark. At the end, this article delineates the challenges currently faced and points out prospective avenues for research and development.

**Index Terms**—Large language model, retrieval-augmented generation, natural language processing, information retrieval.

---

## I. INTRODUCTION

ARGE language models (LLMs) have achieved remarkable success, though they still face significant limitations, especially in domain-specific or knowledge-intensive tasks [1], notably producing “hallucinations” [2] when handling queries beyond their training data or requiring current information. To overcome challenges, Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant document chunks from external knowledge bases through semantic similarity calculation. By referencing external knowledge, RAG effectively reduces the problem of generating factually incorrect content. Its integration into LLMs has resulted in widespread adoption, establishing RAG as a key technology in advancing chatbots and enhancing the suitability of LLMs for real-world applications.

RAG technology has rapidly developed in recent years, and the technology tree summarizing related research is shown in Figure 1. The development trajectory of RAG in the era of large models exhibits several distinct stage characteristics. Initially, RAG’s inception coincided with the rise of the Transformer architecture, focusing on enhancing language models by incorporating additional knowledge through Pre-Training Models (PTM). This early stage was characterized by foundational work aimed at refining pre-training techniques [3]–[5]. The subsequent arrival of ChatGPT [6] marked a pivotal moment, with LLM demonstrating powerful in context learning (ICL) capabilities. RAG research shifted towards providing better information for LLMs to answer more complex and knowledge-intensive tasks during the inference stage, leading to rapid development in RAG studies. As research progressed, the enhancement of RAG was no longer limited to the inference stage but began to incorporate more with LLM fine-tuning techniques.

The burgeoning field of RAG has experienced swift growth, yet it has not been accompanied by a systematic synthesis that could clarify its broader trajectory. This survey endeavors to fill this gap by mapping out the RAG process and charting its evolution and anticipated future paths, with a focus on the integration of RAG within LLMs. This paper considers both technical paradigms and research methods, summarizing three main research paradigms from over 100 RAG studies, and analyzing key technologies in the core stages of “Retrieval,” “Generation,” and “Augmentation.” On the other hand, current research tends to focus on methods, lacking analysis and summarization of how to evaluate RAG. This paper comprehensively reviews the downstream tasks, datasets, benchmarks, and evaluation methods applicable to RAG. Overall, this paper sets out to meticulously compile and categorize the foundational technical concepts, historical progression, and the spectrum of RAG methodologies and applications that have emerged post-LLMs. It is designed to equip readers and professionals with a detailed and structured understanding of both large models and RAG. It aims to illuminate the evolution of retrieval augmentation techniques, assess the strengths and weaknesses of various approaches in their respective contexts, and speculate on upcoming trends and innovations.

Our contributions are as follows:
- In this survey, we present a thorough and systematic review of the state-of-the-art RAG methods, delineating its evolution through paradigms including native RAG.
```

### --- Page 0002 ---

```markdown
![Technology tree of RAG research](assets/page_0002_img_1.png)

Fig. 1. Technology tree of RAG research. The stages of involving RAG mainly include pre-training, fine-tuning, and inference. With the emergence of LLMs, research on RAG initially focused on leveraging the powerful in context learning abilities of LLMs, primarily concentrating on the inference stage. Subsequent research has delved deeper, gradually integrating more with the fine-tuning of LLMs. Researchers have also been exploring ways to enhance language models through the pre-training stage through retrieval-augmented techniques.

advanced RAG, and modular RAG. This review contextualizes the broader scope of RAG research within the landscape of LLMs.
- We identify and discuss the central technologies integral to the RAG process, specifically focusing on the aspects of “Retrieval”, “Generation” and “Augmentation”, and delve into their synergies, elucidating how these components intricately collaborate to form a cohesive and effective RAG framework.
- We summarize the current assessment methods of RAG, covering 26 tasks, nearly 50 datasets, outlining the evaluation objectives and metrics, as well as the current evaluation benchmarks and tools. Additionally, we anticipate future directions for RAG, emphasizing potential enhancements to tackle current challenges.

The paper unfolds as follows: Section II introduces the main concept and current paradigms of RAG. The following sections explore core components—“Retrieval”, “Generation” and “Augmentation”, respectively. Section III focuses on optimization methods in retrieval, including indexing, query and embedding optimization. Section IV concentrates on post-retrieval process and LLM fine-tuning in generation. Section V analyzes the three augmentation processes. Section VI focuses on RAG’s downstream tasks and evaluation system. Section VII mainly discusses the challenges that RAG currently faces and its future development directions. At last, the paper concludes in Section VIII.

II. OVERVIEW OF RAG

A typical application of RAG is illustrated in Figure 2. Here, a user poses a question to ChatGPT about a recent, widely discussed news. Given ChatGPT's reliance on training data, it initially lacks the capacity to provide updates on recent developments. RAG bridges this information gap by sourcing and incorporating knowledge from external databases. In this case, it gathers relevant news articles related to the user’s query. These articles, combined with the original question, form a comprehensive prompt that empowers LLMs to generate a well-informed answer.

The RAG research paradigm is continuously evolving, and we categorize it into three stages: Naive RAG, Advanced RAG, and Modular RAG, as shown in Figure 3. Despite RAG method are cost-effective and surpass the performance of the native LLM, they also exhibit several limitations. The development of Advanced RAG and Modular RAG is a response to these specific shortcomings in Naive RAG.

A. Naive RAG

The Naive RAG research paradigm represents the earliest methodology, which gained prominence shortly after the
```

### --- Page 0003 ---

```markdown
![A representative instance of the RAG process applied to question answering. It mainly consists of 3 steps: 1) Indexing. Documents are split into chunks, encoded into vectors, and stored in a vector database. 2) Retrieval. Retrieve the Top K chunks most relevant to the question based on semantic similarity. 3) Generation. Input the original question and the retrieved chunks together into LLM to generate the final answer.](assets/page_0003_img_1.png)

widespread adoption of ChatGPT. The Naive RAG follows a traditional process that includes indexing, retrieval, and generation, which is also characterized as a “Retrieve-Read” framework [7].

**Indexing** starts with the cleaning and extraction of raw data in diverse formats like PDF, HTML, Word, and Markdown, which is then converted into a uniform plain text format. To accommodate the content limitations of language models, text is segmented into smaller, digestible chunks. Chunks are then encoded into vector representations using embeddings and stored in vector database. This step is crucial for enabling efficient similarity searches in the subsequent retrieval phase.

**Retrieval.** Upon receipt of a user query, the RAG system employs the same encoding model utilized during the indexing phase to transform the query into a vector representation. It then computes the similarity scores between the query vector and the vector of chunks within the indexed corpus. The system prioritizes and retrieves the top K chunks that demonstrate the greatest similarity to the query. These chunks are subsequently used as the expanded context in prompt.

**Generation.** The posed query and selected documents are synthesized into a coherent prompt to which a large language model is tasked with formulating a response. The model’s approach to answering may vary depending on task-specific criteria, allowing it to either draw upon its inherent parametric knowledge or restrict its responses to the information contained within the provided documents. In cases of ongoing dialogues, any existing conversational history can be integrated into the prompt, enabling the model to engage in multi-turn dialogue interactions effectively.

However, Naive RAG encounters notable drawbacks:

**Retrieval Challenges.** The retrieval phase often struggles with precision and recall, leading to the selection of misaligned relevant chunks, and the missing of crucial information.

**Generation Difficulties.** In generating responses, the model may face the issue of hallucination, where it produces content not supported by the retrieved context. This phase can also suffer from irrelevance, toxicity, or bias on the outputs, detracting from the quality and reliability of the information.

**Augmentation Hardships.** Integrating retrieved information with the different task can be challenging, sometimes resulting in disjointed or incoherent outputs. The process may also encounter redundancy when similar information is retrieved from multiple sources, leading to repetitive responses. Determining the significance and relevance of various passages and ensuring stylistic and tonal consistency add further complexity. Facing complex issues, a single retrieval based on the original query may not suffice to acquire adequate context information. Moreover, there’s a concern that generation models might overly rely on augmented information, leading to outputs that simply echo retrieved content without adding insightful or synthesized information.

**B. Advanced RAG**  
Advanced RAG introduces specific improvements to overcome the limitations of Naive RAG. Focusing on enhancing retrieval quality, it employs pre-retrieval and post-retrieval strategies. To tackle the indexing issues, Advanced RAG refines its indexing techniques through the use of a sliding window approach, fine-grained segmentation, and the incorporation of metadata. Additionally, it incorporates several optimization methods to streamline the retrieval process [8].
```

### --- Page 0004 ---

```markdown
![Comparison between the three paradigms of RAG.](assets/page_0004_img_1.png)

Pre-retrieval process. In this stage, the primary focus is on optimizing the indexing structure and the original query. The goal of optimizing indexing is to enhance the quality of the content being indexed. This involves strategies: enhancing data granularity, optimizing index structures, adding metadata alignment optimization, and mixed retrieval. While the goal of query optimization is to make the user’s original question clearer and more suitable for the retrieval task. Common methods include query rewriting, query transformation, query expansion and other techniques [7], [9]–[11].

Post-Retrieval Process. Once relevant content is retrieved, it’s crucial to integrate it effectively with the query. The main methods in post-retrieval process include re-rank chunks and context compressing. Re-ranking the retrieved information to relocate the most relevant content to the edges of the prompt is a key strategy. This concept has been implemented in frameworks such as Llamanide2, LangChain3, and HayStack [12]. Feeding all relevant documents directly into LLMs can lead to information overload, diluting the focus on key details with irrelevant content. To mitigate this, post-retrieval efforts concentrate on selecting the essential information, emphasizing critical sections, and shortening the context to be processed.

C. Modular RAG

The modular RAG architecture advances beyond the former two RAG paradigms, offering enhanced adaptability and versatility. It incorporates diverse strategies for improving its components, such as adding a search module for similarity searches and refining the retriever through fine-tuning. Innovations like restructured RAG modules [14] have been introduced to tackle specific challenges. The shift towards a modular RAG approach is becoming prevalent, supporting both sequential processing and integrated end-to-end training across its components. Despite its distinctiveness, Modular RAG builds upon the foundational principles of Advanced and Naive RAG, illustrating a progression and refinement within the RAG family.

1) New Modules: The Modular RAG framework introduces additional specialized components to enhance retrieval and processing capabilities. The Search module adapts to specific scenarios, enabling direct searches across various data sources like search engines, databases, and knowledge graphs, using LLM-generated code and query languages [15]. RAG Fusion addresses traditional search limitations by employing a multi-query strategy that expands user queries into diverse perspectives, utilizing parallel vector searches and intelligent re-ranking to uncover both explicit and transformative knowledge [16]. The Memory module leverages the LLM’s memory to guide retrieval, creating an unbounded memory pool that
```

### --- Page 0005 ---

```markdown
aligns the text more closely with data distribution through iterative self-enhancement [17], [18]. Routing in the RAG system navigates through diverse data sources, selecting the optimal pathway for a query, whether it involves summarization, specific database searches, or merging different information streams [19]. The Predicted module aims to reduce redundancy and noise by generating context directly through the LLM, ensuring relevance and accuracy [13]. Lastly, the Task Adapter module tailors RAG to various downstream tasks, automating prompt retrieval for zero-shot inputs and creating task-specific retrievers through few-shot query generation [20], [21]. This comprehensive approach not only streamlines the retrieval process but also significantly improves the quality and relevance of the information retrieved, catering to a wide array of tasks and queries with enhanced precision and flexibility.

2) New Patterns: Modular RAG offers remarkable adaptability allowing for module substitution or reconfiguration to address specific challenges. This goes beyond the fixed structures of Naive and Advanced RAG, characterized by a simple “Retrieve” and “Read” mechanism. Moreover, Modular RAG expands this flexibility by integrating new modules or adjusting interaction flow among existing ones, enhancing applicability across different tasks.

Innovations such as the Rewrite-Retrieve-Read [7] model leverage the LLM’s capabilities to refine retrieval queries through a rewriting module and a LLM-feedback mechanism to update retrieving role, improving task performance. Similarly, approaches like the Hybrid Retrieve-Read model [8] emphasize retrieval from model weights, enhancing the model’s ability to handle knowledge-intensive tasks. Hybrid retrieval strategies integrate keyword, semantic, and vector searches to cater to diverse queries. Additionally, employing sub-queries and hypothetical document embeddings (HyDE) [11] seeks to improve retrieval relevance by focusing on embedding similarities between generated answers and real documents.

Adjustments in modular architecture and interaction, such as the Demonstrate-Search-Project (DSP) [23] framework and the iterative Retrieve-Read-Retrieve-Read flow of ITER-GETGEN [14], showcase the dynamic use of module outputs to bolster another module’s functionality, illustrating a sophisticated understanding of enhancing module synergy. The flexible orchestration of Modular RAG Flow showcases the benefits of adaptive retrieval through techniques such as FLARE [24] and Self-RAG [25]. This approach transcends the fixed RAG retrieval process by evaluating the necessity of retrieval based on different scenarios. Another benefit of a flexible architecture is that the RAG system can more easily integrate with other technologies (such as fine-tuning or reinforcement learning) [26]. For example, this can involve fine-tuning the retriever for better retrieval results, fine-tuning the generator for personalized outputs, or engaging in collaborative fine-tuning [27].

### D. RAG vs Fine-tuning

The augmentation of LLMs has attracted considerable attention due to their growing prevalence. Among the optimization methods for LLMs, RAG is often compared with Fine-tuning (FT) and prompt engineering. Each method has distinct characteristics as illustrated in Figure 4. We used a quadrant chart to illustrate the differences among the methods in two dimensions: external knowledge requirements and model adaptation requirements. Prompt engineering leverages a model’s inherent capabilities with minimum necessity for external knowledge and model adaptation. RAG can be likened to providing a model with a tailored textbook for information retrieval, ideal for precise information retrieval tasks. In contrast, FT is comparable to a student internalizing knowledge over time, suitable for scenarios requiring replication of specific structures, styles, or formats.

RAG excels in dynamic environments by offering real-time knowledge updates and effective utilization of external knowledge sources with high interpretability. However, it comes with higher latency and ethical considerations regarding data retrieval. On the other hand, FT is more static, requiring a retraining of updates but enabling the customization of the model’s behavior and style. It demands significant computational resources for dataset preparation and training, and while it can reduce hallucinations, it may face challenges with familiar data.

In multiple evaluations of their performance on various knowledge-intensive tasks across different domains, [28] demonstrated that while unsupervised fine-tuning shows some improvement, RAG significantly outperforms it, for both retrieval and knowledge accumulation during supervised fine-tuning. The choice between RAG and FT depends on the specific needs for data dynamics, customization, and computational capabilities in the application context. RAG and FT are not mutually exclusive and can complement each other, enhancing a model’s capabilities at different levels. In some instances, their combined use may lead to optimal performance. The optimization process involving RAG and FT may require multiple iterations to achieve satisfactory results.

### III. RETRIEVAL

In the context of RAG, it is crucial to efficiently retrieve relevant documents from the data source. There are several key issues involved, such as the retrieval source, retrieval granularity, pre-processing of the retrieval, and the corresponding embedding model.

#### A. Retrieval Source

RAG relies on external knowledge to enhance LLMs, while the type of retrieval source and the granularity of retrieval units both affect the final generation results.

1) Data Structure: Initially, text is the mainstream source of retrieval. Subsequently, the retrieval source expanded to include semi-structured data (PDF) and structured data (Knowledge Graph, KG) for enhancement. In addition to retrieving from original external sources, there is also a growing trend in recent researches towards utilizing content generated by LLMs themselves for retrieval and enhancement purposes.
```

### --- Page 0006 ---

```markdown
# TABLE I  
## SUMMARY OF RAG METHODS

| Method            | Retrieval Source                     | Retrieval Data Type | Retrieval Granularity | Augmentation Stage | Retrieval process |
|-------------------|--------------------------------------|----------------------|-----------------------|--------------------|-------------------|
| CoG               | Wikipedia                            | Text                 | Phrase                | Pre-training        | Iterative          |
| DenseX            | FactoidWiki                         | Text                 | Proposition           | Inference           | Once              |
| EAR               | Dataset-base                        | Text                 | Sentence              | Tuning              | Once              |
| UPRISE            | Dataset-base                        | Text                 | Sentence              | Tuning              | Once              |
| RAST              | Dataset-base                        | Text                 | Sentence              | Tuning              | Once              |
| Self-Mem          | Dataset-base                        | Text                 | Sentence              | Tuning              | Iterative          |
| FLARE             | Search Engine, Wikipedia            | Text                 | Sentence              | Adaptive            | Once              |
| PGRA              | Wikipedia                            | Text                 | Sentence              | Inference           | Once              |
| PILCO             | Wikipedia                            | Text                 | Sentence              | Inference           | Once              |
| RADA              | Dataset-base                        | Text                 | Sentence              | Inference           | Once              |
| Filter-rank       | Synthesized dataset                 | Text                 | Sentence              | Inference           | Once              |
| R-GCN             | Dataset-base                        | Text                 | Sentence              | Pre-training        | Once              |
| LLM-K            | Dataset-base                        | Text                 | Sentence              | Pre-training        | Iterative          |
| TIGER             | Dataset-base                        | Text                 | Item-base             | Pre-training        | Once              |
| LM-Index          | Dataset-base                        | Text                 | Item-base             | Tuning              | Once              |
| BEQ-UL           | Dataset-base                        | Text                 | Item-base             | Tuning              | Once              |
| CT-RAG           | Synthesized dataset                 | Text                 | Item-base             | Tuning              | Once              |
| Atlas             | Wikipedia, Common Crawl             | Text                 | Chunk                 | Pre-training        | Once              |
| RAVEN             | Wikipedia                            | Text                 | Chunk                 | Pre-training        | Once              |
| RETRO+           | Pre-training Corpus                 | Text                 | Chunk                 | Pre-training        | Iterative          |
| INSTRUCTRETRO     | Pre-training corpus                 | Text                 | Chunk                 | Tuning              | Once              |
| RRR               | Search Engine                       | Text                 | Chunk                 | Tuning              | Once              |
| RA2E             | Dataset-base                        | Text                 | Chunk                 | Tuning              | Once              |
| PROMPTAGTOR       | BEIR                                | Text                 | Chunk                 | Tuning              | Once              |
| AAR               | MS MARCO, Wikipedia                 | Text                 | Chunk                 | Tuning              | Once              |
| RA-DPT            | Common Crawl, Wikipedia             | Text                 | Chunk                 | Tuning              | Once              |
| RAG-FOUR          | Wikipedia                            | Text                 | Chunk                 | Tuning              | Once              |
| Self-RAG          | Dataset-base                        | Text                 | Chunk                 | Inference           | Once              |
| RGM               | Dataset-base                        | Text                 | Chunk                 | Inference           | Once              |
| CoG               | Wikipedia                            | Text                 | Chunk                 | Inference           | Once              |
| Token-Elimination  | Dataset-base                        | Text                 | Chunk                 | Inference           | Once              |
| PaperQ-A          | Arxiv, Online Database, PubMed      | Text                 | Chunk                 | Inference           | Once              |
| NoiseQA           | FactoidWiki                         | Text                 | Chunk                 | Inference           | Once              |
| MoM-ILACL         | Search Engine, Wikipedia            | Text                 | Chunk                 | Inference           | Once              |
| ToC               | Search Engine, Wikipedia            | Text                 | Chunk                 | Inference           | Recursive          |
| SRK               | Dataset-base                        | Text                 | Chunk                 | Inference           | Once              |
| IRFG              | Wikipedia                            | Text                 | Chunk                 | Inference           | Once              |
| RAG-LongConv      | Dataset-base                        | Text                 | Chunk                 | Inference           | Once              |
| IRET-RETGEN      | Wikipedia                            | Text                 | Chunk                 | Inference           | Once              |
| IRCT-01           | Wikipedia                            | Text                 | Chunk                 | Inference           | Recursive          |
| LLM-Knowledge-Boundary | Wikipedia                     | Text                 | Chunk                 | Inference           | Once              |
| RAPTOR            | Dataset-base                        | Text                 | Chunk                 | Inference           | Iterative          |
| RECTE             | LLMS                                | Text                 | Chunk                 | Inference           | Once              |
| ICRALM-64        | Pile, Wikipedia                     | Text                 | Chunk                 | Inference           | Iterative          |
| Retrieve-and-Sample | Dataset-base                      | Text                 | Chunk                 | Inference           | Once              |
| Zemi              | C4                                  | Text                 | Doc                   | Tuning              | Once              |
| CRAG              | Arxiv                               | Text                 | Doc                   | Inference           | Once              |
| 1 PAGER           | Wikipedia                            | Text                 | Doc                   | Inference           | Once              |
| PRCA              | Dataset-base                        | Text                 | Doc                   | Inference           | Once              |
| QLM-Doc-ranking   | Dataset-base                        | Text                 | Doc                   | Inference           | Iterative          |
| Recomp            | Wikipedia                            | Text                 | Doc                   | Inference           | Once              |
| EOP               | Dataset-base                        | Text                 | Doc                   | Inference           | Iterative          |
| RePLUG            | Pile                                | Text                 | Doc                   | Inference           | Iterative          |
| ArM-RAG           | Dataset-base                        | Text                 | Doc                   | Inference           | Iterative          |
| GenRead           | LLMS                                | Text                 | Doc                   | Inference           | Once              |
| UniMS-RAG         | Dataset-base                        | Text                 | Crosslingual, Text    | Sentence            | Inference           | Once              |
| CREA-ICL          | Dataset-base                        | Text                 | Tabular, Text         | Chunk               | Inference           | Once              |
| SANTA             | Dataset-base                        | Code, Text           | Item                  | Pre-training        | Once              |
| SURGE             | Dataset-base                        | KG                   | Sub-Graph             | Tuning              | Once              |
| Mu-KT-DB          | Dataset-base                        | KG                   | Entity                | Tuning              | Once              |
| Dual-Feedback-TDB  | Dataset-base, Graph                | KG                   | Entity Sequence       | Tuning              | Multi-time         |
| FABULA            | Dataset-base, Graph                 | KG                   | Entity                | Inference           | Once              |
| HYBGL             | Dataset-base                        | KG                   | Entity                | Inference           | Once              |
| KMPG              | Dataset-base                        | KG                   | Entity                | Inference           | Once              |
| GRAIL             | Dataset-base                        | KG                   | Sub-Graph             | Inference           | Once              |
| Freebase          | TextGraph                           | KG                   | Sub-Graph             | Inference           | Once              |
```

### --- Page 0007 ---

```markdown
![RAG compared with other model optimization methods in the aspects of “External Knowledge Required” and “Model Adaptation Required”.](assets/page_0007_img_1.png)

Unstructured Data, such as text, is the most widely used retrieval source, which are mainly gathered from corpus. For open-domain question-answering (ODQA) tasks, the primary retrieval sources are Wikipedia Dump with the current major versions including HotpotQA⁴ (1st October , 2017), DPR⁵ (20 December, 2018), in addition to encyclopedia dump, the specific data source includes cross-lingual text [19] and domain-specific data such as medical [67] and legal domains [29].

Semi-structured data, typically refers to data that contains a combination of text and table information, such as PDF. Handling semi-structured data poses challenges for conventional RAG systems due to two main reasons. Firstly, text extraction processes may introduce inadvertently separate tables, leading to data corruption during retrieval. Secondly, reconstructing tables into the data can complicate semantic similarity searches. When dealing with semi-structured data, one approach involves leveraging the code capabilities of LLMs to execute Text-2-SQL queries on tables within databases, such as TableGPT [85]. Alternatively, tables can be transformed into text format for further analysis using text-based methods [75]. However, both of these methods are not optimal solutions, indicating substantial research opportunities in this area.

Structured data, such as knowledge graphs (KGs) [86], which are typically verified and can provide more precise information. KnowledgeGPT [15] generates KB search queries and stores knowledge in a personalized base, enhancing the RAG model’s knowledge richness. In response to the limitations of LLMs in understanding and answering questions about textual graphs, G-Retriever [84] integrates Graph Neural Networks (GNNs), LLMs and RAG, enhancing graph comprehension and question-answering capabilities through soft prompting of the LLM, and employs the Prize-Collecting Steiner Tree (PCST) optimization problem for targeted graph retrieval. On the contrary, it requires additional effort to build, validate, and maintain structured databases. 

LLMs-Generated Content. Addressing the limitations of external auxiliary information in RAG, some research has focused on exploiting LLMs’ internal knowledge. SKR [68] classifies questions as known or unknown, applying retrieval enhancement selectively. GenRead [13] replaces the retrieval with an LLM generator, finding that LLM-generated contexts often contain more accurate answers due to better alignment with the pre-training objectives of causal language modeling. Selmer [17] iteratively creates an unbounded memory pool with a retrieval-enhanced generator, using a memory selector to choose outputs that serve as dual problems to the original question, thus self-enhancing the generative model. These methodologies underscore the breadth of innovative data source utilization in RAG, striving to improve model performance and task effectiveness.

2) Retrieval Granularity. Another factor besides the data format of the retrieval source is the granularity of the retrieved data. Coarse-grained retrieval units theoretically can provide more relevant information for the problem, but they may also contain redundant content, which could distract the retriever and language models in downstream tasks [50], [87]. On the other hand, fine-grained retrieval unit granularity increases the burden of retrieval and does not guarantee semantic integrity and meeting the required knowledge. Choosing
```

### --- Page 0008 ---

```markdown
# Page 0008

The appropriate retrieval granularity during inference can be a simple and effective strategy to improve the retrieval and downstream task performance of dense retrievers.

In text, retrieval granularity ranges from fine to coarse, including Token, Phrase, Sentence, Proposition, Chunks, Document. Among them, DenseX [30] proposed the concept of using propositions as retrieval units. Propositions are defined as atomic expressions in the text, each encapsulating a unique factual segment and presented in a concise, self-contained natural language format. This approach aims to enhance retrieval precision and relevance. On the Knowledge Graph (KG), retrieval granularity includes Entity, Triplet, and sub-Graph. The granularity of retrieval can also be adapted to downstream tasks, such as retrieving Item IDs [40] in recommendation tasks and Sentence pairs [38]. Detailed information is illustrated in Table 1.

## B. Indexing Optimization

In the Indexing phase, documents will be processed, segmented, and transformed into Embeddings to be stored in a vector database. The quality of index construction determines whether the correct context can be obtained in the retrieval phase.

1) **Chunking Strategy**: The most common method is to split the document into chunks on a fixed number of tokens (e.g., 100, 256, 512) [88]. Larger chunks can capture more context, but they may introduce noise, requiring longer processing time and higher costs. While smaller chunks may not fully convey the necessary context, they do have less noise. However, this leads to truncation within sentences, prompting the optimization of a recursive split and sliding window methods, enabling expanded retrieval by merging globally related information across multiple retrieval processes [89]. Nevertheless, these approaches still cannot strike a balance between semantic completeness and context length. Therefore, methods like Small2Big have been proposed, where sentences (small) are used as the retrieval unit, and the preceding context sentences are provided as (big) context to LLMs [90].

2) **Metadata Attachments**: Chunks can be enriched with metadata information such as page number, file name, author, category, timestamp. Subsequently, retrieval can be filtered based on this metadata, limiting the scope of the retrieval. Assigning different weights to document timestamps during retrieval can achieve time-aware RAG, ensuring the freshness of knowledge and avoiding outdated information.

In addition to extracting metadata from the original documents, metadata can also be artificially constructed. For example, adding summaries of paragraphs, as well as introducing hypothetical questions. This method is also known as Reverse HyDE. Specifically, using LLM to generate questions that can be answered by the document, then calculating the similarity between the original question and the hypothetical question during retrieval to reduce the semantic gap between the question and the answer.

3) **Structural Index**: One effective method for enhancing information retrieval is to establish a hierarchical structure for the documents. By constructing in structure, RAG system can expedite the retrieval and processing of pertinent data.

### Hierarchical index structure

Files are arranged in parent-child relationships, with chunks linked to them. Data summaries are stored at each node, aiding in the swift traversal of data and assisting the RAG system in determining which chunks to extract. This approach can also mitigate the illusion caused by block extraction issues.

### Knowledge Graph index

Utilize KG in constructing the hierarchical structure of documents contributes to maintaining consistency. It delineates the connections between different concepts and entities, markedly reducing the potential for illusions. Another advantage is the transformation of the information retrieval process into instructions that LLM can comprehend, thereby enhancing the accuracy of knowledge retrieval and enabling LLM to generate contextually coherent responses, thus improving the overall efficiency of the RAG system. To capture the logical relationship between document content and structure, KGP [91] proposed a method of building an index between multiple documents using KG. This KG consists of nodes (representing paragraphs or structures in the documents, such as pages and tables) and edges (indicating semantic/lexical similarity between paragraphs or relationships within the document structure), effectively addressing knowledge retrieval and reasoning problems in a multi-document environment.

## C. Query Optimization

When considering the primary challenges with Naive RAG is its direct reliance on the user’s original query as its retrieval. Formulating a precise and clear question is difficult, and imprudent queries result in subpar retrieval effectiveness. Sometimes, the question itself is complex, and the language is not well-organized. Another difficulty lies in language complexity ambiguity. Language models often struggle when dealing with specialized vocabulary or ambiguous abbreviations with multiple meanings. For instance, they may not discern whether “LLM” refers to large language model or a Master of Laws in a legal context.

1) **Query Expansion**: Expanding a single query into multiple queries enriches the context of the query, providing further context to address any lack of specific nuances, thereby ensuring the optimal relevance of the generated answers.

**Multi-Query**: By employing prompt engineering to expand queries via LLMs, these queries can then be executed in parallel. The expansion of queries is not random, but rather meticulously designed.

**Sub-Query**: The process of sub-question planning represents the generation of the necessary sub-questions to contextualize and fully answer the original question in mind. This process of adding relevant context is, in principle, similar to query expansion. Specifically, a complex question can be decomposed into a series of simpler sub-questions using the least-to-most prompting method [92].

**Chain-of-Verification(CoVe)**: The expanded queries undergo validation by LLM to achieve the effect of reducing hallucinations. Validated expanded queries typically exhibit higher reliability [93].
```

### --- Page 0009 ---

```markdown
2) Query Transformation: The core concept is to retrieve chunks based on a transformed query instead of the user’s original query.

**Query Rewrite.** The original queries are not always optimal for LLM retrieval, especially in real-world scenarios. Therefore, we can prompt LLM to rewrite the queries. In addition to using LLM for query rewriting, specialized smaller language models, such as RRR (Rewrite-retrieve-read) [7]. The implementation of the query rewrite method in the Taobao, known as BEQUE [9] has notably enhanced recall effectiveness for long-tail queries, resulting in a rise in GMV.

Another query transformation method is to use prompt engineering to let LLM generate a query based on the original query for subsequent retrieval. HyDE [11] construct hypothetical documents (assumed answers to the original query). It focuses on embedding similarity from answer to answer rather than seeking embedding similarity for the problem or query. Using the Step-back Prompting method [10], the original query is abstracted to generate a high-level concept question (step-back question). In the RAG system, both the step-back question and the original query are used for retrieval, and both results are utilized as the basis for language model answer generation.

**3) Query Routing:** Based on varying queries, routing to distinct RAG pipeline, which is suitable for a versatile RAG system designed to accommodate diverse scenarios.

**Metadata Router/ Filter.** The first step involves retrieving keywords (notably from the query), focusing on the keywords and metadata within the chunks to narrow down the search scope.

**Semantic Router** is another method of routing involving leveraging the semantic information of the query. Specific approach see Semantic Router 6. Certainly, a hybrid routing approach can also be employed, combining both semantic and metadata-based methods for enhanced query routing.

### D. Embedding

In RAG, retrieval is achieved by calculating the similarity (e.g. cosine similarity) between the embeddings of the question and document chunks, where the semantic representation capability of embedding models plays a key role. This mainly includes a sparse encoder (BM25) and a dense retriever (BERT architecture Pre-training language models). Recent research has introduced prominent embedding models such as AngIE, Vogo, BGE, etc [94–96], which are benefit from multi-task instruct tuning. Hugging Face’s MTEB leaderboard 7 evaluates embedding models across 8 tasks, covering 58 datasets. Additionally, C-MTEB focuses on Chinese capability, covering 6 tasks and 35 datasets. There is no one-size-fits-all answer to “which embedding model to use.” However, some specific models are better suited for particular use cases.

**1) MixHybrid Retrieval:** Sparse and dense embedding approaches capture different relevance features and can benefit from each other by leveraging complementary relevance information. For instance, sparse retrieval models can be used to provide initial search results for training dense retrieval models. Additionally, re-training language models (PLMs) can be utilized to learn term weights to enhance sparse retrieval. Specifically, it also demonstrates that sparse retrieval models can enhance the zero-shot retrieval capability of dense retrieval models and assist dense retrievers in handling queries containing rare entities, thereby improving robustness.

**2) Fine-tuning Embedding Model:** In instances where the context significantly deviates from pre-training corpus, particularly within highly specialized disciplines such as healthcare, legal practice, and other sectors replete with proprietary jargon, fine-tuning the embedding model on your own domain dataset becomes essential to mitigate such discrepancies.

In addition to supplementing domain knowledge, another purpose of fine-tuning is to align the retriever and generator, for example, using the results of LLM as the supervision signal for fine-tuning, known as LSR (LLM-supervised Retriever). PROMPTAGATOR [21] utilizes the LLM as a few-shot query generator to create task-specific retrievers, addressing challenges in supervised fine-tuning, particularly in data-scarce domains. Another approach, LLM-Embedder [97], exploits LLMs to generate reward signals across multiple downstream tasks. The retriever is fine-tuned with two types of supervised signals: hard labels for the dataset and soft rewards from LLMs. This dual-signal approach fosters a more effective fine-tuning process, tailoring the embedding model to derive downstream applications. REPLIE [72] utilizes a retriever and LLM to calibrate the probability distributions of the retrieved documents and then performs supervised training by computing the KL divergence. This straightforward and effective training method enhances the performance of the retrieval model by using an LLM as the supervisory signal, eliminating the need for specific cross-attention mechanisms. Moreover, inspired by RLHF (Reinforcement Learning from Human Feedback), utilizing LLM-based feedback to reinforce the retrieval through reinforcement learning.

### E. Adapter

Fine-tuning models may present challenges, such as integrating functionality through an API or addressing constraints arising from limited local computational resources. Consequently, some approaches opt to incorporate an external adapter to aid in alignment.

To optimize the multi-task capabilities of LLM, UPRISE [20] trained a lightweight prompt retriever that can automatically retrieve prompts from a pre-built prompt pool that are suitable for a given zero-shot task input. AAR (Augmentation-Adapted Retriever) [47] introduces a universal adapter designed to accommodate multiple downstream tasks. While PRCA [69] add a pluggable reward-driven contextual adapter to enhance performance on specific tasks. BGM [26] keeps the retriever and LLM fixed, and trains a bridge Seq2Seq model in between. The bridge model aims to transform the retrieved information into a format that LLMs can work with effectively, allowing it to not only rerank but also dynamically select passages for each query, and potentially employ more advanced strategies like repetition. Furthermore, PKG
```

### --- Page 0010 ---

```markdown
# PAGE_NAME: page_0010

introduces an innovative method for integrating knowledge into white-box models via directive fine-tuning [75]. In this approach, the retriever module is directly substituted to generate relevant documents according to a query. This method assists in addressing the difficulties encountered during the fine-tuning process and enhances model performance.

## IV. GENERATION

After retrieval, it is not a good practice to directly input all the retrieved information to the LLM for answering questions. Following will introduce adjustments from two perspectives: adjusting the retrieved content and adjusting the LLM.

### A. Context Curation

Redundant information can interfere with the final generation of LLM, and overly long contexts can also lead LLM to the “Lost in the middle” problem [98]. Like humans, LLM tends to only focus on the beginning and end of long texts, while forgetting the middle portion. Therefore, in the RAG system, we typically need to further process the retrieved content.

1) **Context Reranking**: Reranking fundamentally reorders document chunks to highlight the most pertinent results first, effectively redefining the overall document pool, serving a dual purpose in information retrieval, acting as both an enhancer and a filter, delivering refined inputs for more precise language model processing [70]. Reranking can be performed using rule-based methods that depend on predefined metrics like Diversity, Relevance, and MRR, or model-based approaches like Encoder-Decoder models from the BERT series (e.g., SpanBERT), specialized reranking models such as CohereRank or pre-reranker-large, and general large language models like GPT [12, 99].

2) **Content Selection/Compression**: A common misconception in the RAG process is the belief that retrieving as many relevant documents as possible and concatenating them to form a lengthy retrieval prompt is beneficial. However, excessive context can introduce more noise, diminishing the LLM’s perception of key information.

(Long) LLMingua [100], [101] utilize small language models (SLMs) such as GPT2-Small or LLaMa-7B, to detect and remove unimportant tokens, transforming it into a form that is challenging for humans to comprehend but well understood by LLMs. This approach presents a direct and practical method for prompt compression, eliminating the need for additional tuning of LLMs while balancing language integrity and compression ratio. PRCA tackled this issue by training an information condenser using contrastive learning [71]. Each training data point consists of one positive sample and five negative samples, and the encoder undergoes training using contrastive loss throughout this process [102].

In addition to compressing the content, reducing the number of documents also helps improve the accuracy of the model’s answers. Me et al. [103] propose the “Filter-Reranker” paradigm, which combines the strengths of LLMs and SLMs.

In this paradigm, SLMs serve as filters, while LLMs function as reordering agents. The research shows that instructing LLMs to rearrange challenging samples identified by SLMs leads to significant improvements in various Information Extraction (IE) tasks. Another straightforward and effective approach involves having the LLM evaluate the retrieved content before generating the final answer. This allows the LLM to filter out documents with poor relevance through LLM critique. For instance, in Chatal [104], the LLM is prompted to self-suggest on the referenced legal provisions to assess their relevance.

### B. LLM Fine-tuning

Targeted fine-tuning based on the scenario and data characteristics on LLMs can yield better results. This is also one of the greatest advantages of using on-premise LLMs. When LLMs lack data in a specific domain, additional knowledge can be provided to the LLM through fine-tuning. Huggingface’s fine-tuning data can also be used as an initial step.

Another benefit of fine-tuning is the ability to adjust the model’s input and output. For example, it can enable LLM to adapt to specific data formats and generate responses in a particular style as instructed [37]. For retrial tasks that engage with structured data, the SANTA framework [76] implements a tripartite training regimen to effectively encapsulate both structural and semantic nuances. The initial phase focuses on the query and document embeddings.

Aligning LLM outputs with human or retriever preferences through reinforcement learning is a potential approach. For instance, manually annotating the final generated answers and then providing feedback through reinforcement learning. In addition to aligning with human preferences, it is also possible to align with the preferences of fine-tuned models and retrievers [79]. When circumstances prevent access to powerful proprietary models or larger parameter open-source models, a simple and effective method is to distill the more powerful models (e.g. GPT-4). Fine-tuning of LLM can also be coordinated with fine-tuning of the retriever to align preferences. A typical approach, such as RA-DRT [27], aligns the scoring functions between Retriever and Generator using KL divergence.

## V. AUGMENTATION PROCESS IN RAG

In the domain of RAG, the standard practice often involves a singular (once) retrieval step followed by generation, which can lead to inefficiencies and sometimes is typically insufficient for complex problems demanding multi-step reasoning, as it provides a limited scope of information [105]. Many studies have optimized their retrieval process in response to this issue, and we have summarised them in Figure 5.

## VI. Iterative Retrieval

Iterative retrieval is a process where the knowledge base is repeatedly searched based on the initial query and the text generated so far, providing a more comprehensive knowledge
```

### --- Page 0011 ---

```markdown
# Page 0011

![Diagram illustrating iterative, recursive, and adaptive retrieval processes](assets/page_0011_img_1.png)

base for LLMs. This approach has been shown to enhance the robustness of subsequent answer generation by offering additional contextual references through multiple retrieval iterations. However, it may be affected by semantic disconnection and the accumulation of irrelevant information. IR-REGTEN [14] employs a synergistic approach that leverages “retrieval-enhanced generation” alongside “generation-enhanced retrieval” for tasks that necessitate the reproduction of specific information. The model harnesses the content required to address the input task as a contextual basis for retrieving pertinent knowledge, which in turn facilitates the generation of improved responses in subsequent iterations.

## B. Recursive Retrieval

Recursive retrieval is often used in information retrieval and NLP to improve the depth and relevance of search results. The process involves iteratively refining search queries based on the results obtained from previous searches. Recursive Retrieval aims to enhance the search experience by gradually converging on the most pertinent information through a feedback loop. IRCtoT [61] uses chain-of-thought to guide the retrieval process and enriches the CoT with the obtained retrieval results. ToC [57] creates a clarification tree that systematically optimizes the ambiguous parts in the Query. It can be particularly useful in complex search scenarios where the user’s needs are not entirely clear from the outset or where the information sought is highly specified or nuanced. The recursive nature of the process allows for continuous learning and adaptation to the user’s requirements, often resulting in improved satisfaction with the search outcomes.

To address specific data searches, recursive retrieval and multi-hop retrieval techniques are utilized together. Recursive retrieval involves a structured index to process and retrieve data in a hierarchical manner, which may include summarizing sections of a document or lengthy PDF before performing a retrieval based on this summary. Subsequently, a secondary retrieval within the document refines the search, embodying the recursive nature of the process. In contrast, multi-hop retrieval is designed to develop deeper into graph-structured data sources, extracting intermediate information [106].

## C. Adaptive Retrieval

Adaptive retrieval methods, exemplified by Flare [24] and Self-RAG [25], refine the RAG framework by enabling LLMs to actively determine the optimal means and content for retrieval, thus enhancing the efficiency and relevance of the information sourced.

These methods are part of a broader trend wherein LLMs employ active judgment in their operations, as seen in model agents like AutoGPT, Toolformer, and Graph-Toolformer [107]–[109]. Graph-Toolformer, for instance, divides its retrieval process into distinct steps where LLMs proactively use retrievers, apply Self-Ask techniques, and employ few-shot prompts to initiate search queries. This proactive stance allows LLMs to decide when to search for necessary information, akin to how an agent utilizes tools. WebGPT [110] integrates a reinforcement learning framework to train the GPT-3 model in autonomously using a search engine during text generation. It analyzes this process using special tokens that facilitate actions such as search engine queries, browsing results, and citing references, thereby expanding GPT-3’s capabilities through the use of external search engines. Flare automates timing retrieval by monitoring the confidence of the generation process, as indicated by the
```

### --- Page 0012 ---

```markdown
# PAGE_NAME: page_0012

## VI. TASK AND EVALUATION

The rapid advancement and growing adoption of RAG in the field of NLP have propelled the evaluation of RAG models to the forefront of research in the LLMs community. The primary objective of this evaluation is to comprehend and optimize the performance of RAG models across diverse application scenarios. This chapter will mainly introduce the evaluation tasks of RAG models, both downstream and evaluation targets.

### A. Downstream Task

The core task of RAG remains Question Answering (QA), including traditional single-hop/multi-hop QA, multiple-choice, domain-specific QA as well as long-form scenarios suitable for RAG. In addition to QA, RAG is continuously being expanded into multiple downstream tasks, such as Information Extraction (IE), dialogue generation, code search, etc. The main downstream tasks of RAG and their corresponding datasets are summarized in Table II.

### B. Evaluation Target

Historically, RAG models assessments have centered on their execution in specific downstream tasks. These evaluations employ established metrics suitable to the tasks at hand. For instance, question answering evaluations might rely on EM and F1 scores [7], [45], [59], [72], whereas fact-checking scores often hinge on Accuracy as the primary metric [4], [11], [42]. BLEU and ROUGE metrics are also commonly used to evaluate answer quality [26], [32], [52], [78]. Tools like RALIE, designed for the automatic evaluation of RAG applications, similarly base their assessments on these task-specific metrics [160]. Despite this, there is a notable paucity of research dedicated to evaluating the distinct characteristics of RAG models. The main evaluation objectives include:

1. **Retrieval Quality.** Evaluating the retrieval quality is crucial for determining the effectiveness of the context sourced by the retriever component. Standard metrics from the domains of search engines, recommendation systems, and information retrieval systems are employed to measure the performance of the RAG retrieval module. Metrics such as Hit Rate, MRR, and NDCG are commonly utilized for this purpose [161], [162].

2. **Generation Quality.** The assessment of generation quality centers on the generator's capacity to synthesize coherent and relevant answers from the retrieved context. This evaluation can be categorized based on the content’s objectives: unlabeled and labeled content. For unlabeled content, the evaluation encompasses the faithfulness, relevance, and non-harmfulness of the generated answers. In contrast, for labeled content, the focus is on the accuracy of the information produced by the model [161]. Additionally, both retrieval and generation quality assessments can be conducted through manual or automatic evaluation methods [29], [161], [163].

### C. Evaluation Aspects

Contemporary evaluation practices of RAG models emphasize three primary quality scores and four essential attributes, which collectively inform the evaluation of the two principal targets of the RAG model: retrieval and generation.

1. **Quality Scores:** Quality scores include context relevance, answer faithfulness, and answer relevance. These quality scores evaluate the efficiency of the RAG model from different perspectives in the process of information retrieval and generation [164]–[166].

   - **Context Relevance** evaluates the precision and specificity of the retrieved context, ensuring relevance and minimizing processing costs associated with extraneous content.
   - **Answer Faithfulness** ensures that the generated answers remain true to the retrieved context, maintaining consistency and avoiding contradictions.
   - **Answer Relevance** requires that the generated answers are directly pertinent to the posed questions, effectively addressing the core inquiry.

2. **Required Abilities:** RAG evaluation also encompasses four abilities indicative of its adaptability and efficiency: noise robustness, negative rejection, information integration, and counterfactual robustness [167], [168]. These abilities are critical for the model’s performance under various challenges and complex scenarios, impacting the quality scores.

   - **Noise Robustness** appraises the model’s capability to manage noise documents that are question-related but lack substantive information.
   - **Negative Rejection** assesses the model’s discernment in refraining from responding when the retrieved documents do not contain the necessary knowledge to answer a question.
   - **Information Integration** evaluates the model’s proficiency in synthesizing information from multiple documents to address complex questions.
   - **Counterfactual Robustness** tests the model’s ability to recognize and disregard known inaccuracies within documents, even when instructed about potential misinformation.

Context relevance and noise robustness are important for evaluating the quality of retrieval, while answer faithfulness, answer relevance, negative rejection, information integration, and counterfactual robustness are important for evaluating the quality of generation.
```

### --- Page 0013 ---

```markdown
# TABLE II  
DOWNSTREAM TASKS AND DATASETS OF RAG

| Task       | Sub Task               | Dataset                          | Method                                   |
|------------|-----------------------|----------------------------------|------------------------------------------|
| QA         | Single-hop            | Natural Question(NQ) [111]      | [26], [30], [34], [42], [45], [50], [52], [59], [64], [82]  |
|            |                       | TriviaQA(TQA) [113]             | [3], [4], [22], [40], [43], [54], [62], [71], [112]         |
|            |                       | SQuAD [114]                     | [20], [23], [30], [32], [45], [69], [112]                    |
|            |                       | Web Questions(WebQ) [115]       | [3], [4], [13], [30], [50], [68]                             |
|            |                       | PopQA [116]                     | [7], [25], [67]                                           |
|            |                       | MS MARCO [117]                  | [4], [40], [52]                                           |
|            | Multi-hop             | HotpotQA [118]                  | [23], [26], [31], [47], [51], [61], [82]                    |
|            |                       | 2WikiMultiHopQA [119]          | [14], [24], [48], [59], [61], [91]                          |
|            |                       | MuSiQue [120]                   | [14], [51], [61], [91]                                   |
|            | Long-form QA          | ELI5 [121]                      | [27], [34], [43], [49], [51]                               |
|            |                       | NarrativeQ2A(NQA) [122]        | [45], [60], [63], [123]                                   |
|            |                       | ASQA [124]                      | [24], [57]                                               |
|            |                       | QMSum(QM) [125]                 | [60], [123]                                             |
|            | Domain QA             | Qasper [126]                    | [60], [63]                                               |
|            |                       | COVID-QA [127]                  | [35], [44]                                               |
|            |                       | CMB [128]MMCU_Medical [129]    | [81]                                                    |
|            | Multi-Choice QA       | QuALITY [130]                   | [60], [63]                                               |
|            |                       | ARC [131]                       | [25], [67]                                               |
|            |                       | CommonsenseQA [132]            | [58], [66]                                               |
| Graph QA   |                       | GraphQA [84]                    | [84]                                                    |
| Dialog     | Dialog Generation      | Wizard of Wikipedia (WoW) [133] | [13], [27], [34], [42]                               |
|            |                       | Personal Dialog [134]           | [74], [135]                                             |
|            |                       | DuleMoN [136]                   | [74]                                                    |
|            | Task-oriented Dialog   | CamRest [137]                   | [78], [79]                                             |
|            | Recommendation        | Amazon(Toys,Sport,Beauty) [138] | [39], [40]                                             |
| IE         | Event Argument Extraction | WikiEvent [139]               | [13], [27], [37], [42]                               |
|            |                       | RAMS [140]                      | [36], [37]                                               |
|            |                       | T-REx [141], ZsRE [142]        | [27], [51]                                               |
| Reasoning  | Commonsense Reasoning  | HellaSwag [143]                 | [20], [66]                                               |
|            |                       | CoT Reasoning [144]            | [27]                                                    |
|            |                       | Complex Reasoning [145]         | [55]                                                    |
| Others     | Language Understanding  | MMLU [146]                      | [7], [27], [28], [42], [43], [47], [72]                    |
|            | Language Modeling      | WikiText-103 [147]              | [5], [29], [64], [71]                                   |
|            |                       | StrategyQA [148]                | [14], [48], [51], [55], [58]                             |
|            | Fact Checking/Verification | FEVER [149]                  | [4], [13], [27], [34], [42], [50]                          |
|            |                       | PubHealth [150]                 | [25], [67]                                               |
|            | Text Generation       | Biography [151]                  | [67]                                                    |
|            | Text Summarization    | WikiAPI [152]                   | [24]                                                    |
|            |                       | XSum [153]                      | [17]                                                    |
|            | Text Classification    | ViLeons [154]                   | [19]                                                    |
|            |                       | TREC [155]                      | [33]                                                    |
|            | Sentiment             | SST-2 [156]                     | [20], [33], [38]                                       |
|            | Code Search           | CodeSearchNet [157]             | [76]                                                    |
|            | Robustness Evaluation  | NoMIRACL [56]                   | [56]                                                    |
|            | Math                  | GSM8K [158]                     | [73]                                                    |
|            | Machine Translation    | JRC-Acquis [159]                | [17]                                                    |
```

### --- Page 0014 ---

```markdown
# SUMMARY OF METRICS APPLICABLE FOR EVALUATION ASPECTS OF RAG

| Context Relevance | Faithfulness | Answer Relevance | Noise Robustness | Negative Rejection | Information Integration | Counterfactual Robustness |
|-------------------|--------------|------------------|-------------------|---------------------|-------------------------|---------------------------|
| ✓                 | ✓            | ✓                | ✓                 | ✓                   | ✓                       | ✓                         |
| Accuracy          | ✓            |                  |                   |                     |                         |                           |
| EM                | ✓            |                  |                   |                     |                         |                           |
| Recall            | ✓            |                  |                   |                     |                         |                           |
| Precision         | ✓            |                  |                   |                     |                         |                           |
| R-Rate            | ✓            |                  |                   |                     |                         |                           |
| Cosine Similarity | ✓            |                  |                   |                     |                         |                           |
| Hit Rate          | ✓            |                  |                   |                     |                         |                           |
| MRR               | ✓            |                  |                   |                     |                         |                           |
| NDCG              | ✓            |                  |                   |                     |                         |                           |
| BLEU              | ✓            |                  |                   |                     |                         |                           |
| ROUGE/ROUGE-L     | ✓            |                  |                   |                     |                         |                           |

The specific metrics for each evaluation aspect are summarized in Table III. It is essential to recognize that these metrics, derived from related work, are traditional measures and do not yet represent a mature or standardized approach for quantifying RAG evaluation aspects. Custom metrics tailored to the nuances of RAG models, though not included here, have also been developed in some evaluation studies.

## D. Evaluation Benchmarks and Tools

A series of benchmark tests and tools have been proposed to facilitate the evaluation of RAG. These instruments for quantitative metrics not only gauge RAG model performance but also enhance comprehension of the model’s capabilities across various evaluation aspects. Prominent benchmarks such as RGB, RECALL and CRUD [167]–[169] focus on ascertaining the essential abilities of RAG models. Currently, state-of-the-art automated tools like RAGAS [164], ARES [165], and TruLens [9] employ LLMs to adjudicate the quality scores. These tools and benchmarks collectively form a robust framework for the systematic evaluation of RAG models, as summarized in Table IV.

## VII. DISCUSSION AND FUTURE PROSPECTS

Despite the considerable progress in RAG technology, several challenges persist that warrant in-depth research. This chapter will mainly introduce the current challenges and future research directions faced by RAG.

### A. RAG vs Long Context

With the deepening of related research, the context of LLMs is continuously expanding [170]–[172]. Presently, LLMs can effortlessly manage contexts exceeding 200,000 tokens. This capability signifies that long-document question answering, previously reliant on RAG, can now incorporate the entire document directly into the prompt. This has also sparked discussions on whether RAG is still necessary when LLMs are not constrained by context. In fact, RAG still plays an irreplaceable role. On one hand, providing LLMs with a large amount of context at once will significantly impact its inference speed, while chunked retrieval and on-demand input can significantly improve operational efficiency. On the other hand, RAG-based generation can quickly locate the original references for LLMs to help users verify the generated answers. The entire retrieval and reasoning process is observable, while generation solely relying on long context remains a black box. Conversely, the expansion of context provides new opportunities for the development of RAG, enabling it to address more complex problems and integrative or summary questions that require reading a large amount of material to answer. Developing new RAG methods in the context of super-long contexts is one of the future research trends.

### B. RAG Robustness

The presence of noise or contradictory information during retrieval can detrimentally affect RAG’s output quality. This situation is figuratively referred to as “Misinformation can be worse than no information at all”. Improving RAG’s resistance to such adversarial or counterfactual inputs is gaining research momentum and has become a key performance metric [48], [50]. Cuconato et al. [54] analyze which type of documents should be retrieved, evaluate the relevance of the documents to the prompt, their position, and the number included in the context. The research findings reveal that including irrelevant documents can unexpectedly increase accuracy by over 30%, contradicting the initial assumption of reduced quality. These results underscore the importance of developing specialized strategies to integrate retrieval with language generation models, highlighting the need for further research and exploration into the robustness of RAG.

### C. Hybrid Approaches

Combining RAG with fine-tuning is emerging as a leading strategy. Determining the optimal integration of RAG and fine-tuning whether sequential, alternating, or through end-to-end joint training—and how to harness both parameterized
```

### --- Page 0015 ---

```markdown
# SUMMARY OF EVALUATION FRAMEWORKS

| Evaluation Framework | Evaluation Targets         | Evaluation Aspects                     | Quantitative Metrics               |
|----------------------|---------------------------|----------------------------------------|------------------------------------|
| RGB†                 | Retrieval Quality         | Noise Robustness                       | Accuracy                            |
|                      | Generation Quality        | Negative Rejection                     | EM                                 |
|                      |                           | Information Integration                | Accuracy                            |
|                      |                           | Counterfactual Robustness              | Accuracy                            |
| RECALL†              | Generation Quality        | Counterfactual Robustness              | R-Rate (Reappearance Rate)         |
| RAGAS†               | Retrieval Quality         | Context Relevance                      | *                                  |
|                      | Generation Quality        | Faithfulness                           | *                                  |
|                      |                           | Answer Relevance                       | Cosine Similarity                  |
| ARES†                | Retrieval Quality         | Context Relevance                      | Accuracy                            |
|                      | Generation Quality        | Faithfulness                           | Accuracy                            |
|                      |                           | Answer Relevance                       | Accuracy                            |
| TruLens†             | Retrieval Quality         | Context Relevance                      | *                                  |
|                      | Generation Quality        | Faithfulness                           | *                                  |
|                      |                           | Answer Relevance                       | *                                  |
| CRUD†                | Retrieval Quality         | Creative Generation                    | BLEU                               |
|                      | Generation Quality        | Knowledge-intensive QA                 | ROUGE-L                             |
|                      |                           | Error Correction                       | Summarization                      |

† represents a benchmark, and ‡ represents a tool. * denotes customized quantitative metrics, which deviate from traditional metrics. Readers are encouraged to consult pertinent literature for the specific quantification formulas associated with these metrics, as required.

## D. Scaling Laws of RAG

End-to-end RAG models and pre-trained models based on RAG are still one of the focuses of current research. The parameters of these models are one of the key factors. While scaling laws are established for LLMs, their applicability to RAG remains uncertain. Initial studies like RETRO have begun to address this, yet the parameter count in RAG models still lags behind that of LLMs. The possibility of an Inverse Scaling Law, where smaller models outperform larger ones, is particularly intriguing and merits further investigation.

## E. Production-Ready RAG

RAG’s practicality and alignment with engineering requirements have facilitated its adoption. However, enhancing retrieval efficiency, improving document recall in large knowledge bases, and ensuring data security—such as preventing inadvertent disclosure of customer sources or metadata by LLMs—are critical engineering challenges that remain to be addressed.

The development of the RAG ecosystem is greatly impacted by the progression of its technical stack. Key tools like LangChain and LlamaIndex have quickly gained popularity with the emergence of ChatGPT, providing extensive RAG-related APIs and becoming essential in the realm of LLMs. The emerging technology stack, while not as rich in features as LangChain and LlamaIndex, stands out through its specialized products. For example, Flowise AI prioritizes a low-code approach, allowing users to deploy AI applications, including RAG, through a user-friendly drag-and-drop interface. Other technologies like HayStack, Meltano, and Other Coral are also gaining attention for their unique contributions to the field.

In addition to AI-focused vendors, traditional software and cloud service providers are expanding their offerings to include RAG-centric services. Weaviate’s Verba is designed for personal assistant applications, while Amazon’s Kendra offers intelligent enterprise search services, enabling users to browse various content repositories using built-in connectors. In the development of RAG technology, there is a clear trend towards different specialization directions, such as: 1) Customization - tailoring RAG to meet specific requirements. 2) Simplification - making RAG easier to use to reduce the
```

### --- Page 0016 ---

```markdown
![Summary of RAG ecosystem](assets/page_0016_img_1.png)

## RAG Ecosystem

### Downstream Tasks
- Dialogue
- Question answering
- Summarization
- Fact verification

### Technology Stacks
- Langchain
- LlamaIndex
- FlowiseAI
- AutoGen

### The RAG Paradigm
- **Naive RAG**
- **Advanced RAG**
- **Modular RAG**

### Techniques for Better RAG
- Chunk Optimization
- Iterative Retrieval
- Retriever Fine-tuning
- Query Transformation
- Context Selection
- Adaptive Retrieval
- Dual Fine-tuning

### Key Issues of RAG
- What to refine
- How to use Retrieval

### RAG Prospect
- **Challenges**
  - RAG in Long Context Length
  - Hybrid
  - Robustness
  - Scaling-laws for RAG
  - Production-ready RAG
- **Evaluation of RAG**
  - **Evaluation Target**
    - Retrieval Quality
    - Generation Quality
  - **Evaluation Aspects**
    - Answer Relevance
    - Context Relevance
    - Answer Faithfulness
  - **Evaluation Framework**
    - Benchmarks
      - CRUD
      - Trulens
      - RGB
      - RAGAS
      - ARES

## Multi-modal RAG

RAG has transcended its initial text-based question-answering confines, embracing a diverse array of modal data. This expansion has spawned innovative multimodal models that integrate RAG concepts across various domains:

- **Image**: RA-CM3 [176] stands as a pioneering multimodal model of both retrieving and generating text and images. BLIP-2 [177] leverages frozen image encoders alongside LLMs for efficient visual language pre-training, enabling zero-shot image-to-text conversions. The “Visualize Before You Write” method [178] employs image generation to steer the LM's text generation, showing promise in open-ended text generation tasks.

- **Audio and Video**: The GSS method retrieves and stitches together audio clips to convert machine-translated data into speech-translated data [179]. UEOP marks a significant advancement in end-to-end automatic speech recognition by incorporating external, multimodal machine-aided solutions for voice-to-text conversion [180]. Additionally, KNN-based attention fusion leverages audio embeddings and semantically related text embeddings to refine ASR, thereby accelerating domain adaptation.

- **Code**: RBPS [182] excels in small-scale learning tasks by retrieving code examples that align with developers' objectives through encoding and frequency analysis. This approach has demonstrated efficacy in tasks such as test assertion generation and program repair. For structured knowledge, the CoK method [106] first extracts facts pertinent to the input query from a knowledge graph, then integrates these facts as hints within the input, enhancing performance in knowledge graph question-answering tasks.

## VIII. CONCLUSION

The summary of this paper, as depicted in Figure 6, emphasizes RAG's significant advancement in enhancing the capabilities of LLMs by integrating parameterized knowledge from external knowledge bases with extensive non-parameterized data from various tasks. The analysis outlines three developmental paradigms within the RAG framework: Naive, Advanced, and Modular RAG, each representing a progressive enhancement over its predecessors. RAG's technical integration with other AI methodologies, such as fine-tuning and reinforcement learning, has further expanded its capabilities. Despite the progress in RAG technology, there are research opportunities to improve its robustness and its ability to handle extended contexts. RAG's application scope is expanding into multimodal domains, adapting its principles to interpret and process diverse data forms like images, videos, and code. This expansion highlights RAG's significant practical implications for AI deployment, attracting interest from academic and industrial sectors.
```

### --- Page 0017 ---

```markdown
The growing ecosystem of RAG is evidenced by the rise in RAG-centric AI applications and the continuous development of supportive tools. As RAG's application landscape broadens, there is a need to refine evaluation methodologies to keep pace with its evolution. Ensuring accurate and representative performance assessments is crucial for fully capturing RAG's contributions to the AI research and development community.

## REFERENCES

| No. | Citation                                                                                                           |
|-----|--------------------------------------------------------------------------------------------------------------------|
| [1] | N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, “Large language models struggle to learn long-tail knowledge,” in International Conference on Machine Learning, PMLR, 2023, pp. 15 696–15 707. |
| [2] | Y. Zhang, Y. Li, C. Cui, D. Liu, T. Xu, H. Huang, E. Zhao, Y. Zhang, Y. Chen et al., “Siren’s song in the ocean: A survey on hal- lucination in large language models,” arXiv preprint arXiv:2309.02129, 2023. |
| [3] | D. Arora, A. K. Sin, R. Chowdhury, and N. Sharma, “Giraffe-mets: A graph- based paradigm for zero-shot information retrieval,” arXiv preprint arXiv:2301.21508, 2023. |
| [4] | J. Lewis, E. Perez, A. Piktus, F. Petroni, V. Kapurkhina, N. Goyal, J. Kittler, M. Lewis, W. Y. Yu, T. Rocktshild et al., “Retrieval- augmented generation for knowledge-intensive tasks,” Advances in Neural Information Processing Systems, vol. 33, pp. 4959–4974, 2020. |
| [5] | S. Borgeaud, A. Mensch, J. Hoffmann, T. C. E. Rutherford, K. Mill- gan, G. B. Van Den Driessche, J.-B. Lespiau, B. Damoc, A. Clark et al., “Improving language models by retrieving from millions of tokens,” in International Conference on Machine Learning, PMLR, 2022, pp. 202–212. |
| [6] | L. Ouyang, J. Wu, K. Jiang, D. Almeida, C. Wainwright, M. Pinski, H. Zhang, D. Xu, A. Slavkovik, A. et al., “Training language models to follow instructions with human feedback,” in Advances in Neural Information Processing Systems, 2022, pp. 7370–7374. |
| [7] | X. Ma, Y. Gong, P. He, H. Zhao, and N. Xu, “Query rewriting for retrieval-augmented language models,” arXiv preprint arXiv:2210.01756, 2022. |
| [8] | I. BERT, J. Lin, “Knowledge-augmented language models: An overview,” https://github.com/knowledge-augmented-lm/knowledge-augmented-lm. |
| [9] | W. Peng, G. Li, Y. Jiang, Z. Wang, D. Ou, X. Zeng, E. Chen et al., “Retrieval-augmented long-tail knowledge in taboola search,” arXiv preprint arXiv:2311.03758, 2023. |
| [10]| S. S. Zhen, S. Mishra, X. Chen, H. C. Cheng, H. Chi, Q. V. Le, and D. Zhou, “Take a step back: Re-thinking bias in large language models,” arXiv preprint arXiv:2301.06157, 2023. |
| [11]| X. Luo, X. Ma, J. Lin, and C. Liang, “Precise zero-shot dense retrieval without relevance labels,” arXiv preprint arXiv:2210.22044, 2023. |
| [12]| V. Blagoev, “Enhancing rags pipelines in haystack: Introducing diver- sity skinner and indistinct-identifier,” https://haystack.deepset.ai/community/blog/enhancing-rag-pipelines-in-haystack-4514b2e5b5, 2023. |
| [13]| W. Yu, D. Iyer, S. Wang, Y. Xu, M. J. Sanyal, C. Mu, Z. Meng, and M. Jiang, “Generate rather than retrieve: Language models as strong content generators,” arXiv preprint arXiv:2206.10032, 2022. |
| [14]| Z. Shao, Y. Gong, Y. Shen, M. Huang, N. Duan, and W. Chen, “En- hancing retrieval-augmented language models with iterative retrieval- generative synergy,” arXiv preprint arXiv:2305.15963, 2023. |
| [15]| X. Wang, Q. Yang, Y. Ou, J. Liang, Q. He, Z. Gu, Y. Xiao, and W. Wang, “Knowledget: Enhancing large language models with retrieval and storage access on knowledge bases,” arXiv preprint arXiv:2301.18017, 2023. |
| [16]| A. H. Radmard, “Forget the future: rag-fusion,” https://towardsdatascience.com/forget-the-future-rag-fusion-114728d584d6, 2023. |
| [17]| X. Cheng, D. Luo, X. Chen, L. Liu, D. Zhao, and R. Yan, “Lift your retrieval-augmented text generation with self memory,” arXiv preprint arXiv:2305.02473, 2023. |
| [18]| S. Wang, Y. Xu, Y. Fang, Y. Liu, S. Sun, R. Su, C. Zeng, and M. Zeng, “Training data is more valuable than you think: A simple and effective method by retrieving from training data,” arXiv preprint arXiv:2303.08773, 2022. |
| [19]| X. Li, E. Nie, and S. Liang, “From classification to generation: insights into crosslingual pretrained icl,” arXiv preprint arXiv:2311.05025, 2023. |
| [20]| D. Cheng, S. Huang, J. Bi, Y. Zhan, J. Liu, Y. Wang, H. Sun, Y. Wei, D. Deng, and Q. Zhang, “Unise: Universal prompt retrieval for improving zero-shot evaluation,” arXiv preprint arXiv:2303.08815, 2023. |
| [21]| Z. Dai, Y. Y. Zhao, H. Ma, Y. Luan, J. Ni, A. Bakalov, K. Guo, K. B. Hall, and M.-W. Chang, “Promptbar: Few-shot dense retrieval from few examples,” arXiv preprint arXiv:2307.15502, 2023. |
| [22]| Z. Sun, X. Wang, Y. Yang, and X. Zhou, “Recitato: augmented language models,” arXiv preprint arXiv:2210.01196, 2022. |
| [23]| O. Khatab, K. Santhanam, X. Li, D. Hall, P. Liang, C. Potts, and A. Zaha, “Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive mp,” arXiv preprint arXiv:2212.14204, 2022. |
| [24]| Z. Jiang, F. Xu, K. Zhao, Z. Sun, Q. Liu, J. Dowdle-Yu, Y. Yang, J. Cullian, and G. Nguib, “Active retrieval augmented generation,” arXiv preprint arXiv:2305.06083, 2023. |
| [25]| A. Asizi, Z. Wu, X. Wang, A. Sil, and H. Hajishirzi, “Self-arg: Learning to retrieve, generate, and critique through self-reflection,” arXiv preprint arXiv:2301.11507, 2023. |
| [26]| Z. Ke, W. Kong, C. Li, M. Zhang, Q. Mei, and M. Bendersky, “Bridging the preference gap between models and humans,” arXiv preprint arXiv:2406.05941, 2023. |
| [27]| X. Y. Lin, X. Chen, M. Shen, M. L. Montel, R. James, P. Roz- zoni, J. Zhang, G. Chen, and S. Wei, “R-ask: Retrieval- augmented instructional tuning,” arXiv preprint arXiv:2301.01352, 2023. |
| [28]| O. Ovatida, M. Brief, M. Misheva, and O. Elishkin, “Fine-tuning of retrieval-augmented language models in,” arXiv preprint arXiv:2301.05943, 2023. |
| [29]| T. Lan, D. Chen, Y. He, X. Huang, and X. Ma, “Co-Training Representation Learning for Retrieval-Augmented Generation,” arXiv preprint arXiv:2301.05943, 2023. |
| [30]| H. Zhang, “Dense retrieval with retrieval-augmented generation,” arXiv preprint arXiv:2212.04683, 2023. |
| [31]| L. Liu and M. Zhuang, “Divide & conquer for entailment-aware multi-hop evidence retrieval,” arXiv preprint arXiv:2311.02616, 2023. |
| [32]| J. Liu, X. Liu, Y. Hu, Y. He, P. Huang, Y. Li, and X. Ma, “Diversity design space with retrieval-augmented training,” arXiv preprint arXiv:2210.14590, 2023. |
| [33]| Z. Guo, S. Cheng, Y. Wang, Y. Li, and Y. Lin, “Prompt-guided retrieval-augmented generation for knowledge-intensive tasks,” arXiv preprint arXiv:2305.17653, 2023. |
| [34]| M. Langari, A. Zardosht, E. Nguib, and G. Nguib, “Learning to filter context for retrieval-augmented generation,” arXiv preprint arXiv:2311.08377, 2023. |
| [35]| M. J. Seabrook, J. L. Joshi, and K. M. Lang, “Retrieval-augmented generation for low-resource domain tasks,” arXiv preprint arXiv:2402.14842, 2023. |
| [36]| Y. Ma, Y. Cao, Y. Hong, and S. An, “Large language model is not a good soft- information extractor, but a good ranker for hard information,” arXiv preprint arXiv:2303.05923, 2023. |
| [37]| D. Xu and H. Ji, “Retrieval-augmented generative question answering for event argument extraction,” arXiv preprint arXiv:2212.10767, 2022. |
| [38]| L. Wang, N. Yang, and F. Wei, “Learning to retrieve in context examples for large language models,” arXiv preprint arXiv:2307.07167, 2023. |
| [39]| S. Rajput, N. Mehta, A. Singh, R. Keshavanu, T. V. Lu, H. Heldt, L. Hong, Y. Yao, Q. Yan, S. Ransom et al., “Recommender systems with generative retrieval,” arXiv preprint arXiv:2305.05065, 2023. |
| [40]| J. Lin, B. H. Eng, G. Wang, C. Chen, T. Li, Z. Wang, Z. Li, Y. Li, H. Li, et al., “Language models as sensitive indexes,” arXiv preprint arXiv:2301.01785, 2023. |
| [41]| A. Ramath, T. Behl, D. Vokhmin, and S. Chapidi, “Context tuning for retrieval-augmented generation,” arXiv preprint arXiv:2312.07586, 2023. |
| [42]| J. Gazdar, P. Lewis, M. Loneli, K. Hossain, F. Petroni, T. Schick, J. Dowdle, V. A. Julian, S. Riedel, and E. Grave, “Few-shot learning with retrieval-augmented language models,” arXiv preprint arXiv:2309.22029, 2023. |
| [43]| J. Huang, W. Ping, P. Xu, M. Shoebyk, K. C.-C. Chang, and B. Catan- zaro, “Raven: Long-context learning with retrieval-augmented language models,” arXiv preprint arXiv:2308.07922, 2023. |
```

### --- Page 0018 ---

```markdown
| Reference                                                                 | Citation                                                                 |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [44] B. Wang, W. Ping, P. Yu, L. McAfee, Z. Liu, M. Shoeybi, Y. Dong,    | [66] Z. Wang, X. Pan, D. Yu, D. Yu, J. Chen, and H. J. "Zemi: Learning    |
| and K. Liu, "Shall we return autoregressive language models with retrieval?"| shorter semi-parametric models from multiple tasks," arXiv              |
| arXiv preprint arXiv:2304.07672, 2023.                                   | preprint arXiv:2102.00825, 2023.                                       |
| [45] A. K. Dubey, L. McAfee, P. Yu, L. M. Shoeybi, and D. Catania,       | [67] S.-Q. Yan, Y.-C. Gu, Y. Zhu, and Z.-H. Ling, "Corrective retrieval  |
| "Structure: Instruction-tuning for retrieval-augmented pretraining,"     | generation," arXiv preprint arXiv:2310.15884, 2023.                    |
| arXiv preprint arXiv:2307.01373, 2023.                                   | [68] B. Jain, L. B. Soares, and T. Kwiatkowski, "I-paper: One pass answer |
| [46] S. Sriramadhikari, R. Wetrasekera, E. Worn, T. K. Chandak, R. Rama, | generation and evidence retrieval," arXiv preprint arXiv:2306.16853,   |
| and S. N. Nayakakshara, "Improving the domain adaptation of retrieval    | 2023.                                                                   |
| augmented generation for knowledge-intensive question answering,"         | [69] H. Yang, Z. Li, Y. Zhang, J. Wang, N. Cheng, M. Li, and J. Xiao,    |
| Transactions of the Association for Computational Linguistics,           | "Prefitting black-box large language models for retrieval question answer-|
| vol. 11, pp. 1–17, 2023.                                                | ing via pluggable reward-driven control," arXiv preprint arXiv:2301.18447,|
| [47] A. Xu, C. Xiong, S. Yu, and Z. Liu, "Augmentation-enabled retrieval  | 2023.                                                                   |
| for diverse generative language models as a service plug-in," arXiv      | [70] Z. Zhang, B. Liu, B. Koopman, and G. Zuccon, "On-structure large    |
| preprint arXiv:2301.17331, 2023.                                        | language models are strong across short query likelihood models for      |
| [48] O. M. Trotter, D. Watson, O. Ram, and J. Berant, "Making retrieval-  | document ranking," arXiv preprint arXiv:2310.13243, 2023.              |
| augmented language models robust to irrelevant context," arXiv           | [71] F. Xu, W. Shi, and E. Choi, "Recomp: Improving retrieval-augmented  |
| preprint arXiv:2101.05125, 2023.                                        | models with compression and selective augmentation," arXiv preprint     |
| [49] T. H. Cheng, F. Xu, S. Aroca, and E. Choi, "Understanding the      | arXiv:2310.04085, 2023.                                               |
| retrieval-augmented long-form question answering," arXiv preprint       | [72] W. Shi, M. Min, M. Yasunga, S. Reams, M. Lewis, L. Zettlemoyer,    |
| arXiv:2210.12150, 2022.                                                  | and M. Y. "Repl: Retrieving augmented black-box language models," arXiv  |
| [50] W. Hu, Y. Zhang, J. Cheng, and D. Yu, "On-context retrieval:        | preprint arXiv:2210.12506, 2023.                                       |
| Enhancing black-box models with retrieval-augmented language models,"    | [73] E. Melhem, "Enhancing intelligent generation," arXiv preprint      |
| arXiv preprint arXiv:2210.12190, 2023.                                   | arXiv:2311.04177, 2023.                                               |
| [51] S. Xu, L. Zhang, J. Cheng, and T. Zhang, "Search-in-the-          | [74] H. Wang, W. Huang, Y. Deng, K. Fang, Y. Wang, F. M. J. P. J. Pan,   |
| knowledge: accurate, credible and traceable large language models for     | and K. W. "Unis: A unified multi-source retrieval-augmented generation   |
| knowledge-intensive tasks," arXiv preprint arXiv:2304.13242, 2023.     | for retrieval-augmented and personalized dialogue systems," arXiv       |
| [52] L. Ma, K. N. Izadi, A. Cacuril, L. Dagan, and M. Waisberg,          | preprint arXiv:2301.02563, 2024.                                       |
| "Optimizing retrieval-augmented reader models via token elimination,"    | [75] Z. Liu, C. Xu, P. Zhao, X. Geng, C. Ma, J. Qiao, L. Jin, and D. Jiang,|
| arXiv preprint arXiv:2301.13682, 2023.                                   | "Augmenting large language models with retrieval-augmented generation,"   |
| [53] I. O. Odugbemi, A. Shridhar, S. Cos, K. G. Roberson,                | arXiv preprint arXiv:2305.04757, 2023.                                 |
| and A. D. White, "Paper: Retrieval-augmented generation for             | [76] A. Xu, Z. Liu, C. Xiong, Y. Yu, Z. Li, Z. Lin, and G. Yu, "Structure- |
| document summarization," arXiv preprint arXiv:2305.02529, 2023.        | aware knowledge graph construction based on retrieval on structure,"     |
| [54] Y. Liu, J. Zhang, J. Finkel, S. Pilehvar, C. Ma, and A. K.         | arXiv preprint arXiv:2305.01921, 2023.                                 |
| K. K. "Retrieval-augmented generation: A survey," arXiv preprint       | [77] H. Shen, Y. Q. Huang, F. W. N. Wu, and B. W. "Retrieval-          |
| arXiv:2301.05149, 2023.                                                  | generation generation for end-to-end task-oriented dialogue systems,"    |
| [55] Z. Zhang, A. Y. Ren, S. Shi, M. Han, W. R. Lai, and                 | arXiv preprint arXiv:2301.08876, 2023.                                 |
| C. Zhao, "The logic, indications, and implementation framework for       | [78] T. Shi, L. Li, Z. Liu, Z. Qian, and Q. Wang, "Dual-faceted         |
| retrieval-augmented generation," in Proceedings of the 17th             | knowledge retrieval for task-oriented dialogue systems," arXiv         |
| International Conference on Natural Language Processing, 2023, pp. 1–14.| preprint arXiv:2301.15825, 2023.                                       |
| [56] N. Thakur, L. Bontchev, J. Li, Q. Liu, D. R. Rehbagh, and           | [79] R. Amado and A. J. "Fable: Flexible narrative generation using     |
| D. Alfonso-Hernandez, "You don’t know how to retrieve without           | retrieval-augmented narrative construction," arXiv preprint            |
| retrieval-augmented generation," arXiv preprint arXiv:2301.11361, 2023. | arXiv:2301.18846, 2023.                                               |
| [57] G. Kim, S. Kim, B. Jeon, J. Park, and J. Kang, "Two-tiered clarifi- | [80] X. Jiang, B. Zhang, Y. Xu, Q. Li, Y. Feng, Z. Wang, J. Tang,       |
| cation for answering ambiguous questions with retrieval-augmented        | H. Ding, X. Chao, J. Zhao et al., "Think and retrieve: A hypothesis     |
| language models," arXiv preprint arXiv:2310.14506, 2023.                | driven graph enhanced medical large language models," arXiv preprint   |
| [58] Y. Wang, P. Li, X. Liu, "Self-knowledge guided retrieval for       | arXiv:2312.15883, 2023.                                               |
| large language models," arXiv preprint arXiv:2301.00520, 2023.         | [81] J. Baek, S. Jeong, M. Jung, J. C. Park, and S. J. Hwang,          |
| [59] E. Zheng, K. Ren, D. Zhao, M. Yang, and D. Q. "Retrieval-         | "Knowledge-aware retrieval generation," arXiv preprint                 |
| generation for augmented large language models," arXiv preprint         | arXiv preprint arXiv:2301.12896, 2023.                                 |
| arXiv:2301.01549, 2023.                                                  | [82] L. Luo, Y.-F. Li, A. Jaffari, and P. San, "Reasoning on graphs:     |
| [60] P. Xu, W. Yu, L. McAfee, C. Zhu, Z. Liu, S. Zeng,                  | Faithful and interpretable large language model reasoning," arXiv      |
| B. Bakhtin, M. Shoeybi, and D. Catania, "On retrieval-                  | preprint arXiv:2301.06100, 2023.                                      |
| context large language models," arXiv preprint arXiv:2301.20325, 2023. | [83] X. He, X. Y. Tan, N. Sun, N. V. Chawla, T. Laurent, Y. LeCun,     |
| [61] H. Trivedi, N. Balasubramanian, T. Koth, and A. Sabharwal,         | Z. Besson, and B. H. Bodé, "G-retriever: Retrieval-augmented generation |
| "Interleaving retrieval with chain-of-thought reasoning for              | for textual graph understanding and question answering," arXiv         |
| knowledge-intensive multi-step queries," arXiv preprint                 | preprint arXiv:2402.07030, 2024.                                       |
| arXiv:2202.01250, 2022.                                                  | [84] Z. Zhao, L. Li, B. Wang, G. Huang, S. Yang, J. Yuan, C. Su,      |
| [62] R. Ren, Y. Wang, Y. Xu, X. Zhao, L. Tian, H. Wu, J. Li, and       | L. A. Su et al., "TableBot: Towards utilizing tables, natural language  |
| H. Wang, "Investigating factual knowledge boundary of large language     | and commands into one agent," arXiv preprint arXiv:2307.08674, 2023.  |
| models with retrieval-augmented generation," arXiv preprint             | [85] M. Kaur, M. G. Gunatna, R. Srinivasan, and H. Jin, "Some recent    |
| arXiv:2307.11019, 2023.                                                 | seeking language generation using retrieval-augmented retrieval and      |
| [63] A. Sarith, S. Adhikari, T. S. Khanna, A. Goldie, and D. C.        | knowledge graph generation," in Proceedings of the 2023 International  |
| Manning, "Raptor: Recursive assistance processing for organized           | Conference on Artificial Intelligence, vol. 36, no. 10, pp. 102–107,   |
| retrieval," arXiv preprint arXiv:2401.15089, 2023.                     | 2023.                                                                   |
| [64] R. Vam, Y. Levine, I. Dalal, D. Moshyia, A. Shashku, L. Leyton-   | [86] F. Shi, X. Chen, K. Min, S. Scales, D. Doan, F. Chen, H. Scharli,  |
| models," arXiv preprint arXiv:2302.00083, 2023.                         | and D. Xu, "Large language models can be easily adapted to retrieval   |
| [65] A. R. G. Poo, F. Fang, W. Ma, and Z. Lin, "Retraining: Document     | relevant context," in International Conference on Machine Learning,     |
| level event extraction via knowledge augmentation," in Proceedings of    | 2023, pp. 312–321.                                                    |
| the 61st Annual Meeting of the Association for Computational Linguistics  | [87] R. Title, "Evaluating the ideal chunk size for a retrieval system   |
| (Volume 1: Long Papers), 2023, pp. 293–306.                             | using llamaindex," https://www.llamaindex.ai/blog/evaluating-the-ideal-|
|                                                                           | chunk-size-for-a-rap-system-using-llamaindex-6207e5d3ec5, 2023.       |
```

### --- Page 0019 ---

```markdown
| Reference | Citation |
|-----------|----------|
| [89]      | Langchain, "Recurrently split by character," https://python.langchain.com/docs/modules/data_connection/document_transformers/retrieval_text_splitter, 2023. |
| [90]      | So, Y., "Advanced rage 01: Small-to-medium language models," https://towardsdatascience.com/advanced-rage-01-small-to-medium-language-models-1218b396d43, 2023. |
| [91]      | Wang, H., Li, N., da Rosa, A., Siu, R., Zhang, D., and Derr, "Knowledge prompting for multi-document question answering," arXiv preprint arXiv:2301.17303, 2023. |
| [92]      | Zhu, N., Schirhagl, H., Wei, N., Scalevs, X., Wang, D., Shumams, C., Cui, O., Bousquet, G. L., "Least-to-most prompting enables complex reasoning in large language models," arXiv preprint arXiv:2205.10625, 2022. |
| [93]      | Dhulipala, M., Koneli, J., Xu, R., Raileanu, X. L., A. Celiklimiz, J., and Weston, "Chain-of-verification reduces hallucination in large language models," arXiv preprint arXiv:2309.14915, 2023. |
| [94]      | Li, X. and Li, J., "Angle-optimized text embeddings," arXiv preprint arXiv:2309.12871, 2023. |
| [95]      | Vaggelas, V., "Language embedding methods," https://blogs.oyacel.com/2023/02/09/language-embedding-methods/ |
| [96]      | BAAL, "FlagEmbedding," https://github.com/FlagOpen/FlagEmbedding, 2023. |
| [97]      | Zhang, F., Xiao, Z., Liu, D., Dou, J., and Yi, J., "Retrieving answers to augment large language models," arXiv preprint arXiv:2310.07554, 2023. |
| [98]      | Pei, L., Liu, H., Hewitt, A., Paredes, R., Belciug, A., Petroni, F., and Li, X., "Lost in the middle: How language models use long contexts," arXiv preprint arXiv:2307.13702, 2023. |
| [99]      | Gao, T., Sheng, Y., Xiang, Y., Xiong, H., Wang, J., and Zhang, C., "Towards interactive and explainable human-AI dialogue," arXiv preprint arXiv:2303.14524, 2023. |
| [100]     | Anderson, C., Wilson, S., and D. S. Richardson, "Language addressing for service interaction and content generation," in Proceedings of the 2022 International Conference on Machine Translation, pp. 202-209, 2022. Available: https://aclanthology.org/2022.amta-1.19. |
| [101]     | Huang, Q., Liu, L., Luo, D. L., C. Li, Y. Lin, Y., "Unifying knowledge augmentation and enhancing links in long context language model compression," arXiv preprint arXiv:2301.08369, 2023. |
| [102]     | Karapkin, B., Oğuz, S., Min, P., Levis, L. W., Edunov, D., Chen, W., and Y. H., "Dense passage retrieval for open domain question answering," arXiv preprint arXiv:2204.09220, 2022. |
| [103]     | Mo, D., Yang, Y., Liu, Y. A., Zaid, X., Muttura, R., Jha, A., and Awadallah, A., Celiklimiz, Y., "Qusum: A new benchmark for query-based multi-domain meeting summarization," arXiv preprint arXiv:2207.05023, 2022. |
| [104]     | Dasgupta, K., Li, L., Bellay, A., Cahan, N., and Sharma, M., "Gardens of information-seeking bots and answers, and their secrets," arXiv preprint arXiv:2105.00101, 2021. |
| [105]     | Müller, A., Reina, R., Jayakumar, M., and Pitsch, "Covid-qa: A new semantic dataset for covid-19," ACL 2021, https://aclanthology.org/2021.covid-19.200, 2021. |
| [106]     | Wang, G. H., Chen, D. Song, Z., Zhang, Z., Chen, Q., Xiao, F., Jiang, J., Li, X., Wang, B., Wang, Y., "Chain-A: Comprehension benchmark in Chinese," arXiv preprint arXiv:2308.08823, 2023. |
| [107]     | Yang, S., Yue, and Y., "Auto-ep for online decision making: Benchmarks and additional opinions," arXiv preprint arXiv:2306.03224, 2023. |
| [108]     | T. Schick, I. Dziuda, V. R. Desai, R. Rinaldi, L. Monelli, L. Zettlemoyer, N. Candea, and T. Scialom, "Toolformer: Language models teach themselves to use tools," arXiv preprint arXiv:2302.06741, 2023. |
| [109]     | Zhang, "Graph-to-compiler: To improve with graph reasoning ability via prompt augmented by chatgpt," arXiv preprint arXiv:2306.14110, 2023. |
| [110]     | Rakhesh, J., Hilton, S., Balaji, W. J., O. L. Gunter, C. Kim, I. Esses, J., Jain, K., Vossaraju, W., Saunders et al., "WebGPT: Browser-assisted question answering with human feedback," arXiv preprint arXiv:2301.09213, 2022. |
| [111]     | Kwiatkowski, J., Polamoki, O., Redfield, M., Collins, A., Prikhodko, C., Albert, B., Epstein, I., Poloshkin, J., Devlin, J., and Weston, "Natural questions: a benchmark for question answering research," Transactions of the Association for Computational Linguistics, vol. 7, pp. 453–466, 2019. |
| [112]     | Liu, S., Yazar, R., Meng, M., Moorthy, S., Joy, C., Xiong, Y., and Zhou, "Exploring the integration strategies of retriever and large language models," arXiv preprint arXiv:2205.12754, 2022. |
| [113]     | Joshi, C., Choi, D. S., Weld, D. I., and Zettlemoyer, T., "Triviaqa: A large scale dataset for reading comprehension," arXiv preprint arXiv:1705.10840, 2017. |
| [114]     | Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P., "Squad: 100,000 questions for reading comprehension of text," arXiv preprint arXiv:1606.05250, 2016. |
| [115]     | Jernite, Y., Choi, F., Frostig, P., and Liang, P., "Semantic parsing on freebase from question-answer pairs," in Proceedings of the 2016 conference on empirical methods in natural language processing, 2013, pp. 1533–1544. |
| [116]     | Malin, A., Asai, Y., Dey, R., Hajiishir, D., and Khoshabi, "When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric methods," arXiv preprint arXiv:2212.10517, 2022. |
| [117]     | Nguyen, N., Rosenberg, X., Song, J., Gao, S., Tiwary, R., Majumder, R., and Deng, M., "More human-readable compression datasets," 2026. |
| [118]     | Yang, P., Di, S., Zhang, Y., Beniog, W., Chen, R., Sakhalikhov, M., and C. D. M., "Diverse dialogue: A dataset for diverse, explainable multi-hop question answering," arXiv preprint arXiv:2307.06090, 2023. |
| [119]     | Ho, X. H., A. D. Nguyen, S. Sugawara, and A. Aizawa, "Constructing a benchmark for comprehensive evaluation of reasoning steps," arXiv preprint arXiv:2011.06016, 2020. |
| [120]     | Thirunavukarasu, R., Kith, M., and Sabharwal, "Missing links: A new benchmark for knowledge-based conversational agents," Transactions of the Association for Computational Linguistics, vol. 10, pp. 539–554. |
| [121]     | Kokić, S., Shvarz, P., Blunsom, P., Cyer, K. M., Hermann, G., Mehta, E., "Greensets: The transferable reasoning comprehension challenge," in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 317-328, 2022. |
| [122]     | K.-H. Lee, C. Reth, J. Furtak, J. Cerny, and J. Fischer, "A human-inspired reading agent with self-improvement," arXiv preprint arXiv:2204.92772, 2022. |
| [123]     | Zhang, M., Liu, H., Ding, J., and M.-W. Chang, "Asqa: Factoid questions from answers," arXiv preprint arXiv:2204.09220, 2022. |
| [124]     | Mo, D., Mo, Y., Liu, Y. A., Zaid, X., Muttura, R., Jha, A., and Awadallah, A., Celiklimiz, Y., "Qusum: A new benchmark for query-based multi-domain meeting summarization," arXiv preprint arXiv:2207.05023, 2022. |
| [125]     | Dasgupta, K., Li, L., Bellay, A., Cahan, N., and Sharma, M., "Gardens of information-seeking bots and answers, and their secrets," arXiv preprint arXiv:2105.00101, 2021. |
| [126]     | Müller, A., Reina, R., Jayakumar, M., and Pitsch, "Covid-qa: A new semantic dataset for covid-19," ACL 2021, https://aclanthology.org/2021.covid-19.200, 2021. |
| [127]     | Wang, G. H., Chen, D. Song, Z., Zhang, Z., Chen, Q., Xiao, F., Jiang, J., Li, X., Wang, B., Wang, Y., "Chain-A: Comprehension benchmark in Chinese," arXiv preprint arXiv:2308.08823, 2023. |
| [128]     | Yang, S., Yue, and Y., "Auto-ep for online decision making: Benchmarks and additional opinions," arXiv preprint arXiv:2306.03224, 2023. |
| [129]     | T. Schick, I. Dziuda, V. R. Desai, R. Rinaldi, L. Monelli, L. Zettlemoyer, N. Candea, and T. Scialom, "Toolformer: Language models teach themselves to use tools," arXiv preprint arXiv:2302.06741, 2023. |
| [130]     | Zhang, "Graph-to-compiler: To improve with graph reasoning ability via prompt augmented by chatgpt," arXiv preprint arXiv:2306.14110, 2023. |
| [131]     | Rakhesh, J., Hilton, S., Balaji, W. J., O. L. Gunter, C. Kim, I. Esses, J., Jain, K., Vossaraju, W., Saunders et al., "WebGPT: Browser-assisted question answering with human feedback," arXiv preprint arXiv:2301.09213, 2022. |
| [132]     | Kwiatkowski, J., Polamoki, O., Redfield, M., Collins, A., Prikhodko, C., Albert, B., Epstein, I., Poloshkin, J., Devlin, J., and Weston, "Natural questions: a benchmark for question answering research," Transactions of the Association for Computational Linguistics, vol. 7, pp. 453–466, 2019. |
| [133]     | Liu, S., Yazar, R., Meng, M., Moorthy, S., Joy, C., Xiong, Y., and Zhou, "Exploring the integration strategies of retriever and large language models," arXiv preprint arXiv:2205.12754, 2022. |
| [134]     | Joshi, C., Choi, D. S., Weld, D. I., and Zettlemoyer, T., "Triviaqa: A large scale dataset for reading comprehension," arXiv preprint arXiv:1705.10840, 2017. |
| [135]     | Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P., "Squad: 100,000 questions for reading comprehension of text," arXiv preprint arXiv:1606.05250, 2016. |
| [136]     | Jernite, Y., Choi, F., Frostig, P., and Liang, P., "Semantic parsing on freebase from question-answer pairs," in Proceedings of the 2016 conference on empirical methods in natural language processing, 2013, pp. 1533–1544. |
| [137]     | Malin, A., Asai, Y., Dey, R., Hajiishir, D., and Khoshabi, "When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric methods," arXiv preprint arXiv:2212.10517, 2022. |
| [138]     | Nguyen, N., Rosenberg, X., Song, J., Gao, S., Tiwary, R., Majumder, R., and Deng, M., "More human-readable compression datasets," 2026. |
| [139]     | Yang, P., Di, S., Zhang, Y., Beniog, W., Chen, R., Sakhalikhov, M., and C. D. M., "Diverse dialogue: A dataset for diverse, explainable multi-hop question answering," arXiv preprint arXiv:2307.06090, 2023. |
| [140]     | Ho, X. H., A. D. Nguyen, S. Sugawara, and A. Aizawa, "Constructing a benchmark for comprehensive evaluation of reasoning steps," arXiv preprint arXiv:2011.06016, 2020. |
| [141]     | Thirunavukarasu, R., Kith, M., and Sabharwal, "Missing links: A new benchmark for knowledge-based conversational agents," Transactions of the Association for Computational Linguistics, vol. 10, pp. 539–554. |
| [142]     | Kokić, S., Shvarz, P., Blunsom, P., Cyer, K. M., Hermann, G., Mehta, E., "Greensets: The transferable reasoning comprehension challenge," in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 317-328, 2022. |
| [143]     | K.-H. Lee, C. Reth, J. Furtak, J. Cerny, and J. Fischer, "A human-inspired reading agent with self-improvement," arXiv preprint arXiv:2204.92772, 2022. |
| [144]     | Zhang, M., Liu, H., Ding, J., and M.-W. Chang, "Asqa: Factoid questions from answers," arXiv preprint arXiv:2204.09220, 2022. |
```

### --- Page 0020 ---

```markdown
| Reference                                                                 | Citation                                                                 |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [136] X. Z. Guo, W. Wu, Z.-Y. Ni, W. Hu, W. Wang, and S. Wang,           | "Learning to use open-domain conversation with long-term persona memory," |
|     "arXiv preprint arXiv:2203.05797, 2022."                             |                                                                         |
| [137] T. H. W. M. Gasic, N. Mrksic, I. M. Rojas-Barahona, P.-H.         | "On the use of conditional generation and subtask learning in neural     |
|     Su, S. U. Vandyke, and S. Young,                                    | dialogue systems," arXiv preprint arXiv:1606.03326, 2016.              |
| [138] R. He and I. McAuley, "Ups and downs: Modeling the visual evolution | "Proceedings of the 25th international conference on world wide web,"   |
|     of fashion trends with one-class collaborative filtering,"           | 2016, pp. 507–517.                                                    |
|     arXiv preprint arXiv:1704.09199, 2021.                              |                                                                         |
| [139] S. Li, H. Ji, and J. Han, "Document-level event argument extraction | "arXiv preprint arXiv:2104.05919, 2021."                               |
|     by conditional generation,"                                           |                                                                         |
| [140] J. Ebner, R. K. Culkin, K. Rawlins, and B. Van Durme, "Multi-     | "arXiv preprint arXiv:1911.03710, 2019."                              |
|     sentence argument linking,"                                           |                                                                         |
|     arXiv preprint arXiv:1911.03710, 2019.                              |                                                                         |
| [141] H. Elshaar, P. Wouklouklis, A. Remaci, C. Graxier, J. Hart, E.    | "LIM-OUT-RET: A large scale alignment of dialogue and knowledge base    |
|     Talbot, and L. M. L. Ture, "An evaluation of the Evaluation          | "International Conference on Language Resources and Evaluation (LREC)   |
|     Framework for Evaluating Human-Computer Interaction,"                | 2018, 2018.                                                            |
|     arXiv preprint arXiv:2203.05780, 2022.                              |                                                                         |
| [142] O. Levy, M. Seo, C. Choi, and L. Zettlemoyer, "Zero-shot           | "arXiv preprint arXiv:1706.01145, 2017."                              |
|     extraction via reading comprehension,"                                |                                                                         |
|     arXiv preprint arXiv:1706.01145, 2017.                              |                                                                         |
| [143] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi,        | "Hawkes: Can a machine really finish your sentence?"                    |
|     "Hellaswag: A new benchmark for grounded language understanding,"    | "arXiv preprint arXiv:1905.07830, 2019."                              |
|     arXiv preprint arXiv:1905.07830, 2019.                              |                                                                         |
| [144] S. Kim, S. J. Lo, D. Kim, J. Jang, S. Ye, J. Shin, and M. Wang,   | "The role of context in language understanding,"                        |
|     "Context modeling for thought-in-time language generation,"          | "arXiv preprint arXiv:2003.10445, 2020."                              |
|     arXiv preprint arXiv:2003.10445, 2020.                              |                                                                         |
| [145] A. K. Kothari, M. K. Khapar, K. Sankaranarayanan, and S. Chandra, | "Retrieving relevant answer pairs with a knowledge graph,"              |
|     "A knowledge graph-based approach for question answering,"           | "Journal of the Association for Information Science and Technology, vol. |
|     32, no. 1, 2021.                                                    |  102–110, 2021."                                                       |
| [146] D. Hendricks, C. Burns, S. Basart, A. Zou, M. Naciri, D. Song,    | "Measuring multi-task language understanding,"                          |
|     and J. Steinhardt, "Measuring robust multi-task language             | "arXiv preprint arXiv:2109.03000, 2021."                              |
|     understanding,"                                                       |                                                                         |
|     arXiv preprint arXiv:2109.03000, 2021.                              |                                                                         |
| [147] X. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer         | "arXiv preprint arXiv:1606.06784, 2016."                              |
|     sentinel memory networks,"                                           |                                                                         |
|     arXiv preprint arXiv:1606.06784, 2016.                              |                                                                         |
| [148] M. Wu, D. Khashabi, E. Segal, T. Khot, D. Brant, and J. Brant,    | "Did available to suggest a question answering,"                        |
|     "Implicit knowledge strategies,"                                     | "Transactions of the Association for Computational Linguistics, vol. 9,  |
|     "Proceedings of the 19th International Conference on Computational  | pp. 346–361, 2021."                                                  |
|     Linguistics, 2022."                                                 |                                                                         |
| [149] M. Alona, V. Alschuler, C. Christodoulopoulos, and A. Mitall,     | "Preventing large-scale abuse for extraction and verification,"          |
|     "arXiv preprint arXiv:2103.00355, 2021."                            |                                                                         |
| [150] J. Kothari and F. Tofigh, "Explainable automated feedback for      | "Public health data," arXiv preprint arXiv:2102.00620, 2021."         |
|     multi-turn dialogue,"                                                |                                                                         |
|     arXiv preprint arXiv:2102.00620, 2021.                              |                                                                         |
| [151] R. Tabel, D. Granger, and M. Auli, "Neural text generation for    | "A case study with application to the biography domain,"                |
|     summarization,"                                                      | "arXiv preprint arXiv:2007.07116, 2020."                              |
| [152] H. Hayashi, F. Budanov, P. Wang, C. Ackerson, R. Neervan,       | "G. and Neugib, "Wiki-qa: A dataset for multi-domain aspect-based      |
|     summarization,"                                                      | "Transactions of the Association for Computational Linguistics, vol. 9,  |
|     pp. 211–225, 2021."                                                 |                                                                         |
| [153] S. Naryanas, S. B. Cohen, and M. Lapa, "Don't give me the details,| "Just the summary: topic-aware conversational neural models for ex-     |
|     "arXiv preprint arXiv:2008.08745, 2018."                            |                                                                         |
| [154] S. Saha, J. A. Moundi, A. Saleh, A. Sharma, R. M. Rahouti,       | "I. Ahmed, N. Mohammed, and M. R. Amin, "Vico-lens: A novel          |
|     "image for extracting knowledge from different forms of annotated    | "Proceedings of the 2022 Workshop on Bangla Language Processing (BLP-  |
|     social networks,"                                                    | 2022), 2022, pp. 72–79."                                             |
| [155] X. Li and D. Roth, "Learning question classifiers,"                | "in COLING 2022: The 19th International Conference on Computational    |
|     Linguistics, 2022."                                                 |                                                                         |
| [156] S. R. K. Peryal, Y. Wu, J. Chuang, C. D. Mungania, A. Y. Ng,     | "and C. P. Rots, "Recursive deep models for semantic composition,"      |
|     "in Proceedings of the 2013 conference in empirical methods in      | "natural language processing, 2013, pp. 1631–1642."                   |
|     natural language processing, 2013, pp. 1631–1642."                  |                                                                         |
| [157] H. Hussain, H.-H. Wu, T. Gazit, M. Allamanis, and M. Brockschmidt,| "Codesearch challenge: Evaluating the state of the art in code search," |
|     "arXiv preprint arXiv:1909.03496, 2019."                            |                                                                         |
| [158] K. Cobe, V. Kosaraju, M. Baartman, M. Chen, H. Jun, L. Kaiser,   | "M. Puppel, J. Tworkowski, H. Rzhanov, and N. Zakharov, "Training        |
|     "to solve real world problems,"                                      | "arXiv preprint arXiv:2104.11468, 2021."                              |
| [159] R. Steinberg, K. Pouchkarev, A. Mijares, C. E. Eyrie, E. Tulis,   | "and D. Varga, "The re-acquisition of multilingual and parallel corpus  |
|     "with 20+ languages," arXiv preprint arXiv:0609.0506, 2006."       |                                                                         |
| [160] Y. Hoshi, D. Hiyama, Y. N. K. Tsutsumi, Y. Morikado, O. Tori,     | "and I. Deguchi, "Nile: A framework for developing and evaluating      |
|     "retrieval-augmented large language models," arXiv preprint arXiv:  |
|     2308.12063, 2023.                                                   |                                                                         |
| [161] J. Liu, "Building production-ready rag applications,"              | "https://www.ai.engineer/summit/schde/building-production-ready-rag-   |
|     applications, 2023."                                                |                                                                         |
| [162] I. Nguyen, "Evaluating rag: How to evaluate document retrieval,"   | "https://www.deepai.org/blog/rag-evaluation-retieval, 2023."          |
| [163] L. Leung, K. Uhlrich, and A. Polyzos, "Best practices for         | "in evaluation of rag applications," https://www.databirks.com/blog/   |
|     LIM-out-eval-best-practices-RAG, 2023."                             |                                                                         |
| [164] E. S. M. Le, P. Cress, D. Potts, and M. Zaharia, "Ragas:          | "Automated evaluation for retrieval-augmented generation," arXiv       |
|     preprint arXiv:2203.15217, 2023."                                   |                                                                         |
| [165] J. Saadoun, A. Kothari, C. Potts, and M. Zaharia, "An            | "automated evaluation framework for retrieval-augmented generation      |
|     systems," arXiv preprint arXiv:2211.04976, 2023."                  |                                                                         |
| [166] J. Chen, H. Lin, X. Han, and L. Sun, "Benchmarking large language  | "models," arXiv preprint arXiv:2203.50132, 2022."                     |
| [167] Y. Liu, Z. Li, S. Chen, H. Lin, Z. Zhou, F. Meng, J. Zhou, and    | "Y. Xu, L. Li, S. Wang, and X. Zhang, "Evaluating large language models,"|
|     "arXiv preprint arXiv:2301.08473, 2023."                            |                                                                         |
| [168] Y. Liu, Z. Li, S. Wang, and X. Zhang, "Evaluating large language   | "models," arXiv preprint arXiv:2301.08473, 2023."                     |
| [169] Y. Yu, Z. Li, X. Niu, K. Feng, B. Tang, W. Wang, H. Liu, H.      | "X. Liu, and E. Chen, "Cud-rag: A benchmark for generative models in    |
|     retrieval-augmented generation of large language models,"            | "arXiv preprint arXiv:2401.74021, 2024."                              |
| [170] P. Xu, W. Ping, X. Wu, L. McLee, C. Zhu, Z. Liu, S. Bursztyn,     | "E. Bakhturina, M. Shoyb, and B. Zadorozhny, "Retrieval systems based   |
|     "on retrieval-augmented generation," arXiv preprint arXiv:2301.20325,| 2023."                                                                  |
| [171] C. Packer, Y. Fang, S. G. Patil, K. I. Stowell, and J. E. Gon-    | "zalez, "Multi-tasking in retrieval-augmented systems," arXiv         |
|     preprint arXiv:2301.68032, 2023."                                   |                                                                         |
| [172] G. Xiao, Y. Tian, B. Chen, S. Han, and J. Wells, "Efficient       | "streaming language models with attention sinks," arXiv preprint       |
|     arXiv preprint arXiv:2301.68032, 2023."                             |                                                                         |
| [173] Y. Zhang, S. Patil, A. Jain, S. Shen, M. Zaharia, I. Stoica, and   | "E. Gonzalez, "Rag: Adapting language model to domain specific rag,"    |
|     "arXiv preprint arXiv:2401.01351, 2023."                            |                                                                         |
| [174] A. Kaplan, S. McCandlish, T. Heinmuth, T. B. Brown, B. Chess,    | "R. Chui, G. Ray, A. Radford, W. D. Amodeo, "Scaling laws for         |
|     "real language models," arXiv preprint arXiv:2201.08861, 2022."    |                                                                         |
| [175] U. Alon, F. Xu, J. H. Lee, S. Sengupta, D. Roth, and G. Neugib,   | "Neuro-symbolic language modeling with automated-augmented retrieval,"   |
|     "in International Conference on Machine Learning, PMLR, 2022, pp.   |
|     468–485."                                                           |                                                                         |
| [176] M. Yavuz, A. Aghayev, A. W. Raines, J. Jeske, L. Pizlo, J.        | "Lezkov, P. L. Z. Liu, L. Zettlemoyer, and W. T. Y. "Retrieval-augmented  |
|     multi-modal language model," arXiv preprint arXiv:2211.08743, 2022.|                                                                         |
| [177] M. Liu, D. Li, S. S. Wares, and S. Ho, "BERT-2-Bootstrap language   |
|     image generation with freezing and merging large language models,"    |
|     arXiv preprint arXiv:2201.25292, 2022.                             |                                                                         |
| [178] W. Zhu, A. Yan, L. Yu, X. Xu, E. M. Eckstein, and W. Y.          | "Wang, "Visualize before you write: maximizing guided open-end         |
|     generation," arXiv preprint arXiv:2203.01205, 2022.                |                                                                         |
| [179] J. Zhao, G. Haffar, and E. Sharif, "Generating speech from        | "speech for speech translation," arXiv preprint arXiv:2201.08173, 2022.|
|     [180] D. Chan, S. Ghosh, A. Rastogi, and B. Hoffmeister, "Using     | "external speech recognition," arXiv preprint arXiv:2301.27368, 2023."|
|     automated end-to-end speech recognition,"                            |                                                                         |
|     arXiv preprint arXiv:2301.27368, 2023.                             |                                                                         |
```

### --- Page 0021 ---

```markdown
| Reference                                                                                                                             |
|---------------------------------------------------------------------------------------------------------------------------------------|
| [181] A. Yang, A. Nagarani, P. H. Seo, A. Miech, J. Pont-Tuset, I. Laptev, J. Šivic, and C. Schmid, “Vid2Sag: Large-scale pretraining of a visual language model for dense video captioning,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 10714–10726. |
| [182] N. Nashid, M. Sinha, and A. Mesbah, “Retrieval-based prompt selection for code-related few-shot learning,” in *2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)*, 2023, pp. 2450–2462. |
```


# ArXiv 2403.01432

### --- Page 0001 ---

```markdown
# Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge

**Heydar Soudani**  
Radboud University  
Nijmegen, The Netherlands  
heydar.soudani@ru.nl  

**Evangelos Kanoulas**  
University of Amsterdam  
Amsterdam, The Netherlands  
e.kanoulas@uva.nl  

**Faegheh Hasibi**  
Radboud University  
Nijmegen, The Netherlands  
faegheh.hasibi@ru.nl  

## ABSTRACT
Language Models (LMs) provide a vast amount of factual knowledge, exhibiting strong performance across diverse tasks and domains. However, it has been observed that the performance of these models diminishes when dealing with less-popular or low-frequency concepts, for example, in domain-specific applications. The two prominent approaches to enhance the performance of LMs on less frequent topics are Retrieval Augmented Generation (RAG) and fine-tuning (FT) over synthetic data. This paper explores and evaluates the impact of RAG and FT on customizing LMs in handling low-frequency entities in question answering tasks. We conduct extensive experiments on twelve LMs of varying size and type and different FT methods, data augmentation, and retrieval models. Our findings indicate that while FT boosts performance across entities of varying popularity, RAG surpasses FT by a large margin, particularly for less popular factual knowledge. Additionally, the success of both RAG and FT approaches is amplified by improving retrieval and data augmentation strategies. While beneficial for small LMs, requires extensive resources. To address this issue, we propose the new Stimulus RAG approach that surpasses the effectiveness of fine-tuning based approaches, thereby eliminating the need for the costly data augmentation and fine-tuning steps for enriching LMs with less popular factual knowledge. The code is available at https://github.com/informag/RAGvsFT.

## CCS CONCEPTS
- Computing methodologies → Natural language generation;  
- Information systems → Question answering; Novelty in information retrieval; Language models.

## KEYWORDS
Retrieval Augmented Generation, Fine Tuning, Data Augmentation

## ACM Reference Format:
Heydar Soudani, Evangelos Kanoulas, and Faegheh Hasibi. 2024. Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge. In *Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP '24), December 9–12, 2024, Tokyo, Japan*. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3673791.3689145

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s).  
SIGIR-AP '24, December 9–12, 2024, Tokyo, Japan  
© 2024 Copyright held by the owner/author(s).  
ACM ISBN 978-1-4503-7472-7/24/12. https://doi.org/10.1145/3673791.3689145

![Figure 1: Comparison of RAG and fine-tuning on StabelLM2 performance in question answering over factual knowledge. RAG-based approaches significantly enhance the performance of the vanilla StabelLM2, outperforming fine-tuning by a large margin. Our proposed sRAG approach outperforms all models, including the fine-tuning based approaches.](assets/page_0001_img_1.png)

## 1 INTRODUCTION
Language Models (LMs) exhibit outstanding capabilities in executing tasks that demand extensive memorization of factual data [13]. However, their memorization capabilities are constrained when dealing with less frequent entities [15, 27, 39, 51], and even the largest models may encounter the well-known "hallucination" problem [47] and temporal degradation [29]. Consequently, when LMs are intended for deployment in less resourced domains, customization becomes imperative to ensure optimal performance. A common example is within the industrial setup, where chatbots or Question Answering (QA) systems need to accurately answer users' questions about a proprietary knowledge graph or intra-company terminology with limited textual description [48, 60]. Retrieval-Augmented Generation (RAG) and Fine-Tuning (FT) stand out as the two prominent approaches for adapting LMs to specific domains [17, 41, 43, 48]. RAG retrieves relevant information from a document corpus and enhances LM's response generation through the implementation of in-context learning (ICL) [15, 60]. Conversely, FT approaches update model weights to become adept at recalling specific information and enhances memorization capabilities during inference [6]. In the context of less popular knowledge, where limited data is available, data augmentation methods are utilized to generate synthetic training data, serving as an initial step towards FT [49, 50]. Despite existing research on enhancing LM's memorization with RAG [39], no work to our knowledge
```

### --- Page 0002 ---

```markdown
# SIGIR-AP '24, December 9–12, 2024, Tokyo, Japan

## 1 INTRODUCTION

has compared RAG with knowledge obtained through FT, particularly for less popular knowledge.

In this paper, we aim to understand which approach is more appropriate for customizing LMs for less popular knowledge. Specifically, we seek to answer this research question: 

**(RQ1)** How does RAG compare to fine-tuning for question answering over less popular factual knowledge, and which factors affect their performance? To address this question, we conduct a comprehensive comparison of RAG and fine-tuning methods for less popular knowledge, assuming that textual descriptions, albeit limited, are available for a specific domain and application. We, therefore, collect Wikipedia documents related to QA datasets over long-tail entities and apply two methods of knowledge injection: parametric knowledge injection using FT and non-parametric knowledge injection using RAG. For the FT approach, the LM is fine-tuned with synthetically generated data from these documents using data augmentation approaches [3, 55]. For the RAG approach, we use retrieval to know the most relevant documents for a query. We investigate how the effectiveness of these methods is affected by the following aspects: (i) fine-tuning method; i.e., full FT vs. parameter efficient fine-tuning (PEFT), (ii) data augmentation method, (iii) LM type and size, i.e., decoder only vs. encoder-decoder models and varying size, ranging from 80M to 1B parameters, and (iv) retrieval model preferences. Through exhaustive experimentation on relevant LMs and different setup of fine-tuning, data augmentation, and retrieval models, we arrive at the following conclusions:

- **Fine-tuning method**: Comparing full FT with PEFT (i.e., QLora) for 1B LMs with less than 1 billion parameters, PEFT is more effective than PEFT in the downstream tasks. PEFT, however, preserves the reasoning ability of LMs (needed for RAG) and outperforms full FT when fine-tuned models are used in combination with RAG (Table 2).

- **Data augmentation method**: Comparing prompt-based and a state-of-the-art fine-tuned [55] QA generation models, the prompt-based method demonstrates better performance for the downstream task. This suggests that the high-quality synthetic data generated by large LMs can better assist LMs with memorizing new knowledge, compared to the greater volume of data generated by the fine-tuned model (Table 2).

- **LM type and size**: Comparing decoder-only with encoder-decoder LMs (Flan-T5 models of various sizes), decoder-only models outperform encoder-decoder models of similar size. Interestingly, larger LMs generally do not benefit from fine-tuning, while smaller ones do. Therefore, a small fine-tuned LM with RAG can perform on par or better than a large LM; e.g., StableLM 12B vs. Llama3 (8B) (Table 3).

- **Retrieval model**: Comparing retrievers with varying performance in the RAG system, we observe that as the popularity of factual knowledge increases, the performance of the retriever decreases (Figure 7). Moreover, the performance of the RAG system increases by using higher performance retrievers (Figures 1 and 8).

- **Fine-tuning vs. RAG**: Comparing these two knowledge injection methods, RAG substantially outperforms fine-tuning. Fine-tuned LMs combined with RAG either outperform or perform on par with vanilla LMs with RAG in all but one case (Figure 1).

While fine-tuning improves accuracy in answering factual questions, both with and without RAG, it demands a considerable amount of effort and resources. This leads us to our second research question: 

**(RQ2)** Can we avoid the costs of fine-tuning by developing an advanced RAG approach that surpasses the performance of a fine-tuned LM with RAG? To answer this question, we develop Stimulus RAG (SRAG), a new RAG approach that simulates an LM to generate the correct response based on the provided hint in the prompt. The hint is extracted from the top retrieved documents by the retrieval model. Our results demonstrate that Stimulus RAG outperforms all other combinations of fine-tuning, both with and without retrieve-then-generate RAG.

To summarize, this paper makes the following contributions:

- We study the effectiveness of fine-tuning and RAG approaches for question answering over less popular factual knowledge and compare the performance of these models across distinct setups: vanilla and fine-tuned models, both with and without RAG, using different data augmentation methods.

- We perform extensive experiments to understand how fine-tuned and RAG models are affected by four different factors: data augmentation method, fine-tuning method, LM type and size, and retrieval model.

- We propose a new RAG approach Stimulus RAG that outperforms all RAG and fine-tuning setups, thereby bypassing the need for expensive fine-tuning.

## 2 RELATED WORK

Parametric and Non-parametric Knowledge. It is demonstrated that larger pre-trained LMs reinforce a significant amount of world knowledge in their parameters (parametric knowledge) [39]. FT can update the parametric knowledge embedded in LMs and customize it for a specific domain [19, 41]. One of the guiding principles for FT is data availability, which is limited especially in specialized domains [17, 43]. Data augmentation (DA) addresses the data scarcity problem by generating task- and domain-relevant samples from existing unlabeled texts. A common DA approach for the RAG task is generating question-answer pairs through a four-step Pipeline, consisting of passage selection, answer extraction, question generation, and consistency filtering [30, 32, 55]. Ushio et al. [55] conducted an empirical study comparing the E2E generation approaches: Pipeline, Multitask, and End-to-End (E2E) and showed the E2E approach outperforms others in downstream tasks.

Furthermore, a large body of work shows that augmenting LMs with nonparametric knowledge (i.e., retrieved text chunks) enables much smaller models to match the performance of larger models [31]. In this method, known as Retrieval Augmented Generation (RAG), an information retrieval system is utilized to find relevant documents and adds them to the input prompt to enhance response generation of LMs [4, 5]. 

As interest grows in refining pre-trained LMs for particular tasks, the comparison of FT and RAG strategies under suitable conditions is becoming increasingly important. Mosbach et al. [41] explored the effectiveness of few-shot FT versus ICL for classification tasks in general domains. e.g. Luis Balaguer et al. [17] compared FT and RAG in answering agriculture and geography-specific questions.
```

### --- Page 0003 ---

```markdown
![Overview of parametric and non-parametric knowledge injection for less popular factual knowledge](assets/page_0003_img_1.png)

Ovadia et al. [43] assessed the performance on multiple-choice questions in specialized areas like anatomy, astronomy, biology, and prehistory. In contrast to these studies, we directly address the integration of less popular factual knowledge into LMs, comparing various retrievers, data augmentation, and fine tuning methods.  
**Less Popular Knowledge.** An entity's popularity in LMs is gauged by its frequency in the model's pre-training data [20, 40], often assessed through the entity’s occurrences in a large corpus [27] via entity linking [16, 56]. Due to the practical challenges of direct counting, e.g., annotation of large-scale collections with defined entities [26], proxies are defined to approximate the popularity of factual knowledge. Sun et al. [51] use the retrieval metrics and content density, while Macwe et al. [38] introduce the co-occurrence of the subject entity and related articles as a popularity proxy. Wikipedia experiences are among the most prevalent methods for measuring the popularity of entities [10, 36].  
**RAG Development.** RAG introduces a new approach in AI, combining the strengths of both retrieval-based and generative models [15]. The concept of RAG was created and popularized by Lewis et al. [31], who introduced a model that combines a dense passage retriever with a sequence-to-sequence model, called Retrieve-then-Generate. This approach demonstrated substantial improvements in knowledge-intensive tasks. Several parameters affect a RAG system's accuracy, including the relevance of the passages included in the prompt context, their position, and their number [6, 15].  
However, several works have argued that the Retrieve-then-Generate approach is not optimal for more complex tasks, and more advanced RAG systems are needed. Adaptive-RAG [23] defines a classification-based RAG system to decide which RAG model should be used based on the question type. RATT [60] trains a model to ignore documents that don’t help in answering the questions, thereby adapting LMs to domain-specific RAG. Generate-then-Retrieve (GTR) [1] argues that the Retrieve-then-Generate paradigm is insufficient when the answer must be obtained from multiple documents. This theory introduces the RGC pipeline, which first generates multiple queries and then retrieves information for the generated queries. Mallen et al. [39] found that for popular knowledge, using RAG can hurt performance, so they defined an adaptive retrieval system to ensure retrieval only where it is beneficial.  
We discuss that increasing the number of documents is not helpful and may introduce more noise into the LM's input. To address this problem, we propose a new stimulus RAG system that highlights parts of the input text most likely to contain the correct answer. This approach aids the LM in accurately identifying and extracting relevant information. Highlighting has been used in the literature of information retrieval for various purposes. Askari et al. [8] aim to generate synthetic documents for queries, highlighting keywords to create high-quality documents. Cho et al. [12] propose generating sub-entities and summary highlights to overlay on source documents, enabling users to quickly navigate through content. Li et al. [33] introduce a new prompting framework to provide black-box LMs with fine-grained, instance-specific guidance toward desired outputs. Our work differs from these in that we highlight sentences using a simple yet effective reranker model, which directly improves RAG’s performance.

## 3 METHODOLOGY

In this section, we introduce our evaluation framework (Figure 2), which is designed to assess the effectiveness of two knowledge injection methods: the parametric method using FT and the non-parametric method using RAG.

### 3.1 Task Definition

This study specifically focuses on factual knowledge [2] of entities, defined as information about particular attributes and characteristics of target entities, among various types of world knowledge [39]. Factual knowledge is defined as a triplet (subject, relationship, object) [39]. In this context, the question involves the subject and the relationship, while the answer corresponds to the object. By using these template questions, we ensure that LMs understand the question and derive the answers from the knowledge base. We select Wikipedia-based question-answering datasets focused on factual knowledge. This enables us to measure the popularity of entities and based on Wikipedia pages and also obtain the corresponding evidence document for each entity from Wikipedia. For each dataset, we select Wikipedia pages whose corresponding entities appear in the test dataset. This setup mirrors real-world industry practices, where entities and their textual descriptions relate to companies’ specific internal processes.

### 3.2 Knowledge Injection with Fine-tuning

LMs are primarily pre-trained on general domains. To customize LMs for specific knowledge or a particular domain, fine-tuning (FT), also known as parametric knowledge injection, is commonly used. However, FT requires a substantial amount of training examples, which are often unavailable for specific applications, such as those within a company. Data augmentation (DA) offers a solution to this training and shortage. To achieve parametric knowledge injection, we fine-tune LMs using synthetically generated question-answer (QA) pairs. Formally, given a set of documents $D = \{d_1, d_2, \ldots, d_n\}$, where $n$ is the number of documents, a QA pair generator $g_q$ is tasked with generating as many QA pairs as possible:

$$
g_q(d_i) = \{(q_i, a_i^1), (q_i, a_i^2), \ldots, (q_i, a_i^{m_i})\}
$$
```

### --- Page 0004 ---

```markdown
# SIGIR-AP '24, December 9–12, 2024, Tokyo, Japan

## LLM Input

You are a question-answer generator. Your goal is to generate question-answer pairs given the Context.

Example output: `{"question": "", "answer": ""}`

**Context:** SCONTEXT

1. Step 1: Identify spans that are likely to be answers to questions, identify as many as possible.
2. Step 2: For each identified span, generate a question.
3. Step 3: Respond to the question in only a few tokens concisely.
4. Step 4: Output in JSON format following [...]

Ensure that you distinctly label and delineate Steps 1, 2, 3, and 4. Let's think step by step:

### Figure 3: Input prompt for prompt-based QA pair generation.
We define a CoT prompt to outline the generation steps.

where $q_i$ and $d_m$ denote a question and answer generated for the document $d$, $m$ is the total number of generated QAs for the document, and the generated set $Q = \bigcup_{j=1}^m Q_j$ is then used for fine-tuning the LLM, which consists of a set of learnable parameters to predict the probabilities of future or masked tokens.

We employ two QA generation methods. The first one is the end-to-end (E2E) QA generation, where a fine-tuned sequence-to-sequence model generates QA pairs from $d$. We utilize the E2E approach by Ushio et al. [55], who employs a trained T5-large for paragraph-level QA generation and has been shown to be more robust and effective than the established pipeline approach. The term "E2E" is used because the QA generation process is divided into two sequential components, i.e., answer extraction and question generation. Instead, a QA pair $q_i$ is generated in one go. To train the E2E model, the training question-answer pairs $(q_i, a_i)$ are input to the training document $d_m$ converted into a flattened sentence $y_i$ the following transformation:

$$
T(q) = "question: {q}, answer: {a}"
$$

$y_i = "T(q_i, d_i) | | T(q_{i+1}, d_i) | | \ldots$

where each pair is textualized with the function $T(q, a)$, and the textualized QA pairs are concatenated using the separator `|`. The E2E QA generation function $g$ is then obtained by maximizing the conditional log-likelihood:

$$
g = \arg\max_{g} p(q | d)
$$

The second QA generation method is the prompt approach, in which the QA generator $g_a$ is an instruction-tuned LLM capable of reasoning over the input prompt $I$. In this approach, the QA pairs are generated as follows: $g_a(q) = E(M(d), p)$. We utilize Zephyr [54] with Chain of Thought (CoT) [57] prompting for QA generation, as demonstrated in Figure 3.

### Figure 4: Our proposed Stimulus RAG method. The Hint Extractor identifies the most relevant sentence from top-K documents ranked by the retriever. This sentence is then added to the beginning of the input prompt.

## 3.3 Knowledge Injection with RAG

The non-parametric knowledge injection is performed using RAG, which consists of two components: the Retriever and the Generator [6, 15].

**Retriever.** The first key component in a RAG system is a retriever $R$, which builds an index for a document corpus $D$. During inference, given an input sequence $q$, the retriever identifies and ranks relevant documents $D_q = R(d, q)$. In our retrieval process, we employ both sparse and dense retrievers. We utilize BM25 [4] as a sparse retriever due to its popularity and effectiveness. For dense retrievers, we employ DPR [28] and Contriever [22] methods. Both models convert textual data into vector representations using a transformer network. The similarity between the query $q$ and document $d$ is defined as $S(q, d) = \overline{q} \cdot \overline{d}$, where we compute the product between embedding vectors $\overline{q}$ and $\overline{d}$. We employ two independent BERT models, trained discriminatively using query-document pairs with negative samples from BM25. Contriever, on the other hand, is trained using a shared BERT model for query and document encoding, optimized using a contrastive loss. We also employ a two-step retrieval pipeline, which includes first-stage retrieval using BM25 and reranking using DPR [1, 34].

**Generator.** The second step involves a generator component responsible for synthesizing an answer, typically implemented via LMs. Generative LMs take a query and top-K ranked documents from $D_q$, denoted as $D^K = [d_1, \ldots, d_K]$, and generates a response by sequentially predicting the next token. Our RAG prompt prepends the documents before the query, following [15, 39].

## 3.4 Stimulus RAG

While the generic retrieve-then-generate framework of RAG is effective in answering factual knowledge [19], it sometimes struggles
```

### --- Page 0005 ---

```markdown
# Table 1: Statistics of the factual knowledge-based datasets.

| Dataset  | # QA | Rel. Typ. | Question form |
|----------|------|-----------|---------------|
| PorQA    | 14K  | 16        | Template      |
| WITQA    | 14K  | 32        | Model-assisted |
| EQ       | 17.3K| 24        | Template      |

# Table 2: Accuracy of vanilla and fine-tuned LMs, both with and without RAG. The RAG results are based on ideal retrieval. Statistically significant differences in the PEFT-Prompt rows are compared with other rows. Superscripts (a), (b), (c), and (d) indicate statistically significant differences (better or worse) compared to vanilla LM, PEFT-E2E, Full-E2E, and Full-Prompt, respectively, determined by the Wilcoxon test (p-value < 0.01).

| FT      | QA  | +F-RAG | +FT-RAG | +F-RAG |
|---------|-----|--------|---------|--------|
| PorQA   |     |        |         |        |
| FlanT5-small | 2.69 | 47.46  | 2.84    | 27.36  |
| PEFT    | E2E | 6.06   | 48.40   | 7.86   | 33.76  |
| PEFT    | Prompt | 7.01(a, c) | 63.94(b) | 9.04(a) | 42.68(a, b) |
| Full    | 5.19 | 12.63  | 10.98   | 21.87  |
| Full    | 8.55 | 46.88  | 15.52   | 53.92  |
| FlanT5-base | 6.01 | 73.03  | 6.07    | 53.92  |
| PEFT    | E2E | 7.53   | 70.34   | 0.88   | 51.30  |
| PEFT    | Prompt | 9.11(b) | 73.94(b) | 12.98(a,b,c,d) | 57.63(a,b,c,d) |
| Full    | 7.42 | 41.76  | 10.91   | 31.22  |
| Full    | 10.06 | 51.80  | 17.36   | 54.07  |
| FlanT5-large | 8.84 | 68.56  | 16.94   | 52.64  |
| PEFT    | E2E | 8.69   | 67.47   | 15.33  | 53.25  |
| PEFT    | Prompt | 11.24(a) | 71.27(a,b) | 18.76(a) | 60.64(a) |
| Full    | 11.75 | 27.31  | 14.79   | 23.17  |
| StableLM | 17.01 | 76.14  | 17.92   | 60.72  |
| PEFT    | E2E | 16.39 | 74.82  | 23.62  | 53.21  |
| PEFT    | Prompt | 21.75(a,b) | 82.90(a,b) | 27.23(a,b,c) | 68.21(a,b,c) |
| Full    | 14.23 | 57.87  | 13.22  | 24.26  |
| Full    | 25.73 | 32.50  | 23.22  | 30.07  |

![Distribution of sample counts across popularity buckets, defined by log10(pageviews) for PorQA and WITQA and log2(pageviews) for EQ.](assets/page_0005_img_1.png)

## 4 EXPERIMENTAL SETUP

**Datasets.** We conducted our experiments on three datasets focused on factual knowledge: PorQA [9], WITQA [3], and Entity Question (EQ) [46], all of which include long-tail entities; see Table 1 for statistics. PorQA is an open-domain QA dataset about long-tail entities, constructed from 16 diverse relationship types in Wikidata. EQ is another popular open-domain QA dataset that covers a long-tail entity distribution, using Wikipedia hyperlink counts as a proxy for entity frequency and sampling knowledge triples from Wikidata based on these frequency distributions. Since EQ does not provide Wikidata IDs for each entity, we use only about 80% of the questions, where the mention of the subject entity has a unique match with a Wikidata entity. WITQA is another entity-centric dataset that defines a different proxy for popularity. They argue that the popularity metric should be based on the occurrence of both the subject entity and the relation (unlike the subject-based popularity in PorQA and EQ). Therefore, they define the S-R count, which is the co-occurrence of the subject entity and relation predicate. However, they report pageviews in their dataset, and we use the pageview-based popularity to enable comparison across datasets. To analyze performance with respect to popularity, we divide the entities into five buckets based on their popularity levels; see Figure 5.

**Evaluation Metric.** Following previous studies [38, 39, 46], we report on the accuracy metric, where a prediction is considered correct if one of the ground truth responses matches a substring of the predicted response. While widely used [15, 27, 36], this metric is not without issues. A principal problem arises in determining
```

### --- Page 0006 ---

```markdown
| Dataset   | PorQA         | WITQA         | EQ           |
|-----------|---------------|---------------|--------------|
| Model     | #P  | -FT  | +FT  | -FT  | +FT  | -FT  | +FT  | -FT  | +FT  | -FT  | +FT  |
|-----------|-----|------|------|------|------|------|------|------|------|------|------|
|           |     |      |      |      |      |      |      |      |      |      |      |
| FlanT5-small | 80m | 2.69 | 7.26 | 47.46 | 61.93 | 8.76 | 18.30 | 40.76 | 59.47 | 2.84 | 39.27 | 36.42 | 62.68 |
| FlanT5-base  | 250m | 6.01 | 9.11 | 73.08 | 71.34 | 16.52 | 23.32 | 73.43 | 74.34 | 6.07 | 12.98 | 53.92 | 57.63 |
| FlanT5-large | 780m | 8.44 | 11.24 | 68.56 | 71.27 | 24.52 | 28.85 | 74.37 | 77.24 | 16.94 | 18.17 | 52.64 | 60.06 |
| TinyLama   | 1.1B | 17.50 | 18.24 | 74.39 | 74.87 | 45.12 | 67.65 | 78.84 | 80.26 | 21.12 | 24.57 | 61.01 | 61.01 |
| StableLM2  | 1.6B | 17.01 | 21.75 | 76.14 | 82.09 | 42.18 | 51.66 | 81.08 | 86.19 | 17.92 | 27.23 | 60.72 | 68.82 |
| MiniCPM    | 2B   | 14.16 | 15.47 | 75.86 | 75.86 | 37.61 | 44.94 | 73.71 | 75.43 | 15.31 | 22.92 | 54.63 | 62.70 |
| FlanT5-3B  | 3B   | 12.24 | 13.65 | 73.31 | 74.71 | 31.24 | 36.78 | 76.97 | 79.67 | 15.28 | 20.49 | 52.70 | 62.74 |
| Mistral    | 7B   | 21.47 | 30.70 | 80.25 | 78.44 | 51.36 | 58.29 | 86.05 | 83.90 | 25.78 | 34.01 | 68.60 | 69.96 |
| Zephyr     | 7B   | 23.25 | 35.48 | 86.75 | 80.20 | 58.33 | 63.74 | 83.68 | 89.69 | 29.20 | 38.52 | 62.45 | 67.89 |
| Llama2-chat| 7B   | 26.09 | 27.71 | 81.15 | 80.15 | 53.88 | 56.38 | 86.50 | 84.71 | 27.13 | 32.84 | 68.29 | 67.64 |
| Llama3-chat| 3B   | 32.27 | 35.29 | 81.54 | 61.88 | 61.58 | 85.71 | 85.51 | 83.65 | 31.73 | 36.85 | 68.64 | 68.64 |
| FlanT5-xl  | 11.3B| 11.26 | 15.94 | 75.19 | 74.98 | 30.40 | 42.89 | 78.19 | 81.44 | 12.48 | 23.16 | 59.67 | 62.80 |

![Table 3: Accuracy of vanilla and fine-tuned LMs, both with and without RAG. The RAG results are based on Ideal retrieval. The PFET method is used for fine-tuning, and prompting is utilized for QA generation. The best results in each column are shown in bold, and the best results in each row are underlined. Statistically significant differences in the +FT-RAG columns are compared with the -FT-RAG columns. Superscript (a) indicates statistically significant differences (better or worse) as determined by the Wilcoxon test (p-value < 0.01).](assets/page_0006_img_1.png)

For example, comparing the predicted response "Nathanson" with the ground truth "Jeff Nathanson," the prediction is considered incorrect. Another problem is that when the model generates multiple entries, any name from the ground truth entity, the response is incorrectly considered as correct. Recognizing these limitations, we acknowledge the necessity for a more advanced analysis of answer variations, which we leave for future research.

Language Models. We use several LMs, focusing on two main features: the backbone architecture (i.e., decoder-only and encoder-decoder) and model size, which ranges from 80 million to over 1 billion parameters. For the encoder-decoder models, we utilize five versions of the FlanT5 family [14], spanning from small to XXL. For the decoder-only models, we employ smaller LMs such as TinyLama [59], StableLM2 [9], and MiniCPM [21], which range from 1 to 2 billion parameters. Additionally, we incorporate larger LMs like Mistral [24], Zephyr [54], Llama2 [53], and Llama3 [5], which range from 7 to 8 billion parameters. As for instructions, we apply zero-shot prompting for generative prediction using a straightforward template. For non retrieved-augmented input, we use template: "Question: `<question>`", and for retrieved-augmented input, we use template: "Context: `<context>` Question: `<question>`".

Fine Tuning. We leverage training data for the PFET approach using two distinct data augmentation methods. To ensure a fair comparison between PFET and RAG with an ideal retriever, we generate QAs exclusively using the summary sections of Wikipedia pages. After generating QAs, we proceed to fine-tune LMs.

RAG. We utilize a variety of retrieval models to obtain relevant documents for the RAG approach, including BM25 [45], Contriever [22], DPR [28], and a two-stage re-ranker that combines BM25 with DPR, all implemented according to the BEIR benchmark [52]. Additionally, since the selected datasets do not contain grounded document evidence, we assume that the summary section of each Wikipedia page is the answer-containing document. We define an ideal retriever model as one that returns the summary paragraph as the top-ranked document, referred to as the Ideal retriever throughout the paper. We acknowledge that this assumption is not entirely accurate as some answers may be found in other subsections. However, our evaluation of the downstream task (Figures 6 and 8) demonstrates that the ideal retriever outperforms other retrievers.

Stimulus RAG. We select a DPR retriever for the hint extractor and set $K = 3$. We report on two variations of the SRAG approach: (i) SRAG(D), which utilizes the top-1 sentence as the hint, and (ii) SRAG(D), which inserts the entire document contacting the top-ranked sentence. For the instruction, we used the following template: "Context: `<hint>`<context> Question: `<question>`".

# RESULTS
In the following, we evaluate fine-tuning and RAG methods on different setups and answer our two research questions listed in Section 1.
```

### --- Page 0008 ---

```markdown
# SIGIR-AP '24, December 9–12, 2024, Tokyo, Japan
## Heydar Soudani, Evangelos Kanoulas, & Fagheh Hasibi

![The performance of the LMs with different combinations of FT and RAG. The +FT+RAG setup outperforms other setups across all models and datasets.](assets/page_0008_img_1.png)

Figure 6: The performance of the LMs with different combinations of FT and RAG. The +FT+RAG setup outperforms other setups across all models and datasets.

The RAG prompt. The results indicate a direct correlation between the performance of the retrieval model and the overall QA accuracy, underscoring the significant impact of the retrieval model on the effectiveness of the downstream task for both vanilla and fine-tuned LMs. Since the performance of RAG with DPR and BM25+DPR is comparable, we use DPR as the retrieval model for subsequent experiments.

### Analysis of RAG and FT per popularity.
Figure 6 illustrates QA accuracy across different popularity buckets, providing insight into the effectiveness of RAG and FT (using ideal retriever). It is evident that RAG significantly increases accuracy for the least popular entities, which aligns with the findings of Malam et al. [3]. Additionally, these figures demonstrate that FT enhances QA accuracy across all popularity levels, with the most notable improvements observed in the least popular buckets for StableLM and Llama3.

### Analysis of retrieved models per popularity.
Figure 7 compares the performance of retrieval models against the ideal retriever across different popularity buckets. The results indicate that retrieval effectiveness is higher for less popular entities compared to more popular entities. This is likely due to the limited occurrences of noisy documents for less popular entities. Figure 8 shows the QA system's accuracy using various retrieval models within the RAG framework across different popularity buckets for the FlanT5-base and StableLM2 models. The left figures display results for vanilla LMs, while the right figures show results for fine-tuned LMs. FT does not alter the pattern of accuracy across popularity buckets but shifts the overall accuracy higher. Interestingly, the accuracy decreases from the less popular bucket to the fourth popularity bucket across different retrieval types increases in the most popular bucket. This reduction in accuracy can be interpreted by the finding in Figure 7, which shows that the retriever's performance decreases as popularity increases. However, it appears that for popular entities, the LMs can ignore noisy information in the input prompt and rely on their internal knowledge to answer questions, resulting in a sudden increase in accuracy despite the retriever's lower performance in the most popular bucket.

![Recall@1 for retrieval models across different popularity levels shows that retrievers perform more effectively with less popular knowledge compared to more popular ones.](assets/page_0008_img_2.png)

Figure 7: Recall@1 for retrieval models across different popularity levels shows that retrievers perform more effectively with less popular knowledge compared to more popular ones.

![Performance as the answer generator acts across popularity buckets on PopQA. FT does not alter the overall pattern. Accuracy decreases as the retrieval models’ performance drops from the least popular bucket to the fourth bucket. Interestingly, accuracy increases for the most popular bucket, indicating that LMs rely on their embedded information for popular entities.](assets/page_0008_img_3.png)

Figure 8: Performance as the answer generator acts across popularity buckets on PopQA. FT does not alter the overall pattern. Accuracy decreases as the retrieval models’ performance drops from the least popular bucket to the fourth bucket. Interestingly, accuracy increases for the most popular bucket, indicating that LMs rely on their embedded information for popular entities.

## 5.2 Stimulus RAG performance
The second research question (R2Q) concerns whether our proposed Stimulus RAG method can surpass the performance of fine-tuned models. To evaluate the effect of the stimulus RAG, we first need to investigate how increasing the number of documents in the input affects the accuracy of LMs. Table 6 presents the results of RAG with top-1, top-3 and top-5 documents, shown as (ID), (3D), and (5D), respectively. It shows that using top-3 documents leads to notable accuracy improvements in all cases, both before and after FT. For DPR, these results align with those in Table 4, where there is a significant increase from Recall@1 to Recall@3. For ideal retriever, it is important to note that the Ideal retriever is not 100 percent accurate; for some queries, the answer is found in other paragraphs, not just in the summary paragraph. Another notable observation is that increasing the number of input documents to five results in either negligible accuracy improvement or a decrease in accuracy. This occurs despite Table 7.
```

### --- Page 0009 ---

```markdown
| Model     | -FT-RAG         | -FT-RAG         | SRAG            |
|-----------|------------------|------------------|------------------|
|           | (3D)             | (1D)             | (D)              |
|           | PopQA            |                  |                  |
|-----------|------------------|------------------|------------------|
| FlanT5-base | DPR 53.36  | 56.67  | 52.20  | 53.46  |
|           | Ideal 74.50      | 73.05  | 71.34  | 72.07  | 71.75 |
|           | 64.34  | 63.54  | 55.33  |          |
|           | 76.14  | 80.24  | 80.29  | 82.98  | 82.99 |
| Mistral   | DPR 60.09  | 62.25  | 62.13  | 63.93  |
|           | Ideal 80.25      | 81.84  | 78.44  | 80.34  | 80.34 |
|           | 61.46  | 66.64  | 67.40  | 66.43  | 66.43 |
| Llama3    | DPR 81.29  | 82.56  | 83.27  | 82.63  |
|-----------|------------------|------------------|------------------|
| FlanT5-base | DPR 49.54  | 52.94  | 52.03  | 48.02  |
|           | Ideal 53.92      | 58.18  | 57.63  | 57.54  |
|           | 54.51  | 63.14  | 57.15  | 61.94  | 64.76 |
|           | 54.61  | 69.16  | 69.26  | 67.64  | 54.52 |
| Mistral   | DPR 54.51  | 62.68  | 61.28  | 63.94  | 64.73 |
|           | Ideal 68.60      | 71.34  | 71.98  | 69.64  | 70.83 |
|           | 67.37  | 67.68  | 68.41  | 65.64  | 65.45 |
| Llama3    | DPR 68.67  | 71.50  | 72.46  | 71.81  |
|-----------|------------------|------------------|------------------|

![Detailed description of the chart](assets/page_0009_img_1.png)

In this paper, we aimed to determine the most suitable approach for customizing language models (LMs) for less-resourced domains. We examined the effectiveness of retrieval augmented generation (RAG) and fine-tuning (FT) methods, focusing on four key aspects: (i) fine-tuning methods, specifically full fine-tuning versus parameter-efficient fine-tuning (PEFT), (ii) data augmentation techniques, (iii) the type and size of LMs, including decoder-only versus encoder-decoder models ranging from 80 million to 11 billion parameters, and (iv) the performance of retrieval models. Our findings reveal several key points. First, PEFT enhances downstream task performance and preserves the reasoning abilities of LMs while incorporating new knowledge. Second, prompt-based QA generation exhibits superior performance in factual QA tasks. Third, a small fine-tuned LM with RAG can perform on par with or even surpass a larger LM model. Additionally, RAG’s performance improves with higher-performing retrievers. Notably, when comparing knowledge injection methods, RAG significantly outperforms FT. We addressed the cost of fine-tuning by developing Stimulus RAG (SRAG), a novel RAG approach that prompts an LM to generate correct responses based on hints provided in the prompt. This method eliminates the need for extensive fine-tuning, making it a cost-effective solution for enhancing LM performance in less-resourced domains.

## ACKNOWLEDGMENTS
This publication is part of the project LESSEN with project number NWA.1389.20.183 of the research program NWA ORC 2020/21 which is (partly) financed by the Dutch Research Council (NWO).
```

### --- Page 0010 ---

```markdown
# REFERENCES

[1] Zahra Abedinihashem and Mohammad Alimohammadi. 2022. Generate then Retrieve: Conversational Response Retrieval Using LLMs as Answer and Query Generators. arXiv preprint arXiv:2304.19302.

[2] Nabil Abdennour. 2015. Bloom’s taxonomy: a cognitive learning objectives. Journal of the Medical Library Association, JMLA 103, 3 (2015), 152.

[3] Sybil A. Briscoe, Andrei A. Filippov, Joe B. Dehghan, and Michael Collins. 2019. The 57th Conference of the Association for Computational Linguistics, ACL 2019. 4686–4713.

[4] Aidan Asher, Sewon Min, Zezong Zhang, and Danqi Chen. 2023. A Retrieval-based Language Models and Applications. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: Student Abstracts, ACL 2023. 41–46.

[5] Aidan Asher, Zeyu Liu, Yizhong Wang, Aviroop S. Saha, and Hannah Shridhar. 2021. Self-AGI: Learning to Retrieve, Generate, and Configure through Self-Reflection. In The Tenth International Conference on Learning Representations, ICLR (2021).

[6] Aidan Asher, Zezong Zhang, Danqi Chen, Piyush K. Volek, and Elizabeth H. H. Huang. 2021. Retrieval-based Language Models with Retrieval. CoRR (2020).

[7] Anis T. Alkhateeb and Evangelos Kanoulas. and Saurabh Verma. 2013. A First Look at Using Distributed Representations for Training SearchGPT vs. Human Experts. In Proceedings of the 33rd International Conference on Information Knowledge Management, CIKM (2013), 513–520.

[8] Aidan Asher, Mohammad Alimohammadi, and Saurabh Verma. 2023. Expand, Highlight, Generate: Richly-Driven Document Generation for Passage Ranking. In Proceedings of the 2023 Conference on Empirical Methods for Natural Language Processing, EMNLP (2023). 1007–1018.

[9] Jonathan M. Bock, Jonathan Yoo, Dakota Mahan, Duy Nguyen, Mahyar Zareh, Rishabh Adhithya, James Baicun, Ben Brooks, Nathan Cooper, Bita Merig Lee, Emad Mostaque, Michael Pineda, and Chris Roush. 2022. Stable LLM 1.2.6 Technical Report. CoRR abs/2202.17834 (2022).

[10] Aidan Asher, Danqi Chen, Piyush K. Volek, and Elizabeth H. H. Huang. 2021. Retrieval-based Language Models with Retrieval. CoRR (2020).

[11] Xin Chen, Li Luo, Xiying Chen, Leman Liu, Dongyan Zhou, and Rui Yan. 2023. Unify Your LLMs: The Retrieval-Augmented Generation with Self-Memory. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) (2023).

[12] Sangwoo Cho, Kaijang Song, Chen Li, Dong Yu, Hassan Firooz, and Fei Liu. 2020. Better Highlighting: Creating Sub-Sentence Summary Highlights in Pre-trained Language Models. In Empirical Methods in Natural Language Processing, EMNLP (2020), 6282–6300.

[13] Hyungwon Chung, Seung-hwan Noh, Jacob W. Nair, Nathan Basnor, Gurkan M. Akbas, Robert H. Parris, Hyunwoo Song, and Young Chung. 2022. FALM: Scaling Language-Modeling with Many-Label Datasets. In Proceedings of the 2023 Conference on Empirical Methods for Natural Language Processing, EMNLP (2023). 240–251.

[14] Hyungwon Chung, J. Lee, M. Song, Jeongure, Bertram Y. Kim, and Yoonho Fock. 2023. Xuefu Wang, Mustafa Dehghan, Siddharth Bhardwaj, Albert Wehling, Shanshan Zhang, Guozi Dai, Misung Yeom, Yifan Zhang, Abhankara S. Shukla, and Sangwan Nam. 2023. Aidan Asher, Y. Chen, and Y. Zhang. 2023. A Unified Approach to Entity Recognition. Journal of Machine Learning Research 25 (2023), 701–715.

[15] Florin Chocan, Giovanni Trappolini, Federico Siliano, Simone Flice, Cesare Campagnolo, Volek Marek, Nicola Tontodonati, and Fabio Piretti. 2022. The Power of Noise: Redefining Retrieval for RAG Systems. In Proceedings of the 2023 International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR (2023), 719–727.

[16] Nicole De Gua, Ozair Izaz, Sebastian Riedel, and Fabio Piretti. 2021. A Progressive Entity Retrieval. In 9th International Conference on Learning Representations, ICLR (2021).

[17] Nelson E. F. K. Liew, Lin, Iboh Ezeh, and Parinaz Mohajerani. 2023. The Visible Potential of Entity Retrieval. In Proceedings of the 2023 Conference on Empirical Methods for Natural Language Processing, EMNLP (2023). 241–251.

[18] Jimmy Lin, Rodrigo Pratomo Notogeno, and Andrew Yates. 2021. Pretrained Transformers for Text Ranking: BERT and Beyond. Morgan & Claypool Publishers, Hokkaido Univ. Eric K. Tan, Masahiro Miyoshi, Tohoku University, and Daisuke Hirano. 2022. New Shift Parameter-Effective Fine-Tuning is Better and Cheaper in the Context Learning. In Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems (NeurIPS) (2022).

[19] Nelson E. F. K. Liew, Lin, Iboh Ezeh, and Parinaz Mohajerani. 2023. The Visible Potential of Entity Retrieval. In Proceedings of the 2023 Conference on Empirical Methods for Natural Language Processing, EMNLP (2023). 241–251.

[20] Seiji Makino, Hayato Izumi, and Nika Zubarev. 2022. Retrieval Augmentation for a Deeper Dive into the Efficacy of Retrieval Augmentation in Language Models. In Proceedings of the 2023 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022. 5506–5521.
```

### --- Page 0011 ---

```markdown
# Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge

| Reference | Citation |
|-----------|----------|
| [46] | Alex Mallen, Akari Asai, Victor Zong, Rajarshi Das, Daniel Khashabi, and Hamed Hajishirzi. 2023. When Not to Trust Language Models: Investigating the Effects of Parametric and Non-Parametric Reminders. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), (2023). |
| [47] | Jowin N. Wei, Zhi Mike Lewis, Nathan G. Chen, Wen-Tsai Li, Hannah H. P. S. S. M. Liu, and Jiekun Zhang. 2023. Nonparametric-Masked Language Modeling. In Findings of the Association for Computational Linguistics: ACL, 2077–2118. |
| [48] | Marius Mosbach, Timo Finkel, Shauli Ravfogel, Dietrich Klakow, and Yani Elazar. 2023. Fast-Track Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation. In Findings of the Association for Computational Linguistics: ACL, (2023), 1234–1241. |
| [49] | Muhammad Nawaz, Asad Ullah Khan, Ghi Muhammad Saqib, Saeed Anwar, Muhammad Israr, Nick Manea, and Aijmal M. 2023. A Comprehensive Overview of Large Language Models. arXiv 2307.06363 (2023). |
| [50] | Odeh Ovidaru, Meenakshi Beri, Moksli Mishra, and Oren Elinas. 2023. Fine-Tuning vs. Retrieval-Augmented Knowledge Injection in LLMs. CoRR abs/2301.02139 (2023). |
| [51] | Stephen E. Robertson and Mere Walker. 1994. Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted Retrieval. In Proceedings of the 17th International ACM SIGIR Conference on Research and Development in Information Retrieval. 232–241. |
| [52] | Stephen E. Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. In Found. Trends Inf. Retr. 3, 4 (2009), 333–389. |
| [53] | Christopher S. Wallace, Zhaohui Zhang, Jingyuk Lee, and Dong Chen. 2023. Entity-Centric Question Challenge Datasets for Fine-Tuning Language Models. In Findings of the Association for Computational Linguistics: ACL, 2023, 1420–1428. |
| [54] | Kurt Shuster, Spencer Poff, Myra Chen, Dowrie Kella, and Jason Williams. 2023. Retrieval-Augmented Reinforcement Hallucination in Conversational Agents. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 3758–3768. |
| [55] | Heydar Soltani, Evangelos Kanoulas, and Fragkiskos Hatzivassiloglou. 2023. Data Augmentation for Conversational AI. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 1232–1238. |
| [56] | Heydar Soltani, Roxana Petcu, Evangelos Kanoulas, and Fragkiskos Hatzivassiloglou. 2024. A Survey on Recent Advances in Conversational Data Generation. arXiv preprint arXiv:2403.01308 (2024). |
| [57] | Kai Sun, Yifan Ehan Xu, Huamen Zha, Yue Liu, and Xin Lun Dong. 2024. Head-to-Tail: How Knowledgeable are Large Language Models (LLMs)? A.K.A. Will LLMs Replace Knowledge Graphs?. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL (2024), 311–325. |
| [58] | Nandan Thaker, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. REIR: A Heterogeneous Benchmark for Zero- and Few-Shot Retrieval-Based Models. In Proceedings of the 44th International Conference on Information Retrieval, NeurIPS (2021). |
| [59] | Fabio Trovato, Lucas Martin, Kevin Storey, Peter Albert, Ahmad Alhindi, Yassir Ben Dali, David Eidsvik, Cristian Contian-Ferreyra, Nino Guillen, Guillerme Cuthbert, Rachael Eidsvik, Jeremy W. Wentz, Hironobu Hoshino, Rui Hu, Hakan Inan, Marek Krajewski, Viktor Kerker, Alimah Khasib, Isabell Koumoun, Artem Konorev, Punit Singh Koori, Marie-Anne Laxhabet, Thabo Laviri Jeya, Lena Lisovskikh, Tingluo Ma, Xavkier Martindale, Lordi Meysian, Roshni Raghava, Jina Sehlke, Tana Silas, Eric Michael Smith, Ranjan Subramanian, Xiyong Ellen Tan, Bin Fang, Rosy Taylor, Aiming Puskar, Jin Yang, Puxun, Zhu Yang, Yuran, Yuchen Zhang, Angela Jan, Meike Kunda, Shuxian Wang, Aizhen Li, and Robert Sirois. 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv 2307.07258 (2023). |
| [60] | Levis Tzeng, Edward Buehler, Hannah Lambert, Nezareth Rajani, Kadir H. Yurtsever, Loush Belda, Sheng Huang, Leandro von Werra, Clementine Fourrier, Thomas W. Stokes, and Samy Bengio. 2023. Generative Retrieval and Thomas: Towards Efficient Retrieval and Distillation of LLMs. CoRR abs/2309.16944 (2023). |
| [61] | Ashish Dutta, Elvira A. Mavridis, and James Camacho-Collados. 2023. A Friendly Comparison of Fine-Tuning and Retrieval-Augmented Generation. In Findings of the Association for Computational Linguistics: ACL, 2023, 1420–1428. |
| [62] | Jae-Hoon Lee and Tatsuya Hasegawa. 2022. RAIL: Entity Linker Training on the Shoulders of Giants. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL (2022), 1–9. |
| [63] | Peiyuan Zhang, Guantong Zeng, Tianhao Wang, and Wei Liu. 2024. TinyLlama: An Open-Source Small Language Model. (2024). |
| [64] | Tanyan Zhang, Shisir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E. Gonzalez. 2024. RAFT: Adapting Language Model to Domain Specific RAG. (2024). |
```


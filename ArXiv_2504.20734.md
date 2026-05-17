# ArXiv 2504.20734

### --- Page 0001 ---

```markdown
# UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities

**Woongyeong Yeo\textsuperscript{1}**  **Kangsang Kim\textsuperscript{*}**  **Soyeong Jeong\textsuperscript{*}**  **Jinheon Baek\textsuperscript{1}**  **Sung Ju Hwang\textsuperscript{1,2}**  
KAIST\textsuperscript{1}  DeepAuto.ai\textsuperscript{2}  
{\{wgcyo, kksan07, starsuzi, jinheon.baek, sungju.hwang\}@kaist.ac.kr}  
https://universalrag.github.io

## Abstract

Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge source cannot address. To address this, we introduce UniversalRAG, designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single aggregated corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose modality-aware routing, which dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it, and further justifies its effectiveness with a theoretical analysis. Moreover, beyond modality, we organize modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 10 benchmarks of multiple modalities, showing its superiority over various modality-specific and unified baselines.

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable performance across various tasks, and have been widely adopted to assist users in everyday life (Anil et al., 2023; OpenAI, 2025). However, LLMs often generate factually incorrect or misleading information, especially on topics they were less or not exposed to during train-

*Equal contribution

![Conceptual illustration comparing existing RAG strategies with our proposed UniversalRAG.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
![t-SNE plot of the unified embedding space](assets/page_0002_img_1.png)

modality- and corpus-specific; therefore, they may be suboptimal to serve as a universal, one-for-all framework that can flexibly handle the wide range of queries, whose knowledge requirements vary. 

In this work, we present UniversalRAG, a novel RAG framework that brings together knowledge distributed across multiple modality-specific corpora, and leverages them to generate grounded responses to queries in a universal workflow. To operationalize this, one straightforward approach might be to aggregate all entries from the collected, heterogeneous knowledge corpora, and embed them into a unified space using a multimodal encoder (which is typically trained to align inputs from different modalities if they are semantically similar). However, despite such alignment efforts, we find that this strategy suffers from modality gaps (Zhang et al., 2025b; Bolya et al., 2025; Meng et al., 2025), the tendency that inputs are clustered based on their modality rather than their semantic meaning (visualized in Figures 2 and 7). As a result, retrieval becomes biased toward knowledge sources that share the same modality as the query, overlooking relevant content from other modalities.

To address this challenge, rather than forcing all modalities into a single embedding space, we take a different direction and introduce modality-aware routing. UniversalRAG predicts its modality requirements and routes retrieval to the corresponding modality-specific corpora (potentially multiple, when the query calls for cross-modal evidence), after which the retrieved knowledge is jointly used for grounding. Notably, this strategy not only sideskips modality gaps by avoiding every cross-modal comparison but also enables seamless integration of new modalities by extending the routing logic without modifying existing modality-specific retrievals.

Beyond modality, data granularity (i.e., the size or unit of each entry in the corpus) also affects retrieval precision and generation quality (Chen et al., 2024b; Zhong et al., 2025), since different queries benefit from different granularities even within the same modality: overly fine-grained entries can dilute context, while overly coarse ones may bundle unrelated information. For example, complex analytical questions may require full documents or videos, while simple factoid questions are better served with a single paragraph or short video clip.

To accommodate this, we further decompose each modality into multiple granularity levels, organizing them into distinct corpora: documents are additionally segmented into paragraphs and stored in a paragraph-level corpus, and similarly, full-length videos are divided into short clips and stored, while images are kept intact since they are inherently piecemeal. Overall, with these modality- and granularity-aware corpora (including paragraphs, documents, tables, images, clips, and videos) in place, as well as an additional no-retrieval option to efficiently handle straightforward queries (that require no external knowledge), our UniversalRAG dynamically routes each query to the most relevant knowledge sources, ultimately supporting the diverse information needs of real-world users.

We validate UniversalRAG on 10 datasets spanning diverse modalities and granularities, where it outperforms all baselines by large margins on average, confirming its effectiveness in handling diverse types of queries. Moreover, UniversalRAG improves efficiency via modality-aware retrieval and appropriate granularity selection, while maintaining robustness on out-of-distribution datasets.

## 2 Method

We begin by describing the preliminaries.

### 2.1 Preliminaries

Large Vision Language Models Let us first define LLMs, which take an input sequence of tokens $x = [x_1, x_2, \ldots, x_n]$ and generate an output sequence of tokens $y = [y_1, y_2, \ldots, y_m]$, as follows: $y = \text{LLM}(x)$, where $x$ and $y$ are represented in text. Building on top of LLMs, Large Vision-Language Models (LVLMs) extend their capability to support multimodal understanding by incorporating visual encoders (Bai et al., 2023; Chen et al., 2024c; Liu et al., 2024; Li et al., 2025a), to process both the textual and visual inputs. Formally, similar to LLMs, LVLMs can be functionalized as $y = \text{LVLM}(x)$, where input token $x_i$ is extended to either textual or visual. However, although they are extensively trained, LVLMs themselves are limited to their parametric knowledge, and often struggle with queries that require fine-grained or up-to-date information, less or not exposed for training.
```

### --- Page 0003 ---

```markdown
# Retrieval-Augmented Generation

To address the aforementioned limitations of using only the parametric knowledge, RAG has been widely used, whose core idea is to retrieve query-relevant information from a large corpus and incorporate it into the generation process. Formally, in response to a query $q$, a retrieval model $T$ fetches the relevant context from a corpus $C: c = T(q; c)$. Then, in the subsequent generation step, LVM generates a response $a$ conditioned on the query and retrieved context: $a = \text{LVM}(q, c)$. However, most existing RAG approaches are restricted to retrieving from a single corpus consisting of entries from a single modality (such as only the textual documents), limiting their ability to handle diverse queries with knowledge requirements that vary across them.

## 2.2 UniversalRAG

We introduce UniversalRAG that dynamically identifies and routes queries to the most appropriate modality and granularity for targeted retrieval.

### Challenges in Multi-Corpus Retrieval

To accommodate the diverse knowledge needs of real-world queries, which may involve heterogeneous sources spanning different modalities, we consider a set of modality-specific corpora, where each corpus $C_m$ contains items of modality $m$. Notably, one straightforward approach to operationalize this is to aggregate all corpora into a unified corpus $C_{\text{unified}} = \bigcup_{m \in M} C_m$ and embed all items into a shared space using a multimodal encoder, as for retrieval over a single corpus: $c = T(q; C_{\text{unified}})$. However, we find this approach suffers from modality gap (Figures 2 and 7), where queries, being textual, align more closely with elements in the text corpus regardless of the modality required. Therefore, instead of forcing all heterogeneous elements into a unified corpus, we propose selectively engaging the most relevant corpus needed for queries.

### Modality-Aware Retrieval

To sidestep the issue of modality gap (introduced by handling all modalities over the unified space), we instead propose to break down the overall retrieval process into two subsequent stages: (1) identifying the most relevant set of modalities for the query; and (2) performing targeted retrieval within the selected modality-specific corpus. Specifically, instead of aggregating all modality-specific corpora, we preserve each corpus in its original form with an independent embedding space. After that, to direct queries to their best-aligned knowledge sources, we introduce a routing module $R$ that dynamically predicts the modalities best suited for a query $q$, yielding $R(q) = M_q$ where $M_q$ is the set of modalities for $q$. Retrieval is then restricted to the corresponding corpora $\{C_m | m \in M_q\}$, using any off-the-shelf retriever $T_m$ tailored to each modality, thereby avoiding the modality gap issue present in a unified space. Proposition 1 formalizes the advantage of modality-aware routing over unified embeddings, and we provide its proof in Appendix C.

**Proposition 1.** Let the similarity score in a unified embedding space $C_{\text{unified}}$ be defined as

$$
s(q, c) = \alpha \cdot 1\{m(q) = m(c)\} + \beta \cdot r(q, c) + \epsilon,
$$

where $\alpha > 0$ induces modality bias and $r(\cdot, \cdot)$ measures relevance. If $\alpha$ dominates the variability of $r$, modality-aware routing retrieves items from the required modality $m^*(q)$ with higher probability than unified embedding retrieval.

However, while this routing principle mitigates the modality gap, organizing corpora solely by the modality might still be suboptimal since different queries require varying levels of granularity.

### Granularity-Aware Retrieval

To accommodate the varying complexity and information scope of different queries, we extend UniversalRAG to operate not only across modalities but also across different levels of granularity within each modality. To be specific, rather than treating each modality-specific corpus as a flat collection of items, we organize it into representations at multiple resolutions, enabling retrieval to target either fine-grained details or broader context as required by the query. To reflect this richer organization of corpora, the routing module $R$ expands its prediction space to include modality-granularity pairs best suited to a query, as well as a no-retrieval option for cases where external context is unnecessary: 

$$
R : Q \rightarrow \{ \emptyset \} \cup \left( \bigcup_{m \in M} G_m \right),
$$ 

where $M$ is the set of modalities and $G_m$ is the set of granularities available for modality $m$. Once the router predicts the relevant pairs, retrieval is performed over the corresponding corpora, using retrievers specialized for each modality to obtain the relevant context. Finally, the LVM generates the answer $a$ with $c$, customized to the modality and granularity for each individual query, thereby enabling the universal, one-for-all RAG framework.
```


### --- Page 0004 ---

```markdown
## 2.3 Router Implementation Strategies

A key component of UniversalRAG is the router, which is responsible for determining the optimal modality and granularity of knowledge for a query.

### Training-based Router

To perform the routing task, we first consider training the available models to predict the appropriate modality–granularity pair for each query. However, since ground-truth labels (for the modality and granularity the query should be routed to) are not available, we leverage inductive biases in existing benchmarks, mapping each dataset to routing targets that match its task characteristics, allowing us to automatically obtain a labeled corpus without manual annotation. We then train open-source LVLMs to serve as the router using a multi-hot label representation and cross-entropy loss. At inference time, the router produces a sigmoid distribution over modality-granularity pairs and returns all configurations whose scores exceed a predefined threshold, enabling cross-modal and multi-granularity retrieval when necessary.

### Training-free Router

Alternatively, we also explore a training-free approach that leverages the broad knowledge and robust reasoning capabilities of modern frontier models, such as Gemini (Anil et al., 2023). Instead of learning from labeled data, the model is directly prompted to act as a router. To achieve this, we first design the prompt template (used to elicit routing), which describes the objective and includes examples demonstrating how different types of queries correspond to specific retrieval targets (See Figure 8 for details). Then, at inference time, the model is prompted with this template to predict the most suitable modality-granularity pairs from a predefined set. This eliminates the need for supervised labels or task-specific training, offering the flexibility to adapt to new domains.

## 3 Experiment

### 3.1 Experimental Setup

We now explain the experimental setup, including datasets, models, and implementation details.

#### Datasets

To evaluate UniversalRAG, we compile a comprehensive benchmark covering RAG tasks across seven modalities and granularities. For the no-retrieval setting, we use MMLU (Hendrycks et al., 2021). For text-based RAG, we include Natural Questions (NQ) (Kwiatkowski et al., 2019) for single-hop, paragraph-level retrieval, and HotpotQA (Yang et al., 2018) for multi-hop, document-level retrieval. To consider diverse scenarios, we further incorporate HybridQA (Chen et al., 2020) for reasoning over text and tables, MRAG-Bench (MRAG) (Hu et al., 2025) for image RAG, and WebQA (Chang et al., 2022) and InfoSeek (Chen et al., 2023a) for cross-modal RAG over text and images. Lastly, for the RAG with videos, we use LVBench (Wang et al., 2025a) for queries over short or localized video segments, as well as VideoRAG-Wiki and VideoRAG-Synth (Jeong et al., 2025) for queries grounded on long-form or complete videos. Please refer to Appendix A for more details.

#### Knowledge Corpora

To support the aforementioned, diverse RAG scenarios with various modalities and granularities, we consider their corresponding corpora. Recall that we define seven routing pathways: None, Paragraph, Document, Table, Image, Clip, and Video, with cross-modal routing allowing queries to span multiple modalities. For the paragraph and document corpora, we use Wikipedia at the levels of paragraphs (Karpujhin et al., 2020) and documents (Jiang et al., 2024b). The table corpus is built by collecting tables from the HybridQA benchmark. For the image, we adopt corpora from MRAG-Bench, WebQA, and InfoSeek datasets. Lastly, we construct two video corpora at different scales: a video-level corpus consisting of full-length videos from LVBench and VideoRAG datasets, and a clip-level constructed by segmenting these videos into multiple short clips.

#### Methods

We compare our UniversalRAG to a diverse set of 12 baselines, grouped into four categories. The first is Naïve, which directly answers queries without retrieving external knowledge. In addition, the group of Unimodal RAGs includes ParagraphRAG, DocumentRAG, TableRAG, ImageRAG, ClipRAG, and VideoRAG methods, which retrieve information exclusively from their respective corpora and leverage it for response generation. The third group of Unified Embedding Multimodal RAGs uses multimodal encoders to align different modalities into a shared embedding space for retrieval, including UniRAG (Sharifmoghaddam et al., 2025), GME (Zhang et al., 2025b), PECore (Bolya et al., 2025), and VLMZwec-V2 (Meng et al., 2025). MultiRAG is included in the last group of Multi-corpus Multimodal RAGs, which performs retrieval over all the avail-
```

### --- Page 0005 ---

```markdown
| Models         | MMLU  | NQ   | HeteroQA | HybridQA | MIRAG | WebQ  | Infersent | LVBench | VideoRAG | VideoRAG-Synth |
|----------------|-------|------|----------|----------|-------|-------|-----------|---------|----------|-----------------|
|                | Acc   | EM   | F1       | EM       | F1    | R@1   | R@L      | Acc     | R@1      | R@L             |
| NaïveParagraphRAG | 74.39 | 38.56 | 21.30    | 25.91    | 52.11 | 92.78 | 18.70     | 28.97   | 86.74    | 35.59           |
| ImageRAG      | 73.71 | 38.15 | 21.34    | 25.30    | 51.75 | 91.95 | 19.15     | 17.64   | 85.61    | 35.36           |
| VideoRAG      | 73.11 | 37.69 | 21.45    | 25.16    | 51.68 | 87.67 | 19.67     | 15.67   | 86.36    | 34.76           |
| UniRAG        | 72.73 | 37.15 | 21.16    | 24.80    | 51.68 | 86.45 | 19.15     | 15.12   | 85.61    | 34.36           |
| T5GEMMA-2 270 | 71.71 | 36.50 | 20.80    | 24.30    | 50.80 | 85.67 | 18.50     | 14.80   | 84.91    | 33.96           |
| VideoRAG-Synth | 70.11 | 35.00 | 20.50    | 23.80    | 50.00 | 84.50 | 18.00     | 14.00   | 83.21    | 33.56           |
| PMF           | 71.00 | 36.00 | 20.80    | 24.00    | 50.50 | 85.00 | 18.50     | 14.50   | 84.00    | 33.00           |
| VLM-Ex-V2     | 71.78 | 39.25 | 21.80    | 25.50    | 52.00 | 87.00 | 19.80     | 16.00   | 87.38    | 34.91           |
| MultiRAG      | 71.00 | 36.00 | 20.80    | 24.00    | 50.50 | 85.00 | 18.50     | 14.50   | 84.00    | 33.00           |
| **UniversalRAG** |       |      |          |          |       |       |           |         |          |                 |
| **Final Result** | 74.39 | 38.56 | 21.60    | 34.61    | 11.68 | 63.25 | 70.82     | 19.36   | 32.77    | 28.85           |
| **Qwen3-VL-8B-Instruct** | 74.39 | 38.56 | 21.30    | 30.93    | 14.70 | 63.25 | 70.82     | 19.36   | 32.77    | 28.85           |
| T5GEMMA-2 270 | 74.82 | 38.54 | 25.00    | 30.00    | 15.00 | 63.00 | 69.00     | 21.05   | 39.81    | 29.91           |
| **Training-Free Results** |       |      |          |          |       |       |           |         |          |                 |
| Qwen3-VL-8B-Instruct | 74.27 | 36.40 | 24.15    | 21.32    | 4.95  | 57.91 | 63.85     | 21.84   | 32.61    | 36.94           |
| InternVL3.5-1B | 74.09 | 34.50 | 24.90    | 25.00    | 5.94  | 54.00 | 64.67     | 29.05   | 32.82    | 36.84           |
| Oracle        | 73.39 | 35.32 | 25.36    | 34.72    | 15.20 | 52.15 | 71.97     | 20.32   | 27.09    | 36.05           |

![Comparison of averaged evaluation results across different RAG methods and LVLMs.](assets/page_0005_img_1.png)

### 3.2 Experimental Results and Analyses
Now we present the overall results across diverse RAG scenarios, followed by a detailed analysis of the observed performance improvements.

#### Overall Results
We present the modality- and granularity-specific results in Table 1, along with the averaged results with different LVLMs in Figure 3, from which we observe that UniversalRAG consistently achieves the best performance on average. Specifically, in Table 1, the results corroborate our hypothesis that retrieving from the modality (or granularity) that aligns best with the information needs of the queries achieves the highest accuracy; however, mismatches between the query and retrieval source result in significant degradation, which supports our claim that considering diverse modalities in the universal workflow is necessary for realistic RAG. Also, the level of granularity within each modality affects performance, suggesting that fine-grained retrieval and generation are necessary. In addition to them, UniversalRAG significantly outperforms unified embedding multimodal RAG baselines, confirming the issue of the modality gap inherent within them (See Figures 2 and 7). Lastly, when compared with the MultiRAG.
```

### --- Page 0006 ---

```markdown
# Modality Selection Rate (%)

| 100 | 85 | None | Text | Image | Video |
|-----|----|------|------|-------|-------|
| 0   | 0  | 0    | 1    | 2     | 3     |
|     |     |      |      |       |       |
| **Oracle** | 23 | 0 | 24 | 23 | 0 |

Figure 4: Distribution of the retrieved data modalities.

---

# Performance Comparison of Uni-modal and Cross-modal Approaches Across Different Router Models

| Models       | Retrieval | EM   | R1   | R-BERT |
|--------------|-----------|------|------|--------|
| Qwen3-VL-2   | Uni-modal | 9.60 | 14.56| 67.93  | 95.58  |
|              | Cross-modal | 11.65| 16.23| 70.22  | 95.86  |
| InternV1.3.5 | Uni-modal | 9.65 | 13.86| 67.90  | 95.49  |
|              | Cross-modal | 10.25| 14.79| 69.14  | 95.72  |
| GPT-5        | Uni-modal | 4.75 | 8.57 | 60.54  | 94.04  |
|              | Cross-modal | 4.95 | 8.79 | 62.38  | 94.52  |

---

# Modality Accuracy

| Models       | Modality | Recall | R1  | R@5 |
|--------------|----------|--------|-----|-----|
| UniRAG       |          | 25.00  | 0.01| 0.03|
| GME          |          | 26.27  | 13.84| 17.29|
| PExC         |          | 25.00  | 0.67| 1.85|
| VLM2Vec-V2   |          | -25.00 | 2.39| 3.69|
| UniversalRAG (Qwen3-VL-2) | 95.28 | 21.36 | 39.48 |
| UniversalRAG (InternV1.3.5) | 92.39 | 19.61 | 31.82 | 39.20 |
| UniversalRAG (GPT-5) | 68.22 | 16.33 | 27.31 | 31.41 |

---

Figure 5: Retrieval latency per query across corpus size.

Figure 6: Router accuracy with varying the router model size.

---

## Effectiveness of Cross-Modal Retrieval

Many queries can be addressed by using a single, most prominent modality; certain tasks benefit from integrating evidence across multiple modalities. For instance, HybridQA requires reasoning that spans both structured tables and accompanying textual sources, while WebQA involves visually grounded questions that pair text with images. Table 2 shows that, compared to uni-modal retrieval, for which each query is routed to a single relevant source, cross-modal retrieval achieves consistently stronger performance. By enabling queries to be routed across multiple modalities, the cross-modal retrieval can leverage complementary evidence that would otherwise be missed by the uni-modal approach. These highlight the effectiveness of UniversalRAG’s flexible routing mechanism, which dynamically retrieves information from multiple sources rather than relying on a single modality.

## Effectiveness of Modality Routing

To investigate the effectiveness of our routing method, we compare the distribution of retrieved modalities for VLM2Vec-V2, GME, and UniversalRAG (with Qwen3-VL-2) in Figure 4. Using 200 sampled queries per benchmark and normalizing distributions, we find that VLM2Vec-V2 retrieves exclusively text, while GME similarly exhibits a strong bias toward text regardless of the query’s required modality, reflecting the modality gap inherent to unified embedding spaces. In contrast, UniversalRAG retrieves more evenly across modalities, indicating that the router effectively mitigates modality bias and adaptively selects appropriate knowledge sources. This leads to higher modality retrieval accuracy, and consequently, higher retrieval recall, as shown in Table 3. While GME achieves comparable recall on text and image corpora, its inability to accurately retrieve from the correct modality leads to lower recall on multimodal corpora that include videos. Yet, UniversalRAG consistently retrieves from the correct modality, enabling it to achieve higher recall than baselines across all scenarios.

## Effectiveness of Multigranularity

Given the observed benefits of corpus selection in Table 1, we investigate its impact beyond modality by comparing UniversalRAG at varying levels of granularity. Table 4 shows that incorporating granularity-aware corpus selection leads to consistent performance gains by avoiding the retrieval of context that is either insufficient (e.g., a short paragraph lacking key entities for multi-hop reasoning) or excessive (e.g., a full video when only a short clip is relevant).
```

### --- Page 0007 ---

```markdown
| Models         | #Gn | EM    | FI    | Acc   |
|----------------|-----|-------|-------|-------|
| GPT-3          | 1   | 23.30 | 31.38 | 31.92 |
| GPT-5          | 2   | 24.35 | 32.71 | 32.30 |
|                 | 3   | 24.20 | 32.64 | 32.43 |
|                 | 4   | 24.70 | 33.25 | 32.85 |
| Qwen3-VL-8B    | 1   | 23.85 | 32.54 | 31.53 |
|                 | 2   | 24.65 | 33.12 | 32.43 |
|                 | 3   | 24.70 | 33.23 | 32.82 |
|                 | 4   | 25.05 | 33.70 | 33.20 |

| Models         | In-Domain Router | Avg Score | Out-Domain Router | Avg Score |
|----------------|------------------|-----------|-------------------|-----------|
| Random         | -                | 14.29     | -                 | 37.85     |
| P-E*           | -                | 3.36      | -                 | 19.08     |
| VLMXDev-V2     | -                | 33.31     | -                 | 39.89     |
| UniversalRAG   | (qwen-v1.2)      | 95.61     | 42.60             | 71.93     |
| UniversalRAG   | (inter-v1.5-1B)  | 93.16     | 41.68             | 47.80     |
| UniversalRAG   | (GPT-5)          | 72.23     | 41.68             | 43.89     |
| Ensemble       | (Majority Voting) | 98.33     | 42.83             | 78.56     |

## Generalizability on Out-of-Domain Scenarios
As shown in Table 1, UniversalRAG with trained routers outperforms the training-free router (sometimes even approaching oracle performance), and a natural follow-up question is how these routers behave on unseen, out-of-domain (OOD) datasets. To investigate this, we evaluate on six OOD datasets (detailed in Appendix A.2), with results presented in Tables 5 and 10. In contrast to the in-domain setting, trained routers exhibit noticeable performance degradation, whereas the training-free router generalizes robustly and even surpasses the trained variants. Nevertheless, UniversalRAG remains effective in OOD scenarios and consistently outperforms all baselines, including those using the unified embedding space or random modality and granularity assignment, highlighting the benefit of adaptive, modality- and granularity-aware retrieval.

## Ensemble Strategy for Robust Routing
Building on the trade-off between the high in-domain accuracy of trained routers and the strong OOD generalization of training-free routers, we propose ensemble strategies that leverage their complementary strengths. Specifically, we explore confidence-based ensembling, which uses the trained router’s prediction when its confidence exceeds a threshold and otherwise falls back to the training-free router, as well as majority voting, which selects the majority prediction from three routers (training-based and free) with random tie-breaking. Table 5 shows that UniversalRAG with the ensemble routing achieves a robust balance between accuracy and generalization, making it well suited for real-world scenarios with unseen or shifting distributions.

## Case Study
We present a case study of UniversalRAG in Table 6. The query asks for the number of statues of people on the Michigan Soldiers and Sailors Monument. Both TextRAG and ImageRAG retrieve the relevant and correct evidence; however, each modality alone is insufficient to determine the...
```
![Detailed description of the chart](assets/page_0007_img_1.png)
![Detailed description of the chart](assets/page_0007_img_2.png)
```

### --- Page 0008 ---

```markdown
| Question                                                                 | Answer: Nine statues of people.                                   |
|--------------------------------------------------------------------------|------------------------------------------------------------------|
| TextRAG                                                                  | Retrieved: the next section which is surrounded by four male     |
|                                                                          | figures depicting the Navy, Infantry, Cavalry, and Artillery     |
|                                                                          | branches of the United States Army. Four female allegorical      |
|                                                                          | figures, resting on pedestals, are above the male statues and ...|
| Response: Eight people ✗                                                | ImageRAG                                                            |
|                                                                          | Retrieved: the next section which is surrounded by four male     |
|                                                                          | figures depicting the Navy, Infantry, Cavalry, and Artillery     |
|                                                                          | branches of the United States Army. Four female allegorical      |
|                                                                          | figures, resting on pedestals, are above the male statues and ...|
| Response: Six people ✗                                                  | VideoRAG                                                            |
|                                                                          | Retrieved:                                                          |
|                                                                          | full count. TextRAG lacks the information needed to aggregate    |
|                                                                          | all statues, while ImageRAG suffers from partial occlusion.      |
|                                                                          | VideoRAG fails to retrieve relevant evidence, as the video corpus |
|                                                                          | does not contain information useful for this query. In contrast,  |
|                                                                          | UniversalRAG routes the query to both the “Paragraph” and “Image”|
|                                                                          | corpora, allowing cross-modal reasoning and correctly identifying  |
|                                                                          | all nine statues. More case studies are provided in Appendix F.  |

## 4 Related Work

### Large Vision Language Models
Building on the impressive performance of LLMs (Anil et al., 2023; Hurst et al., 2024), recent studies have extended to visual domains. Liu et al. (2023) incorporates a CLIP-based (Radford et al., 2021) image encoder to align visual inputs with language representations, followed by models using diverse encoders (Bai et al., 2023; Chen et al., 2024c; Li et al., 2024) and extensions to video (Li et al., 2025a; Wang et al., 2025b; Bai et al., 2025). However, despite improved performance on multimodal benchmarks (Mathew et al., 2021; Yue et al., 2024; Li et al., 2024; Fu et al., 2025) from larger datasets and with improved architectures, LVLMs still often suffer from hallucinations (Huang et al., 2025) when relying solely on parametric knowledge.

### Retrieval-Augmented Generation
To address the aforementioned limitation of parametric-only models, RAG incorporates external knowledge during response generation. While conventional RAG focuses on the textual corpus (Lewis et al., 2020; Ram et al., 2023), recent work extends it to multimodal sources such as images and videos (Chen et al., 2022; Riedler and Langer, 2024; Jeong et al., 2025). However, these approaches assume a fixed single-modality retrieval, making them less adaptable to real-world queries that may require information from different modalities. Multimodal encoders (Radford et al., 2021; Zhang et al., 2025b; 

### Retrieval Granularity
While most of the existing RAG methods operate at fixed granularity (e.g., full documents, passages, or sentences), real-world queries often require information at varying levels of specificity depending on the knowledge needed, which in turn impacts performance and efficiency in both textual (Chen et al., 2024b; Liu et al., 2025b; Zhong et al., 2025) and video-based retrieval systems (Chen et al., 2023b). In contrast, UniversalRAG performs query-level routing across modality and granularity dimensions, enabling retrieval from the most relevant source at the appropriate level.

## 5 Conclusion
In this paper, we proposed UniversalRAG, a novel RAG framework designed to retrieve from corpora of diverse modalities and granularities. Through a modality- and granularity-aware routing mechanism, UniversalRAG dynamically selects the most suitable knowledge sources for each query, effectively addressing the limitations posed by modality gaps and fixed-granularity retrieval, which we further justify with theoretical results. Empirical evaluations across 10 benchmarks demonstrate that
```


### --- Page 0009 ---

```markdown
UniversalRAG outperforms both modality-specific and unified baselines, showcasing robust performance across diverse modalities. Also, our analyses highlight the importance of fine-grained retrieval and the complementary strengths of training-free and trained routers. We believe these findings demonstrate the potential of UniversalRAG as an adaptive solution for grounding LVLMs with heterogeneous external knowledge, paving the way for the one-for-all RAG that unifies the fragmented landscape of existing corpus-specific RAGs.

## Limitations

The proposed UniversalRAG is designed for leveraging heterogeneous, multimodal corpora at RAG, enabling corpus-aware routing to flexibly utilize modality- and granularity-specific corpora. It is worth noting that the routing mechanism is its central part, and to improve its accuracy, high-quality samples for training may be required; however, existing datasets or benchmarks lack ground-truth labels indicating ideal modality or granularity for each query. Nonetheless, we address this by automatically annotating queries (based on inductive biases inherent in datasets or downstream performance measured with all the available corpora), as detailed in Appendix A. However, since they may contain some noise, constructing high-quality, human-annotated routing datasets would be a valuable direction for future work. Also, due to similar reasons: the absence of annotated data (specifically, the query-granularity pairs), we segment each (text and video) modality into two levels of granularity to obtain supervision signals for router training. Again, collecting more fine-grained annotations that cover a wider range of query-modality and query-granularity pairs would be an exciting direction to expand the applicability of UniversalRAG.

## Ethical Considerations

The proposed UniversalRAG can be seamlessly integrated with any LVLMs and compatible retrieval corpora, reducing hallucination with the corpus-specific routing. However, there can be potential private, harmful, or biased content present in the retrieved or generated outputs, depending on the nature of the underlying corpora or the internalized knowledge within LVLMs. To mitigate such risks, it is recommended to apply safeguard mechanisms and filtering techniques in retrieval and generation, to ensure the safe and responsible deployment.

## References

Mohammad Mahdi Abootorab, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammadhakni, Bardia Mohammadi, Omid Ghahroodi, Mahdih Soleymaani Baghsas, and Ehsaneddin Asgari. 2025. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented generation. Preprint, arXiv:2502.08826.

Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiayi Yu, Raúl Scorcia, John Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Saur Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schritterwe, Amelia Glaese, Jilin Chen, Emily Pilter, Timothy P. Lili- crap, and 33 others. 2023. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Owen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. Preprint, arXiv:2308.12966.

Shuai Bai, Yuxuan Cai, Ruize Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen's-vl technical report. arXiv preprint arXiv:2511.21631.

Valeriya Bolotova-Baranova, Vladislav Blinov, Sofya Filippova, Falk Scholer, and Mark Sanderson. 2023. WikiHowQA: A comprehensive benchmark for multi-document non-factoid question answering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5291–5314, Toronto, Canada. Association for Computational Linguistics.

Daniel Bolya, Po-Ya Huang, Peite Su, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoosh Rashed, Junke Wang, Marco Monteiro, Hu Xiuyong Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph Feichtenhofer. 2025. Perception encoder: The best visual embeddings are not out of the network. Preprint, arXiv:2504.13181.

Brandon Castellano. 2014. PySceneDetect.

Yingshan Chang, Guihong Cao, Mridu Narang, Jianfeng Gao, Hisami Suzuki, and Yontan Bisk. 2022. Webqa: Multihop and multimodal QA. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 16474–16483. IEEE.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024a. Benchmarking large language models in retrieval-augmented generation. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial
```

### --- Page 0010 ---

```markdown
| **Author(s)** | **Title** | **Conference/Journal** | **Year** | **Pages** |
|---------------|-----------|------------------------|----------|-----------|
| Wangqing Cui, KePing Bi, Jiafeng Guo, and Xueqi Cheng | 2024. MORE: Multi-mOdal REtRieval augmented performance commonsense reasoning. | In Findings of the Association for Computational Linguistics: ACL 2024 | 2024 | pages 1178–1192 |
| Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Yong Du | 2024. Dense X retrieval: What retrieval granularity should we use? | In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024 | 2024 | pages 15159–15177 |
| Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, and Xueqi Cheng | 2025. Rowen: Adaptive retrieval-augmented generation for hallucination mitigation in lms. | In Proceedings of the 2025 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, SIGIR-AP 2025 | 2025 | pages 12–21 |
| Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W. Cohen | 2022. Mugan: Multimodal retrieval-augmented generator for open question answering over images and text. | In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022 | 2022 | pages 5558–5570 |
| Matthijs Douze, Alexandr Gubarev, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Marzé, Maria Lomeli, Lucas Hosseni, and Hervé Jégou | 2025. The fais library. | IEEE Transactions on Big Data | 2025 | pages 1–17 |
| Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shen, and 2 others | 2025. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal lms in video analysis. | In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) | 2025 | pages 24108–24118 |
| Yunfan Gao, Xin Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofeng Wang | 2024. Retrieval-augmented generation for large language models: A survey. | Preprint, arXiv:2312.10997 | 2024 | - |
| Ziriu Guo, Xubin Ren, Lingrui Xu, Jiahua Zhang, and Chao Huang | 2025. Rag-anything: All-in-one rag framework. | arXiv preprint arXiv:2510.12323 | 2025 | - |
| Dan Hendrycks, Collin Burns, Steven Basart, Andy Zau, Montas Maizeika, Dawn Song, and Jacob Steinhardt | 2021. Measuring massive multitask language understanding. | In 9th International Conference on Learning Representations, ICLR 2021 | 2021 | Virtual Event, May 3-7 |
| Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa | 2020. Constructing a multi-hop QA dataset for comprehensive evaluation of big QA datasets. | In Proceedings of the 28th International Conference on Computational Linguistics | 2020 | pages 6609–6625 |
| Wenbo Hu, Jia-Chen Gu, Zi-Yi Duo, Mohsen Fayyaz, Pan Lu, Kai-Wei Chang, and Nanyun (Violet) Pery | 2025. Mrag-bench: Vision-centric evaluation for retrieval-augmented multimodal models. | In International Conference on Representation Learning | 2025 | volume 2025, pages 95558–95581 |
```

### --- Page 0011 ---

```markdown
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans. Inf. Syst., 43(2).

Aaron Hurst, Adam Lerner, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and others. 2024. GPT-4o system card. arXiv preprint arXiv:2410.12726.

Shayekh Bin Islam, Md Asib Rahman, K. S. M. Toza-Mel Hossain, Enaam Houqe, Shafiq Joty, and Md. Rizwan Parvez. 2024. Open-graph: Enhanced retrieval augmented reasoning with open-source large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, pages 14231–14244. Association for Computational Linguistics.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. 2024. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), MACAL 2024, Mexico City, Mexico, June 16-21, 2024, pages 7036–7050. Association for Computational Linguistics.

Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju Hwang. 2025. VideoRAG: Retrieval-augmented generation over video corpus. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21278–21298, Vienna, Austria. Association for Computational Linguistics.

Xingyu Ji, Parker Glenn, Aditya G. Parameswaran, and Madelon Hulsbos. 2025. TARGET: Benchmarking table retrieval for generative tasks. Preprint, arXiv:2505.15145.

Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Daqing Wang, and Fuzhen Zhuang. 2024. E5-v: Universal embeddings with multimodal large language models. Preprint, arXiv:2407.12580.

Zhenghao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Diwedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 7969–7972. Association for Computational Linguistics.

Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024b. Longrach: Enhancing retrieval-augmented generation with long-context llms. Preprint, arXiv:2406.15139.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Vladimir Karpukhin, Barlas Oguz, Seow Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Ken Lee, Kristina Toutanova, Linh Doan, Matthew Keckley, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466.

Patrick Lewis, Ethan Perez, Aleksandr Piktus, Fabio Petroni, Vladimir Karpukhin, Namrata Goyal, Ehud Reiter, Mikey Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2025a. LLaVA-ovension: Easy visual task transfer. Transactions on Machine Learning Research.

Kunchang Li, Yali Wang, Yinan He, Yizhou Li, Yi Wang, Yi Liu, Zun Wang, Jilian Xu, Guo Chen, Ping Lou, Limin Wang, and Yu Qiao. 2024. Mvbench: A comprehensive multi-modal video understanding benchmark. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 22195–22206. IEEE.

Yangning Li, Yinghui Li, Xinyu Wang, Zhen Zhang, Xinran Zheng, Yong Jin, Hui Wang, Hai-Tao Zheng, Fei Huang, and Jingren Zhou. 2025b. Benchmarking multimodal retrieval augmented generation with dynamic VQA dataset and self-adaptive planning agent. In AAAI 2025 Workshop on Preventing and Detecting LLM Misalignment (PDLM).

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. 2025. MM-EMBED: Universal multimodal retrieval with multimodal LLMS. In The Thirteenth International
```

### --- Page 0012 ---

```markdown
# Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025.

| Author(s) | Title | Source |
|-----------|-------|--------|
| Stephanie Lin, Jacob Hilton, and Owain Evans. | TruthfulQA: Measuring how models mimic human falsehoods. | In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics. |
| Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. | 2024. Improved baselines with visual instruction tuning. | In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 26286–26296. IEEE. |
| Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. | 2023. Visual instruction tuning. | In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023. |
| Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, and Jun Ma. | 2025a. Hm-rag: Hierarchical multi-agent multimodal retrieval augmented generation. | In Proceedings of the 33rd ACM International Conference on Multimedia, MM ’25, page 2781–2790, New York, NY, USA. Association for Computing Machinery. |
| Zuhong Liu, Charles-Élie Simon, and Fabien Caspari. | 2025b. Passage segmentation of documents for extractive question answering. | In Advances in Information Retrieval, ECIR 2025, Lucca, Italy, April 6-10, 2025, Proceedings, Part III, page 345–352, Berlin, Heidelberg. Springer-Verlag. |
| Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Ceunca, Cyril Zakka, Louba Ben Allal, Anton Lozhkov, Nouamene Tazi, and others. | 2025. Smoll: Redefining small and efficient multimodal models. | arXiv preprint arXiv:2504.05299. |
| Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. | 2021. Docvqa: A dataset for VQA on documents images. | In IEEE Winter Conference on Applications of Computer Vision, WACV 2021, Waikoloa, HI, USA, January 3-8, 2021, pages 2199–2208. IEEE. |
| Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, and others. | 2025. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. | arXiv preprint arXiv:2507.04590. |
| Antoine Michin, Dmitrii Zikhov, Jean-Baptiste Alayrac, Markarak Tapaswi, Ivan Laptev, and Josef Sivic. | 2019. Howto100M: Learning a text-video embedding by watching hundreds of millions narrated video clips. | In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). |
| OpenAI. | 2025. Gpt-5 system card. |  |
| Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Grish Sash, Amankr Aksell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. | 2021. Learning transferable visual models from natural language supervision. | In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR. |
| Pranay Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. | 2016. SQuAD: 100,000+ questions for machine comprehension of text. | In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics. |
| Ori Ram, Yoav Levine, Itay Dalmedigo, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. | 2023. In-context retrieval-augmented language models. | Transactions of the Association for Computational Linguistics, 11:1316–1331. |
| Ruchit Rawal, Khalid Saifullah, Ronen Basri, David Jad, Gowthami Somepalli, and Tom Goldstein. | 2025. CiniPLe: A long video question answering dataset and benchmark. | In Workshop on Video-Language Models @ NeurIPS 2024. |
| Monica Riedel and Stefan Langer. | 2024. Beyond text: Optimizing req with multimodal inputs for industrial applications. | Preprint arXiv:2410.21943. |
| Sahel Sharifiyoghadam, Shivani Upadhyay, Wenhui Chen, and Jimmy Lin. | 2025. UniRAG: Universal retrieval augmentation for large vision language models. | In Findings of the Association for Computational Linguistics: ACL 2025, pages 2026–2039, Albuquerque, New Mexico. Association for Computational Linguistics. |
| Xiaoqiang Tang, Qiang Gao, Jian Li, Nan Du, Qi Li, and Shihong Xie. | 2025. MBA-RAG: a bandit approach for adaptive retrieval-augmented generation through question complexity. | In Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 3248–3254. Association for Computational Linguistics. |
| Weihan Wang, Zehai He, Wenyu Hong, Yean Cheng, Xihaoan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, and Jie Tang. | 2025a. Lybchenk: An extreme long video understanding benchmark. | In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22958–22967. |
| Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xinggeuang Wei, Zhaoying Liu, Linglin Jin, Shengyong Ye, Jie Shao, and others. | 2025b. termv3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. | arXiv preprint arXiv:2508.18265. |
```

### --- Page 0013 ---

```markdown
| Authors                                                                 | Title                                                                                                   |
|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Yin Wu, Quanyu Long, Jing Li, Jianfei Yu, and Wenya Wang. 2025.       | Visual-rag: Benchmarking text-to-image retrieval augmented generation for visual knowledge intensive queries. Preprint, arXiv:2502.16366. |
| Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024.            | Corrective retrieval augmented generation. Preprint, arXiv:2401.15884.                                 |
| Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,                   | HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics. |
| Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao, Linmei Hu, Liu Weichuan, Lei Hou, and Juanzi Li. 2025. | SeaKR: Self-aware knowledge retrieval for adaptive retrieval augmented generation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27022–27043, Vienna, Austria. Association for Computational Linguistics. |
| Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Rubin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhen Yang, Yibo Liu, Wenhao Huang, and others. 2024. | MMUU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 9556–9567. IEEE. |
| Biao Zhang, Paul Suganthan, Gaël Liu, Ilya Philippov, Sahil Dua, Ben Hora, Kat Black, Gus Martins, Omar Sansevierio, Shreya Pathak, and 1 others. 2025a. | T5gem 2: Seeing, reading, and understanding longer. arXiv preprint arXiv:2512.14856.                   |
| Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yavor Artzi. 2020. | BERTScore: Evaluating text generation with BERT. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. |
| Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. 2025b. | Bridging modalities: Improving multimodal retrieval by multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9274–9285. |
| Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, and 1 others. 2025c. | Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176. |
| Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Zeng Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. | Siren’s song in the ocean: A survey on hallucination in large language models. Preprint, arXiv:2309.01219. |
| Zihan Zhang, Meng Fang, and Ling Chen. 2024.                           | Retrievalal: Assessing adaptive retrieval-augmented generation for short-form open-domain question answering. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 6963–6975. Association for Computational Linguistics. |
| Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang, and Zengchang Qin. 2025. | Mix-of-granularity: Optimize the chunking granularity for retrieval-augmented generation. In Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 5756–5774. Association for Computational Linguistics. |
```

### --- Page 0014 ---

```markdown
# A Additional Details on Dataset

Table 7 provides an overview of all datasets and their corresponding knowledge corpora used in our experiments, including the target modality type as well as the size of the queries and corpora. We divide each dataset into a 3:7 ratio for training and testing. We offer the detail of each dataset below.

## A.1 In-Domain Dataset

**MMLU** As a dataset comprising queries that can be answered without the need for retrieval, we use MMLU (Hendrycks et al., 2021), a benchmark that spans a wide range of tasks, including problem-solving abilities (e.g., elementary mathematics, computer science) and world knowledge (e.g., law, world religions). Specifically, we use questions from all tasks in the development split.

**Natural Questions (NQ)** We also use Natural Questions (Kwiatkowski et al., 2019), a question answering dataset consisting of real user queries issued to the Google search engine, with answers annotated based on supporting Wikipedia articles. We randomly sample 2,000 QA pairs from the dev split, and formulate the text corpus by segmenting the Wikipedia corpus into paragraphs of at most 100 words.

**HotpotQA** HotpotQA (Yang et al., 2018) is a Wikipedia-based QA benchmark, but it contains complex queries that are annotated to reason over multiple articles. We utilize 2,000 randomly sampled QA pairs of the test split. As it requires multi-hop reasoning over multiple documents, we formulate the text corpus by grouping multiple related documents following LongRAG (Jiang et al., 2024b), which can be longer than 4K tokens.

**HybridQA** HybridQA (Chen et al., 2020) is a benchmark that requires reasoning over both tabular and textual information. Each question is grounded in a Wikipedia table, but often requires linking to associated text information to locate the correct answer. We randomly sample 2,000 QA pairs from the dev split. Unlike the original benchmark, which directly connects tables and textual evidence, we separate them into distinct table and text corpora to better validate our modality-specific routing-based retrieval framework.

**MRAG-Bench (MRAG)** We utilize MRAG-Bench (Hu et al., 2025), a vision-centric RAG benchmark that requires only relevant images and does not rely on other modalities, and evaluate on all 1,353 questions. Unlike conventional text-only queries, each query in MRAG-Bench is multimodal, consisting of a textual question interleaved with a query image. We construct a single image corpus by collecting all images across questions.

**WebQA** WebQA (Chang et al., 2022) is a benchmark designed to evaluate the ability of LVLMs to reason over multiple sources of information, including both text and images, in an open-domain setting. As the dataset is originally constructed with question-specific retrieval sources that combine text and images, we extract a subset of questions that require retrieval of image for answering. We then further filter these using GPT-4o (Hurst et al., 2024) with the prompt shown in Figure 10 to make sure questions are not grounded to a certain image, resulting in a final set of 2,000 QA pairs. Finally, we construct separate text and image corpora by extracting and aggregating evidence from each modality.

**InfoSeek** InfoSeek (Chen et al., 2023a) is an open-domain benchmark comprising questions interleaved with images, which are best answered by retrieving relevant textual and visual information. For our experiments, we sample 2,000 QA pairs from dev split and collect the text and image evidence associated with each question to construct the corresponding text and image corpora.

**LVBench** LVBench (Wang et al., 2025a) is a benchmark developed for long video understanding, featuring questions generated by annotators based on YouTube videos with an average duration of over one hour. Since the benchmark was originally designed for non-RAG tasks, we rephrase the original text-video interleaved queries into a text-only format to align with our experimental setup using GPT-4o, with video metadata and a prompt (Figure 11). Each query is associated with a specific video and a corresponding time range. Notably, the majority of queries are annotated with timestamps spanning less than five minutes, thereby focusing on short segments within the longer videos. Since some videos are currently unavailable, we conduct our evaluation on the available videos and their corresponding questions. For training, we use these short-timestamp queries as a clip-level dataset.

**VideoRAG** We also utilize VideoRAG-Wiki and VideoRAG-Synth benchmarks, introduced in Vide-
```

### --- Page 0015 ---

```markdown
| Dataset            | Query Modality | Target Retrieval Modality | # Queries | Corpus Size | Avg Length   |
|--------------------|----------------|---------------------------|-----------|-------------|--------------|
| **In-Domain Datasets** |                |                           |           |             |              |
| MMLU               | Text           | None                      | 1,710     | 850k        | 100 tokens   |
| Natural Questions   | Text           | Paragraph                 | 2,000     | 509k        | 693 tokens   |
| HotpotQA           | Text           | Document                  | 2,000     | 2,000       | 15k          |
| HybridQA           | Text           | Paragraph + Table         | 2,000     | 15k         | -            |
| MRQA-Bench         | Text + Image   | Image                     | 1,353     | 60k         | -            |
| WebQA              | Text           | Paragraph + Image         | 2,000     | 20k         | -            |
| InfoSeek           | Text + Image   | Paragraph + Image         | 2,000     | 20k         | -            |
| LVBench            | Text           | Clip/Video                | 777       | 89          | 3,865        |
| VideoRAG-Wiki      | Text           | Clip/Video                | 374       | 9k          | 3788         |
| VideoRAG-Synth     | Text           | Clip/Video                | 374       | 9k          | 378s         |
| **Out-of-Domain Datasets** |          |                           |           |             |              |
| TruthfulQA         | Text           | None                      | 790       | -           | -            |
| TriviaQA           | Text           | Paragraph                 | 661       | 661k        | 100 tokens   |
| SQuAD              | Text           | Paragraph                 | 2,000     | 1.19M       | 100 tokens   |
| 2WikiMultiHopQA    | Text           | Document                  | 2,000     | 12k         | 562 tokens   |
| Visual-RAG         | Text           | Image                     | 374       | 2k          | -            |
| CinePle            | Text           | Clip/Video                | 1,440     | 144         | 158s         |
```
![Dataset summary for in-domain and out-of-domain benchmarks](assets/page_0015_img_1.png)
```

### --- Page 0016 ---

```markdown
# CinePile

CinePile (Rawal et al., 2025) is a long-video question-answering benchmark that features questions based on movie clips from YouTube. Since the benchmark was originally designed for video understanding tasks rather than RAG, we reformulate each query using the same procedure as LVBench. For each of the 144 available videos, we randomly select 10 questions from the test split. Since CinePile does not provide granularity annotations, we classify the questions into two categories (such as clip-level and full-video-level granularity) using GPT-4, following the same approach used in VideoRAG.

## A.3 Evaluation Metrics

We report results with standard metrics. For datasets with multiple-choice questions, we report Top-1 Accuracy (Acc), the proportion of questions answered correctly. For short-answer datasets, we use Exact Match (EM) and F1, which respectively measure exact agreement and word-level overlay between predictions and references; for InfoSeek, we use the custom accuracy metric defined in the original paper and official repository. For datasets with longer free-form answers, we use ROUGE-L, which captures the longest common subsequences between the prediction and reference (Lin, 2004), and BERTScore, which assesses their semantic similarity (Zhang et al., 2020). We report the average score by averaging first within each modality, then across modalities. Results are obtained from a single run under limited computational resources, while we validate the generality of our framework across multiple backbone models.

## B Additional Implementation Details

To effectively leverage both visual and textual information for visual element retrieval, we employ an ensemble approach that combines visual and textual similarity scores with a weighting ratio of 0.8 for visual information. The textual information consists of image captions for images and scripts for videos. To handle long videos, we utilize PySceneDetect (Castellano, 2014), an open-source tool that detects scene boundaries by analyzing content changes (e.g., color histogram differences or threshold-based detection), to segment long videos into shorter clips with an average length of no more than 3 minutes. Moreover, for both the retrieval and generation stages, we uniformly sample 32 frames per video. For baseline models that do not natively support video input, specifically UniRad (which utilizes CLIP) and GME, we average the embeddings of these sampled frames to obtain a single representative embedding vector.

Training-based routers employ a lightweight classifier head on top of the backbone model to produce logits over multi-label prediction. Multi-label targets are converted into multi-hot vectors, and training is performed via binary cross-entropy loss between these targets and the predicted logits. The router is trained for 5 epochs with a learning rate of $2e-5$ and a LoRA rank of $r = 32$. At inference time, routing decisions are made using a predefined threshold of 0.8, selecting all modality-granularity combinations whose sigmoid probabilities exceed the threshold. In contrast, for the training-free variant, we prompt the model using a curated prompt that specifies task objectives and few-shot examples, as shown in Figure 8. Most experiments are conducted on NVIDIA RTX Pro 6000 Max-Q GPUs with 96GB of VRAM.

## C Theoretical Analyses of UniversalRAG

In this section, we present formal analyses of each module in UniversalRAG, including the effectiveness of modality routing (Section C.1) and multi-granularity (Section C.2), as well as the efficiency of modality-aware routing (Section C.3).

### C.1 Effectiveness of Modality Routing

For a rigorous analysis of the effectiveness of modality routing, we restate Proposition 1 with additional detail and provide a complete proof.

**Proposition 1 (Restated).** Let the similarity score in the unified embedding space of $U_{unified}$ be defined as

$$
s(q, c) = \alpha \cdot 1\{m(q) = m(c)\} + \beta \cdot r(q, c) + \epsilon,
$$

where $\alpha > 0$ is a modality bias, $m(\cdot)$ denotes the modality, and $r(\cdot, \cdot)$ measures semantic relevance. If $\alpha$ is sufficiently large relative to the variance of $r$, the probability of retrieving items from the required modality $m^*(q)$ is less than under modality-aware routing followed by within-modality retrieval.

**Proof.** Without loss of generality, let us consider the top-1 retrieval, as the extension to the top-k case follows directly. Let the unified retrieval corpus
```


### --- Page 0017 ---

```markdown
Cunified be decomposed into three disjoint sets:

| $S = \{c : m(c) = m(q)\}$ | $R = \{c : m(c) = m^*(q)\}$ | $O = Cunified \setminus (S \cup R).$  |
|----------------------------|-----------------------------|---------------------------------------|
| (1)                        |                             |                                       |

Let us consider the scenario where $m^*(q) \neq m(q)$ and $S, R \neq \emptyset$. Define $X_c = \beta \cdot r(q, c) + \epsilon_c$ and suppose $\{X_c\}_{c \in Cunified}$ are independent, mean-zero, sub-Gaussian with variance proxy $\sigma^2 = \beta^2 \cdot Var[r(q, c)] + Var[e_c]$. Then the similarity scores can be expressed as

$$
s(q, c) = 
\begin{cases} 
\alpha + X_c, & c \in S \\ 
X_c, & c \in R \cup O. 
\end{cases}
$$
(2)

Let $M_S = \max_{c \in S} X_s$, $M_R = \max_{c \in R} X_r$, and $M_O = \max_{c \in O} X_o$. Under the unified embedding retrieval, the top-1 item lies in $R$ if and only if 

$$
M_R \geq \alpha + \max\{M_S, M_O\}.
$$

Hence, we can obtain the upper bound of the probability where top-1 retrieval comes from $R$:

$$
P(Tunified(q; Cunified) \in R) = P(M_R \geq \alpha + \max\{M_S, M_O\}) \leq P(M_R - M_S \geq \alpha).
$$
(3)

As $\{M_R - M_S \geq \alpha\} \subset \{(r,s) \in R \times S | X_r - X_s \geq \alpha\}$, by the union bound we have

$$
P(M_R - M_S \geq \alpha) \leq \sum_{(r,s) \in R \times S} P(X_r - X_s \geq \alpha).
$$

As $X_r - X_s$ is sub-Gaussian with variance proxy $2\sigma^2$, the Chernoff bound of the tail probability combined with Equation (3) leads to

$$
P(Tunified(q; Cunified) \in R) \leq |R| |S| \exp\left(-\frac{\alpha^2}{4\sigma^2}\right).
$$
(4)

By contrast, if the retrieval is done at the modality-specific corpus after modality-aware routing with accuracy $r$, the probability where the top-1 item is in $R$ is 

$$
P(Tunified(q; Cunified) \in R) \leq |R| |S| \exp\left(-\frac{\alpha^2}{4\sigma^2}\right).
$$
(5)

C.2 Effectiveness of Multigranularity

In Section 3.2 and Appendix D.2, we show that routing with multiple granularities within each modality improves performance (see Tables 4 and 9). We also provide a simple statement and proof that support these empirical findings.

**Proposition 2.** Let $F(Q; m, g)$ be the expected response quality when retrieving from modality $m$ using granularity $g$. If there exist queries $q_1, q_2$ and granularities $g_1, g_2$ such that $F(q_1; m, g_1) > F(q_1; m, g_c)$ and $F(q_2; m, g_c) > F(q_2; m, g_1)$, then a routing policy that assigns $g_1$ to $q_1$ and $g_c$ to $q_2$ attains strictly higher expected quality than any fixed-granularity policy.

**Proof.** Consider any fixed policy that always uses a single granularity $g \in \{g_f, g_c\}$. If $g = g_f$, then we have

$$
F(q_1; m, g_f) < F(q_1; m, g_f) < F(q_2; m, g_f).
$$
(6)

Similarly, if $g = g_c$, then we have

$$
F(q_1; m, g_c) < F(q_2; m, g_c).
$$
(7)

In both cases, the sum of response quality with the routing policy that applies $g_f$ to $q_1$ and $g_c$ to $q_2$ strictly exceeds that of any fixed granularity policy.

C.3 Efficiency of Modality-Specific Retrieval

While the empirical results in Section 3.2 demonstrate the efficiency benefits of modality-aware routing (with latency trends shown in Figure 5),
```

### --- Page 0018 ---

```markdown
we provide a more rigorous analysis on its computational advantages. Let $N$ denote the size of each modality- and granularity-specific corpus, assuming uniform corpus sizes for simplicity, and let $k$ be the number of available routing choices (i.e., the number of modality-granularity pairs). Under a unified embedding approach, retrieval is performed over a single aggregated corpus of size $kN$, incurring a search cost that scales with the total corpus size. In contrast, UniversalRAG first performs lightweight routing to select the most relevant modality-granularity subset, and then conducts retrieval over only a small selected subset.

**Proposition 3.** Let $T(m)$ denote the expected retrieval latency of a single query over a corpus of size $m$ under a fixed retrieval backend, and let the routing cost be a fixed constant $C$, independent of the number of available routing choices $k > 1$. Then, UniversalRAG achieves lower latency than unified embedding space retrieval on large-scale corpora.

**Proof.** Under unified embedding, all modality-granularity corpora are merged into a single index of size $kN$. Then, the expected per-query retrieval latency is $T_{\text{unified}} = T(kN)$. Under UniversalRAG, routing incurs a constant overhead $C$ and then retrieval is executed only on a small number of routed corpora. Assuming retrieval calls of selected corpus are executed in parallel, the end-to-end latency of whole retrieval process is $T_{\text{routing}} = C + T(N)$. Let us first consider the case of exact retrieval with embeddings, where the backend exhibits linear scaling $T(m) = \Theta(m)$, then we obtain

$$
T_{\text{unified}} \approx \frac{kN}{N + C} = \frac{k}{1 + C/N}. \tag{8}
$$

Taking $N \to \infty$ yields

$$
\liminf_{N \to \infty} T_{\text{unified}} = \Theta(k), \tag{9}
$$

resulting in a linear-in-$k$ speedup. Meanwhile, many modern retrieval systems adopt approximate nearest neighbor search (Douze et al., 2025), which can achieve logarithmic query-time scaling $T(m) = \Theta(\log m)$ (in the best case). Then, for sufficiently large $N$,

$$
T_{\text{unified}} \approx \frac{\log(kN)}{\log N + C} = \frac{\log N + \log k}{\log N + C}. \tag{10}
$$

Letting $N \to \infty$, we have

$$
\liminf_{N \to \infty} T_{\text{unified}} \geq 1. \tag{11}
$$

Thus, even with the approximate retrieval with logarithmic scaling, UniversalRAG achieves a constant factor asymptotic speedup. Combining these results, UniversalRAG attains strictly lower asymptotic retrieval latency than unified embedding space retrieval for any retrieval methods.

### D Additional Experimental Results

#### D.1 Additional Results using Different LVMLs

Table 8 shows detailed generation results of baselines and UniversalRAG models on 10 benchmarks using InternVL3.5-8B and Molmo2-4B as generation models. In both settings, UniversalRAG outperforms all baselines and achieves average scores comparable to Oracle. These results demonstrate that UniversalRAG is robust and generalizable in various LVML generations.

#### D.2 Additional Results on Multigranularity

Table 4 demonstrates the correlation between the number of granularity levels and end-to-end performance for two training-free models, leveraging the flexibility of our approach in scenarios without labeled data. We further extend this analysis to training-based routers, comparing performance with and without granularity. Table 9 reports results across three training-based router models, consistently demonstrating a performance advantage when granularity is incorporated. These findings underscore the efficacy of including granularity in routing decisions for both training-free and training-based approaches.

#### D.3 Detailed Results on Out-of-Domain Dataset

We provide the generation results of UniversalRAG variants and baseline methods on each out-of-domain dataset in Table 10. Overall, UniversalRAG consistently outperforms all baselines on average. Notably, the training-free router variants exhibit strong performance across all datasets, showing their outstanding generalizability to unseen queries. In contrast, trained routers achieve relatively lower performance than on in-domain datasets; nevertheless, they remain robust and still surpass the baseline methods by a large margin.
```


### --- Page 0019 ---

```markdown
| Models               | MMLU  | HotpotQA | HybridQA | MRQA  | WebQ  | InfaSecK | LVBench | VideoRAG-Wiki | VideoRAG-Synth | Avg   |
|----------------------|-------|----------|----------|-------|-------|----------|---------|----------------|----------------|-------|
|                      | Acc   | EM   | F1   | EM   | F1   | Acc      | Acc     | Acc            | Acc            |       |
| Naive                | 71.85 | 17.05 | 24.50 | 31.07 | 10.25 | 14.52    | 47.79   | 61.34          | 15.95          | 36.84 |
| PaperRAG             | 63.66 | 16.40 | 19.40 | 2.75  | 5.50  | 25.50    | 54.56   | 31.25          | 22.67          | 34.41 |
| DocumentRAG          | 60.30 | 8.90  | 8.65  | 14.30 | 3.75  | 7.37     | 36.96   | 19.15          | 19.07          | 30.57 |
| TableRAG             | 72.72 | 16.65 | 19.15 | 18.67 | 11.25 | 14.50    | 39.89   | 11.50          | 21.87          | 30.77 |
| ClipRAG              | 69.93 | 19.25 | 15.80 | 12.63 | 19.85 | 14.92    | 14.80   | 6.00           | 15.37          | 27.80 |
| VideoRAG             | 70.09 | 11.05 | 15.08 | 14.90 | 1.97  | 12.98    | 48.77   | 5.35           | 21.87          | 26.34 |
| UniRAG               | 69.64 | 15.30 | 21.50 | 23.36 | 3.87  | 10.50    | 37.87   | 18.53          | 25.99          | 25.60 |
| GME                  | 69.15 | 14.80 | 17.15 | 21.30 | 6.93  | 14.32    | 34.12   | 17.83          | 12.73          | 25.11 |
| PMEM                 | 67.45 | 14.30 | 17.50 | 23.75 | 6.65  | 9.74     | 37.92   | 10.24          | 24.47          | 25.16 |
| UniversalRAG (Ours) | 68.54 | 15.80 | 21.50 | 21.50 | 7.67  | 12.34    | 39.12   | 18.73          | 15.45          | 25.83 |

| Models               | HotpotQA | LVBench | Acc   |
|----------------------|----------|---------|-------|
|                      | Gn       | EM   | FI   | Acc   |
| Naive                |          | 22.25 | 30.38 | 32.05 |
| PaperRAG             |          | 26.10 | 34.61 | 33.72 |
| DocumentRAG          |          | 23.00 | 30.89 | 32.05 |
| ClipRAG              |          | 25.85 | 34.29 | 33.72 |
| VideoRAG             |          | 22.55 | 30.61 | 31.40 |
| T5Gemma 2 270M      |          | 25.90 | 33.94 | 33.59 |

## E Modality Gap in Unified Embedding Space

Figure 7 visualizes the modality gap within the unified embedding space of five multimodal encoders (Jiang et al., 2022a; Bolya et al., 2025; Lin et al., 2025; Zhang et al., 2025b; Meng et al., 2022). The PCA plot reveals that embeddings cluster by modality, with text embeddings (shown in green) exhibiting larger distances from those of other modalities. Recent methods like E5-V and GME focus on better aligning these modalities to narrow the gap. However, despite these efforts, a noticeable separation between modalities remains, indicating that current multimodal encoders still struggle to fully unify the embedding space across text, images, and videos. Therefore, the modality routing mechanism of UniversalRAG is required to dynamically direct each query to its corresponding modality-specific embedding space, thereby effectively bridging the modality gap and enhancing retrieval performance.

## F Qualitative Results

We present case studies to demonstrate the effectiveness of UniversalRAG. Table 11 compares the results of various RAG approaches, including traditional single-modality methods and UniversalRAG, on queries from the WebQA dataset. Traditional approaches such as TextRAG and VideoRAG fail to generate accurate answers: TextRAG retrieves passages lacking relevant visual details, while VideoRAG is better suited for temporal re...
```

### --- Page 0020 ---

```markdown
| Models            | TruthQA | TriviaQA | SQuAD | 2WikiMultiHop | VisualQA | Cineple  | Acc  | Avg  |
|-------------------|---------|----------|-------|---------------|----------|----------|------|------|
| Naïve             | 70.00   | 53.25    | 61.61 | 16.75         | 32.62    | 37.60    | 82.78| 30.76|
| ParagraphRAG      | 68.86   | 55.82    | 67.38 | 44.47         | 41.50    | 50.86    | 89.58| 50.12|
| DocumentRAG       | 68.10   | 52.95    | 61.38 | 18.10         | 27.40    | 44.50    | 58.19| 8.86 |
| ImageRAG          | 68.48   | 51.39    | 97.95 | 16.12         | 34.01    | 44.01    | 80.32| 29.72|
| ClipRAG           | 69.11   | 51.59    | 14.25 | 23.07         | 43.70    | 41.53    | 10.28| 35.97|
| VideoRAG          | 68.76   | 51.44    | 59.16 | 14.20         | 37.70    | 48.90    | 12.91| 37.56|
| UniRAG            | 68.73   | 52.04    | 59.18 | 14.30         | 22.93    | 47.17    | 91.94| 18.19|
| GME               | 67.97   | 61.74    | 14.90 | 25.95         | 39.40    | 48.09    | 8.65 | 50.23|
| PLEu              | 68.61   | 52.80    | 61.11 | 14.50         | 23.85    | 38.10    | 47.02| 84.88|
| VLM2Vec-V2        | 68.10   | 51.89    | 13.85 | 22.66         | 38.85    | 47.95    | 8.70 | 28.59|
| MultiRAG          | 69.49   | 51.29    | 59.36 | 16.22         | 27.32    | 37.22    | 8.42 | 30.19|
| **UniversalRAG (Ours)** | | | | | | | | |
| Trained Routers   |         |          |       |               |          |          |      |      |
| (Open-V1.2B-Inst) | 69.75   | 54.16    | 23.61 | 41.60         | 45.20    | 54.33    | 10.62| 82.64|
| (IntentV1.3-B)    | 69.87   | 54.46    | 62.35 | 40.97         | 48.55    | 53.89    | 10.88| 72.64|
| (5GCamera 270M)   | 69.24   | 53.71    | 20.91 | 30.60         | 40.84    | 47.70    | 10.52| 82.58|
| **Training-free Routers** | | | | | | | | |
| GIFT-5            | 69.62   | 54.46    | 62.58 | 31.85         | 42.08    | 45.85    | 54.67| 11.27|
| (Open-V1.2L-8B-Inst) | 69.87   | 54.21    | 62.35 | 31.70         | 41.56    | 54.55    | 11.33| 33.82|
| Oracle            | 70.00   | 55.82    | 67.38 | 34.40         | 44.27    | 48.40    | 58.19| 11.64|
```
![Results of diverse RAG methods on out-of-domain dataset with QwenV1.2L-8B-Inst across modalities.](assets/page_0020_img_1.png)

### --- Page 0021 ---

```markdown
![Visualization of the unified embedding space across various multimodal encoders.](assets/page_0021_img_1.png)

a query requiring temporally localized visual evidence is incorrectly routed from clip-level retrieval to static image retrieval. Moreover, the router sometimes exhibits difficulty in determining the appropriate retrieval granularity. Queries that lie near the boundary between different granularity levels are sometimes misrouted, as shown in the third and fourth examples. Finally, the router occasionally fails to recognize cross-modal information needs, leading to incorrect routing decisions for queries that require joint reasoning across modalities, as illustrated in the last example.
```

### --- Page 0022 ---

```markdown
# Table 11: Case study comparing RAG on single modality against UniversalRAG (Ours). Traditional RAG models are limited by modality-specific retrieval, often missing relevant knowledge from other modalities. UniversalRAG (Ours) dynamically routes each query to the most appropriate modality, enabling more accurate retrieval and response generation.

## Comparison of UniversalRAG to Single-modality RAG

| Question | What color balloons were displayed on the USNS Carl Brashear during its ship launching ceremony in San Diego? |
|----------|----------------------------------------------------------------------------------------------------------------|
| Ground Truth | The balloons were red, white, and blue that were displayed on the USNS Carl Brashear during the ship launching ceremony in San Diego. |
| No Retrieval | The USNS Carl Brashear was adorned with blue and gold balloons during its ship launching ceremony in San Diego, reflecting the colors of the U.S. Navy and honoring the legacy of Carl Brashear. X |

| Method | Retrieval Status | Response |
|--------|------------------|----------|
| TextRAG | Retrieved | Operation Tomodachi, "Carl Brashear" loaded more than 800 pallets of humanitarian cargo at Sasebo’s Juliet pier on March 20 and set sail later that day to join the Navy ships operating off northern Japan. "Brashear" completed 17 underway replenishment missions, delivering more than 1 million gallons (3,800 m³) of fuel to ... The documents provided do not mention the color of balloons displayed during the ship launching ceremony of the USNS Carl Brashear in San Diego. X |
| TableRAG | Retrieved | **Inmate Name** | **Register Number** | **Status** | **Details** |
|          | Qian Xuesen | Unlisted | Held at Terminal Island ... | Chinese-born rocket scientist ... |
|          | Liz Reavy | Unlisted | Held at FCI Terminal Island ... | Girlfriend of Los Angeles mob kingpin Mickey Cohen ... |
|          |          |          |          | The color of balloons displayed on the USNS Carl Brashear during its ship launching ceremony in San Diego is not mentioned in the provided documents. X |
| VideoRAG | Retrieved | ![Video stills related to the USNS Carl Brashear](assets/page_0022_img_1.png) The video does not provide any information about the USNS Carl Brashear or its ship launching ceremony in San Diego, so it is not possible to determine the color of balloons displayed during that event based on the given content. X |
| UniversalRAG (Ours) | Routed to: Image | Retrieved: ![Image of the USNS Carl Brashear during the ceremony](assets/page_0022_img_2.png) During its ship launching ceremony in San Diego, the USNS Carl Brashear was adorned with red, white, and blue balloons. ✓ |
```

### --- Page 0023 ---

```markdown
# Table 12: Case study comparing different levels of text granularity. The user query requires complex retrieval involving multiple entities. ParagraphRAG retrieves limited context centered around a single entity, leading to an incorrect answer. UniversalRAG (Ours) routes the query to the document corpus and retrieves richer document-level information, allowing it to capture both relevant entities and generate the correct response.

| Comparison of Using Different Text Granularity |  |
|------------------------------------------------|--|
| **Question**                                    | George Reed was, along with which Drake University player, one of the greatest running backs in CFL history? |
| **Ground Truth**                               | Johnny D. Bright |
| **ParagraphRAG**                               | Retrieved: Matthew Reed (born November 30, 1951) is an American former gridiron football player. He played professionally as a quarterback in the World Football League (WFL) and the Canadian Football League (CFL). A graduate of Grambling State University, Reed was a tenth-round selection (240th overall pick) of the Buffalo Bills in the 1973 NFL Draft but did not play in the league. Reed began his pro career with the Birmingham Americans of the WFL. In 1974, he backed up George Mira, completing 77 of 188 passes (41%) for 1345 yards and 11 touchdowns and 12 interceptions. He was part of their championship team. <br> **Response**: The question contains an error: Matthew Reed is a quarterback, not a running back, and there is no mention of him being associated with Drake University. Additionally, George Reed (a famous CFL running back) is not mentioned in the provided text as being paired with a Drake University player. The text does not contain enough information to answer the question accurately. ✗ |
| **UniversalRAG (Ours)**                        | Routed to: Document <br> Retrieved: George Reed (Canadian football) George Robert Reed, CM, SOM (born October 2, 1939), is a former American college football and Canadian Football League player. Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history. In November 2006, Reed was voted one of the CFL's Top 50 players (#2) of the league's modern era by Canadian sports network. <br> ... <br> Johnny Bright Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League. He played college football at Drake University. He is a member of the Canadian Football Hall of Fame, the National Football Foundation’s College Football Hall of Fame, the Missouri Valley Conference Hall of Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame. <br> **Response**: Johnny Bright ✓ |
```

### --- Page 0024 ---

```markdown
| **Comparison of Using Different Video Granularity** |
|-----------------------------------------------------|
| **Question**                                        |
| What does the protagonist observe through the window after being taken to the utility room in the full episode of Blue Eye Samurai on Netflix? |
| (A) A group of monks sitting cross-legged in the snow |
| (B) A group of citizens chatting together           |
| (C) A group of warriors practicing swords           |
| (D) A group of samurais eating                      |
| **Ground Truth**                                   |
| C                                                   |

| **VideoRAG**                                       |
| **Retrieved:**                                     |
| ![VideoRAG Retrieved Frames](assets/page_0024_img_1.png) |
| (Timestamp Range: 00:00--1:01:05)                  |
| **Response:** A X                                  |

| **UniversalRAG (Ours)**                            |
| **Routed to:** Clip                                |
| **Retrieved:**                                     |
| ![UniversalRAG Retrieved Frames](assets/page_0024_img_2.png) |
| (Timestamp Range: 33:46--36:56)                    |
| **Response:** C ✓                                  |
```

### --- Page 0025 ---

```markdown
| Table 14: Case study comparing UniversalRAG across uni-modal and cross-modal scenarios. In the uni-modal setup, where only a single prominent modality is used, information can sometimes be incomplete as they require evidence across modalities. UniversalRAG (Ours), with its cross-modal capability, gathers evidence from multiple modalities to generate a more comprehensive response. |

| Comparison of UniversalRAG across Uni- and Cross-Modal Retrieval |
| --- |
| **Question** | What year did an artist known by the mid-1960s in soul and jazz circles for his recording skills release an American record company and label founded by brothers Max and Sol Weiss in 1949? |
| **Ground Truth** | 2000 |
| **UniversalRAG (Uni-modal)** | Routed to: Paragraph |
| **Retrieved:** | David Axelrod (April 17, 1931 [nb 1] - February 5, 2017) was an American composer, arranger, and producer. After starting out as a staff producer for record companies specializing in jazz, Axelrod became known by the mid-1960s in soul and jazz circles for his recording skills. In 1968, Axelrod embarked on a solo career and released several eccentric albums during the 1970s that showcased his characteristic sound, which combined heavily microphoned drums and baroque orchestration, and avant-garde themes ranging from the environment to heightened mental awareness. With his early solo projects, Axelrod was one of the first recording artists to fuse elements of jazz, rock, and R & B. One of his most important records, Song of Innocence (1968), featured instrumental interpretations of 18th-century poet William Blake's poetry collection of the same name done in a contemporary musical vein, leading one critic at the time to coin the term jazz fusion and numerous hip hop producers to sample the album's music decades later. |
| **Response:** | 1960 X |
| **UniversalRAG (Ours)** | Routed to: Paragraph+Table |
| **Retrieved:** | (Above Paragraph with the following table) |
| | Year | Album | Artist | Genre | Label | Credit |
| | --- | --- | --- | --- | --- | --- |
| | 1998 | Greatest Hits | Joe Cocker | Rock | EMI Histories | Trombones You Can Leave Your Hat On |
| | 2000 | The Axelrod Collection | David Axelrod | Jazz, Rock | Hip-O, A & M | Here as You Can Leave Your Hat On. |
| | 2004 | Ultimate Collection | Joe Cocker | Rock | ... | ... |
| **Response:** | 2000 ✓ |

| Table 15: Failure cases in modality routing with UniversalRAG (Ours). |

| **Question** | **Ground Truth** | **UniversalRAG (Ours)** |
| --- | --- | --- |
| What language does the French word polytechnique come from? | Paragraph | No |
| Who is seated to the right of Kobe in the Jimmy Kimmel tribute show? | Clip | Image |
| Which book by William A. Dembski summarizes the concepts he introduced about intelligent design in another of his works? | Document | Paragraph |
| What is the main cause of Lee Chong Wei losing points in the first half of his semi-final match against Lin Dan in the Rio 2016 Olympics replay? | Video | Clip |
| What is at the top of Hanbit Tower at Expo Science Park? | Paragraph+Image | Paragraph |
```

### --- Page 0026 ---

```markdown
Classify the following query into one or more categories from: [No, Paragraph, Document, Table, Image, Clip, Video], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:

- **No**: The query can be answered directly with common knowledge, reasoning, or computation without external data.
- **Paragraph**: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- **Document**: The query requires multi-hop reasoning, combining information from multiple sources or documents to form a complete answer.
- **Table**: The query requires information that is best represented in a tabular format, often involving comparisons or structured data.
- **Image**: The query focuses on visual aspects like appearances, structures, or spatial relationships.
- **Clip**: The query targets a short, specific moment or event within a video, without needing full context.
- **Video**: The query requires understanding dynamic events, motion, or sequences over time in a video.

### Examples:
- "What is the capital of France?" → No
- "What is the birth date of Alan Turing?" → Paragraph
- "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?" → Document
- "Among the recipients of the Turing Award, who had the earliest birth year?" → Table
- "Describe the appearance of a blue whale." → Image
- "Describe the moment Messi scored his goal in the 2022 World Cup final." → Clip
- "Explain how Messi scored his goal in the 2022 World Cup final." → Video
- "Solve 12 x 8." → No
- "Who played a key role in the development of the iPhone?" → Paragraph
- "Which Harvard University graduate played a key role in the development of the iPhone?" → Document
- "What is the cheapest iPhone model available in 2023?" → Table
- "Describe the structure of the Eiffel Tower." → Image
- "Describe the moment Darth Vader reveals he is Luke’s father in Star Wars." → Clip
- "Analyze the sequence of events leading to the fall of the Empire in Star Wars." → Video
- "Describe the visual appearance and habitat of the blue whale." → Paragraph+Image
- "Compare the architectural features shown in Gothic and Renaissance cathedrals." → Image+Table
- "Describe the moment of the moon landing and explain the mission details." → Paragraph+Clip

Classify the following query: `{query}`  
Provide only the category or categories combined with ‘+’.

![Prompt for query routing in a training-free manner. The prompt defines each category with concise criteria and illustrative examples.](assets/page_0026_img_1.png)
```

### --- Page 0027 ---

```markdown
Classify the following query into one or more categories from: [No, Paragraph, Passage, Section, Document, ..., Clip, Sequence, Segment, Video], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:

- **Paragraph:** The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- **Passage:** The query requires a detailed block of text (a few paragraphs) from a single source, with added context.
- **Section:** The query requires retrieving an extensive section of a document explaining a sub-topic, possibly with examples or elaboration.
- **Document:** The query requires multi-hop reasoning, combining information from multiple sources or documents to form a complete answer.
- **Clip:** The query targets a short, specific moment or event within a video, without needing full context.
- **Sequence:** The query targets a continuous stretch of related shots (about 10 minutes) that together form a self-contained mini-narrative or process, providing more context and flow than a standalone clip.
- **Segment:** The query targets a longer portion of a video (about 30 minutes) capturing a meaningful sub-scene or subplot-rich and cohesive enough to serve as its own chapter-like unit.
- **Video:** The query requires understanding dynamic events, motion, or sequences over time in a video.

### Examples:
- "What is the birth date of Alan Turing?" → **Paragraph**
- "Summarize Alan Turing’s concept of the Turing Machine." → **Passage**
- "Explain Alan Turing’s contributions to cryptography during WWII." → **Section**
- "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?" → **Document**
- "Describe the moment Messi scored his goal in the 2022 World Cup final." → **Clip**
- "Detail the sequence of passes and movements leading to Messi’s goal in the 2022 World Cup final." → **Sequence**
- "Describe the build-up sequence during the mid-game period of the 2022 World Cup final." → **Segment**
- "Analyze how Argentina won the 2022 World Cup." → **Video**

Classify the following query: `{query}`  
Provide only the category or categories combined with ‘+’.

![Prompt for query routing in a training-free manner with additional granularity choices](assets/page_0027_img_1.png)
```

### --- Page 0028 ---

```markdown
Evaluate whether the query can be answered using general knowledge about the image’s subject rather than relying solely on details unique to the provided image, and verify that the answer is obtainable from the image and the query.

- Respond "yes" if:
  1. The query can be fully answered using general knowledge about the subject.
  2. The answer can be derived solely from the image and the query, without needing image-specific details.
- Respond "no" if either condition is not met.

Example 1:
- Image: A portrait of Donald Trump
- Query: What is the color of Trump’s hair?
- Answer: White
- Response: "yes"

Example 2:
- Image: A close-up photo of a light bulb
- Query: What is the color of the light bulb in this image?
- Answer: Yellow
- Response: "no"

Figure 10: Prompt to filter queries for WebQA.

You will receive a query from a video QA dataset and the title of the corresponding video on YouTube. I want you to paraphrase the query by replacing "in the video?", "of the video", or similar phrases with references to the video content naturally. The output should sound as if a human is asking ChatGPT, and should not explicitly mention the exact name of the video or even parts of the title. However, the rephrased query should contain enough implicit information about the video to allow the model to identify it. Try to reduce the chance of the model getting confused between multiple possible video candidates. If there could be multiple video matches for a given query, try to include more information in the rephrased query.

Example 1:
- Query: What year appears in the opening caption of the video?
- Video Title: Blue Eye Samurai | Hammerscale | Full Episode | Netflix
- Upload Date: 2023-11-05
- Channel Name: Netflix
- Rephrased Output: What year appears in the opening caption of the Blue Eye Samurai episode on Netflix?

Example 2:
- Query: After the vlogger sees a dog with an advertisement from the company named Smitten, camera changes to the scene with ___.
- Video Title: My ICELAND Experience | Ultimate Travel Vlog
- Upload Date: 2022-10-26
- Channel Name: Kallmekris
- Rephrased Output: After spotting a dog with a Smitten advertisement, what scene does the camera transition to in Kallmekris’s Iceland travel vlog from 2022?

Figure 11: Prompt to rephrase queries using video metadata for LVBench and CinePile.
```


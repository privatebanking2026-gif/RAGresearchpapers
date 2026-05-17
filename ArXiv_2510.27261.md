# ArXiv 2510.27261

### --- Page 0001 ---

```markdown
# RegionRAG: Region-level Retrieval-Augmented Generation for Visual Document Understanding

**Yinglu Li, Zhiying Lu, Zhihang Liu, Yiwei Sun, Chuanbin Liu*, Hongtao Xie**  
University of Science and Technology of China, Hefei, China  
{lupluq, ariesiecrack, liuzhihang, syw95}@mail.ustc.edu.cn, {liub92, htxie}@ustc.edu.cn

## Abstract

Multi-modal Retrieval-Augmented Generation (RAG) has become a critical method for empowering LLMs by leveraging candidate visual documents. However, current methods consider the entire document as the basic retrieval unit, introducing substantial irrelevant visual content in two ways: 1) Relevant documents often contain large regions unrelated to the query, diluting the focus on salient information; 2) Retrieving multiple documents to increase recall further introduces redundant and irrelevant documents. These redundant contexts distract the model’s attention and further degrade the performance. To address this challenge, we propose RegionRAG, a novel framework that shifts the retrieval paradigm from the document level to the region level. During training, we design a hybrid supervision strategy from both labeled data and unlabeled data to pinpoint relevant patches. During inference, we propose a dynamic pipeline that intelligently groups salient patches into complete semantic regions. By delegating the task of identifying relevant regions to the retriever, RegionRAG enables the generator to focus solely on concise, query-relevant visual content, improving both efficiency and accuracy. Experiments on six benchmarks demonstrate that RegionRAG achieves state-of-the-art performance, it improves retrieval accuracy by 10.02% in R@1 on five datasets and boosts question answering accuracy by 3.56% while yielding an 11.42% visual tokens compared with prior methods.

**Code —** [https://github.com/Aery666/RegionRAG](https://github.com/Aery666/RegionRAG)  
**Extended version —** [https://arxiv.org/pdf/2510.27261](https://arxiv.org/pdf/2510.27261)

## 1 Introduction

Retrieval-Augmented Generation (RAG) is a powerful paradigm that equips Large Language Models (LLMs) with external knowledge by retrieving relevant context from a dynamic database (Chen et al. 2022; Yasunaga et al. 2022; Liu et al. 2025b). As RAG achieves significant results on text databases, researchers have shifted their focus more to visual document databases with complex visual layouts that hinder knowledge retrieval and grounding (Yu et al. 2024; Fayys et al. 2024; Dong et al. 2025; Tanaka et al. 2025). Early visual document RAG approaches relied on brittle pipelines that first extracted text via Optical Character Recognition (OCR) and layout analysis (Zhang et al. 2024). This process is not only complex but also discards crucial visual information (e.g., layout, style) for holistic comprehension. To address these limitations, the development of Vision-Language Models (VLMs) has spurred a new wave of frameworks that operate directly on document images. Representative works such as VisRAG (Yu et al. 2024), Col-Rep (Fayys et al. 2024), and VDocRAG (Tanaka et al. 2025) leverage VLMs to retrieve documents based on holistic visual and textual features, thereby bypassing the error-prone text extraction stage, preserving richer semantics.

---

**Question:** In which city did partner banks of Always in Beta project come together?

| Retrieval | Score: 0.525 | Image-level RAG | Score: 0.490 |
|-----------|--------------|-----------------|---------------|
|           |              |                 |               |
|           |              |                 |               |
|           |              |                 |               |
|           |              |                 |               |

**Ground truth:** Barcelona  
**RegionRAG (Ours):** Barcelona

![Comparison of (a) Image-level RAG with (b) our proposed RegionRAG. While traditional methods retrieve entire or coarse-grained document images, our RegionRAG identifies and forwards only the most salient regions to the generator. This focused, region-level approach significantly improves final generation accuracy.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
# 2 Related Works

Despite recent progress, existing VLM-based RAG systems typically operate at the document level, treating the entire image as a retrieval unit. This coarse granularity introduces two major types of noise that degrade generation performance. The first type is from the irrelevant information in a document. Even relevant documents often contain large regions unhelpful to the query, diluting the model’s focus on truly salient content. The second type is from the multiple documents specially introduced in RAG. A common strategy is to feed the top-k retrieved documents into the generator (Yao et al. 2025) to compensate for retrieval errors. However, this introduces a performance paradox—while intended to increase ground-truth recall, our experiment shown in Figure 1(a) reveals that increasing k actually leads to a decrease in performance. This may be because including redundant and irrelevant documents significantly increases visual tokens and distracts the model’s attention, thereby outweighing the limited performance gains provided by the small amount of valid information.

To tackle this problem, an intuition is to shift the retrieval granularity from the entire document to only the most relevant semantic regions. Therefore, we propose a novel framework named RegionRAG that enhances retrieval at both the training and inference stages. During training, we improve the alignment between queries and document regions through a hybrid supervision strategy. Specifically, RegionRAG jointly leverages manually annotated bounding boxes from labeled data and a weakly-supervised similarity signs from unlabeled data. A unified loss integrates both super-minimizing annotation cost. During inference, we introduce a dynamic retrieval proposal algorithm that employs query-specific similarity maps. Since relevant information often spans multiple neighboring patches (e.g., parts of a table or paragraph), we apply a neighbor-based grouping strategy that merges patches into coherent regions. This enhances matching precision by isolating concise, context-complete visual segments relevant to the query. As shown in Figure 1(b), our region-level RAG can effectively improve performance as $k$ increases. Extensive experiments demonstrate that RegionRAG achieves a 3.56% average accuracy gain across five document visual question answering (VQA) benchmarks, and 28.58% costs of visual tokens compared to prior methods.

Our main contributions are summarized as follows:

- To the best of our knowledge, we are the first to propose a region-level multi-modal RAG framework (RegionRAG) for visual documents.
- We introduce a dual-objective training strategy that achieves fine-grained patch alignment using different types of data, and introduce a neighbor-based grouping inference strategy to set visual regions.
- Experiments show our RegionRAG achieves more accurate retrieval and efficient generation compared with previous methods.

## Image-Level Retrieval-Augmented Generation (RAG)

RAG enhances Large Language Models (LLMs) by incorporating external knowledge, a paradigm that was initially successful in natural language processing (NLP) tasks (Guu et al. 2020; Borgou et al. 2022; Ram et al. 2023). It typically involves retrieving relevant information from an external knowledge base to guide answer generation (Shi et al. 2023; Yu et al. 2023; Liu et al. 2024). Beyond NLP, RAG has been extended to visually rich documents. Recent works often relied on brittle OCR pipelines (Yu et al. 2023), which discarded visual cues and introduced fragile pre-processing steps. However, this approach discards visual cues and introduces brittle pre-processing steps. Recently, the rapid development of Vision-Language Models (VLMs) (Lu et al. 2021, 2023, 2025a), such as VisRAG (Yu et al. 2024), CoLiF (Yasuda et al. 2024), and DocRAG (Tanaka et al. 2025) has enabled the direct retrieval of entire document images. While pioneering, these methods operate at a coarse, document-level granularity. This forces the generator to process entire images, introducing substantial noise from query-irrelevant content, which dilutes contextual relevance, degrades generation quality, and reduces computational efficiency.

## General Visual Document Understanding

Another line of research focuses on end-to-end understanding of individual visual documents. Traditional OCR-based methods require a corpus to extract cues crucial for reasoning. With the emergence of model LLMs, models like DocFormer (Papadopoulos et al. 2021), Vown-VL (Bai et al. 2023), and InternV (Chen et al. 2024) demonstrate unified reasoning over document images. However, these models are not designed for large-scale retrieval. Retrieval-augmented methods like VisRAG (Yu et al. 2024) and TabPedia (Zhao et al. 2024) leverage visual grounding but still retrieve entire documents, introducing redundant context. In contrast, RegionRAG retrieves fine-grained, query-relevant regions for precise and efficient content construction.

## Region-Based VQA and Grounding

Inspired by the need for finer granularity, another category of models explores region-level reasoning. Studies such as RegionGPT (Guo et al. 2024), region-based VQA methods (Wu et al. 2022), VLM-R3 (Jiang et al. 2025), and DeepEyes (Zheng et al. 2025) adopt an end-to-end generate-and-ground paradigm. However, their tasks also differ crucially from ours: they primarily perform localization or grounding within a single, pre-selected document as known a query. They are not retrieval systems designed to search a corpus. While one could envision an generic, multi-step pipeline for corpus-level retrieval using these models, such approaches are prohibitively slow for efficient retrieval. In contrast, RegionRAG is the first to formalize an explicit, decoupled region-retrieval stage for the RAG paradigm, bridging the gap between efficient retrieval and fine-grained understanding.
```

### --- Page 0003 ---

```markdown
![The framework of RegionRAG. During training, the model jointly learns global (document-query) and local (region-query) alignments. At inference time, it first identifies and retrieves the most relevant visual regions and subsequently generates an answer based on this retrieved context.](assets/page_0003_img_1.png)

## 3 Methods

### 3.1 Preliminary
Our RegionRAG framework redefines the standard RAG pipeline for visually-rich documents. We begin with a corpus $C$ containing $M$ documents, $C = \{D_1, D_2, \ldots, D_M\}$. Each document $D_m$ is segmented into $N$ non-overlapping visual patches $\{p_1, p_2, \ldots, p_N\}$. These patches serve as our fundamental retrieval units, contrasting with traditional methods that operate on whole documents.

Our pipeline consists of two main components. Firstly, a RegionRetriever ($R$), is responsible for identifying the most relevant set of patches $P_R$ from a document $D_m$ given a text query $q$. It uses a shared VLM to embed both the query and the visual patches into a common latent space for fine-grained similarity matching; its function is represented as $P_R = R(q, D_m)$.

We train ($R$) with two levels of alignment to develop its region retrieval ability, a global document-query level and a fine-grained region-query level. Secondly, we employ a Generator ($G$) to synthesize the final textual answer $A$. As an MLM, it is conditioned on the original query $q$ and the set of retrieved image patches $P_R$; this process is denoted as $A = G(q, P_R)$.

### 3.2 Global Document-Query Alignment
While our ultimate goal is to retrieve specific regions, the model must first identify the correct parent document. Our framework first performs Global Document-Query Alignment to obtain a holistic document representation and ensure basic retrieval ability among documents.

We follow (Fayye et al. 2024; Yu et al. 2024) to employ global contrastive learning. To create a holistic document representation from its constituent patches, we aggregate their features into a single global vector. We begin by segmenting a document image $D$ into $N$ patches. A VLM then processes these patches to produce a set of $N$ embeddings, $E_D = \{e_1, e_2, \ldots, e_N\}$, where each $e_i \in \mathbb{R}^d$ and is the embedding dimension. We apply an element-wise max-pooling operation across the $N$ patch embeddings $E_D$ to obtain the single global representation $v_D$. During training, we sample a batch of $B$ document-query pairs $\{(D_i, q_i)\}_{i=1}^B$. For each pair, we compute the document embedding $v_D$ and the query embedding $q_i$. The similarity between the $i$-th query and the $j$-th document is measured by cosine similarity:

$$
s(e_i, v_D) = \frac{e_i \cdot v_D}{\|e_i\| \|v_D\|} \tag{1}
$$

The objective function for global contrastive learning is based on InfoNCE loss (He et al. 2020). For the $i$-th pair in the batch, we treat $(e_i, v_D)$ as the positive pair and $(e_j, v_D)$ for all $j \neq i$ as the negative pairs. The loss for the entire batch is:

$$
L_{global} = -\frac{1}{b} \sum_{i=1}^b \log \left( \frac{\exp(s(e_i, v_D)/\tau)}{\sum_{j=1}^N \exp(s(e_j, v_D)/\tau)} \right) \tag{2}
$$
```

### --- Page 0004 ---

```markdown
# Page 0004

![Our neighbor-based grouping strategy. Patches with high saliency are binarized and then grouped into bounding boxes.](assets/page_0004_img_1.png)

## 3.3 Fine-grained Region-Query Alignment

The coarse-grained global alignment only provides inter-document supervision and cannot achieve precise region-level activation within a document, thus failing to support fine-grained region retrieval. Therefore, we introduce Fine-grained Region-Query Alignment to enable intra-document supervision. This is primarily implemented through the local objective function $L_{local}$, which guides the model to pinpoint the specific regions most relevant to the query. The core of our region-level approach is a hybrid supervision strategy that leverages two distinct signal sources: gold-standard annotations from labeled data (with bounding boxes) and weakly-supervised signals from unlabeled data (without bounding boxes). Both supervision types share a unified loss formulation.

We consider training a batch of $B$ document-query pairs, denoted as $(D_i, q_i)_{i=1}^B$. For each pair, the VLM provides a query embedding $e_q \in \mathbb{R}^d$ and a set of $N$ patch embeddings for the document $D$, denoted as $E_D = \{e_{i,1}, e_{i,2}, \ldots, e_{i,N}\}$. Each patch embedding $e_{i,k} \in \mathbb{R}^d$ represents a specific document region. The core of our fine-grained stage is to compute the similarity between the query $e_q$ and each patch embedding $e_{i,k}$ to identify the most relevant regions.

### Supervision from Labeled Data

For labeled data, each document-query pair $(D_i, q_i)$ is accompanied by ground-truth bounding boxes that delineate the regions essential for answering the query. We use this precise supervision to construct positive and negative sets of patch embeddings. The positive set, $P^+$, consists of all patch embeddings $\{e_k\}$ corresponding to image patches centered within any ground-truth box. Conversely, the negative set, $P^-$, consists of all patch embeddings for patches lying entirely outside these boxes. The local loss $L_{local}$ for a labeled sample encourages the query embedding $e_q$ to be closer to the embeddings in $P^+$ than to those in $P^-$.

### Supervision from Unlabeled Data

A key advantage of our method is its ability to learn from unlabeled data, which contains only weakly-supervised document-query pairs $(D_i, q_i)$ without region annotations. To extract supervisory signals, we generate pseudo-ground-truth labels by leveraging the model's similarity estimates between the query and document patch embeddings. This is feasible because a well-pretrained VLM, guided by the global loss and supervision from labeled data, naturally exhibits some patch-level attention capability. We compute a saliency map that captures the interaction between the query $q_i$ and the document patch embeddings $E_D$. This map is then binarized using a predefined threshold $\theta$ to ensure stable selection of positive samples; we adopt a relatively low value for $\theta$. Patch embeddings with similarity scores above $\theta$ are included in the positive set $P^+$, while those below are assigned to the negative set $P^-$.

Given the positive set $P^+$ and negative set $P^-$ (derived from either ground-truth or pseudo-labels), the unified contrastive loss $L_{local}$ is formulated as:

$$
L_{local} = -\log \left( \frac{\sum_{p^+ \in P^+} \exp(s(e_q, p^+)/\tau)}{\sum_{p^- \in P^-} \exp(s(e_q, p^-)/\tau)} \right) \tag{3}
$$

where $\tau$ is a temperature parameter.

## 3.4 Overall objective

The overall RegionRAG learning objective is a weighted sum of the global and fine-grained alignment loss:

$$
L_{RegionRAG} = L_{Global} + \beta L_{local}, \tag{4}
$$

where the weights $\alpha$ and $\beta$ are hyper-parameters.

## 3.5 Inference

Building upon the model’s capacity for local region attention developed during training, our inference process extracts specific areas to perform region-level retrieval and subsequent answer generation. The pipeline unfolds in three steps: first, we obtain a proposal mask from a retrieval saliency map. Second, we identify connected components of salient patches based on a defined neighborhood size. Third, we extract the minimum bounding rectangle for each element. Finally, we feed the retrieved regions into a generator $G$ to produce the final answer.

### Region Proposal via Saliency Mapping

The inference process begins by identifying candidate regions within a document image $D$ that are potentially relevant to the input query $q$. We first compute a saliency map $S = \{s(e_q, p_k) | k = 1, 2, \ldots, N\}$, which is a 2D grid of cosine similarity corresponding to the layout of the $N$ document patches. To filter out patches with low relevance, we binarize this saliency map into a mask $M_k$ using a predefined threshold $\eta \in [0, 1]$. Patches with scores above this threshold are considered salient candidates:

$$
M_k = \begin{cases}
1, & \text{if } S_k \geq \eta \\
0, & \text{if } S_k < \eta
\end{cases} \tag{5}
$$

This step isolates a sparse set of potentially relevant regions, as shown in Figure 3, preparing them for the subsequent grouping and merging process.
```

### --- Page 0005 ---

```markdown
Algorithm 1: Bounding Box Proposal from Saliency Map

Input: Saliency map $S$, threshold $\eta$, neighbor range $r$, image size $(H_{img}, W_{img})$, grid size $(H_p, W_p)$  
Output: A set of bounding boxes $B$  
1: $M \leftarrow S \geq \eta$  \hspace{1cm}  $\rightarrow$ Binarize saliency map  
2: $R \leftarrow \text{FindComponentsBFS}(M, r)$  \hspace{1cm}  $\rightarrow$ Find connected components  
3: $B \leftarrow \emptyset$  
4: for each component $R_i \in R$ do  
5: \hspace{1cm} $B \leftarrow \text{CalculateMinBBox}(R_i, (H_{img}, W_{img}))$  
6: \hspace{1cm} $B \leftarrow B \cup \{B\}$  
7: end for  
8: return $B$  

Region Merging and Bounding Box Generation. To prevent excessive fragmentation of semantically related areas, we group salient patches from the mask $M$ into connected components. Two patches are considered connected if the distance between their spatial coordinates is less than or equal to a neighborhood radius $r$. We use the Chebyshev distance as our metric:

$$
D((x_1, y_1), (x_2, y_2)) = \max(|x_1 - x_2|, |y_1 - y_2|). \tag{6}
$$

For each connected component found, we compute its minimum enclosing rectangle. These rectangles can be considered our final retrieved regions, and can be further sent into the generator to complete the question answering. For each image, the entire region retrieval process is summarized in the Algorithm 1.

## 4 Experiments

### 4.1 Experimental Settings

**Datasets.** We train our model using the unlabeled dataset from VisRAG (Yu et al. 2024) in-domain data and the document-focused subset of Visual-CoT (Shao et al. 2024) for labeled, bounding-box level supervision. For evaluation, we test our method on a diverse suite of benchmarks including DocVQA (Tito, Karatzas, and Valveny 2023), PlotVQA (Methani et al. 2020), ArxivQA (Li et al. 2024), InfoVQA (Mehta et al. 2022), SlideVQA (Tanaka et al. 2023), and VidDoc (Faysse et al. 2024).

**Evaluation Metrics.** We report the retrieval and generation performance on the evaluation sets of the datasets sourced from the VQA datasets. For retrieval, we use Recall@1, Recall@10, and nDCG@5. For generation, we follow VisRAG (Yu et al. 2024) to report the answer accuracy, employing a relaxed extract metric that allows a 5% error margin for numeric responses.

**Implementation Details.** We initialized RegioRetriever with Qwen2.5-VL-3B (Bai et al. 2025) for training, while

| Models         | #Para | Arxiv | Doc Info | Slide | Avg  |
|----------------|-------|-------|----------|-------|------|
| BM25           | 2M    | 54.3  | 86.8     | 76.0  | 78.3 |
| SigIR          | 883M  | 45.0  | 68.0     | 58.3  | 69.0 |
| bGE(large)     | 335M  | 87.4  | 68.2     | 73.1  | 73.7 |
| NV-Embed-V2    | 758M  | 79.1  | 99.5     | 80.7  | 86.8 |
| ColPal         | 2.92M | 79.4  | 92.8     | 92.8  | 86.5 |
| VisRAG         | 362K  | 78.8  | 77.4     | 91.7  | 92.4 |
| ColQwen2.5*   | 128K  | 84.5  | 97.4     | 85.7  | 85.0 |
| RegionRAG (ours) | 3B  | 92.5  | 99.4     | 99.1  | 96.4 |

Table 1: Retrieval performance comparison between our RegionRAG and other SOTA methods on multiple types of DocumentVQA benchmarks in Recall@10. It notes that we reproduce the evaluation using their official checkpoints.

| Models         | #Data | Recall@1 | #Slide | VidRo |
|----------------|-------|----------|--------|-------|
| VisRAG         | 362K  | 66.2     | 78.4   | 76.4  |
| ColQwen2.5*   | 128K  | 62.3     | 82.2   | 83.3  |
| RegionRAG      | 220K  | 78.4     | 86.8   | 82.6  |

Table 2: Retrieval performance comparison in Recall@1 and nDCG@5.* denotes that we reproduce the evaluation based on their official checkpoints.

The generator is an off-the-shelf model without training. The model is trained for five epochs on a mixture of 98k labeled and 122k unlabeled data. The batch size per GPU is set to 64. The temperature parameter $t$ is set to 0.02 for the global loss and 0.25 for the local loss.

### 4.2 Performance Comparison

We compare our RegionRAG with the following state-of-the-art methods, including OCR-based pipelines BM25 (Robertson, Zaragoza et al. 2009), BGE-large-en v1.5 (Kiao et al. 2024), and NV-Embed-V2 (Lee et al. 2024), and VLM-based methods SigLIP (Zhai et al. 2023), VisRAG (Yu et al. 2024), VDocRAG (Tanaka et al. 2025), ColPal and ColQwen2.5 (Faysse et al. 2024).

**Retrieval Performance.** As shown in Table 1, our RegionRAG outperforms existing methods on the Recall@10 metric across all benchmarks. Compared to VisRAG, which has a comparable number of model parameters (3B) and is trained on 362k samples, our model achieves superior performance with less data (220k). On average, RegionRAG surpasses VisRAG by 4.0 percentage points in Recall@10. When compared with ColQwen2.5, which also adopts Qwen2.5-VL-3B as its backbone, the advantage of RegionRAG becomes even more pronounced. Its average Recall@10 increase is 11.4 percentage points higher (96.4 vs. 85.0), with a huge margin on the Plot benchmark (92.4 vs.
```

### --- Page 0006 ---

```markdown
| Methods        | Region | DocVQA | InfoVQA | Average |
|----------------|--------|--------|---------|---------|
| ColQwen2.5     | Image  | 0.062  | 0.109   | 0.086   |
|                 | Bbox   | 0.112  | 0.138   | 0.125   |
| RegionRAG (Ours)| Image  | 0.119  | 0.226   | 0.173   |
|                 | Bbox   | 0.271  | 0.346   | 0.309   |

Table 3: Comparison of average similarity scores between text queries and different visual scopes (full image vs. ground-truth bounding box) on DocVQA and InfoVQA.

47.5). These results demonstrate that our proposed method of supervising the most query-relevant image regions is even superior on the whole document retrieval than employing image-level supervision. This success can be attributed to our tailored training scheme and loss design. We further compare RegionRAG with recent VLM-based methods on Recall@1 and nDCG@5 metrics in Table 2. Our model obtains leading performance on most benchmarks, highlighting its consistent superiority. While its scores on ViDRe are marginally behind those of the top performer, this evaluation constitutes a partial zero-shot setting for our model due to the limited subset domain coverage in our training data.

### Region-Test Alignment Capability
We evaluate the model's ability to localize query-relevant regions by comparing the average retrieval scores of our method and the ground-truth bounding box (Bbox) from the Visual-Context box, as shown in Table 3, ColQwen2.5 and our RegionRAG achieve higher similarity on Bbox regions, indicating that relevant information is locally concentrated. Notably, RegionRAG yields a larger similarity gain (0.136 vs. 0.039 for ColQwen2.5), highlighting the effectiveness of our region-level contrastive loss in capturing fine-grained, query-relevant visual cues.

### Generation Performance
We evaluate our method under both fixed and dynamic resolution settings. In the fixed setup, we process the retrieved region at either 256² or 512² pixel resolutions. As shown in Table 4, accuracy improves notably at both resolutions, and using interleaved boxes consistently outperforms full-image inputs under the same resolution. In the dynamic setting, which better reflects real-world inference, the model uses the original resolution with up to 512². On average, the bbox input consumes only 71.4% of the image tokens while maintaining or even surpassing full-image performance. For instance, it achieves 63.79% higher accuracy on InfoVQA. Notably, the full-image method is superbounded by the “oracle” setting, which inputs the ground truth and a fixed resolution. Interestingly, our bounding box approach can even surpass this baseline. For example, on InfoVQA at 512² resolution, our top-4 bbox method attains 63.79% accuracy, exceeding the oracle’s 51.39%. This is because downscaling affects smaller cropped regions less severely than entire images—preserving finer details that enable the model to better perceive the relevant content. Overall, these results highlight the advantages of our region retrieval in both efficiency and fidelity.

| Pixel methods   | Arxiv  | DocVQA | SlideVQA |
|------------------|--------|--------|----------|
| top1 image       | 62.59  | 19.97  | 24.23    | 16.69   | 39.49   |
| top1 bbox        | 62.13  | 31.13  | 39.13    | 17.15   | 39.92   |
| top4 image       | 61.40  | 18.95  | 22.47    | 38.31   | 62.42   |
| top4 bbox        | 62.62  | 42.3   | 44.15    | 19.40   | 44.42   |
| oracle           | 64.71  | 19.97  | 24.37    | 30.39   | 39.39   |

Table 4: Performance comparison under fixed and dynamic resolutions across five VQA benchmarks. “top image/bbox” denotes processing the retrieved full image or bounding box, “oracle” uses ground-truth images, “dynamic” means using the original input with up to 512².

| Methods         | ArxivQA | DocVQA | SlideVQA |
|------------------|---------|--------|----------|
| w/o Lglobal      | 77.08   | 84.94  | 79.24    | 88.90   |
| w/o Lglobal      | 77.71   | 68.55  | 81.14    | 78.24   |
| w/o Llocal       | 59.31   | 84.24  | 82.40    | 78.74   |
| w/o Unlabeled    | 73.90   | 67.75  | 71.07    | 75.00   |

Table 5: Main ablation study (%) of retrieval results for global contrastive learning ($L_{global}$), regional contrastive learning ($L_{local}$) with labeled and unlabeled data.

### 4.3 Ablation
Main Ablation. The core components of our method include the global contrastive loss ($L_{global}$), the regional contrastive loss ($L_{local}$), and a semi-supervised strategy that utilizes both labeled and unlabeled data. As shown in Table 5, removing either $L_{global}$ or $L_{local}$ consistently leads to performance degradation across all datasets, indicating that both global document-level and fine-grained region-level contrastive learning are crucial for effective retrieval. Moreover, the results underscore the effectiveness of our semi-supervised data strategy. While discarding labeled data causes a substantial performance collapse, excluding unlabeled data also yields a noticeable decline (e.g., a 4.53% drop in R@1 on ArxivQA). These findings confirm that our model not only relies on the supervised signal but also effectively leverages large-scale unlabeled data to enhance its generalization and robustness.
```

### --- Page 0007 ---

```markdown
![Visualization of similarity maps. The relevant regions are highlighted with red boxes for better reading. Our RegionRAG obtains better localization ability, as it exhibits higher and more concentrated similarity in relevant regions.](assets/page_0007_img_1.png)

Figure 5: Ablation study on generation inputs. We compare using BBox versus Image as input to the generator. (a) and (b) show accuracy versus input resolution for a fixed $k = 4$. (c) and (d) show accuracy as a function of top-$k$ candidates at a fixed resolution.

### Ablation Study on Generation Inputs
We conduct an ablation study to analyze the impact of input granularity on the generator. Specifically, we compare providing cropping bounding box (BBox) regions against full image pages, focusing on two factors: input resolution (i.e., visual token count) and the number of top-$k$ retrieved candidates. As shown in Figure 5(a) and (b), when inputs are constrained to a fixed resolution, the BBox method consistently outperforms the Image method. The advantage is most pronounced at lower resolutions like $256^2$ and $448^2$, where BBox accuracy exceeds Image by up to $21.73$ points. This improvement arises because resizing a small BBox crop enlarges its informative content compared to a down-sampled full image. Consequently, our approach achieves higher performance with fewer tokens (e.g., BBox at $256^2$ rivals Image at $448^2$ on InfoVQA). At very high resolution ($1024^2$), the gap narrows as Qwen2.5-VL already localizes answers in high-resolution images. Overall, our method strikes a better balance between performance and efficiency. Figure 5(c) and (d) further analyze the effect of the top-$k$ setting at a fixed resolution.

### 4.4 Qualitative Analysis
To qualitatively investigate how RegionRAG achieves its strong performance, we visualize the similarity heatmaps in Figure S1. The figure clearly shows where the model gives the relevant regions to retrieve from a given query, comparing our method with ColQwen2.5, which processes the entire image. The similarity map of RegionRAG is significantly more focused on the query-relevant visual parts. In contrast, ColQwen2.5 focuses diffusely across the image, indicating a difficulty in distinguishing relevant information from background noise. For example, for the author Jeffry Float, our model points to the exact ‘1984’ entry on the timeline. Similarly, for the other example, RegionRAG precisely pinpoints the relevant charts and data points. This shows that our region-level retrieval mechanism effectively guides the model to the most salient information, which is key to its superior performance.

### 5 Conclusion
In this work, we presented RegionRAG, an innovative framework that enhances visual document RAG by retrieving fine-grained semantic regions instead of entire documents. By leveraging a dual-objective training strategy and a neighbor-based region grouping algorithm, RegionRAG effectively filters irrelevant context and provides the generator with precise visual evidence. Our method achieves state-of-the-art performance across six benchmarks, significantly improving both retrieval (+10.21% in R@1) and question answering accuracy (+3.56%) while cutting visual token costs by 28.58%. In the future, we aim to build more efficient and scalable RAG systems, extending to more general scenarios.
```

### --- Page 0008 ---

```markdown
# Acknowledgments

This work is supported by the National Nature Science Foundation of China (6245141, 62121002,  U123B208, 62272436). We thank the support of the GPU cluster built by MCC Lab of Information Science and Technology Institution, USTC, and USTC supercomputing center for providing computational resources for this project.

## References

Appalaraju, S.; Jasani, B.; Kota, B. U.; Xie, Y.; and Mantra, R. 2021. Boostformer: End-to-end transformer for document understanding. In Proceedings of the IEEE/CVF international conference on computer vision, 993–1003.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-V: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12694, 1(2).

Bai, S.; Chen, X.; Liu, K.; Wang, S.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen-5 v1 technical report. arXiv preprint arXiv:2502.13923.

Beyer, L.; Steiner, A.; Pinto, A. S.; Kolesnikov, A.; Wang, X.; Salz, D.; Neumann, M.; Alabdulmohsin, I.; Tschannen, M.; Bugliarello, E.; et al. 2024. Paligemma: A versatile 3D visual transformer. arXiv preprint arXiv:2407.07726.

Borgeaud, S.; Mena, A.; Hoffmann, J.; Cai, T.; Rutherford, E.; Millican, K.; van Deursen, B.; Lespiau, J.-B.; Damour, B.; Clark, A.; et al. 2022. Improving language models by retrieving from trillions of tokens. In International Conference on Machine Learning, 2202–2212.

Chen, W.; Hu, H.; Saharia, C.; and Cohen, W. W. 2022. Reimage: Retrieving relevant text-to-image generator. arXiv preprint arXiv:2209.14941.

Chen, Z.; Liu, Y.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhang, M.; Zhang, Q.; Zhu, X.; Lu, L.; et al. 2024. Instruct: Scaling vision foundation models and aligning for service visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 24185–24198.

Deng, C.; Yuan, J.; Bu, P.; Wang, P.; Li, Z.-Z.; Xu, J.; Li, X.; Huang, Y.; Song, J.; Zheng, B.; et al. 2024. Longdoc: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. arXiv preprint arXiv:2412.18424.

Dong, Y.; Veda, N.; Bors, K.; Ito, D.; Sera, T.; and Oyamada, M. 2025. SCAN: Semantic Document Layout Analysis for Textual and Visual Retrieval-Augmented Generation. arXiv preprint arXiv:2505.14381.

Faysee, M.; Sibille, H.; Wu, T.; Omrani, B.; Viaud, G.; Hudelot, C.; and Colombo, P. 2024. Coplati: Efficient document retrieval with vision language models. In The Thirteenth International Conference on Learning Representations.

Guo, Q.; De Mello, S.; Yin, H.; Byeon, W.; Cheung, K. C.; Yu, Y.; and Lin, S. 2024. Regoint: Towards rethinking understanding vision language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13796–13806.

Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M. 2020. Retrieval augmented language model: pre-training. In International conference on machine learning, 3929–3938. PMLR.

He, K.; Fan, H.; Wu, Y.; Xie, S.; and Girshick, R. 2020. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9729–9738.

Jiang, C.; Heng, Y.; Ye, Y.; Yang, H.; Xu, H.; Yan, M.; Zhang, J.; Huang, F.; and Shang, S. 2025. VLM-R 3: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought. arXiv preprint arXiv:2505.16192.

Lee, C.; Roy, R.; Xu, M.; Raiman, J.; Shoeybi, M.; Catanzaro, B.; and Ping, W. 2024. NV-embed: Improved techniques for training LLMs as generalist embedding models. arXiv preprint arXiv:2405.17482.

Li, L.; Wang, Y.; Xu, R.; Wang, P.; Feng, X.; Kong, L.; and Liu, Q. 2024. Multimodal long video dataset for improving scientific comprehension for vision-language models. arXiv preprint arXiv:2403.00231.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual instruction tuning. Advances in neural information processing systems, 36: 34829–34816.

Liu, Z.; Li, J.; Xie, H.; Li, P.; Ge, J.; Liu, S.-A.; and Jin, L. 2024. Towards balanced and efficient multimodal modeling for video information retrieval. In Proceedings of the AAAI conference on artificial intelligence, 3853–3860.

Liu, Z.; Xie, C.-W.; Li, P.; Zhao, L.; Tang, L.; Zheng, Y.; Liu, C.; and Xie, H. 2025a. Hybrid-level instructions for video token compression in multi-modal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 8568–8578.

Liu, Z.; Xie, C.-W.; Wen, B.; Yu, F.; Li, P.; Zhang, B.; Yang, N.; Gao, Z.; Zheng, Y.; Xie, H.; et al. 2025b. CApability: A Comprehensive Visual QA Benchmark for Evaluating Both Correctness and Thoroughness. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Mathew, M.; Bagal, V.; Tifo, R.; Tazakis, D.; Valvency, E.; and Jawahar, C. 2022. Infographicho. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 1697–1706.

Mathew, M.; Karatzas, D.; and Jawahar, C. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2200–2209.

Methani, N.; Ganguly, P.; Khapra, M. M.; and Kumar, P. 2024. Ploqit: Reasoning over scientific plots. In Proceedings of the ieee/cvf winter conference on applications of computer vision, 1527–1536.

Ramo, Q.; Levine, Y.; Dalmidos, I.; Muhlbay, D.; Shashua, A.; Leyton-Brown, K.; and Shoham, Y. 2023. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11: 1316–1331.
```

### --- Page 0009 ---

```markdown
Robertson, S.; Zaragoza, H.; et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends® in Information Retrieval, 3(4): 333–389.

Shao, H.; Qian, S.; Xiao, H.; Song, G.; Zong, Z.; Wang, L.; Liu, Y.; and Li, H. 2024. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for channel-of-thought reasoning. Advances in Neural Information Processing Systems, 37: 8167–8642.

Shi, W.; Min, S.; Yasunaga, M.; Seo, M.; James, R.; Lewis, M.; Zettlemoyer, L.; and Yih, W.-t. 2023. Replug: Retrieval-augmented black-box language models. arXiv preprint arXiv:2301.12652.

Sidorov, O.; Hu, R.; Rohrbach, A.; and Singh, A. 2020. Textcaps: a dataset for image captioning with reading comprehension. In European conference on computer vision, 742–758. Springer.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Ba, D.; Parikh, D.; and Rohrbach, M. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8317–8326.

Tanaka, R.; Iki, T.; Hasegawa, T.; Nishida, K.; Saito, K.; and Kuroki, J. 2025. Vdocars: Retrieval-augmented generation over visually-rich documents. In Proceedings of the Computer Vision and Pattern Recognition Conference, 24827–24837.

Tanaka, R.; Nishida, K.; Nishida, K.; Hasegawa, T.; Saito, I.; and Kuroki, J. 2023. Visual question answering on multiple images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 13636–13645.

Tito, R.; Karatzas, D.; and Valveny, E. 2023. Hierarchical multimodal transformers for multiple page docvqa. Pattern Recognition, 114: 109834.

Yu, Z.; Xiong, C.; Yu, S.; and Liu, Z. 2023. Augmentation-adapted retriever improves generalization of language models as generic plug-in. arXiv preprint arXiv:2305.17331.

Zhai, X.; Mustafa, B.; Kolesnikov, A.; and Beyer, L. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, 11975–11986.

Zhang, G.; Qu, S.; Liu, J.; Zhang, C.; Lin, C.; Yu, L.; Pan, D.; Cheng, E.; Liu, J.; Lin, Q.; et al. 2024. Map-neo: Highly capable and transparent bilingual large language model series. arXiv preprint arXiv:2405.19327.

Zhao, W.; Feng, H.; Liu, Q.; Tang, J.; Wu, B.; Liao, L.; Wei, S.; Ye, Y.; Liu, H.; Zhou, W.; et al. 2024. Tabpedia: Towards comprehensive visual table understanding with concept synthesis. Advances in Neural Information Processing Systems, 37: 7185–7212.

Zheng, Z.; Yang, M.; Hong, J.; Zhao, C.; Xu, G.; Yang, L.; Shen, C.; and Yu, X. 2025. DeepKeys: Incentivizing Thinking with Images via Reinforcement Learning. arXiv preprint arXiv:2505.14562.

Yao, J.; Liu, S.; Wang, Y.; Mei, L.; Bi, B.; Ge, Y.; Li, Z.; and Cheng, X. 2025. Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation. arXiv preprint arXiv:2506.11063.

Yasunaga, M.; Aghayan, A.; Shi, W.; James, R.; Leskovec, J.; Liag, P.; Lewis, M.; Zettlemoyer, L.; and Yih, W.-t. 2022. Retrieval-augmented multimodal language modeling. arXiv preprint arXiv:2211.12561.

Yu, S.; Tang, C.; Xu, B.; Cui, J.; Ran, J.; Yan, Y.; Liu, Z.; Wang, S.; Han, X.; Liu, Z.; et al. 2024. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594.
```

### --- Page 0010 ---

```markdown
# Technical Appendix

Overview. In the Appendix, we introduce more methods details in Sec. A, more implementation details in Sec. B, more dataset details in our training in Sec. C. Then we add more experiments in Sec. D, such as hyper-parameter studies, inference efficiency analysis.

## A Methods Details

### Algorithm S1: Bounding Box Proposal from Saliency Map (Detailed)

**Input:** Saliency map $S$, bbox threshold $\eta$, neighborhood range $r$, image size $(H_{img}, W_{img})$, patch size $(H_p, W_p)$  
**Output:** A set of bounding boxes $B$

1. $M \gets \{(x,y) \mid S(x,y) \geq \eta\}$  \# Binarize saliency map to get a mask  
2. $P_{salient} \gets \text{get\_grid.coordinates}(M) \quad \text{Get } D \text{ indices of salient patches}$  
3. $R \gets \emptyset$  \# Initialize list of connected regions  
4. $visited \gets \emptyset$  \# Initialize set of visited patch coordinates  
5. for each patch coordinate $p \in P_{salient}$ do  
6. \quad if $p \notin visited$ then  
7. \quad \quad $Q \gets \text{collections.deque}()$  \# Initialize a queue for BFS  
8. \quad \quad $current \gets \emptyset$  \# Initialize the current region  
9. \quad \quad $Q.\text{append}(p)$  
10. \quad \quad $visited.\text{add}(p)$  
11. \quad \quad while $Q$ is not empty do  
12. \quad \quad \quad $current \gets Q.\text{popleft}()$  
13. \quad \quad \quad $R_{current} \gets \text{current}$  
14. \quad \quad \quad for each neighbor $p_{neighbor}$ of $current$ within range $r$ do  
15. \quad \quad \quad \quad if $p_{neighbor} \in P_{salient}$ and $p_{neighbor} \notin visited$ then  
16. \quad \quad \quad \quad \quad $visited.\text{add}(p_{neighbor})$  
17. \quad \quad \quad \quad \quad $Q.\text{append}(p_{neighbor})$  
18. \quad \quad \quad end if  
19. \quad \quad end for  
20. \quad \quad end while  
21. $R.\text{add}(R_{current})$  
22. end for  
23. $B \gets \emptyset$  \# Initialize the set of final bounding boxes  
24. for each region $R \in R$ do  
25. \quad $(x_{min}, y_{min}) \gets (\min_{p \in R.p.x}, \min_{p \in R.p.y})$  \# Find min grid coordinates  
26. \quad $(x_{max}, y_{max}) \gets (\max_{p \in R.p.x}, \max_{p \in R.p.y})$  \# Find max grid coordinates  
27. \quad $x_1 \gets x_{min} \times W_p$  
28. \quad $y_1 \gets y_{min} \times H_p$  
29. \quad $x_2 \gets \min((x_{max} + 1) \times W_{img})$  
30. \quad $y_2 \gets \min((y_{max} + 1) \times H_{img})$  
31. \quad $B.\text{add}((x_1, y_1, x_2, y_2))$  
32. end for  
33. return $B$  

To further clarify our method for extracting candidate regions from the saliency map (as described in Section 3.5 of the main text), this section provides a detailed algorithmic description. Algorithm S1 aims to effectively merge spatially adjacent patches with saliency scores above a threshold into meaningful semantic regions and to generate a minimum bounding box for each region. Specifically, the process begins by filtering for all patches with saliency scores exceeding a preset threshold $\eta$. It then employs a Breadth-First Search (BFS) to identify connected components, grouping adjacent patches that fall within a neighborhood range $r$. Once a connected region of patches is identified, the algorithm calculates its minimum bounding box by determining the external grid coordinates of all constituent patches and converting them into actual pixel coordinates on the image. This ultimately yields a set of precise candidate regions for the subsequent generator.

## B Implementation Details

We initialized RegionRetriever with Qwen2.5-VL-3B (Bai et al. 2025) for training, while the generator is an off-the-shelf model without training. The model is trained using bfloat16 mixed-precision, and we leverage the flash attention-2 implementation for efficiency. Instead of full fine-tuning, we employ Parameter-Efficient Fine-Tuning (PEFT) using the LoRA methodology. The LoRA configuration features a rank of $32$ and an $l_{0}$ of $32$, with a dropout rate of $0.1$. LoRA adapters are applied to all linear projection layers. We used the AdamW optimizer with an initial learning rate of $2 \times 10^{-4}$ and a warmup phase of $100$ steps. The temperature for the global contrastive loss ($\tau$) is set to $0.02$, while the temperature of the local contrastive loss is $0.25$. Given the global loss weighted by $\alpha = 1$, the local loss component was weighted by a coefficient $\beta = -0.01$ in the final loss calculation.

## C Dataset Details

Our model is trained on a hybrid dataset composed of both unlabeled and labeled data, with statistics detailed in Table S1. For unlabeled data training, we utilize the VisRAG (Yu et al. 2024) in-domain dataset, which covering document types from scientific figures to industrial documents, is crucial for enhancing our model's generalization capability. For fine-grained supervision, the labeled data is sourced from Visual-CoT, which is composed of several benchmarks, including TextVQA (Singh et al. 2019), TextCaps (Sidorov et al. 2020), DocVQA (Mathew, Karatzas, and Jawahar 2021), and InfographicVQA (Mathew et al. 2022).

Our ablation studies (as shown in Table S2) reveal the explicit and distinct contributions of each data type, as the impact of removing them varies across datasets. For instance, on DocVQA, removing unlabeled data causes a more significant drop in R@1 (from $86.80\%$ to $71.07\%$) than removing labeled data (to $82.40\%$). Conversely, on ArxivQA, the
```

### --- Page 0011 ---

```markdown
| Domain                | Source Dataset               | Size  | Dataset Description                     |
|-----------------------|------------------------------|-------|-----------------------------------------|
| Unlabeled data        | VisRAG-in-domain             | 122k  | Diverse document images with text       |
| Labeled data          | Visual-CoT TextVQA           | 18k   | Images with text                        |
|                       | Visual-CoT TextCaps          | 32k   | Images with text                        |
|                       | Visual-CoT DocVQA            | 33k   | Document images                         |
|                       | Visual-CoT InfographicsVQA   | 15k   | Infographic                             |

Table S1: The overview of the document-focused subset of Visual-CoT. The dataset includes four source datasets.

![TextVQA example](assets/page_0011_img_1.png)
Question: what year was the book made?  
Answer: 2009  
Bbox: [320, 860, 396, 898]  

![TextCaps example](assets/page_0011_img_2.png)
Question: Which city is indicated on the license plate?  
Answer: ROMA  
Bbox: [135, 176, 381, 323]  

![DocVQA example](assets/page_0011_img_3.png)
Question: Who is the program supervisor for the internship program?  
Answer: Mark Beckman  
Bbox: [524, 1084, 825, 1127]  

![InfoVQA example](assets/page_0011_img_4.png)
Question: How many have found home working very difficult?  
Answer: 22%  
Bbox: [82, 884, 140, 909]  

Figure S1: Examples of four sources covered in the Visual-CoT dataset, with corresponding question-answer annotations and bboxes. The red bounding boxes in the images highlight the critical image regions that provide necessary and related information for answering the questions.

effect is reversed and even more pronounced: removing labeled data leads to a dramatic drop in R@1 (from 78.43% to 59.31%), far greater than the impact of removing unlabeled data (to 73.90%). This complex interaction suggests that the value of labeled data is not simply to boost a retrieval score, but to provide an invaluable and unique type of supervision signal. Specifically, tasks in datasets like DocVQA and TextVQA (e.g., “What year was the book made?”) require the model to directly analyze textual content from the image, rather than retrieving answers from an external knowledge base. Therefore, while such samples may not be ideal for training a conventional retrieval module, they are highly beneficial for guiding our model to accurately locate question-relevant information within the visual content, which is central to our RegionRAG framework.

D.1 Retrieval Performance  
To more comprehensively evaluate the retrieval performance of our proposed RegionRAG model, we provide more detailed experimental results in Table S2, expanding upon Ta-
```

### --- Page 0012 ---

```markdown
| Methods         | R@1   | R@2   | R@5   | R@10  | N@1   | N@2   | N@5   | N@10  |
|------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| **ArxivQA**      |       |       |       |       |       |       |       |       |
| VisRAG           | 69.12 | 76.59 | 83.16 | 87.62 | 69.12 | 73.88 | 76.98 | 78.36 |
| ColQwen2.5      | 62.25 | 70.83 | 80.27 | 84.56 | 62.25 | 67.67 | 71.83 | 72.37 |
| RegionRAG(Ours) | 78.43 | 83.33 | 89.95 | 92.52 | 78.43 | 81.52 | 84.52 | 85.35 |
| **DocVQA**       |       |       |       |       |       |       |       |       |
| VisRAG           | 66.16 | 77.73 | 86.97 | 91.20 | 66.16 | 73.20 | 77.72 | 79.13 |
| ColQwen2.5      | 80.80 | 86.63 | 94.59 | 97.46 | 80.20 | 84.26 | 87.89 | 88.85 |
| RegionRAG(Ours) | 86.63 | 92.89 | 98.24 | 99.32 | 86.63 | 90.58 | 92.98 | 93.39 |
| **InfoVQA**      |       |       |       |       |       |       |       |       |
| VisRAG           | 78.83 | 89.83 | 95.68 | 97.08 | 78.83 | 85.77 | 88.47 | 88.91 |
| ColQwen2.5      | 82.31 | 89.97 | 96.94 | 98.21 | 82.31 | 87.15 | 90.31 | 90.85 |
| RegionRAG(Ours) | 89.00 | 95.96 | 99.03 | 99.44 | 89.00 | 93.39 | 94.81 | 94.95 |
| **PlQnA**        |       |       |       |       |       |       |       |       |
| VisRAG           | 49.13 | 61.88 | 80.76 | 68.38 | 49.13 | 57.11 | 65.54 | 68.38 |
| ColQwen2.5      | 9.27  | 14.83 | 28.62 | 47.51 | 9.27  | 12.78 | 18.96 | 25.05 |
| RegionRAG(Ours) | 54.60 | 68.13 | 80.38 | 92.35 | 54.69 | 63.17 | 69.84 | 72.88 |
| **SlideVQA**     |       |       |       |       |       |       |       |       |
| VisRAG           | 76.35 | 87.77 | 95.05 | 97.12 | 79.01 | 87.09 | 90.90 |       |
| ColQwen2.5      | 76.03 | 87.95 | 94.60 | 96.76 | 76.80 | 83.57 | 86.74 |       |
| RegionRAG(Ours) | 80.04 | 91.37 | 97.30 | 98.20 | 80.04 | 87.18 | 90.01 |       |

![Overall retrieval performance in Recall@k and nDCG@k for k = 1, 2, 5, 10.](assets/page_0012_img_1.png)

Figure S2: Analysis of the impact of key hyper-parameters on retrieval performance (Recall@1). (a) The effect of the positive pseudo-label threshold ($\theta$), used for training on unlabeled data, on Recall@1. (b) The effect of the local loss weight ($\beta$) on Recall@1. The results indicate that the model achieves optimal performance when $\theta = 0$ and $\beta = 0.01$.

Table S2: Overall retrieval performance in Recall@k and nDCG@k for $k = 1, 2, 5, 10$.

D.2 Hyper-parameter Analysis

Analysis of Positive Pseudo-label Threshold $\theta$. This threshold is critical for automatically partitioning pseudolabeled patches from the similarity map when training on unlabeled data, directly impacting the model’s retrieval performance. Figure (a) illustrates the impact of this threshold on the model’s retrieval performance (Recall@1) across four datasets. We tested three values: -0.2, 0, and 0.2. The results clearly show that Recall@1 most benchmarks peaks when the threshold is set to 0. This aligns with our intuition: a threshold that is too low (e.g., -0.2) might introduce excessive noise by incorrectly labeling irrelevant patches as positive. Conversely, a threshold that is too high (e.g., 0.2) 
```

### --- Page 0013 ---

```markdown
![Hyper-parameter analysis: (a) The effect of neighbor range on accuracy. (b) The effect of box threshold on accuracy. A box threshold of -1 is equivalent to using the full image as input.](assets/page_0013_img_1.png)

| Model         | R@1   | R@2   | R@10  |
|---------------|-------|-------|-------|
| ColQwen2.5   | 10.13 | 13.50 | 20.08 |
| RegionRAG    | 12.58 | 16.91 | 23.24 |

Table S3: Comparison of retrieval results between RegionRAG and ColQwen2.5 on LongDocURL.

| Method | Retrieval Time | Generation Time |
|--------|----------------|------------------|
| Image  | 217.3          | 435.5            |
| Bbox   | 309.6          | 291.1            |

Table S4: Comparison of retrieval, generation, and total inference time (in seconds) between image-level and region-level methods from InfoVQA.

### D.3 Long-context Experiment
To further evaluate the robustness of RegionRAG in long-context scenarios, we conducted experiments on Long-DocURL (Jeng et al. 2024), a benchmark involving multi-page (often >50) and lengthier document inputs. As shown in Table S3, RegionRAG consistently outperforms ColQwen2.5 across all retrieval metrics. These results demonstrate that RegionRAG effectively scales to long-document settings, where query-relevant content is more sparsely distributed and retrieval requires stronger discrimination across extensive visual contexts. The consistent gains confirm that the proposed region-level retrieval mechanism remains effective under longer input contexts, supporting its generalization beyond short and medium length document benchmarks.

### D.4 Inference Efficiency Analysis
To further assess the computational efficiency of RegionRAG, we measure the inference latency on InfoVQA, comparing image-level (Image) and region-level (Bbox) pipelines. As shown in Table S4, the Image approach incurs slightly higher retrieval time (309.6 vs. 217.3) due to additional region selection and merging operations. How-
```

### --- Page 0014 ---

```markdown
| Pixel Methods | Archive | Doc | Info | Plot | Slide |
|---------------|---------|-----|------|------|-------|
| 256           | top1   | bbox | 60.17 | 34.88 | 17.17 | 46.76 |
|               | top1   | bbox | 61.52 | 36.37 | 49.44 | 17.92 | 46.94 |
|               | top4   | image | 59.31 | 31.30 | 38.76 | 17.03 | 48.56 |
|               | top4   | bbox | 59.44 | 45.69 | 54.17 | 17.69 | 49.11 |
|               | oracle | 61.52 | 36.55 | 42.34 | 20.05 | 51.44 |
| 512           | top1   | bbox | 59.31 | 65.31 | 52.08 | 17.38 | 53.59 |
|               | top1   | bbox | 59.21 | 73.44 | 56.55 | 18.08 | 54.16 |
|               | top4   | image | 58.82 | 65.99 | 48.05 | 16.69 | 55.83 |
|               | top4   | bbox | 60.42 | 76.64 | 49.86 | 17.57 | 58.09 |
|               | oracle | 63.24 | 70.89 | 55.71 | 23.43 | 56.12 |

Table S5: Performance comparison under fixed and dynamic resolutions across five VQA benchmarks (GPT-4o as generator). “top1 image/bbox” denotes processing the retrieved image or bounding box, “oracle” uses ground-truth images.

---

| Model data | Archive | Doc | Info | Plot | Slide |
|------------|---------|-----|------|------|-------|
|            | 362k    | 87.62 | 91.20 | 97.08 | 68.38 | 97.12 |
|            | visrag  | 220k   | 81.31 | 92.53 | 61.87 | 96.86 |
|            | Our     | 220k   | 92.52 | 99.94 | 42.35 | 98.20 |

Table S6: Retrieval performance (Recall@10) comparison between RegionRAG and fine-tuned baseline (VisRAG) on our 220k training set.

---

| Method | Bbox | Image |
|--------|------|-------|
| Accuracy | 33.1 | 20.7 |

Table S7: Generation performance on InfoVQA using PaliGemma-3B as the retriever. The Bbox input still outperforms the Image input under the top-1 QA setting.

---

D.5 Evaluation with GPT-4o as Generator  
To verify the generalizability of our approach beyond Qwen2.5-VL, we further conduct experiments by replacing the generator with GPT-4o while keeping the same retrieval pipeline. As shown in Table S5, our region-level tuning strategy consistently performs the image-level counterpart across all benchmarks and resolutions. The advantage is especially evident at lower resolutions (e.g., $256^2$), where the Bbox-based input achieves up to 10.6 points higher accuracy on InfoVQA. This demonstrates that our region-aware retrieval mechanism provides clear and focused visual grounding that benefits even a powerful generator like GPT-4o. Moreover, the performance gap remains stable at higher resolutions ($512^2$), indicating that the effectiveness of RegionRAG is model-agnostic and transfers well to stronger large multi-modal models. These results confirm that the improvements brought by our method stem from better input organization rather than being tied to a specific generator architecture.

D.6 Fine-tuning Baselines on Our Dataset  
To ensure a fair comparison, we further fine-tune VisRAG and ColQwen2.5 on our 220k training set and compare their retrieval performance. ColQwen2.5 shares the same backbone as our base model (Qwen2.5-L-3B), and its fine-tuning configuration corresponds to the “w/o local loss” setting in our ablation study (Table 5). As shown in Table S6, VisRAG exhibits a noticeable performance drop when trained on our smaller dataset (220k vs. its original 362k samples). In contrast, RegionRAG consistently outperforms both fine-tuned VisRAG and ColQwen2.5 across all benchmarks, achieving substantial improvements particularly on the PlotQA and ArxivQA datasets. These results demonstrate that the superior performance of RegionRAG stems primarily from our proposed regional contrastive learning and retrieval strategy, rather than from finetuning setup.

D.7 More Qualitative Analysis  
To intuitively demonstrate the superior localization ability of our RegionRAG model, we provide a qualitative comparison of similarity heatmaps against the ColQwen2.5 baseline on the VideoQA and InfoVQA datasets in Figure S4 and Figure S5. These visualizations clearly show that for a given question, RegionRAG generates highly focused and intense “hotspots” that precisely cover the text or chart area containing the answer. For instance, in a table-based question from DocVQA (Figure S4, fourth from left), RegionRAG accurately focuses its attention on the specific cell corresponding to “no reported meal reaction revisions”, whereas the attention from ColQwen2.5 is comparatively diffuse and fails to precisely lock onto the information. This precise localization capability is a direct result of our proposed region-level contrastive learning objective, which enables the model to effectively filter irrelevant information and focus on the most critical evidence, thereby significantly boosting performance on complex visual document understanding tasks.

D.8 Generalization Across Model Architectures  
PaliGemma. PaliGemma (Beyer et al. 2024) is a versatile 3B vision-language model built upon the Gemma language backbone and the Pali visual encoder. Despite its parameter size, it demonstrates strong transferability across diverse multimodal tasks such as VQA and document understanding. In our experiments, we adopted PaliGemma-3B.
```

### --- Page 0015 ---

```markdown
| Q: What is the cat.no of Envelopes-plain Manila-9.1? 12/12/12? | Q: Where in Winston-Salem, NC accommodation arranged? | Q: What is the percentage shown under “no reported metal reaction revisions”? |
|-------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------|
| ![Qualitative comparison of similarity heatmaps on the DocVQA dataset](assets/page_0015_img_1.png) | ![Qualitative comparison of similarity heatmaps on the DocVQA dataset](assets/page_0015_img_2.png) | ![Qualitative comparison of similarity heatmaps on the DocVQA dataset](assets/page_0015_img_3.png) |
| RegionRAG (Ours)                                           | RegionRAG (Ours)                                    | RegionRAG (Ours)                                                              |
| ![Qualitative comparison of similarity heatmaps on the DocVQA dataset](assets/page_0015_img_4.png) | ![Qualitative comparison of similarity heatmaps on the DocVQA dataset](assets/page_0015_img_5.png) | ![Qualitative comparison of similarity heatmaps on the DocVQA dataset](assets/page_0015_img_6.png) |
| ColQwen2.5                                               | ColQwen2.5                                        | ColQwen2.5                                                                    |

Figure S4: Qualitative comparison of similarity heatmaps on the DocVQA dataset. For various questions, our RegionRAG model generates significantly more focused and intense activation heatmaps on the ground-truth regions (indicated by red boxes) compared to the more diffuse heatmaps from the ColQwen2.5 baseline.

| Q: What is the percentage chance of infection when a person is not wearing an eye protection 3.1%, 16.0%, or 5.5%? | Q: What is the amount designated to start landing pads to connect Australian entrepreneurs to global innovation hubs? | Q: What percentage of online adults use Twitter as of September 2013? |
|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| ![Qualitative comparison of similarity heatmaps on the InfoVQA dataset](assets/page_0015_img_7.png)               | ![Qualitative comparison of similarity heatmaps on the InfoVQA dataset](assets/page_0015_img_8.png)            | ![Qualitative comparison of similarity heatmaps on the InfoVQA dataset](assets/page_0015_img_9.png)               |
| RegionRAG (Ours)                                                                                                   | RegionRAG (Ours)                                                                                              | RegionRAG (Ours)                                                    |
| ![Qualitative comparison of similarity heatmaps on the InfoVQA dataset](assets/page_0015_img_10.png)              | ![Qualitative comparison of similarity heatmaps on the InfoVQA dataset](assets/page_0015_img_11.png)          | ![Qualitative comparison of similarity heatmaps on the InfoVQA dataset](assets/page_0015_img_12.png)             |
| ColQwen2.5                                                                                                         | ColQwen2.5                                                                                                    | ColQwen2.5                                                        |

Figure S5: Qualitative comparison of similarity heatmaps on the InfoVQA dataset. Similar to its performance on DocVQA, RegionRAG demonstrates superior localization ability on infographics, generating sharp heatmaps that are precisely focused on small, answer-bearing regions (indicated by red boxes), while the baseline’s attention is more scattered.
```

### --- Page 0016 ---

```markdown
| Model  | Retriever        | ArxivQA | DocVQA | InfoVQA | PlotQA | SlideVQA | Avg.  |
|--------|------------------|---------|--------|---------|--------|----------|-------|
| ColPali| PaliGemma-3B    | 79.41   | 90.18  | 88.86   | 31.41  | 92.81    | 76.53 |
| RegionRAG | PaliGemma-3B | 82.72   | 95.26  | 95.54   | 85.05  | 97.12    | 91.14 |
|        | Qwen2.5-VL-3B    | 92.52   | 99.32  | 99.44   | 92.35  | 98.20    | 96.37 |

Table S8: Retrieval performance (Recall@10) comparison using different base models. RegionRAG maintains superior results over ColPali when both adopt the same PaliGemma-3B backbone.

as the backbone to assess the generalization of RegionRAG across different model architectures (Table S8 and Table S7).

### Retrieval Performance
To evaluate the generalization of RegionRAG across different architectures, we replace the Qwen2.5-VL-3B backbone used in the main experiments with PaliGemma-3B, the same base model adopted by ColPali. As shown in Table S8, RegionRAG still achieves significantly higher retrieval performance across all benchmarks, with an average Recall@10 improvement of 14.61 percentage points over ColPali. In particular, the gap is most pronounced on PlotQA, where RegionRAG surpasses ColPali by over 50 points (85.05 vs. 31.41). These results confirm that the advantages of our method do not stem from the backbone architecture, but rather from its region-level retrieval design and fine-grained contrastive learning strategy. This demonstrates that RegionRAG generalizes effectively across model architectures and tasks, reinforcing the robustness and versatility of our framework.

### Generation Performance
To further illustrate how the advantage of RegionRAG extends beyond retrieval, we evaluate its generation performance on InfoVQA using PaliGemma-3B as the backbone. As shown in Table S7, the model using retrieved bounding boxes achieves markedly higher accuracy (33.1% vs. 20.7%) compared to the full-image input under the top-1 QA setting. This result mirrors our earlier findings with Qwen2.5-VL, confirming that fine-grained, region-level inputs lead to more focused and efficient reasoning. Even with a different architecture, RegionRAG effectively preserves its performance advantage, demonstrating strong generalization of our region retrieval design across both model backbones and task stages.
```


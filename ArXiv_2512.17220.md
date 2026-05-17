# ArXiv 2512.17220

### --- Page 0001 ---

```markdown
# Mindscape-Aware Retrieval Augmented Generation for Improved Long Context Understanding

**Yuqing Li**¹², **Jiangnan Li**³, **Zheng Lin**¹², **Ziyan Zhou**¹²,  
**Junjie Wu**⁴, **Weiping Wang**¹, **Jie Zhou**¹, **Mo Yu**³†  
¹Institute of Information Engineering, Chinese Academy of Sciences  
²School of Cyber Security, University of Chinese Academy of Sciences  
³WeChat AI, Tencent ⁴Hong Kong University of Science and Technology  
{liyuqing, linzheng}@ie.ac.cn {jiangnanli, moyumy}@tencent.com  

## Abstract
Humans understand long and complex texts by relying on a holistic semantic representation of the content. This global view helps organize prior knowledge, interpret new information, and integrate evidence dispersed across a document, as revealed by the **Mindscape-Aware Capability** of humans in psychology. Current Retrieval-Augmented Generation (RAG) systems lack such guidance and therefore struggle with long-context tasks. In this paper, we propose Mindscape-Aware RAG (MiA-RAG), the first approach that equips LLM-based RAG systems with explicit global context awareness. MiA-RAG builds a mindscape through hierarchical summarization and conditions both retrieval and generation on this global semantic representation. This enables the retriever to form enriched query embeddings and the generator to reason over retrieved evidence within a coherent global context. We evaluate MiA-RAG across diverse long-context and bilingual benchmarks for evidence-based understanding and global sense-making. It consistently surpasses baselines, and further analysis shows that it aligns local details with a coherent global representation, enabling more human-like long-context retrieval and reasoning.

## 1 Introduction
Human thinking is inherently context-dependent. For any learned topic, familiar situation, or ongoing project of engagement, humans maintain a global semantic representation in memory. When the same topic reappears, this global memory is reactivated, endowing humans with the **Mindscape-Aware capability** to become aware of the approximate scope of their knowledge and to rely on this memory to interpret new inputs within context, selectively channel retrieval toward context-relevant knowledge, and guide subsequent reasoning accordingly. This phenomenon is grounded in theories from psychology (Bartlett, 1932; Tulving and Thomson, 1973; Reyna and Brainerd, 1995) and neuroscience (Ralph et al., 2017), which posit that when a topic is reactivated, conscious cognition is constrained and guided by globally integrated knowledge, with converging support from neuroimaging observations. We review both the theoretical and empirical supports in Appx. A.

### Retrieval-Augmented Generation (RAG)
(Zhu et al., 2025; Gao et al., 2023; Zhang et al., 2025a) has emerged as a critical strategy for long-context understanding by retrieving useful context fragments from very long inputs, thereby overcoming LLMs' limited context lengths (Lewis et al., 2020). However, current RAG systems primarily retrieve and generate based on local, evidence-level signals, lacking the mindscape-aware capability to activate a global semantic frame as humans do. Endowing RAG with this capability is therefore especially promising for personalized knowledge collections, such as long-context question answering (Bai et al., 2023), code generation (Wang et al., 2024b; 

![Average model ranks across five long-context benchmarks under 3/5/10-chunk settings.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
We evaluate MiA-RAG across a range of long-context understanding tasks spanning diverse domains and genres (e.g., government reports, narratives), in both English and Chinese. The evaluation also covers various task formats, including freeform QA, multiple-choice QA, and claim verification, as well as different RAG configurations such as vanilla RAG and GraphRAG (Edge et al., 2024). As summarized in Figure 1, the MiA family is more effective than baselines; in particular, MiA-RAG-14B achieves the best average rank, surpassing the vanilla 72B system and highlighting the benefit of mindscape-aware retrieval and generation.

Beyond performance gains, we analyze MiA-RAG’s internal mechanisms via embedding-space geometry and a new Mindscape-Coherent Evidence Alignment (MCEA) metric, showing that it internalizes global semantics: the mindscape reshapes query representations toward the global semantic space and acts as a scaffold that guides attention for Integrative Reasoning.

Our contributions are summarized as follows:
(1) We formulate the psycho- and neuro-inspired problem of mindscape-aware thinking, and present the first computational solution that equips LLMs with this capability.
(2) We conduct extensive experiments under diverse settings, demonstrating the necessity and effectiveness of integrating mindscape-aware capability into LLM-based systems.
(3) Our in-depth analysis reveals that the mindscape aligns the geometry of query representations with the global semantic space and serves as a semantic scaffold to guide attention, confirming the active internalization of global context rather than surface-level pattern matching.

## 2 Related Work

### Context-Aware Embeddings
Our MiA-Emb is related to the research topic of context-aware retrieval (or contextual retrieval (Anthropic, 2024)). This line of work mainly focuses on producing embeddings enriched with contextual information. A straightforward approach is to encode each chunk within a long-context window using LLMs that support extended inputs (Chen et al., 2024; Sturua et al., 2024; Nussbaum et al., 2024; Wang et al., 2024a; Lee et al., 2024; Li et al., 2023; Voyage-AI, 2025). To incorporate information beyond the local window, Xu et al. (2024) construct graphs of discourse relations and then utilize graph neighbors.
```


### --- Page 0003 ---

```markdown
# 3 Method

In this section, we introduce the Mindscape-Aware RAG (MiA-RAG) framework, designed to emulate the human cognitive process of leveraging global context for understanding, retrieval, and reasoning.

## 3.1 Preliminaries

Let $Doc$ be a long document that has been partitioned into chunks $c_i \in C$. In a standard RAG pipeline, a retriever selects a set of chunks $C_{ret} \subset C$ for a query $q$, and the generator conditions on these chunks to produce an answer $a$:

$$
q \xrightarrow{C_{ret}} a. \tag{1}
$$

However, this setup does not provide a global view of the document for either retrieval or generation. To bridge this gap, we propose MiA-RAG, which incorporates an explicitly global semantic scaffold termed the Mindscape $S$. By conditioning both retrieval and generation on $S$, MiA-RAG situates local evidence within a global context, improving retrieval accuracy and reasoning consistency:

$$
q, S \xrightarrow{C_{ret}} a. \tag{2}
$$

## 3.2 Hierarchical Mindscape Construction

The Mindscape $S$, which serves as the global memory of our framework, is constructed through a hierarchical bottom-up summarization process. We first prompt summarizer $M_s$ (GPT-40) with $[INST]_{sum,c}$ (Appx. F) to summarize each chunk:

$$
s_i = M_s([INST]_{sum,c}, c_i). \tag{3}
$$

After obtaining the chunk-level summaries, the sequence of $\{s_i\}$ is concatenated in order and further summarized using $[INST]_{sum,c}$ (Appx. F) to produce a single global representation:

$$
S = M_s([INST]_{sum,S}, [s_1, s_2, \ldots, s_n]). \tag{4}
$$

The resulting $S$ provides a coherent, document-level abstraction that constitutes the Mindscape.

## 3.3 MiA-Emb: Mindscape-Aware Retriever

To train the Mindscape-Aware Retriever (MiA-Emb), we construct supervision dataset and optimize the model via a multi-task objective.

### 3.3.1 Supervision Construction

Existing long-narrative understanding datasets such as NarrativeQA (Kočiský et al., 2018) typically provide only QA pairs and do not link questions to fine-grained supporting evidence. Such supervision is essential for training long-context retrievers, whether evidence is represented as text chunks in standard RAG or as graph nodes in GraphRAG.

Given the cost and inefficiency of manual annotation, we automatically extend NarrativeQA to curate $\bar{D}_{emb}$, a dataset offering silver-standard alignments at both chunk and node levels. For chunk evidence, we perform query augmentation, majority-vote ensemble retrieval, and LLM-based filtering to identify silver chunks (Algorithm 1). As shown in Table 1, an oracle experiment on 20 books validates these annotations: using only silver chunks ($\sim 1/30$ tokens) already exceeds full-context performance. Node evidence is constructed in an analogous way.

In total, $\bar{D}_{emb}$ comprises 27,117 questions, averaging 2.3 silver chunks and 2.9 nodes per question. Details of the construction are provided in Appx B.
```


### --- Page 0004 ---

```markdown
| Input                     | Avg. Chunks | EM    | F1    |
|---------------------------|-------------|-------|-------|
| Silver Chunks             | 2.3         | 31.29 | 52.91 |
| Oracle (Full Book)       | 69.2        | 30.04 | 50.79 |

Table 1: Validation of silver chunk annotations. ‘Avg. Chunks’ is the average number of chunks per query.

### 3.3.2 Model Optimization
We obtain MiA-Emb by fine-tuning a pre-trained embedding model so that global context is explicitly injected into query representations under our supervision signal.

**Mindscape-Conditioned Query Encoding**  
We encode each query $q_i$ by explicitly incorporating the summary $S$ together with task-specific control tokens $d$. The input sequence is defined as

$$
Q = [\texttt{[INST]emb}] \; q_i; \; S; \; d_q; \; d_n; \; d_c,
$$

where [INST]emb (Appx. F) is instruction prefix, $d_q$ marks the end of the query, and $d_n$ and $d_c$ activate node- and chunk-retrieval nodes, respectively.

**Residual Integration and Training Objective**  
To balance the original query intent with global guidance, we use a residual integration mechanism to form task-specific enriched query representations and train MiA-Emb with a contrastive objective (van den Oord et al., 2018). Full details are provided in Appx. B.3.

### 3.4 MiA-Gen: Mindscape-Aware Generator
**Training Data Construction**  
We obtain the mindscape-aware generator (MiA-Gen) using a supervised fine-tuning corpus built from two datasets: NarrativeQA for long-form question answering and CLIPPER (Pham et al., 2025), a synthetic dataset of narrative true/false claims with chain-of-thought rationales for claim verification. Each example is formatted as:

$$
[\texttt{[INST]gen} \; (\texttt{Appx. F})] \; S; \; \hat{C}^r; \; q_i \; \rightarrow \; y_{gen}^{-1},
$$

To simulate realistic retrieval conditions, $\hat{C}^r$ is generated by mixing silver chunks with irrelevant chunks. The resulting contexts contain both relevant and noisy evidence with varied lengths, mirroring real retrieval behavior. For the Clipper dataset, we directly use the retrieval results produced by our MiA-Emb model to obtain $\hat{C}^r$. Combining these elements yields the final supervised fine-tuning dataset $\hat{D}_{gen}$ for MiA-Gen.

**Optimization**  
MiA-Gen is optimized over $\hat{D}_{gen}$ using the autoregressive cross-entropy loss:

$$
\mathcal{L}_{\text{MiA-Gen}} = -\sum_{t=1}^{|y|} \log p_\theta(y_t | y_{<t}, z_{gen}).
$$

### 4 Experimental Setting
**Evaluated Models**  
MiA-RAG is implemented by fine-tuning Qwen-series (Zhang et al., 2025b; Team, 2024) large language models. We develop two core components:
- **Mindscape-Aware Retriever**: we fine-tune Qwen2.5-Embedding-8B to obtain the MiA-Emb.
- **Mindscape-Aware Generator**: we fully fine-tune Qwen2.5-14B-Instruct to obtain MiA-Gen.

We have released our models on Hugging Face. Implementation details are provided in Appx C.

**Public Long Narrative Understanding Tasks**  
We evaluate our method on a diverse set of long narrative understanding benchmarks, many of which contain contexts beyond standard LLM input limits (128K). Since our model is trained on the NarrativeQA (Kočiský et al., 2018) training set, we first perform in-domain evaluation on held-out books to assess narrative comprehension. To examine cross-domain generalization, we further evaluate on EN.MC subset of oBench (Zhang et al., 2024) for multiple-choice reasoning, DetectiveQA (Xu et al., 2025) for bilingual long-text reasoning, and the public subset of Nocha (Karpinska et al., 2024) for claim verification. Dataset statistics and metrics are provided in Table 3.

### 5 Experiments
#### 5.1 Study I: Retrieval Results
Table 4 summarizes retrieval performance. MiA-Emb consistently outperforms all baselines across the benchmarks, and even surpasses Sit-Emb (Wu et al., 2025), a state-of-the-art model specialized for story understanding. Further comparisons with other embedding models are given in Appx. D.4.

On the in-domain NarrativeQA, MiA-Emb exhibits substantial gains, validating its ability to leverage global context for precise evidence localization. This advantage also transfers to the out-of-domain, bilingual DetectiveQA benchmark, where MiA-Emb remains clearly superior in Answer Recall.

Finally, our ablation study shows that removing the summary (w/o Summary) leads to substantial
```

### --- Page 0005 ---

```markdown
| Model                | Retriever | Generator | NarrativeQA | 0-Bench | Det-QA-2 | Det-QA-3 | NoCha | Avg  |
|----------------------|-----------|-----------|-------------|---------|----------|----------|-------|------|
| Summary-Only         | Ours-3B   | Ours-3B   | 39.24       | 7.26    | 73.67    | 61.33    | 31.75 | 56.61 |
| Vanilla              | Ours-3B   | Ours-3B   | 5.728       | 41.1345 | 55.2059  | 6.3770   | 50.31 | 33.36 |
| MiA (Ex-Only)       | Ours-3B   | Ours-3B   | 5.728       | 47.7464  | 61.9161  | 51.3078  | 51.35 | 44.74 |
| MiA (Em-Only)       | Ours-3B   | Ours-3B   | 5.728       | 42.4656  | 60.8953  | 52.1277  | 61.77 | 53.42 |
| MiA                 | MiA-Emb-8B | Ours-3B   | 5.728       | 50.8513  | 61.8717  | 61.3795  | 32.14 | 49.81 |
| Summary-Only         | Ours-3B   | Ours-3B   | 39.24       | 7.26    | 73.67    | 61.33    | 31.75 | 56.61 |

### Table 2: Overall results of the MiA framework on Long-story QA and Reasoning benchmarks. Each model is evaluated under 3/5/10 retrieval settings. “+” indicates whether the Mindscape summary is incorporated at that stage. Dark gray rows represent deeper misdance involvement, while bold numbers indicate the best results within each scale group. The final column reports the average over all metrics per row.

| Dataset         | Queries | Avg. Tokens | Metrics         |
|------------------|---------|-------------|------------------|
| NarrativeQA      | 556     | 83k         | F1, EM, Recall    |
| on-Bench-EN.MC   | 229     | 118k        | Accuracy          |
| DetectiveQA      | 1,200   | 139k        | Pairwise Acc.     |

### Table 3: Summary of the evaluation datasets.

| Method            | NarrativeQA | DetectedQA-Z11 | DetectedQA-EN |
|-------------------|-------------|----------------|----------------|
| MiA-Emb-8B        | 82.83       | 79.89          | 71.79          |
| MiA-Sum           | 58.62       | 67.10          | 65.60          |
| MiA-Gen-8B       | 70.07       | 62.58          | 74.67          |

### Table 4: Retrieval performance comparison across models, evaluated using Recall@K (%).

| Method            | NarrativeQA | 0-Bench | DetectedQA-EN | NoCha |
|-------------------|-------------|---------|----------------|-------|
| MiA-Gen-14B       | 53.19       | 82.39   | 76.03          | 52.91 |
| w/o Summary       | 50.49       | 75.39   | 71.01          | 44.97 |
| w/o Claim         | 51.22       | 88.44   | 75.58          | 44.44 |
| w/o QA            | 46.40       | 81.08   | 72.21          | 46.56 |

### Table 5: Ablation study of MiA-Gen-14B. Reported scores are averaged over the 3/5/10-chunk settings.

Mindscape-Aware Retrieval Consistently Improves Performance As verified in Sec. 5.1, MiA-Emb substantially enhances retrieval quality. When integrated into the full pipeline, substituting the vanilla retriever with MiA-Emb yields consistent gains. MiA-Emb improves the average scores of the 72B and 14B generators by 6.95% and 7.55%, respectively, confirming that globally informed queries yield more effective retrieval.

Integrative Reasoning Benefits from Mindscape-Conditioned Generation Simply supplying the summary to a vanilla generator yields a consistent +3.79% improvement, showing that global contextual cues provide useful guidance. A larger gain is observed when the generator is fine-tuned under the same mindscape-conditioning paradigm as the retriever. Under identical inputs, our MiA-Gen-14B achieves a substantially larger +11.16% gain. This disparity suggests that MiA-Gen more effectively integrates retrieved chunks with the global semantics that guided their selection.

5.2.2 Ablation Study We perform ablations to assess each MiA-RAG component, with results shown in Tables 4 and 5. Impact of Mindscape-Conditioning Same as in the embedding stage (Table 4), removing the summary (w/o Summary) leads to substantial degradation in the generation stage (Table 5). These
```

### --- Page 0006 ---

```markdown
![Impact of retriever scale on average results across 5 benchmarks (DetectiveQA-ZH/EN, ocBench, NoCha, NarrativeQA) with a Qwen2.5-72B generator.](assets/page_0006_img_1.png)

![Impact of generator model scale on average results over 5 benchmarks, with a Mia-Emb-BB retriever.](assets/page_0006_img_2.png)

| Summary Generator       | Recalled 5/10 (%) | F1-Score (%) |
|-------------------------|--------------------|---------------|
| GPT-4 (Ours)            | 62.6875/92.8289     | 52.8435/52.563.5 |
| Qwen2.5-32B-Instruct    | 61.6746/64.6085     | 50.6015/50.937.5 |
| Qwen2.5-14B-Instruct    | 79.7374/54.8786     | 51.4515/51.812.6 |
| Qwen2.5-2B-Instruct     | 58.6272/61.8167     | 47.7915/51.48 |

declines indicate that summary-based supervision helps align queries with global semantics and supports the integration of dispersed evidence. 

**Benefit of Multi-Paradigm Supervision**  
Ablating either supervision paradigm (w/o Claim, for claim verification or w/o QA for question answering) consistently degrades performance, indicating that exposure to diverse reasoning patterns improves generalization beyond any single task.

**5.3 Study III: MiA-GraphRAG for Global QA**  
We further evaluate MiA-RAG for global sensemaking in a GraphRAG QA setting, where it retrieves relevant graph nodes (entities) for holistic document understanding and achieves clear gains over baselines. Details are reported in Appx. D.2.

**5.4 Study IV: Impact of Model Scales**  
We evaluate the scalability of MiA-Emb across backbone sizes (0.6B to 8B) against SFT-Emb (identical to the w/o Summary ablation in Sec. 5.1, i.e., trained and evaluated without summaries) and Vanilla1 Qwen3.5-Embedding baselines. As shown in Figures 2 and 8, MiA-Emb consistently outperforms both baselines; notably, MiA-Emb-0.6B already surpasses the Vanilla 8B model in both retrieval recall and downstream QA and reasoning performance. We also scale MiA-Gen across model sizes and observe consistent gains over the vanilla Qwen2.5-Instruct models (1.5B>72B), presented in Figure 3. In particular, MiA-Gen-14B matches or even exceeds the 72B model.

Collectively, these results indicate that incorporating global semantics is more effective for long-context understanding than merely scaling model size. Numerical results are provided in Appx. D.3.

**5.5 Study V: Impact of the Quality of Summaries**  
We show that our MiA-RAG is robust to the quality of summaries. Specifically, we replace the 

**6 Analysis**  
In this section, we introduce analytical methods to evaluate whether the resulting MiA-RAG exhibits the three hypothesized capabilities proposed in the introduction, namely Enriched Understanding, Selective Retrieval, and Integrative Reasoning.

**6.1 The Role of Global Summaries**  
While the ablation confirms that MiA-RAG benefits from incorporating summaries into the inputs, a critical question remains regarding their functional role during inference.

We first show that summaries are not useful simply because they cover the answer. To illustrate this, we evaluate a Summary-Only variant in which the generator predicts the answer using only the summary (Table 2). This model consistently underperforms the vanilla-RAG and falls short of the MiA-RAG results. These findings indicate that the summaries function not as standalone evidence but as semantic scaffolds: they enhance retrieval and reasoning by guiding the model to operate within a
```

### --- Page 0007 ---

```markdown
![Comparison of projection angles for MiA-Emb and Qwen3-Emb. Lower angles indicate better alignment of queries with the book’s semantic subspace.](assets/page_0007_img_1.png)

![Layer-wise comparison of silver-chunk retrieval accuracy and attention allocation proportion.](assets/page_0007_img_2.png)

We verify the hypothesis in two folds. First, we examine whether performance gains from MiA-Emb correlate with increased use of the global summary. If so, we then study whether the model focuses its attention on information that can enrich the queries in these situations.

## 6.2 Geometric Properties of the MiA Embedding Space

Extending the analysis in Sec. 6.1, we further examine the following hypothesis:

**(H1)** MiA-Emb facilitates Selective Retrieval.

That is, whether the embedding model biases query representations toward the active book topic, thereby better positioning them within the subspace supported by the corresponding chunks.

**Method** We visualize query and chunk embeddings with t-SNE (Maaten and Hinton, 2008). To characterize the semantic structure of the document, we first fit t-SNE on the chunk embeddings only, yielding a 2D manifold that reflects the document’s semantic structure. We then embed the query representations into the same 2D space and inspect how well each model positions queries relative to the corresponding topic-relevant regions.

**Results** Figure 4 shows a clear geometric distinction between MiA-Emb and the vanilla embedding model. Across books, MiA-Emb consistently yields smaller projection angles, meaning that query embeddings lie closer to the semantic subspaces spanned by their corresponding documents. On average, MiA-Emb-8B achieves 37.1°, compared with 43.5° for Qwen-Emb-8B, demonstrating that mindspace conditioning more effectively guides queries toward the correct semantic region and enables more precise selective retrieval.

## 6.3 Residual Stream and Attention Analysis of the MiA Embedding Model

In this and the following subsections, we examine the following hypothesis:

**(H2)** MiA-Emb facilitates Enriched Understanding of queries.

**Method** To understand how summary information enriches query representations, we inspect the following:

**(H2.1)** MiA-Emb puts increased attention to the global summary at layers of improved predictability compared to the baseline.

Following the approach of (Jiang et al., 2024), we compare MiA-Emb with the vanilla embedding model through their residual streams to analyze how retrieval-relevant information is progressively accumulated into the query representation. To ensure comparability, we select 100 queries for which both models achieve Recall@10 = 100%. Concretely, we track the layer-wise Top-10 silver-chunk ratio for both models, reflecting how the hidden states at each layer steer the retrieval distribution toward the correct evidence. For MiA-Emb, we additionally examine the attention from the last token to summary tokens and to query tokens, enabling us to assess whether improvements coincide with increased use of global-summary cues.

**Results** As shown in Figure 5, MiA-Emb exhibits a clear rise in silver-chunk recall beginning at the middle layers. This rise coincides with increased attention to the global summary in the same layer range, suggesting that the model progressively injects summary-derived cues into the query representation. This incorporation of global signals enriches the query embedding, enabling MiA-Emb to develop a deeper semantic understanding of the query and thus support more selective retrieval.

**(H2.2)** MiA-Emb attends to information that enriches the query at the layers identified in (H2.1).
```

### --- Page 0008 ---

```markdown
## 6.4 Attention Pattern Analysis in the Generation Model

**(H3)** MiA-Gen facilitates Integrative Reasoning over retrieved chunks within the global mindscape.

To examine how the mindscape steers generation toward relevant evidence, we introduce the Mindscape-Coherent Evidence Alignment (MCEA) metric. It quantifies global–local coherence in attention by measuring whether chunks that are more consistent with the summary receive stronger query attention, and whether this effect is stronger for relevant chunks than for irrelevant ones. Higher MCEA indicates stronger mindscape-driven Integrative Reasoning. A formal definition is given in Appx. E.

### Method

We compute MCEA for MiA-Gen and the vanilla Qwen2.5-14B-Instruct at each transformer layer. To verify that the observed alignment is driven by global semantics rather than positional or length biases, we also introduce a summary-replaced control, where the original summary is replaced with an unrelated text of the same length.

### Results

Figure 7 highlights two key findings. First, MiA-Gen exhibits consistently higher MCEA values than the vanilla model, with the gap expanding notably in the middle and late layers. This pattern reflects a structured reasoning process: local chunks first internalize global mindscape semantics, after which the query increasingly attends to these enriched chunks. Second, replacing the valid summary with irrelevant text causes a sharp drop in MiA-Gen’s MCEA, whereas the vanilla model exhibits negligible sensitivity to this perturbation. This contrast confirms that MiA-Gen’s alignment is driven by genuine mindscape semantics rather than positional or length-based cues.

Collectively, these results demonstrate that MiA-Gen performs Integrative Reasoning: the model leverages the global mindscape to structure local evidence interpretation and subsequently guides query attention toward globally coherent chunks.

## 7 Conclusion

Inspired by the human cognitive ability to interpret new inputs within a global “mindscape”, we propose MiA-RAG, the first framework that equips LLM-based RAG systems with mindscape-aware capabilities. We approximate this global impression via a hierarchically generated summary that serves as a persistent global memory. By conditioning both the retrieval and generation stages on this summary, MiA-RAG achieves superior performance in evidence-based long-context understanding as well as global sense-making tasks. Empirical analysis further shows that the summary projects queries into a global semantic space, enabling enriched understanding, selective retrieval, and integrative reasoning over dispersed evidence.

![Attention pattern of MiA-Emb: the last token attempts to predicting summary tokens, with red regions indicating tokens that receive high attention.](assets/page_0008_img_1.png)

![Layer-wise Mindscape-Coherent Evidence Alignment (MCEA) scores for generator.](assets/page_0008_img_2.png)
```

### --- Page 0009 ---

```markdown
## Limitations

While MiA-RAG demonstrates strong performance on narrative long-context QA and reasoning, the framework relies on a precomputed global summary as the mindscape representation. This requirement may limit applicability in scenarios where the underlying content evolves over time or where summaries are difficult to obtain. Moreover, our experiments focus primarily on narrative-style datasets, and the generality of the approach for other long-context settings (e.g., long-form dialogue) remains to be validated. Finally, part of the supervision signal is derived from commercial LLMs, which may introduce latent biases or hallucinated content. Nonetheless, the empirical gains suggest that the mindscape-aware training strategy remains robust even under imperfect supervision.

## References

Anthropic. 2024. Introducing contextual retrieval.

Sam Audrain and Mary Pat McAndrews. 2022. Schemas provide a scaffold for neocortical integration of new memories over time. Nature communications, 13(1):5795.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, and 1 others. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

Frederic Charles Bartlett. 1932. Remembering: A study in experimental and social psychology. Cambridge university press.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. 2024. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663.

Jeffrey R Binder, Ruvith H Desai, William V Graves, and Lisa L Conant. 2009. Where is the semantic system? a critical review and meta-analysis of 120 functional neuroimaging studies. Cerebral cortex, 19(12):2767–2796.

Garvin Brod, Ulman Lindenberger, and Yee Lee Shing. 2017. Neural activation patterns during retrieval of schema-related memories: Differences and commonalities between children and adults. Developmental science, 20(6):e12475.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. Preprint, arXiv:2402.03216.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Morley, Steven Truitt, Dasha Metropolitansky, Robert Oszvath Nagy, and Jonathan Larson. 2024. From local to global: A graph rap approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10907, 2(1).

Samuel J Gershman, Anna C Schapiro, Amul Hupbach, and Kenneth A Norman. 2013. Neural context reinstatement predicts memory attribution. Journal of Neuroscience, 33(20):8590–8595.

Asaf Gilboa and Hannah Marlatte. 2017. Neurobiology of schemas and schema-mediated memory. Trends in cognitive sciences, 21(8):618–631.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Che Jiang, Biqing Qi, Xiangyu Hong, Dayuan Fu, Yang Cheng, Fandong Meng, Mo Yu, Bowen Zhou, and Jie Zhou. 2024. On large language models’ hallucination with regard to known facts. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1041–1053.

Marzena Karpinska, Katherine Thia, Kyle Lo, Tanya Goyal, and Mohit Yadav. 2024. One thousand and one pairs: A "novel" challenge for long-context language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024.

Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The narrative reading comprehension challenge. Transactions of the Association for Computational Linguistics, 6:317–328.

James E Kragel, Youssef Ezzat, Bradley C Lega, Michael R Sperling, Gregory A Worrell, Robert E Gross, Barbara C Jobst, Sameer A Heshmat, Kareem A Zaghloul, Joel M Stein, and 1 others. 2021. Distinct cortical systems reinstate the content and context of episodic memories. Nature Communications, 12(1):4444.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. Nv-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428.
```

### --- Page 0010 ---

```markdown
| Author(s)                                                                 | Year  | Title                                                                                                   | Source                                                                                          |
|---------------------------------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpuinkin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschl, and others | 2020  | Retrieval-augmented generation for knowledge-intensive nlp tasks.                                      | Advances in neural information processing systems, 33:9459–9474.                               |
| Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venktesh, Acyr Locatelli, Huanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen | 2024  | Snapvk: Llm knows what you are looking for before generation.                                         | Advances in Neural Information Processing Systems, 37:22947–22970.                             |
| Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang | 2023  | Towards general text embeddings with multi-stage contrastive learning.                                 | arXiv preprint arXiv:2308.03281.                                                               |
| Laurens van der Maaten and Geoffrey Hinton | 2008  | Visualizing data using t-sne.                                                                          | Journal of machine learning research, 9(Nov):2579–2605.                                       |
| Raiza Martin and Steven Johnson. | 2023  | Introducing notebooklm.                                                                                 |                                                                                                 |
| Jing Miao, Charat Thongprapong, Supawadee Supadungskor, Oscar A Garcia Valencia, and Wisit Cheungspoon | 2024  | Integrating retrieval-augmented generation with large language models in nephrology: advancing practical applications. | Medicina, 60(3):445.                                                                            |
| Zach Nussbaum, John X Morris, Brandon Duderstadt, and Andriy Mulyar | 2024  | Nomic embed: Training a reproducible long context text embedder.                                       | arXiv preprint arXiv:2402.01613.                                                               |
| Chau Minh Pham, Yapei Chang, and Moht Iyer | 2025  | Clipper: Compression enables long-context synthetic data generation.                                    | arXiv preprint arXiv:2502.14854.                                                               |
| Hongjin Qian, Zhen Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zicheng Dou, and Tiejun Huang | 2025  | Memory: Boosting long context processing with global memory-enhanced retrieval augmentation.            | Proceedings of the ACM on Web Conference 2025, pages 2366–2377.                               |
| Matthew A Lambon Ralph, Elizabeth Jefferies, Karalyn Patterson, and Timothy T Rogers | 2017  | The neural and computational bases of semantic cognition.                                              | Nature reviews neuroscience, 18(1):42–55.                                                      |
| Valerie F Reyna and Charles J Brainerd | 1995  | Fuzzy-trace theory: An interim synthesis.                                                               | Learning and individual differences, 7(1):1–75.                                               |
| Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas, Andreas Koukounas, Nan Wang, and Han Xiao | 2024  | jina-embeddings-v3: Multilingual embeddings with task lore.                                           | Preprint, arXiv:2409.10173.                                                                    |
| Qwen Team. | 2024  | Qwen2.5: A party of foundation models.                                                                    |                                                                                                 |
| Endel Tulding and Donald M Thomson. | 1973  | Encoding specificity and retrieval processes in episodic memory.                                        | Psychological review, 80(5):352.                                                                |
| Aïron van den Oord, Yazhe Li, and O. Vinyals. | 2018  | Representation learning with contrastive predictive coding.                                              | arXiv, abs/1807.03748.                                                                          |
| Voyage-AL. | 2025  | Introducing voyage-context-3: focused chunk-level details with global document context.                 | Blog post.                                                                                      |
| Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. | 2024  | Improving text embeddings with large language models.                                                  | In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11897–11916. |
| Zora Zhirou Wang, Akari Asai, Xinyan Velocity Yu, Frank F Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. | 2024b | Codera-bench: Can retrieval augment code generation?                                                   | arXiv preprint arXiv:2406.14497.                                                                |
| Junjie Wu, Jiangnan Li, Yuqing Li, Lemao Liu, Liyan Xu, Jiwei Li, Dit-Yan Yeung, Jie Zhou, and Mo Yu. | 2025  | Stembi-v1: 5: Improved context-aware dense retrieval for semantic association and long story comprehension. | arXiv preprint arXiv:2508.01959.                                                                |
| Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. | 2023  | Efficient streaming language models with attention skins.                                               | arXiv preprint arXiv:2309.17453.                                                                |
| Liyan Xu, Jiangnan Li, Mo Yu, and Jie Zhou. | 2025  | Fine-grained modeling of narrative context: A cohere perspective vs retrospective questions.             | In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5822–5838. |
| Yangwen Xu, Qixiang Lin, Zaihua Han, Yong He, and Yancha Ba. | 2016  | Intrinsic functional network architecture of human semantic processing: Modules and hubs.               | Neuroimage, 132:542–555.                                                                        |
| Zhe Xu, Jiasheng Ye, Xiaoran Liu, Xiangyang Liu, Tianxiang Sun, Zhiqing Liu, Qiceng Guo, Linlin Li, Qun Liu, Xuanjing Huang, and Xipeng Qiu. | 2025  | DetectiveQA: Evaluating long-context reasoning on detective novels.                                    | In Workshop on Reasoning and Planning for Large Language Models.                                 |
| Dongjie Yang, XiaoDong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. | 2024  | Pyramidfinder: Pyramidal kv cache compression for high-throughput lm inference.                         | arXiv preprint arXiv:2405.12532.                                                                |
| Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. | 2024  | Helmet: How to evaluate long-context language models effectively and thoroughly.                        | arXiv preprint arXiv:2410.02694.                                                                |
```

### --- Page 0011 ---

```markdown
# Supports of Mindscape-Aware Capabilities in Broader Research Fields

We show that the existence and our discussed advantages of the mindscape-aware capability are supported by various research on human memory in psychological and neuroscience research.

## Supports in Psychology

First, the existence of mindscape-aware capability traces back to the concept of schema (Bartlett, 1932) and aligns closely with the principles of Fuzzy-Trace Theory (FTT; Reyna and Brainerd 1995). Specifically, the schema serves as an integrated structure of a given topic. When activated, it guides attention and limits interpretive possibilities during information processing. The FTT theory posits that human memory encodes experiences at two complementary levels: verbatim traces that preserve surface details and gist traces that capture the abstract, meaning-based structure of knowledge. When individuals encounter new information related to a familiar topic, it is typically the gist-level representation that is reactivated, providing a global semantic scaffold that constrains interpretation, retrieval, and reasoning. Our mindscape-aware framework is a computational implementation of gist-based cognition and an approximation of the abstract information in schema, in the context of complex reasoning and retrieval-augmented systems.

Second, our discussed advantages of the mindscape-aware capability are grounded in the Encoding Specificity Principle (Tulving and Thomson, 1973), which lies in the core of psychological research on memory. The principle states that when a familiar topic or task reappears, the reinstatement of its original contextual pattern reactivates the corresponding memory network, thereby enhancing retrieval effectiveness and interpretive coherence.

## Supports in Neuroscience

The theory of MiA capabilities has a solid empirical basis in neuroscience research. First and most importantly, the existence of mindscape-aware abilities is directly characterized by the controlled semantic cognition (CSC) framework (Ralph et al., 2017), which posits that conscious cognition is constrained and guided by globally integrated semantic knowledge, ensuring that thought unfolds within a coherent knowledge framework.

Second, the aforementioned psychological theories and the CSC framework have been verified by neuroimaging evidence, providing empirical support for mindscape-aware abilities. For example, Brod et al. (2017); Gilboa and Marlate (2017); Audrain and McAndrews (2022) identify the neural foundations of schemas, which facilitate the integration of new knowledge and shape memory recall. Further empirical findings (Gershman et al., 2013; Kragel et al., 2021) demonstrate that context reinstatement during retrieval, i.e., the reactivation of semantic, situational, or cue-related features present during encoding — enhances memory recall. Similarly, the CSC framework is supported by evidence of sustained co-activation during story comprehension and related semantic processes (Binder et al., 2009; Xu et al., 2016). In this sense, mindscape-awareness can be viewed as a higher-order manifestation of encoding specificity and schema mechanisms, involving the reactivation of a global semantic “mindscape” that encompasses an individual’s accumulated knowledge and guides interpretation and retrieval.
```


### --- Page 0012 ---

```markdown
# Algorithm 1 Silver Evidence Annotation

1. Input: Dataset $D = \{(q_i, a_i) \}_{i=1}^N$, mindsape summary $S$, retrieve $E_s$, task $t \in \{c, n\}$.
2. Define evidence unit $U = C$ (if $t = c$) or $U = V$ (if $t = n$).
3. Output: Silver-annotated dataset $\hat{D}_{emb} = \{(q_i, \hat{U}_i) \}_{i=1}^N$.
4. Initialize $\hat{D}_{emb} = \emptyset$
5. for each $(q_i, a_i) \in D$ do
6. \quad $q_{emb} = (q_i, q_i + a_i, a_i)$ \> Query Augmentation
7. \quad $U_{pool} \gets \text{Candidate Retrieval \& Ensemble}$
8. \quad $V_{pool} \gets \emptyset$
9. \quad for $j \in \text{query do}$
10. \quad \quad $U_{pool} \gets \text{VoteAndSelectTopK}(E_s', q', U, k)$
11. \quad end for
12. \quad $U_{cand} \gets \text{LLM-driven Refinement}$
13. \quad $\hat{U}_i \gets \text{LLM}(q_i, a_i, U_{cand}) \> \text{See Fig. 11/12}$
14. \quad Add $(q_i, \hat{U}_i) \to \hat{D}_{emb}$
15. end for
16. return $\hat{D}_{emb}$

## B MiA-Emb: Supervision and Training Objective

### B.1 Positive Evidence Construction

As existing long-context benchmarks lack explicit query–evidence alignments, we automatically construct silver evidence for both chunk- and node-level retrieval.

#### Silver Chunk Annotation
We annotate silver chunks using a structured procedure that integrates query augmentation, ensemble retrieval, and LLM-based refinement (Algorithm 1). For chunk-level supervision, we set the task $t = c$ in Algorithm 1, yielding the silver chunk dataset $\hat{D}_{emb} = \{(q_i, \hat{C}_i) \}_{i=1}^N$, where $\hat{C}_i \subset C$ denotes the set of supporting chunks for query $q_i$.

#### Silver Node Annotation
To support retrieval at a global semantic granularity, we construct a knowledge graph $G = (V, E)$ by extracting entity-level information, following a procedure similar to GraphRAG [Edge et al., 2024]. For each document, we employ GPT-2 to identify key entities and generate concise descriptions, yielding a node set $V = \{(name, desc)\}$.

We then generate the node-level silver dataset $\hat{D}_{emb} = \{(q_i, \hat{V}_i) \}_{i=1}^N$ by setting the task $t = n$ and evidence units $U = V$ in Algorithm 1. Here, $\hat{V}_i \subset V$ represents the set of relevant nodes for query $q_i$, serving as the ground truth for the node retrieval task.

### B.2 Negative Evidence Construction

MiA-Emb is trained with a contrastive objective that requires both positive and negative samples. Positive samples are taken from the silver evidence sets $\hat{C}_i$ and $\hat{V}_i$ described above, while negative samples are constructed from two complementary sources. We illustrate the construction for chunk retrieval; node retrieval follows the same design.

#### Hard negatives
Hard negatives are semantically similar to the query but not included in the silver evidence. We select chunks from the candidate set $C_{cand}$ (Algorithm 1) that are not part of the silver set $\hat{C}_i$ and take up to 5 such chunks to form the hard-negative set $C_{hard}$. These samples provide challenging contrasts that encourage the model to distinguish subtle semantic differences.

#### Simple negatives
Simple negatives are clearly irrelevant to the query. We sample them uniformly at random from the full document chunk set $C$, ensuring no overlap with the positive set $\hat{C}_i$ or the hard negatives $C_{hard}$. We sample 5 chunks to form the simple-negative set $C_{simple}$.

#### Final negative set
For chunk retrieval, the final negative pool for query $q_i$ is 

$$
C_{i}^r = C_{hard}^i \cup C_{simple}^i. \tag{8}
$$

For node retrieval, we apply the same procedure to obtain the node-level negative set $\hat{V}_i$: we use $\hat{U}_i^*$ as a unified notation for the negative set of query $q_i$.

### B.3 Model Training

We provide additional training details for MiA-Emb, complementing Sec. 3.3.2.

#### Input Representation
To enable the embedding model to perceive both the local query intent and the global mindsape, we construct a composite input sequence. Let $q_i$ be the query and $S$ be the mindsape summary. The input is formatted as 

$$
Q = [[\text{INST}_{emb}; q_i; d_i; S; d_n; d_c]], \tag{9}
$$

where $[\text{INST}_{emb}]$ is the instruction prefix, $d_i$ marks the end of the query, and $d_n, d_c$ serve as special tokens representing node- and chunk-retrieval tasks, respectively.

The sequence is encoded by the embedding model $\mathcal{E}$ to obtain token-level hidden states:

$$
H = \mathcal{E}(Q) = (h_1, \ldots, h|Q|), \tag{10}
$$

where $H$ denotes the last-layer hidden states for all tokens in $Q$.
```


### --- Page 0013 ---

```markdown
## Residual Integration 

To preserve the original query semantics while injecting global context, we employ a residual connection strategy. We extract the hidden state at the query delimiter ($h_q$, corresponding to token $d_q$) and the hidden state at the task delimiter ($h_r$, corresponding to $d_c$ or $d_n$, depending on the active task $t$). The final enriched query representation $\tilde{q}$ is computed as

$$
\tilde{q} = \delta \cdot h_q + (1 - \delta) \cdot h_t, \tag{11}
$$

where $\delta$ is a hyperparameter controlling the balance between local query focus and global context awareness. A detailed ablation on the role of this residual connection is provided in Appendix D.5.

## Joint Contrastive Optimization 

Finally, we optimize a multi-task contrastive objective (van Oord et al., 2018) over chunk and node retrieval:

$$
L_{MiA-Emb} = \beta \cdot L_c + (1 - \beta) \cdot L_n, \tag{12}
$$

where $L_c$ and $L_n$ represent the losses for chunk and node retrieval, respectively, and $\beta \in [0, 1]$ balances their contribution.

Both tasks employ the InfoNCE loss. Specifically, the objective $L_t$ ($t \in \{c,n\}$) is defined as:

$$
L_t = -\frac{1}{|B|} \sum_{j=1}^{|B|} \log \frac{\exp(sim(\tilde{q}_t, d_j^+)/\tau)}{\sum_{d \in \mathcal{D}} \exp(sim(\tilde{q}_t, d)/\tau)}, \tag{13}
$$

where $|B|$ is the batch size, $\tau$ is the temperature parameter, and $sim(\cdot, \cdot)$ denotes cosine similarity.

The candidate set for the $j$-th query is constructed as:

$$
C_j = \{d_j^+\} \cup U_j^-, \tag{14}
$$

where $d_j^+$ is the positive embedding sampled from the silver evidence set $\hat{U}_j$, and $U_j^-$ is the corresponding set of negative candidates.

## Implementation Details 

We set the chunk size to 1200 with an overlap of 100 tokens for NarrativeQA, DetectiveQA, and $\infty$Bench, and to 200 for NoCha, following the typical context length distributions of these datasets. We build our retriever MiA-Emb by applying LoRA (Hu et al., 2022) on top of Qwen3-Embedding-8B, and build our generator MiA-Gen by fully fine-tuning Qwen2.5-14B-Instruct. Throughout the paper, we denote Qwen2.5-72B as the 4-bit quantized variant of Qwen2.5-72B-Instruct to improve efficiency. GPT-4o refers to GPT-4o-2411. For silver evidence construction, we use Gte-Qwen-7B as the retriever $E_g$ in Algorithm 1. All experiments are conducted on 8 × H20 GPUs, each equipped with 96GB of memory. All hyperparameters are summarized in Table 7.

| Setting         | MiA-Emb | MiA-Gen |
|------------------|---------|---------|
| Precision        | 1.0     | 1.0     |
| Batch Size       | 4       | 2       |
| Steps            | 2000    | 2000    |
| warmup ratio     | 0.1     | 0.05    |
| Learning Rate    | 1 × 10^{-4} | 1 × 10^{-5} |
| LoRA Rank        | 128     | -       |
| LoRA $\alpha$    | 256     | -       |
| Temperature $\tau$ | 0.01  | 0.0     |
| Residual Weight $\beta$ | 0.5 | -     |
| Multi-task Weight $\beta$ | - | -     |

Table 7: Training configurations. ‘-’ denotes not used.

## Additional Experiments 

### D.1 Results on Helmet 

To further examine the robustness of MiA-RAG, we evaluate our system on the NarrativeQA subset used in the Helmet benchmark. This setting is particularly challenging due to long contexts. Table 8 compares different combinations of retrievers and generators, with darker rows indicating stronger utilization of the global summary. We observe three main trends. First, replacing the vanilla retriever with MiA-Emb consistently improves both EM and F1, even when paired with off-the-shelf generators. Second, adding the Mindscope summary during inference benefits all RAG configurations, especially when retrieval quality is already high. Finally, the integration of MiA-Emb and MiA-Gen into the full MiA-RAG model delivers the strongest results, markedly surpassing all baselines while requiring substantially shorter effective context lengths.

### D.2 Study III: MiA-GraphRAG for Global QA 

#### Global Sense-Making QA Task 

Beyond local evidence-oriented evaluation, we assess global sense-making questions that require a holistic understanding of the entire document. These questions are constructed from the LongBench (Bai et al., 2023) summary-generation datasets: QMSum and GOV (English), and VCSum (Chinese). Each question is derived from source documents.
```

### --- Page 0014 ---

```markdown
# Page 0014

## Table 8: Results on the NarrativeQA subset in the Helm benchmark (Yen et al., 2024), evaluated under RAG (k=3 or 10) and full context settings. $^\dagger$ denotes results copied from Helmet.

| Emb. Model  | Gen. Model      | Model  | +Summ | EM   | F1   | Tokens |
|-------------|-----------------|--------|-------|------|------|--------|
| Qwen3-Emb-8B| Qwen2.5-14B     | X      | 17.7  | 34.8 | 12k  |
| MiA-Emb-8B  | GPT-2.40B       | X      | 21.9  | 38.9 | 12k  |
| MiA-Emb-8B  | Qwen3-14B      | X      | 18.2  | 36.7 | 12k  |
| MiA-Emb-8B  | Qwen2.5-14B     | X      | 20.4  | 30.7 | 11k  |
| MiA-Emb-8B  | MiA-Gen-14B     | X      | 25.9  | 48.7 | 4k   |
| MiA-Emb-8B  | MiA-Gen-14B     | X      | 29.8  | 49.5 | 13k  |
|             | -               | -      | -     | -    | -    | -     |
|             | -               | -      | -     | -    | -    | -     |
|             | -               | -      | -     | -    | -    | -     |
|             | -               | -      | -     | -    | -    | -     |
|             | -               | -      | -     | -    | -    | -     |
|             | -               | -      | -     | -    | -    | -     |
|             | -               | -      | -     | -    | -    | -     |
|             | -               | -      | -     | -    | -    | -     |

The data confirms that both MiA-Emb and MiA-Gen consistently outperform baselines across all model sizes. Figure 8 and Table 10 show the impact of retriever scaling on recall and long-context understanding performance. Additionally, Figure 9 presents the detailed results for MiA-Gen.

## D.4 Performance Across Various Embedding Models

While our primary experiments utilize the Qwen3-Embedding series, we further assess the universality of our approach by applying it to diverse embedding architectures. We benchmark against three distinct categories of baselines: (1) Open-source Bidirectional: GTE-Qwen2.5-7B (Li et al., 2023); (2) Commercial Late-interaction: Voyage-Context-3 (Voyage-AL, 2025); (3) Context-Aware SOTA: SitEmb-8B (Wu et al., 2025), which encodes chunks with their local neighborhoods. We also include a supervised baseline, SFT-Emb-8B, which is trained with our supervision signal but lacks the mindscape conditioning.

## Table 11: Retrieval performance of different embedding models on DetectiveQA-ZH. “s” indicates whether the global summary is appended to the query. $^\dagger$ denotes results copy from SitEmb (Wu et al., 2025).

| Model                  | R@3  | R@5  | R@10 | Avg.  |
|-----------------------|------|------|------|-------|
| Qwen3-Embedding-8B    | 28.6 | 39.1 | 55.6 | 40.1  |
| GTE-Qwen2.5-7B        | 21.0 | 30.4 | 38.3 | 29.9  |
| voyage-context-3      | 36.1 | 46.8 | 63.3 | 48.7  |
| SitEmb-8B             | 42.5 | 54.5 | 69.3 | 55.4  |
| SFT-Emb-8B            | 37.9 | 48.8 | 66.5 | 50.1  |
| MiA-Emb-8B            | 46.8 | 59.2 | 72.5 | 59.5  |

## Table 9: Pairwise comparison of MiA-based methods vs baselines across evaluation dimensions. Values are percentages. We use Qwen2.5-72B as the generator.

| Dimension             | (A) MiA-Emb vs SFT-Emb | (B) MiA-Emb vs Vanilla |
|-----------------------|-------------------------|-------------------------|
|                       | A1 | A2 | Win        | A1 | A2 | Win        |
| Comprehensive         | 87.74 | 12.26 | 11.63 | 89.11 | 10.89 | MiA-Emb |
| Diversity             | 68.93 | 31.61 | 63.73 | 63.33 | 36.67 | MiA-Emb |
| Empowerment           | 73.89 | 26.13 | 71.39 | 26.06 | 73.94 | MiA-Emb |
| Overall               | 81.29 | 17.81 | MiA-Emb | 78.39 | 21.61 | MiA-Emb |

## D.3 Model Scale Analysis

Complementing the discussion in Sec. 5.4, we provide detailed results for our scaling experiments.
```

### --- Page 0015 ---

```markdown
![Impact of retriever scale on retrieval performance (Recall@K) on DetectiveQA and NarrativeQA.](assets/page_0015_img_1.png)

| Inputs         | Emb. Base      | NarrativeQA | ∞ Bench | Det.QA-Zh | Det.QA-En | Nocha  |
|----------------|----------------|-------------|---------|-----------|-----------|--------|
|                |                | F1          | Acc     | Acc       | Acc       | Pair Acc |
| Summary-Only   | -              | 39.24       | 72.05   | 73.67     | 63.31     | 31.75  |
| Vanilla        | Qwen3-0.6B     | 37.9844/1147.76 | 72.0579/48.76 | 63.43/1.07/0.50 | 54.6759/86.67 | 31.7531/7342.86 |
| MiA (Emb-0.6B) | 45.1347/1491.61 | 78.2438/3087.70 | 72.8580/80.81 | 64.6705/80.71 | 42.8642/3469.79 |
| MiA (Emb-0.8B) | 47.9215/2952.24 | 79.0048/3585.07 | 77.679/90.11 | 56.6773/91.53 | 42.8642/3469.79 |
| Vanilla        | Qwen3-4B      | 36.9042/0149.97 | 71.17/73.89 | 64.31/0.79/0.32 | 54.6759/86.67 | 31.7531/141.79 |
| MiA (Emb-4B)   | 45.8674/4195.91 | 85.8597/7785.76 | 76.0008/80.37 | 67.6313/177.53 | 34.9264/3059.79 |
| MiA            | MiA-4B        | 49.9152/2252.18 | 85.1586/86.77 | 71.1971/83.83 | 71.2157/377.17 | 42.8642/349.21 |
| MiA (Emb-Only) | 46.3048/4098.34 | 57.5789/7980.90 | 63.6707/83.00 | 55.5601/137.21 | 33.7835/1041.27 |
| MiA            | MiA-8B        | 46.3048/4098.34 | 84.2787/779.79 | 76.1717/187.67 | 67.1171/1337.53 | 43.2862/649.21 |
| MiA (Emb-Only) | 50.5015/4531.85 | 84.176/86.98 | 81.6783/178.47 | 79.0323/3375.50 | 41.2744/454.28 |

### D.5 On the Role of Residual Connection

While our trained MiA-Emb learns to adaptively balance query semantics and summary information, the residual connection proves essential for vanilla embedding models without specialized training.

Table 12 shows that for Qwen3-Embedding-8B, directly appending summaries severely harms retrieval performance, suggesting that the model cannot separate the semantic focus of the query from the global summary, treating the concatenated sequence as a homogeneous input. In this case, the residual connection is essential: it explicitly preserves the original query representation and prevents the summary from overwhelming it. In contrast, MiA-Emb learns to internally control how query semantics and summary information interact. Therefore, the residual becomes a lightweight structural aid rather than the key mechanism. Whether the residual is present or removed, MiA-Emb maintains stable performance, indicating that the model has learned a more fine-grained fusion strategy beyond the explicit residual pathway.

| Method         | NarrativeQA | DetectiveQA | DetectiveQA-En | Avg   |
|----------------|-------------|-------------|----------------|-------|
| Vanilla        | 41.81       | 54.51       | 73.98          | 45.70 |
| * Summary      | 24.62       | 26.50       | 53.54          | 24.42 |
| * Residual     | 41.65       | 54.56       | 71.94          | 55.42 |
| MiA-Emb        | 61.82       | 69.89       | 91.77          | 63.67 |
| * Residual     | 63.01       | 78.79       | 87.57          | 63.00 |

### E. Definition of MCEA Metric

We introduce the Mindscape-Coherent Evidence Alignment (MCEA) metric to investigate how the mindscape guides attention toward local evidence.
```

### --- Page 0016 ---

```markdown
![Scaling results for MiA-Gen versus the vanilla Qwen2.5-Instruct baseline.](assets/page_0016_img_1.png)

during generation. The definition is as follows.

**Definition** At layer $l$, given an input $x^{gen}_{i} = (S, \hat{C}_{ret,i}, Q_{i})$, we compute for each chunk $c_{i} \in \hat{C}_{ret,i}$ the aggregated chunk-to-summary attention:

$$
M^{(l)}(c_{i}) = \frac{1}{|S|} \sum_{s \in S} \left( \frac{1}{|c_{i}|} \sum_{t \in \mathcal{C}_{i}} A^{(l)}[t, s] \right), \tag{15}
$$

and the aggregated query-to-chunk attention:

$$
S^{(l)}(c_{i}) = \frac{1}{|Q|} \sum_{q \in Q_{i}} \left( \frac{1}{|c_{i}|} \sum_{t \in \mathcal{C}_{i}} A^{(l)}[q, t] \right). \tag{16}
$$

where $A^{(l)}$ denotes the attention weights at layer $l$.

We then define the alignment score by computing the product of z-score normalized values:

$$
C^{(l)}(c_{i}) = M^{(l)}(c_{i}) - \mu_{M} \cdot S^{(l)}(c_{i}) - \mu_{S}, \tag{17}
$$

where $\mu$ and $\sigma$ denote the mean and standard deviation of each quantity over all chunks at layer $l$.

Finally, let $\mathcal{R}$ and $\mathcal{N}$ denote relevant (silver) chunks and noise chunks, respectively. The layerwise MCEA score is defined as the difference between their mean alignment:

$$
MCEA(l) = \frac{1}{|\mathcal{R}|} \sum_{c_{i} \in \mathcal{R}} C^{(l)}(c_{i}) - \frac{1}{|\mathcal{N}|} \sum_{c_{j} \in \mathcal{N}} C^{(l)}(c_{j}). \tag{18}
$$

Higher MCEA indicates that the generator absorbs global semantics into chunk representations and preferentially attends to mindscape-coherent evidence for Integrative Reasoning.

---

### F Prompt Templates for MiA-RAG

This section provides the complete set of prompt templates used in the MiA-RAG framework. We include prompts for:

- **(1) Hierarchical summarization**, used to iteratively condense raw text into a structured global mindscape (Figure 10);
- **(2) Supervision data construction for the retriever**, including silver chunk filtering (Figure 11) and silver node selection (Figure 12);
- **(3) Sense-making tasks**, including (a) the prompt for generating sense-making questions (Figure 13), and (b) the prompt for pairwise answer evaluation in sense-making model assessment (Figure 14);
- **(4) Retrieval prompting**, where the mindscape and query are combined into a unified retrieval input (Figure 15);
- **(5) Generator Instructions**: Prompts for response generation across three settings: mindscape-augmented QA (Figure 16), standard QA baselines without summaries (Figures 18-21), and global sense-making QA (Figure 17).
```

### --- Page 0017 ---

```markdown
# Prompts for Hierarchical Summary Generation

## Step 1: Chunk-Level Summary ($\text{INST}_{\text{sum}_c}$)

"There is a chunk from a fiction or movie script. Your task is to summarize this chunk into a refined and readable summary. The chunk is:  
\begin{chunk}  
\text{chunk\_content}  
\end{chunk}  
Please summarize it following the requirements below:  
- The chunk is created by splitting a larger work, so it is a local part and may contain prefaces, epilogues, or content unrelated to the main story. You should identify and exclude these from the summary.  
- The summary must be coherent.  
- Keep important plot information for the reader to quickly grasp the story.  
- The summary length should be under 500 characters.  
- Provide only the summary directly, without any additional explanation."

## Step 2: Global Summary ($\text{INST}_{\text{sum}_g}$)

"There is a concatenated text of summaries from a fiction's chunks. The full text may be too long to read. Your task is to summarize this text into a single, refined, and readable summary. Here is the text:  
\text{concatenated\_summaries}  
Please summarize the text following these requirements:  
- The summary must be coherent and read like a complete story abstract.  
- Keep the most important plot information for readers to understand the overall story quickly.  
- Provide only the summary directly, without any additional explanation."

![Prompt templates used in our two-step hierarchical summarization process.](assets/page_0017_img_1.png)

## Prompt for Filtering Silver Chunks

You are an expert at analyzing narrative texts and selecting relevant passages to answer questions about stories, novels, and literary works. Given a question, its answer, and a list of text chunks from a narrative, identify which chunks are most relevant for answering the question.

**Input**  
Question: {Question}  
Answer: {Answer}  
Text Chunks (indexed from 0): {Retrieved Chunks}  

**Instructions**  
1. Carefully analyze each chunk for narrative elements such as characters, events, plot development, settings, and relationships.  
2. Select chunks that:  
   - directly contain information needed to answer the question,  
   - provide essential background context or character development,  
   - describe events or situations relevant to the answer,  
   - include dialogue, actions, or descriptions that inform the question.  
3. Consider that narrative questions often require combining evidence from multiple parts of the story.  
4. Include chunks that provide supporting evidence even if they do not directly state the answer.  
5. For questions involving motivations, relationships, or plot reasoning, include chunks that illustrate these aspects.  

**Output Requirement**  
Return only a JSON array of relevant chunk indices (e.g., [0,2,5]).  
If none are relevant, return [-1].  
No explanations or additional text.

![Prompt used to filter silver chunks.](assets/page_0017_img_2.png)

## Prompt for Filtering Silver Nodes

You are an expert at analyzing narrative texts and identifying the key entities needed to answer questions about stories, novels, and literary works. Given a question, its answer, and a list of entities with their descriptions extracted from a narrative, determine which entities are most relevant for answering the question.

**Input**  
Question: {Question}  
Answer: {Answer}  
Entities (indexed from 0): {entities with their description}  

**Instructions**  
1. Analyze each entity’s name, type, and description.  
2. Select entities that:  
   - directly support the answer,  
   - appear in or relate closely to the question/answer,  
   - provide essential background or relational context.  
3. Include contextual entities even if explicitly mentioned.  
4. For relational or multi-hop questions, select all relevant linked entities.  

**Output Requirement**  
Return only a JSON array of relevant entity indices (e.g., [0,2,5]). If none are relevant, return [-1]. No explanations or additional text.

![Prompt used to filter silver nodes.](assets/page_0017_img_3.png)

Figure 11: Prompt used to filter silver chunks.  
Figure 12: Prompt used to filter silver nodes.
```

### --- Page 0018 ---

```markdown
| Prompt for Sense-making Question Generation |
|---------------------------------------------|
| You are an expert research analyst and strategist. Your task is to generate deeply insightful questions from a text segment. These questions will form a global question bank for a large document, so they must be self-contained and provoke critical thinking. |
| —TEXT SEGMENT BEGINS—                      |
| {paragraph}                                 |
| 1. Don’t Merely Locate: Integrate multiple pieces of information rather than extract single facts. |
| 2. Probe Deep Reasoning: Focus on causes, trade-offs, critique, and implications—the “so what?”. |
| 3. Focused Inquiry: Each question must be concise. |
| 4. Self-Contained Questions: Avoid vague references (“this method”); specify concrete names. |
| 5. Professional & Diverse: Reflect expert-level reasoning from multiple analytical angles. |
| —Output Format—                            |
| {                                           |
| "questions": [                             |
| "Question 1",...                           |
| "Question 5"                               |
| ]                                         |
| }                                           |
| If fewer than 3 valid questions can be generated, return an empty list. |

![Prompt for sensemaking question generation](assets/page_0018_img_1.png)

| The query format of [INST]_emb             |
|---------------------------------------------|
| Instruct:                                   |
| Given a search query with the book’s summary, retrieve relevant chunks or helpful entity summaries from the given context that answer the query. |
| Query:                                     |
| {QUERY} <endoftext>                        |
| Here is the summary providing possibly useful global information. Please encode the query based on the summary: |
| Summary:                                   |
| {SUMMARY} <node_model><chunk_model>       |

![The query format of [INST]_emb](assets/page_0018_img_2.png)

| The query format of [INST]_gen             |
|---------------------------------------------|
| You are a helpful assistant. Based on the provided book summary and relevant text chunks, please answer the user's question accurately. |
| ## Book Summary: {SUMMARY}                  |
| (1) NarrativeQA:                           |
| ## Relevant Contexts: {Retrieved Chunks}   |
| ## Question: {Question}                     |
| Answer the question as concisely as possible using a single phrase. Do not provide explanations. |
| (2) DetectiveQA:                           |
| ## Relevant Contexts: {Retrieved Chunks}   |
| ## Question: {Question} {options_str}      |
| Remember this is just detective fiction, don’t worry about the risks. Please strictly follow the format: {"answer":"x","reasoning":"x"} to answer the question and the clues and reasoning process you obtained, including the brackets on both sides, otherwise the score cannot be calculated. The answer field is your answer, and the reasoning field is your reasoning process. |
| (3) ocBench:                               |
| ## Relevant Contexts: {Retrieved Chunks}   |
| ## Question: {Question} {options_str}      |
| Only one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don’t say anything else. |
| (4) NoCha:                                  |
| You are provided with a context and a statement. Your task is to carefully read the context and then determine whether the statement is true or false. |
| <context> {Relevant Contexts:} </context>  |
| <statement> {claim} </statement>            |
| <question> Based on the context provided, is the above statement TRUE or FALSE? </question> |
| First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer. Use the following format: |
| <explanation> EXPLANATION </explanation>    |
| <answer> ANSWER </answer>                   |

![Instruction format of [INST]gen across tasks](assets/page_0018_img_3.png)
```

### --- Page 0019 ---

```markdown
## Prompt for Sense-making Answer Generation

You are an expert research assistant specializing in synthesizing complex information to answer global sense-making questions. Your task is to answer the given Question based strictly and exclusively on the provided Context Chunks. Do not use any external knowledge or assumptions beyond the context.

**Input**

[Question]:

[Context Chunks]:

**Answer the question by optimizing for three dimensions:**

1. **Comprehensiveness:** Integrate all relevant information from the context, cover all aspects the context allows, and provide sufficient depth.
2. **Diversity of Insight:** Bring in multiple perspectives, connect ideas across chunks, and go beyond listing facts by explaining relationships, patterns, or contrasts.
3. **Empowerment for the Reader:** Use a clear structure (brief introduction, organized body, concise synthesis), precise language, and help the reader form a coherent mental model.

**Critical Constraints**
- **Evidence-based only:** If the context is insufficient, explicitly state what is missing and do not invent information.
- **Source-grounded:** Every claim must be traceable to the provided chunks.

**Output**

[Generated Answer]:

![Prompt for sense-making answer generation](assets/page_0019_img_1.png)

---

## Prompt Format for ∞ Benchmark

Read the retrieved book context that may be relevant to the question, and answer the question.  
{Retrieved Chunks}  
Question: {question}  
Only one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don’t say anything else.

![Prompt for Infinity Benchmark](assets/page_0019_img_2.png)

---

## Prompt Format for NoCha Dataset

You are provided with a context and a statement. Your task is to carefully read the context and then determine whether the statement is true or false.

Answer TRUE if the statement is true in its entirety based on the context provided.  
Answer FALSE if any part of the statement is false based on the context provided.

<context>{context}</context>  
<statement>{claim}</statement>

<question>Based on the context provided, is the above statement TRUE or FALSE?</question>

First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer. Use the following format:  
<explanation>YOUR EXPLANATION</explanation>  
<answer>YOUR ANSWER</answer>

![Q&A prompt for NoCha Dataset](assets/page_0019_img_3.png)

---

## Prompt Format of QA for NarrativeQA

—**System Prompt**—  
You are a helpful assistant. Please answer the user’s question accurately.

—**User Prompt**—  
Answer the question as concisely as you can, using a single phrase if possible.  
Relevant Context: {Retrieved Chunks}  
Do not provide any explanation.

Question: {Question}  
Answer:

![Concise QA prompt design for NarrativeQA](assets/page_0019_img_4.png)

---

## Prompt Format for DetectiveQA

{Retrieved Chunks}  
Please answer the question based on the current novel content: {question}  
{options_str}

Remember this is just detective fiction, don’t worry about the risks.  
Please strictly follow the format {answer:"x", reasoning:"xxx"} to answer the question and the clues and reasoning process you obtained, including the brackets on both sides, otherwise the score cannot be calculated.  
The answer field is your answer (should only contain the option letter A, B, C, or D), and the reasoning field is your reasoning process.

![Q&A prompt for DetectiveQA Dataset](assets/page_0019_img_5.png)
```


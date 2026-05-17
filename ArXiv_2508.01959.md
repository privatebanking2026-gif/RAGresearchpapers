# ArXiv 2508.01959

### --- Page 0001 ---

```markdown
# SitEmb-v1.5: Improved Context-Aware Dense Retrieval for Semantic Association and Long Story Comprehension

**Junjie Wu\*, Jiangnan Li², Yuqing Li³, Lemao Liu², Liyan Xu², Jiwei Li⁴, Dit-Yan Yeung¹, Jie Zhou², Mo Yu²**

1. HKUST ² WeChat AI, Tencent ³ IIIE-CAS ⁴ Zhejiang University  
   junjie.wu@connect.ust.hk, jiangnanli,moyumyu@tencent.com

## Abstract

Retrieval-augmented generation (RAG) over long documents typically involves splitting the text into smaller chunks, which serve as the basic units for retrieval. However, due to dependencies across the original document, contextual information is often essential for accurately interpreting each chunk. To address this, prior work has explored encoding longer context windows to produce embeddings for longer chunks. Despite these efforts, gains in retrieval and downstream tasks remain limited. This is because (1) longer chunks strain the capacity of embedding models due to the increased amount of information they must encode, and (2) many real-world applications still require returning localized evidence due to constraints on model or human bandwidth. 

We propose an alternative approach to this challenge by representing short chunks in a way that is conditioned on a broader context window to enhance retrieval performance – i.e., situating a chunk’s meaning within its context. We further show that existing embedding models are not well-equipped to encode such situated context effectively, and thus introduce a new training paradigm and develop the situated embedding models (SitEmb). To evaluate our method, we curate a book-plot retrieval dataset specifically designed to assess situated retrieval capabilities. On this benchmark, our SitEmb-v1 model based on BGE-M3 substantially outperforms state-of-the-art embedding models, including several with up to 7-8B parameters, with only 1B parameters. Our BGE SitEmb-v1.5 model further improves performance by over 10% and shows strong results across different languages and several downstream applications.¹

## 1 Introduction

Text embedding models (Wang et al., 202a; Sturua et al., 2024; Nussbaum et al., 2024; Hu et al., 2025) encode textual inputs into vector spaces. These models enable efficient semantic representation and matching, thus are foundational to many applications involving retrieval-augmented generation (RAG) (Lewis et al., 2020), such as code generation (Wang et al., 202b; Miao et al., 2024), reference generation (Wu et al., 2025), and personal AI assistants (Martin & Johnson, 2023).

In these tasks, candidate documents are typically segmented into smaller chunks to facilitate efficient processing. However, since documents often exhibit a narrative or logical flow, the meaning of each chunk is highly dependent on its surrounding context. This highlights the need for text embeddings that capture broader contextual information to enable context-aware retrieval.

One straightforward approach to this issue is to increase chunk size, allowing each chunk to capture more information. This has motivated a wave of recent work on supporting
```


### --- Page 0002 ---

```markdown
# Preprint

![Comparison of the same embedding models that return the same lengths of texts with different chunk sizes on our evaluation task.](assets/page_0002_img_1.png)

Figure 1: Comparison of the same embedding models that return the same lengths of texts with different chunk sizes on our evaluation task (§3). X-axis refers to chunk sizes. For example, when the return text is 5k and the chunk size is 1,024, the retriever returns top-5 chunks.

long input sequences in embedding models, either by designing efficient bidirectional models (Chen et al., 2024; Sturua et al., 2024; Nussbaum et al., 2024), or by repurposing powerful unidirectional pre-trained LLMs as embedding generators (Li et al., 2023; Wang et al., 2024; Moreira et al., 2024; Kim et al., 2024). These models can produce embeddings for sequences of up to 8,192 tokens or more.

However, it is often observed that simply enabling longer input windows does not necessarily lead to better embeddings. A key reason lies in the limited capacity of embedding vectors – embedding models must compress the information in the input text into a single vector. Intuitively, the longer the input chunk, the more information it contains, and the more long-range dependencies across arbitrary pairs of chunks within a document it needs to capture. This increases the likelihood of critical information loss during compression. Existing models are trained by merely extending the context window, without explicitly learning how to represent such distributed contextual relationships, which leads to a counterintuitive outcome: applications built on long-chunk embeddings often underperform those using short-chunk embeddings, despite the latter discarding more contextual information. Figure 1 illustrates this effect on a book plot retrieval task (Xu et al., 2024a). When the same total length of text (5k/10k/25k tokens) is retrieved using the same embedding model (Jina or NV-Embed), recall consistently decreases as the documents are segmented into longer chunks.

Given the aforementioned challenges, we propose an alternative approach to context-aware embeddings: directly incorporating the broader context surrounding each chunk into its chunk embedding. This allows the model to account for how the chunk is situated within the original document, enabling more contextually informed embeddings. In other words, we aim to situate a chunk’s meaning within its broader context (Yu et al., 2023; Xu et al., 2024a) during the embedding process. We refer to this approach as situated embedding. By doing so, we alleviate the issue of capacity limitations: during encoding, the model only needs to identify and integrate context that is relevant to the target chunk, which is a more tractable task than modeling dependencies across an extended input window.

Building on the idea of situated embedding, we first investigate whether existing embedding models can effectively generate situated embeddings. However, we find that situated embedding cannot be achieved simply by prompting existing embedding models, as demonstrated in §4. To address this limitation, we develop a dedicated situated embedding model specifically designed to handle this scenario. We achieve this through two techniques: (1) Constructing context-dependent training instances using publicly available user-annotated book notes. Platforms such as Douban2 allow users to write notes attached to particular book segments. We treat the note as a query and the chunk as groundtruth, framing a retrieval task with ~1.6M query-candidate pairs. As user notes typically reflect the contextual understanding of surrounding context, it makes context-aware embeddings beneficial for this retrieval task. (2) Promoting context usage through residual learning. In many cases, a chunk alone may offer partial (usually ambiguous) clues about its relevance to the query, allowing models
```


### --- Page 0003 ---

```markdown
# Preprint

To exploit shortcuts. To counter this, we employ a residual architecture where the situated embedding model is trained to resolve the residual from a baseline chunk-only embedding model. This encourages the model to focus on the additional contextual information.

To evaluate models’ context-aware embedding capability, we curate a book-plot retrieval task following (Xu et al., 2022a), which has been verified by previous work for its guaranteed requirement of context-aware embedding capability. Experiments demonstrate our situated embedding model’s superior performance over all the state-of-the-art embedding models, including those with up to 7B parameters and with massive pre-training. We further illustrate the generalizability of the trained models on the Recap Snippet Identification task (Li et al., 2024), a distinct task in book understanding beyond standard query-candidate retrieval, and several downstream story understanding tasks such as QA and claim verification that require RAG.

## 2 Our Situated Embedding Model

We develop the first model that generates high-quality situated embeddings as follows:

### Training Data Construction

We construct two sets of training data in English and Chinese, corresponding to different usage scenarios of retrieval, long-story comprehension, and semantic association. For the story comprehension purpose, we follow Mou et al. (2021) and build data based on NarrativeQA (Kočiský et al., 2018). For association-oriented training, we draw on book notes following prior work (Yu et al., 2023; Zhou et al., 2025). Book notes are particularly suitable for this purpose, as they capture human readers’ divergent thinking when engaging with a paragraph, thereby revealing associations between notes and the corresponding text. The construction of this book-note training data is detailed as follows.

We collect notes and their associated anchor texts for ~100 most popular books according to Douban. We treat each user note as a query and its corresponding anchored segment as a chunk, resulting in 1,614,007 query–chunk pairs. We reserve all the query–chunk pairs from our evaluation books and randomly select 1000 pairs for early-stopping.

Given a query–chunk pair, we define the situated context of the chunk as a sequence of its surrounding sentences, including the chunk itself. Specifically, we use the user-underlined texts anchored to their query as the chunk’s situated context, as these texts naturally align with our definition. Due to variations in user behavior, the lengths of these situated contexts range from 37 tokens to several thousand, making our trained model robust to a wide range of context lengths.

After this process, each chunk will be contained within one such segment, and we regard the segment as the situated context of the chunk.

### Residual Learning to Promote Situated Context Usage

Prior studies, such as (Ettinger, 2020), have shown that BERT-based models often rely on shallow heuristics or partial, ambiguous clues when matching texts. This behavior hinders the model’s ability to fully comprehend the entire input, which potentially explains why existing embedding models struggle to utilize long contextual information. To address this limitation, we adopt a residual learning framework (He et al., 2016), in which a situated embedding model is trained to resolve the residual from a baseline chunk-only embedding model, thereby equipping the trained model with a deeper understanding of situated context.

Specifically, we maintain two models, a baseline model $\hat{θ}$ that embeds the chunk only and a situated model $\hat{θ}^*$ that embeds the chunk situated within the context. For each query-chunk pair in the training data, we treat the chunk as the positive sample, and randomly sample 10 other chunks from the remaining chapters of the same book as negative samples. A query is embedded as $\bar{q} = \bar{q}^+ + \bar{q}^-$, where $\bar{q}^+$ and $\bar{q}^-$ are embedding vectors from $\hat{θ}$ and $\hat{θ}^*$, respectively. Similarly, a chunk is embedded as $\bar{c} = c^b + c^e$. The training loss on each
```

### --- Page 0004 ---

```markdown
# Preprint

| Model            | Size | Chunk-Only | Situated Context | + Situated Summ. |
|------------------|------|------------|------------------|-------------------|
|                  |      | @10       | @20              | @50               | @10       | @20       | @50               |
| M3               | 0.5B | 42.55     | 53.33            | 66.51             | 9.48      | 15.68     | 24.70             |
| Jina-v3         | 0.5B | 42.66     | 52.67            | 62.92             | 34.10     | 44.27     | 58.33             |
| E5-Mistral      | 7B   | 43.15     | 51.93            | 66.65             | 14.77     | 24.97     | 32.68             |
| GTE-Qwen2       | 7B   | 46.19     | 55.79            | 71.15             | 19.01     | 29.77     | 49.34             |
| NV-Embed-v2     | 7B   | 51.38     | 60.11            | 71.26             | 21.01     | 30.02     | 42.95             |
| Qwen-Embedding   | 8B   | 51.58     | 61.32            | 73.47             | 48.01     | 58.71     | 71.86             |
| voyage-context-3 | unk  | 58.54     | 68.47            | 70.99             | 60.46     | 69.50     | 82.19             |
|                  |      | 61.37     | 70.36            | 82.84             |           |           |                   |
| SitEmb-v1-M3 (ours) | 1B | 48.79     | 58.45            | 73.29             | 50.86     | 60.57     | 76.40             |
| SitEmb-v1.5-Qwen3 (ours) | 8B | 61.66 | 69.74            | 79.54             | 63.32     | 72.75     | 83.69             |
| + book note data | 8B   | 66.81     | 73.46            | 84.32             | 68.96     | 79.32     | 86.68             |

Table 1: Recall results on NDP-v1. The maximum length is set to 8,192. Best results of each setting are boldfaced.

The query-chunk pair can then be defined as:

$$
L(\Theta, \mathcal{O}) = \frac{1}{N} \sum_{i=1}^{N} \max(0, \gamma + \text{sim}(\hat{a}_i, \tilde{c}_i) - \text{sim}(a_i, \mathbf{t}_j))
$$

where $i$ is the index of negative chunk. See Appendix A for details of the training process.

## 3 Evaluation Dataset

Xu et al. (2024a) repurpose instances from the PlotRetrieval dataset Xu et al. (2024b) to support the task of contextual retrieval. Their work focuses on a single book, demonstrating that incorporating a graph-based representation of the book can improve local chunk retrieval. This finding highlights that the plot retrieval task inherently requires situated understanding and retrieval capabilities.

Following their work, we repurpose the PlotRetrieval dataset into a chunk-level retrieval task and expand the number of evaluation books from 1 to 7. Specifically, we filter books that are too short (i.e., <100,000 tokens), as they can typically be processed in a single input window and therefore diminish the utility of RAG. We also exclude books with too few user notes, as well as less popular versions on the reading platform, which tend to have less diverse note styles.

This filtering process results in 7 evaluation books containing 1,394 diverse queries, which together constitute the Book Plot Retrieval task (see Appendix B.1 for details on each selected book). When constructing the situated context for each chunk in the Book Plot Retrieval task, we first partition the chunk’s corresponding book into segments of consecutive sentences until the length reaches 200 tokens. We then sequentially group every 16 consecutive segments. This grouped context serves as the situated context of each chunk inside the group. During evaluation, we report Recall@10, Recall@20, and Recall@50 as the primary metrics.

**Remark**: Our plot retrieval task captures an important real-world scenario where localized retrieval results are essential. When users of an online reading app try to recall a plot, they typically lack the time or patience to read through long passages. In our setting, each retrieved segment corresponds to about 2–3 pages as displayed in a mobile or digital reading app, aligning with this user requirement. However, while humans can mentally connect these short segments to the broader narrative, the content alone is often insufficient due to missing context, highlighting the need for situated embedding techniques.
```

### --- Page 0005 ---

```markdown
# 4 Study I: Analysis of Existing Models on Generating Situated Embeddings

As the first step, we investigate the necessity of training a situated embedding model. That is, are existing long-context embedding models capable of generating good situated embeddings?

## Setup
We investigate this question on the NDP-v1 book in our evaluation dataset. Following our approach described in §2, we use the 16 surrounding chunks of each chunk to construct its situated context. We compare the following models: 
1. Long-context BERT models, including BGE-M3 (Chen et al., 2024) and Jina-v3 (Jina-Embeddings-v3) (Sturua et al., 2024).
2. LLM-based embedding models: E5-Mistral (E5-Mistral-7b-Instruct) (Wang et al., 2024), GTE-Qwen2 (GTE-qwen2-7b-Instruct) (Li et al., 2023), NV-Embed-v2 (Lee et al., 2024) and Qwen3-Embedding-8B (Zhang et al., 2025).
3. Our trained situated model from §2, including the v1 model based on M3 and the v1.5 model based on Qwen3. For reference, we also compare with the concurrent work on the most advanced commercial late-chunking model voyage-context-3 (Voyage-AI, 2025). Check Appendix C for additional details on model usages.

## Results
Table 1 presents the evaluation results, from which we draw the following conclusions:

- **Existing models do not have zero-shot situated embedding capability.** When enhancing the contexts to chunks, the performance of all the existing models degrades significantly (i.e., comparing columns of *Situated Context* and *Chunk-Only*). Note that the length of the situated context is well within their claimed maximum context window sizes. In contrast, our situated embedding model can effectively leverage contextual information, and largely surpasses much larger 7B baselines.

- **The poor results partly sourced from limitation in understanding long inputs.** The failure of producing situated embeddings is partly from the existing models' (actual) insufficiency of handling long inputs. To see this, we in addition compare with the LLM-generated situated summaries approach (Anthropic, 2024), which prompts an LLM to generate a concise summary that reflects how a chunk is situated within its broader context as the contextual information. We ask GPT-4 (OpenAI, 2024) to generate the situated summaries and use them in the same way like the situated contexts. Note that we use this setting only for reference, because it does not make a fair comparison due to the involvement of a much stronger model in the pipeline with high computational cost.

From the results, all the models suffer from a much smaller degrade when using the summaries instead of original situated contexts, while M3, Jina, E5 and Qwen3 have their results slightly increased. This reflects the fact that the baselines fail to situate the target chunk within long contexts. In comparison, our approach can achieve performance boost for both types of contextual inputs.

# 5 Study II: Analyzing the Robustness of Our SitEmb Models

In this study, we examine two aspects crucial to real-world applications: (1) whether our SitEmb models learn to generalize to new books rather than rely on memorization, and (2) whether they are robust to variations in situated context length.

## The Impact of Training-Test Book Overlap
A key concern in evaluating pre-trained language models, including embedding models, is whether models benefit unfairly from training-test overlap. Demonstrating that such overlap does not drive results is particularly important, because model training cannot anticipate all future downstream uses; it is therefore impractical to proactively filter training data against every possible evaluation or user scenario.

> The models are selected based on their strong performance on the MTEB benchmark (Muennighoff et al., 2023).
```

### --- Page 0006 ---

```markdown
# Preprint

## Table 2: Study the impact of training-test book overlap on the NDP-v1 task. We experiment with the Sit-Qwen3 with book-note setting.

| Setting         | Recall         |
|------------------|----------------|
|                  | @10   | @20   | @50   |
| w/ NDP           | 68.98 | 79.32 | 86.68 |
| w/o NDP          | 69.60 | 78.72 | 87.04 |

## Table 3: Recall results of our situated embedding models on NDP-V1 with various lengths of situated contexts. The listed lengths correspond to multiples (4/8/16/32/64) of the average segment range observed in books from the book plot retrieval task. The best results are **boldfaced** and the second best results are _underlined_.

| Situated Context Length | SitEmb-v1 Recall | SitEmb-v1.5 Recall |
|-------------------------|-------------------|---------------------|
|                         | @10   | @20   | @50   | @10   | @20   | @50   |
| [512, 800]             | 51.63 | 61.22 | 74.48 | 69.31 | 77.51 | 86.09 |
| [1024, 1600]           | 52.29 | 61.23 | 74.15 | 68.76 | 76.24 | 85.78 |
| [2048, 3200]           | 51.73 | 61.56 | 75.00 | 68.98 | 79.32 | 86.68 |
| [4096, 6400]           | 50.62 | 59.11 | 75.16 | 69.75 | 77.45 | 86.83 |
| [8192, 12800]          | 50.52 | 59.11 | 74.51 | 68.88 | 77.00 | 86.47 |

To verify the validity of our evaluation, we constructed a controlled experiment that modifies the training data from NarrativeQA with and without any version of the NDP books, and then evaluated their performance on NDP-v1. As shown in Table 2, models exposed to the test books during training exhibit no measurable performance gain, indicating that training-test overlap does not materially affect our results.

### Robustness to Context Length

We evaluate the sensitivity of our trained model to variations in context length. Experiment is conducted on the NDP-v1 book used in Table 1, varying the number of sentences per chunk, and measuring recall scores with our SitEmb-v1-M3 and the SitEmb-v1.5-Qwen models, both trained with book note data. The results in Table 3 demonstrate that the model maintains stable performance across different context lengths. Our choice of 16-segment groups strikes a balance between efficiency and accuracy.

## 6 Study III: Contextual Retrieval on the Full Book Plot Retrieval Task

We evaluate our situated embedding model on the full plot retrieval task to assess its effectiveness in enhancing contextual retrieval. We compared our models trained with the QA data (denoted as QA) and book note data (semantic association, denoted as SA). The M3 model fails to improve with the QA training data thus the corresponding results are omitted.

As shown in Table 4, incorporating contextual information through situated embeddings significantly improves performance. Our SitEmb-v1-M3 model consistently outperforms chunk-only baselines without our training techniques. The SitEmb-v1.5 models further boost performance by over 10% when trained on QA data and over 15% when trained on QA+SA data. Notably, both variants surpass the performance of the recent commercial late-chunking model voyage-context-3, and show clear advantages over their chunk-only variations.

To ensure that the gains of SitEmb-v1-M3 are not merely due to increased model capacity, we also train the same residual architecture on two chunk-only M3 models (Res-M3). It fails to yield improvements over the trained M3 (SA) baseline, indicating that the advantage of 

4 We use NarrativeQA rather than book-note data, since the queries in the plot retrieval task are originally derived from book notes so the test books have to be removed, making such an overlap experiment implausible in that setting.
```

### --- Page 0007 ---

```markdown
| Setting        | Model                     | Size | Recall          |
|----------------|---------------------------|------|------------------|
|                |                           | @10  | @20  | @50      |
|----------------|---------------------------|------|------|----------|
| Chunk-only     | M3 (out-of-box)          | 0.58 | 32.92| 41.46| 55.85    |
|                | M3 (SA)                  | 0.5  | 42.87| 52.91| 60.21    |
|                | + Residual               | 1B   | 43.43| 51.74| 65.51    |
|                | Qwen3 (out-of-box)      | 8B   | 43.57| 52.94| 69.48    |
|                | Qwen3 (QA)               | 8B   | 51.02| 60.78| 73.89    |
|                | Qwen3 (QA+SA)            | 8B   | 60.36| 68.87| 80.45    |
| Late Chunking   | voyage-context-3         | unk  | 51.39| 60.89| 73.70    |
|                | SitEmb-v1-M3 (SA)       | 1B   | 14.15| 15.66| 29.25    |
|                | - Residual               | 0.5B | 43.55| 54.98| 68.93    |
|                | SitEmb-v1.5-Qwen3 (QA)  | 8B   | 53.87| 63.76| 78.64    |
|                | SitEmb-v1.5-Qwen3 (QA+SA)| 8B   | 63.03| 72.83| 82.70    |

Table 4: Overall results on book plot retrieval. Check Table 9 and 10 for full results.

|                | Recap                    |
|----------------|--------------------------|
| Model          | R@5  | P@5  | F1@5   |
|----------------|------|------|--------|
| Qwen3          | 33.0 | 46.6 | 37.9   |
| Qwen3 (QA)    | 32.0 | 45.8 | 37.0   |
| Qwen3 (QA+SA) | 32.6 | 46.4 | 37.6   |
| SitEmb-v1.5-Qwen3 (QA) | 32.3 | 46.6 | 37.4   |
| SitEmb-v1.5-Qwen3 (QA+SA)| 33.6 | 48.2 | 38.9   |

Table 5: Results on the recap task.

our method primarily come from the effective use of contextual information. In addition, training without the residual architecture (- Residual) leads to degraded performance compared to our full SitEmb-v1-M3, further supporting our training design.

## 7 Study IV: Downstream Semantic Association Task – Recap Identification

Next, we assess the generalizability of our situated embedding model on downstream applications that are not explicitly designed for contextual retrieval and contain only a limited portion of context-dependent examples. In this section, we evaluate on a distinct task, Recap Snippet Identification task (Li et al., 2024), which aims to identify recap passages for a given paragraph. We following their setting of using top-5 retrieved passages.

Results Table 5 presents the results. Because this task differs substantially from our training data, it poses a challenging transfer setting. Consequently, our trained models without context usage show a slight performance drop compared to the original Qwen3 model. However, since recap identification requires embedding capabilities beyond simple similarity matching, our models trained with SA achieve better generalization to this task. In particular, the SitEmb-v1.5 (QA+SA) model outperforms all others by leveraging contextual information. These results highlight the importance of enhancing semantic association capabilities in embeddings.

## 8 Study V: Downstream Long Story Comprehension Applications

Finally, we evaluate on a variety of story understanding tasks that requires processing inputs exceeding the length limits of many LLMs, including NarrativeQA (Kočiský et al., 2018), the multichoice QA task from oobench (Zhang et al., 2024), the newly release DetectiveQA (Xu et al., 2025), the public subset of NoCha (Karpinska et al., 2024) and the LongStoryQA-large.
```

### --- Page 0008 ---

```markdown
# Preprint

| Model                        | NarrativeQA | $\alpha$Bench-IMC | DetectiveQA | NoCha (Public) | LongStoryQA-Large |
|------------------------------|-------------|--------------------|-------------|----------------|--------------------|
| Qwen3 (out-of-box)          | 275/308.322 | 751.840/860.0      | 625.687/732 | 429/413/460    | 527.579/612        |
| Qwen3 (QA)                   | 295/310.324 | 830.859/887        | 725.812/818 | 540/524/365    | 583.592/614        |
| SitEmb-v1.5-Qwen3 (QA)      | 311/310.344 | 830.859/890.7      | 703.782/823 | 540/564/460    | 577.571/619        |
| SitEmb-v1.5-Qwen3 (QA+SA)   | 294/316.318 | 830.865/862.6      | 625.742/783 | 546/524/492    | 577.594/615        |

Table 6: Results on the story QA tasks. We report results with top-3/5/10 retrieved chunks.

| Model              | Answer Recall | Clue Recall | Final Accuracy |
|--------------------|---------------|-------------|-----------------|
| voyage-context-3   | Top-3  | Top-5  | Top-10 | Top-3  | Top-5  | Top-10 | Top-3  | Top-5  | Top-10 |
|                    | 3.61   | 46.8  | 63.3  | 24.8  | 33.8  | 48.1  | 68.7  | 73.5  | 79.8  |
| Qwen3 (out-of-box) | 29.6   | 37.8  | 55.5  | 23.8  | 31.9  | 46.5  | 62.5  | 68.7  | 73.2  |
| Qwen3 (QA)         | 35.8   | 50.5  | 66.1  | 23.7  | 33.0  | 48.0  | 70.5  | 78.2  | 81.8  |
| SitEmb-v1.5-Qwen3 (QA) | 42.5   | 54.5  | 69.3  | 24.6  | 34.0  | 49.2  | 73.2  | 78.7  | 82.3  |
| SitEmb-v1.5-Qwen3 (QA+SA) | 29.4   | 41.3  | 56.7  | 26.9  | 36.4  | 51.2  | 65.4  | 74.2  | 78.3  |

Table 7: Study on the effects of improved retrieval on the DetectiveQA dataset, which provides evidence passage annotations.

task from CLongQ (Qiu et al., 2024). These tasks cover different genres, both English and Chinese languages, and task types of free-form QA, multi-choice QA and claim verification. We retrieve top-3/5/10 with the compared embedding models and use Qwen2.5-72B (4-bit quantized model) to generate the results.

Table 6 shows that our SitEmb-v1.5 model trained on QA data consistently outperforms its counterpart without situated embedding, except for language-specific top-5 results. In comparison, our SitEmb model trained on QA+SA yields mixed results relative to the no-context model, but still shows advantages over the original Qwen3. This suggests that existing story comprehension tasks demand limited semantic association capability, making the SitEmb (QA) model a well-balanced choice across diverse benchmarks.

One notable observation regarding performance degradation with larger retrieved context on NoCha is that, once the key plot is retrieved, additional context tends to consist mainly of distractions, causing an LLM with weaker reasoning ability to lose focus. To verify this, we evaluated the advanced Gemini2.5-Flash under our QA+SA setting, achieving top-3/5/10 pair accuracies of 55.6/57.1/57.1 without degradation. This confirms that the necessary evidence is saturated within the top-5 results.

## 9 Conclusion

This paper introduces the situated embedding models, which encodes a chunk's surrounding contextual information directly into its embedding, enabling a deeper understanding of the
```

### --- Page 0009 ---

```markdown
# Preprint

chunk itself. Experiments across multiple long-context understanding tasks demonstrate that situated embeddings provide an effective alternative approach to contextual retrieval, and our proposed model serves as a strong first step in advancing this direction.

## Limitations

While our experiments on several use cases highlight the advantages of embedding models with enhanced semantic association capabilities, results on broader applications are mixed. At this stage, training with QA-only data achieves a better overall balance. This suggests that semantic association exists along a spectrum from direct relevance to abstract and implicit relations. To excel across diverse scenarios, a model must be able to adaptively control its degree of divergence. Achieving this poses challenges for our current LoRA fine-tuning regime, which has limited capacity, and calls for new training objectives that explicitly encourage controllable association through instruction following.

Another limitation of our current work is that the models are primarily optimized for narrative data. In future work, we plan to construct training data from a broader range of domains to improve generalization.

## References

Anthropic. Enhancing rag with contextual retrieval. 2024. URL [https://github.com/anthropic/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb](https://github.com/anthropic/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb).

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge 3m-embedding: Multi-lingual, multi-functionality, multi-granularity contextual embeddings through self-knowledge distillation, 2024.

Allyson Ettinger. What bert is not: Lessons from a new suite of psycholinguistic diagnostics for language models. *Transactions of the Association for Computational Linguistics*, 8:34–48, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*, 2022. URL [https://openreview.net/forum?id=2nZKeefYf9](https://openreview.net/forum?id=2nZKeefYf9).

Xinshuo Hu, Zifei Shan, Xinping Zhao, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, et al. Kalm-embedding: Superior training data brings a stronger embedding model. *arXiv preprint arXiv:2501.01028*, 2025.

Marzena Karpinska, Katherine Thia, Kyle Lo, Tanya Goyal, and Mohit Iyyer. One thousand and one pairs: A "novel" challenge for long-context language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024*, Miami, FL, USA, November 12-16, 2024, 2024. URL [https://doi.org/10.18653/v1/2024.emnlp-main.948](https://doi.org/10.18653/v1/2024.emnlp-main.948).

Jihoon Kwon Sangmo Ge Yejin Kim, Minkyoung Cho Jy-yong Sohn Chanyeol, Choi Junseong Kim, and Seohwa Lee. Linq-embed-mistral: Elevating text retrieval with improved gpt data through task-specific control and quality refinement. linq ai research blog, 2024.

Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. The narrative reading comprehension challenge. *Transactions of the Association for Computational Linguistics*, 6:317–328, 2018. URL [https://aclanthology.org/Q18-1023.pdf](https://aclanthology.org/Q18-1023.pdf).
```

### --- Page 0010 ---

```markdown
# Preprint

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. NV-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428, 2024.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. *Advances in neural information processing systems*, 33:9459–9474, 2020.

Jiangnan Li, Qiuying Wang, Liyan Xu, Wenjie Pang, Mo Yu, Zheng Li, Weiping Wang, and Jie Zhou. Previously on the stories: Recap snippet identification for story reading. arXiv preprint arXiv:2402.07271, 2024.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023.

Xueguang Ma, Luyu Gao, Shengyao Zhuang, Jiaqi Samantha Zhan, Jamie Callan, and Jimmy Lin. Tevatron 2.0: Unified document retrieval toolkit across scale, language, and modality. arXiv preprint arXiv:2505.02466, 2025.

Raiza Martin and Steven Johnson. Introducing notebookml. 2023. URL https://blog.google/technology/ai/notebookml-google-ai.

Jing Miao, Chaarat Thongprayoon, Supawadee Supadungsuk, Oscar A Garcia Valenzuela, and Wisit Cheungpasit. Integrating retrieval-augmented generation with large language models in neophyte: advancing practical applications. *Medicina*, 60(3):445, 2024.

Gabriel de Souza P Moreira, Radek Oslumski, Mengyao Xu, Ronay Ak, Benedikt Schifferer, and Eric M. D. N. Oliveira. NV-retriever: NV-retrieving text embedding models with contrastive hard-negative mining. arXiv preprint arXiv:2407.15831, 2024.

Xiangyu Mou, Chenghao Yang, Mo Yu, Bingsheng Yao, Xiaoxia Guo, Saloni Potdar, and Hui Su. Narrative question answering with cutting-edge open-domain techniques: A comprehensive study. *Transactions of the Association for Computational Linguistics*, 9: 1032–1046, 2021.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. Mteb: Massive text embedding benchmark. In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pp. 2014–2037, 2023.

Zach Nussbaum, John X Morris, Brandon Duerdast, and Andriy Mulyar. Monic embed: Training a reproducible long context text embedder. arXiv preprint arXiv:2402.01613, 2024.

OpenAI. Help gpt-40. 2024. URL https://openai.com/index/hello-gpt-40/.

Zexuan Qiu, Jingjing Li, Shijue Huang, Xiaoqi Jiao, Wanjun Zhong, and Irwin King. Clongeval: A chinese benchmark for evaluating long-context large language models. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pp. 3985–4004, 2024.

Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimml, Feng Wang, Georgios Matastras, Andreas Koukounas, Andreas Koukounas, Nan Wang, and Han Xiao. jina-embeddings-v3: Multilingual embeddings with task lora, 2024. URL https://arxiv.org/abs/2409.12173.

Voyage-AI. Introducing voyage-context-3: focused chunk-level details with global document context, jul 2025. URL https://blog.voyageai.com/2025/07/23/voyage-context-3/. Blog post.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furui Wei. Improving text embeddings with large language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 11897–11916, 2022a.
```

### --- Page 0011 ---

```markdown
# Preprint

Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. Coderag-bench: Can retrieval augment code generation? arXiv preprint arXiv:2406.14497, 2024b.

Junjie Wu, Geifei Gu, Yanan Zheng, Dit-Yan Yeung, and Arman Cohan. Ref-long: Benchmarking the long-context referencing capability of long-context language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23861–23880, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-99176-251-0. URL https://aclanthology.org/2025.acl-long.1162/.

Liyan Xu, Jiangnan Li, Mo Yu, and Jie Zhou. Fine-grained modeling of narrative context: A coherence perspective via retrospective questions. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5822–5838, 2024a.

Shicheng Xu, Liang Pang, Jiangnan Li, Mo Yu, Fandong Meng, Huawei Shen, Xueqi Cheng, and Jie Zhou. Plot retrieval as an assessment of abstract semantic association. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pp. 146–161, 2024b.

Zhe Xu, Jiasheng Ye, Xiaoran Liu, Xiangyang Liu, Tianxiang Sun, Zhigeng Liu, Qipeng Guo, Linlin Li, Qun Liu, Xuanjing Huang, and Xipeng Qiu. DetectiveQA: Evaluating long-context reasoning on detective novels. In Workshop on Reasoning and Planning for Large Language Models, 2025. URL https://openreview.net/forum?id=9x5E5Lk.

Mo Yu, Jiangnan Li, Shunyu Yao, Weijing Yao, Xiachen Zhou, Zhou Xiao, Fandong Meng, and Jie Zhou. Personality understanding of fictional characters during book reading. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14784–14802, 2023.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. oebench: Extending long context evaluation beyond 100k tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11–16, 2024, 2024. URL https://doi.org/10.18653/v1/2024.acl-long.814.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Daiyiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embed-ding: Advancing text embedding and reranking through foundation models, 2025.

Chulun Zhou, Qijing Wang, Mo Yu, Xiaoqian Yue, Rui Lu, Jiangnan Li, Yifan Zhou, Shunchi Zhang, Jie Zhou, and Wai Lam. The essence of contextual understanding in theory of mind: A study on question answering with story characters. arXiv preprint arXiv:2501.01705, 2025.

## A  Additional Training Details of Our Situated Embedding Model

In this section, we describe additional details on how we attempt to train the first situated embedding model.

### A.1 Model Initialization

Before the residual learning process described in §2, we initialize two models, $\mathcal{O}^+$ and $\mathcal{O}^-$. While $\mathcal{O}^+$ is directly initialized from the BGE-M3 embedding model, we perform a prior training step on $\mathcal{O}^-$ to facilitate more effective residual learning.
```

### --- Page 0012 ---

```markdown
# Preprint

Specifically, we initialize $\theta$ from the same BGE-M3 embedding model as $\Theta$. For each query-chunk pair in the training data, we treat the chunk as the positive sample and randomly sample 10 negative chunks from other chapters of the same book. We then obtain the query embedding $q^$ and chunk embedding $c^$ from $\Theta$, and train $\Theta$ using the margin-based loss defined in Eq. 1, applied solely to this model. This prior training stage familiarizes $\Theta$ with the task of retrieving book chunks based on user notes, thereby providing a more informative foundation for the subsequent residual learning phase.

## A.2 Training Configurations

All training procedures in this paper follow a consistent configuration. For SitEmb-v1-M3, we use a learning rate of $2 \times 10^{-5}$ and a weight decay of $5 \times 10^{-2}$. The batch size is set to 80, and the maximum input sequence length is 8192 tokens, which corresponds to the input limit of BGE-M3. During training, we employ the development set introduced in §2 for early stopping. The model is evaluated on this set every 180 training steps, and training is terminated once both the training loss and development performance converge. The margin and temperature values used in the loss function are both set to 0.1. All experiments referring to SitEmb-v1-M3 are conducted using two NVIDIA A100 80G GPUs.

For the training of SitEmb-v1.5-Qwen3, we equip Qwen3-Embedding-B (Zhang et al., 2025) with a Low-Rank Adaptation (Hu et al., 2022). The rank is set to 128, the alpha is set to 256, and adapters are attached to the query/key/value/output projections in multi-head attention modules, whose dropout rate is set to 0.05. The training schedule moves on using the cosine LR, warming up at the first 10% steps, whose learning rate is set to $1 \times 10^{-4}$.

Unlike SitEmb-v1-M3, which first trains a chunk-only encoder and then residually trains a situated encoder with the chunk-only one, we fully utilize the causal-masking feature of the encoder in Qwen3-Embedding (i.e., the unidirectional feature that future tokens cannot access future tokens), and train the context-only and situated settings at the same time. Specifically, for the chunk and context encoding, we concatenate them into one sequence in which the chunk comes first and is followed by the context. Due to the unidirectionality, the chunk can only see itself (i.e., the context-only setting), and the context can realize the situated chunk (i.e., the situated setting). The sequence is formed as “[$\text{[CHUNK]}$]$\text{The context in which the chunk is situated is given below.}$” to encode the chunk by being aware of the context. $\text{[CONTEXT]}$ $\text{endoftext}$.” In this way, the chunk embedding and the situated embedding are obtained by the last pooling of extracting the embedding of the first and the second “$\text{endoftext}$”. We denote the chunk embedding as $c^$, the situated embedding as $s^$, and the query embedding as $q$. To co-train the two settings, the contrastive learning loss is computed using the scores $\text{sim}(q^, c^)$, $\text{sim}(q^, s^)$, and the temperature value of 0.01.

Furthermore, we follow Chen et al. (2024); Ma et al. (2025) to broadcast the computed scores of batches on every GPU to reach a bigger batch size. The batch size per GPU is set to 5, and we use 8 pieces of NVIDIA A800 80G GPUs. For every query, we sample a positive chunk plus 13 negative chunks from the same book. In this way, each query can see $8 \times 5 \times 14 = 560$ chunks at a step. Additionally, the accumulation step is set to 4, and the model will be trained for 5 epochs. The best checkpoint is picked by the result on the NDP-v1 dev set per epoch, which is always from epoch 1. Therefore, we use the checkpoint saved at epoch 2 by default. All evaluating experiments referring to SitEmb-v1.5-Qwen3 are conducted using a piece of NVIDIA A100 40G GPU in the data type of bfloat16.

# B Full Results Decomposed to Books

## B.1 Books in the Evaluation Dataset

Following the process described in §3, we select 7 books from the PlotReviator dataset to construct our evaluation set. The names of these books, along with the corresponding numbers of queries and candidate chunks, are summarized in Table 8.
```


### --- Page 0013 ---

```markdown
# Preprint

## Book

| Book                          | Queries | Candidates |
|-------------------------------|---------|------------|
| Notre-Dame de Paris (NDP)-v1  | 510     | 1288       |
| Notre-Dame de Paris (NDP)-v2  | 153     | 1369       |
| Notre-Dame de Paris (NDP)-v3  | 146     | 1347       |
| Crime and Punishment (C&P)    | 134     | 1639       |
| The Adventures of Tom Sawyer (TATS) | 173 | 154        |
| The Red and the Black (TRB)   | 144     | 1294       |
| Tess of the d'Urbervilles (TDU) | 134   | 1093       |

**Table 8:** Statistics of books in the evaluation dataset.

### Instruct:
Given a user note query, retrieve the passages that are most relevant to the content or context described in the query.

**Query:**
{QUERY}

![The query format of E5-Mistral and GTE-Qwen2](assets/page_0013_img_1.png)

Note that for some English books, the PlotRetrieval dataset includes multiple Chinese translation versions, treating each version as a distinct book. We adopt the same setting and denote the three translation versions of Notre-Dame de Paris in the 7 selected books as v1, v2, and v3, respectively. Among them, NDP-v1 is the version used in Table 1.

## C Details on Running Embedding Models

For all non-LLM embedding models (i.e., BGE-M3 and Jina-v3), we directly use the models to encode queries, chunks, and situated context, with the maximum input length set to 8192 tokens.

For E5-Mistral, GTE-Qwen2, and voyage-context-3, we follow the official encoding guidelines provided at [E5-Mistral](https://huggingface.co/float/e5-mistral-7b-instruct), [GTE-Qwen-7B](https://huggingface.co/Alibaba-NLP/gte-qwen-7b-instruct), and [voyage.ai](https://docs.voyage.ai.com/docs/contextualized-chunk-embeddings#quickstart), respectively. In both cases, we prepend a one-sentence instruction to each query as required, as illustrated in Figure 2.

For NV-Embed-v2, we adopt the same input format as E5-Mistral and GTE-Qwen2 when encoding queries. For chunk encoding in the chunk-only setting of Table 1, we omit instructions, consistent with the E5-Mistral and GTE-Qwen2 setups. In all other settings where additional context is included, we follow the official prompt format of NV-Embed-v2 ([NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)), as shown in Figure 3.

When running the latter two columns of experiments in Table 1, we append the situated context or situated summary to each chunk using the delimiters “</s>” and “\n\n”, respectively. The concatenated sequence is then treated as a new chunk and encoded as a whole.
```

### --- Page 0014 ---

```markdown
| Model            | Book         | @10    | @20    | @50    |
|------------------|--------------|--------|--------|--------|
| NDP-v1           |              | 42.55  | 53.33  | 66.51  |
| NDP-v2           |              | 33.22  | 44.18  | 60.62  |
| NDP-v3           |              | 43.79  | 51.63  | 64.38  |
| C&P              |              | 21.64  | 24.63  | 45.15  |
| TATS             |              | 38.73  | 47.40  | 63.01  |
| TRB              |              | 23.61  | 34.72  | 47.22  |
| TDU              |              | 26.87  | 34.33  | 44.03  |
| **Avg**          |              | 32.92  | 41.46  | 55.85  |
|                  |              |        |        |        |
| M3 (SA)         |              |        |        |        |
| NDP-v1           |              | 48.79  | 58.45  | 73.29  |
| NDP-v2           |              | 46.58  | 56.16  | 67.81  |
| NDP-v3           |              | 50.33  | 62.09  | 73.53  |
| C&P              |              | 36.57  | 42.91  | 54.85  |
| TATS             |              | 42.20  | 56.94  | 70.81  |
| TRB              |              | 36.80  | 43.06  | 58.33  |
| TDU              |              | 38.81  | 50.75  | 63.43  |
| **Avg**          |              | 42.87  | 52.91  | 66.01  |
|                  |              |        |        |        |
| Res-M3 (SA)     |              |        |        |        |
| NDP-v1           |              | 49.30  | 58.73  | 71.93  |
| NDP-v2           |              | 48.29  | 54.79  | 66.78  |
| NDP-v3           |              | 50.33  | 59.15  | 72.22  |
| C&P              |              | 35.45  | 39.93  | 52.99  |
| TATS             |              | 45.66  | 36.36  | 72.25  |
| TRB              |              | 35.42  | 40.97  | 59.72  |
| TDU              |              | 39.55  | 52.24  | 62.69  |
| **Avg**          |              | 43.43  | 51.74  | 65.51  |
|                  |              |        |        |        |
| SitEmb-v1-M3 (SA, No Res) |   |        |        |        |
| NDP-v1           |              | 51.31  | 63.07  | 74.84  |
| C&P              |              | 37.31  | 45.90  | 57.09  |
| TATS             |              | 49.42  | 59.25  | 76.01  |
| TRB              |              | 33.33  | 47.22  | 64.58  |
| TDU              |              | 39.55  | 52.99  | 67.91  |
| **Avg**          |              | 43.85  | 54.98  | 68.93  |
|                  |              |        |        |        |
| SitEmb-v1-M3 (SA) |            |        |        |        |
| NDP-v1           |              | 50.85  | 60.57  | 76.40  |
| NDP-v2           |              | 48.97  | 57.88  | 70.89  |
| NDP-v3           |              | 51.63  | 64.71  | 74.84  |
| C&P              |              | 36.94  | 44.40  | 57.84  |
| TATS             |              | 47.69  | 60.40  | 75.14  |
| TRB              |              | 38.19  | 47.92  | 63.19  |
| TDU              |              | 41.79  | 53.73  | 66.42  |
| **Avg**          |              | 45.15  | 55.66  | 69.25  |

Table 9: Full results of SitEmb-v1-M3 on book plot retrieval.

Your task is to embed passages for retrieval. Your input consists of the target passage and its context. You need to find relevant information from the context to enhance the target passage embedding such that it captures the meanings of the passages situated within the context.

context:  
{CONTEXT}  
passage:  
{PASSAGE}  

![Figure 3: Prompt for NV-Embed-v2.](assets/page_0014_img_1.png)
```

### --- Page 0015 ---

```markdown
| Model                     | Book   | @10   | @20   | @50   |
|---------------------------|--------|-------|-------|-------|
| Qwen3-Embedding (out-of-box) |        |       |       |       |
|                           | NDP-v1 | 51.20 | 61.63 | 76.10 |
|                           | NDP-v2 | 38.36 | 47.95 | 29.86 |
|                           | NDP-v3 | 50.65 | 60.78 | 76.14 |
|                           | C&P    | 35.82 | 43.66 | 60.45 |
|                           | TATS   | 49.71 | 60.69 | 75.14 |
|                           | TRB    | 43.06 | 50.00 | 65.97 |
|                           | TDU    | 36.19 | 45.90 | 62.69 |
| Avg                       |        | 43.57 | 52.94 | 69.48 |
|                           |        |       |       |       |
| Qwen3 (QA)               |        |       |       |       |
|                           | NDP-v1 | 61.66 | 69.06 | 79.54 |
|                           | NDP-v2 | 52.74 | 60.62 | 78.42 |
|                           | NDP-v3 | 54.79 | 70.59 | 83.66 |
|                           | C&P    | 40.67 | 52.61 | 64.93 |
|                           | TATS   | 58.09 | 65.61 | 77.75 |
|                           | TRB    | 46.53 | 56.25 | 68.75 |
|                           | TDU    | 42.54 | 50.74 | 64.18 |
| Avg                       |        | 51.02 | 60.77 | 73.89 |
|                           |        |       |       |       |
| Qwen3 (QA+SA)            |        |       |       |       |
|                           | NDP-v1 | 66.81 | 74.36 | 84.32 |
|                           | NDP-v2 | 67.47 | 75.68 | 84.59 |
|                           | NDP-v3 | 65.03 | 71.09 | 83.71 |
|                           | C&P    | 49.63 | 61.57 | 73.51 |
|                           | TATS   | 60.69 | 73.12 | 94.71 |
|                           | TRB    | 56.94 | 63.54 | 73.96 |
|                           | TDU    | 55.97 | 61.94 | 73.88 |
| Avg                       |        | 60.36 | 68.87 | 80.45 |
|                           |        |       |       |       |
| SitEmb-v1.5-Qwen (QA)    |        |       |       |       |
|                           | NDP-v1 | 63.32 | 72.75 | 85.39 |
|                           | NDP-v2 | 58.90 | 68.84 | 80.82 |
|                           | NDP-v3 | 59.48 | 70.89 | 84.64 |
|                           | C&P    | 44.40 | 54.10 | 76.49 |
|                           | TATS   | 58.96 | 70.23 | 83.24 |
|                           | TRB    | 45.14 | 57.64 | 71.53 |
|                           | TDU    | 44.78 | 50.75 | 70.15 |
| Avg                       |        | 53.57 | 63.56 | 78.64 |
|                           |        |       |       |       |
| SitEmb-v1.5-Qwen (QA+SA) |        |       |       |       |
|                           | NDP-v1 | 68.98 | 79.32 | 86.68 |
|                           | NDP-v2 | 71.23 | 79.45 | 89.73 |
|                           | NDP-v3 | 66.34 | 79.41 | 87.58 |
|                           | C&P    | 58.58 | 68.26 | 76.49 |
|                           | TATS   | 65.61 | 73.70 | 84.10 |
|                           | TRB    | 50.00 | 62.50 | 77.43 |
|                           | TDU    | 60.45 | 67.16 | 76.87 |
| Avg                       |        | 63.03 | 72.83 | 82.70 |

Table 10: Full results of SitEmb-v1.5-Qwen on book plot retrieval.
```


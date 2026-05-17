# ArXiv 2310.04408

### --- Page 0001 ---

```markdown
# RECOMP: IMPROVING RETRIEVAL-AUGMENTED LMs WITH COMPRESSION AND SELECTIVE AUGMENTATION

Fangyuan Xu¹, Weijia Shi², Eunsol Choi¹  
Department of Computer Science  
¹The University of Texas at Austin  
²University of Washington  
{fangyuan,eunsol}@utexas.edu, swj0419@cs.washington.edu

## ABSTRACT

Retrieving documents and prepending them in-context at inference time improves performance of language model (LMs) on a wide range of tasks. However, these documents, often spanning hundreds of words, make inference substantially more expensive. We propose compressing the retrieved documents into textual summaries prior to in-context integration. This not only reduces the computational costs but also relieves the burden of LMs to identify relevant information in long retrieved documents. We present two compressors – an extractive compressor which selects useful sentences from retrieved documents and an abstractive compressor which generates summaries by synthesizing information from multiple documents. Both compressors are trained to improve LMs’ performance on end tasks when the generated summaries are prepended to the LMs' input, while keeping the summary concise. If the retrieved documents are irrelevant to the input or offer no additional information to LM, our compressor can return an empty string, implementing selective augmentation. We evaluate our approach on language modeling task and open domain question answering task. We achieve a compression ratio of more than 6% with minimal loss in performance for both tasks, significantly outperforming the off-the-shelf summarization models. We show that our compressors trained for one LM can transfer to other LMs on the language modeling task and provide summaries largely faithful to the retrieved documents.¹

## 1 INTRODUCTION

Retrieval-augmented language models (RALMs) (Khandewal et al., 2019; Izacard et al., 2022; Lewis et al., 2020; Borgeaud et al., 2022) have shown impressive performance on knowledge-intensive tasks (Kwiatkowski et al., 2019; Petroni et al., 2021). Simply prepending retrieved documents to the input without adapting the language models (LMs) (Shi et al., 2023b; Ram et al., 2023; Si et al., 2022) allows retrieval augmentation even for black-box LMs, but such approach comes with limitations. First, it increases computational costs as LMs now encode substantially more tokens. Second, even if we manage to adapt LMs to efficiently incorporate longer context (Beltagy et al., 2020; Zaheer et al., 2020), these models struggle to use all information in the context, frequently missing information placed in the middle (Liu et al., 2023). Third, prepending a large number of documents in-context can further confuse LMs with irrelevant information, degrading model performance (Mallen et al., 2022; Shi et al., 2023a).

To overcome such limitations, we propose RECOMP (Retrieve, Compress, Prepend), an intermediate step for RALMs which compresses retrieved documents into a textual summary prior to in-context augmentation. Figure 1 illustrates our approach. The generated summary should be concise to maximize efficiency, be faithful to the retrieved evidence documents, and guide RALM to generate desired outputs when prepended to the input. To satisfy both efficiency and effectiveness constraints, our compressor strategically performs selective augmentation by generating an empty summary when the retrieved documents are irrelevant or unhelpful for target task.

¹Our code is available at [https://github.com/carriex/recomp](https://github.com/carriex/recomp).
```

### --- Page 0002 ---

```markdown
![An illustration of RECOMP, which compresses retrieved documents into a textual summary before prepending it as input to a language model at inference time. The compressed summary guides the LM to generate the correct answer, while significantly reducing the computation costs required to encode the documents.](assets/page_0002_img_1.png)

## 2 PROBLEM FORMULATION: RECOMP

Given an input sequence $x$, a target output sequence $y$ and a set of $N$ retrieved documents $D$ $(d_1, d_2, \ldots, d_N)$, RECOMP compresses retrieved documents $D$ with respect to $x$ into a summary $s$ which captures core information in $D$ relevant to $x$ with significantly fewer tokens than $D$. Our architecture consists of two modules: compressor $c$ and LM $M$. In this work, we assume a blackbox LM and train the compressor. Given the set of retrieved $N$ documents $(d_1, d_2, \ldots, d_N)$ and the input sequence $x$, a compressor returns a token sequence $s$. We design our compressor to be substantially smaller than LM $M$, as we aim to reduce computational costs of encoding a set of retrieved documents.

The output from compressor $s$ should be: (1) Concise: The summary should be as short as possible to optimize efficiency. If the retrieved documents do not contain relevant information or retrieval augmentation is not necessary, $s$ can be an empty sequence. (2) Effective: when $s$ is prepended to input sequence $x$ and provided to LM $M$ as a prompt, LM should generate the target output sequence $y$. (3) Faithful: $s$ should be a faithful and interpretable summary of the input document set (i.e., $s$ must be entailed by the input document set $(d_1, d_2, \ldots, d_N)$). We focus on training compressors for conciseness and effectiveness. We summarize the key ideas for our two compressors, extractive compressors and abstractive compressor here, and discuss their training schemes formally in Section 3.
```

### --- Page 0003 ---

```markdown
# Extractive Compressor

Given $n$ sentences $[s_1, s_2, \ldots, s_n]$ in the input document set $(d_1, d_2, \ldots, d_N)$, we train a dual encoder model $enc_g$ which embeds sentence $s_i$ and the input sequence $x$ into fixed-dimensional embeddings respectively. Their inner product represents how helpful it would be for the LM $M$ to prepend $s_i$ to the input $x$ to generate $y$. The final summary $y$ from the compressor will be a concatenation of top $N$ sentences ranked by their inner product with the input. As this approach is extractive, we assume the faithfulness criteria is mostly satisfied.³

# Abstractive Compressor

We train an encoder-decoder model $enc_{dec}$ to serve as an abstractive compressor, which takes the input sequence $x$ and a concatenation of retrieved document set $D$ $[d_1; d_2; \ldots; d_N]$ and output a summary $s$. Although we do not have human annotations to train this model, prior work (Goyal et al., 2022; Chen et al., 2023; Polturi et al., 2023) suggests that the extreme-scale LMs can generate good query-focused summaries when prompted carefully. Yet, using an extreme-scale model as the compressor is not desirable as we want the compressor to be substantially smaller than the LMs. Thus, we perform distillation (Hinton et al., 2015) of extreme-scale LMs to build a lightweight abstractive compressor $enc_{dec}$. We do not train specifically for faithfulness, but later manually evaluate the faithfulness in Section 6.

# 3 LEARNING THE COMPRESSORS

Our compressor resembles text summarization models in the output should be faithful to the original input, yet the main goal is different. Instead of capturing salient information for humans readers, compressors aim to produce a concise text that are useful for a LM on an end task. In this section, we describe how to train the extractive compressor (§3.1) and the abstractive compressor (§3.2) leveraging end task signals. Further training details can be found in the Appendix A.2.

## 3.1 EXTRACTIVE COMPRESSOR

As we formulate extractive compression as a ranking problem, training extractive compressors involves training a reranker for the retrieved documents with two differences. First, our compressor considers a different granularity of input (sentences) compared to the initial retrieval unit (paragraphs). Second, the sentence is evaluated based on whether it is useful as input for the LM on the downstream task (Shi et al., 2023b; Ram et al., 2023).

### Model

We train a dual-encoder model $enc_g$ which encodes the input context $x$ and the candidate sentence $s$ separately. We obtain an embedding of $x$ and $s$ by taking the representation of the $[CLS]$ token respectively, and compute their similarity by calculating the inner product of the two. We initialize our model with the encoder checkpoint (Izacard et al., 2021). This model consists of 110M parameters, satisfying the efficiency desideratum of compressor.

### Training

Figure 2 presents pseudocode for training an extractive compressor with contrastive loss for the language modeling task. For each input query $x$, we identify positive and negative sentences from retrieved documents.

For each pair of input sequence $x_i$ and candidate sentences $s_j$, we measure

Input: Base LM $M$, Compressor $enc_g$, Training data $\{x_i, s_j\}_{i,j}$ where $x_i$ is input, $S = \{s_j\}$ is a set of candidate sentences from the retrieved documents for $x_i$, $y$ is the target answer, and score threshold $\epsilon$.

Output: An updated extractive compressor encoder $enc_g$.

1. $T \gets \emptyset$
2. for $i \gets 1, \ldots, T$ do
3. $p_i \gets \text{argMax}_{s_j \in S} \text{Score}(M, y_i; [s_i; x_i])$
4. for $j \gets 1, \ldots, n$ do
5. $L \gets \emptyset$
6. if $\text{Score}(M, y_i; [s_j; x_i]) < \text{Score}(M, y_i; [p_i; x_i])$ then
7. $L \gets L \cup s_j$
8. if $|L| > 0$ then
9. $\mathcal{L} \gets \text{argTop}_{s_j \in (enc_g(x_i))}$
10. $T \gets T \cup (\{x_i, p_i, L\})$
11. $enc_g = \text{Finetune}(enc_g, T)$

![Learning an extractive compressor for language modeling task.](assets/page_0003_img_1.png)
```

### --- Page 0004 ---

```markdown
# 3.2.1 CREATING DATASET FOR DISTILLATION

**Score**$(M, y_i, [s_j; x_i]) = \log P_M(y_i \mid [s_j; x_i])$, log likelihood assigned to target output according to LM $M$ when candidate sentence is prepended to the input. We consider the sentences with the highest log likelihood as a positive example $P_i$ (line 3). To construct negative examples $N_i = \{n_k\}_{k=1}^K$, we choose up to five sentences with top contriver score that has the log likelihood lower than the positive sentence for a threshold (line 6).

Training a compressor for QA task works similarly, but scoring will evaluate whether the LM will generate the correct answer with summary prepended (change in line 6). Pseudocode for the QA tasks is in Figure 6 in the Appendix. We train our encoder with a contrastive loss (Karukhin et al., 2020a), maximizing the similarity between positive pairs $(x_i, p_i)$ and minimize negative pairs $(x_i, N_i)$. The training objective is to minimize:

$$
-\log \left( e^{\text{sim}(x_i, p_i)} \right) + \sum_{n \in N_i} e^{\text{sim}(x_i, n)}
$$

Data: For the language modeling task, we generate training data using the training split of the Wikitet-103 dataset, selecting the top 20 sentences from the top 5 BM25 retrieved documents for each input context $x$. For the QA tasks, we generate training data using the training split and consider the top 20 sentences from the top 5 contriever-ms-marco 5 retrieved documents. We report detailed statistics for the training data in Table 5 in the appendix. For each sentence from the retrieved documents, we prepend the Wikipedia page title to it to for decontextualization.

## 3.2 ABSTRACTIVE COMPRESSION

To train an abstractive compressor, we distill the query-focused summarization ability of existing scale LM by generating training dataset from it, filter the generated data, and train an encoder decoded from the filtered dataset (West et al., 2022). In contrast to prior work (Gun et al., 2023) which uses intrinsic summarization metric for filtering, we use the LM’s performance on the task to rank the generated summaries prepended for filtering. Fig. 3 presents pseudo algorithm for the abstractive compressor.

### Generation From Teacher Model

For the language modeling task, we manually construct prompt set $\{p_i\}$. Given an input $x_i$, a retrieved document $D_i$ and a prompt $p$ to summarize the document with respect to the input, GPT-3.5 generates a summary (line 3).

### Filtering with Critic

After generating a summary for each prompt template, we select the summary which results in the highest end task performance for each example ($s_t$) as the target summary (line 4-8). **Score**$(M, y_i, [s_j; x_i])$ is the same as the extractive compressor above. We then compare the end task performance with the target summary prepended and with input $x$ only (i.e., no retrieval) on base model $M$ (line 6). If the end task performance gets worse (e.g., increase in perplexity) when prepending the summary, we set the target summary to an empty string (line 7), otherwise we append the target summary to the training set (line 9). This allows for selective augmentation and mitigates the risk of prepending irrelevant documents.

``` 

![Learning an abstractive compressor for language modeling task.](assets/page_0004_img_1.png)
```

### --- Page 0005 ---

```markdown
Constructing training datasets for the question answering tasks works similarly, with the following modifications. As summarization for the question answering task is more straightforward, we use a single prompt for each dataset. We filter out examples where prepending the summary does not lead to performance improvement. Pseudo code for the QA tasks is in Figure 7 in the Appendix.

## Model & Training
We use encoder-decoder LM (775M), initialized from T5-large checkpoint (Raffel et al., 2020). This model has been trained with summarization datasets (Hermann et al., 2015).

## Data
We summarize top 5 retrieved documents for both language modeling and question answering tasks. We generate training examples using 2% of the training set for the WikiText-103 dataset. We generate training examples from the entire NQ training set and TriviaQA training set. For HotpotQA, we only generate summaries for the training data where the gold answer is in the retrieved documents (56% of the training data) to reduce API costs. We report percentage of data filtered and percentage of empty summaries in Table 5 in A.1.

# 4 EXPERIMENTAL SETTINGS
We evaluate our approach on language modeling and open-domain QA following prior work (Shi et al., 2023b; Ram et al., 2023). For both tasks, we report the task performance as a measure of effectiveness and the number of tokens provided in context as a measure of efficiency.

## 4.1 LANGUAGE MODELING
We evaluate language modeling perplexity on WikiText-103 (Merity et al., 2016) benchmark on three open-sourced LMs of varying scale: GPT2 (117M), GPT2-XL (1.5B; Radford et al. (2019)) and GPT-J (6B; Wang & Komatsuzaki (2012)). We train our compressors using GPT2 as the base model and evaluate whether the trained compressor transfer to GPT2-XL and GPT-J. We use the BM25 retriever (Robertson & Zaragoza, 2009) to retrieve from the Wikipedia corpus for DCP. The articles are then truncated into non-overlapping documents of 100 words. During retrieval, articles containing the input sequence x is removed from the corpus to prevent data contamination. Following Ram et al. (2023), we perform retrieval every 32 tokens.

## 4.2 OPEN-DOMAIN QA
### Datasets
We evaluate our model on three benchmark datasets: Natural Questions (NQ) (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017) and HotpotQA (Yang et al., 2018). We report results on development set of NQ, test set of TriviaQA and randomly sampled 500 examples from HotpotQA development set. We report Exact Match (EM) and F1token-level F1 of answer strings to measure end task performance.

### Base Language Models & Retrieval Corpus
We use Flan-UL2 (20B)(Chung et al., 2022), a large scale instruction-tuned LM. We use contriever model trained on MS MARCO dataset (Campos et al., 2016) as a retriever on Wikipedia corpus from Dec. 20, 2018 for all three datasets. The articles are truncated into non-overlapping documents of 100 words.

### Prompt Format
We include few-shot in-context examples in the prompt, followed by the retrieved documents and the question. We use randomly sampled training examples as in-context examples, which constitutes 110, 147, and 149 tokens on average for NQ, TQA and HotpotQA respectively. For retrieved documents, we concatenate them in ascending order of retrieval score, with the highest scored document closest to the question (Si et al., 2022). We do not include the retrieved documents in in-context examples as it did not improve performance. An example input can be found in Appendix Table 7.

## 4.3 BASELINES AND ORACLES
### Baselines
We first consider two heuristic token and phrase-level compression methods: BoW, which converts the retrieved documents to a list of ordered unigrams and concatenates them together.
```

### --- Page 0006 ---

```markdown
| Table 1: Results on language modeling task. We report results on GPT-2, GPT2-XL and GPT-J with compressors trained with GPT-2. |
|-------------------------------------------------------------------------------------------------------------------------------------|
| **In-Domain**                | **Out-Domain**                |
| **GPT2 (117M)**             | **GPT2-XL (1.5B)**           | **GPT-J (6B)**                |
| **In-context Evidence**      | # tokens | PPL  | # tokens | PPL  | # tokens | PPL  |
|------------------------------|---------|------|----------|------|----------|------|
| -                            | 0       | 37.84| 0        | 19.89| 0        | 11.44|
| **RALM without compression** |         |      |          |      |          |      |
| Top 1 document               | 141     | 32.90| 14       | 17.86| 141      | 10.57|
| Top 5 documents              | 512     | 35.53| -        | -    | -        | -    |
| **Phrase/Token level compression** |   |      |          |      |          |      |
| Top 1 document (BoW)        | 66      | 36.13| 66       | 18.85| 66       | 10.97|
| Top 1 document (NE)         | 34      | 37.23| 33       | 19.67| 33       | 11.39|
| **Extractive compression of Top 5 documents (select top 1 sentence)** | | | | | |
| Oracle (w/ gpt2)            | 32      | 30.36| 32       | 16.58| 31       | 9.92 |
| Oracle (w/ gpt2)            | 32      | 30.36| 32       | 16.99| 32       | 10.22|
| Random                       | 27      | 36.98| 27       | 19.55| 27       | 11.32|
| BM25                         | 33      | 36.68| 33       | 19.02| 33       | 11.08|
| Contriever                   | 33      | 35.54| 33       | 18.98| 33       | 11.05|
| Ours (init. w/ Contriever)  | 31      | 33.67| 31       | 18.19| 31       | 10.73|
| **Abstractive compression of Top 5 documents** | | | | | |
| Oracle                       | 68      | 30.67| 68       | 16.87| 65       | 10.10|
| Oracle (w/ gpt2)            | 68      | 30.67| 68       | 17.23| 68       | 10.37|
| GPT-3                        | 33      | 34.84| 33       | 18.70| 33       | 10.96|
| T5                           | 15      | 37.80| 15       | 19.92| 15       | 11.15|
| Ours (init. w/ T5)         | 13      | 33.64| 15       | 18.09| 15       | 10.66|

---

**Named Entities** (NE), which extracts a list of ordered named entities from retrieved documents and concatenates them. For the extractive compressor on the language modeling task, we use BM25 and Contriever [Lazaridou et al., 2021], which rank the sentences by their similarity to the input $x$ baselines. For the QA datasets, we report results using BM25, Contriever finetuned on MS MARCO and DPR (Karpukhin et al., 2020b) fine-tuned on NQ. We also report a Random baseline which randomly selects a sentence from the retrieved documents. For abstractive compression, we report the performance of the off-the-shelf T5 (large, 770M) model and that of GPT-3.5 model. As we experimented with multiple prompts for the language modeling task, we report the performance of the summaries generated by GPT-3.5 model with the best single prompt.

**Oracle** We explore the performance upper bound of compression by considering two oracle approaches. For the extractive approach, we construct oracle compressor by considering all sentences $s_i$ in the evidence document set and choosing the sentence that leads to the best end task performance (i.e., lowest perplexity or highest answer accuracy) for each example. For the abstractive approach, we consider summaries generated from different prompts ($\{s_j\}_{j=1}^n$ in Figure 3) and empty summary, and choose the one that leads to the best end task performance. As oracle compression is model dependent, we also report model-independent results by always using GPT-2 as a reference LM (Oracle w/ gpt2) to test how well oracle sentences for one model transfer to other models for the language modeling task.

---

### 5 RESULTS

**Language modeling** Table 1 reports the results on language modeling task. All retrieval augmentation methods improve perplexity over no retrieval setting across three LMs. Heuristic token / phrase-level compression methods (BoW and NE) are worse than prepending uncompressed documents, potentially due to the disfluency of the prepended text.

Both oracle settings show substantial gain over prepending the entire document set, with only 6-13% of tokens. More tokens are not always better: prepending top 1 document outperforms prepending top 5 documents. This confirms that the naive retrieve-and-prepend approach has a significant room for improvements, as prepending irrelevant documents can hurt performances.
```

### --- Page 0007 ---

```markdown
| Table 2: Open-domain QA results with Flan-UL2 (20B) as the LM M. We report number of tokens provided as in-context evidence document, excluding the in-context examples. We train separate compressors (one extractive, one abstractive) for each dataset. Extractive compressor selects one sentence for NQ/TQA, and two sentences for HotpotQA. |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

| In-Context evidence | # tok | NQ EM | F1 | TQA EM | F1 | HotpotQA EM | F1 |
|---------------------|-------|--------|----|---------|----|-------------|----|
| -                   | 0     | 21.99  | 29.38 | 49.33   | 54.85 | 0           | 17.80 | 26.10 |
| RALM without compression | Top 1 documents | 132 | 33.07 | 41.45 | 136 | 57.84 | 64.94 | 138 | 28.80 | 40.58 |
|                     | Top 5 documents | 660 | 39.39 | 42.88 | 677 | 62.37 | 70.09 | 684 | 32.80 | 43.90 |
| Phraselevel compression | Top 5 documents (NE) | 338 | 23.60 | 31.02 | 128 | 54.96 | 61.19 | 157 | 22.20 | 31.89 |
|                     | Top 5 documents (Bow) | 450 | 28.48 | 36.84 | 259 | 58.16 | 65.15 | 255 | 25.60 | 36.00 |

| Extractive compression of top 5 documents | # tok | NQ EM | F1 | TQA EM | F1 | HotpotQA EM | F1 |
|-------------------------------------------|-------|--------|----|---------|----|-------------|----|
| Oracle                                     | 34    | 60.22  | 64.25 | 32 | 79.29 | 82.06 | 70 | 41.80 | 51.07 |
| Random                                     | 32    | 23.27  | 21.09 | 31 | 50.18 | 56.24 | 61 | 21.00 | 29.86 |
| BM25                                       | 26    | 25.82  | 33.68 | 37 | 54.67 | 61.74 | 29.60 | 28.02 | 30.82 |
| DPR                                        | 39    | 34.32  | 43.81 | 56 | 68.62 | 78.07 | 28.13 | 29.80 | 38.15 |
| Contriver                                  | 36    | 30.06  | 31.92 | 40 | 53.67 | 60.01 | 78 | 28.50 | 39.48 |
| Ours                                       | 37    | 36.57  | 44.22 | 38 | 58.99 | 65.26 | 75 | 30.40 | 40.14 |

| Abstractive compression of top 5 documents | # tok | NQ EM | F1 | TQA EM | F1 | HotpotQA EM | F1 |
|---------------------------------------------|-------|--------|----|---------|----|-------------|----|
| Oracle                                       | 51    | 45.68  | 53.66 | 37 | 71.01 | 76.38 | 102 | 35.80 | 46.25 |
| GPT-3.5                                     | 56    | 37.12  | 36.35 | 41 | 62.03 | 69.66 | 107 | 31.60 | 42.65 |
| Ours                                        | 10    | 25.30  | 36.74 | 55 | 58.12 | 62.34 | 29 | 23.19 | 37.99 |
| Ours                                        | 37    | 37.04  | 45.47 | 32 | 58.68 | 66.34 | 64 | 28.20 | 37.91 |

Our trained extractive compressor significantly outperforms other extractive baselines (Contriver and BM25) across all three LMs, while prepending slightly fewer tokens. Comparing to prepending one document, we achieve a compression ratio of 25% at minimum performance drop. Our trained abstractive compressor performs the best across the board, achieving the lowest perplexity and the highest compression ratio. Our abstractive compressor achieves high compression rate through selective augmentation, prepending summaries to only 33% of examples (length distribution of generated summaries is in Fig. 8).

Open-domain QA
We report the results on QA tasks in Table 2. Similar to the language modeling task, all retrieval augmentation methods improve performance over no retrieval setting, across three datasets, consistent with previous study on other LMs (Shi et al., 2023b; Mahlen et al., 2022; Si et al., 2022). Unlike language modeling, prepending five documents shows significant gains over prepending a single document, motivating the use of compression to incorporate more contexts.

We find that extractive oracle outperforms the abstractive one in all datasets. Extractive oracle selects the best one from N candidate sentences, while abstractive oracle tests two options – prepending GPT-3.5 summary or prepending nothing. Both oracles show improvements over prepending all information, suggesting that removing irrelevant information benefit the model. 

Among extractive baselines, DPR performs the best as it has been fine-tuned on high-quality NQ data. On NQ, selecting the top 1 DPR ranked sentences from top 5 documents outperforms prepending top 1 document, with much fewer tokens (39 vs. 132). However, its performance degrades in out of domain datasets. Off-the-shelf summarization model (T5) boasts the highest level of compression, achieving 4-6 points gains in EM while adding mere 7-10 tokens.

The trained compressors, both extractive and abstractive, shows promising performances. On NQ and TQA, the abstractive approach is more effective. On NQ, it achieves a compression ratio of 5% tokens while losing 2 EM points compared to prepending full documents. On TQA, we observe similar trends, compression ratio of 50 tokens while losing 3.7 EM points compared to prepending full sets of documents. On HotpotQA that requires multihop understanding of documents, we 

% We provide an example where our compressed summary yields correct answer while prepending full document does not in Table 9 in the appendix.
```

### --- Page 0008 ---

```markdown
![Histogram of abstractive summary length (# tokens) distribution.](assets/page_0008_img_1.png)

# 6 ANALYSIS AND DISCUSSIONS

## Transferring Across Different LMs
One benefit of textual summary is that they can transfer to other LMs, unlike approaches such as soft prompts (Wingate et al., 2022; Chevalier et al., 2023; Mu et al., 2023). We evaluate whether our compressors trained to achieve high performance with respect to a specific LM (GPT2 for language modeling, FlanT5 for open domain QA) can transfer to other LMs. For language modeling, we find that trained compressor transfers well to other LMs (GPT2-LM and GPT-J), despite they are larger LMs (Table 1). For open domain QA, we test transferring our compressors to LLaMA-13B (Touvron et al., 2023) model. The results can be found in Table 10 in the appendix. Overall, the performance is worse than the LM from which compressors are trained on, sometimes unable to outperform other compression baselines (e.g., no clear gain from using contriver vs. our trained contriver on TQA/HotpotQA), leaving considerable gap to the oracle compressions for LLaMA itself. Yet, on NQ/TQA, our compressor obtains 5% compression ratio with less than 5 EM drop compared to full document setting, showing the robustness of our retrieve-compress-prepare paradigm.

## How do the length of the summaries vary?
Can the learned compressor reliably determine when LMs require irrelevant documents or not? As evidenced below, we found that the model performances for some input queries, 4-24% of training examples for abstractive compressors contain empty summary. Fig. 4 presents the length distribution of abstractive summaries on NQ and Wikitet. Histograms for other datasets is in Fig. 8 in the appendix. The input document lengths do not vary significantly across summary length significantly in length, suggesting abstractive compressor enables selective retrieval augmentation. We have not experimented selective compression with extractive compressor, fixing the number of prepended sentences for the entire dataset summarizer can be a promising direction for future work.

## How does model leverage the in-context documents?
We evaluate whether retrieval augmented LMs tend to copy answers verbatim from in-context evidence documents or generate answers not present in the documents. This is a desired behavior only when the gold answer is in the evidence. We first report how frequently a gold answer span is present in evidence text (% Gold in Evi). As expected, full documents contain the answer most frequently, followed by NE and GPT-3. However, having more gold answers in the evidence doesn’t equate to better performance, as the model cannot always identify the correct answer from the evidence (84 % for NE vs. 98% for T5(ours)).

We observe that model can be easily distracted by irrelevant contexts, copying from a document span even when it does not contain a gold answer, echoing findings from prior work (Shi et al., 2022a). Prepending top 5 documents has 

| Evidence | EM  | % Gold in Evi. | % Pred in Evi. |
|----------|-----|----------------|-----------------|
| Top 1    | 33.1| 92 / 51        |                 |
| Top 5    | 39.3| 97 / 96        |                 |
| NE       | 26.0| 84 / 48        |                 |
| Oracle   | 60.2| 34 / 93        |                 |
| Contriver| 30.2| 25 / 88        |                 |
| Ours     | 36.6| 28 / 90        |                 |
| GPT-3.5  | 37.1| 45 / 98        |                 |
| T5       | 25.9| 30 / 52        |                 |
| Ours     | 37.0| 34 / 98        |                 |
```

### --- Page 0009 ---

```markdown
# Page 9

a higher frequency (81%) of copying incorrectly compared to top 1 document (51%), and GPT-3's compression leads to an even higher incorrect copying frequency (85%), potentially as query-focused summarization generates sentences that seemingly contains the answer. Our compressor successfully reduces such erroneous behavior to 39%.

## Is generated summary faithful and comprehensive? 

We (the authors) manually evaluate outputs of the abstractive compressors on two axes (Chen et al., 2023): **Faithfulness**: whether the summary can be enabled by the retrieved documents, **Comprehensiveness**: whether the summary contains sufficient information to answer the question, regardless of whether the generated information comes from the retrieved documents. For both, we select one of three labels: Yes, Partially, No, and report the % of useful summaries which are both faithful and comprehensive. Annotation sample can be found in Table 11 in the appendix. We evaluate the summaries generated by GPT-3.5 and our abstractive compressor. We randomly sample 30 non-empty summaries from the test set.

### Table 4: Manual analysis on abstractive summaries generated for NQ, TQA and HotpotQA (HQA) dataset.

| Dataset | Model   | % Faithful |  Y  |  P  |  N  | % Compre. |  Y  |  P  |  N  | % Use. |
|---------|---------|------------|-----|-----|-----|-----------|-----|-----|-----|--------|
| NQ      | GPT-3.5 | 90         |  0  | 10  | 97  |  0        |  3  | 83  |  0  |  80    |
| NQ      | Ours    | 80         | 13  |  7  |100  |  0        |  0  | 80  |  0  |  80    |
| TQA     | GPT-3.5 | 97         |  0  |  3  | 90  |  0        |  1  | 83  |  0  |  83    |
| TQA     | Ours    | 83         |  3  |  4  | 96  |  0        |  4  | 77  |  0  |  77    |
| HQA     | GPT-3.5 | 74         |  0  | 26  | 78  |  0        |  22 | 50  |  0  |  40    |
| HQA     | Ours    | 67         |  0  | 33  | 74  |  0        |  26 | 40  |  0  |  40    |

7. **Related Work**

**Efficient RAML** He et al. (2021) improves efficiency of RAMLs by improving retrieval components, such as data store compression, dimensionality reduction for neural retrieval. A line of work also introduces reducing retrieval frequency through selective retrieval (He et al., 2021; Mallen et al., 2022) or using a larger stride (Martins et al., 2022). In this work, we improve efficiency of RAML by compressing retrieved documents into a concise summary or an empty sequence, facilitating selective retrieval augmentation.

**Prompt Compression** Recent work (Wingate et al., 2022; Chevalier et al., 2023; Mu et al., 2023) proposes compressing long contexts into summary vectors (soft prompts) that can be used by LMs, rather than shorter textual summaries. Such soft prompts can serve as efficient replacements for plain-text demonstrations, minimizing the computational costs during inference. Another related line of work proposes context distillation (Snell et al., 2022; Choi et al., 2022; Padmanabhan et al., 2023), which injects the prepended context into the parameters of an LM. Compared to above approaches, our approach yields more interpretable textual summary that can transfer across different LMs, and can be applied to black box LMs without requiring gradient updates. Prior work has studied textual compression for other tasks, such as political fact checking (Chen et al., 2023) and instruction learning (Yin et al., 2023).

**Distillation / Goal Oriented Summarization** Recent work introduces symbolic knowledge distillation (West et al., 2022), which transfers knowledge from a teacher model by generating a training dataset with the teacher model and train a student model on it. For better performance, they introduce strict criteria, which filter undesirable examples from generated training dataset. Such distillation technique has been applied for various applications including summarization (Jung et al., 2023), which aims to generate high quality summaries while we optimize for generating effective summary for downstream LMs. One work that is similar to our setting is Hsu & Tan (2021) which trains an
```

### --- Page 0010 ---

```markdown
# 8 CONCLUSION

We introduce RECOMP, a method which compresses retrieved documents into textual summaries before prepending them to improve in-context retrieval augmented language models. We present two compression models – an extractive compressor and an abstractive compressor. We design a training scheme which leverages end task signals from a blackbox LM to generate useful summaries and allowing the compression models to perform selective augmentation. Our experiments show that our compressors can improve the efficiency of retrieval augmented LMs significantly with minimal drop in performances.

## ACKNOWLEDGEMENT

We thank the members of the UT and UW NLP community for feedback on the project. We especially thank Alisa Liu, Junyi Jessy Li and Greg Durrett for providing comments on the draft. The project is partially funded by NSF grant (IIS-2312948).

## ETHICS STATEMENT

We use commercial language model to generate training data for our compressors, which might include factual error. We conduct careful human evaluation on the data generated and present our analysis in the paper.

## REPRODUCIBILITY STATEMENT

We release our codes, prompt, and data generated with API access publicly.

## REFERENCES

1. Tz Belavkin, Matthew E. Peters, and Arman Cohen. Longformer: The long-document transformer. arXiv, 2020. URL https://api.semanticscholar.org/CorpusID: 215737171.

2. Sebastian Borgwardt, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Milican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Henighan, Saffron Huang, Loron Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of tokens. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 2206–2240. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/borgeaud22a.html.

3. Daniel Fernando Campos, Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Ranju Majumder, Li Deng, and Bhaskar Mitra. Marco: A human generated machine reading comprehension dataset. arXiv, 2016. URL https://api.semanticscholar.org/CorpusID: 1289517.

4. Jifan Chen, Grace Kim, Aniruddh Sriram, Greg Durrett, and Eunsol Choi. Complex claim verification with evidence retrieved in the wild. ArXiv, 2023. URL https://api.semanticscholar.org/CorpusID: 258282852.

5. Alexis Chevalier, Alexander Wettig, Aniruddh Ajith, and Danqi Chen. Adapting language models to compress contexts. arXiv preprint arXiv:2305.14788, 2023.
```


### --- Page 0011 ---

```markdown
| Author(s)                                                                 | Title                                                                                                   | Source                                                                                          |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Eunbi Choi, Yongrae Jo, Joel Jang, and Minjoon Seo                      | Prompt injection: Parameterization of fixed inputs.                                                    | ArXiv, https://api.semanticscholar.org/CorpusID:249953762.                                   |
| Hyung Woo Chung, Le Hou, S. Longpre, Barret Zoph, Yi Tay, William Fedus, | Scaling instruction-finetuned language models.                                                         | ArXiv, https://arxiv.org/abs/2210.11416, 2022.                                               |
| Eric Li, Kxueh Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, |                                                                                                         |                                                                                                 |
| Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Akanksha Chowdhery, |                                                                                                         |                                                                                                 |
| Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Wei Yu, Vincent Zhao, |                                                                                                         |                                                                                                 |
| Yanping Huang, Andrew M. Dai, Hongkun Yu, Slay Fetrov, Ed Huai hsin Chi, |                                                                                                         |                                                                                                 |
| Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei |                                                                                                         |                                                                                                 |
| Tanya Goyal, Junyi Jessy Li, and Greg Durrett.                          | News summarization and evaluation in the era of gpt-3.                                               | arXiv preprint arXiv:2209.12356, 2022.                                                        |
| Junxian He, Graham Neubig, and Taylor Berg-Kirkpatrick.                 | Efficient nearest neighbor language models.                                                             | In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5703–5714, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi:10.18653/v1/2021.emnlp-main.461. URL: https://aclanthology.org/2021.emnlp-main.461. |
| Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, | Teaching machines to read and comprehend.                                                              | ArXiv, https://api.semanticscholar.org/CorpusID:6203757.                                     |
| Will Kay, Mustafa Suleyman, and Phil Blunsom.                           |                                                                                                         |                                                                                                 |
| Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean.                    | Distilling the knowledge in a neural network.                                                         | ArXiv, https://api.semanticscholar.org/CorpusID:7200347.                                     |
| Matthew Honnibal, Ies Montani, Sophie Van Landeghem, and Boyd Adriaens. | spaCy: Industrial-strength natural language processing in Python, 2020.                               |                                                                                                 |
| Chao-Chun Hsu and Chenhao Tan.                                          | Decision-focused summarization.                                                                         | In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 117–132, Online, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi:10.18653/v1/2021.emnlp-main.10. URL: https://aclanthology.org/2021.emnlp-main.10. |
| Gautier Izacard, Mathieu Caron, Lucas Hosseni, Sebastian Riedel,        | Unsupervised dense information retrieval with contrastive learning.                                     | ArXiv, https://arxiv.org/abs/2112.09118, 2021.                                               |
| Piotr Bojanowski, Armand Joulin, and Edouard Grave.                     |                                                                                                         |                                                                                                 |
| Gautier Izacard, Patrick Lewis, Maria Lomelli, Lucas Hosseni, Fabio Petroni, | Few-shot learning with retrieval augmented language models.                                            | ArXiv, https://arxiv.org/abs/2208.02392, 2022.                                               |
| Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.                 |                                                                                                         |                                                                                                 |
| Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer.         | TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension.              | arXiv preprint 1705.03551, 2017.                                                               |
| Jaehoon Jung, Peter West, Liwei Jiang, Faeez Brahman, Ximing Lu,        | Impossible distillation: from low-quality model to high-quality dataset & model for summarization and paraphrasing. | arXiv preprint arXiv:2305.16535, 2023.                                                        |
| Jillian Fisher, Taylor Sorensen, and Yejin Choi.                        |                                                                                                         |                                                                                                 |
| Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu,   | Dense passage retrieval for open-domain question answering.                                            | In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781, Online, November 2020a. Association for Computational Linguistics. doi:10.18653/v1/2020.emnlp-main.550. URL: https://aclanthology.org/2020.emnlp-main.550. |
| Sergey Edunov, Danqi Chen, and Wen tau Yih.                             |                                                                                                         |                                                                                                 |
| Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu,   | Dense passage retrieval for open-domain question answering.                                            | In Conference on Empirical Methods in Natural Language Processing, 2020b. URL: https://api.semanticscholar.org/CorpusID:215737187. |
| Sergey Edunov, Danqi Chen, and Wen tau Yih.                             |                                                                                                         |                                                                                                 |
```

### --- Page 0012 ---

```markdown
| Authors                                                                 | Title                                                                                                   | Source                                                                                                   |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. | Generalization through memorization: Nearest neighbor language models.                                 | ArXiv, abs/1911.00172, 2019.                                                                             |
| Diederik P. Kingma and Jimmy Ba.                                       | Adam: a method for stochastic optimization.                                                             | CoRR, abs/1412.6980, 2014.                                                                               |
| Tom Kwiatkowski, Jenimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. | Natural questions: A benchmark for question answering research.                                        | Transactions of the Association for Computational Linguistics, 7:452–466, 2019. URL: https://aclweb.org/anthology/Q19-1026. |
| Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpuinkin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. | Retrieval-augmented generation for knowledge-intensive tasks.                                          | ArXiv, abs/2011.11401, 2020.                                                                             |
| Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. | Lost in the middle: How language models use long contexts.                                             | ArXiv preprint arXiv:2307.03172, 2023.                                                                   |
| Alex Mallen, Akari Asai, Victor Asai, Rajarshi Das, Hannah Hajishirzi, and Daniel Khashabi. | When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories. | ArXiv, abs/2212.10511, 2022.                                                                             |
| Pedro Henrique Martins, Zita Marinho, and André F. T. Martins.         | Chunk-based nearest neighbor machine translation.                                                        | In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 4228–4245, Abu Dhabi, United Arab Emirates, December 2022. URL: https://aclweb.org/anthology/2022.emnlp-main.284. |
| Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher.    | Pointer sentinel mixture models.                                                                         | ArXiv, abs/1609.07843, 2016.                                                                             |
| Jesse Mu, Xiang Lisa Li, and Noah D. Goodman.                          | Learning to compress prompts with gists.                                                                | ArXiv, abs/2304.08467, 2023. URL: https://api.semanticscholar.org/CorpusID: 258179012.                  |
| Shankar Ramabhadran, Yasumasa Onoe, Michael J.Q. Zhang, Greg Durrett, and Eunsol Choi. | Propagating knowledge updates to lms through distillation.                                              | ArXiv, abs/2306.09306, 2023. URL: https://api.semanticscholar.org/CorpusID: 259165330.                 |
| Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicolo De Cao, James Thorne, Yacine Jernite, Vladimir Karpuinkin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. | KILT: a benchmark for knowledge intensive language tasks.                                               | In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2523–2544, Online, June 2021. URL: https://aclweb.org/anthology/2021.naacl-main.200. |
| Abhilash Potluri, Fangyuan Xu, and Eunsol Choi.                       | Concise answers to complex questions: Summarization of long-form answers.                              | In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9709–9728, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.541. URL: https://aclweb.org/anthology/2023.acl-long.541. |
| Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. | Language models are unsupervised multitask learners.                                                    | 2019.                                                                                                    |
| Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. | Exploring the limits of transfer learning with a unified text-to-text transformer.                       | The Journal of Machine Learning Research, 21(1):5485–5551, 2020.                                       |
```

### --- Page 0013 ---

```markdown
| Author(s)                                                                 | Title                                                                                                           |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Ori Ram, Yoav Levine, Itay Dalmedigo, Dor Muhlgy, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. | In-context retrieval-augmented language models. ArXiv, abs/2302.00083, 2023.                                   |
| Nils Reimers and Iryna Gurevych.                                         | Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3982–3992, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1410. URL https://aclanthology.org/D19-1410. |
| Stephen E. Robertson and Hugo Zaragoza.                                   | The probabilistic relevance framework: Bm25 and beyond. Found. Trends Inf. Retr.; 3:333–389, 2009.            |
| Chantal Shaib, Millicent Li, Sebastian Joseph, Ian Marshall, Junyi Jessy Li, and Byron Wallace. | Summarizing, simplifying, and synthesizing medical evidence using GPT-3 (with varying success). In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 1387–1407, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-short.119. URL https://aclanthology.org/2023.acl-short.119. |
| Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Huai hsien Chi, Michael Scharli, and Denny Zhou. | Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning, 2023. URL https://api.semanticscholar.org/CorpusID:256459776. |
| Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoo Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. | Replug: Retrieval-augmented black-box language models. ArXiv, 2023. doi: 10.2307/12652.2023. |
| Chengli Si, Zhe Gan, Zhenyuan Yang, Shuanghong Wang, Jianfeng Wang, Jordan L. Boyd-Graber, and Lijuan Wang. | Prompting gpt-3 to be reliable. ArXiv, abs/2210.91500, 2022.                                                   |
| Charles Burton Snell, Dan Klein, and Ruiji Zhong.                        | Learning by distilling context. ArXiv, abs/2209.15189, 2022. URL https://api.semanticscholar.org/CorpusID:252668389. |
| Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahari, Yasmin Babaei, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargav, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cristian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esibou, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanj Goswami, Roman Goyal, Anthony S. Harthorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marinka Kardas, Viktor Kerekez, Mikhail Lavril, Isabel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Dima Liskovich, Yining Liu, Puning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Mylobog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rishab Rungta, Kalyan Saladi, Alan Schelten, Rauna Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Ilyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. | Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288, 2023. URL https://api.semanticscholar.org/CorpusID:259590998. |
| Ben Wang and Aran Komatsuzaki.                                           | GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021. |
| Peter West, Chandra Bhargavataula, Jack Hessel, Jena Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, and Yejin Choi. | Symbolic knowledge distillation: from general language models to commonsense models. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4602–4607, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.341. URL https://aclanthology.org/2022.naacl-main.341. |
```

### --- Page 0014 ---

```markdown
David Wingate, Mohammad Shoeybi, and Taylor Sorensen. Prompt compression and contrastive conditioning for controllability and toxicity reduction in language models. In 

| Findings of the Association for Computational Linguistics: EMNLP 2022, | Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| pp. 5621–5634. doi: 10.18653/v1/2022.findings-emnlp.412. URL https://aclanthology.org/2022.findings-emnlp.412. |                                                                                          |

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. Huggingface’s transformers: State-of-the-art natural language processing. ArXiv, abs/1910.03771, 2019.

Yumo Xu and Mirella Lapata. Coarse-to-fine query focused multi-document summarization. In 

| Conference on Empirical Methods in Natural Language Processing, 2020. | URL https://api.semanticscholar.org/CorpusID:226262229. |
|--------------------------------------------------------------------------|----------------------------------------------------------|

Zhiling Yang, Peng Qi, Saizheng Zhang, Yosua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In 

| Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018. | |
|--------------------------------------------------------------------------------| |

Fan Yin, Jesse Yigit, Philippe Laban, Shafiq R. Joy, Caiming Xiong, and Chien-Sheng Wu. Did you read the instructions? rethinking the effectiveness of task definitions in instruction learning. In 

| Annual Meeting of the Association for Computational Linguistics, 2023. | URL https://api.semanticscholar.org/CorpusID:259063796. |
|--------------------------------------------------------------------------|----------------------------------------------------------|

Manzil Zaheer, Guru Gururangan, Kumar Avinav Dubey, Joshua Ainslie, Chris Alberti, Santiago Onofrio, Philip Pham, Anirudh Raval, Qifan Wang, Li Yang, and Andrew M. Big bird: Transformers for longer sequences. ArXiv, abs/2007.14062, 2020. URL https://api.semanticscholar.org/CorpusID:220831004.

Shiyue Zhang, David Wan, and Mohit Bansal. Extractive is not faithful: An investigation of broad unfaithfulness problems in extractive summarization. ArXiv, abs/2209.03549, 2022. URL https://api.semanticscholar.org/CorpusID:252118883.
```

### --- Page 0015 ---

```markdown
![Data distribution on NQ dev set, TriviaQA dev set, and HotpotQA dev set](assets/page_0015_img_1.png)

# APPENDIX

## A.1 COMPRESSOR TRAINING DATA GENERATION

We report the statistics of the data used to train compressors in Table 5. We use SpaCy (Honnibal et al., 2020) to extract named entities.

**Extractive Data Generation**  
We generate data using the training data for the four datasets we tested (Wikitext, NQ, TQA and HotpotQA). We use the NLTK package to perform sentence splitting. We remove examples without any negatives.

**Abstractive Data Generation**  
We report prompt used to generate summaries in Table 8. We queried the OpenAI API with temperature of 0.7 and top p = 1. For the language modeling task, we use an ensemble of four prompts and choose the one which leads to the lowest perplexity as the target. If none of the summaries lead to perplexity decrease, we treat an empty summary as target. We queried the OpenAI API with temperature of 0.7 and top p = 1. We generate four summaries per example for randomly sampled 2% of the training data (48,013 examples).

## A.2 COMPRESSOR TRAINING DETAILS

**Extractive Compressor**  
For language modeling, we use the contriever checkpoint trained with unsupervised data. For the QA tasks, we use the contriever checkpoint fine-tuned on the MSMARCO task (Campos et al., 2016)¹⁰, following prior work (Si et al., 2022; Shi et al., 2023b). We implement the model using the Transformers (Wolf et al., 2019) and the sentence-transformer library (Reimers & Gurevych, 2019). We train with Adam optimizer (Kingma & Ba, 2014), using a batch size of 64, learning rate of 2e-5 and 1000 warmup steps for 3 epochs. We report results on the model with the best ranked perplexity on our validation set for the language modeling task and the best reranked accuracy for the QA tasks.

¹⁰ [https://huggingface.co/facebook/contriever](https://huggingface.co/facebook/contriever)  
¹⁰ [https://huggingface.co/facebook/contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco)
```

### --- Page 0016 ---

```markdown
| Input: Base LM $M$, Compressor encoder $enc_c$, Training data $\{(x_i, S_i, y_i)^T\}$ where $x_i$ is input, $S_i$ is a set of candidate sentences from the retrieved document for $x_i$, $y_i$ is the target answer. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Output: An updated extractive compressor encoder $enc_c$                                                                                                                                         |
| 1: $T \gets \emptyset$                                                                                                                                                                           |
| 2: for $i \in \{1, \ldots, T\}$ do                                                                                                                                                                |
| 3: $p_i \gets \arg\max_{s_j \in S_i} Score(M, y_i | [s_j; x_i])$                                                                                                                               |
| 4: for $j \in \{1, \ldots, n\}$ do                                                                                                                                                                 |
| 5: $l \gets \emptyset$                                                                                                                                                                            |
| 6: if $Score(M, y_i | [s_j; x_i]) < Score(M, y_i | [p_i; x_i])$ then                                                                                                                           |
| 7: $l \gets l \cup s_j$                                                                                                                                                                           |
| 8: if $|l| > 0$ then                                                                                                                                                                              |
| 9: $N_i \gets \arg\top_{s_j \in S_i \setminus (enc_c(s_i), enc_c(x_i))}$                                                                                                                       |
| 10: $T \gets T \cup \{(x_i, p_i, l)\}$                                                                                                                                                            |
| 11: $enc_c \gets Finetune(enc_c, T)$                                                                                                                                                             |

| Input: Teacher LM $M_t$, Base LM $M$, Summarization prompt $p$, Compressor $enc_{c\theta}$, Training data $\{(x_i, D_i, y_i)^T\}$ where $x_i$ is input, $D_i$ is the set of retrieved document for $x_i$, $y_i$ is the target answer. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Output: An updated $enc_{c\theta}$                                                                                                                                                                |
| 1: $T \gets \emptyset$                                                                                                                                                                           |
| 2: for $i \in \{1, \ldots, T\}$ do                                                                                                                                                                |
| 3: $s_i \gets Decode(M_t, [p; x_i; D_i])$                                                                                                                                                        |
| 4: $e_i \gets Score(M, y_i | [s_i; x_i])$                                                                                                                                                        |
| 5: $v_i \gets Score(M, y_i | [x_i])$                                                                                                                                                             |
| 6: if $v_i < e_i$ then                                                                                                                                                                            |
| 7: $T \gets T \cup \{(x_i, D_i, s_i)\}$                                                                                                                                                          |
| 8: break                                                                                                                                                                                           |
| 9: $T \gets T \cup \{(x_i, D_i, s_i)\}$                                                                                                                                                          |
| 10: $enc_{c\theta} \gets Finetune(enc_{c\theta}, T)$                                                                                                                                           |

![Learning an extractive compressor for QA task. The Score here is the exact match between the decoded answer and the gold answers.](assets/page_0016_img_1.png)
![Learning an abstractive compressor for QA task. The Score here is the exact match between the decoded answer and the gold answers.](assets/page_0016_img_2.png)
```

### --- Page 0017 ---

```markdown
| Dataset   | Extractive                |                             | Abstractive                |                             |
|-----------|---------------------------|-----------------------------|----------------------------|-----------------------------|
|           | Train      | Validation   | % filtered | $|V|$ | Train      | Validation   | % filtered | % empty |
|-----------|------------|--------------|------------|-----|------------|--------------|------------|---------|
| NQ        | 42,149     | 9,769       | 46         | 4.44| 39,466     | 4,931        | 50         | 25      |
| TQA       | 70,032     | 8,753       | 56         | 4.37| 48,322     | 5,887        | 32         | 16      |
| HotpotQA  | 24,526     | 3,068       | 69         | 4.33| 26,556     | 2,937        | 42         | 4       |
| Wikidata  | 1,398,318  | 1,5483      | 41         | 4.04| 38,410     | 9,603        | 0          | 24      |

![Histogram of abstractive summary length (#tokens) distribution for testing data of NQ, TQA, HotpotQA and Wikidata.](assets/page_0017_img_1.png)
![Histogram of abstractive summary length (#tokens) distribution for testing data of NQ, TQA, HotpotQA and Wikidata.](assets/page_0017_img_2.png)
![Histogram of abstractive summary length (#tokens) distribution for testing data of NQ, TQA, HotpotQA and Wikidata.](assets/page_0017_img_3.png)
![Histogram of abstractive summary length (#tokens) distribution for testing data of NQ, TQA, HotpotQA and Wikidata.](assets/page_0017_img_4.png)

Abstractive Compressor We implement the model using the Transformers (Wolf et al., 2019). We train abstractive summarizer with Adam optimizer (Kingma & Ba, 2014), using a batch size of 16, learning rate of $1e-5$ and 1000 warmup steps for 3 epochs.
```

### --- Page 0018 ---

```markdown
| Table 6: Example abstractive and extractive compression on wikitext-103 dev set and NQ. |
|----------------------------------------------------------------------------------------|
| **Wikitext-103 input**                                                                 |
| present in most of the Mediterranean Sea, only missing from the section east of Crete, and along only the north e- |
| west coast of the Black Sea                                                            |
| **Original Top 1 document**                                                            |
| Sea of Crete* Sea of Crete The Sea of Crete*, “Kritiko Pelagos”* or Cretean Sea, is a sea, part of the Aegean Sea, located in |
| Southern extremity. The sea stretches to the North of the island of Crete. East of the islands of Rhyakos and Kassos. The |
| boundary extends West to the Ionian Sea. To the Northwest is the Myrtoan Sea, a subdivision of the Mediterranean Sea that lies |
| between the Cyclades and Peloponnese. To the East-S.E is the rest of the Mediterranean Sea, |
| **Method**                                                                             |
| **BoW**                                                                                |
| present Mediterranean Sea missing section east Crete along north west coast Black The Kritiko Pelagos Cretan sea part |
| Aegean Sea located Southern extremity stretches North East island Kythira Antikythera West Dodecanese Rhodes Karpathos Kassos |
| bounding Ionian To Northwest Myrtoan subdivisions Ion Peloponnese Sea rest Kythira the Aegean Sea the Ionian Sea Southern South Rhodes the Myrtoan Sea Cretean Sea Antikythera Dodecanese Kassos Karpathos the Black Sea Sea of Crete Peloponnese the Mediterranean Sea Cyclades to the Northwest is the Myrtoan Sea, a subdivision of the Mediterranean Sea that lies between the Cyclades and Peloponnese. |
| **Extractive compression**                                                             |
| **NQ Input**                                                                          |
| who got the physics nobel prize in physics                                             |
| **Original Top 1 document**                                                            |
| receive a diploma, and a medal and a document confirming the prize amount. Nobel Prize in Physics The Nobel Prize in Physics |
| is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions and marked in the field of physics. It is one of the Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded in 1901. The Nobel Prize in Physics was awarded to physicist Wilhelm Röntgen in recognition of the extraordinary services he was awarded the Ahe. In addition, eight formalidades have gone to receive the Nobel Prize in Physics: Claude-Tanoudji, Pierre-Gilles de Gennes, Albert Einstein, Alfred Kastler, Gabriel Laméguin, Louis de Broglie, Jean Baptiste Perrin and Serge Haroche, while other ENS physicists include such major figures as Paul Langevin, famous for developing Langevin dynamics and the Langevin equation. Alumni Post-Salient won the Nobel Prize in Chemistry. A ranking of universities worldwide based on results of alumni to Nobel prize-winner published in 2016 by American scholars Skopky Husa and Jonathan Wai placed ENS as the first university worldwide, reaffirmed by the discovery of the remarkable rays (or x-rays). This award is administered by the Nobel Foundation and widely regarded as the most prestigious award that a scientist can receive in physics. It is presented in Stockholm at an annual ceremony on 10 December, the anniversary of Nobel's death. Throughout 2018, a total of 290 individuals have been awarded the prize. Only three women (1.4% of laureates) have won the Nobel Prize in Physics: Marie Curie in 1903, Maria Goeppert Mayer in 1963, and Donna Strickland in 2018. Alfred Nobel, in his last will and testament, stated that his |
| **Method**                                                                             |
| **TS**                                                                                 |
| Wilhelm Röntgen received the first Nobel Prize in Physics in recognition of its extraordinary services. It is one of the five Nobel Prizes established by Alfred Nobel in 1895 and awarded since 1901. The first Nobel Prize in Physics was awarded to physicist Wilhelm Röntgen in 1901 for his discovery of the remarkable rays (or x-rays). Since then, 290 individuals have been awarded the prize, with only three women (1.4% of laureates) having won. |
| **Table 7: Example input to the Flan-UL2 for NQ with in-context examples and retrieved documents.** |
| **Dataset**                                                                            |
| **Prompts**                                                                            |
| **NQ**                                                                                |
| who won a million on deal or no deal Answer: Tomorrow Rodriguez                       |
| who is the actor that was playing the car in cool hand luke Answer: Roy Harmon        |
| who said it’s better to have loved and lost Answer: Alfred, Lord Tennyson            |
| who is the first indian woman to be crowned as miss world Answer: Reita Faria        |
| **Retrieved Docs**                                                                     |
| **Question**                                                                          |
| **Answer:**                                                                           |
```

### --- Page 0019 ---

```markdown
| Dataset   | Prompts                                                                                                                                                                                                                     |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NQ        | Compress the information in the retrieved documents into a 2-sentence summary that could be used to answer the question: Question: `query` Retrieved documents: `docs` <br> Compressed documents:                          |
| TQA       | Compress the information in the retrieved documents into a 2-sentence summary that could be used to answer the question: Question: `query` Retrieved documents: `docs` <br> Compressed documents:                          |
| HotpotQA  | Source documents: `docs` Question: `query` Generate a reasoning chain to answer the question:                                                                                                                             |
| Wikitext  | Generate the next two sentences of the given query using the information from the provided documents. <br> Source Documents: `docs` <br> Query: `query` <br>                                                                 |
| Wikitext  | Select sentences from the retrieved docs that are most likely to be in the next sentence. <br> Source Documents: `docs` <br> Query: `query` <br>                                                                           |
| Wikitext  | Generate the next one sentence of the given query using the information from the provided documents <br> Source Documents: `docs` <br> Query: `query` <br>                                                                  |
| Wikitext  | Summarize the information from the provided documents <br> Source Documents: `docs` <br> Query: `query` <br>                                                                                                            |
```
![Table of prompts used to generate summaries from GPT-3.5-turbo](assets/page_0019_img_1.png)
```

### --- Page 0020 ---

```markdown
# Table 9: Case study of how compressing the retrieved documents helps the model to identify the right answer from NQ dev set.

| Question: host of the late show who was once a correspondent for the daily show. | Gold answer: Stephen Colbert |
|-------------------------------------------------------------------------------|------------------------------|
| Type                                                                          | In-context documents          | Predicted Answers            |
| None                                                                          | Chelsea Handler               | Samantha Bee                 |
| Top 5                                                                         |                              |                              |
| 1. By Conan O'Brien, in 2009. Leno explained that he did not want to see a repeat of the hard feelings and controversy that occurred when he was given the show over David Letterman following Carson's retirement in 1992. O'Brien's last "Late Night" episode was taped on February 30, 2009. Former Saturday Night Live host Jimmy Fallon took over host of "Late Night with Jimmy Fallon" on March 2, 2009. The Colbert Report aired for days a week on Comedy Central from October 17, 2005, was hosted by Stephen Colbert, one of the regulars on Comedy Central's The Daily Show as has begun with a notable interview with former British prime minister Tony Blair. The interview occurred the night before a book signing at 2100 which attracted international attention when Blair was pulled into shoes and eggs and successfully evaded an attempted citizen's arrest on charges of war crimes. On 1 February 2013, Kat Kenny returned to host that night's edition when Tudy's father did. | 1. Paul Murdock's in-depth choice of questions when interviewing Anti-Austerity Alliance TD Paul Murphy in relation to the campaign against the implementation of a water tax was much criticized. Opponents of the 'Michigan', interviewing Emmet. Colbert has been given near-full control of the show, with little interference from CBS management in regard to format. Colbert brought most of his staff from "The Colbert Report" with him to "The Late Show", as well as outsiders such as Brian Stack, who is best known for his work on Conan O'Brien's programs, and Jon Stewart, former host of Colbert's predecessor program "The Daily Show", who is credited as executive producer. Colbert no longer uses the character he had portrayed on "The Colbert Report", jokingly referring to Big Bush that "I used to play a character" and has had three regular hosts: Jay Byrne, Tim Kenny and Ryan Tubridy. Colbert was deputized by Byrne for three seasons in the 1990s. There have been at least four occasions on which another presenter has hosted the show. The first was when Byrne became unexpectedly and seriously ill. Frequent panelist Ted Bonner presented instead. The second time was towards the end of a show about feminism, when Byrne suffered a voting Maria Flannery into this seat to present the remainder of the show. On another occasion, radio broadcaster and former news reader Andy O'Mahoney replaced Byrne for an interview popular acclaim. Colbert would host the program and he was chosen to replace David Letterman as host of CBS's "Late Show" in 2015. Ed Helms, a former correspondent from 2002 to 2006, also starred on NBC's "The Office" and was a main character in the 2009 hit "The Hangover". After filling in as host during Stewart's two-month absence in the summer of 2013, John Oliver went on to host his own show on HBO, Last Week Tonight with John Oliver. In 2016, former correspondent Samantha Bee launched her own late-night talk show "Full Frontal with Samantha Bee". Best bounded Jason |
| GPT-3.5-turbo                                                                | Former Daily Show correspondent Stephen Colbert was chosen to replace David Letterman as host of CBS's "Late Show" in 2015, while Ed Helms, a former correspondent from 2002 to 2006, also starred on NBC's "The Office" and Jon Oliver, who filled in as host during Jon Stewart's absence in 2013, went on to host his own show on HBO. | Stephen Colbert              |
| T5 (ours)                                                                     | Stephen Colbert was a former correspondent for The Daily Show and later became host of CBS's "Late Show" in 2015. He has since brought most of his staff from "The Colbert Report" with him to "The Late Show", with little interference from CBS management in regard to format. |
```

### --- Page 0021 ---

```markdown
# Table 10: Open-domain QA results on LLaMA-13B. We report the results of oracle compressions with Flan-UL2, the base model for the compressors, (Oracle w/ FLAN) and the oracle compression results for LLaMA-13B.

| In-context evidence          | NQ          |                | TQA         |                | HotpotQA     |                |
|------------------------------|-------------|----------------|-------------|----------------|--------------|----------------|
|                              | # tok       | EM             | # tok       | EM             | # tok        | EM             |
| -                            | 0           | 30.89          | 0           | 65.00          | 0            | 24.20          |
| **RALM without compression** |             |                |             |                |              |                |
| Top 1 document               | 132         | 33.35          | 136         | 66.62          | 138          | 34.40          |
| Top 5 documents              | 660         | 37.04          | 667         | 70.61          | 684          | 37.00          |
| **Phrase / token level compression** |   |                |             |                |              |                |
| Top 5 documents (BoW)       | 450         | 33.05          | 259         | 66.59          | 255          | 30.00          |
| Top 5 documents (NE)        | 338         | 34.60          | 128         | 65.88          | 157          | 29.20          |
| **Extractive compression of top 5 documents** | |          |             |                |              |                |
| Oracle                       | 31          | 56.62          | 68.89       | 31             | 84.61        | 80.46          |
| Oracle (w/ FLAN)            | 34          | 40.89          | 50.06       | 32             | 68.52        | 74.96          |
| Random                       | 32          | 30.33          | 39.85       | 31             | 62.80        | 69.25          |
| Contriever                   | 36          | 32.52          | 42.01       | 40             | 65.88        | 72.44          |
| Ours (init. w/ T5)          | 37          | 34.38          | 44.15       | 38             | 65.28        | 71.85          |
| **Abstractive compression of top 5 documents** | |        |             |                |              |                |
| Oracle                       | 50          | 45.60          | 84.87       | 38             | 74.37        | 79.83          |
| Oracle (w/ FLAN)            | 51          | 38.98          | 49.40       | 37             | 69.86        | 76.46          |
| T5                           | 1           | 33.38          | 43.54       | 37             | 63.18        | 70.92          |
| Ours (init. w/ T5)          | 36          | 36.32          | 46.10       | 32             | 66.27        | 73.12          |
```


### --- Page 0022 ---

```markdown
| Dataset | Model | Query, Passages and Summary | Evaluation |
|---------|-------|-----------------------------|------------|
| NQ      |       | Passages: When will miraculous ladybug season 2 episode 12 come out? 2016 to 2016 (TVNZ's 1). In Japan, Disney Channel streamed the episode "Stormy Weather" through its mobile application on July 2018, before the official premiere on 23 July in the same year. The second season premiere is scheduled for a global launch around September–November 2017 in Europe, at a panel at San Diego Comic-Con 2017, it was announced that the second season would have its North American release on Netflix in December 2017, with 31 episodes to be released. Kickstart will air season 3 of this show in the US starting 30 August 2018, marking the first time that Korea 1 on 1 September 2018 on ERSL. In the United States, the series debuted on Nickelodeon on December. In the United Kingdom and Ireland, the show premiered on 30 January 2016 on Disney Channel. A Christmas special was released in 2016 and the second season premiered in France on TF1 and in English on Disney Channel UK in 2017. Netflix acquired the U.S. video-on-demand streaming rights and further success are in production. As mentioned, the series focuses on the adventures of Marinette Dupain-Cheng and Adrien Agreste. When this airs, the Miraculous transformations take her superhero persona Ladybug, while Adrien transforms into Cat Noir. Eastern Europe and Scandinavia, and free-to-air rights in Spain, Germany, France, and Turkey. South Korea has the exclusive rights to stream "Ladybug," with 63 episodes until November 2016, with spread through February 2016, and its second half of the season airing 1 March 2016. 26K Broadcaster, having participated in the production, provided the episodes in a partnership with Disney. Exclusively to subscribers of their IPTV platform B TV about a half-hour following the South Korean broadcast of Season 2 of this show airing on a network after re-television. A third season is also in development. On 22 January 2018, Zag posted on Instagram that the crew was working on a fourth and fifth season. On 31 August premiered a new series with Miraculous: Tales of Ladybug & Cat Noir which this film is in the style. On 5 September 2018, Syldance Media has acquired the rights to the news media such as "The Huffington Post", wrote that the show has inspired teens and adults to create Channel 2 on Disney Channel. The show was brought to the United Kingdom and Netflix subscribers in the UK. Among the franchise's terrestrial television broadcast in the Republic of Ireland, Raidiú F́éilte premiered the show in 2018 on RTÉ's television channel. The series has been broadcast in various countries, including the United States, the UK, and its channel ABC's on 22 March 2016. In New Zealand, the show premiered on 27 April 2017 on Kidzone. Miraculous Ladybug season 3 is scheduled for release in 2018 on Netflix in North America. The show is presented in South Korea on 1 September 2017 and has been aired in various countries, including in the United States, the UK, and | Not faithful, Comprehensive |
| TQA     |       | Question: In which wood does Flora Pose move to live with her relatives, the Sturkdakers?  In Sussex? Passages:  The Sturkdakers are a writer, decides that the only way for her to live without researching her writing is to stay with relatives. Her city-based relatives show no interest, so she seeks something more. There are a few who, most of them, unsatisfied, but one is intriguing. Flora decides to stay for a while with the Sturkdakers family on their rundown farm. The Sturkdakers are an assortment of rustic, uncouth, and quirky eccentric characters, for which she decides she has the right to be critical, emotional, or playful to overcome reaching her potential. Flora quickly realises that, relatives at the isolated Cold Comfort Farm in the fictional village of Howling in Sussex. The inhabitants of the farm – Aunt Ada Doom, some long-festering emotional problem caused by lingering hatred, fear, and the farm has been humbly. Flora, being a liberal, husband, woman to the family, and several years later based on "Conference at Cold Comfort Farm," when Flora is married with several children, was broadcast. In 1995 a television film was produced which was generally well-received, with critics. Janet Maslin in the "New York Times" wrote that screen version "gets it exactly right." The film starred Kate Beckinsale as Flora, Joanna Lumley as her friend and mentor Mary Smiling, Rufus Sewell as Seth, Mel McKellen as Amos Starkadder, and Atkins as Judith, Stephen Fry as Myth, Miranda Maguire as Mrs. Beet, and what. There comes also suddenly of a heart attack and Lady Place is exited out, with the view that this is a grown-up, will return to the home for the business. After several years of living, a new family finds herself feeling increasingly isolated by her obligations to the family and living in London. When shopping for flowers on the Moscow Road, Laura decides she has the right to call her brother, and again, a good wish for her respect as the visible of local Map to see her own.  Flora moves to the ancestral home in Newfoundland, which she has abandoned for a few days. Realizing that through it is also the ground, she finds him to meet with. While struggling to rebuild his life, the quirky eccentric house, and for his daughter, Quirky meets social resident Wayne Proves, a widow who has a pre-teen son with learning disability. | Faithful, Not comprehensive |
```


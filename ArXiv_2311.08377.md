# ArXiv 2311.08377

### --- Page 0001 ---

```markdown
# Learning to Filter Context for Retrieval-Augmented Generation

**Zhiruo Wang**¹  **Jun Araki**¹  **Zhengbao Jiang**¹  
**Md Rizwan Parvez**²  **Graham Neubig**¹  
¹Carnegie Mellon University  ²Bosch Research  
{zhiruo,zhengbaj,neubig}@cs.cmu.edu

## Abstract
On-the-fly retrieval of relevant knowledge has proven an essential element of reliable systems for tasks such as open-domain question answering and fact verification. However, because retrieval systems are not perfect, generation models are required to generate outputs given partially or entirely irrelevant passages. This can cause over- or under-reliance on context, and result in problems in the generated output such as hallucinations. To alleviate these problems, we propose FILCo, a method that improves the quality of the context provided to the generator by (1) identifying useful context based on lexical and information-theoretic approaches, and (2) training context filtering models that can filter retrieved contexts at test time. We experiment on six knowledge-intensive tasks with FLAN-T5 and LLaMA2, and demonstrate that our method outperforms existing approaches on extractive question answering (QA), complex multi-hop and long-form QA, fact verification, and dialog generation tasks. FILCo effectively improves the quality of context, whether or not it supports the canonical output.¹

## 1 Introduction
Retrieval augmented approaches to generation have been shown effective for many knowledge-intensive language tasks such as open-domain question answering and fact verification, producing more faithful (Khandelwal et al., 2020; Lewis et al., 2020; Shuster et al., 2021; Komeili et al., 2022), interpretable (Guu et al., 2020), and generalizable (Khandelwal et al., 2021) outputs. While the de facto approach is to provide the top retrieved passages to the generator indiscriminately, imperfect retrieval systems often return irrelevant or distracting content. Generation models are then trained to produce canonical outputs with the guidance of partially or entirely irrelevant passages, and thus are prone to hallucination or spurious memorization.

![Figure 1: FILCo filters out irrelevant content (marked in red) and leaves precisely supporting content, making it easier for the generator to predict the correct answer.](assets/page_0001_img_1.png)

Ideally, a model should be grounded on the precisely supporting content to generate the correct output. However, this ideal grounding is hard to achieve with an imperfect retrieval system alone. On one hand, positive passages (i.e., passages that support the output) sometimes contain distracting content. For example in Figure 1, while the passage containing the actual supporting content is successfully retrieved, the model still fails to pay sufficient attention to the supporting content, and is distracted by surrounding sentences that share similar topics (Shi et al., 2023). On the other hand, models learn to over-utilize negative passages in the same way as using positive passages, e.g., extracting a span from the irrelevant passage, which would inevitably be incorrect. This potentially degrades accuracy, as training with higher-quality context often leads to better performance (Dou et al., 2021).

Some works have attempted to optimize the provided content on the passage level, by reranking more relevant passages rise to the top of the retrieved list (Wang et al., 2018; Nogueira and Cho, 2020; Mao et al., 2021), selecting only evidential passages to include (Asai et al., 2022), or only retrieving passages when generation models need assistance (Mallen et al., 2023; Jiang et al., 2023). Choi et al. (2021) proposed to decontextualize sen-
```

### --- Page 0002 ---

```markdown
![The FILCo pipeline: (i) filtering retrieved passages, (ii) generation with filtered context.](assets/page_0002_img_1.png)

## 2 Generation with Filtered Contexts

In this section, we first outline notation (§2.1), then introduce three oracle filtering strategies (§2.2). Next, we describe how to train context filtering models with oracle filtered context (§2.3) and learn to generate with filtered contexts (§2.4).

### 2.1 Problem Statement

In retrieval-augmented generation, we are given an input query $q$ and annotated output $o$ from an example $e = \{q, o\}$, and want to improve the output of a generative model $M_{gen}$. We assume a set of retrieved passages $P = \{p_i\}_{i \in K}$, each consisting of $n_i$ text spans $p_i = \{t_{i,1}, \ldots, t_{i,n_i}\}$. We can provide the model with one or more selected text spans $T = \{t_j\}_{j}$ when generating output $o$, namely $M_{gen}(o | T)$. In traditional retrieval-based methods, however, all text spans in the top-$K$ passages $\{t_{i,j} \,|\, \forall i \in K\}$ are provided to the model. In experiments, we split passages into sentences using the spaCy tokenizer² as candidate text spans. Later in §5, we will show that sentence-wise splitting performs the best among our comparisons.

### 2.2 Obtaining Oracle Contexts

In this section, we propose methods that select oracle text spans that can be used to train a context filtering model. We select text spans using a filtering function $F(\cdot)$, denoted as $F(T, P)$, where text spans in $T = \{t_j\}$ are selected by the underlying score function $f(\cdot)$ according to individual filtering methods. We select a single best span $T = \{t_j\}$, $j \in \text{argmax}_{j} f(t_{i,j})$, when using oracle filtering, as it outperforms multi-span filtering in our preliminary studies.
```

### --- Page 0003 ---

```markdown
We now introduce three approaches to filtering potentially useful content from retrieved passages.

| **String Inclusion** | The STRING measure $f_{inc}(t, o) \in \{0, 1\}$ that makes a binary decision on whether text span $t$ lexically contains the output $o$. We enumerate the ranked passages retrieved $\{p_1, p_2, \ldots\}$ and select the first text span that contains the output $f_{inc}(t, o) = 1$. This measure is effective when the supporting document $p_{gold}$ contains the exact output text $o$. However, $f_{inc}$ may fail to distinguish supporting context from spurious ones, that accidentally contain the output but do not answer the question. Applying $f_{inc}$ to other abstractive tasks may result in selecting zero spans since no exact matches exist. |

| **Lexical Overlap** | We next introduce a more flexible LEXICAL measure $f_{u1} \in [0, 1]$ that calculates the unigram overlap between the example $e$ and the candidate text span $t$. Intuitively speaking, higher lexical overlap indicates greater topic similarity, hence higher utility at generation time. We select sentences $t$ using different parts of the example $e$ for tasks of different types. We measure the $F_1$ score $f_{u1}(t, o) \in [0, 1]$ between $t$ and output $o$ for tasks having responses grounded on provided knowledge, i.e., QA and dialog generation. We measure $t$ using query $q$ for fact verification as $f_{u1}(t, q)$ since $q$ is a one-word binary label. We select the sentence $t_i^j$ with the highest similarity to example $e$ and above a pre-defined threshold $\lambda = 0.5$, where $(i, j) = \text{arg max}_{i,j}(f_{u1}(t_i^j, e))$, and $i, j \in \{i, j | f_{u1}(t_i^j, e) > \lambda\}$. Nonetheless, for tasks having queries that may be factually incorrect (e.g., fact verification), spans of high lexical overlap may reinforce the misinformation and lead to incorrect generations. |

| **Conditional Cross-Mutual Information (CXMI)** | We adopt a measure $f_{cmi}$ from the conditional cross-mutual information (CXMI) score in contextual machine translation (Fernandes et al., 2021). Given a pair of input sequences with and without context augmentation, $t \, \& \, q$, we measure the probability difference in model $M_{gen}$ generating the expected output as $f_{cmi}(t, e) = \frac{M_{gen}(o | t, q)}{M_{gen}(o)} \in \mathbb{R}$, as illustrated in Figure 3. We select the text span $t_i^j$ having the highest CXMI score above a pre-defined threshold $\lambda = 0.4$, where $(i, j) = \text{arg max}_{i,j}(f_{cmi}(t_i^j, e))$, and $i, j \in \{i, j | f_{cmi}(t_i^j, e) > \lambda\}$. $f_{cmi}$ can overcome the lexical barrier and is applicable to all tasks, however at the cost of more computation. |

![An example illustration of context filtering with the CXMI strategy.](assets/page_0003_img_1.png)

### 2.3 Learning to Filter Contexts
While the previous section described how to identify useful contexts at training time when the gold standard answer is known, we also need methods that can apply at test time when the answer is unknown. To this end, we train the context filtering models, $M_{ctx}$, using context filtered with the three measures in §2.2. To create training data for $M_{ctx}$, for each training example with query $q$, we concatenate the retrieved passages $P$ and query $q$ as input, then, we apply the filter method $f$ to obtain filtered context $t_{silver}$ as output. We use $silver$ instead of $oracle$ to represent the non-perfect filtering result due to unknown gold labels for non-extractive tasks. As shown in Figure 2, we train $M_{ctx}$ by feeding in query $q$ and retrieved passages $P$, and ask it to generate filtered context $t_{silver}$, formalized as $M_{ctx}(t_{silver} | q \, \& \, P)$. At test time, given the retrieved passages $P$ for each test query $q$, we leverage $M_{ctx}$ to predict filtered context $t_{pred}$, formalized as $t_{pred} = M_{ctx}(q \, \& \, P)$. $t_{pred}$ is subsequently provided to the generation model $M_{gen}$ together with the query $q$ to predict the output.

### 2.4 Generation With Filtered Contexts
As illustrated in Figure 2, we similarly use $t_{silver}$ filtered context for training and model predicted output.
```

### --- Page 0004 ---

```markdown
context $t_{pred}$ for inference.

For each training example $(q, o)$, we prepend the silver filtered context $t_{silver}$ to the example query $q$, and obtain the model input $q \oplus t_{silver}$. We feed this input into the generation model $M_{gen}$ and train it to output the canonical response, formulated as 

$$
M_{gen}(o | t_{silver} \oplus q).
$$

At inference time, we provide the context $t_{pred}$ filtered by model $M_{act}$ for generation, denoted as 

$$
M_{gen}(o | t_{pred} \oplus q) = M_{gen}(o | M_{act}(q, P) \oplus q).
$$

In comparison to appending all retrieved text spans $P \oplus q$, including only selected text can effectively reduce the computational cost by $|P| / |t|$ at both training and inference time.

## 3 Knowledge-Intensive Language Tasks

We experiment on six knowledge-intensive language tasks that necessitate retrieval augmentation for generation (§3.1), where a limited portion of examples are supported by retrieved passages (§3.2).

### 3.1 Tasks and Datasets

We use six datasets built from Wikipedia articles as supporting documents for answer, response, and judgment generation, as listed in Table 1.

#### Open-Domain Question Answering

We adopt NaturalQuestions (NQ) (Kwiatkowski et al., 2019) and TriviaQA (TQA) (Joshi et al., 2017) to experiment with the open-domain QA task.

Each example in NQ has a question $q$ and annotated short answers $o$. We experiment with the processed version (Lee et al., 2019) that includes all examples having short answers of no more than five tokens. For the TQA dataset, each example has a question $q$ and answers $o$, which are extracted spans from supporting Wikipedia articles $P$. Following Lewis et al. (2020), we use the Exact Match (EM) metric to evaluate model predictions.

#### Multi-Hop Question Answering

We also adopt more complex QA scenarios, the first of which is multi-hop QA, where each question $q$ requires reasoning over a chain of passages $P$ to obtain the correct answer $o$. For this task, we use the HotpotQA (Yang et al., 2018) dataset containing 113K question-answer pairs created based on Wikipedia pages. Because the answers do not always appear in the ground-truth supporting documents $P$, this dataset belongs to abstractive generation, in contrast to the extractive nature of answers in NQ and TQA. Following Yang et al. (2018) and accommodating its abstractive nature, we use unigram F1 to evaluate answer correctness.

#### Long-Form Question Answering

Another complex QA task is generating long, abstract answers given the question, i.e., long-form QA. For this we use the ELI5 (Fan et al., 2019) dataset, which requires elaborate and in-depth answers to open-ended questions. The dataset comprises 270K threads from the Reddit forum “Explain Like I’m Five” (ELI5) and features diverse questions requiring multi-sentence answers. We experiment with the generative short setting, and evaluate model predictions using unigram F1.

#### Fact Verification

We use the Fact Extraction and VErification (FEVER) dataset (Thorne et al., 2018) aggregated by the KILT benchmark (Petroni et al., 2021). It contains claims $q$ generated by rephrasing sentences in Wikipedia articles. A claim has the label $o = “SUPPORTS”$ if it preserves the fact in the Wikipedia reference, otherwise is labeled as “REFUTES” due to the fact contradiction. Following the original baseline (Thorne et al., 2018), we use accuracy for evaluation.

#### Knowledge-Crowned Dialog Generation

We adopt the Wizard of Wikipedia (WoW) dataset (Dinan et al., 2019) from KILT, which aims to generate the next dialog by grounding on Wikipedia articles. In each example, the input $q$ is the conversation history involving multiple utterance turns, and the next-turn response is the output $o$. We evaluate with unigram F1 following Petroni et al. (2021).

| Dataset  | # Examples | Dev | Test | Evaluation Metric |
|----------|------------|-----|------|--------------------|
| NQ       | 79.2      | 8.7 | 3.6  | EM                 |
| TQA      | 78.8      | 8.8 | 1.3  | EM                 |
| HotpotQA | 88.9      | 5.6 | 5.6  | F1                 |
| ELI5     | 273.0     | 1.5 | 0.6  | F1                 |
| FEVER    | 105.0     | 10.4| 1.0  | Accuracy           |
| WoW      | 63.7      | 3.1 | 2.9  | F1                 |

Table 1 lists the dataset statistics. Because test sets are not available for datasets adopted from the KILT benchmark (i.e., HotpotQA, ELI5, FEVER, WoW), we report the development set results.

### 3.2 Wikipedia Passage Retrieval

To better understand the quality of passages provided in the generation stage, we evaluate the per...
```

### --- Page 0005 ---

```markdown
# 4 Experiments and Analysis

We first introduce the experimental setup (§4.1) and baseline approaches for comparison (§4.2). Then, we evaluate model performance on both end generation (§4.3) and context filtering (§4.5).

## 4.1 Experimental Setup

We use FLAN-T5 (Chung et al., 2022) and LLAMA 2 (Touvron et al., 2023) as the backbone model architectures, because of their potential superior performance among open-source models. We fine-tune both models for (i) the context filtering task as $M_{ctx}$ and (ii) the end generation task as $M_{gen}$.

### FLAN-T5
FLAN-T5 is a family of instruction-tuned encoder-decoder models for seq2seq generation tasks, which makes it suitable for our retrieval-augmented generation setting. Due to constraints in computational resources, we use the XL version with 3B parameters. We load model checkpoints from and implement training using HuggingFace Transformers (Wolf et al., 2020).

### LLAMA 2
LLAMA 2 represents a collection of foundation model ranging from 7B to 70B parameters, particularly optimized for dialog use cases, but also achieve good performance on many other tasks. We train the 7B model version with LoRA (Hu et al., 2022) using the xTuring platform. 

### Implementation Details
For both models, we allow a maximum length of 1024 tokens for all sequences at training and inference. $M_{ctx}$ is configured to generate at most 512 tokens as filtered context for all tasks. We allow $M_{gen}$ to generate at most 128 tokens for extractive QA, fact verification, and dialog generation tasks. We use greedy decoding for generating both filtered context and end-generation output. Unless otherwise specified, we train all $M_{ctx}$ and $M_{gen}$ models for 3 epochs, using a learning rate of $5e^{-5}$ and batch size of 32.

## 4.2 Experiment Methods
We describe two baselines FULL and PSG, our main approach FilCo, and the SILVER setting.

### Baseline 1: Augmenting with Full Passages
The most common approach for retrieval-augmented generation is to concatenate all passages into the input. We denote this method as FULL and adopt it as our first baseline. To conduct a fair comparison with sufficient training for

| Dataset | Recall (pos. + neg.) | Precision (pos.) |
|---------|-----------------------|------------------|
|         | 1     | 5     | 1   | 5   |
| NQ      | 50.1  | 74.1 | 2.5 | 2.7 |
| TQA     | 61.2  | 77.8 | 4.5 | 4.8 |
| HotpotQA| 16.7  | 27.3 | 2.1 | 0.4 |
| ELI5    | 13.1  | 25.7 | 97.7| 55.1|
| FEVER   | 57.0  | 75.9 | 1.3 | 1.4 |
| WoW     | 34.9  | 54.8 | 16.4| 17.7|

Table 2: Recall of the top 1 and top 5 DPR-retrieved passages, and precision on positive passages.

## Noise in Positive Passages
To measure the ratio of precisely supporting context in retrieved passages, we further calculate their unigran precision with regard to the annotated output, as shown in Table 2. In general, the precision is pretty low: scoring less than 20.0 for WoW, and less than 5.0 for NQ, TQA, HotpotQA, and FEVER. ELI5 has exceptionally high top-1 precision, because its output often aggregates large text chunks from multiple passages. However, precision drops by over 40 points when adding 4 more passages. These numbers indicate the potential existence of redundant content, which could distract the model and deteriorate the final generation.

To retrieve Wikipedia passages for all examples, we use the adversarial Dense Passage Retriever (DPR) (Karpukhin et al., 2020) to retrieve the top 5 passages from all Wikipedia passages.
```

### --- Page 0006 ---

```markdown
![Generation performance when passages are filtered with different approaches.](assets/page_0006_img_1.png)

generation in a full-context style, we fine-tune the FLAN-T5 and LLAMA 2 models to generate outputs using the full content of the top-1 passages under the same experiment setting as in §4.1.

## Baseline 2: Passage-Wise Filtering
An alternative method inspired by Asai et al. (2022) is to filter context on a passage level. Specifically, for each passage among the top-1 retrieved ones, the model decides whether to include the entire piece of the passage in the input. In comparison, our model operates in a finer granularity (i.e., on the sentence level) and could trained with multiple filtering strategies. To show the empirical advantage of our method, we denote this approach as PSG and adopt it as another baseline.

## Main Approach: Augmenting with Filtered Context
As described in §2, we train $M_{ctx}$ to filter the top-1 retrieved passage $P$ to $t_{silver}$, and $M_{gen}$ to generate output $o$ with $t_{silver}$. To create $t_{silver}$, we use the STRINC measure for NQ and TQA, LEXICAL for FEVER, and CXMI for WoW, HotpotQA, and ELI5. These measures are shown to be the optimal settings based on further analysis in §5.

At test time, we provide model-filtered context $t_{pred}$ to $M_{gen}$, and denote the results as FILCo. To demonstrate the prospective performance upper bound, we also evaluate $M_{gen}$ generation by providing silver-filtered context $t_{silver}$, and denote these results as SILVER.

## 4.3 Generation Performance
Results using four methods and two models are shown in Figure 4. In general, applying context filtering beforehand significantly improves the results on all datasets than FULL. Moreover, filtering in a finer granularity is better than PSG.

Compared to providing $M_{gen}$ with SILVER filtered contexts, using contents predicted by the filter model, i.e., FILCo achieves comparable performance on all six tasks, indicating effective training of the context filtering process.

For extractive QA tasks, our method achieves +4.3 and +8.6 EM increase in NQ with FLAN-T5 and LLAMA2 models, +1.1 and +0.2 EM increase in TQA. As exemplified by Figure 1, our context filter effectively removes distracting alternative answers and irrelevant passages, hence enabling the generation model to hit the correct answer span with higher precision and lower effort.

For more complex QA tasks, our method brings +1.0 and +1.3 F1 increase in HotpotQA with FLAN-T5 and LLAMA2 models, and +0.6, +2.6 EM increase in ELI5. The overall improvement is less significant, compared to extractive QA tasks, presumably due to the increased task difficulty.

For abstractive generation tasks, our method brings about even larger improvements: +6.2 and +4.3 accuracy increase for FEVER with FLAN-T5 and LLAMA2, and +3.5, +1.1 F1 increase for WoW. As could be partially countered from the low precision in Table 2, filtering irrelevant content helps the model focus on the concerned knowledge.

## 4.4 Generation With Filtered Positive and Negative Passages
We decompose datasets into examples with positive and negative top-1 retrieved passages, to examine improvements under both scenarios. As shown in Figure 5, for both positive and negative passages retrieved, applying FILCo effectively improves the context quality, hence yields better end generation results, particularly for abstractive generation tasks such as FEVER and WoW. Align-
```

### --- Page 0007 ---

```markdown
![Improvement on examples retrieved with positive (top) and negative passages (bottom)](assets/page_0007_img_1.png)

![Number of input tokens after filtering retrieved contexts with different strategies](assets/page_0007_img_2.png)

| Method | FULL | PSG | FILCo | SILVER |
|--------|------|-----|-------|--------|
| NQ     | 2.5  | 1.3 | 5.1   | 7.3    |
| TQA    | 4.5  | 3.0 | 8.4   | 4.6    |
| HotpotQA | 2.6 | 2.6 | 10.8  | 17.1   |
| ELI5   | 92.9 | 92.5| 98.8  | 98.8   |
| FEVER  | 1.2  | 1.2 | 5.1   | 4.4    |
| WoW    | 10.8 | 35.5| 62.9  | 71.5   |

As shown in Table 3, context after filtering achieves much higher precision for all tasks. Particularly for abstractive tasks, SILVER filtering increases the precision by +14.5 on HotpotQA and +60.7 on WoW. Moreover, model-filtered contexts (FILCo) are largely comparable to SILVER, and sometimes even better, such as +3.8 points in TQA. For other tasks, the small gaps between them minimally affect the end generation, as already shown in Figure 4. We conjecture these lost contents are not essential for models, particularly if they only involve common entities (Mallen et al., 2023).

However, filtering with the PSG baseline often leads to precisions lower than the FULL setting, despite the fact that it has higher output scores than FULL. Coarse granularity for context filtering may be one major reason for its loss in precision.

Shorter Inputs In Figure 6, we measure the average number of tokens in model inputs after filtering the retrieved contexts using different methods. More specifically, we do not filter context in the FULL setting, filter context by passage in the PSG setting, and filter context in the sentence level with FILCo. Model inputs contain the original query and (filtered) context. Footnote 7 effectively reduces input length by 4 – 64%.

Higher Precision To evaluate the amount of potentially redundant information in the context, we measure the unigrams precision of outputs with respect to filtered or unfiltered contexts.
```

### --- Page 0008 ---

```markdown
| Measure  | STRINC | LEXICAL | CXMI  |
|----------|--------|---------|-------|
| NQ       | 44.7   | 30.0    | 39.9  |
| TQA      | 59.2   | 39.0    | 45.3  |
| HOTTQA   | 59.2   | 57.4    | 60.0  |
| EL15     | 73.6   | 73.9    | 74.2  |
| FEVER    | 80.9   | 86.4    | 95.8  |
| WoW      | 63.4   | 69.3    | 66.6  |

| Measure  | STRINC | LEXICAL | CXMI  |
|----------|--------|---------|-------|
| NQ       | 43.3   | 35.2    | 41.8  |
| TQA      | 60.7   | 57.1    | 60.7  |
| HOTTQA   | 59.5   | 61.1    | 61.3  |
| EL15     | 78.6   | 78.8    | 72.8  |
| FEVER    | 86.6   | 88.4    | 92.3  |
| WoW      | 65.5   | 66.0    | 65.4  |

![An example in the FEVER dataset illustrating filtering outcomes using different strategies. STRINC yields empty context, LEXICAL and CXMI-filtered context are highlighted in red and green, respectively.](assets/page_0008_img_1.png)

6 Generation with Multiple Passages
It is often helpful to integrate multiple passages as context input to the model. Particularly, some tasks such as multi-hop QA may naturally necessitate using multiple passages to perform the task. To demonstrate the generality of our proposed method, we further experiment using multiple passages as source context. We experiment with FLAN-T5 since it has more consistent behaviors across tasks.

6.1 Baseline and Settings
We experiment with top-K passages, where $K = 5$, to minimize the loss from length truncation due to model input limitations, compared to larger $K$s, and hence produce more fair comparisons.

Similarly to the single-passage setting, we compare FULL and PSG as baseline methods, where FULL inputs all passages unfiltered and PSG picks zero or more passages. We also include the results of top-performing methods such as RAG (Lewis et al., 2020), FiD (Izacard and Grave, 2021), and evidentiality-guided (EV1) generation (Asai et al., 2022). In comparison to baselines, we report the sentence-wise filtering method as FILCo and the canonical setting by SILVER.
```

### --- Page 0009 ---

```markdown
6.2 Generation Performance

As shown in Table 5, our main method FilCo surpasses the full-context (FULL) and passage-filtering (PSGs) settings by a large margin, +1.2 – 14.2 points in all six tasks. FilCo also outperforms existing performance baselines. Compared to using top-1 passages only, performance increases on extractive tasks when aggregating multiple top-ranked passages. Interestingly, performance on FEVER and WoW drop by –3.2 and –2.3 points, potentially due to the decreased retrieval quality of lower-ranked passages, as the top-1 retrieval recall is relatively high.

| Context | NQ  | TQA | HotpotQA | ELI5 | FEVER | WoW  |
|---------|-----|-----|----------|------|-------|------|
| **BASLINE, TOP 5** |     |     |          |      |       |      |
| RAG     | 44.5 | 56.8 | 88.1     | 13.8 |       |      |
| FiD     | 48.3 | 67.2 | -        | 89.5 | 16.9  |      |
| Evl     | 49.8 | 67.8 | -        | 89.8 | 17.9  |      |
| **FILCo, TOP 1** |     |     |          |      |       |      |
| FilCo   | 44.7 | 59.0 | 73.8     | 94.2 | 68.3  |      |
| **FILCo, TOP 5** |     |     |          |      |       |      |
| FULL    | 47.6 | 67.3 | 61.5     | 72.1 | 88.0  | 64.8 |
| PSGS    | 52.9 | 69.1 | 62.3     | 73.7 | 90.4  | 64.6 |
| FilCo   | 61.8 | 71.1 | 65.0     | 73.9 | 91.4  | 66.0 |
| SILVER  | 62.0 | 71.1 | 65.2     | 73.9 | 92.2  | 66.1 |

Table 5: Generation results when providing top-5 retrieved passages filtered by passages or sentences. RAG, FiD, and Evl are top-performing methods. We bold-type the best results that do not use silver contexts.

7 Related Work

Augmented Generation Providing additional contexts to generation has shown to be effective (Lewis et al., 2020; Guu et al., 2020; Mialon et al., 2023) across many knowledge-intensive tasks (Petroni et al., 2021). While the most common approach with a set of retrieved passages is to append them all to the input, some works explored the optimal granularity and strategy to do this. Wang et al. (2019) identify 100 words to be the optimal size for candidate passages, which then became the de facto length. Many works explored retrieval at varied granularity, including paragraph (Lee et al., 2019; Feldman and El-Yaniv, 2019), phrase (Lee et al., 2021), and even token levels (Khandelwal et al., 2020; Alon et al., 2022), which all reveal a trade-off in difficulty between retrieval and generation: retrieving longer sequences is easier, but it is harder to generate correct output from them. In fact, Shi et al. (2023) shows that model performance can dramatically decrease when irrelevant information is included in output-supporting documents. Our method alleviates this in-passage distraction by allowing arbitrary passage sizes at retrieval time, and providing precisely useful content for generation.

Optimizing Retrieval for Augmentation Many works focus on post-process retrieved content to augment the generation. A common approach is to rerank retrieved passages and provide only the top few under limited input capacity, based on the similarity between query and passages (Nogueira and Cho, 2020), the majority of reader predictions (Mao et al., 2021), and utility for generation (Wang et al., 2018). Asai et al. (2022) measures the evidentiality of retrieved passages to improve context quality, by removing irrelevant passages and skipping the retrieval step (Mallen et al., 2023). Nonetheless, these methods operate on the coarse passage level, thus still suffering from in-passage distractions. Our method has similarities to answer sentence selection (Yu et al., 2014), which can operate at a more fine-grained sentence level. Yet further, our filtering can apply to text split in arbitrary granularity that optimizes the tasks of interest, and capture more subtle variances in context.

8 Conclusion and Future Work

We propose a context filtering method, FilCo, to provide precisely supportive content to assist model generations, which effectively removes distracting content in both passages partially supporting and irrelevant to the queries. Applying our method brings an average of 2.8 and 3.0 point increase with FLAN-T5 and LLaMA2, across six knowledge-intensive language datasets from question answering, fact verification, to knowledge-grounded dialog generation. Our work also reveals varied recipes to effectively filter context for different tasks. We hope that FilCo can facilitate more developments toward faithful generations in more scenarios.

Limitations

Our proposed method has been shown effective across various tasks, however, may be in certain data domains, under automatic evaluation metrics, and with sufficient computational resources.

Our approach is domain-agnostic in principle, however, all the datasets we experiment with are built from Wikipedia articles, i.e., the open domain. Tasks of other domains such as news (Trischler
```

### --- Page 0010 ---

```markdown
## Method Evaluation

We evaluate model retrieval, filtering, and generation performance using automatic metrics such as Exact Match and Unigram F1, which have become the standard metrics. Beyond lexical-based metrics, we keep open to neural- or human-based evaluations, given the potentially inaccurate automatic measures, especially with increasingly complex tasks (Puglisi et al., 2019) and models of greater capacities (Kamalloo et al., 2023).

Our method requires training models to (i) filter context, and (ii) generate output, which necessitates certain computational resources, according to the model architecture and size of choice. Nonetheless, our method costs less computation compared to traditional full-passage augmentation. As shown by §5, a generation model with filtered content requires at least 4.7 times less computation, at both training and inference time.

## Acknowledgements

This work was supported in part by a grant from Bosch. We thank the members of CMU LTI for their helpful discussion and feedback on this work.

## References

| Author(s) | Title | Source |
|-----------|-------|--------|
| Uri Alon, Frank F. Xu, Junxian He, Sudipta Sengupta, Dan Roth, and Graham Neubig. | Neuro-symbolic language modeling with automaton-augmented retrieval. | In ICML 2022 Workshop on Knowledge Retrieval and Language Models. |
| Akari Asai, Matt Gardner, and Hannah Hajishirzi. | Evidentiality-guided generation for knowledge-intensive NLP tasks. | In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2226–2243, Seattle, United States. Association for Computational Linguistics. |
| Eunsol Choi, Jennimaria Palomaki, Matthew Lamm, Tom Kwiatkowski, Dijana Das, and Michael Collins. | Decontextualization: Making sentences stand-alone. | Transactions of the Association for Computational Linguistics, 9:447–461. |
| Hyung Won Chung, Le Hou, Shayne Longpre, Barrett Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirae Suzy, Xinyun Chen, Akanksha Choudhary, Alex Castro-Ros, Marie Pellatt, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. | Scaling instruction-finetuned language models. | arXiv preprint arXiv:2210.11416. |
| Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. | 2019. Wizard of Wikipedia: Knowledge-powered conversational agents. | In International Conference on Learning Representations. |
| Zi-Yi Dou, Pengfei Liu, Hiroaki Hayashi, Zhengbao Jiang, and Graham Neubig. | GSum: A general framework for guided neural abstractive summarization. | In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4830–4842, Online. Association for Computational Linguistics. |
| Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. | ELI5: Long form question answering. | In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3558–3567, Florence, Italy. Association for Computational Linguistics. |
| Yair Feldman and Ran El-Yaniv. | 2019. Multi-hop paragraph retrieval for open-domain question answering. | In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 2296–2309, Florence, Italy. Association for Computational Linguistics. |
| Patrick Fernandes, Kayo Yin, Graham Neubig, and André F. T. Martins. | Measuring and increasing context usage in context-aware machine translation. | In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6467–6478, Online. Association for Computational Linguistics. |
| Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. | 2020. REALM: Retrieval-augmented language model pre-training. | In International Conference on Machine Learning. JMLR.org. |
| Edward J. Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. | 2022. LoRA: Low-rank adaptation of large language models. | In International Conference on Learning Representations. |
| Gautier Izacard and Edward Grave. | 2021. Leveraging passage retrieval with generative models for open domain question answering. | In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume. |
```

### --- Page 0011 ---

```markdown
| **Authors**                                                                 | **Title**                                                                                                   | **Source**                                                                                          |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Jinhyuk Lee, Alexander Wettig, and Danqi Chen.                            | Phrase retrieval learns passage retrieval, too.                                                             | In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3661–3672, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. |
| Kenton Lee, Ming-Wei Chang, and Kristina Toutanova.                       | Latent retrieval for weakly supervised open domain question answering.                                      | In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6806–6809, Florence, Italy. Association for Computational Linguistics. |
| Patrick Lewis, Ethan Perez, Aleksandr Piktus, Fabio Petroni, Vladimir Karpukhin, Namana Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäshel, Sebastian Riedel, and Douwe Kiela. | Retrieval-Augmented Generation for knowledge intensive NLP tasks.                                           | In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474.                    |
| Alex Mallen, Akari Asai, Victor Zheng, Rajarshi Das, Daniel Khashabi, and Hanan Hajishirzi. | When not to trust language models: Investigating effectiveness of parametric and non-parametric memory.   | arXiv preprint arXiv:2302.10511.                                                                    |
| Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen. | Reader-guided passage ranking for open-domain question answering.                                          | In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 344–350, Online. Association for Computational Linguistics. |
| Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christofores Nampanis, Ram Pasmuru, Roberta Reinala, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Aslı Celikyilmaz, Ender Ayan LeCun, and Thomas Scialom. | Augmented language models: a survey.                                                                         | arXiv preprint arXiv:2302.07842.                                                                    |
| Anastasios Nentidis, Anastasía Krithara, Georgios Paliouras, Eulália Ferrer-Madurell, Salvador Llorenç-López, and Martin Krallinger. | Biosac @ clef2023: The eleventh edition of the large-scale biomedical semantic indexing and question answering challenge. | In Advances in Information Retrieval.                                                               |
| Rodrigo Nogueira and Kyunghyun Cho.                                        | Passage re-ranking with bert.                                                                               | arXiv preprint arXiv:1901.04085.                                                                    |
| Fabio Petroni, Aleksandr Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vasilis Plachouras, Tim Rocktäshel, and Sebastian Riedel. | KILT: a benchmark for knowledge intensive language tasks.                                                   | In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2523–2545, Online. Association for Computational Linguistics. |
| Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer.             | TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension.                   | In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics. |
| Ehsan Kamalloo, Nouha Dziiri, Charles LA Clarke, and Davood Rafiei.      | Evaluating open-domain question answering in the era of large language models.                              | arXiv preprint arXiv:2305.06984.                                                                    |
| Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. | Dense passage retrieval for open-domain question answering.                                                  | In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics. |
| Urvashi Khandelwal, Angela Fan, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. | Nearest neighbor machine translation.                                                                         | In International Conference on Learning Representations.                                             |
| Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. | Generalization through memorization: Nearest neighbor language models.                                      | In International Conference on Learning Representations.                                             |
| Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. | The NarrativeQA reading comprehension challenge.                                                             | Transactions of the Association for Computational Linguistics, 6:317–328.                          |
| Mojtaba Komeili, Kurt Shuster, and Jason Weston.                          | Internet-augmented dialogue generation.                                                                      | In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8460–8478, Dublin, Ireland. Association for Computational Linguistics. |
| Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Kent Lee, Kristina Toutanova, Lilian Jones, Matthew Kelez, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. | Natural questions: A benchmark for question answering research.                                             | Transactions of the Association for Computational Linguistics, 7:452–466.                          |
```

### --- Page 0012 ---

```markdown
# PAGE_NAME: page_0012

Hemant Pugaliyya, James Route, Kaixin Ma, Yixuan Geng, and Eric Nyberg. 2019. Bend but don’t break? multi-challenge stress test for QA models. In *Proceedings of the 2nd Workshop on Machine Reading for Question Answering*, pages 125–136, Hong Kong, China. Association for Computational Linguistics.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Ohan, Ed Chi, Nathaniel Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. arXiv preprint arXiv:2302.00093.

Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. Retrieval augmentation reduces hallucination in conversation. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 3784–3803, Punta Cana, Dominican Republic. Association for Computational Linguistics.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and April Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Adam Trischler, Tong Wang, Xingdi Yuan, Justin Har- ris, Alessandro Sordoni, Philip Bachman, and Kaheer Suleman. 2017. NewsQA: A machine comprehension dataset. In *Proceedings of the 2nd Workshop on Representation Learning for NLP*, pages 191–200, Vancouver, Canada. Association for Computational Linguistics.

Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang, Tim Klinger, Wei Zhang, Shiyu Chang, Gerry Tesauro, Bowen Zhou, and Jing Jiang. 2018. R3: Reinforced ranker-reader for open-domain question answering. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 32.

Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallapati, and Bing Xiang. 2019. Multi-passage BERT: A globally normalized BERT model for open-domain question answering. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP- IJCNLP)*, pages 5878–5882, Hong Kong, China. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pier- ric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Marianna Bruna, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online. Association for Computational Linguistics.

Ying Xu, Dakuo Wang, Mo Yu, Daniel Ritchie, Bing- sheng Yao, Tongshuang Wu, Zheng Zhang, Toby Li, Nora Bradford, Branda Sun, Tran Hong, Yisi Sang, Yufang Hou, Xiaojun Ma, Diyi Yang, Nanyun Peng, Zhou Yu, and Mark Warschauer. 2022. Fantastic questions and where to find them: FairytaleQA – an authentic dataset for narrative comprehension. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 447–460, Dublin, Ireland. Association for Computational Linguistics.

Zhiling Yan, Peng Qi, Saizheng Zhang, Yosua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

Lei Yu, Karl Moritz Hermann, Phil Blunsom, and Stephen Pulman. 2014. Deep learning for answer sentence selection. arXiv preprint arXiv:1412.1632.
```


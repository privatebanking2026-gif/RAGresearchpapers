# ArXiv 2310.01352

### --- Page 0001 ---

```markdown
# RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING

**Xi Victoria Lin\*  Xilun Chen\*  Mingda Chen\***  
Weijia Shi  Maria Lomeli  Rich James  Pedro Rodriguez  Jacob Kahn  
Gergely Szilvasy  Mike Lewis  Luke Zettlemoyer  Scott Yih  

FAIR at Meta  
{victorialin,xilun,mindachen,scottyih}@meta.com  

## ABSTRACT

Retrieval-augmented language models (RALMs) improve performance by accessing long-tail and up-to-date knowledge from external data stores, but are challenging to build. Existing approaches require either expensive retrieval-specific modifications to LM pre-training or use post-hoc integration of the data store that leads to suboptimal performance. We introduce Retrieval-Augmented Dual Instruction Tuning (RA-DIT), a lightweight fine-tuning methodology that provides a third option by retrofitting any LLM with retrieval capabilities. Our approach operates in two distinct fine-tuning steps: (1) one updates a pre-trained LM to better use retrieved information, while (2) the other updates the retriever to return more relevant results, as preferred by the LM. By fine-tuning over tasks that require both knowledge utilization and contextual awareness, we demonstrate that each stage yields significant performance improvements, and using both leads to additional gains. Our best model, RA-DIT 65B, achieves state-of-the-art performance across a range of knowledge-intensive zero- and few-shot learning benchmarks, significantly outperforming existing in-context RALM approaches by up to +8.9% in 0-shot setting and +1.4% in 5-shot setting on average.

## 1 INTRODUCTION

Large language models (LLMs) excel as zero- and few-shot learners across various tasks (Brown et al., 2020; Chowdhery et al., 2022; Touvron et al., 2022b; Anil et al., 2023; OpenAI, 2023). However, because knowledge is represented only in the model parameters, they struggle to capture long-tail knowledge (Trimble et al., 2022; Sun et al., 2023) and require substantial resources to be kept up-to-date (Miller, 2023). Retrieval-Augmented Language Modeling (RALM) integrates LLMs with non-parametric information retrieval to overcome these limitations (Guu et al., 2020; Borgeaud et al., 2022; Izacard et al., 2022b; Shi et al., 2023; Ram et al., 2023). By explicitly decoupling knowledge retrieval with the backbone language model, such architectures have exhibited superior performance on knowledge intensive tasks such as open-domain question answering (Lewis et al., 2020; Izacard et al., 2022b) and live chat interactions (Liu, 2022).

Existing RALM architectures focus on two high-level challenges: (i) enhancing the LLM's capability to incorporate retrieved knowledge (Lewis et al., 2020; Izacard et al., 2022b) and (ii) refining the retrieval component to return more relevant content (Shi et al., 2023b; Izacard et al., 2022). Previous work has also introduced retrieval capabilities at different stages of the model training process. REALM (Guu et al., 2020) and RETRO (Borgeaud et al., 2022) opt for end-to-end training, incorporating the retrieval component from the outset. As (Izacard et al., 2022b) builds upon the T5 language model (Raffel et al., 2020), and continuously pre-trains the framework over unsupervised text. REP-LM (Shi et al., 2023a) and In-Context RALM (Ram et al., 2023) combine off-the-shelf LLMs with general-purpose retrievers, showing that these two components can be effectively used through the emergent in-context learning capabilities of LLMs. However, extensive pre-training of such architectures is expensive, and the off-the-shelf fusion approach also has limitations, particularly as the LLMs are not inherently trained to incorporate retrieved content.

\*Equal contribution
```

### --- Page 0002 ---

```markdown
![Figure 1: The RA-DIT approach separated fine-tunes the LLM and the retriever. For a given example, the LM-ft component updates the LLM to maximize the likelihood of the correct answer given the retrieval-augmented instructions. The R-ft component updates the retriever to minimize the KL-Divergence between the retriever score distribution and the LLM reference.](assets/page_0002_img_1.png)

In this work, we show lightweight instruction tuning (Chung et al., 2022b; Iyer et al., 2022; Zhou et al., 2023) also can significantly boost the performance of RALMs, especially in knowledge intensive scenarios. We propose Retrieval-Augmented Dual Instruction Tuning (RA-DIT), an approach that retrofits any LLM with retrieval capabilities via fine-tuning over a set of tasks selected to cultivate knowledge utilization and contextual awareness in the language model predictions. We initialize the framework using pre-trained LLaMA (Touvron et al., 2023a) and a state-of-the-art dual-encoded dense retriever, DRAGON+ (Lin et al., 2023). Following Shi et al. (2023b), we retrieve relevant text chunks based on the language model prompt. Each retrieved chunk is processed in parallel and concatenated to produce the final output.

We reform instruction-tuning in two separate steps. For language model fine-tuning (LM-ft), we adopt the label-loss objective (Chung et al., 2022b; Iyer et al., 2022) and augment fine-tuning prompt with a retrieved “background” field prepped according to the instruction prompt. We also leverage the design of existing NLP tasks and populate this field with the ground truth context for tasks such as reading comprehension and summarization. By incorporating the background text during fine-tuning, we guide the LLM to optimally utilize the retrieved information and ignore distracting content (Shi et al., 2023a). For retriever fine-tuning (R-ft), we update the query encoder using a generalized LM-Supervised Retrieval (LSR, Shi et al., 2023b) training objective computed over a combination of supervised tasks and unsupervised text completion. This way we enable the retriever to yield more contextually relevant results, aligned with the preferences of the LLM.

We demonstrate that fine-tuning both gives significant performance gains, and that the fine-tuned LLM and retriever can be combined to achieve further improvements. Our largest model, RA-DIT 65B, attains state-of-the-art performance in zero- and few-shot settings on knowledge intensive benchmarks, notably surpassing the tuned-in-context RALM approach on datasets including MMLU (Hendrycks et al., 2021a) (+8.2% 0-shot; +0.7% 5-shot) and Natural Questions (Kwiatkowski et al., 2019) (+22% 0-shot; +3.8% 5-shot). In addition, RA-DIT 65B also substantially outperforms ATLAS 118B on 8 knowledge-intensive tasks (+7.2% on average in the 64-shot fine-tuning setting). This suggests that language models and retrievers, when optimized independently and then fused through instruction-tuning, can compete effectively with RALMs that have undergone extensive continuous pre-training. We further conduct a comprehensive model analysis, showing the effectiveness of our approach across LLMs of varying sizes, as well as evaluating the influence of different fine-tuning strategies and retriever configurations.¹

¹ We release the scripts for indexing Common Crawl data and generating our fine-tuning and inference prompts at: https://github.com/facebookresearch/RA-DIT.
```

### --- Page 0003 ---

```markdown
## 2 METHOD

### 2.1 ARCHITECTURE

Language Model We focus on retrieval-augmenting pre-trained auto-regressive language models (Brown et al., 2020). In particular, we use LLAMA (Touvron et al., 2023a), a family of open-sourced language models pre-trained on millions of tokens.

Retriever We adopt a dual-encoder based retriever architecture, since it can be easily fine-tuned and is efficient at the inference stage (Lewis et al., 2020; Izacard et al., 2022b; Shi et al., 2023b). Given a corpus $C$ and a query $q$, the document encoder maps each text chunk $c \in C$ to an embedding $E_d(c)$ and the query encoder maps $q$ to an embedding $E_q(q)$. The top-k relevant texts for $q$ are retrieved based on the query-document embedding similarity, which is often computed via dot product:

$$
s(q, c) = E_q(q) \cdot E_d(c). \tag{1}
$$

We initialize the retriever using DRAGON+ (Lin et al., 2023), a state-of-the-art dual-encoder model trained with a contrastive learning objective and large-scale data augmentation.

Parallel In-Context Retrieval-Augmentation Following Shi et al. (2023b), for a given language model prompt $x$, we retrieve the top-k relevant text chunks $C' \subset C, |C'| = k$. To stay within the context window size limit, each retrieved chunk is prepended to the prompt $x$, and the language model predictions from multiple augmented prompts are computed in parallel. The final output probability is a mixture of the probability from each augmented prompt weighted by the chunk relevance score:

$$
PLM(y|x, C') = \sum_{c' \in C'} PLM(y|c \odot x) \cdot p_R(c|x), \tag{2}
$$

where $c$ denotes sequence concatenation, and $p_R(c|x) = \frac{\exp a(c)}{\sum_{c' \in C'} \exp a(c')}$ are the retriever scores re-normalized among top-k relevant chunks.

### 2.2 FINE-TUNING DATASETS

We choose a set of fine-tuning tasks aimed at boosting the language model’s ability to utilize knowledge effectively and improving its contextual awareness in generating predictions. As shown in Table 1, our language model fine-tuning datasets ($D_L$) consists of 20 datasets across 5 distinct categories: dialogue, open-domain QA, reading comprehension, summarization and chain-of-thought reasoning. For retriever fine-tuning datasets $D_R$, we opt for the QA datasets in our collection featuring standalone questions, and we additionally include two QA datasets, FreebaseQA (Jiang et al., 2019) and MS-MARC (Nogueira et al., 2016). The examples of each dataset are serialized for instruction tuning using manually compiled templates (Table 10). For tasks in $D_L \cap D_R$, we use the same template for both fine-tuning tasks. In addition, we observe that supplementing the instruction-tuning data with unsupervised text leads to additional performance gains for both language model and retriever fine-tuning, and we detail data mixture used in Appendix B.

### 2.3 RETRIEVAL AUGMENTED LANGUAGE MODEL FINE-TUNING

To improve the language model’s ability to utilize retrieved information, we fine-tune it on selected datasets $D_L$ with in-context retrieval augmentation. Formally, we separate each fine-tuning sequence into an instruction segment ($x$) and an output segment ($y$). For each example $(x_i, y_i) \in D_L$:

2 We use a pair of text (“Background”) and end(“\n\n”) tokens to demarcate the retrieved segment in the augmented prompt. The complete set of our instruction-tuning templates are shown in Appendix C.

3 Our reading comprehension (RC) fine-tuning datasets include SQuAD 2.0 (Rajpurkar et al., 2018), which is the standard method to determine whether a question can be answered using a given passage, and to provide an answer only when the passage is relevant (otherwise the response is set to “I don’t know”). As shown in Appendix F, fine-tuning on this dataset promotes a desirable behavior: the instruction-tuned model tends to respond with “I don’t know” when the retriever presents an incorrect passage. We leave further exploring this behavior to improve answer generation as a future work.
```

### --- Page 0004 ---

```markdown
# Table 1: Our instruction tuning datasets. All datasets are downloaded from Hugging Face (Lhoest et al., 2021), with the exception of those marked with †, which are taken from Iyer et al. (2022).

| Task                | HF identifier | Dataset name                                      | D_t | D_R | Train  |
|---------------------|---------------|--------------------------------------------------|-----|-----|--------|
| Dialogue            | oasst1       | OpenAssistant Conversations Dataset (Kopf et al., 2023) | -   | -   | 51,589 |
| Open-Domain QA      | commoncrawl   | CommonsenseQA (Talmor et al., 2019)              | -   | -   | 9,741  |
|                     | web.questions  | Web Questions (Berant et al., 2013)              | -   | -   | 3,778  |
|                     | wiki.qa       | Wiki Question Answering (Yang et al., 2015)      | -   | -   | 20,360 |
|                     | yahoo.answers  | Yahoo! Answers QA (Jiang et al., 2019)           | -   | -   | 27,362 |
|                     | freebase.qa   | FreebaseQA (Jiang et al., 2019)                  | -   | -   | 20,358 |
|                     | ms.marco      | MS MARCO (Nguyen et al., 2016)                   | -   | -   | 80,143 |
|                     | coqa          | Conversational Question Answering (Reddy et al., 2019) | -   | -   | 10,647 |
|                     | drop          | Discrete Reasoning Over Paragraphs (Dua et al., 2019) | -   | -   | 77,400 |
| Reading Comprehension | newsqa      | NewsQA (Trischler et al., 2017)                  | -   | -   | 74,160 |
|                     | pubmed.qa     | PubMedQA (Jin et al., 2019)                       | -   | -   | 1,000  |
|                     | quad          | QA for Artificial Intelligence (Rogers et al., 2020) | -   | -   | 10,246 |
|                     | quac          | QuAC (Radford et al., 2019)                       | -   | -   | 1,914  |
|                     | squad.v2      | SQuAD v2.0 (Rajpurkar et al., 2018)               | -   | -   | 130,319 |
| Summarization       | cnndm         | CNN/DailyMail (Nallapati et al., 2015)           | -   | -   | 287,171 |
|                     | algebra.qa    | Algebra QA with Rationales (Ling et al., 2017)   | -   | -   | 7,467  |
| Chain-of-Thought    | gsm8k         | Grade School Math (Cobbe et al., 2021)           | -   | -   | 7,473  |
| Reasoning           | competition.math | MATH (Hendrycks et al., 2021b)                  | -   | -   | 7,500  |
|                     | strategyqa    | StrategyQA (Geva et al., 2021)                   | -   | -   | 2,290  |

*We only used the question-and-answer pairs in the MS MARCO dataset.*

In addition to fine-tuning the language model with retrieval augmentation, we also fine-tune the retriever to better align its output with the language model. In particular, we adopt a generalized version of LSR (LM-Supervised Retrieval, Shi et al., 2023b) training that leverages the language model itself to provide supervision for retriever fine-tuning.

For a training sample $(x, y)$ in the retriever fine-tuning dataset $D_R$, we define the LSR score for a retrieved chunk as follows:

$$
p_{LSR}(c|x) = \frac{\exp(p_{LM}(y|c \odot x)/\tau)}{\sum_{c' \in C} \exp(p_{LM}(y|c' \odot x)/\tau)} \approx \frac{\exp(p_{LM}(y|c \odot x)/\tau)}{\sum_{c' \in C'} \exp(p_{LM}(y|c' \odot x)/\tau)}
$$

where $\tau$ is a temperature hyperparameter, and $C$ denotes the top-k retrieved chunks for $x$. A higher LSR score indicates that $c$ is more effective at improving the language model's chance of the original example.
```

### --- Page 0005 ---

```markdown
predicting the correct answer. The goal of LSR training is for the retriever to assign higher scores to chunks that can improve the LLM's likelihood of generating the correct answer. To achieve this, we minimize the KL-divergence between $p_{LSR}$ and the retriever scores $p_R$ defined in Eq. 2:

$$
L(D_R) = \mathbb{E}_{(x,y) \in D_R} KL(p_R(c|x) \| p_{LSR}(c|x,y)) \tag{5}
$$

In practice, we only update the query encoder of the retriever, as fine-tuning both encoders hurts the performance (§5.1). While previous work (Shi et al., 2023b) relies solely on unlabeled texts (denoted as corpus data) for LSR training, we show that LSR can be generalized to incorporate the multi-task instruction data introduced in §2.2 (denoted as MTI data). The MTI data provide direct supervision to the retriever to return relevant information that enhances the language model in various downstream tasks. As shown in §5.1, combining both types of data yields the best results and outperforms using either source alone.

## 3 EXPERIMENT SETUP

### 3.1 RETRIEVER

We initialize the retriever in our framework with DRAGON+ (Lin et al., 2023) and also use it to study various retriever configurations. To build the retriever corpus, we combine the text chunks from the Dec. 20, 2021 Wikipedia dump released by Izacard et al. (2022b) with additional ones from the 2017-2020 CommonCrawl dumps. We detail the corpus pre-processing and indexing in Appendix A. Our final retrieval data store, with the two data sources combined, contain 399M text chunks with a maximum length of 200 words. In Appendix E.3, we conduct an analysis on the impact of using various subsets of the retrieval corpus, as well as different Wikipedia snapshots. We obtain the retrieval queries used for our fine-tuning and evaluation tasks using manually6 constructed templates (Table 10 and 12).

### 3.2 BASELINES

We focus on comparing our approach to the base Llama models (Touvron et al., 2023a) and RE-PLUG (Shi et al., 2023b), a state-of-the-art approach that integrates off-the-shelf LLMs and retrievers, in the zero-shot and in-context few-shot learning settings. We instantiate RE-PLUG using Llama and DRAGON+. In addition, we also compare RA-DIT to ATLAS (Izacard et al., 2022b) in a 64-shot fine-tuning setting (§4).

### 3.3 EVALUATION

We primarily conduct evaluation on knowledge-intensive tasks that are not included in our fine-tuning datasets, including MMLU (Hendrycks et al., 2021a), Natural Questions (NQ; Kwiatkowski et al., 2019), TriviaQA (TQA; Joshi et al., 2017), and a subset of the tasks in the KILT benchmark (Petroni et al., 2021). We use the development split of the KILT subset excluding ELI5 to determine fine-tuning hyperparameters (Appendix B). This enables us to report genuine few-shot evaluation results for 4 out of the 10 evaluation tasks. For the remaining tasks, we report few-shot results assuming access to in-domain development data. In addition, we also evaluate the models on commonsense reasoning tasks to measure the impact of the proposed approach on the LLM's parametric knowledge and reasoning capabilities. Details of our evaluation datasets, including the evaluation metrics, template and the scoring functions used, can be found in Appendix D.

## 4 MAIN RESULTS

### Knowledge-Intensive Tasks

We report the main results in Table 2. In particular, RA-DIT is compared to Llama (Touvron et al., 2023a) as well as RE-PLUG (Shi et al., 2023b), in both 0-shot and
```
![Detailed description of the chart](assets/page_0005_img_1.png)
```

### --- Page 0006 ---

```markdown
| Table 2: Main results: Performance on knowledge intensive tasks (test sets). |
|-------------------------------------------------------------------------------|
| MMLU | NQ  | TQA | ELI5 | HoPo | FEY | AIDA | zRE | T-REx | WoW | Avg* | Avg  |
|------|-----|-----|------|------|-----|------|-----|-------|-----|------|------|
| LLAMA 65B      | 51.2 | 52.5 | 55.8 | 19.5 | 15.2 | 59.3 | 0.6 | 6.7 | 1.3 | 15.6 | 32.9 | 22.8 |
| LLAMA 65B RePLUG | 59.7 | 28.8 | 72.6 | 19.1 | 32.0 | 73.4 | 18.0 | 36.3 | 16.1 | 45.1 | 43.1 |
| RA-DIT 65B     | 64.6 | 35.2 | 75.4 | 2.12 | 39.7 | 80.7 | 4.51 | 7.3 | 53.1 | 16.4 | 49.1 | 50.5 |

| 5-shot in-context |
|--------------------|
| LLAMA 65B      | 63.4 | 31.6 | 71.8 | 22.6 | 81.5 | 48.2 | 39.4 | 52.1 | 17.4 | 47.2 | 45.0 |
| LLAMA 65B RePLUG | 64.4 | 42.3 | 74.9 | 22.8 | 41.1 | 89.4 | 46.0 | 64.0 | 16.8 | 51.1 | 52.7 |
| RA-DIT 65B     | 64.9 | 43.9 | 75.1 | 23.2 | 40.7 | 90.7 | 55.8 | 72.64 | 17.3 | 51.8 | 52.5 |

| Table 3: Performance on commonsense reasoning tasks (dev sets) without retrieval augmentation. |
|------------------------------------------------------------------------------------------------|
| 0-shot | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-E | OBOA | Avg |
|--------|-------|------|------|-----------|------------|-------|------|-----|
| LLAMA 65B      | 85.3 | 82.8 | 52.3 | 84.2 | 77.0 | 78.9 | 56.0 | 60.2 | 72.1 |
| LLAMA 65B RePLUG | 86.7 | 83.7 | 57.9 | 85.1 | 79.8 | 83.7 | 60.5 | 68.8 | 74.5 |

5-shot settings. We first observe that RePLUG works much better than the base LLAMA 65B, confirming the benefits of RALMs on knowledge-intensive tasks. Furthermore, RA-DIT significantly outperforms RePLUG (+8.9% in 0-shot and +1.4% in 5-shot on average over MMLU, NQ, TQA and ELI5) and achieves the best performance on most datasets. This corroborates our claim that combining off-the-shelf LLMs and retrievers is sub-optimal, and our dual instruction tuning approach is an effective way of retrofitting LLMs with retrieval capabilities.

We also compare with ATLAS, a state-of-the-art encoder-decoder based RALM that jointly pretrains the language model and the retriever. Here we adopt a 64-shot setting similar to Laczard et al. (2022b) with the following differences. While ATLAS conducts 64-shot fine-tuning for each individual task and reports the performance of task-specific models, we continuously fine-tune the RA-DIT checkpoint using the 64-shot examples from all tasks combined, and report the performance of a single model across tasks. As shown in Table 2, despite using a single model, RA-DIT outperforms ATLAS by an average of 4.1 points, achieving higher performance on 6 out of the 8 datasets.

Commonsense Reasoning We benchmark RA-DIT 65B on a set of commonsense reasoning tasks to evaluate the impact of retrieval-augmented instruction tuning on the LLM’s parametric knowledge and reasoning capabilities. We hence do not perform retrieval augmentation in this experiment. As shown in Table 3, RA-DIT demonstrates improvements over the base LLAMA models on 7 out of 8 evaluation datasets, indicating that the parametric knowledge and reasoning capabilities of the LLM component are in general preserved. As discussed in Appendix F, maintaining the parametric knowledge in the LLM component is vital as a safety net when the retriever makes mistakes.

5. ANALYSIS

5.1 FINE-TUNING STRATEGIES

Language Model Fine-tuning We compare LLAMA instruction-tuned with retrieval augmentation (RA-DIT 65B) to the base language model, as well as LLAMA that is instruction-tuned.
```

### --- Page 0007 ---

```markdown
| **Table 4:** Ablation of language model fine-tuning strategies. All rows report dev set performance. |
|----------------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 0 / 5-shot     | HoPo | FEV  | AIDA | zRE  | T-REX| WoW  | Avg  |
| LLaMA 65B      | 12.5 | 23.8 | 38.8 | 0.9  | 64.1 | 97.360| 12.523| 15.717| 16.464| 16.46| 29.46|  |
| IT 65B         | 200.0| 200.0| 67.832| 8.9 | 58.5 | 109.4 | 173.53| 16.4| 16.5| 29.46|  |
| RA-FT 65B      | 26.8 | 29.9 | 62.584| 10.7| 52.9 | 30.9 | 35.2 | 24.152| 15.16| 29.46|  |
| **top-1 / chunk** |      |      |      |      |      |      |      |
| LLaMA 65B + DRAGON* | 25.8 | 39.4 | 72.898| 39.1 | 50.7 | 48.8 | 51.469| 15.871| 19.30| 54.3|  |
| IT 65B + DRAGON* | 33.3 | 38.8 | 70.901| 39.503| 56.582| 44.767| 15.156| 46.532|  |
| RA-FT 65B + DRAGON* | 37.6 | 39.1 | 10.904| 41.623| 59.67| 49.658| 16.6| 16.67| 47.537|  |
| **top-5 / chunks** |      |      |      |      |      |      |      |
| LLaMA 65B + DRAGON* | 29.6 | 40.7 | 90.3 | 41.528| 55.9 | 62.7 | 37.70| 16.6| 17.6| 55.8|  |
| IT 65B + DRAGON* | 35.2 | 40.8 | 57.912| 49.523| 62.61| 39.68| 15.6| 15.68| 41.549|  |
| RA-FT 65B + DRAGON* | 39.9 | 40.8 | 24.912| 52.634| 63.61| 63.58| 16.6| 16.76| 50.157|  |

| **Table 5:** Ablation of retriever fine-tuning strategies. All rows use the LLaMA 65B model and report 5-shot performance on the dev sets. |
|----------------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| **top-1**      | MMLU | NQ   | TQA  | HoPo | FEV  | AIDA | zRE  | T-REX| WoW  | Avg  |
| DRAGON*        | 62.6 | 41.5 | 72.9 | 41.5 | 90.6 | 54.1 | 63.7 | 72.1 | 17.5 | 56.6 |
| MTL instruction tuning | 61.4 | 36.5 | 91.4 | 66.6 | 67.2 | 72.1 | 57.5 |  |
| corpus data (FTI benchmarks) | 61.7 | 42.3 | 73.8 | 42.8 | 68.9 | 53.5 | 17.5 | 56.6 |
|  | 62.9 | 41.1 | 81.6 | 91.6 | 54.2 | 17.5 | 56.6 |
|  | 62.3 | 42.1 | 74.912| 54.9 | 65.2 | 71.6 | 17.5 | 57.0 |

* Average over the 6 KILT development tasks.

conventionally$^8$ (IT 65B) on the same set of tasks. We evaluate all models with in-context retrieval augmentation using the DRAGON* retriever, adjusting the number of retrieved chunks to 0, 1 or 10. As shown in Table 4, while both instruction tuning methods substantially enhance the 5-shot performance, their offers marginal improvements or even hurt the model performance in the 5-shot setting for most tasks except for HotPotQA$^9$. When in-context retrieval-augmented instruction is applied, all models show substantial gains in both settings, even when limited to the top-1 chunk. The model performance consistently improves as we include more retrieved chunks. In the 0-shot setting with top-10 retrieved chunks, the RA-IT 65B model outperforms the IT 65B model by a large margin (51.0 vs. 47.7%). This observation aligns with the findings from previous instruction-tuning literature (by Ert et al., 2022). HotPotQA is an exception likely because it is from a task category covered in our instruction-tuning data.
```

### --- Page 0008 ---

```markdown
# Published as a conference paper at ICLR 2024

## 5.2 DUAL INSTRUCTION TUNING ABLAION

Table 6: The impact of LM and Retriever fine-tuning in our RA-DIT method, comparing the RE-PLUG baseline, LM-ft only, R-ft only, and RA-DIT. 5-shot dev set performance is reported.

| 5-shot  | MMLU | NQ  | TQA | HoPo | FEV | AIDA | zRE | T-REX | WoW | Avg  |
|---------|-------|-----|-----|------|-----|------|-----|-------|-----|------|
| LLAMA 65B + Dragon*  | 61.7  | 41.7 | 30.2 | 22.1 | 41.6 | 90.8 | 54.0 | 63.7 | 71.9 | 172.2 | 53.8 |
| LLAMA 65B + FHeD Dragon* | 63.0  | 42.2 | 74.9 | 22.2 | 41.4 | 91.6 | 54.9 | 65.1 | 71.4 | 174.4 | 54.4 |
| RIT 65B + Dragon* | 64.8  | 42.8 | 73.1 | 23.6 | 41.2 | 53.6 | 62.9 | 69.0 | 16.6 | 53.9 |
| RIT 65B + FHeD Dragon* | 64.3  | 43.8 | 75.0 | 23.4 | 42.0 | 92.3 | 52.8 | 62.5 | 70.1 | 17.3 | 54.6 |

We isolate the impact of the language model fine-tuning from retriever fine-tuning in our RA-DIT method, and illustrate the benefit of each. 11 According to Table 6, both LM-ft and R-ft are beneficial when used alone, and outperform the RePLUG using LLAMA 65B and the Dragon* retriever. On the other hand, the most gain can be achieved when combining LM-ft and R-ft in our RA-DIT method, which outperforms the RePLUG baseline by 0.8 points on average. In our preliminary experiments, we also attempted dual instruction tuning by fine-tuning the retriever using LSR scores from the RA-IT LM or conduct the RA-IT step using passages returned by the fine-tuned retriever, for one or two such iterations, but did not observe further gains. We leave the exploration of multi-step RA-DIT to future work.

## 5.3 RETRIEVER SETTINGS

Table 7: Retriever settings: We report 5-shot dev set performance using LLAMA 65B and various retrievers in the RePLUG setting.

| 5-shot  | MMLU | NQ  | TQA | HoPo | FEV | AIDA | zRE | T-REX | WoW | Avg  |
|---------|-------|-----|-----|------|-----|------|-----|-------|-----|------|
| LLAMA 65B | 61.3  | 30.9 | 70.6 | 23.8 | 83.7 | 50.2 | 36.0 | 52.3 | 17.4 | 45.0 |

Retriever ablation using LLAMA 65B and the 399M CC + Wiki corpus

| Retriever | MMLU | NQ  | TQA | HoPo | FEV | AIDA | zRE | T-REX | WoW | Avg  |
|-----------|-------|-----|-----|------|-----|------|-----|-------|-----|------|
| Contriever | 59.3  | 41.2 | 73.0 | 32.4 | 88.1 | 45.0 | 56.1 | 17.2 | 21.6 | 47.5 |
| Contriever-msmarco | 62.0  | 42.1 | 74.1 | 37.8 | 89.3 | 49.0 | 62.9 | 17.4 | 21.8 | 51.8 |
| Dragon* | 61.7  | 41.7 | 73.0 | 40.8 | 90.8 | 48.8 | 63.7 | 71.9 | 17.8 | 53.4 |

We study the impact of various retriever choices in our framework. We use LLAMA 65B as the language model and combine it with different retrievers. Table 7 first compares Dragon* (Lin et al., 2023) with other state-of-the-art retrievers such as Contriever (Izacard et al., 2022a). All retrieval-augmented models substantially improve over the LLAMA baseline, and Dragon* significantly outperforms both Contriever and Contriever-MSMARCO. We hence adopt Dragon* as our base retriever in all experiments.

## 6 RELATED WORK

### Retrieval-Augmented Language Models
RALMs augment LMs with a non-parametric memory to facilitate external knowledge access and provide provenance (Guu et al., 2020; Lewis et al., 2020; ...)

10 In early experiments, we also tested other mixtures and found that using 5% or 10% MTI data worked the best. (They perform similarly to each other.)
11 Minor performance differences may be observed for the LLAMA 65B + Dragon* model in different ablations due to the differences in few-shot example truncation in long prompts. We ensure all rows within the table are comparable.
```

### --- Page 0009 ---

```markdown
Published as a conference paper at ICLR 2024

Borgeaud et al., 2022; Shi et al., 2023b). Previous work has proposed different ways of fusing the LM and the non-parametric component. For example, RETRO (Borgeaud et al., 2022) and FiD (Izacard & Grave, 2021; Höffert et al., 2022) leverage separate encoder modules to encode the retrieved content, which are integrated with the backbone LM via cross-attention. A more widely adopted approach directly augments the LM input with the retrieved content (Guu et al., 2020; Lewis et al., 2020; Shi et al., 2023b). This approach yields competitive results with a moderate inference cost increase, as the LM can effectively contextualize the retrieved content and the original prompt through multi-layer self-attention. RA-DIT is grounded in the in-context RA framework for its simplicity and practicality. Instead of performing extensive pre-training (Guu et al., 2020; Borgeaud et al., 2022; Izacard et al., 2022b), we propose a lightweight fine-tuning recipe that primarily utilizes downstream data, and demonstrate improved few-shot generalization of the fine-tuned RALM on knowledge-intensive language tasks.

### Instruction Tuning

Instruction tuning has been proposed to align pre-trained LLMs to follow natural language instructions and avoid extensive prompt engineering (Ouyang et al., 2022; Wei et al., 2022; Chung et al., 2022; Wang et al., 2022; Ye et al., 2022). We propose retrieval-augmented instruction tuning (RA-IT) as part of our dual instruction tuning framework to improve the LM's ability to leverage retrieved information. Concurrent work has also applied instruction tuning to other RALM architectures. Notably, Wang et al. (2023) fine-tunes the backbone LM in the RETRO framework while freezing the cross-attention module and the memory encoder. In comparison, RA-DIT fine-tunes both the LM and the retriever while decoupling the fine-tuning processes of the two components.12 Asai et al. (2023) fine-tunes an LM to adaptively retrieve passages on demand and reflect on the relevance of the retrieved passages and its generation using special-token markups. The most related work to ours is SAIL (Luo et al., 2023), an approach that fine-tunes the LM with instructions augmented with retrieved content, and examines it on public instruction benchmarks (Taoir et al., 2023; Chiang et al., 2023) using a moderately sized model (78 parameters). In contrast, RA-DIT performs a dual retrieval-augmented instruction tuning on passages while SAIL concatenates them in the LM context. Furthermore, RA-DIT adopts a holistic view of the RALM architecture by employing a learnable neural retriever and proposing a dual optimization framework. SAIL, in comparison, leans on non-differentiable retrievers such as BM25 and focuses on improving the LM (e.g., it proposes an in-context retrieval selection technique to guide the model focus towards informative content).

### Information Retrieval

Retrieval methods include sparse retrievers that does matching over a sparse bag-of-words representation (Robertson & Zaragoza, 2009; Formal et al., 2021), dense retrievers that embed queries and documents into a fixed-size dense vector for nearest-neighbor search (Karpukhin et al., 2020; Xiong et al., 2021), and multi-vector retrievers which uses multiple vectors as the representation and more complex search algorithms for increased accuracy (Khatak et al. & Zaharia, 2020; Li et al., 2023). We adopt a state-of-the-art dense retriever, DRAGON (Lin et al., 2023), as our base retriever, because of its simplicity, state-of-the-art accuracy, high retrieval efficiency on GPUs, and the ease of further fine-tuning.

### 7 CONCLUSION

In this paper, we propose RA-DIT, a lightweight Retrieval-Augmented Dual Instruction Tuning framework that can effectively retrofit any pre-trained LLM with retrieval capabilities. RA-DIT updates the LLM with retrieval-augmented instruction tuning to make better use of retrieved knowledge and ignore irrelevant or distracting information. It also fine-tunes the retriever with supervision from the LLM to retrieve texts that can better help the LLM generate correct outputs. RA-DIT achieves state-of-the-art performance in zero- and few-shot evaluations on knowledge intensive benchmarks, surpassing un-tuned in-context RALM approaches such as ReFLUG and compete effectively against methods that require extensive pre-training such as ATLAS.

12 Although the differences in the base LMs, fine-tuning datasets and inference settings make direct comparisons between the two models challenging, RA-DIT 65B compares favorably to InstructRetro 48B (Wang et al., 2023) in zero-shot setting on the shared evaluation datasets.
```

### --- Page 0010 ---

```markdown
# REFERENCES

Shourya Aggarwal, Diyanshu Mandowara, Vishwajeet Agrawal, Dinesh Khandelwal, Parag Singla, and Dinesh Garg. Explanations for CommonsenseQA: New dataset and models. In Chengping Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021*, pp. 3050–3065. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021-Long-238. URL [https://doi.org/10.18653/v1/2021-Long-238](https://doi.org/10.18653/v1/2021-Long-238).

Aida Amini, Saadia Gorbani, Shanchun Lin, Riki Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 2357–2367, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1245. URL [https://aclanthology.org/N19-1245](https://aclanthology.org/N19-1245).

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Samik Shaker, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michelle Catasta, Yong Cheng, Colin Cherry, Christopher A. Chouquette-Chad, Aakanksha Chowdhery, Clement Cryer, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vaid Feinberg, Fangxiao Guo, Guri Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Kahn, Yikang Liu, Shikha Kuduk, Chang Lin, Katherine Lee, Benjamin Lee, Eric Li, Maxin Liu, YaGuan Li, Jian Li, Heyu Chen, Lanhua Lin, Zhongtao Liu, Frederick Liu, Marcello M. Mazzocchi, Aidan Meisner, Vedant Misra, Maysam Mousselly, Zachary Nado, John Nhan, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pelant, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ross, David R. Xo, Brennan Sate, Rajkumar Samuel, Renee Shelby, Ambrose Sione, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijaya Vasudevan, Yordhalli, Xuezh Wnag, Pidong Wang, Zirui Wang, Tuo Wang, John Wieting, Yuhaui Wu, Keliu Xu, Yuhan Xu, Linting Xue, Pengcheng Yin, Jianhui Wu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Salvatore Petros, and Yonghui Wu. PALM 2 technical report, 2023.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sii, and Hannaneh Hajishirzi. Self-reg: Learning to retrieve, generate, and critique through self-reflection, 2023.

Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-answer pairs. In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pp. 1533–1544, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URL [https://aclanthology.org/D13-1160](https://aclanthology.org/D13-1160).

Yaozong Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In *Thirty-Fourth AAAI Conference on Artificial Intelligence*, 2020.

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruthford, Katie Millen, Léonard van der Driessche, Jean-Baptiste Lespiau, Bogdan Danciu, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Henighan, Saffron Huang, Loren Maggiore, Chris Olah, Aidan Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of tokens. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), *International Conference on Machine Learning, ICLR 2022, 17-23 July 2022, Baltimore, Maryland, USA*, volume 162 of *Proceedings of Machine Learning Research*, pp. 2262–2240. PMLR, 2022. URL [https://proceedings.mlr.press/v162/borgeaud22a.html](https://proceedings.mlr.press/v162/borgeaud22a.html).
```

### --- Page 0011 ---

```markdown
Published as a conference paper at ICLR 2024

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafula Dhariwal, Arvind Neelakantan, Pranay Singh, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Chad, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bcf496714bfb8ac14f264fa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bcf496714bfb8ac14f264fa-Paper.pdf).

Wei-Lin Chiang, Zhuohan Li, Zi Li, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing GPT-4 with 90%+ chatgpt quality. March 2023. URL [https://lmsys.org/blog/2023-03-30-vicuna/](https://lmsys.org/blog/2023-03-30-vicuna/).

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Shun, Ksenia Shasha, Sasha Tsayvashchenko, Aayush Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Kanupriya Shazeer, Vindhya Kumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Renée Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Odukoya, Lervin Skaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Heyotenk Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agarwal, Mark Ommerick, Andrew M. Dai, Thamalayan Sankaranarayanan Pillai, Marie Peliat, Alzhor Lewkowycz, Eric M. Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhil Wang, Brendan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, and Jeff Dean. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2202.02311, 2022.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Li, Kuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Li, Zhuyun Dai, Mirac Suzgur, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Peliat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022b. URL [https://arxiv.org/abs/2210.11416](https://arxiv.org/abs/2210.11416).

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploiting the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1300. URL [https://aclanthology.org/N19-1300](https://aclanthology.org/N19-1300).

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try arc, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05437, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Basar, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reichiho Nakan, Christopher Hess, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. URL [https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168).

Emily Dinan, Stephen Roller, Kurt Schuster, Angela Fan, Michael Auli, and Jason Weston. Wizard of wikipedia: Knowledge-powered conversational agents. In International Conference on Learning Representations, 2019. URL [https://openreview.net/forum?id=1713RjKfm](https://openreview.net/forum?id=1713RjKfm).
```

### --- Page 0012 ---

```markdown
# Published as a conference paper at ICLR 2024

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 2368–2378, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1246. URL https://aclanthology.org/N19-1246.

Hady Elsahar, Pavlos Viouglouklis, Arslan Ramedi, Christophe Gravier, Jonathon Hare, Frederique Laforest, and Elena Simperl. T-REx: A large scale alignment of natural language with knowledge base triples. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan, May 2018. European Language Resources Association (ELRA). URL https://aclanthology.org/L18-1544.

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. ELI5: Long form question answering. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 3558–3567, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1346. URL https://aclanthology.org/P19-1346.

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. Spalde: Sparse lexical and expansion model for first stage ranking. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*. Association for Computing Machinery, 2021. ISBN 9781450380379. doi: 10.1145/3408835.3463098. URL https://doi.org/10.1145/3408835.3463098.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. *Transactions of the Association for Computational Linguistics*, 9:346–361, 2021. doi: 10.1162/tacl_a_00221. URL https://aclanthology.org/2021.tacl-1.21.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Retrieval augmented language model pre-training. In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pp. 3929–3938. PMLR, 2020. URL http://proceedings.mlr.press/v119/guu20a.html.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. *Proceedings of the International Conference on Learning Representations (ICLR), 2021a*. 

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In *Joaquin Vanschoren and Sai-Kit Yeung (eds.)*, *Proceedings of the NeurIPS Datasets and Benchmarks 2021, December 2021*, virtual, 2021b. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ec0db773eb2d1c0ba178361a4-Abstract-round2.html.

Karl Moritz Hermann, Tomáš Kociský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In *Corinna Cortes, Neil D. Lawrence, Daniel D. Lee, Masashi Sugiyama, and Roman Garnett (eds.)*, *Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pp. 1693–1701, 2015*. URL https://proceedings.neurips.cc/paper/2015/hash/afec7005c9114302c049714f0369e5-Abstract.html.

Johannes Hoffart, Mohamed Amir Yosef, Ilaria Bordino, Hagen Fürstenau, Manfred Pinkal, Marc Spaniol, Bilyana Tzaneva, Stefan Thater, and Gerhard Weikum. Robust disambiguation of named entities in text. In *Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing*, pp. 782–792, Edinburgh, Scotland, UK, July 2011. Association for Computational Linguistics. URL https://aclanthology.org/D11-1072.
```

### --- Page 0013 ---

```markdown
# Published as a conference paper at ICLR 2024

Sebastian Hofstätter, Jiecao Chen, Karthik Raman, and Hamed Zamani. Multi-task retrieval-augmented text generation with relevance sampling, 2022.

Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Daniel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Kourza, Xian Li, Brian O'Koro, Gabriel Pereyra, Jeff Wang, Christopher Dewan, Asli Celikyilmaz, Luke Zettlemoyer, and Ves Stoyanov. OPT-IML: scaling language model instruction meta learning through the lens of generalization. CoRR, abs/2212.12017, 2022. doi: 10.48550/arXiv.2212.12017. URL https://doi.org/10.48550/arXiv.2212.12017.

Gautier Izard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 874–880, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021-eacl-main.74. URL https://aclanthology.org/2021.eacl-main.74.

Gautier Izard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=JkNplX7b0.

Kelvin Ling, Dekun Wu, and Hui Jiang. FreebaseQA: A new factoid QA data set matching trivia-style question-answer pairs with Freebase. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 318–323, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1028. URL https://aclanthology.org/N19-1028.

Qiao Jin, Bhuwan Dhingra, Zhengqing Liu, William Cohen, and Xinghua Lu. PubMedQA: A dataset for biomedical research question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2567–2577, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1259. URL https://aclanthology.org/D19-1259.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL https://aclanthology.org/P17-1147.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Frolov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550.

Omar Khatab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 39–48. Association for Computing Machinery, 2020. ISBN 9781450380164. doi: 10.1145/3397271.3401075. URL https://doi.org/10.1145/3397271.3401075.

Tomas Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. The NarrativeQA reading comprehension challenge. Transactions of the Association for Computational Linguistics, 6:317–328, 2018. doi: 10.1162/tacl.a.00023. URL https://aclanthology.org/Q18-1023.
```

### --- Page 0014 ---

```markdown
Published as a conference paper at ICLR 2024

Andreas Kofler, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tan, Keith Stevens, Abdallah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richard Nagyfi, et al. Openassistant conversations—democratizing large language model alignment. arXiv preprint arXiv:2304.07327, 2023. 

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Albert, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelleher, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl.a.00276. URL https://aclanthology.org/Q19-1026.

Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension. In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), pp. 333–342, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/K17-1034. URL https://aclanthology.org/K17-1034.

Patrick S. H. Lewis, Ethan Perez, Aleksandr Piktus, Fabio Petroni, Vladimir Karpukhin, Nam Goyal, Heinrich Küttler, Mike Lewis, Wen-tai Yin, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Husan Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/6b492302305f78e0b1e26945f4781e5-Abstract.html.

Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Saško, Gunjan Chhablani, Bhavvitya Malik, Simon Brandies, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Paytor, Angelina McMillan-Major, Philipp Schmidt, Sylvain Guerguiev, Clément Delangue, Théo Matussière, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar, François Largouet, Alexander Rush, and Thomas Wolf. Datasets: A community library for natural language processing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 175–184, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-demo.21. URL https://aclanthology.org/2021-emnlp-demo.21.

Minghan Li, Sheng-Chieh Lin, Barlas Oguz, Aish Gohsal, Jimmy Lin, Yashar Mehdad, Wen-tao Yih, and Xiul Chen. CTADLE: Conditional token interaction via dynamic lexical routing for efficient and effective multi-vector retrieval. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1189–1197, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.663. URL https://aclanthology.org/2023.acl-long.663.

Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xiul Chen. How to train your DRAGON: Diverse augmentation towards generalizable dense retrieval. arXiv preprint arXiv:2302.07452, 2023.

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 158–167, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1015. URL https://aclanthology.org/P17-1015.

Jerry Liu. LlamaIndex, 11 2022. URL https://github.com/jerryliua/llama_index.

Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James R. Glass. SAIL: search-augmented instruction learning. CoRR, abs/2305.15225, 2023. doi: 10.48550/arXiv.2305.15225. URL https://doi.org/10.48550/arxiv.2305.15225.
```

### --- Page 0015 ---

```markdown
# Published as a conference paper at ICLR 2024

Alex Malin, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannah Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 9802–9822, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.546. URL https://aclanthology.org/2023.acl-long.546.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pp. 2381–2391, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1260. URL https://aclanthology.org/D18-1260.

Katharine Miller. How Do We Fix and Update Large Language Models?, 2023. URL https://hai.stanford.edu/news/how-do-we-fix-and-update-large-language-models.

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. Ms marco: A human generated machine reading comprehension dataset. CoRR, abs/1611.09268, 2016. URL http://dblp.uni-trier.de/db/journals/corr/corr1611.html#NguyenRSGTMD16.

OpenAI. Gpt-4 technical report, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kell, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Sondhi. Training language models to follow instructions with human feedback. In *Advances in Neural Information Processing Systems*, volume 35, pp. 2730–2744. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efe3b58e364a7391458805a001731-Paper-Conference.pdf.

Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, Y James Thorne, Yacine Serignat, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. KILT: a benchmark for knowledge intensive language tasks. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp. 2523–2534, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.200. URL https://aclanthology.org/2021.naacl-main.200.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *J. Mach. Learn. Res.*, 21:140:1–140:67, 2020. URL http://jmlr.org/papers/v21/20-074.html.

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for SQuAD. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pp. 784–789, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-2124. URL https://aclanthology.org/P18-2124.

Ori Ram, Yoav Levine, Itay Dalmiaogdos, Dor Muthalay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models, 2023.

Siva Reddy, Danqi Chen, and Christopher D Manning. Coqa: A conversational question answering challenge. *Transactions of the Association for Computational Linguistics*, 7:249–266, 2019.

Stephen E. Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4):333–389, 2009. doi: 10.1561/1500000019. URL https://doi.org/10.1561/1500000019.
```

### --- Page 0016 ---

```markdown
| **Authors**                                                                 | **Title**                                                                                          |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Anna Rogers, Olga Kovaleva, Matthew Downey, and Anna Rumshisky.           | Getting closer to AI complete question answering: A set of prerequisite real tasks. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirtieth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY USA, February 7-12, 2020, pp. 8722-8731. AAAI Press, 2020. URL: [https://aaai.org/ojs/index.php/AAAI/article/view/6398](https://aaai.org/ojs/index.php/AAAI/article/view/6398). |
| Keisuke Sakaguchi, Roman Le Bras, Chandra Bhagavatula, and Yejin Choi.   | Winogrande: An adversarial winograd schema challenge at scale. arXiv preprint arXiv:1907.10641, 2019. |
| Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi.   | Social IQa: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4463-4473, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1454. URL: [https://aclanthology.org/D19-1454](https://aclanthology.org/D19-1454). |
| Freda Shi, Xinyun Chen, Kanishka Sharma, Nathan Sacles, David Dohan, Ed H. Chi, Nathaniel Schiari, and Denny Zhou. | Large language models can be easily distracted by irrelevant context. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Siwan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 4th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 3120–3127. PMLR, 23–29 Jul 2023a. URL: [https://proceedings.mlr.press/v202/shi23a.html](https://proceedings.mlr.press/v202/shi23a.html). |
| Weijia Shi, Yewon Kim, Michihiro Yasunaga, Mijoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. | REPLUG: retrieval-augmented black-box language models. CoRR, abs/2301.12625, 2023. doi: 10.48550/arXiv.2301.12625. URL: [https://arxiv.org/abs/2301.12625](https://arxiv.org/abs/2301.12625). |
| Kai Sun, Ruin Ehtan Xu, Hanwen Zhao, Yue Liu, and Xin Luna Dong.           | Head-to-tail: How knowledgeable are large language models (llm)? a.k.a. will llms replace knowledge graphs?, 2023. |
| Oyvind Tafjord, Peter Clark, Matt Gardner, Wen-tau Yih, and Ashish Sabharwal. | QUAERE: A dataset and models for answering questions about qualitative relationships. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pp. 7063-7071. AAAI Press, 2019. doi: 10.1609/aaai.v33i01.3301763. URL: [https://doi.org/10.1609/aaai.v33i01.3301763](https://doi.org/10.1609/aaai.v33i01.3301763). |
| Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant.        | Commonsense: A question answering challenge targeting commonsense knowledge. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pp. 4149-4158. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1421. URL: [https://doi.org/10.18653/v1/N19-1421](https://doi.org/10.18653/v1/N19-1421). |
| Rohan Tatori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. | Stanford alpaca: An instruction-following llama model. URL: [https://github.com/tatsu-lab/stanford_alpaca, 2023](https://github.com/tatsu-lab/stanford_alpaca, 2023). |
| James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. | FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 809-819, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1074. URL: [https://aclanthology.org/N18-1074](https://aclanthology.org/N18-1074). |
| Kushal Tiwalra, Aram H. Markosyan, Luke Zettlemoyer, and Armen Aghajanyan. | Memorization without overfitting: Analyzing the training dynamics of large language models. In |
```

### --- Page 0017 ---

```markdown
Published as a conference paper at ICLR 2024

NeurIPS, 2022. URL: http://papers.nips.cc/paper_files/2022/hash/ fa050f4d6807e2b465715b2d249-Abstract-Conference.html.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Éric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Édouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babad, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargava, Shruti Bhosale, Dan Bikle, Lukas Blecher, Cristian Cantor Ferrer, Moya Chen, Guillem Cucurull, David Esibou, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Guo, Vedanju Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Maidin Khabas, Isabel Klonunov, Artem Korenov, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenny Lee, Diana Liskovich, Yinghui Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybo, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalpana Salathiel, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Tian, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Ilany Zaron, Yuchin Zhang, Angela Fan, Melania Scalambro, Sharan Narang, Aurélien Rodriguez, Robert Stojanovic, Sergey Edunov, and Nikos Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023b.

Adam Tsichling, Tong Wang, Xinqi Yuan, Justin Harris, Alessandro Sordoni, Philip Bachman, and Karcher Sulman. NewsQ: A machine comprehension dataset. In Proceedings of the 2nd Workshop on Representation Learning for NLP, pp. 191–200, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-2623. URL: https://aclanthology.org/w17-2623.

Yizhong Wang, Swaroop Mishra, Pegah Alimoradiabshar, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karakosmanolakis, Haizhi Lai, Ishan Purohit, Hison Mandal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrdad Moradshahi, Mihir Parmar, Milad Pirooz, Neeraj Varshney, Phani Rohitha Kaza, Pukit Verma, Ravesh Singh, Puri Rushing Karia, Savan Doshi, Shailaja Keyar Sampat, Siddharth Mishra, Sujan Reddy A, Sumanta Padhy, Tanay Dixit, and Xudong Shen. Super-Instructioners: Generalizing via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5085–5109, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.340. URL: https://aclanthology.org/2022.emnlp-main.340.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guo, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Trained language models are zero-shot learners. In International Conference on Learning Representations, 2022. URL: https://openreview.net/ forum?id=gEZrGozcdR.

Lee Xiong, Chenyang Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junadi Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense retrieval. In International Conference on Learning Representations, 2021. URL: https://openreview.net/forum?id=ZrEfyZlIn.

Yi Yang, Wen-tau Yih, and Christopher Meek. WikiQA: A challenge dataset for open-domain question answering. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 2013–2018, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1237. URL: https://aclanthology.org/D15-1237.

Zhiling Yang, Peng Qi, Saizheng Zhang, Yousha Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2024.
```

### --- Page 0018 ---

```markdown
Published as a conference paper at ICLR 2024

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1472. URL https://aclanthology.org/ P19-1472.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. Lima: Less is more for alignment, 2023.

Processing, pp. 2369–2380, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1259. URL https://aclanthology.org/ D18-1259.
```

### --- Page 0019 ---

```markdown
# A RETRIEVAL CORPUS

We combine the text chunks from the Dec. 20, 2021 Wikipedia dump released by Izacard et al. (2022b) with additional ones from the 2017-2020 CommonCrawl dumps. The Wikipedia dump includes lists and infoboxes in addition to regular articles. The articles are split by section, where long sections are further split into text chunks of equal sizes and contain less than 200 words, leading to a total of 37M text chunks. We randomly sample a subset of articles from the CommonCrawl dumps, and split them into equal-sized text chunks that contain less than 100 white-space-separated words, leading to a total of 362M text chunks.

We use a GPU-based exact $k$-nearest-neighbor search index implementation\footnote{released by Izacard et al. (2022b).}

# B IMPLEMENTATION DETAILS

## Fine-tuning Dataset Selection

Prior work (Chung et al., 2022b; Iyer et al., 2022) have demonstrated that jointly fine-tuning the language model on a diverse collection of instruction-based datasets leads to improved model generalization for unseen instructions. We adopt a similar strategy by combining five categories of fine-tuning tasks to enhance the language model’s knowledge utilization (dialogue, open-domain QA, chain-of-thought reasoning) and to improve its contextual awareness for prediction generation (reading comprehension, summarization). These categories were selected due to their representativeness of practical knowledge-intensive language tasks.

## Retrieval-augmented LM Fine-tuning

We use the top-3 retrieved text chunks for a given example (i.e., $k = 3$) to generate the fine-tuning instances. To improve fine-tuning efficiency, we pack multiple examples up to the language model context window limit (2048 tokens). Each example is demarcated by a pair of `<bos>` and `<eos>` tokens, and we adopt the technique from (Iyer et al., 2022) such that a token only attends to the previous tokens in the same example. We use a dataset mixture that contains 10% unsupervised text and 5% OASST-1 data. For the remaining datasets, we establish a cap on the number of examples per dataset at $\eta = 7500$ based on the model performance on our development set.\footnote{We then randomly sample batches in accordance with this adjusted mixture probability.}

We fine-tune the 7B, 13B and 65B LLaMA models using 8, 16 and 64 A100 GPUs, respectively. The fine-tuning hyperparameters are detailed in Table 8. Similar to Zhou et al. (2023), we found that the best generalization performance on the dev set can be achieved using a small number of fine-tuning steps. We evaluate the models every 100 steps, and select the best checkpoint based on the average dev set performance over the 6 development KILT tasks shown in Table 11 (early stopping).

### Table 8: Hyperparameters for retrieval-augmented LM fine-tuning.

| Model      | peak lr | scheduler | warm-up | # steps | early stopping | batch size | model parallel | seq len |
|------------|---------|-----------|---------|---------|----------------|------------|----------------|---------|
| RA-DIT 7B  | 1e-5   | 1e-7     | cosine  | 200     | 500            | 64         | 1              | 2048    |
| RA-DIT 13B | 1e-5   | 1e-7     | cosine  | 200     | 500            | 400       | 128            | 2,048   |
| RA-DIT 65B | 1e-5   | 1e-7     | cosine  | 200     | 500            | 128        | 8              | 2048    |

## 64-shot Task Fine-tuning

Table 9 summarizes our hyperparameters for 64-shot fine-tuning on the 9 KILT eval tasks shown in Table 12 except for MMLU. Given the small amount of examples used (64 × 9 = 576), the fine-tuning for a significantly less number of steps at this stage without using warm-up. We evaluate the model every 50 steps, and select the best checkpoint based on the average dev set performance over the 6 development KILT tasks shown in Table 11.

## Retriever Fine-tuning

We employ both unsupervised text and downstream tasks for retriever fine-tuning. For the corpus data, we randomly sample 900k text chunks from our retrieval corpus to 

\footnote{https://github.com/facebookresearch/atlas}
\footnote{We did not thoroughly tune this parameter to avoid overfitting to the development sets.}
```


### --- Page 0020 ---

```markdown
| Model         | peak lr | end lr | scheduler | warm-up | # steps | early stopping | batch size | model parallel | seq len |
|---------------|---------|--------|-----------|---------|---------|----------------|------------|----------------|---------|
| LLAMA 65B     | 1e-5   | 1e-6  | linear    | 0       | 100     | 100            | 8          | 8              | 2048    |
| RA-DIT 13B    | 1e-5   | 1e-6  | linear    | 0       | 100     | 50             | 32         | 2              | 2048    |
| RA-DIT 65B    | 1e-5   | 1e-6  | linear    | 0       | 100     | 50             | 32         | 8              | 2048    |

We form a set of self-supervised data, using the first 50 tokens of each chunk as the input $x$ and the last 50 tokens as the ground-truth output $y$. In addition, we leverage the multi-task instruction tuning datasets (MTI data) as shown in Table 1, including 10 open-domain question answering and dialog tasks, with a total of 286k training examples. As discussed in §5.1, we observe that, when used alone, the corpus data works slightly better than the downstream tasks. However, combining both types of fine-tuning data yields the best results and outperforms using either source alone. Therefore, we adopt a mixture of 95% corpus data and 5% downstream tasks for retriever fine-tuning in our final model.

We fine-tune the DRAGON+ retriever on 16 A100 GPUs using the drp-scale codebase. The retriever is fine-tuned using a learning rate of $1e^{-5}$ with 1237 warmup steps (DRAGON default), a per-GPU batch size of 32, and a temperature $\tau = 0.01$, for a single epoch over a combination of 5% MTI data and 95% corpus data. We adopt the KL-divergence loss as discussed in Section 2.4 using the top-10 retrieved chunks for each example. For simplicity and efficiency, we produce the top-10 retrieved chunks and their LSR scores (Eqn. 4) using LLAMA 65B and DRAGON+, and do not update them during R-ft. Furthermore, as only the query encoder is fine-tuned, there is no need to update the chunk embeddings in the retriever index. Model validation is performed once every 500 steps using the same mean reciprocal rank (MRR) metric as in the original DRAGON paper (Lin et al., 2023), on a combined validation set from the 10-task MTI data.

Inference under further specification, we use the top-10 retrieved text chunks for a given sample (i.e. $k = 10$) and ensemble their predictions during inference. For multi-choice tasks, we compute the weighted average probability of each choice item according to Eq. 2 and select the choice with the highest probability. For generation tasks, we perform decoding using each generated prompt independently, compute the weighted average probability of each using generated answer, and output the answer with the highest probability. When computing probabilities of output answers, we use several scoring functions: “nll”, “nll_char”, “nll_token”, and “nll_comp”. “nll” is the sum of negative log likelihood across all tokens in the sequence. “nll_char” and “nll_token” are “nll” divided by the numbers of characters and subword units in output answers respectively. “nll_comp” selects answers based on the probability divided by the probability of the answer given “Answer:”.

### C Fine-tuning Dataset Templates

Table 10 shows the templates we used to serialize our instruction tuning datasets. Following Chung et al. (2022b) and Iyer et al. (2022), we randomize the field markers used during training to avoid overwriting. In particular, when serializing a task example, we randomly sample from `{"Q: ", "Question: ", "Q: "}` for `inst_set`, set `inst_eb` to `"\n"` and randomly sample from `{"A: ", "Answer: "}` for `<answer>`.

### D Evaluation Datasets and Templates

Table 11 shows the evaluation datasets used in our experiments. For dev set evaluation, we use a maximum of 2500 randomly sampled examples from the respective official dev sets to reduce the sample for experimentation as it performs reasonably well and allows us to execute inference with fewer GPUs.
```

### --- Page 0021 ---

```markdown
| Category                | Instruction Template                                                                 | Query Template                          |
|------------------------|-------------------------------------------------------------------------------------|-----------------------------------------|
| Dialogue                | Background: {retrieved passage} \n\n Q: {turn} A: {turn} {turn} {turn} A: {turns} A: | - <inst_s> <answer_s> {answer} -       |
| Open-domain QA          | Background: {retrieved passage} \n\n Q: {turn} A: {turn} {turn} {turn} A: {turns} A: | - <inst_s> <answer_s> {question} -     |
| Reading Comprehension   | Background: {context} \n\n Q: {inst_s} {question} <inst_s> {question}             | - <answer_s> {answer} -                |
| Summarization           | Background: {context} \n\n Summarize this article: <inst_s>                       | - <answer_s> {summary} -               |
| Chain-of-thought Reasoning | Background: {retrieved passage} \n\n Q: {inst_s} {instructions}                  | - <answer_s> {answer} -                |

Table 11: Our evaluation datasets. * indicates the development datasets we used to select fine-tuning hyperparameters.

| Task                  | Dataset name                          | Acronym | Metric | Score   |
|----------------------|---------------------------------------|---------|--------|---------|
| Open-domain QA       | Natural Questions (Kwiatkowski et al., 2017) | NQ      | EM     | nll     |
| QA                   | TriviaQA (Joshi et al., 2017)       | TQA     | EM     | nll     |
| QA                   | HotpotQA (Yang et al., 2018)        | HoPo    | EM     | nll     |
| Fact Checking        | FEVER (Thorne et al., 2018)         | FEV     | Acc.   | nll     |
| Entity Linking       | AIDA CoNLL-YAGO (Hoffart et al., 2011) | AIDA    | Acc.   | nll     |
| Dialogue             | Wizard of Wikipedia (Dinan et al., 2019) | WoW     | FT     | nll     |
| Reasoning            | BoolQ (Clark et al., 2019)          | BoolQ   | Acc.   | nll     |
| Commonsense Reasoning | PIQA (Bisk et al., 2020)            | PIQA    | Acc.   | nll.char|
|                      | SQuAD (Rajpurkar et al., 2016)      | SQA     | Acc.   | nll.char|
|                      | HellaSwag (Zellers et al., 2019)    | HellaSwag | Acc. | nll.char|
|                      | WinoGrande (Sakaguchi et al., 2019) | WinoGrande | Acc. | nll.char|
|                      | ARC-E (Clark et al., 2018)          | ARC-E   | Acc.   | nll.char|
|                      | ARC-Challenge (Clark et al., 2018)  | ARC-C   | Acc.   | nll.char|
|                      | OpenBookQA (Mihaylov et al., 2018)  | OBQA    | Acc.   | nll.comp|

E ADDITIONAL EXPERIMENTS

E.1 SCALING LAWS OF RETRIEVAL AUGMENTED LANGUAGE MODEL FINE-TUNING

We investigate the impact of the base language model size when retrieval-augmented instruction tuning is applied, and summarize the results in Figure 2. We combine the fine-tuned models with the base DRAGON* retriever in this set of experiments.

Overall, all models substantially benefit from retrieval augmentation, with smaller models witnessing even bigger improvements. We further note that retrieval augmentation can be an effective strategy for enhancing the performance of smaller models (hence reducing pre-training and inference costs), given the 7B model leveraging > 1 teraflop which outperformed the vanilla 65B model on several tasks. This trend also differs across tasks. For tasks that primarily...
```

### --- Page 0022 ---

```markdown
| Table 12: Language model prompts and retriever query templates used for our evaluation datasets. We did not perform retrieval for commonsense reasoning tasks evaluation. |
| --- |
| **Task** | **LLM Prompt Template** |
| **Knowledge-Intensive Tasks** |  |
| MMLU | Background: `{retrieved passage}\n\nQuestion: {question}\nA. {choice}\nB. {choice}\nC. {choice}\nD. {choice}\nA: {answer}` |
| HotpotQA | Background: `{retrieved passage}\n\nQuestion: {question}\nA: {answer}` |
| AIDA | Background: `{“retrieved” passage}\n\n{context}\nOutput: “the Wikipedia page title of the entity mentioned between [START_ENT] and [END_ENT] in the given text}\nA: {answer}` |
| FEVER | Background: `{“retrieved” passage}\n\n{this statement true? “{statement}”}\nA: {answer}` |
| T-REX | Background: `{retrieved passage}\n\n{entity} | {SEP} | {relation}\nA: {answer}` |
| WoW | Background: `{retrieved passage}\n\nQ: {turn}\nA: {answer}` |
| **Commonsense Reasoning Tasks** |  |
| ARC-E.C. | Question: {question}\nAnswer: {answer} |
| BoolQ | Question: {question}\nAnswer: {answer} |
| HellaSwag | {context} | {ending} |
| OpenBookQA | Question: {question}\nAnswer: {answer} |
| PIQA | Question: {question}\nAnswer: {answer} |
| SIQA | {context} | Q: {question} A: {answer} |
| WinoGrande | {prefix} {answer} {suffix} |

![Model size performance across different tasks](assets/page_0022_img_1.png)

Figure 2: RA-IT model performance (combined with DRAGON+) across sizes 7B, 13B and 65B on our development tasks. 0-shot performance: dashed lines; 5-shot performance: solid lines. 

measure one-hop look-up abilities (such as Zero-Shot RE and T-REX), retrieval augmentation provides significant improvements across all model sizes and can bring the performance of smaller models closer to that of their larger counterparts. For more complex tasks (such as HotpotQA and WoW), the advantage of using a larger LLM remains prominent.
```

### --- Page 0023 ---

```markdown
| **Table 13**: Comparison between parallel retrieval-augmentation and chunk concatenation. The results are obtained using the base DRAGON* retriever. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0-shot** | **HoPo** | **FEV** | **AIDA** | **zRE** | **T-REx** | **WoW** | **Avg** |
| top-3 chunks | RA-IT 65B (parallel) | 39.9 | 82.4 | 45.2 | 63.4 | 52.8 | 16.6 | 50.1 |
|  | RA-IT 65B (concat) | 39.5 | 83.9 | 52.2 | 65.2 | 47.9 | 16.6 | 50.9 |

## E.2 COMPARE PARALLEL RETRIEVAL AUGMENTATION TO CHUNK CONCATENATION

We adopt the parallel retrieval-augmentation approach proposed by Shi et al. (2023b) to reduce the prompt length, which is necessary in the few-shot settings (§2.1). However, this approach is computationally expensive when the individual prompts share long common prefixes (as in the few-shot setting). In addition, by separately encoding the text chunks, this approach is potentially less effective for knowledge synthesis compared to concatenating the retrieved text chunks in a single prompt. To understand the impact of using parallel retrieval-augmentation, we compare it to the chunk concatenation approach under the setting with top-3 retrieved text chunks. We conduct this experiment using the RA-IT 65B model and 0-shot evaluation.

According to Table 13, the two approaches perform closely on average with chunk concatenation demonstrating a small benefit. Specifically, parallel retrieval-augmentation under-performs chunk concatenation on FEVER and Zero-shot Relation Extraction, and perform on par on Wizard of Wikipedia. It also performs slightly better on HotpotQA, which is somewhat unexpected, given the dataset is specifically designed to necessitate multiple evidence sources for answering a question. We observe wider performance gaps between the two approaches on CoNLL-YAGO and T-REx, where concatenation performs much better on the former but worse on the latter.

It is worth noting that the RA-IT 65B model has been fine-tuned using parallel retrieval augmentation, which potentially provides a benefit to using the same configuration during inference. We defer the investigation of fine-tuning with chunk concatenation to future studies. This direction appears promising, especially considering that state-of-the-art language models are progressively being trained with ever-larger context windows. 

## E.3 RETRIEVAL CORPUS ABLATION

| **Table 14**: Retriever settings: We report 5-shot dev set performance using LLAMA 65B and various retrievers in the REPLUG setting. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **5-shot** | **MLLM** | **NQ** | **TQA** | **HoPo** | **FEV** | **AIDA** | **zRE** | **T-REx** | **WoW** | **ELI5** | **Avg** |
| LLAMA 65B | 61.3 | 30.9 | 70.6 | 23.8 | 83.7 | 50.2 | 36.0 | 52.3 | 17.4 | 23.4 | 45.0 |

Retriever corpus ablation using LLAMA 65B and the DRAGON* retriever

CC only | 62.8 | 39.6 | 72.6 | 34.4 | 89.5 | 54.8 | 30.3 | 46.2 | 17.1 | 29.7 | 47.0 |
Wiki 2021 + infobox | 62.2 | 42.0 | 71.4 | 89.8 | 62.5 | 65.3 | 17.1 | 17.7 | 22.2 | 54.8 |
Wiki 2021 | 62.2 | 41.8 | 71.0 | 87.9 | 62.1 | 65.2 | 73.6 | 17.6 | 22.2 | 54.7 |
Wiki 2018 | 61.5 | 42.6 | 70.7 | 40.4 | 90.8 | 62.1 | 51.3 | 59.8 | 17.6 | 22.5 | 51.9 |

Table 14 shows the impact of varying the retrieval corpus. In particular, we consider several subsets of our 399M retrieval corpus, namely CommonCrawl only (362M) and Wikipedia only (with and without infoboxes). We further compare with another Wikipedia snapshot (Wiki 2018) commonly used in the literature (Karpukhin et al., 2020). We observe that retrieving from Wikipedia only is beneficial for a number of KILT tasks such as AIDA and zRE, as Wikipedia was the intended corpus for KILT tasks. We find that Wiki 2018 works better for NQ since the corpus is closer to the date of its data collection, similar to the observations by Izacard et al. (2022b). This indicates that our retrieval-augmented LM is faithful to the supplied retrieval corpus, and up-to-date information can be provided by updating the retrieval index at test time.
```

### --- Page 0024 ---

```markdown
# F  EXAMPLES

In this section, we show the task prompts, the corresponding retrieved passages and model predictions generated by LLAMA 65B instruction-tuned with retrieval augmentation (RA-IT 65B) and LLAMA 65B instruction-tuned conventionally (IT 65B) on selected task examples.

## F.1  HOTPOTQA

We analyze the performance of the two models on the development set of HotpotQA in the zero-shot setting since under this setting RA-IT 65B outperforms IT 65B by a large margin. Table 15 shows two examples from the HotpotQA development set where RA-IT 65B makes a correct prediction while IT 65B makes a wrong prediction. First, we observed that the dense retriever struggles to return useful text chunks for the multi-hop questions in the HotpotQA dataset and most of the returned text chunks contains no information that helps the prediction. In this case, the IT 65B model shows a stronger tendency to be misled by distractors within the retrieved text chunk, since it has not been trained with noisy passages during fine-tuning. It also tend to predict “I don’t know” more frequently18, while the RA-IT 65B can ignore the noisy passages retrieved and predict the correct answer based on its parametric knowledge (Mallene et al., 2023). We also observe that in cases where both models generate wrong predictions because of the distractors (e.g. for the third text chunk in the second example), the generation probability of the wrong answer from RA-IT 65B is much lower; and in cases where both models ignore the noisy passages and rely on the parametric knowledge to make a prediction, RA-IT 65B outputs the correct answer with a higher probability (e.g. for the second text chunk in the first example).

18 As discussed in §2.2, this behavior is induced by fine-tuning on SQuAD v2.0 (Rajpurkar et al., 2018), which trains the model to predict “I don’t know” for passages that does not match with the given question.
```

### --- Page 0025 ---

```markdown
| Prompt                                                                                                                                                                                                 | $P_R$ | Output | $n_{ll,M}$ | RA-IT | RA-IT  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|--------|------------|-------|--------|
| Input: Charlotte Hatherley initially came to prominence in a band formed in what year? Label: 1992. RA-IT 65B final prediction: 1992 $\times$                                                                 | 0.27  | 1992   | 1.16       | 1.01  |        |
| Background: Charlotte Hatherley Born in London, Hatherley was brought up in West London (1992) and attended Chiswick Community School. Her music career began at the age of 15, when she joined British punk band Nightnurse. Two years later, with Ash looking for a guitarist to add to their indie sound, Hatherley was hired after frontman Tim Wheeler saw her play at a Nightnurse gig. Hatherley’s Ash debut was at the Beaford Festival in Limelight in 00 August 1997, and the following week the newly formed band played the 1997 V Festival in front of 50,000 people. Her recording career with Ash began last year with the single “A Life Less Ordinary” and continued on with Nu-Clear Sounds in 1998. Hatherley was a full-time member of Ash for eight years, playing on three studio albums, and wrote a handful of the band’s songs, most notably Grey Area, on the B-side to the single “There’s a Star.” The song was a cult favorite among fans, and eventually became the title track of Hatherley’s debut solo album. On 20 January 2006 it was announced that Hatherley would be leaving Ash in an amicable break-up.\n\nQ: Charlotte Hatherley initially came to prominence in a band formed in what year? $\n$ | 0.46  | 1992   | 0.94       | 0.38  |        |
| Background: Charlotte Hatherley Charlotte Franklin Hatherley (born 20 June 1979) is an ex-member of Ash. She has pursued a solo career and acted as a touring instrumentalist for Bryan Ferry, KT Tunstall, and others. She has also been a touring member of The Long Paddock. Since leaving Ash in 2006, she has been a co-writer for Alex Lines and is currently musical director for South African artist Thandiswa.\n\nQ: Charlotte Hatherley initially came to prominence in a band formed in what year? $\n$ | 0.13  | 1992   | 1.00       | 0.54  | 0.72   |
| Input: Oxley Highway ends at a coastal town that had how many inhabitants in June 2016? Label: 45,698. RA-IT 65B final prediction: 1.00 $\times$                                                                 | 0.25  | 10000  | 7.27       | 0.61  |        |
| Background: Oxley Electorate: Ipswich Motorway: The Ipswich Motorway is a vital link up-trail. Oxley Electorate: Ipswich Motorway The Ipswich Motorway is a vital link supporting Queensland industry. It forms part of the national freight network providing connectivity for industry to the Acacia Ridge intermodal facility, the major industrial area of Wacol and the Brisbane markets at Rocklea2014 in the member for Morrison’s electorate. The motorway is the state’s largest fruit and vegetable market and is a major centre for produce on the east coast. The section of the motorway is over capacity with 30,000 vehicles per average day, including up to 12,000 freight vehicles. Numbers are increasing each year as an average of fun.\n\nQ: Oxley Highway ends at a coastal town that had how many inhabitants in June 2016? $\n$ | 0.15  | 45,698 | 45,698     | 0.18  | 0.38   |
| Background: Post Offices For Sale NSW — Lotto — Newseagents — Marlow & Co. Co South Wales about 390 km north of Sydney, and 570 km south of Brisbane. The town is located on the Tasman Sea coast, at the mouth of the Hastings River, and at the eastern end of the Oxley Highway. The town with its suburbs had a population of 45,698 in June 2016. Port Macquarie is a retirement destination, known for its extensive beaches and waterways. Port Macquarie has a humid sub-tropical climate with warm, humid summers and mild winters, with frequent rainfall spread throughout the year. Port Macquarie’s central business district contains two shopping centres, a marina, the beginning of the Oxley Highway and a coastal town that had many inhabitants in June 2016.\n$\n$ | 0.15  | 45,698 | 45,698     | 0.18  | 0.38   |
```


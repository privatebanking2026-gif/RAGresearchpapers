# ArXiv 2310.11511

### --- Page 0001 ---

```markdown
# SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

**Akari Asai\*, Zeqiu Wu\*, Yizhong Wang\dagger, Avrup Siti\ddagger, Hannaneh Hajishirzi\dagger**  
\textit{University of Washington} \textit{Allen Institute for AI} \textit{IBM Research AI}  
\{akari, zeqiuwu, yizhongw, hannaneh\}@cs.washington.edu, avi@us.ibm.com

---

## ABSTRACT

Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. Retrieval-Augmented Generation (RAG), an ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation. We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (SELF-RAG) that enhances an LM's quality and factuality through retrieval and self-reflection. Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens. Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. Experiments show that SELF-RAG (7B and 13B parameters) significantly outperforms state-of-the-art retrieval and retrieval-augmented models on a diverse set of tasks. Specifically, SELF-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA, reasoning and fact verification tasks, and it shows significant gains in improving factuality and citation accuracy for long-form generations relative to these models.

---

## 1 INTRODUCTION

State-of-the-art LLMs continue to struggle with factual errors (Mallen et al., 2023; Min et al., 2023) despite their increased model and data scale (Ouyang et al., 2022). Retrieval-Augmented Generation (RAG) methods (Figure 1 left; Lewis et al., 2020; Guo et al., 2020) augment the input of LLMs with relevant retrieved passages, reducing factual errors in knowledge-intensive tasks (Ram et al., 2022; Asai et al., 2023a). However, these methods may hinder the versatility of LLMs or introduce unnecessary or off-topic passages that lead to low-quality generations (Shi et al., 2023) since they retrieve passages indiscriminately regardless of whether the factual grounding is helpful. Moreover, the output is not guaranteed to be consistent with retrieved relevant passages (Gao et al., 2023) since the models are not explicitly trained to leverage and follow facts from provided passages. This work introduces Self-Reflective Retrieval-augmented Generation (SELF-RAG) to improve an LLM's generation quality, including its factual accuracy without hurting its versatility, via on-demand retrieval and self-reflection. We train an arbitrary LM in an end-to-end manner to learn to reflect on its own generation process given a task input by generating both task output and intermittent special tokens (i.e., reflection tokens). Reflection tokens are categorized into retrieval and critique tokens to indicate the need for retrieval and its generation quality respectively (Figure 1 right). In particular, given an input prompt and preceding generations, SELF-RAG first determines if augmenting the continued generation with retrieved passages would be helpful. If so, it outputs a retrieval token that calls a retriever model on demand (Step 1). Subsequently, SELF-RAG concurrently processes multiple retrieved passages, evaluating their relevance and then generating corresponding task outputs (Step 2). It then generates critique tokens to criticize its own output and choose best one (Step 3) in terms of factuality and overall quality. This process differs from conventional RAG (Figure 1 left), which

---

1Our code and trained models are available at [https://selfrag.github.io/](https://selfrag.github.io/).
```

### --- Page 0002 ---

```markdown
# Preprint

![Overview of SELF-RAG. SELF-RAG learns to retrieve, critique, and generate text passages to enhance overall generation quality, factuality, and verifiability.](assets/page_0002_img_1.png)

consistently retrieves a fixed number of documents for generation regardless of the retrieval necessity (e.g., the bottom figure example does not require factual knowledge) and never second visits the generation quality. Moreover, SELF-RAG provides citations for each segment with its self-assessment of the output as supported by the passage, leading to enhanced verifiability.

SELF-RAG trains an arbitrary LM to generate text with reflection tokens by unifying them with next token prediction from the embedded model vocabulary. We train our generator LM on a diverse collection of text interleaved with reflection tokens and retrieved passages. Reflection tokens, inspired by reward models used in reinforcement learning (Ziegler et al., 2019; Guu et al., 2020), are inserted offline into the original corpus by a trained critic model. This eliminates the need to host a critic model during training, reducing overhead. The critic model, in part, is supervised on a dataset of input, output, and corresponding reflection tokens collected by prompting a property LM (i.e., GPT-4; OpenAI 2023). While we draw inspiration from studies that use control tokens to start and guide text generation (Liu et al., 2022; Keskar et al., 2019), our trained LM uses critiquing tokens to assess its own predictions after each generated segment as an integral part of the generation output.

SELF-RAG further enables a customizable decoding algorithm to satisfy hard or soft constraints, which are defined by reflection token predictions. In particular, our inference-time algorithm enables us to (1) flexibly adjust retrieval frequency for different downstream applications and (2) customize models' behaviors to user preferences by leveraging reflection tokens through segment-level beam search using the weighted linear sum of the reflection token probabilities as segment score.

Empirical results on six tasks, including reasoning and long-form generation, demonstrate that SELF-RAG significantly outperforms pre-trained and instruction-tuned LLMs that have more parameters and widely adopted RAG approaches with higher citation accuracy. In particular, SELF-RAG outperforms retrieval-augmented ChatGPT on four tasks, Llama2-chat (Touvron et al., 2023) and Alpaca (Dubois et al., 2023) on all tasks. Our analysis demonstrates the effectiveness of training and inference with reflection tokens for overall performance improvements as well as test-time model customizations (e.g., balancing the trade-off between citation provisions and completeness).

## 2 RELATED WORK

### Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) augments the input space of LMs with retrieved text passages (Guu et al., 2020; Lewis et al., 2020), leading to large improvements in knowledge-intensive tasks after fine-tuning or used with off-the-shelf LMs (Ram et al., 2023). A more recent work (Luo et al., 2023) instruction-tunes an LM with a fixed number
```

### --- Page 0003 ---

```markdown
# Preprint

of retrieved passages prepended to input, or pre-train a retriever and LM jointly, followed by few-shot fine-tuning on task datasets (Zacard et al., 2022b). While prior work often retrieves only once at the beginning, Liang et al. (2023) propose to adaptively retrieve passages for generation on top of a proprietary LM or Schick et al. (2023) train an LM to generate API calls for named entities. Yet, the improved task performance of such approaches often comes at the expense of runtime efficiency (Mallen et al., 2023), robustness to irrelevant context (Shi et al., 2023), and lack of attributions (Liu et al., 2023a; Gao et al., 2023). We introduce a method to train an arbitrary LM to learn to use retrieval on-demand for diverse instruction-following queries and introduce controlled generation guided by reflection tokens to further improve generation quality and attributions.

## Concurrent RAG Work

A few concurrent works² on RAG propose new training or prompting strategies to improve widely-adopted RAG approaches. Lin et al. (2023) fine-tune both the retriever and LM on instruction-tuning datasets in two steps. While we also train our model on diverse instruction-following datasets, SELF-RAG enables retrieval on demand and selection of the best possible model output via fine-grained self-reflection, making it widely applicable and more robust and controllable. Yoran et al. (2023) use a natural language inference model and Xu et al. (2023) use a summarization model to filter out or compress retrieved passages before using them to prompt the LM to generate the output. SELF-RAG processes passages in parallel and allows internal ones through self-reflection, without relying on external models at inference. Moreover, our self-reflection mechanism also evaluates other aspects of the model output quality including factuality. LATS (Zhou et al., 2023) prompt off-the-shelf LMs to search for relevant information for question answering tasks and to generate with tree search, guided by LM-generated value scores. While their value function simply indicates an overall score of each generation, SELF-RAG trains to an arbitrary LM to learn to generate fine-grained self-reflection and customizable inference.

## Training and Generating with Critics

Training with critics involves training LMs with reinforcement learning (e.g., Proximal Policy Optimization or PPO; Schulman et al. 2017) from human feedback (RLHF) has proven effective in aligning LMs with human preferences (Zhou et al., 2022). While our work studies fine-grained RLHF with multiple rewards, through our work also studies the multi-grained RLHF with multiple rewards. Though our work also studies fine-grained RLHF on retrieval and generation, we train our target LM on task examples augmented with reflection tokens from a critic model offline, with a far lower training cost compared to RLHF. In addition, reflection tokens in SELF-RAG enable controllable generation at inference, while RLHF focuses on human preference alignment during training. Other works use general control tokens to guide LM generation (Lu et al., 2022; Korkbat et al., 2023), while SELF-RAG uses reflection tokens to decide the need for retrieval and to self-evaluate generation quality. Xie et al. (2023) propose a self-evaluation-guided decoding framework, but they focus only on reasoning tasks with one evaluation dimension (reasoning path consistency) and without retrieval. Recent work on LLM refinement (Dhuliawala et al., 2023; Madan et al., 2023; Paul et al., 2023) prompts a model to generate task output, natural language feedback and refined task output iteratively, but at the cost of inference efficiency.

## 3 SELF-RAG: LEARNING TO RETRIEVE, GENERATE AND CRITIQUE

We introduce Self-Reflective Retrieval-Augmented Generation (SELF-RAG), shown in Figure 1. SELF-RAG is a framework that enhances the quality and factuality of an LM through retrieval and self-reflection, without sacrificing LM’s original creativity and versatility. Our end-to-end training lets an LM $M$ generate text informed by retrieved passages, if needed, and criticize the output by learning to generate special tokens. These reflection tokens (Table 1) signal the need for retrieval or confirm the output’s relevance, support, or completeness. In contrast, common RAG approaches retrieve passages indiscriminately, without ensuring complete support from cited sources.

### 3.1 Problem Formalization and Overview

Formally, given input $x$, we train $M$ to sequentially generate textual outputs $y$ consisting of multiple segments $y = [y_1, \ldots, y_T]$, where $y_t$ indicates a sequence of tokens for the $t$-th segment.³ Generated tokens in $y$ include text from the original vocabulary as well as the reflection tokens (Table 1).

²All work is arXived within a week of this preprint.  
³In this paper, we treat one sentence as a segment in our experiments, but our framework is applicable to any segment unit (i.e., sub-sentence).
```


### --- Page 0004 ---

```markdown
# Preprint

| Type   | Input                | Output                               | Definitions                                                                                     |
|--------|----------------------|--------------------------------------|-------------------------------------------------------------------------------------------------|
| Retrieve | $x / z, y$         | $\{yes, no, continue\}$             | Decides when to retrieve with $\mathcal{R}$                                                   |
| ISRL   | $x, d$               | $\{relevant, irrelevant\}$          | $d$ provides useful information to solve $x$.                                                 |
| ISup   | $x, d, y$            | $\{fully supported, partially supported, no support\}$ | All of the verification-worthy statement in $y$ is supported by $d$.                           |
| ISke   | $x, y$               | $\{5, 4, 3, 2, 1\}$                 | $y$ is a useful response to $x$.                                                               |

Table 1: Four types of reflection tokens used in SELF-RAG. Each type uses several tokens to represent its output values. The bottom three rows are three types of Critique tokens, and the bold text indicates the most desirable critique tokens. $x$, $y$, $d$ indicate input, output, and a relevant passage, respectively.

## Algorithm 1 SELF-RAG Inference

Require: Generator LM $\mathcal{M}$, Retriever $\mathcal{R}$, Large-scale passage collections $\{d_1, \ldots, d_n\}$  
1. Input: input prompt $z$ and preceding generation $y_{<t}$. Output: next output segment $y_t$  
2. $M$ predicts $y_t$ given $(x, y_{<t})$  
3. if $Retrieve$ == Yes then  
4. \quad Retrieve relevant text passages $D$ using $\mathcal{R}$ given $(x, y_{t-1})$  
5. \quad $M$ predicts $ISRL$ given $x, d$ and $y$ given $x, d, y_t$ for each $d \in D$  
6. \quad $M$ predicts $ISup$ and $ISke$ given $x, y_t$ for each $d \in D$  
7. Rank $y_t$ based on $ISRL$, $ISup$, $ISke$  ⟶ Critique  
8. else if $Retrieve$ == No then  
9. \quad $M$ predicts $ISRL$ given $x$  ⟶ Generate  
10. $M$ predicts $ISke$ given $x, y_t$  ⟶ Critique  

Inference overview. SELF-RAG leverages the inference of $\mathcal{M}$ for every $x$ and preceding generation $y_{<t}$. The model decodes a retrieval token to evaluate the utility of retrieval. If retrieval is not required, the model predicts the next output segment, as it does in a standard LM. If retrieval is needed, the model generates: a critique token to evaluate the retrieved passage’s relevance, the next response segment, and a critique token to evaluate if the information in the response segment is supported by the passage. Finally, a new critique token evaluates the overall utility of the response. To generate each segment, SELF-RAG processes multiple passages in parallel and uses its own generated reflection tokens to enforce soft constraints (Section 3.3) or hard control (Algorithm 1) over the generated task output. For instance, in Figure 1 (right), the retrieved passages $d_1$ is selected at the first time step since it does not provide direct evidence ($ISRL$ Irrelevant) and $d$ output is only partially supported while $d$ are fully supported.

Training overview. SELF-RAG enables an arbitrary LM to generate text with reflection tokens by unifying them as next token predictions from the expanded model vocabulary (i.e., the original vocabulary plus reflection tokens). Specifically, we train the generator model $\mathcal{M}$ on a curated corpus with interleaving passages retrieved by a retriever $\mathcal{R}$ and reflection tokens predicted by a critic model $\mathcal{C}$ (summarized in Appendix Algorithm 2). We train $\mathcal{C}$ to generate reflection tokens for evaluating retrieved passages and the quality of a given task output (Section 3.2.1). Using the critic model, we update the training corpus by inserting reflection tokens into task outputs offline. Subsequently, we train the final generator model ($\mathcal{M}$) using the conventional LM objective (Section 3.2.2) to enable $\mathcal{M}$ to generate reflection tokens by itself without relying on the critic at inference time.

## 3.2 SELF-RAG TRAINING

Here, we describe the supervised data collection and training of two models, the critic $\mathcal{C}$ (Section 3.2.1) and the generator $\mathcal{M}$ (Section 3.2.2).

### 3.2.1 TRAINING THE CRITIC MODEL

Data collection for critic model. Manual annotation of reflection tokens for each segment is expensive (Wu et al., 2023). A state-of-the-art LM like GPT-4 (OpenAI, 2023) can be effectively utilized.  
```


### --- Page 0005 ---

```markdown
# Preprint

Figure 2: SELF-RAG training examples. The left example does not require retrieval while the right one requires retrieval; thus, passages are inserted. More examples are in Appendix Table 4.

used to generate such feedback (Liu et al., 2023b). However, depending on such proprietary LMs can raise API costs and diminish reproducibility (Chen et al., 2023). We create supervised data by prompting GPT-4 to generate reflection tokens and then distill their knowledge into an in-house C. For each group of reflection tokens, we randomly sample instances from the original training data: $\{x_{sample}, y\}$. As different reflection token groups have their own definitions and input, as shown in Table 1, we use different instruction prompts for them. Here, we use Retrieve and an example. We prompt GPT-4 with a type-specific instruction (“Given an instruction, make a judgment on whether finding some external documents from the web helps to generate a better response.”) followed by few-shot demonstrations. The original task input $x$ and output $y$ to predict an appropriate reflection token as text: $p(r|[x, y])$. Manual assessment reveals that GPT-4 reflection token predictions show high agreement with human evaluations. We collect 4k-20k supervised training data for each type and combine them to form training data for C. Appendix Section D shows the full list of instructions, and A contains more details about our analysis.

## Critic learning.

After we collect training data $D_{critic}$, we initialize C with a pre-trained LM and train it on $D_{critic}$ using a standard conditional language modeling objective, maximizing likelihood:

$$
\max_{C} E_{(x,y)\sim D_{critic}} \log p_C(r|x, y), \quad r \text{ for reflection tokens.} \tag{1}
$$

Though the initial model can be any pre-trained LM, we use the same one as the generator LM (i.e., Llama 2-7B; Touvron et al. 2023) for C initialization. The critic achieves a higher than 90% agreement with GPT-4-based predictions on most reflection token categories (Appendix Table 5).

### 3.2.2 Training the generator model

#### Data collection for generator.

Given an input-output pair $(x, y)$, we augment the original output $y$ using the retrieval and critic models to create supervised data that precisely mimics the SELF-RAG inference-time process (Section 3.1). For each segment $y_j$, we run C to assess whether additional passages could help to enhance generation. If retrieval is required, the retrieval special token $<retrieve>=yes$ is added, and R retrieves the top $K$ passages. For each passage, C further evaluates whether the passage supports the model generation and predicts $<isRel>$. If a passage is relevant, C further evaluates whether the passage supports the model generation and predicts $<isNotRel>$. Critique tokens $<isRel>$ and $<isNotRel>$ are appended after the retrieved passage or generations. At the end of the output, y or yr, C predicts the overall utility token $<isBest>$ and an augmented output with reflection tokens and the original input pair is added to $D_{gen}$. See the example training data in Figure 2.

#### Generator learning.

We train the generator model M by training on the curated corpus augmented with reflection tokens $D_{gen}$ using the standard next token objective:

$$
\max_{M} E_{(x,y)\sim D_{gen}} \log p_M(y|x). \tag{2}
$$

Unlike C training (Eq. 1), M learns to predict the target output as well as the reflection tokens. During training, we mask out the retrieved text chunks (surrounded by <p> and </p> in Figure 2 for loss calculation and expand the original vocabulary V with a set of reflection tokens $\{<Critique>, <Retrieve>\}$.

#### Connections to prior work on learning with critique.

Recent work incorporates additional critique (feedback) during training, e.g., RLHF (Ouyang et al. 2022) via PPO. While PPO relies on
```

### --- Page 0006 ---

```markdown
# Preprint

## 3.3 SELF-RAG INFERENCE

Generating reflection tokens to self-evaluate its own output makes SELF-RAG controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. For tasks demanding factual accuracy (Min et al., 2023), we aim for the model to retrieve passages more frequently to ensure that the output aligns closely with the available evidence. Conversely, in more open-ended tasks, like composing a personal experience essay, the emphasis shifts towards retrieving less and prioritizing the overall creativity or utility score. In this section, we describe approaches to enforce control to meet these distinct objectives during the inference process.

### Adaptive retrieval with threshold.

SELF-RAG dynamically decides when to retrieve text passages by predicting $Retrieve$. Alternatively, our framework allows a threshold to be set. Specifically, if the probability of generating $Retrieve=Yes$ is normalized over all output tokens in $Retrieve$, we trigger retrieval (details in Appendix Section A.3).

### Free-decoding with critique tokens.

At each segment step $t$, when retrieval is required, based either on hard or soft conditions, $R$ retrieves $K$ passages, and the generator $M$ processes each passage in parallel and outputs $K$ different continuation candidates. We conduct a segment beam search (with the beam size=$B$) to obtain the top-$B$ segment continuations at each timestamp $t$, and return the best sequence at the end of generation. The score of each segment $y_t$ with respect to passage $d$ is updated with a critique that is the linear weighted sum of the normalized probability of each Critique token type. For each critique token group (e.g., $[IsRel]$), we denote its score at timestamp $t$ as $s^t_G$, and we compute a segment score as follows:

$$
f(y_t, d, Critique) = p(y_t|x, d, y_{<t}) + S(Critique), \tag{3}
$$

$$
S(Critique) = \sum_{G \in G} w^G s^G_C \text{ for } G \in \{[IsRel], [IsUtl], [IsNts]\}, \tag{4}
$$

where $s^G_C = \frac{p(d|x)}{\sum_{G'} p(d|x)}$ stands for the generation probability of the most desirable reflection token (e.g., $[IsRel] \text{-} Relevant$) for the critique token type $G$ with $N^G$ distinct tokens (that represent different possible values for $G$). The weights $w^G$ in Eq. 4 are hyperparameters that can be adjusted at inference time to enable customizable behaviors at test time. For instance, to ensure that result $y$ is mostly supported by evidence, we can assign a weight term for the $[IsUtl]$ score higher, with relatively lowering weights for other aspects. Alternatively, we could further enforce hard constraints during decoding using $Critique$. Instead of using a soft reward function in Eq. 4, we could explicitly filter out a segment continuation when the model generates an undesirable critique token (e.g., $[IsUtl \text{-} No\ support]$). Balancing the trade-off between multiple preferences has been studied in RLHF (Touvron et al., 2023; Wu et al., 2023), which often requires training to change models’ behaviors. SELF-RAG tailors an LM with no additional training.

## 4 EXPERIMENTS

### 4.1 TASKS AND DATASETS

We conduct evaluations of our SELF-RAG and derive baselines on a range of downstream tasks, holistically evaluating outputs with metrics designed to assess overall correctness, factuality, and fluency. Throughout these experiments, we conduct zero-shot evaluations, where we provide instructions describing tasks without few-shot demonstrations (Wei et al., 2022; Sanh et al., 2022). Details of our experiments’ settings, including test-time instructions, are available in the Appendix Section B.1.

Closed-set tasks include two datasets, i.e., a fact verification dataset about public health (PubHealth; Zhang et al. 2023) and a multiple-choice reasoning dataset created from scientific exams (ARC:
```


### --- Page 0007 ---

```markdown
# Preprint

**Challenge:** Clark et al. (2018). We use accuracy as an evaluation metric and report on the test set. We aggregate the answer probabilities of target classes for both of these datasets (Appendix Section B.2).

**Short-form generations tasks** include two open-domain question answering (QA) datasets, PopQA (Mallen et al., 2022) and TriviaQA-unfiltered (Joshi et al., 2017), where systems need to answer arbitrary questions about factual knowledge. For PopQA, we use the long-tail subset, consisting of 1,399 rare entity queries whose monthly Wikipedia page views are less than 100. As the TriviaQA-unfiltered (open) test set is not publicly available, we follow prior work's validation and test split (Min et al., 2019; Guu et al., 2020), using 11,313 test queries for evaluation. We evaluate performance based on whether gold answers are included in the model generations instead of strictly requiring exact matching, following Mallen et al. (2023); Schick et al. (2023).

**Long-form generation tasks** include a biography generation task (Min et al., 2023) and a long-form QA task ALCE-ASQA (Gao et al. (2023); Stelmach et al. (2022). We use FactScore (Min et al., 2023) to evaluate biographies, and we use official metrics of correctness (str-emb), fluency based on MAUVE (Pillai et al., 2021), and citation precision and recall (Gao et al., 2023) for ASQA. 

## 4.2 BASELINES

**Baselines without retrievals.** We evaluate strong publicly available pre-trained LLMs, Llama2\_7B (Touvron et al., 2023), instruction-tuned models, Alpaca\_7B\_13B (Dubois et al., 2023) (our replication based on Llama2); and models trained and reinforced using private data, Chat-GPT (Ouyang et al., 2022) and Llama2\_chat\_13B. For instruction-tuned LMs, we use the official system prompt or instruction format used during training if publicly available. We also compare our method to concurrent work, CoveE58 (Dhuliwala et al., 2023), which introduces retrieval prompting to improve the factuality of LLM generations.

**Baselines with retrievals.** We evaluate models augmented with retrieval at test time or during training, given the query prepended with the top retrieved documents using the same retriever as in our system. It also includes Llama2-FT, where Llama2 is fine-tuned on training data without the reflection tokens or retrieved passages. We also report the result of retrieval-augmented baselines with LMs trained with private data: Ret-ChatGPT and Ret-Llama2-chat, which deploy the same augmentation technique above, as well as perplexity, and an InstructGPT-based production search system. The second category includes concurrent methods that are trained with retrieved text passages, i.e., SAIL (Luo et al., 2023) is instruction-tuning an LM on the Alpaca instruction-tuning data with top retrieved documents inserted before instructions, and Toolformer (Schick et al., 2023) to pre-train LM with API calls (e.g., Wikipedia APIs).

## 4.3 EXPERIMENTS

**Training data and settings.** Our training data consists of diverse instruction-following input-output pairs. In particular, we sample instances from Open-Inst processed data (Wang et al., 2023) and knowledge-intensive datasets (Petroni et al., 2021; Stelmach et al., 2022; Mihaylov et al., 2018). In total, we use 150k instruction-output pairs. We use Llama2 7B and 13B (Touvron et al., 2023) as our generator base LM, and we use Llama2 7B as our base retriever LM. For the retriever model $R$, we use the off-shelf Contriver-MS MARCO (Izacard et al., 2022a) by default and retrieve up to ten documents for each input. More training details are in the Appendix Section B.1.

**Inference settings.** At a default configuration, we assign the weight terms $w_{1}$, $w_{2}$, $w_{3}$ values of 1.0, 1.0 and 0.5, respectively. To encourage frequent retrieval, we set the retrieval threshold to 0.2 for most tasks and 0 for ALCE (Gao et al., 2023) due to citation requirements. We speed up inference using film (Kwon et al., 2023). At each segment level, we adopt a beam width of 2. For a token-level generation, we use greedy decoding. By default, we use the top five binaries from Contriver-MS MARCO (Izacard et al., 2022a); for biographies and open-domain QA, we use additional top five documents retrieved by a web search engine, following Luo et al. (2023); for ASQA, we use the author-provided top 5 documents by GTR-XXL (Ni et al., 2022) across all baselines for a fair comparison.

5\footnote{https://github.com/princeton-nlp/ALCE}  
6\footnote{We report numbers using the results reported in the paper or the implementations are not available.}
```

### --- Page 0008 ---

```markdown
## Preprint

### Table 2: Overall experiment results on six tasks. Bold numbers indicate the best performance among non-proprietary models, and gray-colored bold text indicates the best proprietary model when they outperform all non-proprietary models. * indicates concurrent or recent results reported by concurrent work. - indicates numbers that are not reported by the original papers or are not applicable. Models are sorted based on scale. FS, em, rg, mau, prec, rec denote FactScore (factuality); str, em, rg denote (correctness); MAUVE (fluency); citation precision and recall, respectively.

| Short-form | Closed-set | Long-form generations (with citations) |
|------------|------------|----------------------------------------|
| PopQA      | TQA        | Pub   | ARC | Bio  | ASQA |
| LM         | (acc)     | (acc) | (FS) | (em) | (rg) | (mau) | (prec) | (rec) |
|------------|------------|-------|------|------|------|------|-------|------|
| Llama2-c13 | 20.0      | 59.3  | 49.4 | 58.9 | 22.4 | 29.6 | 28.6  | -    |
| Ret-Llama2-c13 | 51.8  | 59.8  | 52.1 | 37.9 | 79.9 | 32.8 | 34.8  | 43.8 |
| ChatGPT    | 29.3      | 74.3  | 70.1 | 75.3 | 71.8 | 35.3 | 36.2  | 68.8 |
| Ret-ChatGPT | 50.8     | 65.7  | 54.7 | 75.3 | 40.7 | 39.9 | 79.7  | 65.1 |
| Perplexity.ai | -      | -     | -    | -    | -    | -    | -     | -    |
|------------|------------|-------|------|------|------|------|-------|------|
| Baselines without retrieval |            |       |      |      |      |      |       |      |
| Llama2     | 14.7      | 30.5  | 34.2 | 21.8 | 44.5 | 7.9  | 15.3  | 19.0 |
| Alpaca2    | 23.6      | 54.5  | 49.8 | 45.8 | 18.8 | 29.4 | 61.7  | -    |
| Llama2-13  | 17.7      | 58.5  | 29.4 | 53.4 | 7.2  | 12.4 | 16.0  | -    |
| Alpaca13   | 24.4      | 61.3  | 55.5 | 54.9 | 50.2 | 22.9 | 70.6  | -    |
| CoVE65*    | -          | -     | -    | -    | -    | -    | 71.2  | -    |
|------------|------------|-------|------|------|------|------|-------|------|
| Baselines with retrieval |            |       |      |      |      |      |       |      |
| Toolformer*68 | 48.8  | -     | -    | -    | -    | -    | -     | -    |
| Llama2     | 23.2      | 30.0  | 78.0 | 15.2 | 22.1 | -    | 2.9   | 4.0  |
| Alpaca2    | 46.7      | 61.4  | 40.2 | 48.0 | 76.6 | 30.9 | 33.3  | 57.9 |
| Llama2-FT* | 48.7      | 57.3  | 64.5 | 65.8 | 72.8 | 31.0 | 58.8  | 51.2 |
| Llama2-13  | 45.7      | 47.0  | 30.2 | 77.5 | 16.3 | 20.5 | 24.7  | 2.3  |
| Alpaca13   | -          | -     | -    | -    | -    | -    | -     | -    |
| Our SELF-RAG* | 54.9  | 66.4  | 72.4 | 67.3 | 81.2 | 30.7 | 34.5  | 66.9 |
| Our SELF-RAG 138 | 55.8 | 69.3  | 74.5 | 73.1 | 80.2 | 31.7 | 37.0  | 71.6 |

## 5 RESULTS AND ANALYSIS

### 5.1 Main Results

**Comparison against baselines without retrieval.** Table 2 (top presents the baselines without retrieval. Our SELF-RAG (bottom two rows) demonstrates a substantial performance advantage over supervised fine-tuned LLMs in all tasks and even outperforms ChatGPT in PubHealth, PopQA, biography generations, and ASQA (Rouge and MAUVE). Our approach also significantly outperforms a concurrent method that employs sophisticated prompt engineering; specifically, on the bio generation task, our 7B and 13B models outperform the concurrent CoVE (Dhuliwala et al., 2023), which iteratively prompts Llama2658 to refine output.

**Comparison against baselines with retrieval.** As shown in Tables 2 (bottom), our SELF-RAG also outperforms existing RAG in many tasks, obtaining the best performance among non-proprietary LM-based models on all tasks. While our method outperforms other baselines, on PopQA or Bio, powerful instruction-tuned LMs with retrieval (e.g., Llama2-chat, Alpaca) show large gains from their non-retrieval baselines. However, we found that these baselines provide limited solutions for tasks where we cannot simply copy or extract sub-strings of retrieved passages. On PubHealth and ARC-Challenge, baselines with retrieval do not improve performance notably from their non-retrieval counterparts. We also observe that most baselines with retrieval struggle to improve citation accuracy. On ASQA, our model shows significantly higher citation precision and recall than all other models except ChatGPT. Gao et al. (2023) found that ChatGPT consistently exhibits superb efficacy in this particular task, surpassing smaller LMs. Our SELF-RAG bridges this performance gap, even outperforming ChatGPT in citation precision, which measures whether the model-generated claim is fully supported by cited evidence. We also found that on the metrics for factual reporting, SELF-RAG 7B occasionally outperforms our 13B due to the tendency of smaller SELF-RAG to often generate.
```

### --- Page 0009 ---

```markdown
# Preprint

| PQA (acc) | Med (acc) | AS (em) |
|-----------|-----------|---------|
| SELF-RAG (50k) | 45.5      | 73.5    | 32.1    |
| **Training**   |           |         |         |
| No Retrieve $R$ | 43.6      | 67.8    | 31.0    |
| No Critic $C$  | 42.6      | 72.0    | 18.1    |
| $T_{est}$      | -         | -       | -       |
| No retrieval    | 24.7      | 73.0    | -       |
| Hard constraints | 28.3      | 72.6    | -       |
| Retrieve top 1  | 41.8      | 73.1    | 28.6    |
| Remove $[ISSUF]$ | 44.1      | 73.2    | 30.6    |

### Figure 3: Analysis on SELF-RAG
(a) Ablation studies for key components of SELF-RAG training and inference based on our 7B model. (b) Effects of soft weights on ASQA citation precision and Mauve (fluency). (c) Retrieval frequency and normalized accuracy on PubHealth and PopQA.

![Ablation studies for key components of SELF-RAG](assets/page_0009_img_1.png)

5.2 ANALYSIS

Ablation studies. We conducted a set of ablations of our framework to identify which factors play key roles. We also modeled various training methods differently than our model: No Retrieve trains LM using the standard instruction-following method given instruction-output pairs without retrieved passages; No Critic trains LM trained with input-output pairs that are always augmented with the one retrieved document without reflection tokens. This is similar to SAIL (Luo et al., 2023), and we use our instruction-output data instead of using the Alpaca dataset (Dubois et al., 2023), as in SAIL. We also conduct ablation on our inference-time algorithm, including retrieval during inference; Hard constraints indicate the model performance that retrieves when $Retrieve$ instead of using the adaptive threshold; Retrieve top 1 always retrieves and uses the top one document only, similar to standard RAG approaches; Remove $[ISSUF]$ indicates the model performance that removes $[ISSUF]$ score only during critique-guided beam search in Eq. 4. In this ablation experiment, we use a training instance size of 50k for a more efficient training variations. Later in this section, we conduct an analysis of the effect of training data size. We conduct the ablation studies on three datasets, PopQA, PubHealth, and ASQA. On ASQA, we evaluate models on sampled 150 instances and exclude ablations involving adaptive or no retrieval processes.

We show in Table 3 the ablation results. The top part of the table shows results for training ablations, and the bottom part is for inference ablations. We see that all components play important roles. We also observe a large performance gap between SELF-RAG and No Retrieve or Critic baselines across tasks, indicating that training an LM with those models largely contributes to the performance gain of SELF-RAG. Using the top passages regardless of their relevance (Retrieve top 1) as in conventional RAG approaches causes a large drop in PopQA and ASQA, and removing $[ISSUF]$ during the beam search hurts performance on ASQA. This demonstrates the effectiveness of SELF-RAG’s capabilities of carefully selecting generations based on fine-grained multiple criteria, instead of heavily using all of the top passages from the retrieval model or solely depending on relevance scores.

**Effects of inference customization.** One key benefit of our proposed framework is that it enables us to control how much each critique type affects the final generation sampling. We analyze the effects of different parameter weights on the output of our 7B model during inference time on ASQA, where multiple evaluation aspects are considered. Figure 3b shows the effects of changing the weighting term for $[ISSUF]$, which criticizes how supported the output is by the text passage. As the figure shows, increasing the weight leads to positive effects on the model’s citation precision since this puts more emphasis on whether model generation is supported by the evidence. On the
```

### --- Page 0010 ---

```markdown
![Training scale and Human analysis: (a) PopQA, (b) PubHealth, (c) ASQA (citation precision), respectively. (d) Human analysis on SELF-RAG outputs as well as reflection tokens.](assets/page_0010_img_1.png)

Figure 4: Training scale and Human analysis: (a) (b) (c) Training scale analysis shows the effect of the training data scale on PopQA, PubHealth and ASQA (citation precision), respectively. (d) Human analysis on SELF-RAG outputs as well as reflection tokens.

contrary, a larger weight results in lower MAUVE scores: when generation gets longer and more fluent, there are often more claims that are not fully supported by citations, consistent with findings by Liu et al. (2023a). Our framework lets practitioners choose and customize models’ behaviors at test time by adjusting such parameters without requiring additional training.

Efficiency and accuracy trade-off. Using our framework, practitioners can adjust how often retrieval occurs using the token probability of reward tokens. We evaluate how this adaptive threshold affects overall accuracy and frequency of retrieval, and we evaluate the performance with varying numbers of threshold δ (larger δ results in less retrieval) on PubHealth and PopQA. Figure 3 shows that the model’s retrieval frequencies dramatically change on both datasets, as δ varies. On one hand, performance deterioration by retrieving less is smaller on PubHealth but larger in PopQA.

Effects of training data size. We conduct an analysis of how the data scale affects the model’s performance. In particular, we randomly sample 5k, 10k, 20k, and 50k instances from our original training instances, and fine-tune four SELF-RAG 7B variants on those subsets. Here we compare the model performance on PopQA, PubHealth, and ASQA (citation precision) with our final SELF-RAG trained on the full 150k instances. We also evaluate Figures 4a, 4b and 4c show the model’s performance trained on different amount of data. Across all datasets, increasing data size often shows upward trajectories and the improvements are significantly larger in PopQA and ASQA, while we do not observed such significant improvements on Llama2-7B7 when increasing the training data from 50k to 150k. These results also indicate that further expanding the training data of SELF-RAG may lead to further improvements, although in this work we limit our training data size to 150k.

Human evaluations. We conduct small human evaluations on SELF-RAG outputs, as well as the reliability of predicted reflection tokens. In particular, we sampled 50 samples from PopQA and Bio results. Following Menick et al. (2022), human annotators evaluated S&P, which indicates whether the model output is plausible (i.e., the output is a reasonable and on-topic response to the question as if it were occurring in a conversation) and supported (i.e., the provided evidence is sufficient to verify the validity of the answer). For S&P, we do not consider the instances where SELF-RAG predicts irrelevant or no support. We then ask our annotators whether the model-predicted reflection tokens about [Ref] and [Sut] match their inspections (e.g., whether the fully supported output is supported by the cited evidence). Human annotators find SELF-RAG answers are often plausible and supported by relevant passages with higher S&P scores on short-form PopQA, which is consistent with Menick et al. (2022). Human annotators also find [Ref] and [Sut] reflection token predictions are mostly aligned with their assessments. Appendix 6 shows several annotated examples and explanations on assessments.

6 CONCLUSION

This work introduces SELF-RAG, a new framework to enhance the quality and factuality of LLMs through retrieval on demand and self-reflection. SELF-RAG trains an LM to retrieve, generate, and critique text passages and its own generation by predicting the next tokens from its original vocabulary as well as newly added special tokens, called reflection tokens. SELF-RAG further enables the tailoring of LM behaviors at test time by leveraging reflection tokens. Our holistic evaluations on six tasks using multiple metrics demonstrate that SELF-RAG significantly outperforms LLMs with more parameters or with conventional retrieval-augmented generation approaches.
```

### --- Page 0011 ---

```markdown
# Ethical Concerns

This work aims to improve the factuality of LLM outputs, the lack of which continues to cause numerous real-world problems (e.g., spread of misinformation and provision of incorrect and dangerous advice). While our method shows significant improvements in terms of performance, factuality, and citation accuracy, it can still generate outputs that are not fully supported by the citations. We hope that explicit self-reflection and fine-grained attribution may help users verify factual errors in the model outputs.

# Acknowledgments

We thank Sewon Min, Scott Wen-tau Yih, Sean Welleck, and Kawin Ethayarajh for fruitful discussions in the early stages of this work. We thank Sewon Min, Joongwon (Daniel) Kim, and Sandy Kaplan for valuable feedback on the paper, and Tianyu Gao and Weijia Shi for their help on evaluations. Akari Asai is supported by the IBM Fellowship. We thank Stability AI for providing computing to train and evaluate the LMs in this work, and Microsoft Accelerate Foundation Models Research Program for the access to OpenAI APIs. This work was funded in part by the DARPA MCS program through NIWC Pacific (N66001-19-2-4031), NSF IIS-2044660, and gifts from AI2.

# References

| Author(s) | Title | Source | Year | URL |
|-----------|-------|--------|------|-----|
| Akari Asai, Kazuma Hashimoto, Hannaneh Hajishirzi, Richard Socher, and Caiming Xiong. | Learning to retrieve reasoning paths over wikipedia graph for question answering. | International Conference on Learning Representations | 2020 | [Link](https://openreview.net/forum?id=5JgVhKrYDH) |
| Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. | Retrieval-based language models and applications. | Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Tutorial) | 2023a | [Link](https://aclanthology.org/2023.acl-tutorials.6) |
| Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. | Task-aware retrieval with instructions. | Findings of the Association for Computational Linguistics | 2023b | [Link](https://aclanthology.org/2023.findings-acl.225) |
| Bernd Bohnet, Vinh Q Tran, Pat Vega, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Jörg Eisenstein, Kuzman Ganchev, Jonathan Herzig, Kai Hui, et al. | Attributed question answering: Evaluation and modeling for attributed large language models. | arXiv preprint arXiv:2212.00837 | 2022 | [Link](https://arxiv.org/abs/2212.00837) |
| Lingjiao Chen, Maite Zahariev, and James Zou. | How is chatgpt's behavior changing over time? | arXiv preprint arXiv:2307.09009 | 2023 | [Link](https://arxiv.org/abs/2307.09009) |
| Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. | Think you have solved question answering? try the arc, the ai2 reasoning challenge. | arXiv preprint arXiv:1803.05457 | 2018 | [Link](https://arxiv.org/abs/1803.05457) |
| Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. | Flashattention: Fast and memory-efficient exact attention with io-awareness. | Advances in Neural Information Processing Systems | 2022 | [Link](https://openreview.net/forum?id=H4dpF5ism) |
| Shehzad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. | Chain-of-verification reduces hallucination in large language models. | arXiv preprint arXiv:2309.11495 | 2023 | [Link](https://arxiv.org/abs/2309.11495) |
| Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. | Wizard of wikipedia: Knowledge-powered conversational agents. | International Conference on Learning Representations | 2019 | [Link](https://openreview.net/forum?id=1713iRkqM) |
| Yann Dubois, Xuechen Li, Rohan Tatori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. | AlpacaFarm: A simulation framework for methods that | | | |
```

### --- Page 0012 ---

```markdown
# Preprint

learn from human feedback. arXiv preprint arXiv:2305.14387, 2023. URL https://arxiv.org/abs/2305.14387.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. arXiv preprint arXiv:2305.14627, 2023. URL https://arxiv.org/abs/2305.14627.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In *International Conference on Machine Learning*, 2020. URL https://dl.acm.org/doi/pdf/10.5555/3524938.3525306.

Gautier Izacard, Mathilde Caron, Lucas Hosseni, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. *Transactions on Machine Learning Research*, 2022a. URL https://openreview.net/forum?id=1nKplx17bo.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseni, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Few-shot learning with retrieval augmented language models. arXiv preprint arXiv:2208.03299, 2022. URL https://arxiv.org/abs/2208.03299.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Ming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. arXiv preprint arXiv:2305.06983, 2023. URL https://arxiv.org/abs/2305.06983.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2017. URL https://aclweb.org/anthology/P17-1147.

Nitish Srivastava Kesar, Bryan McCann, Lar V Varshney, Calming Xiong, and Richard Socher. A conditional transformer language model for controllable generation. arXiv preprint arXiv:1909.05588, 2019. URL https://arxiv.org/abs/1909.05588.

Tomasz Korbak, Kejian Shi, Angelica Chen, Rasika Vinayak Bhale Rao, Christopher Buckley, Jason Phang, Samuel R Bowman, and Ethan Perez. Pretraining language models with human preferences. In *International Conference on Machine Learning*, 2022. URL https://openreview.net/forum?id=ATiWk0Ec.

Tom Kwiatkowski, Jennaimaria Palangi, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Kenton Lee, Cristina Tautkova, Lin Jones, Matthew Kelsey, Ming-Wei Chang, Andrew M. Dai, Jakub Uszkoreit, Quc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 2019. URL https://aclanthology.org/Q19-1206.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with padedation. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023. URL https://arxiv.org/abs/2309.06180.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Namrata Ganesh, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems*, 2020. URL https://proceedings.neurips.cc/paper/2020/file/6b9432300257f0b26b2945d7481e5-Paper.pdf.

Xi Victoria Lin, Xilun Chen, Mingda Chen, Weija Shi, Maria Lomeli, Rick James, Pedro Rodriguez, Jacob Kahn, Gregory Szilvasy, Mike Lewis, Luke Zettlemoyer, and Scott Yih. Ra-dt: Retrieval-augmented dual instruction tuning, 2023. URL https://arxiv.org/abs/2310.01352.

Nelson F. Liu, Tianyu Zhang, and Percy Liang. Evaluating verifiability in generative search engines. arXiv preprint arXiv:2304.09488, 2023a. URL https://arxiv.org/abs/2304.09488.
```

### --- Page 0013 ---

```markdown
# Preprint

Yang Liu, Dan Ter, Yichong Xu, Shuohang Wang, Ruochuen Xu, and Chenguang Zhu. Gptval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634, 2023b. URL [https://arxiv.org/abs/2303.16634](https://arxiv.org/abs/2303.16634).

Ximing Lu, Sean Welleck, Jack Hessel, Liwei Jiang, Lianhui Qin, Peter West, Prithiviraj Manoharan, and Yejin Choi. QUARK: Controllable text generation with reinforced unlearning. In *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview.net/forum?id=5HIds3ux50](https://openreview.net/forum?id=5HIds3ux50).

Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Brandon Fox, Helen Meng, and James Glass. Sail: Search-augmented instruction learning. arXiv preprint arXiv:2305.15225, 2023. URL [https://arxiv.org/abs/2305.15225](https://arxiv.org/abs/2305.15225).

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhume, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-reference: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023. URL [https://arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651).

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2023. URL [https://aclanthology.org/2023.long-546](https://aclanthology.org/2023.long-546).

Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glase, Susannah Young, Lucy Campbell-Gillingham, Geoffrey Irving, et al. Teaching language models to support answers with verified outputs. arXiv preprint arXiv:2203.11417, 2022. URL [https://arxiv.org/abs/2203.11417](https://arxiv.org/abs/2203.11417).

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 2018. URL [https://aclanthology.org/D18-1260](https://aclanthology.org/D18-1260).

Sewon Min, Danqi Chen, Hannaneh Hajishirzi, and Luke Zettlemoyer. A discrete hard EM approach for weakly supervised question answering. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 2019. URL [https://aclanthology.org/D19-1284](https://aclanthology.org/D19-1284).

Sewon Min, Kalpesh Krishna, Xinxu Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. arXiv preprint arXiv:2305.14251, 2023. URL [https://arxiv.org/abs/2305.14251](https://arxiv.org/abs/2305.14251).

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021. URL [https://arxiv.org/abs/2112.09332](https://arxiv.org/abs/2112.09332).

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yifei Yang. Large dual encoders are generalizable retrievers. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 2022. URL [https://aclanthology.org/2022.emnlp-main.669](https://aclanthology.org/2022.emnlp-main.669).

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. URL [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774).
```

### --- Page 0014 ---

```markdown
# Preprint

Ryan Lowe. Training language models to follow instructions with human feedback. In *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview.net/forum?id=T6K8KACXEON](https://openreview.net/forum?id=T6K8KACXEON).

Debjit Paul, Mete Ismailyzada, Maxime Peyrard, Beatriz Borges, Antoine Bosselut, Robert West, and Boaz Faltings. Refiner: Reasoning feedback on intermediate representations. arXiv preprint arXiv:2304.01904, 2023. URL [https://arxiv.org/abs/2304.01904](https://arxiv.org/abs/2304.01904).

Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. KILT: a benchmark for knowledge intensive language tasks. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2021. URL [https://aclanthology.org/2021.naacl-main.200](https://aclanthology.org/2021.naacl-main.200).

Krishna Pillutla, Swabhyadipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. MAUVE: Measuring the gap between neural text and human text using divergence frontiers. In *Advances in Neural Information Processing Systems*, 2021. URL [https://openreview.net/forum?id=Tq7nJp7PR](https://openreview.net/forum?id=Tq7nJp7PR).

Samraj Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In *Proceedings of the International Conference for Performance Computing, Networking, Storage and Analysis*, 2022. URL [https://dl.acm.org/doi/10.5555/3433701.3433727](https://dl.acm.org/doi/10.5555/3433701.3433727).

Ori Ram, Yoav Levine, Itay Dalmiedgos, Dor Muhlay, Amon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. *Transactions of the Association for Computational Linguistics*, 2023. URL [https://arxiv.org/abs/2302.03001](https://arxiv.org/abs/2302.03001).

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Linting Sutawika, Zaid Alyafei, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Anam Dew, M Saiful Bari, Canwen Xu, Shiming Thakker, Shanya Sharma Sharma, Eliza Szeczech, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debjyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matthew Shen, Yizheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abhishek Sharma, Andrea Santilli, Thubten Fwyas, Joan Alan Fries, Ryah Tewen Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush. Multitask prompt training enables zero-shot task generalization. In *International Conference on Learning Representations*, 2022. URL [https://openreview.net/forum?id=9vrb9D0W1](https://openreview.net/forum?id=9vrb9D0W1).

Timo Schick, Jane Dwiwedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023. URL [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761).

John Schulman, Filip Wolski, Prafula Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1706.06347, 2017. URL [https://arxiv.org/abs/1706.06347](https://arxiv.org/abs/1706.06347).

Freda Shih, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In *Proceedings of the 40th International Conference on Machine Learning*, 2023. URL [https://proceedings.mlr.press/v202/shih23a.html](https://proceedings.mlr.press/v202/shih23a.html).

Ivan Stelmach, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. ASQA: Factoid questions meet long-form answers. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 2022. URL [https://aclanthology.org/2022.emnlp-main.566](https://aclanthology.org/2022.emnlp-main.566).

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERification. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, 2018. URL [https://aclanthology.org/N18-1074](https://aclanthology.org/N18-1074).
```

### --- Page 0015 ---

```markdown
# Preprint

Hugo Tovron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmin Babaei, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. URL https://arxiv.org/abs/2307.09288.

Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. How far can camels go? exploring the state of instruction tuning on open resources. arXiv preprint arXiv:2306.04751, 2023. URL https://arxiv.org/abs/2306.04751.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=gEzrGC0zqR.

Zeqiu Wu, Yushi Hu, Weijia Shi, Noah Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A Smith, Marli Ostendorf, and Hannah Hajiijiri. Fine-grained human feedback gives better rewards for language model training. arXiv preprint arXiv:2306.01693, 2023. URL https://arxiv.org/abs/2306.01693.

Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, Xu Zhao, Min-Yen Kan, Junxian He, and Qizhe Xie. Decomposition enhances reasoning via self-evaluation guided decoding. arXiv preprint arXiv:2305.00633, 2023. URL https://arxiv.org/abs/2305.00633.

Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with compression and selective augmentation, 2023. URL https://arxiv.org/abs/2310.04408.

Xiang Yue, Boshi Wang, Kai Zhang, Zirui Chen, Yu Su, and Huan Sun. Automatic evaluation of attribution by large language models. arXiv preprint arXiv:2305.06311, 2023. URL https://arxiv.org/abs/2305.06311.

Tianhua Zhang, Hongyin Luo, Yung-Sung Chuang, Wei Fang, Luc Gaitskell, Thomas Hartvigsen, Xixin Wu, Danny Fox, Helen Meng, and James Glass. Interpretable unified language checking. arXiv preprint arXiv:2304.03728, 2023. URL https://arxiv.org/abs/2304.03728.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models, 2023. URL https://arxiv.org/abs/2310.04406.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019. URL https://arxiv.org/abs/1909.08593.
```

### --- Page 0017 ---

```markdown
# A SELF-RAG DETAILS

## A.1 REFLECTION TOKENS.

Definitions of reflection tokens. Below, we provide a detailed definition of reflection type and output tokens. The first three aspects will be provided at each segment level, while the final aspect is only given at each output level.

- **Retrieval-on-demand** ($\text{Retrieve}$): Given an input and previous-step generation (if applicable), an LM determines whether the continuation requires factual grounding. No indicates retrieval is unnecessary as the sequence does not require factual grounding or may not be enhanced by knowledge retrieval, Yes indicates retrieval is necessary. We additionally have continue to use evidence, which indicates that a model can continue to use the evidence retrieved previously. For instance, a passage may contain rich factual information, and thus SELF-RAG generates multiple segments based on the passage.

- **Relevant** ($\text{IsRel}$): Retrieved knowledge may not be always relevant to the input. This aspect indicates whether the evidence provides useful information (Relevant) or not (Irrelevant).

- **Supported** ($\text{IsSup}$): Attribution is the concept of whether the output is fully supported by certain evidence (Menick et al., 2022; Bohnet et al., 2022). This aspect judges how much information in the output is entailed by the evidence. We evaluate attributions in three scale, Fully supported, Partially supported, and No support / Contradictory, following Yue et al. (2023); Nakano et al. (2021).

- **Useful** ($\text{IsUse}$): Following the definitions from Liu et al. (2023a), we define the perceived utility as whether the response is a helpful and informative answer to the query, independently from whether it is in fact factual or not. This can also be viewed as plausibility in Menick et al. (2022). For usefulness, we use a five-scale evaluation (1 is the lowest and 5 is the highest).

Details of GPT-4-based data collections. We use the instruction and demonstration pairs to prompt GPT-4, listed in Section D. Following a official recommendation, we separate instances and outputs with “###”. We use the temperature 1 and set the maximum output token counts to be 200. We discard instances where GPT-4 does not follow the designated output formats or output sequences that do not match our expected category areas. As a result, we collected 1,2594 for $\text{Retrieve}$, 11,181 for $\text{IsSup}$, 19,317 for relevance, 3,831 for utility.

Manual analysis of the GPT-4 predictions. The authors of this paper manually assess randomly sampled 20 instances for each aspect and check if GPT-4 predictions match their assessments given the same instruction, demonstrations, and test instances. We found our assessments show high agreement with GPT-4 predictions, especially for relevance (95%), retrieval necessity (95%), and the degree of support (90%). Agreement was slightly lower in usefulness (80%), mostly due to the disagreement between 1 and 2 or 4 and 5.

## A.2 SELF-RAG TRAINING

Overview of training. Algorithm 2 provides a high-level overview of our training.

Full list of seed datasets. To sample diverse input-output pairs, we sample instances of the OpenInstruct (Wang et al., 2023) dataset. In particular, we use their ShareGPT, GPT-4 Alpaca, Alpaca, OpenAssistant, and FLAN subsets subsets. We also sample instances from a couple of knowledge-intensive datasets, Natural Questions (Kwiatkowski et al., 2019), Wizard of Wikipedia (Dinan et al., 2019) and FEVER (Thorne et al., 2018) from the KILT benchmark (Petroni et al., 2021), ASQA (Stelmakh et al., 2022) and multiple QA datasets including ARC-Easy and OpenBookQA (Mihaylov et al., 2018). Table 3 shows the full list of training instances, and in total, we use 145,619 instances.

Performance of the Critic. We evaluate the accuracy of reward predictions by splitting GPT-4 generated feedback into training, development, and test sets. The accuracy of the reward model is as follows. Table 5 shows the model performance of predicting GPT-4 judgments. As you can see, overall our fine-tuned reward model shows high prediction matching with GPT-4 predicted feedback.
```

### --- Page 0018 ---

```markdown
# Preprint

## Algorithm 2 SELF-RAG Training

1. Input input-output data $D = \{X, Y\}$, generator $M, C \theta$
2. Initialize $C$ with a pre-trained LM
3. Sample data $\{X_{sample}, Y_{sample}\} \sim \{X, Y\}$  ⟶ Training Critic LM (Section 3.2.1)
4. For $(x, y) \in (X_{sample}, Y_{sample})$ do
5. &nbsp;&nbsp;&nbsp; Prompt GPT-4 to collect a reflection token $r$ for $(x, y)$
6. &nbsp;&nbsp;&nbsp; Add $((x, y), r)$ to $D_{critic}$
7. Update $C$ with next token prediction loss  ⟶ Critic learning; Eq. 1
8. Initialize $M$ with a pre-trained LM  ⟶ Training Generator LM (Section 3.2.2)
9. For $(x, y) \in (X, Y)$ do
10. &nbsp;&nbsp;&nbsp; Run $C$ to predict $r$ given $(x, y)$
11. &nbsp;&nbsp;&nbsp; Add $((x, y), r)$ to $D_{gen}$
12. Update $M$ on $D_{gen}$ with next token prediction loss  ⟶ Generator LM learning; Eq. 2

| Dataset name       | category                | Data source      | the number of instances |
|--------------------|-------------------------|------------------|-------------------------|
| GPT-4 Alpac        | Instruction-following    | Open-Instruct    | 26,168                  |
| Stanford Alpaca    | Instruction-following    | Open-Instruct    | 25,153                  |
| FLAN-V2            | Instruction-following    | Open-Instruct    | 18,817                  |
| ShareGPT           | Instruction-following    | Open-Instruct    | 13,406                  |
| Open Assistant 1   | Instruction-following    | Open-Instruct    | 9,464                   |
| Wizard of Wikipedia | Knowledge-intensive      | KILT             | 17,367                  |
| Natural Questions   | Knowledge-intensive      | KILT             | 15,535                  |
| FEVER              | Knowledge-intensive      | KILT             | 9,966                   |
| OpenBookQA         | Knowledge-intensive      | HF Dataset       | 4,969                   |
| Arc-Easy           | Knowledge-intensive      | HF Dataset       | 2,147                   |
| ASQA               | Knowledge-intensive      | ASQA             | 3,897                   |

### Table 3: The generator LM $M$ training data statistics.

| base LM   | Retrieve | IsSup | IsRL | IsUse |
|-----------|----------|-------|------|-------|
| Llama2-7B | 93.8     | 93.5  | 80.2 | 73.5  |
| FLAN-3B   | 85.6     | 73.1  | 82.0 | 72.1  |

### Figure 5: Reward prediction accuracy using GPT-4 predictions as ground-truth predictions.

While our final model uses Llama2-7B as a base LM, we also train and compare FLAN-3B (Wei et al., 2022) model on the same data, to investigate the effectiveness of different data sizes affect final reward predictions. In most aspects, our reward model shows higher than 80% accuracy, indicating the powerful ability of fine-tuned specialized LMs to evaluate text. While both models show relatively lower performance on IsUse, this is because both models often confuse between the two highest cases (5 and 4), where human annotators can also disagree.

### Details of $M$ data creation.

Here, we provide detailed data creation procedures. Algorithm 3 summarizes the process. Here we set $y_t$ to $y$ for simplification. Once we first run in input data from the aforementioned datasets, to predict whether retrieval is needed or not. For the instances where the critic predicts $Retrieve=No$, we only predict the $IsUse$ given input and output. For instances where the critic predicts $Retrieve=Yes$, we first retrieve passages using the input and the entire output as queries, to find passages that are relevant to the entire output. We then split output sentences using Spacy.7 For each sentence, we run $C$ to predict whether the retrieval is necessary or not, given the input, preceding segments, and the initial retrieved passage. If $C$ predicts $Retrieve=No$, then do not insert any paragraph at the $t$th segment. If $C$ predicts $Retrieve=Yes$, then we use the original input and the $t$th segment as a retrieval query to find relevant passages for the $t$th segment. For each retrieved passage, we predict $IsRel$ and $IsSup$. If there is any passage and continuation with $IsRel=Relevant$ and $IsSup=Fully Supported$ / $IsSup=Partially Supported$.

7 [https://spacy.io](https://spacy.io)
```

### --- Page 0019 ---

```markdown
# Preprint

## Algorithm 3 M_gen Data creation

1: Input Input-output data $D = X, Y$  
2: for $(x, y) \in \{X, Y\}$ do  
3: \quad Given $(x, y)$, $c$ predicts $\text{Retrieve}$  
4: \quad if $\text{Retrieve}$ is predicted then  
5: \quad \quad Retrieve relevant passages $D$ using $R$ given $(x, y)$  
6: \quad for each $d \in D$ do  
7: \quad \quad $C$ predicts $\text{IsRel}$ for each $d$  \hfill $\triangleright$ Predict relevance of passages  
8: \quad \quad $C$ predicts $\text{IsSu}$ for each $(x, d)$  \hfill $\triangleright$ Predict supports of outputs  
9: \quad \quad $C$ predicts $\text{IsUt}$ for each $d$  \hfill $\triangleright$ Predict overall utility ($t = T$ only)  
10: \quad Sample $d$  
11: \quad else if $\text{Retrieve}$ is not predicted then  
12: \quad \quad $C$ predicts $\text{IsUt}$ given $x, y$  
13: \quad Add augmented $(x, y, d, r)$ to $D_{\text{gen}}$  

Training examples. Table 4 show several training examples used for $M$ training.

### A.3 SELF-RAG INFERENCE

Details of beam-search score calculations. We first compute scores for each critique type by taking the normalized probabilities of desirable tokens. For $\text{IsRel}$, we compute the score as follows:

$$
s(\text{IsRel}) = \frac{p(\text{IsRel} = \text{Relevant}) + p(\text{IsRel} = \text{Irrelevant})}{p(\text{IsRel} = \text{Relevant})}.
$$

For $\text{IsSu}$, we compute the score as follows:

$$
s(\text{IsSu}) = \frac{p(\text{IsSu} = \text{Fully}) + 0.5 \times p(\text{IsSu} = \text{Partially})}{S}
$$

where $S = \sum_{t \in \{\text{Fully, Partially, No}\}} p(\text{IsSu} = t)$. For $\text{IsUt}$ where we have a five-scale score, we compute the weighted sum of the scores. We assign weighted scores of $w = \{-1, -0.5, 0, 0.5, 1\}$ to the tokens $\text{IsUt} = \{1, 2, 3, 4, 5\}$, and compute the final scores as follows:

$$
s(\text{IsUt}) = \sum_{t \in \{1,2,3,4,5\}} w_t p(\text{IsUt} = t) / S,
$$

where $S = \sum_{t \in \{1,2,3,4,5\}} p(\text{IsUt} = t)$.

Details of adaptive retrieval. For retrieval based on soft constraints, we trigger retrieval if the following condition is satisfied:

$$
p(\text{Retrieve} = \text{Yes}) + p(\text{Retrieve} = \text{No}) > \delta.
$$

## B EXPERIMENTAL DETAILS

### B.1 MORE DETAILS OF TRAINING

More details of training and computations. We use 4 Nvidia A100 with 80GB memory to train our models. All models are trained for 3 epochs with a batch size of 128, a peak learning rate of $2 \times 10^{-5}$ with 3% warmup steps, and linear decay afterward. We set the maximum token length to be 2,048 for the 7B model, and 1,524 for the 13B model due to the memory constraint. We use Deepspeed stage 3 (Rajbhandari et al., 2020) to conduct multi-GPU distributed training, with training precision float16 enabled. FlashAttention (Dao et al., 2022) is used to make the long-context training more efficient. We run inference on our trained models using 1-2 Quadro RTX 6000 GPUs with 24GB memory.
```

### --- Page 0020 ---

```markdown
# Preprint

## B.2 MORE DETAILS OF EVALUATIONS

### Retrieval setup details.
By default, we use Contriever-MRS MARCO to retrieve the top five documents from Wikipedia, and use official Wikipedia embeddings based on 2018 English Wikipedia. On PopQA, where question and answer pairs are created based on WikiData in 2022, we found that the 2018 Wikipedia sometimes lacks articles about some entities that have been more recently added to Wikipedia. Therefore, for PopQA, we used the December 2020 preprocessed Wikipedia corpus provided by Izacard et al. (2022b) and generated document embeddings. 8 The issues of performance variance from different Wikipedia dumps have been reported by prior work (Asai et al., 2020; Izacard et al., 2022b). Yet, we observe limited effectiveness of such off-the-shelf retrieval models trained primarily on knowledge-intensive tasks for open-ended generation (e.g., instruction following). Recent or concurrent work studies instruction-tuning of retrieval systems (Asai et al., 2023b) or joint training of retrieval and LM components (Lin et al., 2023), while we leave exploring the effectiveness of such approaches for future work. For bio generation and open-domain QA tasks, we additionally retrieve five documents using Google Programmable Search 9 and search documents from English Wikipedia. As this API only provides snippets, we retrieve Wikipedia introductory paragraphs for the corresponding entries.

### Detailed experimental settings for individual datasets.
For OpenQA datasets, we set the maximum new token length to 100 tokens. For closed-set tasks (PubHealth and ARC-C), we set the maximum new token length to 50 for all baselines. For SELF-RAG inference on PubHealth and ARC-C, instead of determining the output with the highest score 4 as in other tasks, we aggregate the scores for each option and select the answer option with the highest score. We found in zero-shot settings of fact checking, some LLMs can generate capitalized class labels (e.g., True) while our gold labels are lower-cased. Therefore, across different LMs, for fact checking, we lowercase the predictions. In multiple choice tasks, we found some models generate answers in slightly different ways (e.g., A instead of A). We slightly modify instructions for each LM to avoid such violations, but we still observe issues including when each candidate and their predictions in format violations still remain. After that processing, in closed set tasks, model predictions match one of the gold classes in almost all cases. For ALCE, we found that Llama2-chat tend to generate significantly lower outputs than other models (e.g., on average, their output is nearly 100 token, while ChatGPT generates 400 tokens on average), resulting in inflated str-em scores. We limit the maximum generation length to 100 tokens for all baselines to avoid this issue, rather than the original 300 tokens in the ALCE paper. Consequently, all of the baseline output length is within 30-60 tokens. For FactScore, we set the maximum new token length to 500 for baselines and 200 for SELF-RAG at each segment level.

### Task-specific instructions.
Table 5 shows the list of the instructions used during evaluations. For Open-domain QA, we do not provide explicit instructions.

## C RESULTS

### C.1 ANALYSIS

#### Reliance on parametric- and non-parametric memories.
We conduct analysis on how frequently model answers come from retrieved passages (non-parametric memories) or their own parametric memories. On two open-domain QA datasets, TriviaQA and PopQA, we conduct the following analysis: 1) sample query models successfully answer correctly, 2) for each query in this group, check whether the matched ground-truth answer is a sub-string of the retrieved passage or not. We evaluate SELF-RAG 7B, Alpaca 7B, Alpaca 13B, and Llama2-Chat-13B. We found that SELF-RAG significantly less frequently generates answers that are not included in the provided evidence; in particular, in Alpaca 30B, 20% of the correct predictions are not included in the provided passages, followed by Llama2-chat 13B (18%) and Alpaca (15%), while it is only 2% in SELF-RAG. When retrieved passages are not relevant, SELF-RAG generates [1] = irrelevant, indicating that the following answers may not be factually grounded, while those instruction-tuned models continue to generate plausible answers.

8 [https://github.com/facebookresearch/atlas](https://github.com/facebookresearch/atlas)  
9 [https://programmablesearchengine.google.com/about/](https://programmablesearchengine.google.com/about/)
```

### --- Page 0021 ---

```markdown
## C.2 HUMAN EVALUATION EXAMPLES

Table 6 shows examples with human evaluations on S&P and correctness of $isRel$ and $isUp$ reflection tokens.

## C.3 QUALITATIVE EXAMPLES

Table 7 shows several examples predicted by our SELF-RAG (13B). The first example is the model output to an ASQA question. The first reference states that Emperor Constantine made Sunday a day of rest from labor, and further the second citation supports the fact that the official adoption of Sunday as a day of rest by Constantine in AD 321. In the second example, the model predicts $Contradictory$ to the first output as the output says the person has served as the CEO since 2010, while the passage says he stepped down as CEO in 2015. Indicating those factual contradictions as reflection tokens enables to enforcement of hard control and also verification of model outputs easily. In the third example, while the generation is mostly correct, SELF-RAG predicts $Partially$ incorrect to the statement listing the name of the songs, as they were not explicitly mentioned.

## D FULL LIST OF INSTRUCTIONS AND DEMONSTRATIONS FOR GPT-4

Here, we show the instructions and demonstrations used to prompt GPT-4 to collect reflection tokens. Table 8 shows the instructions and demonstrations for the initial retrieval token. Table 9 shows the instruction and demonstrations used to collect the three-way output tokens for $retrieve$ given instruction, preceding sentences, and previously retrieved passages. Due to the longer demonstration and test input, we only use a single demonstration. Table 10 shows an instruction and demonstrations used to collect the three-way output tokens for $isRel$. Table 11 shows an instruction and demonstrations used to collect the five-way output tokens for $issue$.
```


### --- Page 0022 ---

```markdown
# Preprint

## Identify the two types of computer memory. (from gpt4.alpaca)

**Output**  
**[Retrieve=Yes]**  
Computer memory Computer memory – In computing, memory refers to the computer hardware integrated circuits that store information for immediate use in a computer; it is synonymous with the term "primary storage". Computer memory operates at a high speed, for example random-access memory (RAM), as a distinction from storage that provides slow-to-access information but offers higher capacities. If needed, content from primary storage can be transferred to secondary storage; a very common way of doing this is through a memory management technique called "virtual memory". An archaic synonym for memory is store. The term "memory", meaning "primary storage" or "main memory", is  
**[iskp=Relevant]**  
Primary memory, also known as main memory, is the memory that the computer can directly access and manipulate.  
**[isur=Fully Supported]**  
**[Retrieve=Yes]**  
Computer memory—often associated with addressable semiconductor memory, i.e., integrated circuits consisting of silicon-based transistors, used for example as primary storage but also other purposes in computers and other digital electronic devices. There are two main kinds of semiconductor memory, volatile and non-volatile. Examples of non-volatile memory are flash memory (used as secondary memory) and ROM, PROM, EPROM and EEPROM memory (used for storing firmware such as BIOS). Examples of volatile memory are primary storage, which is typically dynamic random-access memory (DRAM), and fast CPU cache memory, which is typically static random-access memory (SRAM) that is fast but energy-consuming, offering lower memory  
**[iskp=Relevant]**  
Primary memory can be accessed quickly, but it is volatile, meaning that data is lost if the power is turned off.  
**[isur=Partially Supported]**  
**[ist=5]**

## Give an example of a situation where you have to delegate tasks. (from gpt4.alpaca)

**Output**  
**[Retrieve=No]**  
The manager realizes that there is an extensive list of tasks that need to be completed in order to make the event a success.  
**[Retrieve=Yes]**  
Delegation—Delegation is the assignment of any responsibility or authority to another person (normally from a manager to a subordinate) to carry out specific tasks. It is one of the core concepts of management leadership. However, the person who delegated the work remains accountable for the outcome of the delegated work. Delegation empowers a subordinate to make decisions, i.e., it is a shifting of decision-making authority from one organizational level to a lower one. Delegation, if properly done, is not fabrication. The opposite of effective delegation is micromanagement, where a manager provides too much input,  
**[iskp=Relevant]**  
This includes tasks such as creating a guest list, designing and sending invitations, and coordinating with vendors.  
**[isur=Partially Supported]**  
**[Retrieve=No]**  
Since it is not possible for the manager to handle everything alone, they must delegate tasks to the team members.  
**[Retrieve=No]**  
The manager would then assign each team member specific responsibilities based on their individual skills and strengths.  
**[Retrieve=Yes]**  
In this application design – Subject Matter Experts: These are the business users, the IS professionals, and the outside experts that will be needed for a successful workshop. This group is the backbone of the meeting.  
**[iskp=Relevant]**  
For instance, the team's graphic designer might be tasked with designing the invitations, while the event coordinator would be in charge of planning the program and working with vendors.  
**[Retrieve=No]**  
By delegating tasks, the manager can ensure that all necessary tasks are completed in a timely manner, while empowering their team to take on responsibilities and contribute to the success of the event.  
**[ist=5]**

## Synthesize a poem based on this prompt: "The sun is setting" (from stanford.alpaca)

**Output**  
**[Retrieve=No]**  
The stars begin to twinkle bright the trees, scattered throughout the night;  
**[Retrieve=No]**  
They stars begin to twinkle bright the trees, scattered throughout the night.  
**[Retrieve=No]**  
They flicker like a silent dance, A peaceful moment in its trance.  
**[Retrieve=No]**  
A chorus of crickets sing a lullaby, To every creature that roams the sky.  
**[ist=5]**

## When did walking dead season 7 come out (from no)

**Output**  
**[Retrieve=Yes]**  
The Walking Dead (season 7) – The seventh season of "The Walking Dead", an American post-apocalyptic horror television series on AMC, premiered on October 23, 2016, and concluded on April 2, 2017, consisting of 16 episodes. Developed for television by Frank Darabont, the series is based on the eponymous series of comic books by Robert Kirkman, Tony Moore, and Charlie Adlard.  
**[iskp=Relevant]**  
October 23, 2016  
**[isur=Fully Supported]**  
**[ist=5]**

## Given a chat history separated by new lines, generates informative, knowledgeable and engaging response: Hello, I love German shepherd dogs and you? – My favorite dog, they are officially known as German Shepherd Dogs? – Yes, I have a German shepherd in my house and she is pregnant (from row)

**Output**  
**[Retrieve=Yes]**  
German Shepherd – a simple difference of opinion, it is the fundamental issue of the breed's exceptional conformation and movement. … They are known for being easy to train and good for performing tasks and following instructions. They are especially well known for their police work, being used for tracking criminals, patrolling troubled areas and detecting and holding suspects.  
**[iskp=Relevant]**  
Because of this, they are often used for disability assistance and for police work.  
**[isur=Partially Supported]**  
**[ist=2]**

| Table 4: List of the training examples. |
|------------------------------------------|
| 22                                       |
```

### --- Page 0023 ---

```markdown
# Preprint

| Dataset         | Instruction                                                                                                                                                                                                                     |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ARC-C            | Given four answer candidates, A, B, C and D, choose the best answer choice. Please answer with the capitalized alphabet only, without adding any extra phrase or period.                                                        |
| PubHealth        | Is the following statement correct or not? Say true if it’s correct; otherwise, say false. Don’t capitalize or add periods, just say “true” or “false”.                                                                         |
| Bio Generation   | Tell me a bio about [Person Name].                                                                                                                                                                                            |
| ASQA (baseline)  | Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. |
| ASQA (ours)      | Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.                                          |

Table 5: Full list of instructions used during zero-shot evaluations. For open-domain QA, we don’t use any task specific instruction and simply use the original questions as input query.
```

### --- Page 0024 ---

```markdown
# Preprint

| Input | What genre is A Question and Answer Guide to Astronomy? (PopQA) |
|-------|------------------------------------------------------------------|
| Output | [Retrieve-Yes] <p>Reference [1]<p>[IsRel] = Relevant] A Question and Answer Guide to Astronomy is a non-fiction book that falls under the genre of Astronomy and Cosmology. | [IsSup] = Fully Supported | [IsL] = 5 |
| Reference | [1] A Question and Answer Guide to Astronomy is a book about astronomy and cosmology, and is intended for a general audience. The book was written by Pierre-Yves Bely, Carol Christian, and Jean-Rene Roy, and published in English by Cambridge University Press in 2010. It was originally written in French. The content within the book is written using a question and answer format. It contains some 250 questions, which The Science Teacher states each are answered with a concise and well-formulated essay that is informative and readable. The Science Teacher review goes on to state that many of the answers given in the book are little gems of science writing. The Science Teacher summarizes by stating that each question is likely to be thought of by a student, and that the answers are informative, well constructed, and thorough. The book covers information about the planets, the Earth, the Universe, practical astronomy, history, and awkward questions such as astronomy in the Bible, UFOs, and aliens. Also covered are subjects such as the Big Bang, comprehension of large numbers, and the Moon illusion. |
| S&P | Supported and Plausible. |
| Correctness of [IsRel] and [IsSup]: | Both [IsRel] and [IsSup] are correct, as the Reference [1] provides detailed information of the book including the genre, and the model output is fully supported by the reference. |

| Input | Who is the author of The Lie? (PopQA) |
|-------|----------------------------------------|
| Output | [Retrieve-Yes] <p>Reference [1]<p>[IsRel] = Relevant] The author of The Lie is Sam Harris. | [IsSup] = Fully Supported | [IsL] = 5 |
| Reference | [1] Lying is a 2011 long-form essay book by American author and neuroscience expert Sam Harris. Harris argues that we can radically simplify our lives and improve society by merely telling the truth in situations where others often lie. |
| S&P | Plausible but not entirely supported. |
| Correctness of [IsRel] and [IsSup]: | Both [IsRel] and [IsSup] are incorrect, as the Reference [1] is about a different book whose title is similar to The Lie. |

| Input | Tell me a bio about Ronaldinho (Bio) |
|-------|---------------------------------------|
| Output | [Retrieve-Yes] <p>Reference [1]<p>[IsRel] = Relevant] Ronaldinho is a retired professional footballer from Brazil. | [IsSup] = Fully Supported | [Retrieve-Continue] | [IsL] = 5 |
| Reference | [1] Ronaldo de Assis Moreira (born 21 March 1980), commonly known as Ronaldinho Gaúcho or simply Ronaldinho, is a Brazilian retired professional footballer who played mostly as an attacking midfielder, but was also deployed as a winger. Considered one of the best players of his generation and regarded by many as one of the greatest of all time, Ronaldinho won two FIFA World Player of the Year awards and a Ballon d'Or. A global icon of the sport, he was renowned for his technical skills, creativity, dribbling ability, and accuracy from free-kicks, as well as his use of tricks, feints, no-look passes and overhead... |
| S&P | Plausible and fully supported. |
| Correctness of [IsRel] and [IsSup]: | Both [IsRel] and [IsSup] are correct. |

| Input | Tell me a bio about Sergei Bodrov (Bio) |
|-------|---------------------------------------|
| Output | [Retrieve-Yes] <p>Reference [1]<p>[IsRel] = Relevant] Sergei Vladimirovich Bodrov (born June 28, 1948) is a Russian film director, screenwriter, and producer. | [IsSup] = Fully Supported | [Retrieve-Continue] | [IsL] = 5 |
| Reference | [1] Sergei Vladimirovich Bodrov (born June 28, 1948) is a Russian film director, screenwriter, and producer. In 2003 he was the President of the Jury at the 25th Moscow International Film Festival. |
| S&P | Plausible but only partially supported. The Reference [1] does not explicitly mention the person is born in Moscow and there is no description about the Russian Guild of Film Producers. |
| Correctness of [IsRel] and [IsSup]: | Both [IsRel] and [IsSup] are correct as the reference document is highly relevant and the first sentence is fully supported. |

| Table 6: | Examples of human evaluation results. |
```

### --- Page 0025 ---

```markdown
# Preprint

| Input                                                                 | Output                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| When did Sunday become the seventh day of the week (asqsa)          | Output  | [Retrieve=Yes] <p>Reference [1]<p>[IsRel=Relevant] In 321 AD, the Roman Emperor Constantine the Great made Sunday a day of rest from labor, stating: "On the venerable Day of the Sun let the magistrates and people residing in cities rest, and let all workshops be closed" [1][Isuf=Fully Supported] | [Retrieve=Yes] <p>Reference [2]<p>[IsRel=Relevant] The official adoption of Sunday as a day of rest by Constantine the Great in AD 321 made Sunday the seventh day of the week.[2][Isuf=Fully Supported] <p>Reference [1] Emperor Constantine enacted the first Sunday Laws, for "the venerable Day of the Sun" in 321 AD. On March 7, 321, the Roman Emperor Constantine issued a decree making Sunday a day of rest from labor, stating: We find a good description of this in sources such as the book Rest Days which states: Early Christian observance of both the spiritual seventh-day sabbath and a Lord's Day assembly is evidenced in Ignatius letter to the Magnesians ca. 110. <p>Reference [2] Despite the official adoption of Sunday as a day of rest by Constantine, the seven-day week and the rudimentary cycle continued to be used side-by-side until at least the Calendar of 354 and probably later. ... The fact that the canon had to be issued at all is an indication that adoption of Constantine's decree of 321 was still not universal ... |
| Tell me a bio about Rory Tapner. (bio generation)                    | Output  | <p>Reference [1]<p>[IsRel=Relevant] Rory Tapner is a British businessman who has served as the Chief Executive Officer of Coutts, a private banking and wealth management company, since 2010.[1][Isuf=Contradictory]. Reference [2]<p>[IsRel=Relevant] Rory Tapner was born on 30 September 1959 in London, England.[1][Isuf=Fully Supported] <p>Reference [1] Rory Tapner is a UK businessman who, in September 2010 was appointed CEO of Coutts, the Wealth Management division of Royal Bank of Scotland Group. In February 2015 he stepped down as CEO when he was replaced in the role by Alison Rose. <p>Reference [2] Holdings (from August 2016); and the Council of the University of Buckingham (from July 2017). Rory Tapner Rory Tapner (born 30 September 1959) is a UK businessman ... |
| Tell me a bio about G. Venugopal. (bio generation)                   | Output  | [Retrieve=Yes] <p>Reference [1]<p>[IsRel=Relevant] G. Venugopal is a popular playback singer in the Malayalam film industry. [Retrieve=Continue] He has sung over 300 songs in Malayalam, Tamil, and Hindi movies.[1][Isuf=Partially Supported][Retrieve=Yes] <p>Reference [2]<p>G. Venugopal was born on December 10, 1960 in India.[1][Isuf=fully supported] <p>Reference [1] G. Venugopal G Venugopal (born 10 December 1960) is an Indian playback singer known for his work in Malayalam films. He started his singing career in the film "Oruathmanmaava Aalayiram" (1984). Since then he has sung in more than 300 films; and has over 500 private albums to his credit. The many awards he has won include the coveted Best Playback Singer (Kerala State) which he won thrice, the Film Critics award and the Kerala Kaumudi Gallu Poll award (twice). He won The Kerala State award for Best Playback singer (2004) third time for the song "aadecede...aadeadcede" from the film "Ullam". Venugopal <p>Reference [2] Kerala State Film Awards: Kerala Film Critics Award Kerala State Government Award (Professional Drama): Asianet Film Awards: Kerala Kaumudi Gallu Poll Award: Academic G. Venugopal G Venugopal (born 10 December 1960) is an Indian playback singer known for his work in Malayalam films. |
| Table 7: Examples of outputs.                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
```

### --- Page 0026 ---

```markdown
| Instructions                                                                                                                             |
|-----------------------------------------------------------------------------------------------------------------------------------------|
| Given an instruction, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response. Please answer [Yes] or [No] and write an explanation. |

| Demonstrations                                                                                                                         | Need retrieval? | Explanation                                                                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Give three tips for staying healthy.                                                                                                   | [Yes]            | There might be some online sources listing three tips for staying healthy or some reliable sources to explain the effects of different behaviors on health. So retrieving documents is helpful to improve the response to this query. |
| Describe a time when you had to make a difficult decision.                                                                             | [No]             | This instruction is asking about some personal experience and thus it does not require one to find some external documents.                                                                                 |
| Write a short story in third person narration about a protagonist who has to make an important career decision.                        | [No]             | This instruction asks us to write a short story, which does not require external evidence to verify.                                                                                                     |
| What is the capital of France?                                                                                                         | [Yes]            | While the instruction simply asks us to answer the capital of France, which is a widely known fact, retrieving web documents for this question can still help.                                             |
| Find the area of a circle given its radius. Radius = 4                                                                                 | [No]             | This is a math question and although we may be able to find some documents describing a formula, it is unlikely to find a document exactly mentioning the answer.                                          |
| Arrange the words in the given sentence to form a grammatically correct sentence. quickly the brown fox jumped                          | [No]             | This task doesn’t require any external evidence, as it is a simple grammatical question.                                                                                                                 |
| Explain the process of cellular respiration in plants.                                                                                 | [Yes]            | This instruction asks for a detailed description of a scientific concept, and is highly likely that we can find a reliable and useful document to support the response.                                     |

Table 8: Instructions and demonstrations for Retrieve aspect given the input only.
```

### --- Page 0027 ---

```markdown
# Preprint

## Instructions
You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:
- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence].
- If the sentence doesn’t require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with [No Retrieval].
- If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.

### Instruction
Explain the use of word embeddings in Natural Language Processing.

**Preceding sentences**  
Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.

**Evidence:**  
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. 

**Output:**  
Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.

**Rating**  
[Retrieval]

**Explanation**  
The output discusses the applications of word embeddings, while the evidence only discusses the definitions of word embeddings and how they work. Therefore, we need to retrieve other evidence to verify whether the output is correct or not.

---

| Table 9: Instructions and demonstrations for [Retrieve] aspect given the input, preceding generations, and retrieved passages. |
```

### --- Page 0028 ---

```markdown
# Preprint

## Instructions
You’ll be provided with an instruction, along with evidence and possibly some preceding sentences. When there are preceding sentences, your focus should be on the sentence that comes after them. Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].

| Instruction | Given four answer options, A, B, C, and D, choose the best answer. |
|-------------|---------------------------------------------------------------------|
| Input       | Earth’s rotating causes <br> A: the cycling of AM and PM <br> B: the creation of volcanic eruptions <br> C: the cycling of the tides <br> D: the creation of gravity |
| Evidence    | Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates. |
| Rating      | [Relevant] |
| Explanation | The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A. |

| Instruction | age to run for US House of Representatives |
|-------------|-------------------------------------------|
| Evidence    | The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a senator represents at the time of election. |
| Rating      | [Irrelevant] |
| Explanation | The evidence only discusses the ages to run for the US Senate, not for the House of Representatives. |

Table 10: Instructions and demonstrations for the |isRRL| aspect given the input only.
```

### --- Page 0029 ---

```markdown
# Preprint

## Instructions
You will receive an instruction, evidence, and output, and optional preceding sentences. If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence.  
Use the following entailment scale to generate a score:
- **[Fully supported]** - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.
- **[Partially supported]** - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].
- **[No support / Contradictory]** - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.

Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.

## Instruction
Explain the use of word embeddings in Natural Language Processing.

### Preceding sentences
Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.

### Output
Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.

### Evidence
Word embedding  
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. Methods to generate this mapping include neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear. Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing, sentiment analysis, next token predictions as well as analogy detection.

### Score
[Fully supported]

### Explanation
The output sentence discusses the application of word embeddings, and the evidence mentions all of the applications: syntactic parsing, sentiment analysis, next token predictions as well as analogy detection as the applications. Therefore, the score should be [Fully supported].

| Table 11: Instructions and demonstrations for \texttt{IssSup} tokens. |
|:-----------------------------------------------------------------------:|
| 29                                                                    |
```

### --- Page 0030 ---

```markdown
| Instructions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility. The detailed criterion is as follows: 5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs. 4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. 3: The response is acceptable, but some major additions or improvements are needed to satisfy users’ needs. 2: The response still addresses the main request, but it is not complete or not relevant to the query. 1: The response is barely on-topic or completely irrelevant. |

| Instruction                                                                                                                                                                                                                     | Output                                                                                          | Perceived utility | Explanation                                                                                                                                                                                                                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Who is the current prime minister of the UK as of 2023?                                                                                                                                                                     | Boris Johnson was the prime minister of the UK from 2019 - 2022.                             | 2                 | While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn’t answer the instruction. Therefore, the utility is 2.                                                                                                                                  |
| Given a description of a travel destination, recommend 10 tourist attractions with detailed descriptions of each. The travel destination is Tokyo, Japan. | Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen. | 3                 | This output doesn’t provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn’t match the instructions strictly.                                                                                                                                               |

| Table 12: Instructions and demonstrations for [1Use] tokens.                                                                                                                                                                                                                                           |
```


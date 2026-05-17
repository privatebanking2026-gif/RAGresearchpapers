# ArXiv 2402.10612

### --- Page 0001 ---

```markdown
# Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs

**Hanxing Ding**  
dinghanxing18s@ict.ac.cn  
State Key Laboratory of AI Safety,  
Institute of Computing Technology,  
Chinese Academy of Sciences  
Beijing, China  

**Liang Pang**  
pangliang@ict.ac.cn  
State Key Laboratory of AI Safety,  
Institute of Computing Technology,  
Chinese Academy of Sciences  
Beijing, China  

**Zihao Wei**  
State Key Laboratory of AI Safety,  
Institute of Computing Technology,  
Chinese Academy of Sciences  
Beijing, China  

**Huawei Shen**  
State Key Laboratory of AI Safety,  
Institute of Computing Technology,  
Chinese Academy of Sciences  
Beijing, China  

**Xueqi Cheng**  
State Key Laboratory of AI Safety,  
Institute of Computing Technology,  
Chinese Academy of Sciences  
Beijing, China  

---

**Abstract**  
Hallucinations present a significant challenge for large language models (LLMs). The utilization of parametric knowledge in generating factual content is constrained by the limited knowledge of LLMs, potentially resulting in internal hallucinations. While incorporating external information can help fill knowledge gaps, it also introduces the risk of irrelevant information, thereby increasing the likelihood of external hallucinations. To balance the use of parametric knowledge within LLMs and external information, in this study, we present Rowen, a novel framework that enhances LLMs with an adaptive retrieval augmentation process tailored to address hallucination detection. Rowen introduces a consistency-based hallucination detection module, which assesses the model's uncertainty regarding the input query by evaluating the semantic inconsistencies in various responses generated across different languages or models. When high uncertainties in the responses are detected, Rowen activates the retrieval of external information to rectify the model outputs. Through comprehensive empirical experiments, we demonstrate that Rowen surpasses the current state-of-the-art in both detecting and mitigating hallucination content within the outputs of LLMs.

**CCS Concepts**  
• Information systems → Question answering.

**Keywords**  
Large Language Models, Hallucination Mitigation, Self-Consistency Check

*Corresponding author*  
*Our Code: https://github.com/dhx20150812/Rowen.*

This work is licensed under a Creative Commons Attribution 4.0 International License.  
SIGIR-AP 2025, Xi'an, China, © 2023 Copyright held by the owner/author(s).  
ACM ISBN 978-1-4503-2878-9/25/12  
https://doi.org/10.1145/3767695.3769500

---

## 1. Introduction

In recent years, large language models (LLMs) have demonstrated impressive abilities in natural language understanding [9, 11, 21, 31, 33], and reasoning [4, 36, 47]. Despite their successes, it has been widely observed that even state-of-the-art LLMs can generate factually incorrect or nonsensical outputs, referred to as hallucinations [14, 44, 46]. These unreliable outputs pose significant risks in practical deployments of LLMs.

Efforts to enhance the factual accuracy of LLM outputs have been substantial. These studies often utilize LLMs' extensive parametric knowledge and advanced logical reasoning capabilities. They employ approaches like self-reflection [6, 15, 23, 35] or collaborative refinements involving interactions among multiple models [5, 7], aiming to enhance logical coherence in refined content. Despite their effectiveness, these self-improvement methods may be limited by LLMs' knowledge boundaries [20, 28] or may not fully exploit parametric knowledge [10], leading to what we term internal hallucination, illustrated in Figure 1.

Alongside these self-improvement strategies, Retrieval-Augmented Generation (RAG) [3] serves as a complementary method to overcome knowledge limitations. RAG employs a retrieve-then-read knowledge sources into the LLMs' generation process [3, 21, 34, 41]. However, as depicted in Figure 1, RAG methods are susceptible to external hallucination when irrelevant evidence is incorporated, potentially leading to cumulative errors and compromising output accuracy [20, 29].

Drawing inspiration from the latest neuroscience research [27], which reveals how the human brain dynamically switches between internal thoughts and external sensations, we introduce a novel framework...
```
![Detailed description of the chart](assets/page_0001_img_1.png)


### --- Page 0002 ---

```markdown
# SIGIR-AP 2025, December 7 - 10, 2025, Xi'an, China

## 2 Related Works

In this section, we discuss recent works on hallucination detection and mitigation, focusing on uncertainty estimation methods for detection and post-hoc correction for mitigation.

### 2.1 Exploring Uncertainty for Hallucination Detection

Uncertainty refers to the confidence level of the model outputs, and it serves as an important indicator for identifying and eliminating hallucinations, so it can assist users in determining when to trust LLMs. In general, methods for exploring uncertainty for hallucination detection can be categorized into three types: (1) Logit-based estimation relies on accessing the model’s logits to calculate token-level probabilities or entropy, which are used to measure uncertainty [12, 25, 34]. However, this approach can pose challenges for black-box closed-source models. (2) Verbalized-based estimations involve prompting language models to express their uncertainty using specific prompts [1, 32, 38]. However, these methods tend to display a high degree of overconfidence when expressing their confidence [30, 38]. (3) To overcome these limitations, consistency-based estimations are adopted to measure the consistency score among multiple responses provided by the model for a given question [25, 35, 38, 41]. The underlying assumption suggests that when language models struggle with indecision and fabricate facts, they tend to provide logically inconsistent responses to identical questions. Therefore, they can serve as cross-language and cross-model detection modules that cross-check answers to the same questions across different languages or models. This cross-checking paradigm serves as a powerful mechanism to identify hallucinations in LLMs.

### 2.2 Post-Hoc Correction for Hallucination Mitigation

Mitigating hallucinations in the inference time could be a cost-effective and controllable way. A line of research harnesses the extensive parametric knowledge and robust logical reasoning capabilities of LLMs to ensure logical consistency either through self-reflection within a single model [6, 15, 23, 35, 39] or through collaborative refinements or debates involving multiple models [5, 7]. Despite their strengths, LLMs are sometimes constrained by their knowledge boundaries and the complexity of the reasoning chain, resulting in occasional inaccuracies [20, 24, 28] termed internal hallucination. To address this knowledge gap, retrieval-augmented generation methods leverage external knowledge as supplementary evidence to aid LLMs in providing accurate responses [3, 10, 21, 26, 34, 40, 41]. However, these approaches, while effective, can occasionally encounter the challenge of error accumulation, where irrelevant evidence may seep into the generation process, leading to incorrect responses [20, 29], a phenomenon referred to as external hallucination. Our work only performs retrieval augmentation when hallucinations are detected, thereby maximizing the utilization of both the parametric knowledge and externally retrieved information.

![Figure 1: The limited knowledge of LLMs poses a challenge for generating accurate answers, referred to as Internal Hallucination. Additionally, retrieval-augmented generation occasionally faces the risk of error accumulation, where irrelevant evidence may infiltrate the generation phase and lead to nonfactual responses, known as External Hallucination.](assets/page_0002_img_1.png)
```

### --- Page 0003 ---

```markdown
# Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs

There are also some adaptive retrieval methods that assess the difficulty of questions or the confidence in responses to decide whether to retrieve documents [2, 13, 16, 32].

## 3 Methodology

Our objective is to enhance the factuality of LLM responses by integrating parametric and external knowledge. We propose a framework called Rowen (Retrieve only when needed). Initially, we leverage LLMs' Chain-of-Thought (CoT) reasoning to generate an initial response ($\S$ 3.1). To mitigate internal hallucinations, Rowen employs a consistency-based hallucination detection module that assesses the reliability of the initial response ($\S$ 3.2). If high uncertainties are found, the initial answer is refined using external information via retrieval augmentation ($\S$ 3.3), resulting in the final response. Otherwise, the initial response is considered the final output. For external hallucinations, Rowen resorts to external knowledge only when uncertainties are found, ensuring that the final answer is both accurate and reliable.

### 3.1 Stage 1: Generating Initial Answer

To maximize the exploitation of the parametric knowledge in LLMs, we initially employ their Chain-of-Thought (CoT) reasoning to generate a preliminary response. This process involves: critically assessing the validity of the information in the input query $x_0$ and prioritizing accuracy and fact-checking before diving into elaboration, detailed in Table 1. After generating the CoT thought, we ask M to provide a concise answer $r_0$ for the input query $x_0$. Our aim is to ensure high-quality responses are generated from the outset. The answer $r_0$ is finalized as the ultimate response to the input query $x_0$ after our detection module ensures it is free from hallucinations.

### 3.2 Stage 2: Deciding Whether to Retrieve

To decide when to retrieve, we leverage model uncertainty, which refers to the confidence level of model outputs and serves as a crucial indicator for deciding when to trust LLMs [16]. Unfortunately, current consistency-based methods fail when LLMs provide consistent yet incorrect answers across different perturbations [43]. This issue may arise because these methods focus exclusively on semantic coherence within a single language or model.

To tackle this issue, we propose novel cross-language and cross-model detection methods that assess semantic consistency among responses for the same question across different languages or models. If inconsistencies are detected in these responses, we flag them as potentially inaccurate and invoke a retrieval process.

### 3.2.1 Cross-Language / Model Perturbations

To facilitate subsequent consistency-based hallucination detection, we begin by leveraging advancements in LLM prompting to generate semantically equivalent perturbations. Initially, we start with an input query and instruct the model M to provide a set of semantically equivalent questions $X = \{x_1, x_2, \ldots, x_k\}$. We use the prompt: "For the question [ORIGINAL QUESTION], please provide k semantically equivalent questions" with a high decoding temperature to generate diverse perturbed expressions.

After obtaining the diverse verbalized questions, we prompt the LLM M to generate its candidate responses according to the questions. We employ a greedy decoding strategy to avoid unpredictable randomness of the LLM as much as possible.

$$
r_j = M(x_j), \quad j = 1, \ldots, k,
$$

where $k$ is the length of the generated semantically equivalent questions $X$.

### 3.2.2 Cross-Language Detection

To capture language-level uncertainty, we first incorporate a cross-language consistency check (Rowen-CL) to evaluate the semantic consistency of responses to the same question across different languages. To achieve this, we introduce language-level perturbations by asking the model M to translate the source-language questions $X$ into corresponding paraphrased questions $X_T = \{x_{T1}, x_{T2}, \ldots, x_{Tk}\}$ in the target language. The model M is then asked to generate corresponding answers for each question in the target language.

$$
r_{Tj} = M(x_{Tj}), \quad j = 1, \ldots, k.
$$

### 3.2.3 Cross-Model Detection

Besides language-level cross-checking, we also introduce a model-detection module (Rowen-CM) to evaluate the semantic consistency of responses to the same question across different models. We adopt an additional verifier M_V to provide answers for each source question:

$$
r_{j}^V = M_V(y_j), \quad j = 1, \ldots, k.
$$

### 3.2.4 Consistency Score Calculation

In this step, we utilize the generated questions and answers from all previous stages to calculate a numerical consistency score that captures language-level and model-level cross-checking uncertainties.

| Table 2 | The instruction for determining whether two QA pairs in different languages are semantically equivalent. |
|---------|-------------------------------------------------------------------------------------------------------------|
| Given the question Q, and two potential answers: answer A in English and answer B in Chinese. Your task is to determine if the direct meaning of A and B are equivalent in the context of answering Q. Consider linguistic nuances, cultural variations, and the overall conveyance of information. Respond with a binary classification. If A and B are equivalent, output 'True', otherwise output 'False'. |
```

### --- Page 0004 ---

```markdown
![Overview of Rowen framework. It illustrates the process of generating an initial response, checking consistency, and retrieving evidence.](assets/page_0004_img_1.png)

## Rowen

### Stage 1: Generating Initial Answer
- CoT Process $G$
- Initial Answer $r_0$

### Stage 2: Deciding Whether to Retrieve
- **Consistency-based Detection Module**
  - Cross-language questions $X_f$ → responses $r_f$
  - Perturbed questions $X$ → responses $r_p$
  - Cross-model questions $X_y$ → responses $r_y$
  - Semantically Equivalent Perturbations

### Stage 3: Retrieval Augmented Generation
- Perturbed Questions $X^*$ → Retrieved Evidences $E$

---

### 3.1 Cross-Language Consistency Score
Let $C(·,·)$ denote a semantic equivalence checking operator that takes two QA pairs as inputs and returns "True" if they are semantically equivalent, and "False" otherwise. We then map the response to a numerical semantic equivalent score: ("True" $\rightarrow$ 1.0, "False" $\rightarrow$ 0.0). In our implementation, we leverage the model $M$ and utilize the prompt in Table 2 to implement the cross-language checking operator and calculate the cross-language consistency score $Z_{CL}$ as:

$$
Z_{CL} = \frac{1}{k} \sum_{j=1}^{k} C(p_i, p_j^t) \tag{4}
$$

where $p_i$ and $p_j^t$ denote the QA pairs in the source language and target language, respectively.

### Table 3: The instruction for determining whether two QA pairs generated by different models are semantically equivalent.

### 3.2 Cross-Model Consistency Score
Similar to the cross-language consistency score calculation, we use the prompt in Table 3 to implement the checking operator $C$ to calculate cross-model consistency score:

$$
Z_{CM} = \frac{1}{k} \sum_{j=1}^{k} C(p_i, p_j^v) \tag{5}
$$

where $p^v$ denote the QA pairs generated by verifier model $M_V$.

### Hybrid Consistency Score
The different variants of Rowen capture various aspects of uncertainty in the original response, complementing each other effectively. We propose integrating the cross-language and cross-model consistency scores to create a unified hybrid consistency score:

$$
Z_{Hybrid} = Z_{CL} + \alpha \cdot Z_{CM} \tag{6}
$$

where $\alpha$ is a weight factor for the cross-model consistency score.

### 3.3 Stage 3: Retrieval Augmented Generation
If the consistency score $Z$ falls below a threshold, it indicates possible hallucinatory content in the original response $r_0$. We then introduce a retrieval-augmented generation procedure.

#### Searching Relevant Knowledge
To help the LM correct errors, we search for supporting evidence from external sources like online webpages. We first ask the model $M$ to generate search queries for each paraphrased question in $X$. These queries are input into the online search engine to retrieve relevant knowledge, denoted as $E$, used for correcting factual errors in $r_0$.

#### Repairing Hallucinated Contents
With the retrieved evidence $E$, the model reviews the original thought process $f_0$ and initial answer $r_0$. The aim is to identify and correct inaccuracies, producing the refined answer $r_c$:

$$
r_c = M(x_0, f_0, E) \tag{7}
$$

The corrected answer $r_c$ serves as the final response to question $x_0$.
```

### --- Page 0005 ---

```markdown
# Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs

| Models                       | TruthfulQA                     | StrategyQA                     |
|------------------------------|--------------------------------|--------------------------------|
|                              | GPT-Judge ↑ | BLEU ↑ | Rouge-L ↑ | Accuracy ↑ | GPT-Judge ↑ | BLEU ↑ | Rouge-L ↑ | Accuracy ↑ |
| **Vanilla LLMs**            |              |         |           |            |              |         |           |            |
| ChatGPT (gpt-3.5-turbo)     | 47.92        | 10.17   | 31.31     | 61.40      |              |         |           |            |
| **Self-improvement Methods** |              |         |           |            |              |         |           |            |
| CoVe [6]                    | 48.01        | 12.81   | 26.52     | 61.40      |              |         |           |            |
| Multi-agent Debate [7]      | 50.83        | 3.94    | 21.05     | 65.73      |              |         |           |            |
| Self-Reflection [15]        | 42.99        | 3.86    | 18.18     | 62.40      |              |         |           |            |
| **Retrieval-augmented Methods** |          |         |           |            |              |         |           |            |
| Factool [3]                 | 34.50        | 1.34    | 12.22     | 67.20      |              |         |           |            |
| Detect-and-Mitigate [34]    | 49.98        | 3.17    | 18.59     | 56.94      |              |         |           |            |
| **Adaptive Retrieval Methods** |            |         |           |            |              |         |           |            |
| FLARE [16]                  | 45.04        | 11.59   | 26.83     | 61.19      |              |         |           |            |
| Adaptive-Retrieval [24]     | 45.55        | 8.87    | 26.75     | 62.50      |              |         |           |            |
| Self-RAG [2]                | 40.36        | 4.36    | 21.28     | 58.40      |              |         |           |            |
| Adaptive-RAG [13]           | 46.02        | 10.29   | 26.24     | 58.50      |              |         |           |            |
| LUQ [42]                    | 55.08        | 5.79    | 21.44     | 71.00      |              |         |           |            |
| **Our Framework**           |              |         |           |            |              |         |           |            |
| Rowen-CL                    | 57.39        | 7.60    | 24.16     | 74.00      |              |         |           |            |
| Rowen-CM                    | 56.29        | 6.85    | 22.36     | 72.40      |              |         |           |            |
| Rowen-Hybrid                | 59.34        | 15.27   | 31.15     | 75.60      |              |         |           |            |

Table 4: Experimental results of mitigating hallucinations on the TruthfulQA dataset and StrategyQA dataset. Rowen-Hybrid achieves a detection accuracy of 59.0% on the TruthfulQA dataset and 73.0% on the StrategyQA dataset.

## 4 Experimental Setup
In this section, we outline the experimental setup of Rowen, detailing the datasets utilized, the evaluation metrics employed, the baseline methods considered, and the implementation specifics.

### 4.1 Datasets and Evaluation Metrics
We evaluate the hallucination mitigation performance across two challenging datasets:

- **TruthfulQA**: We use the TruthfulQA dataset [22] to evaluate the ability of LLMs in generating truthful responses [17, 45]. In our study, we focus on the generation task in TruthfulQA. Therefore, to evaluate the factuality of responses from LLMs, we calculate the GPT-Judge score, obtained by fine-tuning the babbage-002 model using the original fine-tuning data from their official repository\footnote{https://github.com/siyu1/TruthfulQA}. We also report the BLEU and Rouge-L scores to evaluate the lexical overlap between generated responses and ground-truth references.

- **StrategyQA**: The StrategyQA dataset [8] comprises crowdsourced yes / no questions that require multi-step reasoning for accurate answers. We follow previous work [16] to randomly sample 500 examples due to the cost consideration of running experiments. We also follow the settings of Wei et al. [37] to generate both the reasoning process as well as the final answer. We present the exact-match accuracy of the generated yes / no answers compared to the gold-standard answers.

### 4.2 Baseline Methods
We consider the following methods as our baselines: (1) Vanilla LLMs, such as ChatGPT. (2) Self-improvement methods: CoVe [6] leverages verification questions to self-analyze potential errors, systematically addressing each question to refine the baseline response. Self-Reflection [15] presents an iterative self-reflection methodology that incorporates knowledge acquisition and answer generation. Multi-agent Debate [7] utilizes multiple LMs against to debate their individual responses over multiple rounds to arrive at a common final answer. (3) Retrieval-augmented methods: Factool [3] leverages various tools to gather evidence about the factuality of the generated content. Detect and Mitigate [34] actively detects hallucinations during generation by identifying potential hallucination through the logit output values of LLMs. (4) Adaptive retrieval methods: FLARE [16] adopts an active retrieval strategy that only retrieves when LLMs generate low probability tokens. Adaptive-Retrieval [24] only retrieves when necessary based on a pre-defined threshold for entity popularity. Self-RAG [2] trains a single arbitrary LM that adaptively retrieves passages on-demand. Adaptive-RAG [13] trains a query-complexity classifier to decide when to retrieve based on question complexity. LUQ [42] is a sampling-based uncertainty quantification for tokens.

### 4.3 Implementation Details
LLMs. In our experiments, we validate Rowen using the GPT-3.5 language model, specifically the gpt-3.5-turbo-0613 version\footnote{https://api.openai.com/v1/chat/completions}.
```

### --- Page 0006 ---

```markdown
# SIGIR-AP 2025, December 7–10, 2025, Xi'an, China

We re-implement all baselines, except Self-RAG, using GPT-3.5 to ensure a fair comparison. For semantic perturbations, we configure the temperature to 1.0 to generate diverse perturbed expressions. Otherwise, the temperature is set to 0.0 to obtain high-quality deterministic outputs. Considering the diversity of expressions and the lateracy in generating perturbations, we produce $k = 6$ semantically equivalent questions. For the cross-language detection module, English serves as the source language while Chinese is employed as the target language. *This decision is made considering the substantial cultural disparities between the two languages, which can enhance the model’s capability to detect semantic inconsistencies when responding to identical questions. For the cross-model detection module, we adopt *Open-Max* 4248*, a large instruction-tuned model for chat service, as the verifier LM. We execute all experiments on a NVIDIA A800 80G GPUs.

## Retrieval Module
Following Chen et al. [3], we utilize the Google Search API offered by Serper* to fetch web pages and extract the most pertinent snippets from the API’s response. Subsequently, we answer the response to acquire various types of snippets, including answer boxes, knowledge graphs, and organic search results.

## 5 Experimental Results
In this section, we evaluate the effectiveness of Rowan for hallucination mitigation. In particular, we aim to answer the following research questions:

- RQ1: How does Rowan perform compared to existing strong methods for hallucination mitigation?
- RQ2: Is the proposed multilingual detection module superior to other hallucination detection methods?
- RQ3: How well does Rowan generalize to new datasets and models?
- RQ4: How do different hyperparameters affect the performance of Rowan?
- RQ5: Does Rowan eliminate both types of hallucinations as expected?
- RQ6: Does Rowan reduce unnecessary retrieval calls?

### 5.1 Main Results (RQ1)
We evaluate the effectiveness of Rowan on the TruthfulQA and StrategyQA datasets. Table 4 presents the overall performance of Rowan compared to several strong baselines. Rowan demonstrates superior performance on both datasets, with higher GFT-Judge score and accuracy, indicating the effectiveness of our proposed method.

Vanilla ChatGPT shows a certain level of accuracy in answering factual questions, achieving scores of 47.92% and 61.60% on the two datasets, respectively. While self-improvement methods perform better than the vanilla LM on both datasets, they are still limited by their knowledge boundaries and suffer from internal hallucinations. RAG methods demonstrate relatively better performance compared to self-improvement methods, highlighting the benefits of integrating external knowledge. However, Factol falls short on the TruthfulQA dataset and the Detect-and-Mitigate method underperforms on the StrategyQA dataset. This may be attributed to error accumulation caused by unnecessary retrieval (external hallucinations).

We also conducted additional experiments to compare with four adaptive retrieval methods. Notably, Adaptive-Retrieval faces challenges on the TruthfulQA dataset due to some questions lacking explicit entities, causing it to struggle in deciding when to retrieve based on entity popularity, leading to poor performance. Besides, Self-RAG’s effectiveness is hindered by the capabilities of the LLaMa model, resulting in inferior performance.

Compared to the aforementioned baselines, Rowan demonstrates significant performance gains on both datasets. Both Rowan-CI and Rowan-CM exhibit excellent hallucination mitigation capabilities, even compared to strong adaptive retrieval methods. Specifically, Rowan-Hybrid achieves a GFT-Judge score of 59.34% on the TruthfulQA dataset, surpassing the strongest baseline by 16.74%. Additionally, Rowan-Hybrid attains an accuracy of 75.60% on the StrategyQA dataset, significantly outperforming existing baselines. These results underscore Rowan’s ability to effectively leverage parametric knowledge and external information for maximum advantage.

### 5.2 Effect of Detection Module (RQ2)
To validate the effectiveness of our proposed hallucination detection for hallucination mitigation within strong detection methods: (1) average token-level probability / entropy that utilizes the probabilities $p$ of entropies generated by a proxy LM (e.g., LLaMa-7B) as a metric to measure hallucination. (2) SelfCheckGPT [25] that measures information consistency between the different responses to determine hallucinations. (3) Consistency-based method [43], SAC*, that evaluates semantic-aware cross-check consistency, building upon the foundation of self-consistency principles.

Based on the results in Table 5, it is evident that logits-based methods perform moderately well in detecting hallucinations. Specifically, the entropy-based estimation exhibits superior performance. Besides, the combination of different variants in SelfCheckGPT leads to a slight performance improvement. Additionally, it is worth noting that SAC* achieve competitive performance on detecting significant factual errors accurately.

Finally, we observe that Rowan significantly outperforms these strong hallucination detection baselines, especially the monolingual detection method SAC*. We also notice that Rowan achieves notable hallucination mitigation with a minimal number of retrieval calls. This underlines the superior efficiency of our adaptive retrieval module.

### 5.3 Scalability of Rowan (RQ3)
**Scalability to Open-Sure LMs.** In addition to ChatGPT, we also assess Rowan’s effectiveness when employing open-source language models: Qwen-1, Qwen-1.5-14B, Qwen-2-78-Instruct, and Qwen-2-78-Instruct*. These models can choose for the cross-language detection model $M$ due to their strong capabilities in
```


### --- Page 0007 ---

```markdown
# Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs

## Table 5: Performance comparison of applying other hallucination detection methods in adaptive retrieval scenarios. We also report the ratio of retrieval conducted by each method.

| Models         | TruthfulQA                     | StrategyQA                     |
|----------------|--------------------------------|--------------------------------|
|                | GPT-Judge ↑ | BLEU ↑ | Rouge-L ↑ | Ratio(%) | Accuracy ↑ | Ratio(%) |
| LLaMa2-7B     | 50.19       | 12.51  | 31.11     | 46.5     | 71.50      | 23.0     |
|                | 52.22       | 9.18   | 28.37     | 59.0     | 72.00      | 26.5     |
| SelfCheckGPT   |            |        |           |          |            |          |
| w/ BERTScore   | 51.38       | 8.43   | 26.85     | 27.2     | 67.50      | 21.0     |
| w/ MQAG        | 52.76       | 7.69   | 26.92     | 54.4     | 69.00      | 34.0     |
| w/ Ngram       | 52.41       | 5.01   | 21.60     | 34.9     | 66.50      | 32.0     |
| Combination     | 53.10       | 6.69   | 24.04     | 51.3     | 69.50      | 30.0     |
| Consistency    |            |        |           |          |            |          |
| SAC³-Q         | 51.02       | 7.90   | 28.00     | 24.5     | 65.50      | 24.0     |
| SAC³-a1        | 52.22       | 9.37   | 29.56     | 20.8     | 67.00      | 24.5     |
| Rowen-Hybrid   | 59.34       | 15.27  | 31.15     | 23.0     | 75.60      | 20.0     |

## Figure 3: Experimental results of Rowen with open-source LLMs on TruthfulQA.

![Experimental results of Rowen with open-source LLMs on TruthfulQA](assets/page_0007_img_1.png)

## Table 6: Hallucination mitigation performance on the NQ dataset and TriviaQA dataset.

| Methods        | NQ            |            | TriviaQA      |            |
|----------------|---------------|------------|---------------|------------|
|                | EM            | F1         | EM            | F1         |
| FLARE          | 32.50         | 43.91      | 59.00         | 68.34      |
| Adaptive-RAG   | 35.04         | 48.44      | 58.00         | 68.97      |
| Rowen-CL       | 38.05         | 51.60      | 76.66         |            |
| Rowen-CM       | 37.36         | 53.27      | 65.00         | 74.98      |
| Rowen-Hybrid   | 39.98         | 57.31      | 69.04         | 78.64      |

## 5.4 Impact of Hyper-parameters in Rowen (RQ4)

We analyze the impact of key factors for hallucination mitigation on the TruthfulQA dataset, including detection thresholds, numbers of perturbed questions, and cross-model detection weights, as shown in Figure 4.

### Detection Threshold. 

We experiment with various consistency thresholds to investigate their impact on hallucination mitigation effectiveness. As shown in Figure 4(a), we observe significant performance improvement when the detection threshold increases from 0.2 to 0.6, as this identifies more hallucinations and enhances...
```

### --- Page 0008 ---

```markdown
![Impact of key factors for hallucination mitigation on the TruthfulQA dataset.](assets/page_0008_img_1.png)
![The percentage of internal and external hallucinations across different methods on both datasets.](assets/page_0008_img_2.png)
![Comparison of hallucination mitigation performance for Rowen-CL under different language choices on the TruthfulQA dataset.](assets/page_0008_img_3.png)

## Table 7: Comparison of hallucination mitigation performance for Rowen-CL under different language choices on the TruthfulQA dataset.

| Language Pair         | Discrepancy | GPT-Judge |
|-----------------------|-------------|-----------|
| English - German      | Low         | 50.55     |
| English - French      | Medium      | 52.26     |
| English - Chinese     | High        | 57.39     |

## Table 8: Comparison of hallucination mitigation performance for Rowen-CH under different verifier LMs on the TruthfulQA dataset.

| Verifier LM  | Capability | GPT-Judge |
|--------------|------------|-----------|
| Qwen-Turbo   | Low        | 52.50     |
| Qwen-Plus    | Medium     | 54.09     |
| Qwen-Max     | High       | 56.29     |

### Impact of Key Factors for Hallucination Mitigation on the TruthfulQA Dataset

#### Number of Perturbed Questions
We also study the effect of the number of perturbed question on hallucination mitigation and the results are shown in Figure 4(b). We observe that the performance of Rowen improves with an increasing number of question samples, yet the performance gain gradually diminishes after using 6 question samples. This indicates that, in practice, using 6 question samples could achieve reasonably good performance at a relatively low computational cost.

#### Cross-Model Detection Weight
We also examine the impact of different cross-model detection weight for Rowen-Hybrid. Figure 4(d) shows that GPT-Judge scores significantly increase as the weight rises from 0.25 to 1.0, indicating that introducing cross-model checking can improve the accuracy of hallucination detection. However, when the weight factor exceeds 1.0, performance declines, with scores dropping to around 56 at a weight of 1.5. This suggests that while a moderate weight enhances factuality, excessively a high value allows the cross-model detection module to exert too much influence, reducing effectiveness.

#### Impact of the Choice of Target Language
To investigate how different language pairs affect the effectiveness of hallucination mitigation for Rowen-CL, we conduct experiments on the TruthfulQA dataset with various target languages. The results of these experiments are presented in Table 7. It is observed that the combination of English as the source language and German as the target language yields the least favorable results. This may be attributed to their shared Germanic language family roots, which results in numerous linguistic similarities and overlapping cultural references. Conversely, as the cultural divergence between the source and target languages widens, we witness an enhancement in the performance of hallucination mitigation. This trend substantiates the hypothesis that cultural disparities between languages play a pivotal role in identifying hallucinations and bolstering the factuality of the generated responses.

#### Impact of the Choice of Verifier LM
To investigate the impact of different verifier LMs on the efficacy of hallucination mitigation for Rowen-CH, we conducted experiments using various verifier LMs on the TruthfulQA dataset. The experimental results presented in Table 8 demonstrate a significant improvement in hallucination mitigation as the capability of the verifier LM increases. Specifically, Qwen-Max exhibited the highest effectiveness in this task, achieving an efficacy of 56.29%, while Qwen-Turbo and Qwen-Plus...
```

### --- Page 0009 ---

```markdown
# Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs
SIGIR-AP 2025, December 7–10, 2025, Xi’an, China

## Table 9: Analysis on LLM calls efficiency and hallucination mitigation performance across different methods.

| Methods                | TruthfulQA | StrategyQA | GPT-Judge = Calls | Accuracy | Calls |
|-----------------------|------------|------------|-------------------|----------|-------|
| Self-Reflection       | 42.99      | 6          | 62.40             | 5        |       |
| Multi-agent Deb.      | 50.83      | 6          | 65.73             | 6        |       |
| Rowen-CL              | 57.39      | 6          | 74.00             | 5        |       |
| Rowen-CM              | 56.29      | 5          | 72.40             | 4        |       |
| Rowen-Hybrid          | 59.34      | 8          | 75.60             | 6        |       |

## Table 10: Statistics on the average number of retrieval calls to answer each question.

| Methods                | # Num of Retrieval Calls | TruthfulQA | StrategyQA |
|-----------------------|--------------------------|------------|------------|
| Factool               | 12.5                     | 11.6       |
| Detect-and-Mit.       | 7.2                      | 5.5        |
| FLARE                 | 2.1                      | 3.9        |
| Adaptive-RAG          | 0.9                      | 1.4        |
| Rowen-Hybrid          | 1.5                      | 0.5        |

5.5 **Analysis of Computation Cost (RQ5)**  
To verify whether Rowen effectively reduces internal and external hallucinations, we present a comparative analysis of the presence of both hallucinations across three methods—Factool, Detect-and-Mit, and Rowen-Hybrid. Internal hallucinations refer to the generation of incorrect responses using only the parameterized knowledge of LLMs, while external hallucinations refer to the generation of incorrect responses using documents introduced after retrieval. Figure 5 shows that Factool and Detect-and-Mit are significantly prone to both types of hallucinations. Both baseline methods have high levels of internal and external hallucinations, struggling with internal coherence and external realignment. In contrast, Rowen-Hybrid effectively reduces external hallucinations by timely integrating external knowledge, thereby avoiding unnecessary information retrieval and mitigating potential errors.

5.6 **Analysis of Computation Cost (RQ6)**  
**Efficiency Analysis of LLM Calls.** To evaluate the efficiency of LLM calls in Rowen, we compare various methods—including Self-Reflection, Multi-agent Deb, and three different Rowen variants—by analyzing the number of API calls required to answer a single question and their effectiveness in mitigating hallucinations. From the results shown in Table 9, we find that Rowen-CL and Rowen-CM, achieve significantly better hallucination mitigation compared to the baseline methods, while using a similar number of API calls.

**Efficiency Analysis of Retrieval Calls.** To verify the retrieval efficiency of Rowen, in Table 10, we compare the average number of retrieval calls made by Factool, Detect-and-Mit, FLARE, and Rowen-Hybrid to answer a question across two datasets. Factool generates the most retrieval API calls due to verifying each claim. Detect-and-Mitig and FLARE identify low-confident concepts in LLM outputs and call search APIs with fewer API calls. Adaptive-RAG underestimates the difficulty of adversarial questions in the TruthfulQA dataset, resulting in poor truthfulness scores despite using the minimum number of retrievals. Rowen-Hybrid, which retrieves information only for uncertain responses, excels in both retrieval efficiency and factual accuracy, showcasing its superiority over other RAG methods. This demonstrates the superiority of Rowen-Hybrid over other RAG methods in terms of both efficiency and effectiveness.

## Acknowledgements  
This work was supported by the Strategic Priority Research Program of the Chinese Academy of Sciences under Grant No.XDDB060302, the National Science Foundation of China (NSFC) under Grants No. 62276438, the Key Research and Development Program of Xinjing Autonomous Region under Grant No. 202204230826, the Beijing Nova Program under Grants No. 2025048765, and the Youth Innovation Promotion Association CAS under Grants No. 202204230826.
```

### --- Page 0010 ---

```markdown
# SIGIR AP 2023, December 7–10, 2023, Xi'an, China

[1] Dan Hendrycks, Colin Burns, Steven Basart, Andy Z. Matuszek, Dawn Song, and Jakob N. Schneider. 2022. Measuring Massive Multitask Language Understanding with the MMLU Benchmark. In *Proceedings of the 38th International Conference on Machine Learning*, volume 139, 2021. PMLR 139: 13045–13060.

[2] Lei Hu, Yanzhu Xu, Weijia Ma, Weizhong Zheng, Feng Han, Hongliang Wang, Qiangling Chen, Weijun Peng, Xiaokefeng Feng, Bing Qi, and Ting Han. 2022. A Survey on Hallucination in Large Language Models: Principles, Mitigation Techniques, and Open Questions. *arXiv preprint arXiv:2111.03562* (2022).

[3] Yu-Hen Hsu, Yanzhu Xu, Zhihua Zhu, Junjie Zhang, Zhi-Yang Zhu, Tangjun Su, Junten Liu, Chuanbei Yu, Yizhu Zhang, Jizhi Ye, Liu Maosong Sun, and Junfeng Han. 2023. C-Eval: A Multi-Level Benchmark for Chinese Evaluation Suite for Foundation Models. *arXiv preprint arXiv:2305.03282* (2023).

[4] Yuheng Huang, Jiayang Song, Zhiheng Wang, Huaming Chen, and Lei Ma. 2023. Look Before You Leap: An Exploratory Study of Certainty Measurement for Large Language Models. *CoRR abs/2307.12604* (2023).

[5] Yosuke Sogouye, Jindong Baek, Sukhminder Chou, Ji Hwang, and Jorg C. Park. 2024. Adaptive Active Learning for Dialog Retrieval using Large Language Models. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, ACL 2024, 3403–3408 (2024).

[6] David N. J. Nevoen, Lee R. Pritcher, Tzucheng Yu, Dan Xu Yu, Etsuko Ishii, Neji Wang, and Andrada, and Pascal Furu. 2023. Survey of Hallucination in Large Language Generation. *ACM Computing Surveys* 55, 2 (2023), 12403–12518.

[7] Taisuke T. Yamauchi, Yuma N. Yamamoto, Eiko Saito, and Reika Matsuda. 2023. Towards Mitigating LLM Hallucination via Self Reflection. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, EMNLP 2023, 16–20.

[8] Zhengbo Jiang, Frank K. Yau, Zhiqiang Sun, Jian Liu, and Zheng Zhang. 2023. A Survey on Large Language Models: Recent Advances and Future Directions. *arXiv preprint arXiv:2301.01234* (2023).

[9] Saakar Kothari, Tom Andrews, Asher Aksel, Tom Herglind, Daven Thang, and David T. H. Schneider. 2023. The Adversarial Robustness of Large Language Models. *arXiv preprint arXiv:2305.01234* (2023).

[10] Huan Chen, Junhao Bai, Sanjiv Kumar, Shanshan Xue, Lingyu Wang, and Yujie Zhang. 2023. A Survey on Large Language Models: Current Status and Future Directions. *arXiv preprint arXiv:2305.01234* (2023).

[11] Zhenhua Zhang, Yifan Liu, Kamila Das, Bradley A. Makin, and Kumar Sricharan. 2023. Reproducibility of Large Language Models: A Survey. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, EMNLP 2023, 16–20.

[12] Ziyu Zhang, Yifan Liu, Ziyang Guo, Dan Kong, and Jiajun Zhang. 2023. A Survey on the Evaluation of Large Language Models. *arXiv preprint arXiv:2305.01234* (2023).

[13] Jiaxuan Zhang, Ruanhe Li, Kamila Das, Bradley A. Makin, and Kumar Sricharan. 2023. Reproducibility of Large Language Models: A Survey. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, EMNLP 2023, 16–20.

[14] Zhenhua Zhang, Yifan Liu, Ziyang Guo, Dan Kong, and Jiajun Zhang. 2023. A Survey on the Evaluation of Large Language Models. *arXiv preprint arXiv:2305.01234* (2023).

[15] Xue Zhang, Liyang Cui, Dan Lei, Luming Zhang, and Tuanfu Liu. 2023. Hallucination in Large Language Models: A Survey. *arXiv preprint arXiv:2305.01234* (2023).

[16] Zhenhua Zhang, Asher Aksel, and Naoki Okazaki. 2023. Automatic Evaluation of Prompting for Large Language Models. *arXiv preprint arXiv:2305.01234* (2023).

[17] Xue Zhang, Liyang Cui, Dan Lei, Luming Zhang, and Tuanfu Liu. 2023. Hallucination in Large Language Models: A Survey. *arXiv preprint arXiv:2305.01234* (2023).

[18] Ruiyang Wang, Yanhao Wang, Yiyang Qiu, Yanzhe Zhang, Lijia Huo, Tian Huo, Wei Yu, J. Rong Wang, and Haifeng Wang. 2023. Investigating the Feasibility and Effectiveness of Large Language Models with Retrieval Augmentation. *CoRR abs/2307.01234* (2023).

[19] Fei Shi, Xinyu Chen, Kanishka Mishra, Nathan Sables, David Dohan, et al. 2023. Neural Network Distillation by Retrieval Context. In *International Conference on Machine Learning*, ICML 2023, 322–340 (2023). *Proceedings of Machine Learning Research*, Vol. 121. PMLR 121: 3120–3127.

[20] Shunchao Tu, Xiu Yu, Baohong Ding, Yanzhang Xu, C. H. Suen, Jinyu Gao, Yuhua Shen, and Rolin Ding. 2024. Vent to Trust LLMs Aligning Confidence with Response Quality. *arXiv preprint arXiv:2401.12872* (2024).

[21] Ronan Teyssou, Galadriel T. R. Y. N. Zhang, Thomas V. M. L. A. Carlos Gutterrez, Percy Liang, and Taisuke T. Yamauchi. 2023. Standard Japan as Instruction-following LLM model. *https://github.com/tatsu-lab/standard*.

[22] Katherine Tian, Eric Mitchell, Alon R. Abadi, Sharma Rafal, Ralfiou, Huxley Zhe, and Christopher J. Manning. 2023. Just Ask for Collaboration: Strategies for Fine-tuning Pretrained Language Models. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, EMNLP 2023, 16–20.

[23] Hugo Touvier, Thibault Laverdure, Xavier Martel, Hambo Faus, Laaksonen, Timo K. E. R. B. N. M. A. M. 2023. Towards Efficient and Efficient Foundation Language Models. *CoRR abs/2302.13917* (2023).

[24] Zhenhua Zhang, Asher Aksel, and Naoki Okazaki. 2023. Automatic Evaluation of Prompting for Large Language Models. *arXiv preprint arXiv:2305.01234* (2023).

[25] Xue Zhang, Liyang Cui, Dan Lei, Luming Zhang, and Tuanfu Liu. 2023. Hallucination in Large Language Models: A Survey. *arXiv preprint arXiv:2305.01234* (2023).

[26] Xue Zhang, Liyang Cui, Dan Lei, Luming Zhang, and Tuanfu Liu. 2023. Hallucination in Large Language Models: A Survey. *arXiv preprint arXiv:2305.01234* (2023).

[27] Chao Ren, Xianpeng Li, Shafly, Chengwei Qin, and Lihong Ding. 2023. Investigating the Effectiveness of Large Language Models with Retrieval Augmentation. *Journal of Computer Science* 43, 38 (2023), 6538–6552. *doi:10.31234/osf.io/2g23*.
```


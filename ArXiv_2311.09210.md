# ArXiv 2311.09210

### --- Page 0001 ---

```markdown
# Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models

**Wenhau Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li, Hongwei Wang, Dong Yu**  
Tencent AI Lab  
wenhaowyu@global.tencent.com

## Abstract

Retrieval-augmented language model (RALM) represents a significant advancement in mitigating factual hallucination by leveraging external knowledge sources. However, the reliability of the retrieved information is not always guaranteed, and the retrieval of irrelevant data can mislead the response generation. Moreover, standard RALMs frequently neglect their intrinsic knowledge due to the interference from retrieved information. In instances where the retrieved information is irrelevant, RALMs should ideally utilize their intrinsic knowledge or, in the absence of both intrinsic and retrieved knowledge, opt to respond with "unknown" to avoid hallucination. In this paper, we introduce CHAIN-OF-NOTE (CoN), a novel approach to improve robustness of RALMs in facing noisy, irrelevant documents and in handling unknown scenarios. The core idea of CoN is to generate sequential reading notes for each retrieved document, enabling a thorough evaluation of their relevance to the given question and integrating this information to formulate the final answer. Our experimental results show that GPT-4, when equipped with CoN, outperforms the CHAIN-OF-THOUGHT approach. Besides, we utilized GPT-4 to create 10K document data, subsequently trained on LLaMa-2 7B model. Our experiments across four open-domain QA benchmarks show that fine-tuned RALMs equipped with CoN significantly outperform standard fine-tuned RALMs.

## 1 Introduction

Retrieval-augmented language models (RALMs) represent a novel framework that significantly advances large language models (Touvron et al., 2023; OpenAI, 2023) by addressing key limitations such as reducing factual hallucinations (Ji et al., 2023; Zhang et al., 2023a), injecting up-to-date knowledge in a plug-and-play manner (Dhingra et al., 2022; Vu et al., 2023), and enhancing domain-specific expertise (Li et al., 2023; Qin et al., 2023).

![Figure 1: Compared with the current RALMs, the core idea behind CHAIN-OF-NOTE (CoN) is to generate sequential reading notes for the retrieved documents, ensuring a systematic assessment of their relevance to the input question before formulating a final response.](assets/page_0001_img_1.png)

These enhancements primarily stem from integrating large language models (LLMs) with external knowledge sources (Guu et al., 2020; Lewis et al., 2020; Borgeaud et al., 2022; Shi et al., 2023c). In a typical RALM setup, a query is first processed by a retriever that searches a vast evidence corpus for pertinent documents. A reader then examines these documents, extracting useful information and formulating the final output answer.

However, there exist several issues with the current RALM framework. First, there is no guarantee that the information retrieval (IR) system will always yield the most pertinent or trustworthy information. The retrieval of irrelevant data can lead to misguided responses (Shi et al., 2023a; Yoran et al., 2023), and potentially causing the model to overlook its inherent knowledge, even when it possesses adequate information to address the query (Mallen
```

### --- Page 0002 ---

```markdown
et al., 2023). Secondly, state-of-the-art LLMs often hallucinate when addressing fact-oriented questions, a deficiency that can be risky and may discourage users (Ji et al., 2023; Zhang et al., 2023). Ideally, an intelligent system should be capable of determining whether it has enough knowledge, both intrinsic and retrieved, to provide an accurate answer. In cases where knowledge is insufficient, the system should respond with “unknown” when the answer cannot be determined. Based on the shortcomings of the standard RALM system, in this paper, we aim to improve the robustness of RALMs, mainly focusing on two pivotal aspects:

| **(1) Noise Robustness:** The ability of a RALM to discern and disregard noisy information present in irrelevant documents, while appropriately leveraging its intrinsic knowledge. |
| **(2) Unknown Robustness:** The capacity of a RALM to acknowledge its limitations by responding with “unknown” when given a query it does not have the corresponding knowledge to answer, and the relevant information is not found within the retrieved documents. |

In this work, we introduce a novel framework named CHAIN-OF-NOTE (CoN), designed to enhance the robustness of RALMs. The cornerstone of CoN is to generate a series of reading notes for retrieved documents, enabling a comprehensive assessment of their relevance to the input query. This approach not only evaluates each document’s pertinence but also pinpoints the most critical and reliable information therein. This process effectively filters out irrelevant or less credible content, leading to responses that are more precise and contextually relevant, as exemplified in Figure 1. Besides, CoN enhances the capability of RALM to handle queries fall outside the scope of training data. In cases where the retrieved documents do not provide any relevant information, CoN can guide the model to acknowledge its limitations and respond with an “unknown” or provide possible explanation based on available data, enhancing reliability.

To validate the effectiveness of the CoN idea, we first conducted a comparison with CHAIN-OF-THOUGHT (CoT) (Wei et al., 2022) using GPT-4 as the reader, showing CoN is more effective than CoT in retrieval-augmented scenarios. Next, we prompted GPT-4 (OpenAI, 2023) to generate a 10K training examples based on questions collected from NQ (Kwiatkowski et al., 2019), and subsequently trained on the LLaMa-2 7B, to valid the note-taking ability for smaller-sized models. Our evaluation of the RALM, integrated with CoN and compared to the standard RALM system, focused on three major aspects: (1) overall QA performance using DPR-retrieved documents, (2) noise robustness, assessed by introducing noise information to the system, and (3) unknown robustness, evaluated through queries not covered in the LLaMa-2 pre-training data, i.e., real-time questions. The evaluations were conducted on the NQ datasets, namely TriviaQA (Joshi et al., 2017), WebQ (Berant et al., 2013), and RealTimeQA (Kasai et al., 2023). Our experiments show that CoN not only improves overall QA performance when employed with DPR-retrieved documents but also significantly enhances robustness in both noise and unknown aspects. This includes a +7.9 increase in accuracy (measured by the exact match score) with noisily retrieved documents, and a +10.5 increase in the rejection rate for real-time questions1 that are beyond the pre-training knowledge scope.

## 2 Proposed Method

### 2.1 Overview
In this section, we introduce CHAIN-OF-NOTE, an innovative advancement for retrieval-augmented language models (RALMs). Specifically, CoN framework generates sequential reading notes for the retrieved documents, which enables a systematic evaluation of the relevance and accuracy of information retrieved from external documents. By creating sequential reading notes, the model not only assesses the pertinence of each document to the query but also identifies the most critical and reliable pieces of information within these documents. This process helps in filtering out irrelevant or less trustworthy content, leading to more accurate and contextually relevant responses.

### 2.2 Background of Existing RALMs
RALMs signify a transformative development in language models, enhancing their output by incorporating external knowledge. These models operate by introducing an auxiliary variable, denoted as $d$, which represents retrieved documents. This inclusion allows them to consider a range of possible documents, thereby producing responses that are more informed and precise (Lazaridou et al., 2023).
```

### --- Page 0003 ---

```markdown
# Figure 2: Illustration of the CHAIN-OF-NOTE (CoN) framework

| (a) CoN: Relevant → Find the answer | (b) CoN: Irrelevant → Infer the answer | (c) CoN: Irrelevant → Answer Unknown |
|--------------------------------------|-----------------------------------------|---------------------------------------|
| Wikipedia: "When was the Deadpool 2 being released?" <br> Wikipedia: "Deadpool 2 is a 2018 American super-hero film based on the Marvel Comics character Deadpool distributed by 20th Century Fox ... This movie was followed by a release in a particular state over the next days, including the United States on June 2, 2018." <br> Wikipedia: "Deadpool 2 premiered at Leicester Square in London on May 18, 2018. It was released in the United States on May 18, 2018, having been previously scheduled for a release of June 1, based on information provided above, the answer is May 18, 2018." | Wikipedia: "Who wrote the song 'When I was 17'?" <br> Wikipedia: "The nostalgic and melodic song reconstructs the lives of girls with whom the singer had relationships at various years when he was 17, 'small-town girls ... on the village green'." <br> Wikipedia: "... 'It was a Very Good Year', commonly unknown as 'When I was 17', was more famous by Frank Sinatra's version in 1966, which won the Grammy Award for Best Male Vocal Performance in 1966." | Wikipedia: "When is the fourth movie of the Divergent series coming out?" <br> Wikipedia: "The Divergent Series is a feature film trilogy ... distributed by Summit Entertainment and Lionsgate Films, the series' three science fiction films set in a dystopian society: Divergent, Insurgent, and Allegiant." <br> Wikipedia: "The film (Divergent) was a box office bomb, grossing $137 million against its budget of $142 million, becoming the low-grossing 'Divergent' film. The initial release date of the fourth film in Divergent series was delayed due to budget cuts for a potential fourth movie. However, neither passage confirms the release date of the fourth film in Divergent series. Given this lack of specific information and my current knowledge, the answer is unknown." |

2022; Shi et al., 2023c). The RALM models can be represented as $p(y|x) = \sum_{i} p(y|d_i, x)p(d_i|x)$. Here, $x$ represents the input query, and $y$ signifies the model’s generated response. In practice, it is infeasible to compute the sum over all possible documents due to the vast number of potential sources. Consequently, the most common approach involves approximating the sum over $d_i$ using the $k$ highest ranked documents, and providing all these documents as part of the input. We assume, w.l.o.g., that these documents are $d_1, \ldots, d_k$, yielding $p(y|x) = \sum_{i=1}^{k} p(y|d_i, x)p(d_i|x)$.

However, existing RALMs suffer from several limitations:

- **Risk of Surface-Level Processing**: When directly generating an answer, language models might rely on surface-level information without deep comprehension. Thus, they could easily overlook the nuances of question or documents, particularly in complex or indirect questions.

- **Difficulty in Handling Contradictory Information**: When faced with documents containing contradictory information, directly generating an answer becomes challenging. The model may struggle to assess which piece of information is more credible or relevant.

## 2.3 The CHAIN-OF-NOTE Framework

The CHAIN-OF-NOTE (CoN) framework presents a solution to the challenges faced by retrieval-augmented language models (RALMs). This framework significantly enhances the ability of RALMs to critically assess retrieved documents through a structured note-taking process. Specifically, it involves generating concise and contextually relevant summaries or notes for each document. This method allows the model to systematically evaluate the relevance and accuracy of information drawn from external documents. By creating sequential reading notes, CoN not only assesses...
```

### --- Page 0004 ---

```markdown
# PAGE_NAME: page_0004

## 2.3.1 CHAIN-OF-NOTE Format Design

The framework primarily constructs three types of reading notes, as shown in Figure 2, based on the relevance of the retrieved documents to the input question: First, when a document directly answers the query, the model formulates the final response based on this relevant information, as shown in Figure 2(a). Second, if the retrieved document does not directly answer the query but provides useful context, the model leverages this information along with its inherent knowledge to deduce an answer, as shown in Figure 2(b). Third, in cases where the retrieved documents are irrelevant, and the model lacks sufficient knowledge to answer, it defaults to responding with "unknown", as shown in Figure 2(c). This nuanced approach mirrors human information processing, striking a balance between direct retrieval, inferential reasoning, and the acknowledgment of knowledge gaps.

## 2.3.2 Data Collection and Model Training

To equip the model with the ability to generate such reading notes, it’s essential to gather appropriate training data. Manual annotation for each reading note is resource-intensive, so we employ a state-of-the-art language model – GPT-4 – to generate the notes data. This method is both cost-effective and enhances reproducibility. We initiate this process by randomly sampling 10k questions from the NQ (Kwiatkowski et al., 2019) training dataset. GPT-4 is then prompted with specific instructions and in-context examples to generate the three distinct types of note generation (detailed in Appendix A.5). The quality of GPT-4’s predictions is subsequently assessed through human evaluations on a small subset of the data before proceeding to the entire set. The NQ dataset is chosen as our primary dataset due to its diverse range of real queries from search engines. However, to ensure the model’s adaptability, we also test its performance on three additional open-domain datasets, including TriviaQA, WebQ, and RealTimeQA, showing its generalization capabilities to out-of-domain (OOD) data.

After collecting 10K training data from GPT-4, the next step involves using them to train a LLaMA-2 7B model (Touvron et al., 2023), to validate the feasibility of generating CHAIN-OF-NOTE (CoN) outputs. To do this, we concatenate the instruction, question and document as a prompt and train the model to generate notes and answer in a standard supervised way. Our in-house model learns to sequentially generate reading notes for each document to assess their relevance to the input query. Responses are generated based on the document’s relevance, enhancing accuracy and reducing misinformation. If all documents are irrelevant, the model either relies on inherent knowledge for an answer or responds with "unknown" if the answer cannot be determined accurately.

## 2.3.3 Hybrid Training for Better Efficiency

Generating CHAIN-OF-NOTE (CoN) would increase inference cost, potentially hindering real-world usage. To address this, we experimented with a simple yet effective strategy for internalizing CoN reasoning, called Hybrid Training.

Specifically, we allocate 50% of the training time to the standard RALM, which involves directly generating answers without notes, and the other 50% to RALM with CoN. This strategy allows the model to internalize intermediate reasoning steps during training. Additionally, we add two different prompt words before each category of data.

During the inference phase, we exclusively use the standard RALM prompt to guide the model, prompting it to output answers without relying on explicit reading notes. This approach leverages the hidden states developed during training for implicit CoN reasoning. The model trained with the hybrid training strategy maintains the same inference time while achieving only slightly lower performance wit CoN. The results will be introduced in §3.5.
```


### --- Page 0005 ---

```markdown
| Datasets  | Full size | IR Recall | Subset size |
|-----------|-----------|-----------|-------------|
| NQ        | 3,610     | 73.82     | 2,086       |
| TriviaQA  | 7,993     | 89.95     | 7,074       |
| WebQ      | 2,032     | 64.22     | 1,231       |

Table 1: Dataset statistics. The recall evaluation is based on DPR retrieval on the full test set.

## 3 Experiments

### 3.1 Experimental Settings and Evaluations

#### 3.1.1 Datasets and Splits

We conducted comprehensive experiments using three benchmark datasets in open-domain question answering (QA): NQ (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), and WebQ (Berant et al., 2013), with further details provided in Appendix A.3. Additionally, we employed RealTimeQA (Kasai et al., 2023) as a special case to evaluate “unknown” robustness.

The evaluation was conducted based on two evaluations sets: full set and subset evaluation. Firstly, akin to traditional open-domain QA evaluation, we assessed the models using all questions from the test set to evaluate the overall QA performance. The documents were retrieved using DPR, and the top-k documents were fed into the generator. We adhered to the same test splits for the open-domain QA setting as used by Izacard and Grave (2021); Karpukhin et al. (2020). For TriviaQA, evaluations from LLaMa-2 (Touvron et al., 2023) were conducted on the Wikipedia dev set comprising 7,993 examples. Therefore, we also follow the same evaluation on this dev set to facilitate comparisons with their performance. Secondly, to assess the model’s noise robustness and unknown robustness, we extracted subsets from the above test sets that contained relevant documents in the retrieved list. We then enumerated each retrieved document to determine if it was a golden document for the given question. Based on the noise ratio $r$, for instance, if the top-k documents are needed for the generator, then $k - r$ would be the number of noisy documents, and $k$ would be the number of relevant documents. For example, when noise ratio is 20% and top-5 documents are needed, then 4 are relevant documents, and 1 is irrelevant documents. During the enumeration of the retrieved documents in data pre-processing, we populated two lists; when one list reached its limit, we stopped adding more documents to that list until both lists were complete. In instances where no relevant documents are retrieved by the DPR for certain questions, we exclude these from robustness evaluation. Therefore, the subset is smaller than the original test set, as shown in Table 1.

#### 3.1.2 Baseline Methods

CHAIN-OF-NOTE (CON) is built upon the traditional retrieve-then-read pipeline (Lewis et al., 2020). Recent implementations such as Lazaridou et al. (2022); Shi et al. (2023a); Luo et al. (2023) integrate large language models to achieve better performance. Therefore, we primarily compare our approach against these retrieve-read methods. As outlined in the §2.3, we denote an input question as $x$ and its corresponding answer as $y$. Besides, $d_i$ represents the $i$-th retrieved document, and $y_d$ is the associated reading note for that document. Here we show the difference of methods to compare.

QA fine-tune w/o IR are trained to directly generate an answer from the input question, without relying on any external retrieved information. Essentially, it learns the function $f: x \rightarrow y$, transforming the question $x$ directly to answer $y$.

Retrieve-Read (Shi et al., 2023c) are trained to generate an answer not only from the question but also by incorporating retrieved documents. It learns the function $f: \{x, d_1, \ldots, d_k\} \rightarrow y$, meaning it transforms the question $x$ and a set of retrieved documents $\{d_1, \ldots, d_k\}$ into an answer $y$.

Retrieve-Read with CHAIN-OF-NOTE are trained to generate reading notes for each retrieved document before formulating the final answer. It learns the function $f: \{x, d_1, \ldots, d_k\} \rightarrow \{y_{d_1}, \ldots, y_{d_k}\}$, thereby enabling the model to process the question $x$ and retrieved documents $\{d_1, \ldots, d_k\}$ to produce reading notes $\{y_{d_1}, \ldots, y_{d_k}\}$ and the final answer $y$.

For fair comparability, we trained all LLaMa-2 models on same training set, with the main difference being in the input and output formats. We also note that the experiments conducted with GPT-4 were performed in a zero-shot setting. The prompts used for various experimental conditions are detailed in Appendix A.5.

#### 3.1.3 Evaluation Metrics

For the evaluation of open-domain QA performance, we have employed two widely recognized metrics: Exact Match (EM) and F1 score, as suggested by prior work in the Chen et al. (2017);
```

### --- Page 0006 ---

```markdown
| Models                                   | NQ         |            | TriviaQA   |            | WebQ       |            | Average    |            |
|------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|
|                                          | EM         | F1         | EM         | F1         | EM         | F1         | EM         | F1         |
|------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|
| Backbone language model: LLaMa-2 7B     |            |            |            |            |            |            |            |            |
| QA fine-tune w/o IR                     | 28.80      | 37.53      | 63.19      | 68.61      | 28.30      | 42.77      | 35.98      | 44.27      |
| SAIL (Luo et al., 2023)*                | 36.20      | 44.23      | 73.20      | 80.92      | 27.92      | 40.65      | 45.77      | 55.27      |
| Retrieve-Read (Shi et al., 2023c)      | 47.39      | 55.81      | 74.92      | 81.53      | 29.58      | 43.51      | 48.49      | 56.97      |
| + CHAIN-OF-NOTE (ours)                  | 48.92      | 57.53      | 76.72      | 82.25      | 32.33      | 46.68      | 50.46      | 58.78      |
|                                          | (+1.53)    | (+1.72)    | (+1.35)    | (+0.72)    | (+2.75)    | (+3.17)    | (+1.97)    | (+1.81)    |
|------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|
| Backbone language model: GPT-4-116 6†   |            |            |            |            |            |            |            |            |
| QA prompt w/o IR                        | 54.0       | 74.2       | 56.2       | 61.5       |            |            |            |            |
| Retrieve-Read (Shi et al., 2023c)      | 61.8       | 70.6       | 56.8       | 63.1       |            |            |            |            |
| + CHAIN-OF-THOUGHT                      | 63.6       | 71.2       | 58.4       | 64.4       |            |            |            |            |
| + CHAIN-OF-NOTE (ours)                  | 63.8       | 74.6       | 58.8       | 65.7       |            |            |            |            |
|                                          | (+2.0)     | (+4.0)     | (+2.0)     | (+2.6)     |            |            |            |            |

Table 2: The RALM, when equipped with CHAIN-OF-NOTE (CoN), demonstrates a marginal improvement over the standard RALM in full test set evaluations. Significantly, it outperforms the standard RALM system in scenarios with noisy documents, suggesting that CoN can substantially enhance the model’s noise robustness. 
* SAIL was designed for retrieval-augmented instruction tuning, and as such, may not be ideally factual QA. 
† Evaluating GPT-4 outputs with EM score is challenging; we opt for Accuracy, with reasons outlined in §3.1.3.

Karpukhin et al. (2020); Zhu et al. (2021). For EM score, an answer is deemed correct if its normalized form – obtained through the normalization procedure delineated by (Karpukhin et al., 2020) – corresponds to any acceptable answer in the provided list. Similar to EM score, F1 score treats the prediction and ground truth as bags of tokens, and computes the average overlap between the prediction and ground truth answer (Chen et al., 2017). Besides, we use reject rate (RR) to evaluate the unknown robustness when given questions beyond a language model’s knowledge scope.

Finally, since GPT-4 is not directly trained on open-domain QA benchmarks, employing EM / F1 for evaluation is challenging. Therefore, we adopt the approach outlined in Mallen et al. (2023); Kandpal et al. (2023), utilizing accuracy as the evaluation metric. Accuracy considers a prediction correct if any substring of the prediction exactly matches any of the provided correct answers.

### 3.2 Evaluation on Overall QA Performance
Table 2 demonstrates that the RALM consistently outperforms the directly fine-tuned LLaMa-2 with QA pairs, without retrieval. This improvement is closely tied to the effectiveness of the retrieval process. As indicated in Table 1, DPR demonstrates markedly superior retrieval performance on the NQ and TriviaQA datasets compared to WebQ. Consequently, the benefits of retrieval are more pronounced on NQ and TriviaQA. Furthermore, when comparing our enhanced RALM, which integrates CoN, with the standard RALM, our method persistently shows better performance. There is an average improvement of +1.97 in EM scores across all three datasets when using LLaMa-2 as backbone language model. Delving deeper, we find that this improvement varies depending on whether DPR successfully retrieves relevant documents. Specifically, the average improvement is +1.2 when DPR retrieves relevant documents and +2.3 when it does not on the NQ dataset. This disparity suggests that our CoN improve RALM’s in scenarios where more noisy documents are fetched in the first retrieval stage. This observation aligns with our findings on noise robustness, which are elaborated in the subsequent sections detailing our experimental results.

Furthermore, the dynamics observed with larger language models differ from those noted in experiments with smaller-sized models due to their superior factual knowledge. The impact of utilizing retrieval is observed to be less pronounced with larger models and can even be detrimental in certain cases, such as with TriviaQA, where questions are mostly straightforward. Concerning the comparison between CoN and the baseline, the performance trend remains consistent with that observed in smaller-sized models, suggesting that CoN maintains its significance across different model sizes.
```

### --- Page 0007 ---

```markdown
| Models                                   | Noise Ratio | NQ EM  | NQ F1  | TriviaQA F1 | WebQ F1 | Average EM | Average F1 |
|------------------------------------------|-------------|--------|--------|--------------|---------|------------|------------|
| Retrieve-Read + CHAIN-OF-NOTE           | 100%        | 34.28  | 41.74  | 55.30        | 61.67   | 29.58      | 46.34      |
|                                          |             | (±7.55)| (±7.84)| (±9.00)      | (±8.33) | (±7.27)    | (±6.73)    |
| Retrieve-Read + CHAIN-OF-NOTE           | 80%         | 54.28  | 61.03  | 73.83        | 80.00   | 35.46      | 52.70      |
|                                          |             | (±2.35)| (±2.20)| (±2.46)      | (±2.14) | (±5.14)    | (±3.84)    |
| Retrieve-Read + CHAIN-OF-NOTE           | 60%         | 61.44  | 67.94  | 78.44        | 83.65   | 37.01      | 54.16      |
|                                          |             | (±1.99)| (±1.39)| (±0.35)      | (±0.42) | (±2.25)    | (±2.20)    |
| Retrieve-Read + CHAIN-OF-NOTE           | 40%         | 64.62  | 71.12  | 80.56        | 86.76   | 38.40      | 55.60      |
|                                          |             | (±1.29)| (±1.10)| (±1.16)      | (±0.35) | (±3.76)    | (±2.07)    |
| Retrieve-Read + CHAIN-OF-NOTE           | 20%         | 67.21  | 73.69  | 81.73        | 87.89   | 39.95      | 56.66      |
|                                          |             | (±2.79)| (±2.39)| (±1.13)      | (±0.35) | (±4.41)    | (±3.47)    |
| Retrieve-Read + CHAIN-OF-NOTE           | 0%          | 69.23  | 75.57  | 83.34        | 89.44   | 42.24      | 58.59      |
|                                          |             | (±4.05)| (±4.29)| (±0.18)      | (±0.50) | (±3.92)    | (±2.72)    |

![Evaluation on Noise Robustness](assets/page_0007_img_1.png)

| Models                                   | RealTimeQA EM | F1   | RR  |
|------------------------------------------|---------------|------|-----|
| Retrieve-Read (Shi et al., 2023c)       | 15.6          | 19.9 | 6.1 |
| + CHAIN-OF-NOTE (ours)                  | 15.7          | 20.3 | 13.0|

### 3.3 Evaluation on Noise Robustness

As illustrated in Table 2, when faced with entirely noisy documents, both the standard RALM and our CHAIN-OF-NOTE enhanced RALM underperformed compared to the non-retrieval setting. This suggests that RALMs can be misled by noisy information, leading to more hallucinations.

Notably, equipping the model with CoN enables it to perform nearly as well as the baseline model directly fine-tuned with QA pairs without retrieval, showcasing its robustness to noise and its ability to disregard irrelevant information. The CoN approach is effective not only in fine-tuned, smaller-sized models but also in large language models, such as GPT-4, with adjustments made only to the prompt. Besides, in comparison to the CHAIN-OF-THOUGHT technique, commonly utilized in reasoning scenarios, CoN presents a more efficient strategy for retrieval-augmented settings, particularly in addressing knowledge-intensive tasks.

### 3.4 Evaluation on Unknown Robustness

Table 4 illustrates that our RALM equipped with CoN exhibits superior robustness in handling unknown scenarios, particularly evident in the RealTimeQA benchmark. This benchmark falls completely outside the model’s domain and contains real-time information that was not part of the LLaMa-2 pre-training data. Despite this, models are still capable of providing correct answers in some cases, as the answers remain consistent over time. In comparison to the standard RALM system, our method shows a significant improvement, exceeding +10.5 in its ability to reject to answer questions in unknown scenarios. The evaluation is based on reject rate (RR), i.e., number of rejected
```

### --- Page 0008 ---

```markdown
![Using a hybrid training strategy demonstrates slightly lower robustness across various noise ratios but consistently better performance than standard RALMs.](assets/page_0008_img_1.png)

| Models ↓                     | Inference Time(s) |
|------------------------------|--------------------|
| Retrieve-Read                | 0.6104             |
| + CHAIN-OF-NOTE              | 12.0192            |
| + CHAIN-OF-NOTE (hybrid)     | 0.6074             |

### 3.5 Evaluation on Hybrid Training Strategy

As illustrated in Figure 3 and Table 5, our proposed RALM equipped with a hybrid strategy demonstrates slightly lower robustness across various noise ratios while keeping similar efficient decoding time consumption to the standard RALM. This indicates that our CHAIN-OF-NOTE framework, when implemented with a hybrid training strategy, is highly applicable to a wide range of real-world business scenarios. This enhancement in robustness without significant time overhead highlights the practical value and efficiency of our approach, making it a viable solution for environments where QA accuracy can vary but inference time is crucial.

## 4 Related Work

Retrieval-Augmented Language Models (RALMs) represent a significant advancement in natural language processing, combining the power of large language models with the specificity and detail provided by external knowledge sources (Guu et al., 2020; Lewis et al., 2020; Izacard et al., 2022). Recent studies highlight the impact of context relevance on language model performance (Creswell et al., 2022; Shi et al., 2023a; Yoran et al., 2023). Notably, Creswell et al. (2022) demonstrated that incorporating random or irrelevant contexts could adversely affect QA performance. In contrast, Shi et al. (2023a) discovered that adding irrelevant context to exemplars or task-specific instructions can sometimes enhance model performance, implying that models might intrinsically possess capabilities, developed during pre-training, to manage such scenarios. Most pertinent to our research is the study by Yoran et al. (2023), which focused on training RALMs to disregard irrelevant contexts. This approach, while distinct from our proposed solution, underscores the importance of context relevance in enhancing the effectiveness of RALMs.

Besides, we present more related Chain-of-Xs approaches (e.g., Chain-of-Thought (CoT) (Wei et al., 2022)) in the Appendix A.1.1 and A.2.

## 5 Conclusion

In this paper, we introduce the CHAIN-OF-NOTE (CoN) framework, a novel methodology designed to enhance the robustness of RALMs. The central concept of CoN revolves around the generation of sequential reading notes for each retrieved document. This process allows for an in-depth assessment of document relevance to the posed question and aids in synthesizing this information to craft the final answer. Our experiments show that GPT-4, when equipped with CoN, outperforms the CHAIN-OF-THOUGHT approach. Besides, we utilized GPT-4 to create 10K CoN data, subsequently trained on a LLaMa-2 7B model. Our experiments across four open-domain QA benchmarks show that RALMs equipped with CoN significantly outperform standard fine-tuned RALMs.

## 6 Limitations

One major limitation of the CHAIN-OF-NOTE (CoN) approach is its increased inference cost due to the sequential generation of notes. While CoN is beneficial for assessing the relevance and integrating external knowledge, it results in longer response times, which is problematic for time-sensitive applications. Moreover, the system’s efficiency depends on the conciseness and relevance of the generated notes, which can fluctuate based on the complexity of the retrieved documents.
```

### --- Page 0009 ---

```markdown
# References

Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic parsing on freebase from question-answer pairs. In **EMNLP**, pages 1533–1544.

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George BM Van Den Driesche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022. Improving language models by retrieving from trillions of tokens. In **International conference on machine learning**, pages 2206–2240. PMLR.

Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading wikipedia to answer open-domain questions. In **Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)**, pages 1870–1879.

Hao Cheng, Yelong Shen, Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. 2021. Uniteqa: A hybrid approach for open domain question answering. In **Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)**, pages 3080–3090.

Antonia Creswell, Murray Shanahan, and Irina Higgins. 2022. Selection-inference: Exploiting large language models for interpretable logical reasoning. arXiv preprint arXiv:2205.09712.

Bhuvan Dhingra, Jeremy R Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W Cohen. 2022. Time-aware language model as temporal knowledge bases. **Transactions of the Association for Computational Linguistics**, 10:257–273.

Shehzad Dhuliawala, Mojtaba Koemlili, Jing Xu, Robert Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2023. Chain-of-verification reduces hallucination in large language models. arXiv preprint arXiv:2309.11945.

Kelvin Guo, Kenton Lee, Zora Tung, Panungpop Pasupat, and Ming-Wei Chang. 2020. Realm: Retrieval-augmented language model pre-training. arXiv preprint arXiv:2002.08909.

Fan Huang, Haewoon Kwak, and Jisun An. 2023. Chain of explanation: New prompting method to generate quality natural language explanation for implicit hate speech. In **Proceedings of the ACM Web Conference 2023**, pages 90–93.

Gautier Izacard and Edouard Grave. 2021. Leveraging passage retrieval with generative models for open domain question answering. In **EACL**, pages 874–880.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseni, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022. Few-shot learning with retrieval augmented language models. arXiv preprint arXiv:2208.03299.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascal Fung. 2023. Survey of hallucination in natural language generation. **ACM Computing Surveys**, 55(12):1–38.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In **ACL**, pages 1601–1611.

Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. 2023. Large language models struggle to learn long-tail knowledge. In **International Conference on Machine Learning**, pages 15696–15707. PMLR.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open domain question answering. In **Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)**, pages 6769–6781.

Jungo Kasai, Keisuke Sakaguchi, Yoichi Takashima, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A Smith, Yejin Choi, and Kentaro Inui. 2023. Realtime qa: What’s the answer right now? **Advances in Neural Information Processing Systems**.

Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020. Generalization through memorization: Nearest neighbor language models. In **International Conference on Learning Representations**.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916.

Tom Kwiatkowski, Jenni Rinaldi, Jasminka Paliakou, Vidya Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epsten, Ilia Polosukhin, Jacob Devlin, Ken Lee, et al. 2019. Natural questions: A benchmark for question answering research. **TACL**, pages 452–466.

Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Goyahl. 2022. Internet-augmented language models through few-shot prompting for open-domain question answering. arXiv preprint arXiv:2203.05115.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. **Advances in Neural Information Processing Systems**, 33:9459–9474.
```

### --- Page 0010 ---

```markdown
| **Authors**                                                                 | **Title**                                                                                          |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Xianzhi Li, Xiaodan Zhu, Zhiqiang Ma, Xiaomu Liu, and Sameena Shah. 2023.  | Are chatgpt and gpt-4 general-purpose solvers for financial text analytics? an examination on several typical tasks.  [arXiv preprint arXiv:2305.05862](https://arxiv.org/abs/2305.05862) |
| Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James Glass. 2022. | Sail: Search-augmented instruction learning. [arXiv preprint arXiv:2305.15225](https://arxiv.org/abs/2305.15225) |
| Ji Ma, Ivan Korkotkov, Yifei Yang, Keith Hall, and Ryan McDonald. 2021.   | Zero-shot neural passage retrieval via domain-targeted synthetic question generation. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 1075–1088. |
| Kaixin Ma, Hao Cheng, Yu Zhang, Xiaodong Liu, Eric Nyberg, and Jianfeng Gao. 2023. | Chain-of-skills: A configurable model for open-domain question answering. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*. |
| Alex Melian, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannah Hajishirzi. 2023. | When not to trust language models: Investigating the effectiveness of parametric and non-parametric memories. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 9802–9822. |
| OpenAI. 2023. | GPT-4 technical report. [arXiv preprint arXiv:2303.08774](https://arxiv.org/abs/2303.08774) |
| Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiao Chen, Michihiro Yasunaga, and Diyi Yang. 2023. | chatgpt a general-purpose natural language processing task solver? [arXiv preprint arXiv:2302.06476](https://arxiv.org/abs/2302.06476) |
| Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. | Rocketap: An optimized training approach to dense passage retrieval for open-domain question answering. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 5835–5847. |
| Jeff Rasley, Samya Rajbhandari, Olaitan Ruwaise, and Yuxiong He. 2020. | Deepseed: System optimizations enable training deep learning models with over 100 billion parameters. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 3055–3066. |
| Devendra Singh Sachan, Mike Lewis, Dani Yogatama, Luke Zettlemoyer, Joelle Pineau, and Manzil Zaheer. 2022. | Questions are all you need to train a dense passage retriever. [arXiv preprint arXiv:2206.10658](https://arxiv.org/abs/2206.10658) |
| Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathaniel Schärli, and Denny Zhou. 2023a. | Large language models can be easily distracted by irrelevant context. In *International Conference on Machine Learning*, pages 31210–31227. PMLR. |
| Freda Shi, Mirac Suzgur, Markus Freitag, Xuezhi Wang, Suraj Srivats, Sorush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou. 2023b. | Language models are multilingual chain-of-thought reasoners. In *The Eleventh International Conference on Learning Representations*. |
| Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023c. | Retrieval-augmented black-box language models. [arXiv preprint arXiv:2301.12652](https://arxiv.org/abs/2301.12652) |
| Devendra Singh, Siva Reddy, Will Hamilton, Chris Dyer, and Daini Yogatama. 2021. | End-to-end training of multi-document reader and retriever for open-domain question answering. *Advances in Neural Information Processing Systems*, 34:29658–29831. |
| Hugo Touvron, Louis Martin, Kevin Stone, Peter Alrf, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargava, Shruti Bhosele, et al. 2023. | Llama 2: Open foundation and fine-tuned chat models. [arXiv preprint arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |
| Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. | Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*. |
| Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, et al. 2023. | Refreshing large language models with search engine augmentation. [arXiv preprint arXiv:2310.03214](https://arxiv.org/abs/2310.03214) |
| Jianing Wang, Qiushi Sun, Nuo Chen, Xiang Li, and Ming Gao. 2023a. | Boosting language models responding with chain-of-knowledge prompting. [arXiv preprint arXiv:2306.06427](https://arxiv.org/abs/2306.06427) |
| Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong, and Zhang Xiong. 2023b. | Knowledge-driven: Exploring faithful reasoning in LMs for knowledge-intensive question answering. [arXiv preprint arXiv:2302.13829](https://arxiv.org/abs/2302.13829) |
| Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. | Chain of thought prompting enables reasoning in large language models. [arXiv preprint arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| Ori Yoran, Tomer Wolfson, Ofir Ran, and Jonathan Berant. 2023. | Making retrieval-augmented language models robust to irrelevant context. [arXiv preprint arXiv:2310.01558](https://arxiv.org/abs/2310.01558) |
```

### --- Page 0011 ---

```markdown
Donghan Yu, Chenguang Zhu, Yuwei Fang, Wenhao Yu, Shuohang Wang, Yichong Xu, Xiang Ren, Yiming Yang, and Michael Zeng. 2022. Kg-fid: Infusing knowledge graph in fusion-in-decoder for open-domain question answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4961–4974.

Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Yu, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2023a. Generate rather than retrieve: Large language models are strong context generators. International Conference for Learning Representation (ICLR).

Wenhao Yu, Zhihan Zhang, Zhenwen Liang, Meng Jiang, and Ashish Sabharwal. 2023b. Improving language models via plug-and-play retrieval feedback. arXiv preprint arXiv:2305.14002.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingting Fu, Yinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. 2023a. Siren’s song in the ai ocean: A survey on hallucination in large language models. arXiv preprint arXiv:2309.01219.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 2023b. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923.

Zexuan Zhong, Tao Lei, and Danqi Chen. 2022. Training language models with memory augmentation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5657–5673.

Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and Tat-Seng Chua. 2021. Retrieving and reading: A comprehensive survey on open-domain question answering. arXiv preprint arXiv:2101.00774.
```

### --- Page 0012 ---

```markdown
# A Appendix

## A.1 More Related Work

### A.1.1 Retrieval-Augmented Language Models

Retrieval-Augmented Language Models (RALMs) represent a significant advancement in natural language processing, combining the power of large language models with the specificity and detail provided by external knowledge sources (Guu et al., 2020; Lewis et al., 2020; Izacard et al., 2022). These models first leverage a retriever to scan a vast evidence corpus, such as Wikipedia, to identify a set of documents pertinent to the user’s query. Following this, a reader component is employed to meticulously analyze these documents and formulate a response. This two-pronged approach ensures both relevance and depth in the generated answers. Recent follow-up work has mainly focused on improving the retriever (Karpukhin et al., 2020; Qu et al., 2021; Sachan et al., 2022; Ma et al., 2023) or the reader (Izacard and Grave, 2021; Cheng et al., 2021; Yu et al., 2022), training the system end-to-end (Lewis et al., 2020; Singh et al., 2021), and integrating the retrieval systems with large-scale black-box language models (Yu et al., 2023a; Shi et al., 2023b; Yu et al., 2023b; Trivedi et al., 2023). Another line of RALMs such as kNN-LM (Khandewal et al., 2020; Zhong et al., 2022) retrieves a set of tokens and interpolates between the next token distribution and kNN distributions computed from the retrieved tokens at inference. The evolution has also led to the emergence and popularity of retrieval-augmented products, such as ChatGPT plugin, Langchain, and New Bing.

### A.2 Chain-of-X Approaches in Large Language Models

Recent research shows that large language models (LLMs) are capable of decomposing complex problems into a series of intermediate steps, pioneered by the concept of Chain-of-Thought (CoT) prompting (Wei et al., 2022; Kojima et al., 2022). The CoT approach mirrors human problem-solving methods, where complex issues are broken down into smaller components. By doing so, LLMs can tackle each aspect of a problem with focused attention, reducing the likelihood of overlooking critical details or making erroneous assumptions. This sequential breakdown makes the reasoning process more transparent, allowing for easier identification and correction of any logical missteps.

The CoT methodology has been effectively applied in various contexts, including multi-modal reasoning (Zhang et al., 2023b), multi-lingual scenarios (Shi et al., 2023b), and knowledge-driven applications (Wang et al., 2023b). Additionally, there has been a surge in the development of other chain-of-X methods, addressing diverse challenges in LLM applications. These include chain-of-explanation (Huang et al., 2023), chain-of-knowledge (Wang et al., 2023a), chain-of-verification (Dhuliwala et al., 2023) and IR chain-of-thought (Trivedi et al., 2023). For instance, Chain-of-Verification (Dhuliwala et al., 2023) generates an initial response, formulates verification questions, and revises the response based on these questions, reducing factual errors and hallucinations in the response. Closely related to our work is IR chain-of-thought (Trivedi et al., 2023), which employs CoT to infer and supplement unretrieved information, thereby improving the accuracy of complex reasoning tasks. While chain-of-X approaches have shown promise in enhancing LLMs' performance across various domains, their application in RALMs, particularly for improving robustness in noisy and unknown scenarios, is relatively unexplored. This gap signifies further research in applying these strategies to augment RALMs, thereby enhancing their robustness and reliability.

## A.3 Dataset Information

- **TriviaQA** (Joshi et al., 2017) contains a set of trivia questions with answers originally scraped from trivia and quiz-league websites.
- **WebQ** (Berant et al., 2013) consists of questions selected using Google Suggest API, where the answers are entities in Freebase.
- **NQ** (Kwiatkowski et al., 2019) were collected from real Google search queries and the answers are one or multiple spans in Wikipedia articles identified by human annotators.

## A.4 Implementation Details

In the retrieval phase, we employed DPR (Karpukhin et al., 2020) to retrieve documents from Wikipedia. We accessed the model via direct loading from the official DPR repository hosted on GitHub. Subsequent to retrieval, our fine-tuning process for the LLaMA-2 (Touvron et al., 2023) model runs for 3 epochs with a batch size set to 128, leveraging the DeepSpeed library (Rasley et al., 2020) and the ZeRO optimizer (Ma et al., 2021), with
```

### --- Page 0013 ---

```markdown
bfloat16 precision. The learning rates are set to $\{1e-6, 2e-6, 5e-6, 1e-5, 2e-5\}$, and the empirical results indicated that $5e-6$ yielded the best model performance, hence we standardized the learning rate for all reported numbers. Greedy decoding is applied during inference on all experiments to ensure deterministic generations.

## A.5 Instruction Prompts

(1) For standard RALM, the instruction is:  
**Task Description:** The primary objective is to briefly answer the question.

(2) For RALM with CoT, the instruction is:  
**Task Description:** Think step by step to answer the question.

(3) For RALM with CoN, the instruction is:  
**Task Description:**  
1. Read the given question and five Wikipedia passages to gather relevant information.  
2. Write reading notes summarizing the key points from these passages.  
3. Discuss the relevance of the given question and Wikipedia passages.  
4. If some passages are relevant to the given question, provide a brief answer based on the passages.  
5. If no passage is relevant, directly provide an answer without considering the passages.

## A.6 Case Studies

In our case studies, as illustrated in Table 6, we compare the responses generated by the standard RALM and our enhanced RALM with CoT. These examples highlight the differences in how each model processes and interprets information from retrieved documents.

The first case shows a question pertains to the most recent Summer Olympics held in the USA. The standard RALM is misled by the mention of "Chicago’s bid for the 2016 Summer Olympics." Lacking a deep comprehension of the content, it incorrectly focuses on the more recent year (2016), resulting in an inaccurate answer. In contrast, the RALM with CoN carefully analyzes the information. It notes that while Chicago bid for the 2016 Olympics, there’s no confirmation of it being a successful bid. This leads to the correct conclusion that the most recent Olympics in the USA were held in 1996. The second case study involves identifying the language of the first Jnanpith Award recipient. Here, the standard RALM fails to synthesize information across documents. It identifies G. Sankara Kurup as the award recipient but does not connect this information to the language of his work. Conversely, the RALM with CoN effectively combines details from both documents. It recognizes that while the first document mentions Kurup’s award, the second document provides the missing language detail, leading to the correct answer of Malayalam.

## A.7 Licenses

The four open-domain QA benchmarks, LLaMa-2 models are all released under MIT License. They are all for research purposes, and our experiments are consistent with their intended usage.
```

### --- Page 0014 ---

```markdown
| Question: When was the last Summer Olympics in the USA? | Answer: 1996 |
|--------------------------------------------------------|---------------|
| Wikipedia document #1: In Chicago’s 2016 Summer Olympics bid, the games would have been held from July 22 to August 7, with the Paralympics held between August 12 and 28. The plan emphasized use of Chicago Park District ... (Wikipedia page: Chicago bid for the 2016 Summer Olympics) | Wikipedia document #2: The 1996 Olympics are the most recent edition of the Summer Olympics to be held in the United States. Los Angeles will host the 2028 Summer Olympics, 32 years after the games were held in Atlanta. ... (Wikipedia page: 1996 Summer Olympics) |
| (Other retrieved Wikipedia documents are omitted.) | |
| Standard RALM: 2016  |  X |
| RALM with CHAIN-OF-NOTE: Wikipedia passage #1 discusses Chicago’s bid for the 2016 Summer Olympics, but does not mention whether it is a successful bid. Wikipedia passage #2 confirms that the 1996 Summer Olympics were the most recent edition of the Summer Olympics to be held in the United States. Based on the information given in above passages, the answer is 1996 |

| Question: The first Jnanpith Award recipient was an author in which language? | Answer: Malayalam |
|--------------------------------------------------------|---------------|
| Wikipedia document #1: The first Jnanpith Award was given to G. Sankara Kurup in 1965 for his collection of poems, Odakkuzhal (... (Wikipedia page: Jnanpith Award) | Wikipedia document #2: ... The final round had four authors; Kazi Nazrul Islam (Bengali), D. V. Gundappa (Kannada), Viswanatha Satyanarayana (Telugu), and G. Sankara Kurup (Malayalam). On 19 November 1966, Kurup presented with... (Wikipedia page: Jnanpith Award) |
| (Other retrieved Wikipedia documents are omitted.) | |
| Standard RALM: G. Sankara Kurup  |  X |
| RALM with CHAIN-OF-NOTE: Wikipedia passage #1 states that the first Jnanpith Award was given to G. Sankara Kurup in 1965 for his collection of poems, Odakkuzhal. Wikipedia passage #2 mentions that the Jnanpith Award was given to G. Sankara Kurup in 1966 for his work in Malayalam. Based on the information given in above passages, the answer is Malayalam ✔ |

| Table 6: Case Study. Compared to Standard RALM, our RALM with CHAIN-OF-NOTE exhibits a deeper understanding of how documents reveal information relevant to the question. It goes beyond merely capturing surface-level terms, leading to more accurate responses. |
```



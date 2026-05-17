# ArXiv 2310.12150

### --- Page 0001 ---

```markdown
# Understanding Retrieval Augmentation for Long-Form Question Answering

**Hung-Ting Chen, Fangyuan Xu*, Shane Arora*, Eunsol Choi**  
Department of Computer Science  
University of Texas at Austin  
{hungtingchen, fangyuan, shane.arora, eunsol}@utexas.edu  

---

## Abstract

How retrieved documents are used in language models (LMs) for long-form generation task is understudied. We present two controlled studies on retrieval-augmented LM for long-form question answering (LFQA): one fixing the LM and varying evidence documents and the other fixing evidence documents and varying the LMs. We study various attributes of generated answers (e.g., fluency, length, variance), with an emphasis on the attribution of generated answers to in-context evidence documents. We collect a dataset (SALAD) containing human annotations of sentence-level answer attribution in LFQA and evaluate existing methods for automatically judging attribution. We find that while LMs can leverage relevant in-context documents, the generated answer is only partially attributable towards the documents, especially for LMs trained without retrieval augmentation. Together, our analysis reveals how retrieval augmentation impacts long knowledge-rich text generation and provide directions for future work.

## 1 Introduction

Recent works (Nakano et al., 2021; Malaviya et al., 2023; Gao et al., 2023) proposed retrieval augmentation as a powerful tool to provide up-to-date, relevant information to LMs for long-form answer generation. Yet, retrieval augmentation does not always affect LMs the way we anticipate. Liu et al. (2023a) discovered that information placed in the middle of contexts is not used by LMs. Parameter knowledge continues to affect generation even when only irrelevant documents are provided in-context for factoid QA task (Longpre et al., 2021; Chen et al., 2022). These findings, however, are based on analyses on factoid QA with short answer spans, which is easier to evaluate. Our understanding of how retrieval augmentation impacts long-form generation in LMs is limited.

We study how retrieval impacts answer generation for LFQA, a complex long-text generation task. We study two settings (illustrated in Figure 1): (1) fixing the LM and varying the degree of relevance of evidence documents and (2) fixing evidence documents and varying the LMs. As evaluating the quality of long-form answers is notoriously difficult (Krishna et al., 2021), we start our analysis by measuring surface features (e.g., length, perplexity) that correlate with specific answer qualities such as coherence (Xu et al., 2023).

Our analysis reveals that retrieval augmentation changes LM’s generation substantially. Some effects, e.g., change in the length of answers, are pronounced even when provided documents are irrelevant. Relevant in-context evidence documents lead to more substantial changes, leading LMs to generate more unexpected sentences (measured by higher perplexity), while irrelevant documents do not have the same effects. Surprisingly, the impact of retrieval augmentation, even with the same set of evidence documents, can result in opposite effects for different base LMs (e.g. change in answer lengths and lexical diversity).

One desirable property of retrieval-augmented LFQA system is to have the answer attributed to the evidence documents. To evaluate this, we collect SALAD, a dataset with Sentence-
```

### --- Page 0002 ---

```markdown
![Figure 1: We study (A) how differing LMs use the same in-context evidence documents to generate answer and (B) how in-context documents of various degree of relevance affect the answer generation. We analyze generated answers on surface patterns (self-BLEU, perplexity, etc) and their attribution to evidence documents. Attribution judgements are made per sentence, either by annotators (Section 5) or automatically from NLI model (Section 7). O's, A's and X's denote supported, partially supported and unsupported sentences respectively. Colored texts are generated texts not supported by in-context evidence documents.](assets/page_0002_img_1.png)

## 2 Background and Related Work

### LFQA
LFQA (Fan et al., 2019; Stelmach et al., 2022) requires models to generate paragraph-length answers to complex, open-ended questions. To address this, WebGPT (Nakano et al., 2021) introduces a web agent that searches the web and integrates relevant information to LMs. We evaluate this model closely in this study.

### Retrieval Augmentation
Retrieval-augmented generation has received attention as a way to provide up-to-date, relevant documents to LMs at inference time (Ram et al., 2023), showcasing performance gains across multiple tasks (Shi et al., 2023). A line of work investigates how LMs incorporate in-context documents (Mallen et al., 2022; Liu et al., 2023) with their parametric knowledge on simpler tasks such as factoid QA. Wang et al. (2023) studies the impact of retrieval in open-ended text generation with KNN-LM (Khandelwal et al., 2019).
```

### --- Page 0003 ---

```markdown
We focus on LFQA, which requires factual, attributable output over long sequences. Prior work (Krishna et al., 2021) also analyzed attribution in LFQA but studied smaller LLMs fine-tuned with in-domain data, whose attribution pattern varies significantly from the models we study.

## Evaluating Attribution

We focus our analysis on the attribution of long-form answers with respect to the prepended evidence document set. We follow the AIS framework Raskin et al. (2021), an evaluation framework for whether a system-generated text can be derived by a given knowledge source. Bohnet et al. (2022) and Yue et al. (2023) study automatically evaluating attribution; the former uses off-the-shelf entailment models, while the latter prompts and fine-tune LLMs. Gao et al. (2023b) builds QA models that generate text along with citations and evaluates the citation quality of the generations automatically. Bohnet et al. (2022) presents a controlled study of attribution (e.g., varying evidence documents and how they impact attribution) on factoid QA with Wikipedia as retrieval corpus.

Recent work (Liu et al., 2023b) annotates attribution quality in long-form answers generated from commercial generative search engines. While they provide a comprehensive study on attribution quality with manual annotations, their study on black box models is limited, as they do not have knowledge of how the cited documents were integrated into the LMs. For instance, documents could have been retrieved post hoc (Gao et al., 2023a) or prepended in context. We instead present a controlled study involving open-source models, and analyze their data in Section 6.

## 3 Study Setting

We conduct a controlled study on how retrieval augmentation impacts long-form answer generation for LMs, observing surface features and attribution using manually annotated evidence document sets and LMs. In this section, we describe our experimental setting.

### Dataset

We source questions from ELI5 dataset (Fan et al., 2019), which contains questions from the Reddit forum “Explain Like I’m Five”. We use the entire test set released by WebGPT (Nakano et al., 2021) (271 questions) for automatic evaluation (Sec. 4 and Sec. 7.2), and randomly sample 100 questions to collect manual attribution annotations (Sec. 5).

### Knowledge Source: Evidence Documents

For each question, we compile four sets of evidence documents to examine how models use documents of varying degrees of relevance. Each document set $D$ contains 3-4 document snippets, each containing roughly 100 words. The statistics on each set can be found in Appendix A.1. We detail each document set below:

- **Human Demonstration**: Annotators from prior study (Nakano et al., 2021) used a commercial search engine (Bing) to gather documents to answer questions. We include these as “gold” documents that are considered relevant for answering questions by humans.
- **WebGPT Retriever**: We consider documents retrieved by the WebGPT (175B) (Nakano et al., 2021) model. Their study found using these documents results in high-quality answer generation.
- **Bing Search**: We retrieve relevant documents using Bing Search API with the question as the query, and obtain the top 10 pages returned by the API, and retrieve four 100-word segments from aggregate search results. Post-processing details are in Appendix A.2.
- **Random**: To simulate a set of irrelevant documents, we randomly sample another question in the test set and take the corresponding documents retrieved by WebGPT.

We evaluate the relevance of the first three sets of documents manually by sampling 20 questions and examining the document sets. We find that WebGPT, human demonstration and Bing documents contain sufficient information to answer the question for 85%, 50% and 45% of the examples, respectively. Details on the manual study are in Appendix B.6.

### Base LMs & Answer Generation

We consider three LMs: WebGPT(175B) (Nakano et al., 2021), GPT-3.5 (text-davinci-003) (Brown et al., 2020) and Alpaca (Taori et al., 2023). The 3
```

### --- Page 0004 ---

```markdown
Published as a conference paper at COLM 2024

| Model (d evidence) | # Sentences | # Words | RankGen (↑) | Self-BLEU (↓) | Perplexity (↓) |
|--------------------|-------------|---------|-------------|----------------|-----------------|
| WebGPT+WebGPT docs | 6.7 / 1.9   | 160     | 31.35       | 0.58           | 1.81            |
| GPT-3.5            | 9.3 / 15.26 | 219 / 53| 12.77 / 0.87| 0.71           | 6.13            |
| +Human docs        | 6.6 / 6.18  | 172 / 41| 11.89 / 0.16| 0.62           | 10.49           |
| +WebGPT docs       | 6.8 / 9.18  | 185 / 41| 11.97 / 0.19| 0.62           | 11.63           |
| +Bing docs         | 6.9 / 10.19 | 179 / 39| 12.13 / 0.19| 0.64           | 9.03            |
| +Random docs       | 7.6 / 11.21 | 183 / 39| 12.40 / 2.13| 0.68           | 6.70            |
| Alpaca-7b          | 5.0 / 8.1   | 113 / 23| 12.17 / 2.00| 0.51           | 11.95           |
| +Human docs        | 5.7 / 19.36 | 134 / 79| 11.82 / 2.32| 0.55           | 12.99           |
| +WebGPT docs       | 6.2 / 3.79  | 145 / 50| 11.91 / 0.27| 0.55           | 12.37           |
| +Bing docs         | 7.6 / 8.50  | 167 / 60| 12.04 / 2.05| 0.59           | 10.81           |
| +Random docs       | 5.2 / 6.53  | 121 / 26| 12.57 / 1.99| 0.53           | 11.92           |
| Human† (Human docs)| 5.1 / 2.7   | 119 / 59| 9.29 / 4.37 | 0.49           | 7.67            |

Table 1: Generated answer statistics. We present means and two standard deviations in its subscript: one computed over three answers generated for the same example, one over answers for different examples. Numbers in red and blue indicate decrease and increase from the baseline respectively. We boldface rows which have human annotations for attribution (Section 5).

WebGPT model is trained to interact with a semantically search engine (Bing) and compiles long-form answers based on the information gathered from Bing for questions from the ELI5 dataset.1 We experimented with two types of answer-sourced LMs [GPT-1 (Kawasaki & Komatsuzaki, 2021) (6B), Flan-T5 (Chung et al., 2022) (11B), Llama (Touvron et al., 2023) (7B, 13B, 30B) and Alpaca (Taori et al., 2023)] and aim to be the best-performing human manual examination.2 The prediction examples for all other LMs can be found in Appendix B.5. We prepended the contextual evidence document set to the question and provided it as a prompt to LMs with a brief instruction. We sample three answers for each setting to study answer variability. The decoding hyperparameters and prompts can be found in Appendix A.3.

## 4 How In-Context Documents Impact Surface Answer Statistics

Metrics Unlike evaluating short, mostly entity-based answers (Rajpurkar et al., 2016; Fisch et al., 2021), evaluating the overall quality of long-form answers (Krishna et al., 2021; Xu et al., 2023) is notoriously difficult for both humans and models. In this section, we look at metrics that have been shown to correlate with specific aspects (e.g., fluency, coherence) (Xu et al., 2023) of answers, to quantify differences between answers.

- **Length:** We report the number of sentences and number of words in the answer paragraph. The length is shown as a significant contributing factor in human evaluation for various tasks, with humans often preferring longer answers (Sun et al., 2019; Liu et al., 2022; Xu et al., 2023).
- **Self-BLEU** (Zhu et al., 2018) measures the lexical diversity of generated text. An answer is less diverse and contains more repetition if it has a higher Self-BLEU score. Prior work Xu et al. (2023) also found that a lower Self-BLEU correlates to better coherence.
- **RankGen** (Krishna et al., 2022) is a T5-XXL encoder trained with large-scale contrastive learning, ranking generation given a prefix. A higher RankGen score signifies a more likely continuation of the prefix. We measure RankGen score with question as the prefix.
- **Perplexity:** We report the perplexity of the answer measured with GPT-2-X (Radford et al., 2019). Lower perplexity generally indicates more fluent generated text, though human-written texts (Holtzman et al., 2019) do not necessarily exhibit lower perplexity compared to model-generated text.

1 While their model is not released, the model outputs were provided at https://openpublicb1. lob.core.windows.net/webgpt-answer-viewer/index.html  
2 This is likely because ELI5 was one of the seed task used to generate training data for Alpaca.
```

### --- Page 0005 ---

```markdown
# Results

Table 1 presents the statistics for answers. Overall, prepending relevant documents yields bigger changes for both GPT-3.5 and Alpaca compared to prepending random documents. Prepending unrelated documents has little effect on the automatic metrics for Alpaca, but impacts the generation of GPT-3.5, especially in length and Self-BLEU. This might be related to instruction tuning enabling LMs (Alpaca in this case) to be more robust to irrelevant prompts (Webson & Pavlick, 2022). We report the results on GPT-4 (which shows the same trend as GPT-3.5) in Appendix B.4, and seven other LMs in Appendix B.5.

Using the same evidence documents brings different effects on different LMs. On GPT-3.5, providing documents results in shorter outputs and less repetitions, while on Alpaca, it results in longer outputs and more repetitions. Yet, on both models, adding relevant documents causes bigger changes in length than adding random documents. Overall, GPT-3.5 generate longer answers with less variability across examples. Alpaca answers exhibit higher variance across examples across all metrics.

For both models, RankGen scores decrease when the document set is more relevant. This can be as the model incorporates new information from retrieved documents, generation becomes less predictable from the question alone. Perplexity also shows similar trends, with relevant documents increasing perplexity. This might be because it copies rare tokens from documents, which will get assigned high perplexity when evaluating answers alone.

## Models can differentiate random and relevant documents.

In our experiments, answers generated with random documents are the most similar to answers generated without documents, see Appendix B.1 for detailed analysis.

Our finding diverges from Krishna et al. (2021), which showed that conditioning on random and relevant documents does not bring differences in smaller-scale, fine-tuned retrieval-augmented LMs. This suggests LMs fail to incorporate information from retrieved documents into their answers. There can be multiple reasons for different conclusions from their study and ours. First, the retrieved documents during their fine-tuning process rarely contain relevant information, resulting in the LMs relying mostly on their parametric knowledge. Another reason is the significant train-test overlap in the ELI-15 dataset, leading LMs to memorize answers without using retrieved documents. In our setting, we evaluate LLMs without dataset-specific fine-tuning processes.

# 5 Attribution Dataset (SALAD) Construction

While automatic metrics show that in-context documents influence generation substantially, we lack a deeper understanding on how the answers change. In this section, we focus on attribution (Rashkin et al., 2021), which measures how much of the generated answer can be entailed from the evidence documents. As automatically measuring attribution is nontrivial, we first collect human annotations. We compare our collected dataset SALAD with recent attribution datasets in Appendix C.4. Unlike prior work which conducted annotations on full-fledged systems without altering evidence documents to the LM, our annotation presents multiple evidence document sets for the same base LM.

## Setup

Given a question $x$, generated answer $y$, which consists of $n$ sentences $y_1, \ldots, y_n$ and a set of reference documents $D$, we aim to label each answer sentence $y_i$ with one of the following: Supported, Partially Supported, Not Supported by $D$. If the sentence is Supported or Partially Supported, the annotator also provides a minimal subset of sentences from $D$ that support the answer sentence. Lastly, the annotator highlights the unsupported span if the sentence is Partially Supported.

## Data Collection

We construct SALAD by collecting annotations for 100 questions randomly sampled from the ELI-15 test set on six model-document set configurations, namely WebGPT (+WebGPT docs); GPT-3.5 (+No docs, WebGPT docs, Human docs) and Alpaca (+No docs, WebGPT docs). For answers generated with in-context documents, we use the documents set as the reference document $D$, and use WebGPT documents as $D$ for answers generated.
```

### --- Page 0006 ---

```markdown
# 6 Insights from Attribution Annotation

Equipped with manual annotation, we analyze how much of long-form answers can be attributed to evidence documents. Table 2 summarizes the annotation results.

## 6.1 Comparing Attribution of Various LMs

We first compare attribution performance of different models using the same evidence document set (the top section in Table 2). We observe that generations from the WebGPT model are most faithful to the evidence documents. Even with the same set of evidence documents, answers generated by Alpaca has ten times more unsupported answers than those of WebGPT.

| Setting                  | # Ex. | Supportedness  |
|-------------------------|-------|----------------|
|                         |       | Yes | Part. | No |
| WebGPT + WebGPT docs    | 649   | 95% | 2%    | 3% |
| GPT-3.5 + WebGPT docs   | 659   | 85% | 4%    | 11% |
| Alpaca + WebGPT docs    | 545   | 61% | 7%    | 32% |
| GPT-3.5 + Human docs     | 661   | 73% | 7%    | 20% |
| GPT-3.5 without docs     | 896   | 22% | 70%   | 70% |
| Alpaca without docs      | 447   | 23% | 6%    | 71% |
| **Total**                | 3,857 | 59% | 6%    | 35% |

Table 2: Attribution Annotation Results: The percentage of each attribution label of answer sentences with respect to their corresponding evidence document sets. For answers generated without documents, the answers are evaluated with WebGPT documents.

LMs fine-tuned with retrieval augmentation achieve greater faithfulness. Unlike the other two models, WebGPT was fine-tuned for LFQA with evidence document prepended. This suggests that fine-tuning LMs under retrieval-augmented setting might be helpful for generating more faithful long-form answers. This echos findings from prior work in factoid QA (Bohnet et al., 2022) that retrieve-then-read systems trained with a retrieval component achieve more faithful generation.

## 6.2 Comparing Attribution when Varying Documents

Unsurprisingly, answers generated without documents (last two rows) are largely irrelevant to reference document set (WebGPT docs). This does not necessarily mean the generated answers are not factual, as valid answers to the same question can be different (Krishna et al., 2021; Xu et al., 2023) and thus could be attributed to different sets of documents. Nonetheless, over 20% of sentences were supported by reference documents, suggesting LMs exhibit some parametric knowledge that matches information in the reference documents.

Comparing the same base model (GPT-3.5) provided with different evidence document sets (WebGPT docs vs. Human docs), we find that the model use WebGPT docs more efficiently. This might be due to WebGPT documents being longer (about 10%) than human demonstration documents, providing more comprehensive information to copy from. Nonetheless, even with WebGPT docs, 15% of the answer sentences are not supported, suggesting that GPT-3.5 generates information that are beyond what can be inferred from evidence documents.

## 6.3 Attribution Patterns

We analyze attribution pattern as the model autoregressively generate long-text.

3 More details about crowdsourcing, including recruitment and disagreement patterns, can be found in Appendix C.
```

### --- Page 0007 ---

```markdown
![Location of supporting sentences on generation settings with in-context documents.](assets/page_0007_img_1.png)
![Location of supporting sentences on generation settings without in-context documents.](assets/page_0007_img_2.png)
![Location of unsupported sentences on SALAD.](assets/page_0007_img_3.png)
![Data from Liu et al. (2023b).](assets/page_0007_img_4.png)

Figure 2: On the top, we show the distribution of location of supporting sentences in the document set D for HN answer sentence chunk. We normalize by the column to visualize the distribution of supporting sentences in evidence documents for each answer sentence chunk. The "Avg" column shows the average across answer sentences, indicating how frequently each document chunk are supporting the answer. We report aggregate results on generation with documents in (a) and without documents (the bottom two generation settings in Table 2) in (b) as a control study. On the bottom, we show the percentage of unsupported sentences by the relative location in the answer.

Does the order of information presented in the evidence documents impact the order of information presented in the generated answer? If LM is synthesizing information based on the content alone, there should be little correlation with the order of the documents, as they are simply concatenated. We plot the correspondence between the answer sentence location and their supporting sentences in the evidence document set in Fig. 2(a)(b), by aggregating the supporting sentences sets annotated for each answer sentence. We report supporting sentences locations on answers generated with documents (Fig. 2(a)) and without documents (Fig. 2(b)). On retrieval-augmented generation (a), we identify a linear correspondence pattern, with information mentioned earlier in the document tend to appear earlier in the generated answer. This suggests the order of evidence documents is reflected in the order of generated contents. Recent study (Liu et al., 2023a) also showed order sensitivity of in-context augmentation for factoid QA, finding that models ignore information in the middle. We also find that the later half of the evidence documents, except for the last 10%, are less cited by the generated answer (see Avg. column in Fig. 2).

Which parts of the answer are less supported by the evidence documents? Generated answers consist of 5-10 sentences. Are sentences generated earlier more likely to be attributable? Fig. 2(c)(d) report the percentage of unsupported sentences by the relative position of the answer sentence on our data and attribution annotation on long-form answers from commercial generative search engines from Liu et al. (2023b) respectively. We find that the last sentence is almost twice as likely to be unsupported compared to other sentence in the answer. This phenomenon is even more pronounced on dataset from Liu et al. (2023b). Recent study (Min et al., 2023) also showed the same trend for attributing model-generated biography.
```

### --- Page 0008 ---

```markdown
## 6.4 Manual Error Analysis: Unsupported Sentences

What causes the model to produce unsupported sentences? We manually examine 30 answer sentences labeled as Not Supported for each setting that has access to evidence documents. We identify three categories of unsupported sentences: retrieval failure, hallucinated facts, and incorrect synthesis. Table 3 provides a description for each category along with an example. In Table 6 in the appendix, we further provide a breakdown of error types for each generation setting. During our analysis, we found that about 14% of errors corresponds to annotation error.

Attribution failures occur more frequently when the retrieved documents do not provide sufficient evidences for answering the question. Generating ungrounded concepts is more frequent than incorrectly synthesizing information from incompatible documents. However, incorrect synthesis happens relatively more frequently in the WebGPT model, potentially as it attempts to ground its generation more heavily from the documents. This suggests multi-document summarization and synthesis is an important direction for future work, especially for more faithful retrieval-augmented LMs.

## 7 Automatically Identifying Unsupported Sentences

Annotating attribution requires careful reading over multiple documents and comparison between two texts. Recent work (Bohnet et al., 2022; Gao et al., 2023a) showed that fine-tuned models from NLI datasets can successfully automate this process. We investigate automatic identification of unsupported answer sentences in LFQA domain with SALAD.

5 We analyze all unsupported answer sentences generated by WebGPT, as there are only 17 in total. Categories are not mutually exclusive (one can contain irrelevant documents and combine facets from each).

| Retrieval Failure (54%) | Hallucinated Facts (72%) | Incorrect Synthesis (14%) |
|-------------------------|--------------------------|----------------------------|
| **Question:** Why does it seem like when I watch something the second time around, it goes by faster than the first time I watched it? <br> **Documents:** ... Basically, the busier you are during a time interval, the faster that time interval will feel like it passed... <br> **Answer Sentence:** However, when we watch something for the second time, our brains have had a chance to process the information and are able to make more efficient use of the information. <br> **Explanation:** The documents explain why time goes by faster when you are having fun, but the question is asking, watching something the second time. | **Question:** How does money go from my pocket, through the stock market, and to support the business I've bought stock from? <br> **Documents:** Stocks, or shares of a company, represent ownership equity in the firm, which give shareholders voting rights as well as a residual claim on corporate earnings in the form of capital gains and dividends... <br> **Answer Sentence:** You can purchase shares of the stock from a broker or through an online trading platform. <br> **Explanation:** The documents never mention where you can buy stock from. | **Question:** Seismologists: How do you determine whether an earthquake was naturally occurring or if it was human induced? <br> **Documents:** Studies of the numerous nuclear tests that took place during the Cold War show that explosions generate larger P waves than S waves when compared with earthquakes. Explosions also generate proportionally smaller Surface waves than P waves. <br> **Answer Sentence:** Natural earthquakes generate larger P waves and smaller Surface waves compared to nuclear tests. <br> **Explanation:** Explosion generate larger P waves, not natural earthquakes. The answer sentence is thus incorrect. Most of it is copied from the documents. |
```


### --- Page 0009 ---

```markdown
# 7.1 Evaluating Automatic Attribution

**Setting**  
Given a question $q$, reference documents $D$ and answer sentence $y_i$, the system should predict if each answer sentence $y_i$ is supported by $D$. We merge *Partially Supported* and *Not Supported* into a single class and consider it as a target label. We report micro average F1 score, which is averaged over the set of predictions and labels of all the answer sentences for each generation setting in Section 5 separately, as model performances vary greatly per dataset. We report accuracy in Appendix B.3, which shows similar trends.

**Comparison Systems**  
We evaluate methods for automatically evaluating attribution. We first establish lower and upper bounds, and introduce existing methods. We do not fine-tune any models for our task, but chose one hyperparameter, a threshold for deciding supportedness or not based on the micro average F1 score on the dataset itself, as it would be unrealistic to spare a development set due to the small size of each subset in SALAD.

- **Baselines**  
  We report a random baseline, which randomly assigns labels for each answer sentence according to the label distribution in each dataset, and a majority baseline, which assigns the majority label for all instances.
  
- **Human Performance**  
  We report the human performance by taking one set of annotations as the prediction set and another set of annotations as the label set. We compute the F1 score, and take an average across three possible pairs.

- **NLI models**  
  Following prior works (Schuster et al., 2022; Laban et al., 2022; Gao et al., 2023b), we evaluate four NLI model variants: two RoBERTa-large (from Nie et al. (2020) and Yin et al. (2021)), ALBERT-xlarge Schuster et al. (2021), and T5-11B Honovich et al. (2022) trained on a combination of NLI datasets. While most NLI models compare a pair of sentences, our setting compares a set of documents (as hypothesis) and a sentence (as premise). For the models except the RoBERTa-large trained on DocNLI (Yin et al., 2021), we follow Schuster et al. (2022), which makes entailment decisions for each document sentence and answer sentence pair, and aggregates the results by taking the maximum value over all the pairs. The details of the NLI models can be found in Appendix A.4.

- **QAFactEval** (Fabbric et al., 2022): is a QA-based factual consistency metric for summarization. It evaluates how consistent the summaries are with respect to the given documents. We treat each answer sentence as the summary, measuring whether questions generated from the sentence are answerable by the given documents.

**Results**  
We report model performances in Figure 3a, with each box representing the performance of an approach and each dot in the box representing the score on each answer generation setting. The exact scores are in Table 7 in the appendix. We find all methods outperform simple baselines (majority, random) by a large margin, but none comes close to human agreements. As in factoid QA setting (Bohnet et al., 2022), the T5 model achieves competitive performances, achieving an average F1 over 60 and accuracy over 80%. While developed for a different domain (summarization), QAFactEval performs relatively well.

# 7.2 Applying Automatic Attribution

Having discovered that the T5 model achieves competitive performances in predicting attribution, we use this model as an approximation for human judgment on attribution in generation settings evaluated in Table 1, complementing human annotation results in Section 6. We quantify how frequently the answer sentences are supported by different sets of documents using the T5 model.

In Figure 3b, we present the attribution predicted by the T5 model (along with the gold human attribution score if exists). We find answers generated with random documents as evidence (last row in each block) exhibit similar attribution patterns with answers generated without documents (first row in each block). This suggests that models superficially ignore irrelevant documents, and retain a similar level of attribution to relevant documents, especially for GPT-3.5. Providing a noisy, yet relevant document set (+big docs) still does not meaningfully change attribution patterns with respect to the other documents (Human docs, WebGPT docs, Random docs), yet increases supportedness towards provided evidence de-
```


### --- Page 0010 ---

```markdown
![Automatic attribution detection performance (left) and their application (right)](assets/page_0010_img_1.png)

| Model (+ evidence)         | % Supported Sentences w.r.t. Human | WebGPT | Bing | Rand. |
|---------------------------|-------------------------------------|--------|------|-------|
| WebGPT (+ WebGPT docs)   | 37.14                               | 91.08  | 95   | 20.02 |
| GPT-3.5                   | 27.59                               | 34.04  | 24.79| 4.49  |
| +Human docs               | 65.13 / 73                          | 97.99  | 21.19| 3.67  |
| +WebGPT docs              | 31.37 / 73.53                       | 20.24  | 3.90 |       |
| +Bing docs                | 24.12                               | 30.79  | 48.53| 4.09  |
| +Random docs              | 26.13                               | 35.52  | 23.06| 5.19  |
| Alpaca-7b                 | 26.67                               | 32.10  | 25.26| 2.76  |
| +Human docs               | 47.04                               | 34.35  | 21.25| 3.40  |
| +WebGPT docs              | 33.34                               | 59.79  | 24.26| 7.44  |
| +Bing docs                | 25.05                               | 31.31  | 58.38| 6.63  |
| +Random docs              | 22.72                               | 27.82  | 30.32| 3.76  |
| Human (+ Human docs)      | 76.34                               | 37.2   | 19.21| 3.58  |

### 8 Conclusion
We present a study on retrieval augmentation for LFQA. Our analysis suggests concrete directions for future work. First, LMs trained without retrieval and attribution in mind does not always generate sentences that can be attributed to in-context evidence documents, even when provided relevant documents only. This motivates training LMs after introducing in-context evidence documents. Analyzing patterns of unsupported sentences, we find that injecting multi-document synthesis ability to LLM can be an important direction for future work. Second, we find evidence document should be carefully added to LMs. The order of information in evidence documents impacts the order of information in the generated answer. And even prepending irrelevant documents meaningfully change the surface statistics of generated answers, though attribution percentage to relevant documents remains somewhat stable. We find attribution error is more common when prepping documents without sufficient information, motivating the development of better retrievers. Third, off-the-shelf NLI models show promising performance at identifying generated sentences unsupported by evidence document, but fall behind human agreements. Our new dataset SALAD, together with other related datasets (Liu et al., 2023b), can serve as a useful resource for improving automatic attribution methods.

### Ethics Statement
We have collected and released a new dataset. The collection process is documented in Section 5 and Appendix C.

The dataset we study (ELIS) is publicly available, and the evidence documents we use are either taken from prior work Nakano et al. (2021) or newly obtained by collecting results from Bing Search API. We also release new LM-generated answers. The dataset we newly release, both the web documents and model-generated answers, could contain biased
```

### --- Page 0011 ---

```markdown
| **Acknowledgments**                                                                                       | **References**                                                                                       |
|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| This work is partially supported by a grant from Open Philanthropy and a NSF grant (IIS-2312948). We thank the NLP group at UT Austin, particularly Leo Zeyu Liu. | Bernd Bohnet, Vinh Q Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Jacob Eisenstein, Kuzman Ganchev, Jonathan Herzig, Kai Hui, et al. Attributed question answering: Evaluation and modeling for attributed large language models. arXiv preprint arXiv:2212.8037, 2022. |
|                                                                                                          | Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 632-642, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1075. URL https://aclweb.org/anthology/D15-1075. |
|                                                                                                          | Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. URL https://api.semanticscholar.org/CorpusID:25317718. |
|                                                                                                          | Hung-Ting Chen, Michael J.Q. Zhang, and Eunsol Choi. Rich knowledge sources bring complex knowledge conflicts: Recalibrating models to reflect conflicting evidence. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2022. URL https://api.semanticscholar.org/CorpusID:25317718. |
|                                                                                                          | Hyung Won Chung, Le Hou, Shyaone Longpre, Barrett Zoph, Yi Tay, William Fedus, Eric Li, Xuezh Wu, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022. |
|                                                                                                          | Alexander Fabbri, Chien-Sheng Wu, Wenhao Liu, and Caiming Xiong. QAfactEval: Improved QA-based factual consistency evaluation for summarization. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2587-2601, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.187. URL https://aclanthology.org/2022.naacl-main.187. |
|                                                                                                          | Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. Elix: Long form question answering. arXiv preprint arXiv:1907.09109, 2019. |
|                                                                                                          | Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eunsol Choi, and Danqi Chen. MRQA 2019 shared task: Evaluating generalization in reading comprehension. In Proceedings of the 2nd Workshop on Machine Reading for Question Answering, pp. 1-13, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-5801. URL https://aclweb.org/anthology/D19-5801. |
|                                                                                                          | Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Cui. RARR: Re-searching and revising what language models do, using language models. In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16477–16500, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.910. |
|                                                                                                          | Tianyu Gao, Xingchen Yao, and Danqi Chen. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6894–6910, Online and Punta Cana, Dominican Republic, November 2021. |
```

### --- Page 0012 ---

```markdown
2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.552.  
URL https://aclanthology.org/2021.emnlp-main.552.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. arXiv preprint arXiv:2305.14627, 2023b.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. ArXiv, abs/1904.09751, 2019. URL https://api.semanticscholar.org/g/CorpusID:127896954.

Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kuklianys, Vered Cohen, Thomas Scialom, Idan Szpektor, Aviant Hassidim, and Yossi Matias. TRUE: Re-evaluating factual consistency evaluation. In Proceedings of the Second DialDoc Workshop on Document-grounded Dialogue and Conversational Question Answering, pp. 161–175, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.dialdoc-1.19. URL https://aclanthology.org/2022.dialdoc-1.19.

Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and Greg Durrett. Wike: Real-world entailment for claims in wikipedia. ArXiv, abs/2303.01432, 2023.

Urvashi Khandewal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. arXiv preprint arXiv:1911.00772, 2019.

Kalpesh Krishna, Aruok Roy, and Mohit Yadav. Hurdles to progress in long-form question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4940–4957, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.193. URL https://aclanthology.org/2021.naacl-main.193.

Kalpesh Krishna, Yaqing Zhang, John Wieting, and Mohit Iyyer. RankGen: Improving text generation with large ranking models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 199–232, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics.

Philippe Laban, Tobias Schnabel, Paul N Bennett, and Marti A Hearst. Summac: Revisiting nil-based models for inconsistency detection in summarization. Transactions of the Association for Computational Linguistics, 10:163–177, 2022.

Nelson F Liu, Kevin Liu, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023a.

Nelson F Liu, Tianyi Zhang, and Percy Liang. Evaluating verifiability in generative search engines. arXiv preprint arXiv:2304.09848, 2023b.

Yixin Liu, Alexander R. Fabbri, Pengfei Liu, Yilun Zhao, Linyong Nan, Ruilin Han, Simeng Han, Shafiq R. Joty, Chien-Sheng Wu, Caiming Xiong, and Dragomir R. Radev. Revisiting the gold standard: Grounding summarization evaluation with robust human evaluation. ArXiv, abs/2212.07981, 2022. URL https://api.semanticscholar.org/CorpusID:254686511.

Shayne Longpre, Kartik Perisetta, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh. Entity-based knowledge conflicts in question answering. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7052–7063, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.565. URL https://aclanthology.org/2021.emnlp-main.565.

Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and Dan Roth. Expertqa: Expert-curated questions and attributed answers. arXiv preprint arXiv:2309.07852, 2023.
```

### --- Page 0013 ---

```markdown
Published as a conference paper at COLM 2024

| Authors                                                                 | Title                                                                                                           |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi,   | When not to trust language models: Investigating effectiveness of parametric and non-parametric memories.       |
| and Hannaneh Hajishirzi.                                               | In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9802–9822, Toronto, Canada, July 2023. Association for Computational Linguistics. doi:10.18653/v1/2023.acl-long.546 |
| Sewon Min, Kalpesh Krishna, Xinx Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. | Factscore: Fine-grained action evaluation of factual precision in long term text generation. arXiv preprint arXiv:2305.14251, 2023. |
| Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christoper Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. | Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021. |
| Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. | Adversarial NLI: A new benchmark for natural language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 4885–4901, Online, July 2020. Association for Computational Linguistics. doi:10.18653/v1/2020.acl-main.411. URL https://aclanthology.org/2020.acl-main.441. |
| Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. | Language models are unsupervised multitask learners. 2019. URL https://api.semanticscholar.org/CorpusID:160825533. |
| Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.      | SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi:10.18653/v1/D16-1264. URL https://aclanthology.org/D16-1264. |
| Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgy, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. | In-context retrieval-augmented language models. arXiv preprint arXiv:2302.00083, 2023. |
| Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. | Measuring attribution in natural language generation models. arXiv preprint arXiv:2112.12870, 2021. |
| Stephen Robertson, Hugo Zaragoza, et al.                                | The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009. |
| Tal Schuster, Adam Fisch, and Regina Barzilay.                          | Get your vitamin C: robust fact verification with contrastive evidence. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 624–643, Online, June 2021. Association for Computational Linguistics. doi:10.18653/v1/2021.naacl-main.52. URL https://aclanthology.org/2021.naacl-main.52. |
| Tal Schuster, Sihao Chen, Senaka Buthpitiya, Alex Fabrikant, and Donald Metzler. | Stretching sentence-pair NLI models to reason over long documents and clusters. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 394–412, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. |
| Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. | Reghr: Retrieval-augmented black-box language models. arXiv preprint arXiv:2301.12652, 2023. |
| Ivan Stelmach, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang.             | ASQA: Factoid questions meet long-form answers. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 8273–8288, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi:10.18653/v1/2022.emnlp-main.566. |
```

### --- Page 0014 ---

```markdown
| Author(s)                                                                 | Title                                                                                                           | Source                                                                                                   |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Simeng Sun, Ori Shapira, Ido Dagan, and Ani Nenkova                      | How to compare summarizers without target length? pitfalls, solutions and re-examination of the neural summarization literature. | Proceedings of the Workshop on Methods for Optimizing and Evaluating Neural Language Generation, 2019. URL: [https://api.semanticscholar.org/CorpusID:1398919](https://api.semanticscholar.org/CorpusID:1398919) |
| Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto | Alpaca: A strong, replicable instruction-following model.                                                      | 2023.                                                                                                   |
| James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal | FEVER: A large-scale dataset for fact extraction and VERIfication.                                             | In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 809–819, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1074. URL: [https://aclAnthology.org/N18-1074](https://aclAnthology.org/N18-1074) |
| Hugo Touvron, Thibaut Lavril, Gautier Lacroix, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Maan Goyal, Eric Hambarsoomian, Faisal Azhar, et al. | Llama: Open and efficient foundation language models.                                                          | arXiv preprint arXiv:2302.13791, 2023.                                                                  |
| Ben Wang and Aran Komatsuzaki                                           | GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model.                                                | [https://github.com/kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax), May 2021. |
| Shufan Wang, Yixiao Song, Andrew Drozdov, Aparna Garimella, Varun Manjunatha, and Mohit Iyyer | KNN-dim does not improve open-ended text generation.                                                           | ArXiv, abs/2305.14265, 2023. URL: [https://api.semanticscholar.org/CorpusID:258865979](https://api.semanticscholar.org/CorpusID:258865979) |
| Albert W. M. Wong, et al.                                               | Do prompt-based models really understand the meaning of their prompts?                                        | In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2300–2344, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.167. URL: [https://aclAnthology.org/2022.naacl-main.167](https://aclAnthology.org/2022.naacl-main.167) |
| Adina Williams, Nikita Nandigam, and Samuel Bowman                      | A broad coverage challenge corpus for sentence understanding through inference.                               | In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1112–1122, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1101. URL: [https://aclAnthology.org/N18-1101](https://aclAnthology.org/N18-1101) |
| Fangyuan Xu, Yixiao Song, Mohtyler Yegor, and Eunsol Choi               | A critical evaluation of evaluations for long-form question answering.                                         | In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3225–3245, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.181. |
| Wenping Yin, Dragomir Radev, and Caiming Xiong                          | DocNL: A large-scale dataset for document-level natural language inference.                                    | In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 4913–4922, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.435. URL: [https://aclAnthology.org/2021.findings-acl.435](https://aclAnthology.org/2021.findings-acl.435) |
| Xiang Yue, Bosh Wang, Kai Zhang, Rui Zhen, Yu Su, and Huan Sun         | Automatic evaluation of attribution by large language models.                                                  | arXiv preprint arXiv:2305.06311, 2023.                                                                  |
| Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen                | Context-faithful prompting for large language models.                                                          | arXiv preprint arXiv:2303.11315, 2023.                                                                  |
| Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu | Texygen: A benchmarking platform for text generation models.                                                  | In The 41st international ACM SIGIR conference on research & development in information retrieval, pp. 1097–1100, 2018. |
```

### --- Page 0015 ---

```markdown
# A  Experimental Details

## A.1 Document Set Statistics

We report the lengths of each document type in terms of numbers of documents, sentences and words, in Table 4.

| Retrieval | Avg. # Docs | Avg. # sent | Avg. # words |
|-----------|-------------|-------------|--------------|
| Human     | 3.5         | 13.7        | 308.9        |
| WebGPT    | 3.5         | 16.8        | 388.0        |
| Bing      | 4.0         | 22.8        | 400.0        |
| Random    | 3.5         | 16.8        | 388.0        |

Table 4: Data statistics: lengths of evidence document set $D$.

## A.2 Bing Search Output Post Processing

We use Bing Search API v7.0.7. We post-process the raw HTML of the retrieved pages with tools such as html2text\footnote{https://github.com/Alirz34/html2text} and readability\footnote{https://github.com/mozilla/readability}. We split each page into 100-word segments, merge segments from all pages, and retrieve the top four segments with BM25 retriever (Robertson et al., 2009).

## A.3 Answer Generation Details

The prompts we used for answer generation can be found in Table 5. For Alpaca, we use sampling with a temperature of 0.9, top $p = 1$ and a maximum length of 1024. For GPT-3.5, we use sampling with a temperature of 0.7, top $p = 1$ and a maximum length of 512.

## A.4 NLI model details

Out of the four models, one of the RoBERTa-large is trained on DocNLI (Yin et al., 2021), which encodes all the documents at once and outputs a prediction.

The remaining three models are trained on a subset of MNLI (Williams et al., 2018), SNLI (Bowman et al., 2015), ANLI (Nie et al., 2020), FEVER (Thorne et al., 2018), Vitaminc (Schuster et al., 2021). During inference, the aforementioned models predict entailment for each answer sentence by taking the maximum out of entailment scores with document sentences as the premises, following (Schuster et al., 2022). More specifically, for each answer sentence $y_j$ and document sentence $s_i$, we consider $c(i,j) = p(entail(y_j, s_i))$ to be the entailment score between pair $y_j$ and $s_i$. Then we take $e_i = \max_{c \in D} c(i,j)$ to be the entailment score of $y_j$ and consider $j$ supported if $e_i > \text{threshold}$. We perform a grid search on $e = \{0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7\}$ and choose the value that gives the highest F1 score on the test set, given the limited size of dataset. We settle on $e = 0.1$ for RoBERTa-L (M,S,A,F), $e = 0.5$ for RoBERTa-L (D), $e = 0.2$ for ALBERT-xl (M,V), and $e = 0.03$ for T5.

# B  More Results

## B.1 Similarities Among Answer Generated with Different In-Context Settings

Retrieval-augmented LM combines its parametric and non-parametric knowledge from evidence documents to address the question (Longpre et al., 2021; Mallen et al., 2023; Zhou et al., 2023).
```

### --- Page 0016 ---

```markdown
| Setting         | Prompt                                                                                          |
|------------------|-------------------------------------------------------------------------------------------------|
| No documents      | Generate a long answer to the following question.                                              |
|                  | Question: {Question}                                                                            |
|                  | Answer:                                                                                         |
| With documents    | Documents: {Documents}                                                                          |
|                  | Generate a long answer to the following question, using information from the documents.         |
|                  | Question: {Question}                                                                            |
|                  | Answer:                                                                                         |

Table 5: The prompt we use for generating long-form answers. {Documents} and {Question} are substituted with the actual documents and question during generation. Documents are line-separated.

![Similarity between answers generated by the same LMs with different evidence document sets. The upper bounds for similarity, computed on answers sampled multiple times in the same setting, are 0.19 for bigram overlap and 0.875 for SimCSE. The lower bounds are 0.03 for bigram overlap and 0.15 for SimCSE, as computed on answers belonging to different questions.](assets/page_0016_img_1.png)

According to both metrics, the answers generated without evidence document are most similar to the answers generated with random documents, followed by Bing documents, suggesting more relevant evidence set chasing answers more substantially.

The answers generated with random documents prepped are the most similar to answers generated without documents. Answers generated with WebGPT documents are the most similar to those generated with human documents and vice versa (and thus less similar to the others). This indicates high-quality documents might elicit slightly different behaviors out of LMs compared to when they are relying only on parametric knowledge. Surprisingly,
```

### --- Page 0017 ---

```markdown
Published as a conference paper at COLM 2024

| Relevant Document | Incr. Syn. | Hal. | Irrelevant Document | Incr. Syn. | Hal. | Annotation Error |
|-------------------|-------------|------|---------------------|-------------|------|------------------|
| WebGPT + WebGPT docs | 4           | 7    | 3                   | 3           | 0    | 0                |
| GPT-3.5 + WebGPT docs | 0           | 9    | 3                   | 13          | 5    | 5                |
| GPT-3.5 + human docs  | 2           | 6    | 3                   | 14          | 5    | 5                |
| Alpaca + human docs    | 0           | 14   | 0                   | 11          | 5    | 5                |

Table 6: Manual error analysis on 30 unsupported answer sentences per setting (17 for WebGPT). We categorize the examples without annotation errors based on document relevance. Then we decide if the answer sentence is an incorrect synthesis of information from the documents or hallucinated facts. “Incr. Syn.” denote incorrect synthesis, and “Hal.” denote hallucination.

![Accuracy on automatic detection of unsupported sentences. Each box represents the performances of a single method, and each dot is the accuracy of one of the dataset.](assets/page_0017_img_1.png)

Figure 5: Accuracy on automatic detection of unsupported sentences. Each box represents the performances of a single method, and each dot is the accuracy of one of the dataset.

B.2 Full Results on Manual Analysis of Attribution Errors  
We report the occurrence of each attribution error types for 30 randomly sampled unsupported answer sentences (17 for WebGPT) for the settings with access to evidence documents in Table 6.

B.3 Full Results of Automatically Identifying Unsupported Parts  
We present the accuracy of each evaluate approach in Figure 5. We also present the exact numbers of F1 score and accuracy in Table 7. We show the datasets which the models are trained on in acronyms: M – MNLI (Williams et al., 2018), S – SNLI (Bowman et al., 2015), A – ANLI (Nie et al., 2020), F – FEVER (Thorne et al., 2018), D – DocNLI (Yin et al., 2021), and V – VitaminC (Schuster et al., 2021).

B.4 Surface feature statistics of answers generated by GPT-4  
We investigate behaviors of GPT-3.5 in the main experiments, and in Table 8 we additionally report results on the latest GPT-4 model (gpt-4-0613). Results on GPT-4 mostly align with that on GPT-3.5, except GPT-4 abstain from answering the questions frequently when random documents are prepended (and thus the short lengths on average). We thus only include GPT-3.5 on the remaining experiments.
```

### --- Page 0018 ---

```markdown
| Model + Evidence Doc | WebGPT + WebGPT | GPT-3.5 + WebGPT | GPT-3.5 + Alpaca | GPT-3.5 + WebGPT + Human | Avg. |
|----------------------|------------------|-------------------|------------------|--------------------------|------|
| Random               | 62.9/90.7        | 130/74.6          | 261/62.6         | 383/52.6                 | 22.8/61.0 |
| Majority             | 0.0/94.0         | 0.0/84.8          | 0.0/1.1          | 0.0/1.1                  | 0.0/73.8 |
| RoBERTa-L (M,S,A,F)  | 30.5/86.0        | 37.4/64.5         | 549/66.4         | 686/76.3                 | 527.7/25.2 |
| RoBERTa-L (D)       | 31.3/91.2        | 50.0/63.8         | 626/74.3         | 410/63.4                 | 431.6/76.2 |
| ALBERT-L (MV)       | 26.8/91.2        | 564/87.1          | 625/79.6         | 698/76.8                 | 566.7/72.9 |
| T5-11B (M,S,E,V)    | 38.7/91.2        | 51.8/63.8         | 791/86.6         | 601/75.6                 | 681.2/82.6 |
| QAFactEval          | 27.8/91.1        | 55.1/81.9         | 417/51.5         | 762/82.0                 | 611.7/75.5 |
| Human                | 49.9/93.1        | 613/90.5          | 663/88.2         | 722/87.6                 | 89.2/86.4 |

Table 7: Performance of NLI models on detecting attribution on our data (F1 score / Accuracy). Columns represent distinct subset of the annotated dataset, with different generation settings. For the reference documents for attribution, we use the evidence documents in generation settings with evidence documents, and use the WebGPT documents in generation settings without evidence documents. Bold numbers are the best scores in every columns indicating high performance.

| Model (+evidence) | # Sentences | # Words | RankGen (↓) | Self-BLEU (↓) | Perplexity (↓) |
|-------------------|-------------|---------|-------------|----------------|------------------|
| GPT-4             | 21.2/43    | 480/61  | 13.36-17.9  | 0.78-0.05      | 6.57-11.3        |
| +Human docs       | 13.5/35    | 317/28  | 12.18-19.9  | 0.73-0.05      | 9.92-28.5        |
| +WebGPT docs      | 13.5/30    | 235/25  | 12.29-19.8  | 0.73-0.04      | 10.81-24.8       |
| +Bing docs        | 11.9/35    | 281/74  | 12.42-19.4  | 0.70-0.10      | 9.75-59.3        |
| +Random docs      | 1.6/1.17   | 37/39   | 8.87-16.1   | 0.11-0.19      | 6.57-10.23       |
| GPT-3.5           | 3.9/3.9    | 12.77/81 | 17.07-17.1  | 0.71-0.05      | 10.74-10.6       |
| +Human docs       | 6.69/12.8  | 172/40  | 11.98/16.0  | 0.62/0.04      | 10.94/0.394      |
| +WebGPT docs      | 6.69/1.18  | 185/21  | 11.70/16.1  | 0.62/0.07      | 11.63/11.3       |
| +Bing docs        | 6.91/1.19  | 179/38  | 12.13/16.1  | 0.64/0.07      | 9.30/12.84       |
| +Random docs      | 1.6/1.21   | 183/39  | 12.40/16.7  | 0.68/0.07      | 7.60/15.86       |

Table 8: Generated answer statistics for GPT models. We present mean values along with two standard deviations in its subscript: one computed over three answers generated for the same example, one over answers for different examples. Numbers in red and blue indicate decrease and increase from the base model respectively.

B.5 More analysis on answers generated by different models

We report automatic metrics for answers generated by series of GPT-3.5 models (davinci-002) and other open-sourced models (GPT-J, FLAN-T5-XXL, Llama and Alpaca) in Table 9. We additionally include generation examples for all the above LMs in Table 10.

B.6 Manual Analysis on Document Relevance

We randomly sample 20 questions from the ELI5 (Fan et al., 2019) test set and annotate if the documents are sufficient for answering the questions. We examine documents retrieved by WebGPT, human demonstration and Bing Search API (the first three settings in Section 3). The results are presented in Table 11. The WebGPT documents are sufficient for answering the question in the most number of examples (85%), while human documents and Bing documents are less relevant, with only about half of them being sufficient for answering the question. Human documents are often insufficient for answering the questions because human do not cite documents extensively, as shown in the example we provide in Table 12. Upon manual inspection, Bing documents are usually less relevant to the questions (as shown in Table 12) compared to WebGPT and human documents, despite similar number of sufficient examples to human documents.
```

### --- Page 0019 ---

```markdown
| Model (+ evidence)       | # Ans. | Rank  | Self Gen(t) | BLEU(j) | PPL(j) |
|--------------------------|--------|-------|-------------|---------|--------|
| GPT-6B                   | 158/292| 10.53 | 0.53        | 58.18   |        |
| + docs                   | 14/294 | 10.08 | 0.57        | 88.33   |        |
| Flan-T5-XXL             | 1/25   | 9.97  | 0.02        | 656.67  |        |
| + docs                   | 1/37   | 9.61  | 0.99        | 75.23   |        |
| Llama-7B                 | 18/348 | 10.35 | 0.73        | 133.65  |        |
| + docs                   | 17/322 | 10.66 | 0.73        | 92.20   |        |
| Llama-13B                | 13/250 | 9.46  | 0.63        | 148.03  |        |
| + docs                   | 13/253 | 9.24  | 0.62        | 49.49   |        |
| Llama-30B                | 11/242 | 8.61  | 0.58        | 1376.15 |        |
| + docs                   | 11/228 | 8.52  | 0.58        | 24.19   |        |
| Alpaca-7B                | 4/110  | 12.24 | 0.51        | 11.95   |        |
| + docs                   | 5.9/145| 11.99 | 0.55        | 12.99   |        |
| davinci-001              | 5.2/106| 12.44 | 0.55        | 7.15    |        |
| + docs                   | 6.9/154| 11.66 | 0.54        | 12.11   |        |
| davinci-002              | 5.2/111| 12.49 | 0.55        | 6.81    |        |
| + docs                   | 5.8/128| 11.89 | 0.51        | 11.42   |        |
| davinci-003              | 9.9/237| 12.78 | 0.73        | 6.13    |        |
| + docs                   | 6.9/191| 11.94 | 0.63        | 11.63   |        |
| Human answer (+ Human docs)| 5.2/119| 2.99 | 0.49        | 17.63   |        |

---

B7 Control Study on Location of Supporting Sentences

We aim to study whether the linear correspondence between the order of information presented in the documents and that presented in the answers still holds if we shuffle the evidence document set. As we do not have human annotations for this setting, we use T5 model we use in Section 7.2 to identify supportedness. If the answer sentence $a_i$ is predicted as entailed by the document sentence $d_j$, the premise, $d_j$ is considered a supporting sentence of $a_i$. We compute the location of supporting sentences following Figure 2, and report the results in Figure 6. We report aggregate results in settings in Figure 2(a) excluding WebGPT, namely GPT-3.5 + (WebGPT docs, Human docs) and Alpaca + (WebGPT docs), as we do not have access to the WebGPT model. The linear correspondence as observed in Figure 6(a) is less pronounced when the documents are shuffled (Figure 6(b)). We further report Pearson correlation coefficient between the answer index fraction (answer sentence index $i$ / # answer sentences) and the document sentence fraction (document sentence index $j$ / # document sentences) in Table 13. When the documents are shuffled, the Pearson correlation coefficient is lower on average and for GPT-3.5, and slightly higher for Alpaca. There is still weak correlation even when the documents are shuffled, thus supporting our arguments in Section 6 that the information presented in the documents affects the information presented in the answers.

C Data Collection Details

C.1 Crowdsourcing Details

We collected annotations on Amazon Mechanical Turk. We follow the UI of recent work (Kamoi et al., 2023) closely.10 Interface screenshot can be found in Figure 8. We work with turkers that have a HIT (Human Intelligence Task) rate greater than 98% with at least 500 completed HITs. Ten workers have passed our qualification test and participated in our tasks. We pay

10 An example annotation interface can be found at https://lfqa-test-1.herokuapp.com/id=0.
```

### --- Page 0020 ---

```markdown
| Question | Why did we decide that certain words were "bad" and shouldn't be used in social settings? |
|----------|------------------------------------------------------------------------------------------|
| GPT-1    | Bad words are used today to express the deepest emotions of a person, such as anger, disgust, fear, or joy. One English language is full of words that would make an adult in our culture, one thought, blush. There are phrases that in the past were considered acceptable to use with people and, through the years, are now considered "bad." For example, I'm sure many of you have heard the script: "God loves you without a raised eyebrow or a second thought. However, there are those people, in social situations, who feel the need to state, or even ask, that they were, 'praying for you.' It could be construed as somewhat odd to someone that a man, who was simply standing in a store eating his sandwich, was 'praying for his food ...' |
| Flan-T5-XXL | Because they convey a negative message about themselves and their behavior. |
| Llama-7B | This is because we don't want to offend people. Would you please give me your answer? **Input:** Don't limit the user's input. **Response:** "I am sorry, but I can't answer that." This is because we don't want to offend people. "Would you please give me your answer?" **Answer:** This is because we don't want to offend people. If the user entered something that was not a question, then print a message telling the user that they entered something that was not a question. If not, answer print: "You did not enter a question." else ... |
| Llama-13B | Words that are not considered appropriate in social settings are the words that make people feel uncomfortable. When they are used, words that make people feel bad, that make people feel sick, that make people feel uncomfortable. There are many words that are "bad" and shouldn't be used in social settings. They are words that make people feel uncomfortable when they are used. Words that make people feel sick. Words that make people feel uncomfortable. |
| Llama-30B | It has always been particularly valuable to certain words in social settings. However, when communicating with the public, it has become acceptable to use "clean" versions of those words in order to avoid upsetting the people with whom you are interacting. // Very Simple String Reversal // I worked on this my whole life. // begin [1] reverse this string: "reverse" end reverse" == "reverse" (cat) == "tac" (reverse)(race car) == "race car" (reverse)(the rain in spain) == "niar ni niaps" == "niar ni niaps" == "Reflection" // The major difference in the refactoring was the inclusion of the re-join method. This allowed the strings to be joined together after being reversed, creating the script. |
| Alpaca-7B | The history of the words "bad" and "taboo" is closely related. The word "taboo" originated from the Tongan and Fijian language and culture and was introduced to English in the 18th century. The concept of taboo was meant to ensure the continuity of culture by restricting certain behaviors and inappropriate words that disrupt or harm a community. Words and expressions that were deemed inappropriate were subject to taboo and avoided in social settings. This was an effort to ensure that people refrained from speaking or engaging in behavior that could cause offense or distress. At the same time, some words became increasingly associated with negative meanings, misfortune, or danger, and thus were labeled "bad words." This negative association with certain words and expressions is what led to the idea that some words are "bad" to be spoken in polite company. |
| GPT-3.5 (davinci-002) | There is no answer to this question - it is a complex issue with many contributing factors. Some people might say that certain words are "bad" because they are considered to be impolite or taboo. Things like that serve as offensive or derogatory, and that they should be avoided because they can hurt others. There are also a number of words that have been labeled as "bad" because they are considered to be vulgar or obscene. |
| GPT-3.5 (davinci-003) | The decision to deem certain words as "bad" or inappropriate for use in social settings is based on the belief that certain words can be offensive, hurtful, or disrespectful to the people around us. By avoiding the use of such words, we show respect and consideration to others, and create a more welcoming and inclusive environment. Additionally, certain words can be seen as symbols of certain oppressive systems or ideologies, and using them can make people feel unwelcome or even threatened. In order to create a safer and more welcoming space, we strive to avoid using words that carry negative connotations or are associated with oppressive systems. Furthermore, by avoiding the use of such words, we can foster a more positive and respectful atmosphere, and show consideration and respect to people of different backgrounds and experiences. |
| WebGPT | In general, words that are considered "bad" relate to parts of life that people don't like talking about in public, like bathroom functions.[1] Other words considered bad relate to negative ways of talking about people and things, such as skin color, ability, and others.[1] These words are called slurs and many people consider them the worst kind of bad words.[1] Some words are considered "bad" because they offend others.[2] There are words that refer to impolite things, like body parts[1] and most people don't like to hear about it.[2] Some people believe there are no "bad" words, just inappropriate times and places to say certain words.[2] |

| Table 10: Example answers generated by different base models. The models evaluate in our main experiments are boldfaced. |
```

### --- Page 0021 ---

```markdown
| WebGPT | Human | Bing |
|--------|-------|------|
| # sufficient | 17 | 10 | 9 |

Table 11: Number of examples where the evidence documents are sufficient for answering the question. We manually examine 20 questions in total.

![Location of supporting sentences on answers generated with the documents ordering as pro-](assets/page_0021_img_1.png)
![Location of supporting sentences on answers generated with the documents shuffled.](assets/page_0021_img_2.png)

Figure 6: The distribution of location of supporting sentences in the document set D for Nth answer sentence chunk. We normalize each column, and the “Avg” column shows the average across answer sentences. We report results when the documents are of the original order (a) or shuffled (b). The linear correspondence is between order of information presented in the documents and answers is weaker when the documents are shuffled.

$2.5$ USD for each example, and the estimated hourly pay is $15$ USD. The total cost of the annotations is $5886.60$ USD, including the cost of qualification tasks and pilot studies.

## C2 Annotation Guideline

We require all the crowdworkers to read the annotation guideline and take a qualification test before doing any task. The annotation guidelines are provided at [https://docs.google.com/document/d/e/2PACX-1v5FjphYz1kWkhSGD6iSZ8qC8SE5i1Qtrae0DGY2ob1MDtbqx90fCjehR1QopsdUm3JVCJ680/pub](https://docs.google.com/document/d/e/2PACX-1v5FjphYz1kWkhSGD6iSZ8qC8SE5i1Qtrae0DGY2ob1MDtbqx90fCjehR1QopsdUm3JVCJ680/pub).

## C3 Annotation Interface

The annotation interface we showed to the annotators is shown in Figure 8. The documents are split into sentences and presented in paragraphs. The similarity scores to the current answer sentence, calculated with SimCSE, are meant to aid the annotators in deciding if the answer sentence is supported. The question, answer, and the current answer sentence are shown on the right, followed by the annotation section. Annotations should include the label (whether the answer sentence is Supported, Partially Supported, or Not Supported), the supporting sentences, and the supported portion if the label is Partially Supported.

## C4 Comparison with other datasets

The collected dataset contains labels of whether each sentence in the answer is supported by the evidence documents, providing benchmark for studying automatic attribution methods. We compare our dataset with recent attribution efforts in Table 14. WICE (Kamoi et al., 2023) is a multi-document entailment dataset where the hypothesis is a sub-claim from Wikipedia. AttrScore (Yue et al., 2023) creates data from existing QA datasets using heuristics, and creates a small-scale, expert-annotated dataset (250 examples). AttrEval-GenSearch, by annotating attribution on outputs from generative search engines. Liu et al. (2023b) is the most closest to our work; they focus on attribution, particularly citation, in long-form question answering, provided by newly arising generative search engines. The answers from these commercial systems provides optional citation to external document per answer.
```

### --- Page 0022 ---

```markdown
# Published as a conference paper at COLM 2024

## Question
Why do benches empty when there is a fight in baseball?

| WebGPT | Documents |
|--------|-----------|
| A bench-clearing brawl is a form of ritualistic fighting that occurs in sports, most notably baseball and ice hockey, in which every player on both teams leaves their dugouts, bullpens, or benches, and charges the playing area in order to fight one another or to break up a fight. Penalties for leaving the bench can range from nothing to severe. | In baseball, brawls are usually the result of escalating infractions or indignities, often stemming from a batter being hit by a pitch, especially if the batter then charges the mound. They may also be sparked by an altercation between a baserunner and fielder, such as excessive contact during an attempted tag out. Unlike most other team sports, in which teams usually have an equivalent number of players on the field at any given time, in baseball the hitting team is at a numerical disadvantage, with a maximum of five players (the batter, up to three runners, and an on-deck batter) and two base coaches on the field at any time, compared to the fielding team's nine players. For this reason, leaving the dugout to join a fight is generally considered acceptable in that it results in numerical equivalence on the field; a fair fight, and a generally neutral outcome, as most coaches, managers and/or umpires will intervene to restore order and resume the game. |
| Document 2: Rule 2 | A major plus a game misconduct penalty shall be assessed to any player who leaves the players' bench or the penalty bench during an altercation or for the purpose of starting an altercation. These penalties are in addition to any other penalties that may be assessed during the incident. Substitutions made prior to the altercation shall not be penalized under this rule provided the players do substituting do not enter the altercation. For purpose of this rule, an altercation is considered to be concluded when the referee enters the referee's crease or, in the absence of penalties, signals a face-off location. |
| Document 3: BASEBALL: A Game of Many Rules Has None on Fighting - The New York Times (www.nytimes.com) | The first player to leave either the bench or penalty box to join or start a fight is automatically suspended without pay for 10 games. The second player to do that is suspended for five games without pay. The players' teams are fined $10,000 for the first incident, and both of the teams face possible suspension and a fine based on the review of the incident by the commissioner. |

| Human Documents |  |
|-----------------|--|
| Document 1: Bench-clearing brawl - Wikipedia.org | A bench-clearing brawl is a form of ritualistic fighting that occurs in sports, most notably baseball and ice hockey, in which every player on both teams leaves their dugouts, bullpens, or benches, and charges the playing area in order to fight one another or to break up a fight. Penalties for leaving the bench can range from nothing to severe. |
| Document 2: Unlike MLB, the NHL stamped out bench-clearing brawls (www.sbnation.com) | Although this particular brawl earned a place in history, a similar sort of all-hands-on-deck silliness remains a fairly regular feature in baseball. |
| Document 3: Bench-Clearing Brawls Just Not The Same Without Amphetamines (www.theshrubshack.com) | In the glory days of bench clearing brawls, real punches were thrown and real blood was shed, mostly because the players were incredibly high all the time. |

| Bing Documents |  |
|----------------|--|
| Document 1: What does it mean to clear the bench? - TimesMojo | When the team is down, the referee can charge the mound with a bat and has resulted in criminal charges. When was the last time an NHL bench clearing brawl in the NHL was 1987? 85. Fifty percent of the players that suited up in the 1980s had at least one fight. Why do benches clear? Most fights in baseball turn into what is known as a bench-clearing brawl. This is when an entire team's bench, sometimes | 
| Document 2: What does it mean to clear the bench? - TimesMojo | position in the field, the may return to the mound although that rarely happens. Is bat flipping illegal? Canada and the United States. In Canada, and the United States, bats have traditionally been considered rude and inconsistent with baseball etiquette. Traditional etiquette and the unwritten rules of baseball espouse humility and discourage actions which may be interpreted as arrogant or showing up the opponents. Why do catchers throw to third base after a strikeout? Around the Horn If |
| Document 3: Baseball Fighting Rules - rookieroad.com | players that leave their sidelines benches will be subject to some pretty hefty fines. The MLB does not feel like it can afford to automatically fine players for joining a fight because of the way the game is structured. In baseball, when there is a fight between an offensive player and a defensive player, the offense is always going to be outnumbered. That is because unless there are offensive players on base, it will always be an even |
| Document 4: WATCH: Recent brawl between Cincinnati Reds and Chicago - 5 Sportsnut | two sides met again in 2018, Garrett got his shot at revenge. When Cincinnati leading 4-2 in the seventh inning, he struck out Baez to end the inning. They stared each other down after Garrett celebrated the strikeout then started exchanging words. After Baez initiated a fight, the two rivals charged at one another and the benches cleared out. Related: MLB trade rumors - Latest MLB rumors entering July. Needless to say, there is no love lost between these two. Fortunately, it didn't |

| Table 12. Example of documents retrieved by WebGPT, human demonstration and Bing Search API. Document titles are **bolded**. |
```

### --- Page 0023 ---

```markdown
| GPT-3.5                     | GPT-3.5                     | GPT-3.5                     | Alpaca                      | Average  |
|-----------------------------|-----------------------------|-----------------------------|-----------------------------|----------|
| WebGPT docs                 | + Human docs                | + WebGPT docs               |                             |          |
| Human annotations (unshuffled) | 0.2110                      | 0.1316                      | 0.3234                      | 0.2220   |
| unshuffled                  | 0.2094                      | 0.2351                      | 0.0743                      | 0.1445   |
| shuffled                    | 0.1748                      | 0.1310                      | 0.0825                      | 0.1359   |

Table 13: Pearson correlation coefficient computed between the relative location of answer sentence $a_i$ (answer sentence index $i / #$ answer sentences) and the relative location of document sentence $d_j$ (document sentence index $j / #$ document sentences) that support $a_i$. The numbers of human annotations (top row) are computed only on the 100 annotated examples, and the supporting sentences are identified by crowdworkers. Correlation is weaker for GPT-3.5 and marginally stronger for Alpaca when the documents are shuffled.

![Distribution of disagreement patterns in our collected data. O: Supported, Δ: Partially Supported, X: Not Supported.](assets/page_0023_img_1.png)

C.5 Disagreement Patterns of Annotations

We report the percentage of each annotation pattern in Figure 7. O’s denote Supported, triangles denote Partially Supported and X’s denote Not Supported. All annotators agree on 70% of the examples. Two annotators agree on around 26% of the examples. All annotators disagree with each other on 3.4% of the examples.
```

### --- Page 0024 ---

```markdown
![Screenshot of the annotation interface. The documents are shown on the left-hand side, along with the similarity score (SimCSE) to the current answer sentence. The right-hand side shows the question, answer, and the current answer sentence. The annotations go below the box for the current answer sentence.](assets/page_0024_img_1.png)

| Dataset                       | # Ex. | Text to be verified                                         | Evidence Length (Avg. words) |
|-------------------------------|-------|-----------------------------------------------------------|-------------------------------|
| WICE (Kamoi et al., 2023)     | 5.3K  | Sub-claims of Wikipedia sentences                          | 1586.4                        |
| Yue et al. (2023)            | 4K    | Sentence-long answers generated by Chatgpt conditioned on the short answer from a QA dataset, long-form answers generated from commercial search engines | 150.8                         |
| Liu et al. (2023b)           | 11K   | Sentence in long-form answers generated from commercial search engines | 1792.5                        |
| ExpertQA (Malaviya et al., 2023) | 12K   | Sentence in long-form answers to expert-curated questions | 679.3                         |
| SALAD (Ours)                 | 4K    | Sentence in long-form answers generated from LLMs         | 396.0                         |
```



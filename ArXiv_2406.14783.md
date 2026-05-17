# ArXiv 2406.14783

### --- Page 0001 ---

```markdown
# Evaluating RAG-Fusion with RAGE1o: an Automated Elo-based Framework

**Zachary Rackauckas\***  
Columbia University  
New York, NY, USA  
zcr2150@columbia.edu  

**Arthur Câmara**  
Zeta Alpha  
Amsterdam, The Netherlands  
camara@zeta-alpha.com  

**Jakub Zavrel**  
Zeta Alpha  
Amsterdam, The Netherlands  
zavrel@zeta-alpha.com  

---

## ABSTRACT

Challenges in the automated evaluation of Retrieval-Augmented Generation (RAG) Question-answering (QA) systems include hallucination problems in domain-specific knowledge and the lack of gold standard benchmarks for company-internal tasks. This results in difficulties in evaluating RAG variations, like RAG-Fusion (RAGF) in the context of a product QA task at Infineon Technologies. To solve these problems, we propose a comprehensive evaluation framework, which leverages Large Language Models (LLMs) to generate large datasets of synthetic queries based on real user queries in Elo scoring, as well as LLM-as-a-judge to rate retrieved documents and answers, evaluates the quality of answers, and ranks different variants of Retrieval-Augmented Generation (RAG) agents with RAGE1o's automated Elo-based competition. LLM-as-a-judge rating of a random sample of synthetic queries shows a moderate, positive correlation with domain expert scoring in relevance, accuracy, completeness, and precision. While RAGF outperformed RAG in Elo scoring, as a side effect, our results show that RAGF significantly outperforms RAG in completeness, but underperforms in precision. In addition, RAGE1o's RAGF assistant demonstrated significantly higher performance in document relevance based on MRRG scores. We find that RAGE1o positively aligns with the preferences of human annotators, though the evaluation is still required. Finally, RAGF's approach leads to more complete answers than the expert annotations and better answers overall based on RAGE1o's evaluation criteria.

## 1 INTRODUCTION

The text-generating capabilities of LLMs, together with their text understanding abilities, have evolved a new conversational Question-Answering (QA) systems to experience a considerable leap in performance, with near-human text quality and reasoning capabilities [7]. However, these systems can be prone to hallucinations [19, 35], as they sometimes produce seemingly plausible but factually incorrect answers.

The general inability of such models to identify unanswerable questions [2, 38] can exacerbate hallucinations, especially in enterprise settings. In such scenarios, user questions may require specific domain knowledge to be answered properly. This knowledge is usually out-of-domain for most LLMs, but is present in private and confidential internal documents from the company.

One such company is Infineon, a leading manufacturer of semiconductors. Given its wide range of equipment, information about its products is spread across multiple, highly technical documents, including datasheets and detailed design guides of hundreds of pages.

*Work conducted while the author was affiliated with Infineon Technologies.*
```


### --- Page 0002 ---

```markdown
| Table 2: Sample of synthetic queries for evaluating Infinion’s RAG assistant. GPT4 refers to OpenAI’s gpt-4-turbo-2022-84-09 model. Opus, Sonnet and Haiku refer to Anthropic’s Claude 3 models opus-20240229, sonnet-20240229 and haiku-20240230, respectively. |
|----------------|----------------------------------------------------------------------------------------------------------------------------------|
| model          | Query                                                                                                                            |
| GPT4           | What are some typical consumer applications for V11496x-STA/B sensors?                                                          |
| GPT4           | What specific ISO 26262 readiness is available for the KP253 sensor?                                                            |
| Opus           | How small of a form factor can I achieve for a battery-powered air quality device using Infinion’s PAS CO2 sensor?              |
| Sonnet         | Can Infinion’s sensors support bus configurations or daisy-chaining for simplified wiring and reduced complexity in IoT systems? |
|                | Which TLE4971 current sensor models are available in the TISON-8-6 package?                                                     |

| Table 1: Sample of questions submitted by users to the Infinion RAG-Fusion system |
|-------------------------------------------------------------------------------------|
| User-submitted queries                                                               |
| What is the security of origin of IM721D28, and how does geopolitical exposure affect the market and my SM for the microphone? |
| What is the IP rating of mounted IM721D28?                                         |
| Tell me microphones that have been released since January 2023 based on the dataset revision history. |
| We need to confirm whether the IFX waterproof MIC has a sleeping mode and wake-up functions. |

## 2 RELATED WORK
Several evaluation systems for RAG have been proposed to address flaws in current evaluation methods. For instance, Facts as a Function (FaaF) [20] is an end-to-end factual evaluation algorithm specially created for RAG pipelines. By extracting functions from ground truth facts, FaaF focuses on the quality of generation and retrieval by calling LLMs. FaaF has substantially increased efficiency and cost-effectiveness, achieving reduced error rates compared to traditional evaluation methods. The reliance on a set of ground truths does not meet our goal of applying an automated evaluation tool to our pipelines. Recently, researchers have worked to eliminate the need for ground truths. This is especially important when automatically evaluating grants, which typically include technical documents and datasets, such as the Infinion RAG conversational agent. RAGL demonstrates this reliance by using LLMs as-a-judge, a method studied in numerous recent works.

SelfCheckGPT demonstrates the ability to leverage LLMs to detect and rank factual errors with two resources [25]. In addition, it has been demonstrated that GPT3.5 Turbo outperforms ground truth baselines in fact-checking with a "1/2-obs" method [40]. A model built to classify statements as true or false based on the activations of an LLM’s hidden layers had up to 83% classification accuracy [6]. This evidence supports RAG1.0’s usage of LLMs-as-judge.

Automated evaluation metrics can also be applied to RAG-based systems. BARTScore, an automated metric based on the BART architecture, has also outperformed most metrics on categories including factuality [23, 39]. Besides automated evaluation metrics, several automated evaluation frameworks have been created with a similar goal to RAGL. Focusing on faithfulness, answer relevance, and content relevance, RAGAS leverages LLM prompting to focus on situations where ground truths and human annotations are not present in a dataset [13]. Prediction-powered inference aims to decrease the number of human annotations needed for machine learning prediction on a dataset of images of galaxies with approximately 300,000 annotations [3]. The ARES toolkit leverages prediction-powered inference to evaluate RAG systems with fewer human annotations. Like RAGL, ARES automatically evaluates RAG systems using synthetically generated data [31].

ARAGGOG highlights Hypothetical Document Embedding (HyDE) and LLM reranking as effective methods for enhancing retrieval precision while also exploring the effectiveness of Sentence-Wise Retrieval and the potential of the Document Summary Index in improving RAG systems [12].
```

### --- Page 0003 ---

```markdown
# Evaluating RAG-Fusion with RRF as an Automated Elo-based Framework

![A traditional Retrieval-Augmented Generation pipeline compared to a RAG-Fusion pipeline. While a traditional RAG agent submits only the original query to the search system, a RAGF agent first generates variations of the user query and combines the rankings induced by these queries into a final ranking using RRF. The resulting top-k passages are fed into the LLM for generating the answer to the user's query.](assets/page_0003_img_1.png)

While the aforementioned frameworks evaluate answers on relevance, faithfulness, and correctness metrics, RAG can also be evaluated on noise and counterfactual robustness, negative rejection, and information integration [8].

In addition to answers, frameworks have also been created to evaluate documents. Corrective Retrieval Augmented Generation (CRAG) builds on RAG by employing a retrieval evaluator to ensure that only the optimal documents are fed into the LLM prompt during the retrieval phase [36].

Due to its Elo-based ranking system for answers, its use of LLM-as-a-judge, and its reliance on the intermediate retrieval steps in a RAG pipeline, RAGF is a unique evaluation toolkit. In this study, we set out to compare a simple RAG versus a more sophisticated RAGF system on a knowledge-intensive industry-specific domain.

## 3 RETRIEVAL AUGMENTED QA WITH RANK FUSION

While answers generated by traditional retrieval augmented systems are based on a number of documents retrieved from a single query, RAGF introduces additional variation into the retrieval process. Upon receiving a query from the user, a RAGF agent leverages a large language model to generate a set of queries based on the agent's understanding of the user query [15]. Table 3 shows examples of queries generated by the agent based on the user query, “How to cross-sell a MEMS microphone and a XENSIV sensor to customers?”.

After generating the variations of the user query, the RAGF agent submits the original and the generated queries to a retrieval system [29] that returns the top-k relevant documents $d_1, d_2, \ldots, d_k$ from the set of all documents D for each query. The rankings induced by these queries are then combined using reciprocal rank fusion (RRF) [9] into a final, higher-quality set of passages. The intuition behind RAGF is that submitting variations of the same query and combining the final rankings increases the likelihood of relevant passages being injected into the LLM prompt. In contrast, non-relevant passages retrieved by a single query are discarded [2].

Table 3: Queries Generated from “How to cross-sell a MEMS microphone and a XENSIV sensor to customers?”

| LLM-Generated Query |
|----------------------|
| What are the key features of Infineon’s MEMS microphones and XENSIV sensors that can be highlighted while cross-selling? |
| How can Infineon’s MEMS microphones and XENSIV sensors be integrated for enhanced audio and motion sensing capabilities in various applications? |
| What are the most suitable applications and industries for Infineon’s MEMS microphones and XENSIV sensors to maximize cross-selling potential? |

$$
RRFScore(D) = \sum_{i=1}^{k} \frac{1}{r(d_i) + k}
$$

## 4 DEVELOPMENT OF A SYNTHETIC TEST SET

As previously discussed, one of the main issues when evaluating the quality of a QA system in an enterprise setting is that, frequently, companies do not have a large enough existing collection of queries to evaluate such systems' quality. Therefore, in this work, we propose to adopt a strategy previously used by methods for generating synthetic queries for training retrieval systems, such as InPars [18] and Promptgator [10].

Similar to these approaches, we randomly sample passages from documents within our collection and prompt LLM to generate questions that users may ask about these portions. However, one difference in our approach to generating training queries is the size of these passages. When generating queries for training a retrieval system, we ideally want to keep the passages short for the system's encoder's relatively short context windows. However, when generating queries for evaluating QA systems (including retrieval augmented), we are not bound to the limit of the embedding model used for retrieval. Rather, a longer passage may yield questions
```

### --- Page 0004 ---

```markdown
![Process for creating synthetic queries. We prompt multiple LLMs to generate queries based on existing documents. We include some existing user queries in the prompt as few-shot examples.](assets/page_0004_img_1.png)

## 5 LLM-AS-A-JUDGE FOR RAG PIPELINES

Even with a suitable set of synthetic queries for evaluating our RAG conversational agent, assessing whether a given answer properly answers a question is not trivially done. If a ground-truth “golden answer” is available, one can use traditional syntactic-based metrics such as BLEU, METEOR or ROUGE [22, 24, 27]. Without such reference answers, one would require human raters with a considerable understanding of the question’s topic to manually assess the quality of the answers produced by each system. However, this is a costly process.

Alternatively, several LLM-as-a-Judge methods have been proposed, where another LLM is asked to evaluate the quality of answers generated by other LLMs. Nevertheless, in an enterprise setting, the answers usually require the LLM to access knowledge not present in their training datasets but rather contained in documents internal to the company. This is usually accomplished using a RAG pipeline like the one described above. Therefore, the judging LLM also needs access to similar knowledge to accurately evaluate the agent’s answers’ quality.

Therefore, in this work, we rely on RAGLEO, an open-source RAG evaluation toolkit that evaluates the answers generated by the agent and the documents retrieved by them. By leveraging the annotation of retrieved documents, provided by the agents being evaluated, on the answer evaluation step, this method allows the judging LLM to evaluate if the generated answers are viable to use all the information available about the question properly and to check for any hallucinations. As the documents used for generating the answers are included in the answer evaluating prompt, an agent that incorrectly cites information from a source it has not seen in these documents is likely hallucinating and should have its evaluation adjusted accordingly. As we explore in Section 8, this two-step process results in a high correlation between human expert annotators and the judging LLM, enabling higher reliability and trust when evaluating different RAG pipelines. This process is also illustrated in Figure 3.

### 5.1 Evaluation aspects

While our main evaluation focuses on the pairwise comparison between the two agents, RAGELO also allows us to evaluate answers pointwise. In this setting, similar to other works [33], we prompt the judging LLM to evaluate the answers according to multiple criteria:

- **Relevance:** Does the answer address the user’s question?
- **Accuracy:** Is the answer factually correct, based on the documents provided?
- **Completeness:** Does the answer provide all the information needed to answer the user’s question?
- **Precision:** If the user’s question is about a specific product, does the answer provide the answer for that specific product?
```

### --- Page 0005 ---

```markdown
![The RAGE10 evaluation pipeline. First, documents retrieved by the agents are evaluated positive according to their relevance to the user's question. Then, the agents' answers are evaluated pairwise, using the retrieved relevant documents from both agents as reference.](assets/page_0005_img_1.png)

## 6 RETRIEVAL PIPELINES
We not only experiment with different search agents (i.e., RAG and RAGF. We are also interested in how different retrieval methods may impact the quality of the final answers generated by these agents.

### 6.1 Retrieval methods
Our corpus consists of passages extracted from the Infineon XENSIV Product Selection Guide, a 117-page document with detailed information on every product in the XENSIV family. This document included technical information about all Infineon XENSIV sensors, consumer and automotive sensor applications, guidance in selecting the correct sensor, and other comprehensive and detailed information about the product line.

The passages are embedded using multilingual-e5-base [34] and indexed using OpenSearch, allowing us to perform both KNN-based vector search, keyword-based search with BM25 [30], and RRF based hybrids thereof.

### 6.2 QA Systems Implementation
We mainly evaluate two agents: a native RAG pipeline, where the agent first retrieves top-k passages that are then templated into a prompt, and the Infineon RAG-Fusion (RAGF) agent. Upon receiving a query, a native RAG agent takes the following actions:
1. Retrieve the top k most relevant passages from the search system.
2. Perform a Chat Completions API call, prompting the LLM with instructions for generating an answer based on the five relevant passages.
3. Process and output the Chat Completions response.

Meanwhile, the Infineon RAGF conversational assistant uses a similar framework and performs the following steps upon receiving a query:
1. Perform a Chat Completions API call to generate four new queries based on the original query using a prompt tailored to the agent's original goal.
2. Retrieve the top k most relevant passages for each query.
3. Using RRF, combine the top-k passages induced by all queries into a final answer.

## 7 EXPERIMENTS
### 7.1 Comparing LLM-as-a-judge to expert annotators
While LLM-as-a-judge is a theoretically viable algorithm for rating RAG and RAGF answers, we established whether the results agree with the annotations of domain experts.

Figure 4 provides a Bland-Altman plot to visually represent the LLM and human judgments' agreement.

![Bland-Altman plot to visualize the comparison between LLM-as-a-judge and expert answers.](assets/page_0005_img_2.png)

The bias of approximately 0.12 indicates that, on average, LLM scores were slightly higher than human scores. The limits of agreement ranged from approximately -1.17 to 1.41, demonstrating substantial variability in the difference between LLM and human evaluators.
```

### --- Page 0006 ---

```markdown
Next, we compared LLM-as-a-judge to expert annotators with Kendall’s $τ$. Kendall’s $τ$ is a nonparametric measure that quantifies the degree of association between two monotonic continuous or ordinal variables by calculating the proportion of concordance and discordance among pairwise ranks, offering valuable insight into their rank correlation [11, 28]. We used the SciPy Stats Kendall function to calculate a tau score and a p-value for the combined ratings of all columns, flattened into a 1-D array with RAG and RGF ratings combined [1]. The tau-b value, a nonparametric measure of association, is calculated using the following formula [21]:

$$
τ_b = \frac{(P - Q + T)}{(P + Q + U)} \tag{2}
$$

$P$ represents the number of concordant pairs, $Q$ represents the number of discordant pairs, $T$ represents the number of ties exclusive to $x$, and $U$ represents the number of ties exclusive to $y$.

This test returned $τ = 0.56$, indicating a moderate, positive correlation [32] with $p < 0.01$ given very high probabilities of no association at $p < 0.001$99% confidence level]. For comparison, in similar experiments judging human versus LLM judgments, Fagioli et al. found $τ$ values of $τ = 0.76$ and $τ = 0.86[14].

Following the same methodology, we also calculated Spearman’s $ρ$, a similar nonparametric correlation measure. This resulted in $ρ = 0.59$ with $p < 0.01$, demonstrating a statistically significant, moderate positive correlation [28].

### 7.2 RAG vs RGF

#### 7.2.1 Quality of retrieval

We assessed retrieval quality using Mean Reciprocal Rank@5 (MRR@5), which averages the inverse ranks of the first relevant document within the top five positions across all queries. The formula is given by

$$
MRR@5 = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i} \tag{3}
$$

where $|Q|$ is the total number of queries and $rank_i$ is considered only if it’s within the top five, otherwise it counts as zero [17]. MRR@5 scores were calculated for each agent and each retrieval method considering two categories:

(1) MRR@5 score for documents deemed “somewhat relevant” or “very relevant.”

The results can be seen below in Table 4.

| Agent   | Retrieval Method | Very Relevant | Somewhat Relevant |
|---------|------------------|---------------|-------------------|
| RAG     | KNN              | 0.407         | 0.828             |
| RAG     | Hybrid           | 0.821         | 0.955             |
| RAG-F   | KNN              | 0.746         | 0.949             |
| RAG-F   | BM25             | 0.396         | 0.810             |
| RAG-F   | Hybrid           | 0.855         | 0.970             |
| RAG-F   | Hybrid           | 0.758         | 0.961             |

#### 7.2.2 Pairwise evaluation of answers.

When we ran RAGEL 209 to evaluate end-to-end answer quality of RAG vs RGF with different base retriever configurations a task that cannot rely on standard Information Retrieval metrics. These RAGEL 209 results show more victories for RAG than RGF; for example, when using BM25 as a base retriever, RAGF won 49% of the games, RAG won 14.5%, and RAG and RGF are tied in 36.5% of the times. The resulting ELO scores for all six variants are shown in table 6, which gives a robust ranking of the systems, without reliance on a gold standard. It is interesting to see that, for both RAGF as well as RAG, BM25 is a strong baseline that is not surpassed by generative embeddings in these experiments.

Next, we compared the RAGEL 209 outcome to the preference of our Infinet human annotator. We performed two-tailed paired t-tests to compare RAG against RAGF on each category from the Infinet representatives’ human evaluations with a $α = .05$. As expected, due to larger variety of retrieved results, RAGF significantly outperforms RAG in completeness at the 95% confidence level with $p < 0.01$. However, on the precision metrics, RAG significantly outperformed RAGF at the 95% confidence level with $p < 0.04$.

### 8 DISCUSSION

As observed above, we found statistically significant, moderate positive correlations between LLM ratings and human annotations. This indicates a consistent association between the ratings from LLM-as-a-judge and those by Infinet workers. We find that on average, LLM scores are slightly higher than those of human annotations which could not be fully reliable, and IR metrics derived from LLM-as-a-judge should not be equated with regular relevance scores without further calibration, we can still make good use of this approach to rank-order systems. These findings collectively support the validity of our LLM evaluation method, which assesses conversational system outputs based on a combination of relevance, accuracy, completeness, and recall.

The style of evaluation and the different dimensions it takes into account are specified in the prompts given to the LLM in the RAGEL 209 evaluation, which are provided in Appendix A. Specifically, while the initial LLM-as-a-judge is given precise criteria to focus on only four categories, we instructed RAGEL 209 to include more than the initial four categories:

> Your evaluation should consider factors such as comprehensiveness, correctness, helpfulness, completeness, accuracy, depth, and level of detail of their responses.

Since RAGF significantly outperformed RAG in the completeness category, the RAGEL 209 judge LLM likely weighted completeness higher than precision. In addition, based on manual observation of a small random sample of answers, RAGF produced more comprehensive answers and featured higher depth and level of detail due to the multiple query generation. However, games where RAG won were most likely influenced by a significantly more precise answer than that of RAGF. While RAGF values comprehensive answers that offer multiple perspectives to the user, RAG produces shorter answers that answer the original query only. Since completeness is defined as the extent to which a user’s question was answered, it can be presumed that RAGF’s longer and more comprehensive answers
```

### --- Page 0007 ---

```markdown
| **Table 5: RAG vs RAGF Win percentage between pairwise comparison of the agent’s answers using GPT-40 as a judge with RAGE10.** |
|:----------------------------------------------------------:|
| **Agent** | **BM25** | **KNN** | **Hybrid** | **AVG** |
|-----------|----------|---------|------------|---------|
|           | RAG      | RAGF    | RAG        | RAGF    |
| **BM25**  | RAG      | —       | 14.5%      | 49.5%   | 52.5%   | 29.0%   | 28.5%   | 34.8%   |
|           | RAGF     | 49.0%   | 58.5%      | 51.5%   | 53.5%   | 30.5%   | 48.6%   |
| **KNN**   | RAG      | 33.0%   | 27.0%      | —       | 20.0%   | 26.0%   | 31.0%   | 27.4%   |
|           | RAGF     | 34.5%   | 30.0%      | 37.0%   | 20.0%   | 30.5%   | 32.0%   | 32.8%   |
| **Hybrid**| RAGF     | 41.5%   | 21.0%      | 51.5%   | 48.0%   | 20.5%   | 36.0%   | 44.3%   |

| **Table 6: Elo Ranking for all agents averaged over 500 tournaments.** |
|:----------------------------------------------------------:|
| **Agent** | **Retrieval** | **Elo score** |
|-----------|---------------|----------------|
| RAGF      | BM25          | 571.0          |
| RAGF      | Hybrid        | 550.0          |
| RAG       | Hybrid        | 497.0          |
| RAGF      | BM25          | 487.0          |
| RAGF      | KNN           | 470.0          |
| RAG       | KNN           | 436.0          |

## 9 CONCLUSION
Overall, we found that the evaluation framework proposed by RAGE10 positively aligns with the preferences of human annotators for RAG and RAGF with due caution due to a moderate action and variability of scoring. We found that the RAGF approach leads to better answers most of the time, according to the RAGE10 evaluation. According to expert scoring, the RAGF approach significantly outperforms in completeness compared to RAG but significantly underperforms in precision compared to RAG. Based on these results, we cannot confidently assert that RAGF's approach leads to better answers generally. However, the results do support that RAGF's approach leads to more complete answers and a higher proportion of better answers under evaluation by RAGE10.

Since RAGE10 is generally applicable to retrieval-augmented algorithms, in future work, we also intend to test different models other than RAG and RAGF, including those with different LLMs. In addition, due to RAGF's underperforming in sensitivity in expert ratings, especially where the LLMs should be on relevant human sensitivities.

## Acknowledgments
We would like to thank the Inference team for his support during this work. We thank the Inference team as well for providing valuable feedback.

## REFERENCES
[1] [a.i.](https://arxiv.org/abs/2305.13712). Retrieved April 24, 2024 from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html#scipy.stats.kendalltau.  
[2] Anastasios K. Anastasopoulos, Stephen B. H. Clare, Flare Anning, Michael I. Jordan, and Tanjin Zame. 2023. Prediction-Proofed Retrieval: arXiv:2305.13712.  
[3] Anthropic. 2024. The Claude 3 Model Family. Opus, Sonnet, Haiku, and the Prototyped Claude.  
[4] Negral Amadei and Charles J. Clarke. 2024. A Comparison of Methods for Evaluating Generative R. https://arxiv.org/abs/2401.00414.  
[5] Anna Zaria and Tom Ritchie. 2023. The Internal State of LLM Knows When It’s Wrong. arXiv:2301.17341.  
[6] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Tarlav Bhardwaj, Arvind Neelakantan, Pranay Shyam, Girish Sastry, Amanda Askell, Dhruv Gaur, Aditi Ramesh, Yotam Tsur, Hemang Jha, Rewon Child, Asim A. Alhassan, Daniel M. Ziegler, Kelsey M. Gibbons, Winter Stokes, Jack Clark, Christopher Berner, Misha Laskin, Sam McCandlish, Jared Sutzker, and Ori Anav. 2020. Language Models are Few-Shot Learners. https://doi.org/10.48550/arxiv.2005.14165.  
[7] Lawrence Chen, Hongyu Lin, Amina Le and Leo Sun. 2023. Improving Language Models in Retrieval-Augmented Learning.  
[8] Gordon V. Cormack, Charles A. L. Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms condensed individual rank learning methods.
```

### --- Page 0008 ---

```markdown
# Zachary Rackauckas, Arthur Cimara, and Jakub Zawel

**SIGIR 2023 (Boston, MA, USA) (SIGIR '23). Association for Computing Machinery, New York, NY, USA, 754–759. https://doi.org/10.1145/1671191.1671214**

[1] Ziyuan Zhao, Vincent Y. F. Ma, Y. Lian, Jun Ping, J. Anton Bakalov, Kevin E. Y. Chen, and Ming-Wang Chang. 2022. In-Context Few-Shot Data Retrieval. arXiv:2201.05896 [cs.CL]  
[2] Nicholas D. Edwards, Eric J. B. G. and Stephen T. R. E. F. 2023. Graphing Networks for Explanations. arXiv:2306.08468 [stat.ML]  
[3] Matouš Bělohlávek, Shivang Nagpal, and Alexander Fred-Ojala. 2023. ARAGOG: Advanced Graph Augmented Grading. arXiv:2401.00767 [cs.CL]  
[4] Ashish R. B. H. J. James, Luis Espinosa-Anke, and Steven Schockaert. 2023. RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15717 [cs.CL]  
[5] Guglielmo Faggioli, Laura Dietz, Charles L. A. Caire, Gianluca Demartini, Matthias Fagni, Claudia H. Forth, Kondo Vanda, Evgeny Zaitsev, Martin Potthast, Benno Stein, and H. H. Wang. 2023. Perspectives on Large Language Models for Relevance Judgment. In Proceedings of the 2023 ACM SIGIR International Conference on Theory of Information Retrieval (ICTIR '23). ACM, https://doi.org/10.1145/3631675.3631676  
[6] Fatih F. A. 2024. Toward Optimising a Retrieval Augmented Generation Pipeline. arXiv:2401.00767 [cs.CL]  
[7] He Huang, Lingyu Zhang, and Lijuan Ma. 2022. An Empirical Study of Multi-Task Learning for Document Judge Models. arXiv:2401.00767 [cs.CL]  
[8] Yujing Jiang, Ruoi Biondich, Hugo Zhonina, Marika Fardel, Roberto Lott, Zhenyu Zhao, and Reforging Networks. 2023. Rerankers: 12 Papers, 12 Languages. arXiv:2401.00767 [cs.CL]  
[9] A. M. A. A. 2023. The impact of its long-term impacts. Biometrics, 30(1), 1-10. https://doi.org/10.1016/j.biometrics.2023.01.001  
[10] M. K. A. 2023. The evaluation with high levels of correlation with human judgments. In SMiRT 2007 (Proc. Cape Reefs) (JUR-77). Association for Computational Linguistics, 423-428.  
[11] Mike Lewis, Yihan Liu, Naman Goyal, Marjan Ghazvininejad, Aldebaran Shakib, and Yevgeny Lee. 2020. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. arXiv:1910.13461 [cs.CL]  
[12] Ian T. S. R. 2023. A Package for Generating Evaluations of Summarises. Resource Spain, 74–81. https://andromeda.org/Wor-703.131  
[13] P. A. 2023. Multilingual Document Detection for Generative Language Models. arXiv:2305.38956 [cs.CL]  
[14] OpenAI. 2022. GPT-3 Turbo and GPT-4. arXiv:2206.03798 [cs.CL]  
[15] Philip Resnik, Salim Roukos, Ted Briscoe, and Wei-Jing Zhu. 2002. BLEU: a method for automatic evaluation of machine translation. In ACL 2002 (Philadelphia, Pennsylvania) (ACL '02). Association for Computational Linguistics, USA, 311–318. https://doi.org/10.3115/1073083.1073135  
[16] Samy P. 2022. Efficient inference for Kendall's tau. arXiv:2206.01419 [cs.LG]  
[17] Zachary Rackauckas. 2023. Rag-Fusion: A New Take on Retrieval Augmented Generation. International Journal on Natural Language Computing 13, 1 (Feb. 2022), 97–47. https://doi.org/10.5121/ijnlc.2023.11303  
[18] David A. D. S. 1994. Query Expansion. In Proceedings of the Third Text Retrieval Conference, TREC-3, (Gaithersburg, Maryland, USA, November 2–4, 1994). NIST Special Publication No. 500-225. 139–146. http://trec.nist.gov/pubs/trec3/proceedings/exp.pdf  
[19] Jon S. A. 2021. Omar Khattab, Christopher Potts, and M. Ait Ziane. 2024. Towards an Automated Framework for Retrieval-Augmented Generation Systems. arXiv:2311.04925 [cs.CL]  
[20] Patrick Sobolev, Christof Bär, and Lothar A. Schwartze. 2018. Correlation Coefficients: A New Approach to User Interpretation. Aesthetics & Analogies 5, 1 (May 2018), 165–176. https://doi.org/10.2121/0000000000000063  
[21] David Thomas, Seth Spideman, Nick Crawsell, and Bhaskar Mitra. 2023. arXiv:2306.10261 [cs.IR]  
[22] Liang Wang, Nan Yang, Xiaolong Huang, Yunqi Wang, Rangan Majumder, and Furui Wei. 2024. Multilingual ES Text Embeddings: A Technical Report. arXiv:2401.00257  
[23] Yujin Zhao and William Yang Wang. 2021. On Hallucination and Predictive Uncertainty in Natural Language Generation. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume. Association for Computational Linguistics, 125–135. https://doi.org/10.18653/v1/2021.eacl-main.12  
[24] Shi-Qi Jian, Jie Chen, Guo Zhu, and Zhen-Hua Ling. 2024. Corrective Retrieval Augmented Generation. arXiv:2401.15884 [cs.CL]  
[25] Yizhang Yang, Alistair Moffat, and R. P. 2023. Prompt Judgments: Preference, Absolute, and Ratio. In Proceedings of the 9th Australasian Document Computing Symposium (Dunedin, New Zealand) (ADCS '23). Association for Computing Machinery, New York, NY, USA, Article 3, 5 pages. https://doi.org/10.1145/3292199.3292199  
[26] Zhangyi Zhou, Xin Zhang, Yuxin Chen, Xiying Dong, and Xuan-Jing Huang. 2023. Do Large Language Models Know What They Don’t Know? In Finding of the Association for Computational Linguistics: ACL 2023, 8653–8655.  
[27] Weiwei Guo, Guansheng He, and Pengfei Liu. 2023. BERTScore: Revisited. arXiv:2401.00767 [cs.CL]  
[28] Zhenyu Zhang, Hongxin Wu, Danyong He, and Jinless Glass. 2023. Integrating Hybrid Language Checking. arXiv:2401.00767 [cs.CL]  
[29] Lianming Zhang, Wei Ling, Sheng Sheng, Siyuan Zhang, Zhanhao Wu, Yonghe Chen, Zai Lin, Zhiouan Li, and Zhangjie Li. 2023. RAG with MFT-based Document and Answers. arXiv:2403.03658  

## A. RAGELS PROMPTS AND CONFIGURATIONS

You are an expert document annotator. Your job is to evaluate whether a document contains relevant information to answer a user’s question.

Please act as an impartial relevance annotator for a search engine. Your goal is to evaluate the relevancy of the documents given as your input.

You should write one sentence explaining why the document is relevant or not for the user question. A document can be:
- Not relevant: The document is not on topic.
- Somewhat relevant: The document is on topic but does not fully answer the user question.
- Very relevant: The document is on topic and answers the user’s question.

[user question]  
`{query}`  
`{document content}`  
`{document}`  

## A.2 Answer evaluators

For the positive evaluator used in Section 5.1, we used the following prompt with RAGEL's CustomPromptAnswerEvaluator:

You are an impartial evaluator for evaluating the quality of the responses provided by an AI assistant tasked to answer users’ questions about the catalogue of IoT sensors produced by Infinitum.

You will be given the user’s question and the answer produced by the assistant.  
The answer’s was generated based on a set of documents retrieved by a search engine.
```

### --- Page 0009 ---

```markdown
# Evaluating RAG-Fusion with RAGEl10: an Automated Elo-based Framework

You will be provided with the relevant documents retrieved by the search engine. Your task is to evaluate the answer's quality based on the response's relevance, accuracy, and completeness.

## Rules for evaluating an answer:
- **Relevance**: Does the answer address the user's question?
- **Accuracy**: Is the answer factually correct, based on the documents provided?
- **Completeness**: Does the answer provide all the information needed to answer the user's question?
- **Precision**: If the user's question is about a specific product, does the answer provide the answer for that specific product?

## Steps to evaluate an answer:
1. **Understand the user's intent**: Explain in your own words what the user's intent is, given the question.
2. **Check if the answer is correct**: Think step-by-step to determine whether the answer correctly answers the user's question.
3. **Evaluate the quality of the answer**: Evaluate the quality of the answer based on its relevance, accuracy, and completeness.
4. **Assign a score**: Produce a single JSON object with the following keys, each with a single score between 0 and 2, where 2 is the highest score on that aspect:
   - **relevance**:
     - 0: The answer is not relevant to the user's question.
     - 1: The answer is partially relevant to the user's question.
     - 2: The answer is fully relevant.
   - **accuracy**:
     - 0: The answer is factually incorrect.
     - 1: The answer is partially correct.
     - 2: The answer is fully correct.
   - **completeness**:
     - 0: The answer does not provide enough information to answer the user's question.
     - 1: The answer only answers some aspects of the user's question.
     - 2: The answer fully answers the user's question.
   - **precision**:
     - 0: The answer does not mention the same product or product line as the user's question.
     - 1: The answer mentions a similar product or product line, but not the same as the user's question.
     - 2: The answer mentions the exact same product or product line as the user's question.

The last line of your answer must be a SINGLE LINE JSON object with the keys "relevance", "accuracy", "completeness", and "precision", each with a single score between 0 and 2.

[DOCUMENTS RETRIEVED]
- (documents)
- [User Query]
- [query]
- [Agent answer]
- [answer]

For the pairwise evaluation between agents used for the results in Tables 5 and 6, we used RAGEl10's `PairwiseAnswerEvaluator` with the following parameters:

```python
pairwise_evaluator_config = PairwiseEvaluatorConfig(
    n_games_per_query=15,
    has_citations=False,
    include_raw_documents=True,
    include_annotations=True,
    document_relevance_threshold=2,
    factors=['the comprehensiveness', 'correctness', 'helpfulness', 'completeness', 'accuracy', 'depth', and level of detail of their responses. Answers are comprehensive if they show the user multiple perspectives in addition to but still relevant to the intent of the original question.'],
)
```

This generates 15 random games between two agents per query (i.e., all possible unique games for a given pair) and tells the evaluator that:
- The answers do not include specific citations to any passage (has_citations=False).
- Include the full text of the retrieved passages in the evaluation prompt (include_raw_documents=True).
- Inject the output of the retrieval evaluation into the prompt (include_annotations=True).
- Ignore any passage with a relevance score below 2 (document_relevance_threshold=2).
- Consider these factors when selecting the best answer factors.

These parameters produce the following final prompt used for evaluation:

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the question below based on a set of documents retrieved by a search engine.

You should choose the assistant that best answers the user question based on a set of reference documents that may or may not be relevant.

For each reference document, you will be provided with the text of the document as well as reasons why the document is or is not relevant.

Your evaluation should consider factors such as comprehensiveness, correctness, helpfulness, completeness, accuracy, depth, and level of detail of their responses. Answers are comprehensive if they show the user multiple perspectives in addition to but still relevant to the intent of the original question.

Details are only useful if they answer the user's question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Begin your evaluation by explaining why each answer correctly answers the user's question. Then, you should compare the two responses and provide a short explanation of their differences. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
```

### --- Page 0010 ---

```markdown
# Zackary Rackauckas, Arthur Camara, and Jakub Zavrel

| **After providing your explanation, output your final verdict by strictly following this format:** | **{documents}** |
|---------------------------------------------------------------------------------------------------|------------------|
| if assistant A is better, "[EA]" if assistant B is better, and "[EC]" for a tie.                 |                  |

| **[User Question]** | **{query}** |
|----------------------|--------------|
|                      |              |

| **[Reference Documents]** |                  |
|---------------------------|------------------|
|                           |                  |

| **[The Start of Assistant A's Answer]** | **(answer_a)** |
|------------------------------------------|-----------------|
|                                          |                 |
| **[The End of Assistant A's Answer]**   |                 |

| **[The Start of Assistant B's Answer]** | **(answer_b)** |
|------------------------------------------|-----------------|
|                                          |                 |
| **[The End of Assistant B's Answer]**   |                 |
```



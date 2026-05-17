# ArXiv 2305.06983

### --- Page 0001 ---

```markdown
# Active Retrieval Augmented Generation

**Zhengbao Jiang\***, **Frank F. Xu\***, **Luyu Gao\***, **Zhiqing Sun\***, **Qian Liu\***, **Jane Dwivedi-Yu\***, **Yiming Yang\***, **Jamie Callan\***, **Graham Neubig\***  
1Language Technologies Institute, Carnegie Mellon University  
2Sea AI Lab  
3FAIR, Meta  
{zhengbaj,fangzh,luuyg,zhiqings,neubig}@cs.cmu.edu  

## Abstract

Despite the remarkable ability of large language models (LMs) to comprehend and generate language, they have a tendency to hallucinate and create factually inaccurate output. Augmenting LMs by retrieving information from external knowledge resources is one promising solution. Most existing retrieval augmented LMs employ a retrieve-and-generate setup that only retrieves information once based on the input. This is limiting, however, in more general scenarios involving generation of long texts, where continually gathering information throughout generation is essential. In this work, we proposed a generalized view of **active retrieval augmented generation**, methods that actively decide when and what to retrieve across the course of the generation. We propose Forward-Looking Active Retrieval augmented generation (FLARE), a generic method which iteratively uses a prediction of the upcoming sentence to anticipate future content, which is then utilized as a query to retrieve relevant documents to regenerate the sentence if it contains low-confidence tokens. We test FLARE along with baselines comprehensively over 4 long-form knowledge-intensive generation tasks/datasets. FLARE achieves superior or competitive performance on all tasks, demonstrating the effectiveness of our method.¹

## 1 Introduction

Generative language models (LMs) (Brown et al., 2020; Ouyang et al., 2022; OpenAI, 2023; Chowdhery et al., 2022; Zhang et al., 2022; Tourvon et al., 2023; Zhao et al., 2023) have become a foundational component in natural language processing (NLP) systems with their remarkable abilities. Although LMs have memorized some world knowledge during training (Petroni et al., 2019; Roberts et al., 2020; Jiang et al., 2020), they still tend to hallucinate and create imaginary content (Maynez et al., 2020; Zhou et al., 2021). Augmenting LMs with retrieval components that look up relevant information from external knowledge resources is a promising direction to address hallucination (Khandewal et al., 2020; Izacard et al., 2022).

Retrieval augmented LMs commonly use a retrieve-and-generate setup where they retrieve documents based on the user’s input, and then generate a complete answer conditioning on the retrieved documents (Chen et al., 2017; Guu et al., 2020; Lewis et al., 2020; Izacard and Grave, 2021; Sachan et al., 2021; Lee et al., 2021; Jiang et al., 2022; Izacard et al., 2022; Nakano et al., 2021; Qian et al., 2023; Lazaridou et al., 2022; Shi et al., 2023). These single-time retrieval augmented LMs outperform purely parametric LMs, particularly for short-form knowledge-intensive generation tasks such as factoid question answering (QA) (Kwiatkowski et al., 2019; Joshi et al., 2017), where the information needs are clear in the user’s input, and it is sufficient to retrieve relevant knowledge once solely based on the input.

Increasingly powerful large LMs have also demonstrated abilities in more complex tasks that involve generating long-form output, such as long-form QA (Fan et al., 2019; Stelmach et al., 2022), open-domain summarization (Cohen et al., 2021; Hayashi et al., 2021; Giorgi et al., 2022), and (chain-of-thought; CoT) reasoning (Wei et al., 2022; Ho et al., 2020; Geva et al., 2021; Hendricks et al., 2020). In contrast to short-form generation, long-form generation presents complex information needs that are not always evident from the input alone. Similar to how humans gradually gather information as we create content such as papers, essays, or books, long-form generation with LMs would require gathering multiple pieces of knowledge throughout the generation process. For example, to generate a summary about a particular topic, the initial retrieval based on the topic name
```

### --- Page 0002 ---

```markdown
![An illustration of forward-looking active retrieval augmented generation (FLARE).](assets/page_0002_img_1.png)

Step 1  
$s_1$  
Joe Biden (born November 20, 1942) is the 46th president of the United States.  
$s_{1}^1$  
Joe Biden (born November 20, 1942) is the 46th president of the United States.  

Step 2  
$s_2$  
Joe Biden attended the University of Pennsylvania, where he earned a law degree.  
$s_{2}^1$  
He graduated from the University of Delaware in 1965 with a Bachelor of Arts in history and political science.  

Step 3  
$s_3$  
Joe Biden announced his candidacy for the 2020 presidential election on August 18, 2019.  
$s_{3}^1$  
Joe Biden announced his candidacy for the 2020 presidential election on April 25, 2019.  

(e.g., Joe Biden) may not cover all aspects and details. It is crucial to retrieve extra information as needed during generation, such as when generating a certain aspect (e.g., Joe Biden’s education history) or a specific detail (e.g., the date of Joe Biden’s presidential campaign announcement).

Several attempts have been made to retrieve multiple times throughout generation. These attempts include methods that passively use the past context to retrieve additional information at a fixed interval (Khandelwal et al., 2020; Borgeaud et al., 2022; Ram et al., 2023; Trivedi et al., 2022) which might not accurately reflect what LMs intend to generate in the future or retrieve at inappropriate points. Some works in multi-hop QA decompose the full question into sub-questions, each of which is used to retrieve extra information (Press et al., 2022; Yao et al., 2022; Khot et al., 2022; Khattab et al., 2022).

We ask the following question: can we create a simple and generic retrieval augmented LM that actively decides when and what to retrieve throughout the generation process, and are applicable to a variety of long-form generation tasks? We provide a generalized view of active retrieval augmented generation. Our hypothesis regarding when to retrieve is that LMs should retrieve information only when they lack the required knowledge to avoid unnecessary or inappropriate retrieval that occurs in passive retrieval augmented LMs (Khandelwal et al., 2020; Borgeaud et al., 2022; Ram et al., 2023; Trivedi et al., 2022). Given the observation that large LMs tend to be well-calibrated and low probability/confidence often indicates a lack of knowledge (Kadavath et al., 2022), we adopt an active retrieval strategy that only retrieves when LMs generate low-probability tokens. When deciding what to retrieve, it is important to consider what LMs intend to generate in the future, as the goal of active retrieval is to benefit future generations. Therefore, we propose anticipating the future by generating a temporary next sentence, using it as a query to retrieve relevant documents, and then regenerating the next sentence conditioning on the retrieved documents. Combining the two aspects, we propose Forward-Looking Active REtrieval augmented generation (FLARE), as illustrated in Figure 1. FLARE iteratively generates a temporary next sentence, use it as the query to retrieve relevant documents if it contains low-probability tokens and regenerate the next sentence until reaches the end.

FLARE is applicable to any existing LMs at inference time without additional training. Con-
```

### --- Page 0003 ---

```markdown
## 1. Introduction

Considering the impressive performance achieved by GPT-3.5 (Ouyang et al., 2022) on a variety of tasks, we examine the effectiveness of our methods on text-davinci-003. We evaluate FLARE on 4 diverse tasks/datasets involving generating long outputs, including multihop QA (2WikiMultihopQA), commonsense reasoning (StrategyQA), long-form QA (ASQA), and open-domain summarization (WikiAsp) (Ho et al., 2020; Geva et al., 2021; Stelmach et al., 2022; Hayashi et al., 2021). Over all tasks, FLARE achieves superior or competitive performance compared to single-time and multi-time retrieval baselines, demonstrating the effectiveness and generalizability of our method.

## 2. Retrieval Augmented Generation

We formally define single-time retrieval augmented generation and propose the framework of active retrieval augmented generation.

### 2.1 Notations and Definitions

Given a user input $x$ and a document corpus $D = \{d_i\}_{i=1}^{|D|}$ (such as all Wikipedia articles), the goal of retrieval augmented LMs is to generate the answer $y = [s_1, s_2, \ldots, s_m] = [w_1, w_2, \ldots, w_n]$ containing $m$ sentences or $n$ tokens leveraging information retrieved from the corpus.

In retrieval augmented LM, the LM typically pairs with a retriever that can retrieve a list of documents $D_q = \text{ret}(q)$ for a query $q$; the LM conditions on both the user input $x$ and retrieved documents $D_q$ to generate the answer. Since we focus on examining various methods of determining when and what to retrieve, we follow existing methods (Ram et al., 2023; Trivedi et al., 2022) to preempt the retrieved documents before the user input to aid future generation for both baselines and our method for fair comparisons: 

$$
y = \text{LM}(D_q, x).
$$

### 2.2 Single-time Retrieval Augmented Generation

The most common choice is to directly use the user input as the query for retrieval and generate the complete answer at once $y = \text{LM}(D_q, x)$.

### 2.3 Active Retrieval Augmented Generation

To aid long-form retrieval with retrieval, we propose active retrieval augmented generation. It is a generic framework that actively decides when and what to retrieve through the generation process, resulting in the interleaving of retrieval and generation. Formally, at step $t (t \geq 1)$, the retrieval query $q_t$ is formulated based on both the user input $x$ and previously generated output $y_{<t} = [y_0, \ldots, y_{t-1}]$:

$$
q_t = \text{qry}(x, y_{<t}),
$$

where $\text{qry}(\cdot)$ is the query formulation function. At the beginning ($t = 1$), the previous generation is empty ($y_{<1} = \emptyset$), and the user input is used as the initial query ($q_1 = x$). Given retrieved documents $D_q$, LMs continually generate the answer until the next retrieval is triggered or reaches the end:

$$
y_t = \text{LM}(D_q, x, y_{<t}),
$$

where $y_t$ represents the generated tokens at the current step $t$, and the input to LMs is the concatenation of the retrieved documents $D_q$, the user input $x$, and the previous generation $y_{<t}$. We discard previously retrieved documents $U_{<t}D_q$ and only use the retrieved documents from the current step to condition the next generation to prevent reaching the input length limit of LMs.

## 3. FLARE: Forward-Looking Active REtrieval Augmented Generation

Our intuition is that (1) LMs should only retrieve information when they do not have the necessary knowledge to avoid unnecessary or inappropriate retrieval, and (2) the retrieval queries should reflect the intents of future generations. We propose two forward-looking active retrieval augmented generation (FLARE) methods to implement the active retrieval augmented generation framework. The first method prompts the LM to generate retrieval queries when necessary while generating the answer using retrieval-encouraging instructions, denoted as FLARE$_\text{instr}$. The second method directly uses the LM's generation as search queries, denoted as FLARE$_\text{direct}$, which iteratively generates the next sentence to gain insight into the future topic, and if uncertain tokens are present, retrieves relevant documents to regenerate the next sentence.

### 3.1 FLARE with Retrieval Instructions

Inspired by Toolformer (Schick et al., 2023), a straightforward way of expressing information needs for retrieval is to generate “Search(query)?” when additional information is needed (Schick et al., 2023), e.g., “The colors on the flag of Ghana have the following meanings. Red is for [Search(Ghana flag red meaning)] the blood of martyrs, ...” When working with GPT-3.5 models that
```


### --- Page 0004 ---

```markdown
![An illustration of forward-looking active retrieval augmented generation with retrieval instructions (FLARE$^{\text{struct}}$). It iteratively generates search queries to retrieve relevant information to aid future generations.](assets/page_0004_img_1.png)

## 3.2.1 Confidence-based Active Retrieval
As shown in Figure 1, at step $t$, we first generate a temporary next sentence $\hat{y}_t = \text{LM}([x, y_{t-1}])$ without conditioning on retrieved documents. Then we decide whether to trigger retrieval and formulate search queries $q_t$ to retrieve relevant documents, and then regenerate the next sentence $y_t$. The reason we utilize sentences as the basis of our iteration is due to their significance as semantic units that are neither too short nor too lengthy like phrases and paragraphs. However, our approach can also utilize phrases or paragraphs as the basis.

Since LMs tend to be well-calibrated that low probability/confidence often indicates a lack of knowledge (Jiang et al., 2021; Kadavath et al., 2022; Varshney et al., 2022), we actively trigger retrieval if any token of $\hat{y}_t$ has a probability lower than a threshold $\theta \in [0, 1]$. $\theta = 0$ means retrieval is never triggered, while $\theta = 1$ triggers retrieval every sentence.

$$
y_t = 
\begin{cases} 
\hat{y}_t & \text{if all tokens of } \hat{y}_t \text{ have probs } \geq 0 \\ 
\hat{y}_t = \text{LM}([D_{q_t}, x, y_{<t}]) & \text{otherwise} 
\end{cases}
$$

where the query $q_t$ is formulated based on $\hat{y}_t$.

## 3.2.2 Confidence-based Query Formulation
One way to perform retrieval is to directly use the next sentence $\hat{y}_t$ as the query $q_t$. This shares a similar spirit with methods that use generated hypothetical titles or paragraphs from LMs as retrieval queries or evidences (Gao et al., 2022; Sun et al., 2022; Yu et al., 2022; Mao et al., 2021). We generalize such techniques to long-form generation where active information access is essential.

We found retrieving with the next sentence achieves significantly better results than with the previous context, as shown later in subsection 6.2. However, it has a risk of perpetuating errors contained in it. For example, if the LM produces the sentence “Joe Biden attended the University of Pennsylvania” instead of the correct fact that he attended the University of Delaware, using this erroneous sentence as a query might retrieve mislead...
```

### --- Page 0005 ---

```markdown
queries $q_t$ are formulated based on $\hat{s}$ as follows:

$$
q_t = \begin{cases} 
\emptyset & \text{if all tokens of } \hat{s} \text{ have probs } \leq \theta \\ 
\text{mask}(\hat{s}) & \text{otherwise} 
\end{cases}
$$

### 3.3 Implementation Details

**Base LM** We validate our method on one of the most advanced GPT-3.5 LMs text-davinci-003 by iteratively querying their API. 

**Document corpus and retrievers.** Since we focus on the integration of retrieval and generation, we use off-the-shelf retrievers that take queries as inputs and return a list of relevant documents. For datasets that mainly rely on knowledge from Wikipedia, we use the Wikipedia dump from Karpukhin et al. (2020) and employ BM25 (Robertson and Zaragoza, 2009) as the retriever. For datasets that rely on knowledge from the open web, we use the Bing search engine as our retriever. 

**Retrieved document formatting.** Multiple retrieved documents are linearized according to their ranking and then added to the beginning of the user input using Prompt D.1.

Other implementation details such as sentence tokenization and efficiency are included in Appendix A.

### 4 Multi-time Retrieval Baselines

Existing passive multi-time retrieval augmented LMs can also be formulated using our framework (subsection 2.3). In this section, we formally introduce three baseline categories based on when and what to retrieve. These baselines are not exact reproductions of the corresponding paper because many design choices differ which makes direct comparisons impossible. We implemented them using the same settings, with the only variation being when and what to retrieve.

**Previous-window** approaches trigger retrieval every $l$ tokens, where $l$ represents the window size. Generated tokens from the previous window are used as the query:

$$
q_t = y_{t-l} \quad (t \geq 2), \quad y_t = [w_{(t-1)+1}, \ldots, w_t].
$$

Some existing methods in this category are RETRO (Borgeaud et al., 2022), IC-RALM (Ram et al., ...
```

![Figure 3: Implicit and explicit query formulation. Tokens with low probabilities are marked with underlines.](assets/page_0005_img_1.png)

### --- Page 0006 ---

```markdown
![Detailed description of the chart](assets/page_0006_img_1.png)

2023), which retrieve every few tokens, and KNN-LM (Khandelwal et al., 2020), which retrieves every token. We follow Ram et al. (2023) to use a window size of $l = 16$.

Previous-sentence approaches trigger retrieval every sentence and use the previous sentence as the query, and IRCoT (Trivedi et al., 2022) belongs to this category:

$$
q_t = y_{t-1} \quad (t \geq 2),
$$
$$
y_t = g_t.
$$

Question decomposition approaches manually annotated task-specific exemplars to guide LMs to generate decomposed sub-questions while producing outputs. For example, self-ask (Press et al., 2022), a method in this category, manually inserts sub-questions in exemplars using Prompt D.2. For the test case, retrieval is triggered dynamically whenever the model generates a sub-question.

The aforementioned approaches can retrieve additional information while generating. However, they have notable drawbacks: (1) Using previously generated tokens as queries might not reflect what LMs intend to generate in the future. (2) Retrieving information at a fixed interval can be inefficient because it might occur at inappropriate points. (3) Question decomposition approaches require task-specific prompt engineering, which restricts their generalizability in new tasks.

5 Experimental Setup  
We evaluate the effectiveness of FLARE on 4 diverse knowledge-intensive tasks using few-shot in-context learning (Radford et al., 2019; Brown et al., 2020; Liu et al., 2023). We follow previous works (Trivedi et al., 2022) to sub-sample at most 500 exemplars from each dataset due to the cost of running experiments. Datasets, metrics, and settings are summarized in Table 7 of Appendix B. The hyperparameters of FLARE are selected based on the development set and listed in Table 9. FLARE refers to FLARE$_{direct}$ if not specifically stated.

Multihop QA  
The goal of multihop QA is to answer complex questions through information retrieval and reasoning. We use 2WikiMultihopQA (Ho et al., 2020) which contains 2-hop complex questions sourced from Wikipedia articles that require composition, comparison, or inference, e.g., “Why did the founder of Versus die?” We follow Wang et al. (2022) to generate both the chain-of-thought and the final answer. Experimental setting details are included in Appendix B.

We use regular expressions to extract the final answer from the output and compare it with the reference answer using exact match (EM), and token-level F1, precision, and recall.

Commonsense reasoning  
Commonsense reasoning requires world and commonsense knowledge to generate answers. We use StrategyQA (Geva et al., 2021) which is a collection of crowdsourced yes/no questions, e.g., “Would a pear sink in water?” We follow Wei et al. (2022) to generate both the chain-of-thought and the final yes/no answer. Details are included in Appendix B.

Long-form QA  
Long-form QA aims to generate comprehensive answers to questions seeking complex information (Fan et al., 2019; Stelmach et al., 2022). We use ASQA (Stelmach et al., 2022) as our testbed where inputs are ambiguous questions with multiple interpretations, and outputs should cover all of them. For example, “Where do the Philadelphia Eagles play their home games?” could be asking about the city, sports complex, or stadium. We found in many cases it is challenging even for humans to identify which aspect of the question is ambiguous. Therefore, we created another setting (ASQA-hint) where we provide a brief hint to guide LMs to stay on track when generating answers. The hint for the above case is “This question is ambiguous in terms of which specific location or venue is being referred to.” Experimental setting details are included in Appendix B.

We use metrics from Stelmach et al. (2022), including EM, RoBERTa-based QA score (Disambiguating F1), ROUGE (Lin, 2004), and an overall score combining Disambig-F1 and ROUGE (DR).

Open-domain summarization  
The goal of open-domain summarization is to generate a comprehensive summary about a topic by gathering information from open web (Giorgi et al., 2022). We use WikiAsp (Hayashi et al., 2021) which aims to generate aspect-based summaries about entities from 20 domains in Wikipedia, e.g., “Generate a summary about Echo School (Oregon) including the
```

### --- Page 0007 ---

```markdown
![Comparison between FLARE and baselines across all tasks/datasets. We report the primary metric for each dataset: EM for 2WikiMultihopQA, StrategyQA, and ASQA, and UniEval for WikiAsp.](assets/page_0007_img_1.png)

| Methods                     | EM   | F1   | Prec. | Rec.  |
|----------------------------|------|------|-------|-------|
| No retrieval               | 28.2 | 36.8 | 36.5  | 38.6  |
| Single-time retrieval      | 39.4 | 48.8 | 48.6  | 51.5  |
| Multi-time retrieval       |      |      |       |       |
| Previous-window            | 43.2 | 52.3 | 51.7  | 54.5  |
| Previous-sentence          | 39.0 | 49.2 | 48.9  | 51.8  |
| Question decomposition      | 47.8 | 56.4 | 56.1  | 58.6  |
| FLARE$_{inst}$ (ours)     | 42.4 | 49.8 | 49.1  | 52.5  |
| FLARE$_{edit}$ (ours)     | 51.0 | 59.7 | 59.1  | 62.6  |

6 Experimental Results

We first report overall results across 4 tasks/datasets and compare the performance of FLARE with all the baselines introduced in section 4. We then run ablation experiments to study the efficacy of various design choices of our method.

6.1 Comparison with Baselines

Overall results. The overall performance of FLARE and baseline across all tasks/datasets are reported in Figure 4. FLARE outperforms all baseline on all tasks/datasets, indicating that FLARE is a generic method that can effectively retrieve additional information throughout the generation.

Among various tasks, multihop QA shows the most significant improvement. This is largely due to the task’s clear definition and specific objective of producing the final answer through a 2-hop reasoning process, which makes it easier for LMs to generate on-topic output. In contrast, ASQA and WikiAsp are more open-ended, which increases the difficulty of both generation and evaluation. The improvement on ASQA-hint is larger than that of ASQA because identifying ambiguous aspects is challenging even for humans in many cases, and providing a generic hint helps LMs to stay on topic.

Thorough comparisons with baselines. The performance of all baselines on 2WikiMultihopQA are reported in Table 1. FLARE outperforms all baselines by a large margin, which confirms that forward-looking active retrieval is highly effective. Most multi-time retrieval augmented approaches outperform single-time retrieval but with different margins. The improvement of retrieving using the previous sentence is relatively small which we hypothesize is mainly because the previous sentence often describes entities or relations different from those in the next sentence in 2WikiMultihopQA. While the previous-window approach might use the first half of a sentence to retrieve information potentially helpful for generating the second half. Among all baselines, the question decomposition approach (Press et al., 2022) achieves the best performance, which is not surprising since the in-context exemplars manually annotated with decomposed sub-questions (Prompt D.2) guide LMs to generate sub-questions that align with the topic/intention of future generations. FLARE outperforms this baseline, indicating that manual exemplar annotation is not necessary for effective future-aware retrieval. The gap between FLARE$_{inst}$ and question decomposition is large, indicating that teaching LMs to generate search queries using task-generative retrieval instructions and exemplars is challenging.

We report all metrics for the other datasets in Table 2. FLARE outperforms baselines with respect to all metrics. Retrieval using the previ...
```

### --- Page 0008 ---

```markdown
| Datasets         | StrategyQA | ASQA         | ASQA-hint    | WikiAsp      |
|------------------|------------|--------------|--------------|--------------|
| Metrics          | EM         | EM D-F1 R-L DR | EM D-F1 R-L DR | UniEval E-F1 R-L |
| No retrieval      | 72.9      | 33.8 24.2 33.3 28.4 | 40.1 32.5 36.4 34.4 | 47.1 14.1 26.4 |
| Single-time retrieval | 68.6 | 40.0 27.4 34.0 30.4 | 43.2 34.8 37.4 36.0 | 52.4 17.4 26.9 |

### Multi-time retrieval

|                  | Previous-window | Previous-sentence | FLARE (ours) |
|------------------|-----------------|--------------------|---------------|
|                  | 71.2 39.9 27.0 34.3 30.4 | 43.7 35.7 37.5 36.6 | 77.3 41.3 28.2 34.3 31.1 |

Table 2: Comparison between FLARE and baselines on StrategyQA, ASQA, ASQA-hint, and WikiAsp. D-F1 is Disambig-F1, R-L is ROUGE-L, and E-F1 is named entity-based F1.

|                  | 2WikiMultihopQA | ASQA-hint       |
|------------------|------------------|------------------|
|                  | EM  F1  Prec. Rec. | EM D-F1 R-L DR   |
| Previous         | 39.0 49.2 48.9 51.8 | 42.5 34.1 36.9 35.5 |
| Next             | 48.8 57.6 57.1 60.5 | 45.9 35.7 37.5 36.6 |

Table 3: A head-to-head comparison between using the previous sentence and the next sentence for retrieval.

| #Tokens | EM   | F1   | Prec. | Rec. |
|---------|------|------|-------|------|
| 16      | 43.2 | 52.3 | 51.7  | 54.5 |
| 32      | 43.6 | 52.4 | 52.0  | 55.0 |
| 48      | 40.0 | 49.3 | 49.0  | 52.0 |
| All     | 39.0 | 48.5 | 48.2  | 51.1 |

Table 4: Previous-window approaches using different numbers of tokens as queries.

![Performance (EM) of FLARE with respect to the percentage of steps/sentences with retrieval on 2WikiMultihopQA and StrategyQA.](assets/page_0008_img_1.png)

In the past hurts the performance, further confirming our hypothesis that previous context might not be relevant to intent of future generations.

### Importance of active retrieval
Next, we investigate how active retrieval threshold θ affects performance. To alter our method from not retrieving to retrieving every sentence, we adjust the confidence threshold θ that determines when to trigger retrieval from 0 to 1. We then calculate the proportion of steps/sentences where retrieval is activated, and present the performance based on it. As shown in Figure 5, on 2WikiMultihopQA, the performance plateaus when the retrieval percentage exceeds 60%, indicating that retrieval when LMs are confident is not necessary. On StrategyQA, the performance drops when the retrieval percentage exceeds 50%, indicating that unnecessary retrieval can introduce noise and impede the original generation process. We found triggering retrieval for 40%-80% of sentences usually leads to a good performance across tasks/datasets.

### Effectiveness of different query formulation methods
We study implicit query formation by masking and explicit query formulation through question generation. In Table 5, we compare the performance of FLARE with different masking methods.
```

### --- Page 0009 ---

```markdown
| $\beta$ | EM    | F1    | Prec. | Rec.  |
|---------|-------|-------|-------|-------|
| 0.0     | 0.488 | 0.576 | 0.571 | 0.605 |
| 0.2     | 0.498 | 0.588 | 0.582 | 0.616 |
| 0.4     | 0.510 | 0.597 | 0.591 | 0.627 |
| 0.6     | 0.506 | 0.593 | 0.586 | 0.622 |

Table 5: Performance of FLARE with respect to the masking threshold $\beta$ on 2WikiMultiHopQA.

| ASQA-hint         | EM D.F. | R.L. DR | WikiAsp         | UniEval E.F. | R.L. |
|-------------------|---------|---------|------------------|--------------|------|
| Implicit 46.7     | 36.9    | 37.7    | 53.4             | 18.2         | 27.7 |
| Explicit 46.2     | 36.7    | 37.7    | 53.4             | 18.9         | 27.6 |

Table 6: A comparison between implicit and explicit query formulation methods in FLARE.

8 Conclusion  
To aid long-form generation with retrieval augmentation, we propose an active retrieval augmented generation framework that decides when and what to retrieve during generation. We implement this framework with forward-looking active retrieval that iteratively uses the upcoming sentence to retrieve relevant information if it contains low-confidence tokens and regenerates the next sentence. Experimental results on 4 tasks/datasets demonstrate the effectiveness of our methods. Future directions include better strategies for active retrieval and developing efficient LM architectures for active information integration.

9 Limitations  
We also conduct experiments on Wizard of Wikipedia (Dinan et al., 2019) and ELI5 (Fan et al., 2019), and found that FLARE did not provide significant gains. Wizard of Wikipedia is a knowledge-intensive dialogue generation dataset where the output is relatively short (~20 tokens on average) so retrieving multiple disparate pieces of information might not be necessary. ELI5 (Fan et al., 2019) is a long-form QA dataset requiring in-depth answers to open-ended questions. Due to issues mentioned in Krishna et al. (2021) such as difficulties of grounding generation in retrieval and evaluation, both single-time retrieval and FLARE did not provide significant gains over not using retrieval. From an engineering perspective, interleaving generation and retrieval with a naive implementation increases both overheads and the cost of generation. LMs need to be activated multiple times (once for each retrieval) and a caching-free implementation also requires recomputing the previous activation each time after retrieval. This issue can be potentially alleviated with special architectural designs that encode the retrieved documents $D_q$ and the input/generation $(x/y_{<t})$ independently.

Acknowledgements  
This work was supported in part by a grant from the Singapore Defence Science and Technology Agency and the IBM PhD Fellowship. We thank Chunting Zhou, Amanda Bertsch, Uri Alon, Hiroaki Hayashi, Harsh Trivedi, Patrick Lewis, Timo Schick, Kaixin Ma, Shuyan Zhou, and Songwei Ge for their insightful discussions and help with the experiments.
```

### --- Page 0010 ---

```markdown
# References

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Milligan, George van den Driesche, Jean-Baptiste Lespiau, Bogdan Damoc, Aïda Mencik, Diego de Las Casas, Aurelia Guy, Jacob Mincik, Roman Ring, Tom Henighan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Improving language models by retrieving from tokens. In *International Conference on Machine Learning*, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of *Proceedings of Machine Learning Research*, pages 2206–2240. PMLR.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prahav Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Fon H. Heighian, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In *Advances in Neural Information Processing Systems* 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading wikipedia to answer open-domain questions. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics*, ACL 2017, Vancouver, Canada, July 30 - August 4, volume 1: Long Papers, pages 1870–1879. Association for Computational Linguistics.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Sharma, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Church, Kensen Shi, Sasha Tsuchiya, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Popp, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ar, Pengcheng Yin, Tujo Duke, Anselm Levslyak, Sanjay Ghemawat, Sinipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Japhy Ippolito, David Luan, Hyenautek Lim, Barrett Zoppa, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanujhalam Sankaranarayanan Pillai, Marie Petell, Aitor Lewkowycz, Erica Moreira, Drew Choi, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Yuzhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. CoRR, abs/2204.02311.

Nachshon Cohen, Oren Kalinsky, Yifat Ziser, and Alessandro Moschitti. 2021. Vikisum: Coherent summarization dataset for efficient human-evaluation. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 2: Short Papers)*, Virtual Event, August 1-6, 2021, pages 212–219. Association for Computational Linguistics.

Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2019. Wizard of wikipedia: Knowledge-powered conversational agents. In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net.

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: long form question answering. In *Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28-August 2, 2019, Volume 1: Long Papers*, pages 3558–3567. Association for Computational Linguistics.

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022. Precise zero-shot dense retrieval without relevance labels. CoRR, abs/2212.10496.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. *Transactions of the Association for Computational Linguistics*, 9:346–361.

John M. Giorgi, Luca Soldaini, Bo Wang, Gary D. Bader, Kyle Lo, Lucy Liu Wang, and Aaron Cohen. 2022. Exploring the challenges of open domain multi-document summarization. CoRR, abs/2212.10526.

Kelvin Guu, Kenton Lee, Zara Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: retrieval-augmented language model pre-training. CoRR, abs/2002.08909.

Hiroaki Hayashi, Prashant Budania, Peng Wang, Chris Ackersson, Raj Neervannan, and Graham Neubig. 2021. Wikisum: A dataset for multi-domain aspect-based summarization. *Trans. Assoc. Comput. Linguistics*, 9:211–225.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. CoRR, abs/2009.00310.

Xanh Ho, Anh-Khoa Duong Nguyen, Sak Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In *Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020*.
```

### --- Page 0011 ---

```markdown
| **Authors**                                                                 | **Title**                                                                                                   | **Source**                                                                                                   |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Gautier Izacard and Edouard Grave. 2021.                                   | Leveraging passage retrieval with generative models for open domain question answering.                     | In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021, pages 874–880. Association for Computational Linguistics. |
| Gautier Izacard, Patrick S. H. Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Y, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022. | Few-shot learning with retrieval augmented language models.                                                  | CoRR, abs/2208.03299.                                                                                       |
| Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. 2021.           | How can we know when language models know? on the calibration of language models for question answering.    | Trans. Assoc. Comput. Linguistics, 9:962–977.                                                                |
| Zhengbao Jiang, Luyu Gao, Jun Araki, Haibo Ding, Zhiruo Wang, Jamie Callan, and Graham Neubig. 2022. | Retrieval as attention: End-to-end learning of retrieval and reading within a single transformer.          | CoRR, abs/2212.02027.                                                                                       |
| Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. 2020.          | How can we know what language models know.                                                                  | Trans. Assoc. Comput. Linguistics, 8:423–438.                                                                |
| Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017.    | Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension.                   | In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, pages 1601–1611. Association for Computational Linguistics. |
| Saurav Kadavath, Tom Conery, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Dany Hernandez, Josh Jacobson, Jack Kermion, Shauna Kravce, Liane Lovitt, Kalemal Ndouse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. 2022. | Language models (mostly) know what they know. | CoRR, abs/2207.05221.                                                                                       |
| Vladimir Karpukhin, Barlas Ozyurt, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. | Dense passage retrieval for open-domain question answering.                                                  | In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 6769–6781. Association for Computational Linguistics. |
| Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020. | Generalization through memorization: Nearest neighbor language models.                                      | In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net. |
| Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. 2022. | Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive NLP.            | CoRR, abs/2212.10421.                                                                                       |
| Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yu Fa, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2022. | Decomposed prompting: A modular approach for solving complex tasks.                                        | CoRR, abs/2210.02406.                                                                                       |
| Kalpesh Krishna, Aurok Roy, and Mohit Iyer. 2021.                          | Hurdles to progress in long-form question answering.                                                        | In North American Association for Computational Linguistics.                                                  |
| Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Ken-ton Lee, Kristina Tounatova, Lilon Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. | Natural questions: a benchmark for question answering research.                                             | Trans. Assoc. Comput. Linguistics, 7:452–466.                                                                |
| Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. 2022. | Internet-augmented language models through few-shot prompting for open-domain question answering.           | CoRR, abs/2203.05115.                                                                                       |
| Haejun Lee, Akhil Kedia, Jongwon Lee, Ashwin Paranjape, Christopher D. Manning, and Kyung-Gu Woo. 2021. | You only need one model for open-domain question answering.                                                  | CoRR, abs/2212.07381.                                                                                       |
| Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. | Retrieval-augmented generation for knowledge-intensive NLP tasks.                                          | In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual. |
| Junyi Li, Tianyi Tang, Wayne Xin Zhao, Jingyuan Wang, Jian-Yun Nie, and Ji-Rong Wen. 2023. | The web can be your oyster for improving large language models.                                             | CoRR, abs/2305.10998.                                                                                       |
| Chin-Yew Lin. 2004.                                                         | A package for automatic evaluation of summaries.                                                            | In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics. |
```

### --- Page 0012 ---

```markdown
| **Authors**                                                                 | **Title**                                                                                                   |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. | Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv., 55(9):1–195:35. |
| Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hanneah Hajishirzi, and Daniel Khashabi. 2022. | When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories. CoRR, abs/2212.10511. |
| Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen. 2021. | Generation-augmented retrieval for open-domain question answering. In Proceedings of the 9th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume I: Long Papers), Virtual Event, August 1-6, 2021, pages 4089–4100. Association for Computational Linguistics. |
| Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. | On faithfulness and factuality in abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1906–1919. On-line. Association for Computational Linguistics. |
| Reichi Hiro Nakano, Jacob Hilton, Shashi Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vincent Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyra Eloundou, Gretchen Krueger, Kevin Button, Mathew Knight, Benjamin Schlesinger, and John Schulman. 2021. | WebGPT: Browser-assisted question-answering with human feedback. CoRR, abs/2112.09332. |
| OpenAI. 2023. | GPT-4 technical report. CoRR, abs/2303.08774. |
| Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll K. Wainwright, Pamela Mishkin, Chong Zhang, Sandhni Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simons, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. | Training language models to follow instructions with human feedback. CoRR, abs/2303.02155. |
| Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhuo Yu, Weizhu Chen, and Jianfeng Gao. 2023. | Check your facts and try again: Improving large language models with external knowledge and automated feedback. CoRR, abs/2302.12813. |
| Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick S. H. Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander H. Miller. 2019. | Language models as knowledge bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 2463–2473. Association for Computational Linguistics. |
| Orf Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. 2022. | Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350. |
| Hongjing Qian, Yutao Zhu, Zhicheng Dou, Haoqi Qu, Xinyu Zhang, Zheng Liu, Ruofei Lai, Zhao Cao, Jian-Yun Nie, and Ji-Rong Wen. 2023. | Webbrain: Learning to generate factually correct articles for queries by grounding on large web corpus. CoRR, abs/2304.04358. |
| Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao Liang, Kunlun Zhu, Yankai Liu, Xu Han, Ning Ding, Huadong Wang, Ruobing Xie, Fanchao Qi, Zhiyuan Liu, Maosong Sun, and Jie Zhou. 2023. | Webpm: Interactive web search for chinese long-form question answering. CoRR, abs/2305.06849. |
| Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. | Language models are unsupervised multitask learners. OpenAI Report, 1(8). |
| Ori Ram, Yoav Levine, Itay Dalmedig, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. | In-context retrieval-augmented language models. arXiv preprint arXiv:2302.00083. |
| Adam Roberts, Colin Raffel, and Moataz Shazeer. 2020. | How much knowledge can you pack into the parameters of a language model? In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 5418–5426. Association for Computational Linguistics. |
| Stephen E. Robertson and Hugo Zaragoza. 2009. | The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr., 3(4):333–389. |
| Devendra Singh Sachan, Siva Reddy, William L. Hamilton, Chris Dryer, and Dandapani Votagamma. 2021. | End-to-end training of multi-document reader and retriever for open-domain question answering. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 25968–25981. |
| Timo Schick, Jane Dwiwedi-Yu, Roberto Deisi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. | Toolformer: Language models can teach themselves to use tools. |
| Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. | REPLUG: retrieval-augmented black-box language models. CoRR, abs/2301.12652. |
```

### --- Page 0013 ---

```markdown
| Authors                                                                 | Title                                                                                                   |
|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Ivan Stelmach, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang.           | ASQA: factoid questions meet long-form answers. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 8273–8288. Association for Computational Linguistics. |
| Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou.       | Recitation-augmented language models. CoRR, abs/2210.2196.                                            |
| Hugo Touvron, Thibault Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lacroix, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. | Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.                           |
| Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. | Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. CoRR, abs/2212.10590. |
| Neeraj Varshney, Man Luo, and Chitta Baral.                           | 2022. Can open-domain QA reader utilize external knowledge efficiently like humans? CoRR, abs/2211.12707. |
| Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, and Denny Zhou. | Self-consistency improves chain of thought reasoning in language models. CoRR, abs/2203.11171.         |
| Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. | Chain of thought prompting elicits reasoning in large language models. CoRR, abs/2201.11903.           |
| Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Ishak Shafran, Karthik Narasimhan, and Yuan Cao. | React: Synergizing reasoning and acting in language models. CoRR, abs/2210.03629.                     |
| Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Xu, Soumya Sanyal, Chenguang Zhu, Michael Zhang, and Meng Jiang. | Generate rather than retrieve: large language models are strong context generators. CoRR, abs/2209.10603. |
| Wenhao Yu, Zhihan Zhang, Zhenwen Liang, Meng Jiang, and Ashish Sabharwal. | Improving language models via plug-and-play retrieval feedback. CoRR, abs/2305.14002.                  |
| Fengji Zhang, Bei Chen, Yue Zhang, Jin Liu, Daqiang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. | Reprocessor: Repository-level code completion through iterative retrieval and generation. CoRR, abs/2303.12570. |
| Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Devries, Mona Diab, Xian Li, Xi Victoria Lin, Todd M. Haywood, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Shridhar, Tianlu Wang, and Luke Zettlemoyer. | Opt: Open pre-trained transformer language models. ArXiv, abs/2205.01068.                             |
| Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqing Jin, Beichen Zhang, Junjie Zhang, Zicon Dong, Yifan Du, Chen Yang, Yushou Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. | A survey of large language models. CoRR, abs/2303.18223.                                             |
| Ming Zhong, Yang Liu, Da Yin, Yuning Mo, Yizhu Jiao, Fengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei Han. | Towards a unified multidimensional evaluator for text generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 2023–2038. Association for Computational Linguistics. |
| Chunting Zhou, Graham Neubig, Jiatao Gu, Mona Diab, Francisco Guzmán, Luke Zettlemoyer, and Marjan Ghazvininejad. | Detecting hallucinated content in conditional neural sequence generation. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 1393–1404, Online. Association for Computational Linguistics. |
```

### --- Page 0014 ---

```markdown
# A FLARE Implementation Details

**FLAREstruct implementation details**  
We found that LMs can effectively combine retrieval and downstream task-related skills and generate meaningful search queries while performing the task. However, there are two issues: (1) LMs tend to generate fewer search queries than necessary. (2) Generating excessive search queries can disrupt answer generation and adversely affect performance. We address these issues using two methods respectively. First, we increase the logit of the token “I” by 2.0 to improve the chances of LMs generating “I[Search(query)]”. Second, whenever LMs generate a search query, we use it to retrieve relevant information, promptly remove it from the generation, and generate the next few tokens while forbidding “I” by adding a large negative value to the logit of “I”.

**The initial query of FLARE.**  
FLARE starts with the user input $\mathbf{x}$ as the initial query to retrieve documents to generate the first sentence $\hat{s}_1 = \text{LM}(\mathcal{D}_{\text{x}})$ to bootstrap the iterative generation process. For the following steps, the temporary forward-looking sentence is generated without retrieved documents.

**Sentence tokenization.**  
For each step $t$, we generate 64 tokens which are longer than most sentences, and use NLTK sentence tokenizer\footnote{https://www.nltk.org/api/nltk.tokenize.PunktSentenceTokenizer.html} to extract the first sentence and discard the rest.

**Efficiency**  
As shown in subsection 6.2, on average retrieval is triggered for 30% ∼ 60% of sentences depending on downstream tasks. In comparison, KNN-LM (Khandhlev et al., 2020) retrieves every token, RETRO or IC-RALM (Borgeaud et al., 2022; Ram et al., 2023) retrieves every 4∼32 tokens, and IRCoT (Trivedi et al., 2022) retrieves every sentence. Compared to single-time retrieval, however, interleaving retrieval and generation with a naive implementation indeed increases overheads, which we discuss in the limitation section (section 9).

# B Datasets and Settings

Datasets, metrics, and experimental settings are summarized in Table 7.

\footnote{https://www.nltk.org/api/nltk.tokenize.PunktSentenceTokenizer.html}

**Multihop QA**  
For “Why did the founder of Versus die?”, the output we aim to generate is “The founder of Versus was Gianni Versace. Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997. So the answer is shot.” We use 8 exemplars from Trivedi et al. (2022) listed in Prompt D.4 for in-context learning, BM25 as the retriever, and Wikipedia articles as the retrieval corpus. Similar to the observation in Trivedi et al. (2022), we found incorporating retrieval results for exemplars improves the performance, we use the input $\mathbf{x}$ of each exemplar to retrieve several documents and then add them using the format in Prompt D.1. We found increasing the number of retrieval documents often increases performance. Therefore, we use the maximum number of documents that can fit within the input length limit of $text-davinci-003$, which is 2 for 2Wiki-MultihopQA.

**Commonsense Reasoning**  
For “Would a pear sink in water?”, the output we aim to generate is “The density of a pear is about 0.6g/cm^3$, which is less than water. Objects less dense than water float. Thus, a pear would float. So the final answer is no.” We use 6 exemplars from Wei et al. (2022) listed in Prompt D.5, BM25 on the Wikipedia corpus, and 3 retrieved documents to run experiments.

**Long-form QA**  
For “Where do the Philadelphia Eagles play their home games?”, the output we aim to generate is “We need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium. Therefore, this question has 3 interpretations and the answers are: (1) The city is Philadelphia. (2) The sports complex is the South Philadelphia Sports Complex. (3) The stadium is the Lincoln Financial Field stadium.” For both the original setting (ASQA) and the setting with hints (ASQA-hint), we manually annotate 8 exemplars (Prompt D.6 and D.8), use BM25 on the Wikipedia corpus, and 3 retrieved documents to run experiments.

**Open-domain Summarization**  
The original WikiAsp dataset is designed for multi-document summarization and provides a list of references to systems. We converted it into the open-domain setting by removing the associated references and instead gathering information from the open web. For “Generate a summary about Echo School (Ore...
```


### --- Page 0015 ---

```markdown
## C Hyperparameters

Hyperparameters of FLARE on different datasets are listed in Table 9.

## D Prompts and Few-shot exemplars

The prompt used to linearize multiple documents is shown in Prompt D.1. The prompt used in self-ask (Press et al., 2022) is shown in Prompt D.2. Prompts and exemplars of different tasks/datasets are shown in Prompt D.3, D.4, D.5, D.6, D.8, and D.10, respectively.

### Prompt D.1: document formatting

| Search results: |
|------------------|
| [1] Document 1   |
| [2] Document 2   |
| ...              |
| The user input x |

### Prompt D.2: multihop QA with self-ask

**Question:** Who lived longer, Theodor Haecker or Harry Vaughan Watkins?  
Are follow up questions needed here: Yes.  
**Follow up:** How old was Theodor Haecker when he died?  
**Intermediate answer:** Theodor Haecker was 65 years old when he died.  
**Follow up:** How old was Harry Vaughan Watkins when he died?  
**Intermediate answer:** Harry Vaughan Watkins was 69 years old when he died.  
So the final answer is: Harry Vaughan Watkins.
```


### --- Page 0016 ---

```markdown
| Settings         | 2WikiMultihopQA (Ho et al., 2020) | StrategyQA (Geva et al., 2021) | ASQA (Stelmakh et al., 2022) | WikiAsp (Hayashi et al., 2021) |
|------------------|-----------------------------------|--------------------------------|------------------------------|--------------------------------|
| Task             | multihop QA                       | commonsense QA                 | long-form QA                 | open-domain summarization      |
| #Examples        | 500                               | 229                            | 500                          | 500                            |
| Metrics          | EM, F1, Prec., Rec.              | EM                             | EM, Disambig-F1, ROUGE, DR UniEval, entity-F1, ROUGE |
| Corpus           | Wikipedia                         | Wikipedia                      | Wikipedia                    | open web                      |
| Retriever        | BM25                              | BM25                           | BM25                         | Bing                           |
| Top-k            | 2                                 | 3                              | 3                            | 5                              |
| #Exemplars       | 8                                 | 6                              | 8                            | 4                              |
| Ret. for exemplars | ✓                               | ✗                              | ✓                            | ✗                              |

Table 7: Dataset statistics and experimental settings of different tasks.

---

wikipedia.org, wikiwand.com, wiki2.org, wikimedia.org

Table 8: Wikipedia-related domains excluded from Bing’s search results.

---

| Dataset                | θ   | β   | Query formulation | Combine single- & multi-time retrieval |
|-----------------------|-----|-----|-------------------|---------------------------------------|
| 2WikiMultihopQA      | 0.8 | 0.4 | implicit          | ✗                                     |
| StrategyQA            | 0.4 | 0.4 | implicit          | ✗                                     |
| ASQA & ASQA-hint     | 0.8 | 0.4 | explicit          | ✓                                     |
| WikiAsp               | 0.8 | 0.4 | explicit          | ✓                                     |

Table 9: Hyperparameters of FLARE on different datasets.
```

### --- Page 0017 ---

```markdown
# Prompt D.3: retrieval instructions for 2WikiMultihopQA

## Skill 1. 
Use the Search API to look up relevant information by writing “[Search(term)]” where “term” is the search term you want to look up. For example:

**Question:** But what are the risks during production of nanomaterials?  
**Answer (with Search):** [Search(nanomaterial production risks)] Some nanomaterials may give rise to various kinds of lung damage.

**Question:** The colors on the flag of Ghana have the following meanings.  
**Answer (with Search):** Red is for [Search(Ghana flag red meaning)] the blood of martyrs, green for forests, and gold for mineral wealth.

**Question:** Metformin is the first-line drug for what?  
**Answer (with Search):** [Search(Metformin first-line drug)] patients with type 2 diabetes and obesity.

## Skill 2. 
Answer questions by thinking step-by-step. First, write out the reasoning steps, then draw the conclusion. For example:

**Question:** When did the director of film Hypocrite (Film) die?  
**Answer (with step-by-step):** The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013.

**Question:** Are both Kurram Garhi and Trojkrist located in the same country?  
**Answer (with step-by-step):** Kurram Garhi is located in the country of Pakistan. Trojkrist is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no.

**Question:** Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?  
**Answer (with step-by-step):** Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan’s nationality is India. Karl Freund’s nationality is Germany. Thus, they do not have the same nationality. So the answer is no.

**Question:** Who is Borachin (Wife Of Ögedei)’s father-in-law?  
**Answer (with step-by-step):** Borachin is married to Ögedei Khan. Ögedei Khan’s father is Genghis Khan. Thus, Borachin’s father-in-law is Genghis Khan. So the answer is Genghis Khan.

**Question:** Who was born first out of Martin Hodge and Ivania Martinich?  
**Answer (with step-by-step):** Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1955. Thus, Martin Hodge was born first. So the answer is Martin Hodge.

**Question:** When did the director of film Laughter In Hell die?  
**Answer (with step-by-step):** The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is August 25, 1963.

**Question:** Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?  
**Answer (with step-by-step):** The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two. So the answer is Twenty Plus Two.

**Question:** Who is the grandchild of Krishna Shah (Nepalese Royalty)?  
**Answer (with step-by-step):** Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvijit Shah. Thus, Krishna Shah has a grandchild named Prithvijit Shah. So the answer is Prithvijit Shah.

Now, combine the aforementioned two skills. First, write out the reasoning steps, then draw the conclusion, where the reasoning steps should also utilize the Search API “[Search(term)]” whenever possible.

**Question:** Where did Minbyauk Thihapate’s wife die?  
**Answer (with step-by-step & Search):
```

### --- Page 0018 ---

```markdown
# Prompt D.4: exemplars of 2WikiMultihopQA

| Question                                                                 | Answer                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| When did the director of film Hypocrite (Film) die?                     | The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013.                                                                                                   |
| Are both Kurram Garhi and Trojkrsti located in the same country?        | Answer: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no.                                   |
| Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality? | Answer: Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan’s nationality is India. Karl Freund’s nationality is Germany. Thus, they do not have the same nationality. So the answer is no. |
| Who is Boraqchin (Wife Of Ögedei)’s father-in-law?                      | Answer: Boraqchin is married to Ögedei Khan. Ögedei Khan’s father is Genghis Khan. So, Boraqchin’s father-in-law is Genghis Khan.                                                                                       |
| Who was born first out of Martin Hodge and Ivania Martinich?           | Answer: Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is Martin Hodge.                                                           |
| When did the director of film Laughter In Hell die?                     | Answer: The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is August 25, 1963.                                                                             |
| Which film has the director died later, The Gal Who Took the West or Twenty Plus Two? | Answer: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two. So the answer is Twenty Plus Two. |
| Who is the grandchild of Krishna Shah (Nepalese Royal)?                 | Answer: Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah. So the answer is Prithvipati Shah.                         |
| Which country the director of film Citizen Mavzik is from?              | Answer:                                                                                                                                                                                                                     |
```

### --- Page 0019 ---

```markdown
# Prompt D.5: exemplars of StrategyQA

Generate a yes or no answer to the following question.  
**Question:** Do hamsters provide food for any animals?  
**Answer:** Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the final answer is yes.

Generate a yes or no answer to the following question.  
**Question:** Could Brooke Shields succeed at University of Pennsylvania?  
**Answer:** Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the final answer is yes.

Generate a yes or no answer to the following question.  
**Question:** Hydrogen’s atomic number squared exceeds number of Spice Girls?  
**Answer:** Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the final answer is no.

Generate a yes or no answer to the following question.  
**Question:** Is it common to see frost during some college commencements?  
**Answer:** College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the final answer is yes.

Generate a yes or no answer to the following question.  
**Question:** Could a llama birth twice during War in Vietnam (1945-46)?  
**Answer:** The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the final answer is no.

Generate a yes or no answer to the following question.  
**Question:** Would a pear sink in water?  
**Answer:** The density of a pear is about 0.6g/cm³, which is less than water. Objects less dense than water float. Thus, a pear would float. So the final answer is no.

Generate a yes or no answer to the following question.  
**Question:** Did Spiderman fight against Falcon in the MCU?  
**Answer:**
```

### --- Page 0020 ---

```markdown
# Prompt D.6: exemplars of ASQA

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** Who played Bonnie in Gone with the Wind?  
**Answer:** This question is ambiguous in terms of which version or adaptation of Gone with the Wind is being referred to. In order to figure out its interpretations, we need to consider different versions or adaptations of Gone with the Wind. Gone with the Wind has two versions or adaptations: the 1939 film Gone with the Wind or the 2008 musical Gone with the Wind. Therefore, this question has 2 interpretations: (1) Who played Bonnie in the 1939 film Gone with the Wind? (2) Who played Bonnie in the 2008 musical Gone with the Wind? The answers to all interpretations are: (1) The 1939 film Gone with the Wind's character Bonnie was played by Eleanor Cammack "Cammie" King. (2) The 2008 musical Gone with the Wind's character Bonnie was played by Leilah de Meza.

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** What is the second largest city in the USA?  
**Answer:** This question is ambiguous in terms of the criteria being used to determine the second largest city in the USA. In order to figure out its interpretations, we need to consider different criteria to determine a city’s size. City size can be measured by two criteria: population or area. Therefore, this question has 2 interpretations: (1) What is the second largest city in the USA by population? (2) What is the second largest city in the USA by area? The answers to all interpretations are: (1) The second largest city in the USA by population is Los Angeles, California. (2) The second largest city in the USA by area is Juneau, Alaska.

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** When was Bohemian Rhapsody released as a single?  
**Answer:** This question is ambiguous in terms of which country’s release of the single is being referred to. In order to figure out its interpretations, we need to consider different countries where Bohemian Rhapsody is released. Bohemian Rhapsody was released in the United Kingdom and in the United States on different dates. Therefore, this question has 2 interpretations: (1) When was Bohemian Rhapsody released as a single in the United Kingdom? (2) When was Bohemian Rhapsody released as a single in the United States? The answers to all interpretations are: (1) Bohemian Rhapsody was released as a single in the United Kingdom on 31 October 1975. (2) Bohemian Rhapsody was released as a single in the United States on December 1975.

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** Where do the Philadelphia Eagles play their home games?  
**Answer:** This question is ambiguous in terms of which specific location or venue is being referred to. In order to figure out its interpretations, we need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium. Therefore, this question has 3 interpretations: (1) What city do the Philadelphia Eagles play their home games? (2) In what sports complex do the Philadelphia Eagles play their home games? (3) What stadium do the Philadelphia Eagles play their home games? The answers to all interpretations are: (1) Philadelphia Eagles play their home games in the city Philadelphia. (2) Philadelphia Eagles play their home games in the South Philadelphia Sports Complex. (3) Philadelphia Eagles play their home games in the Lincoln Financial Field stadium.

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** When did Xbox One come out in Australia?  
**Answer:** This question is ambiguous in terms of which specific version of the Xbox One is being referred to. In order to figure out its interpretations, we need to consider the different versions of the Xbox One that have been released. Xbox One has two versions: the Xbox One video game console or the Xbox One X high-end model. Therefore, this question has 2 interpretations: (1) When did the Xbox One release in Australia? (2) When did the Xbox One X release in Australia? The answers to all interpretations are: (1) The Xbox One video game console was released in Australia on November 22, 2013. (2) The Xbox One X video game console was released in Australia on November 7, 2017.

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** When does the movie Summer of '84 come out?  
**Answer:** This question is ambiguous in terms of which release of the movie Summer of '84 is being referred to. In order to figure out its interpretations, we need to consider different releases of the movie Summer of '84. The movie Summer of '84 is first released at the Sundance Festival before it’s released throughout the US. Therefore, this question has 2 interpretations: (1) When did the movie Summer of '84 first release at the Sundance Festival? (2) When did the movie Summer of '84 first release throughout the US? The answers to all interpretations are: (1) Summer of '84 was released at the Sundance Festival on January 22, 2018. (2) Summer of '84 was released throughout the US on August 10, 2018.
```

### --- Page 0021 ---

```markdown
# Prompt D.7: exemplars of ASQA (cont.)

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** What was Roy Orbison’s first number one hit?  
**Answer:** This question is ambiguous in terms of which specific chart or region is being referred to. In order to figure out its interpretations, we need to consider the different charts and regions where Roy Orbison’s music was popular. Roy Orbison is popular in both the US Hot 100 and Canada, and the UK and Ireland. Therefore, this question has 2 interpretations: (1) What was Roy Orbison’s first number one hit in the US Hot 100 and Canada? (2) What was Roy Orbison’s first number one hit in the UK and Ireland? The answers to all interpretations are: (1) "Running Scared" was the first number one hit for Roy Orbison in the US Hot 100 and Canada. (2) "Only the Lonely (Know the Way I Feel)" was the first number one hit for Roy Orbison in the UK and Ireland.

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** What is the criminal’s name in The Breakfast Club?  
**Answer:** This question is ambiguous in terms of which specific name is being referred to - the character’s name or the actor’s name. In order to figure out its interpretations, we need to consider both possibilities: the character’s name or the actor’s name. Therefore, this question has 2 interpretations: (1) What is the criminal’s character name in The Breakfast Club? (2) What is the name of the actor who played the criminal in The Breakfast Club? The answers to all interpretations are: (1) John Bender was the name of the criminal’s character in The Breakfast Club. (2) Judd Nelson was the actor of the criminal in The Breakfast Club.

Given an ambiguous question, figure out its interpretations and answer them one by one.

**Question:** How many state parks are there in Virginia?  
**Answer:**
```

### --- Page 0022 ---

```markdown
# Prompt D.8: exemplars of ASQA-hint

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.

| Question | Answer |
|----------|--------|
| What played Bonnie in gone with the wind? | Hint: This question is ambiguous in terms of which version or adaptation of Gone with the Wind is being referred to. Answer: In order to figure out its interpretations, we need to consider different versions or adaptations of Gone with the Wind. Therefore, this question has 2 interpretations: (1) Who played Bonnie in the 1939 film Gone with the Wind? (2) Who played Bonnie in the 2008 musical Gone with the Wind? The answers to all interpretations are: (1) The 1939 film Gone with the Wind's character Bonnie was played by Eleanor Cammack "Cammie" King. (2) The 2008 musical Gone with the Wind's character Bonnie was played by Leilah de Meza. |

| Question | Answer |
|----------|--------|
| What is the second largest city in the usa? | Hint: This question is ambiguous in terms of the criteria being used to determine the second largest city in the USA. Answer: In order to figure out its interpretations, we need to consider different criteria to determine a city's size. City size can be measured by two criteria: population or area. Therefore, this question has 2 interpretations: (1) What is the second largest city in the USA by population? (2) What is the second largest city in the USA by area? The answers to all interpretations are: (1) The second largest city in the USA by population is Los Angeles, California. (2) The second largest city in the USA by area is Juneau, Alaska. |

| Question | Answer |
|----------|--------|
| When was bohemian rhapsody released as a single? | Hint: This question is ambiguous in terms of which country’s release of the single is being referred to. Answer: In order to figure out its interpretations, we need to consider different countries where Bohemian Rhapsody is released. Bohemian Rhapsody was released in the United Kingdom and in the United States on different dates. Therefore, this question has 2 interpretations: (1) When was Bohemian Rhapsody released as a single in the United Kingdom? (2) When was Bohemian Rhapsody released as a single in the United States? The answers to all interpretations are: (1) Bohemian Rhapsody was released as a single in the United Kingdom on 31 October 1975. (2) Bohemian Rhapsody was released as a single in the United States on December 1975. |

| Question | Answer |
|----------|--------|
| Where do the philadelphia eagles play their home games? | Hint: This question is ambiguous in terms of which specific location or venue is being referred to. Answer: In order to figure out its interpretations, we need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium. Therefore, this question has 3 interpretations: (1) What city do the Philadelphia Eagles play their home games? (2) In what sports complex do the Philadelphia Eagles play their home games? (3) What stadium do the Philadelphia Eagles play their home games? The answers to all interpretations are: (1) Philadelphia Eagles play their home games in the city Philadelphia. (2) Philadelphia Eagles play their home games in the South Philadelphia Sports Complex. (3) Philadelphia Eagles play their home games in the Lincoln Financial Field stadium. |

| Question | Answer |
|----------|--------|
| When did xbox one come out in australia? | Hint: This question is ambiguous in terms of which specific version of the Xbox One is being referred to. Answer: In order to figure out its interpretations, we need to consider the different versions of the Xbox One that have been released. Xbox One has two versions: the Xbox One video game console or the Xbox One X high-end model. Therefore, this question has 2 interpretations: (1) When did the Xbox One release in Australia? (2) When did the Xbox One X release in Australia? The answers to all interpretations are: (1) The Xbox One video game console was released in Australia on November 22, 2013. (2) The Xbox One X video game console was released in Australia on November 7, 2017. |

| Question | Answer |
|----------|--------|
| When does the movie summer of 84 come out? | Hint: This question is ambiguous in terms of which release of the movie is being referred to. Answer: In order to figure out its interpretations, we need to consider different releases of the movie Summer of '84. The movie Summer of '84 is first released at the Sundance Festival before it’s released throughout the US. Therefore, this question has 2 interpretations: (1) When did the movie Summer of '84 first release at the Sundance Festival? (2) When did the movie Summer of '84 first release throughout the US? The answers to all interpretations are: (1) Summer of '84 was released at the Sundance Festival on January 22, 2018. (2) Summer of '84 was released throughout the US on August 10, 2018. |
```

### --- Page 0023 ---

```markdown
# Prompt D-9: exemplars of ASQA-hint (cont.)

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.

| Question | What was Roy Orbison’s first number one hit? |
|----------|-----------------------------------------------|
| Hint     | This question is ambiguous in terms of which specific chart or region is being referred to. |
| Answer   | In order to figure out its interpretations, we need to consider the different charts and regions where Roy Orbison’s music was popular. Roy Orbison is popular in both the US Hot 100 and Canada, and the UK and Ireland. Therefore, this question has 2 interpretations: (1) What was Roy Orbison’s first number one hit in the US Hot 100 and Canada? (2) What was Roy Orbison’s first number one hit for Roy Orbison in the UK and Ireland? The answers to all interpretations are: (1) Running Scared was the first number one hit for Roy Orbison in the US Hot 100 and Canada. (2) Only the Lonely (Know the Way I Feel) was the first number one hit for Roy Orbison in the UK and Ireland. |

| Question | What is the criminal’s name in The Breakfast Club? |
|----------|-----------------------------------------------------|
| Hint     | This question is ambiguous in terms of which specific name is being referred to - the character’s name or the actor’s name. |
| Answer   | In order to figure out its interpretations, we need to consider both possibilities: the character’s name or the actor’s name. Therefore, this question has 2 interpretations: (1) What is the criminal’s character name in The Breakfast Club? (2) What is the name of the actor who played the criminal in The Breakfast Club? The answers to all interpretations are: (1) John Bender was the name of the criminal’s character in The Breakfast Club. (2) Judd Nelson was the actor of the criminal in The Breakfast Club. |

| Question | How many state parks are there in Virginia? |
|----------|-----------------------------------------------|
| Hint     | This question is ambiguous in terms of the time frame or period being referred to. |
| Answer   |  |
```


### --- Page 0024 ---

```markdown
# Prompt D.10: exemplars of WikiAsp

Generate a summary about Aslanhane Mosque including the following aspects: location, history with one aspect per line.  
## # Location  
The mosque is in the old quarter of Ankara next to Ankara Castle. With an altitude of 947 metres (3,107 ft) it overlooks Ankara at 39°56′12″N 32°51′55″E.  
## # History  
The mosque is one of the oldest mosques in Turkey still standing. It was built during the reign of Mesud II of the Anatolian Seljuks in 1290. Its architect was Ebubekir Mehmet. It was commissioned by two Ahi leaders named Hüsamettin and Hasaneddin. However, in 1330, it was repaired by another Ahi leader named Şerefettin after whom the mosque was named. After several minor repairs the mosque was restored by the directorate general of foundations in 2010-2013 term.

Generate a summary about Untold Legends: The Warrior's Code including the following aspects: reception, gameplay, development with one aspect per line.  
## # Reception  
The game received "mixed or average reviews" according to video game review aggregator Metacritic.  
## # Gameplay  
The Warrior's code is a hack n' slash action role-playing game, which concentrates on action-oriented combat.  
## # Development  
As a pre-order bonus, the game was shipped with a small action figure of the Guardian class.

Generate a summary about Raid on St. Augustine including the following aspects: aftermath, background with one aspect per line.  
## # Aftermath  
Once the English had gone Menéndez and the rest of the Spanish settlers returned to find a smoldering ruins and very little left. He soon and begged for help from the viceroy of Cuba and the settlement took a while to build itself back up. The destroyed fort was replaced with the present day Castillo de San Marcos.  
## # Background  
War had already been unofficially declared by Philip II of Spain after the Treaty of Nonsuch in which Elizabeth I had offered her support to the rebellious Protestant Dutch rebels. The Queen through Francis Walsingham ordered Sir Francis Drake to lead an expedition to attack the Spanish New World in a kind of preemptive strike. Sailing from Plymouth, England, he struck first at Santiago in November 1558 then across the Atlantic at the Spanish new world city of Santo Domingo of which was captured and ransomed on 1 January 1586 and following that successfully attacked the important city of Cartagena on 19 February. Drake wanted to strike at another Spanish city on the Main before finally visiting and replenishing Sir Walter Raleigh's new colony of Roanoke Colony on the American East Coast. Then after his hope to make the Transatlantic crossing back to England. The fleet headed north, and in late April Drake put into the Spanish Cuban mainland and his men dug wells in search of fresh water and gathered supplies to help counter an outbreak of dysentery after which he moved on. The fleet traveled north within sight of land on the Florida peninsula sailing past the West coast. On 27 May 1586 as they approached further north a small fort was spotted on the shore, with a small inlet close by. This was the location of St. Augustine, the most northerly town in Spain's New World Empire, and the oldest permanent colonial settlement in North America. Drake knew of the place and was also aware of the fact that the Spanish under Pedro Menéndez de Avilés had ordered all of the French Huguenot colonists that tried to settle in the area executed. Drake decided on one final opportunity to raid and plunder, and a chance to avenge his fellow Protestants.

Generate a summary about Lakewood (Livingston, Alabama) including the following aspects: architecture, history with one aspect per line.  
## # Architecture  
The house has a plan that is relatively rare in early Alabama architecture. The plan features a brick ground floor that is topped by one-and-a-half-stories of wood-frame construction. The ground floor originally contained domestic spaces, with the formal rooms on the principle floor and bedrooms on the upper floor. A central hallway is present on all levels. The facade is five bays wide, with central entrance doors on the ground and principle floors. The bays are divided by two-story Doric pilasters, with the middle third of the facade occupied by a two-tiered tetrastyle Doric portico. Two curved wrought iron staircases ascend from ground level to the front center of the upper portico, leading to the formal entrance.  
## # History  
Lakewood was built for Joseph lake, a native of North Carolina, by Hiram W. Bardwell, a master builder. Construction was completed in 1840. Located adjacent to the University of West Alabama, Julia Strudwick Tutwiler, a Lake relative, periodically resided in the house from 1881 to 1910 while she served as president of the university. It was then known as Livingston Normal College. The house was extensively photographed by Alex Bush for the Historic American Buildings Survey in November and December 1936. Lakewood has continued to be owned by descendants of the Lake family to the current day. The house and its surrounding 10 acres (4.0 ha) of grounds were listed on the Places in Peril in 2012 due to the immediate threat of its acquisition by developers.

Generate a summary about Carlos Moedas including the following aspects: biography, early life, political career with one aspect per line.  
## # Biography  
Carlos Moedas is a Portuguese politician and engineer.  
## # Early Life  
He was born on 10th March 1969 in Lisbon, Portugal.  
## # Political Career  
He served as the Secretary of State for the Digital Transition in the Portuguese government from 2015 to 2019.
```


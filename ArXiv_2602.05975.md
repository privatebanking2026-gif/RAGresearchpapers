# ArXiv 2602.05975

### --- Page 0001 ---

```markdown
# SAGE: Benchmarking and Improving Retrieval for Deep Research Agents

Tiansheng Hu¹  Yilun Zhao²  Canyu Zhang¹  Arman Cohan²  Chen Zhao¹,³  
¹ NYU Shanghai  ² Yale University  ³ Center for Data Science, New York University  
[https://github.com/HughieHu/Sage](https://github.com/HughieHu/Sage)

## Abstract
Deep research agents have emerged as powerful systems for addressing complex queries. Meanwhile, LLM-based retrievers have demonstrated strong capability in following instructions or reasoning. This raises a critical question: can LLM-based retrievers effectively contribute to deep research agent workflows? To investigate this, we introduce SAGE, a benchmark for scientific literature retrieval comprising 1,200 queries across four scientific domains, with a 200,000 paper retrieval corpus. We evaluate six deep research agents and find that all systems struggle with reasoning-intensive retrieval. Using DR Tulu as backbone, we further compare BM25 and LLM-based retrievers (i.e., ReasonIR and tie-Qwen2-7b-instruct) as alternative search tools. Surprisingly, BM25 significantly outperforms LLM-based retrievers by approximately 30%, as existing agents generate keyword-oriented sub-queries. To improve performance, we propose a corpus-level test-time scaling framework that uses LLMs to augment documents with metadata and keywords, making retrieval easier for off-the-shelf retrievers. This yields 8% and 2% gains on short-form and open-ended questions, respectively.

## 1 Introduction
Like human experts, deep research agents (OpenAI, 2025b; GoogleDeepmind, 2024; Perplexity, 2025; Shao et al., 2025a) address complex queries by iteratively searching and synthesizing information across multiple sources. With the help of recent advances in the agentic capabilities of large language models (LLMs), these systems demonstrate strong and robust performance in benchmarks across multiple domains (Aqsa et al., 2025a; Zheng et al., 2025; Li et al., 2025a; Wang and Yuan, 2025; Chervonyi et al., 2025; Zhao et al., 2025). 

At the core of deep research agents lies their retrieval stack (Zheng et al., 2025; Besrour et al., 2025). Recent advances in LLM-based retrievers have shown strong promise, particularly in their ability to follow instructions and support reasoning-intensive retrieval (Shao et al., 2025b; Muennighoff et al., 2025a; Weller et al., 2025a). However, most existing commercial deep research agents adopt proprietary search APIs over large web corpora, which rely on surface-form matching. We thus ask the following research question: **Whether LLM-based retrievers can effectively contribute to deep research agent workflows?**

We propose to systematically study the retrieval behaviors of deep research agents with a scientific literature search task. As shown in Figure 1, 

![SAGE task overview. Given a complex question, the deep research agent (e.g., DR Tulu) iteratively reasons, generates keyword-based sub-queries, searches for relevant papers, and outputs a final answer. We first evaluate the agents with their native web-search tool, and then modify DR Tulu’s MCP service to replace web search with retrievers that performs corpus search over our paper collection.](assets/page_0001_img_1.png)

```

### --- Page 0002 ---

```markdown
# 2 Related Work

## Deep Research Agents
Deep research agents represent a new paradigm of autonomous AI systems designed to tackle complex, multi-step information-seeking tasks (Huang et al., 2025). Commercial systems including OpenAI's Deep Research (OpenAI, 2025b), Google’s Gemini Deep Research (GoogleDeepmind, 2025a) have demonstrated impressive performance on challenging benchmarks such as BrowseComp (Wei et al., 2025). In parallel, open-source efforts have rapidly advanced, with systems such as SearchR1, WebThinker, and Tongyi Deep Research approaching competitive performance \cite{Li et al., 2025b; Jin et al., 2025; Li et al., 2025c; Team et al., 2025}. Notably, DR Tulu (Shao et al., 2025a) is the first open model explicitly trained for open-ended, long-form deep research via reinforcement learning, achieving results comparable to proprietary systems on benchmarks. Despite these advances, existing deep research agents rely primarily on web search or proprietary retrieval backends. Whether such agents can function as plug-and-play solutions when paired with LLM-based retrievers over closed-domain corpora remains largely unexplored, which we systematically investigate in this work.

## LLM-based Retrievers
The advent of large-scale contrastive learning marked a significant advancement for retrievers (Ni et al., 2022; Gao et al., 2023; Li et al., 2023; Wang et al., 2022; Chen et al., 2024). More recently, decoder-based retrievers such as LLM2Vec (BehnamGhadir et al., 2024) and GritLM (Mueninghoff et al., 2025a) have emerged, repurposing generative LLMs for embedding tasks. Beyond general-purpose embeddings, recent work has explored training LLM-based retrievers to enhance specific capabilities. Promptiver (Weller et al., 2025a) introduces instruction-trained retrievers that can be prompted like language models. ReasonIT (Shao et al., 2025b) presents the first retriever specifically trained for reasoning-intensive tasks such as finding similar coding problems. However, whether these retrievers can collaborate effectively with agentic search paradigms remains unexplored, and our work bridges this gap.

## Test-time Scaling for Retrieval
Test-time scaling has emerged as an effective paradigm for en-
```


### --- Page 0003 ---

```markdown
### 3 SAGE Benchmark

This section introduces SAGE Benchmark. We begin with motivating SAGE (§3.1), then present data curation and evaluation metric for shot-form questions (§3.2), followed by those for open-ended questions (§3.3) and corpus construction (§3.4).

#### Problem Formulation

Unlike traditional RAG system, which given a query $q$, retrieves documents $D = \text{Retrieve}(q)$ and generates a response conditioned on $D$ in one shot, a deep research agent is an agentic system composed of one or more LLMs augmented with search tools. Such agents autonomously plan multi-step research procedures, retrieve information from online sources, and synthesize evidence into a comprehensive, well-cited answer. Specifically, the agent selects an action $a_i \in \{ \text{think}, \text{tool}, \text{answer} \}$ at each step: reasoning internally, issuing a sub-query $q_i$ to retrieve documents $D_i = \text{Retrieve}(q_i)$, or producing the final answer conditioned on the accumulated evidence $\bigcup D_i$. This formulation enables the agent to decompose complex questions into subqueries $q_1, q_2, \ldots, q_n$, progressively building an evidence base across multiple retrieval rounds.

#### 3.1 Why Scientific Literature Search?

Our primary goal is to study the retrieval behavior of deep research agents. To achieve this, we choose scientific literature search as our testbed for several reasons: (1) **Task is Common and Impactful.** Searching for relevant literature is an integral part of the research process, whether it is to verify if an idea has been explored before or to collect related work. Therefore a strong agentic system could significantly accelerate the scientific discovery process. (2) **Controllable Domain Specific Corpus.** Existing deep research tasks rely on entire web as a corpus, limited by the use of commercial search APIs. In contrast, scientific literature search adopts collections of papers as a controlled corpus for precise evaluation of different retrievers. (3) **Existing Datasets Fall Short.** While several datasets exist for scientific literature search, they fail to evaluate deep research agents. This is because the papers used in these datasets are outdated and often include LLMs' pre-existing knowledge. However, scientific literature is a rapidly evolving field, with new papers published daily. Our dataset uses up-to-date papers to better study the retrieval behavior of deep research agents in a dynamic environment.

Based on these reasons, we construct SAGE, which includes 1,200 questions spanning short-form and open-ended types. These questions cover four critical scientific domains: Computer Science, Natural Science, Healthcare, and Humanities. For each domain, we curate a corpus of 50,000 up-to-date papers. The statistics of our dataset are presented in Table 1.

#### 3.2 Short-form Questions

The first type of questions in SAGE is short-form questions. Similar to existing deep research benchmarks (Wei et al., 2025; Chen et al., 2025), short-form questions emphasize two key characteristics: (1) **Intensive Reasoning.** These questions require deep research agents to browse multiple papers, synthesize detailed and scattered information, and derive a final answer; (2) **Verifiability.** The answer to each question is unique and fixed, therefore the correctness is easily verifiable. An example of short-form question can be found at Figure 2.

| Property                | Com. Sci. | Nat. Sci. | Health. | Human. |
|------------------------|-----------|-----------|---------|--------|
| **Short-Form Questions**|           |           |         |        |
| Query Num              | 150       | 150       | 150     | 150    |
| Query Length           | 201.5     | 180.3     | 187.6   | 188.3  |
| GT Documents           | 1.00      | 1.00      | 1.00    | 1.00   |
| DB Size                | 47,637    | 50,000    | 50,000  | 39,032 |
| **Open-Ended Questions**|          |           |         |        |
| Query Num              | 150       | 150       | 150     | 150    |
| Query Length           | 99.6      | 103.9     | 101.5   | 101.2  |
| GT Documents           | 17.62     | 12.67     | 10.83   | 9.94   |
| DB Size                | 46,756    | 48,879    | 47,745  | 37,506 |

Table 1: Our Benchmark statistics. Domains: Computer Science (Com. Sci.), Natural Science (Nat. Sci.), Healthcare (Health.), and Humanities (Human.). Query length is in tokens. GT Documents = average ground truth papers per query.
```

### --- Page 0004 ---

```markdown
![Overview of short-form questions that require intensive reasoning over metadata, paper details and inter-paper relationships. Each question consists of three parts and has only one ground-truth answer.](assets/page_0004_img_1.png)

![Overview of open-ended questions that are grounded on real-world scenarios. Each question consists of three parts and multiple ground-truth papers weighted by their relevance.](assets/page_0004_img_2.png)

## Data Curation
We construct question-answer pairs from three sources: extracted paper metadata (e.g., author count, title length), figures and tables extracted using PyMuPDF (McKie, 2025), and inter-paper relationships established via reference overlap. To establish inter-paper relationships, we compute the citation overlap between papers, which we consider two papers as related if they share at least four common references in their reference lists. Specifically, we first sample a seed paper and a related paper published after 2024 from major venues in each domain (e.g., ACL, ICML, NeurIPS for computer science). Next, we extract the corresponding metadata, figures, tables, and inter-paper relationships. We then prompt LLMs (GPT-5-mini (OpenAI, 2025a) in this case) to generate questions that require reasoning across these multiple sources. The answer to each question is the seed paper itself.

## Evaluation Metric
We use Exact Match (EM) as the metric to evaluate whether the ground truth answer is included in the output text or citations.

### 3.3 Open-Ended Questions
Unlike short-form questions, which primarily aim to objectively measure and compare different deep research systems (Rodriguez and Boyd-Graber, 2021), open-ended questions are grounded in real-world scenarios. They mimic the types of questions researchers encounter when conducting literature reviews and exploring new ideas. An example of an open-ended question can be found in Figure 3.

## Data Curation
The open-ended questions consist of two components: (1) the background context of the research topic, and (2) the shared citations between a pair of papers. We construct questions through the following pipeline: First, we leverage the reference-overlap information from Section 3.2 to select paper pairs. For each selected pair, we adopt GPT-5-mini (OpenAI, 2025a) to analyze the inter-relationship between the two papers and the reasons for their shared citations. Based on this analysis, GPT-5-mini (OpenAI, 2025a) generates corresponding questions. Note that each open-ended question has multiple ground truth papers, so we create the ground-truth using a hierarchical structure. The most relevant papers are the selected seed paper, followed by those cited by both papers.

## Evaluation Metric
Given the list of ground-truth papers for open-ended questions, we first assign discrete relevance scores $r \in \{2, 1, 0\}$: Most Relevant ($r=2$) for the two seed papers; Relevant ($r=1$) for the intersection of the core papers’ references; and Not Relevant ($r=0$) for all others. We report Weighted Recall to capture all papers from both the output text and citation lists:

$$
\text{Weighted Recall} = \sum_{d \in L} g(rel(d)) \bigg/ \sum_{d \in G} g(rel(d)) \, (1)
$$

where $L$ is the set of retrieved documents, $G$ is the set of all relevant documents, and $g(r) = r$ is the linear gain function.

### 3.4 Corpus Construction
For each domain, we construct a 50k-paper corpus using only open-access PDFs to ensure accessibility. The corpus begins with the following: (1) the ground-truth target paper and its highest-overlap partner from the computed reference-overlap information, (2) the intersection of their references, and (3) the union of their references. We then expand the corpus by sampling papers published in or after 2020 from major venues in the respective domain until the desired corpus size is reached. Due to the limited availability of papers in the humanities, this
```

### --- Page 0005 ---

```markdown
# 4 Experiment

In this section, we first describe the experiment setup for deep research agents with web search (§4.1) and report their results on SAGE (§4.2). We then move to a controlled setting by evaluating retriever performance within the same deep-research agent (i.e., DR Tulu) using a retrieval corpus we constructed (§4.3 and §4.4). At last, we present ablation results on short-form questions (§4.5).

## 4.1 Web-Search Experiment Setup

We evaluate two categories of deep research agents: (1) **Proprietary deep research agents**, including GPT-5 (OpenAI, 2025a), GPT-5-mini (OpenAI, 2025a), GPT-5-nano (OpenAI, 2025a), Gemini-2.5-Pro (GoogleDeepmind, 2025b), and Gemini-2.5-Flash (GoogleDeepmind, 2025a), by using the official APIs; (2) **Open-source deep research agents**, notably AI2’s recently released DR Tulu (Shao et al., 2025a), which sets a new SOTA among open-source deep-research agents. For GPT series¹, we set the “reasoning effort” to “medium”, and enable web search functionality. For Gemini series, we set “thinkingBudget” to “-1” to enable dynamic thinking and give web search permission. For DR Tulu (Shao et al., 2025a), we deploy the model on a server equipped with one H100 GPU and perform inference using VLLM.

## 4.2 Web-Search Results

Table 2 presents the results of deep research agents with web search. We have the following findings:

**GPT-5 leads overall on short-form questions, while open-ended questions vary more by domain and model.** On short-form questions, the GPT-5 series delivers the strongest performance across all domains, with GPT-5 achieving the best EM (71.69%). In contrast, open-ended questions induce more heterogeneous outcomes: GPT-5-nano performs best in healthcare, while Gemini-2.5-flash is competitive in computer science and humanities. Notably, DR Tulu outperforms the closed-source Gemini-2.5 series agents on short-form questions, indicating that open-source deep research agents can match or exceed proprietary systems in precise, retrieval-heavy settings.

**Search quantity is not the main driver of accuracy.** On short-form questions, Gemini-2.5-flash issues nearly twice as many web-search calls as GPT-5, and DR Tulu returns an exceptionally large number of references (37.32 on average), yet both trail GPT-5 by a substantial margin. This pattern suggests that brute-force searching or reference accumulation is insufficient for precise retrieval. Instead, stronger models appear to benefit from more accurate query decomposition and more targeted evidence selection, achieving higher accuracy with fewer, better-aligned searches.

**Agents adapt search effort differently across query types.** When moving from short-form to open-ended questions, DR Tulu and the Gemini series reduce the number of searches, consistent with looser constraints and potentially earlier stopping. In contrast, GPT-5 increases search activity on open-ended questions and attains the best overall results, with only a modest and acceptable increase in the number of references compared with other agents.

**Query decomposition strategies differ across agents.** As shown in Figure 7 and Figure 8 in Appendix, the proprietary models tend to decompose queries into more phrasal, semantically structured search queries, whereas DR Tulu sub-queries more often resemble less structured keyword concatenations. This difference aligns with the observed efficiency gap, where more structured decomposition corresponds to fewer but higher-yield searches and improved retrieval precision.

## 4.3 Corpus-Search Experiment Setup

Motivated by our web-search results on the dataset, we next investigate how LLM-based retrievers integrate with deep research workflows. We use DR Tulu (Shao et al., 2025a) as the backbone agent for all corpus-search experiments. We modify DR Tulu’s MCP service so it can only use our provided retriever as the search tool. We study three retrievers, which are as follows: BM25 (Robertson et al., 1994), a sparse retriever; get-Qwen-2-7B-instruct (Li et al., 2023), a LLM-based retriever; and ReasonIR (Shao et al., 2025b), a reasoning-intensive retriever.

**Retrieval Index Construction.** Before experiment, we first download all PDFs according to
```


### --- Page 0006 ---

```markdown
| Method   | Com. Sci. | Healthcare | Humanities | Nat. Sci. | Avg. Searches | Avg. Refs | Avg. Perf. |
|----------|-----------|------------|-------------|-----------|---------------|-----------|------------|
| **WEB SEARCH** |           |            |             |           |               |           |            |
| Short-Form Questions (Exact Match) |           |            |             |           |               |           |            |
| GPT-5    | 57.3      | 78.7       | 79.1        | 71.7      | 8.78          | 6.54      | 71.7       |
| GPT-5-mini | 40.0    | 72.0       | 70.7        | 66.4      | 8.15          | 4.69      | 62.3       |
| GPT-5-nano | 30.7    | 46.7       | 62.0        | 44.3      | 8.92          | 7.06      | 45.9       |
| DR Tulu  | 36.0      | 58.0       | 49.3        | 44.7      | 7.35          | 37.32     | 42.0       |
| Gemini-2.5-pro | 27.7 | 47.8      | 40.3        | 37.7      | 14.02         | 1.43      | 38.5       |
| Gemini-2.5-flash | 30.3 | 43.7     | 41.6        | 36.4      | 15.64         | 3.79      | 38.0       |
| Open-Ended Questions (Weighted Recall) |           |            |             |           |               |           |            |
| GPT-5    | 35.1      | 25.0       | 18.8        | 26.2      | 13.69         | 29.02     | 26.3       |
| GPT-5-mini | 27.4    | 17.8       | 13.9        | 22.4      | 10.07         | 30.70     | 20.4       |
| GPT-5-nano | 25.9    | 29.8       | 15.7        | 21.1      | 8.59          | 30.27     | 20.6       |
| DR Tulu  | 18.0      | 17.2       | 14.1        | 20.2      | 4.25          | 35.95     | 17.4       |
| Gemini-2.5-pro | 28.2 | 10.1      | 15.7        | 9.8       | 6.50          | 22.12     | 16.0       |
| Gemini-2.5-pro | 20.0 | 5.2       | 12.2        | 6.7       | 10.77         | 14.18     | 11.0       |
| **COURSE SEARCH** |           |            |             |           |               |           |            |
| Short-Form Questions (Exact Match) |           |            |             |           |               |           |            |
| BM25 k=10 | 63.3    | 88.8       | 84.4        | 88.4      | 6.42          | 40.0      | 81.2       |
| BM25 k=5  | 56.0    | 79.9       | 82.0        | 85.3      | 7.54          | 22.4      | 75.8       |
| get-Qwen k=10 | 44.4 | 69.7      | 73.3        | 64.4      | 5.88          | 33.2      | 63.0       |
| ReasonIR k=10 | 28.0 | 57.0      | 61.3        | 50.7      | 7.51          | 35.8      | 49.3       |
| get-Qwen k=5 | 32.9  | 52.1      | 62.6        | 40.0      | 4.82          | 14.2      | 46.9       |
| ReasonIR k=5 | 25.9  | 42.2      | 48.2        | 38.4      | 8.78          | 17.4      | 38.7       |
| Open-Ended Questions (Weighted Recall) |           |            |             |           |               |           |            |
| get-Qwen k=10 | 28.9 | 33.5      | 34.6        | 33.0      | 4.54          | 29.3      | 30.7       |
| BM25 k=10 | 20.3    | 36.2       | 34.0        | 32.3      | 4.17          | 29.9      | 30.7       |
| ReasonIR k=10 | 16.1 | 29.5      | 32.0        | 27.4      | 4.44          | 26.8      | 26.2       |
| get-Qwen k=5 | 22.4  | 26.3      | 29.3        | 26.2      | 4.79          | 15.4      | 26.0       |
| BM25 k=5  | 18.2    | 29.4       | 28.3        | 26.3      | 4.72          | 16.7      | 25.5       |
| ReasonIR k=5 | 11.3  | 18.5      | 22.3        | 17.0      | 6.21          | 15.8      | 17.3       |

Table 2: Performance comparison across two question types. Avg. Perf. denotes the average performance across all domains. Bold indicates the best result and underline indicates the second best. For Avg. Searches: dark red = highest, light red = lowest. For Avg. Refs: dark green = highest, light green = lowest.

The URLs in the SAGE dataset (detailed in Section 3.4). We then convert them to markdown using PyMuPDF (McKie, 2025) for text and PDF-Plumber (Singer-Vine and Jain, 2025) for tables. Next, we embed the first 32,000 tokens of each markdown file with the corresponding retriever to ensure that the vast majority of each PDF’s content is retained while matching the maximum input length of get-Qwen-2-7B-instruct. We embed each document individually, setting the batch size to 1 to avoid unnecessary padding. Both ReasonIR and get-Qwen-2-7B-instruct embeddings are computed on a single H100 GPU. We present the paper length distribution for all four domains in the Appendix Figure 9, Figure 10, Figure 11 and Figure 12.

**Retrieval Setup.** During experiments, the DR Tulu agent is deployed on two H100 GPUs, where one running vLLM for answer generation and the other running MCP powered by the selected retriever. We set the maximum search iteration to 10 and for each retriever we evaluate two settings for the number of results returned per search, which are top-5 and top-10 (i.e., k=5 and k=10). Each retrieval step returns a list of paper titles together with their abstracts.

**4.4 Corpus-Search Results** Table 2 presents our results of DR Tulu using in-house retrievers as search tools. We have the following main findings:

BM25 dominates LLM-based retrievers on short-form questions, while the gap for open-ended questions is narrower. On short-form questions, BM25 significantly outperforms LLM-based retrievers by roughly 30%, suggesting that sparse lexical matching is better aligned with multiconstraint evidence retrieval in this setting. On open-ended questions, BM25 and get-Qwen-2-7B...
```

### --- Page 0007 ---

```markdown
| Method         | EM      | ΔMet.  | ΔDet. | ΔRel.  |
|----------------|---------|--------|-------|--------|
| Web Search      |         |        |       |        |
| GPT-5          | 71.69   | -24.45 | -3.85 | -16.92 |
| DR Tulu        | 42.00   | -17.92 | -21.60| -6.88  |
| Gemini-2.5-Pro | 38.50   | -11.43 | -14.69| -5.67  |
| Corpus Search   |         |        |       |        |
| DR Tulu (BM25) | 75.84   | -8.17  | -1.94 | 24.59  |
| DR Tulu (ReasonIR) | 38.72 | -5.33  | -9.84 | 15.99  |

### Figure 4: An illustrative case where LLM-based retrieval fails due to semantic drift. The query seeks a paper that uses physics-informed heuristics. ReasonIR over-emphasizes title-level keywords (highlighted in red) and thus retrieves wrong papers. The retrieved content then reinforces this focus in subsequent retrieval steps, creating a feedback loop that increasingly prioritizes “physics-informed” in title. In contrast, BM25 remains anchored by lexical matching in similar subqueries and avoids this drift.

LLM-based retrievers suffer from reduced diversity under long-document constraints. LLM-based retrievers also face information loss when documents approach the maximum input length, and embedding convergence can further reduce per-search diversity. We define Unique References per Search (URS) as the average number of retrieved documents returned per search call, computed as the ratio of the average number of documents to the average number of searches. Under top-5 on short-form questions, BM25 achieves URS of 2.97, whereas ReasonIR attains only URS of 1.98. This indicates that LLM-based retrievers are less effective to surface the target document under a fixed search budget.

Low-diversity decomposition blunts retriever differences on open-ended queries. DR Tulu exhibits relatively low diversity in its query decomposition. BM25 appears more compatible with DR Tulu’s decomposition and is more robust to long documents, but it does not open a clear advantage on open-ended queries. A plausible explanation is that DR Tulu’s sub-queries cover only a limited portion of the evidence space, so even when retrievers behave differently, multiple ground-truth targets are only partially retrieved.

### 4.5 Ablation
We conduct ablation studies using short-form questions, as their answers are easier to verify. As discussed earlier, these questions span three aspects of query information: paper metadata, multimodal details, and inter-paper relationships. Manual inspection shows that leveraging any two of these components is sufficient to locate 93.67% of the target.
```

### --- Page 0008 ---

```markdown
# Test-Time Corpus Scaling

Our analysis in Section 4 reveals a fundamental limitation of existing deep research agents: certain papers requiring intensive reasoning are inherently difficult to retrieve. Prior work (SU et al., 2025; Shao et al., 2025b) proposes to address this challenge through test-time scaling on the query side, augmenting queries with reasoning chains. In contrast, we propose an alternative form of test-time scaling at the document corpus side. The key intuition is that, rather than increasing query complexity, we incorporate reasoning-derived information into documents, making them easier to retrieve for off-the-shelf retrievals.

## 5.1 Method

Since DR Tulu primarily issues keyword-based queries, we augment each document’s Markdown by prepending salient keywords to improve retrieval effectiveness. Specifically, we first obtain key bibliographic metadata, including publication venue, year, authors, and citation counts. In addition, we use Qwen3-Next-80B-A3B-Instruct (Qwen, 2025) to process the Markdown and extract eight topic-relevant keywords that summarize the paper’s core contributions. These fields are formatted as emphasized keywords and prepended to each document, so that both bibliographic signals and high-level semantic cues are surfaced for effective keyword-based retrieval. 

2 We scale the corpus by augmenting documents with additional information in bag of keywords. With LLMs, future work could explore more aggressive corpus scaling strategies.

## 5.2 Results

In this experiment, we set the maximum number of search iterations to 10 and retrieve the top-5 results per search. Table 4 reports the results of DR Tulu with three different retrievers, both before and after applying test-time corpus scaling.

### BM25 benefits most from corpus scaling.

On short-form questions, BM25 achieves absolute gain of 8.18%, LLM-based retrievers exhibit only modest improvements. This is largely because BM25 is more sensitive to keyword signals, while LLM-based retrievers, as discussed, struggle when documents approach input-length limits. Therefore, the added information makes documents only marginally easier for them.

### Limited improvement on open-ended questions.

All three retrievers show only marginal improvements on open-ended questions. This result aligns with our earlier observation at section 4.4 that DR Tulu’s (and other deep research agents) generated query lacks diversity, which limits retrieval breadth and prevents corpus-level scaling from fully translating into downstream performance gains.

## 6 Conclusion

We introduce SAGE, a benchmark for reasoning-intensive scientific literature retrieval. Through extensive evaluation, we reveal a critical finding: LLM-based retrievers underperform BM25 by approximately 30% in deep research agent workflows, as existing agents generate keyword-oriented subqueries. To address this limitation, we propose corpus-level test-time scaling, which enriches papers with metadata and LLM-generated keywords, and achieves consistent improvements. Our work highlights that effective collaboration between retrievers and agents requires further adaptation.
```

![Performance before and after corpus-level test-time scaling. Short-form is evaluated by Exact Match (EM) (%) and open-ended evaluated by Weighted Recall (%). Improvements are shown with green background.](assets/page_0008_img_1.png)

### --- Page 0009 ---

```markdown
| **Limitations and Future Work**                                                                                                           | **Acknowledgements**                                                                                     |
|-------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| We acknowledge limitations in our study. We do not perform instruction fine-tuning or alignment on the open-source deep-research agents. As a result, we are unable to assess whether training agents to adapt their query generation strategies based on the underlying retriever type could improve performance. Exploring such retriever-aware agent training remains a valuable direction for future work. Additionally, most of our behavioral analysis is conducted on DR Tulu, whose post-training procedures may significantly influence the observed agent behaviors. Consequently, our findings may not fully generalize to agents with different training recipes or base model architectures. | Tiansheng Hu and Chen Zhao were supported by NYU Shanghai Center for Data Science. This work was supported in part through the NYU IT High Performance Computing resources, services, and staff expertise. |

| **References**                                                                                                                                                                                                 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. 2025. Agent S2: A compositional generalist-specialist framework for computer use agents. In Conference on Language Modeling. |
| Pariashd BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. LLM2vec: Large language models are secretly powerful text encoders. In Conference on Language Modeling. |
| Ines Bessrour, Jingbo He, Tobias Schreiber, and Michael Rafter. 2025. Reagrant: Multi-agent retrieval-augmented generation for attributed question answering. arXiv preprint arXiv:2506.16988. |
| Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3-embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In Findings of the Association for Computational Linguistics. |
| Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifmoghaddam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. 2025. Browscomp-plus: A more fair and transparent evaluation benchmark of deep-research agents. arXiv preprint arXiv:2508.06600. |
| Yuri Chervoniy, Trieu H. Trinh, Miroslav Oltak, Xi-aomeng Yang, Hoang Nguyen, Marcelo Menegali, Junhyuk Jung, Junsu Kim, Vikas Verma, Quoc V. Le, and Thang Luong. 2025. Gold-medalist performance in solving olympiad geometry with alphageometry2. arXiv preprint arXiv:2502.03544. |
| Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the Annual Meeting of the Association for Computational Linguistics. |
| GoogleDeepmind. 2024. Gemini deep research. Official blog post introducing Gemini Deep Research. |
| GoogleDeepmind. 2025a. Gemini 2.5 flash best for fast performance on everyday tasks. Official blog post introducing Gemini 2.5 Flash models. |
| GoogleDeepmind. 2025b. Gemini 2.5: Our most intelligent ai model. Official blog post introducing Gemini 2.5 Pro models. |
| Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, and Jun Wang. 2025. Deep research agents: A systematic examination and roadmap. arXiv preprint arXiv:2506.18096. |
| Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training LLMs to reason and leverage search engines with reinforcement learning. In Conference on Language Modeling. |
| Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, and Xiangmin Xu. 2025a. QuantAgents: Towards multiparadigm financial system via simulated trading. In Findings of the Conference on Empirical Methods in Natural Language Processing. |
| Xiaoxi Li, Guantong Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025b. Search-o1: Agentic search-enhanced large reasoning models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. |
| Xiaoxi Li, Jiajie Jin, Guantong Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and Zhicheng Dou. 2025c. Webthinker: Empowering large reasoning models with deep research capability. arXiv preprint arXiv:2504.21776. |
| Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281. |
| Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query rewriting in retrieval-augmented large language models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. |
```

### --- Page 0010 ---

```markdown
| Author(s)                                                                 | Title                                                                                                   |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Jory K. McKie. 2025.                                                      | Pymupdf: Python bindings for mupdf. Version 1.26.7.                                                   |
| Niklas Muenchhoff, Hongjin SU, Liang Wang, Nan Yang, Furui Wei, Tao Yu, | Generative representational instruction tuning. In The International Conference on Learning Representations. |
| Amrapneet Singh, and Douwe Kiela. 2025a.                                 |                                                                                                         |
| Niklas Muenchhoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei,  | s1: Simple test-time scaling. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. |
| Hannah Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and  |                                                                                                         |
| Tatsunori Hashimoto. 2025b.                                               |                                                                                                         |
| Jianno Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego,      | Large dual encoders are generalizable retrievers. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. |
| Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yifei Yang. 2022. |                                                                                                         |
| OpenAI. 2025a.                                                            | Introducing gpt-5. Official blog post introducing GPT-5 models.                                        |
| OpenAI. 2025b.                                                            | Official deep research. Official blog post introducing OpenAI Deep Research.                           |
| Perplexity. 2025.                                                         | Introducing perplexity deep research. Official blog post introducing Perplexity Deep Research.         |
| Qwen. 2025.                                                               | Qwen3-next: Towards multi-task training & inference efficiency. Official blog post introducing Qwen3-next family. |
| Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, | Okapi at TREC-3. In Proceedings of The Text REtrieval Conference.                                      |
| and Mike Gattford. 1994.                                                 |                                                                                                         |
| Pedro Rodriguez and Jordan Boyd-Graber. 2021.                             | Evaluation paradigms in question answering. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. |
| Rulin Shao, Akari Asai, Shannon Zeigang Shen, Hamish Irwin, Varsha Kishore, | Reinforcement learning with evolving priors for deep research. In Conference on Language Modeling.     |
| Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, |                                                                                                         |
| Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soltani, Faez Brahman,     |                                                                                                         |
| Wen tu Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannah Hajishirzi, |                                                                                                         |
| and Peng Wei. 2025a.                                                     |                                                                                                         |
| Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muenchhoff, Xi Victoria Lin, | ReRank: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations. |
| Daniela Rus, Bryan Hsiang Low, Sewon Min, Wen tu Yih, Pang Wei Koh, and  |                                                                                                         |
| Luke Zettlemoyer. 2025b.                                                 |                                                                                                         |
| Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Ihtak Shafran, Karthik R Narasimhan, | PromptRetriever: Instruction-trained retrievers can be prompted like language models. In The International Conference on Learning Representations. |
| and Yuna Cozman. 2023.                                                   |                                                                                                         |
| Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Avi Kumar. 2025.      | Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The International Conference on Learning Representations. |
| Hongjin SU, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muenchhoff,     | BRIGHT: A realistic and challenging benchmark for reasoning-intensive retrieval. In The Thirteenth International Conference on Learning Representations. |
| Han Yu Wang, Liu Haisu, Quan Shi, Zachary S Siegel, Michael Tang, Roux Xion, |                                                                                                         |
| Jinsung Yoon, Sceron A. Orik, Danqi Chen, and Tao Yu. 2025.              |                                                                                                         |
| Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, | Multilingual euqal 65 text embeddings: A technical report. arXiv preprint arXiv:2402.05672.          |
| Guanguo Li, Xuion Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, |                                                                                                         |
| Lianqiao Si, Lutu Ou, Liwen Zhang, Peng Xie, Rui Ye, Wenbiao Yin, Ximiao Yu, |                                                                                                         |
| Xinyu Wang, Xixi Wu, Xuanzhong Chen, Yida Zhao, Zhen Zhang, Zhengwei Tao, |                                                                                                         |
| Zhongwang Zhang, Zile Qiao, Chen Wang, Donglei Yu, Gang Fu, Haiying Shen, |                                                                                                         |
| Jiyin Yang, Jun Lin, Junkai Zhang, Kui Zeng, Li Yang, Hailong Yin, Maokai Song, |                                                                                                         |
| Mingpeng Liao, Peng Xia, Qian Xiao, Rui Min, Ruikui Ding, Runnan Fang,   |                                                                                                         |
| Shaowei Chen, Shen Huang, Shihang Wang, Shihao Cai, Weizhong Shen, Xiaolong Wang, |                                                                                                         |
| Xin Guan, Xinyu Geng, Yingcheng Shi, Yuning Wu, Zhuo Chen, Zijian Li, and Yong Jiang. 2025. | Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701.                               |
| Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangang Malyugin, and  | Multilingual euqal 65 text embeddings: A technical report. arXiv preprint arXiv:2402.05672.          |
| Furu Wei. 2024.                                                          |                                                                                                         |
| Ziqi Wang and Boqin Yuan. 2025.                                          | L-mars: Legal multi-agent workflow with orchestrated reasoning and agent search. arXiv preprint arXiv:2509.00761. |
| Jason Wei, Zhiqiang Sun, Spencer Papay, Scott McKinnon, Jeffrey Han,    | A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516.              |
| Huyong Chen, Alex Tachard Passos, William Fedus, and Amelia Glascott.   |                                                                                                         |
| Orion Weller, Benjamin Van Durme, Dawn Lawrie, Ashwin Paranjape, Yuhao Zhang, | PromptRetriever: Instruction-trained retrievers can be prompted like language models. In The International Conference on Learning Representations. |
| and Jack Hessel. 2025a.                                                  |                                                                                                         |
| Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, and  | Rank1: Test-time compute for re-ranking in information retrieval. In Conference on Language Modeling.  |
| Benjamin Van Durme. 2025b.                                               |                                                                                                         |
```

### --- Page 0012 ---

```markdown
# A Appendix

## A.1 Query-Answer Example

**Query:**  
Find me the paper that matches this requirement:  
- Published between 2023 and 2025 (inclusive), presented at the Annual Meeting of the Association for Computational Linguistics, and building on 40 prior works.  
- A key visualization shows their method dramatically improves T5-small versus the vanilla baseline: typical JGA gains of 0.15–0.25, with a striking 40.25 jump on Task 4 (their method T5 = 0.58 vs vanilla T5 = 0.33).  
- This paper and “APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference” both cite “Movement Pruning: Adaptive Sparsity by Fine-Tuning” for weight-gradient salience scoring to guide which parameters to prune or tune, and “Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning” for sensitivity, smoothing, and uncertainty measures to allocate and remove tuning parameters. They share 7 citations in total and are specifically related to salience-driven pruning and adaptive budget allocation for parameter-efficient fine-tuning.

**Answer:**  
TaSL: Continual Dialog State Tracking via Task Skill Localization and Consolidation

![Example of a Short-Form question](assets/page_0012_img_1.png)

---

## A.2 Query-Decomposition Case Study

**Query:**  
Find me the paper that matches this requirement:  
- Published in the period from 2021 to 2025, presented at the Annual Meeting of the Association for Computational Linguistics conference, and written by 5 authors.  
- Its primary figure demonstrates that the architecture explicitly leverages both source KBQA training data and target KBQA few-shot examples to influence retrieval, reranking, and generation, highlighting transfer from source domains plus few-shot adaptation to the target as a core design choice.  
- This paper and “RetinaQA: A Robust Knowledge Base Question Answering Model for both Answerable and Unanswerable Questions” both cite “Few-shot In-context Learning on Knowledge Base Question Answering” for its few-shot in-context generate-then-ground sketching pipeline and “Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases” for the GrailQA dataset and s-expression logical-form formalism. They share 17 citations in total and are specifically related by emphasis on transferable sketch-based logical-form generation and evaluation on GrailQA.

**Decomposed Search Queries:**
1. site:aclanthology.org ACL 2021 knowledge base question answering GrailQA sketch generation transfer few-shot
2. site:aclanthology.org 2024.acl knowledge base question answering few-shot transfer GrailQA sketch
3. RetinaQA arXiv PDF 2403.10849
4. Semantic Scholar RetinaQA 2403.10849 references
5. site:semanticscholar.org RetinaQA 2403.10849 references GrailQA
6. site:aclanthology.org RetinaQA 2024.acl-long
7. Shu and Yu 2024 KBQA few-shot in-context learning TIARA retriever ACL
8. ACL 2024 KBQA transfer few-shot retriever reranking generation figure source target few shots site:aclanthology.org 2024.acl-long
9. Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases ACL Anthology 2021

**Answer:**  
Few-shot Transfer Learning for Knowledge Base Question Answering: Fusing Supervised Models with In-Context Learning

![GPT-5 Query Decomposition Example](assets/page_0012_img_2.png)
```

### --- Page 0013 ---

```markdown
## Query:
Find me the paper that matches this requirement:
- Published in the period from 2021 to 2025, presented at the Annual Meeting of the Association for Computational Linguistics conference, and written by 5 authors.
- Its primary figure demonstrates that the architecture explicitly leverages both source KBAQ training data and target KBAQ few-shot examples to influence retrieval, reranking, and generation, highlighting transfer from source domains plus few-shot adaptation to the target as a core design choice.
- This paper and “RetinaQA: A Robust Knowledge Base Question Answering Model for both Answerable and Unanswerable Questions” both cite “Few-shot In-context Learning on Knowledge Base Question Answering” for its few-shot in-context generate-then-ground sketching pipeline and “Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases” for the GrailQA dataset and s-expression logical-form formalism. They share 17 citations in total and are specifically related by emphasis on transferable sketch-based logical-form generation and evaluation on GrailQA.

### Decomposed Search Queries:
1. RetinaQA knowledge base question answering ACL
2. FUSIC-KBA ACL few-shot transfer learning KBAQ figure retrieval reranking LLM
3. RetinaQA ACL 2024 authors GrailQA s-expression citations “Few-shot In-context Learning on Knowledge Base Question Answering”
4. RetinaQA ACL authors arXiv 2403.10849 GrailQA s-expression
5. RetinaQA ACL 2024 authors arXiv 2403.10849 GrailQA s-expression citations

### Answer:
Few-shot Transfer Learning for Knowledge Base Question Answering: Fusing Supervised Models with In-Context Learning.

![Distribution of markdown length (in tokens) for 1,000 randomly sampled documents from the Computer Science domain.](assets/page_0013_img_1.png)
Figure 9: Distribution of markdown length (in tokens) for 1,000 randomly sampled documents from the Computer Science domain.

![Distribution of markdown length (in tokens) for 1,000 randomly sampled documents from the Healthcare domain.](assets/page_0013_img_2.png)
Figure 10: Distribution of markdown length (in tokens) for 1,000 randomly sampled documents from the Healthcare domain.

![Distribution of markdown length (in tokens) for 1,000 randomly sampled documents from the Humanities domain.](assets/page_0013_img_3.png)
Figure 11: Distribution of markdown length (in tokens) for 1,000 randomly sampled documents from the Humanities domain.

Figure 8: Dr-Tulu Query Decomposition Example.
```

### --- Page 0014 ---

```markdown
![Distribution of markdown length (in tokens) for 1,000 randomly sampled documents from the Natural Science domain.](assets/page_0014_img_1.png)

### A.3 Document Length Distribution

### A.4 Comparison with BrowseComp-Plus: Retriever Behavior

We observe a retriever ranking that differs from BrowseComp-Plus (Chen et al., 2025). In our experiments, BM25 consistently outperforms LLM-based dense retrievers (e.g., gt-Qwen2-7B-Instruct), whereas BrowseComp-Plus reports stronger performance from dense retrievers such as Qwen3-Embed-8B. We attribute the discrepancy to differences in (i) task characteristics, (ii) retriever implementations, and (iii) agent model strength and query decomposition.

Longer documents and weaker answer locality in our setting. BrowseComp-Plus uses substantially shorter documents on average (6733 tokens vs. 1376 in ours) and exhibits strong early-answer locality: truncating documents to the first 512 tokens still preserves the ground-truth answer in at least one gold document for 86.5% of queries. This property favors dense retrievers that primarily represent the document prefix. In contrast, our documents are longer and evidence is more dispersed, reducing the effectiveness of limited-window dense encoding.

Asymmetric text coverage can favor dense retrievers under early-answer locality. BrowseComp-Plus encodes only the first 4096 tokens for Qwen3-Embed-8B, while BM25 indexes the full document. When answers are front-loaded, prefix-only dense encoding can act as an implicit denoising mechanism, whereas full-text BM25 may incur additional lexical noise. This asymmetric coverage therefore biases the comparison toward dense retrievers. In our experiments, we allow up to 32,000 tokens per document, so dense retrievers do not benefit from short-prefix encoding.

Agent strength and query decomposition modulate retriever sensitivity. BrowseComp-Plus further suggests that stronger agent models (e.g., GPT-5 and o3) are less sensitive to retriever choice, as the gap between BM25 and Qwen3-Embed-8B narrows compared to weaker models (e.g., Qwen3-32B and Gemini-2.5 Flash). Moreover, BrowseComp-Plus adopts a ReAct-style framework (Yao et al., 2023) that produces natural-language sub-queries, while our setup (GPT-5 series, Gemini-2.5 series, and DR-Tulu) uses more keyword-oriented decomposition. This difference in query formulation can shift the relative advantage between lexical and dense retrievers.

### A.5 Further Experiments with SearchR1-32B

As a supplement to our main experiments with DR Tulu, we evaluate another open-source research agent, SearchR1-32B (Jin et al., 2025). Table 5 summarizes corpus-search performance across domains.

SearchR1-32B exhibits near single-shot retrieval. SearchR1-32B issues only 1.1–1.2 searches per question, leaving limited room for iterative query refinement. Consequently, end-to-end performance is primarily determined by the initial query formulation and the base retriever.

Natural-language querying does not obviate lexical matching. Although SearchR1-32B produces natural-language queries rather than keyword-style decompositions, BM25 remains markedly stronger on short-form questions. On open-ended questions, BM25 and gt-Qwen are closer in performance, while ReasonIR remains substantially worse, consistent with previous findings. Importantly, the average number of references is similar across retrievers, suggesting that the observed differences are driven by retriever quality rather than references counts.

### A.6 Prompt Templates
```

### --- Page 0015 ---

```markdown
| Method       | Com. Sci. | Healthcare | Humanities | Nat. Sci. | Avg. Searches | Avg. Refs | Avg. Perf. |
|--------------|-----------|------------|------------|-----------|---------------|-----------|------------|
| **CORPUS SEARCH (SEARCHR1-32B)** |           |            |            |           |               |           |            |
| Short-Form Questions (Exact Match) |           |            |            |           |               |           |            |
| BM25 $k=5$  | 19.57     | 45.26      | 34.13      | 40.91     | 1.21          | 5.7       | 34.97      |
| gte-Qwen $k=5$ | 4.29   | 23.44      | 26.23      | 16.18     | 1.24          | 5.7       | 17.54      |
| ReasonIR $k=5$ | 3.79   | 20.90      | 15.70      | 11.28     | 1.21          | 5.5       | 12.92      |
| Open-Ended Questions (Weighted Recall) |           |            |            |           |               |           |            |
| BM25 $k=5$  | 6.80      | 12.58      | 13.52      | 10.20     | 1.16          | 5.6       | 10.77      |
| gte-Qwen $k=5$ | 7.57   | 11.78      | 12.73      | 7.48      | 1.10          | 5.2       | 9.89       |
| ReasonIR $k=5$ | 4.65   | 5.52       | 8.19       | 2.93      | 1.11          | 5.4       | 5.32       |

Table 5: Corpus-search results with SearchR1-32B.

---

### Academic Paper Keyword Generation Prompt

Based on the following academic paper content, generate exactly 8 keywords that best represent the main topics, methods, or contributions of this paper.  
Content: `[content]{20000}`  
Return ONLY the 8 keywords separated by commas, nothing else. Example format: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6, keyword7, keyword8

Figure 13: Prompt of Academic Paper Keyword Generation

---

### Prompt for Analyzing Shared Reference Functions

**System:** You are a research assistant helping to analyze academic citations. Provide concise, accurate summaries.  
**User:**  
Shared Citation Paper: `"{shared_title}" by {shared_authors}`  
Paper 1: Target Paper  
- Citing Contexts:  
  1. `{context_1}`  
  2. `{context_2}`  
  ...  
- Intents: `{intents}`  
Paper 2 (shared `{shared_count}` papers with Target Paper): `"{cited_paper_title}"`  
- Citing Contexts:  
  1. `{context_1}`  
  2. `{context_2}`  
  ...  
- Intents: `{intents}`  
**Task:** Summarize in ONE sentence what role this shared citation paper played for both papers. Focus on the specific contributions or methods it provided to each paper.

Figure 14: Prompt for Analyzing the Functional Role of Shared References Between Two Papers
```

### --- Page 0016 ---

```markdown
# Prompt for Generating Comprehensive Summaries of Paper Relationships

**System:** You are a research assistant analyzing paper relationships. Write concise, specific summaries:
- **Maximum 2 sentences**
- **Mention ONLY 2 most important shared citations by title**
- **Use *This paper* for Paper 1, full title for Paper 2**
- **Be specific about what the citations are used for**
- **Avoid generic phrases and long lists**

**User:**
- Paper 1 (This Paper): “{paper1_title}”
- Paper 2: “{cited_paper_title}”
- These two papers share {shared_count} common citations.

**Analysis of shared citations:**
1. “{citation_title}” by {citation_authors}
   - Paper 1 uses it for: {intents}
   - Paper 2 uses it for: {intents}
   - How both papers use it: {summary}
2. ... (additional shared citations)

**Task:** Write a concise summary (2 sentences maximum) that:
1. Uses *This paper* to refer to Paper 1
2. Uses the full title when referring to Paper 2
3. Selects and mentions 2 important shared citations by title (in quotes). Avoid famous papers mentioned only as convention, e.g., “Attention is All You Need”
4. Briefly explains what commonality these key citations reveal
5. Mentions that they share {shared_count} citations in total
6. Be specific and concrete—avoid generic statements

**Format:**
- First sentence: Introduce the 2 key shared citations and their specific role
- Second sentence: State “They share {N} citations in total and are specifically related by {brief commonality}.”

**Additional constraints:** No parentheses or brackets; keep each part simple and direct.

---

![Prompt for Generating Comprehensive Summaries of Shared References Relationships Between Paper Pairs](assets/page_0016_img_1.png)

# Prompt for Selecting the Most Characteristic Summary

**System:** You are a helpful assistant that selects the most characteristic summary. Return only a single integer.

**User:**
- You are given a list of paper summaries. Each summary describes the relationship between a source paper and a cited paper.

**Your task:** Select exactly 1 summary that is MOST characteristic and informative.
- Choose the summary that provides the most specific, concrete technical details
- Prefer summaries that mention distinctive methods, techniques, or research approaches
- Avoid generic or vague descriptions

**Summaries:**
- [0] {summary_0}
- [1] {summary_1}
- [2] {summary_2}
- ... (additional summaries)

Return ONLY a single integer index (e.g., 0, 1, 2, etc.)  
No explanation needed, just the number.

---

**Query Construction:** After selecting index i, the final query is constructed as:  
Find me the paper that have the following characteristics: {selected_summary}
```



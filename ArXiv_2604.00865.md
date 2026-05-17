# ArXiv 2604.00865

### --- Page 0001 ---

```markdown
# Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation

**Shuguang Jiao**  
Harbin Institute of Technology  
Shenzhen, China  
245151028@stu.hit.edu.cn  

**Chengkai Huang**  
Macquarie University and UNSW  
Sydney, Australia  
chengkai.huang1@unsw.edu.au  

**Xuan Wang**  
Harbin Institute of Technology  
Shenzhen, China  
xuanwgan@cs.hitsz.edu.cn  

**Yifan Li**  
Harbin Institute of Technology  
Shenzhen, China  
liyifan@cs.hitsz.edu.cn  

**Shuhan Qi**  
Harbin Institute of Technology  
Shenzhen, China  
shuhani@cs.hitsz.edu.cn  

**Lina Yao**  
UNSW and CSIRO's Data61  
Sydney, Australia  
lina.yao@unsw.edu.au  

---

## Abstract

Agentic Retrieval-Augmented Generation (Agentic RAG) has become a widely adopted paradigm for multi-hop question answering and complex knowledge reasoning, where retrieval and reasoning are interleaved at inference time. As reasoning trajectories grow longer, failures become increasingly common. Existing approaches typically address such failures by either stopping at diagnostic analysis or rerunning the entire retrieval–reasoning pipeline, which leads to substantial computational overhead and redundant reasoning. In this paper, we propose Doctor-RAG (DR-RAG), a unified diagnose-and-repair framework that corrects failures in Agentic RAG through explicit error localization and prefix reuse, enabling minimal-cost intervention. DR-RAG decomposes failure handling into two consecutive stages: (i) Trajectory-level failure diagnosis and localization, which attributes errors to a coverage-gated taxonomy and identifies the earliest failure point in the reasoning trajectory; and (ii) Tool-conditioned local repair, which intervenes only at the diagnosed failure point while maximally reusing validated reasoning prefixes and retrieved evidence. By explicitly separating error attribution from correction, DR-RAG enables precise error localization, thereby avoiding expensive full-pipeline reruns and enabling targeted, efficient repair. We evaluate DR-RAG across three multi-hop question answering benchmarks, multiple agentic RAG baselines, and different backbone models. Experimental results demonstrate that DR-RAG substantially improves answer accuracy while significantly reducing token consumption compared to rerun-based repair strategies.

## CCS Concepts
- Information systems → Information retrieval.

## Keywords
Retrieval-Augmented Generation, Large Language Model, Agentic RAG

## 1 Introduction

In recent years, Agentic Retrieval-Augmented Generation (Agentic RAG) has emerged as a dominant paradigm for multi-hop question answering and complex knowledge reasoning [16, 21, 29, 36, 45]. By dynamically generating search queries, invoking retrievers, and performing multi-step reasoning during inference, Agentic RAG enables models to progressively plan and solve complex problems, substantially improving coverage and generalization over static RAG systems [15, 32, 34, 35, 46, 48].

However, as reasoning chains become longer and interaction rounds increase, failures in Agentic RAG become more frequent and difficult to fully avoid [1, 25, 33]. Errors in multi-step reasoning may arise at arbitrary stages and accumulate over time, simply improving single-step reasoning or retrieval performance is often insufficient to ensure stable system behavior [2, 8, 41]. As a result, the challenges faced by Agentic RAG extend beyond whether a correct answer can be generated, and increasingly concern:

**How can an Agentic RAG system localize the cause of a failure and repair it with minimal recomputation once the failure occurs?**

Existing studies have analyzed and modeled failures in RAG systems from different perspectives [1, 3, 11, 25, 33]. One line of work focuses on systematically categorizing and diagnosing errors in RAG outputs [33, 47], but does not further address automatic repair or performance improvement after failures occur. Another line of work incorporates error analysis into existing workflows [8], triggering subsequent repair based on diagnostic feedback after a complete reasoning execution; however, such repair typically relies on a replanning–rerun strategy, discarding the original trajectory and re-executing the retrieval–reasoning pipeline. Although

![The statistics of different error types in Agentic RAG under the ReAct baseline on HotpotQA (500 samples).](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
# Conference acronym 'XX', June 03–05, 2018, Woodstock, NY
Shuguang Jiao et al.

these two approaches differ in how error analysis is conducted and utilized, they either stop at diagnostic evaluation or depend on expensive full-pipeline re-execution, and do not reuse validated per-fixes and previously retrieved evidence, and thus recompute most steps during repair. Consequently, a fundamental question remains unanswered: once a failure has occurred, how can an Agentic RAG system repair it at minimal cost? Figure 1 illustrates the distribution of failure types observed in agentic RAG under the ReAct baseline on PhotoQA. The results show that failures arise from diverse sources, rather than a single dominant cause. This diversity suggests that failures are systematic and multi-faceted, making unconditional reruns or monolithic repair strategies inherently inefficient.

To address this problem, we propose Doctor-RAG (DR-RAG), a unified diagnosis-and-repair framework for Agentic RAG, designed to enable efficient failure correction through explicit error attribution and minimal-cost intervention. Unlike existing approaches that directly retry or replan the entire retrieval-reasoning pipeline upon failure, DR-RAG explicitly models failure correction as two consecutive stages: error diagnosis and conditional repair.

Specifically, DR-RAG consists of two tightly coupled core components. (1) Taxonomy-Constrained Error Diagnosis and Localization. When a reasoning failure occurs, DR-RAG treats the agentic RAG reasoning trajectory as the object of diagnosis. Leveraging evidence coverage signals, the diagnosis module performs coarse-to-fine failure attribution, mapping continuous reasoning failures to a set of executable error types. DR-RAG further localizes the earliest failure point at the trajectory level, identifying the first reasoning event that deviates from a valid solution path, thereby providing structured signals required for minimal intervention in subsequent repair. (2) Tool-Conditioned Local Repair. Given the diagnosed error type and failure localization, DR-RAG selects a corresponding tool-conditioned repair operator rather than regenerating a complete solution plan. Each repair operator intervenes only at the identified failure point while maximizing reuse of previously validated reasoning prefixes and retrieved evidence. Concretely, repair operations include lightweight answer rewriting for former errors, evidence-preserving re-reasoning under full coverage, query rewriting and retrieval enhancement for retrieval failures, and planning-based repair for reasoning-induced retrieval errors. By constraining repair to the necessary components, DR-RAG significantly reduces the cost of retrieval and reasoning repair while preserving coherence across repaired trajectories.

Our main contributions are summarized as follows:

- We are the first to formally formulate failure handling in Agentic RAG at the system level, modeling corrections as diagnosis-guided minimal-cost intervention rather than full-pipeline retry.
- We propose a coverage-gated error taxonomy and a trajectory-based diagnosis method that explicitly distinguishes retrieval and reasoning failures while localizing the earliest failure point.
- We design a set of tool-conditioned repair operators that enable local repair with prefix reuse, reducing correction cost.

- Experiments across multiple datasets, baselines, and model scales show that DR-RAG substantially improves accuracy while greatly reducing retrieval calls and reasoning token usage.

## 2 Related Work

### 2.1 Retrieval Augmented Generation

Large Language Models (LLMs) are prone to hallucinations and lack access to up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses these limitations by integrating external knowledge sources after inference time, improving factual accuracy without requiring model retraining [12, 23]. Early work, exemplified by the original RAG framework [26], established the retrieval-enhanced generation architecture that underpins retrieval-enhanced generation. Building on this paradigm, subsequent studies progressively extended RAG with more advanced reasoning and retrieval mechanisms, including Chain-of-Thought [45], Tree-of-Thought reasoning [20, 44], and adaptive retrieval strategies [18]. These advances enable effective multi-step reasoning in RAG systems.

### 2.2 Agentic RAG

In recent years, Agentic RAG (e.g., ReAct [45], Search-RL [27], Search-RAG [21]) has formalized reasoning and tool invocation (retrieval) [7], memory [28], and summarization [5] as a unified sequential decision-making process. Through leveraging [6, 9], these approaches enable models to autonomously select search actions [4, 10, 39] typically optimizes reasoning and retrieval policies using outcome-level rewards or preference signals, thereby improving performance on multi-step search and complex reasoning tasks.

More recently, several studies have investigated robustness and failure diagnostics in RAG and agentic RAG systems. RAGChecker [33] is a claim-level entailment-based diagnostic framework that evaluates RAG system performance but does not address how to repair or improve the system after errors are identified. RAG-Critic [8] responds to detected errors by replanning a new reasoning and retrieval trajectory, without localizing the earliest error or reusing validated prefix trajectories. As a result, failed trajectories are still treated as invisible errors, leading to redundant computation and limited error correction efficiency. In contrast, how to repair a failed agentic RAG trajectory in a fine-grained and cost-efficient manner remains underexplored [14, 17, 31]. Prior work largely lacks explicit error localization within trajectories and does not consider reusing validated reasoning prefixes or retrieved evidence. This gap motivates our work, which treats agentic RAG failures as structured objects that can be diagnosed, localized, and selectively repaired rather than blindly rerun.

## 3 Agentic RAG Task Formulation

We consider the agentic retrieval-augmented generation (RAG) setting for multi-step question answering, where a language model solves a query through an explicit sequence of interleaved reasoning, search, and information actions.

Given a question $x \in X$, the system executes a sequence of actions, each corresponding to one of the following modules:
```

### --- Page 0003 ---

```markdown
# Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation
Conference acronym 'XX', June 03–05, 2018, Woodstock, NY

## Table 1: Coverage-gated error taxonomy for agentic RAG trajectories with localized failure attribution.

| Coverage | Error Type        | Localized Component |
|----------|-------------------|---------------------|
| Cov(y) = 1 | Format Error      | answer              |
|            | Reasoning Error   | reasoning           |
| Cov(y) < 1 | Format Error      | answer              |
|            | Retriever Error    | information         |
|            | Search Error      | search              |

The goal of the diagnosis module is to determine (i) why the failure trajectory fails, in terms of the error category; (ii) when the failure first occurs along the trajectory at the action level. In contrast to RAG-Critic, our diagnosis process does not use the ground-truth answer as auxiliary supervision and relies solely on the generated trajectory. Formally, we define a diagnosis operator:

$$
D_\mathcal{g} : (x, y) \mapsto (c, k^h),
$$

where $c \in \mathcal{C}$ denotes an error category from a predefined taxonomy, and $k^h$ is an optional location index indicating the earliest action in the trajectory that is identified as erroneous.

### 4 Methodology

#### 4.1 DR-RAG Overview

Figure 2 presents the overall workflow of DR-RAG, a diagnose-and-repair framework for agentic RAG. Throughout this section, trajectories are defined as the sequence of actions taken by the agent in response to an atomic reasoning or retrieval action. Unlike prior approaches that rerun the entire reasoning-retrieval pipeline, DR-RAG treats a failed trajectory as a reusable object and corrects it through targeted and localized intervention.

We focus on repairable agentic RAG failures; Dataset Noise (e.g., ambiguous questions) are considered non-actionable and excluded from the diagnosis and repair pipeline. The framework then proceeds in two sequential stages: (1) taxonomy-constrained error diagnosis and localization, and (2) tool-conditioned local repair.

In the diagnosis stage, DR-RAG first assesses the sufficiency of the retrieved documents and then performs coverage-constrained error classification and localization based on observable trajectory signals, which serves as a gating signal that constrains the space of plausible failure causes. In the repair stage, DR-RAG selects a specialized repair operator based on the diagnosis result and intervenes only at the identified failure point, while efficiently reusing all previously validated prefix steps.

#### 4.2 Diagnosis Module: Taxonomy-Constrained Error Diagnosis and Localization

This section presents a unified diagnosis module that jointly defines (i) a taxonomy of agentic RAG failure modes and (ii) a coverage-guided procedure for error classification and localization. The module analyzes a failed trajectory without modifying it and outputs a structured diagnosis that directly confirms downward repair.

Given an input question $x$, the agentic RAG system generates an action-level trajectory $y = \{p(0), \ldots, p(k), a_{pred}\}$, where $a_{pred}$ is treated as the terminal action. We focus on failure cases where the predicted answer is incorrect, i.e., $EM_{qa}(a_{pred}) = 0$.
```

### --- Page 0004 ---

```markdown
# Conference acronym 'XX', June 03–05, 2018, Woodstock, NY

## Figure 2: Overview of DR-RAG. Given a failed agent RAG trajectory, the framework performs taxonomy-constrained diagnosis and localization, followed by tool-conditioned local repair that intervenes at the failure point while reusing the trajectory.

### (a) Taxonomy-Constrained Error Diagnosis and Localization

![Taxonomy-Constrained Error Diagnosis and Localization](assets/page_0004_img_1.png)

- **Error Candidates**
  - Format Error
  - Reasoning Error
  - Retriever Error
  - Search Error

- **Error Taxonomy Constraints**
  - $c$: Error Type
  - $k$: Error Location

### (b) Tool-Conditioned Local Repair

![Tool-Conditioned Local Repair](assets/page_0004_img_2.png)

- **Repair Router**
  - Chose operator from (Format, Reasoning, Retriever, Search)
  - Trajectory function at $t$: $Trajectory(t_0, \ldots, t_k)$
  
- **Repair**
  - Reasoning with LLM
  - Searching with Retriever
  - Rewriting with LLM

### The admissible error categories are therefore limited to:

- **Format-Invalid Answer (Format Error)**, where the predicted answer is semantically correct but violates strict output or EM format constraints, and is localized to the answer prediction step;

- **Reasoning Logic Error (Reasoning Error)**, where the model performs incorrect reasoning over sufficient evidence, leading to an incorrect answer. This error type admits a well-defined failure point and requires trajectory localization.

Partial Evidence Coverage ($Cov(y) < 1$). When the retrieved documents are insufficient, failures may arise either from retrieval itself or from earlier reasoning steps that misdirect retrieval. In this case, the admissible error categories include:

- **Format-Invalid Answer (Format Error)**, where the predicted answer is semantically correct but violates strict output or EM format constraints, and is localized to the answer prediction step, regardless of evidence coverage;

- **Retriever Failure (Retriever Error)**, where valid and well-formed queries fail to retrieve relevant information, indicating a limitation of the retrieval component; and

- **Reasoning-Induced Missing Retrieval (Search Error)**, where incorrect reasoning leads to invalid or incomplete queries, causing relevant evidence to be missed.

This coverage-gated taxonomy ensures mutually exclusive and trajectory-consistent error classification, and directly determines when and where local repair is applied.

#### 4.2.3 Taxonomy-Constrained Error Localization

Within the coverage-constrained candidate set, the diagnosis module performs fine-grained error classification based on observable trajectory signals, including reasoning steps, issued queries, retrieved documents, and the final predicted answer. No additional retrieval is triggered, and no ground-truth answer or oracle information is used during diagnosis. For error types that admit partial trajectory reuse, the module further performs error localization at the action level. Given the action-level trajectory $y = (y_1, \ldots, y_k)$ defined above, the diagnosis module performs error localization at the action level.

The error localization index is defined as:

$$
k^* = \min\{k | \rho(k) \text{ is classified as erroneous}\}, \tag{3}
$$

where the classification is performed under the coverage-constrained error taxonomy. Intuitively, $k^*$ identifies the earliest action that is
```

### --- Page 0005 ---

```markdown
# Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation
Conference acronym 'XX', June 03-05, 2018, Woodstock, NY

## 4.3 Repair Module: Tool-Conditioned Local Repair

Given the diagnosis output $(c, k')$ from the diagnosis module, the repair stage performs targeted intervention on the failed agentic RAG trajectory. After restarting the entire reasoning-retrieval process, DR-RAG applies a tool-conditioned repair operator selected according to the diagnosed error type. This design enables partial reuse of conditionally valid trajectory prefixes and avoids redundant computation.

Formally, we define a set of repair operators: $F = \{f_1, \ldots, f_n\}$, where each operator $f_i$ is specialized for repairing a specific error category. Operator $f$ is defined as $f: S \to C$ where $S: C \to \mathfrak{R}$ maps an error type to its repair operator.

### 4.3.1 Repair under Full Evidence Coverage ($Cov(y) = 1$)

When the diagnosis module determines that the retrieved documents in the trajectory are sufficient, repair occurs by correcting reasoning or answer generation, without issuing additional retrieval actions.

### Format-Invalid Answer Repair. 

As in the full-coverage case, formatting invalid answer errors under partial coverage are handled by rewriting the final answer. No reasoning or retrieval actions are altered.

### Retriever Failure Repair. 

Retriever failure errors arise when well-formed queries fail to retrieve sufficient information. We truncate the trajectory at $k'$ and retain the prefix $y_{prefix}$. All retrieval queries issued prior to the failure are collected as $Q = \{q_1, \ldots, q_{k'-1}\}$. These queries are rewritten to improve recall while preserving their original intent. In addition, the retriever is configured with an increased top-$k$ to return a larger candidate document set. The model then re-generates the final answer based on the retained prefix and the augmented retrieval results.

### Reasoning-Induced Missing Retrieval Repair. 

Since failure originates from an incorrect reasoning action that misguides retrieval, both reasoning and retrieval steps beyond $k'$ are re-generated. After truncating the trajectory at $k'$, the model is prompted to generate a high-level solution plan conditioned on the original question and the retained prefix. This plan decomposes the task into a sequence of reasoning and retrieval steps, which are then executed to produce a new trajectory suffix. During retriever failure repair, both reasoning and retrieval paths are allowed to diverge from the original trajectory beyond the prefix.

### 4.3.3 Summary of Repair Operators. 

Across all error categories, repair operators share a common structure: they preserve conditionally valid trajectory prefixes, intervene only at the diagnosed failure point $k'$, and restrict modifications to the main components required by the error type. This tool-conditioned design forms a coherent interface with the diagnosis module and enables efficient, localized repair for agentic RAG trajectories.

## 5 Experiments

### 5.1 Research Questions

In this section, we address the following Research Questions (RQs):

- **RQ1:** Does DR-RAG improve overall RAG performance while reducing token consumption compared to full rerun baselines?
- **RQ2:** Do error taxonomy and error localization each play an essential role in DR-RAG?
- **RQ3:** What mechanisms enable DR-RAG to achieve lower token usage than retry-based repair methods?
- **RQ4:** How accurate are error diagnosis and localization across different error types?
- **RQ5:** How does repair success vary across different error types?
- **RQ6:** What is the maximum achievable repair success rate under oracle diagnosis and localization?

### 5.2 Experiment Settings

#### 5.2.1 Datasets and Metrics. 

We conduct experiments on three multi-hop question answering benchmarks: HotpotQA, 2Wiki, and MuSiQ. HotpotQA [43] is a large-scale benchmark for multi-hop reasoning over Wikipedia, 2WikiMultihopQA (2Wiki) [13] focuses
```

### --- Page 0006 ---

```markdown
| Baseline   | Method       | Tokens | Cost ($) | Repair | EM   | AEM  | HotpotQA | 2Wiki | MusiQue |
|------------|--------------|--------|----------|--------|------|------|----------|-------|---------|
| ReAct [45] | Step-wise    | 8,113  | 15.7     | 17.4   | 19.0 | 18.6 | 11.6     | 15.0  | 14.9    |
|            | RAG-Critic [8] | 9,279  | 11.5     | 12.4   | 14.4 | 18.8 | 9.4      | 14.0  | 13.9    |
|            | DR-RAG       | 5,610  | 17.9     | 18.6   | 13.4 | 14.5 | 8.6      | 8.1   | 8.5     |
| Search-01 [27] | Step-wise | 6,809  | 7.3      | 6.9    | 7.1  | 3.4  | 5.9      | 6.0   | 3.6     |
|            | RAG-Critic [8] | 7,263  | 7.7      | 6.6    | 6.8  | 5.8  | 5.4      | 7.6   | 3.4     |
|            | DR-RAG       | 7,963  | 9.7      | 6.2    | 6.6  | 7.0  | 6.3      | 7.6   | 8.5     |
| Search-R1 [21] | Step-wise | 6,632  | 17.2     | 12.3   | 13.0 | 15.6 | 15.9     | 16.2  | 8.2     |
|            | RAG-Critic [8] | 7,057  | 7.3      | 8.4    | 5.7  | 5.2  | 8.6      | 10.9  | 1.4     |
|            | DR-RAG       | 1,099  | 11.9     | 7.4    | 8.0  | 10.2 | 9.3      | 10.1  | 7.8     |
| Search-R1 [21] | Step-wise | 5,395  | 20.6     | 12.4   | 13.9 | 18.8 | 18.0     | 18.5  | 10.5    |
|            | RAG-Critic [8] | 12,065 | 5.6      | 6.0    | 5.0  | 4.7  | 2.0      | 3.6   | 3.5     |
|            | DR-RAG       | 6,609  | 13.9     | 12.8   | 10.6 | 6.6  | 6.7      | 4.7   | 1.7     |
| Search-R1 [21] | Step-wise | 13,162 | 5.3      | 6.0    | 5.0  | 4.7  | 3.4      | 5.0   | 2.5     |
|            | RAG-Critic [8] | 5,519  | 2.6      | 2.0    | 0.7  | 2.4  | 1.5      | 1.7   | 1.0     |
|            | DR-RAG       | 7,268  | 12.2     | 9.4    | 9.0  | 2.8  | 3.5      | 8.5   | 3.9     |

![Detailed description of the chart](assets/page_0006_img_1.png)
```

### --- Page 0007 ---

```markdown
# Doctor-RAG: Failure-Aware Repair for Agentit Retrieval-Augmented Generation

## 5.2.3 Implementation Details
Following prior work on agentic RAG systems [19], we conduct experiments using three backbone language models: 2B-3B [42], LLaMA-3.1-3B-Instruct [37], and Qwen-3B [42]. All baselines and diagnosis-repair strategies are evaluated under identical backbone models to ensure fair comparison. Following Karpukhin et al. [23], we use the full Wikipedia 2018 dump as the retrieval corpus. Our main experiments follow the retrieval setup of Jiang et al. [19], employing FAISS [22] for indexing, DR-large-en-v1.5 [40] as retriever, and retrieving top 5 documents. All methods share the same retrieval settings. All methods are evaluated in a post-hoc manner on failed trajectories produced by the original agentic RAG baselines, without modifying their underlying retrieval or reasoning components. For correctness, we employ the evaluation metrics defined in LLM-based sufficiency judge with the same backbone model as the evaluated methods, which accounts to golden evidence. Notably, the diagnostic module of DR-RAG does not use ground-truth answers and operates solely on the generated trajectories and recoveries. All reported token costs for DR-RAG include tokens consumed during both diagnosis and repair. Experiments are conducted on four NVIDIA RTX 7900 GPUs, using vLLM for inference [24].

## 5.3 Main Results (RQ1)
Table 2 reports the main experimental results across three multi-hop question answering benchmarks. Across all baselines and datasets, DR-RAG consistently achieves substantial performance improvements while maintaining lower token consumption. Compared with full rerun and step-wise retry baselines, DR-RAG yields significant gains in EM, F1, and ROUGE-L. For instance, under the Qwen-3B backbone with the ReAct baseline, DR-RAG improves EM by +25.8% on HotpotQA, +19.6% on 2Wiki, and +10.0% on MusIQue, while reducing token usage by 26.8% relative to the rerun strategy. Across all settings, DR-RAG consistently incurs the lowest token cost, validating that localized repair is more efficient than full trajectory re-execution.

DR-RAG achieves simultaneous gains in effectiveness and efficiency by adopting a failure-attribution-driven local repair paradigm. Conventional rerun or step-wise methods implicitly treat the entire trajectory as unreliable after a failure, leading to redundant regeneration of reasoning and retrieval steps. In contrast, DR-RAG first diagnoses where the failure originates from formatting, reasoning, or retrieval, and then localizes the earliest point of failure, thereby avoiding recomputation of previously verified reasoning.

## 5.4 Ablation Study (RQ2)
Table 3: Ablation study of DR-RAG. We evaluate error taxonomy and error localization across three benchmarks, showing gains in accuracy and efficiency.

| Baseline  | HotpotQA | 2Wiki | MusIQue |
|-----------|----------|-------|---------|
|           | Tokens   | EM    | Tokens  | EM    | Tokens  | EM    |
| DR-RAG    | 4,483    | 17.7  | 5,697  | 17.7 | 5,464  | 10.0  |
| w/o Taxonomy | 4,840 | 13.9  | 5,794  | 13.7 | 5,507  | 6.9   |
| w/o Localization | 6,012 | 14.3 | 7,193  | 14.9 | 6,585  | 8.6   |

These results confirm that taxonomy and localization are complementary: taxonomy enables effective repair, while localization ensures minimal intervention.

## 5.5 Token and Time Efficiency (RQ3)
We evaluate the computational efficiency of DR-RAG in terms of token consumption and end-to-end inference time. As shown in Table 4, DR-RAG consistently achieves the lowest cost across all datasets and agent frameworks under both metrics. Full reruns strategies regenerate entire reasoning trajectories after a failure, resulting in substantial overhead in token usage and runtime. Step-wise retry reduces some redundancy but still repeats downstream reasoning and retrieval beyond the failure point, as it does not distinguish different error types and regenerates all subsequent steps once a failure is detected. In contrast, DR-RAG performs...
```

### --- Page 0008 ---

```markdown
# Conference acronym 'XX', June 03–05, 2018, Woodstock, NY
Shuguang Jiao et al.

## Table 4: Efficiency comparison of different repair strategies.
Tok. denotes the average token consumption, and Time represents the total time cost (in seconds). The best results (minimum values) are highlighted in bold.

| Baseline Method | HotpotQA | 2Wiki  | MusiQue |
|------------------|----------|--------|---------|
|                  | Tok.     | Time   | Tok.    | Time   |
| ReAct            | Rerun    | 8,714   | 113,934 | 82,920 | 927     |
|                  | Step-wise| 11,524  | 1,031   | 9,822  | 11,315 | 990    |
|                  | RAG-Critic| 14,196 | 5,036   | 16,605 | 10,726 | 1,111  |
| DR-RAG           | 4,514    | 279,583 | 5,687   | 199    | 1,790  |
| Search-o1        | Rerun    | 7,844   | 3,839   | 16,837 | 817    | 1,871  |
|                  | Step-wise| 6,861   | 631     | 2,862  | 1,634  |
|                  | RAG-Critic| 17,910 | 475,850 | 5,795  | 9,711  | 1,790  |
| DR-RAG           | 4,340    | 274,629 | 5,384   | 4,963  | 220    |
| Search-R1        | Rerun    | 6,457   | 2,377   | 477    | 3,930  |
|                  | Step-wise| 5,848   | 3,766   | 2,629  | 734    |
|                  | RAG-Critic| 15,203 | 511,959 | 19,371 | 7,897  | 3,852  |
| DR-RAG           | 4,594    | 270,520 | 5,820   | 5,743  | 366    |

### React | Search-o1 | Search-R1 | Avg.

![Confusion matrix for automated error diagnosis.](assets/page_0008_img_1.png)

### Figure 4: Automated error diagnosis and repair analysis (a) Confusion matrix aggregated across multiple datasets (2Wiki, HotpotQA, and MusiQue) and baselines (ReAct, Search-o1, and Search-R1), showing high consistency between the diagnostic model and human annotations. (b) Comparison of average repair rates across error types on HotpotQA, aggregating results from three baselines under repair strategies.

## Table 5: Repair success rates stratified by error type and retrieval coverage.
Results are broken down by failure category (format, reasoning, search, and retriever errors) and reported separately for full coverage (Cov-1) and partial coverage (Cov-1) across datasets and baselines.

| Baseline Method | HotpotQA | 2Wiki  | MusiQue |
|------------------|----------|--------|---------|
|                  | Cov-1    | Format Err | Search Err | Retr. Err |
| ReAct            | 43.00    | 60.71   | 27.22     | 39.77     | 15.00    |
|                  | 2Wiki    | 30.32   | 52.63     | 27.27     | 25.61    | 20.50    |
|                  | MusiQue  | 61.54   | 100.00    | 19.27     | 13.00    |
| Search-o1        | HotpotQA | 40.00   | 34.78     | 19.15     | 28.75    | 33.33    |
|                  | 2Wiki    | 21.85   | 16.48     | 21.13     | 10.00    |
|                  | MusiQue  | 43.75   | 75.00     | 0.972    | 14.56    | 12.00    |
| Search-R1        | HotpotQA | 36.36   | 21.88     | 22.27     | 14.29    |
|                  | 2Wiki    | 27.27   | 17.24     | 29.00     | 23.45    |
|                  | MusiQue  | 5.00    | 8.64      | 6.10     | 3.57    |

### 5.6 Error Diagnosis Reliability (RQ4)
We evaluate the reliability of the automated diagnosis module in DR-RAG by comparing its predictions with human annotations across datasets and agent frameworks. Figure 3 reports diagnosis accuracy stratified by evidence coverage. Across all baselines and datasets, diagnosis accuracy consistently falls within the 50–70% range. On average, DR-RAG achieves an accuracy of 61.6% under full coverage (Cov-1) and 60.3% under partial coverage (Cov-1), indicating stable diagnostic performance regardless of evidence completeness. This consistency suggests that the diagnosis module generalizes well across agent architectures and provides sufficiently reliable signals to guide localized repair.

### 5.7 Repair Performance by Error Type (RQ5)
We first analyze repair performance by error type using Figure 4(b), which compares rerun, step-wise repair, and DR-RAG on HotpotQA.
```

### --- Page 0009 ---

```markdown
# Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation
## Conference acronym 'XX', June 03-05, 2018, Woodstock, NY

### Question: Which of these is further south in China, the Pulandian District or Kaiyuan, Liaoning?

![Detailed description of the chart](assets/page_0009_img_1.png)

Figure 5: Case study of a reasoning logic error under full evidence coverage. DR-RAG localizes the earliest faulty reasoning step and repairs the error by reusing retrieved evidence, avoiding unnecessary retrieval and full trajectory regeneration.

(a) DR-RAG achieves the highest repair rates across all five error types, with the largest gains on reasoning and format errors under full coverage (Cov-1). These errors are typically easier to correct than retrieval errors, enabling targeted correction after failing retrieval. In contrast, rerun and partial level repair do not model error courses, limiting their improvements. 

(b) For retrieval errors and search errors, DR-RAG still outperforms the baselines, but the overall repair rates are lower. These two error types exhibit different limiting factors. For retrieval errors, repair is constrained by missing or incomplete evidence, while search errors typically arise in more complex multi-hop questions that require harder reasoning and planning beyond local correction.

Table 5 reports repair success rates under different error types and retrieval coverage conditions. Overall, repair and partial coverage (Cov-1) is consistently more difficult than full coverage (Cov-1). When retrieval coverage is incomplete, many errors arise from complex reasoning and planning failures, which reflect the upper bound of the model’s ability to analyze and solve hard tasks. In such cases, even with a correct diagnosis, localized repair becomes less effective. We also observe clear dataset-level differences. Across baselines and error types, repair rates on MusiQue are generally lower than those on HotpotQA and 2Wiki. This is expected, as MusiQue contains more complex multi-hop questions and longer reasoning chains, which place higher demands on the model's reasoning capacity and limit the effectiveness of repair.

### 5.8 Oracle-based Upper Bound Analysis (RQ6)

Finally, we estimate the upper bound of DR-RAG by assuming oracle access to error types and failure locations. As shown in Table 6, oracle-guided DR-RAG achieves higher repair rates and downstream QA performance across all datasets and baselines.

| Baseline Dataset | Tokens | Repair Rate | MRR @1 | AF1 @1 |
|------------------|--------|-------------|--------|--------|
| HotpotQA         | 4,520  | 39.4        | 27.8   | 26.9   |
| 2Wiki            | 5,034  | 29.2        | 21.0   | 20.6   |
| MusiQue          | 5,681  | 21.1        | 11.2   | 11.3   |
| HotpotQA         | 4,459  | 29.0        | 15.4   | 13.6   |
| 2Wiki            | 5,825  | 24.6        | 16.6   | 16.4   |
| MusiQue          | 5,482  | 12.6        | 12.2   | 12.1   |
| Search-R1       | 4,571  | 21.8        | 13.6   | 14.4   |
| 2Wiki            | 6,231  | 29.0        | 18.3   | 18.9   |
| MusiQue          | 5,965  | 11.3        | 9.8    | 9.9    |

5.9 Case Study
We present a representative example from HotpotQA to illustrate how DR-RAG repairs reasoning errors under full evidence coverage. The question asks: "Which of these is further south in China, the Pulandian District or Kaiyuan, Liaoning?" The agent retrieves the correct documents for both locations, indicating full evidence coverage. However, it incorrectly answers "Kaiyuan, Liaoning," as the retrieved evidence stating that Pulandian District is located in the south of Liaoning province, while Kaiyuan is in the northeast.
```

### --- Page 0010 ---

```markdown
According to our taxonomy, this failure is classified as a reasoning logic error for null coverage, where the error originates from incorrect reasoning rather than missing evidence. Restart-based baselines respond to this failure by regenerating the entire retrieval-reasoning trajectory, even though the retrieved documents contain sufficient information to answer the question.

In contrast, DR-RAG diagnoses the earliest reasoning failure and performs localized repair by reusing the retrieved evidence and recomputing only the affected reasoning steps. This targeted intervention corrects the error without additional retrieval and avoids unnecessary recomputation, demonstrating the effectiveness of diagnosis-guided localized repair.

## 6 Conclusion

In this work, we presented DR-RAG, a diagnose-and-repair framework for Agentic Retrieval-Augmented Generation that enables efficient failure correction without full-plenitude retrains. DR-RAG models failure handling as trajectory-level diagnosis and localized repair by identifying error types and earliest failures within the reasoning trajectories. Experiments show that DR-RAG consistently improves accuracy and efficiency, supported by reliable automated diagnostics and stable repair behavior. Overall, this work reframes failure handling in Agentic RAG as a structured repair problem rather than a costly reset, providing a practical framework for more robust and efficient agentic reasoning systems in multi-step reasoning applications.

## References

1. Garima Agrawal, Tharindu Kumara, Zeyad Alghamdi, and H. J. Z. M. M. A. A. 2020. A Study of Points of Failure in Retrieval-Augmented Generation. In *FLML*, 162–177.
2. Self-RAG: Learning to Retrieve, Generate, and Critique through Reinforcement. In *ICLR OpenReview*.
3. Shuyang Chen, Ziqiang Zhang, Akihiro Yokoo, Akari Saito, and Masateru Sato. 2025. Enhancing Retrieval in a Linguistic Variation Task. *ACL* 2025:5831–5835.
4. Guochen Zhang, Xiaozhong Chen, Dongwei Yu, Haotian Xu, Wang Xie, Zhenhua Song, Wenbo Tian, Huifeng Chen, and Jianfeng Zhang. 2023. Improving Agentic Retrieval Agents via Markovian State Reconstruction. *CoRR* abs/2315.12307 (2023).
5. Yishan Du, Yang Ke, Tang Jun, Pingyuan Zheng, Zhaoxi Chen, Zhauni, and Zhanghua Wang. 2023. Enhancing RAG Models via Answer-Supervised Learning. *CoRR* abs/2509.00727 (2025).
6. Gaunting Dong, Libo Zhang, Zhongwen Wang, Xiaozhu Zhao, Jinshan Wang, Huanyu Zhao, Fuzheng Zhang, Kun Guo, Guozhao Zha, Yizhu Jin, J-Rong Wen, and Zicheng Dong. 2025. Agentic Entropy-Halved Policy Optimization. *CoRR* abs/2516.14435 (2025).
7. Zhaoxi Li, Weizhong Jian, Xiaoli Ji, Hongjin Qiu, Tian Zhao, Fanpu-ning, and J-Rong Wen. 2023. RAG-CI: Leveraging Attention Critic-Guided Agentic Workflow for Retrieval-Augmented Generation. In *ACL* (2023).
8. Yunting Dong, Shuang Mo, Kai Li, Hanzhe Liu, Ruicheng Liu, and Zhaoxi Zhang. 2023. Zhaoxi Zhang, Jianfeng Zhang, Huijuan Wang, Zhaoxi Dong, and J-Rong Wen. 2023. Agentic Reinforced Policy Optimization. *CoRR* abs/2517.09329 (2025).
9. Jaxuan Qian, Wu M., Mingyang Xu, Shusheng Xu, Chiyu Li, Zhiyu He, Shubian Chen, and Yi Wu. 2023. Beyond Simple Token Truncating: Long-Context Inference with Large-Scale Synthesis. *CoRR* abs/2585.02705 (2023).
10. Gauthier Guiet, Kévin Ondaïra-Terrail, Aroreo Bonnet, and L. A. 2024. Automated Evaluation of Retrieval-Augmented Language Models with Task-Specific Exam Generation. In *ICML OpenReview*.
11. Kelvin G. K. Lee, Kento Lee, Taro Pungprapap, and Ming-Wai Cheng. 2020. Retrieval-Augmented Generation in Natural Language Processing. In *ICML* (Proceedings of the 37th International Conference on Machine Learning, Vol. 119), 3929–3938.
12. Yanchun Hu, Akash Khanna, Deepak Supreeth, and Vikas Gupta. 2020. Constructing a Rich-Data RAG Dataset for Semantic Evaluation of Reasoning Steps. In *Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020*. International Committee on Computational Linguistics, 6620–6625.
13. Chengkai Hu, Xiaodai Chen, Hongtao Huang, Zeng Zhang, and Lina Yao. 2020. Generative Chain of Behavior for User Trajectory Prediction. *arXiv preprint arXiv:2012.18373 (2020)*.
14. Chengkai Hu, Junda Wu, Yuxia Ziu, Zihuan Wang, Yuji Tang, Ruixi Zhang, Ryan A. Ross, Brayan K. Temprano, Zhong Li, et al. 2023. Towards agentic recommender systems in the era of multimodal large language models. *arXiv preprint arXiv:2305.16783 (2023)*.
15. Chengkai Hu, Yu Xu, Yiran Kaige Xie, Tong Yu, Julian McAuley, and Lina Yao. 2023. Embedding-based adaptive retrieval-augmented generation of large language models. In *Proceedings of the 31st International Conference on Computational Linguistics*, 1433–1442.
16. Yinan Zhao, Jiarong Zhao, Sukun Ding, Shile Wang, Xin Zhang, Song Tang, and Zhaoxi Zhang. 2023. RAG-STAR: Enhancing Deliberative Reasoning with Retrieval-Augmented Learning and Reinforcement. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. *arXiv preprint arXiv:2306.11074 (2023)*.
17. Shuangqi Jiao, Xinyu Xiao, Yunfan Wu, Shuhan Q. Chengkai Huang, Quan Z. Ma, Chengxian Shen, and Zhaoxi Zhang. 2023. RAG-ROUGE: Guided-Deliberate RAG with Reinforcement Learning. *arXiv preprint arXiv:2307.11041 (2023)*.
18. Huxon Jin, Hansel Zeng, Dong Wang, Hamed Zamani, and Javis Zhang. 2023. RAG-REINFORCE: A Reinforcement Learning Framework for Retrieval-Augmented Generation. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. *arXiv preprint arXiv:2308.11041 (2023)*.
19. Jeff H. Johnson, Matthew L. Dyer, and Hervé J. 2021. Billion-Scale Multilingual Pretraining with EFL. *IEEE Trans. Neural Netw. Learn. Syst.* 32, 1 (2021), 1–15.
20. Vladimir Karpukhin, Aroreo Bonnet, and J. 2021. Retrieval-Kernel Search: Led by W. Serafini, Dang Chen, and Wen Yu. 2021. Dense Passage Retrieval for Open-Domain Question Answering. In *ACM SIGIR Conference on Research and Development in Information Retrieval*.
21. Wosuk Kwon, Zhuohui Li, Siyuan Zhang, Ying Shen, Lianmin Zheng, and Zhaoxi Zhang. 2023. RAG-META: Meta-Learning for Retrieval-Augmented Generation. In *SOPF* 2021, 611–626.
22. Alex K. M. M. M. M. 2023. Automated Belief-Sharing via Agentic Learning. *arXiv preprint arXiv:2301.13975 (2025)*.
23. Zhaoxi Li, Ethan P. Frechette, Alexis Petroni, Viktoriia Kovalchuk, Namrata Gokhale, Heinrich Kittler, Timo Kresch, Sebastian Reidel, and Duro Kale. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In *NeurIPS*.
24. Zhaoxi Li, Guanting Dong, Jiajun Li, Yizhu Yang, Yizhu Zhao, Peiten Zhang, and Zhaoheng Dong. 2025. Search3: Agentic Search-Enhanced Large Reasoning Models. *CoRR* abs/2516.13926 (2025).
25. Xiaoxi Li, Weizhong Jian, Jiarong Liu, Guanting Dong, Jiajun Li, Yizhu Wang, Yizhu Zhao, Jirong Wen, Wan Liu, and Zhaoxi Dong. 2025. DeepAgent: A Great Recurrent Agent with Learning Tools. *CoRR* abs/2508.21612 (2025).
26. Yifan Li, Kun Wang, Shen O., Chengfu Huang, L. Jiang, Lijia Lian, Jaxuan Qian, and Jiajia Zhang. 2021. Self-supervised learning-based weight adaptive (RAG) for fast cross-modal retrieval. *Signal Image Video Process* 15, 1 (2021), 57–71.
27. Haruo Ichikawa, Chikafumi Hasegawa, and K. 2023. SpecAgent: An End-to-End Mobile Infrastructure for Speculatively Humanly Assistant. *arXiv preprint arXiv:2307.11041 (2023)*.
28. Qili Liu, Zhaoxi Zhang, Yiming Wang, Shuran Tian, Zhiwei Chen, and J. 2023. MARC-R: Beyond Single Retrieval via Reinforcement-Learned Multi-Token Retrieval. *CoRR* abs/2517.09329 (2025).
29. Dongyu Fu, Lin Qian, Keqiang Hu, Tanshan Zhang, Shubian Chen, Cheng Huang, Guixing Wang, Sichao Sun, Huanyu Li, Zhaoxi Zhang, Bing Wang, Jianrong Jiang, Tong He, Zhiguo Wang, Fengfei Liu, Yao Zhang, and Zheng Zhang. 2023. RAG-REINFORCE: A Reinforcement Learning Framework for Retrieval-Augmented Generation. *arXiv preprint arXiv:2308.11041 (2023)*.
```

### --- Page 0011 ---

```markdown
Doctor-RAG: Failure-Aware Report for Agentic Retrieval-Augmented Generation

| Reference                                                                 | Citation                                                                 |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Zhang, 2024. RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation. In NeurIPS. | [14] Zhang, 2024. RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation. In NeurIPS. |
| Zheng, H., Si, Y., Liu, H., Hattab, L., Weiwei, S., Niu, S., Yang, L., Yu, R., Fan, B., Wang, J., Zhang, H., Xiong, M., Qiu, J., Guo, Q., Yang, J., Ouyang, A., Zhe, T., Huang, W., Yang, Z., Zhang, Y., Miao, Y., Zhang, T., and Janchun, R. 2023. Deep Research: A Systematic Survey. arXiv:2312.10368. | [15] Zheng, H., Si, Y., Liu, H., Hattab, L., Weiwei, S., Niu, S., Yang, L., Yu, R., Fan, B., Wang, J., Zhang, H., Xiong, M., Qiu, J., Guo, Q., Yang, J., Ouyang, A., Zhe, T., Huang, W., Yang, Z., Zhang, Y., Miao, Y., Zhang, T., and Janchun, R. 2023. Deep Research: A Systematic Survey. arXiv:2312.10368. |
| Habibzadeh, J., Janda, J., Yingxin, J., Min, E., Chen, Z., Zhang, C., Wayne, K., Zhao, L., Fang, J., and Rong, W. 2022. R-Sheriff: Reinventing the Search Capability in LMs via Reinforcement Learning. CoRR abs/2855.2655 (2022). | [16] Habibzadeh, J., Janda, J., Yingxin, J., Min, E., Chen, Z., Zhang, C., Wayne, K., Zhao, L., Fang, J., and Rong, W. 2022. R-Sheriff: Reinventing the Search Capability in LMs via Reinforcement Learning. CoRR abs/2855.2655 (2022). |
| Liana, T., 2024. The Illusion of Harder Models. CoRR abs/2077.4712 (2024). | [17] Liana, T., 2024. The Illusion of Harder Models. CoRR abs/2077.4712 (2024). |
| Tharvi, T., Mrinmoy, B., Balasubramanian, T., Jash Khat, and Ashish Sabharwal. 2023. Empirical Methods Questioning Single-Stop Question Composition. Trans. Assoc. Comput. Linguistics 10 (2022), 553–594. | [18] Tharvi, T., Mrinmoy, B., Balasubramanian, T., Jash Khat, and Ashish Sabharwal. 2023. Empirical Methods Questioning Single-Stop Question Composition. Trans. Assoc. Comput. Linguistics 10 (2022), 553–594. |
| Peilin, W., Hanzhang, H., Wan, Weishan, Zhaoyang, K., Kiyu, D., and Ziyu, H. 2023. MHR: Hierarchical Process for Agentic Retrieval-Augmented Generation. CoRR abs/2855.8774 (2023). | [19] Peilin, W., Hanzhang, H., Wan, Weishan, Zhaoyang, K., Kiyu, D., and Ziyu, H. 2023. MHR: Hierarchical Process for Agentic Retrieval-Augmented Generation. CoRR abs/2855.8774 (2023). |
| Zhao, Y., Yang, C., Yuxin, Z., and Huan, L. 2024. Corrective Retrieval-Augmented Generation. CoRR abs/2482.15884 (2022). | [20] Zhao, Y., Yang, C., Yuxin, Z., and Huan, L. 2024. Corrective Retrieval-Augmented Generation. CoRR abs/2482.15884 (2022). |
| Junyang, L., Kai, Deng, Ruan, Keqin, Yang, W., Li, Lianghua, Deng, Mei, Li, Shizhen, Yu, Mingjie, Li, Pei, Ping, Wang, Q., Zhu, Min, Ren, Xiao, Shinken, Li, Shuang, Luo, Tianyi, Lu, Tianheng, Weibo, Yu, Xinchang, Ren, Xinyu, Zhang, Y., Xu, Yuancheng, Zeng, Fan, Yang, Yu, Chen, Zhang, Ziyang, Zhou, and Zhan, Q. 2022. Quesh: Technical Report. CoRR abs/2855.2985 (2022). | [21] Junyang, L., Kai, Deng, Ruan, Keqin, Yang, W., Li, Lianghua, Deng, Mei, Li, Shizhen, Yu, Mingjie, Li, Pei, Ping, Wang, Q., Zhu, Min, Ren, Xiao, Shinken, Li, Shuang, Luo, Tianyi, Lu, Tianheng, Weibo, Yu, Xinchang, Ren, Xinyu, Zhang, Y., Xu, Yuancheng, Zeng, Fan, Yang, Yu, Chen, Zhang, Ziyang, Zhou, and Zhan, Q. 2022. Quesh: Technical Report. CoRR abs/2855.2985 (2022). |
| Zhibin, T., Yang, P., Qisi, Zhang, Yuxin, Bengi, William W. Cohen, Raslan Salakhutdinov, and Christopher D. Manning. 2018. HoPQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018. Association for Computational Linguistics, 2369–2380. | [22] Zhibin, T., Yang, P., Qisi, Zhang, Yuxin, Bengi, William W. Cohen, Raslan Salakhutdinov, and Christopher D. Manning. 2018. HoPQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018. Association for Computational Linguistics, 2369–2380. |
| Shubhy, Y., Diana, K., Jeffrey, A., Nida, Shafak, K., Narasimhan, V., and Yu, J. 2022. Rethinking Reasoning Rescaling and Inferring Language Models. In ICLR OpenReview. | [23] Shubhy, Y., Diana, K., Jeffrey, A., Nida, Shafak, K., Narasimhan, V., and Yu, J. 2022. Rethinking Reasoning Rescaling and Inferring Language Models. In ICLR OpenReview. |
| Jiedian, X., Ke, J., Yixiu, W., Chengkai, H., Lanshu, N., Liao, Y., and Long, H. 2022. Knowledge Remains: Analyzing Retrieval-Augmented Generation. In Long-Short-Horizon Action Reasoning. arXiv:2201.18740. | [24] Jiedian, X., Ke, J., Yixiu, W., Chengkai, H., Lanshu, N., Liao, Y., and Long, H. 2022. Knowledge Remains: Analyzing Retrieval-Augmented Generation. In Long-Short-Horizon Action Reasoning. arXiv:2201.18740. |
| Asaf, H., Yudai, L., Eitan, R., Yotam, Peretz, Roy, Bar-Haim, and Michal Shmueli. 2022. Towards CLEAR: Prioritizing AIs via LLM-as-a-Module. Sage J. | [25] Asaf, H., Yudai, L., Eitan, R., Yotam, Peretz, Roy, Bar-Haim, and Michal Shmueli. 2022. Towards CLEAR: Prioritizing AIs via LLM-as-a-Module. Sage J. |
| Zhao, Z., Hu, Y., Wang, D., Haoxin, Z., Yanjie, L., Yuhua, Chen, Shushen, T., Xiong, K., Xiangyu, Z., and Enchong, D. 2023. Towards an Efficient Agentic Retrieval-Augmented Generation Framework. CoRR abs/2855.2985 (2022). | [26] Zhao, Z., Hu, Y., Wang, D., Haoxin, Z., Yanjie, L., Yuhua, Chen, Shushen, T., Xiong, K., Xiangyu, Z., and Enchong, D. 2023. Towards an Efficient Agentic Retrieval-Augmented Generation Framework. CoRR abs/2855.2985 (2022). |
```


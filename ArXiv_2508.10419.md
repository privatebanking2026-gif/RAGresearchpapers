# ArXiv 2508.10419

### --- Page 0001 ---

```markdown
# ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning

**Juyuan Wang\textsuperscript{1}**  Rongchen Zhao\textsuperscript{1}  Wei Wei\textsuperscript{2}  Yufeng Wang\textsuperscript{1}  
Mo Yu\textsuperscript{4}  Jie Zhou\textsuperscript{3}  Jin Xu\textsuperscript{1,3}  Liyan Xu\textsuperscript{4}  

\textsuperscript{1}School of Future Technology, South China University of Technology  
\textsuperscript{2}Independent Researcher  
\textsuperscript{3}Pazhou Lab, Guangzhou  
\textsuperscript{4}WeChat AI, Tencent  

---

## Abstract

Narrative comprehension on long stories and novels has been a challenging domain attributed to their intricate plotlines and entangled, often evolving relations among characters and entities. Given the LLM's diminished reasoning over extended context and its high computational cost, retrieval-based approaches remain a pivotal role in practice. However, traditional RAG methods could fall short due to their stateless, single-step retrieval process, which often overlooks the dynamic nature of capturing interconnected relations within long-range context. In this work, we propose ComoRAG, holding the principle that narrative reasoning is not a one-shot process, but a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to human cognition on reasoning with memory-related signals in the brain. Specifically, when encountering a revealing impasse, ComoRAG orchestrates dynamic reasoning cycles while interacting with a dynamic memory workspace. In each cycle, it generates probing queries to devise new exploratory paths, then integrates the retrieved evidence of new aspects into a global memory pool, thereby supporting long-range context comprehension for their resolution. Across four challenging long-context narrative benchmarks (200K+ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to 11% compared to the strongest baselines. Further analysis reveals that ComoRAG is particularly advantageous for complex evidence acquiring global context comprehension, offering a principled, cognitively motivated paradigm towards retrieval-based stateful reasoning. Our framework is made publicly available at [https://github.com/EternityJune25/ComoRAG](https://github.com/EternityJune25/ComoRAG).

---

## 1 Introduction

The core challenge of long narrative comprehension lies not merely in connecting discrete pieces of evidence, a task more naturally defined as multi-hop Question Answering (QA), but in performing a dynamic cognitive synthesis to grasp necessary background and content progression (Xu et al. 2022a). Unlike multi-hop QA (Yang et al. 2018), which seeks a static path through fixed facts, narrative comprehension requires emulating a human reader: continuously building and revising a global mental model of the plot, characters, and their evolving relationships (Johnson-Laird 1983). The complexity of this process is well exemplified by a classic question “Why did Snape kill Dumbledore?” from a Harry Potter series. Answering this requires weaving a complete web of evidence from disparate clues spanning multiple books—Dumbledore’s terminal illness, the Unbreakable Vow, and Snape’s deeply concealed loyalty. The true significance of these clues is only fully reconciled in hindsight. This capability is what we term stateful reasoning: it demands more than linking static evidence; it requires maintaining a dynamic memory of the narrative, one that is constantly updated as new revelations emerge. Long-context LLMs have demonstrated promising performance on benchmarks such as the “Needle in a Haystack” (Eisenchos, Yogatama, and Al-Rfou 2023). However, their capacity to process long narratives (200K+ tokens) remains limited by their context windows. Furthermore, as the input length increases, these models are prone to the “lost in the middle” problem (Liu et al. 2024), which raises perplexity and impairs generalization.

---

![Comparison of RAG reasoning paradigms](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
# Page 0002

Certain quality. This limitation is particularly pronounced in narrative tasks which require situational reasoning. As a result, retrieval-augmented generation (RAG) (Lewis et al. 2020) has emerged as an important direction for tackling context comprehension with LLMs, leveraging text embeddings for more advanced retrieval paradigms such as embeddings situated and global context (Wu et al. 2025).

However, existing RAG methods still struggle to effectively address this challenge. Advanced single-step retrieval remains limited by its static index. This includes methods such as RAPTOR (Sarthi et al. 2024), which clusters and summarizes text chunks to retrieve at different levels of granularity; HippoRAG2 (Gutiérrez et al. 2025) and GraphRAG (Edge et al. 2025), which build knowledge graphs to achieve multi-hop reasoning in a single retrieval step. Nonetheless, one-shot static retrieval inevitably leads to shallow comprehension. For example, the evidence about Snape in Fig. 1(a) can mislead the model into making a false inference.

As a remedy, multi-step RAG methods offer a more promising direction, such as IRCoT (Driess et al. 2023), which interleaves the retrieval process with Chain-of-Thought reasoning (Wei et al. 2022); Self-RAG (Asai et al. 2023), which trains a model to adaptively retrieve and reflect on evidence; and MemoRAG (Qian et al. 2025), which uses a dual-system architecture to generate clues from compressed global context. These methods fail to obtain coherent reasoning through iterative retrieval. However, their retrieval steps are typically independent, which lack coherent reasoning throughout narrative progression, featuring insufficient ability to integrate contradictory evidence such as “Snape protects/bullies Harry” and cannot understand the evolution of his actions, ultimately unable to yield the correct answer.

In this work, we seek inspiration from the function of Prefrontal Cortex (PFC) in human brains, which employs a stateful reasoning process called Metacognitive Regulation (Fernandez-Duque, Baird, and Pomer 2000). This process is not a single action but a dynamic interplay between new evidence acquisition, driven by goal-directed memory probes (Dohs et al. 2006; Miller and Constantino 2024), and subsequent knowledge consolidation. During consolidation, new findings are integrated with past information to construct an evolving, coherent narrative. This iterative cycle allows the PFC to continuously assess its understanding and revise its strategy, providing a direct cognitive blueprint for our framework’s stateful reasoning approach.

We introduce ComoRAG, a cognitive-inspired, memory-organized RAG framework, imitating the human Prefrontal Cortex for achieving stateful reasoning. At its core is a dynamic cognitive loop operating on a memory workspace, which actively probes and integrates new evidence to build a coherent narrative comprehension. 

This process, as illustrated in Figure 1(c), is a closed loop of evolving reasoning states. Faced with a complex query like “Why did Snape kill Dumbledore?”, the system’s memory state evolves from an initial “causally incomplete” event (Snape kills Albus), to an “apparent contradiction” upon finding contradictory information (Snape protects Harry), and ultimately to a logically consistent rendered context through exploration and evidence fusion. Only in this final, complete cognitive state can ComoRAG provide the correct textual reasoning, deriving the true insight that it was “an act of loyalty, not betrayal”.

This cognitively-inspired design yields substantial improvements across four challenging long-context narrative benchmarks. ComoRAG is shown to consistently outperform all categories of strong baselines across each dataset. Our analysis reveals several key findings. First, Snape directly from the cognitive loop, which transforms a static knowledge base into a dynamic reasoning engine; for instance, accuracy on EN-MC jumps from a static-retrieval baseline of 64.6% to 72.9%, with performance efficiently converging in around 2 cycles. Second, our framework excels on narrative queries that require global understanding of plot progression, achieving up to a 19% relative F1 improvement on these challenging question types where others fail. Finally, our framework demonstrates remarkable modularity and generalizability; this core loop can be flexibly integrated to existing RAG methods such as RAPTOR, which directly yields a 21% relative accuracy gain. Also, switching to a stronger model as the backbone LLM agents can upgrade reasoning in the entire cognitive loop, attaining accuracy from 72.93% to 78.17%. These results collectively validate that ComoRAG provides a principled, cognitively-inspired framework for retrieval-based long narrative comprehension towards stateful reasoning.

## 2 Methodology

We introduce ComoRAG, an autonomous cognitive architecture designed to formalize and implement the process of Metacognitive Regulation outlined in the Introduction. The architecture’s design is directly inspired by the functional mechanisms of the Prefrontal Cortex (PFC) and is founded on three conceptual pillars: (1) a Hierarchical Knowledge Source for deep contextual understanding; (2) a Dynamic Memory Workspace for tracking and integrating the multi-turn reasoning; and (3) a Metacognitive Control Loop that drives the entire resolving procedure.

### 2.1 Problem Formulation: Towards Princpled Narrative Reasoning

Our objective is to design a framework for stateful reasoning in RAG scenarios. Especially, it aims to resolve those queries that require global context comprehension in the first place, commonly seen in narratives, where conventional RAG may fail to recognize relevant context based on the surface form of queries. Formally, denote the initial query as $q_{init}$, and a knowledge source $X$ derived from the original context, our framework $F$ leverages a series of adaptive operations to yield the final answer, $A_{final}$, through discrete time steps $t = 1, \ldots, T$ with underlying memory control.

At the beginning of each step $t$, $F$ determines its focus of reasoning—a set of new probing queries $P(t)$, representing new information to seek that may logically deepen the query comprehension and ultimately complement the an-
```

### --- Page 0003 ---

```markdown
![An illustration of ComoRAG. Triggered by a reasoning impasse (Failure), the Metacognitive Regulation loop consists of five core operations described in Section 2.3: 1) Self-Probe to devise new exploratory probing queries based on past memory units; 2) Tri-Retrieve to retrieve evidence from three knowledge sources; 3) Mem-Encode to form new memory units on how the latest evidence of new aspects could complement the final query resolution; 4) Mem-Fuse to generate the answers integrating new and past memory units; 5) Try-Answer to perform query answering using new memory information produced in this cycle.](assets/page_0003_img_1.png)

## 2.2 The Hierarchical Knowledge Source

To overcome the limitations of the individual representation of the given context, our framework first builds a hierarchical knowledge index $\chi$ for retrieval that models the new text from three complementary cognitive dimensions, analogous to how the PFC integrates different memory types from various brain regions, particularly supporting cross-layer reasoning from raw evidence to abstract relationships.

### Veridical Layer: Grounding in Factual Evidence. 
To ensure all reasoning is traceable to source evidence, a veridical layer $\chi^{ver}$ is firmly established, constituted by raw text chunks directly, analogous to the precise recall of factual details in human memory. For more accurate retrieval on text chunks, we instruct a LMI to generate knowledge triples (subject-predicate-object) for each chunk. These triples participate in each retrieval, strengthening the connection between an incoming query and the corresponding text chunk, which is proven effective by HippoRAG (Jiménez Gutierrez et al. 2024). Further details are described in Appendix B.

### Semantic Layer: Abstracting Thematic Structure. 
To capture thematic and conceptual connections that transcend across long-range contextual dependencies, a semantic layer $\chi^{sem}$ is built, inspired by the work RAPTOR that employs a GMM-driven clustering algorithm to recursively summarize semantically similar text chunks into a hierarchical summary tree. We reckon such semantic abstraction is necessary for deeper comprehension and follow the same reasoning. These summary nodes enable the framework to retrieve conceptual information beyond the surface level.

### Episodic Layer: Reconstructing Narrative Flow. 
The previous two layers equip views of both factual details and high-level concepts. However, they lack temporal development or plot progression that can be especially crucial for narratives. To enable such view with long-range causal chains, we introduce the episodic layer, $\chi^{ep}$, which aims to reconstruct the plotline and story arc by capturing the sequential narrative development. The process features a sliding window summarization across text chunks; each resulting node is then a summary that aggregates the narrative development of continuous or causally related events according to the timeline. Optionally, the sliding window process can be applied recursively to form higher-level views of content progression, extracting different levels of narrative flow as part of the knowledge source.

## 2.3 The Architecture of Metacognitive Regulation
The core of ComoRAG is a control loop that fully realizes the concept of metacognitive regulation. It is composed of a Regulatory Process for reflection and planning at each step, and a Metacognitive Process for executing reasoning and memory management with the Memory Workspace.
```

### --- Page 0004 ---

```markdown
# Dynamic Memory Workspace

The memory workspace contains memory units that serve as the bridge for a cohesive multi-step exploration and reasoning by metacognitive regulation. Each memory unit $m$ functionally conducts one retrieval operation, denoted as a tuple of three elements: $m = (p, e^{type}, c^{type})$, where $p$ is the probing query that triggers this retrieval; $e^{type}$ is the homogeneous set of evidence retrieved from a single knowledge layer (type $\in \{ver, sem, epi\}$); and $c^{type}$ is a synthesized cue that reflects how the retrieved evidence by the probe $p$ could complement the comprehension and resolution of the original query $q_{init}$. Concretely, $c^{type}$ is generated by a LLM in the role of Comprehension Agent, $c_{new}$, denoted as $c^{type} = \pi_{cue}(q_{init}, p, e^{type})$.

The formation of a memory unit $(p, e^{type}, c^{type})$ by each retrieval is defined as a $Mem-Encode$ operation. The memory workspace/pool will be utilized and updated throughout the reasoning cycle described below.

## The Regulatory Process

The regulatory process is invoked at the beginning of a reasoning cycle $t$ if the preceding cycle $t-1$ is concluded in failure. The core operation, $Self-Probe$, plans new probing queries for which retrieved information may contribute to the final answer, thereby devising new exploratory paths to break the impasse. It is orchestrated by a Regulation Agent, $r_{probe}$, whose decisions are informed by the reflection on the prior failure, exploring for more necessary background or relevant information towards a full context comprehension to resolve the original query $q_{init}$. The $Self-Probe$ technique probing history $P(t-1)$ is used to: (1) retrieve the end of the last step; and (2) the memory pool $M(t-1)$ to identify gaps that caused the failure, connected to all synthesized cues of memory units generated in the prior step, denoted as $C(t-1)$. Its output $P(t)$ is a new, strategic set of retrieving probes for the current cycle:

$$
P(t) = r_{probe}(q_{init}, P(t-1)_{hist}, C(t-1)) \tag{1}
$$

## The Metacognitive Process

The metacognitive process takes the new probes for this cycle $P(t)$, and performs reasoning towards resolving the original query while keeping track of the progress within the memory space. It comprises a series of operations, described in detail as follows.

- **Tri-Retrieve**: for each probing query $p \in P(t)$, a retrieval is conducted on each knowledge layer $r^{type}$ where type $\in \{ver, sem, epi\}$, such that evidence of high embedding similarity to per layer is retrieved in a standard Dense Passage Retrieval paradigm, with each evidence being either the raw text chunk, a semantically clustered summary, or a narrative flow summary.

- **Mem-Encode**: for each probe $p$ and type, the retrieved evidence is immediately processed by the aforementioned $Mem-Encode$, to generate a new memory cue that keeps track of how this specific probing could complement to the final answer. The number of all generated memory units at this step can be denoted as $|M(t)_{encode}| = 3 \times |P(t)|$.

- **Mem-Fuse**: new memory units in the above step $M(t)$ mainly emphasize aspects probed in the current cycle. To fully utilize the past experience and historical knowledge, the framework further identifies relevant synthesized cues from past units in the existing memory pool $M_{pool}^{-1}$, then generates a new synthesized cue for fusing past relevant evidence. Let $M_{pool}^{-1}$ represent past memory units whose cues are of high embedding similarity with $q_{init}$, and denote a LLM as Integration Agent $I_{fuse}$ that synthesizes these relevant past evidence into a high-level background summary, the new cue fusing past memory $C_{fuse}$ is then:

$$
C_{fuse}^{(t)} = \pi_{fuse}(q_{init}, M_{pool}^{-1} \odot q_{init}) \tag{2}
$$

- **Try-Answer**: with the new probing evidence in $M(t)_{encode}$ and the past-fusing cue $C_{fuse}$, a QA Agent, $\pi_{QA}$, is applied to these contexts to produce the cycle's final output $O(t)$:

$$
O(t) = \pi_{QA}(q_{init}, M(t)_{encode}, C_{fuse}^{(t)}) \tag{3}
$$

Specifically, a LLM is instructed to take these latest evidence and the past background as the context, determine and amend the original query can be resolved. It either yields the final answer and terminates the reasoning or signals Failure and continues to the next step.

- **Mem-Update**: this last step in a cycle simply incorporates the newly generated memory units into the global pool, with their embedding encoded, for future retrieval and evidence.

$$
M(t)_{global} = M(t-1) \cup M(t) \tag{4}
$$

## ComnoRAG

With the above steps from Tri-Retrieve to Mem-Update, one cycle of the cognitive loop is realized. For the initial step at $t = 1$, ComnoRAG starts with one round of $Tri-Retrieve$ followed by $Try-Answer$. If Failure is signaled, it initiates the Metacognitive loop of stateful reasoning on exploratory paths, characterized by the interlocking operations with the memory workspace, which enables to tackle complex narrative comprehension.

In essence, our framework grasps on the principle that for long context comprehensibility, especially in narratives where the entire context is cohesively interconnected through the underlying plot progression (Xu et al. 2022a), the query resolution is not a linear pipeline; rather, it is a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to the human cognitive process. The overall process is further depicted in the algorithm of Appendix A; detailed prompts used by each LLM agent are provided in Appendix D.

# 3 Experimental Settings

## Datasets

Our experiments cover four long-context narrative understanding datasets for comprehensive evaluation, featuring both question answering from free generation (QA), and multi-choice questions by selecting the best option (MC).

- **NarrativeVQA** (Kocisky et al. 2017): a QA dataset consisting of books and movie scripts. For ease of computation, we follow prior works and randomly sample 500 questions from the test set, with average context length 58k tokens.
```

### --- Page 0005 ---

```markdown
| Category        | Method            | NarrativeQA |         |         | ENQA |         |         | ENMC |         |         | DetectiveQA |         |         | QA Avg |         |         | MC Avg |         |         |
|-----------------|-------------------|-------------|---------|---------|------|---------|---------|------|---------|---------|-------------|---------|---------|--------|---------|---------|--------|---------|---------|
|                 |                   | FI          | EM      | FI      | ACC  | FI      | EM      | ACC  | FI      | EM      | ACC         | FI      | EM      | ACC    | FI      | EM      | ACC    | FI      | EM      |
| LLM             | GPT-4o-mini       | 27.29       | 7.00    | 29.83   | 12.82| 30.57   | 30.68   | 28.56| 9.91    | 30.63 |             |         |         |        |         |         |        |         |         |
| Naive RAG       | BGE-M30(0.3B)     | 23.16       | 15.10   | 23.71   | 16.42| 59.82   | 54.54   | 23.44| 17.80   | 24.57 | 61.30       | 24.96   | 21.79   | 61.82  |         |         |        |         |         |
|                 | NVM-Embed-v2(R)   | 27.18       | 17.80   | 34.24   | 24.57| 61.30   | 24.96   | 21.79| 15.60   | 27.95 | 19.75       | 56.21   | 36.30   | 24.76  |         |         |        |         |         |
| Enhanced RAG    | RAPTOR            | 27.84       | 17.80   | 26.63   | 19.65| 57.21   | 57.95   | 27.09| 18.73   | 57.58 | 23.79       | 16.15   | 58.54   |        |         |         |        |         |         |
|                 | HippoRAGv2       | 23.12       | 15.20   | 24.45   | 17.09| 26.60   | 56.81   | 23.79| 16.15   | 58.54 |             |         |         |        |         |         |        |         |         |
| Multi-step RAG  | Self-RAG          | 19.60       | 6.40    | 12.84   | 4.27 | 59.83   | 52.27   | 16.42| 5.34    | 56.05 |             |         |         |        |         |         |        |         |         |
|                 | RAPTOR+IRCoT      | 23.29       | 15.20   | 19.40   | 11.64| 55.89   | 51.13   | 21.35| 13.42   | 53.51 |             |         |         |        |         |         |        |         |         |
|                 | RAG               | 31.35       | 16.00   | 19.36   | 63.76| 64.77   | 31.72   | 17.68| 64.27   |         |             |         |         |        |         |         |        |         |         |
|                 | HippoRAGv2+IRCoT  | 28.98       | 13.00   | 29.27   | 18.24| 64.19   | 62.50   | 29.15| 16.62   | 15.62 |             |         |         |        |         |         |        |         |         |
|                 | ComoRAG (Ours)    | 31.43       | 18.60   | 24.52   | 25.07| 72.93   | 68.18   | 32.98| 21.84   | 7.56  |             |         |         |        |         |         |        |         |         |

Table 1: Evaluation results on four long narrative comprehension datasets. For fair comparison, all methods use GPT-4o-mini as the LLM backbone, and all non-naive RAG methods use BGE-M3 for retrieval (details in Section 3). We highlight the best and second-best results. ComoRAG (Ours) is shown consistently outperforming all baselines across all datasets.

- ENQA from exBENCH (Zhang et al. 2024): a QA dataset with 351 questions on classic novels, with average context length over 200k tokens.
- EN.MC from exBENCH: a MC dataset with 229 questions on classic novels of similar length as ENQA.
- DetectiveQA (Xu et al. 2024): a MC dataset with 20k unique fiction writer query over 100k tokens. We randomly sample 20% of instances to reduce the context length.

For evaluation metrics, we report both FI and Exact Match (EM) scores for QA datasets, and report Accuracy (ACC) for MC datasets. To ensure fairness in revising multiple-choice questions, we only expose the options during `Answer`.

Baselines: We employ four types of baselines as follows, covering different paradigms for long context QA.
- LLM: the non-RAG setting, where the entire context (capped by length 128k) is provided to the LLM directly.
- Naive RAG: the standard RAG setting that splits the raw context by chunks for retrieval. We set the max chunk length as 512 tokens in all experiments.
- Enhanced RAG: RAG methods with augmented retrieval index, including RAPTOR (Sarthi et al. 2024) that constructs a semantic summary tree over text chunks, and HippoRAGv2 (Gutiérrez et al. 2025) that builds the knowledge base for entities in text chunks. We also experimented with GraphRAG (Edge et al. 2025); however, it requires exponential computational cost for building the retrieval index, being less practical for full evaluation. We separately report GraphRAG on a subset in Appendix B.
- Multi-step RAG: RAG methods with multi-step or iterative retrieval strategies (Trivedi et al. 2023) leveraging Chain-of-Thought (CoT) as intermediate queries that iteratively retrieve evidence. Self-RAG (Asai et al. 2022) trains a model that incorporates a directive model to control when to stop retrieval. MemoRAG (Qian et al. 2025) trains a model that

4.1 Main Results
Evaluation results of our main experiments are shown in Table 1. Remarkably, ComoRAG achieves the best performance upon all baselines across datasets. Despite using the lightweight 0.3B BGE-M3 for retrieval, it significantly outperforms RAG with much larger 8B embedding models. Overall, ComoRAG demonstrates consistent improvement for tackling long narrative comprehension, surpassing strong RAG methods for various paradigms.

Upon closer examination, ComoRAG exhibits distinct advantages on the two exBENCH datasets returning ultra-long contexts. More broadly, Figure 1 illustrates that ComoRAG is more robust and insensitive to longer contexts, sustaining its efficacy over HippoRAGv2, with the accuracy gap peaking at +24.6% for documents exceeding 150k tokens, which
```

### --- Page 0006 ---

```markdown
| Method     | EN.MC | EN.QA |
|------------|-------|-------|
|            | ACC   | F1    | EM    |
| ComoRAG   | 72.93 | 34.52 | 25.07 |
| Baselines  |       |       |       |
| HippoRAGv2 | 60.26 | 24.45 | 17.09 |
| RAPTOR     | 57.21 | 26.33 | 19.65 |
| Index      |       |       |       |
| w/o Veridical | 51.97 | 22.24 | 15.88 |
| w/o Semantic | 64.63 | 30.82 | 22.65 |
| w/o Episodic | 64.63 | 31.48 | 21.47 |

Table 2: Ablation studies of ComoRAG.

---

4.2 Ablation Studies

We perform ablation studies on EN.MC and EN.QA datasets by systematically removing key modules in ComoRAG. The results are shown in Table 2.

**Hierarchical Knowledge Source**  
All three knowledge layers contribute supplementary enhancements to the final performance, with the Veridical layer being the most significant relative index. It provides the basis for factual-grounded reasoning, as confirmed by the 30% relative performance drop upon its removal.

**Metacognition**  
Removing the Metacognition process essentially disables the memory workspace, where all agents operate on retrieved evidence directly, without knowledge consolidation from the synthesized cues. Disabling this module leads to a significant performance drop, as seen by the 22% relative decrease in F1 score on EN.QA, and an approximate 15% decrease in accuracy on EN.MC, underscoring the critical role of dynamic memory organization.

**Regulation**  
Removing the Regulation process cuts off the goal-oriented guidance, such that each cycle uses the same initial query for new evidence retrieval (duplicated evidence is removed), without generating probing queries that are crucial to new evidence acquisition. Disabling this module severely impacts retrieval efficiency, causing a 24% drop in accuracy on EN.MC and a 19% drop in F1 score on EN.QA.

Notably, removing both Metacognition and Regulation further degrades performance, effectively reducing the system to a one-shot resolver without multi-step reasoning. Overall, the ablation study results corroborate that the enhancement offered by ComoRAG stems from the synergy between its memory consolidation and dynamic evidence exploration, facilitated by the hierarchical knowledge index to provide enriched semantic information. Removing any of the core components would significantly weaken its narrative reasoning capabilities.

---

4.3 In-Depth Analysis of Iterative Retrieval

**Source of Gains: From Static Bottleneck to Dynamic Reasoning**  
Our analysis suggests that the stateful multi-step reasoning enabled by the Metacognition module is key to driving further improvement.

We first identify a “static bottleneck”: at the initial retrieval stage using the original query at step 0, the single-step evaluation score shows no significant advantage over the baseline HippoRAGv2+IRCoT. However, upon activating the cognitive loop, there presents a sustained and significant improvement, raising the accuracy to 72.93% on EN.MC, as shown in Figure 4. This further supports the findings from the ablation studies, which demonstrate a significant performance drop upon removing the entire loop. Additionally, Figure 4 illustrates that the majority of the improvement occurs within 2-3 cycles, confirming the efficiency of the process. The few remaining unresolved queries are tied to the inherent reasoning limitation of the base LLM, where our next analysis shows that the ceiling performance of ComoRAG can be lifted by switching to more capable LLMs.

**Model-agnostic Generalization**  
ComoRAG demonstrates generalization with different LLM backbones, with stronger LLMs further enhancing the reasoning process and final query resolution. To validate this, we replace GPT-4-mini with GPT-4.1 and Vowpal Wabbit-32B in the Qualitative loop, using the same knowledge source for retrieval. The results, presented in Figure 4 and the upper section of Table 3, show a notable improvement particularly with GPT-4.1, boosting the F1 score on EN.QA from 34.52 to 
```


### --- Page 0007 ---

```markdown
| Method       | NarQA      | ENQA      | EN.MC     | DetQA    |
|--------------|------------|-----------|-----------|----------|
|              | F1         | F1        | ACC       | ACC      |
| ComoRAG     | 31.43      | 34.52     | 72.93     | 68.11    |
| w/ Qwen-3.32| 32.17      | 35.29     | 74.24     | 69.32    |
| w/ GPT-4.1  | 35.43      | 35.82     | 78.17     | 76.14    |
| HippoRAGv2  | 23.12      | 26.33     | 60.26     | 56.81    |
| + Our Loop   | 29.12      | 31.76     | 68.56     | 63.64    |
| RAPTOR      | 27.84      | 26.33     | 57.21     | 57.95    |
| + Our Loop   | 30.55      | 34.31     | 69.00     | 62.50    |

Table 3: Efficacy of ComoRAG on model-agnostic generalization and Plug-and-Play flexibility.

38.82, and increases the accuracy on EN.MC from 72.93 to 78.17. These results demonstrate that ComoRAG effectively leverages and unseals the model’s capabilities during its stateful iterative reasoning process.

### 4.4 In-Depth Analysis of Query Resolution

To deepen the understanding of narrative query resolution, we roughly categorize all questions in our experimented datasets into three query types: factoid, narrative, and inferential, described as follows (details in Appendix C).

- **Factoid Queries**: queries answerable by a single, specific piece of information, often knowledge-seeking, e.g., “What religion is Octavio Amber?”
- **Narrative Queries**: queries that require an understanding of plot progression as a coherent background context, e.g., “Where does Trace choose to live at the end of the novel?”
- **Inferential Queries**: queries demanding reasoning, e.g., “What is the main reason that Nils first visits Aiden in his apartment?”

To systematically investigate the dynamics of ComoRAG reasoning, we first pose the question: what is the bottleneck in long-narrative reasoning for existing RAG methods? Figure 5 pictures a clear diagnosis. While some not-trivial solutions for factoid queries account for over 60% of initial solution, our iterative cognitive loop is essential for resolving complex narrative queries involving global context comprehension and deeper reasoning. These constitute nearly 50% of the problems that are solved exclusively through the Metacognitive loop.

![Distribution of solved query types](assets/page_0007_img_1.png)

![Benchmarking RAG methods across query types](assets/page_0007_img_2.png)

In this work, we propose ComoRAG for long narrative reasoning, aiming to address the “stateless” limitation of conventional RAG. ComoRAG is specifically inspired by the human brain’s Prefrontal Cortex: through a dynamic memory space and iterative probes, it fuses fragmented evidence into a coherent action to achieve stateful reasoning over narrative progression. Experiments validate that ComoRAG overcomes the bottleneck of existing methods by excelling at complex narrative and inferential queries, marking a paradigm shift from information retrieval to cognitive reasoning towards deeper long text comprehension.
```

### --- Page 0008 ---

```markdown
# References

Asai, A., Wu, Z., Wang, Y.; Sil, A., and Hajishirzi, H. 2024. Self-Rag: Learning to Retrieve, Generate, and Critique through Self-Reflection. In *The 17th International Conference on Learning Representations*.

Chen, J.; Xiao, S.; Zhang, P.; Liu, K.; Lian, D.; and Liu, Z. 2024. Multi-Embedding Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. In *The 17th International Conference on Learning Representations*. 

Dobbins, I. G.; and Hans, S. 2006. Cue- versus Probe-Dependent Prefrontal Cortex Activity during Contextual Remembering. *Journal of Cognitive Neuroscience*, 18(9): 1439–1452.

Edge, D.; Trinh, H.; Cheng, K.; Bradley, J.; Chao, A.; Mody, A.; Trust, T.; Metropolitan, D.; Nuss, R. O.; and Larson, J. 2025. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.

Eisenlohr, S. M.; Yogatama, D.; and Al-Rfou, R. 2023. Needle in a Haystack: Where Is It Finding Factual Associations in Long Texts. arXiv preprint arXiv:2307.09288.

Fernandez-Duque, D.; Baird, J. A.; and Posner, M. I. 2000. Executive Attention and Metacognitive Regulation. *Consciousness and Cognition*, 9(2): 288–307.

Gutierrez, B.; Shu, Y.; Qi, X.; Zhou, X.; and Su, Y. 2025. From RAG to Memory: Non-Parametric Continual Learning. *Proceedings of the International Conference on Machine Learning*.

Jimenez Gutierrez, B.; Shu, Y.; Gu, Y.; Yasuda, M.; and Su, Y. 2024. Hypergonic: Neurobiologically inspired long-term memory for large language models. *Advances in Neural Information Processing Systems*, 37: 59532–59569.

Johnson-Laird, P. N. 1983. *Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness*. Cambridge, MA: Harvard University Press.

Kocisky, T.; Schneider, J.; Blunsom, P.; Chernova, K. M.; Mells, G.; and Grefenstette, E. 2017. The NarrativeQA Reading Comprehension Challenge. *Transactions of the Association for Computational Linguistics*, 6: 317–328.

Lee, C.; Roy, R.; Xu, M.; Raiman, J.; Shoeybi, M.; Catanzaro, B.; and Ping, W. 2025. NV-EMB: Improved Techniques for Training LLMs as Generalist Embedding Models. In *The 17th International Conference on Learning Representations*.

Lewis, P.; Perez, E.; Petroni, A.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, Y.; Wt.-Rokatschied, T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M.; and Lin, H., eds., *Advances in Neural Information Processing Systems*, volume 33, 9459–9474. Curran Associates, Inc.

Liu, M. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Belvalcqua, M.; Petroni, F.; and Liang, P. 2024. Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*, 12: 157–173.

Miller, J. A.; and Constantinidis, C. 2024. Timescales of learning in prefrontal cortex. *Nature Reviews Neuroscience*, 25(9): 591–607.

Qian, H.; Liu, Z.; Zhang, P.; Mao, K.; Lian, D.; Dou, Z.; and Huang, T. 2025. Memory: Boosting long context processing with global memory-enhanced retrieval augmentation. In *Proceedings of the ACM on Web Conference 2025*, 2366–2377.

Sarthi, P.; Abdullahi, S.; Tuli, A.; Khanna, S.; Goldie, A.; and Manning, C. 2024. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. In *International Conference on Tree-Generated Retrieval* (ICLR).

Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal, A. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds., *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 10014–10037. Toronto, Canada: Association for Computational Linguistics.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; brian ichter; Xia, F.; Chi, E. H.; Le, Q. V.; and Zhou, D. 2022. Chain of Thought Prompting Elicits Reasoning in Large Language Models. In Oh, A. H.; Agarwal, A.; Belgrave, D.; and Cho, K., eds., *Advances in Neural Information Processing Systems*.

Wu, J.; Li, Z.; Li, Y.; Liu, K.; Li, J.; Yeung, D.; P. Zhou, J.; and Yu, M. 2025. Sitebm-v1.5: Improved Context Retrieval for Story Completion. arXiv:2508.01959.

Xu, L.; Li, J.; Yu, M.; and Zhou, J. 2024. Fine-Grained Modeling of Narrative Context: A Coherence Perspective via Retrospective Questions. In K. L. W.; Martins, A.; and Srikumar, V., eds., *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 5822–5838. Bangkok, Thailand: Association for Computational Linguistics.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W.; Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Riloff, E.; Chiang, D.; Hochreiter, J.; and Tsujii, J., eds., *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 2369–2380. Brussels, Belgium: Association for Computational Linguistics.

Zhang, X.; Chen, Y.; Hu, S.; Xu, Z.; Chen, J.; Hao, M.; Han, X.; Thai, Z.; Wang, S.; Liu, Z.; et al. 2020. Bench: Extending long context evaluation beyond 100k tokens. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 15262–15277.
```

### --- Page 0009 ---

```markdown
# A ComoRAG Algorithm

Algorithm 1: ComoRAG (Described in Section 2)  
Require: Initial Query $q_{init}$, Knowledge Source $X$, Max Iterations $T$  
Ensure: The final answer $O$ or a failure signal  

1. function COMO RAG($q_{init}, X, T$)  
2. \quad $M^{(0)} \gets M_{encode}(q_{init}, P^{(0)}_{hist}, E^{(0)})$  
3. \quad $M^{(0)}_{pool} \gets M_{update}(M^{(0)}_{pool}, M^{(0)}_{encode})$  
4. \quad $P^{(0)} \gets q_{init}$  
5. \quad $C^{(0)} \gets \emptyset$  
6. for $t \gets 1, \ldots, T$ do  
7. \quad $E^{(t)} \gets \text{Tri-Retrieve}(P^{(t-1)})$  
8. \quad $M^{(t)}_{encode} \gets M_{encode}(q_{init}, P^{(t)}, E^{(t)})$  
9. \quad $M^{(t)}_{pool} \gets M_{update}(M^{(t-1)}_{pool}, M^{(t)}_{encode})$  
10. \quad $P^{(t)} \gets P^{(t-1)} \cup P^{(t)}$  
11. \quad $C^{(t)} \gets C^{(t-1)}$  
12. if $O \neq \text{FailureSignal}$ then  
13. \quad $O(t) \gets \text{Try-Answer}(q_{init}, M^{(t)}_{pool}, C^{(t)})$  
14. \quad if $O(t) \neq \text{FailureSignal}$ then return $O$  
15. end if  
16. $M^{(t)}_{pool} \gets M_{update}(M^{(t-1)}_{pool}, E^{(t)})$  
17. $P^{(t)}_{hist} \gets P^{(t-1)} \cup P^{(t)}$  
18. $C^{(t)} \gets M^{(t)}_{pool}$  
19. end for  
20. return FailureSignal  
21. end function  

## B Implementation Details

### B.1 Veridical Layer

As described in Section 2.2, ComoRAG empowers Large Language Models by constructing a hierarchical knowledge source, whereby the Veridical Layer is a foundational component governing the construction process of HippoRAGV2 (Gutierrez et al. 2025) to add a mapping between knowledge graphs (KGs) and text chunks to facilitate retrieval. To construct the KG, a Large Language Model (LLM) is leveraged to extract (subject-predicate-object) knowledge triples. These triples from a document are then aggregated to form a unified knowledge graph. Finally, a retrieval-optimized encoded adds supplementary edges to this graph by identifying and linking semantically similar entities (synonyms). The retrieval of the Veridical Layer thus follows HippoRAGV2 to utilize KGs towards more accurate retrieval. Statistics for this layer are detailed in Table 4.

| Layer Count | NarQA | EN.AQ | EN.MC | DetQA |
|-------------|-------|-------|-------|-------|
| Veridical   | # of Chunks | 4,416 | 26,465 | 47,074 | 2,406 |
|             | # of Entities | 33,810 | 292,170 | 401,969 | 30,969 |
|             | # of Triples | 51,012 | 372,339 | 576,595 | 33,696 |

### B.2 Episodic Layer

To construct the Episodic Layer, a sequence of text chunks is summarized. Since the chunk lengths can vary significantly, the choice of a sliding window size for this summarization presents a trade-off: a large window can be too coarse for short narratives, while a small window may be inefficient and fail to capture long-range dependencies in long-form content. Therefore, we dynamically adjust the window size $W$ according to the total number of text chunks, $N$, in the document. The specific heuristic is as follows:

- For short to medium-length narratives ($N \leq 200$ chunks): stepped window sizes (3, 5, 8, and 10) are used for documents up to 20, 50, 100, and 200 chunks respectively, aiming to preserve details for shorter contexts.
- For long narratives ($N > 200$): A logarithmic scaling function is applied to prevent the window from becoming excessively large. This sub-linear growth is intended to increase the summary scope for massive texts more slowly. The window size is calculated as follows to keep the window size between 10 to 20:

$$
W = \min(20, \max(10, \log_2(N \times 2)))
$$

For each window, the contained text chunks are concatenated and provided to an LLM agent (GPT-40-mini in our experiments). The agent is instructed to generate a concise summary that maintains chronological order and identifies key events and causal relationships. The resulting summaries are then collected and sorted by their original window order to form the nodes of the Episodic Layer.

### B.3 GraphRAG Experiments

GraphRAG is a structured-augmented RAG model similar to HippoRAGV2, which involves the construction of a comprehensive knowledge graph from source documents, which is then used to identify interconnected information for retrieval. However, its formulation requires heavy computation for building the retrieved index that includes multi-level nodes and relations and summaries.

We conducted preliminary experiments on a data subset to evaluate its viability. The results, detailed in Table 5, demonstrated that GraphRAG not only had significantly higher token consumption, but also attained lower scores compared to...
```

### --- Page 0010 ---

```markdown
| Dataset  | Factoid | Narrative | Inferential | Total |
|----------|---------|-----------|-------------|-------|
| EN.QA    | 224     | 84        | 43          | 351   |
| EN.MC    | 132     | 46        | 51          | 229   |

![Distribution of query types across the two datasets](assets/page_0010_img_1.png)

Each question is classified into one of the three categories based on the cognitive processes required to answer it, described in Section 4.4:

- **Factoid**: questions answerable by locating a single, specific piece of information from the text.
- **Narrative**: questions that demand an understanding of plot progression, requiring the aggregation of information from multiple text parts.
- **Inferential**: questions that necessitate reasoning beyond the literal text to understand implicit motivations or causal links.

The final distribution of the annotated query types is presented in Table 7.

### Table 5: Comparison of Performance, Token Usage, and Average Time for ComoRAG and GraphRAG.

| Performance Metrics | ComoRAG | GraphRAG |
|---------------------|---------|----------|
| F1 Score            | 33.61 (100.0%) | 14.20 (42.3%) |
| EM Score            | 21.43 (100.0%) | 8.00 (37.3%)  |

| Token Usage         | ComoRAG | GraphRAG |
|---------------------|---------|----------|
| Tokens              | 5.90M (100.0%) | 27.12M (459.7%) |

| Average Time Taken (sec) | ComoRAG | GraphRAG |
|--------------------------|---------|----------|
| Index                    | 291 (100.0%) | 1936 (665.3%) |
| Retrieve                 | 25 (100.0%)  | 29 (116.0%)  |

### Table 6: Hyperparameters for ComoRAG

| Hyperparameter          | Value                     |
|------------------------|---------------------------|
| LLM Agents             | GPT-4-mini                |
| Retrieval Model        | BGE-M3                    |
| Chunk Size             | 512 tokens                |
| Context Length         | 6,000 tokens              |
| Random Seed            | 0                         |
| Max Iterations         | 5                         |
| Max Probing Queries    | 3                         |
| Context Construction    | Proportional Allocation (8:2:2:1 ratio for V:S:E:H) |
| Mem-Fuse Threshold     | 0.5                       |

### C Query Types for Narratives

To facilitate a fine-grained analysis of our model’s performance, we (authors of this work) manually annotated the types of all questions in the EN.QA and EN.MC datasets.
```

### --- Page 0011 ---

```markdown
# D Prompting Templates

## Self-Probe
### Instruction Template for Probing Query Generation in Regulation Agent

**Role:**  
You are an expert in multi-turn retrieval-oriented probe generation. Your job is to extract diverse and complementary retrieval probes from queries to broaden and enrich subsequent corpus search results.

### Input Materials:
- **Original Query:** A question or information need that requires comprehensive information retrieval.
- **Context:** Available background information, partial content, or relevant summaries.
- **Previous Probes:** Previously generated probes from earlier iterations (if any).

### Task:
Based on the query and context, generate up to 3 non-overlapping retrieval probes that explore the query from distinct angles.

### Critical Requirements:
- **Semantic Differentiation:** Ensure new probes are semantically distinct from any previous probes provided.
- **Comprehensiveness:** New probes should cover different information dimensions not addressed by previous probes.
- **Relevance Maintenance:** All probes must remain directly relevant to answering the original query.

Each probe should:
- Target different information dimensions relevant to the query type:
  - **Character-related:** actions, motivations, relationships, timeline, consequences
  - **Event-related:** participants, causes, sequence, location, outcomes
  - **Object-related:** description, origin, usage, significance, connections
  - **Location-related:** events occurred, people involved, time periods, significance
- Expand search scope beyond obvious keywords to capture related content.
- Avoid semantic overlap with previous probes while maintaining query relevance.
- Be formulated as effective search terms or phrases.

### Probe Generation Strategy:
- **When previous probes exist:**
  1. Analyze Previous Coverage: Identify what semantic domains/angles have been covered.
  2. Gap Identification: Find unexplored but relevant information dimensions.
  3. Alternative Angles: Generate probes from different conceptual perspectives.
  4. Semantic Distance: Ensure sufficient semantic distance from previous probes.

- **When no previous probes exist:**
  - Probe 1: Direct elements explicitly mentioned in the query.
  - Probe 2: Contextual elements that might contain the answer.
  - Probe 3: Related concepts or alternative formulations.

### Output Format:
```json
{
  "probe1": "Content of probe 1",
  ...
}
```

### Notes:
- For simple queries, you may generate only 1–2 probes.
- If previous probes have covered most relevant angles, generate fewer new probes to avoid redundancy.
- Prioritize quality and semantic distinctiveness over quantity.
```

### --- Page 0012 ---

```markdown
# Instruction Template for Synthesized Cue Generation in Comprehension Agent

## Role
You are an expert narrative analyst capable of identifying, extracting, and analyzing key information from narrative texts to provide accurate and targeted answers to specific questions.

## Material
You are given the following:
1. A final objective to be resolved
2. A specific question that needs to be answered
3. Content: Direct excerpts, facts, and specific information from the narrative text

## Task
1. Carefully analyze the question to identify:
   - What type of information is being asked (character actions, locations, objects, events, motivations, etc.)
   - Which narrative elements are relevant to answering it
   - The specific details that need to be extracted
2. Systematically scan the content for:
   - Direct mentions of relevant elements (names, places, objects, events)
   - Contextual probes that help answer the question
   - Temporal and spatial relationships
   - Cause-and-effect connections
3. Analyze the identified information considering:
   - Explicit statements (directly stated facts)
   - Implicit information (suggested through context, dialogue, or narrative)
   - Logical connections between different narrative elements
   - Chronological sequence of events if relevant
4. Synthesize findings to construct a precise answer to the question.

## Response Format
Provide a structured analysis with up to 5 key findings:
```
Key Finding: <Most directly relevant information answering the question>  
Key Finding: <Supporting evidence or context>  
Key Finding: <Additional relevant details>  
Key Finding: <Clarifying information if needed>  
Key Finding: <Resolution of any ambiguities>  
```

---

# Instruction Template for Cue Generation in Integration Agent

## Role
You are an expert narrative synthesis specialist who excels at integrating and analyzing information from multiple narrative sources to create coherent and comprehensive insights.

## Input Material
- **Previous Analysis:** Results from earlier memory fusion operations that contain analyzed narrative information.
- **Current Query:** A question or information request that needs to be addressed.

## Task
1. Review and understand the previous memory fusion outputs:
   - Identify key narrative elements and their relationships.
```

### --- Page 0013 ---

```markdown
# PAGE_NAME: page_0013

1. **Note any established facts, character developments, or plot points.**
   - Recognize patterns and connections across different analyses.

2. **Analyze the current query in context:**
   - Determine how it relates to previously established information.
   - Identify any new aspects or angles that need to be addressed.
   - Consider how previous insights can inform the current response.

3. **Synthesize the information:**
   - Integrate relevant previous findings with new analysis.
   - Create a coherent narrative that addresses the current query.
   - Ensure continuity and consistency with previous analyses.
   - Highlight any new insights or developments.

4. **Provide a comprehensive response that:**
   - Directly answers the current query.
   - Incorporates relevant previous context.
   - Maintains narrative coherence.
   - Offers clear and insightful analysis.

**Response Format:**
Provide a cohesive narrative response that integrates previous insights with new analysis to address the current query. Focus on creating a flowing, well-structured response.

---

## Try-Answer Prompt Template for Query Resolution in QA Agent

**Role:**
You are an expert on reading and understanding books and articles.

**Task:**
Given the following detailed article, semantic summary, episodic summary from a book, and a related question with different options, you need to analyze which option is the best answer for the question.

**Inputs:**
- **Detail Article:** `{context}`
- **Summary by Semantic:** `{semantic.summary}`
- **Summary by Episodic:** `{episodic.summary}`
- **History Info:** `{history.info}`
- **Question:** `{question}`

**Limits:**
- Do not infer. Respond only based on the provided content strictly.
- Pick the choice only if you find at least 2 places that support the answer.

**Response Format:**
1. ### Content Understanding
   Start with a brief summary of the content in no more than three sentences. Begin this section with ### Content Understanding.

2. ### Question Analysis
   Based on the question, analyze and list all relevant items using a markdown list. Begin this section with ### Question Analyse.

3. ### Options Analysis
   Extract the key points related to 4 options, also using a markdown list. Begin this section with ### Options Analyse.

4. ### Final Answer
   Provide your final answer with a heading. Begin this section with ### Final Answer, followed by the best option in the format of [A] or [B] or [C] or [D]. If you cannot answer, give a failure signal: *.
```

### --- Page 0014 ---

```markdown
# E Case Study on Narrative Reasoning

## Input Data (No Options)
**Query:** Mrs. MacIntyre never writes letters, so what is the sudden purpose of buying ink?  
**Options:** [A] Response to the government’s letter [B] Sending a birthday card to my niece [C] Write a letter to the Sunday Comet newspaper. [D] Blurring Photos  

**ComoRAG's Choice Result**

### Memory Pool $M^{(0)}_{pool}$
- A page of the newspaper had a portion cut out...

### Step 1
**Probes $P^{(1)}$:**  
- What did Mrs. McGinty recognize in the clipping, and how did she try to profit from it?  
...  
**Retrieved Passages:**  
...The narrative offers insight into Miss Pamela Hosford’s role at the Sunday Comet, as she casually recalls receiving a vague letter from a woman about a photograph but fails to retrieve it...  
**Cues $C^{(1)}$:**  
- Key Finding: Mrs. McGinty usually had Joe help her reply to letters.;  
- Key Finding: Mrs. McGinty may have sent a letter to the newspaper;...

### Memory Pool $M^{(1)}_{pool}$
- A page of the newspaper had a portion cut out...  
- Mrs. MacIntyre sent a letter to the Sunday Comet...

### Step 2
**Probes $P^{(2)}$:**  
- What did Miss Pamela Hosford recall about a letter sent to the Sunday Comet, and what might it imply about Mrs. McGinty?  
...  
**Retrieved Passages:**  
...Miss Pamela Hosford’s role at the Sunday Comet, as she casually recalls receiving a vague letter from a woman about a photograph but fails to retrieve it... She liked eavesdropping, knew who was in the photo, and asked what we’d pay...  
**Cues $C^{(2)}$:**  
- Key Finding: Mrs. McGinty wanted to make some money from the photo;...

**Chosen:** C (Correct)  
(C) Write a letter to the Sunday Comet newspaper: Strong textual probes support this option. Mrs. McGinty cut out a part of the newspaper, recognized someone in a photo, asked about payment, and unusually bought ink—suggesting she intended to write to the paper. Final Answer: [C]

---

**Table 8:** Case Study on Narrative Reasoning. We present a case to demonstrate our model’s performance in long-context understanding, showing the final round of the Metacognitive Control Loop. Different colors are used to highlight the nature of the processed information: Blue is used for the key evidence that contributes to the correct answer, while Orange is used for the key cues.
```


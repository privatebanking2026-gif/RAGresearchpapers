# Latent Abstraction for Retrieval-Augmented Generation

### --- Page 0001 ---

```markdown
# Latent Abstraction for Retrieval-Augmented Generation

**Ha Lan N.T.*, Minh-Anh Nguyen*, Dung D. Le**  
Center for AI Research, VinUniversity, Vietnam  
{lan.nth, minh.na2, dung.ld}@vinuni.edu.vn  
*Equal contribution*

## Abstract

Retrieval-Augmented Generation (RAG) has become a standard approach for enhancing large language models (LLMs) with external knowledge, mitigating hallucinations, and improving factuality. However, existing systems rely on generating natural language queries at each hop and maintaining a strict architectural separation between retriever and generator, preventing them from leveraging the full representational capacity of the LLM. We propose LaNR (Latent Abstraction for RAG), a unified framework in which a single LLM jointly performs encoding, retrieval, and generation entirely within its own latent space. Rather than generating textual queries, LaNR produces dense retrieval vectors from the hidden states of a designated [PRED] token and uses them to match against encoded document representations from the same model. Furthermore, LaNR adaptively decides when sufficient evidence has been retrieved using a lightweight MLP control head over those same hidden states, enabling explicit token-level stopping reasoning. Extensive experiments on five QA benchmarks spanning single-hop and multi-hop settings demonstrate that LaNR outperforms existing RAG methods, while achieving improved inference efficiency through reduced number of retrieval calls and model integration.

## 1 Introduction

Retrieval-Augmented Generation (RAG) has emerged as a standard paradigm for enhancing LLMs with external knowledge, improving factuality, and mitigating hallucinations [2, 13, 29]. By retrieving relevant documents from external corpora and conditioning generation on this information, RAG systems enable BLLMs to access up-to-date and domain-specific knowledge beyond their parametric memory. Despite these advantages, existing RAG frameworks exhibit several limitations. First, they rely heavily on explicit text-based retrieval, where the model must generate natural language queries to interact with a separate retrieval module. Second, they enforce a strict architectural separation between the retriever and the generator, often requiring independently trained components or additional fine-tuning via reinforcement learning (RL) or supervised fine-tuning (SFT) LLMs to work with retrieval modules [33, 19, 37, 31]. These design choices introduce substantial computational overhead, limit the full utilization of the LLM’s shared representational capacity, and increase inference latency due to the need for explicit token-level generation at each step, illustrated in Figure 1.

Recent advances in latent reasoning suggest an alternative paradigm, where LLMs perform reasoning directly in hidden representation space rather than through fully verbalized intermediate steps [51, 32, 9, 44]. Such approaches demonstrate that latent trajectories in hidden states can encode rich reasoning processes without explicit token generation. However, this paradigm remains largely unexplored in the context of retrieval-augmented systems [9]. Extending latent reasoning to RAG is, however, non-trivial. First, unlike standard reasoning tasks, RAG lacks high-quality chain-of-thought supervision for constructing retrieval queries, making it difficult to distill text-based reasoning.
```

![Detailed description of the chart](assets/page_0001_img_1.png)

### --- Page 0002 ---

```markdown
![Comparison between conventional RAG and LAnR for multi-hop QA.](assets/page_0002_img_1.png)

Figure 1: Comparison between conventional RAG and LAnR for multi-hop QA. Conventional RAG performs explicit reasoning at each hop, including generating intermediate text, forming search queries, and deciding whether to continue retrieval. In contrast, LAnR operates in latent space: a special token [PRED] produces query vectors from hidden states, while a lightweight MLP controls the retrieval process, enabling more efficient and integrated reasoning.

![Comparison of inference time, generated tokens, and Exact Match accuracy between prior RAG methods and LAnR.](assets/page_0002_img_2.png)

Figure 2: Comparison of inference time, generated tokens, and Exact Match accuracy between prior RAG methods and LAnR. Latent retrieval reduces latency and token generation while maintaining strong performance.

into latent representations [15, 43]. Second, existing RAG architectures typically rely on separate embedding models for retrieval, which necessitates explicit natural language queries. Consequently, integrating both reasoning traces and retrieval queries into a shared latent space requires a unified model that simultaneously supports retrieval and generation.

To address these challenges, we propose a unified latent abstraction retrieval-augmented generation (LAnR) framework, wherein retrieval is conducted directly within the LLM's internal representation space. Instead of generating textual queries, the model produces latent query vectors derived from the hidden states of designated special tokens [PRED]. These vectors are used to retrieve documents from a vector database constructed using representations from the same LLM, enabling tight integration between retrieval and generation, detailed in Section 3. To support multi-hop reasoning, we further introduce a lightweight multi-layer perceptron (MLP) [28] that predicts whether additional retrieval steps are necessary. This design is motivated by empirical observations that uncertainty signals, such as entropy in the token distribution, correlate with the need for further information acquisition, illustrated in Appendix C.

Extensive experiments across multiple benchmarks demonstrate that our approach achieves superior performance compared to state-of-the-art RAG methods, while significantly reducing inference latency due to minimal text generated, illustrated in Figure 2 and eliminating the need for serving.
```

### --- Page 0003 ---

```markdown
# Page 3

a separately trained dense retriever. We provide our code at the anonymous repository https://anonymous.4open.science/r/LAnR-48EF/. We summarize our main contributions as follows:

1. We propose a novel latent RAG framework that leverages the internal representation space of an LLM to jointly perform document encoding, retrieval, and generation, thereby unifying these components within a single architecture.

2. We design an implicit retrieval control mechanism, implemented as a lightweight auxiliary head, that use the LLM's hidden representation to adaptively determines the necessity of additional retrieval without relying on explicit text-based reasoning.

3. We introduce a new approach for constructing retrieval query vectors directly from the LLM's latent representations, eliminating the need for intermediate natural language query generation.

## 2 Latent Query Construction

We begin by investigating latent query construction, where the LLM bypasses explicit query generation and instead produces a dense retrieval vector directly from its hidden representations. To enable this capability, the LLM is trained to internalize query formulation, allowing it to infer search intent from the input and map it directly into a retrieval vector. Formally, let $x = (x_1, \ldots, x_T)$ denote an input query token sequence and $\mathcal{M}$ an autoregressive language model with last-layer hidden states $h_t \in \mathbb{R}^d$. We append a designated token $[PRED]$ to the input, forming $\bar{x} = (x_1, \ldots, x_T, [PRED])$. Through causal self-attention, its hidden state attends to the full preceding context, yielding a latent query vector:

$$
q = h_{[PRED]} \in \mathbb{R}^d. \tag{1}
$$

Optionally, $N \geq 1$ consecutive $[PRED]$ tokens can be appended, allowing the model to construct the query representation over multiple latent steps, which promotes higher-level abstraction and extends the LLM's latent reasoning process; we denote $N=1$ and study this design choice in Appendix E.

Unlike prior LLM-based encoders [39, 40], where the query vector encodes a fully-formed retrieval query, here $q$ is produced by the model’s own reasoning over context. This distinction becomes essential in the multi-turn setting, detailed in Section 3, where intermediate sub-queries have no canonical textual form and must be inferred from the running context of previously retrieved evidence. Each representation of the document $D_i$ are obtained from the same model $\mathcal{M}$ via last-token pooling [10, 13]:

$$
d_i = h_{b}^{last} \in \mathbb{R}^d. \tag{2}
$$

Embedding queries and documents within a shared representation space induced by the same parameters eliminates the need for a separate retriever and preserves the model’s generative capability for downstream answer production. We train the $[PRED]$ representation with a standard contrastive objective [38, 5]:

$$
L_{CL} = - \log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q, d^+)/\tau) + \sum_{j=1}^{N^-} \exp(\text{sim}(q, d_j^-)/\tau)}, \tag{3}
$$

where $d^+$ is a positive document, $\{d_j^-\}_{j=1}^{N^-}$ are negatives mined from the model’s own retrieval errors and periodically refreshed following ANCE [42], $\text{sim}(\cdot, \cdot)$ is cosine similarity, and $\tau$ is a temperature parameter. This setup gives us a query vector at any point in the model’s forward pass. Building on this foundation, Section 3 introduces a multi-turn training objective designed to leverage both properties effectively.

## 3 End-to-End Framework and Training Objective

Building on the latent query construction in Section 2, we extend the framework from single-step implicit retrieval to a multi-turn setting. The central challenge is to determine when to trigger additional retrieval and what to retrieve at each step, while relying solely on latent representations in hidden space without generating intermediate text tokens. We address both through: (1) an MLP-based retrieval control head that decides whether further retrieval is necessary, and (2) an adaptive
```

### --- Page 0004 ---

```markdown
![Overview of LaNr. Queries are injected into the LLM and combined with a [PRED] token to form a latent query from hidden representations. This latent query is used for retrieval and to decide whether further retrieval is needed via a lightweight MLP Retrieval Control Head. The LLM then generates the answer from the retrieved context.](assets/page_0004_img_1.png)

contrastive target mechanism that dynamically updates the retrieval objective based on remaining unretrieved evidence.

At each retrieval turn $r = 1, 2, \ldots, R$, the model constructs a latent query from the current context, which now includes both the original question and previously retrieved documents, and decides whether to retrieve again or to proceed with answer generation. Let $c^{(r)}$ denote the accumulated context at turn $r$:

$$
c^{(r)} = (x, D^{(1)}, D^{(2)}, \ldots, D^{(r-1)}),
$$

where $D^{(s)}$ denotes the set of top-$K$ documents retrieved at turn $s$. At each turn, we append [PRED] to $c^{(r)}$ and extract the latent query vector: $q^{(r)} = c^{(r)}_{[PRED]} \in \mathbb{R}^d$, which is used both for retrieval via similarity matching against the document embeddings, detailed in Equation 2 and as input to the retrieval control head described below. The process repeats until the control head signals termination or a maximum number of turns $R$ is reached, after which the model generates the final answer conditioned on the full accumulated context $c^{(R+1)}$.

MLP-based Retrieval Control Head. An additional component of our framework is a lightweight MLP head that determines whether the currently retrieved evidence is sufficient to answer the query, or whether additional retrieval is required. At each turn $r$, the Retrieval Control Head $f_0 takes $q^{(r)}$ as input and produces a binary decision:

$$
\hat{y}^{(r)} = f_0(q^{(r)}) = \sigma (MLP(q^{(r)})) \in [0, 1],
$$

where $\sigma$ is the sigmoid function. A prediction $\hat{y}^{(r)} < 0.5$ indicates that all necessary evidence has been retrieved and the model should proceed to generation; $\hat{y}^{(r)} \geq 0.5$ signals that additional retrieval is needed. The training label is derived from the model's own retrieval state. Let $P$ denote the full set of positive (gold) documents required to answer the query, and $R^{(r)} = D^{(1)} \cup \ldots \cup D^{(r)}$ the documents retrieved up to turn $r$. The ground-truth label is:

$$
y^{(r)} = \begin{cases}
0, & \text{if } P \subset R^{(r)}, \\
1, & \text{otherwise.}
\end{cases}
$$

i.e., the label is 0 (stop) when all positive documents have been successfully retrieved, and 1 (continue) otherwise. This self-supervised formulation requires no external oracle: the signal is generated entirely.
```

### --- Page 0005 ---

```markdown
from the model’s own retrieval accuracy at training time. The control head is trained with binary cross-entropy:

$$
L_{ctrl} = - \sum_{r=1}^{R} \left[ y^{(r)} \log \hat{y}^{(r)} + (1 - y^{(r)}) \log(1 - \hat{y}^{(r)}) \right]. \tag{6}
$$

In the multi-turn setting, naively using the same contrastive target at every turn is suboptimal: once a positive document has been retrieved, it should no longer serve as the target for subsequent queries. At turn $r$, the set of leftover positive documents is: $P(r) = P \setminus R^{(r-1)}$, where $R^{(0)} = \emptyset$. The contrastive loss at turn $r$ uses a positive document sampled from $P(r)$:

$$
L_{CL}^{(r)} = - \log \frac{\exp(\text{sim}(q^{(r)}, d^{(r)})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(q^{(r)}, d_{j}^{(r)})/\tau)} \tag{7}
$$

where $d^{(r)}$ is the embedding of a document drawn from $P(r)$. This iterates each successive turn toward the missing evidence, avoiding redundant retrieval. The full training loss combines next-token prediction, multi-turn contrastive retrieval, and the control head objective:

$$
L = \sum_{r=1}^{R} \left[ \lambda L_{CL}^{(r)} + \lambda L_{NTP} + \mu L_{ctrl} \right]. \tag{8}
$$

where $\lambda$ and $\mu$ are hyperparameters balancing the three objectives. $L_{NTP}$ denotes the standard next-token prediction loss, which maintains the generative capability of the LLM. We apply loss masking to retrieved tokens and [PRE] tokens, so that the optimization objective is computed only over tokens generated by the LLM, excluding retrieved content from gradient updates, following prior works [19, 33].

In each step, the model performs iterative retrieval beginning from the input query. At each step, the model appends the [PRE] token, extracts the latent query representation $q^{(r)}$, and feeds it into the retrieval control head. If the controller predicts that additional retrieval is needed ($\hat{y}^{(r)} \geq 0.5$), the model retrieves the top-$K$ documents, appends them to the current context, and continues the retrieval process. Otherwise, the model stops retrieving and generates the final answer conditioned on the accumulated evidence. To prevent unbounded retrieval, we impose a maximum limit of $R = T$ retrieval rounds. Additional qualitative examples are provided in Appendix H.

## 4 Experiments

In this section, we investigate the following research questions (RQs):

**RQ1.** How effective is LaNR compared to conventional agentic RAG systems in answer quality across single-hop and multi-hop benchmarks, and can its implicit retrieval match text embedding models in retrieval accuracy?

**RQ2.** Does LaNR’s retrieval control head adaptively allocate search hops based on query complexity, and does each retrieval hop find relevant evidence more effectively than text-based iterative search?

**RQ3.** Can LaNR reduce inference time by minimizing retrieval iterations and improving efficiency through more abstract query representations?

**RQ4.** What is the individual contribution of each training objective ($L_{NTP}, L_{CL}, L_{ctrl}$) and the retrieval control head to LaNR’s end-to-end performance?

### 4.1 Experimental Setup

**Datasets.** We evaluate our method on five open-domain QA benchmarks, including two single-hop datasets, Natural Questions (NQ) [23] and TriviaQA [20], and three multi-hop datasets: HotpotQA [46], 2WikiMultiHopQA (2Wiki) [16] and MusiQue [36]. We report Exact Match (EM) as the primary metric for answer generation. For retrieval evaluation on datasets with ground-truth documents, we use Recall as the main metric.
```

### --- Page 0006 ---

```markdown
| **Table 1:** (RQ1) Accuracy comparison of LaNr and baselines on QA benchmarks. Bold and underline denote best and second-best results, respectively. "-Instruct" and "-Base" indicate the corresponding Qwen2.5-3B backbone variants. |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| **Methods**    | **Single-Hop QA** |                |                |                | **Multi-Hop QA** |                |
|                | NQ             | TriviaQA       | HotpotQA       | 2Wiki          | MusiQue        | Avg.           |
| **w/o Retrieval** |                |                |                |                |                |                |
| Direct Generation | 0.106         | 0.208          | 0.149          | 0.244          | 0.020          | 0.134          |
| SFT            | 0.249         | 0.292          | 0.186          | 0.248          | 0.044          | 0.176          |
| **w/ Single-Hop Retrieval** |                |                |                |                |                |                |
| Naïve RAG [24] | 0.348         | 0.544          | 0.255          | 0.226          | 0.047          | 0.270          |
| **Multi-Hop Retrieval** |                |                |                |                |                |                |
| Search-1 [26]  | 0.238         | 0.472          | 0.221          | 0.218          | 0.054          | 0.255          |
| IRCoT [37]     | 0.111         | 0.312          | 0.164          | 0.171          | 0.067          | 0.181          |
| ReSearch-Instruct [7] | 0.365         | 0.571          | 0.351          | 0.272          | 0.095          | 0.331          |
| ReSearch-Base [7] | 0.427         | 0.597          | 0.305          | 0.272          | 0.074          | 0.319          |
| Search-R1-Instruct [19] | 0.397         | 0.565          | 0.331          | 0.310          | 0.124          | 0.336          |
| Search-R1-Base [19] | 0.421         | 0.583          | 0.297          | 0.274          | 0.066          | 0.312          |
| AutoRefine-Instruct [33] | 0.436         | 0.597          | 0.404          | 0.380          | 0.169          | 0.396          |
| AutoRefine-Base [33] | 0.467         | 0.620          | 0.405          | 0.393          | 0.157          | 0.405          |
| LaNr-Instruct  | 0.460         | 0.613          | 0.419          | 0.408          | 0.193          | 0.413          |
| LaNr-Base      | 0.455         | 0.610          | 0.417          | 0.402          | 0.187          | 0.414          |

**Baselines.** We compare LaNr against three categories of methods: (1) generation without retrieval, including direct LLM generation and supervised fine-tuning (SFT) without retrieval; (2) single-hop retrieval methods, such as naive RAG [24] that retrieves once using the input query; and (3) multi-hop retrieval approaches, including agenting and iterative systems such as Search-1 [26], IRCoT [37], Search-R1 [19] and AutoRefine [31].

**Implementation Details.** To simulate a realistic retrieval setting, we remove ground-truth content from the QA datasets and instead utilize the 2018 Wikipedia dump [21] as the external knowledge source. By default, retrieval returns the top-$k$ documents at each step, with $k = 3$ following the setting of Search-R1 for fair comparison. All models are trained on the combined NQ and HotpotQA dataset, following the setup of prior works [19, 31]. Detailed training configurations are described in Appendix B.

**4.2 Main Results**

**Overall Performance (RQ1).** Table 1 reports EM accuracy across five QA benchmarks covering both single-hop and multi-hop reasoning. Retrieval-free methods perform poorly, while Naive RAG improves single-hop QA but remains ineffective on compositional tasks such as HotpotQA, 2Wiki, and MusiQue. Among multi-hop approaches, Search-R1 and AutoRefine provide strong baselines, but LaNr consistently achieves the best overall performance, especially on challenging multi-hop benchmarks. LaNr-Instruct reaches 0.419 EM on HotpotQA, 0.408 on 2Wiki, and 0.193 on MusiQue, outperforming AutoRefine-Instruct across all three datasets. Although AutoRefine remains competitive on single-hop benchmarks, LaNr achieves the highest average EM overall, indicating a stronger balance between retrieval and reasoning. Despite being trained only on NQ and HotpotQA, LaNr also generalizes effectively to unseen datasets in a zero-shot setting.

**Retrieval Quality vs. Text Embedding Retrievers.** Table 2 compares LaNr-Instruct with sparse and dense retrievals across three multi-hop QA benchmarks. While BM25, BGE, and ES rely on fixed top-$k$ retrieval and are trained solely for retrieval, LaNr-Instruct jointly performs adaptive retrieval and generation. Despite this joint objective, LaNr-Instruct achieves competitive retrieval recall using substantially fewer retrieved documents. On HotpotQA and 2WikiQA, it attains recalls of 0.840 and 0.715 with only ~5 retrieved documents, matching or exceeding the Recall@5–10 performance of specialized retrievers. On the more challenging MusiQue benchmark, LaNr-Instruct achieves 0.516 recall with ~7.9 documents, outperforming BM25 and remaining competitive with dense retrievers. In contrast, the single-step LaNr-Instruct-static variant consistently underperforms the adaptive version, highlighting the benefit of iterative retrieval with learned control.
```

### --- Page 0007 ---

```markdown
| Dataset     | Method                | R@1   | R@3   | R@5   | R@10  | Recall (budget)   |
|-------------|-----------------------|-------|-------|-------|-------|--------------------|
|             | BM25 [30]            | 0.401 | 0.603 | 0.670 | 0.751 | —                  |
|             | BGE [5]              | 0.477 | 0.782 | 0.832 | 0.880 | —                  |
|             | E5 [38]              | 0.462 | 0.773 | 0.826 | 0.875 | —                  |
|             | LaNr-Instruct-static  | 0.451 | 0.769 | 0.800 | 0.830 | 0.840±0.012 (−5.5 docs) |
|             | LaNr-Instruct         | —     | —     | —     | —     | —                  |
|             | BM25 [30]            | 0.360 | 0.555 | 0.605 | 0.656 | —                  |
|             | BGE [5]              | 0.411 | 0.643 | 0.680 | 0.715 | —                  |
|             | E5 [38]              | 0.415 | 0.666 | 0.687 | 0.717 | —                  |
|             | LaNr-Instruct         | 0.407 | 0.641 | 0.684 | 0.713 |  —                 |
|             | LaNr-Instruct         | —     | —     | —     | —     | 0.715±0.009 (−5.1 docs) |
|             | BM25 [30]            | 0.216 | 0.310 | 0.350 | 0.400 | —                  |
|             | BGE [5]              | 0.280 | 0.435 | 0.500 | 0.573 | —                  |
|             | E5 [38]              | 0.290 | 0.417 | 0.454 | 0.503 | —                  |
|             | LaNr-Instruct         | 0.265 | 0.393 | 0.449 | 0.487 | 0.516±0.011 (−7.9 docs) |

![Search Count Distribution](assets/page_0007_img_1.png)

![Step-wise Gold Recall](assets/page_0007_img_2.png)

![Productive Hop Rate](assets/page_0007_img_3.png)

### Latent RAG Effectiveness (RQ2)
To understand how LaNr retrieves rather than just how well, we analyse the search-level behaviour of the retrieval control head across all five evaluation benchmarks.

#### Adaptive search count.
Figure 4(a) shows the distribution of search calls per query across all five benchmarks. Single-hop datasets are dominated by a single search call, while compositional benchmarks require progressively more, showing the control head adapts to query complexity rather than applying a fixed budget.

#### Step-wise retrieval quality.
Figure 4(b) compares cumulative gold recall across retrieval heads between LaNr and AutoRefine on the three multi-hop benchmarks. LaNr consistently achieves higher recall at every step, with the advantage appearing from the first hop and widening on more compositional datasets such as 2Wiki and MuSiQue. This suggests that latent query vectors better capture residual
```

### --- Page 0008 ---

```markdown
# Page 8

## Information Needs

Figure 4(c) reports the productive hop rate: the fraction of additional search calls triggered by a "continue" decision that successfully retrieved at least one new gold document. On Wiki, 75% of additional calls are productive, confirming that the control heads follow-up searches only when evidence is genuinely incomplete. On MuSiQue, the rate is lower at 23%, reflecting the difficulty of locating the precise gold documents in highly compositional three-hop and four-hop chains where missing any single query may fail to bridge the full reasoning gap.

### Figure 5: (RQ3) Per-dataset EM distributions for LaNR, AutoRefine, and Search-R1. LaNR achieves competitive or higher EM with the fewest retrieval calls and consistently narrow variance.

![Per-dataset EM distributions for LaNR, AutoRefine, and Search-R1](assets/page_0008_img_1.png)

## Inference Efficiency of Latent Retrieval (RQ3)

Figure 5 directly tests the core efficiency claim of LaNR: each subplot shows the EM distribution for all three methods on a single benchmark, with the colored co-click labels reporting each method’s average number of retrieval calls. Across all five datasets, LaNR issues the fewest retrieval calls (1.54 – 2.47), yet its median EM matches or exceeds AutoRefine (2.02 – 2.13 calls) and Search-R1 (2.01 – 3.13 calls). The narrow interquartile ranges confirm that LaNR's accuracy is stable across evaluation instances rather than driven by a subset of easy queries. Notably, on the multi-hop benchmarks, LaNR's gap over Search-R1 widens substantially despite Search-R1 issuing nearly twice as many retrievals, indicating that latent query vectors retrieve more relevant evidence per call than token-based queries.

The token and latency costs are quantified in Figure 2. LaNR generates on average only ~5 output tokens per query, roughly 30x fewer than Search-R1 (~163) or AutoRefine (~168), because the retrieval signal is encoded as a latent vector rather than verbalized text. This reduction translates directly to wall-clock savings: LaNR completes a query in 0.81 – 1.61 s compared to 2.01 – 2.77 s for Search-R1 and 2.15 – 2.34 s for AutoRefine, a 1.5 – 2.7x speedup across all benchmarks.

### Key Takeaway

By replacing explicit query generation with compact latent vectors and an adaptive retrieval controller, LaNR performs fewer but more effective search calls, achieves higher cumulative gold recall, and maintains stable EM performance with lower computational cost.

## 4.3 Ablation Studies

### Component Ablation (RQ4)

Table 3 isolates the contribution of three design choices: the MLP retrieval control head, NTP loss masking, and the NTP loss weight $\lambda_{NTP}$. Removing the control head and replacing it with a fixed 5-document budget leaves single-hop performance essentially unchanged but causes large drops on multi-hop tasks, confirming that adaptive retrieval is the primary driver of multi-hop gains. Removing NTP loss masking introduces a moderate but consistent degradation on all three multi-hop benchmarks, suggesting the masking prevents the model from conflating retrieval-query generation with next-token prediction over retrieved content. Reducing $\lambda_{NTP}$ to 0.5 incurs a small performance drop, while setting it to 0 collapses performance to near zero, demonstrating that the NTP objective is indispensable for answer generation.
```

### --- Page 0009 ---

```markdown
| Table 3: Component ablation study (RQ4). $\Delta$Avg denotes the absolute decrease in average performance relative to the full LaNr model. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Variant | NQ | TriviaQA | HoptotQA | 2Wiki | MusiQure | Avg | $\Delta$Avg |
| LaNr-Instruct | 0.460 | 0.613 | 0.419 | 0.408 | 0.193 | 0.418 | — |
| w/ ctrl head | 0.459 | 0.592 | 0.342 | 0.266 | 0.095 | 0.350 | -0.068 |
| w/o NTP loss masking | 0.458 | 0.582 | 0.395 | 0.371 | 0.182 | 0.398 | -0.020 |
| $\lambda_{NTP} = 0.5$ | 0.455 | 0.578 | 0.411 | 0.402 | 0.189 | 0.407 | -0.011 |
| $\lambda_{NTP} = 0$ | 0.005 | 0.007 | 0.004 | 0.000 | 0.000 | 0.003 | -0.415 |

![EM at various fractions of training data used (5\% to 100\%). Most of the performance gain is captured by the first 50\% of the data.](assets/page_0009_img_1.png)

### Additional Experimental Results.
Figure 6 shows EM as a function of training-set fraction for both LaNr-3B and LaNr-7B across all five benchmarks. The learning curve steeply in the first 10-20\% of data, where the four-dataset average EM climbs from 0.13 to 0.38. Notably, 2WikiMultiHopQA starts near zero at 5\% and gains most rapidly between 10-50\%, reflecting the higher sample complexity of multi-hop compositional reasoning compared to single-hop benchmarks (NQ, TriviaQA), which plateau earlier. Beyond 50\% gains flatten to within 3-5\% of the full-data training budgets. The 7B variant yields a consistent improvement over 3B across all data fractions, indicating that model scale and training data contribute largely independently to final performance.

Additional analyses are provided to further examine LaNr from multiple perspectives, including: (i) a detailed evaluation of the Retrieval Control Head, Appendix E.1; (ii) model scaling experiments spanning 3B to 7B backbones, Appendix E.2; (iii) statistical significance analysis across multiple random seeds, Appendix E.3; and (iv) the effect of varying the number of [PRED] tokens, where larger token counts improve abstraction capacity, Appendix E.4. Together, these findings further demonstrate the robustness, scalability, and retrievability effectiveness of LaNr across diverse QA benchmarks.

## 5 Conclusion
We introduced latent abstraction retrieval-augmented generation (LaNr), a unified framework that performs retrieval and reasoning directly in the latent space of a single LLM. By replacing explicit text-based query generation with latent query vectors derived from hidden states, LaNr eliminates the need for a separate retrieval and reduces reliance on token-level reasoning. We further proposed a lightweight retrieval control head that adaptively determines when additional retrieval is required, enabling efficient multi-hop reasoning without explicit intermediate text. Empirical results across multiple benchmarks demonstrate that LaNr achieves competitive performance compared to existing RAG systems while significantly reducing inference latency. These findings highlight the potential of latent-space reasoning as a scalable alternative to conventional RAG pipelines.
```

### --- Page 0010 ---

```markdown
# References

| No. | Citation                                                                                                                                                                                                 |
|-----|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [1] | Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. A survey on rg with llms. *Procedia computer science*, 246:3781–3790, 2024.                                                        |
| [2] | Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. Retrieval-based language models and applications. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)*, pages 41–46, 2023. |
| [3] | Akari Asai, Zeqiu Wu, Yizhong Wang, Aviyup Sil, and Hanneah Hajishirzi. Self-rap: Learning to retrieve, generate, and critique through self-reflection. In *The Twelfth International Conference on Learning Representations*, 2023. |
| [4] | Parisahed BehnamGhadir, Vaibhav Adharka, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. LlmZee: Large language models are secretly powerful text encoders, 2022. URL https://arxiv.org/abs/2404.5961, 2024. |
| [5] | Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. *arXiv preprint arXiv:2402.03216*, 45(5), 2024. |
| [6] | Mingyang Chen, Linzhung Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z Pan, Wen Zhang, Huajun Chen, et al. Learning to reason with search for llms via reinforcement learning. *arXiv preprint arXiv:2503.19470*, 2025. |
| [7] | Mingyang Chen, Linzhung Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z Pan, Wen Zhang, Huajun Chen, et al. Learning to reason with search for llms via reinforcement learning. *arXiv preprint arXiv:2503.19470*, 2025. |
| [8] | Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning. *arXiv preprint arXiv:2505.16782*, 2025. |
| [9] | Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning. *arXiv preprint arXiv:2505.16782*, 2025. |
| [10]| Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishui Zhang, and Dongyan Zhao. xrac: Extreme text compression for retrieval-augmented generation with token. *Advances in Neural Information Processing Systems*, 37:109487–109516, 2024. |
| [11]| Debrup Das, Sam O’Nuaillan, and Razieh Rahimi. Radar: Reasoning-aware dense retrieval models. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pages 19981–20008, 2025. |
| [12]| Jingcheng Deng, Zhongtao Jiang, Liang Pang, Zihao Wei, Liwei Chen, Kun Xu, Yang Song, Huwei Shen, and Xueqi Cheng. Following the autoregressive nature of ilm embeddings via compression and alignment. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pages 12672–12688, 2025. |
| [13]| Tao Ge, Jing Hu, Lei Wang, Xin Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model. *arXiv preprint arXiv:2307.06945*, 2023. |
| [14]| Sachin Goyal, Zewad J. Rizvi, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with paused tokens. *arXiv preprint arXiv:2310.02226*, 2023. |
| [15]| Shibo Hao, Sainbayar Sukhabaatar, Djiia Su, Xian Li, Zhiteng Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. *arXiv preprint arXiv:2412.06769*, 2024. |
| [16]| Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In *Proceedings of the 28th International Conference on Computational Linguistics*, pages 6609–6625, 2020. |
```

### --- Page 0011 ---

```markdown
# References

[17] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Diweidi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 7969–7992, 2023.

[18] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O Arik. Long-context llms meet rag: Overcoming challenges for long inputs in rag. arXiv preprint arXiv:2410.05983, 2024.

[19] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

[20] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.

[21] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 6769–6781, 2020.

[22] Chaeeun Kim and Seungone Kim. Freeson: Retriever-free retrieval-augmented reasoning via cross-traversing mats. arXiv preprint arXiv:2505.16409, 2025.

[23] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.

[24] Patrick Lewis, Ethan Perez, Aleksandr Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

[25] Jindong Li, Yali Fu, Li Fan, Jiahong Liu, Yao Shu, Chengwei Qin, Menglin Yang, Irwin King, and Rex Ying. Implicit reasoning in large language models: A comprehensive survey. arXiv preprint arXiv:2509.02350, 2025.

[26] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhiqing Dou. Search-ol: A genetic search-enhanced large reasoning models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5420–5438, 2025.

[27] Zhijie Nie, Zhangchi Feng, Mingxin Li, Cunwang Zhang, Yanzhao Zhang, Dingkun Long, and Richong Zhang. When text embedding meets large language model: a comprehensive survey. arXiv preprint arXiv:2412.09165, 2024.

[28] Leonardo Noriega. Multilayer perceptron tutorial. School of Computing, Staffordshire University, 4(5):444, 2005.

[29] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgy, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316–1331, 2023.

[30] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond, volume 4. Now Publishers Inc, 2009.

[31] Timo Schick, Jane Diweidi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36: 68539–68551, 2023.
```

### --- Page 0012 ---

```markdown
| Reference                                                                                                           | Year |
|---------------------------------------------------------------------------------------------------------------------|------|
| [32] Dachuan Shi, Abedelkadir Aisi, Keying Li, Xiangchi Yuan, Leyan Pan, Wenke Lee, and Wen Xiao. Swireasoning: Switch-thinking in latent and explicit for pareto-superior reasoning llms. arXiv preprint arXiv:2510.05609, 2025. | 2025 |
| [33] Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An Zhang, and Xiang Wang. Search and refine during think: Facilitating knowledge refinement for improved retrieval-augmented reasoning. arXiv preprint arXiv:2505.11277, 2025. | 2025 |
| [34] Levy Silva and Luciano Barbosa. Improving dense retrieval models with ilm augmented data for dataset search. Knowledge-based systems, 294:111740, 2024. | 2024 |
| [35] Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Repetition improves language model embeddings. arXiv preprint arXiv:2402.15449, 2024. | 2024 |
| [36] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022. | 2022 |
| [37] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 10014–10037, 2023. | 2023 |
| [38] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533, 2022. | 2022 |
| [39] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. In Proceedings of the 26th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11897–11916, 2024. | 2024 |
| [40] Xiaoqiang Wang, Suyuchen Wang, Yun Zhu, and Bang Liu. System-1.5 reasoning: Traversal in language and latent spaces with dynamic shortcuts. arXiv preprint arXiv:2505.18692, 2025. | 2025 |
| [41] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. | 2022 |
| [42] Lee Xiong, Chenyuan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808, 2020. | 2020 |
| [43] Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot: Soft chain-of-thought for efficient reasoning with llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23336–23351, 2025. | 2025 |
| [44] Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot++: Test-time scaling with soft chain-of-thought reasoning. arXiv preprint arXiv:2505.11484, 2025. | 2025 |
| [45] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hu, Bo Zheng, Bowen Yu, Chang Gao, Cheng Huang, Chenxu Lu, et al. Qwen's technical report. arXiv preprint arXiv:2505.09388, 2025. | 2025 |
| [46] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018. | 2018 |
| [47] Shunya Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In the eleventh international conference on learning representations, 2022. | 2022 |
```

### --- Page 0013 ---

```markdown
# PAGE_NAME: page_0013

[48] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

[49] Xinlei Yu, Zhangquan Chen, Yongbo He, Tianyu Fu, Cheng Yang, Chengming Xu, Yue Ma, Xiaobin Hu, Zhe Cao, Jie Xu, et al. The latent space: Foundation, evolution, mechanism, ability, and outlook. arXiv preprint arXiv:2604.02029, 2026.

[50] Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin, Dong Wang, Xuanhui Wang, and Michael Bendersky. Inference scaling for long-context retrieval augmented generation. arXiv preprint arXiv:2410.04343, 2024.

[51] Zhenrui Yue, Bowen Jin, Huimin Zeng, Honglei Zhuang, Zhen Qin, Jinsung Yoon, Lanyu Shang, Jiawei Han, and Dong Wang. Hybrid latent reasoning via reinforcement learning. arXiv preprint arXiv:2505.18544, 2025.

## A Related Works

### A.1 Latent Reasoning

A key limitation of dominant reasoning paradigms such as explicit chain-of-thought (CoT) [41, 48, 14, 49] lies in their reliance on discrete token generation during inference. In standard CoT decoding, the model commits to a single token at each step, sampled from the predicted distribution. While this process enhances interpretability by verbalizing intermediate reasoning steps, it inherently collapses the full probability distribution into a single trajectory, discarding uncertainty and eliminating alternative reasoning paths that may be informative. To address this limitation, recent work has explored latent reasoning [15, 43, 22, 51, 40], where reasoning unfolds directly in the continuous hidden-state space rather than through discrete text. This paradigm offers two primary advantages over CoT: (1) increased representational capacity per step, as continuous vectors can encode substantially richer information than individual tokens [8]; and (2) the ability to implicitly preserve multiple reasoning hypotheses without prematurely collapsing them into a single token sequence [25]. However, despite these advances, prior work on latent reasoning has primarily focused on improving reasoning quality in standalone language modeling settings. Its application to RAG remains largely unexplored. In particular, existing methods do not address how latent representations can be leveraged for retrieval itself, nor how they can guide adaptive retrieval decisions.

### A.2 Retrieval Augmented Generation

RAG enhances LLMs by incorporating external knowledge sources to mitigate hallucinations and to address knowledge gaps [24, 50, 1]. A central challenge in RAG systems lies in determining when and how to retrieve relevant information, as naive single-step retrieval often introduces irrelevant or insufficient context [17, 18]. Early approaches employ supervised fine-tuning (SFT) to train models for query generation and retrieval integration [3, 37, 47, 31]; however, these methods rely on high-quality annotated trajectories and often exhibit limited generalization to out-of-distribution settings. Leveraging the inherent structure of LLMs, an alternative direction is to fine-tune them as document encoders for text embedding [4, 19, 35, 12], or to use LLMs to generate synthetic data to support embedding training [11, 27, 34]. More recent work explores iterative and adaptive retrieval strategies, where models interleave reasoning and retrieval in a multi-step process, commonly described as "search-during-think" [19, 33, 6, 22]. However, existing approaches primarily operate in the discrete text space, relying on explicit query generation and token-level signals to control retrieval. Moreover, they often lack mechanisms for directly assessing retrieval sufficiency or refining retrieved evidence beyond surface-level interactions.

## B More Implementation Details

### Training Details

We train LaNR using full-parameter fine-tuning on 2 NVIDIA H100 (80GB) GPUs. The training data is constructed by combining Natural Questions (NQ) [23] and HoptotQA [46], following a consistent setup across LaNR and all training-based baselines. We employ fully
```

### --- Page 0014 ---

```markdown
| Table 4: Primary hyperparameters used by LAnR. |
|-------------------------------------------------|
| Hyper-parameter                | Value          |
|--------------------------------|----------------|
| Training Batch Size            | 256            |
| Chunk size                     | 100 words      |
| Micro Training Batch Size      | 32             |
| Learning Rate                  | $5 \times 10^{-5}$ |
| Validation Batch Size          | 256            |
| Total Training Steps           | 250            |
| Max Search Actions             | 4              |
| $\alpha$ for NTP Loss         | 1              |
| $\mu$ for Retrieval Control Head Loss | 1      |

| Table 5: Statistics of the datasets used in this paper. |
|---------------------------------------------------------|
| NQ       | TriviaQA | HotpotQA | 2Wiki  | MusiQue   |
|----------|----------|----------|--------|-----------|
| Train    | 79168    | 78785    | 90447   | 15,000   | 19,938   |
| Dev      | 8757     | 8837     | 7405    | 12576   | 2417     |
| Test     | 3610     | 11313    | -       | -       | -        |

Sharded Data Parallelism (FSDP) for distributed training and use bfloat16 precision for both training and evaluation. Table 4 summarizes the main hyperparameters. During inference, we sample with a temperature of 1.0 and allow up to 4 retrieval steps per query. Retrieved documents are concatenated and truncated to a maximum length of 512 tokens. For direct inference and supervised fine-tuning (SFT) baselines, we adopt Qwen2.5-3B-Instruct [45] as the backbone model.

**Dataset Statistics.** All datasets are obtained from the FlashRAG Datasets collection. Detailed statistics are reported in Table 5. The LAnR training set is constructed from the training splits of NQ and HotpotQA, comprising 169,615 examples. For evaluation, we aggregate the test or development splits from seven benchmarks: test splits are used for datasets that provide them (e.g., NQ, TriviaQA), while development splits are used otherwise (e.g., HotpotQA, 2Wiki, MusiQue).

**Computational Serving.** Although LAnR-Instruct introduces higher indexing cost than conventional text embedding approaches, it reduces serving-time computational overhead by unifying retrieval and generation within a single model. Table 6 compares LAnR-Instruct with Auto-Refine-Instruct using the same Qwen2.5-3B backbone. Unlike Auto-Refine-Instruct, which requires an additional external embedding retriever during inference, LAnR-Instruct performs retrieval directly through latent representations inside the generation model. As a result, LAnR-Instruct requires lower VRAM usage and simplifies deployment by eliminating the need to maintain a separate text retrieval encoder.

### B.1 Retrieval Control Head Model Design

The retrieval control head $f_θ$ is a single affine projection followed by a sigmoid:

$$
\hat{y}(r) = \sigma(W q^{(r)} + b), \quad W \in \mathbb{R}^{1 \times d}, \quad b \in \mathbb{R},
$$

where $q^{(r)} = h^{(r)}$ is the last-layer hidden state of the final [PRED] token at retrieval turn $r$, and $d$ is the LLM's hidden dimension ($d=2048$ for Qwen-2.5-3B, $d=3584$ for Qwen-2.5-7B). The head introduces $d+1$ trainable parameters, 2,049 for the 3B backbone and 3,585 for the 7B backbone, adding negligible capacity (under 0.002% of total model parameters) and requiring zero additional LLM forward passes at inference.
```

### --- Page 0015 ---

```markdown
| Method               | Retrieval Architecture | Extra Retriever | VRAM Usage (GB BF16) |
|----------------------|-----------------------|-----------------|-----------------------|
| Auto-Refine-Instruct | Text-based retrieval   | Yes             | 6.48                  |
| LaNr-Instruct        | Unified latent retrieval| No              | 5.85                  |

![Entropy distribution on RAG benchmarks using Qwen models](assets/page_0015_img_1.png)

### C. Latent Retrieval Control

Existing iterative RAG systems rely on explicit token-level signals to determine retrieval termination [37, 26, 19], entangling the stopping decision with surface-level text generation rather than the model’s internal assessment of evidence sufficiency. This raises a natural question: does the hidden state in an LLM already encode a reliable signal of whether the retrieved context is adequate, even before any answer tokens are produced?

**Setup.** Given an input query token sequence $x$ with an instruction to produce an answer, and a subset $G_k \subset P$ of gold supporting documents sampled from the full set $P$, we construct a probe sequence that enforces answer generation conditioned on the provided context:

$$
\bar{x} = (x, \text{GOLD}, \text{PRED})
$$

By causal attention, the hidden state $h_t := h_{\text{pdb}}(\bar{x}) \in \mathbb{R}^d$ attends to the full query and gold context. The LM head produces a distribution over the first answer token, $p_k(v) = \text{softmax}(W_h h_v)$, with predictive entropy:

$$
H_k = -\sum_{v \in V} p_k(v) \log p_k(v), \quad V \text{ denotes the vocabulary.}
$$

**Experimental Findings.** We vary $k$ from 0, as no gold evidence to $k_{\text{max}} = |P|$, as all gold evidence) and measure the resulting entropy distribution. Figure 7 reports results across four multi-hop QA benchmarks: HotPotQA, TriviaQA, MusiSQue, and 2WikiMultiHopQA, using model Qwen-2.5. Instruct at the 3B and 7B scales without finetuning. Across all datasets and model combinations, $H_k$ decreases monotonically as $k$ increases; at $k = k_{\text{max}}$, the model exhibits high entropy, reflecting substantial uncertainty over the answer space; at $k = k_{\text{max}}$, the distribution concentrates near $H_k < 1.0$. This separation is most pronounced on compositional multi-hop benchmarks, such as MusiSQue and 2WikiMultiHopQA, where the absence of bridge documents prevents shortcuts via parametric recall.

\footnote{https://huggingface.co/datasets/RUC-NLPLR/FlashRAG_datasets}
```

### --- Page 0016 ---

```markdown
Observation 1 (Entropy-Sufficiency Correlation) $H_k$ decreases monotonically in $k$; in particular, $E[H_k | P \subset G_k] < E[H_k | P \not\subset G_k]$.

Implication for retrieval control. Observation 1 establishes that $H_k$ is a reliable signal of evidence sufficiency. Moreover, $H_k$ is a deterministic function of the same hidden state $h_k$ used for retrieval, as in Eq. 1, so entropy-based stopping decisions can, in principle, be inferred from $h_k$ alone, without requiring full text generation at each retrieval step. Building on this, Section 3 introduces a lightweight MLP-based Retrieval Control Head that operates directly on $h_{[PRED]}$ to predict whether further retrieval is needed.

## D Control-Head Supervision under Varying Annotation Regimes

The control-head label $y^{(r)}$ defined in Eq. 5 requires access to a set of supporting documents $\mathcal{P}$. Such annotations are directly available in standard multi-hop QA datasets (e.g., HotpotQA, 2WikiMultiHopQA, MuSiQue), and are commonly used by prior retrieval-augmented methods for supervision or reward design [19, 33, 37]. Our framework therefore operates under the same supervision assumptions as competitive baselines in the multi-hop setting.

### Span-containment labeling for single-hop datasets. 
For single-hop datasets (e.g., NQ, TriviaQA), passage-level annotations are not directly aligned with the retrieval corpus. We adopt a standard Dense Passage Retrieval (DPR) [21] strategy to construct $\mathcal{P}$.

### Chunking. 
We first segment the Wikipedia corpus $\mathcal{C}$ into fixed-length passages (100-word chunks), which serve as the retrieval units. This preprocessing step is deterministic and independent of model training.

### Positive passage assignment. 
We then construct a single positive passage per query using dataset-specific heuristics:

(i) Datasets with annotated contexts (e.g., NQ). We align the original annotated gold passage to the closest matching chunk in $\mathcal{C}$ based on answer overlap. The matched chunk is treated as the positive passage. Examples for which no alignment is found are discarded.

(ii) Datasets without annotated contexts (e.g., TriviaQA). We adopt distant supervision by retrieving passages using a sparse retriever (e.g., BM25, E5), and selecting the highest-ranked passage that contains the answer string as the positive passage. If no retrieved passage within the top-$k$ contains the answer, the example is removed.

This procedure yields weakly supervised "gold" passages that are not exhaustively annotated but have been shown to be effective in prior work. The resulting set $\mathcal{P}$ is used consistently for both retrieval training and control-head supervision, ensuring a unified notion of relevance without requiring additional manual annotation.

## E More Experimental Results

### E.1 Retrieval Control Head Analysis

Figure 8 shows ROC curves for the retrieval control head on both LaNr-3B-Base and LaNr-3B-Trust across all five benchmarks. Both variants achieve strong AUC on multi-hop datasets, confirming that the $[PRED]$ hidden state carries a reliable evidence-sufficiency signal when supporting documents are compositionally distributed. Performance of both model variants is lower on single-hop datasets (Base: 0.730, 0.725; Instruct: 0.704, 0.712), where the distinction between "enough" and "not enough" evidence is less clear because a single retrieved document often suffices. The consistent improvement of the Instruct variant across all datasets and the clear stratification by task complexity confirm that the control head learns a genuine sufficiency signal rather than a dataset-specific heuristic.
```

### --- Page 0017 ---

![ROC curves for the retrieval control head across five benchmarks](assets/page_0017_img_1.png)  
![ROC curves for the retrieval control head across five benchmarks](assets/page_0017_img_2.png)  

Table 7: EM and F1 across methods and model sizes (Qwen-2.5-Base, 3B and 7B) on five QA benchmarks.

| Model               | Metric | NQ    | TriviaQA | HotpotQA | 2Wiki | MuSiQue | Avg   |
|---------------------|--------|-------|----------|----------|-------|---------|-------|
| **Qwen-2.5-3B-Base** |        |       |          |          |       |         |       |
| SearchR1-3B-Base    | EM     | 0.421 | 0.583    | 0.297    | 0.274 | 0.066   | 0.328 |
|                     | F1     | 0.476 | 0.650    | 0.380    | 0.322 | 0.123   | 0.390 |
| AutoRefine-3B-Base  | EM     | 0.467 | 0.620    | 0.405    | 0.393 | 0.157   | 0.408 |
|                     | F1     | 0.534 | 0.689    | 0.503    | 0.453 | 0.233   | 0.482 |
| LaNr-3B-Base (ours) | EM     | 0.455 | 0.610    | 0.402    | 0.183 | 0.410   | 0.410 |
|                     | F1     | 0.535 | 0.654    | 0.589    | 0.603 | 0.256   | 0.527 |
| **Qwen-2.5-7B-Base** |        |       |          |          |       |         |       |
| SearchR1-7B-Base    | EM     | 0.469 | 0.627    | 0.410    | 0.272 | 0.173   | 0.390 |
|                     | F1     | 0.552 | 0.710    | 0.517    | 0.327 | 0.346   | 0.466 |
| AutoRefine-7B-Base  | EM     | 0.484 | 0.659    | 0.451    | 0.187 | 0.437   | 0.434 |
|                     | F1     | 0.574 | 0.729    | 0.573    | 0.607 | 0.283   | 0.553 |
| LaNr-7B-Base (ours) | EM     | 0.482 | 0.631    | 0.459    | 0.415 | 0.205   | 0.438 |
|                     | F1     | 0.570 | 0.724    | 0.590    | 0.612 | 0.295   | 0.558 |

E.2 Effect of Model Size

Table 7 compares Search-R1, AutoRefine, and LaNr across Qwen-2.5-Base 3B and 7B backbones using both EM and F1 metrics. Scaling from 3B to 7B consistently improves performance for all methods, with the largest gains observed on multi-hop benchmarks that require compositional reasoning. At both model scales, LaNr achieves the strongest overall multi-hop performance. With the 3B backbone, LaNr-3B-Base attains the highest EM on HotpotQA (0.417), 2Wiki (0.402), and MuSiQue (0.183), while also substantially outperforming prior methods in F1, particularly on HotpotQA (0.589 vs. 0.503 for AutoRefine) and 2Wiki (0.603 vs. 0.453). Similar trends hold at 7B scale, where LaNr-7B-Base achieves the best EM and F1 across all three multi-hop datasets, reaching 0.459/0.590 on HotpotQA, 0.415/0.612 on 2Wiki, and 0.205/0.295 on MuSiQue. Although AutoRefine remains competitive on single-hop datasets such as NQ and TriviaQA, the results show that LaNr benefits more consistently from increased model capacity, suggesting that latent retrieval scales effectively with stronger backbone representations and is particularly advantageous for complex multi-step reasoning tasks.

### --- Page 0018 ---

```markdown
| Table 8: Statistical analysis against search-during-think baselines. The p-value column represents the T-test result of LAnR-Instruct vs. AutoRefine-Instruct and Search-R1-Instruct. |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Modelled       | NQ             | TriviaQA       | HotpotQA       | 2Wiki          | MultiQA        | p-value        |
| LAnR-Instruct  | 0.460 ± 0.008  | 0.613 ± 0.005  | 0.419 ± 0.006  | 0.406 ± 0.006  | 0.193 ± 0.005  | 7.49 × 10^{-39} |
| AutoRefine-Instruct | 0.461 ± 0.010 | 0.620 ± 0.007 | 0.405 ± 0.012  | 0.420 ± 0.010  | 0.145 ± 0.011  |                |
| Search-R1-Instruct | 0.410 ± 0.009 | 0.597 ± 0.019 | 0.315 ± 0.016  | 0.254 ± 0.023  | 0.124 ± 0.005  |                |

![EM accuracy by training step for models trained with 1, 2, and 3 [PRED] tokens across four benchmarks. N=1 converges fastest and performs best on multi-hop datasets; additional tokens offer marginal gains only on single-hop TriviaQA.](assets/page_0018_img_1.png)

### E.3 Variance Analysis
Table 8 reports the mean and standard deviation across three runs with different random seeds to evaluate the robustness of search-during-think methods. LAnR-Instruct consistently achieves lower variance than competing baselines across most datasets, indicating more stable retrieval and reasoning behavior. To assess statistical significance, we perform paired T-tests between LAnR-Instruct and AutoRefine-Instruct. The resulting low p-values ($p \ll 0.01$) confirm that the performance differences are statistically significant, demonstrating that the gains of latent retrieval are reliable rather than arising from random variation.

### E.4 Ablation on Numbers of [PRED] Tokens
We further analyze the effect of training with multiple [PRED] tokens, detailed in Figure 9. Increasing the number of [PRED] tokens encourages the model to construct more abstract latent queries. On single-hop benchmarks such as TriviaQA and NQ, LAnR consistently benefits from additional [PRED] tokens, with performance improvements emerging early in training. In contrast, on more challenging multi-hop benchmarks, models with multiple [PRED] tokens require substantially longer training before surpassing the performance of the single-[PRED] setting. These results suggest that while additional [PRED] tokens can improve abstraction capability, they also increase optimization complexity.
```

### --- Page 0019 ---

```markdown
# F  Limitations

**Corpus encoding overhead.** LaNR encodes the entire retrieval corpus using the same LLM used for generation. While this enables tight representational alignment, it is more computationally expensive than conventional retrieval approaches, including sparse lexical methods such as BM25 [30] and lightweight dense retrievers such as BGE [5]. Re-encoding is also required whenever the backbone model is updated, which may limit practicality in frequently updated knowledge bases.

**Interpretability of latent queries.** Unlike text-based RAG systems, where retrieval queries are human-readable and easily auditable, LaNR’s latent query vectors are not directly interpretable. This makes it harder to diagnose retrieval failures or explain why specific documents were retrieved, which may be a concern in high-stakes deployments.

**Evaluation scope.** All experiments use English-language open-domain QA benchmarks with Wikipedia as the knowledge source. Performance on non-English, domain-specific, or long-document corpora remains untested, and results may not generalize to settings with significantly different document distributions or answer styles.

# G  Broader Impact

LaNR is foundational research on latent retrieval-augmented generation efficiency and does not target any specific deployment. Nevertheless, we briefly discuss potential societal implications.

**Positive impacts.** By reducing inference latency by up to 2.7× and token generation by ∼ 30× compared to existing RAG systems, LaNR lowers the computational cost of knowledge-grounded language model inference, improving accessibility for resource-constrained practitioners and reducing energy consumption at scale. Improved factuality in question answering also has broad benefits in educational, medical, and scientific information access.

**Potential negative impacts.** As with any system that improves the fluency and factual grounding of LLM outputs, LaNR could lower the barrier to generating more convincing disinformation or fabricated content at scale. The efficiency gains in particular may make high-volume automated generation more feasible. Additionally, because LaNR retrieves from large external corpora, any biases present in those corpora may be silently propagated into generated answers without explicit query traces, potentially making such biases harder to audit than in text-based retrieval pipelines where queries are inspectable.

# H  Case Studies

The following examples trace LaNR’s retrieval process hop by hop. At each search, the model runs one forward pass over a growing context ending with [PRE0]. The hidden state at that position encodes all evidence seen so far, drives the next dense retrieval query, and is fed to the MLP control head to decide whether to continue. The context after it searches is:

$$
[Q] \text{ question } [d^{(1)}_1] \cdots [d^{(K)}_1] \cdots [d^{(1)}_r] \cdots [d^{(K)}_r] [PRE]
$$

Gold supporting documents are underlined with *.
```


### --- Page 0020 ---

```markdown
# Example 1 - MuSiQue (3-hop compositional)

## 2WikiMultihopQA - 2-hop compositional

**Question:** "Where do Greyhound buses leave from in the city where the performer of Darlings formed?"

**Reasoning chain:** Darlings → performer: Kevin Drew → formation city: Toronto → bus terminal

### Turn 1
**Context:** [Q] Where do Greyhound buses leave from . . . formed? [PRED]

**Retrieved:** Kevin Drew*, Spirit If. . ., Detropia

[Kevin Drew*] Kevin Drew (born September 9, 1976) is a Canadian musician and songwriter who, together with Brendan Canning, founded the expansive Toronto baroque-pop collective Broken Social Scene.

**Turn 2**
**Context:** [Q] ...[Kevin Drew: ...Toronto baroque-pop collective...] [Spirit If:...] [Detropia:...] [PRED]

**Retrieved:** Toronto Coach Terminal*, Alvarado Transportation Center, Dufferin St. Bridges

[Toronto Coach Terminal*] The Toronto Coach Terminal is the central bus station for inter-city services in Toronto, Ontario, Canada. It is located at 610 Bay Street, in the city’s Downtown.

**Turn 3**
**Context:** [Q] ...[Kevin Drew:...] [Toronto Coach Terminal: ...Toronto...Bay St...] [PRED]

**Retrieved:** Darlings (Kevin Drew album)*, Peace Center, Darlington County

[Darlings (Kevin Drew album)*] Darlings is the second solo album by Broken Social Scene co-founder Kevin Drew. It was released on March 18, 2014.

**Answer:** The Toronto Coach Terminal ✓ Correct * gold document

The [PRED] state at Search 1 already encodes Toronto implicitly through the Kevin Drew article, letting the model jump directly to the bus-terminal query at Search 2 without materialising "Toronto" as an explicit intermediate result. All three gold documents are retrieved across three latent searches, and the control head stops as soon as the evidence chain is complete.
```

### --- Page 0021 ---

```markdown
# Example 2 - 2WikiMultiHopQA (2-hop)

## 2WikiMultiHopQA - 2-hop compositional

**Question:** "Where was the director of film The Green Fog born?"  
**Reasoning chain:** The Green Fog → director: Guy Maddin → birthplace: Winnipeg, Manitoba  

| Turn | Context                                                                 | Pcttl  | Action     |
|------|-------------------------------------------------------------------------|--------|------------|
| 1    | [Q] Where was the director of film The Green Fog born? [PRED]         | 0.9999 | CONTINUE   |
|      | Retrieved: The Green Fog, Walon Green, Adam Green (filmmaker)         |        |            |
|      | [The Green Fog] The Green Fog is an experimental film directed by Guy Maddin, Evan Johnson, and Galen Johnson that loosely revisits the plot of Alfred Hitchcock's Vertigo through found footage. |        |            |
| 2    | [Q] [The Green Fog: ...directed by Guy Maddin...] [Walon Green:...] [Adam Green:...] [PRED] | 0.9992 | CONTINUE   |
|      | Retrieved: Guy Maddin, The Heart of the World, The Forbidden Room      |        |            |
|      | [Guy Maddin] Guy Maddin (born February 28, 1956) is a Canadian screenwriter, director, author, cinematographer, and film editor of both features and short films, from Winnipeg, Manitoba. |        |            |
| 3    | [Q] [The Green Fog:...] [Guy Maddin: ...Winnipeg, Manitoba...] [PRED] | 0.0000 | STOP       |
|      | (Control head halts; no new documents retrieved.)                       |        |            |
| **Answer:** Winnipeg, Manitoba ✓ Correct |

This example shows the transition across all three datasets: Pcttl holds at 0.9999 and 0.9992 while evidence accumulates, then drops to 0.0000 the moment Guy Maddin and his birthplace are both in context. No text query names "Guy Maddin" at any step; the latent query at Search 2 is shaped entirely by the {PRED} representation formed over the The Green Fog article.
```


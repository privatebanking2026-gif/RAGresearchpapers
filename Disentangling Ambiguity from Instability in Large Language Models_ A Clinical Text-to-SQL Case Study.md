# Disentangling Ambiguity from Instability in Large Language Models  A Clinical Text-to-SQL Case Study

### --- Page 0001 ---

```markdown
# Disentangling Ambiguity from Instability in Large Language Models: A Clinical Text-to-SQL Case Study

**Angelo Ziletti\***  **Leonardo D’Ambrosi**  
Bayer AG, Berlin, Germany  
angelo.ziletti@bayer.com

## Abstract

Deploying large language models for clinical Text-to-SQL requires distinguishing two qualitatively different causes of output diversity: (i) input ambiguity that should trigger clarification, and (ii) model instability that should trigger human review. We propose CLUES, a framework that models Text-to-SQL as a two-stage process (interpretations → answers) and decomposes semantic uncertainty into an ambiguity score and an instability score. The instability score is computed via the Schur complement of a bipartite semantic graph matrix. Across AmbigQA/SituatedQA (gold interpretations) and a clinical Text-to-SQL benchmark (known interpretations), CLUES improves failure prediction over state-of-the-art Kernel Language Entropy. In deployment settings, it remains competitive while providing a diagnostic decomposition unavailable from a single score. The resulting uncertainty regimes map to targeted interventions - query refinement for ambiguity, model improvement for instability. The high-ambiguity/high-instability regime contains 51% of errors while covering 25% of queries, enabling efficient triage.

## 1 Introduction

Text-to-SQL systems powered by Large Language Models (LLMs) promise to democratize access to Electronic Health Records (EHRs) and claims databases, enabling epidemiologists to query complex data using natural language Lee et al. (2023); Ziletti and D’Ambrosi (2024); Koretsky et al. (2025). While retrieval-augmented approaches (RAG) have shown promising results Ziletti and D’Ambrosi (2024), deployment faces a critical barrier: the risk of syntactically correct but semantically erroneous queries that return plausible yet incorrect results.
```


### --- Page 0002 ---

```markdown
A key challenge is latent ambiguity: ambiguity that is not evident to the user but significantly affects results Elazar et al. (2024); Gupta et al. (2025). Consider “How many patients over 18 have atopic dermatitis?”: does “over 18” refer to age at diagnosis or current age? First diagnosis or any? Each interpretation yields a different SQL query and potentially different results. Left unresolved, such ambiguities undermine reproducibility of epidemiological research Hemkens et al. (2016); Herrett et al. (2017).

Latent ambiguity also complicates uncertainty estimation. When a model produces diverse outputs, is this diversity a legitimate reflection of query ambiguity, or a sign of model instability? Existing methods provide a single score that conflates two fundamentally distinct sources: uncertainty arising from inherent ambiguity in the input (akin to aleatoric uncertainty), and uncertainty stemming from the model’s internal instability or lack of knowledge (akin to epistemic uncertainty). This distinction enables targeted diagnosis of system failures for model improvement. For a deployed system, it is also operationally critical: high ambiguity should trigger a clarification dialogue, while high instability should flag the query for human review.

Contributions. (1) CLUES, a framework that decomposes semantic uncertainty into ambiguity ($H_I$) and conditional instability ($H_{RI}$) via the Schur complement of a bipartite similarity matrix, enabling different interventions (clarification vs. review) and applicable to any black-box LLM. (2) An interpretation-generation procedure for epidemiological questions and a clinical Text-to-SQL dataset with multiple interpretations.¹

¹Code and dataset will be released upon acceptance.

(3) Empirical evidence across open-domain QA and clinical Text-to-SQL that CLUES improves failure prediction and yields regime-based error concentration useful for targeted routing.

Paper Organization. Sec. 3 formalizes our two-stage generative framework and introduces the Schur complement construction. We validate the decomposition across three settings of increasing complexity: open-domain QA with gold interpretations (Sec. 4), a real-world clinical Text-to-SQL dataset (Sec. 5), and production deployment with on-the-fly interpretation generation (Sec. 6).

## 2 Related Work

Semantic entropy and kernel-based uncertainty. Semantic entropy clusters generations into equivalence classes for hallucination detection Kuhn et al. (2023); Farquhar et al. (2024); Kernel Language Entropy (KLE) generalizes this by encoding pairwise semantic similarities. Nikitin et al. (2024) Recent work derives kernel entropy via bias-variance-covariance decomposition, Kamkari et al. (2024) combines semantic signals with token-level uncertainty, Huang et al. (2025) and exploits geometric measures such as semantic density and volume. Sun et al. (2025); Kumar et al. (2025); Lee et al. (2024a) However, these methods operate on a single
```

### --- Page 0003 ---

```markdown
![Detailed description of the chart](assets/page_0003_img_1.png)

## 3 Methodology: A Decomposition Framework for Semantic Uncertainty

Decomposing aleatoric and epistemic uncertainty. This distinction is fundamental in probabilistic machine learning and critical for deployment in safety-critical domains. Kendall and Gal (2017); Hüllermeier and Waegeman (2021) Recent LLM work trains meta-models or uses temperature sensitivity to identify epistemic uncertainty, but these approaches require additional supervision or pipeline changes. Desai and Durrett (2024); Mozes et al. (2025); Zhang et al. (2025) In the EHR-QA domain, Kim et al. (2022) decompose uncertainty to detect ambiguous questions, but operate at the token level without modeling multi-stage generation.

Text-to-SQL and clinical uncertainty. Text-to-SQL has evolved from cross-domain benchmarks Yu et al. (2018) to clinical applications for EHR access and clinical trial recruitment Ziletti and D’Ambrosi (2025); Ghosh et al. (2025); Deng et al. (2022). Benchmarks like EHRSQL Lee et al. (2023) and BiomedSQL Koretsky et al. (2025) evaluate clinical Text-to-SQL, while multi-turn approaches improve robustness Park et al. (2024). Existing systems prioritize execution accuracy but provide limited uncertainty quantification Park et al. (2022); Kim et al. (2024); Ziletti and D’Ambrosi (2024). Reliability work focuses on unanswerable detection, ensemble methods, or token entropy thresholds Kim et al. (2024); Lee et al. (2024b), but does not decompose uncertainty by source or model semantic equivalence.
```

### --- Page 0004 ---

```markdown
![Bipartite Semantic Graph for Regime II: Ambiguity](assets/page_0004_img_1.png)
![Bipartite Semantic Graph for Regime III: Instability](assets/page_0004_img_2.png)

| Regime II: Ambiguity | Regime III: Instability |
|----------------------|-------------------------|
| ![Bipartite Semantic Graph](assets/page_0004_img_1.png) | ![Bipartite Semantic Graph](assets/page_0004_img_2.png) |
| Full System Matrix $W$ | Full System Matrix $W$ |
| ![Full System Matrix W](assets/page_0004_img_3.png) | ![Full System Matrix W](assets/page_0004_img_4.png) |

Figure 1: Examples of uncertainty regimes in CLUES. (Left) Regime II, Ambiguity: Interpretations (brown) are semantically distinct (low $W_{II}$ off-diagonal), but results (green) cluster tightly within each interpretation (high intra-cluster $W_{RR}$), yielding high $H_I$, low $H_{R|I}$. (Right) Regime III, Instability: Interpretations are similar (high $W_{II}$ off-diagonal), yet results vary substantially within interpretations, yielding low $H_I$, high $H_{R|I}$. Red dashed edges: interpretation-result assignments ($W_R$). Bottom: corresponding system matrices $W$.

Our methodology extends the geometric view of uncertainty from KLE to a multi-stage generative process. We construct a bipartite semantic graph representing the entire system and use tools from linear algebra and spectral graph theory to decompose its structural complexity.

### 3.1 Interpretation-Augmented Generation

We formalize the Text-to-SQL system as a two-stage generative process. Given an initial natural language query $q$:
```

### --- Page 0005 ---

```markdown
## 3.2 Constructing the Bipartite Semantic Graph

We define a similarity function $k(\cdot) \in [0, 1]$ measuring semantic similarity between text strings. The full system matrix $W$ serves as the graph’s weighted adjacency matrix and has a natural bipartite block structure:

$$
W = \begin{pmatrix}
W_{II} & W_{IR} \\
W_{RI} & W_{RR}
\end{pmatrix} \in \mathbb{R}^{(N+M) \times (N+M)}
$$

where:

- $W_{II} \in \mathbb{R}^{N \times N}$: The interpretation similarity matrix, with $(W_{II})_{ij} = k(I_i, I_j)$. This block encodes the semantic relationships among interpretations.
  
- $W_{RR} \in \mathbb{R}^{M \times M}$: The answer similarity matrix (flattening $(n, m) \mapsto j$), with $(W_{RR})_{ij} = k(R_i, R_j)$. This block encodes the semantic relationships among all generated answers.
  
- $W_{IR} \in \mathbb{R}^{N \times M}$ and $W_{RI} = W_{R}^{T}$: The assignment matrices, with $(W_{IR})_{ij} = 1$ if answer $R_j$ was generated from interpretation $I_i$, and 0 otherwise.

The diagonal blocks $W_{II}$ and $W_{RR}$ encode graded semantic similarity, interpreted as weighted edge connectivity in the graph. The off-diagonal blocks $W_{IR}$ and $W_{RI}$ encode the generative provenance between interpretations and their answers. We represent this provenance as binary connectivity, since each answer traces to exactly one interpretation. This mixed representation is consistent with the graph-theoretic framework: edge weights encode connection strength, derived from continuous similarity scores or binary structural links. The KLE framework Nikitin et al. (2024) constructs graphs from natural-language-inference-based similarity scores; our extension introduces a second node type (answers) connected to the first (interpretations) via known generative links rather than inferred similarity.
```

### --- Page 0006 ---

```markdown
# PAGE_NAME: page_0006

## Prompt-Based Similarity
The similarity function $k(·,·)$ is implemented via an LLM with a task-specific prompt that defines the relevant notion of equivalence. For Text-to-SQL, the prompt assesses whether two interpretations would yield logically equivalent SQL queries; for QA benchmarks, whether they would yield the same factual answer. This ensures that $W_{II}$ and $W_{RR}$ encode task-relevant semantics rather than generic lexical similarity.

## Heat Kernel Regularization
Following KLE Nikitin et al. (2024), we apply a heat diffusion kernel $K_{\tau} = e^{-\tau L}$, where $L = D - W$ is the graph Laplacian and $D$ is the degree matrix. This transforms raw pairwise similarities into a smoothed representation where the hyperparameter $\tau$ controls granularity: small $\tau$ preserves fine-grained distinctions, while large $\tau$ diffuses similarity across the graph, merging nearby nodes into coarse clusters. To select $\tau$ without a validation set, we calibrate against an idealized baseline: $W_{RR}$ of perfectly identical items (all ones) should yield near-zero entropy ($< 0.001$ bits). This yields $\tau = 10$ for our experiments. Sensitivity analysis (Table 4) confirms robustness across $\tau \in [2, 20]$.

## 3.3 Decomposing Uncertainty with the Schur Complement
Our central goal is to decompose the total uncertainty of the system into interpretable components: input ambiguity and system instability. The resulting scores $H_I$ and $H_{R | I}$ are not intended to replace aggregate measures like $H_R$ for failure prediction; rather, they provide complementary information about the source of uncertainty, enabling differential interventions (input refinement vs. generation review) that a single score cannot support.

### 3.3.1 Baseline: Result Entropy $H_R$
The state-of-the-art KLE approach Nikitin et al. (2024) computes entropy over generated outputs without modeling their provenance. In our framework, this yields $H_R$ from $W_{RR}$ alone. High $H_R$ indicates diverse outputs but does not distinguish the source of uncertainty: input ambiguity or system instability. This is the baseline we aim to improve upon.

### 3.3.2 Joint Uncertainty $H(R,I)$
We define total system uncertainty as the joint entropy over the full bipartite graph, computed from the complete similarity matrix $W$ (Eq. 1). This captures uncertainty across both interpretations and results, along with their generative relationships encoded in $W_{IR}$.

### 3.3.3 Ambiguity Score $H_I$
We define the ambiguity score as the entropy over the set of interpretations, estimating ambiguity in the user’s query. We compute $H_I$ by applying the KLE framework to $W_{I}$:
```

### --- Page 0007 ---

```markdown
### 3.3.4 Instability Score $H_{R | I}$

We define the instability score as the conditional entropy of results given interpretations: “Given a clear interpretation, how much does the answer still vary?”

The Naive Subtraction Approach. A natural approach is to estimate conditional entropy via subtraction: 
$$
H_{R | I} = H_{R,I} - H_I.
$$ 
However, this relies on the Shannon entropy chain rule, which does not generally hold for von Neumann entropy on heat-kernel density matrices \cite{Nikitin et al. (2024)}. The joint and marginal kernels have different sizes (see Eq. (1)) and undergo independent diffusion processes, so $H_{R | I}$ is not guaranteed to be positive. Indeed, $\hat{H}$ yields negative values in 37–60% of cases (Tables 1 and 3).

The Schur Complement Approach. We leverage the block structure of $W$ (Eq. (1)) to construct the conditional similarity structure directly via the Schur complement \cite{Schur (1917)}; \cite{Zhang (2005)}:
$$
S = W_{RR} - W_{R I}(W_{II} + \epsilon I)^{-1}W_{IR}
$$
where $\epsilon = 10^{-3}$ ensures invertibility. The Schur complement provides a principled way to “condition on” the interpretation structure. The projection term $W_{R}W_{I}^{\dagger}W_{IR}$ captures the portion of result similarity explained by interpretations. The residual $S$ retains only the similarity structure that input ambiguity cannot account for. This mirrors the Gaussian case, where the Schur complement yields conditional covariance $Cov(R | I)$ \cite{Boyd and Vandenberghe (2004)}. The entropy $H_{R | I}$ computed from $S$ therefore quantifies the system’s internal inconsistency. Since $S$ is not guaranteed to be positive semidefinite (PSD), we project onto the PSD cone via eigendecomposition, clipping negative eigenvalues to zero \cite{Higham (2002)}, then apply the KLE recipe (Eq. (2)). Unlike the subtraction approach above, this guarantees $H_{R | I} \geq 0$. We term this decomposition framework CLUES (Conditional Language Uncertainty via Entropy and Schur). Crucially, CLUES distinguishes uncertainty from query ambiguity (potentially addressable via clarification) from system instability (requiring model-level intervention).

### 3.3.5 Uncertainty regimes and recommended interventions

We define four regimes by thresholding $H_I$ (ambiguity) and $H_{R | I}$ (instability):

- **Regime I:** Confident (low $H_I$, low $H_{R | I}$): auto-answer.
- **Regime II:** Ambiguity (high $H_I$, low $H_{R | I}$): input refinement (clarification).
- **Regime III:** Instability (low $H_I$, high $H_{R | I}$): generation review.
```

### --- Page 0008 ---

```markdown
# 4 Experiments on Open Answer Datasets

| Model       | AUROC       | Regime      | Acc. | $H_R$ | $H_{R|I}$ | $A$ | $B$ | Acc. | $H_R$ | $H_{R|I}$ | $A$ | $B$ |
|-------------|-------------|-------------|------|-------|-----------|-----|-----|------|-------|-----------|-----|-----|
| GPT-3.5    | 27.0       | 0.641      | 52.8 | 9.81  | 1.457     | 25.67 | 0.634 | 37.4 | 4.4  | 3.1       |
| Model X    | 50.6       | 0.584      | 81.0 | 1.81  | 3.76      | 0.0   | 0.0   | 0.0  | 0.0  | 0.0       |
| Model Y    | 35.7       | 0.536      | 75.0 | 37.5  | 5.10      | 30.7  | 502  | 49.5 | 44.4 | 4.20      |
| Model Z    | 27.3       | 0.538      | 61.0 | 1.81  | 3.76      | 0.0   | 0.0   | 0.0  | 0.0  | 0.0       |

*Table 1: Multi-Model Validation on Open-Domain QA Benchmarks. We evaluate $N_q = 300$ questions per dataset across 5 LLMs (pooled across 5 splits). The naive subtraction $H_{R|I} - H_R$ yields negative values in 53%/37% of cases (AmbigQA/SituatedQA).*

We first validate our decomposition on two open-domain QA benchmarks: AmbigQA Min et al. (2020), where questions admit multiple valid interpretations due to semantic ambiguity, and SituatedQA Zhang and Choi (2021), where answers vary based on temporal or geographical context. Both datasets provide gold-standard interpretations paired with corresponding answers, allowing us to test whether $H_{R|I}$ provides a more accurate failure signal than $H_R$.

## 4.1 Experimental Setup

**Dataset and Protocol.** From each dataset, we select the first $N_q = 300$ questions with at least two gold interpretations, capping at three interpretations per question to limit inference costs. Following the two-stage framework in Sec. 3, the interpretation stage is given by the dataset’s gold interpretations. For the generation stage, we sample $M = 3$ answers per interpretation, yielding $N \times M$ results per question ($N \in \{2, 3\}$). We construct the bipartite semantic graph $W$ and compute $H_R$, $\hat{H}_{R|I}$, and $H_{R|I}$.
```

### --- Page 0009 ---

```markdown
# Large Language Models

For answer generation, we evaluate five frontier LLMs: GPT-OSS-120B (hereafter GPT-OSS) OpenAI et al. (2025), Kimi K2 Thinking (KimiK2) Team et al. (2025) and Qwen-3-VL-235B-A22B (Qwen3) Team (2025) are open-weights; Gemini 3 Pro (Gemini3Pro) Gemini Team (2023) and Claude Sonnet 4.5 (Claude4.5S) Anthropic AI (2023) are proprietary. All models are sampled with temperature $T = 1$. Throughout this paper, we use Gemini 2.5 Flash ($T = 0$) for semantic similarity ($W_{IJ}, W_{RR}$) and answer correctness evaluation (unless otherwise noted).

## 4.2 Results and Analysis

Results are presented in Table 1. Predicting failures on frontier LLMs is inherently challenging; the baseline $H_R$ achieves AUROC scores of 0.47–0.66 depending on model and dataset. Our proposed $H_{R | I}$ consistently outperforms $H_R$, achieving pooled AUROC of 0.627 on AmbigQA (+0.077) and 0.641 on SituatedQA (+0.074). Per-model results show consistent improvements, with the largest gains when $H_R$ itself carries predictive signal; for models where $H_R$ performs near chance (Claude4.5S, Qwen3), all accuracy measures show very limited discriminative power. The naive subtraction approach $\hat{H}_{R | I}$ achieves near-chance AUROC, validating the need for the Schur complement construction (Eq. 3).

The improvement over $H_R$ stems from CLUES’s ability to account for input structure. By ignoring interpretations, $H_R$ errs in both directions: diversity inflation, where legitimate variation across interpretations ($W_{IJ} \approx I, W_{RR} \approx I$) is flagged as uncertainty; and mode collapse, where identical outputs across distinct interpretations ($W_{RR} \approx 1$) yield false confidence. The Schur complement detects both failure modes.
```


### --- Page 0010 ---

```markdown
| Input Question                                                                 | Disambiguated Question                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| How many patients > 17 yo have atopic dermatitis (all codes). Breakdown by code | Calculate the count of unique patients who have a first diagnosis of Atopic Dermatitis at any time in their patient history, where the patient’s age at the time of this first diagnosis was greater than 17 years. Provide this count broken down by the specific concept code of Atopic Dermatitis. |
| How many patients with chronic kidney disease never took heparin before their chronic kidney disease diagnosis? | Count the number of unique patients who have a first occurrence of Chronic Kidney Disease and who have no record of Heparin administration at any time prior to their first Chronic Kidney Disease diagnosis.                                   |

Table 2: Disambiguation examples resolving temporal, demographic, and event ordering ambiguities. Bold text indicates specifications that were implicit or absent in the original question.

## Regime Analysis

The four-regime framework (Sec. 3.3.5) partitions by $H_r$ and $H_{R|I}$ to guide interventions. Here, we test whether $H_{R|I}$ provides discriminative value beyond $H_r$ alone. We focus on high-uncertainty queries where $H_r$ exceeds its median, then partition by median $H_{R|I}$ into Regime A (low $H_{R|I}$) and Regime B (high $H_{R|I}$). Despite identical $H_r$ profiles, accuracy differs substantially: 56.6% vs. 21.1% for AmbigQA and 46.8% vs. 13.3% for SituatedQA. These differences are highly significant (Chi-square $p < 10^{-22}$), with odds ratios of 4.9 and 5.7: queries in Regime B are 5–6× more likely to fail. This pattern holds across all five LLMs without model-specific calibration.

# 5 Clinical Text-to-SQL with Known Interpretations

## 5.1 Resolving Ambiguity in Epidemiological Questions

Epidemiological questions in natural language frequently contain implicit ambiguities requiring explicit specification for accurate analysis (Table 2). While rule-based templates (e.g., OHDISI ATLAS OHDISI Community (2024)) lack flexibility and interactive clarification introduces user fatigue, our iterative disambiguation approach automatically infers the most plausible interpretation through successive refinement rounds, presenting users with a transparent final specification they can verify or edit before SQL generation.
```

### --- Page 0011 ---

```markdown
# Disambiguation Pipeline

The disambiguation pipeline operates iteratively until convergence. Given an input question, each round prompts the LLM to:

- Identify common epidemiological ambiguities including patient count semantics (unique patients vs. all records), temporal relationships (before/after, within timeframes), population definitions (inclusion/exclusion criteria), event ordering (first vs. any occurrence), and demographic specifications (age at diagnosis vs. current age).
  
- Assign an ambiguity score $s \in [0,1]$ and generate multiple candidate interpretations ordered by clinical plausibility.
  
- Select the most plausible interpretation for the next round.

The prompt embeds domain conventions as defaults reflecting common practice in observational health research (e.g., standard age bands; defaulting to entire patient history when unspecified; unique patient counts). The process terminates when ambiguity falls below a threshold ($s < 0.1$); in practice, within 1–3 iterations. We use Gemini 2.5 Flash with structured output schemas.

This iterative disambiguation offers three advantages: (1) transparency, as users can review and edit candidates before SQL generation; (2) educational value, as outputs serve as templates for well-formed questions; and (3) systematic application of epidemiological best practices without user-specified routine parameters.

In qualitative evaluation, epidemiologists confirmed that disambiguated outputs accurately reflect domain conventions and produce clinically appropriate disambiguations.

## 5.2 Experimental Setup

### Source Dataset and Interpretation Construction

We build on EpiAskKB Ziletti and D’Ambrosi (2025), which contains question-SQL pairs for epidemiological research on EHR. Questions may be ambiguous, reflecting how epidemiologists naturally pose queries (Table 2). For each question, we construct a multi-interpretation setup following Sec. 4.1. From the question and gold SQL, we generate a disambiguated question reflecting the gold SQL, plus three alternative interpretations (Sec. 5.1). For each alternative, we generate SQL by prompting the LLM with the original question, alternative interpretation, and gold SQL to guide query structure. This constrained context minimizes structural errors. We use Gemini3Pro for all generation.

### Generation and Execution Pipeline

For each question, we sample $M = 3$ SQL queries per interpretation, yielding $3 \times 3 = 9$ generated queries per model-temperature configuration. SQL generation, entity resolution, self-correction, and execution follow Ziletti and D’Ambrosi (2024). Queries are run on Optum’s de-identified...
```


### --- Page 0012 ---

```markdown
## 5.3 Results

Results are presented in Table 3. $H_{R|I}$ outperforms $H_R$ as a failure predictor in 17 of 20 model-temperature configurations, achieving pooled AUROC of 0.762 [95% CI: 0.737, 0.787] versus 0.600 [0.575, 0.627] for $H_R$. This difference is significant ($p < 10^{-10}$), demonstrating that conditioning on interpretation structure provides substantial discriminative value. The naive subtraction $\hat{H}_{R|I}$ achieves only 0.461 AUROC with 60% negative values, validating the theoretical necessity of the Schur complement (Sec. 3.3).
```


### --- Page 0013 ---

```markdown
| Model       | Temperature | Accuracy | Metric 1 | Metric 2 | Metric 3 | Change |
|-------------|-------------|----------|----------|----------|----------|--------|
| KimiK2     | T=0.7      | 76.1     | 0.632    | 0.446    | 0.815    | +4     |
|             | T=1.0      | 74.3     | 0.548    | 0.417    | 0.817    | +4     |
|             | T=1.5      | 75.2     | 0.635    | 0.444    | 0.791    | +4     |
|             | T=2.0      | 78.9     | 0.599    | 0.418    | 0.853    | +2     |
| Claude4.5S | T=0.7      | 78.9     | 0.593    | 0.395    | 0.611    | +     |
|             | T=1.0      | 77.6     | 0.595    | 0.466    | 0.694    | +1     |
|             | T=1.5      | 78.0     | 0.633    | 0.400    | 0.744    | +3     |
|             | T=2.0      | 78.9     | 0.575    | 0.426    | 0.622    | +1     |
| Gemini3Pro | T=0.7      | 93.3     | 0.708    | 0.623    | 0.604    | +     |
|             | T=1.0      | 93.3     | 0.620    | 0.532    | 0.715    | +2     |
|             | T=1.5      | 88.8     | 0.642    | 0.578    | 0.593    | +     |
|             | T=2.0      | 88.8     | 0.635    | 0.542    | 0.595    | +1     |
| Pooled      |             | 74.4     | 0.600    | 0.461    | 0.762    | +     |

Table 3: Clinical Text-to-SQL benchmark results (N = 2,180; 109 questions × 5 models × 4 temperatures). Δ = Regime A — Regime B accuracy gap. $\hat{R}_{I|I}$ produces negative values in 60% of cases.
```

### --- Page 0014 ---

```markdown
Regime Analysis Confirms Discriminative Value. The accuracy gap $\Delta$ between Regime A (high $H_r$, low $H_{R|I}$) and Regime B (high $H_r$, high $H_{R|I}$), split at median values, is positive in 19 of 20 configurations (pooled: +33.6 pp). This replicates the open-domain QA pattern (Sec. 4.2): among queries with equally high output diversity, those with low $H_{R|I}$ are substantially more likely to succeed.

Temperature Modulates Uncertainty Signal. For most models, moderate temperatures ($T = 1.0$–1.5) yield the strongest $H_{R|I}$ signal. Claude 4.5 illustrates this pattern: regime separation increases from $\Delta = 0$ at $T=0.7$ to $\Delta = +30.8$ at $T=1.5$ (0 $\rightarrow$ +30.8), suggesting higher sampling diversity exposes latent instability. Gemini 3 Pro is the exception: its errors appear to stem from consistent but incorrect outputs rather than instability.

Sensitivity to Diffusion Parameter $\tau$. Table 4 shows $H_{R|I}$ is robust for $\tau \ge 2$ (AUROC 0.75–0.76), while $H_R$ degrades at high $\tau$ (0.74 $\rightarrow$ 0.58) as diffusion blurs pairwise similarities toward uniformity. The Schur complement resists this effect: both raw similarities and the projection term smooth proportionally, preserving residual signal.

| $\tau$ | 1    | 2    | 3    | 5    | 10   | 15   |
|--------|------|------|------|------|------|------|
| $H_I$  | 0.571| 0.574| 0.576| 0.578| 0.572| 0.569| 0.567|
| $H_R$  | 0.669| 0.742| 0.738| 0.676| 0.601| 0.586| 0.582|
| $H_{R|I}$| 0.675| 0.746| 0.764| 0.765| 0.762| 0.761| 0.760|

Table 4: Pooled AUROC sensitivity to heat kernel parameter $\tau$. $H_{R|I}$ remains stable across $\tau \in [2, 20]$ while $H_R$ degrades at high diffusion.

6 Clinical Text-to-SQL in Deployment Settings

6.1 Experimental Setup

Having established that $H_{R|I}$ outperforms $H_R$ with known interpretations, we now test whether these gains extend to production deployment. This setting differs from previous experiments in three key ways.

Interpretations are generated, not given. Interpretations are produced on the fly via our disambiguation procedure (Sec. 5.1) and may imperfectly capture the true ambiguity structure, unlike the validated gold/silver labels used previously.
```

### --- Page 0015 ---

```markdown
Uncertainty serves as a proxy signal. We evaluate final-answer correctness only, meaning $H_{R|I}$ computed over interpretations serves as a proxy for reliability rather than directly measuring per-interpretation accuracy.

Limited statistical power. Evaluation is coarser (one label per question rather than per interpretation), and models achieve high accuracy (Table 5), yielding sparse errors that challenge statistical estimation.

We use the same pipeline as Sec. 5.2, testing with and without disambiguation and RAG. RAG follows the methodology in Ziletti and D'Ambrosi (2024).

## 6.2 Results

Table 5 summarizes performance in the deployment setting. All models achieve 84–100% accuracy, higher than Sec. 5.3. This reflects the evaluation setup: strict path consistency requires correctness across multiple interpretations including less natural ones, while here we evaluate a single answer against the original query, which typically matches the model’s default interpretation.

|         | Qwen | 3 GPT | OSM | KIM | CLA | GEN | Pooled |
|---------|------|-------|-----|-----|-----|-----|--------|
| Dis     | RAG  |       |     |     |     |     |        |
| ×       | ×    | 85.3 | 93.6 | 99.1 | 95.4 | 100 | 94.7   |
| ✓       | ×    | 88.1 | 98.2 | 99.1 | 94.5 | 97.2 | 95.4   |
| ×       | ✓    | 90.8 | 99.1 | 97.2 | 97.2 | 100 | 96.9   |
| ✓       | ✓    | 94.5 | 92.7 | 94.5 | 96.3 | 100 | 95.6   |
| Pooled  |      | 89.7 | 95.9 | 97.5 | 95.9 | 99.3 | 95.6   |

Table 5: Model Accuracy by Configuration. Dis. = disambiguated input. All values in %.

Uncertainty Separates Correct from Incorrect. Despite sparse errors, median uncertainty values are systematically higher for incorrect predictions. Pooled across all configurations ($N = 2, 180$), incorrect predictions show median $H_I$ of $0.189$ vs. $0.069$ for correct, $H_R$ of $0.442$ vs. $0.301$, and $H_{R|I}$ of $0.072$ vs. $0.003$.
```

### --- Page 0016 ---

```markdown
# RAG Stabilizes Output

RAG substantially reduces $H_{R | I}$ for correct predictions: GPT-OSS drops from 0.113 to 0.030, Qwen3 from 0.046 to 0.006, and KimiK2 from 0.062 to 0.025. However, RAG also reduces $H_{R | I}$ for incorrect predictions (pooled median: 0.104 → 0.052), which may suppress the uncertainty signal that would otherwise flag errors (e.g., GPT-OSS: 0.296 → 0.025). Claude4.5S and Gemini3Pro show minimal $H_{R | I}$ (0.002) with or without RAG, suggesting sufficient internalized domain knowledge.

## AUROC and regime-based diagnostics

In this setting (Sec. 6.1), pooled AUROC is 0.687 [0.629, 0.740] for $H_{R}$ versus 0.648 [0.600, 0.701] for $H_{R | I}$ (not significant, $p = 0.20$). Crucially, $H_{R}$ cannot distinguish whether a high-uncertainty query needs clarification, model review, or both. CLUES is designed to decompose uncertainty rather than maximize single-score prediction, providing diagnostic resolution unavailable from $H_{R}$ alone (Sec. 6.3).

## 6.3 Uncertainty Regime Analysis

Given sparse per-condition errors, we pool all data for regime analysis. We partition queries using pooled median thresholds for $H_{I}$ and $H_{R | I}$, yielding a $2 \times 2$ structure (Table 6).

| Regime   | $H_{I}$ | $H_{R | I}$ | C/I | Err  |
|----------|---------|-------------|-----|------|
| I: Confident  | Low     | Low         | 534/8 | 1.5% |
| II: Ambiguity | High    | Low         | 530/18 | 3.3% |
| III: Instability | Low  | High        | 529/21 | 3.8% |
| IV: Compound   | High   | High        | 492/48 | 8.9% |
| **Total**      |         |             | 2085/95 | 4.4% |

**Table 6:** Uncertainty Regime Analysis. $2 \times 2$ decomposition by $H_{I}$ and $H_{R | I}$ median thresholds. C/I = Correct/Incorrect.

## Error Gradient and Routing Implications

Error rates increase from Regime I (1.5%) to Regime IV (8.9%). Each regime contains $\sim$ 25% of queries, yet Regime IV concentrates 51% of all errors (48/95). A routing strategy sending only Regime IV to human review would examine one quarter of queries while catching half of all failures. Within-regime AUROC in Regime IV ($\approx 0.52$ for all metrics) indicates that continuous entropy values provide limited additional signal once the regime is determined. The decomposition’s value is further demonstrated among high-uncertainty queries where both $H_{I}$ and $H_{R}$ are above median ($N = 712$): these appear identical under conventional metrics, yet $H_{R | I}$ separates them into Regime A (4.3% error) and Regime B (9.2% error).
```

### --- Page 0017 ---

```markdown
# 7 Summary

We introduced CLUES, a framework that decomposes semantic uncertainty into ambiguity ($H_I$) and conditional instability ($H_{R | I}$) via a Schur-complement construction over an interpretation–result bipartite graph. Across open-domain QA and clinical Text-to-SQL, $H_{R | I}$ is competitive with, and often improves upon, state-of-the-art Kernel Language Entropy Nikitin et al. (2024) for failure prediction, while providing diagnostic resolution unavailable from a single score. Regime analysis based on $(H_R, H_{R | I})$ surfaces distinct failure patterns and maps them to targeted interventions: clarification for high ambiguity, human review for high instability. CLUES is model-agnostic and requires only output sampling; future work will explore adaptive routing and uncertainty propagation in agentic pipelines.

# 8 Limitations

## Computational Cost. 
CLUES requires multiple LLM calls per query (interpretations × samples; 9 in our setup). Calls can be parallelized, though cost remains a consideration for large-scale deployment.

## Interpretation Quality. 
CLUES assumes generated interpretations meaningfully capture query ambiguity. If the disambiguation procedure produces trivial or redundant interpretations, $H_I$ may underestimate true input ambiguity, and the Schur complement may not isolate instability effectively. We observed strong results with frontier LLMs, but interpretation quality likely degrades with weaker models.

## Routing Efficiency in Sparse Error Regimes. 
In high-accuracy settings (95.6% in deployment), regime-based routing catches 51% of errors by reviewing 25% of queries, twice the efficiency of random selection, though not yet sufficient for fully automated triage. The fundamental challenge is sparse errors: with only 95 failures (∼ 4% error rate), fine-grained uncertainty signals have limited discriminative power. Applications with higher base error rates may benefit more substantially from regime-based routing.

# References

Anthropic AI (2023)  
Anthropic AI. 2023.  
Model card and evaluations for claude models.
```


### --- Page 0018 ---

```markdown
| Citation                        | Reference                                                                                                           |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Boyd and Vandenberghe (2004)   | Stephen Boyd and Lieven Vandenberghe. 2004. *Convex Optimization*. Cambridge University Press, Cambridge, England.  |
| Deng et al. (2022)             | Naihao Deng, Yuwei Chen, and Yue Zhang. 2022. Recent advances in text-to-SQL: A survey of what we have and what we expect. arXiv preprint arXiv:2208.10099. |
| Desai and Durrett (2024)       | Shreshth Desai and Greg Durrett. 2024. Distinguishing the knowable from the unknowable with language models. arXiv preprint arXiv:2402.03563. |
| Elazar et al. (2024)           | Izhak Elazar, Roei Aharoni, Jonathan Berant, and Reut Tsarfaty. 2024. AMBROSIA: A benchmark for parsing ambiguous questions into database queries. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 11875–11893. |
| Farquhar et al. (2024)         | Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. *Nature*, 630(8017):625–630. |
| Gemini Team (2023)             | Gemini Team. 2023. *Gemini: A family of highly capable multimodal models*. Technical report, Google. Accessed: February 15, 2024. |
|                                 | [Link to document](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf). Accessed: February 15, 2024. |
```

### --- Page 0019 ---

```markdown
| Citation                     | Details                                                                                                                                                                                                 |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ghosh et al. (2025)          | Shrestha Ghosh, Moritz Schneider, Carina Reinicke, and Carsten Eickhoff. 2025. [A survey on LLM-assisted clinical trial recruitment](https://example.com). In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 625–646, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics. |
| Gupta et al. (2025)         | Vivek Gupta, Rui Zhang, and Yue Zhao. 2025. Disambiguate first parse later: Generating interpretations for ambiguity resolution in semantic parsing. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP). |
| Hemkens et al. (2016)       | Lars G Hemkens, Despina G Contopoulos-Ioannidis, and John PA Ioannidis. 2016. Research reproducibility in longitudinal multicenter studies using data from electronic health records. EGEMS (Generating Evidence & Methods to improve patient outcomes), 4(2). |
| Herrett et al. (2017)       | Emily Herrett, Arlene M Gallagher, Krishnan Bhaskaran, Harriet Forbes, Rohini Mathur, Tjeerd van Staa, and Liam Smeeth. 2017. Methods for enhancing the reproducibility of biomedical research findings using electronic health records. BioData Mining, 10(1):1–8. |
| Higham (2002)               | Nicholas J. Higham. 2002. [Computing the nearest correlation matrix—a problem from finance](https://example.com). IMA Journal of Numerical Analysis, 22(3):329–343.                                   |
```

### --- Page 0020 ---

```markdown
| Citation                          | Reference                                                                                                           |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Huang et al. (2025)               | Jiuding Huang, Dianzhi Yang, Kit Lau, Yuxuan Liu, Yu Zhang, Chen Lyu, and Jifeng Chen. 2025. Integrating token-level uncertainty, bidirectional nli, and semantic entropy for robust hallucination detection in large language models. In *IEEE International Conference on Big Data (BigData)*, pages 2945–2954. IEEE. |
| Hüllermeier and Waegeman (2021)   | Eyke Hüllermeier and Willem Waegeman. 2021. Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. *Machine Learning*, 110(3):457–506. |
| Kamkari et al. (2024)             | Priyank Jaini Kamkari, Aitor Lewkowycz, and David Duvenaud. 2024. A bias-variance-covariance decomposition of kernel scores for generative models. *arXiv preprint arXiv:2310.05833*. |
| Kendall and Gal (2017)            | Alex Kendall and Yarin Gal. 2017. What uncertainties do we need in Bayesian deep learning for computer vision? In *Advances in Neural Information Processing Systems (NeurIPS)*, pages 5574–5584. |
| Kim et al. (2022)                 | Daeyoung Kim, Seongsu Bae, Seungho Kim, and Edward Choi. 2022. Uncertainty-aware text-to-program for question answering on structured electronic health records. In *Proceedings of the Conference on Health, Inference, and Learning*, pages 138–151. PMLR. |
| Kim et al. (2024)                 | Yongrae Kim, Hyunseok Park, Seongsu Seo, Gyubok Lee, and Edward Choi. 2024. LG AI Research & KAIST at EHRSQL 2024: Self-training large language models with pseudo-labeled unanswerable questions for a reliable text-to-SQL system on EHRs. In *Proceedings of the Clinical NLP Workshop*. |
```

### --- Page 0021 ---

```markdown
| Citation                     | Authors                                           | Year | Title                                                                                          | Venue                                                   |
|-----------------------------|---------------------------------------------------|------|------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| Koretsky et al. (2025)      | Mathew J Koretsky, Maya Willey, and 1 others     | 2025 | BiomedSQL: Text-to-SQL for scientific reasoning on biomedical knowledge bases.                 | arXiv preprint arXiv:2505.20321.                       |
| Kuhn et al. (2023)          | Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar   | 2023 | Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. | International Conference on Learning Representations (ICLR). |
| Kumar et al. (2025)         | Aditya Kumar, Yuxin Zhang, and Jiahao Chen       | 2025 | Semantic volume: Quantifying and detecting both external and internal uncertainty in LLMs.    | arXiv preprint arXiv:2502.212139.                      |
| Lee et al. (2023)           | Gyubok Lee, Hyeonji Hwang, Seongsu Bae,          | 2023 | EHRSQL: A practical text-to-SQL benchmark for electronic health records.                     | In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track. |
|                             | Yeonsu Kwon, Woncheol Shin, Seongjun Yang,       |      |                                                                                                |                                                         |
|                             | Minjoon Seo, Jong C Lee, and Edward Choi.        |      |                                                                                                |                                                         |
| Lee et al. (2024a)          | Joonho Lee, Seonghyeon Kim, Seoyoung Park,       | 2024a| Improving uncertainty quantification in large language models via semantic embeddings.         | arXiv preprint arXiv:2410.22685.                       |
| Lee et al. (2024b)          | Sangryul Lee, Jiyoun Kim, and Seoyeon Park.      | 2024b| ProbGate at EHRSQL 2024: Enhancing SQL query generation accuracy through probabilistic threshold filtering and error handling. | In Proceedings of the Clinical NLP Workshop.            |
```

### --- Page 0022 ---

```markdown
| Citation | Authors | Title | Source |
|----------|---------|-------|--------|
| Min et al. (2020) | Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. | AmbigQA: Answering ambiguous open-domain questions. | In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5783–5797, Online. Association for Computational Linguistics. |
| Mozes et al. (2025) | Maximilian Mozes, Robert Bamler, and José Miguel Hernández-Lobato. | Semantic uncertainty in advanced decoding methods for llm generation. | arXiv preprint arXiv:2506.17296. |
| Nikitin et al. (2024) | Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka Marttinen. | Kernel language entropy: fine-grained uncertainty quantification for ilms from semantic similarities. | In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA. Curran Associates Inc. |
| OHDSI Community (2024) | OHDSI Community. | Atlas: Open source software for observational data analysis. | [https://github.com/OHDSI/Atlas](https://github.com/OHDSI/Atlas). Accessed: 2026-01-28. |
| OpenAI et al. (2025) | OpenAI, ; Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, and 108 others. | gpt-oss-120b and gpt-oss-20b model card. | Preprint, arXiv:2508.10925. |
```

![Detailed description of the chart](assets/page_0022_img_1.png)

### --- Page 0023 ---

```markdown
| Citation                     | Reference                                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Park et al. (2022)           | Daeyoung Park, Suji Choi, Sunjae Kim, Jongwuk Lee, and Jaegul Choo. 2022. Uncertainty-aware text-to-program for question answering on structured electronic health records. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pages 1914–1929. |
| Park et al. (2024)           | Jaehee Park, Nan Zhang, Xiaohui Xiao, and 1 others. 2024. EHR-SeqSQL: A sequential text-to-SQL dataset for interactively exploring electronic health records. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL). |
| Schur (1917)                 | J Schur. 1917. Über potenzreihen, die im innern des einheitskreises beschränkt sind. J. Reine Angew. Math., 1917(147):205–232. |
| Sun et al. (2025)            | Yixuan Sun, Yichi Wang, and Yang Liu. 2025. Enhancing uncertainty quantification in large language models through semantic graph density. arXiv preprint. |
| Team et al. (2025)           | Kimi Team, Yifan Bai, Yiping Bao, Guandu Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, and 150 others. 2025. Kimi k2: Open agent intelligence. Preprint, arXiv:2507.20534. |
| Team (2025)                  | Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388. |
```

### --- Page 0024 ---

```markdown
| Citation                     | Reference                                                                                                           |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Yu et al. (2018)            | Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3911–3921. |
| Zhang (2005)                | Fuzhen Zhang, editor. 2005. The schur complement and its applications, 2005 edition. Numerical Methods and Algorithms. Springer, New York, NY. |
| Zhang and Choi (2021)       | Michael Zhang and Eunsol Choi. 2021. SituatedQA: Incorporating extra-linguistic contexts into QA. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7371–7387, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. |
| Zhang et al. (2025)         | Yuxin Zhang, Zinan Gao, Zhiming Xu, and Peng Cui. 2025. Inv-entropy: A fully probabilistic framework for uncertainty quantification in language models. arXiv preprint arXiv:2506.09684. |
| Ziletti and D'Ambrosi (2024) | Angelo Ziletti and Leonardo D'Ambrosi. 2024. Retrieval augmented text-to-SQL generation for epidemiological question answering using electronic health records. In Proceedings of the 6th Clinical Natural Language Processing Workshop, pages 47–53, Mexico City, Mexico. Association for Computational Linguistics. |
```

### --- Page 0025 ---

```markdown
| Ziletti and D’Ambrosi (2025) | Angelo Ziletti and Leonardo D’Ambrosi. 2025. Generating patient cohorts from electronic health records using two-step retrieval-augmented text-to-sql generation. Preprint, arXiv:2502.21107. |
```


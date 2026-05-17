# HypRAG  Hyperbolic Dense Retrieval for Retrieval Augmented Generation

### --- Page 0001 ---

```markdown
# HypRAG: Hyperbolic Dense Retrieval for Retrieval Augmented Generation

| Hiren Madhu       | Ngoc Bui         | Ali Maatouk      |
|-------------------|------------------|------------------|
| Leandros Tassiulas| Smita Krishnaswamy| Menglin Yang     |
| Sukanta Ganguly   | Kiran Srinivasan | Rex Ying         |

## Abstract

Embedding geometry plays a fundamental role in retrieval quality, yet dense retrievers for retrieval-augmented generation (RAG) remain largely confined to Euclidean space. However, natural language exhibits hierarchical structure from broad topics to specific entities that Euclidean embeddings fail to preserve, causing semantically distant documents to appear spuriously similar and increasing hallucination risk. To address these limitations, we introduce hyperbolic dense retrieval, developing two model variants in the Lorentz model of hyperbolic space: HyTE-FH, a fully hyperbolic transformer, and HyTE-H, a hybrid architecture projecting pre-trained Euclidean embeddings into hyperbolic space. To prevent representational collapse during sequence aggregation, we introduce the Outward Einstein Midpoint, a geometry-aware pooling operator that provably preserves hierarchical structure. On MTEB, HyTE-FH outperforms equivalent Euclidean baselines, while on RAGBench, HyTE-H achieves up to 29% gains over Euclidean baselines in context relevance and answer relevance using substantially smaller models than current state-of-the-art retrievers. Our analysis also reveals that hyperbolic representations encode document specificity through norm-based separation—with over 20% radial increase from general to specific concepts—a property absent in Euclidean embeddings, underscoring the critical role of geometric inductive bias in faithful RAG systems.¹

¹ The code is available at: [https://anonymous.4open.science/r/HypRAG-30C6](https://anonymous.4open.science/r/HypRAG-30C6).

---

## Machine Learning, ICML

### 1 Introduction

Dense retrieval forms the backbone of retrieval-augmented generation (RAG) systems (Lewis et al., 2020; Fan et al., 2024), where embedding quality directly determines whether generated responses are grounded in evidence or hallucinated. By retrieving relevant documents and conditioning generation on this context, RAG systems produce responses that are more attributable and aligned with verifiable sources (Ni et al., 2025). Yet, despite
```

### --- Page 0002 ---

advances in retrieval architectures, current systems continue to rely on Euclidean embeddings, a choice inherited from standard neural networks rather than from language structure itself.

Natural language inherently exhibits strong hierarchical organization (He et al., 2025b; Robinson et al., 2024), with semantic structure giving rise to locally tree-like neighborhoods. Euclidean spaces struggle to represent such branching hierarchies due to polynomial volume growth (He et al., 2025b), introducing shortcuts between hierarchically distinct regions that distort semantic relationships. In retrieval settings, these distortions can cause semantically distant documents to appear spuriously similar (Radovanovic et al., 2010; Bogolin et al., 2022), degrading retrieval precision (Reimers and Gurevych, 2021): a query about a specific subtopic may retrieve documents from sibling or parent categories that share similarity but lack the required specificity.

![Hierarchies in Text. (A) Documents naturally organize into branching hierarchies where general topics spawn increasingly specific subtopics. Euclidean spaces distort such hierarchies due to crowding effects, while hyperbolic geometry preserves hierarchical relationships through exponential volume growth. (B) Ricci curvature analysis of document embeddings from strong baselines reveals predominantly negative curvature, indicating tree-like semantic structure.](assets/page_0002_img_1.png)

To further see why geometry matters for retrieval, consider a query about transformer attention mechanisms (Figure 1A). Relevant documents form a natural hierarchy—from general concepts like NLP, to transformers, to specific components like multi-head attention—inducing tree-like semantic structure. Euclidean embeddings struggle to preserve this organization: representing both broad topics and specialized descendants forces a trade-off between semantic proximity and fine-grained separation, causing neighborhood crowding and distortion.

### --- Page 0003 ---

```markdown
Hyperbolic geometry resolves this tension through exponential volume growth, allowing general concepts to remain compact while specific documents spread outward. To test whether such structure appears empirically, we analyze Ollivier–Ricci curvature (Ni et al., 2019)—a measure of local geometry where negative values indicate tree-like branching—on graphs built from MS MARCO document embeddings (Bajaj et al., 2016). Across several strong models (Linq Embed Mistral, LLaMA Nemotron 8B, Qwen3 Embedding 4B), curvature distributions are predominantly negative (Figure 1B), providing empirical evidence that retrieval-relevant embeddings exhibit intrinsic hyperbolic structure and motivating hyperbolic geometry as a natural inductive bias for dense retrieval.

Recent work has been exploring hyperbolic geometry for language modeling and RAG systems, though with different focus areas. HELM (He et al., 2025a) introduces a family of hyperbolic language models that operate entirely in hyperbolic space, but these models target text generation rather than retrieval. In the RAG setting, HyperbolicRAG (Cao et al., 2025) projects embeddings into the Poincaré ball to encode hierarchical depth rankings. However, HyperbolicRAG relies on Euclidean encoders to produce the initial embeddings, leaving the fundamental geometric mismatch. Moreover, aggregating token embeddings into document representations poses a challenge that existing work in hyperbolic learning does not address (Yang et al., 2024). As we show in Proposition 4.3, naively averaging tokens in Euclidean space before projecting to hyperbolic space causes representations to collapse toward the origin, destroying the hierarchical structure that is meant to be preserved.

To this end, we introduce hyperbolic dense retrieval for RAG, framing embedding geometry as a design choice for improving evidence selection and grounding at the representation level. We study this through two complementary instantiations. First, HyTE-FH (Hyperbolic Text Encoder, Fully Hyperbolic) operates entirely in the Lorentz model of hyperbolic space, enabling end-to-end representation learning. Second, HyTE-H (Hybrid) maps embeddings from off-the-shelf Euclidean encoders into hyperbolic space, allowing us to build on existing pre-trained Euclidean models. The Lorentz model’s intrinsic geometry enables parameter-efficient scaling: HyTE-H outperforms Euclidean baselines several times (2-3x) its size, reducing memory footprint in resource-constrained settings. To address the aggregation challenge in both instantiations, we introduce the Outward Einstein Midpoint, a geometry-aware pooling operator that amplifies tokens farther from the origin, provably preserving hierarchical structure during pooling.

Through extensive evaluation on RAGBench, we demonstrate that both hyperbolic variants consistently outperform Euclidean baselines in answer relevancy across multiple datasets, while achieving competitive performance on MTEB. Our experiments validate three key findings: (1) hyperbolic retrieval substantially improves RAG performance, with up to 29% gains over Euclidean baselines in context relevance and answer relevance; (2) hyperbolic models naturally encode concept-level hierarchies in their radial structure, with the fully hyperbolic model achieving a 20.2% radius increase from general to specific concepts, while Euclidean
```

### --- Page 0004 ---

```markdown
![Detailed description of the chart](assets/page_0004_img_1.png)

## 2 Related Works

### Text Embeddings and Dense Retrieval
Dense retrieval embeds queries and documents into a shared vector space and ranks candidates by similarity (e.g., dot product or cosine). Transformer bi-encoders (e.g., BERT (Devlin et al., 2019)) are widely used in this context due to their scalability with maximum inner product search (Karpukhin et al., 2020; Reimers and Gurevych, 2019). Most methods train with contrastive objectives using in-batch and hard negatives (Gao et al., 2021; Izacard et al., 2021; Xiong et al., 2021), often following large-scale pretraining plus task-specific fine-tuning (Wang et al., 2022; Li et al., 2023; Nussbaum et al., 2025). More recently, decoder-only embedding models initialize from LLMs to exploit their pretrained linguistic knowledge (Muennighoff et al., 2024; Lee et al., 2024; Zhang et al., 2025). However, most retrievers remain reliant on inner products or distances in Euclidean geometry-an inductive bias often misaligned with the hierarchical structure of language and document collections. We address this gap by introducing hyperbolic geometry for text embeddings to better capture such a hierarchy.

### Retrieval Augmented Generation
RAG grounds LLMs in retrieved evidence to improve factuality and access external knowledge (Oche et al., 2025). It typically retrieves top-k contexts (often via dense retrieval) and conditions generation on them (Lewis et al., 2020). Since the context window is limited, retrieval quality is a key bottleneck for relevance and faithfulness (Friel et al., 2024a). Several methods improve reliability after retrieval: Self-RAG (Asai et al., 2024) and CRAG (Yan et al., 2024) use learned critics to filter or re-rank evidence, while GraphRAG (Han et al., 2024) leverages knowledge graphs for structured subgroup retrieval. These approaches operate downstream of the embedding space and are complementary to our geometric approach. Our goal is to improve RAG upstream by enhancing the retriever representations so that the initial top-k evidence is more reliable under realistic efficiency constraints.

### Hyperbolic Representation Learning
Hyperbolic geometry is primarily known for its ability to better capture hierarchical, tree-like structures (Yang et al., 2023; Peng et al., 2021), which enhances performance in various tasks, including molecular generation (Liu et al., 2019), recommendation (Yang et al., 2021; Li et al., 2021), image retrieval (Khrulkov et al., 2020; Wei et al., 2024; Bui et al., 2025), and knowledge graph embedding (Ganea et al., 2018a; Dhingra et al., 2018). More recently, hyperbolic geometry has also shown promise for multi-modal embedding models (Desai et al., 2023; Ibrahim et al., 2024; Pal et al., 2024) and foundation models (Yang et al., 2025; He et al., 2025a). In contrast to these works, we study how hyperbolic representations can improve retrieval in RAG systems. Concurrently, Cao et al. (2025) use hyperbolic geometry to improve RAG rankings, but obtain hyperbolic embeddings via a simple projection from Euclidean encoders; by contrast, we build on fully hyperbolic encoders trained end-to-end and address key challenges in this setting, including providing the theoretically grounded geometry-aware pooling for document-level representations.
```

### --- Page 0005 ---

```markdown
# 3 Hyperbolic Space Preliminaries

In this section, we go over all the preliminaries of Lorentz model of hyperbolic space and introduce the basic building blocks that create HyTE-FH.

## 3.1 Lorentz Model of Hyperbolic Space

We represent all embeddings in $d$-dimensional hyperbolic space $\mathbb{H}^d_K$ with constant negative curvature $K < 0$ using the Lorentz (hyperboloid) model. In the Lorentz model, hyperbolic space is realized as the upper sheet of a two-sheeted hyperboloid embedded in $\mathbb{R}^{d + 1}$,

$$
\mathbb{H}^d_K = \left\{ x \in \mathbb{R}^{d + 1} \mid \langle x, x \rangle_L = \frac{1}{K}, x_0 > 0 \right\},
$$

where the Lorentzian inner product is defined as $\langle x, y \rangle_L = -x_0 y_0 + \sum_{i=1}^d x_i y_i$. This formulation admits closed-form expressions for geodesic distances, barycentric operations, and parallel transport, and expresses similarity directly through Lorentzian inner products. The geodesic distance between two points $x, y \in \mathbb{H}^d_K$ is given by 

$$
d_K(x, y) = \frac{1}{\sqrt{-K}} \cosh^{-1} \left( \langle x, y \rangle_L \right),
$$ 

which is a monotone function of the Lorentzian inner product.

To support optimization, we make use of exponential and logarithmic maps between the manifold and its tangent spaces. For a point $x \in \mathbb{H}^d_K$, the logarithmic map $\log_x(\cdot)$ maps nearby points to the tangent space $T_x \mathbb{H}^d_K$, while the exponential map $\exp_x(\cdot)$ maps tangent vectors back to the manifold. These operators are used only where necessary for gradient-based updates, ensuring that all representations remain on $\mathbb{H}^d_K$ and preserving the hierarchical structure induced by negative curvature.

## 3.2 Hyperbolic Transformer Components

Standard operations cannot be applied directly in hyperbolic space, as they may violate the manifold constraint (Yang et al., 2024). To address this, we introduce hyperbolic components that serve as the building blocks for our embedding model. These operations are performed via a re-centering procedure that applies Euclidean operations in a latent space and maps the result back to the Lorentz model. By doing so, the resulting vector is constructed to satisfy the Lorentz constraint, thereby preserving the hyperbolic structure of representations. We present these operations as follows.

### Lorentz Linear Layer

Given curvatures $K_1, K_2$, and parameters $W \in \mathbb{R}^{(n + 1) \times m}$ and $b \in \mathbb{R}^m$ with 

$$
z = W^T x + b,
$$ 

the Lorentzian linear transformation (Yang et al., 2024) is the map HLT: $\mathbb{L}^{k,n} \to \mathbb{L}^{k,m}$ given by,
```

### --- Page 0006 ---

```markdown
# Hyperbolic Layer Normalization

Given token embeddings $X = \{x^n_{i}\}_{i=1}^n \subset \mathbb{H}^d_K$, hyperbolic layer normalization is defined as

$$
\text{HypLayerNorm}(X) = \left( \sqrt{\frac{K_1}{K_2}} \| \mathbf{z} \|^2_2 - \frac{1}{K_2} \sqrt{\frac{K_1}{K_2}} \mathbf{z} \right)
$$

where $z = f_{LN} \left( x_{[1:d]} \right)$, $f_{LN}(\cdot)$ denotes standard Euclidean LayerNorm applied to the spatial components of the embedding, and $K_1, K_2 > 0$ are input and output curvature respectively.

## Lorentz Residual Connection

Let $x, f(x) \in \mathbb{K}^n$ where $x$ is an input vector and $f(x)$ is the output of a neural network $f$. Then, the Lorentzian residual connection (He et al., 2025e) is given by $x \oplus_L f(x) = \alpha_1 x + \alpha_2 y$, where

$$
\alpha_i = w_i / \left( \sqrt{-R} \| w \| x + w_2 f(x) \|_c \right), \quad \text{for } i \in \{0, 1\},
$$

where $\alpha_1, \alpha_2$ are weights parametrized by constants $(w_1, w_2) \in \mathbb{R}^2 \setminus \{(0, 0)\}$.

## Hyperbolic Self-Attention

In hyperbolic attention, similarity is governed by hyperbolic geodesic distance (Ganea et al., 2018b). Given token embeddings $X = \{x^n_{i}\}_{i=1}^n \subset \mathbb{H}^d_K$, queries, keys, and values are computed via Lorentz-linear transformations as $Q = \text{HLT}(X; W^0, b^0)$, $V = \text{HLT}(X; W^V, b^V)$, and $K = \text{HLT}(X; W^K, b^K)$, where $\text{HLT}(\cdot)$ denotes a linear map in Lorentz space. Attention weights are computed using squared hyperbolic geodesic distances (He et al., 2025c; Chen et al., 2022) as

$$
v_{i,j} = \frac{\exp\left(-d^2_L(q_j, k_j) / \sqrt{m}\right)}{\sum_{l=1}^n \exp\left(-d^2_L(q_l, k_l) / \sqrt{m}\right)},
$$

with head dimension $m$. This prioritizes geodesic proximity rather than angular similarity. The attended representation is obtained via a Lorentzian weighted midpoint

$$
\text{Att}_\ell(x_i) = \frac{\sum_{j=1}^n \lambda_j v_j}{\sqrt{-K} \| \sum_{j=1}^n \lambda_j v_j \|_c},
$$

where $\lambda_j = \mu_j, 0$ is the Lorentz factor. Unlike Euclidean averaging, this aggregation remains on $\mathbb{H}^d_K$ and preserves radial structure during contextualization.
```


### --- Page 0007 ---

```markdown
# 4 Method

We now outline our approach to hyperbolic dense retrieval. We begin by introducing the two proposed HyTE architectures, followed by an analysis of why naïve pooling strategies fail in hyperbolic space, and conclude by presenting our geometry-aware aggregation operator.

## 4.1 Architecture

![HyTE Architecture. A) HyTE-FH Encoder Block, B) HyTE-FH architecture, C) HyTE-H Architecture.](assets/page_0007_img_1.png)

The hyperbolic encoder components described in Section 3 form the building blocks (Figure 2A) of HyTE-FH, our fully hyperbolic transformer (Figure 2B). By operating entirely within hyperbolic geometry, HyTE-FH preserves hierarchical structure throughout token-level contextualization, aggregation, and similarity computation, with semantic abstraction and specificity encoded along radial dimensions. HyTE-H (Figure 2C) instead projects pretrained Euclidean representations into hyperbolic space, which allows hyperbolic geometry to be leveraged with a strong initialization and avoiding the need to train a fully hyperbolic encoder from scratch.
```

### --- Page 0008 ---

```markdown
While hyperbolic self-attention enables geometry-consistent contextualization at the token level, dense retrieval requires aggregating variable-length sequences into fixed-dimensional representations. Standard approaches map representations to tangent space, aggregate in Euclidean space, then map back to the manifold (Yang et al., 2024; Desai et al., 2023), but this distorts hierarchical structure encoded in radial depth in both the models. In the following subsections, we analyze this failure mode formally and introduce a pooling operator designed to preserve hierarchical information.

## 4.2 Failure of Naïve Hyperbolic Pooling

Naïve pooling strategies that aggregate in Euclidean space (Yang et al., 2024; Desai et al., 2023) systematically contract representations toward the origin. This follows from hyperbolic convexity: for any $\{x_i\}^n_{i=0} \subset \mathbb{H}^d_K$, the barycenter lies strictly closer to the origin than the maximum-radius point unless all points coincide. Consequently, document-level embeddings lose the radial separation that encodes document specificity through hierarchical depth. To address this failure mode, we first establish notation for projecting ambient vectors onto the hyperboloid and measuring radial depth.

### Definition 4.1 (Lorentz Projection).
For $v \in \mathbb{R}^{d+1}$ with $\langle v, v \rangle_L < 0$ and $\nu_0 > 0$, let $\Pi_K(v) = \frac{v}{\sqrt{-\langle v, v \rangle_L}}$ denote the unique positive rescaling satisfying 

$$
(\Pi_K(v), \Pi_K(v))_L = 1 / K.
$$

### Definition 4.2 (Radial Depth).
The radial depth of $x \in \mathbb{H}^d_K$ is $r(x) = x_0$. Since $x_0 = \frac{1}{-K} \cosh(-K \rho)$ where $\rho = d_K(0, x)$, ordering by $x_0$ is equivalent to ordering by intrinsic hyperbolic distance from the origin.

Semantically, radial depth encodes concept specificity: general concepts should be near the origin while fine-grained entities should have larger radii. This provides a measurable signature for evaluating whether models learn meaningful hierarchical structure. The simplest aggregation strategy is Euclidean averaging in the ambient space followed by reprojection. However, this approach provably contracts representations toward the origin (Ganea et al., 2018a; Chami et al., 2019), destroying hierarchical structure encoded in radial depth. We formalize this in the following proposition.

### Proposition 4.3 (Euclidean Mean Contracts).
Let $\{x_i\}^n_{i=1} \subset \mathbb{H}^d_K$ with $n \geq 2$. Define the Euclidean mean $x = \frac{1}{n} \sum^n_{i=1} x_i$ and its projection onto the hyperboloid $m_{Euc} = \Pi_K(x)$. Then, we have
```

### --- Page 0009 ---

```markdown
# Page 0009

$$
r(mEuc) \leq \frac{1}{n} \sum_{i=1}^{n} r(x_i),
$$

with equality if and only if all $x_i$ are identical.

The proof of this Proposition is available in Appendix A.2. This failure motivates a precise characterization of desirable pooling behavior. We formalize the requirement that pooling should preserve, rather than collapse, radial structure.

### Definition 4.4 (Outward Bias).
A pooling operator $\mathcal{P} : \mathcal{H}^n_K \to \mathcal{H}^n_K$ is outward-biased if $r(\mathcal{P}(\{x_i\}_{i=1}^{n})) \geq r$, where $r$ is the weighted mean radius.

A natural alternative is a weighted aggregation scheme in which token contributions are modulated by their relative importance. For example, Zhu et al. (2020) adopt the Einstein midpoint, the canonical barycenter in hyperbolic space (Gulcure et al., 2019), to emphasize semantically specific tokens during pooling: since points near the boundary receive higher weight via the Lorentz factor $\lambda_i = x_{i,0}$, more informative content should dominate the aggregate. However, we show this intuition is misleading: the implicit radial weighting is fundamentally insufficient to counteract hyperbolic contraction at the document level.

### Proposition 4.5 (Implicit Radial Weighting is Insufficient).
The Einstein midpoint weights points by the Lorentz factor $\lambda_i = x_{i,0}$, but this weighting grows as $\exp(-\sqrt{-K} \rho)$ while hyperbolic volume grows as $\exp((d-1) \sqrt{-K} \rho)$. Specifically, for a point $x \in \mathcal{H}^n_K$ at hyperbolic distance $\rho$ from the origin $o = (1 / \sqrt{-K}, 0, \ldots, 0)$, we have

$$
x_0 = \frac{1}{\sqrt{-K}} \cosh(\sqrt{-K} \rho) \sim \frac{1}{2\sqrt{-K}} \exp(\sqrt{-K} \rho)
$$

as $\rho \to \infty$. Thus, the Lorentz factor weighting undercompensates for the exponential growth of hyperbolic balls at large radii by a factor of $\exp((d - 2) \sqrt{-K} \rho)$.

These results establish that neither Euclidean averaging nor the standard Einstein midpoint satisfies the outward-bias property required for hierarchy-preserving aggregation. This motivates the design of a pooling operator with explicit radial amplification. The proof of this Proposition is available in Appendix A.3.

### 4.3 Outward Einstein Midpoint Pooling
```

### --- Page 0010 ---

```markdown
![Outward Einstein Midpoint. Size of token shows its contribution towards aggregation.](assets/page_0010_img_1.png)

To mitigate radial contraction during aggregation, we introduce the Outward Einstein Midpoint, a geometry-aware pooling operator that explicitly amplifies the contribution of tokens with larger hyperbolic radius. Let $\{x_i\}_{i=1}^n \subset \mathbb{H}^d_k$ denote a sequence of token embeddings, with optional attention weights $w_i \geq 0$, and $\lambda_i$ denoting the Lorentz factors. We define a radius-dependent weighting function

$$
\phi(x_i) = x_{i,0}^p, \quad p > 0,
$$

which is monotone in the radial coordinate. The Outward Einstein Midpoint is then given by

$$
m^{OEM}_{k,p} = \frac{\sum_{i=1}^n (w_i \phi(x_i)) \lambda_i x_i}{\sum_{i=1}^n (w_i \phi(x_i)) \lambda_i},
$$

followed by reprojection onto the hyperboloid $\mathbb{H}^d_k$.

As shown in Figure 3, by construction, this operator assigns disproportionately higher weight to tokens located farther from the origin, counteracting the contraction inherent to naïve averaging. We now establish theoretical guarantees for the Outward Einstein Midpoint, showing that it systematically improves upon the standard Einstein midpoint in preserving radial structure.

**Theorem 4.6 (OEM Pre-Projection Bound).**  
Let $v = \sum_{i=1}^n w_i x_i^{p+1}$ where $w_i \propto w_ix_{i,0}^p$ are the normalized OEM weights. Then, for $p \geq 0$, we have
```

### --- Page 0011 ---

```markdown
We apply Chebyshev’s sum inequality to the co-monotonic sequences $a_i = x_{0}^{p+1}$ and $b_i = x_{i,0}$ to prove this. Full proof can be found in Appendix A.4. While projection onto $H^d_K$ contracts the radial coordinate, the OEM’s concentration of weight on high-radius tokens inflates the pre-projection average, counteracting this effect. Theorem 4.6 establishes that OEM increases the pre-projection radial coordinate. The following theorem shows a stronger result: OEM provably dominates the standard Einstein midpoint in preserving radial structure.

### Theorem 4.7 (OEM Outward Bias).
Let $m^{\text{Ein}}_K$ denote the standard Einstein midpoint ($p = 0$) and $m^{\text{OEM}}_{K,p}$ the Outward Einstein Midpoint. Then, for all $p \ge 1$:

$$
r(m^{\text{OEM}}_{K,p}) \ge r(m^{\text{Ein}}_K)
$$

The OEM weights $w_i \propto w^{p}_{x_{i,0}}$ concentrate more mass on high-radius points than the Einstein weights $w_p x_{i,0}$, increasing the pre-projection time component while reducing pairwise dispersion. Full proof in Appendix A.5. Together, these results establish that the Outward Einstein Midpoint provably preserves hierarchical structure during aggregation, in contrast to both Euclidean averaging and the standard Einstein midpoint. We validate this empirically through concept-level hierarchy analysis (Section 5.2), showing that models using OEM pooling maintain monotonically increasing radii across semantic specificity levels—a property absent in Euclidean baselines.

## 4.4 Training Methodology

We train the hyperbolic encoder in three stages, with all objectives operating directly on the Lorentz manifold using geodesic-based similarity.

### Stage 1: Hyperbolic Masked Language Modeling.
We initialize via masked language modeling (MLM), following the standard BERT objective in hyperbolic space. Contextualization is performed through hyperbolic self-attention, with all intermediate representations on the hyperboloid. Predictions are produced using a Lorentzian multinomial logistic regression (LorentzMLR) (Bdeir et al., 2024) head, which defines class logits via Lorentzian inner products. Only HyTE-FH is trained on MLM, while for HyTE-H we choose a pre-trained Euclidean model as the MLM base to leverage a stronger initialization in low-resource settings.

### Stage 2: Unsupervised Contrastive Pre-Training.
We fine-tune the resulting MLM model on query–document pairs by minimizing unsupervised contrastive loss. Similarity is defined as negative geodesic distance.
```


### --- Page 0012 ---

```markdown
$$
s(q, d) = -d_k(q, d). \text{ The contrastive loss over in-batch negatives is }
$$
$$
\mathcal{L}_{ctr} = -\frac{1}{N} \sum_{i=1}^{N} \log \exp\left(\frac{s(q_i, d_i)}{\tau}\right),
$$
where $\tau > 0$ is a temperature parameter.

### Stage 3: Supervised Contrastive Learning Fine-tuning.
In the final stage of training, we further fine-tune the encoder using supervised contrastive learning on labeled query–document data. Given a query $q_i$, a set of relevant documents $D_i^+$, and a set of non-relevant documents $D_i^-$, the supervised contrastive objective encourages the query representation to be closer to all relevant documents than to non-relevant ones.

$$
\mathcal{L}_{sup} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\sum_{d^+ \in D_i^+} e^{p_h} \exp\left(\frac{s(q_i, d^+)}{\tau}\right)}{\sum_{d \in D_i^+} \nu_{d_i} \exp\left(\frac{s(q_i, d)}{\tau}\right)},
$$
where $\tau > 0$ is a temperature parameter. This stage explicitly aligns hyperbolic distances with supervised relevance signals, refining retrieval behavior beyond unsupervised co-occurrence structure.

### Retrieval-Augmented Generation.
At inference time, the trained hyperbolic encoder is used to retrieve the top-$k$ documents $C$ for a given query. These retrieved documents are then provided as context to a downstream generative language model. Prompt formatting and generation follow standard practice and are provided in Appendix B. We present runtime and computational complexity in Appendix D.

## 5 Experiments and Results

| Model          | Mean (Task) | Mean (TaskType) |
|----------------|-------------|------------------|
| EucBERT        | 54.11       | 51.31            |
| HyTE-H$Euc$    | 54.57       | 53.71            |
| HyTE-FH        | 56.41       | 53.75            |

Table 1: Performance on MTEB benchmark. We report mean scores across tasks and task types. HyTE-FH performs best among the three models.
```

### --- Page 0013 ---

```markdown
| Model        | F CR  | AR CR | AR F  | CR F  | CR AR | IF CR | AR IF | CR IF | CR AR | IF AR | CR AR | IF CR | AR CR | IF CR | AR F  | CR F  | AR F  | CR F  | AR F  | CR F  |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Euclidean    | 0.598 | 0.798 | 0.647 | 0.685 | 0.583 | 0.582 | 0.654 | 0.644 | 0.641 | 0.642 | 0.646 | 0.625 | 0.679 | 0.475 | 0.272 | 0.062 | 0.314 | 0.719 | 0.835 | 0.841 |
| HyTE-H       | 0.709 | 0.677 | 0.857 | 0.773 | 0.743 | 0.717 | 0.857 | 0.773 | 0.743 | 0.717 | 0.857 | 0.773 | 0.743 | 0.717 | 0.857 | 0.773 | 0.743 | 0.717 | 0.857 | 0.773 |

### Table 2: RAG benchmark results comparing our model variants.

| Model        | F CR  | AR CR | AR F  | CR F  | CR AR | IF CR | AR IF | CR IF | CR AR | IF AR | CR AR | IF CR | AR CR | IF CR | AR F  | CR F  | AR F  | CR F  | AR F  | CR F  |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ModernBERT   | 0.617 | 0.748 | 0.632 | 0.656 | 0.895 | 0.573 | 0.632 | 0.709 | 0.746 | 0.567 | 0.639 | 0.656 | 0.657 | 0.575 | 0.678 | 0.675 | 0.675 | 0.675 | 0.675 | 0.675 |

### Table 3: RAG benchmark results comparing our hybrid model with state-of-the-art embedding models. HyTE-H demonstrates competitive performance particularly in context relevance and answer relevance.

5.1 Experimental Setup

**Datasets.** We pre-train our models using publicly available corpora following the data curation and filtering protocols introduced in nomic-embed (Nussbaum et al., 2025). For masked language modeling (MLM), we use the high-quality 2023 Wikipedia dump, which provides broad topical coverage and long-form text suitable for learning general-purpose semantic representations. For contrastive pre-training, we leverage approximately 235 million text pairs curated and filtered as described in (Nussbaum et al., 2025), designed to encourage semantic alignment across paraphrases and related content at scale. Finally, for task-specific fine-tuning, we use the training splits of the BEIR benchmark (Thakur et al., 2021), which comprises a diverse collection of retrieval tasks spanning multiple domains and query styles.

**Evaluation Benchmarks.** We evaluate our approach on two complementary benchmarks: (1) the Massive Text Embedding Benchmark (MTEB) (Muennighoff et al., 2023) to assess embedding quality across diverse tasks, and (2) RAGBench (Friel et al., 2024b) for end-to-end RAG system evaluation. In MTEB, we particularly use the English part of the benchmark. RAGBench evaluates RAG systems on domain-specific question-answering datasets including CovidQA, Cuad, Emanual, DelucionQA, and ExpertQA.
```

### --- Page 0014 ---

```markdown
## Baselines

We adopt different baseline strategies for our two models based on their training paradigms. For HyTE-FH, which is pre-trained from scratch, we train a fully Euclidean equivalent called EucBERT using the same architecture and training setup. This controlled comparison isolates the contribution of hyperbolic geometry. We also evaluate HyTE-H$^{Euc}$, a hybrid hyperbolic model initialized with EucBERT. The three models are evaluated on MTEB and RAGBench. For HyTE-H$^{bert}$, which is fine-tuned with modernbert-base (Warner et al., 2024) as base model, we compare against state-of-the-art embedding models smaller than 500M parameters, including e.g. multilingual-base (Zhang et al., 2024), KaLM-embedding-multilingual-mini-v1 (Hu et al., 2025), and embeddinggemma-300 (Vera et al., 2025).

## Metrics

For MTEB, we report mean scores across tasks and task types. For RAG evaluation, we measure three key metrics using RAGAS (Es et al., 2024): (1) Faithfulness, which assesses whether generated answers are grounded in the retrieved context; (2) Context Relevance, which measures how relevant the retrieved documents are to the query; and (3) Answer Relevance, which evaluates how well the generated answer addresses the user’s question.

## Implementation

We implement all hyperbolic models using HyperCore (He et al., 2025d) and train on NVIDIA H100 GPUs. All three models, HyTE-FH, HyTE-H, and EucBERT, share the same architecture, each containing 149M parameters with 12 transformer layers and 768-dimensional embeddings. For generation and judging, we use Llama-3.1-8B-Instruct (Weerawardenia et al., 2025). For RAG benchmarks, we fix the retrieval context window size to 5 for all models to ensure a controlled comparison; we additionally report ablations with larger context sizes in Appendix Table A3.

## 5.2 Results

### MTEB Benchmark

Table 1 reports performance on the MTEB benchmark. HyTE-FH achieves the highest mean score across tasks (56.41), outperforming both EucBERT (54.11) and HyTE-H$^{Euc}$ (54.57). On the task-type mean, HyTE-FH and HyTE-H$^{Euc}$ perform comparably (53.75 and 53.71, respectively), with both surpassing EucBERT (51.31). These results demonstrate that hyperbolic representations not only improve RAG retrieval but also remain competitive on general-purpose embedding benchmarks. We present task-wise results in Table A1.
```

### --- Page 0015 ---

```markdown
![Empirical validation of hierarchical encoding. Left: Euclidean models show flat or decreasing norms. Middle: HyTE-H demonstrate increasing norms with fine-tuning enhancing this trend. Right: HyTE-FH achieves +20.2% total increase from L1 to L5. Bottom: Normalized comparison and percent change summary highlighting the contrasting behaviors of different geometric approaches.](assets/page_0015_img_1.png)

## RAG Benchmark Results

Table 2 presents RAG benchmark results across five datasets. HyTE-FH achieves the best average performance across all three metrics: faithfulness (0.732), context relevance (0.848), and answer relevance (0.765). HyTE-H$_{Euc}$ ranks second overall, with both hyperbolic variants substantially outperforming EucBERT. On individual datasets, HyTE-FH leads on CovidQA, Cuad, DelusionQA, and ExpertQA, while HyTE-H$_{Euc}$ achieves the best context and answer relevance on Emanual. These results demonstrate that hyperbolic geometry consistently improves retrieval quality for RAG across diverse domains.

Table 3 reports RAG performance across five datasets. HyTE-H$_{bert}$ consistently outperforms strong Euclidean embedding baselines across all metrics, with particularly large gains in context relevance and answer relevance. These improvements indicate that hyperbolic representations are more effective at retrieving structurally relevant evidence, which is critical for downstream generation quality in RAG pipelines. In qualitative case studies shown in Appendix E.1, we observe that Euclidean models frequently fail to retrieve key supporting passages.
```

### --- Page 0016 ---

```markdown
## Concept-Level Hierarchy Analysis

A central motivation for hyperbolic embeddings is their capacity to preserve hierarchical relationships (Section 4.2). To understand how models capture document hierarchy, we analyze learned radii (distances from the origin in the Poincaré ball) across five hierarchical levels: from Level 1 (most general, e.g., document-level topics) to Level 5 (most specific, e.g., fine-grained entities). Figure 4 presents these results. The fully hyperbolic model demonstrates clear hierarchical organization with radii increasing monotonically from Level 1 (2.902) to Level 5 (3.488, +20.2%). This shows the model naturally places general concepts near the origin and specific details toward the boundary, consistent with hyperbolic geometry, where proximity to the origin represents generality. Euclidean models show flat or decreasing distributions. Baselines maintain constant norms across levels or decrease norm by 30%, reflecting inverted structure. Hybrid models exhibit substantially larger radii from the hyperbolic component. The fine-tuned hybrid increases from 116.9 to 146.7, showing that fine-tuning induces structured hierarchy. We have attached the dataset for this case study in the supplementary material. The concept level hierarchy data is available in Appendix C.

## Ablation Studies

We compare two pooling strategies for aggregating token embeddings into document representations: CLS token pooling and OEM pooling. CLS pooling uses the representation of a special classification token, while OEM pooling performs geometry-aware aggregation directly in hyperbolic space. Table 4 shows that OEM pooling yields higher performance across both mean task and mean task-type metrics on MTEB retrieval tasks, indicating more effective document-level aggregation in the hyperbolic setting. We also show that using geodesic distance in the contrastive objective outperforms the Lorentz inner product (Appendix Table A2), suggesting better alignment of representations on the manifold. Additionally, hyperbolic models maintain strong performance with smaller retrieval budgets, whereas Euclidean baselines require larger context windows to achieve comparable results (Appendix Table A3).

### Table 4: Comparison of pooling strategies on MTEB tasks. OEM pooling leverages hyperbolic geometry for improved performance.

| Pooling Strategy | Mean (Task) | Mean (TaskType) |
|------------------|-------------|------------------|
| CLS Token        | 49.33       | 48.90            |
| OEM              | 56.41       | 53.75            |
```


### --- Page 0017 ---

```markdown
## 6 Conclusion

We introduced hyperbolic dense retrieval for RAG, showing that aligning embedding geometry with the hierarchical structure of language improves faithfulness and answer quality. Our approach preserves document-level structure during aggregation through a geometry-aware pooling operator, addressing a key failure mode of Euclidean retrieval pipelines. Across evaluations, we observe consistent gains using models substantially smaller than current state-of-the-art retrievers, highlighting the effectiveness of hyperbolic inductive bias over scale alone. Case studies further show that hyperbolic representations organize documents by specificity through norm-based separation, a property absent in Euclidean embeddings. These findings suggest that embedding geometry is a central design choice for reliable retrieval in RAG systems, with implications for future scalable and multimodal retrieval architectures.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, specifically dense retrieval for retrieval-augmented generation systems. By improving the geometric fidelity of document embeddings, our approach aims to reduce retrieval errors that can lead to hallucinated or poorly grounded responses in RAG systems. We believe more accurate retrieval contributes positively to the reliability of AI-generated content. Additionally, our fully hyperbolic model demonstrates improved parameter efficiency, which may reduce computational costs and environmental impact associated with training and deploying embedding models. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi (2024)

P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao, X. Liu, R. Majumder, A. McNamara, B. Mitra, T. Nguyen, et al. (2016)

- **Self-RAG: learning to retrieve, generate, and critique through self-reflection.**  
  In The Twelfth International Conference on Learning Representations,  
  External Links: [Link](#)  
  Cited by: §2.

- **Ms marco: a human generated machine reading comprehension dataset.**  
  arXiv preprint arXiv:1611.09268.  
  Cited by: §1.
```

### --- Page 0018 ---

```markdown
| Authors                                           | Title                                                                                          | Source                                                                                          | External Links | Cited by |
|---------------------------------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|----------------|----------|
| A. Bdeir, K. Schwethelm, and N. Landwehr (2024)  | Fully hyperbolic convolutional neural networks for computer vision.                          | In The Twelfth International Conference on Learning Representations,                            | [Link](#)      | $4.4$    |
| S. Bogolin, I. Croitoru, H. Jin, Y. Liu, and S. Albanie (2022) | Cross modal retrieval with querybank normalisation.                                          | In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5194–5205. |                | $1$      |
| N. Bui, M. Yang, R. Chen, L. Neves, M. Ju, R. Ying, N. Shah, and T. Zhao (2025) | Learning along the arrow of time: hyperbolic geometry for backward-compatible representation learning. | In Forty-second International Conference on Machine Learning,                                   | [Link](#)      | $2$      |
| L. Cao, R. Wang, J. Li, Z. Zhou, and M. Yang (2025) | HyperbolicRAG: enhancing retrieval-augmented generation with hyperbolic representations.   | arXiv preprint arXiv:2511.18808.                                                               |                | $1$, $2$ |
| I. Chami, Z. Ying, C. Ré, and J. Leskovec (2019) | Hyperbolic graph convolutional neural networks.                                             | Advances in neural information processing systems 32.                                          |                | $4.2$    |
| W. Chen, X. Han, Y. Lin, H. Zhao, Z. Liu, P. Li, M. Sun, and J. Zhou (2022) | Fully hyperbolic neural networks.                                                            | In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5672–5686. |                | $3.2$    |
```

### --- Page 0019 ---

```markdown
| Authors                                           | Title                                                                                          | Source                                                                                                           | Cited by |
|---------------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------|
| K. Desai, M. Nickel, T. Rajpurohit, J. Johnson, and S. R. Vedantam (2023) | Hyperbolic image-text representations.                                                        | In International Conference on Machine Learning, pp. 7694–7731.                                                 | Cited by: $2, $4.1, $4.2. |
| J. Devlin, M. Chang, K. Lee, and K. Toutanova (2019) | Bert: pre-training of deep bidirectional transformers for language understanding.              | In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186. | Cited by: $2. |
| B. Dhingra, C. Shallue, M. Norouzi, A. Dai, and G. Dahl (2018) | Embedding text in hyperbolic spaces.                                                         | In Proceedings of the Twelfth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-12), G. Glavaš, S. Somasundaran, M. Riedl, and E. Hovy (Eds.), New Orleans, Louisiana, USA, pp. 59–69. External Links: [Link](#), [Document](#) | Cited by: $2. |
| S. Es, J. James, L. E. Anke, and S. Schockaert (2024) | Ragas: automated evaluation of retrieval augmented generation.                                  | In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pp. 150–158. | Cited by: $5.1. |
| W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T. Chua, and Q. Li (2024) | A survey on rag meeting llms: towards retrieval-augmented large language models.              | In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pp. 6491–6501.         | Cited by: $1. |
```

### --- Page 0020 ---

```markdown
| Authors                                      | Title                                                                                                   | Source                                                                                          | Cited by |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|----------|
| R. Friel, M. Belyi, and A. Sanyal (2024a)   | Ragbench: explainable benchmark for retrieval-augmented generation systems.                            | arXiv preprint arXiv:2407.11005.                                                                | §2      |
| R. Friel, M. Belyi, and A. Sanyal (2024b)   | Ragbench: explainable benchmark for retrieval-augmented generation systems.                            | arXiv preprint arXiv:2407.11005.                                                                | §5.1    |
| O. Ganea, G. Bécigneul, and T. Hofmann (2018a) | Hyperbolic neural networks.                                                                             | Advances in neural information processing systems 31.                                          | §2, §4.2 |
| O. Ganea, G. Bécigneul, and T. Hofmann (2018b) | Hyperbolic neural networks.                                                                             | In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, Red Hook, NY, USA, pp. 5350–5360. | §3.2    |
| T. Gao, X. Yao, and D. Chen (2021)          | SimCSE: simple contrastive learning of sentence embeddings.                                             | In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, M. Moens, X. Huang, L. Specia, and S. W. Yih (Eds.), Online and Punta Cana, Dominican Republic, pp. 6894–6910. | §2      |
| C. Gulcehre, M. Denil, M. Malinowski, A. Razavi, R. Pascanu, K. M. Hermann, P. Battaglia, V. Bapst, D. Raposo, A. Santoro, and N. de Freitas (2019) | Hyperbolic attention networks.                                                                          | In International Conference on Learning Representations.                                       | §4.2    |
```

### --- Page 0021 ---

```markdown
| Citation                                                                                          | Cited By  |
|---------------------------------------------------------------------------------------------------|-----------|
| H. Han, Y. Wang, H. Shomer, K. Guo, J. Ding, Y. Lei, M. Halappanavar, R. A. Rossi, S. Mukherjee, X. Tang, et al. (2024) | $2$       |
| Retrieval-augmented generation with graphs (graphrag).                                          |           |
| arXiv preprint arXiv:2501.00309.                                                                |           |
|                                                                                                   |           |
| N. He, R. Anand, H. Madhu, A. Maatouk, S. Krishnaswamy, L. Tassiulas, M. Yang, and R. Ying (2025a) | $1$, $2$  |
| HELM: hyperbolic large language models via mixture-of-curvature experts.                        |           |
| In The Thirty-ninth Annual Conference on Neural Information Processing Systems,                   |           |
| External Links: [Link](#)                                                                        |           |
|                                                                                                   |           |
| N. He, J. Liu, B. Zhang, N. Bui, A. Maatouk, I. King, M. Yang, M. Weber, and R. Ying (2025b)    | $1$       |
| Position: beyond euclidean – foundation models should embrace non-euclidean geometries.         |           |
| In The Fourth Learning on Graphs Conference,                                                     |           |
| External Links: [Link](#)                                                                        |           |
|                                                                                                   |           |
| N. He, H. Madhu, N. Bui, M. Yang, and R. Ying (2025c)                                          | $3$       |
| Hyperbolic deep learning for foundation models: a survey.                                       |           |
| In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2,   |           |
| pp. 6021–6031.                                                                                   |           |
|                                                                                                   |           |
| N. He, M. Yang, and R. Ying (2025d)                                                              | $5.1$     |
| Hypercore: the core framework for building hyperbolic foundation models with comprehensive modules. |           |
| arXiv preprint arXiv:2504.08912.                                                                |           |
|                                                                                                   |           |
| N. He, M. Yang, and R. Ying (2025e)                                                              | $3.2$     |
| Lorentzian residual neural networks.                                                              |           |
| In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1,    |           |
| KDD ’25, New York, NY, USA, pp. 436–447.                                                        |           |
| External Links: ISBN 9798400712456, [Link](#), [Document](#)                                    |           |
```

### --- Page 0022 ---

```markdown
| Authors                                                                 | Title                                                                                                   | Citation Information                      |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------|
| X. Hu, Z. Shan, X. Zhao, Z. Sun, Z. Liu, D. Li, S. Ye, X. Wei, Q. Chen, B. Hu, et al. (2025) | Kalm-embedding: superior training data brings a stronger embedding model.                               | arXiv preprint arXiv:2501.01028. Cited by: §5.1 |
| S. Ibrahimi, M. G. Atigh, N. V. Noord, P. Mettes, and M. Worring (2024) | Intriguing properties of hyperbolic embeddings in vision-language models.                               | Transactions on Machine Learning Research. Cited by: §2 |
| G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, and E. Grave (2021) | Unsupervised dense information retrieval with contrastive learning.                                     | arXiv preprint arXiv:2112.09118. Cited by: §2 |
| V. Karpukhin, B. Oguz, S. Min, P. S. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih (2020) | Dense passage retrieval for open-domain question answering.                                             | In EMNLP (1), pp. 6769–6781. Cited by: §2 |
| V. Khrulkov, L. Mirvakhabova, E. Ustinova, I. Oseledets, and V. Lempitsky (2020) | Hyperbolic image embeddings.                                                                             | In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6418–6428. Cited by: §2 |
| C. Lee, R. Roy, M. Xu, J. Raiman, M. Shoeybi, B. Catanzaro, and W. Ping (2024) | Nv-embed: improved techniques for training llms as generalist embedding models.                         | arXiv preprint arXiv:2405.17428. Cited by: §2 |
| P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschl, et al. (2020) | Retrieval-augmented generation for knowledge-intensive nlp tasks.                                       | Advances in neural information processing systems 33, pp. 9459–9474. Cited by: §1, §2 |
```

### --- Page 0023 ---

```markdown
| Authors                                                                 | Title                                                                                                   | Source                                                                                                   | Cited by |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------|
| Y. Li, H. Chen, X. Sun, Z. Sun, L. Li, L. Cui, P. S. Yu, and G. Xu (2021) | Hyperbolic hypergraphs for sequential recommendation.                                                  | In Proceedings of the 30th ACM international conference on information & knowledge management, pp. 988–997. | $2$      |
| Z. Li, X. Zhang, Y. Zhang, D. Long, P. Xie, and M. Zhang (2023)        | Towards general text embeddings with multi-stage contrastive learning.                                 | arXiv preprint arXiv:2308.03281.                                                                         | $2$      |
| Q. Liu, M. Nickel, and D. Kiela (2019)                                 | Hyperbolic graph neural networks.                                                                      | In Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.), Vol. 32, pp. . | $2$      |
| N. Muennighoff, S. Hongjin, L. Wang, N. Yang, F. Wei, T. Yu, A. Singh, and D. Kiela (2024) | Generative representational instruction tuning.                                                        | In The Thirteenth International Conference on Learning Representations,                                   | $2$      |
| N. Muennighoff, N. Tazi, L. Magne, and N. Reimers (2023)               | MTEB: massive text embedding benchmark.                                                                 | In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, A. Vlachos and I. Augenstein (Eds.), Dubrovnik, Croatia, pp. 2014–2037. | $5.1$    |
| B. Ni, Z. Liu, L. Wang, Y. Lei, Y. Zhao, X. Cheng, Q. Zeng, L. Dong, Y. Xia, K. Kenthapadi, et al. (2025) | Towards trustworthy retrieval augmented generation for large language models: a survey.                | arXiv preprint arXiv:2502.06872.                                                                         | $1$      |
| C. Ni, Y. Lin, F. Luo, and J. Gao (2019)                                | Community detection on networks with ricci flow.                                                      | Scientific reports 9 (1), pp. 1–12.                                                                      |          |
```


### --- Page 0024 ---

```markdown
| Authors                                                                 | Year | Title                                                                                                         | Source                                                                                          | Cited by |
|-------------------------------------------------------------------------|------|---------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|----------|
| Z. Nussbaum, J. X. Morris, A. Mulyar, and B. Duderstadt                | 2025 | Nomic embed: training a reproducible long context text embedder.                                             | In Transactions on Machine Learning Research, External Links: [Link](#)                        | §1      |
| A. J. Oche, A. G. Folashade, T. Ghosal, and A. Biswas                  | 2025 | A systematic review of key retrieval-augmented generation (rag) systems: progress, gaps, and future directions. | arXiv preprint arXiv:2507.18910.                                                               | §2      |
| A. Pal, M. van Spengler, G. M. D. di Melendugno, A. Flaborea, F. Galasso, and P. Mettes | 2024 | Compositional entailment learning for hyperbolic vision-language models.                                     | External Links: [2410.06912](#), [Link](#)                                                   | §2      |
| W. Peng, T. Varanka, A. Mostafa, H. Shi, and G. Zhao                   | 2021 | Hyperbolic deep neural networks: a survey.                                                                    | IEEE Transactions on pattern analysis and machine intelligence 44 (12), pp. 10023–10044.       | §2      |
| M. Radovanovic, A. Nanopoulos, and M. Ivanovic                         | 2010 | Hubs in space: popular nearest neighbors in high-dimensional data.                                           | Journal of Machine Learning Research 11 (sept), pp. 2487–2531.                               | §1      |
| N. Reimers and I. Gurevych                                             | 2019 | Sentence-BERT: sentence embeddings using Siamese BERT-networks.                                              | In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan (Eds.), Hong Kong, China, pp. 3982–3992. External Links: [Link](#), [Document](#) | §2      |
```

### --- Page 0025 ---

```markdown
| Authors                                      | Title                                                                                                           | Source                                                                                                         | Cited by  |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------|
| N. Reimers and I. Gurevych (2021)           | The curse of dense low-dimensional information retrieval for large index sizes.                                 | In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pp. 605–611. | §1        |
| M. Robinson, S. Dey, and S. Sweet (2024)    | The structure of the token space for large language models.                                                    | arXiv preprint arXiv:2410.08993.                                                                              | §1        |
| N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych (2021) | BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models.                       | In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), External Links: [Link](#) | §5.1      |
| H. S. Vera, S. Dua, B. Zhang, D. Salz, R. Mullins, S. R. Panyam, S. Smoot, I. Naim, J. Zou, F. Chen, et al. (2025) | Embeddinggemma: powerful and lightweight text representations.                                                  | arXiv preprint arXiv:2509.20354.                                                                              | §5.1      |
| L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei (2022) | Text embeddings by weakly-supervised contrastive pre-training.                                                  | arXiv preprint arXiv:2212.03533.                                                                              | §2        |
| B. Warner, A. Chaffin, B. Clavié, O. Weller, O. Hallström, S. Taghadouini, A. Gallagher, R. Biswas, F. Ladhak, T. Aarsen, N. Cooper, G. Adams, J. Howard, and I. Poli (2024) | Smarter, better, faster, longer: a modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. | External Links: [Link](#)                                                                                     | §5.1      |
| S. Weerawardhena, P. Kassianik, B. Nelson, B. Saglam, A. Vellore, A. Priyanshu, S. Vijay, M. Aufiero, A. Goldblatt, F. Burch, et al. (2025) | Llama-3-1-foundational-securityllm-8b-instruct technical report.                                               | arXiv preprint arXiv:2508.01059.                                                                              | §5.1      |
```

### --- Page 0026 ---

```markdown
| Authors                                           | Title                                                                                          | Publication Details                                                                 | External Links | Cited by  |
|---------------------------------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|----------------|-----------|
| R. Wei, Y. Liu, J. Song, Y. Xie, and K. Zhou (2024) | Exploring hierarchical information in hyperbolic space for self-supervised image hashing.     | IEEE Transactions on Image Processing 33 (), pp. 1768–1781.                        | [Document](#)  | $§2$     |
| L. Xiong, C. Xiong, Y. Li, K. Tang, J. Liu, P. N. Bennett, J. Ahmed, and A. Overwijk (2021) | Approximate nearest neighbor negative contrastive learning for dense text retrieval.          | In International Conference on Learning Representations.                           | [Link](#)     | $§2$     |
| S. Yan, J. Gu, Y. Zhu, and Z. Ling (2024)        | Corrective retrieval augmented generation.                                                    | arXiv preprint arXiv:2401.15884.                                                  |                | $§2$     |
| H. Yang, H. Chen, L. Li, S. Y. Philip, and G. Xu (2021) | Hyper meta-path contrastive learning for multi-behavior recommendation.                       | In 2021 IEEE International Conference on Data Mining (ICDM), pp. 787–796.         |                | $§2$     |
| M. Yang, R. S. B. A. Feng, B. Xiong, J. Liu, I. King, and R. Ying (2025) | Hyperbolic fine-tuning for large language models.                                            | In The Thirty-ninth Annual Conference on Neural Information Processing Systems.     | [Link](#)     | $§2$     |
| M. Yang, H. Verma, D. C. Zhang, J. Liu, I. King, and R. Ying (2024) | Hypformer: exploring efficient transformer fully in hyperbolic space.                        | In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD '24, New York, NY, USA, pp. 3770–3781. | [ISBN](9798400749041), [Link](#), [Document](#) | $§1$, $§2$, $§3$, $§4.1$, $§4.2$ |
```


### --- Page 0027 ---

```markdown
M. Yang, M. Zhou, R. Ying, Y. Chen, and I. King (2023)  
**Hyperbolic representation learning: revisiting and advancing.**  
In International Conference on Machine Learning, pp. 39639–39659.  
Cited by: §2.

X. Zhang, Y. Zhang, D. Long, W. Xie, Z. Dai, J. Tang, H. Lin, B. Yang, P. Xie, F. Huang, et al. (2024)  
**MGTE: generalized long-context text representation and reranking models for multilingual text retrieval.**  
In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 1393–1412.  
Cited by: §5.1.

Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, et al. (2025)  
**Qwen3 embedding: advancing text embedding and reranking through foundation models.**  
arXiv preprint arXiv:2506.05176.  
Cited by: §2.

Y. Zhu, D. Zhou, J. Xiao, X. Jiang, X. Chen, and Q. Liu (2020)  
**HyperText: endowing FastText with hyperbolic geometry.**  
In Findings of the Association for Computational Linguistics: EMNLP 2020, T. Cohn, Y. He, and Y. Liu (Eds.), Online, pp. 1166–1171.  
External Links: [Link](#), Document  
Cited by: §4.2.

## Appendix A  Proofs

Throughout, we work in the Lorentz model with curvature $K < 0$, where  
$$\mathcal{H}^d_K = \{x \in \mathbb{R}^{d+1} : \langle x, x \rangle_L = 1 / K, x_0 > 0\}$$  
and $\langle x, y \rangle_L = -x_0 y_0 + \sum_{i=1}^d x_i y_i$ denotes the Lorentzian inner product.

### A.1 Auxiliary Lemma

**Lemma A.1 (Lorentzian Inner Product Bound).**
```

### --- Page 0028 ---

```markdown
For any $x, y \in \mathbb{H}^d_K$, we have $K(x,y)_L \geq 1$, with equality if and only if $x = y$.

**Proof.**  
The geodesic distance on $\mathbb{H}^d_K$ satisfies

$$
d_K(x,y) = \frac{1}{\sqrt{-K}} \cosh^{-1}(K(x,y)_L).
$$

Since $\cosh^{-1} : [1, \infty) \to [0, \infty)$ and $d_K(x,y) \geq 0$ with equality if and only if $x = y$, we conclude $K(x,y)_L \geq 1$ with equality if and only if $x = y$. $\blacksquare$

### A.2 Proof of Proposition 4.3

**Proposition A.2 (Euclidean Mean Contracts).**  
Let $\{x_i\}^n_{i=1} \subset \mathbb{H}^d_K$ with $n \geq 2$. Define the Euclidean mean $x = \frac{1}{n} \sum_{i=1}^n x_i$ and its projection onto the hyperboloid $m_{Euc} = \Pi_K(x)$. Then:

$$
r(m_{Euc}) \leq \frac{1}{n} \sum_{i=1}^n r(x_i)
$$

with equality if and only if all $x_i$ are identical.

**Proof.**  
We first verify that the projection is well-defined, then establish the contraction inequality.

We must show $\langle x, x_i \rangle_L < 0$ and $x_0 > 0$. The latter is immediate since $x_0 = \frac{1}{n} \sum_i x_{i,0} > 0$. For the former, compute:

$$
K\langle x, x_i \rangle_L = \left\langle \sqrt{\frac{1}{n} \sum_i x_i, \frac{1}{n} \sum_j x_j} \right\rangle_L = \frac{1}{n^2} \sum_{i,j} K(x_i, x_j)_L.
$$

By Lemma A.1, each term satisfies $K(x_i, x_j)_L \geq 1$. Therefore:

$$
K\langle x, x_i \rangle_L \geq \frac{1}{n^2} \cdot n^2 = 1 > 0.
$$

Since $K < 0$, this implies $\langle x, x_i \rangle_L < 0$, confirming projectability.

The projection is given by $m_{Euc} = x / \sqrt{K(x, x)_L}$, so the radial depth satisfies:
```

### --- Page 0029 ---

```markdown
$$
r^{(mEuc)} = \frac{x_0}{\sqrt{K(x, x)_L}}.
$$

From Step 1, we have $K(x, x)_L \geq 1$, hence $K(x, x)_L \geq 1$. Therefore:

$$
r^{(mEuc)} = \frac{x_0}{\sqrt{K(x, x)_L}} \leq x_0 = \frac{1}{n} \sum_{i=1}^{n} x_{i, 0} = \frac{1}{n} \sum_{i=1}^{n} r(x_i).
$$

Equality holds if and only if $K(x, x)_L = 1$, which by Step 1 requires $K(x_i, x_j)_L = 1$ for all pairs $i, j$. By Lemma A.1, this occurs if and only if $x_i = x_j$ for all $i, j$, i.e., all points are identical.

### A.3 Proof of Proposition 4.5

**Proposition A.3 (Implicit Radial Weighting is Insufficient).**  
The Einstein midpoint weights points by the Lorentz factor $k_i = x_{i, 0}$, but this weighting grows as $\exp(-K \rho)$ while hyperbolic volume grows as $\exp((d - 1) -K \rho)$. The Lorentz factor weighting therefore undercompensates by a factor of $\exp((d - 2) -K \rho)$ for $d \geq 3$.

**Proof.**  
We establish the asymptotic growth rates of the Lorentz factor and hyperbolic volume separately, then compare them.

**Step 1: Lorentz factor asymptotics.**  
The hyperbolic distance from the origin $o = (1 / \sqrt{-K}, 0, \ldots, 0)$ to a point $x \in \mathbb{H}^d_K$ is:

$$
\rho = d_K(o, x) = \frac{1}{\sqrt{-K}} \cosh^{-1}(K(o, x)_L).
$$

Computing the inner product:

$$
\langle o, x \rangle_L = -\frac{x_0}{\sqrt{-K}},
$$

so $K(o, x)_L = -K \left(-x_0 / \sqrt{-K}\right) = -\sqrt{-K} x_0$. Thus:

$$
\rho = \frac{1}{\sqrt{-K}} \cosh^{-1}(-\sqrt{-K} x_0).
$$

Inverting this relation:
```

### --- Page 0030 ---

```markdown
For large $\rho$, using $\cosh(t) \sim \frac{1}{2} e^{t}$:

$$
x_0 = \frac{1}{\sqrt{-K}} \cosh(\sqrt{-K} \rho).
$$

Hence the Lorentz factor $\lambda = x_0$ grows as $\exp(-\sqrt{-K} \rho)$.

### Step 2: Hyperbolic volume asymptotics.

The volume of a geodesic ball of radius $\rho$ in $\mathbb{H}^d$ is:

$$
\text{Vol}(B_\rho) = \frac{\omega_{d-1}}{(-K)(d-1)/2} \int_0^{\rho} \sinh^{d-1}(\sqrt{-K} t) \, dt,
$$

where $\omega_{d-1} = 2\pi^{d/2} / \Gamma(d/2)$ is the surface area of the unit $(d-1)$-sphere. For large $\rho$, using $\sinh(t) \sim \frac{1}{2} e^{t}$:

$$
\text{Vol}(B_\rho) \sim C_d \exp((d-1)\sqrt{-K} \rho),
$$

where $C_d$ is a dimension-dependent constant.

### Step 3: Compensation deficit.

The ratio of volume growth to Lorentz factor growth is:

$$
\frac{\text{Vol}(B_\rho)}{\lambda} \sim \frac{\exp((d-1)\sqrt{-K} \rho)}{\exp(\sqrt{-K} \rho)} = \exp((d-2)\sqrt{-K} \rho).
$$

For $d \geq 3$, this ratio diverges as $\rho \to \infty$, demonstrating that the Lorentz factor provides insufficient compensation for the exponential growth of hyperbolic space at large radii. 

---

### A.4 Proof of Theorem 4.6

**Theorem A.4 (OEM Pre-Projection Bound).**  
Let $v = \sum_{i=1}^{n} w_i x_i^{p+1,0}$ where $w_i \propto w_i x_i^{p,0}$ are the normalized OEM weights. Then, for $p \geq 0$:

$$
v_0 = \frac{\sum_{i=1}^{n} w_i x_i^{p+2,0}}{\sum_{i=1}^{n} w_i x_i^{p+1,0}} \geq \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i} = r_w.
$$

**Proof.**  
We apply Chebyshev’s sum inequality. Define sequences:
```

### --- Page 0031 ---

```markdown
$$
a_i = x_{i,0}^{p+1}, \quad b_i = x_{i,0}.
$$

Since $x_{i,0} > 0$ for all $i$ (points lie on the upper sheet of the hyperboloid) and $p \geq 0$, both sequences are strictly positive. Moreover, the sequences are co-monotonic: for any $i, j$,

$$
x_{i,0} \geq x_{j,0} \iff x_{i,0}^{p+1} \geq x_{j,0}^{p+1},
$$

since $t \mapsto t^{p+1}$ is strictly increasing on $(0, \infty)$.

Chebyshev’s sum inequality states that for co-monotonic sequences $\{a_i\}, \{b_i\}$ and non-negative weights $\{w_i\}$ with $\sum_i w_i > 0$:

$$
\left( \sum_i w_i a_i \right) \left( \sum_i w_i b_i \right) \geq \left( \sum_i w_i a_i \right) \left( \sum_i w_i b_i \right).
$$

Substituting $a_i = x_{i,0}^{p+1}$ and $b_i = x_{i,0}$:

$$
\left( \sum_i w_i x_{i,0}^{p+2} \right) \left( \sum_i w_i \right) \geq \left( \sum_i w_i x_{i,0}^{p+1} \right) \left( \sum_i w_i x_{i,0} \right).
$$

Dividing both sides by $\sum_i w_i x_{i,0}^{p+1} \sum_i w_i > 0$:

$$
\frac{\sum_i w_i x_{i,0}^{p+2}}{\sum_i w_i x_{i,0}^{p+1}} \geq \frac{\sum_i w_i x_{i,0}}{\sum_i w_i}.
$$

The left-hand side equals $\nu_0$ and the right-hand side equals $\rho_w$, completing the proof. Equality holds if and only if all $x_{i,0}$ are identical.

---

### A.5 Proof of Theorem 4.7

**Theorem A.5 (OEM Outward Bias).**  
Let $m^{\text{Ein}}_k$ denote the standard Einstein midpoint ($p = 0$) and $m^{\text{OEM}}_{k,p}$ the Outward Einstein Midpoint. Then for all $p \geq 1$:

$$
r(m^{\text{OEM}}_{k,p}) \geq r(m^{\text{Ein}}_k).
$$

**Proof.**  
For a general exponent $q \geq 0$, define the weighted average with weights proportional to $w_i x_{i,0}^{q+1}$.
```

### --- Page 0032 ---

```markdown
The projected point is $m^{(q)} = \Pi_K(v^{(q)})$ with radial depth:

$$
v^{(q)} = \frac{\sum_i w_i x_i^{0+1} x_i}{\sum_i w_i x_i^{0+1}}.
$$

The radial depth is 

$$
r(m^{(q)}) = \frac{u^{(q)}_0}{\sqrt{K(v^{(q}), v^{(q)})_L}}.
$$

The Einstein midpoint corresponds to $q = 0$ and the OEM to $q = \rho$.

We show that $q \mapsto r(m^{(q)})$ is non-decreasing for $q \geq 0$. Define the normalized weights 

$$
\alpha^{(q)}_i = \frac{w_i x_i^{0+1}}{\sum_j w_j x_j^{0+1}}. 
$$ 

As $q$ increases, these weights concentrate toward indices with larger $x_{i,0}$: if $x_{i,0} > x_{j,0}$, then 

$$
\frac{\alpha^{(q)}_i}{\alpha^{(q)}_j} = \frac{w_i}{w_j} \left( \frac{x_{i,0}}{x_{j,0}} \right)^{q + 1}
$$ 

is strictly increasing in $q$.

The radial depth after projection satisfies:

$$
r(m^{(q)})^2 = \frac{(u^{(q)}_0)^2}{K(v^{(q}), v^{(q)})_L}.
$$

We analyze numerator and denominator separately.

**Numerator:** We have $u^{(q)}_0 = \sum_i \alpha^{(q)}_i x_{i,0}$. By Chebyshev’s inequality (Theorem 4.6), this is non-decreasing in $q$.

**Denominator:** We have 

$$
K(v^{(q)}, v^{(q)})_L = \sum_{i,j} \alpha^{(q)}_i \alpha^{(q)}_j K(x_i, x_j)_L.
$$ 

By Lemma A.1, $K(x_i, x_j)_L \geq 1$ with equality iff $x_i = x_j$. As weights concentrate on fewer points (larger $q$), the sum decreases toward 1.

Thus as $q$ increases: the numerator $(u^{(q)}_0)^2$ increases, while the denominator $K(v^{(q)}, v^{(q)})_L$ decreases. Both effects increase $r(m^{(q)})^2$, establishing monotonicity.
```

### --- Page 0033 ---

```markdown
For $p \geq 1 > 0$, we conclude $r(m^{\text{OEM}}_{k,p}) = r(m(p)) \geq r(m(0)) = r(m^{\text{Ein}}_{k})$.

## Appendix B  RAG Prompt

For each retrieval-augmented generation (RAG) query, we construct a single inference prompt by concatenating the top-$|c|$ retrieved documents into the context window of the language model. The documents are provided verbatim, without re-ranking or compression, and are ordered according to their retrieval score. The language model is then instructed to generate an answer conditioned solely on the retrieved context and the user query, ensuring that any factual content in the response must be supported by the retrieved evidence.
```

### --- Page 0034 ---

```markdown
# RAG Prompt

Based on the following context, answer the question.

**Context:**  
\{Doc[0], Doc[1], …, Doc[C]\}

**Question:** \{query\}

**Answer:**

---

## Appendix C  Hierarchical Document Probe
```

### --- Page 0035 ---

```markdown
Dense retrieval models implicitly induce a geometry over documents. While Euclidean encoders often cluster documents based on surface-level similarity, they struggle to faithfully represent hierarchical relationships that arise naturally in language and knowledge organization. Hyperbolic spaces, by contrast, are well suited for embedding tree-like and taxonomic structures due to their exponential volume growth.

To qualitatively assess whether hyperbolic encoders recover such hierarchical organization, we construct a controlled document set that exhibits a clear semantic hierarchy while ensuring that individual documents remain self-contained and independent.

We design a synthetic yet semantically natural hierarchy consisting of five levels of increasing specificity: Science → Mathematics → Algebra → Linear Algebra → Linear Transformations. At each level, we generate five independent paragraphs that are topically coherent but do not explicitly reference parent or child topics. Importantly, documents at deeper levels refine the semantic scope of higher-level topics without sharing explicit lexical markers or cross-document dependencies. This construction isolates hierarchical structure as a latent semantic property rather than an artifact of explicit cues.

All paragraphs are embedded independently with no hierarchical supervision. We analyze the resulting embeddings by inspecting their relative organization in the learned space. Hyperbolic models are expected to organize documents according to semantic specificity, with broader concepts closer to the origin and more specific concepts at increasing radial depth. We present a few of them here and we have attached the data file in supplementary material.
```

### --- Page 0036 ---

# Hierarchical Document Texts (Verbatim)

![Hierarchical Document Texts](assets/page_0036_img_1.png)

### --- Page 0037 ---

```markdown
![Empty page with no content](assets/page_0037_img_1.png)
```

### --- Page 0038 ---

```markdown
![Empty page with no content](assets/page_0038_img_1.png)
```

### --- Page 0039 ---

```markdown
![Empty page with no content](assets/page_0039_img_1.png)
```

### --- Page 0040 ---

```markdown
![Empty page with no content](assets/page_0040_img_1.png)
```

### --- Page 0041 ---

```markdown
![Empty page with no content](assets/page_0041_img_1.png)
```

### --- Page 0042 ---

```markdown
# Appendix D  Runtime and Computational Complexity

In this section, we analyze the computational complexity of the proposed hyperbolic dense retrieval system and compare it to a standard Euclidean transformer-based retriever. Let $n$ denote the input sequence length, $d$ the hidden dimension, $h$ the number of attention heads, and $L$ the number of transformer layers.

## Hyperbolic Transformer Encoder

The HyTE-FH encoder follows the standard transformer structure, with all linear, normalization, attention, and residual operations replaced by their Lorentzian counterparts. Crucially, these operations preserve the same asymptotic complexity as their Euclidean analogues.

Each Lorentz linear transformation (HLT) consists of a matrix multiplication followed by a constant number of scalar operations and a reprojection. This incurs $O(nd^2)$ time per layer, identical to a Euclidean linear layer up to constant factors.

Hyperbolic self-attention computes pairwise geodesic distances between queries and keys. In the Lorentz model, each geodesic distance $d_k(q_i, k_j)$ is computed using a Lorentzian inner product, which costs $O(d)$. Thus, attention score computation scales as $O(n^2d)$ per layer, matching standard dot-product attention.

The Lorentzian weighted midpoint used for value aggregation requires a weighted sum and normalization in $\mathbb{R}^{d + 1}$, contributing $O(nd)$ time per token, and is dominated by the attention score computation.

## [MATHEMATICS]

Mathematics also plays a central role in modeling complex systems. By formalizing assumptions and relationships, mathematical models help clarify underlying mechanisms and enable precise predictions under well-defined conditions.

## [ALGEBRA]

Modern algebra emphasizes structural relationships over explicit computation. Rather than focusing on individual equations, it studies entire systems of elements and operations, revealing patterns that persist across different mathematical settings.

## [LINEAR ALGEBRA]

The power of linear algebra lies in its balance between abstraction and computation. While grounded in rigorous theory, it offers efficient numerical techniques that scale to high-dimensional problems.

## [LINEAR TRANSFORMATIONS]

Linear transformations form the foundation of many applied systems, from computer graphics to neural networks. Understanding their behavior is essential for analyzing stability, expressiveness, and computational efficiency.
```


### --- Page 0043 ---

```markdown
Overall, the time complexity of a single HyTE-FH layer is $O(n^2d + nd^2)$, and the total encoder complexity is $O(L(n^2d + nd^2))$, which matches the asymptotic complexity of a standard Euclidean transformer.

HyTE-H introduces an additional projection from Euclidean to hyperbolic space at the input, costing $O(nd)$, which is negligible compared to the encoder cost.

### Pooling via Outward Einstein Midpoint
Given a sequence of $n$ token embeddings, the Outward Einstein Midpoint computes radius-dependent weights, a weighted sum in $\mathbb{R}^{d + 1}$, and a single reprojection. This requires $O(nd)$ time and $O(d)$ memory, identical in order to standard mean pooling or the Einstein midpoint.

### Training Objectives
All training objectives operate on fixed-dimensional query and document embeddings. Geodesic similarity computation costs $O(d)$ per query–document pair.

Unsupervised contrastive pre-training with in-batch negatives of size $N$ requires $O(N^2d)$ per batch, matching the complexity of standard contrastive learning. Supervised contrastive fine-tuning has the same asymptotic cost.

Masked language modeling introduces no additional asymptotic overhead beyond the encoder forward pass.

### Retrieval and RAG Inference
At inference time, dense retrieval over a corpus of size $|\mathcal{D}|$ requires computing hyperbolic distances between a query embedding and document embeddings, with total cost $O(|\mathcal{D}|d)$. This matches Euclidean dense retrieval up to constant factors.

Approximate nearest neighbor indexing can be applied without modification, as retrieval relies only on pairwise distance computations. In summary, the proposed hyperbolic dense retrieval system has the same asymptotic computational complexity as a Euclidean transformer-based retriever:

$$O(L(n^2d + nd^2))$$

for encoding, and $O(|\mathcal{D}|d)$ for retrieval. The additional cost of hyperbolic geometry manifests only as constant-factor overhead from Lorentzian inner products and reprojection, while enabling geometry-aware modeling of hierarchical structure.

## Appendix E Additional results

### Full MTEB Results
Table A1 presents the performance of our proposed models on the MTEB benchmark across seven task types. All models share the same architecture with 149M parameters, 768 dimensions, and 12
```
![Detailed description of the chart](assets/page_0043_img_1.png)
```

### --- Page 0044 ---

```markdown
# PAGE_NAME: page_0044

layers. HyTE-H achieves the best overall performance with a Mean (Task) score of 59.89, outperforming both HyTE-FH and the Euclidean baselines. Notably, HyTE-H ranks first in six out of seven task categories, demonstrating the effectiveness of the hybrid hyperbolic-Euclidean approach. HyTE-FH shows competitive performance in clustering tasks, securing second place, which suggests that hyperbolic geometry is particularly beneficial for capturing hierarchical relationships. The Euclidean equivalent (ModernBert-embed*) achieves second place in most categories but falls short of HyTE-H, indicating that incorporating hyperbolic components provides meaningful improvements over purely Euclidean representations.

## Table A1: MTB Benchmark Results. * Euclidean equivalent of HyTE

---

## Similarity function. 

We evaluate two distance metrics for the contrastive loss in hyperbolic space: the Lorentz inner product and hyperbolic geodesic distance. While the Lorentz inner product provides a computationally convenient similarity measure, geodesic distance directly reflects intrinsic distances on the manifold. As shown in Table A2, using geodesic distance in the contrastive objective leads to improved performance across both mean task and mean task-type metrics, suggesting more effective alignment of representations in hyperbolic space.

## Table A2: Comparison of loss functions for hyperbolic embeddings. Both Lorentz inner product and geodesic distance are evaluated for their effectiveness in learning hierarchical representations.

| Loss Function            | Mean (Task) | Mean (TaskType) |
|-------------------------|-------------|------------------|
| Lorentz Inner Product    | 52.59       | 51.60            |
| Geodesic Distance        | 56.41       | 53.75            |

---

## Context size. 

Table A3 shows the effect of increasing the retrieval context size from $|C| = 5$ to $|C| = 10$. The Euclidean baseline (Gemma) exhibits a large performance gain across all metrics with the larger context window, but still falls short of the fully hyperbolic HyTE-H model. HyTE-FH also improves with increased context, while
```

### --- Page 0045 ---

```markdown
HyTE-H remains comparatively stable, achieving consistently strong performance at both context sizes. This suggests that fully hyperbolic retrieval is less sensitive to context expansion and maintains effectiveness even under smaller retrieval budgets.

### Table A3: Average performance on RAG Bench with varying context window size.

| Model     | $|C| = 5$ | F     | CR    | AR    | $|C| = 10$ | F     | CR    | AR    |
|-----------|----------|-------|-------|-------|-----------|-------|-------|-------|
| Gemma     |          | 0.603 | 0.735 | 0.684 |           | 0.756 | 0.846 | 0.836 |
| HyTE-FH   |          | 0.732 | 0.848 | 0.765 |           | 0.770 | 0.912 | 0.784 |
| HyTE-H    |          | 0.763 | 0.904 | 0.832 |           | 0.787 | 0.913 | 0.847 |

F = Faithfulness, CR = Context Relevance, AR = Answer Relevance. Best results in bold.

## E.1 Case Study: The Impact of Retrieval Geometry on Answer Quality

To illustrate how embedding geometry affects end-to-end RAG performance, we analyze a representative query from the Emanuel dataset where models exhibit markedly different behaviors.

### Query.
“What is the feature of Bixby guide?”

This query requires retrieving documentation about a specific Samsung TV feature: the Bixby tutorial that appears when users first interact with the voice assistant. The correct answer is contained in a single passage within the hierarchically organized e-Manual.

### Quantitative Comparison.
Table A4 presents the evaluation metrics for each model. Only HyTE-H successfully retrieves relevant context and generates a faithful response.
```

### --- Page 0046 ---

```markdown
| Table A4: Case study metrics for the query “What is the feature of Bixby guide?” CR = Context Relevance, F = Faithfulness, AR = Answer Relevancy. |
| --- | --- | --- | --- | --- |
| Model     | CR  | F   | AR  | Failure Mode          |
| HyTE-H    | 1.0 | 1.0 | 1.0 | None (Success)       |
| GTE       | 0.5 | 1.0 | 0.0 | Honest Refusal       |
| Gemma     | 1.0 | 0.25| 0.94| Ungrounded Extrapolation |

## Qualitative Analysis.

We identify four distinct outcomes based on retrieval quality and LLM response behavior:

(1) **Successful Retrieval (HyTE-H).** The hyperbolic hybrid model retrieves the exact passage describing the Bixby guide: “When you press the button for the first time, the Using Bixby button appears… a tutorial on using Bixby is shown.” This enables a concise, accurate response:

> “The feature of Bixby guide is a tutorial on using Bixby, which is shown when you press the button on your Samsung Smart Remote after the first time, and then press the Select button.”

(2) **Retrieval Collapse with Honest Refusal (GTE).** GTE retrieves its default “hub” documents—generic content about picture quality, SmartThings, and antenna connections—regardless of the query. The LLM correctly identifies the context mismatch:

> “Unfortunately, the provided context does not mention the Bixby guide.”

While this response is faithful (F=1.0), it provides no utility (AR=0.0). This pattern, where Euclidean embeddings collapse to retrieving the same generic documents, occurred for 100% of queries in GTE’s results.

(3) **Partial Context with Topic Drift (Gemma).** Gemma retrieves tangentially related content about the Samsung Smart Remote that mentions Bixby but not the Bixby guide. The LLM provides information about running Bixby generally rather than the guide feature:
```

### --- Page 0047 ---

```markdown
“Unfortunately, the provided context does not mention the Bixby guide. However, based on general knowledge, the Bixby guide is a feature in Samsung TVs that provides users with a personalized interface to access various features and settings. It offers a simplified and intuitive way to navigate the TV’s menu…”

This response achieves high answer relevancy (AR=0.94) because it sounds plausible, but low faithfulness (F=0.25) because the claims are fabricated. This “confident hallucination” failure mode is particularly dangerous: users receive authoritative-sounding misinformation.

## JSON Representation

The complete retrieval outputs are: **HyTE-H (Successful Retrieval)**:

```json
{
  "query": "What is the feature of Bixby guide?",
  "model": "HyTE-H",
  "metrics": {
    "context_relevance": 1.0,
    "faithfulness": 1.0,
    "answer_relevancy": 1.0
  },
  "top_retrieved_context": "Running Bixby. Press and hold the button on your Samsung Smart Remote... To view the Bixby guide, press: When you press the button for the first time, the Using Bixby button appears at the bottom of the screen. Press the Select button. T up window appears and a tutorial on using Bixby is shown.",
  "response": "The feature of Bixby guide is a tutorial on using Bixby, which is shown when you press the button on your Samsung Smart first time, and then press the Select button.",
  "outcome": "success"
}
```

## GTE (Retrieval Collapse):
```

### --- Page 0048 ---

```markdown
| Query                                   | What is the feature of Bixby guide? |
|-----------------------------------------|-------------------------------------|
| Model                                   | GTE                                 |
| Metrics                                 |                                     |
| - Context Relevance                     | 0.5                                 |
| - Faithfulness                          | 1.0                                 |
| - Answer Relevancy                     | 0.0                                 |
| Top Retrieved Context                   | Adjusting the Picture Quality. Change the Picture Mode and adjust Expert Settings. |
| Response                                | Unfortunately, the provided context does not mention the Bixby guide. |
| Outcome                                 | retrieval_collapse_honest_refusal   |

---

| Query                                   | What is the feature of Bixby guide? |
|-----------------------------------------|-------------------------------------|
| Model                                   | Gemma                               |
| Metrics                                 |                                     |
| - Context Relevance                     | 0.5                                 |
| - Faithfulness                          | 0.25                                |
| - Answer Relevancy                     | 0.94                                |
| Top Retrieved Context                   | Removing channels from a favorites list. Select channels in a favorites list, and then select the icon. |
| Response                                | Unfortunately, the provided context does not mention the Bixby guide. However, based on general knowledge, the Bixby is a feature in Samsung TVs that provides users with a personalized interface to access various features and settings. It offers a simplified way to navigate the TV’s menu and access popular features. |
| Outcome                                 | ungrounded_extrapolation            |
| Warning                                 | High answer_relevancy (0.94) masks unfaithful content (0.25) |
```


# ArXiv 2605.06285

### --- Page 0001 ---

```markdown
# LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG

**Yijia Zheng**  
University of Amsterdam, Amsterdam, the Netherlands  
{y.zheng, m.worring}@uva.nl  

**Marcel Worring**  

## Abstract

Single-step retrieval-augmented generation (RAG) provides an efficient way to incorporate external information for simple question answering tasks but struggles with complex questions. Agentic RAG extends this paradigm by replacing single-step retrieval with a multi-step process, in which the large language model (LLM) acts as a search agent that generates intermediate thoughts and subqueries to iteratively interact with the retrieval system. This iterative process incurs substantial latency due to the autoregressive generation of lengthy thoughts and subqueries. To address this limitation, we propose LatentRAG, a novel framework that shifts both reasoning and retrieval from discrete language space to continuous latent space. Unlike existing explicit methods that generate natural language thoughts or subqueries token-by-token, LatentRAG produces latent tokens for thoughts and subqueries directly from the hidden states in a single forward pass. We align LLMs with dense retrieval models in the latent space, enabling retrieval over latent subquery tokens and supporting end-to-end joint optimization. To improve transparency and encourage semantically meaningful latent representations, we incorporate a parallel latent decoding mechanism that translates latent tokens back into natural language. Extensive experiments on seven benchmark datasets show that LatentRAG achieves performance comparable to explicit agentic RAG methods while reducing inference latency by approximately 90%, substantially narrowing the latency gap with traditional single-step RAG.

## 1 Introduction

Large language models (LLMs) have demonstrated strong capabilities in answering complex questions [1–3], but these capabilities are fundamentally bounded by their static internal knowledge [4, 5]. Solely relying on internal knowledge limits their performance on questions that require up-to-date information or proprietary knowledge [6, 7] and increases the risk of hallucinations [8, 9]. To improve both the factuality and transparency of LLM-generated outputs, retrieval-augmented generation (RAG) [10, 11] retrieves question-relevant information from an external retrieval system to augment the LLM inputs [12, 13]. Traditional RAG methods provide an efficient way to access external knowledge, but their single-step retrieval design limits their effectiveness on complex questions that require iterative reasoning and retrieval [14, 15].

Motivated by the success of tool-using LLM agents [16, 17], recent agentic RAG approaches [18, 19] replace traditional single-step retrieval with a multi-step agentic search process that alternates between generation and retrieval. In this process, the agent generates a thought via chain-of-thought (CoT) reasoning [20] and then produces the next action, which can be either a subquery for the next retrieval step or the final answer. Each generated subquery is used to retrieve relevant documents. Unlike static single-step retrieval in traditional RAG, this multi-step agentic search process enables complex questions to be answered more effectively.

Preprint.
```


### --- Page 0002 ---

```markdown
![Comparison of performance and latency on multi-hop QA datasets. LatentRAG achieves comparable performance to competitive agentic RAG methods such as Search-R1 and AutoRefine, while maintaining efficiency on par with naive single-step RAG. Search-R1 incurs substantial latency in thought and subquery generation, whereas LatentRAG substantially reduces the time spent in these two stages, leading to the observed efficiency gains. Detailed stage-wise latency breakdowns are provided in Appendix E.5.](assets/page_0002_img_1.png)

decomposed and effectively solved step by step \[21, 22\]. Although agentic RAG methods demonstrate strong performance on tasks with complex questions \[15, 23\], they incur substantial latency due to the additional multi-step interactions \[24, 25\].

To identify the latency bottlenecks of agentic RAG, we measure the average inference time across different stages for both naive single-step RAG and agentic RAG methods. As shown in Fig. 1, on multi-hop question answering (QA) datasets, the inference time of a representative RAG method, Search-R1 \[19\], requires 16–22× the inference time of naive RAG. This overhead is primarily driven by the thought and subquery generation stages, which together account for approximately 90% of the total latency. Both stages involve a sequential token-by-token generation of long outputs, where each output token depends on previously generated tokens, leading to multiple sequential LLM forward passes with limited parallelism. In contrast, prefill, retrieval, and final answer generation take far less time than the other two stages. The inference time comparison indicates that the latency bottlenecks of agentic RAG lie in the thought and subquery generation stages.

To reduce the thought and subquery generation latency in agentic RAG, we draw inspiration from another technique called latent reasoning. Latent reasoning \[26, 27\] is an efficient reasoning paradigm that performs reasoning within the continuous hidden states of the LLM, also referred to as latent tokens, without explicitly generating discrete language tokens. Compared to explicit reasoning, latent reasoning avoids allocating computation to non-semantic tokens that are produced solely for linguistic fluency \[28, 26\]. Furthermore, continuous latent tokens allow the LLM to directly leverage high-level semantic representation, avoiding the inefficiency of explicit token-by-token generation and thereby enabling more parallelizable computation \[29–31\]. Although latent reasoning offers a promising avenue for enhancing reasoning efficiency, its application to agentic RAG remains unexplored.

In this work, we pioneer the integration of latent reasoning into the agentic RAG paradigm and, more importantly, propose a latent retrieval mechanism. Unlike generation-only tasks studied in prior work on latent reasoning \[26, 32\], agentic RAG requires the LLM to emit explicit subquery tokens to invoke external retrieval. This explicit token generation not only incurs significant decoding overhead but also prevents gradient propagation, thereby hindering direct optimization of the LLM using retrieval signals. To overcome these limitations, we investigate whether latent tokens generated by an LLM can effectively serve as subqueries for retrieval. This introduces two challenges. (1) Data scarcity: Training retrieval models typically requires large-scale paired data, often comprising hundreds of millions of query–document pairs \[33, 34\]. In contrast, agentic RAG systems are commonly developed under a training setup that provides only tens of thousands of question–answering pairs, without explicit supervision on the ground-truth documents for intermediate subqueries \[19, 35\]. This data scarcity makes it difficult to learn effective retrieval capability using conventional training paradigms for retrieval models. (2) Transparency: Latent tokens inherently obscure the intermediate thoughts and subqueries, which is particularly problematic for agentic RAG, as lengthy and redundant
```

### --- Page 0003 ---

```markdown
# Page 3

retrieved documents make answer verification and evidence attribution [36, 37] time-consuming without explicit intermediate steps.

To address the aforementioned challenges, we introduce LatentRAG, an efficient agentic RAG framework that conducts reasoning and retrieval in the latent space. Specifically, we feed a sequence of special thought and subquery tokens into the LLM and use the corresponding fast hidden states as latent thought and subquery tokens, respectively. These latent tokens are obtained in a single forward pass, enabling parallel computation and avoiding the inefficiency of autoregressive generation. To address challenge (1), we align the LLM with a pretrained dense retrieval model in the latent space. The latent subquery tokens are used as inputs to the retrieval model to generate latent subquery embeddings. We then minimize the KL divergence between the similarity distribution over documents induced by latent subquery embeddings and that induced by natural language subquery embeddings. This design enables fully differentiable end-to-end joint optimization of the LLM and the retrieval model. To address challenge (2) and encourage the latent tokens to capture meaningful semantics, we incorporate a parallel latent decoding mechanism that converts latent tokens into natural language thoughts and subqueries. During inference, this latent decoding process is optional, enabling a trade-off between transparency and efficiency. Since this latent decoding process depends only on the latent tokens, all thoughts and subqueries across different steps can be decoded in parallel, reducing the latency of the decoding process. Our main contributions are summarized as follows:

- We introduce LatentRAG, a novel agentic RAG framework that performs reasoning and retrieval in the latent space, reducing the latency overhead of explicit thought and subquery generation.
- We propose a latent-space alignment objective that jointly optimizes the LLM and retrieval model, enabling latent tokens to serve as effective retrieval queries while supporting end-to-end training.
- We incorporate a parallel decoding mechanism that translates latent tokens into explicit thoughts and subqueries, improving transparency while maintaining more efficient than explicit agentic RAG.

Extensive experiments on seven benchmark datasets show that LatentRAG achieves performance while significantly reducing latency overhead by approximately 90% on average, approaching the latency of traditional single-step RAG.

## 2 Related Work

### Agentic RAG
Recent advances in RAG have shifted beyond traditional single-step methods [10, 11] toward agentic RAG approaches [23, 21, 15], which perform multi-step retrieval by iteratively generating intermediate thoughts and subqueries. Early agentic RAG methods [14, 38, 18] primarily rely on prompting strategies to enable LLMs to interact with retrieval systems. To improve the retrieval ability of LLMs, Self-RAG [40] and AutoRAG [41] construct synthetic training data from RAG benchmark datasets for supervised fine-tuning. Some methods [24, 42, 43] further introduce mechanisms to balance internal knowledge and external retrieval, enabling LLMs to retrieve only when internal knowledge is insufficient. To mitigate the reliance on supervised training data and promote more flexible search strategies, a growing line of work [19, 44–46] formulates agentic RAG as a Markov decision process, where LLMs learn an optimal decision policy to interact with the retrieval system via reinforcement learning (RL). Recent RL-based approaches further incorporate fine-grained intermediate reward functions [47–50] and explore parallel retrieval strategies [51, 25, 52]. As discussed in the introduction, all these existing methods require generating long sequences of thoughts and subqueries in the language space, leading to substantial latency. In contrast to existing approaches, we explore performing reasoning and retrieval in the latent space, avoiding long textual thought and subquery generation and achieving significant efficiency gains.

### Latent Reasoning
Latent reasoning [29, 53] reduces the latency overhead of explicit chain-of-thought (CoT) reasoning [20] by operating in the continuous hidden states of LLMs, but existing work primarily focuses on generation-only tasks [32, 26] without external retrieval. Early research explores adding filler tokens to enable LLMs to allocate more computation within the hidden states before generating outputs [32, 54]. Coconut [26] proposes an autoregressive latent reasoning paradigm, where each latent token, i.e., a generated hidden state, is recursively fed back into the LLM to generate the next latent token. While the training process of Coconut is only supervised by the final answer, some methods [55, 28, 56, 57] further utilize information generated by explicit CoT.
```

### --- Page 0004 ---

```markdown
# Page 4

as intermediate supervision to improve the training process. To enhance semantic consistency and address the distributional mismatch between the latent token space and the model input space, recent approaches [58–60] constrain latent representations as mixtures of the language token embeddings. Some methods [61, 62] introduce lightweight assistant models to generate latent tokens, thereby improving efficiency while avoiding disrupting the capabilities of the base LLM. Latent reasoning has been extended to practical applications, including retrieval. CLRa [63] leverages latent reasoning to compress retrieved information in single-step RAG, while a concurrent work, LaSER [64], develops a dense retrieval model based on latent reasoning. Despite the rapid advancement of latent reasoning, its application to agentic RAG introduces several challenges as discussed in the introduction, leaving this area largely unexplored. In this paper, we pioneer the integration of latent reasoning into agentic RAG and further propose a latent retrieval mechanism, significantly reducing latency overhead.

## 3 Preliminaries

Following the standard setting in prior RAG research [19, 25], we study the question-answering (QA) task defined as follows. Given a question $q$, the objective is to generate an answer $a$ by retrieving the necessary information from a large corpus $D = \{d_1, d_2, \ldots, d_n\}$, where each $d_i$ represents a document. To simplify notation, for each symbol that represents natural language text (e.g., $d_i$), we use the same symbol to denote its token sequence.

LLMs are widely used for solving the QA task. An LLM maps an input sequence to an output sequence through two stages: prefill and decoding. In the prefill stage, all input tokens are processed in parallel to compute the key-value (KV) cache. In the decoding stage, output tokens are generated autoregressively, where each token is produced based on the KV cache of the input tokens and previously generated tokens. Due to autoregressive dependencies, the decoding stage can only generate output tokens in a token-by-token manner, leading to substantial latency for long outputs.

RAG systems augment LLMs with information retrieved from an external retrieval system. Two types of retrieval models are widely adopted [15, 65]: sparse retrieval models, which rely on exact token-level matches, and dense retrieval models, which encode the query and documents into continuous embeddings and select top-k documents based on cosine similarity. Dense retrieval models capture deeper semantic similarity than sparse retrieval models, leading to superior performance on RAG benchmarks [66, 67]. Thus, in this work, we focus on dense retrieval models.

Agentic RAG methods perform multi-step generation and retrieval, as shown in Fig. 2(a). At each iteration, the LLM generates a reasoning thought and a corresponding subquery, which is then used to retrieve relevant information from an external retrieval system. Formally, at iteration $t$, the historical interaction trajectory is denoted as a sequence:

$$
I_t = (t_0, s_0, \ldots, \tau_{t-1}, s_{t-1}, c_{t-1}),
$$

where $\tau_i$ represents the $i$-th thought, $s_i$ is the $i$-th generated subquery, and $c$ comprises the contents of the top-k documents retrieved using $s_i$. Conditioned on the question $q$ and the interaction trajectory $I_t$, the agent first performs reasoning by producing the next thought $\tau_t$ and subsequently generates the next subquery $s_t$, denoted jointly as $(\tau_t, s_t) = g_{\text{LLM}}(q, I_t; \omega_{\text{LLM}})$, where $\omega_{\text{LLM}}$ represents the parameters of the LLM. After the reasoning process, if the agent concludes that sufficient information has been gathered, it generates a final answer, expressed as $(\tau_t, a) = g_{\text{LLM}}(q, I_t; \omega_{\text{LLM}})$.

## 4 Methodology

LatentRAG adopts a similar procedure to traditional explicit agentic RAG described in Sec. 3, where the LLM agent iteratively generates thoughts and subqueries, and the subqueries are then used to retrieve relevant information. Unlike explicit agentic RAG methods that generate thoughts and subqueries in the language space, LatentRAG operates in the latent space and only produces latent tokens, i.e., the last-layer hidden states, for thoughts and subqueries (Sec. 4.1). The latent subquery tokens are then used as inputs to retrieve relevant documents (Sec. 4.2). To improve transparency, the latent thought and subquery tokens can be decoded into natural language via the latent decoding process (Sec. 4.3). The model is trained with a joint objective that combines losses from different components (Sec. 4.4). The overall framework is shown in Fig. 2.
![Detailed description of the chart](assets/page_0004_img_1.png)
```

### --- Page 0005 ---

```markdown
![Traditional explicit agentic RAG methods and LatentRAG pipeline](assets/page_0005_img_1.png)

## 4.1 Generation with Latest Tokens

We replace the explicit thoughts $t_i$ and subqueries $s_i$ in Eq. 1 with sequences of special tokens $t_f$ and $s_f$, respectively. Here $t_f = \langle \text{think}_1, \ldots, \text{think}_m \rangle$ denotes a sequence of $m$ special thought tokens, and $s_f = \langle \text{query}_1, \ldots, \text{query}_y \rangle$ denotes a sequence of $n$ special subquery tokens. At iteration $t$, the interaction trajectory $I_t$ is denoted as a sequence:

$$
I_t = \langle t_f^0, c_0, \ldots, t_f^{t-1}, s_f^{1}, c_{t-1} \rangle. \tag{2}
$$

The special tokens serve as latent composition slots, allowing the LLM to allocate additional internal computation without generating explicit natural language thoughts and subqueries. During the prefill stage, the special tokens are processed and parallel, producing their last hidden states $H_t^f$ and $H_s^f$, which are referred to as latent thought and subquery tokens, respectively.

Given the question $q$ and the interaction trajectory $I_t^f$, we append the input with the special thought tokens $t_f$ and let the LLM decide an action token from the last latent thought token:

$$
\alpha_t = \text{glue}(q, t_f^t; \theta_{LLM}), \tag{3}
$$

where $\alpha_t \in \langle \text{query}, \text{answer} \rangle$ represents whether to proceed with retrieval by generating a subquery or to terminate by producing the final answer. If $q = \langle \text{query} \rangle$, we append the tokens $s_f$ to the input sequence $(q, t_f; t_f)$ and let the LLM generate the latent subquery tokens:

$$
H_s^f = \text{fuse}(s_f; q, I_t^f; \theta_{LLM}). \tag{4}
$$

where $\theta_{LLM}$ denotes a single forward pass of the LLM. The obtained latent subquery tokens $H_s^f$ are used to retrieve relevant top-$k$ documents, which constitute the retrieved content (see Sec. 4.2). If $\alpha_t = \langle \text{answer} \rangle$, the model continues to generate the final answer a.
```

### --- Page 0006 ---

```markdown
## 4.2 Latent Retrieval

We use the generated latent subquery tokens $H^s_t$ to retrieve relevant content $c_t$. Since these latent tokens reside in the output space of the LLM and are not directly compatible with the input space of the retrieval model, we add a lightweight projector module $Proj_{ret}$ to bridge the two spaces. The projector is composed of a bidirectional self-attention layer and a position-wise feed-forward network (FFN) layer. The projected latent subquery tokens are fed into a trainable retrieval model to obtain the latent subquery embedding:

$$
v^s_t = f_{ret}(Proj_{ret}(H^s_t; \theta_{ret})). \tag{5}
$$

Here $\theta_{ret}$ denotes the parameters of the retrieval model, which are initialized from a pretrained model and will be optimized during the fine-tuning process. Since ground truth documents are not available in our setting, we train the model to produce latent subquery embeddings that approximate the retrieval behavior induced by the corresponding natural language subqueries. Specifically, each natural language subquery $s_t$ in the trajectory is encoded using a reference retrieval model to produce a reference embedding $v^*_t$:

$$
v^*_t = f_{ret}(s_t; \theta_{ret}), \tag{6}
$$

where the reference retrieval model is initialized from the same pretrained model as the trainable one, but its parameters $\theta_{ret}$ remain frozen during the fine-tuning process. The reference embeddings are used to retrieve top-$k$ documents from the corpus, which are treated as pseudo-relevant documents.

To learn subquery embeddings that align with relevant documents, a common practice is to use the InfoNCE loss [68], which pulls query embeddings closer to positive documents while pushing them away from negatives. However, in our setting, pseudo-relevant documents are not grounded truths and may contain substantial noise. In addition, unlike large-scale dense retrieval pretraining settings that rely on hundreds of millions of labeled query–document pairs [33, 34], an aggregate RAG is typically trained with only tens of thousands of samples [19, 15]. Such data limitations lead us to leverage the retrieval model's well-suited features in the subsequent RAG setting.

To better leverage the prior knowledge encoded in the pretrained retrieval models, we introduce a retrieval objective based on Kullback–Leibler (KL) divergence. Specifically, for each subquery $s_t$ and each candidate document $d_k$, we compute the following cosine similarity-based probabilities using the reference subquery embedding and the corresponding latent subquery embedding:

$$
p_i(s_t) = \frac{\exp(\cos(v^*_t, d_k)/\beta)}{\sum_{j=1}^{N_d} \exp(\cos(v^*_t, d_j)/\beta)}, \quad q_i(s_t) = \frac{\exp(\cos(v^*_t, d_k)/\beta)}{\sum_{j=1}^{N_d} \exp(\cos(v^*_t, d_j)/\beta)} \tag{7}
$$

where $\beta$ is the temperature parameter that controls the sharpness of the distribution. As the number of candidate documents, where the candidate set consists of all in-batch pseudo-relevant documents. The retrieval loss function is defined as the KL divergence between both distributions:

$$
L_{ret} = \frac{1}{|\mathcal{B}|} \sum_{i=1}^{N_d} \sum_{s_t \in \mathcal{E}_s} p_i(s_t) \log \left( \frac{p_i(s_t)}{q_i(s_t)} \right) \tag{8}
$$

where $\mathcal{E}_s$ denotes all the subqueries in a training batch. An alternative objective is to directly align $v^*_t$ and $v^*_s$, by minimizing cosine distance. However, our ablation experiments show that it yields lower performance compared to the KL objective.

## 4.3 Latent Decoding

To improve transparency of the decision-making process and enhance latent representation learning, we introduce a latent decoding objective. The key idea is to optimize the LLM to reconstruct the corresponding natural language sequences directly from the generated latent tokens.

We add projector modules $Proj_t$ and $Proj_s$ to map latent thought and subquery tokens into the LLM input space, respectively. The projector modules follow the same structure as the projector introduced in Sec 4.2. The projected latent thought tokens and latent subquery tokens are then fed into the LLM to decode the corresponding natural language thought $\pi_t$ or subquery $s_t$:

$$
\pi_t = g_{LLM}(Proj_t(H^t; \theta_{LLM})), \quad s_t = g_{LLM}(Proj_s(H^s_t; \theta_{LLM})). \tag{9}
$$
```

### --- Page 0007 ---

```markdown
| Methods                | NQ    | TQA    | P@1    | P@5    | P@10   | R@1    | R@5    | R@10   | MRR   | Avg.  |
|-----------------------|-------|--------|--------|--------|--------|--------|--------|--------|-------|-------|
| Re-Rank               | 452.1 | 451.5  | 3.14   | 5.27   | 7.29   | 1.30   | 2.54   | 3.39   | 1.19  | 25.04 |
| Search-Rank           | 458.6 | 453.0  | 3.14   | 5.27   | 7.29   | 1.30   | 2.54   | 3.39   | 1.19  | 25.04 |
| Dense-Rank            | 458.6 | 453.0  | 3.14   | 5.27   | 7.29   | 1.30   | 2.54   | 3.39   | 1.19  | 25.04 |
| Dense-Search          | 458.6 | 453.0  | 3.14   | 5.27   | 7.29   | 1.30   | 2.54   | 3.39   | 1.19  | 25.04 |
| LatentRAG             | 458.6 | 453.0  | 3.14   | 5.27   | 7.29   | 1.30   | 2.54   | 3.39   | 1.19  | 25.04 |
| LatentRAG+            | 458.6 | 453.0  | 3.14   | 5.27   | 7.29   | 1.30   | 2.54   | 3.39   | 1.19  | 25.04 |
| Search-Rank+         | 458.6 | 453.0  | 3.14   | 5.27   | 7.29   | 1.30   | 2.54   | 3.39   | 1.19  | 25.04 |
| LatentRAG+           | 458.6 | 453.0  | 3.14   | 5.27   | 7.29   | 1.30   | 2.54   | 3.39   | 1.19  | 25.04 |

The prompts used to format these inputs are provided in Appendix D. The decoding process is optimized using the standard cross-entropy loss between the generated sequence and the corresponding natural language target. This results in two decoding losses: a thought decoding loss $L_{dec}^{thought}$ and a subquery decoding loss $L_{dec}^{subquery}$. The latent decoding loss is the combination of both terms:

$$
L_{dec} = L_{dec}^{thought} + L_{dec}^{subquery} \tag{10}
$$

During inference, this latent decoding process is optional, allowing the LLM agent to perform reasoning and retrieval entirely in the latent space for efficiency. When required, latent tokens can be decoded into natural language for transparency. Since each decoding process depends only on its corresponding latent tokens, all thoughts and subqueries across multiple steps can be decoded in parallel, thus reducing the latency of generating these natural language sequences.

### 4.4 Overall Training Objective

The overall training objective is defined as a weighted combination of the generation loss, retrieval loss, and latent decoding loss:

$$
L = L_{gen} + \lambda_{ret} L_{ret} + L_{dec} \tag{11}
$$

where $\lambda_{ret}$ controls the relative scale of the retrieval loss. We do not introduce additional scaling factors for $L_{gen}$ and $L_{dec}$ since both are derived from the standard LLM cross-entropy objective and thus have comparable magnitudes.

### 5 Experiments

#### 5.1 Experimental Setup

**Datasets.** We evaluate LatentRAG using seven common benchmark QA datasets, comprising three general QA datasets (NQ [69], TriviaQA [70], and PopQA [71]) and four multi-hop QA datasets (HotpotQA [72], 2wiki [73], Musique [74], and Bamboogle [75]). We use the 2018 Wikipedia dump [76] as the corpus for retrieval. More details of the datasets can be found in Appendix C.
```

### --- Page 0008 ---

```markdown
## Baselines

We compare LatentRAG against a diverse set of baselines covering direct inference (Direct Inf.), traditional single-step RAG (Naive RAG [10]), prompt-based agentic RAG (Iter-RefGen [77], Search-oI [18]), and training-based agentic RAG (R1-Searcher [45], ZeroSearch [78], DeepRAG [24], Search-R1 [19], AutoRefine [35]).

## Implementation details

Following previous works [19, 35], we adopt Qwen2.5-7B [79] as the default LLM for all methods. For training-based baselines, we utilize their published model weights to ensure the faithful reproduction of their reported performance. Training trajectories are constructed from a combined training set of NQ and HotpotQA using Search-R1 and AutoRefine. Variants trained on trajectories generated by Search-R1 and AutoRefine are denoted as LatentRAG* and LatentRAG†, respectively. To reduce computational costs, we conduct main experiments using lightweight retrieval models with fewer than 1B parameters, which are among the top-performing models on the MTEB benchmark [80] and cover diverse model architectures, including Qwen3-Embedding-0.6B [83], e5-base-v2 [34], jina-embeddings-v5-text-nano [81], harrier-osv1-270m1, and FZLM-1v2-330M [82]. Unless otherwise specified, we use Qwen3-Embedding-0.6B as the default retriever. To evaluate the trade-off between performance and latency, we report the exact match (EM) score [19] and the average latency per question. Latency is measured on a single NVIDIA H100 GPU with 94 GB memory by default. More implementation details are in Appendix B.

## 5.2 Main Results

### Overall performance and latency

As shown in Table 1, advanced agentic RAG methods such as Search-R1 and AutoRefine achieve superior performance over naive single-step RAG, but incur substantially higher latency, with an average overhead of around 15× the latency in single-step RAG. This latency gap is more pronounced on multi-hop QA datasets. In contrast, LatentRAG trained on trajectories from Search-R1 and AutoRefine achieves comparable performance, with retrieval differences within 5%, while significantly reducing latency by approximately 90%. This advantage holds consistently across diverse retrieval models. Fig. 1 shows that LatentRAG significantly reduces latency in both the retrieval and subquery generation.

Compared to other retrieval models, we observe a relatively larger performance drop when using e5-base-v2. To investigate the source of this discrepancy, we analyze the embedding spaces of different retrieval models. As shown in Fig. 4 in the Appendix, e5-base-v2 exhibits severe anisotropy [83, 84], indicating that the embeddings produced by the model are highly concentrated within a narrow cone on a hypersphere. This skewed distribution makes it difficult for the LLM to adapt to the retrieval space. More analysis is provided in Appendix E.1.

### Latent decoding efficiency

Latent decoding is an option for improving transparency at the cost of additional latency. To quantify this overhead, Table 2 reports the latency of LatentRAG with and without latent decoding. Latent decoding increases the overall latency of LatentRAG by approximately 4–5×. Nevertheless, it still reduces latency by 63.3% and 47.4% compared to Search-R1 and AutoRefine, respectively. The efficiency gain stems from the removal of sequential dependencies, enabling parallel decoding across steps. The actual speedup is bounded by the longest sequence in the batch, which determines the number of decoding steps required. We report the max length ratio in Table 2, defined as the fraction of tokens in the longest thought or subquery sequence over the total decoding length. A higher ratio indicates a more imbalanced distribution of sequence lengths. In particular, LatentRAG* exhibits a larger max length ratio, which explains its less pronounced efficiency gains. Further analysis is provided in Appendix E.2, along with case studies of decoded examples in Appendix E.7.

### Scaling model size

We study scalability along two orthogonal dimensions. For retrieval model scaling, we evaluate Qwen3-Embedding-0.6B, 4B, and 8B [33] with a fixed 7B LLM. For LLM scaling, we evaluate Qwen2.5-3B, 7B, and 14B with a fixed Qwen3-Embedding-0.6B retrieval model. Larger retrieval models produce higher-dimensional embeddings, resulting in a substantially larger

| Methods          | Lat. (ms) | Max Len. Ratio (%) |
|------------------|-----------|---------------------|
| Search-R1        | 532.17    | 37.7                |
| LatentRAG* w/o decoding | 593.40    | 39.0                |
| LatentRAG* w/ decoding | 1,170.63  | 33.70               |
| AutoRefine       | 582.77    | 43.14               |
| LatentRAG       | 2,540.47  | 42.13               |

`https://huggingface.co/microsoft/harrier-osv1-270m`
```

### --- Page 0009 ---

```markdown
![Performance and latency results across different retrieval model and LLM sizes.](assets/page_0009_img_1.png)

| Methods                | EM (%) | Success (%) | Overlap (%) |
|-----------------------|--------|-------------|-------------|
| Search-R1             | 42.47  | 61.17       | 79.41       |
| LatentRAG (w/ KL loss)| 43.46  | 60.15       | 69.31       |
| (-) Cosine loss       | 42.56  | 61.76       | 68.31       |
| (-) InfoNCE loss      | 41.86  | 58.60       | 67.08       |
| (-) w/o retriever     | 41.81  | 59.07       | 57.92       |
| (-) w/o decoding      | 40.61  | 60.64       | 57.38       |

As shown in Fig. 3, performance improves with increasing model size along both dimensions. Scaling the retrieval model introduces negligible latency overhead, as the retrieval process can be efficiently parallelized. In contrast, scaling the LLM leads to substantial latency increases for SearchR1 due to increased decoding time for thought and subquery generation. Our method achieves comparable performance across most settings and yields improvements in the 3B LLM setting while significantly reducing inference latency.

5.3 Ablation Studies  
We conduct ablation studies on key design choices to validate their effectiveness. Specifically, we replace the KL-based retrieval objective in Eq. 8 with two alternative choices: (i) a cosine loss, which directly minimizes the cosine distance between the latent subquery embedding $v_s$ and the corresponding reference subquery embedding $v_r$, and (ii) a standard InfoNCE loss [68], which is widely used for training retrieval models. We further consider two ablation settings: (iii) removing the pretrained retrieval model and relying solely on the LLM to produce subquery embeddings, and (iv) removing the latent decoding loss in Eq. 10 during training. We report the average EM score as well as two retrieval related metrics: (a) retrieval success rate, defined as the proportion of successful iterative retrievals where the retrieved documents contain the ground truth answer, and (b) retrieval overlap, defined as the proportion of documents retrieved by Search-R1 that are also retrieved by our model.

As shown in Table 3, LatentRAG with the proposed KL-based objective achieves better EM score and retrieval success rate compared to the cosine and InfoNCE alternatives. The cosine loss yields the highest retrieval overlap ratio, indicating closer imitation of the teacher model Search-R1. However, its performance is lower than that of the KL-based variant, suggesting that overly aligning with the teacher model may limit model capacity and lead to suboptimal performance. Removing the pretrained retrieval model also degrades the performance, highlighting the importance of the inductive bias provided by the pretrained retrieval model. Finally, removing the latent decoding loss leads to performance degradation, suggesting that latent decoding not only improves transparency at inference time, but also facilitates the learning of latent representations during training.

6 Conclusion  
In this paper, we propose LatentRAG, an efficient semantic RAG framework that shifts both reasoning and retrieval from discrete language space to continuous latent space. Experiments show that LatentRAG achieves performance comparable to existing RAG methods while reducing latency by approximately 90%. To improve transparency, the latent tokens can be optionally decoded into natural language with additional latency overhead, while still achieving an overall 40–60% reduction in latency compared to the corresponding baselines. Experiments across different model scales further demonstrate the general applicability of LatentRAG.
```

### --- Page 0010 ---

```markdown
# References

[1] Jinqi Lai, Wensheng Gan, Jiayang Wu, Zhenlian Qi, and Philip S Yu. Large language models in law: A survey. *AI Open*, 2024.

[2] Peng-Yuan Wang, Tian-Shuo Liu, Chenyang Wang, Ziniu Li, Yidi Wang, Shu Yan, Chengxing Jia, Xu-Hai Liu, Xinwei Chen, Jiacheng Xu, et al. A survey on large language models for mathematical reasoning. *ACM Comput. Surv.*, 2025.

[3] Karan Singhal, Tao Tu, Juraj Gottwies, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question answering with large language models. *Nat. Med.*, 2025.

[4] Cunking Wang, Xiaozie Liu, Yuanhua Yue, Qipeng Guo, Xiangkun Hu, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunhai Yao, Xuming Hu, Zehan Qi, Wenyang Gao, Yidong Wang, Linyi Yang, Jindong Wang, Xing Xie, Zheng Zhang, and Yue Zhang. Survey on factuality in large language models. *ACM Comput. Surv.*, 2025.

[5] Yuxia Wang, Minghan Wang, Muhammad Arslan Manzoor, Fei Liu, Georgi Nenkov Georgiev, Rocktim J. Das, and Preslav Nakov. Factuality of large language models: A survey. In *EMNLP*, 2024.

[6] Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Cheng, Chen Chen, and Jundong Li. Knowledge editing for large language models: A survey. *ACM Comput. Surv.*, 2024.

[7] Mingyang Wang, Alisa Stoll, Lukas Lange, Heike Adel, Hinrich Schütze, and Jannik Strodthoff. Bring your own knowledge: A survey of methods for LLM knowledge expansion. *arXiv preprint arXiv:2502.12598*, 2023.

[8] Ziwei Ji, Naycon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Matdoto, and Pascal Fung. Survey of hallucination in natural language generation. *ACM Comput. Surv.*, 2023.

[9] Li Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. *ACM Trans. Inf. Syst.*, 2023.

[10] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-wei Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. In *NeurIPS*, 2020.

[11] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In *ICML*, 2020.

[12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*, 2023.

[13] Boci Peng, Yun Zhu, Yongchao Liu, Xiaobo He, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. Graph retrieval-augmented generation: A survey. *ACM Trans. Inf. Syst.*, 2025.

[14] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In *ACL*, 2023.

[15] Aditi Singh, Abu Elhessam, Saket Kumar, and Tala Talaei Kheir. Agentic retrieval-augmented generation: A survey on agentic RAG. *arXiv preprint arXiv:2501.09136*, 2025.

[16] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In *ICLR*, 2023.

[17] Timon Schick, Jane Dwidweid-Yu, Roberto Desi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In *NeurIPS*, 2023.

[18] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhu, Yujia Zhou, Yutao Zhu, Peitan Zhang, and Zhicheng Dou. Search-01: Agentic search-enhanced large reasoning models. In *EMNLP*, 2025.

[19] Bowen Jin, Hans Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. In *COAL*, 2025.
```

### --- Page 0011 ---

```markdown
| Reference | Citation |
|-----------|----------|
| [20] | Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V. Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022. |
| [21] | Yanqing Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei, Henry Peng Zou, Xiao Luo, Yusheng Zhao, et al. Towards agentic RAG with deep reasoning: A survey of RAG-reasoning systems in LLMs. In Findings of EMNLP, 2025. |
| [22] | Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan O. Arik, and Jiawei Han. An empirical study on reinforcement learning for reasoning-search interleaved LLM agents. arXiv preprint arXiv:2505.15117, 2025. |
| [23] | Minhua Lin, Zongyu Wu, Zhichao Xu, Hui Liu, Xianfeng Tang, Qi He, Charu Aggarwal, Xiang Zhang, and Suhang Wang. A comprehensive survey on reinforcement learning-based agentic search: Foundations, roles, optimizations, evaluations, and applications. arXiv preprint arXiv:2510.16724, 2025. |
| [24] | Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and Jie Zhou. DeepRAG: Thinking to retrieve step by step for large language models. In ICLR, 2026. |
| [25] | Zhiwen Tan, Jiaming Huang, Qintong Wu, Hongxuan Zhang, Chenyi Zhuang, and Jinlie Gu. RAG-R1: Incentivizing the search and reasoning capabilities of LLMs through multi-query parallelism. In AAAI, 2024. |
| [26] | Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. In COLT, 2025. |
| [27] | Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on latent reasoning. arXiv preprint arXiv:2505.16782, 2025. |
| [28] | Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171, 2024. |
| [29] | Rui-Fei Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfà Huang, Dawei Zhu, Hao Wang, Kexin Wu, Xuanliang Zhang, Yong Shan, et al. A survey on latent reasoning. arXiv preprint arXiv:2507.62023, 2023. |
| [30] | Loïc Barrault, Paul-Ambriose Duquesne, Maha Elbayad, Artem Kozhevnikov, Belen Alastruoy, Pierre Andrews, Mariano Corria, Guillaume Couloiran, Renaud T. Costa-jussa, David Dalle, et al. Large concept models: Language modeling in a sentence representation space. arXiv preprint arXiv:2412.08821, 2024. |
| [31] | Hong Jin Tack, Jack Lanchantin, Jane Yu, Andrew Cohen, Ilia Kulikov, Janice Lan, Shibo Hao, Yuan-dong Tian, Jason Weston, and Xian Li. LLM pretraining with continuous concepts. arXiv preprint arXiv:2502.05824, 2025. |
| [32] | Sahil Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnav Nagarajan. Think before you speak: Training language models with pause tokens. In ICLR, 2024. |
| [33] | Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Daihyung Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025. |
| [34] | Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majmudar, and Fur Wei. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.05332, 2022. |
| [35] | Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An Zhang, and Xiang Wang. Search and refine during think: Facilitating knowledge refinement for improved retrieval-augmented reasoning. In NeurIPS, 2025. |
| [36] | Jirui Qi, Gabriel Sarti, Raquel Fernández, and Arianna Bisazara. Model internals-based answer attribution for trustworthy retrieval-augmented generation. In EMNLP, 2024. |
| [37] | João Eduardo Batista, Emil Vatai, and Mohamed Wahib. SAFE: Improving LLM systems using sentence-level in-generation attribution. arXiv preprint arXiv:2505.12621, 2025. |
| [38] | Zhenghao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In EMNLP, 2023. |
```

### --- Page 0012 ---

```markdown
| Reference | Citation |
|-----------|----------|
| [39]      | Zhao Xinjie, Fan Gao, Xinyu Song, Yingjian Chen, Rui Yang, Yanran Fu, Yuyang Wang, Yusuke Iwasawa, Yutaka Matsuo, and Irene Li. ReAgent: Reversible multi-agent reasoning for knowledge-enhanced multi-hop QA. In EMNLP, 2025. |
| [40]      | Akari Asai, Zeqiu Wu, Yizhong Wang, Anirup Sill, and Hannanah Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In ICLR, 2024. |
| [41]      | Dongkyu Kim, Byoungwook Kim, Donggeon Han, and Matous Eichbt. AutoRAG: Automated framework for optimization of retrieval augmented generation pipeline. arXiv preprint arXiv:2410.20872, 2024. |
| [42]      | Yujin Cheng, Xiaonan Li, Shimin Li, Qin Zhu, Zhangyue Yin, Yunfan Shao, Liangyu Li, Tianxiang Sun, Hang Yan, and Xipeng Qiu. Unified active retrieval for retrieval augmented generation. In Findings of EMNLP, 2024. |
| [43]      | Soyoung Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-RAG: Learning to adapt retrieval-augmented large language models through question complexity. In NAACL, 2024. |
| [44]      | Mingyang Chen, Linzhang Sun, Tianpeng Li, Chenzheng Zhu, Haofen Wang, Jeff Z Pan, Wen Zhang, Huijun Chen, Fan Yang, Zexuan Zhou, et al. ReSearch: Learning to reason with search for LLMs via reinforcement learning. In NeurIPS, 2025. |
| [45]      | Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R-Searcher: Incentivizing the search capability in LLMs via reinforcement learning. arXiv preprint arXiv:2503.05952, 2025. |
| [46]      | Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. DeepResearcher: Scaling deep research via reinforcement learning in real-world environments. In EMNLP, 2025. |
| [47]      | Yuta Xie, Nathaniel Thomas, Nicklas Hansen, Yang Fu, Li Eran Li, and Xiaolong Wang. TIPS: Turn-level information-potential reward shaping for search-augmented LLMs. In ICLR, 2026. |
| [48]      | Fei Lin, WZhan Zhang, Kun Wan, Wentian Zhang, Kaiyu He, Xinya Du, and Zhiyu Chen. HiPRAG: Hierarchical process rewards for efficient generative retrieval augmented generation. In ICLR, 2026. |
| [49]      | Fengji Zhang, Xinyao Niu, Chengying Yang, Guancheng Lin, Zhongkai Hao, Zhou Ren, Cheng Huang, Jacky Keung, Bei Chen, and Junyang Lin. A*Search: Ambiguity-aware question answering with reinforcement learning. In ICLR, 2026. |
| [50]      | Qingfei Zhao, Roubing Wang, Dingling Xu, Daren Zha, and Limin Liu. R-Search: Empowering LLM reasoning with search via multi-reward reinforcement learning. arXiv preprint arXiv:2506.04185, 2025. |
| [51]      | Shiu Zhao, Tan Yu, Abhank Uka, Japinder Singh, Aditya Shukla, and Rama Akkiraju. ParallelSearch: Train your LLMs to decompose queries and search sub-queries in parallel with reinforcement learning. arXiv preprint arXiv:2508.09030, 2025. |
| [52]      | Zelai Xu, Zhexuan Xu, Ruize Zhang, Chunyang Zhu, Shi Yu, Weilin Liu, Quanlu Zhang, Wenbo Ding, Chao Yu, and Yu Wang. WideSeek-R1: Exploring width scaling for broad information seeking via multi-agent reinforcement learning. arXiv preprint arXiv:2602.04634, 2026. |
| [53]      | Xinlei Yu, Xianghuan Chen, Yongbo He, Tianyu Fu, Cheng Yang, Chengming Xu, Yue Ma, Xiaobin Hu, The Cao, Jie Xu, et al. The latent space: Foundation, evolution, mechanism, ability, and outlook. arXiv preprint arXiv:2604.02209, 2026. |
| [54]      | Jacob Pfau, William Merrill, and Samuel R Bowman. Let’s think dot by dot: Hidden computation in transformer language models. In COLM, 2024. |
| [55]      | Zhenyi Shen, Hanjian Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. CODI: Compressing chain-of-thought into continuous space via self-distillation. In EMNLP, 2025. |
| [56]      | Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. SimCOT: Supervised implicit chain-of-thought. In ICLR, 2026. |
| [57]      | Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chengyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. Soft thinking: Unlocking the reasoning potential of LLMs in continuous concept space. In NeurIPS, 2025. |
```

### --- Page 0013 ---

```markdown
| Reference | Citation |
|-----------|----------|
| [59]     | Yufa Zhou, Yixiao Wang, Xunjian Yin, Shuyan Zhou, and Anru R Zhang. The geometry of reasoning: Flowing logics in representation space. In ICLR, 2026. |
| [60]     | Jingfeng Zheng, Liang Pang, Zihao Wei, Shichen Xu, Zenghao Duan, Kun Xu, Yang Song, Huawei Shen, and Xueqi Cheng. Latent reasoning in LLMs as a vocabulary-space suppression. arXiv preprint arXiv:2510.15222, 2025. |
| [61]     | Yinhai He, Wendy Zheng, Yaochen Zhu, Zaiyi Zheng, Lin Su, Sriram Vasudevan, Qi Guo, Liangjie Heng, and Jundong Li. SemCoT: Accelerating chain-of-thought reasoning through semantically-aligned implicit tokens. In NeurIPS, 2025. |
| [62]     | Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. SoftCoT: Soft chain-of-thought for efficient reasoning with LLMs. In ACL, 2025. |
| [63]     | Jie He, Richard He Bai, Sinead Williamson, Jeff Z Pan, Navdeep Jaitly, and Yizhe Zhang. CLaR: Bridging retrieval and generation with continuous latent reasoning. arXiv preprint arXiv:2511.18659, 2025. |
| [64]     | Jiajie Jin, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengxin Xie, Yutao Zhu, and Zhicheng Dou. LASER: Internalizing explicit reasoning into latent space for dense retrieval. arXiv preprint arXiv:2603.01425, 2026. |
| [65]     | Waqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on RAG meeting LLMs: Towards retrieval-augmented large language models. In KDD, 2024. |
| [66]     | Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao Yang, and Ji-Rong Wen. FlashRAG: A modular toolkit for efficient retrieval-augmented generation research. In WWW, 2025. |
| [67]     | Yuanjie Liu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjing Wang, Hao Wu, Huanyong Liu, Tong Xu, and Enhong Chen. CRUD-RAG: A comprehensive Chinese benchmark for retrieval-augmented generation of large language models. ACM Trans. Inf. Syst., 2025. |
| [68]     | Aaron van der Oord, Yizhe Li, and Dorian Yaze. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. |
| [69]     | Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Daniel Epstein, Ilia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. TACL, 2019. |
| [70]     | Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In ACL, 2017. |
| [71]     | Alex Malin, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannah Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In ACL, 2023. |
| [72]     | Zhilin Yang, Peng Qi, Saizheng Zhang, Yosua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In EMNLP, 2018. |
| [73]     | Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In COLING, 2020. |
| [74]     | Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. TACL, 2022. |
| [75]     | Ofir Press, Munu Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the computational gap in language models. In Findings of EMNLP, 2023. |
| [76]     | Vladimir Karpukhin, Barlas Oguz, Seung Won Kim, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In EMNLP, 2020. |
| [77]     | Zhihong Xu, Yeyun Gong, Yelong Shen, Minihie Huang, Nan Duan, and Weizhu Chen. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. In Findings of EMNLP, 2023. |
| [78]     | Hao Sun, Zile Qiao, Jiyuan Guo, Xuanbo Fan, Yinyang Hou, Yong Jiang, Penglin Xie, Yan Zhang, Fei Huang, and Jinjren Zhou. ZeroSearch: Incentivize the search capability of LLMs without searching. arXiv preprint arXiv:2505.04588, 2025. |
```

### --- Page 0014 ---

```markdown
| Reference                                                                                                           |
|---------------------------------------------------------------------------------------------------------------------|
| [79] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhou Tu, Jianwei Zhang, Jianxin Yang, Jiayi Yang, Jinger Zhou, Junyang Lin, Kai Dong, Kenming Lu, Keqin Bao, Kevin Yang, Le Yu, Miki Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qu. Qwen2 technical report. arXiv preprint arXiv:2412.15115, 2024. |
| [80] Niklas Muennighoff, Nouamane Tazi, Loïc Maguire, and Nils Reimers. MTEB: Massive text embedding benchmark. In EACL, 2023. |
| [81] Mohammad Kalim Akram, Saba Sturua, Nastia Havrishenko, Quentin Herreros, Michael Günther, Maximilian Werk, and Han Xiao. jina-embeddings-v5-text: Task-targeted embedding distillation. arXiv preprint arXiv:2602.15547, 2026. |
| [82] Ziyin Zhang, Zihan Liao, Hang Yu, Peng Di, and Rui Wang. P2LM-v2: Inclusive, performant, and efficient embeddings for a multilingual world. arXiv preprint arXiv:2603.19223, 2026. |
| [83] Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. On the sentence embeddings from pre-trained language models. In EMNLP, 2020. |
| [84] Wenxuan Zhou, Bill Yuchen Lin, and Xiang Ren. IsoBN: Fine-tuning BERT with isotropic batch normalization. In AAAI, 2021. |
| [85] nostalgebraist. Interpreting GPT: the logit lens. https://www.lesswrong.com/posts/acRKB8wDpa6R6v/interpreting-gpt-the-logit-lens, 2020. |
```

### --- Page 0015 ---

```markdown
# A  Broader Impacts and Limitations

## Broader impacts. 
This work proposes an efficient agentic RAG framework that performs reasoning and retrieval in the latent space. The proposed approach can be applied to a wide range of information-seeking scenarios, such as legal or clinical question answering [1, 3], and improve overall efficiency in these tasks. More broadly, as most existing work focuses on training agents to use search engines originally designed for humans, this work suggests a shift from human-oriented text-based search engines to agent-oriented embedding-based search engines that better support agent usage. This provides a potential direction for rethinking search engines in the era of agentic systems.

## Limitations & future work. 
Our method relies on SFT over trajectories generated by existing agentic RAG methods, and its performance is therefore partly bounded by the quality of the training data. This hinders the model from directly learning an optimal retrieval policy through interactions with the retrieval system. Nevertheless, our approach yields strong and efficient initial models that serve as an effective foundation for future research. Future work could investigate reinforcement learning to improve performance by encouraging exploration and exploitation.

# B  Implementation Details

## Training data construction. 
As described in the main paper, we combine the training sets from NQ and HotpotQA to construct a unified training dataset. We then build training trajectories using interaction data generated by Search-R1 and AutoRefine on this unified training dataset. Each trajectory consists of the question, intermediate reasoning thoughts, subqueries, retrieved document chunks, and the final generated answer. AutoRefine introduces an additional refinement stage to improve the initially retrieved documents. To maintain a consistent trajectory format with Search-R1, we merge the refinement text into the reasoning thoughts. We retain only those trajectories that correctly format and include all relevant finer-grained components from the different elements in the generated trajectories, we introduce a set of special tokens to explicitly mark structural elements in the output, such as `<Answer>`...`</Answer>`. In contrast, these tags are typically tokenized into multiple subword units in Search-R1 and AutoRefine. This difference may introduce minor variations in generation time. However, its impact is negligible compared to the overall latency reduction achieved by our framework.

## Computing resources & parallelization strategies. 
For training, we optimize LatentRAG on a single compute node equipped with two NVIDIA H100 GPUs, each with 94 GB of memory. Each training job takes about 24 to 48 hours to complete. To reduce GPU memory consumption, we enable gradient checkpointing to minimize the storage of intermediate activations. For distributed training, we adopt DeepSpeed ZeRO1, which shards the optimizer states across GPU devices while keeping gradients and model parameters fully replicated. This design avoids additional communication overhead associated with parameter and gradient sharding, thereby maintaining efficient data-parallel training. To address the imbalance in trajectory lengths, we implement a binned batching strategy. Specifically, we partition trajectories into 200 bins according to their lengths and construct each batch by sampling from a single bin. This binned batching strategy ensures that samples within each batch have similar sequence lengths and therefore reduces padding overhead and improves computational efficiency. We use bfloat16 precision and FlashAttention-2 during training. We adopt LoRA with a rank of 16 for parameter-efficient fine-tuning, which significantly reduces the number of trainable parameters, thereby lowering memory and computational costs.

For evaluation, we conduct experiments on a single NVIDIA H100 GPU with 94 GB of memory by default. We deploy the retrieval system using Faiss on the GPU with half-precision indexing and load the LLM on the same device. To ensure a fair comparison across different methods, we measure both LLM prefill and decoding latency using the standard forward pass implemented in Hugging Face Transformers6. For scaling experiments, larger retrieval models produce higher-dimensional parameters.

2. https://github.com/deeppeedai/DeepSpeed  
3. https://github.com/facebookresearch/flash-attention  
4. https://github.com/microsoft/LoRA  
5. https://github.com/facebookresearch/faiss  
6. https://huggingface.co/docs/transformers  
```

### --- Page 0016 ---

```markdown
| NO  | NQ     | TriviaQA | PopQA  | HotpotQA | 2wiki  | Musique | Bambogle |
|-----|--------|----------|--------|----------|--------|---------|----------|
| # Train | 79,168 | –        | –      | 90,447   | –      | –       | –        |
| # Test  | 3,610  | 11,313   | 14,267 | 7,405    | 12,576 | 2,417   | 125      |

document embeddings that exceed the memory capacity of a single GPU. For example, the index built from Qwen3-Embedding-88 occupies approximately 160 GB even with float16 precision. To accommodate this, for all the scaling experiments, we deploy the retrieval system across three H100 GPUs, while using a separate H100 GPU to serve the LLM. This setup ensures sufficient GPU resources for both retrieval and generation, allowing us to report latency under sufficient computational resources, where system bottlenecks are minimized.

### Hyperparameters
We fine-tune the model using LoRA with rank 16 and scaling factor 64, applied to all position weights. The model is trained for 5 epochs with a learning rate of $1 \times 10^{-4}$. The maximum trajectory length is capped at 3000 tokens. For the KL divergence loss, we set the target distribution based on the similarity scores between queries and documents. Specifically, we select the temperature factor that makes the cumulative probability of the top-3 retrieved documents approach 0.5. In practice, this corresponds to setting the temperature to $\beta = 0.03$ in most cases. The loss weight for the retrieval objective is set to $\lambda_{ret} = 1$. For the retrieval model, we remove dropout to reduce noise in the target distribution, while for the LLM we apply a dropout rate of 0.1. We use $m = 4$ thought tokens for each thought generation step and $n = 16$ subquery tokens for each subquery generation step. The training batch size is set to 16. The model is optimized using AdamW optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.999$, and a weight decay of 0.01. For retrieval loss calculation, we retrieve the top-10 identified documents as pseudo-relevant documents and combine them with in-batch negative samples, i.e., the pseudo-relevant documents from other subqueries within the same batch, to enhance the retrieved document set, which is then used for evaluation.

### Evaluation metrics & measurement protocol
We adopt exact match (EM) as the primary performance evaluation metric. The EM score measures whether the final predicted answer exactly matches the ground-truth answer. Before evaluation, both predicted and ground-truth answers are normalized by removing articles (e.g., a, an, the), stripping whitespace, removing punctuation, and converting all text to lowercase. For all retrieval-based metrics, we retrieve the top-3 documents per query. The maximum number of retrieval iterations is set to 4. For efficiency, we report latency, which captures the end-to-end response time from receiving a query to generating the final answer. We sample the first 100 questions from each dataset to estimate latency. To enable fine-grained latency analysis, we report a breakdown of the latency across different stages, including preflight generation, subquery generation, retrieval, and answer generation.

Following prior work [19, 35], we use Qwen2.5 Instruct for inference-based methods due to its stronger instruction-following capabilities. For training-based baselines, we adopt checkpoints released in the original papers that are based on the Base variant of Qwen2.5, which demonstrated better performance in prior work compared to the Instruct variant under training-based settings [19]. We also initialize and fine-tune our model from Qwen2.5 Base for fair comparison.

For stage-wise latency measurement, the embedding time of a natural language subquery is included in the retrieval stage. For our method, we reduce the number of vectors transmitted to the retrieval system, we generate subquery embeddings on the model side from the latent tokens and pass only the resulting embedding vector to the retrieval system. Therefore, the embedding computation time is attributed to the subquery generation stage instead of the retrieval stage. This design leads to higher measured subquery generation time and lower retrieval time for our method. However, this difference does not affect the computation of the overall latency.

### C Dataset Description
We conduct our experiments on seven benchmark QA datasets, following previous works [19, 35]. These datasets include three general QA datasets (NQ [69], TriviaQA [70], and PopQA [71]) and four multi-hop QA datasets (HotpotQA [72], 2wiki [73], Musique [74], and Bambogle [75]). Instead of
```

### --- Page 0017 ---

```markdown
## D Prompt Templates

In this section, we provide all prompt templates used in our framework. Double curly braces {{ ... }} denote runtime placeholders. Prompt D.1 presents the template for latent thought and subquery generation. The latent thought and subquery tokens are derived from the hidden states at the positions of the corresponding special tokens. An action token is predicted based on the final thought token. If the action token `<answer>`, the special subquery tokens in the prompt template are replaced with the answer token `<answer>` to trigger the answer generation process. Prompt D.2 and Prompt D.3 present the templates for latent thought and subquery decoding, respectively.

### Prompt D.1. Prompt template for thought and subquery generation

```
Answer the following question by reasoning step by step and retrieving necessary information at each step:
{{QUESTION}}
<think>...<think>...<query>...<query>,>
<information>
Doc 1 {{[TOP_1_DOCUMENT]}}
Doc 2 {{[TOP_2_DOCUMENT]}}
...
<think>...<think>...<query>...<query>,>
</information>
```

### Prompt D.2. Prompt template for latent thought decoding

```
Decode the thought based on the latent representation: {{[LATENT_THOUGHT_TOKENS]}}
```

### Prompt D.3. Prompt template for latent subquery decoding

```
Decode the subquery based on the latent representation: {{[LATENT_SUBQUERY_TOKENS]}}
```

## E More Experimental Results

### E.1 Embedding Space Analysis of Retrieval Models

In Table 1 of the main paper, compared to other retrieval models, our method exhibits a relatively larger performance drop when using e5-base-v2. To further investigate the source of this discrepancy, we analyze the differences in the geometric properties of the embedding space across different retrieval models. Specifically, for each retrieval model, we generate $l_2$-normalized embeddings for the entire Wikipedia corpus. We then compute the mean direction of all document embeddings produced by that model. Next, we measure the cosine similarity and the angular distance between each document embedding and this mean direction and visualize their respective distributions. A distribution that is skewed toward higher cosine similarities (or lower angles) indicates that the embeddings are concentrated around the mean direction rather than being uniformly distributed over the hypersphere, thereby reflecting a stronger anisotropy [83, 84] in the embedding space.
```

### --- Page 0018 ---

```markdown
![Distribution of cosine similarity and angle between document embeddings and their mean direction.](assets/page_0018_img_1.png)
![Latency reduction using batch latent decoding vs. max length ratio.](assets/page_0018_img_2.png)

| Methods        | NQ    | TriviaQA | PopQA | HotpotQA | ZwiQ | Musig | Bamboo | Average   |
|----------------|-------|----------|-------|----------|------|-------|--------|-----------|
| Search-R@10    | 3.583 | 5.198    | 3.588 | 69.69    | 62.95| 8.686 | 49.94  | 5.372     |
| LatentRAG* w/ decoding | 1,491.62 | 2,675.08 | 2,680.95 | 704.87 | 739.86 | 623.47 | 593.87 | 593.87 |
| LatentRAG*    | 1,456.63 | 1,537.63 | 1,634.54 | 1,710.62 | 2,234.67 | 2,356.67 | 2,320.92 | 1,970.63 |
| AutoRAG        | 4.782 | 42.23    | 49.97 | 534.64   | 526.64 | 585.54 | 482.34 | 4.827     |
| LatentRAG* w/o decoding | 4,916.25 | 4,309.95 | 2,576.14 | 2,541.69 | 607.85 | 629.85 | 1,526.81 | 512.87 |
| LatentRAG* w/o decoding | 7,306.25 | 4,304.43 | 2,576.14 | 2,102.68 | 1,230.86 | 2,130.67 | 2,154.87 | 2,540.47 |

As shown in Fig. 4, the embeddings generated by e5-base-v2 exhibit extremely high cosine similarity and low angular deviation with respect to the mean direction, demonstrating severe anisotropy. This suggests that the embeddings are highly concentrated around a narrow region of the hypersphere, rather than being well spread out. As a result, small approximation errors in the embedding space may lead to completely different retrieval outputs, making it challenging to train a model to faithfully approximate the behavior of the original retrieval model. Moreover, such a skewed distribution may force the LLM to deviate from its original representation geometry to adapt to this skewed concentrated space, which could negatively affect the performance of the LLM.

### E.2 Latent Decoding Efficiency Analysis

As discussed in the main paper, latent decoding improves transparency at the cost of additional latency. A good property of our method is that the decoding of thoughts and subqueries is conditionally independent given the latent tokens. This property allows us to perform parallel decoding across different steps, in contrast to existing agentic RAG methods that generate these sequences sequentially.

To quantify the effect of reduced latency enabled by our parallel decoding strategy, we report detailed latency measurements across multiple datasets and compare them with baseline methods. As shown in Table 5, using latent decoding increases latency by approximately 4–5× compared to the setting without latent decoding. Nevertheless, compared to corresponding baseline methods, our method with latent decoding reduces overall latency by approximately 23-68% across different datasets.

The efficiency gains from parallel decoding are more pronounced when sequence lengths are balanced, as this reduces padding overhead and avoids unnecessary computation. To characterize the impact of sequence lengths...
```

### --- Page 0019 ---

```markdown
| Methods        | NQ   | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboo | Average   |
|----------------|------|----------|-------|----------|-------|---------|--------|-----------|
| Near RAG       | 51.21| 49.74    | 91.94 | 69.44    | 61.84 | 49.11   | 49.85  | 51.45     |
| Answer Gen.    | 39.54| 79.19    | 57.85 | 36.25    | 29.61 | 26.19   | 55.75  | 41.74     |
| Total          | 90.75| 128.93   | 149.79| 105.69   | 91.45 | 75.30   | 105.60 | 93.19     |
|                |      |          |       |          |       |         |        |           |
| Search-R1      | 160.61| 195.35  | 156.45| 111.68   | 153.09| 129.18  | 108.29 | 143.43    |
| Subquery Gen.  | 268.05| 301.75  | 295.15| 242.50   | 236.76| 196.06  | 210.80 | 232.43    |
| Answer Gen.    | 18.14 | 45.48   | 16.81 | 20.10    | 26.76 | 2.06    | 24.76  | 18.41     |
| Total          | 446.80| 542.58  | 468.41| 374.28   | 415.61| 327.30  | 343.85 | 394.27    |
|                |      |          |       |          |       |         |        |           |
| AutoRefine     | 183.29| 207.12  | 182.78| 134.19   | 120.37| 118.79  | 117.90 | 151.73    |
| Subquery Gen.  | 132.55| 139.67  | 123.01| 102.43   | 118.29| 119.80  | 118.29 | 123.76    |
| Answer Gen.    | 18.29 | 45.48   | 16.81 | 20.10    | 26.76 | 2.06    | 24.76  | 18.41     |
| Total          | 334.13| 392.27  | 322.60| 256.72   | 265.42| 240.65  | 261.95 | 290.90    |
|                |      |          |       |          |       |         |        |           |
| LatentRAG      | 184.70| 205.10  | 182.78| 134.19   | 120.37| 118.79  | 117.90 | 151.73    |
| Subquery Gen.  | 132.55| 139.67  | 123.01| 102.43   | 118.29| 119.80  | 118.29 | 123.76    |
| Answer Gen.    | 18.29 | 45.48   | 16.81 | 20.10    | 26.76 | 2.06    | 24.76  | 18.41     |
| Total          | 334.13| 392.27  | 322.60| 256.72   | 265.42| 240.65  | 261.95 | 290.90    |

### E.3 Detailed Stage-wise Latency Comparison
Table 6 shows the detailed stage-wise latency breakdown when using the Qwen3-Embedding-0.6B retrieval model. Compared to naive single-step RAG, Search-R1 and AutoRefine introduce significant latency overhead. The average latency across all datasets is approximately 15× that of near RAG. This overhead mainly comes from the thought and subquery generation stages, which together account for about 90% of the total latency. In contrast, our method, trained on trajectories generated by Search-R1 and AutoRefine, significantly reduces the overall latency by approximately 90% compared to the corresponding baseline.

### E.4 Impact of Trajectory Quality on Model Performance
To investigate the effect of trajectory quality on model performance, we train the same model using trajectories generated by LLMs of different sizes. As shown in Fig. 3 in the main paper, larger LLMs consistently achieve better performance, suggesting that they tend to produce higher-quality trajectories.
```


### --- Page 0020 ---

```markdown
| Methods            | Token Counts         | # Forward Passes |
|--------------------|----------------------|------------------|
|                    | Thought  | Subquery  | Answer  | Others  | Total  |                  |
|--------------------|-----------|-----------|---------|---------|--------|------------------|
| Search-R1          | 121.8 (in) | 379 (out) | 96.0 (in) | 1325.5 (in) | 1694.0 | 169.4            |
| LatentRAG w/o decoding | 13.8 (in)  | 39.0 (in)  | 5.8 (out)  | 1225.5 (in) | 58.0 (in) | 11.7             |
| LatentRAG w/ decoding | 13.8 (in) + 117.7 (out) | 390 (in) + 293 (out) | 5.8 (out) | 1436.2 (in) | 1480.0 | 1528.0 | 52.8             |

![Performance under different numbers of latent thought tokens](assets/page_0020_img_1.png)

E.5 Influence of Latent Token Numbers

To investigate the impact of latent token numbers, we vary the number of latent thought tokens $m$ and the number of subquery tokens $n$ and evaluate the exact match scores under different configurations. As shown in Fig. 6, performance remains relatively stable across different settings. It increases slightly at first and reaches a peak when using 4 thought tokens and 16 subquery tokens, suggesting that while additional latent tokens can provide more expressive capacity and increase performance, excessive tokens may introduce redundancy. Therefore, in our experiments, we set $m = 4$ and $n = 16$.

E.6 Average Token Counts and Number of Forward Passes

To analyze token usage efficiency, we report the average token counts per question. We distinguish between input and output tokens, as output tokens are generated autoregressively and cannot fully be parallelized, typically incurring higher latency and being more costly in practice. For example, in the OpenAI API pricing$^*$, output tokens are typically priced about 6x higher than input tokens. We also report the number of forward passes per question. The number of forward passes corresponds to how many sequential LLM forward computations are required, which typically relates to the overall latency under sufficient hardware resources. Moreover, this latency can be easily reduced by simply scaling up GPU computational resources as it is fundamentally constrained by sequential dependencies in the generation process.

As shown in Table 8, Search-R1 generates substantially more output tokens due to explicit thought and subquery generation, which in turn leads to a large number of LLM forward passes and explains its high latency reported in the main paper. As an option to improve transparency at the cost of additional latency, latent decoding increases the number of output tokens in our method to a level comparable to Search-R1. However, since in our method the thought and subquery sequences across different steps are conditionally independent given the latent tokens, these sequences can be decoded in parallel, which significantly reduces the number of LLM forward passes. Moreover, the decoding process depends only on the latent tokens rather than attending to the full iteration history, which can further reduce computational overhead in practice.

$^*$https://openai.com/api/pricing/
```

### --- Page 0021 ---

```markdown
E.7 Case Studies
To qualitatively analyze the behavior of our method, we present several case studies of the reasoning and retrieval processes of LatentRAG.

Success case analysis. As shown in Success Case 1 & 2, our method successfully learns the reasoning and retrieval patterns of the respective baseline models. Models trained on trajectories from different baselines generate thoughts and subqueries that are similar to those of the original models. For instance, the decoded thoughts of LatentRAG$^2$ are able to capture the refinement structure in the reasoning process of AutoRefine. In Success Case 3, although both models arrive at the correct answer after a sequence of reasoning and retrieval steps, they exhibit redundant retrieval in the final stage. This suggests that undesirable behaviors of the teacher model may also be learned by our method, highlighting the importance of trajectory quality as discussed in Appendix E.4.

Failure case analysis. As shown in Failure Case 1 and 2, although the reasoning and retrieval processes of our method are correct, the model sometimes fails to produce fully consistent outputs, leading to incorrect answers under exact match evaluation. This might indicate that latent representations facilitate the learning of abstract concepts but are less effective for precise lexical output. Nevertheless, our method maintains competitive performance while significantly reducing overall latency by 90%, highlighting the value of latent reasoning and retrieval in generating accurate information for answer generation.

LogitLens analysis. To investigate what information is encoded in each latent thought or subquery token, we leverage LogitLens [85] to analyze the generated latent tokens. LogitLens projects hidden states into the vocabulary space using the unembedding matrix of the LLM, enabling inspection of the token-level information encoded in the hidden states. Fig. 7 & 8 present the top-5 predicted language tokens by logits for each latent token. Although we do not explicitly constrain latent tokens to align with the LLM vocabulary space, the model still distributes these latent representations around semantically related vocabulary regions. In particular, the decoded vocabulary tokens from the thought and subquery tokens of the first step are closely related to the first subquery, while those from later steps gradually shift toward vocabulary regions associated with the second subquery and eventually the final answer. Additionally, unlike natural language tokenization, which typically represents text through a fixed subword decomposition that may split semantic units across multiple tokens, a latent token can encode the whole semantic concept, such as Christianity Today or William Goldman. These findings suggest that performing reasoning and retrieval in the latent space may provide more flexibility and expressivity than operating in natural language space.
```

### --- Page 0022 ---

```markdown
# Success Case 1: Search-R1° vs. LatentRAG°

| **Question** | What is the place of birth of Clara Novello's father? *(Ground truth answer: London)* |
|--------------|---------------------------------------------------------------------------------------|

## Search-R1°

| **[Thought]** | I need to find the place of birth of Clara Novello’s father. I’ll search for it. |
|---------------|-----------------------------------------------------------------------------------|
| **[Subquery]** | Clara Novello's father |
| **[Docs]**    | Chunk_11243233 [Clara Novello]: Ivor Novello carried on this family name. Clara Novello Clara Anastasia Novello (10 June 1818 – 12 March 1908) was an acclaimed soprano, the fourth daughter of Vincent Novello, a musician and music publisher, and his wife, Mary Sibilia Heil. Her acclaimed soprano and pure style made her one of the greatest vocalists, alike in opera, oratorio and on the concert stage, from 1833 onwards. In 1843 she married Count Gigliotti, and retired in 1861. Charles Lamb wrote a poem ("To Clara N.") in her praise. She was born in Oxford Street, London, on 10 June 1818, the fourth daughter. |
| **[Thought]** | I found out that Clara Novello’s father is Vincent Novello. Now I need to find the place of birth of Vincent Novello. |
| **[Subquery]** | place of birth of Vincent Novello |
| **[Docs]**    | Chunk_10890160 [Vincent Novello]: Vincent Novello (6 September 1781 – 9 August 1861), English musician, son of an Italian who married an English wife, was born in London. He was a chorister and organist, but he is best known for bringing to England many works now considered major standards, and with his son he created a major publishing house. As a boy, Novello was a chorister at the Sardinian Embassy Chapel in Duke Street, Lincoln’s Inn Fields, where he learnt the organ from Samuel Webbe; and from 1796 to 1822 he became in succession organist of the Sardinian, Spanish (in Manchester Square) and Portuguese |
| **[Thought]** | I found out that the place of birth of Vincent Novello is London. Now I can provide the answer. |
| **[Answer]**  | London |

## LatentRAG°

| **[Thought]** | I need to find the place of birth of Clara Novello’s father. I’ll search for it. |
|---------------|-----------------------------------------------------------------------------------|
| **[Subquery]** | Clara Novello's father |
| **[Docs]**    | Chunk_11243233 [Clara Novello]: Ivor Novello carried on this family name. Clara Novello Clara Anastasia Novello (10 June 1818 – 12 March 1908) was an acclaimed soprano, the fourth daughter of Vincent Novello, a musician and music publisher, and his wife, Mary Sibilia Heil. Her acclaimed soprano and pure style made her one of the greatest vocalists, alike in opera, oratorio and on the concert stage, from 1833 onwards. In 1843 she married Count Gigliotti, and retired in 1861. Charles Lamb wrote a poem ("To Clara N.") in her praise. She was born in Oxford Street, London, on 10 June 1818, the fourth daughter. |
| **[Thought]** | I found out that Clara Novello’s father is Vincent Novello. Now I need to find his place of birth. |
| **[Subquery]** | Vincent Novello place of birth |
| **[Docs]**    | Chunk_10890160 [Vincent Novello]: Vincent Novello (6 September 1781 – 9 August 1861), English musician, son of an Italian who married an English wife, was born in London. He was a chorister and organist, but he is best known for bringing to England many works now considered major standards, and with his son he created a major publishing house. As a boy, Novello was a chorister at the Sardinian Embassy Chapel in Duke Street, Lincoln’s Inn Fields, where he learnt the organ from Samuel Webbe; and from 1796 to 1822 he became in succession organist of the Sardinian, Spanish (in Manchester Square) and Portuguese |
| **[Thought]** | I found out that the place of birth of Vincent Novello is London. Now I can provide the answer. |
| **[Answer]**  | London |
```

### --- Page 0023 ---

```markdown
# Success Case 2: AutoRefine® vs. LatentRAG®

| **Question** | What is the place of birth of Clara Novello's father? (Ground truth answer: London) |
|--------------|-------------------------------------------------------------------------------------|
| **AutoRefine®** | To answer this question, I need to find out who Clara Novello's father is and then determine his place of birth. |
| **Thought** | Clara Novello's father |
| **Subquery** | Clara Novello's father: Ivor Novello carried on this family name. Clara Novello Clara Anastasia Novello (10 June 1818 – 12 March 1908) was an acclaimed soprano, the fourth daughter of Vincent Novello, a musician and music publisher, and his wife, Mary Sabilla Hell. Her acclaimed soprano and pure style made her one of the greatest vocalists, alike in opera, oratorio and on the concert stage, from 1833 onwards. In 1843 she married Count Gigliocetti, and retired in 1861. Charles Barnet wrote a poem ("To Clara N.") in her praise. She was born in Oxford Street, London, on 10 June 1818, the fourth daughter. |
| **Docs** | <refine>From the documents, I learned that the father of Clara Novello was Vincent Novello. </refine> |
| **Thought** | <refine>From the documents, I learned that the father of Clara Novello was Vincent Novello. </refine> |
| **Subquery** | What is the place of birth of Vincent Novello |
| **Docs** | Chunk_10891040 [Vincent Novello]: Vincent Novello (6 September 1781 – 9 August 1861), English musician, son of an Italian who married an English wife, was born in London. He was a chorister and organist, but he is best known for bringing to England many works now considered major standards, and with his son he created a major publishing house. As a boy, Novello was a chorister at the Sardinian Embassy Chapel in Duke Street, Lincoln’s Inn Fields, where he learnt the organ from Samuel Webbe; and from 1796 to 1822 he became in succession organist of the Sardinian, Spanish (in Manchester Square) and Portuguese. |
| **Thought** | <refine>From the documents, I learned that Vincent Novello was born in London. </refine> |
| **Thought (dec)** | To answer this question, I need to find out who Clara Novello's father is and then determine his place of birth. |
| **Subquery (dec)** | Clara Novello's father |
| **Docs** | Chunk_1124233 [Clara Novello]: Ivor Novello carried on this family name. Clara Novello Clara Anastasia Novello (10 June 1818 – 12 March 1908) was an acclaimed soprano, the fourth daughter of Vincent Novello, a musician and music publisher, and his wife, Mary Sabilla Hell. Her acclaimed soprano and pure style made her one of the greatest vocalists, alike in opera, oratorio and on the concert stage, from 1833 onwards. In 1843 she married Count Gigliocetti, and retired in 1861. Charles Barnet wrote a poem ("To Clara N.") in her praise. She was born in Oxford Street, London, on 10 June 1818, the fourth daughter. |
| **Answer** | <refine>From the documents, I learned that Vincent Novello was born in London. </refine> |
| **Answer** | London |
```

### --- Page 0024 ---

```markdown
# Success Case 3: Search-R1° vs. LatentRAG°

## [Question]
Which film was released first, The Beginning Or The End or Hold Back The Night? (Ground truth answer: The Beginning or the End)

## Search-R1°
| Thought | I need to determine which film, The Beginning or The End or Hold Back The Night, was released first. I'll search for the release years of both films. |
|---------|------------------------------------------------------------------------------------------------------------------------------------------|
| Subquery | The Beginning or The End |
| Docs | Chunk_15130472 [The Beginning Was the End]: The Beginning Was the End is a 1971 pseudo-scientific book written by ... (irrelevant retrieval results) |
| Thought | I found out that The Beginning or The End is a book. Now I need to find the release year of the film Hold Back The Night compare it to the release year of the film based on the book The Beginning or The End. |
| Subquery | Hold Back The Night release year |
| Docs | Chunk_15896613 [Hold Back the Night]: Hold Back the Night is a 1956 American war film about the Korean War based on the 1951 novel by Pat Frank, who had been a war correspondent in Korea. The film was directed by Allan Dwan; his third film with John Payne and his third film about the United States Marine Corps, the others being "Abroad with Two Yanks" (1944) and "Sands of Iwo Jima" (1949). The film tells the story in flashbacks of a bottle of scotch carried by a World War II Marine lieutenant and Korean War captain, Sam MacKenzie. His girl Anne |
| Thought | I found out that Hold Back The Night is a film released in 1956. Now I need to find the release year of The film based on the book The Beginning or The End. |
| Subquery | The Beginning or The End release year |
| Docs | Chunk_10610413 [The Beginning or the End]: seldom rises above cheery imbecility. According to MGM records, "The Beginning or the End" was made on a budget of $2,632,000.0, but earned $21,200.0 in the United States and Canada and $721,000.0 elsewhere, resulting in a loss to the studio of $1,596,000.0. The Beginning or the End is a 1947 American docudrama film about the development of the atomic bomb in World War II, directed by Norman Taurog, starring Brian Donlevy and Hume Cronyn, and released by Metro-Goldwyn-Mayer. The film dramatizes the creation of the atomic bomb in the Manhattan Project. |
| Thought | I found out that The Beginning or The End was released in 1947. Now I can compare the release year of Hold Back The Night. |
| Subquery | Release year of Hold Back The Night (Redundant retrieval) |
| Docs | Chunk_15896613 [Hold Back the Night]: ... |
| Thought | I found out that Hold Back The Night was released in 1956. Now I can compare it to the release year of The Beginning or the End, which I found to be 1947. |
| Answer | The Beginning or the End |

## LatentRAG°
| Thought | I need to determine which film, The Beginning or The End or Hold Back the Night, was released first. I'll search for the release years of both films. |
| Subquery | The Beginning Or The End |
| Docs | Chunk_10610413 [The Beginning or the End]: seldom rises above cheery imbecility. According to MGM records, "The Beginning or the End" was made on a budget of $2,632,000.0, but earned $21,200.0 in the United States and Canada and $721,000.0 elsewhere, resulting in a loss to the studio of $1,596,000.0. The Beginning or the End is a 1947 American docudrama film about the development of the atomic bomb in World War II, directed by Norman Taurog, starring Brian Donlevy and Hume Cronyn, and released by Metro-Goldwyn-Mayer. The film dramatizes the creation of the atomic bomb in the Manhattan Project. |
| Thought | I found out that The Beginning or The End is a 1947 American documentary film. Now I need to find the release year of Hold Back the Night. |
| Subquery | Hold Back The Night |
| Docs | Chunk_15896613 [Hold Back the Night]: Hold Back the Night is a 1956 American war film about the Korean War based on the 1951 novel by Pat Frank, who had been a war correspondent in Korea. The film was directed by Allan Dwan; his third film with John Payne and his third film about the United States Marine Corps, the others being "Abroad with Two Yanks" (1944) and "Sands of Iwo Jima" (1949). The film tells the story in flashbacks of a bottle of scotch carried by a World War II Marine lieutenant and Korean War captain, Sam MacKenzie. His girl Anne |
| Thought | I found out that Hold Back The Night was released in 1956. Now I can compare the release years of both films, The Beginning Or The End was released in 1947, and Hold Back The Night was released in 1956. |
| Subquery | release year of The Beginning of the End (Redundant retrieval) |
| Docs | Chunk_10610413 [The Beginning or the End]: ... |
| Thought | I found out that The Beginning or The End (1947 film) was released in 1947. Now I can compare the release years of both films. |
| Answer | The Beginning or the End |
```

### --- Page 0025 ---

```markdown
# Failure Case 1: Search-R1° vs. LatentRAG°

## Question
Who led the enquiry into the Iraq war? (Ground truth answer: Sir John Chilcot)

### Search-R1°
- **Thought**: I need to find out who led the enquiry into the Iraq war. I'll search for it.
- **[Subquery]**: Who led the enquiry into the Iraq war?
- **[Docs]**: 
  - Chunk_6411881 [Iraq Inquiry]: The Iraq Inquiry (also referred to as the Chilcot Inquiry after its chairman, Sir John Chilcot) was a British public inquiry into the nation’s role in the Iraq War. The inquiry was announced in 2009 by Prime Minister Gordon Brown and published in 2016 with a public statement by Chilcot. On 6 July 2016, Sir John Chilcot announced the report's publication, more than seven years after the inquiry was announced. Usually referred to as the Chilcot report by the news media, the document stated that at the time of the invasion of Iraq in 2003, Saddam Hussein did
- **[Thought]**: I found out that the Iraq inquiry, also referred to as the Chilcot Inquiry, was led by Sir John Chilcot. Now I can provide the answer.
- **[Answer]**: Sir John Chilcot

## LatentRAG°
- **[Thought (dec)]**: I need to find out who led the inquiry into the Iraq War. I'll search for it.
- **[Subquery (dec)]**: Who led the inquiry into the Iraq War?
- **[Docs]**: 
  - Chunk_6411881 [Iraq Inquiry]: The Iraq Inquiry (also referred to as the Chilcot Inquiry after its chairman, Sir John Chilcot) was a British public inquiry into the nation’s role in the Iraq War. The inquiry was announced in 2009 by Prime Minister Gordon Brown and published in 2016 with a public statement by Chilcot. On 6 July 2016, Sir John Chilcot announced the report's publication, more than seven years after the inquiry was announced. Usually referred to as the Chilcot report by the news media, the document stated that at the time of the invasion of Iraq in 2003, Saddam Hussein did
- **[Thought]**: I found out that the Iraq War was led by Sir John Chilworth. Now I can provide the answer.
- **[Answer]**: Sir John Chilcot

---

# Failure Case 2: Search-R1° vs. LatentRAG°

## Question
What was the name of the dog in the novel "Three Men in a Boat"? (Ground truth answer: Montmorency)

### Search-R1°
- **[Thought]**: I need to find the name of the dog in the novel "Three Men in a Boat". I'll search for it.
- **[Subquery]**: What was the name of the dog in the novel "Three Men in a Boat"?
- **[Docs]**: 
  - Chunk_8506768 [Three Men in a Boat]: is how undated it appears to modern readers – the jokes have been praised as fresh and witty. The three men are based on Jerome himself (the narrator Jerome K. Jerome) and two real-life friends, George Wingrave (who would become a senior manager at Barclays Bank) and Carl Hentschel (the founder of a London printing business, called Harris in the book), with whom Jerome often took boating trips. The dog, Montmorency, is entirely fictional but, "as Jerome admits, developed out of that area of inner consciousness which, in all Englishmen, contains an element of the dog". The trip is
- **[Thought]**: I found out that the dog in the novel "Three Men in a Boat" is named Montmorency. Now I can provide the answer.
- **[Answer]**: Montmorency

## LatentRAG°
- **[Thought (dec)]**: I need to find the name of the dog in the novel "Three Men in a Boat". I'll search for it.
- **[Subquery (dec)]**: What was the name of the dog in the novel "Three Men in a Boat"?
- **[Docs]**: 
  - Chunk_8506768 [Three Men in a Boat]: is how undated it appears to modern readers – the jokes have been praised as fresh and witty. The three men are based on Jerome himself (the narrator Jerome K. Jerome) and two real-life friends, George Wingrave (who would become a senior manager at Barclays Bank) and Carl Hentschel (the founder of a London printing business, called Harris in the book), with whom Jerome often took boating trips. The dog, Montmorency, is entirely fictional but, "as Jerome admits, developed out of that area of inner consciousness which, in all Englishmen, contains an element of the dog". The trip is
- **[Thought]**: I found out that the name of the dog in the novel "Three Men in a Boat" is Montmorency. Now I can provide the answer.
- **[Answer]**: Montmorency
```

### --- Page 0026 ---

```markdown
![Latent thought and subquery tokens in LogitLens Case Study 1 on LatentRAG](assets/page_0026_img_1.png)

| Question | The author of The Thing of It Is... | What Nationality? (Ground-truth answer: American) |
|----------|-------------------------------------|--------------------------------------------------|
| LogitLens | I need to know the nationality of the author of The Thing of It Is... If I search for it. | |
| Subquery | The author of The Thing of It Is... | |
| LogitLens | I need to know the nationality of William Goldman. Now I need to add the nationality. | |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |                                                  |
|          |                                     |


# ArXiv 2511.03475

### --- Page 0001 ---

```markdown
# CONTEXTPILOT: FAST LONG-CONTEXT INFERENCE VIA CONTEXT REUSE

**Yinsicheng Jiang**¹ | **Yeqi Huang**¹ | **Liang Cheng**¹ | **Cheng Deng**¹ | **Xuan Sun**¹ | **Luo Mai**¹

## ABSTRACT
AI applications increasingly depend on long-context inference, where LLMs consume substantial context to support stronger reasoning. Common examples include retrieval-augmented generation, agent memory layers, and multi-agent orchestration. As input contexts get longer, prefill latency becomes the main bottleneck. Yet today’s prefill acceleration techniques face a trade-off: they either preserve reasoning quality but deliver little KV-cache reuse, or improve reuse at the cost of degraded reasoning quality.

We present CONTEXTPILOT, a system that accelerates prefill by introducing context reuse as a new mechanism for faster long-context inference. CONTEXTPILOT introduces a context index to identify overlapping context blocks across LLM interactions (e.g., across users and turns). It further proposes context alignment and de-duplication techniques to maximize KV-cache reuse. To preserve reasoning quality under reuse, it introduces succinct context annotations that prevent quality degradation. Finally, CONTEXTPILOT is built around a modular architecture with a clean interface that integrates with existing inference engines. Extensive evaluation shows that CONTEXTPILOT reduces LLM prefill latency by up to 3x compared to state-of-the-art methods while preserving reasoning quality. At longer context lengths, it can even improve reasoning quality. CONTEXTPILOT is open-sourced at: https://github.com/EfficientContext/ContextPilot.

## 1 INTRODUCTION
Long-context inference is now central to many AI applications. Whether through retrieval-augmented generation (RAG) (Lewis et al., 2020), AI memory layers such as MemNN (Chikhar et al., 2025), multi-agent orchestration, or personal AI assistants that interact with external data across conversations (e.g., OpenClaw), workloads routinely feed LLMs tens to hundreds of thousands of tokens. In a typical prefill pipeline, a retriever (e.g., FAISS, Qdrant, ElasticSearch), memory store, or agent tool (e.g., file reading, web search) fetches relevant documents, chunks, or memories for a user query, and an inference engine (e.g., SGLang, vLLM, TensorRT-LLM) consumes them as input context.

We call these discrete units of external context context blocks (CBs). During the prefill phase, the engine computes key-value (KV) caches, which are then reused during decode to generate output tokens sequentially. The key performance metric of prefill is to reduce time-to-first-token (TTFT). To that end, inference engines use a prefix cache.

**Co-Primary Authors:** University of Edinburgh, Correspondence to: Yinsicheng Jiang <x.jiang@ed.ac.uk>, Yeqi Huang <yeqi.huang@ed.ac.uk>, Liang Cheng <L.cheng@ed.ac.uk>, Cheng Deng <cheng.deng@ed.ac.uk>, Xuan Sun <xuan.sun@ed.ac.uk>, Luo Mai <luo.mai@ed.ac.uk>.

Proceedings of the 9th MSLys Conference, Bellevue, WA, USA, 2026. Copyright 2026 by the author(s).
```

### --- Page 0002 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

## 1 Introduction

Long-context workloads often retrieve large sets of documents or memories in varying orders, leaving most KV caches unused. The second category, approximate KV-cache matching, exemplified by CacheBlend (Yao et al., 2025) and PromptCache (Gim et al., 2024), matches KV caches by floating-point similarity rather than exact prefixes. While this increases reuse and shortens TTFT, we observe in evaluation that it can significantly degrade model accuracy.

To reduce TTFT for long-context inputs without sacrificing accuracy, we propose a new approach based on the observation that real-world long-context workloads often exhibit overlapping context blocks, commonly (i) across multiple turns within the same conversation and (ii) among parallel sessions (e.g., prompts or user queries) in domain-specific applications. Leveraging this observation, we identify three opportunities for context reuse with negligible accuracy loss: (1) Aligning context blocks with previously cached prefixes, improving cache-hit ratios; (2) De-duplicating context blocks to avoid recomputation for already cached content; and (3) Adding context annotations to inform the model of original relevance ranking and deduplicated block locations, maintaining accuracy loss.

In this paper, we present CONTEXTPILOT, a system that accelerates prefill by introducing context reuse as a new target for long-context inference. CONTEXTPILOT delivers strong performance across diverse baselines and real-world datasets. Across long-context workloads—including RAG (multi-turn, multi-session, and hybrid), and emerging agent systems (Mem0), emerging multi-agent reasoning paradigms, and real-world agent deployments (OpenClaw)—it reuses contexts to accelerate prefill, outperforming state-of-the-art systems (CacheBlend, LMCache, RadixCache, and RAGCache) by 1.5–3× on MultihopRAG, NarrativeQA, QASPER, and MT-RAG with negligible accuracy loss. As context length grows, CONTEXTPILOT can even improve response quality and answer accuracy, thanks to its novel context-annotation design.

CONTEXTPILOT also scales to very large MoE models (Jiang et al., 2025): on DeepSeek-R1 (671B), it improves prefill throughput by 1.52–1.81× on 16–32 GPUs. Beyond cloud deployments, CONTEXTPILOT reduces prefill latency by 63.6% in a real-world agent pipeline (OpenClaw on a single RTX 5090), and achieves ~2.4× latency reduction on Apple Silicon laptops, demonstrating broad applicability from data center to edge.

In the following sections, we are working towards broader academic and industry adoption with multiple adopters, and have open-sourced CONTEXTPILOT on GitHub. Expect it to be deployed in more challenging multi-user services scenarios (Fu et al., 2024) and can serve as an extensible software foundation for context engineering (Hua et al., 2022; Zhan et al., 2025; Roy et al., 2025), and optimization (Kang et al., 2025; Yu et al., 2025).

## 2 Background and Motivation

### 2.1 Long-context inference systems

Long-context inference systems augment LLMs with external context blocks—retrieved documents, chunks, or memories—to enhance factual grounding and reasoning. We use the term context block throughout this paper to refer to any discrete unit of external context injected into the model, whether a retrieved document, a document chunk, or a memory entry. Two dominant paradigms driving this trend: 

1. **Retrieval-augmented generation (RAG)** (Lewis et al., 2020; Gao et al., 2024) retrieves the top-K most relevant documents per query from an external corpus, serving both online latency-sensitive services (e.g., semantic search, dialogue, deep research (Zilliz, 2022; Guo et al., 2024b)) and offline throughput-oriented pipelines (e.g., large-scale annotation, synthetic data generation (Shen et al., 2025; Zhou et al., 2024; NVIDIA, 2024; Zhang et al., 2025c)).
2. **AI** 
```


### --- Page 0003 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

![Overview of a long-context inference system with prefix caching.](assets/page_0003_img_1.png)

## 2.2 Emergence: growing context lengths

Long-context inference systems face a critical prefill latency bottleneck as modern LLMs demand expanding context windows. This is driven by two reasons: (1) increasing the number of retrieved context blocks to broaden information coverage (Li et al., 2024; Jin et al., 2024; Yue et al., 2025; Laban et al., 2024; Chung et al., 2025), and (2) enriching contextual information by retrieving complete documents or full memory histories and applying context engineering methods (Rajasekaran et al., 2025).

Analysis of our workload data reveals that both approaches deliver significant accuracy gains. Scaling the retrieval parameter ($k$) from lower to higher values enhances accuracy by as much as 20%, while retrieving full documents achieves similar performance improvements, confirmed by recent context engineering studies (Zhang et al., 2025b).

However, expanded context windows (i.e., longer context block inputs) introduce substantial prefill overhead and can even degrade reasoning quality beyond a certain length (Du et al., 2025; Raju et al., 2026). Our trace data shows that LLM inference engines process 20k–10k prefill tokens, leading to 3–10 second latency when executing 32B dense models on a single H100 GPU. For larger models such as Mixture-of-Experts (MoEs), the prefill latency can be even higher. As a result, the prefill becomes the dominant bottleneck, downgrading user experience and preventing long-context applications from being widely deployed.

## 2.3 Designs of existing KV cache reuse methods

To address the growing cost of longer retrieved contexts, existing KV-cache reuse methods exhibit several issues:

### Exact-prefix matching yields low KV-cache reuse.

Existing prefix-caching mechanisms rely heavily on exact token-level matching, e.g., RadixCache (Zheng et al., 2024), or document-level matching, e.g., LMCache (Cheng et al., 2025) and RAGCache (Jin et al., 2024b): even minor variations, such as whitespace differences or slightly reordered tokens and documents, prevent reuse. Our evaluation (Section 7.1) shows that despite substantial overlap in retrieved documents across related queries, cache hit ratios remain abysmally low. For example, for the dataset MultihopRPG with Qwen3-32B, the KV cache hit ratio is only 4.6%, indicating low KV cache reuse. For NarrativeQA with Llama3.7B, the hit ratio is also only 5.5%, leaving most cache unused.

### Approximate KV-cache matching degrades quality.

To improve low cache-hit ratios, recent techniques such as CacheBlend (Yao et al., 2025) adopt approximate KV-caching matching. Instead of exact-prefix matching, they measure similarity in KV values (floating-point vectors) and reuse cached states when the proximity exceeds an empirically decided threshold. However, KV-value similarity is not a reliable indicator of whether cached states can be reused across different contexts and requests. Approximate matching degrades accuracy, with errors compounding over multi-turn interaction. Our evaluations (Section 7.1) show that across multiple models (e.g., Qwen3-32B, Qwen3-4B, Llama3.7-70B) and datasets (e.g., MultihopRPG, NarrativeQA, QASPER), approximate matching can degrade accuracy by 9–11% (dropping from around 60% to approximately 50%), preventing its deployment in many services where high fidelity is necessary.

## 3 DESIGN OVERVIEW

### 3.1 Observation: significant overlap in long context

Our design is motivated by a key observation: real-world long-context workloads exhibit substantial overlap in context blocks across both sessions and conversation turns.
```

### --- Page 0004 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

![Multi-session overlap](assets/page_0004_img_1.png)

## Table 1. Reproducing DeMo ordering study with newer models. Modern LLMs show negligible ordering gaps even on datasets that showed large gaps in the original study.

| Dataset                     | GPFT-3.5 | GPFT-5.1 | Random | DeMo |
|-----------------------------|----------|----------|--------|------|
| SST2 (Socher et al., 2013)  | 93.8     | 92.0     | 93.8   |      |
| SNLI (Bowman et al., 2015)  | 72.6     | 72.6     | 83.2   |      |
| SUBJ (Pang & Lee, 2004)     | 71.3     | 71.6     | 77.5   | 77.0 |
| CR (Hu & Liu, 2004)         | 93.8     | 94.7     | 92.9   |      |
| **Avg**                     | 82.9     | 83.0     | 86.9   | 86.7 |

Specifically, we identify three opportunities that commonly arise in real-world long-context applications:

1. **Aligning context blocks with the prefix cache across sessions boosts KV-cache reuse.** As shown in Figure 2a, the context blocks for the second and third users are aligned to match the first user’s sequence, all three contexts would share an identical prefix, achieving 100% KV-cache reuse.

   Trace-based alignment experiments on MultihopRAG, NarrativeQA, and QASPER confirm this potential. Aligning context block order with prefix-cache structure raises KV-cache hit ratio to 38.9%, 20.2%, and 16.5%, respectively, representing 3–8× higher utilization than the baseline (Section 7.4). This structure allows the model to avoid redundant prefill computation across users.

   Crucially, such alignment incurs minor accuracy loss: only 0.1–0.3% on the same datasets (Section 7.4). As shown in Table 1, our reproduction of the DeMo ordering study (Guo et al., 2022a) with newer models confirms that modern LLMs are substantially less sensitive to input ordering than earlier generations, with near-zero variance on datasets (SST2 (Socher et al., 2013), SNLI (Bowman et al., 2015), SUBJ (Pang & Lee, 2004), CR (Hu & Liu, 2004)) that showed large gaps in the original study. The small residual degradation arises because prefix-optimal alignment can occasionally move important context blocks toward the middle of the list, exposing them to the lost-in-the-middle effect (Liu et al., 2023).

   We later discuss strategies to largely recover this minor loss. Note that context block alignment poses no additional privacy or security risks, sharing the same guarantees as prior KV-cache reuse methods (e.g., RadixCache).

2. **De-duplicating multi-turn overlaps reduces prefill cost.** Figure 2b shows that multi-turn retrievals often return overlapping context blocks across conversation turns. By de-duplicating these blocks and processing only new content together with dialogue history, the amount of contextual data during prefill can be greatly reduced, lowering computation cost. Beyond whole-block duplication, distinct context blocks often share content at a finer granularity—as
```

### --- Page 0005 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

## 3.3 ContextPilot system overview

Prior work treats accuracy (e.g., context graphs, agentic memory) and system performance (exact prefix caching) in isolation. CONTEXTPILOT uniquely bridges this gap through three contributions: (1) a context index with a novel distance function that actively aligns documents with the prefix cache to maximize reuse—converting misses into hits that prior systems cannot achieve; (2) succinct context annotations that allow LLMs to recover semantic accuracy loss; and (3) multi-turn context traversal that identifies and deduplicates previously memorized documents, reducing prefill overhead especially under model context length constraints. This co-design enables simultaneous gains in efficiency and quality that neither approach achieves alone.

![System Overview of ContextPilot](assets/page_0005_img_1.png)

## 4 CONTEXT INDEX

The context index is designed to: (1) efficiently track the inference engine’s prefix-cache status to enable KV-cache reuse; (2) support fast lookup of previously stored KV caches via prefix matching, enabling cross-session context reuse when overlaps exist; and (3) traverse KV caches in multi-turn conversations to detect duplicated context.

### 4.1 Key designs for context index

Figure 4 illustrates the structure of the context index with an example. The left panel shows the index tree, and the right panel shows the corresponding prefix-cache status. The index is organized as a tree whose root represents an empty context. Each node corresponds to a prefix stored in the context index.

![Context Index construction with prefix-cache semantics](assets/page_0005_img_2.png)
```

### --- Page 0006 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

## 6

the prefix cache and contains child nodes that extend this prefix. Every node maintains four attributes: (1) the context containing context block IDs, (2) the search path from the root to this node, (3) an access frequency counter for cache eviction, and (4) the clustering distance at which the node was created.

### Index creation. 
The index is built via hierarchical clustering based on prefix matching. First, we compute pairwise distances between all contexts using their overlap rate. Next, we iteratively merge the closest pair, creating a virtual node whose context is the shortest intersection representing their shared prefix. Finally, each leaf node records its search path from the root, enabling efficient traversal for both cross-session prefix matching and multi-turn duplicate detection.

As shown in Figure 4, the process begins with C1 {2, 1, 3}, C2 {2, 6, 1}, and C3 {4, 1, 0} as leaf nodes. Since C1 and C2 have the smallest distance (sharing {1, 2}), they merge first into a virtual node C4 with context {1, 2}. C3 then merges with C4 to form the root C5 with context {1}. The resulting tree has C1–C3 as leaves storing their search paths from C5, while C4 and C5 serve as virtual nodes representing shared prefixes for cache reuse.

This construction runs in $O(N^2)$ time, where $N$ is the number of contexts, and is fully parallelizable on CPUs and GPUs. It takes $0.8$ s on CPUs and $0.2$ s on GPUs. The space complexity is $O(N \cdot K)$, where $K$ is the average number of blocks per query. Because the index stores only block IDs and metadata rather than full texts, its space overhead is minimal. The complete hierarchical clustering pseudocode is provided in Algorithm 4 (Appendix H).

### Quantifying the overlapping between contexts. 
A key challenge in index construction is quantifying the overlap between contexts. We propose a context distance function that satisfies two requirements: (1) it captures the number of shared documents between contexts, and (2) it accounts for their positional alignment, since retrieval systems rank documents by query relevance.

To illustrate the need for this design, consider four contexts: A {3, 5, 1, 7}, B {2, 6, 3, 5}, C {3, 5, 8, 9}, and D {2, 6, 4, 0}. A naive overlap-only metric assigns identical distances (0.5) to pairs A–B, B–C, and B–D because each shares two documents. However, B and D share {2, 6} at positions 1–2, while A and B share {3, 5} at different positions. Our distance function (Equation 1) assigns a smaller distance to B–D, as their overlaps occur in similar positions, reflecting both overlap magnitude and positional alignment. Such patterns cannot be captured by conventional distance measures like cosine, L1, or L2 similarity, which ignore positional structure. More formally, our distance function is defined as:

$$
d_{ij} = 1 - \frac{|S_{ij}|}{\max(|C_i|, |C_j|)} + \alpha \cdot \sum_{k \in S_{ij}} |p_i(k) - p_j(k)| \cdot \frac{1}{|S_{ij}|}
$$

where $S_{ij}$ denotes the set of shared documents, $p_i(k)$ is the position of document $k$ in context $i$, and $\alpha \in [0.001, 0.01]$ ensures overlap count remains the dominant factor while incorporating positional alignment.

### Index update. 
The context index stays synchronized with the inference engine’s prefix cache through lightweight request ID tracking. Each leaf node is associated with a request ID maintained by the engine. When the engine evicts cached entries, it sends the corresponding request IDs to ContextPilot, which looks up the affected nodes via a request-to-node mapping and removes them. Empty parent nodes are recursively pruned to keep the tree compact. The overall update cost is $O(h)$, where $h$ is the tree height, requiring only a single traversal per eviction.

## 4.2 Key operations with context index

The context index provides two key operations:

### Context search. 
ContextPilot frequently searches for previously stored contexts based on the current one to enhance reuse. The index search algorithm (Algorithm 1, Appendix D) efficiently locates matching contexts by greedily descending from the root, selecting at each level the child with the minimum distance while recording positions to form a search path. The search stops upon reaching a leaf or when internal nodes are equidistant, indicating the longest shared prefix. Updates are localized and efficient: matching an internal node appends the new context at height $O(1)$, while matching a leaf creates a new internal node with their intersection (O(1)). Unlike K-means re-clustering or HNSW graph rebuilding, these updates require no tree restructuring, enabling dynamic index maintenance with minimal overhead.

For example, given context C6 {2, 1, 4}, we search the index in Figure 4. C6 first compares with the root’s child C5 and finds a shared prefix {1}, descending to C5 and recording its position [0]. At C5, C6 charts {1, 2} with C4 but only {1} with C3, so it selects C4 and appends another [0], yielding [0,0]. At C4’s children C1 {1, 2, 3} and C2 {1, 2, 6}, all have equal distance, so the search stops and identifies C4 as the best match with [0,0]. C6 is then inserted into C4’s children list at position 2, forming the final search path [0,0,2].

Search complexity varies with tree height. For contexts with common prefixes, $h = O(\log n) |O(C| \cdot \log n)$ empirically, where $n$ denotes the number of stored contexts. Empirically, search takes approximately $0.068$ ms per request (Appendix D), negligible compared to prefill.
```

### --- Page 0007 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

latency.

Context traversal. In multi-turn conversations, CONTEXTPILOT updates node context lengths by traversing the index using the stored search path. Starting from the root, it sequentially follows indices along the path until reaching the target node, then performs the update. Traversal costs $O(h)$ and is subsumed by the search overhead above.

## 5 CONTEXT ALIGNMENT

The context alignment mechanism aims to: (1) align incoming contexts with the current prefix cache to maximize KV-cache reuse; (2) schedule the aligned contexts to the inference engine with awareness of cache generation and eviction policies to enhance hit ratio; and (3) insert context annotations that recover necessary semantics and maintain accuracy.

### 5.1 Context alignment algorithm

Formally, the context alignment algorithm (Algorithm 2, Appendix H) takes a batch of requests with their context blocks as input, aligns them with the prefix cache based on prefix matches from the context index, and returns aligned contexts with maximized shared prefixes.

As illustrated in Figure 5, we begin with initial contexts $C1 \{2, 1, 3\}, C2 \{2, 6, 1\}, C3 \{4, 1, 0\}$, followed by new contexts $C6 \{2, 1, 4\}, C7 \{5, 7, 8\},$ and $C8 \{1, 2\}$. Initialization contexts inherit prefixes from their parent nodes (C1, C2 from C1 \{1, 2\} from C5 \{1\}), while new contexts search the index (C6 and C8 match C4 and inherit \{1, 2\}). Each context then concatenates its matched prefix with remaining documents in their original order, producing $C1 \rightarrow \{1, 2, 3\}, C2 \rightarrow \{1, 2, 6\}, C6 \rightarrow \{1, 2, 4\},$ and $C8 \rightarrow \{1, 2, 9\}$. Unmatched contexts (e.g., C7) remain unchanged and forstandable branches. This strategy ensures overlapping contexts share common prefixes while preserving the ranking of non-shared documents.

The algorithm is invoked whenever CONTEXTPILOT processes a new request. It runs in $O(|C| \cdot \log n)$ time, where $n$ is the number of stored contexts, taking approximately 0.047 ms per request (Appendix D.3)—negligible compared to prefill.

### 5.2 Scheduling requests with aligned contexts

After aligning contexts, CONTEXTPILOT must schedule their execution to match the inference engine’s KV-cache generation and eviction policies; otherwise, cache reuse becomes ineffective. We therefore design a scheduling algorithm that: (1) reuses the search paths obtained during context alignment to avoid redundant tree lookups; (2) groups contexts by the first element of their search path, naturally separating cache regions; and (3) sorts contexts within each group by path length in descending order, ensuring longer prefix matches execute before shorter ones.

Figure 6 illustrates this process. In the baseline order C6, C3, C7, C8, limited cache capacity allows only one context: C6 caches \{1, 2, 4\}, but C3 reuses only \{1\} and evicts \{2, 4\}. C7 causes a full miss, caching \{5, 7, 8\} and evicting all previous entries, which then forces another miss for C8 despite its shared prefix \{1, 2\} with C6. This inefficiency arises because contexts with shared prefixes are not executed consecutively.

Our scheduler rearranges the execution to C6, C8, C3, C7, grouping prefix-sharing contexts together. C6 first caches \{1, 2, 4\}, then C8 immediately reuses \{1, 2\} before eviction. C3 and C7 run afterward without disrupting this reuse, maximizing cache hit ratio.

Our scheduler performs $O(N)$ grouping by root-prefix path and $O(N \log N)$ in-group sorting over $N$ contexts, with negligible real-time overhead. In contrast, existing indexing methods such as RAGCache and SGLang’s LPM use a global prefix selection that rescans a radix tree with $M$ nodes at each decision point, yielding $O(N \log M) + O(N \log N)$ overall cache usage. By draining groups sequentially, our method avoids re-scanning KV budgets, and keeps complexity independent of $M$. The full scheduling pseudocode is given in Algorithm 5 (Appendix H).

### 5.3 Context annotation for context alignment

Why aligning is safe. As shown in Section 3.2, our reproduction of the DEMo study (Guo et al., 2022a) confirms that modern LLMs are substantially less sensitive to input ordering (Table 1), explaining why aligning for cache efficiency introduces only minor accuracy perturbation and making lightweight correction mechanisms sufficient.

Why annotations still help. Despite the reduced sensitivity, aligning can still cause minor accuracy perturbation on some datasets (e.g., -1.1% on QASPER). Annotations mitigate this by reducing the model’s reliance on positional signals to infer relevance. On multi-hop tasks where chaining evidence across context blocks benefits from explicit guidance, annotations not only recover lost accuracy but actively improve it beyond the no-alignment baseline (e.g., +4.0% F1 on MultihopRAG with Qwen3-32B; see Appendix D.2). Gains are consistent across model scales: Qwen3-4B gains +1.4% on MultihopRAG and +1.3% on NarrativeQA, while Qwen3-32B gains +4.0% and +1.2% respectively. Attention map analysis (Appendix B) confirms that annotations reshape internal attention, aligning it with semantic rather than positional priority.
```


### --- Page 0008 ---

```markdown
![Example for aligning context with prefix cache](assets/page_0008_img_1.png)

![Example of scheduling requests with aligned contexts](assets/page_0008_img_2.png)

## Annotation mechanism

We provide the LLM with succinct annotations indicating the original relevance ranking of context blocks. Aligning contexts alters this ranking, which enhances document relevance critical for answers quality. Consider context C6, where the retriever returns documents in order {1, 2, 4}. The baseline prompt is:

```
[system prompt] → [CB.2] → [CB.1] → [CB.4] → [question]
```

After aligning to {1, 2, 4} for cache efficiency, we append an order annotation before the question:

```
[system prompt] → [CB.1] → [CB.2] → [CB.4] → [order annotation] → [question]
```

The annotation explicitly specifies the original relevance priority:

> “Please read the content in the following priority order: [CB.2] > [CB.1] > [CB.4] and answer the question.”

This short instruction adds negligible token overhead during prefill yet effectively preserves the model’s ability to attend to documents by their original relevance ranking. As a result, CONTEXTPILOT achieves aggressive cache optimization with minor accuracy perturbation (≤1% on most datasets), and often improved accuracy on multi-hop reasoning tasks.

## 6 CONTEXT DEDUPLICATION

The context-deduplication mechanism has two goals: (1) eliminate redundant content—both entire context blocks repeated across turns and shared content across distinct text blocks—whose KV caches are already stored, thereby minimizing redundant prefill computation; and (2) provide annotations informing the LLM which content has been duplicated and where the corresponding information resides in the earlier context.

Algorithm 3 (Appendix H) formalizes the full de-duplication procedure, which operates at two levels. The algorithm runs in $O(|C|)$ time, taking approximately 0.6 ms per request (Appendix D.3)—negligible compared to prefill.

### Context-block-level de-duplication

The context index maintains a per-conversation record of all context blocks processed in prior turns, provided that the first turn's context is aligned and inserted into the index (Section 5). Given a new context, the algorithm checks the previously indexed blocks, identifies exact matches, and generates a location annotation for each duplicate. The current turn's blocks are then registered for future comparisons.

We illustrate with an example: consider a session where user $U$ initially retrieves context {1, 2, 4} in the first turn. During context alignment (Section 5), these blocks are inserted into the context index and recorded in C6's conversation history. In the second turn, a new query yields {1, 5, 2}. The algorithm queries the index's conversation record, identifies {1, 2} as already cached from the first turn, and replaces them with local annotations, leaving only the novel block {5} to be fully processed.

### Content-level de-duplication

Content-block-level de-duplication handles exact matches across turns, but distinct context blocks often share overlapping content—as illustrated in Figure 2b, where information about Kennedy’s death date appears across multiple context blocks. Inspired by content-defined chunking in deduplication storage systems (Muthitacharoen et al., 2001), we split each novel context block into variable-length sub-blocks at boundaries where $HASH(\ell)$ mod $M = 0$ for each turn $\ell$. Unlike fixed-size chunking, where a single insertion or deletion shifts all subsequent boundaries and prevents hash matches, content-defined boundaries are determined solely by local content, ensuring that identical text always produces the same sub-blocks regardless of its offset within different contexts.
```

### --- Page 0009 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

context blocks.

Each sub-block is hashed. Any sub-block matching a hash from a different context block is replaced with a location annotation pointing to the first occurrence. After deduplication, the context index is updated to reflect these changes for the request.

## Context annotation for de-duplicated context blocks.

Simply removing duplicates can degrade answer quality. To maintain quality, we insert location annotations (e.g., “Please refer to [CB-1] in the previous conversation”) that direct the LLM to corresponding context blocks in the conversation history, guiding the LLM to prior context without repeating prefill. For C6’s second turn, the prompt changes from:

```
[first turn context] → [first turn Q&A] → [CB-1] → [CB-5] → [CB-2] → [second turn question]
```

to:

```
[first turn context] → [first turn Q&A] → [annotation_1] → [CB-5] → [annotation_2] → [second turn question]
```

## 7 Evaluation

Our evaluation of CONTEXTPILOT shows: (1) CONTEXTPILOT improves prefill throughput by up to 4.0% over numerous state-of-the-art systems and methods across multi-turn, multi-session, and hybrid RAG workloads, (2) it outperforms strong baselines by 1.5-3× in throughput and 1.9-3.7× in accuracy in emerging academic AI applications, including real-world agent deployments, (3) each component (context alignment, de-duplication, and annotation) yields clear gains and robustness with negligible overhead, (4) benefits scale with longer contexts and larger retrieval sizes, and (5) gains extend to edge devices, achieving ~2.4× latency reduction on consumer hardware without GPU servers.

### Evaluation setup.

Our CONTEXTPILOT implementation supports SGLang 0.4.6 and vLLM 0.10.0, requiring only request ID tracking in each engine’s prefix cache without affecting existing functionality, making the changes easy to upstream and merge.

We compare CONTEXTPILOT against the following baselines: (i) LMCACHE (version 0.3), representing the state of the art in prompt caching; (ii) CACHEBLEND, the state of the art in KV-cache matching, integrated with LMCACHE; and (iii) RADIXCACHE, based on SGLang’s implementation using a Longest-Prefix-Match scheduling policy.

We omit additional baselines that achieve comparable performance to those above; (iv) HiCACHE (Xie et al., 2025) extends RadixCache by expanding prefix caches to lower-tier memory; since it directly builds on RadixCache, we compare against RadixCache instead. (v) RAGCACHE adopts a similar radix-tree structure at document granularity and shows comparable performance to RadixCache. It is not open-sourced and thus excluded from our evaluation.

Our evaluation is conducted on two GPU clusters: (1) a 16× H100 GPU cluster and (2) a 12× A6000 GPU cluster. For all baselines, we tune system parameters for optimal performance and accuracy, aligning our configurations with the best results reported in their respective papers. We set $\alpha = 0.001$ for the distance metric in Equation 1 across all experiments.

### 7.1 Performance in Retrieval-Augmented Generation

We evaluate on four RAG datasets: QASPER (Dasigi et al., 2021), MultihopRAG (Tang & Yang, 2024), NarrativeQA (Kociský et al., 2018), and MT-RAG (Katsai et al., 2025). For QASPER, MultihopRAG, and NarrativeQA, we use a chunk size of 1024 following (Bhat et al., 2025), while MT-RAG performs document-level retrieval without chunking. We use $g=\text{Qwen-7B-Instruct}$ as the embedding model, FAISS for similarity search on MultihopRAG and NarrativeQA, and BM25 for QASPER and MT-RAG. This setup demonstrates CONTEXTPILOT’s effectiveness across diverse retrieval paradigms.

#### Multi-session RAG.

We evaluate multi-session RAG on QASPER, MultihopRAG, and NarrativeQA using three models—Qwen-3B-Instruct-2507, Qwen-32B, and Llama3-7B-Instruct—on H100 GPUs with top-k=15.

| Model                | F1 Score | Prefill Throughput |
|----------------------|----------|--------------------|
| CONTEXTPILOT         | 0.3×     | 2.05×              |
| LMCACHE              | 0.2×     | 2.13×              |
| RadixCache           | 1.3–1.6× | Gains on NarrativeQA and QASPER, by aligning contexts with the prefix cache to maximize overlap. In contrast, LMCACHE and RadixCache depend on exact prefix matching, causing recomputation even for overlapping content, while LMCACHE also incurs high CPU offloading costs for long contexts. CONTEXTPILOT also maintains or improves accuracy via order annotations (e.g., 6.04–6.44 on MultihopRAG with Qwen-32B), while CacheBlend degrades sharply (F1 drops to 11.3 on NarrativeQA with Qwen-32B).
```


### --- Page 0010 ---

```markdown
| Dataset         | Model                | F1 (%) | Prefill Throughput | F1 (%) | Prefill Throughput | F1 (%) | Prefill Throughput | F1 (%) | Prefill Throughput |
|------------------|---------------------|--------|--------------------|--------|--------------------|--------|--------------------|--------|--------------------|
|                  |                     | LMCache | CacheBlend         | Radix Cache | ContextPilot (Ours) |
| MultihopRAG      | Qwen3-4B-Instruct-2507 | 35.2   | 34710.9            | 34.8   | 35043.5            | 35.2   | 58901.1            | 36.6   | 16709.5            |
|                  | Qwen3-2B            | 60.4   | 14706.8            | 51.1   | 36126.6            | 64.4   | 17626.6            | 64.4   | 3269.1             |
|                  | Llama3-3.70B-Instruct | 62.9   | 11596.4            | 54.9   | 14134.2            | 62.9   | 14771.1            | 62.9   | 30046.7            |
| NarrativeRAG     | Qwen3-4B-Instruct-2507 | 16.0   | 39276.4            | 11.3   | 42189.5            | 16.0   | 34942.4            | 17.3   | 57613.1            |
|                  | Qwen3-2B            | 28.4   | 15514.0            | 19.8   | 16913.2            | 28.4   | 15598.8            | 29.6   | 22780.4            |
|                  | Llama3-3.70B-Instruct | 37.8   | 12575.7            | 31.3   | 13710.5            | 37.8   | 12644.9            | 38.4   | 18468.8            |
| QASPER           | Qwen3-2B            | 27.9   | 2939.2             | 21.9   | 36725.7            | 27.9   | 33034.6            | 26.8   | 64619.9            |
|                  | Llama3-3.70B-Instruct | 36.0   | 15584.4            | 29.3   | 20289.9            | 36.0   | 17520.3            | 34.9   | 24734.3            |
|                  | Qwen3-2B            | 27.9   | 14289.7            | 33.8   | 15307.3            | 38.3   | 19601.0            |

Qwen3-4B due to approximate KV matching disrupting coherence.

### Effectiveness with even larger MoE models.
We further evaluate DeepSeek-RI (671B) on a GPU cluster with 32 H2O GPUs, provided by a potential industry adopter. On MultihopRAG, CONTEXTPILOT increases the cache hit ratio from 5% to 60%; on NarrativeQA, it raises the hit ratio from 6% to 38%. These improvements translate into 18.1x and 1.52x higher prefill throughput on 16xH2O, respectively. Scaling to 32xH2O yields similar speedups, confirming that our approach generalizes to larger reasoning models and multi-node deployments. We provide a more detailed analysis in the appendix.

### Multi-turn RAG.
We evaluate multi-turn RAG on the MTRAG dataset using Qwen3-4B-Instruct-2507, Llama3-1B-Instruct, and Qwen3-30B-A3B-Thinking-2507 on a single H100 GPU. Models are evaluated with varying windows size to handle the growing conversation history. Answer accuracy is measured via the LLM-as-a-judge method from RABDM (Kuo et al., 2025) with GPT-5, as recommended by MT-RAG.

Table 3 reports accuracy and time-to-first-token (TTFT). CONTEXTPILOT cuts TTFT by removing redundant document processing across turns through context de-duplication. It achieves 3.45x, 3.35x, and 3.90x speedups over LMCache on Qwen3-4B, Llama3-1B, and Qwen3-30B, respectively, and up to 2.00x over RadixCache and 1.55x over CacheBlend. CONTEXTPILOT also preserves accuracy via location annotations that direct models to previously seen documents (e.g., 62.56%→64.27% on Qwen3-4B), while CacheBlend drops to 50.3% due to approximate KV matching disrupting multi-turn coherence.

### Multi-session, multi-turn RAG.
We evaluate the combined multi-session and multi-turn scenario under real-world deployment using Qwen3-4B-Instruct-2507 on H100 GPUs, varying concurrency from 2 to 32 sessions.

Table 5 shows that CONTEXTPILOT achieves the lowest TTFT at all concurrency levels by aligning contexts with the prefix cache. At 2 sessions, it delivers 3.38x, 1.92x, and 1.67x speedups over LMCache, RadixCache, and CacheBlend, respectively; at 32 sessions, the gains remain substantial at 2.61x, 1.49x, and 1.20x. These improvements confirm that context retention enhances throughput over concurrency scales.

### 7.2 Performance in Agentic Applications
We study three representative agentic work scenarios, increasingly common in production—multi-agent reasoning, multi-role, and real-world deployment—along with repeatedly surfaced long contexts across turns and sessions, which are critical for context reuse.

### Multi-agent reasoning.
We evaluate CONTEXTPILOT with Chain-of-Agent (CoA) (Zhang et al., 2022), where worker agents handle document segments and a manager aggregates results. CONTEXTPILOT enhances CoA through agent-aware routing: in multi-session settings, recurring documents are routed to the agent that previously processed them for KV-cache reuse; in multi-turn conversions, re-peated documents are duplicated with location annotations directing agents to prior context. We deploy CoA configurations on MultihopRAG, each with its 15 agents using Llama3-1B, Llama3-2B, or Qwen3-4B-Instruct ($k=15$). With Qwen3-4B, accuracy increases from 48.3% to 50.2% and throughput by 1.8x; with Llama3-1B, accuracy rises from 50.7% to 54.4% with a 2.1x speedup.

### Agentic memory systems.
We evaluate CONTEXTPILOT with MemO (Chhikara et al., 2025), a popular AI memory system that repeatedly retrieves user-specific memories as context blocks, creating substantial cross-request overlap. Using Qwen3-4B on LoCoMo (Maharana et al., 2024) with GPT-4.1 as judge, CONTEXTPILOT indexes and aligns fetched memories with the prefix cache in online as $k=100$, it reduces TTFT from 1.01 to 0.055 (18.3x speedup) with a minor accuracy trade-off (0.437→0.420). At $k=20$, although LoCoMo memory conversions are relatively short (~26k tokens across all turns on average), CONTEXTPILOT still improves both TTFT (0.038→0.031s, ...
```

### --- Page 0011 ---

```markdown
| Metric                | Method                     | Avg   | P99   |
|-----------------------|---------------------------|-------|-------|
| Prompt Tokens         | Baseline                  | 34,601| 43,981|
|                       | + ContextPilot            | 26,218| 35,385|
|                       |                           |       |       |
| Prefill Latency       | Baseline                  | 7.2   | 5.8   |
|                       | + ContextPilot            | 6.7   | 2.2   |
|                       |                           |       |       |
| Wall Time             | Baseline                  | 26.1  | 68.5  |
|                       | + ContextPilot            | 21.7  | 17.8  |
|                       |                           |       |       |
| Δ                     |                           | -20.7%| -12.4%|

![Performance metrics across different tasks](assets/page_0011_img_1.png)

### 7.3 Deployment on Edge Devices

Real-world agent deployment. To evaluate CONTEXTPILOT in an end-to-end agent pipeline, we integrate it with OpenClaw served via SGLang on a single RTX 3090. The OpenClaw agent issues requests through the CONTEXTPILOT proxy, which aligns prompts with the prediction cache, deduplicating overlapping context both across turns and within context blocks, and forwards optimized prompts to the inference engine (Figure 14 in Appendix E). We evaluate on the claw-tasks benchmark using Qwen-3B-Inst-2507, covering two workload types: document analysis (60 tasks, 22 documents, ~250 turns) and coding (10 tasks). Document analysis is prefill-heavy (average ~45K prompt tokens, only 984 decode tokens), while coding tasks generate longer outputs, making prefill a smaller fraction of wall time.

### 7.4 Performance breakdown, overhead and robustness

Contribution of alignment and scheduling. We analyze how aligning and scheduling each contribute on Multihybrid rRAG (k=15) across SGLang and vLLM on H100 GPUs. As shown in Figure 7, each component adds incremental cache hit ratio gains. For SGLang with Qwen3-32B, hit ratio rises from 8.49% to 20.56% (+ aligning) to 33.97% (+ scheduling)—a 4% improvement. vLLM with Llama3-70B follows a similar trend: 10.7% → 30.8% → 43.2%, directly translating to reduced prefill computation.

Performance with long-running workloads. Time-series analysis (Appendix D.1) confirms that CONTEXTPILOT's gains are sustained: it maintains ~34% cache hit ratio throughout the entire workload, with both alignment and scheduling contributing to the improvement.
```

### --- Page 0012 ---

```markdown
![Performance breakdown of key components.](assets/page_0012_img_1.png)

![Prefill throughput under different top k values.](assets/page_0012_img_2.png)

## 8 RELATED WORKS

**RAG system optimization.** System-level approaches such as METIS (Ray et al., 2022) and Chameleon (Jiang et al., 2024) optimize workflow and hardware efficiency, jointly tuning retrieval settings or using heterogeneous accelerators (He et al., 2025) to reduce latency and boost throughput. CONTEXTPILOT complements them by improving KV-cache reuse.

**Reranking in retrieval systems.** Rerankers refine retrieval results via learned ranking models (Adeyemi et al., 2024; Li et al., 2023; Zhang et al., 2025d; HyperRAG (An et al., 2025) further enables KV reuse at the reranking stage. CONTEXTPILOT instead operates downstream, requiring reranked document IDs with the prefix cache while preserving relevance through order annotations.

**Fine-tuning with positional re-encoding.** Methods like BlockAttention, KVLink, and TurboRAG (Ma et al., 2025; Yang et al., 2025a; Lu & Tang, 2024) fine-tune models to reuse KV caches via position re-encoding and pre-stored states. They improve efficiency but require heavy training and large cache storage. CONTEXTPILOT is training-free and can be applied to any model as an extra optimization framework.

**Faster KV-cache compute.** CacheBlend (Yao et al., 20225) and related works (Liu et al., 2024; Agarwal et al., 2025; Yang et al., 2025b; Deng et al., 2025; Liu et al., 2025a; Du et al., 2026; Pan et al., 2025; Tang et al., 2024; Wu et al., 2025; Chen et al., 2025; Zhang et al., 2025a) optimize KV-cache computing through compression, decoding, parallel encoding, cache sharing, or FLOP-aware admission and eviction policies. CONTEXTPILOT complements these by operating at the context level, aligning and de-duplicating inputs to maximize reuse, and can be combined with them for further gains.

## 9 CONCLUSION

We presented CONTEXTPILOT, a context-reuse system that accelerates long-context prefill with negligible accuracy loss. CONTEXTPILOT uniformly represents retrieval inputs as context blocks and applies a common set of mechanisms—context indexing, alignment, de-duplication, and succinct annotations to boost prefix KV cache reuse across diverse workloads. Its modular design enables new system and algorithmic research on context engineering, management, and optimization for long-context AI. Looking ahead, we envision CONTEXTPILOT as a co-optimization framework that jointly decides what context to feed and how to serve it, maximizing both inference efficiency and output quality.
```

### --- Page 0013 ---

```markdown
# Acknowledgements

We sincerely thank our shepherd and the MLSys reviewers for insightful feedback that significantly improved this manuscript. We also thank Minjie Wang (formerly Amazon AI Labs, now The University of Hong Kong (Shayku X-Lab)) for valuable early-stage discussions, and Ryan Tsui (The University of Edinburgh) for contributions to the implementation. We further thank the Tencent WeChat (Weixin) Group for access to H2O GPU clusters for large-scale MoE evaluation and for insightful feedback. We also thank the UK Advanced Research and Invention Agency (ARIA) for funding this project. Finally, we thank the School of Informatics, University of Edinburgh, UK Isambard-AI and the Edinburgh International Data Facility (EIDF) for providing GPU resources that supported this research.

# References

| Author(s) | Title | Source | Year | URL |
|-----------|-------|--------|------|-----|
| AbouEllaa, S., Zabihitari, P., Ibrahim, N., Afshar, M., and Khaef, R. | Exploring rag solutions to reduce hallucinations in llms | 2025 IEEE International systems Conference (SysCon), | 2025 | [doi:10.1109/SysCon64521.2025.1104810](https://doi.org/10.1109/SysCon64521.2025.1104810) |
| Adeyemi, M., Oladipo, A., Pradeep, R., and Lin, J. | Zero-shot cross-lingual parsing with large language models | In Ku, L., Marit, S., and Srikanth, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), | 2024 | [doi:10.18653/v1/2024.acl-short.59](https://aclanthology.org/2024.acl-short.59/) |
| Agarwal, S., Sundaresan, S., Mitra, S., Mahapatra, D., Gupta, A., Sharma, R., Kapp, N. J., Yu, T., and Saini, S. | Cachecraft: Managing chunk-caches for efficient retrieval-augmented generation | Proc. ACM Manag. Data, | 2023 | [doi:10.1145/3725273](https://doi.org/10.1145/3725273) |
| Alzubi, S., Brooks, C., Chiniya, P., Contente, E., von Gerlach, C., Irwin, L., Jiang, Y., Kaz, A., Nguyen, W., Oh, S., Tyagi, H., and Wiswanath, P. | Open deep search: Democratic reasoning with search-powered agents | 2025 | [arXiv:2504.20201](https://arxiv.org/abs/2504.20201) |
| An, Y., Cheng, V., Park, S. J., and Jiang, J. | Hypergraph: Enhancing quality-efficiency tradeoffs in retrieval-augmented generation with reranker-k-explore | 2025 | [arXiv:2504.09221](https://arxiv.org/abs/2504.09221) |
| Ayala, O. and Bechard, R. | Reducing hallucination in structured outputs via retrieval-augmented generation | In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), | 2024 | [doi:10.18653/v1/2024.naacl-industry.19](http://dx.doi.org/10.18653/v1/2024.naacl-industry.19) |
| Bhat, S. R., Rudat, M., Spiekermann, J., and Flores-Herr, N. | Rethinking chunk size for long-document retrieval: A multi-dataset analysis | 2025 | [arXiv:abs/2505.21700](https://arxiv.org/abs/2505.21700) |
| Bowman, S. R., Angeli, G., Potts, C., and Manning, C. D. | A large annotated corpus for learning natural language inference. | In EMNLP, | 2015 | |
| Chang, E. Y. and Geng, L. | Task management, validation, and transaction guarantees for multi-agent planning. | arXiv preprint arXiv:2503.19515, | 2025 | |
| Chen, G., Feng, Q., Ni, J., Li, X., and Shieh, M. Q. | RAPID: Long-context inference with retrieval-augmented speculative decoding. | In International Conference on Machine Learning (ICML), | 2021 | PMLR, 2025 |
| Cheng, Y., Liu, Y., Yao, J., An, Y., Chen, X., Feng, K., Huang, Y., Shen, S., Du, K., and Jiang, J. | Lmcache: An efficient way to cache layer for long-context inference | 2025 | [arXiv:abs/2510.09665](https://arxiv.org/abs/2510.09665) |
| Chikhiara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. | Memo: Building production-ready ai agents with scalable long-term memory tools. | 2025 | [arXiv:abs/2504.19413](https://arxiv.org/abs/2504.19413) |
| Chung, Y., Kakkar, G. T., Gan, Y., Milne, B., and Ozcan, F. | Is long context all you need? leveraging LLM's extended context for NLS2QL. | In Proceedings of the VLDH ebdowm, | 2025 | [arXiv:abs/2501.12372](https://arxiv.org/abs/2501.12372) |
| Daisgi, P., Lo, K., Beltagy, I., Cohen, A., Smith, N. A., and Gardner, M. | A dataset of information-seeking questions and answers anchored in research papers, | 2021 | [arXiv:abs/2105.03011](https://arxiv.org/abs/2105.03011) |
| Deng, Y., You, Z., Xiang, L., Li, Q., Yuan, P., Hong, Z., Zheng, Y., Li, W., Li, R., Liu, H., Mouratidis, K., Yiu, M. L., Li, H., Shen, Q., Mao, R., and Tang, B. | Alaydb: The data foundation for efficient and effective long-context lm inference. | In Companion of the 2025 International Conference on Management of Data, | 2025 | [doi:10.1145/3722212.3724482](https://doi.org/10.1145/3722212.3724482) |
```

### --- Page 0014 ---

```markdown
| **Reference**                                                                                                                                                                                                                     | **URL**                                                                                          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Du, D., Cao, S., Cheng, J., Mai, L., Cao, T., and Yang, M. Bitdecoding: Unlocking tensor cores for long-context llms with low-bit kv cache. In 2026 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2026. | [Link](https://www.usenix.org/conference/osdi25/presentation/du)                                 |
| Du, Y., Tian, M., Ronanki, S., Rongali, S., Bodapati, S., Galstyan, A., Wells, A., Schwartz, R., Huerta, E. A., and Peng, H. Context length alone hurts LLM performance despite perfect retrieval. In Findings of the Association for Computational Linguistics: EMNLP, 2025. | [Link](https://arxiv.org/abs/2510.05381)                                                        |
| Fu, Y., Xue, L., Huang, Y., Brabete, A., Ustiugov, D., Patel, Y., and Mai, L. Serverlessllm: Low-latency servers inference for large language models. In Gavrilovska, A. and Terry, D. B. (eds.), 18th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2024, 195-135. USENIX Association, 2024. | [Link](https://www.usenix.org/conference/osdi24/presentation/fu)                                 |
| Gao, X., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, R., Wang, H., and Wang, H. Retrieval-augmented generation for large language models: A survey, 2024.                                                                 | [Link](https://arxiv.org/abs/2312.10997)                                                        |
| Gim, I., Chen, G., seob Lee, S., Sarda, N., Khandewal, A., and Zhong, L. Prompt cache: Modular attention reuse for low-latency inference. URL.                                                                                   | [Link](https://arxiv.org/abs/2311.04934)                                                        |
| Guo, Q., Wang, L., Wang, Y., Ye, W., and Zhang, S. What makes a good order of examples in in-context learning. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 14892–14904, Bangkok, Thailand, August 2024. Association for Computational Linguistics. | [Link](https://doi.org/10.18653/v1/2024.findings-acl.884)                                      |
| Guo, S., Deng, C., Wen, Y., Chen, H., Chang, Y., and Wang, J. Ds-agent: Automated data science by empowering large language models with case-based reasoning. arXiv preprint arXiv:2402.17453, 2026.                             | [Link](https://arxiv.org/abs/2601.20975)                                                        |
| He, C., Huang, Y., Mu, P., Miao, Z., Xue, J., Ma, L., Yang, F., and Mai, L. Waferlm: Large language model inference at wafer scale. In Zhou, L. and Zhou, Y. (eds.),                                                                 |                                                                                                  |
| Hu, M. and Liu, B. Mining and summarizing customer reviews. In KDD, 2004.                                                                                                                                                     |                                                                                                  |
| Hu, Y., Liu, S., Yue, Y., Zhang, G., Liu, B., Zhu, F., Lin, J., Guo, H., Dou, S., Xi, Z., et al. Memory in the age of AI agents, 2025.                                                                                          | [Link](https://arxiv.org/abs/2512.13564)                                                        |
| Hua, Q., Ye, L., Fu, D., Xiao, Y., Cai, X., Wu, Y., Lin, J., Wang, J., and Liu, P. Context engineering. arXiv preprint arXiv:2510.26493, 2025.                                                                                 |                                                                                                  |
| Jiang, W., Zeller, W., Waleffe, R., Hoefler, T., and Alonso, G. Chameleon: a heterogeneous and disaggregated executor system for retrieval-augmented language models. Proc. VLDB Endow., 18(1):42–52, 2024.                     |                                                                                                  |
| Jiang, F., Yu, Y., Huang, Y., Nie, P., Liu, Z., Xue, L., He, C., Sit, M.-K., Xue, J., Dong, L., Miao, Z., Du, D., Xu, T., Zou, K., Ponti, E., and Lee, M.-C. 2024: Benchmarking cost, accuracy and performance of spark mixture-of-experts systems. In Advances in Neural Information Processing Systems (NeurIPS), 2025. | [Link](https://arxiv.org/abs/2410.05983)                                                        |
| Jin, C., Zhang, Z., Jiang, X., Liu, F., Liu, X., Liu, X., and Jin, X. Ragecache: Efficient knowledge caching for retrieval-augmented generation, 2024.                                                                             | [Link](https://arxiv.org/abs/2404.12457)                                                        |
| Kang, M., Chen, W.-N., Han, D., Inan, H. A., Wutschitz, L., Chen, Y., Sim, R., and Rajmohan, S. Acorn: Optimizing context compression for long-horizon llm agents. arXiv preprint arXiv:2510.00615, 2025.                      |                                                                                                  |
| Katsis, Y., Rosenthal, S., Fadnis, K., Gunasekara, C., Lee, Y.-S. P., Shah, V., Zhu, H., Contractor, D., and Danilevsky, M. Mtrag: A multi-turn conversational benchmark for evaluating retrieval-augmented generation systems, 2025. | [Link](https://arxiv.org/abs/2501.03468)                                                        |
| Kirsch, L., Harrison, J., Sohl-Dickstein, J., and Metz, L. General-purpose in-context learning by meta-learning transformers. arXiv preprint arXiv:2212.04458, 2022.                                                          |                                                                                                  |
```

### --- Page 0015 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

Kočiský, T., Schwarz, J., Blunson, P., Dyer, C., Hermann, K. M., Melis, G., and Grefenstette, E. The narratveqa reading comprehension challenge, 2017. URL: https://arxiv.org/abs/1712.07040.

Kuo, T.-L., Liao, F.-T., Hsieh, M.-W., Chang, F.-C., Hsu, P.-C., and Shiu, D.-S. Rad-bench: Evaluating large language models capabilities in retrieval augmented dialogues, 2025. URL: https://arxiv.org/abs/2409.12558.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention, 2023. URL: https://arxiv.org/abs/2309.06180.

Laban, P., Fabbri, A. R., Xiong, C., and Wu, C.-S. Summary of a haystack: A challenge to long-context LLMs and RAG systems. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 9885–9903, 2024.

Lewis, P., Perez, E., Piktus, F., Perrotta, F., Karpujkin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., et al. Retrieval-augmented generation for knowledge-intensive NLP. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Li, C., Liu, Z., Xiao, S., and Shao, Y. Making large language models accessible: foundation for efficient retrieval, 2023.

Li, G., Zhou, X., Zhang, X., and Zhang, H. Database perspective on LLM inference systems. Proceedings of the VLDB Endowment, 18, 2025. URL: https://www.vldb.org/pvldb/vol15/p504-1.pdf.

Li, Z., Li, C., Zhang, M., Mei, Q., and Benderys, M. Retrieval augmented generation or long-context LLMs? a comprehensive study and hybrid approach. arXiv preprint arXiv:2407.16833, 2024.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts, 2023. URL: https://arxiv.org/abs/2307.03172.

Liu, Y., Li, H., Cheng, Y., Ray, S., Huang, Y., Zhang, Q., Du, K., Yao, J., Lu, S., Ananthanarayanan, G., Maire, M., Hoffmann, H., Holtzman, A., and Jiang, J. Cacheken: K cache compression and streaming for fast large language model serving, 2024. URL: https://arxiv.org/abs/2310.07240.

Liu, Y., Huang, Y., Yao, J., Feng, S., Gu, Z., Du, K., Li, H., Cheng, Y., Jiang, J., Lu, S., Musavathi, M., and Choukse, E. Droidspeak: K cache sharing for cross-llm communication and multi-llm serving, 2025. URL: https://arxiv.org/abs/2411.02820.

Liu, Y., Si, C., Narasimhan, K. R., and Yao, S. Contextual experience replay for self-improvement of language agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14179–14198, 2025b.

Lu, S. and Tang, Y. Turborag: Accelerating retrieval-augmented generation with precomputed k chunks for each chunked text. arXiv preprint arXiv:2410.07590, 2024.

Lumer, E., Nizar, F., Jangiti, A., Frank, K., Gulati, A., Phadate, M., and Subbiah, V. K. Don’t break the cache: An evaluation of prompt caching for long-horizon agentic tasks, 2026. URL: https://arxiv.org/abs/2601.06007.

Ma, D., Wang, Y., and Tian, L. Block-attention for efficient prefiling, 2025. URL: https://arxiv.org/abs/2409.13535.

Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. Evaluating very long-term conversational memory of LLM agents. In ACL, 2024.

Muthitacharoen, A., Chen, B., and Mazières, D. A low-bandwidth network for llm inference. In SOSP, 2001.

NVIDIA. How nvidia uses nlp to generate synthetic data for ml improvement. https://developer.nvidia.com/blog, 2024.

Pan, R., Wang, Z., Jia, Z., Karakus, C., Zancotto, L., Dao, T., Wang, Y., and Netravali, R. Prefilng caching for the era of hybrid LLMs. In Proceedings of Machine Learning and Systems (MLSys), 2025. URL: https://arxiv.org/abs/2411.19379.

Pang, B. and Lee, L. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In ACL, 2004.

Rajashekar, P., Dixon, E., Ryan, C., and Hadfield, J. Effective context engineering for AI agents, 2025. URL: https://www.anthropic.com. Anthropic Engineering.

Raju, R., Ji, M., Upasani, S., Li, B., and Thakker, U. The limits of long-context reasoning in automated bug fixing, 2026. URL: https://arxiv.org/abs/2602.16069.

Ray, S., Pan, R., Gu, Z., Du, K., Feng, S., Ananthanarayanan, G., Netravali, R., and Jiang, J. Metrics: Fast quality-aware ray systems with configuration adaptation. In Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles, SOSP ’25, pp. 606–622, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 97894008718700.
```

### --- Page 0016 ---

```markdown
doi: [10.1145/3731569.3764855](https://doi.org/10.1145/3731569.3764855). URL [https://doi.org/10.1145/3731569.3764855](https://doi.org/10.1145/3731569.3764855).

Shen, H., Yan, H., Xing, Z., Liu, M., Li, Y., Chen, Z., Wang, Y., Wang, J., and M. Y. Ragsytn: Synthetic data for robust and faithful rag component optimization, 2025. URL [https://arxiv.org/abs/2505.10989](https://arxiv.org/abs/2505.10989).

Shuster, K., Poff, S., Chen, M., Kiela, D., and Weston, J. Retrieval augmentation reduces hallucination in conversation, 2021. URL [https://arxiv.org/abs/2104.07567](https://arxiv.org/abs/2104.07567).

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., and Potts, C. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP, 2013.

Tang, J., Zhao, Y., Zhu, K., Xiao, G., Kasiski, B., and Han, S. QUEST: Query-aware sparsity for efficient long-context LLM inference. In Proceedings of the 41st International Conference on Machine Learning (ICML), pp. 4791–4794, 2021.

Tang, Y. and Yang, Y. Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries, 2021. URL [https://arxiv.org/abs/2109.15391](https://arxiv.org/abs/2109.15391).

Varma, S., Voice, T., Sun, Y., Chen, Z., Yu, R., and V. K. Hilbert: Recursively building formal proofs with informal reasoning, 2025. URL [https://arxiv.org/abs/2509.22819](https://arxiv.org/abs/2509.22819).

Wu, W., Pan, Z., Fu, K., Wang, C., Chen, L., Bai, Y., Wang, T., Wang, Z., and Xiong, H. TokenSelect: Efficient long-context inference and length extrapolation for LLMs via dynamic token-level KV cache reduction. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 21275–21292, 2025.

Xie, Z., Xu, Z., Zhao, M., An, Y., Mailthody, V. S., Mahlke, S., Garland, M., and Kozyrakis, C. Strata: Hierarchical context caching for long context language model serving, 2025. URL [https://arxiv.org/abs/2508.18572](https://arxiv.org/abs/2508.18572).

Yang, J., Hou, B., Wei, W., Bao, Y., and Chang, S. Kivlink: Accelerating language models via efficient kv cache reuse, 2025. URL [https://arxiv.org/abs/2502.16022](https://arxiv.org/abs/2502.16022).

Yang, X., Chen, T., and Chen, B. Ape: Faster and longer context-augmented generation via adaptive parallel encoding, 2025b. URL [https://arxiv.org/abs/2502.05431](https://arxiv.org/abs/2502.05431).

Yao, J., Li, H., Liu, Y., Ray, S., Cheng, Y., Zhang, Q., Du, K., Lu, S., and Jiang, J. Cachefed: Fast large language model serving for rag with cached knowledge fusion. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 94–109, 2025. URL [https://doi.org/10.1145/3689013.3696098](https://doi.org/10.1145/3689013.3696098).

Yu, H., Chen, T., Feng, J., Chen, J., Dai, W., Yu, Q., Zhang, Y.-Q., Ma, W.-Y., Liu, J., Wang, M., and Zhou, H. MemAgent: Reshaping long-context LLM with multimodal RL-based memory agents, 2025. URL [https://arxiv.org/abs/2205.02259](https://arxiv.org/abs/2205.02259).

Zue, Z., Zhuang, H., Bai, A., Hui, K., Jagerma, R., Zeng, H., Qin, Z., Wang, D., Wang, X., and Bendersky, M. Inference scaling for long-context retrieval and generation. In The Thirteenth International Conference on Learning Representations (ICLR), 2025.

Zhan, H., Guo, Y., Wen, B., Li, Z., Ling, C., and Zhao, L. A survey of context engineering for large language models, 2025. URL [https://arxiv.org/abs/2507.13334](https://arxiv.org/abs/2507.13334).

Zhang, Q., H, X., Chen, H., Yi, X., Fu, F., Miao, X., Nie, X., Chen, W., and Cui, B. PQCache: Product quantization-based KVCache for long-context LLMs. In the ACM on Management of Data, 3(1):30–50, 2025a.

Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamarun, V., Rainton, J., Wu, C., Ji, M., Li, H., Thakker, U., Zou, J., and Oultook, K. Agent context engineering: Evolving contexts for self-improving language models, 2025b. URL [https://arxiv.org/abs/2510.04168](https://arxiv.org/abs/2510.04168).

Zhang, X., Zhang, P., Liu, S., Tang, J., Wan, Y., Yang, B., and Huang, F. Culturesynth: A hierarchical taxonomy-guided and retrieval-augmented framework for cultural question-answer synthesis, 2025. URL [https://arxiv.org/abs/2509.10886](https://arxiv.org/abs/2509.10886).

Zhang, Y., Sun, R., Chen, Y., Pfister, T., Zhang, R., and Arik, S. Chain of agents: Large language models collaborating on long-context tasks. Advances in Neural Information Processing Systems, 37:132208–132237, 2024.

Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., and Zhou, J. Qwen3 embedding: Advancing text embedding and reranking through foundational models. arXiv preprint arXiv:2506.05716, 2025d.

Zheng, L., Lin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barret, C., and Sheng, Y. Sglang: Efficient execution of structured language model programs, 2024. URL [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104).
```

### --- Page 0017 ---

```markdown
| Citation                                                                                          | Year |
|---------------------------------------------------------------------------------------------------|------|
| Zhou, J., Liu, Z., Liu, Z., Xiao, S., Wang, Y., Zhao, B., Zhang, C. J., Lian, D., and Xiong, Y.  | 2024 |
| Megapairs: Massive data synthesis for universal multimodal retrieval.                            |      |
| URL: [https://arxiv.org/abs/2412.14475](https://arxiv.org/abs/2412.14475)                       |      |
| Zilliz. Deepsearcher: Open-source deep research on private data.                                | 2025 |
| GitHub repository. URL: [https://github.com/zilliztech/deep-searcher](https://github.com/zilliztech/deep-searcher) |      |
```

### --- Page 0018 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

## A DEEPSEEK-R1 RESULTS

Table 6 presents end-to-end results for DeepSeek-R1 on MultihopRAG and NarrativeQA datasets, evaluated on 16×H20 and 32×H20 GPUs. ContextPilot achieves 1.81× and 1.52× throughput improvements on MultihopRAG and NarrativeQA respectively, with these speedups remaining consistent across both 16 and 32 GPU deployments using context-aware routing.

Table 6. DeepSeek-R1 results formatted for vertical readability. By stacking hardware configurations, the table width is minimized while preserving all data points.

| Method                | Hardware       | Prefill (Toks) | Cache Hit | F1 (%)  |
|-----------------------|----------------|----------------|-----------|---------|
| Vanilla               | 16×H20         | 969.69         | 5.12%     | 64.15   |
|                       | 32×H20         | 1860.08        | 4.17%     | 64.15   |
| ContextPilot w/o Annotations | 16×H20 | 1797.75        | 60.37%    | 64.09   |
|                       | 32×H20         | 3204.64        | 58.34%    | 64.09   |
| ContextPilot          | 16×H20         | 1799.75        | 61.47%    | 64.68   |
|                       | 32×H20         | 3207.64        | 58.41%    | 64.68   |

![DeepSeek-R1 results](assets/page_0018_img_1.png)

## B ATTENTION MAP ANALYSIS

Since context engineering (Rajasekaran et al., 2025) and in-context learning (Kirsch et al., 2022) strongly influence model inference, we analyze attention patterns when explicit annotations are introduced to recall the original relevance ranking.

Figure 9 and Figure 10 compare the final-layer attention maps of Qwen3 and LLaMA3.3 under this setup. When given explicit document-priority cues, both models exhibit consistent attention behaviors despite architectural differences. They correctly focus on document tokens ([Doc.1], [Doc.2], [Doc.3]), reflecting awareness of the mismatch between the aligned and original sequences, as indicated by intersections between queries in the annotation region and keys in the context region. As the context re-aligns with the original sequence, both models emphasize ([Doc.2]) while parsing ([Doc.1]) and ([Doc.3]), showing that the cue ([Doc.2] > [Doc.1] > [Doc.3]) effectively directs cross-document attention. Hence, explicit annotations reshape internal attention, aligning it with semantic rather than positional priority.

This finding supports a central hypothesis of ContextPilot: explicit annotations can make a comeback from the accuracy lost on alignment by re-establishing alignment with the original retrieval semantics.

![Attention map of the last layer attention of Qwen3](assets/page_0018_img_2.png)

## C DOCUMENT ACCESS DISTRIBUTION

Figure 11 shows the cumulative distribution of document access frequency across three RAG datasets. A small fraction of documents accounts for the majority of retrievals: the top 

![Cumulative distribution of document access frequency](assets/page_0018_img_3.png)
```

### --- Page 0019 ---

```markdown
20\% most frequently accessed documents cover 79.2\% of retrieval events on MultihopRAG, 57.4\% on NarrativeQA, and 49.6\% on QASPER. This heavy-tailed distribution confirms that real-world RAG workloads exhibit substantial context overlap across sessions, motivating context reuse through prefix-aligned alignment.

![Document access distribution (CDF) across three datasets. The vertical dashed line marks 20\% of documents; horizontal lines indicate the cumulative percentage of accesses from these top documents.](assets/page_0019_img_1.png)

## D ADDITIONAL EVALUATION DETAILS

### D.1 Time-Series Metrics

Figure 12 shows how cache hit ratio evolves as the workload progresses. ContextPilot maintains approximately 34\% cache hit ratio compared to baseline’s 7\% throughout the entire workload, demonstrating a sustained 5\% improvement that is not a transient warm-up effect.

![Cache hit ratio over workload progress for Llama-3.3-70B and Qwen3-32B. ContextPilot maintains $\sim$5\% higher hit ratio throughout execution.](assets/page_0019_img_2.png)

Figure 13 presents cumulative cached tokens as a metric for radix tree prefix reuse. ContextPilot achieves 10.33M cached tokens versus baseline’s 2.42M at completion on Llama3.3-70B-Instruct (4.27×) and 10.50M versus 2.75M on Qwen3-32B (3.82×). The “w/o Scheduling” variant (6.85M, 2.83×) confirms that both alignment and scheduling contribute to the improvement.

![Cumulative cached tokens (radix tree reuse) over workload progress for Llama-3.3-70B and Qwen3-32B. ContextPilot achieves 4× better prefix reuse.](assets/page_0019_img_3.png)

### D.2 Accuracy Breakdown

Table 7 provides a detailed breakdown of accuracy contributions from each ContextPilot component.

| Model       | Configuration      | MultihopRAG | NarrativeQA |
|-------------|--------------------|-------------|-------------|
| Qwen3-32B   | Baseline           | 60.4\%      | 28.4\%      |
|             | Alignment          | 60.0\%      | 28.2\%      |
|             | + Annotation       | 64.4\%      | 29.6\%      |
|             | + Scheduling       | 64.4\%      | 29.6\%      |
| Qwen3-4B    | Baseline           | 35.2\%      | 16.0\%      |
|             | + Alignment        | 34.5\%      | 15.2\%      |
|             | + Annotation       | 36.6\%      | 17.3\%      |
|             | + Scheduling       | 36.6\%      | 17.3\%      |

### D.3 Per-Request Overhead

Table 8 reports the per-request overhead of ContextPilot components, measured on 2K requests with $k = 15$ on NVIDIA A6000.

| Component   | Latency (ms) |
|-------------|--------------|
| Search      | 0.068        |
| Alignment   | 0.047        |
| De-duplication | 0.600    |
| **Total**   | **∼0.7**     |
```

### --- Page 0020 ---

```markdown
# E OPENCLAW PIPELINE DIAGRAM

Input Proposal: [system, user: 200K], Output Proposal: [system, user: 20K],  
$C_{1}$: assistant, user: 50K, $C_{2}$: assistant, user: 20K

![OpenClaw + ContextPilot pipeline diagram](assets/page_0020_img_1.png)

Figure 14. The OpenClaw + ContextPilot pipeline. The OpenClaw agent sends requests to the ContextPilot proxy, which aligns prompts with the prefix cache of the inference engine, deduplicates overlapping context both across turns and within context blocks, and forwards optimized prompts to the inference engine.

## F SYSTEM OVERHEAD WITH ZERO CONTEXT OVERLAP

Zero context overlap represents the worst case for ContextPilot, as it isolates system overhead with no retrieval benefit. Using a synthetic RAG workload with no retrieval overlap, ContextPilot adds only $0.72$ of prefill latency for 1K contexts (one-hour job), demonstrating that querying the context index during operation incurs negligible overhead.

## G IMPACT OF PREFIX CACHE SIZE

The prefix KV-cache stores precomputed attention states for reuse across requests; a larger cache retains more contexts, increasing the chance of a cache hit and reducing redundant prefill computation. We evaluate this effect on MultihopRAG across A6000 (48 GB) and H100 (80 GB) GPUs. Because ContextPilot aligns contexts with the prefix cache to maximize overlap, it benefits disproportionately from larger caches: SGLan hit ratios improve from $29.64\%$ to $33.97\%$, and VLLM from $35.90\%$ to $43.4\%$, translating directly into higher prefill throughput. In contrast, baselines see smaller gains since their context alignment does not systematically exploit the additional capacity.

## H DETAILED ALGORITHM PSEUDOCODE

This appendix collects all algorithm pseudocode referenced in the main text. Algorithm 1 details the context index tree search, Algorithm 2 describes context alignment for prefix sharing, Algorithm 3 formalizes context de-duplication, Algorithm 4 provides the full pseudocode for context index construction via hierarchical clustering, and Algorithm 5 details the tree-based request grouping and scheduling procedure.

### Algorithm 1 Context Index Tree Search

Require: Context $C = \{b_{1}, \ldots, b_{k}\}$, root node $R$  
Ensure: Search path, best matching node  
1: $cur \gets R$, path $\gets []$  
2: while $cur$ has children do  
3: \quad $best \gets \text{arg min}_{c \in cur.children} C_{R}.blocks \neq \emptyset \text{Dist}(C, c)$  
4: \quad if no overlapping child found then  
5: \quad \quad break  
6: \quad end if  
7: \quad $path.\text{append}( \text{index of best})$  
8: \quad if $best$ is leaf then  
9: \quad \quad return path, best  
10: \quad end if  
11: \quad $cur \gets best$  
12: end while  
13: return path, cur  

### Algorithm 2 Context Alignment for Prefix Sharing

Require: Context $C$ with context blocks, Context Tree root node $R$  
Ensure: Aligned context $C'$  
1: $bestMatch \gets \text{FIND\_BEST\_MATCH\_NODE}(C, R)$  
2: if $bestMatch = \text{null}$ then  
3: \quad $C' \gets C$  
4: \quad return $C'$  
5: end if  
6: $prefix \gets \text{bestMatch.context}$  
7: $remaining \gets C \setminus prefix$  
8: $C' \gets prefix \cup remaining$  
9: return $C'$  

### function FINDBESTMATCHNODE(C, R)

1: if $C$ is initialization context then  
2: \quad return $C.parent$  
3: else  
4: \quad return SEARCHTREE(C, R)  
5: end if  
```

### --- Page 0021 ---

```markdown
# ContextPilot: Fast Long-Context Inference via Context Reuse

## Algorithm 3 Context De-duplication

**Require:** New context $C_{new}$, context index $Z$, conversation ID $id$, chunk modulus $M$  
**Ensure:** Deduplicated context $C'$, location annotations $H$

1. $S \gets T_{seen}.subblocks[id] \; \text{blocks indexed in prior turns}$
2. $C' \gets []$
3. $H \gets []$
4. for each block $b \in C_{new}$ do
5. &nbsp;&nbsp;if $b \in S$ then
6. &nbsp;&nbsp;&nbsp;&nbsp;$h \gets \text{"refer to } [b] \text{ in previous conversation"}$
7. &nbsp;&nbsp;&nbsp;&nbsp;$C'.append(h); H.append(h)$
8. &nbsp;&nbsp;else
9. &nbsp;&nbsp;&nbsp;&nbsp;$\{s_1, \ldots, s_m\} \gets CDC(b, M) \; \text{context-defined chunking}$
10. &nbsp;&nbsp;&nbsp;&nbsp;for $j = 1 \; \text{to} \; m$ do
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$f \gets HASH(s_j)$
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $f \in V$ and $V[f] \neq b$ then
13. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Replace $s_j$ with location annotation to $V[f]$
14. &nbsp;&nbsp;&nbsp;&nbsp;else
15. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$V[f] \gets b$
16. &nbsp;&nbsp;&nbsp;&nbsp;end if
17. &nbsp;&nbsp;end for
18. &nbsp;&nbsp;$C'.append(CONCAT(s_1, \ldots, s_m))$
19. &nbsp;&nbsp;end if
20. end for
21. $S \gets S \cup C_{new} \; \text{register for future turns}$
22. $T_{seen}.subblocks[id] \gets V \; \text{persist sub-block hashes}$
23. return $C', H$

## Algorithm 4 Context Index Construction via Hierarchical Clustering

**Input:** Batch of $n$ contexts $S = \{s_1, s_2, \ldots, s_n\}$, distance function $d( \cdot, \cdot )$  
1. // **Phase 1:** Distance computation and clustering  
2. $D \gets \text{pairwise distances using } d(s_i, s_j) \; \text{by GPU or CPU}$  
3. $Z \gets \text{LINKAGE}(D) \; \text{hierarchical clustering}$  
4. // **Phase 2:** Build tree with deduplication  
5. for $i = 1 \; \text{to} \; n$ do  
6. &nbsp;&nbsp;Create leaf node $v_i$ with $v_i.content \gets s_i$  
7. &nbsp;&nbsp;Redirect $v_i \rightarrow v_j; v_j.freq \gets 1 \; \text{dedup}$  
8. end if  
9. end for  
10. for each node $(c_1, c_2, \delta)$ in $Z$ do  
11. &nbsp;&nbsp;$v \gets \text{new internal node}$  
12. &nbsp;&nbsp;$v.content \gets c_1.content \cup c_2.content$  
13. &nbsp;&nbsp;$v.children \gets [c_1, c_2]; c_1.parent, c_2.parent \gets v$  
14. end for  
15. Remove empty internal nodes; relink children to grandparents  
17. // **Phase 3:** Top-down prefix alignment  
18. Compute search paths from root to all nodes  
19. for each node $v$ in BFS order from root do  
20. &nbsp;&nbsp;if $v$ is not root and $v.parent.docs \neq \emptyset$ then  
21. &nbsp;&nbsp;&nbsp;&nbsp;$v.docs \gets v.parent.docs \oplus (v.docs \cdot v.parent.docs)$  
22. &nbsp;&nbsp;end if  
23. end for  
24. return aligned contexts from leaf nodes, search paths  

## Algorithm 5 Search-Path-Based Request Grouping and Scheduling

**Input:** $N$ contexts with search paths $\{P_1, \ldots, P_N\}$ from alignment  
**Output:** Scheduled execution order  
1. // **Phase 1:** Group by root prefix  
2. $G \gets \text{empty map}$  
3. for $i = 1 \; \text{to} \; N$ do  
4. &nbsp;&nbsp;$key \gets P_i[0] \; \text{first element of search path}$  
5. &nbsp;&nbsp;$G[key].append(i)$  
6. end for  
7. // **Phase 2:** Sort within each group  
8. for each group $G \in G$ do  
9. &nbsp;&nbsp;Sort $G$ by $|P_i|$ descending \; \text{longer paths first}  
10. end for  
11. // **Phase 3:** Order groups and flatten  
12. Sort groups by $|G|$ descending \; \text{largest groups first}  
15. return concatenation of all groups  
```


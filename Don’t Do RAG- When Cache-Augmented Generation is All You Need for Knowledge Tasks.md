# Don’t Do RAG- When Cache-Augmented Generation is All You Need for Knowledge Tasks

### --- Page 0001 ---

```markdown
# Don’t Do RAG:  
When Cache-Augmented Generation is All You Need for Knowledge Tasks

**Brian J. Chan\***  
**Chao-Ting Chen\***  
**Jui-Hung Cheng\***  
Department of Computer Science  
National Chengchi University  
Taipei, Taiwan  
{110730365, 110730338, 110730307}@nccu.edu.tw  

**Hen-Hsen Huang**  
Institute of Information Science  
Academia Sinica  
Taipei, Taiwan  
hhhuang@iis.sinica.edu.tw  

---

## Abstract  
Retrieval-augmented generation (RAG) has gained traction as a powerful approach for enhancing language models by integrating external knowledge sources. However, RAG introduces challenges such as retrieval latency, potential errors in document selection, and increased system complexity. With the advent of large language models (LLMs) featuring significantly extended context windows, this paper proposes an alternative paradigm, cache-augmented generation (CAG) that bypasses real-time retrieval. Our method involves preloading all relevant resources, especially when the documents or knowledge for retrieval are of a limited and manageable size, into the LLM's extended context and caching its runtime parameters. During inference, the model utilizes these preloaded parameters to answer queries without additional retrieval steps. Comparative analyses reveal that CAG eliminates retrieval latency and minimizes retrieval errors while maintaining context relevance. Performance evaluations across multiple benchmarks highlight scenarios where long-context LLMs either outperform or complement traditional RAG pipelines. These findings suggest that, for certain applications, particularly those with a constrained knowledge base, CAG provides a streamlined and efficient alternative to RAG, achieving comparable or superior results with reduced complexity.

## CCS Concepts  
- Computing methodologies → Discourse, dialogue and pragmatics; Natural language generation; Information systems → Specialized information retrieval.

## Keywords  
Large Language Models, Retrieval Augmented Generation, Retrieval-Free Question Answering.

## 1 Introduction  
The advent of retrieval-augmented generation (RAG) [1, 3] has significantly enhanced the capabilities of large language models (LLMs) by dynamically integrating external knowledge sources. RAG systems have proven effective in handling open-domain questions and specialized tasks, leveraging retrieval pipelines to provide contextually relevant answers. However, RAG is not without its drawbacks. The need for real-time retrieval introduces latency, while errors in selecting or ranking relevant documents can degrade the quality of the generated responses. Additionally, integrating retrieval and generation components increases system complexity, necessitating careful tuning and adding to the maintenance overhead.

This paper proposes an alternative paradigm, cache-augmented generation (CAG), leveraging the capabilities of long-context LLMs to address these challenges. Instead of relying on a retrieval pipeline, as shown in Figure 1, our approach involves preloading the LLM with all relevant documents in advance and precomputing the key-value (KV) cache, which encapsulates the inference state of the LLM. The preloaded context enables the model to provide rich, contextually accurate answers without the need for additional retrieval during runtime. This approach eliminates retrieval latency, mitigates retrieval errors, and simplifies system architecture, all while maintaining high-quality responses by ensuring the model processes all relevant context holistically.

![Figure 1: Comparison of Traditional RAG and our CAG Workflows: The upper section illustrates the RAG pipeline, including real-time retrieval and reference text input during inference, while the lower section depicts our CAG approach, which preloads the KV-cache, eliminating the retrieval step and reference text input at inference.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
# Conference acronym 'XX', June 03–05, 2018, Woodstock, NY
Brian J Chan, Chao-Ting Chen, Jui-Hung Cheng, and Hen-Hsen Huang

Recent advances in long-context LLMs have extended their ability to process and recover substantial textual inputs. By accommodating larger context windows, these models can assimilate extensive information in a single inference step, making them well-suited for tasks like document comprehension, multi-turn dialogue, and summarization of lengthy texts. This capability eliminates the dependency on real-time retrieval, as all necessary information can be preloaded into the model. These developments create opportunities to streamline workflows for knowledge-intensive tasks, potentially reducing or even eliminating the need for traditional RAG systems.

Recent studies [2, 4] have investigated the performance of long-context models in RAG tasks, revealing that state-of-the-art models like GPT-0, GPT-4, and Claude 3.5 can effectively process large amounts of retrieved data, outperforming traditional systems in many scenarios. Findings suggest that as long as documents fit within the extended context length, traditional RAG systems can be replaced by these long-context models. Similarly, Liu et al. [5] has demonstrated the benefits of precomputed KV caching to improve efficiency, albeit with the need for position ID rearrangement to enable proper functioning. Nonetheless, these methods remain vulnerable to retrieval failures inherent to RAG systems.

Through a series of experiments comparing traditional RAG workflows with our proposed approach, we identify scenarios where long-context LLMs outperform RAG in both efficiency and accuracy. By addressing the technical and practical implications, this paper aims to provide insights into how long-context LLMs may serve as a streamlined, effective alternative to RAG, particularly for cases where the documents or knowledge to retrieve are of limited, manageable size. Our findings challenge the default reliance on RAG for knowledge integration tasks, offering a simplified, robust solution to harness the growing capabilities of long-context LLMs. Our contributions are threefold as follows:

1. **Retrieval-Free Long-Context Paradigm**: Introduced a novel approach leveraging long-context LLMs with preloaded documents and precomputed KV caches, eliminating retrieval latency, errors, and system complexity.
2. **Performance Comparison**: Conducted extensive experiments showing scenarios where long-context LLMs outperform traditional RAG systems, especially with manageable knowledge bases.
3. **Practical Insights**: Provided actionable insights into optimizing knowledge-intensive workflows, demonstrating the viability of retrieval-free methods for specific applications. Our CAG framework is released publicly.¹

## 2 Methodology

Our CAG framework leverages the extended context capabilities of long-context LLMs to enable retrieval-free knowledge integration. By preloading external knowledge sources, such as a collection of documents $D = \{d_1, d_2, \ldots\}$, and precomputing the key-value (KV) cache $KV$, we address the computational challenges and inefficiencies inherent to real-time retrieval in traditional RAG systems. The operation of our framework is divided into three phases:

1. **External Knowledge Preloading**  
   In this phase, a curated collection of documents $D$ relevant to the target application is preprocessed and formatted to fit within the model's extended context window. The LLM $M$, with parameters $\theta$, processes $D$, transforming it into a precomputed KV cache:
   $$
   C_{KV} = KV\text{-Encode}(D) \tag{1}
   $$

   This KV cache, which encapsulates the reference state of the LLM, is stored on disk or in memory for future use. The computational cost of processing $D$ is incurred only once, regardless of the number of subsequent queries.

2. **Inference**  
   During inference, the precomputed KV cache $C_{KV}$ is loaded alongside the user's query $Q$. The LLM utilizes this cached context to generate responses:
   $$
   R = M(Q | C_{KV}) \tag{2}
   $$

   By preloading the external knowledge, this phase eliminates retrieval latency and reduces risks of errors or omissions that arise from dynamic retrieval. The combined prompt $P = \text{Concat}(D, Q)$ ensures a unified understanding of both the external knowledge and the user query.

3. **Cache Reset**  
   To maintain system performance across multiple inference sessions, the KV cache, stored in memory, can be reset efficiently. As the KV cache grows, it can be appended-only manner, leading to the need for rapid reinitialization without reloading the entire cache from disk, ensuring sustained speed and responsiveness:
   $$
   C_{KV}^{\text{reset}} = \text{Truncate}(C_{KV}, t_1, t_2, \ldots, t_k) \tag{3}
   $$

   The proposed methodology offers several significant advantages over traditional RAG systems:
   - **Reduced Inference Time**: By eliminating the need for real-time retrieval, the inference process becomes faster and more efficient, enabling quicker responses to user queries.
   - **Unified Context**: Preloading the entire knowledge collection into the LLM provides a holistic and coherent understanding of the documents, resulting in improved response quality and consistency across a wide range of tasks.
   - **Simplified Architecture**: By removing the need to integrate retrievers and generators, the system becomes more streamlined, reducing complexity, improving maintainability, and lowering development overhead.

Looking forward, our approach is poised to become even more powerful with the anticipated advancements in LLMs. As models continue to expand their context length, they will be able to process increasingly larger knowledge collections in a single inference step. Additionally, the improved ability of these models to extract and utilize relevant information from long contexts will further enhance their performance. These two trends will significantly extend the usability of our approach, enabling it to handle more complex and diverse applications. Consequently, our methodology is well-positioned to become a robust and versatile solution.

¹ https://github.com/hhhuang/CAG
```

### --- Page 0003 ---

```markdown
# Don’t Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks

Conference acronym 'XX', June 03-05, 2018, Woodstock, NY

## Table 1: Overview of the SQuAD and HotPotQA test sets with varying reference text lengths, highlighting the number of documents, questions, and associated responses for each configuration.

| Source   | Size   | # Docs | # Tokens | # QA Pairs |
|----------|--------|--------|----------|------------|
| HotPotQA | Small  | 16     | 21k      | 1,392      |
|          | Medium | 32     | 43k      | 1,056      |
|          | Large  | 64     | 85k      | 1,344      |
| SQuAD   | Small  | 3      | 21k      | 500        |
|          | Medium | 4      | 32k      | 500        |
|          | Large  | 7      | 50k      | 500        |

## 3 Experiments

### 3.1 Experimental Setup

To evaluate the effectiveness of our proposed method, we conducted experiments using two widely recognized question-answering benchmarks: the Stanford Question Answering Dataset (SQuAD) 1.0 [6] and the HotPotQA dataset [7]. These datasets provide complementary challenges, with SQuAD focusing on precise, context-aware answers within passages and HotPotQA emphasizing multi-hop reasoning across multiple documents. Each test dataset consisted of documents $D = \{d_1, d_2, \ldots\}$ paired with questions $Q = \{q_1, q_2, \ldots\}$ and golden responses $R = \{r_1, r_2, \ldots\}$. These datasets provide a robust platform for assessing both single-context question answering and complex multi-hop reasoning.

To investigate how different levels of reference text length influence retrieval difficulty, we created three sets for each dataset, varying the size of the reference text. For example, in the HotPotQA-small configuration, we sampled 16 documents $D_s$ from the HotPotQA dataset to form a long reference text. QA pairs associated with $D_s$ were selected as test instances. The same methodology was applied to create test sets for SQuAD.

The dataset statistics are summarized in Table 1. As the number of documents (and hence the length of the reference text) increases, the task becomes more challenging, particularly for RAG systems. Longer reference texts increase the difficulty of accurately retrieving the correct information, which is critical for LLMs to generate high-quality responses.

The primary task involves generating accurate and contextually relevant answers $\hat{R} = \{ \hat{r}_1, \hat{r}_2, \ldots \}$ for the SQuAD and HotPotQA questions, based on the respective preloaded passages. By leveraging the precomputed key-value cache $C_{KV} = KV\text{-}Encode(D)$, our system generates responses $\hat{r} = M(q | C_{KV})$ without relying on retrieval mechanisms during inference. This unified approach allows for direct performance comparisons against traditional RAG systems, highlighting the strengths and limitations of our method across diverse QA challenges.

The experiments were executed on Tesla V100 32G x 8 GPUs. For all experiments, we used the Llama 13B Instruction model as the underlying LLM across all systems, including both the RAG systems and our proposed method. This model supports input sizes of up to 128k tokens, enabling the processing of extensive contexts. For our proposed method, the context of each dataset was preloaded into the model via a precomputed key-value (KV) cache. For SQuAD, the documents $D_s$ were encoded into a KV cache $C_{KV} = KV\text{-}Encode(D_s)$, while for HotPotQA, the documents $D_h$ were encoded into $C_{KV} = KV\text{-}Encode(D_h)$. These caches were stored offline and loaded during inference to facilitate real-time retrieval, ensuring comprehensive access to all relevant information for each dataset.

### 3.2 Baseline Systems

The baseline RAG systems were implemented using the Llamalab-tex framework, employing two retrieval strategies: BM25 for sparse retrieval and OpenAI Indexes for dense retrieval. Each dataset—SQuAD and HotPotQA—was evaluated separately, with retrieval systems configured to fetch passages exclusively from the respective dataset to ensure focused and fair evaluation. The details of each baseline system are as follows:

1. **Sparse Retrieval System (BM25)**: The first baseline system employed BM25 indexes for retrieval. BM25, a sparse retrieval algorithm, ranks documents based on term frequency-inverse document frequency (TF-IDF) and document length normalization. Given a query $q$, BM25 retrieves the top-k passages $P_k = \{p_1, p_2, \ldots, p_k\}$ from the indexed collection $D$. These passages are then fed into the generator, M, to synthesize answers:

   $$
   \hat{r} = M(q | P_k)
   $$

   BM25 provides a robust and interpretable retrieval mechanism, suited for tasks involving keyword matching.

2. **Dense Retrieval System (OpenAI Indexes)**: The second baseline utilized OpenAI indexes, which employ dense embeddings to represent both documents and queries in a shared semantic space. For a query $q_i$, dense retrieval selects the top-k passages $P_k$ that are semantically aligned with the query, offering improved contextual understanding compared to sparse methods. These passages were similarly passed to the generator for answer synthesis as Equation 4. This system is particularly effective for questions requiring more contextual matching beyond exact term overlap.

Our experiments were conducted on both the SQuAD and HotPotQA datasets to evaluate the performance of different systems in terms of similarity to ground-truth answers, measured using BERTScore [8]. For the RAG baselines, the top-1, top-3, top-5, and top-10 retrieved passages were used for inference. In contrast, our CAD utilized the preloaded context specific to each dataset to generate answers without retrieval constraints.

### 3.3 Results

As shown in Table 2, the experiments revealed clear distinctions between our proposed method and traditional RAG systems. Our proposed approach achieved the highest BERTScore in most cases.
```

### --- Page 0004 ---

```markdown
# Conference acronym 'XX', June 03–05, 2018, Woodstock, NY
## Brian J Chan, Chao-Ting Chen, Jui-Hung Cheng, and Jen-Hsen Huang

## Table 2: Experimental Results

| Size  | System      | HotPotQ   | SQuAD    | BERT-Score |
|-------|-------------|-----------|----------|------------|
|       |             | Top-K    |          |            |
| Small | Sparse RAG  | 1 0.0673  | 0.7469   |            |
|       |             | 3 0.0673  | 0.7999   |            |
|       |             | 5 0.7549  | 0.8022   |            |
|       |             | 10 0.7461 | 0.8191   |            |
|       | Dense RAG   | 1 0.7079  | 0.6445   |            |
|       |             | 3 0.7509  | 0.7304   |            |
|       |             | 5 0.7414  | 0.7583   |            |
|       |             | 10 0.7516 | 0.8035   |            |
|       | CAG (Ours)  |           | 0.7759   | 0.8265     |
| Medium| Sparse RAG  | 1 0.6527  | 0.7036   |            |
|       |             | 3 0.6191  | 0.7411   |            |
|       |             | 5 0.7616  | 0.7467   |            |
|       |             | 10 0.7238 | 0.7420   |            |
|       | Dense RAG   | 1 0.7135  | 0.6188   |            |
|       |             | 3 0.7446  | 0.6899   |            |
|       |             | 5 0.7287  | 0.7407   |            |
|       |             | 10 0.7451 | 0.7350   |            |
|       | CAG (Ours)  |           | 0.7696   | 0.7512     |

## Table 3: Comparison of Generation Time

| Dataset Size | System      | Generation Time |
|--------------|-------------|------------------|
| HotPotQA     | Small       | CAG 0.85292      |
|              | Medium      | w/o CAG 9.24734  |
|              |             | CAG 16.81642     |
|              | Large       | w/o CAG 2.32667  |
|              |             | w/o CAG 94.34917 |
| SQuAD        | Small       | CAG 10.06950     |
|              | Medium      | w/o CAG 10.29593 |
|              |             | CAG 17.13114     |
|              | Large       | w/o CAG 13.35784 |
|              |             | CAG 2.40577      |
|              |             | w/o CAG 31.08368 |

situations, outperforming both RAG systems. By preloading the external context from the test set, our system eliminates retrieval errors and ensures holistic reasoning over all relevant information. This advantage is particularly evident in scenarios where RAG systems might retrieve incomplete or irrelevant passages, leading to suboptimal answer generation. The results underscore the robustness and efficiency of our method, especially for tasks requiring a unified understanding of the source material. When the retrieval method exhibits biases, our approach is inherently limited by their dependence on retrieval accuracy and ranking heuristics. Our approach bypasses these limitations, leveraging the long-context capabilities of the Llama 3 model to achieve superior performance.

Table 3 compares our CAG approach with standard in-context learning, where the reference text is provided dynamically during inference, requiring real-time KV-cache computation. The results demonstrate that CAG dramatically reduces generation time, particularly as the reference text length increases. This efficiency stems from preloading the KV-cache, which eliminates the need to process the reference context on-the-fly.

Moreover, CAG is also faster than traditional RAG systems, as it bypasses the retrieval stage entirely. Unlike RAG, CAG does not require retrieval or reference text input during inference, streamlining the process and further enhancing efficiency. These advantages make CAG an optimal solution for scenarios with extensive reference contexts, offering substantial time savings without compromising performance.

## 4 Conclusion

As long-context LLMs evolve, we present a compelling case for our approach, which balances efficiency and performance in our work compared with existing methods. By providing a foundational context and using retrieval only to augment dense or highly specific queries, we would balance the efficiency of retrieval with the flexibility of retrained, making it suitable for scenarios where context completeness and adaptability are equally important.

## References
1. [Yunfa Guo, Yun Xiong, Xiong Zhang, Kangjun Jiang, Rui Yu, Yi Bi, and Jiajie Mao. 2023. Retrieving-and-generating for generative language models. arXiv preprint arXiv:2301.02253.](https://arxiv.org/abs/2301.02253)
2. [Patrick Lewis, Ethan Perez, Aleksandra Piktus, Floortje Bockting, Vladimir Karpukhin, Naman Goyal, Heinrich Kützing, Mike Lewis, Wei-Yun Yu, Tikhon Kolesnichenko, et al. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems 33 (2020): 9459–9470.](https://arxiv.org/abs/2005.11401)
3. [Zhuowan Li, Cheng Li, Mingyang Zhang, Qianhao Mei, and Michael S. Bernstein. 2024. Structural Knowledge Generation in Long Contexts: LLMs: A Comprehensive Study and Hybrid Approach. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, Francisco Bontchev, Daniel Teylor-Pick, and Anastasios Shirmarov (Eds.). Association for Computational Linguistics, Miami, Florida, US, 881–893.](https://doi.org/10.18653/v1/2024.emnlp-industry.76)
4. [Songbai Hu, Li Hao, Wuying Zhang, Zhi Chen, and Yaohua Tang. 2024. TurboRAG: Accelerated Retrieval Generation with Pre-computed KV Caches for Chunked Text. arXiv preprint arXiv:2405.01709.](https://arxiv.org/abs/2405.01709)
5. [Prabhakar Raju, Jilan Zhang, Konstantinos Lappas, and Jerry Ling. 2019. SQuAD: A 100,000-Question for Machine Comprehension of Text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, Jian Su, Kevin Duh, and Xinyu Zhang (Eds.). Association for Computational Linguistics, Austin, Texas, 2383–2392.](https://doi.org/10.18653/v1/D16-1264)
6. [Zhilin Yang, Peiyu Qi, Salim Benferhat, Yosuke Inoue, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
```

### --- Page 0005 ---

```markdown
# Don’t Do RAG:  
## When Cache-Augmented Generation is All You Need for Knowledge Tasks  
### Conference acronym 'XX, June 03-05, 2018, Woodstock, NY  

| Reference | Citation |
|-----------|----------|
| [8]       | Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. [n.d.] BERTScore: Evaluating Text Generation with BERT. In *International Conference on Learning Representations*. |
|           | Diverse, Explainable Multi-hop Question Answering. In *Conference on Empirical Methods in Natural Language Processing (EMNLP)*. |
```


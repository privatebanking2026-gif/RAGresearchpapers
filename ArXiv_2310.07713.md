# ArXiv 2310.07713

### --- Page 0001 ---

```markdown
# InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

**Boxin Wang**¹  **Wei Ping**¹  **Lawrence McAfee**¹  **Peng Xu**¹  **Bo Li**²  **Mohammad Shoeybi**¹  **Bryan Catanzaro**¹

---

## Abstract

Pretraining auto-regressive large language models (LLMs) with retrieval demonstrates better perplexity and factual accuracy by leveraging external databases. However, the size of existing pretrained retrieval-augmented LLM is still limited (e.g., Retro has 7.5B parameters), which limits the effectiveness of instruction tuning and zero-shot generalization. In this work, we introduce Retro 48B, the largest LLM pretrained with retrieval. Specifically, we continue to pretrain a 43B GPT model on additional 100 billion tokens using the Retro augmentation method by retrieving from 1.2 trillion tokens. Notably, the obtained foundation model, Retro 48B, largely outperforms the counterpart GPT 43B trained on 1.7 tokens in terms of perplexity with only 2.58% additional GPU hours, demonstrating the significant scaling potential of the method. After instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on a wide range of zero-shot tasks. Specifically, the average improvement of InstructRetro is 7% over its GPT counterpart across 8 short-form QA and reading comprehension tasks, 10% over GPT across 6 challenging long-form QA tasks, and 16% over GPT across 3 summarization tasks. Surprisingly, we find that one can ablate the encoder from InstructRetro architecture and directly use its decoder backbone, while achieving comparable results. Our results highlight the promising direction to obtain a better GPT decoder through continued pretraining with retrieval before instruction tuning. Our code and checkpoints are publicly available at: [https://huggingface.co/nvidia/retro-48b-instruct-4k](https://huggingface.co/nvidia/retro-48b-instruct-4k).

¹NVIDIA ²UIC. Correspondence to: Boxin Wang <boxinw@nvidia.com>, Wei Ping <wping@nvidia.com>.

Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).

---

## 1. Introduction

Retrieval helps large language models (LLM) to handle current events, detailed knowledge, proprietary information not in pretraining, and to improve factual grounding (e.g., Nakano et al., 2021; Thoppilan et al., 2022; Borgeaud et al., 2022). In the previous study, pretraining auto-regressive language model with retrieval (i.e., Retro) demonstrates successes in reducing perplexity (Borgeaud et al., 2022) and improving factual accuracy (Wang et al., 2023a).

In the past year, the decoder-only auto-regressive LLMs have demonstrated remarkable successes (e.g., OpenAI, 2022; 2023), because i) LLMs have been scaled to hundreds of billion parameters (Brown et al., 2000a; Rae et al., 2021; Smith et al., 2022; Chowdhery et al., 2022; Hoffmann et al., 2022; Tourvone et al., 2023b), and ii) instruction tuning (Wei et al., 2022a; Chung et al., 2022) and reinforcement learning from human feedback (RLHF) (Chung et al., 2022) recipes have been applied on these pretrained LLMs.

In contrast, the pretrained retrieval-augmented language models still have a relatively small number of parameters trained with a limited number of tokens. For example, the auto-regressive Retro has 7.5B parameters and is trained on 1.7 tokens (Borgeaud et al., 2022), Retro+ has 9.5B parameters and is trained on 330B tokens (Wang et al., 2023a), and TS-based Atlas has 11B parameters and is trained with retrieval on maximum 327M tokens (Lazard et al., 2022b). In addition, none of previous models have been applied with instruction tuning and RLHF to enhance usability. The lack of scaling could also limit the effectiveness of instruction tuning (Wei et al., 2022a) and other intriguing properties that exist in large language models (Wei et al., 2022b).

In this work, we scale up Retro to 48B parameters, trained on 1.7K tokens in total, i.e., 1.1T tokens for pretraining its GPT backbone, 100B tokens for continued retrieval-augmented pretraining while retrieving from 1.2T tokens. As a result, we can mitigate the zero-shot generalization gap on a wide range of tasks after applying instruction tuning.

Specifically, we make the following contributions:

1. We introduce Retro 48B, the largest LLM pretrained with retrieval. To save the computation budget, we continue to pretrain a 43B parameter GPT
```

### --- Page 0002 ---

```markdown
# InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

model (originally trained on 1.1T tokens) on additional 100B tokens by retrieving from 1.2T tokens. In contrast to Retro-finetuning (Borgeaud et al., 2022), that freezes pretrained decoder weights, we unfreeze the decoder, jointly train all the parameters and find better perplexity. ¹ Notably, with only 2.58% additional GPU hours, the perplexity improvement of Retro 48B over its GPT 43B counterpart is still significant even at this scale, demonstrating that the value of retrieval does not diminish with scaling model size.

2. After instruction tuning, InstructRetro 48B demonstrates strong zero-shot capability to incorporate context for various downstream tasks, and significantly outperforms instruction-tuned GPT with retrieval-augmented generation (RAG). The training pipeline of InstructRetro is shown in Figure 1.

3. Perhaps surprisingly, we find that one can directly ablate the encoder from InstructRetro 48B. The obtained decoder-only InstructRetro 43B still achieves very comparable results on downstream tasks. This highlights the promising direction of obtaining better decoder-only LLMs through continued pretraining and retrieval before instruction tuning.

We organize the rest of the paper as follows. We discuss the details of Retro 48B in § 3 and the instruction tuning recipe in § 4. We report results in Section 5 and conclude the paper in § 6.

## 2. Related Work

Retrieval-augmented language models have been established for open domain question answering for years (Karpukhin et al., 2020; Lewis et al., 2020; Guu et al., 2020; Borgeaud et al., 2022; Izacard et al., 2022b). In previous study, language models have been augmented with retrieval at inference (Khandwala et al., 2020; Yogatama et al., 2021), fine-tuning (Karpukhin et al., 2020; Lewis et al., 2020; Guu et al., 2020; Huang et al., 2023; Shi et al., 2023), and pretraining (Borgeaud et al., 2022; Izacard et al., 2022b; Wang et al., 2023a; Shi et al., 2023a). Retrieval-augmented pretraining is particularly interesting, as it can largely reduce model perplexity (Borgeaud et al., 2022), enhance factuality (Wang et al., 2023a), and improve downstream task accuracy after task-specific fine-tuning (Izacard et al., 2022b) and reasoning capability (Shi et al., 2023a).

¹ Note that, it turns out that unfreezing of decoder is an important design not only for better perplexity, and it eventually leads to the interesting finding after instruction tuning.

around 10B parameters (Borgeaud et al., 2022; Wang et al., 2023a; Izacard et al., 2022a), which largely limits the zero-shot generalization capability after instruction tuning (Wei et al., 2022a; Ouyang et al., 2022; Chung et al., 2022). For example, Wei et al. (2022a) find instruction tuning to be more effective when the decoder-only LLM has around 50B parameters.

Instruction tuning aims to teach LLMs to follow natural language instructions (Wei et al., 2022a; Ouyang et al., 2022; Sanh et al., 2022b; Mishra et al., 2022), which becomes an indispensable ingredient to build the state-of-the-art LLMs for downstream tasks (OpenAI, 2022; 2023; Tuyls et al., 2023b). In the past years, many high-quality instruction tuning datasets have been created, including FLAN (Chung et al., 2022), OpenAssistant (Köpf et al., 2023), Self-Instruct (Wang et al., 2022a), Dolly (Conover et al., 2023), Unnatural Instructions (Honovich et al., 2022). A concurrent work, RA-DIT (Lin et al., 2024), focuses on retrieval-augmented instruction tuning and further augments 20 instruction tuning datasets with retrieval, which supports fine-tuning both LLM and retrieval to yield high-quality neighbors. In contrast, our work focuses on retrieval-augmented pretraining, which extends the scale of the retrieval access to trillions of tokens. Although the two work are orthogonal, InstructRetro 43B outperforms RA-DIT 65B on certain tasks, as shown in Table 1. We leave it as an interesting future direction to apply RA-DIT retrieval-augmented instruction tuning data to the instruction tuning stage of InstructRetro for further performance improvement of retrieval-augmented LLMs.

## 3. Continued Pretraining of GPT with Retrieval

In this section, we start by introducing the preliminary of Retro (Borgeaud et al., 2022) and highlight some key differences between Retro and GPT. We then go through the pretraining details of how we scale up the size of Retro to 48B, a size that has never been studied before.

### 3.1 Preliminaries of Retro

Retro (Borgeaud et al., 2022) is an auto-regressive language model pretrained with retrieval augmentation. While Retro shares the backbone of GPT models, Retro differs from GPT by incorporating an additional Retro encoder. The Retro encoder is adept at encoding features of retrieved neighbors from external knowledge bases. Furthermore, Retro adds chunk-wise cross-attention layers within its decoder transformer architecture to integrate retrieved information from the Retro encoder effectively. This design paradigm also makes Retro different from the encoder-decoder architecture (e.g., T5 (Raffel et al., 2020) and Atlas (Izacard et al., 2022a)). The success of scaling decoder-only autoregres-
```

### --- Page 0003 ---

```markdown
# InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

![Training pipeline for InstructRetro 48B and InstructRetro 43B.](assets/page_0003_img_1.png)

## Stage 1: Pretraining Stage

- **Pretraining corpus (1.11)**  
  Pretrained GPT 43B

## Stage 1: Continued Pretraining w/ Retrieval

- **Pretraining corpus (100B)**  
  Retro 48B  
  **Retrieval**  
  Retrieval database (1.27)

## Stage 2: Instruction Tuning

- **Instruction tuning data**  
  InstructRetro 48B  
  Disable Encoder  
  InstructRetro 43B

### Retrieval Abstraction

Borgeaud et al. (2022) demonstrates that retrieval-augmented pretraining can significantly benefit from large-scale retrieval of billions of tokens. To build the retrieval database, we utilize the entire pretraining corpus, but holding about 1% as validation set. This ensures that both Retro and GPT models are pretrained on an equivalent volume of information from the pretraining corpus. Our retrieval database is a key-value database, where values are the chunks of tokens split from the pretraining corpus, and the keys are corresponding BERT embeddings (Devlin et al., 2018). The pretraining corpus consists of 1.2 trillion tokens of English corpus. More details of the pretraining corpus can be found in Appendix §A.1. In summary, our retrieval database comprises 19 billion chunks, with each chunk containing 64 tokens.

### Chunk-wise cross-attention

Aligning with the chunk-wise design of the retrieval database, Retro splits the input tokens into a sequence of chunks \( C = (C_1, \ldots, C_n) \). Retro retrieves nearest neighbor chunks using the previous chunk and fuses this information with the context from preceding chunks to guide the generation of the next chunk. Formally, given an input sequence \( X \) with tokens \( X = (x_1, \ldots, x_m) \), Retro splits \( X \) into a sequence of \( k \) chunks \( (C_1, \ldots, C_n) \) with chunk size \( m = 7 \). From a high-level perspective, Retro uses the last \( (i - 1) \)-th chunk \( C_{i-1} \) to retrieve \( k = 2 \) nearest neighbor chunks \( N(C_{i-1}) \) from the retrieval database, and fuses the contextual information from the previous chunks \( (C_1, \ldots, C_{i-1}) \) and retrieval information from \( N(C_{i-1}) \) by cross-attention to guide the generation of the next \( i \)-th chunk \( C_i \). To avoid breaking the causality, the autoregressive generation of \( i \)-th chunk \( C_i \) can only use the nearest neighbors of the previous chunk \( N(C_{i-1}) \) instead of \( N(C_i) \). In our work, we follow Borgeaud et al. (2022) and retrieve top-\( k = 2 \) nearest neighbors for each chunk, with chunk size \( m = 64 \) and the maximum number of tokens \( n = 4096 \).

### 3.2. Retro-fitting: continued pretraining with retrieval

There are two main challenges of scaling up Retro: the large-scale retrieval database and the pretraining cost of LLMs. To overcome the challenges, we leverage the Faiss index (Johnson et al., 2019) to achieve fast approximate nearest neighbor search and retro-fitting technique to reuse the pretrained GPT parameters and save computational cost.

### Retrieval index to the large-scale retrieval database

We use the Faiss index (Johnson et al., 2019) as the implementation for the dense retriever to search for approximate nearest neighbors in the BERT embedding space. We configure the Faiss index to cluster the dense embeddings into 222 centroids accelerated with Hierarchical Navigable Small World (HNSW) graphs (Malkov & Yashunin, 2018) to speed up the query. We also encode the embeddings with optimized product quantization (Gray & Neuhoff, 1998; Ge et al., 2014) to compress memory overhead and further improve the query throughput. As a result, we can achieve 4ms per query over the whole pretraining corpus averaged for each chunk on a DGX-A100 node. One may find more details in Appendix §B.

### Base pretrained GPT

We leverage continued pretraining (i.e., GPT-fitting and retro-fitting) based on pretrained GPT models. Specifically, we pretrain from scratch a set of GPT models with the following parameters size: 823M, 2.25B, 5.8B, 22B, and 43B. All of the models are based on Transformer (Vaswani et al., 2017) with different hidden dimen-
```

### --- Page 0004 ---

```markdown
# InstructRev: Instruction Tuning post Retrieval-Augmented Pretraining

## 4. Instruction Tuning

Instruction tuning can significantly improve the ability of foundational LLMs to follow instructions, thus improving zero-shot results on downstream tasks (e.g., Wei et al., 2022a; Chung et al., 2022). In this section, we further enhance Retro via instruction tuning.

### 4.1. Datasets Blending

Existing instruction tuning methods mainly leverage supervised fine-tuning on a blend of instruction following datasets (Wei et al., 2022a; Chung et al., 2022; Sanh et al., 2022a; Wang et al., 2023b).

We use a blend of high-quality instruction tuning datasets to train LLMs to follow instructions in conversational formats, which include: i) a high-quality social dialogue dataset SODA (Kim et al., 2022), ii) a long-form QA dataset ELI5 that requires elaborate answers (Fan et al., 2019), iii) LLM-generated instructions: Self-Instruct (Wang et al., 2022b) and Unnatural Instructions (Honovich et al., 2022), iv) FLAN and Chain-of-thought datasets (Chung et al., 2022; Wei et al., 2022c; Longpre et al., 2023), v) a private crowd-sourced conversational dataset and public human-annotated conversation datasets OpenAssistant (Köpf et al., 2023) and Dolly (Conover et al., 2023), and vi) samples from the pre-trained models.

The format of all the instruction tuning data is unified in a conversational way with three roles: “system”, “assistant”, and “user”. The “system” role sets up the tone and style of LLM assistants to give helpful, detailed, and polite answers to the user’s questions. The “user” and “assistant” role contains the questions and the corresponding answers from the instruction tuning datasets. We show an example format of the instruction data in Appendix C.1. In total, we collected a total of 128K high-quality samples for instruction tuning.

### 4.2. Training details

For each training sample, we take the multi-turn conversations between the user and the assistant as context and apply the loss mask only to the last response from the assistant. We use the standard language modeling loss with teacher forcing. Since Wei et al. (2022a) suggests that instruction tuning is most effective with larger language models, we apply instruction tuning to the GPT-fitting 43B model and the Retro 48B model, naming them “InstructGPT 43B” and “InstructRetro 48B”, respectively. We fine-tune the LLMs by taking the loss only on the answer part with a batch size of 128 and a learning rate of 5e-6 for 1000 steps with a weight decay of 0.01. We use the Adam optimizer (Kingma & Ba, 2015).
```

### --- Page 0005 ---

```markdown
![Perplexity evaluation of pretrained GPT models, GPT-fitting, and Retro-fitting models across various parameter sizes on the held-out validation set.](assets/page_0005_img_1.png)

![Validation perplexity of Retro-fitting (823M) when we freeze or unfreeze Retro decoder during continued pretraining on 100B tokens.](assets/page_0005_img_2.png)

5. Experiments
In this section, we conduct comprehensive studies on the zero-shot capabilities of InstructRetro and its GPT counterpart with RAG (InstructGPTRAG) across various downstream tasks to unveil the potential of Retro model after instruction tuning.

5.1 Experimental setup
Datasets. To demonstrate the generalization of instruction tuning, we follow FLAN (Wei et al., 2022a) and primarily focus on zero-shot evaluation of downstream tasks. Specifically, we consider two categories of open-ended QA tasks as well as text summarization tasks: (1) short-form QA or reading comprehension, which expects short answers (e.g., a few tokens) to be generated or extracted from the context, including Natural Question (NQ) (Kwiatkowski et al., 2019), TriviQA (Joshi et al., 2017), NewsQA (Trischler et al., 2016), SQuAD 1.1 (Rajpurkar et al., 2016), SQuAD 2.0 (Rajpurkar et al., 2018), Qasper (Dasigi et al., 2019), NarrativeQA (Kočiský et al., 2018), DROP (Dua et al., 2019). To compare with baselines, we use the split from KILT benchmark (Petroni et al., 2021) for NQ and TriviQA. For the other tasks, we use the official splits: (2) long-form QA, which expects longer answer spans within a few sentences, including doc2dial (Feng et al., 2020), two proprietary annotated or manual datasets (people ask questions about the particular car models), and other proprietary annotated T5 demonstration dataset; (3) summarization, which expects to summarize a long passage or context within a few sentences, including QMSum (Zhong et al., 2021), SummFrenD (Chen et al., 2021), and GovReport (Huang et al., 2021).

Retrieval-augmented generations (RAG). At pretraining, we use BERT embeddings to embed the retrieval database and support retrieval from trillions of tokens. For downstream task evaluation, we follow Retro (Borgeaud et al., 2022) and use task-specific corpus and state-of-
```

### --- Page 0006 ---

```markdown
![Simplified architecture diagram of InstructRetro](assets/page_0006_img_1.png)

## 5. Zero-shot evaluation on QA tasks

We present the zero-shot evaluation results across eight short-form QA and reading comprehension datasets in Table 1. We also apply InstructRetro to four open-ended QA tasks, as detailed in Table 2. These datasets are representative of real-world applications, including chatbots for IT support and customer service.

### Instruction tuning post retrieval-augmented pretraining yields a better GPT decoder.

From Table 1, we observe that InstructRetro 43B shows consistent accuracy improvement upon its counterpart InstructGPTₐₜₕ 43B across different datasets for short-form QA or reading comprehension tasks. Notably, the average relative improvement of InstructRetro across all the short-form datasets is around 7%. Given that both InstructRetro 43B and InstructGPTₐₜₕ 43B are pretrained and instruction tuned with identical datasets, hyper-parameters, and evaluation prompts, we attribute this consistent improvement to the training recipe of InstructRetro, which leverages continued pretraining with retrieval before instruction tuning. We hypothesize that retrieval-augmented pretraining enhances the capability of LLMs to utilize the information within the context (from both Retro encoder and decoder). The subsequent phase of instruction tuning further amplifies the effectiveness of InstructRetro in solving knowledge-intensive tasks. To have a deeper understanding, we provide an ablation study in §5.4.

From Table 1, we also show that InstructRetro 43B provides compelling performance than other state-of-the-art LLMs.
```

### --- Page 0007 ---

```markdown
| Task                | NQ   | TriviaQA | NewsQA | SQuAD 2.0 | SQuAD 1.1 | Quoref | NarrativeQA | DROP  |
|---------------------|------|----------|--------|-----------|-----------|--------|-------------|-------|
| Metric              | EM   | EM       | FI     | F1 / EM   | F1 / EM   | FI     | F1         | FI    |
| GPT-3 175B          | (Brown et al.) | 14.6 | 64.3     | 59.5 / 52.6 | -         | -         | -       | 23.6  |
| PaLM 2 -L          | (Chowdhery et al.) | 37.5 | -        | -         | -         | -         | -       | -     |
| GlaM 64B           | (Du et al.) | 24.7 | 71.3     | 71.1 / 64.7 | -         | -         | -       | 57.3  |
| FLAN-LMDA 137B     | (Wei et al.) | 20.7 | 68.1     | 44.2 / -   | 80.1 / -  | -         | -       | 22.7  |
| Llama 2 RAG 70B    | (Touvron et al.) | 37.7 | 65.6     | 53.4 / 64.1 | 73.4 / 66.2 | 69.7 | 52.7 | 57.2  |
| Retro 7.5B         | (Borgeaud et al.) | 8.9  | 36.0     | -         | -         | -         | -       | -     |
| Retro++ 9B         | (Wang et al.) | 25.8 | 48.3     | -         | -         | -         | -       | -     |
| Atlas 118          | (Izacard et al.) | 26.7 | 56.9     | -         | -         | -         | -       | -     |
| RA-DT 65B          | (Lin et al.) | 35.2 | 75.4     | -         | -         | -         | -       | -     |
| InsturctRetro 43B  | 37.0 | 78.1     | 52.4    | 70.7 / 64.3 | 72.4 / 65.8 | 71.5 | 53.9 | 51.8  |
| InsturctRetro 48B  | 38.9 | 78.3     | 75.6 / 69.3 | 71.1 / 70.4 | 76.2 | 60.0 | 54.8  |
| InsturctRetro 48B  | (encoder, Avg. -67) | 38.6 | 77.8     | 57.0 / 74.8 | 67.7 / 69.0 | 76.1 | 59.8 | 54.6  |

For example, InsturctRetro 43B achieves better accuracy than Llama 2 with RAG on multiple tasks, close to FLAN-LMDA 137B, which is 3x the size of InsturctRetro 43B.

**Impact of Retro encoder for downstream tasks.** We also notice that InsturctRetro 48B and 43B perform very comparably from Table 1. We enable the Retro encoder for retrieval-augmented pretraining, while disabling the Retro encoder due to the lack of retrieved high-quality neighbors for instruction tuning. Note that we still perform retrieval-augmented generation for downstream tasks, where the retrieved contexts are put into the decoder of both InsturctRetro 48B and 43B as part of the prompts. The only difference is whether we enable the cross attention gate in Figure 4 to attend the Retro encoder in InsturctRetro 48B or disable it in InsturctRetro 43B. When enabling the Retro encoder, we put the top-2 neighbors in the encoder to align with the pretraining behavior.

This suggests that although Retro is proficiently trained to infer both with and without the neighbors in the encoder, it is more beneficial to align with the instruction tuning protocols and bypass the Retro encoder to solely serve as a GPT decoder during evaluation. We think it is important and promising future research direction to explore retrieval-augmented instruction tuning with the Retro encoder activated, especially when high-quality retrieval-augmented instruction tuning data is available.

InsturctRetro demonstrates larger improvement on long-form QA datasets. When comparing the results of InsturctRetro on short-form QA datasets and long-form QA datasets, we observe InsturctRetro 43B demonstrates large
```


### --- Page 0008 ---

```markdown
![Natural Question ACC (EM) before instruction tuning](assets/page_0008_img_1.png)
![Natural Question ACC (EM) before and after instruction tuning](assets/page_0008_img_2.png)

Table 2. Zero-shot evaluation on four long-form QA datasets. We use F1 as the evaluation metric. Car #1 and #2 are short for two annotated car manual datasets. The average relative improvement of InstructRetro 43B across the long-form QA tasks is 10% over InstructGPT 43B.

| Model            | doc2dial | Car #1 | Car #2 | IT Doc |
|------------------|----------|--------|--------|--------|
| Llama 2 RAG 70B  | 32.33    | 49.63  | 45.89  | 25.70  |
| InstructGPT RAG 43B | 32.87    | 50.88  | 31.40  |
| InstructReto 48B | 35.74    | 63.52  | 57.49  | 34.08  |
| (w/o encoder, Avg. +10%) | (35.79) | (63.12) | (61.69) | (34.85) |
| InstructRetro 43B | 35.95    | 63.16  | 56.82  | 34.07  |
| (w/o encoder, Avg. +10%) | (37.95) | (64.17) | (61.67) | (34.50) |

Table 3. From Table 3, we observe that InstructRetro consistently outperforms the InstructGPT RAG on these summarization tasks, especially on the GovReport dataset with 4.87 ROUGE score improvement. Moreover, InstructRetro 43B consistently outperforms Llama 2 RAG 70B across three datasets. This experiment further confirms the generalizability of InstructRetro after instruction tuning and indicates that instruction tuning post retrieval-augmented pretraining yields a better GPT decoder.

5.4. Ablation studies
In this section, we conduct ablation studies to understand the source of improvements for InstructRetro. We find that both retrieval-augmented pretraining and instruction tuning are indispensable to unlock the potential of retrieval-augmented LLMs.

To understand how instruction tuning improves retrieval-augmented pretraining, we show the zero-shot accuracy (Ex-Act Match score) for Retro and GPT on the Natural Question dataset before and after instruction tuning, as detailed in Figure 5. We observe that Retro achieves significantly better zero-shot accuracy than GPT when the number of parameters is relatively small (e.g., 2.25B). However, when scaling the size of parameters, the zero-shot performances of both GPT and Retro start to saturate. We hypothesize that this saturation is mainly due to the poor instruction following capability of both pretrained foundation GPT and Retro.

To remove the instruction-following bottleneck, we apply instruction tuning to further fine-tune both Retro 48B and GPT 43B. Instruction tuning largely mitigates the instruction following bottleneck for both GPT and Retro, resulting in a significant increase of their zero-shot performance on downstream tasks, respectively. Furthermore, once this bottleneck is alleviated, the benefits of retrieval augmentation at pretraining become more pronounced, as InstructRetro excels in leveraging and integrating evidence from retrieved context. Thus, we observe significant improvement.
```

### --- Page 0009 ---

```markdown
# InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

of InstructRetro over InstructGPT$_{\text{RAG}}$ again in Figure 5b. This ablation study confirms that our training recipe - both retrieval-augmented pretraining and instruction tuning - is important for achieving high performance in QA tasks.

## 5.5 Evaluation on MT-Bench

We further evaluate InstructRetro and InstructGPT on the MT-Bench chat benchmark (Zheng et al., 2024) to access Retro’s performance on general chat tasks. One may find more details in Appendix D.

## 6. Conclusion

In this paper, we introduce InstructRetro 48B, the largest LLM with retrieval-augmented pretraining and instruction tuning. Specifically, we start from a pretrained GPT model, and continue to pretrain the model with retrieval, which yields the retrieval-augmented foundation model Retro 48B. After applying instruction tuning to Retro, InstructRetro 48B unveils the potential of retrieval-augmented pretraining and demonstrates significant zero-shot accuracy improvement over its GPT counterpart through our extensive experiments on a wide range of downstream tasks. Moreover, our novel findings show that by using the GPT decoder backbone, i.e., InstructRetro 43B, can achieve comparable performance with models which shed light on providing direction to obtain a better GPT decoder through retrieval-augmented pretraining before instruction tuning.

## Impact Statement

Our InstructRetro, similar to the line of RAG studies, offers significant advancements in addressing the practical deployment and applications of LLMs, particularly in areas of factuality, downstream task accuracy, contextual understanding, and model efficiency. Specifically, it significantly improves the generation of factual and grounded text with retrieval from high-quality databases, which helps mitigate misinformation and enhance public trust. Additionally, InstructRetro boosts the accuracy of LLMs in various downstream tasks and demonstrates better capability in contextual understanding, which is important for reasoning tasks. Remarkably, it achieves comparable performance to LLMs two to three times its size, enhancing computational efficiency and environmental sustainability. This positions InstructRetro as a practical tool in the ethical and safe deployment of LLMs.

## References

Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., Van Den Driessche, G. B., Lespiau, J.-B., Damoc, B., Clark, A. et al. Improving language models by retrieving from millions of tokens. In ICML, 2022.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. NeurIPS, 2020a.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020b.

Chen, M., Chu, Z., Wiseman, S., and Gimpel, K. Summscreen: A dataset for abstractive screenplay summarization. ArXiv, abs/2104.07091, 2021. URL https://api.semanticscholar.org/CorpusID:233240741.

Chowdhery, A., Narang, S., Devlin, J., Bosma, A., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Sugzun, M., Chen, X., Chowdhery, A., Castro-Ros, A., Pellat, N., Robinson, K., Valter, D., Narang, S., Mishra, G., Yu, A., Zhao, V., Hou, Y., Dai, A., Hu, P., Hsiao, E. H., Dean, J., Devlin, J., Roberts, A., Zhou, D. L., Q. V., and Wei, J. Scaling instruction-finetuned language models. arXiv preprint arXiv:2201.11416, 2022. URL https://arxiv.org/abs/2210.11416v5.

Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., and Xin, R. Free Tidy: Introducing the world’s first truly open instruction-tuned lm. databricks, 2023.

Dasigi, P., Liu, R. F., Marasović, A., Smith, N. A., and Gardner, M. Quoref: A reading-comprehension dataset with questions requiring coreferential reasoning. Conference on Empirical Methods in Natural Language Processing, 2019. doi: 10.18653/v1/D19-1606.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Kriukov, M., Zhou, Y., Yu, A., Friat, O., Zoph, B. F., Fedus, L., Bosma, M., Zhou, Z., Wang, T., K. E., Webster, K., Pellat, N., Robinson, K., Meier-Hellstern, K., Duke, T., Dixon, L., Zhang, K., Le, Q. V., Wu, Y., Chen, Z., and Cui, G. Efficient scaling of language models with mixture-of-experts. International Conference on Machine Learning, 2021. URL https://arxiv.org/abs/2112.06905v2.
```

### --- Page 0010 ---

```markdown
| Citation                                                                                                           | Citation                                                                                                           |
|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., and Gardner, M. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. North American Chapter of the Association for Computational Linguistics, 2019. doi: 10.18653/v1/P19-1246. | Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-Yu, J., Joulin, A., Riedel, S., and Grave, E. Atlas: Few-shot learning with retrieval augmented language models. arXiv preprint arXiv: 2208.03299, 2022a. |
| Fan, A., Jernite, Y., Perez, E., Grangier, D., Weston, J., and Auli, M. Eilsf: Long form question answering. Annual Meeting of the Association for Computational Linguistics, 2019. doi: 10.18653/v1/P19-1346. | Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-Yu, J., Joulin, A., Riedel, S., and Grave, E. Few-shot learning with retrieval augmented language models. arXiv preprint arXiv:2208.03299, 2022b. |
| Feng, S., Wan, H., Gunasekara, R. C., Patel, S., Joshi, S., and Lastras, L. doc2dial: A goal-oriented document-grounded dialogue dataset. Conference on Empirical Methods in Natural Language Processing, 2020. doi: 10.18653/v1/2020.emnlp-main.652. | Johnson, J., Douze, M., and Jégou, H. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7 (3):535–547, 2019. |
| Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2021. | Joshi, M., Choi, E., Weld, D. S., and Zettlemoyer, L. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. Annual Meeting of the Association for Computational Linguistics, 2017. doi: 10.18653/v1/P17-1147. |
| Lee, T. H., Ke, Q., and Sun, J. Optimized product quantization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(4):744–755, 2014. doi: 10.1109/TPAMI.2013.240. | Karpu kin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W.-t. Dense passage retrieval for open-domain question answering. In EMNLP, 2020. |
| Gray, R. and Neub, H. Quantization. IEEE Transactions on Information Theory, 44(6):2325–2338, 1998. doi: 10.1109/18.720541. | Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and Lewis, M. Generalization through memorization: Nearest neighbor language models. In ICLR, 2020. |
| Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M. REALM: Retrieval augmented language model pre-training. In ICMI, 2020. | Kim, H., Hessel, J., Jiang, L., West, P., Lu, X., Yu, Y., Zhou, P., Bras, R. L., Alikhani, M., Kim, G., Sam, C., Choi, Y. Soda: Million-scale dialogue distillation with social commonsense contextualization. arXiv preprint arXiv: 2212.10465, 2022. URL https://arxiv.org/abs/2212.10465v2. |
| Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cati, T., Routhier, F., Casas, D. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. | Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization, 2014. |
| Honovich, O., Scialom, T., Levy, O., and Schick, T. Unnatural instructions: Tuning language models with (almost) no human labor. Annual Meeting of the Association for Computational Linguistics, 2022. doi: 10.48550/arXiv.2212.09689. | Kočiský, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann, K. M., Melis, G., and Grefenstette, E. The narrativeqa reading comprehension challenge. Transactions of the Association for Computational Linguistics, 6:317–328, 2018. |
| Huang, J., Ping, W., Xu, P., Shoeybi, M., Chang, K. C.-C., and Catanzaro, B. Raven: In-context learning with retrieval augmented encoder-decoder language models. arXiv preprint arXiv:2308.07922, 2023. | Kudo, T. and Richardson, J. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. Conference on Empirical Methods in Natural Language Processing, 2018. doi: 10.18653/v1/D18-2012. |
| Huang, L. R., Cao, S., Puriain, N. N., Ji, H., and Wang, L. Efficient attentions for long document summarization. arXiv, abs/2104.02112, 2021. URL https://api.semanticscholar.org/CorpusID:233033613. | Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Tutanov, K., Jones, L., Kellecy, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., and Petrov, S. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019. |
```

### --- Page 0011 ---

```markdown
| **Reference**                                                                                                           | **Details**                                                                                     |
|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z.-R., Stevens, K., Barhoum, A., Duc, N. M., Stanley, O., Nagyfi, R. ES., Suri, S., Glushkov, D., Dantu-luri, A., Maguire, A., Schumann, C., Nguyen, H., and Mattick, A. Openassistant conversations - democratizing large language model alignment.  *arXiv preprint arXiv: 2304.03727*, 2023. |                                                                                                 |
| Laurençon, H., Saulnier, L., Wang, T., Akiki, C., Vilanova del Moral, A., Le Secq, T., Von Werra, L., Mou, C., González Ponceferada, E., Nguyen, H., et al. The biscience corpus: A 1.6 tb composite multilingual dataset. *Advances in Neural Information Processing Systems*, 35:31809–31826, 2022. |                                                                                                 |
| Lewis, P., Perez, E., Pitroni, F., Karpu-khin, V., Goyal, N., Küttler, H., Lewis, M., W.-t., Rockẗschel, T., et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *In NeurIPS*, 2020. |                                                                                                 |
| Lin, X. V., Chen, X., Chen, M., Shi, W., Lomeli, M., James, R., Rodriguez, P., Kahn, J., Szul, G., Lewis, M., Zettlemoyer, L., and tau Yih, W. RA-DIT: Retrieval-augmented dual instruction tuning. *In The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?id=2207tubg](https://openreview.net/forum?id=2207tubg) |                                                                                                 |
| Longpre, S., Hou, L., Yu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J., and Roberts, A. The Flan collection: Designing data and methods for effective instruction tuning. *International Conference on Machine Learning*, 2023. doi: 10.48550/arXiv.2301.13688. |                                                                                                 |
| Malkov, Y. A. and Yashunin, V. D. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE transactions on pattern analysis and machine intelligence*, 42(4):824–836, 2018. |                                                                                                 |
| Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H. Cross-task generalization via natural language crowdsourcing instructions. *In ACL*, 2022. |                                                                                                 |
| Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., et al. WebGPT: Browser-assisted question-answering with human feedback. *arXiv preprint arXiv:2112.09332*, 2021. |                                                                                                 |
| OpenAI. ChatGPT. [https://chat.openai.com](https://chat.openai.com), 2022.                                          |                                                                                                 |
| OpenAI. GPT-4 technical report. *arXiv*, 2023.                                                                        |                                                                                                 |
| Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slam, K., Ray, A., et al. Training language models to follow instructions with human feedback. *NeurIPS*, 2022. |                                                                                                 |
| Petroni, F., Piktus, A., Fan, A., Lewis, P., Yazdani, M., De Cao, N., Thorne, J., Jernite, Y., Karpu-khin, V., Mailard, J., Plachouras, V., Rockẗschel, T., and Riedel, S. KILT: a benchmark for knowledge intensive language tasks. *In NAACL*, 2021. |                                                                                                 |
| Rae, J. W., Borgeaud, S., Cai, T., Millikan, K., Hoffmann, J., Song, F., Aslanides, J., Henderson, S., Ring, R., Young, S., et al. Scaling language models: Methods, analysis & insights from training gopher. *arXiv preprint arXiv:2112.11446*, 2021. |                                                                                                 |
| Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Liu, P. J., et al. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140):1–67, 2020. |                                                                                                 |
| Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. Squad: 100,000+ questions for machine comprehension of text. *Conference on Empirical Methods in Natural Language Processing*, 2016. doi: 10.18653/v1/D16-1264. |                                                                                                 |
| Rajpurkar, P., Jia, R., and Liang, P. Know what you don’t know: Unanswerable questions for squad. *Annual Meeting of the Association for Computational Linguistics*, 2018. doi: 10.18653/v1/P18-2142. |                                                                                                 |
| Sanh, V., Webson, A., Raffel, C., Bach, S., Sutawika, L., Alyafea, Z., Chaffin, A., Stiegler, A., Raja, A., Dey, M., Bari, M. S., Xu, C., Thakker, U., Sharma, S. S., Szczecha, E., Kim, T., Chhablani, G., Nayak, N., Datta, D., Chang, J., Jiang, T.-J., Wang, H., Manica, M., Shen, S., Yong, Z. X., Pandey, H., Bawden, R., Wang, T., Neeraj, T., Rozen, J., Sharma, A., Santilli, A., Fevry, T., Fries, J. A., Tehan, R., Scao, T. L., Biderman, S., Gao, L., Wolf, T., and Rush, A. M. Multitask prompted training enables zero-shot task generalization. *In International Conference on Learning Representations*, 2022a. URL [https://openreview.net/forum?id=9Vr9D9W4](https://openreview.net/forum?id=9Vr9D9W4) |                                                                                                 |
| Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafea, Z., Chaffin, A., Stiegler, A., Scao, T. L., Raja, A., et al. Multitask prompted training enables zero-shot task generalization. *In ICLR*, 2022b. |                                                                                                 |
```

### --- Page 0012 ---

```markdown
| Author(s)                                                                 | Year  | Title                                                                                                      | Source                                                                                      |
|---------------------------------------------------------------------------|-------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Shi, W., Min, S., Lomeli, M., Zhou, C., Li, M., Lin, V., Smith, N. A., Zettlemoyer, L., Yih, S., and Lewis, M. | 2023a | In-context pretraining: Language modeling beyond document boundaries.                                     | arXiv preprint arXiv:2310.10638.                                                          |
| Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer, L., and Yih, W.                     | 2023b | Replug: Retrieval-augmented black-box language models.                                                   | arXiv preprint arXiv:2310.12652.                                                          |
| Smith, S., Patwardhan, M., Norick, B., LeGresley, P., Rajbhandari, S., Casper, J., Liu, Z., Prabhume, S., Zerveas, G., Korthikanti, V., Zhang, E., Child, R., Aminabadi, R. Y., Bernauer, J., Song, X., Shoeybi, M., He, Y., Houston, M., Tiwary, S., and Catanzaro, B. | 2022  | Using deepseed and megatron to train megatron-large model.                                              | arXiv, 2022.                                                                              |
| Thoppilan, R., De Freitas, D., Hall, J., Shaheer, N., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos, T., Baker, L., Du, Y., et al. | 2022  | Language models are foundation models.                                                                     | arXiv preprint arXiv:2201.08239.                                                          |
| Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M., Azhar, F., Rodriguez, A., Joliot, A., Grave, E., and Lam, P. | 2023  | Llama: Open foundation and fine-tuned chat models.                                                       | arXiv preprint arXiv:2302.13971.                                                          |
| Tschiher, A., Wang, T., Yuan, K., Harris, J., Sordoni, A., Barham, P., and Suleman, K. | 2016  | Newsqa: A machine comprehension dataset.                                                                   | REPL4NLP@ACL, 2016. doi: 10.18653/v1/W17-2623.                                           |
| Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. | 2017  | Attention is all you need.                                                                                 | NIPS, 2017.                                                                                |
| Wang, B., Ping, W., Xu, P., McAfee, E., Liu, Z., Shoeybi, M., Dong, Y., Kuchaeva, O., Li, B., Xiao, C., et al. | 2023  | Shall we pretrain autoregressive language models with retrieval? A comprehensive study.                  | EMNLP, 2023a.                                                                              |
| Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, A., and Hajishirzi, H. | 2022  | Self-instruct: Aligning language models with self-generated instructions.                                  | arXiv preprint arXiv:2212.10560, 2022.                                                   |
| Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, A., and Hajishirzi, H. | 2023  | Self-instruct: Aligning language models with self-generated instructions.                                  | Meeting of the Association for Computational Linguistics, 2023. doi: 10.48550/arXiv.2212.10560. |
| Wang, Y., Ivison, H., Dasigi, P., Hessel, J., Khot, T., Chandu, K. R., Wadden, D., MacMillan, K., Smith, N. A., Beltagy, I., and Hajishirzi, H. | 2023  | How far can a camel's go? Exploring the state of instruction tuning on open resources.                    | arXiv preprint arXiv:2306.04751, 2023. URL: https://arxiv.org/abs/2306.04751.            |
| Wei, J., Bosma, M., Zhao, Y. V., Guu, K., Yu, A. W., Lester, J., Du, N., Dai, A. M., and Le, Q. V. | 2022  | Finetuned language models are zero-shot learners.                                                          | In ICLR, 2022.                                                                              |
| Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgwardt, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. | 2022b | Emergent abilities of large language models.                                                               | arXiv preprint arXiv:2206.06728.                                                           |
| Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E. L., E. Q. Zhou, D., et al. | 2023  | Chain-of-thought prompting elicits reasoning in language models.                                          | Advances in Neural Information Processing Systems, 35: 24824–24837, 2022.                 |
| Yogatama, D., de Masson d'Autume, C., and Kong, L. | 2021  | Adaptive semiparametric language models.                                                                   | Transactions of the Association for Computational Linguistics, 2021.                       |
| Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. | 2024  | Judging lm-as-a-judge with bench and chatbot arena.                                                      | Advances in Neural Information Processing Systems, 36, 2024.                             |
| Zhong, M., Yin, D., Yu, T., Zaidi, A., Mutuma, M., Sha, R., Awadallah, A. H., Celikyilmaz, A., Liu, Y., Qiu, X., et al. | 2021  | Qmsum: A new benchmark for query-based multi-domain meeting summarization.                               | arXiv preprint arXiv:2104.05938, 2021.                                                    |
```

### --- Page 0013 ---

```markdown
# InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

## A. Details of Pretraining

### A.1. Pretraining corpus

We prepared a pretraining dataset consisting of around 1.2 trillion tokens from English natural language data. Specifically, it consists of web-crawl data from Common Crawl, news data, conversational data, book data (e.g., Book3 and Book-Corpus2 from the Pile dataset (Gao et al., 2020)), scientific and multi-domain data (e.g., Wikipedia and the BigScience ROOTS corpus (Laurencot et al., 2022)).

### A.2. Continued pretraining schedules

Based on pretrained GPT models, we further pretrain Retro with retrieval augmentation on additional 100 billion tokens, which is around 25M samples with sequence length set to 4096. We list the pretraining hyper-parameter details of Retro-fitting in Table 4. GPT-fitting uses the same training schedules as Retro-fitting.

All models use Adam optimizer (Kingma & Ba, 2014) with $\beta_1 = 0.9$ and $\beta_2 = 0.95$. We employ the learning rate (LR) decay schedules with LR warmup samples of 16667 and LR decay samples of 23750000.

| Models Size | LR min | LR Decay Styles | Batch Size | Pretraining Steps |
|-------------|--------|------------------|------------|-------------------|
| 823M       | $2e^{-5}$ | $2e^{-6}$ | cosine | 128 | 195.2k |
| 2.25B      | $2e^{-5}$ | $2e^{-6}$ | cosine | 256 | 97.6k |
| 8.5B       | $1e^{-5}$ | $1e^{-6}$ | cosine | 512 | 48.8k |
| 22B        | $1e^{-6}$ | $1e^{-6}$ | cosine | 512 | 48.8k |
| 43B        | $9e^{-6}$ | $9e^{-7}$ | cosine | 768 | 32.5k |

### A.3. Computational cost

We present the detailed computational cost of the continued pretraining step on additional 100B tokens for both Retro and GPT across different sizes in Table 5. We can see that pretraining Retro brings around additional 35% computational overhead than pretraining GPT, which mainly comes from the Retro encoder and cross-chunk attention to incorporate and fuse the retrieved neighbor information. Moreover, we can see that scaling up the size of Retro does not bring more computational overhead and remains around 35%, shedding light on a promising way to retrieval-augmented pretraining.

A more useful perspective is looking at the overall pretraining cost. Since our Retro 48B starts from a pretrained GPT 43B on 1.1T tokens, it only needed 2.58% additional GPU hours in contrast to pretraining the GPT 43B on 1.2T tokens.

$$
1.1T \times 1 + 0.1T \times (1 + 31\%) = 102.58\%
$$

| GPT | training on 100B token | Retro | training on 100B token | Additional Overhead (on 100B tokens) | Additional Overall Overhead (on 1.2T tokens) |
|-----|-------------------------|-------|-------------------------|--------------------------------------|-----------------------------------------------|
| 823M | 140G GPU Hours         | 878M  | 1920 GPU Hours         | 36%                                  | 3.00%                                         |
| 2.25B | 3226 GPU Hours        | 2.5B  | 4096 GPU Hours         | 27%                                  | 2.25%                                         |
| 8.5B | 1269G GPU Hours        | 9.5B  | 17395 GPU Hours        | 37%                                  | 3.08%                                         |
| 22B  | 3788G GPU Hours        | 24B   | 52152 GPU Hours        | 37%                                  | 3.08%                                         |
| 43B  | 53329 GPU Hours        | 48B   | 69995 GPU Hours        | 31%                                  | 2.58%                                         |

## B. Details of retrieval database

Retrieval Database. We use the whole pretraining corpus as our retrieval database, consisting of 1.2 trillion tokens as mentioned in Appendix A.1. Our pretraining dataset with 1.2 trillion tokens yields a retrieval database consisting of 19B chunks in total with chunk size $m = 64$. To support fast similarity searches with billions of chunks, we implement the database index with Faiss index (Johnson et al., 2019). Given the BERT embeddings of an input chunk $C_i$, Faiss can return the approximate $k$ nearest neighbor of $C_i$ within a few milliseconds.
```

### --- Page 0014 ---

```markdown
# InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

## B.1. Faiss index configuration

We use the Faiss index (Johnson et al., 2019) as the implementation for the dense retriever to search for approximate nearest neighbors in the BERT embedding space. We configure the Faiss index as follows:

- **Preprocessing:** We use Optimized Product Quantization (Ge et al., 2014) to apply a rotation to the input vectors to make them more amenable to PQ coding (Gray & Neuhoff, 1998).

- **Indexer:** We use Inverted File Index (IVF) with $2^{22}$ centroids and accelerate it with Hierarchical Navigable Small World (HNSW) graphs (Malkov & Yashunin, 2018).

- **Encoding:** We adopt PQ encoding that compresses the dense embedding vector into 64 bits.

As a result, we can achieve 4ms per query over the whole pretraining corpus via batch queries averaged for each chunk with less than 1TB memory usage as our max throughput. Given a single query, the latency of the response is around 0.1s per query. We also note that increasing the number of $K$ in the query does not yield slower query speed. During pretraining, we follow Borgeaud et al. (2022) to pre-compute the nearest neighbors and save the data for pretraining.

## B.2. Computational cost on building retrieval database

Building a Faiss index involves several steps. We detail each step with its associated computational cost as below:

- **Embedding the retrieval database into dense BERT embeddings.** Given the chunk size of $m = 64$ tokens, we embed every chunk of text corpus with BERT-large-cased. The computational cost to embed the text corpus is around 6.22M chunks per GPU hour given one A100 GPU. For our 19B chunk database, it takes around 3054 GPU hours in total.

- **Train the Faiss index.** This involves determining a smaller number of centroids to cluster the whole corpus embeddings and initializing the HNSW graph. The computational cost for training the Faiss index depends on the number of centroids and the number of centers. Given our setup, we train the faiss index based on 600M chunks uniformly sampled from the retrieval database. The computational cost of this step is less than 4 hours with one DGX A100 node.

- **Add the embedded corpus to the Faiss index.** After the index has been trained, the index centroids and HNSW graph are determined, but the index itself is still empty. In this step, we add the whole dense corpus embeddings to the index graph structure. The computational cost of adding the corpus to the index is around 192 CPU hours within one DGX A100 node. Moreover, it can be purely done within a CPU node to save computational cost.

- **Query the Faiss index.** As mentioned above, we can achieve 4ms per query over the whole pretraining corpus via batch queries averaged for each chunk with less than 1TB memory usage as our max throughput. The computational cost to this step can also be purely done within a CPU node to save computational cost and can run in parallel to further speed up the querying.

In summary, the overall computational cost of building Faiss index is marginal compared to the pretraining cost, especially considering the benefits of retrieval-augmentation pretraining, which further unlocks the potential of instruction tuning. 

## B.3. Ablation studies on Faiss index configurations

**Faiss training-time configuration.** We conduct ablation studies on the quantization techniques using two index configurations on two datasets: the whole pretraining dataset and the Wikipedia Corpus. We highlight the configuration setup in Table 6 below.

Following the official guide of Faiss¹, we initialize two Faiss indexes based on the sizes of two retrieval databases: the full pretraining corpus with 19B chunks and the Wikipedia corpus with 66M chunks. We applied product quantization (Ge et al., 2014; Gray & Neuhoff, 1998) to the full pretraining corpus to reduce the dimensionality and save the index memory to support loading the full pretraining corpus, while applying uncompressed flat encoding to the Wikipedia corpus as a comparison. We benchmark the querying speed for a batch of 40K dense embeddings and evaluate the query speed for two indexes.

¹ https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
```

### --- Page 0015 ---

```markdown
# InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

## Table 6. Ablation studies on Faiss product quantization (PQ) on two different retrieval databases.

| Retrieval Index for Full Pretraining Corpus | Retrieval Index for Wikipedia Corpus |
|---------------------------------------------|-------------------------------------|
| # chunks                                   | 19B                                 | 66M                                 |
| Configuration                              |                                     |                                     |
| Dimension Reduction                        | OPQ64_128                            | No Reduction                        |
| Approximate Search Encoding                | IVF49194304_HNSW32                  | IVF261244_HNSW32                   |
| Query Speed                                |                                     |                                     |
| K=2                                       | 0.004 s/query                       | 0.01 s/query                       |
| K=20                                      | 0.004 s/query                       | 0.01 s/query                       |
| K=200                                     | 0.0045 s/query                      | 0.01 s/query                       |
| K=2000                                    | 0.004 s/query                       | 0.01 s/query                       |

From Table 6, we can see that applying product quantization can not only help compress the index and save memory usage but also help improve the query speed, which is critical when scaling up the retrieval database. We can also see that increasing the number of $K$ for $K$ nearest neighbor searchers barely impacts the query speed.

Faiss query-time configuration. For our index configuration with inverted file index structures and HNSW graph, the hyper-parameter $nprobe$ and $efSearch$ play important roles in the query time of Faiss, as detailed in Table 7.

## Table 7. Important querying-time hyper-parameters for our Faiss index.

| index type | Index class | runtime parameter | comments |
|------------|-------------|-------------------|----------|
| IVF*, IM12* | IndexIVF*  | $nprobe$          | the main parameter to adjust the speed-recision tradeoff |
| HNSW*      | IndexHNSW  | $efSearch$        | the depth of the HNSW search |

To select a proper set of query-time hyper-parameters with a good tradeoff of recall and speed, we conduct ablation studies with varying $nprobe$ and $efSearch$. Specifically, we use the retrieval index built on the whole pretraining corpus, query the index with randomly sampled 10K chunks from the pretraining corpus, and evaluate the recall accuracy of retrieving the query chunk itself given top-$K = 2000$. The query time and corresponding recall accuracy with different hyper-parameters are shown in Table 8. We thus select $nprobe=4096$ and $efSearch=32$ in our retrieval-augmented pretraining, considering its fast querying time without sacrificing the recall accuracy significantly.

## Table 8. Trade-off between speed and recall accuracy given different query-time hyper-parameters.

| $efSearch$ | $nprobe$ | Time (per query) | Acc (K=2000) |
|------------|----------|-------------------|--------------|
| 256        | 65536    | 21ms              | 1.00         |
| 64         | 16384    | 6ms               | 0.96         |
| 32         | 4096     | 4ms               | 0.93         |
| 16         | 4096     | 2.5ms             | 0.88         |
```

### --- Page 0016 ---

```markdown
# C. Qualitative examples

## C.1 An example From the instruction tuning data

Table 9 below is a data example from the instruction tuning data.

**Table 9.** A sample conversation from our instruction tuning dataset. The black part serves as the context, while the black part is masked as the answer mask, on which the LM loss is computed.

| Speaker   | Conversation                                                                                                                                                                                                                     |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| System    | This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user’s questions based on the context. The assistant should also indicate when the answer cannot be found in the context. |
| User      | My bicycle tire has a puncture and while I have a pump with me, I don’t have a patch repair kit. Is there a way to keep the tire from running out of air until I can get to a bicycle shop?                                     |
| Assistant | Yes, you could use duct tape to seal the puncture.                                                                                                                                                                            |
| User      | Can you give me a tutorial on how to use the duct tape to seal the puncture?                                                                                                                                                 |
| Assistant | Sure, here’s a tutorial on how to use duct tape to seal a puncture in a tire: 1. Start by cleaning the area around the puncture with a damp cloth. 2. Cut a piece of duct tape large enough to cover the puncture, plus a few millimeters of extra. 3. Remove the backing from the duct tape, and press it firmly over the puncture. 4. Rub the tape firmly with your fingers to ensure a good seal. 5. You can also use a roller or a flat item like a ruler to make sure that the tape is pressed firmly against the puncture. 6. Trim away any excess tape, and your puncture should now be sealed. |
```

### --- Page 0017 ---

```markdown
## C.2. An example From the downstream QA dataset: SQuAD 1.1

We re-format all of the QA tasks in the conversational format. An example from the SQuAD 1.1 dataset with the conversational format is shown in Table 10 below.

### Table 10. A sample conversation prompt template from the SQuAD 1.1 dataset.

| Speaker   | Conversation                                                                                                                                                                                                                                                                                                                                 |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| System    | System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.                                                                 |
| User      | title: , source: The pound-force has a metric counterpart, less commonly used than the newton: the kilogram-force (kgf) (sometimes kilopond), which is the force exerted by standard gravity on one kilogram of mass. The kilogram-force leads to an alternate, but rarely used unit of mass: the metric slug (sometimes mug or hyl) is that mass that accelerates at 1 ms-2 when subjected to a force of 1 kgf. The kilogram-force is not a part of the modern SI system, and is generally deprecated; however it still sees use for some purposes as expressing aircraft weight, jet thrust, bicycle spoke tension, torque wrench settings and engine output torque. Other arcane units of force include the sthené, which is equivalent to 1000 N, and the kip, which is equivalent to 1000 lbf. <br> Based on the above article, answer a question. What is the seldom used force unit equal to one thousand newtons? |
| Assistant | The answer is                                                                                                                                                                                                                                                                                                                                  |
```

### --- Page 0018 ---

```markdown
## D. Experimental results on MT Bench

We evaluate InstructRetro and InstructGPT on the MTBench chat benchmark (Zheng et al., 2024) to understand Retro performance on general chat tasks. We use the Tulu-v2 (Wang et al., 2023b) alignment dataset as the instruction tuning dataset to enhance the chat capabilities for both pretrained GPT model and Retro model. The detailed breakdown of the MT-Bench result is shown in Table 11.

### Table 11. Performance comparison of MT-Bench models

| MT-Bench | InstructRetro-Tulu-v2-43B | InstructGPT-Tulu-v2-43B |
|----------|----------------------------|---------------------------|
| Writing  | 8.85                       | 8.15                      |
| Roleplay | 7.75                       | 7.80                      |
| Reasoning| 5.40                       | 4.75                      |
| Math     | 3.15                       | 2.35                      |
| Coding   | 3.40                       | 4.10                      |
| Extraction| 6.80                      | 6.75                      |
| STEM     | 8.58                       | 8.53                      |
| Humanities| 9.68                      | 9.10                      |
| Turn 1   | 6.89                       | 6.67                      |
| Turn 2   | 6.51                       | 6.21                      |
| Avg      | 6.70                       | 6.44                      |

In Table 11, we show that on average InstructRetro outperforms InstructGPT across different turns. We also observe that Retro performs slightly lower than GPT in domains such as role play and coding. We think that the main reason is that we do not cover coding and role-playing related datasets in the pretraining dataset of the base GPT model, and thus retrieval-augmented pretraining could make little difference.

## E. Potential Negative Social Impacts

In this section, we discuss a few potential negative social impacts shared by the current line of LLM research. First, similar to other very capable LLMs, InstructRetro can generate non-factual but persuasive text across a wide range of topics. This ability can be maliciously exploited to create and spread disinformation or misinformation at scale. Second, InstructRetro is trained on vast datasets collected from the internet, which may include personal or private information. In addition, the retrieval database used in InstructRetro may contain private information as well. Third, although the Retro framework was found to be effective in reducing toxic generations (Wang et al., 2023a), it still reflects and can amplify the biases present in the training data as other LLMs.
```


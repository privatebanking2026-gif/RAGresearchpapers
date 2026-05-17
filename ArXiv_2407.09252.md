# ArXiv 2407.09252

### --- Page 0001 ---

```markdown
# Context Embeddings for Efficient Answer Generation in RAG

David Rau¹  
University of Amsterdam  
Amsterdam, Netherlands  
d.m.rau@uva.nl  

Hervé Déjean  
Naver Labs Europe  
Grenoble, France  
herve.dejean@naverlabs.com  

Shuai Wang†  
The University of Queensland  
Brisbane, Australia  
shuai.wang@uq.edu.au  

Stéphane Clinchant  
Naver Labs Europe  
Grenoble, France  
stephane.clinchant@naverlabs.com  

---

## ABSTRACT

Retrieval-Augmented Generation (RAG) allows overcoming the limited knowledge of LLMs by extending the input with external information. As a consequence, the contextual inputs to the model become much longer which slows down decoding time directly translating to the time a user has to wait for an answer. We address this challenge by presenting COCOM, an effective context compression method, reducing long contexts to only a handful of Context Embeddings speeding up the generation time by a large margin. Our method allows for different compression rates trading off decoding time for answer quality. Compared to earlier methods, COCOM allows for handling multiple contexts more effectively, significantly reducing decoding time for long inputs. Our method demonstrates an impressive speed-up of up to 5.69x while achieving higher performance compared to existing efficient context compression methods. Model checkpoints: [https://huggingface.co/naver/ecom-v1-128-mistral-7b](https://huggingface.co/naver/ecom-v1-128-mistral-7b).

---

## KEYWORDS

Context Compression, LLM, RAG

---

## ACM Reference Format:

David Rau, Shuai Wang, Hervé Déjean, and Stéphane Clinchant. 2024. Context Embeddings for Efficient Answer Generation in RAG. In ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

---

## 1 INTRODUCTION

Large Language Models (LLMs) are pre-trained on massive amounts of textual data; for instance, Llama 2 [32] has been trained on 3 trillion tokens during pre-training. Through billions of learnable parameters, LLMs not only excel at modeling language but at the same time, build up a knowledge base that could be later used for question answering. On the other hand, the model is limited to the knowledge contained in the pre-training data. In knowledge-intensive scenarios, relying solely on the parametric memory of the model is often insufficient. To alleviate this, context can be provided explicitly from an external source through a preceding retrieval step (Retrieval-Augmented Generation–RAG). Although LLMs show notable improvements when given additional relevant context in knowledge-intensive tasks, this approach has limitations. A key drawback is that adding more context to the input considerably slows down generation during inference. This occurs because the self-attention mechanism in transformers grows exponentially in space and memory requirements with increasing input length. At the same time, previous research has shown providing multiple documents as context can improve RAG performance [10, 12]. This is particularly critical for QA applications where reasoning over context for multiple documents is necessary, such as multi-doc QA tasks [7, 16, 35]. In fact, the observation that modern transformers can naturally cope with many context documents for answer generation in open domain QA tasks was central to the development of RAG [6, 11]. However, as the input length becomes larger,

---

![Figure 1: COCOM: Compressing multiple contexts for RAG into a small set ($\mathcal{S} \in \{4, 16, 128\}$) of Context Embeddings leads to a massive speed up in answer generation while maintaining higher performance compared to other methods. Results are shown for the ASQA dataset.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
# Conference'17, July 2017, Washington, DC, USA Rau et al.

the position bias in LLMs might further complicate the extraction of relevant information [21].

Previous work has shown that the increased generation time in RAG can be alleviated by reducing the model’s input through context compression. This can be achieved either by applying lexical-based compression, where unimportant terms or tokens in the context are identified and filtered out during generation [13], or by embedding-based compression, where embedding models transform the context into fewer embedding tokens in the LLM input [3, 8, 24, 31]. Notably, state-of-the-art embedding-based compression methods often achieve higher effectiveness and lower latency compared to lexical-based compression methods [3].

However, despite the current embedding-based compression approaches achieving lower latency in RAG systems, several limitations remain:

| **Limitation**                | **Description**                                                                 |
|-------------------------------|---------------------------------------------------------------------------------|
| Large compressor model        | These methods rely on large compression models to achieve high effectiveness, such as [23, 24]. |
| Low effectiveness             | The effectiveness of current embedding-based compression methods underestimates the potential of LLMs for answer generation, as they only tune parts of model components and leave the decoder LLM untouched. We hypothesize that freezing the decoder hinders the use of compressed contexts. |
| Fixed compression rate        | Current methods do not offer different compression rates with respect to the length of context, which may detract from their generation quality at high effectiveness. |
| Single document limitation     | Current effective methods only support using a single document context to generate answers. |

We address the described limitations, similar to concurrently developed methods, by compressing contexts into a small number of context embeddings which are then provided as input to the LLM. This allows us to reduce the input size to a fraction of its surface form, which has led to an increased decoding time for answer generation. We call our model COCOM (Context Compression Model), a multi-context compression method leveraging a single model for context generation and answer generation.

Additionally, we further show that with appropriate pretraining and tuning approaches, our compressing model achieves significantly higher effectiveness than current context compressing approaches (see Figure 1). We summarize our contributions as follows:

- We present COCOM, an effective context compression method, reducing long contexts to only a handful of context embeddings spending up the generation time while achieving higher performance compared to other methods.
- In an efficiency study, we demonstrate the efficiency-effectiveness trade-offs achievable with different compression rates. We further illustrate the time and memory required for compression. We reduce inference time by up to 5.69 × and GFLOPs by up to 22 × while maintaining high performance.

The rest of this paper is structured in the following way. Section 2 discusses related work on RAG, efficiency, and compression approaches. We continue in Section 3 discussing the RAG task and our novel COCOM approach to effective context compression. Section 4 details the experimental setup in terms of the RAG models and the five QA tasks. In Section 5, we present the main COCOM results in terms of effectiveness and efficiency. Section 6 conducts further analysis of how compression affects the model. We end with discussion and conclusions in Section 7, and limitations in Section 8.

## 2 RELATED WORK

In this section, we discuss related work on RAG, efficiency, and compression approaches.

The initial motivation for this work stems from a recent study by Morris et al. [23], which demonstrates that a bag-of-words representation of the original surface terms can be recovered from text embeddings. This observation that embeddings can encapsulate the content of an entire passage inspired the idea to provide context in the form of an embedding rather than the original context in token form to an LLM.

The underlying motivation in the context of RAG to reduce the input size is, as mentioned earlier, due to the computational cost of contextualizing long inputs and as a consequence thereof increased generation time [1]. We address this by reducing the context to only a handful of context embeddings that are provided the LLM head-on.

Reducing the input to RAG models is an active research field, with many works being done concurrently with ours. Among those works, two primary lines of research have emerged: embedding-based and lexical-based context compression. We discuss them in the following:

### 2.1 Lexical-based Compression

Lexical-based compression focuses on either selecting tokens from the context [20] or summarizing contexts [33], both aiming to retain essential information while reducing overall context size. LLMUiniga comprises a query-dependent token filtering module that uses a LLM to first select important tokens in the context. Then, a query-dependent token classifier is used to select tokens to form the compressed context.

On the other hand, Zhu et al. [36] do not consider compression at the term level, but at the document level. Retrieved documents are either included or excluded with respect to the query. Only the included documents form the context for answer generation. It is worth noting that current lexical-based compression approaches all rely on specific query inputs. Therefore, compression needs to be virtually processed online not allowing to compress documents offline, slowing down generation time.

### 2.2 Embedding-based Compression

Embedding-based compression approaches focus on compressing the context into one or multiple summary embeddings that can be directly interpreted by the decoder model. This first work of
```


### --- Page 0003 ---

```markdown
# Context Embeddings for Efficient Answer Generation in RAG
Conference'17, July 2017, Washington, DC, USA

## Table 1: Comparison to previous works on Embedding-based Context Compression.

| Work                | Light Compressor | Decoder Tuning | Adaptable | Multi-Doc | Efficient Answer Generation |
|---------------------|------------------|----------------|-----------|-----------|-----------------------------|
| GridLM [24]         | ✗                | ✓              | ✓         | ✗         | ✗                           |
| AutoCompressor [4]  | ✓                | ✓              | ✓         | ✗         | ✗                           |
| ICAE [8]            | ✗                | ✗              | ✓         | ✓         | ✓                           |
| xRAG [3]           | ✗                | ✗              | ✓         | ✓         | ✓                           |
| COCOM-light (ours)  | ✗                | ✓              | ✓         | ✓         | ✓                           |
| COCOM (ours)        | ✗                | ✓              | ✓         | ✓         | ✓                           |

this line is called AutoCompressor [4]. This approach attempts to compress contextual information by segmenting it into randomly segmented chunks, subsequently aggregating these into summary embeddings through an iterative process until target embedding size is met. However, the training of the summary embeddings relies exclusively on new token prediction tasks, raising concerns about their ability to effectively encapsulate relevant contextual data. Furthermore, AutoCompressor is designed primarily for long contexts, generating a minimum of 50 summary embeddings. Such a configuration is not suitable for RAG pipelines where short passages are retrieved, such as KILT.

Building up on AutoCompressor, ICAE by Ge et al. [8] explores training a context compressor using the same LLM as the decoder model, and compress only once to get the summary embeddings. However, their approach limits the model’s capacity by using a frozen decoder, preventing the accumulation of gradients from the decoder part during training. In this paper, we argue that decoder training is an important factor that strongly impacts the performance of the model. We illustrate this argument in Section 4.2.

Furthermore, GridLM Muenchhof et al. [24] addresses the issue of double decoding the same context first for retrieval and then again as the provided context to the LLM. They use the same LLM for ranking and generation which allows them to cache all representations during encoding the context and to reuse them during generation. This approach compared to ours is limited to only a single context, does not speed up decoding time, and results in very large storage requirements.

Cheng et al. [3] propose xRAG concurrently to our method. They directly reuse frozen ranking representations based on embedding models while freezing the decoder. Although this approach successfully resolves the double decoding problem, it suffers from low effectiveness because the representation is not trained prior to its application to compression tasks. This issue becomes particularly challenging when light-weight encoder models, such as DPR with 109 million parameters, are used as compressors. In such cases, the model achieves similar effectiveness to the Mistral-7b model when retrieval is not applied¹. On the other hand, using retrieval representations for lightweight models for compression is counter-intuitive. Representations gathered from retrieval tasks may lack sufficient information to fully recover the context. Conversely, representation learned for compression demonstrates its capacity to reconstruct the original context [8]. This suggest that, upon further adjustment, it may show a higher potential to serve as an effective retriever.

### 2.3 Overview
In Table 1 we contrast our method with the described related works on embedding-based compression. It is important to note that most previous works mentioned so far have only considered cases that may not directly apply to RAG settings but rather for long-context question answering. In their setting, only one relevant document is used for each query to fulfill the user request.

Therefore, such models are not naturally able to deal with effectively multiple documents. Furthermore, their reported effectiveness may not directly indicate the final performance in RAG systems, where the document may be potentially irrelevant, and often multiple top-retrieved documents are used. As a decoder model, by design, we should be able to handle multiple context representations, we argue that fine-tuning the decoder is a simple yet necessary solution compared to existing works.

## 3 METHODOLOGY
In this section, we detail the RAG task and our novel COCOM approach to effective context compression.

### 3.1 Task Definition: RAG
RAG employs a ranking system $\mathcal{R}$ and a parametric generative language model $\mathcal{L}_{LM}$, where the ranking system can be multi-staged. First, the ranking system builds a search index $I$ based on a collection. Then, at request time, the index $I$ is searched yielding context segments $C$ that are relevant to the user input $x$: 

$$
f_{\mathcal{R}}: \{x\} \rightarrow C.
$$

Next, the LLM generates a response $r$ based on the context $C$ and user input $x$:

$$
\theta_{LLM}: \{C, x\} \rightarrow r \tag{1}
$$

Note how in RAG the context is added to the input of the LLM dramatically increasing the input to the LLM, as $|C| \gg |x|$.

### 3.2 COCOM: Effective Context Compression
The main idea of COCOM is to enhance efficiency by compressing the context, which is typically given as surface forms as input tokens into a smaller set of context embeddings which then serve as the input to the LLM. An overview of our entire pipeline is given in Figure 2. More formally, our approach can be described as follows:

²The segments can be at different granularities for instance sentences, passages, or entire documents. In this work, we focus on passages.
```

### --- Page 0004 ---

```markdown
![Overview of our COCOM (-light) model pipeline](assets/page_0004_img_1.png)

Given a context $C$ tokenized into a sequence of tokens $\{t_1, t_2, \ldots, t_n\}$, a compressor model $\phi_{cmp}$, we compress $C$ into context embeddings $E$, a smaller set of embeddings $\{e_1, e_2, \ldots, e_k\}$, where $k \ll n$. Each embedding $e_i \in \mathbb{R}^d$, with $d$ being the LLM's hidden dimension.

$$
\phi_{cmp} : \{t_1, t_2, \ldots, t_n\} \rightarrow \{e_1, e_2, \ldots, e_k\} \in \mathbb{R}^d \tag{2}
$$

Next, based on the compressed context embeddings $E$ and the user input to the LLM $\phi_{LLM}$ generates a response $r$:

$$
\phi_{LLM} : \{E, x\} \rightarrow r \tag{3}
$$

The $\phi_{cmp}$ model is trained to generate context embeddings that capture the content of the input tokens in a compressed form. As both models are trained jointly, $\phi_{LLM}$ learns to decode these context embeddings, extracting the relevant information required to answer user queries.

COCOM compresses the context-embeddings question independently. This means not only do individual contexts have to be contextualized by an LLM only once, but they can also be pre-computed offline and stored, drastically reducing computational costs of the LLM at inference time. Further, by only feeding a small number of context embeddings instead of the long context, the input size is reduced to a fraction leading to a massive speed-up for answer generation.

For COCOM, we utilize the same model for compression and answer generation $\phi_{cmp} = \phi_{LLM}$. Therefore, we effectively train a single model on the two tasks. For the compression task, we prepend a special token <EOT> to the input and depending on $\xi$ append a different number of context embedding tokens <CTX> at the end of the sequence. We directly use the representations of the last hidden layer as our context embeddings as input - to the same model - for the answer generation.

As demonstrated later in the experiments, our method also allows us to potentially employ any embedding model as a compressor; including more lightweight encoder-only models such as BERT ³.

### 3.2.1 Adaptable Compression Rate

The number of context embeddings $k = |E|$ can be varied and allows to control the level of compression of the original context $C = \{t_1, \ldots, t_n\}$. We calculate the number of context embeddings $k$ per context $C$ based on a compression rate $\xi$ and the length of the tokenized input $n = |C|$.

$$
\xi = \frac{n}{|E|} \tag{4}
$$

For instance, when compressing a context with length $n = 128$ and a compression rate $\xi = 4$ we obtain 32 context embeddings, reducing the input by 64 times.

### 3.2.2 Multiple Contexts Embeddings

We propose two auto-regressive variations of the next-token prediction task to learn to compress context into embeddings and to use these context embeddings as input to the LLM. Following our earlier notation, the objective function for the standard next token prediction for input $\mathbf{x} = \{x_1, x_2, \ldots, x_T\}$ can be written as:

$$
L(\phi_{LLM}) = - \sum_{x_t \in \mathcal{X}} \log \phi_{LLM}(x_t | x_1, x_2, \ldots, x_{t-1}) \tag{5}
$$

### 3.3 Pre-training Context Embeddings

We modify the next token prediction task to recover the original input tokens from the compressed context embeddings $E$. This way we jointly train the...
```

### --- Page 0005 ---

```markdown
# Context Embeddings for Efficient Answer Generation in RAG

$$
E_\phi = \phi_{\text{com}}(x_1, x_2, \ldots, x_T) \tag{6}
$$

$$
L(\theta_{LLM}, \phi_{\text{com}}) = - \sum_{x_k \in X} \log P_{\theta_{LLM}}(y | E_k, x_1, \ldots, x_{k-1}) \tag{7}
$$

This task serves as a preliminary step toward our final objective of answering questions based on context embeddings. For this objective, we first aim to learn to compress and decompress the same input effectively.

### 3.3.2 Language Modeling from Context Embeddings

Our final task is to answer questions based on the context embeddings. To this end, in our language modeling task, we train the model to continue a given input conditioned on context embeddings. This way the model learns not only to compress a given input but also to leverage the content of the context embeddings effectively.

We split input $X = \{x_1, x_2, \ldots, x_T\}$ into $X_A = \{x_1, \ldots, x_j\}$ and $X_B = \{x_{j+1}, \ldots, x_T\}$. After compressing the first part $X_A$ into $E_A$, we learn to generate the conditional - namely the second part $X_B$ - based on the compressed representations $E_A = \phi_{\text{com}}(X_A)$. This can be seen as a variation of the next token prediction task but conditioned on context embeddings.

$$
L(\theta_{LLM}, \phi_{\text{com}}) = - \sum_{r_k \in R} \log P_{\theta_{LLM}}(y | \phi_{\text{com}}(X_A), x_1, \ldots, x_{k-1}) \tag{9}
$$

## 4 EXPERIMENTAL SETUP

In this section, we detail our experimental setup in terms of the RAG models and the five QA tasks.

### 4.1 Implementation Details

We use Mistral-7B-Instruct-v0.2 as our backbone LLM for answer generation. For context compression in COCOM, we utilize the same model. For our more light-weight context compression, in COCOM-light, we employ bert-base-uncased. We apply three different compression rates $r = 1, 16, 128$. We employ SPADLE-v3 [18] with re-ranking top-$k$ using DeBERTa-v3 [9] as our retrieval system. For all our experiments we use the top-5 documents as context. We release our strongest model checkpoints on Huggingface.

### 4.2 Training

For both pre-training and fine-tuning, we apply parameter-efficient LoRA tuning.

#### 4.2.1 Pre-training

For our pre-training, we employ the two earlier mentioned pre-training autoencoding and language modeling tasks. Samples are drawn randomly with equal probability from both tasks. We first define our dataset but found this to perform best to ensure efficient batch processing, which requires that every sample in a batch contains a fixed-length tokenized input. Thus we split the Wikipedia-KILT [27] corpus into chunks of 128 tokens using the Llama-2-7b tokenizer. We pre-train on 101m samples. Training hyperparameters can be found in the Appendix in Table 10.

#### 4.2.2 Fine-tuning

The BERGEN library [29] is used to fine-tune the model. We fine-tune our models on various datasets concurrently. To construct our fine-tuning dataset *, we combine training samples from Natural Questions [17], MS MARCO [25], adversarial QA [2], HotpotQA [35], WikiQA [34], SQuAD [15], ASQA [30], and PopQA [22].

##### 4.3 Evaluation

We evaluate our model on several widely used QA datasets. Natural Questions [17], TriviaQA [16], HotpotQA [35], ASQA [30], and PopQA [22].

###### 4.3.1 Metrics

As our main metric, following the standard protocol to evaluate fine-tuned models we use Exact Match (EM). To compare our results to previous works, which partially rely on untuned decoders and therefore produce verbose answers, we revert to the Match metric (M), which indicates whether the label is contained (as an exact match) in the generated answer.

### 4.4 Baselines without Context Compression

We fine-tune the base model (Mistral-7B-Instruct-v0.2):

- RAG - upper bound. The model receives the top-5 retrieved contexts, alongside the query and answers the question. This model serves as an upper bound in our experiment not applying context compression.
```


### --- Page 0006 ---

```markdown
# Conference'17, July 2017, Washington, DC, USA
Rau et al.

## 4.5 Baselines with Context Compression

We compare our models to the context compression methods mentioned below. As mentioned earlier these models tune only parts of their model components on the downstream data but leave their decoder LLM untuned applying it zero-shot. We argue this to be a major limitation, as answering questions from context embeddings differs fundamentally from the standard language modeling hindering the model to effectively leverage the context embeddings.

To ensure comparability among approaches we use the same retrieval system as mentioned earlier in Section 4.1.

- **Autocompressor** [4]: We use the Princeton-ai/AutoCompressor-Llama-2-7b-ckpt checkpoint producing 50 summary vectors. As their model is limited to compressing one single context, we just use the top retrieved document as context.
- **ICAE** [8]: We use the Mistral-7B-Instruct-v2.0-Lra-checkpoint which uses the same base LLM as ours and is therefore directly comparable. ICAE is fine-tuned to compress a single long context, however, in our work we use multiple contexts. To alleviate this we concatenate the top five retrieved contexts together as the context input for the model and truncate as the maximum length of 512 tokens. Note the model has a maximum output length of 128 compressed tokens, which is our original concatenated context input.
- **xRAG**: We utilize the xRAG-7b-1, and 8x7b mixture-of-experts model trained alongside their strongest SFR compressor. The base model is again the same as ours, to ensure comparability. As their model is limited to compressing a single context into a single compressed representation, we use the top retrieved context for the xRAG setting. We apply their predefined dropping criteria for the answer generation, which aims at cutting the verbose nature of a untrained decoder LLM.

## 5 RESULTS

In this section, we present the main COCOM and COCOM-light results in terms of effectiveness and efficiency.

### 5.1 Main Results

The main results for COCOM are presented in Table 2. We measure performance following the standard practice for fine-tuned models using the Exact Match (EM) metric. Compared to existing context compression methods [4], our approach demonstrates a significantly (Tested with paired t-test (p<0.05)), higher effectiveness across different compression rates for all datasets tested. COCOM even outperforms the much stronger xRAG Mistral-8x7b model by a large margin having 8 times more parameters than COCOM. The highest performance is observed at a low compression rate ($\epsilon = 4$). Increasing the compression rate results in a slight performance decline, which we will analyze further in Section 6.1.

Compared to our upper bound baseline RAG without compression, we reduce the context by up to 128 times while still maintaining relatively high performance on average over datasets.

Performance decreases on average 4 points for our strongest model (COCOM $\mathcal{F} = 4$) and 10 points for the highest compression rate (COCOM $\mathcal{F} = 128$). Compared to our lower bound baseline LLM without provided context we gain up to 17 points, adding only a small number of additional context embeddings to the input.

Note, while EM is a standard metric for evaluating tuned models, it might underestimate zero-shot decoder methods that do not adapt the model to generate answers. To address this, we also provide results using the Match metric in the appendix in Table 9. Although models that do not tune their decoder achieve relatively higher performance when measured in Match, our model's effectiveness compared to other methods still remains consistently significantly higher.

Overall, considering the effectiveness and the efficiency gains from context compression (discussed further in Section 5.3), COCOM shows a very favorable trade-off.

### 5.2 COCOM-light

To alleviate the dimensional mismatch between the bert-based compressor and the - typically larger - LLM, we learn a linear projection layer $W_{sxd}$, where $s$ is the compressed size, $d$ is the hidden dimension of BERT, and $d$ the hidden dimension of the LLM. To obtain a set of Context embeddings we leverage the last hidden representation of each input token. We simply split the hidden representations into blocks of length $f$ and project each block into a single Context Embedding. Thus, we learn a block-wise aggregation of the input representations that depending on the input length, and the compression rate $f$ yields a different number of Context Embeddings per input. Note that a similar approach is applied in xRAG, where a projection layer is used on the embedding to resolve the dimension mismatch. However, we argue that compressing using a single vector embedding could significantly restrict the compression quality, especially using lightweight encoder models such as BERT. This restriction can result in much lower effectiveness compared to using a larger model [3].
```


### --- Page 0007 ---

```markdown
| **Table 2**: Results in Exact Match (EM) comparing COCOM (-light) to other context compression works. For Match metric (M) see Table 9 in Appendix. All methods use 5 context passages unless indicated otherwise. * Method limited to single context. ⊕ upper baseline. ↓ lower baseline. * indicates statistical non-significance (p<0.05) with respect to COCOM (-4). |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| **Zero-shot**   |                | **NQ**         | **TriviaQA**   | **HotpotQA**   | **ASQA**       | **PopQA**      | **Average**    |
| **Decoder**     | **Method**     | **Compression rate (ξ)** |                |                |                |                |                |
| Mistral-7B-v0.2 | AutoCompressor [4] ★ | × 4          | 0.000          | 0.000          | 0.000          | 0.000          | 0.000          |
|                 | ICAE [8] ★    | × 4            | 0.210          | 0.592          | 0.184          | 0.222          | 0.290          |
|                 | xRAG [3]      | × 128          | 0.184          | 0.622          | 0.185          | 0.199          | 0.274          |
|                 |                | × 128          | 0.265          | 0.744          | 0.239          | 0.292          | 0.318          |
|                 | RAG (no compression) | -      | 0.597          | 0.883          | 0.500          | 0.622*         | 0.514          |
|                 | LLM* (without context) | -    | 0.359          | 0.708          | 0.264          | 0.516          | 0.199          |
|                 | COCOM (-light) | × 4           | 0.539          | 0.849          | 0.490          | 0.621*         | 0.531          |
|                 |                | × 16           | 0.492          | 0.823          | 0.367          | 0.565          | 0.385          |
|                 |                | × 128          | 0.444          | 0.794          | 0.321          | 0.550          | 0.314          |
|                 | COCOM (ours)  | × 4            | 0.554          | 0.859          | 0.430          | 0.679          | 0.474          |
|                 |                | × 16           | 0.539          | 0.852*         | 0.426*         | 0.465          | 0.577          |
|                 |                | × 128          | 0.511          | 0.835          | 0.378          | 0.585*         | 0.391          |

| **Table 3**: Decoding efficiency in generation time, GPU Memory, and number of operations (GFLOPs) for COCOM (-light) on dataset NQ. ξ the compression rate. Efficiency speedup compared against RAG (no compression) is indicated in brackets. |
|----------------|----------------|----------------|----------------|----------------|
| **Model**      | **ξ**          | **Decoding Time (ms)** | **GPU Memory (GB)** | **GFLOPs**     |
| Mistral-7B-v0.2 |                |                |                |                |
| RAG (no comp.) | 1064           | 18.1           | 25031          |
| LLM (no context) | 159          | 14.1           | 207            |
| COCOM (-light) | × 4            | 15.1 (× 2.87)  | 7016 (× 3.57)  |
|                 | × 16           | 213 (× 5.00)   | 4.129 (× 16.16)|
|                 | × 128          | 128 (× 2.17)   | 22.00          |

| **Table 4**: Compression efficiency and storage requirements. Compressing 24m contexts using an A100 80GB GPU. |
|----------------|----------------|----------------|----------------|
| **Compression** | **ξ**         | **Time (h)**   | **Index Size (TB)** |
| COCOM          | 4              | 89             | 6.06               |
|                 | 128            | 77             | 1.51               |
| COCOM-light    | 16             | 1.51           | 1.51               |
|                 | 128            | 1               | 0.19               |

We show our efficiency results for answer generation in Table 3 for different compression rates ξ and compare them to RAG without context compression. Context compression with COCOM reduces answer generation time, GPU memory, and the number of operations drastically with up to 5.69× less inference time cost, 1.27× GPU memory, and 22× GFLOPs compared to no compression. In addition, Table 4 presents the compression costs for all documents in the kit-100w (24m contexts) collection using COCOM-light models at various compression rates. COCOM-light models demonstrate significantly faster compression speeds compared to the COCOM model by employing a much computationally lighter compressing module (up to 89×). Index size varies inversely with compression rate: higher compression rates result in smaller index storage requirements. However, this trade-off leads to lower quality in answer generations, as shown in Table 2.

### 5.4 Ablations
In the following section, we run additional ablation experiments for COCOM and COCOM-light. Most results can be found in Table 6. We report performance in Exact Match on two datasets (NQ and ASQA).
```

### --- Page 0008 ---

```markdown
# Table 5: Impact of the number of provided contexts ($k$) on COCOM measured in EM on datasets NQ and ASQA.

| Model  | $\xi$ | $k=1$ | $k=5$ |
|--------|-------|-------|-------|
| COCOM  | 4     | 0.499 | 0.554 | 0.558 | 0.609 |
|        | 16    | 0.491 | 0.539 | 0.541 | 0.602 |
|        | 128   | 0.482 | 0.511 | 0.544 | 0.585 |

## 5.4.1 Handling multiple contexts.
In Table 5, we compare the performance of COCOM with 1 retrieved context ($k = 1$) versus our default setup $k = 5$. On both datasets and for all compression rates, we observe a substantial gain when using more contexts. Moreover, COCOM with 1 retrieved context is still significantly better compared to other baselines relying on single retrieved document (ICEA, xRAG) in Table 2. As a decoder model by design should be able to handle multiple context representations, we argue that fine-tuning the decoder is a simple yet necessary solution compared to existing works.

## 5.4.2 Pre-training Context Compression.
Central to our approach is the compression of context into a small number of Context embeddings. We argue that context compression fundamentally differs from the language modeling objective on which the model was originally trained. Consequently, we have employed auto-encoding and language-modeling-from-context-embedding tasks to learn how to effectively compress the context and utilize the compressed representations during decoding. We show the results of the impact of the pre-training tasks on the downstream performance after fine-tuning. Our results suggest that the dedicated pre-training tasks for context compression can improve performance for downstream QA performance, subsequently exposing possible explanations. Either context compression is too complex to be learned concurrently with the downstream task, or larger fine-tuning datasets are necessary to effectively learn how to compress contexts.

## 5.4.3 Pre-training Corpus.
Our method employs an initial pre-training stage aimed at initializing context compression. We train auto-revisely on the same target corpus, which is later used to retrieve relevant contexts. In this experiment, our objective is to assess how variations in the pre-training corpus impact downstream QA performance, thereby testing the robustness of our approach. To explore this, we additionally re-train the model on the "sample-10B" subset of FineWeb [26]. We employ the same training methodology described in Section 4.2, where we segment the collection into non-overlapping passages of 128 tokens using the Llama-2-7b tokenizer and train on a subset of 10 million tokens, similar to the target corpus. The results presented in Table 6 indicate a slight decrease in performance when using a different target corpus for pre-training. Nonetheless, our approach demonstrates robustness in handling variations in the pre-training corpus, highlighting its adaptability and effectiveness in context compression.

## 5.4.4 Decoder LLM Tuning.
Existing context compression methods do only the compression while producing the decoder, responsible for generating the answer, frozen. A core distinction from these methods is that we tune all components including the decoder, in COCOM. We hypothesize that context embeddings differ significantly from the input token embeddings the model was trained on, thereby hindering effective utilization without dedicated tuning. We investigate the consequences of freezing the decoder and solely tuning the compressor, akin to existing methods. Our findings show the criticality of tuning the decoder to achieve high effectiveness. This reinforces our hypothesis that specific tuning of context embeddings is essential for better performances.

# Table 6: Impact of pre-training corpus, pre-training, and decoder tuning on downstream performance (EM). Compression rate $\xi = 128$.

| Ablation                | Datasets  |
|------------------------|-----------|
|                        | NQ        | ASQA      |
| COCOM-light (baseline) | 0.444     | 0.550     |
| w/o pre-training       | 0.423     | 0.524     |
| pre-training on FineWeb| 0.427     | 0.545     |
| w/o tuning decoder     | 0.353     | 0.438     |
| COCOM (baseline)      | 0.519     | 0.585     |
| w/o pre-training       | 0.490     | 0.565     |
| pre-training on FineWeb| 0.503     | 0.581     |
| w/o tuning decoder     | 0.421     | 0.521     |

![Impact on zero-shot transferability of fine-tuning on multiple datasets (multi) concurrently vs. on a single dataset for COCOM. Compression rate $\xi = 128$](assets/page_0008_img_1.png)
```

### --- Page 0009 ---

```markdown
| Model         | $\xi$ | Rouge-L | LMCE   |
|---------------|-------|---------|--------|
| COCOM-light   | 4     | 0.9734  | 0.1882 |
|               | 16    | 0.9643  | 0.1800 |
|               | 128   | 0.7938  | 0.1618 |
| COCOM         | 4     | 0.9979  | 0.2045 |
|               | 16    | 0.9912  | 0.1991 |
|               | 128   | 0.5545  | 0.1771 |

### 6 ANALYSIS
In this section, we conduct further analysis on how compression affects the model.

#### 6.1 Context compression
In our earlier results in Section 5.1, we observe a decline in performance with higher compression rates, particularly for the lightweight compressor in COCOM-light. To investigate potential reasons for this drop, we assess the model’s ability to perform the two pre-training tasks: (i) compressing and compressing input representations after pre-training.

Table 7 shows the results of these evaluations. Both the full and lightweight models effectively master the auto-encoding task at lower compression rates ($\xi = 4.16$). However, they exhibit significant difficulties in reconstructing the input when the compression rate increases ($\xi = 128$). This problem is notably more pronounced in our decoder-based compression model (COCOM).

We identify two possible explanations: First, compressing longer contexts tends to render embeddings inherently less presentable due to the inevitable information loss at higher compression rates. Second, the dimension of linear projection layers in the COCOM-light model is dependent on the compression rate; thus, a higher compression rate results in an increased parameter count, leading to its linear layer to manage context compression. In contrast, the COCOM model employs large tuning, where the size of the components is not dependent on the compression rate. This fundamental difference in handling compression may explain why the COCOM-light model could potentially achieve higher effectiveness under conditions of high compression, due to its higher parameter count. In terms of the second pre-training task, our results indicate that COCOM consistently outperforms COCOM-light, this finding correlates to the final effectiveness of the answer-answering tasks, as indicated in table 2.

#### 6.2 Case Study Answer Quality
We investigate the answers generated with different models. For this, we randomly select a query from the NQ dataset and compare the responses generated by each method. Table 8 presents the responses to the selected question.

### 7 CONCLUSION
In this paper, we presented our novel COCOM approach for context compression. Our main finding is that COCOM accelerates answer generation by reducing the model's input, by

![Generated responses using different methods. Dataset: NQ](assets/page_0009_img_1.png)
```

### --- Page 0010 ---

```markdown
# Conference'17, July 2017, Washington, DC, USA

compressing multiple contexts into context embeddings that, once pre-computed, serve to augment the answer generation.

Our approach maximizes the potential of the LLM by tuning all components outperforming existing methods for context compression in RAG. By offering a trade-off between efficiency and effectiveness, our method allows for the selection of varying numbers of context compression tokens. This flexibility enables us to balance higher answer quality against faster generation times as needed. Unlike previous methods, our approach allows for the input of multiple contexts, which enhances generation quality and optimally makes use of the reduced decoding time. This is because only for very long inputs, the distinction between the context in token form and a reduced set of embeddings becomes most apparent.

We hope that our work will inspire further research in context compression and pave the way for efficient and effective deployment of Retrieval-Augmented Generation (RAG) models in real-world applications.

## 8 LIMITATIONS

We end this paper by discussing the limitations of our method and of our experiments.

Our approach offers great potential to reduce the computational footprint of RAG. However, in our experiments we were constrained by available computational resources, which limits us to utilizing a relatively small model of only 7 billion parameters. This constraint prevents us from exploring the capabilities of larger models such as LLaMA3 or GPT-3.5, which might yield significantly better performance.

Additionally, the evaluation of our method has been restricted to a single task, namely Question Answering (QA) tasks and using English corpora. A more comprehensive assessment, encompassing diverse tasks and multilingual datasets, would be necessary to thoroughly understand the model’s capabilities in more varied scenarios.

## REFERENCES

[1] Akari Asai, Zezhou Zhang, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer, Hanan Hibberd, and Wen-tau Yih. 2024. Reliable, Adaptable, and Attributable Language Models with Retrieval. arXiv preprint arXiv:2403.01837 (2024).

[2] Max Bartolo, Alistair Roberts, Johannes Welbl, Sebastian Hein, and Poonam S. B. S. Tertön. 2020. Beat the Bot: Investigating Adversarial Reinforcement Learning for Reading Comprehension. Transactions of the Association for Computational Linguistics 8 (2020), 662–678. https://doi.org/10.1162/tacl_a_00338 arXiv:1805.10162.

[3] Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Wei Hu, Hsiu-Hsuan Yang, and Dongyan Zhao. 2024. xRAG: Extreme Context Compression for Retrieval-Augmented Generation with One Token. arXiv preprint arXiv:2403.01976 (2024).

[4] Alexs Chervonenkis, Alexander Metin, Amrit Ajith, and Danqi Chen. 2023. Adapting Language Models to Compress Contexts. arXiv:2305.14785 [cs.LG].

[5] Florent Couprie, Giovanni Trippas, Federico Sibilia, Simone Fricci, Ceasar Campagnon, Voile Maarek, Nicola Tonellotto, and Fabrizio Silvestri. 2024. The Power of Noise: Redefining Retrieval for RAG Systems. arXiv:2401.14857 [cs.LG].

[6] Mostafa Dehghani, Hossein Arshamzadeh, Juan Kamps, and Maarten de Rijke. 2019. Learning to Transform, Combine, and Answer in Open-Domain Question Answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL 2019, Florence, Italy, 3556–3567. https://doi.org/10.18653/v1/P19-136.

[7] Tao Ge, Jing Luo, Yifan Yang, Si-Qing Chen, and Furu Wei. 2024. In-context Autoregressive for Content Compression in a Large Language Model. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=H4gEJcE6.

[8] Pengcheng Li, Jianfeng Gao, and Weizhi Chen. 2021. DebertaV3: Improving deberta using extra-style pre-training with gradient-disentangled embedding. arXiv preprint arXiv:2104.02353 (2021).

[9] Jennifer Hwa, Arwen Shai, Zhihong Wei, and Graham Neubig. 2022. RAG-EX: Robust Design of Retrieval-Augmented Generation. arXiv:2201.03904 (2022).

[10] Gautier Cazalens, David Ritchie, Lendlin Lussier, Riosendi, Fabio Devries, Shick, Jane Dwyer, Y-S. Armand, and Sebastian Redel. 2022. Atlas: Few-shot Learning with Retrieval-Augmented Language Models. https://arxiv.org/abs/2203.02939.

[11] Huijuan Jiang, Qianhui Wu, Chih-Yen Lee, Yongyi Yang, and Lili Liu. 2023. Multi-Language Compressing for Pre-trained Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 1336–1376. https://doi.org/10.18653/v1/2023.emnlp-1.103.

[12] Kelvin Jiang, Dekou Wu, and Ming Yan. 2019. PreFace: A New Heuristic for Data Set Matching Using Style Transfer. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Minneapolis, MN, USA, 1–10. https://doi.org/10.18653/v1/P19-1028.

[13] Matt Gardner, Chandra Bhagavatula, and Luke Zettlemoyer. 2019. Multiple Choice Science Questions. arXiv preprint arXiv:1902.26049.

[14] M. A. H. S. Choi, Chih-Chung Chiu, and Wei-Lun Zettlemoyer. 2019. A Large Scale Duality-Supervised Dataset for Reading Comprehension. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Florence, Italy, 1–10. https://doi.org/10.18653/v1/P19-117.

[15] Tom Kwiatkowski, Tushar Khot, Ashish Sabharwal, and Manohar B. K. 2019. The Natural Questions Dataset. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 1–10. https://doi.org/10.18653/v1/D19-1001.

[16] Yujing Liu, Li Li, and Peter Liang. 2021. Pre-Training, Optimizing Continuous Prompts for Generation. In Proceedings of the 9th Annual International Conference on Natural Language Processing (Volume 1: Long Papers), Changqing Zong, Fei Xu, Weijie Li, and Robert Navigli (Eds.). Association for Computational Linguistics, Online, 452–487. https://doi.org/10.18653/v1/2021.acl-long.35.

[17] Youngeet Liu. 2023. Unlocking Context Constraints of LLMs: Enhancing Context Efficiency of LLMs with Self-Normalized Content Filtering. arXiv:2401.12102 [cs.LG].

[18] Nelson E. Tiu, Kevin Lin, John Hewitt, Ashwin Paranjpe, Nadeem Beliwalla, Pablo F. Cortes, and Percy Liang. 2021. Lost in the Middle: Language Models and Context Length. arXiv:2403.03177 (2021).

[19] Alex Malin, Akari Asai, Victor Zhong, Rajrushi Das, Daniel Khabshid, and Hanan Hibberd. 2023. When to Retrieve? Investigating Effectiveness of Parameters and Non-Parameter Methods. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1–10. https://doi.org/10.18653/v1/2023.acl-long.54.

[20] Text Embedding Review (Mason) and A New Text. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Houda Boumara, Juan Pino, and Kalika Bati (Eds.). Association for Computational Linguistics.
```

### --- Page 0011 ---

```markdown
Content Embeddings for Efficient Answer Generation in RAG

Singapore, 1248–1260. https://doi.org/10.18653/v1/2023.emnlp-main.765  
Nikita Shcherbakov, Hongfu Liu, Liang Wang, Nan Yang, Furu Wei, Tao Yu. Amane-  
preet Singh and Bowe Liu. 2022. Generative Representational Instruction  
Tuning. arXiv preprint arXiv:2205.09596 [cs.CL].  

Tsyganov, M., Mirzayan, K., Sko, J., Jianfeng Gao, Saurabh Tiwary, Rangan  
Mujumdar, and L. Ben. 2016. MARS: A human-generated machine reading  
comprehension dataset. 2016.  

Guilherme De Almeida, Fynd Klyuchnikov, Ben Allan, Anton Likhotvor, Mar-  
garet Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. 2024.  
The FiveW Dataset: Enriching the Web for the First Text Data at Scale.  
arXiv preprint arXiv:2406.17557 [cs.CL].  

Fabio Petroni, Aleksandra Piktus, Anjina F. Patrick Lewis, Majid Yazdani,  
Nicola De Cao, James Thornton, Yacine Jernite, Vladimir Karpukhin, Jean Maillard,  
et al. 2020. KILT: A benchmark for knowledge intensive language tasks. arXiv  
preprint arXiv:2009.02283 [cs.CL].  

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.  
SQuAD: 100,000+ Questions for Machine Comprehension of Text. In Proceed-  
ings of the 2016 Conference on Empirical Methods in Natural Language Process-  
ing, Austin, Texas, 2383–2392. https://doi.org/10.18653/v1/D16-1261  

David V. Hader, Neelesh, Nadezhda Chervonenkis, Thibaut Formal, Shuai Wang,  
Vasilisa Nikulshina, and Stephanie Chiarito. 2022. BEKENE: A Benchmarking  
Dataset for Knowledge Intensive Generation. arXiv preprint arXiv:2407.01012 [cs.CL]. https://  
arxiv.org/abs/2407.01012.  

Yuan Shen, Kylian Buvens, Dhruvan Dhingra, and Ming-Wei Chang. 2022. ASQA:  
Fact-Conditioned Long-Form Answers. In Proceedings of the 2022 Conference  
on Empirical Methods in Natural Language Processing, Yejin Choi, Kevin  
Gimpel, and Yue Zhang (eds.). Association for Computational Linguistics.  
https://doi.org/10.18653/v1/2022.emnlp-main.765.  

Yuan Shen, Liu Yi, Shikhar Patil, Ziyang Wu, Tianjun Zhang, Kurt Keutzer,  
Yuan Zhang, and Eric Xing. 2022. LCoCo: Learning Long Contexts  
for Open-Domain Question Answering. arXiv preprint arXiv:2204.10020 [cs.CL].  

Yasuda, A., Kato, K., Kawai, K., Kevin Soper, Albert, Aymad Almahdi Yasir-  
ullin, Hoon Lee, Jaehoon Jeong, Priyal Bhargava, Shahir Hossaini,  
Shuai Zhang, Yujin Kim, Jaehoon Lee, Brian Fuller, Ciyun Liu,  
Vandav Gouswami, Naman Goyal, Anthony Hartshorn, Sahar Yosif,  
Rui Han, Hoda Rahman, Marek Kacprzak, Viktor Kejzlar, Madih Khabas, Isabel  
Kloumann, Anton Pomeroy, Saurabh Singh Kumar, Marie Anne Zachary, Thibaut  
Laurin, Jeeyoung Lee, Daria Lisikovich, Yinghui Lu, Yuning Mao, Xavier Parthier,  
Jeremy Schubert, Rishi Thakur, Jae Mohyung, Yisin Nie, Andrew Poulton,  
Jeyhun Mammadov, Rishi Raghavan, Kalyan Sabadi, Allen Scheben, Ruan Silva,  
Eric Michael Smith, Ranjan Subramanian, Kileen Tan, Bin Tan, Ross  
Zhang, Yuxin Zhang, Angela Fan, Xianqian Xu, Zhen Yang, Alana Zarov,  
Robert Stojan, Sergey Edunov, and Thomas Scialom. 2023. Llama 2:  
Open Foundation and Fine-Tuned Models. arXiv preprint arXiv:2307.09295 [cs.CL].  

Yi Wang, Wen-Tsan Li, and Christopher Meek. 2015. WQG: A Challenge Dataset  
for Open-Domain Question Generation. In Proceedings of the 2015 Conference on  
Empirical Methods in Natural Language Processing, Luis Marquez, Chris Callison-  
Burch, and Jian Su (eds.). Association for Computational Linguistics, Lisbon,  
Portugal, 201–205. https://doi.org/10.18653/v1/D15-1247  

Shilan Yang, Peng Li, Shizheng Zhang, Yushou Bending, William Cohen, Ruslan  
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for  
Diverse, Explainable Multi-Hop Question Answering. In Proceedings of the 2018  
Conference on Empirical Methods in Natural Language Processing, Ellen Riloff,  
Tao Yu, Angela Fan, Joakim Nivre, and Jun’ichi Tsujii (eds.). Association for  
Computational Linguistics, Brussels, Belgium, 2369–2380. https://doi.org/10.  
18653/v1/D18-1259  

Yin Wang, Jia-Chen Lo, Caitlin Stork, Ho Ko, Yinxiao Liu, Chi-Cheng Lin, Lei  
Shu, Jianfeng Gao, Lei Meng, Bang Liu, et al. 2021. Accelerating Inference of  
Retrieval-Augmented Generation via Sparse Context Selection. arXiv preprint  
arXiv:2205.16783 [cs.CL].  
```

### --- Page 0012 ---

```markdown
# A APPENDIX

Table 9: Results in Match (M) comparing COCOM (-light) to other context compression works. All methods use 5 context passages unless indicated otherwise. * Method limited to single context. ^ upper baseline. v lower baseline. * indicates statistical non-significance (p<0.05) with respect to COCOM $\xi=4$.

| Decoder                | Method                     | Compression rate ($\xi$) | Dataset         | NQ    | TriviaQA | HotpotQA | ASQA  | PopQA | Average |
|-----------------------|---------------------------|--------------------------|------------------|-------|----------|----------|-------|-------|---------|
| **Zero-shot**         |                           |                          |                  |       |          |          |       |       |         |
|                       | AutoCompressor [1] ★     | $\times 4$               |                  | 0.351 | 0.703    | 0.314    | 0.574 | 0.237 | 0.435   |
|                       | ICAE [8]                  | $\times 4$               |                  | 0.421 | 0.784    | 0.293    | 0.649 | 0.426 | 0.479   |
|                       | xRAG [3]                  | $\times 128$             |                  | 0.316 | 0.766    | 0.267    | 0.339 | 0.326 | 0.403   |
|                       | Mistral-7B-v0.2           | $\times 128$             |                  | 0.405 | 0.852    | 0.326    | 0.457 | 0.412 | 0.490   |
|                       | RAG^a (no compression)    | -                        |                  | 0.637 | 0.917    | 0.544    | 0.665 | 0.543 | 0.661   |
|                       | LLM^v (no context)        | -                        |                  | 0.403 | 0.753    | 0.283    | 0.573 | 0.208 | 0.444   |
| **Fine-tuned**        |                           |                          |                  |       |          |          |       |       |         |
|                       | COCOM-light (ours)        | $\times 4$               |                  | 0.579 | 0.882    | 0.439    | 0.633 | 0.478 | 0.601   |
|                       |                           | $\times 16$              |                  | 0.529 | 0.857    | 0.395    | 0.604 | 0.395 | 0.556   |
|                       |                           | $\times 128$             |                  | 0.479 | 0.828    | 0.347    | 0.586 | 0.326 | 0.513   |

Table 10: Hyperparameters for Pretraining

| Hyperparameter         | Assignment               |
|-----------------------|-------------------------|
| learning Rate         | $1 \times 10^{-4}$      |
| lr scheduler type     | linear                  |
| warmup ratio          | 0.05                    |
| weight decay          | 0.1                     |
| overall batch size    | 256                     |
| optimizer             | AdamW                   |
| epochs                | 1                       |
| LoRa layers           | all linear layers       |
| LoRa alpha            | 32                      |
| LoRa dropout          | 0.1                     |
| LoRa r               | 16                      |
| LoRa bias             | None                    |
| GPU                   | 8 x A100 80GB           |
| context max length    | 128                     |
```

### --- Page 0013 ---

```markdown
# Table 11: Hyperparameters for Fine-tuning

| Hyperparameter        | Assignment          |
|----------------------|---------------------|
| learning Rate        | $1 \times 10^{-4}$  |
| lr scheduler type    | linear              |
| warmup ratio         | 0.05                |
| weight decay         | 0.1                 |
| overall batch size    | 64                  |
| optimizer            | AdamW               |
| epochs               | 2                   |
| LoRa layers          | all linear layers    |
| LoRa dropout         | 0.1                 |
| LoRa alpha          | 32                  |
| LoRa r              | 16                  |
| LoRa bias           | None                |
| GPU                  | 8 x A100 80GB       |
| retriever(s)        | SPLADE-v3 (+ DeBERTa-v3) |
| num passages         | 5                   |

# Table 12: Datasets contained in the multi-dataset collection used for fine-tuning our COCOM (light). We filtered out queries with more than 128 tokens and labels of more than 64 tokens.

| Dataset      | Number examples |
|--------------|-----------------|
| NQ           | 87,925          |
| MSMARCO      | 100,000         |
| Adversarial QA | 30,000        |
| HotpotQA     | 88,869          |
| WikiQA      | 873             |
| SciQ         | 11,679          |
| ASQA         | 4,353           |
| Wiki QA      | 61,517          |
| Freebase     | 20,358         |
| SQuAD        | 87,599          |
| **Total**    | **493,473**     |
```



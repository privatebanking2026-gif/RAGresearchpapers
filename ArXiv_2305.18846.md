# ArXiv 2305.18846

### --- Page 0001 ---

```markdown
# Knowledge Graph-Augmented Language Models for Knowledge-Grounded Dialogue Generation

Minki Kang\(^1,2\); Jin Myung Kwak\(^1,*\); Jinheon Baek\(^1,*\); Sung Ju Hwang\(^1\)  
KAIST\(^1\), AIITRICS\(^2\)  
\{zzxc1133, kwak.jinmyung, jinheon.baek, sjhwang82\}@kaist.ac.kr

## Abstract

Language models have achieved impressive performances on dialogue generation tasks. However, when generating responses for a conversation that requires factual knowledge, they are far from perfect, due to an absence of mechanisms to retrieve, encode, and reflect the knowledge in the generated responses. Some knowledge-grounded dialogue generation methods tackle this problem by leveraging facts from Knowledge Graphs (KGs); however, they do not guarantee that the model utilizes a relevant piece of knowledge from the KG. To overcome this limitation, we propose SURgeon Retrieval-augmented GEnration (SURGE), a framework for generating context-relevant and knowledge-grounded dialogues with the KG. Specifically, our SURGE framework first retrieves the relevant subgraph from the KG, and then enforces consistency across facts by perturbing their word embeddings conditioned on the retrieved subgraph. Then, we utilize contrastive learning to ensure that the generated texts have high similarity to the retrieved subgraphs. We validate our SURGE framework on OpenDialKG and KOMODIS datasets, showing that it generates high-quality dialogues that faithfully reflect the knowledge from KG.

## 1 Introduction

Dialogue systems aim to engage in ongoing conversations with humans by generating human-like responses. While pre-trained language models (PLMs) [20, 30, 37] are capable of generating fluent responses, they often generate factually incorrect responses due to a lack of explicit knowledge [38]. To overcome such limitations, recent works access external knowledge sources such as Wikipedia [6], Web [15], Freebase [31] and Wikidata [45] to retrieve the relevant knowledge for the dialogue context. In this work, we focus on the Knowledge Graphs (KGs)-based dialogue generation as existing works [39, 47, 49, 4, 5, 1, 8, 19]. KGs represent facts in the most compact and effective symbolic structured form (See the leftmost of Figure 1), consisting of entities and nodes and their relation as an edge. Each of them is formed with a triplet, which can help generate knowledge-grounded responses.

Most of the dialogue generation models with KGs [8, 19] utilize all the triplets associated with entities in the dialogue context. However, through observation, we found not all of the facts are actually relevant to the conversation (e.g., Jane Austen was born in Steventon in Figure 1), which could mislead the models to generate factually incorrect responses. 87% of facts from 1-hop KG are irrelevant to the context in the OpenDialKG dataset [24]. Moreover, encoding all the facts including the unnecessary ones is computationally inefficient [8, 36]. On the other hand, even after correctly retrieving the relevant facts, it is not straightforward to combine two heterogeneous modalities: the dialogue context is represented as a text; the knowledge is represented as a graph. In other words, since PLMs already have tons of pre-trained parameters trained on the unstructured texts, properly conditioning the structured graph to PLMs is highly challenging. Otherwise, PLMs may
```
![Detailed description of the chart](assets/page_0001_img_1.png)

### --- Page 0002 ---

```markdown
![Motivation: Existing knowledge-grounded dialogue generation models with a KG often utilize the multi-hop subgraph associated with entities in the dialogue context (e.g., Jane Austen). However, they suffer from a couple of the following problems: (1) irrelevant knowledge when only 12.6% of facts from the 1-hop KG are useful to generate the target responses given a dialogue context, and (2) inconsistent generation where generated texts include the factually incorrect statements.](assets/page_0002_img_1.png)

generate inconsistent responses disregarding the knowledge from the retrieved subgraph, which is a phenomenon known as hallucination [34], where the generated responses exhibit their own memorized yet incorrect knowledge.

In this work, we tackle such challenging and fundamental issues of knowledge-grounded dialogue generation with the KGs. We propose an end-to-end dialogue generation framework that considers all aspects from knowledge retrieval, encoding, and reflection along the generation process. As a first step, we propose a context-relevant subgraph retriever that retrieves only the relevant triplets from KGs to prevent the model from generating context-irrelevant responses. Notably, our subgraph retrieval method embeds the KG considering its relational structure with the Graph Neural Network (GNN) [14] instead of using PLMs as in previous work [19]. Furthermore, without any labels for pairs of dialogue contexts and their relevant subgraphs, our method is end-to-end trainable jointly with the generation objective by marginalizing the likelihood of the generated sentences over the latent retrieved subgraph [19]. Then, to encode the retrieved subgraph into the input text sequence, we propose a graph encoding based on its permutation and relation invariant yet efficient. Specifically, we design the graph encoding method that retains the order of the representation space of PLMs, instead of prepending them in front of the text sequence to avoid the computational burden. Furthermore, to ensure that the model does make use of the encoded knowledge when generating responses, we propose a multi-modal contrastive learning objective between two different graph-text modalities to enforce consistency across the retrieved facts and the generated texts. We call our framework SURGE (Subgraph Retrieval-augmented Generation).

We validate our SURGE framework on the OpenDialKG [24] and KOMODIS [7] datasets. Note that, when evaluating the generated responses from dialogue models, conventional metrics (e.g., BLEU [25]; Rouge [20]) cannot measure how faithfully the generated responses reflect the related knowledge in KGs. Thus, in evaluation, we further introduce an additional performance metric, referred to as Knowledge-verifying Question Answering (KQA), which evaluates whether the generated responses contain the correct knowledge with an additional extrinsic question answering scheme. The experimental results show that SURGE generates responses that not only agree with the gold knowledge but are also consistent with the retrieved knowledge from KGs. Our contributions are summarized as follows:

- We propose a GNN-based context-relevant subgraph retriever that extracts the context-relevant piece of knowledge from KGs, for generating appropriate responses to the ongoing conversation.
- We propose an invariant yet efficient graph encoder and a graph-text contrastive learning objective to ensure that the generated responses faithfully reflect the retrieved knowledge.
- We validate our SURGE framework against relevant baselines, demonstrating its efficacy in generating responses that are more informative by retrieving and reflecting the relevant knowledge.

## 2 Related Work

### Language Models
Pre-trained Language Models (PLMs) [29, 16, 10] that use a Transformer-based [43] encoder-decoder architecture have achieved great successes on language generation tasks. As they can accurately contextualize the given context and then generate human-like sentences, they are recently used as the base architecture for neural dialogue systems [50, 11]. Moreover, when PLMs become larger, dialogue models are capable of generating high-quality responses [1], suggesting that pre-trained parameters do contain certain knowledge [28]. However, despite the fluency of
```

### --- Page 0003 ---

```markdown
# Dialogue Generation with KGs

Regarding dialogue generation tasks with KGs that we target, Moon et al. [24] introduce a knowledge-grounded dialogue dataset, where each dialogue turn comes with facts from the large-scale KG. Several following works [39, 47, 49, 4, 51] suggest sequence-to-sequence models trained from scratch, which focus on generating dialogue by conditioning the output word distribution with entities from the KG. Further, Galetzka et al. [8] propose an efficient method that encodes all facts in the k-hop neighbors of entities that appear in the dialogue history, in order to reduce the number of input tokens forwarded in PLMs. On the other hand, Rony et al. [36] propose to mask out model weights for irrelevant facts in PLMs. However, all of these methods simply match and retrieve all facts for entities that appear in the dialogue context, which either may mislead models to generate out-of-context responses from irrelevant facts, or can increase the computational overheads from prepending all tokens for all facts in PLMs. Our work differs from them since we aim at retrieving only a context-relevant subgraph among all associated facts with its retriever, which is end-to-end trainable along with a generative model.

## 3 Method

In this section, we first discuss the basic ingredients: Transformer and Graph Neural Network. We then formalize the dialogue generation problem and describe key components for our Subgraph Retrieval-augmented Generation (SURGE) framework: context-relevant subgraph retrieval, invariant graph encoding, and graph-text contrastive learning. Figure 2 illustrates the overview of our framework.

### 3.1 Preliminaries

As we use two different modalities, namely text and graph, we first define them, and then describe the neural networks to encode them. In particular, a text is defined as a sequence of tokens $x = [x_1, \ldots, x_l]$, $\forall i, \; x_i \in V$, where $V$ is a predefined vocabulary formed with specific tokenization algorithms [37]. On the other hand, a knowledge graph (KG) is a type of multi-relational graphs $G = (E, R, E') \in E \times R \times E'$, where $e, e' \in E$ are head and tail entities (nodes) along their relation (edge) $r$ and $E$ and $R$ are sets of entities and relations, respectively.

To easily access different modalities in the same framework, we define the tokenization (mapping) function that maps entities and relations to the tokens used in Pre-trained Language Models (PLMs), represented as follows: $f: \mathcal{U} \rightarrow V^l$ where $l$ is an arbitrary length varying across different entities and relations. In other words, any entity $e \in E$ and relation $r \in R$ is tokenized to a sequence of tokens $x \in V^l$: $e(\cdot) = xe$ and $r(\cdot) = xr$. For instance, an entity New York (i.e., e) is tokenized into two tokens "New" and "York", i.e., $xe = ['New', 'York']$.

#### Transformer

A Transformer [43] is the most basic building block of recent PLMs [5, 29]. Given a sequence $x = [x_1, \ldots, x_n]$, $\forall i \in V$, generative transformers generate a sequence $y_{1:t-1} = [y_1, \ldots, y_{t-1}]$, $\forall y_i \in V$ with encoder Enc, decoder Dec, and token embedding $f$. A hidden state at time t for generating $y_t$ is $h_t = \text{Dec}(Enc(X), Y_{1:t-1})$, where $X = f(x_1), \ldots, f(x_n)$ and $y_{1:t-1} = [y_{1}, \ldots, y_{t-1}]$. Both Enc and Dec functions are permutation sensitive with positional embedding.

#### Graph Neural Network

A Graph Neural Network (GNN) represents a node with its neighboring nodes over graphs [10], as follows:

$$
GNI(e_i; G) = \text{UPD}(e_i; \text{AGG}(e_j \; | \; \forall e_j \in N(e_i; G))),
$$

where $N(e_i; G) = \{e_h | (e_h, e_i, r_e) \in G\}$ is a set of neighboring entities of $e_i$; $e_i$ and $e_h$ are embeddings of entities (nodes) and edge; $\text{AGG}$ is a function that aggregates embeddings of neighboring entities; and $\text{UPD}$ is a function that updates $e_i$ with the aggregated messages from $\text{AGG}$.
```

### --- Page 0004 ---

```markdown
![Framework Overview. Our framework, SURGE, consists of three parts. First, a context-relevant subgraph retriever $R(Z|x)$ retrieves the subgraph $Z$ relevant to the given dialogue history from a knowledge graph $G$ (e.g., 1-hop KGC from entity Jane Austen); a. Specifically, we measure the similarity of a context and triplet embedding to compose the retrieval distribution $p(Z|x)$ (3.3). Then, we encode the retrieved subgraph $Z$ using the graph encoding $\Psi_G(Z)$ (3.4). Finally, we use contrastive learning to enforce the model to generate a knowledge-grounded response with the retrieved subgraph $Z$ (3.5).](assets/page_0004_img_1.png)

## 3.2 Problem Statement

Given a dialogue history $x = [x_1, \ldots, x_N]$, a model with generative PLMs models a conditional distribution $p(y|x)$ to generate an output response $y = [y_1, \ldots, y_T]$. To generate knowledge-grounded dialogue, this problem requires a piece of specific knowledge for an ongoing conversation.

To that end, given a dialogue history, we aim at retrieving a subgraph $Z \subset G$ consisting of a set of triplets $z \in Z$ where $z = (e_h, r, e_t)$, which encodes relevant knowledge for ongoing conversation. The distribution of the context-relevant subgraph is $p(Z|x)$ and our final likelihood for response generation becomes $p(y|z, Z)$. To jointly optimize the objective of graph retrieval with response generation, we treat $Z$ as a latent variable and then marginalize the likelihood of the generative model over all possible latent variables for retrieved subgraphs $Z$, formalized as follows:

$$
p(y|x) = \sum_{Z \subset G} p_\theta(Z|x) p(y|z, Z) = \sum_{Z \subset G} p_\theta(Z|x) \prod_{t=1}^T p(y_t|z, y_{0:t-1}), \tag{1}
$$

where $y_0$ is the start token for the generation, $p(Z|x)$ is an output distribution of the context-relevant triplet retrieval, and $p(y|z, Z)$ is the target distribution of the knowledge-augmented generator, parameterized as $\phi$ and $\theta$, respectively, which we specify in next few subsections.

## 3.3 Context-Relevant Subgraph Retriever

We now provide a concrete description of our context-relevant subgraph retriever, i.e., $R(Z|x)$, formalized in Equation 1. Given the dialogue history $x$, we assume that retrieval of each triplet in $Z = \{z_1, \ldots, z_h\}$ is independent. Then, for simplicity, we decompose the retrieval of a set of triplets $p(Z|x)$ into the product of individual triplet retrieval, represented as follows: $p(Z|x) = p(z_1|x)p(z_2|x) \cdots p(z_h|x)$. From this decomposition, it is now sufficient to focus on a single triplet retrieval. We define the score for the single triplet with the inner product between embeddings of the dialogue history $x$ and the candidate triplet $z \in Z$, as follows:

$$
p_\theta(z|x) \propto \exp(d(e_h, e_t)), \tag{2}
$$

where $d$ is a triplet embedding function and $s$ is a dialogue context embedding function. For implementing $s$, we can use any off-the-shelf PLMs, but for $d$, we need another effective approach that captures the property of the graph. Therefore, we utilize the Graph Neural Networks (GNNs) for the triplet embedding function $d$ to consider the relational structure between entities in the KG.

More specifically, we consider a set of triplets associated to the entities that appear in the given dialogue context: $\{(e_h, r, e_t) \,|\, (e_h, r, e_t) \in \mathcal{E} \, \text{and} \, e_h, e_t \in x\}$ as the retrieval candidates. Then, to effectively represent triplets consisting of entities and their relations as items, we use GNNs described in Section 3.1. In our triplet retriever, utilizing both nodes and edges, which are equally essential.
```

### --- Page 0005 ---

```markdown
### 3.4 Invariant Graph Encoding

We now specify graph encoding, which aims to condition the structural graph $Z$ along with the text sequence $x$ over PLMs to generate $y$. Let $\pi(x, Z)$ be a graph encoding function. Then, the simplest way to encode graphs into PLMs is to prepend the tokens of entities and relations to the input text $z$ \cite{19, 22}. Formally, given a text $x = [x_1, \ldots, x_n]$ and a graph $Z = (a_1, r_1, b_1), (b_1, r_2, a_2), (a_1, r_3, c_1), \ldots, (a_n, r_n, c_n)$, a naive graph encoding is defined as follows: 

$$
\psi(x, Z) = f([x_1, b_1, b_2, r_1, b_1, a_2, r_2, c_1, \ldots, r_n, c_n])
$$ 

where $a = (a_1, r_1)$, $b = (b_1, r_1)$, and so on. Here $f$ is a token embedding and $g$ is a mapping function defined in Section 3.1. However, it violates important properties for consistent encoding of a multi-relational graph into PLMs: permutation invariance \cite{48} and relation-inversion invariance, formalized in Definition 3.1 and 3.2 as follows:

**Definition 3.1. (Permutation Invariance)** For any permutation $\pi \in S_n$, $\psi(x, Z) = \psi(x, \pi \cdot Z)$.

**Definition 3.2. (Relation Inversion Invariance)** Let $\pi^{-1}$ be an inverse relation to $r$ if $(a, r, b) = (b, r^{-1}, a)$ for any subgraph $Z$. Formally, given $x$ and $Z$, 

$$
\psi(x, Z \cup \{(a, r, b)\}) = \psi(x, Z \cup \{(b, r^{-1}, a)\}) 
$$ 

for any triplet $u$ to the naïve encoding. We first define a SORT operator that returns the same graph regardless of the order of input elements, as follows:

$$
SORT(\pi \cdot Z) = SORT(\pi' \cdot Z'), \forall \pi, \pi' \in S_n, 
$$ 

where $S_n$ is a set of all possible permutations for elements. We then define an INV operator that adds the inverse triplet of each triplet in $Z$, as follows:

$$
INV(Z) = Z \cup \{(e^{-1}, e_h) | (e_h, e) \in Z\}. 
$$

Based on them, our graph encoding function, $\psi(x, SORT(INV(Z)))$, satisfies both invariances.

**Invariant and Efficient Graph Encoding** 

However, the above encoding is not efficient since it requires $O(n)$ space complexity with triplets. Thus, we newly define $\psi$ that encodes the sorted sequence of only the unique entities, as follows:

$$
\hat{\psi}(x, SORT(ENT(Z))) = f([a_1, b_1, c_1, \ldots, x_n]), 
$$ 

where $ENT(Z)$ returns the set of unique entities in $Z$. This encoding meets both invariance properties but also efficient since it only costs $O(k)$, for the $k$ entity where $k < n$. However, as it does not retain the relational information in $Z$, we further perturb the token embeddings of each entity $f(\cdot)$ in PLMs with respect to their graph representations in $Z$. Specifically, for each entity $a \in ENT(Z)$, we apply a learnable affine transformation \cite{27} on the token embedding of $a$ as follows:

$$
\beta(f(a), Z) = (1 + \gamma) f(a) + \delta, 
$$ 

where MLP is a Multi-Layer Perceptron, $\gamma \in R^{d} \to R^{d}$ perturbs the embedding according to $Z$, $R-GNN$ is the relation-aware GNN \cite{42}. In sum, we denote a relation-aware and invariant yet efficient encoding $\psi^*(x, Z) = \beta(\hat{\psi}(x, SORT(ENT(Z))), INV(Z))$.

We conclude that our graph encoding satisfies both properties. For further details on the proof and comprehensive illustration, please refer to Appendix D.
```

### --- Page 0006 ---

```markdown
# 3.5 Knowledge-Grounded Generation with Graph-Text Contrastive Learning

Our framework now can retrieve and encode the context-relevant subgraph given the user input. Then, reflecting the subgraph into the model is important when generating a knowledge-grounded response. The generative model should be able to generate different sequences when providing different subgraphs, for the same dialogue history.

However, we only access the single ground-truth response regardless of the retrieved knowledge, while the generative model is trained with a teacher forcing. Thus, this setting can raise the problem of exposure bias [32]: the model is never exposed to other generated tokens during training. To overcome such limitations, we introduce a graph-text contrastive learning method. Formally, for a single pair of a graph and text, the contrastive learning objective is defined as follows:

$$
L_{cont} = \frac{1}{2} \log \frac{\exp(\text{sim}(c(z), c(h))/\tau)}{\sum_{y} \exp(\text{sim}(c(z'), c(h))/\tau)} + \frac{1}{2} \log \frac{\exp(\text{sim}(c(z), c(h))/\tau)}{\sum_{y'} \exp(\text{sim}(c(z'), c(h))/\tau)} \tag{7}
$$

where $z = [z_1, z_2, \ldots, z_n]$, $h = \frac{1}{T_h} \sum_{t=1}^{T_h} h_t$ is the mean of decoder representations, $\text{sim}$ is the cosine similarity, and $\tau$ is a learnable linear temperature parameter. Furthermore, $\sum_{y'}$ indicates the summation over negative samples, which are other texts or graphs within a same mini-batch.

# 3.6 Training

We train the entire model by maximizing the log-likelihood log $p(y|x)$ defined in Equation 1 with respect to parameters of both the retriever $\phi$ and the generator $\theta$. Since computing the marginal probability over entire subgraphs is infeasible, we approximate it by summing over $k$ sampled subgraphs [9, 17]. Our end-to-end training objective for retrieval-augmented generation is then defined as follows:

$$
L_{ret} = \log \frac{p_{\phi}(Z|x)p_{\theta}(y|x, Z)}{Z_{GI}} \tag{8}
$$

where $\Pi = \text{sample}(p_{\rho}(\cdot|x))$ denotes sampling $k$ subgraphs over the subgraph distribution and each subgraph sampling is decomposed into sampling $n$ triplets from $p(z_i|x)\forall i \in [1, n]$ as in Section 3.3. We further assume that the gold subgraph $Z^*$ is partially available in training. In this case, we utilize the following supervised loss to train the retriever: $L_{sup} = \log p_{\phi}(Z^*|x)$. By combining all objectives in Equation 7, 8, and $L_{sup}$, our training objective is defined as $L = L_{ret} + L_{sup} + L_{cont}$.

# 4 KQA Metric: Knowledge-verifying QA

Existing automatic evaluation metrics, namely BLEU and ROUGE [25, 20], are limited in that they only consider the lexical overlaps of words without measuring the factual correctness. For instance, as shown in Figure 3 (a), there could be multiple correct responses, but existing metrics score them lower due to the lexical mismatch with the gold response. To solve this issue, we propose Knowledge-verifying Question Answering (KQA) which measures whether generated responses contain factual correct knowledge given the dialogue history. To realize this, we formulate extrinsic QA task [31] by automatically deriving QA pairs from the dialogue and the large-scale KG in each dataset (See Figure 3). Then, we fine-tune BERT [5] on synthetic KQA pairs to build a QA model.
```


### --- Page 0007 ---

```markdown
| Method                       | KQA         | BLEU         | ROUGE         | F1         | R2         | R1         |
|------------------------------|-------------|---------------|---------------|------------|------------|------------|
|                              | EM  | F1  | B-1 | B-2 | B-3 | B-4 | R-1 | R-2 | R-1 | F1  |
| Baselines                    |     |     |    |    |    |    |    |    |    |     |
| Space Efficient Encoding      | 36.60 | 42.64 | 16.15 | 10.93 | 6.63 | 4.65 | 20.45 | 15.36 | 20.24 | 24.54 |
| EARL                         | 38.44 | 43.11 | 16.03 | 12.82 | 8.62 | 6.81 | 21.68 | 16.41 | 21.68 | 25.84 |
| DiffKG                      | 32.87 | 38.95 | 15.89 | 9.53 | 3.96 | 1.95 | 19.50 | 17.87 | 18.41 | 22.86 |
| Random Retrieval             | 31.72 | 39.05 | 15.70 | 9.62 | 2.99 | 20.21 | 7.68 | 19.57 | 23.28 | 23.28 |
| Retrieval                    |             |               |               |            |            |            |
| variants                     |             |               |               |            |            |            |
| Dense Retrieval (Bi-encoder) | 46.07 | 52.52 | 16.70 | 10.74 | 4.91 | 21.41 | 8.75 | 19.57 | 23.53 | 23.53 |
| Dense Retrieval (Poly-encoder)| 46.50 | 57.67 | 11.01 | 7.45 | 5.06 | 20.64 | 19.67 | 24.24 | 24.24 | 24.24 |
| SURGE (unsupervised)         | 51.90 | 57.67 | 17.11 | 12.76 | 7.58 | 21.48 | 20.57 | 25.47 | 25.47 | 25.47 |
| SURGE (semi-supervised)     | 50.43 | 67.70 | 17.29 | 11.04 | 7.34 | 21.83 | 8.98 | 20.48 | 25.10 | 25.10 |
| Oracle                       | 63.37 | 80.12 | 12.47 | 7.93 | 6.29 | 24.90 | 19.97 | 20.43 | 19.97 | 20.43 |
| Gold Response                | 93.30 | 92.51 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

### Table 3: Knowledge-grounded generation results by using the modified model sub- 
### as instead of the retrieved ones, we evaluate the efficacy of corrective learn- 
### ing, with F1 and KFI metrics.

| Method                       | F1  | KFI  |
|------------------------------|-----|------|
| SURGE (unsupervised)         | 27.78 | 24.90 |
| SURGE (semi-supervised)     | 28.30 | 26.38 |
| SURGE (contrastive)         | 27.81 | 27.58 |

### Figure 4: Retrieval results on the OpenDialKG dataset, with the metrics of 
### MRR and Hits @3.

5 Experiment
5.1 Experimental Setup
We conduct experiments on OpenDialKG [24], which consists of 15K dialogues with 91K utterances associated with a large-scale KG, namely Freebase [3] with 100K entities and facts. We note that among them, 49% of the utterances come with gold knowledge, whereas others are not. We randomly split the dataset into training (70%), validation (15%), and test sets (15%). We also use KOMODIS [7], which contains 7.5K dialogues associated with the KG having 88K facts. As retrieval candidates, we use 1-hop KG associated with entities in the given dialogue for OpenDialKG and 2-hop KG for KOMODIS. Except Table 2, most of the experiments are conducted on OpenDialKG. We use T5-small [30] for all experiments. For more details, see Appendix E.

5.2 Baselines and Our Models
No Knowledge. The model only with the dialogue history. All Knowledge. The model with entire facts within a k-hop subgraph. Gold Knowledge. The model with the exact gold knowledge if it exists. Space Efficient Encoding. The model with all facts from the k-hop subgraph. We use two variants from [8], namely series and parallel. EARL. The latest RNN-based model, where the entities are conditioned in response generation [51]. DiffKG. PLM-based model with differentiable path traversal [40]. Random/Sparse Retrieval. The model with selected facts via random sampling or sparse retrieval [33]. Dense Retrieval. A variant of our framework where T5 encoder [30] is used for in Equation 3 instead of GNNs similar to Bi- and Poly-encoder [12]. SURGE (unsupervised). Ours with retrieved context-relevant facts from k-hop subgraph, where the retrieval is trained without any supervision. SURGE (semi-supervised). Ours but the retriever is trained with $L_{sup}$ if the gold exists. SURGE (contrastive). Ours with $L_{ret}$, $L_{sup}$, and $L_{cont}$ in Section 3.
```

### --- Page 0008 ---

```markdown
| Table 4: Examples of responses from the baseline (Space Efficient, parallel) and responses from SURGE. |
| --- |
| **Context** | **Gold response** | **Baseline response** | **SURGE response** |
| (a) I love Moby Dick. Can you recommend something similar? | It was written by Herman Melville in 1851. It’s one of the best novels. | Do you like Moby Dick? | Moby Dick was written by Herman Melville. |
| (b) Do you know anything about the movie that stars Adam Brown? | Yes, the whale was in the movie The Hobbit: An Unexpected Journey. | Adam Brown started in King Kong. | Adam Brown starred in The Hobbit: The Desolation of Smaug and The Hobbit: The Battle of the Five Armies. |

| Table 5: Experimental results on OpenDialKG with additional three metrics other than KQA for measuring whether the generated responses contain appropriate knowledge. |
| --- |
| **Method** | **KQA EM** | **KQA Knowledge F1** | **Entity F1** | **String Matching** |
| --- | --- | --- | --- | --- |
| Baseline | 43.54 | 50.60 | 18.91 | 21.10 | 44.25 |
| Space Efficient (Parallel) | 38.54 | 44.34 | 17.43 | 18.93 | 40.56 |
| Space Efficient (Pub-encoded) | 40.25 | 52.37 | 21.84 | 48.41 |  |
| SURGE (contrastive) | 51.00 | 57.63 | 21.87 | 23.40 | 59.75 |
| Gold Response | 93.90 | 25.62 | 29.86 | 20.35 | 85.75 |

| Table 6: Performance comparisons of Table 7. Human evaluation on evaluations of graph encodings, described sintency, informativeness, and fluency in Section 3.4. (Inv= Invariant) |
| --- |
| **Method** | **KQA Knowledge** | **Method Consistency** | **Fluency** |
| --- | --- | --- | --- |
| New Encoding | 52.73 | 62.31 |  |
| Invariant (entirely) |  |  |  |
| Space Efficient | 2.47 | 1.75 | 2.46 |
| SURGE (c) | 2.71 | 2.39 | 2.92 |

![Embedding visualization of graph (star) - text (circle) contrastive learning.](assets/page_0008_img_1.png)

### 5.3 Evaluation Metrics
We use BLEU [25], ROUGE [20], and F1 score as metrics. We also use our new metric, KQA (Section 4), which measures whether the generated responses contain proper knowledge. We use Knowledge F1 (KFI) [38] to measure unigram overlaps between the retrieved knowledge and generated response.

### 5.4 Experimental Results and Analysis
In Table 1, we report the knowledge-grounded response generation performances of baselines and our SURGE on OpenDialKG. As shown in Table 1, our models significantly outperform all the baseline models, excluding oracle, in all evaluation metrics. The high BLEU, ROUGE, and F1 refer that ours sufficiently learns the syntactic and semantic structure of the responses. Our models also achieve high F1 and EM scores in KQA. The high KQA scores indicate that the generated responses are formed with the correct facts, which are relevant to the dialog context. Even the baseline models such as All Knowledge, Space Efficient Encoding [8], EARL [51], and DijKG [40], which are provided with all k-hop facts, underperform than ours. The result demonstrates that selecting relevant knowledge is critical in knowledge-augmented response generation. In Table 2, we additionally report the experimental results on KOMODIS to show the applicability of our method to other datasets. Our SURGE (contrastive) also outperforms other baselines in KOMODIS. For results with all metrics, please see Appendix F.3.

Knowledge Retrieval Figure 4 shows performances of retrievers, for which we measure the performance on 45% of test dialogues containing the gold knowledge, with Mean Reciprocal Rank (MRR) and Hits@k as metrics. Our models outperform all baselines. Further, our model with contrastive learning and semi-supervised retriever training outperforms an unsupervised version. See Appendix H for examples.
```

### --- Page 0009 ---

```markdown
## Knowledge-Grounded Generation

We conduct an ablation study on our models to validate the knowledge consistency performance of the response generation by computing the Knowledge F1 (KFI) score [38]. We use the gold knowledge rather than the retrieved one to focus solely on the case where a given knowledge is consistently reflected in the generated responses. We randomly modify the tail entity of each gold knowledge to ensure that responses are generated from the given knowledge rather than the trained knowledge. Table 3 shows that our model with a contrastive learning term outperforms all others in the KFI, implying that the generated responses accurately reflect the encoded knowledge.

## Retrieval and Generation Examples

Table 4 shows the examples of generated responses along with the retrieved knowledge. We compare our SURGE against Space Efficient (parallel) baseline. In example (a), the baseline response contains an incorrect fact distracted by the contextually irrelevant entity ‘sailor’. Contrarily, SURGE successfully retrieves relevant facts from the KG and then generates the factually correct response. In (b), the baseline generates the response with a wrong fact, meanwhile, SURGE retrieves context-relevant facts and generates an informative response.

## Automatic Evaluations on Knowledge Groundedness

In Table 5, we measure Knowledge F1 (KFI in Table 3), string matching (check whether at least one of answer entities exists the generated response), and entity F1 (measuring F1 score with each entity in answer candidates) for representative baselines and our SURGE (semi-supervised) on OpenidalKG. For KFI, we measure the F1 score regarding the concatenation of the question (head entity and relation) and all answer candidates (available tail entities) in KQA as the gold response. The results show that all metrics show the same tendency with KQA and our proposed method still outperforms other baselines by generating responses with more proper knowledge. See Appendix F.6 for more details.

## Sensitive Analysis on Graph Encoding

We conduct an analysis on graph encoding variants introduced in Section 3.4. The knowledge length in Table 6 indicates the average token length used in encoding. Our encoding uses Equation 6 for embeddings the best against other variants while using the lesser space at the graph encoding phase.

## Human Evaluation

We sample 30 responses of SURGE, All Knowledge, and Space Efficient on the test set of OpenidalKG, then conduct a human study of them. We recruit 46 annotators and ask them to evaluate the quality of the generated responses with consistency, informativeness, and fluency criteria using a 3-point Likert-like scale. As shown in Table 7, ours obtains significantly higher scores than others in all criteria, which is another evidence that our framework generates consistent, informative, and fluent responses. We observe that the informativeness score and KQA F1 score have a 0.42 Pearson correlation coefficient. This allows us to confirm that our KQA metric positively correlates with the human evaluation results.

## Embedding Space Visualization

We further visualize the latent space of graph and text learned from Equation 7 in Figure 5. The visualization shows that, for the same dialogue with different subgraphs, our SURGE with graph-text contrastive learning (right) generates distinct response embeddings pertaining to different subgraphs, unlike the one without contrastive learning which shows less variety over responses for the same dialogue (left). We include zoomed Figure 5 in Appendix H.

## 6 Conclusion

In this work, we proposed a novel end-to-end framework for knowledge-grounded dialogue generation which retrieves context-relevant subgraph, encodes a subgraph with the text, and generates natural and informative responses based on the retrieved subgraph, called Subgraph Retrieval-augmented GEnration (SURGE). Our results demonstrate the effectiveness of our framework in both quantitative and qualitative experiments in knowledge retrieval and response generation tasks. The analysis shows the contribution of each proposed component: retrieval, encoding, and graph-text representation learning. Our work suggests a new direction to generate informative responses for knowledge graph-based dialogue task by empirically showing the importance of retrieving the more relevant subgraph knowledge rather than using all the relevant knowledge graphs when generating knowledge-grounded responses.
```

### --- Page 0010 ---

```markdown
# References

1. Daniel Adiwardana, Minh-Thang Luong, David R. So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang, Apoorv Kulshreshtha, Garrett Newade, Yifeng Lu, and Quoc V. Le. Towards a human-like open-domain chatbot. CoRR, abs/2001.09977, 2020.

2. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.

3. Kurt D. Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings of the ACM SIGMOD International Conference on Management of Data, SIGMOD 2008, Vancouver, BC, Canada, June 10-12, 2008, pages 1247–1250, 2008.

4. Fuwei Cui, Hui Di, Hongjie Ren, Kazushige Ouchi, Ze Liu, and Jinan Xu. Syntactically diverse adversarial network for knowledge-grounded conversation generation. In Marie-Francine Moens, Xuanjing Huang, Lucila Specia, and Scott Wen-tau Yih, editors, Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November, 2021, pages 4602–4630. Association for Computational Linguistics, 2021. URL https://doi.org/10.18653/v1/2021.findings-emnlp.394.

5. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186, 2019. URL http://doi.org/10.18653/v1/n19-1423.

6. Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael A. Maddox, and Jason Weston. Wizard of Wikipedia: Knowledge-powered conversational agents. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

7. Fabian Galetzka, Chukwumeeka Uchenna Emeh, and David Schlangen. A corpus of controlled opinionated and knowledgeable movie discussions for training neural conversation models. In Nicolette Calzolari, Frédéric Béchet, Philippe Blache, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asunción Moreno, Jan Odijk, and Stelios Piperidis, editors, Proceedings of The 12th Language Resources and Evaluation Conference, LREC 2020, Marseille, France, May 11-16, 2020, pages 565–573. European Language Resources Association, 2020. URL https://aclanthology.org/2020.lrec-1.71/.

8. Fabian Galetzka, Jewgienij Rose, David Schlangen, and Jens Lehmann. Space efficient context encoding for non-task-oriented dialogue generation with graph attention transformer. In Proceedings of the 9th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 7028–7041. Association for Computational Linguistics, 2021.

9. Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Retrieval augmented language model pre-training. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 3929–3938. PMLR, 2020.

10. William L. Hamilton. Graph representation learning. Synthesis Lectures on Artificial Intelligence and Machine Learning, 14(3):1–159, 2020.

11. Ehsan Hosseini-Asl, Bryan McCann, Chien-Sheng Wu, Semih Yavuz, and Richard Socher. A simple language model for task-oriented dialogue. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
```

### --- Page 0011 ---

```markdown
| Reference                                                                                                           |
|---------------------------------------------------------------------------------------------------------------------|
| [12] Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=HygxnNFh. |
| [13] Jaehyeong Jo, Jinheon Baek, Seul Lee, Dongki Kim, Minki Kang, and Sung Ju Hwang. Edge representation learning with hypergraphs. CoRR, abs/2106.15845, 2021. |
| [14] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. |
| [15] Mojtaba Komeili, Kurt Shuster, and Jason Weston. Internet-augmented dialogue generation. CoRR, abs/2107.07566, 2021. |
| [16] Mike Lewis, Yinhun Liu, Naman Goyal, Marjan Ghazvininejad, Abdelfattah Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 7871–7880. Association for Computational Linguistics, 2020. |
| [17] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Lin Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Huan Sun-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. |
| [18] Jiwei Li, Michael Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. A diversity-promoting objective function for neural conversation models. In Kevin Knight, N. Kencova, and Owen Rambow, editors, NAACL HLT 2016, The 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego California, USA, June 12-17, 2016, pages 110–119. The Association for Computational Linguistics, 2016. URL https://doi.org/10.18653/v1/n16-1014. |
| [19] Yu Li, Baolin Peng, Yelong Shen, Yi Mao, Lars Liden, Zhou Yu, and Jianfeng Gao. Knowledge-grounded dialogue generation with a unified knowledge interface. In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 206–218. Association for Computational Linguistics, 2022. URL https://aclanthology.org/2022.naacl-main.15. |
| [20] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013. |
| [21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. URL https://openreview.net/forum?id=BkgR6iCqY7. |
| [22] Kaixin Ma, Hao Cheng, Xiaodong Liu, Eric Nyberg, and Jianfeng Gao. Open domain question answering with a unified knowledge interface. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 1605–1620. Association for Computational Linguistics, 2022. URL https://doi.org/10.18653/v1/2022.acl-long.113. |
| [23] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan T. McDonald. On faithfulness and factuality in abstractive summarization. In Dan Jurafsky, Joyce Chai, Natalie Schluter, |
```

### --- Page 0012 ---

```markdown
| Reference                                                                                                           |
|---------------------------------------------------------------------------------------------------------------------|
| [24] Seungwhan Moon, Pararth Shah, Anuj Kumar, and Rajen Subba. OpenDialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs. In Anna Korhonen, David R. Traum, and Lluis Marquez, editors, Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28-August 2, 2019, Volume 1: Long Papers, pages 845–854. Association for Computational Linguistics, 2019. |
| [25] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, July 6-12, 2002, Philadelphia, PA, USA, pages 311–318. ACL, 2002. |
| [26] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Ben Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32, pages 8024–8035, 2019. |
| [27] Ethan Perez, Florian Strub, Harme de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In Sheila A. McIlraith and Kilian Q. Q. Weinberger, editors, Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th Innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 7-8, 2018, pages 3942–3951. AAAI Press, 2018. URL https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16528. |
| [28] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick S. H. Lewis, Anton Bakhtin, Yuankai Wu, and Alexander H. Miller. Language models as knowledge bases? In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 2463–2473. Association for Computational Linguistics, 2019. URL https://doi.org/10.18653/v1/D19-1250. |
| [29] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 1(8), 2019. |
| [30] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:1401:1–1406:20, 2020. |
| [31] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000 questions for machine comprehension of text. In Jian Su, Xavier Carreras, and Kevin Duh, editors, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016, pages 2383–2392. The Association for Computational Linguistics, 2016. URL https://doi.org/10.18653/v1/D16-1264. |
| [32] Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. In Yoshua Bengio and Yann LeCun, editors, 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016. |
| [33] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr., 3(4):333–389, 2009. |
```

### --- Page 0013 ---

```markdown
| Reference                                                                                                           |
|---------------------------------------------------------------------------------------------------------------------|
| [34] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun'ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 4035–4045. Association for Computational Linguistics, 2018. URL https://doi.org/10.18653/v1/d18-1437. |
| [36] Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhun Liu, Jing Xu, Myle Ott, Eric Michael Smith, Y-Lan Boureau, and Jason Weston. Recipes for building an open-domain chatbot. In Paola Merlo, Jörg Tiedemann, and Reut Tsarfaty, editors, Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021, pages 300–325. Association for Computational Linguistics, 2021. |
| [36] Md. Rashad Al Hasan Rony, Ricardo Usbeck, and Jens Lehmann. Dialogk: Knowledge-structure aware task-oriented dialogue generation. In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruiz, editors, Findings of the Association for Computational Linguistics: NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 2575–2571. Association for Computational Linguistics, 2022. URL https://doi.org/10.18653/v1/2022.findings-naacl.195. |
| [37] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation for rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers. The Association for Computer Linguistics, 2016. |
| [38] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces hallucination in conversation. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November, 2021, pages 3784–3803. Association for Computational Linguistics, 2021. |
| [39] Yi-Lin Tuan, Yun-Nung Chen, and Hung-yi Lee. Dyckchat: Benchmarking dialogue generation grounded on dynamic knowledge graphs. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 1855–1865. Association for Computational Linguistics, 2019. |
| [40] Yi-Lin Tuan, Sajjad Beygi, Maryam Fazel-Zarani, Qiaozio Gao, Alessandra Cercone, and William Tang Wang. Towards large-scale interpretable knowledge graph reasoning for dialogue systems. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 383–395. Association for Computational Linguistics, 2022. URL https://doi.org/10.18653/v1/2022.findings-acl.33. |
| [41] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008. URL http://jmlr.org/papers/v9/vandermaaten08a.html. |
| [42] Shikhar Yashshith, Soumya Sanyal, Vikram Nitin, and Partha P. Talukdar. Composition-based multi-relational graph convolutional networks. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. |
| [43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Ilia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. |
| [44] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In 6th International Conference on Learning Representations, |
```

### --- Page 0014 ---

```markdown
ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018.

| Reference | Citation |
|-----------|----------|
| [45] | Denny Vrandečić and Markus Krötzsch. Wikidata: a free collaborative knowledgebase. Commun. ACM, 57(10):78–85, 2014. |
| [46] | Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Sheffer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Pl, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In EMNLP 2020 - Demos, Online, November 16-20, 2020, pages 38–45, 2020. URL https://doi.org/10.18653/v1/2020.emnlp-demos.6. |
| [47] | Sixing Wu, Ying Li, Dawei Zhang, Yang Zhou, and Zhonghai Wu. Diverse and informative dialogue generation with context-specific commonsense knowledge awareness. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 5811–5820. Association for Computational Linguistics, 2020. URL https://doi.org/10.18653/v1/2020.acl-main.515. |
| [48] | Manzil Zaheer, Jatwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. Deep sets. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 3391–3401, 2017. |
| [49] | Houyu Zhang, Zhenghao Liu, Chenyang Xiong, and Zhiyuan Liu. Grounded conversation generation using guided traverses on commonsense knowledge graphs. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 2031–2043. Association for Computational Linguistics, 2020. URL https://doi.org/10.18653/v1/2020.acl-main.184. |
| [50] | Yizhe Zhang, Sigi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. Dialogpt: Large-scale generative pre-training for conversational response generation. In ACL, system demonstration, 2020. |
| [51] | Hao Zhou, Minlie Huang, Yong Liu, Wei Chen, and Xiaoyan Zhu. EARL: informative knowledge-grounded conversation generation with entity-agnostic representation learning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 2383–2395. Association for Computational Linguistics, 2021. |
```

### --- Page 0015 ---

```markdown
# Appendix

## A  Limitations

As discussed in Appendix H, our work is limited in a variety of dimensions primarily in terms of the lack of a well-formulated public dataset, retrieval accuracy, and generation quality. First, the public benchmark dataset for knowledge-consistent dialogue generation is highly limited. Despite the fact that there are several public Knowledge Graphs (KGs) available [45, 3], only one dataset [24] provides the diverse set of dialogue and its corresponding large-scale KG. This circumstance may limit the rigorous evaluation of our framework’s adaptability in various settings. Future work may study applying our approach for a wider range of dialogue datasets based on Wikipedia [6] by leveraging existing public large-scale KG such as Wikidata [45]. Second, the search space for retrieving context-relevant subgraphs can be expanded. Our SURGE framework now runs on a $k$-hop KG that is rooted in entities in the given dialogue history. Finding the entity within the text, on the other hand, necessitates precise named entity extraction and entity linking. Therefore, future work may investigate extending our approach to a framework that can retrieve the context-relevant subgraph among the entire KG instead of $k$-hop KG. Third, there is still room for improvement in generation quality since we generate knowledge-enhanced responses with a small-scale Pre-trained Language Model (PLM) for efficiency. Such PLMs occasionally fail to generate high-quality natural sentences [30]. Future work could improve generation quality based on a larger PLM.

## B  Broader Impacts

Our proposed knowledge-grounded dialogue generation model is essential for designing user-friendly real-world AI systems. Among various types of dialogue generation models, knowledge-grounded dialogue models are trained to interact with users and convey factual information to users in natural languages. Their conversational interfaces can be adapted to any user interface that connects bilateral interaction between humans and computers. We believe that conversational interfaces can enhance the users’ experiences and reduce the users’ efforts in learning how to use the systems. However, knowledge-grounded dialogue models can become vulnerable to generating offensive and harmful content or responses with misinformation depending on the users or data. When deploying the models in the real world, in addition to generating realistic responses, they also need to be robust to adversarial feedback from malicious users and biases inherited in pre-training or training corpus, or else they could malfunction. Therefore, along with the quantitative and qualitative evaluations of generated responses, it would be worthwhile to examine the robustness of the dialogue models.

## C  Notations

We organize the notations we used for formally describing our method in Table 8.

## D  Intuitions & Proofs for Graph Encoding

### D.1  Intuitions

In Section 3.4, we focus to introduce the novel graph encoding which meets both permutation and relation inversion invariances. However, one may draw the question of why such invariances are important for graph encoding with the pre-trained language models (PLMs) and need more detailed explanations on this.

First of all, we want to recapitulate why the permutation invariance is important in encoding multi-relational graphs into the PLMs along with the text sequence. As noted in Section 3.1, PLMs are permutation sensitive since the meaning of the sentence can vary when we change the order of words in the sentence (e.g., "A is born in C" ≠ "C is born in A"). However, the multi-relational graphs are permutation invariant since they are represented as a set of triplets. For instance, given the multi-relational graphs with two triplets, $(a, born-in, c)$, $(b, born-in, d)$, the order of elements (triplets) does not affect the entire semantic of the graph. (e.g., $((a, born-in, c), (b, born-in, d)) = ((b, born-in, d), (a, born-in, c))$).
```

### --- Page 0016 ---

```markdown
| Notation | Description |
|----------|-------------|
| $\mathcal{V}$ | pre-defined vocabulary of tokens for pre-trained language models (text) |
| $\mathcal{E}$ | pre-defined vocabulary of entities (symbol) |
| $\mathcal{R}$ | pre-defined vocabulary of relations (symbol) |
| $a, \ldots, z$ | knowledge graph symbols written in typewriter font |
| $\mathbf{x}$ | input sequence (vector) |
| $\mathbf{x}_1, \ldots, \mathbf{x}_N$ | input tokens (scalar) |
| $\mathbf{y}$ | $[\mathbf{y}_1, \ldots, \mathbf{y}_T]$ output sequence and tokens |
| $\mathcal{G}$ | multi-relational graph, such as knowledge graph |
| $\mathcal{Z}$ | retrieved subgraph: $\mathcal{Z} \subset \mathcal{G}$ |
| $z$ | triplet (edge): $z \in \mathcal{Z}$ |
| $q$ | tokenization (mapping) function of KG symbol to the sequence of tokens |
| $s(\cdot)$ | text representation function for retrieval |
| $d(\cdot)$ | triplet representation function for retrieval |
| $\text{Enc}$ | Transformer Encoder |
| $\text{Dec}$ | Transformer Decoder |
| $f$ | token (word) embedding function |
| $\theta$ | generator parameter |
| $\phi$ | retriever parameter |
| $\psi$ | set encoding function |
| $\beta$ | perturbation function |
| $n$ | the number of triplets in a retrieved subgraph $\mathcal{Z}$ |
| $k$ | the number of samples in a marginalization term |
| $z$ | encoder hidden state (single token) |
| $\mathcal{Z}$ | encoder hidden states (sequence of tokens) |
| $h$ | decoder hidden state (single token) |
| $\mathcal{H}$ | decoder hidden states (sequence of tokens) |
| $\mathbf{X}$ | input embeddings after token embedding function (sequence) |
| $\mathbf{Y}$ | output embeddings after token embedding function (sequence) |

With a naïve encoding, the PLM yields different representations for different orders of triplets in the subgraph. Therefore, if the PLM is only fine-tuned with the input $[\mathbf{A}, \text{born-in}, \mathbf{C}, \mathbf{B}, \text{born-in}, \mathbf{D}, \text{where was A born?}]$, there is no guarantee that the PLM will output the exact same response given the input with a permuted subgraph $[\mathbf{B}, \text{born-in}, \mathbf{D}, \text{A}, \text{born-in}, \mathbf{C}, \text{where was A born?}]$ in the inference since the PLM is order-sensitive due to its positional encoding. In order to prevent the aforementioned scenarios, we decide to design the permutation-invariant graph encoding which yields stable results regardless of the order of triplets in the graph. Similarly, the inversion of the triplet yields the same semantic (e.g., $({\text{A}, \text{born-in}, \text{C}}) = ({\text{C}, \text{born-in}, \text{A}})$), but the graph encoding without considerations for the inverse relation results in different representations from PLM given the triplet and its inversed one.

### D.2 Proofs

In this section, we first show that a naïve encoding function $\psi$ in Section 3.4 is neither permutation invariant nor relation inversion invariant, formalized in Proposition D.1. After that, we prove that our invariant and efficient encoding function $\psi^*$ with graph-conditioned token embedding perturbation is both permutation invariant and relation inversion invariant, formalized in Proposition D.2.

**Proposition D.1.** A naïve encoding function $\psi$ is neither permutation invariant nor relation inversion invariant.

**Proof.** We prove this by contradiction.

Suppose $\mathbf{x} = [\mathbf{x}_1, \ldots, \mathbf{x}_n]$ and $\mathcal{Z} = \{(\mathbf{a}, \mathbf{r}_1, \mathbf{b}), (\mathbf{b}, \mathbf{r}_2, \mathbf{a}), (\mathbf{a}, \mathbf{r}_1, \mathbf{c})\}$. Moreover, let $\mathcal{Z}' = \{(\mathbf{b}, \mathbf{r}_2, \mathbf{a}), (\mathbf{a}, \mathbf{r}_1, \mathbf{b}), (\mathbf{a}, \mathbf{r}_1, \mathbf{c})\}$ be one of permutations of $\mathcal{Z}$ with the permutation order $\pi = (2, 1, 3)$.
```

### --- Page 0017 ---

```markdown
With a naïve encoding, $\psi(x, Z) = [a, r_1, b, b, r_2, a, a, r_2, c, x_1, \ldots, x_n]$ and $\psi(x, Z') = [b, z, a, r_1, b, a, r_1, c, x_1, \ldots, x_n]$. Therefore, it is easy to notice that $\psi(x, Z) \neq \psi(x, Z')$, thus the naïve encoding is not permutation invariant.

We then show that a naïve encoding is not relation invariant invariant. Suppose $Z'' = [(a, r_1), (b, a), (c, "^{-1}")],$ where $(a, r_1) \in Z$ is changed to its inverse relation $(c, "^{-1}")$. Then, $\psi(x, Z'') = [a, d, b, e, c, d_a, z_1, \ldots, z_n]$ that is different against $\psi(x, Z'')$: $\psi(x, Z) \neq \psi(x, Z'')$. Therefore, the naïve encoding function is not relation invariant.

In conclusion, from the above two counterexamples, we prove that a naïve encoding function $\psi$ is neither permutation invariant nor relation invariant.

We now provide proof of the permutation invariance and the relation inversion invariance of our invariant and effective graph encoding $\psi^*$, described in Section 3.4. Before starting the proof, we first revisit the permutation invariant property of graph neural networks that sum, mean and max operators are permutation invariant for the input set of AGGR. Thus, if we use sum, mean, or max for AGGR, then the token embedding perturbation function $\beta$ naturally satisfies the permutation invariance property. In other words, $\beta(X, Z) = \beta(X, \pi \cdot Z)$. Hence, $X = \psi^*(x, SORT(ENT(Z)))$ for any permutation $\pi$.

**Proposition 2.** Invariant and efficient encoding $\psi^*$ is both permutation invariant and relation inversion invariant.

Proof. Suppose $X = [x_1, \ldots, x_n]$ and $Z = [(a, r_1), (b, r_2), (a, r_1), (c, x_1), \ldots]$. We first consider the permutation invariance for any permuted set $Z' = \pi \cdot Z$. While $Z$ and $Z'$ can have different orders of elements thus the outputs of $ENT(Z)$ and $ENT(Z')$ could be different, we always obtain the same output with the usage of the SORT operator for encoding. In other words, $SORT(ENT(Z'))$ holds to be identical to the encoding of $SORT(ENT(Z))$. Thus, $\psi(x, SORT(ENT(Z))) = \psi(x, SORT(ENT(Z')))$. 

Further, since the token embedding perturbation function $\beta(\cdot, Z)$ along with sum, max, or mean in AGGR is also permutation invariant with regards to any permutation on $Z$, we conclude our invariant and efficient encoding $\psi^*$ is permutation invariant.

We finally prove the relation inversion invariance property of $\psi^*$. Suppose $Z'' = (Z \cup t') \setminus t$ where $t \in Z$ is any triplet in a set and $t'$ is inverse of $t$. Then, $ENT(Z') = ENT(Z'')$ that is trivial as $ENT(Z)$ returns the set of only unique nodes in $Z$. Therefore, $\psi(x, SORT(ENT(Z))) = \psi(x, SORT(ENT(Z'')))$ correspondingly holds.

The remaining step to conclude the proof is to show the following equality: $\beta(\cdot, INV(Z)) = \beta(\cdot, SORT(ENT(Z''))) = \psi^*(x, Z) = \beta(\cdot, SORT(ENT(Z'')))$, INV($Z$) = $\beta(\cdot, SORT(ENT(Z'')))$, INV($Z'')$. We note that INV($Z$) = INV($Z''$), as INV makes any graph as bidirectional one by the definition in Eq. 6 of the main paper. Therefore, $\beta(\cdot, INV(Z''))$ holds, and the relation inversion invariance property of $\psi^*$ holds.

## E Experimental Setup

In this section, we introduce the detailed experimental setups for our models and baselines. Specifically, we describe the details on implementation, dataset, training and model in the following subsections of E.1, E.2, E.3 and E.4, one by one.

### E.1 Implementation Details

We use the T5-small [30] as the base Pre-trained Language Model (PLM) for all experiments. For the pre-trained checkpoint, we use the version that the authors released. For all implementations, we use PyTorch [26]. To easily implement the language model, we use the huggingface transformers library [46].
```


### --- Page 0018 ---

```markdown
![GNN-based Triplet Representation for Retrieval. To represent each triplet with regards to its graph structure, we use the message passing on both nodes and edges. (a) Node-level Message Passing. To represent the entity Sense and Sensibility, the message from its neighbors – the entity Jane Austen – is aggregated. (b) Edge-level Message Passing. To represent the relation written_by, the messages from relations associated to a green hyperedge are aggregated. We do not draw self-loops and inverse edges for simplicity.](assets/page_0018_img_1.png)

## Retriever Details

In this paragraph, we describe the implementation details of our context-relevant subgraph retriever, including the triplet embedding and dialogue context embedding for the retriever.

For the dialogue history embedding function $q$, we use the existing pre-trained language model (PLM). Specifically, we use the encoder part of the T5-small model \cite{30} and freeze the parameters of it not to be trained. We then instead add a Multi-Layer Perceptron (MLP) on top of it, to give a point-wise attention \cite{2} to each token, whereby all tokens are not equally considered in the sentence encoding. Formally,

$$
q(x) = \sum_{i=1}^{n} \alpha_i * z_i, \quad Z = [z_1, \ldots, z_n] = \text{Enc}(X),
$$

where $\alpha_i$ is a scalar, and MLP is a Multi-Layer Perceptron consisting of two linear layers and ReLU nonlinearity.

For obtaining triplet representations, we need to embed the entity (node) and relation (edge) into the latent space. Similar to the token embedding matrix used in PLMs, we can introduce the entity and relation embedding matrices. However, since the number of entities used in Freebase of OpenDialKG \cite{24} is too large compared to the number of tokens in T5 (100,814 vs 32,000) \cite{30}, it is inefficient to introduce the trainable entity embedding matrix for the retriever. Furthermore, the use of standalone entity embedding matrix might be sub-optimal in terms of generalization since there is no evidence that all entities in a large-scale KG emerge in training dataset.

Thus, we instead reuse the contextualized representation from the PLM encoder, to embed each node if the corresponding entity exists in the dialogue context. Formally, suppose that there is a triplet $(e_k, r, e_j)$ in the 1-hop subgraph $G$, which satisfies the following condition: $e_k \in r(e) \subseteq E$. If so, we can know the position of the mapped entity within the dialogue history: $[start_{k}, \ldots, end_{k}] = q(e_k) \subseteq Z$. Therefore, the node embedding for the entity $e_k$ is obtained by 

$$
\text{EntEmb}(e_k) = \frac{1}{|start_k|} \sum_{j \in start_k} \text{Enc}(X)_j \quad \text{if } e_k \in E. 
$$

If the entity does not exist in the dialogue history, we use the zero vector as the node embedding. For edge embedding, we use the trainable relation embedding matrix $R \in \mathbb{R}^{|R| \times 128}$ to represent the edge, since the number of relations is relatively small (1,357).

With our node and edge representations, we now focus on representing the triplet in Eq. 4 of the main paper for its retrieval. In particular, we use the Graph Neural Networks (GNNs) for encoding triplets, where we obtain the node representations from the Graph Convolutional Network (GCN) \cite{14} that is a widely used architecture for representing the nodes with respect to their graph structures. However, for representing the edges, we use the Edge Hypergraph Graph Neural Network (EHGNN) used in Jo et al. \cite{13}, due to its simplicity but effectiveness for edge representations. We summarize our triplet representation in Figure 6.
```

### --- Page 0019 ---

```markdown
![Comprehensive diagram for Invariant and Efficient graph encoding.](assets/page_0019_img_1.png)

## Graph Encoding Details
In this paragraph, we describe the implementation details of the token embedding perturbation function $\beta$ used in our Invariant and Efficient graph encoding introduced in Section 3.4. To be aware of the relation of the graph over GNNs, we use the simplified version of the inverse relation from its opposite relation, we use the same linear layer. Also, we use subtraction as the specific composition operator for reflecting relations in CompGCN.

Then, we form the learnable affine transformation based on the aggregated representation from GNN as follows:

$$
\eta = R-GNN(f(\alpha); Z) = UPD(f(\delta), AGGR(\{f(\beta), x | \, \forall \, b \in N(\alpha; Z)\}))
$$

where $MLP_1(\eta), \, \delta = MLP_2(\eta), \, \beta(f(\alpha), Z) = (1 + \gamma) * f(\alpha) + \delta$,

$\gamma = MLP_1(\eta), \, \delta = MLP_2(\eta), \, \beta(f(\alpha), Z) = (1 + \gamma) * f(\alpha) + \delta$.

In Figure 7, we illustrate comprehensive diagram of Equation 6, which enables our Invariant and Efficient graph encoding to understand the structure of the retrieved subgraph $Z$.

## Contrastive Learning Details
For contrastive learning, we initialize $\tau$ in Equation 7 as 0.01.

## KQA Details
In this paragraph, we describe the implementation details for our Knowledge-verifying Question Answering (KQA) introduced in Section 4. For building the QA dataset, we first gather the dialogue sessions where the gold response contains the entity from the whole OpenDialKG dataset. Then, we extract the triplet from the given whole KG where the head entity is placed within the dialogue history and the tail entity is placed within the gold response. We build a QA training dataset based on the extracted triplets and a corresponding dialogue session. To diversify the training data, we replace the tail entity of each triplet with plausible candidate entities within KG and change the entity in the response following the changed entity on the triplet. As a result, we obtain the QA dataset size of 200K. We train the BERT-base [5] with the constructed QA dataset. We hold out 10% of data for validation and obtain the fine-tuned BERT model with 88.89 F1 score on the hold-out validation set. When we apply the fine-tuned BERT model on the evaluation of the generated responses, we rebuilt the QA evaluation set with the generated response instead of a gold response as illustrated in Figure 3 of the main paper.

## E.2 Dataset Details
We mainly conduct experiments on OpenDialKG [24], which provides the parallel dialogue corpus corresponding to the existing large-scale Knowledge Graph (KG) named Freebase [3]. The provided large-scale KG consists of 1,190,658 fact triplets over 100,813 entities and 1,358 relations. This dataset is collected from 15K human-to-human role-playing dialogues, having multi-turns, from which we pre-process that each assistance response is the label and its corresponding dialogue history is the input. Although some of the data contain the gold knowledge that is useful for generating the response on the ongoing conversation, we found that 51% of data has no gold knowledge.
```

### --- Page 0020 ---

```markdown
## E.3 Training Details

All experiments are constrained to be done with a single 48GB Quadro 8000 GPU. SURGE training needs 12 GPU hours. For all experiments, we select the best checkpoint on the validation set. We fine-tune the SURGE for 30 epochs on the training set, where we set the learning rate as 1e-4, weight decay as 0.01, learning rate decay warmup rate as 0.06, maximum sequence length for dialogue history as 256, maximum sequence length for knowledge as 128, and batch size as 24. For retrieval, we use the subgraph size as 3, and sample size k for marginalization as 4. We use the AdamW [21] optimizer for training. For fair evaluation, we apply the same training setting to all baselines if applicable. All experimental results are reported with a single run.

## E.4 Model & Baselines Details

In this subsection, we describe the details of baselines and our models used in our experiments, as follows:

1. **No Knowledge**: This model is provided with only the dialog history. No knowledge is used to generate responses.
   
2. **Gold Knowledge**: This model is provided with the dialogue history along with its exact gold knowledge for the gold response. Thus, since this model uses such gold knowledge, we expect the results of it as the upper bound of the task.
   
3. **Space Efficient (parallel)**: This model is mostly the same as the above model – space Efficient (series) – except the knowledge encoding part. Specifically, it encodes the entities in the words like the above, whereas, encoding the relation between entities in the segmentation block of the entities [8].
   
4. **EARL**: This model uses the RNN-based encoder-decoder architecture with the entity-agnostic representation learning [51], with all the provided knowledge associated with the entities in the dialogue history. Specifically, this model first calculates the probability of words obtained by encoding the entities in the KG, and then uses such probabilities to generate a word in the decoding phase.
   
5. **DiffKG**: This model [40] uses a differentiable path reasoning, which is jointly trainable along with the dialogue generation. After the path reasoning, the entities in the reasoning path are naively appended in front of the dialogue history, then concatenated input is forwarded to the pre-trained language model.
   
6. **Random Retrieval**: This model is provided with entire facts from k-hop subgraphs of entities that appeared in the dialogue history. However, instead of encoding all the knowledge in one-hop subgraph as Space Efficient, this model randomly samples them, which are then used for generating responses.
   
7. **Sparse Retrieval (BM25)**: This model is also provided with entire facts from k-hop subgraphs of entities. To sample relevant facts to the dialogue history among the entire facts, this model uses BM25 [33] that is a sparse retrieval model. To be specific, let assume we have a dialogue history and its corresponding facts from k-hop subgraphs of matched entities. Then, to run BM25, we first concatenate components of each consisting of two entities and one relation, and tokenize the dialogue history and the facts for obtaining corpus and queries, respectively, for BM25. After that, BM25 calculates the lexical overlapping score between the dialogue context (corpus) and the one-hop fact (query), from which we use the relevant facts having top-k scores by BM25.
```


### --- Page 0021 ---

```markdown
![Graph showing the variation of the number of facts in retrieved subgraphs](assets/page_0021_img_1.png)

| Method                          | MRR   | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|---------------------------------|-------|--------|--------|--------|---------|
| Random Retrieval                | 6.87  | 3.11   | 7.09   | 13.03  | 17.64   |
| Space Retrieval (BM25)         | 7.76  | 6.27   | 9.79   | 16.71  | 26.61   |
| Dense Retrieval (Bi-encoder)    | 20.39 | 11.99  | 30.94  | 39.26  | 67.26   |
| Dense Retrieval (RNN-encoder)   | 21.47 | 12.13  | 23.12  | 31.51  | 41.63   |
| SURGE (unsupervised)           | 21.24 | 10.46  | 31.97  | 41.77  | 74.74   |
| SURGE (semi-supervised)        | 22.53 | 12.40  | 24.79  | 31.94  | 42.67   |
| SURGE (contrastive)            | 25.98 | 16.67  | 28.50  | 42.67  | 74.31   |

9. **Dense Retrieval** (Bi-encoder, Poly-encoder): This model uses a pre-trained language model for the triplet embedding of the retriever instead of using GNN. Specifically, we consider each triplet as a single sentence (e.g., “Jane Austen, write, Susan”) → “Jane Austen write Susan” and embed them with the pre-trained model. 

10. **SURGE (unsupervised)**: Our basic subgraph retrieval-augmented generation framework that is provided with entire facts from k-hop subgraphs of entities. In particular, this model trains the structure-aware subgraph retriever without any guidance of the gold knowledge (i.e., ground truth knowledge for the dialogue history is not given). In other words, for the given dialogue context, this model implicitly learns to retrieve the context-relevant knowledge, and then generates the response with the retrieved knowledge.

11. **SURGE (semi-supervised)**: Our subgraph retrieval-augmented generation framework with supervised learning of graph retrieval, with provided entire facts from k-hop subgraphs of entities. Unlike the unsupervised version of SURGE, this model trains the retriever to select the gold knowledge if the dialogue context has such knowledge during training.

12. **SURGE (contrastive)**: Our full subgraph retrieval-augmented generation framework with the contrastive learning of graph-text modalities as well as the semi-supervised learning of graph retrieval, with provided entire facts from k-hop subgraphs of entities. Unlike aforementioned frameworks of ours, this additionally enforces the model to faithfully reflect the retrieved knowledge in the input, to the generated response with contrastive learning.

F. **Additional Experiments**

F.1 **Varying the Number of Facts in Subgraphs**

We experiment our SURGE framework with varying the number of facts in retrieval, which are then used in our graph encoding function to condition the encoded graph information for response generation. Specifically, in Figure 8, we report the length of sequence for knowledge (knowledge length) and F1 scores measured by our KQA for our SURGE framework, with different numbers of facts within a retrieved subgraph: $n = \{3, 5, 10\}$. Note that, in this experiment, we only use the semi-supervised model without the contrastive loss. We expect that the performance of our SURGE will increase as we increase the number of facts within the retrieved subgraph, since the model can leverage more numbers of knowledge for response generation. As shown in Figure 8, we observe the significant performance improvements on using ten facts against using three and five.
```

### --- Page 0022 ---

```markdown
| Method                          | KQA         | BLEU       | ROUGE      | Unigram   |
|---------------------------------|-------------|------------|------------|-----------|
|                                 | EM  | F1  | B-1 | B-2 | B-3 | B-4 | R-L | R-L | F1  |
| Random                          | 12.41 | 14.70 | 7.74 | 4.02 | 2.46 | 1.68 | 21.79 | 4.00 | 21.44 | 16.29 |
| Space Efficient (Series)       | 12.41 | 14.70 | 8.34 | 5.13 | 3.77 | 2.36 | 4.85 | 2.60 | 17.37 |
| Space Efficient (Parallel)     | 16.46 | 18.70 | 9.33 | 5.66 | 4.06 | 3.20 | 22.80 | 4.12 | 22.47 | 17.72 |
| SURGE (unsupervised)           | 16.18 | 18.51 | 11.46 | 7.10 | 5.15 | 4.07 | 23.49 | 5.77 | 23.09 | 18.70 |
| SURGE (semi-supervised)        | 16.62 | 19.43 | 11.28 | 6.98 | 5.05 | 3.98 | 23.58 | 5.79 | 23.21 | 18.68 |
| SURGE (contrastive)            | 17.30 | 19.50 | 11.51 | 7.18 | 5.20 | 4.10 | 21.43 | 6.17 | 23.74 | 19.51 |

![Experimental results on KOMODIS dataset with T5-small](assets/page_0022_img_1.png)

## F2  Discussions on Using Larger PLMs

Notably, we observe that the use of larger Pre-trained Language Models (PLMs) – three times more number of parameters compared to T5-small that we use – does not result in better performance for knowledge-grounded dialogue tasks. Specifically, in Table 9, we report the experimental results of selected baselines and our SURGE semi-supervised model with BART-base [16] as the base PLM. We want to clarify that the BART-base model has 220M parameters, which is about three times larger than the number of parameters of the T5-small model (60M).

We first observe that BART-base shows decent performance without any knowledge (No Knowledge) compared to the no-knowledge case of T5-small, verifying that the larger PLM generally contains more factual knowledge infused in its trained parameters. Moreover, BART-base obtains higher scores in the simple word overlap metrics such as BLEU [25] and ROUGE [20], whose results further confirm that a larger PLM can generate more natural or syntactically better sentences than the smaller one, thanks to its parameter size.

On the other hand, we find that BART-base is less suffered from the irrelevant knowledge issue (i.e., conditioning irrelevant knowledge for the given context when generating responses) than T5-small, therefore, the performance of Space Efficient Encoding on KQA is quite high. However, the use of BART-base does not result in significant improvement on the KQA metric for our SURGE framework. Moreover, ours with T5-small shows better performance than ours with BART-base in terms of KQA scores, when the number of facts within the retrieved subgraph is $n = 10$. This result suggests that the quality of the generated response – having relevant knowledge to the given context – might depend on the performance of the subgraph retriever whose goal is to retrieve the context-relevant knowledge, rather than the inherent performance of PLMs.

## F3  Full Experimental Results on KOMODIS

In the main paper, we mostly focus on OpenDialKG dataset [24], since it is the largest and most realistic public datasets that provides both dialogues across diverse domains and corresponding large-scale Knowledge Graph (KG) [3]. To verify the effectiveness of our SURGE framework, the existence of the large-scale KG and the importance of relevant fact searching is important since we focus on the real-world scenario where the response generation requires the relevant fact acquired from the large-scale KG.

However, one can raise the question regarding the versatility of our method on other datasets. To alleviate the issue, we conduct additional experiments on another dataset named KOMODIS [7], which is also KG-based dialogue dataset. Compared to OpenDialKG, KOMODIS does not provide the corresponding large-scale KG and most of responses do not require the knowledge. Therefore, we only measure the automatic evaluation to evaluate the performance of each method on KOMODIS dataset. In Table 10, we present the experimental results on the KOMODIS dataset. Results obviously show that our SURGE framework shows superior performance against baselines on the additional dataset. Therefore, we can conclude that our method can generalize to other datasets beyond the opendialKG dataset.
```

### --- Page 0023 ---

```markdown
## F.4 Diversity Evaluation

In the main paper, we evaluate model generation performance primarily on its quality. We measure the distinct metric [18], which is one of the most popular metrics for evaluating the diversity of the generated model, to evaluate the performance of each model in more diverse aspects. In Table 11 left, we report the performance of baselines and our models in distinct metric. Our SURGE framework generates more diverse responses than all other baselines, according to the results.

## F.5 Ablations Studies on GNN Design Choices

We use two different types of Graph Neural Networks (GNN) in our SURGE framework. One is the Graph Convolutional Network (GCN) [14], which is used to embed each node entity on the entire 1-hop subgraph in the triplet embedding function $d$ of the main paper Equation 4. Another is Composition-Based Multi-Relational Graph Convolutional Networks (CompGCN) [42], which is used to embed each entity by considering the relations between entities in the token embedding perturbation function $\beta$ of the main paper Equation 6. In this subsection, we conduct ablation studies on both GNN design choices. First of all, we replace the GCN in Equation 4 with Graph Attention Network (GAT) [44] to validate the effect of the GNN design choices on the node embedding in the triplet embedding function. Then, we run experiments by changing CompGCN in Equation 6 to GCN to see how important the relationships are in the graph encoding. We present the results on Table 11 right. Results indicate that the use of GAT in Equation 3 does not have any impact on the performance a lot. However, the use of relation-aware GNN is highly important in efficiency and efficient graph encoding, since removing the relation awareness of GNN reduces the performance of our model a lot.

## F.6 Automatic Evaluations on Knowledge Groundedness

Our KQA metric introduced in Section 4 is useful to evaluate the knowledge groundedness of the generated response. Since the knowledge of the KQA metric stems from the use of the KG to resolve the issue from the missing knowledge by only considering the gold response for evaluation, we can also utilize other rule-based metrics like string matching (check whether at least one of the answer candidates of KQA presents in the generated response by string matching) or Entity F1 score (measuring the F1 score against each entity in answer candidates instead of the gold response). As we know, automatic evaluations can be imperfect when compared to human evaluations. However, we also believe that using a variety of credible automatic evaluation metrics will strengthen the validity of the experimental results. Therefore, we supplement the experimental results with three more evaluation metrics for measuring whether the generated responses contain appropriate knowledge.

In Table 12, we measure Knowledge F1 (KF1 in Table 3 in the main paper), string matching, and entity F1 for representative baselines and our SURGE (semi-supervised) in OpenDialKG, as an extension of Table 1 in the main paper. For KF1, we measure the F1 score regarding the concatenation of the question (head entity and relation) and all answer candidates (available tail entities) in KQA as the gold response. The results show that all metrics show the same tendency with KQA and our proposed method still outperforms other baselines by generating responses with more proper knowledge. Although three rule-based metrics are useful for assessing the knowledge groundedness of generated responses, they do have some drawbacks. KF1 and Entity F1 are affected by the length of the generated responses and answer candidates. String matching is too strict since it may miss some responses that only contain partial words of knowledge (e.g., the response only contains the first name of the author whereas the candidate answers contain the full name of the author). As a result, the use of KQA is also beneficial since the trained QA model can compensate for the shortcomings of rule-based metrics.

## G Human Evaluation

In this section, we describe the details of human evaluation used in Table 7 of the main paper. We request the annotators to evaluate the responses generated from two baselines (i.e., ALL Knowledge and Space Efficient) and our SURGE framework in response to the given dialogue context, according to three criteria – consistency, informativeness, and fluency. Figure 9 is the instructions provided to each annotator. Specifically, regarding the consistency metric, we ask annotators to check whether
```
![Detailed description of the chart](assets/page_0023_img_1.png)
```

### --- Page 0024 ---

```markdown
| Method                        | Dev1: B=2 | KQA   | BLEU | ROUGE | Unigram |
|-------------------------------|-----------|-------|-------|-------|---------|
| No Knowledge                  | 0.66      | 15.73 |       |       |         |
| All Knowledge                 | 2.43      | 34.75 |       |       |         |
| SEE (series)                 | 3.89      | 21.79 |       |       |         |
| SEE (parallel)               | 5.83      | 22.69 |       |       |         |
| EABR                         | 12.61     | 16.68 |       |       |         |
| Sparse Retrieval (B=2)       | 17.85     | 17.63 |       |       |         |
| SURGE (semi-supervised)      | 18.18     | 27.85 |       |       |         |

| Method                        | EM   | F1   | Knowledge F1 | Entity F1 | String Matching |
|-------------------------------|------|------|--------------|-----------|------------------|
| No Knowledge                  | 12.25| 20.69| 13.80        | 9.33      | 13.03            |
| Random Knowledge               | 31.72| 38.95| 16.29        | 16.49     | 32.71            |
| All Knowledge                 | 43.58| 50.61| 18.91        | 21.10     | 44.25            |
| Space Efficient (Parallel)    | 38.44| 44.34| 17.43        | 18.93     | 40.56            |
| Dense Retrieval (Poly-encoder) | 40.62| 52.57| 19.72        | 21.46     | 48.41            |
| DiffKG                        | 12.25| 20.99| 14.44        | 9.37      | 13.23            |
| SURGE (semi-supervised)      | 51.00| 57.63| 17.70        | 11.21     | 8.25             |

![Human Evaluation Instructions](assets/page_0024_img_1.png)
```


### --- Page 0025 ---

```markdown
![Large version of Figure 5 in the main paper. Stars indicate the embedding of graph and circles indicate the embedding of decoder hidden states (text), respectively.](assets/page_0025_img_1.png)

the generated response makes sense in the context of the conversation. For informativeness, we ask annotators to check whether the response contains correct and enough information, whereby experiment participants are recommended to use the internet search, to check whether the response contains correct facts. In addition to this, we also provide the dialogue-related facts from Freebase as a reference for fact checking for annotators. For fluency, we ask annotators to check whether the response is grammatically correct and naturally sound.

# H Retrieval and Generation Examples

In this section, we provide examples for knowledge retrieval and response generation, for the given dialogue history.

## Embedding Space Visualization

In Figure 10, we present a larger version of Figure 5 in the main paper. Specifically, we embed the hidden representations before the projection layer for each graph (star) and the embedding of the generated text (circle) through the dimensionality reduction using t-SNE [41]. As mentioned in the main paper, the visualization highlights that our SURGE framework with graph-text contrastive learning generates more distinct responses to different subgraphs, unlike the one without graph-text contrastive learning which shows less variety over responses even with different graphs.

## Retrieval Examples

We provide the retrieval examples of various models, such as random retrieval, sparse retrieval and our SURGE models. In particular, in the first (top) example of Figure 11, we are given a dialogue context in regard to books for Richard Maxwell, and baselines including random and BM25 retrievers select the facts associated to the entity Richard Maxwell, which are irrelevant to the ongoing conversion, for example, (Richard Maxwell, is a Theatre director). Also, as shown in the second (bottom) example of Figure 11, we observe that the simple term-based matching model (i.e., BM25) cannot contextualize the current and previous dialogues, but retrieves the facts associated to frequent words, for example, song, which are less meaningful for the user’s question. In contrast to baselines, as our SURGE framework trains a retriever in an end-to-end fashion, it first contextualizes the given dialogue context, and then accurately retrieves relevant knowledge.

## Generation Examples

We provide the generation examples from our model. To be specific, we provide the dialogue context along with its corresponding retrieved subgraph and generated response obtained from our SURGE framework. In Figure 12 and Figure 13, we provide the correct examples: our model retrieves a context-relevant subgraph, but also generates a factual response from retrieved knowledge. On the other hand, in Figure 14, we provide the failure cases. In particular, as shown in the first row of Figure 14, the fact in the knowledge graph could be ambiguous or inaccurate, as it defines the release year of the book - Wicked - as both 2008 and 2014. Moreover, we further provide the failure example on retrieval in the second row of Figure 14, where the user asks about the Bourne Legacy, while the dialogue agents retrieve the irrelevant knowledge to the question. Finally, we show the common problem in PLMs in the last row of Figure 14, where the generative model repeats the meaningless words at the end, while the retriever correctly selects the relevant knowledge.
```

### --- Page 0026 ---

```markdown
| Dialogue Context                                                                 |
|----------------------------------------------------------------------------------|
| A: Could you recommend any books written by Richard Maxwell?                    |

| Gold Knowledge                                                                   |
|----------------------------------------------------------------------------------|
| Richard maxwell, --written_by, a tale of two cities                             |

| Random Knowledge                                                                 |
|----------------------------------------------------------------------------------|
| Richard maxwell, sibling, jan maxwell                                            |
| Screenwriter, --is_a, Richard maxwell                                            |
| Theatre director, --is_a, Richard maxwell                                        |

| BM25 Knowledge                                                                   |
|----------------------------------------------------------------------------------|
| Richard maxwell, --is_a, Theatre director                                        |
| Screenwriter, --is_a, Richard maxwell                                            |
| Richard maxwell, organization founded, new york city players                     |

| Our Knowledge                                                                    |
|----------------------------------------------------------------------------------|
| Richard maxwell, --written_by, a tale of two cities                             |
| Richard maxwell, sibling, Jan maxwell                                            |

| Dialogue Context                                                                 |
|----------------------------------------------------------------------------------|
| A: I like Adam Levine.                                                           |
| B: OMG me too! I love that song Moves Like Jagger.                               |
| A: Yes, Love that too. It is really fun. Can you tell me more.                  |
| B: Did you know it's considered a power pop song?                                |
| A: No, I didn't. Do you know Love the way you lie?                              |

| Gold Knowledge                                                                   |
|----------------------------------------------------------------------------------|
| Song, --kind_of_composition, Love the way you lie                                |
| Love the way you lie, composer, Eminem                                           |

| Random Knowledge                                                                 |
|----------------------------------------------------------------------------------|
| Blue monday, kind of composition, Song                                            |
| The look of love, kind of composition, Song                                       |
| Bad romance, kind of composition, Song                                            |

| BM25 Knowledge                                                                   |
|----------------------------------------------------------------------------------|
| Song, --kind_of_composition, This charming man                                   |
| Behati prinsloo, --spouse (or domestic partner), Adam levine                     |
| Song, --kind_of_composition, safe & sound                                        |

| Our Knowledge                                                                    |
|----------------------------------------------------------------------------------|
| g minor, --key, Love the way you lie                                             |
| Eminem, --composer, Love the way you lie                                         |
| Skylar grey, --composer, Love the way you lie                                    |

![Examples of the dialogue history with its corresponding gold knowledge](assets/page_0026_img_1.png)
```

### --- Page 0027 ---

```markdown
| Dialogue Context                                                                 |                                                                                     |
|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| A: Do you like Rob Marshall? My friends have recommended his films.             |                                                                                     |
|                                                                                  |                                                                                     |
| Retrieved Knowledge                                                               |                                                                                     |
| Nine, has_genre, Drama                                                           |                                                                                     |
| Pirates of the Caribbean: On Stranger Tides, directed_by, Rob Marshall         |                                                                                     |
| Memoirs of a Geisha, directed_by, Rob Marshall                                  |                                                                                     |
|                                                                                  |                                                                                     |
| Generated Response                                                               |                                                                                     |
| B: Yes, he wrote Pirates of the Caribbean: On Stranger Tides, Memoirs of a Geisha Nine and many others. Have you seen either of those? |                                                                                     |
|                                                                                  |                                                                                     |
| Dialogue Context                                                                 |                                                                                     |
| A: Are there any works by Chris Cleave, which you can recommend me?            |                                                                                     |
| B: Certainly, wrote The Other Hand, Incendiary, and Old. Have you read them?   |                                                                                     |
| A: No, I haven't read The Other Hand out of the ones you mentioned. What genre is it? |                                                                                     |
|                                                                                  |                                                                                     |
| Retrieved Knowledge                                                               |                                                                                     |
| The Other Hand, release_year, 2008                                              |                                                                                     |
| 2008, -release_year, The Other Hand                                             |                                                                                     |
| The Other Hand, has_genre, Literary fiction                                      |                                                                                     |
|                                                                                  |                                                                                     |
| Generated Response                                                               |                                                                                     |
| B: It's Literary fiction, and was released in 2008.                             |                                                                                     |
|                                                                                  |                                                                                     |
| Dialogue Context                                                                 |                                                                                     |
| A: Who wrote Of Mice and Men?                                                   |                                                                                     |
| B: It was written by John Steinbeck.                                            |                                                                                     |
| A: Didn't he also write The Red Pony?                                           |                                                                                     |
| B: Yes He also wrote The Red Pony.                                              |                                                                                     |
| A: Can you suggest other titles by him to read?                                 |                                                                                     |
|                                                                                  |                                                                                     |
| Retrieved Knowledge                                                               |                                                                                     |
| Cannery Row, written_by, John Steinbeck                                         |                                                                                     |
| The Grapes of Wrath, written_by, John Steinbeck                                 |                                                                                     |
| Tortilla Flat, written_by, John Steinbeck                                       |                                                                                     |
|                                                                                  |                                                                                     |
| Generated Response                                                               |                                                                                     |
| B: Yes he also wrote Tortilla Flat.                                             |                                                                                     |

Figure 12: Examples of the dialogue history with its corresponding retrieved knowledge and generated response from our SURGE framework. The fact is represented as the format of (head, relation, tail), where ~symbol in the front of relation (i.e., ~relation) in the retrieved knowledge denotes the inverse relation. In this example, we only provide the correct cases of both retrieval and generation.
```

### --- Page 0028 ---

```markdown
| Dialogue Context                                                                 | Retrieved Knowledge                                                                                     | Generated Response                                                   |
|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| A: Could you recommend some movies by director Simon Wood?                      | The One That Got Away, written_by, Simon Wood                                                        | B: Simon Wood directed The One That Got Away. Have you seen that?   |
|                                                                                  | Simon Wood, ~written_by, The One That Got Away                                                       |                                                                     |
|                                                                                  | Author, ~is_a, Simon Wood                                                                             |                                                                     |
| Dialogue Context                                                                 | Retrieved Knowledge                                                                                     | Generated Response                                                   |
| A: I like David McCullough. Could you recommend any books of him?              | The Greater Journey: Americans In Paris, release_year, 2011                                          | B: Sure. He wrote The Greater Journey: Americans In Paris. Also, he wrote some documentary and indie films. |
| B: Thank you for the information. When was The Greater Journey: Americans In Paris released? | 2011, ~release_year, The Greater Journey: Americans In Paris                                         | A: It was released in 2011. It's a great book.                    |
|                                                                                  | Paris, written_by, Edward Rutherfurd                                                                 |                                                                     |
| Dialogue Context                                                                 | Retrieved Knowledge                                                                                     | Generated Response                                                   |
| A: I like Tony Tanner, what do you think?                                       | Sense and Sensibility, written_by, Tony Tanner                                                       | B: I think he is a great actor. He starred in Sense and Sensibility and Mansfield Park. |
|                                                                                  | Tony Tanner, ~written_by, Mansfield Park                                                              |                                                                     |
|                                                                                  | Mansfield Park, written_by, Tony Tanner                                                                |                                                                     |

Figure 13: Examples of the dialogue history with its corresponding retrieved knowledge and generated response from our SURGE framework. The fact is represented as the format of (head, relation, tail), where ~symbol in the front of relation (i.e., ~relation) in the retrieved knowledge denotes the inverse relation. In this example, we only provide the correct cases of both retrieval and generation.
```

### --- Page 0029 ---

```markdown
| Dialogue Context                                                                                     |                                                                                     |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| A: Do you know what books Jennifer L. Armentrout has written?                                      | B: She wrote Wicked, White House Kiss and Wait For You. Are you interested in her other books as well? |
| A: No thanks. What genre is Wicked?                                                                  | B: Wicked is a Play that won the Pulitzer Prize for Drama. Would you like more information about it? |
| A: Sure, sounds interesting                                                                           |                                                                                     |
| **Retrieved Knowledge**                                                                               |                                                                                     |
| Wicked, release_year, 2008                                                                            | 2008, ~release_year, Wicked                                                         |
| Wicked, release_year, 2014                                                                            |                                                                                     |
| **Generated Response**                                                                                |                                                                                     |
| B: It is a play written by Jennifer L. Armentrout and released in 2008.                            |                                                                                     |
| **Dialogue Context**                                                                                 |                                                                                     |
| A: I like the movie Armageddon.                                                                      | B: Did you know Armageddon is written by Tony Gilroy?                             |
| A: Oh, didn't he also write The Bourne Legacy?                                                      |                                                                                     |
| **Retrieved Knowledge**                                                                               |                                                                                     |
| Michael Clayton, starred_actors, Tony Gilroy                                                         |                                                                                     |
| Michael Clayton, written_by, Tony Gilroy                                                             |                                                                                     |
| Proof of Life, written_by, Tony Gilroy                                                               |                                                                                     |
| **Generated Response**                                                                                |                                                                                     |
| B: Yes, he did. He also wrote Michael Clayton.                                                      |                                                                                     |
| **Dialogue Context**                                                                                 |                                                                                     |
| A: I like Deadpool or any movies like Deadpool.                                                     |                                                                                     |
| **Retrieved Knowledge**                                                                               |                                                                                     |
| Deadpool, starred_actors, Ryan Reynolds                                                               |                                                                                     |
| Deadpool, starred_actors, Stefan Kapicic                                                             |                                                                                     |
| Deadpool, starred_actors, Ed Skrein                                                                  |                                                                                     |
| **Generated Response**                                                                                |                                                                                     |
| B: Deadpool is a great movie. Stefan Kapicic starred in it. Stefan Kapicic also starred in The Last Man and The Last Man. |                                                                                     |

Figure 14: Examples of the dialogue history with its corresponding retrieved knowledge and generated response from our SURGE framework. The fact is represented as the format of (head, relation, tail), where ~symbol in the front of relation (i.e., ~relation) in the retrieved knowledge denotes the inverse relation. In this example, we only provide the failure cases due to the problem on data (first row), retrieval (second row), and generation (third row).
```


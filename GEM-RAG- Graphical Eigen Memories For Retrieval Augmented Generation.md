# GEM-RAG- Graphical Eigen Memories For Retrieval Augmented Generation

### --- Page 0001 ---

```markdown
# GEM-RAG: Graphical Eigen Memories For Retrieval Augmented Generation

Brendan Hogan Rappazzo, Yingheng Wang, Aaron Ferber, Carla Gomes  
Department of Computer Science  
Cornell University  
Ithaca, New York 14850  
Email: {bhr54, yw2349, amf272, cpe5}@cornell.edu

---

**Abstract**—The ability to form, retrieve, and reason about memories in response to stimuli serves as the cornerstone for general intelligence - shaping entities capable of learning, adaptation, and intuitive insight. Large Language Models (LLMs) have proven their ability, given the proper memories or context, to reason and respond meaningfully to stimuli. However, they are still unable to optimally encode, store, and retrieve memories - the ability to do this would unlock their full ability to operate as AI agents, and to specialize in niche domains. To remedy this, one promising area of research is Retrieval Augmented Generation (RAG), which aims to augment LLMs by providing them with rich in-context examples and information. In question-answering (QA) applications, RAG methods embed the text of interest in chunks, and retrieve the most relevant chunks for a prompt using text embeddings. Motivated by human memory encoding and retrieval, we aim to improve over standard RAG methods by generating and encoding higher-level representations and tagging the chunks based on their utility. We introduce Graphical Eigen Memory for Retrieval Augmented Generation (GEM-RAG). GEM-RAG works by embedding each chunk of text in a given text corpus with LLM generated “utility” questions, connecting chunks in a graph based on the similarity of both their text and utility questions, and then using the eigen-decomposition of the memory graph to build higher level summary modes that capture the main themes of the text. As a result, GEM-RAG not only provides a more principled method for RAG tasks, but also synthesizes graphical eigen memory (GEM) which can be useful for both exploring text and understanding which components are relevant to a given question. We evaluate GEM-RAG, using both UnifiedQA and GPT-3.5 Turbo as the LLMs, with SBERT, and OpenAI’s text encoders on four standard QA tasks, showing that GEM-RAG outperforms other state-of-the-art RAG methods on these tasks. We also discuss the implications of having a robust RAG system and future directions.

## I. INTRODUCTION

The ability to create intelligent machines has long occupied the fascination of humankind, from the automata of the medieval era, formalization of logic in the 17th century, and evolving with the emergence of computing theory and artificial intelligence concepts in the 19th century. Now in the modern era, the possibility of generating Artificial General Intelligence (AGI) [1] seems as close as ever, in particular just within the past three years with the advent of massive scale machine learning systems, especially Large Language Models (LLMs). LLMs have emerged as remarkably powerful general knowledge stores, with the ability to perform impressively on a large variety of tasks [2], [3], [4]. Further, given their large context lengths, it has been shown that they have a powerful ability to perform in-context learning, whether it be for chat-like applications where it can reference parts of the conversation dynamically, adapting to classification tasks [5], or reasoning tasks through chain-of-thought prompting [6]. LLMs appear to have solved one part of the general intelligence equation, given the proper context, much like a human mind given the proper working memory stream, they can reason and respond to questions reasonably. But without a way to encode, store and retrieve information that extends outside of their context, LLMs are missing an extremely important part of general intelligence, they are unable to form long-term, and ongoing memories as AI agents, and are unable to adapt to new niche domains.

If LLMs were able to perfectly encode, store and retrieve memories it would open the possibility for AI agents to remember and reference conversations, research publications, literary works, and more, building hierarchies of memories and knowledge about the world. It would also advance QA models to read and encode huge amounts of other text documents, and provide robust conversations or QA citing content in those documents. Expanding this idea for LLMs, this would enable the ability to retrieve text, memories, audio, imagery, video, and more, paving the way for more potent models and use cases.

Motivated by this problem, retrieval augmented generation (RAG) has been an increasingly active area of research, which aims to adapt static LLMs to new and niche domain applications [7], [8], [9], [10], [11]. While RAG methods can work for any type of text-based information, whether it be memories [12] or images in the case of vision LLMs [13], one standard setting is that concerning large corpora of text from stories, articles, books, etc. In these settings, RAG generally works by first splitting the text into chunks, and obtaining an embedding for each chunk. Given a new prompt, the RAG system will embed it, return the top k nearest chunks in embedding space, and then prompt the LLM to answer the prompt given the context chunks [14].

Current RAG systems empirically work reasonably well, but are far from optimal and have several issues, of which we will discuss now. Firstly, while it is a good approximation, purely finding the similarity between the prompt and each
```

### --- Page 0002 ---

```markdown
![An overview of the graph construction and retrieval process for GEM-RAG. Given a corpus of text, GEM-RAG then generates utility questions for each chunk of text using an LLM, where a utility question asks something that could be answered given the text chunk as context. Next GEM-RAG uses these utility questions and respective embeddings to build a weighted graph. Summary nodes are then generated using the graph's spectral decomposition, using the eigenvectors to represent different orthogonal nodes or "eigenmodes" of the text. For retrieval, GEM-RAG embeds the question or prompt, and searches the graph for the optimal nodes or context to return.](assets/page_0002_img_1.png)

| Step | Description |
|------|-------------|
| Step 1 | Text Chunking & Utility Question Generation |
| Step 2 | Question Embedding & Similarity Computing |
| Step 3 | Building Weighted Complete Graph |
| Step 4 | Theme Summarization via Spectral Decomposition |
| Step 5 | Context Retrieval via Similarity Search |

To justify the claim that taxons are essential for growth, it is possible to ask: "What are the most relevant aspects of the text being analyzed?" In this context, the LLM can generate utility questions that help to investigate the matter. 

The chunk in the embedding space may not be the best method to retrieve the most relevant chunks to answer a question. The texts may be similar in style or phrasing, giving a superficially high similarity score, while the context or purpose of the text unrelated to the question. 

Secondly, often answering complex questions requires a synthesis of information across many chunks of text, and understanding how different chunks are interrelated and connected to each other conceptually is important for retrieval.

In aiming to solve these problems, we took inspiration from human cognition, specifically how humans encode and retrieve information based on the information’s relevance and how the information is tested on the material, emphasizing the importance of understanding the utility of the encoded information. 

Further, the more often memories are retrieved together, the more humans will synthesize this data into higher level summaries.

Motivated by human cognition, we propose Graphical Eigen Memories for Retrieval Augmented Generation (GEM-RAG). Given a corpus of text, our method splits the text into chunks and then generates several relevant "utility" questions using an LLM. The embeddings of these utility questions help build a complete weighted graph, where the weight between two nodes is the similarity of the utility questions. 

With this memory graph, and motivated by the observation that humans tend to synthesize information that is often retrieved together, we perform a random-walk analysis on the graph, by using the spectral decomposition of the normalized graph Laplacian. The intuition is that the eigenvectors of this decomposition provides an understanding of the key orthogonal themes present in the passage, by providing the different modes based on memory node similarity. 

We use the components of important eigenvectors to produce "eigenmodes" or summary nodes, which capture the higher-level structure of the text. This schematic structure constitutes the graphical eigen memory, or GEM. At retrieval time, we find the most similar utility question to the prompt/question, and then perform best first search on the GEM, to retrieve a sub-graph containing the context chunks ultimately passed to the LLM. A schematic of our method can be seen in Figure 1.

We believe developing a robust RAG system will enable LLMs to become rich AI-agents, adapting smartly to niche scientific domains, and unlock other important applications dependent on actively using memory. To best constrain and study the effectiveness of our model, we quantitatively evaluate the accuracy of our model on two QA datasets, QuALITY and Qaspar. We compare our method against a baseline of the standard RAG procedure, as well as against a recent state-of-the-art method, RAPTOR. For each method we explore using the SBERT and OpenAI's text-embedding-ada-002 embedding models, as well as the UniLmeQA and GPT-3 as LLMs. We show that in most cases our method gives better performance, and we provide ablation experiments.
```

### --- Page 0003 ---

```markdown
# PAGE_NAME: page_0003

## I. INTRODUCTION

To give an understanding of the effect of the number of eigen summary nodes and utility questions. Lastly, while the primary use-case of the synthesized graphical eigen memory (GEM) is to perform RAG for an LLM, we also explore how the GEM is a standalone object agnostic to the LLM. Ultimately, the GEM can be used to explore and visualize data, which we showcase in a web demo for an example graph.

Our contributions are as follows: 1) We introduce a novel RAG system inspired by human cognition that encodes, stores and retrieves information by its utility. 2) We further formulate generating summary nodes as a random walk problem, and use eigendecomposition to generate summary nodes. 3) We demonstrate the effectiveness of our RAG method on two QA datasets, using multiple different embedding and language models. We ran several ablation experiments to better understand the effectiveness of our method. 4) We release an interactive demo of an example GEM, to showcase how the graph works, and to emphasize its use as a standalone object.

## II. RELATED WORK

### Large Language Models (LLMs)

LLMs have proven to be extremely powerful general knowledge stores [24, 22], [25]. It has also been shown in some cases, that fine-tuning can produce data specific models [26]. The advent of better hardware and algorithms has allowed them to handle larger context lengths, which can handle more information to be encoded or retrieved. However, it has been shown that longer contexts have diminishing returns, can lead to a loss of information [27], [28], still necessitating retrieving relevant contextual information.

It has also been shown that while large context length models can perform very well on short story QA tasks [29]. However, it should be noted that these models retrieve the entire story as context, with thousands of tokens, where as we study only returning hundreds, as is the case in RAG methods. The motivation for this is discussed in the experimental section.

### Retrieval Augmentation Methods

The area of retrieval augmented generation has seen recent interest. In the work [7] initially introduced the idea of augmenting the context of an LLMs with the retrieved information. Further work shown by [8], [9], [10], [14] expanded on the idea, conceptualizing how LLMs could use retrieved context to trace where its response was coming from. In the work [30] proposed learning the retriever and LLM model jointly where as another work proposed using a tree-decoding algorithm for multi-answer retrieval [31]. Many works use different hierarchies of data summary [32], [33]. The work in [34] also showed how hierarchies can be built using recursive summarization. In the work [35] showed that RAG methods, while improving over baselines, often fail to provide enough key context to properly answer the question. It has also been shown how to use LLMs to generate summaries of chunks of text for improved accuracy. And the benefits of doing RAG with a custom encoder module [36], [37] Recently, Recursive Abstractive Processing for Tree-Organized Retrieval [20], studied the effects of building hierarchical trees of chunk summaries based on aggregating textual chunks by similarity. However, this approach fails to consider the biases of textual similarity and limits the node synthesis to a tree structure.

## III. METHODS

### A. Connection to Human Cognition

GEM draws its design and motivation from the intricacies of human cognition, namely, the processes by which humans encode, store and retrieve information. Specifically, we draw inspiration from how it is believed the human brain prioritizes information based on its utility, and that information that is most often retrieved together gets summarized together [15].

Our first observation from psychology is the so called “testing effect”, which observes that if humans are tested on subject material, they are more likely to accurately remember it [16]. This may be because, by testing, the information can be associated to a specific utility in our cognition, and thus is “tagged” with information that makes it easier to retrieve. To this end, we aim to better tag each chunk of information, by “tagging” it with LLM generated utility questions, which help express what information a specific text chunk has, and why it might be useful.

Our second observation is that in human cognition, the more often memories are retrieved together, the more likely the will continue to be retrieved together [17]. In the context of utility questions and their corresponding text embeddings, nodes, and the strength of their connections as the similarity of their question embeddings. We can then, inspired by this observation from human cognition, try to build higher level summary nodes based off the nodes of the graph, i.e. nodes that are likely to be retrieved together given a prompt. Our intuition is that by performing an eigendecomposition, each eigenvector will capture different mode across the line, that would often be retrieved together, and thus should be used to generate summary nodes.

### B. Graphical Eigen Memories

Our method at a high level involves several steps to first construct the graph, by chunking the text, generating utility questions, building the initial graph, and finally using the eigenvectors from spectral decomposition of normalized graph Laplacian to build the summary nodes. Then, with the GEM produced, we show how it can be used to perform RAG. The specifics of our implementation are discussed in the experiments section. A schematic of our method can be seen in Figure 1.

#### 1. Memory Graph Construction: Chunking

Text chunking is standard practice in RAG where the corpus of text of interest C is split apart into chunks, $(CHUNK_1, \ldots, CHUNK_n)$, where each chunk, $CHUNK_k$, has some number $T$ tokens. In practice the text can be split by number of characters or number of tokens. For a corpus of text we first chunk the text into $n$ chunks, where each chunk is $T$ tokens long, and $n = \frac{N}{T}$, where $N$ is the total number of tokens in the text.
```

### --- Page 0004 ---

```markdown
# Generating Utility Questions

Given each text CHUNK, we prompt an LLM to generate some number of utility questions. This can be represented by a function $Q$ that takes a chunk of text, and an integer, $n$, and generates $n$ utility questions. For every CHUNK, we compute $Q(CHUNK_i)$, which give a set of utility questions $Q_{i1}, Q_{i2}, \ldots, Q_{im_i}$.

## Text Embeddings

In order to quantify the similarity of the utility question to each other and/or to a prompt, we need to embed the text into an high-dimensional feature space, given by a text encoder. More specifically, given a text embedding function $E$, for each node $i$, we compute the embedding $v_i$ as $v_i = E(CHUNK_i)$. Similarly, we also compute this embedding for each utility question by the same function. However, in order to best encode the information of the utility question we take the embedding the average of the questions embedding, and the base text’s embedding, that is $v_{ij} = \frac{E(Q_{ij}) + E(CHUNK_i)}{2}$.

## Building the Weighted Complete Graph

Given the embedding of each chunk of text, as well as each chunk’s corresponding utility questions and respective embeddings, we then generate the fully connected graph. For each node/chunk pair $(i, j)$, we consider the sum of the similarity metric between all of node $i$’s utility question embeddings, to that of node $j$’s base text embedding. More formally, let $G = (N, E)$ be the memory graph we are constructing. Let the nodes be given by $N = \{CHUNK_1, CHUNK_2, \ldots, CHUNK_k\}$, where each node $i$ has utility questions $Q_i = \{q_{i1}, q_{i2}, \ldots, q_{im_i}\}$, and base text $x_i$. For each $i \in \{1, 2, \ldots, m\}$, generate an edge between $i$ and $v$ with weight $\sum_{n_i \in Q_i} SIM(v_{n_i}, v_j)$. Any function could be used to compute the similarity, but in this case we use standard cosine similarity, i.e., $SIM(v_i, v_j) = \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|}$.

## Building the Summary Nodes

In order to build an embedding system that encodes this higher level information we formulate this as a random walk or spectral decomposition problem. Intuitively, in this context, each eigenvalue and its corresponding eigenvector reveal a distinct ‘theme’ or conceptual dimension in the graph. By summarizing the top components and each eigenvector components, using an LLM, we can understand the most significant relationships and conceptual clusters within the graph.

More specifically, let $S = (s_{ij})_{n \times n}$ be the similarity matrix for the graph, where $s_{ij}$ is the sum weight between the $m$ SIM values of each utility questions of nodes $i$ and to node $j$. By attaching each $s_{ij}$ to a weighted edge $e_{ij}$, we can map the similarity matrix $S$ onto the memory graph $G$. Thus, we can better understand the relationship between different text pieces of the document by analyzing the properties and behaviors of $G$.

Since different documents possess different connectivity and node centrality, the spectra will also be different scales. To better quantify how influential each node is without degree bias, we transform $S$ to a variant of normalized graph Laplacian $L$, which is $L = D^{-1/2}(S - D)D^{-1/2}$, where $D = diag(d_i)$ is the degree matrix and $I$ is the identity matrix. Then we conduct spectral decomposition by solving

$$
\lambda \mathbf{L} = \mathbf{L} \mathbf{v}
$$

with the resulting eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$ ordered in non-increasing order of their magnitude. The corresponding eigenvectors $\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_r$ represent the principal themes. Then, for each eigenvector $\mathbf{z}_k$, select the top components, $x_1, x_2, \ldots, x_k$, representing the most relevant nodes for the $k$-th theme. Then we prompt an LLM, given the text passages associated with $x_1, x_2, \ldots, x_k$, to summarize the text, summarizing any high-level information. With the produced summary text, we introduce it as a new context to every other node in the graph in the manner previously described.

## Analysis of Graph Spectrum

From spectral graph theory [38, 39], the eigenvalues of $L$ exhibit these properties: (1) $\sum_i \lambda_i = 0$, (2) each eigenvalue $\lambda_i$ falls within the range $[-1, 1]$, and (3) the largest eigenvalue $\lambda_1$ in the similarity matrix $S^{-1}$, where $S$ is zero diagonal, has $(n^2 - n)$ non-zero elements, leading to an $O(n^2)$ complexity for the normalized Laplacian transformation. Subsequent spectral decomposition via the Lanczos algorithm [40] incurs an $O(n^3)$ computational complexity.

Given that all text chunks nodes are clustered into different themes via spectral decomposition, we can observe some interesting properties on $S$ from such clustering behavior.

**Remark 1:** Suppose the document includes $k$ essential themes. The modularity of $S$ can be formulated as follows:

$$
S = \begin{bmatrix}
\hat{S}_{11} & \hat{S}_{1k} \\
\hat{S}_{k1} & \hat{S}_{kk}
\end{bmatrix}
$$

where $\hat{S}_{ij}$ has $r$ rows. Thus, each diagonal block $\hat{S}_{ii}$ satisfies $0 < n_i - 1 \| \hat{S}_{ii} \|_F > 0$ for each off-diagonal block. Then we can characterize $S$’s spectrum via the behavior of its eigenvalues, i.e., $0 < \lambda_k < 1$, $\forall i \in \{1, 2, \ldots, k\}$ and $|\lambda| < 1$, $\forall i \in \{k + 1, k + 2, \ldots, n\}$.
```


### --- Page 0005 ---

```markdown
| Embedding        | LLM          | RAG      | Acc     | HARD Acc |
|------------------|--------------|----------|---------|----------|
| SBERT            | UnifiedQA    | GEM-RAG  | 52.14%  | 44.70%   |
| SBERT            | UnifiedQA    | RAPTOR   | 51.51%  | 43.30%   |
| SBERT            | UnifiedQA    | Embed    | 51.04%  | 44.06%   |
| SBERT            | GPT3.5       | GEM-RAG  | 61.84%  | 51.60%   |
| SBERT            | GPT3.5       | RAPTOR   | 60.13%  | 50.32%   |
| SBERT            | GPT3.5       | Embed    | 58.61%  | 47.25%   |
| OpenAI           | UnifiedQA    | GEM-RAG  | 52.81%  | 44.83%   |
| OpenAI           | UnifiedQA    | RAPTOR   | 53.48%  | 46.96%   |
| OpenAI           | UnifiedQA    | Embed    | 52.14%  | 42.53%   |
| OpenAI           | GPT3.5       | GEM-RAG  | 63.37%  | 51.81%   |
| OpenAI           | GPT3.5       | RAPTOR   | 60.32%  | 50.96%   |
| OpenAI           | GPT3.5       | Embed    | 60.32%  | 49.55%   |
| OpenAI           | GPT3.5       | GEM-RAG (k-Means) | 61.42%  | 50.83%   |

**TABLE**  
RESULTS ON THE QUALITY DEV DATASET FOR ALL EMBEDDING, LLM AND RAG PAIRS.

These properties confer on $S$ effective clustering capabilities, where text chunks within the same theme exhibit higher similarity and lower similarity across different themes.

**Remark 2:** Consider a sequence where each term is the ratio $\beta_k / \beta_i \geq 2$ to the largest eigenvalue $\lambda_k$. If there exists some index $i \geq 2$ such that $\beta_i \leq 1$, and $\beta_i - \beta_{i+1} > c$ where $c$ indicates the cutoff that identifies the first significant gap between a pair of adjacent ratios, then $d$ is the estimated number of essential themes in the documents.

**Remark 3:** Let $\Lambda = \sum_{i} \alpha_i$. Here, $\alpha_i$ indicates more differentiation between themes becomes ambiguous.

**2) Retrieval:** Given a built GEM, our method responds to readily answer prompts/question about the given dataset. The process works as follows: given a prompt/question $p$, and some budget $B$ of nodes to return, we first produce an embedding of the prompt the query $v_p = E(p)$. We then find, out of the entire graph, the utility question that has the highest $S$. Specifically, let $Q = \{q_1, q_2, \ldots, q_n\}$ be the set of all utility questions, find $q^* = \arg\max_{q \in Q} \text{sim}(v_p, E(q))$. Then, from the associated node $q^*$ we perform a best first search, to find the next node to visit $B$ to include in the context. Full details can be seen in Algorithm 1.

The LLM is then given this context, followed by the question and prompted to answer.

**C. Method Trade-Offs**

While the robustness of our method leads to our improved results, it does come with some trade-offs that are important to discuss, particularly in terms of computational complexity, and potential costs.

First, generating utility questions via an LLM can become a significant cost, either computationally or monetarily, depending on the number of nodes in the dataset, and the number of utility questions. Generating a graph with $n$ nodes and $d$ utility questions requires my LLM calls. The graph building is all pre-computed, so in most cases this extra cost is okay, but it is worth noting.

Secondly, in order to generate higher level summary nodes we do eigendecomposition which has a complexity of $O(n^3)$ where $n$ is the number of nodes. With a large number of nodes this complexity may require consideration.

IV. EXPERIMENTS

Our method, and RAG methods in general can be used for a host of tasks, including AI agent memory, fine-tuning LLMs for niche scientific domains etc. For purposes of comparison we evaluate the efficiency of our model in the context of QA for model fine-tuning.

**A. Setting**

It should be noted that RAG methods in general aim to study the specific problem of retrieving minimal context windows from large data sets and returning them to an LLM for processing. However, this setting is of ongoing debate and change within the research community, as large LLMs have increasingly large context lengths, making it possible to fit massive amounts of data within a single prompt. We feel, that even with larger and larger context windows, the problem of retrieval is still of interest because in a real-world setting, the data sets of interest are often still much larger than the current context windows of the biggest models, and returning massive amounts of context tokens is impractical and costly. Secondly, even with a larger context, it has been shown that LLMs can be prone to forget information in the context, and can be prone to hallucinate information. Lastly, with more precise retrieval it makes it easier to verify and track the information that an LLM is using to answer a question. For these reasons, we feel that the retrieval methods are still highly motivated despite the evergrowing context length of large models. Retrieval also requires no fine-tuning, and uses LLMs "off-the-shelf".

To properly compare our method to other recent work we study datasets where often the entire data set can fit in the context of an LLM. For this reason, we outline our method that outperforms other retrieval methods, which are all limited to 400 tokens of context, there has been work that uses thousands of tokens of context and outperforms retrieval metrics such as
```

### --- Page 0006 ---

```markdown
| Embedding | LLM      | RAG    | F1     |
|-----------|----------|--------|--------|
| SBERT     | GPT-3.5 | GEM-RAG| 18.53% |
| SBERT     | GPT-3.5 | RAPTOR | 18.51% |
| SBERT     | GPT-3.5 | Embed  | 19.24% |
| OpenAI    | GPT-3.5 | GEM-RAG| 20.13% |
| OpenAI    | GPT-3.5 | RAPTOR | 18.13% |
| OpenAI    | GPT-3.5 | Embed  | 19.07% |

![Results on the Qasper dataset for all embedding, LLM and RAG pairs](assets/page_0006_img_1.png)

### B. Data

#### a) QuALITY: The Question Answering with Long Input Texts, Yes! (QuALITY) dataset [18] contains 230 medium length passages (about 500 tokens), for which each passage has associated multiple choice questions and ground truth answers. For additionally, each set of questions has a subset of HARD questions which are particularly challenging. Specifically we use the ‘dev’ data split. For the ablations experiments we use the first 50 passages from this set, whereas for the main experiments in Table I we use the remaining 180.

#### b) Paper: The Qasper data set [19] contains over 1500 academic NLP papers. Each paper has associated multiple choice questions and ground truth answers. We use the first 100 papers in the data set.

#### C. Baselines

We compare our method to the standard RAG method, of embedding each chunk of text, and embedding the given prompt, and finding the most similar chunks to the prompt, up to a specific budget. Additionally, we compare to recent work that showed promising results, Recursive Abstractive Processing for Tree-Organized Retrieval (RAPTOR) method [20]. RAPTOR primarily aims to tackle the problem of producing hierarchies of nodes, that summarizes the text passage appropriately.

#### D. Evaluation

The QuALITY dataset has multiple choice questions with ground truth answers, for this dataset we calculate the accuracy and report the overall accuracy, as well as the accuracy on the HARD subset. The Qasper dataset has answer types that are either Abstractive, Extractive, Yes/NO or Answerable/Unanswerable. We use the F1 score to evaluate efficacy of the methods for this dataset.

### E. Experimental Parameters

Our objective with these experiments is to evaluate the efficacy of our model across different text embedding models and LLMs. We use the SBERT [21], and the OpenAI text-embedding-ada-002 text embedding models. We also use the UnifiedQA [22] and GPT-3.5 Turbo LLMs. For the QuALITY data set we consider all possible combinations of text encoders, LLMs and RAG methods. For the Qasper dataset we consider all embedding and RAG models but only using GPT-3.5 Turbo as the LLM.

For all experiments we use a chunk size of 100 tokens, and we allow 400 tokens of context, meaning four nodes of context. Even though GPT-3.5 can support a much larger context, we aim to study the setting where only few nodes/chunks can be used, to better isolate the effectiveness of the RAG method in question, rather than the attention mechanism of the LLM.

For the summarization method for GEM-RAG and RAPTOR, as well at the utility question method for GEM-RAG we use GPT-3.5 Turbo as the LLM. All similarity measurements were done using cosine similarity. For the GEM-RAG method we use two view eigendecompositions, and five utility questions. In our ablation experiments we then explore the effect of varying both of these parameters. For the ablation experiments we evaluate the accuracy on the first 25 articles of the QuALITY dataset, whereas for the main experiments we evaluate on the latter 180 articles.

### F. Results

QuALITY Results The results from our experimentation can be seen in Table I. We observe that in all settings, except for those using the OpenAI embedding model, and UnifiedQA LLM, our model gives the best performance, in both overall accuracy and accuracy on the HARD subset. Also notably, the difference is exaggerated on the HARD subset. Also, using OpenAI’s embedding, and GPT-3.5 Turbo, suggesting this difference is exaggerated the more robust the embedding and LLM models used are.

Additionally, we looked at the performance of our method if we use k-means instead of spectral clustering. We observe a performance drop, which indicates the spectral graph analysis performs better than standard clustering.

Qasper Results The results from our experimentation can be seen in Table II. In this setting we only use GPT-3.5 Turbo as the LLM and consider using SBERT and the OpenAI text embedding models. We see that the OpenAI text-embedding-ada-002 embedding model, and gives the best overall score. However, it performs behind the standard RAG method when considering SBERT.

#### GEM For Data Visualization

We would like to stress that while GEM has been formulated to be primarily used as a method for RAG, the produced GEM is a standalone object that can be used with any LLM to do QA work, as well as a tool to visualize and organize data. We provide an example visualization for a single story in the QuALITY dataset at the following url: https://detailed-swan.sota.domains/GEM.html.

#### Ablation Study

In order to test the tuneable hyperparameters of our model, the number of eigencomponents and the number of utility questions, we performed three ablation experiments, the results of which can be seen in Figure 2. In the first we keep the number of components constant at ten, and vary the number of utility questions. We observe that the accuracy with respect to the number of utility questions
```

### --- Page 0007 ---

```markdown
![Ablation study on the effect of number of utility questions and eigencomponents](assets/page_0007_img_1.png)

| Eigen-components | Percent of nodes returned that are eigen/summary |
|------------------|--------------------------------------------------|
| 0                | 0.0                                              |
| 2                | 14.4                                             |
| 4                | 24.5                                             |
| 6                | 32.8                                             |
| 8                | 38.1                                             |
| 10               | 42.6                                             |

V. DISCUSSION AND CONCLUSION

We developed GEM-RAG, a method for RAG inspired by human cognition, that tags each memory or chunk by the specific utility of its information and relation to other memories. Further, we use these utility questions to formulate a weighted fully connected graph. We perform an eigendecomposition on this memory graph to robustly extract “eigenthemes”, and create summary nodes for each theme. We observe in most cases, for multiple text embeddings and LLMs, our method out-performs standard baselines. We also show that a produced GEM is a standalone object: it can be used with any LLM to be searchable and conversable, and provides a principled visualization for understanding text data. We believe the optimal RAG method has the ability to greatly improve the ability of LLMs, enabling real AI agents that can leverage massive histories of conversations, or adapt to massive niche data sets without fine-tuning. Further, these RAG methods can extend to LLMs, retrieving articles, videos, sound, etc., and bringing us closer to simulating human cognition.

VI. ACKNOWLEDGEMENTS

This project is partially supported by the National Science Foundation (NSF); the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, a program of Schmidt Sciences, LLC; the National Institute of Food and Agriculture (US-DA/NIFA); the Air Force Office of Scientific Research (AFOSR), and Toyota Research Institute (TRI).

REFERENCES

[1] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamvar, P. Lee, Y. T. Lee, Y. Liu, S. Lundberg, H. Nori, H. Palangi, T. R. Ribeiro, and Y. Zhang, “Sparks of artificial general intelligence: Early experiments with gpt-4,” 2023.

[2] F. Petroni, T. Rocktäschel, P. Lewis, A. Bakhtin, Y. Wu, A. H. Miller, and S. Riedel, “Language models as knowledge bases” 2019. [Online]. Available: https://arxiv.org/abs/1909.01066

[3] Z. Jiang, F. F. K. Ju, J. Araki, and G. Neubig, “How can we know what language models know?” CoRR, vol. abs/1911.12543, 2019. [Online]. Available: http://arxiv.org/abs/1911.12543
```

### --- Page 0008 ---

```markdown
| **References**                                                                                                           |
|--------------------------------------------------------------------------------------------------------------------------|
| [4] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. K. mar, P. Lee, Y. T. Lee, Y. S. Lundberg, H. Nori, H. Plank, M. T. Riemer, P. Y. Yang, "Sparks of artificial general intelligence: Early experiments with GPT-4," 2023. |
| [5] A. Ravichandran, M. P. Chen, and S. Ganguli, "Pretraining task selection: Rethinking the role of non-bias in context learning for regression," 2023. |
| [6] N. Narasimhan, D. Yu, J. Zhao, I. Shafran, T. R. Griffiths, Y. Cao, and K. Narasimhan, "The other half: Deliberate problem solving with large language models," 2023. |
| [7] P. S. H. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpuškin, N. Goyal, K. Kütler, M. Lewis, W. Yi, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," CoRR, vol. abs/2005.11401, 2020. [Online]. Available: https://arxiv.org/abs/2005.11401 |
| [8] "Retrieval-augmented generation for knowledge-intensive NLP tasks," CoRR, vol. abs/2005.11401, 2020. [Online]. Available: https://arxiv.org/abs/2005.11401 |
| [9] A. W. Yu, D. Ohan, M. Luong, R. Zhao, K. Chen, M. Norouzi, and O. Y. Le, "Agent: Combining knowledge with goal self-attention for reading comprehension," CoRR, vol. abs/1804.09541, 2018. [Online]. Available: https://arxiv.org/abs/1804.09541 |
| [10] R. Sun, R. Levine, D. Mildenglos, D. Waly, A. Shashua, T. Levon-Brown, and Y. Shobham, "In-context retrieval-augmented language modeling," 2023. |
| [11] J. Gizard, Z. Lemoine, M. Loneli, I. Hossn, F. Petroni, T. Schick, A. D. Yudkevich, A. L. Jones, S. Riedel, and E. Griffiths, "Atlas: Few-shot learning with retrieval augmented language models," 2022. |
| [12] Y. Sun, D. O. Menzia, B. Zhao, P. Neumann, K. Kunze, and M. Guo, "Rag-udr: Generative driving explaining language with retrieval-augmented generation," 2023. |
| [13] A. K. K. Liu, H. F. Liu, B. Xiong, I. Leng, J. Andreas, and K. H. Lee, "My hybrid architecture for retrieval-augmented question answering," in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 1–10. [Online]. Available: https://aclanthology.org/2022.emnlp-main.1 |
| [14] G. Arivazhagan, L. Liu, P. Qi, X. Chen, W. Wang, and M. Huang, "My hybrid architecture for retrieval-augmented question answering," in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, ACL, 2021. |
| [15] L. Henry, E. Reeder and J. D. Karpicke, "Test-enhanced learning: Taking memory retrieval to new heights," Psychological Science, vol. 17, no. 3, pp. 249–255, 2006. [Online]. Available: https://doi.org/10.1111/j.1467-9280.2006.01670.x |
| [16] M. Goldstein, E. G. Edwards, "Editors' introduction: Abstract concepts: Structure, processing, and modeling," Topics in Cognitive Science, vol. 10, no. 3, pp. 490–506, 2018. |
| [17] Y. P. Yang, A. P. Morris, N. Joshi, N. Nanga, J. P. Zhang, A. Chen, M. Palukuri, A. M. J. Thompson, H. He, and S. R. Bowman, "Quality: Question answering when input text is very short," 2021. |
| [18] P. Dasigi, K. Liu, I. Beltagy, A. Cohan, N. S. Smith, and M. Gardner, "A dataset of information-seeking questions and answers anchored in research papers," in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, K. Toutanova, A. R. Rummels, L. Zettlemoyer, D. Hakkani-Tür, I. Beltagy, S. Bethard, R. Cottrell, T. Chakraborty, and Y. Zhou, Eds. Online: Association for Computational Linguistics, Jun. 2021, pp. 499–506. [Online]. Available: https://aclanthology.org/2021.naacl-main.56 |
| [19] P. Sarith, S. Abdulhalim, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning, "Raptor: Recursive abstractive processing for tree-organized retrieval," 2021. |
| [20] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," 2019. [Online]. Available: https://arxiv.org/abs/1908.10084 |
| [21] D. Khashabi, T. Khot, A. Asharaf, O. Trafford, P. Clark, and H. Hajishirzi, "Unified: Closing from sound boundaries with a single QA system," CoRR, vol. abs/2005.00780, 2020. [Online]. Available: https://arxiv.org/abs/2005.00780 |
| [22] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, D. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, H. Herbert-Voss, G. Krueger, T. M. K. M. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigal, M. Lewis, S. Gray, B. Chen, J. Clark, C. Berner, S. McCan, A. Radford, I. Sutskever, and D. Amodei, "Language models are few-shot learners," CoRR, vol. abs/2005.14165, 2020. [Online]. Available: https://arxiv.org/abs/2005.14165 |
| [23] "Language models are few-shot learners," CoRR, vol. abs/2005.14165, 2020. [Online]. Available: https://arxiv.org/abs/2005.14165 |
| [24] OpenAI, "GPT-4 technical report," 2023. |
| [25] A. Roberts, C. Raffel, and S. Narasimhan, "How much knowledge can you pack into the parameters of a language model?" in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), B. Webber, T. Cohen, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 5418–5426. [Online]. Available: https://aclanthology.org/2020.emnlp-main.371 |
| [26] P. F. Liu, K. Lin, I. Hewitt, A. Parikh, M. DeMello, F. Petroni, and P. Liang, "Lost in the middle: How language models use context," 2023. |
| [27] S. Sun, K. Krishna, A. Mattarella-Micke, and M. Iyer, "Do large language models actually use long-range context?" in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, M. Post, M. Xu, H. S. P. S. and W. Y. T. Y. H. Eds. Online: Association for Computational Linguistics, Nov. 2021, pp. 1–10. [Online]. Available: https://aclanthology.org/2021.emnlp-main.627 |
| [28] A. Shashua, M. H. K. I. Brant, and L. Z. "Zero-shot: A zero-shot benchmark for long-range context," 2023. [Online]. Available: https://arxiv.org/abs/2305.14156 |
| [29] J. Gizard, P. Lewis, M. Loneli, I. Hossn, F. Petroni, T. Schick, A. D. Yudkevich, A. L. Jones, S. Riedel, and E. Griffiths, "Atlas: Few-shot learning with retrieval augmented language models," 2022. |
| [30] S. Min, K. Lee, W. R. T. Y. and H. J. "Retrieving for more: A new retrieval-augmented generation framework," in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 697–708. [Online]. Available: https://aclanthology.org/2022.emnlp-main.1 |
| [31] Y. Liu, K. Haimovich, Y. Zhou, S. Zhuang, X. Ciong, and P. Yu, "Dense hierarchical retrieval for open-domain question answering," in Findings of the Association for Computational Linguistics, EMNLP 2021, F. Moens, X. Huang, L. Specia, and S. W. Y. T. Eds. Online: Association for Computational Linguistics, Nov. 2021, pp. 1–10. [Online]. Available: https://aclanthology.org/2021.findings-emnlp.1 |
| [32] P. Christiano, "Recursively summarizing books with human feedback," 2021. |
| [33] B. Nensim, R. Soldanki, R. Fok, A. Cohans, and K. Lo, "A question answering framework for decontextualizing user-facing snippets from scientific documents," 2023. |
| [34] T. Gao, H. Yen, Y. Yu, and D. Chen, "Enabling large language models to generate text with citations," 2023. |
| [35] Y. Karpuškin, B. Ogan, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Y. "Dense passage retrieval for open-domain question answering," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), B. Webber, T. Cohen, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 6769–6781. [Online]. Available: https://aclanthology.org/2020.emnlp-main.1 |
| [36] R. F. G. "Spectral graph theory," American Mathematical Soc., 1997, vol. 92. |
| [37] W. Li, W. K. Y. Liu, and K.-L. "Enhancing the effectiveness of clustering with spectral analysis," IEEE Transactions on Knowledge and Data Engineering, vol. 19, no. 7, pp. 897–902, 2007. |
| [38] D. Lanzcos, "An iteration method for the solution of the eigenvalue problem for hermitian and differential operators," J. Res. Nat. Bur. Stand., vol. 45, pp. 255–282, 1950. |
```


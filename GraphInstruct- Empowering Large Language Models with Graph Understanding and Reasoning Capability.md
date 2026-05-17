# GraphInstruct- Empowering Large Language Models with Graph Understanding and Reasoning Capability

### --- Page 0001 ---

```markdown
![GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability](assets/page_0001_img_1.png)

# GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability

Zihan Luo¹, Xiran Song¹, Hong Huang¹, Jianxun Lian³, Chenhao Zhang², Jinqi Jiang², Xing Xie³, Hai Jin¹

1. National Engineering Research Center for Big Data Technology and System, Services Computing Technology and System Lab, Cluster and Grid Computing Lab, School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan 430074, China  
2. School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan 430074, China  
3. Microsoft Research Asia, Beijing 100190, China  

Received month dd, yyyy; accepted month dd, yyyy  
E-mail: honghuang@hust.edu.cn.  
© Higher Education Press 2025  

## Abstract

Improving the general capabilities of Large Language Models (LLMs) is an active research topic. As a common data structure in many real-world domains, understanding graph data is a crucial part of advancing general intelligence. To this end, we propose a dynamic benchmark named GraphInstruct in this paper, which comprehensively includes 21 classical graph reasoning tasks, providing diverse graph generation pipelines and detailed intermediate reasoning steps for each sample. Based on GraphInstruct, we develop GraphSolver via efficient instruction-tuning, which demonstrates prominent graph understanding capability compared to other open-sourced LLMs. To further endow LLMs with multi-step graph reasoning capability, we propose a label-masking training strategy and build GraphSolver, which leverages masked inversion on intermediate reasoning tokens to emphasize crucial node-identification signals. As one of the pioneering efforts to enhance the graph understanding and reasoning abilities of LLMs, extensive experiments have shown that our LLMs sincerely outperform existing LLMs on various tasks. Our code and data are released publicly at: https://github.com/GGCL-codes/GraphInstruct.

## Keywords

LLM; Graph reasoning; Instruction tuning

## 1 Introduction

Recently, the rise of Large Language Models (LLMs) has showcased their powerful generalization abilities in various domains such as mathematics [1, 2], code [3, 4], and dialogue [5, 6], and researchers are aspiring to leverage LLMs to achieve the capabilities of artificial general intelligence. With this goal in mind, many researchers have begun efforts to assist LLMs in understanding diverse forms of input data, including images [7, 8], audios [9], and others. However, this body of work has overlooked the capability of LLMs to comprehend data in the form of graphs. In fact, as a common data structure in the real world, graphs are prevalent in many scenarios like social networks [10], fraud detection [11], and biological models [12, 13]. What’s more, many logical reasoning tasks also fundamentally exist in the form of graphs [14], and facilitating LLMs in understanding graph data and graph reasoning may be a crucial step toward advancing their general capabilities in mathematics and code intelligence [15, 16].

In fact, numerous researchers have already started investigating the understanding capabilities of LLMs on graph data, such as GPT4Graph [17], NLGraph [18], GraphArena [19], and others. Fatemi et al. [20] also examine the impact of different graph description languages on the understanding of graph data by LLMs. These exploratory works consistently find that, in most cases, LLMs perform significantly below expectations. As shown in Figure 1, we conduct case studies on a classical graph reasoning task, implying that existing LLMs struggle to comprehend and address reasoning tasks based on graph data. In response to these challenges, a number of work begins to focus on enhancing the capabilities of LLMs on graph data. GraphLLM [21] and GraphGPT [22] contemplate ways to enhance the performance of LLMs in graph-related tasks by concatenating encoded information from Graph Neural Networks (GNNs) into LLMs. GraphWiz [23] utilizes supervised fine-tuning (SFT) and direct preference optimization (DPO) to align LLMs with graph data. These works aid in understanding graph structure information but has only been tested on a limited set of tasks, with insufficient generalization capabilities. In summary, addressing the challenge of enabling LLMs to truly understand graph data remains an unresolved issue.

In this paper, to facilitate LLMs in comprehending graph-structured data and thereby equipping them with the ability to address graph-related tasks, we have curated and introduced a comprehensive dynamic benchmark – GraphInstruct designed for assessing and fine-tuning LLMs.
```

### --- Page 0002 ---

```markdown
# Zihan Lu et al.  GraphInstuct: Empowering Large Language Models with Graph Understanding and Reasoning Capability

## Question:
Given a directed graph:  
Node 4 is connected to node 3.  
Node 2 is connected to node 7.  
Node 3 is connected to node 1.  
Node 1 is connected to nodes 0, 5.  
Node 5 is connected to nodes 4, 7.  
Node 6 is connected to node 3.  
Which are the predecessor nodes of node 3? A predecessor of $n$ is a node such that there exists a directed edge from $n$ to $n$.

Let us answer this question step by step to make it correct.

## Answer:
GPT-3.5 Turbo

Now, let’s find the predecessors of node 3:  
Node 4 has a directed edge to node 3.  
Node 2 has a directed edge to node 3.  
Node 6 has a directed edge to node 3.  
Therefore, the predecessor nodes of node 3 are nodes 4, 2, and 6.

Ours  
$[4, 2, 6]$

![Case study on graph reasoning tasks that general LLMs like GPT-3.5 Turbo fail to handle](assets/page_0002_img_1.png)

## 2 Related Work
### 2.1 LLMs on Graph Reasoning
With the rise of LLMs, researchers are increasingly exploring the capability of LLMs on solving graph reasoning tasks like DFS and the shortest path. Existing research can be broadly categorized into three domains: 
1. Empirical performance evaluation. Early researchers conducted empirical performance evaluations, such as GraphGR[1], NL-Graph[18], GraphArena [19], GraphOM [20], and GraphOM [26]. These early works pave the way for applying LLMs to the field of graph and consistently reveal the potential of LLMs in understanding graph data.
2. Fine-tuning graph-specific reasoning. Many researchers have started fine-tuning LLMs on graph-related tasks to enhance their reasoning capability on graph-related tasks, as seen in works like GraphLML [21], GraphWiz [23], and GraphSilo [15] etc. However, their scope has remained relatively limited to several specific tasks, and the challenge of enabling LLMs to possess a universal understanding of graph data remains an urgent problem to be addressed.
3. Tool augmentation and code generation. Different from the existing work, the third line of researchers do not ask LLMs to solve graph reasoning tasks in an explicit step-by-step reasoning manner, but utilizes tool augmentation (e.g., code generation) to equip LLMs with explicit graph algorithmic skills [27, 28]. Our work, GraphInstuct, belongs to the second category. Our aim is to help LLMs genuinely master graph reasoning by advancing fine-tuning methodologies and promoting universal understanding of graph data.

### 2.2 LLM Instruction Tuning
Although LLMs are widely applied in various fields such as conversational [26], mathematics [29], and code programming [30] due to their outstanding generalizations abilities, researchers have found that fine-tuning with specific instructions can further enhance LLMs’ performance in more targeted domains. For instance, Bao et al. [31] utilize instruction tuning to apply LLMs to recommendation scenarios. Li et al. [32] create a dedicated dataset, EcomInstuct, for the e-commerce domain and fine-tune a specialized LLM, EcomGPT. Moreover, researchers have explored the application of LLMs in various areas such
```


### --- Page 0003 ---

```markdown
# Front. Comput. Sci., 2023, 0(0): 1

## Graph Structure Distribution

### (a) Graph Generation

| Random         | Scale-free     | Small-world     |
|----------------|----------------|------------------|
| Mini graphs    | Small graphs   | Medium graphs    | Large graphs    |

### Edge List
```
[(3, 4), 
 (3, 1), 
 (3, 2), 
 (4, 2)]
```

### Adjacency Table
```
{3: [4, 1, 0], 
 4: [3, 2], 
 1: [3], 
 0: [4]}
```

### Adjacency Table in natural language
Node 3 is connected to nodes 4, 1, 0.  
Node 4 is connected to ...

### (b) Graph Description

### Integer Node ID
Node 3 is connected to nodes 4, 1, 0.  
Node 4 is connected to ...

### Letter Node ID
Node XTR is connected to nodes SEQ, PWE, NCS.  
Node NCS is connected to ...

## (c) Question & Answering

### Question
Given an undirected graph:  
...  
Does the graph have a cycle?

### Intermediate steps
Let's solve it step by step.  
We can use the topological sorting algorithm to detect a cycle in the graph.  
...  
The result of topological sorting: `[3, 7, 1, 4, 5, 6, 2, 0]` contains all the nodes in the graph, so the answer is

### Answer
No

---

### Fig. 2
The overview of GraphInstruct benchmark. We provide diverse options during the construction of GraphInstruct, including graph structure distributions, graph sizes, graph description languages, and node IDs. For improving the reasoning capability of LLMs, GraphInstruct also provides precise intermediate results for each task.

### Fig. 3
The complete task schema of GraphInstruct

---

## 3 The GraphInstruct Benchmark
In this section, we will introduce our GraphInstruct benchmark, which is designed to evaluate and enhance the graph understanding and graph reasoning of LLMs. In general, GraphInstruct covers various graph structure distributions, graph sizes, and graph descriptions. The overview of GraphInstruct is illustrated in Figure 2, where samples are dynamically generated with various options. As shown in Figure 3, we comprehensively consider 21 classic graph reasoning tasks in GraphInstruct, encompassing tasks at Node level, Node-pair level, and Graph level, respectively, and cover most classical graph reasoning tasks such as Depth-First Search (DFS) and Shortest Path. The following subsections will further introduce the details of GraphInstruct.

### 3.1 Graph Generation
As the foundation in GraphInstruct, the graphs generated in GraphInstruct should be as diverse as possible, so that the performance on these graphs could reflect the intrinsic capability of LLMs. To this end, we provide various kinds of options during the graph generation process, as shown in Figure 2 (a), and can be summarized as follows:

- **Graph Structure Distribution**  
To provide a diverse range of graph structures, we mainly consider three representative graph structure distributions, including (1) Erdős–Rényi random networks; (2) Barabási–Albert scale-free networks; and (3) small-world networks.

- **Erdős–Rényi random networks:** Given $N$ nodes, they are connected pairwise with a probability of $p$ to form a graph.

- **Barabási–Albert scale-free networks:** Networks in which the node degree distribution follows a power-law distribution. Various networks such as social networks, biological networks, and trade networks exhibit scale-free network characteristics.

- **Small-world networks:** In small-world networks, most nodes are not directly connected to each other, but the majority of nodes can be reached from any other node through a few intermediate steps.
```

### --- Page 0004 ---

```markdown
# Zihan Luo et al.  GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability

In addition, both directed and undirected graphs are considered and randomly generated in GraphInstruct during graph generation, providing diverse graph structure distributions.

## Graph Size
Following the line of previous work [18,21], GraphInstruct consists of four levels of graph size, namely mini graphs, small graphs, medium graphs and large graphs, consisting of 5-7 nodes, 8-15 nodes, 16-25 nodes and 26-35 nodes, respectively. Intuitively, tasks with larger graph sizes are always incorporated with higher difficulty levels because of the increase in context length and required reasoning steps. The impact of graph size will be further analyzed in our experiments.

### 3.2 Graph Description
GraphInstruct is a text-based dataset that transforms the generated graphs into diverse Graph Description Languages (GDL). Given the prior investigation provided by [20], GraphInstruct provides three effective graph description languages as shown in Figure 2 (b): edge list, adjacency table, and adjacency table in natural language (the default setting during training). Besides different graph description languages, we also provide two optional node ID representations, including an integer ID like `1` and random letters like `XRT`.

### 3.3 Intermediate Reasoning Generation
As a CoT manner, we generate detailed intermediate reasoning steps for each training sample through graph prompts. As illustrated in Figure 2 (c), the experimentation process includes accurate intermediate steps, which can be used as either supervision signals or CoT prompts. With the help of intermediate steps, we can empower LLMs with stronger graph reasoning capability, which will be discussed in Section 5.

### 3.4 Task Definition
For each sample, GraphInstruct D provides four elements, including the graph G, query Q, answer A and intermediate reasoning steps S. For each randomly generated graph $G = (V, E)$, V is the set of nodes and E is the set of edges. In the following, we briefly introduce the definitions for the 21 graph reasoning tasks involved in GraphInstruct:

- **Neighbor** Given a node $u \in V$, output its neighbors $N(u)$.
- **Degree** Given a node $u \in V$, output the degree of it: $|N(u)|$.
- **Predecessor** Given a directed graph $G$ and a node $u \in V$, output the set of predecessor nodes of it: $P(u) = \{v | (v,u) \in E\}$.
- **PageRank** Find the node with the largest PageRank value [38] after several rounds of iteration. The damping value is 0.85, and the iteration number is 3. The initialized PageRank value for each node is $1/|V|$.
- **Clustering Coefficient** Calculate the clustering coefficient value of node $u \in V$. For a node $u$, let $T$ be the degree number among its neighbors and D its degree (out-degree in directed graphs). The clustering coefficient of it is $C = \frac{2E}{T(T-1)}$ for directed graphs and $C = \frac{E}{N(N-1)}$ for undirected graphs.
- **Common Neighbor** Given a node pair $(u, v)$, output the number of common neighbors: $C = |N(u) \cap N(v)|$.

### 3.5 Evaluation
We parse the LLM’s output and use exact-match accuracy for evaluation. There are mainly the following kinds of output type: boolean (e.g. Connectivity), integer (e.g. Degree), float (e.g. Jaccard), single node (e.g. PageRank), node list (e.g. Depth-First Search), node set (e.g. Neighbor), and edge list (e.g. Bipartite). For each output type, we require a relative error of 3% or less. For tasks where multiple solutions exist (e.g. Depth-First Search), we examine if the output is a valid answer.

### 3.6 GraphInstruct Statistics
As summarized in Table 1, we compare GraphInstruct with several existing datasets on graph reasoning tasks. To the best of our knowledge,
```
![Detailed description of the chart](assets/page_0004_img_1.png)
```

### --- Page 0005 ---

```markdown
| Table 1 | Comparisons of GraphSolver among other related datasets |
|---------|----------------------------------------------------------|
| Task    | Node Class. [19] | Graph Structure | Node ID | Graph Description |
|---------|------------------|----------------|--------|------------------|
| NciGraph [11] | ✓ | ✓ | ✓ | ✓ |
| GraphQA [20] | ✓ | ✓ | ✓ | ✓ |
| GraphPrompt [21] | ✓ | ✓ | 10 | ✓ |
| SAGE [22] | ✓ | 8 | 4 | ✓ |
| Node2Vec [23] | ✓ | ✓ | 9 | 4 |
| GraphLogic [24] | ✓ | ✓ | 13 | ✓ |
| GraphInstruct (Ours) | ✓ | ✓ | 21 | ✓ |

GraphInstruct is the most comprehensive dataset, encompassing 21 classical tasks and providing detailed intermediate reasoning steps for each sample. In addition, GraphInstruct could provide a variety sample distribution due to the diverse options during graph generations, making it a fertile soil for developing LLMs' graph understanding capability.

## 4 GraphSolver: Towards Graph Understanding
Thanks to the diversity inherent in GraphInstruct, we posit that this benchmark holds the potential to enhance LLMs with graph understanding capability. With LLaMA-3.1-8B-Instruct as the foundational model, we first construct a model named GraphSolver through instruction tuning on the GraphInstruct. In detail, given a sample $G, \mathcal{A}, \mathcal{R} \in \mathcal{P}(Q|(G, \mathcal{A}))$, we aim to minimize the following loss function:

$$
L = -\mathbb{E}_{G, \mathcal{A}, \mathcal{R}} \log P(Q|(G, \mathcal{A}))
$$

Note that, at this stage, our main aim is to develop fundamental graph understanding capability for LLMs, thus we do not include reasoning steps here. The role of intermediate reasoning steps will be analyzed in Section 5.

Specifically, we employ LoRA [40] as the fine-tuning strategy, which is adapted to attention models and feed-forward networks, and the rank dim is set to 8. Adam [41] is used as the optimizer, with the learning rate set of $1e^{-4}$ and weight decay of $0.01$. The batch size is set to 8 during the training process, and a cosine learning rate scheduler is employed. We fine-tune the model for 4 epochs based on LLaMA-3.1-8B [42], and all experiments are implemented on a machine with 4 32GB NVIDIA V100 GPUs. As an 8-bit-parameter LLaMA model, the computation cost of GraphSolver is actually comparable to other LLMs of similar size. To further facilitate the inference efficiency, we employ vLLM [43] techniques for all models, which greatly reduces the inference time cost.

### 4.1 Experimental Settings
#### Dataset split
The tasks in GraphInstruct are divided into two groups according to whether the tasks appear in the training data during fine-tuning the model, which we call In-domain tasks and Out-of-domain tasks. In detail, we choose four relatively difficult tasks named BFS, Cycle, Clustering Coefficient, and Euler Path as the Out-of-domain tasks, and the rest as In-domain tasks. The reasons are two-fold: 1) The four tasks are inherently challenging, demanding advanced graph comprehension and reasoning capabilities from LLMs for successful completion; 2) These four tasks comprehensively cover node-level, node pair-level, and graph-level aspects, and the required output format for each task involves boolean, numerical values, and node sets, which are quite diverse. For each in-domain task, 800 samples are randomly selected as the training set, which consists of 400 samples with in-level graph size and 400 samples with small-level graph size. As for test, we randomly select 100 samples as the test set for each task, which results in a final test set with 2100 samples. Note that, graphs generated in GraphInstruct are described with adjacency tables in natural language and integer node ID by default. The generalization analysis to more graph description options will be given in Section 4.5.

### Baselines
Besides LLaMA-3.1-8B-Instruct1, we also evaluate multiple powerful LLMs with comparable model scale, including Qwen-2.5-7B-Instruct2, Phi-1-mini-Instruct3, Mistral-7B-Instruct-v0.34, and two proprietary LLMs namely GPT-3.5 Turbo and GPT-4.0-mini5. For all baselines, the temperature $t$ is set to 0 for obtaining determined answers. It is worth noting that we do not include GNNs like GCN [44] and GraphSAGE [45] here due to their fundamentally different paradigms [19] and the difference of downstream tasks. GNNs are task-specific models and are based on network data, while we aim to introduce a model that has a general graph understanding capability and can handle different graph inputs. In this way, we believe that our model is complementary to existing GNNs.

### 4.2 Main Experiments
In the next, we first investigate the performance of GraphSolver and several powerful LLMs on GraphInstruct. Specifically, the zero-shot accuracy for any in-domain tasks is reported, since we contend that the zero-shot performance can better reflect the LLM's fundamental comprehension of graph data. The results are shown in Figure 1, from which we can observe that:

1. For open-sourced LLMs like Phi-4-mini-Instruct and Qwen-2.5-7B-Instruct with relatively small model sizes, their performance on GraphInstruct is poor, which implies LLMs' lack of capability towards graph data, especially for models with fewer parameters.
2. Following fine-tuning, GraphSolver demonstrates a substantial improvement over LLaMA-3.1-8B-Instruct across the majority of tasks, showcasing performance that is comparable or even superior to GPT-3.5 Turbo. For example, LLaMA-3.1-8B-Instruct achieves a mere 15% accuracy in the Neighbor task, whereas GraphSolver attains an impressive 99% accuracy, surpassing even the performance of GPT-3.5 Turbo and being comparable with GPT-4-mini.

Despite the impressive performance on certain tasks, such as straightforward graph reasoning tasks like Degree and Neighbors, both GPT-4-mini and GraphSolver exhibit less satisfactory results when tackling more complex graph reasoning tasks, including MST and PageRank. This underscores the existing challenges that LLMs face when tackling more complex reasoning tasks.

![Detailed description of the chart](assets/page_0005_img_1.png)
```

### --- Page 0006 ---

```markdown
# Zihan Luo et al. GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability

![Accuracy comparison of several LLMs on 17 in-domain tasks of GraphInstruct](assets/page_0006_img_1.png)

## Table 2 The effect of different prompt engineering techniques

| GraphInstruct-specific tasks | Shared tasks |
|------------------------------|--------------|
|                              | DFS          | Degree       | Common Neighbor | Connectivity | Predecessor |
| GPT-3.5 Turbo                | 0.37         | 0.88         | 0.77           | 0.71         | 0.19        |
| Zero-shot                    | 0.92         | 0.99         | 0.63           | 0.78         | 0.99        |
| GraphSolver                  | 0.99         | 0.99         | 0.64           | 0.79         | 0.99        |
| CoT                          | 0.90         | 0.99         | 0.98           | 0.98         | 0.05        |

![Performance comparisons on both GraphInstruct-specific tasks and shared tasks with two competitor models](assets/page_0006_img_2.png)

## 4.3 Comparison with Other Competitors
In this section, we aim to compare GraphSolver with two highly related competitors, namely GraphWiz and GraphSol. Specifically, we conduct evaluations on (1) tasks that are shared across all works — Cycle, Connectivity, and Maximum Flow, and (2) two tasks that are exclusive to the GraphInstruct — Bipartite and Connected Component. The results are shown in Figure 5, where GraphSolver consistently achieves the best performance on all tasks, especially on two tasks that only GraphInstruct includes. Such observations highlight the significance of introducing a comprehensive training process for developing the capability on graph reasoning tasks for LLMs.

## 4.4 Results with Prompt Engineering
Table 2 presents the effect of prompt engineering techniques such as Few-shot and CoT across five graph reasoning tasks: DFS, Degree, Common Neighbor, Connectivity, and Predecessor. For comparison, we also report the zero-shot results of GPT-3.5 Turbo as a reference. It can be seen that all three prompting-enhanced variants of GraphSolver — zero-shot, one-shot, and zero-shot CoT — exhibit substantially stronger performance across most tasks. However, the differences among the three prompting strategies were minimal. Zero-shot, one-shot, and CoT yielded nearly identical results across all metrics, with only marginal fluctuations. We assume that the limited impact of prompt engineering might be attributed to the intrinsic reasoning capabilities already embedded in the fine-tuned model - GraphSolver.

## 4.5 Generalization Analysis
This section focuses on analyzing the generalization capabilities of GraphSolver from the following four perspectives: increasing graph sizes, distinct graph description languages, various node representation methods, and out-of-domain tasks.
- **Graph size**: Throughout the training process, GraphSolver has exclusively encountered tiny and small graphs. Consequently, we aim to explore whether GraphSolver maintains its graph understanding capabilities when con-
```

### --- Page 0007 ---

```markdown
| Task                     | GPT-3.5 Turbo | Phi-1.4-mini | Qwen-2.5-78 | Mistral-7B-8 | LLaMA-3.1-8 | GraphSolver |
|--------------------------|----------------|---------------|-------------|--------------|--------------|-------------|
| Shortest Path            | 0.06           | 0.33          | 0.42        | 0.34         | 0.01         | 0.13        |
| Hamiltonian Path         | 0.16           | 0.15          | 0.11        | 0.99         | 0.00         | 0.01        |
| Edge                     | 0.32           | 0.16          | 0.65        | 0.62         | 0.37         | 0.76        |
| Connected Component       | 0.28           | 0.16          | 0.18        | 0.12         | 0.00         | 0.10        |
| Connectivity             | 0.78           | 0.74          | 0.37        | 0.58         | 0.59         | 0.38        |
| Processor                | 0.29           | 0.31          | 0.22        | 0.00         | 0.00         | 0.04        |
| Common Neighbor          | 0.42           | 0.31          | 0.22        | 0.31         | 0.99         | 0.31        |
| Diameter                 | 0.45           | 0.48          | 0.10        | 0.12         | 0.04         | 0.26        |

| Task                     | GPT-3.5 Turbo | Phi-1.4-mini | Qwen-2.5-78 | Mistral-7B-8 | LLaMA-3.1-8 | GraphSolver |
|--------------------------|----------------|---------------|-------------|--------------|--------------|-------------|
| Shortest Path            | 0.06           | 0.33          | 0.42        | 0.34         | 0.01         | 0.13        |
| Hamiltonian Path         | 0.16           | 0.15          | 0.11        | 0.99         | 0.00         | 0.01        |
| Edge                     | 0.32           | 0.16          | 0.65        | 0.62         | 0.37         | 0.76        |
| Connected Component       | 0.28           | 0.16          | 0.18        | 0.12         | 0.00         | 0.10        |
| Connectivity             | 0.78           | 0.74          | 0.37        | 0.58         | 0.59         | 0.38        |
| Processor                | 0.29           | 0.31          | 0.22        | 0.00         | 0.00         | 0.04        |
| Common Neighbor          | 0.42           | 0.31          | 0.22        | 0.31         | 0.99         | 0.31        |
| Diameter                 | 0.45           | 0.48          | 0.10        | 0.12         | 0.04         | 0.26        |

![Zero-shot performance (Accuracy) on 4 OOD graph reasoning tasks in GraphInstruct](assets/page_0007_img_1.png)

### Graph description language
To delve deeper into the generalization capabilities of GraphSolver across diverse graph reasoning languages, we conduct experiments on eight classic graph reasoning tasks using ten graph description languages: edge list and adjacency table (adj). The results, as illustrated in Table 3, reveal that there is a slight performance decrease when compared to that using the adjacency table in natural language; GraphSolver still consistently surpasses other LLMs, even including proprietary models like GPT-3.5 Turbo. The above observations verify GraphSolver's capability to retain a substantial portion of its graph understanding capabilities when facing different graph description languages.

### Node ID representation
In contrast to the integer node IDs utilized in the training process (e.g., Node ID) of GraphSolver, we perform generalization tests by substituting node IDs in the test set with random letters (e.g., Node XTE). The results on eight representative graph reasoning tasks, depicted in Table 4, highlight that even with the altered node IDs, GraphSolver consistently outperforms the baseline models like Phi-1.4-mini and GPT-3.5 Turbo, affirming its retention of graph understanding capabilities.

### 5 GraphSolver+: Towards Graph Reasoning
Previous experiments have demonstrated that GraphSolver possesses a certain level of graph understanding capability after fine-tuning. However, relying solely on supervision with final answers might limit its graph reasoning capability. In other words, GraphSolver may only have an intuitive understanding of graph data without being able to provide the reasoning steps for an answer. In this part, we further explore how to utilize the intermediate reasoning steps provided by GraphInstruct to enhance the graph reasoning capability of LLMs.

#### 5.1 Methodology
Different from the training approach in Section 4, we incorporate intermediate steps $S$ as supervised signals into the training data as well to empower LLMs with graph reasoning capability, and construct a new model namely GraphSolver+. Based on instruction-tuning, the loss function utilized at this stage can be formulated as:

$$
L = -\mathbb{E}_{G,Q,S}\left[\log(P(S|G, A); \theta)\right], 
$$

where given a graph $G$ and corresponding query $Q$, GraphSolver+ with parameters $\theta$ are trained to output reasoning steps $S$ before getting the final answer $A$.
```

### --- Page 0008 ---

However, unlike mathematical reasoning tasks [1], intermediate steps $S = \{s_1, s_2, \ldots, s_T\}$ in graph reasoning tasks may contain numerous repetitive and redundant nodes, as shown in Figure 7(a), which could obscure crucial node ID information. Therefore, we employ a label mask training strategy, where the supervised signals corresponding to unimportant tokens are ignored with a certain probability $\rho$, not contributing to the final loss function, while preserving information relevant to node IDs entirely:

$$
\mathcal{L} = -\mathbb{E}_{\text{G},\text{A},\text{S} \sim \mathcal{D}} \left[ \sum_{t=1}^{T} m_t \log P\left(s_t \mid \text{G}, \text{Q}, s_{<t}; \theta\right) + \log P\left(\text{A} \mid \text{G}, \text{Q}, s_{<t}; \theta\right) \right]
$$

where $m_t \in \{0, 1\}$ indicates if the token $s_t$ will be ignored in final loss, and $f(s_t)$ is an indicator function examining if the token $s_t$ contains node ID information. $\gamma$ is a predefined hyperparameter. The benefits of label masking training is two-fold: 1) Through this approach, we can filter out a significant amount of redundant supervised signals, avoiding the overfitting issues on unimportant tokens; 2) With masking, our model can better capture the long-distance dependencies among node IDs.

![Toy graph for illustrating the concept of label mask training strategy](assets/page_0008_img_1.png)

### 5.2 Main Experimental Results
To be concrete, we employ five classical graph reasoning tasks from GraphInstruct: DFS, Predecessor, Degree, Common Neighbor, and Connectivity for training. Each task provides 10,000 training samples, and the token mask probability is set to 0.8 after tuning. To evaluate the reasoning capacity of GraphSolver+, we employ zero-shot CoT techniques that prompt LLMs to generate answers with detailed reasoning steps. As demonstrated in Figure 8, our evaluation across small, medium, and large-scale test sets reveals the following observations:
- GraphSolver+ significantly outperforms all baseline models across all five tasks, even including reasoning models like Qwen-3B when intermediate reasoning steps are incorporated into the supervision signals. Notably, in the medium-size DFS task, GraphSolver+ achieves 53% accuracy - a sixfold improvement over LLaMA-3.1-8B-Instruct's performance (approximately 9%).

![Zero-shot CoT performance (Accuracy) comparisons on five classical graph reasoning tasks. Here we enable the thinking mode for Qwen-3B.](assets/page_0008_img_2.png)

| Model          | DFS  | Degree | Common Neighbor | Connectivity | Predecessor |
|----------------|------|--------|------------------|--------------|-------------|
| GraphSolver    | 0.9  | 0.87   | 0.86             | 0.84         | 0.84        |
| GraphSolver+   | 0.83 | 0.84   | 1.00             | 0.90         | 0.87        |

Additionally, while all models exhibit performance degradation as graph size increases, GraphSolver+ maintains consistent superiority across all graph sizes, and marginal performance degradation on tasks like Degree, Common Neighbor, and Connectivity. This demonstrates the robustness of GraphInstruct in helping models handle increasingly complex graph structures and the generalizability of GraphSolver+.

To our surprise, GraphSolver demonstrates comparable performance with GraphSolver+ on tasks like Connectivity and Predecessor when they are asked to address these graph reasoning tasks step by step. However, our further in-depth step evaluation in the following table indicates that GraphSolver actually poses an unsatisfactory reasoning capacity compared to GraphSolver+. We leave the corresponding analysis to the next part.

### 5.3 Steps Evaluation
To evaluate the reasoning capability of GraphSolver and GraphSolver+, we check the correctness of the reasoning steps output by GraphSolver and GraphSolver+, which mainly contains 100 test samples for each task. Table 6 presents the accuracy of the final answer and intermediate steps output by models on five classical graph reasoning tasks under the zero-shot CoT setting, from which we can observe that: When fine-tuning without using intermediate reasoning steps as supervised signals, GraphSolver's intermediate reasoning steps outputs always turn out to be incorrect. For example, in the Common Neighbor task, although the final answers are correct, none of them has correct intermediate step outputs. The possible reason for this issue is that when only using final answers as supervised signals, the fine-tuned GraphSolver may only develop a shallow understanding of graph data and graph reasoning tasks, unable to provide correct reasoning processes and explanations. In contrast, GraphSolver+ maintains much higher accuracy in intermediate reasoning steps across all five tasks.

### --- Page 0009 ---

```markdown
# Front. Comput. Sci., 2025, 0(0): 1

## 6 Conclusion
To improve the graph understanding capability and graph reasoning capability of LLMs, we propose a dynamic benchmark called GraphInstuct in this paper, which comprehensively provides 21 classic graph reasoning tasks with diverse graph generation pipelines and detailed intermediate reasoning steps. Based on GraphInstuct, we construct GraphSolver and GraphSolver+ through supervised fine-tuning with a novel label mask training strategy. Comprehensive experiments demonstrate that models trained on GraphInstuct could pose superior graph understanding capability and reasoning capability. We expect that GraphInstuct could serve as a fertile soil for future researchers to enhance LLMs' capabilities in graph data mining.

## Limitations and Discussions
Firstly, we observe that our models still under-perform when handling some complex graph reasoning tasks like MST and PageRank, and out-of-domain tasks like Euler Path and Clustering Coefficient. The reasons might be two-fold: 1) The models may struggle with capturing long-range dependencies required by these tasks. Even with intermediate reasoning steps, the GraphSolver+ may face issues of memory loss or information leakage when reasoning over long or complex paths, especially in larger graphs or tasks requiring detailed numerical computations; 2) The training samples for these tasks may still be insufficient. Currently, the current GraphInstuct is limited to abstract graph reasoning tasks without specific applications. While these tasks provide a valuable foundation for evaluating and enhancing graph reasoning capabilities, they do not fully capture the complexity of real-world scenarios where graph-based reasoning plays a critical role. Moving forward, one of the key directions for our future work is to explore the development of more application-specific graph reasoning tasks that are directly relevant to real-world problems, such as urban computing and biological network analysis.

## Acknowledgements
This work was supported by the National Natural Science Foundation of China (Grant No. 62172174).

## Competing Interest
The authors declare that they have no competing interests or financial conflicts to disclose.

## References
1. Hou Z, Lv X, Lu R, Zhang J, Li Y, Yao Z, Li J, Tang J, Dong Y. 2021: Advancing language model reasoning through reinforcement learning and inference scaling. In: Proceedings of the 42nd International Conference on Machine Learning; 2025  
2. Luo H, Sun Q, Xu C, Zhao P, Lou J, Tao C, Geng X, Lin Q, Chen S, Tang Y, Zhang D. WizardMath: Empowering mathematical reasoning

![Ablation study of GraphSolver+](assets/page_0009_img_1.png)
![Case study on the output of GraphSolver+ and GPT-3.5 Turbo](assets/page_0009_img_2.png)
```

### --- Page 0010 ---

```markdown
# Zihan Luo et al.  GraphInstruction: Empowering Large Language Models with Graph Understanding and Reasoning Capability

| Reference | Year |
|-----------|------|
| [1] Luo Z, Xu C, Zhao P, Sun Q, Geng X, Hu W, Tao C, Ma J, Lin Q, Jiang D. WizardCoder: Empowering code large language models with eval-instruct. In: Proceedings of the 13th International Conference on Learning Representations. | 2025 |
| [2] Zhang Z, Li R, Tan M, Yang M, Zhu J, Yang D, Zhao J, Ye G, Li C, Hu X. CPsyCount: A report-based multi-turn dialogue reconstruction and evaluation framework for chinese psychological counseling. In: Findings of the 2022 Annual Meeting of the Association for Computational Linguistics. | 2022 |
| [3] Alharbi A, Donahue L, Puc M, Miech A, Bar H, Hasson Y, Lenc K, Meschi A, Milken K, Reynolds M, Ring R, Rutherford C, Sabri B, Han T, Gong Z, Samanoeedi S, Monteiro M, Nicki L, Borgeaud S, Brook A, Nazmatzadeh A, Sharifzadeh S, Brinowski M, Barreira R, Vinyals O, Zisserman A, Simonyan K. Flamingo: a visual language model for few-shot learning. In: Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022. | 2022 |
| [4] Liu Z, Fang F, Feng X, Du X, Zhang C, Wang H, Bai Y, Zhao L, Fan L, Gan C, Lin H, Li J, Ni Y, Wu H, Narasupalli Y, Zheng Z, Li C, Hu X, Yu R, Chen X, Yang M, Liu J, Liu R, Huang W, Zhang G, Ni S. IBI-Bench: An image implication understanding benchmark for multimodal large language models. In: Advances in Neural Information Processing Systems 2024. | 2024 |
| [5] Huang R, Li M, Yang D, Shi J, Chang Y, Xie Z, Wu Y, Hong Z, Huang J, Liu J, Ren Y, Zou Y, Zhao Z, Watanabe S. AudioGPT: Understanding and generating speech, music, sound, and talking head. In: Proceedings of the 38th AAAI Conference on Artificial Intelligence. | 2024 |
| [6] Xiong X, Lian J, Huang H, Luo Z, Zhu L, Xin W, Mu L, Li C, Xie X, Jin H. xGCN: An extreme graph convolutional network for large-scaled link prediction. In: Proceedings of the ACM Web Conference 2023. | 2023 |
| [7] Zheng K, Zou Y, Xiang S, Jiang C. Graph neural networks for financial fraud detection: A review. Frontiers Comput. Sci., 2025, 19(5): 199505 | 2025 |
| [8] Zhang X, Lei X. Predicting mira-drug interactions via dual-channel network based on TCN and bilstm. Frontiers Comput. Sci., 2025, 19(5): 199505 | 2025 |
| [9] He H, Chen G, Chen C Y C. Integrating sequence and graph information for enhanced drug-target affinity prediction. Science China Information Sciences, 2024, 67(2): 129101 | 2024 |
| [10] Zhu K, Chen J, Wang J, Gong N Z, Yang D, Xie X. DyVal: Graph-informed dynamic evaluation of large language models. In: Proceedings of the 12th International Conference on Learning Representations. | 2024 |
| [11] Peng M, Chen N, Suo Z, Li J. Rewarding graph reasoning processes makes lms more generalized reasoners. In: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. | 2025 |
| [12] Zhang Q, Chen N, Li Z, Peng M, Tang J, Li J. Improving lms' generalized reasoning abilities by graph problems. In: Proceedings of the 2nd Conference on Language Modeling. | 2025 |
| [13] Guo J, Du L, Li H. GPT4Graph: Can large language models understand graph structured data? An empirical evaluation and benchmarking. CoRR, 2023, abs/2305.15066 | 2023 |
| [14] Wang H, Feng S, He T, Tan Z, Han K, Tsvetkov Y. Can language models solve graph problems in natural language? In: Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023. | 2023 |
| [15] Chen N, Li Y, Tang J, Li J. GraphViz: An instruction-following language model for graph computational problems. In: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. | 2024 |
| [16] Kojima T, Gu S, Reid M, Matsuo Y, Iwasawa Y. Large language models are zero-shot reasoners. In: Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022. | 2022 |
| [17] Wei J, Wang X, Schuurmans D, Bosma M, Ichter B, Xia F, Chi E H, Le Q, Zhou D. Chain-of-thought prompting elicits reasoning in large language models. In: Advances in Neural Information Processing Systems 2022. | 2022 |
| [18] Xu H, Jian X, Zhao X, Pang W, Zhang C, Wang S, Zhang Q, Monteiro J, Sun Q, Yu T. GraphPoint: A comprehensive and extendable | 2025 |
```


### --- Page 0011 ---

```markdown
# Front. Comput. Sci., 2025, 0(0): 1

benchmark framework for large language models on graph-theoretic tasks. CoRR, 2025, abs/2504.12766  
[27] Zhang Q., Hong X., Tan J., Chen N., Li Y., Li W., Tang J., Li J. GCoder: Improving large language model for generalized graph problem solving. In: Proceedings of the 34th ACM International Conference on Information and Knowledge Management. 2025, 4149–4159  
[28] Wang R., Liang S., Shen Q., Zhang J., Qin K. GraphTool: Instruction: Revolutionizing graph reasoning in LLMs through decomposed subtask instruction. In: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, V.I. 2025, 1492–1503  
[29] Xu C., Sun Q., Zheng K., Geng X., Zhao P., Feng J., Tao C., Lin Q., Jiang D. WizardLM: Empowering large language models to follow complex instructions. In: Proceedings of the 12th International Conference on Learning Representations. 2023  
[30] Zhuo T., Yu M. C., Chim J., Hu W., Widyasari R., Yusuf I. B., Zhan H., He J., Paul I., Bruner S., Gong C., Hoang T., Zebaze A. R., Hong X., Liu L., Kaddour J., Xu M., Zhang Z., Yadav P., Jain N., Gu A., Cheng Z., Liu J., Liu Q., Wang Z., Lo D., Hui B., Mueninghoff N., Freid D., Du X., Vries H., Werrv A. L. BigCodeBench: Benchmark code generation with diverse function calls and examples. In: Proceedings of the 13th International Conference on Learning Representations. 2025  
[31] Zhang H., Liu Y., Wang Y., Zhang Y., Zhao Y. An effective and efficient editing framework for aligning large language model with recommendation. In: Proceedings of the 17th ACM Conference on Recommender Systems. 2023, 1007–1014  
[32] Li Y., Ma S., Yang X., Huang S., Jiang C., Zheng K., Xie P., Huang F., Jiang Y. ExomGPT: Instruction-tuning large language models with chain-of-thought tasks for e-commerce. In: Proceedings of the 38th AAAI Conference on Artificial Intelligence. 2024, 18582–18590  
[33] Kim Y., Xu X., McDuff D., Breazeal C., Park H. W. Health-LLM: Large language models for health prediction via wearable sensor data. In: Proceedings of the 5th Conference on Health, Inference, and Learning. 2024, 522–539  
[34] Qiu P., Wu C., Zhang X., Lin W., Wang H., Zhang Y., Wang Y., Xie W. Towards building multilingual language model for medicine. Nature Communications, 2024, 15(1): 8384  
[35] Yue S., Liu S., Zhou Y., Shen C., Wang S., Xiao Y., Li B., Song Y., Shen X., Chen W., others. LawLLM: Intelligent legal system with legal reasoning and verifiable retrieval. In: Proceedings of the 29th International Conference on Database Systems for Advanced Applications. 2024, 304–321  
[36] Barabási A. L., Albert R. Emergence of scaling in random networks. Science, 1999, 286(5439): 509–512  
[37] Watts D. J., Strogatz S. H. Collective dynamics of ‘small-world’ networks. Nature, 1998, 393(6684): 440–442  
[38] Ke J., Widom J. Scaling personalized web search. In: Proceedings of the 11th International World Wide Web Conference. 2003, 271–279  
[39] Zhang Y., Wang H., Feng S., Tan Z., Han X., He T., Tsvetkov Y. Can LLM graph reasoning generalize beyond permutation?

In: Findings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024, 229–2305  
[40] Hu E. J., Shen Y., Wallis P., Allen-Zhu Z., Li Y., Wang S., Wang L., Chen W. LoRA: Low-rank adaptation of large language models. In: Proceedings of the 10th International Conference on Learning Representations. 2022  
[41] Kingma D. P., Ba J. Adam: A method for stochastic optimization. In: Proceedings of the 3rd International Conference on Learning Representations. 2015  
[42] Zhang Y., Zhang R., Zhang J., Ye Y., Luo Z. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 2024, 400–410  
[43] Wang K., Li Z., Zhuo S., Sheng Y., Zheng L., Yu H., Gonzalez J. E., Zhang H., Stocca I. Efficient memory management for large language model serving with paddeatention. In: Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles. 2023  
[44] Kipf T. N., Welling M. Semi-supervised classification with graph convolutional networks. In: Proceedings of the 5th International Conference on Learning Representations. 2017  
[45] Hamilton W. L., Ying Z., Leskovec J. Inductive representation learning on large graphs. In: Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems. 2017  
[46] Chen C., Wang X., Lin T. E., Lav A., Wu Y., Gao X., Wen J. R., R. Li Y. Masked thought: Simply masking partial reasoning steps can improve mathematical reasoning learning of language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 2024, 5872–5900  

Zihan Luo received the BEng degree from Huazhong University of Science and Technology, Wuhan, China, in 2020. He is currently working toward the PhD degree in the School of Computer Science at Huazhong University of Science and Technology. His research interests include large language models and graph representation learning.

Xiran Song received his MS degree from Huazhong University of Science and Technology, Wuhan, China, in 2024. He is currently pursuing a PhD in computer science at Mila, Quebec AI Institute, affiliated with Université de Montréal. 

Hong Huang is an Associate Professor at Huazhong University of Science and Technology, China. She received her PhD in Computer Science from University of Göttingen, Germany in 2016, and her M.E. degree in Electronic Engineering from Tsinghua University, Beijing, China in 2012. Her research interests lie in social network analysis, data mining and knowledge graph.
```

### --- Page 0012 ---

```markdown
# Zihan Luo et al.  GraphInfuser: Empowering Large Language Models with Graph Understanding and Reasoning Capability

## Authors

| Name                | Affiliation                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
| Jianxun Lian       | Microsoft Research Asia. He received his Ph.D. degree from University of Science and Technology of China in 2018. His research interests include recommender systems and deep learning techniques. He has published some academic papers on international conferences such as KDD, IJCAI, WWW, SIGIR and CIKM. He serves as a program committee member for several top conferences such as AAAI, WWW and IJCAI. |
| Chenhao Zhang      | An undergraduate student from Huazhong University of Science and Technology, China. He is currently pursuing the BS degree in Computer Science (2023 - 2027). His research interests include Vision-Language Reasoning and Emotional AI. |
| Jinqi Jiang        | A senior undergraduate student at Huazhong University of Science and Technology, majoring in artificial intelligence. His research interests include large language models, efficient AI, and machine learning systems. |
| Dr. Xing Xie       | Currently a senior principal research manager at Microsoft Research Asia, and a guest Ph.D. advisor at the University of Science and Technology of China. He received his B.S. and Ph.D. degrees in Computer Science from the University of Science and Technology of China in 1996 and 2001, respectively. He joined Microsoft Research Asia in July 2001, working on data mining, social computing and ubiquitous computing. |
| Hai Jin            | A Chair Professor of computer science and engineering in China. He received his Ph.D in computer engineering from HUST in 1994. Jin is a Fellow of IEEE, Fellow of CCF, and a life member of the ACM. He has co-authored more than 20 books and published over 900 research papers. His research interests include computer architecture, parallel and distributed computing, big data processing, data storage, and system security. |

---

Frontiers of Computer Science | Issue 0 | Volume 0 | January 2025 | 1–12
```


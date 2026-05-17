# ArXiv 2401.13256

### --- Page 0001 ---

```markdown
# UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems

Hongru Wang, Wenyu Huang, Yang Deng, Rui Wang, Zezhong Wang, Yufei Wang, Fei Mi, Jeff Z. Pan, Kam-Fai Wong

---

**Abstract**—Large Language Models (LLMs) have shown exceptional capabilities in many natural language understanding and generation tasks. However, the personalization issue still remains a much-covered property, especially when it comes to the multiple sources involved in the dialogue system. To better plan and incorporate the use of multiple sources in generating personalized response, we firstly decompose it into three sub-tasks: Knowledge Source Selection, Knowledge Retrieval, and Response Generation. We then propose a novel Unified Multi-Source Retrieval-Augmented Generation system (UniMS-RAG). Specifically, we unify these three sub-tasks with different formulations into the same sequence-to-sequence paradigm during the training, to adaptively retrieve evidences and evaluate the relevance on-demand using special tokens, adding tokens and evaluation tokens. Enabling language models to generate acting tokens facilitates interaction with various knowledge sources, allowing them to adapt their behavior to diverse task requirements. Meanwhile, evaluation tokens can provide the relevance score between the dialogue context and the retrieved evidence. In addition, we carefully design a self-reinforcement mechanism to iteratively refine the generated response considering (1) the consistency scores between the generated response and retrieved evidence; and (2) the relevance scores. Experiments on personalized datasets (DuLeMo and KPB) show that UniMS-RAG achieves better performance than previous strong baselines on the knowledge source selection and response generation task itself as a retriever in a unified manner, and achieves state-of-the-art when using more than one retrieved reference. Extensive analyses and discussions are provided for shedding some new perspectives for personalized dialogue systems.

**Index Terms**—Open-domain Dialogue System, Large Language Models, Retrieval-Augmented Generation

---

## I. INTRODUCTION

The emergence of large language models (LLMs) has revolutionized the field of natural language processing, including many downstream understanding and generation tasks [1]. While these models have undeniably advanced the state of the art in various applications, they also introduce new challenges, particularly the factual error [2, 3] and personalization issues [4, 5] in the realm of dialogue systems. To alleviate this, Retrieval-Augmented Generation (RAG) methods are usually adopted to retrieve relevant passages, aiming to enrich the semantic information of the dialogue context,

Hongru Wang, Rui Wang, Zezhong Wang, and Kam-Fai Wong are with the Department of Systems Engineering and Engineering Management, The Chinese University of Hong Kong. e-mail: hrwang@se.cuhk.edu.hk, Wenyu Huang and Jeff Z. Pan is with EdinburghNLP, The University of Edinburgh, United Kingdom.
```

### --- Page 0002 ---

```markdown
# Knowledge Retrieval (retriever) sequentially retrieves top-n evidence from external sources according to the decisions in the last step.

# Response Generation (reader) produces knowledge-grounded natural language responses to users according to original dialogue context and retrieved evidence.

Then we design a novel framework, Unified Multi-Source Retrieval-Augmented Dialogue System (UniMS-RAG), that unifies three tasks in the above using the same large language models in a sequence-to-sequence (Seq2Seq) manner. In specific, motivated by recent works of assigning different tokens with different roles [1, 6], we carefully introduce two types of tokens: 1) acting tokens to decide the next action (e.g., which source to use), aiming to call different source-on-demand instead of incorporating all of them; and 2) evaluation tokens to evaluate the relevance between dialogue context and retrieved evidence (e.g., the similarity score), in order to force the model attention on more relevant evidence while overlooking noisy ones. Thus we can reformulate the above three tasks as token prediction tasks by generating acting tokens (planner), evaluation tokens (retriever) or original tokens in the vocabulary (reader) during the training. We randomly shuffle the order of retrieved evidences to prevent model learning a shortcut by attributing to evidences in specific positions. To further enhance the quality of generated responses, we incorporate a self-refinement process during the inference stage. This involves reassessing the generated responses by leveraging feedback from evaluation tokens and ensuring consistency between the provided evidence and responses. To sum up, the contributions are summarized as follows:

- We formally propose a multi-source personalized knowledge-grounded dialogue tasks, consisting of three different subtasks: knowledge source selection, knowledge retrieval and final response generation. Our analysis provides detailed insights into the limitations and proficiencies of current LLMs across these sub-tasks.
- We propose a novel method, namely, UniMS-RAG, that tackles all sub-tasks in PerDS with a unified model. To the best of our knowledge, it is the first attempt to utilize LLMs as the planner, retriever and reader at the same time.
- We investigate the different strategies to get the soft label of evaluation tokens during the training stage, including prompting LLMs or using an independent fine-tuned retriever (a.k.a. classification-based and prompting-based methods). Furthermore, we propose a self-refinement mechanism to regenerate the response using updated evidence according to its relevance with the dialogue context and previously generated response.
- Experimental results on two PerDS benchmark datasets show that UniMS-RAG outperforms previous strong baselines, and achieves state-of-the-art performance with external more effective retrieval, resulting in more personalized and factual responses. Extensive analyses provide some new insights into the future of multi-source retrieval-augmented generation tasks.

![Two typical examples of multi-source personalized knowledge-grounded dialogues: upper: An example from DuLeMon [7]; and bottom: An example from KBP [11]. We use same color to indicate the response and corresponding grounded knowledge. We skip the dialogue context for simplicity.](assets/page_0002_img_1.png)

## II. RELATED WORK

### A. Personalized Dialogue System

To build a personalized dialogue agent, Zhang et al. [15] firmly investigated this task with a new dataset Persona-Chat, where a pre-defined persona set is a form of multi-turn dialogue of textual description. Lots of works follow this setting and have taken various approaches [16, 17], persona-sparse scenario [18, 19], long-term persona memory [7], preserving context and personal record bias [21] into consideration. Although some of them complement the insufficient semantics in short persona descriptions by further utilizing an external commonsense knowledge base to extend existing sources [22, 20], they still fail into the conventional framework coupling the knowledge selection with the response generation [23], rendering it infeasible to handle various sources of knowledge. There have also been works showing that the combination of different knowledge sources such as persona descriptions and Wikipedia can further improve the overall performance [24, 25, 26]. However, they still fail to capture possible dependency between knowledge sources. In their framework, knowledge is not used as the role to assist persona-consistent response generation, but as an additional resource to generate a more informative response [2, 3] or select a suitable persona [24, 28]. Furthermore, most existing works overlook the possibilities that the response does not require the involvement of persona descriptions by simply concatenating all personas with the dialogue context to generate the final response [22, 17].

### B. Knowledge-grounded Dialogue System

How to interact with different external sources plays a key role in dialogue systems to retrieve corresponding knowledge, resulting in more helpful, personalized, trustworthy responses [29, 30]. Specifically, most of previous methods rely on different external sources of knowledge to engage the user and improve the conversational experience, including but not limited to system persona to maintain consistency in responses [11], user memory or profile for personalized interactions [31, 32], and
```

### --- Page 0003 ---

```markdown
## C. Retrieval-augmented Generation

Retrieval-augmented Generation (RAG) has been considered as an effective method to overcome several limitations of LLMs, such as hallucinations [36], [41], [37] and long-term memory [7]. Usually, an external retriever is first used to retrieve relevant textual knowledge from one specific knowledge source (e.g., Wikipedia), then the reader takes the relevant textual knowledge as external context for generating knowledge-grounded response [38]. Most of previous works try to optimize the retriever and reader independently [39]. During the initial phases, people use sparse retriever, such as BM25 [40], to make relevance decisions and retrieve corresponding evidence. However, sparse approaches fall short in extracting the semantic features inherent in text content [41]. To overcome this issue, researchers have proposed language model-based dense retrieval, which effectively represent the semantic features of text content [42], [43], [44]. For example, DPR [46] uses language models to encode documents and queries separately, allowing for a more nuanced understanding of the content. More recently, there are a handful of works exploring the performance of LLMs as retriever [45, 46, 47, 48]. In detail, Shen et al. [47] firstly prove that LLMs can be used as strong zero-shot retriever on several benchmark datasets, while Ma et al. [48] propose Listwise Reranker with a Large Language Model (LRL), which achieves strong ranking effectiveness without using any task-specific training data. Distinguishing from previous works, we fine-tune LLM itself to learn a joint distribution of dialogue and evidence using similarity feedback from current most powerful LLMs, such as ChatGPT and GPT-4.

## II. PROBLEM DEFINITION

To consider the responses which require external knowledge and those which do not in practice, we provide a unified definition for the retrieval-augmented dialogue system and non-retrieval dialogue system, following Wang et al. [11]. Let $C = \{c_1, c_2, \ldots, c_t\}$ denote the dialogue context at the current conversation turn, and let $K = \{k_1, k_2, \ldots, k_n\}$ denote the different knowledge sources $K_j = \{k_{j1}, k_{j2}, \ldots, k_{j|K_j|}\}$ indicates the $j^{th}$ source's name of $K_j$ denotes the knowledge in natural language form $k_i$. The goal of retrieval-augmented dialogue system1 is to select suitable knowledge sources, and then generate the helpful, informative and personalized response, depending on which source is chosen [30]. Thus, the problem of PerDS can be decomposed into the following three tasks:

- **Knowledge Source Selection.** At each turn, if given the dialogue context $C_t$ and different knowledge sources $S$, PerDS first select suitable sources denoted as $S' \subseteq S \cup \{\text{NULL}\}$. Notably, there are no constraints imposed on the relationships between different sources within $S$. They can either be independent or interdependent.

- **Knowledge Retrieval.** The second task is to retrieve top-$k$ corresponding evidence $E = \{e_1, e_2, \ldots, e_k\}$ from each selected source if there is. We simply skip this once the PerDS determiner there is no need to call external knowledge.

- **Response Generation.** The final task is to generate a proper response $r$, concerning the dialogue context $C_t$ and necessary evidences $E$ from different external knowledge sources $S$ if there is. The generated response is expected to ground on these evidences, being personalized, informative, up-to-date and helpful according to the distinctions across different sources.

## IV. METHOD

In this section, we first describe the framework that reformulates each task in PerDS into the unified Seq2Seq paradigm, and then introduce the joint-training strategies for the retriever and reader modules and the inference strategy to re-evaluate the quality of response, respectively. The overview and examples of the input and output sequences for UniMS-RAG are illustrated in Figure 2.

### A. UniMS-RAG

Inspired by previous work [11], we propose an innovative methodology termed UniMS-RAG, where $U$ signifies the unification of the training process for planner, retriever and reader, as well as the integration of diverse tasks into a singular comprehensive framework. Recognizing the potential of large sources, and then generate the helpful, informative and personalized response, depending on which source is chosen [30]. Thus, the problem of PerDS can be decomposed into the following three tasks:
```
![Proposed method UniMS-RAG with optimization tasks](assets/page_0003_img_1.png)


### --- Page 0004 ---

```markdown
# Page 0004

language models (LLMs) in orchestrating the utilization of external sources of knowledge, as indicated by recent works, UniMS-RAG extends the capabilities of LLMs to seamlessly connect disparate sources within the context of personalized knowledge-grounded dialogues. This integration streamlines the traditionally separated tasks of retriever and reader training, enabling to adaptively retrieve evidence and evaluate the relevance sources in a unified manner.

To address the interactions between different subtasks, as illustrated in Figure 2, the whole response generation can be modeled into three steps in UniMS-RAG: 1) Planning, to make a series of decisions about whether to use a specific knowledge source given relationship descriptions between different sources; 2) Retrieval, to retrieve top-n results from external databases according to the decisions; 3) Generation, to incorporate retrieved knowledge (if required) into the final response generation. Taking advantage of the decoupling of these different tasks, the UniMS-RAG framework exhibits versatility and scalability in its applicability to various retrieval-augmented response generation tasks. For example, it can be achieved through targeted modifications to the Planning step. By configuring decisions within this phase, the model can seamlessly accommodate different retrieval-augmented tasks without necessitating extensive adjustments to other components.

1) **Planning**: First of all, to incorporate the cases which does not require any sources of external knowledge, we define several additional index tokens, corresponding to different sources, including the token for no knowledge as shown in Table 1:

$$
M : c = NULL,
$$

where $M$ is parameterized by LLMs. Secondly, there are two different scenarios between these multiple knowledge sources: 1) the $K_1, K_2, \ldots, K_n$ is independent, which means there is no interdependent relationship between them; and 2) some or even all of them are not independent, for example, the results obtained from $K_2$ may be contingent on the outcomes derived from $K_1$ and potentially other sources. These two situations cater to different applications. In the first scenario, the independence between knowledge sources offers practical utility for tasks where autonomy and isolation of information are paramount, such as user persona and system persona, which are two independent knowledge sources. On the other hand, in the second scenario, where interdependencies appear, the model accommodates tasks that demand a nuanced consideration of the relationships between different knowledge sources. For example, dependencies between user memory or persona and document introduce a layer of complexity, reflecting real-world scenarios where the fact of hobby, education, and life experience of the user have a major effect on his or her personal preference over external document knowledge [28].

In order to handle both independent and interdependent knowledge sources, thereby extending its applicability across a spectrum of use cases with varying degrees of complexity, the goal of the planning step is to make a series of decisions to decide whether or not the corresponding source of knowledge is required and determine their call order if needed. Since the dependency relationship is previously known, we only need to make sure that a certain knowledge source is called after the sources it depends on. Thus, we formulate this task as sequence-to-sequence generation by directly outputting required sources in execution order as follows:

$$
M : c \leftarrow K_j, K_i, \ldots, K_n,
$$

Then we strictly follow the outputted order to retrieve evidences from corresponding source of knowledge. To offer the flexibility and scalability to plug in an arbitrary number of sources, we can add $K_i, \ldots, K_n$ and NULL as special tokens into the vocabulary of LLMs as special tokens, and expand the set of tokens on the fly following Haco et al. [49]. Besides that, we add other special tokens to indicate the different parts of the input, i.e., $[SOURCE]$ and $[EOS]$ to indicate the start and end positions of sources. In this way, LLM can model the dependency between different sources and learn when and how to call certain sources.

2) **Retrieval**: According to the output of the plan during the planning step, there are two cases in this step: (1) the response does not need any external sources of knowledge, and the dialogue system can skip this step; and (2) the response needs multiple sources of knowledge, and it strictly follows the output source order to retrieve top-n related evidence $k_j$ for the $i^{th}$ source of knowledge according to the dialogue context $c$. If there is a dependency here, it will use preceding retrieved results $k_j$ in the planned execution order as a filter. Specifically, assuming that the output persona is PERSONA, DOCUMENTS in the planning step for a person-consistent system, then we retrieve $k_j$ from DOCUMENTS according to $c$ and $p'$. If there is no dependency, it simply retrieves corresponding evidences source by source (as shown in Figure 1).

To joint train the retriever and reader in a unified manner, we first reformulate the traditional classification task (relevant or irrelevant) for retrieval into a generation task. Then we define several similarity scores as shown in Table 1, ranging from 0 to 1 with the mean as 0.1. In this way, we can force the UniMS-RAG to predict the similarity score by generating corresponding tokens. To get the supervised label of similarity scores, there are two different ways: (1) the similarity scores are generated by independent trained retriever such as DPR [42]; (2) The details can be found in § IV-B. Furthermore, we design an attention mask mechanism to mask unrelated evidence when predicting the similarity score for current evidence, aiming to reduce unnecessary noise in the context as shown in Figure 2.

$$
M : c, K_j \; \text{is} \; sim \; (e_1, 0.1, 0.2, \ldots, 1.0),
$$

In this way, UniMS-RAG can be used to evaluate the relevance between the dialogue context and retrieved evidences after training, serving as a retriever itself. During the inference, we can use it to retrieve top-n evidences $(e_1, \ldots, e_n)$ corresponding sources of knowledge according to Eq. 3.

3) **Generation**: We concatenated all preceding results, including the names of sources, retrieved evidences and corresponding similarity scores, all together with the dialogue context $c$ to generate the response:

$$
M : \; \text{img} \; \rightarrow \; s_t,
$$
```

### --- Page 0005 ---

```markdown
| Token Type | Input                                   | Output                                                                 | Definitions                                                  |
|------------|-----------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------|
| Sources    | $C \{ \text{NULL}, K_1, K_2, K_3, \ldots, K_n \}$ | Decides which source to retrieve or not retrieve.                      |
| Similarity | $C, e_i \{0.0, 0.1, 0.2, \ldots, 1.0\}$ | $e_i$ is useful for current dialogue context.                         |
| Indicator  | $- \{ [\text{SOURCE}], [\text{EOS}], [\text{EVIDENCE}], [\text{EOE}] \}$ | Indicates the start and end position of different parts.              |

where $inp = C_{t} \{ [\text{SOURCE}]K_1, \ldots, [\text{EOS}] [\text{EVIDENCE}] k'_{t} [\text{EOE}][S_{im_t}] \}$. We use $[\text{EVIDENCE}]$ and $[\text{EOE}]$ to represent the start and end positions of the retrieved evidences. The $[S_{im_t}]$ stands for the similarity score of $i_{th}$ evidence, calculated using the retriever ($SIV-B$).

## B. General Framework

Besides the core model, we also value the general framework such as strategies to collect data, unique training and inference strategies. Thus, we first introduce different ways to collect relevance score labels, and then present the training and inference design (Figure 3).

1. **Relevance Score Acquisition**: There are two different methods to get the similarity labels: 1) prompt-based method, which directly prompt the LLMs to assign the similarity scores given dialogue context and evidences [46, 48]; 2) classification-based method, which requires a fine-tuned retriever such as DPR to calculate the similarity score given the same input. In the latter method, the evidence is fed one by one with the dialogue context.

   **Prompt-based method**: We can utilize some off-the-shelf methods to get the similarity score between the dialogue context and the evidence, including some sparse retriever such as TF-IDF [50] and BM25 [40], or simply hard label by assigning a similarity score to the evidence based on whether it is used (set score as 1) or unused (set score as 0). Inspired by recent studies which simply using LLMs as search engine to predict the similarity relationship between any evidence [48], we choose to prompt the LLMs (i.e., ChatGPT) to predict the similarity score by feeding a dialogue context and several evidences in a zero-shot manner. The prompts are shown in Figure 4.

   **Classification-based method**: Following previous works [42], we can train on the off-the-shelf retriever using the ground-truth labels. Specifically, we first build our own finetuning dataset by regarding context, unrelated_evidence as the positive and (context, unrelated_evidence) as the negatives. The positive and negative samples are annotated in the original dialogue dataset, i.e., which persona or documents is used to generate the response. Then we apply the following negative log likelihood (NLL) loss to learn the retrieval-oriented representations following [51]:

$$
L_{\text{retriever}} = -\log \left( \frac{e^{sim(C_i, d^+)}}{e^{sim(C_i, d^+)} + \sum_{j \in B} e^{sim(C_i, d_j^-)} } \right) \tag{5}
$$

$$
\text{sim}(C_i, d) = \cos(C_i, d) \tag{6}
$$

where $sim$ is a similarity function, $B$ is a mini-batch of examples, $d^+$ and $d^-$ are positive evidence and negative evidence for $i_{th}$ dialogue context $C_i$. Once there are no negatives or the negative is the same as the positive, we directly use randomly sampled sample from other session. By including mini-batch negative candidates in the training data, the model is forced to learn to identify the subtle and key information required for the
```
![General framework to utilize UniMS-RAG, including 1) relevance score acquisition; 2) training stage; and 3) inference stage.](assets/page_0005_img_1.png)
![Instructions for zero-shot retriever used to predict similarity score using off-the-shelf LLMs. The grey and yellow blocks indicate the inputs and outputs of the model.](assets/page_0005_img_2.png)
```

### --- Page 0006 ---

```markdown
# Algorithm 1 Inference with Self-refinement

Require: Dialogue Context $C$, UniMS-RAG model $M$, NLI model $N$, Retriever $R$, Generated Response $r$, Sources of Knowledge $K = \{K_1, K_2, \ldots\}$, Retrieved Evidences $E = \{e_1, e_2, \ldots, e_n\}$, Similarity Scores $S_{ce} = \{s_1, s_2, \ldots, s_n\}$, Update Number $\alpha$

Ensure: Refined Response $r_{new}$.

1: $S_{il} = []$ // store all nics scores  
2: for $j = 1, 2, \ldots, n$ do  
3: $n_{il} \in N(e_j, r)$ // get the nil consistency score  
4: $S_{il}.append(n_{il})$  
5: end for  
6: $S = S_{il} \odot S_{ce}$ // element-wise product inspired by $p(e|c) = p(c|e)p(e)$  
7: $S = sorted(S)$ // sort the above scores  
8: $E_{update} = Find(S, E)$ // find the evidences need to be updated according to the sorted scores  
9: $E = E \setminus E_{update}$ // poll all evidences need to be updated from current one  
10: while $E_{update} \neq \emptyset$ do  
11: Pop a evidence $e_i$ from $E_{update}$, then update $E_{update} \leftarrow \{e_i\}$  
12: $e_{new} = R(C, K)$ // retrieve next novel evidence from corresponding source of knowledge according to the dialogue context  
13: $E = E \cup \{e_{new}\}$  
14: $M = M(C, E)$  
15: end while  
16: return $r_{new}$  

---

| Dataset   | DuLeMon | KBP  |
|-----------|---------|------|
| #Dialogue  | 3011    | 2437 |
| Train/Val/Test | 1943/747/316 | 982/121/1279 |
| #Utterances | 24544   | 4822 |
| Source Types | NULL, User-Per-, Bot-Per- | NULL, Persona, Document |
| Resp w/ source | $> 49\%$ | $> 85\%$ |

## V. Experimental Setup

### A. Research Questions
The empirical analysis targets the following research questions:
- **RQ1:** Can large language models serve as a planner to determine whether or not require knowledge, which source of knowledge to call, and when to call?
- **RQ2:** Can large language models serve as a retriever to retrieve highly related evidence from corresponding sources of knowledge?
- **RQ3:** Can large language models serve as a reader to capture related information in the context and incorporate them into the final response generation?

### B. Datasets
We consider two different situations between different knowledge sources, corresponding to two publicly available personalized dialogue datasets: DuLeMon [7] and KBP [11].
```

### --- Page 0007 ---

```markdown
| dataset statistics are presented in Table II. We adopt the same train/val/test split in the original datasets. |
| --- |
| • DuLeMon [7] is the latest open-domain dialogue dataset with long-person persona memory in which a response is grounded on persona information that occurred in historical sessions, leading to better dialogue engagement. There are two versions of DuLeMon: Self and Both, where the persona information comes from only self side (user side) or both side (both user and chat bot side). We choose Both versions to consider the independent relationship between these two sources of persona information: User-Pe r and Chatbot Side (Bot-Pe r). |
| • Knowledge Behind Persona (a.k.a., KBP) [11] is a dialogue dataset, in which the response is grounded on both the persona (Persona) and its corresponding implicit knowledge (Document). We choose it to consider the interdependent relationship between different sources of knowledge. It is worth noting that both of these two datasets contain cases which do not require any external source of knowledge, denoting as NULL. |

## C. Baselines and Evaluation Metrics

### 1) Knowledge Source Selection: 
We begin by discussing prompting-based and classification-based methods. This includes presenting several baseline models to select different sources of knowledge for comparison, and we adopt F1 as our automatic evaluation metric. 
- **Prompting-based methods.** We mainly utilize zero-shot and in-context learning (ICL) to directly instruct ChatGPT (gpt3.5-turbo-1166) and GPT-4 (gpt-4o) to output the required knowledge sources (a.k.a., acting tokens). We use the same demonstration for fair comparison. 
- **BERT [52]** utilizes BERT as the backbone to train a classifier, which determines the required sources. We formulate this task as a multi-label classification task to determine whether each source is required or not required. If none of them is required, then the used source is NULL. We choose different thresholds for different datasets according to the performance at validation dataset 2. 
- **SAFARI [11]** is the first work to formulate the source planning task as a sequence generation task, regarding different sources of knowledge as different tokens in vocabulary at LLMs. However, it does not consider relevance scores or iterative refinement. We only report its best performance for better comparison. 

### 2) Knowledge Retrieval: 
We compare the performance of retrieval with different types of retrievers, and we choose Recall @1 as our primary evaluation metric. This selection is motivated by the predominant use case scenario observed in the original datasets, where a single evidence from each knowledge source is typically sufficient. 
- **BM25 [40]** is a type of sparse retriever, which takes into account term frequencies and inverse document frequency (TF-IDF) to score the relevance of documents to a query. 

2 In detail, we set the threshold 0.3 in kBP and 0.5 in DuLeMon datasets, respectively. 
- **RocketQAv2** [53] is a type of dense retriever, which is a unified list-wise training approach for both the retriever and re-ranker. 
- **DPR [42]** is a method that leverages dense vector representations for passages in a collection. It uses pre-trained language models to encode passages and queries into dense vectors independently, allowing for a more nuanced understanding of the content. 
- **LLMs.** We utilize the same instruction as shown in Figure 4 to prompt gpt-3.5-turbo-1166 and gpt-4 to retrieve top-t evidence from corresponding source knowledge. Furthermore, we also employ in-context learning (ICL) prompting for more comprehensive comparison 3. 

### 3) Response Generation: 
We mainly compare two methods due to limited work focus on our target problem. Regarding the metrics, we choose BLEU, Rouge-L to evaluate the gap between generated response with the ground truth response in the original datasets. Besides that, we select Persona Consistency (P.C) and Knowledge Consistency (K.C) to evaluate the consistency score using our finetuned NLI models [8] with the same definition in [11]. It is important to note we merge the two sources of persona information (User-Pe r, Bot-Pe r) in DuLeMon into one unified P.C score. 
- **FoCuS [54]** aims to minimize the negative log-likelihood of knowledge prompting and sub-tasks: persona grounding and knowledge grounding. It uses either the sigmoid or softmax function to select the knowledge extracting evidence in dialogue context, and then generate the final response. We do not report the performance on DuLeMon dataset since most of the responses in this dataset do not require external sources. 
- **SAFARI [11]** incorporates all retrieved evidences with corresponding source signals to generate the final responses, without considering the relevance between different evidences and dialogue context explicitly. We report performance of both supervised and unsupervised SAFARI. 

## D. Implementation Details

**UniMS-RAG:** We mainly choose ChatGLM-6B [55, 56] as the backbone models during training, we set the batch size as 8, train models with 3 epochs and save the checkpoint with the lowest validation loss. For other hyper-parameter settings, we mainly follow the corresponding official code 4. Due to the computation limit, we conduct training with LoRA [57] at one single 3090 GPU, and it costs about 4-6 hours. For the prompting using LLMs, we choose gpt-3.5-turbo-1166 and gpt-4o, and set both the temperature and top p as 0.1 to reduce the randomness of LLMs. We choose only the top-ranked result for the final experiment to ensure a consistent setting, and we additionally investigate the effects of retrieving different numbers of results to evaluate whether UniMS-RAG can effectively filter out unrelated evidence. According to different sources of similarity scores, we have three variants: UniMS-RAG (ChatGLM-6B), UniMS-RAG (GPT-4o) and UniMS-RAG 3. 
```

### --- Page 0008 ---

```markdown
# (DPR) where the similarity scores come from ChatGPT, GPT-4 and DPR, respectively. At the inference stage, we use self-predicted similarity scores to retrieve the evidence, serving as an indicator. More variants or analyses can be found in SVIIB-B. 

## Others:
We finetune RocketQAv2 and DPR using samples from corresponding dataset by regarding $(content, used\_persona / document)$ as the positive and $(content, unrelated\_persona / document)$ as the negative. We set epochs as 5 and max sequence length as 512, and mainly follow these codebases* for other parameters. For RocketQAv2, we load the weights of pre-trained model zh_dueruer_de_v2 as introduced in the official homepage, which is trained on the largest Chinese QA dataset, and we use 12-layer bert-base-chinese with 110M parameters as backbone model for DPR. We then finetune an NLI model [B] by regarding $(ground\_persona / document, response)$ as the positive and randomly sampled $(unrelated\_persona / document, response)$ as the negative. We also use bert-base-chinese as the backbone model. We concatenate and encode the ground persona/document $k$ and response $r$ in the form of $[CLS][SEP][SEP]$, and we train the model to predict whether responses are consistent with corresponding personas or documents. The batch size for finetuning is 8. The maximal training epoch is 5, and the maximum sequence length of the encoder is 512. In the experiments, we use the AdamW optimizer with a learning rate of 2e-5 and an initial value of 1e-6. We evaluate the NLI model on the KBP test set every 500 iterations during training, and we save the checkpoint with highest performance on the test set. We finetune the model achieves > 95% accuracy for both datasets.

## VI. EXPERIMENTAL RESULTS
In this section, we present the performance of our proposed method UniMS-RAG at different sub-tasks, including the final end-to-end generation task.

### A. Performance of Planning (RQ1)
There are different types of planning decisions in the different datasets: DuLeMon using NULL, User-Bot-Person and both sources of knowledge (and using NULL, Persona, and Both). Table III demonstrates the F1 of planning using different methods under these two datasets. Generally, for prompting-based methods, we can find that two observations: 1) in-context learning (ICL) prompting can not always achieves better performances compared with zero-shot, regardless the dataset and models; and 2) it is not guaranteed that GPT-4 outperforms ChatGPT in all metrics. In detail, GPT-4 tends to predict more NULL compared with ChatGPT, especially on DuLeMon dataset. On the other side, for the supervised methods, the SAFARI model performed slightly worse than the BERT model on the DuLeMon dataset, but their performance was comparable on the KBP dataset. Furthermore, the UniMS-RAG achieves best performance on 4 out of 7 metrics. Specifically, for KBP, both UniMS-RAG and SAFARI most frequently predict Bot, followed by NULL, and then the Persona case, 

mirroring the frequency of these cases in the original dataset. However, UniMS-RAG predicts significantly more NULL cases than SAFARI. Some of these NULL predictions by UniMS-RAG are actually Persona in the original dataset, leading to a lower number of Persona predictions by UniMS-RAG. In addition, we also found that the original data distribution has a serious impact on the final planning performance. For example, there are less than 0.1% samples in the training set of DuLeMon requiring both source of knowledge, resulting in poor performance of all methods at both. A similar phenomenon is also observed on the KBP dataset. Another reason behind this could be the additional token (i.e., evaluation tokens) introduced during the training stage compared with SAFARI. In general, the planning capability of LLMs still needs to be improved to solve the complex multiple sources planning problem in a dialogue system, particularly when dealing with imbalanced or scarce data sources.

### B. Performance of Retrieval (RQ2)
To investigate RQ2, we examine different types of retrievers, including unsupervised methods (BM25, ChatGPT and GPT-4) and supervised methods (RocketQAv2, DPR, and our proposed UniMS-RAG), in order to evaluate the retrieval performance, providing the ground-truth for all models (except NUL). Table IV presents the Recall@1 (R@1) of the different methods.

1) **Performance of baselines**: There are several observations can be concluded from these results: 1) GPT-4 achieves better performance than ChatGPT no matter which prompting method is chosen (zero-shot or in-context), and the prompting method in in-context learning can not always achieves better performance in all datasets. For example, the in-context learning prompting (ICL) performs worse than zero-shot on DuLeMon but better on KBP. We attribute this to the nature of demonstrations and the complexity of different datasets; 2) The overall performance of dense vector (e.g., RocketQAv2 and DPR) mostly is better than ChatGPT and GPT-4, and the performance of ChatGPT is better than sparse retriever (e.g. BM25). The gap of former is bigger than the latter, indicating the large improvement from current methods using LLMs as retriever, particularly at the context of conversational search; 3) It is observed that DPR performs the best out of these retrievers on KBP datasets while RocketQAv2 performs the best on DuLeMon dataset, revealing the importance of dense retrieval models in this task. We attribute the higher performance of RocketQAv2 on DuLeMon to the similar distribution between DuLeMon dataset and pre-training corpus of RocketQAv2; 4) On KBP datasets, all retrievers perform best at Both-PERSONA and worst at Both-DOCUMENTS, indicating the difficulty to retrieve independent source of knowledge since the semantics between different knowledge from Both-DOCUMENTS are similar to the same underlying persona $p^*$, making them more difficult to distinguish. On the other hand, the performance on DuLeMon is even worse than Both-DOCUMENTS, indicating the issue to be the semantic gap between the dialogue context and used persona in DuLeMon (as shown in Figure 1).
```


### --- Page 0009 ---

```markdown
# PAGE 9

## TABLE III: The F1 of different decisions in Planning of different methods under supervised settings. 1) DuLeMon: There are 1686 NULL, 505 USER, 219 BOT and 6 BOTH in the ground planning; 2) KBP: There are 181 NULL, 125 PERSONA and 923 BOTH in the ground planning.

| Model          | NULL  | User-Peer | Bot-Peer | Both  | NULL  | Persona | Both  |
|----------------|-------|-----------|----------|-------|-------|---------|-------|
| ChatGPT-Zero-1 | 1.29  | 24.18 (953) | 0.00     | 0.69 (1449) | 11.45 (116) | 20.67 (238) | 74.88 (880) |
| ChatGPT-Zero-2 | 1.97  | 34.11 (1075) | 0.00     | 0.66 (297) | 27.95 (299) | 23.14 (238) | 41.89 (292) |
| GPT-4-Zero-1   | 2.39  | 40.24 (1278) | 0.00     | 0.97 (31.41) | 31.42 (345) | 58.84 (491) |
| GPT-4-Zero-2   | 57.59 | 18.03 (893) | 0.00     | 0.45 (28.16) | 31.17 (14) | 15.24 (87) |

## TABLE IV: The performance (Recall@1) of Retrieval of different types of retrievers. It is note the UniMS-RAG (w/ DPR) that is the retriever from the DPR model during training, and for our LLMs, the labels comes from response prompting. 1) DuLeMon: There are 511 samples require to use User-Peer and 225 samples require to use Bot-Peer; 2) KBP: There are 125 samples require to use PERSONA and 923 samples require to use BOTH.

| Model          | User-Peer | PERSONA | Both-PERSONA | Both-DOCUMENTS |
|----------------|-----------|---------|---------------|-----------------|
| BM25           | 23.74     | 54.60   | 63.71         | 19.26           |
| ChatGPT-Zero-1 | 21.42     | 53.68   | 61.83         | 19.19           |
| ChatGPT-Zero-2 | 21.53     | 31.11   | 74.20         | 25.67           |
| GPT-4-Zero-1   | 23.40     | 42.80   | 63.60         | 19.26           |
| GPT-4-Zero-2   | 24.07     | 42.66   | 72.05         | 18.58           |

## TABLE V: The performance of Generation of different methods. We follow the official code of SAFARI to conduct evaluation on DuLeMon dataset. We bold the highest performance and underline the second-best performance.

| Models         | DuLeMon         | KBP            |
|----------------|-----------------|----------------|
| BLEU           | Rougel         | PC             | K.C. |
| Unsupervised   | SAFARI (11.21) | 12.11          | 16.96 |
|                | 16.46          | 4.96           | 13.74 |
| Supervised     | SAFARI (11)    | 17.82          | 51.28 |
|                | 23.76          | 76.99          | 29.32 |
|                | 18.34          | 30.32          | 51.71 |
|                | 19.67          | 31.56          | 54.07 |
|                | 18.30          | 32.03          | 54.63 |
|                | 18.31          | 32.14          | 54.63 |

## 2) The performance of UniMS-RAG: Besides, we also present the performance of UniMS-RAG as a retriever using different similarity signals (DPR, ChatGPT, and GPT-4) during the training. Our main focus is to determine whether UniMS-RAG can be used as a retriever. Additionally, if it can be used as a retriever, we aim to evaluate its performance and potential. The results show that UniMS-RAG can be directly used as a retriever, since the performance of all variants of UniMS-RAG are better than sparse retriever (BM25) and some of them even outperforms some promising baselines. In detail, UniMS-RAG w/ DPR even achieves better performance than ChatGPT, revealing the great potential of UniMS-RAG as retriever once we use more high-quality signals (the performance of DPR is better than ChatGPT). Furthermore, UniMS-RAG w/ ChatGPT achieves comparable performance with ChatGPT on DuLeMon and KBP datasets. Since the performance of UniMS-RAG w/ GPT-4 is much better than UniMS-RAG w/ ChatGPT due to more accurate similarity labels provided by GPT-4, we believe that UniMS-RAG can distill the capabilities of original retriever as much as possible if we can provide more data and more fine-grained signals. Therefore, we can derive the answer to RQ2 from this analysis: Large Language Models (LLMs) can serve as retrievers directly, achieving comparable performance compared with original retrievers, showing a great potential towards a more powerful unified RAG framework.

## C. Performance of Generation (RQ3)

To investigate the performance of UniMS-RAG on the response generation task, we conduct two settings: 1) using itself as retriever, which means we use UniMS-RAG to retrieve corresponding evidences from the planned sources of knowledge (i.e., the output of planning step); and 2) using independent retriever (i.e., BM25, DPR, ChatGPT, and GPT-4) to retrieve corresponding evidences from the planned sources of knowledge (i.e., the output of planning step). Table V demonstrates the performance of response generation under both supervised and unsupervised settings.

1) The performance of baselines: On the one hand, we can find that supervised methods mostly achieve better performance.
```

### --- Page 0010 ---

```markdown
than unsupervised methods except the BLEU-1 and Rouge-L of unsupervised SAFARI on DuLeMon dataset. We carefully check the outputs of unsupervised SAFARI and find that it tends to plan to use source of knowledge (70% using both User-Per and Bot-Per) while most of original test samples do not require any sources of knowledge, resulting in extremely low P.C and higher BLEU-1 and Rouge-L. Furthermore, it is evident that SAFARI outperforms FoCus. We emphasise that FoCus treats knowledge selection as a classification task and optimizes it jointly with response generation tasks, leading to efficiency and scalability issues compared with SAFARI and UniMS-RAG.

2) The performance of UniMS-RAG: Referring to Table III and Table IV, the performance of the planning step and retrieval step largely affects the results in the final generation step. Specifically, when using itself as retriever, we can find that UniMS-RAG using signals from DPR (w/ DPR) leads to better performance in contrast to ChatGPT (w/ ChatGPT) or GPT-4 (w/ GPT-4), revealing the effectiveness of better retrieval signals. The results of GPT-4 is also slightly better than ChatGPT due to more accurate similarity scores no matter using itself as retriever or using independent retriever. The gap between DPR and ChatGPT on DuLeMon is relatively small since most of cases here do not require the involvement of external sources of knowledge. Thus, we decide to load parameters of UniMS-RAG w/ DPR to conduct evaluation when using independent retriever. In detail, the results will validate the effectiveness of better retriever (w/ DPR > w/ GPT-4 > w/ ChatGPT > w/ UniMS). We suspect this is highly related to the original data distribution in the test dataset, since most of the samples do not require external personal information. Furthermore, we also find that using independent retriever is mostly better than using itself as retriever except the worst BM25, which is consistent with the findings in the performance of retrieval step. To conclude, it is obvious that our proposed method UniMS-RAG outperforms all other baselines in at least 5 out 2 evaluation metrics no matter using itself as retriever or using independent retriever. Combining the performance of retrieval, and response generation together, we can find that UniMS-RAG is capable of serving as a planner, retriever, and reader in a unified manner, leading to better performance in personalized dialogues.

3) Discussion of UniMS-RAG Member: Despite the clear performance boost from using independent existing retrievers, we emphasize that the performance gain is not solely due to the better relevance scores provided by these retrievers. It is also significantly attributed to our proposed UniMS-RAG, especially the introduction of relevance score prediction task which makes it can attention on relevant evidences while overlooking noisy ones. This conclusion based on two observations: 1) simply using the existing retriever based on SAFARI cannot achieve state-of-the-art results. We have already selected the best-performing version of SAFARI, and all variants of our proposed UniMS-RAG (i.e., using independent retriever and using itself as retriever) largely outperform it, 2) The performance gap between ours and proposed method is relatively smaller than the gap between ours with baselines, revealing the great potential and flexibility of UniMS-RAG. In practice, we advocate for the

![The performance of Generation with different number of retrieved evidences on two datasets.](assets/page_0010_img_1.png)

unified training of these three sub-tasks in PerDS, as depicted in UniMS-RAG. During inference, the retriever should be selected on a case-by-case basis.

VII. ANALYSIS AND DISCUSSIONS

In this section, we choose the best model as shown in our previous experiments to investigate the performance changes under different settings (UniMS-RAG w/ DPR). In detail, we start from the performance of our proposed model with different numbers of retrieved evidence (§VII-A), then we study the effects of the introduced self-refinement during the inference stage by re-evaluating the relationship between generated response with the selected context or retrieved evidence (§VII-B). We present the ablation study to show the effectiveness and rationale of UniMS-RAG (§VII-B), followed by the results of human evaluation.

A. Different Numbers of Retrieved Results

The number of retrieved results plays a key role in the response generation. Striking a balance between accuracy and recall is essential, as too few results may miss important semantics, while too many can introduce noise. Figure 5 shows the performance of UniMS-RAG under different numbers of retrieved results. In DuLeMon dataset, we observe a slight improvement in performance as the number of retrieved results increases. This improvement is likely attributed to infrequent cases requiring evidence in the original test dataset. On the other hand, in KBP dataset, it is obvious that the performance get notably improvement when the number of retrieved evidence increases from 1 to 2, but then experiences a slight decline upon further increase to 3. We hypothesize that this drop is a result of additional noise introduced as the number of retrieved evidence continues to increase. This suggests a delicate balance must be struck to optimize performance, considering the specific characteristics of each dataset.

B. Additional Self-refinement during Inference

To investigate the effects of self-refinement during the inference, we set the number of retrieved evidences as 3 according to the experimental results in the previous section since it
```

### --- Page 0011 ---

```markdown
TABLE VII: The performance of Generation with Self-refinement during the inference. Specifically, we first report the performance of different number of updated evidences (Within self-refinement); and then we fix the update number $l$ and iteratively refine the generated response step by step (Multi-step Self-refinement).

| Number     | BLEU-1 | Rouge-1 | PC | BLEU-1 | PC | KBP  |
|------------|--------|---------|----|--------|----|------|
| UniMS-RAG  | 18.47  | 20.34   | 63.69 | 36.90 | 79.17 | 53.88 |
| Within Self-refinement | 18.75  | 21.61   | 63.76 | 36.76 | 53.53 | 54.83 |
| $l = 1$    | 18.77  | 21.64   | 65.00 | 37.36 | 37.64 | 54.84 |
| Multiple Self-refinement | 18.51  | 21.75   | 64.54 | 36.53 | 53.25 | 54.83 |
| Step 1     | 18.53  | 21.36   | 60.39 | 37.86 | 79.29 | 54.52 |
| Step 2     | 18.70  | 21.65   | 63.56 | 37.35 | 37.85 | 54.84 |
| Step 3     | 19.30  | 24.66   | 65.34 | 37.95 | 38.03 | 54.84 |

Ablation study on the impact of different steps and modules in UniMS-RAG.

| Model     | DuLeMon | KBP  |
|-----------|---------|------|
| UniMS-RAG | 18.47   | 20.34 |
| - Ground Refinement | 16.79   | 18.53 |
| - Context Refinement | 16.76   | 20.69 |
| - Ground & Context | 17.84   | 21.19 |
| - w/o Attention Mask | 18.34   | 20.90 |

The effects of evidence attention mask. There are two notable advantages of adding the evidence attention mask: 1) removing sources of irrelevant evidence to predict the similarity score; and 2) the UniMS-RAG can efficiently operate as a retriever, as it directly captures the relationship between individual contexts and corresponding evidence, optimizing its retrieval capabilities. We can clearly find that removing the evidence attention mask leads to performance degradation, especially in KBP dataset since there are more cases that require external sources of knowledge.

Compared with simple planning strategies. We additionally conduct experiments by adopting two simple planning strategies: 1) always use all sources, User-Per and Bot-Per in DuLeMon, and both in KBP; and 2) always do not use sources, to investigate the effectiveness of the introduced planning steps. The results show that the performance is decreased by 63.18%-1.23% on DuLeMon and 42.07%-36.21% on KBP respectively when using all sources. This highlights that indiscriminate use of all sources not only significantly hampers performance when the majority of samples do not necessitate external references (DuLeMon), but also results in a decline even when a substantial portion of samples requires the utilization of sources (KBP). Conversely, a parallel trend is observed with abandoning from using external sources. When the majority of samples do not require external references, refraining from their usage leads to notable improvement. However, this approach adversely impacts performance when external sources are essential.

Human Evaluation
Human evaluation is conducted to evaluate the quality of generated response in terms of three metrics: coherence score (Coh.), persona consistency score (Per.Cs for both DuLeMon and KBP), and knowledge consistency score (Know.Cs only for KBP). We randomly sample 100 responses from each dataset with grounding information for each model (we choose UniMS-RAG or DPR or ChatGPT and using itself as retriever) and ask three well-educated annotators* to indicate its coherence given persona (1-5) and knowledge (1-10). Table VIII shows ...
```

### --- Page 0012 ---

```markdown
| Model        | DuLeMon | KBP         | 
|--------------|---------|-------------| 
|              | Perc. (%) | Coh. Perc. (%) | Know.C (%) | 
|--------------|---------|-------------| 
| FoCuS        | 2.84    | 14.0        | 53.3       | 44.2       | 33.8       | 
| SAFAIR       | 1.40    | 4.06        | 68.0       | 59.1       | 
| UniMS-RAG    | 2.92    | 3.72        | 4.18       | 65.2       | 6.03      | 
| w/ ChatGPT   | 3.35    | 23.0        | 4.0        | 66.1      | 54.3      | 

### Number of retrieved evidence - 1
| Model        | DuLeMon | KBP         | 
|--------------|---------|-------------| 
|              | Perc. (%) | Coh. Perc. (%) | Know.C (%) | 
|--------------|---------|-------------| 
| w/ DPR      | 2.38    | 6.0         | 34.8       | 6.7       | 
| w/ ChatGPT   | 3.46    | 61.3        | 4.15       | 83.2      | 61.4      | 

### Number of retrieved evidence - 3
| Model        | DuLeMon | KBP         | 
|--------------|---------|-------------| 
|              | Perc. (%) | Coh. Perc. (%) | Know.C (%) | 
|--------------|---------|-------------| 
| w/ DPR      | 3.38    | 6.0         | 34.8       | 6.7       | 
| w/ ChatGPT   | 3.46    | 61.3        | 4.15       | 83.2      | 61.4      | 

In the final result, when the number of retrieved evidence is one, we observe there is a significant improvement on DuLeMon and a slight improvement on KBP datasets when using UniMS-RAG w/ DPR. Moreover, when the number of retrieved evidence increases, it is obvious that the performance is largely improved. This observation reveals the effectiveness and robustness of UniMS-RAG to denoise unrelated information, and capture relevant evidences, thanks to the introduction of task and unique training strategies. In addition, UniMS-RAG w/ ChatGPT performs slightly worse than UniMS-RAG w/ DPR due to worse similarity signals. We also observe that humans are more likely to find persona-inconsistency cases [11]. There are some responses that have intrinsic or intra-contradictions [58], for example, the dialogue responds: “Yes, our license plate starts with ‘Yun B’?”.

## CONCLUSION
In this paper, we focus on personalized knowledge-grounded dialogue tasks in a multi-source setting and decompose the tasks into two sub-tasks: knowledge source selection, knowledge retrieval and response generation. We discern a notable gap in none of the existing literature concerning the multiple sources planning and auto-regressive retriever with LLMs themselves as backbone. To fill this gap, we propose a Unified Multi-Source Retrieval-Augmented Dialogue System (UniMS-RAG), aiming to build a unified personalized dialogue system with LLM serving as planner, retriever and reader simultaneously. Experimental results on two popular personalized datasets show that the UniMS-RAG framework can generate more personalized and factual responses and establish a better performance with self-refinement during inference, significantly outperforming strong baseline models under both automatic and human evaluations.

## ACKNOWLEDGMENT
This paper was partially supported by grants from the RGC General Research Funding Scheme (GRF) 14229222 (CUHK 21151185). Work done when Hongru Wang is visiting EdinburghNLP.

## REFERENCES
[1] Y. Bang and et al., “A multitask, multimodal evaluation of chatgpt on reasoning, hallucination, and interpretability,” 2023.

[2] M. Alex and et al., “When not to trust language models: Investigating effectiveness of parametric and non-parametric memories,” in ACL, 2023.

[3] X. Boyang and et al., “Improving factual consistency for knowledge-grounded dialogue systems via knowledge enhancement and alignment,” in Findings of EMNLP, 2023.

[4] A. Saeidi and et al., “Lamp: When large language models meet personalization,” 2023.

[5] W. Hongru and et al., “Cue-CoT: Chain-of-thought prompting for responding to in-depth dialogue questions with LLMs,” in Findings of EMNLP, 2023.

[6] A. Asai and et al., “Self-rap: Learning to retrieve, generate, and critique through self-reflection,” 2023.

[7] X. Xinchaod and et al., “Long time no see: open-domain conversation with long-term persona memory,” in Findings of ACL, 2022.

[8] M. Andrea and et al., “Personalizing dialogue agents via meta-learning,” in ACL, 2019.

[9] E. Dinan and et al., “Wizard of wikipedia: Knowledge-powered conversational agents,” 2019.

[10] K. Mojtab and et al., “Internet-augmented dialogue generation,” in ACL, 2022.

[11] W. Hongru and et al., “Large language models as source planner for personalized knowledge-grounded dialogues,” in Findings of ACL: EMNLP 2023, 2023.

[12] R. Panda and et al., “Critical contextual augmentation: Landmarks in dialogue generation,” 2023.

[13] G. Kelvin and et al., “Realm: Retrieval-augmented language model pre-training,” in ICMI, ser. ICMI’20, 2020.

[14] O. Rubin and et al., “Long-range language modeling with self-retrieval,” 2023.

[15] Z. Saizheng and et al., “Personalizing dialogue agents: ‘I have a dog, do you have pets too?’” in ACL, 2018.

[16] L. Qian and et al., “You, me, and the rest: Dialogue generation via mutual persona recognition,” in ACL, 2020.

[17] C. Xu and et al., “COSPLAY,” in SIGIR, 2022.

[18] S. Haoyu and et al., “Bob: BERT over BERT for training persona-based dialogue models from limited personalized data,” in ACL, 2021.

[19] W. Charles and et al., “Leveraging similar users for personalized language modeling with limited data,” in ACL, 2022.

[20] Y. Liu and et al., “Improving personality consistency in conversation by persona extending,” in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022.

[21] C. Liang and et al., “Towards robust personalized dialogue generation via order-insensitive representation regularization,” in Findings of ACL, 2023.

[22] M. B. Prasad and et al., “Like hiking? you probably enjoy nature: Persona-grounded dialog with commonsense expansions,” in EMNLP, 2020.

[23] W. Xing and et al., “KSAM: Infusing multi-source knowledge into dialogue generation via knowledge source aware multi-head decoding,” in Findings of ACL, 2022.

[24] Y. Joona and et al., “Call for customized conversation: Customized conversation grounding persona and knowledge,” 2022.
```

### --- Page 0013 ---

```markdown
| Reference Number | Citation                                                                                      |
|------------------|-----------------------------------------------------------------------------------------------|
| [25]             | W. Sxing and et al., “More is better: Enhancing open-domain dialogue generation via multi-source heterogeneous knowledge,” in EMNLP, 2021. |
| [26]             | ——, “Section-aware commonsense knowledge-grounded dialogue generation with pre-trained language model,” in Proceedings of the 29th International Conference on Computational Linguistics, 2022. |
| [27]             | E. Dinan and et al., “Wizard of wikipedia: Knowledge-powered conversational agents,” 2019.    |
| [28]             | F. Tingchen and et al., “There are a thousand hamlets in a thousand people’s eyes: Enhancing knowledge-grounded dialogue with personal memory,” in ACL 2022, 2022. |
| [29]             | H. Minlie and et al., “Challenges in building intelligent open-domain dialogue systems,” ACM Trans. Inf. Syst., 2020. |
| [30]             | H. Wu and et al., “A survey of the evolution of language model-based dialogue systems,” 2023. |
| [31]             | X. Xinchaob and et al., “Long time no see! open-domain conversation with long-term personal memory,” in Find. Conf. of ACL, 2022. |
| [32]             | Z. Zheng and et al., “Memory-augmented dialogue management for task-oriented dialogue systems,” ACM Trans. Inf. Syst., 2019. |
| [33]             | M. Chuan and et al., “Dukenet: A dual knowledge interaction network for knowledge-grounded conversation,” in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, ser. SIGIR '20, 2020. |
| [34]             | R. Nakano and et al., “Webgpt: Browser-assisted question-answering with human feedback,” 2022. |
| [35]             | H. Wang and et al., “Tpe: Towards better compositional reasoning over conceptual tools with multi-persona collaboration,” 2023. |
| [36]             | S. Kurt and et al., “Retrieval augmentation reduces hallucination in conversational,” in Findings of EMNLP, 2021. |
| [37]             | C. Wang and et al., “Survey on factuality in large language models: Knowledge, retrieval and domain-specificity,” 2023. |
| [38]             | L. Patrick and et al., “Retrieval-augmented generation for knowledge-intensive pip tasks,” in Proceedings of the 34th International Conference on Neural Information Processing Systems, ser. NIPS’20, 2020. |
| [39]             | Y. Gao and et al., “Retrieval-augmented generation for large language models: A survey,” 2024. |
| [40]             | R. Stephen and et al., “The probabilistic relevance framework: Bm25 and beyond,” Foundations and Trends® in Information Retrieval, 2009. |
| [41]             | G. Jiafeng and et al., “Semantic models for the first-stage retrieval: A comprehensive review,” ACM Trans. Inf. Syst., 2022. |
| [42]             | K. Vladimir and et al., “Dense passage retrieval for open-domain question answering,” in EMNLP, 2020. |
| [43]             | L. Hang and et al., “Pseudo relevance feedback with deep language models and dense retrievers: Successes and pitfalls,” ACM Trans. Inf. Syst., 2023. |
| [44]             | B. Sebastian and et al., “An analysis of fusion functions for hybrid retrieval,” ACM Trans. Inf. Syst., 2023. |
| [45]             | Y. Zhu and et al., “Large language models for information retrieval: A survey,” 2023.         |
| [46]             | S. Weiwei and et al., “Is ThatGPT good at search? investigating large language models as re-ranking agents,” in EMNLP, 2023. |
| [47]             | T. Shen and et al., “Large language models are strong zeroshot retrievers,” 2023.            |
| [48]             | X. Ma and et al., “Zero-shot listwise document re-ranking with a large language model,” 2023. |
| [49]             | S. Hao and et al., “Toolknet: Augmenting retrieval models with massive tools to vol embeddings,” 2023. |
| [50]             | K. Sparck Jones, “A statistical interpretation of term specificity and its application in retrieval,” Journal of documentation, 1972. |
| [51]             | Y. Shi and et al., “Few-shot conversational dense retrieval,” in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, ser. SIGIR '21, 2021. |
| [52]             | D. Jacob and et al., “BERT: Pre-training of deep bidirectional transformers for language understanding,” in NAACL-HLT, 2019. |
| [53]             | R. Ruiyang and et al., “RocketQAv2: A joint training method for dense passage retrieval and passage ranking,” in EMNLP, 2021. |
| [54]             | J. Yoonma and et al., “Call for customization: A customizable conversation grounding personal and knowledge,” in AAAI, 2022. |
| [55]             | D. Zhengxiao and et al., “Glm: General language model pretraining with autoregressive blank infilling,” in ACL, 2022. |
| [56]             | A. Zeng and et al., “GLM-130B: An open bilingual pre-trained model,” in The Eleventh International Conference on Learning Representations (ICLR), 2023. |
| [57]             | E. J. Hu and et al., “Lora: Low-rank adaptation of large language models,” 2021.              |
| [58]             | Z. Chujie and et al., “CDConv: A benchmark for contradiction detection in Chinese conversations,” in EMNLP, 2022. |
```


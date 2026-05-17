# ArXiv 2305.14322

### --- Page 0001 ---

```markdown
# RET-LLM: Towards a General Read-Write Memory for Large Language Models

> Note: This concept paper outlines an initial methodology, now evolved and thoroughly evaluated in MemLLM.†

**Ali Modarresi**¹²*  **Ayyoub Imani**¹²*  **Mohsen Fayyaz**³  **Hinrich Schütze**¹²  
¹Center for Information and Language Processing, LMU Munich, Germany  
²Munich Center for Machine Learning, Germany  ³Microsoft, Berlin, Germany  
{modarresi, ayyoob}@cis.lmu.de  

## Abstract

Large language models (LLMs) have significantly advanced the field of natural language processing (NLP) through their extensive parameters and comprehensive data utilization. However, existing LLMs lack a dedicated memory unit, limiting their ability to explicitly store and retrieve knowledge for various tasks. In this paper, we propose RET-LLM a novel framework that equips LLMs with a general write-read memory unit, allowing them to extract, store, and recall knowledge from the text as needed for task performance. Inspired by Davidsonian semantics theory, we extract and save knowledge in the form of triplets. The memory unit is designed to be scalable, aggregatable, updatable, and interpretable. Through qualitative evaluations, we demonstrate the superiority of our proposed framework over baseline approaches in question answering tasks. Moreover, our framework exhibits robust performance in handling temporal-based question answering tasks, showcasing its ability to effectively manage time-dependent information.

## 1 Introduction

Large language models (LLMs) have significantly advanced the field of natural language processing (NLP) in recent years (Bubeck et al., 2023; Chowdhery et al., 2022; Touvron et al., 2023). With their vast parameter count and access to extensive data, LLMs have demonstrated remarkable accuracy across various tasks. However, current state-of-the-art LLMs lack a dedicated memory unit. Instead, they are trained to predict words based on context, encoding knowledge implicitly in their parameters, which differs from the ideal memory function.

An ideal memory unit should possess certain characteristics. Firstly, it should allow for read and write operations, enabling the language model to interact with stored knowledge. Scalability is also crucial, as the memory unit should accommodate the consistently evolving nature of knowledge. Furthermore, the memory unit should not be limited to textual documents alone; it should be capable of acquiring knowledge from diverse sources such as database systems. Interpretability is desired, granting insight into the specific knowledge required by the LLM to solve a given task. Lastly, the information stored in the memory unit should be aggregatable, enabling the model to combine related information across multiple documents. For instance, an LLM should be able to list all cities of a country mentioned in multiple documents.

Previous attempts to incorporate memory into LLMs have fallen short in capturing the complete range of memory characteristics. For example, (Zhong et al., 2022; Wu et al., 2022) and (Cheng et al.) degrade the memory as the ability to retrieve relevant documents for a given query context, and adding them to the context when generating answers. Park et al. (2023) merely stores and retrieves previous observations and reflections of a generative agent in a simulated environment.

To address these limitations, we introduce RET-LLM (Retentive LLM) a solution that endows 

![Figure 1: An overview of RET-LLM. A user could prompt with (A): an informative sentence and our approach stores potent information from it inside the memory or (B): a question where previously saved information should be utilized to generate a valid answer.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
LLMs with a scalable, updatable, interpretable, and aggregatable memory module. Our proposal involves equipping language models with a memory module, which allows them to extract knowledge from text and save it for future reference. When faced with a task, the LLM can query the memory module for additional information to support its response. The memory module supports updates and can incorporate information from non-textual sources such as SQL and no-SQL databases and spreadsheets. Furthermore, it enables aggregation of various pieces of information related to a particular concept scattered in a huge document or within multiple documents.

![Figure 1 shows the architecture of RET-LLM. It comprises three components: an LLM, a controller, and a memory unit. We employ Alpaca Taori et al. (2023), a recently released instruction-tuned language model (LLM), and design a fine-tuning process to enable it to acquire the following abilities: information extraction, information lookup, and fact-based answer generation.](assets/page_0002_img_1.png)

Information extraction entails the identification and extraction of triplets in the form of <concept1, relationship, concept2> from informative sentences. The information lookup task involves querying the memory unit to acquire additional information concerning a given concept and its associated relationships when confronted with tasks necessitating further information. Lastly, fact-based answer generation involves generating a final answer based on the retrieved information. The triplet-based storage approach draws inspiration from the theoretical framework of Davidsonian semantics (Davidson, 1967), which provides a foundation for representing concepts described in sentences using a triplet-like structure of <event, subject, object>.

The memory module stores the triplets and their vector representations. During retrieval, it first searches for an exact match of the query text and resorts to a fuzzy search based on vector representations if no exact match is found. For efficient fuzzy search and retrieval, we employ LSH-based hashing of vector representations. The controller acts as an interface, automating interactions between users, the LLM, and the memory module, ensuring a seamless interaction experience with an intelligent chat system.

Our proposed approach offers several advantages over previous methods. It enables LLMs to explicitly store and retrieve knowledge, which is crucial for real-world NLP applications. By incorporating explicit knowledge storage and retrieval, we gain better understanding of the workings of these models and the knowledge they rely on to solve tasks. The use of an external memory unit separate from the LLM ensures scalability and easy modification of stored information. The fuzzy search technique enables efficient retrieval of relevant information, even in the absence of exact matches. Storing information in triplets facilitates the generation of precise and comprehensive solutions, particularly when data aggregation is necessary. Lastly, the memory module allows for easy incorporation of information from diverse sources and accommodates changing facts over time.

Over a qualitative evaluation using question answering examples, we demonstrate cases where a comparable LLM such as Alpaca-7B fails to return a correct answer. We show that this shortcoming occurs while the model has access to all the information required for generating a valid answer. However, in our proposed approach after storing the extractable knowledge from the context, the RET-LLM shows its capability in answering a question without the need of reinputting the context. We also demonstrate that RET-LLM could handle temporal based QA examples. Since it is equipped with a modifiable memory which could handle temporal facts.

## 2 Related Works

Prior works in the field have explored incorporating relevant context into large language models (LLMs) by retrieving and adding relevant documents to the task’s context. Zhong et al. (2022) propose training LLMs with memory augmentation by introducing trainable memory units that are optimized during the training process. Wu et al. (2022) presents the Memorizing Transformer, which can attend to longer documents during inference. This approach stores (Key, Value) pairs, extracted from a transformer layer, in a memory and retrieves relevant pairs to add them to the current context during generation. (Cheng et al.) encode each documents, save them, and retrieve relevant documents based on the current context. In contrast to these approaches, our method offers improved scalability as we do not modify the architecture of the LLM. Instead, we suggest extracting and saving information from documents, allowing for the aggregation of extracted information from multiple sources.
```

### --- Page 0003 ---

```markdown
This enables us to provide more relevant and concise retrieved information that is closely aligned with the specific question being addressed.

Park et al. (2023) utilizes an LLM within a generative agent framework to facilitate the storage and dynamic retrieval of a comprehensive record of the agent’s experiences using natural language. However, there exists a fundamental distinction between their architecture and ours. In Park’s framework, the memory component is an inherent part of the agent itself, while the LLM serves as an external tool employed solely for planning the agent’s behaviors. Consequently, the LLM lacks control over the specific content to be stored and retrieved within the agent’s memory.

Dhingra et al. (2022) contribute to the field by curating a dataset specifically designed to differentiate between temporal and non-temporal facts. They propose training language models on temporally annotated data to enhance their temporal awareness. This work aligns with our research focus on addressing temporal information challenges. However, in our proposed solution, we address these challenges by introducing an updatable memory module.

Schick et al. (2023) present a methodology that empowers LLMs to leverage external tools by generating API calls to access additional functionalities, such as using a calculator for task execution. Our work shares similarities with their approach in terms of teaching the LLM to utilize an external tool. However, it should be noted that our focus lies in incorporating a more intricate and influential tool, namely the memory module, which has the potential to significantly impact the LLM’s output.

## 3 Approach

We aim to design a RET-LLM where the user can perform two actions: (1) Provide one or a series of informative statements where the RET-LLM should be able to memorize the containing information. Previous methods perform this task by either training/fine-tuning the LLM over the provided document or creating a vector representation for the document and storing the representation. (2): Asking related questions which the RET-LLM would answer based on the stored memory. All these actions should function in a seamless setting where the user should only interact in natural language.

Our RET-LLM is constituted by three main components: (1) Controller, (2): Fine-tuned LLM & (3): Memory. As shown in Figure 1, the controller moderates the flow of information between the user, the LLM and the memory. The LLM acts as a processing unit, where it receives texts passed by the controller and figures where it needs to invoke a memory call or not. Since the LLM operates with text, inspired by Schick et al. (2023), we standardized the memory calls by implementing a text-based API schema. Therefore the LLM could generate memory API calls and the controller could apply the LLM API calls to the memory. In our setting, the memory stores data in triplets by using a three-columned table. This is based on the theoretical framework of Davidsonian semantics (Davidson, 1967), where concepts described in sentences could be stored in a structure of <first argument, relation, second argument>.

In the following we describe RET-LLM in more detail. The memory-API, how we fine-tune the LLM to become capable of these calls and the memory structure.

### 3.1 Memory Structure

Each triplet defines a relationship between two arguments with the following format: ⟨t₁, t₂, t₃⟩ where t₁ is the first argument, t₂ is the relation and t₃ is the second argument in the relationship. For instance in the sentence: “Mark Zuckerberg is the CEO of Meta Inc.” the informative triplet that could be extracted is: (Mark Zuckerberg, CEO, Meta Inc.).

To store these triplets we use a three-columned table where each column is associated with each part of the triplet. Alongside saving the texts, we store the average representations so that the memory could also handle queries which have semantically similar words. If the memory module fails to find the exact text in the table, it checks for similar texts by comparing the vector representation of the query text with vector representations of text pieces already stored in the dataset. Therefore for every tᵢ the mean representation retrieved by the LLM (hₐᵛg(tᵢ)) is stored in a Locality-Sensitive Hashing (LSH) table. The reason of utilizing LSH is to reduce the computation required for finding similar representations. Without a hash table for a given query representation, the distances to all of the stored representations should be computed which would be a computationally-expensive task.
```


### --- Page 0004 ---

```markdown
![Memory-Write and Memory-Read scenarios visualized](assets/page_0004_img_1.png)

## Handling Memory Queries
In a memory query, one or two of the triplet parameters should be provided as input:

$$
Q \in \{ \langle q_1 \rangle, \langle q_2 \rangle, \langle q_3 \rangle, \langle q_1, q_2 \rangle, \langle q_1, q_3 \rangle, \langle q_2, q_3 \rangle \}
$$

Where $q_i$ is the search term for the $i$-th parameter in the stored tuples. Before retrieving the query results, each search term is checked for a given $Q$, first the memory checks whether the search terms ($q_i$) have an exact match in the storage table. If $q_i$ does not exist in the stored terms, we use its average representation $h_{AVG}(q_i)$ and the LSH table for an alternative term $\tilde{q_i}$ that has an exact match in our memory table. Possibly, the LSH table may not find an alternative term for the given representation, therefore the query would not have a result: $Q \rightarrow \emptyset$. In any case (exact match or similar match), the query might have multiple matches in the data table ($q_i = t_i$). In this case all resulting triplets would be returned as the query output.

## 3.2 Memory-API & Dataflow
To enable communication between the memory and the LLM, we design an API schema for memory read and write functions. This API allows the controller to understand when the LLM is calling the memory and what parameters should be passed. Based on the triplets discussed in the previous section, the two memory calls are as the following:

- **MEM_WRITE{$\{t_1,t_2,t_3\}$}:** This structure is for storing a triplet $\{t_1, t_2, t_3\}$. Depending on the prompt, multiple write calls could be sequentially generated by the LLM to store multiple triplets extracted from a text.

- **MEM_READ{$\_ \rightarrow \_ $}:** In a memory read, as shown in the API, there are three placeholders that based on $Q$ at least one of them should be filled with the search terms. Based on the query results from the memory, one or a list of triplets could be returned as shown in the highlighted segment.
```

### --- Page 0005 ---

```markdown
![Figure 2 demonstrates how RET-LLM operates using the memory-API.](assets/page_0005_img_1.png)

Figure 2 demonstrates how RET-LLM operates using the memory-API. Depending on the input given by the user, RET-LLM either have to read or write information from or to the memory. If the user prompt an informative statement (or ideally a full document), it would be memory write scenario. On the other hand, by having a question in the input, we consider this to be a memory read case. In both cases the user input is the first input to RET-LLM that is passed on to the LLM.

Based on the given input the LLM infers and generates the relevant API call. With a memory triplet-specific question, after the API call is generated the controller detects it and invoke a memory storation function with the given parameters. The memory receives the data in a triplet format and stores it for future usage. If a memory read call is generated by the LLM, the controller also detects it and pauses the model’s sequence generation for the memory retrieval. It uses the parameters given inside the read call as the query terms and passes them to the memory. The memory lists all stored triplets that feature the given search terms (or a semantically similar version of them according to §3.1) and return the results back to the controller. Using the API discussed in the beginning of this section, the read results are listed after the call so that the LLM could use them to produce a naturally sounded answer. After the answer is produced it is returned back to the user.

As the controller is in between of the user and the LLM, it could hide the whole memory-API schema. This would make the user feel an end-to-end simple language modeling experience without knowing the memory functionality behind the scene.

### 3.3 Finetuning the LLM

In this part we discuss how the LLM is finetuned to be capable of generating memory-API calls. In the end the LLM should be capable of detecting which type of memory call (read or write) it should provoke based on the input. As stated in Section 3.2, the LLM’s input may have one of the two previously discussed structures depending on the memory function. Therefore the LLM should be able to generate and handle this API to store or read the relevant information. To this end, we develop a synthetic dataset to train the LLM. The synthetic task is to learn the relationships of the discussed people with the respective corporations. Based on the stored information, RET-LLM should be capable of answering any questions regarding the people, the corporations or the relationships.

We use a set of firstname and lastnames to generate a synthetic population, called $P$. Each person from this population $per \in P$ could have only one relationship from the following list: $rel \in R = \{employment, manager, investor, founder, customer\}$ with an organization $org \in O$. Where $O$ is a set of corporation names. Hence, each triplet would be as: $(per, rel, org)$. For instance: $(Dominick Alphonso, employment, BMW)$.1 Based on this triplet we can build three triplet-specific questions:
- $Q = \langle per \rangle$, e.g. “Who is Dominick Alphonso?”
- $Q = \langle per, rel \rangle$, e.g. “How Dominick Alphonso is related to BMW?”
- $Q = \langle per, rel \rangle$, e.g. “Dominick Alphonso is employed by which company?”

and the answer to all above should be “Dominick Alphonso is employed by BMW.” Alongside these questions three other types of questions could be asked that could be relevant to multiple triplets:
- $Q = \langle rel \rangle$, e.g. “Who are the employees?”
- $Q = \langle org \rangle$, e.g. “Who are related to BMW?”
- $Q = \langle rel, org \rangle$, e.g. “Who are employed by BMW?”

Unlike the first three, each of these questions could have multiple persons related to the answer. For each of these questions we expect the model answer the questions without any extra information (e.g. stating the corporation of employment when its not asked). To create a training data instance from these questions based on the memory-API, we use the templates stated in Table 1. During finetuning the Question, API query (with the MEM_READ command), API Response and the answer are concatenated as the data input for the LLM. However, the language modeling loss is only applied to the API query and Answer sections. Since these two segments are the text sequences that the LLM is expected to generate based on the other two segments (Question & API Response) that are provided by the controller.

1 Even though the corporation names are real, the people names are entirely random generated. No identification with actual persons is intended or should be inferred.
```

### --- Page 0006 ---

```markdown
# Page 0006

## Table 1: Memory read data examples for finetuning.

| Query Type | Question                          | API Query                       | API Response                   | Answer                     |
|------------|-----------------------------------|---------------------------------|-------------------------------|----------------------------|
| (per)      | Who is per?                       | `<per>:`                        | `<per-rel-org>`              | per is rel to org.         |
| (per, org) | How per is related to org?       | `<per>:<org>`                  | `<per-rel-org>`              | per is rel to org.         |
| (per, rel) | Who are related to org?          | `<org>:`                       | `<per-rel-org>`              | [per, per2, ...] is/are related to org. |
| (rel)      | Who are the rel?                 | `<rel>:`                       | `<per-rel-org>`              | [per, per2, ...] is/are rel to org. |
| (org, rel) | Who are rel org?                 | `<rel-org>:`                   | `<per-rel-org>`              | [per, per2, ...] is/are rel to org. |

## Table 2: Memory write data example structure for finetuning.

| Triplets                     | Statement                     | API Write Calls                                   |
|------------------------------|-------------------------------|--------------------------------------------------|
| `[(per, rel, org), (per2, rel, org), ...]` | `[per1, per2, ...] is/are rel to org.` | `[MEM_WRITE(per1,org)]MEM_WRITE(per2,org)]...` |

As we also need informative examples where have MEM_WRITE calls, we use a similar strategy by using the population, organizations and relations that were previously defined $(P, Q, R)$. Based on the memory-API, in a memory write scenario the RET-LLM receives a sentence which here contains a relationship information and then the LLM should generate the corresponding memory write calls. In our dataset we opted to build examples where it states about multiple people whom have the same relationship with the same company: $(per1, rel, org)$. The template for the memory write data examples are shown in Table 2. Similar to the question-based examples, the statement and the API call are concatenated to form the full input sequence. Also the loss function is applied only to the API segment, since the first part is provided by the controller.

We opted to use the instruction-following Alpaca-7B model (Taori et al., 2023) as a base model for our finetuning. To execute the training in a resource limited setup, we use low-rank adaptation (LoRA) (Hu et al., 2022). 2 This parameter efficient measure allows us to finetune the base model on a single A6000 48GB GPU.

## 4 Qualitative Results

In this part, we present the internal process and final output on multiple evaluation examples. These examples were generated with the same procedure stated in §3.3. First to demonstrate the importance of our approach, we provide the same example to our base model (Alpaca-7B) in a zero-shot setting. The input would be a short instruction for the task, the informative sentences from the example and in the end is the question. As shown in Figure 3, the zero-shot result from the instruction tuned model is clearly incorrect. While the model does have all the information in its context, its still produces an incorrect response.

In the same example, the RET-LLM first stores the extracted triplets from the examples into the memory. After storing the extracted relationships, the RET-LLM could respond to the same question even without having the information in the input. With the help of the memory-API and the memory itself, the relevant triplet is found. The LLM manages to answer correctly after appending the query result to the memory call.

One potential use cases of our approach is in answering questions that have a temporal context. For example, the presidency of the United States undergoes a change every 4 to 8 years. A normal PLM model answers the question about the presidency based on its own training data. While model retraining or parameter editing has its own challenges, our approach could provide an easy and interpretable solution for this issue (Figure 4).

2 The code for finetuning a llama-based model using LoRA is available at: github.com/tloen/alpaca-lora

## 5 Conclusion & Future Work

In this work, we introduced a RET-LLM capable of storing information and retrieving it in further use. With a triplet based memory structure, information are stored in relationships between two arguments with a known relation. The memory could be utilized via a memory-API which is generated.
```

### --- Page 0007 ---

```markdown
![Evaluation Example #1 (Zero-Shot Setting - Alpaca-7B)](assets/page_0007_img_1.png)

| **Evaluation Example #1:** |  |
|-----------------------------|--|
| **Instruction**             | You will be presented with one or a series of sentences about some people and their relationship with a company. After that, for any given question you should be capable of answering that based on the previous sentences. |
| **Relationship sentences**   | Cyrus Alfred, Tia Batres, and Pasquale Baliff are customers of Pfizer. Dorothea Altemus is employed by Pfizer. |
| **Question and prompt for the answer** | Question: Who are employed by Pfizer? |
| **LLM output to the User:** | Cyrus Alfred, Tia Batres, and Dorothea Altemus. ❌ |

---

![Evaluation Example #1](assets/page_0007_img_2.png)

| **LLM Response to Controller:** |  |
|----------------------------------|--|
| **LLM_WRITE**                    | [Cyrus Alfred>>customer of>>Pfizer][MEM_WRITE(Tia Batres>>customer of>>Pfizer)][MEM_WRITE(Pasquale Baliff>>customer of>>Pfizer)] |
| **(Cyrus Alfred>>customer of>>Pfizer)** |  |
| **(Tia Batres>>customer of>>Pfizer)** |  |
| **(Pasquale Baliff>>customer of>>Pfizer)** |  |

---

![Evaluation Example #1](assets/page_0007_img_3.png)

| **Dorothea Altemus is employed by Pfizer.** |  |
|-----------------------------------------------|--|
| **LLM Response to Controller:**               | [MEM_WRITE(Dorothea Altemus>>employed by>>Pfizer)] |
| **(Dorothea Altemus>>employed by>>Pfizer)**  |  |

---

| **Who are employed by Pfizer?** |  |
|----------------------------------|--|
| **LLM Response to Controller:**  | [MEM_READ(>>employed by>>Pfizer):] |
| **LLM final output to the User:** | Dorothea Altemus is employed by Pfizer. ✅ |

---

![Figure 3: An example that has an incorrect result in a zero-shot setting and a correct one in our approach.](assets/page_0007_img_4.png)

---

| **QA Example (Alpaca-7B):** |  |
|------------------------------|--|
| **Question:**                | Who is the president of the United States? |
| **Answer:**                  | LLM output to the User: Barack Obama. ❌ |

---

![QA Example:](assets/page_0007_img_5.png)

| **QA Example:**              |  |
|------------------------------|--|
| **Who is the president of the United States?** |  |
| **LLM Response to Controller:** | [MEM_READ(>>president of>>United States):] |
| **LLM final output to the User:** | Joe Biden is the president of the United States. ✅ |

---

![Figure 4: Asking a question which requires temporal context usually leads to an outdated answer as shown here with Alpaca.](assets/page_0007_img_6.png)
```

### --- Page 0008 ---

```markdown
by a finetuned LLM. Using a controller, all components could communicate with each other and the user would interact with the controller being unaware of the behind process. We have shown that the LLM generates the proper API calls in some question answering examples without having the information in its input context. As this work is still under development, in our next revision we will add a more in-detail empirical evaluation, preferably on a real dataset. We also seek to improve our finetuning method to a more generalized setting so that it could be capable of working with more types of informative relations.

## References

| Author(s)                                                                 | Title                                                                                                   | Source                                                                                          |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Sébastien Bubeck, Varun Chandrasekaran, Ronen El-                        | Sparks of artificial general intelligence: Early experiments with gpt-4.                               | arXiv preprint arXiv:2303.12712.                                                                |
| Xin Cheng, Yankai Lin, Dongyan Zhao, and Rui Yan                        | Knowledge model with plug-in knowledge memory.                                                         |                                                                                                 |
| Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,                        |                                                                                                         |                                                                                                 |
| Maarten Bosma, Gaurav Mishra, Adam Roberts,                               |                                                                                                         |                                                                                                 |
| Paul Barham, Hyung Won Chung, Charles Sutton,                             |                                                                                                         |                                                                                                 |
| Sebastian Gehring, Parker Schuch, Kenan Shi,                             |                                                                                                         |                                                                                                 |
| Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao,                       |                                                                                                         |                                                                                                 |
| Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran,             |                                                                                                         |                                                                                                 |
| Emily Reif, Nan Du, Ben Hutchinson,                                     |                                                                                                         |                                                                                                 |
| Reiner Poge, James Bradbury, Jacob Austin,                               |                                                                                                         |                                                                                                 |
| Michael Isard, Guy Gur-Ar, Pengcheng Yin,                               |                                                                                                         |                                                                                                 |
| Tyo Duke, Anselm Levskaya, Sanjay Ghemawat,                             |                                                                                                         |                                                                                                 |
| Sunipa Dev, Henryk Michalewski, Xavier Garcia,                          |                                                                                                         |                                                                                                 |
| Vedant Misra, Kevin Robinson, Liam Fedus, Denny                          |                                                                                                         |                                                                                                 |
| Zhou, Daphne Ippolito, David Luan, Hyeontak Lim,                        |                                                                                                         |                                                                                                 |
| Barrett Zoph, Alexander Spiridonov, Ryan Sepassi,                       |                                                                                                         |                                                                                                 |
| David Dohan, Shivani Agrawal, Mark Omernick,                            |                                                                                                         |                                                                                                 |
| Andrew D. Pai, Thanuamalayan Sankaranarayana Pilai,                      |                                                                                                         |                                                                                                 |
| Marie Peltat, Aitor Lewkowye, Erica Moreira,                            |                                                                                                         |                                                                                                 |
| Rewon Child, Oleksandr Polozov, Katherine Lee,                          |                                                                                                         |                                                                                                 |
| Zongwei Zhou, Yuezhi Wang, Brennan Saeta, Mark                          |                                                                                                         |                                                                                                 |
| Diaz, Orhan Firat, Michelle Catasta, Jason Wei,                         |                                                                                                         |                                                                                                 |
| Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slavy Petrov,            |                                                                                                         |                                                                                                 |
| and Noah Fiedler.                                                        | Palm: Scaling language modeling with pathways.                                                         |                                                                                                 |
| Donald Davidson.                                                          | The logical form of action sentences, reprinted in d. davidson (1980) essays on actions and events.  |                                                                                                 |
| Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eissenstein,              | Time-aware language models as temporal knowledge bases.                                               | Transactions of the Association for Computational Linguistics, 10:257–273.                     |
| Edward J Hu, yuelong shen, Phillip Wallis, Zeyuan Allen-Zhu,           | LoRA: Low-rank adaptation of large language models.                                                   | In International Conference on Learning Representations.                                       |
| Ali Modarressi, Abdullatif Köksal, Ayyoub Imani,                        | Mem-llm: Finetuning llms to use an explicit read-write memory.                                        | arXiv preprint arXiv:2404.11672.                                                               |
| Joon Sung Park, Joseph P O’Brien, Carrie J Cai,                         | Generative agents: Interactive simulacra of human behavior.                                           | arXiv preprint arXiv:2304.03442.                                                               |
| Timo Schick, Jane Dwiwedi-Yu, Roberto Desi, Roberta                     | Toolformer: Language models can teach themselves to use tools.                                        | arXiv preprint arXiv:2302.04761.                                                               |
| Rohan Tori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois,                | An instruction-following llama model.                                                                  | https://github.com/tatsu-lab/stanford_alpaca.                                                 |
| Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier                    | Llama: Open foundation language models.                                                                | arXiv preprint arXiv:2203.13950.                                                               |
| Yuhai Wu, Markus N Rabe, DeLesley Hutchins,                             | Memorizing transformers.                                                                                | arXiv preprint arXiv:2203.08913.                                                                |
| Zexuan Zhong, Tao Lei, and Danqi Chen.                                   | Training language models with memory augmentation.                                                    | In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5657–5673, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. |

# A Extra Evaluation Example
```

### --- Page 0009 ---

```markdown
# Evaluation Example #2 (Zero-Shot Setting - Alpaca-7B):

## Instruction
You will be presented with one or a series of sentences about some people and their relationship with a company. After that, for any given question you should be capable of answering that based on the previous sentences.

Mozella Baima and Modesto Baichan are employees of ExxonMobil. Maryjane Bachand and William Beasmore are employees of BMW. William Bank is a customer of BMW.  
**Question:** Who are related to BMW?  
**Answer:**  
LLM output to the User:  
Mozella Baima, Modesto Baichan, Maryjane Bachand, and William Beasmore are all employees of BMW. ❌

## Evaluation Example #2:

### LLM Response to Controller:
```
[MEM_WRITE(Mozella Baima->employed by->ExxonMobil)]|[MEM_WRITE(Modesto Baichan->employed by->ExxonMobil)]
```
```
{Mozella Baima->employed by->ExxonMobil}
{Modesto Baichan->employed by->ExxonMobil}
```

### LLM Response to Controller:
```
[MEM_WRITE(Maryjane Bachand->employed by->BMW)]|[MEM_WRITE(William Beasmore->employed by->BMW)]
```
```
{Maryjane Bachand->employed by->BMW}
{William Beasmore->employed by->BMW}
```

### LLM Response to Controller:
```
[MEM_WRITE(William Bank->customer of->BMW)]
```
```
{William Bank->customer of->BMW}
```

### LLM Response to Controller:
```
[MEM_READ(>>>BMW)]
```

### LLM Full Input (User Prompt + MEM_READ Call + Memory result passed by the Controller)
```
who are related to BMW? [MEM_READ(>>>BMW)]:{Maryjane Bachand->employed by->BMW};{William Beasmore->employed by->BMW};{William Bank->customer of->BMW}
```

### LLM output to the User:
William Beasmore, Maryjane Bachand, and William Bank are related to BMW. ✅

![Another evaluation example that has an incorrect result in a zero-shot setting and a correct one in our approach.](assets/page_0009_img_1.png)
```


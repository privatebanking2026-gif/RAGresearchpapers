# ArXiv 2411.02959v1

### --- Page 0001 ---

```markdown
# HtmlIRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems

**Jiejun Tan\***  
Gaoling School of Artificial Intelligence  
Renmin University of China  
Beijing, China  
ztanji@ruc.edu.cn  

**Zhicheng Dou\†**  
Gaoling School of Artificial Intelligence  
Renmin University of China  
Beijing, China  
dou@ruc.edu.cn  

**Wen Wang**  
Baichuan Intelligent Technology  
Beijing, China  
wangwen@baichuan-inc.com  

**Weipeng Chen**  
Baichuan Intelligent Technology  
Beijing, China  
chenweipeng@baichuan-inc.com  

**Ji-Rong Wen**  
Gaoling School of Artificial Intelligence  
Renmin University of China  
Beijing, China  

---

## Abstract
Retrieval-Augmented Generation (RAG) has been shown to improve knowledge capabilities and alleviate the hallucination problem of LLMs. The Web is a major source of external knowledge used in RAG systems, and many commercial systems such as ChatGPT and Perplexity have used Web search engines as their major retrieval systems. Typically, such RAG systems retrieve search results, download HTML sources of the results, and then extract plain texts from the HTML sources. Plain text documents or chunks are fed into LLMs to augment the generation. However, much of the structural and semantic information inherent in HTML, such as headings and table structures, is lost during this plain-text-based RAG process. To alleviate this problem, we propose HtmlIRAG, which is inspired by the use of HTML in modeling knowledge in RAG. We believe HTML is better than plain text in modeling knowledge in external documents, and most LLMs possess the capabilities to understand HTML. However, utilizing HTML presents new challenges. HTML contains additional content such as tags, JavaScript, and CSS specifications, which can divert input tokens and noise in the RAG system. To address this issue, we propose HTML cleaning, compression, and pruning strategies to shorten the HTML while minimizing the loss of information. Specifically, we design a two-step block-tree-based pruning method that prunes useless HTML blocks and keeps only the relevant part of the HTML. Experiments on six QA datasets confirm the superiority of using HTML in RAG systems ¹.

*This work was done when Jiejun Tan was doing internship at Baichuan Intelligent Technology.  
†Corresponding author.  
1Code and datasets are available at https://github.com/plagron/HtmlIRAG.  

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the authors must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.  
Conference acronym: XXX, June 6–05, 2025, Woodside, NY  
© 2020 Copyright held by the owner/author(s). Publication rights licensed to ACM.  
ACM ISBN 978-1-4503-XXXX-X/20/06.  
https://doi.org/XXXXXXXXXXXXXX  

---

## CCS Concepts
- Information systems → Web search engines.

## Keywords
HTML, Retrieval-Augmented Generation, Large Language Model

## ACM Reference Format:
Jiejun Tan, Zhicheng Dou, Wen Wang, Weipeng Chen, and Ji-Rong Wen. 2024. HtmlIRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems. In Proceedings of WebConf 2025 (Conference acronym: XXX). ACM, New York, NY, USA, 14 pages. https://doi.org/XXXXXXXXXXXXXX  

---

## 1 Introduction
Large Language Models (LLMs) have been proven to have powerful capabilities in various natural language processing tasks [42, 44, 46]. However, at the same time, LLMs show deficiencies such as forgetting long-tailed knowledge [28], offering outdated knowledge [39], and hallucination [38, 39, 74]. Retrieval-augmented generation (RAG) utilizes a retrieval system to fetch external knowledge and augment the LLM. It has proved effective in mitigating hallucinations of LLMs [41, 76]. Many RAG systems, such as Perplexity [47] and SearchGPT [43], have been developed, and they commonly use web search engines as the underlying retrieval systems.

Traditional RAG pipelines typically use plain text as the format for retrieved knowledge [21, 63]. HTML documents from the Web are often converted into plain text and concatenated with the user’s query before being fed into the LLM. We found that converting HTML to plain text leads to the loss of structural and semantic information. Figure 1 illustrates that a web page containing tabular forms becomes disordered when converted to plain text. Even worse, original HTML tags, such as “<code>” and “<a>”, denoting important information, are discarded during conversion. Thus, in this paper, we wanted to investigate an intuitive idea: Can we take HTML as the format of external knowledge in RAG systems to preserve the information in HTML documents to a larger extent?
```

### --- Page 0002 ---

```markdown
# Conference acronym 'XX', June 03–05, 2023, Woodstock, NY

![Information loss in HTML to plain text conversion.](assets/page_0002_img_1.png)

documents [6, 15, 17], which means that they inherently possess the ability to understand HTML without requiring further training [26, 73]. Recently, both proprietary and open source LLMs have begun to support increasingly longer input windows, making it feasible to input more complex documents [61, 72]. Furthermore, documents in LaTeX, PDF, and Word formats can be converted to HTML with minimal loss, expanding the potential applicability of LLMs to the format of external knowledge [7, 6, 17].

However, employing HTML as the knowledge format for LLMs also presents the challenge of handling longer input sequences and noisy contexts. Our preliminary experiments show that a HTML document from the Web contains over 80K tokens on average, and over 90% of the tokens are CSS styles, JavaScript, comments, or other meaningless tokens. Compared to the common maximum context window of current LLMs, which ranges from 128K, an individual document length of 80K is unacceptable. The noisy tokens, the aforementioned embedding errors in HTML documents can also affect the generation quality of LLMs. To solve this problem, in this paper, we devise a HTML Cleaning module to remove semantically irrelevant content in HTML documents, while keeping the main content intact. We also adjust the HTML tree structure without losing semantic information, for example, merging multiple layers of single nested HTML tags and removing empty tags. These processes reduce the length of the HTML to 6% of its original size.

Even after cleaning, HTML documents remain relatively long (over 4K each) to LLMs. To shorten the input context and remove the noise contained in the original retrieved documents, existing RAG systems have utilized different types of post-retrieval text refiners [19, 22, 66, 75]. These refiners extract the relevant text chunks or key sentences from the documents, regarding the user’s query and LLMs’ preference, and discard other content. These plain-text-based refiners cannot be directly applied to HTML because simply chunking HTML without considering its structure may generate unreasonable chunks. Hence, we further design an HTML Pruning module, which functions upon the intrinsic tree structure of HTML. The pruning process is comprised of the following steps:

(1) **Building a Block Tree.** Each HTML document can be parsed into a DOM tree [58]. We do not simply prune HTML on the DOM tree because it is too finely-grained [16, 62], which brings much computational cost. Instead, we propose to build a corresponding block tree, in which the original DOM tree nodes are merged into hierarchical blocks. The granularity of the block tree can be adjusted by the degree of merging.

(2) **Pruning Blocks based on Text Embedding.** We then prune the block tree using an on-the-shelf embedding model, because it is a simple but effective way to calculate the block’s relevance scores with the user’s query based on their embedding similarity. We apply a greedy pruning algorithm that removes blocks with lower similarity scores, and gets a pruned block tree. However, we observe that the embedding model may fail to work well with the fine-grained blocks because embeddings learned for these small blocks are usually vague and inaccurate, so the pruning step is limited to coarse-grained block trees.

(3) **Generative Fine-grained Block Pruning.** To prune the block tree further, we expand the leaf nodes of the pruned block tree and build a finer-grained block tree. Since the generative model has a longer context window, it can model the block tree globally and is not limited to modeling one block at a time. Thus we further develop a generative model to prune HTML over the fine-grained blocks. The generative model is supposed to calculate the score for each block, which is given by the prompt indicating the block. The sequence is given by the path of HTML tags, starting from the root and walking down to the block’s last tag (e.g., `<html><body><div><block>...</block>...</div></body></html>`). Finally, according to the block scores, we apply a similar greedy pruning algorithm to the final pruned HTML.

We conduct extensive experiments on six datasets including question answering (QA), natural QA, multi-hop QA, and long-form QA. Experimental results confirm the effectiveness of HTML as the format of external knowledge over plain text.

Our contributions are threefold: (1) We propose to take HTML as the format of knowledge in RAG systems, which retains information from original HTML; (2) We propose a simple but effective HTML cleaning algorithm; (3) We propose a two-stage HTML pruning algorithm. This can be applied to most RAG systems and strikes a balance between efficiency and effectiveness.

## 2 Related Works

### 2.1 Retrieval-Augmented Generation (RAG)

RAG systems augment LLM with external knowledge. A typical RAG pipeline includes components such as a query rewriter [55], a retriever [52, 53], a reranker [53, 66], and a refiner [5, 77]. This typical pipeline is widely used by many prominent RAG frameworks, such as LangChain [8] and Haystack [35]. Many works aim to optimize components in the pipeline, and previous works also manage to enhance the performance of RAG in other ways. Some methods devise new RAG frameworks, like retrieving external knowledge actively when internal knowledge is missing [5, 20, 55], or letting the LLM plan the retrieval process in a straight line or a tree structure [27, 52]. However, most existing
```

### --- Page 0003 ---

```markdown
# HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems
Conference acronym 'XX', June 03-05, 2025, Woodstock, NY

## 2.2 Post-Retrieval Process of RAG
RAG systems usually apply post-retrieval processes (i.e., result refiners) to extract only the useful content to shorten the input context to LLMs. The chunking-based refiner is a widely used solution, which first chunks the text according to certain rules, and then uses a re-ranking model to select top chunks with high relevance [25, 40]. Another solution is abstractive refiner, which utilizes a text-to-text language model to generate abstracts of results [14, 19, 60]. Some works use off-the-shelf abstractive models [70, 71] for fine-tuned abstractive models [19] to summarize retrieved results in a segmented and hierarchical manner. Others leverage the logistics of language models to redefine the importance of words within documents [23, 37].

The aforementioned post-retrieval result refiners are all based on plain text. The existing chunking-based methods cannot be directly applied to HTML because simply chunking HTML without considering its structure may generate unreasonable chunks. Furthermore, the abstractive refiners may have problems such as difficulty in dealing with excessively long HTML, high computational cost, or limited understanding of HTML. To alleviate these problems, in this paper, we propose to prune HTML based on its structure.

## 3 Methodology
In this paper, we propose HtmlRAG, which uses HTML instead of plain text as the format of retrieved knowledge in RAG systems, aiming to keep richer semantic and structured information that is missing in plain text. We emphasize that HTML is a popular data format for documents in a knowledge base and other document formats can be easily converted into HTML.

Taking HTML as the format of external knowledge presents a new challenge of excessively long context. Hence, in HtmlRAG, we propose to prune the original HTML documents into shorter ones progressively. We first apply an HTML cleaning module (§3.2) to remove useless elements and tags. We then propose a two-step structure-aware pruning method to further refine the resulting HTML (§3.4). More specifically, we delete less important HTML blocks with low embedding similarities with the input query (§3.4.1), and then conduct a finer block pruning with a generative model (§3.4.2). The overview of our method is shown in Figure 2.

### 3.1 Problem Definition
In the RAG pipeline, a refiner retrieves a collection of HTML documents $D$ from the web, with a total length of $m$. Meanwhile, we have an LLM $A$ as the reader, which generates an answer $y$. The LLM has a maximum length of context window $l$, considering both efficiency and quality. Our HTML compression algorithm maps $D$ to a shorter HTML document $d$. Its length can fit into the LLM's context window, namely the length of $d$ must be less than or equal to $l$. Our goal is to optimize the compression algorithm to find the best mapping from $D$ to $d$ so that the answer $y$ has the highest quality.

### 3.2 HTML Cleaning
Since the original HTML documents are excessively long (over 80K each), and it's needless to involve semantic features, model-based methods are inappropriate at this step. Thus, we first design a rule-based HTML cleaning, which pre-processes the HTML without considering the user's query. This cleaning process removes irrelevant content and discards redundant structures, retaining all semantic information in the original HTML. The compressed HTML of HTML cleaning is suitable for RAG systems equipped with long-context LLMs and are not willing to lose any information before generation. The cleaned HTML also serves as the basis for following HTML pruning.

#### 3.2.1 HTML Content Cleaning
The HTML documents retrieved from the web contain a large amount of extraneous content that is invisible to human users, such as HTML tags, CSS, JavaScript, etc. Most of the HTML tags provide rich structural information that helps the LLM understand the HTML, while CSS and JavaScript content provide limited assistance. So the specific cleaning steps, which are almost lossless, are as follows: (1) We remove CSS styles, comments, and JavaScript; (2) We keep only meaningful HTML attributes.

#### 3.2.2 Lossless Structural Compression
We find that in most HTML documents, their original HTML structure contains redundancies. We can conduct the following transformation to the HTML structure without losing semantic information: (1) We merge multiple layers of single-nested tags. For example, we simplify `<div><div>some text</div></div>` to `<some-text></>`. (2) We removed empty tags, such as `<p></p>`.

### 3.3 Granularity-Adjustable Block Tree Construction
To prune all retrieved HTML documents as a whole, we first concatenate all retrieved HTML documents together, and use Beautiful Soup [50] to parse the concatenated HTML document to a single DOM tree. Pruning HTML using the DOM tree is the most natural way, but the DOM tree is so finely-grained that numerous nodes and the deep tree structure bring huge computational costs.

Considering the above problem, we propose an optimized structure that models HTML, which is not so fine-grained. Ideally, the granularity of the tree structure can be adjusted for different pruning requirements. We term it as a "block tree", and we set the maximum number of words per block, maxWords to control the granularity of the block tree. In terms of block tree construction, we start from a DOM tree, and we merge fragmented child nodes.
```

### --- Page 0004 ---

```markdown
# Conference acronym 'XX', June 03–05, 2025, Woodstock, NY

## Figure 2: HTML for RAG pipeline overview

![HTML for RAG pipeline overview](assets/page_0004_img_1.png)

### 3.4 Block-Tree-Based HTML Pruning

The block-tree-based HTML pruning consists of two steps, both of which are conducted on the block tree structure. The first step uses an HTML cleaning module to prune the result output by the HTML cleaning module, while the second uses a generative model to prune the result output by the first pruning step.

#### 3.4.1 Pruning Blocks Based on Text Embedding

The refining process is expected to shorten the retrieved results while preserving key information as much as possible. A straightforward idea is to extract plain text in the block and calculate a similarity score with the user’s query using text embeddings. When we use a block tree to prune the block tree by deleting low-similarity blocks and retraining higher ones, in practice, we keep deleting the block with the lowest relevance until the total length of the HTML documents satisfies the context window we set. After block deleting, redundant HTML structures will re-appear, so we re-adjust the HTML structure, meaning multiple layers of single-nested tags are merged and empty tags are removed. The detailed pruning algorithm is demonstrated in Appendix B.

The embedding-based HTML pruning algorithm is lightweight but effective. It adapts to the HTML format better compared to plain-text-based refiners. However, it still has limitations, mainly reflected in the following aspects: (1) The embedding model’s context window is limited to the scope of text within the block each time. It does not directly compare candidate blocks in a single inference. Thus the embedding model lacks a global view of the document information; (2) The embedding model cannot handle blocks with finer granularity, because the text within most blocks is not long enough for the embedding model to obtain semantic features.

#### 3.4.2 Generative Fine-Grained Block Pruning

To further prune blocks with a finer granularity, we expand the leaf nodes of the pruned block tree and get a finer-grained block tree. Given the limitations of the embedding-model-based block pruning, we propose to use a generative model because it has a long context to cover the whole block tree and is not limited to modeling one block at a time. We process the cleaned HTML directly with a generative model because it is cleaner than the original (60K on average), which brings much improvement. Similarly, the generative model is supposed to calculate scores for blocks. Inspired by CFCF [48], which takes the chunk’s sequence token probability as the score for that chunk, we propose to use a sequence of tags to identify a block. Specifically, the sequence consists of tags starting from the root and walking down to the block’s tag, and we term this sequence as "block path". In the inference phase, the generative model follows the structure of the blocks and calculates the scores of blocks in the block tree. The scores of blocks are derived from the token logits, as displayed in Figure 3. At last, we use the same block pruning operation as we mention in §3.4.1 to obtain the refined HTML document.

The details of the generative fine-grained block pruning module are introduced in the remaining section.

(1) **Training a Path-aware Generative Model.** Long-context LLMs are capable of modeling a long-context input containing HTML format and following instructions [10, 36]. Considering the computational cost, we employ an existing lightweight long-context LLM as the foundation model. The model input is the concatenation of an HTML, the query, and an instruction, as demonstrated in Figure 4. The instruction is specially designed to help the LLM understand this path generation task, but we find that the unrefined LLM does not meet our requirements. We attribute this to the fact that existing LLMs have not encountered similar tasks or
```

### --- Page 0005 ---

```markdown
![Block score calculation. The block tree is transformed into the token tree with a tokenizer, and corresponding HTML tags and tokens are marked with the same colors. Token generation probabilities are in the upper right corner, and tokens in dashed boxes do not require inference. In the upper right corner of the block tree, the block probabilities are displayed, which can be derived from the corresponding token probabilities.](assets/page_0005_img_1.png)

## Input:
**HTML:** `{"HTML"}`
**Question:** `{"Question"}`
Your task is to identify the most relevant text piece to the given question in the HTML document. This text piece could either be a direct paraphrase to the fact, or a supporting evidence that can be used to infer the fact. The overall length of the text piece should be more than 20 words and less than 300 words. You should provide the path to the text piece in the HTML document. An example for the output is: `<html><body><div>...</div></body></html>`

## Output:
```html
<html><body><div>...</div></body></html>
```

### Figure 4: The prompt for the generative model.

We propose an efficient tree-based inference, and the tree is termed as the "token tree", which has a one-to-one correspondence with the block tree, given a specific tokenizer. We merge tokenized block paths to get the block tree, as Figure 3 shows. For example, `{"<", "html>", "<", "nav>", ">", "<", "html>", "<", "div>", ">"}` share the same prefix, `"<", "html>", "<"`, and can be merged. Ultimately, the i-th token in the tokenized block path will appear at the i-th level of the token tree. After token tree construction, we calculate the probabilities of tokens in the token tree. The calculation has the following conditions: (1) The probability of the root node is 1.0, which is often "e", depending on the tokenizer; (2) The probabilities of sibling child nodes, which have no siblings, are 10; (3) The probabilities of other nodes are calculated by the generative model GenModel. Suppose token has k siblings, which
```

### --- Page 0006 ---

```markdown
# Conference example 'XX', June 03–05, 2023, Woodstock, NY

## 4 Experiments

We conduct experiments on six QA datasets. We simulate the real industrial working scenario for web search engines and compare our method with baselines from various paradigms.

### 4.1 Datasets

We select six datasets, including: (1) ASQA [54]: a QA dataset consisting of ambiguous questions that can be answered by multiple answers supported by different knowledge sources; (2) HotpotQA [67]: a QA dataset consisting of multi-hop questions; (3) NQ [29]: a QA dataset containing real user’s queries collected by Google; (4) TrivialQA [24]: a QA dataset containing user’s questions; (5) MusIQle [56]: a synthetic multi-hop QA dataset; (6) ELIS [13]: a long-form QA dataset with questions collected from Reddit forum. We randomly sample 400 questions from the test set (if any) or validation set in the original dataset for our evaluation.

To simulate the real industrial web search environment, we require real web pages from the Web in HTML format as retrieved documents. However, the widely used Wikipedia search corpus mainly consists of pre-processed passages in plain text format. So, we apply Bing search API in the US-EN region to search for relevant web pages, and we scrape HTML documents through URLs in returned search results. We provide the URLs and corresponding HTML documents in our experiments for reproduction.

### 4.2 Evaluation Metrics

Our method aims to enhance the overall performance of RAG, so we evaluate the LLM’s response as the end-to-end result. We choose different evaluation metrics for datasets according to their question-and-answer formats. For Hotpot-QA and MusIQle, in which each question is annotated with a single short answer, we report Exact Match. For ASQA, NQ, and Trivial-QA, whose questions are annotated with several short answers, we report Exact Match and Hit@1. Hit@1 means at least one answer of the annotated answer finds the exact match in the LLM’s response. ELIS is annotated with long-form answers, and we report ROUGE-L [34] and BLEU [45].

### 4.3 Baselines

Since the best of our knowledge, we are the first to take HTML as the format of retrieved knowledge in RAG systems, we compare HtmlRAG to baselines that conduct post-retrieval processes. These baselines are mainly based on plain text or Markdown for retrieval. We select three chunking-based retrievers and uniformly follow the chunking method in langChain framework [8]. The reranking component is plug-and-play and we use three different rank models: (1) BM25 [51]: a widely used sparse retrieval model; (2) BGE [65]: an embedding model; BGE-Large-EN with encoder-only structure; (3) E5-Mistral [60]: a embedding model based on an LLM, Mistral-7B [18], with decoder-only structure. Besides we select two abstractive refiners: (1) LongLMlMingua [19]: An abstracting model using Llama7B to select useful context; (2) JinAl Readzer [23]: An end-to-end light-weight LLM with 85 parameters fine-tuned on an HTML to Markdown converting tasks dataset.

### 4.4 Experimental Results

For a fair comparison, all end-to-end QA results are obtained with the latest open-source LLM, Llama-3.1-70B-Instruct and Llama-1.3-8B-Instruct [12] under a 4K context window. As for the implementation details of our method, we construct a block tree with a granularity of 256 words before pruning with the embedding model, and we construct a finer-grained block tree with a granularity of 127 words before pruning with the generative model. We choose BGE-Large-EN [65] as the embedding model for the HTML pruning. We also choose a lightweight Phi-3.5-Mini-Instruct [1] with 35 parameters as the backbone for our generative model. The training data used in fine-tuning the generative model contains 2635 automatically constructed training samples ranging from 2K to 32K in length. More implementation details can be found in Appendix A.

### 4.5 Experimental Results

Main experimental results are demonstrated in Table 1. Our method, HtmlRAG meets or exceeds the baselines across all metrics on the six datasets. This demonstrates the effectiveness of HTML pruning. Additionally, we make the following observations:

1. For chunking-based retrievers, we followed LangChain’s [8] chunking rule, which according to HTML tag headings (h1, h2, etc.). Although chunking strategy considers structural HTML structures, it does not utilize the structural information as effectively as our method. Moreover, converting the final output to plain text still results in a loss of HTML structural and semantic information. Among the three rankers we applied, the sparse retriever BM25 is inferior to two dense retrievers. Among those retrievers, the encoder-based BGE performs better than the decoder-based e5-mistral, despite the latter having more parameters.
```

### --- Page 0007 ---

```markdown
| Method        | ASQA        | Hotpot-QA   | NQ          | Trivia-QA   | MuSiQue     | ELI5       |
|---------------|-------------|-------------|-------------|-------------|-------------|------------|
|               | Hit@1      | EM          | Hit@1      | EM          | Hit@1      | EM         | ROUGE-L    | BLEU      |
| Llama-3-8B-Instruct-4K |  |             |             |             |             |            |
| BM25          | 45.00      | 19.84       | 36.25       | 40.75       | 36.17      | 5.75       | 15.90      | 6.56      |
| BGE           | 68.50      | 31.47       | 43.25       | 59.00       | 92.25      | 27.50      | 10.00      | 15.87     | 6.30      |
| ES-Mistral    | 62.50      | 28.51       | 38.50       | 55.60       | 41.73      | 20.75      | 9.00       | 15.77     | 5.85      |
| LongLMlingua  | 59.25      | 26.34       | 40.75       | 55.25       | 41.82      | 20.02      | 16.08      | 6.45      |
| JinaAI Reader  | 53.50      | 23.14       | 34.00       | 47.25       | 34.41      | 84.75      | 24.83      | 6.75      | 15.80     | 5.65      |
| HtmlRAG       | 71.57      | 33.31       | 43.75       | 61.75       | 49.57      | 27.82      | 8.75       | 15.51     | 5.84      |
|               |             |             |             |             |             |            |
| Llama-3-8B-Instruct-4K |  |             |             |             |             |            |
| BM25          | 49.50      | 21.95       | 38.25       | 47.00       | 35.56      | 88.00      | 25.63      | 16.15     | 6.99      |
| BGE           | 68.00      | 30.57       | 41.75       | 59.05       | 93.00      | 12.50      | 16.20      | 6.64      |
| ES-Mistral    | 63.00      | 27.75       | 36.75       | 50.50       | 44.07      | 27.26      | 11.20      | 16.17     | 6.72      |
| LongLMlingua  | 62.00      | 27.34       | 40.50       | 57.25       | 92.50      | 20.25      | 15.84      | 6.39      |
| JinaAI Reader  | 55.25      | 23.73       | 34.25       | 48.25       | 35.40      | 20.53      | 16.06      | 6.41      |
| HtmlRAG       | 68.50      | 30.53       | 46.25       | 60.50       | 45.26      | 93.07      | 13.25      | 16.37     | 6.77      |

(2) Among the abstracted references, LongLMlingua is not optimized for HTML documents, so its extraction ability is affected when dealing with HTML. Additionally, the plain text output loses structural information, resulting in inferior performance compared to our method. The JinaAI reader generates the refined Markdown given the HTML input. However, token-by-token decoding with long input output lengths is not only challenging for end-to-end generative models, but also has high computational cost.

### 4.6 Further Analysis
#### 4.6.1 The Effectiveness of HTML Cleaning
To validate the priority of HTML as the format of retrieved knowledge, we compare our HTML cleaning model, namely the results of HtmlRAG without pruning, with other rule-based cleaning strategies, including: (1) Vanilla HTML; (2) Plain Text: The plain text extracted with an on-the-self package BeautifulSoup [50]; (3) Markdown: The Markdown converted by an on-the-self converter Markdownify [2]. Additional experiments on token count show that HTML-clean drops over 94.07% tokens of the original HTML, while the number for plain text and Markdown conversion are 96.71% and 90.32% respectively. The cleaned HTML is still long, so we conduct experiments under a long-context setting (128K), as shown in Table 2. When HTML is taken as the format of extended knowledge, HtmlRAG without pruning meets or outperforms plain text and Markdown on most datasets, demonstrating its validity. Besides, we make the following observations: (1) Unprocessed HTML documents contain
```

### --- Page 0008 ---

```markdown
| Method     | ASQA     |         |         | Hotpot-QA |         |         | NQ       |         |         | Trivia-QA |         |         | MusiQue  |         |         |
|------------|----------|---------|---------|-----------|---------|---------|----------|---------|---------|-----------|---------|---------|----------|---------|---------|
|            | Hit@1    | EM      | EM      | Hit@1     | EM      | EM      | Hit@1    | EM      | EM      | Hit@1     | EM      | EM      | Hit@1    | EM      | EM      |
| HtmlIRAG   | 68.50    | 30.53   | 46.25   | 60.50     | 45.26   | 93.50   | 27.03    | 13.25   |         |           |         |         |          |         |         |
| w/o Block Tree | 59.00 (0.05) | 25.50 (0.53) | 42.05 (60.01) | 56.25 (4.25) | 42.07 (3.19) | 92.00 (10.50) | 26.09 (0.44) | 8.00 (6.25) |         |           |         |         |          |         |         |
| w/o Prune-Embed | 56.75 (11.75) | 24.06 (4.84) | 37.50 (7.57) | 49.50 (11.00) | 32.77 (9.99) | 17.75 (15.21) | 26.02 (10.21) | 3.75 (3.50) |         |           |         |         |          |         |         |
| w/o Prune-Gen | 62.00 (6.50) | 26.74 (3.79) | 38.75 (7.50) | 57.75 (2.75) | 42.91 (2.35) | 89.00 (4.00) | 25.55 (1.48) | 7.00 (6.25) |         |           |         |         |          |         |         |

![Experimental results for the impact of block tree granularity](assets/page_0008_img_1.png)

| Result Length | Params | Storage | # in-Tokens | # Out-Tokens |
|---------------|--------|---------|-------------|--------------|
| BGE           | 200M   | 2.5G   | 93.45       | 740.3        |
| Prune-Embed   | 200M   | 2.5G   | 152.26      | 2653         |
| Prune-Gen     | 38     | 7.2G   | 6750        | 287.0        |
| LLM Chat      | 38     | 7.2G   | 3661        | 182.9        |
```

### --- Page 0010 ---

```markdown
# Conference summary: XL, June 03–05, 2023, Woodstock, NY

## Long Papers

1. **Zhao, H., Zhang, L., Liu, W., Wei, K., Ando, T., and V. S. K. Sundararajan.** (2023). Understanding Language Models: A Survey. *Proceedings of the 2023 Conference on Computational Linguistics: EMNLP 2023*, 1575–1590. https://doi.org/10.18653/v1/2023.emnlp-1.14

2. **Zhao, H., Zhang, L., Jiao, M., Hobig, O., Yin, Z., Zhang, H., Jiang, Zhao, and D. R. B. D. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A. M. A.

### --- Page 0011 ---

```markdown
# HTMLRG: HTML is Better Than Plain Text for Modeling Reinforced Knowledge in RAG Systems

## Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.  
[https://proceedings.neurips.cc/paper/2022/hash/ 5b6f3e3d74a9f35173f3a5f3f3a5f3f3-Abstract.html](https://proceedings.neurips.cc/paper/2022/hash/5b6f3e3d74a9f35173f3a5f3f3a5f3f3-Abstract.html)  
K. Choi, and O. G. A. (Eds.).

### References

[15] Ashish S. Saperstein, Simona Roshkov, Todd and Wei-ning Zhang. 2022. Blended Method for Automatic Evaluation of Machine Translation. In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, July 6-12, Philadelphia, PA, USA. ACL, 311-318. https://doi.org/10.18653/v1/2023.acl-311-318.

[16] Aya Patel, Brian M. Mohammad, Sadegh Roshan, Noah Constant, Colin Raffel, and Chris Callison-Burch. 2023. Bidirectional Language Models Are Also Few-shot Learners. In *The Eleventh International Conference on Learning Representations*, ICLR 2023, Kigali, Rwanda, May 3-6, 2023. OpenReview.net: https://openreview.net/forum?id=R3T8P3H4.

[17] PerplexityAI. 2022. Perplexity. https://openai.com/index/search?query=prototype.

[18] Hongjun Ding, Lei Meng, Koi Yu, Yujia Zhou, and Zhihe Zhou. 2022. Exploring the Impact of the 2022 Conference on Computational Linguistics. In *Proceedings of the Annual Meeting of the Association for Computational Linguistics*, Volume 1: Long Papers, ACL 2022, July 3-8, 2022, Dublin, Ireland. https://doi.org/10.18653/v1/2022.acl-long.1.

[19] Yuel Qin, Xunheng Yang, Pengcheng Qian, Liang Li, Hang Shen, Yuzhen Zhang, Xiu Yu, Can Ge, and Ling Sun. 2022. Towards the Design of Data Transformation: A Comprehensive Survey on Data Abstraction. *ACM Transactions on Intelligent Systems and Technology* 28 (2022): 2085-2085. https://doi.org/10.1145/2085.

[20] BeautifulSoup. 2022. Beautiful Soup. https://www.crummy.com/software/BeautifulSoup/.

[21] Stephen E. Robertson and Pedro F. Ferreira. 2009. The Probabilistic Relevance Framework. In *Proceedings of the 32nd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR '09, 333-340. https://doi.org/10.1145/1571941.1571980.

[22] Weiwei Shi, Steven Min, Michihiko Yasunaga, Nianwen Xie, and Eric Lewis. 2022. REPL: Block-Level Multi-Task Learning for Language Models. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, Volume 1: Long Papers, ACL 2022, May 22-27, 2022, Metropolis, New Jersey. 121-134. https://doi.org/10.18653/v1/2022.acl-long.16.

[23] Stefan Bethard (Eds.). *Association for Computational Linguistics*, 837-838. https://doi.org/10.18653/v1/2022.NLP-MAIN-66.

[24] Jess Tan, Zhiwei Liu, Ruonan Zhang, and Mingjun Guo. 2022. ASOA: Factored Contexts Improve Long-Form Answers. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022. https://doi.org/10.18653/v1/2022.emnlp-main.56.

[25] Hisham Thweel, Niranath Balasubramani, Tushar Khot, and Ashish Sabharwal. 2022. Multi-Question Evaluation: A Single-Job Question Composition. *Transactions of the Association for Computational Linguistics* 10 (2022), 595-614. https://doi.org/10.1162/tacl_a_00329.

[26] Shih-Rong Tsai, Hsi-Yu Shih, and Matthew Turk. 2024. A Tool for Parallel in Study Activity with Python and Jupyter. In *Proceedings of the Platform for Artificial Scientific Computing*, Paris 2024, Paris, France, 35-56. https://doi.org/10.1145/963949.963949.

[27] Haschen Wang. 2024. What is the HTML DOM? https://www.w3schools.com/whatis/whatis_html.asp. Accessed 2024-12-01.

[28] Haschen Wang, Fu Huang Dong, and Liang Zhang. 2024. *Document Analysis and Recognition - 10th Annual International Conference, ICRA 2024, September 3-5, 2024, Proceedings, Part I Lecture Notes in Computer Science*, 467-471. https://doi.org/10.1007/978-3-031-07363-1_3.

[29] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Muppaneni, and Fur Wu. 2024. Improving Text Embeddings with Large Language Models. In *Proceedings of the 92nd Annual Meeting of the Association for Computational Linguistics: EMNLP 2024*, Singapore, December 2024, 1-8. Houndmills, Basingstoke, UK: Palgrave Macmillan.

[30] Huanfeng Zhang, Xiao Liu, and Jiewei Zhang. 2023. Extractive Summarization with a Chatbot for Faithful Summary Generation. In *Proceedings of the Association for Computational Linguistics: EMNLP 2023*, Association for Computational Linguistics, 3207-3218. https://doi.org/10.18653/v1/2023.emnlp-main.1124.

[31] Huiping Zhang, Xiao Liu, and Jiewei Zhang. 2023. Summarization Interactive Text Summarization via ChatGPT. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, December 6-10, 2023, Japan, 1064-1067. https://doi.org/10.18653/v1/2023.FINDINGS-EMNLP.1124.

[32] Xinyu Zhang, Zhihe Shen, Shensheng Hu, and Zhiwen Liu. 2023. Multi-Task Learning Enhances Long-Form Evaluation Beyond Task Tools. In *Proceedings of the End Annual Meeting of the Association for Computational Linguistics: Long Papers*, ACL 2023, July 2023, 1-8. https://doi.org/10.18653/v1/2023.acl-long.1.

[33] Royhan B. Khosh, Yujin Kim, and Hsun-Sen Wu. 2023. GPT-4Vision is a Generalist Web Agent. If Grounded, In *A First International Conference for Multimodal Intelligence*, ECE 2023, Austin, Texas, USA. Openreview.net: https://openreview.net/pdf?id=6e9k1e2g.

[34] Zhen Zhou, Wout Schelter, Fernando Martinez-Pineda, Val Moros-Davila, César Ferri, and José Hermando-Ortiz. 2024. Larger and more instructable
```

### --- Page 0012 ---

```markdown
# Conference acronym 'XX', June 03-05, 2023, Woodstock, NY

## A Generative Model Training Details

Here we introduce several critical hyper-parameters that define the training process of the generative model. The model's max training context window is set to 3500 tokens. The model is trained for 3 epochs. The training is conducted on 4 computing nodes, with 32 Nvidia A800 GPUs, each having 80 GB memory. To manage memory usage and computational efficiency, `per_device_train_batch_size` is set to 1, while `gradient_accumulation_steps` is set to 8, effectively simulating a larger batch size during backpropagation.

For parallelism, `seq_parallel_size` is set to 8, indicating that the model will distribute its computations across 8 devices if available. The `learning_rate` is set to 2e-5, striking a balance between rapid convergence and avoiding divergence. The learning rate scheduler (`lr_scheduler_type`) is set to 'constant', meaning the learning rate remains unchanged throughout the training unless manually adjusted. For optimization, the Adam optimizer parameters (`adam_beta1`, `adam_beta2`, and `adam_epsilon`) are chosen as 0.9, 0.98, and 1e-8, respectively, to ensure stable gradient updates. The `max_grad_norm` is set to 1.0 to prevent exploding gradients by clipping them if they exceed this norm. A warmup ratio of 0.01 indicates that the learning rate will be gradually increased during the initial steps of training before stabilizing at the base learning rate. `gradient_checkpointing` is enabled to save memory for the cost of increased computation.

DeepSpeed is configured for efficient distributed training. For gradient clipping, `gradient_clipping` is set to 1.0, ensuring that the gradients do not grow too large, thus preventing potential issues like exploding gradients. The `wall_clock_breakdown` option is set to false, indicating that DeepSpeed will not provide a detailed breakdown of the training time spent on different components of the training loop, which can be useful for profiling but may add some overhead. Mixed precision training using `bfloat16` is set to "auto", indicating that DeepSpeed will decide whether to use bfloat16 based on the capabilities of the system and the requirements of the model.

## B Key Algorithms

In this appendix section, we present all the algorithms mentioned in the main text using pseudo code, including the algorithm for constructing the block tree, the pruning algorithm using the embedding model, and the pruning algorithm using the generative model.

To make it clear, we first define elements under a certain node as follows: All sorts of elements under the node are referred to as `node.content`. Text wrapped by child tags is referred to as `node.children`. Text directly attached to the node is referred to as `node.text`. We show an example according to Figure 6. To discriminate between children with the same HTML tag, we append a number to the end of the original tag name. For example, two children with the same `"<div>"` tag are renamed as `"<div1>"` and `"<div2>"`.

The block tree construction algorithm is demonstrated in Algorithm 1, which transforms a DOM Tree T into a Block Tree T'. In the block tree, a block is the smallest unit that can be pruned in.
```


### --- Page 0013 ---

```markdown
# HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems
**Conference acronym** XX, June 03–05, 2025, Woodstock, NY

## Algorithm 1 Construct Block Tree T' from DOM Tree T
1. procedure `ConstructBlockTree(T)`
2. Declare a queue `nodeQueue`
3. R ← root node of T
4. Enqueue root into `nodeQueue`
5. while `nodeQueue` is not empty do
6.     node ← Dequeue from `nodeQueue`
7.     if node is a leaf node then
8.         node.block ← node.content
9.         node.isLeaf ← True
10.    else
11.        if `node.content` < `maxTokens` then
12.            Merge descendant nodes of node
13.            node.block ← node.content
14.            node.isLeaf ← True
15.        else
16.            Expand children of node
17.            for each child of node do
18.                Enqueue child into `nodeQueue`
19.            end for
20.        end if
21.    end if
22. end while
23. return T
24. end procedure

## Algorithm 2 Greedy Block Pruning
1. procedure `GreedyBlockTreePruning(T)`
2.     nodes ← all nodes with blocks from T
3.     for each node in nodes do
4.         node.score ← `Rel(q, node.block)` ⟶ calculate semantic similarity between node and user request
5.     end for
6.     Sort nodes by key `node.score` in ascending order
7.     while each node in nodes do
8.         node ← the node with the lowest score
9.         if `node.isLeaf` then
10.            parent ← node.parent
11.            delete node
12.            while `parent.content` is empty do
13.                parent ← parent.parent
14.                delete parent
15.            end while
16.        else
17.            delete `node.text`
18.        end if
19.    end while
20. end procedure

## Algorithm 3 Token Probability Calculation
1. procedure `TraverseTokenTree`
2. Declare a queue `nodeStack`
3. t₁ ← root node of T
4. t₁.score ← 1.0 ⟶ Set the score of t₁ as 1.0
5. Push t₁ into `nodeStack`
6. while `nodeStack` is not empty do
7.     t₁₋₁ ← Pop from `nodeStack`
8.     children ← Expand children of node p : (t₁, t₂, …, tₖ)
9.     if K = 0 (p is a leaf node) then
10.        continue
11.    else if K = 1 then
12.        t₁₋₁.score ← 1.0
13.        Push the singleton child t₁₋₁ into `nodeStack`
14.    else
15.        prefix ← {input₁, t₁, …, tₖ₋₁}
16.        for each tₖ in children do
17.            tₖ.score ← exp(logits(tₖ)) / Σₖ exp(logits(tₖ))
18.        end for
19.    end if
20. end while
21. end procedure

<div>
<h1>OpenAI o1-preview</h1>
<div>
<p>
In our tests, the next model update performs similarly to PhD...
</p>
</div>
For complex reasoning tasks this is a significant advancement and represents a new level...
</div>

![Node content explained](assets/page_0013_img_1.png)
```

### --- Page 0014 ---

```
Conference acronym 'XX, June 03–05, 2025, Woodstock, NY  
Tan et al.

node, we delete the block directly. Otherwise, if the block consists of directly attached text under a parent node, we delete only those text. After a block is deleted, the algorithm recursively checks if the parent node is empty. If the parent node is empty, it is to be deleted.

The last key algorithm is token probability calculation, as demonstrated in Algorithm 3. We use a depth-first algorithm to traverse tokens in the token tree so that tokens visited sequentially share the longest prefix sequences. The probability of the root token and singleton child tokens are directly set to 1.0, and does not require calculation.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
```


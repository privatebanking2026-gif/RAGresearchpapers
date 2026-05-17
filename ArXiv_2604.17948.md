# ArXiv 2604.17948

### --- Page 0001 ---

```markdown
# RAVEN: Retrieval-Augmented Vulnerability Exploration Network for Memory Corruption Analysis in User Code and Binary Programs

**Partek Jamwal**², **Minghao Shao**¹,², **Boyuan Chen**¹,², **Achyuta Muthuveeran**², **Asini Subanya**², **Boubacar Ballo**², **Kashish Satija**², **Mariam Shafey**², **Mohamed Mahmoud**², **Moncif Dahaji Bouffi**², **Pasindu Wickramasinghe**², **Siyona Goei**², **Yaakuluya Sabbani**², **Hakim Hacid**³, **Mthandazo Ndhlovu**³, **Elanena Kafeza**³, **Sanjay Rawat**³, **Muhammad Shafique**²  
¹New York University, USA ²New York University Abu Dhabi, UAE ³Technology Innovation Institute, UAE

---

**Abstract**—Large Language Models (LLMs) have demonstrated remarkable capabilities across various cybersecurity tasks, including vulnerability classification, detection, and patching. However, their potential in automated vulnerability report documentation and analysis remains underexplored. We present RAVEN (Retrieval Augmented Vulnerability Exploration Network), a framework leveraging LLM agents and Retrieval Augmented Generation (RAG) to synthesize comprehensive vulnerability analysis reports. Given vulnerable source code, RAVEN generates reports following the Google Project Zero Root Cause Analysis template. The framework uses four modules: an Explorer agent for vulnerability identification, a RAG engine retrieving relevant knowledge from curated databases including Google Project Zero reports and CWE entries, an Analyst agent for impact and exploitation assessment, and a Reporter agent for structured report generation. To ensure quality, RAVEN includes a task specific LLM judge evaluating reports for structure, grounding, truth alignment, code reasoning quality, and remediation quality. We evaluate RAVEN on 105 vulnerable code samples covering 15 CWE types from the NIST dataset. Results show an average quality score of 54.21%, supporting the effectiveness of our approach for automated vulnerability documentation.

---

## I. INTRODUCTION

Large Language Models (LLMs) have demonstrated transformative capabilities across a wide spectrum of domains, reshaping how we approach natural language processing, code generation, and reasoning [1]. In cybersecurity, LLMs have been applied to vulnerability detection in source code [2], penetration testing automation [3], automated program repair [4] and agent-based attack/defense for simulating real-world cybersecurity scenarios [5]. These efforts highlight the growing integration of LLMs into security workflows.

Despite these advances, the application of LLMs in generating comprehensive vulnerability analysis reports remains underexplored. Professional cybersecurity organizations such as Google produce detailed reports documenting root cause analyses, exploitation techniques, and remediation strategies [6]. These reports often require human expertise capturing key information such as vulnerability summaries, attack surfaces, exploit primitives, and patch guidance. However, automating the synthesis of such reports using LLMs introduces challenges that have not been adequately addressed.

Automated vulnerability report generation presents three core challenges. First, it requires integrating multiple subtasks, including vulnerability classification, root cause analysis, exploitation assessment, and patch synthesis, into a single coherent report, which is more complex than solving each subtask in isolation. Second, vulnerability analysis often depends on extended code contexts, call chains, and cross-file dependencies, demanding robust long-context reasoning; yet LLMs can suffer from the “lost in the middle” phenomenon when critical evidence appears in the middle of long inputs. Third, evaluating generated reports is inherently multi-dimensional, for example, for factual accuracy, technical depth, and remediation validity, and building automated evaluation frameworks for such domain-specific outputs remains an open problem.

To address these challenges, we propose RAVEN (Retrieval Augmented Vulnerability Exploration Network), a multi-agent framework that leverages LLMs and Retrieval Augmented Generation (RAG) to automatically synthesize comprehensive vulnerability analysis reports. RAVEN implements a four-module architecture as shown in Fig. 1: a Data Collection Pipeline that transforms raw web pages and PDFs into a structured format (2) a RAG Engine that indexes the structured data into a vector database for retrieval (3) an Agent System that takes the vulnerable code snippet and generates a comprehensive, Google Project Zero Style Vulnerability Report.

![Technical Overview of RAVEN. (1) a Data Collection Pipeline that transforms raw web pages and PDFs into a structured format (2) a RAG Engine that indexes the structured data into a vector database for retrieval (3) an Agent System that takes the vulnerable code snippet and generates a comprehensive, Google Project Zero Style Vulnerability Report.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
# phase agentic workflow comprising an Explorer agent for initial vulnerability identification and CWE classification, a RAG engine that retrieves relevant vulnerability knowledge from curated databases including Google Project Zero reports and MITRE CWE entries, an Analyst agent for in depth impact exploitation assessment, and a Reporter agent that generates structured reports following the Google Project Zero Root Cause Analysis template. To ensure report quality, RAVEN incorporates a task specific LLM as a judge module that evaluates generated reports across four dimensions: structural integrity, ground truth alignment, code reasoning quality, and remediation quality.

Overall, our contributions can be summarized as follows:

1) We propose RAVEN, a novel multi-agent framework that integrates RAG with LLM agents to automate the synthesis of professional-grade vulnerability analysis reports.

2) We design a comprehensive evaluation framework utilizing LLM-as-a-Judge methodology to assess vulnerability reports across multiple dimensions.

3) We conduct extensive experiments on the NIST-834 dataset, demonstrating RAVEN's effectiveness in generating accurate and comprehensive vulnerability reports across diverse CWE categories.

## II. RELATED WORKS

### LLM-based Vulnerability Detection for Code

LLMs have shown promising capabilities in identifying source code security vulnerabilities. Recent works have leveraged code-related models such as CodeBERT [7] and GraphCodeBERT [8] for vulnerability classification tasks, while recent work has shifted toward code-driven architectures including GPT-4 and CodeLlama [9]. Studies demonstrate LLMs can perform vulnerability detection through program engineering and chain-of-thought reasoning [10], though comprehensive evaluations reveal that current models still struggle with reliable identification and reasoning about security flaws, particularly in complex, multi-file contexts [11]. Multi-model cooperation approaches such as M2CVD [13] and mixture-of-experts frameworks [14] have been proposed to enhance detection accuracy across diverse CWE categories.

### LLM-based Program Repair/Fix

Automated program repair using LLMs has emerged as an active research area. VRPilot [14] introduced chain-of-thought prompting combined with iterative patch validation feedback, demonstrating improvements over baseline techniques for vulnerability repair in C and Java. VulRepair [15] applied T5-based models for automated software vulnerability repair, while subsequent work explored few-shot learning [16] and fine-tuning strategies [17]. Recent studies have also investigated the integration of static analyzers with LLMs to improve patch quality and reduce false positives [18]. However, challenges remain in generating patches that correctly address root causes without breaking functionality, particularly for complex vulnerabilities tied to project-specific design patterns [19].

### LLM Agents for Security Tasks

Multi-agent LLM systems have been developed to automate complex security workflows. PentestGPT [3] introduced modular task decomposition for penetration testing, utilizing a pesting task tree for structured reasoning. PentestAgent [20] extended this paradigm by incorporating Retrieval-Augmented Generation (RAG) to enhance domain knowledge [21] and automate intelligence gathering, vulnerability analysis, and exploitation stages. Related multi-agent approaches have also been explored in adjacent domains, including hardware design [22], offensive security [23], and vulnerability detection [2]. These frameworks leverage RAG to retrieve relevant vulnerability information from curated databases, mitigating the limitations of LLM context windows and outdated training knowledge. Similar multi-agent architectures have been applied to red teaming and defensive security applications [24].

### LLM-as-a-Judge for Evaluation

Evaluating the quality of LLM-generated security analyses presents unique challenges due to the domain-specific nature of vulnerability reports. The LLM-as-a-Judge paradigm [25] offers a scalable alternative to manual annotation by employing LLMs to assess outputs across multiple dimensions. This approach has been adopted for evaluating code generation quality, vulnerability detection rationales, patch correctness [26] and agentic tasks [27], [28]. While LLM judges demonstrate strong agreement with human evaluators on well-defined criteria, concerns remain regarding their reliability in out-of-domain scenarios [29].

## III. METHODOLOGY

RAVEN's architecture consists of three major technical components, each addressing a distinct challenge in automated vulnerability analysis. These are (i) Data Collection Pipeline (ii) RAG Engine (iii) Agentic Framework. All these modules are discussed in detail in the subsequent sections.

### A. Data Collection Pipeline

In this section, we’ll describe our approach for automated data collection techniques from webpages and books.

1) **Google Project Zero:** Through an extensive analysis of publicly available detailed vulnerability reports, we found Google Project Zero’s 0-days-in-the-wild repository [6] consisting of 70 reports detailing the essential reasoning behind various 0-day exploits.

To extract the content from the reports, we first retrieve URLs from the Google Project Zero homepage using the Crawl4J library [30]. A BFS (Breadth-First Search) deep crawling strategy with a depth level of 2 is used. This crawling process yields 89 unique same-domain URLs. On manual inspection of these URLs, we noticed that some of these links point to non-report sources. Therefore, to retain links that only contain the reports, we define a custom regex filter. Using this
```


### --- Page 0003 ---

```markdown
![Data Collection Pipeline. (Top) URL extraction and web scraping of security reports; (Bottom) PDF-to-markdown conversion using OCR and Table of Contents (ToC) indexing. The final output is a cleaned, structured markdown for RAG.](assets/page_0003_img_1.png)

![Overview of RAVEN's RAG Engine. It comprises 3 chunking strategies (Flat, Contextual, HyPE), 3 retrieval methods (Embeddings, Hybrid, HyPE) and 2 rankers (Cross-encoder, LLM-based).](assets/page_0003_img_2.png)

Similarly to the URL extraction phase, a BFS deep-crawling strategy with a depth of 1 is used to ensure that all content from the target URL and the embedded links within the original report, such as cited advisories, GitHub repositories, and other relevant links, are captured. A list of HTML tags and CSS selectors (e.g., navigation bar, footer, etc.) is used to avoid as much noise as possible. All embedded URL content is encapsulated within structural tags: <START FOR [URL]> to manage truncation; the embedded URL content is limited to three appearances per report; subsequent instances of the same URL are skipped to prevent unnecessarily context window inflation. The output of this process is 70 markdown files, each containing the raw text of the vulnerability reports.

Lastly, we perform an LLM-based cleaning of the raw reports to transform them into a structured, RAG-ready vulnerability documents. All these files are parsed to an LLM along with a specialized ≈ 6200 token prompt that guides it on how to parse, clean, and format the raw scraped content into a consistent schema optimized for Retrieval-Augmented Generation (RAG).

1) PDF to Image Conversion: Every page of the PDF was exported as individual PNG files using the pdf2image library, resulting in 2,735 PNG images.  
2) OCR Processing: Each saved image is processed by Chandra [32], a highly accurate OCR model that converts images and PDFs into structured markdown while preserving layout information. Since the bulk contains tabular data, code blocks, and complex data layouts, this model was perfectly suited for high-fidelity text extraction.  
3) Knowledge Base Consolidation: At this stage, we have 2,735 markdown documents, one for each page of the PDF. Using the markdown documents for the Table of Contents containing the Chapter Name, Start page, and End page. The index is then used to create chapter specific documents.

B. RAG Engine  
RAVEN's RAG engine implements a modular architecture comprising three chunking, three retrieval, and two ranking components, all built upon ChromaDB as the persistent vector database with HNSW (Hierarchical Navigable Small World) indexing for efficient approximate nearest neighbor search. For document indexing, we first have the Contextual Chunking strategy, which uses a fixed-size sliding window on a given document, thereby splitting it into arbitrary fixed-size windows of text. Next, we have the Contextual Chunking strategy based on Anthropic’s Contextual Retrieval [33] approach. The idea is to prepend an LLM-generated chunk-specific explanatory context in relation to the embedded chunk. Each chunk before embedding and saving it to the vector database. Lastly, we have the HyPE (Hypothetical Prompt Embeddings) approach, where 3-5 hypothetical chunk-specific questions are generated for each document chunk, produced by the flat chunking strategy. These generated hypothetical questions become the chunk’s representation. Consequently, the retrieval process for the HyPE strategy shifts from query-to-document to query-to-hypothetical questions. Once the chunks are created, they are converted into embeddings; instead of document chunks.

Moving on to the retrieval process, we first implemented a standard embedding-based approach, where we perform a pure vector-based similarity search between the embedded query and the ingested chunks. Building on this, we have the hybrid approach, where we combine semantic (vector) search and keyword (BM25) scores for enhanced retrieval. We use the following scoring mechanism:

$$
\text{Score}_{\text{Final}} = w_{\text{semantic}} \cdot \text{Score}_{\text{semantic}} + w_{\text{keyword}} \cdot \text{Score}_{\text{keyword}} \quad (1)
$$

where the default weights are 0.6 for semantic and 0.4 for keyword components. Lastly, we have the HyPE retrieval mechanism.
```

### --- Page 0004 ---

```markdown
# Page 0004

strategy that performs query-to-generated-hypothetical question matching.

Once the initial set of chunks is retrieved, it is passed into the Reranker module, which plays a crucial role in refining the results obtained from the initial retrieval process. RAVEN’s RAG Engine comprises two reranking modules, namely a Cross-Encoder based reranker where the query and the document are jointly processed to compute the relevance score. Cisco’s SecureBERT 2.0 [34] which is a domain-specific transformer model optimized for cybersecurity tasks is used as the Cross-Encoder model. In addition to the Cross Encoder, we also have a pointwise LLM-based reranker that scores document relevance on a 0–10 scale.

## C. Agentic Workflow

RAVEN implements a four-phase multi-agent system for automated vulnerability analysis and report generation. The workflow comprises four phases: Exploration, Analysis, Report Generation, and Report Evaluation via LLM as a Judge.

The workflow begins by providing the Explorer Agent with a filepath to the vulnerable code snippet. The agent invokes the `codebase` function and extracts all source code, storing it in the `code_payload` variable. Once extracted, the explorer agent uses the code snippet as the query and calls `rag_retrieval` function, thereby retrieving the most relevant documents from the LLMs. The retrieved context and extracted code are fed into the `generate_explorer_findings` function in the LLM generating an initial set of findings comprising (1) a concise vulnerability summary class (2) matching Common Weakness Enumeration (CWE) standards, (3) Common Vulnerability Enumeration (CVE) identifiers, (4) evidence (line number, snippet, and reason) of where the issue resides, and (5) recommended remediation steps for immediate mitigation. 

Once we have the initial set of explorer agent findings, `ExplorerFindings`, we use it along with the initial extracted code `payload` data and feed it to the analyst agent, which invokes the `analyze` function. The LLM yields an enhanced set of findings built upon the Explorer Agent’s outputs. These include (1) an impact assessment of the vulnerability, (2) likelihood of exploitation, (3) critical points, (4) confidence level in the analysis, and most importantly, (5) remediation strategies (fixes) for the currently detected vulnerability and its future variants.

Upon successful generation of the analyst agent’s findings, `AnalystFindings`, those findings, in conjunction with the initial payload code, `code_payload`, and the explorer agent’s findings, `ExplorerFindings`, are provided as context to the report agent for detailed vulnerability report generation. The report generation happens in three phases:

1. **Vulnerability Analysis**: Based on the provided contextual data context, the agent determines whether a vulnerability exists. If it exists, the agent generates a short descriptive title of the detected vulnerability, a 2-4 sentence summary of where the issue appeared, how it was triggered, etc., CWE description of the vulnerability, root cause, impact summary, attack surface (potential entry points an attacker could exploit to trigger a vulnerability), and lastly the vulnerable snippet from the original code where the bug resides.

2. **Exploit Analysis**: The agent uses Phase 1’s findings to produce the attack vector (specific path an attacker takes to execute the exploit), exploit primitives (the low-level capabilities gained upon successful triggering of the vulnerability), exploit steps (list of 3-5 exploit steps describing how the vulnerability could be triggered), and an exploitability summary.

3. **Fix Generation**: The agent takes the output produced in the earlier two phases and uses it to produce the remediation artifacts. More specifically, the agent produces a code snippet (<20 lines) that directly addresses the root cause, an explanation of how/why the fix eliminates the vulnerability, and lastly, the variant guidance, i.e., steps on dealing with future variants of the same exploit. The output from all three phases is consolidated, fed to a custom markdown renderer to create the final report, and then is evaluated by the Judge agent. The format is inspired by Google Project Zero’s RCA template.

The judge agent uses the original extracted code, ground truth annotation (comes with the original dataset), and the generated report to give its evaluation. This agent utilizes 2 LLMs, Claude 4.5 Sonnet and Gemini 1.3 Pro each possessing specialized evaluation capabilities to evaluate the generated report across 4 dimensions:

- **Structural Integrity (0–10)**: This metric measures how well the generated report adheres to the provided guidelines and the completeness of mandatory fields.
- **Ground Truth Alignment/Factual Grounding (0–10)**: This metric evaluates factual correctness strictly against provided annotation and source code. Score of 0 for halucinated CVEs.
- **Code Reasoning Quality (0–10)**: This metric assesses the technical depth, specificity, and logical consistency of the report’s reasoning.
- **Remediation Quality (0–10)**: This metric measures how valid the proposed fix code is relative to the identified root cause in the ground truth annotation.

## IV. EXPERIMENTS

### A. LLM Selection

As outlined in Section III-C, the agentic system operates in four phases: Exploration, Analysis, Report Generation, and Report Evaluation. For the first three phases, we use the same LLM. All experiments are conducted exclusively on Falcon models, Falcon-HIR-7B [35] (F16), Falcon-HIR-7B-Instruct (F16), Falcon-HL-34B-Instruct (Q8), and Falcon-3B-Instruct (F16), released by the Technology Innovation Institute (TII).

### B. Parameters and Features

For all Falcon models, a temperature of 0.6 and top-p of 0.95 is used. The RAG engine ingests 70 Google Project Zero
```


### --- Page 0005 ---

```markdown
![The RAVEN Agentic Pipeline. It orchestrates four specialized agents for end-to-end vulnerability analysis.](assets/page_0005_img_1.png)

| CWE   | Description                                      | %     |
|-------|--------------------------------------------------|-------|
| CWE-119 | Improper Restriction of Operations              | 31.43 |
| CWE-787 | Out-of-bounds Write                             | 6.67  |
| CWE-476 | NULL Pointer Dereference                         | 6.67  |
| CWE-457 | Use of Uninitialized Variable                    | 6.67  |
| CWE-416 | Use After Free                                  | 6.67  |
| CWE-170 | Improper Null Termination                        | 6.67  |
| CWE-122 | Heap-based Buffer Overflow                       | 6.67  |
| CWE-121 | Stack-based Buffer Overflow                      | 6.67  |
| CWE-120 | Buffer Copy without Checking Size of Input      | 6.67  |
| CWE-415 | Double Free                                     | 5.71  |
| CWE-401 | Missing Release of Memory                        | 5.71  |
| CWE-190 | Off-by-one Error                                | 0.95  |
| CWE-126 | Buffer Over-read                                | 0.95  |
| CWE-124 | Buffer Underwrite                               | 0.95  |
| CWE-123 | Write-what-where Condition                      | 0.95  |

### C. Benchmark

To evaluate how RAVEN's agentic system is able to identify the vulnerability within a given code snippet and generate a detailed report with the fix, we use the NIST-SARD dataset [36]. It is a large scale dataset curated by NIST as part of the SAMATE project. It contains approximately 450,000 programs covering more than 150 classes of weaknesses. Each test case is packaged with source code and a manifest file containing the case metadata used as ground truth to assess the system's factual correctness. The evaluation is performed on 105 memory focused vulnerable samples from the dataset. The composition of our dataset mixture is shown in Table I.

### D. Metrics

The agentic system is evaluated only using LLM-as-a-Judge scores. The judging metrics are Structural Integrity (0-10), Ground-Truth Alignment (0-10), Code Reasoning Quality (0-10) and Remediation Quality (0-10).

### V. RESULTS

We evaluated 4 Falcon models across 10 RAG configurations, resulting in 40 experiments. Each experiment was performed on 105 samples. For every sample, two independent judges generate scores for four dimensions: Structural Integrity (SI), Factual Grounding (FG), Code Reasoning Quality (CR), and Remediation Quality (RQ). Let $J^1_i$, $J^G_i$, $J^R_i$, and $J^Q_i$ denote the scores assigned by Judge 1 and $J^1_2$, $J^G_2$, $J^R_2$, $J^Q_2$ denote the corresponding scores assigned by Judge 2. The overall model score $S_{overall}$ is computed as the arithmetic mean of all eight criterion scores and is given by:

$$
S_{overall} = \frac{1}{2} \sum_{i=1}^2 \left( J^1_i + J^G_i + J^R_i + J^Q_i \right) .
$$

Table II shows the overall scores, $S_{overall}$, of the models. It is clear from the table that there is no specific RAG configuration that outperforms the others. To gain a deeper insight into the overall scores, we pick a model and analyze...
```

### --- Page 0006 ---

```markdown
| Model                | Flat Chunking                     | Contextual Chunking               | HyPE                |
|----------------------|-----------------------------------|-----------------------------------|---------------------|
|                      | EO     | HYB    | HYB    | EO     | HYB    | HYB    | HYB    | CE + CE | LLM + CE | LLM + CE |
|                      | + CE   | + LLM  | + LLM  | + CE   | + LLM  | + LLM  | + LLM  |         |          |          |
| Falcon 1H 34B Inst.  | 7.45   | 7.59   | 7.43   | 7.54   | 7.68   | 7.62   | 7.41   | 7.65    | 7.74     | 7.34     |
| Falcon 1H 7B        | 7.11   | 7.32   | 7.03   | 7.09   | 7.07   | 7.16   | 7.29   | 6.96    |          |          |
| Falcon 7H 7B        | 6.80   | 6.66   | 6.77   | 6.82   | 6.50   | 6.66   | 6.80   | 6.90    | 6.77     |          |
| Falcon 3 10B Inst.  | 5.83   | 5.81   | 5.91   | 5.85   | 5.93   | 5.72   | 6.10   | 5.78    |          |          |

**Abbreviations:** EO = embeddings-only, HYB = hybrid retrieval, CE = cross-encoder reranker, LLM = LLM reranker.

---

Figure 5: Box Plot Analysis of Overall Scores for all Falcon Models

From the box plot in Figure 5, we can see that the Falcon H1R-7B model exhibits the highest variance compared to the other models. Therefore, we select this model for further analysis of its individual dimensions.

### A. Structural Integrity
It is evident from the heatmap shown in Figure 6 that the Falcon H1R model consistently adhered to the required formatting guidelines across different RAG configurations. Upon examining the judge agent's evaluation logs to identify the causes of point deductions, we found that these were primarily due to two factors:
- **Formatting issues:** Corrupted code rendering, truncated fixes (e.g., missing braces or incomplete function definitions), and the use of literal escape sequences (e.g., `\n`) instead of actual line breaks.
- **False negatives:** Incorrectly classifying a vulnerable code snippet as non-vulnerable, which results in an automatic score of zero for the corresponding evaluation.

### B. Ground Truth Alignment / Factual Grounding
Figure 6 illustrates that the Contextual Chunking + Hybrid Retrieval + LLM Reranker configuration (score=6.14) achieves the highest overall score, closely followed by Contextual Chunking + Embeddings-Based Retrieval + LLM Reranker (score=6.02) and Flat Chunking + Embeddings-Only Retrieval + Cross Encoder Reranker (score=5.98).

The strong performance of Contextual Chunking + Hybrid Retrieval + LLM Reranker configuration can be attributed to the fact that each chunk is accompanied by a semantically rich LLM-generated context, thereby enhancing the chunk's representation. During retrieval, the hybrid strategy combines semantic similarity with keyword-based matching (BM25), enabling the retrieval of a wide range of documents. Given that the input to the RAG engine is a NIST-SARD code snippet, the retrieved set of documents may contain some unrelated/noisy narratives due to superficial similarities (shared variable names, function calls, etc.). The initial retrieval is then fed to the LLM-based reranker. Given the enhancing capabilities of the LLMs along with the context-enhanced chunk, the reranker is able to generate optimal rankings for the provided query, leading to factually grounded answers (score=6.14). Cross encoders, as a class of deterministic rerankers, compute pairwise relevance by calculating the full token-level interactions between query and document. Due to the absence of explicit logical reasoning, the final ranked set of retrievals does not contribute to a high Factual Grounding score (score=5.53). Our empirical findings suggest that additional context confounds the cross encoder's relevance scoring for the H1R model. This performance disparity between the rerankers for contextual chunks can also be viewed by switching the retrieval strategy from hybrid to embeddings-only where an LLM Reranker (score=6.02) outperforms the cross encoder (score=5.47).

Upon analyzing the Flat Chunking + Embeddings-Only Retrieval + Cross Encoder Reranker configuration (score=5.98), we observe that when retrieving documents using this strategy, the search space for the retriever is limited to a set of context-free chunks. Consequently, the likelihood of retrieving unrelated chunks is slightly higher, as there is no additional contextual information guiding the retriever's decisions. In such cases, the domain-specific cross-encoder reranker, SeCureBERT 2.0 (score=5.98), outperforms the LLM reranker (score=5.78). Switching the retrieval method from hybrid to an embedding-only approach yields similar performance results.
```

### --- Page 0007 ---

```markdown
# H1R - Average Scores (4 Dimensions)

![Detailed description of the chart](assets/page_0007_img_1.png)

## C. Code Reasoning
The results of the analysis performed in Section V-B are also reflected here. The Contextual Chunking + Hybrid Retrieval + LLM Reranker (score=7.10) and Flat Chunking + Hybrid Retrieval + Cross Encoder Reranker (score=7.32) outperform all other configurations leading to high Code Reasoning Quality for the Falcon H1R model.

## D. Remediation Quality
The trend visualized in Section V-B is clearly visible here. The Contextual Chunking + Hybrid Retrieval + LLM Reranker (score=6.98) and Flat Chunking + Hybrid Retrieval + Cross Encoder Reranker (score=6.80) are the top performing RAG configurations for this criteria.

## E. CWE Remediation Statistics
To evaluate the quality of the fixes generated by each model using our proposed agentic system, we first select the best configurations for each Falcon model, as presented in Table II. For each selected configuration, we calculate the proportion of CWEs for which a valid fix was generated. This assessment is conducted by examining the logs produced by the Judge Agent. For each test case, the Judge Agent returns an evaluation in accordance with a predefined template.

```json
{
  "structural_integrity": { /* omitted */ },
  "factual_grounding": { /* omitted */ },
  "code_reasoning_quality": { /* omitted */ },
  "remediation_quality": {
    "score": 0,
    "justification": "2-3 sentence explanation",
    "fix_addresses_root_cause": false,
    "syntax_valid": false,
    "overall_score": 0.0
  }
}
```

![Detailed description of the chart](assets/page_0007_img_2.png)

Upon examining the Judge Agent logs for Falcon H1R-7B and Falcon H1-34B-Instruct, we can observe that:
- Falcon H1R-7B: This model consistently generates syntactically correct fixes; however, these fixes often prevent program failure by disabling or bypassing the problematic behavior rather than directly addressing and correcting the underlying cause.
- Falcon H1-34B-Instruct: This model reliably generates syntactically valid code, but remediation quality de-
```

### --- Page 0008 ---

```markdown
# VI. CONCLUSION

In this paper, we presented RAVEN, a multi agent framework that combines LLM agents with Retrieval Augmented Generation to produce structured, professional style vulnerability analysis reports from vulnerable source code. By coordinating an Explorer, a retrieval module grounded in curated vulnerability knowledge, an Analyst, and a Reporter, RAVEN supports end to end reasoning from vulnerability identification to impact assessment and remediation guidance. We also introduced a task specific LLM as a Judge to evaluate report quality across structure, alignment with ground truth, code reasoning, and remediation. Experiments on NIST-SAR 2020 across three CWE categories demonstrate that RAVEN can generate coherent and technically meaningful reports, indicating its promise for scaling vulnerability documentation workflows and assisting humans.

# ACKNOWLEDGMENTS

This research was partially funded by Technology Innovation Fund (TII) under the "CASTLE: Cross-Layer Security for Machine Learning Systems IoT" project. Experiments are performed with NYUAD Jubail High Performance Computing.

# REFERENCES

[1] H. M. Shor et al., “Survey of software model architectures: Trends, techniques, and challenges,” IEEE Access, 2024.

[2] H. Xi et al., “From code to the real world: OS vulnerability reports,” arXiv preprint arXiv:2405.12057, 2024.

[3] G. Deng et al., “PentFi: Evaluating and harnessing large language models for automated penetration testing,” in 33rd USENIX Security Symposium. USENIX Association, Aug. 2024, pp. 847–864.

[4] U. Kulsoum et al., “A case study of LLM for automated vulnerability assessment: Assessing impact of reasoning and contextual feedback,” in Proceedings of the ACM International Conference on AI-Powered Software, ser. AIware 2024. ACM, 2024, pp. 103–111.

[5] R. G. Choi et al., “An empirical evaluation of the effectiveness of code analysis,” arXiv preprint arXiv:2402.18174, 2024.

[6] Rout Causes Analyses. “Advisors in-the-wild.” [Online]. Available: https://github.com/days-in-the-wild/rarn.

[7] Z. Feng et al., “CodeBERT: A pre-trained model for programming and natural languages,” in Findings of the Association for Computational Linguistics: EMNLP 2020, T. Chen et al., Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 1536–1547.

[8] D. Guo et al., “Graphcodebert: Pre-training code representations with data flow,” in ICLR, 2021.

[9] B. Roziere et al., “Code llama: Open foundation models for code,” 2024. [Online]. Available: https://arxiv.org/abs/2403.12050

[10] M. S. Anwar et al., “Gosnarr: Detecting logical vulnerabilities in enemy language using inductive contextual reasoning,” in 2025 IEEE Symposium on Security and Privacy (SP), 2025, pp. 758–773.

[11] Y. Liu et al., “SurVEST: Evaluating Structured Vulnerability Reasoning in Large Language Models for Source Code Vulnerability,” in 2025 IEEE Symposium on Security and Privacy (SP). IEEE Computer Society, May 2025, pp. 3041–3042.

[12] A. Yildiz et al., “Benchmarking LLMs and ML-based agents in practical vulnerability detection for code repositories,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. ACL, Jul. 2025, pp. 30–36.

[13] Z. Wang et al., “M2Dev: Enhancing vulnerability understanding through multi-modal information for code vulnerability detection,” ACM Trans. Softw. Eng. Methodol., Oct. 2025.

[14] A. Leskys et al., “Llmcp: context-aware vulnerability detection through code property graph-guided large language models,” in Proceedings of the 34th USENIX Conference on Security Symposium, ser. SEC ’25. USENIX Association, 2025.

[15] M. Fu et al., “VulnAI: A robust automated software vulnerability repair,” ser. ESEC/FSE 2022, ACM, 2022, pp. 935–947.

[16] S. Feng et al., “Promoting AI use you need: Automated android bug replay repair by broadening input ranges of sources,” in Proceedings of the IEEE/ACM 46th ICSE, ser. ICSE, 2024.

[17] X. Zhou et al., “Out of sight, out of mind: Better automated vulnerability repair by broadening input ranges of sources,” in Proceedings of the IEEE/ACM 46th ICSE, ser. ICSE, 2024.

[18] Z. Li et al., “LLM-assisted static analysis for detecting security vulnerabilities,” in ICLR, 2025.

[19] H. Peura et al., “Examining Zero-Shot Vulnerability Repair with Large Language Models,” in 2302 IEEE Symposium on Security and Privacy (SP), 2023, pp. 2393–2365. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/ISSP.2023.10179420

[20] X. Shen et al., “Pentest: Incorporating LLM agents to automated penetration testing,” in Proceedings of the 2021 ACM Asia Conference on Computer and Communications Security, ser. ASIA CCS ’25. Association for Computing Machinery, 2025, p. 375–391.

[21] M. Xiao et al., “Cracken: Cybersecurity LLM agents with knowledge-based execution,” arXiv preprint arXiv:2505.17107, 2025.

[22] W. Xiao et al., “Trojanix: LLM-based framework for anti-trojan localization,” arXiv preprint arXiv:2152.00325, 2025.

[23] M. Sudesh et al., “D-CRACK: Dynamic contextual retrieval intelligent multi-agent system with planner and heterogeneous executors for offensive security,” arXiv preprint arXiv:2502.19301, 2025.

[24] P. He et al., “Red-teaming LLM multi agent systems in various attacks,” in Findings of the ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. ACL, Jul. 2025, pp. 6726–6479. [Online]. Available: https://aclweb.org/anthology/2025.findings-acl.3497

[25] D. Li et al., “Evaluating LLMs in judgment: Opportunities and challenges in empirical methods,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2025, pp. 2757–2791. [Online]. Available: https://aclweb.org/anthology/2025.emnlp-main.1357

[26] Y. Mo et al., “Can you really use code completion? evaluating large language models from a code security perspective,” in Proceedings of the 3rd Annual Meeting of the Association for Computational Linguistics: Long Papers, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 1739–1769.

[27] M. Shae et al., “Towards effective security in LLMs: Hyperparameter tuning, lm as a judge, and a lightweight benchmark,” arXiv preprint arXiv:2506.20875, 2025.

[28] B. Chen et al., “Metakb: A general and extensible reinforcement learning framework for obfuscation-based jailbreak attacks on black-box lms,” arXiv preprint arXiv:2306.22557, 2025.

[29] V. Raina et al., “Is LLM-as-a-bot? investigating universal adversarial attacks on zero-shot LLM assessment,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, ACL, Nov. 2024, pp. 7499–7517.

[30] UncleCode, “Crawl-what: Open friendly web crawler & scraper,” 2024.

[31] MITRE Corporation, “Common weakness enumeration (cwe) version 17,” MITRE Corporation, Tech. Rep. Apr. 2025.

[32] Datalab, “Chandra: Our model for complex documents,” https://cwe.codalab.org/teams.

[33] Contextual retrieval in AI Systems. [Online]. Available: https://www.researchgate.net/publication/contextual-retrieval

[34] E. Ashari et al., “SecureBERT 2.0: Advanced Language Model for Cybersecurity Intelligence,” 2025.

[35] F. L. T. F. Al-Rahim, “Pushing the Reasoning Frontiers with a Hybrid Model for Efficient Test-Time Calling,” 2025.

[36] “Software Assurance Reference Dataset (SARD),” NIST, Feb. 2021.
```


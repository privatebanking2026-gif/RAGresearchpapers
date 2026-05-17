# ArXiv 2408.14484

### --- Page 0001 ---

```markdown
# Agentic Retrieval-Augmented Generation for Time Series Analysis

**Chidash Ravuru**  
IT Dharwad  
India  
200010046@iitdh.ac.in  

**Sagar Srinivas Sakhinna**  
TCS Research  
India  
sagar.sakhinna@tcs.com  

**Venkataramana Runkana**  
TCS Research  
India  
venkataramana.runkana@tcs.com  

---

## ABSTRACT
Time series modeling is crucial for many applications, however, it faces challenges such as complex spatio-temporal dependencies and distribution shifts in learning from historical context to predict task-specific outcomes. To address these challenges, we propose a novel approach using an agentic Retrieval-Augmented Generation (RAG) framework for time series analysis. The framework leverages a hierarchical multi-agent architecture where the master agent orchestrates specialized sub-agents and delegates the end-user request to the relevant sub-agent. The sub-agents utilize smaller, pre-trained language models (SLMs) customized for specific time series tasks through fine-tuning using instruction tuning and direct preference optimization, and retrieve relevant prompts from a shared repository of prompt pools containing distilled knowledge about historical patterns and trends to improve predictions on new data. Our proposed modular, multi-agent RAG approach offers flexibility and achieves state-of-the-art performance on various time series tasks by tackling complex challenges more effectively than task-specific conventional methods across benchmark datasets.

## KEYWORDS
Time Series Analysis, Retrieval Augmentation

---

## 1 INTRODUCTION
Time series modeling underpins a vast spectrum of real-world applications, including demand planning [52], anomaly detection [54], inventory management [24], weather modeling [31], and many others. However, it is not without its challenges. High dimensionality, non-linearity, sparsity, and distribution shifts all pose significant hurdles. Successfully navigating these challenges in time series analysis applications necessitates both considerable domain knowledge and the design of neural network architectures tailored to address task-specific goals, leading to better performance. In contrast to task-specific approaches, which employ different architecture designs for time series analysis, foundational pre-trained large language models (LLMs), such as OpenAI's GPT-4 [29] and Google's Gemini [34, 39], with their strong generalization and logical reasoning capabilities, have shown remarkable versatility across a broad spectrum of natural language processing (NLP) tasks, requiring minimal fine-tuning [17] or only a few demonstrations [2] for adaptation to niche tasks. Open-source, small-scale pre-trained language models (SLMs), such as Google Gemini (40) and Meta LLaMA (1, 41), offer cost-effective domain customization through Parameter Efficient Fine-Tuning (PEFT) [15, 16] techniques using task-specific labeled datasets. Additionally, these smaller models can be further augmented with human preferences using Direct Preference Optimization (DPO) [8], a fine-tuning technique that utilizes paired preference data, such as datasets of preferred and dispreferred responses. However, SLMs may lack the reasoning and generalization capabilities of large-scale proprietary language models. The potential of foundational SLMs designed for universal time series applications (a single-model-fits-all approach), such as diverse time series tasks like classification, anomaly detection, forecasting, imputation, and others, remains largely unexplored but holds great promise. This approach contrasts sharply with the traditional approach of using customized, task-specific methods [43, 49, 50] for time series modeling for various applications. Adapting SLMs designed for NLP tasks for time series modeling to capture trends and patterns within the complex data, though unconventional, offers a clear possibility for producing unique insights. However, this is a challenging task as SLMs are trained primarily on text corpora, which operates on discrete tokens, while time series data is inherently continuous. Furthermore, SLMs may lack the inherent ability to detect and interpret time series patterns and trends like seasonality, cyclicity, or outliers, due to the absence of related pretraining knowledge. Moreover, current LMs designed for time series analysis [14, 20, 56] rely on a fixed-length window of past observations to generate predictions, which may be inadequate for capturing complex patterns and trends present in time series data. The recent trends, Retrieval-Augmented Generation (RAG) or Retrieval-Augmented Language Models [23, 37, 31] combines pre-trained language models with information retrieval for external knowledge bases to augment text generation capabilities for open-ended question-answering (ODQA) tasks, thereby improving language modeling for text summarization, completion, and retrieval accuracy. While regular RAG enables generation with retrieved knowledge for ODQA tasks, Agentic RAGs take this further by being instruction-following agents that can tackle complex goals through multi-step reasoning and iterative refinement cycles using repeated retrievals over a knowledge base to ensure the final response aligns with the end user request. In this work, we propose an Agentic RAG framework for time series analysis to improve task-specific outcomes by addressing challenges like distribution shifts, fixed window limitations in time series data. Figure 1 illustrates the framework. Our Agentic RAG framework presents a hierarchical, multi-agent architecture composed of a master (top-level) agent and specialized sub-agents customized for specific time series tasks. The top-level agent acting as the orchestrator analyzes the incoming user request, determines its nature and complexity, and then routes (or delegates) it to the corresponding task-specific sub-agent to produce the desired output. Similarly to how regular RAG frameworks retrieve relevant information from external knowledge bases like documents, databases, or access the real world through APIs, this Agentic RAG framework leverages distinct prompt pools as internal knowledge bases for each sub-agent focused on specific time series tasks. As specialized
```


### --- Page 0002 ---

```markdown
![The figure illustrates the proposed agentic RAG framework, designed to handle diverse time series analysis tasks. The framework employs a hierarchical, multi-agent architecture. A master agent receives end-user questions and routes them to appropriate specialized sub-agents based on the specific time series task (e.g., forecasting, imputation, classification, anomaly detection). The sub-agents utilize pre-trained SLMs fine-tuned on task-specific datasets using techniques like instruction tuning and direct preference optimization to capture spatio-temporal dependencies within and across the time series datasets. Each sub-agent maintains its own prompt pool as 'key-value' pairs, which stores relevant historical knowledge related to specific trends and patterns within its respective specialized domain. This allows the sub-agents to leverage related past experiences for improved task-specific predictions on new, similar data, and is then relayed back to the user through the master agent.](assets/page_0002_img_1.png)

## 2 PROBLEM FORMULATION
Consider a time series dataset characterized by $N$ univariate time series, and we represent the data collected at a specific timestamp. To refer to data from a specific time series or timestamp, we use subscripts and superscripts, respectively. For instance, $X_t^i$ denotes the data from the $i$-th time series, and $X_{t}^{-1}$ denotes data at timestamp $t$.

### 2.1 Forecasting
We utilize a sliding window $w(10, 16)$ of size $r$ to construct time series subsequences $X = X_{t-r:t} \in \mathbb{R}^{N \times r}$, which observe their previous $r$-steps prior to current time step $t$ to predict about the future values for the next $r$-steps $X_{t+1:t+r} \in \mathbb{R}^{N \times r}$.

### 2.2 Missing Data Imputation
We utilize a binary mask matrix $M \in \{0,1\}^{N \times T}$, where $M_{i,j}$ indicates that the value $X_{i,j}$ is missing, and $M_{i,j} = 1$ indicates that the value is observed in the data matrix $X \in \mathbb{R}^{N \times T}$. Missing data can follow random or block patterns[6, 24, 27] across the $N$ univariate time series and $T$ timestamps. We utilize observed values $X_{obs} = X \odot M$ to estimate the missing values $X_{miss} = X(0 - M) \odot \text{element-wise multiplication}. We utilize a sliding window of size $r$ over the observed samples $X_{obs}$ to construct subsequences $S_{obs} = X_{t-r:t} \in \mathbb{R}^{N \times r}$, which have been observed over previous $r$-steps prior to the current time step $t$. These observed samples are used to predict the missing values for the next $r$-steps, $S_{miss} = X_{miss}^{t+1:t+r} \in \mathbb{R}^{N \times r}$ by leveraging spatio-temporal dependencies within the data.

### 2.3 Anomaly Detection
Assuming the time series dataset exhibits normal behavior during the initial training timesteps, any pattern deviating from the normal behavior in subsequent timesteps $t > T_{train}$ is anomalous. Data observed after training is considered the test dataset. We use a sliding window to construct samples from previous time steps $S_{test} \in \mathbb{R}^{N \times r}$ to predict future values of multiple time series $S_{t+1} \in \mathbb{R}^{N \times r}$. The
```

### --- Page 0003 ---

```markdown
# Agentic Reinforced-Augmented Generation for Time Series Analysis

30th, ACM KDD August 25 - 29, 2024, Barcelona, Spain

---

framework predictions are denoted by $S^{+1} \in \mathbb{R}^{N \times V}$. In the unsupervised anomaly detection task, it computes the robust normalized anomaly scores $A^{+1}_{t}$ for each variable $i$ across the time steps in the training set $T$. This information regarding the variables helps in accurately localizing the anomalies in the test set.

We compute the simple moving average of the maximum value of the anomalous scores $A^{+1}_{t}$ across the multiple variables at time point $t + 1$ over the validation set as given,

$$
T_h = \max_{t' \in [t - \tau_{wd}, t - 1]} \frac{1}{\tau_{wd}} \sum_{i \in [N]} \max(A^{+1}_{t'}) \tag{1}
$$

where $\tau_{wd}$ denotes the number of points in the moving average calculation. $\tau_{val}$ denotes the time points in the validation set. We set the anomaly detection threshold $Th$ as the moving averaged maximum anomaly value for time $t + 1$, $A^{+1}$ over the validation data. During inference, time points with an anomaly score above the threshold were flagged as anomalies.

## 2.4 Classification

We perform unsupervised $K$-means clustering, identifying $(K)$ corresponding temporal clusters or regimes and assigning cluster labels $C \in \mathbb{R}^T$ to each time point in the series $X \in \mathbb{R}^N$. Then, a sliding window approach is employed to predict the cluster labels for the next $t$ steps $S^{t} = X^{t+1}...X^{t+k} \in \mathbb{R}^N$ over the previous $t$ time steps.

## 3 PROPOSED METHOD

The proposed framework offers a novel way to enhance time series by leveraging a hierarchical, multi-agent architecture. It comprises a prompt vector that coordinates specific tasks such as forecasting, anomaly detection, or imputation. These sub-agents employ pre-trained language models and utilize prompt-based information as latent knowledge bases. By retrieving relevant prompts from the pool, the sub-agents can augment their predictions with contextual knowledge about related past patterns, enabling them to adapt to complex time series data. The framework's modular design, combined with the stretching of individual sub-agents, allows for improved performance across various time series analysis tasks, surpassing the limitations of traditional fixed-window methods.

### 3.1 Dynamic Prompting Mechanism

Current time series methods typically utilize past data within a predefined window length to understand historical trends and predict task-specific outcomes. However, this approach may not be optimal because there is no universally ideal window length for all time series data. A larger window length might obscure short-range dependencies, while a smaller window length might fail to capture long-range dependencies. Existing methods fail to capture the full complexity of diverse trends and patterns within the complex data required for accurate time series modeling. Adjusting the window length in real-world scenarios can be challenging and computationally expensive. Achieving this goal is ambitious and task, given the current state of research in this field. To address the challenges of non-stationarity and distributional shifts in real-world data, we utilize a differentiable dynamic prompting mechanism[3]. This mechanism allows traditional time series methods to access related past knowledge by retrieving the same group of prompts from the prompt pool for effective adaptive learning on new, untrained input data. The dynamic prompting approach utilizes a shared pool of prompts stored as key-value pairs. For time series applications, each prompt is represented by a key vector encoding the essential global characteristics associated with that prompt. The corresponding value matrix contains specific knowledge related to those trends or patterns, such as seasonality, cyclicality, irregularities, and other effects. The key vector acts as an identifier or query vector to retrieve relevant prompts from the pool based on similarity to the input new data, providing a form of conditioning or context about historical patterns to enhance the predictions. This allows the time series methods to effectively leverage encoded knowledge from past experiences, enhancing their predictions by recognizing and applying learned patterns from the shared prompt pool to the new input data. The pool of prompts $\mathcal{P} = \{(k_1, v_1), (k_2, v_2), \ldots, (k_M, v_M)\}$, where $M$ is the total number of prompts in the pool, $k_m \in \mathbb{R}^d$ is the key vector of the $m$-th prompt, and $v_m \in \mathbb{R}^k$ is the corresponding prompt value matrix with length $l$ and dimensionality $d$. In order to retrieve the most relevant prompts for a given input time series $S^{t} = X^{t-k} \in \mathbb{R}^e$, we first linearly project it into a $d$-dimensional embeddings $S^{t} \in \mathbb{R}^d$. We then utilize a score-matching function to measure the similarity between the input and each

$$
\gamma(S^{t}, k_m) = \frac{S^{t} \cdot k_m}{\|S^{t}\|} \tag{2}
$$

where $k_m \in \mathbb{R}^{K \times d}$ is the key vector and the top-$K$ prompts with the highest similarity scores are selected, where $1 \leq K \leq M$. Let $j = h_1, \ldots, h_S$ be the list of indices corresponding to the top-$K$ most relevant prompts retrieved from the pool for the input time series $S^{t}$. The selected prompts, along with the original input, are concatenated to form the augmented input $S^{t}_{aug}$:

$$
S^{t}_{aug} = [S^{t}; v_{h_1}; \ldots; v_{h_K}] \tag{3}
$$

where $S^{t} \in \mathbb{R}^{K \times d}$ is the linearly project of $S^{t}$ to $d$-dimensional representation as follows:

$$
S^{j} = W_{s} \tag{4}
$$

where $W_{s} \in \mathbb{R}^{K \times d}$ is a learnable weight matrix. In summary, it aims to improve the time series modeling efficiency on the task-specific performance by allowing the framework to recognize and apply learned patterns across non-stationarity datasets with distributional shifts via the shared prompt representation pool.

### 3.2 Fine-Tuning/Preference Optimization SLMs

Current pretrained SLMs, such as Google’s Gemma and Meta’s Llama-3 models, are designed with a context length of 8K tokens. However, they struggle to process long input sequences that exceed their pretraining context window. This is because the limited length of the context window during pretraining restricts their effectiveness during inference when dealing with longer texts. SLMs with an improved context length can better capture long-term spatio-temporal dependencies and capture patterns that unfold over extended periods, which is essential for accurate predictions and understanding seasonal or cyclic trends. We build upon recent work [19] to improve how SLMs handle long sequences without finetuning. A two-tiered attention mechanism (grouped and neighbor

![Detailed description of the chart](assets/page_0003_img_1.png)
```

### --- Page 0004 ---

```markdown
# 38 different attack scenarios. In addition, we discuss the univariate datasets for forecasting and imputation in the technical appendix.

| Dataset     | Sensors | Timesteps | Time Range   | Data Split | Granularity |
|-------------|---------|-----------|--------------|------------|-------------|
| PeMSDB     | 338     | 26,208    | 09/12 - 11/2018 |            |             |
| PeMSDB     | 307     | 16,992    | 01/2018 - 02/2018 |            |             |
| PeMSDB     | 383     | 28,264    | 05/2017 - 08/2017 | 6 / 2 / 2  |             |
| PeMSDB     | 170     | 17,856    | 07/2016 - 08/2016 |            |             |
| PeMSDB     | 228     | 12,675    | 05/2012 - 06/2012 |            |             |
| METRI-LA   | 207     | 34,272    | 03/2016 - 06/2016 |            |             |
| PeMS-BAY   | 325     | 32,116    | 01/2017 - 05/2017 | 1 / 1 / 2  |             |

Table 1: Summary of the spatio-temporal datasets.

| Dataset     | SWAT | WADI | SMAP | MSL | TEP | HAI |
|-------------|------|------|------|-----|-----|-----|
| Sensors     | 51   | 25   | 55   | 52  | 25  | 55  |
|             |      |      |      |     |     |     |

Table 2: Statistical summary of benchmark datasets. $t$ is the length of the subhypotheses or historical window length.

**Evaluation Metrics:** For forecasting and imputation tasks, the performance of the proposed framework is evaluated using MAE, RMSE, and MAPE metrics on the original scale of the time series data. For classification tasks, we use accuracy. For anomaly detection, we utilize the standard evaluation metrics of precision (P), recall (R) in %, and F-score (F in %). We utilize a multi-metric approach for a fair comparison across the models. To do this, we compute the confusion matrix true positive (TP) for correctly identified anomalies, false negative (FN) for incorrectly identified anomalies, and false positive (FP) for normal points mistakenly identified as anomalies. Precision (TP / (TP + FP)) represents the proportion of correctly detected anomalies among all identified anomalies, while recall (TP / (TP + FN)) represents the proportion of actual anomalies that were correctly detected. The F1-score is calculated as the harmonic mean of precision and recall. The threshold for identifying anomalies is set to the highest anomaly score (for select datasets, 2.3) from the validation set. For the SWAT and WADI datasets, which contain continuously evolving time series, we adopt the point adjustment strategy [36, 51] to flag the entire subsequence as an anomaly if the model predicts one. On the Tennessee Eastman dataset, we utilize the Fault Detection Rate (FDR in %), defined as the ratio of the number of faults detected to the total number of faults that occur, to evaluate the effectiveness of our framework.

**Experimental Settings:** To reduce memory footprint and computational complexity, we segment the time series datasets using a sliding window technique with a predefined historical window size to obtain time series subsequences (smaller, overlapping sequences of a fixed length). We perform instruction-tuning (fine-tuning) of the small-scale language models, such as Self-Extended-Instruct LLaMA-3 [24], Gemini-2B, and Gemini-7B models using the PET technique [4] such as QLoRA[21], on the predicted associated time series tasks using corresponding datasets. We set the following hyperparameters: a batch size of 16, a sequence length of 32K, a learning rate of 1e-5, training for 15 epochs, 50 warmup steps, a weight decay of 0.01, and a gradient accumulation of 2 steps. We used the AdamW optimizer [25] and a linear scheduler to adjust the learning rate during training. We utilized a 4-bit quantization for QLoRA. The QLoRA hyperparameters include the low-rank of
```

### --- Page 0005 ---

```markdown
# Agentic Retrieval-Augmented Generation for Time Series Analysis
30th, ACM KDD August 25 - 29, 2024, Barcelona, Spain

## Table 3: The table compares various methods for 12-sequence-to-12-sequence forecasting tasks on benchmark datasets using multiple evaluation metrics. These methods use 12 past sequences to predict the next 12 sequences.

| Methods | PeMSD3 | PeMSD4 | PeMSD7 | PeMSD5 | PeMSD7D |
|---------|--------|--------|--------|--------|---------|
| HA      | 3.156  | 5.299  | 3.778  | 3.502  | 27.48   | 4.512  | 6.564  | 9.514  | 5.924  | 27.83  | 8.63   | 14.35  |
| ARIMA   | 7.241  | 4.371  | 3.778  | 3.834  | 3.181  | 5.972  | 19.14  | 22.73  | 17.23  | 12.63  | 12.38  | 10.28  |
| VAR     | 2.133  | 3.511  | 2.233  | 2.767  | 15.29  | 9.494  | 13.20  | 29.37  | 19.17  | 4.16   | 7.51   | 10.10  |
| TCN     | 19.827 | 32.45  | 23.68  | 23.87  | 13.50  | 31.03  | 12.80  | 30.82  | 12.83  | 4.73   | 9.44   | 9.14   |
| GRU-ED  | 21.945 | 34.55  | 22.97  | 13.36  | 9.141  | 14.76  | 21.46  | 12.96  | 18.76  | 8.78   | 8.75   | 8.78   |
| DCRNN   | 17.550 | 30.42  | 17.314 | 13.393 | 33.93  | 11.670 | 27.19  | 18.36  | 19.76  | 8.79   | 8.79   | 8.79   |
| STGCN   | 17.903 | 30.31  | 18.344 | 14.177 | 25.32  | 13.61  | 11.62  | 26.26  | 10.29  | 7.81   | 7.81   | 7.81   |
| GraphWaveNet | 19.12 | 32.77 | 18.39 | 29.36 | 19.39 | 11.97 | 18.25 | 12.15 | 3.19 | 6.21 | 6.20 | 8.02 |
| ASTGCN  | 17.934 | 26.129 | 25.322 | 24.07 | 37.07 | 10.73 | 28.06 | 11.63 | 31.41 | 6.18 | 8.12 | 8.12 |
| MSTGCN  | 19.03  | 29.31  | 25.36  | 18.37 | 47.76 | 20.10 | 20.71 | 7.83  | 10.25 | 5.68 | 7.42 | 7.42 |
| LSTGCN  | 17.904 | 29.28  | 16.85  | 21.53 | 11.83 | 21.16 | 19.73 | 26.76 | 11.20 | 5.68 | 7.42 | 7.42 |
| AGCRN   | 16.58  | 28.25  | 12.52  | 12.97 | 36.55 | 12.95 | 15.25 | 9.50  | 5.94  | 7.20 | 5.94 | 7.20 |
| STRGN   | 16.747 | 16.30  | 24.56  | 16.77 | 36.60 | 9.21  | 12.54 | 10.29 | 2.73  | 5.29 | 5.73 | 7.73 |
| Z-GCN   | 16.64  | 21.79  | 15.69  | 12.78 | 31.97 | 9.25  | 15.81 | 15.25 | 2.48  | 2.48 | 2.48 | 2.48 |
| ST-EDGE | 15.00  | 15.00  | 12.96  | 20.45 | 31.97 | 8.25  | 15.81 | 15.25 | 2.48  | 2.48 | 2.48 | 2.48 |

![Table comparing various methods for 12-sequence-to-12-sequence forecasting tasks](assets/page_0005_img_1.png)

## Table 4: The table compares the performance of various forecasting methods on the METR-LA and PeMS-BAY benchmark datasets using multiple evaluation metrics. All methods use 12 past sequences to predict 3, 6, or 12 future sequences.

| Datasets | Methods | MAE  | RMSE | MAPE | MARE | RMSE | MAPE |
|----------|---------|------|------|------|------|------|------|
| METR-LA  | HA      | 8.45 | 3.99 | 9.30 | 10.37 | 5.05 | 12.10 |
|          | SVR     | 8.45 | 3.99 | 9.30 | 10.37 | 5.05 | 12.10 |
|          | FC-ISTM | 6.30 | 3.44 | 9.60 | 7.73  | 7.00 | 6.47  |
|          | DCRNN   | 5.38 | 2.77 | 7.50 | 6.45  | 2.91 | 9.34  |
|          | STGCN   | 5.74 | 2.88 | 7.62 | 7.47  | 9.40 | 12.70 |
|          | GraphWaveNet | 5.15 | 2.69 | 6.92 | 6.24 | 7.37 | 7.50 |
|          | ASTGCN  | 9.27 | 6.61 | 10.71 | 10.39 | 11.52 | 11.61 |
|          | JSTGCN  | 7.82 | 3.81 | 6.96 | 7.71  | 5.13 | 12.61 |
|          | MTGN    | 5.55 | 2.80 | 6.41 | 6.19  | 3.78 | 7.34  |
|          | DMAN    | 5.55 | 2.80 | 6.41 | 6.19  | 3.78 | 7.34  |
| PEMS-BAY | DCRNN   | 2.95 | 1.35 | 2.90 | 1.74  | 3.40 | 2.70  |
|          | STGCN   | 2.96 | 1.36 | 2.97 | 1.81  | 5.49 | 5.79  |
|          | GraphWaveNet | 2.74 | 1.30 | 2.73 | 1.63 | 4.52 | 4.95 |
|          | ASTGCN  | 3.13 | 1.52 | 2.22 | 2.01  | 6.42 | 6.00  |
|          | JSTGCN  | 3.01 | 1.44 | 3.18 | 1.39  | 5.17 | 2.66  |
|          | MTGN    | 2.79 | 1.32 | 2.77 | 1.63  | 4.94 | 1.43  |
|          | DMAN    | 2.81 | 1.29 | 2.68 | 1.53  | 4.12 | 4.37  |
|          | DCRNN   | 2.81 | 1.08 | 1.68 | 2.61  | 1.31 | 1.12  |
|          | SelfExtend-Agentic-RAG W/Gemma-2B | 1.78 | 0.62 | 1.68 | 2.61 | 1.31 | 1.12 |
|          | SelfExtend-Agentic-RAG W/Gemma-1B | 1.62 | 0.81 | 1.63 | 2.52 | 1.21 | 2.31 |
```


### --- Page 0006 ---

```markdown
| Methods            | P(%)  | R(%)  | F(%)  | P(%)  | R(%)  | F(%)  | P(%)  | R(%)  | F(%)  | P(%)  | R(%)  | F(%)  | P(%)  | R(%)  | F(%)  | P(%)  | R(%)  | F(%)  |
|--------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| GAN-I               | 81.03 | 84.97 | 82.72 | 80.63 | 75.71 | 77.16 | 75.79 | 71.02 | 73.86 | 78.13 | 78.36 | 78.25 | 78.76 | 78.13 | 78.45 | 78.76 | 78.13 | 78.45 |
| LSTM-NMT           | 79.12 | 75.08 | 77.25 | 78.54 | 75.86 | 77.15 | 81.96 | 83.64 | 82.80 | 59.34 | 57.26 | 58.29 | 62.46 | 23.65 | 34.20 | 72.57 | 21.75 | 33.12 |
| MTAD-GAT           | 82.01 | 76.78 | 79.25 | 76.82 | 72.57 | 74.67 | 82.96 | 81.94 | 82.45 | 81.57 | 80.27 | 80.92 | 75.27 | 23.81 | 36.12 | 81.57 | 80.27 | 80.92 |
| MAD-GAN            | 89.07 | 78.64 | 83.47 | 77.04 | 71.04 | 74.02 | 82.96 | 81.94 | 82.45 | 81.57 | 80.27 | 80.92 | 75.27 | 23.81 | 36.12 | 81.57 | 80.27 | 80.92 |
| VAR                | 93.55 | 61.12 | 74.96 | 91.79 | 80.36 | 85.92 | 86.42 | 84.27 | 85.32 | 81.87 | 86.41 | 84.12 | 91.04 | 91.11 | 91.07 | 91.04 | 91.11 | 91.07 |
| LSTNet             | 72.15 | 65.34 | 68.52 | 57.02 | 61.53 | 59.25 | 96.30 | 91.72 | 93.01 | 61.87 | 61.37 | 61.62 | 81.37 | 29.15 | 42.59 | 81.37 | 29.15 | 42.59 |
| Deep-SVDD          | 80.42 | 84.54 | 82.49 | 74.18 | 73.24 | 73.71 | 86.54 | 85.62 | 86.08 | 69.61 | 68.93 | 69.26 | 66.33 | 31.35 | 42.77 | 68.33 | 31.35 | 42.77 |
| DADAM              | 89.52 | 82.74 | 86.13 | 74.73 | 74.36 | 74.55 | 86.54 | 85.62 | 86.08 | 69.61 | 68.93 | 69.26 | 66.33 | 31.35 | 42.77 | 68.33 | 31.35 | 42.77 |
| MAMCD              | 82.92 | 69.20 | 74.63 | 69.82 | 67.82 | 68.82 | 81.62 | 61.73 | 70.36 | 58.26 | 34.26 | 43.77 | 71.56 | 32.46 | 42.37 | 71.56 | 32.46 | 42.37 |
| MRC                | 81.69 | 62.87 | 71.67 | 63.67 | 62.43 | 63.05 | 73.81 | 73.15 | 73.48 | 75.45 | 72.80 | 74.10 | 75.45 | 72.80 | 74.10 | 75.45 | 72.80 | 74.10 |
| LSTF               | 63.11 | 51.50 | 56.63 | 69.07 | 67.25 | 68.15 | 72.38 | 73.85 | 73.11 | 73.42 | 73.11 | 73.26 | 73.42 | 73.11 | 73.26 | 73.42 | 73.11 | 73.26 |
| LSTIVAE            | 74.00 | 58.20 | 66.79 | 71.45 | 62.92 | 66.75 | 78.05 | 78.94 | 78.99 | 74.92 | 76.82 | 75.87 | 72.84 | 37.54 | 49.12 | 72.84 | 37.54 | 49.12 |
| BeanGAN            | 81.00 | 78.00 | 79.99 | 72.92 | 69.85 | 71.36 | 69.87 | 67.34 | 68.59 | 69.87 | 67.34 | 68.59 | 69.87 | 67.34 | 68.59 | 69.87 | 67.34 | 68.59 |
| OmniAnomaly        | 81.00 | 78.00 | 79.99 | 72.92 | 69.85 | 71.36 | 69.87 | 67.34 | 68.59 | 69.87 | 67.34 | 68.59 | 69.87 | 67.34 | 68.59 | 69.87 | 67.34 | 68.59 |
| Interface          | 80.99 | 85.80 | 83.90 | 81.27 | 80.27 | 80.77 | 82.96 | 81.94 | 82.45 | 81.57 | 80.27 | 80.92 | 75.27 | 23.81 | 36.12 | 81.57 | 80.27 | 80.92 |
| THIE              | 89.24 | 83.15 | 86.10 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 | 81.13 |
| Agentic-RAG WGemma-2B | 95.80 | 92.55 | 94.16 | 92.58 | 91.85 | 92.21 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 |
| Agentic-RAG WGemma-7B | 97.94 | 95.15 | 96.54 | 99.31 | 99.31 | 99.31 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 | 99.81 |
| Best performance in bold. Second-best with underlined. Agentic-RAG framework variant. |

| Base Model         | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    | 11    | 12    | 13    | 14    | 15    | 16    | 17    | 18    | 19    | 20    |
|--------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| TCN                | 99.61 | 95.48 | 94.53 | 97.71 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 |
| RNN                | 99.67 | 95.48 | 94.53 | 97.71 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 |
| MTAD-GAT           | 99.78 | 98.71 | 96.10 | 98.97 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 | 97.12 |
| LERGEN             | 99.67 | 96.10 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 | 95.12 |
| Agentic-RAG WGemma-2B | 99.66 | 99.12 | 97.05 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 |
| Agentic-RAG WGemma-7B | 99.96 | 99.12 | 97.05 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 | 99.30 |
| Best performance in bold. Second-best with underlined. |

```


### --- Page 0007 ---

```markdown
# REFERENCES

[1] Alhaija, M. E. M., & Micallef, C. (2024). Models 3 Model Card. https://github.com/meta-lambda/blob/main/MODEL_CARD.md

[2] W. C. Boehm, Benjamin Nolen, Nick Rydell, Malinda Subhiabi, Jared D. Kaplan, Pravalabh V. Bairagi, Arvind Neelakantan, Praveen Shyam, Girish Aswani, Amand A. Askell, et al. 2020. Language models as few-shot learners. Advances in Neural Information Processing Systems 33 (2020), 1877–1901.

[3] Wei Cao, Dong S. Serban, O. T. Zhang, Peter T. Zingg, Wen Ye, and Yan Liu. 2021. TEMPO: Prompt-based Generative Time-series Transformer for Time Series Forecasting. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=7M4sYt20u1

[4] Wei Cao, Dong Wang, Jian L. Hao Zhou, Li Li, and Yitian Li. 2018. Bridging recurrent imputation for time series. Advances in neural information processing systems 31 (2018).

[5] Chon Chen, Kerttari Petty, Alexander Skabardonis, Pravin Vangari, and Zhandeng Jia. 2021. Foreway performance measurement system: an online loop detector data. Transportation Research Record 1748, 1 (2001), 90–96.

[6] Zekai Chen, Disong Chen, Xina Zhang, Zuan Yuan, and Yuntian Cheng. 2021. Learning graph structures with Transformer for multivariate time series prediction. In IEEE International Conference on Data Mining (ICDM).

[7] Jenyah Choudhury, Chao Liu, Evelyn Huang, and Dong Park. 2022. Graph neural controlled differential equations for traffic forecasting. In Proceedings of the 41st International Conference on Machine Learning. PMLR 139, 5646–5674.

[8] Paul F. Christiano, Jan Leike, Tom B. Martens, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems 30 (2017).

[9] Andrew C. Choi, Ivan Marisa, and Ceaser Alippi. 2021. Maritime Time Series Anomaly Detection. arXiv preprint arXiv:2103.02741.

[10] Amanda C. Choi, Ivan Marisa, Deni Zambo, and Ceaser Alippi. 2021. Anomaly detection for graph-based spatiotemporal forecasting. In Proceedings of the 2021 IEEE International Conference on Data Science and Advanced Analytics.

[11] Yifan Ding, Ruijie Huo, Zhiqiang Li, and Holtzman, L. K. 2022. Graph neural network-based anomaly detection for time series. Proceedings of the 44th Annual Conference on Artificial Intelligence.

[12] Y. Feng and F. Z. Wang. 2022. MSL-Supervised Masked Anomaly Detection for Multivariate Time Series. arXiv preprint arXiv:2206.02260.

[13] Max Greve, Mark T. Sijbrandij, and G. W. Wijsman. 2022. Graph neural networks for time series forecasting. Advances in Neural Information Processing Systems 36.

[14] Han Guo, Philippe Gengard, Eric P. King, and K. J. Ross. 2023. Low-rank plus diagonal matrix decomposition for efficient language model fine-tuning. arXiv preprint arXiv:2301.12332 (2023).

[15] Eric H. Han, Jiyang Liu, and Qian Chen. 2022. Parameter-efficient fine-tuning for large models: A comprehensive survey. arXiv preprint arXiv:2203.14608 (2022).

[16] K. H. Hwang, Y. Zhang, Philip A. Yelena Allen-Zhu, Xuanli Shi, Bing Wang, L. Wang, and Weizhi Chen. 2021. Low-rank adaptation of language models. arXiv preprint arXiv:2202.08865 (2021).

[17] L. K. Kordzadeh. 2021. Constrained convolutional layer for long short-term memory networks. In Proceedings of the 28th ACM SKGDD International Conference on Knowledge Discovery & Data Mining, 387–395.

[18] Hongjie Liu, Xiaotian Zhang, Jingfeng Wu, Zhenming Liu, Chao-Yuan Cheng, Huijuan Chen, and K. H. Zhao. 2021. Image language: efficient slim context window without tuning. arXiv preprint arXiv:2102.01325 (2021).

[19] Ming Jin, Shiyang Wang, Lintao An, Zhizhang Chen, J. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shunli Pan, et al. 2023. Time-series forecasting by reprogramming large language models. arXiv preprint arXiv:2301.02704 (2023).

[20] Michael Leclerc. 2021. Promotional analysis and forecasting for demand planning: a practical time series approach, with exhibits (2021).

[21] Yequan Li, R. S. S. Shabaz, and Yan Liu. 2023. Distillation for Convolutional Recurrent Neural Networks: Dual-Temporal Forecasting. In Proceedings of the 41st International Conference on Machine Learning.

[22] Xi Victoria Li, Xuefen Chen, Minghao Chen, Weiqi Sha, Larmel B. Kott, Reih, Hameed, and Rachael J. Kahn. 2023. Retro-grade augmented action imitation tuning. arXiv preprint arXiv:2303.07842.

[23] Hengbo Liu, Ziyang Ma, Lijuan Yang, Tian Zhou, Rui Xi, Yi Wang, Qingsong Zhang, and Shuang Wu. 2023. Self-Adaptive Decomposed Interpretative Framework for Electric Load Forecast Under Extreme Events. In IEEE International Conference on Acoustics, Speech and Signal Processing. arXiv preprint arXiv:2111.05101 (2021).

[24] Ivan Marisa, Ceaser Alippi, and Filippo Maria Bianchi. 2022. Graph-based Forecasting with Missing Data through Suboptimal Downsampling. arXiv preprint arXiv:2204.02063.

[25] Ivan Marisa, Ceaser Alippi, and Ceaser Alippi. 2022. Learning to reconstruct time series from partial data and graphs with sparse observations. Advances in Neural Information Processing Systems 35 (2022), 3029–3042.

[26] Yugi Niu, Min H. Nguyen, Phawadee Srichai, and Jyanat Kalangjam. 2023. A Time Series for Kids: Long-term Forecasting with Transformers. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=IdcYf0T0

[27] R. N. 2023. CTF Technical Report. arXiv:2303.6783 (2023).

[28] Boris O’Neal, Dmitri Cerenkov, Nicholas Choudhary, and Yousuf Benjoo. 2022. BEATS: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations.

[29] Jadeep Pathak, Shashank Subramanian, Rex Hartsough, Sangeeta Raja, Abhisek Chattopadhyay, Moreta Mandru, Thorsten Kurth, David Hall, Zongyi Liu, and K. A. A. 2022. Forecasting with a global adaptive hybrid model: a practical approach for neural operators. arXiv preprint arXiv:2202.11211 (2022).

[30] Eduardo R. Ralha, G. Shafran, Eric Mitchell, Christopher D. Manning, Stefano Eraldo, and Francesca Franza. 2022. Direct preference prediction for multi-modal is secretly a neural model. Advances in Neural Information Processing Systems 36 (2022).

[31] Shreyas R. Reddy, Laviya, and Danyel D. Mulhagy. Anom Shuhan, Kevin Leyton-Brown, and Thomas Leibig. 2022. Contextually unaligned language models. Transactions of the Association for Computational Linguistics 10 (2022), 134–161.

[32] Hadi Shiri, Kalyan Suryadevara, and Matthew Leikly. Timothy Leikly, Jean-Baptiste A. P. Boulard, Fabio Suraci, Angeliki Lazaridou, Urban Frant, and M. K. 2021. Contextualized embeddings for multimodal understanding. arXiv preprint arXiv:2104.02101.

[33] Yuxin Shen, Zhenzhen Zhang, and James Newk. 2020. Anomaly detection for time series data using deep learning. In Proceedings of the 2020 IEEE International Conference on Data Mining.

[34] J. S. T. 2023. Repeated Retinal Anomaly Detection. arXiv preprint arXiv:2301.12345.

[35] J. T. 2021. Time series anomaly detection with stochastic clustering. arXiv preprint arXiv:2102.02126 (2021).

[36] Shunyan Wu, Jeffrey Zhan, Dinu Yan, D. Shakthikanth Narasimhan, and K. R. 2022. Robust Graph reasoning and learning with time series. arXiv preprint arXiv:2201.06240.

[37] Xin Xu, Q. Zhang, W. Ren, H. Zhao, H. Liang, H. Pengyang Wang, An, Long, and Zhangdong Guo. 2022. A graph perspective. Advances in Neural Information Processing Systems 36 (2022).

[38] Jinsong Yoon, Daniel Pardi, and A. W. Schaar. 2019. Time series forecasting with recurrent neural networks. Advances in neural information processing systems 32 (2019).
```

### --- Page 0008 ---

```markdown
30th ACM KDD August 25 - 29, 2024, Barcelona, Spain  
Chidhalak Ravuru, Sagar Srinivas Sakhinna, and Venkataramana Runkana  

| Reference                                                                                                           |
|---------------------------------------------------------------------------------------------------------------------|
| [48] Tianping Zhang, Yuhong Wei, Ce Yu, Bian Xiaohan Yi, Shun Zheng, and Jian Li. 2022. Less is more: Fast multivariate time series forecasting with lightweight sampling methods. arXiv preprint arXiv:2207.01136 (2022). |
| [49] Weijie Zhang, Chen Zhang, and Fuget Xu. 2022. GREEN: Multivariate Time Series Forecasting from the Perspective of Graph Relational Learning. In PAKDD. 239–250. |
| [50] Yunchao Zhang and Junichi Yamato. 2022. Crossformer: Transformer utilizing cross-dimension decomposition for multivariate time series forecasting. In The eleventh international conference on learning representations. |
| [51] Jiang Zhao, Yujing Wang, Jianping Guo, Guifeng Huang, Defu Cao, Yuhui Tong, Bixiong Chen, Jiajin Bai, Jie Tong, and Qihang Zhang. 2020. Multivariate time-series anomaly detection via graph attention network. In 2020 IEEE International Conference on Data Mining (ICDM). 841–850. |
| [52] Helen Zhou, Lector A. O’Rourke, and Xitian Wang. 2023. Business Metric-Aware Forecasting for Inventory Management. arXiv preprint arXiv:2302.13113 (2023). |
| [53] Haoxi Zhou, Shangzhang Zhang, Jieqi Sheng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wenzhao Zhang. 2022. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35, 11106–11115. |
| [54] Jiahui Zhu, Shoujie Hu, Haoyu Lin, Jingwen Chen, and Wenchao Meng. 2024. Self-adaptive multivariate time series anomaly detection. IEEE Transactions on Knowledge and Data Engineering (2024). |
| [55] Tianhong Zhang, Yingdong Wen, Xue Wang, Liang Sun, and Rong Jin. 2022. FEDformer: Forecasting with Exponential Decay Transformer for long-term series. In 2022 International Conference on Machine Learning (ICML). 2022, Baltimore, Maryland. |
| [56] Shuang Zhang, Nii Lante Sunkwa, Rong Jin, et al. 2024. One fits all: Power prediction for time series analysis by pretrained lm. Advances in neural information processing systems. |
| [57] Jian Zhang, Yuhong Wei, Liang Sun, and Rong Jin. 2023. On the Power of Pre-trained General Time Series Analysis by Pretrained LM. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=G5v6FvZ7m2 |
```

### --- Page 0009 ---

```markdown
# A MULTIVARIATE SPATIO-TEMPORAL DATASETS

## A.1 Missing Data Imputation

Time series imputation is a critical step in time series analysis. It addresses a common issue in this field: missing values within datasets. These missing values can arise from sensor failures, data transmission errors, or incomplete records. By imputing these gaps, time series imputation ensures the quality and reliability of subsequent analyses. The Agentic-RAG framework achieves this by handling seasonality, trends and capturing the inherent spatio-temporal dependencies within the data. Ultimately, imputation improves data quality, enabling more accurate analysis, modeling, and decision-making. In essence, it plays a vital role by maintaining data integrity and enabling reliable analysis. To evaluate the Agentic-RAG framework's ability to handle missing data, we simulated two types of missingness patterns: point missing and block missing \cite{9}. These patterns represent varying degrees of data availability. To achieve this, we introduced synthetic missingness into time series datasets following these patterns. For point missing, individual values were randomly omitted with a probability threshold ($p$), controlling the overall percentage of missing data. The block missing pattern involves removing contiguous, multi-period, multi-time series segments. This is done by randomly selecting start and end times, as well as start and end time series, to define uniform blocks with an average length of ($\ell$). All data points within each block are removed. Furthermore, two block missing patterns were considered: temporal and spatial. For temporal block missing, contiguous multi-period segments are removed from a given time series. This is done by randomly selecting start and end times, creating stretches of available temporal data. For spatial block missing, contiguous blocks are removed across multiple related time series at specific time points. This involves randomly selecting the start and end time series, resulting in missing spatial data at these time points. Both patterns show varying levels of missing information in the time series data. In summary, point missing refers to sporadic gaps in data, while block missing involves the absence of entire contiguous multi-period and multi-series segments. Block missing can further be categorized into two types: temporal block missing, where contiguous segments are removed within a single time series, and spatial block missing, where contiguous blocks are removed across multiple related time series, mimicking realistic scenarios of faulty data collection. In the context of time series imputation, “in-sample” and “out-of-sample” imputation refer to distinct evaluation settings. In-sample imputation involves the imputation method reconstructing missing values within a given fixed input sequence, $s'$, using all available observed data within that subsequence. Out-of-sample imputation involves training the imputation method using the fixed sequence $s'$ to impute missing points in a future sequence, $s^{*}$. In this work, we utilize out-of-sample settings, as this approach mimics real-world scenarios and rigorously assesses the Agentic-RAG framework's robustness and generalizability by evaluating its ability to handle new, unseen data. The simulated datasets with missing values were then used to evaluate the missing data handling capabilities of the proposed Agentic-RAG framework. We split multiple benchmark datasets in chronological order with a ratio of 7:1:2 for the METR-LA and PEMS-BAY datasets and a ratio of 6:2:2 for the other datasets into training, validation, and test sets. We evaluated the Agentic-RAG framework's performance on simulated data using multiple imputation metrics (e.g., RMSE, MAE, and MAPE). This analysis helps us understand how well the framework handles time series data with missing values, particularly how its performance changes as the percentage of missing data increases. We established the Agentic-RAG framework, trained on complete data (no missing values), as a strong performance benchmark. This benchmark allows us to evaluate the framework's effectiveness in imputing missing data under different conditions of data incompleteness. Tables 7 and 8 present the imputation results on standard benchmark datasets with different missingness patterns, while the framework performs slightly worse than the baseline for minimal missing data. Its accuracy degrades more significantly as the dataset becomes more incomplete, regardless of the specific missingness pattern. Our proposed Agentic-RAG framework demonstrates robustness to missing data by focusing on the available observations for imputing missing values, thereby avoiding the introduction of potentially inaccurate estimates that could obscure the underlying trends and patterns within the time series data. Additionally, the Agentic-RAG framework effectively captures the complex non-linear intra- and inter-time series dependencies and this leads to more reliable imputation. The experiments show that our framework can learn spatio-temporal dependencies from partially observed data with various missingness patterns, resulting in lower imputation errors.

## A.2 Time Series Classification

Time series classification is a crucial task with applications across various domains. In time series analysis, regression, or clusters represent distinct behavioral modes, operating conditions, or states of the system underlying the data. Identifying and characterizing these regimes is crucial for understanding the complex patterns and dynamics within the data. This allows for more accurate modeling, forecasting, and decision-making in applications where time series analysis is essential. The emergence of different regimes in time series can stem from changes in the data generation process, external conditions, or the inherent non-stationarity and multivariate nature of the time series. This reflects the rich information content and complexity often encountered in real-world time series data. To evaluate the proposed Agentic-RAG framework's ability to handle time series classification tasks, an unsupervised clustering approach was employed for data labeling. We first applied k-means clustering to the original time series datasets, determining the optimal number of clusters ($k$) using established techniques such as the elbow method or silhouette analysis. The optimal clusters were treated as class labels, representing distinct regimes within the time series, and each time series was assigned the corresponding cluster label, creating a labeled classification dataset. We adopted a time-based division strategy to split multiple benchmark datasets into training, validation, and testing sets. The METR-LA and PEMS-BAY datasets were split at a 7:1:2 ratio, while other datasets used a 6:2:2 split. We evaluated the framework's performance on the held-out test set using standard classification metrics: accuracy, precision, recall. This methodology allowed us to assess the framework's ability to learn the underlying patterns and relationships associated with
```


### --- Page 0010 ---

```markdown
# 30th ACM KDD August 25 - 29, 2024, Barcelona, Spain
Chidashak Ravura, Sagar Srinivasan, and Venkataramana Runkana

## Table 7: The table presents the Agentic-RAG framework's evaluation results on various metrics for missing data imputation across PeMSD3, PeMSD4, PeMSD7, and METR-LA benchmark datasets with diverse missing data patterns.

| Missing Scheme          | Missing Rate | PeMSD3                | PeMSD4                | PeMSD7                | METR-LA              |
|-------------------------|--------------|-----------------------|-----------------------|-----------------------|----------------------|
|                         |              | RMSE     | MAE     | MAPE    | RMSE     | MAE     | MAPE    | RMSE     | MAE     | MAPE    | RMSE     | MAE     | MAPE    |
| SelfExtend-Agentic-RAG w/Llama-8B | 0%           | 19.48    | 13.01   | 10.35   | 25.54   | 17.46   | 9.52    | 29.97   | 19.02   | 8.03    | 6.23    | 3.12   | 8.53    |
|                         | 10%          | 21.12    | 10.21   | 28.93   | 19.18   | 11.04   | 32.11   | 20.06   | 10.76   | 7.06    | 10.13   |      |      |
|                         | 30%          | 25.25    | 15.33   | 13.32   | 30.61   | 20.62   | 12.63   | 36.48   | 21.16   | 7.82    | 4.51    | 11.02  |      |
|                         | 50%          | 24.16    | 16.39   | 12.49   | 31.27   | 22.14   | 14.08   | 37.24   | 25.15   | 13.21   | 8.57    | 5.03   | 12.18  |
| Block                   |              | 10%       | 25.07   | 17.14   | 15.25   | 31.88   | 15.18   | 39.21   | 29.15   | 14.13   | 9.04    | 5.03   | 13.12  |
|                         | 30%          | 27.21    | 18.45   | 18.28   | 25.12   | 17.23   | 42.32   | 27.07   | 16.09   | 7.67    | 10.62   | 6.47   |      |
|                         | 50%          | 21.98    | 20.09   | 18.43   | 27.11   | 19.16   | 45.27   | 29.03   | 18.12   | 11.63   | 16.07   |      |      |
| Block                   | (Only Spatial) | 10%       | 30.54   | 15.93   | 31.92   | 21.19   | 23.09   | 35.15   | 22.14   | 12.61   | 8.02    | 4.53   |      |
|                         | 30%          | 27.35    | 17.25   | 16.36   | 23.65   | 15.12   | 38.25   | 24.19   | 14.21   | 5.02    | 13.13   | 7.12   | 14.52  |
|                         |              | 50%       | 24.62   | 16.32   | 23.41   | 18.02   | 35.15   | 42.18   | 16.78   | 7.48    | 8.74    | 5.08   | 12.59  |
| Block                   | (Only Temporal) | 10%       | 2.64    | 4.75    | 12.14   | 15.41   | 20.90   | 7.84    | 12.12   | 7.58    | 1.72    | 3.26   | 3.28   |
|                         | 30%          | 2.68    | 5.02    | 6.43    | 16.27   | 23.18   | 8.12    | 1.83    | 3.41   |      |      |      |
|                         | 50%          | 2.89    | 5.27    | 6.73    | 17.92   | 25.78   | 8.48    | 1.79    | 3.34   |      |      |      |
|                         | 30%          | 2.61    | 4.89    | 6.77    | 15.75   | 22.89   | 8.79    | 1.99    | 3.34   |      |      |      |
|                         | 50%          | 2.75    | 5.21    | 6.88    | 16.67   | 23.74   | 8.36    | 1.36    | 3.16   |      |      |      |
|                         |              | 10%       | 2.55    | 4.81    | 6.23    | 15.49   | 26.28   | 7.75    | 15.73   | 3.31   |      |      |      |
|                         | 30%          | 2.78    | 5.12    | 6.56    | 16.67   | 23.78   | 8.36    | 1.36    | 3.16   |      |      |      |
|                         | 50%          | 3.00    | 5.41    | 6.88    | 17.79   | 28.83   | 8.97    | 1.97    | 3.29   |      |      |      |
|                         |              | 10%       | 2.55    | 4.78    | 6.18    | 15.37   | 22.58   | 1.74    | 3.29   |      |      |      |
|                         | 30%          | 2.75    | 5.09    | 6.51    | 16.52   | 23.26   | 8.24    | 1.36    | 3.54   |      |      |      |
|                         | 50%          | 2.98    | 5.38    | 6.83    | 17.75   | 24.76   | 8.96    | 1.96    | 3.58   |      |      |      |

## Table 8: The table presents the performance of the Agentic-RAG framework in imputing missing data on the PeMSD7, PeMSD8, and PeMS-BAY benchmark datasets with various synthetic missing data patterns.

### B UNIVARIATE DATASETS
We conducted several experiments to evaluate the proposed Agentic-RAG framework variants: SelfExtend-Agentic-RAG with Gemma-2B, SelfExtend-Agentic-RAG with Gemma-7B, and SelfExtend-Agentic-RAG with Llama-8B, on the univariate datasets for multiple time series analysis tasks such as forecasting and imputation.

### B.1 Forecasting and Imputation
The ETT (Electricity Transformer) datasets, ETT1, ETT2, ETTm1, and ETTm2, are popular benchmarks used for evaluating and benchmarking univariate time series forecasting methods. They provide a challenging benchmark due to the presence of complex patterns, such as trends, seasonality, and irregularities, which are commonly found in real-world time series data. ETT1 and ETT2 are two hourly time series datasets containing observations of electricity transformers from two different locations. ETTm1 and ETTm2 are two monthly time series datasets containing observations of electricity transformers from two different locations. In this work, we utilize the ETT datasets to evaluate the Agentic-RAG framework for both forecasting and missing data imputation tasks. The Table 1 shows the performance of various methods on the multi-horizon forecasting task using a lookback window of size 512. It presents mean squared error (MSE) and mean absolute error (MAE) for nine models (GPT-4,57, PatchTST,28, TimesNet[42], FEDformer[55], LightTS[48], N-BEATS[30], Agentic-RAG w/Gemma-2B, Agentic-RAG w/Gemma-7B, and Agentic-RAG w/Llama-8B) across four datasets (ETT1, ETT2, ETTm1, ETTm2) at different time horizons (96, 192, 336, 720). This allows for a comprehensive analysis of forecasting accuracy and robustness of Agentic-RAG framework across varying prediction lengths. The performance of various methods for imputing missing data (point and block missing) and their effectiveness in out-of-sample imputation settings are compared in Tables 12 and 13. The evaluated methods
```

### --- Page 0011 ---

```markdown
| Dataset                       | PeMSD3         | PeMSD3         | PeMSD3         | PeMSD3         | METR-LA        |
|-------------------------------|----------------|----------------|----------------|----------------|----------------|
|                               | Accuracy       | Precision      | Recall         | Accuracy       | Precision      | Recall         |
| SelfExtend-Agentic-RAG/WGemma-2B | 91.235        | 95.548         | 90.375         | 92.915         | 93.048         | 92.915         |
| SelfExtend-Agentic-RAG/WGemma-7B | 92.126        | 90.791         | 93.231         | 92.947         | 91.015         | 91.015         |
| SelfExtend-Agentic-RAG/WLlama-8B | 93.015        | 91.961         | 92.319         | 92.875         | 93.054         | 91.942         |
| LSTM                          | 83.015         | 83.265         | 84.565         | 85.265         | 87.805         | 86.565         |
| MLP                           | 82.015         | 80.561         | 81.028         | 81.848         | 82.425         | 83.528         |

Table 10: The table shows the evaluation results of the Agentic-RAG framework variants performance on various metrics for time series classification on the PeMSD3, PeMSD4, PeMSD7, and METR-LA benchmark datasets.

---

### C. ENVIRONMENTAL IMPACT

Our Agentic-RAG framework training process, involving multiple variants running for extended periods, increases our energy consumption and carbon footprint. Accurate quantification of the carbon footprint of deep learning experiments is essential for promoting sustainable practices in artificial intelligence research and development. A crucial aspect of this endeavor is estimating the energy consumption and associated greenhouse gas emissions during the computationally intensive training processes. This is calculated by determining the Total Graphics Power (TGP), which represents the maximum power draw of the GPU, including the GPU chip itself and other components like memory and additional circuitry. For example, the NVIDIA P100 GPU has a TGP of 300 watts, while the NVIDIA T4 GPU has a TGP of 70 watts. By multiplying the TGP by the training time, we can estimate the energy consumption, which is then converted to carbon emissions using a region-specific carbon intensity factor. This factor accounts for the energy mix (coal, natural gas, renewables, etc.) used to generate electricity in the geographic area where the computations are performed. Considering a 725-GPU hours training experiment and using an estimated carbon intensity factor of 0.0007 metric tons CO2e per kWh for the year 2024 (for more information on the carbon intensity of electricity, you can visit CO2 Intensity - Our World in Data), the calculated carbon footprint would be 152.25 kg CO2e for the NVIDIA T4 GPU and 35.525 kg CO2e for the NVIDIA T4 GPU. Note: K2Co2 stands for kilograms of carbon dioxide equivalent. The average person in the United States emits approximately 43.8 kg of carbon dioxide equivalent (CO2e) per day. Given the emissions of 152.25 kg CO2e for the NVIDIA P100 GPU and 35.525 kg CO2e for the T4 GPU, it would take approximately 0.83 days (or 19 hours) to match the emissions of the T4 GPU. While the calculated carbon footprint provides valuable insight, the actual energy consumption and resulting emissions may vary due to factors like GPU utilization and workload sources. Nonetheless, understanding the carbon footprint is a step towards understanding and mitigating the environmental impact of deep learning research, paving the way for more sustainable and responsible practices in artificial intelligence.

### D. HYPERPARAMETER OPTIMIZATION

Hyperparameter optimization involves training the Agentic-RAG framework variants multiple times with different hyperparameter settings. This can be computationally expensive, especially for complex pre-trained language models or large datasets. We optimized the hyperparameters for the best-performing Agentic-RAG w/Llama-8B variant. For simplicity and in the interest of time, we have utilized the same settings for evaluating the performance of Agentic-RAG w/Gemma-2B and w/Gemma-7B variants for both multivariate and univariate datasets across all tasks. In our experiments, we optimized the training process for supervised fine-tuning using a learning rate from (1e−5, 5e−4). The training was conducted over epochs in the range of (10, 15) with a warmup step count from (500, 1000, 1000) and a decay for regularization from (0.01, 0.05, 0.1). We used gradient accumulation steps for stabilized training convergence from (2, 4, 8) and employed the AdamW optimizer. To manage memory and computational efficiency, we applied 4-bit quantization for QLoRA, with hyperparameters including a low-rank ('r') from (16, 32, 64) and a ('c') from (32, 64, 128), and a dropout from (0.05, 0.1, 0.2).
```

### --- Page 0012 ---

```markdown
| Methods   | GPTATS | PatchTS | TimesNet | FEDFormer | LightsN | N-BEATS | ARAG-W*2 | ARAG-W*7 | ARAG-W*8 |
|-----------|--------|---------|----------|-----------|---------|---------|----------|----------|----------|
| Metric    | MSE    | MAE     | MSE      | MAE       | MSE     | MAE     | MSE      | MAE      | MSE      | MAE      |
| ETTh1     | 96     | 0.376  | 3.907    | 0.334     | 0.376   | 0.419   | 0.424    | 0.432    | 0.399    | 0.428    |
|           | 720    | 0.416  | 0.413    | 0.436     | 0.420   | 0.448   | 0.475    | 0.462    | 0.464    | 0.465    |
| ETTh2     | 336    | 0.442  | 0.433    | 0.422     | 0.436   | 0.463   | 0.474    | 0.863    | 0.672    | 1.454    |
|           | 720    | 0.476  | 0.465    | 0.521     | 0.560   | 0.507   | 0.537    | 0.583    | 0.628    | 0.682    |
| ETTm1     | 96     | 0.332  | 0.372    | 0.367     | 0.426   | 0.410   | 0.407    | 0.315    | 0.336    | 0.310    |
|           | 720    | 0.366  | 0.394    | 0.366     | 0.429   | 0.438   | 0.408    | 0.396    | 0.384    | 0.392    |
| ETtm2     | 96     | 0.173  | 0.262    | 0.165     | 0.257   | 0.203   | 0.287    | 0.209    | 0.308    | 0.197    |
|           | 720    | 0.326  | 0.321    | 0.329     | 0.325   | 0.336   | 0.336    | 0.319    | 0.411    | 0.401    |

Table 11: The table compares various methods for the multi-horizon forecasting task with a lookback window of size 512.

| Methods   | GPTATS | PatchTS | TimesNet | FEDFormer | LightsN | N-BEATS | ARAG-W*2 | ARAG-W*7 | ARAG-W*8 |
|-----------|--------|---------|----------|-----------|---------|---------|----------|----------|----------|
| Metric    | MSE    | MAE     | MSE      | MAE       | MSE     | MAE     | MSE      | MAE      | MSE      | MAE      |
| ETTh1     | 20%    | 0.460  | 0.480    | 0.540     | 0.560   | 0.500   | 0.505    | 0.485    | 0.455    | 0.485    |
|           | 720    | 0.670  | 0.650    | 0.665     | 0.605   | 0.610   | 0.670    | 0.690    | 0.675    | 0.675    |
| ETTh2     | 20%    | 0.406  | 0.411    | 0.379     | 0.426   | 0.463   | 0.474    | 0.863    | 0.672    | 1.454    |
|           | 720    | 0.472  | 0.469    | 0.471     | 0.475   | 0.473   | 0.472    | 0.469    | 0.469    | 0.471    |
| ETtm1     | 20%    | 0.520  | 0.525    | 0.523     | 0.581   | 0.646   | 0.603    | 0.602    | 0.651    | 0.551    |
|           | 720    | 0.530  | 0.531    | 0.526     | 0.514   | 0.511   | 0.511    | 0.507    | 0.508    | 0.504    |
| ETtm2     | 20%    | 0.378  | 0.410    | 0.362     | 0.385   | 0.403    | 0.421    | 0.415    | 0.587    | 0.519    |
|           | 720    | 0.500  | 0.613    | 0.577     | 0.620   | 0.615    | 0.627    | 0.627    | 0.610    | 0.611    |

Table 13: The table evaluates the effectiveness of various missing data imputation techniques (including point-wise and block-wise methods) for out-of-sample imputation, using a 512-step historical window to predict missing values in subsequent 720-step future data.
```

### --- Page 0013 ---

```markdown
# Agentic Retrieval-Augmented Generation for Time Series Analysis
30th, ACM KDD August 25 - 29, 2024, Barcelona, Spain

## E ABLACTION STUDY

To understand the contribution of each component within our proposed Agentic-RAG framework, we designed an ablation study. By systematically evaluating the impact of removing individual components, we gain valuable insights into their role in the framework's overall performance. The following ablation experiments were conducted:

| Component | Description |
|-----------|-------------|
| (a) Effect of dynamic prompting mechanism (DPM): | We compared the performance of the Agentic-RAG framework with and without the dynamic prompting mechanism. |
| (b) Role of sub-agent specialization (SAS): | We evaluated the Agentic-RAG framework using a single universal sub-agent for all tasks versus specialized sub-agents for each task. |
| (c) Instruction-tuning (IT) vs. no fine-tuning (NIT): | We compared the performance of SLMs with instruction-tuning against their performance without any fine-tuning. |
| (d) Effectiveness of direct preference optimization (DPO): | We evaluated the framework's performance with and without DPO and assessed how aligning SLMs with preferred outcomes impacts the accuracy and reliability of predictions. |

Our study investigates the impact of different components on the overall performance of the framework, *SelfExtend-Agentic-RAG W/Llama 3 - 8*, in time series forecasting, anomaly detection, and classification tasks across various benchmark datasets. We systematically disable each component (dynamic prompting mechanism (DPM), sub-agent specialization (SAS), instruction-tuning (IT), or direct preference optimization (DPO)) and compare the results to the full framework. Tables 14 and 15 detail the forecasting performance, highlighting that the original framework consistently achieves the lowest error rates in MAE, RMSE, and MAPE across different horizons and datasets. This indicates the crucial role of each component in improving forecasting accuracy. Table 16 focuses on anomaly detection tasks, showing the original framework's superior precision, recall, and F1-score compared to its ablated variants. The original framework consistently achieves higher metrics scores across anomaly benchmark datasets such as SwAT, WADI, SMAP, MSL, and HAL. The significant performance drop observed in the ablated variants underscores the importance of the integrated components, demonstrating their synergistic contribution to enhancing anomaly detection capabilities.
```

### --- Page 0014 ---

```markdown
| Methods                          | PeM3DS | PeM5D4 | PeM5D7 | PeM5D8 | PeM5D7(M) |
|----------------------------------|--------|--------|--------|--------|-----------|
| Baseline W/O DPM                 | 15.31  | 23.37  | 12.63  | 20.35  | 11.42     |
| Baseline W/O SAS                 | 14.46  | 21.85  | 11.81  | 20.37  | 10.75     |
| Baseline W/O IT                  | 21.04  | 31.01  | 15.85  | 30.48  | 17.10     |
| Baseline W/O DPO                 | 13.53  | 20.67  | 11.86  | 18.59  | 10.83     |
| SelfExtend-Agentic-RAG W/Llama-3 8 | 13.01  | 20.45  | 10.53  | 17.26  | 16.55     |

**Table 14:** The table shows the ablation study results for 12-sequence-to-12-sequence forecasting tasks on benchmark datasets using multiple evaluation metrics. The performance of the ablated variants drops compared to the original framework.

| Datasets   | Methods                          | Horizon @3 | Horizon @6 | Horizon @12 |
|------------|----------------------------------|------------|------------|-------------|
| METR-LA    | Baseline W/O DPM                 | 4.84       | 2.42       | 6.28        |
|            | Baseline W/O SAS                 | 4.19       | 2.63       | 5.97        |
|            | Baseline W/O IT                  | 7.05       | 3.23       | 8.69        |
|            | Baseline W/O DPO                 | 4.19       | 2.12       | 5.36        |
|            | SelfExtend-Agentic-RAG W/Lama-3 8 | 4.03       | 2.02       | 5.05        |
| PEM5-BAY   | Baseline W/O DPM                 | 1.94       | 0.97       | 1.96        |
|            | Baseline W/O SAS                 | 1.79       | 1.82       | 2.79        |
|            | Baseline W/O IT                  | 2.84       | 1.38       | 2.77        |
|            | Baseline W/O DPO                 | 1.69       | 0.85       | 1.62        |
|            | SelfExtend-Agentic-RAG W/Lama-3 8 | 1.62       | 1.63       | 2.52        |

**Table 15:** The table presents the ablation study results for the forecasting task performed on the METR-LA and PEM5-BAY datasets, evaluated using multiple metrics. All methods utilized 12 historical sequences to forecast 3, 6, or 12 future sequences.

| Dataset                          | PeM3DS | PeM5D4 | PeM5D7 | PeM5D8 | PeM5D7(M) |
|----------------------------------|--------|--------|--------|--------|-----------|
| Baseline W/O DPM                 | 79.57  | 78.52  | 83.49  | 76.92  | 82.37     |
| Baseline W/O SAS                 | 85.54  | 84.64  | 83.73  | 88.77  | 81.38     |
| Baseline W/O IT                  | 39.79  | 39.26  | 37.04  | 39.56  | 36.30     |
| Baseline W/O DPO                 | 95.49  | 93.87  | 97.04  | 94.78  | 98.31     |
| SelfExtend-Agentic-RAG W/Lama-3  | 94.97  | 91.58  | 92.63  | 91.97  | 90.84     |

**Table 16:** The table showcases the experimental findings from the ablation study conducted on anomaly detection benchmark datasets, reporting the precision, recall, and F1-score metrics.

| Dataset                          | PeM3DS | PeM5D4 | PeM5D7 | PeM5D8 | PeM5D7(M) |
|----------------------------------|--------|--------|--------|--------|-----------|
| Baseline W/O DPM                 | 77.12  | 75.45  | 76.89  | 77.25  | 76.47     |
| Baseline W/O SAS                 | 81.25  | 79.45  | 80.78  | 82.67  | 80.55     |
| Baseline W/O IT                  | 25.45  | 22.78  | 21.42  | 22.67  | 20.56     |
| Baseline W/O DPO                 | 88.67  | 87.23  | 84.25  | 91.02  | 88.56     |
| SelfExtend-Agentic-RAG W/Lama-3  | 93.01  | 91.56  | 92.31  | 95.65  | 90.35     |

**Table 17:** The table presents the ablation study results, evaluating the performance across various metrics for time series classification tasks on the PeM3D, PeM5D4, PeM5D7, and METR-LA benchmark datasets.

| Dataset                          | PeM5D7(M) | PeM3DS | PeM5D4 | PeM5D7 | PEM5-BAY |
|----------------------------------|-----------|--------|--------|--------|----------|
| Baseline W/O DPM                 | 75.41     | 73.21  | 74.26  | 76.02  | 71.75    |
| Baseline W/O SAS                 | 82.43     | 82.56  | 81.41  | 81.32  | 82.01    |
| Baseline W/O IT                  | 37.61     | 36.12  | 36.54  | 38.02  | 36.17    |
| Baseline W/O DPO                 | 90.62     | 88.73  | 92.54  | 39.32  | 91.01    |
| SelfExtend-Agentic-RAG W/Lama-8  | 94.03     | 92.46  | 93.05  | 95.04  | 91.05    |

**Table 18:** This table presents the results of an ablation study comparing the performance of various Agentic-RAG framework variants. The study evaluates performance on three benchmark datasets - PeM5D7(M), PeM3DSB, and PEM5-BAY - across different metrics for time series classification tasks.
```


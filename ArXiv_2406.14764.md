# ArXiv 2406.14764

### --- Page 0001 ---

```markdown
# RE-AdaptIR: Improving Information Retrieval through Reverse Engineered Adaptation

**William Fleshman** and **Benjamin Van Durme**  
Johns Hopkins University  
will.fleshman@jhu.edu

## Abstract

Large language models (LLMs) fine-tuned for text-retrieval have demonstrated state-of-the-art results across several information retrieval (IR) benchmarks. However, supervised training for improving these models requires numerous labeled examples, which are generally unavailable or expensive to acquire. In this work, we explore the effectiveness of extending reverse engineered adaptation to the context of information retrieval (RE-AdaptIR). We use RE-AdaptIR to improve LLM-based IR models using only unlabeled data. We demonstrate improved performance both in training domains as well as zero-shot in domains where the models have seen no queries. We analyze performance changes in fine-tuning scenarios and offer findings of immediate use to practitioners.

## 1 Introduction

Information retrieval (IR) is a fundamental component of various modern applications, powering search engines, recommendation systems, and various data analytics pipelines. Recently, large language models (LLMs) have achieved state-of-the-art results on dense text retrieval, identifying and ranking the most relevant text for a given query by comparing learned vector representations of the text (Reimers and Gurevych, 2019; Khattab and Zaharia, 2020; Karpukhin et al., 2020; Izacard et al., 2022; Ma et al., 2023; Jiang et al., 2023; Weller et al., 2024). The effectiveness of text retrieval systems have a direct impact on numerous domains, including healthcare, finance, and social media, where accurate and timely access to information is critical. Retrieval is also critical in the context of retrieval augmented generation (RAG), enabling LLMs access to external resources when constructing a response (Lewis et al., 2020). For these reasons, we seek a practical and efficient approach for improving existing text retrieval models.

Supervised fine-tuning of LLMs for text retrieval tasks has become a widely adopted approach, leveraging their pretrained language understanding capabilities to achieve state-of-the-art results on various benchmarks (Ma et al., 2023; Jiang et al., 2023; Weller et al., 2024). However, adapting an LLM for text retrieval requires labeled datasets, with numerous example queries and documents both related and unrelated to forming a helpful response. This poses a significant challenge to improving these systems, as data annotation or synthetic generation can be too expensive, difficult, and error-prone (Fredriksson et al., 2020; Desmond et al., 2021). Making matters worse, fine-tuning an existing LLM on new domains can cause forgetting, a decreased performance on previously capable tasks (McCloskey and Cohen, 1989; Kotha et al., 2024). Fleshman and Van Durme (2024) recently proposed reverse engineered adaptation (RE-ADAPT), an approach for solving similar dilemmas faced when fine-tuning existing instruction-tuned models. Here we introduce RE-AdaptIR, an extension of RE-ADAPT.

![In RE-AdaptIR, RE-ADAPT is extended to an existing retrieval model to isolate what was learned during labeled contrastive training. The pretrained model is fine-tuned on unlabeled in-domain documents and readapted for text retrieval. The new retriever outperforms the original on both in-domain and zero-shot information retrieval tasks.](assets/page_0001_img_1.png)
```

### --- Page 0002 ---

```markdown
# ADAPT to IR models, leveraging unlabeled data to improve existing text-retrieval LLMs (Figure 1). Specifically we:

- Extend RE-ADAPT to the information retrieval setting and apply RE-ADAPTIR to two state-of-the-art text retrieval models: RepLLaMA and e5-Mistral;
- Demonstrate improved performance both in-domain and zero-shot across 14 datasets; and
- Explore the importance of fine-tuning on data relevant to test-time queries and the impact different scenarios have on performance.

## 2 Background

### 2.1 Retrieval Models

The transformer architecture is a natural choice for text retrieval models, as it embeds text into dense vector representations (Vaswani et al., 2023). LLMs pretrained on massive amounts of text have demonstrated the ability to capture useful semantic meaning in these vector representations (Radford et al., 2019; Tourvon et al., 2023; Jiang et al., 2023). To ensure these models are capable for retrieval, a secondary fine-tuning stage is used to encourage the model to map similar texts to similar vectors (Ma et al., 2023; Jiang et al., 2023; Weller et al., 2024). This is generally done via some form of supervised contrastive training such as with InfoNCE (van den Oord et al., 2019). After training, the models can be used to create a vector database of documents, and rank them given a query representation by using a similarity function. Related documents should have more similar representations to the query than those unrelated. We experiment with two such models in this work: RepLLaMA (Ma et al., 2023) and e5-Mistral (Wang et al., 2024).

**RepLLaMA.** Ma et al. (2023) introduced RepLLaMA to demonstrate that state-of-the-art LLMs could surpass the previous results achieved with smaller retrieval models, especially when evaluated zero-shot on datasets not seen during training. They construct RepLLaMA by fine-tuning LLaMA-2-7B (Tourovn et al., 2023) on approximately 500k labeled examples from the training split of the MS-MARCO dataset (Bajaj et al., 2018).

**e5-Mistral.** In a similar fashion, Wang et al. (2024) fine-tune the Mistral-7B LLM (Jiang et al., 2023) using a combination of synthetic data, MS-MARCO, and multiple other labeled datasets. The resulting e5-Mistral model achieves state-of-the-art results on several text retrieval benchmarks.

In both cases, labeled data was needed to achieve the best results, and it is unclear how to effectively incorporate the copious amount of unlabeled text additionally available. For example, MS-MARCO contains almost 9 million passages, but only a fraction of this data is used in training, due to the limited number of associated queries available. In this work, we use RE-ADAPTIR to leverage this unlabeled data and improve the performance of both RepLLaMA and e5-Mistral.

### 2.2 Reverse Engineered Adaptation

Fleshman and Van Durme (2024) introduced reverse engineered adaptation (RE-ADAPT) as a new method to efficiently update instruction-tuned models with unlabeled data lacking the previously required instruction-tuning annotations (Mishra et al., 2022; Wei et al., 2022; Ouyang et al., 2022). RE-ADAPT works by first isolating what has been learned from instruction-tuning by taking the difference between the weights of the instruction-tuned and pretrained versions of a model. This difference can be thought of as an adapter (Houlsey et al., 2019) or as a multi-task version of task-vectors (Ilharco et al., 2023). Given this RE-ADAPT $\Delta$, the pretrained weights $\Theta$ can be fine-tuned with a new knowledge adapter $\Psi$ without impacting the previous instruction-tuning. Finally, the model can be re-instantiated with weights $\Theta + \alpha \cdot \Delta$ where $\alpha$ and $\beta$ are partial adaptation scalars used to control the strength of fine-tuning (Fleshman and Van Durme, 2024). The authors show that RE-ADAPT improves the performance of instruction-tuned models in the new domain while preserving or improving performance out-of-domain.

## 3 Re-AdaptIR

In this work, we explore the effectiveness of RE-ADAPT in the context of information retrieval. While an instruction-tuned model still leverages the pretraining capabilities of next-token prediction, most text retrieval models do not. RepLLaMA and e5-Mistral both discard the next-token prediction from their respective LLMs and fine-tune the model to produce a single vector representation per document (Ma et al., 2023; Wang et al., 2024). It is therefore unclear whether continued fine-tuning of the pretrained LLM with next-token prediction will improve down-stream retrieval.
```

### --- Page 0003 ---

```markdown
To answer this question, we first fine-tune the pretrained LLM on unlabeled documents from a new domain. We then construct a RE-AdaptIR for the retrieval model by discarding the pretrained next-token predictor weights as well as the corresponding weights from the knowledge adapter. We can then follow the RE-ADAPT procedure using the remaining weights, and evaluate the readapted IR model using queries from the new domain.

## 4 Experiments

We first replicate the shared text retrieval experiments conducted by Ma et al. (2023) and Wang et al. (2024) and compare the base model performance before and after applying RE-AdaptIR. We then conduct further analysis to understand how different fine-tuning scenarios impact performance.

### 4.1 Datasets

Both RepLLaMA and e5-Mistral utilized MS-MARCO (Bajai et al., 2018) as part of their training data, and we include it in our evaluations to help measure any benefits from using RE-AdaptIR in-domain. Specifically, RE-AdaptIR allows for fine-tuning over the entire 8.84M passages, where only a subset of those passages were used for retrieval training due to limited availability of query-passage pairs (Ma et al., 2023; Wang et al., 2024). Additionally, we use the same 13 public datasets from the BeIR IR benchmark (Thakur et al., 2021) used by Ma et al. (2023) to assess RepLLaMA’s zero-shot performance across a diverse set of IR tasks. Of these, we note that the training splits of FEVER (Thorne et al., 2018), HotPotQA (Yang et al., 2018), NQ (Kwiatkowski et al., 2019), and Quora (DataCanary et al., 2017) were also used by Wang et al. (2024) in the training of e5-Mistral, providing more in-domain insight to our experiments. These datasets are zero-shot for RepLLaMA, as is the remainder of BeIR for both models. We use the same prompts used by Ma et al. (2023) and Wang et al. (2024) for each dataset.

### 4.2 Adapters

We use parameter efficient fine-tuning (Mangulkar et al., 2022) with DoRA (Li et al., 2024) to adapt the pretrained LLaMA-2-7B (Touvron et al., 2023) and Mistral-7B (Jiang et al., 2023) LLMs. We train each adapter for a single epoch on all passages from the dataset under evaluation. Specific training details are included in Appendix A. As in Fleshman and Van Durme (2024), we use a scalar of 0.5 with our knowledge adapters to minimize interference with existing retrieval ability.

| Dataset       | RepLLaMA Base | RepLLaMA RA | e5-Mistral Base | e5-Mistral RA |
|---------------|----------------|--------------|------------------|----------------|
| MS-MARCO      | 46.5           | 46.1         | 36.5             | 40.1           |
| FEVER         | 84.0           | 83.8         | 85.1             | 87.7           |
| HotPotQA      | 67.2           | 67.6         | 72.5             | 73.4           |
| NQ            | 61.8           | 62.1         | 53.2             | 52.4           |
| Quora         | 60.1           | 82.8         | 85.0             | 88.0           |
| Arguana       | 52.3           | 52.6         | 52.9             | 59.0           |
| Climate-FEVER | 30.8           | 30.4         | 29.4             | 31.4           |
| DBPedia       | 43.4           | 43.5         | 47.2             | 47.1           |
| FiQA          | 42.4           | 45.5         | 49.9             | 52.3           |
| NFCorpus      | 38.0           | 38.6         | 40.8             | 39.6           |
| SciDocs       | 17.7           | 18.3         | 18.6             | 18.7           |
| SciFact       | 74.5           | 76.3         | 71.4             | 73.3           |
| TREC-COVID    | 84.0           | 85.6         | 83.9             | 81.0           |
| Touche-2020   | 27.5           | 29.0         | 30.1             | 30.1           |
| **Average**    | **53.7**       | **54.3**     | **53.5**         | **55.4**       |
| **Average Z-Shot** | **54.3**   | **54.9**     | **46.3**         | **48.2**       |

Table 1: nDCG@10 across test splits for MS-MARCO and BeIR datasets. The results highlighted in orange indicate the dataset’s train split was used for training the corresponding model and are not zero-shot. Base is the unmodified model and RA is the model RE-Adapted after fine-tuning on the domain.

### 4.3 Results

Our main results are compiled in Table 1. We see that RE-ADAPTIR improves results in the majority of cases, increasing the average zero-shot nDCG@10 by 0.6 and 1.9 points for RepLLaMA and e5-Mistral respectively. Importantly, these performance gains required no additional labeled data and are achieved by simply fine-tuning the pretrained model over the document database being used for retrieval. The default partial adaptation scalar of 0.5 was used for this experiment, but we note that optimizing this value per dataset does improve results. While not applicable to our zero-shot analysis, in practice this value can be set using withheld queries.

We notice the few cases where performance was reduced tend to occur with the larger corpora. We plot this relationship in Figure 2 and do observe a slightly negative correlation. This relationship is purely observational and likely caused by latent topic or task diversity among the larger datasets used in these experiments. For reference, we include the dataset sizes in Appendix B.

Are queried passages all that matter? One reason the in-domain results could be better than baseline is because the passages being queried for at
```

### --- Page 0004 ---

```markdown
| RE-Adapted Model | Arguna | FiQA | NFCorpus | SciFact |
|------------------|--------|------|----------|---------|
|                  | w/o    | w/   | w/o      | w/      |
| RepLLaMA         | +0.5   | +0.2 | +0.3     | +1.0    |
| e5-Mistral       | +3.9   | +5.1 | +7.0     | +2.0    |

Table 2: Change in nDCG@10 over original retriever when pretrained model is fine-tuned using only documents with (w/) or without (w/o) corresponding queries in the test set, or with (both) subsets.

![The observed relationship between the corpus size and the change in performance when fine-tuning with RE-ADAPTIR.](assets/page_0004_img_1.png)

e5-Mistral only used a subset to supplement the otherwise synthetically generated data (Ma et al., 2023; Wang et al., 2024). In both cases, the extra data improved the performance and indicates that retrieval models can generally benefit from additional unlabeled training using RE-ADAPTIR.

| Dataset     | RepLLaMA | e5-Mistral |
|-------------|----------|------------|
|             | Base     | Dom        | Base     | Dom        |
| FEVER       | +0.1     | +0.3      | +1.0     | +2.7      |
| HotPotQA    | -0.4     | -0.8      | +0.5     | -0.4      |
| NQ          | -0.1     | -0.4      | +0.7     | +7.9      |
| Quora       | +2.0     | +0.7      | +2.4     | +0.0      |
| Arguna      | +0.8     | +0.5      | +10.1    | +31.1     |
| Climate-FEVER| +1.0    | +1.4      | +1.1     | +7.6      |
| DBPedia     | +0.1     | +0.0      | +0.5     | +0.6      |
| FiQA        | +0.2     | +1.1      | +1.7     | +0.7      |
| NFCorpus    | 0.0      | -0.6      | +0.7     | -0.5      |
| SCDOCS      | -0.1     | -0.5      | +0.6     | +0.5      |
| SciFact     | -0.9     | -2.7      | +2.1     | +0.4      |
| TREC-COVID  | -0.4     | -2.0      | +2.6     | +5.5      |
| Touche-2020  | -0.1    | +0.4      | -0.5     | -1.6      |
| **Average**  | +0.2    | +0.5      | +2.0     | +0.3      |

Table 3: Change in nDCG@10 when RE-ADAPTIR is applied with pretrained model fine-tuned on MS-MARCO instead of the domain under evaluation. Base indicates the change with respect to the original model, Dom the change with respect to the model RE-Adapted on the evaluated domain.

5 Conclusion  
In this work, we introduced RE-ADAPTIR, an extension of RE-ADAPT for using unlabeled data to improve the zero-shot and in-domain performance of text retrieval models. We demonstrated RE-ADAPTIR improves two state-of-the-art models: RepLLaMA and e5-Mistral. We find that fine-tuning on the documents being queried for at test-time is not required, and still see increased performance when they are excluded. RE-ADAPTIR improved baseline performance in two cases: one where in-domain data was used, and the other, using additional unlabeled data relative to the models’ original training corpus. Combined, these results reinforce the wide applicability of our approach, and our findings ensure RE-ADAPTIR is of immediate use to text-retrieval practitioners.
```

### --- Page 0005 ---

```markdown
# References

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Via Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS MARCO: A human generated machine reading comprehension dataset. Preprint, arXiv:1611.09268.

DataCanary, hifilakaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, and tomtung. 2017. Quora question pairs.

Michael Desmond, Evelyn Duesterwald, Kristina Brimi- join, Michelle Bachram, and Qian Pan. 2021. Semi-automated data labeling. In Proceedings of the NeurIPS 2020 Competition and Demonstration Track, volume 133 of Proceedings of Machine Learning Research, pages 156–169. PMLR.

William Fleshman and Benjamin Van Durme. 2024. Readapt: Reverse engineered adaptation of large language models. Preprint, arXiv:2405.15007.

Teodor Fredriksson, David Issa Mattos, Jan Bosch, and Helena Holmström Olsson. 2020. Data labeling: An empirical investigation into industrial challenges and mitigation strategies. In Product-Focused Software Process Improvement, pages 202–216. Cham. Springer International Publishing.

Neil Housby, Andrei Giurgiu, Stanislav Jastrzebski, Bruna Morrone, Quentin De Laroussihe, Andrea Gesmundo, Moona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790–2799. PMLR.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannah Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations.

Gautier Izard, Mathilde Caron, Lucas Hossien, Sébastien Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. Preprint, arXiv:2112.09118.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men- ch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guil- laume Lample, Lucile Saulnier, Lélio Renard Lavau, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William E. Sayed. 2023. Mistral 7B. Preprint, arXiv:2310.06285.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open- domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781. Online. Association for Computational Linguistics.

Omar Khatab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, page 39–48, New York, NY, USA. Association for Computing Machinery.

Suhas Kotha, Jacob Springer, and Aditi Raghunathan. 2024. Understanding catastrophic forgetting in lan- guage models via implicit inference. In NeurIPS 2023 Workshop on Distribution Shifts: New Frontiers with Foundation Models.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Ken- don Lee, Kristina Tsioutauna, Lion Jones, Matthew Kellezy, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Saury Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock- täschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge- intensive NLP tasks. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA. Curran Associates Inc.

Shih-yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024. DORA: Weight- decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning.

Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2023. Fine-tuning llama for multi-stage text retrieval. Preprint, arXiv:2310.08319.

Sourab Mangulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. 2022. Petf: State-of-the-art parameter- efficient fine-tuning methods. https://github.com/huggingface/petf.

Michael McCloskey and Neal J. Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. volume 24 of Psychology of Learning and Motivation, pages 109–165. Academic Press.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannah Hajishirzi. 2023. Cross-task generaliza- tion via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3470–3487, Dublin, Ireland. Association for Computational Linguistics.
```

### --- Page 0006 ---

```markdown
| Authors                                                                 | Year  | Title                                                                                                   | Source                                                                                      |
|-------------------------------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Long Ouyang, Jeff Wu, Xu Jiang, Diego Almeida, Carroll L. Wainwright,  | 2022  | Learning to follow instructions with human feedback.                                                   | ArXiv, abs/2302.01525.                                                                     |
| Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slaman, Alex   |       |                                                                                                         |                                                                                             |
| Ray, John Schulman, Jacob Hilton, Fraser Kelleyn, Luke E. Melcher,     |       |                                                                                                         |                                                                                             |
| Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, |       |                                                                                                         |                                                                                             |
| Jan Leike, and Ryan J. Lowe.                                            |       |                                                                                                         |                                                                                             |
| Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya | 2019  | Language models are unsupervised multitask learners.                                                    | Preprint, arXiv:2401.00368.                                                                |
| Sutskever.                                                              |       |                                                                                                         |                                                                                             |
| Nils Reimers and Iryna Gurevych.                                       | 2019  | Sentence-BERT: Sentence embeddings using Siamese BERT-networks.                                       | In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics. |
| Nandan Thakur, Nils Reimers, Andreas Rücelk, Abhishek Srivastava, and  | 2021  | BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models.               | In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). |
| Iryna Gurevych.                                                         |       |                                                                                                         |                                                                                             |
| James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit  | 2018  | FEVER: A large-scale dataset for fact extraction and VERification.                                     | In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics. |
| Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Arijit Almahari,| 2023  | Llama 2: Open foundation and fine-tuned chat models.                                                  | Preprint, arXiv:2307.09288.                                                                |
| Shashank Bansal, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui  |       |                                                                                                         |                                                                                             |
| Hakan Inan, Marcin Kardaś, Viktor Kerkez, Madián Khabsa, Isabel Kloumann,|       |                                                                                                         |                                                                                             |
| Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril,  |       |                                                                                                         |                                                                                             |
| Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet,   |       |                                                                                                         |                                                                                             |
| Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Raish Rung, Kalyan      |       |                                                                                                         |                                                                                             |
| Saladi, Alan Scheitlen, Ruan Silva, Eric Michael Smith, Ranjan Subramani,|       |                                                                                                         |                                                                                             |
| Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang  |       |                                                                                                         |                                                                                             |
| Kuan, Puxin Xu, Zheng Yan, Ilyan Zarov, Yuchen Zhang, Angela Fan,      |       |                                                                                                         |                                                                                             |
| Melanie Kambadur, Sharan Narang, Aurélien Rozière, Robert Stojnic,     |       |                                                                                                         |                                                                                             |
| Sergey Edunov, and Thomas Scialom.                                      |       |                                                                                                         |                                                                                             |
| Aaron van den Oord, Yazhe Li, and Oriol Vinyals.                       | 2019  | Representation learning with contrastive predictive coding.                                             | Preprint, arXiv:1807.03748.                                                                 |
| Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Lilion     |       | Attention is all you need.                                                                               | Preprint, arXiv:1706.03762.                                                                 |
| Kaiser, and Illia Polosukhin.                                           |       |                                                                                                         |                                                                                             |
| Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder,    | 2024  | Improving text embeddings with large language models.                                                  | Preprint, arXiv:2401.00368.                                                                 |
| and Furu Wei.                                                           |       |                                                                                                         |                                                                                             |
| Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu,                     | 2022  | Fine-tuned language models are zero-shot learners.                                                     | In International Conference on Learning Representations.                                    |
| Andrew M. Dai, and Quoc V. Le.                                          |       |                                                                                                         |                                                                                             |
| Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan,    | 2021  | Follower: Evaluating and teaching information retrieval models to follow instructions.                 | Preprint, arXiv:2403.15246.                                                                 |
| Benjamin Van Durme, Dawn Lawrie, and Luca Soldaini.                     |       |                                                                                                         |                                                                                             |
| Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,                   | 2018  | HotpotQA: A dataset for diverse, explainable multi-hop question answering.                             | In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics. |
```

### --- Page 0007 ---

```markdown
# A Adapter Details

We trained DoRA adapters for all attention key, query, and value layers as well as the up and down projection layers. All adapters used rank 32 with alpha 64 and a LoRA dropout of 0.05.

We used a batch size of 4 using the AdamW optimizer with learning rate of 0.0002 with linear scheduling. Unless otherwise stated, all adapters were trained over 1 epoch of the respective corpus with a max length of 1024 for any example.

All training and evaluation was done using a single NVIDIA A100 GPU with 80GB of memory.

# B Corpus Sizes

| Dataset        | Corpus Size |
|----------------|-------------|
| MS-MARCO       | 8.84M       |
| Climate-FEVER  | 5.42M       |
| FEVER          | 5.42M       |
| HotPotQA       | 5.23M       |
| DBPedia        | 4.63M       |
| NQ             | 2.68M       |
| Quora          | 523K        |
| Touche-2020    | 382K        |
| TREC-COVID     | 171K        |
| FiQA           | 57K         |
| SCIDOCS       | 25K         |
| ArguAna        | 8.67K       |
| SciFact        | 5K          |
| NFCorpus       | 3.6K        |
```


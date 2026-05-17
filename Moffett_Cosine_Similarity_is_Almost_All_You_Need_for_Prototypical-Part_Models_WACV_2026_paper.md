# Moffett Cosine Similarity is Almost All You Need for Prototypical-Part Models WACV 2026 paper

### --- Page 0001 ---

```markdown
# Cosine Similarity is Almost All You Need (for Prototypical-Part Models)

Luke Moffett¹  
Luke.Moffett@duke.edu  

Frank Willard¹  
FrankieWillard@gmail.com  

Maximillian Machado¹  
Maximilian.Machado@duke.edu  

Emmanuel Mokel¹  
Emmanuel.Mokel@duke.edu  

Jon Donnelly¹  
Jon.Donnelly@duke.edu  

Zhicheng Guo¹  
Zhicheng.Guo@duke.edu  

Adam Costarino¹  
Adam.Costarino@duke.edu  

Julia Yang¹  
Julia.Yang@duke.edu  

Giyoung Kim¹  
Giyoung.Kim@duke.edu  

Alina Jade Barnett²  
Alina.Barnett@uri.edu  

Cynthia Rudin¹  
Cynthia.Rudin@duke.edu  

¹Duke University  
²University of Rhode Island  

## Abstract

Prototypical-part networks are a popular interpretable alternative to black-box deep learning models for computer vision because of their faithful, prototype-based self-explanations. However, in practice, they have proven difficult to train because they are highly sensitive to hyperparameter tuning and difficult to comprehend because they contain a large number of prototypes. We show that replacing $l_2$ distance with an angular prototype similarity in the original ProtoP-Net greatly improves robustness to hyperparameter selection and is sufficient to produce accuracy and sparsity competitive with state-of-the-art on many backbones and datasets. We also show cosine similarity leads to superior accuracy for five different ProtoPNet architectures (ProtoPNet, TestNet, Deformable ProtoPNet, ProtoTree, and ST-ProtoPNet). Finally, we demonstrate ProtoPNet with cosine similarity produces better semantics than $l_2$ prototypes from cosine models, scores better on prototype quality metrics and are perceived as more similar 3:2 in a user study.¹

## 1. Introduction

Prototypical-Part Networks (PPNs) are deep neural networks that perform classification by comparing example image patches (“prototypes”) from their training set to input image patches using latent representations extracted at neural network work. These comparisons are available to users as inference.

¹Code is available at [https://github.com/lmoffett/cosine-is-almost](https://github.com/lmoffett/cosine-is-almost)

![Validation Accuracy Distribution](assets/page_0001_img_1.png)

![Optimization Trajectory](assets/page_0001_img_2.png)

![Cosine](assets/page_0001_img_3.png)

![Euclidean](assets/page_0001_img_4.png)
```

### --- Page 0002 ---

```markdown
![Visual Comparison Cosine and $\ell_2$ Similarity Maps. For two randomly selected prototype images, the five most similar training images are displayed for each (of left) a model using cosine similarities and (right) a model using $\ell_2$ (Euclidean) similarities. Heatmaps represent model-assessed similar patches, and bounding boxes cover 90th-percentile and above similarity-scored patches. Models that use cosine similarity produce more semantically consistent and concentrated prototype localization, which we show quantitatively through a user study and prototype quality metrics.](assets/page_0002_img_1.png)

original ProtoPNet took over a day of GPU computation to train due to its slow convergence – this is despite the training regime’s computational requirements being almost identical to finetuning a pretrained convolutional neural network. PPNs are further hindered by the latent structure’s sensitivity to hyperparameter selection, affecting not only the downstream performance but also the user comprehension of the model reasoning.

In this work, we found a simple and efficient way to train any kind of PPN architecture much more effectively – cosine similarity (Figure 1). This trick was, in some sense, the turning point of our noses all along. The original ProtoPNet employed an $\ell_2$-distance based measure with a logarithmic kernel to measure similarity, an approach largely followed with minor changes in later models [39, 40, 48, 49]. The TesNet [59] authors demonstrated an improvement in performance to diversifying and disentangling the set of prototypes; we believe that TesNet’s actual improvement in performance was due not primarily to the diversity of prototypes but largely due to the incidental use of cosine similarity as a result of their introduction of orthogonality loss functions. While TesNet is one of the leading performers among PPNs, we show it is possible to achieve near-parity using other variants (including the original ProtoPNet) by simply switching to cosine similarity. Conversely, switching PPNs such as Deformable ProtoPNet [17] and ST-ProtoPNet [58] from cosine similarity to $\ell_2$-distance hurts performance.

We summarize our contributions as the following:
1. We demonstrate that cosine similarity provides superior performance, training stability, and model sparsity for five different PPN architectures (ProtoPNet, Deformable ProtoPNet, TesNet, ProtoTree, and ST-ProtoPNet). Changing the original ProtoPNet to use a cosine similarity measure and systematically tuning hyperparameters moves ProtoPNet onto the Pareto frontier of accuracy and sparsity of ProtoPNet architectures.
2. We further show that models trained with cosine similarity have better semantics as measured by prototype quality metrics and user preferences.
3. We create a novel early stopping criterion for PPNs that trains PPNs in low single digit hours, enabling fair, fixed-GPU-computation comparisons.

## 2. Related Work
In contrast to methods that provide post-hoc “explanations” for deep learning models’ predictions [4, 31, 35, 52, 54], a family of work aims to develop ad-hoc inherently interpretable models. Our work focuses on class-based DNNs [11, 34]. ProtoPNet was introduced by Chen et al. [11], which learns a set of prototypical parts for each class, and forms predictions for each new instance by comparing each learned prototype to the input instance. Several follow-up works to ProtoPNet have altered the mechanism by which these prototype comparisons are translated into class predictions: ProtoTree [39] forms predictions by using the similarity of its prototypes to traverse a soft decision tree and uses an exponential kernel for similarity. ProtoPShare [47] and ProtoPool [49] introduce a prototype sharing mechanism that allows a given prototype to be shared by multiple classes, the latter using a modified logarithmic kernel for similarity. ProtoPath [61] retains the logarithmic kernel but visualizes similarities using Shapley values. In another direction, TesNet [59] encouraged prototypes to form orthogonal bases for class subspaces. TesNet’s orthogonality loss was adopted by Deformable ProtoPNet [17], which extended ProtoPNet’s prototypes to allow spatial deformations, and ST-ProtoPNet [58], which moves prototypes closer to the de-
```

### --- Page 0003 ---

```markdown
# 3. Methods

This section is organized as follows: In Subsection 3.1, we introduce the fundamentals of PPNs. We then investigate the implications of different similarity functions (Subsection 3.2) and the hyperparameter optimization method used to further compare them (Subsection 3.3). Finally, we describe a user-friendly prototype of similarity functions (Subsection 3.4).

## 3.1 Background and Notation

Let $X \in \mathbb{R}^{H \times W \times C}$ denote an image of height $H$ and width $W$ with $C$ channels, and let $Y \in \{0, \ldots, K - 1\}$ denote its class label. A PPN consists of three primary components: an embedding layer $f : \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{D \times K \times W'}$, which extracts a high-level map of D dimensional feature vectors from an image $x_i$; a prototypical layer $g : \mathbb{R}^{D \times K \times W'} \to \mathbb{R}^{P \times K}$, which computes the similarity of each of the learned prototypes to the input at each of $H' \times W'$ center locations; and a class prediction head $h : \mathbb{R}^{P \times K} \to \mathbb{R}^K$, which uses these prototype similarities to compute an output logit for each class. The PPN's predictions are $\hat{Y} = \arg \max_{k \in \{1, \ldots, K\}}(h \circ g \circ f)(x_i)$.

In our study, we focus on the implementation of $g$, borrowing multiple backbones (implementations of $f$) and prediction heads (implementations of $h$) from the literature. The prototype layer $P = \{p_j\}_{j=1}^J$, where each $p_j \in \mathbb{R}^{D \times K \times W'}$ is a tensor representing latent patches. $H_P$ and $H_W$ are hyperparameters that define the height and width in latent space of each learned prototypical-part. For each prototype $p_j$, we compute a patch-level similarity map $\phi_j$ $\in \mathbb{R}^{H' \times W'}$,

$$
\phi_{j,h,w}(z) = (p_j)_{:,h,w} \cdot (H_P + H_W + w) + \gamma : \mathbb{R}^{D \times H_P \times W_P} \to \mathbb{R}^+ 
$$

as a similarity scoring function. Finally, we compute the "overall similarity" of prototype $p_j$ to sample $z$ as $\gamma(z) := \max_{h,w} \phi_{j,h,w}$.

The focus of this work is the choice of $\gamma$. The original ProtoPNet used the similarity function

$$
\gamma_c(p_j, z) := \log \left( \frac{\|v(z) - v(p_j)\|_2^2 + 1}{\|v(z)\|_2^2 + 1} \right) + \epsilon
$$

where $v : \mathbb{R}^{D \times K \times W'} \to \mathbb{R}^{D H' \times W'}$ denotes the flattening of a tensor to a vector. We refer to $\gamma$ as $\ell_2$-similarity. Note that this form includes a logarithmic kernel to convert the $\ell_2$ distances into a positive similarity measure. We also test the application of normalized $\ell_2$ similarity:

$$
\gamma_E(p_j, z) := \log \left( \frac{\|v(z)\|_2^2}{\|v(p_j)\|_2^2} + 1 \right) + \epsilon
$$

We define cosine similarity as

$$
\gamma_{cos}(p_j, z) := v(z) \cdot v(p_j)
$$

The models are trained to minimize a loss of the form

$$
\ell_{overall}(f, g, h, X, Y) = C \cdot h(o \circ f(X), Y)
$$

where the overall loss $\ell_{overall}$ is the sum of the standard cross entropy loss, denoted as $C$, and a weighted sum over interpretability-encouraging terms denoted collectively as $\ell_{interp}$. $\ell_{interp}$ encourages the latent embedding space produced by $f$ to be well clustered by class and prototypes to recover representative samples from each class. For instance, cluster loss, $\ell_{clust} = \frac{1}{N} \sum_{i=1}^N \text{max}_{j:p_j \in P} g_j(z_i)$, where $P_j$ denotes the set of prototypes associated with class $j$, encourages patches from a class to lie from other classes' prototypes.

In almost all PPNs, a prototype projection step is performed during or at the end of training. This projection step replaces each prototype with the nearest latent patch from a training sample: $p_j' = \text{arg max}_{k} g_j(z_i)$, where $D = \{i : y_i = \text{class}(p_j)\}$ and class$(p_j)$ denotes the class associated with the $j$-th prototype. This ensures that each prototype has a pixel-space representation for comparison. The projection step is important for determining the number of prototypes in the fully trained model because the same patch can replace multiple learned prototypes. As a result, the hyperparameter $P$ specifying the number of prototypes in a model is an upper bound on the ultimate number of prototypes in the model. To determine the actual number of prototypes, we report the number of distinct latent vectors as measured by $\gamma$.
```


### --- Page 0004 ---

```markdown
## 3.2 Prototype Metrics

The key benefit of PPN models is their interpretability; as such, we are interested in quantifying and optimizing for interpretability when comparing similarity functions. We compute three measures of model interpretability: stability and consistency, as introduced in [26], and a novel prototype sparsity metric. These are not an exhaustive list of prototype quality metrics – which are an active area of research – but constitute a reasonably diverse set of interpretability criteria against which to test joint optimization. In order to jointly optimize these metrics alongside accuracy, we ensure that each metric has a range of $[0, 1]$.

Recall that $K$ denotes the number of classes in the dataset, $H$ the height of the latent space, and $W$ the width of the latent space. For a model with $P$ distinct prototypes of shape $D \times H \times W_p$, we compute a sparsity score $sparsity$ as: 

$$
sparsity = \frac{(K + H \cdot W_p)}{(P + H \cdot W_p)}
$$

This construction captures four desirable properties: First, sparsity and the number of distinct prototype are inversely proportional; doubling the number of distinct prototypes increases the sparsity score by 1 if it has $K$ prototype images and $K$ prototypical parts the model learns. This defines how many prototype sample images and what proportion of those images a user must consider when examining a given prediction.

Prototype Stability and Consistency. Prototype stability, $stab$, measures the invariance of prototype activation to Gaussian noise added to the input image. Stable prototypes do not change activation in response to noise. Prototype consistency, $consist$, measures how frequently a prototype activates on the same semantic part of the image; for instance, the head of a bird and not the head and the feet. Both metrics are formally defined in [26].

Optimization Objectives. In separate experimental runs, we optimize for only classification accuracy (denoted $Acc$) or for both accuracy and prototype quality scores (denoted $Acc-PS$). For each metric $m$, let $v_{m,e}$ denote its value at epoch $e$, and $obj_{m}$ its optimization objective. For accuracy, $obj_{Acc} = v_{e}$. When directly optimizing interpretability, we define a single prototype score as the average of the three prototype metrics, which can be generally defined as $v_{Acc-PS} = \sum_{m=1}^{M} \lambda_m \cdot v_{m}$ where $M$ is the total number of prototype metrics, $v_{m}$ is the m-th prototype metric, and $\lambda_m$ is the weight for the m-th metric. The weights on these metrics can be determined either through priors or explicit goals to optimize for specific metrics. In our experiments, we are interested in studying joint optimization generally using three metrics for which we have even weighting yielding the objective, $v_{Acc-PS} = (v_{Acc} + v_{PS} + v_{obj})$. We maximize $v_{Acc-PS} = v_{Acc-PS}$. This multiplicative construction emphasizes complementarity between accuracy and prototype quality.

## 3.3 Hyperparameter Optimization for Fair Similarity Function Comparison

In order to fairly compare similarity functions, we perform hyperparameter optimization over loss coefficients, training length, and learning rates for each of our similarity measures using Bayesian hyperparameter optimization. Of special note is the number of prototypical parts $P$ (which represent image patches) initialized the model. The original Prototypical Net fixed this number (arbitrarily to 10 classes), which inspired many future works to reduce the number of prototypes through prototype sharing (e.g., [39, 47, 49]), often citing improved interpretability via a smaller number of prototypes. Since the number of initialized prototypes $P$ is not generally equivalent to the number of learned prototypes, prototype sparsity has neither been consistently optimized nor reported in previous studies as we do here. A complete description of the method and its hyperparameters is embedded in layers of a PPN, the latent prototypical parts may be decoupled from their source images. The projection step ties prototypical parts to source images. This makes traditional early stopping, which relies on saturation of a target metric at an arbitrary epoch, unsuitable for PPNs.

To address this, we introduce a novel early stopping scheme specific to PPNs. Let $E_{proj} = \{j : epoch \, j \text{ is a projection epoch} \land E_{proj} = j + 1 \text{ for } j \in E_{proj}\}$. $v_{Acc}$, where $j \in E_{proj}$ captures the model’s current performance; whereas $E_{proj}$ captures the model’s potential to improve performance after future projections. Patience is exhausted at projection epoch $e \in E_{proj}$ when there has been no improvement in $v_{Acc}$ before or after projection for a specified number of consecutive projection epochs. Formally, improvement is defined as 

$$
v_{Acc, e} \geq \max_{j \in E_{proj}} v_{Acc, j} \text{ or } v_{Acc, e-1} > 1 \text{ max }_{j \in E_{proj}, j \leq e-1} v_{Acc, j}.
$$

In our experiments, we use a patience of 2 projection epochs and halt training once exhausted. We select the model with the best $v_{Acc}$ that has undergone no embedding layer training prior.

## 3.4 User and Model Perception of Similarity

In addition to assessing model semantics on quality metrics, we performed a user study assessing how well human perception of prototype similarity matches the model’s latent space.
```

### --- Page 0005 ---

```markdown
![Accuracy-Prototype Count Frontier of Cosine Ablation. Best Test models and their corresponding sparsities. Top Row: Cub200. Bottom Row: Stanford Dogs. Across all datasets and backbones, the accuracy-sparsity frontier for cosine models (dashed gray line) is beyond that of $l_2$ models (dotted gray line). ProtoNet, TestNet, and ST-ProtoNet all improve on both dimensions using cosine similarity. Neither ProtoTree nor Deformable improve sparsity with cosine, but their accuracies without it are not competitive with other models. Complete results are available in Table 12.](assets/page_0005_img_1.png)

4. Evaluating Causes of Accuracy Improvement

We performed two sets of hyperparameter optimizations on the accuracy objective directly comparing $l_2$ models to cosine models. In the Cosine Ablation experiments, we ran hyperparameter optimization with fixed GPU-hour budgets for five PPN variants. We compared cosine and $l_2$ similarities to establish how far cosine abounds the Pareto frontier of accuracy and sparsity. In the Optimized ProtoPNet experiments, we instead ran hyperparameter optimization on ProtoPNet using optimal configuration.

For both sets of experiments, we considered three families of CNN backbone: DenseNet-161 [24], ResNet-50 [22], and VGG-19 [51]. All backbones were pretrained on ImageNet.

4.1. Cosine Ablation

We performed hyperparameter optimization five different models each with (1) ProtoPNet [11], (2) Deformable ProtoPNet [17], with a slight variation in the implementation of the Deformable layer (see Supplemental Section 9), (3) TestNet [5], (4) ST-ProtoPNet [58], and (5) ProtoTree [39] which uses a numerically stabilized $z$ from [39]. We also apply
```

### --- Page 0006 ---

```markdown
| Metric         | Acc #Proto | Acc #Proto | Acc #Proto |
|----------------|------------|------------|------------|
| Backbone       | DN161      | RN50       | VGG19      |
| ProtoNet       | 75.4       | 1300       | 72.6       |
| ST-ProtoNet    | 82.2       | 550       | 80.8       |
| Deformable     | 81.2       | 4400       | 75.7       |
| ProtoTree      | 82.2       | 202       | 76.2       |
| BOL2-PPN       | 69.7       | 2540       | 2493       |
| BOL2-PPN*      | 76.6       | 1932       | 78.3       |
| BOCs-PPN       | 80.5       | 1129       | 85.9       |
| BOCs-PPN*      | 82.2       | 633       | 87.0       |

Successive Halving [32] to eliminate the bottom 50% of runners after the second project step, doubling projection patience thereafter.

**Result: Cosine Similarity Provides Superior Performance to Euclidean Distance.** Figure 3 shows that the Pareto frontier of accuracy-number of prototypes is advanced by replacing $l_2$ similarity with cosine – in every setting, changing the metric used by a given model from $l_2$ to cosine improved accuracy. Moreover, the worst cosine model performed better than the best $l_2$ model with one exception – only ProtoTree has worse accuracy using cosine than any $l_2$ model for any dataset we tested (Supplementary Table 13). In these optimizations (which use a maximum of 16 prototypes per class), ProtoNet and ST-ProtoNet are the most accurate for two different backbone-dataset combinations, respectively, and accuracy and sparsity between the ProtoNet and ST-ProtoNet are similar throughout: the biggest difference is in DenseNet-161 for Stanford Dogs, where ProtoNet has 86.4% accuracy with 706 prototypes and ST-ProtoNet has 81.4% with 800 prototypes.

Normalization alone does not resolve the differences between cosine and $l_2$ models. Supplementary Table 12 shows that cosine and normalized $l_2$ models performed similarly between the best models (≤ 0.3% difference in optimal accuracy) when using DenseNet-161 and ResNet-50, but that VGG-19's normalized $l_2$ performance is almost identical to unnormalized $l_2$ on CUB-200. This arises, in part, because of the quality of the clustering structure when the first project step occurs during training – cosine and normalized-$l_2$ models with ResNet-50 and DenseNet-161 backbones consistently have $\epsilon_{act}$ that exceeds their separation $\epsilon_{sep}$ at that point. This indicates the clustering structure already has good discriminatory power. Conversely, while VGG-19 cosine models exhibit this property, VGG-19 normalized-$l_2$ models do not. No unnormalized $l_2$ model does so consistently. See Supplementary Section 10.

**4.2 Optimized ProtoPNet**

We hyperparameter-optimized ProtoPNet with $l_2$ and cosine similarities for 216 GPU hours for multiple backbone-dataset combinations to isolate the effects of hyperparameter optimization. We evaluated on four datasets: CUB-200 (both cropped and un-cropped), Stanford Cars [14], and Stanford Dogs [63]. The models resulting from the table are BOL2-For-Bayesian-Optimized: BOL2-ProtoNet, and BOCs-ProtoNet.

These baselines do not provide the number of prototypes per optimization – only the initial hyperparameter. We thus estimated sparsity of the baseline models using the average number of prototypes in the top 10% of models in our Cosine Ablation experiments.

**Result: Hyperparameter Tuning with Cosine Similarity Approaches the Accuracy-Sparsity Frontier.** As shown in Table 1, simply changing the similarity metric of ProtoPNet to cosine similarity achieves accuracy competitive with the state-of-the-art and models 2 to 10 times sparser than those without shared prototypes (ProtoTree, ProtoPool) using the ResNet-50 backbone. Tuning yields inferior results for VGG-19. Tuned models are 2-6 percentage points behind leading models using DenseNet-161 but are also 2 to 10 times sparser than those models. In total, this evidence suggests that a large part of the superior accuracy of models such as Deformable ProtoPNet [17], TesNet [59] and ST-ProtoPNet [58] can be attributed to their use of cosine similarity. Of the gap that remains, part of the explanation lies in the number of prototypes, which appears to be particularly important for VGG-19 models. This is not incidental – aside from the use of cosine similarity, all three models made contributions that enable learning a greater diversity of prototypes. While cosine similarity alone does not grant ProtoPNet state-of-the-art accuracy, it substantially closes the gap and moves ProtoPNet to the accuracy-sparsity frontier for DenseNet-161 and ResNet-50.
```

### --- Page 0007 ---

```markdown
![Overall Cosine and Euclidean Preference Pie Chart](assets/page_0007_img_1.png)

![Distribution of Cosine and Euclidean Preference by Sample Bar Chart](assets/page_0007_img_2.png)

5. Optimizing for Interpretability
-----------------------------------

Having shown that simply applying cosine similarity and carefully tuning hyperparameters is sufficient to achieve a large part of the recent performance gain in the literature, we now turn to investigate whether these corresponded to improved prototype semantics. First, we performed a user study to understand if cosine-learned prototypes better captured human perception of similarity. Then, we assessed CUB200 models on prototype quality metrics. We ran Bayesian hyperparameter optimization for 12 computational days using the prototype index $z_{proto, score}$.

### Users Prefer Cosine-Trained Prototypes
Users in our user study assessed the cosine trained prototypes to be more similar than $l_2$ prototypes to their reference patch for 34 of 54 samples (63.3%). This was a total of 816 of 1360 (60.4%) responses. The preference is statistically significant ($p < 0.001$) after accounting for individual reader effects. There is a wide distribution of preferences across the samples; users had unanimous preferences on only 11 samples (7 cosine, 4 $l_2$, Figure 4). In total, users provided us clear evidence in favor of cosine-learned prototypes.

### Cosine Prototypes Improve over $l_2$ in Semantics
We computed the accuracy, stability, consistency, and sparsity of the prototypes of models trained during the accuracy-only optimization and joint optimization, and compared these metrics. Table 2 shows the results of this evaluation. Cosine prototypes score better on every measure than $l_2$ prototypes, regardless of the optimization strategy. In particular, cosine prototypes have consistency $\geq 90.0$ on all backbones, indicating they have tight localization to specific parts of the image. Joint optimization does not solve the prototype metric problem for $l_2$. Supplementary Figure 7 shows this is 

| Metric | BOL2-PPN | BOCs-PPN |
|--------|----------|----------|
|        | Acc A-PS | Acc A-PS |
| DIn161 | 69.7     | +0.2     | 78.0     | 75.6     |
| Sparsity | 7.3     | +5.5     | 37.0     | 42.8     |
| Consistency | 66.1 | 83. +17.7 | 100.0 | 100.0 |
| Stability | 42.1 | -0.5 | 58.1 | 56.8 |

| Metric | BOL2-PPN | BOCs-PPN |
|--------|----------|----------|
|        | Acc A-PS | Acc A-PS |
| REns0  | 85.4     | 83.6     | 86.9     | 85.7     |
| Sparsity | 6.7     | +1.2     | 21.7     | 42.6     |
| Consistency | 53.3 | 35.0 | 99.9 | +0.5 |
| Stability | 31.7 | -3.4 | 51.5 | 53.6 |

| Metric | BOL2-PPN | BOCs-PPN |
|--------|----------|----------|
|        | Acc A-PS | Acc A-PS |
| VGG16  | 67.6     | 67.1     | 71.1     | 70.8     |
| Sparsity | 7.6     | -0.9     | 30.5 | 41.7 |
| Consistency | 81.7 | +0.4 | 100.0 | 100.0 |
| Stability | 46.4 | +2.0 | 63.8 | 66.2 |
```

### --- Page 0008 ---

```markdown
| **6. Discussion**                                                                                                           | **7. Conclusion**                                                                                                           |
|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Similarity metrics for high-dimensional analysis have been studied generally [1, 7, 45, 56] and in the context of neural network embeddings [38, 53, 64]. However, a satisfying theoretical justification for the choice between Euclidean, cosine, and other similarity measures remains elusive [18, 43]. In the absence of strong theoretical justification, neural network training has largely defaulted to normalized similarity measures like cosine similarity that improve numerical stability [12, 13, 16, 27, 36, 44]. | We thoroughly studied the impact of cosine similarity on the performance of prototypical-part networks (PPNs). We introduced a novel interpretability metric and stopping criteria tailored for case-based DNN’s which allowed us to systematically apply hyperparameter tuning to PPNs. These experiments allow us to attribute accuracy and interpretability improvements. We used this approach to train models with superior accuracy, showing that cosine similarity is responsible for much of the performance improvement in recent PPN models. We observed that models optimized with cosine similarity for interpretability were sparser and more stable. Finally, we demonstrated that these improvements accord with human judgments of prototype similarity. |
| Part of the challenge is the well-known fact that cosine similarity is proportional to normalized $l_2$ distance. The neighborhoods induced in nearest-neighbor space are equivalent between normalized $l_2$ and cosine [19]. We tested a normalized version of ProtoPNet’s similarity function ($\gamma_g$) to establish normalization explained the difference in training stability and accuracy between $l_2$-similarity and cosine similarity models (detailed in Supplementary Section 10). While normalization did provide some performance improvement, cosine-similarity models were more accurate and robust to hyperparameter selection. This implicates the logarithmic kernel from ProtoPNet’s original similarity function. The only remaining difference between the two settings. Critically, normalized $l_2$ distance with the logarithmic kernel is not proportional to cosine similarity. | This work has substantial implications for future research on case-based DNNs and for their practical use. For researchers, our results suggest that future case-based DNNs should default to cosine similarity to improve training stability, model performance, and prototype semantics. Additionally, researchers should be aware of the trade-off between sparsity and accuracy in the current generation of PPNs and work to reduce this limitation in the future. For practical applications, the training stability of cosine similarity suggests that similar variants of PPNs are likely to provide competitive performance for many applications if implemented with cosine distance and systematic hyperparameter tuning. |
| In Supplementary Section 10, we demonstrated that $\gamma_{cos}$ consistently induces a better clustering structure during the training warm-up when compared to $\gamma_e$ and $\gamma_g$ (normalized $l_2$) as measured by $\ell_{ent}$ and $\ell_{sep}$. We further showed that the quality of this initial clustering structure is correlated with final model performance. However, the clustering structure arises from the entire PPN training regime, of which the similarity measure $\gamma$ is only one part. Improving this clustering structure through other means could similarly | Limitations. There are other desirable qualities of prototypes that are not accounted for in the prototype metrics used in this study. As such, we cannot say that our optimization is objective represents and aids in the selection of prototype qualities, only that it expresses some desirable qualities. The even weighting of our optimization was chosen because of uncertainty of what optimized values were achievable, but it may not represent an appropriate weighting for any given task. While we did optimize over the length of training before the first project phase, it is possible that training longer pre-project phases would remove the differences between $l_2$ and cosine-similarity models based on improved clustering at the first project. If this were the case, reduced training time would still justify the use of cosine similarity. |
| improve stability, performance, and semantics of PPNs.                                                                    | Societal Impact. The question of how to make deep learning algorithms for computer vision interpretable is one of the most important considerations for trust in AI. The answers to this could heavily impact self-driving cars, radiology, and facial recognition in policing. |
```

### --- Page 0009 ---

```markdown
# Acknowledgments
This work was funded by the National Science Foundation Division of Computer and Network Systems under grant HRD-2222336. Additionally, this material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE 2139754. Thank you to the anonymous reviewers whose feedback greatly improved this manuscript.

# References
| No. | Citation |
|-----|----------|
| [1] | Charu C. Aggarwal, Alexander Hinneburg, and Daniel A. Keim. On the surprising behavior of distance metrics in high dimensional space. In *Database Theory — ICDT 2001*, pages 420–434, Berlin, Heidelberg, 2001. Springer Berlin Heidelberg. |
| [2] | Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. |
| [3] | Varun Babbar, Zhicheng Guo, and Cynthia Rudin. What is different between these datasets? arXiv preprint arXiv:2403.05652, 2024. |
| [4] | Sebastian Bach, Alexander Binder, Grégoire Montavon, Frederick Klauschen, Lukas-Robert Müller, and Wojciech Samek. On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation. *PloS One*, 10 (2015):e0130140. |
| [5] | Alina Jade Barnett, Fides Regina Schwartz, Chaofan Tao, Chaofan Chen, Yinda Ren, Joseph Y. Lo, and Cynthia Rudin. A case-based interpretable deep learning model for classification of mass lesions in digital mammography. *Nature Machine Intelligence*, 3(12):1061–1070, 2021. |
| [6] | Alina Jade Barnett, Zhicheng Guo, Jin Jing, Wendong Ge, Peter W. Karp, Wan Yee Kong, and Karissa Aline Herlopian, Lakshman Aroor Jayapagal, Olga Taraschenko, et al. Improving clinician performance in classifying eye patterns on the ictal-interictal injury continuum using deep learning. *NEJM JI*, 16(1):A102303301, 2024. |
| [7] | Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, and Uri Shaft. When is “nearest neighbor” meaningful? In *Database Theory—ICDT’99: 7th International Conference Jerusalem, Israel, January 10–12, 1999 Proceedings 7*, pages 217–235, 1999. |
| [8] | Lukas Biawiel. Experiment Tracking with Weights and Biases, 2020. Software available from wandb.com. |
| [9] | Bernd Bischl, Martin Binder, Michel Lang, Tobias Plelik, Jakob Richter, Stefan Coors, Janek Thomas, Theresa Ullmann, Marek Becker, Anne-Laure Boulesteix, et al. Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 13(2):e1484, 2023. |
| [10] | Merle Leendert de Bouter, Javier Llorente, Zeno Gerads, and Marten Wiering. ProtoExplorer: Interpretative forensic analysis of debriefed videos using prototype exploration and visualization. *Information Visualization*, 2023. |
| [11] | Chaofan Chen, Oscar Li, Daniel Tio, Alina Barnett, Cynthia Rudin, and Jonathan K. Su. This Looks Like That: Deep Learning for Interpretative Image Recognition. *Advances in Neural Information Processing Systems*, 32, 2019, 1-2. |
| [12] | Ting Chen, Simon Kornblith, Mohammad Noroozi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In *International conference on machine learning*, pages 1597–1607. PMLR, 2020. |
| [13] | Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 15750–15758, 2021. |
| [14] | Afshin Dehghan, Seyed Zain Masoodi, Guang Shu, Enrique Ortiz, et al. View independent vehicle make, model and color recognition using convolutional neural network. arXiv preprint arXiv:1702.07121, 2017. |
| [15] | Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pages 248–255. IEEE, 2009. |
| [16] | Jianfeng Zhang, Jia Guo, Naman Xu, and Stanculescu Zafeiriou. Aferace: Adding average margin loss for deep face recognition. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 4690–4699, 2019. |
| [17] | Jon Donnelly, Alina Jade Barnett, and Chaofan Chen. Deformable ProtoNet: An Interpretable Image Classifier using Deformable Prototypes. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022. |
| [18] | Stephen France and Douglas Carroll. Is the distance one persistent effect overstated? Some theory and experimentation. In *Machine Learning and Data Mining in Pattern Recognition Conference, MLDM 2009*, Leipzig, Germany, July 6–8, 2009. Proceedings 6, pages 280–294. Springer, 2009. |
| [19] | Stephen L. France, J Douglas Carroll, and Hui Xiong. Distance metrics for high dimensional nearest neighbor recovery: Compression and normalization. *Information Sciences*, 184 (1):92–110, 2012. |
| [20] | Srishti Gautam, Anchee Beckitt, Shine Tan, Suabia Salahuddin, Robert Jensen, Marina Hohne, and Michael Kampffmeyer. Prototracer: A trustworthy self-explaining probabilistic variational model. In *Advances in Neural Information Processing Systems*, pages 17940–17952. Curran Associates, Inc., 2022. |
| [21] | Jean-Bastien Gril, Florian Strub, Florent Althé, Corentin Tallec, Pierre Richernon, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap over your latent-a new approach to self-supervised learning. *Advances in neural information processing systems*, 33:21217–21284, 2020. |
| [22] | Kaiming He, Xiangyu Zhang, Shaogang Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016. |
| [23] | Tianyu Hua, Wenxiao Wang, Zhixi Xue, Sucheng Ren, Yue Wang, and Hang Zhao. On feature decorrelation in self-supervised learning. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021. |
```

### --- Page 0010 ---

```markdown
| Reference                                                                 | Page(s)     |
|---------------------------------------------------------------------------|-------------|
| [24] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. Densely Connected Convolutional Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4700–4708, 2017. 5 | 9598–9608   |
| [25] Qihan Huang, Mengqi Xue, Wengui Huang, Haofei Zhang, Jie Song, Yongcheng Jing, and Mingli Song. Evaluation and Improvement of Interpretability for Self-Explainable Prototype Networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2011–2020, 2023. | 9609–9610   |
| [26] Qihan Huang, Mengqi Xue, Wengui Huang, Haofei Zhang, Jie Song, Yongcheng Jing, and Mingli Song. Evaluation and Improvement of Interpretability for Self-Explainable Prototype Networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2011–2020, 2023. | 9611–9612   |
| [27] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pages 448–456, 2015. | 9613–9614   |
| [28] Max Kaeberger, Karen Simonyan, Andrew Zisserman, et al. Spatial Transformer Networks. Advances in Neural Information Processing Systems, 28, 2015. | 9615–9616   |
| [29] Lin Jin, Pascal Vincent, Zoran LeCun, and Yoshua Bengio. Understanding dimensionality reduction in contrastive self-supervised learning. arXiv preprint arXiv:2107.00428, 2021. | 9617–9618   |
| [30] Eoin M. Kenny, Mycal Kutz, and Julie Shain. Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes. In The Eleventh International Conference on Learning Representations, 2023. | 9619–9620   |
| [31] Ben Kim, Martin Wattenberg, J. Justin Gilmer, Carrie Cai, James Wexler, Fernando Viegas, et al. Interpretable By-Design Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV). In International Conference on Machine Learning, pages 2686–2677. PMLR, 2018. 2 | 9621–9622   |
| [32] Manoj Kumar, George E Dahl, Viyasu Vasu, and Mohammad Norouzi. Parallel architecture and hyperparameter search for visual explanations with classification. arXiv preprint arXiv:1805.10255, 2018. 6 | 9623–9624   |
| [33] Aaron J. Li, Robin Netzorg, Zhihan Cheng, Zhuoqin Zhang, and Bin Yu. Improving prototypical visual explanations with reward reweighting, resolution, and retraining. 2024. | 9625–9626   |
| [34] Oscar Li, Hao Liu, Chaofan Chen, and Cynthia Rudin. Deep Learning for Case-Based Reasoning Through Prototypes: A Neural Network That Explains Its Predictions. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. 1, 2 | 9627–9628   |
| [35] Scott L. Lundberg and Su-In Lee. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30, 2017. 2 | 9629–9630   |
| [36] Chunjie Liu, Jianfeng Zhang, Xiaole Xue, Lei Wang, Rui Ren, and Qiang Yang. Cosine normalization: Using cosine similarity instead of product in neural networks. In Artificial Neural Networks and Machine Learning–ICANN 2018: 27th International Conference on Artificial Neural Networks, | 9631–9632   |
| [37] Chi Yu Ma, Brandon Zhao, Chaofan Chen, and Cynthia Rudin. This Looks Like Those: Illuminating Prototypical Concepts Using Multiple Visualizations. Advances in Neural Information Processing Systems, 36, 2024. | 9633–9634   |
| [38] Pascal Mettes, Elise Ver Def, and Ceas Snook. Hyperspherical Prototype Networks. Advances in Neural Information Processing Systems, 32, 2019. | 9635–9636   |
| [39] Meike Nauta, Ron Van Bree, and Christin Seifert. Neural Prototype Trees for Interpretable Fine-Grained Image Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14933–14943, 2021. 1, 2, 4, 5 | 9637–9638   |
| [40] Meike Nauta, Jörg Schlotterer, Marieke van Keulen, and Christian Seifert. PRT-Net: Patch-Based Inductive Prototypes for Interpretable Image Classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2744–2753, 2021. 1, 2, 3 | 9639–9640   |
| [41] Mateusz Pach, Dawid Rymarczyk, Koryna Lewandowska, Jacek Tabor, and Bartosz Zieliński. Unambiguous prototypical parts network for user-centric interpretable computer vision, 2023. | 9641–9642   |
| [42] Adam Paszek, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelsheva, Antigoni, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems, 32, 2019. | 9643–9644   |
| [43] Chuba Peng, Zhipeng Gui, and Huayi Wu. Interpreting the curse of dimensionality from distance concentration and manifold effect, 2025.  | 9645–9646   |
| [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2 | 9647–9648   |
| [45] Miloš Radanović, Alexandros Nanopoulos, and Mirjana Ivanović. Nearest neighbors in high-dimensional data: The emergence and influence of hubs. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 865–872, 2009. | 9649–9650   |
| [46] Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5):206–215, 2019. | 9651–9652   |
| [47] Dawid Rymarczyk, Łukasz Struski, Jacek Tabor, and Bartosz Zieliński. Prototypical parts sharing for similarity discovery in interpretable image classification. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 1420–1430, 2021. | 9653–9654   |
| [48] Dawid Rymarczyk, Łukasz Struski, Michał Górczak, Koryna Lewandowska, Jacek Tabor, and Bartosz Zieliński. Interpretable Image Classification with Differentiable Prototypes. | 9655–9656   |
```

### --- Page 0011 ---

```markdown
# References

| No. | Citation                                                                                                                                                                                                 |
|-----|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [50] | Mikolaj Sacha, Dawid Rymarczyk, Lukasz Struski, Jacek Tabor, and Bartosz Zielinski. ProtoSeg: Interpretable Semantic Segmentation with Prototypical Parts. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1481–1492, 2023. |
| [51] | Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. In Proceedings of the 3rd International Conference on Learning Representations (ICLR), 2015.                                           |
| [52] | Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep Inside Convolutional Networks: Visualising image Classification Models and Saliency Maps. arXiv preprint arXiv:1312.6034, 2013.                                                  |
| [53] | Harald Steck, Chaitanya Ekanadham, and Nathan Kallus. Is cosine-similarity of embeddings really about similarity? In Companion Proceedings of the ACM Web Conference 2024, pages 887–890, 2024.                                               |
| [54] | Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic Attribution for Deep Networks. In International Conference on Machine Learning, pages 3319–3328. PMLR, 2017.                                                                 |
| [55] | Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The iNaturalist Species Classification and Detection Dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8769–8778, 2018. |
| [56] | Robin Vandale, Bo Kang, Tijl De Bie, and Yvan Saeys. The curse revisited: When are distances informative for the ground truth in noisy high-dimensional data? In International Conference on Artificial Intelligence and Statistics, pages 2158–2172. PMLR, 2022. |
| [57] | C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.                                                                 |
| [58] | Chong Wang, Yuyun Liu, Yunhong Chen, Fengbei Liu, Yu Tian, Davis McCarthy, Helen Frazer, and Gustavo Carneiro. Learning support and prototype diversity for interpretable image classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2062–2072, 2023. |
| [59] | JiaqI Wang, Huafeng Liu, Xinyue Wang, and Liping Jing. Interpretable Image Recognition by Constructing Transparent Embedding Space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 895–904, 2021.                  |
| [60] | Tom Nuno Wolf, Sebastian Pöslterl, and Christian Wachinger. Don’t PANIC: Prototypical Additive Neural Network for Interpretable Classification of Alzheimer’s Disease. In International Conference on Information Processing in Medical Imaging, pages 82–94. Springer, 2023. |
| [61] | Tom Nuno Wolf, Fabian Bongartz, Anne-Marie Rickmann, Sebastian Pöslterl, and Christian Wachinger. Keep the faith: Faithful explanations in convolutional neural networks for case-based reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5921–5929, 2024. |
| [62] | Julia Yang, Alina Jade Barnett, Jon Donnelly, Savtik Kishore, Jerry Fang, Fides Regina Schwartz, Chaofan Chen, Joseph Y Lo, and Cynthia Rudin. FPN-IAI-BI: A multi-scale interpretable deep learning model for classification of mass margins in digital mammography. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5003–5009, 2021. |
| [63] | Peison Zhao, Lingxi Xie, Ya Zhang, and Qi Tian. Universal-to-specific framework for complex action recognition. IEEE Transactions on Multimedia, 23:3441–3453, 2020.                                                                 |
| [64] | Kaitlyn Zhou, Kawin Ethayarajh, Dallas Card, and Dan Jurafsky. Problems with cosine as a measure of embedding similarity for high frequency words, 2022.                                                                                     |
```


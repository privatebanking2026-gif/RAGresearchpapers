# Nested Hyperbolic Spaces for Context-Aware LLM Reasoning- A Geometric Framework

### --- Page 0001 ---

```markdown
# Hikage Morino¹
¹Affiliation not available

February 11, 2026
```


### --- Page 0002 ---

```markdown
# Nested Hyperbolic Spaces for Context-Aware LLM Reasoning: A Geometric Framework

**Hikage Morino**  
Independent Researcher  
morchella@spring.nifty.jp  

January 14, 2026

## Abstract

We propose a dual hyperbolic space architecture for context-aware reasoning in large language models, featuring nested hierarchical structures and ridge-based secondary space positioning. The framework addresses fundamental challenges in managing multiple conversational contexts through principled geometric design. Our key contributions are: (1) a dual space architecture maintaining geometrically distinct primary and secondary hyperbolic spaces for parallel reasoning, (2) ridge-based positioning that places secondary centers at optimal transition points between contexts, (3) nested hierarchical structures with adaptive depth assignment integrating geometric distance and semantic relevance, and (4) a comprehensive theoretical framework connecting hyperbolic geometry, information theory, and cognitive principles. We provide detailed implementation proposals and evaluation protocols, with empirical validation remaining as important future work. The geometric substrate we propose may have applications beyond context management, offering insights into the organization of conceptual knowledge and the structure of multi-perspective reasoning.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in multi-turn dialogue and complex reasoning tasks. However, they face fundamental challenges in context management: limited token windows necessitate careful selection of what to retain, while context switching between topics incurs significant computational costs. Existing approaches—from memory-augmented architectures to retrieval-augmented generation—address these challenges primarily through engineering solutions, lacking a unified geometric framework for understanding the underlying information access patterns.

Human reasoning, by contrast, naturally operates across multiple parallel contexts. We maintain “surface” conversations while simultaneously processing background thoughts—an inner dialogue that often leads to sudden insights when these parallel streams converge. Moreover, humans routinely synthesize ideas across multiple conversational partners: one might discuss abstract theory with a colleague, explore practical implementations with another, then integrate insights from both. Current LLM architectures, operating within a single flat context window, cannot naturally support such multi-threaded reasoning. We propose that hyperbolic geometry provides the mathematical substrate for architectures that can.

Building on this insight, we introduce a dual hyperbolic space architecture for LLM reasoning. Our framework features: (1) a primary hyperbolic space centered on the current conversational context, (2) secondary hyperbolic spaces for parallel exploration of related concepts, (3)
```


### --- Page 0003 ---

```markdown
ridge-based positioning that places secondary space centers at optimal “viewpoints” between contexts, and (4) nested hierarchical structures where concepts are organized in self-similar hierarchies with adaptive depth. This architecture provides a principled geometric substrate for managing context switching, information retrieval, and multi-scale reasoning.

Our key contributions are:

- A dual hyperbolic space architecture unifying context management and parallel reasoning
- Ridge-based secondary space positioning for optimal context transitions
- Nested hierarchical structures with adaptive depth assignment
- A comprehensive theoretical framework with proposed implementation and evaluation protocols

The remainder of this paper is organized as follows. Section 2 reviews related work. Sections 3–6 develop our theoretical framework. Section 7 presents our proposed implementation. Section 8 discusses implications and limitations. Section 9 concludes.

## 2 Related Work

**Hyperbolic Neural Networks.** The use of hyperbolic geometry in machine learning has gained significant attention following Nickel and Kiela’s introduction of Poincaré embeddings [14], which demonstrated that hyperbolic spaces naturally capture hierarchical structures with far fewer dimensions than the Euclidean alternatives. Subsequent work extended this to the Lorentz model [15], hyperbolic attention mechanisms [8], and hyperbolic graph neural networks [3]. However, these approaches primarily focus on single-space embeddings optimized for specific tasks. Our work differs by proposing a dual space architecture where multiple hyperbolic spaces operate in parallel, with explicit mechanisms for transitioning between them.

**Context Management in LLMs.** Managing context in long conversations remains a central challenge for LLMs. Memory-augmented transformers [16] and neural Turing machines [7] introduce external memory mechanisms, while retrieval-augmented generation (RAG) [10] leverages information retrieval to expand effective context. More recent work explores hierarchical context representations [5] and dynamic context compression [13]. These approaches treat context management primarily as an engineering problem—selecting what to retain or retrieve. In contrast, we provide a geometric framework where context relationships are embedded in the structure of hyperbolic space itself, and “context switching” corresponds to movement through this geometry.

**Information Theory and Computational Costs.** The relationship between information access and computational cost has been explored in algorithmic information theory [11] and more recently in the concept of epiplexity [6], which measures information complexity relative to a computationally bounded observer. This perspective aligns with our view that distance in hyperbolic space should encode not just semantic similarity but also the computational cost of context transitions. However, existing information-theoretic frameworks do not provide explicit geometric architectures for managing these costs in neural systems.

**Hierarchical and Multi-Scale Representations.** Self-similar structures have been explored in neural architecture design [1, 12], often motivated by biological neural organization. Multi-scale attention mechanisms [4] and hierarchical transformers [17] capture information at multiple resolutions. Our nested hyperbolic structures share this multi-scale philosophy but add geometric principles: each level of the hierarchy is itself a hyperbolic space, and the “cost” of accessing information scales geometrically with depth.
```

### --- Page 0004 ---

```markdown
## 3 Theoretical Framework

We develop a geometric framework for context-aware reasoning based on hyperbolic spaces and their hierarchical organization. This section establishes the mathematical foundations for our dual space architecture.

### 3.1 Hyperbolic Geometry Primer

Hyperbolic geometry provides a natural substrate for representing hierarchical structures. We work primarily in the Poincaré disk model, where hyperbolic n-space $H^n$ is represented as the open unit ball in $\mathbb{R}^n$:

$$
H^n = \{x \in \mathbb{R}^n : \|x\| < 1\} \tag{1}
$$

equipped with the Riemannian metric:

$$
ds^2 = \frac{4 \|dx\|^2}{(1 - \|x\|^2)^2} \tag{2}
$$

The hyperbolic distance between points $x, y \in H^n$ is:

$$
d_H(x, y) = \operatorname{arcosh} \left( 1 + \frac{2 \|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)} \right) \tag{3}
$$

**Key properties:** (1) Geodesics (shortest paths) are circular arcs perpendicular to the boundary, (2) the volume of a ball of radius $r$ grows exponentially as $\sim e^{(n-1)r}$, enabling efficient embedding of tree-like structures, and (3) distances grow rapidly near the boundary, naturally encoding hierarchy.

For computational implementation, we use the exponential and logarithmic maps to convert between Euclidean embeddings and hyperbolic coordinates:

$$
\exp_{H^0}(v) = \tanh(\|v\|) \frac{v}{\|v\|} \tag{4}
$$

$$
\log_{H^0}(x) = \operatorname{arctanh}(\|x\|) \frac{x}{\|x\|} \tag{5}
$$

### 3.2 Conceptual Space as Hyperbolic Manifold

We model the space of concepts as a hyperbolic manifold $H^n$ ($n = 768$ for typical transformer embeddings). Each concept $c$ is represented as a point in this space, with the following geometric interpretation:

**Center point:** The origin (or current center) represents the active focus of attention. Concepts near the center are readily accessible—they form the current working context.

**Distance as cost:** The hyperbolic distance $d_H(c_0, c)$ from the center $c_0$ to concept $c$ encodes the computational and cognitive cost of shifting attention to $c$. This distance can be interpreted as an epiplexity measure [6]—the information complexity relative to a computationally bounded observer.
```


### --- Page 0005 ---

```markdown
# Hierarchical organization

Parent-child relationships in conceptual hierarchies correspond to radial paths from center to periphery. Abstract concepts reside near the center; specific instances spread toward the boundary. The exponential volume growth naturally accommodates the combinatorial explosion of specific instances.

## 3.3 Energy Landscape Interpretation

We interpret the hyperbolic space as an energy landscape where reasoning corresponds to navigation through this geometry:

$$
E(c) = \alpha \cdot d_{H}(c_{center}, c) \tag{6}
$$

where $\alpha > 0$ is a scaling parameter. This energy represents the computational cost (attention, memory access, context retrieval) required to bring concept $c$ into focus.

**Valleys:** Local minima of $E$ correspond to stable contexts—coherent clusters of related concepts. The current conversation occupies one such valley.

**Ridges:** Saddle points between valleys represent transitional concepts that provide “visibility” into multiple contexts. These ridge points will play a crucial role in our dual space architecture (Section 5).

**Context switching:** Moving from one valley to another requires traversing the energy landscape. The minimal path is a geodesic, but the cost is determined by the height of the intervening ridge. Large context switches require crossing high ridges—they are energetically expensive.

This energy interpretation connects our geometric framework to information theory: the “potential energy” $E(c)$ corresponds to the computational resources needed to access information at $c$, aligning with the cognitive perspective [6] where information depends on the observer’s computational constraints.

## 3.4 Nested Hierarchical Structures

Our framework extends beyond a single hyperbolic space to a nested hierarchy where concepts are organized in self-similar structures with adaptive depth:

$$
ConceptNode(c) = (e_{global}, H_{local}, children) \tag{7}
$$

Here $e_{global} \in \mathbb{H}^{n}$ is the concept’s position in the global space, $H_{local} \approx \mathbb{H}^{m} \ (m \ll n)$ is a smaller hyperbolic space encoding the concept’s internal structure, and children are nested ConceptNodes.

**Motivation:** Evidence from cognitive science suggests that human semantic memory exhibits self-similar organizational principles—concepts at different scales of abstraction show similar structural patterns. While proving true mathematical fractal structure would require infinite recursion, we capture this principle through finite-depth nested hierarchies.

**Adaptive depth assignment:** Rather than using distance alone, we combine geometric proximity with semantic relevance:

$$
score(c) = \beta \cdot \left( 1 - \frac{d_{H}(c_{0}, c)}{d_{max}} \right) + (1 - \beta) \cdot relevance(c|c_{0}, ctx) \tag{8}
$$

$$
depth(c) = 
\begin{cases} 
3 & \text{if } score(c) > \tau_{3} \\ 
2 & \text{if } \tau_{2} < score(c) \leq \tau_{3} \\ 
1 & \text{if } \tau_{1} < score(c) \leq \tau_{2} \\ 
0 & \text{otherwise} 
\end{cases} \tag{9}
$$
```

### --- Page 0006 ---

```markdown
where $\beta \in [0, 1]$ balances geometric distance and semantic relevance, and $\tau_1, \tau_2, \tau_3$ are threshold parameters. This allows concepts at the same distance to receive different levels of detail based on their contextual importance.

### Efficient relevance computation:
The relevance score combines precomputed static similarity (O(1) lookup from a cached matrix) with lightweight context-dependent attention:

$$
\text{relevance}(c(c_0), ctx) = (1 - \alpha) \cdot \text{sim}_{\text{static}}(c, c_0) + \alpha \cdot \langle \phi(c), h_{\text{ctx}} \rangle \tag{10}
$$

where $h_{\text{ctx}}$ is the current context representation and $\phi(c)$ is concept $c$'s embedding.

### Information access cost:
To access information at depth $d$ in the hierarchy, one must traverse $d$ nested spaces. The total cost scales as:

$$
C(d) = \sum_{i=1}^{d} \alpha_i \cdot d_{i} \cdot \text{dist}(c^{\text{center}}_i, c^{\text{target}}_i) \tag{11}
$$

where $c^{\text{center}}_i$ and $c^{\text{target}}_i$ are positions within the $i$-th level space. Under appropriate conditions (similar distances at each level), this grows exponentially: $C(d) \sim k^d$ for some $k > 1$.

### Capacity vs. cost trade-off:
The nested structure provides exponential storage capacity (volume grows as $e^{(n-1)}$ at each level) but also exponential access cost for deeply nested information. This naturally implements a form of importance weighting: frequently accessed information should reside at shallow depths, while specialized knowledge can be stored deeply with higher retrieval costs.

## 4 Dual Space Architecture
Having established the geometric foundations, we now describe our dual space architecture for managing parallel reasoning contexts in LLMs.

### 4.1 Motivation: Parallel Reasoning
Human cognition naturally maintains multiple concurrent thought processes. During a conversation, we simultaneously track the explicit dialogue while processing background thoughts—connections to past experiences, alternative framings, potential objections. These parallel streams occasionally converge, producing insights that neither stream alone would generate.

Current LLM architectures operate within a single context window, forcing all reasoning into a sequential, monolithic representation. This creates tension: maintaining detailed information about the current topic leaves little capacity for exploring tangential but potentially relevant concepts.

Our dual space architecture resolves this tension by maintaining one primary space for the current conversational focus and one or more secondary spaces for parallel exploration. Crucially, these are not merely separate attention heads or memory buffers—they are geometrically distinct hyperbolic spaces, each with its own center and organization.

### 4.2 Primary Space
The primary space ($H^n_p, c_p$) represents the current conversational context, where $c_p \in H^n_p$ is the center concept (current focus) and $n = 768$ is the embedding dimension.

#### Organization:
Concepts are organized by their hyperbolic distance from $c_p$, with the nested structure (Section 3.4) determining the level of detail. The primary space is continuously updated as the conversation progresses:
```
![Description of the dual space architecture](assets/page_0006_img_1.png)
```

### --- Page 0007 ---

```markdown
### 4.2 Center Updates

- **Center updates:** When the topic shifts substantively, $c_p$ moves to the new focus concept via a geodesic path.

- **Hierarchy recomputation:** The nested structure is rebuilt around the new center, with depth assignments reflecting the changed perspective.

### Attention Mechanism

Standard transformer attention operates primarily within the primary space, with attention weights modulated by hyperbolic distance:

$$
\text{Attention}(Q,K,V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} - \lambda \cdot D_H \right) V \tag{12}
$$

where $D_H$ is the matrix of hyperbolic distances from the center, and $\lambda > 0$ controls the strength of geometric bias.

### 4.3 Secondary Spaces

Secondary spaces $[H^m_n, c_s]$ are spawned when the system anticipates a context switch or needs to explore alternatives. Each secondary space:

1. Has its own center $c_s$, distinct from $c_p$.
2. Operates in a typically lower-dimensional space ($m = 64-256 < n$).
3. Maintains its own nested hierarchy around $c_s$.
4. Evolves independently from the primary space.

**When to spawn:** Secondary spaces would be created when:

- A ridge point is reached in the primary space (Section 5).
- The system detects a potential topic branch.
- Explicit exploration is requested.

**Multiple secondary spaces:** The architecture supports multiple concurrent secondary spaces, enabling parallel exploration. In practice, we propose limiting this to 2–3 active secondary spaces to control computational cost.

**Independence:** Secondary spaces are not mere “views” of the primary space. They have independent centers and organizations. A concept $c$ may appear near the center in one space (high detail, low access cost) and at the periphery in another (low detail, high cost).

### 4.4 Transition Dynamics

**Entering a secondary space:** When spawning a secondary space centered at $c_s$:

$$
\varphi_{p \to s} : H^m_p \to H^m_s \tag{13}
$$

This mapping involves:

1. **Dimension reduction:** Projecting from $n$ to $m$ dimensions.
2. **Re-centering:** Applying a Möbius transformation that moves $c_s$ to the origin.
```

### --- Page 0008 ---

```markdown
3. Hierarchy rebuild: Constructing a new nested structure around $c_s$

The computational cost of this transition is:

$$
C_{enter} = \alpha \cdot d_{h}(c_p, c_s) + \beta \cdot \text{build\_cost}(c_s, m) \tag{14}
$$

Returning to primary space: The return transition is asymmetric. Since the primary space remains intact, returning is computationally cheaper:

$$
C_{return} = \gamma \ll C_{enter} \tag{15}
$$

This asymmetry is cognitively motivated: humans can easily “snap back” to their main train of thought after a brief digression, whereas initiating the digression requires mental effort. 

Integration: When returning, insights from the secondary space can be integrated into the primary space by updating relevance scores or introducing new concepts near $c_p$.

4.5 Mathematical Formalization

We formalize the dual space system as:

$$
S = \left( \{(H_{inp}, c_p, F_p)\} \cup \{(H_{i}^{m}, c_{si}, F_{si})\}_{i=1}^{k} \right) \tag{16}
$$

where $F_p$ and $F_{si}$ represent the nested structures, and $k$ is the number of active secondary spaces (typically $0-3$).

Transition costs:

$$
C(\text{spawn}(c_s)) = d_{h}(c_p, c_s) + \text{build\_cost}(c_s) \tag{17}
$$

$$
C(\text{return}()) = O(1) \quad (\text{primary space cached}) \tag{18}
$$

5 Ridge-Based Secondary Space Positioning

Having established the dual space architecture, we now address a critical design question: where should we center secondary spaces? This section introduces our ridge-based positioning strategy.

5.1 The Positioning Problem

When spawning a secondary space to explore a related but distinct concept, a naive approach would center it on the most abstract common parent of the current and target concepts. However, abstract concepts reside far from both the current focus and the exploration target, making both contexts difficult to access.

Energy landscape analogy: The current context occupies a valley, while the exploration target lies in another valley. The abstract concept sits at a high point between them. We need a ridge point—a saddle between valleys that provides good visibility into both while minimizing transition costs.

5.2 Ridge Points as Optimal Viewpoints

We define a ridge point $r^*$ as the point on the geodesic between current center $c_p$ and predicted next center $c_{next}$ that maximizes bilateral visibility:

$$
r^* = \arg \max_{r \in \gamma_{(c_p,c_{next})}} \left[ V(r, c_p) + V(r, c_{next}) \right] \tag{19}
$$
```

### --- Page 0009 ---

```markdown
where $\gamma(c_p, c_{next})$ is the geodesic path and $V(r, c)$ measures how well position $r$ can “observe” the neighborhood of concept $c$.  
**Approximate solution:** The geometric midpoint of the geodesic often serves as a good approximation:  
$$
r_{mid} = \exp_{H}^{c_p} \left( \frac{1}{2} \log_{H}^{c_p}(c_{next}) \right) \tag{20}
$$

### 5.3 Bilateral Visibility

The visibility function $V(r, c)$ quantifies how much information about concept $c$'s local neighborhood is accessible from position $r$:

$$
V(r, c) = \sum_{c' \in N(c)} w(c') \cdot \exp(-\lambda \cdot d_H(r, c')) \tag{21}
$$

where $N(c)$ is the neighborhood of $c$, $w(c')$ is the importance weight, and $\lambda > 0$ controls visibility decay.  

**Why bilateral visibility matters:** A secondary space centered at $r^*$ can:
- **Maintain context:** Access information about the current focus $c_p$  
- **Explore ahead:** Access information about the target $c_{next}$  
- **Facilitate transitions:** Bridge the two contexts with minimal total distance  

### 5.4 Computational Advantages

Ridge positioning provides several benefits:  
**Reduced transition cost:** By positioning between contexts:  
$$
C_{total} = d_H(c_p, r^*) + d_H(r^*, c_{next}) \approx d_H(c_p, c_{next}) \tag{22}
$$

**Graceful prediction failure:** If $c_{next}$ is mispredicted, $r^*$ still provides some visibility toward the actual next concept.  
**Natural integration:** Insights from a ridge-positioned secondary space are easily integrated because $r^*$ is “close enough” to $c_p$ to share relevant concepts.  

## 6 Information Access Costs in Nested Hierarchies

Our nested hierarchical structure naturally encodes differential information access costs: concepts near the center are cheap to retrieve, while deeply nested information requires traversing multiple levels.  

### 6.1 The Multi-Level Access Problem

Consider accessing a specific piece of information when the current focus is at the top level. In our nested hierarchy:
1. **Level 0 (primary space):** Current center  
2. **Level 1:** Related concepts in the primary’s local space  
```

### --- Page 0010 ---

```markdown
3. Level 2: Sub-concepts in Level 1’s local spaces

4. Level 3: Detailed information in Level 2’s local spaces

Each level traversal incurs computational and attentional cost.

## 6.2 Cost Components

We model the total access cost as comprising three components:

1. **Distance Cost (Geometric)**:

   $$
   C_{dist}(d) = \sum_{i=1}^{d} \alpha_i \cdot d_{H_i}^{(center, target)} \quad (23)
   $$

   In hyperbolic geometry, this scales superlinearly: $C_{dist} \sim d^{\beta}, \, \beta > 1$.

2. **Contextual Reorientation Cost**:

   $$
   C_{reorient}(d) = \gamma \cdot d \quad (24)
   $$

   This linear term reflects the overhead of loading and interpreting each new local coordinate system.

3. **Decompression Cost**:

   $$
   C_{decompress}(d) = \delta \cdot \sum_{i=1}^{d} entropy(H_i) \quad (25)
   $$

**Total Cost**:

$$
C_{total}(d) = C_{dist}(d) + C_{reorient}(d) + C_{decompress}(d) \quad (26)
$$

This multi-component model can be seen as a geometric realization of epiplexity [6].

## 6.3 Scaling with Depth

Under simplifying assumptions, total cost grows exponentially:

$$
C_{total}(d) \approx k^d, \, k > 1 \quad (27)
$$

This exponential cost scaling is balanced by exponential capacity growth. At depth $d$, total capacity is approximately:

$$
Capacity(d) \sim m^d \cdot e^{(m-1)^{physical}} \quad (28)
$$

## 6.4 Trade-offs and Optimization

**Comparison with flat embeddings**:

| Property          | Flat Embedding         | Nested Hierarchy         |
|-------------------|------------------------|---------------------------|
| Capacity          | Linear in dim          | Exponential in depth      |
| Avg access cost   | $O(1)$                 | $O(k^d)$                  |
| Adaptivity        | None                   | High                      |

**Optimization strategies**:
- **Adaptive depth**: Place important concepts at appropriate depths
```

### --- Page 0011 ---

```markdown
- **Caching:** Frequently accessed deep concepts cached at shallow levels
- **Lazy evaluation:** Materialize deep levels only when needed
- **Dynamic adjustment:** Promote/demote concepts based on usage

Connection to human cognition: This cost structure mirrors human memory access: immediate working memory is fast but limited, while long-term memory is vast but slow to retrieve.

# 7 Proposed Implementation

We propose an implementation strategy for our dual hyperbolic space architecture. This section describes the technical realization without empirical validation, which remains important future work.

## 7.1 System Architecture

Our proposed implementation builds on a pretrained transformer LLM with the following modifications:

- **Hyperbolic embedding layer:** Replace standard Euclidean embeddings with Poincaré ball embeddings. Token embeddings are projected into $H^n$ using exponential map:

$$
e_{hyp} = \tanh\left(\frac{||e_{cul}||}{2} \cdot ||e_{cul}||\right) \cdot ||e_{cul}||
$$

### Dual space manager:

A controller module would maintain:

- **Primary space:** $(H^{768}, c_p, F_p)$ with adaptive nested hierarchy
- **Secondary spaces:** Pool of up to $k = 3$ active spaces
- **Transition logic:** Ridge detection, space spawning, integration

### Nested hierarchy construction:

For each active space center $c$:

1. Compute relevance scores using cached similarity matrix
2. Select top-100 concepts by integrated score
3. Assign depths $\{0, 1, 2, 3\}$ based on score thresholds
4. For concepts with depth $> 0$, recursively build local spaces

## 7.2 Key Algorithms

### Ridge point computation:

```python
def compute_ridge_midpoint(c_p, c_next):
    # Log map: c_next to tangent space at c_p
    v = log_map(c_p, c_next)
    # Map to tangent space
    v_mid = 0.5 * v
    # Exp map: back to hyperbolic space
    r = exp_map(c_p, v_mid)
    return r
```
```

### --- Page 0012 ---

```markdown
# Adaptive depth assignment:

```python
def assign_depth(concept, center, context, beta=0.5):
    # Distance component
    dist = hyperbolic_distance(concept, center)
    dist_score = 1.0 - (dist / d_max)

    # Relevance component
    static_sim = similarity_cache[concept, center]
    context_sim = attention(concept, context)
    relevance = (1-alpha) * static_sim + alpha * context_sim

    # Integrated score
    score = beta * dist_score + (1-beta) * relevance

    # Depth thresholds
    if score > 0.75: return 3
    elif score > 0.5: return 2
    elif score > 0.25: return 1
    else: return 0
```

## 7.3 Computational Considerations

### Complexity analysis:
- Hyperbolic operations: $O(d)$ per operation (numerically stable implementations exist)
- Ridge computation: $O(d)$ (geodesic midpoint)
- Depth assignment: $O(V)$ for vocabulary size $V$ (with caching)
- Hierarchy construction: $O(k \cdot 2^{d_{max}})$ for $k$ selected concepts

### Memory optimization:
- Lazy hierarchy construction
- Caching frequently accessed deep concepts
- Pruning inactive secondary spaces

### Proposed hyperparameters:
- Primary dimension: $n = 768$
- Secondary dimension: $m = 64$
- Max secondary spaces: $k = 3$
- Max hierarchy depth: $d_{max} = 3$
- Distance/relevance balance: $\beta = 0.5$
- Static/context balance: $\alpha = 0.3$
```

### --- Page 0013 ---

```markdown
## 7.4 Proposed Evaluation Framework

We propose the following evaluation protocols for future empirical validation:

### Datasets:
- Multi-turn dialogue with topic shifts
- Long-context QA requiring distant context
- Multi-topic synthesis tasks

### Baselines:
- Standard transformer
- Retrieval-augmented generation
- Flat hyperbolic embeddings
- Hierarchical Euclidean spaces

### Metrics:
- Coherence (human evaluation)
- Context recall accuracy
- Integration quality
- Computational cost (inference time, memory)
- Epilexity correlation (predicted vs. measured access cost)

Expected advantages: The architecture is designed to provide improvements in multi-topic integration and context recall while maintaining reasonable computational overhead. Empirical validation of these expectations remains crucial future work.

## 8 Discussion

### 8.1 Theoretical Implications

Our dual hyperbolic space architecture with nested hierarchies provides a geometric framework unifying several previously disparate concepts in AI and cognitive science.

**Geometry of reasoning:** By interpreting reasoning as navigation through hyperbolic space, we provide a geometric substrate for understanding computational costs. Distance encodes not just semantic dissimilarity but the cognitive effort required to shift contexts—a perspective aligned with recent work on computational epistemology [6, 11].

**Information access under constraints:** Our nested hierarchies implement a natural stratification where frequently accessed information resides at shallow depths (low cost) while specialized knowledge can be stored deeply (high cost, high capacity). This mirrors fundamental principles in computer architecture (cache hierarchies) and human memory (working vs. long-term storage).

**Epilexity realization:** The access cost model provides a concrete architectural instantiation of epilexity [6]—measuring information complexity relative to a computationally bounded observer.
```

### --- Page 0014 ---

```markdown
## 8.2 Modeling Human Reasoning Patterns

Our architecture operationalizes hypotheses about human cognition: (1) humans maintain parallel reasoning contexts, and (2) conceptual knowledge exhibits nested hierarchical organization.

- **Parallel contexts:** The dual space design directly models the phenomenon of “inner dialogue”—maintaining a surface conversation while processing background thoughts. This is not merely attention reweighting but geometrically distinct organizations that can evolve independently.

- **Multi-perspective integration:** Human reasoning often benefits from viewing problems through multiple frameworks. Also, opposite opinions sometimes must be integrated, especially in the case of moral/ethical issues. Our architecture provides a geometric substrate for such multi-perspective thinking, where each perspective corresponds to a different space organization.

- **Cognitive plausibility:** The asymmetry in transition costs (entering a secondary space is expensive, returning to primary is cheap) mirrors human experience. We easily “snap back” to our main train of thought after a digression, but initiating the digression requires mental effort.

## 8.3 Local Opposition within Global Similarity

A further implication of ridge-based positioning concerns the geometric relationship between contexts as viewed from the secondary space. When we center a secondary space at ridge point $r$, the Möbius transformation that moves $r$ to the origin maps the primary center $c_p$ and predicted next center $c_{next}$ to positions that are approximately opposite directions from the new origin—in the case where $r$ is the geodesic midpoint, they are precisely $180^\circ$ apart.

This geometric configuration provides a mathematical framework for understanding a phenomenon extensively studied in social psychology: how effective cross-partisan dialogue often involves finding shared higher-level values (the ridge point) from which participants can then acknowledge their genuine disagreements without dismissing the other side. Haidt’s moral foundations theory similarly demonstrates how political opponents share abstract commitment to “morality” while differing sharply on which moral dimensions to prioritize—precisely the geometric structure our ridge-based architecture encodes.

This suggests that our nested hyperbolic architecture may capture fundamental patterns in human reasoning about contested concepts. The geometry naturally encodes how mediation requires occupying a position from which both sides appear opposed, even as they share common ground—a principle observed across domains from political discourse to scientific paradigm debates.

## 8.4 Practical Applications

Beyond theoretical insights, our architecture suggests practical applications:

- **Long-form dialogue systems:** Nested hierarchies enable storing conversation history at varying levels of detail, with recent exchanges at high resolution and older context summarized.

- **Multi-topic assistance:** Secondary spaces allow exploring digressions without losing the primary thread.

- **Collaborative AI systems:** Multiple AI systems could maintain different space organizations, with a meta-system integrating their perspectives.

- **Knowledge base organization:** The nested hyperbolic structure provides a natural schema for organizing large knowledge bases.
```

### --- Page 0015 ---

```markdown
8.5 Limitations

| **Limitation**                     | **Description**                                                                                                                                                                                                 |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Computational overhead              | Hyperbolic operations are more expensive than Euclidean. Nested hierarchies add complexity. Theoretical analysis suggests modest overhead, but this requires empirical validation.                             |
| Prediction accuracy                 | Ridge positioning requires predicting the next conversational center. Mis-prediction reduces effectiveness.                                                                                                     |
| Approximation nature                | Our nested hierarchies are finite-depth ($d_{max} = 3$) approximations of hypothesized self-similar structures. True mathematical fractals would require infinite recursions; our pragmatic choice balances theoretical motivation with computational feasibility. |
| Lack of empirical validation         | While we provide a comprehensive theoretical framework and implementation proposal, empirical validation is needed across diverse tasks, languages, and model scales. This represents the most significant limitation and the primary direction for future work. |
| Hyperparameter sensitivity          | Our approach introduces several hyperparameters. While we provide reasonable defaults, optimal values may be task-dependent.                                                                                   |

8.6 Future Work

| **Future Direction**               | **Description**                                                                                                                                                                                                 |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Empirical validation                | The most critical next step is implementing the proposed architecture and conducting experiments on the evaluation protocols outlined in Section 7.                                                            |
| Deeper hierarchies                 | Exploring $d_{max} > 3$ or adaptive maximum depth could move closer to truly self-similar organization.                                                                                                       |
| Learned transitions                 | Rather than heuristic ridge detection, learned models could predict when and where to spawn secondary spaces.                                                                                                  |
| Multi-modal extensions              | Applying nested hyperbolic geometries to vision-language models.                                                                                                                                              |
| Theoretical analysis                | Proving convergence properties, establishing optimality of ridge positioning, and deriving bounds on integration quality.                                                                                     |

9 Conclusion

We have proposed a dual hyperbolic space architecture for context-aware reasoning in large language models, featuring nested hierarchical structures and ridge-based secondary space positioning. This framework addresses fundamental challenges in managing multiple conversational contexts through principled geometric design.

Key contributions:

1. **Dual space architecture**: We introduced geometrically distinct primary and secondary hyperbolic spaces that enable parallel reasoning contexts. Unlike attention-based approaches operating within a single representational space, our architecture maintains truly independent organizations—each with its own center, distance metric, and hierarchical structure.

2. **Ridge-based positioning**: Our strategy for centering secondary spaces at ridge points—geometric saddles between contexts—minimizes transition costs while maximizing bilateral visibility. The resulting 180° opposition between contexts as viewed from the ridge reveals deep structure in conceptual relationships, formalizing patterns observed in human discourse and conflict resolution.

3. **Nested hierarchies with adaptive depth**: We developed a framework for organizing concepts in self-similar nested hyperbolic spaces, where depth assignment integrates geometric distance with semantic relevance. This provides exponential storage capacity while implementing natural...
```

### --- Page 0016 ---

```markdown
information access costs. The approach operationalizes recent information-theoretic work on epiplexity.

4. Comprehensive formalization: We provided mathematical foundations spanning hyperbolic geometry, energy landscape interpretation, access cost modeling, and transition dynamics, along with detailed implementation proposals.

Broader implications: Beyond technical contributions, our framework suggests new perspectives on the geometry of thought. The hyperbolic structure—where distance encodes cognitive effort rather than mere dissimilarity—provides a mathematical language for reasoning about bounded intelligence. The nested organization mirrors universal patterns in information systems. And the emergence of local opposition from ridge-based viewing offers geometric insight into phenomena ranging from scientific paradigm conflicts to political polarization.

Looking forward: While we have focused on language models, the geometric principles may extend to multi-modal systems, collaborative AI architectures, and human-AI interaction design. The fundamental trade-off between storage capacity and access cost appears universal; our nested hyperbolic framework provides one principled way to navigate it.

The architecture we propose is not a claim that semantic spaces are hyperbolic or exhibit fractal structure—these remain hypotheses requiring deeper investigation. Rather, we demonstrate that assuming such structures leads to architectures with desirable theoretical properties. Whether these geometric principles reflect deep truths about cognition or merely provide useful engineering frameworks, they offer a productive lens for designing more capable and cognitively plausible AI systems.

Empirical validation remains the most critical next step. We have provided detailed implementation proposals and evaluations to facilitate this work. We hope this theoretical foundation will inspire both practical implementations and further theoretical development of geometric approaches to AI reasoning.

## Acknowledgments

This work was motivated by the hypothesis that human semantic spaces exhibit self-similar hierarchical organization, and constructed through conversations between the author and AIs. Making theoretical insights was supported from Claude and ChatGPT, and writing main text and formulas was helped from Claude. We thank the anonymous reviewers for their valuable feedback.

## References

[1] Y. Bengio, A. Courville, and P. Vincent. Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8):1798–1828, 2013.

[2] D. Brockman and J. Kalla. Durably reducing transphobia: A field experiment on door-to-door canvassing. Science, 352(6282):220–224, 2016.

[3] I. Chami, Z. Ying, C. Ré, and J. Leskovec. Hyperbolic graph convolutional neural networks. NeurIPS, 2019.

[4] R. Child, S. Gray, A. Radford, and I. Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
```

### --- Page 0017 ---

```markdown
| Reference Number | Citation                                                                                          |
|------------------|---------------------------------------------------------------------------------------------------|
| [5]              | Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. Le, and R. Salakhutdinov. Transformer-XL: Attention-based language models beyond a fixed-length context. ACL, 2019. |
| [6]              | M. Finzi, A. Wilson, and others. Epiplexity: Measuring information complexity for bounded observers. arXiv preprint arXiv:2601.03220, 2026. |
| [7]              | A. Graves, G. Wayne, and I. Danihelka. Neural Turing machines. arXiv preprint arXiv:1410.5401, 2014. |
| [8]              | C. Gulcehre, M. Denil, M. Malinowski, and others. Hyperbolic attention networks. ICLR, 2019.      |
| [9]              | J. Haidt. The Righteous Mind: Why Good People Are Divided by Politics and Religion. Pantheon Books, 2012. |
| [10]             | P. Lewis, E. Perez, A. Piktus, and others. Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS, 2020. |
| [11]             | M. Li and P. Vitányi. An Introduction to Kolmogorov Complexity and Its Applications. Springer, 3rd edition, 2008. |
| [12]             | J. Lindsey, A. Ocko, S. Ganguli, and S. Deny. A unified theory of early visual representations from retina to cortex through anatomically constrained deep CNNs. ICLR, 2019. |
| [13]             | J. Mu, X. Li, and N. Goodman. Learning to compress prompts with gist tokens. arXiv preprint arXiv:2304.08467, 2023. |
| [14]             | M. Nickel and D. Kiela. Poincaré embeddings for learning hierarchical representations. NeurIPS, 2017. |
| [15]             | M. Nickel and D. Kiela. Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. ICML, 2018. |
| [16]             | S. Sukhbaatar, A. Szlam, J. Weston, and R. Fergus. End-to-end memory networks. NeurIPS, 2015.    |
| [17]             | X. Zhang, F. Wei, and M. Zhou. HIBERT: Document level pre-training of hierarchical bidirectional transformers for document summarization. ACL, 2019. |
```



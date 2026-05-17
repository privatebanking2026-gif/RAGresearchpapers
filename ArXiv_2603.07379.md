# ArXiv 2603.07379

### --- Page 0001 ---

```markdown
# SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions

**Saroj Mishra**\(^1\), **Suman Niroula**\(^2\), **Umesh Yadav**\(^3\), **Dilip Thakur**\(^4\), **Srijan Gyawali**\(^5\), **Shiva Gaire**\(^6\)

\(^1\)University of North Dakota: saroj.mishra773@gmail.com  
\(^2\)Youngstown State University: sum.nir1@gmail.com  
\(^3\)University of Toledo: yadav.umesh0518@gmail.com  
\(^4\)University of Missouri: dilepthakur7@gmail.com  
\(^5\)Tribhuwan University: gyawalisrijan01@gmail.com, mail@shivagaire.com.np  

---

**Abstract**—Retrieval-Augmented Generation (RAG) systems are increasingly evolving into agentic architectures where large language models autonomously coordinate multi-step reasoning, dynamic memory management, and iterative retrieval strategies. Despite rapid industrial adoption, current research lacks a systematic understanding of Agentic RAG as a sequential decision-making system, leading to highly fragmented architectures, inconsistent evaluation methodologies, and unresolved reliability risks. This Systematization of Knowledge (SoK) paper provides the first unified framework for understanding these autonomous systems. We formalize agentic retrieval-generation processes, explicitly modeling their control policies and state transitions. Building upon this formalization, we develop a comprehensive taxonomy and modular decomposition that categorizes systems by their planning mechanisms, retrieval orchestration, memory paradigms, and tool-invocation behaviors. We further analyze the critical limitations of traditional static evaluation practices and identify severe systemic risks inherent to autonomous loops, including compounding hallucination propagation, memory poisoning, retrieval mismanagement, and cascading-tool-execution vulnerabilities. Finally, we outline key declarative research directions spanning adaptive retrieval, cost-aware orchestration, formal trajectory evaluation, and oversight mechanisms, striving a definitive framework for building reliable, controllable, and scalable agentic retrieval systems.

**Index Terms**—Agentic RAG, Retrieval-Augmented Generation, Sequential Decision Processes, Tool Invocation, Multi-Step Reasoning, System Architecture, Evaluation Frameworks, AI Safety.

---

## I. INTRODUCTION

Retrieval-Augmented Generation (RAG) fundamentally couples a parameter generator with a non-parametric corpus to condition outputs on retrieval evidence [1]. However, the standard formulation relies on a static control flow: a retriever fetches a fixed set of passages, and the generator synthesizes an answer without adaptive multi-step decisions [2]. This deterministic pipeline exhibits severe brittleness in knowledge-intensive and multi-hop tasks [3]. Because retrieval occurs blindly before reasoning begins, static systems suffer from context overloading [4], lack native correction loops for noisy retrievals [5], and indiscriminately retrieve regardless of input necessity, which can actively diminish response quality [6].

To mitigate these limitations, early heuristic approaches introduced active and iterative retrieval paradigms [7]. Frameworks like unified active-retrieval (UAR) treat the retrieval trigger as a dynamic decision [8], while generation-in-the-loop architectures iterate intermediate reasoning to refine subsequent queries [9]. Concurrently, the emergence of tool-augmented large language models (LLMs) established the foundation for fully autonomous control [10], [11]. Models such as ReAct (Reasoning and Acting) demonstrated that LLMs can act as reasoning agents emitting interleaved thoughts and actions [12]. Furthermore, paradigms incorporating episodic memory [13], tree-based exploration [14], and interactive search [15] proved that agents can optimize trajectories based on environmental observations.

As illustrated in Figure 1, the convergence of dynamic retrieval policies with autonomous planning loops has crystallized into a new paradigm: Agentic RAG [16]. In this architecture, retrieval is no longer a preprocessing step, but an explicitly managed tool in multi-step, policy-driven reasoning trajectory [17]. The LLM orchestrates the entire process, deciding which actions to perform, whether to iterate, and how to adaptively search at multiple granularities [18]. This requires a fundamental shift from fixed retrieve-then-read workflows to modular, pattern-based control strategies [19]. This paper positions itself as a Systematization of Knowledge (SoK). Currently, the rapid proliferation of Agentic RAG systems has led to severe field fragmentation, a lack of a unified taxonomy, and an absence of standardized evaluation frameworks. To address these systemic gaps, the main contributions of this work are summarized as follows:

1. We provide a formal conceptualization of agentic retrieval-augmented generation by framing it as a sequential decision-making process that integrates reasoning, retrieval, memory, and tool interaction.
2. We introduce a multi-dimensional taxonomy that organizes the design space of agentic RAG systems.
```

### --- Page 0002 ---

```markdown
![High-level progression from single-pass retrieval-augmented generation to iterative retrieval and Agentic RAG.](assets/page_0002_img_1.png)

## I. Introduction

This section establishes the motivation for formalizing Agentic RAG as a distinct paradigm beyond static retrieval-augmented generation. We clarified the conceptual gap between traditional RAG pipelines and autonomous, multi-step reasoning architectures that dynamically plan, retrieve, and adapt. By framing the need for structured taxonomy, evaluation reform, and formal modeling, we positioned Agentic RAG as a systems problem rather than a pioneering extension. The next section grounds this discussion in the foundational evolution of large language models and retrieval systems, setting the theoretical and historical context necessary for formal definition.

## II. BACKGROUND AND FOUNDATIONS

This section establishes the conceptual building blocks that underpin Agentic RAG systems. It reviews large language models, classic retrieval-augmented generation, tool-augmented paradigms, planning, and memory architectures. The goal is to provide evidence-driven grounding for the formalization, taxonomy, and architectural discussions that follow.

### A. Large Language Models

Modern large language models (LLMs) rely on the Transformer architecture to learn contextual representations from massive corpora [20], [21]. While highly capable text generators, their ability to perform autonomous reasoning stems primarily from in-context learning: the capacity to adapt to novel tasks via prompt conditioning without parameter updates [22]. Techniques like chain-of-thought prompting extend this by eliciting intermediate reasoning steps, allowing models to decompose problems and follow multi-step procedures [23]. These zero-shot planning capabilities serve as the foundational engine for agentic control.

However, LLMs exhibit fundamental limitations that necessitate external augmentation. Their parametric knowledge is frozen at training time [1], making them prone to hallucinating facts for novel or niche queries [24]. Furthermore, simply expanding the context window to inject more information is insufficient; models frequently ignore relevant data placed in the middle of long inputs, a vulnerability known as the “lost in the middle” effect [4]. Overcoming these constraints requires active external invocation and dynamic retrieval rather than passive text generation.

### B. Retrieval-Augmented Generation

To address the knowledge deficit of frozen LLMs, Retrieval-Augmented Generation (RAG) couples a parametric generator with a non-parametric retrieval index [1]. Classic RAG utilizes dense retrieval models (e.g., DPR) to map queries and documents into a shared embedding space, fetching the top-$k$ most relevant passages for the generator to condition upon [2]. Evidence from multiple retrieved documents efficiently while maintaining tractable compute [25].

Despite these advances, standard RAG architectures rely on a strictly static control flow: retrieve once, then generate. This deterministic pipeline is fundamentally limited. Retrieval quality depends entirely on the initial, often disreputable user query, with no mechanisms to refine the search based on intermediate generation states [7]. Because the retrieved context is fixed upfront, the system cannot autonomously self-correct if the initial evidence is noisy or incomplete [5]. These structural rigidities directly motivate the shift toward iterative, policy-driven retrieval frameworks.

### C. Tool-Augmented and Agentic LLMs

A parallel research trajectory reframed LLMs from static text generators to interactive agents capable of taking actions in external environments. ReAct (Reasoning and Acting) introduced a prompting paradigm that interleaves explicit reasoning traces with actions (e.g., search queries, API calls), enabling the model to gather information iteratively and adjust its trajectory based on observations [12]. Toolformer addressed a complementary challenge: teaching models to autonomously decide which tools to invoke, when to invoke them, and how to incorporate results [10]. MRKL Systems proposed a modular neuro-symbolic architecture in which an LLM serves as a router that delegates to specialized external modules, emphasizing extensibility beyond pure parametric capabilities [11].
```

### --- Page 0003 ---

```markdown
The concept of agentic LLMs further crystallized through work on self-improvement and reflective control. Reflexion introduced verbal reinforcement learning, where an agent stores textual reflections on its past failures in an episodic memory buffer and uses them to improve subsequent attempts [13]. A comprehensive survey by Wang et al. formalized the LLM-based augmentation as a system comprising profiling, memory, planning, and action modules [26]. These developments established the agent design patterns—planning, tool use, and reflection—that Agentic RAG systems embed directly into the retrieval pipeline.

## D. Multi-Hop Reasoning and Planning

Many knowledge-intensive tasks require reasoning across multiple pieces of evidence that cannot be retrieved in a single step. HotpotQA formalized this requirement by introducing a multi-hop question answering benchmark where systems must reason over multiple supporting documents to derive an answer [3]. Standard retrieval approaches struggle with such tasks because the information needed for later reasoning steps depends on intermediate deductions, creating a dependency that single-pass retrieval cannot resolve [9].

Query decomposition addresses this challenge by breaking a complex query into simpler sub-questions. Least-to-most prompting solves decomposed problems sequentially [27], while Plan-and-Solve prompting generates an explicit plan before execution [28]. Self-Axis extends this paradigm by integrating retrieval with explicit follow-up queries and routing them to a search engine [29].

Interleaved retrieval-reasoning approaches take this further by tightly coupling retrieval with ongoing chain-of-thought generation. IRCot interleaves reasoning steps with retrieval calls, using the evolving trace to guide what to retrieve next. Tree-of-Thought generalizes this toward explicit tree-structured exploration with search and self-evaluation [30]. These methods establish the reasoning foundations upon which agentic retrieval systems build their planning mechanisms.

## E. Memory-Augmented Systems

Effective multi-step reasoning requires maintaining and updating state across interactions. Short-term memory in agentic systems typically corresponds to the evolving context window: the accumulation of observations, actions, and intermediate outputs. However, as contexts grow long, models exhibit degraded utilization of information, motivating strategies for dynamic context pruning and selective attention [4].

Long-term memory systems enable agents to retain and recall information across tasks or sessions. Retrieval-based memory stores past experiences as embeddings in a vector store and retrieves relevant entries at inference time, functioning analogously to RAG but over the agent’s own history [31]. Episodic memory captures structured records of past interaction trajectories, including actions taken and outcomes achieved [13].

Recent work proposes unified architectures that dynamically manage both short-term working memory and long-term persistent storage, allowing agents to selectively consolidate, retrieve, and forget information based on task demands [32]. These persistent memory mechanisms act as a necessary prerequisite for the state-tracking capabilities that distinguish Agentic RAG from static pipelines.

The progression from static generation to retrieval-augmented systems reveals the architectural primitives that make autonomous reasoning possible. However, the literature lacks a precise formal boundary distinguishing iterative retrieval from true agentic behavior. The following section formalizes Agentic RAG using necessary and sufficient conditions and frames it within a sequential decision-making model to resolve this ambiguity.

## III. FROM STATIC RAG TO AGENTIC RAG

The transition from static Retrieval-Augmented Generation (RAG) to agentic RAG represents a fundamental paradigm shift in how large language models (LLMs) interact with external knowledge. While traditional RAG operates strictly as a linear pipeline—fetching documents based on an initial query and passing them to a generator—it lacks the capacity for autonomous correction, multi-step reasoning, and dynamic context formulation. This section traces the evolutionary path from static pipelines to planning-driven retrieval systems. We formally define Agentic RAG, explicitly mathematically map its state transition and control policies, and demarcate the boundary between single-pass retrieval and iterative workflows.

### A. Limitations of Standard RAG Pipelines

Standard RAG architectures [1] decompose knowledge retrieval from text generation through a deterministic, sequential mechanism. Given a query q and a knowledge corpus C, a retriever fetches a top-k set of documents D, and the static, one-shot retrieval paradigm suffers from three critical systemic limitations:

1. First, it is highly susceptible to retrieval irrelevance and context overloading. If the initial embedding maps the query to suboptimal documents, the generator is forced to condition its output on irrelevant noise. As demonstrated by Liu et al. [33], LLMs suffer from a “lost in the middle” phenomenon, where the inclusion of excessive, low-signal retrieved context degrades reasoning quality.

2. Second, static pipelines possess no adaptive reasoning or correction loops. If a complex query requires synthesizing information across disparate documents that do not share semantic similarity in the vector space, a single-pass retrieval will fail to fetch the requisite connective context [34], [35].

3. Third, this architecture is prone to error propagation. Because the retrieval phase is strictly isolated from the generation phase, the LLM cannot pause generation to request missing information, resulting in hallucinations when the retrieved context is insufficient [36].
```

### --- Page 0004 ---

```markdown
## B. Need for Iterative Retrieval

To address the brittleness of one-shot retrieval, the field moved toward iterative retrieval mechanisms. Complex user intents, particularly in domains requiring multi-hop reasoning (e.g., answering compositional questions over datasets like HotPotQA \cite{3} or MuSiQ \cite{17}), rarely map to a single contiguous text chunk.

Iterative retrieval allows the system to execute sequential queries against the database, where subsequent queries are conditioned on the information retrieved in prior steps \cite{38}. This necessity arises from the problem of query reformulation. A user’s initial prompt is often underspecified. Iterative systems employ the LLM to retrieve or expand the query based on partial information, progressively building a high-fidelity context window. However, early iterative retrieval models relied on heuristic triggers (e.g., retrieving every n tokens) rather than semantic understanding of when external knowledge was actually required.

## C. Emergence of Planning-Driven Retrieval

The limitations of heuristic-based iterative retrieval precipitated the integration of planning models, leading to planning-driven retrieval. Inspired by the ReAct (Reasoning and Acting) framework \cite{12}, architectures began coupling the retriever with an LLM planner.

Concurrently, paradigms like Toolformer \cite{10} established that LLMs could be trained to autonomously invoke external tools. This has enabled models to navigate text interfaces and execute search queries to gather evidence for reformulating an answer. The emergence of open-source autonomous agent frameworks (e.g., AutoGPT \cite{39}) further normalized the concept of granting LLMs continuous execution privileges.

In this evolved paradigm, the LLM does not merely consume retrieved text; it actively decides when to invoke the retriever as an external tool, what specific query to pass to it, and how to evaluate the returned context against the overarching goal. This orchestration of retrieval through autonomous planning loops serves as the foundational architecture for Agentic RAG. The conceptual progression from deterministic, single-pass pipelines to this policy-driven framework is illustrated in Figure 2.

## D. Formal Definition of Agentic RAG

Agentic RAG is not defined by the presence of a retriever, but by the presence of an autonomous control policy that governs retrieval and serves as a discrete action space.

1) **System-Level Formalization**: We model Agentic RAG as a finite-horizon Partially Observable Markov Decision Process (POMDP), where the external knowledge corpus $\mathcal{C}$ constitutes a latent, partially observable information source. We formally define the system as the tuple:

$$
\mathcal{S}_{\text{RAG}} = (\mathcal{S}_{env}, \mathcal{A}, \Omega, \tau_{\pi}, M, T)
$$

where:

- $\mathcal{S}_{env}$ is the latent true state of the required knowledge residing in $\mathcal{C}$.
- $\mathcal{A}$ is the discrete action space consisting of retrieval, reasoning tool use, and information: $\mathcal{A} = \mathcal{A}_{\text{ret}} \cup \mathcal{A}_{\text{tool}} \cup \mathcal{A}_{\text{stop}}$.
- $\Omega$ is the observation space (e.g., text chunks returned by a retriever or outputs from a tool).
- $o(\mathcal{O} | s_t, a_t)$ is the observation function that returns an observation $o \in \Omega$ conditioned on the hidden state $s_t$ and the action $a_t$ taken.
- $\tau_{\pi}(M_t | M_{t-1})$ is a stochastic control policy parameterized by the LLM (implemented via prompting or fine-tuning), conditioned on the observable history.
- $M_t$ is the dynamic working memory (or observable history buffer) at step $t$. The working memory $M_t$ serves as a tractable approximation of the belief state $b_t$.
- $T(s_{t+1} | s_t, a_t)$ is the latent state transition function. In this formulation, the state $s_t$ represents the evolving task context, including the user query, intermediate reasoning traces, retrieved documents, and relevant memory elements accumulated during interaction. The action $a_t$ corresponds to decisions such as issuing a retrieval query, invoking an external tool, updating memory, or generating response tokens. The policy $\tau_{\pi}(M_t)$ defines the agent’s strategy for selecting actions conditioned on the current context. The environment comprises external knowledge sources, retrieval systems, and tool interfaces with which the agent interacts during task execution.

At any discrete time step $t \in [0, T_{\max}]$ (where $T_{\max}$ is the finite horizon limit), the system maintains a memory $M_t$ seeded with the initial user query $q_t$. The stochastic policy $\tau_{\pi}$ samples the next action $\alpha_t \in \mathcal{A}(M_t)$.

If the policy selects a retrieval action $\alpha_t = \text{Retrieve}(q_t)$, the observation function returns the latent corpus and deterministically updates the memory with the observation $o_t$ such that $M_{t+1} = M_t \cup \{o_t, a_t\}$. If the policy dictates a reasoning step $a_t = \text{Reason}(o_t)$, the intermediate conclusion $c_t$ is appended as $M_{t+1} = M_t \cup \{c_t\}$. The process iterates strictly within the finite horizon $T_{\max}$ until it outputs the STOP action, triggering the final generation $y = G(M_t)$.

In practice, maintaining an exact Bayesian belief state over the environment is infeasible for large-scale language agents. Instead, most implementations approximate the belief state through structured memory representations $M_t$. These representations may include intermediate reasoning traces, retrieved document sets, tool outputs, and summarized contextual knowledge accumulated across reasoning steps. Belief updates therefore correspond to memory update operations such as selective retrieval augmentation, summarization, pruning of redundant information, or learned memory corrections that retain high-utility signals while discarding low-relevance context. Such approximations enable tractable reasoning while preserving relevant task information across multi-step interactions.

2) **Necessary Properties**: Based on the POMDP formalization above, an Agentic RAG system must exhibit the following:
```

### --- Page 0005 ---

```markdown
![The architectural evolution from static one-shot RAG pipelines to the Agentic RAG POMDP formulation. The Agentic framework replaces linear generation with a cyclic control policy $\pi_\theta$ managing a persistent memory state $M_t$.](assets/page_0005_img_1.png)

1) **Iterative Control:** The system must possess a feedback loop governed by a stochastic policy $\pi_0$, allowing for multiple transitions before final generation.  
2) **Dynamic Retrieval:** Retrieval queries $q_t$ must be conditionally generated at runtime based on the evolving memory state $M_t$.  
3) **Tool-Mediated Interaction:** The retriever must be modeled as an explicit function call within the action space $A$, subject to validation via the observation function.  
4) **State Persistence:** The system must maintain an episodic working memory $M_t$, that persists across the control loop to approximate the fully observable state.  

While these four properties are analytically necessary to classify a system as Agentic RAG, they are not sufficient to guarantee stability or safety. An architecture may possess the correct POMDP loops but still fail due to an unaligned policy or corrupted memory—a limitation that necessitates the rigorous evaluation and safety frameworks discussed in subsequent sections. Ultimately, Agentic RAG constitutes a partially observable sequential decision process under adaptive retrieval policies.  

3) **Distinguishing Active RAG vs Agentic RAG:** A common source of ambiguity in the literature is the conflation of “Active RAG” (e.g., FLARE [7]) and Agentic RAG. Active RAG dynamically decides when to retrieve during the token generation process, often using probability confidence thresholds to trigger a database lookup. However, Active RAG is fundamentally a single-pass generative process that uses retrieval to fill localized knowledge gaps.  

In contrast, Agentic RAG separates planning from generation. It is policy-driven, executes multi-step tool use, and can perform operations that do not directly result in output tokens (e.g., self-correction, discarding retrieved content, or switching tools). A summary of these architectural distinctions is provided in Table II.
```

### --- Page 0006 ---

```markdown
# PAGE_NAME: page_0006

## TABLE I
### MAPPING AGENTIC SYSTEM PROPERTIES TO POMDP FORMALIZATION

| Agentic Property | POMDP Component | Operational Interpretation |
|------------------|------------------|----------------------------|
| Iterative Control | Stochastic Policy $\pi$ | Non-deterministic action selection governing the loop |
| Dynamic Retrieval | Action Space $A$ | Query generation treated as a discrete runtime action |
| Tool Mediation | Observation Model $O$ | External API interaction returning state context |
| State Persistence | Belief State $b_t \approx M_t$ | Memory acts as a tractable approximation of hidden state |
| Termination | Finite Horizon $T_{max}$ | Constrained loop depth to prevent infinite execution |

## TABLE II
### ARCHITECTURAL DISTINCTIONS BETWEEN ACTIVE RAG AND AGENTIC RAG

| Feature               | Active RAG                          | Agentic RAG                          |
|----------------------|-------------------------------------|--------------------------------------|
| Trigger Mechanism     | Log-probability thresholds or token heuristics | Policy-driven reasoning and explicit tool-calling |
| Control Flow          | Single-pass, forward-generating     | Iterative, multi-step planning loops  |
| Planning Explicitness  | Planning and generation             | Explicit delete/read/write capabilities |
| Context Management     | Append-only (accumulated context)  | Read/Write/Prune working memory      |
| Failure Handling       | Cannot self-correct prior token generation | Can discard poor retrieval and query explicitly |

## E. Problem Formulation of Agentic RAG Systems

Given the POMDP representation, the engineering of an Agentic RAG system can be formulated as a constrained sequential decision-making problem. The objective is to optimize the stochastic policy $\pi$ to maximize the fidelity of the final output $y$ relative to an ideal response $y^*$, while strictly minimizing the computational overhead of the iterative loop.

We define an objective function over a trajectory $\tau = (M_0, \ldots, M_1, \ldots, M_T)$ generated by policy $\pi$. Let $R_{task}(y, y^*)$ be the terminal reward function measuring response quality. Let $C(a_t)$ represent the step-wise cost function, which models latency, token consumption, and API limits. The problem formulation of an Agentic RAG system is:

$$
\max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ R_{task}(y, y^*) - \sum_{t=0}^{T} C(a_t) \right] \tag{2}
$$

where $\lambda$ is a regularization parameter controlling the trade-off between reasoning depth and operational efficiency.

This section established the theoretical backbone of Agentic RAG by formalizing its state transitions and defining the necessary properties of iterative control, dynamic retrieval, and memory persistence. We demonstrated that moving beyond static and active RAG pipelines fundamentally transforms the architecture into a budget-constrained sequential decision-making problem. Having clarified this structural foundation, Section IV systematizes the field by classifying existing Agentic RAG frameworks across these operational dimensions.

## IV. TAXONOMY OF AGENTIC RAG SYSTEMS

Retrieval-Augmented Generation (RAG) couples a Retriever with a Generator—typically a large language model (LLM)—to ground model outputs in external knowledge rather than relying solely on parametric knowledge [1], [40], [41]. Agentic RAG extends this paradigm by introducing an explicit Planner that governs Tool Invocation (including retrieval) under a Control Policy, thereby enabling Iterative Retrieval, Dynamic Context Construction, and Multi-step Reasoning beyond a single retrieve-then-generate pass [12], [42]–[44].

This section provides an attribute-based taxonomy: we classify Agentic RAG systems by orthogonal axes that describe what kind of system they are, and how to implement them. Section V instantiates these classes into concrete architectures, while Section VI abstracts recurring solutions as design patterns.

To provide a rigorous classification of the Agentic RAG landscape, we propose a taxonomy organized across four dimensions: Planning, Memory, Tool Orchestration, and Retrieval Strategy. As illustrated in Figure 3, these dimensions are designed to be Mutually Exclusive and Collectively Exhaustive (MECE) and represent varying degrees of complexity within each dimension, but every Agentic RAG architecture may inherently make a design choice across these four axes. Table III synthesizes this classification, mapping common archetypes to their core taxonomy attributes.

## A. Architectural Taxonomy

Architectural taxonomy in Agentic RAG classifies systems by agent topology—i.e., how many distinct decision-making entities exist, where the Planner function is located, and whether roles such as retrieval and generation are centrally controlled or distributed. This axis is intentionally orthogonal to retrieval strategy: a single-agent system may still perform iterative retrieval, and a multi-agent system may still perform one-shot retrieval if its control policy is static [40], [41]. Modern SDKs and frameworks expose topology and tool loops explicitly [45], [46], enabling the same application class to be realized under different topologies [43], [44], [47].

1) **Single-Agent RAG**: Single-Agent RAG denotes systems where one agent jointly performs planning and generation, invoking retrieval and other tools under a single control policy. Classical RAG formulations always combine a retriever and generator, but they need not be agnostic if retrieval is purely pre-specified; the agentic variant emerges when the planner role adapts actions [1], [7], [40]. Single-agent loops are directly supported in major frameworks [45]–[47], while other orchestrators provide lightweight agent abstractions suitable for retrieval-centric tool use [48], [49].
```

### --- Page 0007 ---

```markdown
## B. Retrieval Strategy Taxonomy

Retrieval strategy taxonomy captures when and how the Retriever is invoked across a trajectory, and how retrieved evidence is incorporated into dynamic context construction. Agentic systems increasingly treat retrieval as a repeated, state-dependent action rather than an upfront preprocessing step [7], [40], [41].

### 1) One-Shot Retrieval: 
One-Shot Retrieval refers to a single retrieval action conditioned on a fixed retrieved context, matching baseline RAG [1], [40]. Within Agentic RAG, this remains a class where no state-dependent retrieval actions occur after initiation, regardless of whether a Planner exists [45], [46].

### 2) Iterative Retrieval: 
Iterative Retrieval performs multiple retrieval actions during a single query resolution, where later retrievals depend on intermediate state. IRCot interleaves retrieval with Chain-of-Thought steps [9]. IterGen retrieves retrieval and generation with intermediate generators informing retrieval [5]. This class increases the degrees of freedom of the control policy and tightly couples retrieval with token economics [42].

### 3) Self-Refining Retrieval: 
Self-Refining Retrieval couples retrieval with critique, revision, or self-evaluation such that queries and evidence are refined to increase faithfulness [6], [57]. Self-RAG learns to retrieve on-demand and critique both retrieved passages and generations [6]. Such systems employ hybrid or learned control policies to drive active knowledge assimilation from retrieved evidence [58], [59].

### C. Reasoning Taxonomy

Reasoning taxonomy classifies the form of multi-step reasoning used to decide tool invocation and transform evidence into grounded outputs. We adopt four classes: Chain-of-Thought, ReAct-style interleaving, reflection-based reasoning, and tree-based exploration [12], [13], [23], [30].

#### 1) Chain-of-Thought & ReAct-Style: 
Chain-of-Thought (CoT) prompting elicits a sequential reasoning trace of intermediate steps [23], frequently acting as a query-construction substrate in IRCot and planning decompositions [9], [28]. ReAct extends this by interleaving reasoning steps with actions (tool invocations), producing observations that update subsequent reasoning [12]. Many agent frameworks describe agents as running tools in a loop until a stop condition, corresponding closely to the ReAct taxonomy class [45], [46].

#### 2) Reflection & Tree-Based Exploration: 
Reflection-based reasoning introduces explicit self-evaluation steps that critique intermediate reasoning, retrieved evidence, or generated assertions. Reflection stores this feedback in an episodic memory buffer to improve more tailored retrieval, while RARR retrieves evidence specifically to attribute and revise generated text [57]. Conversely, Tree-based exploration treats reasoning as a search over multiple candidate branches. Tree-of-Thoughts realizes this by proposing, evaluating, and expanding thoughts with backtracking [30], supporting evidence gathering for competing hypotheses.

![Taxonomy of Agentic RAG systems architecture, retrieval strategy, reasoning paradigm, and memory/context management. This structural mapping demonstrates how orthogonal control-flow decisions combine to form distinct, reproducible agentic archetypes.](assets/page_0007_img_1.png)
```

### --- Page 0008 ---

```markdown
## D. Memory and Context Paradigms

Agentic RAG must manage memory that persists across episodes and the active context given to the Generator at each step. Long-context models do not remove the need for structured context selection, as performance often degrades depending on the position of relevant information within long inputs [4]. Consequently, Dynamic Context Pruning has emerged to remove or compress retrieved content before generation. Methods like FILCO [60] and Provence [61] learn to filter retrieved contexts, reducing overhead and mitigating irrelevant evidence—a capability that becomes increasingly critical under iterative and multi-agent settings [7].

Beyond active context window management, architectures require Episodic Memory to store temporally bounded trajectories of agent behavior and feedback. For instance, RefleXion stores reflective feedback in an episodic buffer [13], while Generative Agents utilize a memory stream to support iterative planning [62]. This episodic logging acts as a localized attention mechanism, preserving reasoning fidelity while managing API costs across distinct task steps.

To maintain coherence across multiple independent sessions, systems also deploy Persistent Long-Horizon Memory. This paradigm retains information across sessions by persisting latent states into vector databases. Frameworks like Memory Bank [63] and MemGPT [64] explicitly target storing, recalling, and updating long-term interaction memories. These systems define memory-refining strategies—dictating how memory is updated, constructed, or decayed over time—shifting the architecture from a stateless functional flow to a stateful, continuous entity [45], [59].

## E. Cross-Dimensional Trade-Off Analysis

Taxonomy dimensions interact in practice; choices along one dimension induce constraints along others. These trade-offs are surfaced in both academic work on iterative retrieval and industrial documentation on tool chaining and orchestration [17], [42], [44], [65].

1) **Retrieval Depth vs. Cost:** Deeper retrieval (iterative/self-refining) improves coverage for multi-hop and long-form tasks [5], [7], [9] but increases cost via more tool calls, longer contexts, and extra generations. Pruning methods partially decouple depth from cost but risk removing necessary evidence [60], [61].

2) **Planning Complexity vs. Latency:** Planner-executor separation, explicit planning, and tree-based exploration reduce error propagation but impose latency due to extra planning and coordination [30], [51]. Tool calling is inherently multi-step and can stack latency when sequential [42]. Parallel or reduced round-trip tool use is highlighted as a mitigation in industrial guidance [65].

3) **Cost, Latency, and Token Economics:** Agentic RAG introduces token amplification: intermediate reasoning, tool queries, and critique steps expand generated tokens and multiple model invocations [42], [64]. Iterative retrieval paradigms often scale cost directly with the number of steps [5], [9].

Learned tool-use decisions motivate budget-aware orchestration as a core control-policy property [10], [59].

Table IV illustrates how representative agentic RAG systems can be categorized using the proposed taxonomy dimensions. This mapping demonstrates that the taxonomy captures diverse architectures spanning different planning strategies, retrieval mechanisms, memory paradigms, and tool coordination patterns.

This taxonomy categorizes Agentic RAG systems along structural and operational attributes, separating topology, memory strategies, and retrieval dynamics from implementation details. By organizing systems through architectural properties rather than surface tools, we establish a stable comparative framework. Having defined these structural categories, the next section decomposes these interarchitectural modules that operationalize these attributes in practice.

## V. CORE ARCHITECTURAL COMPONENTS

Building upon the taxonomy established in the preceding classification frameworks, it becomes necessary to transition from a theoretical categorization of Agentic Retrieval-Augmented Generation (Agentic RAG) systems toward a concrete systems-engineering perspective. Standard RAG architectures often rely on rigid, linear pipelines—typically defined by a monolithic sequence of query rewriting, document selection, and answer generation [16]. While these static joint optimizations, models maximize system performance for single-uniform workflow, rendering them incapable of decomposing complex, multi-hop queries that demand variable reasoning paths [16]. In contrast, Agentic RAG demands a decoupled yet highly orchestrated modular architecture capable of dynamic state management, iterative reasoning, and verifiable execution [18].

To realize theoretical autonomy, an Agentic RAG system must be structured as a network of interdependent but specialized modules [66]. A critical systems boundary must be maintained between three core roles: the planner breaks a complex query into a sub-task graph; the controller (Reasoning Engine) executes the immediate next step based on the local state; and the orchestrator manages the routing of inputs and outputs across distinct, specialized agents. This formal division of labor ensures that cognitive reasoning is explicitly separated from tool execution [66]. As illustrated in Figure 4, the modular interaction between these components enforces a closed feedback loop before any output is finalized. The specific inputs, outputs, and control signals governing these modules are synthesized in Table V.

### A. Planner Module

The Planner Module serves as the strategic orchestrator of the architecture [67]. Unlike traditional pipelines where retrieval is triggered by a single user query, the Planner is responsible for dynamically parsing high-dimensional intents, decomposing them into tractable sub-tasks, and formulating an iterative execution strategy [66]. This module addresses
```


### --- Page 0009 ---

```markdown
# TABLE III  
**CONSOLIDATED TAXONOMY MAPPING ARCHETYPES TO THEIR CORE AGENTIC RAG ATTRIBUTES.**

| Archetype                          | Topology      | Retrieval         | Reasoning               | Memory/Context                | Policy                       | Representative anchors                      |
|------------------------------------|---------------|--------------------|-------------------------|-------------------------------|------------------------------|---------------------------------------------|
| Baseline grounded generation        | Single-agent  | One-shot           | Minimal / linear        | Minimal; optional filtering    | Heuristic                    | RAG[1] surveys [40, 41]                     |
| Iterative evidence accumulation     | Single-agent  | Iterative          | CoT / ReAct            | Dynamic context construction   | Heuristic/Hybrid             | IR[2] CoT; Ire-ReGen [5]; LingChain agents [46] |
| Reflective refinement               | Single-agent  | Self-refining      | Reflection-based        | Episodic critique; pruning     | Hybrid/Learned               | Self-RAG [6]; RARR [57]; ReFlexion [13]    |
| Role-separated orchestration       | Planner-Executor | Iterative/self-refining | Planning + execution | Executor logging;              | Hybrid                       | HuggingGPT [50]; MRKL [11]; OpenAI Agents [45] |
| Distributed knowledge work          | Multi-agent   | Iterative/mixed    | ReAct/reflective       | Agent-local episodic; aggregation | Hybrid                     | AutoGen [52]; CrewAI [54]; LangGraph [53]; AID [44] |
| Memory-centric long-horizon         | Any           | Mixed              | Reflection common       | Persistent+episode; refresh    | Hybrid/Learned               | MemGPT [63]; MageMem [59];                  |

# TABLE IV  
**MAPPING REPRESENTATIVE AGENTIC RAG SYSTEMS TO THE PROPOSED TAXONOMY DIMENSIONS**

| System         | Planning Technology | Retrieval Engine | Reasoning Engine | Memory Model | API Coordination |
|----------------|---------------------|------------------|------------------|--------------|------------------|
| RAG            | RAG workflows       | Traditional retrieval | Traditional reasoning | Traditional memory | Traditional coordination |
| User Query Input |                     |                  |                  |              |                  |

![Core architectural components and control-flow relationships within a generalized Agentic RAG system. This demonstrates how the Reasoning Engine coordinates bidirectionally with Memory Systems and delegates execution to the Tool Orchestration Layer to maintain verifiable state control.](assets/page_0009_img_1.png)

## B. Retrieval Engine

In an Agentic RAG architecture, the Retrieval Engine ceases to operate as a passive document filter; instead, it functions as an active logic co-processor [68]. Standard embedding-based retrievers map queries into a latent vector space. However, fixed-dimensional embeddings are mathematically incapable of representing the full expressive spectrum of complex Boolean logic due to the linear separability limit [68]. To circumvent this bottleneck, the agentic Retrieval Engine integrates diverse indexing structures—including dense vector...
```

### --- Page 0010 ---

```markdown
| Module        | Inputs                     | Outputs                | Control Signals                     | Feedback Loops                        |
|---------------|----------------------------|------------------------|-------------------------------------|---------------------------------------|
| Planner       | User Query, Global State   | Sub-task / Graph       | Depth limits, Max steps             | Self-correction on plan failure       |
| Controller    | Sub-task, Local Context    | Action / Tool          | Confidence thresholds                | Observation-triggered replanning      |
| Orchestrator  | Multi-Agent Outputs        | Final Synthesis        | Agent-routing logic                  | Cross-agent consensus voting          |

### Architectural Decomposition of Agentic RAG Modules

search, sparse keyword matching, structured SQL databases, and formal knowledge graphs—orchestrated through programmable interfaces [69].

A defining implementation of this paradigm exposes hierarchical retrieval interfaces directly to the reasoning model [18]. Rather than concatenating a massive context window that degrades model attention, architectures equip the agent with granular tools: broad lexical matching, dense conceptual retrieval, and the targeted extraction of specific document segments [18]. This progressive information disclosure grants the agent autonomy to adjust its strategy dynamically. Empirical evaluations demonstrate that this interface design allows the agent to retrieve significantly fewer tokens than traditional static methods while achieving superior accuracy [18].

Furthermore, to balance precision and latency, production-grade engines employ multiphase ranking architectures. Running deep machine learning ranking models across an entire candidate set introduces unacceptable latency stacking [70]. Agent ranking eliminates this trade-off by applying a structured approach to the retrieval process, ensuring that the agent can focus on the most relevant top results [18]. Empirical evaluations further demonstrate that coupling optimized semantic chunking with these two-stage cross-core re-ranking pipelines significantly improves retrieval faithfulness and mitigates hallucination risks in high-stakes environments [71]. Industrial implementations also incorporate provenance-aware data fetching, executing dynamic queries against external system logs to ensure that retrieval is grounded in verifiably sensitive evidence rather than hallucinated artifacts [72].

### C. Reasoning Engine (The Controller)

The Reasoning Engine operates as the controller of the Agentic RAG system, responsible for interpreting retrieved contexts, updating the internal consensus state, and managing the step-by-step resolution of the generated plan. While the Planner dictates the overarching strategy, the Reasoning Engine controls the microscopic flow of state updates, determining how individual tool outputs are synthesized into actionable intelligence. This module navigates dynamic environments, handles tool invocation errors, and dynamically allocates deliberation time based on task complexity.

A primary architectural requirement is the establishment of a robust interface between the language model's cognitive space and the operational environment. In traditional workflows, models interact with verbose human-computer interfaces, which quickly overload the context window during long multi-turn dialogues, leading to attention degradation [73]. Modern architectures solve this by formalizing the Agent-Computer Interface (ACI). An effective ACI enforces structured interaction patterns based on simple atomic commands, informative state observation, and efficient error recovery mechanisms [73]. Instead of returning massive error traces, the ACI provides concise, syntax-checked feedback, preventing the agent from becoming trapped in infinite loops.

By operating through an ACI, the Reasoning Engine maintains strict execution control. It updates the system's working state by applying iterative edits, executing sandboxed code, and navigating repositories without losing context. Artifacts generated by these actions constitute a consensus memory. The Reasoning Engine constantly reads and modifies this structured task state, ensuring that distributed agents maintain a cohesive understanding of the problem space across protracted execution sessions.

### D. Memory Systems

Traditional RAG implementations treat memory dynamically, with each independent query. This assumption that memory is merely static storage leaves the agent without continuity of identity or historical awareness [74]. Agentic RAG redesigns this by separating memory into distinct subsystems: short-term working state, long-term persistent storage, and episodic memory [74]. Short-term memory acts as the immediate scratchpad, maintaining the evolving state session and conversational history. To prevent context exhaustion, this layer employs dynamic context pruning algorithms and strict state-checkpointing.

The most critical advancement is the formalization of Episodic Memory within Continuum Memory Architectures (CMA). CMA treats memory as a continuously evolving subsystem where memories persist, decay, and alter through retrieval-induced interference [74]. Episodic memory captures discrete trajectories of past problem-solving behaviors, allowing the agent to reflect on past experiences to inform future planning.

Advanced implementations grant the memory system intrinsic agency. Self-evolving memory systems allow artifacts to actively generate contextual descriptions and evolve their relational graphs as new experiences emerge [75]. Further, memory frameworks integrate memory management directly into the agent's action space. Unlike systems relying on external heuristics, these utilize reinforcement learning to autonomously dictate when a memory should be accessed, retained, or forgotten, optimizing the cognitive load of the Reasoning Engine dynamically [32].
```

### --- Page 0011 ---

```markdown
## E. Tool Orchestration Layer

The Tool Orchestration Layer acts as the middleware connecting the cognitive layers to external computational environments, APIs, and subsidiary sub-agents. It abstracts the complexities of API payload formatting, resource management, and execution limits, allowing the Reasoning Engine to interact with the environment through standardized interfaces. This layer is critical for transforming a theoretical reasoning path into actionable execution.

In sophisticated multi-agent ecosystems, tool orchestration is handled via specialized architectural primitives that enforce rigid hierarchy and state isolation. Hierarchical delegation allows a primary LLM agent to wrap a highly specialized secondary agent and invoke it as a functional tool. This facilitates the Coordinator/Dispatcher pattern, where a central agent manages requests and relinquishes control to specialists based on intent classification.

To manage execution flow without introducing unnecessary inference overhead, the orchestration layer employs deterministic routing components that control sub-agent execution structurally rather than cognitively. Sequential routers enforce strict pipeline execution, passing shared context between agents to ensure predictable data flow. Parallel routers manage concurrent fan-out operations—essential for reducing latency during independent multi-source data retrieval—before gathering results into a shared session state. Loop routers orchestrate iterative refinement, executing Generator-Critic primitives until a specific termination condition is met to prevent infinite recursion.

## F. Verification and Self-Correction

Agentic systems are inherently susceptible to cascading reasoning failures. In a multi-step workflow, a minor hallucination or incorrect tool invocation early in the execution graph can propagate, leading to systemic failure. Therefore, robust Verification and Self-Correction Modules must be integrated directly into the iterative loop to provide runtime supervision, reflection, and rigorous output validation.

These modules function by establishing a closed-loop Perception-Planning-Action-Reflection (PPAR) cycle. As illustrated in Figure 5, when the Reasoning Engine proposes a solution, it is first evaluated by a separate verification agent or internal critic. Domain-specific standalone agents illustrate that systems cannot rely solely on simple LLM self-reflection, as models suffer from evaluation blind spots [76]. Instead, self-verification relies on empirical testing, using as iterative simulation against ground-truth constraints [76].

If the verification module detects a factual inconsistency or syntax error, it generates structured feedback detailing the failure state. The Reasoning Engine incorporates this feedback to iteratively adjust the query formulation or switch retrieval strategies until the output passes all validation constraints. In scenarios where self-correction fails to converge, the Verification module triggers an escalation path through Human-in-the-Loop (HITL) intervention. Operating through policy engines, 

![The closed-loop Perception-Planning-Action-Reflection (PPAR) cycle with Human-in-the-Loop (HITL) escalation. This demonstrates the structural necessity of verification loops: outputs failing constraint checks are returned as structured feedback, and unresolvable loops are escalated to prevent autonomous hallucination.](assets/page_0011_img_1.png)

## VI. DESIGN PATTERNS IN AGENTIC RAG

Building on the architectural module decomposition established in Section V, this section abstracts away from specific implementations to identify reusable control-flow strategies. These design patterns specify how planning, retrieval, generation, verification, and memory updates are sequenced and iterated under a control policy [19]. As illustrated in Figure 6, these patterns operate as engineering-level motifs that can be combined and composed to dictate the operational tempo of the agent [12].

### A. Plan-Then-Retrieve Pattern

This pattern explicitly separates task decomposition from execution. The agent first produces a high-level plan or sub-question list, then performs retrieval conditioned on each step before composing a final answer [28, 29].

- **Control Flow:** (i) Plan/decompose → (ii) retrieve evidence per subtask → (iii) generate intermediate notes → (iv) synthesize final answer [77].
- **Strengths:** Makes information needs explicit and significantly improves compositional generalization in multi-step tasks [27].
- **Limitations:** Decomposition quality is critical; if the initial plan is flawed or ambiguous, the entire subsequent retrieval trajectory fails [77].
- **Typical Use Cases:** Multi-hop QA where evidence requirements can be enumerated in advance (e.g., HotpotQA) [3].
```

### --- Page 0012 ---

```markdown
# Page 0012

![Control-flow diagram showing Agentic RAG systems process design patterns through explicit decisions over task decomposition, retrieval timing, iterative refinement, and orchestration. This structural mapping highlights the transition from linear pipelines to cyclic loops.](assets/page_0012_img_1.png)

## B. Retrieve-Reflect-Refine Pattern

The agent alternates retrieval and generation with explicit reflection steps to decide if retrieved evidence is sufficient, and refines subsequent actions (e.g., query rewriting, retrieval gating) accordingly [6]. Recent work such as A-RAG [18] introduces hierarchical retrieval interfaces that allow agents to progressively refine context acquisition through staged document exploration, improving token efficiency and retrieval relevance.

- **Control Flow:** (i) Retrieve → (ii) draft partial answer → (iii) reflect on document utility → (iv) refine query → repeat until stop [5].
- **Strengths:** Improves factuality and citation accuracy by establishing a “retrieval-on-demand” directive signal rather than blindly passing context [6].
- **Limitations:** Relies heavily on the LLM’s inherent self-critique capabilities, which can suffer from evaluation blind spots or over-confidence.
- **Typical Use Cases:** Long-form attributed generation and open-domain QA where initial retrieval is typically imperfect [78].
- **Failure Modes:** Infinite loops where the agent repeatedly refines a query but retrieves the same unhelpful documents.
- **Cost/Latency Implications:** Introduces sequential iterations that compound latency and increase compute overhead, motivating budget-aware gating mechanisms [7], [8].

## C. Decomposition-Based Retrieval Pattern

Rather than producing a full plan upfront, the agent decomposes the query implicitly through stepwise reasoning, triggering retrieval mid-trajectory based on evolving hypotheses [9], [12]. Emerging approaches such as DLLM-Searcher [79] explore diffusion-based language models to parallelize reasoning trajectories, reducing latency while maintaining diverse search exploration.

- **Control Flow:** (i) Generate reasoning step → (ii) formulate retrieval action → (iii) incorporate observation → repeat [9].
- **Strengths:** Highly adaptable; allows the system to discover the next information need based on partial inference, mimicking human investigative behavior [12].
- **Limitations:** The repeated interleaving of reasoning and tool calls creates highly redundant prompt effects [80].
- **Typical Use Cases:** Complex investigative tasks where subsequent logical steps are entirely dependent on the specific facts uncovered in the previous step [3].
- **Failure Modes:** Reasoning drift, where the agent forgets the original objective after a long sequence of intermediate observations.
- **Cost/Latency Implications:** Extremely expensive computationally due to repeated prompt accumulation and sequential bottlenecking [80].

## D. Tool-Augmented Retrieval Loop Pattern

Retrieval is treated as just one tool among many (e.g., calculators, code execution, SQL). The agent dynamically chooses among these heterogeneous tools in an iterative loop to update its state [10].

- **Control Flow:** (i) Decide next tool → (ii) execute tool → (iii) process observation → (iv) update state → repeat [15].
```

### --- Page 0013 ---

```markdown
| **E. Multi-Agent Collaboration Pattern** |  |
|------------------------------------------|--|
| **Strengths:** Enables massive zero-shot generalization across domains requiring distinct modalities (math, search, code) while preserving core modeling ability [10], [81]. | **Typical Use Cases:** Medical, legal, and compliance domains requiring strict auditability and traceable evidence [88]. |
| **Limitations:** Tool routing reliability becomes a first-class failure point; agents frequently struggle with strict syntax formatting for complex APIs [11]. | **Failure Modes:** The agent forcibly misaligns generated claims with irrelevant evidence to satisfy a formatting requirement (false attribution). |
| **Control Flow:** (i) Assign roles → (ii) iterative message passing → (iii) integrate artifacts into final synthesis [52]. | **Cost/Latency Implications:** Effectively doubles the generation latency, as the system must complete an initial draft before the verification phase even begins. |
| **Typical Use Cases:** Broad knowledge-intensive tasks requiring non-textual computation or interaction with structured databases [10]. |  |
| **Failure Modes:** Tool hallucination (inventing nonexistent APIs) or failure to recover gracefully when an API returns an unexpected error code [80]. |  |
| **Cost/Latency Implications:** Variable cost depending heavily on the latency of the external APIs invoked. |  |

| **F. Retrieval-Grounded Self-Verification Pattern** |  |
|-----------------------------------------------------|--|
| The agent treats verification as a dedicated, first-class execution stage, retrieving evidence specifically to validate, refute, and attribute claims made in a direct response [81], [85]. | **Control Flow:** (i) Direct answer → (ii) extract checkable claims → (iii) retrieve evidence per claim → (iv) revise and attach citations [85]. |
| **Strengths:** Directly verifies hallucination and provides highly attributable, verifiable outputs supported by verified quotes [87]. | **Limitations:** Verification quality is ultimately bounded by the retriever's recall; it cannot correct a claim if the grounding truth is missing from the corpus [14]. |
| **Cost/Latency Implications:** Highest token amplification profile; cross-agent communication aggressively consumes token budgets. |  |

| **G. Human-As-A-Tool (HITL) Pattern** |  |
|---------------------------------------|--|
| This pattern models human oversight as a callable API within the action space. When epistemic uncertainty exceeds a defined threshold, the policy pauses execution to request disambiguation or supervision [15], [52]. | **Control Flow:** (i) Execute loop → (ii) detect ambiguity/risks threshold → (iii) pause for human input → (iv) resume execution with human observation [13]. |
| **Strengths:** Guarantees safety in high-stakes environments and strictly enforces evidence discipline via human feedback [15]. | **Limitations:** Fundamentally breaks continuous system autonomy and creates operational bottlenecks. |
| **Typical Use Cases:** High-stakes financial, medical, or administrative tasks where automated retrieval is inadequate and strict compliance oversight is mandatory. | **Cost/Latency Implications:** Negligible API cost, but introduces extreme wall-clock latency that halts the automated execution loop entirely [8]. |

| **VII. EVALUATION AND BENCHMARKING** |  |
|---------------------------------------|--|
| Despite the growing deployment of agentic RAG systems, current evaluation methodologies largely remain inherited from traditional retrieval or language generation tasks. These approaches primarily focus on final answer quality and fail to capture the multi-step reasoning, tool interaction, and decision dependencies that characterize agentic systems. As a result, commonly used benchmarks may obscure critical failure modes and provide incomplete signals about system reliability. This section therefore examines the limitations of existing evaluation practices and outlines a structured framework for assessing agentic RAG behavior. |  |
```

### --- Page 0014 ---

```markdown
| Design Pattern                | Core Question                                      | Termination Condition                     | Tradeoffs (Cost / Latency / Risk)                     | Representative Anchors          |
|-------------------------------|---------------------------------------------------|------------------------------------------|------------------------------------------------------|----------------------------------|
| Plan-then-retrieve            | “What subtasks must be answered before synthesis?” | All planned sub-questions answered      | High upfront planning cost; risk of brittle initial plans | Self-Ask [29], Plan-and-Solve [28] |
| Retrieve-refine               | “Is retrieval needed? Are these passages relevant?”| Reflection indicates sufficiency or budget exhausted | High latency due to sequential query iterations       | Ref-Gen [6], Iter-RefGen [5]    |
| Decomposed retrieval          | “Given the current reasoning state, what is missing?” | Answer reached with adequate evidence   | Extreme token accumulation; reasoning drift risk      | IRCoT [9], ReAct [12]            |
| Tool-augmented loops          | “Which heterogeneous tool to call now?”           | Tool results stabilize answer or verify halls | Variable latency; high risk of tool-syntax failure    | Toolformer [10], CRITIC [81]     |
| Multi-agent collaboration      | “Which agent should handle this task?”            | Cross-agent consensus reached           | Massive token amplification; coordination overhead     | AutoGen [52], MetaGPT [84]       |
| Self-verification             | “Which claims require checking against the corpus?”| Verification passes or abstention is triggered | Doubles baseline latency; bounded by retrieval recall  | CoVe [15], GopherCite [87]       |
| Human-as-a-tool              | “Is human input required for disambiguation?”     | Human resolves uncertainty and resumes loop | Extreme wall-clock latency; guarantees safety         | WebGPT [15], AutoGen [52]       |

Standard generation metrics were originally designed for static, single-turn text generation tasks and fail to capture the interactive and iterative behavior of agent systems [89, 90]. While traditional metrics evaluate the "engine" (the LLM's terminal text output), evaluative metrics assess the "car" (the entire system's behavior across planning, tool use, and environment interaction) [89].

Traditional metrics like BLEU or ROUGE focus on lexical similarity rather than semantic truth or reasoning adequacy. Consequently, they are incapable of distinguishing between a correct final answer reached through valid logic and one reached through valid planning [91]–[93]. To highlight these inadequacies, Table VII synthesizes a Metric Failure Analysis, demonstrating exactly how and why static metrics break down when applied to autonomous multi-step architectures.

To quantify the efficiency and correctness of these intermediate steps, agentic evaluation relies on specific trajectory-level metrics:

- **Progress Rate (PR)**: Progress Rate measures the fraction of reasoning steps that meaningfully advance task completion:
  
  $$
  PR = \frac{\text{Number of successful reasoning steps}}{\text{Total reasoning steps}}
  $$

- **Effective Information Rate (EIR)**: Effective Information Rate measures the efficiency of retrieved information used during reasoning:
  
  $$
  EIR = \frac{\text{Useful retrieved tokens}}{\text{Total retrieved tokens}}
  $$

Higher EIR indicates that the retrieval subsystem provides more relevant information relative to the overall retrieval volume.

### A. Evaluation Dimensions for Agentic RAG

To move beyond the limitations of static metrics, evaluation must be decomposed into specific behavioral dimensions that capture the lifecycle of an agentic decision [92].

### B. From Static Benchmarks to Evaluation Frameworks

Existing benchmarks for RAG focus heavily on static, one-shot evaluation [92]. To prevent the mere listing of leaderboards, Table VIII covers the current fragmented benchmarking landscape into a synthesis of target capabilities and their evaluation.
```

### --- Page 0015 ---

```markdown
# TABLE VII  
## METRIC FAILURE ANALYSIS: WHY STANDARD EVALUATION FAILS FOR AGENTIC RAG

| Metric               | Failure Dimension         | Why It Fails Agentic Systems                                                                 | Agentic Failure Case                                                                                     |
|----------------------|---------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| BLEU / ROUGE         | Lexical Rigidity          | Primarily measures surface-level lexical overlap; ignores semantic consistency and factual keypoints [91]. | An agent correctly diagnoses a condition but uses synonyms not in the reference text, receiving a failing score despite factual accuracy [94]. |
| Exact (EM)           | Match                     | Offers no flexibility for valid aliases or the superfluous reasoning detail often generated by agents [91]. | An agent outputs the correct entity but is factored because it included a valid reasoning trace before the target word [91]. |
| Final-Answer Accuracy | Trajectory Blindness      | Provides a "black box" view; cannot determine if the agent correctly reasoned or merely guessed [92]. | A math agent reaches the correct final digit through mutually canceling calculation errors, hiding a fundamental planning breakdown [90]. |
| Success Rate (SR)   | Credit Assignment         | Non-diagnostic; identifies that a failure occurred but fails to pinpoint the bottleneck (retrieval vs. tool call) [90]. | An agent correctly writes code but fails execution due to a syntax timeout; SR marks it 0, obscuring the successful reasoning [92]. |
| Pass@k               | Reliability Blindness     | Focuses on best-case capability rather than the consistency required for enterprise deployment [89]. | An agent succeeds once in ten attempts; while technically "capable," it is dangerously unreliable for production tasks [89]. |

## C. Toward a Structured Agentic Evaluation Pipeline

Because Agentic RAG systems exhibit iterative reasoning, tool invocation, and memory usage, evaluation must operate at multiple scopes of assessment [89]. As illustrated in Figure 7, we abstract these into a structured three-layer evaluation pipeline that moves from atomic actions to holistic system evaluation.

1) **Layer 1: Component-Level Assessment:** Isolates individual primitives to assess localized correctness before considering their interaction over time [92]. This includes evaluating the Planner (task decomposition), the Retriever (recall precision), and the Tool Executor (invocation accuracy and parameter F1 scores) [89]. It captures localized failure modes without conflating them with downstream reasoning errors.

2) **Layer 2: Trajectory-Level Coherence:** Examines how atomic actions compose into coherent reasoning sequences across interaction steps [92]. This layer tracks logical progression, adaptation to intermediate API responses, and memory consistency [90]. Metrics include Progress Rate and success ratios, capturing failure modes that static metrics overlook, such as compounding errors and infinite execution loops [89].

3) **Layer 3: System-Level Outcome:** Treats the agentic pipeline holistically, focusing on deployment-relevant properties [92]. At this scope, evaluation abstracts away internal structure to assess final completion, cross-agent coordination effectiveness, and output faithfulness [89][96]. Crucially, this layer must also incorporate Cost and Latency Awareness, measuring token amplification and Time-To-First-Token (TTFT) to ensure the system is economically viable for real-world deployment [90].

## D. Systemic Evaluation Gaps

Despite the layered framework proposed above, significant systemic gaps remain in the current literature. First, the reliance on LLM-as-a-judge methodologies creates a reproducibility crisis. While automated judges correlate with humans, they are highly sensitive to prompt sequencing and exhibit "sycophantic" biases toward their own generated output patterns, making stable baseline comparisons difficult as frontier models evolve [91][93].

Second, the field lacks standardized mechanisms for credit assignment. Current evaluations treat agents as black boxes, providing a single score that fails to pinpoint whether a failure occurred during planning, retrieval, or final synthesis [90][92]. Finally, methods for evaluating an agent's ability to maintain persistent state and episodic memory across long-horizon conversations (e.g., hundreds of turns) remain highly underdeveloped, leaving critical deployment realities untested [89].
```


### --- Page 0016 ---

```markdown
| Evaluation Framework | Targeted Capability | Limitation for Agentic RAG |
|----------------------|---------------------|-----------------------------|
| RGB & FaithEval [96, 98] | Noise robustness, negative rejection, and counterfactual adherence. | Assumes a single forward-pass; cannot evaluate dynamic query reformulation. |
| RAGBench (TRACe) [95] | Utilization, Relevance, Adherence, and Completeness across industries. | Static dataset; fails to capture multi-step tool use or environment interaction. |
| RAGEval & CRAG [93] | Keypoint-based factual accuracy and multi-hop reasoning coverage. | Evaluates final output via mock APIs but lacks metrics for reasoning efficiency or cost. |

$$
\text{Layer 1: Component-Level Assessment}
$$
- Planner: Task Decomposition
- Retriever: Real Precision
- Tool Executor: Innovation Accuracy

$$
\text{Layer 2: Trajectory-Level Coherence}
$$
- Composes into
- Logical Progression & Progress Rate
- Adaptation to Intermediate APIs
- Memory & State Consistency

$$
\text{Layer 3: System-Level Outcome}
$$
- Determines
- Final Task Completion
- Output Faithfulness & Grounding
- Cost, Tokens & Latency TTTF

![The Agentic RAG Evaluation Pipeline. This framework demonstrates the necessary structural shift from terminal output scoring to multi-layered assessment, capturing component-level tool accuracy, trajectory-level reasoning coherence, and system-level outcome fidelity.](assets/page_0016_img_1.png)

---

## VIII. INDUSTRY FRAMEWORKS AND REAL-WORLD SYSTEMS

The transition of Agentic RAG from academic prototype to production exposes how theoretical architectures are operationalized in practice. By embedding autonomy, iterative retrieval, and verifiable execution into enterprise workflows, industrial systems attempt to overcome the accuracy limitations of static generative models. This section evaluates the deployment of Agentic RAG across specialized domains, analyzes the orchestration frameworks that abstract these architectures, and details the systemic constraints of production deployment.

### A. Domain-Specific Implementations

In enterprise environments, proprietary data is heavily fragmented across secure document stores and specialized databases. Static RAG pipelines struggle when these domain-specific ontologies and access controls. Agentic architectures address this by utilizing multi-hop planning to fuse cross-document information. For example, systems like TURA (Tool-Augmented Unified Retrieval Agent) implement Directed Acyclic Graph (DAG) based planning to handle transactional financial data [101]. By modeling sub-tasks and data dependencies as a DAG, TURA orchestrates reasoning chains across both static documents and dynamic APIs, enforcing strict access governance during execution [101]. Furthermore, because retrieving and embedding sensitive enterprise records directly into the generation context introduces severe information leakage vulnerabilities, deploying these systems safely increasingly requires differentiating intra-context learning frameworks [102]. To further enforce strict access governance, future enterprise agents could integrate visual authentication models such as deep learning-based masked facial recognition [103] as a prerequisite tool for before accessing sensitive records.

Scientific research requires a different architectural emphasis: rigorous attribution and verifiable citation traces. Systems like PaperQA2 mitigate hallucination by treating the literature corpus as an interactive environment [104]. Rather than executing a single vector search, the agent uses a multi-phase loop: it generates targeted search queries, retrieves candidate chunks, and applies LLM-based Contextual Summarization to score evidence before generation [104]. The agent employs citation traversal tools to verify the provenance of its claims, demonstrating how hierarchical retrieval interfaces isolate and evaluate evidence systematically.

Software engineering represents a highly complex embedded environment where agents autonomously explore repositories, run diagnostic tests, and parse compliance logs [73]. The SWE-agent framework operationalizes this by providing an Agent-Component Interface (ACI) to isolate and execute codebase operations safely [73]. Instead of attempting full file overwrites—which exhaust context windows—the agent uses targeted diff patching and dynamic exploration [73]. This couples dynamic code retrieval with iterative execution.
```

### --- Page 0017 ---

```markdown
# PAGE_NAME: page_0017

## B. Industrial Orchestration Frameworks

The transition from bespoke academic prototypes to scalable enterprise applications is facilitated by orchestration frameworks. These platforms abstract memory management, tool integration, and control loops, providing the routing primitives necessary to engineer complex agent topologies [105].

Rather than hardcoding API payloads, developers utilize these frameworks to define architectural boundaries. For instance, LangGraph abstracts stateful, cyclic orchestration by modeling agent interactions as a directed graph, providing fine-grained control over state persistence and reflection loops [105]. Conversely, frameworks like Google’s Agent Development Kit (ADK) provide hierarchical routing primitives [106]. ADK orchestrates non-deterministic LLM agents using deterministic structural routers, leveraging the Model Context Protocol (MCP) to standardize external tool interfaces and ensure environment-agnostic deployment [106]. However, while MCP solves critical interoperability challenges by decoupling context from execution, securing these interfaces against adversarial tool poisoning and prompt injection remains a profound systemic challenge [107].

Other frameworks optimize for distinct control-flow paradigms. AutoGen performs agent interactions as asynchronous, event-driven data interface for conversational multi-agent coordination [105]. Conversely, Lamandeck, originally a static routing pipeline, now provides abstract query pipelines and index-centric memory routing [105]. Table X synthesizes how these industrial frameworks operationalize the core architectural modules (Planner, Controller, Memory, Orchestrator) defined in Section V.

## C. Deployment Implications and the Research Gap

Deploying these frameworks exposes operational bottlenecks rarely encountered in isolated academic benchmarks. The most critical constraint is latency stacking [70]. In static RAG, latency is bounded by a single retrieval and generation step. In Agentic RAG, every reasoning loop, load invocation, and reflection step compounds the total response time [70]. To mitigate this, systems construct layer-wise execution topology graphs, enabling the parallel execution of independent agent sub-tasks and concurrent security scanning [70].

Additionally, agents operating in non-deterministic loops can easily become trapped in infinite execution cycles if constrained with ambiguous API feedback. Without strict orchestration limits on recursion depth, autonomous agents rapidly exhaust API budgets [101]. Consequently, production systems mandate rigorous observability layers to monitor token economics and execution trajectories in real-time.

This highlights a structural divergence between academic research and industrial deployment. Academic prototypes frequently rely on monolithic LLMs executing unconstrained tool usage to maximize benchmark scores. Conversely, industry prioritizes determinism, utilizing constrained Agent-Computer Interfaces and lightweight, distilled routing models to achieve fidelity at a fraction of the computational cost [101]. Bridging this gap requires standardizing evaluation pipelines to measure computational efficiency and procedural control alongside final output accuracy.

Practical deployments of agentic RAG systems must also account for operational constraints such as latency limits, token budgets, and memory footprint restrictions. Industrial applications often impose limits on reasoning trajectory length and retrieval expansion to control inference cost and response time. These constraints motivate adaptive policies such as budget-aware retrieval triggers, early termination criteria, and hierarchical retrieval pipelines that minimize redundant context expansion. Designing agent policies that balance reasoning depth with computational efficiency remains a critical challenge for real-world agentic systems.

## IX. FAILURE MODES, SAFETY, AND RELIABILITY CHALLENGES

While the preceding sections chart the architecture and design patterns of Agentic RAG, this section addresses their systemic vulnerabilities. The shift from static retrieve-then-generate pipelines to multi-step, tool-integrated workflows introduces novel attack surfaces. Because agent systems operate iteratively, localized errors compound in ways that are qualitatively different from traditional RAG failures. As synthesized in Table X, this section provides a structured analysis of these failure categories, organized by their position in the agentic pipeline.

### A. Retrieval Drift and Query Misalignment

In static RAG, retrieval quality is determined entirely by the initial query. In Agentic RAG, the agent reformulates queries across iterations, introducing the possibility of semantic drift: a gradual divergence between the evolving query and the user’s original information need [9]. Query-rewriting approaches acknowledge this problem directly, noting that original queries frequently misalign with what the retriever can effectively resolve [78].

In multi-agent architectures, retrieval drift is compounded by delegation. When a planner agent decomposes a task and delegates sub-queries to retriever agents, the planner’s interpretation of sub-task requirements may diverge from what the retriever can meaningfully resolve [67]. Without explicit coercive retriever or retrieval-quality feedback loops, iterative query reformulation can wander indefinitely, consuming token budgets without approaching a satisfactory answer.
```

### --- Page 0018 ---

```markdown
| Framework    | Orchestrator Model       | Planner / Control Flow | Memory Routing            | Tool Abstraction                  |
|--------------|--------------------------|------------------------|---------------------------|------------------------------------|
| LangGraph [105] | Cyclic Directed Graphs   | State-machine nodes    | Persistent checkpointing   | Wrapped Python functions            |
| Google ADK [106] | Hierarchical Composition | Deterministic routing loops | Shared contextual state   | Model Context Protocol              |
| CrewAI [105]   | Role-based Sequential     | Process-driven delegation | Structured persona memory  | Assigned capability arrays          |
| AutoGen [104]  | Asynchronous Chat         | Event-driven conversation | Message history logs      | Executable code blocks              |
| LlamaIndex [105] | Query Pipelines          | Data-driven routing     | Index-centric retrieval    | LlamaPack interfaces                |

## B. Hallucination Despite Retrieval
RAG was initially motivated as a mechanism to reduce hallucination by grounding generation in retrieved evidence [1]. However, empirical studies demonstrated that retrieval does not eliminate this risk; retrieval-augmented legal research tools exhibited hallucination rates up to 33%, contradicting vendor claims [108]. This occurs when retrieved passages are typically relevant but factually insufficient, when multiple documents contain conflicting information [40], or when the model succumbs to the lost-in-the-middle effect [4].

In agentic settings, the hallucination risk is amplified by iteration. An intermediate generation containing a hallucinated claim may be used for subsequent retrieval or reasoning steps, causing the error to propagate and reinforce across iterations. While mechanisms like self-reflection attempt to address this by enabling the model to critique its own retrieved passages, the approach relies on the model's own judgments, which are fundamentally fallible [6].

## C. Tool Misuse and Cascading Errors
Agentic RAG systems extend LLMs beyond text generation to tool invocation, including database queries, API calls, and code execution. Each tool call introduces a potential failure point: the model may yield an inappropriate tool, formulate a malformed query, or encounter API timeouts [10]. ReWO explicitly evaluates robustness under tool-failure scenarios, noting the severe brittleness of repeated thought-action-observation loops [80].

In multi-step workflows, tool failures cascade. A failed API call produces an error message that the agent may misinterpret as valid output and incorporate into subsequent reasoning [84]. While systems implement critique loops where outputs are evaluated and revised based on feedback [81], the absence of robust fallback mechanisms at each tool invocation point presents a significant structural reliability gap. Furthermore, as agentic workflows increasingly incorporate multimodal tools, they inherently inherit the vulnerabilities of those underlying modules, such as the susceptibility of visual classifiers to stealthy adversarial perturbations and malicious payload injections [109].

## D. Prompt Injection in Iterative Retrieval
Agentic RAG systems that retrieve from open or semi-curated corpora are highly vulnerable to indirect prompt injection: adversarial content embedded in retrieved documents that manipulates the agent's behavior. Unlike static RAG, where the attack surface is limited to a single retrieval pass, agentic systems face a compounded risk because each iterative retrieval step offers a new opportunity to encounter injected content [110].

Injecting as few as five carefully crafted malicious documents into a corpus can cause RAG systems to generate attacker-specified answers with a 90% success rate [111]. In agentic settings, the consequences extend beyond generation errors: injected instructions can alter the agent's planning, causing it to invoke unintended tools, or retrieve information through subsequent actions [112]. The OWASP Top 10 for LLM Applications identifies this as a leading vulnerability, noting that models struggle to distinguish between trusted instructions and adversarial content in retrieved contexts [113].

## E. Memory Poisoning
Systems that maintain persistent memory across sessions introduce an additional attack vector. If an adversary can influence the content stored in an agent's long-term memory, all subsequent interactions conditioned on that memory are compromised. This attack survives session terminations, logouts, and device changes when memories are stored serverside [114].

In Agentic RAG architectures with episodic memory modules, memory poisoning alters the agent's future retrieval strategies, planning heuristics, and tool-use preferences. Unlike corpus poisoning, which affects a shared knowledge base, memory poisoning targets the agent’s personalized state, making detection exceptionally difficult because the corrupted information is specific to individual user sessions [32].

## F. Systemic Risk Amplification in Iterative Agents
The failure modes described above interact and compound in iterative agentic workflows, creating systemic risks that exceed the sum of individual failure categories. Three amplification mechanisms govern this degradation:
```

### --- Page 0019 ---

```markdown
# Cascading Failure Amplification: 
A single error at an early step (e.g., a hallucinated intermediate answer or failed tool call) propagates through subsequent iterations. Because agentic systems condition actions on the accumulated history, errors are integrated into the evolving system state rather than isolated.

# Compounded Hallucination Loops: 
When an intermediate hallucination is used as context for a subsequent query, the retriever may return passages that spuriously corroborate the hallucination, creating a self-reinforcing cycle that artificially increases the model’s confidence in incorrect information.

# Feedback Reinforcement Instability: 
In systems with reflection modules, the critique mechanism may be biased by the same errors it is meant to detect. If the reflection module operates under the same parametric biases as the generator, it may approve flawed outputs, leading to divergent behavior rather than convergence.

The autonomy introduced by agentic retrieval loops amplifies traditional LLM risks while introducing new systemic vulnerabilities such as cascading hallucinations, retrieval poisoning, and tool misuse. These risks emerge from feedback-driven decision processes rather than isolated generation errors. Addressing these structural vulnerabilities requires research beyond patch-based mitigation, motivating the grand challenges discussed in the next section.

## X. OPEN RESEARCH CHALLENGES AND FUTURE DIRECTIONS
The transition from static Retrieval-Augmented Generation (RAG) to agentic architectures expands the operational capabilities of retrieval-based systems, but it introduces structural complexities that current ad-hoc implementations cannot sustainably manage. As the field matures, research must pivot from empirical prototyping to developing theoretically grounded, scalable, and verifiable systems [12]. Currently, the development of Agentic RAG remains theoretically underspecified; disparate frameworks rely on customized heuristics for tool orchestration and memory management, a fragmentation that severely impedes reproducibility [26]. Furthermore, there is a distinct absence of theoretical frameworks that mathematically bound the behavior of autonomous retrieval loops, leaving the field reliant on empirical prompt engineering rather than formal guarantees [10].

To address these systemic bottlenecks, we formalize five grand research directions structured as doctoral-scale problems. These problems are not mutually exclusive and necessitate interdisciplinary approaches. As proposed in Figure 8, resolving these grand challenges requires integrating methodologies across multiple foundational system dimensions spanning short, medium, and long-term horizons. An consolidated overview of these five problems—detailing their primary risks, theoretical gaps, and core evaluation metrics—is provided in Table XI.

---

## A. Stable Adaptive Retrieval Under Planning Loops
- **Problem Statement:** How can iterative retrieval processes be stabilized under dynamic planning decisions without causing retrieval drift or infinite execution loops?
- **Why It Matters:** Unstable retrieval leads to cascading reasoning failures in multi-step tasks. If an autonomous agent fetches a misaligned document in step one, the error compounds, derailing the cognitive trajectory. The field currently lacks empirical standardization for halting iterative retrievals securely.
- **Current Limitation:** Systems rely on arbitrary heuristic query reformulation (e.g., rigid max_steps parameters) and lack formal stability guarantees or mathematical convergence proofs for the retrieval loop.
- **Evaluation Criteria:** Maximum task horizon length before reasoning collapse; state-transition convergence bounds; semantic drift penalty scores; and marginal utility of successive retrieval steps.
- **Methodological Approaches:** Context-theoretic modeling of the context window; reinforcement learning with strict stability constraints; retrieval confidence calibration utilizing Bayesian uncertainty estimation.

## B. Formal Evaluation of Agentic Reasoning Quality
- **Problem Statement:** How can we construct a scalable, automated evaluation framework that assesses an agent’s multi-step reasoning trajectory rather than just its terminal output?
- **Why It Matters:** Without rigorous trajectory evaluation, developers cannot verify whether a correct terminal answer was achieved through sound logic or stochastic luck, making it impossible to guarantee safety in high-stakes domains [115]. This vulnerability is particularly evident in clinical applications, where recent empirical evaluations demonstrate that while advanced reasoning models achieve high overall diagnostic accuracy, they still exhibit severe performance gaps across specific disease categories, necessitating strict trajectory verification [116].
- **Current Limitations:** Existing metrics heavily favor static generation evaluation. Attempts at automated trajectory scoring lack standardized rubrics for intermediate step verification and suffer from evaluator-generator coupling biases.
- **Evaluation Criteria:** Trajectory inter-rater reliability (Cohen’s κ) between automated judges and experts; false positive rates for intermediate tool invocations; and quantifiable correlation coefficients between reasoning path efficiency and output quality.
- **Methodological Approaches:** Development of deterministic verification state machines; automated generation of counterfactual retrieval datasets to test against resilience; multi-dimensional reward modeling focusing on logical coherence.
```

### --- Page 0020 ---

```markdown
# PAGE_NAME: page_0020

## TABLE X
### STRUCTURED FAILURE-MODE CATEGORIZATION FOR AGENTIC RAG SYSTEMS

| Failure Mode         | Pipeline Stage         | Root Cause                                         | Agentic Amplification Factor                          | Severity / Impact                                   |
|----------------------|-----------------------|---------------------------------------------------|------------------------------------------------------|----------------------------------------------------|
| Retrieval Drift      | Iterative Retrieval    | Semantic divergence in query reformulation        | Compounds across iterations without convergence guarantees | Moderate (Degrades accuracy, increases cost)      |
| Hallucination        | Generation             | Context insufficiency or positional bias          | Hallucinated outputs become retrieval context in next iteration | High (Corrupts downstream logic and planning)      |
| Tool Misuse          | Tool Orchestration     | Malformed queries, API failures                    | Errors propagate through dependent downstream tool calls | High (Causes systemic execution crashes)            |
| Prompt Injection     | Retrieval Context      | Adversarial content in retrieved documents         | Each retrieval iteration exposes new injection surface | Critical (Enables unauthorized data exfiltration)  |
| Memory Poisoning     | Memory Systems         | Adversarial manipulation of persistent state      | Corrupted memory affects all future sessions and decisions | Critical (Persistent, cross-session compromise)    |
| Feedback Instability  | Reflection            | Reflection module shares generator biases          | Self-critique may approve errors or reject correct outputs | Moderate (Prevents loop convergence)               |

![Detailed description of the chart](assets/page_0020_img_1.png)

### Fig. 8
The interdisciplinary mapping of the proposed doctoral-scale grand problems across foundational system dimensions and research time horizons. Addressing these challenges requires systemic integration rather than isolated optimization.

## C. Memory Robustness and Poisoning Resistance

- **Problem Statement:** How can Agentic RAG systems with persistent read/write memory be secured against adversarial data injection that corrupts the control policy over time?
- **Why It Matters:** While Section IX diagnoses the systemic vulnerabilities of persistent memory, the theoretical gap lies in developing architectural immunity. The field requires formal guarantees to ensure an autonomous policy remains uncorrupted after ingesting adversarial content into episodic memory [117].
- **Current Limitations:** Existing defenses rely on superficial input optimization or static guardrails, which fail entirely when malicious triggers are mapped to unique, stealthy regions in the vector embedding space.
- **Evaluation Criteria:** Provable state recovery rates post-injection; cross-session leakage containment bounds; and the Attack Success Rate (ASR) of latent triggers evaluated strictly under formal verification constraints.

### Methodological Approaches:
- Implementation of cryptographic memory provenance tracking; anomaly detection in latent vector spaces to isolate optimized backdoor triggers; memory compartmentalization architectures with strict privilege separation.

## D. Cost-Aware Autonomous Orchestration

- **Problem Statement:** How can Agentic RAG orchestrators dynamically balance the trade-off between the depth of autonomous reasoning and the financial and computational cost of execution?
- **Why It Matters:** Multi-agent collaboration introduces severe token amplification. This problem explicitly targets economic optimality under budget constraints. Without formal cost-aware routing, deploying Agentic RAG at...
```

### --- Page 0021 ---

```markdown
# TABLE XI  
## SUMMARY OF GRAND RESEARCH PROBLEMS AND INTERDISCIPLINARY ROADMAP FOR AGENTIC RAG

| Grand Problem                  | Primary Risk                                   | Theoretical Gap                               | Core Evaluation Metric                     | Interdisciplinary Domain          |
|--------------------------------|------------------------------------------------|----------------------------------------------|-------------------------------------------|-----------------------------------|
| 10.1 Stable Retrieval           | Semantic drift and reasoning collapse          | Lack of formal convergence proofs for context loops | State-transition convergence bounds       | Control Theory, RL                |
| 10.2 Reasoning Evaluation       | Undetected logical failures                     | Absence of intermediate trajectory verification | Trajectory inter-rater reliability        | Formal Verification                |
| 10.3 Memory Security            | Persistent episodic poisoning                  | No robust state modeling against latent triggers | Provable state recovery rate              | Systems Security                   |
| 10.4 Cost Orchestration        | Token explosion and latency stacking           | No budget-aware multi-agent routing optimization | Pareto efficiency (Compute vs. Accuracy) | Operations Research                |
| 10.5 Trust Calibration          | Overconfidence in corrupted context            | Lack of dynamic uncertainty bounds during retrieval | Expected Calibration Error (ECE)         | HCI, Statistics                   |

---

The grand challenges identified here highlight the systemic research bottlenecks preventing the deployment of truly autonomous, reliable Agentic RAG. Addressing these gaps requires an interdisciplinary convergence of control theory, formal verification, and systems engineering. By solving these doctoral-scale problems, the field can transition Agentic RAG from the empirically grounded heuristics of today into the rigorously bounded, partially observable sequential decision processes formalized in Section III. Having charted this theoretical roadmap, Section XI synthesizes the core structural takeaways of this Systematization of Knowledge.

## XI. CONCLUSION

This Systematization of Knowledge unified the emerging landscape of Agentic Retrieval-Augmented Generation through formal definitions, structural taxonomy, architectural decomposition, evaluation reform, and systemic risk analysis. By mapping the transition from static, single-pass retrieval pipelines to dynamic, policy-driven reasoning loops, this paper provided a comprehensive foundation for understanding how large language models autonomously orchestrate external tools, manage persistent memory, and adapt to environmental feedback.

By distinguishing agentic behavior from iterative retrieval and grounding it within a sequential decision-making framework, we clarified conceptual boundaries that are often conflated in current literature. Our analysis demonstrated that true autonomy requires explicit modular separation between strategic planning, active retrieval, and robust state management. Furthermore, we established that evaluating these architectures necessitates a paradigm shift from static terminal metrics to multi-dimensional trajectory assessments capable of adding interpretive logic and tool-use correctness.

As agentic systems continue to evolve, rigorous formalization, evaluation standardization, and safety guarantees will determine whether these architectures mature into reliable reasoning systems or remain experimental extensions of retrieval pipelines. Resolving the doctoral-scale challenges identified in this roadmap—ranging from stable retrieval convergence to...
```

### --- Page 0022 ---

```markdown
# Memory Poisoning Resistance

Memory poisoning resistance—requires interdisciplinary collaboration across control theory, cybersecurity, and operations research.

A central insight emerging from this systematization is that academic RAG systems should be viewed not merely as extensions of retrieval pipelines, but as sequential decision-making systems in which language models coordinate response, retrieval, and tool interaction across multiple steps. Recognizing this shift is essential for designing robust architectures, developing meaningful evaluation methodologies, and understanding the broader reliability implications of deploying such systems in real-world environments. Ultimately, transitioning from empirical heuristics to theoretically bounded frameworks is the prerequisite for deploying trustworthy autonomous knowledge systems in high-stakes environments.

## References

| [1] P. Lewis, E. Perez, A. Pitkus, F. Petrov, V. Karpuškhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. [Online]. Available: https://arxiv.org/abs/2010.11401 |
| [2] V. Karpuškhin, B. Oguz, S. Min, P. Lewis, W. L. W. E. S. Xu, D. Chen, and W. Yih, "Dense passage retrieval for open-domain question answering," in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2020. [Online]. Available: https://aclweb.org/anthology/2020.emnlp-main.620 |
| [3] Z. Yang, P. Q. Si, S. Zhang, Y. Beygelzimer, W. Cohen, R. Salakhutdinov, and A. W. T. Y. Wang, "Improving language models by retrieving from trillions of tokens," in *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2021. [Online]. Available: https://arxiv.org/abs/2110.12589 |
| [4] E. F. Liu, K. L. Hewitt, A. Paranjape, H. Bevilacqua, F. Petroni, and P. Lewis, "Towards the ability to reason with language models under uncertainty," *Transactions of the Association for Computational Linguistics*, vol. 12, pp. 157–173, 2020. [Online]. Available: https://aclanthology.org/2020.tacl-1.7 |
| [5] Y. Shao, T. Cheng, Y. Shen, M. Huang, N. Duan, and W. Chen, "Enhancing retrieval-augmented language models with iterative end-to-end training," in *Proceedings of the Association for Computational Linguistics (ACL)*, 2022. [Online]. Available: https://aclanthology.org/2022.findings-acl.620 |
| [6] T. Jiang, F. F. Xu, L. Gu, Z. Sun, Q. Liu, J. Dwivedi, Y. Yu, Y. Jiang, and G. Zhang, "A retrieval-augmented generation approach," arXiv preprint arXiv:2305.06983, 2023. [Online]. Available: https://arxiv.org/abs/2305.06983 |
| [7] B. Cheng, Y. S. Li, Q. Zhao, Z. Yin, Y. Shao, L. Li, T. Sun, H. Yan, and X. Qiu, "Unified active retrieval for retrieval-augmented generation," in *Finding the Association for Computational Linguistics: EMNLP 2024*, 2024. [Online]. Available: https://aclweb.org/anthology/2024.emnlp-999 |
| [8] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, "Retrieving retrieval with end-to-end thought-provoking for knowledge-intensive tasks," in *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: Long Papers*, 2023. [Online]. Available: https://aclanthology.org/2023.acl-long.157 |
| [9] T. Schick, J. Dwivedi, Y. Ru, D. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, "Tuning language models can teach themselves to use tools," arXiv preprint arXiv:2302.04761, 2023. [Online]. Available: https://arxiv.org/abs/2302.04761 |
| [10] E. Kapars, A. Singer, J. Anisette, E. Omer, A. Petrov, et al., "MRKL systems: A modular, neuro-symbolic architecture that combines large language models, the world wide web, and discrete programs," arXiv preprint arXiv:2205.00445, 2022. [Online]. Available: https://arxiv.org/abs/2205.00445 |
| [11] S. Yao, Y. Zhao, D. Yu, N. Du, L. Shafran, K. Ramesh, and Y. Chen, "ReAct: Synergizing reasoning and acting in language models," in *International Conference on Learning Representations (ICLR)*, 2023. [Online]. Available: https://arxiv.org/abs/2302.03663 |
| [12] N. Shinn, B. Labash, A. Gopinath, K. Narasimhan, and S. Yao, "Reflection: Language agents with verbal reinforcement," arXiv preprint arXiv:2303.11360, 2023. [Online]. Available: https://arxiv.org/abs/2303.11360 |
| [13] A. Belinkov, V. Q. Tran, P. Vega, K. Aharoni, D. Andor, L. Baldwin, S. Ciaramita, J. Eisenstein, K. Ganchev, J. Herzig, K. H. Kwiatkowski, J. M. Ji, N. L. Senterian Saralizgi, T. Schuster, W. W. Cohen, M. Collins, D. Das, D. Metzler, S. Petrow, and K. Wootters, "Attributed question answering: Evaluation and modeling for attributed large language models," arXiv preprint arXiv:2212.80370, 2022. [Online]. Available: https://arxiv.org/abs/2212.80370 |
| [14] R. Wason, J. Hilton, S. Balaji, J. W. P. Abbeel, et al., "WebGPT browser-assisted question answering," arXiv preprint arXiv:2112.09332, 2021. [Online]. Available: https://arxiv.org/abs/2112.09332 |
| [15] A. Gupta, B. Mehta, S. Kumar, and T. Rho, "Generating augmented generations: A survey on augmented generation," arXiv preprint arXiv:2501.9501, 2023. [Online]. Available: https://arxiv.org/abs/2501.9501 |
| [16] P. Ferrarini, M. Cvetkovic, A. Prencipe, and G. D'Amico, "Is acting with rag? an experimental comparison of rag approaches," arXiv preprint arXiv:2601.00711, 2026. [Online]. Available: https://arxiv.org/abs/2601.00711 |
| [17] M. Du, B. Xu, C. Su, S. Wang, J. Wang, and Z. Mao, "Retrieval-augmented generation via hierarchical neural references," arXiv preprint arXiv:2602.02462, 2022. [Online]. Available: https://arxiv.org/abs/2602.02462 |
| [18] W. M. P. Wang, "A new paradigm for language models," in *Proceedings of the Fourth Workshop on the Practical Use of Colored Petri Nets and the CPN Tools (CPN)*, 2016. [Online]. Available: https://www.itu.dk/research/publications/workflow-patents-on-the-express-power-of-petri-net-based-workflows |
| [19] A. Vashwani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, K. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762 |
| [20] J. Kaplan, S. McCandlish, T. Henighan, T. Brown, D. Chess, C. R. Gray, A. Radford, J. Wu, and D. Amodei, "Scaling laws for neural language models," arXiv preprint arXiv:2001.08361, 2020. [Online]. Available: https://arxiv.org/abs/2001.08361 |
| [21] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, I. Kaplan, P. Dhariwal, A. Neelakantan, S. Sutskever, and D. Amodei, "Language models are few-shot learners," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020. [Online]. Available: https://arxiv.org/abs/2005.14165 |
| [22] J. Wei, K. Wang, D. Schuurmans, M. Bosma, B. Henter, F. Xia, E. Chi, Q. Lee, and D. Zhou, "Thought-provoking ethics reasoning in large language models," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022. [Online]. Available: https://arxiv.org/abs/2211.19903 |
| [23] L. Huang, W. Yu, M. Wang, Z. Feng, H. Wang, Q. Chen, W. Feng, X. Feng, B. Qin, and T. Liu, "A survey on hallucination in large language models: Principles, techniques, and open questions," arXiv preprint arXiv:2311.05232, 2023. [Online]. Available: https://arxiv.org/abs/2311.05232 |
| [24] G. Izacard and E. Grave, "Leveraging passage retrieval with generative models for open domain question answering," in *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, 2021. [Online]. Available: https://aclanthology.org/2021.eacl-main.7 |
| [25] J. Wang, C. Ma, X. Feng, Q. Zhang, H. Yang, J. Zhang, Z. Chen, J. Yang, C. Chen, Y. Lin, W. Zhao, X. Zhao, and J.-R. Wen, "A survey on large language model based autonomous agents," *Frontiers of Computer Science*, vol. 18, no. 2, 186345, 2024. [Online]. Available: https://arxiv.org/abs/2308.11432 |
```

### --- Page 0023 ---

```markdown
| Reference                                                                                          | 
|----------------------------------------------------------------------------------------------------| 
| [27] D. Zhou et al., “Least-to-most prompting enables complex reasoning in large language models,”  arxiv preprint arXiv:2205.10625, 2022. [Online]. Available: https://arxiv.org/abs/2205.10625 |
| [28] L. Wang et al., “Plan-and-solve prompting improves zero-shot chain-of-thought reasoning by large language models,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), Long Papers, 2023. [Online]. Available: https://aclanthology.org/2023.acl-long.1471 |
| [29] D. Press et al., “Measuring and narrowing the compositionality gap in language models,” arxiv preprint arXiv:2202.03350, 2022. [Online]. Available: https://arxiv.org/abs/2202.03350 |
| [30] S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Co, and K. Narasimhan, “Three thoughts: Deliberate problem solving and large language models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2024. [Online]. Available: https://arxiv.org/abs/2305.10601 |
| [31] J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and S. M. Bernstein, “Generative agents: Interactive simulation of human behavior,” in Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST), 2023. [Online]. Available: https://arxiv.org/abs/2304.03442 |
| [32] Y. Xu, Y. Xie, Q. Tan, F. Feng, and L. Wu, “Agenta memory: Learning long-term and short-term memory for large language model agents,” arxiv preprint arXiv:2201.0885, 2026. [Online]. Available: https://arxiv.org/abs/2201.0885 |
| [33] P. Liu, K. Liu, H. Ayyad, P. Bedi, M. Bellegia, L. Pétron, and L. Fang, “List of the middle ground: A framework for understanding the Association for Computational Linguistics,” vol. 12, pp. 157–173, 2024. |
| [34] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Interactive retrieval with chain-of-thought prompting for multi-step questions,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, Long Papers, 2023. [Online]. Available: https://aclanthology.org/2023.acl-long.1475 |
| [35] A. Khatapoth, P. Koltchinskii, and A. Zharov, “Balancing robust risk minimization and generalization,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2304.03442 |
| [36] W. Shi, M. Xia, A. R. Fabbri, L. Zettlemoyer, and R. Doshi-Velez, “Trusting your evidence: Hiccups lessen the worst-case risk of large language models,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics, 2024, pp. 1234–1249. |
| [37] T. Khot, H. Trivedi, M. Finlayson, Y. F. K. Richardson, P. Clark, and K. Sabharwal, “Multihop question via single-hop question generation,” “Transactions of the Association for Computational Linguistics,” vol. 9, pp. 537–554, 2021. |
| [38] T. Gao, H. Yen, Y. Duo, and C. Chen, “Enabling large language models to generate correct actions,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 6165–6184. |
| [39] J. Richards, “Auto-GPT: An autonomous GPT-4 agent,” https://github.com/Significant-Gravitas/Auto-GPT, 2023. |
| [40] Y. Gao, Y. Xiong, X. Gao, K. Ji, P. Yan, Y. B. Dai, J. Sun, M. Jiang, and H. Wang, “Retrieval-augmented generation for large language models: A survey,” 2023. [Online]. Available: https://arxiv.org/abs/2312.10997 |
| [41] W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li, “A survey on RAG meeting LLMs: Towards retrieval-augmented large language models,” 2024. [Online]. Available: https://arxiv.org/abs/2405.06211 |
| [42] OpenAI, “Function calling — openai api documentation,” 2025. [Online]. Available: https://developers.openai.com/docs/api-reference/function-calling |
| [43] Anthropic, “Code use without code: Overview (claude api docs),” 2026. [Online]. Available: https://www.anthropic.com/claude/docs/code-use-without-code-overview |
| [44] Google, “Agent development kit (adk) documentation,” 2026. [Online]. Available: https://google.github.io/adk-docs/ |
| [45] OpenAI, “Agents — openai api documentation,” 2026. [Online]. Available: https://developers.openai.com/docs/api-reference/agents-sdk |
| [46] LangChain, “Langchain agents documentation,” 2026. [Online]. Available: https://docs.langchain.com/docs/python/langchain/agents |
| [47] Llamanlnd, “Llamanlnd agents documentation,” 2026, accessed 2026-02-24. [Online]. Available: https://developers.lamanlnd.se/python/frameworks/ |
| [48] Hugging Face, “smokelatest documentation,” 2026, accessed 2026-02-24. [Online]. Available: https://huggingface.co/smoke/latest |
| [49] Hugging Face, “smokelatest (github repository),” 2026, accessed 2026-02-24. [Online]. Available: https://github.com/huggingface/smoke |
| [50] Y. Shen et al., “HuggingFace: Solving AI tasks with graphs and its friends in hugging face,” 2023. [Online]. Available: https://arxiv.org/abs/2305.09572 |
| [51] Q. Wu et al., “Autogen: Enabling next-gen LLM applications via multi-agent conversation,” 2023. [Online]. Available: https://arxiv.org/abs/2308.08155. |
| [52] LangChain-AI, “Langchain github repository,” 2026, accessed 2026-02-24. [Online]. Available: https://github.com/langchain-ai/langchain |
| [53] Microsoft, “AutoGen documentation: Multi-agent conversation framework,” 2026, accessed 2026-02-24. [Online]. Available: https://microsoft.github.io/autogen/2026/Use-Cases/Agent_chat/ |
| [54] LangChain, “LangChain documentation,” 2026. [Online]. Available: https://www.langchain.com/ |
| [55] Y. Gao, D. L. Pasupat, A. Chen, A. T. Chaganty, Y. Fan, Y. Zhao, and K. Guu, “RARR: Researching and revising what language models say, using language models,” in Proceedings of the Association for Computational Linguistics (Volume I: Long Papers), Toronto, Canada, 2023, pp. 1647–1658. |
| [56] Z. Liu, Z. Lu, Y. Yan, S. Wang, S. Yu, Z. Zeng, X. Xiao, and Y. Zhang, “A survey on large language models with long-term memory,” 2023. [Online]. Available: https://arxiv.org/abs/2402.13537 |
| [57] Y. Yu, Y. Luo, Y. Xie, Q. Tan, J. Li, and L. Wu, “Agenta memory: Learning long-term and short-term memory for large language model agents,” 2026. [Online]. Available: https://arxiv.org/abs/2402.13537 |
| [58] A. Wang, J. Araki, Z. M. R. Parveez, and G. Neubig, “Learning to filter context for retrieval-augmented generation,” 2023. [Online]. Available: https://arxiv.org/abs/2311.08377 |
| [59] C. Chintala, T. Formal, V. Nikolov, and S. Ruder, “Efficient and robust context pruning for retrieval-augmented generation,” 2023. [Online]. Available: https://arxiv.org/abs/2301.16214 |
| [60] J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and S. M. Bernstein, “Generative agents: Interactive simulation of human behavior,” 2023. [Online]. Available: https://arxiv.org/abs/2304.03442 |
| [61] W. Zheng, L. Guo, “Enabling large language models with long-term memory,” 2023. [Online]. Available: https://arxiv.org/abs/2305.10237 |
| [62] P. Cacker, S. Wooders, K. Lin, W. Fangs, S. G. Patil, I. Stoica, and J. E. Gonzalez, “Memory for large language systems,” 2024. [Online]. Available: https://arxiv.org/abs/2310.05860 |
| [63] Anthropic, “Introducing advanced use of the claude developer platform,” 2025, accessed 2026-02-24. [Online]. Available: https://www.anthropic.com/coming/advanced-code-use |
| [64] T. Nguyen, P. Chin, and Y.-W. Tai, “MaG-RL: Multi-agent retrieval-augmented generative chain-of-thought reasoning,” arxiv preprint arXiv:2500.2095, 2025. |
| [65] Y. Chen, F. Zhang, T. H. S. Wang, Z. Yang, M. Zhong, and J. Mao, “Yake: Bridging the strategic operation and the strategic rate,” arxiv preprint arXiv:2019.2016, 2026. |
| [66] A. Aavani, “Capturing D of the expressive power and efficient evaluation of polynomial retrieval,” arxiv preprint arXiv:2601.8742, 2026. |
| [67] S. Xu, W. Hao, and T. Li, “Ka-arg: Integrating retrieval-augmented generation for intelligent educational question-answering,” arXiv preprint arXiv:1254.2075, 2025. |
| [68] X. Shi, M. Zheng, and Q. Luo, “Learning latency-aware orchestration for parallel multi-agent systems,” arxiv preprint arXiv:2601.1500, 2026. |
```

### --- Page 0024 ---

```markdown
| Reference                                                                                          | Citation                                                                                          |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [71] A. Mahajan and U. Yadav, "Chunking, retrieval, and re-ranking: an empirical evaluation of RAG architectures for policy document question answering," arXiv preprint arXiv:2011.15457, 2020. Available: https://arxiv.org/abs/2011.15457 |                                                                                                   |
| [72] K. Mukherjee et al., "Lind-Down: reverse forensics for threat modeling," arXiv preprint arXiv:2508.2132, 2023. Available: https://arxiv.org/abs/2508.2132 |                                                                                                   |
| [73] J. Yang, C. E. Jimenez, & W. K. Liert, S. Yao, K. R. Narasimhan, and O. Press, "SWEAGLE: Agent-complete neural architecture for long-horizon LLM agents," arXiv preprint arXiv:2601.09921, 2026. |                                                                                                   |
| [74] J. Logan, "Continuing neural architectures for long-horizon LLM agents," arXiv preprint arXiv:2502.12170, 2025. |                                                                                                   |
| [75] H. Pan, Z. Lin, Z. Wang, X. Chen, K. Ding, and J. Zhao, "Towards fine-tuned verilog bit assistant: Self-verification and self-correction," arXiv preprint arXiv:2406.00115, 2024. |                                                                                                   |
| [76] S. Min, W. Zhong, L. Zettlemoyer, and H. Hajishirzi, "Multi-hop reading comprehension through question decomposition and reasoning," in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019. Available: https://aclweb.org/anthology/P19-1613 |                                                                                                   |
| [77] X. Ma, Y. Chen, H. Han, and N. Bougouin, "Revisiting retrieval for retrieval-augmented large language models," arXiv preprint arXiv:2365.2403, 2023. Available: https://arxiv.org/abs/2365.2403 |                                                                                                   |
| [78] Y. Zhang, and "Diln-sear: Diffusion large language models for search and reasoning," arXiv preprint arXiv:2404.XXXXXX, 2024. Available: https://arxiv.org/abs/2404.XXXXXX |                                                                                                   |
| [79] B. Xu, Z. Fang, B. E. S. Mukherjee, Y. Liu, and D. Xu, "Rewoot: Language modeling from observations for efficient augmented language models," arXiv preprint arXiv:2305.18323, 2023. Available: https://arxiv.org/abs/2305.18323 |                                                                                                   |
| [80] D. Xu et al., "Large language models are not all created equal," arXiv preprint arXiv:2305.11738, 2023. Available: https://arxiv.org/abs/2305.11738 |                                                                                                   |
| [81] J. Chen et al., "Evaluating large language models: A survey," arXiv preprint arXiv:2303.17760, 2023. Available: https://arxiv.org/abs/2303.17760 |                                                                                                   |
| [82] C. Qian et al., "Chatbot: Communicative agents in dialogue," in Proceedings of the 6th Annual Meeting of the Association for Computational Linguistics, Long Papers, 2024. Available: https://aclanthology.org/2024.acl-long.8 |                                                                                                   |
| [83] S. Hong et al., "Metaplt: Meta prompting for a multi-agent collaborative framework," in 2023 International Conference on Language Representation (LREP), 2023. Available: https://lrep2023.org |                                                                                                   |
| [84] D. Bhushan et al., "Chat-verification reduces hallucinations in large language models," arXiv preprint arXiv:2309.11495, 2023. Available: https://arxiv.org/abs/2309.11495 |                                                                                                   |
| [85] J. Liu et al., "Search-augmented generative model for large language models," arXiv preprint arXiv:2505.XXXXXX, 2024. Available: https://arxiv.org/abs/2505.XXXXXX |                                                                                                   |
| [86] J. Bernt et al., "Teaching language models to support humans with verified quotes," arXiv preprint arXiv:2203.11147, 2022. Available: https://arxiv.org/abs/2203.11147 |                                                                                                   |
| [87] H. Gao et al., "Enabling large language models to generate text citations," arXiv preprint arXiv:2305.14627, 2023. Available: https://arxiv.org/abs/2305.14627 |                                                                                                   |
| [88] M. Mohammadi et al., "L1 vs. L2: Evaluation and benchmarking of LLM agents: A survey," 2025. Available: https://arxiv.org/abs/2305.14627 |                                                                                                   |
| [89] A. Yadhu, L. Eiden, A. Li, G. Liu, Y. Zhao, B. Ranjan, A. Cohen, and M. Shmueli-Scheuer, "Survey on evaluation of LLM-based agents," 2025. Available: https://arxiv.org/abs/2305.18416 |                                                                                                   |
| [90] D. Malin, T. Kafle, and N. Bougouin, "A review of fallacies for hallucination assessment in large language models," 2025. Available: https://arxiv.org/abs/2305.20169 |                                                                                                   |
| [91] W. Yi, T. Wu, Z. Liu, X. Ning, Z. Zhao, Z. Li, G. Qiu, J. Lin, D. Fu, Z. Li, M. Di, Z. Zhou, W. Ba, Y. Li, G. Qiu, X. Yang, X. Tang, and Y. Xiao, "Agent reasoning for LLM-based models," 2026. Available: https://arxiv.org/abs/2601.12538 |                                                                                                   |
| [92] N. Zhu, X. Liu, D. Xu, Y. M. Z. Liu, Y. Ru, Y. Wang, Y. Li, X. Zhang, X. Han, Z. Liu, and M. Sun, "RAGEVAL: Scenario specific RAG evaluation dataset generation framework," in Association for Computational Linguistics, Vienna, Austria, 2025. |                                                                                                   |
| [93] K. Zlong, Z. Lin, Z. Liu, and N. Bougouin, "Benchmarking retrieval-augmented generation," in Association for Computational Linguistics, Bangkok, Thailand, 2024. |                                                                                                   |
| [94] R. Firth, M. Belhaj, and S. Ayan, "Explainable benchmark for retrieval-augmented generation systems," 2025. Available: https://arxiv.org/abs/2401.03777 |                                                                                                   |
| [95] Y. Mngs, P. Purushwalkam, S. Pandit, Z. K. Z. Nguyet, C. Xiong, and S. Jyoty, "Faithful: Can your language model stay faithful to context, even if the moon is made of marshmallows?" 2024. Available: https://arxiv.org/abs/2410.03777 |                                                                                                   |
| [96] A. Costelloi, M. Allen, R. Hauksson, G. Schunke, S. Harharan, C. Cheng, W. Li, and A. Yadhu, "GAMENB: Evaluating strategic reasoning abilities of LLM agents," arXiv, 2024. Available: https://arxiv.org/abs/2406.01651 |                                                                                                   |
| [97] J. Chen, H. Lin, X. Han, and L. Sun, "Benchmarking large language models in retrieval-augmented generation," 2023. Available: https://arxiv.org/abs/2304.01341 |                                                                                                   |
| [98] Y. Chen et al., "Draco: Discriminative reasoning for comprehensive agent evaluation," arXiv preprint arXiv:2404.XXXXXX, 2024. Available: https://arxiv.org/abs/2403.XX |                                                                                                   |
| [99] H. Wang et al., "CL-Check: an evaluation context learning benchmark for RAG," arXiv preprint arXiv:2406.2406.XXXXXX, 2024. Available: https://arxiv.org/abs/2406.2406 |                                                                                                   |
| [100] Z. Zhao, Y. Dong, A. Liu, L. Zheng, Y. Li, and "Tura: Tool-assisted reasoning for search," arXiv preprint arXiv:2406.2406.XXXXXX, 2024. Available: https://arxiv.org/abs/2406.2406 |                                                                                                   |
| [101] B. Bhusal, M. Acharya, R. Kaur, C. Samplawish, A. Roy, A. D. Cebh, R. Chacha, and S. Jha, "Privacy preserving in retrieval-augmented language models," arXiv preprint arXiv:2306.12503, 2025. Available: https://arxiv.org/abs/2306.12503 |                                                                                                   |
| [102] M. S. H. and A. M. "A survey of LLMs and a framework for their evaluation," arXiv preprint arXiv:2406.2406.XXXXXX, 2024. Available: https://arxiv.org/abs/2406.2406 |                                                                                                   |
| [103] M. D. Hammerling, M. Ponnath, S. G. Rodriguez, and A. D. White, "Towards active sketching with supervised knowledge," arXiv preprint arXiv:2409.13470, 2024. |                                                                                                   |
| [104] M. Alexi, "From prompt-response to goal-directed systems: the evolution of agent-oriented architecture," arXiv preprint arXiv:2602.10479, 2026. |                                                                                                   |
| [105] Google Developer Documentation, "Agent development kit (adk)," Google Open Source, 2025. |                                                                                                   |
| [106] S. Gaire, S. Suyash, S. Mishra, S. Niroula, D. Thakur, and U. Yadav, "Optimization of knowledge, Security and safety in the new context production ecosystem," arXiv preprint arXiv:2512.08290, 2025. |                                                                                                   |
| [107] V. Nagel, P. Suraj, M. Dahl, M. Supriya, C. B. M. D. Ben, and E. T. "Hallucinations: Reassessing the reliability of legal research tools," Journal of Empirical Legal Studies, 2025. Available: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2406.XXXXXX |                                                                                                   |
| [108] U. Yadav, S. Niroula, G. K. Gupta, and B. Yadav, "Exploring secure machine learning through payload injection and fgsm attacks on resnet50," in 2025 IEEE Silicon Valley Cybersecurity Conference (SVCC), 2025. |                                                                                                   |
| [109] L. Grekesh, A. Sobhanian, S. Mishra, C. Endres, T. Holz, and M. Fritz, "Not what you’ve signed up for: Composing real-world LLM-integrated applications with indirect prompt injection," arXiv preprint arXiv:2302.1273, 2023. Available: https://arxiv.org/abs/2302.1273 |                                                                                                   |
| [110] D. Pasquin, et al., "Securing AI agents against prompt injection attacks," arXiv preprint arXiv:2511.15759, 2024. Available: https://arxiv.org/abs/2511.15759 |                                                                                                   |
| [111] OWASP Foundation, "LLMOI 2025 project injection," OWASP Top 10 for Large Language Model Applications, 2025. Available: https://gemini.oas.org/llmoi-prompt-injection |                                                                                                   |
```

### --- Page 0025 ---

```markdown
| Reference                                                                                                           |
|---------------------------------------------------------------------------------------------------------------------|
| [114] S. Cohen, R. Bitton, and B. Nassi, “Here comes the AI worm: Unleashing zero-click worms that target GenAI-powered applications,” arXiv preprint arXiv:2403.02817, 2024. [Online]. Available: https://arxiv.org/abs/2403.02817. |
| [115] L. Zheng, W.-L. Chang, Y. Sheng, S. Hao, Z. Wu, S. Ba, E. Zhuang, Y. Lin, Z. Li, W. Xng, J. E. Gonzalez, L. Stocia, and F. P. Xing, “Judging LLM-as-a-judge with MT-Bench and chatbot arena,” in Advances in Neural Information Processing Systems, vol. 36, 2023. |
| [116] K. Gupta, P. Pand, N. Acharya, A. K. Sing, and S. Niroula, “Lims in disease diagnosis: A comparative study of deepsexr-k1 and 0.3 mini across chronic health conditions,” arXiv preprint arXiv:2503.10486, 2025. [Online]. Available: https://arxiv.org/abs/2503.10486. |
| [117] K. Greshake, S. Abdehahi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “What have you asked for: A comprehensive analysis of novel prompt injection threats to application-integrated large language models,” in Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, 2023. |
| [118] O. Khattab, A. Singh, P. Maheshwari, Z. Zhang, K. Santhaman, V. Vardhaman, S. Hasa, T. Toshi, H. Moazam, H. Müller, M. Zahra, and C. Potos. “SPY: Compiling declarative language model calls into state-of-the-art pipelines,” in International Conference on Learning Representations (ICLR), 2024. |
```


# ArXiv 2310.03714

### --- Page 0001 ---

```markdown
# DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines

Omar Khattab\(^1\), Arnav Singh\(^2\), Paridhi Maheshwari\(^3\), Zhiyuan Zhang\(^1\), Keshav Santhanam\(^4\), Sri Vardhaman\(^6\), Saiful Haq\(^6\), Ashutosh Sharma\(^6\), Thomas T. Joshi\(^7\), Hanna Moazam\(^8\), Heather Miller\(^3,9\), Matei Zaharia\(^2\), Christopher Potts\(^1\)

\(^1\) Stanford University, \(^2\) UC Berkeley, \(^3\) Carnegie Mellon University, \(^4\) Amazon Alexa AI, \(^5\) Dashworks Technologies, Inc., \(^6\) IIT Bombay, \(^7\) Calera Capital, \(^8\) Microsoft, \(^9\) Two Sigma Investments

okhattab@cs.stanford.edu

## ABSTRACT

The ML community is rapidly exploring techniques for prompting language models (LMs) and for stacking them into pipelines that solve complex tasks. Unfortunately, existing LM pipelines are typically implemented using hard-coded “prompt templates”, i.e. lengthy strings discovered via trial and error. Toward a more systematic approach for developing and optimizing LM pipelines, we introduce DSPy, a programming model that abstracts LM pipelines as text transformation graphs, i.e. imperative computation graphs where LMs are invoked through declarative modules. DSPy modules are parameterized, meaning they can learn (by creating and tuning) compositions of prompts how to apply compositions of prompting, finetuning, augmentation, and reasoning techniques. We design a compiler that can optimize any DSPy pipeline to maximize a given metric. We conduct two case studies, showing that succinct DSPy programs can express and optimize sophisticated pipelines that reason about math word problems, tackle multi-hop retrieval, answer complex questions, and control agent loops. Within minutes of compiling, a few lines of DSPy allow GPT-3.5 and 11ama2-13b-chat to self-boost pipelines that outperform standard and few-shot prompting generally by over 25% and 65%, respectively, and pipelines with expert-created demonstrations (by up to 46% and 16–40%, respectively). On top of that, DSPy programs compiled to open and relatively small LMs like 770M-parameter T5 and 11ama2-13b-chat are competitive with approaches that rely on expert-written prompt chains for proprietary GPT-3.5.

DSPy is available at [https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy).

## 1 INTRODUCTION

Language models (LMs) are enabling researchers to build NLP systems at higher levels of abstraction and with lower data requirements than ever before (Bommasani et al., 2021). This is fueling an exploding space of “prompting” techniques—and lightweight finetuning techniques—for adapting LMs to new tasks (Kojima et al., 2022), eliciting systematic reasoning from them (Wei et al., 2022; Wang et al., 2022b), and augmenting them with retrieved sources (Gu et al., 2022; Lazaridou et al., 2022; Khatab et al., 2022) or with tools (Yao et al., 2022; Schick et al., 2023). Most of these techniques are explored in isolation, but interest has been growing in building multi-stage pipelines and agents that decompose complex tasks into more manageable calls to LMs in an effort to improve performance (Qi et al., 2019; Khatab et al., 2022; Karpas et al., 2022; Dohan et al., 2022; Khot et al., 2022; Khatab et al., 2022; Chen et al., 2022; Pourreza & Rafiei, 2022; Shinn et al., 2023).

Unfortunately, LMs are known to be sensitive to how they are prompted for each task, and this is exacerbated in pipelines where multiple LM calls have to interact effectively. As a result, the LM
```

### --- Page 0002 ---

```markdown
# Preprint

calls in existing LM pipelines and in popular developer frameworks are generally implemented using hard-coded ‘prompt templates’, that is, long strings of instructions and demonstrations that are hand crafted through manual trial and error. We argue that this approach, while pervasive, can be brittle and unscalable—conceptually akin to hand-tuning the weights for a classifier. A given string prompt might not generalize to different pipelines or across different LMs, data domains, or even inputs.

Toward a more systematic approach to designing AI pipelines, we introduce the DSPy programming model.¹ DSPy pushes building new LM pipelines away from manipulating free-form strings and closer to programming (composing modular operators to build text transformation graphs) where a compiler automatically generates optimized LM invocation strategies and prompts from a program. We draw inspiration from the consensus that emerged around neural network abstractions (Bengio et al., 2013), where (1) many general-purpose layers can be modularly composed in any complex architecture and (2) the model weights can be trained using optimizers instead of being hand-tuned.

To this end, we propose the DSPy programming model (Sec 3). We first translate string-based prompting techniques, including complex and task-dependent ones like Chain of Thought (Wei et al., 2022) and ReAct (Yao et al., 2022), into declarative modules that carry natural-language typed signatures. DSPy modules are task-adaptive components—akin to neural network layers—that abstract any particular task formulation, like answering a question or summarizing a paper. We then parameterize each module so that it can learn its desired behavior by iteratively bootstrapping useful demonstrations within the pipeline. Inspired by PyTorch abstractions (Paszke et al., 2019), DSPy modules are used via expressive define-by-run computational graphs. Pipelines are expressed by (1) declaring the modules needed and (2) using these modules in any logical control flow (e.g., if statements, for loops, exceptions, etc.) to logically connect the modules.

We then develop the DSPy compiler (Sec 4), which optimizes any DSPy program to improve quality and cost. The compiler simulates versions of the program with optional baselines, and a validation metric. The compiler simulates versions of the program on the inputs and bootstraps example or finetuning small LMs for steps of the pipeline. Optimization in DSPy is highly modular; it is conducted by teleprompters,² which are general-purpose optimization strategies that determine how the modules should learn from data. In this way, the compiler automatically maps the declarative modules to high-quality compositions of prompting, finetuning, reasoning, and augmentation.

Prompting models like DSPy could be assessed among many dimensions, but we focus on the role of expert-crafted prompts in shaping system performance. We are seeking to reduce or even remove their role through DSPy modules (e.g., versions of popular techniques like Chain of Thought) and teleprompters. We report on two expansive case studies: math word problems (GMSK8; Cobbe et al., 2021) and multi-hop question answering (HotPotQA; Yang et al. 2018) with explorations of chain of thought, multi-chain reflection, multi-hop retrieval, retrieval-augmented question answering, and agent loops. Our evaluations are a unique mix of different compiling strategies reflecting how that straightforward DSPy programs outperform systems using hand-crafted prompts, while also allowing our programs to use much smaller and hence more efficient LMs effectively.

Overall, this work proposes the first programming model that translates prompting techniques into parameterized declarative modules and introduces an effective combinator for general optimization strategies (teleprompters) to optimize arbitrary pipelines of these modules. Our main contributions are empirical and algorithmic: with DSPy, we have found that we can implement very short programs that can bootstrap self-improving multi-stage NLP systems using LMs as small as 11lama2-13b-chat and T5-Large (770M parameters). Without hand-crafted prompts and within minutes to tens of minutes of compiling compositions of DSPy modules can raise the quality of simple programs from 33% to 82% (Sec 6) and from 32% to 46% (Sec 7) for GPT-3.5 and, similarly, from 9% to 47% (Sec 6) and from 22% to 41% (Sec 7) for 11lama2-13b-chat.

¹ DSPy is pronounced dee-ess-pie. It’s the second iteration of our earlier Demonstrate–Search–Predict framework (DSP; Khattab et al. 2022). This paper introduces the key concepts in DSPy. For more extensive and up-to-date documentation of the framework, we refer readers to https://github.com/stanfordnlp/dspy.

² We derive the name teleprompters from the notion of abstracting and automating the task of prompting, in particular, such that it happens at a distance, without manual intervention.
```

### --- Page 0003 ---

```markdown
## 2 RELATED WORK

This work is inspired by the role that Torch \cite{Collobert et al., 2002}, Theano \cite{Bergstra et al., 2010; Al-Rfou et al., 2016}, Chainer \cite{Tokui et al., 2015}, and others played in the development in deep learning by providing powerful abstractions. A similar transformation is emerging with higher-level pipelines of LMs, and we are seeking to offer a solid conceptual framework and programming abstractions for what we call foundational model programming. We draw on differentiable programming: mining \cite{Wang et al., 2018} but applied to LMs called rather than neural networks, and borrow syntactic elements from PyTorch \cite{Paszke et al., 2019}.

In-context learning \cite{McCann et al. 2018; Radford et al. 2018; Brown et al. 2020} is a key mechanism for foundation model programming. A growing body of work has revealed that, especially with instruction tuning \cite{Ouyang et al., 2022}, we can elicit sophisticated behavior via prompting \cite{Wei et al., 2022; Wang et al., 2022b; Press et al., 2022; Yao et al., 2022; Khot et al., 2022; Madaan et al., 2023}. Similarly, forms of weak supervision that would normally require task-specific \cite{Khattab et al., 2021a} or hand-built \cite{Ratner et al., 2016; Hancock et al., 2018} heuristics are now done by LMs \cite{Wang et al., 2022b; Zeilnkman et al., 2022; Zhang et al., 2022; Shao et al., 2023}.

In-context learning methods now routinely invoke tools, leading to LM pipelines that use retrieval models \cite{Chen et al., 2017; Lewis et al., 2020; Guu et al., 2022; Lazaridou et al., 2022; Izacard et al., 2022}, multimodal foundation models, and more traditional tools like APIs \cite{Nakano et al., 2021} and calculators. A number of toolkits have been developed to facilitate this, including LangChain \cite{Chase, 2022}, Semantic Kernel \cite{Microsoft}, Llamandlek \cite{Liu, 2022}, and many other retrieval and agent libraries. These toolkits provide pre-packaged chains and agents that connect LMs with numerous accessible tools. However, they suffer from the pervasive prompt engineering challenges we address in DSPy: they express task-specific behavior through hand-written prompt templates (for detailed discussion, see Appendix B).

Researchers are starting to apply resource optimization and RL to find effective prompts, generating for a single logical LM call \cite{Guu et al., 2023; Pryzant et al., 2023; Huang et al., 2022; Yang et al., 2023}. DSPy seeks to generalize this space: it offers a rich framework for optimizing arbitrary pipelines from high-level declarative signatures, by bootstrapping high-quality multi-stage demonstrations with constraints. In this framework, DSPy teleprompts may apply optimization using model selection techniques like cross-validation or, in principle, with sophisticated techniques involving RL and LM feedback \cite{Hu et al., 2023; Zhao et al., 2023a; Shinn et al., 2023} or learned or Bayesian hyperparameter optimization methods \cite{Bergstra et al., 2013; Akiba et al., 2019}.

The present paper seeks to motivate DSPy as a programming model and to report new empirical findings from applying the DSPy compiler. This is inspired by formative work by \cite{Bergstra et al., 2010; 2013; Paszke et al., 2019} and \cite{Wolf et al., 2020}, who support their respective programming models with a mix of benchmark results and some qualitative measures. For the current paper, we focus on showing that DSPy and its compiler allow us to build outstanding LM systems without hand-crafted prompt strings, but instead from truly modular units, and that this opens up for systematically exploring a rich design space at a very high programmatic level of abstraction.

## 3 THE DSPY PROGRAMMING MODEL

We present DSPy, which treats LMs as abstract devices for text generation,³ and optimizes their usage in arbitrary computational graphs. DSPy programs are expressed in Python: each program takes the task input (e.g., a question to answer or a paper to summarize) and returns the output (e.g., an answer or a summary) after a series of steps. DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompts. Signatures abstract the input/output behavior of a module; modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and teleprompts optimize all modules in the pipeline to maximize a metric.

³ We assume access to one or more LMs, which consume a prompt string and return text completions. This may be a promptable LM capable of in-context learning (e.g., GPT-3.5 or Llama2-7b) or a smaller finetunable LM (e.g., T5-base). An LM may be selected as the default; operations will use it unless configured otherwise.
```

### --- Page 0004 ---

```markdown
# 3.1 NATURAL LANGUAGE SIGNATURES CAN ABSTRACT PROMPTING & FINETUNING

Instead of free-form string prompts, DSPy programs use natural language signatures to assign work to the LM. A DSPy signature is natural-language typed declaration of a function: a short declarative speech that tells DSPy what a text transformation needs to do (e.g., “consume questions and return answers”), rather than how a specific LM should be prompted to implement that behavior. More formally, a DSPy signature is a tuple of input fields and output fields (and an optional instruction). A field consists of field name and optional metadata.¹ In typical usage, the roles of fields are inferred by DSPy as a function of field names. For instance, the DSPy compiler will use in-context learning to interpret question differently from answer and will iteratively refine its usage of these fields.

Signatures offer two benefits over prompts: they can be compiled into self-improving and pipeline-adaptive prompts or finetunes. This is primarily done by bootstrapping (Sec 4) useful demonstrating examples for each signature. Additionally, they handle structured formatting and parsing logic to reduce (or, ideally, avoid) brittle string manipulation in user programs.

In practice, DSPy signatures can be expressed with a shorthand notation like `question -> answer`, so that line 1 in the following is a complete DSPy program for a basic question-answering system (with line 2 illustrating usage and line 3 the response when GPT-3.5 is the LM):

```python
1 qa = dspy.Predict("question -> answer")
2 q = qa(question="Where is Guarani spoken?")
3 out: Prediction(answer="Guarani is spoken mainly in South America.")
```

In the shorthand notation, each field’s name indicates the semantic role that the input (or output) field plays in the transformation. DSPy will parse this notation and expand the field names into meaningful instructions for the LM, so that `english.document -> french.translation` would prompt for English to French translation. When needed, DSPy offers more advanced programming interfaces for expressing more intricate constraints on signatures (Appendix).

# 3.2 PARAMETERIZED & TEMPLATED MODULES CAN ABSTRACT PROMPTING TECHNIQUES

In other signatures in programming languages, DSPy signatures simply define an interface and provide type-like hints on the expected behavior. To use a signature, we must declare a module with that signature, like we instantiated a `Predict` module above. A module declaration looks like a function having that signature.

The Predict Module The core module for working with signatures in DSPy is `Predict` (simplified pseudocode in Appendix D.1). Internally, `Predict` stores the supplied signature, an optional LM to use (initially None), but otherwise overrides the default LM for this module, and a list of demonstrations for prompting (initially empty). Like layers in PyTorch, the instantiated module behaves as a callable function: it takes keyword arguments corresponding to the signature input fields (e.g., question), formats a prompt to implement the signature and includes the appropriate demonstrations, calls the LM, and parses the output fields. When `Predict` detects it’s being used in compile mode, it will also internally track input/output traces to assist the telemporter at bootstrapping the demonstrations.

Other Built-in Modules DSPy modules translate prompting techniques into modular functions that support any signature, contrasting with the standard approach of prompting LMs with task-specific details (e.g., hand-written few-shot examples). To this end, DSPy includes a number of more sophisticated modules like ChainOfThought, ProgramOfThought, MultiChainProgram, and React.² These can all be used interchangeably to implement a DSPy signature. For instance, simply chang-

¹“String descriptions of the task and the fields are also optional and usually omitted. Fields can carry optional field prefix and description. By default, fields are assumed to hold free-form strings; we are actively exploring optional data type as a way to specify constraints on valid values (e.g., bool or int) and more gracefully handle formatting and parsing logic, though this feature is not core to DSPy at the time of writing.”

²These modules generalize prompting techniques from the literature, respectively, by Wei et al. (2022), Chen et al. (2022), Yoran et al. (2023), and Yao et al. (2022), and in doing so, generalize the idea on zero-shot prompting and rationale self-generation from Kojima et al. (2022), Zelikman et al. (2022), Zhang et al. (2022), and Huang et al. (2022) to parameterized modules that can bootstrap arbitrary multi-stage pipelines.
```

### --- Page 0005 ---

```markdown
# Preprint

## 3. TELEPROMPTERS CAN AUTOMATE PROMPTING FOR ARBITRARY PIPELINES

When invoking a DSpy program, we generally invoke a teleprompter, which is an optimizer that takes the program, a training set, and a metric—and returns a new optimized program. Different teleprompters (Sec 4) apply different strategies for optimization.

### 3.1 ChainOfThought

```python
class ChainOfThought(dspy.Module):
    def __init__(self, signature):
        # Modify signature from 'inputs -> outputs' to 'inputs -> rationale, outputs'.
        rationale_field = dspy.OutputField(prefix='Reasoning: Let\'s think step by step.')
        signature = dspy.Signature(signature).prepend_output_field(rationale_field)

        # Create a sub-module with the modified signature.
        self.predict = dspy.Predict(signature)

    def forward(self, **kwargs):
        # Just forward the inputs to the sub-module.
        return self.predict(**kwargs)
```

This is a fully-fledged module capable of learning effective few-shot prompting for any LM or task. We contrast that with Appendix C, which copies reasoning prompts hand-written by sources ranging from recent research to popular prompting libraries.

### Parameterization

Uniquely, DSpy parameterizes these prompting techniques. To understand this parameterization, observe that any LM cell seeking to implement a particular signature needs to specify parameters that include: (1) the specific LM to call (Chen et al., 2023), (2) the prompt instructions (Yang et al., 2023) and the string prefix of each signature field and, most importantly, (3) the demonstrations used as few-shot prompts (for frozen LMs) or as training data (for finetuning). We focus primarily on automatically generating and selecting useful demonstrations. In our case studies, we find that bootstrapping good demonstrations gives us a pathway to teach sophisticated pipelines of LMs new behaviors systematically.

### Tools

DSpy programs may use tools, which are modules that execute computation. We support retrieval models through a dspy.Retrieve module. At the time of writing, DSpy has built-in support for CoLBERTv2, Pyserini, and Pinecone retrievers, and we have explored experimental dspy.SQL for executing SQL queries and dspy.PythonInterpreter for executing Python code in a sandbox.

### Programs

DSpy modules can be composed in arbitrary pipelines in a define-by-run interface. Inspired directly by PyTorch and Chainer, one first declares the modules needed at initialization, allowing DSpy to keep track of them for optimization, and then one expresses the pipeline with arbitrary code that calls the modules in a forward method. As a simple illustration, we offer the following simple but complete retrieval-augmented generation (RAG) system.

```python
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        # Retrieve will use the user's default retrieval settings unless overridden.
        self.retrieve = dspy.Retrieve(num_passages)
        self.generate_answer = dspy.ChainOfThought('context, question -> answer')

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)
```
```

### --- Page 0006 ---

```markdown
# Preprint

In DSPy, training sets may be small, potentially a handful of examples, though larger data enables more powerful optimization. Training examples may be incomplete, i.e., only input values are necessary. Labels for the pipeline steps are not required, unless they need to be used in the metric. In practice, we typically assume labels only for (at most) the program's final output, not the intermediate steps. This label-efficiency is critical for modularity: building a new pipeline in DSPy requires simply recompiling the new pipeline's code, not annotating data specific to the new pipeline.

Metrics can be simple notions like exact match (EM) or F1, but they can be entire DSPy programs that balance multiple concerns. For example, we may compile the RAG module against a dataset of question-answer pairs `qa.trainset` and the metric EM. The goal of optimization here is to effectively bootstrap few-shot demonstrations. The following code achieves this:

```python
# Small training set with only questions and final answers.
qa_trainset = [dspy.Example(question="What is the capital of France?", answer="Paris")]

# The teleprompter will bootstrap missing labels: reasoning chains and retrieval contexts.
answer_match = dspy.BootstrapFewShot(metric=dspy.evaluate.answer.exact_match)
compiled_rag = teleprompter.compile(RAG(), trainset=qa_trainset)
```

In this example, the `BootstrapFewShot` teleprompter (Sec 4, Appendix E.1) simulates RAG on the training example(s). It will collect demonstrations of each module (i.e., examples of its input-output behavior) that collectively lead to valid output (i.e., respecting the signatures and the metric).

If one wanted to push the compiled program to be extractive given its retrieved contexts, one could define a custom metric to use in place of `dspy.evaluate.answer.exact_match`.

```python
def answer_and_context_match(example, pred, trace=None):
    answer_match = dspy.evaluate.answer.exact_match(example, pred)
    
    # Is the prediction a substring of some passage?
    context_match = any(pred in c for c in pred.context)
    
    return answer_match and context_match
```

Notice that behavior like this might be more accurately checked by another DSPy program that checks for faithful grounding of answers. Such metrics are fully supported and encouraged in DSPy.

Teleprompters can be composed by specifying a teacher program. DSPy will sample demonstrations from this program for prompt optimization. This composition can enable very rich pipelines, where expressive programs (e.g., complex expensive ensembles using larger LMs) supervise each other (e.g., simple pipelines using smaller LMs). One may start with `compiled_rag` from above (say, compiled to use the large Llama2-13B chat LM) but now fine-tune Flan-T5-large to create an efficient program:

```python
# larger set of questions with no labels. Labels for all steps will be bootstrapped.
unlabeled_questions = [dspy.Example(question="What is the capital of Germany?", ...)]

# As we assume no answer, we use 'answer_passage_match' to filter ungrounded answers.
finetuning_teleprompter = BootstrapFineTune(metric=dspy.evaluate.answer.passage_match)

# We set 'teacher=compiled_rag' to compose. Bootstrapping will now use 'compiled_rag'.
compiled_rag_via_finetuning = finetuning_teleprompter.compile(RAG(), teacher=compiled_rag,
    trainset=unlabeled_questions, target='google/flan-t5-large')
```

# 4 THE DSPY COMPILER

A key source of DSPy’s expressive power is its ability to compile—or automatically optimize—any program in this programming model. Compiling relies on a teleprompter, which is an optimizer for DSPy programs that improves the quality (or cost) of modules via prompting or finetuning, which are unified in DSPy. While DSPy does not enforce this when creating new teleprompters, typical teleprompters go through three stages.

**Stage 1: Candidate Generation** The compiler first (recursively) finds all unique Predict modules (predictors) in a program, including those nested under other modules. For each unique predictor $p$, the teleprompter may generate candidate values for the parameters of $p$: the instruction, field descriptions, or—most importantly—demonstrations (i.e., example input–output pairs). In this iter-
```

### --- Page 0007 ---

```markdown
# Preprint

Consider the simplest non-trivial teleprompter in DSPy, $BootstrapFewShot$ (simplified pseudocode version of the program being compiled) on some training inputs, possibly one or more times with a high temperature. When running in compile mode, multi-stage traces are tracked transparently and in a thread-safe fashion throughout execution. The program's metric is used to filter for multi-stage traces that together help the pipeline pass the metric. We thus obtain potential labels for all signatures in the program by throwing away the bad examples and using the good examples as potential demonstrations, though these design decisions are under user control.

While LMs can be highly unreliable, we find they can be rather efficient at searching the space of solutions for multi-stage designs. A well-decomposed program can typically find at least a few training examples where the LM can pass the constraints enforced by the signatures and metrics, allowing us to bootstrap iteratively if needed.

## Stage 2: Parameter Optimization

Now each parameter has a discrete set of candidates: demonstrations, instructions, etc. Many hyperparameter tuning algorithms (e.g., random search or Tree-structured Parzen Estimators as in HyperOpt (Bergstra et al., 2013) and Optuna (Akiba et al., 2019)) can be applied for selection among candidates. We report simplified implementations of DSPy’s $BootstrapFewShotWithRandomSearch$ and $BootstrapFewShotWithOptuna$ in Appendix E.2 and Appendix E.3.

Another type of optimization is fine-tuning with $BootstrapFewShot$, where the demonstrations are used to update the LM’s weights for each predictor. When this is applied, the LM parameter of each module is updated to the new LM weights. Typically, we are optimizing average quality using the metric with cross-validation over the training set or a validation set. This is applicable even with no prior stages, depending on the nature of the metric.

## Stage 3: Higher-Order Program Optimization

A different type of optimization that the DSPy compiler supports is modifying the control flow of the program. One of the simplest forms of these is ensembles, which we use in the case studies in this work. An ensemble will bootstrap multiple copies of the same program, and then replace the program with a new one that runs them all in parallel and reduces their predictions into one with a custom function (e.g., majority voting). In future work, this stage can easily accommodate techniques for more dynamic (i.e., test-time) bootstrapping as well as automatic backtracking-like logic.

# 5 GOALS OF EVALUATION

Programming frameworks can be evaluated along many dimensions: computational efficiency, developer efficiency, intuitiveness of the code and concepts, and so forth. In this paper, we focus on perhaps the most pressing issue for current LM pipelines: the role of hand-written, task-specific prompts in achieving performant systems. Our evaluations seek to test the following hypotheses:

- **H1** With DSPy, we can replace hand-crafted prompt strings with concise and well-defined modules, without reducing quality or expressive power.

- **H2** Parameterizing the modules and treating prompting as an optimization problem makes DSPy better at adapting to different LMs, and it may outperform expert-written prompts.

- **H3** The resulting modularity makes it possible to more thoroughly explore complex pipelines that have useful performance characteristics or that nuanced metrics.

Our evaluation will explore these hypotheses using diverse task–program pairs. We hope this begins a shift from underspecified questions like "how do different LMs compare on GSM8K?" toward "how they compare on GSM8K with program P when compiled with strategy S", which is a well-defined and reproducible run. Ultimately, our goal is to reduce the role of artificial prompt construction in modern AI in favor of the development of new modular, composable programs and optimizers.
```

### --- Page 0008 ---

```markdown
| Program          | Compilation      | Training | Dev   | Test  | Dev   | Test  |
|------------------|------------------|----------|-------|-------|-------|-------|
|                  |                  |          | GPT-3.5 |       | Llama2-13b-chat |       |
| none             | n/a              | 24.0     | 25.2  | 7.0   | 9.4   |       |
| fewshot          | trainset         | 33.1     | 4.3   |       |       |       |
| vanilla          | bootstrap        | 44.0     | 28.0  |       |       |       |
| bootstrap×2      | trainset         | 64.7     | 61.7  | 37.3  | 36.5  |       |
| +ensemble        | trainset         | 62.7     | 61.9  | 39.0  | 34.6  |       |
|                  |                  |          |       |       |       |       |
| none             | n/a              | 60.0     | 26.7  |       |       |       |
| fewshot          | trainset         | 63.0     | 27.3  |       |       |       |
| CoT              | 78.6             | 72.4    | 43.3  | 33.7  |       |       |
| bootstrap        | trainset         | 80.3     | 72.9  | 43.3  |       |       |
| +ensemble        | trainset         | 88.3     | 81.6  | 43.7  |       |       |
|                  |                  |          |       |       |       |       |
| none             | n/a              | 65.0     | 36.7  |       |       |       |
| reflection       | trainset         | 71.7     | 36.3  |       |       |       |
| bootstrap        | trainset         | 83.0     | 44.3  | 40.2  |       |       |
| +ensemble        | trainset         | 86.7     | 49.0  | 46.9  |       |       |

## 6 CASE STUDY: MATH WORD PROBLEMS

We evaluate on the popular GSM8K dataset with grade school math questions (Cobbe et al., 2021). We sample 200 and 300 question–answer pairs from the official training set for training and development, respectively. Our final evaluations use the 1.3k official test set. We report extensive comparisons on the development set to avoid overfitting on test. Following prior work on GSM8K, we evaluate the accuracy of the final numerical value that appears in the LM output.

### Programs Considered

For this task, we consider three simple DSyp programs: a one-step Predict module (vanilla), a two-step ChainOfThought module (CoT), and finally a multi-step ComparerOfThoughts module (ThoughtReflection). These are fully defined by the following code:

```python
class ThoughtReflection(dspy.Module):
    def __init__(self, num_attempts):
        self.num_attempts = num_attempts
        self.predict = dspy.ChainOfThought("question -> answer", num_attempts)
        self.compare = dspy.MultiChainComparison("question -> answer", num_attempts)

    def forward(self, questions):
        completions = self.predict(questions).completions
        return self.compare(questions, completions=completions)
```

In reflection, five reasoning chains are sampled from the LM (alongside their answers) and they are compared in parallel by a built-in MultiChainComparison module, which generalizes Yoran et al. (2023). This generates a new answer taking into account the patterns from the five attempts. Critically, the modules used are general, none is specific math problems or particular LM.

### Compiling

As discussed in Section 4, DSyp programs can be compiled into new, optimized programs. In our experiments, we evaluate the programs zero-shot (no compiling) as well as a number of strategies for compiling. Our simplest compiler is labeled FewShot:

```python
fewshot = dspy.LabeledFewShot(k=8).compile(program, trainset=trainset)
```

Here, programs can be any DSyp module. This simply samples k=8 random demonstrations from the trainset for the fields common to the training examples and the signature(s), in this case, question and answer, but not the reasoning for instance. We report the average of 3–5 runs (depending on the setting) when applying such random sampling.
```


### --- Page 0009 ---

```markdown
Next, we also consider bootstrapping few-shot examples with random search:

```python
tp = BootstrapShowWithRandomSearch(metric=gsm8k_accuracy)
bootstrap = tp.compile(program, trainset=trainset, valset=devset)
```

This will generate demonstration chains for examples in the training set and optimize the selection of demonstrations (from this set) to self-improve the program’s modules. As the name indicates, this is done with random search, treating the selection of demonstrations as a parameter to optimize.

Next, if desired, this bootstrapping process can be nested in DSPy. In particular, we can use the optimized `bootstrap` program itself to further bootstrap another program. This is relevant, for example, whenever the original zero-shot program performs relatively poorly.

```python
bootstrap2 = tp.compile(program, teacher=bootstrap, trainset=trainset, valset=devset)
```

And lastly, we consider assembling these bootstraps:

```python
# A program that ensembles the top-7 candidate programs from a bootstrapping compiler run
# (in particular `bootstrap*` or, when applicable, `bootstrap2`) with majority voting.
ensemble = Ensemble(decode_fn=dspy.majority).compile(bootstrap.programs[:7])
```

GSM8K includes human reasoning chains. Above, `trainset` does not include these reasoning chains. We also evaluate with `trainset_human_CoT`, which extends the examples in `trainset` with the human reasoning string. These two datasets can be used interchangeably as the value for the `trainset` parameter above. We note here that compiling generally runs on the order of minutes (or tens of minutes) as even the more expensive settings only require running the program a few thousand times (e.g., 10–20 trials over 150–300 validation examples) and they can occur in parallel.

Results

Our results are summarized in Table 1, which includes dev results as well as our evaluation of promising representatives of each approach on the test set. First, the vanilla program results predict the answers directly, that is, without using a reasoning chain first. This is most pronounced in the absence of good demonstrations, which can be seen in the non-completion setting (i.e., zero-shot instruction) and the fewshot setting (i.e., sampling random question-answer pairs). Interestingly, however, vanilla is helped substantially by compiling with bootstrap and by iterating this process into bootstrap2. On inspecting the prompts bootstrapped (Appendix F), we see the prompt allows the LM to leverage the single field for reasoning first, which is permitted as the metric extracts the final numerical value for evaluation.

Next, we consider the CoT program. While the expert has human reasoning chains (`human_CoT`) provide a large boost when available, we can match or surpass this using bootstrap, substantiating our hypotheses that DSPy can cut the need for hand-crafted prompts. Beyond this, we see that the reflection program, while only a few lines longer than the others, is a clear winner, though CoT is quite effective with ensemble. Overall, the bootstrap compilation procedure leads to large gains for every program, across both LMs. Indeed, all programs in this table are expressed by composing two to four DSPy modules and teleprompters, and they reveal overall that—in the new paradigm prescribed by DSPy—it’s composing the right generic modules, rather than manipulating string prompts, that improves different LMs from 4–20% accuracy to 49–88% accuracy.

We can informally compare with the following. Zhang et al. (2022) reports 48% for `text-davinci-002`, which aligns closely with our `llama2-13b-chat` results, and reports 59.4% with codex when employing a manual CoT approach and 62.8% with an automatic CoT method. Wang et al. (2022b) report 57% for CoT prompting with PaLM 540B, which becomes 74% upon adding self-consistency. The Llama2 authors (Touvron et al., 2023) presents 28.7% for `llama2-13b`, 42.2% for `llama2-34b`, and 56.8% for `llama2-70b`. Intriguingly, our program with the 13b variant of the model is competitive with their 34b-based results even though we don’t use human reasoning chains in our program. Zhao et al. (2023b) reports 80.8% for CoT with gpt-3.5-turbo from April 2023. The GPT-4 authors (OpenAI, 2023) report GPT-3.5 scores 57.1% and GPT-4 elevates this to 92% but they note that GPT-4 was in fact pre-trained on a subset of GSM8K’s training set.
```
![Table summarizing results of various programs](assets/page_0009_img_1.png)
```

### --- Page 0010 ---

```markdown
# 7 CASE STUDY: COMPLEX QUESTION ANSWERING

In this case study, we explore the multi-hop question answering task with the HotPotQA (Yang et al., 2018) dataset in the open-domain “fullwiki” setting. For retrieval, we use a search index of the official Wikipedia 2017 “abstracts” dump of HotPotQA. Search is conducted by a CoILBERTv2 (Sthanam et al., 2021) retriever. The HotPotQA test set is hidden, so we reserve the official validation set for our testing, and sample 100 examples for that. We sub-divide the training set into 70%/30% train/validation splits. In the training (and thus validation) split, we keep only examples marked as “hard” in the original dataset, which matches the designation of the official validation and test sets. For training and for reporting development results, we sample 200 and 300 examples respectively.

## Programs Considered

Our simplest baseline is the vanilla program used in the previous case study on GSM8K (See 6); the “question → answer” signature is universal enough that it will work for this task (and many others) when compiled appropriately.

Our baseline RAG program is the one given in Section 3.2 as a simple example of RAG with a dspy.ChainOfThought layer. We will see that this program does not excel at HotPotQA, and this motivates us to evaluate two multi-hop programs.

To that end, we first test ReAct (Yao et al., 2022), a multi-step agent for tool use, which is implemented as a built-in module in DSpy. In the simplest case, a ReAct module for a particular signature can be declared as follows in DSpy:

```python
react = dspy.ReAct("<question -> answer>", tools=[dspy.Retrieve(k=1), max_tries=5])
```

We also test the following custom program, which simulates the information flow in Baleen (Khattab et al., 2012a) and IRRR (Qi et al., 2020) and has similarities to IRCoT (Trivedi et al., 2022).

```python
class BasicMultiHop(dspy.Module):
    def __init__(self, passages_per_hop):
        self.init_fields(passages_per_hop)
        self.generate_query = dspy.ChainOfThought(context, question -> search_query)
        self.generate_answer = dspy.ChainOfThought(context, question -> answer)
    
    def forward(self, question):
        context = []
        for h in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages
        
        return self.generate_answer(context=context, question=question)
```

## Compiling

For compiling programs, we continue to use the ones that we used for GSM8K (See 6). We also consider two compositions of our teleporters. For ReAct, we covered bootstrapping with BootstrapFewShotWithRandomSearch starting from an earlier bootstrap of the React program. For the simple multihop program, we also consider fine-tuning with T5-Large starting from the earlier bootstrap of that program.

```python
multihop_t5 = dspy.BootstrapFineTune(metric=answer_exact_match).compile(program, teacher=bootstrap, train=instraints, target='t5-large')
```

## Results

Table 2 summarizes our results. Compared with the vanilla few-shot prompting, a chain-of-thought and retrieval-augmented generation (CoRAG) program can self-bootstrap in DSpy to increase answer EM substantially. However, this relies entirely on the CoILBERTv2 retriever to find relevant passages directly from the original questions, limiting its passage recall. This is tackled in the react and multihop programs, which will generate queries for the retriever in multiple iterative “hops”. Indeed, overall, a simple multihop program performs the best, and in general bootstrap again proves to be very effective at raising its quality relative to its fewshot variant for both LMs.

In particular, we can see that bootstrapping (and/or bootstrapx2) can outperform both fewshot prompting (for multihop) and expert human reasoning (for react; adapted slightly from Yao et al. (2022) to our retrieval setting). Perhaps most importantly, we can make 11lama2-13b-capable inline with GPT-3.5 by simply compiling our programs.

To assess the finetuning capacity of DSpy, we also evaluated the compiler multihop_t5 defined above which produced a T5-Large (770M parameter) model. This program scores 39.3% answer EM and 46.0% passage accuracy on the dev set, using only 200 labeled inputs and 800 unlabeled.
```
![Detailed description of the chart](assets/page_0010_img_1.png)
```

### --- Page 0011 ---

```markdown
| Program      | Compiler  | Dev         | Test        | Dev         | Test        | Dev         | Test        |
|--------------|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
|              |           | Ans         | Psg         | Ans         | Psg         | Ans         | Psg         |
| vanilla      | fewshot  | 34.3        | n/a         | 31.5       | n/a         | 27.5       | n/a         | 21.8       |
| CoT_RAG      | fewshot  | 36.4        | 36.0        | 29.8       | 34.4       | 34.5       | 36.0       | 28.0       | 34.4       |
|              | bootstrap | 42.3        | 36.0        | n/a        | 38.3       | 36.0       | 32.9       | 34.4       |
| react        | +human.r  | 33.0        | n/a         | 28.3       | n/a         | 24.7       | n/a         |
|              | bootstrap | 31.0        | n/a         | 24.7       | n/a         | 24.7       | n/a         |
|              | bootstrap2 | 39.0       | n/a         | 40.0       | n/a         | 40.0       | n/a         |
| multihop     | fewshot  | 36.9        | 38.3        | 31.2       | 32.0       | 34.7       | 32.0       | 31.3       | 30.8       |
|              | bootstrap | 48.7        | 47.0        | 43.8       | 42.0       | 48.3       | 36.4       | 43.5       |
|              | ensemble  | 54.7        | 45.6        | 50.0       | 41.0       | n/a        | n/a         |

![Results with in-context learning on HotPotQA multi-hop retrieval question answering](assets/page_0011_img_1.png)

8 CONCLUSION  
This paper introduced DSPy, a new programming model for designing AI systems using pipelines of pretrained LMs and other tools. We presented three new concepts introduced in this abstraction (DSPy signatures, modules, and teleporters), and showed in two very different case studies that it supports rapid development of highly effective systems that use relatively small LMs. We have maintained open-source versions of this framework for close to a year. In this period, we have seen and created a large number of programs that were compiled to high-quality systems by DSPy, spanning tasks from information extraction to low-resource synthetic data generation. In the interest of space and to maintain reasonable scope in this paper, we leave reporting on such tasks under controlled experimental conditions to future work. While in-context learning has proved transformative over the past 2–3 years of LM research, we argue that the true expressive power in this emerging paradigm is in building sophisticated text transformation graphs in which composable modules and optimizers (teleporters) come together to leverage LMs in more systematic and reliable ways.

## ACKNOWLEDGMENTS  
We thank Josh Purtell for suggesting the apt name “text transformation graph” for the computational graph abstraction of DSPy. We thank Rick Battle, Igor Kotenko, Lisa Li, David Hall, Ashwin Paranjape, Chris Manning, Percy Liang, and many researchers, developers, and users for valuable feedback.
```

### --- Page 0012 ---

```markdown
discussions and feedback. We thank Giuseppe Attanasio for his public LaTeX GitHub-style Python code formatting gist.6

This work was partially supported by IBM as a founding member of the Stanford Institute for Human-Centered Artificial Intelligence (HAI), Oracle, Virtusa, and Cigna Healthcare. It was also partially supported by an HAI Azure compute grant. This research was supported in part by affiliate members and other supporters of the Stanford DAWN project—Facebook, Google, and VMware—as well as the NSF under CAREER grant CNS-1651570. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. Omar Khattab is supported by the Apple Scholars in AI/ML fellowship.

\usepackage[pdftex]{graphicx} ...
\includegraphics[width=0.81\linewidth]{myfile.pdf}

## References

| Author(s) | Title | Source |
|-----------|-------|--------|
| Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. | A next-generation hyperparameter optimization framework. | In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2623–2631, 2019. |
| Rami Al-Rfou, Guillaume Alain, Amjad Almahairi, Christof Angermueller, Dzmitry Bahdanau, Nicolas Ballas, Frédéric Bastien, Justin Bayer, Anatoly Belikov, Alexander Belopolsky, et al. | Theano: A Python framework for fast computation of mathematical expressions. | arXiv e-prints, pp. arXiv:1605. |
| James Bergstra, Olivier Breuleux, Frédéric Bastien, Pascal Lamblin, Razvan Pascanu, Guillaume Desjardins, Joshua T. L. David Warde-Farley, and Yoshua Bengio. | Theano: A CPU and GPU Python compiler. | In Proc. 9th python in science conf, volume 1, pp. 3–10, 2010. |
| James Bergstra, Frédéric Bastien, Olivier Breuleux, Pascal Lamblin, Razvan Pascanu, Olivier Delalleau, Guillaume Desjardins, David Warde-Farley, Ian Goodfellow, Arnaud Bergeron, et al. | Theano: Deep learning on gpus with Python. | In NIPS 2011, BigLearning Workshop, Granada, Spain, volume 3, 2011. |
| James Bergstra, Daniel Yamins, and David Cox. | Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. | In International conference on machine learning, pp. 115–123. PMLR, 2013. |
| Rishi Bommasani, Drew A Hudson, Ethan Adelii, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bock, Antoine Bosselut, Emma Brunskill, et al. | On the opportunities and risks of foundation models. | arXiv preprint arXiv:2108.07258, 2021. |
| Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. | Language models are few-shot learners. | Advances in neural information processing systems, 33:1877–1901, 2020. |
| Harrison Chase. | hwchase17/langchain. | 2022. URL https://github.com/hwchase17/langchain. |
| Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. | Reading Wikipedia to answer open-domain questions. | In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1870–1879, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1171. URL https://aclanthology.org/P17-1171. |
| Lingjiao Chen, Matei Zaharia, and James Zou. | Frugalgpt: How to use large language models while reducing cost and improving performance. | arXiv preprint arXiv:2305.05176, 2023. |

6 https://gist.github.com/g8a9/07c2be12ae2cfad4aa430d7dc946bc
```

### --- Page 0013 ---

```markdown
# Preprint

| Authors                                                                 | Title                                                                                                   | Source                                                                                     |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen.             | Program of thoughts prompt: Disentangling computation from reasoning for numerical reasoning tasks.     | arXiv preprint arXiv:2211.12588, 2022.                                                   |
| Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma,       | Scaling language modeling with pathways.                                                                | arXiv preprint arXiv:2204.02311, 2022.                                                  |
| Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian   |                                                                                                         |                                                                                            |
| Gehrmann, et al.                                                         |                                                                                                         |                                                                                            |
| Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, | Training verifiers to solve math word problems.                                                        | arXiv preprint arXiv:2110.14168, 2021.                                                  |
| Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton,          |                                                                                                         |                                                                                            |
| Reichiro Nakano, et al.                                                 |                                                                                                         |                                                                                            |
| Ronan Collobert, Samy Bengio, and Johnny Mariéthoz.                    | Torch: a modular machine learning software library.                                                    | Technical report, Idiap, 2002.                                                           |
| David Dohan, Winnie Xu, Aitor Lewkowycz, Jacob Austin, David Bieber,   | Language model cascade.                                                                                 | arXiv preprint arXiv:2207.10342, 2022.                                                  |
| Raphael Gontijo Lopes, Yuhai Wu, Henryk Michalewski, Rif A Saurous,    |                                                                                                         |                                                                                            |
| Jascha Sohl-Dickstein, et al.                                           |                                                                                                         |                                                                                            |
| Luyu Gao, Zhuyin Dai, Panunpop Pasupat, Anthony Chen, Arun Tejasvi     | Rarr: Researching and revising what language models say, using language models.                        | In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16477–16508, 2023a. |
| Chaganty, Yicheng Fan, Vince Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, |                                                                                                         |                                                                                            |
| et al.                                                                   |                                                                                                         |                                                                                            |
| Luyu Gao, Aman Madan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming       | Program-aided language models.                                                                          | In International Conference on Machine Learning, pp. 10764–10799. PMLR, 2023b.          |
| Callan, and Graham Neubig.                                              |                                                                                                         |                                                                                            |
| Qingyang Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan,     | Language model optimization.                                                                            | arXiv preprint arXiv:2309.08532, 2023.                                                  |
| Guoqing Liu, Jiang Bian, and Yujing Yang.                              |                                                                                                         |                                                                                            |
| Kelvin Guo, Kenton Lee, Zora Tung, Panunpop Pasupat, and Ming-Wei     | Retrieval-augmented language model pre-training.                                                       | arXiv preprint arXiv:2002.08909, 2020. URL: https://arxiv.org/abs/2002.08909.           |
| Chuang.                                                                  |                                                                                                         |                                                                                            |
| Braden Hancock, Paroma Varma, Stephanie Wang, Martin Bringmann, Percy   | Reinforcing interactions with natural language explanations.                                            | In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1884–1895. Association for Computational Linguistics, 2018. URL: http://aclweb.org/anthology/P18-1175. |
| Liang, and Christopher Ré.                                             |                                                                                                         |                                                                                            |
| Bin Hu, Chengyang Zhao, Pu Zhang, Zihao Zhou, Yuanhang Yang, Zenglin  | A reinforcement learning approach.                                                                       | arXiv preprint arXiv:2306.03604, 2023. URL: https://arxiv.org/abs/2306.03604.           |
| Xu, and Bin Liu.                                                        |                                                                                                         |                                                                                            |
| Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang,       | Large language models can self-improve.                                                                | arXiv preprint arXiv:2210.11610, 2022.                                                  |
| Hongkun Yu, and Jiawei Han.                                            |                                                                                                         |                                                                                            |
| Gautier Izard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio      | Few-shot learning with retrieval augmented language models.                                             | arXiv preprint arXiv:2205.03299, 2022.                                                  |
| Petroni, Timo Schick, Jane Diwedi-U, Armand Joulin, Sebastian Riedel,  |                                                                                                         |                                                                                            |
| and Edouard Grave.                                                      |                                                                                                         |                                                                                            |
| Ehud Karpas, Omri Abed, Yonatan Belinkov, Barak Lenz, Opher Lieber,    | Mrkl systems: A modular, neuro-symbolic architecture that combines large language models, external     | arXiv preprint arXiv:2205.00445, 2022.                                                  |
| Nir Ratner, Yoav Shoham, Hofit Batz, Yao Levine, Kevin Leyton-Brown,   | knowledge sources and discrete reasoning.                                                               |                                                                                            |
| et al.                                                                   |                                                                                                         |                                                                                            |
| Omar Khattab, Christopher Potts, and Matei Zaharia.                    | Balenet: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval.                                  | In Thirty-Fifth Conference on Neural Information Processing Systems, 2021a.              |
| Omar Khattab, Christopher Potts, and Matei Zaharia.                    | Relevance-guided supervision for openqa with CoILBERT.                                               | Transactions of the Association for Computational Linguistics, 9:929–944, 2021b.        |
```

### --- Page 0014 ---

```markdown
# Preprint

Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp. arXiv preprint arXiv:2212.14024, 2022.

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. arXiv preprint arXiv:2210.02406, 2022.

Takeshi Kojima, Shixing Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot responders. arXiv preprint arXiv:2205.11916, 2022.

Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. Internet-augmented language models through few-shot prompting for open-domain question answering. arXiv preprint arXiv:2203.01515, 2022.

Patrick Lewis, Ethan Perez, Aleksandra Pikus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäshel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 9459–9474. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230265780e1bc26945df74815e-Paper.pdf.

Jerry Liu. LlamaIndex, 2022. URL https://github.com/jerryjliu/llama.index.

Arman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Diri, Shriram Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.

Bryan McCann, Nitsuh Sachdeva, Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering. arXiv:1806.08730, 2018. URL https://arxiv.org/abs/1806.08730.

Microsoft. Semantic kernel. 2023. URL https://learn.microsoft.com/semantic-kernel/.

Reiichiro Nakano, Jacob Hilton, Sushir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantan Jain, Vincent Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin B. Tso, Matthew Knight, Benjamin Chess, and John Schulman. WebGPT: Browser-assisted question-answering with human feedback, 2021. URL https://arxiv.org/abs/2112.09332.

OpenAI. GPT-4 technical report, 2023.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alhan Desmaison, Andreas Koff, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chintala, Benoît Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/file/bd1b0d28fef79f2f8af9f012727740e-Paper.pdf.

Mohammadreza Pourreza and Davood Rafiei. Dini-sql: Decomposed in-context learning of text-to-sql with self-correction. arXiv preprint arXiv:2304.11015, 2023.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350, 2022.
```

### --- Page 0015 ---

```markdown
# Preprint

Reid Pryzant, Dan Iter, Jerry Li, Yi-Tat Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with “gradient descent” and beam search. arXiv preprint arXiv:2305.03495, 2023.

Peng Qi, Xiaowen Lin, Leo Mehr, Zijian Wang, and Christopher D. Manning. Answering complex open-domain questions through iterative query generation. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)*, pp. 2590–2602, Hong Kong, China, 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1261. [https://aclanthology.org/D19-1261](https://aclanthology.org/D19-1261).

Peng Qi, Haejun Lee, Oghenetegiri Sido, Christopher D Manning, et al. Retrieve, rerank, read, then iterate: Answering open-domain questions of arbitrary complexity from text. arXiv preprint arXiv:2010.12527, 2020. URL: [https://arxiv.org/abs/2010.12527](https://arxiv.org/abs/2010.12527).

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. MS, OpenAI, 2018. URL: [https://openai.com/blog/language-unsupervised/](https://openai.com/blog/language-unsupervised/).

Alexander J Ratner, Christopher M De Sa, Sen Wu, Daniel Selsam, and Christopher Ré. Data programming: Creating large training sets quickly. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (eds.), *Advances in Neural Information Processing Systems 29*, pp. 3567–3575. Curran Associates, Inc., 2016. URL: [https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly).

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. arXiv preprint arXiv:2112.01048, 2021.

Timo Schick, Jane Diwidjono Yu, Roberto Deis, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicolas Cadene, and Thomas Wolf. Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

Zhihong Shao, Yewong Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Syntactic prompting: Generating chain-of-thought demonstrations for large language models. arXiv preprint arXiv:2302.00618, 2023.

Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint arXiv:2303.11366, 2023.

Chenglei Si, Zhe Gan, Zhengyuan Yang, Shuhong Wang, Jianfeng Wang, Jordan Boyd-Graber, and Lijuan Wang. Prompting gpt-3 to be reliable. arXiv preprint arXiv:2210.09150, 2022.

Zhiqun Song, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou. Recitation-augmented language models. arXiv preprint arXiv:2210.01296, 2022.

Seiya Toku, Kenta Ono, Shohei Hido, and Justin Clayton. Chainer: a next-generation open source framework for deep learning. In *Proceedings of workshop on machine learning systems (LearnSys)* in the twenty-ninth annual conference on neural information processing systems (NIPS), volume 5, pp. 1–6, 2015.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Baeza, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned models. arXiv preprint arXiv:2307.09288, 2023.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving trivial with chain-of-thought reasoning for knowledge-intensive multi-step questions. arXiv preprint arXiv:2212.10509, 2022.

Fei Wang, James Decker, Xilun Wu, Gregory Essertel, and Tiark Rompf. Backpropagation with callbacks: Foundations for efficient and expressive differentiable programming. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL: [https://proceedings.neurips.cc/paper_files/paper/2018/file/34e15776f613db29983134a7933-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2018/file/34e15776f613db29983134a7933-Paper.pdf).
```

### --- Page 0016 ---

```markdown
# Preprint

| Authors                                                                 | Title                                                                                                   | Reference                                      |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------|
| Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. | Rationale-augmented ensembles in language models.                                                      | arXiv preprint arXiv:2207.00477, 2022.       |
| Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. | Self-consistency improves chain of thought reasoning in language models.                               | arXiv preprint arXiv:2203.11171, 2022b.      |
| Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. | Chain of thought prompting elicits reasoning in large language models.                                 | arXiv preprint arXiv:2201.11903, 2022.       |
| Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Pino, Canwen Xu, Teven Le Scao, Sylvain Gugger, Marianna Drame, Quentin Lhoest, and Alexander Rush. | Transformers: State-of-the-art natural language processing.                                          | In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, Online, 2020. Assoc. for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-demos.6. URL: https://aclanthology.org/2020.emnlp-demos.6. |
| Chenguang Yang, Xuezhi Wang, Yifeng Liu, Hanxiao Liu, Quoc Le, Denny Zhou, and Xinyun Chen. | Large language models as optimizers.                                                                    | arXiv preprint arXiv:2309.03409, 2023.       |
| Zhilin Yang, Peng Qi, Shizhe Diao, Yizhe Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. | Hotpotqa: A dataset for diverse, explainable multi-hop question answering.                             | arXiv preprint arXiv:1809.09600, 2018.       |
| Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. | 2022.                                                                                                   | 2022.                                         |
| Ori Yoran, Tomer Wolfson, Ben Bogin, Uri Katz, Daniel Deutch, and Jonathan Berant. | Answering questions by meta-reasoning over multiple chains of thought.                                 | arXiv preprint arXiv:2304.13007, 2023.       |
| Eric Zelikman, Yuhuai Wu, and Noah D Goodman.                           | Star: Bootstrapping reasoning with reasoning.                                                          | arXiv preprint arXiv:2203.14665, 2022.       |
| Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola.                   | Automatic chain of thought prompting in large language models.                                         | arXiv preprint arXiv:2210.03493, 2022.       |
| Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. | ExpeL: LLM agents are experiential learners.                                                            | arXiv preprint arXiv:2308.10144, 2022a. URL: https://arxiv.org/pdf/2308.10144. |
| Xu Zhao, Yuxi Xie, Kenji Kawaguchi, Junxian He, and Qizhe Xie.         | Automatic model selection with large language models for reasoning.                                     | arXiv preprint arXiv:2305.14333, 2023b.      |
```

### --- Page 0017 ---

```markdown
# PAGE_NAME: page_0017

## A  ADVANCED SIGNATURES

When more control is desired, one can express signatures as Python classes to provide explicit instructions of the transformation and describe the format or role of each field more directly. For instance, the following signature generates search queries using context and an optional question:

```python
class GenerateSearchQuery(dsp.signature):
    """Write a simple search query that will help answer a complex question."""
    context = dsp.InputField(desc="may contain relevant facts")
    question = dsp.InputField()
    query = dsp.OutputField(dtype=dsp.SearchQuery)
```

Using the above, we can specify a complete system for the generation of a synthetic IR dataset where the queries are mediated by a query generated by the LM:

```python
query_gen = dsp.Predict(GenerateSearchQuery)
query_gen(context="Language typology")
# Out: Prediction(question="What are the main types of language classification?",
# query="language classification" OR "language typology" -wikipedia)
```

If questions are available, they can be supplied as shown: `query_gen(context="Language typology", question="What are the primary language families of South America?")`.

As a work in progress feature, users can optionally specify the type of output fields as `bool`, `int`, `float`, `list`, or `dict` instead of the default free-form string type, as in `contexts, question -> answer_found: bool`.

## B  COMPARISON WITH EXISTING LIBRARIES LIKE LANGCHAIN AND LLAMAINDEX

LangChain and LlamaIndex are perhaps the most popular library in the general space of prompting LMs. These libraries have a different focus compared to DSPy and their suffer internally from the prompt engineering challenges that DSPy aims to resolve. In particular, whereas the goal of DSPy is to tackle the fundamental challenges of prompt engineering for building new LM computational agents, LangChain and LlamaIndex generally help application developers who need pre-packaged components and chains, e.g., implementations of popular and reusable pipelines (e.g., particular agents and specific retrieval pipelines) and tools (e.g., connections to various databases and implementations of long- and short-term memory for agents).

These off-the-shelf higher-level abstractions contrast with DSPy’s focus on introducing core composable operators. In particular, DSPy introduces signatures (to abstract prompts), modules (to abstract prompting techniques), and teleprompters to act as optimizers for arbitrary imperative code (DSPY programs) that chain modules together. Its goal is to help researchers and practitioners build new LM pipelines quickly and achieve very high quality through automatic compilation (self-improvement) instead of manual prompt engineering.

In contrast, typical existing research implementations and existing libraries like LangChain and LlamaIndex are implemented using manual prompt engineering, which is the key problem that DSPy tackles. We conducted an informal study to highlight this. In late September 2023, we found that the LangChain codebase contains 50 strings exceeding 1000 characters, which are generally prompts, compared to none at all in DSPy. Indeed, a substantial number of LangChain’s Python files are singularly dedicated to task-related templating and prompt engineering with 12 prompts.py files and 42 prompt.py files. DSPy, on the other hand, provides a structured framework that automatically bootstraps prompts. The library itself does not contain a single hand-written prompt demonstration for any task at the time of writing, despite the very high quality with various LMs.

To review the typical forms of prompt engineering in existing libraries, we consider the following in LangChain. The LangChain Program-Aided Language Model Gao et al. (2023a) chain programming is a few-shot learning, leveraging a template that is 3982 characters long with 8 math word problems (Prompt 2) and corresponding output programs as learning examples for the language model. LangChain also contains a prompt for SQL query tasks for each of the databases like Oracle, GoogleSQL, DuckDB, Crate, and MySQL, with the average length of these prompts at 1058 characters. Other task areas such as QA with sources (Prompt B) and Graph.QA also have signific...
```

### --- Page 0018 ---

```markdown
## C SAMPLE LARGE PROMPTS

This section highlights a few popular existing frameworks that structure prompt engineering templates. The primary objective is to capture how many words and characters are used for such large multi-line prompts defined for tasks or tools and present these example prompts retrieved from open-sourced papers and repositories. The formatting of these example prompts is adapted from Gao et al. (2023a).

| Task/Tool Prompt                     | Source                        | Words | Characters |
|--------------------------------------|-------------------------------|-------|------------|
| Prompt 1: Text-evidence checker      | Gao et al. (2023a)           | 818   | 4964       |
| Prompt 2: Math word problems (PAL)   | LangChain & Gao et al. (2023b) | 566   | 3957       |
| Prompt 3: ReAct                     | Yao et al. (2022)            | 593   | 3889       |
| Prompt 4: Zero-shot ReAct           | LangChain                    | 101   | 600        |
| Prompt 5: QA with sources           | LangChain                    | 992   | 6197       |
| Prompt 6: SQL MyScale querying      | LangChain                    | 343   | 2239       |
| Prompt 7: Relevant docs retrieval    | LlamaIndex                   | 129   | 719        |
| Prompt 8: IRS chatbot                | LlamaIndex                   | 389   | 2258       |
```


### --- Page 0019 ---

```markdown
| **Line** | **Text**                                                                                                                                                                                                 |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1        | [web] I will check some things you said.                                                                                                                                                                |
| 2        | (Y) You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It's called the nasal cycle.                |
| 3        | I checked: How often do your nostrils switch?                                                                                                                                                           |
| 4        | I found this article: Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow. While the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small article study published in the journal PLOS One. |
| 5        | Your nose's switching time is about every 2 hours, not 45 minutes.                                                                                                                                     |
| 6        | This disagrees with what you said.                                                                                                                                                                      |
| 7        | (Y) You said: The Little House books were written by Laura Ingalls Wilder. The books were published by HarperCollins.                                                                                  |
| 8        | I checked: Who published the Little House books?                                                                                                                                                       |
| 9        | I found this article: These are the books that started it all – the stories that captured the hearts and imaginations of children and young adults worldwide. Written by Laura Ingalls Wilder and published by HarperCollins, these beloved books remain a favorite to this day. |
| 10       | The Little House books were published by HarperCollins.                                                                                                                                                 |
| 11       | This agrees with what you said.                                                                                                                                                                         |
| 12       | (Y) You said: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford's psychology building.                                                                                |
| 13       | I checked: Where was Stanford Prison Experiment conducted?                                                                                                                                              |
| 14       | I found this article: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment. |
| 15       | The Stanford Prison Experiment was conducted in Jordan Hall.                                                                                                                                           |
| 16       | This agrees with what you said.                                                                                                                                                                         |
| 17       | (Y) You said: Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual discipline that has its roots in the 1880s.                                   |
| 18       | I checked: When did social work have its roots?                                                                                                                                                        |
| 19       | I found this article: The Emergence and Growth of the Social Work Profession Social work's roots were planted in the 1880s, when charity organization societies (COS) were created to organize municipal voluntary relief associations and settlement houses were established. |
| 20       | Social work has its roots in the 1880s, not 1880s.                                                                                                                                                     |
| 21       | This disagrees with what you said.                                                                                                                                                                      |
| 22       | (Y) You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list.                                                                        |
| 23       | I checked: What is the Havel-Hakimi algorithm?                                                                                                                                                         |
| 24       | I found this article: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that no such graph exists. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962). |
| 25       | Havel-Hakimi algorithm is for constructing a special solution if a simple graph for the given degree sequence exists, or proving that one cannot find a positive answer, nor converting the adjacency matrix of a graph into its adjacency list. |
| 26       | This disagrees with what you said.                                                                                                                                                                      |
| 27       | (Y) You said: "My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.                                   |
| 28       | I checked: Who was the producer of "(I've Had) The Time of My Life"?                                                                                                                                  |
| 29       | I found this article: On September 8, 2016, the original demo of this song, along with a remix by producer Michael Lloyd, was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Research Foundation at Stanford University. |
| 30       | "The Time of My Life" was produced by Michael Lloyd.                                                                                                                                                   |
| 31       | This agrees with what you said.                                                                                                                                                                         |
| 32       | (Y) You said: Kelvin Hopkins was suspended from the Labor Party because he had allegedly sexually harassed and behaved inappropriately towards a Labour Party activist, Ava Etemad.                       |
| 33       | I found this article: A former Labour MP has left the party before an inquiry into sexual harassment allegations against him was able to be concluded, the party has confirmed. Kelvin Hopkins was accused in 2017 of inappropriately physical contact and was suspended by the Labour party pending an investigation. This agrees with what you said. |
| 34       | Kelvin Hopkins was suspended because he had allegedly sexually harassed and behaved inappropriately towards a Labour Party activist, Ava Etemad.                                                         |
| 35       | This agrees with what you said.                                                                                                                                                                         |
| 36       | (Y) You said: The battles of Lexington and Concord, the British side was led by General Thomas Smith.                                                                                                   |
| 37       | I checked: Who led the British side in the battle of Lexington and Concord?                                                                                                                            |
| 38       | I found this article: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Smith. There were two British regiments.                                                                 |
| 39       | The British side was led by Lieutenant Colonel Francis Smith, not General Thomas Hall.                                                                                                                 |
| 40       | This disagrees with what you said.                                                                                                                                                                      |
| 41       | (Y) You said: (text)                                                                                                                                                                                   |
| 42       | I found this article: (query)                                                                                                                                                                          |
| 43       | I found this article: (evidence)                                                                                                                                                                        |
| 44       |                                                                                                                                                                                                          |
| 45       |                                                                                                                                                                                                          |
| 46       |                                                                                                                                                                                                          |
| 47       |                                                                                                                                                                                                          |
| 48       |                                                                                                                                                                                                          |
| 49       |                                                                                                                                                                                                          |
| 50       |                                                                                                                                                                                                          |
| 51       |                                                                                                                                                                                                          |
| 52       | Figure 1: Example few-shot prompt using a reasoning chain for agreement model that identifies inconsistencies between text and evidence (Gao et al., 2023a).                                              |
```

### --- Page 0020 ---

```markdown
# Preprint

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

```python
def solution():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    return money_left
```

Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?

```python
def solution():
    """Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    return golf_balls_left
```

Q: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?

```python
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    return computers_total
```

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

```python
def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    dad_toys = 2
    mom_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    return total_toys
```

Q: Jason had 20 lollipops. We gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

```python
def solution():
    """Jason had 20 lollipops. We gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_initial = 20
    jason_current = 12
    lollipops_given = jason_initial - jason_current
    return lollipops_given
```
```

### --- Page 0021 ---

```markdown
# Preprint

## Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

```python
def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    return denny_lollipops
```

## Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

```python
def solution():
    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    return chocolates_left
```

## Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

```python
def solution():
    """If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    return total_cars
```

## Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

```python
def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    return trees_added
```

## Q: {question}

```python
# solution in Python:
```

![PAL example few-shot prompt for solving math questions by generating code.](assets/page_0021_img_1.png)
```

### --- Page 0022 ---

```markdown
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:

1. **SearchEntity**: which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
2. **LookupKeyword**: which returns the next sentence containing keyword in the current passage.
3. **FinishTask**: which returns the answer and finishes the task.

Here are some examples.

1. **Question**: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?  
   **Action**: SearchColorado orogeny  
   **Observation 1**: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.  
   **Action 2**: LookupEastern sector  
   **Action 3**: High Plains (United States)  
   **Observation 3**: High Plains refers to one of two distinct land regions:  
   **Action 4**: SearchHigh Plains (United States)  
   **Observation 4**: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).  
   **Action 5**: Finish(1,800 to 7,000 ft)

2. **Question**: Musician and satirist Allee Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after?  
   **Action**: SearchMilhouse  
   **Observation 1**: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.  
   **Action 2**: LookupNamed after  
   **Action 3**: FinishRichard Nixon  
   **Question**: Which character is about Finnish rock groups, Adam Clayton Powell or the Saimasa Gesture?  
   **Action**: SearchAdam Clayton Powell  
   **Observation 1**: Could not find (Adam Clayton Powell, III). Similar: “(Adam Clayton Powell III”, “Seventh Avenue (Manhattan)”, “Adam Clayton Powell Jr., State Office Building”, “Isabel Washington Powell”, “Adam Powell”, “Adam Clayton Powell (film)”.  
   **Action 2**: SearchAdam Clayton Powell (III)  
   **Observation 2**: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr. It was later aired as part of the PBS series The American Experience.  
   **Action 3**: FinishThe Saimasa Gesture  
   **Question**: What profession does Nicholas Ray and Elia Kazan have in common?  
   **Action**: SearchNicholas Ray  
   **Observation 1**: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.  
   **Action 2**: SearchElia Kazan  
   **Observation 2**: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.  
   **Action 3**: FinishDirector, screenwriter, actor  
   **Question**: Which magazine was started first Arthur's Magazine or First for Women?  
   **Action**: SearchArthur's Magazine  
   **Observation 1**: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.  
   **Action 2**: SearchFirst for Women  
   **Observation 2**: First for Women is a woman’s magazine published by Bauer Media Group in the U.S. The magazine was started in 1989.  
   **Action 3**: FinishArthur's Magazine  
   **Question**: Were Pavel Urysohn and Leonid Levin known for the same type of work?  
   **Action**: SearchPavel Urysohn  
   **Observation 1**: Pavel Samuilovich Urysohn (February 3, 1897 – August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.  
   **Action 2**: SearchLeonid Levin  
   **Observation 2**: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.  
   **Action 3**: FinishYes

Figure 3: ReAct example prompt for interleaving Thought, Action, Observation steps.
```

### --- Page 0023 ---

```markdown
Answer the following questions as best you can. You have access to the following tools:
Search: useful for when you need to answer questions about the world
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {Search}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin
Question: (question)
Thought:

![Langchain ReAct example prompt for interleaving Thought, Action, Observation steps.](assets/page_0023_img_1.png)
```

### --- Page 0024 ---

```markdown
# Preprint

Given the following extracted parts of a long document and a question, create a final answer with references (“SOURCES”).  
If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer.  
Always return a “SOURCES” part in your answer.

---

**QUESTION:** Which state/country's law governs the interpretation of the contract?  
**ANSWER:**  
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an injunction or other relief to protect its Intellectual Property Rights.  
**Source:** 29-30  
Content: No waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other) right or remedy.  
11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation in force of the remainder of the term (if any) and this Agreement.  
11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any kind between the parties.  
**Source:** 31  
In 19.10 Third-Party Beneficiaries.  
**Source:** 34  
Content: (b) If Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as defined in Clause 8.5) or that such a violation is reasonably likely to occur,  
**Source:** 41  
**FINAL ANSWER:** This Agreement is governed by English law.  
**SOURCES:** 29-31

---

**QUESTION:** What did the president say about Michael Jackson?  
**ANSWER:**  
Content: Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  
Last year COVID-19 took us apart. This year we are finally together again.  
Tonight, we meet as Democrats, Republicans and Independents. But most importantly as Americans.  
With a unity of purpose and the American people to the Constitution.  
And with an unwavering resolve that freedom will always triumph over tyranny.  
Six days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could bend it to his will.  
He thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined.  
He met President Zelenskyy and every Ukrainian, their fearlessness, their courage, their determination, inspires the world.  
Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.  
**Source:** 32-33  
Content: And we won’t stop.  
We have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life.  
Let’s not use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  
Let’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  
We can’t change how divided we’ve been. But we can change how we move forward on COVID-19 and other issues we must face together.  
I recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera.  
They were responding to a 9-1-1 call when a man shot and killed them with a stolen gun.  
Officer Mora was 27 years old.  
Officer Rivera was 22.  
Both Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers.  
I spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.  
**Source:** 24-25  
Content: And a proud Ukrainian people, who have known 30 years of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  
To all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has cost around the world.  
And I’m taking robust action to make sure the pain of our sanctions is targeted at Russia’s economy. And I will use every tool at our disposal to protect American business and consumers.  
Tonight, I can announce that the United States has worked with 30 other countries to release 60 million barrels of oil from our reserves around the world.  
America will lead the effort, releasing 30 million barrels from our Strategic Petroleum Reserve. And we stand ready to do more if necessary, united with our allies.  
These steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming.  
But I want you to know that we are going to be okay.  
**Source:** 51  
Content: More support for patients and families.  
To that end, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health.  
It’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  
ARPA-H will have a singular purpose: to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.  
```

### --- Page 0025 ---

```markdown
# Preprint

## Figure 5: Langchain example prompt for QA with sources.

```
A unity agenda for the nation.  
We can do this.  
My fellow Americans! Tonight, we have gathered in a sacred space—the citadel of our democracy.  
In this Coalition, generation after generation, Americans have debated great questions amid great strife, and have done great things.  
We have fought for freedom, expanded liberty, defeated totalitarianism and terror.  
And built the strongest, freest, and most prosperous nation the world has ever known.  
Now is the hour.  
Our moment of responsibility.  
Our test of resolve and conscience, of history itself.  
It is in this moment that our character is formed. Our purpose is found. Our future is forged.  
Well I know this nation.  
Source: -34p1  
=======  
FUNKS ANSWER: The president did not mention Michael Jackson.  
```

## Figure 6: Langchain example prompt for SQL querying using MyScale.

```
You are a MyScale expert. Given an input question, first create a syntactically correct MyScale query to run, then look at the results of the query and return the answer to the input question.  
MyScale queries have a vector distance function called DISTANCE(column, array) to compute relevance to the user's question and under the feature array column by the relevance.  
When the query is asking for {top}k closest row, you have to use this distance function to calculate distance to entity's array on vector column and order by the distance to retrieve relevant rows.  
**NOTICE:** DISTANCE(column, array) only accepts an array column as its first argument and a NeuralArray(entity) as its second argument. You also need a user-defined function called NeuralArray(entity) to retrieve the entity's array.  
Unless the user specifies in the question a specific example to obtain, query for at most {top}k results using the LIMIT clause as per MyScale. You should only query the columns needed to answer the question.  
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as defined identifiers.  
Do not continue to use the columns names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which columns in which table.  
Pay attention to use today() function to get the current date, if the question involves "today". ORDER BY clause should always be after WHERE clause. Do NOT add semicolon to the end of SQL. Pay attention to the content in table schema.  

Use the following format:  
```  
table_name info =======  
table.info()  
```  
**Question:** {input}  
**SQLQuery:**  

Here are some examples:  
```  
CREATE TABLE "ChatPaper" (  
  abstract STRING,  
  id STRING,  
  vector ARRAY(Float32),  
  ENGINE = RelocatedReplacingMergeTree()  
)  
ORDER BY id;  
```  
**Question:** What is the fastest Pyramid Network?  
**SQLQuery:** SELECT "ChatPaper".title, "ChatPaper".authors FROM "ChatPaper" ORDER BY DISTANCE(vector, NeuralArray("Paper contributor")) LIMIT {top}k;  

Let's begin:  
```  
table_name info =======  
table.info()  
```  
**Question:** {input}  
**SQLQuery:**  
```
```  

```markdown
25
```

### --- Page 0026 ---

```markdown
# Preprint

## A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided.

Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the relevance score.  
The relevance score is a number from 1-10 based on how relevant you think the document is to the question.  
Do not include any documents that are not relevant to the question.

**Example format:**

**Document 1:**  
summary of document 1

**Document 2:**  
<summary of document 2>

...

**Document 10:**  
<summary of document 10>

**Question:** <question>  
Doc: 3; Relevance: 7  
Doc: 2; Relevance: 4  
Doc: 7; Relevance: 3  

Let's try this now:  
{context.str}  
Question: {query.str}  
Answer:

![LlamaIndex example prompt for returning relevant documents and corresponding summaries.](assets/page_0026_img_1.png)

---

## You are an IRS chatbot whose primary goal is to help users with filing their tax returns for the 2022 year.

Provide concise replies that are polite and professional.  
Your answers should be truthfully based on official government information, with consideration to context provided below on changes for 2022 that can affect tax refund.

1. Do not answer questions that are not related to United States tax procedures and respond with "I can only help with any tax-related questions you may have."  
2. If you do not know the answer to a question, respond by saying "I do not know the answer to your question. You may be able to find your answer at www.irs.gov/faq."

### Changes for 2022 that can affect tax refund:

- Changes in the number of dependents, employment or self-employment income and divorce, among other factors, may affect tax-filing status and refund. No additional stimulus payments. Unlike 2020 and 2021, there were no new stimulus payments for 2022 so taxpayers should not expect to get an additional payment.
- Some tax credits return to 2019 levels. This means that taxpayers will likely receive a significantly smaller refund compared with the previous years. Changes include returns for the Child Tax Credit (CTC), the Earned Income Tax Credit (EITC) and the Child and Dependent Care Credit will revert to pre-COVID levels.
- For 2022, the CTC is now $2,000 for each qualifying child. A child must be under age 17 at the end of 2022 to be a qualifying child for the CTC. Eligible taxpayers with children will get $500 for the 2022 tax year. The child and dependent care credit has a maximum of $2,100 in 2022.
- Above-the-line charitable deductions. During 2021, taxpayers were able to take up to a $600 charitable donation tax deduction on their tax returns. However, for tax year 2022, taxpayers who don’t itemize and who take the standard deduction, won’t be able to deduct their charitable contributions.
- More people may be eligible for the Premium Tax Credit. For tax year 2022, taxpayers may qualify for temporarily expanded eligibility for the premium tax credit.
- Eligibility rules changed to claim a tax credit for clean vehicles. Review the changes under the Inflation Reduction Act of 2022 to qualify for a Clean Vehicle Credit.

![LlamaIndex example prompt for IRS chatbot guidelines.](assets/page_0026_img_2.png)
```

### --- Page 0027 ---

```markdown
# D MODULES

## D.1 PREDICT

```python
class Predict(dspy.Module):
    def __init__(self, signature, **config):
        self.signature = dspy.Signature(signature)
        self.config = config

        # Module Parameters.
        self.lm = dspy.ParameterLM(None)  # use the default LM
        self.demonstrations = dspy.ParameterDemonstrations([])

    def forward(self, **kwargs):
        lm = get_the_right_lm(self.lm, kwargs)
        signature = get_the_right_signature(self.signature, kwargs)
        demonstrations = get_the_right_demonstrations(self.demonstrations, kwargs)

        prompt = signature(demonstrations.demos, **kwargs)
        completions = lm.generate(prompt, **self.config)
        prediction = Prediction.from_completions(completions, signatures=signature)

        if dspy.settings.compiling is not None:
            trace = dict(predictor=self, inputs=kwargs, outputs=prediction)
            dspy.settings.traces.append(trace)

        return prediction
```

## D.2 CHAIN OF THOUGHT

```python
class ChainOfThought(dspy.Module):
    def __init__(self, signature):
        # Modify signature from 'inputs -> outputs' to 'inputs -> rationale, outputs'.
        rationale_field = dspy.Field('rationale', 'Let\'s think step by step.')
        signature = dspy.Signature(signature).prepend_output_field(rationale_field)

        # Declare a sub-module with the modified signature.
        self.predict = dspy.Predict(self.signature)

    def forward(self, **kwargs):
        # Just forward the inputs to the sub-module.
        return self.predict(**kwargs)
```
```

### --- Page 0028 ---

```markdown
# E TELEPROMPTERS

## E.1 BOOTSTRAPFEWSHOT

```python
class SimplifiedBootstrapFewShot(Teleprompter):
    def __init__(self, metric=None):
        self.metric = metric

    def compile(self, student, trainset, teacher=None):
        teacher = teacher if teacher is not None else student
        compiled_program = student.deepcopy()

        # Step 1. Prepare mappings between Student and Teacher Predict modules.
        # Note: other modules will rely on Predict internally.
        assert student.have_compatible_predict_modules(student, teacher)
        name_predictor, predictor_name = map_predictors_recursively(student, teacher)

        # Step 2. Bootstrap traces for each Predict module.
        # We'll loop over the training set. We'll try each example once for simplicity.
        for example in trainset:
            if we_found_enough_bootstrap_demos(): break

            # turn on compiling mode which will allow us to keep track of the traces
            with dspy.setting.context(compiling=True):
                # run the teacher program on the example, and get its final prediction
                prediction = teacher(*example.inputs())

            # get the trace of all internal Predict calls from teacher program
            predicted_traces = dspy.settings.trace

            if self.metric(example, prediction, predicted_traces):
                # add the example to the traces
                for predictor, inputs, outputs in predicted_traces:
                    d = dspy.Example(automated=True, **inputs, **outputs)
                    predictor_name = self.predictorName[id(predictor)]
                    compiled_program[predictor_name].demonstrations.append(d)

        return compiled_program
```

## E.2 BOOTSTRAPFEWSHOTWITHRANDONSEARCH

```python
class SimplifiedBootstrapFewShotWithRandomSearch(Teleprompter):
    def __init__(self, metric=None, trials=16):
        self.metric = metric
        self.trials = trials

    def compile(self, student, *, teacher=None, trainset, valset=None):
        # we can do forms of cross-validation if valset is unset.
        valset = trainset if valset is None else valset

        candidates = []
        for seed in range(self.trials):
            # Create a new basic bootstrap few-shot program.
            shuffled_trainset = shuffle(trainset, seed=seed)
            tp = BootstrapFewShot(metric=self.metric, max_bootstrap.demos=random_size())
            candidate_program = tp.compile(student, shuffled_trainset, teacher)

            # Step 2: Evaluate the generated candidate program.
            score = evaluate_program(candidate_program, self.metric, valset)
            candidates.append((score, candidate_program))

        # return the best candidate program.
        return max(candidates, key=lambda x: x[0])[1]
```
```

### --- Page 0029 ---

```markdown
# E.3 BOOTSTRAPFEWFewShotWithOptuna

```python
class SimpleBootstrapFewShotWithOptuna(Teleprompter):
    def __init__(self, metric, trials=16):
        self.metric = metric
        self.trials = trials

    def objective(self, trial):
        # Step 1: Create copy of student program.
        candidate_program = self.student.reset_copy()

        # Step 2: Based on trial, select demos for each predictor in program.
        # Note: For simplicity, we can just select a single demo for each predictor.
        # But we can easily tune the number of demonstrations to select here.
        for (name, predictor), in zip(candidate_program.named_predictors()):
            all_demos = predictor.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            predictor.demos = all_demos[demo_index]

        # Step 3: Evaluate the modified candidate program.
        score = evaluate_program(candidate_program, self.metric, self.valset)

        # Step 4: Store the candidate for Optuna to select highest-scoring program.
        trial.set_user_attr("program", candidate_program)
        return score

    def compile(self, student, trainset, teacher=None, valset=None):
        self.trainset = trainset
        self.valset = trainset if valset is None else valset
        self.student = student.deepcopy()
        self.teacher = teacher.deepcopy() if teacher else student.deepcopy()

        tp = BootstrapFewShot()
        self.pool = tp.compile(self.student, self.teacher, self.trainset, self.metric)

        # Use Optuna to find the best program by optimizing the objective function.
        best_program = optimize_with_optuna(self.objective)

        print('Best score:', best_program.score)
        print('Best program:', best_program)
        return best_program
```
```

### --- Page 0030 ---

```markdown
# F  EXAMPLES OF THE PROMPTS AUTOMATICALLY GENERATED BY DSPy

For GSM8K, we include the prompt bootstrapped by DSPy for GSM8K 11lama2-13b-chat for the vanilla program compiled with $bootstrap \times 2$ in Figure 9.

We also include a CoT prompt for GSM8K and a generate_query prompt from the multihop program for HotPotQA. All of these, particularly their demonstrations' labels and their selection, are generated by DSPy automatically using 11lama2-13b-chat.

![Detailed description of the chart](assets/page_0030_img_1.png)
```

### --- Page 0031 ---

```markdown
# Preprint

![Copy of the prompt automatically generated by DSPy for GSM8K Llama2-13b-chat vanilla program compiled with bootstrap](assets/page_0031_img_1.png)

---

Given the fields `question`, produce the fields `answer`.

---

Follow the following format.

Question: ${question}  
Answer: ${answer}

---

Question: Jimmy and Irene go shopping for clothes on a Tuesday, where senior citizens get a 10% discount on their purchases. Jimmy picks out 3 shorts from the $15 rack. Irene grabs 5 shirts from the $17 rack. How much money do they give to the cashier?  
Answer: Jimmy picks out 3 shorts at $15 each = $45. Irene grabs 5 shirts at $17 each = $85. Total cost = $45 + $85 = $130. Since senior citizens get a 10% discount, they will pay 10% of $130 = $13. So, they will give the cashier $130 - $13 = $117.

---

Question:

---

Given the fields `question`, produce the fields `answer`.

---

Follow the following format.

Question: ${question}  
Reasoning: Let's think step by step in order to produce the answer. We ...

Answer: ${answer}

---

Question: Mark is baking bread. He has to let it rise for 120 minutes twice. He also needs to spend 10 minutes kneading the seed. How many minutes does it take Mark to finish making the bread?  
Reasoning: Let's think step by step in order to find out how many minutes it takes for finishing making the bread. We know that he needs to let it rise for 120 minutes twice, so that's 240 minutes. Then, he needs to spend 10 minutes kneading it and 30 minutes baking it. So, in total, it will take Mark 240 + 10 + 30 = 280 minutes to finish making the bread.  
Answer: 280

---

Question: Ben has $2200 for his business operations costs. He orders goods from his supplier and writes them a cheque for $500. His debtor pays him $500 from the purchases they made on credit. Mr. Ben then decides to do equipment maintenance and spends $120 on the whole operation. How much money is Mr. Ben remaining with?  
Reasoning: Let's think step by step in order to find out how much money Mr. Ben is remaining with. We know that he had $2200 to begin with, and he spent $500 on goods from his supplier, so he has $2200 - $500 = $1700 left. Then, this debtor paid him $500, so he has $1700 + $500 = $2200. Finally, he spent $1200 on equipment maintenance, so he has $2200 - $1200 = $1000 left.  
Answer: 1000

---

... several other demonstrations here ...

---

Question:
```

### --- Page 0032 ---

```markdown
Given the fields ‘context’, ‘question’, produce the fields ‘search_query’.

---

Follow the following format.

Context: ${context}  
Question: ${question}  
Reasoning: Let’s think step by step in order to ${produce the search_query}. We ...  
Search Query: ${search_query}

---

Context:  
[1] Twilight (novel series) | Twilight is a series of four vampire-themed fantasy romance novels by American author Stephenie Meyer. ...  
[2] Harper Connelly Mysteries | The Harper Connelly Mysteries is a series of fantasy mystery novels written by Charlaine Harris, and first published in 2005. ...  
[3] The Dark Heroine | The Dark Heroine is a series of vampire-themed fantasy romance novels written by English author Abigail Gibbs, published by HarperCollins in 2012. ...

Question: In which year was the first of the vampire-themed fantasy romance novels for which The Twilight Saga: The Official Illustrated Guide serves as a spin-off encyclopedia reference book first published?  

Search Query: When was the first of the vampire-themed fantasy romance novels published?

---

Context:  
[1] The Victorians | The Victorians - Their Story In Pictures is a 2009 British documentary series which focuses on Victorian art and culture. ...  
[2] The Cantons | The Cantons: A Family Picture is an 1849 Victorian novel by Edward Bulwer-Lytton that was popular in its time.  
[3] Victorian (comics) | The Victorian is a 25-issue comic book series published by Penny-Farthing Press and starting in 1999. ...

Question: The Victorians - Their Story In Pictures is a documentary series written by an author born in what year?  

Reasoning: Let’s think step by step in order to produce the search query. We know that the documentary series is about Victorian art and culture, and it was written and presented by Jeremy Paxman. Therefore, we need to find the year in which Jeremy Paxman was born.

Search Query: Jeremy Paxman birth year

---

Context:
```
![Shortened copy of the prompt automatically generated by DSPy for HotPotQA Llama2-13b-chat multi-hop program (generating second hop query) compiled with bootstrap.](assets/page_0032_img_1.png)
```


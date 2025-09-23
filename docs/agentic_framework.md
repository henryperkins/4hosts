# Unified Framework for Context and Prompt Engineering in Agentic Research Systems

## Introduction

The development of applications powered by Large Language Models (LLMs) is undergoing a critical transformation, evolving from an art of clever "prompt hacking" into a rigorous engineering discipline. At the heart of this evolution lie two symbiotic practices: **Context Engineering** and **Prompt Engineering**. Building reliable, scalable, and predictable AI systems requires a holistic, architectural approach that moves beyond crafting isolated instructions. This report posits that the mastery of these intertwined disciplines is the cornerstone of creating robust, production-grade agentic systems.

**Context Engineering** is the architectural discipline concerned with the systematic, programmatic selection, shaping, and sequencing of the entire information payload an LLM receives during inference. It is not merely about prompt design; it is about architecting the complete "mental world" in which the model operates, encompassing everything from retrieved documents and tool outputs to conversational memory and internal reasoning artifacts. In contrast, **Prompt Engineering** is the crucial, instruction-level component of this broader practice. It focuses on crafting the specific text template that frames the curated context and directs the model's behavior, style, and output schema within that engineered world.

The necessity of this unified approach is most pronounced in the context of the **agentic paradigm**. An agentic research system operates through an autonomous, multi-step loop of planning, acting, and reflecting to accomplish complex, open-ended goals. This iterative process, where each step consumes context from previous steps and emits new artifacts that become context for future steps, creates a dynamic feedback loop. The stability and effectiveness of this loop are entirely dependent on the precise management of this flow of information.

This report provides a comprehensive, unified framework for practitioners aiming to build, debug, and maintain resilient agentic research systems. It will proceed from foundational concepts and mental models to a deep dive into the technical pillars of context engineering. It will then detail high-impact prompt engineering patterns specifically for agents, followed by a concrete architectural blueprint for an integrated research service. Finally, it will establish a robust framework for evaluation and observability before concluding with a practical analysis of common failure modes and their mitigation strategies. The objective is to equip engineers and architects with the principles and patterns necessary to transform fragile sequences of LLM calls into resilient, explainable, and maintainable AI systems.

---

## Section 1: Foundational Concepts and Mental Models

A shared and precise understanding of core terminology is essential before delving into the architectural and implementation specifics of agentic systems. This section formally defines the disciplines of Context and Prompt Engineering, clarifies their hierarchical relationship, and deconstructs the operational framework of the agentic research loop.

### 1.1 Context Engineering: The Architecture of Awareness

Context Engineering is formally defined as the systematic, programmatic optimization of the information payload supplied to a Large Language Model during inference. It transcends simple prompt design to become a form of systems design or software architecture for LLMs. The discipline is not concerned with what to say to the model at a single moment in time, but rather with what the model knows when you say it, and why it should care. It is the practice of designing the entire mental world the model inhabits, ensuring it performs consistently and predictably across varied sessions, users, and tasks.

The scope of Context Engineering is expansive, encompassing the entire data lifecycle within an agentic loop. This includes:
- **Data Retrieval**: Sourcing information from external knowledge bases.
- **Filtering**: Removing irrelevant or noisy data.
- **Ranking**: Prioritizing the most relevant information.
- **Compression**: Reducing the token footprint of information to fit within constraints.
- **Ordering**: Sequencing information strategically within the prompt.
- **Window-Packing**: Assembling the final context payload within the model's token limit.
- **Memory Eviction**: Deciding which information to discard from short-term or long-term memory.

The primary objective of this discipline is to maximize **relevance density**—the concentration of useful, grounding information per token—while simultaneously ensuring that all grounding sources remain traceable for fact-checking and explainability. This focus on systematic design makes Context Engineering the foundation for building production systems that require predictability and scalability, moving beyond the often hit-or-miss nature of simple prompting.

### 1.2 Prompt Engineering: The Art of Instruction

Prompt Engineering is the craft of designing the specific text template that frames the curated context and instructs the model on its role, task, style, and output schema. If Context Engineering builds the stage, Prompt Engineering writes the script for the actor. It operates within the context window that has been meticulously assembled, focusing on the precise wording and structure needed to elicit a desired response for a single input-output pair.

The scope of Prompt Engineering is more focused, covering several key instructional domains:
- **Role and Persona Framing**: Assigning the model a character to adopt (e.g., "You are a senior research analyst at a global consulting firm") to guide its tone and perspective.
- **Task Specification**: Clearly and unambiguously defining the goal of the interaction using action verbs (e.g., "Summarize the key findings," "Generate a JSON list").
- **Style and Format Constraints**: Specifying the desired presentation of the output, such as length, format (e.g., bullet points, table), and writing style.
- **Reasoning Directives**: Instructing the model on how to think, most notably through Chain-of-Thought (CoT) prompting ("Think step-by-step") to unlock more complex reasoning abilities.
- **Tool Invocation Hints**: Providing clear syntax, examples, and schemas for how the model should call external functions or tools.

### 1.3 The Hierarchy and Interplay: A Unified View

A common misconception is to view Context Engineering and Prompt Engineering as competing or equivalent practices. A more accurate mental model establishes a clear hierarchy: **Prompt Engineering is a subset of Context Engineering**. The most brilliantly crafted prompt will fail if the context surrounding it is noisy, irrelevant, or poorly structured. For instance, a precise instruction can be rendered useless if it is buried behind thousands of tokens of irrelevant chat history or badly formatted retrieved documents.

This relationship can be understood through a practical analogy: Prompt Engineering is akin to writing a perfectly clear and persuasive letter. Every word is chosen with care to ensure the recipient understands the message perfectly. Context Engineering, however, is like running the entire household. It involves juggling schedules, budgets, moods, and unexpected events to ensure everyone is fed, rested, and prepared for what comes next. A design leader, or an AI architect, must run the household.

This distinction highlights a crucial maturation in the field of AI development. Early interactions with LLMs were dominated by the creative craft of prompt "hacking," focused on achieving impressive one-off results. However, building reliable, scalable, and maintainable systems—the kind needed for enterprise applications—demands the rigor of Context Engineering, a discipline closer to software architecture. The former is sufficient for a flashy demo; the latter is a prerequisite for a production-grade system that delivers consistent value over thousands of interactions. The formalization of Context Engineering as a field of study, exemplified by dedicated academic surveys, signals that the industry is moving beyond simple chatbots and toward complex, stateful applications that demand architectural discipline.

**Table 1: Context Engineering vs. Prompt Engineering: A Comparative Analysis**

| Discipline          | Scope                                                                 | Primary Goal                                                                 | Key Techniques & Tools                                                                 | Debugging Focus                                                                 | Failure Mode Example                                                                 |
|---------------------|-----------------------------------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Context Engineering | The entire information payload: retrieval, memory, tools, history, scratchpads. Manages what the model knows. | Maximize relevance density per token and ensure consistent, predictable performance across many sessions. | RAG systems, vector databases, memory modules, context caching, compression algorithms (Map-Reduce, Semantic Hashing). | Inspecting the full context window, token flow, retrieval scores, and memory state. | The system hallucinates because the retriever pulled irrelevant documents, or it crashes due to exceeding the token limit. |
| Prompt Engineering  | The instructional text template: role, task, style, format, reasoning directives. Manages what the model does. | Elicit a specific, high-quality response for a single input-output pair.     | Few-shot examples, Chain-of-Thought (CoT), Re-Act patterns, structured output schemas (JSON/XML), self-critique instructions. | Rewording instructions, adjusting examples, and analyzing the final output for stylistic or factual errors. | The output is in the wrong format, the tone is off, or the reasoning is flawed despite having the correct information. |

### 1.4 The Agentic Research Loop: The Operational Framework

Agentic AI systems are defined by their capacity to act with autonomy, initiative, and adaptability to pursue goals in dynamic environments. They achieve this through an iterative operational framework, often referred to as the **agentic research loop**. This loop is not merely a software pattern but an attempt to implement a simplified cognitive process, enabling the system to plan, act, learn, and improve over time. This cycle is particularly relevant because it is where the interplay between context and prompt engineering becomes most critical and dynamic. The loop can be deconstructed into five core phases:
1. **Plan**: Faced with a complex or open-ended user query, the agent first decomposes the problem into a sequence of smaller, more manageable sub-tasks or questions. This planning phase is essential for navigating problems where the path to a solution cannot be hardcoded in advance.
2. **Retrieve/Act**: The agent executes a single step from its plan. This action typically involves interacting with the external world by calling a tool (e.g., performing a web search, executing code, querying a database) or retrieving information from its internal memory systems. This is the primary context-gathering phase of the loop.
3. **Think/Reason**: The LLM, serving as the agent's "brain," processes the newly acquired context from the previous step. It integrates this new information with its instructions (the prompt), its existing knowledge, and any intermediate results stored in its scratchpad to reason about the state of the problem and decide on the next course of action.
4. **Write/Record**: The outcome of the reasoning step is recorded as a structured artifact. This could be a summary of a retrieved document, the output of a tool, or a piece of code. These artifacts are often stored in a "scratchpad" or working memory, forming part of the context for subsequent steps.
5. **Reflect/Critique**: The agent assesses its progress against the overall goal. It may update its long-term memory with new learnings, critique its own work to identify errors, and refine or update its plan for the next iteration. This metacognitive step is crucial for reducing hallucinations, enforcing quality, and ensuring the agent does not get stuck in unproductive loops.

Viewing this agentic loop through a cognitive lens reveals its architectural depth. Academic research on cognitive architectures explicitly maps the components of such systems to the "common model of cognition," where the LLM's context window functions as "working memory," external vector stores serve as "long-term memory," and the prompt acts as the "procedural memory" that dictates how to operate on them. This perspective suggests that improving agent performance is not solely a matter of developing more powerful LLMs. It is equally about designing more effective cognitive architectures—a core task of Context Engineering. This involves optimizing the efficiency of memory retrieval, the quality of the planning module, and the robustness of the reflection mechanism to create a more capable and reliable system.

---

## Section 2: The Pillars of Advanced Context Engineering

Mastering Context Engineering requires a deep, technical understanding of the components that manage the flow of information into the LLM. This section provides a detailed examination of the three foundational pillars: **Retrieval-Augmented Generation (RAG)** for external knowledge, **Memory Systems** for statefulness, and **Context Window Management** for handling resource constraints.

### 2.1 Retrieval-Augmented Generation (RAG) Architectures

RAG is the fundamental technique for grounding LLMs in external, up-to-date, or proprietary knowledge. By retrieving relevant information before generation, RAG systems significantly mitigate the risks of factual hallucination and overcome the static nature of a model's training data. A production-grade RAG pipeline is a multi-stage process.

#### 2.1.1 Pre-Retrieval: Query Analysis and Routing

Before any documents are fetched, a sophisticated RAG system must first understand the user's intent. A powerful technique is to use an LLM as a preliminary query classifier. This involves a separate, lightweight LLM call that analyzes the user's query and routes it accordingly. This routing can determine which data source to query—for example, directing a technical question to an internal engineering wiki versus an SEO-related question to a marketing knowledge base. This classification can also be used to select the most appropriate system message or prompt template for the task at hand, ensuring the agent is properly configured before it even begins retrieval. This pre-retrieval step prevents queries from being sent to irrelevant data stores, improving both efficiency and the relevance of the retrieved context.

#### 2.1.2 Core Retrieval: Vector Search Strategies

The heart of a RAG system is its retriever, which is typically powered by a vector database. After indexing documents by splitting them into chunks and generating vector embeddings, the retriever finds the most relevant chunks for a given query. Two primary strategies dominate this space:
- **Similarity Search**: This is the baseline approach, where the retriever fetches the top-k document chunks whose vector embeddings are most similar to the query's embedding, often measured by cosine similarity. While straightforward and effective for finding closely related content, it can suffer from a lack of diversity, often returning multiple chunks that repeat the same information.
- **Maximal Marginal Relevance (MMR)**: MMR is a more advanced and often preferable retrieval algorithm that optimizes for both relevance and diversity. After finding an initial set of relevant documents, MMR iteratively selects chunks by balancing their similarity to the original query against their dissimilarity to the chunks already selected. This process, controlled by a parameter often denoted as λ, helps prevent redundancy in the context window. By including more diverse perspectives, MMR provides the LLM with a richer and more comprehensive set of information, which can lead to more nuanced and complete answers.

#### 2.1.3 Post-Retrieval: Ensuring Traceability with Cite-Aware RAG

A trustworthy AI system must be able to justify its claims. Cite-aware RAG is an architectural pattern that ensures the final output is transparent and verifiable. Implementation requires two key features. First, during the indexing process, each document chunk must be stored with a unique source identifier (e.g., document ID, page number, paragraph number). Second, the generation prompt must explicitly instruct the LLM to include these source IDs as inline citations in its response. For example, a generated sentence might look like: "The company's revenue grew by 3% in Q2 2023." This creates a clear audit trail from the generated claim back to the source document, allowing users to verify the information and building trust in the system's output. Advanced techniques like saliency-based attribution can even link specific phrases in the output to the most relevant sentences in the source material, providing granular traceability.

### 2.2 Memory Systems for Long-Horizon Agency

LLMs are inherently stateless; they have no memory of past interactions beyond the information provided in the current context window. Therefore, memory is not a feature of the model itself but a critical component of the surrounding agentic framework that must be explicitly engineered. Memory systems are typically divided into two categories.

#### 2.2.1 Short-Term (Working) Memory

Short-term memory holds context-specific information relevant to the current, ongoing task or conversation. It is analogous to a computer's RAM—fast but volatile. The LLM's own context window serves as the most immediate form of working memory. For conversations that exceed the window size, common techniques include:
- **Sliding Window**: Keeping only the N most recent conversational turns.
- **Summarization**: Periodically using an LLM to summarize older parts of the conversation, replacing the detailed history with a more compact summary.
- **Checkpointers**: More advanced agentic frameworks like LangGraph use "checkpointers" to save the state of a multi-step execution, allowing it to be paused and resumed. This state acts as a form of short-term memory for the specific task.

#### 2.2.2 Long-Term (Persistent) Memory

Long-term memory enables an agent to retain and recall information across multiple sessions, allowing it to learn from past interactions, remember user preferences, and build a persistent knowledge base. These systems are almost universally implemented using vector databases. The process involves storing memories—such as user facts, past conversations, or successful problem-solving strategies—as vectorized chunks. These memories are then retrieved based on their semantic similarity to the current query or situation.

Drawing from cognitive science, long-term memory in agents can be structured into different types:
- **Episodic Memory**: Stores specific events and past interactions (e.g., "The user previously asked about travel to London in July").
- **Semantic Memory**: Stores general knowledge and facts (e.g., "London is the capital of the UK").
- **Procedural Memory**: Stores "how-to" knowledge and learned skills (e.g., "The most efficient way to book a flight involves checking these three websites first").

Advanced agents can even be prompted to reflect on their experiences and generate their own memories to store, or to formulate specific queries to retrieve information from their long-term memory store, creating a powerful loop of learning and adaptation.

### 2.3 Strategic Context Window Management

While the context windows of modern LLMs are expanding dramatically, simply having a larger window is not a silver bullet. Large contexts introduce a distinct set of engineering challenges, including:
- **Declining Accuracy**: Models can struggle to identify the most critical information when it is "lost in the middle" of a very long context.
- **Increased Latency and Cost**: The computational complexity of the attention mechanism in transformers scales quadratically with the sequence length, meaning larger contexts lead to significantly slower and more expensive inference.
- **The "Needle in a Haystack" Problem**: Finding a single, critical piece of information becomes exponentially harder as the size of the surrounding "haystack" of text increases.

This reality necessitates a shift from a "fill-it-up" mentality to a strategic approach to context management, treating the context window as a constrained and valuable resource. This involves two key classes of techniques: **compression** and **caching**.

#### 2.3.1 Context Compression Techniques

Compression techniques aim to reduce the token count of information while preserving its semantic essence, allowing more knowledge to fit into a fixed budget.
- **Map-Reduce Summarization**: This is a powerful, albeit lossy, technique for handling documents that are too long to fit in the context window. The process involves:
  1. **Map**: The long document is split into smaller, manageable chunks that each fit within the context window. An LLM is then called in parallel to summarize each individual chunk.
  2. **Reduce**: The summaries from the map step are concatenated. If this combined text is still too long, the process is repeated—the summaries are themselves summarized. This recursive "collapsing" can be applied until the final summary is within the desired token limit.
- **Semantic Hashing**: This is a more advanced compression technique that converts text into a compact binary vector, or "hash". Unlike cryptographic hashes, semantic hashes are designed such that similar texts produce similar hashes (in terms of Hamming distance). This has two primary applications in context engineering. First, it can be used as a highly efficient retrieval mechanism, as seen in frameworks like Hash-RAG, where searching for similar hashes is much faster than searching in a full vector space. Second, the hash itself can serve as a highly compressed representation of the original text, preserving semantic meaning in a very small number of bits. This is also being explored for compressing the Key-Value (KV) cache within the transformer architecture itself to speed up inference on long contexts.

#### 2.3.2 Cost & Latency Optimization: Context Caching

For applications where parts of the prompt are static across many requests (e.g., a complex system prompt, few-shot examples, or a large piece of reference text), context caching is a critical optimization strategy. Caching works by pre-processing and storing the tokenized representation of these static prompt components in a fast-access memory. When a new request arrives, the system only needs to process the new, dynamic parts of the prompt and can reuse the cached components. This dramatically reduces the number of tokens that need to be processed for each call, leading to significant cost savings and improved time-to-first-token (TTFT). Major model providers offer different implementations of this feature, with varying technical specifications that make them suitable for different use cases.

**Table 2: Comparison of Context Caching Strategies**

| Provider       | Compatible Models                          | Minimum Token Threshold          | Cache Lifetime              | Pricing Impact/Structure                          | Best Use Case                                                                 |
|----------------|--------------------------------------------|----------------------------------|-----------------------------|---------------------------------------------------|-------------------------------------------------------------------------------|
| Google Gemini  | Stable versions of Gemini 1.5 Pro & Flash  | 32,768 tokens                    | 1 hour (default, customizable) | Standard token price + hourly maintenance fee.    | High-volume applications with very large, consistent prompts (e.g., analyzing large documents repeatedly). |
| Anthropic Claude | Claude 3.x series (Haiku, Sonnet, Opus)    | 1,024 (Sonnet/Opus) or 2,048 (Haiku) tokens | 5 minutes (refreshed on read) | 25% premium on write, 90% discount on read.       | Applications with frequent reuse of medium-sized prompts where requests are closely spaced. |
| OpenAI         | gpt-4.5-preview, gpt-4o, gpt-4o-mini, o1-mini | 1,024 tokens                     | 5-10 minutes (up to 1 hour off-peak) | Automatic 50% cost reduction for eligible requests; no special pricing tiers. | General-purpose or mixed workloads where simplicity and automatic optimization are valued. |

---

## Section 3: High-Impact Prompt Engineering Patterns for Agents

Once the context has been meticulously engineered, the prompt serves as the precise set of instructions that guides the agent's reasoning and actions. Effective prompt engineering for agents is about reducing ambiguity and enforcing consistency, transforming the prompt from a simple question into a configuration layer for the agent's behavior. This section details the foundational, tool-oriented, and metacognitive patterns that are essential for building reliable agents.

### 3.1 Foundational Patterns for Guiding Behavior

These patterns establish the agent's core operational parameters and bias its output toward the desired style and format.
- **System Instructions**: The system prompt is the most powerful tool for setting an agent's immutable policy, persona, and high-level purpose. It is the first instruction the model receives and sets the tone for the entire interaction. Best practices dictate that system instructions should be concise (ideally under 100 tokens), clear, and free of contradictions. For example: "You are a helpful AI assistant. Your task is to provide a comprehensive plan to execute the given user task. Think step-by-step in great detail. Assume you have access to a web browser."
- **Few-Shot Examples**: To guide the model toward a specific output format, style, or reasoning process, providing a few examples (few-shot learning) is highly effective. These examples act as demonstrations, showing the model exactly what is expected. For models from providers like OpenAI and Google, these examples are most effective when placed in the prompt just before the final user message, as this positioning strongly influences the subsequent generation.
- **Chain-of-Thought (CoT) Prompting**: CoT is a seminal technique that dramatically improves an LLM's performance on tasks requiring logical, mathematical, or multi-step reasoning. By simply adding the instruction "Let's think step-by-step" or a similar phrase, the prompt compels the model to externalize its reasoning process before providing a final answer. This decomposition of a complex problem into smaller, sequential steps reduces the likelihood of errors. For user-facing applications, the verbose chain of thought can be hidden or post-processed, so the user only sees the final, clean answer.

### 3.2 Prompts for Tool and Function Integration

The ability to use tools is a defining characteristic of an agent. Prompting for reliable tool use requires absolute clarity and consistency.
- **Structured Tool Usage**: The key to reliable tool invocation is a clear, unambiguous format. Prompts must include explicit instructions on the expected output format for a tool call, such as a specific JSON schema or an XML-like syntax. Providing concrete examples of correctly formatted tool calls within the prompt is also a critical best practice. This not only helps the model use the tools correctly but also simplifies debugging for developers, as the structured output is easier to parse and validate.
- **Defining the "World"**: The prompt must present the model with a complete and consistent picture of its operational environment. This includes a thorough description of all available tools, their specific purposes, their required parameters (with data types), and how they should be used. It is paramount that the description of a tool in the system prompt is perfectly consistent with the tool's actual programmatic definition. Any discrepancy can confuse the model and lead to misuse or failure to use the tool.
- **Handling Tool Failures**: Agents will inevitably attempt to call tools with incorrect parameters. A robust system anticipates this. When a tool call fails, the function should not return a generic error. Instead, it should return a descriptive error message that is passed back to the agent as the observation. For example: "Tool call failed. Parameter date expects format YYYY-MM-DD, but received 'today'." Models are often capable of parsing these error messages and self-correcting on the next attempt.

### 3.3 Eliciting Metacognition: Prompts for Self-Reflection

The most advanced agents are capable of a form of metacognition—reasoning about their own thinking and actions. This can be elicited through specific prompting patterns.
- **The Re-Act Pattern (Reason + Act)**: This pattern structures the agent's turn into a synergistic loop of reasoning and acting. In each step, the model is prompted to generate:
  1. A **Thought**: Reasoning about the current state and what action is needed next.
  2. An **Action**: The specific tool call to be executed.
  3. An **Observation**: The result returned from the tool call.
  The agent then iterates on this cycle, using the observation from the previous step to inform its next thought. This explicit, structured loop is a core pattern for many agentic systems, making the agent's behavior more transparent and controllable.
- **Self-Critique**: A powerful and direct way to improve output quality and reduce hallucinations is to build a self-reflection step directly into the prompt chain. After generating a draft response, a subsequent prompt instructs the model to act as its own quality gate. For example: "Critique your draft answer. Check for factual accuracy against the provided sources, ensure all parts of the user's question have been addressed, and verify that all claims are properly cited. Provide a final, improved answer only after this critique." This forces the model to perform a validation pass on its own work before presenting it to the user.

Across all these patterns, the unifying principle is the reduction of ambiguity. A well-designed prompt suite creates a predictable and consistent operational environment, minimizing the model's need to guess or infer. This approach treats prompting less as an exercise in finding "magic words" and more as a form of software engineering, specifically "design by contract." The prompt defines the contract for the agent's behavior—specifying its inputs, outputs, and processes—and the goal is to make that contract as unambiguous as possible for the non-deterministic "runtime" that is the LLM.

---

## Section 4: A Blueprint for an Integrated Agentic Research Service

This section synthesizes the preceding concepts into a concrete, end-to-end architectural blueprint for a production-grade agentic research service. It details a modular, observable, and testable system designed for resilience and maintainability.

### 4.1 The Layered Context Pipeline: An Architectural Deep Dive

A robust agentic system is best architected using an "orchestrator-worker" pattern, where a central orchestrator manages the overall workflow while delegating specific tasks to specialized sub-agents or tools. This architecture can be understood through three logical layers: the Tool Layer (interfacing with data and APIs), the Reasoning Layer (the LLM core), and the Action/Orchestration Layer (managing the loop and executing actions). The flow of information through this system, from user query to final LLM call, forms the context pipeline.

The journey of a user query through a modular, injectable context assembly pipeline proceeds as follows:
1. **[User Query] -> Query Classifier**: The process begins when the user query is first analyzed by a query classification module. This module, often a lightweight LLM call, determines the user's intent and routes the query to the appropriate downstream resources, such as selecting a specific vector database or a specialized tool.
2. **-> Retriever / Tool Executor**: Based on the classified intent, the system fetches relevant context. This involves retrieving passages from vector stores using strategies like MMR, executing tool calls to access real-time information from APIs like web search, and recalling relevant long-term memories from a memory store.
3. **-> Memory & Scratchpads**: The system gathers the agent's current state, which includes long-term memories, the short-term history of the current session, and any intermediate results (e.g., Chain-of-Thought reasoning, previous tool outputs) stored in a scratchpad.
4. **[Assembly] -> Context Packager**: This component is the unsung hero of a resilient agentic architecture. It acts as a token budget allocator, a critical control point that manages the finite context window of the LLM. It is responsible for assembling the final context payload by prioritizing essential information (e.g., system instructions, tool schemas), deciding what to compress (e.g., long documents, chat history), and what to evict when the token budget is exceeded. The sophistication of this packager—its ability to apply explicit rules for budgeting, prioritization, and compression—is a direct measure of the system's maturity. A naive system simply concatenates context until it breaks; a production-grade system treats the context window like RAM in a constrained operating system.
5. **[Final Output] -> Prompt Templater**: Finally, the meticulously assembled context block is programmatically inserted into a final, step-specific prompt template before being sent to the LLM for the reasoning step.

To ensure the system is debuggable and optimizable, it is crucial to engineer for observability from the outset. This means emitting detailed telemetry and logs at each stage of the pipeline, including token counts (in and out), retrieval scores and hit rates, component latencies, and final output quality scores like factuality and relevance.

### 4.2 Designing Step-Specific Prompt Templates

A powerful design pattern for agentic systems is to decouple the task logic from the context assembly by using short, focused, step-specific prompt templates for each phase of the research loop. This approach leverages the fact that LLMs perform better on simpler, more focused tasks. It separates the "what to do" (the prompt) from the "what to know" (the context block), which improves modularity and maintainability. This allows a data science team to A/B test different retrieval strategies (modifying the context) without altering the prompts, while a prompt engineering team can refine instructional wording without needing to understand the complexities of the context pipeline.

Below are examples of such templates:
- **Planning Step Prompt**: "You are a meticulous research planner. Your task is to decompose the user's query into a series of logical, non-overlapping sub-questions that can be researched independently. Output a JSON list containing these sub-questions."
- **Search Step Prompt**: "You are a search query formulation expert. For the following sub-question: {{subQ}}, generate three diverse and effective search queries suitable for a web search engine. The queries should be designed to retrieve comprehensive and complementary information. Output a JSON list of the queries."
- **Read/Summarize Step Prompt**: "You are a research analyst. Your task is to read the following document text provided below. Identify and extract the key findings that are directly relevant to answering the question: {{subQ}}. Summarize these findings as a concise list of 5 bullet points. For each bullet point, you must cite the specific paragraph number(s) from the document from which the information is derived."
- **Synthesis Step Prompt**: "You are an expert research synthesizer. The following bulleted lists are summaries extracted from multiple source documents. Your task is to synthesize this information into a single, cohesive, and well-written paragraph that provides a comprehensive answer to the original user query. It is critical that every claim in your final answer is followed by its corresponding source citation, formatted as [source ID]."

### 4.3 Engineering for Modularity and Testability

To build a system that can evolve and be improved over time, it is essential to design for modularity and testability. The most effective way to achieve this is by architecting the pipeline using Dependency Injection (DI). Each component of the pipeline—the query classifier, the retriever, the context packager, the prompt templates, and the LLM itself—should be implemented as an injectable module. This design allows individual components to be easily swapped out for A/B testing, benchmarking, or upgrades without requiring changes to the core orchestration logic. For example, the team could test a new vector database or a different embedding model simply by injecting a new retriever module. Frameworks like crewAI and LangGraph are built on this principle, promoting the use of modular, interchangeable agents and tasks to construct complex workflows.

---

## Section 5: Evaluation and Observability

A system that cannot be measured cannot be reliably improved. For non-deterministic, complex systems like AI agents, a robust framework for evaluation and observability is not an afterthought but a foundational requirement for development, debugging, and production monitoring.

### 5.1 A Multi-Dimensional Evaluation Framework

Evaluating an agentic system requires a multi-layered strategy that mirrors its architecture. It is insufficient to simply evaluate the final answer; one must assess the performance of each key component—the retriever, the generator, and the end-to-end system—to effectively diagnose and isolate issues. A low "Groundedness" score in the final output, for instance, could be caused by a hallucinating generator or by a retriever that fed it irrelevant context. Only by measuring both can the true root cause be identified.

A comprehensive evaluation suite should include metrics across three dimensions:

#### 5.1.1 Context-Awareness Metrics

These metrics assess the quality of the context being fed to the LLM.
- **Context Utilization / Sufficiency**: This measures how effectively the generated response uses the relevant information that was provided in the context. It answers the question: "Of the relevant information retrieved, how much was actually used in the answer?" A low score indicates that the model ignored key pieces of data provided by the retriever.
- **Context Relevance / Precision**: This is the flip side of utilization, measuring the signal-to-noise ratio of the retrieved context itself. It answers the question: "Of all the context retrieved, how much of it was actually relevant to the user's query?" This metric evaluates the performance of the retriever.

#### 5.1.2 Generation Quality Metrics

These metrics assess the final output generated by the agent.
- **Groundedness / Faithfulness**: This is arguably the most critical metric for RAG systems. It evaluates how well the claims made in the generated answer are supported by the provided source context. It is a direct measure of factuality and is key to quantifying and mitigating hallucinations. This is often calculated using an LLM-as-a-judge approach, which breaks the response into individual claims and then verifies each claim against the source documents.
- **Answer Relevance**: This metric assesses how pertinent and on-topic the final answer is to the original user query, ensuring the agent has correctly understood and addressed the user's intent.

#### 5.1.3 Operational Metrics

These metrics measure the efficiency and cost-effectiveness of the system.
- **Latency & Throughput**: This includes time-to-first-token, total response time, and the number of requests the system can handle per second. These are critical for user experience.
- **Token Usage & Cost**: Diligently tracking the number of input and output tokens for each LLM call is essential for managing the operational costs of the application.

**Table 3: Key Evaluation Metrics for Agentic Systems**

| Metric Name          | What It Measures                                                                 | How It's Calculated (High-Level)                                                                 | Why It's Important                                                                 |
|----------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Context Precision   | The relevance of retrieved context (evaluates the Retriever).                    | The proportion of retrieved documents that are relevant to the query. Often calculated using LLM-as-a-judge to score each document's relevance. | A low score indicates the retriever is providing noisy or irrelevant information, which is a primary cause of hallucinations. |
| Context Utilization | The extent to which the final answer uses the provided context (evaluates the Generator). | The proportion of relevant information in the context that is actually reflected in the final answer. | A low score means the model is ignoring the provided context and likely relying on its parametric memory, defeating the purpose of RAG. |
| Groundedness / Faithfulness | The factual accuracy of the final answer against the provided context (evaluates the Generator). | Breaks the generated answer into individual claims and verifies each claim against the source context. The score is the ratio of supported claims. | This is the primary metric for measuring and reducing hallucinations. It ensures the agent's outputs are trustworthy and verifiable. |
| Answer Relevance    | The pertinence of the final answer to the user's query (evaluates the End-to-End System). | An LLM-as-a-judge scores the alignment between the user's query and the final generated answer on a numeric scale. | Ensures the agent has correctly understood and addressed the user's intent, rather than answering a related but different question. |

### 5.2 Debugging Workflows and Tooling

The non-deterministic and complex nature of agentic systems renders traditional debugging tools inadequate. This necessitates a new class of tools and workflows centered on the principle of **observability**. In a production environment, you cannot control, govern, or secure what you cannot see. Comprehensive tracing and logging are therefore not merely "nice-to-haves" for debugging; they are fundamental requirements for reliability, security, and governance.

- **Full-Trace Logging**: The cornerstone of agent debugging is the ability to capture and visualize a complete, step-by-step trace of every action the agent takes. A full trace should include every LLM call with the exact prompt and raw output, every tool invocation with its parameters and results, and every change to the agent's memory or state. Specialized platforms like LangSmith are designed specifically for this purpose, providing a visual "run tree" that allows developers to drill down into each step of a complex agent execution. For local development, enabling verbose=True or debug=True in frameworks like LangChain offers a command-line version of this detailed logging.
- **Advanced Debugging Techniques**: Beyond simple logging, several advanced techniques are crucial for diagnosing complex failures:
  - **Behavior Tracing**: This is the fundamental practice of logging every decision and its context to reconstruct the agent's behavior path, allowing for post-mortem analysis of failures.
  - **Time-Travel Debugging**: This involves recording and comparing snapshots of the system's state at different points in time. It is invaluable for identifying the root cause of regressions by comparing a failing run to a previous successful run.
  - **Intent Inference and Goal Tracking**: This technique involves monitoring the agent's high-level goals (as defined in its plan) and comparing them against its actual sequence of actions to detect when the agent has deviated from its intended path or entered an inefficient loop.
  - **Real-time Workflow Debugging**: The next generation of debugging tools, such as Dify 1.5.0, allows for interactive debugging of live workflows. Developers can inspect the values of all variables in real-time, modify them on the fly, and re-run individual nodes of the agentic graph without having to re-execute the entire, often expensive, upstream chain. This dramatically accelerates the iteration and debugging cycle.

---

## Section 6: Failure Mode Analysis and Mitigation

Even well-architected systems can fail. A key to building resilient agents is to anticipate common failure modes and implement robust mitigation strategies. Analysis of these failures reveals a common theme: seemingly distinct symptoms often stem from a single, systemic root cause—poor context management. A token limit crash is an obvious symptom of a flawed context assembler, but hallucinations, tool failures, and infinite loops are often more subtle manifestations of the same underlying disease. Effective debugging, therefore, begins not with tweaking the final prompt, but with a thorough inspection of the full context payload that was sent to the model.

This section provides a practical troubleshooting guide, mapping common symptoms to their likely causes and concrete solutions.

### 6.1 Symptom: Factual Hallucinations

- **Description**: The agent generates text that is factually incorrect, inconsistent with the provided sources, or entirely fabricated.
- **Likely Root Causes**:
  - **Context Failure**: The retriever either failed to find the relevant information or the correct information was retrieved but buried in a large volume of irrelevant noise, causing the model to overlook it.
  - **Knowledge Staleness**: The agent is relying on its static, pre-trained parametric knowledge because the RAG system did not provide up-to-date information.
  - **Prompt Ambiguity**: A vague or poorly structured prompt can lead the model to "fill in the gaps" with plausible-sounding but invented details.
- **Fixes and Mitigation Strategies**:
  - **(Context Engineering) Improve Retrieval**: The most effective solution is to enhance the RAG pipeline. This can involve tightening retrieval filters, increasing the number of retrieved documents (k), improving query rewriting and expansion, or implementing more advanced techniques like Contextual Retrieval, which enriches chunks with metadata before embedding.
  - **(Prompt Engineering) Improve Instructions**: Make prompts more specific. Explicitly instruct the model to cite its sources for every claim and to rely only on the information provided in the context. Adding a self-critique step (e.g., "Verify all facts against the provided sources before finalizing the answer") can also significantly reduce hallucinations.

### 6.2 Symptom: Function Call Omissions or Errors

- **Description**: The agent either fails to call a necessary tool or calls it with improperly formatted or incorrect parameters.
- **Likely Root Causes**:
  - **Poor Tool Definition**: The tool's name, description, or parameter schema within the prompt is unclear, inconsistent, or poorly documented, making it difficult for the model to understand its purpose or how to use it correctly.
  - **Context Window Overflow**: The tool's schema, which is critical for correct invocation, was pushed out of the limited context window by other, less important information like a long chat history.
- **Fixes and Mitigation Strategies**:
  - **(Prompt & Context Engineering) Enhance Prompt Clarity**: Ensure tool names and descriptions are crystal clear and accurately reflect their function. Use the Context Packager to prioritize tool schemas, ensuring they appear early in the prompt and are not evicted. Provide few-shot examples of correct tool usage to guide the model.
  - **(System Design) Implement Robust Error Handling**: The tool's code should validate all incoming parameters. If a call is invalid, it should return a clear, descriptive error message to the agent (e.g., "Error: Invalid value for parameter 'user_id'. Expected an integer."). The agent can often parse this feedback and self-correct on its next attempt.

### 6.3 Symptom: Infinite Reasoning Loops

- **Description**: The agent becomes stuck, repeating the same sequence of thoughts and actions without making progress toward its goal or reaching a termination condition.
- **Likely Root Causes**:
  - **Goal Amnesia**: The agent has lost the context of its overall goal or the specific termination criteria due to context window limitations. This is a form of memory failure.
  - **Flawed Reflection**: The agent's reasoning is flawed, causing it to believe it is making progress when it is not. It may fail to recognize that a tool's output is not providing any new, useful information.
  - **Unstable Tool Behavior**: Certain models may be inherently more prone to looping when a tool call does not return the expected information.
- **Fixes and Mitigation Strategies**:
  - **(System Design) Add Explicit Guardrails**: The agent orchestrator must enforce hard limits as a safety net. This includes a step counter or iteration limit (max_iterations) to forcibly terminate a run that exceeds a reasonable threshold.
  - **(Prompt Engineering) Improve Reflection Prompt**: Enhance the agent's self-assessment capabilities by adding an explicit instruction to its reasoning prompt, such as: "Before each new step, critically assess whether the last action yielded new and relevant information. If you are not making material progress towards the final goal, terminate the process and report your findings."
  - **(System Design) Implement Human-in-the-Loop**: For critical or complex workflows, design the agent to pause and request human input if it detects it is in a loop or unable to make progress on its own.

### 6.4 Symptom: Token Limit Crashes

- **Description**: The application throws an error because the size of the final assembled context exceeds the model's maximum context window.
- **Likely Root Causes**:
  - **Unbounded Context Assembly**: The Context Packager is naively concatenating all available context (chat history, RAG results, tool outputs) without a defined budget or prioritization scheme.
  - **Conversational Drift**: In a long-running conversation, the accumulated history naturally grows to exceed the context limit.
- **Fixes and Mitigation Strategies**:
  - **(Context Engineering) Implement a Budgeted Context Assembler**: The Context Packager must be designed with an explicit token budget. It should have a clear prioritization hierarchy (e.g., system prompt and tool schemas are most important) and defined strategies for what to do when the budget is exceeded.
  - **(Context Engineering) Use Compression and Summarization Fallbacks**: When the context is too large, the packager should trigger a fallback strategy. This could involve using map-reduce summarization on long documents or the conversational history to create a more compact representation.
  - **(Context Engineering) Apply Strategic Truncation**: Implement intelligent truncation methods, such as a sliding window that keeps only the most recent turns of a conversation, or a method that summarizes the oldest parts of the chat history while keeping recent turns detailed.

The analysis of these failure modes reinforces a critical principle: a production-ready agent is not one that never makes a mistake, but one that is architected to detect, recover from, and learn from its mistakes. This requires building explicit guardrails, validation checks, and robust error-handling feedback loops directly into the agentic workflow. Reliability is an emergent property of a system designed for resilience.

**Table 4: Troubleshooting Guide for Agentic Research Systems**

| Symptom                  | Likely Root Cause(s)                                                                 | Recommended Fixes (Context Engineering)                                                                 | Recommended Fixes (Prompt Engineering)                                                                 |
|--------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Hallucinated Facts      | Missing or low-relevance context from retriever; outdated parametric knowledge.      | Improve retriever quality (increase k, better chunking, query rewriting); implement cite-aware RAG to enforce grounding. | Add self-critique step ("Verify facts against sources"); instruct model to admit uncertainty; use Chain-of-Thought for logical consistency. |
| Function Call Omissions / Errors | Tool schema is unclear or was pushed out of the context window; inconsistent tool definitions. | Prioritize tool schemas in the context packager; ensure tool definitions are consistent across the system. | Make tool names and descriptions unambiguous; provide clear few-shot examples of correct tool invocation syntax. |
| Infinite Reasoning Loop | Agent forgets termination criteria (memory failure); flawed reflection mechanism; unstable tool behavior. | Add a step counter/iteration limit as a hard guardrail; implement human-in-the-loop escalation.           | Add explicit instruction to stop if no new, relevant information is being gathered; refine the agent's overall goal and plan. |
| Token Limit Crashes     | Context assembler is naively concatenating context without a budget; unbounded chat history. | Implement a token budget allocator; use summarization (e.g., map-reduce) or compression (e.g., semantic hashing) as a fallback for oversized context. | (N/A - This is fundamentally a context engineering failure).                                           |

---

## Conclusion and Future Directions

The successful development of robust, scalable, and reliable agentic AI systems hinges on a paradigm shift away from isolated prompt tweaking and towards a holistic, architectural approach. The intertwined disciplines of Context Engineering and Prompt Engineering provide the necessary framework for this evolution. By mastering the principles and patterns outlined in this report, engineering teams can move from building fragile demos to deploying production-grade systems that can plan, search, reason, and write with minimal hallucinations and maximum maintainability.

### Key Takeaways: The Principles of Resilient Agent Design

The analysis yields four core principles that should guide the design of any advanced agentic system:
1. **Treat Context as a Managed Resource**: The LLM's context window is not an infinite scratchpad but a constrained and expensive resource, analogous to RAM in a traditional operating system. Resilient architectures must therefore be designed with explicit, upfront strategies for context budgeting, prioritization, compression, and eviction.
2. **Separate Context from Instruction**: A clean separation between what the model needs to know (the context, managed by the context pipeline) and how it should behave (the instructions, defined in prompt templates) is paramount. This decoupling makes both the data pipeline and the agent's logic more modular, independently testable, and easier to maintain and optimize.
3. **Instrument Everything for Observability**: You cannot improve, debug, or govern what you cannot measure. Implementing comprehensive, end-to-end logging and evaluation for token counts, retrieval metrics, latencies, and output quality scores is not optional. It is the foundational requirement for the iterative hardening of the agentic research loop.
4. **Build for Modularity and Testability**: Architect the system using dependency injection and modular components. This ensures that every part of the system—retrievers, compressors, prompt templates, and even the core LLMs—can be independently A/B tested, benchmarked, and upgraded without disrupting the entire orchestration layer.

### Future Outlook: The Next Wave of Agentic Systems

The field of agentic AI is advancing at a rapid pace. Several emerging trends, hinted at throughout this report, are poised to define the next generation of these systems:
- **Multi-Agent Systems**: The future lies in collaborative ecosystems where multiple, specialized agents delegate tasks to one another. A "lead researcher" agent might orchestrate a team of "search agents," "data analysis agents," and "report writing agents," each with its own specialized tools and prompts. This modular approach promises to handle even greater complexity.
- **Automated Optimization**: The processes of prompt engineering and even workflow design are themselves becoming targets for AI automation. We are seeing the emergence of LLMs used to automatically generate and optimize prompts, and even to suggest more efficient agentic workflows based on performance data.
- **Standardized Communication**: As multi-agent systems become more common, the need for interoperability will become critical. Emerging standards like Agent Communication Protocols (ACP) aim to create a universal language for agents to communicate and collaborate, regardless of their underlying framework or provider.
- **Evolving Long-Context Strategies**: As foundational models with ever-larger context windows become available, the specific tactics of context engineering will evolve. While the need for some form of RAG will likely persist for accessing real-time or proprietary data, the trade-offs between retrieval, in-context learning, and advanced compression techniques will continue to shift, requiring ongoing research and re-evaluation of architectural best practices.

By jointly mastering the crafts of Context and Prompt Engineering, developers and architects will be well-equipped to not only navigate the challenges of today but also to build the more sophisticated, collaborative, and autonomous AI systems of tomorrow.

---

## Appendix: Related Notes

### Building Effective AI Agents

The note [[Building Effective AI Agents]] from Anthropic discusses the distinction between **workflows** and **agents**:
- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents** are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

The note emphasizes the importance of starting with simple solutions and only increasing complexity when necessary. It also highlights the use of frameworks like LangGraph, Amazon Bedrock, Rivet, and Vellum, while cautioning against over-reliance on them due to potential abstraction layers that can obscure debugging.

### Context Engineering - Short-Term Memory Management with Sessions from OpenAI Agents SDK

The note [[Context Engineering - Short-Term Memory Management with Sessions from OpenAI Agents SDK  OpenAI Cookbook 1]] explores **context trimming** and **context summarization** as two proven techniques for managing context effectively using the `Session` object from the OpenAI Agents SDK. It provides practical examples and code snippets for implementing these techniques, emphasizing the importance of maintaining coherence, reducing latency and cost, and improving tool-call accuracy.

### Context Engineering Strategies for Supplying Agents with Essential Information

The note [[Context Engineering Strategies for Supplying Agents with Essential Information]] categorizes context engineering strategies into four main buckets: **write, select, compress, and isolate**. It discusses various techniques such as scratchpads, memories, RAG, summarization, trimming, and multi-agent systems, and explains how LangGraph and LangSmith support these strategies.

---

## Works Cited

1. [2507.13334] A Survey of Context Engineering for Large Language Models - arXiv, accessed July 19, 2025, https://arxiv.org/abs/2507.13334
2. A Survey of Context Engineering for Large Language Models - arXiv, accessed July 19, 2025, https://arxiv.org/html/2507.13334v1
3. Context Engineering vs Prompt Engineering | by Mehul Gupta | Data Science in Your Pocket, accessed July 19, 2025, https://medium.com/data-science-in-your-pocket/context-engineering-vs-prompt-engineering-379e9622e19d
4. 10 Best Practices for Prompt Engineering with Any Model - PromptHub, accessed July 19, 2025, https://www.prompthub.us/blog/10-best-practices-for-prompt-engineering-with-any-model
5. How we built our multi-agent research system - Anthropic, accessed July 19, 2025, https://www.anthropic.com/engineering/built-multi-agent-research-system
6. What is Agentic AI? | UiPath, accessed July 19, 2025, https://www.uipath.com/ai/agentic-ai
7. Prompt Engineer vs Context Engineer: Why Design Leadership Needs to See the Bigger Picture | by Elizabeth Eagle-Simbeye | Bootcamp | Jul, 2025 | Medium, accessed July 19, 2025, https://medium.com/design-bootcamp/prompt-engineer-vs-context-engineer-why-design-leadership-needs-to-see-the-bigger-picture-24eec7ea9a91
8. Prompt Engineering for AI Guide | Google Cloud, accessed July 19, 2025, https://cloud.google.com/discover/what-is-prompt-engineering
9. Best Practices for Building Prompt Templates - Salesforce Help, accessed July 19, 2025, https://help.salesforce.com/s/articleView?id=sf.prompt_builder_best_practices.htm&language=en_US&type=5
10. How to Write AI Prompts for Project Managers - Smartsheet, accessed July 19, 2025, https://www.smartsheet.com/content/ai-prompts-project-management
11. Advanced Prompt Engineering Techniques - Mercity AI, accessed July 19, 2025, https://www.mercity.ai/blog-post/advanced-prompt-engineering-techniques
12. Prompt Engineering for AI Agents - PromptHub, accessed July 19, 2025, https://www.prompthub.us/blog/prompt-engineering-for-ai-agents
13. AI Agents in LangGraph - DeepLearning.AI - Learning Platform, accessed July 19, 2025, https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/qyrpc/introduction
14. Building Agentic RAG with LlamaIndex - DeepLearning.AI, accessed July 19, 2025, https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/
15. What Are Agentic Workflows? Patterns, Use Cases, Examples, and More | Weaviate, accessed July 19, 2025, https://weaviate.io/blog/what-are-agentic-workflows
16. How to Build AI Agents Using Plan-and-Execute Loops - WillowTree Apps, accessed July 19, 2025, https://www.willowtreeapps.com/craft/building-ai-agents-with-plan-and-execute
17. Designing Agentic AI Systems, Part 1: Agent Architectures – Vectorize, accessed July 19, 2025, https://vectorize.io/designing-agentic-ai-systems-part-1-agent-architectures/
18. The Need to Improve Long-Term Memory in LLM-Agents, accessed July 19, 2025, https://ojs.aaai.org/index.php/AAAI-SS/article/download/27688/27461/31739
19. Retrieval Augmented Generation (RAG) for LLMs - Prompt Engineering Guide, accessed July 19, 2025, https://www.promptingguide.ai/research/rag
20. RAG vs. CAG: A Deep Dive into Context-Aware AI Generation Techniques - DZone, accessed July 19, 2025, https://dzone.com/articles/rag-vs-cag-context-aware-ai
21. Improve Retrieval Augmented Generation Through Classification | A ..., accessed July 19, 2025, https://www.a-cx.com/improving-retrieval-augmented-generation/
22. medium.com, accessed July 19, 2025, https://medium.com/@shravankoninti/mastering-rag-a-deep-dive-into-retriever-2ac7957106b7#:~:text=Similarity%20Search%3A%20Retrieves%20documents%20based,redundancy%20and%20ensuring%20diverse%20results.
23. 2.5 Semantic Search. Advanced Retrieval Strategies - LLMOps ..., accessed July 19, 2025, https://boramorka.github.io/LLM-Book/CHAPTER-2/Answers%202.5/
24. Effective Source Tracking in RAG Systems - Chitika, accessed July 19, 2025, https://www.chitika.com/source-tracking-rag/
25. LLM Source Citation with Langchain - Ready Tensor, accessed July 19, 2025, https://app.readytensor.ai/publications/llm-source-citation-with-langchain-dXocR7whrOTR
26. Build smarter AI agents: Manage short-term and long-term memory ..., accessed July 19, 2025, https://redis.io/blog/build-smarter-ai-agents-manage-short-term-and-long-term-memory-with-redis/
27. What is LLM's Context Window?:Understanding and Working with the Context Window | by Tahir | Medium, accessed July 19, 2025, https://medium.com/@tahirbalarabe2/what-is-llms-context-window-understanding-and-working-with-the-context-window-641b6d4f811f
28. LLM Context Windows: Basics, Examples & Prompting Best Practices, accessed July 19, 2025, https://swimm.io/learn/large-language-models/llm-context-windows-basics-examples-and-prompting-best-practices
29. How to summarize text through parallelization | 🦜️ LangChain, accessed July 19, 2025, https://python.langchain.com/docs/how_to/summarize_map_reduce/
30. How to use LLMs: Summarize long documents : r/PromptEngineering - Reddit, accessed July 19, 2025, https://www.reddit.com/r/PromptEngineering/comments/1cfxd6y/how_to_use_llms_summarize_long_documents/
31. What is Semantic Hashing - Activeloop, accessed July 19, 2025, https://www.activeloop.ai/resources/glossary/semantic-hashing/
32. HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation - arXiv, accessed July 19, 2025, https://arxiv.org/html/2505.16133v1
33. ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference - Hugging Face, accessed July 19, 2025, https://huggingface.co/papers/2502.00299
34. Extending Context Window of Large Language Models via Semantic Compression - ACL Anthology, accessed July 19, 2025, https://aclanthology.org/2024.findings-acl.306.pdf
35. Optimizing LLM Costs: A Comprehensive Analysis of Context ..., accessed July 19, 2025, https://phase2online.com/2025/04/28/optimizing-llm-costs-with-context-caching/
36. Epic: Efficient Position-Independent Context Caching for Serving Large Language Models, accessed July 19, 2025, https://arxiv.org/html/2410.15332v1
37. How to build your Agent: 11 prompting techniques for better AI ..., accessed July 19, 2025, https://www.augmentcode.com/blog/how-to-build-your-agent-11-prompting-techniques-for-better-ai-agents
38. How to Prevent LLM Hallucinations: 5 Proven Strategies - Voiceflow, accessed July 19, 2025, https://www.voiceflow.com/blog/prevent-llm-hallucinations
39. Tools | 🦜️ LangChain, accessed July 19, 2025, https://python.langchain.com/docs/concepts/tools/
40. Build agentic systems with CrewAI and Amazon Bedrock | Artificial Intelligence - AWS, accessed July 19, 2025, https://aws.amazon.com/blogs/machine-learning/build-agentic-systems-with-crewai-and-amazon-bedrock/
41. Top Tools & Techniques for Debugging Agentic AI Systems, accessed July 19, 2025, https://www.amplework.com/blog/debugging-agentic-ai-tools-techniques/
42. Chain complex prompts for stronger performance - Anthropic API, accessed July 19, 2025, https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts
43. Mastering AI-Powered Research: My Guide to Deep Research, Prompt Engineering, and Multi-Step Workflows : r/ChatGPTPro - Reddit, accessed July 19, 2025, https://www.reddit.com/r/ChatGPTPro/comments/1in87ic/mastering_aipowered_research_my_guide_to_deep/
44. Why your AI Agents Need Prompt Templates And How to Use Them Effectively | by Mariem Jabloun | Towards Dev - Medium, accessed July 19, 2025, https://medium.com/towardsdev/why-your-ai-agents-need-prompt-templates-and-how-to-use-them-effectively-d6ae235c2bb0
45. A practical guide to building agents - OpenAI, accessed July 19, 2025, https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
46. Exercise 2: Ground a Prompt with Einstein Search | Salesforce Developers, accessed July 19, 2025, https://developer.salesforce.com/agentforce-workshop/rag/2-einstein-search
47. Use prompt templates | Generative AI on Vertex AI - Google Cloud, accessed July 19, 2025, https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-templates
48. A Complete Guide to Meta Prompting - PromptHub, accessed July 19, 2025, https://www.prompthub.us/blog/a-complete-guide-to-meta-prompting
49. langchain-ai/rag-research-agent-template - GitHub, accessed July 19, 2025, https://github.com/langchain-ai/rag-research-agent-template
50. I Created a Prompt That Turns Research Headaches Into Breakthroughs : r/PromptEngineering - Reddit, accessed July 19, 2025, https://www.reddit.com/r/PromptEngineering/comments/1i12zew/i_created_a_prompt_that_turns_research_headaches/
51. A Systematic Prompt Template Analysis for Real-world LLMapps - arXiv, accessed July 19, 2025, https://arxiv.org/html/2504.02052v2
52. The Art and Science of RAG: Mastering Prompt Templates and Contextual Understanding, accessed July 19, 2025, https://medium.com/@ajayverma23/the-art-and-science-of-rag-mastering-prompt-templates-and-contextual-understanding-a47961a57e27
53. Context Utilization - UpTrain, accessed July 19, 2025, https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-utilization
54. LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide - Confident AI, accessed July 19, 2025, https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation
55. Building an LLM evaluation framework: best practices - Datadog, accessed July 19, 2025, https://www.datadoghq.com/blog/llm-evaluation-framework-best-practices/
56. RAG Evaluation Metrics: Best Practices for Evaluating RAG Systems - Patronus AI, accessed July 19, 2025, https://www.patronus.ai/llm-testing/rag-evaluation-metrics
57. Mastering RAG Evaluation: Best Practices & Tools for 2025 - Orq.ai, accessed July 19, 2025, https://orq.ai/blog/rag-evaluation
58. Monitoring evaluation metrics descriptions and use cases (preview ..., accessed July 19, 2025, https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-model-monitoring-generative-ai-evaluation-metrics?view=azureml-api-2
59. LLM Evaluation Metrics Every Developer Should Know - Comet, accessed July 19, 2025, https://www.comet.com/site/blog/llm-evaluation-metrics-every-developer-should-know/
60. 7 Key LLM Metrics to Enhance AI Reliability | Galileo, accessed July 19, 2025, https://galileo.ai/blog/llm-performance-metrics
61. LLM Metrics: Key Metrics Explained - Iguazio, accessed July 19, 2025, https://www.iguazio.com/blog/llm-metrics-key-metrics-explained/
62. Exploring Agentic AI Systems: A Hands-On Guide to Building Secure Agent Workflows, accessed July 19, 2025, https://www.arthur.ai/blog/exploring-agentic-ai-systems
63. LangSmith - LangChain, accessed July 19, 2025, https://www.langchain.com/langsmith
64. Mastering Debugging in LangChain: Essential Tips & Tricks - Arsturn, accessed July 19, 2025, https://www.arsturn.com/blog/debugging-langchain-tips-tricks
65. How to debug your LLM apps - LangChain.js, accessed July 19, 2025, https://js.langchain.com/docs/how_to/debugging
66. How to debug your LLM apps - 🦜️ LangChain, accessed July 19, 2025, https://python.langchain.com/docs/how_to/debugging/
67. Dify 1.5.0: Real-Time Workflow Debugging That Actually Works ..., accessed July 19, 2025, https://dify.ai/blog/dify-1-5-0-real-time-workflow-debugging-that-actually-works
68. Understanding LLM Hallucinations. Causes, Detection, Prevention, and Ethical Concerns, accessed July 19, 2025, https://medium.com/@tam.tamanna18/understanding-llm-hallucinations-causes-detection-prevention-and-ethical-concerns-914bc89128d0
69. LLM Hallucinations 101 - Neptune.ai, accessed July 19, 2025, https://neptune.ai/blog/llm-hallucinations
70. Introducing Contextual Retrieval - Anthropic, accessed July 19, 2025, https://www.anthropic.com/news/contextual-retrieval
71. Agent can't stop on function calls and keeps iterating in an infinite ..., accessed July 19, 2025, https://github.com/langchain-ai/langchain/discussions/18279
72. The unreasonable effectiveness of an LLM agent loop with tool use | Hacker News, accessed July 19, 2025, https://news.ycombinator.com/item?id=43998472
73. Deepseek LLM infinite loop on tool usage - Questions - n8n Community, accessed July 19, 2025, https://community.n8n.io/t/deepseek-llm-infinite-loop-on-tool-usage/69191
74. Semantic Compression With Large Language Models, accessed July 19, 2025, https://www.dre.vanderbilt.edu/~schmidt/PDF/Compression_with_LLMs.pdf
75. All Courses - DeepLearning.AI, accessed July 19, 2025, https://www.deeplearning.ai/courses/?courses_date_desc%5B
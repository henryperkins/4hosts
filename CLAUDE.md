# CodeViz Research Context

> **Note**: This file contains research context from CodeViz. Most recent contexts are at the bottom.

---

## Research Query

Investigate the "Research Orchestration & AI" component in detail. Focus your analysis on these key files:
1. paradigm_search.py (4hosts/four-hosts-app/backend/services/paradigm_search.py)
2. answer_generator_continued.py (4hosts/four-hosts-app/backend/services/answer_generator_continued.py)
3. search_apis.py (4hosts/four-hosts-app/backend/services/search_apis.py)
4. classification_engine.py (4hosts/four-hosts-app/backend/services/classification_engine.py)
5. llm_client.py (4hosts/four-hosts-app/backend/services/llm_client.py)
6. context_engineering.py (4hosts/four-hosts-app/backend/services/context_engineering.py)
7. answer_generator.py (4hosts/four-hosts-app/backend/services/answer_generator.py)
8. research_orchestrator.py (4hosts/four-hosts-app/backend/services/research_orchestrator.py)

Provide insights about the component's main responsibilities and functionality and how it integrates with other system components

*Session: fc298aa118929dccbee84704f35270e7 | Generated: 7/28/2025, 6:56:49 AM*

### Analysis Summary

# Research Orchestration & AI Component Analysis

The "Research Orchestration & AI" component is a sophisticated system designed to conduct targeted research and generate nuanced answers based on distinct "host paradigms." It integrates query classification, context engineering, multi-API search, and AI-driven content generation to provide comprehensive and paradigm-aligned research outputs.

## Core Responsibilities and Functionality

This component's main responsibilities include:
*   **Query Classification**: Understanding the user's intent and aligning it with one or more host paradigms.
*   **Context Engineering**: Transforming the classified query into a detailed research plan, optimizing for search and synthesis.
*   **Multi-API Search**: Executing searches across various external data sources, applying paradigm-specific strategies.
*   **Answer Generation**: Synthesizing search results into structured, paradigm-aligned answers using Large Language Models (LLMs).
*   **Orchestration & Integration**: Managing the flow between these sub-components, ensuring seamless execution and data transfer.

## Component Breakdown

### **1. Paradigm-Specific Search Strategies**
The [paradigm_search.py](four-hosts-app/backend/services/paradigm_search.py) module implements specialized search approaches tailored to each of the four host paradigms: Dolores (revolutionary), Teddy (devotion), Bernard (analytical), and Maeve (strategic).

*   **Purpose**: To generate and refine search queries, and filter/rank search results according to the specific priorities and biases of a given paradigm.
*   **Internal Parts**:
    *   `DoloresSearchStrategy`: Focuses on investigative journalism, exposing systemic issues.
    *   `TeddySearchStrategy`: Emphasizes community support and care resources.
    *   `BernardSearchStrategy`: Prioritizes academic research and data.
    *   `MaeveSearchStrategy`: Concentrates on business intelligence and strategic insights.
    *   `SearchContext`: A data class [SearchContext](four-hosts-app/backend/services/paradigm_search.py:30) that encapsulates the original query, primary and secondary paradigms, and user preferences for search.
    *   `get_search_strategy()`: A factory function [get_search_strategy](four-hosts-app/backend/services/paradigm_search.py:300) to retrieve the appropriate strategy instance.
*   **External Relationships**:
    *   **Uses**: [search_apis.py](four-hosts-app/backend/services/search_apis.py) for executing the actual searches.
    *   **Uses**: [credibility.py](four-hosts-app/backend/services/credibility.py) to assess source credibility, influencing result ranking.
    *   **Used by**: [research_orchestrator.py](four-hosts-app/backend/services/research_orchestrator.py) to apply paradigm-specific search logic during research execution.

### **2. Answer Generation (Base and Continued)**
The answer generation system is split across two files:
*   [answer_generator.py](four-hosts-app/backend/services/answer_generator.py)
*   [answer_generator_continued.py](four-hosts-app/backend/services/answer_generator_continued.py)

#### **2.1. Base Answer Generation**
The [answer_generator.py](four-hosts-app/backend/services/answer_generator.py) module defines the foundational components for generating paradigm-specific answers.

*   **Purpose**: To provide an abstract framework for answer generation and implement the Dolores and Teddy paradigm-specific generators.
*   **Internal Parts**:
    *   `BaseAnswerGenerator`: An abstract base class [BaseAnswerGenerator](four-hosts-app/backend/services/answer_generator.py:60) defining the common interface for all paradigm generators.
    *   `DoloresAnswerGenerator`: Implements the revolutionary paradigm's answer generation logic [DoloresAnswerGenerator](four-hosts-app/backend/services/answer_generator.py:100), focusing on exposing injustices.
    *   `TeddyAnswerGenerator`: Implements the devotion paradigm's answer generation logic [TeddyAnswerGenerator](four-hosts-app/backend/services/answer_generator.py:300), focusing on support and care.
    *   `Citation`: A data class [Citation](four-hosts-app/backend/services/answer_generator.py:25) for managing source citations.
    *   `AnswerSection`: A data class [AnswerSection](four-hosts-app/backend/services/answer_generator.py:36) representing a section of the generated answer.
    *   `GeneratedAnswer`: A data class [GeneratedAnswer](four-hosts-app/backend/services/answer_generator.py:45) encapsulating the complete generated answer.
    *   `SynthesisContext`: A data class [SynthesisContext](four-hosts-app/backend/services/answer_generator.py:54) providing context for answer synthesis.
*   **External Relationships**:
    *   **Uses**: [llm_client.py](four-hosts-app/backend/services/llm_client.py) for generating content using LLMs.
    *   **Uses**: [cache.py](four-hosts-app/backend/services/cache.py) and [credibility.py](four-hosts-app/backend/services/credibility.py) for managing and assessing source information.
    *   **Used by**: [answer_generator_continued.py](four-hosts-app/backend/services/answer_generator_continued.py) to extend its functionality.

#### **2.2. Continued Answer Generation & Orchestration**
The [answer_generator_continued.py](four-hosts-app/backend/services/answer_generator_continued.py) module extends the answer generation system with Bernard and Maeve generators and introduces the main orchestration logic.

*   **Purpose**: To provide the Bernard and Maeve paradigm-specific answer generators and orchestrate the overall answer generation process, including multi-paradigm synthesis.
*   **Internal Parts**:
    *   `BernardAnswerGenerator`: Implements the analytical paradigm's answer generation logic [BernardAnswerGenerator](four-hosts-app/backend/services/answer_generator_continued.py:15), focusing on empirical evidence.
    *   `MaeveAnswerGenerator`: Implements the strategic paradigm's answer generation logic [MaeveAnswerGenerator](four-hosts-app/backend/services/answer_generator_continued.py:240), focusing on actionable strategies.
    *   `AnswerGenerationOrchestrator`: The central class [AnswerGenerationOrchestrator](four-hosts-app/backend/services/answer_generator_continued.py:470) that manages all individual paradigm generators and handles single or multi-paradigm answer requests.
*   **External Relationships**:
    *   **Extends**: `BaseAnswerGenerator` from [answer_generator.py](four-hosts-app/backend/services/answer_generator.py).
    *   **Uses**: [llm_client.py](four-hosts-app/backend/services/llm_client.py) for all LLM interactions.
    *   **Uses**: `mesh_network_service` from [mesh_network.py](four-hosts-app/backend/services/mesh_network.py) for integrating multi-paradigm answers.
    *   **Used by**: The main application flow to generate final research answers.

### **3. Search API Integrations**
The [search_apis.py](four-hosts-app/backend/services/search_apis.py) module provides a unified interface for interacting with various external search engines.

*   **Purpose**: To abstract away the complexities of different search APIs, providing a standardized way to retrieve search results.
*   **Internal Parts**:
    *   `SearchResult`: A data class [SearchResult](four-hosts-app/backend/services/search_apis.py:25) to standardize the format of search results.
    *   `SearchConfig`: A data class [SearchConfig](four-hosts-app/backend/services/search_apis.py:40) for configuring search requests.
    *   `RateLimiter`: A utility class [RateLimiter](four-hosts-app/backend/services/search_apis.py:49) to manage API call rates.
    *   `BaseSearchAPI`: An abstract base class [BaseSearchAPI](four-hosts-app/backend/services/search_apis.py:70) for all search API implementations.
    *   `GoogleCustomSearchAPI`: Implementation for Google Custom Search [GoogleCustomSearchAPI](four-hosts-app/backend/services/search_apis.py:85).
    *   `ArxivAPI`: Implementation for academic paper search on ArXiv [ArxivAPI](four-hosts-app/backend/services/search_apis.py:125).
    *   `BraveSearchAPI`: Implementation for Brave Search [BraveSearchAPI](four-hosts-app/backend/services/search_apis.py:165).
    *   `PubMedAPI`: Implementation for medical and life science papers on PubMed [PubMedAPI](four-hosts-app/backend/services/search_apis.py:280).
    *   `SearchAPIManager`: Manages multiple search APIs, handling failover and aggregation [SearchAPIManager](four-hosts-app/backend/services/search_apis.py:370).
    *   `create_search_manager()`: A factory function [create_search_manager](four-hosts-app/backend/services/search_apis.py:440) to initialize the `SearchAPIManager` with configured APIs.
*   **External Relationships**:
    *   **Used by**: [paradigm_search.py](four-hosts-app/backend/services/paradigm_search.py) to execute paradigm-specific searches.
    *   **Used by**: [research_orchestrator.py](four-hosts-app/backend/services/research_orchestrator.py) as the primary interface for external data retrieval.

### **4. Classification Engine**
The [classification_engine.py](four-hosts-app/backend/services/classification_engine.py) module is responsible for classifying user queries into one or more host paradigms.

*   **Purpose**: To accurately determine the underlying intent and perspective of a user's query, enabling paradigm-specific processing.
*   **Internal Parts**:
    *   `HostParadigm`: An Enum [HostParadigm](four-hosts-app/backend/services/classification_engine.py:25) defining the four consciousness paradigms (Dolores, Teddy, Bernard, Maeve).
    *   `QueryFeatures`: A data class [QueryFeatures](four-hosts-app/backend/services/classification_engine.py:33) for extracted query features.
    *   `ParadigmScore`: A data class [ParadigmScore](four-hosts-app/backend/services/classification_engine.py:42) for scoring paradigms.
    *   `ClassificationResult`: A data class [ClassificationResult](four-hosts-app/backend/services/classification_engine.py:50) for the complete classification output.
    *   `QueryAnalyzer`: Analyzes queries to extract features like tokens, entities, intent signals, and domain [QueryAnalyzer](four-hosts-app/backend/services/classification_engine.py:65).
    *   `ParadigmClassifier`: Classifies queries using a hybrid approach combining rule-based methods and LLM inference [ParadigmClassifier](four-hosts-app/backend/services/classification_engine.py:220).
*   **External Relationships**:
    *   **Uses**: [llm_client.py](four-hosts-app/backend/services/llm_client.py) for LLM-based classification when enabled.
    *   **Used by**: [context_engineering.py](four-hosts-app/backend/services/context_engineering.py) as the initial step in the context engineering pipeline.

### **5. LLM Client**
The [llm_client.py](four-hosts-app/backend/services/llm_client.py) module provides a unified client for interacting with various Large Language Models.

*   **Purpose**: To abstract LLM interactions, providing a consistent interface for generating text, structured outputs, and handling tool calls across different LLM providers (Azure OpenAI, OpenAI).
*   **Internal Parts**:
    *   `LLMClient`: The main client class [LLMClient](four-hosts-app/backend/services/llm_client.py:60) managing connections and requests to LLMs.
    *   `_PARADIGM_MODEL_MAP`: Maps paradigms to specific LLM models [llm_client.py](four-hosts-app/backend/services/llm_client.py:35).
    *   `_SYSTEM_PROMPTS`: Stores system prompts tailored for each paradigm [llm_client.py](four-hosts-app/backend/services/llm_client.py:42).
    *   `generate_completion()`: A core method [generate_completion](four-hosts-app/backend/services/llm_client.py:105) for chat-style completions.
    *   `generate_structured_output()`: A method [generate_structured_output](four-hosts-app/backend/services/llm_client.py:200) for generating JSON output based on a schema.
    *   `generate_with_tools()`: A method [generate_with_tools](four-hosts-app/backend/services/llm_client.py:217) for enabling LLM tool calling.
*   **External Relationships**:
    *   **Used by**: [answer_generator.py](four-hosts-app/backend/services/answer_generator.py) and [answer_generator_continued.py](four-hosts-app/backend/services/answer_generator_continued.py) for all content generation.
    *   **Used by**: [classification_engine.py](four-hosts-app/backend/services/classification_engine.py) for LLM-based query classification.

### **6. Context Engineering Pipeline**
The [context_engineering.py](four-hosts-app/backend/services/context_engineering.py) module implements the W-S-C-I (Write-Select-Compress-Isolate) pipeline, which refines a classified query into a detailed research execution plan.

*   **Purpose**: To transform a high-level classified query into a precise, actionable context for research and answer generation, ensuring paradigm alignment throughout the process.
*   **Internal Parts**:
    *   `ContextLayer`: An abstract base class [ContextLayer](four-hosts-app/backend/services/context_engineering.py:59) for pipeline layers.
    *   `WriteLayer`: Documents the research focus and themes based on the paradigm [WriteLayer](four-hosts-app/backend/services/context_engineering.py:75).
    *   `SelectLayer`: Selects appropriate search methods, sources, and generates specific search queries [SelectLayer](four-hosts-app/backend/services/context_engineering.py:190).
    *   `CompressLayer`: Determines compression strategies and token budgets for information processing [CompressLayer](four-hosts-app/backend/services/context_engineering.py:300).
    *   `IsolateLayer`: Defines criteria and patterns for extracting key findings [IsolateLayer](four-hosts-app/backend/services/context_engineering.py:390).
    *   `ContextEngineeredQuery`: A data class [ContextEngineeredQuery](four-hosts-app/backend/services/context_engineering.py:49) representing the output of the entire pipeline.
    *   `ContextEngineeringPipeline`: The main orchestrator [ContextEngineeringPipeline](four-hosts-app/backend/services/context_engineering.py:500) that runs the query through all W-S-C-I layers.
*   **External Relationships**:
    *   **Uses**: [classification_engine.py](four-hosts-app/backend/services/classification_engine.py) to receive the initial classified query.
    *   **Used by**: [research_orchestrator.py](four-hosts-app/backend/services/research_orchestrator.py) to obtain the detailed research plan before executing searches.

### **7. Research Orchestrator**
The [research_orchestrator.py](four-hosts-app/backend/services/research_orchestrator.py) module is the central component that ties together context engineering, search execution, and result processing.

*   **Purpose**: To execute the research process end-to-end, from receiving a context-engineered query to delivering filtered and credible search results.
*   **Internal Parts**:
    *   `ResearchExecutionResult`: A data class [ResearchExecutionResult](four-hosts-app/backend/services/research_orchestrator.py:20) for the complete research execution output.
    *   `ResultDeduplicator`: Removes duplicate search results [ResultDeduplicator](four-hosts-app/backend/services/research_orchestrator.py:35).
    *   `CostMonitor`: Tracks API costs and provides budget alerts [CostMonitor](four-hosts-app/backend/services/research_orchestrator.py:120).
    *   `ParadigmAwareSearchOrchestrator`: The main orchestrator class [ParadigmAwareSearchOrchestrator](four-hosts-app/backend/services/research_orchestrator.py:170) that manages the research flow.
*   **External Relationships**:
    *   **Receives input from**: The output of [context_engineering.py](four-hosts-app/backend/services/context_engineering.py) (`ContextEngineeredQuery`).
    *   **Uses**: [search_apis.py](four-hosts-app/backend/services/search_apis.py) for executing external searches.
    *   **Uses**: [paradigm_search.py](four-hosts-app/backend/services/paradigm_search.py) to apply paradigm-specific search strategies and result ranking.
    *   **Uses**: [credibility.py](four-hosts-app/backend/services/credibility.py) for assessing source credibility.
    *   **Uses**: [cache.py](four-hosts-app/backend/services/cache.py) for caching search results and tracking costs.
    *   **Outputs to**: The answer generation system (e.g., [answer_generator_continued.py](four-hosts-app/backend/services/answer_generator_continued.py)) for synthesis.

## Integration and Flow

The "Research Orchestration & AI" component operates as a pipeline:

1.  **Query Input**: A user query is received.
2.  **Classification**: The [classification_engine.py](four-hosts-app/backend/services/classification_engine.py) analyzes the query and determines its primary (and potentially secondary) host paradigm. This step may involve the [llm_client.py](four-hosts-app/backend/services/llm_client.py) for LLM-based classification.
3.  **Context Engineering**: The classified query is passed to the [context_engineering.py](four-hosts-app/backend/services/context_engineering.py) pipeline. This pipeline (W-S-C-I layers) refines the query into a detailed `ContextEngineeredQuery`, specifying search priorities, source preferences, and content processing strategies aligned with the inferred paradigm.
4.  **Research Execution**: The `ContextEngineeredQuery` is then fed to the [research_orchestrator.py](four-hosts-app/backend/services/research_orchestrator.py).
    *   The orchestrator uses the paradigm-specific search queries and preferences generated by [context_engineering.py](four-hosts-app/backend/services/context_engineering.py) and [paradigm_search.py](four-hosts-app/backend/services/paradigm_search.py).
    *   It executes searches via the [search_apis.py](four-hosts-app/backend/services/search_apis.py) manager, which handles interactions with various external search engines.
    *   Results are deduplicated by `ResultDeduplicator` and their credibility assessed using `credibility.py`.
    *   The `CostMonitor` tracks API usage and costs.
    *   The orchestrator returns a `ResearchExecutionResult` containing filtered and ranked search results.
5.  **Answer Generation**: The `ResearchExecutionResult` (specifically the filtered search results) is passed to the [answer_generator.py](four-hosts-app/backend/services/answer_generator.py) and [answer_generator_continued.py](four-hosts-app/backend/services/answer_generator_continued.py) modules.
    *   The `AnswerGenerationOrchestrator` selects the appropriate paradigm-specific generator (Dolores, Teddy, Bernard, or Maeve).
    *   These generators use the [llm_client.py](four-hosts-app/backend/services/llm_client.py) to synthesize the search results into a structured answer, complete with sections, summaries, action items, and citations, all aligned with the chosen paradigm's tone and focus.
    *   For multi-paradigm queries, the `mesh_network_service` (from [mesh_network.py](four-hosts-app/backend/services/mesh_network.py)) integrates outputs from multiple generators.

This integrated flow ensures that research is not only comprehensive but also deeply aligned with the specific perspective or "paradigm" inferred from the user's query, leading to highly relevant and contextually appropriate answers.

---

## Research Query

Investigate the "context_engineering.py" component in detail. Focus your analysis on these key files:
1. context_engineering.py (four-hosts-app/backend/services/context_engineering.py)

Provide insights about the component's main responsibilities and functionality and how it integrates with other system components

*Session: ad985929b878e3c9ec6e7f5e99f00da9 | Generated: 7/28/2025, 7:01:18 AM*

### Analysis Summary

# Context Engineering Component Analysis

The **Context Engineering** component, primarily implemented in [context_engineering.py](four-hosts-app/backend/services/context_engineering.py), is a crucial part of the Four Hosts application's backend. Its main responsibility is to transform a user's raw query, after it has been classified, into a highly refined and contextually rich set of parameters suitable for research execution. This process is inspired by the W-S-C-I (Write-Select-Compress-Isolate) methodology, which aims to optimize the research process by tailoring it to specific "Host Paradigms."

## Core Responsibilities and Functionality

The **Context Engineering Pipeline** ([ContextEngineeringPipeline](four-hosts-app/backend/services/context_engineering.py:449)) orchestrates four distinct layers, each contributing to the refinement of the query context:

1.  **Write Layer**: ([WriteLayer](four-hosts-app/backend/services/context_engineering.py:100))
    *   **Purpose**: Documents the query according to a paradigm-specific focus. It enriches the query with relevant themes, narrative frames, and search priorities based on the primary Host Paradigm identified during classification.
    *   **Internal Parts**: It contains `paradigm_strategies` that define how each [HostParadigm](four-hosts-app/backend/services/classification_engine.py) (Dolores, Teddy, Bernard, Maeve) influences the documentation process. It uses internal methods like `_extract_query_themes` and `_generate_search_priorities` to derive these elements from the initial query and its classification features.
    *   **External Relationships**: It takes a [ClassificationResult](four-hosts-app/backend/services/classification_engine.py) as input, which is generated by the **Classification Engine**. Its output, [WriteLayerOutput](four-hosts-app/backend/services/context_engineering.py:29), feeds into subsequent layers.

2.  **Select Layer**: ([SelectLayer](four-hosts-app/backend/services/context_engineering.py:200))
    *   **Purpose**: Selects appropriate research methods and sources based on the determined paradigm. It generates enhanced search queries, specifies source preferences, exclusion filters, and identifies suitable research tools.
    *   **Internal Parts**: It utilizes `selection_strategies` to define how each [HostParadigm](four-hosts-app/backend/services/classification_engine.py) influences query modification, source type selection, and tool recommendations. Methods like `_generate_search_queries` and `_generate_secondary_queries` are used to create a diverse set of search queries.
    *   **External Relationships**: It receives the [ClassificationResult](four-hosts-app/backend/services/classification_engine.py) and optionally the output from the **Write Layer** ([WriteLayerOutput](four-hosts-app/backend/services/context_engineering.py:29)). Its output, [SelectLayerOutput](four-hosts-app/backend/services/context_engineering.py:37), informs the subsequent research execution.

3.  **Compress Layer**: ([CompressLayer](four-hosts-app/backend/services/context_engineering.py:320))
    *   **Purpose**: Determines how information should be compressed or summarized based on the paradigm's priorities. It sets a compression ratio, strategy, and identifies elements to prioritize or remove.
    *   **Internal Parts**: It defines `compression_strategies` for each [HostParadigm](four-hosts-app/backend/services/classification_engine.py), including the desired `ratio`, `strategy`, `priorities`, and `remove` elements. It also calculates a `token_budget` based on query complexity.
    *   **External Relationships**: It takes the [ClassificationResult](four-hosts-app/backend/services/classification_engine.py) as input. Its output, [CompressLayerOutput](four-hosts-app/backend/services/context_engineering.py:45), guides how retrieved information will be processed for conciseness.

4.  **Isolate Layer**: ([IsolateLayer](four-hosts-app/backend/services/context_engineering.py:400))
    *   **Purpose**: Configures the extraction of key findings based on the paradigm's strategy. It defines criteria for key findings, extraction patterns (regex), focus areas, and the desired output structure.
    *   **Internal Parts**: It contains `isolation_strategies` for each [HostParadigm](four-hosts-app/backend/services/classification_engine.py), specifying `strategy`, `criteria`, `patterns`, `focus`, and the `output_structure` for the final research results.
    *   **External Relationships**: It takes the [ClassificationResult](four-hosts-app/backend/services/classification_engine.py) as input. Its output, [IsolateLayerOutput](four-hosts-app/backend/services/context_engineering.py:53), dictates how relevant information will be extracted and structured from the research findings.

## Integration with Other System Components

The **Context Engineering** component is tightly integrated with the **Classification Engine** and serves as a preparatory step for the **Research Orchestrator**.

*   **Classification Engine**: The `ContextEngineeringPipeline`'s primary input is a [ClassificationResult](four-hosts-app/backend/services/classification_engine.py) object, which is produced by the **Classification Engine** ([classification_engine.py](four-hosts-app/backend/services/classification_engine.py)). This result contains the identified primary and secondary Host Paradigms, along with extracted query features (entities, tokens, domain, urgency score). This tight coupling ensures that the context engineering process is highly tailored to the user's intent and the inferred "consciousness" of the query.

*   **Research Orchestrator**: The final output of the `ContextEngineeringPipeline` is a [ContextEngineeredQuery](four-hosts-app/backend/services/context_engineering.py:61). This comprehensive object encapsulates all the refined parameters (search queries, source preferences, compression strategies, isolation criteria) needed by the **Research Orchestrator** ([research_orchestrator.py](four-hosts-app/backend/services/research_orchestrator.py)) to execute the research process effectively. The `ContextEngineeredQuery` acts as a blueprint for the subsequent information gathering and processing stages.

In essence, the **Context Engineering** component acts as an intelligent intermediary, translating a high-level classified query into a detailed, actionable plan for the research system, ensuring that the subsequent research is focused, efficient, and aligned with the underlying Host Paradigm.

---

## Research Query

Investigate the "ClassificationResult" component in detail. Focus your analysis on these key files:
1. classification_engine.py (four-hosts-app/backend/services/classification_engine.py)

Provide insights about the component's main responsibilities and functionality and how it integrates with other system components

*Session: f2ce31a398192ce6511189f35988e284 | Generated: 7/28/2025, 7:17:36 AM*

### Analysis Summary

# ClassificationResult Component Analysis

The `ClassificationResult` component, defined as a dataclass in [classification_engine.py](four-hosts-app/backend/services/classification_engine.py:50), serves as the comprehensive output structure for the query classification process within the Four Hosts application. It encapsulates all relevant information derived from analyzing a user's query and assigning it to one or more "Host Paradigms."

## Main Responsibilities and Functionality

The primary responsibility of `ClassificationResult` is to provide a standardized and detailed report of a classified query. Its functionality is purely data-holding; it does not contain any methods for processing or manipulating data. Instead, it acts as a container for the results generated by the [ParadigmClassifier](four-hosts-app/backend/services/classification_engine.py:159) and [QueryAnalyzer](four-hosts-app/backend/services/classification_engine.py:77) components.

The `ClassificationResult` dataclass includes the following key fields:

*   **`query`**: The original input query string that was classified.
*   **`primary_paradigm`**: The [HostParadigm](four-hosts-app/backend/services/classification_engine.py:29) that the query most strongly aligns with.
*   **`secondary_paradigm`**: An optional [HostParadigm](four-hosts-app/backend/services/classification_engine.py:29) indicating a second-strongest alignment, if applicable.
*   **`distribution`**: A dictionary mapping each [HostParadigm](four-hosts-app/backend/services/classification_engine.py:29) to a float representing its normalized score or probability within the classification.
*   **`confidence`**: A float indicating the overall confidence level of the classification.
*   **`features`**: An instance of [QueryFeatures](four-hosts-app/backend/services/classification_engine.py:35), containing the extracted characteristics of the query (e.g., tokens, entities, intent signals, urgency, complexity, emotional valence).
*   **`reasoning`**: A dictionary providing a list of strings for each [HostParadigm](four-hosts-app/backend/services/classification_engine.py:29), explaining why that paradigm received its score (e.g., matched keywords, LLM reasoning).
*   **`timestamp`**: A `datetime` object indicating when the classification was performed.

## Integration with Other System Components

The `ClassificationResult` component is central to the output of the classification engine and integrates with several other components:

*   **[ParadigmClassifier](four-hosts-app/backend/services/classification_engine.py:159)**: This is the primary component responsible for constructing and returning an instance of `ClassificationResult`. The [classify](four-hosts-app/backend/services/classification_engine.py:164) method of `ParadigmClassifier` orchestrates the entire classification process, including rule-based and LLM-based scoring, score combination, normalization, and confidence calculation, before populating and returning the `ClassificationResult` object.
*   **[QueryFeatures](four-hosts-app/backend/services/classification_engine.py:35)**: An instance of `QueryFeatures` is embedded directly within `ClassificationResult`. The [QueryAnalyzer](four-hosts-app/backend/services/classification_engine.py:77) component generates these features, which are then passed to the `ParadigmClassifier` and ultimately stored in the `ClassificationResult` for comprehensive context.
*   **[HostParadigm](four-hosts-app/backend/services/classification_engine.py:29)**: This Enum is used extensively within `ClassificationResult` to define the types of paradigms (e.g., `primary_paradigm`, `secondary_paradigm`, keys in `distribution` and `reasoning`).
*   **[ClassificationEngine](four-hosts-app/backend/services/classification_engine.py:340)**: This class serves as the main interface to the classification system. Its [classify_query](four-hosts-app/backend/services/classification_engine.py:352) method returns a `ClassificationResult` object. It also uses an internal cache (`self.cache`) to store `ClassificationResult` instances, enabling faster retrieval for repeated queries.
*   **[ParadigmScore](four-hosts-app/backend/services/classification_engine.py:43)**: While not directly part of `ClassificationResult`, `ParadigmScore` objects are intermediate results used by the `ParadigmClassifier` to build up the final `distribution` and `reasoning` within the `ClassificationResult`.


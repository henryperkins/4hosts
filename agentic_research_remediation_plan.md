@# Agentic Research Workflow: End-to-End Remediation Plan

This document outlines a diagnostic analysis and a corrective action plan for the agentic research workflow. The goal is to prevent critical failures, improve retrieval quality, and ensure robust, graceful degradation of the system.

## 1. Root-Cause Analysis

The failure originates from a cascade of issues, starting with query corruption and propagating through the pipeline, ultimately causing a `NoneType` error.

| Log Line / State | Pipeline Stage | Failure Mode & Analysis |
| :--- | :--- | :--- |
| `Optimized query: ... -> 'artificial intelligence tell me about about...'` | **Source Discovery** | **Query Corruption**: The optimization logic is flawed. It incorrectly removes critical keywords ("context engineering", "web"), duplicates stopwords ("about"), and fails to preserve named entities ("maeve"). This guarantees low-quality search results. |
| `Filtered 10 results to 4 with min relevance 0.25` | **Retrieval** | **Weak Relevance Thresholding**: A relevance score of `0.25` is too permissive, allowing irrelevant content into the pipeline. This is a symptom of the poor query quality. |
| `Final results: 6 sources` vs. `Used google for search, got 4 results` | **Retrieval** | **Result-Count Mismatch (Source Aggregation)**: The orchestrator reports 6 sources while the search API reports 4. This indicates a bug in source aggregation, potentially from multiple search engines or internal sources, without proper deduplication or reconciliation. |
| `Starting answer generation with 0 results` | **Synthesis (Handoff)** | **Data Pipeline Break**: This is the most critical failure. The 6 sources from the orchestrator are lost before reaching the answer generator. This points to a broken contract, serialization error, or data loss between `research_orchestrator` and the `main` answer generation service. |
| `Answer generation completed. Type: <class 'NoneType'>` | **Synthesis** | **Missing Nil-Handling**: The answer generator lacks a defensive check for empty or invalid input. It returns `None` instead of a structured "no-answer" response, which is not a predictable or safe output. |
| `ERROR: ... 'NoneType' object has no attribute 'summary'` | **Synthesis (Post-Processing)** | **NoneType Dereference**: A downstream consumer (monitoring or reporting) attempts to access an attribute on the `None` object, causing the application to crash. This demonstrates a lack of defensive programming and contract enforcement. |
| `Triggered event WebhookEvent.RESEARCH_FAILED` | **Reporting** | **Incomplete Failure Reporting**: The system correctly identifies a failure but lacks the specific context (e.g., `REASON_INSUFFICIENT_DATA`, `REASON_INTERNAL_ERROR`) to make the alert actionable. |

## 2. Corrective Design: Revised Architecture & Contracts

We will introduce explicit contracts, defensive patterns, and clear status communication between services.

### Service Contracts (Input/Output Schemas)

**`search_apis` Service:**
- **Input**: `SearchRequest(query: str, num_results: int)`
- **Output**: `SearchResponse(status: Status, results: List[SourceCandidate])`
- **`SourceCandidate` Schema**: `{ "url": str, "title": str, "snippet": str, "raw_score": float }`

**`research_orchestrator` Service:**
- **Input**: `ResearchJob(original_query: str, user_id: str)`
- **Output**: `ResearchResult(status: Status, sources: List[Source], report: ResearchReport)`
- **`Source` Schema**: `{ "id": str, "url": str, "title": str, "relevance_score": float, "content_chunks": List[str], "is_complete": bool }`
- **`ResearchReport` Schema**: `{ "request_id": str, "optimized_queries": List[str], "steps": List[dict] }`

**`answer_generator` Service:**
- **Input**: `GenerationRequest(sources: List[Source], original_query: str)`
- **Output**: `GeneratedAnswer`
- **`GeneratedAnswer` Schema**: `{ "status": Status, "summary": Optional[str], "constituent_ideas": List[str], "cited_sources": List[str], "message": Optional[str] }`

### Status Enum

A mandatory `status` field will be used across all services:
```python
from enum import Enum

class Status(str, Enum):
    OK = "OK"
    SUCCESS = "SUCCESS"
    PENDING = "PENDING"
    FAILED_INTERNAL_ERROR = "FAILED_INTERNAL_ERROR"
    FAILED_DEPENDENCY = "FAILED_DEPENDENCY"
    FAILED_INSUFFICIENT_DATA = "FAILED_INSUFFICIENT_DATA"
    FAILED_QUERY_ERROR = "FAILED_QUERY_ERROR"
```

### Defensive Programming Patterns

- **Null Object Pattern**: The `answer_generator` will **never** return `None`. If it cannot generate an answer, it returns a `GeneratedAnswer` object with `status=FAILED_INSUFFICIENT_DATA` and a descriptive `message`.
- **Retries & Jitter**: All external API calls (e.g., to Google Search) will use `tenacity` or a similar library for exponential backoff with jitter.
- **Timeouts**: A default timeout (e.g., 15 seconds) will be enforced on all network requests.
- **Circuit Breaker**: A library like `pybreaker` will wrap calls to external search APIs. If an API fails consistently, the circuit will open, and the system will immediately fall back to a secondary provider without waiting for a timeout.

### Preconditions & Postconditions

- **`research_orchestrator` Precondition**: Input `original_query` must not be empty.
- **`research_orchestrator` Postcondition**: Output `sources` list must not be `None` (it can be empty). `status` must be set.
- **`answer_generator` Precondition**: Input `sources` must contain at least `K` items (e.g., `K=2`) where `source.is_complete == True`. If not met, immediately return `GeneratedAnswer(status=FAILED_INSUFFICIENT_DATA)`.
- **`answer_generator` Postcondition**: Output must be a valid `GeneratedAnswer` object, never `None`.

## 3. Query Optimization Fix

The query pipeline will be redesigned to preserve intent and improve precision.

**Transformation Pipeline:**
1.  **NER & Keyword Extraction**: Use a lightweight model (e.g., `spaCy` or a fine-tuned DeBERTa) to identify and protect named entities (`"context engineering"`, `"web applications"`, `"Maeve"`). These are treated as atomic units.
2.  **Intent-Preserving Stopword Removal**: Remove stopwords *only if* they are not adjacent to a protected entity.
3.  **Multi-Query Expansion**: Generate a set of queries to run in parallel.
    - **Primary**: The query with protected entities. `("context engineering" AND "artificial intelligence" AND "web applications") OR "maeve"`
    - **Semantic**: A broader query. `(AI OR "artificial intelligence") context management for web apps`
    - **Question**: A rephrased question. `How is context engineering used in AI web applications?`
4.  **Backoff Strategy**: If the primary query returns `< N` results (e.g., `N=5`), automatically execute the semantic and question-based queries.

**Example Corrected Query:**
- **Original**: `'Tell me about context engineering and artificial intelligence in web applications. maeve'`
- **Corrected Primary**: `("context engineering" OR "context management") AND "artificial intelligence" AND "web applications" AND "maeve"`

## 4. Retrieval and Ranking

This stage will be hardened to ensure only high-quality, relevant sources proceed.

- **Relevance Scoring**: A weighted score will be calculated: `0.6 * semantic_similarity(snippet, query) + 0.3 * title_match_score + 0.1 * url_keyword_match`. A result is kept only if `relevance_score > 0.65`.
- **Deduplication**:
    - **URL-based**: Normalize URLs (remove tracking params, http/s) and discard exact duplicates.
    - **Content-based**: Use MinHash or SimHash on snippets to discard near-duplicates (similarity > 0.9).
- **Source Quorum & Reconciliation**:
    - The `research_orchestrator` is the single source of truth. It must collect results from all search engines, deduplicate them, and create a final list.
    - **Minimum Viable Results Policy**: The orchestrator must produce a `ResearchResult` with at least `2` sources having `relevance_score > 0.7`. If this quorum is not met, the process stops and returns `status=FAILED_INSUFFICIENT_DATA`. This prevents the "6 sources -> 0 results" discrepancy.
- **Data Model with Completeness Flags**:
    ```python
    class Source:
        id: str
        url: str
        title: str
        relevance_score: float
        raw_content: Optional[str]
        content_chunks: List[str] = []
        # Flag to indicate if fetching and chunking was successful
        is_complete: bool = False
        error_message: Optional[str] = None
    ```

## 5. Synthesis and Generation

The answer generator will be made more resilient and communicative.

- **Preflight Check**:
    ```python
    MIN_EVIDENCE_CHUNKS = 3
    complete_sources = [s for s in sources if s.is_complete]
    evidence_chunks = sum([len(s.content_chunks) for s in complete_sources])

    if evidence_chunks < MIN_EVIDENCE_CHUNKS:
        return GeneratedAnswer(
            status=Status.FAILED_INSUFFICIENT_DATA,
            message=f"Cannot generate answer. Required {MIN_EVIDENCE_CHUNKS} evidence chunks, but found only {evidence_chunks}."
        )
    ```
- **Safe Defaults & No-Answer Output**: The `GeneratedAnswer` model ensures a structured response is always returned. The default `summary` is `None`, not an empty string, to be explicit.

## 6. Observability and Alerts

We will implement metrics, logs, and traces to monitor the health of the new design.

- **Metrics & Logs (Prometheus/Grafana format)**:
    - `research_query_total{status="SUCCESS/FAILED"}`: Counter for research jobs.
    - `query_mutation_drift_ratio`: Gauge measuring Levenshtein distance between original and final query.
    - `result_count_mismatch_total`: Counter for discrepancies between service layers.
    - `null_output_total{service="answer_generator"}`: Counter for any function returning a raw `None`.
- **Traces**: All services will propagate a `X-Request-ID` header. Use OpenTelemetry to trace the entire lifecycle of a research request.
- **Alert Rules (Alertmanager)**:
    - `ALERT HighQueryMutationDrift IF query_mutation_drift_ratio > 0.4 FOR 5m`
    - `ALERT HighFailureRate IF sum(rate(research_query_total{status="FAILED"}[5m])) / sum(rate(research_query_total[5m])) > 0.1 FOR 10m`
    - `ALERT ResultCountMismatchDetected IF increase(result_count_mismatch_total[1m]) > 0`
- **Minimal Dashboard Layout**:
    - **Top Row**: Key KPIs (Total Requests, Success Rate %, P95 Latency).
    - **Query Health**: Time-series graphs for Query Mutation Drift, Zero-Result Rate.
    - **Pipeline Health**: Funnel chart showing request count at each stage (Orchestrator -> Search -> Generator). Graph for `result_count_mismatch_total`.
    - **Error Logs**: A table view of structured logs filtered for `level=ERROR`.

## 7. Test Plan

- **Unit Tests**:
    - `test_query_optimizer.py`: Test with the exact failing query and assert that entities are preserved.
    - `test_answer_generator.py`: Test with `sources=[]` and `sources=[incomplete_source]` and assert it returns a valid `GeneratedAnswer` with `status=FAILED_INSUFFICIENT_DATA`.
- **Integration Tests**:
    - `test_orchestrator_to_generator_handoff`: Create a mock `research_orchestrator` that returns a fixed `ResearchResult` and assert that the `answer_generator` receives it correctly.
- **Chaos & E2E Tests**:
    - Use a tool like `toxiproxy` to simulate search API failures and latency; assert that the circuit breaker and fallbacks trigger correctly.
    - **Synthetic Incident Test**: Create an E2E test that uses the original user query and asserts that the final output is a structured `GeneratedAnswer` (either `SUCCESS` or `FAILED_INSUFFICIENT_DATA`), and that no `NoneType` exceptions are logged.
- **Regression Tests**: The synthetic incident test will serve as the primary regression test for this specific failure.

## 8. Acceptance Criteria

The remediation is considered complete when the following conditions are met:

1.  **No Critical Errors**: Executing the original query (`'Tell me about context engineering...'`) 100 times results in zero `NoneType` exceptions or unhandled errors.
2.  **Graceful Degradation**: When search APIs are disabled, the system returns a `GeneratedAnswer` with `status=FAILED_DEPENDENCY` or `FAILED_INSUFFICIENT_DATA` within 20 seconds.
3.  **Query Integrity**: The `query_mutation_drift_ratio` metric remains below `0.2` for 99% of production traffic.
4.  **Data Consistency**: The `result_count_mismatch_total` alert does not fire for 24 hours of normal operation.
5.  **Observability**: The new Grafana dashboard is deployed and correctly displays all metrics defined in section 6.
6.  **Test Coverage**: All tests defined in the test plan are implemented and passing in the CI/CD pipeline.

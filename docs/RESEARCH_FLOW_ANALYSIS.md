# Four Hosts Research Query Flow Analysis

## Complete Query Flow (End-to-End)

### 1. Frontend Query Submission
- **File**: `frontend/src/components/ResearchFormEnhanced.tsx:133-154`
- User enters query in form with minimum 10 character validation
- Options configured: paradigm override, depth, max_sources, language, region
- Form submits to API with ResearchOptions payload

### 2. Backend API Entry Point
- **File**: `backend/routes/research.py:922-1021`
- POST `/research/query` endpoint receives request
- User authentication and role verification
- Rate limiting checks (per-user limits)
- Initial paradigm classification run (can toggle LLM on/off)
- Research data stored with status PROCESSING
- Background task launched via `execute_real_research`

### 3. Paradigm Classification Pipeline
- **File**: `backend/services/classification_engine.py:350-406`
- Query features extracted: tokens, entities, intent signals
- Rule-based classification using keywords and patterns
- Optional LLM classification (Azure OpenAI GPT-4)
- Weighted score combination (rule_weight + llm_weight + domain_weight)
- Primary and secondary paradigms determined with confidence scores
- Distribution across all four paradigms calculated

### 4. Context Engineering (W-S-C-I)
- **File**: `backend/services/context_engineering.py:222-258`
- **Write Layer**: Documents according to paradigm focus
  - Extracts key themes from query
  - Generates paradigm-specific search priorities
- **Select Layer**: Chooses search strategies and sources
- **Compress Layer**: Optimizes for token budget
- **Isolate Layer**: Defines extraction patterns and focus areas

### 5. Research Orchestration
- **File**: `backend/services/research_orchestrator.py:900-1100`
- Query optimization and compression
- Deterministic search execution across multiple APIs:
  - Google Custom Search
  - Brave Search
  - ArXiv
  - PubMed
  - Semantic Scholar
- Result deduplication using simhash
- Credibility scoring per domain
- Optional deep research integration
- Progress updates via WebSocket

### 6. Answer Generation
- **File**: `backend/services/answer_generator.py:2263-2311`
- Paradigm-specific generators instantiated
- Evidence selection within token budget
- LLM synthesis using Azure OpenAI GPT-4
- Section generation with citations
- Action items extraction
- Multi-paradigm synthesis for certain combinations (e.g., Maeve + Dolores)

### 7. Response Construction
- **File**: `backend/routes/research.py:844-863`
- Final result structure assembled with:
  - Paradigm analysis
  - Answer with sections and citations
  - Source list with credibility scores
  - Metadata including processing metrics
  - Optional integrated/mesh synthesis
- Stored in research_store
- WebSocket progress completion event

### 8. Frontend Display
- **File**: `frontend/src/components/ResultsDisplayEnhanced.tsx:39-100`
- Receives results via WebSocket or polling
- Displays answer sections, citations, sources
- Shows paradigm analysis and confidence
- Credibility indicators and bias checks
- Export options (PDF, Markdown, JSON)

## ACTUAL Architectural Problems Found

### 1. Triple Paradigm Override Chaos
- **Location**: `backend/routes/research.py`
- **Problem**: Paradigm can be overridden at THREE different points:
  - Line 968-995: Initial classification during submission
  - Line 159-193: Reconciling stored vs new classification during execution
  - Line 196-244: User override from request options
- **Impact**: Each override artificially sets confidence to 0.9, making confidence scores meaningless
- **Evidence**: The same research can have different paradigms at different stages of execution

### 2. Critical Race Condition in Result Storage
- **Location**: `backend/routes/research.py:862-868`
- **Problem**: Non-atomic updates create inconsistent states:
  ```python
  await research_store.update_field(research_id, "results", final_result)  # Line 862
  await research_store.update_field(research_id, "status", ResearchStatus.COMPLETED)  # Line 863
  # ...
  await progress_tracker.complete_research(research_id, {...})  # Line 868
  ```
- **Impact**: Frontend can receive "research_completed" WebSocket event but get "still processing" when fetching results
- **Evidence**: If frontend polls between lines 862-863, it sees incomplete data despite completion notification

### 3. Progress Tracking is Complete Fiction
- **Location**: Throughout `execute_real_research`
- **Problem**: Hardcoded progress values with no relation to actual work:
  - 5% after classification (instant)
  - 15% after context engineering (~200ms)
  - 25% after search starts
  - 98% at completion
  - Missing: 73% of progress bar unaccounted for during actual search/synthesis work
- **Impact**: Users see misleading progress that can jump from 25% to 98% instantly or hang at 25% for minutes

### 4. Systematic Silent Failure Pattern
- **Location**: Throughout codebase
- **Problem**: 44+ instances of `except Exception:` in `routes/research.py` alone
- **Example Impact**:
  - Paradigm override failures silently ignored (lines 148-149)
  - WebSocket updates silently fail (lines 126-127)
  - Critical metadata assembly errors swallowed (lines 177-178)
- **Consequence**: System appears to work but produces degraded results with no error reporting

### 5. Query Refinement Layer Doesn't Work
- **Location**: `backend/services/context_engineering.py:1077-1092`
- **Problem**: `refined_queries` is populated from `optimize_output` variations, but:
  - OptimizeLayer only runs if ENABLE_QUERY_LLM=1 (line 889)
  - Falls back to original query silently if empty
  - No validation that variations actually improve search
- **Evidence**: Most searches use original query despite complex W-S-C-I pipeline

### 6. Deep Research Backwards Logic
- **Location**: `backend/routes/research.py:358-361`
- **Problem**: Enables deep research when `enable_real_search` is FALSE:
  ```python
  if not getattr(research.options, "enable_real_search", True) and not enable_deep:
      if user_role in [PRO, ENTERPRISE, ADMIN]:
          enable_deep = True
  ```
- **Impact**: Deep research (which needs search APIs) activates when search is disabled

### 7. Answer Synthesis Happens Even Without Results
- **Location**: `backend/services/research_orchestrator.py`
- **Problem**: No validation that search returned meaningful results before synthesis
- **Impact**: LLM generates hallucinated answers when search fails
- **Evidence**: `synthesize_answer` flag always honored regardless of result quality

### 8. WebSocket and Store Desynchronization
- **Location**: Multiple points in execution flow
- **Problem**: WebSocket progress events and research_store updates are independent:
  - Progress tracker shows 100% but store still says IN_PROGRESS
  - Store has results but WebSocket hasn't sent completion
  - No transaction or coordination between systems
- **Impact**: Frontend shows conflicting information from different sources

### 9. Result Shape Mutations Without Validation
- **Location**: `backend/routes/research.py:844-860`
- **Problem**: Final result structure built manually with nested try/except blocks:
  - No schema validation
  - Fields added conditionally based on various flags
  - Shape changes based on paradigm combinations
- **Impact**: Frontend receives unpredictable data structures causing rendering failures

### 10. Context Engineering Metrics are Fake
- **Location**: `backend/routes/research.py:542-561`
- **Problem**: Context layer metrics assembled from optional fields that may not exist:
  - Uses `getattr` with defaults everywhere
  - No validation that layers actually ran
  - Reports success even when layers failed
- **Impact**: Metrics dashboard shows false positive data

## Critical Fixes Needed (Priority Order)

### P0 - Data Corruption Risks
1. **Fix Race Condition**: Make result storage atomic
   - Combine results + status update in single transaction
   - Send WebSocket completion AFTER store update confirms
   - Add version/sequence numbers to prevent stale reads

2. **Fix Paradigm Override Logic**: Single source of truth
   - Remove paradigm override from execution phase
   - Apply overrides ONCE during submission
   - Never artificially boost confidence scores
   - Log all paradigm changes for debugging

### P1 - User Experience Failures
3. **Fix Progress Tracking**: Make it real
   - Track actual search API calls completed
   - Report synthesis progress from LLM
   - Remove hardcoded magic numbers
   - Add estimated time remaining based on historical data

4. **Add Error Visibility**: Stop swallowing exceptions
   - Log all exceptions with context
   - Return degraded-quality indicators to frontend
   - Add error telemetry for monitoring
   - Implement circuit breakers for failing components

### P2 - System Reliability
5. **Fix Query Refinement**: Make it actually work
   - Don't require ENABLE_QUERY_LLM for basic variations
   - Validate that variations differ from original
   - Track which queries actually return results
   - A/B test refined vs original queries

6. **Fix Deep Research Logic**: Correct the backwards condition
   - Deep research should enhance normal search, not replace it
   - Should activate when MORE search is needed, not less
   - Document clear activation criteria

### P3 - Quality Improvements
7. **Validate Before Synthesis**: Don't generate without data
   - Minimum result count threshold before synthesis
   - Quality score threshold for results
   - Fallback to "insufficient data" response
   - Track synthesis quality metrics

8. **Add Data Contracts**: Prevent shape mutations
   - Define TypeScript interfaces and Pydantic models
   - Validate at API boundaries
   - Version the API contracts
   - Add integration tests for contract compliance

## Root Cause Analysis

The core issues stem from:
1. **No Transactional Thinking**: Updates treated as independent operations instead of atomic transactions
2. **Optimistic Error Handling**: Assuming success and silently ignoring failures
3. **Presentation Over Precision**: Progress bars and metrics optimized for appearance rather than accuracy
4. **Accretion Without Refactoring**: Features added without restructuring, leading to logic spread across multiple locations
5. **Weak Contracts**: No enforcement of data shapes between components

## Architecture Recommendations

1. **Introduce Saga Pattern**: For multi-step research execution with proper rollback
2. **Event Sourcing**: Track all state changes as events for debugging and replay
3. **CQRS**: Separate write path (research execution) from read path (result fetching)
4. **Circuit Breakers**: Prevent cascade failures when components degrade
5. **Observability**: Add structured logging, distributed tracing, and metrics
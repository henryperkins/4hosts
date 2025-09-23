# Research Progress Tracking Flow (Back to Front)

## Overview
The Four Hosts application tracks research progress through a WebSocket-based real-time system that flows from backend services through to the frontend UI. Progress updates are broadcast at each major phase of the research pipeline.

## Architecture Components

### 1. Core Progress Infrastructure

#### ProgressTracker (`services/websocket_service.py:598`)
- Central hub for all progress events
- Manages research state and phase transitions
- Weighted progress model with phases:
  - `classification`: 10%
  - `context_engineering`: 15%
  - `search`: 40%
  - `analysis`: 10% (dedup/credibility/filtering)
  - `agentic_loop`: 10%
  - `synthesis`: 15%
  - `complete`: 0% (terminal state)

#### Progress Facade (`services/progress.py`)
- Thin wrapper providing safe no-op behavior when WebSocket unavailable
- Used across all services for progress reporting
- Methods:
  - `update_progress()` - General progress updates
  - `report_credibility_check()` - Credibility scoring events
  - `report_deduplication()` - Deduplication statistics
  - `report_search_started/completed()` - Search phase tracking
  - `report_source_found()` - Individual source discovery
  - `report_synthesis_started/completed()` - Answer generation phase
  - `report_evidence_builder_skipped()` - Evidence building bypass

### 2. WebSocket Event Types (`websocket_service.py:32`)

Research Events:
- `RESEARCH_STARTED` → `research_started`
- `RESEARCH_PROGRESS` → `research_progress`
- `RESEARCH_PHASE_CHANGE` → `research_phase_change`
- `RESEARCH_COMPLETED` → `research_completed`
- `RESEARCH_FAILED` → `research_failed`

Analysis Events:
- `CREDIBILITY_CHECK` → `credibility.check`
- `DEDUPLICATION` → `deduplication.progress`

Search Events:
- `SEARCH_STARTED` → `search.started`
- `SEARCH_COMPLETED` → `search.completed`
- `SOURCE_FOUND` → `source_found`

Synthesis Events:
- `SYNTHESIS_STARTED` → mapped to `research_progress`
- `SYNTHESIS_COMPLETED` → mapped to `research_completed`

## Progress Flow by Phase (Back to Front)

### Phase 1: Query Classification
**Location:** `services/classification_engine.py`
**Progress Reporting:** Updates phase to "classification"
**Details Tracked:**
- Paradigm identification (primary/secondary)
- Classification confidence scores

### Phase 2: Query Optimization (Context Engineering)
**Location:** `services/context_engineering.py:1034-1104`
**Progress Points:**
1. Write Layer (`context_engineering.py:1060`) - "Rewriting query"
2. Select Layer (`context_engineering.py:1074`) - "Optimizing search terms"
3. Compress Layer (`context_engineering.py:1088`) - "Generating search queries"
4. Isolate Layer (`context_engineering.py:1102`) - "Isolating key findings"

**Details Tracked:**
- Original query
- Rewritten variations
- Search term optimization
- Query compression results

### Phase 3: Search & Retrieval
**Location:** `services/research_orchestrator.py:2301-2376`
**Progress Points:**
1. Search Started (`research_orchestrator.py:2304`)
   - Query being executed
   - Stage and label identifiers
2. Search Completed (`research_orchestrator.py:2366`)
   - Number of results found
   - APIs successfully queried
3. Source Found (`research_orchestrator.py:2373`)
   - Individual source metadata (title, URL, snippet)
   - First 3 sources reported for UI preview

**Details Tracked:**
- `total_searches` - Number of search operations planned
- `searches_completed` - Completed search operations
- `sources_found` - Total sources discovered

### Phase 4: Credibility Checking
**Location:** `services/credibility.py`, called via `research_orchestrator.py`
**Progress Reporting:** `websocket_service.py:1286`
**Details Tracked:**
- Domain being evaluated
- Credibility score (0.0-1.0)
- Domain authority metrics
- Bias ratings
- Fact-check ratings

### Phase 5: Ranking & Deduplication
**Location:** `services/query_planning/result_deduplicator.py`
**Progress Reporting:** `research_orchestrator.py:2490`
**Details Tracked:**
- `before_count` - Results before deduplication
- `after_count` - Results after deduplication
- `duplicates_removed` - Number of duplicates eliminated
- `deduplication_rate` - Percentage of duplicates (0.0-1.0)

### Phase 6: Synthesis & Answer Generation
**Location:** `services/answer_generator.py`, `research_orchestrator.py:2018-2137`
**Progress Points:**
1. Synthesis Started (`research_orchestrator.py:2020`)
   - Paradigm being used
   - Number of sources being synthesized
2. Synthesis Progress (`answer_generator.py:962`)
   - Section being generated
   - Token/word targets
3. Synthesis Completed (`research_orchestrator.py:2132`)
   - Number of sections generated
   - Total citations included

**Details Tracked:**
- Section generation progress
- Citation compilation
- Evidence bundle creation

## Real-time Progress Metrics

### Calculated Metrics (`websocket_service.py:761-790`)
- **Overall Progress Percentage:** Weighted sum of phase completions
- **Phase Duration:** Time spent in each phase (milliseconds)
- **Phase Units:** Granular completion tracking within phases
- **Completed Phases:** Set of finished pipeline stages

### Heartbeat System (`websocket_service.py:609`)
- Periodic updates every 20 seconds (configurable)
- Prevents connection timeout during long operations
- Maintains UI responsiveness during LLM synthesis

## Frontend Integration

### WebSocket Message Transformation (`websocket_service.py:523-591`)
- Backend event types mapped to frontend-compatible formats
- ISO timestamp formatting
- Status field injection for cancelled events
- Retry intent preservation for search operations

### Progress Persistence
- Snapshots saved every 2 seconds (`websocket_service.py:630`)
- Message history retention for reconnection (100 messages, 15min TTL)
- Completed research retention for 5 minutes

## Progress Tracking Best Practices

1. **Always Report Phase Changes:** Ensures accurate progress calculation
2. **Granular Updates in Long Operations:** Use `items_done`/`items_total` for sub-phase progress
3. **Error Handling:** Progress updates wrapped in try/except to prevent pipeline disruption
4. **Custom Data:** Additional context via `custom_data` parameter for detailed UI updates
5. **Batch Reporting:** Aggregate similar updates to reduce WebSocket traffic

## Monitoring & Observability

### Structured Logging
- All progress events logged with `structlog`
- Phase transitions tracked with timing metrics
- Error conditions logged without breaking flow

### Telemetry Integration
- Prometheus metrics for phase durations
- Deduplication rates tracked
- Search API performance monitoring

## Example Progress Event Flow

```
1. User submits query
2. RESEARCH_STARTED event broadcast
3. Classification phase (10% progress)
4. Context engineering phases (15% progress)
   - Write, Select, Compress, Isolate sub-phases
5. Search operations (40% progress)
   - Multiple SEARCH_STARTED/COMPLETED cycles
   - SOURCE_FOUND events for discoveries
6. Analysis phase (10% progress)
   - CREDIBILITY_CHECK events
   - DEDUPLICATION event
7. Synthesis phase (15% progress)
   - SYNTHESIS_STARTED (mapped to research_progress)
   - Section generation updates
   - SYNTHESIS_COMPLETED (mapped to research_completed)
8. RESEARCH_COMPLETED event (100% progress)
```

This comprehensive tracking ensures users have real-time visibility into every step of the research process, from initial query optimization through final answer synthesis.
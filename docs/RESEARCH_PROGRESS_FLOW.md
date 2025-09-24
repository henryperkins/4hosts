# Research Progress Tracking Flow (Back to Front)

## Overview
The Four Hosts application tracks research progress through a WebSocket-based real-time system that flows from backend services through to the frontend UI. Progress updates are broadcast at each major phase of the research pipeline.

### Timeout alignment
- Progress sockets honour `PROGRESS_WS_TIMEOUT_MS` (mirrors `VITE_PROGRESS_WS_TIMEOUT_MS`). The backend keeps an internal heartbeat interval derived from this value and announces it to clients via the `/system/frontend-config` endpoint as well as on resume heartbeats so UI pause messaging stays in sync.
- REST polling honours `RESULTS_POLL_TIMEOUT_MS` (mirrors `VITE_RESULTS_POLL_TIMEOUT_MS`). These values surface in both `/research/status/{id}` and `/research/results/{id}` payloads to document the backend-side expectations for long-running jobs.

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
  - `synthesis`: 12%
  - `complete`: 3% (smooths final transition to 100%)

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
**Location:** `services/context_engineering.py:1056-1149`
**Progress Points:**
1. Write Layer (`context_engineering.py:1060`) - "Processing Write layer - documenting paradigm focus"
2. Rewrite Layer (`context_engineering.py:1074`) - "Rewriting query for clarity and searchability"
3. Select Layer (`context_engineering.py:1088`) - "Selecting search methods and sources"
4. Optimize Layer (`context_engineering.py:1102`) - "Optimizing search terms and query variations"
5. Compress Layer (`context_engineering.py:1116`) - "Compressing information by paradigm priorities"
6. Isolate Layer (`context_engineering.py:1130`) - "Isolating key findings extraction patterns"

**Details Tracked:**
- Original query
- Rewritten variations
- Search term optimization
- Query compression results

### Phase 3: Search & Retrieval
**Location:** `services/research_orchestrator.py:2286-2404`
**Progress Points:**
1. Search Started (`research_orchestrator.py:2304`)
   - Query being executed
   - Stage and label identifiers
2. Search Completed (`research_orchestrator.py:2366`)
   - Number of results found
   - Running counters for `searches_completed` / `total_searches`
3. Source Found (`research_orchestrator.py:2373`)
   - Individual source metadata (title, URL, snippet)
   - First 3 sources reported for UI preview

**Details Tracked:**
- `total_searches` - Number of search operations planned
- `searches_completed` - Completed search operations
- `sources_found` - Total sources discovered
- API usage is inferred from successive `search.started` events on the frontend (no longer emitted on `search.completed`)

### Phase 4: Credibility Checking
**Location:** `services/credibility.py`, called via `research_orchestrator.py`
**Progress Reporting:** `websocket_service.py:1379`
**Details Tracked:**
- Domain being evaluated
- Credibility score (0.0-1.0)
- Timestamp (broadcast payload no longer includes auxiliary metrics)

### Phase 5: Ranking & Deduplication
**Location:** `services/query_planning/result_deduplicator.py`
**Progress Reporting:** `research_orchestrator.py:2490` / `websocket_service.py:1394`
**Details Tracked:**
- `before_count` - Results before deduplication
- `after_count` - Results after deduplication
- `removed` - Number of duplicates eliminated
- Timestamp (deduplication rate is now derived server-side only)

### Phase 6: Synthesis & Answer Generation
**Location:** `services/answer_generator.py`, `research_orchestrator.py:2018-2137`
**Progress Points:**
1. Synthesis Started (`research_orchestrator.py:2020`)
   - Paradigm being used
   - Number of sections planned
2. Section Workflow (`answer_generator.py:1219-1275`)
   - Per-section sub-steps emitted via `update_progress_step` (filtering sources, creating citations, generating content, extracting insights, section complete)
3. Synthesis Completed (`research_orchestrator.py:2132`)
   - Number of sections generated
   - Total citations included

**Details Tracked:**
- Section generation progress (step labels per section)
- Citation compilation counts
- Evidence bundle usage (implicit through section steps)

## Real-time Progress Metrics

### Calculated Metrics (`websocket_service.py:761-790`)
- **Overall Progress Percentage:** Weighted sum of phase completions
- **Phase Duration:** Time spent in each phase (milliseconds)
- **Phase Units:** Granular completion tracking within phases
- **Completed Phases:** Set of finished pipeline stages

### Heartbeat System (`websocket_service.py:1026`)
- Periodic updates every 10 seconds (configurable)
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
   - Write, Rewrite, Select, Optimize, Compress, Isolate sub-phases
5. Search operations (40% progress)
   - Multiple SEARCH_STARTED/COMPLETED cycles
   - SOURCE_FOUND events for discoveries
6. Analysis phase (10% progress)
   - CREDIBILITY_CHECK events
   - DEDUPLICATION event
7. Synthesis phase (12% progress)
   - SYNTHESIS_STARTED (mapped to research_progress)
   - Section sub-steps broadcast for each section
   - SYNTHESIS_COMPLETED (mapped to research_completed)
8. Complete phase (3% progress)
   - `complete_research` forces the final 100% update
```

This comprehensive tracking ensures users have real-time visibility into every step of the research process, from initial query optimization through final answer synthesis.

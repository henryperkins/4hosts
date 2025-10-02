# Research Progress Tracking: Gaps and Disconnects Analysis

## Critical Gaps Identified

### 1. **No Explicit Ranking Phase Progress**
- **Gap**: While the system performs ranking and scoring of results, there's no dedicated progress reporting for this phase
- **Location**: Result ranking happens in `research_orchestrator.py` but lacks progress updates
- **Impact**: Users don't see progress during result prioritization
- **Fix Needed**: Add progress reports during ranking operations

### 2. **Missing Agentic Loop Progress Updates**
- **Gap**: The `agentic_loop` phase has a 10% weight allocation but NO actual progress reporting
- **Location**: `run_followups()` at `research_orchestrator.py:963` executes without progress updates
- **Evidence**: No calls to `update_progress()` with phase="agentic_loop" found
- **Impact**: Progress stalls at 75% during iterative follow-ups
- **Fix Needed**: Add progress tracking in `agentic_process.py`

### 3. **Credibility Checking Not Reporting Progress** *(Resolved October 2025)*
- **Gap**: Individual credibility checks didn't surface incremental updates, leaving the analysis phase silent.
- **Resolution**: `analyze_source_credibility_batch` now iterates with `asyncio.as_completed` and calls `progress_tracker.update_progress` on each completion (`four-hosts-app/backend/services/credibility.py:1333`). The orchestrator passes the required tracker context at `four-hosts-app/backend/services/research_orchestrator.py:1358`.
- **Telemetry**: The same helper emits cadence metrics (`analysis_metrics`) that flow into `telemetry_pipeline.record_search_run`, powering new Prometheus series: `analysis_phase_duration_seconds`, `analysis_phase_updates_total`, `analysis_phase_updates_per_second`, and related gap gauges.
- **Impact**: Long-running credibility batches now advance progress smoothly and are observable in dashboards; no further backend action required.

### 4. **Classification Phase Partial Coverage**
- **Gap**: Classification reports progress only at 2 points (feature extraction and rule-based)
- **Missing**: No progress for LLM classification step
- **Location**: `classification_engine.py:463` - LLM classification runs without progress
- **Impact**: Progress appears stuck during LLM classification
- **Fix Needed**: Add progress update for LLM classification phase

### 5. **Error Recovery Disconnect**
- **Gap**: `RESEARCH_FAILED` event exists but isn't consistently triggered on failures
- **Location**: Exception handlers throughout `research_orchestrator.py`
- **Evidence**: Many try/except blocks that silently log errors without broadcasting failure
- **Impact**: Frontend doesn't know when research partially fails
- **Fix Needed**: Systematic error reporting to WebSocket

### 6. **Granular Search Progress Missing**
- **Gap**: Individual API calls within search don't report progress
- **Location**: `search_apis.py` - API calls execute without progress updates
- **Evidence**: Only start/complete events, no intermediate progress
- **Impact**: Long searches appear frozen
- **Fix Needed**: Add per-API progress reporting

### 7. **Synthesis Sub-phase Gaps**
- **Gap**: Evidence building and citation compilation lack progress updates
- **Location**: `answer_generator.py` - only reports at phase start/end
- **Evidence**: No granular progress during section generation
- **Impact**: Synthesis appears stuck during long LLM calls
- **Fix Needed**: Add section-by-section progress

### 8. **Phase Transition Race Conditions**
- **Gap**: Phase changes can occur before previous phase marked complete
- **Location**: `websocket_service.py:783-787` - completion tracking logic
- **Evidence**: `completed_phases` set updated asynchronously
- **Impact**: Progress calculation may be incorrect
- **Fix Needed**: Synchronize phase transitions

### 9. **Progress Persistence Gaps**
- **Gap**: Progress snapshots saved every 2 seconds but may miss rapid updates
- **Location**: `websocket_service.py:630` - `_persist_interval_sec`
- **Impact**: Reconnecting clients may see stale progress
- **Fix Needed**: Force persist on phase changes

### 10. **Weight Model Misalignment**
- **Gap**: Some phases have 0% weight (e.g., "complete")
- **Location**: `websocket_service.py:626` - phase weights
- **Evidence**: "complete" phase has 0.00 weight
- **Impact**: Progress jumps from <100% to 100% abruptly
- **Fix Needed**: Redistribute weights or handle terminal state differently

## Architectural Disconnects

### A. **Progress Facade Inconsistency**
- Some services use `progress` facade, others directly import `websocket_service`
- Creates inconsistent error handling patterns
- Recommendation: Enforce facade usage everywhere

### B. **No Progress Backpressure**
- Progress updates are fire-and-forget
- No mechanism to slow down if WebSocket overwhelmed
- Recommendation: Add queue depth monitoring

### C. **Missing Progress Context**
- Progress updates lack correlation IDs
- Hard to trace which operation triggered which update
- Recommendation: Add operation IDs to all progress events

### D. **No Progress Rate Limiting**
- Rapid progress updates can flood WebSocket
- No throttling mechanism per research ID
- Recommendation: Add rate limiting per research session

## Priority Fixes

1. **HIGH**: Add agentic_loop progress reporting
2. **HIGH**: Implement granular synthesis progress
3. **MEDIUM**: Add credibility check progress callbacks
4. **MEDIUM**: Report individual API search progress
5. **LOW**: Fix phase weight model for smooth progression

## Testing Gaps

- No integration tests for progress flow
- No tests for WebSocket message ordering
- No tests for progress calculation accuracy
- No tests for reconnection progress recovery

## Monitoring Gaps

- No metrics on progress update frequency
- No alerts for stuck progress (same percentage >30s)
- No tracking of WebSocket message drops
- No measurement of progress accuracy vs actual completion

# Focused Implementation Assessment: Query Planner System

## 1. PLANNER INTEGRATION ✅ DEEP & ROBUST

### Integration Points
- **Orchestrator**: Creates QueryPlanner with configurable PlannerConfig (research_orchestrator.py:531)
- **Context Engineering**: OptimizeLayer uses planner for variations (context_engineering.py:913)
- **Fallback Handling**: Graceful degradation to rule_based:primary on planner failure (research_orchestrator.py:540-544)

### Configuration Hierarchy
```python
# Orchestrator path (research_orchestrator.py:1757-1772)
_build_planner_config(limit) -> PlannerConfig
  └── build_planner_config(base) # Applies env overrides
      └── Respects: ENABLE_QUERY_LLM, SEARCH_DISABLE_AGENTIC
      └── Stage caps: SEARCH_PLANNER_CONTEXT_CAP

# Context Engineering path (context_engineering.py:872-895)
_planner_config() -> PlannerConfig
  └── CE-specific: CE_PLANNER_MAX_CANDIDATES, CE_PLANNER_STAGE_ORDER
  └── Disables agentic stage for CE use
```

### Quality Indicators
- ✅ Proper async/await patterns throughout
- ✅ Exception handling with fallbacks
- ✅ Configurable via multiple env vars
- ✅ Stage-based priority ranking (paradigm=1.0, rule_based=0.96, llm=0.9)
- ✅ Jaccard deduplication (threshold=0.92)

## 2. MANAGER/PROVIDER PATH ✅ CLEAN EXECUTION

### Provider Integration (search_apis.py)
```python
# Clean planned candidate execution (1186-1239)
BaseSearchAPI.search_with_variations(planned: Sequence[QueryCandidate])
  ├── Concurrent execution (SEARCH_VARIANT_CONCURRENCY=4)
  ├── Per-candidate annotation:
  │   ├── query_stage, query_label, query_weight
  │   └── query_variant = f"{stage}:{label}"
  └── URL deduplication within provider results
```

### Manager Orchestration (search_apis.py)
```python
SearchAPIManager.search_with_priority(planned) # Line 1846
  ├── Primary provider first (Brave)
  ├── Fallback to academic providers if < MIN_RESULTS_THRESHOLD
  └── _search_single_provider delegates to provider.search_with_variations
```

### Key Features
- ✅ Respects provider rate limits and circuit breakers
- ✅ Progress callbacks integrated (report_search_started/completed)
- ✅ Timeout handling (SEARCH_PER_PROVIDER_TIMEOUT_SEC=15)
- ✅ Concurrent variant execution within providers

## 3. ORCHESTRATOR FOLLOW-UPS ✅ COVERAGE-DRIVEN

### Follow-up Logic (research_orchestrator.py:602-670)
```python
if agentic_config["enabled"]:
  coverage_ratio, missing_terms = evaluate_coverage_from_sources()
  while coverage_ratio < threshold and missing_terms:
    followup_candidates = await planner.followups(
      missing_terms=missing_terms,
      coverage_sources=coverage_sources
    )
    # Filter duplicates, execute, update coverage
```

### Coverage Evaluation
- Uses `evaluate_coverage_from_sources()` from agentic_process
- Tracks `coverage_ratio` and `missing_terms`
- Configurable thresholds:
  - `coverage_threshold`: 0.75 default
  - `max_iterations`: From agentic_config
  - `max_new_queries_per_iter`: Caps follow-ups

### ⚠️ GAP: Coverage metrics computed but not exposed
- `coverage_ratio` calculated but not added to search_metrics
- Missing terms tracked but not telemetrized

## 4. TELEMETRY ⚠️ PARTIAL IMPLEMENTATION

### Working Components

#### Redis Persistence (cache.py:345)
```python
async def get_search_metrics_events(limit=200) -> List[Dict]
  └── Reads from "search_metrics:events" list
async def persist_search_metrics(record: Dict)
  └── LPUSH to Redis with 7-day expiry
```

#### /system/search-metrics Endpoint (routes/system.py:215)
```python
GET /system/search-metrics?window_minutes=60&limit=720
  ├── Fetches events from cache_manager
  ├── Aggregates: provider_usage, costs, dedup_rate
  └── Returns timeline + stats
```

#### Metrics Collection (research_orchestrator.py)
```python
search_metrics = {
  "total_queries": len(executed_candidates),
  "total_results": len(combined),
  "apis_used": [...],
  "deduplication_rate": 1.0 - (len(deduped)/len(combined))
}
```

### ⚠️ MISSING Components

1. **NO Prometheus Export**
   - `monitoring.py` has Prometheus setup but NOT wired to search metrics
   - `MetricsFacade` (metrics.py) is internal-only, no export

2. **NO Persistence Calls**
   - Orchestrator computes metrics but doesn't call `persist_search_metrics()`
   - Metrics stay request-scoped only

3. **NO Coverage Telemetry**
   - `coverage_ratio` computed but not added to metrics
   - Follow-up effectiveness not tracked

## 5. CRITICAL GAPS & RISKS 🚨

### HIGH RISK: BudgetAwarePlanner Not Used
```python
# INSTANTIATED but NEVER CALLED (research_orchestrator.py:158)
self.planner = BudgetAwarePlanner(self.tool_registry, self.retry_policy)
# But actual planner is QueryPlanner, not BudgetAwarePlanner!
planner = QueryPlanner(planner_cfg)  # Line 531
```
**Impact**: No spend caps, unlimited API calls possible

### HIGH RISK: No Unit Tests for Critical Components
- `ResultDeduplicator` - Core dedup logic untested
- `EarlyRelevanceFilter` - Filtering logic untested
- Planner stage adapters - Integration untested

### MEDIUM RISK: Metrics Not Persisted
- Dashboard blind to actual usage
- Can't detect degradation post-deploy
- No alerting possible on dedup_rate drops

### MEDIUM RISK: Circular Import Fixed but Fragile
- `paradigm_search` import moved to function scope (variator.py:105)
- Works but violates import best practices
- Could break with future refactoring

## 6. NEXT STEPS (PRIORITY ORDER)

### 🔴 CRITICAL (This Sprint)

1. **Wire Metrics Persistence**
```python
# In research_orchestrator.py after line 780:
await cache_manager.persist_search_metrics({
    **search_metrics_local,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "research_id": research_id,
    "paradigm": paradigm_code,
})
```

2. **Add Coverage to Metrics**
```python
# After line 609:
search_metrics_local["coverage_ratio"] = coverage_ratio
search_metrics_local["missing_terms_count"] = len(missing_terms)
```

3. **Fix BudgetAwarePlanner Usage**
   - Either remove unused instantiation OR
   - Integrate budget checks in follow-up decision

### 🟡 HIGH (Next Sprint)

4. **Add Unit Tests**
```python
# test_deduplicator.py
def test_simhash_deduplication()
def test_jaccard_threshold()

# test_relevance_filter.py
def test_early_filter_threshold()
def test_paradigm_specific_filtering()
```

5. **Complete Prometheus Export**
   - Wire MetricsFacade to prometheus_client
   - Add /metrics endpoint
   - Configure Grafana dashboards

### 🟢 MEDIUM (Backlog)

6. **Env Consolidation**
   - Migrate to UNIFIED_QUERY_* variables
   - Document all configuration knobs
   - Add validation for conflicting settings

7. **Improve Import Structure**
   - Move paradigm_search types to separate module
   - Eliminate circular dependencies properly

## Summary

The query planner implementation is **functionally complete and operational**, with clean integration throughout the stack. The planned candidates flow correctly from planner → orchestrator → manager → providers, with proper stage:label attribution.

**Critical gaps** exist in telemetry persistence (metrics computed but not saved) and budget enforcement (BudgetAwarePlanner instantiated but unused). These are **configuration/wiring issues** rather than architectural problems, making them relatively quick fixes.

The system successfully handles production traffic with good fallback behavior, but lacks observability for detecting regressions. Immediate focus should be on persisting metrics and adding coverage telemetry, followed by unit tests for the high-risk deduplication and filtering components.
# Implementation Action Plan

## 1. Fix PlannerConfig Stage Order Contribution

### Issue
Context/agentic stages may not contribute if not properly configured in orchestrator.

### Fix Required
**File**: `four-hosts-app/backend/services/research_orchestrator.py:650-654`
```python
def _build_planner_config(self, query_limit: int) -> PlannerConfig:
    """Build planner configuration"""
    base_config = PlannerConfig(
        max_candidates=query_limit,
        enable_agentic=self.agentic_config.get("enabled", False),
        # ADD: Ensure context stage is enabled when we have refined queries
        per_stage_caps={"context": 6}  # Allow context stage to contribute
    )
    return build_planner_config(base=base_config)
```

## 2. Fix OptimizeLayer Caching

### Issue
Cache can become stale when environment variables change.

### Fix Required
**File**: `four-hosts-app/backend/services/context_engineering.py:946-954`
```python
# REPLACE the caching logic with:
async def process(self, classification, previous_outputs=None):
    # ... existing code ...
    planner_cfg = self._planner_config()

    # Option 1: Simple - always create new planner (safest)
    planner = QueryPlanner(planner_cfg)

    # Option 2: Improved caching with env var tracking
    # env_hash = hash(frozenset({
    #     k: v for k, v in os.environ.items()
    #     if k.startswith(("UNIFIED_QUERY_", "CE_PLANNER_"))
    # }.items()))
    # signature = (*self._planner_signature(planner_cfg), env_hash)
    # if not self._cached_planner or signature != self._cached_planner_signature:
    #     planner = QueryPlanner(planner_cfg)
    #     self._cached_planner = planner
    #     self._cached_planner_signature = signature
    # else:
    #     planner = self._cached_planner
```

## 3. Tune Deduplication Thresholds

### Issue
Need provider-specific thresholds for Exa vs traditional search.

### Fix Required
**File**: `four-hosts-app/backend/services/query_planning/result_deduplicator.py:119-126`
```python
@staticmethod
def _adaptive_threshold(result: ResultAdapter, reference: ResultAdapter) -> int:
    domain = result.domain or reference.domain
    result_type = result.get("result_type", reference.get("result_type", "web"))

    # ADD: Provider-specific thresholds
    provider = result.get("provider") or reference.get("provider")
    if provider == "exa":
        return 6  # Exa has richer content, needs looser threshold

    if result_type == "academic" or (domain and (domain.endswith(".edu") or domain.endswith(".gov"))):
        return 8
    if result_type in {"news", "blog"}:
        return 5
    return 3
```

**File**: Add new config for Jaccard thresholds
```python
# In result_deduplicator.py __init__
def __init__(self, similarity_threshold: float | None = None) -> None:
    # Existing code...

    # ADD: Provider-specific Jaccard thresholds
    self.provider_thresholds = {
        "exa": float(os.getenv("DEDUP_SIMILARITY_THRESH_EXA", "0.85")),
        "default": self.similarity_threshold
    }

# In _calculate_content_similarity, use provider-specific threshold:
def _calculate_content_similarity(self, a: ResultAdapter, b: ResultAdapter) -> float:
    # ... existing calculation ...
    provider = a.get("provider", "default")
    threshold = self.provider_thresholds.get(provider, self.similarity_threshold)
    return similarity_score
```

## 4. Fix Search Manager Query Attribution

### Issue
No clear cost attribution per stage/label.

### Fix Required
**File**: `four-hosts-app/backend/services/search_apis.py:2576-2583`
```python
# Organize results by candidate label WITH cost attribution
for result in stage_results or []:
    label = (result.raw_data or {}).get("query_label")
    stage = (result.raw_data or {}).get("query_stage")

    # ADD: Cost attribution
    if not result.raw_data:
        result.raw_data = {}

    result.raw_data["cost_attribution"] = {
        "stage": stage,
        "label": label,
        "provider": result.raw_data.get("provider", "unknown"),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if isinstance(label, str):
        results_by_label.setdefault(label, []).append(result)

    # Existing stage:label annotation
    if isinstance(label, str) and isinstance(stage, str):
        result.raw_data["stage_label"] = f"{stage}:{label}"
```

## 5. Add Agentic Loop Telemetry

### Fix Required
**File**: `four-hosts-app/backend/services/agentic_process.py:273-307`
```python
# After line 273 (evaluate_coverage_from_sources)
logger.info(
    "agentic_coverage_evaluation",
    stage="initial",
    coverage_ratio=coverage_ratio,
    coverage_threshold=coverage_threshold,
    missing_terms_count=len(missing_terms),
    will_trigger_followup=coverage_ratio < coverage_threshold and missing_terms,
    paradigm=paradigm,
    research_id=context.get("research_id"),
)

# After line 402 (second evaluate_coverage call)
logger.info(
    "agentic_coverage_evaluation",
    stage=f"round_{round_num}",
    coverage_ratio=coverage_ratio,
    coverage_threshold=coverage_threshold,
    missing_terms_count=len(missing_terms),
    will_continue=round_num < max_rounds and coverage_ratio < coverage_threshold,
    paradigm=paradigm,
    research_id=context.get("research_id"),
    queries_generated=len(proposed) if proposed else 0,
)

# At function exit (line 428)
logger.info(
    "agentic_loop_complete",
    initial_coverage=initial_coverage,
    final_coverage=coverage_ratio,
    rounds_executed=round_num,
    total_followup_queries=sum(len(v) for v in followup_results.values()),
    threshold_met=coverage_ratio >= coverage_threshold,
    paradigm=paradigm,
    research_id=context.get("research_id"),
)
```

### New Test File
**File**: `four-hosts-app/backend/tests/test_agentic_telemetry.py`
```python
import pytest
from unittest.mock import Mock, patch
from services.agentic_process import run_followups

@pytest.mark.asyncio
async def test_agentic_loop_telemetry():
    """Verify telemetry is emitted for agentic follow-up loops."""

    with patch("services.agentic_process.logger") as mock_logger:
        # Mock low coverage scenario
        mock_sources = [
            {"title": "Test", "snippet": "partial info"}
        ]

        result = await run_followups(
            seed_query="complex technical query",
            paradigm="bernard",
            coverage_sources=mock_sources,
            planner=Mock(),
            search_fn=Mock(return_value=[]),
            context={},
            coverage_threshold=0.9,  # High threshold to force follow-up
        )

        # Verify telemetry calls
        telemetry_calls = [
            call for call in mock_logger.info.call_args_list
            if "agentic" in str(call)
        ]

        assert len(telemetry_calls) >= 2  # Initial and complete

        # Verify initial evaluation logged
        initial_call = telemetry_calls[0]
        assert "agentic_coverage_evaluation" in str(initial_call)
        assert "stage" in initial_call[1]

        # Verify completion logged
        complete_call = telemetry_calls[-1]
        assert "agentic_loop_complete" in str(complete_call)

@pytest.mark.asyncio
async def test_agentic_loop_coverage_threshold():
    """Test that follow-ups trigger correctly based on coverage."""

    # Test with high coverage (no follow-up expected)
    high_coverage_sources = [
        {"title": f"Result {i}", "snippet": "comprehensive information"}
        for i in range(10)
    ]

    result = await run_followups(
        seed_query="simple query",
        paradigm="teddy",
        coverage_sources=high_coverage_sources,
        planner=Mock(),
        search_fn=Mock(),
        context={},
        coverage_threshold=0.5,  # Low threshold, easily met
    )

    # Should not trigger follow-ups
    _, followup_results, _, _, _ = result
    assert len(followup_results) == 0
```

## Testing Commands

```bash
# Test stage contribution
pytest tests/test_query_planner_basic.py -k "test_all_stages" -v

# Test deduplication
pytest tests/test_result_deduplicator.py -v

# Test agentic telemetry
pytest tests/test_agentic_telemetry.py -v

# Integration test with logging
UNIFIED_QUERY_ENABLE_LLM=true \
UNIFIED_QUERY_ENABLE_FOLLOW_UP=true \
pytest tests/test_orchestrator_p0.py -k "agentic" -v -s
```

## Monitoring Dashboard Queries

```sql
-- Agentic loop activation frequency
SELECT
    DATE(timestamp) as day,
    paradigm,
    COUNT(*) FILTER (WHERE will_trigger_followup) as followups_triggered,
    COUNT(*) as total_queries,
    AVG(coverage_ratio) as avg_initial_coverage
FROM logs
WHERE message = 'agentic_coverage_evaluation'
  AND stage = 'initial'
GROUP BY 1, 2;

-- Coverage improvement by round
SELECT
    paradigm,
    stage,
    AVG(coverage_ratio) as avg_coverage,
    COUNT(*) as count
FROM logs
WHERE message = 'agentic_coverage_evaluation'
GROUP BY 1, 2
ORDER BY 1, 2;

-- Deduplication effectiveness
SELECT
    DATE(timestamp) as day,
    AVG(duplicates_removed::float / input_count) as dedup_rate,
    AVG(duplicates_removed) as avg_duplicates,
    COUNT(*) as queries
FROM logs
WHERE stage = 'deduplication'
GROUP BY 1;
```
# Query Planner Implementation Analysis

## 1. PlannerConfig Stage Order and Contribution

### Current Implementation
- **Default stage_order**: `["rule_based", "paradigm", "context", "llm", "agentic"]` (query_planning/types.py:29-35)
- **Stage priors (weights)**: paradigm=1.0, rule_based=0.96, llm=0.9, context=0.88, agentic=0.86

### Key Findings
✅ **Context/Agentic stages DO contribute** when enabled:
- The planner dynamically builds stage generators based on config flags (planner.py:89-124)
- Context stage only generates when `additional_queries` are provided AND `per_stage_caps["context"] > 0`
- Agentic stage only generates when `enable_agentic=True` AND `per_stage_caps["agentic"] > 0`
- Environment variables properly override defaults via `build_planner_config()` (config.py)

### Issues Identified
⚠️ **Conditional activation**: Stages only contribute if:
1. They appear in `stage_order` (configurable)
2. Their per-stage cap is > 0
3. Their enable flag is True (for llm/agentic)
4. For context: additional_queries must be provided

## 2. OptimizeLayer Caching Strategy

### Current Implementation (context_engineering.py:877-954)
```python
self._cached_planner: Optional[QueryPlanner] = None
self._cached_planner_signature: Optional[Tuple[Any, ...]] = None
```

### Issues Identified
⚠️ **Cache invalidation risk**: The cache uses a signature tuple but:
- Environment variables can change between requests
- Config changes may not invalidate cache properly
- Line 954: `planner.cfg = planner_cfg` updates config but doesn't rebuild stages

### Recommendations
```python
# Option 1: Always rebuild (simple, safe)
planner = QueryPlanner(planner_cfg)

# Option 2: Improve signature to include env vars
signature = (cfg_tuple, os.environ.get("UNIFIED_QUERY_*"))

# Option 3: Add explicit invalidation
if config_changed():
    self._cached_planner = None
```

## 3. Deduplication Thresholds

### Current Implementation
- **Query deduplication** (planner.py): Jaccard threshold = 0.92 default
- **Result deduplication** (result_deduplicator.py):
  - SimHash hamming distance thresholds: academic=8, news/blog=5, default=3
  - Content similarity Jaccard: 0.8 default (configurable via DEDUP_SIMILARITY_THRESH)

### Issues with Merged Exa/Deep-Research Results
⚠️ **Potential over/under-deduplication**:
- Different result types use different thresholds
- No specific handling for Exa vs traditional search results
- Merged results may have different content patterns

### Recommendations
```python
# Add provider-specific thresholds
PROVIDER_THRESHOLDS = {
    "exa": 0.85,      # Exa tends to have richer snippets
    "brave": 0.92,    # Standard web results
    "arxiv": 0.95,    # Academic papers need higher threshold
}

# Benchmark with actual data
async def benchmark_dedup():
    # Compare dedup rates across providers
    # Measure false positives/negatives
    # Tune thresholds accordingly
```

## 4. Search Manager Query Mapping & Cost Attribution

### Current Implementation (search_apis.py:2526-2590)
```python
async def search_with_plan(planned, config):
    # Pre-populate all labels
    results_by_label = {c.label: [] for c in planned}

    # Single priority call
    stage_results = await self.search_with_priority(planned, config)

    # Organize by label
    for result in stage_results:
        label = result.raw_data.get("query_label")
        stage = result.raw_data.get("query_stage")
        results_by_label[label].append(result)
```

### Potential Issues
✅ **Mapping appears correct**: Labels are preserved through raw_data
⚠️ **Cost attribution unclear**: No explicit cost tracking per stage/label

### Recommendations
```python
# Add cost tracking
result.raw_data["cost_attribution"] = {
    "provider": provider_name,
    "stage": stage,
    "label": label,
    "api_calls": 1,
    "tokens_used": token_count,
}
```

## 5. Agentic Follow-up Loop Telemetry

### Current Implementation (agentic_process.py:211-430)
- Coverage threshold: 0.75 default
- Max rounds: 3
- Missing terms evaluation drives follow-ups

### Telemetry Gaps
⚠️ **Limited visibility into**:
- How often coverage threshold is met initially
- Number of rounds typically needed
- Which paradigms trigger follow-ups most
- Success rate of follow-up queries

### Recommendations
```python
# Add comprehensive telemetry
logger.info(
    "agentic_followup_metrics",
    initial_coverage=coverage_ratio,
    threshold=coverage_threshold,
    missing_terms_count=len(missing_terms),
    rounds_executed=round_num,
    paradigm=paradigm,
    followup_triggered=coverage_ratio < coverage_threshold,
    final_coverage=final_coverage,
)

# Add test coverage
@pytest.mark.asyncio
async def test_agentic_loop_triggers():
    # Test with low initial coverage
    # Verify follow-ups are triggered
    # Check telemetry is emitted
```

## Priority Actions

1. **HIGH**: Fix OptimizeLayer caching - either remove caching or improve invalidation
2. **HIGH**: Add telemetry for agentic loop to understand real-world usage
3. **MEDIUM**: Benchmark dedup thresholds with actual Exa/deep-research data
4. **MEDIUM**: Add cost attribution tracking to search results
5. **LOW**: Verify stage_order configuration is documented for operators

## Testing Recommendations

```python
# Test 1: Verify stages contribute
async def test_all_stages_contribute():
    cfg = PlannerConfig(
        stage_order=["rule_based", "paradigm", "context", "llm", "agentic"],
        enable_llm=True,
        enable_agentic=True,
    )
    planner = QueryPlanner(cfg)
    candidates = await planner.initial_plan(
        seed_query="test",
        paradigm="bernard",
        additional_queries=["extra1", "extra2"],
    )
    # Assert all stages produced candidates
    stages_present = {c.stage for c in candidates}
    assert stages_present == {"rule_based", "paradigm", "context", "llm", "agentic"}

# Test 2: Agentic loop coverage
async def test_agentic_followup_triggers():
    # Mock low initial coverage
    # Verify follow-up is triggered
    # Check telemetry emission
```
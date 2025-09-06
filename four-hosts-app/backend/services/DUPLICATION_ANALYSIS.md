# Research Orchestrator - Code Duplication Analysis

## Executive Summary
The `research_orchestrator.py` file contains ~3000 lines with approximately 400-500 lines of duplicated code that can be eliminated through refactoring.

## Major Duplication Patterns Identified

### 1. Search Execution with Retry Logic (≈150 lines duplicated)
**Locations:**
- `_execute_searches_deterministic()` lines 1741-1874 (V2 path)
- Loop in `_execute_paradigm_research_legacy()` lines 1149-1257 (Legacy path)

**Pattern:** Both implement identical retry/backoff logic, timeout handling, and metric tracking.

**Solution:** Extract a single helper:
```python
async def execute_search_with_retry(
    query: str, 
    api: str, 
    config: SearchConfig,
    retry_policy: RetryPolicy,
    metrics: Dict
) -> List[SearchResult]
```

### 2. Deduplication Implementations (≈200 lines duplicated)
**Three separate implementations:**
1. `ResultDeduplicator.deduplicate_results()` lines 157-229
2. `DeterministicMerger.merge_results()` lines 105-148  
3. `_merge_duplicate_results()` lines 2360-2380

**Solution:** Keep only `ResultDeduplicator` as the single source of truth.

### 3. Environment Variable Parsing (≈15 lines duplicated)
**`SEARCH_MIN_RELEVANCE_BERNARD` parsed 3 times:**
- Lines 1201-1205
- Lines 1429-1434
- Lines 1796-1800

**Solution:**
```python
def get_bernard_min_relevance() -> float:
    """Get Bernard paradigm minimum relevance threshold."""
    try:
        return float(os.getenv("SEARCH_MIN_RELEVANCE_BERNARD", "0.15"))
    except (ValueError, TypeError):
        return 0.15
```

### 4. Credibility Score Calculation (≈30 lines duplicated)
**Computed twice:**
- `_execute_paradigm_research_legacy()` lines 1503-1515
- `_process_results()` lines 1980-1987

**Solution:** Calculate once in post-processing phase only.

### 5. Early Relevance Filtering (≈80 lines duplicated)
**Two implementations:**
- `EarlyRelevanceFilter.is_relevant()` lines 300-351
- Theme overlap in `_apply_early_relevance_filter()` lines 2429-2458

**Solution:** Merge theme overlap logic into `EarlyRelevanceFilter` class.

### 6. Query Compression & Limiting (≈40 lines duplicated)
**Two paths:**
- V2: lines 723-748 using `query_compressor.compress()`
- Legacy: lines 1121-1141 with different logic

**Solution:**
```python
def prepare_search_queries(
    context_engineered: Any,
    user_context: Any,
    max_queries: Optional[int] = None
) -> List[str]
```

### 7. Placeholder URL Generation (≈10 lines duplicated)
**Deep research citations:**
- `normalize_result_shape()` lines 2484-2487
- `_execute_deep_research_integration()` lines 2612-2615

**Solution:**
```python
def generate_citation_url(
    title: str,
    snippet: str,
    origin_id: str = "deep"
) -> str:
    """Generate stable placeholder URL for unlinked citations."""
    hash_input = (title or snippet or '')[:64]
    hash_val = hash(hash_input) & 0xFFFFFFFF
    return f"about:blank#citation-{origin_id}-{hash_val}"
```

### 8. SearchConfig Construction (≈25 lines duplicated)
**Created 4+ times with similar patterns:**
- Lines 1207-1215
- Line 1435
- Lines 1555-1562
- Line 1801

**Solution:**
```python
def create_search_config(
    paradigm: str,
    max_results: int = 20,
    preferences: Optional[Dict] = None
) -> SearchConfig:
    """Create paradigm-aware search configuration."""
    min_relevance = get_bernard_min_relevance() if paradigm == "bernard" else 0.25
    return SearchConfig(
        max_results=max_results,
        language="en",
        region="us",
        min_relevance_score=min_relevance,
        source_types=preferences.get("source_types", []) if preferences else [],
        exclusion_keywords=preferences.get("exclusions", []) if preferences else []
    )
```

### 9. Result to Dict Conversion (≈100 lines duplicated)
**Multiple implementations:**
- Lines 2146-2158 (compress_search_results prep)
- Lines 2639-2680 (_convert_legacy_to_v2_result)
- Lines 898-922 (_synthesize_answer)

**Solution:**
```python
def normalize_result_to_dict(
    result: Any,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """Convert any result type to normalized dict."""
    # Single implementation handling all cases
```

### 10. Progress Tracking Calls (≈50 lines scattered)
**Inconsistent error handling throughout**

**Solution:** Create wrapper with consistent error handling:
```python
async def report_progress_safe(
    tracker: Any,
    research_id: str,
    **kwargs
) -> None:
    """Report progress with consistent error handling."""
    if not tracker or not research_id:
        return
    try:
        await tracker.update_progress(research_id, **kwargs)
    except Exception as e:
        logger.debug(f"Progress update failed: {e}")
```

## Impact Analysis

### Lines of Code Reduction
- **Current:** ~3000 lines
- **After refactoring:** ~2600 lines
- **Reduction:** ~400 lines (13%)

### Benefits
1. **Consistency:** Single source of truth for each operation
2. **Maintainability:** Bug fixes apply everywhere
3. **Testing:** Test helpers once, use everywhere
4. **Performance:** Easier to optimize single implementations
5. **Readability:** Clearer separation of concerns

### Risk Assessment
- **Low Risk:** Helper extraction (env vars, URLs, configs)
- **Medium Risk:** Consolidating deduplication (needs careful testing)
- **Higher Risk:** Merging search execution paths (behavioral differences)

## Recommended Refactoring Order

1. **Phase 1 - Quick Wins (Low Risk)**
   - Extract env var helpers
   - Create placeholder URL generator
   - Extract search config builder
   - Add progress tracking wrapper

2. **Phase 2 - Consolidation (Medium Risk)**
   - Unify deduplication strategies
   - Merge early filtering implementations
   - Consolidate result normalization

3. **Phase 3 - Major Refactoring (Higher Risk)**
   - Extract common search execution logic
   - Unify query preparation
   - Consider removing legacy path once V2 stable

## Code Quality Metrics

### Duplication Metrics
- **Exact duplicates:** ~100 lines
- **Near duplicates:** ~300 lines  
- **Structural duplicates:** ~100 lines

### Complexity Reduction
- **Cyclomatic complexity:** Would reduce by ~25%
- **Cognitive complexity:** Would reduce by ~30%
- **Test coverage:** Would improve due to focused testing

## Next Steps

1. Create `research_helpers.py` module for extracted utilities
2. Add comprehensive tests for each helper
3. Refactor incrementally with verification at each step
4. Update all callers to use new helpers
5. Remove deprecated code paths
6. Document migration guide for API consumers
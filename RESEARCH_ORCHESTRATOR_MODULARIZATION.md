# Research Orchestrator Modularization Plan

## Overview

This document outlines the strategy for breaking down the monolithic `research_orchestrator.py` file (3,160+ lines) into smaller, focused modules while avoiding duplication with existing services.

## Current State Analysis

### File Structure
- **research_orchestrator.py**: 3,160+ lines with 9 main classes
- **Primary Issues**: Monolithic structure, mixed responsibilities, duplicate functionality with other services

### Key Classes Currently in research_orchestrator.py
1. `ResearchExecutionResult` (data class)
2. `ResultDeduplicator` (advanced simhash-based deduplication)
3. `EarlyRelevanceFilter` (spam detection, relevance filtering)
4. `CostMonitor` (API cost tracking)
5. `RetryPolicy` (retry logic)
6. `ToolRegistry` (tool management)
7. `Budget` / `BudgetAwarePlanner` (budget management)
8. `UnifiedResearchOrchestrator` (main orchestrator - 2,500+ lines)

## Identified Duplications with Existing Services

### 1. Deduplication Logic
- **research_orchestrator.py**: Advanced `ResultDeduplicator` with simhash, content similarity
- **search_apis.py**: Basic URL deduplication (lines 730-745)
- **Resolution**: Remove basic version, use advanced one consistently

### 2. Relevance Filtering
- **research_orchestrator.py**: `EarlyRelevanceFilter` with spam detection
- **search_apis.py**: `ContentRelevanceFilter` with term frequency analysis
- **Resolution**: Merge into unified filter

### 3. Cost Tracking
- **research_orchestrator.py**: `CostMonitor` class
- **deep_research_service.py`: `_estimate_costs()` method
- **metrics.py`: Token accounting and usage tracking
- **Resolution**: Extend existing `metrics.py`

### 4. Query Optimization
- **research_orchestrator.py**: Query compression (`_compress_and_dedup_queries`) and prioritization logic
- **search_apis.py**: `QueryOptimizer` class with cleaning, expansion, and key term extraction
- **llm_query_optimizer.py**: LLM-powered semantic variations
- **paradigm_search.py**: Paradigm-specific query generation strategies
- **Resolution**: Keep query compression in orchestrator (no duplication), use existing QueryOptimizer for cleaning/expansion

### 5. Additional Service Clarifications

#### No Duplication With:
- **llm_critic.py**: Provides LLM-based coverage and claim consistency checks. Used by orchestrator but no overlapping functionality.
- **credibility.py**: Comprehensive credibility scoring with bias analysis. Used by orchestrator but no duplication.
- **paradigm_search.py**: Paradigm-specific query generation and result filtering. Orchestrator uses these strategies but doesn't duplicate them.

## Modularization Strategy

### Phase 1: Extract New Services (No Overlap)

#### 1. services/result_deduplicator.py
**Purpose**: Advanced result deduplication using simhash and content similarity
**Extract from**: `ResultDeduplicator` class (lines 126-328)
**Why New**: No equivalent advanced deduplication exists

```python
# services/result_deduplicator.py
class ResultDeduplicator:
    async def deduplicate_results(self, results: List[SearchResult]) -> DeduplicationResult

class DeduplicationResult(TypedDict):
    unique_results: List[SearchResult]
    duplicates_removed: int
    similarity_threshold: float
```

#### 2. services/relevance_filter.py
**Purpose**: Unified relevance filtering combining existing filters
**Extract from**: Merge `EarlyRelevanceFilter` + `ContentRelevanceFilter`
**Why New**: Replaces both existing filters with unified approach

```python
# services/relevance_filter.py
class UnifiedRelevanceFilter:
    async def filter_results(
        self,
        results: List[SearchResult],
        query: str,
        paradigm: str
    ) -> FilteredResults

    # Methods from both filters:
    - is_relevant() with spam detection
    - calculate_relevance_score() with term frequency
    - paradigm_specific_filtering()
```

#### 3. services/synthesis_orchestrator.py
**Purpose**: Coordinate answer generation and evidence building
**Extract from**: `_synthesize_answer()` method (lines 1644-2000+)
**Why New**: No existing synthesis coordination service

```python
# services/synthesis_orchestrator.py
class SynthesisOrchestrator:
    async def synthesize_answer(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        results: List[SearchResult],
        research_id: str,
        options: Dict[str, Any]
    ) -> SynthesizedAnswer
```

### Phase 2: Extend Existing Services

#### 4. Extend metrics.py with Cost Tracking
**Instead of**: Creating new `cost_monitor.py`
**Add to**: Existing `metrics.py`

```python
# Add to services/metrics.py
class CostTracker:
    async def track_api_cost(self, api_name: str, cost: float, queries_count: int)
    async def get_daily_api_costs(self, date: Optional[str] = None) -> Dict[str, float]
    async def estimate_and_track_aggregate(self, providers: List[str], queries_count: int) -> Dict[str, float]
```

#### 5. Extend search_apis.py SearchManager
**Instead of**: Creating new `search_executor.py`
**Add to**: Existing `SearchManager` class

```python
# Add to services/search_apis.py
class SearchManager:
    # Existing methods...

    async def execute_with_research_context(
        self,
        queries: List[str],
        paradigm: HostParadigm,
        user_context: Any,
        research_id: Optional[str] = None
    ) -> Dict[str, List[SearchResult]]
```

#### 6. Query Compression Utilities (Lightweight)
**Note**: Query compression logic is unique to orchestrator and doesn't duplicate existing services
**Keep in**: orchestrator or extract to lightweight utility

```python
# services/query_utils.py (lightweight)
def compress_and_dedup_queries(queries: List[str]) -> List[str]:
    """Simple query compression - no overlap with QueryOptimizer cleaning/expansion"""

def prioritize_queries(
    queries: List[str],
    limit: int,
    paradigm: Optional[str] = None
) -> List[str]:
    """Simple prioritization - uses paradigm_search strategies but doesn't duplicate them"""
```

### Phase 3: Create Lightweight Utilities

#### 7. services/planning_utils.py
**Purpose**: Simple planning utilities without overlapping complex logic
**Extract from**: Budget, RetryPolicy, ToolRegistry (simplified)

```python
# services/planning_utils.py
@dataclass
class BudgetConfig:
    max_tokens: int
    max_cost_usd: float
    max_wallclock_minutes: int

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay_sec: float = 0.5
    max_delay_sec: float = 8.0
```

### Phase 4: Research-Specific Services

#### 8. services/deep_research_orchestrator.py
**Purpose**: Coordinate deep research iterations
**Extract from**: `_execute_deep_research_integration()` method

```python
# services/deep_research_orchestrator.py
class DeepResearchOrchestrator:
    async def execute_deep_research(
        self,
        context_engineered: ContextEngineeredQuerySchema,
        classification: ClassificationResultSchema,
        mode: DeepResearchMode,
        research_id: str
    ) -> List[Dict[str, Any]]
```

#### 9. services/exa_research_orchestrator.py
**Purpose**: Coordinate Exa research integration
**Extract from**: `_augment_with_exa_research()` method

```python
# services/exa_research_orchestrator.py
class ExaResearchOrchestrator:
    async def augment_with_exa(
        self,
        processed_results: Dict[str, Any],
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        research_id: str
    ) -> None
```

## Refactored Main Orchestrator

### services/research_orchestrator_v2.py
**Reduced from 3,160+ lines to ~500 lines**

```python
# services/research_orchestrator_v2.py
class UnifiedResearchOrchestrator:
    def __init__(self):
        # Inject dependencies
        self.deduplicator = ResultDeduplicator()
        self.relevance_filter = UnifiedRelevanceFilter()
        self.cost_tracker = metrics.CostTracker()
        self.search_manager = SearchManager()
        self.query_optimizer = QueryOptimizer()
        self.synthesis_orchestrator = SynthesisOrchestrator()
        self.deep_research_orchestrator = DeepResearchOrchestrator()
        self.exa_orchestrator = ExaResearchOrchestrator()

    async def execute_research(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any,
        # ... other parameters
    ) -> Dict[str, Any]:
        """Main orchestration method - simplified coordination"""

        # Step 1: Get source queries from context engineering
        source_queries = getattr(context_engineered, "refined_queries", []) or [context_engineered.original_query]

        # Step 2: Compress and dedup queries (orchestrator-specific)
        optimized_queries = compress_and_dedup_queries(source_queries)

        # Step 3: Prioritize queries (uses paradigm_search strategies)
        limited = self._get_query_limit(user_context)
        prioritized_queries = prioritize_queries(
            optimized_queries,
            limited,
            classification.primary_paradigm.value
        )

        # Step 4: Execute searches using paradigm strategies
        from services.paradigm_search import get_search_strategy, SearchContext
        strategy = get_search_strategy(classification.primary_paradigm)
        context = SearchContext(
            original_query=context_engineered.original_query,
            paradigm=classification.primary_paradigm.value
        )

        # Use existing SearchManager for execution
        search_results = await self.search_manager.execute_with_research_context(
            [q['query'] for q in prioritized_queries],
            classification.primary_paradigm,
            user_context,
            research_id
        )

        # Step 5: Filter results using unified filter
        filtered_results = await self.relevance_filter.filter_results(
            search_results,
            context_engineered.original_query,
            classification.primary_paradigm.value
        )

        # Step 6: Deduplicate using advanced deduplicator
        dedup_result = await self.deduplicator.deduplicate_results(filtered_results)

        # ... rest of orchestration using extracted services
```

## Implementation Steps

### Step 1: Remove Duplications
1. Remove basic deduplication from `search_apis.py:730-745`
2. Remove `_estimate_costs()` from `deep_research_service.py`
3. Update imports to use new unified services

### Step 2: Create New Modules
1. Create `result_deduplicator.py`
2. Create `relevance_filter.py` (merging existing filters)
3. Create `synthesis_orchestrator.py`
4. Create research-specific orchestrators

### Step 3: Extend Existing Services
1. Add cost tracking to `metrics.py`
2. Extend `SearchManager` in `search_apis.py`
3. Note: Query optimization already properly separated - no extension needed

### Step 4: Create Main Orchestrator v2
1. Create `research_orchestrator_v2.py`
2. Implement dependency injection
3. Migrate tests
4. Update imports in calling code

### Step 5: Deprecation and Migration
1. Keep `research_orchestrator.py` for backward compatibility
2. Add deprecation warnings
3. Gradually migrate callers to new orchestrator
4. Remove old orchestrator in future release

## Benefits

### Maintainability Improvements
- **70% reduction** in main orchestrator size (3,160 → ~500 lines)
- **Clear separation** of concerns
- **Easier testing** of individual components
- **Reduced cognitive load** for developers

### Code Quality Improvements
- **Eliminated duplication** across services
- **Standardized interfaces** for common operations
- **Better error handling** consistency
- **Improved type safety**

### Performance Benefits
- **Lazy loading** of components
- **Reduced memory footprint**
- **Better caching** opportunities
- **Optimized imports**

## Risk Mitigation

### Backward Compatibility
1. Maintain old orchestrator during transition
2. Use feature flags for gradual rollout
3. Provide migration guide

### Testing Strategy
1. Comprehensive unit tests for each new module
2. Integration tests for service interactions
3. Performance regression testing
4. Load testing with extracted services

### Rollback Plan
1. Quick rollback by switching back to old orchestrator
2. Environment variable controls
3. Monitoring for performance regression
4. Automated canary deployments

## Timeline Estimate

- **Phase 1**: 2-3 days (extract new services)
- **Phase 2**: 1-2 days (extend existing services)
- **Phase 3**: 1 day (create utilities)
- **Phase 4**: 2-3 days (research orchestrators)
- **Testing**: 3-4 days
- **Migration**: 1-2 days

**Total Estimated Time**: 10-15 days

## Success Metrics

1. **Code Quality**
   - Cyclomatic complexity < 10 for all methods
   - 90%+ test coverage
   - Zero duplicate functionality

2. **Performance**
   - No performance regression
   - Faster cold starts
   - Reduced memory usage

3. **Developer Experience**
   - Faster onboarding of new developers
   - Easier debugging
   - Clearer code ownership

## Dependencies

### External Dependencies
- None - uses existing libraries

### Internal Dependencies
- `search_apis.py` - will be extended
- `metrics.py` - will be extended
- `result_adapter.py` - will be used by new modules
- `credibility.py` - no changes needed
- `cache.py` - no changes needed

## Conclusion

This modularization plan significantly improves code maintainability while carefully avoiding duplication with existing services. By extending rather than replacing existing functionality, we maintain stability while creating a more modular architecture.

## Summary of Non-Duplication

### Services Used But Not Duplicated:
- **llm_critic.py**: Orchestrator calls `llm_coverage_and_claims()` when enabled
- **llm_query_optimizer.py**: Used via `propose_semantic_variations()` (controlled by env flag)
- **paradigm_search.py**: Used via `get_search_strategy()` and strategy classes
- **credibility.py**: Used via `get_source_credibility()` calls

### Functionality Properly Separated:
- **Query cleaning/expansion**: Remains in `search_apis.py` QueryOptimizer
- **Semantic variations**: Remains in `llm_query_optimizer.py`
- **Paradigm strategies**: Remains in `paradigm_search.py`
- **Credibility scoring**: Remains in `credibility.py`
- **LLM criticism**: Remains in `llm_critic.py`
- **Query compression**: Unique to orchestrator (no duplication)
- **Cost tracking**: Consolidated in `metrics.py`
- **Result deduplication**: Extracted to dedicated service
- **Relevance filtering**: Merged into unified service
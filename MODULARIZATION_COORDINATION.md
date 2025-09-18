# Coordinated Modularization Strategy

## Overview

This document explains how the **Research Orchestrator Modularization** and **Query Planning Consolidation** plans work together to create a unified, maintainable architecture.

## Relationship Between Plans

### Research Orchestrator Modularization
- **Focus**: Breaking down the monolithic `research_orchestrator.py` (3,160+ lines)
- **Scope**: Orchestration-level concerns (result processing, deduplication, synthesis)
- **Output**: Smaller, focused orchestration services

### Query Planning Consolidation
- **Focus**: Unifying four overlapping query planning systems
- **Scope**: Query-level concerns (cleaning, expansion, variation generation)
- **Output**: Single, unified query planning system

## How They Interact

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Orchestrator                     │
│                      (Modularized)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Context Engineering (W-S-C-I)                  │
│                          │                                  │
│                          ▼                                  │
│                 Unified Query Planner                       │
│                 (Consolidated System)                       │
│                          │                                  │
│          ┌───────────────┼───────────────┐                  │
│          ▼               ▼               ▼                  │
│    Query Cleaning   Variations    Coverage Analysis        │
│          │               │               │                  │
│          └───────────────┼───────────────┘                  │
│                          ▼                                  │
│              Optimized Query Set                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Search Execution                          │
└─────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Context Engineering Layer
```python
# context_engineering.py (after consolidation)
from services.query_planning.planner import UnifiedQueryPlanner

class OptimizeLayer:
    def __init__(self):
        # Old: Multiple separate systems
        # self.query_optimizer = QueryOptimizer()
        # self.llm_optimizer = None  # Optional

        # New: Single unified system
        self.query_planner = UnifiedQueryPlanner()

    async def optimize(self, query: str, paradigm: str) -> ContextEngineeredQuerySchema:
        # Old: Complex coordination between systems
        # variations = self.query_optimizer.generate_query_variations(query, paradigm)
        # if ENABLE_QUERY_LLM:
        #     llm_vars = await self.llm_optimizer.propose_semantic_variations(query, paradigm)
        #     variations.extend(llm_vars)

        # New: Single call to unified planner
        plan = await self.query_planner.plan_queries(query, paradigm)

        return ContextEngineeredQuerySchema(
            original_query=query,
            refined_queries=[v['query'] for v in plan.variations],
            optimization_metadata={
                'complexity_score': plan.complexity_score,
                'strategies_used': plan.metadata.get('strategies_used', []),
                'follow_up_strategy': plan.follow_up_strategy
            }
        )
```

### 2. Research Orchestrator Integration
```python
# research_orchestrator_v2.py (modularized)
class UnifiedResearchOrchestrator:
    async def execute_research(self, classification, context_engineered, user_context):
        # Step 1: Get queries from context engineering
        # Note: Context engineering now uses unified query planner
        source_queries = getattr(context_engineered, "refined_queries", []) or \
                        [context_engineered.original_query]

        # Step 2: Simple compression (orchestrator-level concern)
        optimized_queries = compress_and_dedup_queries(source_queries)

        # Step 3: Execute searches
        search_results = await self.search_manager.execute_with_research_context(
            optimized_queries,
            classification.primary_paradigm,
            user_context,
            research_id
        )

        # Continue with result processing, filtering, etc...
```

## Coordinated Implementation Sequence

### Phase 1: Query Planning Consolidation (Foundation)
1. **Create unified query planner** (`services/query_planning/`)
2. **Consolidate paradigm configurations**
3. **Replace QueryOptimizer in context_engineering.py**
4. **Test query planning functionality**

### Phase 2: Research Orchestrator Modularization
1. **Extract result deduplicator** (`services/result_deduplicator.py`)
2. **Create unified relevance filter** (`services/relevance_filter.py`)
3. **Extend metrics.py with cost tracking**
4. **Create synthesis orchestrator** (`services/synthesis_orchestrator.py`)

### Phase 3: Integration and Enhancement
1. **Enable paradigm search strategies** in unified planner
2. **Add coverage analysis** for follow-up queries
3. **Update research orchestrator** to use new services
4. **Performance testing and optimization**

## Shared Components

### Configuration Management
```python
# Both plans use unified configuration approach
# config/query_planning_settings.py
class QueryPlanningSettings:
    max_variations: int
    enable_llm: bool
    enable_strategic: bool

# config/orchestration_settings.py
class OrchestrationSettings:
    enable_deduplication: bool
    enable_relevance_filter: bool
    enable_synthesis: bool
```

### Common Patterns
```python
# Both use dependency injection pattern
class ServiceContainer:
    query_planner: UnifiedQueryPlanner
    deduplicator: ResultDeduplicator
    relevance_filter: UnifiedRelevanceFilter
    synthesis_orchestrator: SynthesisOrchestrator
```

## Benefits of Combined Approach

### 1. Clear Separation of Concerns
- **Query Planning**: Handles query optimization and variations
- **Orchestration**: Manages research flow and result processing
- **No overlap** between responsibilities

### 2. Reduced Complexity
- **Query planning**: 4 systems → 1 unified system
- **Orchestration**: 3,160 lines → ~500 lines
- **Overall**: Significant code reduction

### 3. Enhanced Capabilities
- **Query Planning**: Enables unused paradigm strategies
- **Orchestration**: Better result processing and synthesis
- **Integration**: Seamless flow between planning and execution

### 4. Improved Maintainability
- **Single source of truth** for query logic
- **Modular services** for orchestration
- **Clear interfaces** between components

## Migration Strategy

### Step 1: Implement Query Planning (Foundation)
```python
# New structure
services/
├── query_planning/
│   ├── config.py        # Paradigm configurations
│   ├── planner.py       # Main planner
│   ├── cleaner.py       # Query cleaning
│   ├── variator.py      # Variation generation
│   └── coverage.py      # Coverage analysis
```

### Step 2: Update Context Engineering
```python
# context_engineering.py
from services.query_planning.planner import UnifiedQueryPlanner

# Replace old QueryOptimizer with UnifiedQueryPlanner
```

### Step 3: Modularize Orchestrator
```python
# Extract services from research_orchestrator.py
services/
├── result_deduplicator.py
├── relevance_filter.py
├── synthesis_orchestrator.py
└── planning_utils.py
```

### Step 4: Create New Orchestrator
```python
# research_orchestrator_v2.py
# Uses new services and unified query planner
```

## Testing Strategy

### 1. Query Planning Tests
- Test each consolidation step
- Ensure behavior parity with old systems
- Test all paradigm strategies

### 2. Orchestration Tests
- Test each extracted service
- Test integration with query planner
- End-to-end research tests

### 3. Integration Tests
- Test complete flow: query → planning → execution → results
- Performance benchmarking
- Load testing

## Risk Mitigation

### 1. Gradual Rollout
- Use feature flags for new components
- Keep old systems during transition
- Easy rollback capability

### 2. Comprehensive Testing
- Test each component independently
- Integration tests for all interactions
- Performance regression testing

### 3. Monitoring
- Track query planning metrics
- Monitor orchestration performance
- Alert on anomalies

## Success Metrics

### Code Quality
- **Total reduction**: 50%+ in targeted modules
- **Duplication elimination**: 100% of identified duplicates
- **Test coverage**: 90%+ for new modules

### Performance
- **Query planning time**: 25% improvement
- **Memory usage**: 30% reduction
- **End-to-end latency**: 15% improvement

### Functionality
- **Parity**: All existing functionality preserved
- **New features**: Paradigm strategies enabled
- **Reliability**: Fewer edge cases and bugs

## Conclusion

The two plans work together synergistically:
1. **Query Planning Consolidation** creates a solid foundation for query optimization
2. **Research Orchestrator Modularization** builds on this foundation with clean orchestration
3. **Together**, they transform the codebase into a maintainable, efficient system

The coordinated approach ensures no duplication, clear separation of concerns, and a smooth migration path to the new architecture.
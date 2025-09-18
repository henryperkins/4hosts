# Query Planning Consolidation Plan

## Problem Statement

The Four Hosts application currently has **four separate query planning/expansion systems** that create significant overlap and inconsistency:

1. **search_apis.py** - QueryOptimizer (rule-based with synonyms)
2. **llm_query_optimizer.py** - LLM-based semantic variations
3. **paradigm_search.py** - Paradigm-specific query generation (currently unused)
4. **agentic_process.py** - Coverage evaluation and follow-up queries (currently unused)

This "four cooks in the kitchen" approach results in:
- Duplicated heuristics across systems
- Inconsistent configuration knobs
- Unclear order of operations
- Unused functionality (paradigm_search and agentic_process query generation)
- Maintenance burden

## Current State Analysis

### Active Systems in Pipeline
1. **QueryOptimizer** (search_apis.py)
   - Cleans queries, removes noise
   - Expands with WordNet synonyms
   - Generates variations (broad, specific, conceptual)
   - Used in context_engineering.py

2. **LLM Query Optimizer** (llm_query_optimizer.py)
   - Optional LLM-powered semantic variations
   - Guarded by `ENABLE_QUERY_LLM`
   - Supplements QueryOptimizer

### Unused But Valuable Systems
3. **Paradigm Search Strategies** (paradigm_search.py)
   - Four sophisticated strategy classes (Dolores, Teddy, Bernard, Maeve)
   - Intelligent query construction with context-aware modifiers
   - **NOT used for query generation** - only for result filtering
   - Has advanced features like adaptive query limits

4. **Agentic Process** (agentic_process.py)
   - Coverage gap analysis
   - Follow-up query generation
   - **NOT used** in main pipeline
   - Designed for iterative research

## Identified Duplications

### 1. Paradigm-Specific Knowledge
```python
# Duplicated across files:
# QueryOptimizer: paradigm_terms dictionary
# ParadigmSearch: query_modifiers per strategy
# AgenticProcess: modifiers dictionary
```

### 2. Query Cleaning Logic
```python
# Duplicated cleaning:
# QueryOptimizer: remove_fluff_words(), protect_entities()
# ParadigmSearch: _clean_query(), _prepare_academic_query() (per paradigm)
```

### 3. Query Complexity Assessment
```python
# Duplicated complexity analysis:
# QueryOptimizer: basic word count
# ParadigmSearch: sophisticated analysis with specificity indicators
```

### 4. Variation Generation
```python
# Duplicated variation patterns:
# QueryOptimizer: rule-based (broad, specific, conceptual)
# LLM Optimizer: semantic variations
# ParadigmSearch: strategic patterns
```

## Consolidation Strategy

### Phase 1: Create Unified Paradigm Configuration

```python
# services/query_planning/config.py
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ParadigmQueryConfig:
    """Centralized configuration for each paradigm's query behavior"""
    name: str
    code: str  # dolores, teddy, bernard, maeve

    # Query modifiers and patterns
    query_modifiers: List[str]
    search_operators: List[str]
    strategic_patterns: List[str]

    # Cleaning rules
    noise_terms: List[str]
    protected_entities: List[str]
    term_replacements: Dict[str, str]

    # Complexity assessment weights
    specificity_indicators: List[str]
    complexity_weights: Dict[str, float]

    # Source preferences
    preferred_sources: List[str]
    source_weights: Dict[str, float]

    # Follow-up patterns
    follow_up_modifiers: List[str]
    gap_analysis_patterns: List[str]

# Load configurations from all existing sources
PARADIGM_CONFIGS: Dict[str, ParadigmQueryConfig] = {
    "dolores": ParadigmQueryConfig(
        name="Dolores (Revolutionary)",
        code="dolores",
        query_modifiers=[
            "controversy", "scandal", "expose", "corrupt",
            "injustice", "systemic", "investigation"
        ],
        # ... other config from existing strategy classes
    ),
    # ... other paradigms
}
```

### Phase 2: Create Unified Query Planner

```python
# services/query_planning/planner.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class QueryPlan:
    original_query: str
    cleaned_query: str
    paradigm: str
    variations: List[Dict[str, Any]]
    complexity_score: float
    follow_up_strategy: Optional[str] = None
    metadata: Dict[str, Any] = None

class UnifiedQueryPlanner:
    """Consolidates all query planning logic into one system"""

    def __init__(self):
        self.configs = PARADIGM_CONFIGS
        self.cleaner = QueryCleaner()
        self.variator = QueryVariator()
        self.complexity_analyzer = ComplexityAnalyzer()

    async def plan_queries(
        self,
        query: str,
        paradigm: str,
        context: Dict[str, Any] = None
    ) -> QueryPlan:
        """Generate a comprehensive query plan"""

        # 1. Clean query using paradigm-specific rules
        cleaned = await self.cleaner.clean(query, paradigm)

        # 2. Assess query complexity
        complexity = await self.complexity_analyzer.assess(cleaned, paradigm)

        # 3. Generate variations using appropriate strategies
        variations = await self.variator.generate_variations(
            cleaned,
            paradigm,
            complexity,
            context
        )

        # 4. Determine follow-up strategy
        follow_up = self._determine_follow_up_strategy(complexity, paradigm)

        return QueryPlan(
            original_query=query,
            cleaned_query=cleaned,
            paradigm=paradigm,
            variations=variations,
            complexity_score=complexity,
            follow_up_strategy=follow_up,
            metadata={
                "complexity_breakdown": complexity,
                "variation_count": len(variations),
                "strategies_used": self._get_used_strategies()
            }
        )
```

### Phase 3: Extract Components

#### Query Cleaner (consolidates cleaning logic)
```python
# services/query_planning/cleaner.py
class QueryCleaner:
    """Consolidates all query cleaning logic"""

    async def clean(self, query: str, paradigm: str) -> str:
        """Clean query using paradigm-specific rules"""
        config = self.configs[paradigm]

        # Apply universal cleaning
        cleaned = self._remove_noise_terms(query, config.noise_terms)
        cleaned = self._protect_entities(cleaned, config.protected_entities)
        cleaned = self._apply_term_replacements(cleaned, config.term_replacements)

        # Apply paradigm-specific cleaning
        cleaned = self._apply_paradigm_cleaning(cleaned, paradigm)

        return cleaned
```

#### Query Variator (consolidates variation generation)
```python
# services/query_planning/variator.py
class QueryVariator:
    """Generates query variations using multiple strategies"""

    async def generate_variations(
        self,
        query: str,
        paradigm: str,
        complexity: float,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate variations using appropriate strategies"""
        variations = []
        config = self.configs[paradigm]

        # 1. Rule-based variations (from QueryOptimizer)
        rule_vars = await self._generate_rule_based_variations(query, paradigm)
        variations.extend(rule_vars)

        # 2. LLM variations (if enabled)
        if self._llm_enabled():
            llm_vars = await self._generate_llm_variations(query, paradigm)
            variations.extend(llm_vars)

        # 3. Paradigm strategic patterns (from ParadigmSearch)
        strategic_vars = await self._generate_strategic_patterns(
            query, paradigm, complexity
        )
        variations.extend(strategic_vars)

        # 4. Adaptive limit based on complexity
        limit = self._calculate_adaptive_limit(complexity)
        return variations[:limit]
```

#### Coverage Analyzer (from agentic_process)
```python
# services/query_planning/coverage.py
class CoverageAnalyzer:
    """Analyzes search coverage and generates follow-up queries"""

    async def analyze_coverage(
        self,
        query: str,
        results: List[Dict[str, Any]],
        paradigm: str
    ) -> CoverageAnalysis:
        """Analyze how well results cover the query"""

        # Extract themes and focus areas
        themes = self._extract_themes(query)
        coverage = self._calculate_theme_coverage(themes, results)

        # Identify gaps
        gaps = self._identify_coverage_gaps(themes, coverage)

        return CoverageAnalysis(
            coverage_score=coverage.score,
            missing_themes=gaps.missing_themes,
            suggested_followups=await self._generate_follow_up_queries(
                query, gaps, paradigm
            )
        )
```

### Phase 4: Integration Points

#### Replace in context_engineering.py
```python
# Before:
from services.search_apis import QueryOptimizer
optimizer = QueryOptimizer()
variations = optimizer.generate_query_variations(query, paradigm)

# After:
from services.query_planning.planner import UnifiedQueryPlanner
planner = UnifiedQueryPlanner()
plan = await planner.plan_queries(query, paradigm)
variations = plan.variations
```

#### Enable in research_orchestrator.py
```python
# Use paradigm strategies for query generation
strategy = get_search_strategy(paradigm)
context = SearchContext(original_query=query, paradigm=paradigm)
paradigm_queries = await strategy.generate_search_queries(context)

# Use coverage analysis for follow-ups
from services.query_planning.coverage import CoverageAnalyzer
analyzer = CoverageAnalyzer()
coverage = await analyzer.analyze_coverage(query, results, paradigm)
```

## Configuration Consolidation

### Remove Redundant Environment Variables
```bash
# Old (multiple):
SEARCH_QUERY_VARIATIONS_LIMIT=5
ADAPTIVE_QUERY_LIMIT=8
ENABLE_QUERY_LLM=1

# New (unified):
UNIFIED_QUERY_MAX_VARIATIONS=8
UNIFIED_QUERY_ENABLE_LLM=1
UNIFIED_QUERY_ENABLE_STRATEGIC_PATTERNS=1
UNIFIED_QUERY_ENABLE_FOLLOW_UP=1
```

### Centralized Configuration
```python
# services/query_planning/settings.py
class QueryPlanningSettings:
    max_variations: int = int(os.getenv("UNIFIED_QUERY_MAX_VARIATIONS", "8"))
    enable_llm: bool = os.getenv("UNIFIED_QUERY_ENABLE_LLM", "1") == "1"
    enable_strategic: bool = os.getenv("UNIFIED_QUERY_ENABLE_STRATEGIC_PATTERNS", "1") == "1"
    enable_follow_up: bool = os.getenv("UNIFIED_QUERY_ENABLE_FOLLOW_UP", "0") == "1"

    # Complexity thresholds
    high_complexity_threshold: float = 0.7
    medium_complexity_threshold: float = 0.4

    # LLM settings
    llm_max_variants: int = 4
    llm_temperature: float = 0.3
```

## Implementation Steps

### Step 1: Create Unified Structure (Week 1)
1. Create `services/query_planning/` directory
2. Implement `ParadigmQueryConfig` with consolidated configs
3. Extract and consolidate cleaning logic
4. Set up basic planner interface

### Step 2: Migrate Active Systems (Week 2)
1. Replace QueryOptimizer usage in context_engineering.py
2. Integrate LLM query optimizer as optional strategy
3. Test with existing functionality

### Step 3: Enable Unused Systems (Week 3)
1. Integrate paradigm search strategy query generation
2. Add coverage analysis for follow-up queries
3. Create integration tests

### Step 4: Cleanup and Optimization (Week 4)
1. Remove old query planning systems
2. Update documentation
3. Performance testing and optimization

## Benefits

### Immediate Benefits
1. **Reduced Complexity**: One system instead of four
2. **Eliminated Duplication**: Single source of truth for query logic
3. **Enabled Unused Features**: Paradigm strategies and coverage analysis now used
4. **Simplified Configuration**: Single set of environment variables

### Long-term Benefits
1. **Better Maintainability**: Easier to modify and extend
2. **Improved Performance**: No redundant processing
3. **Enhanced Testing**: Single system to test
4. **Future-Proof**: Foundation for advanced features

### Risk Mitigation
1. **Gradual Migration**: Keep old systems during transition
2. **Feature Flags**: Control new features with environment variables
3. **Comprehensive Testing**: Ensure behavior parity
4. **Rollback Plan**: Quick revert if issues arise

## Success Metrics

### Code Quality
- Reduce query planning code by 40%
- Eliminate all duplicated heuristics
- Achieve 90%+ test coverage

### Functional
- Maintain all existing query variations
- Enable paradigm-specific strategies (new capability)
- Add coverage-based follow-up queries (new capability)

### Performance
- Reduce query planning time by 25%
- Lower memory usage during query generation
- Improve cache hit rates

## Conclusion

This consolidation will transform the current "four cooks in the kitchen" problem into a single, efficient query planning system. By unifying the overlapping functionality and enabling the unused paradigm strategies, we'll achieve better maintainability, enhanced functionality, and improved performance.
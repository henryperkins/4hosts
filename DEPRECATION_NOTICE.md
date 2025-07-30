# Deprecation Notice - Four Hosts Enhanced Features

## Overview

With the introduction of enhanced features (self-healing, ML pipeline, and enhanced generators), several components are now deprecated. This document outlines the deprecation strategy and migration path.

## Deprecated Components

### 1. Basic Answer Generators (⚠️ DEPRECATED)

**Files Affected:**
- `services/answer_generator_continued.py`
  - `BernardAnswerGenerator` class (lines 27-357)
  - `MaeveAnswerGenerator` class (lines 359-687)

**Replacement:**
- `services/answer_generator_enhanced.py`
  - `EnhancedBernardAnswerGenerator`
  - `EnhancedMaeveAnswerGenerator`

**Migration Path:**
```python
# OLD (Deprecated)
from services.answer_generator_continued import BernardAnswerGenerator, MaeveAnswerGenerator

# NEW (Enhanced)
from services.answer_generator_enhanced import (
    EnhancedBernardAnswerGenerator as BernardAnswerGenerator,
    EnhancedMaeveAnswerGenerator as MaeveAnswerGenerator
)
```

**Key Differences:**
- Enhanced generators include statistical analysis, meta-analysis, SWOT generation
- Better citation handling with study type identification
- Improved confidence calculations based on evidence quality

### 2. Direct Usage of Base Components (⚠️ AVOID)

**Components:**
- Direct instantiation of `AnswerGenerationOrchestrator`
- Direct instantiation of `ClassificationEngine`

**Replacement:**
Use the enhanced versions through `enhanced_integration.py`:
```python
# OLD (Avoid)
from services.answer_generator_continued import answer_orchestrator
from services.classification_engine import classification_engine

# NEW (Preferred)
from services.enhanced_integration import (
    enhanced_answer_orchestrator as answer_orchestrator,
    enhanced_classification_engine as classification_engine
)
```

## Deprecation Timeline

### Phase 1: Current (Immediate)
- Enhanced components are primary
- Basic components maintained for backward compatibility
- All new features use enhanced components

### Phase 2: Next Minor Version (3.1.0)
- Add deprecation warnings to basic generators
- Update all internal usage to enhanced components
- Document migration in changelog

### Phase 3: Next Major Version (4.0.0)
- Remove deprecated basic generators
- Make enhanced components the default
- Clean up redundant code

## Migration Guide

### For API Consumers

No changes required - the API endpoints automatically use enhanced components.

### For Developers

1. **Update Imports:**
   ```python
   # Replace answer_generator_continued imports
   from services.enhanced_integration import enhanced_answer_orchestrator
   ```

2. **Use Enhanced Features:**
   - Submit user feedback via `/research/feedback/{research_id}`
   - Monitor paradigm performance via admin endpoints
   - Let self-healing optimize paradigm selection

3. **New Capabilities:**
   - Statistical insight extraction (Bernard)
   - Competitive intelligence (Maeve)
   - Automatic paradigm switching
   - ML-enhanced classification

## Backward Compatibility

### Maintained Features:
- All existing API endpoints work unchanged
- Response formats remain compatible
- Database schema is extended, not modified

### Breaking Changes:
- None for API consumers
- Internal generator APIs have additional methods

## Code Cleanup Recommendations

1. **Remove Duplicate Logic:**
   - Statistical pattern matching (now in enhanced Bernard)
   - Basic confidence calculations (now evidence-based)

2. **Consolidate Imports:**
   - Use `enhanced_integration.py` as the main entry point
   - Avoid direct imports of deprecated generators

3. **Update Tests:**
   - Test enhanced features (meta-analysis, SWOT)
   - Add tests for self-healing behavior
   - Test ML pipeline integration

## Questions?

Contact the development team or refer to the enhanced features guide in `docs/ENHANCED_FEATURES_GUIDE.md`
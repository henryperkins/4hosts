# Research Workflow Logging & Feedback Recommendations

## Workflow Overview
The Four Hosts research system follows this pipeline:
1. **Query Classification** → Paradigm identification
2. **Context Engineering** → W-S-C-I pipeline
3. **Query Planning** → Strategic query expansion
4. **Search Execution** → Multi-API parallel/priority search
5. **Result Processing** → Deduplication, credibility scoring
6. **Answer Generation** → Paradigm-aligned synthesis

## Current Logging Coverage
✅ **Good coverage:**
- Basic flow transitions
- API call attempts and failures
- Performance metrics (latency, token usage)
- Cost tracking

⚠️ **Gaps identified:**
- Decision point visibility
- Data transformation tracking
- Quality metrics
- User-facing feedback granularity

## Critical Logging Recommendations

### 1. Query Classification Engine
**File:** `services/classification_engine.py`

#### Current State
- Logs classification start/end
- Logs cache hits
- Records final paradigm selection

#### Recommendations
```python
# Add at line 376 (after sanitization, before feature extraction)
logger.info(
    "Query classification input",
    stage="classification_input",
    research_id=research_id,
    query_length=len(query),
    has_special_chars=bool(re.search(r'[^\w\s]', query)),
    language_detected=detect_language(query),  # Add language detection
    query_hash=hashlib.md5(query.encode()).hexdigest()[:8]
)

# Add at line 420 (after feature extraction)
logger.info(
    "Feature extraction complete",
    stage="features_extracted",
    research_id=research_id,
    features={
        "question_words": features.get("question_words", []),
        "action_verbs": features.get("action_verbs", []),
        "domain_indicators": features.get("domain_indicators", []),
        "sentiment": features.get("sentiment_score", 0)
    }
)

# Add at line 480 (when LLM classification differs from rule-based)
if llm_paradigm != rule_paradigm:
    logger.warning(
        "Paradigm classification disagreement",
        stage="classification_conflict",
        research_id=research_id,
        rule_based=rule_paradigm,
        llm_based=llm_paradigm,
        final_choice=final_paradigm,
        confidence_delta=abs(rule_confidence - llm_confidence)
    )
```

### 2. Context Engineering Pipeline
**File:** `services/context_engineering.py`

#### Current State
- Logs layer transitions
- Basic progress updates

#### Recommendations
```python
# Add at line 1050 (after W layer)
logger.info(
    "Write layer output",
    stage="context_write_complete",
    research_id=research_id,
    themes_count=len(write_output.key_themes),
    search_priorities=write_output.search_priorities[:3],
    narrative_frame=write_output.narrative_frame[:100]
)

# Add at line 1080 (after S layer)
logger.info(
    "Select layer output",
    stage="context_select_complete",
    research_id=research_id,
    queries_generated=len(select_output.search_queries),
    source_types=select_output.normalized_source_types,
    authority_domains=len(select_output.authority_whitelist),
    max_sources=select_output.max_sources
)

# Add at line 1110 (after C layer)
logger.info(
    "Compress layer output",
    stage="context_compress_complete",
    research_id=research_id,
    compression_ratio=compress_output.compression_ratio,
    token_budget=compress_output.token_budget,
    removed_elements_count=len(compress_output.removed_elements),
    strategy=compress_output.compression_strategy
)
```

### 3. Search Execution & API Integration
**File:** `services/search_apis.py`

#### Current State
- Logs API calls and responses
- Tracks rate limiting

#### Recommendations
```python
# Add at line 1770 (before primary API search)
logger.info(
    "Search strategy selection",
    stage="search_strategy",
    research_id=research_id,
    primary_api=self.primary_api,
    available_apis=list(self.apis.keys()),
    fallback_order=self._get_fallback_order(),
    min_results_threshold=min_results,
    plan_size=len(planned)
)

# Add at line 1820 (after each API search)
logger.info(
    "API search complete",
    stage="api_search_result",
    research_id=research_id,
    api_name=api_name,
    results_count=len(results),
    unique_domains=len(set(extract_domain(r.url) for r in results)),
    avg_credibility=sum(r.credibility_score for r in results) / len(results) if results else 0,
    response_time_ms=response_time,
    used_cache=was_cached
)

# Add at line 1950 (when falling back)
logger.warning(
    "Search fallback triggered",
    stage="search_fallback",
    research_id=research_id,
    primary_api=self.primary_api,
    primary_results=len(primary_results),
    fallback_api=fallback_api,
    reason="insufficient_results" if len(primary_results) < min_results else "api_error"
)
```

### 4. Query Planning & Optimization
**File:** `search/query_planner/planner.py`

#### Recommendations
```python
# Add after query expansion
logger.info(
    "Query planning complete",
    stage="query_planning",
    research_id=research_id,
    original_query=seed_query[:100],
    candidates_generated=len(candidates),
    strategies_used=list(set(c.stage for c in candidates)),
    paradigm_alignment_score=alignment_score
)

# Add when queries are filtered/ranked
logger.debug(
    "Query candidate ranking",
    stage="query_ranking",
    research_id=research_id,
    candidates=[
        {"query": c.query[:50], "score": c.score, "label": c.label}
        for c in ranked_candidates[:5]
    ]
)
```

### 5. Result Processing & Deduplication
**File:** `services/result_adapter.py` / `services/query_planning/result_deduplicator.py`

#### Recommendations
```python
# Add after deduplication
logger.info(
    "Result deduplication complete",
    stage="deduplication",
    research_id=research_id,
    input_count=original_count,
    output_count=final_count,
    duplicates_removed=original_count - final_count,
    dedup_methods=["url_norm", "title_similarity", "content_hash"],
    unique_domains=len(unique_domains)
)

# Add after credibility scoring
logger.info(
    "Credibility analysis complete",
    stage="credibility_scoring",
    research_id=research_id,
    high_credibility_count=sum(1 for r in results if r.credibility_score > 0.7),
    low_credibility_count=sum(1 for r in results if r.credibility_score < 0.3),
    avg_score=avg_credibility,
    score_distribution=credibility_histogram
)
```

### 6. Answer Generation & Synthesis
**File:** `services/answer_generator.py`

#### Current State
- Logs generation start/end
- Basic section generation

#### Recommendations
```python
# Add at line 1095 (before LLM call)
logger.info(
    "Synthesis context prepared",
    stage="synthesis_context",
    research_id=research_id,
    paradigm=paradigm,
    evidence_quotes=len(context.evidence_quotes),
    unique_sources=len(unique_sources),
    token_budget=context.max_length,
    synthesis_strategy=strategy
)

# Add at line 1120 (for each section)
logger.info(
    "Section generation",
    stage="section_synthesis",
    research_id=research_id,
    section_title=section_def["title"],
    section_index=idx,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    citations_used=len(section.citations),
    confidence=section.confidence
)

# Add at line 1180 (quality check)
logger.info(
    "Answer quality metrics",
    stage="quality_assessment",
    research_id=research_id,
    total_word_count=word_count,
    sections_generated=len(sections),
    unique_citations=len(unique_citations),
    paradigm_alignment_score=alignment_score,
    coherence_score=coherence_score,
    factual_density=facts_per_paragraph
)
```

## User-Facing Feedback Improvements

### 1. Progress Granularity
**File:** `services/progress.py`

```python
# Enhanced progress updates with substeps
async def update_progress_detailed(
    self,
    research_id: str,
    phase: str,
    step: int,
    total_steps: int,
    message: str,
    details: Dict[str, Any] = None
):
    """Provide detailed progress with step counting"""
    await self.update_progress(
        research_id=research_id,
        phase=phase,
        message=f"[{step}/{total_steps}] {message}",
        custom_data={
            "step": step,
            "total_steps": total_steps,
            "percentage": (step / total_steps) * 100,
            **(details or {})
        }
    )
```

### 2. Error Context Enhancement
**File:** `utils/error_handling.py`

```python
# Add error classification and user-friendly messages
ERROR_CLASSIFICATIONS = {
    "rate_limit": {
        "user_message": "Search provider temporarily unavailable. Using alternative sources...",
        "action": "fallback",
        "severity": "warning"
    },
    "no_results": {
        "user_message": "No relevant results found. Trying broader search terms...",
        "action": "expand_query",
        "severity": "info"
    },
    "llm_timeout": {
        "user_message": "AI processing is taking longer than expected. Please wait...",
        "action": "retry",
        "severity": "warning"
    }
}

def classify_and_log_error(
    error: Exception,
    research_id: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Classify errors and provide user-friendly feedback"""
    classification = identify_error_type(error)
    error_info = ERROR_CLASSIFICATIONS.get(
        classification,
        {"user_message": "An unexpected error occurred", "severity": "error"}
    )

    logger.error(
        "Classified error occurred",
        research_id=research_id,
        error_type=classification,
        error_message=str(error),
        severity=error_info["severity"],
        context=context,
        stack_trace=traceback.format_exc() if error_info["severity"] == "error" else None
    )

    return error_info
```

### 3. Quality Indicators
**File:** `frontend/src/components/ResultsDisplayEnhanced.tsx`

```typescript
// Add quality indicators to the UI
interface QualityMetrics {
  paradigmAlignment: number;  // 0-1 score
  sourceCredibility: number;  // 0-1 average
  evidenceCoverage: number;   // 0-1 score
  confidence: number;         // 0-1 score
}

// Display visual indicators
<div className="quality-indicators">
  <QualityBadge
    label="Paradigm Alignment"
    score={metrics.paradigmAlignment}
    tooltip="How well the results align with the selected research paradigm"
  />
  <QualityBadge
    label="Source Quality"
    score={metrics.sourceCredibility}
    tooltip="Average credibility of sources used"
  />
  <QualityBadge
    label="Evidence Coverage"
    score={metrics.evidenceCoverage}
    tooltip="Comprehensiveness of evidence gathered"
  />
</div>
```

## Implementation Priority

### High Priority (Implement First)
1. **Search fallback logging** - Critical for debugging API issues
2. **Query planning visibility** - Understand query expansion decisions
3. **Error classification** - Better user experience during failures
4. **Deduplication metrics** - Track effectiveness of result processing

### Medium Priority
1. **Feature extraction logging** - Understand classification decisions
2. **Section generation metrics** - Track synthesis quality
3. **Progress granularity** - Better user feedback
4. **Credibility distribution** - Source quality insights

### Low Priority
1. **Language detection** - Nice to have for multi-language support
2. **Coherence scoring** - Advanced quality metrics
3. **Token budget tracking** - Cost optimization insights

## Monitoring & Alerting

### Key Metrics to Track
```python
# Add to monitoring dashboard
CRITICAL_METRICS = {
    "classification_accuracy": {
        "threshold": 0.85,
        "alert": "Paradigm classification accuracy below threshold"
    },
    "api_success_rate": {
        "threshold": 0.95,
        "alert": "API success rate degraded"
    },
    "avg_response_time": {
        "threshold": 15.0,  # seconds
        "alert": "Research latency exceeding SLA"
    },
    "deduplication_rate": {
        "threshold": 0.3,  # expect 30% duplicates
        "alert": "Unusual deduplication rate"
    }
}
```

### Log Aggregation Queries
```sql
-- Failed searches by API
SELECT
    api_name,
    COUNT(*) as failure_count,
    AVG(response_time_ms) as avg_response_time
FROM logs
WHERE stage = 'api_search_result'
    AND results_count = 0
GROUP BY api_name
ORDER BY failure_count DESC;

-- Paradigm classification conflicts
SELECT
    DATE(timestamp) as date,
    COUNT(*) as conflicts,
    AVG(confidence_delta) as avg_confidence_gap
FROM logs
WHERE stage = 'classification_conflict'
GROUP BY DATE(timestamp);

-- Query expansion effectiveness
SELECT
    paradigm,
    AVG(candidates_generated) as avg_candidates,
    AVG(CASE WHEN results_count > 10 THEN 1 ELSE 0 END) as success_rate
FROM logs
WHERE stage = 'query_planning'
GROUP BY paradigm;
```

## Testing & Validation

### Unit Tests for Logging
```python
# test_logging_coverage.py
async def test_classification_logging():
    """Ensure all classification decision points are logged"""
    with capture_logs() as logs:
        await classifier.classify("What causes climate change?")

    assert any(l["stage"] == "classification_input" for l in logs)
    assert any(l["stage"] == "features_extracted" for l in logs)
    assert any(l["stage"] == "classification_complete" for l in logs)

async def test_error_classification():
    """Ensure errors are properly classified and logged"""
    with capture_logs() as logs:
        with pytest.raises(RateLimitedError):
            await search_manager.search("test query")

    error_logs = [l for l in logs if l.get("error_type") == "rate_limit"]
    assert error_logs[0]["severity"] == "warning"
    assert "user_message" in error_logs[0]
```

## Rollout Plan

1. **Phase 1 (Week 1)**: Implement high-priority logging in dev environment
2. **Phase 2 (Week 2)**: Add monitoring dashboards and alerts
3. **Phase 3 (Week 3)**: Deploy to staging, gather metrics
4. **Phase 4 (Week 4)**: Production rollout with feature flags

## Success Metrics

- **Reduced debugging time**: Target 50% reduction in time to identify issues
- **Improved user satisfaction**: Clearer error messages and progress feedback
- **Faster incident response**: Alert on degradation before users report
- **Better feature insights**: Data-driven paradigm and query optimization
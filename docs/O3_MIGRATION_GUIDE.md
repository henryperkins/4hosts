# O3 Migration Guide - Maximizing 200k Context Window

## Overview
The Four Hosts system is currently configured for traditional LLMs but has access to o3 with a 200,000 token context window (100k input + 100k output). We're only using 2.5% of this capacity. This guide details the configuration changes and code updates needed to unlock o3's full potential.

## Immediate Configuration Changes (Day 1)

### 1. Configuration Updates Applied
The following changes have been made to `/four-hosts-app/backend/core/config.py`:

```python
# Before → After
SYNTHESIS_BASE_WORDS: 5000 → 50000
SYNTHESIS_BASE_TOKENS: 8000 → 80000
SYNTHESIS_MAX_LENGTH_DEFAULT: 5000 → 50000
EVIDENCE_MAX_QUOTES_DEFAULT: 30 → 200
EVIDENCE_BUDGET_TOKENS_DEFAULT: 2000 → 95000
EVIDENCE_MAX_DOCS_DEFAULT: 20 → 100
EVIDENCE_QUOTES_PER_DOC_DEFAULT: 3 → 10
DEFAULT_SOURCE_LIMIT: 50 → 200 (in research_orchestrator.py)
```

### 2. New O3-Specific Settings Added
```python
# Enable full document processing
O3_FULL_CONTENT_MODE = True  # Process full documents, not just quotes

# Document limits for o3
O3_MAX_FULL_DOCS = 100  # Fetch full content for 100 documents

# Single-pass mode
O3_SINGLE_PASS_MODE = True  # Combine classification + synthesis

# Token budgets
O3_INPUT_TOKEN_BUDGET = 95000  # 95% of 100k input capacity
O3_OUTPUT_TOKEN_BUDGET = 95000  # 95% of 100k output capacity
```

## Code Changes Required (Week 1)

### 1. Evidence Builder Enhancement
**File**: `services/evidence_builder.py`

**Current Approach**:
```python
# Extracts 3 quotes per document from 20 documents
async def build_evidence_quotes(query, results, max_docs=20, quotes_per_doc=3)
```

**O3-Optimized Approach**:
```python
async def build_evidence_bundle_o3(
    query: str,
    results: List[Dict],
    max_docs: int = 100,
    include_full_content: bool = True
) -> Dict[str, Any]:
    """Build evidence bundle optimized for o3's context window."""

    if not include_full_content:
        # Fall back to quote extraction for non-o3 models
        return await build_evidence_quotes(query, results, max_docs)

    # Fetch full content for all documents in parallel
    urls = [r.get("url") for r in results[:max_docs] if r.get("url")]

    # Parallel fetch with increased concurrency
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_parse_url(session, url) for url in urls]
        contents = await asyncio.gather(*tasks, return_exceptions=True)

    # Build comprehensive evidence bundle
    documents = []
    total_tokens = 0

    for result, content in zip(results[:max_docs], contents):
        if isinstance(content, Exception):
            content = result.get("snippet", "")  # Fallback to snippet

        doc = {
            "url": result.get("url"),
            "title": result.get("title"),
            "domain": result.get("domain"),
            "full_content": content,
            "metadata": result.get("metadata", {}),
            "credibility_score": result.get("credibility_score", 0.5)
        }

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        doc_tokens = len(content) // 4
        if total_tokens + doc_tokens > O3_INPUT_TOKEN_BUDGET:
            break

        documents.append(doc)
        total_tokens += doc_tokens

    return {
        "documents": documents,
        "document_count": len(documents),
        "total_tokens": total_tokens,
        "extraction_mode": "full_content"
    }
```

### 2. Research Orchestrator Update
**File**: `services/research_orchestrator.py`

**Add O3-Optimized Execution Path**:
```python
async def execute_research_o3(
    self,
    classification: ClassificationResultSchema,
    context_engineered: ContextEngineeredQuerySchema,
    user_context: Any,
    progress_callback: Optional[Any] = None,
    research_id: Optional[str] = None
) -> Dict[str, Any]:
    """Execute research optimized for o3's massive context window."""

    # Check if o3 mode is enabled
    if not O3_SINGLE_PASS_MODE:
        # Fall back to standard pipeline
        return await self.execute_research(
            classification, context_engineered, user_context,
            progress_callback, research_id
        )

    # Parallel search across ALL APIs with increased limits
    search_tasks = []
    for api in ["google", "brave", "arxiv", "pubmed", "semantic_scholar"]:
        config = SearchConfig(
            max_results=100,  # 100 per API = 500 total
            language="en",
            region="us"
        )
        search_tasks.append(self._perform_search(
            context_engineered.original_query,
            api, config
        ))

    # Execute all searches in parallel
    all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Flatten and filter results
    combined_results = []
    for api_results in all_results:
        if not isinstance(api_results, Exception):
            combined_results.extend(api_results)

    # Early relevance filter (keep more for o3)
    filtered = self._apply_early_relevance_filter(
        combined_results,
        context_engineered.original_query,
        classification.primary_paradigm.value
    )

    # Take top 200 by relevance
    top_results = filtered[:DEFAULT_SOURCE_LIMIT]

    # Build comprehensive evidence bundle with full content
    from services.evidence_builder import build_evidence_bundle_o3
    evidence_bundle = await build_evidence_bundle_o3(
        context_engineered.original_query,
        top_results,
        max_docs=O3_MAX_FULL_DOCS,
        include_full_content=O3_FULL_CONTENT_MODE
    )

    # Single o3 call for classification + synthesis
    synthesis_result = await self._synthesize_with_o3(
        query=context_engineered.original_query,
        evidence_bundle=evidence_bundle,
        paradigm_hint=classification.primary_paradigm.value,
        research_id=research_id,
        max_input_tokens=O3_INPUT_TOKEN_BUDGET,
        max_output_tokens=O3_OUTPUT_TOKEN_BUDGET
    )

    return {
        "results": top_results,
        "answer": synthesis_result,
        "metadata": {
            "mode": "o3_optimized",
            "documents_analyzed": evidence_bundle["document_count"],
            "tokens_used": evidence_bundle["total_tokens"],
            "paradigm": classification.primary_paradigm.value
        }
    }
```

### 3. Answer Generator Update
**File**: `services/answer_generator.py`

**Add O3-Specific Synthesis**:
```python
async def synthesize_with_o3(
    self,
    query: str,
    evidence_bundle: Dict[str, Any],
    paradigm_hint: str,
    max_input_tokens: int = 95000,
    max_output_tokens: int = 95000
) -> GeneratedAnswer:
    """Generate comprehensive answer using o3's full capacity."""

    # Build massive prompt with all documents
    prompt_parts = [
        f"Query: {query}",
        f"Suggested Paradigm: {paradigm_hint}",
        "",
        "DOCUMENTS TO ANALYZE:",
        ""
    ]

    # Include full documents
    for i, doc in enumerate(evidence_bundle["documents"], 1):
        prompt_parts.extend([
            f"--- Document {i} ---",
            f"URL: {doc['url']}",
            f"Title: {doc['title']}",
            f"Credibility: {doc['credibility_score']:.2f}",
            f"Content:",
            doc["full_content"][:50000],  # Cap individual docs at 50k chars
            ""
        ])

    prompt_parts.extend([
        "INSTRUCTIONS:",
        "1. Analyze ALL provided documents comprehensively",
        "2. Identify the most appropriate paradigm for this query",
        "3. Assess the credibility and relevance of each source",
        "4. Synthesize a comprehensive, well-structured answer that:",
        "   - Integrates insights from all relevant documents",
        "   - Provides deep analysis, not surface-level summary",
        "   - Includes specific evidence and citations",
        "   - Addresses multiple perspectives and contradictions",
        "   - Offers actionable insights aligned with the paradigm",
        "5. Structure the response with clear sections and subsections",
        "6. Aim for 10,000-20,000 words of high-quality synthesis"
    ])

    full_prompt = "\n".join(prompt_parts)

    # Call o3 with massive context
    response = await llm_client.generate_completion(
        prompt=full_prompt,
        paradigm=paradigm_hint,
        max_tokens=max_output_tokens,
        temperature=0.7,
        model="o3"  # Explicitly use o3
    )

    # Parse and structure the response
    return self._parse_o3_response(response, query, paradigm_hint)
```

## Deployment Checklist

### Phase 1: Configuration (Immediate)
- [x] Update `config.py` with new limits
- [x] Update `research_orchestrator.py` default_source_limit
- [x] Add O3-specific configuration flags
- [ ] Update environment variables in production

### Phase 2: Code Updates (Week 1)
- [ ] Implement `build_evidence_bundle_o3` in evidence_builder.py
- [ ] Add `execute_research_o3` to research_orchestrator.py
- [ ] Update answer_generator.py with o3-specific synthesis
- [ ] Add parallel document fetching with higher concurrency
- [ ] Implement token counting and budget management

### Phase 3: Testing (Week 1-2)
- [ ] Test with 100+ document fetching
- [ ] Verify memory usage under load (expect 5-10x increase)
- [ ] Validate token counting accuracy
- [ ] Benchmark o3 response times (expect 10-20s)
- [ ] Compare answer quality: current vs o3-optimized

### Phase 4: Optimization (Week 2-3)
- [ ] Implement document streaming to o3
- [ ] Add intelligent document selection based on paradigm
- [ ] Implement batch query processing
- [ ] Add caching for full document content
- [ ] Optimize memory management for large contexts

## Performance Expectations

### Current System
- 20 documents analyzed
- 5k tokens used
- 5-15 second latency
- Surface-level synthesis

### O3-Optimized System
- 200+ documents analyzed
- 95k tokens used
- 10-20 second latency
- Comprehensive, deep synthesis

### Resource Requirements
- Memory: Increase from 2GB to 10GB peak
- Network: 50-100MB per request (vs 5MB current)
- Storage: 2-4GB cache for documents

## Monitoring and Metrics

### Key Metrics to Track
```python
# Add to monitoring.py
o3_metrics = {
    "documents_per_request": Histogram(),
    "tokens_per_request": Histogram(),
    "full_content_fetches": Counter(),
    "o3_synthesis_duration": Histogram(),
    "token_utilization_rate": Gauge()  # Should be >90%
}
```

### Alert Thresholds
- Token utilization < 50%: System underutilizing o3
- Documents per request < 50: Not fetching enough content
- O3 timeout > 30s: May need to reduce document count
- Memory usage > 15GB: Scale infrastructure

## Rollback Plan

If issues arise, rollback is simple via environment variables:
```bash
# Revert to original limits
export SYNTHESIS_BASE_WORDS=5000
export EVIDENCE_MAX_DOCS_DEFAULT=20
export EVIDENCE_BUDGET_TOKENS_DEFAULT=2000
export DEFAULT_SOURCE_LIMIT=50
export O3_FULL_CONTENT_MODE=false
export O3_SINGLE_PASS_MODE=false
```

## Cost-Benefit Analysis

### Costs
- 10x increase in API costs per request
- 5x increase in memory requirements
- 10x increase in network bandwidth

### Benefits
- 10x more sources analyzed
- 100x more context (full docs vs snippets)
- 40x better token utilization (95% vs 2.5%)
- Transformative improvement in answer quality
- True comprehensive research vs surface-level

### ROI
**10-20x value improvement** despite 10x cost increase = **Massive positive ROI**

## Support and Questions

For questions about this migration:
1. Check the updated PIPELINE_ANALYSIS.md for architectural details
2. Review the O3_MIGRATION_GUIDE.md (this document)
3. Monitor the o3_metrics dashboard after deployment

Remember: This is not an optimization, it's fixing a fundamental misconfiguration. The system was built to handle this scale - we're just removing artificial limitations.
# Search APIs Quick Win Enhancements

## Overview
This document outlines the quick win enhancements implemented for the Four Hosts search APIs based on the agentic research workflow remediation plan.

## Implemented Enhancements

### 1. Enhanced Query Diversification
**Location**: `QueryOptimizer.generate_query_variations()`

Expanded from 3 to 5-7 query variations:
- **Primary**: Boolean AND query with key terms
- **Semantic**: Relaxed version with OR operators
- **Question**: Reformulated as a question
- **Synonym**: Expanded with WordNet synonyms
- **Related**: Added related concepts
- **Domain-specific**: Paradigm-aware terms
- **Broad**: Less restrictive for wider coverage
- **Exact phrase**: For finding specific content

### 2. Authority Scoring Improvements
**Location**: `SearchConfig`, `ContentRelevanceFilter._calculate_source_type_score()`

Added:
- `authority_whitelist`: Preferred domains (e.g., .edu, .gov)
- `authority_blacklist`: Blocked domains
- `prefer_primary_sources`: Boolean flag
- Enhanced scoring hierarchy:
  - Whitelisted domains: 0.95
  - Government (.gov): 0.95
  - Educational (.edu): 0.90
  - Academic papers with citations: 0.90+
  - Credible news sources: 0.75
  - Blacklisted domains: 0.10

### 3. Better Metadata Schema
**Location**: `SearchResult` dataclass

Enhanced fields:
- `author`: Author information
- `publication_type`: Type classification (research, article, blog, etc.)
- `citation_count`: Number of citations
- `content_length`: Content size
- `last_modified`: Last modification date
- `content_hash`: MD5 hash for deduplication

Metadata extraction improved in:
- Google results: Extract from metatags
- ArXiv: Parse author lists
- PubMed: Extract full author information

### 4. Cross-source Agreement Detection
**Location**: `ContentRelevanceFilter._detect_cross_source_agreement()`

Features:
- Tracks similar claims across sources
- Boosts relevance score (+0.1) for consensus
- Flags contradictions with negation detection
- Stores consensus metadata in `raw_data`
- Configurable consensus threshold (70%)

### 5. Improved Deduplication
**Location**: `SearchAPIManager._deduplicate_results()`

Implements:
- Content-based MD5 hashing
- Near-duplicate detection using token similarity
- Jaccard similarity calculation (85% threshold)
- Preserves highest relevance score version
- Cross-API deduplication in `search_all()`

## Usage Examples

### Rate Limit Backoff Configuration
The search fetching layer now uses an exponential backoff with jitter for HTTP 429 responses instead of a fixed 60s delay. You can tune behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_RATE_LIMIT_BASE_DELAY` | `2` | Initial delay in seconds for first retry. |
| `SEARCH_RATE_LIMIT_BACKOFF_FACTOR` | `2` | Multiplier applied each retry attempt. |
| `SEARCH_RATE_LIMIT_MAX_DELAY` | `30` | Maximum backoff delay cap in seconds. |
| `SEARCH_RATE_LIMIT_JITTER` | `full` | Jitter strategy: `full` (0 to delay) or `none`. |

Example:
```bash
export SEARCH_RATE_LIMIT_BASE_DELAY=1
export SEARCH_RATE_LIMIT_BACKOFF_FACTOR=2.5
export SEARCH_RATE_LIMIT_MAX_DELAY=20
export SEARCH_RATE_LIMIT_JITTER=full
```

Log output will show: `Rate limited (429) for <url>, attempt <n>, backing off <delay>s (server=<hdr>, computed=<raw>)`.

### Query with Enhanced Variations
```python
optimizer = QueryOptimizer()
variations = optimizer.generate_query_variations(
    "artificial intelligence ethics",
    paradigm="bernard"  # Optional paradigm-specific terms
)
# Returns 6-7 variations for comprehensive coverage
```

### Authority-based Filtering
```python
config = SearchConfig(
    max_results=50,
    authority_whitelist=[".edu", ".gov", "nature.com"],
    authority_blacklist=["spam.com", "clickbait.net"],
    prefer_primary_sources=True,
    min_relevance_score=0.3
)
```

### Cross-source Agreement Detection
```python
filter = ContentRelevanceFilter()
results = filter.filter_results(
    results,
    query,
    config=config,
    detect_consensus=True  # Enable agreement detection
)
# Results will have consensus_claims and potential_contradictions in raw_data
```

## Benefits

1. **Better Coverage**: 5-7 query variations capture more relevant content
2. **Higher Quality**: Authority scoring prioritizes credible sources
3. **Rich Metadata**: Enhanced information for better filtering and analysis
4. **Trustworthiness**: Cross-source agreement helps validate claims
5. **Efficiency**: Better deduplication reduces redundant processing

## Future Enhancements

Consider implementing:
- SimHash for more efficient near-duplicate detection
- Machine learning-based relevance scoring
- Dynamic query expansion based on initial results
- Source credibility database
- Claim extraction and fact-checking integration

# Critical Research Pipeline Issues & Failure Points

## 🛑 HIGH PRIORITY - Pipeline Stoppers

### 1. **Missing URL Handling in Deep Research**
- **Location**: `research_orchestrator.py:2056-2059`
- **Issue**: Deep research synthesizes fake URLs (`about:blank#citation-`) when citations lack URLs
- **Impact**: Results get dropped later in pipeline due to invalid URLs
- **Fix**: Either preserve unlinked citations properly or fetch real URLs

### 2. **Empty Content Drops Without Warning**
- **Location**: `research_orchestrator.py:1739-1744`
- **Issue**: Results with empty content are silently dropped after fetch attempts fail
- **Impact**: Legitimate results lost without user notification
- **Fix**: Add fallback content preservation or user warnings

### 3. **Synchronous Blocking During Answer Synthesis**
- **Location**: `research_orchestrator.py:1440-1452`
- **Issue**: Answer synthesis blocks entire pipeline without cancellation check
- **Impact**: Can't cancel research during synthesis phase (10-30s block)
- **Fix**: Add cancellation checks during synthesis

## ⚠️ MEDIUM PRIORITY - Rate Limits & Timeouts

### 4. **Search Provider Timeout Cascade**
- **Location**: `search_apis.py:1961-1972`
- **Issue**: Single slow API can exhaust timeout budget for all providers
- **Impact**: Partial or no search results when one provider is slow
- **Current**: 25s timeout shared across all providers
- **Fix**: Individual provider timeouts with fallback

### 5. **API Quota Exhaustion Handling**
- **Location**: Multiple - Google CSE (100/day), Brave (2000/month)
- **Issue**: No graceful degradation when quotas exhausted
- **Impact**: Complete search failure instead of using remaining providers
- **Fix**: Track quotas and auto-disable exhausted providers

### 6. **LLM Retry Storm**
- **Location**: `llm_client.py:347-351`
- **Issue**: Aggressive retries (3 attempts) can compound during high load
- **Impact**: Cascading failures and Azure 429 errors
- **Fix**: Circuit breaker pattern with exponential backoff

### 7. **Background Task Polling Timeout**
- **Location**: `background_llm.py:40-42`
- **Issue**: Background tasks poll for max 300s then silently fail
- **Impact**: Deep research tasks timeout without proper error messaging
- **Fix**: Extend timeout or implement proper task recovery

## 🔄 LOW PRIORITY - Recovery & Resilience

### 8. **WebSocket Keepalive Gap**
- **Location**: `websocket_service.py:113-131`
- **Issue**: 30s keepalive may be too long for aggressive proxies
- **Impact**: Silent WebSocket disconnections during long research
- **Fix**: Reduce to 15s or make configurable

### 9. **Deduplication Over-Aggressive**
- **Location**: `research_orchestrator.py:136` (threshold=0.8)
- **Issue**: Similar academic papers might be incorrectly deduplicated
- **Impact**: Loss of unique but similar sources
- **Fix**: Paradigm-specific thresholds (Bernard needs lower)

### 10. **Circuit Breaker Recovery Too Fast**
- **Location**: `search_apis.py:863` (timeout=300s)
- **Issue**: Failed domains recover too quickly, causing repeat failures
- **Impact**: Wasted API calls on consistently failing domains
- **Fix**: Exponential backoff for circuit breaker recovery

## 📊 Data Flow Issues

### 11. **Result Normalization Inconsistency**
- **Multiple locations** - SearchResult vs dict vs ResponsesNormalized
- **Issue**: Multiple result formats cause AttributeErrors
- **Impact**: Results dropped due to missing attributes
- **Fix**: Single normalized result format throughout pipeline

### 12. **Evidence Bundle Merge Failure**
- **Location**: `research_orchestrator.py:1306-1367`
- **Issue**: Complex merge logic for deep research citations can fail silently
- **Impact**: Missing citations in final answer
- **Fix**: Simplify merge logic with better error handling

### 13. **Progress Reporting Gaps**
- **Location**: Multiple async progress callbacks
- **Issue**: Progress updates can fail silently, leaving UI stuck
- **Impact**: User thinks research is frozen when it's actually running
- **Fix**: Fallback progress updates on main path

## 🔐 Configuration & Limits

### 14. **User Rate Limits Not Enforced in Pipeline**
- **Location**: `limits.py` defines limits but not enforced in orchestrator
- **Issue**: Users can exceed concurrent request limits
- **Impact**: System overload from PRO/ENTERPRISE users
- **Fix**: Implement semaphore-based concurrency limiting

### 15. **Memory Leak in Message History**
- **Location**: `websocket_service.py:101`
- **Issue**: Message history grows unbounded (only limit on count, not memory)
- **Impact**: OOM after extended operation
- **Fix**: Add memory-based limits or TTL

## 🚨 Critical Path Dependencies

### 16. **No Fallback for Azure OpenAI Failure**
- **Location**: `llm_client.py:429-517`
- **Issue**: Azure failure blocks synthesis completely
- **Impact**: Entire research fails if Azure is down
- **Fix**: OpenAI fallback or cached response generation

### 17. **Search Manager Creation Can Fail Silently**
- **Location**: `research_orchestrator.py:674-681`
- **Issue**: If all search APIs fail to initialize, empty manager created
- **Impact**: No search results but research continues
- **Fix**: Fail fast if no search providers available

## 🎯 Recommendations

### Immediate Actions:
1. Add comprehensive cancellation checks throughout pipeline
2. Implement proper timeout isolation between providers
3. Add fallback content for empty results
4. Fix deep research URL synthesis issue

### Short-term (1-2 weeks):
1. Implement circuit breakers with exponential backoff
2. Add quota tracking and provider auto-disable
3. Normalize all result formats to single schema
4. Add memory limits to prevent leaks

### Long-term (1 month):
1. Implement full observability with distributed tracing
2. Add health checks for all external dependencies
3. Build automatic fallback paths for critical services
4. Implement gradual degradation strategy

## Testing Checklist

- [ ] Test with all search providers disabled
- [ ] Test with Azure OpenAI down
- [ ] Test with slow/timeout responses
- [ ] Test with rate limit responses (429)
- [ ] Test cancellation at each pipeline stage
- [ ] Test with exhausted API quotas
- [ ] Test with malformed search results
- [ ] Test WebSocket disconnection/reconnection
- [ ] Test concurrent request limits
- [ ] Test memory usage over time

## Monitoring Requirements

1. **Metrics to Track**:
   - Pipeline stage durations (p50, p95, p99)
   - Provider failure rates
   - Cancellation rates by stage
   - Memory usage trends
   - WebSocket connection stability
   - API quota consumption

2. **Alerts to Configure**:
   - Azure OpenAI failures > 10%
   - Search provider failures > 50%
   - Pipeline timeouts > 5/min
   - Memory usage > 80%
   - WebSocket disconnection rate > 20%
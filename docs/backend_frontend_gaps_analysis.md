# Backend-Frontend Gap Analysis - Four Hosts Research Application

## Critical Gaps Identified

### 1. Unused Backend Capabilities
**Deep Research Endpoints**
- `/research/deep` - Dedicated deep research not used (frontend uses generic endpoint)
- `/research/deep/{id}/resume` - Resume capability not implemented
- `/research/deep/status` - Status monitoring not available

**System Features**
- `/sources/credibility/{domain}` - Source credibility lookup unused
- `/system/context-metrics` - W-S-C-I metrics not displayed
- `/paradigms/override` - Paradigm override not in UI
- Webhook management system - No frontend interface

### 2. WebSocket Event Mismatches
**Unhandled Backend Events:**
- `SOURCE_ANALYZING` - Analysis progress not shown
- `SEARCH_RETRY` - Retry attempts not displayed
- `RATE_LIMIT_WARNING` - No user warnings
- `CREDIBILITY_CHECK` - Credibility checks not surfaced
- `DEDUPLICATION` - Dedup process not visible

**Event Type Format Issues:**
- Backend: dotted notation (`research.started`)
- Frontend expects: underscored (`research_started`)

### 3. Data Structure Inconsistencies
- **Paradigm fields**: Backend uses `HostParadigm` enum, frontend expects strings
- **Cost fields**: Backend sends `total`, frontend expects `total_cost`
- **Progress data**: Backend sends complex `custom_data`, frontend expects flat structure
- **Research ID**: Not consistently included in WebSocket messages

### 4. Missing Frontend Implementations
- No deep research UI despite backend support
- Basic error handling vs detailed backend error codes
- No research cancellation despite backend support
- Minimal feedback system despite backend storage
- No source credibility display
- No system health monitoring

### 5. Performance Issues
- Frontend polls every 2s ignoring backend cache TTL
- No leverage of backend Redis cache hints
- Static data (paradigms) not cached on frontend
- Heartbeat messages unused for connection health

## High Priority Fixes

### 1. Deep Research Integration
```typescript
// Add to frontend/src/services/api.ts
async submitDeepResearch(query: string, options: DeepResearchOptions) {
  return this.fetchWithAuth('/research/deep', {
    method: 'POST',
    body: JSON.stringify({ query, ...options })
  })
}
```

### 2. WebSocket Event Alignment
```python
# Backend: Standardize event names in main.py
def _transform_for_frontend(event_type: str) -> str:
    return event_type.replace('.', '_')
```

### 3. Data Field Consistency
```python
# Backend: Align response models
class ResearchCostResponse(BaseModel):
    total_cost: float  # Changed from 'total'
    breakdown: Dict[str, float]
```

## Medium Priority Enhancements

### 1. Source Credibility Component
```typescript
// New component for credibility display
const SourceCredibility = ({ domain, paradigm }) => {
  const { data } = useQuery(['credibility', domain], 
    () => api.getSourceCredibility(domain, paradigm)
  )
  return <CredibilityBadge score={data.score} bias={data.bias} />
}
```

### 2. Progressive Progress Display
- Show phase transitions (classification → context → search → synthesis)
- Display ETA from backend calculations
- Show items_done/items_total progress

### 3. System Health Dashboard
- Leverage `/system/stats` endpoint
- Show API health status
- Display context pipeline metrics

## Implementation Roadmap

**Week 1-2: Core Alignment**
- Fix deep research endpoints
- Standardize data structures
- Add WebSocket handlers
- Improve error handling

**Week 3-4: Feature Enhancement**
- Add credibility display
- Implement progress tracking
- Build health monitoring
- Add cancellation UI

**Week 5-6: Advanced Features**
- Research management UI
- Webhook configuration
- Advanced error recovery
- Cache optimization

## Impact Assessment

**User Experience Impact:**
- Silent failures from WebSocket mismatches
- Missing premium features (deep research)
- Poor error feedback
- No visibility into research quality

**Performance Impact:**
- Inefficient polling patterns
- Unused caching capabilities
- Unnecessary API calls

**Business Impact:**
- Premium features unavailable to users
- Reduced engagement from missing progress details
- Higher infrastructure costs from inefficient polling

## Recommended Actions

1. **Immediate**: Fix WebSocket event naming and data structures
2. **This Sprint**: Implement deep research UI and error handling
3. **Next Sprint**: Add credibility and progress features
4. **Future**: Advanced management interfaces

## Conclusion

The backend provides comprehensive capabilities that aren't fully utilized by the frontend. Addressing these gaps will unlock existing functionality, improve user experience, and reduce technical debt. The modular architecture makes these improvements straightforward to implement incrementally.
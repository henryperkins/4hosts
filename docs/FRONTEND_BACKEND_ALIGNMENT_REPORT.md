# Frontend-Backend Alignment Report

## Executive Summary
After reviewing the backend services (research_orchestrator.py, answer_generator.py, search_apis.py) and frontend implementation, several alignment issues have been identified that need addressing to ensure proper data flow and functionality.

## Key Alignment Issues Found

### 1. Paradigm Value Mismatch
**Issue**: Backend uses HostParadigm enum with different values than frontend
- **Backend**: `HostParadigm.DOLORES` = "revolutionary", `HostParadigm.TEDDY` = "devotion", etc.
- **Frontend**: Expects lowercase values: 'dolores', 'teddy', 'bernard', 'maeve'
- **Location**: 
  - Backend: `/backend/models/context_models.py:31-37`
  - Frontend: `/frontend/src/types.ts:1`

### 2. WebSocket Message Structure
**Issue**: Frontend WSMessage type doesn't fully align with backend progress updates
- **Backend sends**: Complex progress objects with nested metadata
- **Frontend expects**: Simple `WSMessage` with generic `data: unknown`
- **Impact**: Loss of typed information, potential parsing errors

### 3. Research Result Structure Discrepancies
**Issue**: Frontend ResearchResult interface missing several fields sent by backend
- **Missing fields**:
  - `context_engineering` details in paradigm_analysis
  - `execution_metrics` from ResearchExecutionResult
  - `deep_research_content` field
  - `secondary_results` array

### 4. Answer Generation Response Format
**Issue**: Backend answer_generator returns different structure than frontend expects
- **Backend**: Returns `GeneratedAnswer` with `content_md`, `paradigm_sections`, `key_insights`
- **Frontend**: Expects `summary`, `sections`, `action_items`, `citations`
- **Adapter exists but incomplete**: `/backend/routes/research.py:104-111`

### 5. Search API Results Format
**Issue**: SearchResult format differs between services
- **Backend SearchResult**: Contains `metadata` dict with nested fields
- **Frontend SourceResult**: Expects flat structure with direct fields
- **Transformation needed**: `/backend/routes/research.py:114-131` (partially implemented)

### 6. Progress Update Types
**Issue**: Progress tracker sends different event types than frontend handles
- **Backend sends**: 'classification', 'context_engineering', 'search_retrieval', 'synthesis'
- **Frontend handles**: 'status_update', 'progress', 'result', 'error'
- **Missing mapping layer**: WebSocket service needs event type translation

## Recommended Fixes

### Fix 1: Create Paradigm Mapping Utility
```typescript
// frontend/src/utils/paradigm-mapper.ts
export const mapBackendParadigm = (backendValue: string): Paradigm => {
  const mapping: Record<string, Paradigm> = {
    'revolutionary': 'dolores',
    'devotion': 'teddy',
    'analytical': 'bernard',
    'strategic': 'maeve'
  };
  return mapping[backendValue] || 'bernard';
};
```

### Fix 2: Update Frontend Types
```typescript
// frontend/src/types.ts updates
export interface ResearchResult {
  // ... existing fields ...
  execution_metrics?: {
    total_time_seconds: number
    search_time: number
    synthesis_time: number
    tokens_used: number
  }
  deep_research_content?: string
  secondary_results?: SourceResult[]
}

export interface WSMessage {
  type: 'status_update' | 'progress' | 'result' | 'error' | 'classification' | 'context_engineering' | 'search_retrieval' | 'synthesis'
  research_id: string
  data: {
    progress?: number
    message?: string
    metadata?: Record<string, any>
  }
}
```

### Fix 3: Backend Response Normalizer
```python
# backend/services/response_normalizer.py
def normalize_paradigm_value(paradigm: HostParadigm) -> str:
    """Convert HostParadigm enum to frontend-expected string"""
    mapping = {
        HostParadigm.DOLORES: 'dolores',
        HostParadigm.TEDDY: 'teddy',
        HostParadigm.BERNARD: 'bernard',
        HostParadigm.MAEVE: 'maeve'
    }
    return mapping.get(paradigm, 'bernard')
```

### Fix 4: WebSocket Message Adapter
```python
# backend/services/websocket_service.py enhancement
async def send_progress_update(self, research_id: str, event_type: str, progress: int, metadata: dict = None):
    """Send progress update with frontend-compatible format"""
    frontend_type = self._map_event_type(event_type)
    message = {
        "type": frontend_type,
        "research_id": research_id,
        "data": {
            "progress": progress,
            "message": self._get_progress_message(event_type),
            "metadata": metadata or {}
        }
    }
    await self.broadcast(research_id, message)

def _map_event_type(self, backend_type: str) -> str:
    """Map backend event types to frontend types"""
    mapping = {
        'classification': 'progress',
        'context_engineering': 'progress',
        'search_retrieval': 'progress',
        'synthesis': 'progress',
        'complete': 'result',
        'failed': 'error'
    }
    return mapping.get(backend_type, 'status_update')
```

### Fix 5: Answer Response Adapter Enhancement
Update `/backend/routes/research.py:104-111` to properly transform all answer fields:
```python
# Enhanced answer transformation
answer_payload = {
    "summary": getattr(answer_obj, "content_md", "") or getattr(answer_obj, "summary", "") or "",
    "sections": [
        {
            "title": s.get("title", ""),
            "paradigm": normalize_paradigm_value(s.get("paradigm")),
            "content": s.get("content", ""),
            "confidence": s.get("confidence", 0.0),
            "sources_count": s.get("sources_count", 0),
            "citations": s.get("citations", []),
            "key_insights": s.get("key_insights", [])
        }
        for s in (getattr(answer_obj, "paradigm_sections", []) or getattr(answer_obj, "sections", []) or [])
    ],
    "action_items": getattr(answer_obj, "action_items", []) or [],
    "citations": getattr(answer_obj, "citations", []) or []
}
```

## Implementation Priority

1. **High Priority** (Breaking Issues):
   - Fix paradigm value mapping (Fix 1, 3)
   - Update WebSocket message structure (Fix 4)
   - Complete answer response adapter (Fix 5)

2. **Medium Priority** (Data Loss):
   - Update frontend types for missing fields (Fix 2)
   - Enhance source result transformation

3. **Low Priority** (Enhancements):
   - Add execution metrics display
   - Implement secondary results rendering

## Testing Checklist

- [ ] Paradigm values correctly displayed in frontend
- [ ] WebSocket progress updates received and parsed
- [ ] Answer sections properly formatted
- [ ] Citations linked correctly
- [ ] Source results display all metadata
- [ ] Deep research content rendered when available
- [ ] Error messages properly propagated

## Next Steps

1. Implement paradigm mapping utilities in both frontend and backend
2. Update WebSocket service to use proper message format
3. Enhance answer response transformation in research route
4. Update frontend types to match backend response structure
5. Add integration tests for data flow validation
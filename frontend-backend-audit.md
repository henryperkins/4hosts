# Frontend-Backend Integration Audit Report

## Executive Summary

This audit reveals critical integration failures between the Four Hosts frontend (React/TypeScript) and backend (FastAPI/Python) that cause data loss, memory leaks, and race conditions. The most severe issue is the lack of runtime validation for API responses and WebSocket messages, leading to silent failures when data contracts change.

### Update – 2025-09-18
- Fixed a backend failure in the dynamic Action Items generator that caused 400 errors from the Responses API (strict JSON schema). Root cause: schema for `items[].{action,description,priority,timeline,source_ids}` used `strict: true` but did not list all properties under `required`. Remediation: updated schema to require all properties and removed `response_format` from the OpenAI Chat Completions fallback for o‑series models to prevent secondary 400s.
- Code references: `four-hosts-app/backend/services/action_items.py:25`, `four-hosts-app/backend/services/llm_client.py:430`.
- Follow‑ups: add a unit test asserting schema invariants for strict mode and an end‑to‑end smoke test exercising the dynamic actions path.

## 🚨 Critical Findings

### 1. Type Mismatch & Data Contract Violations

#### ContextMetricsPanel.tsx (Line 22)
```typescript
const cp = (data as any)?.context_pipeline || {}
```
- Frontend expects `context_pipeline` but backend sends different structure
- No type validation on API response
- **Impact**: Runtime failures when backend changes

#### ResearchProgress.tsx (Line 290)
```typescript
const data = rawData as WebSocketData
```
- Unsafe type assertion without validation
- WebSocket messages transformed by backend (`_transform_for_frontend`) but frontend doesn't validate transformations
- **Critical**: Event type mapping mismatches (e.g., `WSEventType.RESEARCH_CANCELLED` → `research_progress`)

### 2. WebSocket State Synchronization Issues

#### ResearchProgress.tsx (Lines 278-691)
- **Race Condition**: WebSocket connects before polling completes
- Multiple state updates without batching cause excessive re-renders
- No debouncing for rapid WebSocket events
- **Memory Leak**: WebSocket cleanup in useEffect doesn't guarantee disconnection

#### api.ts (Lines 810-876)
- WebSocket URL construction assumes same-origin without validation
- No reconnection logic for dropped connections
- No message queuing when disconnected

### 3. Authentication & Session Management Flaws

#### api.ts (Lines 141-251)
- **Token Refresh Race**: Multiple 401s trigger parallel refresh attempts
- `failedQueue` pattern can cause request duplication
- CSRF token refresh after auth refresh creates timing vulnerability
- **Critical**: Logout doesn't wait for backend confirmation before clearing state

#### ResearchPage.tsx (Lines 182-189)
- Redirects to login on any error containing "No refresh token"
- No distinction between auth errors and network failures

### 4. Data Transformation Inconsistencies

#### api.ts (Lines 59-100)
- `normalizeAnswerSection()` defaults paradigm to 'bernard' silently
- `normalizeCitation()` generates random IDs when backend ID missing
- **Data Loss**: Failed normalization returns partial objects

#### ResearchPage.tsx (Lines 136-168)
- Manual paradigm classification extraction from results
- Distribution calculation duplicates backend logic
- No validation of confidence scores

### 5. Polling & State Management Issues

#### ResearchPage.tsx (Lines 98-179)
- **Inefficient Polling**: 2-second interval for up to 20 minutes
- No exponential backoff
- **State Desync**: `pollIntervalRef` not synchronized with component lifecycle
- Soft timeout message misleading - research might fail silently

#### ResearchProgress.tsx (Lines 264-277)
- 3-minute timeout hardcoded, not configurable
- Timeout clears before first message received
- No retry mechanism for WebSocket connection failures

### 6. Error Handling Gaps

#### Multiple Components
- Generic error messages lose context
- No error boundaries for component failures
- API errors parsed inconsistently (detail vs error vs message)
- Silent failures in try-catch blocks without logging

### 7. Performance & Memory Issues

#### ResearchProgress.tsx
- **Unbounded Growth**: `updates` array limited to 100 but no time-based cleanup
- **Memory Leak**: `sourcePreviews` array recreated on every update
- Stats calculation (Lines 352-386) runs on every WebSocket message
- No memoization for expensive computations

#### api.ts
- WebSocket connections stored in Map but not cleaned on page navigation
- Message callbacks retained even after component unmount

## 🔴 Most Critical Issues to Fix

### 1. WebSocket Memory Leak & Race Conditions
- ResearchProgress doesn't properly cleanup WebSocket on unmount
- Multiple WebSocket connections can exist for same research_id
- Frontend-backend event type mismatches cause missed updates

### 2. Authentication Token Refresh Race
- Multiple components triggering simultaneous refreshes
- CSRF token timing vulnerability
- No circuit breaker for refresh loops

### 3. Type Safety Violations
- 30+ instances of `as any` casts
- No runtime validation of backend responses
- Data normalization functions silently corrupt data

### 4. State Synchronization
- Research status tracked in 3 places (polling, WebSocket, component state)
- No single source of truth
- Race conditions between polling and WebSocket updates

## 📊 Component-Specific Issues

### ContextMetricsPanel.tsx
- **Type Safety**: Using `any` type for API response
- **Missing Error Boundaries**: No error boundary for component failures
- **No Memoization**: Component re-renders on every parent render
- **Duplicate Labels**: Both Layer Count divs have same label

### EvidencePanel.tsx
- **Complex Key Generation**: Overly complex stable key using btoa
- **Performance**: No memoization for credibilityIcon function
- **Accessibility**: Missing ARIA labels for credibility badges
- **Date Handling**: No fallback for invalid dates

### ResearchProgress.tsx (1237 lines - violates single responsibility)
- **File Too Large**: Should be split into multiple components
- **Complex State**: 24+ state variables indicate component doing too much
- **Performance Issues**: No memoization of heavy computations
- **Magic Numbers**: Constants should be configurable

### ResearchFormEnhanced.tsx
- **Dead Code**: Large blocks of commented code (Lines 69-95, 165-170)
- **Type Safety**: Using `as any` cast for search_context_size
- **Validation Logic**: Query length check hardcoded to 10 chars
- **Accessibility**: Missing fieldset/legend for paradigm selection

### ResearchPage.tsx
- **Type Safety**: Using `as any` casts and loose typing
- **Polling Logic**: Manual setInterval instead of proper polling hook
- **Memory Leak Risk**: pollIntervalRef might not clear properly
- **Error Handling**: Generic error messages without context

### index.css
- **Duplicate Selectors**: `.card-interactive` hover defined twice
- **Browser Compatibility**: Using env() without fallbacks
- **Specificity Issues**: !important flags in reduced motion
- **Performance**: Multiple animation keyframes could impact performance

## 🔥 Critical Race Condition Chain

1. User submits research → Creates `pollIntervalRef`
2. WebSocket connects → Receives events
3. Polling fetches results → Updates same state
4. Component unmounts → WebSocket stays connected (memory leak)
5. Next research → Old WebSocket still firing events

## ⚠️ Authentication Time Bomb

The token refresh queue (`failedQueue`) can trigger parallel refreshes, and the CSRF token clears after auth refresh, creating a ~100ms window where requests fail.

## 📋 Recommendations

### 1. Implement Runtime Validation
```typescript
import { z } from 'zod';

const ContextMetricsSchema = z.object({
  context_pipeline: z.object({
    total_processed: z.number(),
    average_processing_time: z.number(),
    layer_metrics: z.record(z.number())
  })
});

// Use in component
const data = ContextMetricsSchema.parse(await api.getContextMetrics());
```

### 2. Use React Query for API State
```typescript
const { data, error } = useQuery({
  queryKey: ['research', researchId],
  queryFn: () => api.getResearchResults(researchId),
  refetchInterval: (data) => data?.status === 'completed' ? false : 2000,
  staleTime: 1000,
});
```

### 3. Create WebSocket Hook with Cleanup
```typescript
const useResearchWebSocket = (researchId: string) => {
  useEffect(() => {
    const ws = api.connectWebSocket(researchId, handler);
    return () => {
      api.disconnectWebSocket(researchId);
      // Ensure cleanup even if api.disconnect fails
      ws?.close();
    };
  }, [researchId]);
};
```

### 4. Implement Error Boundaries
```typescript
<ErrorBoundary fallback={<ResearchErrorFallback />}>
  <ResearchProgress />
</ErrorBoundary>
```

### 5. Fix Authentication Flow
```typescript
class AuthManager {
  private refreshMutex = new Mutex();

  async refreshToken() {
    return this.refreshMutex.runExclusive(async () => {
      // Single refresh at a time
      return await this.performRefresh();
    });
  }
}
```

### 6. Split Large Components
- Extract WebSocket logic to `useWebSocket` hook
- Create separate `ResearchStats`, `EventLog`, `PhaseTracker` components
- Move polling to `usePolling` hook
- Extract constants to configuration file

### 7. Add Monitoring
```typescript
// Track WebSocket health
const wsHealthCheck = setInterval(() => {
  if (ws.readyState !== WebSocket.OPEN) {
    logger.error('WebSocket disconnected', { researchId });
    reconnect();
  }
}, 5000);
```

## 🎯 Priority Actions

1. **Immediate** (Week 1)
   - Fix WebSocket memory leaks
   - Add runtime validation for API responses
   - Implement error boundaries

2. **Short-term** (Week 2-3)
   - Refactor ResearchProgress.tsx into smaller components
   - Implement React Query for API state management
   - Fix authentication token refresh race condition

3. **Medium-term** (Month 1-2)
   - Add comprehensive TypeScript types (eliminate `any`)
   - Implement proper WebSocket reconnection logic
   - Add performance monitoring and optimization

4. **Long-term** (Quarter)
   - Consider GraphQL or tRPC for type-safe API
   - Implement proper event sourcing for research state
   - Add end-to-end testing for critical flows

## Conclusion

The current implementation has fundamental architectural issues that cause data loss, memory leaks, and poor user experience. The most critical issue is the **1237-line ResearchProgress.tsx component** which violates every principle of separation of concerns - it's simultaneously a WebSocket client, polling manager, stats calculator, UI renderer, and event processor.

Without addressing these issues, the application will continue to experience:
- Silent data corruption
- Memory leaks leading to browser crashes
- Race conditions causing missed updates
- Authentication failures under load
- Poor performance with multiple active researches

The recommended approach is to incrementally refactor starting with the most critical issues (WebSocket cleanup and type safety) while planning a larger architectural improvement to properly separate concerns and establish clear data flow patterns.

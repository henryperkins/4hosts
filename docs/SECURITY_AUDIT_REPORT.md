# Four Hosts Application - Security & Quality Audit Report

## Executive Summary

This comprehensive audit identifies critical security vulnerabilities, data duplication issues, and code quality problems across three key files in the Four Hosts frontend application. The most severe issues include XSS vulnerabilities, unbounded data growth leading to memory leaks, and extensive duplication of metrics display causing user confusion and potential data inconsistencies.

---

## 1. ResearchProgress.tsx - Critical Issues

### 1.1 Security Vulnerabilities

#### XSS Vulnerability (Lines 469-475)
```typescript
const cleanMessage = (msg?: string) => {
  if (!msg) return ''
  try {
    return stripHtml(msg)
  } catch {
    return msg  // ⚠️ CRITICAL: Returns unsanitized HTML on error!
  }
}
```
**Risk**: High - Direct XSS attack vector if stripHtml throws an exception

#### Unbounded Data Growth
- **Line 96**: `MAX_UPDATES` set to 100 but `sourcePreviews` array (line 258) has no limit
- **Line 305**: `stats.apisQueried` Set grows without bounds
- **Line 381**: Updates array slice operation doesn't prevent memory leaks for long-running research

### 1.2 Data Display Problems

#### Empty/Invalid Statistics Display
- **Lines 751-790**: Shows "0" for `sourcesFound` even when no research has started
- **Line 773**: `Quality Rate` can show "NaN%" when `sourcesFound` is 0
- Inconsistent empty state handling (some show "-", others show "0")

#### WebSocket Data Integrity Issues (Lines 169-403)
- No validation that WebSocket messages contain expected data types
- `data.results_count` (line 316) added directly without null checks
- `data.score` (line 321) used in calculations without validation
- Missing bounds checking for progress values

### 1.3 Memory Leaks
- `sourcePreviews` array grows unbounded (line 258)
- `stats.apisQueried` Set grows without limit (line 305)
- No cleanup of refs when component unmounts
- Multiple `setInterval` without proper cleanup checks

---

## 2. Duplicate Metrics and Progress Display

### 2.1 Progress Tracking Duplication

**Multiple Progress States:**
- **Line 99**: `progress` state (overall progress percentage)
- **Line 121**: `determinateProgress` state (done/total items)
- **Line 200**: `ceLayerProgress` state (context engineering progress)
- **Lines 669-676**: Search progress (calculated from `stats.searchesCompleted/stats.totalSearches`)
- **Lines 678-686**: Analysis progress (calculated from `stats.sourcesAnalyzed/analysisTotal`)

**Problem**: Same progress shown in multiple places with potentially conflicting values

### 2.2 Sources Metrics Duplication

#### Sources Found:
- **Line 103**: `stats.sourcesFound`
- **Line 255**: Updated via `source_found` event (`data.total_sources`)
- **Line 316**: Incremented via `search.completed` (`data.results_count`)
- **Lines 750-752**: Displayed in statistics grid

#### Sources Analyzed:
- **Line 104**: `stats.sourcesAnalyzed`
- **Line 274**: Updated via `source_analyzed` event
- **Line 203**: Also tracked via `analysisTotal`
- **Lines 760-764**: Displayed as ratio with `analysisTotal`

**Problem**: Sources counted multiple times through different events

### 2.3 Session Summary vs Statistics Grid Duplication

**Lines 617-628 (Session Summary):**
- ✅ searches completed
- 📊 sources analyzed
- 🔍 high-quality sources
- 🗑️ duplicates removed

**Lines 748-790 (Statistics Grid):**
Shows THE EXACT SAME DATA in a different format

**Problem**: Same metrics displayed twice on same screen

### 2.4 Progress Bar Duplication
- **Lines 660-667**: Overall progress bar
- **Lines 668-676**: Search phase progress bar
- **Lines 678-686**: Analysis phase progress bar

**Problem**: When in search/analysis phase, shows 2-3 progress bars with overlapping information

### 2.5 Phase Tracking Duplication
- **Line 123**: `currentPhase` state
- **Line 211**: Phase updated from `research_progress` message
- **Line 247**: Phase updated from `research_phase_change` message
- **Lines 688-694**: Phase badge display
- **Lines 719-745**: Research phases timeline

**Problem**: Phase shown in badge AND timeline simultaneously

### 2.6 API Usage Duplication
- **Lines 110/626**: `stats.apisQueried.size` (count of APIs)
- **Lines 631-643**: Individual API badges
- **Lines 782-784**: APIs Used in statistics grid

**Problem**: API count shown 3 times

### 2.7 State Update Race Conditions

Multiple WebSocket events update same state:
- `research_progress` → updates progress, determinateProgress, phase
- `source_found` → updates sourcesFound, sourcePreviews
- `source_analyzed` → updates sourcesAnalyzed
- `search.completed` → updates sourcesFound, searchesCompleted

**Problem**: Same metrics updated from multiple events without coordination, causing:
- Inconsistent counts
- Race conditions
- Double counting

---

## 3. sanitize.ts - Security Vulnerabilities

### 3.1 Inadequate HTML Sanitization

```typescript
export function stripHtml(input: string): string {
  if (!input) return ''
  try {
    const txt = input.replace(/<[^>]+>/g, ' ')  // ⚠️ Inadequate!
    return txt.replace(/\s+/g, ' ').trim()
  } catch {
    return input  // ⚠️ Returns unsanitized on error!
  }
}
```

**Issues Not Handled:**
- HTML entities (&lt;, &gt;, etc.)
- JavaScript URLs (javascript:alert(1))
- Data URLs (data:text/html,<script>...)
- Malformed tags (<img src=x onerror=alert(1)>)
- CSS injection
- No HTML entity decoding

**Risk**: High - Multiple XSS attack vectors

---

## 4. api.ts - Critical Security & Logic Issues

### 4.1 Authentication Issues

#### Race Condition in Token Refresh (Lines 214-247)
- Multiple requests can trigger parallel refreshes
- `failedQueue` management is error-prone
- No exponential backoff for failed refreshes

#### CSRF Token Management (Lines 152-159)
- Fallback comment suggests incomplete CSRF protection
- Token cleared on every 401, potentially causing cascading failures

### 4.2 Data Validation Issues

#### Unsafe Type Coercion (Lines 50-100)
- `coerceParadigm` silently converts invalid values to 'bernard'
- `normalizeAnswerSection` creates objects with default values masking missing data
- No validation that required fields actually exist

#### WebSocket Connection Leaks (Lines 806-871)
- Old connections not guaranteed to close before new ones open
- No reconnection logic or heartbeat
- No maximum retry limits
- Maps grow without limit: `wsConnections`, `wsCallbacks`

### 4.3 Error Information Leakage (Lines 268-274)

```typescript
try {
  const err = await response.json()
  throw new Error(err.detail || 'Failed to refresh token')  // ⚠️ Exposes backend errors
}
```

### 4.4 Missing Input Validation
- `researchId` parameters not validated for format/length
- URL construction without proper encoding (line 814)
- No sanitization of user inputs before sending to backend

---

## 5. Frontend-Backend Integration Issues

### 5.1 Source Counting Conflicts

**Backend (websocket_service.py):**
- Line 721: `sources_found` incremented on `report_source_found`
- Line 730-732: Sends `total_sources` in `SOURCE_FOUND` event

**Frontend (ResearchProgress.tsx):**
- Line 255: Receives `total_sources` and SETS total
- Line 316: ADDS `results_count` to `sourcesFound`

**Problem**: Backend sends absolute count, frontend adds incremental = double counting!

### 5.2 Progress State Duplication

Backend sends multiple progress values:
- Lines 651-653: `sources_found`, `sources_analyzed` in every `RESEARCH_PROGRESS`
- Line 654: Always includes `status: "in_progress"`
- Line 621: `progress` percentage (0-100)
- Lines 661-663: `items_done`/`items_total` for determinate progress

Frontend maintains parallel state for each, showing multiple conflicting progress indicators.

### 5.3 Synthesis Progress Double-Reporting

Backend sends every synthesis event TWICE:
1. As specific event (SYNTHESIS_STARTED, SYNTHESIS_COMPLETED)
2. As RESEARCH_PROGRESS event with same information

### 5.4 Message History Accumulation

**Backend**: Stores 100 messages per research, sends last 10 on reconnect
**Frontend**: Keeps last 100 updates, never uses reconnection history

**Problem**: Duplicate message storage, potential memory leak

---

## 6. Recommendations

### 6.1 Immediate Security Fixes

1. **Replace `stripHtml` with DOMPurify or similar library**
2. **Always sanitize without fallback to raw input**
3. **Add input validation for all user-provided data**
4. **Implement proper CSRF token rotation**
5. **Add Content Security Policy headers**

### 6.2 Data Display Fixes

1. **Show loading states instead of zeros**
2. **Validate all numeric operations before display**
3. **Add "No data available" states**
4. **Implement proper null/undefined checks**
5. **Remove duplicate metric displays**

### 6.3 Architecture Improvements

1. **Single Source of Truth**: Each metric should have ONE authoritative state
2. **Backend should be authoritative for all counts/progress**
3. **Remove Session Summary OR Statistics Grid (not both)**
4. **Consolidate Progress**: Use single progress indicator per phase
5. **Unified Phase Display**: Show phase in timeline OR badge, not both
6. **Fix Event Handling**: Prevent double-counting from multiple WebSocket events

### 6.4 Memory Management

1. **Implement proper bounds for all growing arrays**
2. **Add cleanup for WebSocket connections**
3. **Implement connection pooling for WebSockets**
4. **Add proper cleanup in useEffect hooks**
5. **Implement garbage collection for old messages**

### 6.5 Code Quality

1. **Extract sub-components from 934-line ResearchProgress.tsx**
2. **Add comprehensive error logging**
3. **Implement circuit breaker for API calls**
4. **Add Redux or Zustand for state management**
5. **Add comprehensive error boundaries**
6. **Add TypeScript strict mode**
7. **Implement proper async error handling**

---

## 7. Risk Assessment

### Critical (Immediate Action Required)
- XSS vulnerabilities in HTML sanitization
- Memory leaks from unbounded data growth
- Authentication race conditions
- Data double-counting causing incorrect metrics

### High (Address Within Sprint)
- Duplicate metric displays confusing users
- Missing input validation
- Error information leakage
- WebSocket connection leaks

### Medium (Plan for Next Quarter)
- Component size and complexity
- Inconsistent error handling
- Missing state management solution
- Lack of comprehensive testing

---

## 8. Estimated Remediation Effort

- **Security Fixes**: 2-3 days
- **Data Display Consolidation**: 3-4 days
- **Architecture Improvements**: 1-2 weeks
- **Full Refactor**: 3-4 weeks

Total estimated effort for critical and high priority items: **2 weeks**

---

*Generated: December 2024*
*Files Audited: ResearchProgress.tsx, sanitize.ts, api.ts, websocket_service.py*
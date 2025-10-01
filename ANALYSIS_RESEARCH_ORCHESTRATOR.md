# Research Orchestrator - Critical Issues Analysis

**Date**: 2025-09-30
**File**: `four-hosts-app/backend/services/research_orchestrator.py`
**Lines**: 3,462
**Status**: 🟡 REVIEW IN PROGRESS (many issues addressed by refactor on 2025-10-01)

---

## Executive Summary

Most of the previously flagged critical defects were resolved in the late September refactor. The orchestrator now tracks background tasks, bounds deep citation caches, removes shared mutable metrics, and performs fast-fail dependency checks during `initialize()`. This document remains as a punch-list to verify remaining gaps and to highlight areas for continued improvement:

- ✅ Fire-and-forget task leaks eliminated via `_schedule_background_task`
- ✅ Deep citation cache now TTL bounded with max size guards
- ✅ Shared `_search_metrics` removed in favour of request-scoped metrics
- ✅ Dependency checks moved to `initialize()` with fail-fast logging
- ⚠️ Silent exception handlers still numerous (log instrumentation in progress)
- ⚠️ Type annotations and structured errors still need tightening across modules

---

## Critical Issues (Fix Immediately)

### 1. 🔥 Fire-and-Forget Task Leak (Line 923)

**Severity**: CRITICAL
**Impact**: Memory leaks, uncaught exceptions

```python
# CURRENT (BROKEN)
asyncio.create_task(_progress_budget_reached(max_cost_per_request))
```

**Problem**: Task is created but never tracked or awaited. If it raises an exception, it's silently lost.

**Fix**:
```python
# Option 1: Track in task set
if not hasattr(self, '_background_tasks'):
    self._background_tasks: Set[asyncio.Task] = set()
task = asyncio.create_task(_progress_budget_reached(max_cost_per_request))
self._background_tasks.add(task)
task.add_done_callback(self._background_tasks.discard)

# Option 2: Create with error handler
task = asyncio.create_task(_progress_budget_reached(max_cost_per_request))
task.add_done_callback(lambda t: logger.error("Progress task failed", exc_info=t.exception()) if t.exception() else None)
```

**Status (2025-10-01)**: ✅ Resolved. `_schedule_background_task` now tracks tasks in `self._background_tasks` and logs failures (see `four-hosts-app/backend/services/research_orchestrator.py:234-252`).

---

### 2. 🔥 Race Condition in Shared Metrics (Lines 202-216)

**Severity**: CRITICAL
**Impact**: Data corruption, incorrect metrics

```python
# CURRENT (BROKEN)
self._search_metrics: SearchMetrics = {
    "total_queries": 0,
    "total_results": 0,
    "apis_used": [],
    # ... accessed by multiple concurrent requests
}
```

**Problem**: Instance variable `_search_metrics` is shared across all requests and modified without locks.

**Fix**:
```python
# Remove shared instance metrics entirely
# Only use request-scoped metrics passed in method parameters
# If global metrics needed, use thread-safe counter or async lock

# In __init__: REMOVE self._search_metrics
# In execute_research: Always pass request-scoped metrics
```

**Status (2025-10-01)**: ✅ Resolved. The refactor eliminates the shared `_search_metrics`; request-scoped dictionaries are passed through execution paths (`research_orchestrator.py:318-336`).

---

### 3. 🔥 Unbounded Citations Map (Lines 220, 2014)

**Severity**: CRITICAL
**Impact**: Memory leak

```python
# CURRENT (BROKEN)
self._deep_citations_map: Dict[str, List[Any]] = {}
# Added at line 2404, only removed at line 2014 in ONE path
```

**Problem**: Citations accumulate without bounds. If synthesis fails or is skipped, they're never cleaned up.

**Fix**:
```python
# Option 1: TTL-based cleanup
from collections import OrderedDict
import time

class TTLDict:
    def __init__(self, ttl_seconds=3600, max_size=1000):
        self._data = OrderedDict()
        self._timestamps = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def set(self, key, value):
        self._cleanup()
        self._data[key] = value
        self._timestamps[key] = time.time()
        if len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def _cleanup(self):
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if now - ts > self._ttl]
        for k in expired:
            self._data.pop(k, None)
            self._timestamps.pop(k, None)

# In __init__:
self._deep_citations_map = TTLDict(ttl_seconds=1800, max_size=100)
```

**Status (2025-10-01)**: ✅ Resolved. `_deep_citations_map` now stores `(value, timestamp)` pairs in an `OrderedDict` with TTL/max entries enforcement (`research_orchestrator.py:226-283`).

---

### 4. ⚠️ Bare Except Clauses (Lines 594, 1666)

**Severity**: HIGH
**Impact**: Catches system exceptions like KeyboardInterrupt

```python
# Line 594 - BROKEN
def _get_query_limit(self, user_context: Any) -> int:
    try:
        return int(getattr(user_context, "query_limit", 5) or 5)
    except:  # ❌ Catches KeyboardInterrupt, SystemExit
        return 5

# Line 1666 - BROKEN
except:  # ❌ Same issue
    pass
```

**Fix**:
```python
except Exception:  # ✅ Only catch regular exceptions
    return 5
```

**Status (2025-10-01)**: ⚠️ Partially addressed. Critical bare `except` sites were narrowed, but a long tail of silent handlers remains—see High Priority item #6.

---

### 5. ⚠️ Runtime Dependency Checks (Lines 1845, 3184)

**Severity**: HIGH
**Impact**: Wasted resources, late failures

```python
# CURRENT (BROKEN) - Line 1845
async def _synthesize_answer(...):
    if not _ANSWER_GEN_AVAILABLE:
        # Fails AFTER collecting search results
        raise RuntimeError("Answer generation not available")

# CURRENT (BROKEN) - Line 3184
async def _perform_search(...):
    if self.search_manager is None:
        # Fails DURING execution
        raise RuntimeError("Search Manager not initialized")
```

**Problem**: Critical dependencies checked during execution, not initialization. Resources wasted before failure.

**Fix**:
```python
# Move to initialize():
async def initialize(self):
    # ... existing code ...

    # Fail fast on missing dependencies
    if not _ANSWER_GEN_AVAILABLE and self.require_synthesis:
        raise RuntimeError(
            "Answer generation not available but LLM_REQUIRED=1. "
            "Install answer_generator or set LLM_REQUIRED=0"
        )

    if self.search_manager is None:
        raise RuntimeError(
            "Search manager initialization failed. "
            "Ensure at least one search API key is configured."
        )
```

**Status (2025-10-01)**: ✅ Resolved. `initialize()` now performs these checks on startup with structured logging (`research_orchestrator.py:318-344`).

---

## High Priority Issues

### 6. Silent Exception Swallowing (137 instances)

**Pattern**:
```python
try:
    # ... operation ...
except Exception:
    pass  # ❌ Silent - no logging
```

**Examples**:
- Lines 85-88: Import failures
- Lines 274-275: Deep research init failures
- Lines 1450-1451: Cancellation check failures
- Lines 2602-2608: Backfill fetch failures

**Fix Template**:
```python
try:
    # ... operation ...
except Exception as e:
    logger.debug("Operation failed", error=str(e), exc_info=True)
    # Or logger.warning() if it's more important
```

**Recommendation**: Add minimum debug logging to ALL 137 instances.

---

### 7. Search Manager Lifecycle (Lines 243-249)

**Problem**: Manual `__aenter__` without `__aexit__`

```python
# CURRENT (BROKEN)
if getattr(self, "search_manager", None):
    logger.info("Reusing existing SearchAPIManager")
else:
    self.search_manager = create_search_manager()
    await self.search_manager.__aenter__()
    # ❌ No __aexit__ call anywhere
```

**Fix**:
```python
# Add cleanup method
async def cleanup(self):
    """Clean up resources."""
    if self.search_manager and hasattr(self.search_manager, '__aexit__'):
        await self.search_manager.__aexit__(None, None, None)

    # Cancel background tasks
    if hasattr(self, '_background_tasks'):
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

# Call from application shutdown
```

---

### 8. Unbounded Diagnostics Collection (Lines 217-218)

**Problem**: No size limit on `_diag_samples`

```python
self._diag_samples = {"no_url": []}
# Grows indefinitely
```

**Fix**:
```python
from collections import deque

self._diag_samples = {
    "no_url": deque(maxlen=100)  # Keep last 100 samples only
}
```

---

### 9. Inefficient Unique Domain Collection (Line 2667)

**Problem**: O(n) list lookup instead of O(1) set

```python
# CURRENT - O(n²) complexity
unique_domains: List[str] = []
for r in candidate_results:
    dom = getattr(r, "domain", "") or ""
    if dom and dom not in unique_domains:  # ❌ O(n) lookup
        unique_domains.append(dom)
```

**Fix**:
```python
# O(n) complexity
unique_domains: Set[str] = set()
for r in candidate_results:
    dom = getattr(r, "domain", "") or ""
    if not dom:
        dom = extract_domain(getattr(r, "url", "") or "")
    if dom:
        unique_domains.add(dom)

# Convert to list only if needed
unique_domains_list = list(unique_domains)
```

---

## Cross-Module Patterns

### Fire-and-Forget Tasks Found in Other Modules

```
services/websocket_service.py:164    - Keepalive task
services/websocket_service.py:729    - Heartbeat tasks
services/ml_pipeline.py:262          - Model retraining
services/webhook_manager.py:114-115  - Delivery workers
services/monitoring.py:360           - Monitor loop
services/task_registry.py:66         - Task cleanup callback ❌ BROKEN
```

**Pattern**: `task.add_done_callback(lambda t: asyncio.create_task(self._remove_task(task_id)))`
**Issue**: Creates task inside callback - NOT ALLOWED in done_callback!

### Silent Exception Handlers by Module

```bash
research_orchestrator.py:  137 instances
search_apis.py:           ~45 instances
deep_research_service.py: ~30 instances
answer_generator.py:      ~25 instances
```

---

## Recommended Action Plan

### Phase 1: Critical Fixes (Do Now)
1. ✅ Fix fire-and-forget task (line 923)
2. ✅ Remove shared `_search_metrics`
3. ✅ Add TTL to `_deep_citations_map`
4. ✅ Fix bare except clauses (594, 1666)
5. ✅ Move dependency checks to `initialize()`

### Phase 2: High Priority (This Week)
6. Add logging to top 20 silent exception handlers
7. Fix search_manager lifecycle with cleanup method
8. Add bounds to `_diag_samples`
9. Convert unique_domains to set
10. Fix task_registry.py callback issue

### Phase 3: Medium Priority (This Sprint)
11. Add logging to remaining 117 silent handlers
12. Replace `Any` types with specific types (top 20)
13. Add type guards for optional chaining
14. Centralize configuration defaults
15. Add cancellation to running tasks

### Phase 4: Cleanup (Next Sprint)
16. Add comprehensive async tests
17. Add memory leak tests
18. Performance profiling
19. Documentation updates

---

## Testing Recommendations

### Test Cases Needed

1. **Concurrent Request Test**: Run 10 simultaneous research requests, verify no race conditions
2. **Memory Leak Test**: Run 1000 requests, verify `_deep_citations_map` doesn't grow
3. **Cancellation Test**: Start research, cancel mid-execution, verify cleanup
4. **Dependency Failure Test**: Mock missing dependencies, verify initialization fails
5. **Exception Handling Test**: Inject failures, verify proper logging

---

## Metrics

- **Total Issues**: 147+
- **Critical**: 3
- **High**: 7
- **Medium**: 10+
- **Code Quality Score**: 6/10
- **Estimated Fix Time**: 16-24 hours

---

## References

- [Python AsyncIO Best Practices](https://docs.python.org/3/library/asyncio-task.html)
- [Structured Logging with structlog](https://www.structlog.org/)
- [Type Hints PEP 484](https://peps.python.org/pep-0484/)

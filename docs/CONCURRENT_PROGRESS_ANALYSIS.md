# Concurrent Progress Updates – Technical Analysis

_Author: Engineering @ Four-Hosts – 2025-09-23_

This note digs into how **ProgressTracker** handles _simultaneous_ updates from
many asynchronous tasks (search APIs, credibility workers, agentic follow-ups,
LLM calls, etc.).  It documents current behaviour, surfaces race conditions,
and proposes low-risk improvements.

## 1. Current Concurrency Model

| Layer | Concurrency Source | Interaction with ProgressTracker |
|-------|-------------------|----------------------------------|
| **AsyncIO event-loop** | Each long-running unit (`search_with_plan`, `get_source_credibility_safe`, LLM calls) awaits network I/O | Multiple `await progress_tracker.update_progress(...)` can interleave naturally; no explicit lock. |
| **Background tasks** | Heart-beat in `_heartbeat()` per research, Redis persist snapshots | Uses its own `asyncio.create_task`; shares mutable dict `self.research_progress` |
| **WebSocket broadcasts** | For each update we call `connection_manager.broadcast_to_research()` which itself iterates over connected sockets | Not awaited by callers → fire-and-forget tasks may overlap |

### Key internal state

```
self.research_progress: Dict[str, Dict[str, Any]]
self._last_persist: Dict[str, float]
self._heartbeat_tasks: Dict[str, asyncio.Task]
```

All are **unsynchronised** and mutated by multiple coroutines.

## 2. Observed Race Conditions

1. **Phase completion vs. start of next phase**  
   Two tasks may update phase almost simultaneously (e.g. search workers mark
   `search` as complete, while analysis coroutine immediately sets phase to
   `analysis`).  Because `completed_phases` is updated _after_ phase change the
   previous phase may never be recorded, skewing the weighted progress.

2. **Snapshot persistence skip**  
   `_persist_interval_sec` (default 2 s) debounce is keyed by
   `self._last_persist[research_id]`.  If two updates land within the same
   event-loop tick they both read the old timestamp, decide “time elapsed > 2
   s is **false**”, skip persist, then both write back the _same_ timestamp.
   Net effect: we can go ~4 s without snapshot.

3. **Heartbeat cancellation leak**  
   `_cleanup()` cancels the task but multiple concurrent `start_research` →
   `cleanup` sequences for the same research_id can raise `KeyError` on
   `_heartbeat_tasks.pop()`.

4. **WebSocket back-pressure**  
   Rapid concurrent broadcasts (search loop + heart-beat) line up many pending
   `send_json` coroutines on the same WebSocket writer, occasionally hitting
   `RuntimeError: cannot call send() while another coroutine is already
   waiting`.  We mask it with `try/except`, but updates are lost.

## 3. Recommendations

### 3.1 Atomic State Updates

Use an `asyncio.Lock` **per research_id** to serialise mutations of the shared
progress dict.  Wrap `update_progress()` body:

```python
lock = self._locks.setdefault(research_id, asyncio.Lock())
async with lock:
    ... mutate state ...
```

This avoids phases being skipped and ensures `completed_phases` integrity.

### 3.2 Debounce Rework

Instead of last-persist timestamps, push every update into an
`asyncio.Queue` per research session.  A dedicated consumer task persists the
_latest_ snapshot every *n* seconds.  This guarantees that the **final** state
is flushed while coalescing bursty updates.

### 3.3 Broadcast Funnel

Similarly funnel WebSocket messages through a queue to ensure ordering and
avoid overlapping `send_json`.  The consumer awaits `send_json` serially;
other coroutines merely `put_nowait`.

### 3.4 Cleanup Idempotency

Guard all `pop()` operations with `dict.pop(key, None)` and check `if task and
not task.done(): task.cancel()` to avoid double-cancel races.

### 3.5 Telemetry

Add counters for dropped broadcasts and skipped persists to quantify impact
before/after fixes.

## 4. Proposed Task Breakdown

1. Introduce `self._update_locks: Dict[str, asyncio.Lock]` in
   `ProgressTracker` (weight: **4 h**).
2. Replace timestamp debounce with queue-based persister ( **6 h** ).
3. Implement broadcast funnel ( **4 h** ).
4. Add metrics (Prometheus counters) ( **2 h** ).
5. Update docs & unit tests ( **2 h** ).

–– **End** ––


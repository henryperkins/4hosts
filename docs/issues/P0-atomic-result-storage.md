# P0: Atomic Result Storage and Event Ordering

Severity: P0 (Data corruption/inconsistency risk)

Owner: Backend (routes, services)

## Problem
- `results` and `status` are written in two separate calls, then WS completion is emitted, creating a race where clients can observe `RESEARCH_COMPLETED` while `/research/results/{id}` still returns an in‑progress state.
- Evidence in `four-hosts-app/backend/routes/research.py:868` followed by `:872` and WS complete right after `:875`.

## Goal
Make storing final results and status a single atomic write and guarantee WebSocket completion is emitted only after persistence succeeds.

## Scope
- `four-hosts-app/backend/services/research_store.py`
- `four-hosts-app/backend/routes/research.py:860-882`
- No schema changes; record shape stays the same.

## Plan
- Add `update_fields(research_id, patch: dict)` to `ResearchStore` that:
  - Reads current record once, merges the patch, writes back with a single `set()`.
  - For Redis: use a small transaction/pipeline (`WATCH`/`MULTI`/`EXEC`) if available; otherwise a single `SET` write with merged object is acceptable (still atomic on value level).
  - Ensure `_updated_at` and monotonic `version` are set per write (increment `version`, initialize to 1 if missing) for stale read detection.
- Replace the two sequential writes in `routes/research.py` with one call:
  - `await research_store.update_fields(research_id, {"results": final_result, "status": ResearchStatus.COMPLETED})`.
- Emit WS completion only after the store call returns without exception.
- Update failure path to set `{status: FAILED, error: ...}` in one call too.

## Acceptance Criteria
- After `RESEARCH_COMPLETED`, `GET /research/results/{id}` always returns `{status: completed}` and includes `results`.
- No intermediate state where `status` is `PROCESSING/IN_PROGRESS` while WS has already sent `RESEARCH_COMPLETED`.
- Research record includes monotonically increasing `version` and a fresh `_updated_at` ISO timestamp on each write.
- Existing consumers continue to work without payload changes.

## Tests
- Unit: `tests/test_research_store_atomic.py`
  - Simulate concurrent reads around `update_fields` and confirm a single write occurs and `version` increments.
- Integration: `tests/test_routes_research_completion.py`
  - Submit a job, monkeypatch `progress_tracker` to capture events, finalize with `final_result` using the new method, assert order: store updated → WS complete → GET results shows completed.

## Observability
- Log a structured info on final persist including `{research_id, version, size_bytes}`.
- Export Prometheus counter `research_finalized_total` (label `status=completed|failed`).

## Rollback Plan
- Revert to previous two‑call writes; remove `version` field usage (non‑breaking).

## Risks
- Redis transaction complexity; fallback is single `SET` which is acceptable for this P0 if transactional primitives not available.


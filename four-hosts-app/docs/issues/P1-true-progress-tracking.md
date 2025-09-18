# P1: Replace Magic-Number Progress With Real Tracking

Severity: P1 (UX degradation and trust gap)

Owner: Backend (routes, orchestrator, WS service)

## Problem
- Progress is bumped to fixed values (5, 8, 15, 20, 98…) regardless of actual work.
- Users see long stalls at 25% followed by sudden jumps, or completion messages while processing continues.
- Evidence: `four-hosts-app/backend/routes/research.py:115, 160, 200, 874-878`; orchestrator has the hooks (`progress_callback`) to report granular progress but weights are not unified end‑to‑end.

## Goal
Report progress based on real units of work completed across phases: classification, context engineering, search (queries × providers), processing/deduplication, synthesis. Provide ETA using historical phase durations.

## Scope
- `four-hosts-app/backend/services/websocket_service.py` (progress model/ETA)
- `four-hosts-app/backend/services/research_orchestrator.py` (report actual units)
- `four-hosts-app/backend/routes/research.py` (remove magic numbers; map phases only)

## Plan
- Define a canonical phase model in WS tracker:
  - Phases: `classification` (10%), `context` (15%), `search` (45%), `processing` (10%), `synthesis` (20%).
  - Within `search`, compute percent by `(queries_done / queries_total)`; let orchestrator pass `items_done/items_total`.
  - Within `processing`, tie to dedup and credibility checks done.
  - Within `synthesis`, tie to sections generated or tokens consumed if available; fallback to step counters.
- Orchestrator:
  - Ensure calls to `report_search_started`, `report_search_completed`, `report_source_found`, and `report_deduplication` provide `items_done/items_total`.
  - Expose `metrics.search_metrics` totals already collected to WS tracker.
- Routes:
  - Replace hardcoded numeric bumps with phase transitions only (e.g., `update_progress(phase="classification")`) and let WS compute percent.
- ETA:
  - Keep per‑phase moving average in WS (`_phase_stats`) and emit `eta_seconds` when phase active.

## Acceptance Criteria
- Progress increases monotonically to 100 with no >10% jumps unless a phase completes.
- `search` phase progress reflects actual queries completed.
- `synthesis` progress reflects sections/tokens produced (or steps), not a fixed 98% near completion.
- After completion, final progress is 100 and stays consistent with `/status`.

## Tests
- Unit: `tests/test_progress_phases.py`
  - Simulate orchestrator callbacks for N queries; assert progress advances smoothly and caps at phase weight.
- Integration: `tests/test_progress_e2e.py`
  - Submit a query with 5 optimized sub‑queries; capture WS stream; assert expected sequence of phase changes and progress increments.

## Observability
- Prometheus histogram `phase_duration_seconds{phase}` fed by WS tracker.
- Counter `research_completed_total` with labels `{depth, status}`.

## Rollback Plan
- Restore magic numbers in `routes/research.py` (not recommended); keep orchestrator metrics intact.

## Risks
- Slightly more WS chatter; ensure rate limits are respected.


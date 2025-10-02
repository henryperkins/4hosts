# Refactoring Status: Granular Progress for Analysis Phase

## 1. Background and Context

**Problem (resolved):** The user interface previously stalled during the `analysis` phase because the credibility checker ran every domain concurrently via `asyncio.gather` and only emitted progress after *all* tasks finished. This matched the gap called out in `docs/PROGRESS_GAPS_ANALYSIS.md` (Gap #2, "Silent credibility checks").

**Resolution:** Backend support for granular analysis progress landed in the credibility helpers and orchestrator wiring:

- `four-hosts-app/backend/services/credibility.py:1333` rewrote `analyze_source_credibility_batch` around `asyncio.as_completed`, incremental progress emission, and cancellation support.
- `four-hosts-app/backend/services/research_orchestrator.py:1358` now passes the `progress_tracker`, `research_id`, and `check_cancelled` callback so those updates reach the WebSocket facade.

## 2. Current Behaviour Snapshot

- Credibility checks are spawned as individual tasks and surfaced as they complete, so long batches steadily advance the `analysis` phase.
- Every iteration awaits `check_cancelled()` to honour user-initiated aborts quickly.
- An empty-source guard marks the phase complete and logs a friendly message, preventing a lingering "0/0" state.
- Aggregation results retain legacy schema keys while exposing additional metrics such as `high_credibility_count` for downstream consumers.
- The batch helper now records cadence telemetry (`duration_ms`, update gaps, cancellation flag) which is bubbled into `processed_results.metadata.analysis_metrics`.

## 3. Telemetry & Dashboards

- Prometheus now exposes the following series for dashboarding: `analysis_phase_duration_seconds`, `analysis_phase_sources_total`, `analysis_phase_updates_total`, `analysis_phase_updates_per_second`, `analysis_phase_avg_gap_seconds`, `analysis_phase_p95_gap_seconds`, `analysis_phase_first_gap_seconds`, `analysis_phase_last_gap_seconds`, and `analysis_phase_cancelled_total` (all labelled by `paradigm`/`depth`).
- Redis telemetry payloads receive the same fields via `telemetry_pipeline.record_search_run`, so historical comparisons (pre/post refactor) can be queried directly from the metrics store.
- Suggested dashboards: overlay `analysis_phase_updates_per_second` with `analysis_phase_duration_seconds` to visualise smoothness, and set alerts on `analysis_phase_first_gap_seconds > 5s` to catch regressions early.

## 4. Remaining Follow-ups

- **Frontend smoothing:** UI-level buffering/smoothing work remains tracked separately in `docs/CONCURRENT_PROGRESS_ANALYSIS.md`.
- **Documentation hygiene:** Ensure older narratives of the `asyncio.gather` behaviour cite this change (see `docs/PROGRESS_GAPS_ANALYSIS.md`).

## 5. Verification Checklist

- [x] Incremental updates emitted from `analyze_source_credibility_batch`.
- [x] Orchestrator forwards `progress_tracker`, `research_id`, and `check_cancelled`.
- [x] Empty source batches complete the phase cleanly.
- [x] Observability dashboards updated to monitor real-time batch throughput.

## 6. Risk & Mitigation Notes

- **Progress spam:** Tight loops can overwhelm the tracker. Mitigated by the existing `update_progress` coalescing logic in `ProgressTracker`.
- **Cancellation churn:** Cancelled tasks log an info message for visibility; follow-up may add structured telemetry if cancellations increase.
- **Tracker outages:** All tracker interactions stay inside `try/except` so outages remain non-fatal.

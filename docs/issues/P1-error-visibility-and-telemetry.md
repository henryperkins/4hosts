# P1: Stop Swallowing Exceptions — Add Error Visibility & Telemetry

Severity: P1 (Hidden failures, misleading UX, hard to debug)

Owner: Backend (routes + services)

## Problem
- Many `except Exception: pass` blocks hide operational issues (WS updates, metadata assembly, overrides), yielding silent degradation.
- Users see incomplete/empty fields with no error indicators; operators lack telemetry.
- Evidence: multiple `try/except` blocks in `four-hosts-app/backend/routes/research.py` around classification overrides, WS updates, and final assembly.

## Goal
Replace silent catches with structured logs and telemetry, and return degraded‑quality indicators to the frontend when parts fail.

## Scope
- `four-hosts-app/backend/routes/research.py`
- `four-hosts-app/backend/services/research_orchestrator.py` (selected helpers)
- `four-hosts-app/backend/services/websocket_service.py` (optional: error events)
- `four-hosts-app/backend/services/monitoring.py` (metrics integration)

## Plan
- Introduce `utils/error_handling.py` helper:
  - `log_exception(context: str, e: Exception, **fields)` using `structlog.get_logger(__name__)`.
  - `record_error_metric(error_type, severity)` via `PrometheusMetrics.errors_total`.
  - Optional context manager `capture_errors(context, return_value=None, severity="warning")` to replace bare `except` blocks.
- Replace high‑impact `except Exception: pass` with:
  - `log_exception("ws.update_progress")` for WS issues; do not abort pipeline.
  - `log_exception("final_payload.build")` and attach `"degraded": true` to `metadata` when sections fail to assemble.
  - For deep research/classification failures, set `metadata["warnings"].append(...)` so FE can surface a banner.
- Emit WS `WSEventType.ERROR` with non‑fatal warnings when appropriate (rate‑limited to avoid spam).

## Acceptance Criteria
- No bare `except Exception: pass` remains in the targeted hot paths.
- Errors increment `errors_total{error_type, severity}` and produce structured logs with `research_id`.
- Final responses include `metadata.warnings` when non‑fatal steps fail; FE can display a “partial results” notice.

## Tests
- Unit: `tests/test_error_visibility.py`
  - Force WS send failure; verify log call and `errors_total` increment.
  - Force citation mapping failure; verify `metadata.degraded==true` and warning present.
- Integration: `tests/test_nonfatal_errors_e2e.py`
  - Simulate provider outage; ensure response returns with warnings and WS emits an ERROR event.

## Observability
- Prometheus `errors_total` counters with labels; example: `{error_type="ws_send", severity="warning"}`.
- Sample debug log line (JSON): `{ "evt": "final_payload.build", "status": "degraded", "research_id": "...", "error": "KeyError: ..." }`.

## Rollback Plan
- Revert helper usage; keep logs added for future debugging.

## Risks
- More logs noise; mitigate by using `LOG_LEVEL` and targeted contexts.


# P0: Paradigm Override — Single Source of Truth

Severity: P0 (Incorrect classification and misleading confidence)

Owner: Backend (routes, classification)

## Problem
- Paradigm can be overridden in multiple places with artificial confidence inflation (e.g., setting to `0.9`).
- This makes classification drift across the pipeline and invalidates confidence metrics.
- Evidence: `four-hosts-app/backend/routes/research.py:150-244` (execute path override & boost), `:1000-1030` (submission), plus reconciliation logic.

## Goal
Enforce a single, auditable override point (submission) with no artificial confidence boosting, and propagate that forward read‑only.

## Scope
- `four-hosts-app/backend/routes/research.py`
- `four-hosts-app/backend/services/classification_engine.py` (only for types/compat)

## Plan
- Submission path:
  - Apply user `options.paradigm_override` exactly once when persisting the initial research record.
  - Persist an `override: { source: "user", at: <iso>, value: <Paradigm> }` note under `paradigm_classification`.
  - Do not raise confidence; keep engine‑produced `confidence` unchanged.
- Execution path:
  - Remove any re‑application/boosting code inside `execute_real_research`.
  - Trust the stored classification primary for UI mapping; do not mutate distribution/confidence.
- Logging:
  - When override is present, log a structured audit entry once at submission with `{research_id, previous_primary, overridden_to, user_id}`.

## Acceptance Criteria
- Only the submission endpoint mutates classification due to user override.
- Confidence values come solely from the classifier; no hardcoded 0.9 boosts remain.
- Distribution is not zeroed or inflated to match override; at most annotate with `override` metadata.
- E2E: same `primary` and `confidence` visible in status, WS events, and final results.

## Tests
- Unit: `tests/test_paradigm_override_soT.py`
  - With override: submission record reflects override metadata; confidence unchanged.
  - No override: execution does not alter `primary`/`confidence` relative to submission.
- Integration: `tests/test_e2e_override_consistency.py`
  - Submit with override; ensure GET `/status/{id}` and final results align on `primary` and no confidence boost.

## Observability
- Prometheus counter `paradigm_overrides_total{source="user", paradigm=<x>}`.
- Structured audit logs for overrides.

## Rollback Plan
- Re‑enable execution‑time override (not recommended); keep audit logging in place.

## Risks
- Frontend expectations if UI relied on inflated confidence; coordinate FE to avoid regressions.


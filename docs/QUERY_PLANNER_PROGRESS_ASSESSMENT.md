# Query Planner – Progress Assessment (as of September 19, 2025)

TL;DR
- Unified planner is integrated end-to-end: planner emits QueryCandidate items; SearchAPIManager executes the provided plan; orchestrator consumes planner output and drives follow‑ups. Telemetry is wired to Redis and Prometheus, with a new /system/search-metrics endpoint for dashboards. Provider path no longer performs re‑expansion when planned candidates are present.

## Scope Covered
- Planner adapters + deterministic merge/dedup: `four-hosts-app/backend/search/query_planner/planner.py`, `.../types.py`.
- Query planning package (foundations for PR2/PR3): `four-hosts-app/backend/services/query_planning/*` (types, variator, cleaner, relevance_filter, result_deduplicator, config, planning_utils).
- Manager/provider execution path: `four-hosts-app/backend/services/search_apis.py` (planned candidates required for `search_all`, provider helpers call `api.search_with_variations(..., planned=...)`).
- Orchestrator integration + follow-ups loop: `four-hosts-app/backend/services/research_orchestrator.py`.
- CE reuse of planner preview: `four-hosts-app/backend/services/context_engineering.py`.
- Telemetry & persistence: `four-hosts-app/backend/services/telemetry_pipeline.py`, `four-hosts-app/backend/services/cache.py`, `four-hosts-app/backend/services/monitoring.py`, `four-hosts-app/backend/routes/system.py`.

## Current Status
- Planner core (PR1 intent) — COMPLETE
  - Initial plan and follow-ups produce labeled candidates (stage:label) with dedup by canonical form and Jaccard threshold.
- Manager/provider wiring — COMPLETE
  - Manager executes plan across providers; provider method `search_with_variations(..., planned=...)` is the authoritative entry. Result adapter exposes `query_variant` for stage:label.
- Orchestrator seam — COMPLETE
  - Uses planner.initial_plan; counts executed queries in request‑scoped metrics; follow‑ups loop integrated; dedup stats surfaced in `metadata`.
- Telemetry — COMPLETE (Phase A)
  - Per‑run records persisted to Redis (with in‑memory fallback) and exported to Prometheus. New GET `/system/search-metrics` aggregates timeline, provider usage/cost, and dedup rate.
- Env mapping — PARTIAL
  - CE provides `CE_PLANNER_*` knobs; unified `UNIFIED_QUERY_*` variables are documented but not yet the single source.

## Validation & Tests Executed
- Unit/contract
  - `pytest four-hosts-app/backend/tests/test_contracts_models.py` — PASS
  - `pytest four-hosts-app/backend/tests/test_query_planner_basic.py` — PASS
- Telemetry/metrics
  - `pytest four-hosts-app/backend/tests/test_metrics_extended_stats.py` — SKIP for app-bound test; shape/route smoke covered; non-app test parts pass.
- Integration (orchestrator)
  - `pytest four-hosts-app/backend/tests/test_orchestrator_p0.py` — PASS (note: aiohttp session warning, non-blocking)
- Frontend
  - Build OK; export UI targets POST `/v1/export/research/{id}`. Lint has pre‑existing `any` warnings unrelated to planner (to address separately).

## Known Gaps / Risks
- Budget-aware gating not enforced
  - `BudgetAwarePlanner` is instantiated in orchestrator but not consulted to select tools or enforce spend caps. Risk: uncontrolled spend in follow‑ups.
- Coverage telemetry
  - `coverage_ratio` is computed to drive follow‑ups but not emitted as telemetry; prevents alerting for under‑coverage regressions.
- Env drift
  - Legacy flags (`ENABLE_QUERY_LLM`, etc.) still exist alongside planner controls; risk of divergent behavior between services.
- Provider coverage
  - Provider implementations rely on base `search_with_variations(..., planned=...)`; end‑to‑end tests with real provider keys are still sparse.
- Rollback path
  - Planned path is default; a clean “legacy variations” fallback toggle is not wired. Rollback would require a short hotfix if needed.

## Metrics & Observability
- Request‑scoped metrics included in orchestrator response: `total_queries`, `total_results`, `apis_used`, `deduplication_rate`. File: `four-hosts-app/backend/services/research_orchestrator.py:1178`.
- Dedup pipeline: `ResultDeduplicator.deduplicate_results` with simhash bucketing and URL de‑dupe. Files: `.../query_planning/result_deduplicator.py`, metrics update at `.../research_orchestrator.py:2071–2081, 2213–2217`.
- Telemetry pipeline: `services/telemetry_pipeline.py` persists to Redis; Prometheus series include runs, queries, results, processing time histogram, dedup rate gauge, per‑provider usage/cost.
- Dashboards/Exports: GET `/system/search-metrics` returns windowed aggregates for Grafana/Amplitude ingestion.

## API & Contract Notes
- Export API migrated to POST `/v1/export/research/{id}`; legacy GET route removed. Files: `backend/services/export_service.py`, `backend/core/app.py`, `backend/routes/research.py` (removed block). Frontend updated: `frontend/src/services/api.ts:exportResearch`.
- ResultAdapter exposes `query_variant` to retain stage:label on results. File: `backend/services/result_adapter.py`.
- Contracts module remains side‑effect free; fixed missing `Any` import.

## Rollout Plan (Proposed)
- Phase 1 (now): Keep planner path default with provider quotas, increase telemetry sampling, validate dashboard baselines for dedup rate and provider mix.
- Phase 2 (Sep 23 – Oct 4):
  - Enforce budget gating for follow‑ups (turn on `BudgetAwarePlanner` decisions with soft alerts first).
  - Persist coverage telemetry; alert if coverage < threshold and follow‑ups disabled/exhausted.
  - Complete env unification to `UNIFIED_QUERY_*` and remove legacy toggles.
- Phase 3: Provider e2e tests using live keys in nightly (quota‑safe) and synthetic fixtures in CI.

## Action Items
- Telemetry completion
  - Persist `coverage_ratio` and `stage_breakdown` per run; add alerting for dedup drift and provider quota exhaustion.
- Budget enforcement
  - Use `BudgetAwarePlanner.select_tools` for provider choice and `record_tool_spend` gates; expose per‑request caps in user context.
- Env unification
  - Map CE `CE_PLANNER_*` to `UNIFIED_QUERY_*`; remove legacy flags; update docs and CI.
- Tests
  - Add unit tests for `result_deduplicator`, `relevance_filter`, and budget gates; add smoke tests for provider planned path.

## References (Repo paths)
- Planner core: `four-hosts-app/backend/search/query_planner/planner.py`
- Planning package: `four-hosts-app/backend/services/query_planning/*`
- Manager/provider: `four-hosts-app/backend/services/search_apis.py`
- Orchestrator: `four-hosts-app/backend/services/research_orchestrator.py`
- Telemetry: `four-hosts-app/backend/services/telemetry_pipeline.py`, `.../services/cache.py`, `.../services/monitoring.py`, `.../routes/system.py`
- Frontend export client: `four-hosts-app/frontend/src/services/api.ts`

---
Maintainer note: This file reflects repository state validated on September 19, 2025 (US/Eastern). For subsequent changes, update dates and code references as lines move.


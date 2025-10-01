# Research Pipeline Assessment – October 1, 2025

## Current Flow Overview

- **Ingestion:** `routes/research.py` assembles classification/context payloads, tunes depth-specific settings, and invokes `UnifiedResearchOrchestrator.execute_research` with a triage context shim (`four-hosts-app/backend/routes/research.py:401-454`).
- **Core Orchestration:** The orchestrator coordinates planning, multi-provider search, deduplication, credibility scoring, EXA augmentation, and answer synthesis, emitting progress callbacks for search and analysis phases (`four-hosts-app/backend/services/research_orchestrator.py:709-3120`).
- **Operational Layers:** Triage board management (`four-hosts-app/backend/services/triage.py:160-295`), WebSocket progress broadcasting (`four-hosts-app/backend/services/websocket_service.py:700-1075`), and the shared task registry (`four-hosts-app/backend/services/task_registry.py:20-148`) provide runtime visibility and lifecycle control.

## Key Gaps & Risks

1. **Missing Triage Phases** – The orchestrator never signals `classification`, `context`, `synthesis`, or `review`, so the triage board stalls at search/analysis despite documentation promising nine lanes (`four-hosts-app/backend/services/research_orchestrator.py:992-2990`, `four-hosts-app/backend/services/websocket_service.py:1018-1055`).
2. **Silent Failure Surface** – Hundreds of `except Exception` handlers still swallow errors or downgrade them to debug logs, obscuring provider outages and synthesis faults (`four-hosts-app/backend/services/research_orchestrator.py:245-3539`).
3. **Dormant ML Retraining** – The ML pipeline’s background loop is commented out; it lacks task-registry integration and telemetry, so model quality hinges on manual triggers (`four-hosts-app/backend/services/ml_pipeline.py:42-165`).
4. **Task Registry Cleanup Gap** – `_on_task_done` still spawns tasks inside the callback and `_cleanup_tasks` is unmanaged, risking leaked coroutines during shutdown (`four-hosts-app/backend/services/task_registry.py:70-101`).
5. **Limited E2E Coverage** – Only unit tests exercise the triage manager; no integration checks guarantee blocked reasons and lane sequencing propagate across REST + WebSocket paths (`four-hosts-app/backend/tests/test_triage_manager.py:9-88`).

## Recommended Next Steps

1. **Restore Full Lane Fidelity (P0)**
   - Emit progress boundaries for classification, context engineering, synthesis, review, and completion directly from `execute_research` and feed them through the WebSocket layer.
   - Add an integration test that mocks a research run and asserts ordered lane transitions on the triage board.

2. **Tighten Error Reporting (P0)**
   - Replace silent `except Exception` sites in search orchestration, EXA augmentation, and credibility scoring with structured warnings/errors that include `research_id`, phase, and provider metadata.
   - Prioritize external API touchpoints and cancellation handlers before lower-risk metadata merges.

3. **Automate ML Retraining (P1)**
   - Stand up an application-managed retraining task that respects `_retrain_lock`, registers with `TaskRegistry`, and logs cycle metrics via the metrics facade.

4. **Harden Background Task Lifecycle (P1)**
   - Queue cleanup coroutines outside `_on_task_done`, ensure `_cleanup_tasks` drains, and audit orchestrator/websocket services for registry registration.

5. **Extend E2E Verification (P2)**
   - Enhance `scripts/verify-triage-enhanced.sh` and add automated tests that confirm blocked reasons, lane updates, and synthesis metadata survive REST and WebSocket delivery once new phase hooks land.

## Test Coverage

- `cd four-hosts-app/backend && pytest -q tests/test_triage_manager.py` (passes; deprecation warnings for SQLAlchemy/Pydantic remain outstanding repo-wide).


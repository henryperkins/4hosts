# Backend Overlap Analysis

This document summarizes overlapping functionality across backend services, routes, and utilities, and recommends targeted consolidations.

## Authentication

- Password and JWT helpers are implemented in multiple places.
  - `services/auth.py`: `hash_password`, `validate_password_strength`, `create_access_token`, `create_refresh_token`, `decode_token`.
  - `services/auth/password.py` and `services/auth/tokens.py`: parallel implementations of the same.
  - Consolidation: Keep canonical implementations in `services/auth/password.py` and `services/auth/tokens.py`; have `services/auth.py` import and (optionally) re‑export. Update callers to import from a single place.

- “Current user” and role dependencies exist twice.
  - `services/auth.py`: `get_current_user`, `require_role`.
  - `core/dependencies.py`: `get_current_user`, `get_current_user_optional`, `require_role`.
  - Consolidation: Use `core/dependencies.py` as the sole FastAPI dependency layer; keep or drop `services/auth.get_current_user` (prefer drop) and re‑point remaining imports.

## Routes (duplicate or overlapping endpoints)

- Preferences: two endpoints with different paths/verbs.
  - PUT/GET `/auth/preferences` in `routes/auth.py`.
  - POST `/users/preferences` in `routes/users.py`.
  - Consolidation: Keep `/users/preferences` for user‑centric concerns; alias or deprecate `/auth/preferences`.

- History: two endpoints for user research history.
  - GET `/research/history` in `routes/research.py`.
  - GET `/users/history` in `routes/users.py`.
  - Consolidation: Keep `/research/history` and alias/deprecate `/users/history`. Both read from `services.research_store`.

- Export responsibility split between route and service.
  - `routes/research.py` implements GET `/research/export/{research_id}/{fmt}`.
  - `services/export_service.py` provides `create_export_router(...)` to mount a full router.
  - Consolidation: Mount the service router in `core/app.py` and remove in‑route export logic (or forward to the service router).

## LLM / Deep Research

- Overlapping client layers for OpenAI/Azure.
  - `services/llm_client.py`: general chat, tools, streaming, background helper.
  - `services/openai_responses_client.py`: Responses API + deep research helpers + streaming + extraction.
  - `services/background_llm.py`: Background Responses polling and result extraction (duplicates extraction logic).
- Consolidation: Define a single LLM provider façade (keep `llm_client`) and move Responses API + extraction there. Reuse one extraction utility across modules. Keep `deep_research_service` as orchestration, not a second client.

## WebSockets

- Split responsibilities across two modules with overlapping logic.
  - `services/websocket_auth.py`: token extraction/validation, origin checks, and WebSocket‑specific rate limits (`WS_RATE_LIMITS`).
  - `services/websocket_service.py`: connection management, research subscriptions, message routing, and it also imports token decode utilities.
  - Consolidation: Keep auth and rate‑limit logic in `websocket_auth.py` as the only entry for WS security; ensure `websocket_service.py` depends on a small interface (e.g., `authenticate_websocket`, `verify_websocket_rate_limit`) without duplicating token handling. Centralize WS rate‑limit tiers next to API rate limits.

## Rate Limiting & Quotas

- API rate limits and WS rate limits defined in multiple places.
  - API: `services/rate_limiter.py` uses `RATE_LIMITS` from `services/auth.py`.
  - WebSocket: `services/websocket_auth.py` defines independent `WS_RATE_LIMITS`.
  - Consolidation: Create a single `limits.py` (or move to `core/config`) that defines role tiers and per‑channel (API/WS) knobs. Keep `RATE_LIMITS`/`WS_RATE_LIMITS` as views over a single source of truth to prevent drift.

## Research Store vs Cache

- Research state stored and cached in two subsystems.
  - `services/research_store.py`: authoritative research records (Redis with in‑memory fallback, 24h TTL).
  - `services/cache.py`: additional short‑TTL caches for `research_status` and `research_results` plus general KV.
  - Consolidation: Define `research_store` as the single read/write path for research records; allow `cache.py` to provide only transient, read‑through caches with a unified TTL policy. Add a thin helper in `research_store` to expose fast status lookups so routes don’t reach into `cache.py` directly.

## Result Normalization & Adaptation

- Duplicate result adaptation and normalization.
  - `services/result_adapter.py`: rich adapters (`ResultAdapter`, `ResultListAdapter`) and helpers.
  - `services/enhanced_integration.py`: custom `_adapt_search_results` logic repeats similar normalization.
  - `services/research_orchestrator.py`: additional in‑line normalizations.
  - Consolidation: Standardize on `ResultAdapter` utilities; refactor orchestrators/generators to use `adapt_results(...).to_dict_list()` to avoid repeated field access patterns and reduce type‑handling bugs.

## Paradigm Definitions & Prompts

- Paradigm metadata scattered across modules.
  - `core/config.PARADIGM_EXPLANATIONS` – UI labels/descriptions.
  - `services/llm_client._SYSTEM_PROMPTS` – system prompts for each paradigm.
  - `services/classification_engine.py` – paradigm enums and reasoning keywords.
  - `routes/paradigms.get_paradigm_approach_suggestion` – per‑paradigm suggestions.
  - Consolidation: Centralize paradigm constants/prompts in a `models/paradigms_data.py` (or extend `models/paradigms`) with UI text, prompts, and approach suggestions. Import from this single module across routes, clients, and classifiers.

## Context Engineering & Token Budgets

- Mixed budget logic across utilities and services.
  - `utils/token_budget.py`: estimation, trimming, selection, and budget plans.
  - `services/text_compression.py`: custom token budgeting/allocations.
  - `services/llm_client.py`: per‑call max token defaults/overrides.
  - Consolidation: Use `utils/token_budget` as the single budget engine. Have compression and LLM calls read from a shared budget plan (attach to context) instead of each module computing their own caps.

## Orchestrator & Answer Generation

- Overlapping responsibilities around synthesis entrypoints and evidence handling.
  - `services/enhanced_integration.py`: compatibility façade for `generate_answer` (legacy vs new signature) and pre‑adaptation of results.
  - `services/research_orchestrator.py`: prepares synthesis context, evidence quotes, and then calls the generator.
  - `services/answer_generator.py`: heavy lifting for sectioned synthesis, citations, and evidence blocks.
  - Consolidation: Keep `AnswerGenerationOrchestrator` as single synthesis engine; keep signature adapter only in one place (enhanced_integration) and ensure the orchestrator always calls the same façade. Move evidence formatting helpers to one module (generator) and reuse.

## Additional Observations

- Token/config constants are duplicated.
  - `core/config.ACCESS_TOKEN_EXPIRE_MINUTES` vs values in `services/auth.py` and `services/auth/tokens.py`.
  - Consolidation: Read all auth constants from `core/config` (or a dedicated `settings.py`) and inject into services.

- Monitoring vs system routes.
  - Keep stats aggregation in `services/monitoring.py` (or a small `system_service`), make `routes/system.py` thin delegates.

## Concrete Next Actions (Expanded)

1) Centralize role limits
   - Create `core/limits.py`: expose `get_api_limits(role)` and `get_ws_limits(role)`.
   - Update `rate_limiter.py` and `websocket_auth.py` to consume shared config.

2) Unify WebSocket stack
   - Keep security/rate‑limit in `websocket_auth.py`; remove any token decode from `websocket_service.py`.
   - Ensure all WS endpoints enter through `secure_websocket_endpoint` then hand off to `ConnectionManager`.

3) Normalize result handling
   - Replace `_adapt_search_results` usage with `ResultAdapter` flows and remove duplication.
   - Add small helpers in `result_adapter.py` if any fields are missing.

4) Deduplicate paradigm data
   - Move prompts, approach suggestions, and UI explanations to a single module; update `llm_client`, `routes/paradigms`, and UI serializers to read from it.

5) Research store as authority
   - Add `research_store.get_status(research_id)` and `set_status(...)` helpers; have `routes/research.py` and WS progress use these; keep `cache.py` in read‑through mode only.

6) LLM client consolidation
   - Fold Responses API content extraction into `llm_client` (single `_extract_content_safely`).
   - Make `deep_research_service` use only the unified client layer.

7) Auth constants and helpers
   - Move token expiry and algorithm to config; remove duplicates in `services/auth.py` and `services/auth/tokens.py` and re‑export.

## Search / Brave Integrations

- Multiple Brave paths with similar goals.
  - `services/search_apis.py`: general search manager (URL normalization, fetch, ranking, dedupe).
  - `services/brave_grounding.py`: Brave Chat/Summarizer citations.
  - `services/brave_mcp_integration.py`: MCP‑based Brave search config + tool execution.
  - Consolidation: Provide a single `BraveAdapter` with implementations for Direct API (grounding) and MCP; select via config/env. Centralize Brave config parsing to avoid drift. Expose via `search_apis` to callers.

## System / Monitoring

- Stats aggregation split between routes and monitoring service.
  - `routes/system.py`: `stats` and `public-stats` aggregate DB/store snapshots.
  - `services/monitoring.py`: metrics, health checks, system telemetry.
  - Consolidation: Keep business/system stats shaping in a service (monitoring or a small `system_service`) and have routes delegate.

## Docs / OpenAPI

- Custom docs helpers duplicated for test tooling.
  - `utils/custom_docs.py` (used by `core/app.py`).
  - `tests/generate_openapi.py` re‑implements similar helpers.
  - Consolidation: Make the test tool import helpers from `utils/custom_docs.py` to prevent divergence.

## Minor Service Overlaps

- Token management wrappers.
  - `services/token_manager.py` is authoritative for refresh tokens/JTI revocation.
  - `services/auth/tokens.py` and portions of `services/auth.py` re‑expose similar logic.
  - Consolidation: Keep `token_manager` as source of truth; thin wrappers only if necessary for dependency ergonomics.

- Truncation/sanitization utilities exist in several modules.
  - `services/text_compression.py`, `services/search_apis.py` (safe truncation), evidence formatting in `services/evidence_builder.py`.
  - Consolidation: Factor into utils (extend `utils/injection_hygiene.py` with shared helpers) and reuse everywhere.

## Recommended Quick Wins

- Auth unification
  - Canonicalize password/JWT in `services/auth/{password,tokens}.py`; drop duplicates from `services/auth.py` and re‑export for compatibility.
  - Ensure routes depend only on `core/dependencies.py` for auth context.

- Route normalization
  - Keep `/users/preferences`; alias/deprecate `/auth/preferences`.
  - Keep `/research/history`; alias/deprecate `/users/history`.

- Export routes
  - Mount `create_export_router(app.state.export_service)` under `/v1` in `core/app.py` and remove/forward the manual export endpoint in `routes/research.py`.

- LLM client consolidation
  - Centralize response parsing/streaming into one utility used by both `llm_client` and background flows.
  - Have `deep_research_service` call the unified client only.

- Brave consolidation
  - Create `services/brave/` with a single config surface and adapter; wire both grounding and MCP implementations under it; expose via `search_apis`.

## Notable Files Involved

- Services: `services/auth.py`, `services/auth/password.py`, `services/auth/tokens.py`, `services/token_manager.py`, `services/llm_client.py`, `services/openai_responses_client.py`, `services/background_llm.py`, `services/search_apis.py`, `services/brave_grounding.py`, `services/brave_mcp_integration.py`, `services/export_service.py`, `services/monitoring.py`.
- Routes: `routes/auth.py`, `routes/users.py`, `routes/research.py`, `routes/system.py`, `routes/search.py`.
- Utils: `utils/custom_docs.py`, `utils/injection_hygiene.py`, `utils/token_budget.py`.

---

If you want, I can turn the “Recommended Quick Wins” into a concrete PR plan (file changes and sequence) and start applying the smallest refactors first (auth + routes aliases), followed by wiring the export router in `core/app.py`.

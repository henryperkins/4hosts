# Backend Overlap Analysis

This document summarizes overlapping functionality across backend services, routes, and utilities, and recommends targeted consolidations.

## Progress Update — 2025-09-06

- Reviewed and digested the following services for overlap analysis:
  - `services/answer_generator.py`
  - `services/classification_engine.py`
  - `services/research_orchestrator.py`
  - `services/credibility.py`
  - `services/search_apis.py`
  - `services/context_engineering.py`
- Integrated findings below, focusing on shared responsibilities, duplicated logic, and type/contract drift between layers (classification → context engineering → search → credibility → orchestration → answer generation).

## Overlaps Across Answer Generation, Classification, Orchestrator, Credibility, Search, Context

- Paradigm enums, names, and keywords are defined in multiple places.
  - `classification_engine.HostParadigm` defines values like `analytical/strategic/...` while usage elsewhere frequently relies on lowercase code names like `"bernard"`, with normalization helpers in `models.paradigms`. `answer_generator` keeps its own `PARADIGM_KEYWORDS`; `classification_engine.QueryAnalyzer` maintains a different, larger keyword map and regex patterns. Consolidate paradigm identity and keyword canon in one place (see Consolidation Plan).

- Evidence surfaces and signals flow are similar but not unified.
  - `context_engineering.IsolateLayer` outputs `isolated_findings` (matches, by_domain, focus_areas). `answer_generator` consumes `evidence_quotes` and peeks into `context_engineering` dict to build coverage tables; `research_orchestrator` forwards some of these into synthesis metadata. Define a single "EvidenceBundle" payload produced by Context Engineering and passed through Orchestrator to Answer Generation, instead of ad‑hoc dict introspection.

- Progress reporting is embedded in several modules.
  - `answer_generator`, `classification_engine`, `context_engineering`, and `research_orchestrator` all import `websocket_service.progress_tracker` and perform similar `update_progress` calls. Extract a thin `services/progress.py` facade with phase/step helpers to avoid repeated try/except, unify messages, and enable no‑op/testing stubs.

- Source credibility, authority, and consensus logic is split.
  - `services/credibility` computes domain authority (Moz/heuristic), bias/factual ratings, recency decay, cross‑source agreement, controversy; `research_orchestrator` also has an `EarlyRelevanceFilter` with its own low‑quality domain list; `search_apis.ContentRelevanceFilter` evaluates relevance and implements an internal consensus detector; `answer_generator` selects top results by `credibility_score` already attached by upstream layers. Merge authority/bias/recency scoring behind `credibility.get_source_credibility()` and have `search_apis` and the orchestrator call into it for a single score, removing duplicate domain/quality lists.

- URL normalization and deduplication exist in multiple layers.
  - `search_apis.URLNormalizer` and per‑provider shaping; `research_orchestrator.ResultDeduplicator` (URL + content Jaccard); `SearchResult` also carries a `content_hash`. Promote URL normalization to the search layer contract and standardize dedup on `SearchResult.content_hash` + lightweight text similarity to reduce triple maintenance.

- Query rewriting/optimization paths overlap.
  - `search_apis.QueryOptimizer` performs entity protection + variations; `context_engineering.Rewrite/Optimize` layers do similar LLM/heuristic rewrites and variations. Keep optimization in Context Engineering as the canonical producer, and have `search_apis` accept already‑optimized variants when present; fall back to its own optimizer only when none supplied.

- Cross‑source agreement shows up twice.
  - `credibility.CrossSourceAgreementCalculator` and `search_apis.ContentRelevanceFilter._detect_cross_source_agreement` both implement agreement/consensus boosts. Unify on the credibility module and expose a small function that `search_apis` can call for boosting diagnostics to avoid divergent heuristics.

- Recency/freshness is modeled twice.
  - `credibility.RecencyModeler` (exponential decay + breaking boost) and `search_apis` date filtering/ranking. Keep time decay in credibility; have `search_apis` forward timestamps to credibility and apply the returned recency score as a ranking feature rather than re‑implement.

- Type drift between synthesis models.
  - `answer_generator` defines V1 dataclasses (`Citation`, `AnswerSection`, `GeneratedAnswer`) for compatibility while `models.synthesis_models` supplies `SynthesisContext`. Prefer canonical models in `models.synthesis_models` and isolate V1 shims in a compatibility module, not inside the generator.

## Consolidation Plan (Targeted)

- Paradigm Canon [In Progress]
  - Name/code normalization helpers exist in `models/paradigms.py`. Canonical keyword sets remain duplicated in `answer_generator.py` and `classification_engine.py` and should be centralized.

- Evidence Transport [Not Started]
  - Define `EvidenceBundle` in models and pass it from Context Engineering through Orchestrator to Answer Generator; remove ad‑hoc dict peeking.

- Progress Facade [Not Started]
  - Introduce `services/progress.py` and replace inline `progress_tracker` imports across modules.

- Credibility As Source of Truth [Not Started]
  - Replace static low‑quality lists and internal consensus detection with calls into `credibility` (authority/recency/consensus). 

- Dedup Normalization [Partially Completed]
  - URL normalization and `content_hash` exist; switch orchestrator dedup to `content_hash` fast path before Jaccard.

- Query Optimization Ownership [Partially Completed]
  - Orchestrator consumes `refined_queries` from Context Engineering; update `SearchAPIManager` to skip internal variations when provided.

- Synthesis Models [Not Started]
  - Move V1 answer dataclasses out of `answer_generator.py` into a `models/synthesis_compat.py` module.

## Concrete Hotspots and Actions

- Duplicate paradigm keywords
  - Files: `services/answer_generator.py` (PARADIGM_KEYWORDS), `services/classification_engine.py` (QueryAnalyzer.PARADIGM_KEYWORDS). Action: replace both with `models/paradigms.PARADIGM_KEYWORDS` import.

- Mixed paradigm identifiers
  - Files: `classification_engine.HostParadigm` vs string codes in `answer_generator`/`research_orchestrator`. Action: enforce `HostParadigm` end‑to‑end; keep normalization helper for external inputs only.

- Parallel consensus logic
  - Files: `services/credibility.py` vs `services/search_apis.py` agreement code. Action: export `compute_agreement(results_or_domains, key_terms)` from credibility and consume it in search ranking.

- Low‑quality domain lists vs computed credibility
  - Files: `research_orchestrator.EarlyRelevanceFilter`. Action: delete static lists; call `get_source_credibility(domain, paradigm)` and apply thresholds (e.g., overall_score < 0.35 → drop early).

- URL normalization
  - Files: `services/search_apis.URLNormalizer` and downstream consumers. Action: treat normalized URL as part of `SearchResult` construction; ensure orchestrator never receives non‑normalized URLs.

- Progress reporting drift
  - Files: the four services above plus orchestrator. Action: replace inline `progress_tracker` calls with `services/progress.py` to unify phrasing and phase names.

## Suggested Review Sequence (Low‑Risk First)

1) Paradigm canon and keyword unification (pure refactor, no behavior change)
2) Progress facade adoption (no behavior change; easier testing)
3) URL normalization + content_hash‑first dedup (safe, deterministic)
4) Early relevance filter → credibility thresholds (behavior improves, keep feature flag)
5) Consensus/recency consolidation (search ranking parity; A/B flag)
6) EvidenceBundle transport and AnswerGenerator updates (exposes richer UI signals)

## Notes On Current Strengths

- `search_apis.SearchResult` already carries rich metadata (sections, content_hash) that can power dedup/consensus.
- `credibility` implements a thorough, extensible scoring model (DA, bias/factual, recency, controversy, cross‑source agreement) and Brave grounding hooks — ideal as the single authority.
- `context_engineering` provides a clean W‑S‑C‑I pipeline with layer metrics, which is a good seam to standardize query optimization and evidence packaging.

## Authentication

- Password and JWT helpers are implemented in multiple places. [Completed]
  - `services/auth.py` now re‑exports canonical helpers from `services/auth/password.py` and `services/auth/tokens.py`; callers can import from one place.

- “Current user” and role dependencies exist twice. [Completed]
  - Routes use `core/dependencies.py` (`get_current_user`, `get_current_user_optional`, `require_role`); `services/auth.get_current_user` remains for internal use.

## Routes (duplicate or overlapping endpoints)

- Preferences: two endpoints with different paths/verbs. [Completed]
  - `/auth/preferences` now responds with Deprecation/Link headers to `/v1/users/preferences`; `/users/preferences` is primary.

- History: two endpoints for user research history. [Completed]
  - `/users/history` responds with Deprecation/Link headers to `/v1/research/history`; `/research/history` is canonical.

- Export responsibility split between route and service. [Partially Completed]
  - Export router is mounted under `/v1/export` in `core/app.py`. The legacy `/research/export/{research_id}/{fmt}` route still exists — retire or forward it to the service to avoid drift.

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

- API rate limits and WS rate limits defined in multiple places. [Completed]
  - Unified in `core/limits.py` and consumed by `services/rate_limiter.py` and `services/websocket_auth.py`.

## Research Store vs Cache

- Research state stored and cached in two subsystems.
  - `services/research_store.py`: authoritative research records (Redis with in‑memory fallback, 24h TTL).
  - `services/cache.py`: additional short‑TTL caches for `research_status` and `research_results` plus general KV.
  - Consolidation: Define `research_store` as the single read/write path for research records; allow `cache.py` to provide only transient, read‑through caches with a unified TTL policy. Add a thin helper in `research_store` to expose fast status lookups so routes don’t reach into `cache.py` directly.

## Result Normalization & Adaptation

- Duplicate result adaptation and normalization. [Completed]
  - `services/result_adapter.py` is used by `research_orchestrator.py` and `enhanced_integration.py` for normalized access (`adapt_results`, `ResultAdapter`).

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

- Auth unification [Completed]
  - Canonical helpers live in `services/auth/{password,tokens}.py` and are re‑exported by `services/auth.py`; routes use `core/dependencies.py`.

- Route normalization [Completed]
  - `/auth/preferences` and `/users/history` are deprecated in favor of `/v1/users/preferences` and `/v1/research/history`.

- Export routes [Partially Completed]
  - Export router mounted under `/v1/export`; retire/forward legacy route in `routes/research.py`.

- LLM client consolidation [Not Started]
  - Unify `llm_client`, `openai_responses_client`, and `background_llm` extraction/streaming.

- Brave consolidation [Not Started]
  - Introduce a single Brave adapter façade configurable for direct API vs MCP.

## Notable Files Involved

- Services: `services/auth.py`, `services/auth/password.py`, `services/auth/tokens.py`, `services/token_manager.py`, `services/llm_client.py`, `services/openai_responses_client.py`, `services/background_llm.py`, `services/search_apis.py`, `services/brave_grounding.py`, `services/brave_mcp_integration.py`, `services/export_service.py`, `services/monitoring.py`.
- Routes: `routes/auth.py`, `routes/users.py`, `routes/research.py`, `routes/system.py`, `routes/search.py`.
- Utils: `utils/custom_docs.py`, `utils/injection_hygiene.py`, `utils/token_budget.py`.

---

If you want, I can turn the “Recommended Quick Wins” into a concrete PR plan (file changes and sequence) and start applying the smallest refactors first (auth + routes aliases), followed by wiring the export router in `core/app.py`.

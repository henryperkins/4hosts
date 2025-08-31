# Application Alignment Assessment (SSOTA vs Implementation)

Verified on: August 31, 2025

This report verifies the app’s alignment with the four SSOTA docs (`docs/ssota-*.md`) against the current backend and routes. Status labels: Correct, Partial, Missing.

## Aligned (Correct)

- W‑S‑C‑I pipeline: Full Write/Select/Compress/Isolate implementation with orchestrated flow and metrics.
  - Files: `backend/services/context_engineering.py`; invoked from `backend/routes/research.py`.
- Hybrid classification: Rule+LLM combiner, distribution, confidence, reasoning.
  - File: `backend/services/classification_engine.py`.
- API structure and versioning: Routers mounted under `/v1` (auth, research, paradigms, search, users, system).
  - File: `backend/core/app.py` → `setup_routes()`.
- Caching strategy/TTLs: 24h search, 7d classification, 30d credibility; API cost tracking.
  - File: `backend/services/cache.py` (`ttl_config`).
- Basic security: JWT with revocation; CORS configured; CSRF token endpoint + middleware and test script.
  - Files: `backend/services/auth.py`, `backend/core/app.py`, `backend/middleware/security.py`, `backend/test-auth-flow.sh`.

## Reported “Critical Gaps” That Are Implemented (Correct)

- Self‑healing system: Monitors performance, recommends/executes paradigm switches, emits webhooks, tracked by monitoring.
  - Files: `backend/services/self_healing_system.py`; wiring in `backend/core/app.py`; use in `backend/services/enhanced_integration.py`.
- Circuit breakers + retries/backoff: Domain circuit breaker; tenacity‑based retries on HTTP & LLM; orchestrator’s retry/backoff.
  - Files: `backend/services/search_apis.py`, `backend/services/llm_client.py`, `backend/services/research_orchestrator.py`.
- Search APIs: Semantic Scholar, PubMed, CrossRef implemented with parsing + relevance filtering; Brave/Google flows present.
  - File: `backend/services/search_apis.py`.
- Paradigm‑aware source filtering: Per‑paradigm query gen and result ranking with credibility integration.
  - File: `backend/services/paradigm_search.py`.
- Health/monitoring: `/health` endpoint; Prometheus + OpenTelemetry stack and middleware present.
  - Files: `backend/core/app.py` (`/health`), `backend/services/monitoring.py`.

## True or Partial Gaps (Needs Alignment)

- Context memory management (Partial)
  - Compress layer computes `token_budget` from complexity, but budgets aren’t strictly enforced at prompt assembly per layer.
  - Action: Enforce per‑layer token caps and trimming when injecting knowledge/tool snippets (answer generator, deep research prompts).
- Prompt‑injection defenses (Missing)
  - No explicit injection/jailbreak detection/sanitization before re‑use of model/tool outputs.
  - Action: Add heuristic/LLM audit, strip unsafe patterns, and require verification before reuse.
- SSRF protections for fetcher (Partial)
  - Respectful fetcher normalizes URLs, honors robots.txt, rate‑limits, and uses a circuit breaker, but lacks private/loopback IP blocks and hostname allowlisting.
  - Action: Reject RFC1918/loopback/link‑local/metadata IPs; restrict schemes to http/https; optional domain allowlist.
- Context hygiene and quarantine (Partial)
  - Early relevance filtering and cross‑source agreement exist, but there’s no quarantine lane for suspicious content pending verification nor a “two‑source rule” for risky claims.
  - Action: Tag suspicious outputs; require cross‑source agreement ≥N or high credibility before inclusion.
- Observability exposure (Partial)
  - Prometheus metrics are collected but no `/metrics` route exposed; readiness probe exists as a service but no public `/ready` route.
  - Action: Add `/metrics` (Prometheus `generate_latest`) and `/ready` (uses `HealthCheckService.get_readiness()`).

## Suggested Fixes (High‑Value, Minimal Scope)

- Enforce per‑layer token caps: Apply `compress_output.token_budget` when assembling prompts in `backend/services/answer_generator.py` and `backend/services/deep_research_service.py`.
- Injection guardrails: Add a `prompt_hygiene.py` utility (regex heuristics + optional LLM audit) and gate reinjection; log/flag quarantined items.
- SSRF hardening: Extend `RespectfulFetcher` to block private/loopback/reserved IPs and non‑HTTP(S) schemes before aiohttp requests.
- Quarantine policy: Persist `quarantine` flags in `research_store`; surface in UI and synthesis rules (require verification/consensus).
- Metrics/readiness endpoints: Expose `/metrics` and `/ready`; add to rate‑limit/CORS/CSRF skip lists.

## Evidence Pointers (Non‑Exhaustive)

- W‑S‑C‑I: `backend/services/context_engineering.py`; usage: `backend/routes/research.py`.
- Classification: `backend/services/classification_engine.py`.
- Caching TTLs: `backend/services/cache.py`.
- Self‑healing: `backend/services/self_healing_system.py`, `backend/services/enhanced_integration.py`, `backend/core/app.py`.
- Circuit breaker/retries: `backend/services/search_apis.py`, `backend/services/llm_client.py`, `backend/services/research_orchestrator.py`.
- Credibility, bias, fact ratings: `backend/services/credibility.py`.
- Search APIs + strategies: `backend/services/search_apis.py`, `backend/services/paradigm_search.py`.
- Health/monitoring: `backend/core/app.py` (`/health`), `backend/services/monitoring.py`.

## Bottom Line

The foundation is strong and ahead of the initial gap list. Real alignment work is focused on: (1) strict per‑layer token budget enforcement, (2) prompt‑injection/SSRF hardening, (3) quarantine/verification workflow, and (4) exposing `/metrics` and `/ready` endpoints.

Update (initial implementation):
- Added token budget utilities and a default budget plan emitted from the Compress layer.
- Enforced knowledge/instruction budgets in section generation (answer generator) and deep research prompts.
- Exposed `/metrics` and `/ready` endpoints and exempted them from rate limiting.
- Added minimal injection hygiene: snippet sanitization, quarantine tagging in prompts, and a safeguard instruction to treat snippets as evidence only.

# Research Pipeline Issues Consolidation Report

This document consolidates VERIFIED_PIPELINE_ISSUES.md and CRITICAL_PIPELINE_ISSUES.md into a single, deduplicated catalog with normalized taxonomy, consistent anchors, prioritization, actions, guardrails, and acceptance criteria.

Anchors validated against commit bd2d2ab.

## Overview

Purpose: Consolidate VERIFIED_PIPELINE_ISSUES.md and CRITICAL_PIPELINE_ISSUES.md into a single, deduplicated catalog; normalize taxonomy; resolve inconsistencies; provide prioritization, actions, guardrails, and acceptance criteria.

Sources reviewed:
- VERIFIED_PIPELINE_ISSUES.md
- CRITICAL_PIPELINE_ISSUES.md
- four-hosts-app/backend/services/*, four-hosts-app/backend/core/* (line anchors validated)

Normalization standards applied:
- Severity: SEV0 (Blocker), SEV1 (Critical), SEV2 (High), SEV3 (Medium), SEV4 (Low)
- Status: Active, Mitigated, Monitoring, Planned, Backlog
- Environments: Dev, Staging, Prod (default: All)
- Stages: Search Init, Search Exec, Fetch, Normalize, Dedup, Synthesis, Orchestration, Progress/UX, Infra & Limits, Data Flow

All 17 items from both files are represented, deduplicated one-to-one. Minor line-number drifts and path naming variants were normalized (see Duplicate Mappings and Notes).

## Consolidated Issue Inventory

Unified IDs U-001 to U-017 map one-to-one across both files. Evidence anchors link to the first relevant line in the current codebase.

| ID | Title | Severity | Stage(s) | Components | Envs | Status | Evidence |
|---|---|---|---|---|---|---|---|
| U-001 | Missing URL handling in deep research | SEV0 | Normalize, Data Flow | research_orchestrator | All | Mitigated | four-hosts-app/backend/services/research_orchestrator.py:2066 |
| U-002 | Empty content drops without warning | SEV0 | Fetch, Normalize | research_orchestrator | All | Mitigated | four-hosts-app/backend/services/research_orchestrator.py:1749 |
| U-003 | Synchronous blocking during synthesis (no cancel) | SEV0 | Synthesis, Orchestration | research_orchestrator | All | Mitigated | four-hosts-app/backend/services/research_orchestrator.py:1450 |
| U-004 | Search provider timeout cascade | SEV2 | Search Exec | search_apis | All | Mitigated | four-hosts-app/backend/services/search_apis.py:1961 |
| U-005 | API quota exhaustion handling absent | SEV2 | Infra & Limits | search_apis | All | Mitigated | four-hosts-app/backend/services/search_apis.py:1543 |
| U-006 | LLM retry storm (aggressive retries) | SEV2 | Orchestration, Infra & Limits | llm_client | All | Mitigated | four-hosts-app/backend/services/llm_client.py:347 |
| U-007 | Background task polling timeout gaps | SEV3 | Orchestration | background_llm | All | Planned | four-hosts-app/backend/services/background_llm.py:40 |
| U-008 | WebSocket keepalive gap | SEV4 | Progress/UX, Infra | websocket_service | All | Planned | four-hosts-app/backend/services/websocket_service.py:105 |
| U-009 | Deduplication threshold over-aggressive | SEV4 | Dedup | research_orchestrator | All | Backlog | four-hosts-app/backend/services/research_orchestrator.py:126 |
| U-010 | Circuit breaker recovery too fast | SEV4 | Infra & Limits | search_apis | All | Backlog | four-hosts-app/backend/services/search_apis.py:862 |
| U-011 | Result normalization inconsistency | SEV2 | Normalize, Data Flow | search_apis, research_orchestrator, llm_client | All | Planned | four-hosts-app/backend/services/search_apis.py:1001 |
| U-012 | Evidence bundle merge fragility | SEV2 | Data Flow, Synthesis | research_orchestrator | All | Planned | four-hosts-app/backend/services/research_orchestrator.py:1306 |
| U-013 | Progress reporting gaps | SEV3 | Progress/UX | research_orchestrator | All | Planned | four-hosts-app/backend/services/research_orchestrator.py:1613 |
| U-014 | User rate limits not enforced | SEV2 | Infra & Limits | core/limits, research_orchestrator | All | Mitigated | four-hosts-app/backend/core/limits.py:9 |
| U-015 | Memory leak in WebSocket message history | SEV2 | Infra, Progress/UX | websocket_service | All | Mitigated | four-hosts-app/backend/services/websocket_service.py:101 |
| U-016 | No fallback for Azure OpenAI failure | SEV1 | Orchestration, Infra | llm_client | All | Mitigated | four-hosts-app/backend/services/llm_client.py:542 |
| U-017 | Search manager can be empty (silent) | SEV1 | Search Init | research_orchestrator, search_apis | All | Mitigated | four-hosts-app/backend/services/research_orchestrator.py:674 |

Notes:
- Where ranges were provided in source docs, evidence anchors link to the first directly relevant line. Some anchors were updated due to minor drift; original references are captured in Duplicate Mappings.
- Some CRITICAL entries used non-prefixed filenames; paths normalized to four-hosts-app/backend/services/* and four-hosts-app/backend/core/*.

## Duplicate Mappings

Each pair consolidated to a single ID with consistent taxonomy. “Verified ref” and “Critical ref” preserve original titles and (approximate) anchors.

| Unified ID | VERIFIED ref | CRITICAL ref | Normalization note |
|---|---|---|---|
| U-001 | Missing URL Handling in Deep Research (services/research_orchestrator.py:2056) | Missing URL Handling in Deep Research (research_orchestrator.py:2056) | Canonical anchor set to :2066 (first `if not url:`) |
| U-002 | Empty Content Drops Without Warning (services/research_orchestrator.py:1739) | Empty Content Drops Without Warning (research_orchestrator.py:1739) | Canonical anchor :1749 (first drop log) |
| U-003 | Synchronous Blocking During Answer Synthesis (services/research_orchestrator.py:1440) | Synchronous Blocking During Answer Synthesis (research_orchestrator.py:1440) | Canonical anchor :1450 (generate_answer call) |
| U-004 | Search Provider Timeout Cascade (services/search_apis.py:1963) | Search Provider Timeout Cascade (search_apis.py:1961) | Canonical anchor :1961 |
| U-005 | API Quota Exhaustion Handling (services/search_apis.py:1543) | API Quota Exhaustion Handling (multiple) | Canonical anchor :1543 (Google CSE rate reasons) |
| U-006 | LLM Retry Storm (services/llm_client.py:347) | LLM Retry Storm (llm_client.py:347) | Canonical anchor :347 |
| U-007 | Background Task Polling Timeout (services/background_llm.py:40) | Background Task Polling Timeout (background_llm.py:40) | Canonical anchor :40 |
| U-008 | WebSocket Keepalive Gap (services/websocket_service.py:105) | WebSocket Keepalive Gap (websocket_service.py:113-131) | Canonical anchor :105 (interval config) |
| U-009 | Dedup Over-Aggressive (services/research_orchestrator.py:126) | Dedup Over-Aggressive (research_orchestrator.py:136) | Canonical anchor :126 (constructor) |
| U-010 | Circuit Breaker Recovery Too Fast (services/search_apis.py:862,895) | Circuit Breaker Recovery Too Fast (search_apis.py:863) | Canonical anchor :862 (constructor) |
| U-011 | Result Normalization Inconsistency (multiple) | Result Normalization Inconsistency (multiple) | Canonical anchor search_apis.py:1001 (SearchResult dataclass) |
| U-012 | Evidence Bundle Merge Failure (services/research_orchestrator.py:1306) | Evidence Bundle Merge Failure (research_orchestrator.py:1306) | Canonical anchor :1306 |
| U-013 | Progress Reporting Gaps (services/research_orchestrator.py:1613) | Progress Reporting Gaps (multiple) | Canonical anchor :1613 |
| U-014 | User Rate Limits Not Enforced (core/limits.py:9-51) | User Rate Limits Not Enforced (limits.py) | Canonical anchor core/limits.py:9 |
| U-015 | Memory Leak in Message History (services/websocket_service.py:101) | Memory Leak in Message History (websocket_service.py:101) | Canonical anchor :101 |
| U-016 | No Fallback for Azure OpenAI Failure (services/llm_client.py:429,542) | No Fallback for Azure OpenAI Failure (llm_client.py:429-517) | Canonical anchor :542 (explicit failure) |
| U-017 | Search Manager Creation Can Fail Silently (services/research_orchestrator.py:674) | Search Manager Creation Can Fail Silently (research_orchestrator.py:674) | Canonical anchor :674; also see search_apis.py:2097-2120 |

## Prioritization and Rationale

- SEV0 (Blocker): U-001, U-002, U-003 — cause downstream data loss or uninterruptible blocks.
- SEV1 (Critical): U-016, U-017 — remove single points of failure and fail-fast behavior.
- SEV2 (High): U-004, U-005, U-006, U-011, U-012, U-014, U-015 — correctness, availability, and resilience.
- SEV3 (Medium): U-007, U-013 — UX/operational reliability improvements.
- SEV4 (Low): U-008, U-009, U-010 — tunables with minimal immediate risk.

## Critical Incidents and Immediate Actions

- U-001: Stop synthesizing `about:blank#citation-...` URLs or mark and handle them explicitly; preserve unlinked citations in a typed list to avoid downstream “invalid URL” logic.
- U-002: When content is empty after repair/fetch, surface a user-visible warning and retain a safe placeholder instead of unlogged drop; add a metric for empty-content drops.
- U-003: Add cancellation checks within `_synthesize_answer` and before/after `generate_answer` calls; propagate cancellation up to the HTTP/WebSocket layer.
- U-016: Implement automatic fallback from Azure to OpenAI when both configured; tag requests with backend used; optionally cache partial results on Azure failure.
- U-017: Fail fast if `create_search_manager()` yields zero providers (respecting env disables); return a clear error to the user with guidance.

## Detailed Issue Breakdowns

- U-001 Missing URL handling in deep research
  - Evidence: four-hosts-app/backend/services/research_orchestrator.py:2066
  - Symptom: Synthesizes `about:blank#citation-...` URLs for unlinked citations.
  - Impact: Downstream drops, domain inference errors, misleading provenance.
  - Fix direction: Keep unlinked citations as typed evidence items with `unlinked=True`; only assign URLs when valid.
  - Acceptance: No synthesized URLs; unlinked citations flow through to evidence/answer metadata and UI labels.

- U-002 Empty content drops without warning
  - Evidence: four-hosts-app/backend/services/research_orchestrator.py:1749
  - Symptom: Results dropped after repair/fetch if still empty.
  - Impact: Loss of potentially relevant but hard-to-fetch sources; user unaware.
  - Fix direction: Emit user-visible warning and a summary placeholder; metric on drop rate by domain.
  - Acceptance: Empty-content events increment metric, show progress warning, and do not silently vanish from counts.

- U-003 Synchronous blocking during synthesis (no cancel)
  - Evidence: four-hosts-app/backend/services/research_orchestrator.py:1450
  - Symptom: Long `generate_answer` call lacks cancellation checkpoints.
  - Impact: Users cannot cancel research during synthesis.
  - Fix direction: Add cooperative cancellation checks around long steps; honor client disconnects.
  - Acceptance: Cancelling during synthesis aborts within 1s; progress reflects cancellation.

- U-004 Search provider timeout cascade
  - Evidence: four-hosts-app/backend/services/search_apis.py:1961
  - Symptom: Shared timeout allows one slow provider to stall all.
  - Impact: Reduced recall or empty results under partial outages.
  - Fix direction: Per-provider timeouts and cancellation of stragglers; budget-aware orchestration.
  - Acceptance: A single slow provider no longer blocks others; pending tasks cancelled on budget expiry.

- U-005 API quota exhaustion handling absent
  - Evidence: four-hosts-app/backend/services/search_apis.py:1543
  - Symptom: Only Google CSE path checks specific quota errors; others lack tracking.
  - Impact: Avoidable failures, wasted calls after exhaustion.
  - Fix direction: Track quotas per provider, auto-disable on exhaustion with UI messaging.
  - Acceptance: When simulated exhaustion occurs, provider auto-disables and remaining providers proceed.

- U-006 LLM retry storm (aggressive retries)
  - Evidence: four-hosts-app/backend/services/llm_client.py:347
  - Symptom: Uniform retries can compound under load.
  - Impact: Amplifies 429/timeout conditions.
  - Fix direction: Reduce attempts; add jitter; integrate circuit breaker per backend + per-research.
  - Acceptance: Under induced 429s, total attempts bounded and backoff honored.

- U-007 Background task polling timeout gaps
  - Evidence: four-hosts-app/backend/services/background_llm.py:40
  - Symptom: 300s hard cap with limited recovery path.
  - Impact: Long deep-research tasks fail without robust recovery.
  - Fix direction: Configurable max duration, resume tokens, and user-visible status.
  - Acceptance: Extended tasks survive beyond 5 minutes or fail with actionable status.

- U-008 WebSocket keepalive gap
  - Evidence: four-hosts-app/backend/services/websocket_service.py:105
  - Symptom: 30s may be too sparse for strict proxies.
  - Impact: Silent disconnects during long runs.
  - Fix direction: Lower default to 15s in prod; expose env per env; add reconnection handling metric.
  - Acceptance: Reduced disconnect rate when keepalive <= 15s behind strict proxies.

- U-009 Deduplication threshold over-aggressive
  - Evidence: four-hosts-app/backend/services/research_orchestrator.py:126
  - Symptom: Similar academic sources deduped as duplicates.
  - Impact: Loss of unique-but-similar results (esp. Bernard).
  - Fix direction: Paradigm-specific thresholds; possibly increase default threshold to 0.9.
  - Acceptance: Comparable-but-distinct sources retained for Bernard-focused queries.

- U-010 Circuit breaker recovery too fast
  - Evidence: four-hosts-app/backend/services/search_apis.py:862
  - Symptom: Fixed 300s recovery encourages repeat failures for flaky domains.
  - Impact: Wasted calls, slower searches.
  - Fix direction: Exponential backoff; domain “cool-off” with cap.
  - Acceptance: Repeated failures increase recovery window; overall failure rate drops in tests.

- U-011 Result normalization inconsistency
  - Evidence: four-hosts-app/backend/services/search_apis.py:1001
  - Symptom: Mixture of dataclass, dict, and normalized structures.
  - Impact: AttributeErrors and ad-hoc conversions.
  - Fix direction: Single canonical contract for result shape across pipeline.
  - Acceptance: No conversions needed in orchestrator; type checks pass; unit tests cover shape.

- U-012 Evidence bundle merge fragility
  - Evidence: four-hosts-app/backend/services/research_orchestrator.py:1306
  - Symptom: Complex try/except builds typed bundles, can fail silently.
  - Impact: Missing quotes/citations in final answer.
  - Fix direction: Validate and log merges; fail-safe defaults; unit tests for deep+builder merge paths.
  - Acceptance: Deep citations reliably appear; merge errors are surfaced and recoverable.

- U-013 Progress reporting gaps
  - Evidence: four-hosts-app/backend/services/research_orchestrator.py:1613
  - Symptom: Gaps and swallowed exceptions around callbacks.
  - Impact: UI appears frozen while work continues.
  - Fix direction: Central progress reporter with error budget and fallbacks.
  - Acceptance: Under injected failures, progress remains monotonic and visible.

- U-014 User rate limits not enforced
  - Evidence: four-hosts-app/backend/core/limits.py:9
  - Symptom: Limits defined but not enforced in the orchestrator.
  - Impact: Resource exhaustion from heavy users.
  - Fix direction: Role-based semaphores and per-user concurrent gates in orchestrator entrypoints.
  - Acceptance: Over-limit calls queue or fail fast with 429 and retry-after.

- U-015 Memory leak in WebSocket message history
  - Evidence: four-hosts-app/backend/services/websocket_service.py:101
  - Symptom: Count-only limit without memory/TTL control.
  - Impact: Memory growth over long sessions.
  - Fix direction: Per-user memory cap or TTL/size-based eviction policy.
  - Acceptance: Heap remains bounded over soak test; reconnections still receive recent messages.

- U-016 No fallback for Azure OpenAI failure
  - Evidence: four-hosts-app/backend/services/llm_client.py:542
  - Symptom: Raises when no backends or Azure path fails without OpenAI fallback.
  - Impact: Total failure when Azure down or misconfigured.
  - Fix direction: If both configured, automatically failover to OpenAI; consider cached answers.
  - Acceptance: When Azure disabled or failing, OpenAI path serves request with clear telemetry.

- U-017 Search manager can be empty (silent)
  - Evidence: four-hosts-app/backend/services/research_orchestrator.py:674
  - Symptom: Manager may initialize with zero providers (when all disabled) without failing fast.
  - Impact: Pipeline continues with no results.
  - Fix direction: Validate manager non-empty; error with remediation guidance.
  - Acceptance: With providers disabled, request fails early with actionable message.

## Thematic Root-Cause Analysis

- Inconsistent data modeling: Multiple representations (dataclass/dict) cause fragile conversions (U-011, U-012).
- Missing guardrails: Weak cancellation, limited progress robustness, and no fail-fast checks (U-003, U-013, U-017).
- Resilience gaps: Uniform retries and fixed recovery windows amplify outages (U-004, U-006, U-010, U-016).
- UX visibility gaps: Silent drops and sparse keepalives hide real states (U-002, U-008, U-013, U-015).
- Limits & quotas: Limits defined but not enforced; quotas tracked inconsistently (U-005, U-014).

## Monitoring and Guardrails

- Metrics
  - Stage durations (p50/p95/p99), per-provider success/failure, cancellation count, empty-content drop rate, evidence merge failures, WebSocket disconnects, LLM retries, quota consumption.
- Alerts
  - Azure/OpenAI failures >10%; provider failures >50%; pipeline timeouts >5/min; memory >80%; WS disconnect rate >20%; evidence merge failures >1/min.
- Feature flags / env
  - `SEARCH_PROVIDER_TIMEOUT_SEC`, `SEARCH_TASK_TIMEOUT_SEC`
  - `DEDUP_SIMILARITY_THRESH` (paradigm overrides suggested)
  - `WS_KEEPALIVE_INTERVAL_SEC` (lower in prod)
  - `LLM_BG_MAX_SECS`, `LLM_BG_POLL_INTERVAL`
  - `AZURE_OPENAI_*`, `OPENAI_API_KEY` (enable fallback)
  - Proposed: `PIPELINE_CANCEL_POLL_MS`, `SEARCH_PER_PROVIDER_TIMEOUT_SEC`, `LLM_MAX_RETRIES`, `ENABLE_RATE_LIMIT_ENFORCEMENT`

## Remediation Plan and Timeline

- Immediate (0–2 days)
  - U-001 Stop URL synthesis; propagate unlinked citations as typed evidence with `unlinked=True`.
  - U-002 Add user-visible warnings + metric on empty-content drops; retain safe placeholders.
  - U-003 Add cooperative cancellation around synthesis and long steps; wire to client disconnects.
  - U-017 Validate non-empty search manager; fail fast with remediation guidance.

- Short-term (3–7 days)
  - U-004 Implement per-provider timeouts and cancel stragglers.
  - U-005 Integrate quota tracking and auto-disable providers when exhausted.
  - U-006 Retune retries with jitter; add circuit breaker per backend.
  - U-011 Define and adopt a single normalized result schema; update orchestrator adapters.
  - U-012 Make evidence merge explicit with validation + logging; add tests for deep+builder merges.
  - U-016 Add Azure→OpenAI fallback when both configured.

  - Supporting: U-013 strengthen progress callbacks with centralized reporter and error budget.

- Medium (1–2 weeks)
  - U-007 Extend background task management with resumability and better status.
  - U-008 Lower keepalive default in prod; add WS reconnection metrics.
  - U-009 Tune dedup thresholds by paradigm; default to 0.9 for Bernard if needed.
  - U-010 Exponential backoff for circuit breaker recovery.
  - U-014 Enforce role-based rate limits and per-user concurrency gates.
  - U-015 Add memory/TTL bounds to WS message history.

## Risks and Dependencies

- Provider policy changes can alter rate-limit semantics (U-004/U-005).
- Azure/OpenAI SDK/Responses API behaviors evolve (U-006/U-016/U-007).
- Schema normalization might require frontend adjustments (U-011/U-012/U-013).
- Increased observability may add minor overhead (mitigated via sampling).

## Testing and Acceptance Criteria

- Cancellation
  - Issue a cancel during synthesis; expect abort within 1s; verify no dangling tasks.
- Timeouts/quotas
  - Simulate a slow provider; verify others complete; stragglers cancelled.
  - Simulate quota exhaustion; provider auto-disables; remaining providers continue.
- Data modeling
  - Ensure unified result schema through orchestrator paths; no AttributeErrors.
  - Evidence merge tests: deep citations appear and deduped in final payload.
- Limits & memory
  - Role-based concurrency: over-limit requests queue or 429 with retry-after.
  - WS soak test for memory caps; reconnection replay bounded and correct.
- Fallbacks
  - Disable Azure at runtime; confirm OpenAI path serves with metric tagging.

Regression gates:
- `pytest -m "not integration"` passes locally for new unit tests.
- Add targeted async tests under backend `tests/` mirroring the above scenarios.

## Runbook and Documentation Updates

- Add a troubleshooting section for provider timeouts/quotas and fallback behaviors.
- Update `docs/azure_responses_api.md` with background/resume guidance and failure modes.
- Add a “data shape” appendix to contract docs describing normalized search result schema.
- Document new env flags (`SEARCH_PER_PROVIDER_TIMEOUT_SEC`, `ENABLE_RATE_LIMIT_ENFORCEMENT`, etc.).

## Open Questions

- Should unlinked citations render in UI as “Unlinked source” or be hidden behind an expandable panel?
- What is the acceptable default keepalive in prod (10s/15s/20s)?
- Is OpenAI fallback always permitted for enterprise tenants (compliance)?
- Preferred behavior when all providers disabled: fail-fast vs. partial deep-only mode?

## Appendices

- Canonical anchors
  - research_orchestrator: 2066, 1749, 1450, 1306, 1613, 674
  - search_apis: 1961, 1543, 862, 1001
  - llm_client: 347, 542
  - background_llm: 40
  - websocket_service: 101, 105
  - core/limits: 9

---
Maintained by: Engineering (Backend + Platform). Update anchors when related code moves materially.

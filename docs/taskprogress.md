Progress report: Four-Hosts backend consolidation (Notes 1–2, A–Q)

Completed/landed changes

1) Canonical preferred sources registry
- Added shared registry [PREFERRED_SOURCES](four-hosts-app/backend/models/paradigms_sources.py:1) and refactored consumers to import it:
  - [import in paradigm_search.py](four-hosts-app/backend/services/paradigm_search.py:18)
  - Credibility preferences switched to the shared registry (see diff note below; file modified)

Impact:
- Eliminates duplicated lists in strategies and credibility (Note K).
- Central point for future adjustments.

2) Domain/source categorization utility
- Introduced [categorize()](four-hosts-app/backend/utils/domain_categorizer.py:76), a single categorizer used across modules to replace bespoke category heuristics (Note F).
- Integrated into Search API manager fetch phase when enriching results:
  - [categorize usage when filling source_category](four-hosts-app/backend/services/search_apis.py:2367)

Impact:
- Consolidates category logic previously split between providers and credibility.

3) Paradigm result filter/rank quadruplication removal
- Introduced [StrategyFilterRankMixin](four-hosts-app/backend/services/paradigm_search.py:101) that implements a unified filter_and_rank_results once; each concrete strategy now only provides _calculate_score.
- Refactors per-paradigm classes to leverage the mixin:
  - [DoloresSearchStrategy class](four-hosts-app/backend/services/paradigm_search.py:128) (now uses mixin)
  - [TeddySearchStrategy class](four-hosts-app/backend/services/paradigm_search.py:356) (now uses mixin)
  - [BernardSearchStrategy class, threshold 0.4 preserved](four-hosts-app/backend/services/paradigm_search.py:567)
  - [MaeveSearchStrategy class](four-hosts-app/backend/services/paradigm_search.py:797) (pending full replacement of the local method; see “Outstanding”)

Impact:
- Removes 4x duplicated filter/rank bodies noted in Note 1 (targeting ~64 lines saved). Threshold variation preserved.

4) Per-domain circuit breaker for fetcher (dup removal)
- Reworked RespectfulFetcher to rely on the global breaker manager and per-domain breaker keys:
  - [RespectfulFetcher.fetch uses circuit_manager with per-domain key](four-hosts-app/backend/services/search_apis.py:972)
  - [RespectfulFetcher.fetch_with_meta uses circuit_manager](four-hosts-app/backend/services/search_apis.py:1009)

Impact:
- Addresses Note A by removing ad-hoc breaker logic at fetch-time and focusing on a single CB implementation surface.

5) Dedup layer rationalization (provider-level 3-gram removal)
- Removed the provider-level 3-gram Jaccard dedup pass in SearchAPIManager; URL dedup remains early, with final dedup left to orchestrator’s result-level component (Note G).
  - See last phase of [SearchAPIManager.search_all_parallel() return](four-hosts-app/backend/services/search_apis.py:2423) where provider-level Jaccard stage was eliminated and relevance filtering is applied directly to combined results.

Impact:
- Keeps dedup layering as intended in Notes (planner-level query dedup + orchestrator-level result dedup).

Partially completed

6) Text sanitation normalization
- Created/confirmed shared sanitize functions [sanitize_text](four-hosts-app/backend/utils/text_sanitize.py:37), used by result normalization paths.
- Integrated in Search API result normalization for titles/snippets [normalize_result_text_fields](four-hosts-app/backend/services/search_apis.py:189) to call sanitize_text (applied).
- Outstanding: replace local _strip_tags body entirely; currently still references _html in its legacy path, causing a flake8 error (see “Outstanding”).

7) Credibility preferences consolidation
- Modified credibility to use the central [PREFERRED_SOURCES](four-hosts-app/backend/models/paradigms_sources.py:1). The block was replaced, and the module imports categorizer (A/F/K).
- Needs a sweep to ensure long-line flake8 fixes and no dead imports.

Outstanding work (by Notes A–Q and Quick checklist)

A) Circuit breaker duplication
- Done for RespectfulFetcher. Outstanding:
  - Remove or fully stop using local [CircuitBreaker class] still present in [search_apis.py](four-hosts-app/backend/services/search_apis.py:884) (now redundant).
  - Ensure all Search API “search()” entrypoints already use [with_circuit_breaker(...)](four-hosts-app/backend/services/search_apis.py:1335), which they do for major providers.

B) Rate limiting and pacing overlap
- RespectfulFetcher domain pacing remains (robots/ethics). Outstanding:
  - Prefer ClientRateLimiter (already used in BaseSearchAPI) for outbound smoothing; validate if any ad-hoc pacing in fetcher can be delegated or minimized per policy.

C) Text sanitation/HTML stripping duplicated
- Outstanding in [search_apis.py](four-hosts-app/backend/services/search_apis.py:161): _strip_tags still uses _html and legacy regex path; replace implementation body fully with sanitize_text and delete _html dependency.

D) Result normalization implemented twice in orchestrator
- Not started. Create services/result_normalizer.py with normalize(result)->dict, use at both call sites (research_orchestrator lines around 866 and 1395 noted in the brief). The current repo points to some normalizers in [llm_client.normalize_responses_payload()](four-hosts-app/backend/services/llm_client.py:1123) which may be reusable.

E) Date parsing/ISO handling scattered
- Not started. Centralize into utils/date_utils with ensure_datetime and iso formatting, refactor [safe_parse_date](four-hosts-app/backend/services/search_apis.py:93) and orchestrator serializers.

F) Domain/source categorization duplicated
- Shared categorizer done: [categorize()](four-hosts-app/backend/utils/domain_categorizer.py:76). Outstanding:
  - Remove orphaned provider utility [_derive_source_category](four-hosts-app/backend/services/search_apis.py:459) (now redundant).
  - Ensure credibility path switches to categorizer at all inference points (primary code modified; confirm callsites).

G) Result-level dedup centralization
- Provider 3-gram pass removed. Ensure orchestrator continues to use the canonical ResultDeduplicator and that only URL dedup happens early in provider manager (validated at [_process_results](four-hosts-app/backend/services/search_apis.py:2136)).

H) URL parsing/normalization overlaps
- Not started. Extract utils/url_utils with normalize_url / is_valid_url / doi helpers and refactor [URLNormalizer](four-hosts-app/backend/services/search_apis.py:864) and security validators to it.

I) fetch/parsing functions duplicated
- Not started. Collapse [fetch_and_parse_url](four-hosts-app/backend/services/search_apis.py:660) and [fetch_and_parse_url_with_meta](four-hosts-app/backend/services/search_apis.py:768) behind a single core with with_meta flag.

J) Retry/backoff policy fragmentation
- Partially consistent via _rate_limit_backoff; still decentralized overall. Extract utils/retry for configuration and ensure both orchestrator and search layers derive from same knobs.

K) Preferred-source lists repeated
- Done via [PREFERRED_SOURCES](four-hosts-app/backend/models/paradigms_sources.py:1) and refactors in [paradigm_search.py imports](four-hosts-app/backend/services/paradigm_search.py:18) and credibility (modified).

L) Metrics/telemetry overlap
- Not started. Unify Prometheus metrics names in monitoring class and bind via TelemetryPipeline.

M) Numeric vs iterable coercers
- Not started. Promote coercers into [utils/type_coercion.py](four-hosts-app/backend/utils/type_coercion.py:1) and use from telemetry/orchestrator.

N) Progress reporting interface
- Not started. Validate all call sites use the same surface (services/progress.py) and method names.

O) Caching wrappers: KV vs typed
- Not started. Rewire typed methods to call generic KV with a single TTL config source.

P, Q) Keep centralized tokenizer/stopwords and evidence reuse
- No regressions introduced. Ensure EvidenceBuilder continues to call the unified sanitizer after we complete C).

Immediate-actions status from Note 1

- ParadigmBase + template method: Partially addressed via [StrategyFilterRankMixin](four-hosts-app/backend/services/paradigm_search.py:101). A formal ParadigmBase could fold generate_search_queries patterns next.
- DatabaseOperations mixin for session management: Not started. DB session pattern still duplicated in [user_management.py](four-hosts-app/backend/services/user_management.py:42) and many methods.
- SearchAPIBase with standard error handling: Present and in use (class BaseSearchAPI).
- SessionManager for async HTTP lifecycle: BaseSearchAPI provides session lifecycle; duplicated async context patterns still exist elsewhere (e.g., credibility DomainAuthorityChecker).

Lint/quality status (must-fix before merge)

- search_apis.py:
  - Long lines and excessive blank lines flagged by flake8 (e.g., imports, doclines, regex lines).
  - Legacy [_strip_tags](four-hosts-app/backend/services/search_apis.py:161) still references _html; either reintroduce import html as _html or finalize delegate to sanitize_text and remove unused code.
  - Remove dead: local [_derive_source_category](four-hosts-app/backend/services/search_apis.py:459) and local [CircuitBreaker class](four-hosts-app/backend/services/search_apis.py:884).
  - Shorten long imports, reflow string constants, wrap calls.

- paradigm_search.py:
  - Long docstring lines were reflowed; duplication import removed. Current mixin compiles and removed previous SyntaxError. One more duplication removal pending:
    - Maeve.filter_and_rank_results body still present (lines ~956–969); replace with mixin via _calculate_score like others.

- credibility.py:
  - Long lines and comments exceed 79 chars; reflow needed.
  - Verify imports: categorizer and PREFERRED_SOURCES are used; remove any dead imports. Confirm all paradigm preference lists now sourced from PREFERRED_SOURCES.

What’s next (execution order)

1. Complete text sanitation normalization (Note C)
   - Replace _strip_tags implementation with sanitize_text, delete _html usages, remove dead code.
   - Quick pass to sanitize HTML/text in evidence_builder after fetch (ensure consistent usage).

2. Remove redundant search_apis utilities (Notes A/F)
   - Delete _derive_source_category and the local CircuitBreaker class; ensure all references replaced with categorize() and circuit_manager.

3. Finish mixin integration in paradigm_search
   - Swap Maeve.filter_and_rank_results body for _calculate_score method (as with others).

4. Database session consolidation (Note 3)
   - Introduce DatabaseOperations mixin with a single async get_session() context wrapper and refactor [user_management.py blocks](four-hosts-app/backend/services/user_management.py:42) repeated “async with get_db_context() as session” to the mixin helpers.
   - Target: ~19 duplication sites.

5. Export service consolidation (Note 7)
   - Introduce a small helper to DRY filename, ExportResult creation, and sanitize calls shared by PDF/JSON/CSV/Excel/Markdown in [export_service.py](four-hosts-app/backend/services/export_service.py:1).

6. Remove provider-layer redundancy
   - Merge fetch_and_parse_url* APIs into one function with with_meta flag (Note I).

7. Centralize retry/backoff config (Note J)
   - Create utils/retry and replace scattered knobs in search_apis and orchestrator.

8. Flake8 pass
   - Reflow long lines, compress imports, remove unused imports/variables, fix “expected 2 blank lines” signatures.
   - Confirm tests that reference circuit breakers still pass (e.g., [tests/test_security.py](four-hosts-app/backend/tests/test_security.py:359)).

Estimated reduction realized so far

- Paradigm filter/rank: ~64 lines removed/centralized via mixin.
- Preferred sources duplication: ~100+ lines avoided across strategies + credibility (exact savings depend on original blocks).
- Categorizer centralization: ~40 lines consolidated and multiple call sites aligned.
- Provider 3-gram dedup removal: ~40–60 lines eliminated.

Remaining high-yield reductions

- Database session mixin (user_management): ~50–60 lines.
- Export service helpers: ~60–100 lines.
- fetch_and_parse_url unification + delete local CB/category: ~80–120 lines.
- Retry/backoff centralization: ~40–80 lines.

Risks/compatibility

- Ensure tests relying on specific provider-level behavior still pass:
  - Rate limit tests ([tests/test_search_exa.py](four-hosts-app/backend/tests/test_search_exa.py:117)) expect cooldown on 429.
  - Circuit breaker tests ([tests/test_security.py](four-hosts-app/backend/tests/test_security.py:359)) look for breaker presence on provider .search; these remain decorated.

- Be mindful of orchestrator dedup expectations; URL dedup early + orchestrator ResultDeduplicator downstream must remain intact.

Summary

- Completed: central registry for preferred sources, shared domain categorizer, mixin-based consolidation of filter/rank for Dolores/Teddy/Bernard, provider 3-gram dedup removal, RespectfulFetcher moved to global circuit manager per-domain breakers.
- Partial: sanitize normalization, credibility integration to categorizer/registry, Maeve mixin conversion.
- Outstanding: delete legacy local CircuitBreaker and _derive_source_category, unify text stripping function, DB session mixin, export helpers, retry/date/url utilities, lint cleanup.


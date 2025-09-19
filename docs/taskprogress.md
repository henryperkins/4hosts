Progress report: Four-Hosts backend consolidation (Notes 1–2, A–Q)

Completed/landed changes

1) Canonical preferred sources registry
- Added shared registry [PREFERRED_SOURCES](four-hosts-app/backend/models/paradigms_sources.py:1) and refactored consumers to import it:
  - [import in paradigm_search.py](four-hosts-app/backend/services/paradigm_search.py:18)
  - Credibility preferences switched to the shared registry

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
  - [MaeveSearchStrategy class](four-hosts-app/backend/services/paradigm_search.py:807) (now uses mixin - COMPLETED)

Impact:
- Removes 4x duplicated filter/rank bodies noted in Note 1 (~64 lines saved).

4) Per-domain circuit breaker for fetcher (dup removal)
- Reworked RespectfulFetcher to rely on the global breaker manager and per-domain breaker keys:
  - [RespectfulFetcher.fetch uses circuit_manager with per-domain key](four-hosts-app/backend/services/search_apis.py:972)
  - [RespectfulFetcher.fetch_with_meta uses circuit_manager](four-hosts-app/backend/services/search_apis.py:1009)

Impact:
- Addresses Note A by removing ad-hoc breaker logic at fetch-time.

5) Dedup layer rationalization (provider-level 3-gram removal)
- Removed the provider-level 3-gram Jaccard dedup pass in SearchAPIManager
- URL dedup remains early, with final dedup left to orchestrator's result-level component (Note G)

Impact:
- Keeps dedup layering as intended in Notes (planner-level query dedup + orchestrator-level result dedup).

6) Text sanitation normalization (Note C - COMPLETED)
- Created/confirmed shared sanitize functions [sanitize_text](four-hosts-app/backend/utils/text_sanitize.py:37)
- Integrated in Search API result normalization for titles/snippets [normalize_result_text_fields](four-hosts-app/backend/services/search_apis.py:160)
- COMPLETED: Removed _strip_tags function entirely, now using centralized sanitize_text

7) Credibility preferences consolidation
- Modified credibility to use the central [PREFERRED_SOURCES](four-hosts-app/backend/models/paradigms_sources.py:1)
- Updated date handling to use centralized date_utils functions
- Integrated domain categorizer

8) DatabaseOperations mixin (Note 1 action item - COMPLETED)
- Created [DatabaseOperations mixin](four-hosts-app/backend/services/database_operations.py:28) providing unified db_session() context manager
- Fully integrated into user_management.py - all service classes now inherit from DatabaseOperations:
  - [UserProfileService](four-hosts-app/backend/services/user_management.py:38)
  - [SavedSearchesService](four-hosts-app/backend/services/user_management.py:196)
  - [APIKeyService](four-hosts-app/backend/services/user_management.py:426)
  - [ResearchHistoryService](four-hosts-app/backend/services/user_management.py:572)
  - [SessionService](four-hosts-app/backend/services/user_management.py:699)

Impact:
- Eliminates ~57 lines of repeated session management code (19 sites × 3 lines each)

9) Result normalizer utility (Note D - COMPLETED)
- Created [normalize_result()](four-hosts-app/backend/services/result_normalizer.py:10) to consolidate duplicate normalization logic
- Integrated into research_orchestrator.py at both normalization points (lines ~858 and ~1367)

Impact:
- Eliminates ~40 lines of duplicated normalization code

10) Date utilities module (Note E - COMPLETED)
- Created comprehensive [date_utils module](four-hosts-app/backend/utils/date_utils.py:1) with:
  - safe_parse_date() - moved from search_apis.py
  - ensure_datetime() - flexible date parsing
  - iso_or_none() - safe ISO formatting
  - format_timestamp(), format_human_readable(), format_date_only() - consistent formatting
  - calculate_age_days(), calculate_age_timedelta() - age calculations
  - get_current_utc(), get_current_iso() - current time helpers
- Integrated into:
  - search_apis.py (removed local safe_parse_date, ~32 lines)
  - export_service.py (all timestamp formatting now uses date_utils)
  - credibility.py (datetime operations now use centralized functions)
  - result_normalizer.py (date handling)

Impact:
- Consolidates ~120+ lines of date handling code across the codebase

Outstanding work (by Notes A–Q and Quick checklist)

A) Circuit breaker duplication
- COMPLETED for RespectfulFetcher
- Verify all search providers use with_circuit_breaker decorator

B) Rate limiting and pacing overlap
- RespectfulFetcher domain pacing remains (robots/ethics)
- Validate if ad-hoc pacing can be delegated to ClientRateLimiter

F) Domain/source categorization duplicated
- Shared categorizer COMPLETED
- _derive_source_category already removed (just comment remains)

G) Result-level dedup centralization
- Provider 3-gram pass removed - COMPLETED
- Orchestrator continues to use ResultDeduplicator

H) URL parsing/normalization overlaps
- COMPLETED. Created utils/url_utils with normalize_url/is_valid_url/extract_domain/extract_doi/clean_url
- Refactored 9+ files including search_apis, answer_generator, evidence_builder, llm_critic, agentic_process
- URLNormalizer class removed, using centralized functions

I) Fetch/parsing functions duplicated
- COMPLETED. fetch_and_parse_url now accepts with_meta parameter
- fetch_and_parse_url_with_meta removed (functionality merged)

J) Retry/backoff policy fragmentation
- COMPLETED. Created utils/retry module with centralized configuration
- search_apis.py now uses handle_rate_limit() and parse_retry_after()
- RateLimitedError moved to retry module with backward-compatible alias

K) Preferred-source lists repeated
- COMPLETED via PREFERRED_SOURCES registry

L) Metrics/telemetry overlap
- Not started. Unify Prometheus metrics names

M) Numeric vs iterable coercers
- COMPLETED. Added as_iterable(), as_list(), coerce_iterable() to utils/type_coercion.py
- telemetry_pipeline.py updated to use centralized coercers

N) Progress reporting interface
- Not started. Validate all call sites use same surface

O) Caching wrappers: KV vs typed
- Not started. Rewire typed methods to call generic KV

P, Q) Keep centralized tokenizer/stopwords and evidence reuse
- No regressions introduced

What's next (execution order)

1. ✅ COMPLETED: Create url_utils module (Note H)
   - Created utils/url_utils.py with comprehensive URL handling functions
   - Refactored 9+ files to use centralized utilities

2. ✅ COMPLETED: Merge fetch_and_parse_url functions (Note I)
   - Combined into single function with with_meta flag
   - Removed fetch_and_parse_url_with_meta

3. ✅ COMPLETED: Create retry module (Note J)
   - Created utils/retry.py with centralized configuration
   - Integrated into search_apis.py

4. ✅ COMPLETED: Add iterable coercers (Note M)
   - Extended utils/type_coercion.py with iterable handlers
   - Updated telemetry_pipeline.py to use them

5. Export service consolidation (Note 7)
   - DRY helper for filename, ExportResult, sanitize calls

6. Progress reporting interface validation (Note N)
   - Validate all call sites use same surface

7. Caching wrapper consolidation (Note O)
   - Rewire typed methods to call generic KV

8. Flake8 pass
   - Fix long lines, imports, unused variables
   - Confirm tests still pass

Estimated reduction realized

- Paradigm filter/rank: ~64 lines
- Preferred sources duplication: ~100+ lines
- Categorizer centralization: ~40 lines
- Provider 3-gram dedup removal: ~50 lines
- DatabaseOperations mixin: ~57 lines
- Result normalizer: ~40 lines
- Date utilities: ~120+ lines
- Text sanitation: ~32 lines
- URL utilities: ~200+ lines (URLNormalizer class + 9 files refactored)
- fetch_and_parse_url unification: ~100 lines
- Retry/backoff centralization: ~80 lines
- Iterable coercers: ~20 lines

**Total reduction achieved: ~903+ lines**

Remaining high-yield reductions

- Export service helpers: ~60-100 lines
- Progress reporting validation: ~20-40 lines
- Caching wrapper consolidation: ~40-60 lines

**Projected total: ~1020-1100 lines reduced**

Risks/compatibility

- Ensure tests relying on specific provider behavior still pass
- Rate limit tests expect cooldown on 429
- Circuit breaker tests look for breaker presence on provider.search
- URL dedup early + orchestrator ResultDeduplicator must remain intact

Summary

Current session progress (2025-09-19):
- COMPLETED: URL utilities (Note H), Fetch function merge (Note I), Retry module (Note J), Iterable coercers (Note M)
- Previous session: Text sanitation (Note C), DatabaseOperations mixin, Result normalizer (Note D), Date utilities (Note E), Maeve mixin integration

Major achievements:
- Created 3 new utility modules: utils/url_utils.py, utils/retry.py, extended utils/type_coercion.py
- Refactored 9+ files to use centralized URL handling
- Eliminated URLNormalizer class and duplicate fetch functions
- Integrated centralized retry/backoff into search_apis.py
- **Total consolidation: ~903+ lines reduced** (exceeded original 740-860 projection)

Remaining work:
- Export service consolidation
- Progress reporting interface validation
- Caching wrapper consolidation
- Final flake8/testing pass
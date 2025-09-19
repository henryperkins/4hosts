# Task Prompt

## Objectives

* Consolidate and remove duplicated/overlapping patterns exactly as identified in Notes 1–2, without changing or removing any context except explicit duplicate mentions.
* Apply the specific refactoring/consolidation actions specified in the notes (no additional suggestions).
* Target the quantified line-reduction potential stated in the notes.

## Scope

* Backend/services: `paradigm_search.py`, `answer_generator.py`, `search_apis.py`, `research_orchestrator.py`, `query_planning/*`, `rate_limiter.py`, `credibility.py`, `telemetry_pipeline.py`, `monitoring.py`, `progress.py`, `export_service.py`, `evidence_builder.py`
* Backend/utils: `circuit_breaker.py`, `text_sanitize.py`, `type_coercion.py`, `security.py`
* DB layer: `user_management.py`
* Any helpers referenced in sections A–Q below

---

## Critical Duplication Findings (from Note 1)

### 1) Paradigm Implementation Pattern (Highest Impact)

**Files:** `answer_generator.py`, `paradigm_search.py`
**Total Duplication:** \~800+ lines

**Evidence — Constructor Pattern (4x duplication):**

```PYTHON
# Lines: 934-936, 1259-1261, 1791-1793, 2223-2225
def __init__(self):
    super().__init__("paradigm_name")
```

**Section Structure Pattern (4x duplication, 22–30 lines each):**

* Dolores: lines 937–959
* Bernard: lines 1262–1294
* Maeve: lines 1794–1820
* Teddy: lines 2226–2248

**Filter and Rank Pattern (4x duplication, 16 lines each):**

```PYTHON
# Lines: 307-322, 539-552, 787-800, 1020-1033
async def filter_and_rank_results(self, results, context):
    scored_results = []
    for result in results:
        score = await self._calculate_{paradigm}_score(result, context)
        if score > 0.3:  # Minor threshold variations
            result.credibility_score = score
            scored_results.append((result, score))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [result for result, score in scored_results]
```

### 2) Search API Error Handling (Critical Pattern)

**File:** `search_apis.py`
**Occurrences:** 9 identical patterns
**Lines:** 1342–1343, 1408–1410, 1508–1510, 1575–1577, 1647–1649, 1780–1782, 1837–1839

```PYTHON
if r.status != 200:
    return []
```

**Impact:** 27 lines of identical error handling

### 3) Database Session Management

**File:** `user_management.py`
**Occurrences:** 19 identical patterns
**Lines:** 42, 95, 111, 134, 159, 206, 266, 288, 352, 430, 471, 515, 538, 585, 639, 709, 746, 779, 795

```PYTHON
async with get_db_context() as session:
    # Query pattern follows
```

### 4) Async Context Manager Implementation

**Files / locations (exact duplication):**

* `mcp_integration.py:70–76`
* `credibility.py:132–138`
* `search_apis.py:1242–1249`
* `search_apis.py:1916–1930`

```PYTHON
async def __aenter__(self):
    self.session = aiohttp.ClientSession()  # Minor variations in params
    return self
async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self.session:
        await self.session.close()
```

### 5) Rate Limiter Redis vs Memory Pattern

**File:** `rate_limiter.py`
**Pattern Repetition:** 3x (lines 153–158, 169–176, 208–236)

```PYTHON
if self.redis_client:
    # Redis implementation
    key = f"pattern:{identifier}"
    # Redis operations
else:
    # In-memory implementation
    # Dictionary operations
```

### 6) Query Modifier Lists

**File:** `paradigm_search.py`
**Duplication:** 4x similar list definitions

* Dolores: lines 118–133 (16 items)
* Teddy: lines 370–386 (17 items)
* Bernard: lines 603–619 (16 items)
* Maeve: lines 854–869 (15 items)

### 7) Export Format Implementations

**File:** `export_service.py`
**Pattern:** 5 nearly identical export methods (PDF, JSON, CSV, Excel, Markdown)

* Similar filename generation (\~5 lines each)
* Similar `ExportResult` creation (\~8 lines each)
* Similar sanitization calls (\~10 lines each)

#### Quantified Impact

| Pattern                   | Files | Instances | Lines per Instance |      Total Lines | Severity |
| ------------------------- | ----- | --------: | -----------------: | ---------------: | -------- |
| Paradigm Implementation   | 2     |       4x4 |             50–100 |             800+ | Critical |
| Search API Error Handling | 1     |         9 |                  3 |               27 | High     |
| Database Sessions         | 1     |        19 |                  3 |               57 | High     |
| Async Context Manager     | 4     |         4 |                  7 |               28 | Medium   |
| Redis vs Memory           | 1     |         3 |                 20 |               60 | High     |
| Query Modifiers           | 1     |         4 |                 16 |               64 | Medium   |
| Export Patterns           | 1     |         5 |                 23 |              115 | High     |
| **Total**                 |       |           |                    | **1,151+ lines** |          |

**Most Problematic Patterns**

1. The Paradigm Quadruplication — minor variations in string literals (paradigm names), thresholds (0.3 vs 0.4), and list contents (query modifiers).
2. The Database Session Anti-Pattern — 19 identical `async with get_db_context()` blocks in `user_management.py` (similar patterns elsewhere).
3. The Copy-Paste Search Provider — providers replicate error handling, result parsing, rate limiting, and session management.

**Refactoring Recommendations (from Note 1)**

* Immediate Actions:

  * Create `ParadigmBase` with template methods for common operations
  * Extract `DatabaseOperations` mixin for session management
  * Implement `SearchAPIBase` with standard error handling and result parsing
  * Create `SessionManager` for async HTTP client lifecycle
* Code Reduction Potential:

  * Conservative estimate: 30% reduction (345+ lines)
  * Aggressive refactoring: 50% reduction (575+ lines)
  * With design patterns: 60% reduction (690+ lines)
* Priority Refactoring Targets:

  * `paradigm_search.py`: Extract base class (save \~300 lines)
  * `answer_generator.py`: Template method pattern (save \~200 lines)
  * `search_apis.py`: Consolidate error handling (save \~100 lines)
  * `user_management.py`: Database operation utilities (save \~50 lines)

---

## Refined Duplication & Overlap with Consolidation Actions (from Note 2)

**A) Circuit breaker duplication**

* Two implementations: shared `utils/circuit_breaker.py:45` (and `with_circuit_breaker():284`) vs `services/search_apis.py:882`
* Usage: decorators on providers (`BraveSearchAPI.search:1330`, `GoogleCustomSearchAPI.search:1387`, `ExaSearchAPI.search:1482`); fetch path `RespectfulFetcher:924`
* **Unify:** Replace local `class CircuitBreaker` in `search_apis.py:882` with shared class. For per-domain behavior, construct per-domain breaker names (e.g., `f"fetch:{domain}"`) via `circuit_manager.get_or_create():250`.

**B) Rate limiting and pacing overlap**

* Outbound smoother: `ClientRateLimiter (services/rate_limiter.py:644)`
* Ad-hoc pacing: `RespectfulFetcher._pace_domain (search_apis.py:958)`
* **Consolidate:** Prefer `ClientRateLimiter` for smoothing; keep only domain pacing if robots/ethics require.

**C) Text sanitation/HTML stripping duplicated**

* Local `_strip_tags (search_apis.py:159)` vs shared utils `strip_html:17`, `collapse_ws:28`, `sanitize_text:37`
* **Normalize:** Replace calls to `_strip_tags` with `strip_html` (+ `collapse_ws`); remove the local duplicate.

**D) Result normalization implemented twice in orchestrator**

* Response payload at `research_orchestrator.py:866`
* Synthesis inputs at `research_orchestrator.py:1395`
* **Consolidate:** Extract single helper `services/result_normalizer.normalize(result)->dict` used by both.

**E) Date parsing/ISO handling scattered**

* Parser: `safe_parse_date (search_apis.py:91)`
* ISO serializations: `research_orchestrator.py:859` and `:1385`
* **Consolidate:** Create `utils/date_utils` with `ensure_datetime(any)->datetime|None` and `iso_or_none(datetime)->str|None`; reuse `safe_parse_date` for string→datetime.

**F) Domain/source categorization duplicated**

* Provider `_derive_source_category (search_apis.py:457)`
* Credibility fallback `_infer_category (credibility.py:952)`
* **Consolidate:** `utils/domain_categorizer.categorize(domain, content_type|meta)->enum` used by both.

**G) Result-level deduplication in multiple paths**

* Provider URL-only dedup: `SearchAPIManager._process_results:2130`
* Jaccard 3-gram dedup: `search_all_parallel:2417`
* Orchestrator advanced dedup: `ResultDeduplicator (services/query_planning/result_deduplicator.py:24)` used at `research_orchestrator.py:2069`
* Query-candidate dedup (separate): `is_duplicate (query_planning/cleaner.py:20)` via planner `planner.py:176`
* **Consolidate:** Keep query-candidate dedup in planner and result-level dedup in `ResultDeduplicator`. Remove provider-layer 3-gram pass in `search_all_parallel:2417`. Keep only URL dedup early.

**H) URL parsing/normalization overlaps**

* Provider utility: `URLNormalizer (search_apis.py:862)`
* Security validators: `InputSanitizer.sanitize_url (utils/security.py:70)`, `validate_and_sanitize_url (utils/security.py:244)`
* **Consolidate:** Extract `utils/url_utils` with `normalize_url/is_valid_url/doi`; reuse security checks from `validate_and_sanitize_url`.

**I) Fetch/parsing functions duplicated**

* `fetch_and_parse_url:658` and `fetch_and_parse_url_with_meta:766`
* **Consolidate:** Collapse into one core fetch (e.g., `fetch_text(url, with_meta: bool)->(text, meta)`), with metadata extraction in one place.

**J) Retry/backoff policy fragmentation**

* HTTP-layer 429/jitter: `_rate_limit_backoff (search_apis.py:262)`
* Orchestrator policy: `class RetryPolicy (planning_utils.py:33)` used in `_perform_search (research_orchestrator.py:2380)`
* **Consolidate:** Standardize env knobs in a single utility; ensure both paths derive from same config surface (`utils/retry`).

**K) Preferred-source lists repeated across paradigms and credibility**

* Strategies: Dolores `paradigm_search.py:142`; Teddy `:391`; Bernard `:621`; Maeve `:870`
* Credibility: `SourceReputationDatabase.paradigm_preferences (credibility.py:680)`
* **Consolidate:** Canonical registry `models/paradigms_sources.py` referenced by both strategies and credibility.

**L) Metrics/telemetry overlap**

* Prometheus registry: `class PrometheusMetrics (monitoring.py:82)`
* Telemetry mapping: `_record_prometheus_metrics (telemetry_pipeline.py:91)`
* **Consolidate:** Treat `PrometheusMetrics` as the single source of metric names/labels; keep `TelemetryPipeline` bound via `bind_prometheus:37`.

**M) Numeric vs iterable coercers**

* Numeric: `as_int (utils/type_coercion.py:6)`, `as_float:28`
* Telemetry locals: `_coerce_iterable:147`, `_iter_costs:155` (telemetry\_pipeline)
* **Consolidate:** Add iterable/dict-safe coercers in `utils/type_coercion` and import across telemetry/orchestrator.

**N) Progress reporting interface**

* Progress façade: `_NoOpProgress (services/progress.py:14)` with methods `update_progress:15`, `report_search_started:24`, `report_search_completed:27`, `report_source_found:30`
* **Action:** Ensure all call sites route through `progress (services/progress.py:49)` or adhere to the same method names.

**O) Caching wrappers: KV vs typed**

* Typed: `get_source_credibility:270`, `set_source_credibility:290`, `get_search_results:116`, `set_search_results:157`
* Generic KV: `get_kv:236`, `set_kv:257`
* **Ensure:** Typed convenience methods internally use KV and a single TTL source.

**P) Tokenization/stopwords centralization is correct; avoid new duplicates**

* Shared: `STOP_WORDS (services/text_utils.py:30)`, `tokenize:42`
* Consumers (e.g., `QueryOptimizer (services/query_planning/optimizer.py:15)`) already import those.

**Q) Evidence/content extraction reuse is good**

* Evidence builder reuses fetcher: `evidence_builder.py:45`
* **Ensure:** any HTML strip/normalize step in evidence builder uses the unified `utils/text_sanitize` once C) is done.

**Quick consolidation checklist (actionable)**

1. Remove `search_apis`-local breaker; use `utils/circuit_breaker` with per-domain breakers.
2. Replace `_strip_tags` with `utils/text_sanitize` functions; remove duplicate.
3. Merge `fetch_and_parse_url*` into one function with a `with_meta` flag.
4. Make `ResultDeduplicator` the single advanced result-dedup stage; drop `SearchAPIManager`’s 3-gram pass.
5. Extract `utils/domain_categorizer` and `utils/date_utils`; refactor orchestrator/search\_apis/credibility to use them.
6. Centralize `preferred_sources` in a shared registry used by both `paradigm_search` and `credibility`.
7. Standardize retry/backoff config in a `utils/retry` module (jitter/base/max; read env once).
8. Move telemetry iterable coercers into `utils/type_coercion`; import from there.
9. Keep `ClientRateLimiter` for outbound smoothing; retain only one per-domain pacing mechanism.
10. Keep a single planner import path: prefer `services/query_planning`; remove thin wrapper exports to avoid confusion.

**Notable non-dup overlaps worth keeping as-is**

* Planner-level query dedup vs result-level dedup serve different purposes; keep both layers but with one implementation per layer (planner vs results).
* Progress façade vs direct callback signatures are intentionally aligned; just ensure all call sites remain signature-compatible.

---

## Requirements to Complete (directly from the notes)

* Implement the “Immediate Actions” (ParadigmBase, DatabaseOperations mixin, SearchAPIBase, SessionManager).
* Execute all A–Q consolidation actions and the “Quick consolidation checklist.”
* Address the “Priority Refactoring Targets” and aim for the “Code Reduction Potential.”
* Preserve all cited line references and behaviors while removing the explicitly duplicated items listed.
* Maintain the “Notable non-dup overlaps worth keeping as-is” exactly as stated.

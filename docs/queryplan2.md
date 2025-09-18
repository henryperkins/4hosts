Short version: centralizing query planning into `search/query_planner/` touches a handful of places hard (where we currently craft queries) and leaves the rest mostly chill. Below is a precise, file-by-file ripple map for everything else I can see in this repo, plus the exact changes you’ll want.

# Impact map (everything outside the 7 core files)

## High / Medium impact

### agentic\_process.py — **Medium**

* **Now a stage**: Its `propose_queries_enriched(...)` becomes the engine behind `AgenticFollowupsStage`.
* **Keep**: `evaluate_coverage_from_sources`, `summarize_domain_gaps`, and both propose functions.
* **Change**:

  * Guarantee the follow-ups return `List[str]` (no dicts), since the stage wraps into `QueryCandidate`.
  * Add tiny docstrings noting it’s consumed by the planner stage.
  * Don’t call it directly from orchestrators—call `QueryPlanner.followups(...)`.

### llm\_query\_optimizer.py — **Medium**

* **Now a stage**: `propose_semantic_variations(...)` is called only by `LLMVariationsStage`.
* **Change**:

  * Remove/ignore in-module env checks if you prefer; the planner config (`enable_llm`, `per_stage_caps`) becomes the single knob. Or keep the guard and let the stage read it—just don’t have two conflicting switches.
  * Return type stays `List[str]`, which the stage will wrap.

### paradigm\_search.py — **Medium**

* **Now a stage**: `generate_search_queries(SearchContext)` fuels `ParadigmStage`.
* **Keep**: the rich per-paradigm query patterns.
* **Change**:

  * Don’t call these generators directly from anywhere else (some places were importing for ranking; ranking can keep using the same module, see note under *result ranking* below).
  * Ensure each generated item includes `{"query": str, "type": str, "weight": float, "source_filter": Optional[str]}`; the stage maps that onto `QueryCandidate`.

### context\_engineering.py — **Medium**

* **Current**: `OptimizeLayer` instantiates `QueryOptimizer` and calls `generate_query_variations(...)`, `optimize_query(...)`, and `get_key_terms(...)`.
* **Change (PR1)**:
  * Leave `OptimizeLayer` as-is to keep existing tests stable and behavior unchanged.
* **Optional (PR2)**:
  * If the layer needs “preview” variations for UI, import `QueryPlanner` and call `initial_plan(...)` with a low cap (e.g., 5) and show the top-N `QueryCandidate.query`.
  * For key terms, continue using `QueryOptimizer.get_key_terms(...)` unless you add an equivalent `get_key_terms(...)` helper to `classification_engine.QueryAnalyzer` for a single point of truth.

### search\_apis.py — **Medium** (even though it’s one of the 7, it affects other callers)

* **Now consumer, not producer**: `BaseSearchAPI.search_with_variations(...)` must accept a prebuilt `planned: List[QueryCandidate] | None`.
* **Change**:

  * If `planned` is provided, **skip** `self.qopt.generate_query_variations` and just run the given candidates. Keep old behavior for backwards compatibility.
  * Emit a `query_variant` string like `"{stage}:{label}"` on each result (helps downstream analytics).
  * Add `SearchAPIManager.search_with_plan(planned)` to pass planned candidates down to each provider by calling `api.search_with_variations(query, cfg, planned=planned)`.

### research\_store.py / research\_persistence.py — **Low → Medium** (observability)

* **Change (optional but recommended)**:

  * Add a compact **plan trace** field (e.g., `planned_queries: [{"q":"...", "stage":"...", "label":"..."}]`) in the stored research run for reproducibility.
  * No API changes required; just store what orchestrator gives you.

## Low impact

### evidence\_builder.py — **Low**

* Uses `fetch_and_parse_url` only; not a planner consumer. No change.

### answer\_generator.py — **Low**

* Synthesis doesn’t depend on the query strings. No change required.
* **Nice-to-have**: if you display per-source `query_variant`, show `stage:label` to help humans interpret why a source appeared.

### result\_adapter.py — **Low**

* Doesn’t currently expose `query_variant`.
* **Quick win**: add a property `query_variant` that returns `result.get('query_variant','')` (adapter already has the pattern for optional fields).

### deep\_research\_service.py — **Low**

* Builds prompts and sometimes uses paradigm search **config**, not query generation. No change.
* **Optional**: if you trigger any “regular web search” as part of deep research outside the orchestrator, route those through `QueryPlanner` for consistency.

### classification\_engine.py — **Low**

* Provides `QueryAnalyzer` for feature extraction; the planner should source key terms via `QueryOptimizer.get_key_terms(...)` today.
* Optional: add a small `get_key_terms(...)` helper to `QueryAnalyzer` in a later PR if you prefer a single point of truth for term extraction.

### background\_llm.py, llm\_client.py, openai\_responses\_client.py, llm\_critic.py — **None**

* Pure LLM I/O layers; planner change is orthogonal.

### action\_items.py — **None**

* Generates LLM action items from evidence; no planner usage.

### mesh\_network.py — **None**

* Paradigm/host mapping only; no query generation.

### ml\_pipeline.py — **None**

* Training/feedback; no query generation.

### progress.py, websocket\_service.py — **None**

* UI updates; unaffected.

### credibility.py, rate\_limiter.py — **None**

* Scoring and throttling; untouched.

## “Uses but unchanged” sanity list

* **paradigm\_search.py** stays the canonical home for per-paradigm *query patterns* and separately for *result ranking*. Keep the ranking APIs you already use (e.g., `filter_and_rank_results(...)`), because the planner only normalizes **inputs**; it doesn’t touch result ranking.

---

# Cross-cutting changes (one-time)

1. **New types**
   Add `QueryCandidate` and `PlannerConfig` in `search/query_planner/types.py`. Other files shouldn’t construct these directly except orchestrators/tests, but they’ll read `query_variant` from results.

2. **Orchestrator wiring**
   In `research_orchestrator.py` (covered previously), construct the planner and call `SearchAPIManager.search_with_plan(planned)` to pass candidates to providers via the new `planned` parameter, and use `planner.followups(...)` when agentic coverage suggests gaps. This is the only call-site you need to touch to propagate the new plan everywhere.

3. **Env → config**
   Move any query-variation envs (e.g., `SEARCH_QUERY_VARIATIONS_LIMIT`, `ENABLE_QUERY_LLM`) into a single place that builds `PlannerConfig`. Everyone else reads from the config object via the orchestrator—not directly from `os.environ`.

4. **Telemetry**
   Emit a “plan trace” log/metric (stage counts, dedup rate, final `max_candidates`) once per run. If you already ship metrics in `research_orchestrator.py`, tack it on there.

---

# Concrete diff hints per file (ready to implement)

* **search\_apis.py**

  * Signature: `async def search_with_variations(self, query: str, cfg: SearchConfig, planned: list[QueryCandidate] | None = None) -> List[SearchResult]:`
  * First line of the body:

    ```python
    if planned:
        variants = planned
    else:
        # current QueryOptimizer path unchanged
    ```
  * When assembling each result, add: `result["query_variant"] = f"{cand.stage}:{cand.label}"`.

* **research\_orchestrator.py**

  * Construct once: `planner = QueryPlanner(PlannerConfig(...))`.
  * Replace ad-hoc rule-based/paradigm/LLM calls with:

    ```python
    planned = await planner.initial_plan(seed_query=query, paradigm=paradigm)
    results  = await search_manager.search_with_plan(planned)  # thin wrapper that fans out to providers
    ```
  * After coverage analysis:

    ```python
    followups = await planner.followups(seed_query=query, paradigm=paradigm, coverage=coverage, missing_terms=missing, domain_gaps=gaps)
    if followups:
        more = await search_manager.search_with_plan(followups)
        results.extend(more)
    ```

* **context\_engineering.py**

  * In `OptimizeLayer.__init__`, drop `self.optimizer = QueryOptimizer()`.
  * For key terms: use `QueryAnalyzer().get_key_terms(text)` (already imported in this module).
  * If the layer needs variations to show the user: import `QueryPlanner` and call `initial_plan(...)` with a small cap (e.g., 5). Store only the `.query` strings in the layer output.

* **agentic\_process.py**

  * No signature changes needed if you return `List[str]` already (you do).
  * Add a small note in docstrings that this is consumed by `AgenticFollowupsStage`.

* **llm\_query\_optimizer.py**

  * Keep the function as is; remove or leave the env guard. The stage will respect `PlannerConfig.enable_llm`, so the guard can be redundant.

* **paradigm\_search.py**

  * Ensure `generate_search_queries(...)` always returns items with the fields the stage expects (you already do).
  * Do **not** remove result-ranking helpers—they’re used post-search, not in the planner.

* **result\_adapter.py**

  * Add:

    ```python
    @property
    def query_variant(self) -> str:
        return self._result.get('query_variant','') if self._is_dict else getattr(self._result, 'query_variant', '')
    ```

* **research\_store.py / research\_persistence.py**

  * (Optional) Accept and store `planned_queries` metadata from the orchestrator so we can reproduce a run.

---

# Testing you’ll need to update/add

* **Planner golden tests**: small snapshots per paradigm showing the top 5 candidates for a fixed seed → keeps stage order and dedup from drifting.
* **Provider passthrough test**: given `planned`, verify providers don’t call `QueryOptimizer` internally.
* **Orchestrator integration test**: verify agentic follow-ups use `planner.followups` only when coverage < threshold.
* **Adapter test**: `ResultAdapter.query_variant` returns the right `stage:label`.

---

# TL;DR

Only four “other” files are meaningfully touched (`agentic_process.py`, `llm_query_optimizer.py`, `paradigm_search.py`, `context_engineering.py`), plus a tiny adapter tweak. Everything else either keeps working or gains optional observability. The orchestrator is the single integration point; once it passes a `planned` list to providers, the entire stack snaps to the new, consistent plan—no more four cooks in four kitchens.

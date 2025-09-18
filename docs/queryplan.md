Let’s zoom in on the “four cooks” and turn them into one kitchen. I read the code in `search_apis.py` (QueryOptimizer + provider-side variation logic), `llm_query_optimizer.py`, `paradigm_search.py`, and `agentic_process.py`, plus how `research_orchestrator.py` actually wires things. Here’s what’s overlapping, what’s missing, and a concrete, drop-in Planner design that reuses your code with minimal surgery.

# What actually happens today (and where it overlaps)

* **Rule-based (QueryOptimizer, in `search_apis.py`):**

  * Protects quoted phrases, preserves “known entities,” removes noise terms, optional WordNet synonym expansion, adds paradigm terms (simple mapping), and emits a labeled dict of variations: `primary`, `domain_specific`, `exact_phrase` (≤6 tokens), `synonym`, `related`, `broad`, `question`.
  * Providers call this internally in `BaseSearchAPI.search_with_variations()` and pick top N by a hardcoded priority list + `SEARCH_QUERY_VARIATIONS_LIMIT`.
* **LLM-based (`llm_query_optimizer.py`):**

  * `propose_semantic_variations()` (guarded by `ENABLE_QUERY_LLM`) returns up to K semantically diverse variations (structured output). Not used by providers; also not called in `research_orchestrator.py`.
* **Paradigm flavored (`paradigm_search.py`):**

  * Four classes (`Dolores*`, `Teddy*`, `Bernard*`, `Maeve*`) generate paradigm-specific queries with `type`, `weight`, and `source_filter` (e.g., investigative patterns for Dolores, field modifiers for Bernard, etc.).
  * **Orchestrator currently uses these only for *ranking* results (`filter_and_rank_results`)**, not for generating the queries sent to providers. So this generator is parallel-but-unused for query formation.
* **Agentic loop (`agentic_process.py`):**

  * Coverage scoring against `key_themes`/`focus_areas`, domain-gap summary, and follow-up query proposals (combining missing targets with paradigm-enriched modifiers). Imported into `research_orchestrator.py`, but not actually invoked there to expand queries.

**Net effect:** Providers unilaterally generate variants (rule-based), paradigm generators produce richer variants that aren’t used to search, and agentic follow-ups aren’t looped in. The knobs live in three places with different data shapes (`dict[label->query]` vs `[{query, type, weight, source_filter}]` vs `List[str]`).

---

# Target: one **Query Planner** with composable stages

Create `search/query_planner/` with a tiny interface and four stage adapters around your existing code. The planner produces a single, deduplicated, ranked list of **QueryCandidates** with consistent metadata. Providers get handed the plan instead of inventing their own.

## 1) Types (single source of truth)

```python
# search/query_planner/types.py
from dataclasses import dataclass, field
from typing import Literal, Dict, List, Optional

StageName = Literal["rule_based", "llm", "paradigm", "agentic"]

@dataclass
class QueryCandidate:
    query: str
    stage: StageName
    label: str                      # e.g., "primary", "exact_phrase", "investigative_pattern"
    weight: float = 1.0
    source_filter: Optional[str] = None  # e.g., "academic", "investigative", etc.
    tags: Dict[str, str] = field(default_factory=dict)  # arbitrary hints

@dataclass
class PlannerConfig:
    max_candidates: int = 12
    enable_llm: bool = False
    stage_order: List[StageName] = field(default_factory=lambda: ["rule_based", "paradigm", "llm"])
    per_stage_caps: Dict[StageName, int] = field(default_factory=lambda: {"rule_based": 6, "paradigm": 6, "llm": 4, "agentic": 6})
    dedup_jaccard: float = 0.92     # near-duplicate collapse for queries
```

## 2) Stage interface + adapters (wrapping your existing code)

```python
# search/query_planner/base.py
from typing import List, Protocol
from .types import QueryCandidate, PlannerConfig

class QueryStage(Protocol):
    async def generate(self, *, seed_query: str, paradigm: str, key_terms: List[str] | None, cfg: PlannerConfig) -> List[QueryCandidate]:
        ...

# RuleBasedStage: wraps QueryOptimizer from search_apis.py
class RuleBasedStage:
    def __init__(self, qopt):
        self.qopt = qopt  # existing QueryOptimizer instance

    async def generate(self, *, seed_query, paradigm, key_terms, cfg):
        variations = self.qopt.generate_query_variations(seed_query, paradigm=paradigm)
        priority = ["primary","domain_specific","exact_phrase","synonym","related","broad","question"]
        out = []
        for label in priority:
            q = variations.get(label)
            if not q: continue
            # map label to a gentle weight curve
            w = {"primary":1.0,"domain_specific":0.95,"exact_phrase":0.9,"synonym":0.8,"related":0.7,"broad":0.6,"question":0.6}.get(label,0.5)
            out.append(QueryCandidate(query=q, stage="rule_based", label=label, weight=w))
        return out[:cfg.per_stage_caps["rule_based"]]

# LLMVariationsStage: wraps propose_semantic_variations()
class LLMVariationsStage:
    async def generate(self, *, seed_query, paradigm, key_terms, cfg):
        if not cfg.enable_llm:
            return []
        from services.llm_query_optimizer import propose_semantic_variations
        vars_ = await propose_semantic_variations(seed_query, paradigm=paradigm, max_variants=cfg.per_stage_caps["llm"], key_terms=key_terms)
        return [QueryCandidate(query=v, stage="llm", label="semantic", weight=0.8) for v in vars_]

# ParadigmStage: wraps each *SearchStrategy.generate_search_queries*
class ParadigmStage:
    def __init__(self, get_search_strategy):
        self.get_search_strategy = get_search_strategy
    async def generate(self, *, seed_query, paradigm, key_terms, cfg):
        strategy = self.get_search_strategy(paradigm)
        ctx = strategy.SearchContext(original_query=seed_query, paradigm=paradigm)  # reusing your dataclass
        items = await strategy.generate_search_queries(ctx)
        out = []
        for it in items[:cfg.per_stage_caps["paradigm"]]:
            out.append(QueryCandidate(query=it["query"], stage="paradigm", label=it.get("type","paradigm"), weight=float(it.get("weight",0.85)), source_filter=it.get("source_filter")))
        return out

# AgenticFollowupsStage: wraps agentic_process proposals (iteration ≥ 2)
class AgenticFollowupsStage:
    async def generate(self, *, seed_query, paradigm, key_terms, cfg, coverage=None, missing_terms=None, domain_gaps=None):
        if not (missing_terms or domain_gaps):  # must be fed from prior results
            return []
        from services.agentic_process import propose_queries_enriched
        props = propose_queries_enriched(base_query=seed_query, paradigm=paradigm, missing_terms=missing_terms or [], gap_counts=domain_gaps or {}, max_new=cfg.per_stage_caps["agentic"])
        return [QueryCandidate(query=p, stage="agentic", label="followup", weight=0.75) for p in props]
```

## 3) Planner orchestration (deterministic merge + dedup)

```python
# search/query_planner/planner.py
from .types import PlannerConfig, QueryCandidate
from .base import RuleBasedStage, LLMVariationsStage, ParadigmStage, AgenticFollowupsStage
from services.search_apis import QueryOptimizer
from services.paradigm_search import get_search_strategy
# Key-term extraction: use QueryOptimizer.get_key_terms() (or add an equivalent helper to QueryAnalyzer)

def _canon(q: str) -> str:  # canonicalize for dedup
    q = q.strip()
    q = " ".join(q.split())
    return q.lower()

def _jaccard(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))

class QueryPlanner:
    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self.rule = RuleBasedStage(QueryOptimizer())
        self.llm  = LLMVariationsStage()
        self.par  = ParadigmStage(get_search_strategy)
        self.agentic = AgenticFollowupsStage()
        self.qopt_terms = QueryOptimizer()

    async def initial_plan(self, *, seed_query: str, paradigm: str) -> list[QueryCandidate]:
        key_terms = self.qopt_terms.get_key_terms(seed_query)
        stage_map = {"rule_based": self.rule, "llm": self.llm, "paradigm": self.par}
        bag: list[QueryCandidate] = []
        for stage_name in self.cfg.stage_order:
            gens = await stage_map[stage_name].generate(seed_query=seed_query, paradigm=paradigm, key_terms=key_terms, cfg=self.cfg)
            bag.extend(gens)
        return self._merge_and_rank(bag)

    async def followups(self, *, seed_query: str, paradigm: str, coverage, missing_terms, domain_gaps) -> list[QueryCandidate]:
        key_terms = self.qopt_terms.get_key_terms(seed_query)
        bag = await self.agentic.generate(seed_query=seed_query, paradigm=paradigm, key_terms=key_terms, cfg=self.cfg, coverage=coverage, missing_terms=missing_terms, domain_gaps=domain_gaps)
        return self._merge_and_rank(bag)

    def _merge_and_rank(self, items: list[QueryCandidate]) -> list[QueryCandidate]:
        out: list[QueryCandidate] = []
        seen: list[str] = []
        for c in items:
            cq = _canon(c.query)
            # dedup exact and near-duplicate
            if any(s == cq or _jaccard(s, cq) >= self.cfg.dedup_jaccard for s in seen):
                continue
            seen.append(cq)
            out.append(c)
        # rank by (stage prior + weight): paradigm > rule_based > llm by default
        stage_prior = {"paradigm": 1.0, "rule_based": 0.95, "llm": 0.9, "agentic": 0.88}
        out.sort(key=lambda c: (stage_prior.get(c.stage, 0.8) * c.weight), reverse=True)
        return out[: self.cfg.max_candidates]
```

**Key choices:**

* **I/O unification:** everything becomes a `QueryCandidate` with `stage`, `label`, `weight`, and optional `source_filter`.
* **Dedup once:** canonicalize + token Jaccard to collapse near-identical queries (provider code shouldn’t dedup again).
* **Deterministic ranking:** slight prior for paradigm queries, then rule-based, then LLM, then agentic—tunable per mode.

---

# Wiring into the rest of the app (minimal diffs)

1. **Stop provider-side invention.** In `BaseSearchAPI.search_with_variations()`, accept an optional list of `QueryCandidate`s; if provided, **use those** and skip `self.qopt.generate_query_variations`. Keep current behavior as a fallback for backwards compatibility.

   ```python
   async def search_with_variations(self, query: str, cfg: SearchConfig, planned: list[QueryCandidate] | None = None) -> List[SearchResult]:
       variants = planned or [QueryCandidate(query=q, stage="rule_based", label=k) for k,q in self.qopt.generate_query_variations(query).items()]
       # preserve current per-variant execution & annotate `query_variant` = f"{c.stage}:{c.label}"
   ```

2. **In `research_orchestrator.py`:**

   * Build a `QueryPlanner` once per run using env-backed `PlannerConfig`:

     * `ENABLE_QUERY_LLM` → `cfg.enable_llm`
     * `SEARCH_QUERY_VARIATIONS_LIMIT` → `cfg.max_candidates`
     * `ADAPTIVE_QUERY_LIMIT` can still influence `ParadigmStage` through its own strategy.
   * Call `planner.initial_plan(seed_query, paradigm)` to get candidates; when using planned candidates, skip `_compress_and_dedup_queries()` and `_prioritize_queries()` to avoid double-dedup/order drift.
   * Execute via `search_manager.search_with_plan(planned)` (a lightweight wrapper that passes candidates down to providers).
   * After the first batch, compute coverage/gaps using your existing `evaluate_coverage_from_sources` & `summarize_domain_gaps`, then call `planner.followups(...)` to fetch the second round. Stop when coverage ≥ threshold or on budget.

3. **Paradigm ranking stays.** You already use `strategy.filter_and_rank_results(...)`; keep it for result ranking. The planner only standardizes *inputs* to search.

---

# Config by mode/paradigm (practical defaults)

* **Bernard (analytical):** `stage_order = ["paradigm","rule_based","llm"]`, `max_candidates=12`, `per_stage_caps={"paradigm":6,"rule_based":6,"llm":3}`. Enable agentic follow-ups if coverage < 0.7.
* **Maeve (strategic):** favor paradigm stage (industry/market operators), smaller llm cap (2), broader dedup (lower `dedup_jaccard` to 0.88).
* **Dolores (investigative):** prioritize paradigm stage, increase `exact_phrase` weight, and set `source_filter="investigative"` for at least 2 candidates.
* **Teddy (support/community):** more “resource pattern” queries; cap broad variants to keep precision.

Surface these as JSON/YAML once, not scattered across modules.

---

# Telemetry & safety rails

* **Trace each candidate:** include `tags={"reason":"synonym","entity":"<X>"}`, or e.g. `{"missing_term":"right to repair"}` for agentic. Log a compact “plan trace” for debugging.
* **Budget-aware:** enforce a global `planner.max_candidates` and per-stage caps. Respect existing provider CPH/CPM throttles; the planner is just deciding *what* to search.
* **Locale/language:** if the analyzer detects non-English, pass `language` down to `SearchConfig` and mute WordNet synonyms.

---

# Migration plan (2 PRs)

**PR1 (non-breaking):**

* Add `planned: List[QueryCandidate] | None` parameter to `BaseSearchAPI.search_with_variations()`.
* Implement `QueryPlanner` module.
* Add `SearchAPIManager.search_with_plan(planned)` and wire it to pass `planned` down to providers via `.search_with_variations(..., planned=planned)`.
* In `research_orchestrator.py`, construct & call the planner; pass plan to the search manager using `search_with_plan`. Keep the provider fallback intact.

**PR2 (cleanup):**

* Remove internal `qopt.generate_query_variations()` call paths when `planned` is present.
* Deprecate any stray calls to `llm_query_optimizer.propose_semantic_variations()` and `paradigm_search.*generate_search_queries()` outside the planner.
* Consolidate env → `PlannerConfig`.

---

# Example (what the plan looks like)

Seed: `“battery recycling market growth 2024”` (Maeve)

Planner emits (trimmed):

```
1. [paradigm:industry_focus]    battery recycling market growth 2024 industry analysis   (w=0.95, source_filter=None)
2. [rule_based:primary]         battery recycling market growth 2024                     (w=1.00)
3. [rule_based:exact_phrase]    "battery recycling market growth 2024"                   (w=0.90)
4. [paradigm:strategic_pattern] battery recycling TAM SAM SOM forecast                   (w=0.90)
5. [llm:semantic]               global e-waste battery recycling outlook 2024-2026       (w=0.80)
6. [paradigm:source_target]     battery recycling Gartner OR BCG report                  (w=0.85, source_filter="industry")
```

Providers run exactly these, annotate results with `query_variant = stage:label`, and you keep your downstream ranking intact.

---

# Risks & mitigations

* **Double dedup (planner vs orchestrator):** turn off any query-level dedup elsewhere; keep result-level dedup in `research_orchestrator.py`.
* **Stage drift:** all heuristics now live in one place. Add unit tests with golden plans per paradigm.
* **Cost creep with LLM stage:** behind `enable_llm` and a per-stage cap; also cache LLM variations keyed by `(seed_query, paradigm)`.

---

# Tests to add (fast and surgical)

* **Golden plan tests:** for 8–10 fixed seeds × paradigms, snapshot the top N `QueryCandidate`s (stage, label, query).
* **Property tests:** same seed with/without LLM produces superset plan; agentic follow-ups add only unseen candidates.
* **Dedup test:** queries that differ by punctuation/spacing collapse.
* **Locale test:** non-English queries skip WordNet synonyms.

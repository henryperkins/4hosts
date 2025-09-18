# Unified Query Planning and Orchestrator Modularization (Single-Source Plan)

TL;DR
- One planner emits a single, deduplicated, ranked list of QueryCandidates for all providers. When a plan is provided, providers stop inventing their own variations and annotate results with stage:label.
- The orchestrator gains a clean seam via a manager-level search_with_plan(planned) entry point while keeping result ranking, deduplication, credibility, and synthesis in modular services.
- This document consolidates and reconciles: docs/queryplan.md, docs/queryplan2.md, QUERY_PLANNING_CONSOLIDATION.md, RESEARCH_ORCHESTRATOR_MODULARIZATION.md, MODULARIZATION_COORDINATION.md, and the validated convergence plan against current code.

-------------------------------------------------------------------------------

1) Current State (What actually happens today)

Active variation sources
- Rule-based variations inside provider path:
  - Providers call the rule-based optimizer for every query via [services.search_apis.BaseSearchAPI.search_with_variations()](four-hosts-app/backend/services/search_apis.py:1451), which uses [services.search_apis.QueryOptimizer.generate_query_variations()](four-hosts-app/backend/services/search_apis.py:1185).
  - Key-term utility lives in [services.search_apis.QueryOptimizer.get_key_terms()](four-hosts-app/backend/services/search_apis.py:1178).
- Optional LLM variations (behind flag):
  - [services.llm_query_optimizer.propose_semantic_variations()](four-hosts-app/backend/services/llm_query_optimizer.py:32), used by Context Engineering’s OptimizeLayer when ENABLE_QUERY_LLM is on ([services.context_engineering.OptimizeLayer.process](four-hosts-app/backend/services/context_engineering.py:882)).

Valuable but underused generators
- Paradigm-specific query generators are rich but not used for query formation sent to providers:
  - [services.paradigm_search.get_search_strategy()](four-hosts-app/backend/services/paradigm_search.py:1092) and per-paradigm generate_search_queries are used later for result ranking only.
- Agentic follow-ups exist but aren’t looped into searches:
  - Coverage and gap helpers, plus follow-up query proposals: [services.agentic_process.propose_queries_enriched()](four-hosts-app/backend/services/agentic_process.py:151).

Net effect today
- Double expansion and scattered knobs:
  - Context engineering produces refined queries (joining heuristic and paradigm outputs). The orchestrator compresses/prioritizes those, and each provider re-expands again using rule-based variations.
  - Different data shapes and priorities live across multiple modules.

-------------------------------------------------------------------------------

2) Unified Query Planner (Adapter-first, non-breaking; PR1)

Goal
- Introduce a minimal “search/query_planner/” with adapters that wrap existing code to produce a single list of QueryCandidates, ranked and deduplicated once.

Types (single source of truth)
```python
# search/query_planner/types.py
from dataclasses import dataclass, field
from typing import Literal, Dict, List, Optional

StageName = Literal["rule_based", "llm", "paradigm", "agentic"]

@dataclass
class QueryCandidate:
    query: str
    stage: StageName
    label: str
    weight: float = 1.0
    source_filter: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PlannerConfig:
    max_candidates: int = 12
    enable_llm: bool = False
    stage_order: List[StageName] = field(default_factory=lambda: ["rule_based","paradigm","llm"])
    per_stage_caps: Dict[StageName, int] = field(default_factory=lambda: {"rule_based": 6, "paradigm": 6, "llm": 4, "agentic": 6})
    dedup_jaccard: float = 0.92
```

Stage adapters (wrapping existing code)
- RuleBasedStage → [services.search_apis.QueryOptimizer.generate_query_variations()](four-hosts-app/backend/services/search_apis.py:1185)
- LLMVariationsStage → [services.llm_query_optimizer.propose_semantic_variations()](four-hosts-app/backend/services/llm_query_optimizer.py:32)
- ParadigmStage → [services.paradigm_search.get_search_strategy()](four-hosts-app/backend/services/paradigm_search.py:1092).generate_search_queries(...)
- AgenticFollowupsStage → [services.agentic_process.propose_queries_enriched()](four-hosts-app/backend/services/agentic_process.py:151)

Planner orchestration (deterministic merge + dedup)
- Key-term extraction source of truth for planner: [services.search_apis.QueryOptimizer.get_key_terms()](four-hosts-app/backend/services/search_apis.py:1178)
- Canonicalize + Jaccard token dedup once; rank with small stage priors × weights.

Important correctness notes (repo-validated)
- Use QueryOptimizer.get_key_terms rather than a non-existent QueryAnalyzer.get_key_terms on [services.classification_engine.QueryAnalyzer](four-hosts-app/backend/services/classification_engine.py:159).
- Keep paradigms’ result ranking untouched (the planner unifies inputs; ranking stays where it is used now).

-------------------------------------------------------------------------------

3) Manager + Provider Integration (Seam for planned candidates; PR1)

Provider: planned hook
- Extend [services.search_apis.BaseSearchAPI.search_with_variations()](four-hosts-app/backend/services/search_apis.py:1451) to accept planned: Optional[List[QueryCandidate]].
  - If planned is provided:
    - Do not call QueryOptimizer.generate_query_variations.
    - Execute each candidate’s query directly.
    - Annotate result.raw_data["query_variant"] = f"{cand.stage}:{cand.label}".
  - Else:
    - Keep current behavior (rule-based variations + legacy variant label string).

Manager: fan-out planned candidates
- Add SearchAPIManager.search_with_plan(planned) that:
  - Calls each provider’s search_with_variations(query, cfg, planned=planned) (preserving provider concurrency/limits).
  - This is layered alongside [services.search_apis.SearchAPIManager._search_single_provider()](four-hosts-app/backend/services/search_apis.py:2147) and [services.search_apis.SearchAPIManager._search_provider_silent()](four-hosts-app/backend/services/search_apis.py:2212).

Orchestrator: clean seam
- Current orchestrator call-site is [services.research_orchestrator.UnifiedResearchOrchestrator.execute_research()](four-hosts-app/backend/services/research_orchestrator.py:912):
  - Introduce planner.initial_plan(...) to produce planned candidates.
  - Use search_manager.search_with_plan(planned).
  - After initial batch, compute coverage/gaps; if needed, planner.followups(...) → search_with_plan again.
  - When the planned path is active, skip orchestrator’s own query compression/prioritization to avoid double-dedup/order drift.

UI/Adapters: variant exposure
- Add a small getter so downstream consumers can display stage:label without depending on raw_data internals. Reference: [services.result_adapter](four-hosts-app/backend/services/result_adapter.py:1).

-------------------------------------------------------------------------------

4) Impact Map (outside the 7 core files; PR1 safe changes + PR2 optional)

Medium / High
- agentic_process.py: keep signatures (List[str]). Planner consumes propose_queries_enriched for follow-ups.
- llm_query_optimizer.py: leave ENABLE_QUERY_LLM guard in place; planner.enable_llm gates the stage too.
- paradigm_search.py: keep generate_search_queries items with fields {"query","type","weight","source_filter"}; planner maps them to QueryCandidate. Result ranking continues to use strategy.filter_and_rank_results(...).
- context_engineering.py: Leave OptimizeLayer behavior in PR1 (test stability). Optional PR2: adopt planner.initial_plan for preview variations in the UI.

Manager/Provider
- BaseSearchAPI planned hook + SearchAPIManager.search_with_plan(planned) are the only code-path expansions required to pass planned candidates uniformly across providers.

Low (quick wins)
- result_adapter: add property query_variant to expose f"{stage}:{label}" when available.

-------------------------------------------------------------------------------

5) Telemetry & Safety Rails

- Plan trace: log stage counts, dedup rate, and chosen candidates once per run.
- Budget-aware: planner.max_candidates + per-stage caps. Respect provider-side quota/rate limits as today.
- Locale/language: if analyzer detects non-English, pass language to SearchConfig and mute WordNet synonyms in the rule-based path.

-------------------------------------------------------------------------------

6) Tests to Add

- Planner golden plan: fixed seed × paradigm snapshots for the top N QueryCandidates (stage,label,query).
- Property checks: With LLM disabled vs enabled produces a superset plan; agentic follow-ups add new items only.
- Provider passthrough: Given planned candidates, providers do not call QueryOptimizer.generate_query_variations().
- Adapter: ResultAdapter.query_variant returns stage:label (and legacy string still supported when not planned).

-------------------------------------------------------------------------------

7) Query Planning Consolidation (services/query_planning/*; PR2/PR3)

Goal and scope (from QUERY_PLANNING_CONSOLIDATION.md)
- Move from “four cooks” to a unified query planning package:
  - config.py: paradigm query configs (modifiers/operators/source preferences/cleaning rules).
  - cleaner.py: consolidate cleaning (noise removal, entity protection, replacements).
  - variator.py: unify rule-based, LLM, and strategic patterns into a single generator that returns variations with labels and weights.
  - coverage.py: coverage analysis + gap-driven follow-ups (wrapping today’s agentic helpers).

How PR1 feeds PR2/PR3
- PR1 delivers the canonical seam and adapters, keeping behavior unchanged while centralizing the plan. PR2+ can move adapters into variator.py, wire configs into settings, and gradually shift CE to planner previews or full planner output.

Centralized settings (mapping)
- Today’s env:
  - ENABLE_QUERY_LLM → PlannerConfig.enable_llm
  - SEARCH_QUERY_VARIATIONS_LIMIT → PlannerConfig.max_candidates
  - ADAPTIVE_QUERY_LIMIT → influences ParadigmStage/strategy generation
- Future unified settings (example variables):
  - UNIFIED_QUERY_MAX_VARIATIONS, UNIFIED_QUERY_ENABLE_LLM, UNIFIED_QUERY_ENABLE_STRATEGIC_PATTERNS, UNIFIED_QUERY_ENABLE_FOLLOW_UP

-------------------------------------------------------------------------------

8) Research Orchestrator Modularization (extractions; independent of planner)

Intent (from RESEARCH_ORCHESTRATOR_MODULARIZATION.md)
- Shrink the monolith into focused services while preserving existing behavior:
  - result_deduplicator.py: advanced simhash/content similarity (extract from orchestrator).
  - relevance_filter.py: unify EarlyRelevanceFilter + ContentRelevanceFilter.
  - synthesis_orchestrator.py: answer generation + evidence orchestration.
  - deep_research_orchestrator.py / exa_research_orchestrator.py: specialized integrations.
  - planning_utils.py: budget/retry configs used by orchestrator.

Alignment with planner
- The planner only determines “what to search.”
- The modularized orchestrator continues to do “how to process results,” including ranking (via paradigms), dedup, credibility, synthesis, and deep-research enrichment.
- The manager planned seam (search_with_plan) is a stable interface the modularized orchestrator can call.

-------------------------------------------------------------------------------

9) Coordination Plan (from MODULARIZATION_COORDINATION.md + convergence notes)

Phase ordering
- Phase A (PR1): Introduce planner adapters + manager planned seam + provider hook. Orchestrator uses planner.initial_plan and search_with_plan; result ranking unchanged. CE unchanged.
- Phase B (PR2): Begin Query Planning Consolidation package (services/query_planning/*); optionally migrate CE’s OptimizeLayer to planner.initial_plan (low-cap preview) or full planner output for refined_queries. Add plan trace metrics; unify envs → settings.
- Phase C (PR2/PR3): Orchestrator modularization extractions (deduplicator, relevance, synthesis, deep research, cost tracking via metrics). Keep planner path intact.
- Phase D (Cleanup): Remove duplicate or redundant query expansion paths when planned candidates are present; keep result-level dedup.

Skip double work
- When planner is supplying planned candidates, skip orchestrator’s own query compression/prioritization for that path to avoid double-dedup/order drift.
- Preserve orchestrator’s result ranking and credibility pipeline.

-------------------------------------------------------------------------------

10) Migration Slices (PR1/PR2) and Timeline

PR1 (non-breaking; low risk)
- Implement search/query_planner/* (types/base/planner).
- Add planned hook to BaseSearchAPI.search_with_variations(...) and annotate query_variant stage:label for planned results.
- Add SearchAPIManager.search_with_plan(planned) and wire orchestrator call-site to use planner.initial_plan and manager.search_with_plan.
- Keep CE’s OptimizeLayer unchanged to preserve existing tests.
- Add ResultAdapter.query_variant property.

PR2 (consolidation)
- Move adapters into services/query_planning/ (config/cleaner/variator/coverage/settings), retaining behavior parity.
- Optionally refactor CE to use planner.initial_plan for previews/full output.
- Centralize env flags to single settings.
- Begin orchestrator modularization extractions; maintain compatibility with manager planned seam.

Indicative estimates (adjust per team capacity)
- PR1: 2–4 days + tests.
- PR2: 5–10 days for consolidation + orchestrator extractions; tests and benchmarks in parallel.

-------------------------------------------------------------------------------

11) Risks & Mitigations

- Double deduplication (planner vs orchestrator): Skip orchestrator’s compression/prioritization when planned is active; keep result-level dedup.
- Stage drift: Keep all heuristics in planner package and add golden plan tests per paradigm.
- LLM cost creep: Gate with enable_llm and caps; add caching of LLM variations keyed by (seed_query, paradigm).
- Back-compat: Leave provider legacy behavior when planned is None.

-------------------------------------------------------------------------------

12) Testing Strategy and Success Metrics

Testing
- Planner golden snapshots (top N candidates per paradigm).
- Passthrough tests ensuring providers do not call QueryOptimizer when planned exists.
- Orchestrator integration tests for follow-ups firing only when coverage thresholds indicate gaps.
- Adapter test for query_variant stage:label visibility.

KPIs
- Code quality: reduce query planning code duplication; maintain or improve coverage.
- Functional parity: preserve existing variation coverage where planner is disabled; planned path produces equal or better recall.
- Performance: avoid material regressions; reduce redundant expansions.

-------------------------------------------------------------------------------

Appendix: Key References (repo-validated clickable links)

- Rule-based optimizer
  - [services.search_apis.QueryOptimizer.get_key_terms()](four-hosts-app/backend/services/search_apis.py:1178)
  - [services.search_apis.QueryOptimizer.generate_query_variations()](four-hosts-app/backend/services/search_apis.py:1185)
  - [services.search_apis.QueryOptimizer.optimize_query()](four-hosts-app/backend/services/search_apis.py:1270)
- Provider and manager
  - [services.search_apis.BaseSearchAPI.search_with_variations()](four-hosts-app/backend/services/search_apis.py:1451)
  - [services.search_apis.SearchAPIManager._search_single_provider()](four-hosts-app/backend/services/search_apis.py:2147)
  - [services.search_apis.SearchAPIManager._search_provider_silent()](four-hosts-app/backend/services/search_apis.py:2212)
- LLM variations
  - [services.llm_query_optimizer.propose_semantic_variations()](four-hosts-app/backend/services/llm_query_optimizer.py:32)
- Paradigm search
  - [services.paradigm_search.get_search_strategy()](four-hosts-app/backend/services/paradigm_search.py:1092)
- Agentic follow-ups and coverage
  - [services.agentic_process.propose_queries_enriched()](four-hosts-app/backend/services/agentic_process.py:151)
- Orchestrator call-site (planned seam integration)
  - [services.research_orchestrator.UnifiedResearchOrchestrator.execute_research()](four-hosts-app/backend/services/research_orchestrator.py:912)
- Adapter exposure
  - [services.result_adapter](four-hosts-app/backend/services/result_adapter.py:1)
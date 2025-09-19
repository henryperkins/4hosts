from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Callable, Any, Dict

from .optimizer import QueryOptimizer
from services.agentic_process import (
    propose_queries_enriched,
    summarize_domain_gaps,
)
from services.llm_query_optimizer import propose_semantic_variations

from .types import PlannerConfig, QueryCandidate
from .cleaner import canon_query


def _emit_candidates(
    items: Iterable[Any],
    *,
    stage: str,
    cap: int,
    to_query: Callable[[Any, int], Optional[str]],
    make_label: Callable[[Any, int], str],
    make_weight: Callable[[Any, int], float],
    make_tags: Optional[Callable[[Any, int], Optional[Dict[str, str]]]] = None,
    make_source_filter: Optional[Callable[[Any, int], Optional[str]]] = None,
) -> List[QueryCandidate]:
    out: List[QueryCandidate] = []
    for idx, item in enumerate(items):
        raw_q = to_query(item, idx)
        q = canon_query(raw_q or "")
        if not q:
            continue
        tags = make_tags(item, idx) if make_tags else None
        source_filter = make_source_filter(item, idx) if make_source_filter else None
        out.append(
            QueryCandidate(
                query=q,
                stage=stage,  # type: ignore[arg-type]
                label=make_label(item, idx),
                weight=float(make_weight(item, idx)),
                source_filter=source_filter,
                tags=tags or {},
            )
        )
        if len(out) >= cap:
            break
    return out


class RuleBasedStage:
    def __init__(self, optimizer: Optional[QueryOptimizer] = None) -> None:
        self.optimizer = optimizer or QueryOptimizer()

    async def generate(
        self,
        *,
        seed_query: str,
        paradigm: str,
        cfg: PlannerConfig,
    ) -> List[QueryCandidate]:
        variations = self.optimizer.generate_query_variations(seed_query, paradigm=paradigm)
        priority = [
            "primary",
            "domain_specific",
            "exact_phrase",
            "synonym",
            "related",
            "broad",
            "question",
        ]
        weights = {
            "primary": 1.0,
            "domain_specific": 0.95,
            "exact_phrase": 0.92,
            "synonym": 0.85,
            "related": 0.8,
            "broad": 0.72,
            "question": 0.7,
        }
        cap = cfg.per_stage_caps.get("rule_based", len(priority))
        return _emit_candidates(
            priority,
            stage="rule_based",
            cap=cap,
            to_query=lambda label, i: variations.get(label),
            make_label=lambda label, i: str(label),
            make_weight=lambda label, i: float(weights.get(label, 0.7)),
        )


class LLMVariationsStage:
    def __init__(self, optimizer: Optional[QueryOptimizer] = None) -> None:
        self.optimizer = optimizer or QueryOptimizer()

    async def generate(
        self,
        *,
        seed_query: str,
        paradigm: str,
        cfg: PlannerConfig,
    ) -> List[QueryCandidate]:
        if not cfg.enable_llm:
            return []
        key_terms = self.optimizer.get_key_terms(seed_query)[:8]
        variations = await propose_semantic_variations(
            seed_query,
            paradigm,
            max_variants=cfg.per_stage_caps.get("llm", 4),
            key_terms=key_terms,
        )
        cap = cfg.per_stage_caps.get("llm", 4)
        return _emit_candidates(
            variations,
            stage="llm",
            cap=cap,
            to_query=lambda v, i: v,
            make_label=lambda _v, i: f"semantic_{i+1}",
            make_weight=lambda _v, i: 0.8,
        )


class ParadigmStage:
    async def generate(
        self,
        *,
        seed_query: str,
        paradigm: str,
        cfg: PlannerConfig,
    ) -> List[QueryCandidate]:
        from services.paradigm_search import get_search_strategy, SearchContext

        strategy = get_search_strategy(paradigm)
        context = SearchContext(
            original_query=seed_query,
            paradigm=paradigm,
        )
        queries = await strategy.generate_search_queries(context)
        cap = cfg.per_stage_caps.get("paradigm", 6)
        return _emit_candidates(
            queries,
            stage="paradigm",
            cap=cap,
            to_query=lambda it, i: it.get("query", ""),
            make_label=lambda it, i: str(it.get("type", "paradigm")),
            make_weight=lambda it, i: float(it.get("weight", 0.9) or 0.9),
            make_source_filter=lambda it, i: it.get("source_filter"),
            make_tags=lambda it, i: {"paradigm": paradigm},
        )


class ContextStage:
    async def generate(
        self,
        additional_queries: Iterable[str],
        cfg: PlannerConfig,
    ) -> List[QueryCandidate]:
        cap = cfg.per_stage_caps.get("context", 6)
        return _emit_candidates(
            list(additional_queries),
            stage="context",
            cap=cap,
            to_query=lambda raw, i: raw,
            make_label=lambda _raw, i: f"context_{i+1}",
            make_weight=lambda _raw, i: 0.88,
        )


class AgenticFollowupsStage:
    def __init__(self, optimizer: Optional[QueryOptimizer] = None) -> None:
        self.optimizer = optimizer or QueryOptimizer()

    async def generate(
        self,
        *,
        seed_query: str,
        paradigm: str,
        missing_terms: Sequence[str],
        cfg: PlannerConfig,
        coverage_sources: Optional[Sequence[dict]] = None,
    ) -> List[QueryCandidate]:
        if not cfg.enable_agentic:
            return []
        coverage_sources = coverage_sources or []
        domain_gaps = summarize_domain_gaps(list(coverage_sources))
        proposals = propose_queries_enriched(
            base_query=seed_query,
            paradigm=paradigm,
            missing_terms=list(missing_terms),
            gap_counts=domain_gaps,
            max_new=cfg.per_stage_caps.get("agentic", 4),
        )
        cap = cfg.per_stage_caps.get("agentic", 4)
        return _emit_candidates(
            proposals,
            stage="agentic",
            cap=cap,
            to_query=lambda p, i: p,
            make_label=lambda _p, i: f"followup_{i+1}",
            make_weight=lambda _p, i: 0.75,
            make_tags=lambda _p, i: {"missing_term": missing_terms[i] if i < len(missing_terms) else ""},
        )

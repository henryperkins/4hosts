from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .optimizer import QueryOptimizer
from services.agentic_process import (
    propose_queries_enriched,
    summarize_domain_gaps,
)
from services.llm_query_optimizer import propose_semantic_variations

from .types import PlannerConfig, QueryCandidate
from .cleaner import canon_query


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
        out: List[QueryCandidate] = []
        for label in priority:
            query = variations.get(label)
            if not query:
                continue
            out.append(
                QueryCandidate(
                    query=canon_query(query),
                    stage="rule_based",
                    label=label,
                    weight=weights.get(label, 0.7),
                )
            )
            if len(out) >= cfg.per_stage_caps.get("rule_based", len(priority)):
                break
        return out


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
        out: List[QueryCandidate] = []
        for idx, variation in enumerate(variations):
            out.append(
                QueryCandidate(
                    query=canon_query(variation),
                    stage="llm",
                    label=f"semantic_{idx+1}",
                    weight=0.8,
                )
            )
        return out


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
        out: List[QueryCandidate] = []
        for item in queries:
            query = canon_query(item.get("query", ""))
            if not query:
                continue
            label = item.get("type", "paradigm")
            weight = float(item.get("weight", 0.9) or 0.9)
            source_filter = item.get("source_filter")
            out.append(
                QueryCandidate(
                    query=query,
                    stage="paradigm",
                    label=str(label),
                    weight=weight,
                    source_filter=source_filter,
                    tags={"paradigm": paradigm},
                )
            )
            if len(out) >= cap:
                break
        return out


class ContextStage:
    async def generate(
        self,
        additional_queries: Iterable[str],
        cfg: PlannerConfig,
    ) -> List[QueryCandidate]:
        cap = cfg.per_stage_caps.get("context", 6)
        out: List[QueryCandidate] = []
        for idx, raw in enumerate(additional_queries):
            query = canon_query(raw)
            if not query:
                continue
            out.append(
                QueryCandidate(
                    query=query,
                    stage="context",
                    label=f"context_{idx+1}",
                    weight=0.88,
                )
            )
            if len(out) >= cap:
                break
        return out


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
        out: List[QueryCandidate] = []
        for idx, proposal in enumerate(proposals):
            out.append(
                QueryCandidate(
                    query=canon_query(proposal),
                    stage="agentic",
                    label=f"followup_{idx+1}",
                    weight=0.75,
                    tags={"missing_term": missing_terms[idx] if idx < len(missing_terms) else ""},
                )
            )
        return out

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, TYPE_CHECKING

from services.query_planning import (
    QueryCandidate,
    PlannerConfig,
    StageName,
    RuleBasedStage,
    LLMVariationsStage,
    ParadigmStage,
    ContextStage,
    AgenticFollowupsStage,
    canon_query,
    is_duplicate,
)

if TYPE_CHECKING:
    from services.search_apis import QueryOptimizer


class QueryPlanner:
    def __init__(
        self,
        cfg: Optional[PlannerConfig] = None,
        *,
        optimizer: Optional["QueryOptimizer"] = None,
    ) -> None:
        self.cfg = cfg or PlannerConfig()
        self.rule_stage = RuleBasedStage(optimizer)
        self.llm_stage = LLMVariationsStage(optimizer)
        self.paradigm_stage = ParadigmStage()
        self.context_stage = ContextStage()
        self.agentic_stage = AgenticFollowupsStage(optimizer)

    async def initial_plan(
        self,
        *,
        seed_query: str,
        paradigm: str,
        additional_queries: Optional[Iterable[str]] = None,
    ) -> List[QueryCandidate]:
        bag: List[QueryCandidate] = []
        stage_order: Sequence[StageName] = self.cfg.stage_order
        generators = {
            "rule_based": lambda: self.rule_stage.generate(
                seed_query=seed_query,
                paradigm=paradigm,
                cfg=self.cfg,
            ),
            "llm": lambda: self.llm_stage.generate(
                seed_query=seed_query,
                paradigm=paradigm,
                cfg=self.cfg,
            ),
            "paradigm": lambda: self.paradigm_stage.generate(
                seed_query=seed_query,
                paradigm=paradigm,
                cfg=self.cfg,
            ),
        }
        for stage_name in stage_order:
            generator = generators.get(stage_name)
            if not generator:
                continue
            candidates = await generator()
            bag.extend(candidates)
        if additional_queries:
            context_candidates = await self.context_stage.generate(
                additional_queries, self.cfg
            )
            bag.extend(context_candidates)
        return self._merge_and_rank(bag)

    async def followups(
        self,
        *,
        seed_query: str,
        paradigm: str,
        missing_terms: Sequence[str],
        coverage_sources: Optional[Sequence[dict]] = None,
    ) -> List[QueryCandidate]:
        followup_candidates = await self.agentic_stage.generate(
            seed_query=seed_query,
            paradigm=paradigm,
            missing_terms=missing_terms,
            cfg=self.cfg,
            coverage_sources=coverage_sources,
        )
        return self._merge_and_rank(followup_candidates)

    def _merge_and_rank(self, items: Iterable[QueryCandidate]) -> List[QueryCandidate]:
        seen: List[str] = []
        out: List[QueryCandidate] = []
        for cand in items:
            query = canon_query(cand.query)
            if not query:
                continue
            if is_duplicate(query, seen, self.cfg.dedup_jaccard):
                continue
            seen.append(query)
            out.append(
                QueryCandidate(
                    query=query,
                    stage=cand.stage,
                    label=cand.label,
                    weight=cand.weight,
                    source_filter=cand.source_filter,
                    tags=dict(cand.tags),
                )
            )
        stage_prior = {
            "paradigm": 1.0,
            "rule_based": 0.96,
            "llm": 0.9,
            "context": 0.88,
            "agentic": 0.86,
        }
        out.sort(
            key=lambda c: stage_prior.get(c.stage, 0.8) * c.weight,
            reverse=True,
        )
        return out[: self.cfg.max_candidates]

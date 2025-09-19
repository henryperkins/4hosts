from __future__ import annotations

import time
import traceback
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

import structlog

logger = structlog.get_logger(__name__)


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
        start_time = time.time()

        # Materialize additional_queries only if needed to avoid exhausting generators
        aq_for_stage: Optional[Iterable[str]] = None
        aq_count = 0
        if additional_queries is not None:
            try:
                # If it's a Sized iterable, len() won't consume it
                aq_count = len(additional_queries)  # type: ignore[arg-type]
                aq_for_stage = additional_queries
            except TypeError:
                # Generator/iterator: materialize once and reuse
                aq_list = list(additional_queries)
                aq_count = len(aq_list)
                aq_for_stage = aq_list

        logger.info(
            "Starting query planning",
            stage="query_planning_start",
            paradigm=paradigm,
            seed_query=(seed_query[:100] if seed_query else ""),
            config={
                "max_candidates": self.cfg.max_candidates,
                "stage_order": list(self.cfg.stage_order),
                "additional_queries_count": aq_count,
            },
        )

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

            stage_start = time.time()
            try:
                candidates = await generator()
                bag.extend(candidates)

                logger.info(
                    "Query planning stage completed",
                    stage="query_planning_stage",
                    stage_name=stage_name,
                    paradigm=paradigm,
                    duration_ms=(time.time() - stage_start) * 1000,
                    record_count=len(candidates),
                    metrics={
                        "candidates_generated": len(candidates),
                        "total_candidates_so_far": len(bag),
                    },
                )
            except Exception as e:
                logger.error(
                    "Query planning stage failed",
                    stage="query_planning_stage_error",
                    stage_name=stage_name,
                    paradigm=paradigm,
                    error_type=type(e).__name__,
                    stack_trace=traceback.format_exc(),
                )
        if aq_for_stage:
            context_start = time.time()
            context_candidates = await self.context_stage.generate(
                aq_for_stage, self.cfg
            )
            bag.extend(context_candidates)

            logger.info(
                "Context queries processed",
                stage="context_queries",
                paradigm=paradigm,
                duration_ms=(time.time() - context_start) * 1000,
                record_count=len(context_candidates),
            )

        result = self._merge_and_rank(bag)

        logger.info(
            "Query planning completed",
            stage="query_planning_complete",
            paradigm=paradigm,
            duration_ms=(time.time() - start_time) * 1000,
            metrics={
                "initial_candidates": len(bag),
                "final_candidates": len(result),
                "deduplication_removed": len(bag) - len(result),
            },
            queries=[c.query[:100] for c in result[:5]],  # first 5 queries only
        )

        return result

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
        start_time = time.time()
        initial_count = len(list(items)) if hasattr(items, "__len__") else 0

        logger.debug(
            "Starting deduplication and ranking",
            stage="deduplication",
            initial_count=initial_count,
        )

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

        logger.debug(
            "Deduplication and ranking completed",
            stage="deduplication_complete",
            duration_ms=(time.time() - start_time) * 1000,
            metrics={
                "duplicates_removed": len(seen) - len(out),
                "final_count": min(len(out), self.cfg.max_candidates),
            },
        )
        
        return out[: self.cfg.max_candidates]

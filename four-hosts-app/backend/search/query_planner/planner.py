from __future__ import annotations

import time
import traceback
from typing import Iterable, List, Optional, Sequence, TYPE_CHECKING

from services.query_planning import (  # pylint: disable=import-error
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

import os
import structlog
from utils.otel import otel_span as _otel_span

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
        _span = _otel_span(
            "rag.plan.initial",
            {"paradigm": paradigm, "max_candidates": self.cfg.max_candidates},
        ).__enter__()

        # Materialize additional_queries only if needed
        # to avoid exhausting generators
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
        # Build stage generator mapping dynamically so overrides in
        # cfg.stage_order (incl. context/agentic) are honored.  Missing
        # earlier entries caused flags like enable_agentic to be no‑ops.
        generators: dict[str, callable] = {
            "rule_based": lambda: self.rule_stage.generate(
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
        if self.cfg.enable_llm and self.cfg.per_stage_caps.get("llm", 0) > 0:
            generators["llm"] = lambda: self.llm_stage.generate(
                seed_query=seed_query,
                paradigm=paradigm,
                cfg=self.cfg,
            )
        # Context stage surfaces caller-provided additional queries. Only
        # construct generator if there are any and it's requested.
        if additional_queries is not None and self.cfg.per_stage_caps.get("context", 0) > 0:
            generators["context"] = lambda: self.context_stage.generate(
                aq_for_stage or [],  # type: ignore[arg-type]
                self.cfg,
            )
        if self.cfg.enable_agentic and self.cfg.per_stage_caps.get("agentic", 0) > 0:
            # Agentic followups during initial plan currently operate on
            # the seed query with empty missing_terms (bootstrap). Real
            # followups with gaps happen in followups().
            generators["agentic"] = lambda: self.agentic_stage.generate(
                seed_query=seed_query,
                paradigm=paradigm,
                missing_terms=(),
                cfg=self.cfg,
                coverage_sources=None,
            )
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
                val = os.getenv("PLANNER_STRICT_EXCEPTIONS", "0")
                if str(val).lower() in {"1", "true", "yes", "on"}:
                    raise
        # (Legacy path removed) – context queries are now handled as a
        # first-class stage when present in stage_order.

        # Emit summary after expansion and before ranking for visibility
        try:
            strategies_used = list({c.stage for c in bag})
        except Exception:
            strategies_used = []
        logger.info(
            "Query planning summary",
            stage="query_planning",
            paradigm=paradigm,
            original_query=seed_query[:100],
            candidates_generated=len(bag),
            strategies_used=strategies_used,
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
            queries=[c.query[:100] for c in result[:5]],
        )

        if _span:
            try:
                _span.end()
            except Exception:
                pass
        return result

    async def followups(
        self,
        *,
        seed_query: str,
        paradigm: str,
        missing_terms: Sequence[str],
        coverage_sources: Optional[Sequence[dict]] = None,
    ) -> List[QueryCandidate]:
        with _otel_span(
            "rag.plan.followups",
            {"paradigm": paradigm, "missing_terms": len(missing_terms)},
        ):
            followup_candidates = await self.agentic_stage.generate(
                seed_query=seed_query,
                paradigm=paradigm,
                missing_terms=missing_terms,
                cfg=self.cfg,
                coverage_sources=coverage_sources,
            )
        return self._merge_and_rank(followup_candidates)

    def _merge_and_rank(
        self, items: Iterable[QueryCandidate]
    ) -> List[QueryCandidate]:
        start_time = time.time()

        # Materialize once to avoid consuming generators/iterators
        # during counting
        if hasattr(items, "__len__"):
            if isinstance(items, list):
                seq = items
            else:
                seq = list(items)
            initial_count = len(seq)
        else:
            seq = list(items)
            initial_count = len(seq)

        logger.debug(
            "Starting deduplication and ranking",
            stage="deduplication",
            initial_count=initial_count,
        )

        seen: List[str] = []
        out: List[QueryCandidate] = []
        for cand in seq:
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
        stage_prior = getattr(self.cfg, "stage_prior", {}) or {
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

        # Emit candidate ranking preview with scores
        try:
            preview = [
                {
                    "query": c.query[:100],
                    "label": c.label,
                    "stage": c.stage,
                    "score": round(stage_prior.get(c.stage, 0.8) * c.weight, 3),
                }
                for c in out[:5]
            ]
            logger.debug(
                "Query candidate ranking",
                stage="query_ranking",
                candidates=preview,
                total=len(out),
            )
        except Exception:
            pass

        logger.debug(
            "Deduplication and ranking completed",
            stage="deduplication_complete",
            duration_ms=(time.time() - start_time) * 1000,
            metrics={
                "duplicates_removed": len(seen) - len(out),
                "final_count": min(len(out), self.cfg.max_candidates),
            },
        )

        # Emit final candidate preview (up to 5) at debug level
        logger.debug(
            "QueryPlanner final candidates",
            top_queries=[c.query[:120] for c in out[:5]],
            total=len(out),
        )

        return out[: self.cfg.max_candidates]

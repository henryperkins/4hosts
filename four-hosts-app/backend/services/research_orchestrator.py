"""
Unified Research Orchestrator V2
Combines all the best features from multiple orchestrator implementations
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple, cast, TypedDict
from datetime import datetime, timezone
from collections import defaultdict, deque
from urllib.parse import quote_plus
from dataclasses import dataclass, field

# Contracts
from contracts import ResearchStatus as ContractResearchStatus  # type: ignore
# json import removed (unused)
import re
from services.research_store import research_store
from models.base import ResearchStatus as RuntimeResearchStatus

from models.context_models import (
    SearchResultSchema, ClassificationResultSchema,
    ContextEngineeredQuerySchema,
    HostParadigm,
    QueryFeaturesSchema
)
from models.paradigms import normalize_to_internal_code
from services.search_apis import (
    SearchResult,
    SearchConfig,
    create_search_manager,
)
from search.query_planner import QueryPlanner, QueryCandidate
from services.query_planning import PlannerConfig, build_planner_config
from services.exa_research import exa_research_client
from services.credibility import get_source_credibility
from services.paradigm_search import get_search_strategy, SearchContext
from services.cache import (
    cache_manager,
)
from services.text_compression import query_compressor
from core.config import (
    EVIDENCE_MAX_DOCS_DEFAULT,
    EVIDENCE_QUOTES_PER_DOC_DEFAULT,
    EVIDENCE_BUDGET_TOKENS_DEFAULT,
    SYNTHESIS_MAX_LENGTH_DEFAULT,
)
from services.deep_research_service import (
    deep_research_service,
    DeepResearchConfig,
    DeepResearchMode,
    initialize_deep_research,
)
from services.agentic_process import (
    evaluate_coverage_from_sources,
    summarize_domain_gaps,
    propose_queries_enriched,
)
from services.result_adapter import ResultAdapter
from services.query_planning import PlannerConfig, build_planner_config
from services.query_planning.result_deduplicator import ResultDeduplicator
from services.query_planning.relevance_filter import EarlyRelevanceFilter
from services.query_planning.planning_utils import (
    CostMonitor,
    ToolRegistry,
    ToolCapability,
    BudgetAwarePlanner,
    RetryPolicy,
    Plan,
    PlannerCheckpoint,
    Budget,
    SearchMetrics,
)
from services.metrics import metrics
# SearchContextSize import removed (unused)

# Optional: answer generation integration
try:
    from services.answer_generator import answer_orchestrator
    from models.synthesis_models import SynthesisContext as SynthesisContextModel
    _ANSWER_GEN_AVAILABLE = True
except Exception:
    _ANSWER_GEN_AVAILABLE = False
    answer_orchestrator = None  # type: ignore
    SynthesisContextModel = None  # type: ignore

logger = logging.getLogger(__name__)


def bernard_min_relevance() -> float:
    try:
        return float(os.getenv("SEARCH_MIN_RELEVANCE_BERNARD", "0.15"))
    except Exception:
        return 0.15


def default_source_limit() -> int:
    """Resolve the default source limit from env or use a sensible default.
    UPDATED FOR O3: Increased from 50 to 200 to analyze 4x more sources.
    This value is used anywhere a `source_limit` is missing from user context.
    """
    try:
        # UPDATED FOR O3: Changed default from 50 to 200
        v = int(os.getenv("DEFAULT_SOURCE_LIMIT", "200") or 200)
        return max(1, v)
    except Exception:
        return 200  # UPDATED FOR O3: Changed from 50 to 200


@dataclass
class ResearchExecutionResult:
    """Complete research execution result"""
    original_query: str
    paradigm: str
    secondary_paradigm: Optional[str]
    search_queries_executed: List[Dict[str, Any]]
    raw_results: Dict[str, List[SearchResult]]  # Results by API
    # Back-compat: keep original attribute and add a safe alias
    filtered_results: List[SearchResult]
    # Alias to avoid AttributeError in any consumer expecting 'results'
    # Note: property defined below to mirror filtered_results
    credibility_scores: Dict[str, float]  # Domain -> score
    # Non-default fields first
    execution_metrics: Dict[str, Any]
    cost_breakdown: Dict[str, float]
    # Defaults after non-defaults to satisfy dataclass rules
    status: ContractResearchStatus = ContractResearchStatus.OK
    secondary_results: List[SearchResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    deep_research_content: Optional[str] = None

    # Provide a read-only alias so obj.results returns filtered_results
    @property
    def results(self) -> List[SearchResult]:
        return self.filtered_results






class UnifiedResearchOrchestrator:
    """Unified Research Orchestrator combining all features"""

    def __init__(self):
        self.search_manager = None
        self.deduplicator = ResultDeduplicator()
        self.cost_monitor = CostMonitor()
        self.early_filter = EarlyRelevanceFilter()
        self.brave_enabled = False
        self.credibility_enabled = True
        self.research_store = research_store  # Add research store for cancellation checks

        # Orchestration additions
        self.tool_registry = ToolRegistry()
        self.retry_policy = RetryPolicy()
        self.planner = BudgetAwarePlanner(self.tool_registry, self.retry_policy)
        # Enforce that LLM-based synthesis must be available. If set to 1/true,
        # any failure to generate an answer will surface as a job failure rather
        # than silently degrading to evidence-only output. Default: on.
        try:
            self.require_synthesis = os.getenv("LLM_REQUIRED", "1").lower() in {"1", "true", "yes", "on"}
        except Exception:
            self.require_synthesis = True
        # Agentic loop configuration (can be overridden by env)
        self.agentic_config = {
            "enabled": True,
            # Enable LLM critic by default (can be disabled via env)
            "enable_llm_critic": True,
            "max_iterations": 2,
            "coverage_threshold": 0.75,
            "max_new_queries_per_iter": 4,
        }
        try:
            if os.getenv("AGENTIC_ENABLE_LLM_CRITIC", "0") == "1":
                self.agentic_config["enable_llm_critic"] = True
            if os.getenv("AGENTIC_ENABLE_LLM_CRITIC", "0") == "0" and "AGENTIC_ENABLE_LLM_CRITIC" in os.environ:
                self.agentic_config["enable_llm_critic"] = False
            if os.getenv("AGENTIC_DISABLE", "0") == "1":
                self.agentic_config["enabled"] = False
            # Optional tuning knobs
            if os.getenv("AGENTIC_MAX_ITERATIONS"):
                try:
                    self.agentic_config["max_iterations"] = int(os.getenv("AGENTIC_MAX_ITERATIONS", "2") or 2)
                except Exception:
                    pass
            if os.getenv("AGENTIC_COVERAGE_THRESHOLD"):
                try:
                    self.agentic_config["coverage_threshold"] = float(os.getenv("AGENTIC_COVERAGE_THRESHOLD", "0.75") or 0.75)
                except Exception:
                    pass
            if os.getenv("AGENTIC_MAX_NEW_QUERIES_PER_ITER"):
                try:
                    self.agentic_config["max_new_queries_per_iter"] = int(os.getenv("AGENTIC_MAX_NEW_QUERIES_PER_ITER", "4") or 4)
                except Exception:
                    pass
        except Exception:
            pass

        # Metrics
        self._search_metrics: SearchMetrics = {
            "total_queries": 0,
            "total_results": 0,
            "apis_used": [],
            "deduplication_rate": 0.0,
            # diagnostics
            "retries_attempted": 0,
            "task_timeouts": 0,
            "exceptions_by_api": {},
            "api_call_counts": {},
            "dropped_no_url": 0,
            "dropped_invalid_shape": 0,
            "compression_plural_used": 0,
            "compression_singular_used": 0,
        }
        self._diag_samples = {"no_url": []}
        self._supports_search_with_api = False
        # Capture deep research citations per research_id to merge into EvidenceBundle later
        self._deep_citations_map: Dict[str, List[Any]] = {}

        # Performance tracking (cap history to prevent unbounded memory growth)
        self.execution_history = deque(maxlen=100)

        # Diagnostics toggles
        self.diagnostics = {
            "log_task_creation": True,
            "log_result_normalization": True,
            "log_credibility_failures": True,
            "enforce_url_presence": True,
            "enforce_per_result_origin": True
        }

        # Per-search task timeout (seconds) for external API calls.
        # Ensure this exceeds the per-provider timeout so we don't cancel
        # the overall search while individual providers are still running.
        try:
            task_to = float(os.getenv("SEARCH_TASK_TIMEOUT_SEC", "30") or 30)
        except Exception:
            task_to = 30.0
        try:
            prov_to = float(os.getenv("SEARCH_PROVIDER_TIMEOUT_SEC", "25") or 25)
        except Exception:
            prov_to = 25.0
        # Add a small cushion above provider timeout
        self.search_task_timeout = max(task_to, prov_to + 5.0)

    # ──────────────────────────────────────────────────────────────────────
    # Small error/logging helper (applied in a few hot spots)
    # ──────────────────────────────────────────────────────────────────────
    def _log_and_continue(
        self,
        message: str,
        exc: Optional[Exception] = None,
        level: str = "warning",
    ) -> None:
        try:
            if exc:
                getattr(logger, level, logger.warning)(f"{message}: {exc}")
            else:
                getattr(logger, level, logger.warning)(message)
        except Exception:
            # Never raise from logging – absolute last resort
            logger.warning(message)

    # ──────────────────────────────────────────────────────────────────────
    # Metric coercion helpers
    # Ensure safe, typed extraction from heterogeneous _search_metrics dict
    # ──────────────────────────────────────────────────────────────────────
    def _safe_metric_int(self, key: str, default: int = 0) -> int:
        """Safely coerce metric to int for type checkers and runtime safety."""
        try:
            val = self._search_metrics.get(key, default)
            if isinstance(val, bool):
                return int(val)
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                sval = val.strip()
                if sval:
                    # Allow floats encoded as strings
                    return int(float(sval))
            return default
        except Exception:
            return default

    def _safe_metric_float(self, key: str, default: float = 0.0) -> float:
        """Safely coerce metric to float for type checkers and runtime safety."""
        try:
            val = self._search_metrics.get(key, default)
            if isinstance(val, bool):
                return float(val)
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                sval = val.strip()
                if sval:
                    return float(sval)
            return default
        except Exception:
            return default

    async def initialize(self):
        """Initialize the orchestrator. If a search_manager is already
        provided (e.g., injected by the application), reuse it instead of
        creating a new one to avoid duplicate provider sessions.
        """
        if getattr(self, "search_manager", None) and getattr(getattr(self, "search_manager"), "apis", None):
            logger.info("Reusing existing SearchAPIManager for orchestrator")
        else:
            self.search_manager = create_search_manager()
            # SearchAPIManager uses context manager protocol
            await self.search_manager.__aenter__()
        # Fail fast on empty provider set to avoid silent no-op searches
        try:
            if not getattr(self.search_manager, "apis", {}):
                error_msg = (
                    "CRITICAL: No search providers configured - Research pipeline initialization failed. "
                    "Required API keys are missing or invalid. "
                    "Please configure at least one of: "
                    "1) BRAVE_SEARCH_API_KEY for Brave Search "
                    "2) GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX for Google CSE "
                    "3) Academic providers (ArXiv, PubMed, Semantic Scholar). "
                    "You can disable this check by setting SEARCH_DISABLE_FAILFAST=1 if using deep-research-only mode."
                )
                logger.critical(error_msg)
                raise RuntimeError(error_msg)
        except RuntimeError:
            # Allow opt-out via env (useful for deep-research-only scenarios)
            import os as _os
            if _os.getenv("SEARCH_DISABLE_FAILFAST", "0") not in {"1", "true", "yes"}:
                logger.critical("Search provider initialization failed - exiting. Set SEARCH_DISABLE_FAILFAST=1 to bypass.")
                raise
        await cache_manager.initialize()
        await initialize_deep_research()
        # SearchAPIManager only supports search_all, not per-API search
        self._supports_search_with_api = False

        # Try to initialize Brave MCP
        try:
            from services.brave_mcp_integration import initialize_brave_mcp
            self.brave_enabled = await initialize_brave_mcp()
        except Exception as e:
            logger.warning(f"Brave MCP initialization failed: {e}")

        logger.info("✓ Unified Research Orchestrator V2 initialized")
        # Register default tool capabilities for budget tracking
        try:
            self.tool_registry.register(ToolCapability(name="google", cost_per_call_usd=0.005, rpm_limit=100, typical_latency_ms=800))
            self.tool_registry.register(ToolCapability(name="brave", cost_per_call_usd=0.0, rpm_limit=100, typical_latency_ms=600))
        except Exception:
            pass

    async def _record_search_telemetry(
        self,
        research_id: Optional[str],
        paradigm_code: str,
        user_context: Any,
        metrics_snapshot: SearchMetrics,
        cost_breakdown: Dict[str, float],
        processed_results: Dict[str, Any],
        executed_candidates: List[QueryCandidate],
    ) -> None:
        from services.telemetry_pipeline import telemetry_pipeline

        depth = "standard"
        try:
            depth_attr = getattr(user_context, "depth", None)
            if depth_attr:
                depth = str(depth_attr).lower()
        except Exception:
            pass

        try:
            apis_used_raw = metrics_snapshot.get("apis_used", [])  # type: ignore[call-arg]
        except Exception:
            apis_used_raw = []
        if isinstance(apis_used_raw, (list, tuple, set)):
            apis_used = [str(api).lower() for api in apis_used_raw if api]
        elif apis_used_raw:
            apis_used = [str(apis_used_raw).lower()]
        else:
            apis_used = []

        try:
            total_queries = int(metrics_snapshot.get("total_queries", 0))
        except Exception:
            total_queries = 0
        if not total_queries:
            total_queries = len(executed_candidates or [])

        try:
            total_results = int(metrics_snapshot.get("total_results", 0))
        except Exception:
            total_results = 0
        if not total_results:
            try:
                total_results = len(processed_results.get("results", []) or [])
            except Exception:
                total_results = 0

        try:
            dedup_val_raw = metrics_snapshot.get("deduplication_rate")
            dedup_value = float(dedup_val_raw) if isinstance(dedup_val_raw, (int, float)) else None
        except Exception:
            dedup_value = None

        try:
            processing_time = float(
                (processed_results.get("metadata", {}) or {}).get("processing_time_seconds", 0.0) or 0.0
            )
        except Exception:
            processing_time = 0.0

        provider_costs: Dict[str, float] = {}
        for provider, value in (cost_breakdown or {}).items():
            try:
                provider_costs[str(provider).lower()] = float(value)
            except Exception:
                continue

        stage_breakdown: Dict[str, int] = defaultdict(int)
        for candidate in executed_candidates or []:
            try:
                stage_breakdown[str(getattr(candidate, "stage", "unknown"))] += 1
            except Exception:
                stage_breakdown["unknown"] += 1

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "research_id": research_id,
            "paradigm": paradigm_code,
            "depth": depth,
            "total_queries": total_queries,
            "total_results": total_results,
            "apis_used": apis_used,
            "provider_costs": provider_costs,
            "processing_time_seconds": processing_time,
            "stage_breakdown": dict(stage_breakdown),
        }

        if dedup_value is not None:
            record["deduplication_rate"] = dedup_value

        total_cost = sum(provider_costs.values())
        if total_cost:
            record["total_cost_usd"] = total_cost

        await telemetry_pipeline.record_search_run(record)

    async def cleanup(self):
        """Cleanup resources"""
        if self.search_manager:
            # SearchAPIManager uses context manager protocol
            await self.search_manager.__aexit__(None, None, None)
            logger.info("Search manager cleaned up")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get current orchestrator capabilities"""
        return {
            "version": "2.0",
            "features": {
                # Multiple providers + concurrency imply non-determinism in ordering
                # and provider availability; report this accurately.
                "deterministic_results": False,
                "origin_tracking": True,
                "dynamic_compression": True,
                "full_context_preservation": True,
                "user_context_aware": True,
                "paradigm_optimization": True,
                "brave_mcp": self.brave_enabled,
                "deep_research": True,
                "early_relevance_filtering": True,
                "cost_monitoring": True
            }
        }


# Backwards-compatibility alias for legacy imports/tests
class ResearchOrchestrator(UnifiedResearchOrchestrator):
    pass

    async def execute_research(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
        enable_deep_research: bool = False,
        deep_research_mode: Optional[DeepResearchMode] = None,
        synthesize_answer: bool = False,
        answer_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute research (pure V2 flow; legacy path removed)"""

        async def check_cancelled():
            if not research_id:
                return False
            research_data = await self.research_store.get(research_id)
            return bool(research_data and research_data.get("status") == RuntimeResearchStatus.CANCELLED)

        start_time = datetime.now()
        # Use request‑scoped holders to avoid cross‑talk under concurrency
        cost_breakdown: Dict[str, float] = {}
        search_metrics_local: SearchMetrics = {
            "total_queries": 0,
            "total_results": 0,
            "apis_used": [],
            "deduplication_rate": 0.0,
            "retries_attempted": 0,
            "task_timeouts": 0,
            "exceptions_by_api": {},
            "api_call_counts": {},
            "dropped_no_url": 0,
            "dropped_invalid_shape": 0,
            "compression_plural_used": 0,
            "compression_singular_used": 0,
        }

        # Unified query planner path
        limited = self._get_query_limit(user_context)
        refined_queries = list(getattr(context_engineered, "refined_queries", []) or [])
        original_query = getattr(context_engineered, "original_query", "") or getattr(classification, "query", "")
        seed_query = refined_queries[0] if refined_queries else original_query
        if not seed_query:
            seed_query = getattr(classification, "query", "") or ""

        planner_cfg = self._build_planner_config(limited)
        paradigm_code = normalize_to_internal_code(getattr(classification, "primary_paradigm", HostParadigm.BERNARD))
        planner = QueryPlanner(planner_cfg)

        try:
            planned_candidates = await planner.initial_plan(
                seed_query=seed_query,
                paradigm=paradigm_code,
                additional_queries=refined_queries,
            )
        except Exception as e:
            logger.warning(f"Query planning failed; using fallback. {e}")
            fallback_query = seed_query or original_query or getattr(classification, "query", "")
            planned_candidates = [
                QueryCandidate(query=fallback_query, stage="rule_based", label="primary")
            ]

        if not planned_candidates:
            fallback_query = seed_query or original_query or getattr(classification, "query", "") or ""
            planned_candidates = [
                QueryCandidate(query=fallback_query, stage="rule_based", label="primary")
            ]

        if len(planned_candidates) > limited:
            planned_candidates = list(planned_candidates[:limited])

        logger.info(
            "Planned %d query candidates for %s",
            len(planned_candidates),
            getattr(classification.primary_paradigm, "value", classification.primary_paradigm),
        )

        executed_candidates: List[QueryCandidate] = list(planned_candidates)
        search_metrics_local["total_queries"] = int(search_metrics_local.get("total_queries", 0)) + len(executed_candidates)

        # Check for cancellation before executing searches
        if await check_cancelled():
            return {"status": "cancelled", "message": "Research was cancelled"}

        # Fail fast if search manager is missing or empty (misconfiguration)
        try:
            mgr = getattr(self, "search_manager", None)
            if not mgr or not getattr(mgr, "apis", {}):
                if progress_callback and research_id:
                    try:
                        await progress_callback.update_progress(
                            research_id,
                            phase="initialization",
                            message="No search providers available. Check API keys and provider flags.",
                            custom_data={"event": "no_search_providers"},
                        )
                    except Exception:
                        pass
                return {
                    "status": "error",
                    "message": "No search providers configured. Set BRAVE_SEARCH_API_KEY and/or GOOGLE_CSE_*; enable academic providers as needed.",
                }
        except Exception:
            pass

        # Execute searches with deterministic ordering
        search_results = await self._execute_searches_deterministic(
            executed_candidates,
            classification.primary_paradigm,
            user_context,
            progress_callback,
            research_id,
            check_cancelled,  # Pass cancellation check function
            cost_accumulator=cost_breakdown,
            metrics=search_metrics_local,
        )

        # Agentic follow-up loop based on coverage gaps
        if self.agentic_config.get("enabled", True):
            executed_queries = {cand.query for cand in executed_candidates}
            max_iters = max(0, int(self.agentic_config.get("max_iterations", 0)))
            coverage_threshold = float(self.agentic_config.get("coverage_threshold", 0.75))
            max_new_per_iter = max(0, int(self.agentic_config.get("max_new_queries_per_iter", 0)))

            coverage_sources = self._collect_sources_for_coverage(search_results)
            coverage_ratio, missing_terms = evaluate_coverage_from_sources(
                original_query,
                context_engineered,
                coverage_sources,
            )

            iteration = 0
            while (
                iteration < max_iters
                and coverage_ratio < coverage_threshold
                and missing_terms
            ):
                try:
                    followup_candidates = await planner.followups(
                        seed_query=seed_query,
                        paradigm=paradigm_code,
                        missing_terms=missing_terms,
                        coverage_sources=coverage_sources,
                    )
                except Exception as exc:
                    logger.debug("Planner follow-ups failed: %s", exc)
                    break

                followup_filtered: List[QueryCandidate] = []
                for cand in followup_candidates:
                    if cand.query in executed_queries:
                        continue
                    followup_filtered.append(cand)
                    if 0 < max_new_per_iter <= len(followup_filtered):
                        break

                if not followup_filtered:
                    break

                executed_candidates.extend(followup_filtered)
                executed_queries.update(cand.query for cand in followup_filtered)
                search_metrics_local["total_queries"] = int(search_metrics_local.get("total_queries", 0)) + len(followup_filtered)

                followup_results = await self._execute_searches_deterministic(
                    followup_filtered,
                    classification.primary_paradigm,
                    user_context,
                    progress_callback,
                    research_id,
                    check_cancelled,
                    cost_accumulator=cost_breakdown,
                    metrics=search_metrics_local,
                )

                for query, res in followup_results.items():
                    search_results[query] = res

                coverage_sources.extend(
                    self._collect_sources_for_coverage(followup_results)
                )
                coverage_ratio, missing_terms = evaluate_coverage_from_sources(
                    original_query,
                    context_engineered,
                    coverage_sources,
                )
                iteration += 1

        # Record per-query effectiveness (query -> results count)
        try:
            query_effectiveness = [
                {"query": q, "results": len(res or [])}
                for q, res in (search_results or {}).items()
            ]
        except Exception:
            query_effectiveness = []

        # Check for cancellation before processing results
        if await check_cancelled():
            return {"status": "cancelled", "message": "Research was cancelled"}

        # Process and enrich results
        processed_results = await self._process_results(
            search_results,
            classification,
            context_engineered,
            user_context,
            progress_callback,
            research_id,
            metrics=search_metrics_local,
        )

        await self._augment_with_exa_research(
            processed_results,
            classification,
            context_engineered,
            user_context,
            progress_callback,
            research_id,
        )

        # Check for cancellation before deep research
        if await check_cancelled():
            return {"status": "cancelled", "message": "Research was cancelled"}

        # Execute deep research if enabled
        if enable_deep_research:
            # Temporarily allow unlinked citations for deep research path
            prev_allow_unlinked = bool(self.diagnostics.get("allow_unlinked_citations", False))
            self.diagnostics["allow_unlinked_citations"] = True
            try:
                deep_results = await self._execute_deep_research_integration(
                    context_engineered,
                    classification,
                    user_context,
                    deep_research_mode or DeepResearchMode.PARADIGM_FOCUSED,
                    progress_callback,
                    research_id
                )
            finally:
                # Restore previous setting
                self.diagnostics["allow_unlinked_citations"] = prev_allow_unlinked

            if deep_results:
                # Avoid duplicates by URL against ranked results before appending deep results
                seen_urls = {
                    getattr(r, "url", None)
                    for r in processed_results.get("results", [])
                    if getattr(r, "url", None)
                }
                unique_deep: List[dict] = []
                for d in deep_results:
                    try:
                        u = (d.get("url") or "").strip()
                    except Exception:
                        u = ""
                    if u and u not in seen_urls:
                        unique_deep.append(d)
                processed_results["results"].extend(unique_deep)
                processed_results["metadata"]["deep_research_enabled"] = True
                # Update apis_used/sources_used to reflect deep results appended post‑processing
                try:
                    apis = set(processed_results.get("metadata", {}).get("apis_used", []) or [])
                except Exception:
                    apis = set()
                for d in deep_results:
                    try:
                        api = getattr(d, "source", None) or getattr(d, "source_api", None) or getattr(d, "search_api", None) or "deep_research"
                        if api:
                            apis.add(str(api))
                    except Exception:
                        continue
                if apis:
                    processed_results.setdefault("metadata", {})["apis_used"] = list(apis)
                try:
                    used = set(processed_results.get("sources_used", []) or [])
                except Exception:
                    used = set()
                for d in deep_results:
                    try:
                        used.add(getattr(d, "source", None) or getattr(d, "source_api", None) or getattr(d, "search_api", None) or "deep_research")
                    except Exception:
                        continue
                processed_results["sources_used"] = list(used)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        processed_results["metadata"]["processing_time"] = processing_time
        processed_results["metadata"]["processing_time_seconds"] = float(processing_time)

        # Compute UI-facing metrics that depend on the final result set
        credibility_summary = processed_results.setdefault("credibility_summary", {"average_score": 0.0})
        score_distribution = {"high": 0, "medium": 0, "low": 0}
        category_distribution: Dict[str, int] = {}
        total_sources_analyzed = 0
        high_quality_sources = 0

        def _score_band(value: Any) -> str:
            try:
                score = float(value if value is not None else 0.0)
            except Exception:
                score = 0.0
            if score >= 0.8:
                return "high"
            if score >= 0.6:
                return "medium"
            return "low"

        final_results = list(processed_results.get("results", []) or [])
        for result in final_results:
            adapter = ResultAdapter(result)
            score = adapter.credibility_score
            if score is None:
                try:
                    score = float(getattr(result, "credibility_score", 0.0) or 0.0)
                except Exception:
                    score = 0.0
            score = max(0.0, min(1.0, float(score or 0.0)))
            band = _score_band(score)
            score_distribution[band] += 1
            if score >= 0.8:
                high_quality_sources += 1
            total_sources_analyzed += 1

            raw_data = getattr(result, "raw_data", {}) or {}
            source_category = None
            if isinstance(raw_data, dict):
                source_category = raw_data.get("source_category")
            metadata = adapter.metadata or {}
            if not source_category and isinstance(metadata, dict):
                source_category = metadata.get("source_category")
            source_category = (str(source_category).strip().lower() or "general") if source_category else "general"
            category_distribution[source_category] = category_distribution.get(source_category, 0) + 1

        credibility_summary["score_distribution"] = score_distribution
        if total_sources_analyzed:
            credibility_summary["high_credibility_count"] = score_distribution["high"]
            credibility_summary["high_credibility_ratio"] = score_distribution["high"] / float(total_sources_analyzed)

        processed_results["metadata"]["total_sources_analyzed"] = total_sources_analyzed
        processed_results["metadata"]["high_quality_sources"] = high_quality_sources
        processed_results["metadata"]["category_distribution"] = category_distribution
        processed_results["metadata"].setdefault("bias_distribution", {})
        if progress_callback and research_id:
            try:
                await progress_callback.update_progress(
                    research_id,
                    sources_analyzed=total_sources_analyzed,
                    high_quality_sources=high_quality_sources,
                )
            except Exception:
                pass
        processed_results["metadata"].setdefault("agent_trace", [])

        normalized_sources: List[Dict[str, Any]] = []
        try:
            for result in final_results:
                adapter = ResultAdapter(result)
                url = (adapter.url or "").strip()
                if not url:
                    continue
                metadata = adapter.metadata or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                raw_data = getattr(result, "raw_data", {}) or {}
                if isinstance(raw_data, dict):
                    extracted = raw_data.get("extracted_meta")
                    if extracted and "extracted_meta" not in metadata:
                        metadata["extracted_meta"] = extracted
                    if "source_category" in raw_data and "source_category" not in metadata:
                        metadata["source_category"] = raw_data["source_category"]
                    if "credibility_explanation" in raw_data and "credibility_explanation" not in metadata:
                        metadata["credibility_explanation"] = raw_data["credibility_explanation"]
                credibility = adapter.credibility_score
                if credibility is None:
                    try:
                        credibility = float(metadata.get("credibility_score", 0.0) or 0.0)
                    except Exception:
                        credibility = 0.0
                credibility = max(0.0, min(1.0, float(credibility or 0.0)))
                published_date = metadata.get("published_date")
                if published_date is None:
                    published_date = getattr(result, "published_date", None)
                # Ensure JSON-serializable date
                try:
                    from datetime import date, datetime as _dt
                    if isinstance(published_date, (_dt, date)):
                        published_date = published_date.isoformat()
                except Exception:
                    pass
                result_type = metadata.get("result_type") or getattr(result, "result_type", "web")
                normalized_sources.append(
                    {
                        "title": adapter.title,
                        "url": url,
                        "snippet": adapter.snippet,
                        "content": adapter.content,
                        "domain": adapter.domain,
                        "credibility_score": credibility,
                        "published_date": published_date,
                        "result_type": result_type or "web",
                        "source_api": adapter.source_api,
                        "source_category": metadata.get("source_category"),
                        "metadata": metadata,
                    }
                )
        except Exception:
            normalized_sources = []

        # Optional answer synthesis (P2) with data-quality gating
        synthesized_answer = None
        if synthesize_answer:
            # Gate synthesis when there isn't enough or credible evidence
            try:
                min_results = int(os.getenv("MIN_RESULTS_FOR_SYNTHESIS", "3") or 3)
            except Exception:
                min_results = 3
            try:
                min_avg_cred = float(os.getenv("MIN_AVG_CREDIBILITY", "0.45") or 0.45)
            except Exception:
                min_avg_cred = 0.45

            try:
                results_count = len(processed_results.get("results", []) or [])
            except Exception:
                results_count = 0
            try:
                avg_cred = float((processed_results.get("credibility_summary") or {}).get("average_score", 0.0) or 0.0)
            except Exception:
                avg_cred = 0.0

            if results_count < min_results or avg_cred < min_avg_cred:
                md = processed_results.setdefault("metadata", {})
                md["insufficient_data"] = {
                    "reason": "insufficient_results" if results_count < min_results else "low_average_credibility",
                    "results_count": results_count,
                    "avg_credibility": avg_cred,
                    "thresholds": {"min_results": min_results, "min_avg_credibility": min_avg_cred},
                }
                synthesize_answer = False

        if synthesize_answer:
            try:
                # Build evidence quotes from processed results (typed EvidenceQuote)
                evidence_quotes: List[Any] = []
                evidence_bundle = None
                try:
                    from services.evidence_builder import build_evidence_bundle, quotes_to_plain_dicts

                    # Check if we have any sources to build evidence from
                    if not normalized_sources:
                        logger.warning(
                            f"Evidence builder skipped - No search results to process. "
                            f"Research ID: {research_id or 'N/A'}"
                        )
                        evidence_quotes = []
                        evidence_bundle = None
                        evidence_quotes_payload = []
                    else:
                        max_docs = min(
                            EVIDENCE_MAX_DOCS_DEFAULT,
                            int(getattr(user_context, "source_limit", default_source_limit())),
                        )

                        logger.info(
                            f"Building evidence bundle from {len(normalized_sources)} sources, "
                            f"max_docs={max_docs}, quotes_per_doc={EVIDENCE_QUOTES_PER_DOC_DEFAULT}"
                        )

                        evidence_bundle = await build_evidence_bundle(
                            getattr(context_engineered, "original_query", ""),
                            normalized_sources,
                            max_docs=max_docs,
                            quotes_per_doc=EVIDENCE_QUOTES_PER_DOC_DEFAULT,
                            include_full_content=True,
                            full_text_budget=EVIDENCE_BUDGET_TOKENS_DEFAULT,
                        )

                        evidence_quotes = list(getattr(evidence_bundle, "quotes", []) or [])
                        evidence_quotes_payload = quotes_to_plain_dicts(evidence_quotes) if evidence_quotes else []

                        # Log warning if evidence building produced no quotes
                        if not evidence_quotes and normalized_sources:
                            logger.warning(
                                f"Evidence builder produced no quotes from {len(normalized_sources)} sources. "
                                f"Possible causes: "
                                f"1) URL fetching failures (network issues, timeouts) "
                                f"2) Content extraction failures (JavaScript-rendered pages, PDFs) "
                                f"3) No relevant quotes found in fetched content. "
                                f"Research ID: {research_id or 'N/A'}"
                            )
                        else:
                            logger.info(
                                f"Evidence builder succeeded: {len(evidence_quotes)} quotes from "
                                f"{len(getattr(evidence_bundle, 'documents', []))} documents"
                            )

                        # Signal evidence phase completion to UI
                        if progress_callback and research_id:
                            try:
                                await progress_callback.update_progress(
                                    research_id,
                                    phase="evidence",
                                    progress=75,
                                )
                            except Exception:
                                pass

                except ImportError as e:
                    error_msg = (
                        f"CRITICAL: Evidence builder import failed - Cannot build evidence bundle. "
                        f"Error: {str(e)}. "
                        f"Missing dependencies or module not found. "
                        f"Research ID: {research_id or 'N/A'}"
                    )
                    logger.critical(error_msg)
                    evidence_quotes = []
                    evidence_bundle = None
                    evidence_quotes_payload = []
                    # Don't raise - allow synthesis to proceed without evidence

                except Exception as e:
                    error_msg = (
                        f"ERROR: Evidence builder failed - Evidence extraction unsuccessful. "
                        f"Error: {str(e)}. "
                        f"Possible causes: "
                        f"1) Network issues fetching URLs "
                        f"2) Invalid URL formats in search results "
                        f"3) Memory issues with large documents "
                        f"4) Timeout during content fetching. "
                        f"Research ID: {research_id or 'N/A'}, "
                        f"Sources attempted: {len(normalized_sources)}"
                    )
                    logger.error(error_msg)
                    logger.debug("Evidence builder traceback:", exc_info=True)
                    evidence_quotes = []
                    evidence_bundle = None
                    evidence_quotes_payload = []
                    # Don't raise - allow synthesis to proceed without evidence
                # Respect cancellation just before heavy LLM synthesis
                try:
                    if await check_cancelled():
                        return {"status": "cancelled", "message": "Research was cancelled"}
                except Exception:
                    pass

                # Attempt synthesis even if evidence building failed
                if evidence_bundle is None and processed_results.get("results"):
                    logger.warning(
                        f"Proceeding with answer synthesis without evidence bundle. "
                        f"Using {len(processed_results.get('results', []))} raw search results. "
                        f"Research ID: {research_id or 'N/A'}"
                    )

                logger.info(f"Starting answer synthesis for research_id: {research_id}")
                synthesized_answer = await self._synthesize_answer(
                    classification=classification,
                    context_engineered=context_engineered,
                    results=processed_results["results"],
                    research_id=research_id or f"research_{int(start_time.timestamp())}",
                    options=answer_options or {},
                    evidence_quotes=evidence_quotes_payload,
                    evidence_bundle=evidence_bundle,  # Can be None - synthesis should handle
                )
                # Honour cancellation immediately after synthesis returns
                try:
                    if await check_cancelled():
                        return {"status": "cancelled", "message": "Research was cancelled"}
                except Exception:
                    pass
                # Attach contradiction summary to answer metadata if available
                try:
                    contradictions = processed_results.get("contradictions", {})
                    if hasattr(synthesized_answer, "metadata") and isinstance(getattr(synthesized_answer, "metadata"), dict):
                        synthesized_answer.metadata.setdefault("signals", {})
                        synthesized_answer.metadata["signals"]["contradictions"] = contradictions
                except Exception:
                    pass
                try:
                    if evidence_bundle is not None:
                        metrics.record_o3_usage(
                            paradigm=normalize_to_internal_code(classification.primary_paradigm),
                            document_count=len(getattr(evidence_bundle, "documents", []) or []),
                            document_tokens=int(getattr(evidence_bundle, "documents_token_count", 0) or 0),
                            quote_count=len(evidence_quotes or []),
                            source_count=len(normalized_sources),
                            prompt_tokens=None,
                            completion_tokens=None,
                        )
                except Exception:
                    pass
            except asyncio.CancelledError:
                # Preserve cancellation semantics; do not misreport as LLM failure
                raise
            except Exception as e:
                error_msg = (
                    f"CRITICAL: LLM Answer synthesis failed - Cannot generate response. "
                    f"Error: {str(e)}. "
                    f"This typically indicates Azure OpenAI issues: "
                    f"1) Missing/invalid AZURE_OPENAI_API_KEY "
                    f"2) Incorrect AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_DEPLOYMENT "
                    f"3) Azure OpenAI service down/quota exceeded. "
                    f"Research ID: {research_id or 'N/A'}, Query: '{context_engineered.original_query[:100]}...' "
                    f"Search results collected: {len(normalized_sources)} sources"
                )
                logger.critical(error_msg)
                # If synthesis is required, escalate to hard failure so the
                # request never returns a partial result without an LLM answer.
                if getattr(self, "require_synthesis", True):
                    logger.critical(
                        f"Synthesis is required (require_synthesis=True) - Discarding {len(normalized_sources)} search results. "
                        f"Set LLM_REQUIRED=false to allow partial results without synthesis."
                    )
                    raise RuntimeError(error_msg) from e
                else:
                    logger.warning(
                        f"Continuing without synthesis (require_synthesis=False) - Returning {len(normalized_sources)} search results only."
                    )

        # Assemble contract-aligned response payload
        primary_paradigm = getattr(classification, "primary_paradigm", HostParadigm.BERNARD)
        primary_code = normalize_to_internal_code(primary_paradigm)
        secondary_paradigm = getattr(classification, "secondary_paradigm", None)
        secondary_code = normalize_to_internal_code(secondary_paradigm) if secondary_paradigm else None

        distribution_map: Dict[str, float] = {}
        secondary_confidence: Optional[float] = None
        raw_distribution = getattr(classification, "distribution", {}) or {}
        try:
            for host, value in raw_distribution.items():
                key = normalize_to_internal_code(host) if host is not None else str(host)
                distribution_map[key] = float(value or 0.0)
            if secondary_code and secondary_code in distribution_map:
                secondary_confidence = float(distribution_map.get(secondary_code, 0.0))
        except Exception:
            distribution_map = {}

        margin = 0.0
        if distribution_map:
            sorted_scores = sorted(distribution_map.values(), reverse=True)
            margin = float(sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0))

        paradigm_analysis: Dict[str, Any] = {
            "primary": {
                "paradigm": primary_code,
                "confidence": float(getattr(classification, "confidence", 0.0) or 0.0),
            }
        }
        if secondary_code:
            secondary_payload: Dict[str, Any] = {"paradigm": secondary_code}
            if secondary_confidence is not None:
                secondary_payload["confidence"] = secondary_confidence
            paradigm_analysis["secondary"] = secondary_payload

        integrated_synthesis: Optional[Dict[str, Any]] = None
        if synthesized_answer is not None:
            try:
                integrated_synthesis = {
                    "primary_answer": synthesized_answer,
                    "integrated_summary": getattr(synthesized_answer, "summary", ""),
                    "synergies": [],
                    "conflicts_identified": [],
                }
            except Exception:
                integrated_synthesis = None

        apis_used = list(search_metrics_local.get("apis_used", []))

        response = {
            "research_id": research_id or f"research_{int(start_time.timestamp())}",
            "query": getattr(context_engineered, "original_query", classification.query),
            "status": "ok",
            "results": processed_results["results"],
            "sources": normalized_sources,
            "paradigm_analysis": paradigm_analysis,
            "integrated_synthesis": integrated_synthesis,
            "metadata": {
                **processed_results["metadata"],
                "total_results": len(processed_results["results"]),
                "queries_executed": len(executed_candidates),
                "query_effectiveness": query_effectiveness,
                "sources_used": processed_results["sources_used"],
                "credibility_summary": processed_results["credibility_summary"],
                "deduplication_stats": processed_results["dedup_stats"],
                "contradictions": processed_results.get("contradictions", {"count": 0, "examples": []}),
                "processing_time_seconds": float(processing_time),
                "paradigm_fit": {
                    "primary": primary_code,
                    "confidence": float(getattr(classification, "confidence", 0.0) or 0.0),
                    "margin": margin,
                },
                "research_depth": "deep" if enable_deep_research else "standard",
                # Prepare safe metric values for type checking
                # Report request‑scoped metrics, not global cumulative values
                "search_metrics": {
                    "total_queries": int(search_metrics_local.get("total_queries", 0)),
                    "total_results": int(search_metrics_local.get("total_results", 0)),
                    "apis_used": apis_used,
                    "deduplication_rate": float(search_metrics_local.get("deduplication_rate", 0.0)),
                },
                "paradigm": primary_code,
                # Record LLM backend info so callers can verify model/endpoint used
                "llm_backend": self._llm_backend_info(),
            },
            "export_formats": {},
        }

        # Ensure "classification_details" is properly typed using ClassificationDetailsSchema
        try:
            md_obj = response.get("metadata")
            if not isinstance(md_obj, dict):
                md_obj = {}
                response["metadata"] = md_obj

            # Build ClassificationDetailsSchema for type safety
            from models.context_models import ClassificationDetailsSchema

            # Build distribution data with explicit typing
            distribution_data: Dict[str, float] = {}
            if distribution_map:
                for key, value in distribution_map.items():
                    distribution_data[key] = float(value)

            # Build reasoning data properly
            reasoning_data: Dict[str, List[str]] = {}
            reasoning_raw = getattr(classification, "reasoning", {}) or {}
            if isinstance(reasoning_raw, dict):
                for host, steps in reasoning_raw.items():
                    normalized_key = normalize_to_internal_code(host)
                    if steps:
                        steps_list = [str(s) for s in (list(steps) or [])[:4]]
                    else:
                        steps_list = []
                    reasoning_data[normalized_key] = steps_list

            # Create the schema object and convert to dict for storage
            classification_details = ClassificationDetailsSchema(
                distribution=distribution_data,
                reasoning=reasoning_data
            )
            md_obj["classification_details"] = classification_details.model_dump()
        except Exception:
            # Best-effort enrichment; skip on any unexpected shape issues
            pass

        try:
            rid = response["research_id"]
            response["export_formats"] = {
                "json": f"/v1/research/{rid}/export/json",
                "csv": f"/v1/research/{rid}/export/csv",
                "pdf": f"/v1/research/{rid}/export/pdf",
                "markdown": f"/v1/research/{rid}/export/markdown",
                "excel": f"/v1/research/{rid}/export/excel",
            }
        except Exception:
            response["export_formats"] = {}
        # Cost info (search API costs currently tracked; LLM costs not yet integrated)
        try:
            search_cost = sum(float(v) for v in (cost_breakdown or {}).values())
            response["cost_info"] = {
                "search_api_costs": round(search_cost, 4),
                "llm_costs": 0.0,
                "total": round(search_cost, 4),
                # keep backward compat with callers referencing total_cost
                "total_cost": round(search_cost, 4),
                "breakdown": cost_breakdown,
            }
        except Exception:
            response["cost_info"] = {"search_api_costs": 0.0, "llm_costs": 0.0, "total": 0.0, "total_cost": 0.0}
        try:
            await self._record_search_telemetry(
                research_id,
                primary_code,
                user_context,
                search_metrics_local,
                cost_breakdown,
                processed_results,
                executed_candidates,
            )
        except Exception:
            logger.debug("Failed to persist search telemetry", exc_info=True)
        if synthesized_answer is not None:
            # Preserve object for downstream normalizers (routes/research.py)
            response["answer"] = synthesized_answer
            try:
                metadata_obj = getattr(synthesized_answer, "metadata", {}) or {}
                evidence_quotes_copy = metadata_obj.get("evidence_quotes")
                if evidence_quotes_copy:
                    response["metadata"].setdefault("evidence_quotes", evidence_quotes_copy)
            except Exception:
                pass
        # Append execution record for stats (best-effort)
        try:
            # Build a minimal per-execution record
            cred_map: Dict[str, float] = {}
            try:
                for r in processed_results.get("results", []) or []:
                    dom = getattr(r, "domain", None)
                    sc = getattr(r, "credibility_score", None)
                    if dom and isinstance(sc, (int, float)):
                        cred_map[str(dom)] = float(sc)
            except Exception:
                cred_map = {}
            exec_rec = ResearchExecutionResult(
                original_query=getattr(context_engineered, "original_query", ""),
                paradigm=primary_code,
                secondary_paradigm=secondary_code,
                search_queries_executed=[
                    {
                        "query": cand.query,
                        "stage": cand.stage,
                        "label": cand.label,
                    }
                    for cand in executed_candidates
                ],
                raw_results=search_results if isinstance(search_results, dict) else {},
                filtered_results=processed_results.get("results", []) or [],
                credibility_scores=cred_map,
                execution_metrics={
                    "processing_time_seconds": float(processing_time),
                    "final_results_count": len(processed_results.get("results", []) or []),
                    "queries_executed": len(executed_candidates),
                },
                cost_breakdown=cost_breakdown,
                status=ContractResearchStatus.OK,
            )
            self.execution_history.append(exec_rec)
        except Exception:
            pass
        return response

    def _llm_backend_info(self) -> Dict[str, Any]:
        """Provide a small diagnostic snapshot of the active LLM backend."""
        try:
            from services.llm_client import llm_client
            return llm_client.get_active_backend_info()
        except Exception:
            return {"azure_enabled": False, "openai_enabled": False}

    async def _synthesize_answer(
        self,
        *,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        results: List[SearchResult],
        research_id: str,
        options: Dict[str, Any],
        evidence_quotes: Optional[List[Dict[str, Any]]] = None,
        evidence_bundle: Any | None = None,
    ) -> Any:
        """Build a SynthesisContext from research outputs and invoke answer generator."""
        logger.info(f"_synthesize_answer called for research_id: {research_id}, results count: {len(results)}")
        if not _ANSWER_GEN_AVAILABLE:
            error_msg = (
                "CRITICAL: Answer generation module not available - Cannot synthesize response. "
                "This indicates a system configuration issue: "
                "1) Missing answer_generator module or dependencies "
                "2) Import failure during module initialization "
                "3) Azure OpenAI client initialization failed. "
                f"Research ID: {research_id}, Query: '{context_engineered.original_query[:100]}...' "
                f"Available search results: {len(results)} that will be discarded"
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # Prepare minimal list[dict] for generator consumption.  We normalize all
        # result objects through ResultAdapter so downstream helpers receive a
        # consistent, dict-friendly payload (the evidence builder expects
        # `.get(...)` access and previously crashed when handed dataclasses).
        normalized_sources: List[Dict[str, Any]] = []
        for result in results or []:
            try:
                adapter = ResultAdapter(result)
                url = (adapter.url or "").strip()
                if not url:
                    continue

                metadata: Dict[str, Any] = adapter.metadata or {}
                if not isinstance(metadata, dict):
                    metadata = {}

                is_dict_result = isinstance(result, dict)

                if not is_dict_result:
                    try:
                        raw_data = getattr(result, "raw_data", {}) or {}
                        if isinstance(raw_data, dict):
                            extracted = raw_data.get("extracted_meta")
                            if extracted and "extracted_meta" not in metadata:
                                metadata["extracted_meta"] = extracted
                            for key in ("source_category", "credibility_explanation"):
                                if key in raw_data and key not in metadata:
                                    metadata[key] = raw_data[key]
                    except Exception:
                        pass

                credibility = adapter.credibility_score
                if credibility is None:
                    try:
                        credibility = float(metadata.get("credibility_score", 0.0) or 0.0)
                    except Exception:
                        credibility = 0.0

                published_date = metadata.get("published_date")
                if published_date is None and not is_dict_result:
                    published_date = getattr(result, "published_date", None)
                # Ensure JSON-serializable date
                try:
                    from datetime import date, datetime as _dt
                    if isinstance(published_date, (_dt, date)):
                        published_date = published_date.isoformat()
                except Exception:
                    pass

                result_type = metadata.get("result_type")
                if not result_type and not is_dict_result:
                    result_type = getattr(result, "result_type", "web")

                normalized_sources.append(
                    {
                        "title": adapter.title,
                        "url": url,
                        "snippet": adapter.snippet,
                        "content": adapter.content,
                        "domain": adapter.domain,
                        "credibility_score": float(credibility or 0.0),
                        "published_date": published_date,
                        "result_type": result_type or "web",
                        "source_api": adapter.source_api,
                        "metadata": metadata,
                    }
                )
            except Exception:
                logger.debug("[synthesis] Failed to normalize result", exc_info=True)
                continue

        sources: List[Dict[str, Any]] = normalized_sources

        # Best effort context_engineering dict
        try:
            ce = context_engineered.model_dump() if hasattr(context_engineered, "model_dump") else context_engineered.dict()
        except Exception:
            ce = {}

        # Isolation-only extraction: build findings strictly from Isolate layer patterns
        try:
            patterns = []
            iso = getattr(context_engineered, "isolate_output", None)
            if iso:
                raw = getattr(iso, "extraction_patterns", []) or []
                import re as _re
                for p in raw:
                    try:
                        patterns.append(_re.compile(p, _re.IGNORECASE))
                    except Exception:
                        continue
            findings: Dict[str, Any] = {"matches": [], "by_domain": {}, "focus_areas": getattr(getattr(context_engineered, "isolate_output", None), "focus_areas", [])}
            for r in sources:
                try:
                    text = " ".join([
                        (r.get("title") or ""),
                        (r.get("snippet") or ""),
                    ])
                    dom = (r.get("domain") or "").strip()
                except Exception:
                    continue
                matched = []
                for pat in patterns:
                    for m in pat.finditer(text):
                        frag = text[max(0, m.start()-80):m.end()+80]
                        matched.append(frag.strip())
                if matched:
                    findings["matches"].append({"domain": dom, "fragments": matched[:3]})
                    findings["by_domain"].setdefault(dom, 0)
                    findings["by_domain"][dom] += 1
            ce["isolated_findings"] = findings
            ce["isolation_only"] = True
        except Exception:
            ce.setdefault("isolation_only", True)

        # Normalize paradigm code for generator
        from models.paradigms import normalize_to_internal_code
        paradigm_code = normalize_to_internal_code(classification.primary_paradigm)

        # Predeclare evidence bundle holder for type-checkers
        eb = None  # will be optionally assigned below

        # Build synthesis context (typed EvidenceBundle)
        if SynthesisContextModel:
            try:
                from models.evidence import EvidenceBundle, EvidenceQuote, EvidenceMatch  # type: ignore
                # Normalize isolation findings into typed matches
                iso = (ce.get("isolated_findings", {}) or {}) if isinstance(ce, dict) else {}
                raw_matches = list(iso.get("matches", []) or [])

                def _ensure_match(val: Any) -> EvidenceMatch | None:
                    try:
                        if isinstance(val, EvidenceMatch):
                            return val
                        return EvidenceMatch.model_validate(val)
                    except Exception:
                        if isinstance(val, dict) and val.get("domain"):
                            fragments = val.get("fragments") or []
                            if isinstance(fragments, list):
                                return EvidenceMatch(domain=val.get("domain"), fragments=fragments)
                        return None

                matches_typed = []
                for m in raw_matches:
                    hm = _ensure_match(m)
                    if hm is not None:
                        matches_typed.append(hm)

                # Merge quotes from evidence_builder and optional deep research
                deep_q: List[Any] = []
                try:
                    if research_id and hasattr(self, "_deep_citations_map"):
                        deep_q = list(getattr(self, "_deep_citations_map", {}).get(research_id, []) or [])
                        # clear after use to avoid leaks across requests
                        if research_id in self._deep_citations_map:
                            self._deep_citations_map.pop(research_id, None)
                except Exception:
                    deep_q = []

                if evidence_bundle is not None:
                    try:
                        base_bundle = evidence_bundle if isinstance(evidence_bundle, EvidenceBundle) else EvidenceBundle.model_validate(evidence_bundle)
                    except Exception:
                        base_bundle = EvidenceBundle()
                else:
                    base_bundle = EvidenceBundle()

                existing_quotes = list(getattr(base_bundle, "quotes", []) or [])
                quotes_typed: List[EvidenceQuote] = []

                def _ensure_quote(val: Any) -> EvidenceQuote | None:
                    try:
                        if isinstance(val, EvidenceQuote):
                            return val
                        return EvidenceQuote.model_validate(val)
                    except Exception:
                        if isinstance(val, dict) and val.get("quote"):
                            return EvidenceQuote(
                                id=str(val.get("id", f"q{len(quotes_typed)+len(existing_quotes)+1:03d}")),
                                url=val.get("url", ""),
                                title=val.get("title", ""),
                                domain=val.get("domain", ""),
                                quote=val.get("quote", ""),
                                start=val.get("start"),
                                end=val.get("end"),
                                published_date=val.get("published_date"),
                                credibility_score=val.get("credibility_score"),
                                suspicious=val.get("suspicious"),
                                doc_summary=val.get("doc_summary"),
                                source_type=val.get("source_type"),
                            )
                        return None

                for item in existing_quotes:
                    q = _ensure_quote(item)
                    if q is not None:
                        quotes_typed.append(q)
                for item in evidence_quotes or []:
                    q = _ensure_quote(item)
                    if q is not None:
                        quotes_typed.append(q)
                for item in deep_q or []:
                    q = _ensure_quote(item)
                    if q is not None:
                        quotes_typed.append(q)

                dedup_quotes: Dict[str, EvidenceQuote] = {}
                for q in quotes_typed:
                    key = getattr(q, "id", None) or getattr(q, "url", None) or f"idx-{len(dedup_quotes)}"
                    key = str(key)
                    if key not in dedup_quotes:
                        dedup_quotes[key] = q
                quotes_typed = list(dedup_quotes.values())

                base_matches = list(getattr(base_bundle, "matches", []) or [])
                combined_matches: List[EvidenceMatch] = []
                for item in base_matches + matches_typed:
                    hm = _ensure_match(item)
                    if hm is not None:
                        combined_matches.append(hm)
                if combined_matches:
                    seen_match: set[tuple[str, tuple[str, ...]]] = set()
                    unique_matches: List[EvidenceMatch] = []
                    for m in combined_matches:
                        dom = getattr(m, "domain", "") or ""
                        fragments = tuple(getattr(m, "fragments", []) or [])
                        key = (dom, fragments)
                        if key in seen_match:
                            continue
                        seen_match.add(key)
                        unique_matches.append(m)
                    combined_matches = unique_matches

                base_by_domain = dict(getattr(base_bundle, "by_domain", {}) or {})
                for dom, count in (iso.get("by_domain", {}) or {}).items():
                    try:
                        base_by_domain[dom] = base_by_domain.get(dom, 0) + int(count)
                    except Exception:
                        continue

                focus_combined = list(dict.fromkeys((getattr(base_bundle, "focus_areas", []) or []) + list(iso.get("focus_areas", []) or [])))
                documents_combined = list(getattr(base_bundle, "documents", []) or [])
                documents_token_count = int(getattr(base_bundle, "documents_token_count", 0) or 0)

                eb = EvidenceBundle(
                    quotes=quotes_typed,
                    matches=combined_matches,
                    by_domain=base_by_domain,
                    focus_areas=focus_combined,
                    documents=documents_combined,
                    documents_token_count=documents_token_count,
                )
            except Exception:
                eb = None  # type: ignore
            # Validate only for diagnostics; guard to avoid crashing synthesis on validation errors
            try:
                _ = SynthesisContextModel(
                    query=context_engineered.original_query,
                    paradigm=paradigm_code,
                    search_results=sources,
                    context_engineering=ce,
                    max_length=int(options.get("max_length", SYNTHESIS_MAX_LENGTH_DEFAULT)),
                    include_citations=bool(options.get("include_citations", True)),
                    tone=str(options.get("tone", "professional")),
                    metadata={"research_id": research_id},
                    evidence_bundle=eb,
                )
            except Exception:
                # Continue with untyped payload; downstream generator accepts dict-based context
                pass

        # Emit synthesis started (mirrored as research_progress by tracker)
        try:
            from services.progress import progress as _pt
            if _pt:
                await _pt.report_synthesis_started(research_id)
        except Exception:
            pass

        # Optionally pass prompt variant override from research record
        try:
            rec = await self.research_store.get(research_id)
            pv = ((rec or {}).get("experiment") or {}).get("prompt_variant")
            if isinstance(pv, str) and pv in {"v1", "v2"}:
                options = {**options, "prompt_variant": pv}
        except Exception:
            pass

        # Cooperative cancellation just before the long LLM call
        try:
            if research_id:
                rec = await self.research_store.get(research_id)
                if rec and rec.get("status") == RuntimeResearchStatus.CANCELLED:
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

        # Invoke the answer orchestrator using the keyword-friendly signature
        assert answer_orchestrator is not None, "Answer generation not available"
        logger.info(f"Calling answer_orchestrator.generate_answer for research_id: {research_id}")
        answer = await answer_orchestrator.generate_answer(
            paradigm=paradigm_code,
            query=getattr(context_engineered, "original_query", ""),
            search_results=sources,
            context_engineering=ce,
            options={
                "research_id": research_id,
                **options,
                "evidence_quotes": evidence_quotes or [],
                "evidence_bundle": eb if 'eb' in locals() else None,
            },
        )

        # Cooperative cancellation checkpoint after LLM call returns
        try:
            if research_id:
                rec = await self.research_store.get(research_id)
                if rec and rec.get("status") == RuntimeResearchStatus.CANCELLED:
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

        # Attach evidence quotes and isolation findings summary to answer metadata
        # so the frontend can render dedicated panels without re-fetching.
        try:
            if hasattr(answer, "metadata") and isinstance(getattr(answer, "metadata"), dict):
                # Avoid accidental overwrites
                # Publish canonical bundle; legacy field retained only when absent
                if "evidence_bundle" not in answer.metadata and 'eb' in locals() and eb is not None:
                    try:
                        answer.metadata["evidence_bundle"] = eb.model_dump()
                    except Exception:
                        pass
                if "evidence_quotes" not in answer.metadata and (evidence_quotes or []):
                    # Temporary back-compat for frontends expecting raw quotes
                    try:
                        if all(isinstance(q, dict) for q in (evidence_quotes or [])):
                            answer.metadata["evidence_quotes"] = list(evidence_quotes)  # already plain
                        else:
                            from services.evidence_builder import quotes_to_plain_dicts
                            answer.metadata["evidence_quotes"] = quotes_to_plain_dicts(evidence_quotes)  # type: ignore[arg-type]
                    except Exception:
                        # Best-effort fallback: preserve dicts as-is, coerce objects via model_dump/__dict__
                        safe_quotes = []
                        for q in (evidence_quotes or []):
                            if isinstance(q, dict):
                                safe_quotes.append(q)
                            else:
                                try:
                                    safe_quotes.append(q.model_dump())  # type: ignore[attr-defined]
                                except Exception:
                                    safe_quotes.append(getattr(q, "__dict__", {}))
                        answer.metadata["evidence_quotes"] = safe_quotes
                # Summarize isolation findings if available in CE dict
                try:
                    if isinstance(ce, dict) and isinstance(ce.get("isolated_findings"), dict):
                        iso = ce.get("isolated_findings", {})
                        summary = {
                            "matches_count": len(iso.get("matches", []) or []),
                            "domains": list((iso.get("by_domain", {}) or {}).keys()),
                            "focus_areas": list(iso.get("focus_areas", []) or []),
                        }
                        answer.metadata.setdefault("signals", {})
                        answer.metadata["signals"]["isolated_findings"] = summary
                except Exception:
                    pass
        except Exception:
            pass

        # Emit synthesis completed with simple stats
        try:
            sections = 0
            citations = 0
            try:
                sections = len(getattr(answer, "sections", []) or [])
            except Exception:
                pass
            try:
                citations = len(getattr(answer, "citations", []) or [])
            except Exception:
                pass
            from services.progress import progress as _pt
            if _pt:
                await _pt.report_synthesis_completed(research_id, sections, citations)
        except Exception:
            pass

        logger.info(f"Answer generation completed for research_id: {research_id}")
        return answer


    def _get_query_limit(self, user_context: Any) -> int:
        """Get query limit based on user context"""
        # Allow environment overrides per role, else use more generous defaults
        defaults = {"FREE": 5, "BASIC": 8, "PRO": 15, "ENTERPRISE": 25, "ADMIN": 40}
        role = getattr(user_context, "role", "PRO")
        try:
            env_key = f"QUERY_LIMIT_{role}"
            if env_key in os.environ:
                return int(os.getenv(env_key, str(defaults.get(role, 5))) or defaults.get(role, 5))
        except Exception:
            pass
        return defaults.get(role, 5)

    def _build_planner_config(self, limit: int) -> PlannerConfig:
        base = PlannerConfig(max_candidates=max(1, limit))
        cfg = build_planner_config(base=base)

        # Orchestrator-specific overrides
        if os.getenv("SEARCH_DISABLE_AGENTIC", "0") == "1":
            cfg.enable_agentic = False

        context_cap = os.getenv("SEARCH_PLANNER_CONTEXT_CAP")
        if context_cap:
            try:
                cfg.per_stage_caps["context"] = max(0, int(context_cap))
            except Exception:
                pass

        return cfg

    async def _execute_searches_deterministic(
        self,
        planned_candidates: List[QueryCandidate],
        primary_paradigm: HostParadigm,
        user_context: Any,
        progress_callback: Optional[Any],
        research_id: Optional[str],
        check_cancelled: Optional[Any],
        *,
        cost_accumulator: Optional[Dict[str, float]] = None,
        metrics: Optional[SearchMetrics] = None,
    ) -> Dict[str, List[SearchResult]]:
        """Execute the planner's candidates deterministically and collect results."""
        all_results: Dict[str, List[SearchResult]] = {}
        if not planned_candidates:
            return all_results
        max_results = int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())
        paradigm_code = normalize_to_internal_code(primary_paradigm)
        min_rel = bernard_min_relevance() if paradigm_code == "bernard" else 0.25
        # Depth-aware tweaks: allow slightly lower threshold for DEEP to broaden recall
        try:
            depth = str(getattr(user_context, "depth", "")).lower()
            if depth == "deep":
                min_rel = max(0.05, min_rel - 0.05)
        except Exception:
            pass

        # Run per-query searches concurrently with a cap to avoid API bursts
        # Allow per-request concurrency override via user_context
        try:
            concurrency = int(getattr(user_context, "query_concurrency", 0) or 0)
        except Exception:
            concurrency = 0
        if concurrency <= 0:
            concurrency = int(os.getenv("SEARCH_QUERY_CONCURRENCY", "4"))
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _run_query(idx: int, candidate: QueryCandidate) -> Tuple[str, List[SearchResult]]:
            if check_cancelled and await check_cancelled():
                logger.info("Research cancelled before executing query #%d", idx + 1)
                return candidate.query, []
            # Honor per-request flag to disable real web search (frontend/user option)
            try:
                if hasattr(user_context, "enable_real_search") and not bool(getattr(user_context, "enable_real_search")):
                    logger.info("Real web search disabled by user context; skipping query #%d", idx + 1)
                    return candidate.query, []
            except Exception:
                pass
            # Progress (started)
            if progress_callback and research_id:
                try:
                    try:
                        await progress_callback.report_search_started(
                            research_id,
                            candidate.query,
                            f"{candidate.stage}:{candidate.label}",
                            idx + 1,
                            len(planned_candidates),
                        )
                    except Exception:
                        pass
                    await progress_callback.update_progress(
                        research_id,
                        phase="search",
                        message=f"Searching: {candidate.query[:40]}...",
                        progress=20 + int(((idx + 1) / max(1, len(planned_candidates))) * 35),
                        items_done=idx + 1,
                        items_total=len(planned_candidates),
                    )
                except Exception:
                    pass

            # Build SearchConfig using user context (language/region) when provided
            try:
                lang = str(getattr(user_context, "language", "en") or "en")
            except Exception:
                lang = "en"
            try:
                reg = str(getattr(user_context, "region", "us") or "us")
            except Exception:
                reg = "us"

            config = SearchConfig(
                max_results=max_results,
                language=lang,
                region=reg,
                min_relevance_score=min_rel,
            )

            async with sem:
                try:
                    results = await self._perform_search(
                        candidate,
                        config,
                        check_cancelled=check_cancelled,
                        progress_callback=progress_callback,
                        research_id=research_id,
                        metrics=metrics,
                    )
                except Exception as e:
                    logger.error("Search failed for '%s': %s", candidate.query, e)
                    results = []

            # Attribute cost only to providers observed in results (fallback: configured list)
            try:
                seen = {
                    (str(getattr(r, "source_api", None) or getattr(r, "source", "")).strip().lower())
                    for r in results
                }
                seen.discard("")
                if not seen:
                    mgr = getattr(self, "search_manager", None)
                    seen = set(getattr(mgr, "apis", {}).keys()) if mgr else set()
                for name in seen:
                    c = await self.cost_monitor.track_search_cost(name, 1)
                    if cost_accumulator is not None:
                        cost_accumulator[name] = float(cost_accumulator.get(name, 0.0)) + float(c)
            except Exception:
                logger.debug(
                    "Cost attribution failed for query '%s'",
                    candidate.query,
                    exc_info=True,
                )

            # Progress (completed + sample sources)
            if progress_callback and research_id:
                try:
                    await progress_callback.report_search_completed(
                        research_id,
                        candidate.query,
                        len(results),
                    )
                    for r in results[:3]:
                        try:
                            await progress_callback.report_source_found(
                                research_id,
                                {
                                    "title": getattr(r, "title", ""),
                                    "url": getattr(r, "url", ""),
                                    "domain": getattr(r, "domain", ""),
                                    "snippet": (getattr(r, "snippet", "") or "")[:200],
                                    "credibility_score": float(getattr(r, "credibility_score", 0.5) or 0.5),
                                },
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

            return candidate.query, results

        tasks = [
            asyncio.create_task(_run_query(idx, candidate))
            for idx, candidate in enumerate(planned_candidates)
        ]
        try:
            for fut in asyncio.as_completed(tasks):
                # Cooperative cancellation
                if check_cancelled and await check_cancelled():
                    pending = [t for t in tasks if not t.done()]
                    for t in pending:
                        t.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    break
                try:
                    q, res = await fut
                    all_results[q] = res
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Ensure a failed task does not block others
                    continue
        finally:
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return all_results

    def _collect_sources_for_coverage(
        self,
        batches: Dict[str, List[SearchResult]],
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for res_list in (batches or {}).values():
            for entry in res_list or []:
                adapter = ResultAdapter(entry)
                sources.append(
                    {
                        "title": adapter.title,
                        "snippet": adapter.snippet,
                        "url": adapter.url,
                        "domain": adapter.domain,
                    }
                )
        return sources

    async def _process_results(
        self,
        search_results: Dict[str, List[SearchResult]],
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
        *,
        metrics: Optional[SearchMetrics] = None,
    ) -> Dict[str, Any]:
        """
        Combine, validate, dedup, early-filter, rank, and score credibility.
        Returns a dict with keys: results, metadata, sources_used, credibility_summary, dedup_stats, contradictions.
        """
        # 1) Combine & basic validation/repairs (mirror legacy path)
        combined: List[SearchResult] = []
        to_backfill: List[Tuple[SearchResult, str]] = []
        for qkey, batch in (search_results or {}).items():
            if not batch:
                continue
            for r in batch:
                if r is None:
                    continue
                # Enforce URL presence when enabled; count drops separately
                url_val = getattr(r, "url", None)
                if not url_val or not isinstance(url_val, str) or not url_val.strip():
                    try:
                        if self.diagnostics.get("enforce_url_presence", True):
                            if metrics is not None:
                                metrics["dropped_no_url"] = int(metrics.get("dropped_no_url", 0)) + 1
                            if self._diag_samples is not None:
                                self._diag_samples.setdefault("no_url", []).append(getattr(r, "title", "") or "")
                            continue
                    except Exception:
                        pass
                # Minimal repair mirroring legacy behavior
                content = getattr(r, "content", "") or ""
                if not str(content).strip():
                    try:
                        sn = getattr(r, "snippet", "") or ""
                        tt = getattr(r, "title", "") or ""
                        if sn.strip():
                            r.content = f"Summary from search results: {sn.strip()}"
                        elif tt.strip():
                            r.content = tt.strip()
                    except Exception:
                        pass
                # Stage a second‑chance fetch for empty content; perform concurrently later
                if not str(getattr(r, "content", "") or "").strip() and getattr(r, "url", None):
                    to_backfill.append((r, getattr(r, "url")))
                # Require content non-empty after repair/fetch; log and surface drops
                if not str(getattr(r, "content", "") or "").strip():
                    try:
                        logger.debug("[process_results] Dropping empty-content result: %s", getattr(r, "url", ""))
                        if progress_callback and research_id:
                            try:
                                await progress_callback.update_progress(
                                    research_id,
                                    phase="filtering",
                                    message="Dropped result with empty content",
                                    custom_data={
                                        "event": "empty_content_drop",
                                        "url": getattr(r, "url", ""),
                                    },
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                    continue
                combined.append(r)

        # Concurrent backfill for staged empty-content items (bounded concurrency)
        if to_backfill:
            try:
                sm = getattr(self, "search_manager", None)
                if sm is not None:
                    session = sm._any_session()
                    limit = int(os.getenv("SEARCH_BACKFILL_CONCURRENCY", "8") or 8)
                    sem_fetch = asyncio.Semaphore(max(1, limit))

                    async def _fetch_and_set(item: Tuple[SearchResult, str]):
                        res, url = item
                        async with sem_fetch:
                            try:
                                fetched = await sm.fetcher.fetch(session, url)
                                if fetched and not str(getattr(res, "content", "") or "").strip():
                                    res.content = fetched
                            except Exception:
                                return

                    await asyncio.gather(*[_fetch_and_set(it) for it in to_backfill])
            except Exception:
                pass

        # 2) Dedup
        dedup = await self.deduplicator.deduplicate_results(combined)
        deduped: List[SearchResult] = list(dedup["unique_results"])

        # Update metrics (request-scoped when provided)
        if metrics is not None:
            metrics["total_results"] = len(combined)
            if combined:
                try:
                    metrics["deduplication_rate"] = 1.0 - (len(deduped) / len(combined))
                except Exception:
                    metrics["deduplication_rate"] = 0.0
        # Report deduplication stats to UI
        try:
            if progress_callback and research_id:
                await progress_callback.report_deduplication(
                    research_id,
                    before_count=len(combined),
                    after_count=len(deduped),
                )
        except Exception:
            pass

        # 3) Early filtering
        paradigm_code = normalize_to_internal_code(classification.primary_paradigm)
        early = self._apply_early_relevance_filter(
            deduped,
            getattr(context_engineered, "original_query", ""),
            paradigm_code,
        )

        # 4) Paradigm strategy ranking
        strategy = get_search_strategy(paradigm_code)

        # Safely normalize secondary paradigm only when present to satisfy type checker
        secondary_val = getattr(classification, "secondary_paradigm", None)
        secondary_code = (
            normalize_to_internal_code(secondary_val) if secondary_val is not None else None
        )

        search_ctx = SearchContext(
            original_query=getattr(context_engineered, "original_query", ""),
            paradigm=paradigm_code,
            secondary_paradigm=secondary_code,
        )
        ranked: List[SearchResult] = await strategy.filter_and_rank_results(early, search_ctx)

        # 5) Credibility on top-N (batched with concurrency)
        cred_summary = {"average_score": 0.0}
        creds: Dict[str, float] = {}
        user_cap = int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())
        topN = ranked[: min(max(20, user_cap), len(ranked))]

        # Deduplicate by domain, then fetch concurrently
        unique_domains: List[str] = []
        for r in topN:
            d = getattr(r, "domain", "")
            if d and d not in unique_domains:
                unique_domains.append(d)

        if progress_callback and research_id:
            try:
                await progress_callback.update_progress(
                    research_id,
                    phase="analysis",
                    message="Evaluating source credibility",
                    progress=55,
                    items_done=0,
                    items_total=len(unique_domains),
                )
            except Exception:
                pass

        conc = 8
        try:
            conc = int(os.getenv("CREDIBILITY_CONCURRENCY", "8") or 8)
        except Exception:
            pass
        semc = asyncio.Semaphore(max(1, conc))

        async def _fetch_dom(dom: str) -> Tuple[str, float]:
            async with semc:
                try:
                    score, _, _ = await self.get_source_credibility_safe(dom, paradigm_code)
                    return dom, float(score)
                except Exception:
                    return dom, 0.5

        domain_scores: Dict[str, float] = {}
        for idx, (dom, score) in enumerate(await asyncio.gather(*[_fetch_dom(d) for d in unique_domains])):
            domain_scores[dom] = score
            try:
                if progress_callback and research_id:
                    pct = 55 + int(((idx + 1) / max(1, len(unique_domains))) * 20)
                    await progress_callback.update_progress(
                        research_id,
                        phase="analysis",
                        progress=pct,
                        items_done=idx + 1,
                        items_total=len(unique_domains),
                    )
            except Exception:
                pass

        # Assign scores to results (preserve mapping)
        for r in topN:
            dom = getattr(r, "domain", "")
            sc = domain_scores.get(dom, None)
            if sc is not None:
                r.credibility_score = sc
                creds[dom] = sc

        if creds:
            try:
                cred_summary["average_score"] = sum(creds.values()) / float(len(creds))
            except Exception:
                cred_summary["average_score"] = 0.0

        # 6) Final limit by user context
        final_results: List[SearchResult] = ranked[: int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())]

        # Update apis_used from final results
        apis = set()
        for r in final_results:
            api = getattr(r, "source_api", getattr(r, "source", "unknown"))
            if api:
                apis.add(api)
        if apis:
            # Normalize to list for type stability (request‑scoped when provided)
            if metrics is not None:
                metrics["apis_used"] = list(apis)

        meta = {
            "processing_time": None,  # caller fills
            "paradigm": getattr(classification.primary_paradigm, "value", paradigm_code),
            "deep_research_enabled": False,
        }

        return {
            "results": final_results,
            "metadata": meta,
            "sources_used": list({getattr(r, "source", getattr(r, "source_api", "web")) for r in final_results}),
            "credibility_summary": cred_summary,
            "dedup_stats": {
                "original_count": len(combined),
                "final_count": len(final_results),
                "duplicates_removed": dedup.get("duplicates_removed", 0),
            },
            "contradictions": {"count": 0, "examples": []},
        }

    async def _augment_with_exa_research(
        self,
        processed_results: Dict[str, Any],
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any,
        progress_callback: Optional[Any],
        research_id: Optional[str],
    ) -> None:
        """Run Exa research to supplement Brave results when configured."""

        try:
            enabled = os.getenv("ENABLE_EXA_RESEARCH_SUPPLEMENT", "1").lower() in {"1", "true", "yes"}
        except Exception:
            enabled = True
        if not enabled:
            return

        if not exa_research_client.is_configured():
            return

        results = processed_results.get("results", []) or []
        if not results:
            return

        brave_highlights: List[Dict[str, str]] = []
        for res in results:
            try:
                source_name = (getattr(res, "source", "") or "").lower()
                if not source_name and hasattr(res, "source_api"):
                    source_name = (getattr(res, "source_api", "") or "").lower()
            except Exception:
                source_name = ""
            if "brave" not in source_name:
                continue
            brave_highlights.append(
                {
                    "title": str(getattr(res, "title", "") or ""),
                    "url": str(getattr(res, "url", "") or ""),
                    "snippet": str(getattr(res, "snippet", "") or getattr(res, "content", "") or ""),
                }
            )

        if not brave_highlights:
            return

        primary_paradigm = getattr(classification, "primary_paradigm", None)
        focus = self._focus_for_paradigm(primary_paradigm)

        original_query = getattr(context_engineered, "original_query", "") or getattr(user_context, "query", "")
        if not original_query:
            original_query = getattr(classification, "query", "")

        if not original_query:
            return

        if progress_callback and research_id:
            try:
                await progress_callback.update_progress(
                    research_id,
                    phase="analysis",
                    message="Running supplemental Exa research",
                )
            except Exception:
                pass

        exa_output = await exa_research_client.supplement_with_research(
            original_query,
            brave_highlights,
            focus=focus,
        )

        if not exa_output or not exa_output.summary:
            return

        supplemental_sources = [src for src in exa_output.supplemental_sources if src.get("url")]

        content_lines = [f"Summary: {exa_output.summary.strip()}"]
        if exa_output.key_findings:
            content_lines.append("Key Findings:")
            content_lines.extend([f"- {finding}" for finding in exa_output.key_findings])
        if supplemental_sources:
            content_lines.append("Supplemental Sources:")
            for src in supplemental_sources[:5]:
                title = src.get("title") or src.get("url")
                content_lines.append(f"* {title} ({src.get('url')})")

        pseudo_url = f"exa://research/{quote_plus(original_query[:80])}"
        supplement = SearchResult(
            title="Exa Research Synthesis",
            url=pseudo_url,
            snippet=exa_output.summary[:280],
            source="exa_research",
            content="\n".join(content_lines),
            relevance_score=0.85,
            credibility_score=0.65,
            raw_data={
                "key_findings": exa_output.key_findings,
                "supplemental_sources": supplemental_sources,
                "focus": focus,
            },
            result_type="research",
        )

        # Prepend so the synthesis step sees it early without displacing top Brave hits entirely
        processed_results["results"].insert(0, supplement)

        metadata = processed_results.setdefault("metadata", {})
        metadata["exa_research"] = {
            "summary": exa_output.summary,
            "key_findings": exa_output.key_findings,
            "supplemental_sources": supplemental_sources,
        }
        metadata.setdefault("apis_used", [])
        try:
            apis = set(metadata.get("apis_used", []) or [])
            apis.add("exa_research")
            metadata["apis_used"] = list(apis)
        except Exception:
            metadata["apis_used"] = list({"exa_research"})

        try:
            sources_used = set(processed_results.get("sources_used", []) or [])
            sources_used.add("exa_research")
            processed_results["sources_used"] = list(sources_used)
        except Exception:
            processed_results["sources_used"] = ["exa_research"]

        if progress_callback and research_id:
            try:
                await progress_callback.update_progress(
                    research_id,
                    phase="analysis",
                    message="Exa research augmentation complete",
                )
            except Exception:
                pass

    @staticmethod
    def _focus_for_paradigm(paradigm: Optional[HostParadigm]) -> Optional[str]:
        mapping = {
            HostParadigm.BERNARD: "Provide data-backed evidence, recent studies, and quantitative context.",
            HostParadigm.MAEVE: "Highlight strategic implications, competitive dynamics, and actionable levers.",
            HostParadigm.DOLORES: "Surface investigative findings, equity impacts, and accountability angles.",
            HostParadigm.TEDDY: "Emphasize community outcomes, practitioner guidance, and support resources.",
        }
        if paradigm in mapping:
            return mapping[paradigm]
        return None



    async def _perform_search(
        self,
        candidate: QueryCandidate,
        config: SearchConfig,
        check_cancelled: Optional[Any] = None,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
        *,
        metrics: Optional[SearchMetrics] = None,
    ) -> List[SearchResult]:
        """
        Unified single-search with retries/backoff and per-attempt timeout.
        Returns a list of SearchResult; empty list on failure.
        """
        sm = self.search_manager
        if sm is None:
            error_msg = (
                "CRITICAL: Search Manager not initialized - Research pipeline cannot proceed. "
                "This typically means required API keys are missing. "
                "Please ensure at least one of the following is configured: "
                "BRAVE_SEARCH_API_KEY, GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX, "
                "or academic search providers. "
                f"Research ID: {research_id or 'N/A'}, Query: '{candidate.query[:100]}...'"
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # Request-scoped metrics only (avoid shared-state mutation)
        api_name = "aggregate"
        if metrics is not None:
            mcounts = dict(metrics.get("api_call_counts", {}))
            try:
                if self.search_manager is not None:
                    for name in getattr(self.search_manager, "apis", {}).keys():
                        mcounts[name] = int(mcounts.get(name, 0)) + 1
                else:
                    mcounts[api_name] = int(mcounts.get(api_name, 0)) + 1
            except Exception:
                mcounts[api_name] = int(mcounts.get(api_name, 0)) + 1
            metrics["api_call_counts"] = mcounts

        attempt = 0
        delay = self.retry_policy.base_delay_sec
        results: List[SearchResult] = []

        while attempt < self.retry_policy.max_attempts:
            # Cancellation gate before each retry attempt
            if check_cancelled and await check_cancelled():
                logger.info(
                    "Search cancelled during retry phase for q='%s'",
                    candidate.query[:80],
                )
                break

            attempt += 1
            try:
                # Use search_all and emit per-provider progress updates
                coro = sm.search_all(
                    [candidate],
                    config,
                    progress_callback=progress_callback,
                    research_id=research_id,
                )
                task_result = await asyncio.wait_for(coro, timeout=self.search_task_timeout)
                if isinstance(task_result, list):
                    results = task_result
                else:
                    results = []
                break  # success
            except asyncio.TimeoutError:
                if metrics is not None:
                    metrics["task_timeouts"] = int(metrics.get("task_timeouts", 0)) + 1
                if attempt < self.retry_policy.max_attempts:
                    if metrics is not None:
                        metrics["retries_attempted"] = int(metrics.get("retries_attempted", 0)) + 1
                    await asyncio.sleep(min(delay, self.retry_policy.max_delay_sec))
                    delay = min(delay * 2.0, self.retry_policy.max_delay_sec)
                    continue
                else:
                    logger.error(
                        "Search task timeout q='%s'",
                        candidate.query[:120],
                    )
            except asyncio.CancelledError:
                # Preserve cooperative cancellation semantics
                raise
            except Exception as e:
                # Track exceptions by api (request-scoped)
                if metrics is not None:
                    mex = dict(metrics.get("exceptions_by_api", {}))
                    try:
                        if self.search_manager is not None:
                            for name in getattr(self.search_manager, "apis", {}).keys():
                                mex[name] = int(mex.get(name, 0)) + 1
                        else:
                            mex[api_name] = int(mex.get(api_name, 0)) + 1
                    except Exception:
                        mex[api_name] = int(mex.get(api_name, 0)) + 1
                    metrics["exceptions_by_api"] = mex
                if attempt < self.retry_policy.max_attempts:
                    if metrics is not None:
                        metrics["retries_attempted"] = int(metrics.get("retries_attempted", 0)) + 1
                    await asyncio.sleep(min(delay, self.retry_policy.max_delay_sec))
                    delay = min(delay * 2.0, self.retry_policy.max_delay_sec)
                    continue
                else:
                    logger.error(
                        "Search failed err=%s q='%s'",
                        str(e),
                        candidate.query[:120],
                    )

        return results

    def _apply_early_relevance_filter(
        self,
        results: List[SearchResult],
        query: str,
        paradigm: str
    ) -> List[SearchResult]:
        """Apply early-stage relevance filtering (single pass – theme overlap folded into EarlyRelevanceFilter)"""

        baseline: List[SearchResult] = []
        removed = 0
        for result in results:
            try:
                ok = self.early_filter.is_relevant(result, query, paradigm)
            except Exception:
                ok = True
            if ok:
                baseline.append(result)
            else:
                removed += 1
                logger.debug("[early] drop: %s - %s", getattr(result, "domain", ""), (getattr(result, "title", "") or "")[:60])

        if removed > 0:
            logger.info("Early filter removed %d results", removed)
        return baseline

    # [legacy normalizer removed]

    async def get_source_credibility_safe(self, domain: str, paradigm: str) -> Tuple[float, str, str]:
        """Get source credibility with non-blocking error handling"""
        try:
            if not self.credibility_enabled:
                return 0.5, "Credibility checking disabled", "skipped"

            credibility = await get_source_credibility(domain, paradigm)
            # Prefer structured to_dict if available; fall back to attribute
            explanation = ""
            if hasattr(credibility, "to_dict"):
                card = credibility.to_dict()
                explanation = (
                    f"bias={card.get('bias_rating')}, "
                    f"fact={card.get('fact_check_rating')}, "
                    f"cat={card.get('source_category')}"
                )
            elif hasattr(credibility, "explanation"):
                explanation = getattr(credibility, "explanation") or ""
            return getattr(credibility, "overall_score", 0.5), explanation, "success"
        except Exception as e:
            if self.diagnostics.get("log_credibility_failures", True):
                logger.warning("Credibility check failed for %s: %s", domain, str(e))
            return 0.5, f"Credibility check failed: {str(e)}", "failed"

    async def _execute_deep_research_integration(
        self,
        context_engineered: ContextEngineeredQuerySchema,
        classification: ClassificationResultSchema,
        user_context: Any,
        mode: DeepResearchMode,
        progress_callback: Optional[Any],
        research_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Execute deep research and convert to search results format"""

        if progress_callback and research_id:
            try:
                if hasattr(progress_callback, "update_progress"):
                    await progress_callback.update_progress(
                        research_id,
                        phase="deep_research",
                        message="Starting deep research analysis"
                    )
                elif callable(progress_callback):
                    await progress_callback("Starting deep research analysis")
            except Exception:
                pass

        deep_config = DeepResearchConfig(
            mode=mode,
            enable_web_search=True,
            enable_code_interpreter=False,
            background=True,
            include_paradigm_context=True,
        )

        # Propagate request-level options (user_location, search_context_size) when available
        try:
            if research_id:
                rec = await self.research_store.get(research_id)
                opts = (rec or {}).get("options") or {}
                # user_location: pass through as-is if dict
                ul = opts.get("user_location")
                if isinstance(ul, dict):
                    deep_config.user_location = ul
                # search_context_size: map small/medium/large -> LOW/MEDIUM/HIGH
                scs = str(opts.get("search_context_size") or "").lower().strip()
                from services.openai_responses_client import SearchContextSize as _SCS
                if scs in {"small", "low"}:
                    deep_config.search_context_size = _SCS.LOW
                elif scs in {"large", "high"}:
                    deep_config.search_context_size = _SCS.HIGH
                elif scs in {"medium"}:
                    deep_config.search_context_size = _SCS.MEDIUM
        except Exception:
            pass

        # Apply experiment override from research record when available
        try:
            if research_id:
                rec = await self.research_store.get(research_id)
                pv = ((rec or {}).get("experiment") or {}).get("prompt_variant")
                if isinstance(pv, str) and pv in {"v1", "v2"}:
                    deep_config.prompt_variant = pv
        except Exception:
            pass

        try:
            deep_result = await deep_research_service.execute_deep_research(
                query=context_engineered.original_query,
                classification=classification,           # type: ignore[arg-type]
                context_engineering=context_engineered,   # type: ignore[arg-type]
                config=deep_config,
                progress_tracker=progress_callback,
                research_id=research_id,
            )
        except Exception as e:
            error_msg = (
                f"CRITICAL: Deep Research failed - Enhanced research not available. "
                f"Error: {str(e)}. "
                f"This may indicate: "
                f"1) Azure OpenAI issues (missing API key, wrong endpoint/deployment) "
                f"2) Deep research service initialization failure "
                f"3) Token limit exceeded or rate limiting. "
                f"Research ID: {research_id or 'N/A'}, "
                f"Query: '{context_engineered.original_query[:100]}...' "
                f"Mode: {mode.value if mode else 'default'}"
            )
            logger.critical(error_msg)
            logger.critical(f"Deep Research traceback: ", exc_info=True)
            # Return empty list to allow fallback to standard search
            logger.warning("Falling back to standard search results without deep research enhancement")
            return []

        if not getattr(deep_result, "content", None):
            logger.warning(
                f"Deep Research returned no content - Empty result. "
                f"Research ID: {research_id or 'N/A'}, Query: '{context_engineered.original_query[:100]}...'"
            )
            return []

        deep_search_results = []
        # Convert deep research citations into typed EvidenceQuote and stash for bundle merge
        try:
            from services.deep_research_service import convert_citations_to_evidence_quotes
            evq = convert_citations_to_evidence_quotes(
                getattr(deep_result, "citations", []) or [],
                getattr(deep_result, "content", "") or "",
            )
            if research_id and evq:
                self._deep_citations_map[research_id] = evq
        except Exception:
            pass

        for citation in getattr(deep_result, "citations", []) or []:
            title = getattr(citation, "title", "") or ""
            url = getattr(citation, "url", "") or ""
            s = int(getattr(citation, "start_index", 0) or 0)
            e = int(getattr(citation, "end_index", s) or s)
            body = getattr(deep_result, "content", "") or ""
            snippet = body[s:e][:200] if isinstance(body, str) else ""

            # Do not synthesize URLs for unlinked citations. These are preserved
            # via the evidence bundle path and should not be emitted as results.
            if not url or not isinstance(url, str) or not url.strip():
                continue

            try:
                domain = url.split('/')[2] if ('/' in url and len(url.split('/')) > 2) else url
            except Exception:
                domain = ""
            deep_search_results.append({
                "title": title or (url.split('/')[-1] if url else "(deep research)"),
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "result_type": "deep_research",
                "credibility_score": 0.9,
                "origin_query": getattr(context_engineered, "original_query", ""),
                "search_api": "deep_research",
                "source_api": "deep_research",
                "source": "deep_research",
                "metadata": {}
            })

        return deep_search_results
    # [legacy → V2 adapter removed]


    async def execute_deep_research(
        self,
        context_engineered: ContextEngineeredQuerySchema,
        enable_standard_search: bool = True,  # ignored (legacy path removed)
        deep_research_mode: DeepResearchMode = DeepResearchMode.PARADIGM_FOCUSED,
        search_context_size: Optional[Any] = None,
        user_location: Optional[Dict[str, str]] = None,
        progress_tracker: Optional[Any] = None,
        research_id: Optional[str] = None,
    ) -> ResearchExecutionResult:
        """Run only the deep research path and return a minimal ResearchExecutionResult."""
        class _UserCtxShim:
            def __init__(self, role="PRO", source_limit: int = default_source_limit(), is_pro_user=True, preferences=None):
                self.role = role
                self.source_limit = source_limit
                self.is_pro_user = is_pro_user
                self.preferences = preferences or {}
        uc = getattr(context_engineered, "user_context", None)
        user_context = _UserCtxShim(
            role=getattr(uc, "role", "PRO") if uc else "PRO",
            source_limit=getattr(uc, "source_limit", default_source_limit()) if uc else default_source_limit(),
            is_pro_user=getattr(uc, "is_pro_user", True) if uc else True,
            preferences=getattr(uc, "preferences", {}) if uc else {},
        )

        classification = getattr(context_engineered, "classification", None)
        if classification is None:
            try:
                orig_q = getattr(context_engineered, "original_query", "")
                classification = ClassificationResultSchema(
                    query=orig_q,
                    primary_paradigm=HostParadigm.BERNARD,
                    secondary_paradigm=None,
                    distribution={HostParadigm.BERNARD: 1.0},
                    confidence=0.9,
                    features=QueryFeaturesSchema(text=orig_q),
                    reasoning={HostParadigm.BERNARD: ["default"]},
                )
            except Exception:
                class _C:
                    primary_paradigm = HostParadigm.BERNARD
                    secondary_paradigm = None
                    confidence = 0.9
                classification = _C()  # type: ignore

        deep_items = await self._execute_deep_research_integration(
            context_engineered=context_engineered,
            classification=classification,  # type: ignore[arg-type]
            user_context=user_context,
            mode=deep_research_mode,
            progress_callback=progress_tracker,
            research_id=research_id,
        )

        adapted: List[SearchResult] = []
        for d in deep_items or []:
            url = (d.get("url") or "").strip()
            domain = url.split("/")[2] if ("/" in url and len(url.split("/")) > 2) else url
            title = d.get("title") or (url.split("/")[-1] if url else "(deep research)")
            snippet = d.get("snippet") or d.get("summary") or ""
            src = d.get("source_api") or d.get("search_api") or "deep_research"
            try:
                adapted.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=src,
                        domain=domain,
                        content=snippet or title,
                    )
                )
            except Exception:
                continue

        final_results = adapted[: int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())]
        credibility_scores: Dict[str, float] = {}
        for r in final_results[:20]:
            try:
                score, _, _ = await self.get_source_credibility_safe(getattr(r, "domain", ""), "bernard")
                r.credibility_score = score
                if getattr(r, "domain", ""):
                    credibility_scores[getattr(r, "domain", "")] = score
            except Exception:
                pass

        return ResearchExecutionResult(
            original_query=getattr(context_engineered, "original_query", ""),
            paradigm=normalize_to_internal_code(getattr(classification, "primary_paradigm", HostParadigm.BERNARD)),
            secondary_paradigm=None,
            search_queries_executed=[],
            raw_results={"deep_research": adapted},
            filtered_results=final_results,
            secondary_results=[],
            credibility_scores=credibility_scores,
            execution_metrics={
                "queries_executed": 0,
                "raw_results_count": len(adapted),
                "deduplicated_count": len(adapted),
                "final_results_count": len(final_results),
                "deep_research_enabled": True,
            },
            cost_breakdown={"deep_research": 0.0},
            status=ContractResearchStatus.OK if final_results else ContractResearchStatus.FAILED_NO_SOURCES,
        )


    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"message": "No executions yet"}

        total_executions = len(self.execution_history)

        avg_processing_time = (
            sum(
                r.execution_metrics.get("processing_time_seconds", 0)
                for r in self.execution_history
            )
            / total_executions
        )

        paradigm_counts = {}
        for result in self.execution_history:
            paradigm_counts[result.paradigm] = (
                paradigm_counts.get(result.paradigm, 0) + 1
            )

        total_cost = sum(
            sum(result.cost_breakdown.values()) for result in self.execution_history
        )

        return {
            "total_executions": total_executions,
            "average_processing_time": round(avg_processing_time, 2),
            "paradigm_distribution": paradigm_counts,
            "total_cost": round(total_cost, 4),
            "cache_stats": await cache_manager.get_cache_stats(),
            "capabilities": self.get_capabilities()
        }


# Global orchestrator instance (use subclass with execute_research)
research_orchestrator = ResearchOrchestrator()

# Convenience functions for backward compatibility
async def initialize_research_system():
    """Initialize the complete research system"""
    await research_orchestrator.initialize()
    logger.info("Complete research system initialized")

# [legacy convenience execute_research removed]

# Alias for imports expecting different names
research_orchestrator_v2 = research_orchestrator
unified_orchestrator = research_orchestrator
enhanced_orchestrator = research_orchestrator

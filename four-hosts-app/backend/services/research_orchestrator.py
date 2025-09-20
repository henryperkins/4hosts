"""
Unified Research Orchestrator V2
Combines all the best features from multiple orchestrator implementations
"""

# Standard library imports
import asyncio
import os
import structlog
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

# Third-party imports
from dataclasses import dataclass, field

# Local imports - contracts and models
from contracts import ResearchStatus as ContractResearchStatus  # type: ignore
from models.base import ResearchStatus as RuntimeResearchStatus
from models.context_models import (
    ClassificationResultSchema,
    ContextEngineeredQuerySchema,
    HostParadigm
)
from models.paradigms import normalize_to_internal_code

# Local imports - services
from services.agentic_process import (
    evaluate_coverage_from_sources,
    run_followups,
)
from services.cache import cache_manager
from services.credibility import analyze_source_credibility_batch, get_source_credibility_safe
from services.exa_research import exa_research_client
from services.llm_client import llm_client
from services.paradigm_search import get_search_strategy, SearchContext, get_paradigm_focus
from services.query_planning import PlannerConfig, build_planner_config
from services.query_planning.planning_utils import (
    Budget,
    BudgetAwarePlanner,
    CostMonitor,
    Plan,
    PlannerCheckpoint,
    RetryPolicy,
    SearchMetrics,
    ToolCapability,
    ToolRegistry,
)
from services.query_planning.relevance_filter import EarlyRelevanceFilter
from services.query_planning.result_deduplicator import ResultDeduplicator
from services.research_store import research_store
from services.result_adapter import ResultAdapter
from services.result_normalizer import normalize_result, repair_and_filter_results
from services.search_apis import (
    SearchResult,
    SearchConfig,
    create_search_manager,
)

# Local imports - utilities
from utils.cost_attribution import attribute_search_costs
from utils.evidence_utils import deduplicate_evidence_quotes, deduplicate_evidence_matches, ensure_match
from utils.retry import instrumented_retry
from utils.type_coercion import as_int
from utils.url_utils import clean_url, normalize_url, extract_domain, canonicalize_url
from utils.source_normalization import normalize_source_fields, compute_category_distribution, extract_dedup_metrics, dedupe_by_url
from utils.date_utils import get_current_utc, get_current_iso
from utils.token_budget import select_items_within_budget, compute_budget_plan

# Local imports - core
from core.config import (
    EVIDENCE_BUDGET_TOKENS_DEFAULT,
    EVIDENCE_MAX_DOCS_DEFAULT,
    EVIDENCE_QUOTES_PER_DOC_DEFAULT,
    SYNTHESIS_MAX_LENGTH_DEFAULT,
)

# Local imports - search
from search.query_planner import QueryPlanner, QueryCandidate

# Optional: answer generation integration
try:
    from models.synthesis_models import SynthesisContext as SynthesisContextModel
    from services.answer_generator import answer_orchestrator
    _ANSWER_GEN_AVAILABLE = True
except Exception:
    _ANSWER_GEN_AVAILABLE = False
    SynthesisContextModel = None  # type: ignore
    answer_orchestrator = None  # type: ignore

# Optional: deep research integration (guarded)
try:
    from services.deep_research_service import (
        deep_research_service,
        DeepResearchMode,
        DeepResearchConfig,
    )
except Exception:
    deep_research_service = None  # type: ignore
    # Minimal stubs to avoid NameError when feature is disabled/missing
    class DeepResearchMode:  # type: ignore
        PARADIGM_FOCUSED = type("E", (), {"value": "PARADIGM_FOCUSED"})()
    DeepResearchConfig = None  # type: ignore

# Optional: global metrics facade (safe import)
try:
    from services.metrics_facade import metrics as _global_metrics
except Exception:
    _global_metrics = None

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Common utility functions to reduce code duplication
# ──────────────────────────────────────────────────────────────────────



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
    timestamp: datetime = field(default_factory=get_current_utc)
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
        # Initialize deep research if available
        try:
            from services.deep_research_service import initialize_deep_research
            await initialize_deep_research()
        except Exception:
            pass
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

    async def _record_telemetry(
        self,
        research_id: Optional[str],
        paradigm_code: str,
        user_context: Any,
        metrics_snapshot: SearchMetrics,
        cost_breakdown: Dict[str, float],
        processed_results: Dict[str, Any],
        executed_candidates: List[QueryCandidate],
        record_type: str = "search",  # "search" or "global"
    ) -> None:
        """Unified telemetry recording method for both search and global metrics."""
        from services.telemetry_pipeline import telemetry_pipeline

        # Common data extraction logic
        depth = "standard"
        try:
            depth_attr = getattr(user_context, "depth", None)
            if depth_attr:
                depth = str(depth_attr).lower()
        except Exception:
            pass

        try:
            apis_used_raw = metrics_snapshot.get("apis_used", [])
        except Exception:
            apis_used_raw = []
        if isinstance(apis_used_raw, (list, tuple, set)):
            apis_used = [str(api).lower() for api in apis_used_raw if api]
        elif apis_used_raw:
            apis_used = [str(apis_used_raw).lower()]
        else:
            apis_used = []

        total_queries = as_int(metrics_snapshot.get("total_queries"))
        if not total_queries:
            total_queries = len(executed_candidates or [])

        total_results = as_int(metrics_snapshot.get("total_results"))
        if not total_results:
            try:
                total_results = len(processed_results.get("results", []) or [])
            except Exception:
                total_results = 0

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

        # Base record for search telemetry
        record: Dict[str, Any] = {
            "timestamp": get_current_iso(),
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

        # Forward deduplication stats for centralized computation
        try:
            ds = processed_results.get("dedup_stats", {}) or {}
            if isinstance(ds, dict):
                record["dedup_stats"] = {
                    "original_count": as_int(ds.get("original_count")),
                    "final_count": as_int(ds.get("final_count")),
                    "duplicates_removed": as_int(ds.get("duplicates_removed")),
                }
        except Exception:
            pass

        total_cost = sum(provider_costs.values())
        if total_cost:
            record["total_cost_usd"] = total_cost

        # Record search telemetry
        if record_type == "search" or record_type == "both":
            await telemetry_pipeline.record_search_run(record)

        # Record global metrics (when requested and available)
        if (record_type == "global" or record_type == "both") and _global_metrics:
            try:
                # Get global metrics from the MetricsFacade
                latency_distributions = _global_metrics.get_latency_distributions()
                fallback_rates = _global_metrics.get_fallback_rates()
                llm_usage = _global_metrics.get_llm_usage()
                o3_usage = _global_metrics.get_o3_usage_summary()
                paradigm_distribution = _global_metrics.get_paradigm_distribution()

                # Record latency metrics to Prometheus
                for stage, percentiles in latency_distributions.items():
                    for percentile, value in percentiles.items():
                        if value > 0:
                            metric_name = f"research_stage_{stage}_latency_{percentile}"
                            logger.info(
                                "Global metrics snapshot",
                                research_id=research_id,
                                metric_name=metric_name,
                                stage=stage,
                                percentile=percentile,
                                value=value,
                                paradigm=paradigm_code
                            )

                # Record fallback rates
                for stage, rate in fallback_rates.items():
                    if rate > 0:
                        logger.info(
                            "Global fallback rate",
                            research_id=research_id,
                            stage=stage,
                            fallback_rate=rate,
                            paradigm=paradigm_code
                        )

                # Record LLM usage
                for model, usage in llm_usage.items():
                    if usage["calls"] > 0:
                        logger.info(
                            "Global LLM usage",
                            research_id=research_id,
                            model=model,
                            calls=usage["calls"],
                            tokens_in=usage["tokens_in"],
                            tokens_out=usage["tokens_out"],
                            paradigm=paradigm_code
                        )

                # Record O3 usage summary
                if o3_usage:
                    logger.info(
                        "Global O3 usage summary",
                        research_id=research_id,
                        events=o3_usage.get("events", 0),
                        avg_documents=o3_usage.get("avg_documents", 0),
                        avg_quotes=o3_usage.get("avg_quotes", 0),
                        total_document_tokens=o3_usage.get("total_document_tokens", 0),
                        paradigm=paradigm_code
                    )

                # Record paradigm distribution
                for paradigm, count in paradigm_distribution.items():
                    if count > 0:
                        logger.info(
                            "Global paradigm distribution",
                            research_id=research_id,
                            paradigm=paradigm,
                            count=count
                        )

                # Record cost information to global metrics
                if total_cost > 0:
                    logger.info(
                        "Global cost tracking",
                        research_id=research_id,
                        total_cost_usd=total_cost,
                        cost_breakdown=cost_breakdown,
                        paradigm=paradigm_code
                    )

                # Record search effectiveness metrics
                if total_queries > 0:
                    avg_results_per_query = total_results / total_queries
                    logger.info(
                        "Global search effectiveness",
                        research_id=research_id,
                        total_queries=total_queries,
                        total_results=total_results,
                        avg_results_per_query=avg_results_per_query,
                        paradigm=paradigm_code
                    )

                # Record deduplication effectiveness
                dedup_stats = processed_results.get("dedup_stats", {})
                if dedup_stats:
                    dedup_rate, counts = extract_dedup_metrics(dedup_stats)
                    logger.info(
                        "Global deduplication effectiveness",
                        research_id=research_id,
                        original_count=int(counts.get("original_count", 0)),
                        final_count=int(counts.get("final_count", 0)),
                        duplicates_removed=int(counts.get("duplicates_removed", 0)),
                        deduplication_rate_percent=float(dedup_rate),
                        paradigm=paradigm_code
                    )

                # Record credibility metrics
                credibility_summary = processed_results.get("credibility_summary", {})
                if credibility_summary:
                    avg_score = credibility_summary.get("average_score", 0)
                    high_credibility_ratio = credibility_summary.get("high_credibility_ratio", 0)

                    logger.info(
                        "Global credibility metrics",
                        research_id=research_id,
                        average_credibility_score=avg_score,
                        high_credibility_ratio=high_credibility_ratio,
                        paradigm=paradigm_code
                    )

            except Exception as e:
                logger.debug(f"Failed to record global metrics: {e}", exc_info=True)


# Backwards-compatibility alias for legacy imports/tests
class ResearchOrchestrator(UnifiedResearchOrchestrator):
    pass

    def _get_query_limit(self, user_context: Any) -> int:
        """Get query limit from user context or defaults"""
        try:
            return int(getattr(user_context, "query_limit", 5) or 5)
        except:
            return 5

    def _build_planner_config(self, query_limit: int) -> PlannerConfig:
        """Build planner configuration"""
        base_config = PlannerConfig(
            max_candidates=query_limit,
            enable_agentic=self.agentic_config.get("enabled", False),
        )
        return build_planner_config(base=base_config)

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
        logger.info(
            "Starting research execution",
            research_id=research_id,
            paradigm=getattr(classification, "primary_paradigm", "unknown"),
            original_query=getattr(context_engineered, "original_query", ""),
            enable_deep_research=enable_deep_research,
            synthesize_answer=synthesize_answer
        )

        async def check_cancelled():
            # Local cancellation check to avoid circular imports
            if not research_id:
                return False
            research_data = await self.research_store.get(research_id)
            return bool(research_data and research_data.get("status") == RuntimeResearchStatus.CANCELLED)

        start_time = get_current_utc()
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
        logger.info("Initializing query planning", research_id=research_id)
        limited = self._get_query_limit(user_context)
        refined_queries = list(getattr(context_engineered, "refined_queries", []) or [])
        original_query = getattr(context_engineered, "original_query", "") or getattr(classification, "query", "")
        seed_query = refined_queries[0] if refined_queries else original_query
        if not seed_query:
            seed_query = getattr(classification, "query", "") or ""

        logger.info(
            "Query planning initialized",
            research_id=research_id,
            query_limit=limited,
            refined_queries_count=len(refined_queries),
            seed_query=seed_query[:100]
        )

        planner_cfg = self._build_planner_config(limited)
        paradigm_code = normalize_to_internal_code(getattr(classification, "primary_paradigm", HostParadigm.BERNARD))
        planner = QueryPlanner(planner_cfg)

        logger.info("Executing query planner", research_id=research_id, paradigm=paradigm_code)
        try:
            planned_candidates = await planner.initial_plan(
                seed_query=seed_query,
                paradigm=paradigm_code,
                additional_queries=refined_queries,
            )
            logger.info(
                "Query planning successful",
                research_id=research_id,
                candidates_count=len(planned_candidates)
            )
        except Exception as e:
            logger.error(
                "Query planning failed",
                research_id=research_id,
                error=str(e),
                exc_info=True
            )
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
            f"Planned {len(planned_candidates)} query candidates for research_id: {research_id}"
        )

        logger.info(
            "Planned %d query candidates for %s",
            len(planned_candidates),
            getattr(classification.primary_paradigm, "value", classification.primary_paradigm),
        )

        executed_candidates: List[QueryCandidate] = list(planned_candidates)
        search_metrics_local["total_queries"] = int(search_metrics_local.get("total_queries", 0)) + len(executed_candidates)

        # Check for cancellation before executing searches
        # Check cancellation inline
        if research_id:
            logger.info("Checking for cancellation", research_id=research_id)
            research_data = await self.research_store.get(research_id)
            if research_data and research_data.get("status") == RuntimeResearchStatus.CANCELLED:
                logger.warning("Research cancelled before search execution", research_id=research_id)
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

        # Execute searches with deterministic ordering and budget enforcement
        logger.info(
            "Starting search execution",
            research_id=research_id,
            candidates_count=len(executed_candidates)
        )
        search_results = await self._execute_searches_with_budget(
            executed_candidates,
            classification.primary_paradigm,
            user_context,
            progress_callback,
            research_id,
            check_cancelled,  # Pass cancellation check function
            cost_accumulator=cost_breakdown,
            metrics=search_metrics_local,
        )
        logger.info(
            "Search execution completed",
            research_id=research_id,
            results_count=sum(len(results) for results in search_results.values())
        )

        # Agentic follow-up loop based on coverage gaps (extracted)
        if self.agentic_config.get("enabled", True):
            logger.info("Starting agentic follow-up loop", research_id=research_id)
            executed_queries = {cand.query for cand in executed_candidates}
            max_iters = max(0, int(self.agentic_config.get("max_iterations", 0)))
            coverage_threshold = float(self.agentic_config.get("coverage_threshold", 0.75))
            max_new_per_iter = max(0, int(self.agentic_config.get("max_new_queries_per_iter", 0)))

            # Collect initial coverage sources
            coverage_sources: List[Dict[str, Any]] = []
            for res_list in (search_results or {}).values():
                for entry in res_list or []:
                    adapter = ResultAdapter(entry)
                    coverage_sources.append({
                        "title": adapter.title,
                        "snippet": adapter.snippet,
                        "url": adapter.url,
                        "domain": adapter.domain,
                    })

            # Budget helpers
            def _estimate_cost(_cand: QueryCandidate) -> float:
                try:
                    return float(self.planner.estimate_cost("google", 1))
                except Exception:
                    return 0.0

            async def _progress_budget_reached(limit: float) -> None:
                if progress_callback and research_id:
                    try:
                        await progress_callback.update_progress(
                            research_id,
                            phase="search",
                            message=f"Budget limit reached (${limit:.2f}), stopping follow-ups",
                            custom_data={"event": "budget_limit_reached", "limit": limit},
                        )
                    except Exception:
                        pass

            def _can_spend(estimated: float) -> bool:
                try:
                    max_cost_per_request = float(os.getenv("MAX_COST_PER_REQUEST_USD", "0.50"))
                except Exception:
                    max_cost_per_request = 0.50
                current_request_cost = sum(cost_breakdown.values())
                if current_request_cost + estimated > max_cost_per_request:
                    logger.warning(
                        "Budget limit reached for research_id=%s current=$%.3f followups=$%.3f limit=$%.2f",
                        research_id,
                        current_request_cost,
                        estimated,
                        max_cost_per_request,
                    )
                    # Fire-and-forget progress
                    asyncio.create_task(_progress_budget_reached(max_cost_per_request))
                    return False
                # Informational log for notable spend
                if estimated > 0.10:
                    logger.info(
                        "Follow-up queries estimated cost=$%.3f research_id=%s",
                        estimated,
                        research_id,
                    )
                return True

            async def _execute(_cands: List[QueryCandidate]) -> Dict[str, List[SearchResult]]:
                return await self._execute_searches_with_budget(
                    _cands,
                    classification.primary_paradigm,
                    user_context,
                    progress_callback,
                    research_id,
                    check_cancelled,
                    cost_accumulator=cost_breakdown,
                    metrics=search_metrics_local,
                )

            def _to_cov(res_list: List[SearchResult]) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for entry in res_list or []:
                    adapter = ResultAdapter(entry)
                    out.append({
                        "title": adapter.title,
                        "snippet": adapter.snippet,
                        "url": adapter.url,
                        "domain": adapter.domain,
                    })
                return out

            new_cands, follow_map, coverage_ratio, missing_terms, coverage_sources = await run_followups(
                original_query=original_query,
                context_engineered=context_engineered,
                paradigm_code=paradigm_code,
                planner=planner,
                seed_query=seed_query,
                executed_queries=executed_queries,
                coverage_sources=coverage_sources,
                max_iterations=max_iters,
                coverage_threshold=coverage_threshold,
                max_new_per_iter=max_new_per_iter,
                estimate_cost=_estimate_cost,
                can_spend=_can_spend,
                execute_candidates=_execute,
                to_coverage_sources=_to_cov,
                check_cancelled=check_cancelled,
            )

            # Integrate results and metrics
            executed_candidates.extend(new_cands)
            search_metrics_local["total_queries"] = int(search_metrics_local.get("total_queries", 0)) + len(new_cands)
            for query, res in (follow_map or {}).items():
                search_results[query] = res

        # Record per-query effectiveness (query -> results count)
        try:
            query_effectiveness = [
                {"query": q, "results": len(res or [])}
                for q, res in (search_results or {}).items()
            ]
        except Exception:
            query_effectiveness = []

        # Check for cancellation before processing results
        # Check cancellation inline
        if research_id:
            research_data = await self.research_store.get(research_id)
            if research_data and research_data.get("status") == RuntimeResearchStatus.CANCELLED:
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
        # Check cancellation inline
        if research_id:
            research_data = await self.research_store.get(research_id)
            if research_data and research_data.get("status") == RuntimeResearchStatus.CANCELLED:
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
                # Merge deep results with existing results and deduplicate by canonical URL
                try:
                    combined = list(processed_results.get("results", []) or []) + list(deep_results or [])
                    processed_results["results"] = dedupe_by_url(combined)
                except Exception:
                    # Fallback to original unique-deep logic if adapter-based dedupe fails
                    seen_urls: set[str] = set()
                    try:
                        for r in processed_results.get("results", []) or []:
                            ru = getattr(r, "url", None)
                            if ru:
                                canon = canonicalize_url(ru)
                                if canon:
                                    seen_urls.add(canon)
                    except Exception:
                        pass
                    unique_deep: List[dict] = []
                    for d in deep_results:
                        try:
                            u_raw = (d.get("url") or "").strip()
                            u_norm = canonicalize_url(u_raw) if u_raw else ""
                        except Exception:
                            u_norm = ""
                        if u_norm and u_norm not in seen_urls:
                            d["url"] = u_norm
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

        end_time = get_current_utc()
        processing_time = (end_time - start_time).total_seconds()

        # Store only the schema-compliant key; legacy "processing_time" removed
        processed_results["metadata"]["processing_time_seconds"] = float(processing_time)

        # Compute UI-facing metrics that depend on the final result set
        credibility_summary = processed_results.setdefault("credibility_summary", {"average_score": 0.0})
        final_results = list(processed_results.get("results", []) or [])

        # Aggregate credibility using the credibility service utilities
        total_sources_analyzed = 0
        high_quality_sources = 0
        category_distribution: Dict[str, int] = {}

        try:
            sources_for_analysis: List[Dict[str, Any]] = []
            for result in final_results:
                adapter = ResultAdapter(result)
                sources_for_analysis.append({
                    "domain": adapter.domain,
                    "credibility_score": float(adapter.credibility_score or 0.0),
                    "content": adapter.content,
                    "published_date": getattr(result, "published_date", None),
                    "search_terms": [getattr(context_engineered, "original_query", "")],
                })
            if sources_for_analysis:
                stats = await analyze_source_credibility_batch(
                    sources_for_analysis,
                    paradigm=normalize_to_internal_code(classification.primary_paradigm),
                )
                # Map credibility distribution into schema-conformant key
                credibility_summary["average_score"] = float(stats.get("average_credibility", 0.0) or 0.0)
                credibility_summary["score_distribution"] = stats.get("credibility_distribution", {})
                high_quality_sources = int(stats.get("high_credibility_sources", 0) or 0)
                total_sources_analyzed = int(stats.get("total_sources", len(final_results)) or len(final_results))
                if total_sources_analyzed > 0:
                    credibility_summary["high_credibility_ratio"] = high_quality_sources / float(total_sources_analyzed)
                else:
                    credibility_summary["high_credibility_ratio"] = 0.0
                # Explicitly surface the absolute count for UI consumers that expect it
                credibility_summary["high_credibility_count"] = high_quality_sources
            else:
                total_sources_analyzed = 0
                high_quality_sources = 0
                credibility_summary.setdefault("average_score", 0.0)
        except Exception:
            # Fallback: simple aggregation if credibility service unavailable
            total_sources_analyzed = len(final_results)
            high_quality_sources = 0
            for result in final_results:
                try:
                    sc = float(getattr(result, "credibility_score", 0.0) or 0.0)
                    if sc >= 0.8:
                        high_quality_sources += 1
                except Exception:
                    continue

            # Populate summary keys expected by the schema on fallback path as well
            credibility_summary.setdefault("score_distribution", {})
            credibility_summary["high_credibility_count"] = high_quality_sources
            if total_sources_analyzed > 0:
                credibility_summary["high_credibility_ratio"] = high_quality_sources / float(total_sources_analyzed)
            else:
                credibility_summary["high_credibility_ratio"] = 0.0

        # Category distribution will be computed from normalized_sources later using compute_category_distribution

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
                ns = normalize_source_fields(result)
                if ns.get("url"):
                    normalized_sources.append(ns)
        except Exception:
            normalized_sources = []

        # Compute category distribution from normalized sources and update metadata
        try:
            category_distribution, normalized_sources = compute_category_distribution(normalized_sources)
            processed_results["metadata"]["category_distribution"] = category_distribution
        except Exception:
            pass

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
                    from services.evidence_builder import build_evidence_pipeline, quotes_to_plain_dicts

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
                            int(getattr(user_context, "source_limit", int(os.getenv("DEFAULT_SOURCE_LIMIT", "200")))),
                        )

                        logger.info(
                            f"Building evidence bundle via pipeline from {len(normalized_sources)} sources, "
                            f"max_docs={max_docs}, quotes_per_doc={EVIDENCE_QUOTES_PER_DOC_DEFAULT}"
                        )

                        # Apply overall timeout for evidence building phase
                        evidence_timeout = int(os.getenv("EVIDENCE_BUILDER_OVERALL_TIMEOUT", "120"))

                        try:
                            evidence_bundle, evidence_quotes_payload = await asyncio.wait_for(
                                build_evidence_pipeline(
                                    getattr(context_engineered, "original_query", ""),
                                    normalized_sources,
                                    max_docs=max_docs,
                                    quotes_per_doc=EVIDENCE_QUOTES_PER_DOC_DEFAULT,
                                    include_full_content=True,
                                    full_text_budget=EVIDENCE_BUDGET_TOKENS_DEFAULT,
                                ),
                                timeout=evidence_timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"Evidence building timed out after {evidence_timeout} seconds")
                            # Create empty evidence bundle to continue with synthesis
                            evidence_bundle = None
                            evidence_quotes_payload = []

                        evidence_quotes = list(getattr(evidence_bundle, "quotes", []) or [])

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
                    # Check cancellation inline
                    if research_id:
                        research_data = await self.research_store.get(research_id)
                        if research_data and research_data.get("status") == RuntimeResearchStatus.CANCELLED:
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
                    if evidence_bundle is not None and _global_metrics:
                        _global_metrics.record_o3_usage(
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

        # Consolidate APIs used: prefer the list recorded in processed_results metadata
        apis_used: List[str]
        try:
            apis_used = list(processed_results.get("metadata", {}).get("apis_used", []))
        except Exception:
            apis_used = []

        # Fall back to query-time metrics if orchestrator path didn't record them yet
        if not apis_used:
            apis_used = list(search_metrics_local.get("apis_used", []))

        # Ensure the top-level metadata also carries the same canonical list
        if apis_used:
            processed_results.setdefault("metadata", {})["apis_used"] = apis_used

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
                "llm_backend": llm_client.get_active_backend_info(),
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
            await self._record_telemetry(
                research_id,
                primary_code,
                user_context,
                search_metrics_local,
                cost_breakdown,
                processed_results,
                executed_candidates,
                record_type="both",  # Record both search and global metrics
            )
        except Exception:
            logger.debug("Failed to persist telemetry", exc_info=True)
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
        logger.info(f"Finished research execution for research_id: {research_id}")
        return response


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
        logger.info(f"Starting answer synthesis for research_id: {research_id}")
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

                # Use centralized result normalizer
                normalized_result = normalize_result(
                    adapter=adapter,
                    url=url,
                    credibility=credibility,
                    metadata=metadata,
                    is_dict_result=is_dict_result,
                    result=result if not is_dict_result else None
                )
                normalized_sources.append(normalized_result)
            except Exception:
                logger.debug("[synthesis] Failed to normalize result", exc_info=True)
                continue

        # Dedupe and token-budget sources for synthesis input size control
        try:
            deduped_for_synthesis = dedupe_by_url(normalized_sources)
        except Exception:
            deduped_for_synthesis = list(normalized_sources)

        try:
            # Compute knowledge bucket for synthesis source list
            synth_total = int(os.getenv("SYNTHESIS_BUDGET_TOKENS", "8000") or 8000)
            budget_plan = compute_budget_plan(synth_total)
            knowledge_budget = int(budget_plan.get("knowledge", int(synth_total * 0.70)))
        except Exception:
            knowledge_budget = 6000

        try:
            sources, _used, _dropped = select_items_within_budget(
                deduped_for_synthesis,
                max_tokens=knowledge_budget,
            )
        except Exception:
            sources = list(deduped_for_synthesis)

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

                quotes_typed = deduplicate_evidence_quotes(quotes_typed)

                base_matches = list(getattr(base_bundle, "matches", []) or [])
                combined_matches: List[EvidenceMatch] = []
                for item in base_matches + matches_typed:
                    hm = _ensure_match(item)
                    if hm is not None:
                        combined_matches.append(hm)
                if combined_matches:
                    combined_matches = deduplicate_evidence_matches(combined_matches)

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

        logger.info(f"Finished answer synthesis for research_id: {research_id}")
        logger.info(f"Answer generation completed for research_id: {research_id}")
        return answer




    def _build_search_config(
        self,
        user_context: Any,
        primary_paradigm: HostParadigm,
        max_results: Optional[int] = None
    ) -> SearchConfig:
        """Build search configuration from user context and paradigm."""
        if max_results is None:
            default_limit = int(os.getenv("DEFAULT_SOURCE_LIMIT", "200"))
        max_results = int(getattr(user_context, "source_limit", default_limit) or default_limit)

        return SearchConfig(
            max_results=max_results,
            language=str(getattr(user_context, "language", "en") or "en"),
            region=str(getattr(user_context, "region", "us") or "us"),
            # Dynamically lower the relevance threshold to improve recall.
            # A high value (≥0.5) was routinely eliminating *all* candidate
            # results for some paradigms which in turn starved downstream
            # evidence-building and answer synthesis stages.  A more lenient
            # default strikes a better balance between quality and recall.
            min_relevance_score=0.35 if normalize_to_internal_code(primary_paradigm) == "bernard" else 0.15,
        )

    async def _execute_searches_with_budget(
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
        """Execute searches with planner-aware path and attribute costs via CostMonitor."""
        logger.info(
            "Starting search execution with budget enforcement",
            research_id=research_id,
            stage="search_start",
            planned_candidates_count=len(planned_candidates),
        )

        if not planned_candidates:
            return {}

        # Manager-level planned execution
        search_config = self._build_search_config(user_context, primary_paradigm)

        logger.info(
            "Executing search queries",
            research_id=research_id,
            queries=[c.query[:50] for c in planned_candidates[:3]]  # Log first 3 queries
        )
        try:
            results_by_label = await self.search_manager.search_with_plan(planned_candidates, search_config)
            logger.info(
                "Search queries completed",
                research_id=research_id,
                labels=list(results_by_label.keys()) if results_by_label else []
            )

            # Convert results (label -> results) back to query -> results
            all_results: Dict[str, List[SearchResult]] = {}
            for label, results in (results_by_label or {}).items():
                for candidate in planned_candidates:
                    if getattr(candidate, "label", "unknown") == label:
                        all_results[candidate.query] = results
                        break

            # Attribute provider costs per planned candidate
            if cost_accumulator is not None:
                for candidate in planned_candidates:
                    cand_label = getattr(candidate, "label", "unknown")
                    cand_results = results_by_label.get(cand_label, []) if results_by_label else []
                    await attribute_search_costs(
                        cand_results,
                        self.cost_monitor,
                        cost_accumulator,
                        self.search_manager
                    )

            logger.info(
                "Finished search execution with budget enforcement",
                research_id=research_id,
                stage="search_end",
                results_count=sum(len(v) for v in all_results.values()),
            )
            return all_results

        except Exception as e:
            logger.error("Search execution with budget failed: %s", str(e), research_id=research_id, exc_info=True)
            # Fallback to deterministic execution if planned path fails
            logger.warning("Falling back to deterministic search execution", research_id=research_id)
            return await self._execute_searches_deterministic(
                planned_candidates,
                primary_paradigm,
                user_context,
                progress_callback,
                research_id,
                check_cancelled,
                cost_accumulator=cost_accumulator,
                metrics=metrics,
            )

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
        logger.info("Starting search execution", research_id=research_id, stage="search_start", planned_candidates_count=len(planned_candidates))
        all_results: Dict[str, List[SearchResult]] = {}
        if not planned_candidates:
            return all_results
        default_limit = int(os.getenv("DEFAULT_SOURCE_LIMIT", "200"))
        max_results = int(getattr(user_context, "source_limit", default_limit) or default_limit)
        paradigm_code = normalize_to_internal_code(primary_paradigm)
        # Apply the same relaxed threshold that is now used in the budgeted
        # search path so that both execution flows behave consistently.
        min_rel = 0.35 if paradigm_code == "bernard" else 0.15
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

            # Attribute cost to providers observed in results
            await attribute_search_costs(
                results,
                self.cost_monitor,
                cost_accumulator,
                self.search_manager
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

            logger.info(f"Query {candidate.query} for research_id: {research_id} returned {len(results)} results.")
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

        logger.info("Finished search execution", research_id=research_id, stage="search_end", results_count=sum(len(v) for v in all_results.values()))
        return all_results


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
        logger.info("Starting processing of search results", research_id=research_id, stage="processing_start", results_count=sum(len(v) for v in search_results.values()))
        # 1) Combine & basic validation/repairs using shared utility
        combined, to_backfill = repair_and_filter_results(
            search_results,
            enforce_url_presence=self.diagnostics.get("enforce_url_presence", True),
            metrics=metrics,
            diag_samples=self._diag_samples,
            progress_callback=progress_callback,
            research_id=research_id
        )

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
        logger.info("Deduplicated results", research_id=research_id, stage="deduplication", original_count=len(combined), deduplicated_count=len(deduped), duplicates_removed=dedup.get("duplicates_removed", 0))

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
        # Apply early relevance filter inline
        early = []
        removed = 0
        for result in deduped:
            try:
                ok = self.early_filter.is_relevant(result, getattr(context_engineered, "original_query", ""), paradigm_code)
            except Exception:
                ok = True
            if ok:
                early.append(result)
            else:
                removed += 1
                logger.debug("[early] drop: %s - %s", getattr(result, "domain", ""), (getattr(result, "title", "") or "")[:60])
        if removed > 0:
            logger.info("Early filter removed %d results", removed)

        logger.info("Early relevance filtering", research_id=research_id, stage="early_relevance_filtering", original_count=len(deduped), filtered_count=len(early))

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

        logger.info("Ranking", research_id=research_id, stage="ranking", original_count=len(early), ranked_count=len(ranked))

        # 5) Credibility on top-N (batched with concurrency)
        cred_summary = {"average_score": 0.0}
        creds: Dict[str, float] = {}
        default_limit = int(os.getenv("DEFAULT_SOURCE_LIMIT", "200"))
        user_cap = int(getattr(user_context, "source_limit", default_limit) or default_limit)
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
                    score, _, _ = await get_source_credibility_safe(
                        dom, paradigm_code,
                        credibility_enabled=self.credibility_enabled,
                        log_failures=self.diagnostics.get("log_credibility_failures", True)
                    )
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

        logger.info("Credibility scoring", research_id=research_id, stage="credibility_scoring", scored_domains=len(creds), average_score=cred_summary["average_score"])

        # Emit additional distribution metrics for observability
        try:
            highs = sum(1 for v in creds.values() if v > 0.7)
            lows = sum(1 for v in creds.values() if v < 0.3)
            # Build a simple histogram with 0.1 bins
            buckets = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(0, 10)}
            for v in creds.values():
                idx = min(9, max(0, int(v * 10)))
                key = f"{idx/10:.1f}-{(idx+1)/10:.1f}"
                buckets[key] += 1
            logger.info(
                "Credibility analysis complete",
                stage="credibility_analysis",
                research_id=research_id,
                high_credibility_count=highs,
                low_credibility_count=lows,
                avg_score=cred_summary.get("average_score", 0.0),
                score_distribution=buckets,
            )
        except Exception:
            pass

        # 6) Final limit by user context
        default_limit = int(os.getenv("DEFAULT_SOURCE_LIMIT", "200"))
        final_results: List[SearchResult] = ranked[: int(getattr(user_context, "source_limit", default_limit) or default_limit)]

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
            "processing_time_seconds": None,  # caller fills
            "paradigm": getattr(classification.primary_paradigm, "value", paradigm_code),
            "deep_research_enabled": False,
        }

        logger.info("Finished processing of search results", research_id=research_id, stage="processing_end", final_results_count=len(final_results))
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
        focus = get_paradigm_focus(primary_paradigm)

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

        # Dedupe and budget Brave highlights before sending to EXA
        try:
            highlights_dedup = dedupe_by_url(brave_highlights)
        except Exception:
            highlights_dedup = list(brave_highlights)
        try:
            exa_total = int(os.getenv("EXA_HIGHLIGHTS_BUDGET_TOKENS", "4000") or 4000)
            exa_plan = compute_budget_plan(exa_total)
            exa_budget = int(exa_plan.get("knowledge", int(exa_total * 0.70)))
        except Exception:
            exa_budget = 3000
        try:
            highlights_budgeted, _u, _d = select_items_within_budget(
                highlights_dedup,
                max_tokens=exa_budget,
            )
        except Exception:
            highlights_budgeted = highlights_dedup

        exa_output = await exa_research_client.supplement_with_research(
            original_query,
            highlights_budgeted,
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

        # Track retry attempts handled via utils.retry.instrumented_retry

        async def _execute_search():
            """Inner function for retry_with_backoff"""
            # Cancellation check
            if check_cancelled and await check_cancelled():
                logger.info(
                    "Search cancelled during retry phase for q='%s'",
                    candidate.query[:80],
                )
                return []

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
                    return task_result
                else:
                    return []
            except asyncio.TimeoutError as e:
                logger.error(
                    f"Search task timeout for query: {candidate.query}, research_id: {research_id}"
                )
                raise e
            except asyncio.CancelledError:
                logger.error(
                    f"Search cancelled for query: {candidate.query}, research_id: {research_id}"
                )
                # Preserve cooperative cancellation semantics
                raise
            except Exception as e:
                logger.error(
                    f"Search failed for query: {candidate.query}, research_id: {research_id}, error: {e}"
                )
                raise e

        # Use instrumented retry with centralized backoff and metrics callbacks
        def _on_retry(attempt: int, error: Exception, delay: float, will_retry: bool) -> None:
            try:
                if metrics is not None and will_retry:
                    # retries attempted
                    metrics["retries_attempted"] = int(metrics.get("retries_attempted", 0)) + 1
                    # task timeouts
                    if isinstance(error, asyncio.TimeoutError):
                        metrics["task_timeouts"] = int(metrics.get("task_timeouts", 0)) + 1
                    # exceptions by API
                    mex = dict(metrics.get("exceptions_by_api", {}))
                    try:
                        if self.search_manager is not None:
                            for name in getattr(self.search_manager, "apis", {}).keys():
                                mex[name] = int(mex.get(name, 0)) + 1
                        else:
                            mex["aggregate"] = int(mex.get("aggregate", 0)) + 1
                    except Exception:
                        mex["aggregate"] = int(mex.get("aggregate", 0)) + 1
                    metrics["exceptions_by_api"] = mex
            except Exception:
                pass
            logger.info(
                "Search retry scheduled",
                research_id=research_id,
                query=candidate.query[:120],
                attempt=attempt,
                delay=delay,
                error=str(error),
            )

        def _on_give_up(attempt: int, error: Exception) -> None:
            logger.error(
                "Search failed after all retries",
                research_id=research_id,
                attempt=attempt,
                query=candidate.query[:120],
                error=str(error),
            )

        try:
            results = await instrumented_retry(
                _execute_search,
                max_attempts=self.retry_policy.max_attempts,
                base_delay=self.retry_policy.base_delay_sec,
                factor=2.0,
                max_delay=self.retry_policy.max_delay_sec,
                exceptions=(asyncio.TimeoutError, Exception),
                on_retry=_on_retry,
                on_give_up=_on_give_up,
            )
            return results
        except asyncio.CancelledError:
            # Re-raise CancelledError without wrapping
            raise
        except Exception as e:
            logger.error(
                "Search failed after all retries err=%s q='%s'",
                str(e),
                candidate.query[:120],
            )
            return []


    # [legacy normalizer removed]

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
            logger.error(error_msg, exc_info=True)
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
                domain = extract_domain(url)
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



# Global orchestrator instance (use subclass with execute_research)
research_orchestrator = ResearchOrchestrator()

# Convenience functions for backward compatibility
async def initialize_research_system():
    """Initialize the complete research system"""
    await research_orchestrator.initialize()
    logger.info("Complete research system initialized")

# [legacy convenience execute_research removed]

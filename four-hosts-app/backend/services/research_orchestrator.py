"""
Unified Research Orchestrator V2
Combines all the best features from multiple orchestrator implementations
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, deque
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
from services.credibility import get_source_credibility
from services.paradigm_search import get_search_strategy, SearchContext
from services.cache import (
    cache_manager,
)
from services.text_compression import query_compressor
from core.config import (
    EVIDENCE_MAX_DOCS_DEFAULT,
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
    """Resolve the default source limit from env or use a sensible default (50).
    This value is used anywhere a `source_limit` is missing from user context.
    """
    try:
        v = int(os.getenv("DEFAULT_SOURCE_LIMIT", "50") or 50)
        return max(1, v)
    except Exception:
        return 50


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


 


class ResultDeduplicator:
    """Removes duplicate search results using various similarity measures"""

    def __init__(self, similarity_threshold: float = 0.8):
        # Allow tuning via env; default to a less aggressive 0.9 to reduce
        # accidental drops of near-duplicate but distinct items.
        try:
            env_thr = os.getenv("DEDUP_SIMILARITY_THRESH")
            if env_thr is not None:
                similarity_threshold = float(env_thr)
        except Exception:
            pass
        self.similarity_threshold = similarity_threshold

    async def deduplicate_results(
        self, results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Remove duplicate results using URL and content similarity"""
        # Defensive guard – empty input
        if not results:
            return {
                "unique_results": [],
                "duplicates_removed": 0,
                "similarity_threshold": self.similarity_threshold,
            }

        unique_results: List[SearchResult] = []
        duplicates_removed = 0
        seen_urls: set[str] = set()

        # First pass: exact URL‐based deduplication.
        # Skip any items that do not expose a usable ``url`` attribute
        # to avoid unexpected ``AttributeError`` crashes that would halt
        # the entire research pipeline mid-progress (observed by UI
        # getting stuck at the *“Removing duplicate results…”* stage).
        url_deduplicated: List[SearchResult] = []
        for result in results:
            url = getattr(result, "url", None)

            # Verify a non-empty URL exists – otherwise drop the record
            if not url or not isinstance(url, str):
                logger.debug(
                    "[dedup] Dropping result without valid URL: %s", repr(result)[:100]
                )
                duplicates_removed += 1
                continue

            if url not in seen_urls:
                seen_urls.add(url)
                url_deduplicated.append(result)
            else:
                duplicates_removed += 1

        # Second pass: content-similarity deduplication.  Again, guard
        # against malformed results to prevent crashes.
        for result in url_deduplicated:
            is_duplicate = False

            for existing in unique_results:
                try:
                    similarity = self._calculate_content_similarity(result, existing)
                except Exception as e:
                    # Log and treat as non-duplicate so the pipeline can
                    # continue gracefully.
                    logger.debug("[dedup] Similarity calc failed: %s", e, exc_info=True)
                    similarity = 0.0

                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    break

            if not is_duplicate:
                unique_results.append(result)

        logger.info(
            "Deduplication: %s -> %s (%s duplicates removed)",
            len(results),
            len(unique_results),
            duplicates_removed,
        )

        return {
            "unique_results": unique_results,
            "duplicates_removed": duplicates_removed,
            "similarity_threshold": self.similarity_threshold,
        }

    def _calculate_content_similarity(
        self, result1: SearchResult, result2: SearchResult
    ) -> float:
        """Calculate similarity between two search results"""
        try:
            h1 = getattr(result1, "content_hash", None)
            h2 = getattr(result2, "content_hash", None)
            if h1 and h2 and isinstance(h1, str) and isinstance(h2, str) and h1 == h2:
                return 1.0
        except Exception:
            pass
        # Defensive extraction helpers – return empty set when value missing
        def words(text: Optional[str]) -> set[str]:
            if not text or not isinstance(text, str):
                return set()
            return set(text.lower().split())

        # Title similarity (Jaccard index)
        title1_words = words(getattr(result1, "title", ""))
        title2_words = words(getattr(result2, "title", ""))

        if not title1_words or not title2_words:
            title_similarity = 0.0
        else:
            intersection = len(title1_words.intersection(title2_words))
            union = len(title1_words.union(title2_words))
            title_similarity = intersection / union if union > 0 else 0.0

        # Domain similarity – ensure attributes exist
        domain1 = getattr(result1, "domain", "") or ""
        domain2 = getattr(result2, "domain", "") or ""
        domain_similarity = 1.0 if domain1 == domain2 and domain1 else 0.0

        # Snippet similarity
        snippet1_words = words(getattr(result1, "snippet", ""))
        snippet2_words = words(getattr(result2, "snippet", ""))

        if not snippet1_words or not snippet2_words:
            snippet_similarity = 0.0
        else:
            intersection = len(snippet1_words.intersection(snippet2_words))
            union = len(snippet1_words.union(snippet2_words))
            snippet_similarity = intersection / union if union > 0 else 0.0

        # Weighted combination
        overall_similarity = (
            title_similarity * 0.5
            + domain_similarity * 0.2
            + snippet_similarity * 0.3
        )

        return overall_similarity


class EarlyRelevanceFilter:
    """Early-stage relevance filtering to remove obviously irrelevant results"""

    def __init__(self):
        self.spam_indicators = {
            'viagra', 'cialis', 'casino', 'poker', 'lottery',
            'weight loss', 'get rich quick', 'work from home',
            'singles in your area', 'hot deals', 'limited time offer'
        }

        # Domain blocklist can be disabled via EARLY_FILTER_BLOCK_DOMAINS=0
        try:
            if os.getenv("EARLY_FILTER_BLOCK_DOMAINS", "1").lower() in {"1", "true", "yes", "on"}:
                self.low_quality_domains = {
                    'ezinearticles.com', 'articlesbase.com', 'squidoo.com',
                    'hubpages.com', 'buzzle.com', 'ehow.com'
                }
            else:
                self.low_quality_domains = set()
        except Exception:
            self.low_quality_domains = set()

    def is_relevant(self, result: SearchResult, query: str, paradigm: str) -> bool:
        """Check if a result meets minimum relevance criteria"""

        # Check for spam content
        title_val = (getattr(result, "title", "") or "")
        snippet_val = (getattr(result, "snippet", "") or "")
        combined_text = f"{title_val} {snippet_val}".lower()
        if any(spam in combined_text for spam in self.spam_indicators):
            return False

        # Check for low-quality domains
        if (getattr(result, "domain", "") or "").lower() in self.low_quality_domains:
            return False

        # Minimum content check
        if not isinstance(title_val, str) or len(title_val.strip()) < 10:
            return False
        if not isinstance(snippet_val, str) or len(snippet_val.strip()) < 20:
            return False

        # Language detection (basic check for non-English)
        non_ascii_count = sum(1 for c in combined_text if ord(c) > 127)
        if non_ascii_count > len(combined_text) * 0.3:
            return False

        # Query relevance check
        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        if query_terms:
            has_query_term = any(term in combined_text for term in query_terms)
            if not has_query_term:
                return False

        # Check for duplicate/mirror sites
        if self._is_likely_duplicate_site((getattr(result, "domain", "") or "")):
            return False

        # Paradigm-specific early filters
        if paradigm == "bernard" and getattr(result, "result_type", "web") == "web":
            authoritative_indicators = ['.edu', '.gov', 'journal', 'research', 'study', 'analysis',
                                      'technology', 'innovation', 'science', 'ieee', 'acm', 'mit',
                                      'stanford', 'harvard', 'arxiv', 'nature', 'springer']
            tech_indicators = ['ai', 'artificial intelligence', 'machine learning', 'deep learning',
                             'neural', 'algorithm', 'technology', 'computing', 'software', 'innovation']

            has_authority = any(indicator in (getattr(result, "domain", "") or "").lower() or indicator in combined_text for indicator in authoritative_indicators)
            has_tech_content = any(indicator in combined_text for indicator in tech_indicators)

            if not (has_authority or has_tech_content):
                academic_terms = ['methodology', 'hypothesis', 'conclusion', 'abstract', 'citation',
                                'analysis', 'framework', 'approach', 'technique', 'evaluation']
                if not any(term in combined_text for term in academic_terms):
                    return False

        # Folded-in theme overlap (Jaccard) filter
        try:
            thr = float(os.getenv("EARLY_THEME_OVERLAP_MIN", "0.08") or 0.08)
            q_terms = set(query_compressor.extract_keywords(query))
            if q_terms:
                import re as _re
                def _toks(text: str) -> set:
                    return set([t for t in _re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if len(t) > 2])
                rtoks = _toks(combined_text)
                if rtoks:
                    inter = len(q_terms & rtoks)
                    union = len(q_terms | rtoks)
                    jac = (inter / float(union)) if union else 0.0
                    if jac < thr:
                        return False
        except Exception:
            pass

        return True

    def _is_likely_duplicate_site(self, domain: str) -> bool:
        """Check if domain is likely a duplicate/mirror site"""
        duplicate_patterns = [
            r'.*-mirror\.', r'.*-cache\.', r'.*-proxy\.',
            r'.*\.mirror\.', r'.*\.cache\.', r'.*\.proxy\.',
            r'webcache\.', r'cached\.', r'.*\.cc$'
        ]

        for pattern in duplicate_patterns:
            if re.match(pattern, domain.lower()):
                return True

        return False


class CostMonitor:
    """Monitors and tracks API costs"""

    def __init__(self):
        self.cost_per_call = {
            "google": 0.005,  # $5 per 1000 queries
            "brave": 0.0,     # Free tier
            "arxiv": 0.0,     # Free
            "pubmed": 0.0,    # Free
        }

    async def track_search_cost(self, api_name: str, queries_count: int) -> float:
        """Track cost for search API calls"""
        cost = self.cost_per_call.get(api_name, 0.0) * queries_count
        await cache_manager.track_api_cost(api_name, cost, queries_count)
        return cost

    async def get_daily_costs(self, date: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get daily API costs"""
        return await cache_manager.get_daily_api_costs(date)


class RetryPolicy:
    """Standardized retry/backoff policy configuration."""
    def __init__(self, max_attempts: int = 3, base_delay_sec: float = 0.5, max_delay_sec: float = 8.0) -> None:
        self.max_attempts = max_attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec


@dataclass
class ToolCapability:
    name: str
    cost_per_call_usd: float = 0.0
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    typical_latency_ms: Optional[int] = None
    failure_types: List[str] = field(default_factory=list)
    healthy: bool = True
    last_health_check: Optional[datetime] = None


class ToolRegistry:
    """Registry for tool capabilities, costs, limits, and health status."""
    def __init__(self) -> None:
        self._tools: Dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        self._tools[capability.name] = capability

    def get(self, name: str) -> Optional[ToolCapability]:
        return self._tools.get(name)

    def list(self) -> List[ToolCapability]:
        return list(self._tools.values())

    def set_health(self, name: str, healthy: bool) -> None:
        cap = self._tools.get(name)
        if cap:
            cap.healthy = healthy
            cap.last_health_check = datetime.now()


@dataclass
class Budget:
    max_tokens: int
    max_cost_usd: float
    max_wallclock_minutes: int


@dataclass
class PlannerCheckpoint:
    name: str
    description: str
    done: bool = False


@dataclass
class Plan:
    objective: str
    checkpoints: List[PlannerCheckpoint]
    budget: Budget
    stop_conditions: Dict[str, Any] = field(default_factory=dict)
    consumed_cost_usd: float = 0.0
    consumed_tokens: int = 0
    started_at: datetime = field(default_factory=datetime.now)

    def can_spend(self, additional_cost_usd: float, additional_tokens: int) -> bool:
        within_cost = (self.consumed_cost_usd + additional_cost_usd) <= self.budget.max_cost_usd
        within_tokens = (self.consumed_tokens + additional_tokens) <= self.budget.max_tokens
        return within_cost and within_tokens

    def spend(self, cost_usd: float, tokens: int) -> None:
        self.consumed_cost_usd += max(0.0, cost_usd)
        self.consumed_tokens += max(0, tokens)


class BudgetAwarePlanner:
    """Simple budget-aware planner that selects tools and enforces spend."""
    def __init__(self, registry: ToolRegistry, retry_policy: Optional[RetryPolicy] = None) -> None:
        self.registry = registry
        self.retry_policy = retry_policy or RetryPolicy()

    def select_tools(self, preferred: List[str]) -> List[str]:
        tools: List[str] = []
        for name in preferred:
            cap = self.registry.get(name)
            if cap and cap.healthy:
                tools.append(name)
        return tools

    def estimate_cost(self, tool_name: str, calls: int = 1) -> float:
        cap = self.registry.get(tool_name)
        return (cap.cost_per_call_usd * calls) if cap else 0.0

    def record_tool_spend(self, plan: Plan, tool_name: str, calls: int, tokens: int = 0) -> bool:
        cost = self.estimate_cost(tool_name, calls)
        if not plan.can_spend(cost, tokens):
            return False
        plan.spend(cost, tokens)
        return True


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
        self._search_metrics = {
            "total_queries": 0,
            "total_results": 0,
            "apis_used": set(),
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

        # Per-search task timeout (seconds) for external API calls
        try:
            self.search_task_timeout = float(os.getenv("SEARCH_TASK_TIMEOUT_SEC", "20") or 20)
        except Exception:
            self.search_task_timeout = 20.0

    # ──────────────────────────────────────────────────────────────────────
    # Small error/logging helper (applied in a few hot spots)
    # ──────────────────────────────────────────────────────────────────────
    def _log_and_continue(self, message: str, exc: Optional[Exception] = None, level: str = "warning") -> None:
        try:
            if exc:
                getattr(logger, level, logger.warning)(f"{message}: {exc}")
            else:
                getattr(logger, level, logger.warning)(message)
        except Exception:
            # Never raise from logging – absolute last resort
            logger.warning(message)

    async def initialize(self):
        """Initialize the orchestrator"""
        self.search_manager = create_search_manager()
        # SearchAPIManager uses context manager protocol
        await self.search_manager.__aenter__()
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
                "deterministic_results": True,
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
        cost_breakdown: Dict[str, float] = {}

        # Pure V2 query planning
        limited = self._get_query_limit(user_context)
        source_queries = list(getattr(context_engineered, "refined_queries", []) or [])
        if not source_queries:
            source_queries = [getattr(context_engineered, "original_query", "")]
        try:
            optimized_queries = self._compress_and_dedup_queries(source_queries)[:limited]
        except Exception as e:
            logger.warning(f"Query compression failed; using fallback. {e}")
            optimized_queries = [getattr(context_engineered, "original_query", "")][:limited]

        logger.info(
            f"Optimized queries to {len(optimized_queries)} for {classification.primary_paradigm.value}"
        )

        # Check for cancellation before executing searches
        if await check_cancelled():
            return {"status": "cancelled", "message": "Research was cancelled"}

        # Execute searches with deterministic ordering
        search_results = await self._execute_searches_deterministic(
            optimized_queries,
            classification.primary_paradigm,
            user_context,
            progress_callback,
            research_id,
            check_cancelled  # Pass cancellation check function
        )

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
            research_id
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
                processed_results["results"].extend(deep_results)
                processed_results["metadata"]["deep_research_enabled"] = True

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        processed_results["metadata"]["processing_time"] = processing_time

        # Optional answer synthesis (P2)
        synthesized_answer = None
        if synthesize_answer:
            try:
                # Build evidence quotes from processed results
                evidence_quotes: List[Dict[str, Any]] = []
                try:
                    from services.evidence_builder import build_evidence_quotes
                    evidence_quotes = await build_evidence_quotes(
                        getattr(context_engineered, "original_query", ""),
                        processed_results["results"],
                        max_docs=min(EVIDENCE_MAX_DOCS_DEFAULT, int(getattr(user_context, "source_limit", default_source_limit()))),
                        quotes_per_doc=3,
                    )
                except Exception:
                    evidence_quotes = []
                synthesized_answer = await self._synthesize_answer(
                    classification=classification,
                    context_engineered=context_engineered,
                    results=processed_results["results"],
                    research_id=research_id or f"research_{int(start_time.timestamp())}",
                    options=answer_options or {},
                    evidence_quotes=evidence_quotes,
                )
                # Attach contradiction summary to answer metadata if available
                try:
                    contradictions = processed_results.get("contradictions", {})
                    if hasattr(synthesized_answer, "metadata") and isinstance(getattr(synthesized_answer, "metadata"), dict):
                        synthesized_answer.metadata.setdefault("signals", {})
                        synthesized_answer.metadata["signals"]["contradictions"] = contradictions
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Answer synthesis failed: {e}")
                # If synthesis is required, escalate to hard failure so the
                # request never returns a partial result without an LLM answer.
                if getattr(self, "require_synthesis", True):
                    raise

        # Assemble high-level response
        response = {
            "results": processed_results["results"],
            "metadata": {
                **processed_results["metadata"],
                "total_results": len(processed_results["results"]),
                "queries_executed": len(optimized_queries),
                "sources_used": processed_results["sources_used"],
                "credibility_summary": processed_results["credibility_summary"],
                "deduplication_stats": processed_results["dedup_stats"],
                "contradictions": processed_results.get("contradictions", {"count": 0, "examples": []}),
                "search_metrics": {
                    "total_queries": int(self._search_metrics.get("total_queries", 0) if isinstance(self._search_metrics.get("total_queries"), int) else 0),
                    "total_results": int(self._search_metrics.get("total_results", 0) if isinstance(self._search_metrics.get("total_results"), int) else 0),
                    "apis_used": list(self._search_metrics.get("apis_used", set())) if isinstance(self._search_metrics.get("apis_used"), set) else [],
                    "deduplication_rate": float(self._search_metrics.get("deduplication_rate", 0.0) if isinstance(self._search_metrics.get("deduplication_rate"), (int, float)) else 0.0),
                },
                "paradigm": classification.primary_paradigm.value if hasattr(classification, "primary_paradigm") else "unknown",
                # Record LLM backend info so callers can verify model/endpoint used
                "llm_backend": self._llm_backend_info(),
            },
        }
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
        if synthesized_answer is not None:
            response["answer"] = synthesized_answer
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
    ) -> Any:
        """Build a SynthesisContext from research outputs and invoke answer generator."""
        if not _ANSWER_GEN_AVAILABLE:
            raise RuntimeError("Answer generation not available")

        # Prepare minimal list[dict] for generator consumption
        sources: List[Dict[str, Any]] = []
        for r in results:
            try:
                if isinstance(r, dict):
                    sources.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", ""),
                        "domain": r.get("domain", "") or r.get("host", ""),
                        "credibility_score": float(r.get("credibility_score", 0.0) or 0.0),
                        "published_date": r.get("published_date"),
                        "result_type": r.get("result_type", "web"),
                    })
                else:
                    sources.append({
                        "title": getattr(r, "title", ""),
                        "url": getattr(r, "url", ""),
                        "snippet": getattr(r, "snippet", ""),
                        "domain": getattr(r, "domain", ""),
                        "credibility_score": float(getattr(r, "credibility_score", 0.0) or 0.0),
                        "published_date": getattr(r, "published_date", None),
                        "result_type": getattr(r, "result_type", "web"),
                    })
            except Exception:
                continue

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
            for r in results:
                try:
                    text = " ".join([
                        (r.get("title") if isinstance(r, dict) else getattr(r, "title", "")) or "",
                        (r.get("snippet") if isinstance(r, dict) else getattr(r, "snippet", "")) or "",
                    ])
                    dom = (r.get("domain") if isinstance(r, dict) else getattr(r, "domain", "")) or ""
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

        # Build synthesis context (not strictly required by the generator but useful for parity)
        if SynthesisContextModel:
            try:
                from models.context_models import EvidenceBundle  # type: ignore
                eb = EvidenceBundle(
                    matches=list((ce.get("isolated_findings", {}) or {}).get("matches", []) or []),
                    by_domain=dict((ce.get("isolated_findings", {}) or {}).get("by_domain", {}) or {}),
                    focus_areas=list((ce.get("isolated_findings", {}) or {}).get("focus_areas", []) or []),
                    quotes=evidence_quotes or [],
                )
            except Exception:
                eb = None  # type: ignore
            _ = SynthesisContextModel(
                query=context_engineered.original_query,
                paradigm=paradigm_code,
                search_results=sources,
                context_engineering=ce,
                max_length=int(options.get("max_length", SYNTHESIS_MAX_LENGTH_DEFAULT)),
                include_citations=bool(options.get("include_citations", True)),
                tone=str(options.get("tone", "professional")),
                metadata={"research_id": research_id},
                evidence_quotes=evidence_quotes or [],
                evidence_bundle=eb,
            )

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

        # Call the generator using the legacy signature for broad compatibility
        assert answer_orchestrator is not None, "Answer generation not available"
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

        # Attach evidence quotes and isolation findings summary to answer metadata
        # so the frontend can render dedicated panels without re-fetching.
        try:
            if hasattr(answer, "metadata") and isinstance(getattr(answer, "metadata"), dict):
                # Avoid accidental overwrites
                if "evidence_quotes" not in answer.metadata:
                    answer.metadata["evidence_quotes"] = evidence_quotes or []
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

    def _compress_and_dedup_queries(self, queries: List[str]) -> List[str]:
        optimized_list: List[str] = []
        seen_set: Set[str] = set()
        for q in queries or []:
            try:
                compressed = query_compressor.compress(q, preserve_keywords=True)
            except Exception:
                compressed = q
            val = (compressed or q or "").strip()
            if not val:
                continue
            if val not in seen_set:
                seen_set.add(val)
                optimized_list.append(val)
        return optimized_list

    async def _execute_searches_deterministic(
        self,
        optimized_queries: List[str],
        primary_paradigm: HostParadigm,
        user_context: Any,
        progress_callback: Optional[Any],
        research_id: Optional[str],
        check_cancelled: Optional[Any],
    ) -> Dict[str, List[SearchResult]]:
        """
        Deterministically execute searches for V2 path. Returns {query_key: [SearchResult, ...]}.
        """
        all_results: Dict[str, List[SearchResult]] = {}
        max_results = int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())
        paradigm_code = normalize_to_internal_code(primary_paradigm)
        min_rel = bernard_min_relevance() if paradigm_code == "bernard" else 0.25

        for idx, q in enumerate(optimized_queries):
            if check_cancelled and await check_cancelled():
                logger.info("Research cancelled before executing query #%d", idx + 1)
                break

            # Progress
            if progress_callback and research_id:
                try:
                    # Emit explicit search.started so UI can show total planned searches
                    try:
                        await progress_callback.report_search_started(
                            research_id,
                            q,
                            "aggregate",
                            idx + 1,
                            len(optimized_queries),
                        )
                    except Exception:
                        pass
                    await progress_callback.update_progress(
                        research_id,
                        phase="search",
                        message=f"Searching: {q[:40]}...",
                        progress=30 + int((idx / max(1, len(optimized_queries))) * 20),
                        items_done=idx + 1,
                        items_total=len(optimized_queries),
                    )
                except Exception:
                    pass

            # Basic search config for V2 path
            config = SearchConfig(
                max_results=max_results,
                language="en",
                region="us",
                min_relevance_score=min_rel,
            )

            try:
                # Prefer google for determinism; fall back is inside _perform_search()
                results = await self._perform_search(
                    q, "google", config, check_cancelled=check_cancelled
                )
                all_results[q] = results
                # Cost track
                try:
                    _ = await self.cost_monitor.track_search_cost("google", 1)
                except Exception:
                    pass

                if progress_callback and research_id:
                    try:
                        await progress_callback.report_search_completed(
                            research_id, q, len(results)
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
            except Exception as e:
                logger.error("Search failed for '%s': %s", q, e)
                all_results[q] = []

        return all_results

    async def _process_results(
        self,
        search_results: Dict[str, List[SearchResult]],
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Combine, validate, dedup, early-filter, rank, and score credibility.
        Returns a dict with keys: results, metadata, sources_used, credibility_summary, dedup_stats, contradictions.
        """
        # 1) Combine & basic validation/repairs (mirror legacy path)
        combined: List[SearchResult] = []
        for qkey, batch in (search_results or {}).items():
            if not batch:
                continue
            for r in batch:
                if r is None:
                    continue
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
                # Require content non-empty after repair
                if not str(getattr(r, "content", "") or "").strip():
                    continue
                combined.append(r)

        # 2) Dedup
        dedup = await self.deduplicator.deduplicate_results(combined)
        deduped: List[SearchResult] = list(dedup["unique_results"])

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

        # 5) Credibility on top-N
        cred_summary = {"average_score": 0.0}
        creds: Dict[str, float] = {}
        user_cap = int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())
        topN = ranked[: min(max(20, user_cap), len(ranked))]

        # Announce analysis phase with total for UI microbar/ETA
        if progress_callback and research_id:
            try:
                await progress_callback.update_progress(
                    research_id,
                    phase="analysis",
                    message="Evaluating source credibility",
                    progress=55,
                    items_done=0,
                    items_total=len(topN),
                )
            except Exception:
                pass
        for idx, r in enumerate(topN):
            try:
                score, _, _ = await self.get_source_credibility_safe(getattr(r, "domain", ""), paradigm_code)
                r.credibility_score = score
                creds[getattr(r, "domain", "")] = score
                # Stream credibility checks + analyzed counter to UI
                if progress_callback and research_id:
                    try:
                        await progress_callback.report_credibility_check(
                            research_id,
                            getattr(r, "domain", ""),
                            float(score or 0.0),
                        )
                    except Exception:
                        pass
                    try:
                        await progress_callback.report_source_analyzed(
                            research_id,
                            f"src-{idx}",
                            {"credibility": float(score or 0.0)},
                        )
                    except Exception:
                        pass
                    # Occasionally push determinate analysis progress
                    try:
                        if (idx + 1) == len(topN) or ((idx + 1) % 2 == 0):
                            pct = 55 + int(((idx + 1) / max(1, len(topN))) * 15)  # 55→70 during analysis
                            await progress_callback.update_progress(
                                research_id,
                                phase="analysis",
                                progress=pct,
                                items_done=idx + 1,
                                items_total=len(topN),
                            )
                        
                    except Exception:
                        pass
            except Exception:
                pass

        if creds:
            try:
                cred_summary["average_score"] = sum(creds.values()) / float(len(creds))
            except Exception:
                cred_summary["average_score"] = 0.0

        # 6) Final limit by user context
        final_results: List[SearchResult] = ranked[: int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())]

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

    

    async def _perform_search(
        self,
        query: str,
        api: str,
        config: SearchConfig,
        check_cancelled: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Unified single-search with retries/backoff and per-attempt timeout.
        Returns a list of SearchResult; empty list on failure.
        """
        sm = self.search_manager
        if sm is None:
            raise RuntimeError("search_manager is not initialized")

        # Metric: count intended API invocation
        api_counts = self._search_metrics.get("api_call_counts", {})
        api_counts[api] = int(api_counts.get(api, 0)) + 1
        self._search_metrics["api_call_counts"] = api_counts

        attempt = 0
        delay = self.retry_policy.base_delay_sec
        results: List[SearchResult] = []

        while attempt < self.retry_policy.max_attempts:
            # Cancellation gate before each retry attempt
            if check_cancelled and await check_cancelled():
                logger.info("Search cancelled during retry phase for api=%s q='%s'", api, query[:80])
                break

            attempt += 1
            try:
                # Just use search_all - SearchAPIManager doesn't support per-API search
                coro = sm.search_all(query, config)
                task_result = await asyncio.wait_for(coro, timeout=self.search_task_timeout)
                if isinstance(task_result, list):
                    results = task_result
                else:
                    results = []
                break  # success
            except asyncio.TimeoutError:
                self._search_metrics["task_timeouts"] = int(self._search_metrics.get("task_timeouts", 0)) + 1
                if attempt < self.retry_policy.max_attempts:
                    self._search_metrics["retries_attempted"] = int(self._search_metrics.get("retries_attempted", 0)) + 1
                    await asyncio.sleep(min(delay, self.retry_policy.max_delay_sec))
                    delay = min(delay * 2.0, self.retry_policy.max_delay_sec)
                    continue
                else:
                    logger.error("Search task timeout api=%s q='%s'", api, query[:120])
            except Exception as e:
                # Track exceptions by api
                ex_map = self._search_metrics.get("exceptions_by_api", {})
                ex_map[api] = int(ex_map.get(api, 0)) + 1
                self._search_metrics["exceptions_by_api"] = ex_map
                if attempt < self.retry_policy.max_attempts:
                    self._search_metrics["retries_attempted"] = int(self._search_metrics.get("retries_attempted", 0)) + 1
                    await asyncio.sleep(min(delay, self.retry_policy.max_delay_sec))
                    delay = min(delay * 2.0, self.retry_policy.max_delay_sec)
                    continue
                else:
                    logger.error("Search failed api=%s err=%s q='%s'", api, str(e), query[:120])

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
            await progress_callback("Starting deep research analysis")

        deep_config = DeepResearchConfig(
            mode=mode,
            enable_web_search=True,
            enable_code_interpreter=False,
            background=True,
            include_paradigm_context=True,
        )

        # Apply experiment override from research record when available
        try:
            if research_id:
                rec = await self.research_store.get(research_id)
                pv = ((rec or {}).get("experiment") or {}).get("prompt_variant")
                if isinstance(pv, str) and pv in {"v1", "v2"}:
                    deep_config.prompt_variant = pv
        except Exception:
            pass

        deep_result = await deep_research_service.execute_deep_research(
            query=context_engineered.original_query,
            classification=classification,           # type: ignore[arg-type]
            context_engineering=context_engineered,   # type: ignore[arg-type]
            config=deep_config,
            progress_tracker=progress_callback,
            research_id=research_id,
        )

        if not getattr(deep_result, "content", None):
            return []

        deep_search_results = []
        for citation in getattr(deep_result, "citations", []) or []:
            title = getattr(citation, "title", "") or ""
            url = getattr(citation, "url", "") or ""
            s = int(getattr(citation, "start_index", 0) or 0)
            e = int(getattr(citation, "end_index", s) or s)
            body = getattr(deep_result, "content", "") or ""
            snippet = body[s:e][:200] if isinstance(body, str) else ""
            # Synthesize URL when missing for deep_research to avoid downstream drops
            if not url:
                safe_oid = "deep"
                synthesized = f"about:blank#citation-{safe_oid}-{hash(((title or snippet) or '')[:64]) & 0xFFFFFFFF}"
                url = synthesized
            domain = url.split('/')[2] if ('/' in url and len(url.split('/')) > 2) else url
            deep_search_results.append({
                "title": title or (url.split('/')[-1] if url else "(deep research)"),
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "result_type": "deep_research",
                "credibility_score": 0.9,
                "origin_query": getattr(context_engineered, "original_query", ""),
                "search_api": "deep_research",
                "metadata": {"unlinked_citation": True} if url.startswith("about:blank#citation-") else {}
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

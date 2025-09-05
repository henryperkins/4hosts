"""
Unified Research Orchestrator V2
Combines all the best features from multiple orchestrator implementations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Contracts
from contracts import ResearchStatus as ContractResearchStatus  # type: ignore
import hashlib
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
    get_cached_search_results,
    cache_search_results,
)
from services.text_compression import text_compressor, query_compressor, compress_search_results
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


class DeterministicMerger:
    """Ensures deterministic merging of async results"""

    @staticmethod
    def create_result_hash(result: Dict[str, Any]) -> str:
        """Create stable hash for a result"""
        key_string = f"{result.get('url', '')}|{result.get('title', '')}"
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def merge_results(
        result_batches: List[List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        Deterministically merge multiple result batches
        Returns: (merged_results, source_mapping)
        """
        url_to_sources: Dict[str, Set[str]] = defaultdict(set)
        url_to_best_result: Dict[str, Dict[str, Any]] = {}

        for batch_idx, batch in enumerate(result_batches):
            for result in batch:
                url = result.get('url', '')
                if not url:
                    continue

                source = result.get('source_api', f'batch_{batch_idx}')
                url_to_sources[url].add(source)

                if url not in url_to_best_result:
                    url_to_best_result[url] = result
                else:
                    current_content_len = len(
                        url_to_best_result[url].get('snippet', '') +
                        url_to_best_result[url].get('content', '')
                    )
                    new_content_len = len(
                        result.get('snippet', '') +
                        result.get('content', '')
                    )
                    if new_content_len > current_content_len:
                        url_to_best_result[url] = result

        sorted_urls = sorted(url_to_best_result.keys())
        merged_results = []
        source_mapping = {}

        for url in sorted_urls:
            result = url_to_best_result[url]
            result['source_apis'] = sorted(list(url_to_sources[url]))
            merged_results.append(result)
            source_mapping[url] = result['source_apis']

        return merged_results, source_mapping


class ResultDeduplicator:
    """Removes duplicate search results using various similarity measures"""

    def __init__(self, similarity_threshold: float = 0.8):
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

        self.low_quality_domains = {
            'ezinearticles.com', 'articlesbase.com', 'squidoo.com',
            'hubpages.com', 'buzzle.com', 'ehow.com'
        }

    def is_relevant(self, result: SearchResult, query: str, paradigm: str) -> bool:
        """Check if a result meets minimum relevance criteria"""

        # Check for spam content
        combined_text = f"{result.title} {result.snippet}".lower()
        if any(spam in combined_text for spam in self.spam_indicators):
            return False

        # Check for low-quality domains
        if result.domain in self.low_quality_domains:
            return False

        # Minimum content check
        if not result.title or not isinstance(result.title, str) or len(result.title.strip()) < 10:
            return False
        if not result.snippet or not isinstance(result.snippet, str) or len(result.snippet.strip()) < 20:
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
        if self._is_likely_duplicate_site(result.domain):
            return False

        # Paradigm-specific early filters
        if paradigm == "bernard" and result.result_type == "web":
            authoritative_indicators = ['.edu', '.gov', 'journal', 'research', 'study', 'analysis',
                                      'technology', 'innovation', 'science', 'ieee', 'acm', 'mit',
                                      'stanford', 'harvard', 'arxiv', 'nature', 'springer']
            tech_indicators = ['ai', 'artificial intelligence', 'machine learning', 'deep learning',
                             'neural', 'algorithm', 'technology', 'computing', 'software', 'innovation']

            has_authority = any(indicator in result.domain.lower() or indicator in combined_text for indicator in authoritative_indicators)
            has_tech_content = any(indicator in combined_text for indicator in tech_indicators)

            if not (has_authority or has_tech_content):
                academic_terms = ['methodology', 'hypothesis', 'conclusion', 'abstract', 'citation',
                                'analysis', 'framework', 'approach', 'technique', 'evaluation']
                if not any(term in combined_text for term in academic_terms):
                    return False

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
        self.merger = DeterministicMerger()
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
            import os
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
            import os
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
        await self.search_manager.initialize()
        await cache_manager.initialize()
        await initialize_deep_research()
        # Determine capability once
        self._supports_search_with_api = hasattr(self.search_manager, "search_with_api")

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
            await self.search_manager.cleanup()
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
        """Execute research with all enhanced features"""
        
        async def check_cancelled():
            """Check if research has been cancelled"""
            if not research_id:
                return False
            research_data = await self.research_store.get(research_id)
            is_cancelled = research_data and research_data.get("status") == RuntimeResearchStatus.CANCELLED
            if is_cancelled:
                logger.info(f"Research {research_id} has been cancelled")
            return is_cancelled

        start_time = datetime.now()

        # Use the old execute_paradigm_research for backward compatibility
        if hasattr(context_engineered, 'select_output'):
            # Check for cancellation before legacy execution
            if await check_cancelled():
                return {"status": "cancelled", "message": "Research was cancelled"}
                
            # Legacy format - convert and use old method
            result = await self._execute_paradigm_research_legacy(
                context_engineered,
                max_results=user_context.source_limit,
                progress_tracker=progress_callback,
                research_id=research_id,
                check_cancelled=check_cancelled  # Pass cancellation check
            )

            # Convert to new format and, if requested, synthesize an answer
            v2 = self._convert_legacy_to_v2_result(result)
            if synthesize_answer and _ANSWER_GEN_AVAILABLE:
                try:
                    # Build evidence quotes from converted results
                    evidence_quotes: List[Dict[str, Any]] = []
                    try:
                        from services.evidence_builder import build_evidence_quotes
                        evidence_quotes = await build_evidence_quotes(
                            getattr(context_engineered, "original_query", ""),
                            v2.get("results", []),
                            max_docs=min(12, int(getattr(user_context, "source_limit", 10))) if user_context else 10,
                            quotes_per_doc=3,
                        )
                    except Exception:
                        evidence_quotes = []
                    synthesized_answer = await self._synthesize_answer(
                        classification=classification,
                        context_engineered=context_engineered,
                        results=v2.get("results", []),
                        research_id=research_id or f"research_{int(start_time.timestamp())}",
                        options=answer_options or {},
                        evidence_quotes=evidence_quotes,
                    )
                    if synthesized_answer is not None:
                        v2["answer"] = synthesized_answer
                except Exception as e:
                    logger.error(f"Answer synthesis (legacy) failed: {e}")
            return v2

        # Otherwise use V2 implementation
        # Optimize queries based on paradigm and user limits
        # Use QueryCompressor.compress per refined query and cap by limit.
        # ContextEngineeredQuerySchema may expose 'refined_queries' or 'refined_queries' may be absent; fall back to original_query.
        try:
            limited = self._get_query_limit(user_context)
            source_queries = []
            if hasattr(context_engineered, "refined_queries") and isinstance(getattr(context_engineered, "refined_queries"), list):
                source_queries = list(getattr(context_engineered, "refined_queries"))
            else:
                # fallback to single original query
                source_queries = [getattr(context_engineered, "original_query", "")]

            optimized_list = []
            for q in source_queries:
                compressed = query_compressor.compress(q, preserve_keywords=True)
                if compressed:
                    optimized_list.append(compressed)
            # Deduplicate while preserving order
            seen_set = set()
            deduped = []
            for q in optimized_list:
                if q not in seen_set:
                    seen_set.add(q)
                    deduped.append(q)
            optimized_queries = deduped[:limited]
        except Exception as e:
            logger.warning(f"Query compression failed, using fallback queries: {e}")
            fallback_q = getattr(context_engineered, "original_query", "")
            optimized_queries = [fallback_q][: self._get_query_limit(user_context)]

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
            user_context
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
        if synthesize_answer and _ANSWER_GEN_AVAILABLE:
            try:
                # Build evidence quotes from processed results
                evidence_quotes: List[Dict[str, Any]] = []
                try:
                    from services.evidence_builder import build_evidence_quotes
                    evidence_quotes = await build_evidence_quotes(
                        getattr(context_engineered, "original_query", ""),
                        processed_results["results"],
                        max_docs=min(12, int(getattr(user_context, "source_limit", 10))),
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

        # Build synthesis context (not strictly required by the generator but useful for parity)
        if SynthesisContextModel:
            _ = SynthesisContextModel(
                query=context_engineered.original_query,
                paradigm=paradigm_code,
                search_results=sources,
                context_engineering=ce,
                max_length=int(options.get("max_length", 2000)),
                include_citations=bool(options.get("include_citations", True)),
                tone=str(options.get("tone", "professional")),
                metadata={"research_id": research_id},
                evidence_quotes=evidence_quotes or [],
            )

        # Emit synthesis started (mirrored as research_progress by tracker)
        try:
            from services.websocket_service import progress_tracker as _pt
            if _pt:
                await _pt.report_synthesis_started(research_id)
        except Exception:
            pass

        # Call the generator using the legacy signature for broad compatibility
        assert answer_orchestrator is not None, "Answer generation not available"
        answer = await answer_orchestrator.generate_answer(
            paradigm=paradigm_code,
            query=getattr(context_engineered, "original_query", ""),
            search_results=sources,
            context_engineering=ce,
            options={"research_id": research_id, **options, "evidence_quotes": evidence_quotes or []},
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
            from services.websocket_service import progress_tracker as _pt
            if _pt:
                await _pt.report_synthesis_completed(research_id, sections, citations)
        except Exception:
            pass

        return answer

    async def execute_paradigm_research(
        self,
        context_engineered_query,
        max_results: int = 100,
        progress_tracker=None,
        research_id: Optional[str] = None
    ) -> ResearchExecutionResult:
        """Legacy method for backward compatibility"""
        return await self._execute_paradigm_research_legacy(
            context_engineered_query,
            max_results,
            progress_tracker,
            research_id
        )

    async def _execute_paradigm_research_legacy(
        self,
        context_engineered_query,
        max_results: int = 100,
        progress_tracker=None,
        research_id: Optional[str] = None,
        check_cancelled: Optional[Any] = None
    ) -> ResearchExecutionResult:
        """Execute research using legacy format"""

        start_time = datetime.now()
        metrics = {"start_time": start_time.isoformat()}

        # Extract information from context engineering
        original_query = context_engineered_query.original_query
        classification = context_engineered_query.classification
        select_output = context_engineered_query.select_output

        # Normalize paradigm to internal code name
        paradigm = normalize_to_internal_code(classification.primary_paradigm)
        secondary_paradigm = None
        if classification.secondary_paradigm:
            secondary_paradigm = normalize_to_internal_code(classification.secondary_paradigm)

        logger.info(f"Executing research for paradigm: {paradigm}")

        # Initialize a simple plan/budget for the agentic loop
        plan = Plan(
            objective=original_query,
            checkpoints=[PlannerCheckpoint(name="initial_search", description="Run base queries")],
            budget=Budget(max_tokens=200_000, max_cost_usd=1.00, max_wallclock_minutes=2),
            stop_conditions={"max_iterations": int(self.agentic_config.get("max_iterations", 2))}
        )

        # Create search context
        search_context = SearchContext(
            original_query=original_query,
            paradigm=paradigm,
            secondary_paradigm=secondary_paradigm,
        )

        # Get paradigm-specific search strategy
        strategy = get_search_strategy(paradigm)

        # Use search queries from Context Engineering Select layer with guards/normalization
        search_queries = self._select_queries_from_select(select_output, context_engineered_query)
        # Cap number of queries (env override RESEARCH_MAX_QUERIES, default 8)
        try:
            import os
            max_q = int(os.getenv("RESEARCH_MAX_QUERIES", "8") or 8)
        except Exception:
            max_q = 8
        search_queries = search_queries[:max(1, max_q)]

        logger.info(f"Executing {len(search_queries)} search queries")

        # Execute searches with caching
        all_results = {}
        cost_breakdown = {}

        for idx, query_data in enumerate(search_queries):
            # Check for cancellation before each search query
            if check_cancelled and await check_cancelled():
                logger.info(f"Research {research_id} cancelled during legacy search execution")
                break
                
            query = query_data["query"]
            query_type = query_data["type"]
            weight = query_data["weight"]

            # Check cache first
            config_dict = {"max_results": max_results, "language": "en", "region": "us"}

            cached_results = await get_cached_search_results(
                query, config_dict, paradigm
            )

            if cached_results:
                all_results[f"{query_type}_{query[:30]}"] = cached_results
                logger.info(f"Using cached results for: {query[:50]}...")

                if progress_tracker and research_id:
                    await progress_tracker.report_search_completed(
                        research_id, query, len(cached_results)
                    )
            else:
                if progress_tracker and research_id:
                    await progress_tracker.report_search_started(
                        research_id, query, "mixed", idx + 1, len(search_queries)
                    )
                    search_progress = 30 + int((idx / len(search_queries)) * 20)
                    await progress_tracker.update_progress(
                        research_id,
                        phase="search",
                        message=f"Searching: {query[:40]}...",
                        progress=search_progress,
                        items_done=idx + 1,
                        items_total=len(search_queries),
                    )

            # Wire Select layer preferences into SearchConfig
            try:
                sel = getattr(context_engineered_query, "select_output", None)
                # Prefer normalized source types (web/news/academic) when available
                prefs = list(getattr(sel, "normalized_source_types", []) or getattr(sel, "source_preferences", []) or [])
                excludes = list(getattr(sel, "exclusion_filters", []) or [])
                whitelist = list(getattr(sel, "authority_whitelist", []) or [])
            except Exception:
                prefs, excludes, whitelist = [], [], []
            config = SearchConfig(
                max_results=min(max_results, 50),
                language="en",
                region="us",
                source_types=prefs,
                exclusion_keywords=excludes,
                authority_whitelist=whitelist,
            )

            try:
                sm = self.search_manager
                if sm is None:
                    raise RuntimeError("search_manager is not initialized")
                api_results = await sm.search_with_fallback(query, config)

                primary_api = "google"
                cost = await self.cost_monitor.track_search_cost(primary_api, 1)
                cost_breakdown[f"{query_type}_{query[:20]}"] = cost

                for result in api_results:
                    result.credibility_score = (
                        result.credibility_score * weight
                        if hasattr(result, "credibility_score")
                        else weight
                    )

                await cache_search_results(query, config_dict, paradigm, api_results)
                all_results[f"{query_type}_{query[:30]}"] = api_results

                logger.info(f"Got {len(api_results)} results for: {query[:50]}...")

                if progress_tracker and research_id:
                    await progress_tracker.report_search_completed(
                        research_id, query, len(api_results)
                    )
                    for result in api_results[:3]:
                        await progress_tracker.report_source_found(
                            research_id,
                            {
                                "title": result.title,
                                "url": result.url,
                                "domain": result.domain,
                                "snippet": result.snippet[:200] if result.snippet else "",
                                "credibility_score": getattr(result, "credibility_score", 0.5)
                            }
                        )
            except Exception as e:
                logger.error(f"Search failed for '{query}': {str(e)}")
                all_results[f"{query_type}_{query[:30]}"] = []

        # Combine all results with validation
        combined_results = []
        for query_key, query_results in all_results.items():
            if query_results is None:
                logger.warning(f"Query results for '{query_key}' is None, skipping")
                continue

            # Validate each result before adding
            valid_results = []
            for result in query_results:
                if result is None:
                    logger.warning(f"Found None result in query '{query_key}', skipping")
                    continue

                # Check if result has required attributes
                if not hasattr(result, 'title') or not hasattr(result, 'content'):
                    logger.warning(f"Result missing required attributes in query '{query_key}': {type(result)}")
                    continue

                # Check if content is not empty; try to repair from snippet/title if missing
                content = getattr(result, 'content', '')
                if content is None or not str(content).strip():
                    try:
                        snippet_val = getattr(result, 'snippet', '') or ''
                        title_val = getattr(result, 'title', '') or ''
                        if isinstance(snippet_val, str) and snippet_val.strip():
                            result.content = f"Summary from search results: {snippet_val.strip()}"
                            content = result.content
                            result.raw_data = getattr(result, 'raw_data', {}) or {}
                            result.raw_data['content_source'] = 'snippet_repair'
                        elif isinstance(title_val, str) and title_val.strip():
                            result.content = title_val.strip()
                            content = result.content
                            result.raw_data = getattr(result, 'raw_data', {}) or {}
                            result.raw_data['content_source'] = 'title_repair'
                    except Exception:
                        pass

                if content is None or not str(content).strip():
                    logger.warning(f"Result has empty content in query '{query_key}', skipping")
                    continue

                valid_results.append(result)

            combined_results.extend(valid_results)
            logger.debug(f"Query '{query_key}': {len(query_results)} -> {len(valid_results)} valid results")

        logger.info(f"Combined {len(combined_results)} total validated results")

        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="deduplication",
                message="Removing duplicate results...",
                progress=52,
            )

        # Deduplicate results with additional validation
        dedup_result = await self.deduplicator.deduplicate_results(combined_results)
        deduplicated_results = dedup_result["unique_results"]

        # Validate deduplicated results
        validated_results = []
        for result in deduplicated_results:
            if result is None:
                logger.warning("Found None result after deduplication, skipping")
                continue
            validated_results.append(result)

        # Update deduplicated_results with validated ones
        deduplicated_results = validated_results

        logger.info(f"Validation after dedup: {len(dedup_result['unique_results'])} -> {len(deduplicated_results)} valid results")

        # Agentic loop (optional): Plan → Act → Critique → Revise
        try:
            # Initialize agentic loop state for type safety
            initial_sources: List[Dict[str, Any]] = []
            coverage: float = 0.0
            missing: List[str] = []
            if self.agentic_config.get("enabled") and hasattr(context_engineered_query, "write_output"):
                # Build initial sources view for critique
                initial_sources = []
                for r in deduplicated_results[:25]:
                    try:
                        initial_sources.append({
                            "title": getattr(r, "title", ""),
                            "url": getattr(r, "url", ""),
                            "snippet": getattr(r, "snippet", "")
                        })
                    except Exception:
                        continue

                # agentic_process imports moved to module level
                coverage, missing = evaluate_coverage_from_sources(
                    original_query, context_engineered_query, initial_sources
                )

                # Optional LLM critic
            if self.agentic_config.get("enable_llm_critic"):
                from services.llm_critic import llm_coverage_and_claims
                try:
                    themes = getattr(context_engineered_query.write_output, "key_themes", []) or []
                    focus = getattr(context_engineered_query.isolate_output, "focus_areas", []) or []
                    lm = await llm_coverage_and_claims(original_query, paradigm, themes, focus, initial_sources)
                    coverage = max(coverage, float(lm.get("coverage_score", coverage)))
                    if isinstance(lm.get("missing_facets"), list):
                        missing = list({*missing, *[str(x) for x in lm["missing_facets"] if x]})
                    # Record critic diagnostics
                    metrics_entry = {
                        "step": "llm_critic",
                        "coverage": coverage,
                        "warnings": lm.get("warnings", []),
                        "flagged_sources": lm.get("flagged_sources", [])
                    }
                    # Stream coverage delta to UI
                    try:
                        if progress_tracker and research_id:
                            await progress_tracker.update_progress(
                                research_id,
                                phase="agentic_loop",
                                custom_data={
                                    "message": f"Critic coverage: {int(coverage*100)}% | flagged: {len(metrics_entry['flagged_sources'])}",
                                    "coverage": coverage,
                                    "flagged": len(metrics_entry['flagged_sources']),
                                },
                            )
                    except Exception:
                        pass
                    # Attach to execution metrics later via metrics_dict
                except Exception as e:
                    metrics_entry = {"step": "llm_critic", "error": str(e)}
                    # Stash into a local list for merging into metrics_dict below
                    locals().setdefault("_agent_trace", []).append(metrics_entry)

                iterations = 0
                max_iter = int(self.agentic_config.get("max_iterations", 2))
                threshold = float(self.agentic_config.get("coverage_threshold", 0.75))
                while coverage < threshold and iterations < max_iter:
                    iterations += 1
                    if progress_tracker and research_id:
                        try:
                            await progress_tracker.update_progress(
                                research_id,
                                phase="agentic_loop",
                                message=f"Agentic iteration {iterations}",
                                progress=55 + iterations * 5,
                                items_done=iterations,
                                items_total=max_iter,
                            )
                        except Exception:
                            pass

                    gaps = summarize_domain_gaps(initial_sources)
                    proposed = propose_queries_enriched(
                        original_query,
                        paradigm,
                        missing,
                        gaps,
                        max_new=int(self.agentic_config.get("max_new_queries_per_iter", 4)),
                    )

                    # Execute proposed queries and merge minimally
                    for q in proposed:
                        try:
                            # Enforce budget per call
                            if not self.planner.record_tool_spend(plan, "google", 1):
                                logger.info("Agentic loop stopped by budget")
                                break
                            config = SearchConfig(max_results=min(max_results, 30), language="en", region="us")
                            sm = self.search_manager
                            if sm is None:
                                raise RuntimeError("search_manager is not initialized")
                            api_results = await sm.search_with_fallback(q, config)
                            all_results[f"agentic_{q[:30]}"] = api_results
                            for res in api_results:
                                if res and getattr(res, 'url', None):
                                    deduplicated_results.append(res)
                        except Exception as e:
                            logger.warning("Agentic query failed: %s", e)

                    # Recompute coverage after augmentation
                    initial_sources = [{
                        "title": getattr(r, "title", ""),
                        "url": getattr(r, "url", ""),
                        "snippet": getattr(r, "snippet", "")
                    } for r in deduplicated_results[:50]]
                    coverage, missing = evaluate_coverage_from_sources(
                        original_query, context_engineered_query, initial_sources
                    )

                    locals().setdefault("_agent_trace", []).append({
                        "step": "revise",
                        "iteration": iterations,
                        "coverage": coverage,
                        "proposed_queries": proposed,
                        "pool_size": len(deduplicated_results),
                    })
        except Exception as agent_err:
            logger.warning("Agentic loop error: %s", agent_err)

        if progress_tracker and research_id:
            await progress_tracker.report_deduplication(
                research_id, len(combined_results), len(deduplicated_results)
            )

        # Apply early-stage content filtering
        early_filtered_results = self._apply_early_relevance_filter(
            deduplicated_results, original_query, paradigm
        )

        logger.info(f"Early filtering: {len(deduplicated_results)} -> {len(early_filtered_results)} results")

        # Apply paradigm-specific filtering and ranking
        filtered_results = await strategy.filter_and_rank_results(
            early_filtered_results, search_context
        )

        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="filtering",
                message="Filtering & ranking sources...",
                progress=60,
                items_done=len(filtered_results),
                items_total=len(early_filtered_results),
            )

        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="credibility",
                message="Evaluating source credibility...",
                progress=55,
            )

        # Calculate credibility scores
        credibility_scores = {}
        top_results = filtered_results[:20]
        for idx, result in enumerate(top_results):
            credibility_score, credibility_explanation, credibility_status = await self.get_source_credibility_safe(
                result.domain, paradigm
            )
            credibility_scores[result.domain] = credibility_score
            result.credibility_score = credibility_score

            if progress_tracker and research_id and idx % 5 == 0:
                await progress_tracker.report_credibility_check(
                    research_id, result.domain, credibility_score
                )

            # Stream incremental progress every result
            if progress_tracker and research_id:
                try:
                    await progress_tracker.update_progress(
                        research_id,
                        phase="credibility",
                        message="Evaluating source credibility...",
                        progress=55 + int((idx + 1) / max(1, len(top_results)) * 10),
                        items_done=idx + 1,
                        items_total=len(top_results),
                    )
                except Exception:
                    pass

        # Limit final results
        final_results = filtered_results[:max_results]
        # Results now include a `sections` list by default in SearchResult; no dynamic setattr needed

        # Execute secondary search if applicable
        secondary_results = []
        if secondary_paradigm:
            logger.info(
                f"Executing secondary research for paradigm: {secondary_paradigm}"
            )
            secondary_strategy = get_search_strategy(secondary_paradigm)
            secondary_query = f"{original_query} {secondary_paradigm}"

            sel = getattr(context_engineered_query, "select_output", None)
            prefs = list(getattr(sel, "source_preferences", []) or [])
            excludes = list(getattr(sel, "exclusion_filters", []) or [])
            config = SearchConfig(
                max_results=min(max_results // 2, 25),
                language="en",
                region="us",
                source_types=prefs,
                exclusion_keywords=excludes,
            )
            try:
                sm = self.search_manager
                if sm is None:
                    raise RuntimeError("search_manager is not initialized")
                api_results = await sm.search_with_fallback(
                    secondary_query, config
                )
                secondary_results.extend(api_results)
            except Exception as e:
                logger.error(
                    f"Secondary search failed for '{secondary_query}': {str(e)}"
                )

        # Calculate execution metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        metrics_dict = {
            "end_time": end_time.isoformat(),
            "processing_time_seconds": processing_time,
            "queries_executed": len(search_queries),
            "raw_results_count": len(combined_results),
            "deduplicated_count": len(deduplicated_results),
            "final_results_count": len(final_results),
            "secondary_results_count": len(secondary_results),
            "duplicates_removed": dedup_result["duplicates_removed"],
            "credibility_checks": len(credibility_scores),
        }
        # Budget used
        try:
            metrics_dict["agent_budget_spend_usd"] = float(getattr(plan, "consumed_cost_usd", 0.0))
            metrics_dict["agent_budget_tokens"] = int(getattr(plan, "consumed_tokens", 0))
        except Exception:
            pass
        # Attach agent trace if available
        try:
            if "_agent_trace" in locals() and locals()["_agent_trace"]:
                metrics_dict["agent_trace"] = locals()["_agent_trace"]
        except Exception:
            pass
        if isinstance(metrics, dict):
            metrics.update(metrics_dict)
        else:
            metrics = metrics_dict

        # Create and log final execution result via helper
        return self._finalize_legacy_result(
            original_query=original_query,
            paradigm=paradigm,
            secondary_paradigm=secondary_paradigm,
            search_queries=search_queries,
            all_results=all_results,
            final_results=final_results,
            secondary_results=secondary_results,
            credibility_scores=credibility_scores,
            metrics=metrics,
            cost_breakdown=cost_breakdown,
            processing_time=processing_time,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Extracted helpers from legacy implementation (behavior preserved)
    # ──────────────────────────────────────────────────────────────────────
    def _select_queries_from_select(self, select_output: Any, context_engineered_query: Any) -> List[Dict[str, Any]]:
        raw_sq = getattr(select_output, "search_queries", None)
        search_queries: List[Dict[str, Any]] = []
        if isinstance(raw_sq, list):
            for item in raw_sq:
                if isinstance(item, dict):
                    q = (item.get("query") or "").strip()
                    if q:
                        t = item.get("type", "generic")
                        try:
                            w = float(item.get("weight", 1.0) or 1.0)
                        except Exception as e:
                            self._log_and_continue("Invalid weight in select query", e)
                            w = 1.0
                        search_queries.append({"query": q, "type": t, "weight": w})
                elif isinstance(item, str):
                    q = item.strip()
                    if q:
                        search_queries.append({"query": q, "type": "generic", "weight": 1.0})
        elif isinstance(raw_sq, dict):
            q = (raw_sq.get("query") or "").strip()
            if q:
                try:
                    w = float(raw_sq.get("weight", 1.0) or 1.0)
                except Exception as e:
                    self._log_and_continue("Invalid weight in select query (dict)", e)
                    w = 1.0
                search_queries.append({"query": q, "type": raw_sq.get("type", "generic"), "weight": w})

        if not search_queries:
            oq = getattr(context_engineered_query, "original_query", "") or ""
            if oq:
                search_queries = [{"query": oq, "type": "generic", "weight": 1.0}]
        return search_queries

    def _finalize_legacy_result(
        self,
        *,
        original_query: str,
        paradigm: str,
        secondary_paradigm: Optional[str],
        search_queries: List[Dict[str, Any]],
        all_results: Dict[str, List[SearchResult]],
        final_results: List[SearchResult],
        secondary_results: List[SearchResult],
        credibility_scores: Dict[str, float],
        metrics: Dict[str, Any],
        cost_breakdown: Dict[str, float],
        processing_time: float,
    ) -> ResearchExecutionResult:
        # Determine execution status based on availability of results
        execution_status = (
            ContractResearchStatus.FAILED_NO_SOURCES if len(final_results) == 0 else ContractResearchStatus.OK
        )

        result = ResearchExecutionResult(
            original_query=original_query,
            paradigm=paradigm,
            secondary_paradigm=secondary_paradigm,
            search_queries_executed=search_queries,
            raw_results=all_results,
            filtered_results=final_results,
            secondary_results=secondary_results,
            credibility_scores=credibility_scores,
            execution_metrics=metrics,
            cost_breakdown=cost_breakdown,
            status=execution_status,
        )

        # Store in history
        try:
            self.execution_history.append(result)
        except Exception as e:
            self._log_and_continue("Failed to append to execution history", e)

        logger.info(f"Research execution completed in {processing_time:.2f}s")
        logger.info(f"Final results: {len(final_results)} sources")

        # Enhanced logging for debugging (first 5 results)
        if final_results:
            logger.debug("Final results detailed breakdown:")
            for i, r in enumerate(final_results[:5]):
                if r is None:
                    self._log_and_continue(f"Result {i} is None")
                    continue
                try:
                    title = getattr(r, 'title', 'No title')
                    url = getattr(r, 'url', 'No URL')
                    content_length = len(getattr(r, 'content', '') or '')
                    sections = getattr(r, 'sections', [])
                    sections_count = len(sections) if isinstance(sections, list) else 0
                    logger.debug(
                        f"  Result {i}: '{title[:50]}...' ({content_length} chars, {sections_count} sections) - {url}"
                    )
                    if not isinstance(sections, list):
                        self._log_and_continue(f"  Result {i} sections is not a list: {type(sections)}")
                except Exception as e:
                    self._log_and_continue("Failed to log result breakdown", e)
        else:
            self._log_and_continue("No final results produced - this may cause downstream errors")

        return result

    def _get_query_limit(self, user_context: Any) -> int:
        """Get query limit based on user context"""
        base_limits = {
            "FREE": 3,
            "BASIC": 5,
            "PRO": 10,
            "ENTERPRISE": 20,
            "ADMIN": 30
        }
        role = getattr(user_context, "role", "PRO")
        return base_limits.get(role, 3)

    async def _execute_searches_deterministic(
        self,
        queries: List[str],
        paradigm: HostParadigm,
        user_context: Any,
        progress_callback: Optional[Any],
        research_id: Optional[str],
        check_cancelled: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute searches with per-task timeout, retries, and deterministic ordering"""
        sm = self.search_manager
        if sm is None:
            raise RuntimeError("search_manager is not initialized")
        entries: List[Dict[str, Any]] = []
        for qidx, query in enumerate(queries):
            # Check for cancellation before each query
            if check_cancelled and await check_cancelled():
                logger.info(f"Research {research_id} cancelled during search execution")
                return entries  # Return partial results
            for api in self._select_apis_for_paradigm(paradigm, user_context):
                entries.append({"query_index": qidx, "query": query, "api": api})

        if progress_callback and research_id:
            await progress_callback(f"Executing {len(entries)} search operations")

        processed_results: List[Dict[str, Any]] = []
        # Execute sequentially per entry to enable precise retry/backoff and deterministic ordering
        for idx, meta in enumerate(entries):
            # Check for cancellation before each search task
            if check_cancelled and await check_cancelled():
                logger.info(f"Research {research_id} cancelled during search task {idx}/{len(entries)}")
                break  # Exit search loop early
            
            api = meta["api"]
            query = meta["query"]
            qidx = meta["query_index"]

            # Metrics: count calls
            api_counts = self._search_metrics.get("api_call_counts", {})
            api_counts[api] = int(api_counts.get(api, 0)) + 1
            self._search_metrics["api_call_counts"] = api_counts

            attempt = 0
            result_items: List[Any] = []
            delay = self.retry_policy.base_delay_sec
            while attempt < self.retry_policy.max_attempts:
                # Check for cancellation before each retry
                if check_cancelled and await check_cancelled():
                    logger.info(f"Research {research_id} cancelled during search retry")
                    break
                    
                attempt += 1
                try:
                    config = SearchConfig(max_results=20, language="en", region="us")
                    if self._supports_search_with_api:
                        coro = sm.search_with_api(api, query, config)
                    else:
                        # Fallback will ignore api specificity; still tracked for metrics
                        coro = sm.search_with_fallback(query, config)

                    # Per-attempt timeout (configurable via SEARCH_TASK_TIMEOUT_SEC)
                    task_result = await asyncio.wait_for(coro, timeout=self.search_task_timeout)
                    if isinstance(task_result, list):
                        result_items = task_result
                    else:
                        result_items = []
                    break
                except asyncio.TimeoutError:
                    self._search_metrics["task_timeouts"] = int(self._search_metrics.get("task_timeouts", 0)) + 1
                    if attempt < self.retry_policy.max_attempts:
                        self._search_metrics["retries_attempted"] = int(self._search_metrics.get("retries_attempted", 0)) + 1
                        await asyncio.sleep(min(delay, self.retry_policy.max_delay_sec))
                        delay = min(delay * 2.0, self.retry_policy.max_delay_sec)
                        continue
                    else:
                        logger.error("Search task timeout api=%s qid=q%d q='%s'", api, qidx, query[:80])
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
                        logger.error("Search failed api=%s qid=q%d err=%s", api, qidx, str(e))

            # Process results for this entry
            if result_items:
                for item in result_items:
                    if hasattr(item, 'to_dict'):
                        item_dict = item.to_dict()
                    elif hasattr(item, '__dict__'):
                        item_dict = item.__dict__.copy()
                    else:
                        item_dict = dict(item) if isinstance(item, dict) else {}

                    item_dict['origin_query'] = meta['query']
                    item_dict['origin_query_id'] = f"q{meta['query_index']}"
                    item_dict['search_api'] = meta['api']
                    item_dict['result_index'] = idx

                    normalized_item = self.normalize_result_shape(item_dict)
                    if normalized_item:
                        processed_results.append(normalized_item)

        processed_results.sort(key=lambda x: (
            x.get('origin_query_id', ''),
            x.get('search_api', ''),
            x.get('url', ''),
            x.get('title', '')
        ))

        # Update metrics
        try:
            self._search_metrics['total_queries'] += len(queries)
            self._search_metrics['total_results'] += len(processed_results)
        except Exception:
            self._search_metrics['total_queries'] = int(self._search_metrics.get('total_queries', 0)) + len(queries)
            self._search_metrics['total_results'] = int(self._search_metrics.get('total_results', 0)) + len(processed_results)

        if not processed_results:
            logger.warning("No processed search results; downstream pipeline may produce empty output")

        return processed_results

    def _select_apis_for_paradigm(
        self,
        paradigm: HostParadigm,
        user_context: Any
    ) -> List[str]:
        """Select appropriate APIs based on paradigm and user context"""

        paradigm_apis = {
            HostParadigm.DOLORES: ["brave", "google"],
            HostParadigm.BERNARD: ["google", "arxiv", "pubmed"],
            HostParadigm.MAEVE: ["google", "brave"],
            HostParadigm.TEDDY: ["google"]
        }

        selected = paradigm_apis.get(paradigm, ["google"])

        if not getattr(user_context, "is_pro_user", True) and len(selected) > 2:
            selected = selected[:2]

        prefs = getattr(user_context, "preferences", {}) or {}
        if isinstance(prefs, dict) and prefs.get("preferred_sources"):
            selected.extend(prefs["preferred_sources"])

        seen = set()
        unique_apis = []
        for api in selected:
            if api not in seen:
                seen.add(api)
                unique_apis.append(api)
                # apis_used is a set stored in metrics; ensure it's a set before add
                apis_used = self._search_metrics.get('apis_used')
                if isinstance(apis_used, set):
                    apis_used.add(api)
                else:
                    self._search_metrics['apis_used'] = {api}

        return unique_apis

    async def _create_search_task(
        self,
        api_name: str,
        query: str,
        query_index: int
    ) -> Optional[asyncio.Task]:
        """Create a search task for specific API"""
        try:
            config = SearchConfig(max_results=20, language="en", region="us")
            sm = self.search_manager
            if sm is None:
                raise RuntimeError("search_manager is not initialized")
            # Ensure the selected API is actually used by search manager if supported
            if hasattr(sm, "search_with_api"):
                task = asyncio.create_task(
                    sm.search_with_api(
                        api_name,
                        query,
                        config,
                    ),
                    name=f"{api_name}_q{query_index}",
                )
            else:
                # Fallback to generic with_fallback
                task = asyncio.create_task(
                    sm.search_with_fallback(query, config),
                    name=f"{api_name}_q{query_index}"
                )
            if self.diagnostics.get("log_task_creation"):
                logger.debug(
                    "[orchestrator] task created: api=%s qid=q%d q='%s'",
                    api_name,
                    query_index,
                    query[:80],
                )
            return task
        except Exception as e:
            logger.error(f"Failed to create task for {api_name}: {e}")
        return None

    async def _process_results(
        self,
        raw_results: List[Dict[str, Any]],
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any
    ) -> Dict[str, Any]:
        """Process results with deduplication and enrichment"""

        url_groups = defaultdict(list)
        for result in raw_results:
            url = result.get('url', '')
            if url:
                url_groups[url].append(result)

        deduplicated_results = []

        for url, duplicates in url_groups.items():
            merged = self._merge_duplicate_results(duplicates)

            from urllib.parse import urlparse
            merged_url = merged.get('url', '') or ''
            if not merged_url:
                logger.debug("Dropping merged duplicate without URL")
                continue
            domain = urlparse(merged_url).netloc
            credibility_score, credibility_explanation, credibility_status = await self.get_source_credibility_safe(
                domain,
                classification.primary_paradigm.value
            )
            merged['credibility_score'] = credibility_score
            merged['credibility_explanation'] = credibility_explanation
            merged['credibility_status'] = credibility_status
            merged['paradigm_alignment'] = classification.primary_paradigm.value

            # Construct SearchResultSchema with minimal required fields; extra fields placed in metadata
            extra_meta = merged.get('metadata', {}) or {}
            try:
                # Derive domain from URL if missing
                from urllib.parse import urlparse as _urlparse
                _domain = extra_meta.get("domain")
                if not _domain:
                    try:
                        _domain = _urlparse(merged_url).netloc.lower()
                    except Exception:
                        _domain = ""
                # Attempt to parse source category from explanation (e.g., 'cat=academic')
                try:
                    import re as _re
                    m = _re.search(r"cat=([^,\s]+)", credibility_explanation or "")
                    if m:
                        extra_meta["source_category"] = m.group(1)
                except Exception:
                    pass
                extra_meta.update({
                    "credibility_explanation": credibility_explanation,
                    "origin_query": merged.get('origin_query'),
                    "origin_query_id": merged.get('origin_query_id'),
                    "search_api": merged.get('search_api', 'unknown'),
                    "paradigm_alignment": getattr(classification, "primary_paradigm", HostParadigm.BERNARD),
                    "domain": _domain
                })
            except Exception:
                pass

            result_schema = SearchResultSchema(
                title=merged.get('title', ''),
                url=merged_url,
                snippet=merged.get('snippet', ''),
                source=domain or extra_meta.get("domain", "") or merged.get("search_api", "unknown"),
                credibility_score=credibility_score,
                relevance_score=float(merged.get('relevance_score', 0.5) or 0.5),
                metadata=extra_meta
            )

            deduplicated_results.append(result_schema)

        deduplicated_results.sort(
            key=lambda x: (x.credibility_score, x.url),
            reverse=True
        )

        # Optional LLM reranker for top-N relevance (env: LLM_RERANK_ENABLED)
        try:
            import os, json
            if str(os.getenv("LLM_RERANK_ENABLED", "0")).lower() in {"1", "true", "yes"} and deduplicated_results:
                top_n = min(30, len(deduplicated_results))
                sample = deduplicated_results[:top_n]
                # Compose compact passages
                lines = []
                for i, r in enumerate(sample):
                    title = getattr(r, "title", "") or ""
                    snip = getattr(r, "snippet", "") or ""
                    txt = (title + " " + snip)[:400]
                    lines.append(f"{i}) {txt}")
                prompt = (
                    "You are scoring search results for relevance.\n"
                    f"Query: {getattr(context_engineered, 'original_query', '')}\n\n"
                    "For each passage i below, assign a relevance score in [0,5] (float).\n"
                    "Return only JSON matching the schema.\n\n"
                    "Passages:\n" + "\n".join(lines)
                )
                json_schema = {
                    "name": "rerankScores",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "scores": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "score": {"type": "number"}
                                    },
                                    "required": ["index", "score"]
                                }
                            }
                        },
                        "required": ["scores"]
                    },
                    "strict": True
                }
                try:
                    from services.llm_client import llm_client
                    resp = await llm_client.generate_completion(
                        prompt,
                        paradigm="bernard",
                        max_tokens=800,
                        temperature=0.0,
                        json_schema=json_schema,
                    )
                    if isinstance(resp, str):
                        data = json.loads(resp)
                    else:
                        # In case client returns dict with content (unlikely here)
                        try:
                            data = json.loads(getattr(resp, "content", "{}"))
                        except Exception:
                            data = {"scores": []}
                    scores = {int(it.get("index", -1)): float(it.get("score", 0.0)) for it in (data.get("scores") or []) if isinstance(it, dict)}
                    # Stable sort by score desc while preserving prior credibility order
                    indexed = list(enumerate(sample))
                    indexed.sort(key=lambda t: scores.get(t[0], 0.0), reverse=True)
                    deduplicated_results = [it[1] for it in indexed] + deduplicated_results[top_n:]
                    self._search_metrics["llm_rerank_used"] = True
                except Exception as _e:
                    logger.debug("LLM reranker skipped: %s", _e)
        except Exception:
            pass

        limited_results = deduplicated_results[: int(getattr(user_context, "source_limit", 10))]

        # Paradigm post‑ranking: reorder results using the active paradigm strategy
        try:
            strategy = get_search_strategy(classification.primary_paradigm.value)
            sc = SearchContext(
                original_query=getattr(context_engineered, "original_query", ""),
                paradigm=classification.primary_paradigm.value,
                secondary_paradigm=(getattr(classification, "secondary_paradigm", None).value
                                    if getattr(classification, "secondary_paradigm", None) else None),
            )
            # Convert to SearchResult objects for scoring
            tmp_results: List[SearchResult] = []
            for r in limited_results:
                try:
                    tmp_results.append(
                        SearchResult(
                            title=getattr(r, "title", ""),
                            url=getattr(r, "url", ""),
                            snippet=getattr(r, "snippet", ""),
                            source=getattr(r, "source", getattr(r, "domain", "") or "web"),
                            published_date=getattr(r, "published_date", None),
                            domain=getattr(r, "domain", ""),
                            credibility_score=float(getattr(r, "credibility_score", 0.0) or 0.0),
                            result_type=getattr(r, "metadata", {}).get("result_type", "web") if isinstance(getattr(r, "metadata", {}), dict) else "web",
                        )
                    )
                except Exception:
                    continue
            ranked = await strategy.filter_and_rank_results(tmp_results, sc)
            # Reorder limited_results by ranked URL order
            order = {res.url: idx for idx, res in enumerate(ranked)}
            limited_results.sort(key=lambda x: order.get(getattr(x, "url", ""), 1e9))
            self._search_metrics["paradigm_post_ranking"] = True
        except Exception as e:
            logger.debug(f"Paradigm post-ranking skipped: {e}")

        dedup_rate = 1.0 - (len(deduplicated_results) / float(max(len(raw_results), 1)))
        self._search_metrics['deduplication_rate'] = dedup_rate

        # Defensive: ensure SearchResultSchema has to_dict; otherwise map manually
        try:
            result_payload = [r.to_dict() for r in limited_results]
        except Exception:
            result_payload = []
            for r in limited_results:
                payload_item = {
                    "url": getattr(r, "url", ""),
                    "title": getattr(r, "title", ""),
                    "snippet": getattr(r, "snippet", ""),
                    "credibility_score": getattr(r, "credibility_score", 0.5),
                    "metadata": getattr(r, "metadata", {}) or {}
                }
                result_payload.append(payload_item)

        # Some implementations expose compress_search_result (singular). Fallback gracefully.
        # Preserve map for credibility_score and metadata by URL
        preserve_map = {}
        for item in result_payload:
            u = item.get("url", "")
            if u:
                preserve_map[u] = {
                    "credibility_score": item.get("credibility_score"),
                    "metadata": item.get("metadata", {}) or {}
                }

        # Build weighted budgets: credibility × relevance × recency × evidence_density
        def _recency_score(md: dict) -> float:
            try:
                from datetime import datetime, timezone
                pd = md.get("published_date") or md.get("publication_date")
                if isinstance(pd, str):
                    # Best-effort parse
                    import re as _re, datetime as _dt
                    m = _re.match(r"(\d{4})-(\d{2})-(\d{2})", pd)
                    if m:
                        dt = _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    else:
                        return 0.3
                elif hasattr(pd, "year"):
                    dt = pd
                else:
                    return 0.3
                age_days = (datetime.now() - dt).days
                # ~exponential decay with ~1 year half-life
                import math
                return max(0.0, min(1.0, math.exp(-age_days / 365.0)))
            except Exception:
                return 0.3

        def _evidence_density(text: str) -> float:
            if not text:
                return 0.0
            nums = sum(ch.isdigit() for ch in text)
            frac = nums / max(1, len(text))
            bonus = 0.2 if any(k in text.lower() for k in ["doi", "arxiv", "pmid", "%"]) else 0.0
            return min(1.0, frac * 5.0 + bonus)  # cap

        weights = {}
        for item in result_payload:
            cred = float(item.get("credibility_score", 0.0) or 0.0)
            rel = float(item.get("relevance_score", 0.5) or 0.5)
            md = item.get("metadata", {}) or {}
            rec = _recency_score(md)
            evd = _evidence_density((item.get("title", "") + " " + item.get("snippet", ""))[:1000])
            w = max(0.0, 0.5 * cred + 0.25 * rel + 0.15 * rec + 0.10 * evd)
            u = item.get("url") or ""
            if u:
                weights[u] = w

        try:
            compressed_results = compress_search_results(
                result_payload,
                total_token_budget=3000,
                weights=weights,
            )
            self._search_metrics["compression_plural_used"] = int(
                self._search_metrics.get("compression_plural_used", 0)
            ) + 1
        except Exception:
            # Fallback: manual singular compression per item preserving dict shape
            compressed_results = []
            for item in result_payload:
                if not isinstance(item, dict):
                    continue
                title = item.get("title", "") or ""
                snippet = item.get("snippet", "") or ""
                summary = text_compressor.compress_search_result(title, snippet)
                short_title = title if len(title) <= 120 else title[:117] + "..."
                new_item = dict(item)
                new_item["title"] = short_title
                new_item["snippet"] = summary
                content_val = new_item.get("content")
                if content_val:
                    new_item["content"] = text_compressor.compress(str(content_val), max_length=1000)
                compressed_results.append(new_item)
            self._search_metrics["compression_singular_used"] = int(
                self._search_metrics.get("compression_singular_used", 0)
            ) + 1

        # Re-merge preserved fields into compressed output
        merged_compressed = []
        for item in compressed_results or []:
            if not isinstance(item, dict):
                continue
            u = (item.get("url") or "").strip()
            if u and u in preserve_map:
                item.setdefault("metadata", {})
                # Merge metadata, do not overwrite existing keys in item
                preserved_meta = preserve_map[u]["metadata"] or {}
                merged_meta = dict(preserved_meta)
                merged_meta.update(item.get("metadata") or {})
                item["metadata"] = merged_meta
                # Re-inject credibility_score if missing
                if "credibility_score" not in item or item.get("credibility_score") is None:
                    item["credibility_score"] = preserve_map[u]["credibility_score"]
            merged_compressed.append(item)
        compressed_results = merged_compressed

        # Lightweight contradiction detection on compressed snippets
        contradictions = self._detect_contradictions(compressed_results)

        # Category distribution over limited_results
        try:
            cat_counts = {}
            for r in limited_results:
                md = getattr(r, 'metadata', {}) or {}
                cat = md.get('source_category') or md.get('result_type') or 'general'
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        except Exception:
            cat_counts = {}

        # Bias distribution based on credibility_explanation (e.g., 'bias=left, fact=high, cat=academic')
        try:
            import re as _re
            bias_counts = {}
            for r in limited_results:
                md = getattr(r, 'metadata', {}) or {}
                expl = md.get('credibility_explanation') or ''
                m = _re.search(r"bias=([^,\s]+)", expl)
                if m:
                    b = m.group(1).strip().lower()
                    bias_counts[b] = bias_counts.get(b, 0) + 1
            # Normalize common variants
            normalized = {"left": 0, "center": 0, "right": 0, "mixed": 0}
            for k, v in bias_counts.items():
                if k in normalized:
                    normalized[k] += v
                else:
                    normalized["mixed"] += v
        except Exception:
            normalized = {}

        return {
            "results": compressed_results,
            "sources_used": list(self._search_metrics.get('apis_used', set())) if isinstance(self._search_metrics.get('apis_used'), set) else [],
            "credibility_summary": self._calculate_credibility_summary(limited_results),
            "category_distribution": cat_counts,
            "bias_distribution": normalized,
            "dedup_stats": {
                "original_count": len(raw_results),
                "deduplicated_count": len(deduplicated_results),
                "final_count": len(limited_results),
                "deduplication_rate": dedup_rate
            },
            "contradictions": contradictions,
        }

    def _detect_contradictions(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Heuristic cross-source contradiction detector over snippets/titles.

        Looks for opposing signal pairs (increase/decrease, effective/ineffective,
        positive/no/negative correlation, causes/does not cause) across different
        domains. Returns a small summary suitable for metadata/answer attachment.
        """
        if not items or len(items) < 2:
            return {"count": 0, "examples": []}

        import re
        pairs = [
            (re.compile(r"\b(increase(?:d|s)?|rise[s]?)\b", re.I), re.compile(r"\b(decrease(?:d|s)?|drop[s]?|lower)\b", re.I), "increase vs decrease"),
            (re.compile(r"\bpositive\s+correlation|positively\s+correlated\b", re.I), re.compile(r"\bno\s+correlation|negative\s+correlation|negatively\s+correlated\b", re.I), "pos vs no/neg correlation"),
            (re.compile(r"\b(effective|improv(es|ed)|beneficial)\b", re.I), re.compile(r"\b(ineffective|does\s+not\s+work|harms|detrimental)\b", re.I), "effective vs ineffective"),
            (re.compile(r"\b(causes|leads\s+to|results\s+in)\b", re.I), re.compile(r"\b(does\s+not\s+cause|no\s+causal|unrelated)\b", re.I), "causal vs non-causal"),
        ]

        def txt(it: Dict[str, Any]) -> str:
            return f"{it.get('title','')} {it.get('snippet','')}"

        def domain(it: Dict[str, Any]) -> str:
            md = it.get("metadata", {}) or {}
            return (md.get("domain") or "").lower()

        examples = []
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i], items[j]
                da, db = domain(a), domain(b)
                if not da or not db or da == db:
                    continue
                ta, tb = txt(a), txt(b)
                for p1, p2, label in pairs:
                    if p1.search(ta) and p2.search(tb) or p1.search(tb) and p2.search(ta):
                        examples.append({
                            "signal": label,
                            "a_url": a.get("url", ""),
                            "b_url": b.get("url", ""),
                            "a_excerpt": (a.get("snippet", "") or a.get("title", ""))[:180],
                            "b_excerpt": (b.get("snippet", "") or b.get("title", ""))[:180],
                        })
                        if len(examples) >= 5:
                            return {"count": len(examples), "examples": examples}
        return {"count": len(examples), "examples": examples}

    def _merge_duplicate_results(self, duplicates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge duplicate results deterministically"""
        merged = duplicates[0].copy()

        all_sources = set()
        all_queries = set()

        for dup in duplicates:
            all_sources.add(dup.get('search_api', 'unknown'))
            all_queries.add(dup.get('origin_query', ''))

            if len(dup.get('snippet', '')) > len(merged.get('snippet', '')):
                merged['snippet'] = dup['snippet']
            if len(dup.get('content', '')) > len(merged.get('content', '')):
                merged['content'] = dup['content']

        merged['all_sources'] = sorted(list(all_sources))
        merged['all_queries'] = sorted(list(all_queries))
        merged['duplicate_count'] = len(duplicates)

        return merged

    def _calculate_credibility_summary(
        self,
        results: List[SearchResultSchema]
    ) -> Dict[str, Any]:
        """Calculate overall credibility statistics"""
        if not results:
            return {"average_score": 0.0, "high_credibility_count": 0, "high_credibility_ratio": 0.0, "score_distribution": {"high": 0, "medium": 0, "low": 0}}

        scores = [float(getattr(r, "credibility_score", 0.0) or 0.0) for r in results]
        high_cred_count = sum(1 for s in scores if s is not None and s >= 0.7)

        return {
            "average_score": (sum(scores) / float(len(scores))) if scores else 0.0,
            "high_credibility_count": high_cred_count,
            "high_credibility_ratio": (high_cred_count / float(len(results))) if results else 0.0,
            "score_distribution": {
                "high": high_cred_count,
                "medium": sum(1 for s in scores if s is not None and 0.4 <= s < 0.7),
                "low": sum(1 for s in scores if s is not None and s < 0.4)
            }
        }

    def _apply_early_relevance_filter(
        self,
        results: List[SearchResult],
        query: str,
        paradigm: str
    ) -> List[SearchResult]:
        """Apply early-stage relevance filtering"""
        import os, re
        from services.text_compression import query_compressor

        # Baseline rule-based filter
        baseline: List[SearchResult] = []
        removed_rule = 0
        for result in results:
            try:
                ok = self.early_filter.is_relevant(result, query, paradigm)
            except Exception:
                ok = True
            if ok:
                baseline.append(result)
            else:
                removed_rule += 1
                logger.debug("[early] rule-drop: %s - %s", getattr(result, "domain", ""), (getattr(result, "title", "") or "")[:60])

        # Quick semantic theme-overlap filter (configurable)
        try:
            thr = float(os.getenv("EARLY_THEME_OVERLAP_MIN", "0.12") or 0.12)
            q_terms = set(query_compressor.extract_keywords(query))
            if not q_terms:
                return baseline

            def toks(text: str) -> set:
                return set([t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if len(t) > 2])

            filtered: List[SearchResult] = []
            removed_sem = 0
            for r in baseline:
                try:
                    title = getattr(r, "title", "") or ""
                    snippet = getattr(r, "snippet", "") or ""
                except Exception:
                    title = snippet = ""
                rtoks = toks(f"{title} {snippet}")
                if not rtoks:
                    filtered.append(r)
                    continue
                inter = len(q_terms & rtoks)
                union = len(q_terms | rtoks)
                jac = (inter / float(union)) if union else 0.0
                if jac >= thr:
                    filtered.append(r)
                else:
                    removed_sem += 1
                    logger.debug("[early] theme-drop (%.3f<thr): %s - %s", jac, getattr(r, "domain", ""), title[:60])

            removed_total = removed_rule + removed_sem
            if removed_total > 0:
                logger.info("Early filter removed %d results (rule=%d, theme=%d)", removed_total, removed_rule, removed_sem)
            return filtered
        except Exception:
            if removed_rule > 0:
                logger.info("Early filter removed %d results (rule)", removed_rule)
            return baseline

    def normalize_result_shape(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize result to ensure consistent shape with required fields"""
        url = (result.get('url') or '').strip()
        title = result.get('title')
        snippet = result.get('snippet')
        content = result.get('content')
        source_api = result.get('source_api', result.get('search_api', 'unknown'))
        origin_query = result.get('origin_query', '')
        origin_query_id = result.get('origin_query_id', '')
        metadata = result.get('metadata', {}) or {}
        result_type = result.get('result_type', '')

        # Enforce URL presence with diagnostics and optional synthetic handling for deep citations
        if self.diagnostics.get("enforce_url_presence") and not url:
            allow_unlinked = self.diagnostics.get("allow_unlinked_citations", False)
            if allow_unlinked and result_type == "deep_research" and (title or snippet):
                # Synthesize stable placeholder URL for traceability
                safe_oid = origin_query_id or "deep"
                synthesized = f"about:blank#citation-{safe_oid}-{hash(((title or snippet) or '')[:64]) & 0xFFFFFFFF}"
                url = synthesized
                metadata = dict(metadata)
                metadata["unlinked_citation"] = True
            else:
                # record diagnostics sample and counter, then drop
                self._search_metrics["dropped_no_url"] = int(self._search_metrics.get("dropped_no_url", 0)) + 1
                samples = self._diag_samples.get("no_url", [])
                if len(samples) < 5:
                    samples.append({
                        "title": (title or "")[:80],
                        "api": source_api,
                        "origin_query_id": origin_query_id
                    })
                    self._diag_samples["no_url"] = samples
                if self.diagnostics.get("log_result_normalization"):
                    logger.debug(
                        "[normalize] dropped result without URL: title='%s', api=%s",
                        (title or "")[:60],
                        source_api,
                    )
                return None

        normalized = {
            'title': title or (url.split('/')[-1] if url else '(untitled)'),
            'url': url,
            'snippet': snippet or '',
            'content': content or '',
            'source_api': source_api,
            'credibility_score': result.get('credibility_score'),
            'origin_query': origin_query,
            'origin_query_id': origin_query_id,
            'metadata': metadata
        }

        # Validate minimal shape; drop invalid with metric
        if not isinstance(normalized['title'], str) or not isinstance(normalized['snippet'], str):
            self._search_metrics["dropped_invalid_shape"] = int(self._search_metrics.get("dropped_invalid_shape", 0)) + 1
            if self.diagnostics.get("log_result_normalization"):
                logger.debug("[normalize] dropped invalid shape: api=%s oqid=%s", source_api, origin_query_id)
            return None

        # Ensure origin markers
        if self.diagnostics.get("enforce_per_result_origin"):
            if not normalized['origin_query_id']:
                normalized['origin_query_id'] = metadata.get('origin_query_id', '')
            if not normalized['origin_query'] and metadata.get('origin_query'):
                normalized['origin_query'] = metadata['origin_query']

        if self.diagnostics.get("log_result_normalization"):
            logger.debug(
                "[normalize] url=%s api=%s oqid=%s title='%s'",
                normalized["url"],
                normalized["source_api"],
                normalized["origin_query_id"],
                normalized["title"][:60],
            )
        return normalized

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

    def _convert_legacy_to_v2_result(self, legacy_result: ResearchExecutionResult) -> Dict[str, Any]:
        """Convert legacy ResearchExecutionResult to V2 format with consistent dict handling"""
        from .result_adapter import adapt_results

        # Use ResultAdapter to safely handle filtered_results
        try:
            adapter = adapt_results(legacy_result.filtered_results)

            # Use duck-typing without strict attribute checks to avoid Pylance issues
            results_dicts = []
            try:
                valid_results = getattr(adapter, "get_valid_results", lambda: [])()
                iterable = valid_results if isinstance(valid_results, list) else []
            except Exception:
                iterable = []
            for result in iterable:
                try:
                    result_dict = {
                        "url": getattr(result, "url", ""),
                        "title": getattr(result, "title", ""),
                        "snippet": getattr(result, "snippet", ""),
                        "source_api": getattr(result, "source_api", "unknown"),
                        "credibility_score": getattr(result, "credibility_score", 0.5) or 0.5,
                        "origin_query": getattr(legacy_result, "original_query", ""),
                        "paradigm_alignment": getattr(legacy_result, "paradigm", "unknown")
                    }
                    normalized = self.normalize_result_shape(result_dict)
                    if normalized:
                        results_dicts.append(normalized)
                except Exception:
                    continue
            if not results_dicts:
                # Try single adapter fields
                try:
                    result_dict = {
                        "url": getattr(adapter, "url", ""),
                        "title": getattr(adapter, "title", ""),
                        "snippet": getattr(adapter, "snippet", ""),
                        "source_api": getattr(adapter, "source_api", "unknown"),
                        "credibility_score": getattr(adapter, "credibility_score", 0.5) or 0.5,
                        "origin_query": getattr(legacy_result, "original_query", ""),
                        "paradigm_alignment": getattr(legacy_result, "paradigm", "unknown")
                    }
                    normalized = self.normalize_result_shape(result_dict)
                    if normalized:
                        results_dicts = [normalized]
                except Exception:
                    results_dicts = []
        except Exception as e:
            logger.error(f"Error converting legacy results: {e}")
            # Fallback to original approach
            results_dicts = []
            for r in legacy_result.filtered_results:
                try:
                    result_dict = {
                        "url": getattr(r, 'url', ''),
                        "title": getattr(r, 'title', ''),
                        "snippet": getattr(r, 'snippet', ''),
                        "source_api": getattr(r, "source_api", "unknown"),
                        "credibility_score": getattr(r, "credibility_score", 0.5),
                        "origin_query": legacy_result.original_query,
                        "paradigm_alignment": legacy_result.paradigm
                    }
                    normalized = self.normalize_result_shape(result_dict)
                    if normalized:
                        results_dicts.append(normalized)
                except Exception as result_error:
                    logger.warning(f"Skipping problematic result: {result_error}")
                    continue

        return {
            "results": results_dicts,
            "metadata": {
                "total_results": len(results_dicts),
                "queries_executed": len(getattr(legacy_result, "search_queries_executed", []) or []),
                "sources_used": list(set(r.get("source_api", "unknown") for r in results_dicts)) if results_dicts else [],
                "credibility_summary": {
                    "average_score": (
                        sum(list((legacy_result.credibility_scores or {}).values())) / float(max(len((legacy_result.credibility_scores or {})), 1))
                        if getattr(legacy_result, "credibility_scores", None) else 0.0
                    )
                },
                "deduplication_stats": {
                    "original_count": int((legacy_result.execution_metrics or {}).get("raw_results_count", 0)) if getattr(legacy_result, "execution_metrics", None) else 0,
                    "final_count": len(results_dicts)
                },
                "search_metrics": dict(getattr(legacy_result, "execution_metrics", {}) or {}),
                "processing_time": float((legacy_result.execution_metrics or {}).get("processing_time_seconds", 0)) if getattr(legacy_result, "execution_metrics", None) else 0.0,
                "paradigm": getattr(legacy_result, "paradigm", "unknown")
            }
        }

    async def execute_deep_research(
        self,
        context_engineered: ContextEngineeredQuerySchema,
        enable_standard_search: bool = True,
        deep_research_mode: DeepResearchMode = DeepResearchMode.PARADIGM_FOCUSED,
        search_context_size: Optional[Any] = None,
        user_location: Optional[Dict[str, str]] = None,
        progress_tracker: Optional[Any] = None,
        research_id: Optional[str] = None,
    ) -> ResearchExecutionResult:
        """
        Public wrapper to run deep research and (optionally) standard search,
        returning a ResearchExecutionResult compatible object for downstream consumers.
        """
        # 1) Build a fallback user_context-shaped object (role=PRO) if missing
        class _UserCtxShim:
            def __init__(self, role="PRO", source_limit=10, is_pro_user=True, preferences=None):
                self.role = role
                self.source_limit = source_limit
                self.is_pro_user = is_pro_user
                self.preferences = preferences or {}
        try:
            uc = getattr(context_engineered, "user_context", None)
            role = getattr(uc, "role", "PRO") if uc else "PRO"
            source_limit = getattr(uc, "source_limit", 10) if uc else 10
            is_pro_user = getattr(uc, "is_pro_user", True) if uc else True
            preferences = getattr(uc, "preferences", {}) if uc else {}
            user_context = _UserCtxShim(role, source_limit, is_pro_user, preferences)
        except Exception:
            user_context = _UserCtxShim()

        # 2) Determine a classification to pass to deep integration. Prefer context_engineered.classification if present.
        classification = getattr(context_engineered, "classification", None)
        if classification is None:
            # Create a minimal default classification targeting analytical/Bernard to avoid None paths
            try:
                orig_q = getattr(context_engineered, "original_query", "")
                classification = ClassificationResultSchema(
                    query=orig_q,
                    primary_paradigm=HostParadigm.BERNARD,
                    secondary_paradigm=None,
                    distribution={HostParadigm.BERNARD: 1.0},
                    confidence=0.9,
                    features=QueryFeaturesSchema(text=orig_q),
                    reasoning={HostParadigm.BERNARD: ["default"]}
                )
            except Exception:
                # If constructor differs, try attribute assignment fallback
                class _C:  # lightweight shim
                    primary_paradigm = HostParadigm.BERNARD
                    secondary_paradigm = None
                    confidence = 0.9
                classification = _C()  # type: ignore

        # 3) Run deep research via the internal integration
        deep_items: List[Dict[str, Any]] = []
        deep_content: Optional[str] = None
        try:
            deep_items = await self._execute_deep_research_integration(
                context_engineered=context_engineered,
                classification=classification,  # type: ignore[arg-type]
                user_context=user_context,      # shim object accepted internally
                mode=deep_research_mode,
                progress_callback=progress_tracker,
                research_id=research_id
            )
        except Exception as e:
            logger.error(f"Deep research integration failed: {e}")
            deep_items = []

        # Attempt to extract content body from deep research service if available
        # The internal integration currently returns only list[dict]. If upstream adds content,
        # we set it below when available from deep result; otherwise keep None.
        try:
            # Some implementations may attach last deep result content into cache/manager; safe best-effort retrieval.
            deep_content = None
        except Exception:
            deep_content = None

        # 4) Optionally execute legacy/standard search to produce a ResearchExecutionResult baseline
        legacy_result: Optional[ResearchExecutionResult] = None
        if enable_standard_search:
            try:
                legacy_result = await self._execute_paradigm_research_legacy(
                    context_engineered_query=context_engineered,
                    max_results=int(getattr(user_context, "source_limit", 10)),
                    progress_tracker=progress_tracker,
                    research_id=research_id
                )
            except Exception as e:
                logger.error(f"Legacy research path failed: {e}")
                legacy_result = None

        # If legacy result not available, fabricate a minimal ResearchExecutionResult container
        if legacy_result is None:
            try:
                legacy_result = ResearchExecutionResult(
                    original_query=getattr(context_engineered, "original_query", ""),
                    paradigm=getattr(classification.primary_paradigm, "value", str(getattr(classification, "primary_paradigm", "bernard"))),
                    secondary_paradigm=getattr(getattr(classification, "secondary_paradigm", None), "value", None),
                    search_queries_executed=[],
                    raw_results={},
                    filtered_results=[],
                    secondary_results=[],
                    credibility_scores={},
                    execution_metrics={
                        "queries_executed": 0,
                        "raw_results_count": 0,
                        "deduplicated_count": 0,
                        "final_results_count": 0,
                        "deep_research_enabled": True
                    },
                    cost_breakdown={},
                    status=ContractResearchStatus.FAILED_NO_SOURCES
                )
            except Exception as e:
                logger.error(f"Failed to construct minimal ResearchExecutionResult: {e}")
                # As last resort, raise to signal upstream contract mismatch
                raise

        # 5) Transform deep results to SearchResult-like shims and append to filtered_results
        appended = 0
        for d in deep_items or []:
            try:
                url = (d.get("url") or "").strip()
                title = d.get("title") or (url.split("/")[-1] if url else "(deep research)")
                snippet = d.get("snippet") or d.get("summary") or ""
                source_api = d.get("source_api") or d.get("search_api") or "deep_research"
                credibility_score = d.get("credibility_score", 0.8)

                # Build a lightweight SearchResult-like object using SearchResultSchema if available
                try:
                    meta = d.get("metadata", {}) or {}
                    meta.update({
                        "source_api": source_api,
                        "credibility_explanation": d.get("credibility_explanation", ""),
                        "origin_query": d.get("origin_query", getattr(context_engineered, "original_query", "")),
                        "origin_query_id": d.get("origin_query_id", "deep"),
                        "paradigm_alignment": getattr(classification, "primary_paradigm", HostParadigm.BERNARD)
                    })
                    shim = SearchResultSchema(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=(meta.get("domain") or source_api),
                        credibility_score=credibility_score,
                        relevance_score=0.7,
                        metadata=meta
                    )
                    # Append as-is; downstream consumers typically accept SearchResult-like items
                    legacy_result.filtered_results.append(shim)  # type: ignore
                    appended += 1
                except Exception:
                    # Fallback: try the SearchResult dataclass if available from services.search_apis
                    try:
                        shim2 = SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source=source_api,
                            domain=(url.split('/')[2] if url else ""),
                            content=snippet
                        )
                        legacy_result.filtered_results.append(shim2)  # type: ignore
                        appended += 1
                    except Exception:
                        # As last resort, keep a dict
                        legacy_result.filtered_results.append(d)  # type: ignore
                        appended += 1
            except Exception as e:
                logger.warning(f"Skipping deep result append due to error: {e}")

        # 6) Attach deep research content if available
        if deep_content:
            legacy_result.deep_research_content = deep_content

        # 7) Update execution metrics and cost breakdown conservatively
        try:
            legacy_result.execution_metrics = legacy_result.execution_metrics or {}
            legacy_result.execution_metrics["deep_research_appended"] = appended
            legacy_result.execution_metrics["deep_research_enabled"] = True
        except Exception:
            pass

        try:
            legacy_result.cost_breakdown = legacy_result.cost_breakdown or {}
            # If you later compute deep cost, merge here. For now, set zero-cost marker.
            legacy_result.cost_breakdown["deep_research"] = legacy_result.cost_breakdown.get("deep_research", 0.0)
        except Exception:
            pass

        return legacy_result

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

async def execute_research(
    context_engineered_query, max_results: int = 50
) -> ResearchExecutionResult:
    """Convenience function to execute research"""
    return await research_orchestrator.execute_paradigm_research(
        context_engineered_query, max_results
    )

# Alias for imports expecting different names
research_orchestrator_v2 = research_orchestrator
unified_orchestrator = research_orchestrator
enhanced_orchestrator = research_orchestrator

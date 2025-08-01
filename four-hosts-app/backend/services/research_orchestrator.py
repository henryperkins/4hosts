"""
Unified Research Orchestrator V2
Combines all the best features from multiple orchestrator implementations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
# json import removed (unused)
import re

from models.context_models import (
    SearchResultSchema, ClassificationResultSchema,
    ContextEngineeredQuerySchema,
    HostParadigm
)
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
from services.text_compression import text_compressor, query_compressor
from services.deep_research_service import (
    deep_research_service,
    DeepResearchConfig,
    DeepResearchMode,
    initialize_deep_research,
)
# SearchContextSize import removed (unused)

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
    execution_metrics: Dict[str, Any]
    cost_breakdown: Dict[str, float]
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
        if not results:
            return {
                "unique_results": [],
                "duplicates_removed": 0,
                "similarity_threshold": self.similarity_threshold
            }

        unique_results = []
        duplicates_removed = 0
        seen_urls = set()

        # First pass: exact URL deduplication
        url_deduplicated = []
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                url_deduplicated.append(result)
            else:
                duplicates_removed += 1

        # Second pass: content similarity deduplication
        for result in url_deduplicated:
            is_duplicate = False

            for existing in unique_results:
                similarity = self._calculate_content_similarity(
                    result,
                    existing,
                )

                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    break

            if not is_duplicate:
                unique_results.append(result)

        logger.info(
            f"Deduplication: {len(results)} -> {len(unique_results)} "
            f"({duplicates_removed} duplicates removed)"
        )

        return {
            "unique_results": unique_results,
            "duplicates_removed": duplicates_removed,
            "similarity_threshold": self.similarity_threshold
        }

    def _calculate_content_similarity(
        self, result1: SearchResult, result2: SearchResult
    ) -> float:
        """Calculate similarity between two search results"""
        # Title similarity (Jaccard index)
        title1_words = set(result1.title.lower().split())
        title2_words = set(result2.title.lower().split())

        if not title1_words or not title2_words:
            title_similarity = 0.0
        else:
            intersection = len(title1_words.intersection(title2_words))
            union = len(title1_words.union(title2_words))
            title_similarity = intersection / union if union > 0 else 0.0

        # Domain similarity
        domain_similarity = 1.0 if result1.domain == result2.domain else 0.0

        # Snippet similarity
        snippet1_words = set(result1.snippet.lower().split())
        snippet2_words = set(result2.snippet.lower().split())

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
        if not result.title or len(result.title.strip()) < 10:
            return False
        if not result.snippet or len(result.snippet.strip()) < 20:
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

        # Orchestration additions
        self.tool_registry = ToolRegistry()
        self.retry_policy = RetryPolicy()
        self.planner = BudgetAwarePlanner(self.tool_registry, self.retry_policy)

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

        # Performance tracking
        self.execution_history = []

        # Diagnostics toggles
        self.diagnostics = {
            "log_task_creation": True,
            "log_result_normalization": True,
            "log_credibility_failures": True,
            "enforce_url_presence": True,
            "enforce_per_result_origin": True
        }

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

    async def execute_research(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
        enable_deep_research: bool = False,
        deep_research_mode: Optional[DeepResearchMode] = None
    ) -> Dict[str, Any]:
        """Execute research with all enhanced features"""

        start_time = datetime.now()

        # Use the old execute_paradigm_research for backward compatibility
        if hasattr(context_engineered, 'select_output'):
            # Legacy format - convert and use old method
            result = await self._execute_paradigm_research_legacy(
                context_engineered,
                max_results=user_context.source_limit,
                progress_tracker=progress_callback,
                research_id=research_id
            )

            # Convert to new format
            return self._convert_legacy_to_v2_result(result)

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

        # Execute searches with deterministic ordering
        search_results = await self._execute_searches_deterministic(
            optimized_queries,
            classification.primary_paradigm,
            user_context,
            progress_callback,
            research_id
        )

        # Process and enrich results
        processed_results = await self._process_results(
            search_results,
            classification,
            context_engineered,
            user_context
        )

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

        return {
            "results": processed_results["results"],
            "metadata": {
                **processed_results["metadata"],
                "total_results": len(processed_results["results"]),
                "queries_executed": len(optimized_queries),
                "sources_used": processed_results["sources_used"],
                "credibility_summary": processed_results["credibility_summary"],
                "deduplication_stats": processed_results["dedup_stats"],
                "search_metrics": {
                "total_queries": int(self._search_metrics.get("total_queries", 0) if isinstance(self._search_metrics.get("total_queries"), int) else 0),
                "total_results": int(self._search_metrics.get("total_results", 0) if isinstance(self._search_metrics.get("total_results"), int) else 0),
                "apis_used": list(self._search_metrics.get("apis_used", set())) if isinstance(self._search_metrics.get("apis_used"), set) else [],
                "deduplication_rate": float(self._search_metrics.get("deduplication_rate", 0.0) if isinstance(self._search_metrics.get("deduplication_rate"), (int, float)) else 0.0)
            },
                "paradigm": classification.primary_paradigm.value if hasattr(classification, 'primary_paradigm') else 'unknown'
            }
        }

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
        research_id: Optional[str] = None
    ) -> ResearchExecutionResult:
        """Execute research using legacy format"""

        start_time = datetime.now()
        metrics = {"start_time": start_time.isoformat()}

        # Extract information from context engineering
        original_query = context_engineered_query.original_query
        classification = context_engineered_query.classification
        select_output = context_engineered_query.select_output

        # Map enum values to paradigm names
        paradigm_mapping = {
            "revolutionary": "dolores",
            "devotion": "teddy",
            "analytical": "bernard",
            "strategic": "maeve",
        }

        paradigm = paradigm_mapping.get(
            classification.primary_paradigm.value,
            "bernard",
        )
        secondary_paradigm = None
        if classification.secondary_paradigm:
            secondary_paradigm = paradigm_mapping.get(
                classification.secondary_paradigm.value, None
            )

        logger.info(f"Executing research for paradigm: {paradigm}")

        # Create search context
        search_context = SearchContext(
            original_query=original_query,
            paradigm=paradigm,
            secondary_paradigm=secondary_paradigm,
        )

        # Get paradigm-specific search strategy
        strategy = get_search_strategy(paradigm)

        # Use search queries from Context Engineering Select layer with guards/normalization
        raw_sq = getattr(select_output, "search_queries", None)
        search_queries = []
        if isinstance(raw_sq, list):
            # Accept list of dicts or strings; coerce to list[dict]
            for item in raw_sq:
                if isinstance(item, dict):
                    q = (item.get("query") or "").strip()
                    if q:
                        t = item.get("type", "generic")
                        try:
                            w = float(item.get("weight", 1.0) or 1.0)
                        except Exception:
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
                except Exception:
                    w = 1.0
                search_queries.append({"query": q, "type": raw_sq.get("type", "generic"), "weight": w})
        # Fallback if missing/empty
        if not search_queries:
            oq = getattr(context_engineered_query, "original_query", "") or ""
            if oq:
                search_queries = [{"query": oq, "type": "generic", "weight": 1.0}]
            else:
                # last resort default
                search_queries = [{"query": "news", "type": "generic", "weight": 1.0}]
        # Cap to 8
        search_queries = search_queries[:8]

        logger.info(f"Executing {len(search_queries)} search queries")

        # Execute searches with caching
        all_results = {}
        cost_breakdown = {}

        for idx, query_data in enumerate(search_queries):
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
                        f"Searching: {query[:40]}...",
                        search_progress
                    )

                config = SearchConfig(
                    max_results=min(max_results, 50), language="en", region="us"
                )

                try:
                    if not self.search_manager:
                        raise RuntimeError("search_manager is not initialized")
                    api_results = await self.search_manager.search_with_fallback(
                        query, config
                    )

                    primary_api = "google"
                    cost = await self.cost_monitor.track_search_cost(primary_api, 1)
                    cost_breakdown[f"{query_type}_{query[:20]}"] = cost

                    for result in api_results:
                        result.credibility_score = (
                            result.credibility_score * weight
                            if hasattr(result, "credibility_score")
                            else weight
                        )

                    await cache_search_results(
                        query, config_dict, paradigm, api_results
                    )
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

                # Check if content is not empty
                if not getattr(result, 'content', '').strip():
                    logger.warning(f"Result has empty content in query '{query_key}', skipping")
                    continue

                valid_results.append(result)

            combined_results.extend(valid_results)
            logger.debug(f"Query '{query_key}': {len(query_results)} -> {len(valid_results)} valid results")

        logger.info(f"Combined {len(combined_results)} total validated results")

        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id, "Removing duplicate results...", 52
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

            # Ensure sections exist for answer generation
            # Guard attribute access since SearchResult doesn't define 'sections' in dataclass
            if not hasattr(result, 'sections') or result.sections is None:
                try:
                    setattr(result, 'sections', [])
                except Exception:
                    # If object is frozen or doesn't allow new attrs, skip silently
                    pass

            validated_results.append(result)

        # Update deduplicated_results with validated ones
        deduplicated_results = validated_results

        logger.info(f"Validation after dedup: {len(dedup_result['unique_results'])} -> {len(deduplicated_results)} valid results")

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
                research_id, "Evaluating source credibility...", 55
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

        # Limit final results
        final_results = filtered_results[:max_results]

        # Execute secondary search if applicable
        secondary_results = []
        if secondary_paradigm:
            logger.info(
                f"Executing secondary research for paradigm: {secondary_paradigm}"
            )
            secondary_strategy = get_search_strategy(secondary_paradigm)
            secondary_query = f"{original_query} {secondary_paradigm}"

            config = SearchConfig(
                max_results=min(max_results // 2, 25), language="en", region="us"
            )
            try:
                if not self.search_manager:
                    raise RuntimeError("search_manager is not initialized")
                api_results = await self.search_manager.search_with_fallback(
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
        if isinstance(metrics, dict):
            metrics.update(metrics_dict)
        else:
            metrics = metrics_dict

        # Create execution result
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
        )

        # Store in history
        self.execution_history.append(result)

        logger.info(f"Research execution completed in {processing_time:.2f}s")
        logger.info(f"Final results: {len(final_results)} sources")

        # Enhanced logging for debugging
        if final_results:
            logger.debug("Final results detailed breakdown:")
            for i, result in enumerate(final_results[:5]):  # Log first 5 results
                if result is None:
                    logger.warning(f"Result {i} is None")
                    continue

                title = getattr(result, 'title', 'No title')
                url = getattr(result, 'url', 'No URL')
                content_length = len(getattr(result, 'content', ''))
                sections_count = len(getattr(result, 'sections', []))

                logger.debug(f"  Result {i}: '{title[:50]}...' ({content_length} chars, {sections_count} sections) - {url}")

                # Check for sections attribute specifically
                if hasattr(result, 'sections'):
                    if result.sections is None:
                        logger.warning(f"  Result {i} has None sections attribute")
                    elif not isinstance(result.sections, list):
                        logger.warning(f"  Result {i} sections is not a list: {type(result.sections)}")
                else:
                    logger.warning(f"  Result {i} missing sections attribute")
        else:
            logger.warning("No final results produced - this may cause downstream errors")

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
        research_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Execute searches with per-task timeout, retries, and deterministic ordering"""
        entries: List[Dict[str, Any]] = []
        for qidx, query in enumerate(queries):
            for api in self._select_apis_for_paradigm(paradigm, user_context):
                entries.append({"query_index": qidx, "query": query, "api": api})

        if progress_callback and research_id:
            await progress_callback(f"Executing {len(entries)} search operations")

        processed_results: List[Dict[str, Any]] = []
        # Execute sequentially per entry to enable precise retry/backoff and deterministic ordering
        for idx, meta in enumerate(entries):
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
                attempt += 1
                try:
                    config = SearchConfig(max_results=20, language="en", region="us")
                    if self._supports_search_with_api:
                        coro = self.search_manager.search_with_api(api, query, config)
                    else:
                        # Fallback will ignore api specificity; still tracked for metrics
                        coro = self.search_manager.search_with_fallback(query, config)

                    # Per-attempt timeout
                    task_result = await asyncio.wait_for(coro, timeout=10)
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
            # Ensure the selected API is actually used by search manager if supported
            if hasattr(self.search_manager, "search_with_api"):
                task = asyncio.create_task(
                    self.search_manager.search_with_api(
                        api_name,
                        query,
                        config,
                    ),
                    name=f"{api_name}_q{query_index}",
                )
            else:
                # Fallback to generic with_fallback
                task = asyncio.create_task(
                    self.search_manager.search_with_fallback(query, config),
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
                url=merged_url,
                title=merged.get('title', ''),
                snippet=merged.get('snippet', ''),
                credibility_score=credibility_score,
                metadata=extra_meta
            )

            deduplicated_results.append(result_schema)

        deduplicated_results.sort(
            key=lambda x: (x.credibility_score, x.url),
            reverse=True
        )

        limited_results = deduplicated_results[: int(getattr(user_context, "source_limit", 10))]

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

        try:
            compressed_results = text_compressor.compress_search_results(
                result_payload,
                total_token_budget=3000
            )
            self._search_metrics["compression_plural_used"] = int(
                self._search_metrics.get("compression_plural_used", 0)
            ) + 1
        except Exception:
            compressed_results = [
                text_compressor.compress_search_result(item) for item in result_payload
            ]
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

        return {
            "results": compressed_results,
            "sources_used": list(self._search_metrics.get('apis_used', set())) if isinstance(self._search_metrics.get('apis_used'), set) else [],
            "credibility_summary": self._calculate_credibility_summary(limited_results),
            "dedup_stats": {
                "original_count": len(raw_results),
                "deduplicated_count": len(deduplicated_results),
                "final_count": len(limited_results),
                "deduplication_rate": dedup_rate
            }
        }

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
        filtered = []
        removed_count = 0

        for result in results:
            if self.early_filter.is_relevant(result, query, paradigm):
                filtered.append(result)
            else:
                removed_count += 1
                logger.debug(f"Filtered out: {result.domain} - {result.title[:50]}...")

        if removed_count > 0:
            logger.info(f"Early relevance filter removed {removed_count} irrelevant results")

        return filtered

    def normalize_result_shape(self, result: Dict[str, Any]) -> Dict[str, Any]:
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
                synthesized = f"about:blank#citation-{safe_oid}-{hash((title or snippet)[:64]) & 0xFFFFFFFF}"
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
                synthesized = f"about:blank#citation-{safe_oid}-{hash((title or snippet)[:64]) & 0xFFFFFFFF}"
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
                classification = ClassificationResultSchema(
                    primary_paradigm=HostParadigm.BERNARD,
                    secondary_paradigm=None,
                    confidence=0.9
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
                    cost_breakdown={}
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
                        url=url,
                        title=title,
                        snippet=snippet,
                        credibility_score=credibility_score,
                        metadata=meta
                    )
                    # Append as-is; downstream consumers typically accept SearchResult-like items
                    legacy_result.filtered_results.append(shim)  # type: ignore
                    appended += 1
                except Exception:
                    # Fallback: try the SearchResult dataclass if available from services.search_apis
                    try:
                        shim2 = SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet,
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


# Global orchestrator instance
research_orchestrator = UnifiedResearchOrchestrator()

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

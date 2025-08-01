"""
Unified Research Orchestrator V2
Combines all the best features from multiple orchestrator implementations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import re

from models.context_models import (
    SearchResultSchema, ClassificationResultSchema,
    ContextEngineeredQuerySchema, UserContextSchema,
    HostParadigm, QueryFeaturesSchema
)
from services.search_apis import SearchAPIManager, SearchResult, SearchConfig, create_search_manager
from services.credibility import get_source_credibility, CredibilityScore
from services.paradigm_search import get_search_strategy, SearchContext
from services.cache import cache_manager, get_cached_search_results, cache_search_results
from services.text_compression import text_compressor, query_compressor
from services.deep_research_service import (
    deep_research_service,
    DeepResearchConfig,
    DeepResearchMode,
    initialize_deep_research,
)
from services.openai_responses_client import SearchContextSize

logger = logging.getLogger(__name__)


@dataclass
class ResearchExecutionResult:
    """Complete research execution result"""
    original_query: str
    paradigm: str
    secondary_paradigm: Optional[str]
    search_queries_executed: List[Dict[str, Any]]
    raw_results: Dict[str, List[SearchResult]]  # Results by API
    filtered_results: List[SearchResult]
    credibility_scores: Dict[str, float]  # Domain -> score
    execution_metrics: Dict[str, Any]
    cost_breakdown: Dict[str, float]
    secondary_results: List[SearchResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    deep_research_content: Optional[str] = None


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
                similarity = self._calculate_content_similarity(result, existing)

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
            title_similarity * 0.5 + domain_similarity * 0.2 + snippet_similarity * 0.3
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

        # Metrics
        self._search_metrics = {
            "total_queries": 0,
            "total_results": 0,
            "apis_used": set(),
            "deduplication_rate": 0.0
        }

        # Performance tracking
        self.execution_history = []

    async def initialize(self):
        """Initialize the orchestrator"""
        self.search_manager = create_search_manager()
        await self.search_manager.initialize()
        await cache_manager.initialize()
        await initialize_deep_research()

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
        user_context: UserContextSchema,
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
        optimized_queries = query_compressor.optimize_query_batch(
            context_engineered.refined_queries,
            max_queries=self._get_query_limit(user_context),
            paradigm=classification.primary_paradigm.value
        )

        logger.info(
            f"Optimized {len(context_engineered.refined_queries)} queries "
            f"to {len(optimized_queries)} for {classification.primary_paradigm.value}"
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
            deep_results = await self._execute_deep_research_integration(
                context_engineered,
                classification,
                user_context,
                deep_research_mode or DeepResearchMode.PARADIGM_FOCUSED,
                progress_callback,
                research_id
            )

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
                "search_metrics": dict(self._search_metrics),
                "paradigm": classification.primary_paradigm.value if hasattr(classification, 'primary_paradigm') else 'unknown'
            }
        }

    async def execute_paradigm_research(
        self,
        context_engineered_query,
        max_results: int = 100,
        progress_tracker=None,
        research_id: str = None
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
        research_id: str = None
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

        # Use search queries from Context Engineering Select layer
        search_queries = select_output.search_queries[:8]

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
            if hasattr(result, 'sections'):
                if result.sections is None:
                    logger.warning(f"Result has None sections: {getattr(result, 'title', 'Unknown')}")
                    # Create basic sections structure
                    result.sections = []
            elif not hasattr(result, 'sections'):
                # Add sections attribute if missing
                result.sections = []
                
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

        metrics.update(
            {
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
        )

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

    def _get_query_limit(self, user_context: UserContextSchema) -> int:
        """Get query limit based on user context"""
        base_limits = {
            "FREE": 3,
            "BASIC": 5,
            "PRO": 10,
            "ENTERPRISE": 20,
            "ADMIN": 30
        }
        return base_limits.get(user_context.role, 3)

    async def _execute_searches_deterministic(
        self,
        queries: List[str],
        paradigm: HostParadigm,
        user_context: UserContextSchema,
        progress_callback: Optional[Any],
        research_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Execute searches with guaranteed deterministic ordering"""

        search_tasks = []
        task_metadata = []

        for idx, query in enumerate(queries):
            apis_for_query = self._select_apis_for_paradigm(paradigm, user_context)

            for api_name in apis_for_query:
                task = self._create_search_task(api_name, query, idx)
                if task:
                    search_tasks.append(task)
                    task_metadata.append({
                        "query_index": idx,
                        "query": query,
                        "api": api_name
                    })

        if progress_callback and research_id:
            await progress_callback(f"Executing {len(search_tasks)} search operations")

        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        processed_results = []

        for idx, (result, metadata) in enumerate(zip(results, task_metadata)):
            if isinstance(result, Exception):
                logger.error(f"Search failed for {metadata['api']}: {result}")
                continue

            if result and isinstance(result, list):
                for item in result:
                    if hasattr(item, 'to_dict'):
                        item_dict = item.to_dict()
                    elif hasattr(item, '__dict__'):
                        item_dict = item.__dict__.copy()
                    else:
                        item_dict = dict(item) if isinstance(item, dict) else {}

                    item_dict['origin_query'] = metadata['query']
                    item_dict['origin_query_id'] = f"q{metadata['query_index']}"
                    item_dict['search_api'] = metadata['api']
                    item_dict['result_index'] = idx

                    # Normalize the result shape
                    normalized_item = self.normalize_result_shape(item_dict)
                    if normalized_item:  # Only add if URL is present
                        processed_results.append(normalized_item)

        processed_results.sort(key=lambda x: (
            x.get('origin_query_id', ''),
            x.get('search_api', ''),
            x.get('url', ''),
            x.get('title', '')
        ))

        self._search_metrics['total_queries'] += len(queries)
        self._search_metrics['total_results'] += len(processed_results)

        return processed_results

    def _select_apis_for_paradigm(
        self,
        paradigm: HostParadigm,
        user_context: UserContextSchema
    ) -> List[str]:
        """Select appropriate APIs based on paradigm and user context"""

        paradigm_apis = {
            HostParadigm.DOLORES: ["brave", "google"],
            HostParadigm.BERNARD: ["google", "arxiv", "pubmed"],
            HostParadigm.MAEVE: ["google", "brave"],
            HostParadigm.TEDDY: ["google"]
        }

        selected = paradigm_apis.get(paradigm, ["google"])

        if not user_context.is_pro_user and len(selected) > 2:
            selected = selected[:2]

        if user_context.preferences.get("preferred_sources"):
            selected.extend(user_context.preferences["preferred_sources"])

        seen = set()
        unique_apis = []
        for api in selected:
            if api not in seen:
                seen.add(api)
                unique_apis.append(api)
                self._search_metrics['apis_used'].add(api)

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
            return asyncio.create_task(
                self.search_manager.search_with_fallback(query, config),
                name=f"{api_name}_q{query_index}"
            )
        except Exception as e:
            logger.error(f"Failed to create task for {api_name}: {e}")
        return None

    async def _process_results(
        self,
        raw_results: List[Dict[str, Any]],
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: UserContextSchema
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
            domain = urlparse(merged.get('url', '')).netloc
            credibility_score, credibility_explanation, credibility_status = await self.get_source_credibility_safe(
                domain,
                classification.primary_paradigm.value
            )
            merged['credibility_score'] = credibility_score
            merged['credibility_explanation'] = credibility_explanation
            merged['credibility_status'] = credibility_status
            merged['paradigm_alignment'] = classification.primary_paradigm.value

            result_schema = SearchResultSchema(
                url=merged['url'],
                title=merged.get('title', ''),
                snippet=merged.get('snippet', ''),
                source_api=merged.get('search_api', 'unknown'),
                credibility_score=credibility_score,
                credibility_explanation=credibility_explanation,
                origin_query=merged.get('origin_query'),
                origin_query_id=merged.get('origin_query_id'),
                paradigm_alignment=classification.primary_paradigm,
                metadata=merged.get('metadata', {})
            )

            deduplicated_results.append(result_schema)

        deduplicated_results.sort(
            key=lambda x: (x.credibility_score, x.url),
            reverse=True
        )

        limited_results = deduplicated_results[:user_context.source_limit]

        dedup_rate = 1 - (len(deduplicated_results) / max(len(raw_results), 1))
        self._search_metrics['deduplication_rate'] = dedup_rate

        compressed_results = text_compressor.compress_search_results(
            [r.to_dict() for r in limited_results],
            total_token_budget=3000
        )

        return {
            "results": compressed_results,
            "sources_used": list(self._search_metrics['apis_used']),
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
            return {"average_score": 0, "high_credibility_count": 0}

        scores = [r.credibility_score for r in results]
        high_cred_count = sum(1 for s in scores if s >= 0.7)

        return {
            "average_score": sum(scores) / len(scores),
            "high_credibility_count": high_cred_count,
            "high_credibility_ratio": high_cred_count / len(results),
            "score_distribution": {
                "high": high_cred_count,
                "medium": sum(1 for s in scores if 0.4 <= s < 0.7),
                "low": sum(1 for s in scores if s < 0.4)
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
        normalized = {
            'title': result.get('title') or result.get('url', '').split('/')[-1] or '(untitled)',
            'url': result.get('url', ''),
            'snippet': result.get('snippet', ''),
            'content': result.get('content', ''),
            'source_api': result.get('source_api', result.get('search_api', 'unknown')),
            'credibility_score': result.get('credibility_score'),
            'origin_query': result.get('origin_query', ''),
            'origin_query_id': result.get('origin_query_id', ''),
            'metadata': result.get('metadata', {})
        }
        
        # Remove entries with missing URLs (drop if missing)
        if not normalized['url']:
            return None
            
        return normalized

    async def get_source_credibility_safe(self, domain: str, paradigm: str) -> Tuple[float, str, str]:
        """Get source credibility with non-blocking error handling"""
        try:
            if not self.credibility_enabled:
                return 0.5, "Credibility checking disabled", "skipped"
                
            credibility = await get_source_credibility(domain, paradigm)
            return credibility.overall_score, credibility.explanation if hasattr(credibility, 'explanation') else "", "success"
        except Exception as e:
            logger.warning(f"Credibility check failed for {domain}: {str(e)}")
            return 0.5, f"Credibility check failed: {str(e)}", "failed"

    async def _execute_deep_research_integration(
        self,
        context_engineered: ContextEngineeredQuerySchema,
        classification: ClassificationResultSchema,
        user_context: UserContextSchema,
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
            classification=classification,
            context_engineering=context_engineered,
            config=deep_config,
            progress_tracker=progress_callback,
            research_id=research_id,
        )

        if not deep_result.content:
            return []

        deep_search_results = []
        for citation in deep_result.citations:
            deep_search_results.append({
                "title": citation.title,
                "url": citation.url,
                "snippet": deep_result.content[citation.start_index:citation.end_index][:200],
                "domain": citation.url.split('/')[2] if '/' in citation.url else citation.url,
                "result_type": "deep_research",
                "credibility_score": 0.9,
                "origin_query": context_engineered.original_query,
                "search_api": "deep_research"
            })

        return deep_search_results

    def _convert_legacy_to_v2_result(self, legacy_result: ResearchExecutionResult) -> Dict[str, Any]:
        """Convert legacy ResearchExecutionResult to V2 format with consistent dict handling"""
        from .result_adapter import adapt_results
        
        # Use ResultAdapter to safely handle filtered_results
        try:
            adapter = adapt_results(legacy_result.filtered_results)
            
            if hasattr(adapter, 'get_valid_results'):
                # It's a ResultListAdapter
                valid_results = adapter.get_valid_results()
                results_dicts = []
                
                for result in valid_results:
                    result_dict = {
                        "url": result.url,
                        "title": result.title,
                        "snippet": result.snippet,
                        "source_api": result.source_api,
                        "credibility_score": result.credibility_score or 0.5,
                        "origin_query": legacy_result.original_query,
                        "paradigm_alignment": legacy_result.paradigm
                    }
                    # Normalize the result
                    normalized = self.normalize_result_shape(result_dict)
                    if normalized:  # Only add if valid (has URL)
                        results_dicts.append(normalized)
            else:
                # Single result
                if adapter.has_required_fields():
                    result_dict = {
                        "url": adapter.url,
                        "title": adapter.title,
                        "snippet": adapter.snippet,
                        "source_api": adapter.source_api,
                        "credibility_score": adapter.credibility_score or 0.5,
                        "origin_query": legacy_result.original_query,
                        "paradigm_alignment": legacy_result.paradigm
                    }
                    normalized = self.normalize_result_shape(result_dict)
                    results_dicts = [normalized] if normalized else []
                else:
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
                "queries_executed": len(legacy_result.search_queries_executed),
                "sources_used": list(set(r.get("source_api", "unknown") for r in results_dicts)),
                "credibility_summary": {
                    "average_score": sum(legacy_result.credibility_scores.values()) / max(len(legacy_result.credibility_scores), 1)
                },
                "deduplication_stats": {
                    "original_count": legacy_result.execution_metrics.get("raw_results_count", 0),
                    "final_count": len(results_dicts)
                },
                "search_metrics": legacy_result.execution_metrics,
                "processing_time": legacy_result.execution_metrics.get("processing_time_seconds", 0),
                "paradigm": legacy_result.paradigm
            }
        }

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

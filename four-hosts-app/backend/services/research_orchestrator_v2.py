"""
Enhanced Research Orchestrator V2 with Deterministic Operations
Ensures consistent results regardless of async execution order
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import OrderedDict, defaultdict
import hashlib
import json

from models.context_models import (
    SearchResultSchema, ClassificationResultSchema,
    ContextEngineeredQuerySchema, UserContextSchema,
    HostParadigm
)
from services.search_apis import SearchAPIManager, SearchResult, SearchConfig
from services.credibility import get_source_credibility, CredibilityScore
from services.paradigm_search import get_search_strategy, SearchContext
from services.cache import cache_manager, get_cached_search_results, cache_search_results
from services.text_compression import text_compressor, query_compressor

logger = logging.getLogger(__name__)


class DeterministicMerger:
    """Ensures deterministic merging of async results"""

    @staticmethod
    def create_result_hash(result: Dict[str, Any]) -> str:
        """Create stable hash for a result"""
        # Use URL as primary key, title as secondary
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
        # Track which sources provided each result
        url_to_sources: Dict[str, Set[str]] = defaultdict(set)
        url_to_best_result: Dict[str, Dict[str, Any]] = {}

        # Process batches in order
        for batch_idx, batch in enumerate(result_batches):
            for result in batch:
                url = result.get('url', '')
                if not url:
                    continue

                source = result.get('source_api', f'batch_{batch_idx}')
                url_to_sources[url].add(source)

                # Keep result with most content
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

        # Sort results deterministically
        sorted_urls = sorted(url_to_best_result.keys())
        merged_results = []
        source_mapping = {}

        for url in sorted_urls:
            result = url_to_best_result[url]
            result['source_apis'] = sorted(list(url_to_sources[url]))
            merged_results.append(result)
            source_mapping[url] = result['source_apis']

        return merged_results, source_mapping


class ResearchOrchestratorV2:
    """Enhanced orchestrator with deterministic operations and full context preservation"""

    def __init__(self):
        from services.search_apis import create_search_manager
        self.search_manager = create_search_manager()
        self.merger = DeterministicMerger()

        # Metrics
        self._search_metrics = {
            "total_queries": 0,
            "total_results": 0,
            "apis_used": set(),
            "deduplication_rate": 0.0
        }

    async def execute_research(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: UserContextSchema,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Execute research with deterministic result ordering"""

        # Optimize queries based on paradigm and user limits
        optimized_queries = query_compressor.optimize_query_batch(
            context_engineered.refined_queries,
            max_queries=self._get_query_limit(user_context),
            paradigm=classification.primary_paradigm.value
        )

        # Log query optimization
        logger.info(
            f"Optimized {len(context_engineered.refined_queries)} queries "
            f"to {len(optimized_queries)} for {classification.primary_paradigm.value}"
        )

        # Execute searches with deterministic ordering
        search_results = await self._execute_searches_deterministic(
            optimized_queries,
            classification.primary_paradigm,
            user_context,
            progress_callback
        )

        # Process and enrich results
        processed_results = await self._process_results(
            search_results,
            classification,
            context_engineered,
            user_context
        )

        return {
            "results": processed_results["results"],
            "metadata": {
                "total_results": len(processed_results["results"]),
                "queries_executed": len(optimized_queries),
                "sources_used": processed_results["sources_used"],
                "credibility_summary": processed_results["credibility_summary"],
                "deduplication_stats": processed_results["dedup_stats"],
                "search_metrics": dict(self._search_metrics)
            }
        }

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
        progress_callback: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """Execute searches with guaranteed deterministic ordering"""

        # Create search tasks with index tracking
        search_tasks = []
        task_metadata = []

        for idx, query in enumerate(queries):
            # Select APIs based on paradigm
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

        # Execute all tasks concurrently
        if progress_callback:
            await progress_callback(f"Executing {len(search_tasks)} search operations")

        # Use asyncio.gather with return_exceptions to handle failures gracefully
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results with metadata
        processed_results = []

        for idx, (result, metadata) in enumerate(zip(results, task_metadata)):
            if isinstance(result, Exception):
                logger.error(f"Search failed for {metadata['api']}: {result}")
                continue

            if result and isinstance(result, list):
                # Convert SearchResult objects to dicts and add metadata
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

                    processed_results.append(item_dict)

        # Sort results deterministically
        processed_results.sort(key=lambda x: (
            x.get('origin_query_id', ''),
            x.get('search_api', ''),
            x.get('url', ''),
            x.get('title', '')
        ))

        # Update metrics
        self._search_metrics['total_queries'] += len(queries)
        self._search_metrics['total_results'] += len(processed_results)

        return processed_results

    def _select_apis_for_paradigm(
        self,
        paradigm: HostParadigm,
        user_context: UserContextSchema
    ) -> List[str]:
        """Select appropriate APIs based on paradigm and user context"""

        # Base API selection by paradigm
        paradigm_apis = {
            HostParadigm.DOLORES: ["brave", "google"],  # Revolutionary: diverse sources
            HostParadigm.BERNARD: ["google", "arxiv", "pubmed"],  # Analytical: academic
            HostParadigm.MAEVE: ["google", "brave"],  # Strategic: business-focused
            HostParadigm.TEDDY: ["google"]  # Devotion: trusted sources
        }

        selected = paradigm_apis.get(paradigm, ["google"])

        # Adjust based on user tier
        if not user_context.is_pro_user and len(selected) > 2:
            selected = selected[:2]  # Limit APIs for free users

        # Add user's preferred sources if specified
        if user_context.preferences.get("preferred_sources"):
            selected.extend(user_context.preferences["preferred_sources"])

        # Remove duplicates while preserving order
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
            # Use search manager's search_with_fallback method
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

        # Group by URL for deduplication
        url_groups = defaultdict(list)
        for result in raw_results:
            url = result.get('url', '')
            if url:
                url_groups[url].append(result)

        # Deduplicate and enrich
        deduplicated_results = []

        for url, duplicates in url_groups.items():
            # Merge duplicates
            merged = self._merge_duplicate_results(duplicates)

            # Calculate credibility
            from urllib.parse import urlparse
            domain = urlparse(merged.get('url', '')).netloc
            credibility_score = await get_source_credibility(
                domain,
                classification.primary_paradigm.value
            )
            merged['credibility_score'] = credibility_score.overall_score
            merged['credibility_explanation'] = credibility_score.explanation

            # Add paradigm alignment
            merged['paradigm_alignment'] = classification.primary_paradigm.value

            # Convert to schema
            result_schema = SearchResultSchema(
                url=merged['url'],
                title=merged.get('title', ''),
                snippet=merged.get('snippet', ''),
                source_api=merged.get('search_api', 'unknown'),
                credibility_score=credibility_score.score,
                credibility_explanation=credibility_score.explanation,
                origin_query=merged.get('origin_query'),
                origin_query_id=merged.get('origin_query_id'),
                paradigm_alignment=classification.primary_paradigm,
                metadata=merged.get('metadata', {})
            )

            deduplicated_results.append(result_schema)

        # Sort by credibility and relevance
        deduplicated_results.sort(
            key=lambda x: (x.credibility_score, x.url),
            reverse=True
        )

        # Apply user's source limit
        limited_results = deduplicated_results[:user_context.source_limit]

        # Calculate statistics
        dedup_rate = 1 - (len(deduplicated_results) / max(len(raw_results), 1))
        self._search_metrics['deduplication_rate'] = dedup_rate

        # Compress results for efficient processing
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
        # Start with the first result
        merged = duplicates[0].copy()

        # Track all sources and queries
        all_sources = set()
        all_queries = set()

        for dup in duplicates:
            all_sources.add(dup.get('search_api', 'unknown'))
            all_queries.add(dup.get('origin_query', ''))

            # Use longest snippet/content
            if len(dup.get('snippet', '')) > len(merged.get('snippet', '')):
                merged['snippet'] = dup['snippet']
            if len(dup.get('content', '')) > len(merged.get('content', '')):
                merged['content'] = dup['content']

        # Add merged metadata
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


# Global V2 orchestrator instance
research_orchestrator_v2 = ResearchOrchestratorV2()


# Create singleton instance
research_orchestrator_v2 = ResearchOrchestratorV2()

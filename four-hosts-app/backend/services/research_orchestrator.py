"""
Research Orchestration System for Four Hosts Research Application
Integrates Context Engineering Pipeline with Real Search Execution
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from .search_apis import SearchResult, SearchConfig, create_search_manager
from .paradigm_search import get_search_strategy, SearchContext
from .credibility import get_source_credibility
from .cache import cache_manager, get_cached_search_results, cache_search_results

logging.basicConfig(level=logging.INFO)
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


@dataclass
class DeduplicationResult:
    """Result of deduplication process"""

    unique_results: List[SearchResult]
    duplicates_removed: int
    similarity_threshold: float
    clusters: List[List[SearchResult]] = field(default_factory=list)


class ResultDeduplicator:
    """Removes duplicate search results using various similarity measures"""

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    async def deduplicate_results(
        self, results: List[SearchResult]
    ) -> DeduplicationResult:
        """Remove duplicate results using URL and content similarity"""
        if not results:
            return DeduplicationResult([], 0, self.similarity_threshold)

        unique_results = []
        duplicates_removed = 0
        seen_urls = set()
        clusters = []

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

                    # Add to existing cluster or create new one
                    cluster_found = False
                    for cluster in clusters:
                        if existing in cluster:
                            cluster.append(result)
                            cluster_found = True
                            break

                    if not cluster_found:
                        clusters.append([existing, result])
                    break

            if not is_duplicate:
                unique_results.append(result)

        logger.info(
            f"Deduplication: {len(results)} -> {len(unique_results)} "
            f"({duplicates_removed} duplicates removed)"
        )

        return DeduplicationResult(
            unique_results=unique_results,
            duplicates_removed=duplicates_removed,
            similarity_threshold=self.similarity_threshold,
            clusters=clusters,
        )

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

        # Snippet similarity (simplified)
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


class CostMonitor:
    """Monitors and tracks API costs"""

    def __init__(self):
        self.cost_per_call = {
            "google": 0.005,  # $5 per 1000 queries
            "brave": 0.0,  # Free tier (up to 2000/month)
            "moz": 0.01,  # Domain authority calls
            "arxiv": 0.0,  # Free
            "pubmed": 0.0,  # Free
        }

    async def track_search_cost(self, api_name: str, queries_count: int) -> float:
        """Track cost for search API calls"""
        cost = self.cost_per_call.get(api_name, 0.0) * queries_count

        # Update cache with cost tracking
        await cache_manager.track_api_cost(api_name, cost, queries_count)

        return cost

    async def get_daily_costs(
        self, date: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get daily API costs"""
        return await cache_manager.get_daily_api_costs(date)

    async def check_budget_alerts(self, daily_budget: float = 50.0) -> List[str]:
        """Check for budget alerts"""
        costs = await self.get_daily_costs()
        alerts = []

        total_cost = sum(api_data.get("cost", 0.0) for api_data in costs.values())

        if daily_budget <= 0:
            return ["Invalid daily budget configuration"]

        if total_cost > daily_budget * 0.8:
            alerts.append(
                f"Daily budget at {total_cost/daily_budget*100:.1f}% (${total_cost:.2f})"
            )

        if total_cost > daily_budget:
            alerts.append(
                f"Daily budget exceeded: ${total_cost:.2f} > ${daily_budget:.2f}"
            )

        return alerts


class ParadigmAwareSearchOrchestrator:
    """Main orchestrator that integrates everything together"""

    def __init__(self):
        self.search_manager = None
        self.deduplicator = ResultDeduplicator()
        self.cost_monitor = CostMonitor()

        # Performance tracking
        self.execution_history = []

    async def initialize(self):
        """Initialize the orchestrator"""
        self.search_manager = create_search_manager()
        await self.search_manager.initialize()
        await cache_manager.initialize()
        logger.info("Research orchestrator initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.search_manager:
            await self.search_manager.cleanup()
            logger.info("Search manager cleaned up")

    async def execute_paradigm_research(
        self, context_engineered_query, max_results: int = 100, progress_tracker=None, research_id: str = None
    ) -> ResearchExecutionResult:
        """
        Execute research using Context Engineering Pipeline output

        Args:
            context_engineered_query: Output from Context Engineering Pipeline
            max_results: Maximum results to return
        """
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
            "bernard",  # Default to bernard if not found
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
        search_queries = select_output.search_queries[:8]  # Limit to 8 queries

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
                
                # Report search completion from cache
                if progress_tracker and research_id:
                    await progress_tracker.report_search_completed(
                        research_id, query, len(cached_results)
                    )
            else:
                # Report search starting
                if progress_tracker and research_id:
                    await progress_tracker.report_search_started(
                        research_id, query, "mixed", idx + 1, len(search_queries)
                    )
                    # Update overall progress (30-50% range for searches)
                    search_progress = 30 + int((idx / len(search_queries)) * 20)
                    await progress_tracker.update_progress(
                        research_id, 
                        f"Searching: {query[:40]}...", 
                        search_progress
                    )
                
                # Execute real search
                config = SearchConfig(
                    max_results=min(max_results, 50), language="en", region="us"
                )

                try:
                    # Use search manager with fallback
                    api_results = await self.search_manager.search_with_fallback(
                        query, config
                    )

                    # Track costs
                    primary_api = "google"  # Assuming Google is primary
                    cost = await self.cost_monitor.track_search_cost(primary_api, 1)
                    cost_breakdown[f"{query_type}_{query[:20]}"] = cost

                    # Weight results based on query importance
                    for result in api_results:
                        result.credibility_score = (
                            result.credibility_score * weight
                            if hasattr(result, "credibility_score")
                            else weight
                        )

                    # Cache results
                    await cache_search_results(
                        query, config_dict, paradigm, api_results
                    )
                    all_results[f"{query_type}_{query[:30]}"] = api_results

                    logger.info(f"Got {len(api_results)} results for: {query[:50]}...")
                    
                    # Report search completion
                    if progress_tracker and research_id:
                        await progress_tracker.report_search_completed(
                            research_id, query, len(api_results)
                        )
                        
                        # Report top sources found
                        for result in api_results[:3]:  # Report top 3 sources
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

        # Combine all results
        combined_results = []
        for query_results in all_results.values():
            combined_results.extend(query_results)

        logger.info(f"Combined {len(combined_results)} total results")

        # Update progress for deduplication
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id, "Removing duplicate results...", 52
            )
        
        # Deduplicate results
        dedup_result = await self.deduplicator.deduplicate_results(combined_results)
        deduplicated_results = dedup_result.unique_results
        
        # Report deduplication stats
        if progress_tracker and research_id:
            await progress_tracker.report_deduplication(
                research_id, len(combined_results), len(deduplicated_results)
            )

        # Apply paradigm-specific filtering and ranking
        filtered_results = await strategy.filter_and_rank_results(
            deduplicated_results, search_context
        )

        # Update progress for credibility analysis
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id, "Evaluating source credibility...", 55
            )
        
        # Calculate credibility scores
        credibility_scores = {}
        top_results = filtered_results[:20]  # Top 20 results
        for idx, result in enumerate(top_results):
            try:
                credibility = await get_source_credibility(result.domain, paradigm)
                credibility_scores[result.domain] = credibility.overall_score
                result.credibility_score = credibility.overall_score
                
                # Report credibility check
                if progress_tracker and research_id and idx % 5 == 0:  # Report every 5th check
                    await progress_tracker.report_credibility_check(
                        research_id, result.domain, credibility.overall_score
                    )
            except Exception as e:
                logger.warning(
                    f"Credibility check failed for {result.domain}: {str(e)}"
                )
                credibility_scores[result.domain] = 0.5

        # Limit final results
        final_results = filtered_results[:max_results]

        # Execute secondary search if applicable
        secondary_results = []
        if secondary_paradigm:
            logger.info(
                f"Executing secondary research for paradigm: {secondary_paradigm}"
            )
            secondary_strategy = get_search_strategy(secondary_paradigm)
            # Using a simplified query for secondary search for now
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
                "duplicates_removed": dedup_result.duplicates_removed,
                "credibility_checks": len(credibility_scores),
            }
        )

        # Check budget alerts
        budget_alerts = await self.cost_monitor.check_budget_alerts()
        if budget_alerts:
            logger.warning(f"Budget alerts: {budget_alerts}")
            metrics["budget_alerts"] = budget_alerts

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

        return result

    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"message": "No executions yet"}

        total_executions = len(self.execution_history)
        if total_executions == 0:
            return {"message": "No executions yet"}
            
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
        }


# Global orchestrator instance
research_orchestrator = ParadigmAwareSearchOrchestrator()


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


# Example usage and testing
async def test_research_orchestration():
    """Test the complete research orchestration system"""
    print("Testing Research Orchestration System...")
    print("=" * 60)

    # This would normally come from the Context Engineering Pipeline
    # For testing, we'll create a mock context engineered query
    from types import SimpleNamespace

    # Mock classification result
    mock_classification = SimpleNamespace()
    mock_classification.primary_paradigm = SimpleNamespace()
    mock_classification.primary_paradigm.value = "maeve"
    mock_classification.secondary_paradigm = None

    # Mock select output (from Context Engineering Pipeline)
    mock_select_output = SimpleNamespace()
    mock_select_output.search_queries = [
        {
            "query": "small business compete Amazon strategy",
            "type": "strategic",
            "weight": 1.0,
        },
        {
            "query": "Amazon competitive disadvantages",
            "type": "strategic",
            "weight": 0.8,
        },
        {
            "query": "local business advantages over Amazon",
            "type": "strategic",
            "weight": 0.9,
        },
    ]

    # Mock context engineered query
    mock_context_query = SimpleNamespace()
    mock_context_query.original_query = "How can small businesses compete with Amazon?"
    mock_context_query.classification = mock_classification
    mock_context_query.select_output = mock_select_output

    # Initialize system
    await initialize_research_system()

    # Execute research
    try:
        result = await execute_research(mock_context_query, max_results=20)

        print(f"✓ Research completed successfully")
        print(f"  Original query: {result.original_query}")
        print(f"  Paradigm: {result.paradigm}")
        print(
            f"  Processing time: {result.execution_metrics['processing_time_seconds']:.2f}s"
        )
        print(f"  Final results: {len(result.filtered_results)}")
        print(f"  Cost: ${sum(result.cost_breakdown.values()):.4f}")

        # Show top 3 results
        print(f"\nTop 3 Results:")
        for i, res in enumerate(result.filtered_results[:3], 1):
            print(f"  {i}. {res.title[:60]}...")
            print(
                f"     Domain: {res.domain} (credibility: {res.credibility_score:.2f})"
            )
            print(f"     URL: {res.url}")

        # Show stats
        stats = await research_orchestrator.get_execution_stats()
        print(f"\nSystem Stats:")
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Avg processing time: {stats['average_processing_time']}s")
        print(f"  Total cost: ${stats['total_cost']:.4f}")

    except Exception as e:
        print(f"✗ Research execution failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_research_orchestration())

"""
Paradigm-Specific Search Strategies for Four Hosts Research Application
Implements specialized search approaches for each paradigm (Dolores, Teddy, Bernard, Maeve)
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote_plus

from .search_apis import (
    SearchResult,
    SearchConfig,
    SearchAPIManager,
    create_search_manager,
)
from .credibility import get_source_credibility, CredibilityScore
from .cache import cache_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParadigmSearchStrategy:
    """Defines search strategy for a specific paradigm"""

    paradigm: str
    query_modifiers: List[str]
    source_preferences: List[str]
    domain_weights: Dict[str, float]
    search_operators: List[str]
    result_filters: Dict[str, Any]
    max_results: int = 100


@dataclass
class SearchContext:
    """Context for paradigm-aware search"""

    original_query: str
    paradigm: str
    secondary_paradigm: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    depth: str = "standard"  # quick, standard, deep
    region: str = "us"
    language: str = "en"


class DoloresSearchStrategy:
    """Revolutionary paradigm - Focuses on investigative journalism and activism"""

    def __init__(self):
        self.paradigm = "dolores"

        # Query modifiers that expose systemic issues
        self.query_modifiers = [
            "controversy",
            "scandal",
            "expose",
            "corrupt",
            "injustice",
            "systemic",
            "inequality",
            "power abuse",
            "cover-up",
            "investigation",
            "whistleblower",
            "leak",
        ]

        # Preferred source domains
        self.preferred_sources = [
            "propublica.org",
            "theintercept.com",
            "democracynow.org",
            "jacobinmag.com",
            "commondreams.org",
            "truthout.org",
            "motherjones.com",
            "thenation.com",
            "theguardian.com",
            "washingtonpost.com",
            "nytimes.com",
            "icij.org",
        ]

        # Search operators for finding hidden information
        self.search_operators = [
            '"leaked documents"',
            '"internal memo"',
            '"confidential"',
            '"investigation reveals"',
            '"according to sources"',
            '"documents show"',
            '"expose"',
            '"scandal"',
        ]

    async def generate_search_queries(
        self, context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Generate Dolores-specific search queries"""
        base_query = context.original_query
        queries = []

        # Original query with investigative angle
        queries.append(
            {
                "query": base_query,
                "type": "original",
                "weight": 1.0,
                "paradigm": self.paradigm,
                "source_filter": None,
            }
        )

        # Add controversy/scandal angle
        for modifier in self.query_modifiers[:3]:
            modified_query = f"{base_query} {modifier}"
            queries.append(
                {
                    "query": modified_query,
                    "type": "revolutionary_angle",
                    "weight": 0.8,
                    "paradigm": self.paradigm,
                    "source_filter": "investigative",
                }
            )

        # Specific investigative queries
        investigative_patterns = [
            f'"{base_query}" corruption',
            f'"{base_query}" scandal investigation',
            f'"{base_query}" whistleblower',
        ]

        for pattern in investigative_patterns:
            queries.append(
                {
                    "query": pattern,
                    "type": "investigative_pattern",
                    "weight": 0.7,
                    "paradigm": self.paradigm,
                    "source_filter": "alternative_media",
                }
            )

        return queries[:8]  # Limit to 8 queries

    async def filter_and_rank_results(
        self, results: List[SearchResult], context: SearchContext
    ) -> List[SearchResult]:
        """Filter and rank results based on Dolores paradigm priorities"""
        scored_results = []

        for result in results:
            score = await self._calculate_dolores_score(result, context)
            if score > 0.3:  # Minimum threshold
                result.credibility_score = score
                scored_results.append((result, score))

        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [result for result, score in scored_results]

    async def _calculate_dolores_score(
        self, result: SearchResult, context: SearchContext
    ) -> float:
        """Calculate relevance score for Dolores paradigm"""
        score = 0.5  # Base score

        # Bonus for preferred sources
        if result.domain in self.preferred_sources:
            score += 0.3

        # Bonus for investigative keywords in title/snippet
        investigative_keywords = [
            "expose",
            "reveal",
            "investigation",
            "scandal",
            "corruption",
            "injustice",
            "systemic",
            "power",
        ]

        text = (result.title + " " + result.snippet).lower()
        keyword_matches = sum(
            1 for keyword in investigative_keywords if keyword in text
        )
        score += min(0.2, keyword_matches * 0.05)

        # Get credibility score with Dolores paradigm context
        try:
            credibility = await get_source_credibility(result.domain, "dolores")
            dolores_alignment = credibility.paradigm_alignment.get("dolores", 0.5)
            score += dolores_alignment * 0.3
        except:
            pass  # Continue without credibility bonus

        return min(1.0, score)


class TeddySearchStrategy:
    """Devotion paradigm - Focuses on community support and care resources"""

    def __init__(self):
        self.paradigm = "teddy"

        self.query_modifiers = [
            "support",
            "help",
            "resources",
            "community",
            "assistance",
            "care",
            "protection",
            "safety",
            "wellbeing",
            "services",
            "nonprofit",
            "charity",
            "aid",
            "relief",
        ]

        self.preferred_sources = [
            "npr.org",
            "pbs.org",
            "unitedway.org",
            "redcross.org",
            "who.int",
            "unicef.org",
            "doctorswithoutborders.org",
            "goodwill.org",
            "salvationarmy.org",
            "feedingamerica.org",
            "habitat.org",
            "americanredcross.org",
        ]

        self.search_operators = [
            '"support services"',
            '"community resources"',
            '"help available"',
            '"assistance programs"',
            '"care options"',
            '"nonprofit"',
            '"charity"',
            '"volunteer opportunities"',
        ]

    async def generate_search_queries(
        self, context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Generate Teddy-specific search queries"""
        base_query = context.original_query
        queries = []

        # Original query
        queries.append(
            {
                "query": base_query,
                "type": "original",
                "weight": 1.0,
                "paradigm": self.paradigm,
                "source_filter": None,
            }
        )

        # Support-focused queries
        for modifier in self.query_modifiers[:3]:
            queries.append(
                {
                    "query": f"{base_query} {modifier}",
                    "type": "support_focused",
                    "weight": 0.9,
                    "paradigm": self.paradigm,
                    "source_filter": "community",
                }
            )

        # Resource and service queries
        resource_patterns = [
            f'"{base_query}" resources available',
            f"help with {base_query}",
            f"{base_query} support services",
            f"{base_query} community programs",
        ]

        for pattern in resource_patterns:
            queries.append(
                {
                    "query": pattern,
                    "type": "resource_pattern",
                    "weight": 0.8,
                    "paradigm": self.paradigm,
                    "source_filter": "nonprofit",
                }
            )

        return queries[:8]

    async def filter_and_rank_results(
        self, results: List[SearchResult], context: SearchContext
    ) -> List[SearchResult]:
        """Filter and rank results for Teddy paradigm"""
        scored_results = []

        for result in results:
            score = await self._calculate_teddy_score(result, context)
            if score > 0.3:
                result.credibility_score = score
                scored_results.append((result, score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [result for result, score in scored_results]

    async def _calculate_teddy_score(
        self, result: SearchResult, context: SearchContext
    ) -> float:
        """Calculate relevance score for Teddy paradigm"""
        score = 0.5

        # Bonus for community/nonprofit sources
        if result.domain in self.preferred_sources:
            score += 0.3
        elif result.domain.endswith(".org"):
            score += 0.2

        # Bonus for care-related keywords
        care_keywords = [
            "help",
            "support",
            "care",
            "assist",
            "community",
            "service",
            "nonprofit",
            "charity",
            "volunteer",
            "aid",
            "relief",
        ]

        text = (result.title + " " + result.snippet).lower()
        keyword_matches = sum(1 for keyword in care_keywords if keyword in text)
        score += min(0.25, keyword_matches * 0.05)

        # Credibility with Teddy alignment
        try:
            credibility = await get_source_credibility(result.domain, "teddy")
            teddy_alignment = credibility.paradigm_alignment.get("teddy", 0.5)
            score += teddy_alignment * 0.3
        except:
            pass

        return min(1.0, score)


class BernardSearchStrategy:
    """Analytical paradigm - Focuses on academic research and data"""

    def __init__(self):
        self.paradigm = "bernard"

        self.query_modifiers = [
            "research",
            "study",
            "analysis",
            "data",
            "statistics",
            "evidence",
            "findings",
            "peer reviewed",
            "academic",
            "methodology",
            "empirical",
            "systematic review",
        ]

        self.preferred_sources = [
            "nature.com",
            "science.org",
            "arxiv.org",
            "pubmed.ncbi.nlm.nih.gov",
            "scholar.google.com",
            "jstor.org",
            "researchgate.net",
            "springerlink.com",
            "sciencedirect.com",
            "wiley.com",
            "tandfonline.com",
            "cambridge.org",
        ]

        self.search_operators = [
            '"peer reviewed"',
            '"systematic review"',
            '"meta-analysis"',
            '"research study"',
            '"empirical evidence"',
            '"data analysis"',
            '"methodology"',
            '"statistical significance"',
        ]

    async def generate_search_queries(
        self, context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Generate Bernard-specific search queries"""
        base_query = context.original_query
        queries = []

        # Original query
        queries.append(
            {
                "query": base_query,
                "type": "original",
                "weight": 1.0,
                "paradigm": self.paradigm,
                "source_filter": None,
            }
        )

        # Academic-focused queries
        for modifier in self.query_modifiers[:3]:
            queries.append(
                {
                    "query": f"{base_query} {modifier}",
                    "type": "academic_focused",
                    "weight": 0.95,
                    "paradigm": self.paradigm,
                    "source_filter": "academic",
                }
            )

        # Research-specific patterns
        research_patterns = [
            f'"{base_query}" peer reviewed research',
            f"{base_query} systematic review",
            f"{base_query} meta-analysis",
            f'"{base_query}" empirical study',
        ]

        for pattern in research_patterns:
            queries.append(
                {
                    "query": pattern,
                    "type": "research_pattern",
                    "weight": 0.9,
                    "paradigm": self.paradigm,
                    "source_filter": "journal",
                }
            )

        return queries[:10]  # More queries for comprehensive research

    async def filter_and_rank_results(
        self, results: List[SearchResult], context: SearchContext
    ) -> List[SearchResult]:
        """Filter and rank results for Bernard paradigm"""
        scored_results = []

        for result in results:
            score = await self._calculate_bernard_score(result, context)
            if score > 0.4:  # Higher threshold for academic quality
                result.credibility_score = score
                scored_results.append((result, score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [result for result, score in scored_results]

    async def _calculate_bernard_score(
        self, result: SearchResult, context: SearchContext
    ) -> float:
        """Calculate relevance score for Bernard paradigm"""
        score = 0.4  # Lower base score, higher standards

        # Strong bonus for academic sources
        if result.domain in self.preferred_sources:
            score += 0.4
        elif result.domain.endswith(".edu") or "journal" in result.domain:
            score += 0.3
        elif result.result_type == "academic":
            score += 0.35

        # Bonus for research-related keywords
        research_keywords = [
            "study",
            "research",
            "analysis",
            "data",
            "peer reviewed",
            "methodology",
            "findings",
            "evidence",
            "statistical",
        ]

        text = (result.title + " " + result.snippet).lower()
        keyword_matches = sum(1 for keyword in research_keywords if keyword in text)
        score += min(0.3, keyword_matches * 0.05)

        # High credibility requirements
        try:
            credibility = await get_source_credibility(result.domain, "bernard")
            if credibility.overall_score < 0.7:
                score *= 0.7  # Penalty for low credibility
            else:
                bernard_alignment = credibility.paradigm_alignment.get("bernard", 0.5)
                score += bernard_alignment * 0.4
        except:
            score *= 0.8  # Slight penalty if credibility can't be verified

        return min(1.0, score)


class MaeveSearchStrategy:
    """Strategic paradigm - Focuses on business intelligence and strategy"""

    def __init__(self):
        self.paradigm = "maeve"

        self.query_modifiers = [
            "strategy",
            "competitive",
            "market",
            "business",
            "industry",
            "analysis",
            "trends",
            "insights",
            "opportunities",
            "framework",
            "implementation",
            "optimization",
            "roi",
            "performance",
        ]

        self.preferred_sources = [
            "wsj.com",
            "ft.com",
            "bloomberg.com",
            "forbes.com",
            "hbr.org",
            "mckinsey.com",
            "bcg.com",
            "strategy-business.com",
            "bain.com",
            "deloitte.com",
            "pwc.com",
            "kpmg.com",
            "gartner.com",
            "forrester.com",
        ]

        self.search_operators = [
            '"business strategy"',
            '"competitive analysis"',
            '"market trends"',
            '"industry insights"',
            '"best practices"',
            '"case study"',
            '"implementation guide"',
            '"strategic framework"',
        ]

    async def generate_search_queries(
        self, context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Generate Maeve-specific search queries"""
        base_query = context.original_query
        queries = []

        # Original query
        queries.append(
            {
                "query": base_query,
                "type": "original",
                "weight": 1.0,
                "paradigm": self.paradigm,
                "source_filter": None,
            }
        )

        # Strategy-focused queries
        for modifier in self.query_modifiers[:3]:
            queries.append(
                {
                    "query": f"{base_query} {modifier}",
                    "type": "strategy_focused",
                    "weight": 0.9,
                    "paradigm": self.paradigm,
                    "source_filter": "business",
                }
            )

        # Business-specific patterns
        business_patterns = [
            f'"{base_query}" competitive strategy',
            f"{base_query} market analysis",
            f"{base_query} business case study",
            f"how to {base_query} effectively",
        ]

        for pattern in business_patterns:
            queries.append(
                {
                    "query": pattern,
                    "type": "business_pattern",
                    "weight": 0.85,
                    "paradigm": self.paradigm,
                    "source_filter": "industry",
                }
            )

        return queries[:8]

    async def filter_and_rank_results(
        self, results: List[SearchResult], context: SearchContext
    ) -> List[SearchResult]:
        """Filter and rank results for Maeve paradigm"""
        scored_results = []

        for result in results:
            score = await self._calculate_maeve_score(result, context)
            if score > 0.3:
                result.credibility_score = score
                scored_results.append((result, score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [result for result, score in scored_results]

    async def _calculate_maeve_score(
        self, result: SearchResult, context: SearchContext
    ) -> float:
        """Calculate relevance score for Maeve paradigm"""
        score = 0.5

        # Bonus for business sources
        if result.domain in self.preferred_sources:
            score += 0.3
        elif any(
            term in result.domain
            for term in ["business", "market", "industry", "consulting"]
        ):
            score += 0.2

        # Bonus for strategic keywords
        strategy_keywords = [
            "strategy",
            "competitive",
            "market",
            "business",
            "industry",
            "framework",
            "implementation",
            "optimize",
            "roi",
            "performance",
        ]

        text = (result.title + " " + result.snippet).lower()
        keyword_matches = sum(1 for keyword in strategy_keywords if keyword in text)
        score += min(0.25, keyword_matches * 0.05)

        # Actionability bonus - look for implementation words
        action_keywords = [
            "how to",
            "implement",
            "step",
            "guide",
            "framework",
            "process",
        ]
        action_matches = sum(1 for keyword in action_keywords if keyword in text)
        score += min(0.15, action_matches * 0.05)

        # Credibility with Maeve alignment
        try:
            credibility = await get_source_credibility(result.domain, "maeve")
            maeve_alignment = credibility.paradigm_alignment.get("maeve", 0.5)
            score += maeve_alignment * 0.3
        except:
            pass

        return min(1.0, score)


# Factory function to get strategy by paradigm
def get_search_strategy(paradigm: str):
    """Get the appropriate search strategy for a paradigm"""
    strategies = {
        "dolores": DoloresSearchStrategy(),
        "teddy": TeddySearchStrategy(),
        "bernard": BernardSearchStrategy(),
        "maeve": MaeveSearchStrategy(),
    }

    return strategies.get(paradigm, BernardSearchStrategy())  # Default to Bernard


# Example usage and testing
async def test_paradigm_strategies():
    """Test all paradigm search strategies"""
    print("Testing Paradigm Search Strategies...")
    print("=" * 60)

    test_query = "climate change solutions"

    for paradigm in ["dolores", "teddy", "bernard", "maeve"]:
        print(f"\n{paradigm.upper()} PARADIGM:")
        print("-" * 40)

        strategy = get_search_strategy(paradigm)
        context = SearchContext(
            original_query=test_query, paradigm=paradigm, depth="standard"
        )

        queries = await strategy.generate_search_queries(context)

        print(f"Generated {len(queries)} search queries:")
        for i, query in enumerate(queries[:3], 1):
            print(f"  {i}. {query['query'][:60]}...")
            print(f"     Type: {query['type']}, Weight: {query['weight']}")

        print()


if __name__ == "__main__":
    asyncio.run(test_paradigm_strategies())

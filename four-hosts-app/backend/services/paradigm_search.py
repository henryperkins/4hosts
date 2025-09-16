"""
Paradigm-Specific Search Strategies for Four Hosts Research Application
Implements specialized search approaches for each paradigm (Dolores, Teddy, Bernard, Maeve)
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote_plus
import re

from .search_apis import (
    SearchResult,
    SearchConfig,
    SearchAPIManager,
    create_search_manager,
)
from .credibility import get_source_credibility, CredibilityScore
from models.paradigms import normalize_to_internal_code
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


class BaseSearchStrategy:
    """Base class with adaptive query generation"""

    def get_adaptive_query_limit(self, query: str) -> int:
        """Determine optimal number of queries based on complexity"""
        import os

        # Allow override via environment variable
        override = os.getenv("ADAPTIVE_QUERY_LIMIT")
        if override:
            try:
                return int(override)
            except ValueError:
                pass

        word_count = len(query.split())

        # Simple queries (1-3 words) need fewer variations
        if word_count <= 3:
            return 2
        # Moderate complexity (4-7 words)
        elif word_count <= 7:
            return 4
        # Complex queries (8-15 words)
        elif word_count <= 15:
            return 6
        # Very complex queries (>15 words) - max variations
        else:
            return 8

    def assess_query_specificity(self, query: str) -> str:
        """Assess if query is specific or broad"""
        # Check for specific indicators
        specific_indicators = [
            '"',  # Exact phrases
            'site:',  # Site-specific
            'filetype:',  # File type specific
            'inurl:',  # URL specific
            'intitle:',  # Title specific
        ]

        has_specific = any(ind in query.lower() for ind in specific_indicators)
        word_count = len(query.split())

        # Check for proper nouns (capitalized words)
        words = query.split()
        proper_nouns = sum(1 for w in words if w and w[0].isupper())

        if has_specific or proper_nouns >= 2:
            return "specific"
        elif word_count <= 3:
            return "broad"
        else:
            return "moderate"


class DoloresSearchStrategy(BaseSearchStrategy):
    """Revolutionary paradigm - Focuses on investigative journalism and activism"""

    def __init__(self):
        super().__init__()
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
            "antitrust",
            "monopoly",
            "price fixing",
            "cartel",
            "regulatory capture",
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
            'FTC OR DOJ antitrust',
            'state attorney general lawsuit',
            'EU competition commission case',
        ]

    async def generate_search_queries(
        self, context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Generate Dolores-specific search queries"""
        base_query = context.original_query
        queries = []
        
        # Clean up base query - remove common fluff words
        cleaned_query = self._clean_query(base_query)

        # Original query with investigative angle
        queries.append(
            {
                "query": cleaned_query,
                "type": "original",
                "weight": 1.0,
                "paradigm": self.paradigm,
                "source_filter": None,
            }
        )

        # Add targeted controversy/scandal angles based on query content
        relevant_modifiers = self._select_relevant_modifiers(cleaned_query, self.query_modifiers)
        for modifier in relevant_modifiers[:3]:
            # Use more sophisticated query construction
            if len(cleaned_query.split()) > 5:
                # For long queries, insert modifier strategically
                modified_query = self._insert_modifier_strategically(cleaned_query, modifier)
            else:
                modified_query = f"{cleaned_query} {modifier}"
            queries.append(
                {
                    "query": modified_query,
                    "type": "revolutionary_angle",
                    "weight": 0.8,
                    "paradigm": self.paradigm,
                    "source_filter": "investigative",
                }
            )

        # Specific investigative queries with better targeting
        investigative_patterns = self._generate_investigative_patterns(cleaned_query)

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

        # Use adaptive limit based on query complexity
        limit = self.get_adaptive_query_limit(base_query)
        return queries[:limit]
    
    def _clean_query(self, query: str) -> str:
        """Remove common fluff words to improve search relevance"""
        fluff_words = {
            'please', 'help', 'me', 'find', 'information', 'about',
            'can', 'you', 'tell', 'what', 'is', 'are', 'the'
        }
        words = query.split()
        cleaned = [w for w in words if w.lower() not in fluff_words]
        # Keep original if cleaning removes too much
        if len(cleaned) < len(words) / 2:
            return query
        return ' '.join(cleaned)
    
    def _select_relevant_modifiers(self, query: str, modifiers: List[str]) -> List[str]:
        """Select modifiers most relevant to the query content"""
        query_lower = query.lower()
        
        # Score modifiers by relevance
        scored_modifiers = []
        for modifier in modifiers:
            score = 0
            # Check for semantic relevance
            if 'company' in query_lower or 'corporation' in query_lower:
                if modifier in ['corrupt', 'scandal', 'expose']:
                    score += 2
            if 'government' in query_lower or 'policy' in query_lower:
                if modifier in ['cover-up', 'leak', 'whistleblower']:
                    score += 2
            if 'system' in query_lower or 'institution' in query_lower:
                if modifier in ['systemic', 'injustice', 'inequality']:
                    score += 2
            
            scored_modifiers.append((modifier, score))
        
        # Sort by score and return
        scored_modifiers.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in scored_modifiers]
    
    def _insert_modifier_strategically(self, query: str, modifier: str) -> str:
        """Insert modifier at strategic position in query"""
        words = query.split()
        # Find noun phrases or key terms
        for i, word in enumerate(words):
            if word.lower() in ['company', 'corporation', 'government', 'system', 'industry']:
                # Insert modifier before the key noun
                words.insert(i, modifier)
                return ' '.join(words)
        # Default: append at end
        return f"{query} {modifier}"
    
    def _generate_investigative_patterns(self, query: str) -> List[str]:
        """Generate investigative search patterns based on query content"""
        patterns = []
        
        # Analyze query for entity types
        if any(term in query.lower() for term in ['company', 'corporation', 'inc', 'llc']):
            patterns.extend([
                f'"{query}" "internal documents"',
                f'"{query}" lawsuit allegations',
                f'{query} "regulatory violations"'
            ])
        elif any(term in query.lower() for term in ['government', 'agency', 'department']):
            patterns.extend([
                f'"{query}" "freedom of information"',
                f'"{query}" oversight investigation',
                f'{query} accountability report'
            ])
        else:
            # Generic investigative patterns
            patterns.extend([
                f'"{query}" investigation reveals',
                f'"{query}" "hidden truth"',
                f'{query} exposed documents'
            ])
        # Always add regulatory/antitrust angle
        patterns.append(f'{query} antitrust FTC OR DOJ')
        patterns.append(f'{query} EU antitrust investigation')
        
        return patterns[:5]

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


class TeddySearchStrategy(BaseSearchStrategy):
    """Devotion paradigm - Focuses on community support and care resources"""

    def __init__(self):
        super().__init__()
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
            "grant",
            "benefits",
            "program",
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
        
        # Clean and prepare base query
        cleaned_query = self._clean_query_for_support(base_query)

        # Original query
        queries.append(
            {
                "query": cleaned_query,
                "type": "original",
                "weight": 1.0,
                "paradigm": self.paradigm,
                "source_filter": None,
            }
        )

        # Support-focused queries with context-aware modifiers
        relevant_modifiers = self._select_support_modifiers(cleaned_query)
        for modifier in relevant_modifiers[:3]:
            queries.append(
                {
                    "query": f"{cleaned_query} {modifier}",
                    "type": "support_focused",
                    "weight": 0.9,
                    "paradigm": self.paradigm,
                    "source_filter": "community",
                }
            )

        # Generate context-specific resource patterns
        resource_patterns = self._generate_resource_patterns(cleaned_query)

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

        # Adaptive query limit based on complexity
        query_limit = self.get_adaptive_query_limit(base_query)
        return queries[:query_limit]
    
    def _clean_query_for_support(self, query: str) -> str:
        """Clean query for support-focused searches"""
        # Remove emotional or vague terms that don't help with finding resources
        unhelpful_terms = {
            'struggling', 'need', 'help', 'please', 'urgent', 'desperate',
            'looking', 'for', 'find', 'someone', 'anyone'
        }
        words = query.split()
        cleaned = [w for w in words if w.lower() not in unhelpful_terms]
        # Keep at least core terms
        if len(cleaned) < 2:
            return query
        return ' '.join(cleaned)
    
    def _select_support_modifiers(self, query: str) -> List[str]:
        """Select support modifiers based on query context"""
        query_lower = query.lower()
        
        # Prioritize modifiers based on detected need type
        if any(term in query_lower for term in ['mental', 'depression', 'anxiety', 'therapy']):
            return ['mental health support', 'counseling services', 'therapy resources']
        elif any(term in query_lower for term in ['food', 'hunger', 'meal', 'pantry']):
            return ['food assistance', 'food bank', 'meal programs']
        elif any(term in query_lower for term in ['housing', 'homeless', 'shelter', 'rent']):
            return ['housing assistance', 'shelter services', 'rental help']
        elif any(term in query_lower for term in ['medical', 'health', 'doctor', 'clinic']):
            return ['free clinic', 'healthcare services', 'medical assistance']
        else:
            # Generic support modifiers
            return self.query_modifiers[:5]
    
    def _generate_resource_patterns(self, query: str) -> List[str]:
        """Generate resource search patterns based on need type"""
        patterns = []
        query_lower = query.lower()
        
        # Location-aware patterns (if location is mentioned)
        location_terms = self._extract_location(query)
        location_suffix = f" {location_terms}" if location_terms else ""
        
        # Need-specific patterns
        if 'emergency' in query_lower:
            patterns.extend([
                f'emergency {query} hotline{location_suffix}',
                f'24/7 {query} crisis support{location_suffix}',
                f'immediate {query} help{location_suffix}'
            ])
        else:
            patterns.extend([
                f'"{query}" nonprofit organizations{location_suffix}',
                f'{query} "free services"{location_suffix}',
                f'{query} community support groups{location_suffix}',
                f'how to get help with {query}{location_suffix}'
            ])
        
        return patterns[:4]
    
    def _extract_location(self, query: str) -> str:
        """Extract location information from query if present"""
        # Simple location extraction - could be enhanced
        common_location_indicators = ['in', 'near', 'around', 'at']
        words = query.split()
        
        for i, word in enumerate(words):
            if word.lower() in common_location_indicators and i + 1 < len(words):
                # Return the next 1-2 words as location
                location_parts = words[i+1:i+3]
                return ' '.join(location_parts)
        
        return ""

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


class BernardSearchStrategy(BaseSearchStrategy):
    """Analytical paradigm - Focuses on academic research and data"""

    def __init__(self):
        super().__init__()
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
            "meta-analysis",
            "dataset",
            "replication",
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
        
        # Extract key academic concepts
        academic_query = self._prepare_academic_query(base_query)

        # Original query
        queries.append(
            {
                "query": academic_query,
                "type": "original",
                "weight": 1.0,
                "paradigm": self.paradigm,
                "source_filter": None,
            }
        )

        # Academic-focused queries with field-specific modifiers
        field_modifiers = self._identify_research_field_modifiers(academic_query)
        for modifier in field_modifiers[:3]:
            queries.append(
                {
                    "query": f"{academic_query} {modifier}",
                    "type": "academic_focused",
                    "weight": 0.95,
                    "paradigm": self.paradigm,
                    "source_filter": "academic",
                }
            )

        # Generate sophisticated research patterns
        research_patterns = self._generate_academic_patterns(academic_query)

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
            
        # Add citation-based queries for important topics
        if self._is_established_research_topic(academic_query):
            queries.append({
                "query": f'{academic_query} "highly cited" OR "seminal work"',
                "type": "citation_focused",
                "weight": 0.85,
                "paradigm": self.paradigm,
                "source_filter": "academic",
            })

        # Adaptive query limit for academic research (slightly higher base)
        query_limit = min(10, self.get_adaptive_query_limit(base_query) + 2)
        return queries[:query_limit]
    
    def _prepare_academic_query(self, query: str) -> str:
        """Prepare query for academic search by removing colloquialisms"""
        # Remove informal language
        informal_terms = {
            'stuff', 'things', 'basically', 'kind of', 'sort of',
            'really', 'very', 'super', 'totally', 'actually'
        }
        words = query.split()
        cleaned = [w for w in words if w.lower() not in informal_terms]
        
        # Replace colloquial terms with academic equivalents
        academic_replacements = {
            'kids': 'children',
            'teens': 'adolescents',
            'old people': 'elderly',
            'smart': 'intelligent',
            'dumb': 'cognitive impairment'
        }
        
        result = ' '.join(cleaned)
        for colloquial, academic in academic_replacements.items():
            result = result.replace(colloquial, academic)
        
        return result
    
    def _identify_research_field_modifiers(self, query: str) -> List[str]:
        """Identify research field and return appropriate modifiers"""
        query_lower = query.lower()
        
        # Field-specific modifiers
        if any(term in query_lower for term in ['psychology', 'behavior', 'cognitive', 'mental']):
            return ['psychological research', 'behavioral study', 'cognitive science']
        elif any(term in query_lower for term in ['medical', 'disease', 'treatment', 'clinical']):
            return ['clinical trial', 'medical research', 'therapeutic study']
        elif any(term in query_lower for term in ['social', 'society', 'cultural', 'demographic']):
            return ['sociological study', 'social research', 'demographic analysis']
        elif any(term in query_lower for term in ['economic', 'market', 'financial', 'business']):
            return ['economic analysis', 'market research', 'financial study']
        elif any(term in query_lower for term in ['environment', 'climate', 'ecology', 'sustainability']):
            return ['environmental research', 'ecological study', 'climate science']
        else:
            # Generic academic modifiers
            return self.query_modifiers[:5]
    
    def _generate_academic_patterns(self, query: str) -> List[str]:
        """Generate sophisticated academic search patterns"""
        patterns = []
        
        # Methodological patterns
        patterns.extend([
            f'"{query}" quantitative analysis',
            f'"{query}" "research methodology"',
            f'{query} "empirical evidence"'
        ])
        
        # Recent research patterns
        current_year = datetime.now().year
        patterns.append(f'"{query}" "{current_year - 2}..{current_year}"')
        
        # Review and meta-analysis patterns
        patterns.extend([
            f'"systematic review" {query}',
            f'meta-analysis {query}',
            f'"literature review" {query} recent'
        ])
        
        return patterns[:5]
    
    def _is_established_research_topic(self, query: str) -> bool:
        """Check if query relates to established research topics"""
        established_topics = [
            'climate change', 'artificial intelligence', 'machine learning',
            'covid', 'cancer', 'alzheimer', 'quantum', 'genetic',
            'renewable energy', 'sustainable', 'neural network'
        ]
        query_lower = query.lower()
        return any(topic in query_lower for topic in established_topics)

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


class MaeveSearchStrategy(BaseSearchStrategy):
    """Strategic paradigm - Focuses on business intelligence and strategy"""

    def __init__(self):
        super().__init__()
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
            'HBR OR McKinsey OR BCG case study',
        ]

    async def generate_search_queries(
        self, context: SearchContext
    ) -> List[Dict[str, Any]]:
        """Generate Maeve-specific search queries"""
        base_query = context.original_query
        queries = []
        
        # Transform query for strategic focus
        strategic_query = self._prepare_strategic_query(base_query)

        # Original query
        queries.append(
            {
                "query": strategic_query,
                "type": "original",
                "weight": 1.0,
                "paradigm": self.paradigm,
                "source_filter": None,
            }
        )

        # Industry-specific strategic queries
        industry_modifiers = self._identify_industry_modifiers(strategic_query)
        for modifier in industry_modifiers[:3]:
            queries.append(
                {
                    "query": f"{strategic_query} {modifier}",
                    "type": "strategy_focused",
                    "weight": 0.9,
                    "paradigm": self.paradigm,
                    "source_filter": "business",
                }
            )

        # Generate actionable business patterns
        business_patterns = self._generate_strategic_patterns(strategic_query)

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

        # Adaptive query limit based on complexity
        query_limit = self.get_adaptive_query_limit(base_query)
        return queries[:query_limit]
    
    def _prepare_strategic_query(self, query: str) -> str:
        """Prepare query for strategic/business search"""
        # Remove vague business jargon
        jargon_terms = {
            'synergy', 'leverage', 'paradigm', 'proactive', 'holistic',
            'basically', 'essentially', 'innovative', 'disruptive'
        }
        words = query.split()
        cleaned = [w for w in words if w.lower() not in jargon_terms]
        
        # Add strategic context if missing
        strategic_keywords = ['strategy', 'market', 'competitive', 'business', 'industry']
        if not any(keyword in ' '.join(cleaned).lower() for keyword in strategic_keywords):
            # Add appropriate context based on query
            if 'company' in query.lower() or 'product' in query.lower():
                cleaned.append('strategy')
            elif 'improve' in query.lower() or 'increase' in query.lower():
                cleaned.append('optimization')
        
        return ' '.join(cleaned)
    
    def _identify_industry_modifiers(self, query: str) -> List[str]:
        """Identify industry context and return appropriate modifiers"""
        query_lower = query.lower()
        
        # Industry-specific modifiers
        if any(term in query_lower for term in ['tech', 'software', 'saas', 'digital']):
            return ['digital transformation', 'tech strategy', 'SaaS metrics']
        elif any(term in query_lower for term in ['retail', 'ecommerce', 'consumer']):
            return ['retail strategy', 'consumer behavior', 'omnichannel']
        elif any(term in query_lower for term in ['finance', 'banking', 'investment']):
            return ['financial strategy', 'risk management', 'ROI analysis']
        elif any(term in query_lower for term in ['healthcare', 'medical', 'pharma']):
            return ['healthcare market', 'medical device strategy', 'pharma trends']
        elif any(term in query_lower for term in ['manufacturing', 'supply chain', 'logistics']):
            return ['supply chain optimization', 'lean manufacturing', 'logistics strategy']
        else:
            # Generic strategic modifiers
            return ['competitive advantage', 'market positioning', 'growth strategy']
    
    def _generate_strategic_patterns(self, query: str) -> List[str]:
        """Generate actionable strategic search patterns"""
        patterns = []
        
        # ROI and metrics focused
        patterns.append(f'"{query}" ROI "case study"')
        
        # Implementation focused
        patterns.append(f'"how to implement" {query} "best practices"')
        
        # Competitive intelligence
        patterns.append(f'{query} "competitive analysis" benchmark')
        
        # Success stories and failures
        patterns.append(f'{query} "success factors" OR "failure analysis"')
        
        # Framework patterns
        patterns.append(f'{query} framework "step by step"')
        
        # Industry reports
        current_year = datetime.now().year
        patterns.append(f'{query} "industry report" {current_year}')
        # Local advantage / niche strategy
        patterns.append(f'{query} "same-day delivery" local advantage')
        patterns.append(f'{query} "niche strategy" case study')
        
        return patterns[:7]

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
def get_search_strategy(paradigm: Union[str, Any]):
    """Get the appropriate search strategy for a paradigm.

    Accepts internal codes (dolores/bernard/...), enum values (analytical/...),
    or HostParadigm; normalizes to internal code.
    """
    try:
        key = normalize_to_internal_code(paradigm)
    except Exception:
        key = str(paradigm).strip().lower()
    strategies = {
        "dolores": DoloresSearchStrategy(),
        "teddy": TeddySearchStrategy(),
        "bernard": BernardSearchStrategy(),
        "maeve": MaeveSearchStrategy(),
    }

    return strategies.get(key, BernardSearchStrategy())  # Default to Bernard


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

"""
Search API integrations for Four Hosts Research Application
Implements Google Custom Search, Bing Search, and Academic APIs
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import string
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import aiohttp
import fitz  # PyMuPDF
import nltk
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                    wait_exponential)

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URLNormalizer:
    """Handles URL normalization and DOI canonicalization"""

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL to prevent malformed requests"""
        if not url or not isinstance(url, str):
            return ""

        # Remove extra whitespace
        url = url.strip()

        # Handle common DOI patterns
        if url.startswith("10."):
            # Bare DOI - prepend doi.org
            return f"https://doi.org/{url}"

        if "doi.org/" in url:
            # Extract and normalize DOI
            doi_match = re.search(r'doi\.org/(.+)', url)
            if doi_match:
                doi = doi_match.group(1)
                # Avoid double-encoding - decode first if needed
                if '%' in doi:
                    try:
                        doi = unquote(doi)
                    except Exception:
                        pass
                return f"https://doi.org/{doi}"

        # Parse URL components
        try:
            parsed = urlparse(url)

            # Ensure scheme
            if not parsed.scheme:
                url = f"https://{url}"
                parsed = urlparse(url)

            # Normalize domain
            netloc = parsed.netloc.lower()

            # Handle percent-encoded paths carefully
            path = parsed.path
            if path and '%' in path:
                # Only decode if it appears to be over-encoded
                try:
                    decoded = unquote(path)
                    # Check if decoding made it more readable
                    if not re.search(r'%[0-9A-Fa-f]{2}', decoded):
                        path = decoded
                except Exception:
                    pass  # Keep original if decoding fails

            # Reconstruct URL
            normalized = urlunparse((
                parsed.scheme,
                netloc,
                path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))

            return normalized

        except Exception as e:
            logger.warning(f"URL normalization failed for {url}: {e}")
            return url

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is well-formed"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


class CircuitBreaker:
    """Circuit breaker for domains with repeated failures"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts = {}
        self.last_failure_time = {}
        self.blocked_domains = set()

    def should_allow_request(self, domain: str) -> bool:
        """Check if requests to domain should be allowed"""
        if domain not in self.blocked_domains:
            return True

        # Check if recovery period has passed
        if domain in self.last_failure_time:
            elapsed = time.time() - self.last_failure_time[domain]
            if elapsed > self.recovery_timeout:
                logger.info(f"Circuit breaker recovery: allowing requests to {domain}")
                self.blocked_domains.discard(domain)
                self.failure_counts[domain] = 0
                return True

        return False

    def record_failure(self, domain: str):
        """Record a failure for the domain"""
        self.failure_counts[domain] = self.failure_counts.get(domain, 0) + 1
        self.last_failure_time[domain] = time.time()

        if self.failure_counts[domain] >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker triggered for {domain} after "
                f"{self.failure_counts[domain]} failures"
            )
            self.blocked_domains.add(domain)

    def record_success(self, domain: str):
        """Record a success for the domain"""
        if domain in self.failure_counts:
            self.failure_counts[domain] = max(0, self.failure_counts[domain] - 1)


class RespectfulFetcher:
    """Fetches content while respecting robots.txt and rate limits."""

    def __init__(self):
        self.robot_parsers: Dict[str, RobotFileParser] = {}
        self.last_fetch: Dict[str, float] = {}
        self.user_agent = (
            "FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)"
        )
        self.circuit_breaker = CircuitBreaker()

    async def can_fetch(self, url: str) -> bool:
        """Check robots.txt before fetching"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain not in self.robot_parsers:
            try:
                rp = RobotFileParser()
                rp.set_url(f"{domain}/robots.txt")
                await asyncio.to_thread(rp.read)
                self.robot_parsers[domain] = rp
            except Exception as e:
                logger.debug(f"Could not fetch robots.txt for {domain}: {e}")
                # If we can't fetch robots.txt, we'll be conservative and allow
                return True

        return self.robot_parsers[domain].can_fetch(self.user_agent, url)

    async def respectful_fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch with rate limiting and robots.txt compliance"""
        # Normalize URL first
        normalized_url = URLNormalizer.normalize_url(url)
        if not normalized_url or not URLNormalizer.is_valid_url(normalized_url):
            logger.warning(f"Invalid URL after normalization: {url} -> {normalized_url}")
            return None

        domain = urlparse(normalized_url).netloc

        # Check circuit breaker
        if not self.circuit_breaker.should_allow_request(domain):
            logger.debug(f"Circuit breaker blocking requests to {domain}")
            return None

        # Check robots.txt
        if not await self.can_fetch(normalized_url):
            logger.debug(f"Robots.txt disallows fetching {normalized_url}")
            return None

        # Rate limit per domain (1 second between requests)
        if domain in self.last_fetch:
            elapsed = time.time() - self.last_fetch[domain]
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)

        self.last_fetch[domain] = time.time()

        # Fetch with proper headers and record result
        try:
            content = await fetch_and_parse_url(session, normalized_url)
            if content:
                self.circuit_breaker.record_success(domain)
            return content
        except Exception as e:
            self.circuit_breaker.record_failure(domain)
            raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError),
)
async def fetch_and_parse_url(session: aiohttp.ClientSession, url: str) -> str:
    """
    Fetch URL content with ethical headers and parse it to remove HTML tags
    or extract text from PDF.
    """
    headers = {
        "User-Agent": (
            "FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with session.get(
            url, timeout=timeout, headers=headers, allow_redirects=True
        ) as response:
            if response.status == 200:
                content_type = (
                    response.headers.get("Content-Type", "") or ""
                ).lower()
                if "application/pdf" in content_type:
                    pdf_content = await response.read()
                    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                        text = "".join(page.get_text() for page in doc)
                    return text
                else:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, "html.parser")
                    for script in soup(["script", "style"]):
                        script.extract()
                    return soup.get_text(separator=" ", strip=True)
            elif response.status == 403:
                logger.debug(f"Access denied (403) for {url}")
                return ""
            elif response.status == 429:
                logger.warning(f"Rate limited (429) for {url}")
                return ""
            elif 500 <= response.status < 600:
                logger.warning(f"Server error ({response.status}) for {url}")
                response.raise_for_status()
            else:
                logger.warning(f"Failed to fetch {url}, status: {response.status}")
                return ""
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"Network error fetching {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error fetching or parsing {url}: {e}", exc_info=True)
        return ""
    return ""


@dataclass
class SearchResult:
    """Standardized search result across all APIs"""

    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[datetime] = None
    domain: str = ""
    credibility_score: float = 0.0
    bias_rating: Optional[str] = None
    result_type: str = "web"  # web, academic, news
    content: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0  # New field for relevance scoring
    is_primary_source: bool = False  # New field to identify primary sources
    # Enhanced metadata fields
    author: Optional[str] = None
    publication_type: Optional[str] = None  # research, article, blog, report, etc.
    citation_count: Optional[int] = None
    content_length: Optional[int] = None
    last_modified: Optional[datetime] = None
    # Content hash for deduplication
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        if not self.domain and self.url:
            # Extract domain from URL
            from urllib.parse import urlparse

            self.domain = urlparse(self.url).netloc.lower()
        
        # Generate content hash for deduplication
        if self.title and self.snippet and isinstance(self.title, str) and isinstance(self.snippet, str):
            content_str = f"{self.title.lower().strip()}{self.snippet.lower().strip()}"
            self.content_hash = hashlib.md5(content_str.encode()).hexdigest()


@dataclass
class SearchConfig:
    """Configuration for search requests"""

    max_results: int = 50
    language: str = "en"
    region: str = "us"
    safe_search: str = "moderate"
    date_range: Optional[str] = None  # "d", "w", "m", "y" for day/week/month/year
    source_types: List[str] = field(default_factory=list)  # ["academic", "news", "web"]
    # Authority scoring configuration
    authority_whitelist: List[str] = field(default_factory=list)  # Preferred domains
    authority_blacklist: List[str] = field(default_factory=list)  # Blocked domains
    prefer_primary_sources: bool = True
    min_relevance_score: float = 0.25


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, calls_per_minute: int = 100):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = datetime.now(timezone.utc)
        # Remove calls older than 1 minute
        self.calls = [
            call_time
            for call_time in self.calls
            if now - call_time < timedelta(minutes=1)
        ]

        if len(self.calls) >= self.calls_per_minute:
            # Wait until oldest call is > 1 minute old
            wait_time = 60 - (now - self.calls[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self.calls.append(now)


class QueryOptimizer:
    """
    Optimizes search queries by preserving user intent through entity extraction,
    intelligent stopword removal, and multi-query expansion.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # A more conservative set of noise terms
        self.noise_terms = {
            'information', 'details', 'find', 'show', 'tell'
        }
        # Known technical compound terms
        self.known_entities = [
            'context engineering', 'web applications', 'artificial intelligence',
            'machine learning', 'deep learning', 'neural network', 'neural networks',
            'large language model', 'large language models', 'state of the art',
            'natural language processing', 'computer vision', 'reinforcement learning',
            'generative AI', 'transformer models', 'foundation models'
        ]
        # Domain-specific term dictionaries
        self.domain_terms = {
            'medical': ['diagnosis', 'treatment', 'clinical', 'patient', 'therapy'],
            'technical': ['algorithm', 'implementation', 'architecture', 'framework'],
            'business': ['strategy', 'revenue', 'market', 'competitive', 'growth'],
            'academic': ['research', 'study', 'methodology', 'findings', 'hypothesis']
        }

    def _extract_entities(self, query: str) -> Tuple[List[str], str]:
        """Extracts quoted phrases and known entities from the query."""
        entities = []
        temp_query = query

        # 1. Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', temp_query)
        for phrase in quoted_phrases:
            entities.append(phrase)
            temp_query = temp_query.replace(f'"{phrase}"', "")

        # 2. Extract known technical entities
        for entity in self.known_entities:
            if entity in temp_query.lower():
                match = re.search(re.escape(entity), temp_query, re.IGNORECASE)
                if match:
                    entities.append(match.group(0))
                    temp_query = temp_query.replace(match.group(0), "")

        # 3. Extract potential proper nouns (e.g., "Maeve")
        proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", temp_query)
        common_words = {"I", "A"}
        for noun in proper_nouns:
            if noun not in common_words and noun.lower() not in self.stop_words:
                entities.append(noun)
                temp_query = temp_query.replace(noun, "")

        return list(set(entities)), temp_query

    def _intelligent_stopword_removal(self, text: str) -> List[str]:
        """Removes stopwords and noise terms, preserving query structure."""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.noise_terms]
        tokens = [
            t for t in tokens
            if t not in self.stop_words and t not in string.punctuation
        ]
        return tokens

    def get_key_terms(self, query: str) -> List[str]:
        """Extracts all significant entities and keywords from a query."""
        protected_entities, remaining_text = self._extract_entities(query)
        keywords = self._intelligent_stopword_removal(remaining_text)
        # Return entities without quotes and keywords
        return [e.replace('"', '') for e in protected_entities] + keywords

    def generate_query_variations(self, query: str, paradigm: Optional[str] = None) -> Dict[str, str]:
        """
        Generates an expanded set of query variations to improve search recall.
        Now includes synonym expansion, related concepts, and domain-specific terminology.
        """
        protected_entities, remaining_text = self._extract_entities(query)

        # Clean the remaining text
        keywords = self._intelligent_stopword_removal(remaining_text)

        # Combine protected entities (quoted) and keywords
        all_terms = [f'"{e}"' for e in protected_entities] + keywords

        if not all_terms:
            # Fallback to a simple cleaned query if no terms are extracted
            primary_query = ' '.join(self._intelligent_stopword_removal(query))
            if not primary_query: return {"primary": query} # Ultimate fallback
        else:
            primary_query = " AND ".join(all_terms)

        # Generate expanded variations (5-7 total)
        variations = {
            "primary": primary_query
        }

        # 1. Semantic variation: use OR for some terms
        if len(protected_entities) > 1:
            semantic_query = f"({' OR '.join([f'\"{e}\"' for e in protected_entities])}) AND {' '.join(keywords)}"
            variations["semantic"] = semantic_query
        else:
            variations["semantic"] = primary_query.replace(" AND ", " ")

        # 2. Question variation
        question_starters = ['what is', 'how does', 'explain', 'why does', 'when did']
        if not any(query.lower().startswith(s) for s in question_starters):
            variations["question"] = (
                "what is the relationship between "
                f"{' and '.join(all_terms)}"
            )

        # 3. Synonym expansion variation
        synonym_terms = self._expand_synonyms(keywords)
        if synonym_terms != keywords:
            synonym_query = " ".join([f'"{e}"' for e in protected_entities] + synonym_terms)
            variations["synonym"] = synonym_query

        # 4. Related concepts variation
        related_terms = self._get_related_concepts(keywords)
        if related_terms:
            related_query = f"{primary_query} OR ({' OR '.join(related_terms)})"
            variations["related"] = related_query

        # 5. Domain-specific variation (if paradigm provided)
        if paradigm:
            domain_query = self._add_domain_specific_terms(primary_query, paradigm)
            if domain_query != primary_query:
                variations["domain_specific"] = domain_query

        # 6. Broad match variation (less restrictive)
        broad_terms = protected_entities + keywords
        if len(broad_terms) > 2:
            # Use only the most important terms
            broad_query = " ".join(broad_terms[:3])
            variations["broad"] = broad_query

        # 7. Exact phrase variation (for finding specific content)
        if len(all_terms) >= 2 and len(query.split()) <= 6:
            exact_phrase = f'"{query}"'
            variations["exact_phrase"] = exact_phrase

        return variations

    def _expand_synonyms(self, terms: List[str]) -> List[str]:
        """Expand terms with synonyms using WordNet"""
        expanded = []
        for term in terms:
            # Add original term
            expanded.append(term)
            # Get synonyms (limit to 2 per term to avoid explosion)
            synsets = wordnet.synsets(term)
            if synsets:
                synonyms = set()
                for syn in synsets[:2]:  # Limit synsets
                    for lemma in syn.lemmas()[:3]:  # Limit lemmas
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != term.lower():
                            synonyms.add(synonym)
                expanded.extend(list(synonyms)[:2])  # Add max 2 synonyms
        return expanded

    def _get_related_concepts(self, terms: List[str]) -> List[str]:
        """Get related concepts for the terms"""
        related_concepts_map = {
            'AI': ['artificial intelligence', 'machine learning', 'deep learning'],
            'ML': ['machine learning', 'algorithms', 'models'],
            'ethics': ['moral', 'ethical considerations', 'responsible AI'],
            'security': ['cybersecurity', 'privacy', 'data protection'],
            'climate': ['climate change', 'global warming', 'environmental'],
            'health': ['healthcare', 'medical', 'wellness'],
            'finance': ['financial', 'economic', 'investment'],
            'education': ['learning', 'teaching', 'academic']
        }
        
        related = []
        for term in terms:
            term_lower = term.lower()
            for key, concepts in related_concepts_map.items():
                if key.lower() in term_lower or term_lower in key.lower():
                    related.extend(concepts[:2])  # Limit to 2 related concepts
        return list(set(related))[:3]  # Return max 3 unique related concepts

    def _add_domain_specific_terms(self, query: str, paradigm: str) -> str:
        """Add domain-specific terms based on paradigm"""
        paradigm_terms = {
            'dolores': ['investigation', 'expose', 'systemic', 'corruption'],
            'teddy': ['support', 'community', 'help', 'care'],
            'bernard': ['research', 'analysis', 'data', 'evidence'],
            'maeve': ['strategy', 'business', 'optimization', 'market']
        }
        
        if paradigm.lower() in paradigm_terms:
            domain_term = paradigm_terms[paradigm.lower()][0]
            return f"{query} {domain_term}"
        return query

    def optimize_query(self, query: str, paradigm: Optional[str] = None) -> str:
        """
        Returns the primary, most precise query variation.
        This method maintains compatibility with the existing interface.
        """
        variations = self.generate_query_variations(query, paradigm)
        logger.info(f"Generated {len(variations)} query variations for '{query}'")
        return variations.get("primary", query)


class ContentRelevanceFilter:
    """Filters search results for relevance at retrieval time"""

    def __init__(self):
        self.query_optimizer = QueryOptimizer()
        # Track cross-source agreement
        self.claim_tracker: Dict[str, List[SearchResult]] = defaultdict(list)
        self.consensus_threshold = 0.7  # 70% agreement threshold

    def calculate_relevance_score(self, result: SearchResult, original_query: str, key_terms: List[str], config: Optional[SearchConfig] = None) -> float:
        """Calculate relevance score for a search result with enhanced authority scoring"""
        score = 0.0

        # Extract text for analysis
        text_content = f"{result.title} {result.snippet}".lower()

        # 1. Key term frequency (35% weight - reduced to make room for authority)
        term_frequency_score = self._calculate_term_frequency(text_content, key_terms)
        score += term_frequency_score * 0.35

        # 2. Title relevance (25% weight)
        title_score = self._calculate_title_relevance(result.title.lower(), key_terms)
        score += title_score * 0.25

        # 3. Content freshness (10% weight)
        freshness_score = self._calculate_freshness_score(result.published_date)
        score += freshness_score * 0.1

        # 4. Source authority score (15% weight - increased)
        source_score = self._calculate_source_type_score(result, config)
        score += source_score * 0.15

        # 5. Exact phrase matching (10% weight)
        phrase_score = self._calculate_phrase_match_score(text_content, original_query)
        score += phrase_score * 0.1

        # 6. Metadata quality bonus (5% weight)
        metadata_score = self._calculate_metadata_score(result)
        score += metadata_score * 0.05

        return min(1.0, score)

    def _calculate_metadata_score(self, result: SearchResult) -> float:
        """Calculate score based on metadata completeness"""
        score = 0.0
        metadata_fields = [
            result.author is not None,
            result.publication_type is not None,
            result.citation_count is not None,
            result.published_date is not None,
            result.content_length is not None
        ]
        score = sum(metadata_fields) / len(metadata_fields)
        return score

    def _calculate_term_frequency(self, text: str, key_terms: List[str]) -> float:
        """Calculate normalized term frequency score"""
        if not key_terms:
            return 0.0

        matches = sum(1 for term in key_terms if term in text)
        return matches / len(key_terms)

    def _calculate_title_relevance(self, title: str, key_terms: List[str]) -> float:
        """Calculate title relevance score"""
        if not key_terms:
            return 0.0

        # Higher score for terms appearing in title
        title_matches = sum(1 for term in key_terms if term in title)
        base_score = title_matches / len(key_terms)

        # Bonus for exact order matching
        if len(key_terms) >= 2:
            consecutive_matches = 0
            for i in range(len(key_terms) - 1):
                if f"{key_terms[i]} {key_terms[i+1]}" in title:
                    consecutive_matches += 1
            if consecutive_matches > 0:
                base_score = min(1.0, base_score + 0.2)

        return base_score

    def _calculate_freshness_score(self, published_date: Optional[datetime]) -> float:
        """Calculate content freshness score"""
        if not published_date:
            return 0.5  # Neutral score for unknown dates

        # Ensure published_date is timezone-aware
        if published_date.tzinfo is None:
            published_date = published_date.replace(tzinfo=timezone.utc)
            
        days_old = (datetime.now(timezone.utc) - published_date).days

        if days_old <= 7:
            return 1.0
        elif days_old <= 30:
            return 0.8
        elif days_old <= 90:
            return 0.6
        elif days_old <= 365:
            return 0.4
        else:
            return 0.2

    def _calculate_source_type_score(self, result: SearchResult, config: Optional[SearchConfig] = None) -> float:
        """Calculate source type score based on result type and domain with authority scoring"""
        score = 0.5  # Base score
        
        # Check whitelist/blacklist if config provided
        if config:
            if config.authority_blacklist and any(blocked in result.domain for blocked in config.authority_blacklist):
                return 0.1  # Very low score for blacklisted domains
            
            if config.authority_whitelist and any(allowed in result.domain for allowed in config.authority_whitelist):
                score = 0.95  # Very high score for whitelisted domains
                result.is_primary_source = True
                return score
        
        # Academic sources get higher base score
        if result.result_type == "academic":
            score = 0.9
            result.is_primary_source = True
            # Extra points for citation count
            if result.citation_count and result.citation_count > 10:
                score = min(1.0, score + 0.05)
            return score

        # Check for primary source indicators
        primary_indicators = [
            '.gov', '.edu', '.org',
            'official', 'foundation', 'institute',
            'journal', 'research', 'university',
            'academy', 'national', 'federal'
        ]

        domain_lower = result.domain.lower()
        if any(indicator in domain_lower for indicator in primary_indicators):
            result.is_primary_source = True
            # Official government sources get highest score
            if '.gov' in domain_lower:
                return 0.95
            # Educational institutions
            elif '.edu' in domain_lower:
                return 0.9
            # Other primary sources
            else:
                return 0.8

        # News sources - differentiate by credibility
        if result.result_type == "news":
            credible_news = ['reuters', 'ap.org', 'bbc', 'npr', 'wsj', 'nytimes', 'guardian']
            if any(source in domain_lower for source in credible_news):
                return 0.75
            return 0.65

        # Default web sources
        return score

    def _calculate_phrase_match_score(self, text: str, original_query: str) -> float:
        """Calculate score for exact phrase matching"""
        # Look for quoted phrases in the original query
        quoted_phrases = re.findall(r'"([^"]+)"', original_query)

        if not quoted_phrases:
            return 0.5  # Neutral score if no quoted phrases

        matches = sum(1 for phrase in quoted_phrases if phrase.lower() in text)
        return matches / len(quoted_phrases)

    def filter_results(
        self,
        results: List[SearchResult],
        original_query: str,
        min_relevance: float = 0.25,
        config: Optional[SearchConfig] = None,
        detect_consensus: bool = True
    ) -> List[SearchResult]:
        """Filter and rank results by relevance with cross-source agreement detection"""
        # Use config min_relevance if provided
        if config and hasattr(config, 'min_relevance_score'):
            min_relevance = config.min_relevance_score
            
        # Domains known to block scrapers or have heavy paywalls
        blocked_domains = {
            "sciencedirect.com", "springer.com", "wiley.com", "jstor.org",
            "tandfonline.com", "sagepub.com",
        }

        pre_filtered = []
        for result in results:
            if any(blocked in result.domain for blocked in blocked_domains):
                # Skip content fetch for known blocking domains
                result.content = f"Summary from search results: {result.snippet}"
                result.raw_data['content_source'] = 'search_api_snippet'
            pre_filtered.append(result)

        # Extract key terms once
        key_terms = self.query_optimizer.get_key_terms(original_query)

        # Calculate relevance scores
        for result in pre_filtered:
            result.relevance_score = self.calculate_relevance_score(result, original_query, key_terms, config)

        # Filter by minimum relevance
        filtered = [r for r in pre_filtered if r.relevance_score >= min_relevance]

        # Detect cross-source agreement if enabled
        if detect_consensus and len(filtered) > 1:
            filtered = self._detect_cross_source_agreement(filtered, key_terms)

        # Sort by relevance score (descending)
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"Filtered {len(results)} results to {len(filtered)} with min relevance {min_relevance}")

        return filtered

    def _detect_cross_source_agreement(self, results: List[SearchResult], key_terms: List[str]) -> List[SearchResult]:
        """Detect and score cross-source agreement on key claims"""
        # Extract key claims from each result
        claim_groups = defaultdict(list)
        
        for result in results:
            # Extract potential claims (sentences containing key terms)
            text = f"{result.title} {result.snippet}".lower()
            sentences = text.split('.')
            
            for sentence in sentences:
                if any(term in sentence for term in key_terms):
                    # Normalize claim for grouping
                    normalized_claim = self._normalize_claim(sentence)
                    if normalized_claim:
                        claim_groups[normalized_claim].append(result)
        
        # Mark results that have consensus
        consensus_results = set()
        for claim, supporting_results in claim_groups.items():
            if len(supporting_results) >= 2:
                # Multiple sources agree on this claim
                for result in supporting_results:
                    consensus_results.add(id(result))
                    # Boost relevance score for consensus
                    result.relevance_score = min(1.0, result.relevance_score + 0.1)
                    # Add consensus metadata
                    if 'consensus_claims' not in result.raw_data:
                        result.raw_data['consensus_claims'] = []
                    result.raw_data['consensus_claims'].append({
                        'claim': claim,
                        'supporting_sources': len(supporting_results)
                    })
        
        # Flag potential contradictions
        self._flag_contradictions(results, claim_groups)
        
        return results

    def _normalize_claim(self, sentence: str) -> Optional[str]:
        """Normalize a claim for comparison"""
        # Remove extra whitespace and punctuation
        normalized = re.sub(r'[^\w\s]', '', sentence.lower().strip())
        normalized = ' '.join(normalized.split())
        
        # Skip very short claims
        if len(normalized.split()) < 4:
            return None
            
        return normalized

    def _flag_contradictions(self, results: List[SearchResult], claim_groups: Dict[str, List[SearchResult]]):
        """Flag potential contradictions in results"""
        # Simple contradiction detection based on negation patterns
        negation_words = ['not', 'no', 'never', 'none', 'neither', 'nor', 'without']
        
        for claim, supporting_results in claim_groups.items():
            # Check for potential negated version of this claim
            for neg_word in negation_words:
                if neg_word in claim:
                    # Look for similar claim without negation
                    positive_claim = claim.replace(neg_word, '').strip()
                    for other_claim in claim_groups:
                        if other_claim != claim and positive_claim in other_claim:
                            # Potential contradiction found
                            for result in supporting_results:
                                if 'potential_contradictions' not in result.raw_data:
                                    result.raw_data['potential_contradictions'] = []
                                result.raw_data['potential_contradictions'].append({
                                    'this_claim': claim,
                                    'contradicting_claim': other_claim,
                                    'sources': len(claim_groups[other_claim])
                                })


class BaseSearchAPI:
    """Base class for all search APIs"""

    def __init__(self, api_key: str, rate_limiter: Optional[RateLimiter] = None):
        self.api_key = api_key
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None
        self.query_optimizer = QueryOptimizer()
        self.relevance_filter = ContentRelevanceFilter()
        self._session_closed = False

    async def __aenter__(self):
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self._session_closed = False
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()

    async def close_session(self):
        """Safely close the session"""
        if self.session and not self.session.closed and not self._session_closed:
            try:
                await self.session.close()
                self._session_closed = True
                logger.debug(f"Closed session for {self.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Error closing session for {self.__class__.__name__}: {e}")

    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Override in subclasses"""
        raise NotImplementedError

    async def search_with_variations(self, query: str, config: SearchConfig, paradigm: Optional[str] = None) -> List[SearchResult]:
        """Search using multiple query variations for better coverage"""
        variations = self.query_optimizer.generate_query_variations(query, paradigm)
        all_results = []
        seen_urls = set()
        
        # Search with each variation
        for variant_type, variant_query in list(variations.items())[:3]:  # Limit to top 3 variations
            try:
                variant_results = await self.search(variant_query, config)
                
                # Deduplicate by URL
                for result in variant_results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        result.raw_data['query_variant'] = variant_type
                        all_results.append(result)
                        
            except Exception as e:
                logger.warning(f"Search failed for {variant_type} variant: {e}")
                continue
        
        return all_results


class GoogleCustomSearchAPI(BaseSearchAPI):
    """Google Custom Search API implementation"""

    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        super().__init__(api_key, rate_limiter)
        self.search_engine_id = search_engine_id
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def search(
        self, query: str, config: SearchConfig
    ) -> List[SearchResult]:
        """Search using Google Custom Search API with proper error handling."""
        await self.rate_limiter.wait_if_needed()

        # Validate parameters to prevent 400 errors
        if not query or not isinstance(query, str) or not query.strip():
            logger.error("Empty or invalid query provided to Google search")
            return []

        # Generate query variations for better coverage
        variations = self.query_optimizer.generate_query_variations(query)
        optimized_query = variations.get("primary", query)
        logger.info(f"Using primary query variation: '{optimized_query}'")

        if not self.api_key:
            logger.error("Google API key not configured")
            return []

        if not self.search_engine_id:
            logger.error("Google search engine ID not configured")
            return []

        # Clean and validate query
        clean_query = optimized_query.strip()[:2048]  # Google has a query length limit

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": clean_query,
            "num": min(config.max_results, 10),
        }

        # Add optional parameters only if they have valid values
        if config.language and config.language != "auto":
            params["lr"] = f"lang_{config.language}"

        if config.region and config.region.lower() != "global":
            params["gl"] = config.region

        if config.safe_search:
            # Google Custom Search API only accepts "active" or "off"
            # Map "moderate" to "active" for compatibility
            if config.safe_search == "moderate":
                params["safe"] = "active"
            elif config.safe_search in ["active", "off"]:
                params["safe"] = config.safe_search

        if config.date_range:
            params["dateRestrict"] = config.date_range

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = self._parse_google_results(data)
                    # Apply relevance filtering with config
                    filtered_results = self.relevance_filter.filter_results(results, query, config=config)
                    return filtered_results
                elif response.status == 429:
                    logger.warning("Google API rate limit exceeded")
                    raise aiohttp.ClientError("Rate limit exceeded")
                else:
                    # Get response body for detailed error information
                    try:
                        # Try to parse as JSON first for structured error info
                        try:
                            error_data = await response.json()
                            error = error_data.get("error", {})
                            message = error.get("message", "Unknown error")
                            code = error.get("code", "N/A")
                            logger.error(
                                f"Google API error: {response.status} - "
                                f"Code: {code}, Message: {message}"
                            )
                        except (json.JSONDecodeError, aiohttp.ClientError):
                            error_body = await response.text()
                            logger.error(
                                f"Google API error: {response.status} - {error_body}"
                            )
                    except Exception as read_e:
                        logger.error(
                            f"Google API error: {response.status}. "
                            f"Could not read response body: {read_e}"
                        )
                    return []
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Google search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Google search: {e}", exc_info=True)
            return []

    def _parse_google_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Google API response with enhanced metadata extraction."""
        results = []
        items = data.get("items", [])
        if not isinstance(items, list):
            return []

        for item in items:
            if not isinstance(item, dict):
                continue
                
            # Extract enhanced metadata
            pagemap = item.get("pagemap", {})
            metatags = pagemap.get("metatags", [{}])[0] if pagemap.get("metatags") else {}
            
            # Try to extract author
            author = (
                metatags.get("author") or 
                metatags.get("article:author") or 
                metatags.get("dc.creator")
            )
            
            # Try to extract publication date
            date_str = (
                metatags.get("article:published_time") or
                metatags.get("publishdate") or
                metatags.get("dc.date")
            )
            published_date = None
            if date_str:
                try:
                    published_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if published_date.tzinfo is None:
                        published_date = published_date.replace(tzinfo=timezone.utc)
                except:
                    pass
                    
            # Extract content type
            publication_type = metatags.get("og:type", "web")
            
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="google_custom_search",
                domain=item.get("displayLink", ""),
                raw_data=item,
                author=author,
                published_date=published_date,
                publication_type=publication_type,
                content_length=len(item.get("snippet", ""))
            )
            results.append(result)
        return results

    @staticmethod
    def enhance_results_with_snippets(results: List[SearchResult]) -> List[SearchResult]:
        """Enhance results using search snippets when full content unavailable"""
        for result in results:
            if not result.content and result.snippet:
                # Use the snippet as primary content
                result.content = f"Summary from search results: {result.snippet}"
                # Mark that this is snippet-only content
                result.raw_data['content_type'] = 'snippet_only'
                result.raw_data['content_source'] = 'search_api_snippet'
        return results


class ArxivAPI(BaseSearchAPI):
    """ArXiv academic paper search API"""

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        super().__init__("", rate_limiter)  # ArXiv is free, no API key needed
        self.base_url = "http://export.arxiv.org/api/query"

    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search ArXiv for academic papers"""
        await self.rate_limiter.wait_if_needed()

        # Optimize query for academic search
        optimized_query = self.query_optimizer.optimize_query(query)

        params = {
            "search_query": f"all:{optimized_query}",
            "start": 0,
            "max_results": min(config.max_results, 100),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    results = self._parse_arxiv_results(xml_data)
                    # Apply relevance filtering with higher threshold
                    return self.relevance_filter.filter_results(
                        results, query, min_relevance=0.4, config=config
                    )
                else:
                    logger.error(f"ArXiv API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"ArXiv search failed: {str(e)}")
            return []

    def _parse_arxiv_results(self, xml_data: str) -> List[SearchResult]:
        """Parse ArXiv XML response"""
        import xml.etree.ElementTree as ET

        results = []
        try:
            root = ET.fromstring(xml_data)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", namespace):
                title_elem = entry.find("atom:title", namespace)
                summary_elem = entry.find("atom:summary", namespace)
                link_elem = entry.find("atom:id", namespace)
                published_elem = entry.find("atom:published", namespace)

                if title_elem is not None and link_elem is not None and title_elem.text:
                    published_date = None
                    if published_elem is not None and published_elem.text:
                        try:
                            dt_str = published_elem.text.replace("Z", "+00:00")
                            published_date = datetime.fromisoformat(dt_str)
                            if published_date.tzinfo is None:
                                published_date = published_date.replace(
                                    tzinfo=timezone.utc
                                )
                        except (ValueError, TypeError):
                            pass

                    snippet_text = ""
                    if summary_elem is not None and summary_elem.text:
                        snippet_text = (summary_elem.text[:300] + "...")

                    # Extract authors
                    authors = []
                    for author in entry.findall("atom:author", namespace):
                        name_elem = author.find("atom:name", namespace)
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text)
                    
                    author_str = ", ".join(authors[:3]) if authors else None
                    if len(authors) > 3:
                        author_str += f" et al. ({len(authors)} authors)"

                    result = SearchResult(
                        title=title_elem.text.strip(),
                        url=link_elem.text or "",
                        snippet=snippet_text,
                        source="arxiv",
                        published_date=published_date,
                        result_type="academic",
                        domain="arxiv.org",
                        author=author_str,
                        publication_type="research_paper",
                        content_length=len(summary_elem.text) if summary_elem is not None and summary_elem.text else 0
                    )
                    results.append(result)

        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML: {str(e)}")

        return results


class BraveSearchAPI(BaseSearchAPI):
    """Brave Search API implementation with full feature support"""

    def __init__(self, api_key: str, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(api_key, rate_limiter)
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def search(
        self, query: str, config: SearchConfig
    ) -> List[SearchResult]:
        """Search using Brave Search API with comprehensive result parsing."""
        await self.rate_limiter.wait_if_needed()

        # Optimize query
        optimized_query = self.query_optimizer.optimize_query(query)

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        # Core parameters
        params = {
            "q": optimized_query[:400],  # Max 400 chars per API docs
            "count": min(config.max_results, 20),  # Brave max is 20 per request
            "search_lang": config.language,
            "country": config.region.upper() if config.region else "US",
            "safesearch": config.safe_search,
            "text_decorations": "true",  # Include highlighting (as string)
            "spellcheck": "true",  # Enable spell correction (as string)
        }

        # Add freshness filter if date range is specified
        if config.date_range:
            freshness_map = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}
            if config.date_range in freshness_map:
                params["freshness"] = freshness_map[config.date_range]

        # Add result filters based on source types
        if config.source_types:
            # Map our source types to Brave's result filters
            result_filters = []
            if "web" in config.source_types:
                result_filters.append("web")
            if "news" in config.source_types:
                result_filters.append("news")
            if "academic" in config.source_types:
                result_filters.extend(["faq", "discussions"])  # Academic-like content
            if result_filters:
                params["result_filter"] = ",".join(result_filters)

        try:
            async with self.session.get(
                self.base_url, headers=headers, params=params
            ) as response:
                # Check rate limit headers
                if "x-ratelimit-remaining" in response.headers:
                    try:
                        remaining_str = response.headers["x-ratelimit-remaining"]
                        remaining = int(remaining_str.split(",")[0])
                        if remaining < 5:
                            logger.warning(
                                f"Brave API rate limit low: {remaining} "
                                "requests remaining this second"
                            )
                    except (ValueError, IndexError):
                        pass  # Ignore if header is malformed

                if response.status == 200:
                    data = await response.json()
                    results = self._parse_brave_results(data)
                    # Apply relevance filtering with config
                    filtered_results = self.relevance_filter.filter_results(results, query, config=config)
                    return filtered_results
                elif response.status == 401:
                    logger.error("Brave API authentication failed - check API key")
                    return []
                elif response.status == 429:
                    retry_after = response.headers.get("retry-after", "60")
                    logger.warning(
                        f"Brave API rate limit exceeded. "
                        f"Retry after {retry_after} seconds"
                    )
                    raise aiohttp.ClientError(
                        f"Rate limit exceeded. Retry after {retry_after}s"
                    )
                else:
                    logger.error(f"Brave API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Brave search failed: {str(e)}")
            raise

    def _parse_brave_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Brave API response according to documented structure."""
        results = []
        result_types = ["web", "news", "faq", "discussions"]

        for r_type in result_types:
            for item in data.get(r_type, {}).get("results", []):
                if not isinstance(item, dict):
                    continue

                published_date = None
                if item.get("age"):
                    try:
                        dt_str = item["age"].replace("Z", "+00:00")
                        published_date = datetime.fromisoformat(dt_str)
                        if published_date.tzinfo is None:
                            published_date = published_date.replace(
                                tzinfo=timezone.utc
                            )
                    except (ValueError, TypeError):
                        pass

                meta_url = item.get("meta_url", {})
                domain = meta_url.get("hostname", "") if meta_url else ""
                if not domain and item.get("url"):
                    domain = urlparse(item["url"]).netloc

                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    source="brave_search",
                    domain=domain,
                    published_date=published_date,
                    result_type=r_type,
                    raw_data=item,
                )
                results.append(result)
        return results


class PubMedAPI(BaseSearchAPI):
    """PubMed/NCBI API for medical and life science papers"""

    def __init__(
        self, api_key: Optional[str] = None, rate_limiter: Optional[RateLimiter] = None
    ):
        super().__init__(api_key or "", rate_limiter)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search PubMed for medical papers"""
        await self.rate_limiter.wait_if_needed()

        # Optimize query for medical/scientific search
        optimized_query = self.query_optimizer.optimize_query(query)

        # First, search for PMIDs
        search_params = {
            "db": "pubmed",
            "term": optimized_query,
            "retmax": min(config.max_results, 100),
            "retmode": "json",
        }

        if self.api_key:
            search_params["api_key"] = self.api_key

        try:
            # Search for article IDs
            async with self.session.get(
                f"{self.base_url}/esearch.fcgi", params=search_params
            ) as response:
                if response.status != 200:
                    logger.error(f"PubMed search error: {response.status}")
                    return []

                search_data = await response.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])

                if not pmids:
                    return []

            # Fetch article details
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids[:20]),  # Limit to first 20 for performance
                "retmode": "xml",
            }

            if self.api_key:
                fetch_params["api_key"] = self.api_key

            async with self.session.get(
                f"{self.base_url}/efetch.fcgi", params=fetch_params
            ) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    results = self._parse_pubmed_results(xml_data)
                    # Apply relevance filtering with higher threshold
                    return self.relevance_filter.filter_results(
                        results, query, min_relevance=0.4, config=config
                    )
                else:
                    logger.error(f"PubMed fetch error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"PubMed search failed: {str(e)}")
            return []

    def _parse_pubmed_results(self, xml_data: str) -> List[SearchResult]:
        """Parse PubMed XML response"""
        import xml.etree.ElementTree as ET

        results = []
        try:
            root = ET.fromstring(xml_data)

            for article in root.findall(".//PubmedArticle"):
                title_elem = article.find(".//ArticleTitle")
                abstract_elem = article.find(".//AbstractText")
                pmid_elem = article.find(".//PMID")
                pub_date_elem = article.find(".//PubDate/Year")

                if (title_elem is not None and title_elem.text and
                        pmid_elem is not None and pmid_elem.text):
                    pmid = pmid_elem.text
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                    published_date = None
                    if pub_date_elem is not None and pub_date_elem.text:
                        try:
                            year = int(pub_date_elem.text)
                            published_date = datetime(year, 1, 1, tzinfo=timezone.utc)
                        except (ValueError, TypeError):
                            pass

                    abstract = ""
                    if abstract_elem is not None and abstract_elem.text:
                        abstract = (abstract_elem.text[:300] + "...")

                    # Extract authors
                    authors = []
                    author_list = article.find(".//AuthorList")
                    if author_list is not None:
                        for author in author_list.findall(".//Author"):
                            lastname = author.find(".//LastName")
                            forename = author.find(".//ForeName")
                            if lastname is not None and lastname.text:
                                name = lastname.text
                                if forename is not None and forename.text:
                                    name = f"{forename.text} {name}"
                                authors.append(name)
                    
                    author_str = ", ".join(authors[:3]) if authors else None
                    if len(authors) > 3:
                        author_str += f" et al."

                    result = SearchResult(
                        title=title_elem.text,
                        url=url,
                        snippet=abstract,
                        source="pubmed",
                        published_date=published_date,
                        result_type="academic",
                        domain="pubmed.ncbi.nlm.nih.gov",
                        author=author_str,
                        publication_type="medical_research",
                        content_length=len(abstract_elem.text) if abstract_elem is not None and abstract_elem.text else 0
                    )
                    results.append(result)

        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {str(e)}")

        return results


class SemanticScholarAPI(BaseSearchAPI):
    """Semantic Scholar API for free academic paper search"""

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        super().__init__("", rate_limiter)  # No API key needed
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    async def search(
        self, query: str, config: SearchConfig
    ) -> List[SearchResult]:
        """Search Semantic Scholar for academic papers."""
        await self.rate_limiter.wait_if_needed()

        params = {
            "query": query,
            "limit": min(config.max_results, 100),
            "fields": (
                "title,abstract,authors,year,url,"
                "citationCount,influentialCitationCount"
            ),
        }

        try:
            async with self.session.get(
                self.base_url, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_semantic_scholar_results(data)
                else:
                    logger.error(
                        f"Semantic Scholar API error: {response.status}"
                    )
                    return []
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    def _parse_semantic_scholar_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Semantic Scholar API response"""
        results = []

        # Validate data structure
        if not isinstance(data, dict) or "data" not in data:
            logger.warning("Invalid Semantic Scholar response structure")
            return []

        papers = data.get("data", [])
        if not isinstance(papers, list):
            logger.warning("Semantic Scholar 'data' field is not a list")
            return []

        for paper in papers:
            # Skip invalid paper entries
            if not isinstance(paper, dict):
                continue

            # Build URL with validation
            paper_id = paper.get("paperId", "")
            url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""

            # Extract year for published date
            published_date = None
            if year := paper.get("year"):
                try:
                    if isinstance(year, int):
                        published_date = datetime(year, 1, 1, tzinfo=timezone.utc)
                except Exception:
                    pass

            # Build snippet from abstract with None guards
            snippet = paper.get("abstract") or ""
            if not snippet and paper.get("citationCount"):
                citation_count = paper.get("citationCount", 0)
                influential_count = paper.get("influentialCitationCount", 0)
                snippet = f"Citations: {citation_count}, Influential citations: {influential_count}"

            # Ensure snippet is string and handle None case
            if snippet is None:
                snippet = ""
            elif not isinstance(snippet, str):
                snippet = str(snippet)

            result = SearchResult(
                title=paper.get("title") or "",
                url=url,
                snippet=snippet[:300] + "..." if len(snippet) > 300 else snippet,
                source="semantic_scholar",
                published_date=published_date,
                result_type="academic",
                domain="semanticscholar.org",
                raw_data=paper
            )
            results.append(result)

        return results


class CrossRefAPI(BaseSearchAPI):
    """CrossRef API for DOI metadata and open access papers"""

    def __init__(self, email: Optional[str] = None, rate_limiter: Optional[RateLimiter] = None):
        super().__init__("", rate_limiter)  # No API key needed
        self.base_url = "https://api.crossref.org/works"
        self.email = email or os.getenv("CROSSREF_EMAIL", "research@fourhosts.ai")

    async def search(
        self, query: str, config: SearchConfig
    ) -> List[SearchResult]:
        """Search CrossRef for academic papers."""
        await self.rate_limiter.wait_if_needed()

        params = {
            "query": query,
            "rows": min(config.max_results, 100),
            "mailto": self.email,  # Polite request with contact
        }

        try:
            async with self.session.get(
                self.base_url, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_crossref_results(data)
                else:
                    logger.error(f"CrossRef API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return []

    def _parse_crossref_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse CrossRef API response"""
        results = []

        for item in data.get("message", {}).get("items", []):
            # Get best URL (prefer open access)
            url = item.get("URL", "")
            if "link" in item:
                for link in item["link"]:
                    if link.get("content-type") == "unspecified" and link.get("URL"):
                        url = link["URL"]
                        break

            # Parse published date
            published_date = None
            date_parts = item.get("published-print", {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                try:
                    year = date_parts[0][0]
                    month = date_parts[0][1] if len(date_parts[0]) > 1 else 1
                    day = date_parts[0][2] if len(date_parts[0]) > 2 else 1
                    published_date = datetime(year, month, day, tzinfo=timezone.utc)
                except (ValueError, TypeError, IndexError):
                    pass

            # Build snippet
            snippet = item.get("abstract", "")
            if not snippet:
                authors = item.get("author", [])
                if authors:
                    author_names = [f"{a.get('given', '')} {a.get('family', '')}" for a in authors[:3]]
                    snippet = f"Authors: {', '.join(author_names)}"

            result = SearchResult(
                title=" ".join(item.get("title", ["Untitled"])),
                url=url,
                snippet=snippet[:300] + "..." if len(snippet) > 300 else snippet,
                source="crossref",
                published_date=published_date,
                result_type="academic",
                domain="crossref.org",
                raw_data=item
            )
            results.append(result)

        return results


class SearchAPIManager:
    """Manages multiple search APIs with failover and aggregation"""

    def __init__(self, cache_manager=None):
        self.apis: Dict[str, BaseSearchAPI] = {}
        self.fallback_order = []
        self._initialized = False
        self.respectful_fetcher = RespectfulFetcher()
        self.relevance_filter = ContentRelevanceFilter()
        self.cache_manager = cache_manager
        # Deduplication with content hashing
        self.dedup_threshold = 0.85  # Similarity threshold for near-duplicates

    def add_api(self, name: str, api: BaseSearchAPI, is_primary: bool = False):
        """Add a search API"""
        self.apis[name] = api
        if is_primary:
            self.fallback_order.insert(0, name)
        else:
            self.fallback_order.append(name)

    async def search_with_api(self, name: str, query: str, config: SearchConfig) -> List[SearchResult]:
        """
        Thin wrapper to call a specific API by name for compatibility with orchestrator.
        Uses search_with_variations when available for broader coverage.
        """
        # Ensure APIs are initialized
        if not self._initialized:
            await self.initialize()

        api = self.apis.get(name)
        if not api:
            logger.warning(f"Requested search API '{name}' not found; falling back to generic search_with_fallback")
            return await self.search_with_fallback(query, config)

        try:
            if hasattr(api, 'search_with_variations'):
                results = await api.search_with_variations(query, config)
            else:
                results = await api.search(query, config)
            logger.info(f"{name}: fetched {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"search_with_api failed for {name}: {e}")
            return []

    async def search_all(
        self, query: str, config: SearchConfig, paradigm: Optional[str] = None
    ) -> Dict[str, List[SearchResult]]:
        """Search using all available APIs with enhanced deduplication"""
        # Ensure APIs are initialized
        if not self._initialized:
            await self.initialize()

        results = {}
        all_results_for_dedup = []

        tasks = []
        for name, api in self.apis.items():
            # Use search_with_variations if available
            if hasattr(api, 'search_with_variations'):
                task = asyncio.create_task(api.search_with_variations(query, config, paradigm))
            else:
                task = asyncio.create_task(api.search(query, config))
            tasks.append((name, task))

        for name, task in tasks:
            try:
                api_results = await task
                # Fetch full content for each result using the proper session for this API
                api_obj = self.apis.get(name)
                if api_obj and getattr(api_obj, "session", None):
                    content_tasks = [
                        self.respectful_fetcher.respectful_fetch(
                            api_obj.session, result.url
                        )
                        for result in api_results
                        if result.url and not result.content
                    ]
                    if content_tasks:
                        fetched_contents = await asyncio.gather(*content_tasks)
                        for result, content in zip(api_results, fetched_contents):
                            if content:
                                result.content = content

                # Enhance with snippets if content unavailable
                api_results = GoogleCustomSearchAPI.enhance_results_with_snippets(
                    api_results
                )

                # Apply global relevance filtering
                filtered_results = self.relevance_filter.filter_results(
                    api_results, query, min_relevance=0.25, config=config
                )
                
                # Collect for cross-API deduplication
                all_results_for_dedup.extend(filtered_results)
                
                results[name] = filtered_results
                logger.info(
                    f"{name}: {len(filtered_results)} results "
                    f"(from {len(api_results)} raw)"
                )
            except Exception as e:
                logger.error(f"{name} search failed: {str(e)}")
                results[name] = []

        # Apply cross-API deduplication
        deduplicated_results = self._deduplicate_results(all_results_for_dedup)
        
        # Redistribute deduplicated results back to their sources
        final_results = {name: [] for name in results.keys()}
        for result in deduplicated_results:
            source = result.source
            # Find which API this result came from
            for api_name, api_results in results.items():
                if any(r.url == result.url for r in api_results):
                    final_results[api_name].append(result)
                    break
        
        return final_results

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate results using content hashing and similarity detection"""
        if not results:
            return []
            
        deduplicated = []
        seen_hashes = set()
        seen_urls = set()
        similar_content_groups = defaultdict(list)
        
        for result in results:
            # Check exact URL match
            if result.url in seen_urls:
                continue
                
            # Check exact content hash match
            if result.content_hash and result.content_hash in seen_hashes:
                continue
                
            # Check for near-duplicates using SimHash (simplified version)
            is_duplicate = False
            result_tokens = set(self._tokenize_for_similarity(result))
            
            for existing in deduplicated:
                existing_tokens = set(self._tokenize_for_similarity(existing))
                
                # Calculate Jaccard similarity
                intersection = len(result_tokens & existing_tokens)
                union = len(result_tokens | existing_tokens)
                
                if union > 0:
                    similarity = intersection / union
                    if similarity >= self.dedup_threshold:
                        # Near-duplicate found - keep the one with higher relevance score
                        if result.relevance_score > existing.relevance_score:
                            deduplicated.remove(existing)
                            deduplicated.append(result)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_urls.add(result.url)
                if result.content_hash:
                    seen_hashes.add(result.content_hash)
        
        logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)}")
        return deduplicated

    def _tokenize_for_similarity(self, result: SearchResult) -> List[str]:
        """Tokenize result for similarity comparison"""
        text = f"{result.title} {result.snippet}".lower()
        # Simple tokenization - could be enhanced with shingles
        tokens = re.findall(r'\w+', text)
        # Return 3-grams for better similarity detection
        return [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]

    async def initialize(self):
        """Initialize all API sessions"""
        if self._initialized:
            return

        for name, api in self.apis.items():
            if hasattr(api, "__aenter__"):
                await api.__aenter__()
                logger.info(f"Initialized session for {name}")

        self._initialized = True

    async def cleanup(self):
        """Cleanup all API sessions"""
        if not self._initialized:
            return

        cleanup_tasks = []
        for name, api in self.apis.items():
            if hasattr(api, 'close_session'):
                cleanup_tasks.append(api.close_session())
            elif hasattr(api, "__aexit__"):
                cleanup_tasks.append(api.__aexit__(None, None, None))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                logger.info("All search API sessions cleaned up")
            except Exception as e:
                logger.error(f"Error during search API cleanup: {e}")

        self._initialized = False

    async def fetch_with_fallback(self, result: SearchResult, session: aiohttp.ClientSession) -> str:
        """Try multiple methods to get content ethically"""
        # 1. Try respectful fetch with proper headers and robots.txt compliance
        if result.url and not result.content:
            content = await self.respectful_fetcher.respectful_fetch(session, result.url)
            if content:
                result.content = content
                result.raw_data['content_source'] = 'direct_fetch'
                return content

        # 2. Check if we have an academic identifier and use appropriate API
        url_lower = result.url.lower() if result.url else ""

        # Try arXiv ID extraction
        if "arxiv.org" in url_lower:
            arxiv_id = self._extract_arxiv_id(result.url)
            if arxiv_id:
                # Use ArXiv API to get abstract
                logger.info(f"Fetching arXiv paper {arxiv_id} via API")
                result.raw_data['content_source'] = 'arxiv_api'
                # Abstract is already in snippet for arXiv results

        # Try DOI extraction for CrossRef
        doi = self._extract_doi(result.url) or self._extract_doi(result.snippet)
        if doi:
            logger.info(f"Found DOI {doi}, checking CrossRef for open access version")
            # CrossRef results already include abstracts in snippets
            result.raw_data['doi'] = doi
            result.raw_data['content_source'] = 'crossref_metadata'

        # 3. Use search snippet as fallback
        if result.snippet and not result.content:
            result.content = f"Summary from search results: {result.snippet}"
            result.raw_data['content_type'] = 'snippet_only'
            result.raw_data['content_source'] = 'search_api_snippet'
            return result.snippet

        return result.content or ""

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text"""
        if not text:
            return None
        import re
        doi_pattern = r'10\.\d{4,9}/[-._;()/:\w]+'
        match = re.search(doi_pattern, text)
        return match.group(0) if match else None

    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL"""
        if not url:
            return None
        import re
        # Match patterns like 2301.12345 or math.GT/0309136
        arxiv_pattern = r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7})'
        match = re.search(arxiv_pattern, url, re.IGNORECASE)
        return match.group(1) if match else None

    async def search_with_fallback(
        self, query: str, config: SearchConfig
    ) -> List[SearchResult]:
        """Search with automatic failover and performance optimizations"""
        # Ensure APIs are initialized
        if not self._initialized:
            await self.initialize()

        # Normalize query for better cache hits
        normalized_query = self._normalize_query_for_cache(query)
        cache_key = f"{normalized_query}:{config.max_results}:{config.language}"

        # Check cache first
        if self.cache_manager:
            try:
                cached = await self.cache_manager.get(cache_key)
                if cached:
                    logger.info(f"Cache HIT for normalized query: {normalized_query}")
                    # Convert cached data back to SearchResult objects
                    return [SearchResult(**item) for item in cached]
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")

        for api_name in self.fallback_order:
            if api_name in self.apis:
                try:
                    api = self.apis[api_name]
                    results = await api.search(query, config)

                    if results:
                        logger.info(
                            f"Used {api_name} for search, got {len(results)} results"
                        )
                        # Fetch full content for each result
                        if api.session:
                            for result in results:
                                if result.url and not result.content:
                                    content = await self.respectful_fetcher.respectful_fetch(
                                        api.session, result.url
                                    )
                                    if content:
                                        result.content = content
                        # Enhance with snippets if content unavailable
                        results = GoogleCustomSearchAPI.enhance_results_with_snippets(
                            results
                        )

                        # Cache results
                        if self.cache_manager:
                            try:
                                cache_data = [asdict(r) for r in results]
                                await self.cache_manager.set(
                                    cache_key, cache_data, ttl=3600
                                )
                            except Exception as e:
                                logger.warning(f"Cache set failed: {e}")
                        return results
                except Exception as e:
                    logger.warning(f"{api_name} failed, trying next: {str(e)}")
                    continue

        logger.error("All search APIs failed")
        return []

    def _normalize_query_for_cache(self, query: str) -> str:
        """Normalize query to improve cache hit rate"""
        import re
        # Remove extra whitespace, quotes, and punctuation
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        normalized = ' '.join(normalized.split())
        return normalized


def create_search_manager(cache_manager=None) -> SearchAPIManager:
    """Factory function to create search manager with all APIs"""
    manager = SearchAPIManager(cache_manager=cache_manager)

    # Get API keys from environment
    google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    brave_api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    pubmed_api_key = os.getenv("PUBMED_API_KEY")  # Optional

    if brave_api_key:
        brave_api = BraveSearchAPI(
            api_key=brave_api_key,
            rate_limiter=RateLimiter(calls_per_minute=100),
        )
        is_primary = not (google_api_key and google_engine_id)
        manager.add_api("brave", brave_api, is_primary=is_primary)

    # Add Google Custom Search (primary if available)
    if google_api_key and google_engine_id:
        google_api = GoogleCustomSearchAPI(
            api_key=google_api_key,
            search_engine_id=google_engine_id,
            rate_limiter=RateLimiter(calls_per_minute=100),
        )
        manager.add_api("google", google_api, is_primary=True)

    # Add ArXiv (free, for academic content)
    arxiv_api = ArxivAPI(rate_limiter=RateLimiter(calls_per_minute=30))
    manager.add_api("arxiv", arxiv_api)

    # Add PubMed (free, for medical content)
    pubmed_api = PubMedAPI(
        api_key=pubmed_api_key,
        rate_limiter=RateLimiter(calls_per_minute=10),  # Conservative rate
    )
    manager.add_api("pubmed", pubmed_api)

    # Add Semantic Scholar (free, excellent for academic papers)
    semantic_scholar_api = SemanticScholarAPI(
        rate_limiter=RateLimiter(calls_per_minute=10)  # More conservative rate limit
    )
    manager.add_api("semantic_scholar", semantic_scholar_api)

    # Add CrossRef (free, for DOI and open access content)
    crossref_api = CrossRefAPI(
        rate_limiter=RateLimiter(calls_per_minute=50)
    )
    manager.add_api("crossref", crossref_api)

    return manager


# Example usage
async def test_search_apis():
    """Test function for search APIs"""
    manager = create_search_manager()

    config = SearchConfig(max_results=10, language="en", region="us")

    query = "artificial intelligence ethics"

    print(f"Testing search for: {query}")
    print("=" * 50)

    # Test individual APIs
    results = await manager.search_with_fallback(query, config)

    for result in results[:5]:  # Show first 5
        print(f"Title: {result.title}")
        print(f"Source: {result.source}")
        print(f"URL: {result.url}")
        print(f"Snippet: {result.snippet[:100]}...")
        print("-" * 30)


if __name__ == "__main__":
    asyncio.run(test_search_apis())

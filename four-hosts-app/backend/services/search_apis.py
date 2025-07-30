"""
Search API integrations for Four Hosts Research Application
Implements Google Custom Search, Bing Search, and Academic APIs
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import hashlib
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

from bs4 import BeautifulSoup

import fitz  # PyMuPDF

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RespectfulFetcher:
    """Fetches content while respecting robots.txt and rate limits"""
    
    def __init__(self):
        self.robot_parsers = {}
        self.last_fetch = {}
        self.user_agent = "FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)"
        
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
        # Check robots.txt
        if not await self.can_fetch(url):
            logger.info(f"Robots.txt disallows fetching {url}")
            return None
            
        # Rate limit per domain (1 second between requests)
        domain = urlparse(url).netloc
        if domain in self.last_fetch:
            elapsed = time.time() - self.last_fetch[domain]
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
                
        self.last_fetch[domain] = time.time()
        
        # Fetch with proper headers
        return await fetch_and_parse_url(session, url)


async def fetch_and_parse_url(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch URL content with ethical headers and parse it to remove HTML tags or extract text from PDF"""
    # Use proper headers to identify ourselves
    headers = {
        'User-Agent': 'FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        async with session.get(url, timeout=10, headers=headers, allow_redirects=True) as response:
            if response.status == 200:
                content_type = response.headers.get("Content-Type", "").lower()
                if "application/pdf" in content_type:
                    pdf_content = await response.read()
                    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                        text = "".join(page.get_text() for page in doc)
                    return text
                else:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, "html.parser")
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    return soup.get_text(separator=" ", strip=True)
            elif response.status == 403:
                logger.info(f"Access denied (403) for {url} - will use search snippet instead")
                return ""
            elif response.status == 429:
                logger.warning(f"Rate limited (429) for {url} - consider reducing request frequency")
                return ""
            else:
                logger.warning(f"Failed to fetch {url}, status: {response.status}")
                return ""
    except aiohttp.ClientTimeout:
        logger.warning(f"Timeout fetching {url} - site may be slow or blocking automated requests")
        return ""
    except Exception as e:
        logger.error(f"Error fetching or parsing {url}: {e}")
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

    def __post_init__(self):
        if not self.domain and self.url:
            # Extract domain from URL
            from urllib.parse import urlparse

            self.domain = urlparse(self.url).netloc.lower()


@dataclass
class SearchConfig:
    """Configuration for search requests"""

    max_results: int = 50
    language: str = "en"
    region: str = "us"
    safe_search: str = "moderate"
    date_range: Optional[str] = None  # "d", "w", "m", "y" for day/week/month/year
    source_types: List[str] = field(default_factory=list)  # ["academic", "news", "web"]


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, calls_per_minute: int = 100):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = datetime.now()
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
    """Optimizes search queries for better relevance"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Common search modifiers that reduce relevance
        self.noise_terms = {
            'about', 'regarding', 'concerning', 'related', 'information',
            'details', 'explain', 'describe', 'what', 'how', 'why', 'when',
            'where', 'who', 'which', 'should', 'could', 'would', 'might'
        }
        
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for better search relevance"""
        # Tokenize and lowercase
        tokens = word_tokenize(query.lower())
        
        # Remove punctuation and stop words
        key_terms = [
            token for token in tokens
            if token not in string.punctuation
            and token not in self.stop_words
            and token not in self.noise_terms
            and len(token) > 2  # Skip very short words
        ]
        
        # Identify phrases (consecutive key terms)
        phrases = self._extract_phrases(query, key_terms)
        
        return phrases + key_terms
    
    def _extract_phrases(self, original_query: str, key_terms: List[str]) -> List[str]:
        """Extract meaningful phrases from the query"""
        phrases = []
        query_lower = original_query.lower()
        
        # Common phrase patterns
        phrase_patterns = [
            r'"([^"]+)"',  # Quoted phrases
            r'(\w+\s+\w+\s+\w+)',  # Three-word phrases
            r'(\w+\s+\w+)',  # Two-word phrases
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                # Check if the phrase contains at least one key term
                if any(term in match for term in key_terms):
                    phrases.append(match.strip())
        
        return list(set(phrases))  # Remove duplicates
    
    def optimize_query(self, query: str, paradigm: Optional[str] = None) -> str:
        """Optimize query for search engines while maintaining intent"""
        # Extract key terms
        key_terms = self.extract_key_terms(query)
        
        # If query is already short and focused, return as-is
        if len(query.split()) <= 5 and not any(term in query.lower() for term in self.noise_terms):
            return query
        
        # Build optimized query
        if key_terms:
            # For longer queries, focus on key terms
            if len(key_terms) > 5:
                # Take most important terms (beginning and end tend to be more important)
                important_terms = key_terms[:3] + key_terms[-2:]
                return ' '.join(important_terms)
            else:
                return ' '.join(key_terms)
        
        # Fallback to original if no optimization possible
        return query


class ContentRelevanceFilter:
    """Filters search results for relevance at retrieval time"""
    
    def __init__(self):
        self.query_optimizer = QueryOptimizer()
        
    def calculate_relevance_score(self, result: SearchResult, original_query: str, key_terms: List[str]) -> float:
        """Calculate relevance score for a search result"""
        score = 0.0
        
        # Extract text for analysis
        text_content = f"{result.title} {result.snippet}".lower()
        
        # 1. Key term frequency (40% weight)
        term_frequency_score = self._calculate_term_frequency(text_content, key_terms)
        score += term_frequency_score * 0.4
        
        # 2. Title relevance (30% weight)
        title_score = self._calculate_title_relevance(result.title.lower(), key_terms)
        score += title_score * 0.3
        
        # 3. Content freshness (10% weight)
        freshness_score = self._calculate_freshness_score(result.published_date)
        score += freshness_score * 0.1
        
        # 4. Source type bonus (10% weight)
        source_score = self._calculate_source_type_score(result)
        score += source_score * 0.1
        
        # 5. Exact phrase matching (10% weight)
        phrase_score = self._calculate_phrase_match_score(text_content, original_query)
        score += phrase_score * 0.1
        
        return min(1.0, score)
    
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
            
        days_old = (datetime.now() - published_date).days
        
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
    
    def _calculate_source_type_score(self, result: SearchResult) -> float:
        """Calculate source type score based on result type and domain"""
        # Academic sources get higher base score
        if result.result_type == "academic":
            return 0.9
        
        # Check for primary source indicators
        primary_indicators = [
            '.gov', '.edu', '.org',
            'official', 'foundation', 'institute',
            'journal', 'research', 'university'
        ]
        
        domain_lower = result.domain.lower()
        if any(indicator in domain_lower for indicator in primary_indicators):
            result.is_primary_source = True
            return 0.8
        
        # News sources
        if result.result_type == "news":
            return 0.7
        
        # Default web sources
        return 0.5
    
    def _calculate_phrase_match_score(self, text: str, original_query: str) -> float:
        """Calculate score for exact phrase matching"""
        # Look for quoted phrases in the original query
        quoted_phrases = re.findall(r'"([^"]+)"', original_query)
        
        if not quoted_phrases:
            return 0.5  # Neutral score if no quoted phrases
        
        matches = sum(1 for phrase in quoted_phrases if phrase.lower() in text)
        return matches / len(quoted_phrases)
    
    def filter_results(self, results: List[SearchResult], original_query: str, min_relevance: float = 0.3) -> List[SearchResult]:
        """Filter and rank results by relevance"""
        # Extract key terms once
        key_terms = self.query_optimizer.extract_key_terms(original_query)
        
        # Calculate relevance scores
        for result in results:
            result.relevance_score = self.calculate_relevance_score(result, original_query, key_terms)
        
        # Filter by minimum relevance
        filtered = [r for r in results if r.relevance_score >= min_relevance]
        
        # Sort by relevance score (descending)
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered)} with min relevance {min_relevance}")
        
        return filtered


class BaseSearchAPI:
    """Base class for all search APIs"""

    def __init__(self, api_key: str, rate_limiter: Optional[RateLimiter] = None):
        self.api_key = api_key
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None
        self.query_optimizer = QueryOptimizer()
        self.relevance_filter = ContentRelevanceFilter()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Override in subclasses"""
        raise NotImplementedError


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
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        await self.rate_limiter.wait_if_needed()

        # Validate parameters to prevent 400 errors
        if not query or not query.strip():
            logger.error("Empty query provided to Google search")
            return []
            
        # Optimize query for better relevance
        optimized_query = self.query_optimizer.optimize_query(query)
        logger.info(f"Optimized query: '{query}' -> '{optimized_query}'")
            
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
                    # Apply relevance filtering
                    filtered_results = self.relevance_filter.filter_results(results, query)
                    return filtered_results
                elif response.status == 429:
                    logger.warning("Google API rate limit exceeded")
                    raise Exception("Rate limit exceeded")
                else:
                    # Get response body for detailed error information
                    try:
                        # Try to parse as JSON first for structured error info
                        try:
                            error_data = await response.json()
                            logger.error(f"Google API error: {response.status} - {error_data}")
                            if 'error' in error_data:
                                error_details = error_data['error']
                                logger.error(f"Google API error details: {error_details.get('message', 'Unknown error')}")
                                logger.error(f"Error code: {error_details.get('code', 'Unknown')}")
                        except:
                            # If not JSON, get as text
                            error_body = await response.text()
                            logger.error(f"Google API error: {response.status} - {error_body}")
                    except:
                        logger.error(f"Google API error: {response.status} - Could not read response body")
                    
                    return []
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
            return []

    def _parse_google_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Google API response"""
        results = []

        for item in data.get("items", []):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="google_custom_search",
                domain=item.get("displayLink", ""),
                raw_data=item,
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
                    # Apply relevance filtering with higher threshold for academic
                    filtered_results = self.relevance_filter.filter_results(results, query, min_relevance=0.4)
                    return filtered_results
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

                if title_elem is not None and link_elem is not None:
                    published_date = None
                    if published_elem is not None:
                        try:
                            published_date = datetime.fromisoformat(
                                published_elem.text.replace("Z", "+00:00")
                            )
                        except:
                            pass

                    result = SearchResult(
                        title=title_elem.text.strip(),
                        url=link_elem.text,
                        snippet=(
                            (summary_elem.text[:300] + "...")
                            if summary_elem is not None
                            else ""
                        ),
                        source="arxiv",
                        published_date=published_date,
                        result_type="academic",
                        domain="arxiv.org",
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
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search using Brave Search API with comprehensive result parsing"""
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
                    remaining = response.headers["x-ratelimit-remaining"].split(", ")
                    if remaining and int(remaining[0]) < 5:
                        logger.warning(
                            f"Brave API rate limit low: {remaining[0]} requests remaining this second"
                        )

                if response.status == 200:
                    data = await response.json()
                    results = self._parse_brave_results(data)
                    # Apply relevance filtering
                    filtered_results = self.relevance_filter.filter_results(results, query)
                    return filtered_results
                elif response.status == 401:
                    logger.error("Brave API authentication failed - check API key")
                    return []
                elif response.status == 429:
                    retry_after = response.headers.get("retry-after", "60")
                    logger.warning(
                        f"Brave API rate limit exceeded. Retry after {retry_after} seconds"
                    )
                    raise Exception(f"Rate limit exceeded. Retry after {retry_after}s")
                else:
                    logger.error(f"Brave API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Brave search failed: {str(e)}")
            raise

    def _parse_brave_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Brave API response according to documented structure"""
        results = []

        # Parse web results
        web_results = data.get("web", {}).get("results", [])
        for item in web_results:
            # Parse age to published date if available
            published_date = None
            if "age" in item:
                try:
                    # Brave returns ISO format timestamps
                    published_date = datetime.fromisoformat(
                        item["age"].replace("Z", "+00:00")
                    )
                except:
                    pass

            # Extract domain from meta_url if available
            domain = ""
            if "meta_url" in item and "hostname" in item["meta_url"]:
                domain = item["meta_url"]["hostname"]
            elif "url" in item:
                from urllib.parse import urlparse

                domain = urlparse(item["url"]).netloc

            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source="brave_search",
                domain=domain,
                published_date=published_date,
                result_type="web",
                raw_data=item,
            )
            results.append(result)

        # Parse news results
        news_results = data.get("news", {}).get("results", [])
        for item in news_results:
            published_date = None
            if "age" in item:
                try:
                    published_date = datetime.fromisoformat(
                        item["age"].replace("Z", "+00:00")
                    )
                except:
                    pass

            # Extract source info
            source_info = item.get("source", "")
            domain = item.get("domain", "")
            if not domain and "url" in item:
                from urllib.parse import urlparse

                domain = urlparse(item["url"]).netloc

            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source="brave_search",
                domain=domain,
                result_type="news",
                published_date=published_date,
                bias_rating=source_info,  # Store source name as bias rating
                raw_data=item,
            )
            results.append(result)

        # Parse FAQ results
        faq_results = data.get("faq", {}).get("results", [])
        for item in faq_results:
            # FAQ results have question/answer format
            snippet = f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"

            result = SearchResult(
                title=item.get("title", item.get("question", "")),
                url=item.get("url", ""),
                snippet=snippet[:300],  # Limit snippet length
                source="brave_search",
                domain=item.get("meta_url", {}).get("hostname", ""),
                result_type="faq",
                raw_data=item,
            )
            results.append(result)

        # Parse discussion results
        discussions = data.get("discussions", {}).get("results", [])
        for item in discussions:
            if "data" in item:
                disc_data = item["data"]
                snippet = disc_data.get("question", "")
                if "top_comment" in disc_data:
                    snippet += f"\n{disc_data['top_comment']}"

                result = SearchResult(
                    title=disc_data.get("title", ""),
                    url=item.get("url", ""),
                    snippet=snippet[:300],
                    source="brave_search",
                    domain=disc_data.get("forum_name", ""),
                    result_type="discussion",
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
                    # Apply relevance filtering with higher threshold for medical
                    filtered_results = self.relevance_filter.filter_results(results, query, min_relevance=0.4)
                    return filtered_results
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

                if title_elem is not None and pmid_elem is not None:
                    pmid = pmid_elem.text
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                    published_date = None
                    if pub_date_elem is not None:
                        try:
                            year = int(pub_date_elem.text)
                            published_date = datetime(year, 1, 1)
                        except:
                            pass

                    abstract = ""
                    if abstract_elem is not None and abstract_elem.text:
                        abstract = abstract_elem.text[:300] + "..."

                    result = SearchResult(
                        title=title_elem.text,
                        url=url,
                        snippet=abstract,
                        source="pubmed",
                        published_date=published_date,
                        result_type="academic",
                        domain="pubmed.ncbi.nlm.nih.gov",
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
        
    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search Semantic Scholar for academic papers"""
        await self.rate_limiter.wait_if_needed()
        
        params = {
            "query": query,
            "limit": min(config.max_results, 100),
            "fields": "title,abstract,authors,year,url,citationCount,influentialCitationCount"
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_semantic_scholar_results(data)
                else:
                    logger.error(f"Semantic Scholar API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {str(e)}")
            return []
            
    def _parse_semantic_scholar_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Semantic Scholar API response"""
        results = []
        
        for paper in data.get("data", []):
            # Build URL
            paper_id = paper.get("paperId", "")
            url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""
            
            # Extract year for published date
            published_date = None
            if year := paper.get("year"):
                try:
                    published_date = datetime(year, 1, 1)
                except:
                    pass
                    
            # Build snippet from abstract
            snippet = paper.get("abstract", "")
            if not snippet and paper.get("citationCount"):
                snippet = f"Citations: {paper['citationCount']}, Influential citations: {paper.get('influentialCitationCount', 0)}"
                
            result = SearchResult(
                title=paper.get("title", ""),
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
        
    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search CrossRef for academic papers"""
        await self.rate_limiter.wait_if_needed()
        
        params = {
            "query": query,
            "rows": min(config.max_results, 100),
            "mailto": self.email  # Polite request with contact
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_crossref_results(data)
                else:
                    logger.error(f"CrossRef API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"CrossRef search failed: {str(e)}")
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
            if date_parts := item.get("published-print", {}).get("date-parts"):
                try:
                    if date_parts[0]:
                        year = date_parts[0][0]
                        month = date_parts[0][1] if len(date_parts[0]) > 1 else 1
                        day = date_parts[0][2] if len(date_parts[0]) > 2 else 1
                        published_date = datetime(year, month, day)
                except:
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

    def __init__(self):
        self.apis: Dict[str, BaseSearchAPI] = {}
        self.fallback_order = []
        self._initialized = False
        self.respectful_fetcher = RespectfulFetcher()
        self.relevance_filter = ContentRelevanceFilter()

    def add_api(self, name: str, api: BaseSearchAPI, is_primary: bool = False):
        """Add a search API"""
        self.apis[name] = api
        if is_primary:
            self.fallback_order.insert(0, name)
        else:
            self.fallback_order.append(name)

    async def search_all(
        self, query: str, config: SearchConfig
    ) -> Dict[str, List[SearchResult]]:
        """Search using all available APIs"""
        # Ensure APIs are initialized
        if not self._initialized:
            await self.initialize()

        results = {}

        tasks = []
        for name, api in self.apis.items():
            task = asyncio.create_task(api.search(query, config))
            tasks.append((name, task))

        for name, task in tasks:
            try:
                api_results = await task
                # Fetch full content for each result using respectful fetcher
                for result in api_results:
                    if result.url and not result.content:
                        content = await self.respectful_fetcher.respectful_fetch(
                            self.apis[name].session, result.url
                        )
                        if content:
                            result.content = content
                # Enhance with snippets if content unavailable
                api_results = GoogleCustomSearchAPI.enhance_results_with_snippets(api_results)
                
                # Apply global relevance filtering
                filtered_results = self.relevance_filter.filter_results(api_results, query, min_relevance=0.25)
                results[name] = filtered_results
                logger.info(f"{name}: {len(filtered_results)} results (from {len(api_results)} raw)")
            except Exception as e:
                logger.error(f"{name} search failed: {str(e)}")
                results[name] = []

        return results

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

        for name, api in self.apis.items():
            if hasattr(api, "__aexit__"):
                await api.__aexit__(None, None, None)
                logger.info(f"Cleaned up session for {name}")

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
        """Search with automatic failover"""
        # Ensure APIs are initialized
        if not self._initialized:
            await self.initialize()

        for api_name in self.fallback_order:
            if api_name in self.apis:
                try:
                    api = self.apis[api_name]
                    results = await api.search(query, config)

                    if results:
                        logger.info(
                            f"Used {api_name} for search, got {len(results)} results"
                        )
                        # Fetch full content for each result using respectful fetcher
                        for result in results:
                            if result.url and not result.content:
                                content = await self.respectful_fetcher.respectful_fetch(
                                    api.session, result.url
                                )
                                if content:
                                    result.content = content
                        # Enhance with snippets if content unavailable
                        results = GoogleCustomSearchAPI.enhance_results_with_snippets(results)
                        return results
                except Exception as e:
                    logger.warning(f"{api_name} failed, trying next: {str(e)}")
                    continue

        logger.error("All search APIs failed")
        return []


def create_search_manager() -> SearchAPIManager:
    """Factory function to create search manager with all APIs"""
    manager = SearchAPIManager()

    # Get API keys from environment
    google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    brave_api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    pubmed_api_key = os.getenv("PUBMED_API_KEY")  # Optional

    # Add Brave Search (can be primary if Google not available)
    if brave_api_key:
        brave_api = BraveSearchAPI(
            api_key=brave_api_key,
            rate_limiter=RateLimiter(
                calls_per_minute=100
            ),  # Brave allows up to 2000/month on free tier
        )
        # If no Google API is configured, make Brave primary
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
        rate_limiter=RateLimiter(calls_per_minute=100)  # They allow 100 req/sec
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

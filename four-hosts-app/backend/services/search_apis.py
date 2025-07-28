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
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    raw_data: Dict[str, Any] = field(default_factory=dict)

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


class BaseSearchAPI:
    """Base class for all search APIs"""

    def __init__(self, api_key: str, rate_limiter: Optional[RateLimiter] = None):
        self.api_key = api_key
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None

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

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(config.max_results, 10),
            "lr": f"lang_{config.language}",
            "gl": config.region,
            "safe": config.safe_search,
        }

        if config.date_range:
            params["dateRestrict"] = config.date_range

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_google_results(data)
                elif response.status == 429:
                    logger.warning("Google API rate limit exceeded")
                    raise Exception("Rate limit exceeded")
                else:
                    logger.error(f"Google API error: {response.status}")
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


class ArxivAPI(BaseSearchAPI):
    """ArXiv academic paper search API"""

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        super().__init__("", rate_limiter)  # ArXiv is free, no API key needed
        self.base_url = "http://export.arxiv.org/api/query"

    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search ArXiv for academic papers"""
        await self.rate_limiter.wait_if_needed()

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(config.max_results, 100),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    return self._parse_arxiv_results(xml_data)
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

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        # Core parameters
        params = {
            "q": query[:400],  # Max 400 chars per API docs
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
                    return self._parse_brave_results(data)
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

        # First, search for PMIDs
        search_params = {
            "db": "pubmed",
            "term": query,
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
                    return self._parse_pubmed_results(xml_data)
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


class SearchAPIManager:
    """Manages multiple search APIs with failover and aggregation"""

    def __init__(self):
        self.apis: Dict[str, BaseSearchAPI] = {}
        self.fallback_order = []
        self._initialized = False

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
                results[name] = api_results
                logger.info(f"{name}: {len(api_results)} results")
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

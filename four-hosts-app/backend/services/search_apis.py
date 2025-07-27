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
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(minutes=1)]
        
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
    
    def __init__(self, api_key: str, search_engine_id: str, 
                 rate_limiter: Optional[RateLimiter] = None):
        super().__init__(api_key, rate_limiter)
        self.search_engine_id = search_engine_id
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        await self.rate_limiter.wait_if_needed()
        
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(config.max_results, 10),  # Google max is 10 per request
            "lr": f"lang_{config.language}",
            "gl": config.region,
            "safe": config.safe_search
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
                raw_data=item
            )
            results.append(result)
        
        return results

class BingSearchAPI(BaseSearchAPI):
    """Bing Search API implementation"""
    
    def __init__(self, subscription_key: str, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(subscription_key, rate_limiter)
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search using Bing Search API"""
        await self.rate_limiter.wait_if_needed()
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(config.max_results, 50),  # Bing max is 50
            "mkt": f"{config.language}-{config.region}",
            "safeSearch": config.safe_search.capitalize()
        }
        
        if config.date_range:
            # Convert to Bing format
            freshness_map = {"d": "Day", "w": "Week", "m": "Month"}
            if config.date_range in freshness_map:
                params["freshness"] = freshness_map[config.date_range]
        
        try:
            async with self.session.get(self.base_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_bing_results(data)
                elif response.status == 429:
                    logger.warning("Bing API rate limit exceeded")
                    raise Exception("Rate limit exceeded")
                else:
                    logger.error(f"Bing API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Bing search failed: {str(e)}")
            return []
    
    def _parse_bing_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Bing API response"""
        results = []
        
        for item in data.get("webPages", {}).get("value", []):
            # Parse date if available
            published_date = None
            if "dateLastCrawled" in item:
                try:
                    published_date = datetime.fromisoformat(
                        item["dateLastCrawled"].replace("Z", "+00:00")
                    )
                except:
                    pass
            
            result = SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source="bing_search",
                published_date=published_date,
                raw_data=item
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
            "sortOrder": "descending"
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
                        snippet=(summary_elem.text[:300] + "...") if summary_elem is not None else "",
                        source="arxiv",
                        published_date=published_date,
                        result_type="academic",
                        domain="arxiv.org"
                    )
                    results.append(result)
        
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML: {str(e)}")
        
        return results

class PubMedAPI(BaseSearchAPI):
    """PubMed/NCBI API for medical and life science papers"""
    
    def __init__(self, api_key: Optional[str] = None, rate_limiter: Optional[RateLimiter] = None):
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
            "retmode": "json"
        }
        
        if self.api_key:
            search_params["api_key"] = self.api_key
        
        try:
            # Search for article IDs
            async with self.session.get(f"{self.base_url}/esearch.fcgi", params=search_params) as response:
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
                "retmode": "xml"
            }
            
            if self.api_key:
                fetch_params["api_key"] = self.api_key
            
            async with self.session.get(f"{self.base_url}/efetch.fcgi", params=fetch_params) as response:
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
                        domain="pubmed.ncbi.nlm.nih.gov"
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
    
    def add_api(self, name: str, api: BaseSearchAPI, is_primary: bool = False):
        """Add a search API"""
        self.apis[name] = api
        if is_primary:
            self.fallback_order.insert(0, name)
        else:
            self.fallback_order.append(name)
    
    async def search_all(self, query: str, config: SearchConfig) -> Dict[str, List[SearchResult]]:
        """Search using all available APIs"""
        results = {}
        
        tasks = []
        for name, api in self.apis.items():
            if hasattr(api, 'session') and api.session is None:
                async with api:
                    task = asyncio.create_task(api.search(query, config))
                    tasks.append((name, task))
            else:
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
    
    async def search_with_fallback(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search with automatic failover"""
        for api_name in self.fallback_order:
            if api_name in self.apis:
                try:
                    api = self.apis[api_name]
                    if hasattr(api, 'session') and api.session is None:
                        async with api:
                            results = await api.search(query, config)
                    else:
                        results = await api.search(query, config)
                    
                    if results:
                        logger.info(f"Used {api_name} for search, got {len(results)} results")
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
    bing_api_key = os.getenv("BING_SEARCH_API_KEY") 
    pubmed_api_key = os.getenv("PUBMED_API_KEY")  # Optional
    
    # Add Google Custom Search (primary)
    if google_api_key and google_engine_id:
        google_api = GoogleCustomSearchAPI(
            api_key=google_api_key,
            search_engine_id=google_engine_id,
            rate_limiter=RateLimiter(calls_per_minute=100)
        )
        manager.add_api("google", google_api, is_primary=True)
    
    # Add Bing Search (secondary)
    if bing_api_key:
        bing_api = BingSearchAPI(
            subscription_key=bing_api_key,
            rate_limiter=RateLimiter(calls_per_minute=50)
        )
        manager.add_api("bing", bing_api)
    
    # Add ArXiv (free, for academic content)
    arxiv_api = ArxivAPI(rate_limiter=RateLimiter(calls_per_minute=30))
    manager.add_api("arxiv", arxiv_api)
    
    # Add PubMed (free, for medical content)
    pubmed_api = PubMedAPI(
        api_key=pubmed_api_key,
        rate_limiter=RateLimiter(calls_per_minute=10)  # Conservative rate
    )
    manager.add_api("pubmed", pubmed_api)
    
    return manager

# Example usage
async def test_search_apis():
    """Test function for search APIs"""
    manager = create_search_manager()
    
    config = SearchConfig(
        max_results=10,
        language="en",
        region="us"
    )
    
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
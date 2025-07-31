"""
Search APIs V2 Adapter
Ensures search APIs work seamlessly with V2 orchestrator
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from services.search_apis import (
    SearchAPIManager, SearchResult, SearchConfig,
    GoogleSearchAPI, ArxivAPI, PubMedAPI, BraveSearchAPI
)
from services.text_compression import text_compressor, CompressionConfig
from models.context_models import SearchResultSchema

logger = logging.getLogger(__name__)


class SearchAPIManagerV2(SearchAPIManager):
    """Enhanced search API manager with V2 features"""
    
    def __init__(self):
        super().__init__()
        self.compression_enabled = True
        self.origin_tracking_enabled = True
    
    async def search_with_origin_tracking(
        self,
        query: str,
        config: SearchConfig,
        query_id: str,
        api_name: str
    ) -> List[SearchResultSchema]:
        """Execute search with origin tracking"""
        
        # Get the specific API
        api = self.apis.get(api_name)
        if not api or not api.is_available():
            logger.warning(f"API {api_name} not available")
            return []
        
        try:
            # Execute search
            results = await api.search(query, config)
            
            # Convert to V2 schema with origin tracking
            v2_results = []
            for idx, result in enumerate(results):
                # Apply dynamic compression if enabled
                if self.compression_enabled:
                    compressed_snippet = text_compressor.compress_text(
                        result.snippet,
                        CompressionConfig(max_tokens=100)
                    )
                else:
                    compressed_snippet = result.snippet[:300]  # Fallback
                
                # Create V2 schema
                v2_result = SearchResultSchema(
                    url=result.url,
                    title=result.title,
                    snippet=compressed_snippet,
                    full_content=getattr(result, 'full_content', None),
                    source_api=api_name,
                    credibility_score=getattr(result, 'credibility_score', 0.5),
                    origin_query=query,
                    origin_query_id=query_id,
                    metadata={
                        "result_index": idx,
                        "api_name": api_name,
                        "published_date": getattr(result, 'published_date', None),
                        "result_type": getattr(result, 'result_type', 'web')
                    }
                )
                v2_results.append(v2_result)
            
            return v2_results
            
        except Exception as e:
            logger.error(f"Search failed for {api_name}: {e}")
            return []
    
    async def batch_search_with_tracking(
        self,
        queries: List[Dict[str, Any]],
        paradigm: str,
        selected_apis: List[str]
    ) -> Dict[str, List[SearchResultSchema]]:
        """Execute batch searches with full tracking"""
        
        results_by_query = {}
        
        for query_data in queries:
            query = query_data.get("query", "")
            query_id = query_data.get("id", f"q_{hash(query)}")
            
            # Search across selected APIs
            query_results = []
            for api_name in selected_apis:
                if api_name in self.apis:
                    config = SearchConfig(
                        paradigm=paradigm,
                        max_results=10
                    )
                    
                    api_results = await self.search_with_origin_tracking(
                        query, config, query_id, api_name
                    )
                    query_results.extend(api_results)
            
            results_by_query[query_id] = query_results
        
        return results_by_query


class SearchResultAdapter:
    """Adapts between old SearchResult and new SearchResultSchema"""
    
    @staticmethod
    def to_v2_schema(
        old_result: SearchResult,
        origin_query: Optional[str] = None,
        origin_query_id: Optional[str] = None,
        source_api: Optional[str] = None
    ) -> SearchResultSchema:
        """Convert old SearchResult to V2 schema"""
        
        return SearchResultSchema(
            url=old_result.url,
            title=old_result.title,
            snippet=old_result.snippet,
            full_content=getattr(old_result, 'full_content', None),
            source_api=source_api or getattr(old_result, 'source', 'unknown'),
            credibility_score=getattr(old_result, 'credibility_score', 0.5),
            origin_query=origin_query,
            origin_query_id=origin_query_id,
            metadata={
                "domain": old_result.domain,
                "published_date": old_result.published_date,
                "result_type": old_result.result_type
            }
        )
    
    @staticmethod
    def from_v2_schema(v2_result: SearchResultSchema) -> SearchResult:
        """Convert V2 schema to old SearchResult"""
        
        metadata = v2_result.metadata or {}
        
        return SearchResult(
            title=v2_result.title,
            url=v2_result.url,
            snippet=v2_result.snippet,
            domain=metadata.get('domain', urlparse(v2_result.url).netloc),
            published_date=metadata.get('published_date'),
            result_type=metadata.get('result_type', 'web'),
            credibility_score=v2_result.credibility_score
        )
    
    @staticmethod
    def batch_to_v2(
        old_results: List[SearchResult],
        origin_query: str,
        source_api: str
    ) -> List[SearchResultSchema]:
        """Convert batch of old results to V2"""
        
        v2_results = []
        for idx, result in enumerate(old_results):
            v2_result = SearchResultAdapter.to_v2_schema(
                result,
                origin_query=origin_query,
                origin_query_id=f"q_{idx}",
                source_api=source_api
            )
            v2_results.append(v2_result)
        
        return v2_results


# Enhanced API implementations with compression
class GoogleSearchAPIV2(GoogleSearchAPI):
    """Google Search API with V2 enhancements"""
    
    def parse_google_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Google results with dynamic compression"""
        results = []
        
        for item in data.get("items", []):
            # Apply dynamic compression
            snippet = item.get("snippet", "")
            compressed_snippet = text_compressor.compress_text(
                snippet,
                CompressionConfig(max_tokens=100)
            )
            
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=compressed_snippet,
                domain=self.extract_domain(item.get("link", "")),
                published_date=None,
                result_type="web"
            )
            results.append(result)
        
        return results


class ArxivAPIV2(ArxivAPI):
    """ArXiv API with V2 enhancements"""
    
    def parse_arxiv_results(self, entries: List[Any]) -> List[SearchResult]:
        """Parse ArXiv results with compression"""
        results = []
        
        for entry in entries:
            # Compress abstract
            abstract = entry.summary or ""
            compressed_abstract = text_compressor.compress_text(
                abstract,
                CompressionConfig(max_tokens=150, preserve_start_ratio=0.4)
            )
            
            result = SearchResult(
                title=entry.title,
                url=entry.link,
                snippet=compressed_abstract,
                domain="arxiv.org",
                published_date=entry.published.strftime("%Y-%m-%d") if hasattr(entry, 'published') else None,
                result_type="academic"
            )
            
            # Add arxiv-specific metadata
            result.arxiv_id = entry.id.split('/')[-1]
            result.authors = [author.name for author in entry.authors][:3]  # Top 3 authors
            
            results.append(result)
        
        return results


# Global V2-enhanced manager
search_api_manager_v2 = SearchAPIManagerV2()


# Helper functions for migration
def upgrade_search_config(old_config: Union[Dict, SearchConfig]) -> SearchConfig:
    """Upgrade old search config to include V2 features"""
    
    if isinstance(old_config, dict):
        config = SearchConfig(**old_config)
    else:
        config = old_config
    
    # Add V2 features if not present
    if not hasattr(config, 'enable_compression'):
        config.enable_compression = True
    
    if not hasattr(config, 'track_origins'):
        config.track_origins = True
    
    return config


async def initialize_search_apis_v2():
    """Initialize V2-enhanced search APIs"""
    await search_api_manager_v2.initialize()
    logger.info("✓ Search APIs V2 initialized with compression and origin tracking")
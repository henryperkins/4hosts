"""
ResultAdapter - Consistent interface for accessing search results
Handles both dict and object formats to prevent AttributeError crashes
"""

from typing import Any, Dict, Optional, Union, List
import structlog
from utils.url_utils import extract_domain

from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


class ResultAdapter:
    """Adapter for consistent access to search result data regardless of format"""
    
    def __init__(self, result: Union[Dict[str, Any], Any]):
        self._result = result
        self._is_dict = isinstance(result, dict)
    
    @property
    def title(self) -> str:
        """Get title with fallback handling"""
        if self._is_dict:
            title = self._result.get('title', '')
        else:
            title = getattr(self._result, 'title', '')
        
        # Fallback to URL or default
        if not title:
            url = self.url
            if url:
                title = url.split('/')[-1] or url.split('/')[-2] or '(untitled)'
            else:
                title = '(untitled)'
        
        return title
    
    @property
    def url(self) -> str:
        """Get URL with fallback handling"""
        if self._is_dict:
            return self._result.get('url', '')
        else:
            return getattr(self._result, 'url', '')
    
    @property
    def snippet(self) -> str:
        """Get snippet with fallback handling"""
        if self._is_dict:
            return self._result.get('snippet', '')
        else:
            return getattr(self._result, 'snippet', '')
    
    @property
    def content(self) -> str:
        """Get content with fallback handling"""
        if self._is_dict:
            return self._result.get('content', '')
        else:
            return getattr(self._result, 'content', '')
    
    @property
    def source_api(self) -> str:
        """Get source API with fallback handling"""
        if self._is_dict:
            return self._result.get('source_api', self._result.get('search_api', 'unknown'))
        else:
            return getattr(self._result, 'source_api', getattr(self._result, 'search_api', 'unknown'))
    
    @property
    def credibility_score(self) -> Optional[float]:
        """Get credibility score with fallback handling"""
        if self._is_dict:
            return self._result.get('credibility_score')
        else:
            return getattr(self._result, 'credibility_score', None)
    
    @property
    def credibility_explanation(self) -> str:
        """Get credibility explanation with fallback handling"""
        if self._is_dict:
            return self._result.get('credibility_explanation', '')
        else:
            return getattr(self._result, 'credibility_explanation', '')
    
    @property
    def domain(self) -> str:
        """Get domain with fallback handling"""
        if self._is_dict:
            domain = self._result.get('domain', '')
        else:
            domain = getattr(self._result, 'domain', '')
        
        # Extract domain from URL if not present
        if not domain and self.url:
            domain = extract_domain(self.url)
        
        return domain
    
    @property
    def origin_query(self) -> str:
        """Get origin query with fallback handling"""
        if self._is_dict:
            return self._result.get('origin_query', '')
        else:
            return getattr(self._result, 'origin_query', '')
    
    @property
    def origin_query_id(self) -> str:
        """Get origin query ID with fallback handling"""
        if self._is_dict:
            return self._result.get('origin_query_id', '')
        else:
            return getattr(self._result, 'origin_query_id', '')
    
    @property
    def paradigm_alignment(self) -> str:
        """Get paradigm alignment with fallback handling"""
        if self._is_dict:
            return self._result.get('paradigm_alignment', '')
        else:
            return getattr(self._result, 'paradigm_alignment', '')

    @property
    def query_variant(self) -> str:
        """Expose planner stage:label metadata when available."""
        if self._is_dict:
            return str(self._result.get('query_variant', '') or '')
        return str(getattr(self._result, 'query_variant', '') or '')

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata with fallback handling"""
        if self._is_dict:
            return self._result.get('metadata', {})
        else:
            return getattr(self._result, 'metadata', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get arbitrary field with fallback handling"""
        if self._is_dict:
            return self._result.get(key, default)
        else:
            return getattr(self._result, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        if self._is_dict:
            return self._result.copy()
        
        # Convert object to dict
        result_dict = {}
        if hasattr(self._result, 'to_dict'):
            result_dict = self._result.to_dict()
        elif hasattr(self._result, '__dict__'):
            result_dict = self._result.__dict__.copy()
        else:
            # Fallback - try to extract common fields
            for field in ['title', 'url', 'snippet', 'content', 'source_api', 'credibility_score', 'domain']:
                if hasattr(self._result, field):
                    result_dict[field] = getattr(self._result, field)
        
        return result_dict
    
    def has_required_fields(self) -> bool:
        """Check if result has minimum required fields"""
        return bool(self.url)  # URL is the minimum requirement
    
    def __repr__(self) -> str:
        return f"ResultAdapter(title='{self.title[:50]}...', url='{self.url}')"


class ResultListAdapter:
    """Adapter for lists of search results with batch operations"""
    
    def __init__(self, results: List[Union[Dict[str, Any], Any]]):
        self._results = [ResultAdapter(result) for result in results]
    
    def __iter__(self):
        return iter(self._results)
    
    def __len__(self):
        return len(self._results)
    
    def __getitem__(self, index):
        return self._results[index]
    
    def get_valid_results(self) -> List[ResultAdapter]:
        """Get only results with required fields"""
        return [result for result in self._results if result.has_required_fields()]
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all to dict format"""
        return [result.to_dict() for result in self._results]
    
    def filter_by_credibility(self, min_score: float = 0.5) -> List[ResultAdapter]:
        """Filter results by minimum credibility score"""
        return [
            result for result in self._results 
            if result.credibility_score is not None and result.credibility_score >= min_score
        ]
    
    def group_by_domain(self) -> Dict[str, List[ResultAdapter]]:
        """Group results by domain"""
        groups = {}
        for result in self._results:
            domain = result.domain
            if domain:
                if domain not in groups:
                    groups[domain] = []
                groups[domain].append(result)
        return groups
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the result set"""
        valid_results = self.get_valid_results()
        
        return {
            "total_results": len(self._results),
            "valid_results": len(valid_results),
            "unique_domains": len(set(r.domain for r in valid_results if r.domain)),
            "source_apis": list(set(r.source_api for r in valid_results)),
            "avg_credibility": sum(
                r.credibility_score for r in valid_results 
                if r.credibility_score is not None
            ) / max(len([r for r in valid_results if r.credibility_score is not None]), 1)
        }


def adapt_results(results: Union[List[Union[Dict[str, Any], Any]], Union[Dict[str, Any], Any]]) -> Union[ResultListAdapter, ResultAdapter]:
    """Convenience function to create appropriate adapter"""
    if isinstance(results, list):
        return ResultListAdapter(results)
    else:
        return ResultAdapter(results)


# Utility functions for common operations
def extract_links_safe(results: Union[List[Union[Dict[str, Any], Any]], Union[Dict[str, Any], Any]]) -> List[str]:
    """Safely extract links from results regardless of format"""
    adapter = adapt_results(results)
    
    if isinstance(adapter, ResultListAdapter):
        return [result.url for result in adapter.get_valid_results()]
    else:
        return [adapter.url] if adapter.has_required_fields() else []


def extract_snippets_safe(results: Union[List[Union[Dict[str, Any], Any]], Union[Dict[str, Any], Any]]) -> List[str]:
    """Safely extract snippets from results regardless of format"""
    adapter = adapt_results(results)
    
    if isinstance(adapter, ResultListAdapter):
        return [result.snippet for result in adapter.get_valid_results() if result.snippet]
    else:
        return [adapter.snippet] if adapter.has_required_fields() and adapter.snippet else []


def format_results_for_display(results: Union[List[Union[Dict[str, Any], Any]], Union[Dict[str, Any], Any]]) -> str:
    """Format results for display safely"""
    adapter = adapt_results(results)
    
    if isinstance(adapter, ResultListAdapter):
        formatted_lines = []
        for i, result in enumerate(adapter.get_valid_results()[:10], 1):  # Limit to top 10
            formatted_lines.append(
                f"{i}. {result.title}\n   {result.url}\n   {result.snippet[:200]}...\n"
            )
        return "\n".join(formatted_lines)
    else:
        if adapter.has_required_fields():
            return f"1. {adapter.title}\n   {adapter.url}\n   {adapter.snippet[:200]}...\n"
        else:
            return "No valid results found.\n"

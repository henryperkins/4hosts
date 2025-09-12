"""
Text compression utilities for the Four Hosts application
Provides text and query compression functionality
"""

import os
import re
from typing import Optional, List, Dict


class TextCompressor:
    """Simple text compression for search results and content"""
    
    def __init__(self):
        self.min_length = 100
        # Default soft cap; can be overridden per-call
        self.max_length = int(os.getenv("TEXT_COMPRESSION_MAX_LENGTH", "5000") or 5000)
    
    def compress(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Compress text to fit within token limits
        
        Args:
            text: Input text to compress
            max_length: Maximum length (uses default if not specified)
            
        Returns:
            Compressed text
        """
        if not text:
            return ""
            
        max_len = max_length or self.max_length
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If already short enough, return as-is
        if len(text) <= max_len:
            return text
            
        # Truncate and add ellipsis
        return text[:max_len-3] + "..."
    
    def compress_search_result(self, title: str, snippet: str, max_length: int = 500) -> str:
        """
        Compress a search result to essential information
        
        Args:
            title: Result title
            snippet: Result snippet/description
            max_length: Maximum total length
            
        Returns:
            Compressed result text
        """
        # Ensure title isn't too long
        if len(title) > 100:
            title = title[:97] + "..."
            
        # Calculate remaining space for snippet
        remaining = max_length - len(title) - 3  # 3 for separator
        
        if remaining > 50:
            snippet = self.compress(snippet, remaining)
            return f"{title} - {snippet}"
        else:
            return title


class QueryCompressor:
    """Query compression for API rate limit optimization"""
    
    def __init__(self):
        self.max_query_length = 200
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'including', 'until', 'against', 'among', 'throughout', 'despite',
            'towards', 'upon', 'concerning', 'regarding', 'since', 'before',
            'after', 'above', 'below', 'between', 'under', 'over'
        }
    
    def compress(self, query: str, preserve_keywords: bool = True) -> str:
        """
        Compress a search query while preserving important keywords
        
        Args:
            query: Input query
            preserve_keywords: Whether to preserve important keywords
            
        Returns:
            Compressed query
        """
        if not query:
            return ""
            
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # If already short enough, return as-is
        if len(query) <= self.max_query_length:
            return query
            
        # Remove stop words if preserving keywords
        if preserve_keywords:
            words = query.split()
            filtered_words = [w for w in words if w.lower() not in self.stop_words or len(w) > 4]
            query = ' '.join(filtered_words)
            
        # If still too long, truncate
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length-3] + "..."
            
        return query
    
    def extract_keywords(self, query: str) -> list:
        """
        Extract important keywords from a query
        
        Args:
            query: Input query
            
        Returns:
            List of keywords
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
                
        return unique_keywords


def compress_search_results(
    results: List[dict],
    total_token_budget: int = 3000,
    weights: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    Compress search result dicts into concise entries within a rough token budget.

    If `weights` is provided, it should map each result's URL to a non‑negative
    weight; budgets are allocated proportionally (with per‑item caps). Fallbacks
    to equal allocation when missing.
    """
    compressor = text_compressor
    if not results:
        return []
    # Compute per-item budgets
    budgets: Dict[str, int] = {}
    if weights:
        # Normalize weights
        wsum = sum(v for v in weights.values() if isinstance(v, (int, float)) and v > 0)
        if wsum <= 0:
            weights = None
    if weights:
        for r in results:
            u = r.get("url") or ""
            w = float(weights.get(u, 0.0) or 0.0)
            share = (w / wsum) if wsum else 0.0  # type: ignore[name-defined]
            alloc = int(max(150, min(800, total_token_budget * share)))
            budgets[u] = alloc if alloc > 0 else 150
    else:
        per_item = max(200, int(total_token_budget / max(len(results), 1)))
        per_item = min(per_item, 800)
    out: List[dict] = []
    for r in results:
        title = r.get("title") or ""
        snippet = r.get("snippet") or ""
        content = r.get("content") or ""
        u = r.get("url") or ""
        budget = budgets.get(u) if budgets else None
        if budget is None:
            budget = per_item  # type: ignore[name-defined]
        summary = compressor.compress_search_result(title, snippet, max_length=budget)
        short_title = title if len(title) <= 120 else title[:117] + "..."
        new_r = dict(r)
        new_r["title"] = short_title
        new_r["snippet"] = summary
        if content:
            new_r["content"] = compressor.compress(content, max_length=int(budget * 2))
        out.append(new_r)
    return out

# Create singleton instances
text_compressor = TextCompressor()
query_compressor = QueryCompressor()

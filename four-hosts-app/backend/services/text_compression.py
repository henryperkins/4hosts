"""
Text compression utilities for the Four Hosts application
Provides text and query compression functionality
"""

import re
from typing import Optional


class TextCompressor:
    """Simple text compression for search results and content"""
    
    def __init__(self):
        self.min_length = 100
        self.max_length = 2000
    
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


# Create singleton instances
text_compressor = TextCompressor()
query_compressor = QueryCompressor()
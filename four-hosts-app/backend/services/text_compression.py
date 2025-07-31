"""
Dynamic Text Compression and Truncation Service
Replaces hard-coded truncation with intelligent, token-aware compression
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import tiktoken
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Initialize NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except Exception as e:
    logger.warning(f"NLTK initialization failed: {e}")
    NLTK_AVAILABLE = False


@dataclass
class CompressionConfig:
    """Configuration for text compression"""
    max_tokens: int = 500
    preserve_start_ratio: float = 0.3  # Preserve 30% from start
    preserve_end_ratio: float = 0.2    # Preserve 20% from end
    min_sentence_tokens: int = 5       # Minimum tokens per sentence
    enable_summarization: bool = True
    target_compression_ratio: float = 0.7


class TextCompressor:
    """Intelligent text compression with token awareness"""
    
    def __init__(self, model_name: str = "gpt-4"):
        # Initialize tokenizer
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        
        # Stop words for importance scoring
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    def compress_text(
        self, 
        text: str, 
        config: CompressionConfig = CompressionConfig()
    ) -> str:
        """Compress text intelligently based on token budget"""
        if not text:
            return text
        
        # Check if compression needed
        token_count = self.count_tokens(text)
        if token_count <= config.max_tokens:
            return text
        
        # Try different compression strategies
        compressed = self._sentence_compression(text, config)
        
        # If still too long, use hard truncation with ellipsis
        if self.count_tokens(compressed) > config.max_tokens:
            compressed = self._hard_truncate(compressed, config.max_tokens)
        
        return compressed
    
    def _sentence_compression(self, text: str, config: CompressionConfig) -> str:
        """Compress by selecting important sentences"""
        if not NLTK_AVAILABLE:
            return self._simple_truncate(text, config.max_tokens)
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 2:
                return self._simple_truncate(text, config.max_tokens)
            
            # Score sentences
            sentence_scores = []
            for i, sent in enumerate(sentences):
                score = self._score_sentence(sent, i, len(sentences))
                sentence_scores.append((score, i, sent))
            
            # Sort by score
            sentence_scores.sort(reverse=True)
            
            # Select sentences within token budget
            selected_indices = set()
            total_tokens = 0
            
            # Always include first and last sentences if possible
            if sentences:
                first_tokens = self.count_tokens(sentences[0])
                last_tokens = self.count_tokens(sentences[-1])
                
                if first_tokens + last_tokens < config.max_tokens * 0.5:
                    selected_indices.add(0)
                    selected_indices.add(len(sentences) - 1)
                    total_tokens = first_tokens + last_tokens
            
            # Add high-scoring sentences
            for score, idx, sent in sentence_scores:
                if idx in selected_indices:
                    continue
                
                sent_tokens = self.count_tokens(sent)
                if total_tokens + sent_tokens <= config.max_tokens * 0.9:
                    selected_indices.add(idx)
                    total_tokens += sent_tokens
            
            # Reconstruct text maintaining order
            if selected_indices:
                compressed_sentences = []
                for i in sorted(selected_indices):
                    compressed_sentences.append(sentences[i])
                    # Add ellipsis between non-consecutive sentences
                    if i + 1 not in selected_indices and i < len(sentences) - 1:
                        compressed_sentences.append("...")
                
                return " ".join(compressed_sentences)
            
        except Exception as e:
            logger.error(f"Sentence compression failed: {e}")
        
        return self._simple_truncate(text, config.max_tokens)
    
    def _score_sentence(self, sentence: str, position: int, total_sentences: int) -> float:
        """Score sentence importance"""
        score = 0.0
        
        # Position scoring - prefer beginning and end
        if position == 0:
            score += 2.0
        elif position == total_sentences - 1:
            score += 1.5
        elif position < 3:
            score += 1.0
        
        # Length scoring - prefer medium length sentences
        word_count = len(word_tokenize(sentence))
        if 10 <= word_count <= 30:
            score += 1.0
        elif word_count > 50:
            score -= 0.5
        
        # Keyword scoring
        words = word_tokenize(sentence.lower())
        non_stop_words = [w for w in words if w not in self.stop_words and w.isalpha()]
        
        # Check for important patterns
        if self.url_pattern.search(sentence):
            score += 1.0
        if any(kw in sentence.lower() for kw in ['important', 'critical', 'key', 'main']):
            score += 1.5
        if '?' in sentence:  # Questions are often important
            score += 0.5
        
        # Unique word ratio
        if words:
            unique_ratio = len(set(non_stop_words)) / len(words)
            score += unique_ratio
        
        return score
    
    def _simple_truncate(self, text: str, max_tokens: int) -> str:
        """Simple truncation preserving start and end"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Calculate how many tokens to keep from start and end
        start_tokens = int(max_tokens * 0.6)
        end_tokens = max_tokens - start_tokens - 10  # Reserve space for ellipsis
        
        # Decode start and end portions
        start_text = self.encoder.decode(tokens[:start_tokens])
        end_text = self.encoder.decode(tokens[-end_tokens:]) if end_tokens > 0 else ""
        
        # Find clean break points
        start_text = self._clean_break(start_text, at_end=True)
        end_text = self._clean_break(end_text, at_end=False)
        
        return f"{start_text} ... {end_text}" if end_text else f"{start_text} ..."
    
    def _hard_truncate(self, text: str, max_tokens: int) -> str:
        """Hard truncation when other methods fail"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens-10]  # Reserve space for ellipsis
        truncated_text = self.encoder.decode(truncated_tokens)
        
        # Clean break at word boundary
        truncated_text = self._clean_break(truncated_text, at_end=True)
        
        return f"{truncated_text}..."
    
    def _clean_break(self, text: str, at_end: bool = True) -> str:
        """Clean text break at word boundary"""
        if not text:
            return text
        
        if at_end:
            # Break at last complete word
            match = re.search(r'\s\S*$', text)
            if match:
                return text[:match.start()]
        else:
            # Break at first complete word
            match = re.search(r'^\S*\s', text)
            if match:
                return text[match.end():]
        
        return text
    
    def compress_search_results(
        self, 
        results: List[Dict[str, Any]], 
        total_token_budget: int = 3000
    ) -> List[Dict[str, Any]]:
        """Compress a list of search results within token budget"""
        if not results:
            return results
        
        # Calculate per-result budget
        per_result_budget = total_token_budget // len(results)
        
        # Ensure minimum viable budget
        per_result_budget = max(per_result_budget, 100)
        
        compressed_results = []
        for result in results:
            compressed_result = result.copy()
            
            # Compress title if needed
            if 'title' in compressed_result:
                title_tokens = self.count_tokens(compressed_result['title'])
                if title_tokens > 50:
                    compressed_result['title'] = self._hard_truncate(
                        compressed_result['title'], 50
                    )
            
            # Compress snippet/content
            if 'snippet' in compressed_result:
                snippet_budget = per_result_budget - 50  # Reserve for title and metadata
                compressed_result['snippet'] = self.compress_text(
                    compressed_result['snippet'],
                    CompressionConfig(max_tokens=snippet_budget)
                )
            
            compressed_results.append(compressed_result)
        
        return compressed_results
    
    def compress_for_websocket(
        self, 
        data: Dict[str, Any], 
        max_size_bytes: int = 65536
    ) -> Dict[str, Any]:
        """Compress data for WebSocket transmission"""
        import json
        
        # First pass - try without compression
        json_str = json.dumps(data)
        if len(json_str.encode('utf-8')) <= max_size_bytes:
            return data
        
        # Compress large text fields
        compressed_data = data.copy()
        
        # Find and compress text fields
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 1000:
                # Aggressive compression for large fields
                tokens_budget = max(100, min(500, len(value) // 10))
                compressed_data[key] = self.compress_text(
                    value,
                    CompressionConfig(max_tokens=tokens_budget)
                )
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Compress lists of results
                if 'snippet' in value[0] or 'content' in value[0]:
                    compressed_data[key] = self.compress_search_results(value, 1000)
        
        return compressed_data


class QueryCompressor:
    """Compress queries for efficient API usage"""
    
    def __init__(self):
        self.compressor = TextCompressor()
    
    def optimize_query_batch(
        self, 
        queries: List[str], 
        max_queries: int = 10,
        paradigm: Optional[str] = None
    ) -> List[str]:
        """Optimize a batch of queries based on paradigm and limits"""
        if not queries:
            return queries
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        # Score queries by relevance
        scored_queries = []
        for i, query in enumerate(unique_queries):
            score = self._score_query(query, i, paradigm)
            scored_queries.append((score, query))
        
        # Sort by score and take top N
        scored_queries.sort(reverse=True)
        
        # Adjust max queries based on paradigm
        if paradigm:
            paradigm_limits = {
                "analytical": 15,  # Bernard benefits from more queries
                "strategic": 12,   # Maeve needs diverse sources
                "revolutionary": 8,  # Dolores focuses on impactful queries
                "devotion": 6      # Teddy prefers quality over quantity
            }
            max_queries = min(max_queries, paradigm_limits.get(paradigm, 10))
        
        return [query for _, query in scored_queries[:max_queries]]
    
    def _score_query(self, query: str, position: int, paradigm: Optional[str]) -> float:
        """Score query importance"""
        score = 10.0 - position * 0.5  # Position penalty
        
        # Length scoring - prefer medium length
        word_count = len(query.split())
        if 3 <= word_count <= 8:
            score += 2.0
        elif word_count > 15:
            score -= 1.0
        
        # Paradigm-specific scoring
        if paradigm == "analytical" and any(kw in query.lower() for kw in ['data', 'study', 'research']):
            score += 3.0
        elif paradigm == "strategic" and any(kw in query.lower() for kw in ['strategy', 'business', 'market']):
            score += 3.0
        
        return score


# Global instances
text_compressor = TextCompressor()
query_compressor = QueryCompressor()
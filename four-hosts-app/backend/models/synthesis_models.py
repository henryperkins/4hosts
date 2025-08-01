"""
Models for answer synthesis and generation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class SynthesisContext:
    """Context for answer synthesis"""
    query: str
    paradigm: str
    search_results: List[Dict[str, Any]]
    context_engineering: Dict[str, Any]
    max_length: int = 2000
    include_citations: bool = True
    tone: str = "professional"
    metadata: Dict[str, Any] = field(default_factory=dict)
    deep_research_content: Optional[str] = None
    classification_result: Optional[Any] = None


@dataclass
class GeneratedAnswer:
    """Generated answer with metadata"""
    content: str
    paradigm: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    synthesis_quality: Optional[float] = None
    secondary_perspective: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "content": self.content,
            "paradigm": self.paradigm,
            "sources": self.sources,
            "metadata": self.metadata
        }
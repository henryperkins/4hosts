"""
Models for answer synthesis and generation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from core.config import SYNTHESIS_MAX_LENGTH_DEFAULT
try:
    # Typed reference to canonical EvidenceBundle
    from .context_models import EvidenceBundle  # type: ignore
except Exception:  # Fallback type alias for runtime resilience
    EvidenceBundle = Any  # type: ignore


@dataclass
class SynthesisContext:
    """Context for answer synthesis"""
    query: str
    paradigm: str
    search_results: List[Dict[str, Any]]
    context_engineering: Dict[str, Any]
    max_length: int = SYNTHESIS_MAX_LENGTH_DEFAULT
    include_citations: bool = True
    tone: str = "professional"
    metadata: Dict[str, Any] = field(default_factory=dict)
    deep_research_content: Optional[str] = None
    classification_result: Optional[Any] = None
    # New: prioritized quotes selected from top sources
    evidence_quotes: List[Dict[str, Any]] = field(default_factory=list)
    # Canonical: unified evidence payload
    evidence_bundle: Optional[EvidenceBundle] = None


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

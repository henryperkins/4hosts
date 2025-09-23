"""grounding.py – Lightweight grounding coverage helper

The heuristic measures how well the generated answer references evidence
by computing the ratio of answer sentences that include at least one
citation marker (URL, ID, or title snippet).  The computation is kept
simple and dependency-free to avoid adding heavy NLP libraries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Pre-compiled regex for sentence segmentation (period / exclamation / question)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extract_marker(citation: Dict[str, Any]) -> str:
    """Return a minimal string that identifies a citation entry."""
    for key in ("id", "url", "title"):
        value = citation.get(key)  # type: ignore[index]
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def compute_grounding_coverage(
    content: str, citations: List[Dict[str, Any]] | None = None
) -> Tuple[float, int, int]:
    """Compute grounding coverage.

    Parameters
    ----------
    content : str
        The answer content to analyse.
    citations : list[dict] | None
        List of citation objects. Each object should expose at least one of
        the keys: ``id``, ``url``, ``title``.

    Returns
    -------
    tuple[float, int, int]
        ``(coverage_ratio, total_sentences, cited_sentences)`` where
        ``coverage_ratio`` is a float between 0 and 1 inclusive.
    """

    citations = citations or []

    # Split content into sentences (very naive)
    sentences = [s.strip() for s in _SENT_SPLIT.split(content or "") if s.strip()]

    if not sentences:
        return 0.0, 0, 0

    markers = {_extract_marker(c) for c in citations if c}
    markers.discard("")

    cited_indices: set[int] = set()
    if markers:
        for idx, sent in enumerate(sentences):
            for marker in markers:
                if marker in sent:
                    cited_indices.add(idx)
                    break

    coverage = len(cited_indices) / len(sentences)
    return coverage, len(sentences), len(cited_indices)


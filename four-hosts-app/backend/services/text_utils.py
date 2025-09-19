from __future__ import annotations

import os
import re
from typing import List, Set

import nltk

def _ensure_nltk_ready() -> bool:
    """Return True if punkt/stopwords/wordnet are available (download if env allows)."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        if os.getenv("SEARCH_ALLOW_NLTK_DOWNLOADS") == "1":
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("stopwords", quiet=True)
                nltk.download("wordnet", quiet=True)
                return True
            except Exception:
                return False
        return False

_use_nltk = _ensure_nltk_ready()
if _use_nltk:
    try:
        STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
    except Exception:
        STOP_WORDS = {
            "the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to", "with", "by",
            "is", "are", "was", "were", "be", "as", "at", "it", "this", "that", "from",
        }
else:
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to", "with", "by",
        "is", "are", "was", "were", "be", "as", "at", "it", "this", "that", "from",
    }

def tokenize(text: str, *, lower=True) -> List[str]:
    text = text.lower() if lower else text
    tokens = re.findall(r"\w+", text)
    return [t for t in tokens if t not in STOP_WORDS]

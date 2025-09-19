from __future__ import annotations

from typing import Iterable


def canon_query(text: str) -> str:
    return " ".join((text or "").strip().split())


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set((a or "").lower().split())
    set_b = set((b or "").lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / float(len(set_a | set_b) or 1)


def is_duplicate(candidate: str, seen: Iterable[str], threshold: float) -> bool:
    canon = canon_query(candidate)
    for existing in seen:
        if existing == canon or jaccard_similarity(existing, canon) >= threshold:
            return True
    return False

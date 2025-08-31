"""
Token budget utilities for enforcing context window limits.

Lightweight estimator: ~4 characters per token as a heuristic for English text.
If `tiktoken` is available, it can be wired in later; keep this dependency-free.
"""

from __future__ import annotations

from typing import Iterable, Dict, Any, List, Tuple


def estimate_tokens(text: str) -> int:
    """Rough token estimator (~4 chars per token)."""
    if not text:
        return 0
    # Fast path: count characters; avoid splitting to keep it cheap
    length = len(text)
    # 4 chars per token (ceil)
    return (length + 3) // 4


def estimate_tokens_for_result(item: Dict[str, Any]) -> int:
    """Estimate tokens for a search result dict (title + snippet/content)."""
    title = (item.get("title") or "").strip()
    snippet = (item.get("snippet") or item.get("content") or "").strip()
    return estimate_tokens(title) + estimate_tokens(snippet)


def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    """Trim text to approximately max_tokens (by characters)."""
    if max_tokens <= 0 or not text:
        return ""
    # Convert token cap to ~char cap
    cap = max_tokens * 4
    if len(text) <= cap:
        return text
    return text[: max(0, cap - 3)] + "..."


def select_items_within_budget(
    items: Iterable[Dict[str, Any]],
    max_tokens: int,
    per_item_min_tokens: int = 40,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Select as many items as fit within max_tokens, returning (items, used, dropped).

    - Ensures each selected item contributes at least `per_item_min_tokens` budget
      to avoid pathological over-selection of long items.
    """
    selected: List[Dict[str, Any]] = []
    used = 0
    dropped = 0

    for it in items:
        # Ensure each chosen item has a minimum budget slice
        # so we don't exceed max_tokens unexpectedly
        contribution = max(per_item_min_tokens, estimate_tokens_for_result(it))
        if used + contribution > max_tokens:
            dropped += 1
            continue
        selected.append(it)
        used += contribution
    return selected, used, dropped


def compute_budget_plan(total_tokens: int, plan: Dict[str, float] | None = None) -> Dict[str, int]:
    """Translate a fractional plan into integer token buckets.

    Default split (instructions/knowledge/tools/scratch) = 0.15/0.70/0.15/0.0
    """
    plan = plan or {"instructions": 0.15, "knowledge": 0.70, "tools": 0.15, "scratch": 0.0}
    out: Dict[str, int] = {}
    remaining = total_tokens
    keys = list(plan.keys())
    for k in keys[:-1]:
        v = max(0.0, float(plan.get(k, 0.0)))
        bucket = int(total_tokens * v)
        out[k] = bucket
        remaining -= bucket
    # Assign remainder to last bucket (to keep sums consistent)
    last_key = keys[-1]
    out[last_key] = max(0, remaining)
    return out


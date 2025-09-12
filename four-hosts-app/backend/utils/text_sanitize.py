"""
Shared text sanitation utilities for safe display/export.

- strip_html: remove tags like <jats:p> and decode entities
- collapse_ws: collapse consecutive whitespace
- sanitize_text: convenience wrapper applying both and trimming
"""

from __future__ import annotations

import html as _html
import re as _re

__all__ = ["strip_html", "collapse_ws", "sanitize_text"]


def strip_html(text: str) -> str:
    if not text:
        return ""
    try:
        t = _html.unescape(str(text))
        t = _re.sub(r"<[^>]+>", " ", t)
        return t
    except Exception:
        return str(text)


def collapse_ws(text: str) -> str:
    if not text:
        return ""
    try:
        return _re.sub(r"\s+", " ", str(text)).strip()
    except Exception:
        return str(text).strip()


def sanitize_text(text: str, *, max_len: int | None = None) -> str:
    out = collapse_ws(strip_html(text))
    if max_len is not None and len(out) > max_len:
        return out[: max_len - 1] + "…"
    return out


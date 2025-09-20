from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.result_adapter import ResultAdapter
try:
    import structlog  # type: ignore
    logger = structlog.get_logger(__name__)
except Exception:  # pragma: no cover - fallback when structlog missing
    import logging
    logger = logging.getLogger(__name__)


class ResultDeduplicator:
    """Removes duplicate search results using URL and content similarity."""

    def __init__(self, similarity_threshold: float | None = None) -> None:
        if similarity_threshold is None:
            try:
                env = os.getenv("DEDUP_SIMILARITY_THRESH")
                similarity_threshold = float(env) if env is not None else 0.8
            except Exception:
                similarity_threshold = 0.8
        self.similarity_threshold = similarity_threshold

    async def deduplicate_results(self, results: List[Any]) -> Dict[str, Any]:
        if not results:
            return {
                "unique_results": [],
                "duplicates_removed": 0,
                "similarity_threshold": self.similarity_threshold,
            }

        unique_results: List[Any] = []
        duplicates_removed = 0
        seen_urls: set[str] = set()

        url_deduplicated: List[Any] = []
        for result in results:
            adapter = ResultAdapter(result)
            url = adapter.url
            if not url:
                duplicates_removed += 1
                continue
            if url not in seen_urls:
                seen_urls.add(url)
                url_deduplicated.append(result)
            else:
                duplicates_removed += 1

        buckets: Dict[int, List[Tuple[int, Any]]] = {}
        for r in url_deduplicated:
            adapter = ResultAdapter(r)
            basis = f"{adapter.title} {adapter.snippet}"
            sh = self._simhash64(basis)
            key = sh >> 52
            buckets.setdefault(key, []).append((sh, r))

        unique: List[Any] = []
        for _, items in buckets.items():
            kept: List[Tuple[int, Any]] = []
            for simhash, result in items:
                adapter = ResultAdapter(result)
                is_dup = False
                for ref_hash, ref in kept:
                    threshold = self._adaptive_threshold(adapter, ResultAdapter(ref))
                    if self._hamdist64(simhash, ref_hash) <= threshold:
                        is_dup = True
                        break
                    if self._calculate_content_similarity(adapter, ResultAdapter(ref)) >= self.similarity_threshold:
                        is_dup = True
                        break
                if is_dup:
                    duplicates_removed += 1
                else:
                    kept.append((simhash, result))
                    unique.append(result)

        # Emit structured metrics for observability
        try:
            unique_domains = set()
            for r in unique:
                try:
                    d = ResultAdapter(r).domain
                    if d:
                        unique_domains.add(d)
                except Exception:
                    continue
            logger.info(
                "Result deduplication complete",
                stage="deduplication",
                input_count=len(results),
                output_count=len(unique),
                duplicates_removed=duplicates_removed,
                dedup_methods=["url_norm", "simhash", "jaccard_title_snippet"],
                unique_domains=len(unique_domains),
                similarity_threshold=self.similarity_threshold,
            )
        except Exception:
            pass

        return {
            "unique_results": unique,
            "duplicates_removed": duplicates_removed,
            "similarity_threshold": self.similarity_threshold,
        }

    @staticmethod
    def _adaptive_threshold(result: ResultAdapter, reference: ResultAdapter) -> int:
        domain = result.domain or reference.domain
        result_type = result.get("result_type", reference.get("result_type", "web"))
        if result_type == "academic" or (domain and (domain.endswith(".edu") or domain.endswith(".gov"))):
            return 8
        if result_type in {"news", "blog"}:
            return 5
        return 3

    @staticmethod
    def _calculate_content_similarity(a: ResultAdapter, b: ResultAdapter) -> float:
        text_a = f"{a.title} {a.snippet}".lower()
        text_b = f"{b.title} {b.snippet}".lower()
        tokens_a = set(re.findall(r"[a-z0-9]+", text_a))
        tokens_b = set(re.findall(r"[a-z0-9]+", text_b))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))

    @staticmethod
    def _simhash64(text: str) -> int:
        tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        vector = [0] * 64
        for token in tokens[:200]:
            h = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
            for idx in range(64):
                vector[idx] += 1 if (h >> idx) & 1 else -1
        value = 0
        for idx in range(64):
            if vector[idx] >= 0:
                value |= (1 << idx)
        return value

    @staticmethod
    def _hamdist64(a: int, b: int) -> int:
        x = a ^ b
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count

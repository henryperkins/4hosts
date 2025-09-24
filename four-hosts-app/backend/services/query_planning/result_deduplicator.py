from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.result_adapter import ResultAdapter

# Logging — prefer structlog if available. Always ensure that the
# application-wide logging configuration is applied.
try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover – structlog not installed
    import structlog  # type: ignore

from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


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
        self.provider_thresholds: Dict[str, float] = {"default": self.similarity_threshold}
        exa_default = 0.85
        exa_override = os.getenv("DEDUP_SIMILARITY_THRESH_EXA")
        if exa_override is not None:
            try:
                exa_default = float(exa_override)
            except Exception:
                exa_default = 0.85
        exa_clamped = max(0.0, min(1.0, exa_default))
        self.provider_thresholds["exa"] = exa_clamped

    async def deduplicate_results(self, results: List[Any]) -> Dict[str, Any]:
        if not results:
            return {
                "unique_results": [],
                "duplicates_removed": 0,
                "similarity_threshold": self.similarity_threshold,
            }

        duplicates_removed = 0
        duplicate_groups: Dict[str, List[Any]] = {}
        url_index: Dict[str, Tuple[Any, Optional[str]]] = {}
        url_group_map: Dict[int, Optional[str]] = {}
        unique_group_map: Dict[int, Optional[str]] = {}

        url_deduplicated: List[Any] = []
        for result in results:
            adapter = ResultAdapter(result)
            url = adapter.url
            if not url:
                duplicates_removed += 1
                continue
            existing = url_index.get(url)
            if existing:
                duplicates_removed += 1
                representative, group_key = existing
                if not group_key:
                    group_key = f"group_{len(duplicate_groups) + 1}"
                    duplicate_groups[group_key] = [representative]
                    url_index[url] = (representative, group_key)
                    url_group_map[id(representative)] = group_key
                duplicate_groups[group_key].append(result)
                continue
            url_index[url] = (result, None)
            url_group_map[id(result)] = None
            url_deduplicated.append(result)

        buckets: Dict[int, List[Tuple[int, Any]]] = {}
        for r in url_deduplicated:
            adapter = ResultAdapter(r)
            basis = f"{adapter.title} {adapter.snippet}"
            sh = self._simhash64(basis)
            key = sh >> 52
            buckets.setdefault(key, []).append((sh, r))

        unique: List[Any] = []
        for _, items in buckets.items():
            kept: List[Tuple[int, Any, Optional[str]]] = []
            for simhash, result in items:
                adapter = ResultAdapter(result)
                is_dup = False
                target_group: Optional[str] = None
                for idx, (ref_hash, ref, group_key) in enumerate(kept):
                    ref_adapter = ResultAdapter(ref)
                    threshold = self._adaptive_threshold(adapter, ref_adapter)
                    if self._hamdist64(simhash, ref_hash) <= threshold:
                        is_dup = True
                        target_group = group_key
                    else:
                        provider_key = self._provider_key(adapter, ref_adapter)
                        similarity_threshold = self.provider_thresholds.get(
                            provider_key,
                            self.similarity_threshold,
                        )
                        if (
                            adapter.domain
                            and ref_adapter.domain
                            and adapter.domain == ref_adapter.domain
                        ):
                            similarity_threshold = min(similarity_threshold, 0.6)
                        if self._calculate_content_similarity(adapter, ref_adapter) >= similarity_threshold:
                            is_dup = True
                            target_group = group_key
                    if is_dup:
                        duplicates_removed += 1
                        if not target_group:
                            target_group = f"group_{len(duplicate_groups) + 1}"
                            duplicate_groups[target_group] = [ref]
                            kept[idx] = (ref_hash, ref, target_group)
                        unique_group_map[id(ref)] = target_group
                        duplicate_groups[target_group].append(result)
                        break
                if not is_dup:
                    for ref in unique:
                        ref_adapter = ResultAdapter(ref)
                        if (
                            adapter.domain
                            and ref_adapter.domain
                            and adapter.domain != ref_adapter.domain
                        ):
                            continue
                        threshold = self._adaptive_threshold(adapter, ref_adapter)
                        if self._hamdist64(
                            self._simhash64(f"{adapter.title} {adapter.snippet}"),
                            self._simhash64(f"{ref_adapter.title} {ref_adapter.snippet}"),
                        ) <= threshold:
                            dup_candidate = True
                        else:
                            provider_key = self._provider_key(adapter, ref_adapter)
                            similarity_threshold = self.provider_thresholds.get(
                                provider_key,
                                self.similarity_threshold,
                            )
                            if (
                                adapter.domain
                                and ref_adapter.domain
                                and adapter.domain == ref_adapter.domain
                            ):
                                similarity_threshold = min(similarity_threshold, 0.6)
                            dup_candidate = (
                                self._calculate_content_similarity(adapter, ref_adapter)
                                >= similarity_threshold
                            )
                        if dup_candidate:
                            duplicates_removed += 1
                            group_key = unique_group_map.get(id(ref))
                            if not group_key:
                                group_key = f"group_{len(duplicate_groups) + 1}"
                                duplicate_groups[group_key] = [ref]
                                unique_group_map[id(ref)] = group_key
                            duplicate_groups[group_key].append(result)
                            is_dup = True
                            break

                if not is_dup:
                    kept.append((simhash, result, None))
                    unique.append(result)
                    unique_group_map[id(result)] = url_group_map.get(id(result))

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

        filtered_groups = {
            group_id: members
            for group_id, members in duplicate_groups.items()
            if len(members) >= 2
        }

        return {
            "unique_results": unique,
            "duplicates_removed": duplicates_removed,
            "similarity_threshold": self.similarity_threshold,
            "duplicate_groups": filtered_groups,
        }

    @staticmethod
    def _adaptive_threshold(result: ResultAdapter, reference: ResultAdapter) -> int:
        domain = result.domain or reference.domain
        result_type = result.get("result_type", reference.get("result_type", "web"))
        provider = result.get("provider") or reference.get("provider")
        if isinstance(provider, str) and provider.lower() == "exa":
            return 6
        if result_type == "academic" or (domain and (domain.endswith(".edu") or domain.endswith(".gov"))):
            return 8
        if result_type in {"news", "blog"}:
            return 5
        return 3

    @staticmethod
    def _provider_key(result: ResultAdapter, reference: ResultAdapter) -> str:
        provider = (
            result.get("provider")
            or reference.get("provider")
            or result.get("source")
            or reference.get("source")
        )
        if not provider:
            try:
                provider = result.source_api or reference.source_api
            except Exception:
                provider = "default"
        if isinstance(provider, str) and provider:
            return provider.lower()
        return "default"

    @staticmethod
    def _calculate_content_similarity(a: ResultAdapter, b: ResultAdapter) -> float:
        text_a = " ".join(
            part
            for part in [a.title, a.snippet, getattr(a, "content", "")]
            if part
        ).lower()
        text_b = " ".join(
            part
            for part in [b.title, b.snippet, getattr(b, "content", "")]
            if part
        ).lower()
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

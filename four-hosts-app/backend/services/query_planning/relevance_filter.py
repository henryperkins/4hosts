from __future__ import annotations

import os
import re
from typing import Any

from services.search_apis import SearchResult
from services.text_compression import query_compressor


class EarlyRelevanceFilter:
    """Early-stage relevance filtering to remove obviously irrelevant results."""

    def __init__(self) -> None:
        self.spam_indicators = {
            "viagra",
            "cialis",
            "casino",
            "poker",
            "lottery",
            "weight loss",
            "get rich quick",
            "work from home",
            "singles in your area",
            "hot deals",
            "limited time offer",
        }

        try:
            custom_spam = os.getenv("EARLY_FILTER_SPAM_INDICATORS")
            if custom_spam:
                additional_spam = [s.strip().lower() for s in custom_spam.split(",") if s.strip()]
                self.spam_indicators.update(additional_spam)
        except Exception:
            pass

        try:
            if os.getenv("EARLY_FILTER_BLOCK_DOMAINS", "1").lower() in {"1", "true", "yes", "on"}:
                self.low_quality_domains = {
                    "ezinearticles.com",
                    "articlesbase.com",
                    "squidoo.com",
                    "hubpages.com",
                    "buzzle.com",
                    "ehow.com",
                }
            else:
                self.low_quality_domains = set()
        except Exception:
            self.low_quality_domains = set()

    def is_relevant(self, result: SearchResult, query: str, paradigm: str) -> bool:
        title_val = (getattr(result, "title", "") or "")
        snippet_val = (getattr(result, "snippet", "") or "")
        combined_text = f"{title_val} {snippet_val}".lower()

        if any(spam in combined_text for spam in self.spam_indicators):
            return False

        domain_val = (getattr(result, "domain", "") or "").lower()
        if domain_val in self.low_quality_domains:
            return False

        if not isinstance(title_val, str) or len(title_val.strip()) < 10:
            return False
        if not isinstance(snippet_val, str) or len(snippet_val.strip()) < 20:
            return False

        try:
            query_terms = set(query_compressor.extract_keywords(query))
            if not query_terms:
                raise ValueError("empty terms")
            matching_terms = {term for term in query_terms if term in combined_text}
            if not matching_terms:
                import re as _re

                def _tokens(text: str) -> set[str]:
                    return {t for t in _re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 2}

                query_terms = _tokens(query)
                combined_tokens = _tokens(combined_text)
                has_overlap = bool(query_terms & combined_tokens)
                if not has_overlap and len(query_terms) >= 3:
                    min_matches = max(1, len(query_terms) // 3)
                    has_overlap = len(query_terms & combined_tokens) >= min_matches
                if not has_overlap:
                    related_terms = {
                        "llm",
                        "ai",
                        "gpt",
                        "model",
                        "language",
                        "prompt",
                        "context",
                        "engineering",
                    }
                    if not any(term in combined_text for term in related_terms):
                        return False
        except Exception:
            fallback_terms = [term.lower() for term in query.split() if len(term) > 2]
            if fallback_terms and len(fallback_terms) > 2:
                if not any(term in combined_text for term in fallback_terms):
                    return False

        if self._is_likely_duplicate_site(domain_val):
            return False

        if paradigm == "bernard" and getattr(result, "result_type", "web") == "web":
            authoritative_indicators = {
                ".edu",
                ".gov",
                "journal",
                "research",
                "study",
                "analysis",
                "technology",
                "innovation",
                "science",
                "ieee",
                "acm",
                "mit",
                "stanford",
                "harvard",
                "arxiv",
                "nature",
                "springer",
            }
            tech_indicators = {
                "ai",
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "neural",
                "algorithm",
                "technology",
                "computing",
                "software",
                "innovation",
            }

            has_authority = any(indicator in domain_val or indicator in combined_text for indicator in authoritative_indicators)
            has_tech_content = any(indicator in combined_text for indicator in tech_indicators)

            try:
                q_terms = set(query_compressor.extract_keywords(query))
            except Exception:
                q_terms = {t for t in (query or "").lower().split() if len(t) > 2}
            has_direct_overlap = any(t in combined_text for t in q_terms) if q_terms else True

            if not (has_authority or has_tech_content or has_direct_overlap):
                academic_terms = {
                    "methodology",
                    "hypothesis",
                    "conclusion",
                    "abstract",
                    "citation",
                    "analysis",
                    "framework",
                    "approach",
                    "technique",
                    "evaluation",
                }
                if not any(term in combined_text for term in academic_terms):
                    return False

        try:
            thr = float(os.getenv("EARLY_THEME_OVERLAP_MIN", "0.08") or 0.08)
            q_terms = set(query_compressor.extract_keywords(query))
            if q_terms:
                import re as _re

                def _toks(text: str) -> set[str]:
                    return {t for t in _re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if len(t) > 2}

                rtoks = _toks(combined_text)
                if rtoks:
                    inter = len(q_terms & rtoks)
                    union = len(q_terms | rtoks)
                    jac = (inter / float(union)) if union else 0.0
                    whitelist_domains = {
                        "arxiv.org",
                        "scholar.google.com",
                        "pubmed.ncbi.nlm.nih.gov",
                        "nature.com",
                        "science.org",
                        "ieee.org",
                        "acm.org",
                        "springer.com",
                        "wiley.com",
                        "tandfonline.com",
                        "researchgate.net",
                    }
                    if domain_val not in whitelist_domains and jac < thr:
                        return False
        except Exception:
            pass

        return True

    @staticmethod
    def _is_likely_duplicate_site(domain: str) -> bool:
        duplicate_patterns = [
            r".*-mirror\.",
            r".*-cache\.",
            r".*-proxy\.",
            r".*\.mirror\.",
            r".*\.cache\.",
            r".*\.proxy\.",
            r"webcache\.",
            r"cached\.",
        ]

        for pattern in duplicate_patterns:
            if re.match(pattern, domain.lower()):
                return True

        return False

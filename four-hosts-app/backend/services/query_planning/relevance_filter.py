from __future__ import annotations

import os
import re
from typing import Dict, List, Set

from services.search_apis import SearchResult
from services.text_compression import query_compressor
from models.paradigms_sources import PREFERRED_SOURCES


_PARADIGM_KEYWORDS: Dict[str, List[str]] = {
    "bernard": [
        "study",
        "research",
        "statistic",
        "data",
        "analysis",
        "methodology",
        "sample",
        "peer-reviewed",
        "meta-analysis",
        "evidence",
    ],
    "maeve": [
        "roi",
        "return on investment",
        "market",
        "competitive",
        "strategy",
        "revenue",
        "growth",
        "benchmark",
        "roadmap",
        "kpi",
        "profit",
    ],
    "dolores": [
        "systemic",
        "corruption",
        "injustice",
        "power",
        "abuse",
        "inequity",
        "whistleblower",
        "accountability",
        "investigation",
        "lawsuit",
        "disparity",
    ],
    "teddy": [
        "support",
        "resource",
        "assistance",
        "aid",
        "hotline",
        "helpline",
        "services",
        "access",
        "care",
        "nonprofit",
        "community",
        "eligibility",
        "relief",
    ],
}

_PARADIGM_ALIGNMENT_THRESHOLDS: Dict[str, float] = {
    "bernard": 0.35,
    "maeve": 0.3,
    "dolores": 0.35,
    "teddy": 0.3,
}


class EarlyRelevanceFilter:
    """Early-stage relevance filtering.

    Removes obviously irrelevant results before expensive processing.
    """

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
                additional_spam = [
                    token.strip().lower()
                    for token in custom_spam.split(",")
                    if token.strip()
                ]
                self.spam_indicators.update(additional_spam)
        except Exception:
            pass

        try:
            env_flag = os.getenv("EARLY_FILTER_BLOCK_DOMAINS", "1")
            enabled = (env_flag or "1").lower() in {"1", "true", "yes", "on"}
            if enabled:
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

    def is_relevant(
        self,
        result: SearchResult,
        query: str,
        paradigm: str,
    ) -> bool:
        title_val = (getattr(result, "title", "") or "")
        snippet_val = (getattr(result, "snippet", "") or "")
        combined_text = f"{title_val} {snippet_val}".lower()
        paradigm_code = self._normalize_paradigm(paradigm)

        if any(spam in combined_text for spam in self.spam_indicators):
            return False

        domain_val = (getattr(result, "domain", "") or "").lower()
        if domain_val in self.low_quality_domains:
            return False

        if not isinstance(title_val, str) or len(title_val.strip()) < 10:
            return False
        if not isinstance(snippet_val, str) or len(snippet_val.strip()) < 20:
            return False

        query_terms: set[str] = set()
        try:
            query_terms = set(query_compressor.extract_keywords(query))
            if not query_terms:
                raise ValueError("empty terms")
            matching_terms = {
                term
                for term in query_terms
                if term in combined_text
            }
            if not matching_terms:
                import re as _re

                def _tokens(text: str) -> set[str]:
                    tokens = _re.findall(r"[A-Za-z0-9]+", text.lower())
                    return {t for t in tokens if len(t) > 2}

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
            fallback_terms = [
                term.lower()
                for term in query.split()
                if len(term) > 2
            ]
            if fallback_terms and len(fallback_terms) > 2:
                if not any(term in combined_text for term in fallback_terms):
                    return False

        if self._is_likely_duplicate_site(domain_val):
            return False

        if (
            paradigm_code == "bernard"
            and getattr(result, "result_type", "web") == "web"
        ):
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

            has_authority = any(
                indicator in domain_val or indicator in combined_text
                for indicator in authoritative_indicators
            )
            has_tech_content = any(
                indicator in combined_text for indicator in tech_indicators
            )

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
                    tokens = _re.findall(r"[A-Za-z0-9]+", (text or "").lower())
                    return {t for t in tokens if len(t) > 2}

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
                        alignment_score = self._paradigm_alignment_score(
                            paradigm_code,
                            combined_text,
                            domain_val,
                        )
                        # Maeve-oriented research often leans on commercial language
                        # rather than the original query tokens. Allow those results
                        # through when we see clear Maeve signals (at least one keyword
                        # match or preferred source), even if token overlap is low.
                        maeve_override = (
                            paradigm_code == "maeve"
                            and alignment_score >= 0.12
                        )
                        if not maeve_override:
                            return False
        except Exception:
            pass

        # Paradigm alignment is computed but not used as a hard filter.
        # Results that passed the query term matching logic above are considered relevant.
        # Paradigm alignment should be used as a scoring signal in downstream ranking,
        # not as a binary relevance filter that discards valid matches.
        # This prevents dramatic reduction in recall for queries that match query terms
        # but don't contain enough hard-coded paradigm keywords.

        return True

    @staticmethod
    def _normalize_paradigm(paradigm: str) -> str:
        return (paradigm or "").strip().lower()

    @staticmethod
    def _paradigm_alignment_threshold(paradigm_code: str) -> float:
        return _PARADIGM_ALIGNMENT_THRESHOLDS.get(paradigm_code, 0.0)

    def _preferred_domains(self, paradigm_code: str) -> Set[str]:
        sources = PREFERRED_SOURCES.get(paradigm_code, [])
        return {domain.lower() for domain in sources}

    def _paradigm_alignment_score(
        self,
        paradigm_code: str,
        combined_text: str,
        domain: str,
    ) -> float:
        if not paradigm_code:
            return 0.0

        score = 0.0
        domain_lower = (domain or "").lower()
        if domain_lower and domain_lower in self._preferred_domains(paradigm_code):
            score += 0.6

        keywords = _PARADIGM_KEYWORDS.get(paradigm_code, [])
        matches = sum(1 for term in keywords if term in combined_text)
        if matches:
            score += min(0.4, matches * 0.12)

        if paradigm_code == "bernard":
            if re.search(r"\b\d+(?:\.\d+)?\s*%", combined_text):
                score += 0.1
            if re.search(r"\bp\s*[=<>]\s*\d+", combined_text):
                score += 0.1
            if re.search(r"\bn\s*=\s*\d+", combined_text):
                score += 0.05
        elif paradigm_code == "maeve":
            if "swot" in combined_text or "kpi" in combined_text:
                score += 0.05
            if re.search(r"\$[\d,]+", combined_text):
                score += 0.1
            if "timeline" in combined_text:
                score += 0.05
        elif paradigm_code == "dolores":
            if "investigation" in combined_text or "whistleblower" in combined_text:
                score += 0.1
            if "accountability" in combined_text or "power" in combined_text:
                score += 0.05
            if "systemic" in combined_text or "cover-up" in combined_text:
                score += 0.05
        elif paradigm_code == "teddy":
            if "contact" in combined_text or "call" in combined_text:
                score += 0.05
            if "free" in combined_text or "no-cost" in combined_text:
                score += 0.05
            if "support" in combined_text or "access" in combined_text:
                score += 0.05

        return min(score, 1.0)

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

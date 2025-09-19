from __future__ import annotations

import re
import logging
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from ..text_utils import _use_nltk, STOP_WORDS, tokenize
import structlog

logger = structlog.get_logger(__name__)

class QueryOptimizer:
    """
    Generates cleaned-up / expanded query strings that retain user intent
    while maximising recall (quoted-phrase protection, synonym expansion,
    domain-specific boosts, etc.).
    """

    def __init__(self):
        self.use_nltk = _use_nltk
        self.stop_words = STOP_WORDS

        # Terms that rarely influence retrieval quality
        self.noise_terms: Set[str] = {"information", "details", "find", "show", "tell"}

        # Common multi-word technical entities we want to protect (keep quoted)
        self.known_entities = [
            "context engineering",
            "web applications",
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural network",
            "neural networks",
            "large language model",
            "large language models",
            "state of the art",
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "generative ai",
            "transformer models",
            "foundation models",
        ]

        # Paradigm-specific dictionaries (used by _add_domain_specific_terms)
        self.paradigm_terms = {
            "dolores": ["investigation", "expose", "systemic", "corruption"],
            "teddy": ["support", "community", "help", "care"],
            "bernard": ["research", "analysis", "data", "evidence"],
            "maeve": ["strategy", "business", "optimization", "market"],
        }

    # --------------------------------------------------------------------- #
    #                     Internal helper functions                         #
    # --------------------------------------------------------------------- #
    def _extract_entities(self, query: str) -> Tuple[List[str], str]:
        """
        1.  Keep quoted phrases intact.          ->  "climate change"
        2.  Detect known multi-token entities.   ->  machine learning
        3.  Grab obvious proper nouns.           ->  OpenAI, Maeve
        Returns (protected_entities, remaining_text)
        """
        start_time = time.time()

        logger.debug(
            "Starting entity extraction",
            stage="entity_extraction",
            query_length=len(query),
        )

        protected: List[str] = []
        remainder = query

        # 1. quoted phrases -------------------------------------------------
        for phrase in re.findall(r'"([^"]+)"', query):
            protected.append(phrase)
            remainder = remainder.replace(f'"{phrase}"', " ")

        # 2. known entities --------------------------------------------------
        low = remainder.lower()
        for ent in self.known_entities:
            if ent in low:
                protected.append(ent)
                low = low.replace(ent, " ")
                remainder = re.sub(re.escape(ent), " ", remainder, flags=re.I)

        # 3. proper nouns heuristic -----------------------------------------
        proper = re.findall(r"\b[A-Z][a-z]{2,}\b", remainder)
        for p in proper:
            if p.lower() not in self.stop_words:
                protected.append(p)

        # Clean leftover text
        remainder = re.sub(r"\s+", " ", remainder).strip()
        result = list(dict.fromkeys(protected))  # dedup while preserving order

        logger.debug(
            "Entity extraction completed",
            stage="entity_extraction_complete",
            duration_ms=(time.time() - start_time) * 1000,
            entities=result[:10],  # Log first 10 entities
            metrics={
                "protected_count": len(result),
                "quoted_phrases": len(re.findall(r'\"([^\"]+)\"', query)),
                "known_entities_found": sum(1 for e in self.known_entities if e.lower() in query.lower()),
                "proper_nouns_found": len(proper),
            },
        )

        return result, remainder

    def _intelligent_stopword_removal(self, text: str) -> List[str]:
        tokens = tokenize(text)          # shared helper already removes stop-words
        return [t for t in tokens if t not in self.noise_terms]

    # --------------------------------------------------------------------- #
    #                       Public helper methods                           #
    # --------------------------------------------------------------------- #
    def get_key_terms(self, query: str) -> List[str]:
        start_time = time.time()

        logger.info(
            "Extracting key terms",
            stage="key_terms_extraction",
            query_preview=query[:100] if query else "",
        )

        ents, left = self._extract_entities(query)
        terms = self._intelligent_stopword_removal(left)
        result = [e.replace('"', "") for e in ents] + terms

        logger.info(
            "Key terms extraction completed",
            stage="key_terms_complete",
            duration_ms=(time.time() - start_time) * 1000,
            terms=result[:20],  # Log first 20 terms
            metrics={
                "entity_count": len(ents),
                "term_count": len(terms),
                "total_count": len(result),
            },
        )

        return result

    # --------------------------------------------------------------------- #
    #              Query-expansion / variation generation                   #
    # --------------------------------------------------------------------- #
    def generate_query_variations(
        self, query: str, paradigm: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Returns several variations keyed by a label:
            {
                "primary": "...",
                "semantic": "...",
                "question": "...",
                "synonym": "...",
                ...
            }
        We usually send only the first 2-3 variations to each provider.
        """
        start_time = time.time()

        logger.info(
            "Starting query variation generation",
            stage="query_variation_start",
            paradigm=paradigm,
            query_preview=query[:100] if query else "",
            config={
                "has_paradigm": paradigm is not None,
            },
        )

        ents, left = self._extract_entities(query)
        keywords = self._intelligent_stopword_removal(left)

        # PRIMARY (AND-joined) ---------------------------------------------
        if ents or keywords:
            primary = " AND ".join([f'"{e}"' for e in ents] + keywords)
        else:
            primary = query.strip()

        variations: Dict[str, str] = {"primary": primary}

        # SEMANTIC (OR between protected phrases) --------------------------
        if len(ents) > 1:
            quoted_ents = [f'"{e}"' for e in ents]
            variations["semantic"] = f"({' OR '.join(quoted_ents)}) AND {' '.join(keywords)}"
        else:
            variations["semantic"] = primary.replace(" AND ", " ")

        # QUESTION form ----------------------------------------------------
        wh = ["what is", "how does", "why is", "explain", "when did"]
        if not any(query.lower().startswith(w) for w in wh):
            # De-duplicate terms case-insensitively, drop connectors, cap length
            raw_terms = [e for e in ents] + keywords
            seen: set[str] = set()
            clean_terms: List[str] = []
            for t in raw_terms:
                tt = (t or "").strip()
                if not tt:
                    continue
                tl = tt.lower()
                if tl in {"and", "or", "the"}:
                    continue
                if tl not in seen:
                    seen.add(tl)
                    clean_terms.append(tt)
            # Limit to keep queries readable
            clean_terms = clean_terms[:6]
            if clean_terms:
                variations["question"] = (
                    f"what is the relationship between {' and '.join(clean_terms)}"
                )
            else:
                variations["question"] = f"what is {query}"  # fallback

        # SYNONYM expansion -------------------------------------------------
        syn_kw = self._expand_synonyms(keywords)
        if syn_kw != keywords:
            variations["synonym"] = " ".join([f'"{e}"' for e in ents] + syn_kw)

        # RELATED concepts --------------------------------------------------
        rel = self._get_related_concepts(keywords)
        if rel:
            variations["related"] = f"{primary} OR ({' OR '.join(rel)})"

        # DOMAIN-specific ---------------------------------------------------
        if paradigm:
            dom = self._add_domain_specific_terms(primary, paradigm)
            if dom != primary:
                variations["domain_specific"] = dom

        # BROAD (first 3 important terms) ----------------------------------
        broad_terms = ents + keywords
        if len(broad_terms) > 2:
            variations["broad"] = " ".join(broad_terms[:3])

        # EXACT phrase ------------------------------------------------------
        if len(query.split()) <= 6:
            variations["exact_phrase"] = f'"{query.strip()}"'

        logger.info(
            "Query variation generation completed",
            stage="query_variation_complete",
            paradigm=paradigm,
            duration_ms=(time.time() - start_time) * 1000,
            metrics={
                "variation_count": len(variations),
                "entity_count": len(ents),
                "keyword_count": len(keywords),
                "variations_generated": list(variations.keys()),
            },
        )

        return variations

    def optimize_query(self, query: str, paradigm: Optional[str] = None) -> str:
        """Return just the 'primary' variation for convenience."""
        return self.generate_query_variations(query, paradigm)["primary"]

    # --------------------------------------------------------------------- #
    #                       Private expansion helpers                       #
    # --------------------------------------------------------------------- #
    def _expand_synonyms(self, terms: List[str]) -> List[str]:
        if not (self.use_nltk and terms):
            return terms[:]  # unchanged

        from nltk.corpus import wordnet as wn

        expanded: List[str] = []
        for t in terms:
            expanded.append(t)
            try:
                syns = wn.synsets(t)
            except Exception:
                syns = []
            for syn in syns[:2]:                      # limit per term
                for lemma in syn.lemma_names()[:3]:
                    lemma = lemma.replace("_", " ")
                    if lemma.lower() != t.lower() and lemma not in expanded:
                        expanded.append(lemma)
            if len(expanded) >= len(terms) + 4:       # cap overall growth
                break
        return expanded

    def _get_related_concepts(self, terms: List[str]) -> List[str]:
        concept_map = {
            "ai": ["artificial intelligence", "machine learning", "deep learning"],
            "ml": ["machine learning", "algorithms", "models"],
            "ethic": ["responsible ai", "moral", "governance"],
            "security": ["cybersecurity", "privacy", "data protection"],
            "climate": ["global warming", "environmental impact"],
            "health": ["healthcare", "medical", "wellness"],
            "finance": ["economic", "investment", "financial"],
            "education": ["learning", "teaching", "academic"],
        }
        related: List[str] = []
        for t in terms:
            for key, vals in concept_map.items():
                if key in t.lower():
                    related.extend(vals)
        # Also plural/singular normalization triggers
        for t in terms:
            if t.endswith("s"):
                related.append(t[:-1])
        return list(dict.fromkeys(related))[:3]        # unique, max 3

    def _add_domain_specific_terms(self, query: str, paradigm: str) -> str:
        extra = self.paradigm_terms.get(paradigm.lower())
        if not extra:
            return query
        return f"{query} {' '.join(extra)}"

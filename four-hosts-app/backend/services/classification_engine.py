"""
Four Hosts Classification Engine
Core implementation for paradigm classification with 85%+ accuracy target
"""

import re
import hashlib
import json
import traceback
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import hmac, hashlib as _hash
import os
from collections import OrderedDict

# Security utilities
from utils.security import sanitize_user_input, pattern_validator

import structlog

logger = structlog.get_logger(__name__)


# --- Structured Output Definitions ---

# JSON schema used when requesting LLM-driven classifications. Enforces
# well-formed output so downstream parsing does not fail even if the model
# attempts to add commentary around the payload.
CLASSIFICATION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "FourHostsClassification",
    "schema": {
        "type": "object",
        "properties": {
            "dolores": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One or two sentence explanation",
                    },
                },
                "required": ["score", "reasoning"],
                "additionalProperties": False,
            },
            "teddy": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "reasoning": {
                        "type": "string",
                    },
                },
                "required": ["score", "reasoning"],
                "additionalProperties": False,
            },
            "bernard": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "reasoning": {
                        "type": "string",
                    },
                },
                "required": ["score", "reasoning"],
                "additionalProperties": False,
            },
            "maeve": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "reasoning": {
                        "type": "string",
                    },
                },
                "required": ["score", "reasoning"],
                "additionalProperties": False,
            },
        },
        "required": ["dolores", "teddy", "bernard", "maeve"],
        "additionalProperties": False,
    },
}

# Internal imports - deferred to avoid circular dependency
llm_client = None
LLM_AVAILABLE = False

# --- Core Enums and Models ---


class HostParadigm(str, Enum):
    """The four host consciousness paradigms"""

    DOLORES = "revolutionary"
    TEDDY = "devotion"
    BERNARD = "analytical"
    MAEVE = "strategic"


@dataclass
class QueryFeatures:
    """Extracted features from a query for classification"""

    text: str
    tokens: List[str]
    entities: List[str]
    intent_signals: List[str]
    domain: Optional[str]
    urgency_score: float
    complexity_score: float
    emotional_valence: float


@dataclass
class ParadigmScore:
    """Score for a specific paradigm with reasoning"""

    paradigm: HostParadigm
    score: float
    confidence: float
    reasoning: List[str]
    keyword_matches: List[str]


@dataclass
class ClassificationResult:
    """Complete classification result with distribution"""

    query: str
    primary_paradigm: HostParadigm
    secondary_paradigm: Optional[HostParadigm]
    distribution: Dict[HostParadigm, float]
    confidence: float
    features: QueryFeatures
    reasoning: Dict[HostParadigm, List[str]]
    # Optional: structured signals for UI (e.g., matched keywords per paradigm)
    signals: Dict[HostParadigm, Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --- Feature Extraction (To be implemented) ---


class QueryAnalyzer:
    """Analyzes queries to extract classification features"""

    # Canon is imported from models.paradigms to keep system-wide consistency
    from models.paradigms import (
        PARADIGM_KEYWORDS as _CANON_KEYWORDS,
        PARADIGM_PATTERNS as _CANON_PATTERNS,
        DOMAIN_PARADIGM_BIAS as _CANON_DOMAIN_BIAS,
    )
    PARADIGM_KEYWORDS = _CANON_KEYWORDS
    DOMAIN_PARADIGM_BIAS = _CANON_DOMAIN_BIAS

    def __init__(self):
        self.intent_patterns = self._compile_patterns()

    # ---------------------- helper utilities ----------------------

    @staticmethod
    def _to_paradigm(key: Any) -> Optional["HostParadigm"]:
        """Normalize a string or enum key to HostParadigm enum."""
        if isinstance(key, HostParadigm):
            return key
        if isinstance(key, str):
            k = key.strip().lower()
            mapping = {
                "dolores": HostParadigm.DOLORES,
                "revolutionary": HostParadigm.DOLORES,
                "teddy": HostParadigm.TEDDY,
                "devotion": HostParadigm.TEDDY,
                "bernard": HostParadigm.BERNARD,
                "analytical": HostParadigm.BERNARD,
                "maeve": HostParadigm.MAEVE,
                "strategic": HostParadigm.MAEVE,
            }
            return mapping.get(k)
        return None

    def _compile_patterns(self) -> Dict[HostParadigm, List[re.Pattern]]:
        """Compile regex patterns for efficiency with safety checks"""
        compiled: Dict[HostParadigm, List[re.Pattern]] = {}
        # Normalize keys so downstream lookups are reliable
        for raw_key, pattern_list in self._CANON_PATTERNS.items():
            paradigm = self._to_paradigm(raw_key)
            if paradigm is None:
                logger.warning("Unknown paradigm key in patterns", key=raw_key)
                continue
            bucket: List[re.Pattern] = []
            for p in pattern_list:
                safe_pattern = pattern_validator.safe_compile(p, re.IGNORECASE)
                if safe_pattern:
                    bucket.append(safe_pattern)
                else:
                    logger.warning(
                        f"Skipped unsafe pattern for {paradigm}: {p[:50]}..."
                    )
            compiled[paradigm] = bucket
        return compiled

    def analyze(self, query: str, research_id: Optional[str] = None) -> QueryFeatures:
        """Extract comprehensive features from query"""
        # Sanitize input using centralized security utility
        query = sanitize_user_input(query)
        logger.debug(f"Analyzing sanitized query (length={len(query)})")

        # Note: This runs in executor so can't use async progress tracking directly
        # But we can track via the main classify method
        tokens = self._tokenize(query)
        entities = self._extract_entities(query)
        intent_signals = self._detect_intent_signals(query)
        domain = self._identify_domain(query, entities)
        urgency = self._score_urgency(query, tokens)
        complexity = self._score_complexity(tokens, entities)
        emotion = self._score_emotional_valence(tokens)

        return QueryFeatures(
            text=query,
            tokens=tokens,
            entities=entities,
            intent_signals=intent_signals,
            domain=domain,
            urgency_score=urgency,
            complexity_score=complexity,
            emotional_valence=emotion,
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 2]

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (simplified)"""
        entities = []
        words = query.split()
        for i, word in enumerate(words):
            if word[0].isupper() and i > 0:
                entities.append(word)
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        return entities

    def _detect_intent_signals(self, query: str) -> List[str]:
        """Detect query intent patterns"""
        signals = []
        query_lower = query.lower()

        # Check for question types - use 'in' for more flexible matching
        if "how to" in query_lower or query_lower.startswith("how can"):
            signals.append("how_to")
        if query_lower.startswith("why"):
            signals.append("why_question")
        if query_lower.startswith("what"):
            signals.append("what_question")
        if query_lower.startswith("should"):
            signals.append("advice_seeking")

        action_words = ["create", "build", "stop", "prevent", "improve", "help"]
        for word in action_words:
            # Use word boundary matching to prevent false positives and injection
            if re.search(rf'\b{re.escape(word)}\b', query_lower):
                signals.append(f"action_{word}")
        return signals

    def _identify_domain(self, query: str, entities: List[str]) -> Optional[str]:
        """Identify query domain"""
        query_lower = query.lower()
        domain_keywords = {
            "technology": [
                "technology",
                "software",
                "ai",
                "digital",
                "cyber",
                "machine learning",
            ],
            "business": [
                "business",
                "company",
                "market",
                "profit",
                "revenue",
                "strategy",
            ],
            "healthcare": ["health", "medical", "patient", "doctor", "disease", "care"],
            "education": ["education", "school", "student", "learning", "teach"],
            "social_justice": [
                "justice",
                "rights",
                "equality",
                "discrimination",
                "injustice",
            ],
            "science": ["research", "study", "experiment", "scientific", "climate"],
            "nonprofit": ["charity", "volunteer", "nonprofit", "community service"],
        }
        # Check for exact matches first
        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if kw == "ai":
                    if re.search(r"\bai\b", query_lower):
                        return domain
                else:
                    if kw in query_lower:
                        return domain
        return None

    def _score_urgency(self, query: str, tokens: List[str]) -> float:
        """Score query urgency (0-1)"""
        urgency_indicators = [
            "urgent",
            "urgently",
            "immediately",
            "now",
            "asap",
            "quickly",
            "emergency",
            "critical",
            "crisis",
            "desperate",
        ]
        score = sum(1 for word in tokens if word in urgency_indicators)
        return min(score / 3.0, 1.0)

    def _score_complexity(self, tokens: List[str], entities: List[str]) -> float:
        """Score query complexity (0-1)"""
        length_score = min(len(tokens) / 20.0, 1.0)
        entity_score = min(len(entities) / 5.0, 1.0)
        complex_words = [t for t in tokens if len(t) > 8]
        complex_score = min(len(complex_words) / 5.0, 1.0)
        return (length_score + entity_score + complex_score) / 3.0

    def _score_emotional_valence(self, tokens: List[str]) -> float:
        """Score emotional content (-1 to 1)"""
        positive_words = [
            "help",
            "support",
            "love",
            "care",
            "hope",
            "success",
            "happy",
            "grateful",
            "inspire",
            "encourage",
        ]
        negative_words = [
            "fight",
            "attack",
            "hate",
            "angry",
            "unfair",
            "wrong",
            "corrupt",
            "evil",
            "destroy",
            "victim",
        ]
        pos_count = sum(1 for t in tokens if t in positive_words)
        neg_count = sum(1 for t in tokens if t in negative_words)
        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)


# --- Paradigm Classifier (To be implemented) ---


class ParadigmClassifier:
    """Classifies queries into host paradigms using hybrid approach"""

    def __init__(self, analyzer: QueryAnalyzer, use_llm: bool = True):
        self.analyzer = analyzer
        self.use_llm = use_llm
        self.rule_weight = 0.6
        self.llm_weight = 0.4 if use_llm else 0.0
        self.domain_weight = 0.2

    async def classify(self, query: str, research_id: Optional[str] = None) -> ClassificationResult:
        """Perform complete classification with confidence scoring"""
        start_time = time.time()

        logger.info(
            "Starting paradigm classification",
            stage="paradigm_classification_start",
            research_id=research_id,
            query_preview="[redacted]",
            config={"use_llm": self.use_llm, "query_length": len(query)},
        )
        import asyncio
        
        # Get progress tracker if available
        progress_tracker = None
        if research_id:
            try:
                from services.progress import progress as _pt
                progress_tracker = _pt
            except Exception:
                progress_tracker = None
        
        # Track classification steps
        total_steps = 4  # features, rule-based, LLM (optional), combination
        
        # Structured input snapshot (sanitized later in analyzer)
        try:
            logger.info(
                "Query classification input",
                stage="classification_input",
                research_id=research_id,
                query_length=len(query or ""),
                has_special_chars=bool(re.search(r"[^\w\s]", query or "")),
                query_hash=hashlib.md5((query or "").encode("utf-8", errors="ignore")).hexdigest()[:8],
            )
        except Exception:
            pass

        # Run feature extraction in executor to avoid blocking
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="classification",
                message="Extracting query features",
                items_done=0,
                items_total=total_steps
            )
        loop = asyncio.get_running_loop()
        features_task = loop.run_in_executor(None, self.analyzer.analyze, query)
        
        # Start LLM classification later after features are available
        llm_task = None
        
        # Wait for features
        feature_start = time.time()
        features = await features_task

        # Launch LLM classification concurrently using sanitized query text
        if self.use_llm:
            sanitized_query = features.text if features else query
            llm_task = asyncio.create_task(
                self._llm_classification(sanitized_query, features)
            )

        # Emit a compact, structured features snapshot to aid debugging
        try:
            qwords = [w for w in (features.tokens or []) if w in {"who", "what", "when", "where", "why", "how", "which"}][:6]
            actions = [s.replace("action_", "") for s in (features.intent_signals or []) if s.startswith("action_")][:6]
            logger.info(
                "Feature extraction complete",
                stage="features_extracted",
                research_id=research_id,
                features={
                    "question_words": qwords,
                    "action_verbs": actions,
                    "domain_indicators": [features.domain] if getattr(features, "domain", None) else [],
                    "sentiment": round(float(getattr(features, "emotional_valence", 0.0) or 0.0), 3),
                },
            )
        except Exception:
            pass

        logger.info(
            "Feature extraction completed",
            stage="feature_extraction",
            research_id=research_id,
            duration_ms=(time.time() - feature_start) * 1000,
            metrics={
                "intent_signals": len(features.intent_signals) if features else 0,
                "themes": len(getattr(features, 'themes', []) or []),
                "domain_indicators": len(getattr(features, 'domain_indicators', []) or []),
            },
        )
        
        # Run rule-based classification
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="classification",
                message="Applying rule-based classification",
                items_done=1,
                items_total=total_steps
            )

        rule_start = time.time()
        # Use sanitized text from features to ensure consistent inputs
        rule_scores = self._rule_based_classification(features.text, features)

        logger.info(
            "Rule-based classification completed",
            stage="rule_based_classification",
            research_id=research_id,
            duration_ms=(time.time() - rule_start) * 1000,
            paradigm_scores={
                p.value: {"score": s.score, "keywords": len(s.keyword_matches)}
                for p, s in rule_scores.items()
            },
        )
        
        # Wait for LLM results if started
        llm_scores = {}
        if llm_task:
            if progress_tracker and research_id:
                await progress_tracker.update_progress(
                    research_id,
                    phase="classification",
                    message="Running LLM classification",
                    items_done=2,
                    items_total=total_steps
                )
            try:
                llm_start = time.time()
                llm_scores = await llm_task

                logger.info(
                    "LLM classification completed",
                    stage="llm_classification",
                    research_id=research_id,
                    duration_ms=(time.time() - llm_start) * 1000,
                    paradigm_scores={
                        p.value: {"score": s.score} for p, s in (llm_scores or {}).items()
                    },
                )
            except Exception as e:
                logger.error(
                    "LLM classification failed",
                    stage="llm_classification_error",
                    research_id=research_id,
                    error_type=type(e).__name__,
                    stack_trace=traceback.format_exc(),
                )
                llm_scores = {}

        # Surface disagreement between rule-based and LLM tops (if any)
        try:
            if llm_scores:
                rule_top, rule_top_score = max(((p, s.score) for p, s in rule_scores.items()), key=lambda x: x[1])
                llm_top, llm_top_score = max(((p, s.score) for p, s in llm_scores.items()), key=lambda x: x[1])
                if rule_top != llm_top:
                    # Normalize to 0..1 for a comparable delta
                    delta = abs((rule_top_score/10.0) - (llm_top_score/10.0))
                    logger.warning(
                        "Paradigm classification disagreement",
                        stage="classification_conflict",
                        research_id=research_id,
                        rule_based=rule_top.value,
                        llm_based=llm_top.value,
                        confidence_delta=round(delta, 3),
                    )
        except Exception:
            pass

        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="classification",
                message="Combining classification scores",
                items_done=3,
                items_total=total_steps
            )
        final_scores = self._combine_scores(rule_scores, llm_scores, features)
        distribution = self._normalize_scores(final_scores)

        sorted_paradigms = sorted(
            distribution.items(), key=lambda x: x[1], reverse=True
        )
        primary = sorted_paradigms[0][0]
        secondary = (
            sorted_paradigms[1][0]
            if len(sorted_paradigms) > 1 and sorted_paradigms[1][1] > 0.2
            else None
        )

        confidence = self._calculate_confidence(distribution, final_scores)
        reasoning = {p: scores.reasoning for p, scores in final_scores.items()}
        
        # Mark classification complete
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="classification",
                message=f"Classification complete: {primary.value}",
                items_done=total_steps,
                items_total=total_steps
            )
        # Build structured signals for downstream UI
        signals: Dict[HostParadigm, Dict[str, Any]] = {}
        for p, score_obj in final_scores.items():
            try:
                kw = list(score_obj.keyword_matches or [])
            except Exception:
                kw = []
            signals[p] = {
                "keywords": kw,
                # Provide global intent signals for context
                "intent_signals": list(features.intent_signals or []),
            }

        total_duration = (time.time() - start_time) * 1000

        logger.info(
            "Classification completed",
            stage="paradigm_classification_complete",
            research_id=research_id,
            duration_ms=total_duration,
            paradigm_scores={
                "primary": primary.value,
                "secondary": secondary.value if secondary else None,
                "confidence": confidence,
                "distribution": {p.value: score for p, score in distribution.items()},
            },
            metrics={
                "total_keywords_matched": sum(len(s.keyword_matches) for s in final_scores.values()),
                "llm_used": bool(llm_scores),
                "features_extracted": bool(features),
            },
        )

        return ClassificationResult(
            query=query,
            primary_paradigm=primary,
            secondary_paradigm=secondary,
            distribution=distribution,
            confidence=confidence,
            features=features,
            reasoning=reasoning,
            signals=signals,
        )

    def _rule_based_classification(
        self, query: str, features: QueryFeatures
    ) -> Dict[HostParadigm, ParadigmScore]:
        """Rule-based classification using keywords and patterns"""
        logger.debug(
            "Starting rule-based classification",
            stage="rule_classification_detail",
            query_length=len(query),
        )

        scores = {}
        query_lower = query.lower()

        for paradigm in HostParadigm:
            keyword_score = 0.0
            pattern_score = 0.0
            matched_keywords = []
            reasoning = []

            # Canonical keywords may be a flat list; keep backward compat if a dict is provided
            canon = self.analyzer.PARADIGM_KEYWORDS.get(paradigm)
            if isinstance(canon, dict):
                primary_kw = canon.get("primary", [])
                for kw in primary_kw:
                    if kw in query_lower:
                        keyword_score += 2.0
                        matched_keywords.append(kw)
                secondary_kw = canon.get("secondary", [])
                for kw in secondary_kw:
                    if kw in query_lower:
                        keyword_score += 1.0
                        matched_keywords.append(kw)
            else:
                for kw in (canon or []):
                    if kw in query_lower:
                        keyword_score += 1.5
                        matched_keywords.append(kw)

            patterns = self.analyzer.intent_patterns.get(paradigm, [])
            for pattern in patterns:
                if pattern.search(query):
                    pattern_score += 3.0
                    reasoning.append(f"Matches pattern: {pattern.pattern}")

            intent_bonus = self._score_intent_alignment(
                paradigm, features.intent_signals
            )
            total_score = keyword_score + pattern_score + intent_bonus

            if matched_keywords:
                reasoning.append(f"Keywords: {', '.join(matched_keywords[:5])}")
            if intent_bonus > 0:
                reasoning.append(f"Intent alignment: {intent_bonus:.1f}")

            scores[paradigm] = ParadigmScore(
                paradigm=paradigm,
                score=total_score,
                confidence=min(total_score / 10.0, 1.0),
                reasoning=reasoning,
                keyword_matches=matched_keywords,
            )
        return scores

    def _score_intent_alignment(
        self, paradigm: HostParadigm, intent_signals: List[str]
    ) -> float:
        """Score how well intent signals align with paradigm"""
        alignments = {
            HostParadigm.DOLORES: ["action_stop", "action_prevent", "why_question"],
            HostParadigm.TEDDY: ["action_help", "how_to", "advice_seeking"],
            HostParadigm.BERNARD: ["what_question", "why_question"],
            HostParadigm.MAEVE: ["how_to", "action_create", "action_improve"],
        }
        paradigm_intents = alignments.get(paradigm, [])
        matches = sum(1 for signal in intent_signals if signal in paradigm_intents)
        return matches * 1.5

    def _combine_scores(
        self,
        rule_scores: Dict[HostParadigm, ParadigmScore],
        llm_scores: Dict[HostParadigm, ParadigmScore],
        features: QueryFeatures,
    ) -> Dict[HostParadigm, ParadigmScore]:
        """Combine rule-based and LLM scores with domain bias"""
        combined = {}
        for paradigm in HostParadigm:
            rule_score = rule_scores.get(
                paradigm,
                ParadigmScore(
                    paradigm=paradigm,
                    score=0,
                    confidence=0,
                    reasoning=[],
                    keyword_matches=[],
                ),
            )
            total_score = rule_score.score * self.rule_weight

            if paradigm in llm_scores:
                llm_score = llm_scores[paradigm]
                total_score += llm_score.score * self.llm_weight

            if features.domain:
                domain_paradigm = self.analyzer.DOMAIN_PARADIGM_BIAS.get(
                    features.domain
                )
                if domain_paradigm == paradigm:
                    total_score *= 1 + self.domain_weight

            all_reasoning = rule_score.reasoning.copy()
            if paradigm in llm_scores:
                all_reasoning.extend(llm_scores[paradigm].reasoning)
            if (
                features.domain
                and self.analyzer.DOMAIN_PARADIGM_BIAS.get(features.domain) == paradigm
            ):
                all_reasoning.append(f"Domain match: {features.domain}")

            combined[paradigm] = ParadigmScore(
                paradigm=paradigm,
                score=total_score,
                confidence=(
                    (
                        rule_score.confidence
                        + llm_scores.get(paradigm, rule_score).confidence
                    )
                    / 2
                    if llm_scores
                    else rule_score.confidence
                ),
                reasoning=all_reasoning,
                keyword_matches=rule_score.keyword_matches,
            )
        return combined

    def _normalize_scores(
        self, scores: Dict[HostParadigm, ParadigmScore]
    ) -> Dict[HostParadigm, float]:
        """Normalize scores to probability distribution"""
        total = sum(s.score for s in scores.values())
        if total == 0:
            return {p: 0.25 for p in HostParadigm}
        return {p: s.score / total for p, s in scores.items()}

    def _calculate_confidence(
        self,
        distribution: Dict[HostParadigm, float],
        scores: Dict[HostParadigm, ParadigmScore],
    ) -> float:
        """Calculate overall classification confidence"""
        sorted_probs = sorted(distribution.values(), reverse=True)
        spread = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0)
        top_paradigm = max(distribution.items(), key=lambda x: x[1])[0]
        top_score_confidence = scores[top_paradigm].confidence
        confidence = spread * 0.5 + top_score_confidence * 0.5
        return min(confidence, 0.95)

    async def _llm_classification(
        self, query: str, features: Optional[QueryFeatures]
    ) -> Dict[HostParadigm, ParadigmScore]:
        """Use LLM to classify the query into paradigms"""
        def _repair_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
            """Attempt to repair common LLM JSON issues and parse.

            Handles:
            - Markdown code fences (```json ... ```)
            - Leading/trailing text around a JSON object
            - Trailing commas before } or ]
            - Smart quotes → standard quotes
            - Unbalanced braces by appending missing closing braces
            - Single-quoted keys -> double-quoted keys
            """
            if not text or not isinstance(text, str):
                return None

            s = text.strip()
            # Strip Markdown code fences
            if s.startswith("```"):
                s = s.lstrip("`")
                # Remove an optional language tag like json, JSON
                s = s[s.find("\n") + 1 :] if "\n" in s else s
                if s.endswith("```"):
                    s = s[: -3]
                s = s.strip()

            # Extract the largest {...} block to avoid pre/post commentary
            first = s.find("{")
            last = s.rfind("}")
            if first != -1 and last != -1 and last > first:
                s = s[first : last + 1]

            # Normalise quotes
            s = (
                s.replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2018", "'")
                .replace("\u2019", "'")
            )

            # Single-quoted keys to double quotes: 'key': → "key":
            import re as _re
            s = _re.sub(r"'([A-Za-z0-9_]+)'\s*:", r'"\1":', s)

            # Trailing commas before } or ]
            s = _re.sub(r",\s*([}\]])", r"\1", s)

            # Balance braces if obviously short by a few closers
            open_braces = s.count("{")
            close_braces = s.count("}")
            if open_braces > close_braces:
                s = s + ("}" * (open_braces - close_braces))

            # Also balance brackets
            open_brackets = s.count("[")
            close_brackets = s.count("]")
            if open_brackets > close_brackets:
                s = s + ("]" * (open_brackets - close_brackets))

            try:
                return json.loads(s)
            except json.JSONDecodeError as e:
                # Try one more time with additional closing brace
                # (common issue when LLM truncates output)
                if "Expecting" in str(e) and open_braces > 0:
                    try:
                        return json.loads(s + "}")
                    except:
                        pass
                return None
            except Exception:
                return None
        # Create a prompt for the LLM
        features_text = ""
        if features:
            features_text = f"""
Extracted Features:
- Domain: {features.domain or 'General'}
- Intent Signals: {', '.join(features.intent_signals) if features.intent_signals else 'None'}
- Urgency Score: {features.urgency_score:.2f}
- Complexity Score: {features.complexity_score:.2f}
- Emotional Valence: {features.emotional_valence:.2f}"""
        
        prompt = f"""Analyze this query and determine which host paradigm(s) it best aligns with.

Query: "{query}"
{features_text}

Host Paradigms:
1. DOLORES (Revolutionary): Focuses on exposing injustices, fighting oppression, revealing truth
2. TEDDY (Devotion): Emphasizes helping, supporting, protecting vulnerable populations
3. BERNARD (Analytical): Seeks data, evidence, research, objective analysis
4. MAEVE (Strategic): Pursues optimization, competitive advantage, actionable strategies

For each paradigm, provide:
1. A score from 0-10 indicating alignment
2. Brief reasoning (1-2 sentences)

Return as JSON with this structure:
{{
  "dolores": {{"score": 0-10, "reasoning": "..."}},
  "teddy": {{"score": 0-10, "reasoning": "..."}},
  "bernard": {{"score": 0-10, "reasoning": "..."}},
  "maeve": {{"score": 0-10, "reasoning": "..."}}
}}"""

        try:
            # Call LLM with structured output
            if not llm_client:
                logger.warning("LLM client not available")
                return {}
            

            # Generate completion returns a string
            logger.info(f"Sending prompt to LLM for classification, research_id: {getattr(features, 'research_id', None)}")
            timeout_sec = float(os.getenv("CLASSIFICATION_LLM_TIMEOUT_SEC", "20"))
            response_payload = await asyncio.wait_for(
                llm_client.generate_completion(
                    prompt=prompt,
                    paradigm="bernard",  # analytical model/profile
                    json_schema=CLASSIFICATION_JSON_SCHEMA,
                ),
                timeout=timeout_sec,
            )

            # Log structured metadata without exposing raw model output
            present_keys = list(response_payload.keys()) if isinstance(response_payload, dict) else []
            logger.info(
                "LLM response received",
                research_id=getattr(features, "research_id", None),
                keys_present=present_keys,
            )

            # Parse the JSON response with robust error handling/repair
            llm_result: Optional[Dict[str, Any]]
            if isinstance(response_payload, dict):
                llm_result = response_payload
            else:
                response_text = str(response_payload).strip()
                try:
                    llm_result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    snippet = response_text[:1024]
                    logger.error(f"Raw response that failed to parse (≤1kB): {snippet}")
                    llm_result = _repair_and_parse_json(response_text)
                    if llm_result is None:
                        logger.warning("Falling back to rule-based classification after LLM parse failure")
                        return {}

            # Validate the structure
            if not isinstance(llm_result, dict):
                logger.error(f"LLM response is not a dictionary: {type(llm_result)}")
                return {}

            # Convert to ParadigmScore objects
            scores = {}
            paradigm_map = {
                "dolores": HostParadigm.DOLORES,
                "teddy": HostParadigm.TEDDY,
                "bernard": HostParadigm.BERNARD,
                "maeve": HostParadigm.MAEVE,
            }

            for key, paradigm in paradigm_map.items():
                if key in llm_result and isinstance(llm_result[key], dict):
                    score_data = llm_result[key]
                    try:
                        score = float(score_data.get("score", 0))
                        # Clamp score to valid range
                        score = max(0, min(10, score))
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid score for {key}: {score_data.get('score')}")
                        score = 0
                    
                    reasoning = str(score_data.get("reasoning", ""))

                    scores[paradigm] = ParadigmScore(
                        paradigm=paradigm,
                        score=score,
                        confidence=min(score / 10.0, 1.0),  # Normalize to 0-1
                        reasoning=[f"LLM: {reasoning}"] if reasoning else [],
                        keyword_matches=[],
                    )

            return scores

        except Exception as e:
            logger.error(
                "Error in LLM classification", error=str(e), exc_info=True
            )
            # Return empty scores on error
            return {}


# --- Classification Manager ---


class ClassificationEngine:
    """Main interface for the classification system"""

    def __init__(
        self,
        use_llm: bool = True,
        cache_enabled: bool = True,
        cache_size: int = 512,
        cache_ttl_sec: int = 0,
    ):
        self.analyzer = QueryAnalyzer()
        # Try to import LLM client lazily
        global llm_client, LLM_AVAILABLE
        if use_llm and not LLM_AVAILABLE:
            try:
                from .llm_client import llm_client as _llm_client
                llm_client = _llm_client
                LLM_AVAILABLE = True
                logger.info("LLM client successfully imported")
            except ImportError as e:
                logger.warning(
                    f"LLM client not available - using rule-based classification: {e}"
                )
                llm_client = None
                LLM_AVAILABLE = False
        
        actual_use_llm = use_llm and LLM_AVAILABLE
        self.classifier = ParadigmClassifier(self.analyzer, actual_use_llm)

        self.cache_enabled = cache_enabled
        self._cache_size = max(16, int(cache_size))
        self._cache_ttl = max(0, int(cache_ttl_sec))
        # key -> (timestamp, result)
        self.cache: "OrderedDict[str, tuple[float, ClassificationResult]]" = (
            OrderedDict() if cache_enabled else None
        )

    # ---------------- Cache helpers ----------------

    @staticmethod
    def _cache_key(raw_query: str) -> str:
        """Generate a stable HMAC-SHA256 key for the query."""
        try:
            cleaned = sanitize_user_input(raw_query or "")
        except Exception:
            cleaned = (raw_query or "").strip()
        secret = os.getenv("CLASSIFICATION_CACHE_HMAC_KEY", "dev-secret-key").encode()
        return hmac.new(secret, cleaned.encode("utf-8", "ignore"), _hash.sha256).hexdigest()[:32]

    async def classify_query(self, query: str, research_id: Optional[str] = None) -> ClassificationResult:
        """Classify a query with caching and logging"""
        # Input validation
        if not query or len(query) > 10000:
            raise ValueError("Query must be between 1 and 10000 characters")

        logger.info(
            "Starting query classification",
            stage="classification_start",
            research_id=research_id,
            query_length=len(query),
            query_preview="[redacted]",
            config={"use_llm": self.classifier.use_llm, "cache_enabled": self.cache_enabled},
        )

        from time import perf_counter
        start = perf_counter()
        cache_hit = False

        key = self._cache_key(query)
        now = time.time()

        if self.cache_enabled and key in self.cache:
            logger.info(
                "Classification cache hit",
                stage="classification_cache",
                research_id=research_id,
                cache_hit=True,
            )
            cache_hit = True
            ts, result = self.cache.pop(key)
            # TTL check
            if self._cache_ttl and (now - ts) > self._cache_ttl:
                cache_hit = False
            else:
                # reinsert to update LRU order
                self.cache[key] = (ts, result)
        else:
            logger.info(
                "Performing new classification",
                stage="classification_processing",
                research_id=research_id,
                method=("llm" if self.classifier.use_llm else "rule_based"),
            )

            try:
                result = await self.classifier.classify(query, research_id)
                if self.cache_enabled:
                    self.cache[key] = (now, result)
                    # enforce size
                    while len(self.cache) > self._cache_size:
                        self.cache.popitem(last=False)
            except Exception as e:
                logger.error(
                    "Classification failed",
                    stage="classification_error",
                    research_id=research_id,
                    error_type=type(e).__name__,
                    stack_trace=traceback.format_exc(),
                )
                raise

        duration_ms = (perf_counter() - start) * 1000.0

        logger.info(
            "Classification completed",
            stage="classification_complete",
            research_id=research_id,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            paradigm_scores={
                "primary": result.primary_paradigm.value if result else None,
                "secondary": (result.secondary_paradigm.value if (result and result.secondary_paradigm) else None),
                "confidence": (result.confidence if result else None),
            },
            metrics={
                "signals_count": (len(result.signals) if result else 0),
                "reasoning_count": (sum(len(r) for r in result.reasoning.values()) if result else 0),
            },
        )
        # Metrics recording (best-effort, ignore failures)
        try:
            from .metrics import metrics
            metrics.record_stage(
                stage="classification",
                duration_ms=duration_ms,
                paradigm=result.primary_paradigm.value if result else None,
                success=True,
                fallback=not self.classifier.use_llm,
                model="llm" if self.classifier.use_llm else None,
            )
            if cache_hit:
                metrics.increment("classification_cache_hit")
        except Exception:
            pass
        return result


# Global instance
classification_engine = ClassificationEngine()

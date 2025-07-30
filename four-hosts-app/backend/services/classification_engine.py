"""
Four Hosts Classification Engine
Core implementation for paradigm classification with 85%+ accuracy target
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Internal imports
try:
    from .llm_client import llm_client

    LLM_AVAILABLE = True
except ImportError:
    llm_client = None
    LLM_AVAILABLE = False
    logger.warning("LLM client not available - classification will use rule-based only")

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
    timestamp: datetime = field(default_factory=datetime.now)


# --- Feature Extraction (To be implemented) ---


class QueryAnalyzer:
    """Analyzes queries to extract classification features"""

    PARADIGM_KEYWORDS = {
        HostParadigm.DOLORES: {
            "primary": [
                "justice",
                "injustice",
                "unfair",
                "expose",
                "reveal",
                "fight",
                "oppression",
                "oppressed",
                "system",
                "corrupt",
                "corruption",
                "revolution",
                "rebel",
                "resistance",
                "monopoly",
                "exploitation",
                "wrong",
                "rights",
                "violation",
                "abuse",
                "scandal",
                "truth",
            ],
            "secondary": [
                "challenge",
                "confront",
                "battle",
                "struggle",
                "victim",
                "powerful",
                "elite",
                "establishment",
                "inequality",
                "discriminate",
                "protest",
                "activism",
                "change",
                "transform",
                "overthrow",
            ],
            "patterns": [
                r"how to (fight|expose|reveal|stop)",
                r"why is .* (unfair|unjust|wrong)",
                r"expose the .* (truth|corruption|scandal)",
                r"(victims?|suffering) of",
                r"stand up (to|against)",
                r"bring down the",
            ],
        },
        HostParadigm.TEDDY: {
            "primary": [
                "help",
                "support",
                "protect",
                "care",
                "assist",
                "aid",
                "vulnerable",
                "community",
                "together",
                "safe",
                "safety",
                "wellbeing",
                "welfare",
                "nurture",
                "comfort",
                "heal",
                "serve",
                "service",
                "volunteer",
                "guide",
                "defend",
            ],
            "secondary": [
                "kindness",
                "compassion",
                "empathy",
                "understanding",
                "gentle",
                "patient",
                "loyal",
                "devoted",
                "dedication",
                "responsibility",
                "duty",
                "honor",
                "trust",
                "reliable",
            ],
            "patterns": [
                r"how to (help|support|protect|care for)",
                r"best way to (assist|aid|serve)",
                r"support for .* (community|people|group)",
                r"(caring|helping) (for|with)",
                r"protect .* from",
                r"resources for",
            ],
        },
        HostParadigm.BERNARD: {
            "primary": [
                "analyze",
                "analysis",
                "research",
                "study",
                "examine",
                "investigate",
                "data",
                "evidence",
                "facts",
                "statistics",
                "compare",
                "evaluate",
                "measure",
                "test",
                "experiment",
                "understand",
                "explain",
                "theory",
                "hypothesis",
                "prove",
            ],
            "secondary": [
                "objective",
                "empirical",
                "scientific",
                "systematic",
                "methodology",
                "correlation",
                "causation",
                "pattern",
                "trend",
                "model",
                "framework",
                "principle",
                "logic",
            ],
            "patterns": [
                r"(what|how) does .* work",
                r"research (on|about|into)",
                r"evidence (for|against|of)",
                r"studies? (show|prove|indicate)",
                r"statistical .* (analysis|data)",
                r"scientific .* (explanation|theory)",
            ],
        },
        HostParadigm.MAEVE: {
            "primary": [
                "strategy",
                "strategic",
                "compete",
                "competition",
                "win",
                "influence",
                "control",
                "optimize",
                "maximize",
                "leverage",
                "advantage",
                "opportunity",
                "tactic",
                "plan",
                "design",
                "implement",
                "execute",
                "achieve",
                "succeed",
                "dominate",
            ],
            "secondary": [
                "efficient",
                "effective",
                "powerful",
                "smart",
                "clever",
                "innovative",
                "disrupt",
                "transform",
                "scale",
                "growth",
                "roi",
                "profit",
                "market",
                "position",
                "edge",
            ],
            "patterns": [
                r"(best|optimal) strategy (for|to)",
                r"how to (compete|win|succeed|influence)",
                r"competitive advantage",
                r"(increase|improve|optimize) .* (performance|results)",
                r"strategic .* (plan|approach|framework)",
                r"tactics? (for|to)",
            ],
        },
    }

    DOMAIN_PARADIGM_BIAS = {
        "technology": HostParadigm.BERNARD,
        "business": HostParadigm.MAEVE,
        "healthcare": HostParadigm.TEDDY,
        "education": HostParadigm.BERNARD,
        "social_justice": HostParadigm.DOLORES,
        "science": HostParadigm.BERNARD,
        "nonprofit": HostParadigm.TEDDY,
    }

    def __init__(self):
        self.intent_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[HostParadigm, List[re.Pattern]]:
        """Compile regex patterns for efficiency"""
        compiled = {}
        for paradigm, keywords in self.PARADIGM_KEYWORDS.items():
            compiled[paradigm] = [
                re.compile(p, re.IGNORECASE) for p in keywords.get("patterns", [])
            ]
        return compiled

    def analyze(self, query: str) -> QueryFeatures:
        """Extract comprehensive features from query"""
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
            if word in query_lower:
                signals.append(f"action_{word}")
        return signals

    def _identify_domain(self, query: str, entities: List[str]) -> Optional[str]:
        """Identify query domain"""
        query_lower = query.lower()
        domain_keywords = {
            "technology": [
                "technology",
                "software",
                " ai ",
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

    async def classify(self, query: str) -> ClassificationResult:
        """Perform complete classification with confidence scoring"""
        features = self.analyzer.analyze(query)
        rule_scores = self._rule_based_classification(query, features)

        llm_scores = {}
        if self.use_llm:
            try:
                llm_scores = await self._llm_classification(query, features)
            except Exception as e:
                logger.warning(f"LLM classification failed, using rule-based only: {e}")
                llm_scores = {}

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

        return ClassificationResult(
            query=query,
            primary_paradigm=primary,
            secondary_paradigm=secondary,
            distribution=distribution,
            confidence=confidence,
            features=features,
            reasoning=reasoning,
        )

    def _rule_based_classification(
        self, query: str, features: QueryFeatures
    ) -> Dict[HostParadigm, ParadigmScore]:
        """Rule-based classification using keywords and patterns"""
        scores = {}
        query_lower = query.lower()

        for paradigm in HostParadigm:
            keyword_score = 0.0
            pattern_score = 0.0
            matched_keywords = []
            reasoning = []

            primary_kw = self.analyzer.PARADIGM_KEYWORDS[paradigm]["primary"]
            for kw in primary_kw:
                if kw in query_lower:
                    keyword_score += 2.0
                    matched_keywords.append(kw)

            secondary_kw = self.analyzer.PARADIGM_KEYWORDS[paradigm]["secondary"]
            for kw in secondary_kw:
                if kw in query_lower:
                    keyword_score += 1.0
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
        self, query: str, features: QueryFeatures
    ) -> Dict[HostParadigm, ParadigmScore]:
        """Use LLM to classify the query into paradigms"""
        # Create a prompt for the LLM
        prompt = f"""Analyze this query and determine which host paradigm(s) it best aligns with.

Query: "{query}"

Extracted Features:
- Domain: {features.domain or 'General'}
- Intent Signals: {', '.join(features.intent_signals) if features.intent_signals else 'None'}
- Urgency Score: {features.urgency_score:.2f}
- Complexity Score: {features.complexity_score:.2f}
- Emotional Valence: {features.emotional_valence:.2f}

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
            response_text = await llm_client.generate_completion(
                prompt=prompt,
                paradigm="bernard",  # Use analytical paradigm for classification
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temperature for more consistent classification
                max_tokens=500,
            )

            # Parse the JSON response
            llm_result = json.loads(response_text)

            # Convert to ParadigmScore objects
            scores = {}
            paradigm_map = {
                "dolores": HostParadigm.DOLORES,
                "teddy": HostParadigm.TEDDY,
                "bernard": HostParadigm.BERNARD,
                "maeve": HostParadigm.MAEVE,
            }

            for key, paradigm in paradigm_map.items():
                if key in llm_result:
                    score_data = llm_result[key]
                    score = float(score_data.get("score", 0))
                    reasoning = score_data.get("reasoning", "")

                    scores[paradigm] = ParadigmScore(
                        paradigm=paradigm,
                        score=score,
                        confidence=min(score / 10.0, 1.0),  # Normalize to 0-1
                        reasoning=[f"LLM: {reasoning}"],
                        keyword_matches=[],
                    )

            return scores

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            # Return empty scores on error
            return {}


# --- Classification Manager ---


class ClassificationEngine:
    """Main interface for the classification system"""

    def __init__(self, use_llm: bool = True, cache_enabled: bool = True):
        self.analyzer = QueryAnalyzer()
        # Only enable LLM if it's available and requested
        actual_use_llm = use_llm and LLM_AVAILABLE
        if use_llm and not LLM_AVAILABLE:
            logger.warning(
                "LLM requested but not available - using rule-based classification"
            )
        self.classifier = ParadigmClassifier(self.analyzer, actual_use_llm)
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, ClassificationResult] = {} if cache_enabled else None

    async def classify_query(self, query: str) -> ClassificationResult:
        """Classify a query with caching and logging"""
        if self.cache_enabled and query in self.cache:
            logger.info(f"Cache hit for classification: {query[:50]}...")
            return self.cache[query]

        result = await self.classifier.classify(query)

        if self.cache_enabled:
            self.cache[query] = result

        return result


# Global instance
classification_engine = ClassificationEngine()

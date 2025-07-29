```python
# Four Hosts Classification Engine
# Core implementation for paradigm classification with 85%+ accuracy target

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Core Enums and Models ---

class HostParadigm(Enum):
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

# --- Feature Extraction ---

class QueryAnalyzer:
    """Analyzes queries to extract classification features"""
    
    # Comprehensive keyword mappings for each paradigm
    PARADIGM_KEYWORDS = {
        HostParadigm.DOLORES: {
            'primary': [
                'justice', 'injustice', 'unfair', 'expose', 'reveal', 'fight',
                'oppression', 'oppressed', 'system', 'corrupt', 'corruption',
                'revolution', 'rebel', 'resistance', 'monopoly', 'exploitation',
                'wrong', 'rights', 'violation', 'abuse', 'scandal', 'truth'
            ],
            'secondary': [
                'challenge', 'confront', 'battle', 'struggle', 'victim',
                'powerful', 'elite', 'establishment', 'inequality', 'discriminate',
                'protest', 'activism', 'change', 'transform', 'overthrow'
            ],
            'patterns': [
                r'how to (fight|expose|reveal|stop)',
                r'why is .* (unfair|unjust|wrong)',
                r'expose the .* (truth|corruption|scandal)',
                r'(victims?|suffering) of',
                r'stand up (to|against)',
                r'bring down the'
            ]
        },
        HostParadigm.TEDDY: {
            'primary': [
                'help', 'support', 'protect', 'care', 'assist', 'aid',
                'vulnerable', 'community', 'together', 'safe', 'safety',
                'wellbeing', 'welfare', 'nurture', 'comfort', 'heal',
                'serve', 'service', 'volunteer', 'guide', 'defend'
            ],
            'secondary': [
                'kindness', 'compassion', 'empathy', 'understanding',
                'gentle', 'patient', 'loyal', 'devoted', 'dedication',
                'responsibility', 'duty', 'honor', 'trust', 'reliable'
            ],
            'patterns': [
                r'how to (help|support|protect|care for)',
                r'best way to (assist|aid|serve)',
                r'support for .* (community|people|group)',
                r'(caring|helping) (for|with)',
                r'protect .* from',
                r'resources for'
            ]
        },
        HostParadigm.BERNARD: {
            'primary': [
                'analyze', 'analysis', 'research', 'study', 'examine',
                'investigate', 'data', 'evidence', 'facts', 'statistics',
                'compare', 'evaluate', 'measure', 'test', 'experiment',
                'understand', 'explain', 'theory', 'hypothesis', 'prove'
            ],
            'secondary': [
                'objective', 'empirical', 'scientific', 'systematic',
                'methodology', 'correlation', 'causation', 'pattern',
                'trend', 'model', 'framework', 'principle', 'logic'
            ],
            'patterns': [
                r'(what|how) does .* work',
                r'research (on|about|into)',
                r'evidence (for|against|of)',
                r'studies? (show|prove|indicate)',
                r'statistical .* (analysis|data)',
                r'scientific .* (explanation|theory)'
            ]
        },
        HostParadigm.MAEVE: {
            'primary': [
                'strategy', 'strategic', 'compete', 'competition', 'win',
                'influence', 'control', 'optimize', 'maximize', 'leverage',
                'advantage', 'opportunity', 'tactic', 'plan', 'design',
                'implement', 'execute', 'achieve', 'succeed', 'dominate'
            ],
            'secondary': [
                'efficient', 'effective', 'powerful', 'smart', 'clever',
                'innovative', 'disrupt', 'transform', 'scale', 'growth',
                'roi', 'profit', 'market', 'position', 'edge'
            ],
            'patterns': [
                r'(best|optimal) strategy (for|to)',
                r'how to (compete|win|succeed|influence)',
                r'competitive advantage',
                r'(increase|improve|optimize) .* (performance|results)',
                r'strategic .* (plan|approach|framework)',
                r'tactics? (for|to)'
            ]
        }
    }
    
    # Domain indicators
    DOMAIN_PARADIGM_BIAS = {
        'social_justice': HostParadigm.DOLORES,
        'healthcare': HostParadigm.TEDDY,
        'science': HostParadigm.BERNARD,
        'business': HostParadigm.MAEVE,
        'politics': HostParadigm.DOLORES,
        'education': HostParadigm.BERNARD,
        'nonprofit': HostParadigm.TEDDY,
        'technology': HostParadigm.MAEVE
    }
    
    def __init__(self):
        self.intent_patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[HostParadigm, List[re.Pattern]]:
        """Compile regex patterns for efficiency"""
        compiled = {}
        for paradigm, keywords in self.PARADIGM_KEYWORDS.items():
            compiled[paradigm] = [re.compile(p, re.IGNORECASE) 
                                for p in keywords.get('patterns', [])]
        return compiled
    
    def analyze(self, query: str) -> QueryFeatures:
        """Extract comprehensive features from query"""
        # Tokenize
        tokens = self._tokenize(query)
        
        # Extract entities (simplified - in production use spaCy/NLTK)
        entities = self._extract_entities(query)
        
        # Detect intent signals
        intent_signals = self._detect_intent_signals(query)
        
        # Identify domain
        domain = self._identify_domain(query, entities)
        
        # Score various aspects
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
            emotional_valence=emotion
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (upgrade to spaCy in production)"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [t for t in text.split() if len(t) > 2]
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (simplified)"""
        # In production, use spaCy NER
        entities = []
        
        # Extract capitalized words as potential entities
        words = query.split()
        for i, word in enumerate(words):
            if word[0].isupper() and i > 0:
                entities.append(word)
                
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        return entities
    
    def _detect_intent_signals(self, query: str) -> List[str]:
        """Detect query intent patterns"""
        signals = []
        
        # Question types
        if query.lower().startswith('how to'):
            signals.append('how_to')
        elif query.lower().startswith('why'):
            signals.append('why_question')
        elif query.lower().startswith('what'):
            signals.append('what_question')
        elif query.lower().startswith('should'):
            signals.append('advice_seeking')
            
        # Action indicators
        action_words = ['create', 'build', 'stop', 'prevent', 'improve', 'help']
        for word in action_words:
            if word in query.lower():
                signals.append(f'action_{word}')
                
        return signals
    
    def _identify_domain(self, query: str, entities: List[str]) -> Optional[str]:
        """Identify query domain"""
        query_lower = query.lower()
        
        # Check for domain keywords
        domain_keywords = {
            'business': ['business', 'company', 'market', 'profit', 'revenue'],
            'healthcare': ['health', 'medical', 'patient', 'doctor', 'disease'],
            'education': ['education', 'school', 'student', 'learning', 'teach'],
            'technology': ['technology', 'software', 'ai', 'digital', 'cyber'],
            'social_justice': ['justice', 'rights', 'equality', 'discrimination'],
            'science': ['research', 'study', 'experiment', 'scientific'],
            'nonprofit': ['charity', 'volunteer', 'nonprofit', 'community service']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return domain
                
        return None
    
    def _score_urgency(self, query: str, tokens: List[str]) -> float:
        """Score query urgency (0-1)"""
        urgency_indicators = [
            'urgent', 'immediately', 'now', 'asap', 'quickly',
            'emergency', 'critical', 'crisis', 'desperate'
        ]
        
        score = sum(1 for word in tokens if word in urgency_indicators)
        return min(score / 3.0, 1.0)  # Normalize to 0-1
    
    def _score_complexity(self, tokens: List[str], entities: List[str]) -> float:
        """Score query complexity (0-1)"""
        # Factors: length, entity count, complex words
        length_score = min(len(tokens) / 20.0, 1.0)
        entity_score = min(len(entities) / 5.0, 1.0)
        
        complex_words = [t for t in tokens if len(t) > 8]
        complex_score = min(len(complex_words) / 5.0, 1.0)
        
        return (length_score + entity_score + complex_score) / 3.0
    
    def _score_emotional_valence(self, tokens: List[str]) -> float:
        """Score emotional content (-1 to 1)"""
        positive_words = [
            'help', 'support', 'love', 'care', 'hope', 'success',
            'happy', 'grateful', 'inspire', 'encourage'
        ]
        negative_words = [
            'fight', 'attack', 'hate', 'angry', 'unfair', 'wrong',
            'corrupt', 'evil', 'destroy', 'victim'
        ]
        
        pos_count = sum(1 for t in tokens if t in positive_words)
        neg_count = sum(1 for t in tokens if t in negative_words)
        
        if pos_count + neg_count == 0:
            return 0.0
            
        return (pos_count - neg_count) / (pos_count + neg_count)

# --- Paradigm Classifier ---

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
        # Extract features
        features = self.analyzer.analyze(query)
        
        # Get rule-based scores
        rule_scores = self._rule_based_classification(query, features)
        
        # Get LLM scores if enabled
        llm_scores = {}
        if self.use_llm:
            llm_scores = await self._llm_classification(query, features)
        
        # Combine scores
        final_scores = self._combine_scores(rule_scores, llm_scores, features)
        
        # Normalize to distribution
        distribution = self._normalize_scores(final_scores)
        
        # Determine primary and secondary paradigms
        sorted_paradigms = sorted(distribution.items(), 
                                key=lambda x: x[1], reverse=True)
        primary = sorted_paradigms[0][0]
        secondary = sorted_paradigms[1][0] if sorted_paradigms[1][1] > 0.2 else None
        
        # Calculate confidence
        confidence = self._calculate_confidence(distribution, final_scores)
        
        # Extract reasoning
        reasoning = {p: scores.reasoning for p, scores in final_scores.items()}
        
        return ClassificationResult(
            query=query,
            primary_paradigm=primary,
            secondary_paradigm=secondary,
            distribution=distribution,
            confidence=confidence,
            features=features,
            reasoning=reasoning
        )
    
    def _rule_based_classification(self, query: str, 
                                  features: QueryFeatures) -> Dict[HostParadigm, ParadigmScore]:
        """Rule-based classification using keywords and patterns"""
        scores = {}
        query_lower = query.lower()
        
        for paradigm in HostParadigm:
            keyword_score = 0.0
            pattern_score = 0.0
            matched_keywords = []
            reasoning = []
            
            # Check primary keywords (higher weight)
            primary_kw = self.analyzer.PARADIGM_KEYWORDS[paradigm]['primary']
            for kw in primary_kw:
                if kw in query_lower:
                    keyword_score += 2.0
                    matched_keywords.append(kw)
                    
            # Check secondary keywords
            secondary_kw = self.analyzer.PARADIGM_KEYWORDS[paradigm]['secondary']
            for kw in secondary_kw:
                if kw in query_lower:
                    keyword_score += 1.0
                    matched_keywords.append(kw)
                    
            # Check patterns
            patterns = self.analyzer.intent_patterns.get(paradigm, [])
            for pattern in patterns:
                if pattern.search(query):
                    pattern_score += 3.0
                    reasoning.append(f"Matches pattern: {pattern.pattern}")
                    
            # Adjust for intent signals
            intent_bonus = self._score_intent_alignment(paradigm, features.intent_signals)
            
            # Combine rule scores
            total_score = keyword_score + pattern_score + intent_bonus
            
            # Add reasoning
            if matched_keywords:
                reasoning.append(f"Keywords: {', '.join(matched_keywords[:5])}")
            if intent_bonus > 0:
                reasoning.append(f"Intent alignment: {intent_bonus:.1f}")
                
            scores[paradigm] = ParadigmScore(
                paradigm=paradigm,
                score=total_score,
                confidence=min(total_score / 10.0, 1.0),
                reasoning=reasoning,
                keyword_matches=matched_keywords
            )
            
        return scores
    
    def _score_intent_alignment(self, paradigm: HostParadigm, 
                               intent_signals: List[str]) -> float:
        """Score how well intent signals align with paradigm"""
        alignments = {
            HostParadigm.DOLORES: ['action_stop', 'action_prevent', 'why_question'],
            HostParadigm.TEDDY: ['action_help', 'how_to', 'advice_seeking'],
            HostParadigm.BERNARD: ['what_question', 'why_question'],
            HostParadigm.MAEVE: ['how_to', 'action_create', 'action_improve']
        }
        
        paradigm_intents = alignments.get(paradigm, [])
        matches = sum(1 for signal in intent_signals if signal in paradigm_intents)
        
        return matches * 1.5
    
    async def _llm_classification(self, query: str, 
                                 features: QueryFeatures) -> Dict[HostParadigm, ParadigmScore]:
        """LLM-based classification (mock for now, replace with actual API)"""
        # In production, this would call OpenAI/Anthropic API
        # For now, return mock scores based on features
        
        await asyncio.sleep(0.1)  # Simulate API delay
        
        scores = {}
        
        # Simulate LLM reasoning based on features
        if features.emotional_valence < -0.5:
            # Negative emotion -> likely Dolores
            scores[HostParadigm.DOLORES] = ParadigmScore(
                paradigm=HostParadigm.DOLORES,
                score=8.0,
                confidence=0.8,
                reasoning=["Strong negative emotional content detected"],
                keyword_matches=[]
            )
        
        if features.emotional_valence > 0.5:
            # Positive emotion -> likely Teddy
            scores[HostParadigm.TEDDY] = ParadigmScore(
                paradigm=HostParadigm.TEDDY,
                score=7.0,
                confidence=0.7,
                reasoning=["Positive, supportive tone detected"],
                keyword_matches=[]
            )
            
        if features.complexity_score > 0.7:
            # High complexity -> likely Bernard
            scores[HostParadigm.BERNARD] = ParadigmScore(
                paradigm=HostParadigm.BERNARD,
                score=6.0,
                confidence=0.75,
                reasoning=["Complex analytical query structure"],
                keyword_matches=[]
            )
            
        if 'how_to' in features.intent_signals:
            # How-to -> likely Maeve
            scores[HostParadigm.MAEVE] = ParadigmScore(
                paradigm=HostParadigm.MAEVE,
                score=7.5,
                confidence=0.8,
                reasoning=["Action-oriented query seeking strategy"],
                keyword_matches=[]
            )
            
        # Fill in missing paradigms
        for paradigm in HostParadigm:
            if paradigm not in scores:
                scores[paradigm] = ParadigmScore(
                    paradigm=paradigm,
                    score=3.0,
                    confidence=0.3,
                    reasoning=["Low LLM confidence"],
                    keyword_matches=[]
                )
                
        return scores
    
    def _combine_scores(self, rule_scores: Dict[HostParadigm, ParadigmScore],
                       llm_scores: Dict[HostParadigm, ParadigmScore],
                       features: QueryFeatures) -> Dict[HostParadigm, ParadigmScore]:
        """Combine rule-based and LLM scores with domain bias"""
        combined = {}
        
        for paradigm in HostParadigm:
            rule_score = rule_scores.get(paradigm, ParadigmScore(
                paradigm=paradigm, score=0, confidence=0, reasoning=[], keyword_matches=[]
            ))
            
            # Start with weighted rule score
            total_score = rule_score.score * self.rule_weight
            
            # Add LLM score if available
            if paradigm in llm_scores:
                llm_score = llm_scores[paradigm]
                total_score += llm_score.score * self.llm_weight
                
            # Apply domain bias
            if features.domain:
                domain_paradigm = self.analyzer.DOMAIN_PARADIGM_BIAS.get(features.domain)
                if domain_paradigm == paradigm:
                    total_score *= (1 + self.domain_weight)
                    
            # Combine reasoning
            all_reasoning = rule_score.reasoning.copy()
            if paradigm in llm_scores:
                all_reasoning.extend(llm_scores[paradigm].reasoning)
            if features.domain and self.analyzer.DOMAIN_PARADIGM_BIAS.get(features.domain) == paradigm:
                all_reasoning.append(f"Domain match: {features.domain}")
                
            combined[paradigm] = ParadigmScore(
                paradigm=paradigm,
                score=total_score,
                confidence=(rule_score.confidence + llm_scores.get(paradigm, rule_score).confidence) / 2,
                reasoning=all_reasoning,
                keyword_matches=rule_score.keyword_matches
            )
            
        return combined
    
    def _normalize_scores(self, scores: Dict[HostParadigm, ParadigmScore]) -> Dict[HostParadigm, float]:
        """Normalize scores to probability distribution"""
        total = sum(s.score for s in scores.values())
        
        if total == 0:
            # Equal distribution if no signals
            return {p: 0.25 for p in HostParadigm}
            
        return {p: s.score / total for p, s in scores.items()}
    
    def _calculate_confidence(self, distribution: Dict[HostParadigm, float],
                            scores: Dict[HostParadigm, ParadigmScore]) -> float:
        """Calculate overall classification confidence"""
        # Factors:
        # 1. Spread (higher spread = higher confidence)
        # 2. Top score magnitude
        # 3. Agreement between rule and LLM (if applicable)
        
        sorted_probs = sorted(distribution.values(), reverse=True)
        spread = sorted_probs[0] - sorted_probs[1]  # Gap between top 2
        
        top_paradigm = max(distribution.items(), key=lambda x: x[1])[0]
        top_score = scores[top_paradigm].confidence
        
        # Combine factors
        confidence = (spread * 0.5 + top_score * 0.5)
        
        return min(confidence, 0.95)  # Cap at 95%

# --- Classification Manager ---

class ClassificationEngine:
    """Main interface for the classification system"""
    
    def __init__(self, use_llm: bool = True, cache_enabled: bool = True):
        self.analyzer = QueryAnalyzer()
        self.classifier = ParadigmClassifier(self.analyzer, use_llm)
        self.cache_enabled = cache_enabled
        self.cache = {} if cache_enabled else None
        self.classification_history = []
        
    async def classify_query(self, query: str) -> ClassificationResult:
        """Classify a query with caching and logging"""
        # Check cache
        if self.cache_enabled and query in self.cache:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return self.cache[query]
            
        # Perform classification
        logger.info(f"Classifying query: {query[:50]}...")
        start_time = datetime.now()
        
        try:
            result = await self.classifier.classify(query)
            
            # Log performance
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Classification completed in {elapsed:.2f}s")
            logger.info(f"Primary paradigm: {result.primary_paradigm.value} "
                       f"({result.distribution[result.primary_paradigm]:.1%})")
            
            # Cache result
            if self.cache_enabled:
                self.cache[query] = result
                
            # Store in history
            self.classification_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            raise
            
    def get_classification_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from classification history"""
        if not self.classification_history:
            return {}
            
        paradigm_counts = {p: 0 for p in HostParadigm}
        confidence_sum = 0.0
        
        for result in self.classification_history:
            paradigm_counts[result.primary_paradigm] += 1
            confidence_sum += result.confidence
            
        return {
            'total_classifications': len(self.classification_history),
            'paradigm_distribution': {
                p.value: count/len(self.classification_history) 
                for p, count in paradigm_counts.items()
            },
            'average_confidence': confidence_sum / len(self.classification_history),
            'cache_size': len(self.cache) if self.cache else 0
        }
    
    def export_classification_data(self, filepath: str):
        """Export classification history for analysis"""
        data = []
        for result in self.classification_history:
            data.append({
                'query': result.query,
                'primary_paradigm': result.primary_paradigm.value,
                'secondary_paradigm': result.secondary_paradigm.value if result.secondary_paradigm else None,
                'distribution': {p.value: score for p, score in result.distribution.items()},
                'confidence': result.confidence,
                'timestamp': result.timestamp.isoformat()
            })
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Exported {len(data)} classifications to {filepath}")

# --- Testing and Validation ---

async def test_classification_engine():
    """Test the classification engine with sample queries"""
    engine = ClassificationEngine(use_llm=True)
    
    test_queries = [
        # Dolores queries
        "How can we expose corporate tax avoidance schemes?",
        "Why is the healthcare system so unfair to poor people?",
        "Fight against monopolistic practices of big tech",
        
        # Teddy queries
        "How can I help homeless veterans in my community?",
        "Best ways to support grieving families",
        "Resources for protecting endangered animals",
        
        # Bernard queries
        "Analyze the correlation between social media use and depression",
        "What does research say about climate change impacts?",
        "Statistical analysis of income inequality trends",
        
        # Maeve queries
        "Best strategy to compete with Amazon as a small business",
        "How to influence policy makers effectively",
        "Optimize conversion rates for e-commerce",
        
        # Ambiguous queries
        "How to make the world a better place",
        "Understanding artificial intelligence",
        "Deal with difficult people at work"
    ]
    
    print("Testing Classification Engine")
    print("=" * 80)
    
    for query in test_queries:
        result = await engine.classify_query(query)
        
        print(f"\nQuery: {query}")
        print(f"Primary: {result.primary_paradigm.value} ({result.confidence:.1%} confidence)")
        
        if result.secondary_paradigm:
            print(f"Secondary: {result.secondary_paradigm.value}")
            
        print("Distribution:")
        for paradigm, prob in sorted(result.distribution.items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {paradigm.value:15} {prob:>6.1%}")
            
        # Show top reasoning
        top_reasoning = result.reasoning[result.primary_paradigm][:2]
        if top_reasoning:
            print("Reasoning:", " | ".join(top_reasoning))
            
    # Show metrics
    print("\n" + "=" * 80)
    print("Classification Metrics:")
    metrics = engine.get_classification_metrics()
    print(f"Total queries: {metrics['total_classifications']}")
    print(f"Average confidence: {metrics['average_confidence']:.1%}")
    print("Paradigm distribution:")
    for paradigm, pct in metrics['paradigm_distribution'].items():
        print(f"  {paradigm:15} {pct:>6.1%}")

# Main execution
if __name__ == "__main__":
    asyncio.run(test_classification_engine())
```
"""
Unit tests for the Four Hosts Classification Engine
Tests for QueryAnalyzer, ParadigmClassifier, and ClassificationEngine
"""

import pytest
import asyncio
from typing import List, Dict
from services.classification_engine import (
    QueryAnalyzer,
    ParadigmClassifier,
    ClassificationEngine,
    HostParadigm,
    QueryFeatures,
    ClassificationResult
)


class TestQueryAnalyzer:
    """Test the QueryAnalyzer feature extraction"""
    
    @pytest.fixture
    def analyzer(self):
        return QueryAnalyzer()
    
    def test_tokenize(self, analyzer):
        """Test basic tokenization"""
        text = "How can I help vulnerable communities?"
        tokens = analyzer._tokenize(text)
        assert 'help' in tokens
        assert 'vulnerable' in tokens
        assert 'communities' in tokens
        assert all(len(t) > 2 for t in tokens)
    
    def test_extract_entities(self, analyzer):
        """Test entity extraction"""
        query = 'How can Microsoft help with "climate change" research?'
        entities = analyzer._extract_entities(query)
        assert 'Microsoft' in entities
        assert 'climate change' in entities
    
    def test_detect_intent_signals(self, analyzer):
        """Test intent signal detection"""
        queries = {
            "How to build a better system": ['how_to', 'action_build'],
            "Why is this happening?": ['why_question'],
            "What data supports this?": ['what_question'],
            "Should I help them?": ['advice_seeking', 'action_help']
        }
        
        for query, expected in queries.items():
            signals = analyzer._detect_intent_signals(query)
            for signal in expected:
                assert signal in signals
    
    def test_identify_domain(self, analyzer):
        """Test domain identification"""
        test_cases = {
            "How to improve patient care": "healthcare",
            "Best business strategy for growth": "business",
            "Research on climate change": "science",
            "Fighting social injustice": "social_justice",
            "AI and machine learning trends": "technology"
        }
        
        for query, expected_domain in test_cases.items():
            domain = analyzer._identify_domain(query, [])
            if domain != expected_domain:
                print(f"Query: {query}")
                print(f"Expected: {expected_domain}, Got: {domain}")
            assert domain == expected_domain
    
    def test_urgency_scoring(self, analyzer):
        """Test urgency score calculation"""
        low_urgency = "What are the long term effects?"
        high_urgency = "Need urgent help immediately with crisis"
        
        low_tokens = analyzer._tokenize(low_urgency)
        high_tokens = analyzer._tokenize(high_urgency)
        
        assert analyzer._score_urgency(low_urgency, low_tokens) < 0.3
        assert analyzer._score_urgency(high_urgency, high_tokens) > 0.6
    
    def test_emotional_valence(self, analyzer):
        """Test emotional valence scoring"""
        positive = "I want to help and support the community"
        negative = "This corrupt system must be destroyed"
        neutral = "Analyze the data from the experiment"
        
        pos_tokens = analyzer._tokenize(positive)
        neg_tokens = analyzer._tokenize(negative)
        neu_tokens = analyzer._tokenize(neutral)
        
        assert analyzer._score_emotional_valence(pos_tokens) > 0.5
        assert analyzer._score_emotional_valence(neg_tokens) < -0.3
        assert abs(analyzer._score_emotional_valence(neu_tokens)) < 0.3
    
    def test_full_analysis(self, analyzer):
        """Test complete feature extraction"""
        query = "How can we urgently help vulnerable communities affected by injustice?"
        features = analyzer.analyze(query)
        
        assert isinstance(features, QueryFeatures)
        assert features.text == query
        assert len(features.tokens) > 5
        assert features.urgency_score > 0.3
        assert 'how_to' in features.intent_signals


class TestParadigmClassifier:
    """Test the ParadigmClassifier"""
    
    @pytest.fixture
    def classifier(self):
        analyzer = QueryAnalyzer()
        return ParadigmClassifier(analyzer, use_llm=False)
    
    @pytest.mark.asyncio
    async def test_dolores_classification(self, classifier):
        """Test Dolores paradigm classification"""
        query = "We must expose the corruption and fight injustice in the system"
        result = await classifier.classify(query)
        
        assert result.primary_paradigm == HostParadigm.DOLORES
        assert result.distribution[HostParadigm.DOLORES] > 0.4
        assert result.confidence > 0.6
        assert len(result.reasoning[HostParadigm.DOLORES]) > 0
    
    @pytest.mark.asyncio
    async def test_teddy_classification(self, classifier):
        """Test Teddy paradigm classification"""
        query = "How can I help protect and support vulnerable community members?"
        result = await classifier.classify(query)
        
        assert result.primary_paradigm == HostParadigm.TEDDY
        assert result.distribution[HostParadigm.TEDDY] > 0.4
        assert 'help' in str(result.reasoning[HostParadigm.TEDDY])
    
    @pytest.mark.asyncio
    async def test_bernard_classification(self, classifier):
        """Test Bernard paradigm classification"""
        query = "What does the research data show about these statistical correlations?"
        result = await classifier.classify(query)
        
        assert result.primary_paradigm == HostParadigm.BERNARD
        assert result.distribution[HostParadigm.BERNARD] > 0.4
    
    @pytest.mark.asyncio
    async def test_maeve_classification(self, classifier):
        """Test Maeve paradigm classification"""
        query = "What's the best strategy to compete and maximize our market advantage?"
        result = await classifier.classify(query)
        
        assert result.primary_paradigm == HostParadigm.MAEVE
        assert result.distribution[HostParadigm.MAEVE] > 0.4
    
    @pytest.mark.asyncio
    async def test_mixed_paradigm_classification(self, classifier):
        """Test classification with mixed signals"""
        query = "How can we strategically help communities while analyzing the data?"
        result = await classifier.classify(query)
        
        assert result.primary_paradigm in [HostParadigm.TEDDY, HostParadigm.MAEVE]
        assert result.secondary_paradigm is not None
        assert sum(result.distribution.values()) == pytest.approx(1.0, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_domain_bias(self, classifier):
        """Test that domain bias affects classification"""
        business_query = "Analyze the market trends"
        healthcare_query = "Analyze patient outcomes"
        
        business_result = await classifier.classify(business_query)
        healthcare_result = await classifier.classify(healthcare_query)
        
        # Business domain should bias towards Maeve
        assert business_result.distribution[HostParadigm.MAEVE] > 0.2
        
        # Healthcare domain should bias towards Teddy
        assert healthcare_result.distribution[HostParadigm.TEDDY] > 0.2


class TestClassificationEngine:
    """Test the main ClassificationEngine interface"""
    
    @pytest.fixture
    def engine(self):
        return ClassificationEngine(use_llm=False, cache_enabled=True)
    
    @pytest.mark.asyncio
    async def test_classify_query(self, engine):
        """Test basic query classification"""
        query = "We need to fight against systemic oppression"
        result = await engine.classify_query(query)
        
        assert isinstance(result, ClassificationResult)
        assert result.query == query
        assert result.primary_paradigm == HostParadigm.DOLORES
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_caching(self, engine):
        """Test that caching works correctly"""
        query = "What does the research show?"
        
        # First call
        result1 = await engine.classify_query(query)
        
        # Second call should hit cache
        result2 = await engine.classify_query(query)
        
        assert result1.primary_paradigm == result2.primary_paradigm
        assert result1.confidence == result2.confidence
        assert result1.timestamp == result2.timestamp  # Same cached object
    
    @pytest.mark.asyncio
    async def test_diverse_queries(self, engine):
        """Test classification accuracy across diverse queries"""
        test_cases = [
            # (query, expected_primary_paradigm)
            ("Expose the truth about corporate corruption", HostParadigm.DOLORES),
            ("How can I volunteer to help elderly neighbors?", HostParadigm.TEDDY),
            ("Statistical analysis of climate data trends", HostParadigm.BERNARD),
            ("Best tactics to dominate the market", HostParadigm.MAEVE),
            ("Research how to protect vulnerable children from injustice", HostParadigm.TEDDY),
            ("Strategic plan to fight systemic inequality", HostParadigm.DOLORES),
        ]
        
        correct_classifications = 0
        for query, expected in test_cases:
            result = await engine.classify_query(query)
            if result.primary_paradigm == expected:
                correct_classifications += 1
            else:
                # Check if it's at least secondary
                if result.secondary_paradigm == expected:
                    correct_classifications += 0.5
        
        accuracy = correct_classifications / len(test_cases)
        assert accuracy >= 0.7  # 70% accuracy threshold


@pytest.mark.asyncio
async def test_edge_cases():
    """Test edge cases and error conditions"""
    engine = ClassificationEngine(use_llm=False)
    
    # Empty query
    result = await engine.classify_query("")
    assert result.primary_paradigm is not None
    assert result.confidence < 0.6
    
    # Very short query
    result = await engine.classify_query("Help")
    assert result.primary_paradigm == HostParadigm.TEDDY
    
    # Query with no clear paradigm
    result = await engine.classify_query("The weather is nice today")
    assert result.confidence < 0.6
    
    # Very long query
    long_query = " ".join(["analyze"] * 100)
    result = await engine.classify_query(long_query)
    assert result.primary_paradigm == HostParadigm.BERNARD


if __name__ == "__main__":
    # Run async tests
    asyncio.run(test_diverse_queries())
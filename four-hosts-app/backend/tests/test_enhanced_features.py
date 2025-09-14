#!/usr/bin/env python3
"""
Test script for enhanced Four Hosts features:
- Enhanced Bernard and Maeve answer generators
- Self-healing paradigm switching
- ML pipeline for continuous improvement
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import enhanced components
from services.answer_generator import (
    EnhancedBernardAnswerGenerator,
    EnhancedMaeveAnswerGenerator,
)
from services.self_healing_system import self_healing_system
from services.ml_pipeline import ml_pipeline
from services.enhanced_integration import (
    enhanced_answer_orchestrator,
    enhanced_classification_engine,
    record_user_feedback,
    get_system_health_report,
    get_paradigm_performance_metrics,
)
from services.answer_generator import SynthesisContext
from services.classification_engine import HostParadigm, QueryFeatures


async def test_enhanced_bernard_generator():
    """Test enhanced Bernard answer generator with statistical analysis"""
    print("\n" + "="*70)
    print("Testing Enhanced Bernard Answer Generator")
    print("="*70)
    
    generator = EnhancedBernardAnswerGenerator()
    
    # Create test context with research results
    context = SynthesisContext(
        query="What is the correlation between exercise and mental health?",
        search_results=[
            {
                "title": "Meta-analysis: Exercise and Depression",
                "snippet": "A comprehensive meta-analysis of 49 studies (n=266,939) found a correlation of r=-0.48 (p<0.001) between regular exercise and depression scores. Effect size Cohen's d=0.82 indicates large effect.",
                "domain": "pubmed.ncbi.nlm.nih.gov",
                "url": "https://pubmed.example.com/12345",
                "credibility_score": 0.95,
            },
            {
                "title": "Longitudinal Study: Physical Activity and Anxiety",
                "snippet": "10-year longitudinal study (n=15,000) showed 35% reduction in anxiety disorders among those exercising >150 min/week. Confidence interval 95% CI: 0.58-0.72, p=0.002.",
                "domain": "nature.com",
                "url": "https://nature.com/articles/example",
                "credibility_score": 0.92,
            },
            {
                "title": "RCT: Exercise Intervention for Mental Health",
                "snippet": "Randomized controlled trial (n=500) demonstrated 42% improvement in mental health scores after 12-week exercise program. Control group showed 8% improvement. Statistical significance p<0.001.",
                "domain": "sciencedirect.com",
                "url": "https://sciencedirect.com/example",
                "credibility_score": 0.88,
            },
        ],
        max_length=1000,
        metadata={"research_id": "test_bernard_001"},
    )
    
    # Generate answer
    answer = await generator.generate_answer(context)
    
    # Display results
    print(f"\nQuery: {context.query}")
    print(f"\nSummary: {answer.summary}")
    print(f"\nConfidence Score: {answer.confidence_score:.2f}")
    print(f"\nSynthesis Quality: {answer.synthesis_quality:.2f}")
    
    print("\nSections:")
    for section in answer.sections:
        print(f"\n{section.title}:")
        print(f"  Preview: {section.content[:150]}...")
        print(f"  Confidence: {section.confidence:.2f}")
        print(f"  Citations: {len(section.citations)}")
        print(f"  Key Insights: {len(section.key_insights)}")
        if section.key_insights:
            print(f"    - {section.key_insights[0]}")
    
    print("\nStatistical Insights Found:")
    if "statistical_insights" in answer.metadata:
        print(f"  Total insights: {answer.metadata['statistical_insights']}")
    if "meta_analysis_performed" in answer.metadata:
        print(f"  Meta-analysis: {answer.metadata['meta_analysis_performed']}")
    if "peer_reviewed_sources" in answer.metadata:
        print(f"  Peer-reviewed sources: {answer.metadata['peer_reviewed_sources']}")
    
    print("\n✅ Enhanced Bernard generator test completed")


async def test_enhanced_maeve_generator():
    """Test enhanced Maeve answer generator with strategic analysis"""
    print("\n" + "="*70)
    print("Testing Enhanced Maeve Answer Generator")
    print("="*70)
    
    generator = EnhancedMaeveAnswerGenerator()
    
    # Create test context with business/strategic results
    context = SynthesisContext(
        query="How can a small e-commerce business compete with Amazon?",
        search_results=[
            {
                "title": "E-commerce Market Analysis 2024",
                "snippet": "The global e-commerce market size is $5.8 trillion with 15% annual growth rate. Amazon holds 37% market share. Niche markets show 25% CAGR opportunity.",
                "domain": "mckinsey.com",
                "url": "https://mckinsey.com/insights/example",
                "credibility_score": 0.93,
            },
            {
                "title": "Small Business E-commerce Strategy Guide",
                "snippet": "Successful small e-commerce businesses report 150% ROI on personalization strategies. Customer retention costs 5x less than acquisition. Focus on niche markets yields 3x higher margins.",
                "domain": "hbr.org",
                "url": "https://hbr.org/2024/example",
                "credibility_score": 0.90,
            },
            {
                "title": "Competitive Analysis: David vs Goliath in E-commerce",
                "snippet": "Small businesses can save $50K annually by using integrated platforms. Implementation timeline: 3-6 months. Market penetration strategy shows 40% success rate in first year.",
                "domain": "forbes.com",
                "url": "https://forbes.com/sites/example",
                "credibility_score": 0.85,
            },
        ],
        max_length=1000,
        metadata={"research_id": "test_maeve_001"},
    )
    
    # Generate answer
    answer = await generator.generate_answer(context)
    
    # Display results
    print(f"\nQuery: {context.query}")
    print(f"\nSummary: {answer.summary}")
    print(f"\nConfidence Score: {answer.confidence_score:.2f}")
    print(f"\nSynthesis Quality: {answer.synthesis_quality:.2f}")
    
    print("\nStrategic Recommendations:")
    for i, item in enumerate(answer.action_items[:3], 1):
        print(f"\n{i}. {item['action']}")
        print(f"   Priority: {item['priority']}")
        print(f"   Timeline: {item['timeframe']}")
        print(f"   Impact: {item['impact']}")
        print(f"   Dependencies: {', '.join(item['dependencies'][:2])}")
    
    print("\nCompetitive Analysis:")
    if "competitive_analysis" in answer.metadata:
        comp = answer.metadata["competitive_analysis"]
        print(f"  Market leaders: {comp.get('market_leaders', [])}")
        print(f"  Competitive intensity: {comp.get('competitive_intensity', 'unknown')}")
    
    print("\n✅ Enhanced Maeve generator test completed")


async def test_self_healing_system():
    """Test self-healing paradigm switching"""
    print("\n" + "="*70)
    print("Testing Self-Healing System")
    print("="*70)
    
    # Simulate multiple queries with varying performance
    test_queries = [
        {
            "id": "sh_test_001",
            "text": "Analyze the statistical correlation between variables",
            "paradigm": HostParadigm.BERNARD,
            "confidence": 0.85,
            "quality": 0.88,
            "error": None,
        },
        {
            "id": "sh_test_002",
            "text": "Help me understand why my child is struggling in school",
            "paradigm": HostParadigm.TEDDY,
            "confidence": 0.90,
            "quality": 0.92,
            "error": None,
        },
        {
            "id": "sh_test_003",
            "text": "Strategic plan for market expansion",
            "paradigm": HostParadigm.BERNARD,  # Wrong paradigm
            "confidence": 0.45,  # Low confidence
            "quality": 0.50,
            "error": None,
        },
        {
            "id": "sh_test_004",
            "text": "Fight against corporate monopolies",
            "paradigm": HostParadigm.MAEVE,  # Wrong paradigm
            "confidence": 0.40,  # Low confidence
            "quality": 0.45,
            "error": None,
        },
    ]
    
    print("\nRecording query performance...")
    for query in test_queries:
        await self_healing_system.record_query_performance(
            query_id=query["id"],
            query_text=query["text"],
            paradigm=query["paradigm"],
            answer=type('Answer', (), {
                'confidence_score': query["confidence"],
                'synthesis_quality': query["quality"]
            })() if not query["error"] else None,
            error=query["error"],
            response_time=1.5,
        )
        print(f"  Recorded: {query['id']} - {query['paradigm'].value} "
              f"(conf: {query['confidence']:.2f})")
    
    # Test paradigm recommendations
    print("\nTesting paradigm recommendations...")
    test_recommendations = [
        ("Strategic plan for market expansion", HostParadigm.BERNARD),
        ("Fight against corporate monopolies", HostParadigm.MAEVE),
    ]
    
    for query_text, current in test_recommendations:
        recommended = self_healing_system.get_paradigm_recommendation(query_text, current)
        print(f"  Query: '{query_text[:40]}...'")
        print(f"    Current: {current.value}")
        print(f"    Recommended: {recommended.value if recommended else 'No change'}")
    
    # Get performance report
    report = self_healing_system.get_performance_report()
    print("\nSystem Performance Report:")
    print(f"  Total switches: {report['switch_statistics']['total_switches']}")
    print("  Paradigm metrics:")
    for paradigm, metrics in report["paradigm_metrics"].items():
        print(f"    {paradigm}: {metrics['total_queries']} queries, "
              f"{metrics['success_rate']:.2%} success rate")
    
    print("\n✅ Self-healing system test completed")


async def test_ml_pipeline():
    """Test ML pipeline for continuous improvement"""
    print("\n" + "="*70)
    print("Testing ML Pipeline")
    print("="*70)
    
    # Create training examples
    print("\nCreating training examples...")
    
    examples = [
        # Analytical queries
        ("What are the statistical trends in climate data?", HostParadigm.BERNARD, 0.85),
        ("Analyze the correlation between GDP and happiness", HostParadigm.BERNARD, 0.88),
        ("Research methodology for studying sleep patterns", HostParadigm.BERNARD, 0.82),
        
        # Strategic queries
        ("Business strategy for entering Asian markets", HostParadigm.MAEVE, 0.87),
        ("How to compete with established players", HostParadigm.MAEVE, 0.84),
        ("ROI analysis for digital transformation", HostParadigm.MAEVE, 0.86),
        
        # Supportive queries
        ("Help with caring for elderly parents", HostParadigm.TEDDY, 0.89),
        ("Support resources for mental health", HostParadigm.TEDDY, 0.91),
        ("Community programs for homeless youth", HostParadigm.TEDDY, 0.88),
        
        # Revolutionary queries
        ("Fighting systemic inequality in education", HostParadigm.DOLORES, 0.86),
        ("Exposing corporate environmental crimes", HostParadigm.DOLORES, 0.84),
        ("Organizing grassroots political movements", HostParadigm.DOLORES, 0.87),
    ]
    
    # Record examples
    for i, (query, paradigm, satisfaction) in enumerate(examples):
        features = QueryFeatures(
            text=query,
            tokens=query.lower().split(),
            entities=[],
            intent_signals=[],
            domain="general",
            urgency_score=0.5,
            complexity_score=0.6,
            emotional_valence=0.5,
        )
        
        await ml_pipeline.record_training_example(
            query_id=f"ml_test_{i:03d}",
            query_text=query,
            features=features,
            predicted_paradigm=paradigm,
            true_paradigm=paradigm,
            user_feedback=satisfaction,
            synthesis_quality=satisfaction * 0.95,
        )
    
    print(f"  Recorded {len(examples)} training examples")
    
    # Get model info
    model_info = ml_pipeline.get_model_info()
    print("\nModel Information:")
    print(f"  Current version: {model_info['current_version']}")
    print(f"  ML available: {model_info['ml_available']}")
    print(f"  Training examples: {model_info['training_examples']}")
    
    # Get training stats
    stats = ml_pipeline.get_training_stats()
    print("\nTraining Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print("  Examples by paradigm:")
    for paradigm, count in stats["examples_by_paradigm"].items():
        print(f"    {paradigm}: {count}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    
    # Test prediction
    if model_info['ml_available']:
        print("\nTesting ML prediction...")
        test_query = "What's the best strategy for market disruption?"
        test_features = QueryFeatures(
            text=test_query,
            tokens=test_query.lower().split(),
            entities=["market", "disruption"],
            intent_signals=["strategy"],
            domain="business",
            urgency_score=0.6,
            complexity_score=0.7,
            emotional_valence=0.5,
        )
        
        predicted, confidence = ml_pipeline.predict_paradigm(test_query, test_features)
        print(f"  Query: '{test_query}'")
        print(f"  Predicted paradigm: {predicted.value}")
        print(f"  Confidence: {confidence:.2f}")
    
    print("\n✅ ML pipeline test completed")


async def test_integration():
    """Test full integration of enhanced features"""
    print("\n" + "="*70)
    print("Testing Full Integration")
    print("="*70)
    
    # Test enhanced classification
    print("\nTesting enhanced classification...")
    query = "Analyze market trends and develop a strategic response"
    result = await enhanced_classification_engine.classify_query(query, use_llm=False)
    
    print(f"  Query: '{query}'")
    print(f"  Primary paradigm: {result.primary_paradigm}")
    print(f"  Confidence: {result.confidence:.2f}")
    if result.secondary_paradigm:
        print(f"  Secondary paradigm: {result.secondary_paradigm}")
    
    # Test enhanced answer generation
    print("\nTesting enhanced answer generation...")
    context = SynthesisContext(
        query=query,
        search_results=[
            {
                "title": "Market Trends 2024",
                "snippet": "Market showing 12% growth with emerging opportunities...",
                "domain": "example.com",
                "credibility_score": 0.8,
            }
        ],
        max_length=500,
        metadata={"research_id": "integration_test_001"},
    )
    
    answer = await enhanced_answer_orchestrator.generate_answer(
        context,
        result.primary_paradigm,
        result.secondary_paradigm,
    )
    
    print(f"  Generated answer for: {answer.paradigm}")
    print(f"  Sections: {len(answer.sections)}")
    print(f"  Action items: {len(answer.action_items)}")
    
    # Test user feedback
    print("\nRecording user feedback...")
    await record_user_feedback("integration_test_001", 0.85, "strategic")
    print("  Feedback recorded")
    
    # Get system health report
    print("\nSystem Health Report:")
    health = get_system_health_report()
    print(f"  Timestamp: {health['timestamp']}")
    print(f"  Recommendations: {len(health['recommendations'])}")
    for rec in health["recommendations"]:
        print(f"    - {rec['action']} ({rec['component']})")
    
    # Get paradigm metrics
    print("\nParadigm Performance Metrics:")
    metrics = get_paradigm_performance_metrics()
    for paradigm, perf in metrics.items():
        print(f"  {paradigm}: {perf['success_rate']:.2%} success, "
              f"trend: {perf['recent_trend']}")
    
    print("\n✅ Integration test completed")


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Four Hosts Enhanced Features Test Suite")
    print("="*70)
    
    try:
        # Run individual component tests
        await test_enhanced_bernard_generator()
        await test_enhanced_maeve_generator()
        await test_self_healing_system()
        await test_ml_pipeline()
        
        # Run integration test
        await test_integration()
        
        print("\n" + "="*70)
        print("✅ All tests completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test suite error")


if __name__ == "__main__":
    asyncio.run(main())

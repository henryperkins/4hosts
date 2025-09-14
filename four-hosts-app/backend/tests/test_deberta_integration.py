#!/usr/bin/env python3
"""
Test script for DeBERTa zero-shot classifier integration
Tests paradigm classification with the new HF model
"""

import asyncio
import logging
from services.ml_pipeline import ml_pipeline
from services.classification_engine import QueryFeatures, HostParadigm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test queries for each paradigm
TEST_QUERIES = {
    HostParadigm.DOLORES: [
        "How can we fight against systemic inequality in the workplace?",
        "What are the revolutionary changes needed in climate policy?",
        "Expose the corruption in pharmaceutical pricing models",
    ],
    HostParadigm.TEDDY: [
        "How can I support my community during difficult times?",
        "What are the best ways to help elderly neighbors?",
        "How to create a caring and inclusive workplace environment?",
    ],
    HostParadigm.BERNARD: [
        "Analyze the statistical correlation between education and income",
        "What does the research data show about vaccine efficacy?",
        "Conduct a comprehensive study on renewable energy adoption rates",
    ],
    HostParadigm.MAEVE: [
        "Develop a strategic plan to outcompete market rivals",
        "What business strategies can maximize quarterly profits?",
        "How to optimize supply chain for competitive advantage?",
    ],
}

async def test_paradigm_classification():
    """Test the DeBERTa classifier on paradigm-specific queries"""
    print("\n" + "="*60)
    print("Testing DeBERTa Zero-Shot Classifier Integration")
    print("="*60 + "\n")
    
    all_correct = 0
    total_queries = 0
    
    for expected_paradigm, queries in TEST_QUERIES.items():
        print(f"\n--- Testing {expected_paradigm.value.upper()} paradigm ---")
        paradigm_correct = 0
        
        for query in queries:
            total_queries += 1
            
            # Create dummy features (these would normally come from classification engine)
            features = QueryFeatures(
                urgency_score=0.5,
                complexity_score=0.5,
                entities=[],
                intent_signals=[],
                domain="general",
                confidence_score=0.5
            )
            
            # Get prediction
            try:
                predicted_paradigm, confidence = await ml_pipeline.predict_paradigm(
                    query, features
                )
                
                is_correct = predicted_paradigm == expected_paradigm
                if is_correct:
                    paradigm_correct += 1
                    all_correct += 1
                
                print(f"\nQuery: '{query[:60]}...'" if len(query) > 60 else f"\nQuery: '{query}'")
                print(f"Expected: {expected_paradigm.value}")
                print(f"Predicted: {predicted_paradigm.value} (confidence: {confidence:.3f})")
                print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                
            except Exception as e:
                logger.error(f"Error classifying query: {e}")
                print(f"ERROR: Failed to classify query")
        
        accuracy = (paradigm_correct / len(queries)) * 100
        print(f"\n{expected_paradigm.value} accuracy: {accuracy:.1f}% ({paradigm_correct}/{len(queries)})")
    
    overall_accuracy = (all_correct / total_queries) * 100
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {overall_accuracy:.1f}% ({all_correct}/{total_queries})")
    print(f"{'='*60}\n")
    
    return overall_accuracy

async def test_edge_cases():
    """Test edge cases and ambiguous queries"""
    print("\n--- Testing Edge Cases ---")
    
    edge_cases = [
        "Tell me about something",  # Very vague
        "How to analyze market data to help communities?",  # Mixed paradigms
        "Revolutionary research on strategic business models",  # All paradigms
        "",  # Empty query
        "?" * 100,  # Nonsense query
    ]
    
    features = QueryFeatures(
        urgency_score=0.5,
        complexity_score=0.5,
        entities=[],
        intent_signals=[],
        domain="general",
        confidence_score=0.5
    )
    
    for query in edge_cases:
        try:
            if query:
                display_query = query[:50] + "..." if len(query) > 50 else query
            else:
                display_query = "[empty query]"
                
            predicted_paradigm, confidence = await ml_pipeline.predict_paradigm(
                query, features
            )
            
            print(f"\nEdge case: '{display_query}'")
            print(f"Predicted: {predicted_paradigm.value} (confidence: {confidence:.3f})")
            
        except Exception as e:
            print(f"\nEdge case: '{display_query}'")
            print(f"ERROR: {e}")

async def main():
    """Run all tests"""
    try:
        # Test paradigm classification
        accuracy = await test_paradigm_classification()
        
        # Test edge cases
        await test_edge_cases()
        
        # Summary
        if accuracy >= 80:
            print("\n✓ DeBERTa integration is working well!")
        elif accuracy >= 60:
            print("\n⚠ DeBERTa integration is working but could be improved")
        else:
            print("\n✗ DeBERTa integration needs adjustment")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n✗ Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
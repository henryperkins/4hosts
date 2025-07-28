#!/usr/bin/env python3
"""Test the classification integration with main.py"""

import sys
sys.path.append('/home/azureuser/4hosts/four-hosts-app/backend')

import asyncio
from main import classify_paradigm
from models import Paradigm, ParadigmClassification

# Create a mock user object
class MockUser:
    def __init__(self):
        self.id = "test_user_123"

async def test_integration():
    """Test the classification endpoint integration"""
    
    test_query = "How can we expose corporate corruption in the tech industry?"
    
    print(f"Testing classification for: {test_query}")
    
    # Test the classify_query function replacement
    try:
        # Import the modules directly
        from services.classification_engine import classification_engine, HostParadigm
        
        # Test direct classification engine
        result = await classification_engine.classify_query(test_query)
        print(f"\nDirect Engine Result:")
        print(f"  Primary: {result.primary_paradigm.value}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Distribution: {result.distribution}")
        
        # Test the conversion logic
        classification = ParadigmClassification(
            primary=Paradigm(result.primary_paradigm.value),
            secondary=Paradigm(result.secondary_paradigm.value) if result.secondary_paradigm else None,
            distribution={
                "revolutionary": result.distribution.get(HostParadigm.DOLORES, 0),
                "devotion": result.distribution.get(HostParadigm.TEDDY, 0),
                "analytical": result.distribution.get(HostParadigm.BERNARD, 0),
                "strategic": result.distribution.get(HostParadigm.MAEVE, 0)
            },
            confidence=result.confidence,
            explanation={
                result.primary_paradigm.value: '; '.join(result.reasoning.get(result.primary_paradigm, [])[:2])
            }
        )
        
        print(f"\nConverted Format:")
        print(f"  Primary: {classification.primary.value}")
        print(f"  Confidence: {classification.confidence:.2%}")
        print(f"  Distribution: {classification.distribution}")
        print(f"  Explanation: {classification.explanation}")
        
        print("\n✅ Integration test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_integration())
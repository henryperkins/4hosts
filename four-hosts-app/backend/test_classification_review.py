#!/usr/bin/env python3
"""Test the classification engine to verify it's working"""

import asyncio
import sys
sys.path.append('/home/azureuser/4hosts/four-hosts-app/backend')

from services.classification_engine import classification_engine

async def test_classification():
    """Test various queries through the classification engine"""
    
    test_queries = [
        # Dolores queries
        "How can we expose corporate corruption in the tech industry?",
        "Why is the wealth gap so unfair to working families?",
        
        # Teddy queries  
        "How can I help support vulnerable elderly in my community?",
        "What resources are available to protect children from online harm?",
        
        # Bernard queries
        "What does the research data show about climate change impacts?", 
        "Analyze the statistical correlation between education and income",
        
        # Maeve queries
        "How to develop a competitive business strategy for market expansion?",
        "What framework can optimize our company's operational efficiency?"
    ]
    
    print("Testing Classification Engine\n" + "="*50)
    
    for query in test_queries:
        result = await classification_engine.classify_query(query)
        
        print(f"\nQuery: {query}")
        print(f"Primary Paradigm: {result.primary_paradigm.value}")
        if result.secondary_paradigm:
            print(f"Secondary Paradigm: {result.secondary_paradigm.value}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Distribution: {', '.join([f'{p.value}: {s:.1%}' for p, s in result.distribution.items()])}")
        print(f"Features: urgency={result.features.urgency_score:.2f}, complexity={result.features.complexity_score:.2f}, emotion={result.features.emotional_valence:.2f}")
        if result.reasoning.get(result.primary_paradigm):
            print(f"Reasoning: {'; '.join(result.reasoning[result.primary_paradigm][:2])}")

if __name__ == "__main__":
    asyncio.run(test_classification())
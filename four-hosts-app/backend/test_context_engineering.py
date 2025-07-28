#!/usr/bin/env python3
"""Test the context engineering pipeline"""

import asyncio
import sys
sys.path.append('/home/azureuser/4hosts/four-hosts-app/backend')

from services.classification_engine import classification_engine
from services.context_engineering import context_pipeline

async def test_context_pipeline():
    """Test the context engineering pipeline with various queries"""
    
    test_queries = [
        "How can small businesses compete with Amazon?",
        "What support is available for homeless veterans?",
        "Analyze the impact of social media on mental health",
        "Expose corporate tax avoidance schemes"
    ]
    
    print("Testing Context Engineering Pipeline")
    print("=" * 80)
    
    for query in test_queries:
        # Classify
        classification = await classification_engine.classify_query(query)
        
        # Process through pipeline
        engineered = await context_pipeline.process_query(classification)
        
        print(f"\nQuery: {query}")
        print(f"Paradigm: {engineered.classification.primary_paradigm.value}")
        print(f"Confidence: {engineered.classification.confidence:.0%}")
        
        print(f"\nWrite Layer:")
        print(f"  Focus: {engineered.write_output.documentation_focus}")
        print(f"  Themes: {', '.join(engineered.write_output.key_themes[:5])}")
        print(f"  Narrative: {engineered.write_output.narrative_frame}")
        
        print(f"\nSelect Layer:")
        print(f"  Queries Generated: {len(engineered.select_output.search_queries)}")
        print(f"  Top 3 Queries:")
        for q in engineered.select_output.search_queries[:3]:
            print(f"    - [{q['type']}] {q['query'][:60]}... (weight: {q['weight']})")
        print(f"  Tools: {', '.join(engineered.select_output.tool_selections)}")
        print(f"  Max Sources: {engineered.select_output.max_sources}")
        
        print(f"\nCompress Layer:")
        print(f"  Ratio: {engineered.compress_output.compression_ratio:.0%}")
        print(f"  Token Budget: {engineered.compress_output.token_budget}")
        print(f"  Strategy: {engineered.compress_output.compression_strategy}")
        print(f"  Priorities: {', '.join(engineered.compress_output.priority_elements[:3])}")
        
        print(f"\nIsolate Layer:")
        print(f"  Strategy: {engineered.isolate_output.isolation_strategy}")
        print(f"  Focus Areas: {', '.join(engineered.isolate_output.focus_areas[:3])}")
        print(f"  Output Structure: {list(engineered.isolate_output.output_structure.keys())}")
        
        print(f"\nTotal Processing Time: {engineered.processing_time:.3f}s")
        print("-" * 80)
    
    # Show metrics
    metrics = context_pipeline.get_pipeline_metrics()
    print(f"\nPipeline Metrics:")
    print(f"Total Processed: {metrics['total_processed']}")
    print(f"Average Time: {metrics['average_processing_time']:.3f}s")
    print(f"Paradigm Distribution: {metrics['paradigm_distribution']}")

if __name__ == "__main__":
    asyncio.run(test_context_pipeline())
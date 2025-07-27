#!/usr/bin/env python3
"""
Example: Using Brave Search API with Four Hosts Research Application
This example shows how the Brave Search API integrates with the paradigm-based research system
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.search_apis import BraveSearchAPI, SearchConfig, create_search_manager
from services.classification_engine import classification_engine, HostParadigm

async def paradigm_aware_search(query: str):
    """
    Demonstrate how Brave Search can be used with paradigm classification
    """
    print(f"\n{'='*60}")
    print(f"PARADIGM-AWARE SEARCH EXAMPLE")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Step 1: Classify the query
    print("\n1. Classifying query...")
    classification = await classification_engine.classify_query(query)
    
    print(f"\nClassification Results:")
    print(f"  Primary Paradigm: {classification.primary_paradigm.value}")
    if classification.secondary_paradigm:
        print(f"  Secondary Paradigm: {classification.secondary_paradigm.value}")
    print(f"  Confidence: {classification.confidence:.2%}")
    
    # Step 2: Customize search based on paradigm
    print(f"\n2. Customizing search for {classification.primary_paradigm.value} paradigm...")
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        print("Error: BRAVE_SEARCH_API_KEY not set")
        return
    
    brave_api = BraveSearchAPI(api_key=api_key)
    
    # Paradigm-specific search configurations
    paradigm_configs = {
        HostParadigm.DOLORES: {
            "source_types": ["news", "web"],
            "safe_search": "moderate",
            "keywords": ["expose", "investigation", "corruption", "justice"]
        },
        HostParadigm.TEDDY: {
            "source_types": ["web"],
            "safe_search": "strict",
            "keywords": ["help", "support", "community", "resources"]
        },
        HostParadigm.BERNARD: {
            "source_types": ["academic", "web"],
            "safe_search": "moderate",
            "keywords": ["research", "study", "analysis", "data"]
        },
        HostParadigm.MAEVE: {
            "source_types": ["news", "web"],
            "safe_search": "moderate",
            "keywords": ["strategy", "market", "competition", "trends"]
        }
    }
    
    paradigm_config = paradigm_configs.get(classification.primary_paradigm, {})
    
    # Enhance query with paradigm keywords
    enhanced_query = query
    for keyword in paradigm_config.get("keywords", []):
        if keyword.lower() not in query.lower():
            enhanced_query = f"{query} {keyword}"
            break
    
    config = SearchConfig(
        max_results=10,
        source_types=paradigm_config.get("source_types", ["web"]),
        safe_search=paradigm_config.get("safe_search", "moderate")
    )
    
    print(f"  Enhanced Query: {enhanced_query}")
    print(f"  Source Types: {config.source_types}")
    print(f"  Safe Search: {config.safe_search}")
    
    # Step 3: Perform search
    print(f"\n3. Searching with Brave Search API...")
    
    async with brave_api:
        try:
            results = await brave_api.search(enhanced_query, config)
            
            if results:
                print(f"\nFound {len(results)} results:")
                
                # Group by result type
                by_type = {}
                for result in results:
                    if result.result_type not in by_type:
                        by_type[result.result_type] = []
                    by_type[result.result_type].append(result)
                
                # Display results organized by type
                for result_type, type_results in by_type.items():
                    print(f"\n{result_type.upper()} Results ({len(type_results)}):")
                    for i, result in enumerate(type_results[:3], 1):
                        print(f"\n  {i}. {result.title}")
                        print(f"     URL: {result.url}")
                        print(f"     Snippet: {result.snippet[:100]}...")
                
                # Step 4: Paradigm-specific insights
                print(f"\n4. Paradigm-Specific Insights for {classification.primary_paradigm.value}:")
                
                if classification.primary_paradigm == HostParadigm.DOLORES:
                    news_count = len(by_type.get("news", []))
                    print(f"  - Found {news_count} news articles about potential issues")
                    print(f"  - Focus on investigative and exposé content")
                
                elif classification.primary_paradigm == HostParadigm.TEDDY:
                    web_count = len(by_type.get("web", []))
                    print(f"  - Found {web_count} helpful resources")
                    print(f"  - Emphasis on support and community content")
                
                elif classification.primary_paradigm == HostParadigm.BERNARD:
                    academic_count = len(by_type.get("faq", [])) + len(by_type.get("discussion", []))
                    print(f"  - Found {academic_count} analytical/educational results")
                    print(f"  - Focus on research and data-driven content")
                
                elif classification.primary_paradigm == HostParadigm.MAEVE:
                    total_count = len(results)
                    print(f"  - Found {total_count} strategic insights")
                    print(f"  - Emphasis on competitive and market analysis")
                
            else:
                print("No results found")
                
        except Exception as e:
            print(f"Search error: {str(e)}")

async def multi_api_search_example(query: str):
    """
    Demonstrate using Brave Search with other APIs through SearchAPIManager
    """
    print(f"\n{'='*60}")
    print(f"MULTI-API SEARCH EXAMPLE")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Create search manager with all configured APIs
    manager = create_search_manager()
    
    print("\nConfigured Search APIs:")
    for api_name in manager.apis:
        print(f"  - {api_name}")
    
    print(f"\nFallback Order: {' -> '.join(manager.fallback_order)}")
    
    # Search with all APIs
    config = SearchConfig(max_results=5)
    
    print(f"\n1. Searching with all APIs...")
    all_results = await manager.search_all(query, config)
    
    print("\nResults by API:")
    for api_name, results in all_results.items():
        print(f"\n{api_name.upper()}: {len(results)} results")
        if results:
            print(f"  First result: {results[0].title}")
            print(f"  Domain: {results[0].domain}")
    
    # Search with fallback
    print(f"\n2. Testing fallback mechanism...")
    fallback_results = await manager.search_with_fallback(query, config)
    
    if fallback_results:
        print(f"\nFallback search successful!")
        print(f"  Used API: {fallback_results[0].source}")
        print(f"  Found {len(fallback_results)} results")

async def main():
    """Run examples"""
    print("Brave Search API Integration Examples")
    print("====================================")
    
    # Check for API key
    load_dotenv()
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        print("\nError: BRAVE_SEARCH_API_KEY not found in .env file")
        print("Get your API key from: https://api-dashboard.search.brave.com/")
        return
    
    # Example queries for different paradigms
    test_queries = [
        "corporate tax avoidance scandals",  # Dolores
        "how to help homeless communities",   # Teddy
        "quantum computing research papers",  # Bernard
        "best market strategy for startups"   # Maeve
    ]
    
    print("\nSelect an example:")
    print("1. Paradigm-aware search (single query)")
    print("2. Multi-API search comparison")
    print("3. Run all paradigm examples")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        query = input("Enter search query (or press Enter for default): ").strip()
        if not query:
            query = test_queries[0]
        await paradigm_aware_search(query)
    
    elif choice == "2":
        query = input("Enter search query (or press Enter for default): ").strip()
        if not query:
            query = "artificial intelligence trends 2024"
        await multi_api_search_example(query)
    
    elif choice == "3":
        for query in test_queries:
            await paradigm_aware_search(query)
            print("\n" + "-"*60)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Example demonstrating Anthropic Claude Sonnet 4 and Opus 4 integration
with the Four Hosts research application.
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.append('..')
from services.llm_client import llm_client


async def example_anthropic_integration():
    """Demonstrate key features of the Anthropic integration."""
    
    print("🤖 Four Hosts + Anthropic Claude Integration Demo")
    print("=" * 55)
    
    # Check if Anthropic is configured
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set - using mock examples")
        print("   Set environment variable to test with real API")
        print()
    
    # Example 1: Paradigm-specific model selection
    print("1. Paradigm-Specific Model Selection")
    print("-" * 35)
    
    from services.llm_client import _select_model, _PARADIGM_ANTHROPIC_MODEL_MAP
    
    for paradigm, model in _PARADIGM_ANTHROPIC_MODEL_MAP.items():
        print(f"   {paradigm.capitalize():8} → {model}")
    print()
    
    # Example 2: Provider auto-detection
    print("2. Provider Auto-Detection")
    print("-" * 25)
    
    examples = [
        ("claude-3-5-sonnet-20250123", "Sonnet 4 (auto-detected as Anthropic)"),
        ("claude-3-opus-20250123", "Opus 4 (auto-detected as Anthropic)"),
        ("gpt-4o", "GPT-4o (auto-detected as OpenAI)")
    ]
    
    for model, description in examples:
        from services.llm_client import _is_anthropic_model
        provider = "anthropic" if _is_anthropic_model(model) else "openai"
        print(f"   {model:28} → {provider:9} ({description.split('(')[1]}")
    print()
    
    # Example 3: Paradigm-aware content generation (mock)
    print("3. Paradigm-Aware Content Generation Examples")
    print("-" * 44)
    
    research_examples = [
        {
            "paradigm": "dolores",
            "prompt": "Investigate corporate tax avoidance schemes",
            "expected_model": "claude-3-5-sonnet-20250123",
            "style": "Revolutionary, exposing injustice"
        },
        {
            "paradigm": "bernard",
            "prompt": "Analyze statistical patterns in research data",
            "expected_model": "claude-3-opus-20250123", 
            "style": "Analytical, empirical evidence"
        },
        {
            "paradigm": "teddy",
            "prompt": "Provide support resources for mental health",
            "expected_model": "claude-3-5-sonnet-20250123",
            "style": "Compassionate, helpful"
        },
        {
            "paradigm": "maeve",
            "prompt": "Develop competitive market strategy",
            "expected_model": "claude-3-5-sonnet-20250123",
            "style": "Strategic, actionable"
        }
    ]
    
    for example in research_examples:
        print(f"   Paradigm: {example['paradigm'].capitalize()}")
        print(f"   Prompt: {example['prompt']}")
        print(f"   Model: {example['expected_model']}")
        print(f"   Style: {example['style']}")
        
        # Mock API call demonstration
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                # This would make a real API call if credentials are available
                print("   [Making real API call...]")
                response = await llm_client.generate_paradigm_content(
                    prompt=example['prompt'][:50] + "...",  # Truncate for demo
                    paradigm=example['paradigm'],
                    provider="anthropic",
                    max_tokens=100  # Short response for demo
                )
                print(f"   Response preview: {response[:100]}...")
                print()
            except Exception as e:
                print(f"   [API Error: {str(e)[:50]}...]")
                print()
        else:
            print("   [Mock] Would generate paradigm-specific response using Claude")
            print()
    
    # Example 4: Tool calling with Claude
    print("4. Tool Calling with Claude Models")
    print("-" * 33)
    
    search_tool = {
        "type": "function",
        "function": {
            "name": "search_academic_papers",
            "description": "Search academic database for research papers",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "field": {"type": "string", "description": "Academic field"}
                },
                "required": ["query"]
            }
        }
    }
    
    print("   Tool definition:")
    print(f"   - Name: {search_tool['function']['name']}")
    print(f"   - Description: {search_tool['function']['description']}")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            print("   [Making real tool call API request...]")
            result = await llm_client.generate_with_tools(
                prompt="Find recent papers on AI safety research",
                tools=[search_tool],
                paradigm="bernard",  # Analytical paradigm for research
                provider="anthropic"
            )
            print(f"   Content: {result['content'][:100]}...")
            print(f"   Tool calls: {len(result['tool_calls'])} detected")
        except Exception as e:
            print(f"   [API Error: {str(e)[:50]}...]")
    else:
        print("   [Mock] Would invoke Claude tool calling with academic search")
    print()
    
    # Example 5: Multi-provider comparison
    print("5. Multi-Provider Research Comparison")
    print("-" * 36)
    
    comparison_prompt = "What are the key challenges in renewable energy adoption?"
    
    providers_to_test = []
    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append(("OpenAI", "openai"))
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append(("Anthropic", "anthropic"))
    if os.getenv("AZURE_OPENAI_API_KEY"):
        providers_to_test.append(("Azure", "openai"))  # Azure uses OpenAI provider
    
    if providers_to_test:
        print(f"   Testing with prompt: {comparison_prompt}")
        print()
        for provider_name, provider_code in providers_to_test:
            try:
                print(f"   {provider_name} response:")
                response = await llm_client.generate_completion(
                    prompt=comparison_prompt,
                    paradigm="bernard",  # Analytical perspective
                    provider=provider_code,
                    max_tokens=150
                )
                print(f"   {response[:100]}...")
                print()
            except Exception as e:
                print(f"   [Error with {provider_name}: {str(e)[:30]}...]")
                print()
    else:
        print("   [Mock] No API keys configured - would compare responses from:")
        print("   - OpenAI GPT-4o: Technical analysis focused on grid integration")
        print("   - Anthropic Opus: Comprehensive analysis of policy and technical barriers")
        print()
    
    print("✅ Anthropic integration demonstration complete!")
    print()
    print("Configuration tips:")
    print("- Set ANTHROPIC_API_KEY to enable Claude models")
    print("- Use 'provider=\"anthropic\"' parameter for explicit Claude usage")
    print("- Automatic detection works with explicit model names")
    print("- Bernard paradigm gets Opus 4 for maximum analytical capability")


async def example_paradigm_research_workflow():
    """Example of a complete research workflow using different paradigms and providers."""
    
    print("\n🔬 Multi-Paradigm Research Workflow Example")
    print("=" * 45)
    
    research_topic = "Impact of artificial intelligence on employment"
    
    # Step 1: Dolores - Investigative angle (Sonnet 4)
    print("Step 1: Dolores (Revolutionary) - Investigative Research")
    print("- Provider: Anthropic Claude Sonnet 4")
    print("- Focus: Exposing potential systemic issues")
    print("- [Mock] Would investigate: corporate displacement strategies, affected communities")
    print()
    
    # Step 2: Bernard - Analytical deep dive (Opus 4) 
    print("Step 2: Bernard (Analytical) - Statistical Analysis")
    print("- Provider: Anthropic Claude Opus 4")
    print("- Focus: Rigorous data analysis and empirical evidence")
    print("- [Mock] Would analyze: employment statistics, economic modeling, peer-reviewed studies")
    print()
    
    # Step 3: Teddy - Support and solutions (Sonnet 4)
    print("Step 3: Teddy (Devotion) - Support Resources")
    print("- Provider: Anthropic Claude Sonnet 4")
    print("- Focus: Helping affected individuals and communities")
    print("- [Mock] Would provide: retraining programs, support services, community resources")
    print()
    
    # Step 4: Maeve - Strategic recommendations (Sonnet 4)
    print("Step 4: Maeve (Strategic) - Action Plan")
    print("- Provider: Anthropic Claude Sonnet 4") 
    print("- Focus: Actionable strategies and implementation")
    print("- [Mock] Would develop: policy recommendations, corporate strategies, transition plans")
    print()
    
    print("🎯 Result: Comprehensive multi-perspective research leveraging")
    print("   the strengths of different Claude models for each paradigm")


if __name__ == "__main__":
    print("Starting Anthropic Claude integration examples...\n")
    
    try:
        # Run the main integration demo
        asyncio.run(example_anthropic_integration())
        
        # Run the workflow example
        asyncio.run(example_paradigm_research_workflow())
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
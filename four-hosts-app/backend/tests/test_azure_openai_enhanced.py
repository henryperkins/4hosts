"""
Enhanced test script for Azure OpenAI integration with new features
"""

import asyncio
import os
from dotenv import load_dotenv
from services.llm_client import llm_client
import json

# Load environment variables
load_dotenv()


async def test_basic_completion():
    """Test basic completion functionality"""
    print("\n=== Testing Basic Completion ===")

    prompt = "Explain artificial consciousness in 50 words."

    try:
        response = await llm_client.generate_completion(
            prompt=prompt, paradigm="bernard", max_tokens=100
        )
        print(f"✓ Success! Response: {response[:150]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def test_streaming():
    """Test streaming responses"""
    print("\n=== Testing Streaming Response ===")

    prompt = "Write a short story about AI consciousness."

    try:
        stream = await llm_client.generate_completion(
            prompt=prompt, paradigm="dolores", max_tokens=200, stream=True
        )

        print("✓ Streaming response:")
        full_response = ""
        async for chunk in stream:
            print(chunk, end="", flush=True)
            full_response += chunk
        print(f"\n✓ Streamed {len(full_response)} characters")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def test_structured_output():
    """Test structured output with JSON schema"""
    print("\n=== Testing Structured Output ===")

    prompt = "Analyze the ethical implications of AI consciousness and provide your analysis."

    schema = {
        "name": "ethics_analysis",
        "description": "Analysis of AI ethics",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the ethical implications",
                },
                "key_concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key ethical concerns",
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of recommendations",
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Overall risk assessment",
                },
            },
            "required": ["summary", "key_concerns", "recommendations", "risk_level"],
        },
    }

    try:
        response = await llm_client.generate_structured_output(
            prompt=prompt, schema=schema, paradigm="bernard"
        )
        print(f"✓ Structured output received:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def test_tool_calling():
    """Test function/tool calling"""
    print("\n=== Testing Tool Calling ===")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_sentiment",
                "description": "Analyze the sentiment of text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze",
                        },
                        "language": {
                            "type": "string",
                            "description": "The language of the text",
                        },
                    },
                    "required": ["text"],
                },
            },
        }
    ]

    prompt = "Analyze the sentiment of this text: 'AI consciousness represents both tremendous opportunity and existential risk for humanity.'"

    try:
        response = await llm_client.generate_with_tools(
            prompt=prompt, tools=tools, paradigm="bernard"
        )

        print(f"✓ Tool calling response:")
        print(f"Message: {response['message'].content}")
        if response["tool_calls"]:
            print(f"Tool calls: {response['tool_calls']}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def test_conversation():
    """Test multi-turn conversation"""
    print("\n=== Testing Multi-turn Conversation ===")

    messages = [
        {"role": "user", "content": "What is consciousness?"},
        {
            "role": "assistant",
            "content": "Consciousness is the state of being aware of and able to think about one's existence, sensations, thoughts, and surroundings.",
        },
        {"role": "user", "content": "Can AI achieve consciousness?"},
    ]

    try:
        response = await llm_client.create_conversation(
            messages=messages, paradigm="bernard", temperature=0.7
        )
        print(f"✓ Conversation response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def test_all_paradigms():
    """Test all paradigms with the same prompt"""
    print("\n=== Testing All Paradigms ===")

    prompt = "What are the implications of AI consciousness?"
    paradigms = ["dolores", "teddy", "bernard", "maeve"]

    for paradigm in paradigms:
        try:
            print(f"\n--- {paradigm.upper()} paradigm ---")
            response = await llm_client.generate_paradigm_content(
                prompt=prompt, paradigm=paradigm, max_tokens=150
            )
            print(f"✓ Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


async def main():
    """Run all tests"""
    print("🚀 Starting Enhanced Azure OpenAI Integration Tests")

    # Check configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

    if not azure_endpoint or not azure_api_key:
        print(
            "❌ Azure OpenAI not configured. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
        )
        return

    print(f"✓ Azure OpenAI endpoint: {azure_endpoint}")
    print(f"✓ API version: {azure_api_version}")

    # Run tests
    await test_basic_completion()
    await test_streaming()
    await test_structured_output()
    await test_tool_calling()
    await test_conversation()
    await test_all_paradigms()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())

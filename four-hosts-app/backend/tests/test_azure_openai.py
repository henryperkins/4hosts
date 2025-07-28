"""
Test script for Azure OpenAI integration
"""

import asyncio
import os
from dotenv import load_dotenv
from services.llm_client import llm_client

# Load environment variables
load_dotenv()


async def test_azure_openai():
    """Test Azure OpenAI integration"""
    print("Testing Azure OpenAI integration...")

    # Check if Azure OpenAI is configured
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not azure_endpoint or not azure_api_key:
        print(
            "❌ Azure OpenAI not configured. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file"
        )
        return

    print(f"✓ Azure OpenAI endpoint configured: {azure_endpoint}")

    # Test each paradigm
    test_prompt = "Explain the concept of artificial consciousness in 50 words."

    paradigms = ["dolores", "teddy", "bernard", "maeve"]

    for paradigm in paradigms:
        try:
            print(f"\nTesting {paradigm.upper()} paradigm...")
            response = await llm_client.generate_paradigm_content(
                prompt=test_prompt, paradigm=paradigm, max_tokens=100
            )
            print(f"✓ Success! Generated {len(response.split())} words")
            print(f"Response: {response[:100]}...")
        except Exception as e:
            print(f"❌ Error with {paradigm}: {str(e)}")

    print("\n✅ Azure OpenAI integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_azure_openai())

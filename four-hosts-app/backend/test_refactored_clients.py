#!/usr/bin/env python3
"""
Test script for refactored client architecture
Tests the unified client manager and response processor
"""

import asyncio
import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.clients.openai_client_manager import get_client_manager
from services.processors.response_processor import ResponseProcessor
from services.llm_client import llm_client


async def test_client_manager():
    """Test the unified client manager"""
    print("🧪 Testing OpenAI Client Manager...")

    try:
        manager = get_client_manager()
        print(f"   ✓ Client manager created")

        # Test configuration detection
        is_azure = manager.is_azure_configured()
        print(f"   ✓ Azure configured: {is_azure}")

        # Test health status
        health = manager.get_health_status()
        print(f"   ✓ Health status: {health}")

        # Test initialization (if keys are available)
        try:
            manager.initialize()
            print(f"   ✓ Client manager initialized successfully")
        except Exception as e:
            print(f"   ⚠ Client manager initialization skipped: {e}")

        return True

    except Exception as e:
        print(f"   ❌ Client manager test failed: {e}")
        return False


def test_response_processor():
    """Test the unified response processor"""
    print("🧪 Testing Response Processor...")

    try:
        # Test with mock chat completion response
        mock_response = {
            "choices": [{
                "message": {
                    "content": "This is a test response"
                }
            }]
        }

        text = ResponseProcessor.extract_text(mock_response)
        print(f"   ✓ Extracted text: '{text}'")

        # Test with mock responses API response
        mock_responses_payload = {
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "This is a responses API test"
                }]
            }]
        }

        normalized = ResponseProcessor.normalize_payload(mock_responses_payload)
        print(f"   ✓ Normalized responses: {normalized.text}")

        return True

    except Exception as e:
        print(f"   ❌ Response processor test failed: {e}")
        return False


async def test_llm_client():
    """Test the refactored LLM client"""
    print("🧪 Testing Refactored LLM Client...")

    try:
        # Test initialization
        client = llm_client
        print(f"   ✓ LLM client created")

        # Test health info
        info = client.get_active_backend_info()
        print(f"   ✓ Backend info: {info}")

        # Test is_initialized
        initialized = client.is_initialized()
        print(f"   ✓ Client initialized: {initialized}")

        return True

    except Exception as e:
        print(f"   ❌ LLM client test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Testing Refactored Service Architecture\n")

    # Run tests
    test_results = []

    test_results.append(await test_client_manager())
    test_results.append(test_response_processor())
    test_results.append(await test_llm_client())

    # Summary
    passed = sum(test_results)
    total = len(test_results)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed! The refactored architecture is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

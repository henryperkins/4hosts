#!/usr/bin/env python3
"""
Test suite for Anthropic Claude Sonnet 4 and Opus 4 LLM integration
"""

import asyncio
import pytest
import os
from unittest.mock import Mock, AsyncMock, patch
import json

import sys
sys.path.append('..')

from services.llm_client import (
    llm_client, 
    _is_anthropic_model, 
    _select_model,
    _PARADIGM_ANTHROPIC_MODEL_MAP,
    LLMClient
)


class TestAnthropicIntegration:
    """Test Anthropic Claude model integration"""

    def test_anthropic_model_detection(self):
        """Test that Anthropic models are correctly detected"""
        assert _is_anthropic_model("claude-3-5-sonnet-20250123")
        assert _is_anthropic_model("claude-3-opus-20250123")
        assert _is_anthropic_model("claude-3-haiku-20240307")
        assert not _is_anthropic_model("gpt-4o")
        assert not _is_anthropic_model("gpt-4o-mini")
        assert not _is_anthropic_model("o3")

    def test_paradigm_anthropic_model_mapping(self):
        """Test that paradigms map to appropriate Anthropic models"""
        # Check that all paradigms have Anthropic mappings
        paradigms = ["dolores", "teddy", "bernard", "maeve"]
        for paradigm in paradigms:
            assert paradigm in _PARADIGM_ANTHROPIC_MODEL_MAP
            model = _PARADIGM_ANTHROPIC_MODEL_MAP[paradigm]
            assert _is_anthropic_model(model)
            
        # Bernard (analytical) should get Opus 4 for detailed analysis
        assert _PARADIGM_ANTHROPIC_MODEL_MAP["bernard"] == "claude-3-opus-20250123"
        
        # Others should get Sonnet 4
        assert _PARADIGM_ANTHROPIC_MODEL_MAP["dolores"] == "claude-3-5-sonnet-20250123"
        assert _PARADIGM_ANTHROPIC_MODEL_MAP["teddy"] == "claude-3-5-sonnet-20250123"
        assert _PARADIGM_ANTHROPIC_MODEL_MAP["maeve"] == "claude-3-5-sonnet-20250123"

    def test_model_selection_with_provider(self):
        """Test model selection with provider parameter"""
        # Test explicit provider selection
        assert _select_model("bernard", None, "anthropic") == "claude-3-opus-20250123"
        assert _select_model("bernard", None, "openai") == "gpt-4o"
        
        # Test explicit model override
        assert _select_model("bernard", "claude-3-5-sonnet-20250123", "anthropic") == "claude-3-5-sonnet-20250123"
        assert _select_model("bernard", "gpt-4o-mini", "openai") == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_anthropic_client_initialization(self):
        """Test Anthropic client initialization"""
        # Test without API key
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient()
            assert client.anthropic_client is None
            
        # Test with API key
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch('anthropic.AsyncAnthropic') as mock_anthropic:
                client = LLMClient()
                mock_anthropic.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio 
    async def test_anthropic_completion_mock(self):
        """Test Anthropic completion with mocked response"""
        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Test response from Claude")]
        
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response
        
        # Test the completion method
        client = LLMClient()
        client.anthropic_client = mock_client
        
        result = await client._generate_anthropic_completion(
            prompt="Test prompt",
            model="claude-3-5-sonnet-20250123",
            paradigm="bernard",
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
        assert result == "Test response from Claude"
        mock_client.messages.create.assert_called_once()
        
        # Verify call arguments
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-sonnet-20250123"
        assert call_args[1]["max_tokens"] == 100
        assert call_args[1]["temperature"] == 0.7
        assert "analytical researcher" in call_args[1]["system"]  # Bernard's system prompt
        assert call_args[1]["messages"][0]["content"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_anthropic_tool_calling_mock(self):
        """Test Anthropic tool calling with mocked response"""
        # Mock Anthropic response with tool use
        mock_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_1"
        mock_tool_block.name = "search"
        mock_tool_block.input = {"query": "test"}
        
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "I'll search for that information."
        
        mock_response.content = [mock_text_block, mock_tool_block]
        
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response
        
        # Test tool calling
        client = LLMClient()
        client.anthropic_client = mock_client
        
        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        }]
        
        result = await client._generate_anthropic_with_tools(
            prompt="Search for information about AI",
            tools=tools,
            model="claude-3-5-sonnet-20250123",
            paradigm="bernard"
        )
        
        assert result["content"] == "I'll search for that information."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"query": "test"}

    @pytest.mark.asyncio
    async def test_provider_auto_detection(self):
        """Test automatic provider detection from model name"""
        # Create a fresh client instance for this test
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch('anthropic.AsyncAnthropic') as mock_anthropic_class:
                mock_anthropic_client = AsyncMock()
                mock_anthropic_class.return_value = mock_anthropic_client
                
                mock_response = Mock()
                mock_response.content = [Mock(type="text", text="Anthropic response")]
                mock_anthropic_client.messages.create.return_value = mock_response
                
                client = LLMClient()
                
                # Test with explicit Anthropic model - should auto-detect provider
                result = await client.generate_completion(
                    prompt="Test prompt",
                    model="claude-3-5-sonnet-20250123",
                    paradigm="maeve"
                )
                
                assert result == "Anthropic response"
                mock_anthropic_client.messages.create.assert_called_once()

    def test_paradigm_specific_model_assignment(self):
        """Test that paradigms get appropriate Anthropic models"""
        # Bernard (analytical) should get the most capable model (Opus 4)
        bernard_model = _select_model("bernard", None, "anthropic") 
        assert bernard_model == "claude-3-opus-20250123"
        
        # Others should get Sonnet 4 (balanced performance/cost)
        for paradigm in ["dolores", "teddy", "maeve"]:
            model = _select_model(paradigm, None, "anthropic")
            assert model == "claude-3-5-sonnet-20250123"


def main():
    """Run the tests"""
    print("Testing Anthropic Claude Sonnet 4 and Opus 4 integration...")
    
    # Run basic functionality tests
    test_instance = TestAnthropicIntegration()
    
    try:
        test_instance.test_anthropic_model_detection()
        print("✓ Model detection tests passed")
        
        test_instance.test_paradigm_anthropic_model_mapping()
        print("✓ Paradigm model mapping tests passed")
        
        test_instance.test_model_selection_with_provider()
        print("✓ Model selection tests passed")
        
        test_instance.test_paradigm_specific_model_assignment()
        print("✓ Paradigm-specific assignment tests passed")
        
        # Run async tests
        asyncio.run(test_instance.test_anthropic_client_initialization())
        print("✓ Client initialization tests passed")
        
        asyncio.run(test_instance.test_anthropic_completion_mock())
        print("✓ Completion with mocks tests passed")
        
        asyncio.run(test_instance.test_anthropic_tool_calling_mock())
        print("✓ Tool calling with mocks tests passed")
        
        asyncio.run(test_instance.test_provider_auto_detection())
        print("✓ Provider auto-detection tests passed")
        
        print("\n🎉 All Anthropic integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
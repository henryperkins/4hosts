"""
Enhanced tests for the comprehensive groundedness implementation.
Tests all modes: basic, reasoning, and correction with proper mocking.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os

from services.evaluation.context_evaluator import (
    check_content_safety_groundedness,
    _get_groundedness_mode,
    _retry_with_backoff
)


class TestGroundednessMode:
    """Test mode detection logic"""

    def test_basic_mode_default(self):
        """Test that basic mode is default"""
        with patch.dict("os.environ", {}, clear=True):
            assert _get_groundedness_mode() == "basic"

    def test_reasoning_mode(self):
        """Test reasoning mode detection"""
        with patch.dict("os.environ", {"AZURE_CS_GROUNDEDNESS_REASONING": "1"}):
            assert _get_groundedness_mode() == "reasoning"

        with patch.dict("os.environ", {"AZURE_CS_GROUNDEDNESS_REASONING": "true"}):
            assert _get_groundedness_mode() == "reasoning"

    def test_correction_mode(self):
        """Test correction mode detection"""
        with patch.dict("os.environ", {"AZURE_CS_GROUNDEDNESS_CORRECTION": "1"}):
            assert _get_groundedness_mode() == "correction"

    def test_correction_overrides_reasoning(self):
        """Test that correction takes precedence over reasoning"""
        with patch.dict("os.environ", {
            "AZURE_CS_GROUNDEDNESS_REASONING": "1",
            "AZURE_CS_GROUNDEDNESS_CORRECTION": "1"
        }):
            assert _get_groundedness_mode() == "correction"


class TestRetryLogic:
    """Test the retry with backoff functionality"""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt"""
        mock_func = AsyncMock(return_value="success")

        result = await _retry_with_backoff(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failure(self):
        """Test successful execution after failure"""
        mock_func = AsyncMock()
        mock_func.side_effect = [Exception("fail"), "success"]

        result = await _retry_with_backoff(mock_func, max_retries=2, initial_delay=0.01)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test that exception is raised after max retries"""
        mock_func = AsyncMock(side_effect=Exception("always fails"))

        with pytest.raises(Exception) as exc_info:
            await _retry_with_backoff(mock_func, max_retries=2, initial_delay=0.01)

        assert str(exc_info.value) == "always fails"
        assert mock_func.call_count == 2


class TestGroundednessCheck:
    """Test the main groundedness check function"""

    @pytest.mark.asyncio
    async def test_disabled_when_not_configured(self):
        """Test that function returns None when not enabled"""
        with patch.dict("os.environ", {}, clear=True):
            result = await check_content_safety_groundedness(
                "test text",
                ["source text"]
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_disabled_without_endpoint(self):
        """Test that function returns None without endpoint"""
        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1"
        }):
            result = await check_content_safety_groundedness(
                "test text",
                ["source text"]
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_basic_mode_success(self):
        """Test successful basic mode groundedness check"""
        mock_response = {
            "ungroundedDetected": False,
            "ungroundedPercentage": 0.1,
            "ungroundedDetails": []
        }

        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "test-key"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_response_obj = MagicMock()
                mock_response_obj.status = 200
                mock_response_obj.json = AsyncMock(return_value=mock_response)

                mock_session.post.return_value.__aenter__.return_value = mock_response_obj
                mock_session_class.return_value.__aenter__.return_value = mock_session

                result = await check_content_safety_groundedness(
                    "test text",
                    ["source text"]
                )

                assert result is not None
                assert result["ungrounded_detected"] == False
                assert result["ungrounded_percentage"] == 0.1
                assert result["mode"] == "basic"

    @pytest.mark.asyncio
    async def test_reasoning_mode_with_openai(self):
        """Test reasoning mode with Azure OpenAI configured"""
        mock_response = {
            "ungroundedDetected": True,
            "ungroundedPercentage": 0.5,
            "ungroundedDetails": [
                {
                    "text": "incorrect fact",
                    "reason": "Not supported by sources"
                }
            ]
        }

        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "test-key",
            "AZURE_CS_GROUNDEDNESS_REASONING": "1",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_response_obj = MagicMock()
                mock_response_obj.status = 200
                mock_response_obj.json = AsyncMock(return_value=mock_response)

                mock_session.post.return_value.__aenter__.return_value = mock_response_obj
                mock_session_class.return_value.__aenter__.return_value = mock_session

                result = await check_content_safety_groundedness(
                    "test text with incorrect fact",
                    ["source text"]
                )

                assert result is not None
                assert result["mode"] == "reasoning"
                assert result["ungrounded_detected"] == True
                assert "reasoning" in result
                assert "Not supported by sources" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_reasoning_mode_downgrade_without_openai(self):
        """Test that reasoning mode downgrades to basic without OpenAI config"""
        mock_response = {
            "ungroundedDetected": False,
            "ungroundedPercentage": 0.0,
            "ungroundedDetails": []
        }

        # Clear existing AZURE_OPENAI vars and set only CS config
        env_vars = {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "test-key",
            "AZURE_CS_GROUNDEDNESS_REASONING": "1",
            # Explicitly clear Azure OpenAI vars
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_DEPLOYMENT": ""
        }

        with patch.dict("os.environ", env_vars, clear=False):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_response_obj = MagicMock()
                mock_response_obj.status = 200
                mock_response_obj.json = AsyncMock(return_value=mock_response)

                mock_session.post.return_value.__aenter__.return_value = mock_response_obj
                mock_session_class.return_value.__aenter__.return_value = mock_session

                result = await check_content_safety_groundedness(
                    "test text",
                    ["source text"]
                )

                # Verify the request was made without reasoning
                call_args = mock_session.post.call_args
                request_payload = call_args[1]["json"]
                assert "reasoning" not in request_payload
                assert "llmResource" not in request_payload

                assert result["mode"] == "basic"

    @pytest.mark.asyncio
    async def test_correction_mode(self):
        """Test correction mode functionality"""
        mock_response = {
            "ungroundedDetected": True,
            "ungroundedPercentage": 0.3,
            "ungroundedDetails": [
                {"text": "John", "reason": "Name mismatch"}
            ],
            "correctionText": "The patient's name is Jane."
        }

        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "test-key",
            "AZURE_CS_GROUNDEDNESS_CORRECTION": "1",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_response_obj = MagicMock()
                mock_response_obj.status = 200
                mock_response_obj.json = AsyncMock(return_value=mock_response)

                mock_session.post.return_value.__aenter__.return_value = mock_response_obj
                mock_session_class.return_value.__aenter__.return_value = mock_session

                result = await check_content_safety_groundedness(
                    "The patient's name is John.",
                    ["The patient's name is Jane."]
                )

                assert result is not None
                assert result["mode"] == "correction"
                assert result["correction_text"] == "The patient's name is Jane."

    @pytest.mark.asyncio
    async def test_error_handling_401(self):
        """Test handling of 401 authentication error"""
        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "invalid-key"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_response_obj = MagicMock()
                mock_response_obj.status = 401
                mock_response_obj.text = AsyncMock(return_value="Invalid API key")

                mock_session.post.return_value.__aenter__.return_value = mock_response_obj
                mock_session_class.return_value.__aenter__.return_value = mock_session

                result = await check_content_safety_groundedness(
                    "test text",
                    ["source text"]
                )

                assert result is None

    @pytest.mark.asyncio
    async def test_error_handling_429(self):
        """Test handling of 429 rate limit error"""
        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "test-key"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_response_obj = MagicMock()
                mock_response_obj.status = 429
                mock_response_obj.text = AsyncMock(return_value="Rate limit exceeded")

                mock_session.post.return_value.__aenter__.return_value = mock_response_obj
                mock_session_class.return_value.__aenter__.return_value = mock_session

                result = await check_content_safety_groundedness(
                    "test text",
                    ["source text"]
                )

                assert result is None

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of timeout errors"""
        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "test-key",
            "AZURE_HTTP_TIMEOUT_SECONDS": "0.001"  # Very short timeout
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_session.post.side_effect = asyncio.TimeoutError()
                mock_session_class.return_value.__aenter__.return_value = mock_session

                result = await check_content_safety_groundedness(
                    "test text",
                    ["source text"]
                )

                assert result is None

    @pytest.mark.asyncio
    async def test_source_truncation(self):
        """Test that sources are properly truncated"""
        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "test-key"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_response_obj = MagicMock()
                mock_response_obj.status = 200
                mock_response_obj.json = AsyncMock(return_value={
                    "ungroundedDetected": False,
                    "ungroundedPercentage": 0.0,
                    "ungroundedDetails": []
                })

                mock_session.post.return_value.__aenter__.return_value = mock_response_obj
                mock_session_class.return_value.__aenter__.return_value = mock_session

                # Create 25 sources (should be capped at 20)
                many_sources = [f"Source {i}" * 1000 for i in range(25)]

                result = await check_content_safety_groundedness(
                    "test text" * 1000,  # Long text (should be truncated to 7500)
                    many_sources
                )

                # Verify the request
                call_args = mock_session.post.call_args
                request_payload = call_args[1]["json"]

                assert len(request_payload["text"]) <= 7500
                assert len(request_payload["groundingSources"]) <= 20
                assert all(len(s) <= 10000 for s in request_payload["groundingSources"])


class TestIntegrationWithEvaluator:
    """Test integration with the context evaluator"""

    @pytest.mark.asyncio
    async def test_evaluate_context_with_groundedness(self):
        """Test that groundedness check integrates properly with context evaluation"""
        from services.context_packager import ContextPackager
        from services.evaluation.context_evaluator import evaluate_context_package_async

        packager = ContextPackager(total_budget=300)
        package = packager.package(
            instructions=["Follow sources"],
            knowledge=["Test knowledge"],
            tools=[],
            scratchpad=[],
        )

        mock_cs_result = {
            "ungrounded_detected": False,
            "ungrounded_percentage": 0.05,
            "ungrounded_details": [],
            "mode": "basic"
        }

        with patch.dict("os.environ", {
            "AZURE_CS_ENABLE_GROUNDEDNESS": "1",
            "AZURE_CS_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_CS_KEY": "test-key"
        }):
            with patch(
                "services.evaluation.context_evaluator.check_content_safety_groundedness",
                return_value=mock_cs_result
            ):
                report = await evaluate_context_package_async(
                    package,
                    answer="Test answer based on sources",
                    retrieved_documents=["Source document 1", "Source document 2"],
                    check_content_safety=True
                )

                assert report.content_safety_groundedness == 0.95  # 1 - 0.05
                assert report.content_safety_details["mode"] == "basic"
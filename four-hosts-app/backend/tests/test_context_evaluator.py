import pytest
import asyncio
from unittest.mock import patch, MagicMock
from services.context_packager import ContextPackager
from services.evaluation.context_evaluator import (
    evaluate_context_package,
    evaluate_context_package_async,
    check_content_safety_groundedness
)


def test_evaluate_context_package_scores_overlap():
    packager = ContextPackager(total_budget=300)
    package = packager.package(
        instructions=["Follow rigorous sourcing."],
        knowledge=["Carbon emissions are rising rapidly according to UN reports."],
        tools=[],
        scratchpad=[],
    )

    report = evaluate_context_package(
        package,
        answer="Latest analysis shows carbon emissions rising rapidly globally.",
        retrieved_documents=[
            "UN climate report states carbon emissions are rising rapidly across developed nations.",
            "Other article about different topic.",
        ],
    )

    assert report.precision > 0
    assert report.utilization > 0
    assert report.groundedness > 0
    # New fields should be None when using sync version
    assert report.content_safety_groundedness is None
    assert report.content_safety_details is None


@pytest.mark.asyncio
async def test_evaluate_context_package_async_without_content_safety():
    """Test async evaluation without Content Safety configured"""
    packager = ContextPackager(total_budget=300)
    package = packager.package(
        instructions=["Follow rigorous sourcing."],
        knowledge=["Carbon emissions are rising rapidly according to UN reports."],
        tools=[],
        scratchpad=[],
    )

    # Ensure Content Safety is not configured
    with patch.dict("os.environ", {}, clear=True):
        report = await evaluate_context_package_async(
            package,
            answer="Latest analysis shows carbon emissions rising rapidly globally.",
            retrieved_documents=[
                "UN climate report states carbon emissions are rising rapidly across developed nations.",
                "Other article about different topic.",
            ],
            check_content_safety=True
        )

    # Should still have basic scores
    assert report.precision > 0
    assert report.utilization > 0
    assert report.groundedness > 0
    # Content Safety fields should be None when not configured
    assert report.content_safety_groundedness is None
    assert report.content_safety_details is None


@pytest.mark.asyncio
async def test_evaluate_context_package_async_with_content_safety():
    """Test async evaluation with Content Safety mock"""
    packager = ContextPackager(total_budget=300)
    package = packager.package(
        instructions=["Follow rigorous sourcing."],
        knowledge=["The patient name is Jane."],
        tools=[],
        scratchpad=[],
    )

    # Mock Content Safety API response
    async def mock_content_safety_check(text, grounding_sources, task_type="Summarization"):
        return {
            "ungrounded_detected": True,
            "ungrounded_percentage": 0.3,
            "ungrounded_details": [
                {"text": "John", "reason": "Name doesn't match source"}
            ]
        }

    with patch.dict("os.environ", {
        "CONTENT_SAFETY_ENDPOINT": "https://test.cognitiveservices.azure.com/",
        "CONTENT_SAFETY_API_KEY": "test-key"
    }):
        with patch("services.evaluation.context_evaluator.check_content_safety_groundedness", mock_content_safety_check):
            report = await evaluate_context_package_async(
                package,
                answer="The patient name is John.",
                retrieved_documents=["Medical record: Patient Jane, age 45"],
                check_content_safety=True
            )

    # Should have both basic and Content Safety scores
    assert report.precision >= 0
    assert report.utilization >= 0
    assert report.groundedness >= 0
    assert report.content_safety_groundedness == 0.7  # 1 - 0.3
    assert report.content_safety_details["ungrounded_percentage"] == 0.3
    assert len(report.content_safety_details["ungrounded_details"]) == 1


@pytest.mark.asyncio
async def test_check_content_safety_groundedness_direct():
    """Test the Content Safety groundedness check directly"""

    # Test without configuration
    with patch.dict("os.environ", {}, clear=True):
        result = await check_content_safety_groundedness(
            text="Test text",
            grounding_sources=["Source text"]
        )
        assert result is None

    # Test with configuration but mocked HTTP call
    with patch.dict("os.environ", {
        "CONTENT_SAFETY_ENDPOINT": "https://test.cognitiveservices.azure.com/",
        "CONTENT_SAFETY_API_KEY": "test-key"
    }):
        # Mock at the aiohttp level
        import aiohttp
        with patch.object(aiohttp, 'ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200
            async def mock_json():
                return {
                    "ungroundedDetected": False,
                    "ungroundedPercentage": 0.0,
                    "ungroundedDetails": []
                }
            mock_response.json = mock_json

            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await check_content_safety_groundedness(
                text="Test text",
                grounding_sources=["Test text matches source"]
            )

            assert result is not None
            assert result["ungrounded_detected"] == False
            assert result["ungrounded_percentage"] == 0.0


@pytest.mark.asyncio
async def test_content_safety_api_error_handling():
    """Test handling of Content Safety API errors"""

    with patch.dict("os.environ", {
        "CONTENT_SAFETY_ENDPOINT": "https://test.cognitiveservices.azure.com/",
        "CONTENT_SAFETY_API_KEY": "test-key"
    }):
        # Mock at the aiohttp level
        import aiohttp
        with patch.object(aiohttp, 'ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 429  # Too Many Requests
            async def mock_text():
                return "Rate limit exceeded"
            mock_response.text = mock_text

            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Should return None on error, not raise
            result = await check_content_safety_groundedness(
                text="Test text",
                grounding_sources=["Source"]
            )

            assert result is None


def test_evaluation_report_as_dict():
    """Test the as_dict method includes new fields when present"""
    from services.evaluation.context_evaluator import ContextEvaluationReport

    # Test without Content Safety data
    report = ContextEvaluationReport(
        precision=0.8,
        utilization=0.7,
        groundedness=0.9,
        notes=["test note"]
    )

    result_dict = report.as_dict()
    assert result_dict["precision"] == 0.8
    assert result_dict["utilization"] == 0.7
    assert result_dict["groundedness"] == 0.9
    assert "content_safety_groundedness" not in result_dict
    assert "content_safety_details" not in result_dict

    # Test with Content Safety data
    report.content_safety_groundedness = 0.85
    report.content_safety_details = {"ungrounded_percentage": 0.15}

    result_dict = report.as_dict()
    assert result_dict["content_safety_groundedness"] == 0.85
    assert result_dict["content_safety_details"]["ungrounded_percentage"] == 0.15


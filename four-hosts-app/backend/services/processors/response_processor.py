"""
Response Processor
------------------
Unified response extraction and normalization for all OpenAI API responses.
Consolidates logic from llm_client.py and openai_responses_client.py.
"""

import json
from typing import Any, Dict, List, Optional, AsyncIterator

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class Citation(BaseModel):
    """Citation from web search results"""
    url: str
    title: str
    start_index: int
    end_index: int


class ResponsesNormalized(BaseModel):
    """Normalized Responses API content"""
    text: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    web_search_calls: List[Dict[str, Any]] = Field(default_factory=list)


class ResponseProcessor:
    """Unified response extraction and normalization"""

    @staticmethod
    def extract_text(response: Any) -> str:
        """Extract text from various response formats"""
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, dict):
            # Try Responses API format first
            text = ResponseProcessor.extract_responses_final_text(response)
            if text:
                return text.strip()

            # Try Chat Completions format
            choices = response.get("choices", [])
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if content:
                    return str(content).strip()

        # Handle object responses
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                content = choice.message.content
                if content:
                    return str(content).strip()

        return ""

    @staticmethod
    def extract_responses_final_text(payload: Dict[str, Any]) -> Optional[str]:
        """Extract final text from Responses API payload"""
        if not isinstance(payload, dict) or "output" not in payload:
            return None

        for item in reversed(payload.get("output") or []):
            if isinstance(item, dict) and item.get("type") == "message":
                for content in item.get("content", []):
                    if (isinstance(content, dict) and
                            content.get("type") == "output_text"):
                        return content.get("text")
        return None

    @staticmethod
    def extract_citations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from Responses API payload"""
        citations = []
        if not isinstance(payload, dict) or "output" not in payload:
            return citations

        for item in payload.get("output", []):
            if isinstance(item, dict) and item.get("type") == "message":
                for content in item.get("content", []):
                    if (isinstance(content, dict) and
                            content.get("type") == "output_text"):
                        for ann in content.get("annotations", []):
                            if (isinstance(ann, dict) and
                                    ann.get("type") == "url_citation"):
                                citations.append({
                                    "url": ann.get("url"),
                                    "title": ann.get("title"),
                                    "start_index": ann.get("start_index"),
                                    "end_index": ann.get("end_index"),
                                })
        return citations

    @staticmethod
    def extract_tool_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from Responses API payload"""
        tool_calls = []
        if not isinstance(payload, dict) or "output" not in payload:
            return tool_calls

        for item in payload.get("output", []):
            if isinstance(item, dict) and item.get("type") in {
                "web_search_call", "code_interpreter_call", "mcp_call"
            }:
                tool_calls.append(item)
        return tool_calls

    @staticmethod
    def extract_web_search_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract web search calls from Responses API payload"""
        return [
            c for c in ResponseProcessor.extract_tool_calls(payload)
            if isinstance(c, dict) and c.get("type") == "web_search_call"
        ]

    @staticmethod
    def normalize_payload(payload: Dict[str, Any]) -> ResponsesNormalized:
        """Normalize a Responses API payload"""
        return ResponsesNormalized(
            text=ResponseProcessor.extract_responses_final_text(payload),
            citations=ResponseProcessor.extract_citations(payload),
            tool_calls=ResponseProcessor.extract_tool_calls(payload),
            web_search_calls=ResponseProcessor.extract_web_search_calls(payload),
        )

    @staticmethod
    async def process_streaming_chat_response(
        response: AsyncIterator[Any]
    ) -> AsyncIterator[str]:
        """Stream tokens from Chat Completions response"""
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content

    @staticmethod
    async def process_streaming_responses(
        response: AsyncIterator[Any]
    ) -> AsyncIterator[str]:
        """Stream tokens from Responses API"""
        async for event in response:
            if (isinstance(event, dict) and
                    event.get("type") == "response.output_text.delta"):
                if "delta" in event:
                    yield event["delta"]

    @staticmethod
    def extract_structured_json(response: Any) -> Dict[str, Any]:
        """Extract and parse JSON from structured response"""
        text = ResponseProcessor.extract_text(response)
        if not text:
            raise ValueError("No text content found in response")

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output: {e}")
            # Try to extract JSON from partial response
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
            raise ValueError("Response does not contain valid JSON")

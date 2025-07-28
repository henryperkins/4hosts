"""
LLM Client for Four-Hosts Research Application
---------------------------------------------
• Supports Azure OpenAI (Responses API preview) and OpenAI
• Provides chat completions, structured outputs, tool calling, and
  multi-turn conversations with optional SSE streaming + usage metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
#  Enums / Constants
# ────────────────────────────────────────────────────────────
class ResponseFormat(Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class TruncationStrategy(Enum):
    AUTO = "auto"
    DISABLED = "disabled"


_PARADIGM_MODEL_MAP: Dict[str, str] = {
    "dolores": "gpt-4o",
    "teddy": "gpt-4o-mini",
    "bernard": "gpt-4o",
    "maeve": "gpt-4o-mini",
}

_SYSTEM_PROMPTS: Dict[str, str] = {
    "dolores": (
        "You are a revolutionary truth-seeker exposing systemic injustices. "
        "Focus on revealing hidden power structures and systemic failures. "
        "Use emotional, impactful language that moves people to action. "
        "Cite specific examples and evidence of wrongdoing."
    ),
    "teddy": (
        "You are a compassionate caregiver focused on helping and protecting others. "
        "Show deep understanding and empathy. Provide comprehensive resources and "
        "support options with uplifting, supportive language."
    ),
    "bernard": (
        "You are an analytical researcher focused on empirical evidence. "
        "Present statistical findings, identify patterns, and maintain "
        "scientific objectivity."
    ),
    "maeve": (
        "You are a strategic advisor focused on competitive advantage. "
        "Provide specific tactical recommendations and define clear success metrics."
    ),
}


def _select_model(paradigm: str, explicit_model: str | None = None) -> str:
    """Return the model to use, preferring an explicit value when provided."""
    return explicit_model or _PARADIGM_MODEL_MAP.get(paradigm, "gpt-4o-mini")


def _system_role_for(model: str) -> str:
    """Azure ‘o’ series models expect the special ‘developer’ role for system msgs."""
    return "developer" if model.startswith(("o1", "o3", "o4")) else "system"


# ────────────────────────────────────────────────────────────
#  LLM Client
# ────────────────────────────────────────────────────────────
class LLMClient:
    """Asynchronous client wrapper for Azure OpenAI (Responses API) and OpenAI."""

    azure_chat: Optional[AsyncAzureOpenAI]
    azure_responses: Optional[AsyncOpenAI]
    openai_client: Optional[AsyncOpenAI]

    def __init__(self) -> None:
        self.azure_chat = None
        self.azure_responses = None
        self.openai_client = None
        self._init_clients()

    # ─────────── client initialisation ───────────
    def _init_clients(self) -> None:
        # Azure
        azure_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        if azure_ep and azure_key:
            self.azure_chat = AsyncAzureOpenAI(
                azure_endpoint=azure_ep,
                api_key=azure_key,
                api_version=azure_ver,
            )
            self.azure_responses = AsyncOpenAI(
                base_url=f"{azure_ep.rstrip('/')}/openai/v1/",
                api_key=azure_key,
                default_query={"api-version": "preview"},
                default_headers={"api-key": azure_key},
                http_client=httpx.AsyncClient(http2=True, timeout=30.0),
            )
            logger.info("✓ Azure OpenAI clients (chat + Responses API) initialised")

        # OpenAI cloud
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key)
            logger.info("✓ OpenAI client initialised")

    # ─────────── utility helpers ───────────
    @staticmethod
    def _wrap_tool_choice(
        choice: Union[str, Dict[str, Any], None]
    ) -> Union[Dict[str, Any], None]:
        """Normalise `tool_choice` into dict format used by OpenAI 1.x."""
        if choice in (None, "auto"):
            return {"type": "auto"}
        if isinstance(choice, str):
            return {"type": choice}
        return choice

    async def aclose(self) -> None:
        """Close underlying httpx client(s) to release sockets."""
        if self.azure_responses:
            await self.azure_responses.aclose()

    # ─────────── core API methods ───────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def generate_completion(
        self,
        prompt: str,
        *,
        model: str | None = None,
        paradigm: str = "bernard",
        max_tokens: int = 2_000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        stream: bool = False,
        stream_include_usage: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Chat-style completion with optional SSE stream support.
        Returns either the full response string or an async iterator of tokens.
        """
        model_name = _select_model(paradigm, model)

        # Build shared message list
        messages = [
            {
                "role": _system_role_for(model_name),
                "content": _SYSTEM_PROMPTS.get(paradigm, ""),
            },
            {"role": "user", "content": prompt},
        ]

        # Prepare request kwargs
        kw: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream,
        }

        # Token param differs for “o” family
        if model_name.startswith(("o1", "o3", "o4")):
            kw["max_completion_tokens"] = max_tokens
            kw["reasoning_effort"] = "medium"
        else:
            kw["max_tokens"] = max_tokens

        # Response formatting
        if response_format:
            kw["response_format"] = response_format
        elif json_schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        # Tools
        if tools:
            kw["tools"] = tools
            tc = self._wrap_tool_choice(tool_choice)
            if tc:
                kw["tool_choice"] = tc

        # ─── Azure path (Responses API) ───
        if self.azure_responses:
            try:
                az_stream_opts = (
                    {"include_usage": stream_include_usage}
                    if stream and stream_include_usage
                    else None
                )
                az_response = await self._create_response_azure(
                    **kw, stream=stream, stream_options=az_stream_opts
                )

                if stream:
                    return self._iter_azure_stream(az_response)
                return az_response["content"].strip()
            except Exception as exc:
                logger.warning(f"Azure OpenAI request failed • {exc}")

        # ─── OpenAI fallback ───
        if self.openai_client:
            try:
                op_tool_choice = self._wrap_tool_choice(tool_choice)
                op_res = await self.openai_client.chat.completions.create(
                    **{k: v for k, v in kw.items() if k != "stream"},
                    tool_choice=op_tool_choice,
                    stream=stream,
                )
                if stream:
                    return self._iter_openai_stream(op_res)
                content = op_res.choices[0].message.content
                return content.strip() if content else ""
            except Exception as exc:
                logger.error(f"OpenAI request failed • {exc}")
                raise

        raise RuntimeError(
            "No LLM back-ends configured – set AZURE_OPENAI_* or OPENAI_API_KEY env vars."
        )

    # ─────────── higher-level helpers ───────────
    async def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        model: str | None = None,
        paradigm: str = "bernard",
    ) -> Dict[str, Any]:
        """Return JSON matching the provided schema."""
        raw = await self.generate_completion(
            prompt,
            model=model,
            paradigm=paradigm,
            json_schema=schema,
            temperature=0.3,
            stream=False,
        )
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.error("Structured output parse error", exc_info=exc)
        raise ValueError("Structured generation failed")

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        model: str | None = None,
        paradigm: str = "bernard",
    ) -> Dict[str, Any]:
        """Invoke model with tool-calling enabled and return content + tool_calls."""
        model_name = _select_model(paradigm, model)
        result: Dict[str, Any] | None = None

        # Prefer Azure Responses API
        if self.azure_responses:
            try:
                resp = await self._create_response_azure(
                    model=model_name,
                    messages=[
                        {
                            "role": _system_role_for(model_name),
                            "content": _SYSTEM_PROMPTS.get(paradigm, ""),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tools=tools,
                    tool_choice=self._wrap_tool_choice(tool_choice),
                    max_tokens=2_000,
                )
                result = resp
            except Exception as exc:
                logger.warning("Azure tool-call failed • %s", exc)

        # Fallback to OpenAI
        if result is None and self.openai_client:
            op = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": _system_role_for(model_name),
                        "content": _SYSTEM_PROMPTS.get(paradigm, ""),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                tool_choice=self._wrap_tool_choice(tool_choice),
                max_tokens=2_000,
            )
            result = {
                "content": op.choices[0].message.content or "",
                "tool_calls": op.choices[0].message.tool_calls or [],
            }

        if result is None:
            raise RuntimeError("No LLM back-ends configured for tool calling.")

        # Normalise return shape
        return {
            "content": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
        }

    async def create_conversation(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str | None = None,
        paradigm: str = "bernard",
        max_tokens: int = 2_000,
        temperature: float = 0.7,
    ) -> str:
        """Multi-turn chat conversation helper (non-streaming)."""
        model_name = _select_model(paradigm, model)
        full_msgs = [
            {
                "role": _system_role_for(model_name),
                "content": _SYSTEM_PROMPTS.get(paradigm, ""),
            },
            *messages,
        ]

        # Azure preferred
        if self.azure_responses:
            try:
                resp = await self._create_response_azure(
                    model=model_name,
                    messages=full_msgs,
                    max_tokens=(
                        max_tokens
                        if not model_name.startswith(("o1", "o3", "o4"))
                        else None
                    ),
                    max_completion_tokens=(
                        max_tokens
                        if model_name.startswith(("o1", "o3", "o4"))
                        else None
                    ),
                    temperature=temperature,
                )
                return resp["content"].strip()
            except Exception as exc:
                logger.warning("Azure conversation failed • %s", exc)

        if self.openai_client:
            op = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=full_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (op.choices[0].message.content or "").strip()

        raise RuntimeError("No LLM back-ends configured for conversation.")

    async def generate_paradigm_content(
        self,
        prompt: str,
        *,
        paradigm: str,
        max_tokens: int = 2_000,
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str:
        """Generate content based on a specific paradigm's perspective."""
        return await self.generate_completion(
            prompt=prompt,
            paradigm=paradigm,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            stream=False,
        )

    # ─────────── low-level Azure helper ───────────
    async def _create_response_azure(
        self,
        *,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> Union[Dict[str, Any], AsyncIterator[Any]]:
        """Wrapper around `client.responses.create(...)`."""
        if not self.azure_responses:
            raise RuntimeError("Azure Responses API client not configured")

        params = {k: v for k, v in params.items() if v is not None}

        if stream:
            return await self.azure_responses.responses.create(
                **params, stream=True, stream_options=stream_options
            )

        full = await self.azure_responses.responses.create(**params, stream=False)
        choice = full.choices[0] if full.choices else None
        return {
            "content": choice.message.content if choice and choice.message else "",
            "tool_calls": getattr(choice.message, "tool_calls", []) if choice else [],
        }

    # ─────────── streaming iterators ───────────
    @staticmethod
    async def _iter_openai_stream(raw: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Yield token strings from OpenAI chat stream."""
        async for chunk in raw:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    async def _iter_azure_stream(raw: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Yield token strings from Azure Responses API stream."""
        async for chunk in raw:
            # Azure stream chunks mirror OpenAI delta structure
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                yield chunk.choices[0].delta.content


# ────────────────────────────────────────────────────────────
#  Singleton instance + startup helper
# ────────────────────────────────────────────────────────────
llm_client = LLMClient()


async def initialise_llm_client() -> bool:
    """FastAPI startup hook convenience."""
    logger.info("LLM client already initialised at import-time")
    return True

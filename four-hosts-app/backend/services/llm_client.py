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
from openai import AsyncOpenAI
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
    """Asynchronous client wrapper for OpenAI."""

    openai_client: Optional[AsyncOpenAI]

    def __init__(self) -> None:
        self.openai_client = None
        self._init_clients()

    # ─────────── client initialisation ───────────
    def _init_clients(self) -> None:
        # OpenAI cloud
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key)
            logger.info("✓ OpenAI client initialised")
        else:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

    # ─────────── utility helpers ───────────
    @staticmethod
    def _wrap_tool_choice(
        choice: Union[str, Dict[str, Any], None]
    ) -> Union[str, Dict[str, Any], None]:
        """Normalize `tool_choice` for both Azure & OpenAI.

        • OpenAI  → "auto" | "none" | {"type":"function","function": {"name": ...}}
        • Azure   → {"type":"auto"} | {"type":"none"} | {"type":"function","function": {"name": ...}}
        """
        if choice is None or choice == "auto":
            return "auto"          # ← string for OpenAI
        if choice == "none":
            return "none"
        if isinstance(choice, str):
            # explicit function name
            return {"type": "function", "function": {"name": choice}}
        return choice

    async def aclose(self) -> None:
        """Close underlying httpx client(s) to release sockets."""
        # No longer needed as we create httpx clients on demand

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
                kw["tool_choice"] = tc                # keep only when tools exist
        else:
            tool_choice = None                        # ensure we don't forward it

        # ─── OpenAI path ───
        if self.openai_client:
            try:
                # Only include tool_choice if tools were provided
                op_kwargs = {k: v for k, v in kw.items() if k not in ["stream", "tool_choice"]}
                if tools and tool_choice:
                    op_kwargs["tool_choice"] = self._wrap_tool_choice(tool_choice)
                    
                op_res = await self.openai_client.chat.completions.create(
                    **op_kwargs,
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

        wrapped_choice = self._wrap_tool_choice(tool_choice)
        
        # Use OpenAI
        if self.openai_client:
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
                tool_choice=wrapped_choice,       # ← use string/dict directly for OpenAI
                max_tokens=2_000,
            )
            result = {
                "content": op.choices[0].message.content or "",
                "tool_calls": op.choices[0].message.tool_calls or [],
            }
        else:
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


    # ─────────── streaming iterators ───────────
    @staticmethod
    async def _iter_openai_stream(raw: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Yield token strings from OpenAI chat stream."""
        async for chunk in raw:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content



# ────────────────────────────────────────────────────────────
#  Singleton instance + startup helper
# ────────────────────────────────────────────────────────────
llm_client = LLMClient()


async def initialise_llm_client() -> bool:
    """FastAPI startup hook convenience."""
    logger.info("LLM client already initialised at import-time")
    return True

"""Exa Research API integration.

Provides a lightweight client for invoking Exa's research-focused Responses
API models. The client is intentionally decoupled from the main Search API
manager so we can augment Brave search output with higher-quality research
summaries without altering the existing provider orchestration.
"""

from __future__ import annotations

import json
import os
import structlog
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


logger = structlog.get_logger(__name__)


DEFAULT_BASE_URL = "https://api.exa.ai"
DEFAULT_MODEL = "exa-research"
DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass
class ExaResearchOutput:
    """Structured payload returned from Exa research model."""

    summary: str
    key_findings: List[str] = field(default_factory=list)
    supplemental_sources: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)


class ExaResearchClient:
    """Minimal async client for Exa research Responses API."""

    def __init__(self) -> None:
        self.api_key = os.getenv("EXA_API_KEY", "").strip()
        self.base_url = os.getenv("EXA_RESEARCH_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.model = os.getenv("EXA_RESEARCH_MODEL", DEFAULT_MODEL)
        # Default to 120s so we stay within the 90–180s completion window noted in docs
        timeout_raw = os.getenv("EXA_RESEARCH_TIMEOUT", str(int(DEFAULT_TIMEOUT_SECONDS)))
        try:
            self.timeout = float(timeout_raw)
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT_SECONDS

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def supplement_with_research(
        self,
        query: str,
        brave_highlights: List[Dict[str, str]],
        *,
        focus: Optional[str] = None,
    ) -> Optional[ExaResearchOutput]:
        """Run Exa research to augment Brave search findings.

        Args:
            query: Original user question.
            brave_highlights: Representative Brave results (title/url/snippet).
            focus: Optional extra guidance for Exa (e.g., paradigm-specific).
        Returns:
            ExaResearchOutput on success, or None when disabled/failed.
        """

        if not self.is_configured():
            logger.debug("Exa research client skipped", reason="API key not configured")
            return None

        if not query:
            return None

        instructions = self._build_prompt(query, brave_highlights, focus)
        payload = {
            "model": self.model,
            "input": instructions,
            "response_format": self._build_response_format(),
        }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/responses"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Exa research request failed", error=str(exc))
            return None

        data: Dict[str, Any]
        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Failed to decode Exa response JSON", error=str(exc))
            return None

        text = self._extract_output_text(data)
        if not text:
            logger.debug("Empty Exa research output_text")
            return None

        parsed = self._parse_json_payload(text)
        if not parsed:
            logger.debug("Exa research output was not valid JSON",
                        raw_text_preview=text[:200])
            return None

        return ExaResearchOutput(
            summary=str(parsed.get("summary", "")).strip(),
            key_findings=[str(item).strip() for item in parsed.get("key_findings", []) if str(item).strip()],
            supplemental_sources=[
                {
                    "title": str(src.get("title", "")).strip(),
                    "url": str(src.get("url", "")).strip(),
                    "snippet": str(src.get("snippet", "")).strip(),
                }
                for src in parsed.get("supplemental_sources", [])
                if isinstance(src, dict)
            ],
            citations=[
                {
                    "title": str(cite.get("title", "")).strip(),
                    "url": str(cite.get("url", "")).strip(),
                    "note": str(cite.get("note", "")).strip(),
                }
                for cite in parsed.get("citations", [])
                if isinstance(cite, dict) and (cite.get("url") or cite.get("title"))
            ],
        )

    def _build_prompt(
        self,
        query: str,
        highlights: List[Dict[str, str]],
        focus: Optional[str],
    ) -> str:
        """Compose prompt guiding Exa research to complement Brave results."""

        highlight_lines = []
        for idx, item in enumerate(highlights[:5], start=1):
            title = item.get("title") or ""
            url = item.get("url") or ""
            snippet = item.get("snippet") or ""
            highlight_lines.append(
                f"{idx}. Title: {title}\n   URL: {url}\n   Summary: {snippet}"
            )

        context_block = "\n".join(highlight_lines) if highlight_lines else "(No Brave highlights available)"
        focus_line = f"Focus Areas: {focus}" if focus else "Focus Areas: expand with unique insights, data-driven context, and recent developments."

        return (
            "You are an Exa research specialist providing high-precision analysis to complement "
            "general web search findings. Review the Brave search highlights and produce a concise "
            "augmentation that introduces additional perspectives, data points, citations, or sources.\n\n"
            f"Primary Query: {query}\n"
            f"{focus_line}\n\n"
            "Brave Highlights:\n"
            f"{context_block}\n\n"
            "Ensure the response addresses the query directly, surfaces unique key findings, "
            "and includes precise citations for any new facts introduced."
        )

    @staticmethod
    def _build_response_format() -> Dict[str, Any]:
        """Return JSON schema so Exa enforces structured output."""

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "exa_research_supplement",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "supplemental_sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "snippet": {"type": "string"},
                                },
                                "required": ["url"],
                            },
                            "default": [],
                        },
                        "citations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "note": {"type": "string"},
                                },
                                "required": ["url"],
                            },
                            "default": [],
                        },
                    },
                    "required": ["summary"],
                },
            },
        }

    @staticmethod
    def _extract_output_text(payload: Dict[str, Any]) -> str:
        """Best-effort extraction of text from Exa Responses API payload."""

        if not isinstance(payload, dict):
            return ""

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = payload.get("output")
        if isinstance(output_items, list):
            buffer: List[str] = []
            for item in output_items:
                try:
                    contents = item.get("content", []) if isinstance(item, dict) else []
                except AttributeError:
                    continue
                for part in contents:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        txt = part.get("text")
                        if isinstance(txt, str):
                            buffer.append(txt)
            if buffer:
                return "\n".join(buffer).strip()

        return ""

    @staticmethod
    def _parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to salvage JSON by locating first brace pair
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                snippet = text[first : last + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
            return None


exa_research_client = ExaResearchClient()

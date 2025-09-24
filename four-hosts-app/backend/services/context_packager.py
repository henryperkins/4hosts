"""Context packaging helpers for agentic research workflows.

This module introduces a light-weight "context packager" that takes the
outputs of the context-engineering pipeline together with scratchpads,
memory snapshots, and tool schemas and produces a single budgeted payload
ready to be inserted into an LLM prompt.  The approach mirrors the
write/select/compress/isolate (W-S-C-I) vocabulary that the team has been
using and enforces a configurable token budget so we never exceed model
limits when assembling prompts.

The packager is purposely independent from any particular prompt template –
callers receive a structured ``ContextPackage`` that lists what was kept,
what was trimmed, and how many tokens each category consumed.  This makes the
packager suitable for telemetry, A/B experiments, or for swapping in
alternative prompt templates without re-implementing the budgeting logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import structlog

from utils.token_budget import (
    compute_budget_plan,
    estimate_tokens,
    trim_text_to_tokens,
)

logger = structlog.get_logger(__name__)


ContextItem = Dict[str, Any]


@dataclass
class ContextSegment:
    """Represents a logical slice of the packaged context."""

    name: str
    budget_tokens: int
    used_tokens: int
    content: List[ContextItem] = field(default_factory=list)
    dropped: List[ContextItem] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "items": self.content,
            "dropped": self.dropped,
        }


@dataclass
class ContextPackage:
    """Collection of all packaged context segments."""

    total_budget: int
    segments: List[ContextSegment]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_used(self) -> int:
        return sum(seg.used_tokens for seg in self.segments)

    def segment(self, name: str) -> Optional[ContextSegment]:
        for seg in self.segments:
            if seg.name == name:
                return seg
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "total_used": self.total_used,
            "segments": [seg.as_dict() for seg in self.segments],
            "metadata": self.metadata,
        }


def _normalize_items(items: Optional[Iterable[Any]]) -> List[ContextItem]:
    """Normalise caller supplied items into ``ContextItem`` dictionaries."""

    normalised: List[ContextItem] = []
    if not items:
        return normalised

    for raw in items:
        if raw is None:
            continue
        if isinstance(raw, str):
            item = {"content": raw}
        elif isinstance(raw, dict):
            # Copy to avoid mutating caller data
            item = dict(raw)
            if "content" not in item:
                text = item.pop("text", None)
                if text is not None:
                    item["content"] = text
        else:
            item = {"content": str(raw)}
        content = item.get("content", "")
        if not isinstance(content, str):
            content = str(content)
            item["content"] = content
        normalised.append(item)
    return normalised


class ContextPackager:
    """Budget-aware assembler for LLM context windows."""

    def __init__(
        self,
        *,
        total_budget: int = 12000,
        allocation_plan: Optional[Dict[str, float]] = None,
    ) -> None:
        if total_budget <= 0:
            raise ValueError("total_budget must be positive")
        self.total_budget = total_budget
        self.allocation_plan = allocation_plan or {
            "instructions": 0.18,
            "knowledge": 0.62,
            "tools": 0.12,
            "scratch": 0.08,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def package(
        self,
        *,
        instructions: Sequence[Any],
        knowledge: Sequence[Any],
        tools: Sequence[Any],
        scratchpad: Sequence[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextPackage:
        """Package provided segments into a budget respecting bundle."""

        metadata = dict(metadata or {})
        buckets = {
            "instructions": _normalize_items(instructions),
            "knowledge": _normalize_items(knowledge),
            "tools": _normalize_items(tools),
            "scratch": _normalize_items(scratchpad),
        }

        plan = compute_budget_plan(self.total_budget, self.allocation_plan)
        segments: List[ContextSegment] = []

        for name, bucket_items in buckets.items():
            budget = int(plan.get(name, 0))
            segment = self._fill_segment(name, bucket_items, budget)
            segments.append(segment)
        return ContextPackage(total_budget=self.total_budget, segments=segments, metadata=metadata)

    def build_from_context(
        self,
        context_engineered: Any,
        *,
        base_instructions: Optional[str] = None,
        memory_items: Optional[Sequence[Any]] = None,
        tool_schemas: Optional[Sequence[Any]] = None,
        scratchpad: Optional[Sequence[Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextPackage:
        """Convenience wrapper for ``ContextEngineeredQuery`` outputs."""

        knowledge_blocks: List[str] = []
        instructions_blocks: List[str] = []

        try:
            if base_instructions:
                instructions_blocks.append(base_instructions)
        except Exception:
            pass

        try:
            write_output = getattr(context_engineered, "write_output", None)
            if write_output:
                focus = getattr(write_output, "documentation_focus", "")
                if focus:
                    instructions_blocks.append(f"Paradigm focus: {focus}")
                key_themes = getattr(write_output, "key_themes", []) or []
                if key_themes:
                    instructions_blocks.append("Key themes: " + ", ".join(key_themes))
        except Exception as exc:
            logger.debug("context_packager.write_output_error", error=str(exc))

        try:
            select_output = getattr(context_engineered, "select_output", None)
            if select_output:
                queries = getattr(select_output, "search_queries", None) or []
                if queries:
                    formatted = []
                    for q in queries:
                        if isinstance(q, dict):
                            query_text = q.get("query") or ""
                            if query_text:
                                formatted.append(f"• {query_text}")
                    if formatted:
                        knowledge_blocks.append("Preferred search queries:\n" + "\n".join(formatted))
        except Exception as exc:
            logger.debug("context_packager.select_output_error", error=str(exc))

        try:
            isolate_output = getattr(context_engineered, "isolate_output", None)
            if isolate_output:
                focus_areas = getattr(isolate_output, "focus_areas", []) or []
                if focus_areas:
                    knowledge_blocks.append(
                        "Focus areas for extraction:\n" + "\n".join(f"- {fa}" for fa in focus_areas)
                    )
        except Exception as exc:
            logger.debug("context_packager.isolate_output_error", error=str(exc))

        compress_notes: List[str] = []
        try:
            compress_output = getattr(context_engineered, "compress_output", None)
            if compress_output:
                strategy = getattr(compress_output, "compression_strategy", "")
                if strategy:
                    compress_notes.append(f"Compression strategy: {strategy}")
                priority = getattr(compress_output, "priority_elements", []) or []
                if priority:
                    compress_notes.append("Prioritise: " + ", ".join(priority))
        except Exception:
            pass

        instructions_blocks.extend(compress_notes)

        metadata = extra_metadata or {}
        metadata = dict(metadata)
        try:
            metadata.setdefault(
                "layer_durations",
                getattr(context_engineered, "layer_durations", {}),
            )
            metadata.setdefault("paradigm", getattr(getattr(context_engineered, "classification", None), "primary_paradigm", None))
        except Exception:
            pass

        return self.package(
            instructions=instructions_blocks,
            knowledge=knowledge_blocks,
            tools=list(tool_schemas or []),
            scratchpad=list(memory_items or []) + list(scratchpad or []),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_segment(
        self,
        name: str,
        items: List[ContextItem],
        budget_tokens: int,
    ) -> ContextSegment:
        used = 0
        kept: List[ContextItem] = []
        dropped: List[ContextItem] = []

        if budget_tokens <= 0:
            return ContextSegment(name=name, budget_tokens=0, used_tokens=0, content=kept, dropped=items)

        for item in items:
            text = item.get("content", "")
            token_estimate = estimate_tokens(text)
            if used + token_estimate <= budget_tokens:
                kept.append({**item, "tokens": token_estimate})
                used += token_estimate
                continue

            # Not enough budget.  Attempt to partially trim once.
            remaining = max(0, budget_tokens - used)
            if remaining <= 0:
                dropped.append({**item, "tokens": token_estimate, "dropped_reason": "budget_exhausted"})
                continue

            trimmed_text = trim_text_to_tokens(text, remaining)
            trimmed_tokens = estimate_tokens(trimmed_text)
            if trimmed_tokens == 0:
                dropped.append({**item, "tokens": token_estimate, "dropped_reason": "too_large_to_trim"})
                continue

            kept.append({**item, "content": trimmed_text, "tokens": trimmed_tokens, "trimmed": True})
            used += trimmed_tokens
            dropped.append({**item, "tokens": token_estimate, "dropped_reason": "trimmed", "kept_tokens": trimmed_tokens})

        return ContextSegment(
            name=name,
            budget_tokens=budget_tokens,
            used_tokens=used,
            content=kept,
            dropped=dropped,
        )


# Convenience default instance shared by orchestrator/tests
default_packager = ContextPackager()


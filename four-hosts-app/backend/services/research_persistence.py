"""Persistence helpers for research lifecycle.

Synchronises Redis-backed research metadata with PostgreSQL so that
long-lived records survive cache eviction. All writes are best-effort:
errors are logged but never allowed to break the request flow.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from uuid import UUID, uuid5

from sqlalchemy import delete, select

from database.connection import get_db_context
from database.models import (
    ParadigmType,
    ResearchAnswer as DBResearchAnswer,
    ResearchDepth as DBResearchDepth,
    ResearchQuery as DBResearchQuery,
    ResearchSource as DBResearchSource,
    ResearchStatus as DBResearchStatus,
)
from models.base import Paradigm, ResearchDepth as ApiResearchDepth
from utils.date_utils import safe_parse_date

import structlog

logger = structlog.get_logger(__name__)

# Deterministic namespace for mapping external "res_" identifiers to UUIDs
_RESEARCH_NAMESPACE = UUID("7a1c0f50-9f9d-4f0d-80a1-21e9f8e8b7f5")


def _research_uuid(research_id: str) -> UUID:
    """Derive a stable UUID for a string research identifier."""
    return uuid5(_RESEARCH_NAMESPACE, research_id)


def _map_depth(depth: ApiResearchDepth) -> DBResearchDepth:
    """Map API depth (which includes deep_research) to DB enum."""
    if depth == ApiResearchDepth.QUICK:
        return DBResearchDepth.QUICK
    if depth == ApiResearchDepth.DEEP or depth == ApiResearchDepth.DEEP_RESEARCH:
        return DBResearchDepth.DEEP
    return DBResearchDepth.STANDARD


def _safe_paradigm(value: Optional[Paradigm]) -> Optional[ParadigmType]:
    if not value:
        return None
    try:
        return ParadigmType(value.value if isinstance(value, Paradigm) else str(value))
    except Exception:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def record_submission(
    *,
    research_id: str,
    user_id: str,
    query_text: str,
    options: Dict[str, Any],
    classification: Dict[str, Any],
) -> None:
    """Persist the initial research submission in PostgreSQL."""
    try:
        db_id = _research_uuid(research_id)
        user_uuid = UUID(str(user_id))
        depth = _map_depth(ApiResearchDepth(options.get("depth", ApiResearchDepth.STANDARD)))

        primary = classification.get("primary")
        secondary = classification.get("secondary")
        distribution = classification.get("distribution")
        confidence = classification.get("confidence")

        async with get_db_context() as session:
            db_obj = await session.get(DBResearchQuery, db_id)
            if not db_obj:
                db_obj = DBResearchQuery(id=db_id, user_id=user_uuid)
                session.add(db_obj)

            db_obj.user_id = user_uuid
            db_obj.query_text = query_text
            db_obj.query_hash = hashlib.sha256(query_text.encode()).hexdigest()
            db_obj.language = options.get("language", "en")
            db_obj.region = options.get("region")
            db_obj.primary_paradigm = _safe_paradigm(primary) or ParadigmType.BERNARD
            db_obj.secondary_paradigm = _safe_paradigm(secondary)
            db_obj.paradigm_scores = distribution or {}
            if isinstance(confidence, (int, float)):
                db_obj.classification_confidence = float(confidence)
            db_obj.paradigm_override = _safe_paradigm(options.get("paradigm_override"))
            db_obj.depth = depth
            db_obj.max_sources = int(options.get("max_sources", 100))
            db_obj.include_secondary = bool(options.get("include_secondary", True))
            db_obj.custom_prompts = options.get("custom_prompts") or {}
            db_obj.status = DBResearchStatus.PROCESSING
            db_obj.progress = 0
            db_obj.current_phase = "classification"
            db_obj.started_at = _now()
    except Exception:
        logger.exception("research_persistence.submission_failed", extra={"research_id": research_id})


def _extract_float(mapping: Dict[str, Any], key: str) -> Optional[float]:
    value = mapping.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_sections(answer: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    sections = answer.get("sections")
    if isinstance(sections, list):
        return sections
    return []


def _extract_action_items(answer: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    items = answer.get("action_items")
    if isinstance(items, list):
        return items
    return []




async def persist_completion(
    *,
    research_id: str,
    user_id: str,
    final_result: Dict[str, Any],
) -> None:
    """Persist the completed research answer and supporting sources."""
    try:
        db_id = _research_uuid(research_id)
        user_uuid = UUID(str(user_id))
        answer = final_result.get("answer") or {}
        metadata = final_result.get("metadata") or {}
        sources = final_result.get("sources") or []

        async with get_db_context() as session:
            db_query = await session.get(DBResearchQuery, db_id)
            if not db_query:
                db_query = DBResearchQuery(id=db_id, user_id=user_uuid, query_text=final_result.get("query", ""))
                session.add(db_query)

            db_query.user_id = user_uuid
            db_query.status = DBResearchStatus.COMPLETED
            db_query.progress = 100
            db_query.completed_at = _now()
            db_query.sources_found = len(sources)
            db_query.sources_analyzed = len(sources)
            db_query.synthesis_score = _extract_float(metadata, "synthesis_score")
            db_query.confidence_score = _extract_float(metadata, "confidence_score")
            proc_secs = _extract_float(metadata, "processing_time_seconds")
            if proc_secs is not None:
                db_query.duration_seconds = proc_secs

            # Upsert research answer
            existing_answer = await session.execute(
                select(DBResearchAnswer).where(DBResearchAnswer.research_id == db_id)
            )
            db_answer = existing_answer.scalars().first()
            if not db_answer:
                db_answer = DBResearchAnswer(research_id=db_id)
                session.add(db_answer)

            db_answer.executive_summary = answer.get("summary")
            db_answer.paradigm_summary = final_result.get("paradigm_analysis") or {}
            db_answer.sections = list(_extract_sections(answer))
            db_answer.action_items = list(_extract_action_items(answer))
            answer_meta = answer.get("metadata") or {}
            db_answer.key_insights = answer_meta.get("key_insights") or []
            db_answer.synthesis_quality_score = _extract_float(metadata, "synthesis_quality")
            db_answer.confidence_score = _extract_float(metadata, "confidence_score")
            db_answer.completeness_score = _extract_float(metadata, "completeness_score")
            db_answer.generation_model = metadata.get("generation_model")
            db_answer.generation_time_ms = metadata.get("generation_time_ms")
            db_answer.token_count = metadata.get("token_count")
            db_answer.updated_at = _now()

            # Replace sources for this research
            await session.execute(
                delete(DBResearchSource).where(DBResearchSource.research_id == db_id)
            )

            for raw in sources:
                try:
                    source = DBResearchSource(
                        research_id=db_id,
                        url=str(raw.get("url", "")),
                        title=raw.get("title"),
                        domain=raw.get("domain"),
                        content_snippet=raw.get("snippet"),
                        credibility_score=_extract_float(raw, "credibility_score"),
                        relevance_score=_extract_float(raw, "relevance_score"),
                        bias_score=_extract_float(raw, "bias_score"),
                        source_type=raw.get("source_type"),
                        source_metadata=raw,
                        published_date=safe_parse_date(raw.get("published_date")),
                        found_at=_now(),
                    )
                    session.add(source)
                except Exception:
                    logger.debug(
                        "research_persistence.source_skip",
                        exc_info=True,
                        extra={"research_id": research_id, "source": raw.get("url", "")},
                    )
    except Exception:
        logger.exception("research_persistence.completion_failed", extra={"research_id": research_id})


async def persist_failure(
    *,
    research_id: str,
    user_id: str,
    error: str,
) -> None:
    """Mark a research job as failed in PostgreSQL."""
    try:
        db_id = _research_uuid(research_id)
        user_uuid = UUID(str(user_id))
        async with get_db_context() as session:
            db_query = await session.get(DBResearchQuery, db_id)
            if not db_query:
                db_query = DBResearchQuery(id=db_id, user_id=user_uuid)
                session.add(db_query)

            db_query.user_id = user_uuid
            db_query.status = DBResearchStatus.FAILED
            db_query.error_message = error[:2048]
            db_query.completed_at = _now()
    except Exception:
        logger.exception("research_persistence.failure_mark_failed", extra={"research_id": research_id})


__all__ = [
    "record_submission",
    "persist_completion",
    "persist_failure",
]

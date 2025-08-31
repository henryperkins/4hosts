"""
Research routes for the Four Hosts Research API
"""

import logging
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

from models.research import (
    ResearchQuery,
    ResearchOptions,
    ResearchResult,
    ParadigmOverrideRequest
)
from models.base import (
    ResearchDepth,
    ResearchStatus,
    UserRole,
    HOST_TO_MAIN_PARADIGM,
    ParadigmClassification
)
from core.dependencies import get_current_user
from services.research_store import research_store
from services.websocket_service import (
    ResearchProgressTracker,
    progress_tracker as _ws_progress_tracker,
    WSEventType,
    WSMessage,
)
from services.enhanced_integration import (
    enhanced_classification_engine as classification_engine,
)
from services.context_engineering import context_pipeline
from services.research_orchestrator import research_orchestrator
from models.base import HOST_TO_MAIN_PARADIGM, Paradigm
from dataclasses import asdict
from services.webhook_manager import WebhookManager, WebhookEvent
from services.export_service import ExportOptions, ExportFormat, ExportService
import os

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/research", tags=["research"])


# Mock services for now - these will be injected
progress_tracker = _ws_progress_tracker
webhook_manager = None


async def execute_real_research(
    research_id: str,
    research: ResearchQuery,
    user_id: str,
    user_role: UserRole | str = UserRole.PRO,
):
    """Execute the full research pipeline and persist results.

    Steps:
    - Classify query (enhanced engine)
    - Context engineering (W-S-C-I pipeline)
    - Search + retrieval via UnifiedResearchOrchestrator (with WS progress)
    - Paradigm-aware synthesis (enhanced answer orchestrator)
    - Persist results to research_store and broadcast completion
    """
    try:
        # Mark in-progress
        await research_store.update_field(research_id, "status", ResearchStatus.IN_PROGRESS)

        # 1) Classification
        if progress_tracker:
            await progress_tracker.update_progress(research_id, "classification", 5)

        cls = await classification_engine.classify_query(research.query)

        # Stream classification summary
        try:
            if progress_tracker:
                primary = getattr(cls, "primary_paradigm", None)
                secondary = getattr(cls, "secondary_paradigm", None)
                confidence = float(getattr(cls, "confidence", 0.0) or 0.0)
                dist = getattr(cls, "distribution", {}) or {}
                # Build a compact distribution preview
                dist_preview = ", ".join(
                    f"{p.name.lower()}: {dist.get(p, 0.0):.2f}" for p in list(dist.keys())[:2]
                ) if hasattr(primary, 'name') else ""
                await progress_tracker.update_progress(
                    research_id,
                    None,
                    8,
                    custom_data={
                        "message": (
                            f"Classification → primary: {getattr(primary, 'name', primary)}, "
                            f"secondary: {getattr(secondary, 'name', secondary) or 'n/a'}, "
                            f"confidence: {confidence:.2f}"
                            + (f"; dist: {dist_preview}" if dist_preview else "")
                        )
                    },
                )
        except Exception:
            pass

        # 2) Context Engineering
        if progress_tracker:
            await progress_tracker.update_progress(research_id, "context_engineering", 15)

        # Use global pipeline to accumulate metrics
        ce = await context_pipeline.process_query(cls)

        # Stream context engineering summary for transparency
        try:
            if progress_tracker and ce:
                doc_focus = getattr(ce.write_output, "documentation_focus", "") if hasattr(ce, "write_output") else ""
                comp_ratio = getattr(ce.compress_output, "compression_ratio", None) if hasattr(ce, "compress_output") else None
                token_budget = getattr(ce.compress_output, "token_budget", None) if hasattr(ce, "compress_output") else None
                iso_strategy = getattr(ce.isolate_output, "isolation_strategy", "") if hasattr(ce, "isolate_output") else ""
                searches = len(getattr(ce.select_output, "search_queries", []) or []) if hasattr(ce, "select_output") else 0
                await progress_tracker.update_progress(
                    research_id,
                    None,
                    20,
                    custom_data={
                        "message": (
                            "Context: "
                            + (f"focus='{doc_focus[:40]}', " if doc_focus else "")
                            + (f"search_queries={searches}, " if searches else "")
                            + (f"compression={comp_ratio*100:.0f}%, " if isinstance(comp_ratio, (int, float)) else "")
                            + (f"token_budget={token_budget}, " if token_budget else "")
                            + (f"isolation='{iso_strategy}'" if iso_strategy else "")
                        ).rstrip(', ')
                    },
                )
        except Exception:
            pass

        # 3) Search, Retrieval, and Synthesis via unified orchestrator
        if progress_tracker:
            await progress_tracker.update_progress(research_id, "search_retrieval", 25)

        class _UserCtxShim:
            def __init__(self, role: str, source_limit: int):
                self.role = role
                self.source_limit = source_limit

        # Resolve user role string (e.g., 'PRO') for orchestrator context
        user_role_name = (
            user_role.name if hasattr(user_role, "name") else str(user_role).upper()
        ) or "PRO"
        max_sources = int(getattr(research, "options", ResearchOptions()).max_sources)
        user_ctx = _UserCtxShim(user_role_name, max_sources)

        orch_resp = await research_orchestrator.execute_research(
            classification=cls,
            context_engineered=ce,  # legacy shape is supported by the orchestrator
            user_context=user_ctx,
            progress_callback=progress_tracker,
            research_id=research_id,
            enable_deep_research=(research.options.depth == ResearchDepth.DEEP_RESEARCH),
            deep_research_mode=None,
            synthesize_answer=True,
            answer_options={"research_id": research_id, "max_length": 2000},
        )

        # Normalize answer for frontend
        answer_obj = orch_resp.get("answer")
        # Normalize answer to frontend-friendly primitives
        sections_payload = []
        citations_payload = []
        summary_text = ""
        if answer_obj:
            try:
                summary_text = (
                    getattr(answer_obj, "summary", None)
                    or getattr(answer_obj, "content_md", None)
                    or ""
                )

                # Sections
                raw_sections = getattr(answer_obj, "sections", []) or []
                for s in raw_sections:
                    try:
                        sections_payload.append(
                            {
                                "title": getattr(s, "title", ""),
                                "paradigm": getattr(s, "paradigm", "bernard"),
                                "content": getattr(s, "content", ""),
                                "confidence": float(getattr(s, "confidence", 0.8) or 0.8),
                                # Approximate: use number of citations referenced in the section
                                "sources_count": len(getattr(s, "citations", []) or []),
                                "citations": list(getattr(s, "citations", []) or []),
                                "key_insights": list(getattr(s, "key_insights", []) or []),
                            }
                        )
                    except Exception:
                        continue

                # Citations (answer_obj.citations is a dict[str, Citation])
                raw_citations = getattr(answer_obj, "citations", {}) or {}
                if isinstance(raw_citations, dict):
                    primary_paradigm = HOST_TO_MAIN_PARADIGM.get(
                        getattr(cls, "primary_paradigm", None), Paradigm.BERNARD
                    ).value
                    for c in raw_citations.values():
                        try:
                            citations_payload.append(
                                {
                                    "id": getattr(c, "id", ""),
                                    "source": getattr(c, "domain", ""),
                                    "title": getattr(c, "source_title", ""),
                                    "url": getattr(c, "source_url", ""),
                                    "credibility_score": float(
                                        getattr(c, "credibility_score", 0.5) or 0.5
                                    ),
                                    "paradigm_alignment": primary_paradigm,
                                }
                            )
                        except Exception:
                            continue
            except Exception:
                pass

        answer_payload = {
            "summary": summary_text,
            "sections": sections_payload,
            "action_items": list(getattr(answer_obj, "action_items", []) or []),
            "citations": citations_payload,
            "metadata": getattr(answer_obj, "metadata", {}) or {},
        }

        # Convert sources from orchestrator response results (already compressed/normalized)
        sources_payload = []
        for item in orch_resp.get("results", []) or []:
            try:
                md = item.get("metadata", {}) or {}
                sources_payload.append(
                    {
                        "title": item.get("title", "") or "",
                        "url": item.get("url", "") or "",
                        "snippet": item.get("snippet", "") or "",
                        "domain": md.get("domain", "") or "",
                        "credibility_score": float(item.get("credibility_score", 0.5) or 0.5),
                        "published_date": md.get("published_date"),
                        "source_type": md.get("result_type", "web"),
                    }
                )
            except Exception:
                continue

        # Build paradigm analysis summary for UI
        paradigm_analysis = {
            "primary": {
                "paradigm": HOST_TO_MAIN_PARADIGM.get(cls.primary_paradigm, Paradigm.BERNARD).value,
                "confidence": float(getattr(cls, "confidence", 0.75) or 0.75),
                "approach": getattr(ce.write_output, "documentation_focus", "")[:140] if hasattr(ce, "write_output") else "",
                "focus": ", ".join((getattr(ce.write_output, "key_themes", []) or [])[:3]) if hasattr(ce, "write_output") else "",
            }
        }

        # Aggregate metadata
        exec_meta = orch_resp.get("metadata", {}) or {}
        # Compose SSOTA context_layers summary
        context_layers = {
            "write_focus": getattr(ce.write_output, "documentation_focus", "") if hasattr(ce, "write_output") else "",
            "compression_ratio": float(getattr(ce.compress_output, "compression_ratio", 0.0) or 0.0) if hasattr(ce, "compress_output") else 0.0,
            "token_budget": int(getattr(ce.compress_output, "token_budget", 0) or 0) if hasattr(ce, "compress_output") else 0,
            "isolation_strategy": getattr(ce.isolate_output, "isolation_strategy", "") if hasattr(ce, "isolate_output") else "",
            "search_queries_count": len(getattr(ce.select_output, "search_queries", []) or []) if hasattr(ce, "select_output") else 0,
        }

        # Attach agent trace from orchestrator if present
        agent_trace = list(exec_meta.get("agent_trace", []) or []) if isinstance(exec_meta, dict) else []

        metadata = {
            "total_sources_analyzed": len(sources_payload),
            "high_quality_sources": sum(1 for s in sources_payload if s.get("credibility_score", 0) >= 0.7),
            "search_queries_executed": context_layers["search_queries_count"],
            "processing_time_seconds": float(exec_meta.get("processing_time", 0.0) or 0.0),
            "synthesis_quality": getattr(answer_obj, "synthesis_quality", None),
            "paradigms_used": [paradigm_analysis["primary"]["paradigm"]],
            "research_depth": research.options.depth.value if hasattr(research, "options") else "standard",
            "context_layers": context_layers,
            "agent_trace": agent_trace,
            "contradictions": exec_meta.get("contradictions", {"count": 0, "examples": []}),
        }

        # Stream a compact agent trace snapshot for visibility
        try:
            if progress_tracker:
                trace = list(exec_meta.get("agent_trace", []) or [])
                for entry in trace[:5]:
                    if not isinstance(entry, dict):
                        continue
                    action = entry.get("action") or entry.get("name") or "step"
                    duration = entry.get("duration_ms") or entry.get("duration")
                    message = f"Agent: {action}"
                    if duration:
                        try:
                            ms = int(duration)
                            message += f" ({ms} ms)"
                        except Exception:
                            pass
                    await progress_tracker.update_progress(
                        research_id,
                        None,
                        None,
                        custom_data={"message": message},
                    )
        except Exception:
            pass

        final_result = {
            "research_id": research_id,
            "query": research.query,
            "status": ResearchStatus.COMPLETED,
            "paradigm_analysis": paradigm_analysis,
            "answer": answer_payload,
            "sources": sources_payload,
            "metadata": metadata,
            "cost_info": {},
            "export_formats": {
                "pdf": f"/v1/research/{research_id}/export/pdf",
                "markdown": f"/v1/research/{research_id}/export/markdown",
                "json": f"/v1/research/{research_id}/export/json",
            },
        }

        await research_store.update_field(research_id, "results", final_result)
        await research_store.update_field(research_id, "status", ResearchStatus.COMPLETED)

        if progress_tracker:
            await progress_tracker.update_progress(research_id, "finalizing", 98)
            await progress_tracker.complete_research(research_id, {"summary": answer_payload.get("summary", "")[:200]})

    except Exception as e:
        # Log full traceback for easier debugging and include exception type
        logger.exception("execute_real_research failed for %s: %s", research_id, str(e))
        await research_store.update_field(research_id, "status", ResearchStatus.FAILED)
        await research_store.update_field(
            research_id,
            "error",
            f"{e.__class__.__name__}: {str(e)}",
        )
        if progress_tracker:
            await progress_tracker.fail_research(research_id, str(e))


@router.post("/query")
async def submit_research(
    research: ResearchQuery,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    """Submit a research query for paradigm-based analysis"""
    # Check role requirements for research depth
    if research.options.depth in [ResearchDepth.DEEP, ResearchDepth.DEEP_RESEARCH]:
        # Deep research requires at least PRO role
        if current_user.role not in [
            UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN
        ]:
            raise HTTPException(
                status_code=403,
                detail="Deep research requires PRO subscription or higher",
            )

    research_id = f"res_{uuid.uuid4().hex[:12]}"

    try:
        # Classify the query using the new classification engine
        classification_result = await classification_engine.classify_query(
            research.query
        )

        # Convert to the old format for compatibility
        classification = ParadigmClassification(
            primary=HOST_TO_MAIN_PARADIGM[classification_result.primary_paradigm],
            secondary=(
                HOST_TO_MAIN_PARADIGM.get(classification_result.secondary_paradigm)
                if classification_result.secondary_paradigm
                else None
            ),
            distribution={
                HOST_TO_MAIN_PARADIGM[p].value: v
                for p, v in classification_result.distribution.items()
            },
            confidence=classification_result.confidence,
            explanation={
                HOST_TO_MAIN_PARADIGM[p].value: "; ".join(r)
                for p, r in classification_result.reasoning.items()
            },
        )

        # Store research request
        research_data = {
            "id": research_id,
            "user_id": str(current_user.user_id),
            "query": research.query,
            "options": research.options.dict(),
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "created_at": datetime.utcnow().isoformat(),
            "results": None,
        }
        await research_store.set(research_id, research_data)

        # Execute real research
        background_tasks.add_task(
            execute_real_research,
            research_id,
            research,
            str(current_user.user_id),
            current_user.role,
        )

        # Track in WebSocket (if available)
        if progress_tracker:
            await progress_tracker.start_research(
                research_id,
                str(current_user.user_id),
                research.query,
                classification.primary.value,
                research.options.depth.value,
            )

        # Trigger webhook (if available)
        if webhook_manager:
            await webhook_manager.trigger_event(
                WebhookEvent.RESEARCH_STARTED,
                {
                    "research_id": research_id,
                    "user_id": str(current_user.user_id),
                    "query": research.query,
                    "paradigm": classification.primary.value,
                },
            )

        return {
            "research_id": research_id,
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "estimated_completion": (
                datetime.utcnow() + timedelta(minutes=2)
            ).isoformat(),
            "websocket_url": f"/ws/research/{research_id}",
        }

    except Exception as e:
        logger.error("Research submission error: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Research submission failed: {str(e)}"
        )


@router.get("/status/{research_id}")
async def get_research_status(
    research_id: str, current_user=Depends(get_current_user)
):
    """Get the status of a research query"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research["user_id"] != str(current_user.id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    status_response = {
        "research_id": research_id,
        "status": research["status"],
        "paradigm": research["paradigm_classification"]["primary"],
        "started_at": research["created_at"],
        "progress": research.get("progress", {}),
        "cost_info": research.get("cost_info"),
    }

    # Add status-specific information
    if research["status"] == ResearchStatus.FAILED:
        status_response["error"] = research.get("error", "Research failed")
        status_response["can_retry"] = True
        status_response["message"] = "Research failed. You can retry."
    elif research["status"] == ResearchStatus.CANCELLED:
        status_response["cancelled_at"] = research.get("cancelled_at")
        status_response["cancelled_by"] = research.get("cancelled_by")
        status_response["can_retry"] = True
        status_response["message"] = "Research was cancelled by user."
    elif research["status"] == ResearchStatus.COMPLETED:
        status_response["message"] = "Research completed successfully."
    elif research["status"] in [
        ResearchStatus.PROCESSING, ResearchStatus.IN_PROGRESS
    ]:
        status_response["can_cancel"] = True
        status_response["message"] = (
            f"Research is {research['status']}. "
            "Please wait for completion or cancel if needed."
        )

    return status_response


@router.get("/results/{research_id}")
async def get_research_results(
    research_id: str, current_user=Depends(get_current_user)
):
    """Get completed research results"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research["user_id"] != str(current_user.id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    if research["status"] != ResearchStatus.COMPLETED:
        if research["status"] == ResearchStatus.FAILED:
            # Return detailed error information for failed research
            error_detail = research.get("error", "Research execution failed")
            return {
                "status": "failed",
                "error": error_detail,
                "research_id": research_id,
                "message": "Research failed. Please try submitting a new query.",
                "can_retry": True
            }
        elif research["status"] == ResearchStatus.CANCELLED:
            # Return cancellation information
            return {
                "status": "cancelled",
                "research_id": research_id,
                "message": "Research was cancelled by user.",
                "cancelled_at": research.get("cancelled_at"),
                "cancelled_by": research.get("cancelled_by"),
                "can_retry": True
            }
        elif research["status"] in [
            ResearchStatus.PROCESSING, ResearchStatus.IN_PROGRESS
        ]:
            # Return progress information for ongoing research
            return {
                "status": research["status"],
                "research_id": research_id,
                "message": (
                    f"Research is still {research['status']}. "
                    "Please wait for completion or cancel if needed."
                ),
                "progress": research.get("progress", {}),
                "estimated_completion": research.get("estimated_completion"),
                "can_cancel": True,
                "can_retry": False
            }
        else:
            # Handle other statuses (QUEUED, etc.)
            return {
                "status": research["status"],
                "research_id": research_id,
                "message": f"Research is {research['status']}",
                "can_retry": research["status"] != ResearchStatus.PROCESSING,
                "can_cancel": research["status"] in [ResearchStatus.QUEUED]
            }

    return research["results"]


@router.post("/cancel/{research_id}")
async def cancel_research(
    research_id: str, current_user=Depends(get_current_user)
):
    """Cancel an ongoing research query"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research["user_id"] != str(current_user.id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    # Check if research can be cancelled
    current_status = research["status"]
    if current_status in [
        ResearchStatus.COMPLETED,
        ResearchStatus.FAILED,
        ResearchStatus.CANCELLED
    ]:
        return {
            "research_id": research_id,
            "status": current_status,
            "message": (
                f"Research is already {current_status} "
                "and cannot be cancelled"
            ),
            "cancelled": False
        }

    try:
        # Update status to cancelled
        await research_store.update_field(
            research_id, "status", ResearchStatus.CANCELLED
        )
        await research_store.update_field(
            research_id, "cancelled_at", datetime.utcnow().isoformat()
        )
        await research_store.update_field(
            research_id, "cancelled_by", str(current_user.user_id)
        )

        logger.info(
            "Research %s cancelled by user %s",
            research_id, current_user.user_id
        )

        # Broadcast cancellation over WebSocket for live UIs
        if progress_tracker:
            try:
                await progress_tracker.connection_manager.broadcast_to_research(
                    research_id,
                    WSMessage(
                        type=WSEventType.RESEARCH_CANCELLED,
                        data={
                            "research_id": research_id,
                            "status": "cancelled",
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    ),
                )
            except Exception:
                pass

        return {
            "research_id": research_id,
            "status": "cancelled",
            "message": "Research has been successfully cancelled",
            "cancelled": True,
            "cancelled_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to cancel research %s: %s", research_id, e)
        raise HTTPException(status_code=500, detail="Failed to cancel research")


@router.get("/history")
async def get_research_history(
    current_user=Depends(get_current_user),
    limit: int = 10,
    offset: int = 0
):
    """Get user's research history"""
    try:
        # Get research history for user
        user_research = await research_store.get_user_research(
            str(current_user.user_id), limit + offset
        )

        # Sort by creation date (newest first)
        user_research.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        total = len(user_research)
        paginated = user_research[offset: offset + limit]

        # Format the response
        history = []
        for research in paginated:
            history_item = {
                "research_id": research["id"],
                "query": research["query"],
                "status": research["status"],
                "paradigm": research.get("paradigm_classification", {}).get(
                    "primary", "unknown"
                ),
                "created_at": research["created_at"],
                "options": research["options"],
            }

            # Include results summary if completed
            if (research["status"] == ResearchStatus.COMPLETED and
                research.get("results")):
                results = research["results"]
                content_preview = ""
                if results.get("answer", {}).get("sections"):
                    content_preview = (
                        results["answer"]["sections"][0].get("content", "")[:200]
                        + "..."
                    )
                history_item["summary"] = {
                    "answer_preview": content_preview,
                    "source_count": len(results.get("sources", [])),
                    "total_cost": results.get("cost_info", {}).get("total_cost", 0),
                }

            history.append(history_item)

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "history": history
        }
    except Exception as e:
        logger.error("Error fetching research history: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Failed to fetch research history"
        )


@router.post("/feedback/{research_id}")
async def submit_research_feedback(
    research_id: str,
    satisfaction_score: float,
    paradigm_feedback: str = None,
    current_user=Depends(get_current_user),
):
    """Submit feedback for a research query to improve the system"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (research["user_id"] != str(current_user.id) and
        current_user.role != UserRole.ADMIN):
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Store feedback in research data
        await research_store.update_field(research_id, "user_feedback", {
            "satisfaction_score": satisfaction_score,
            "paradigm_feedback": paradigm_feedback,
            "submitted_at": datetime.utcnow().isoformat()
        })

        return {
            "success": True,
            "message": "Feedback recorded successfully",
            "research_id": research_id,
            "satisfaction_score": satisfaction_score
        }
    except Exception as e:
        logger.error("Failed to record feedback: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to record feedback")


@router.get("/export/{research_id}/{fmt}")
async def export_research_result(
    research_id: str,
    fmt: str,
    current_user=Depends(get_current_user),
):
    """Export completed research results in the given format.

    Supported formats: pdf, markdown, json, csv, excel
    """
    # Fetch research
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research["user_id"] != str(current_user.id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    if research.get("status") != ResearchStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Research not completed")

    final = research.get("results") or {}

    # Construct exporter-agnostic payload
    export_payload = {
        "id": research_id,
        "query": research.get("query"),
        "paradigm": ((final.get("paradigm_analysis") or {}).get("primary") or {}).get("paradigm", "unknown"),
        "depth": (research.get("options") or {}).get("depth", "standard"),
        "confidence_score": float(((final.get("paradigm_analysis") or {}).get("primary") or {}).get("confidence", 0.0) or 0.0),
        "sources_count": len(final.get("sources") or []),
        "summary": ((final.get("answer") or {}).get("summary") or ""),
        "answer": final.get("answer") or {},
        "sources": final.get("sources") or [],
        "citations": (final.get("answer") or {}).get("citations") or [],
        "metadata": final.get("metadata") or {},
    }

    # Resolve format
    try:
        fmt_enum = ExportFormat(fmt.lower())
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    options = ExportOptions(format=fmt_enum)

    # Perform export
    export_service = ExportService()
    try:
        result = await export_service.export_research(export_payload, options)
        from fastapi import Response

        return Response(
            content=result.data,
            media_type=result.content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{result.filename}"',
                "X-Export-Format": result.format,
                "X-Export-Size": str(result.size_bytes),
            },
        )
    except Exception as e:
        logger.error("Export failed: %s", e)
        raise HTTPException(status_code=500, detail="Export failed")

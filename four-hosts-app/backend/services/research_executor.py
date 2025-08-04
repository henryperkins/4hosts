"""
Research execution service for background tasks
"""

import logging
from datetime import datetime, timedelta

from models.research import ResearchQuery
from models.base import (
    ResearchDepth,
    ResearchStatus,
    ResearchResult,
    SourceResult,
    HOST_TO_MAIN_PARADIGM
)
from services.research_store import research_store
from services.enhanced_integration import (
    enhanced_classification_engine as classification_engine,
    enhanced_answer_orchestrator as answer_orchestrator
)
from services.research_orchestrator import research_orchestrator
from services.context_engineering import context_pipeline
from services.webhook_manager import WebhookEvent

logger = logging.getLogger(__name__)


async def execute_real_research(
    research_id: str, research: ResearchQuery, user_id: str
):
    """Execute real research using the complete pipeline"""

    async def check_cancellation():
        """Check if research has been cancelled"""
        research_data = await research_store.get(research_id)
        if research_data and research_data.get("status") == ResearchStatus.CANCELLED:
            logger.info("Research %s was cancelled, stopping execution", research_id)
            return True
        return False

    try:
        # Check for cancellation before starting
        if await check_cancellation():
            return

        # Update status
        await research_store.update_field(
            research_id, "status", ResearchStatus.IN_PROGRESS
        )

        # Get classification
        research_data = await research_store.get(research_id)
        if not research_data:
            raise Exception("Research data not found")

        # Get the stored classification result from the new engine
        classification_result = await classification_engine.classify_query(
            research.query
        )

        # Check for cancellation
        if await check_cancellation():
            return

        # Process through context engineering pipeline
        context_engineered_query = await context_pipeline.process_query(
            classification_result
        )

        # Check for cancellation
        if await check_cancellation():
            return

        # Execute research based on depth option
        if research.options.depth == ResearchDepth.DEEP_RESEARCH:
            # Use deep research with o3-deep-research model
            from services.deep_research_service import DeepResearchMode

            # Map paradigms to deep research modes
            deep_mode_mapping = {
                "dolores": DeepResearchMode.PARADIGM_FOCUSED,
                "teddy": DeepResearchMode.PARADIGM_FOCUSED,
                "bernard": DeepResearchMode.ANALYTICAL,
                "maeve": DeepResearchMode.STRATEGIC,
            }

            paradigm_name = context_engineered_query.classification.primary_paradigm.value
            deep_mode = deep_mode_mapping.get(
                paradigm_name, DeepResearchMode.COMPREHENSIVE
            )

            # Get web search settings from research data
            search_context_size = research_data.get("search_context_size")
            user_location = research_data.get("user_location")

            execution_result = await research_orchestrator.execute_deep_research(
                context_engineered_query,
                enable_standard_search=True,
                deep_research_mode=deep_mode,
                search_context_size=search_context_size,
                user_location=user_location,
                research_id=research_id,
            )
        else:
            # Use standard paradigm research
            execution_result = await research_orchestrator.execute_paradigm_research(
                context_engineered_query,
                research.options.max_sources,
                research_id
            )

        # Check for cancellation
        if await check_cancellation():
            return

        # Format results
        formatted_sources = []
        search_results_for_synthesis = []

        # Support both 'filtered_results' and alias 'results'
        legacy_results = getattr(execution_result, "filtered_results", None)
        if legacy_results is None:
            legacy_results = getattr(execution_result, "results", [])

        for result in legacy_results[: research.options.max_sources]:
            formatted_sources.append(
                SourceResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    domain=result.domain,
                    credibility_score=getattr(result, "credibility_score", 0.5),
                    published_date=(
                        result.published_date.isoformat()
                        if result.published_date
                        else None
                    ),
                    source_type=result.result_type,
                )
            )

            search_results_for_synthesis.append(
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "domain": result.domain,
                    "credibility_score": getattr(result, "credibility_score", 0.5),
                    "published_date": result.published_date,
                    "result_type": result.result_type,
                }
            )

        # Check for cancellation before AI generation
        if await check_cancellation():
            return

        # Generate answer using context engineering outputs
        context_engineering = {
            "write_output": {
                "documentation_focus": context_engineered_query.write_output.documentation_focus,
                "key_themes": context_engineered_query.write_output.key_themes[:4],
                "narrative_frame": context_engineered_query.write_output.narrative_frame,
            },
            "select_output": {
                "search_queries": context_engineered_query.select_output.search_queries,
                "source_preferences": context_engineered_query.select_output.source_preferences,
                "max_sources": context_engineered_query.select_output.max_sources,
            },
            "compress_output": {
                "compression_ratio": context_engineered_query.compress_output.compression_ratio,
                "priority_elements": context_engineered_query.compress_output.priority_elements,
                "token_budget": context_engineered_query.compress_output.token_budget,
            },
            "isolate_output": {
                "isolation_strategy": context_engineered_query.isolate_output.isolation_strategy,
                "key_findings_criteria": context_engineered_query.isolate_output.key_findings_criteria,
                "output_structure": context_engineered_query.isolate_output.output_structure,
            },
        }

        # Map enum value to paradigm name
        paradigm_mapping = {
            "revolutionary": "dolores",
            "devotion": "teddy",
            "analytical": "bernard",
            "strategic": "maeve",
        }
        paradigm_name = paradigm_mapping.get(
            context_engineered_query.classification.primary_paradigm.value,
            "bernard",  # Default to bernard if not found
        )

        # Check if we have deep research content
        deep_research_content = getattr(execution_result, "deep_research_content", None)

        # Enhanced logging before answer generation
        logger.info(
            "Starting answer generation with %d results",
            len(search_results_for_synthesis)
        )

        try:
            generated_answer = await answer_orchestrator.generate_answer(
                paradigm=paradigm_name,
                query=research.query,
                search_results=search_results_for_synthesis,
                context_engineering=context_engineering,
                options={
                    "research_id": research_id,
                    "max_length": 2000,
                    "include_citations": True,
                    "deep_research_content": deep_research_content,
                },
            )

            logger.info("Answer generation completed. Type: %s", type(generated_answer))

        except Exception as e:
            logger.error("Answer generation failed: %s", str(e))
            raise

        # Format final result - handle both dict and object formats
        answer_sections = []
        citations_list = []

        # Defensive check: handle different return formats from generators
        if generated_answer is None:
            logger.error("Generated answer is None")
        elif isinstance(generated_answer, dict):
            # Dictionary format from enhanced generators
            sections_count = generated_answer.get("sections", 0)
            if isinstance(sections_count, int):
                # Create basic sections structure for compatibility
                content = generated_answer.get("content", "")
                paradigm = generated_answer.get("paradigm", "bernard")
                answer_sections.append({
                    "title": "Research Summary",
                    "paradigm": paradigm,
                    "content": content,
                    "confidence": generated_answer.get("synthesis_quality", 0.8),
                    "sources_count": len(generated_answer.get("citations", [])),
                })
            else:
                # Legacy sections format
                for section in sections_count:
                    answer_sections.append(
                        {
                            "title": getattr(section, "title", "Untitled"),
                            "paradigm": getattr(section, "paradigm", "bernard"),
                            "content": getattr(section, "content", ""),
                            "confidence": getattr(section, "confidence", 0.8),
                            "sources_count": len(getattr(section, "citations", [])),
                        }
                    )

            # Handle citations from dictionary format
            citations = generated_answer.get("citations", [])
            if isinstance(citations, list):
                for i, cite_id in enumerate(citations):
                    citations_list.append({
                        "id": cite_id,
                        "title": f"Source {i+1}",
                        "source": f"Source {i+1}",
                        "url": "",
                        "snippet": "",
                        "credibility_score": 0.8,
                        "paradigm_alignment": generated_answer.get("paradigm", "bernard"),
                    })
            elif isinstance(citations, dict):
                for cite_id, citation in citations.items():
                    citations_list.append({
                        "id": cite_id,
                        "title": getattr(citation, "source_title", f"Source {cite_id}"),
                        "source": getattr(citation, "source_title", f"Source {cite_id}"),
                        "url": getattr(citation, "source_url", ""),
                        "snippet": getattr(citation, "snippet", ""),
                        "credibility_score": getattr(citation, "credibility_score", 0.8),
                        "paradigm_alignment": generated_answer.get("paradigm", "bernard"),
                    })
        else:
            # Object format (legacy)
            if hasattr(generated_answer, "sections") and generated_answer.sections:
                for section in generated_answer.sections:
                    answer_sections.append(
                        {
                            "title": section.title,
                            "paradigm": section.paradigm,
                            "content": section.content,
                            "confidence": section.confidence,
                            "sources_count": len(section.citations),
                        }
                    )

            if hasattr(generated_answer, "citations") and generated_answer.citations:
                for cite_id, citation in generated_answer.citations.items():
                    citations_list.append(
                        {
                            "id": cite_id,
                            "title": getattr(citation, "source_title", f"Source {cite_id}"),
                            "source": getattr(citation, "source_title", f"Source {cite_id}"),
                            "url": getattr(citation, "source_url", ""),
                            "snippet": getattr(citation, "snippet", ""),
                            "credibility_score": getattr(citation, "credibility_score", 0.8),
                            "paradigm_alignment": context_engineered_query.classification.primary_paradigm.value,
                        }
                    )

        # Normalize generated answer fields to avoid attribute errors
        if generated_answer is None:
            ga_summary = ""
            ga_action_items = []
            ga_synth_quality = 0.0
            ga_gen_time = 0.0
        elif isinstance(generated_answer, dict):
            ga_summary = generated_answer.get("content", "") or generated_answer.get("summary", "")
            ga_action_items = generated_answer.get("action_items", [])
            ga_synth_quality = generated_answer.get("synthesis_quality", 0.0)
            ga_gen_time = generated_answer.get("generation_time", 0.0)
        else:
            ga_summary = getattr(generated_answer, "summary", "")
            ga_action_items = getattr(generated_answer, "action_items", [])
            ga_synth_quality = getattr(generated_answer, "synthesis_quality", 0.0)
            ga_gen_time = getattr(generated_answer, "generation_time", 0.0)

        final_result = ResearchResult(
            research_id=research_id,
            query=research.query,
            status=ResearchStatus.COMPLETED,
            paradigm_analysis={
                "primary": {
                    "paradigm": context_engineered_query.classification.primary_paradigm.value,
                    "confidence": context_engineered_query.classification.confidence,
                    "approach": context_engineered_query.write_output.narrative_frame,
                    "focus": context_engineered_query.write_output.documentation_focus,
                },
                "context_engineering": {
                    "compression_ratio": context_engineered_query.compress_output.compression_ratio,
                    "token_budget": context_engineered_query.compress_output.token_budget,
                    "isolation_strategy": context_engineered_query.isolate_output.isolation_strategy,
                    "search_queries_count": len(
                        context_engineered_query.select_output.search_queries
                    ),
                },
            },
            answer={
                "summary": ga_summary,
                "sections": answer_sections,
                "action_items": ga_action_items,
                "citations": citations_list,
            },
            sources=formatted_sources,
            metadata={
                "total_sources_analyzed": len(execution_result.raw_results),
                "high_quality_sources": len(
                    [s for s in formatted_sources if s.credibility_score > 0.7]
                ),
                "search_queries_executed": len(
                    execution_result.search_queries_executed
                ),
                "processing_time_seconds": execution_result.execution_metrics[
                    "processing_time_seconds"
                ],
                "answer_generation_time": ga_gen_time,
                "synthesis_quality": ga_synth_quality,
                "paradigms_used": [
                    context_engineered_query.classification.primary_paradigm.value
                ],
                "deep_research_enabled": execution_result.execution_metrics.get(
                    "deep_research_enabled", False
                ),
                "research_depth": research.options.depth.value,
            },
            cost_info=execution_result.cost_breakdown,
        )

        # Store results
        await research_store.update_field(
            research_id, "status", ResearchStatus.COMPLETED
        )
        await research_store.update_field(research_id, "results", final_result.dict())
        await research_store.update_field(
            research_id, "cost_info", execution_result.cost_breakdown
        )

        logger.info("✓ Research completed for %s", research_id)

    except Exception as e:
        logger.error("Research execution failed for %s: %s", research_id, str(e))
        await research_store.update_field(research_id, "status", ResearchStatus.FAILED)
        await research_store.update_field(research_id, "error", str(e))

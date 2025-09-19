# flake8: noqa
"""
Integration module for enhanced answer generators, self-healing system, and ML pipeline
Connects new components to the existing Four Hosts application
"""

import logging
from typing import Dict, Optional, Any, List, Union, cast
from datetime import datetime

from .answer_generator import AnswerGenerationOrchestrator
from .self_healing_system import self_healing_system
from .ml_pipeline import ml_pipeline
from .classification_engine import ClassificationEngine, HostParadigm, ClassificationResult
from models.paradigms import normalize_to_enum, normalize_to_internal_code
from models.synthesis_models import SynthesisContext
from .result_adapter import ResultListAdapter, adapt_results
from contracts import (
    GeneratedAnswer as ContractAnswer,
    ResearchStatus as ContractStatus,
    Source as ContractSource,
)
import asyncio

logger = logging.getLogger(__name__)


def _to_contract_source(obj: Any) -> ContractSource:
    if isinstance(obj, ContractSource):
        return obj
    if isinstance(obj, dict):
        return ContractSource.model_validate(obj)
    data = {
        "url": getattr(obj, "url", "http://invalid.local"),
        "title": getattr(obj, "title", ""),
        "snippet": getattr(obj, "snippet", None),
        "score": getattr(obj, "credibility_score", None),
        "metadata": getattr(obj, "metadata", {}),
    }
    return ContractSource.model_validate(data)


class EnhancedAnswerGenerationOrchestrator(AnswerGenerationOrchestrator):
    """Enhanced orchestrator that integrates new components.

    This subclass extends the original ``AnswerGenerationOrchestrator`` while **preserving
    backwards-compatibility** with the original public interface.  There are currently **two**
    calling conventions in the code base:

    1. **Legacy signature** – used by the FastAPI routes in *main.py* and elsewhere.  It matches
       the parent implementation::

           await answer_orchestrator.generate_answer(
               paradigm="bernard",
               query="…",
               search_results=[…],
               context_engineering={…},
               options={…},
           )

    2. **New signature** – used by the new tests in *test_enhanced_features.py* where the first
       argument is a fully populated ``SynthesisContext`` followed by the paradigms::

           await answer_orchestrator.generate_answer(context, HostParadigm.BERNARD)

    Previously we overwrote ``generate_answer`` with the *new* signature which broke the code that
    still depends on the legacy keyword arguments (FastAPI runtime error
    "got an unexpected keyword argument 'paradigm'").

    The implementation below accepts **either** calling style and internally routes the request to
    ``_generate_from_context`` which contains the actual enhanced generation logic.  This keeps the
    public interface stable without duplicating business logic.
    """

    def __init__(self):
        super().__init__()

        # Initialize generators dictionary using parent's method for consistency
        # Accept both enum and string keys for backward compatibility with legacy callers
        self.generators: Dict[Union[HostParadigm, str], Any] = {}

        # Populate generators using parent's _make_generator method to avoid duplication
        for paradigm in HostParadigm:
            # Create generators using INTERNAL code names (dolores/bernard/maeve/teddy)
            code = normalize_to_internal_code(paradigm)
            gen = self._make_generator(code)
            # Store under multiple keys for compatibility (enum, internal code, enum.value)
            self.generators[paradigm] = gen
            self.generators[code] = gen
            self.generators[paradigm.value] = gen

        # Enable self-healing and ML features
        self.self_healing_enabled = True
        self.ml_enhanced = True

        logger.info("Enhanced Answer Generation Orchestrator initialized with advanced features")

    # ------------------------------------------------------------------
    # Public entry point – supports both the legacy and the new signature
    # ------------------------------------------------------------------
    async def generate_answer(self, *args, **kwargs) -> Any:  # type: ignore[override]
        """Generate an answer using either the **legacy** or **new** call signature.

        The method inspects *args* / *kwargs* to determine which signature was used and then
        constructs the appropriate ``SynthesisContext`` before delegating to
        ``_generate_from_context``.
        """

        # Helper local function available to both call-paths -----------------
        def _resolve_paradigm(value: str | HostParadigm | None) -> HostParadigm | None:
            """Normalize a value to HostParadigm, or None if value is None."""
            return normalize_to_enum(value)

        # ------------------------------------------------------------------
        # Case 1 – NEW SIGNATURE: first positional argument is a SynthesisContext instance
        # ------------------------------------------------------------------
        if args and isinstance(args[0], SynthesisContext):
            context: SynthesisContext = args[0]

            # Primary/secondary paradigm can come either positionally or via kwargs
            primary_paradigm: HostParadigm | None = None
            secondary_paradigm: Optional[HostParadigm] = None

            if len(args) > 1:
                primary_paradigm = args[1]
            if len(args) > 2:
                secondary_paradigm = args[2]

            # Fallback to kwargs if not provided positionally
            primary_paradigm = primary_paradigm or kwargs.get("primary_paradigm")
            secondary_paradigm = secondary_paradigm or kwargs.get("secondary_paradigm")

            # If context already stores a paradigm string we try to map it to enum when needed
            if primary_paradigm is None and hasattr(context, "paradigm") and context.paradigm:
                try:
                    primary_paradigm = _resolve_paradigm(context.paradigm)
                except Exception:
                    primary_paradigm = None

            if primary_paradigm is None:
                raise ValueError("primary_paradigm must be supplied when using the context signature")

            # Early zero-source guard (PR1): return structured failure, not None
            try:
                if not context.search_results:
                    return ContractAnswer(
                        status=ContractStatus.FAILED_NO_SOURCES,
                        content_md="",
                        citations=[],
                        diagnostics={"reason": "no_sources"},
                    )
            except Exception:
                # Defensive – continue to normal path
                pass

            return await self._generate_from_context_with_fallback(context, primary_paradigm, secondary_paradigm)

        # ------------------------------------------------------------------
        # Case 2 – LEGACY SIGNATURE: parameters passed individually (possibly as kwargs)
        # ------------------------------------------------------------------
        paradigm: str | HostParadigm | None = None
        query: Optional[str] = None
        search_results: Optional[list] = None
        context_engineering: Optional[dict] = None
        options: Optional[dict] = None

        # Extract either positionally or via kwargs
        if len(args) >= 4:
            paradigm, query, search_results, context_engineering = args[:4]
            if len(args) >= 5:
                options = args[4]
        else:
            paradigm = kwargs.get("paradigm")
            query = kwargs.get("query")
            search_results = kwargs.get("search_results")
            context_engineering = kwargs.get("context_engineering")
            options = kwargs.get("options")

        if paradigm is None or query is None or search_results is None or context_engineering is None:
            raise TypeError("Invalid arguments – expected legacy signature parameters to be provided")

        # Resolve paradigms using the helper defined at the top of this method
        primary_paradigm_enum = _resolve_paradigm(paradigm)

        secondary_paradigm = kwargs.get("secondary_paradigm")
        secondary_paradigm = _resolve_paradigm(secondary_paradigm)

        # Type narrowing for Pylance – ensure non-None
        if primary_paradigm_enum is None:
            raise TypeError("Invalid paradigm – could not resolve to HostParadigm")

        # Build ``SynthesisContext`` – replicate logic from the parent implementation
        options = options or {}

        from core.config import SYNTHESIS_MAX_LENGTH_DEFAULT
        context = SynthesisContext(
            query=query,
            paradigm=primary_paradigm_enum.value if isinstance(primary_paradigm_enum, HostParadigm) else str(primary_paradigm_enum),
            search_results=search_results,
            context_engineering=context_engineering,
            max_length=options.get("max_length", SYNTHESIS_MAX_LENGTH_DEFAULT),
            include_citations=options.get("include_citations", True),
            tone=options.get("tone", "professional"),
        )

        # Attach metadata if provided
        context.metadata = {"research_id": options.get("research_id", "unknown")}

        # Deep-research content may be attached via the options dict
        if dr_content := options.get("deep_research_content"):
            context.deep_research_content = dr_content  # type: ignore[attr-defined]

        # Early zero-source guard (PR1) for legacy call path
        if not context.search_results:
            return ContractAnswer(
                status=ContractStatus.FAILED_NO_SOURCES,
                content_md="",
                citations=[],
                diagnostics={"reason": "no_sources"},
            )

        return await self._generate_from_context_with_fallback(context, cast(HostParadigm, primary_paradigm_enum), secondary_paradigm)

    # ------------------------------------------------------------------
    # Internal implementation – **moved unchanged** from the previous override
    # ------------------------------------------------------------------

    async def _generate_from_context_with_fallback(
        self,
        context: SynthesisContext,
        primary_paradigm: HostParadigm,
        secondary_paradigm: Optional[HostParadigm] = None,
    ) -> Any:
        """Enhanced generation with partial success policy and result adapter.

        This wrapper normalizes incoming search results and then delegates to
        the core implementation. It exists to keep the public entrypoint
        stable while allowing pre/post hooks (e.g., adaptation, fallbacks).
        """
        start_time = datetime.now()
        query_id = context.metadata.get("research_id", f"query_{start_time.timestamp()}")

        # Normalize search results using ResultAdapter
        try:
            adapted_results = self._adapt_search_results(context.search_results)
            context.search_results = adapted_results
        except Exception as e:
            logger.warning(f"Failed to adapt search results: {e}")
            # Continue with original results

        # Delegate to the core generator; it handles exceptions and structured failures
        return await self._generate_from_context(context, primary_paradigm, secondary_paradigm)

    def _adapt_search_results(self, search_results: Any) -> List[Dict[str, Any]]:
        """Safely adapt search results to consistent format"""
        if not search_results:
            return []

        try:
            # Use ResultAdapter to handle both dict and object formats
            adapter = adapt_results(search_results)

            if isinstance(adapter, ResultListAdapter):
                # Get valid results and convert to dict format
                valid_results = adapter.get_valid_results()
                return [result.to_dict() for result in valid_results]
            else:
                # Single result
                if adapter.has_required_fields():
                    return [adapter.to_dict()]
                else:
                    return []
        except Exception as e:
            logger.error(f"Error adapting search results: {e}")
            # Fallback: try to convert to list of dicts manually
            if isinstance(search_results, list):
                adapted = []
                for result in search_results:
                    if isinstance(result, dict):
                        adapted.append(result)
                    elif hasattr(result, 'to_dict'):
                        adapted.append(result.to_dict())
                    elif hasattr(result, '__dict__'):
                        adapted.append(result.__dict__.copy())
                return adapted
            return []

    async def _generate_from_context(
        self,
        context: SynthesisContext,
        primary_paradigm: HostParadigm,
        secondary_paradigm: Optional[HostParadigm] = None,
    ) -> Any:
        """Actual Enhanced generation logic with partial success policy."""
        start_time = datetime.now()
        query_id = context.metadata.get("research_id", f"query_{start_time.timestamp()}")
        errors = []
        failed_paths = []

        try:
            # Check if self-healing recommends a different paradigm
            if self.self_healing_enabled:
                recommended_paradigm = self_healing_system.get_paradigm_recommendation(
                    context.query, primary_paradigm
                )

                if recommended_paradigm and recommended_paradigm != primary_paradigm:
                    logger.info(
                        f"Self-healing system recommends switching from {primary_paradigm} "
                        f"to {recommended_paradigm} for query: {context.query[:50]}..."
                    )

                    # Record the switch decision
                    await self_healing_system.record_query_performance(
                        query_id=query_id,
                        query_text=context.query,
                        paradigm=primary_paradigm,
                        error="paradigm_switch_recommended",
                    )

                    # Use recommended paradigm
                    primary_paradigm = recommended_paradigm

            # Generate answer using appropriate generator (support both enum and string keys)
            generator = self.generators.get(primary_paradigm) or self.generators.get(primary_paradigm.value)
            if generator is None:
                raise ValueError(f"No generator registered for paradigm {primary_paradigm}")
            primary_answer = await generator.generate_answer(context)

            # Enforce non-null return (PR1)
            if primary_answer is None:
                logger.warning("Generator returned None; substituting FAILED_NO_SOURCES answer")

                minimal_sources = [_to_contract_source(r) for r in (context.search_results or [])][:3]

                primary_answer = ContractAnswer(
                    status=ContractStatus.FAILED_NO_SOURCES,
                    content_md="",
                    citations=minimal_sources,
                    diagnostics={"reason": "generator_returned_none"},
                )

            # Record performance metrics
            response_time = (datetime.now() - start_time).total_seconds()
            # Only pass an answer object that exposes the metrics we need; otherwise pass None
            has_metrics = hasattr(primary_answer, "confidence_score") or hasattr(primary_answer, "synthesis_quality")
            perf_answer = primary_answer if has_metrics else None
            await self_healing_system.record_query_performance(
                query_id=query_id,
                query_text=context.query,
                paradigm=primary_paradigm,
                answer=cast(Any, perf_answer),
                response_time=response_time,
            )

            # Record training example for ML pipeline
            cls_res = getattr(context, "classification_result", None)
            if self.ml_enhanced and cls_res is not None:
                synth_quality = getattr(primary_answer, "synthesis_quality", getattr(primary_answer, "quality_score", None))
                await ml_pipeline.record_training_example(
                    query_id=query_id,
                    query_text=context.query,
                    features=cls_res.features,
                    predicted_paradigm=primary_paradigm,
                    synthesis_quality=synth_quality,
                )

            # Handle secondary paradigm if provided (legacy mesh integration removed)
            if secondary_paradigm:
                secondary_generator = self.generators.get(secondary_paradigm) or self.generators.get(secondary_paradigm.value)
                if secondary_generator is None:
                    secondary_generator = generator  # Fallback to primary to avoid crash
                _ = await secondary_generator.generate_answer(context)
                # Record minimal integration metadata without mesh network dependency (if supported)
                meta = getattr(primary_answer, "metadata", None)
                if isinstance(meta, dict):
                    meta["integration"] = {
                        "secondary_paradigm": secondary_paradigm.value,
                        "conflicts": 0,
                        "synergies": 0,
                    }
                    try:
                        setattr(primary_answer, "metadata", meta)
                    except Exception:
                        pass

            return primary_answer

        except Exception as e:
            error_msg = str(e)
            errors.append(error_msg)
            failed_paths.append(f"primary_generation_{primary_paradigm.value}")
            logger.error(f"Error in enhanced answer generation: {e}")

            # Record failure
            await self_healing_system.record_query_performance(
                query_id=query_id,
                query_text=context.query,
                paradigm=primary_paradigm,
                error=error_msg,
                response_time=(datetime.now() - start_time).total_seconds(),
            )

            # Structured failure (PR1): return contracts.GeneratedAnswer with explicit status
            status = (
                ContractStatus.TIMEOUT
                if isinstance(e, (asyncio.TimeoutError,))
                else ContractStatus.TOOL_ERROR
            )
            # Include a few citations when sources exist
            citations = []
            try:
                adapted = context.search_results or []
                for item in (adapted[:3] if isinstance(adapted, list) else []):
                    citations.append(_to_contract_source(item))
            except Exception:
                citations = []

            return ContractAnswer(
                status=status,
                content_md="",
                citations=citations,
                diagnostics={
                    "reason": "generation_exception",
                    "error": error_msg,
                    "failed_paths": failed_paths,
                },
            )

    async def _create_minimal_answer_with_fallback(
        self,
        context: SynthesisContext,
        primary_paradigm: HostParadigm,
        secondary_paradigm: Optional[HostParadigm],
        errors: List[str],
        failed_paths: List[str]
    ) -> Any:
        """Create minimal answer when primary generation fails"""

        # Try fallback paradigm if available
        fallback_answer = None
        fallback_paradigm_used: Optional[HostParadigm] = None
        if self.self_healing_enabled:
            fallback_paradigm = self._get_fallback_paradigm(primary_paradigm)
            if fallback_paradigm:
                logger.info(f"Attempting fallback to {fallback_paradigm} paradigm")
                try:
                    fallback_generator = self.generators.get(fallback_paradigm) or self.generators.get(fallback_paradigm.value)
                    if fallback_generator:
                        fallback_answer = await fallback_generator.generate_answer(context)
                        fallback_paradigm_used = fallback_paradigm
                        logger.info(f"Fallback to {fallback_paradigm} succeeded")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    errors.append(f"fallback_{fallback_paradigm.value}: {str(fallback_error)}")
                    failed_paths.append(f"fallback_generation_{fallback_paradigm.value}")

        # If fallback succeeded, return it with enhanced metadata
        if fallback_answer:
            if hasattr(fallback_answer, "metadata") and isinstance(getattr(fallback_answer, "metadata"), dict):
                meta = getattr(fallback_answer, "metadata")
            else:
                meta = {}
                try:
                    setattr(fallback_answer, "metadata", meta)
                except Exception:
                    meta = {}
            meta.update({
                "partial_success": True,
                "original_paradigm": primary_paradigm.value,
                "fallback_paradigm": fallback_paradigm_used.value if fallback_paradigm_used else "",
                "errors": errors,
                "failed_paths": failed_paths
            })
            return fallback_answer

        # Last resort: create minimal answer with raw search results
        try:
            # Use ResultAdapter to safely extract information
            adapter = adapt_results(context.search_results)

            if isinstance(adapter, ResultListAdapter):
                valid_results = adapter.get_valid_results()[:5]  # Limit to top 5
                links = [result.url for result in valid_results]
                snippets = [result.snippet for result in valid_results if result.snippet]
            else:
                links = [adapter.url] if adapter.has_required_fields() else []
                snippets = [adapter.snippet] if adapter.snippet else []

            minimal_content = f"""I encountered issues generating a complete answer for this query. Here's what I found:

{chr(10).join(f"• {snippet[:200]}..." for snippet in snippets[:3])}

For more information, please check these sources:
{chr(10).join(f"- {link}" for link in links[:3])}"""

            # Build minimal citations as contract Sources (best-effort)
            citations = []
            for link in links[:3]:
                try:
                    citations.append(_to_contract_source({"url": link, "title": link}))
                except Exception:
                    continue

            return ContractAnswer(
                status=ContractStatus.TOOL_ERROR,
                content_md=minimal_content,
                citations=citations,
                diagnostics={
                    "reason": "minimal_fallback",
                    "errors": errors,
                    "failed_paths": failed_paths,
                    "fallback_used": "minimal_answer",
                },
            )

        except Exception as minimal_error:
            logger.error(f"Even minimal answer generation failed: {minimal_error}")
            errors.append(f"minimal_answer: {str(minimal_error)}")
            failed_paths.append("minimal_answer_generation")

            # Absolute last resort: raise with comprehensive error info
            raise RuntimeError(f"Complete answer generation failure. Errors: {'; '.join(errors)}. Failed paths: {', '.join(failed_paths)}")

    def _get_fallback_paradigm(self, failed_paradigm: HostParadigm) -> Optional[HostParadigm]:
        """Get fallback paradigm based on failure"""
        fallback_map = {
            HostParadigm.DOLORES: HostParadigm.TEDDY,  # Revolutionary -> Supportive
            HostParadigm.TEDDY: HostParadigm.BERNARD,  # Supportive -> Analytical
            HostParadigm.BERNARD: HostParadigm.MAEVE,  # Analytical -> Strategic
            HostParadigm.MAEVE: HostParadigm.BERNARD,  # Strategic -> Analytical
        }
        return fallback_map.get(failed_paradigm)


class EnhancedClassificationEngine(ClassificationEngine):
    """Enhanced classification engine that uses ML pipeline"""

    def __init__(self):
        super().__init__()
        self.ml_enhanced = True
        logger.info("Enhanced Classification Engine initialized with ML support")

    async def classify_query(self, query: str, research_id: Optional[str] = None) -> ClassificationResult:
        """Enhanced classification with ML model; preserves base signature."""
        # First get base classification (pass through research_id for progress)
        result = await super().classify_query(query, research_id)

        # If ML is available, also get ML prediction
        if self.ml_enhanced:
            try:
                ml_paradigm, ml_confidence = await ml_pipeline.predict_paradigm(
                    query, result.features
                )

                # If ML confidence is high and differs from rule-based, consider it
                if ml_confidence > 0.8 and ml_paradigm != result.primary_paradigm:
                    logger.info(
                        f"ML model suggests {ml_paradigm} (conf: {ml_confidence:.2f}) "
                        f"vs rule-based {result.primary_paradigm}"
                    )

                    # Blend the results
                    if ml_confidence > result.confidence:
                        # ML is more confident, adjust primary paradigm
                        result.secondary_paradigm = result.primary_paradigm
                        result.primary_paradigm = ml_paradigm
                        result.confidence = (result.confidence + ml_confidence) / 2
                        result.reasoning.setdefault(ml_paradigm, []).append(
                            f"ML model prediction with {ml_confidence:.2f} confidence"
                        )

            except Exception as e:
                logger.error(f"Error in ML classification enhancement: {e}")

        return result


async def record_user_feedback(
    query_id: str,
    satisfaction_score: float,
    paradigm_feedback: Optional[str] = None
) -> None:
    """Record user feedback for continuous improvement"""
    # Record in self-healing system
    await self_healing_system.record_user_feedback(query_id, satisfaction_score)

    # If user suggests a different paradigm, record for ML training
    if paradigm_feedback:
        try:
            suggested_paradigm = HostParadigm(paradigm_feedback)
            # This will be used in next training cycle
            logger.info(
                f"User suggested {suggested_paradigm} paradigm for query {query_id}"
            )
        except ValueError:
            logger.warning(f"Invalid paradigm feedback: {paradigm_feedback}")


def get_system_health_report() -> Dict[str, Any]:
    """Get comprehensive system health report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "components": {
            "self_healing": self_healing_system.get_performance_report(),
            "ml_pipeline": ml_pipeline.get_model_info(),
            "training_stats": ml_pipeline.get_training_stats(),
        },
        "recommendations": [],
    }

    # Add system-wide recommendations
    sh_report = report["components"]["self_healing"]
    ml_info = report["components"]["ml_pipeline"]

    # Check if retraining is needed
    if ml_info.get("training_examples", 0) > 1000:
        report["recommendations"].append({
            "component": "ml_pipeline",
            "action": "Model retraining recommended",
            "reason": f"{ml_info['training_examples']} new examples available",
        })

    # Check paradigm performance
    for paradigm_report in sh_report.get("paradigm_metrics", {}).values():
        if paradigm_report.get("performance_score", 1.0) < 0.6:
            report["recommendations"].append({
                "component": "self_healing",
                "action": f"Investigate {paradigm_report} paradigm performance",
                "reason": "Performance score below threshold",
            })

    return report


# Create enhanced instances for use
enhanced_answer_orchestrator = EnhancedAnswerGenerationOrchestrator()
enhanced_classification_engine = EnhancedClassificationEngine()


# Admin API endpoints for monitoring and control
async def force_paradigm_switch(query_id: str, new_paradigm: str, reason: str) -> Dict[str, str]:
    """Admin endpoint to force paradigm switch"""
    try:
        paradigm = HostParadigm(new_paradigm)
        await self_healing_system.force_paradigm_switch(query_id, paradigm, reason)
        return {"status": "success", "message": f"Paradigm switched to {paradigm.value}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def trigger_model_retraining() -> Dict[str, Any]:
    """Admin endpoint to trigger ML model retraining"""
    if len(ml_pipeline.training_examples) < ml_pipeline.min_training_samples:
        return {
            "status": "error",
            "message": f"Insufficient training examples. Need {ml_pipeline.min_training_samples}, "
                      f"have {len(ml_pipeline.training_examples)}",
        }

    # Trigger retraining
    await ml_pipeline._retrain_model()

    return {
        "status": "success",
        "message": "Model retraining initiated",
        "current_version": ml_pipeline.current_model_version,
    }


def get_paradigm_performance_metrics() -> Dict[str, Any]:
    """Get detailed paradigm performance metrics"""
    metrics = {}

    for paradigm, perf in self_healing_system.performance_metrics.items():
        if perf.total_queries > 0:
            metrics[paradigm.value] = {
                "total_queries": perf.total_queries,
                "success_rate": perf.successful_queries / perf.total_queries,
                "avg_confidence": perf.avg_confidence_score,
                "avg_synthesis_quality": perf.avg_synthesis_quality,
                "avg_user_satisfaction": perf.avg_user_satisfaction,
                "avg_response_time": perf.avg_response_time,
                "recent_trend": "improving" if len(perf.recent_scores) > 10 and
                               sum(s["confidence"] for s in list(perf.recent_scores)[-5:]) / 5 >
                               sum(s["confidence"] for s in list(perf.recent_scores)[-10:-5]) / 5
                               else "stable",
            }

    return metrics

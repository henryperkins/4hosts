"""
Integration module for enhanced answer generators, self-healing system, and ML pipeline
Connects new components to the existing Four Hosts application
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

from .answer_generator_continued import AnswerGenerationOrchestrator
from .answer_generator_enhanced import EnhancedBernardAnswerGenerator, EnhancedMaeveAnswerGenerator
from .self_healing_system import self_healing_system
from .ml_pipeline import ml_pipeline
from .classification_engine import ClassificationEngine, HostParadigm
from .answer_generator import SynthesisContext

logger = logging.getLogger(__name__)


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

        # Replace basic generators with enhanced versions
        self.generators[HostParadigm.BERNARD] = EnhancedBernardAnswerGenerator()
        self.generators[HostParadigm.MAEVE] = EnhancedMaeveAnswerGenerator()

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
            """Translate various string representations (e.g. "bernard", "analytical") to
            ``HostParadigm`` enum members.  Returns *None* if *value* is *None*."""
            if value is None:
                return None
            if isinstance(value, HostParadigm):
                return value
            mapping = {
                "dolores": HostParadigm.DOLORES,
                "revolutionary": HostParadigm.DOLORES,
                "teddy": HostParadigm.TEDDY,
                "devotion": HostParadigm.TEDDY,
                "bernard": HostParadigm.BERNARD,
                "analytical": HostParadigm.BERNARD,
                "maeve": HostParadigm.MAEVE,
                "strategic": HostParadigm.MAEVE,
            }
            key = str(value).lower()
            if key not in mapping:
                raise ValueError(f"Unknown paradigm: {value}")
            return mapping[key]

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

            return await self._generate_from_context(context, primary_paradigm, secondary_paradigm)

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

        # Build ``SynthesisContext`` – replicate logic from the parent implementation
        options = options or {}

        context = SynthesisContext(
            query=query,
            paradigm=primary_paradigm_enum.value if isinstance(primary_paradigm_enum, HostParadigm) else str(primary_paradigm_enum),
            search_results=search_results,
            context_engineering=context_engineering,
            max_length=options.get("max_length", 2000),
            include_citations=options.get("include_citations", True),
            tone=options.get("tone", "professional"),
        )

        # Attach metadata if provided
        context.metadata = {"research_id": options.get("research_id", "unknown")}

        # Deep-research content may be attached via the options dict
        if dr_content := options.get("deep_research_content"):
            context.deep_research_content = dr_content  # type: ignore[attr-defined]

        return await self._generate_from_context(context, primary_paradigm_enum, secondary_paradigm)

    # ------------------------------------------------------------------
    # Internal implementation – **moved unchanged** from the previous override
    # ------------------------------------------------------------------

    async def _generate_from_context(
        self,
        context: SynthesisContext,
        primary_paradigm: HostParadigm,
        secondary_paradigm: Optional[HostParadigm] = None,
    ) -> Any:
        """Actual enhanced generation logic (extracted from the old implementation)."""
        start_time = datetime.now()
        query_id = context.metadata.get("research_id", f"query_{start_time.timestamp()}")
        
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
            
            # Record performance metrics
            response_time = (datetime.now() - start_time).total_seconds()
            await self_healing_system.record_query_performance(
                query_id=query_id,
                query_text=context.query,
                paradigm=primary_paradigm,
                answer=primary_answer,
                response_time=response_time,
            )
            
            # Record training example for ML pipeline
            if self.ml_enhanced and hasattr(context, "classification_result"):
                await ml_pipeline.record_training_example(
                    query_id=query_id,
                    query_text=context.query,
                    features=context.classification_result.features,
                    predicted_paradigm=primary_paradigm,
                    synthesis_quality=primary_answer.synthesis_quality,
                )
            
            # Handle secondary paradigm if provided
            if secondary_paradigm:
                secondary_generator = self.generators.get(secondary_paradigm) or self.generators.get(secondary_paradigm.value)
                if secondary_generator is None:
                    secondary_generator = generator  # Fallback to primary to avoid crash
                secondary_answer = await secondary_generator.generate_answer(context)
                
                # Integrate using mesh network
                integrated = await self.mesh_network.integrate_paradigm_results(
                    primary_answer, secondary_answer
                )
                
                # Update primary answer with integrated content
                primary_answer.secondary_perspective = integrated.secondary_perspective
                primary_answer.metadata["integration"] = {
                    "secondary_paradigm": secondary_paradigm.value,
                    "conflicts": len(integrated.conflicts_identified),
                    "synergies": len(integrated.synergies),
                }
            
            return primary_answer
            
        except Exception as e:
            logger.error(f"Error in enhanced answer generation: {e}")
            
            # Record failure
            await self_healing_system.record_query_performance(
                query_id=query_id,
                query_text=context.query,
                paradigm=primary_paradigm,
                error=str(e),
                response_time=(datetime.now() - start_time).total_seconds(),
            )
            
            # Try fallback paradigm if available
            if self.self_healing_enabled:
                fallback_paradigm = self._get_fallback_paradigm(primary_paradigm)
                if fallback_paradigm:
                    logger.info(f"Attempting fallback to {fallback_paradigm} paradigm")
                    try:
                        fallback_generator = self.generators.get(fallback_paradigm) or self.generators.get(fallback_paradigm.value)
                        if fallback_generator:
                            return await fallback_generator.generate_answer(context)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
            
            raise

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

    async def classify_query(self, query: str, use_llm: bool = True) -> Any:
        """Enhanced classification with ML model"""
        # First get base classification
        result = await super().classify_query(query)
        
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
                        result.reasoning[ml_paradigm.value].append(
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
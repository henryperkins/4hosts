"""
Context Engineering Bridge
Connects existing context engineering pipeline with V2 services
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime

from services.context_engineering import context_pipeline
from services.context_engineering_v2 import context_pipeline_v2
from models.context_models import (
    ClassificationResultSchema, ContextEngineeredQuerySchema,
    QueryFeaturesSchema, HostParadigm
)

logger = logging.getLogger(__name__)


class ContextEngineeringBridge:
    """Bridge between old and new context engineering implementations"""
    
    def __init__(self, use_v2: bool = True):
        self.use_v2 = use_v2
        self.v1_pipeline = context_pipeline
        self.v2_pipeline = context_pipeline_v2
    
    async def process_query(
        self,
        classification_result,
        include_debug: bool = False
    ) -> ContextEngineeredQuerySchema:
        """Process query through appropriate pipeline"""
        
        if self.use_v2:
            # Convert old classification to V2 schema if needed
            if not isinstance(classification_result, ClassificationResultSchema):
                classification_schema = self._convert_to_v2_classification(classification_result)
            else:
                classification_schema = classification_result
            
            # Use V2 pipeline
            return await self.v2_pipeline.process_query(
                classification_schema,
                include_debug=include_debug
            )
        else:
            # Use V1 pipeline and convert result
            v1_result = await self.v1_pipeline.process_query(classification_result)
            return self._convert_v1_to_v2_result(v1_result, classification_result)
    
    def _convert_to_v2_classification(self, old_classification) -> ClassificationResultSchema:
        """Convert old classification format to V2 schema"""
        
        # Handle QueryFeatures conversion
        if hasattr(old_classification, 'features'):
            features = QueryFeaturesSchema(
                text=old_classification.features.text,
                tokens=old_classification.features.tokens,
                entities=old_classification.features.entities,
                intent_signals=old_classification.features.intent_signals,
                domain=old_classification.features.domain,
                urgency_score=old_classification.features.urgency_score,
                complexity_score=old_classification.features.complexity_score,
                emotional_valence=old_classification.features.emotional_valence
            )
        else:
            # Create minimal features
            features = QueryFeaturesSchema(
                text=old_classification.query,
                tokens=[],
                entities=[],
                intent_signals=[],
                domain=None,
                urgency_score=0.5,
                complexity_score=0.5,
                emotional_valence=0.0
            )
        
        # Convert distribution and reasoning
        distribution = {}
        reasoning = {}
        
        if hasattr(old_classification, 'distribution'):
            for paradigm, score in old_classification.distribution.items():
                if isinstance(paradigm, str):
                    try:
                        paradigm_enum = HostParadigm(paradigm)
                        distribution[paradigm_enum] = score
                    except ValueError:
                        logger.warning(f"Unknown paradigm: {paradigm}")
                else:
                    distribution[paradigm] = score
        
        if hasattr(old_classification, 'reasoning'):
            for paradigm, reasons in old_classification.reasoning.items():
                if isinstance(paradigm, str):
                    try:
                        paradigm_enum = HostParadigm(paradigm)
                        reasoning[paradigm_enum] = reasons if isinstance(reasons, list) else [reasons]
                    except ValueError:
                        logger.warning(f"Unknown paradigm in reasoning: {paradigm}")
                else:
                    reasoning[paradigm] = reasons if isinstance(reasons, list) else [reasons]
        
        return ClassificationResultSchema(
            query=old_classification.query,
            primary_paradigm=HostParadigm(old_classification.primary_paradigm.value),
            secondary_paradigm=HostParadigm(old_classification.secondary_paradigm.value) if old_classification.secondary_paradigm else None,
            distribution=distribution,
            confidence=old_classification.confidence,
            features=features,
            reasoning=reasoning
        )
    
    def _convert_v1_to_v2_result(self, v1_result, classification) -> ContextEngineeredQuerySchema:
        """Convert V1 context engineering result to V2 schema"""
        
        # Extract layer outputs
        write_output = {}
        select_output = {}
        compress_output = {}
        isolate_output = {}
        
        if hasattr(v1_result, 'write_output'):
            write_output = asdict(v1_result.write_output) if hasattr(v1_result.write_output, '__dict__') else {}
        
        if hasattr(v1_result, 'select_output'):
            select_output = asdict(v1_result.select_output) if hasattr(v1_result.select_output, '__dict__') else {}
            
        if hasattr(v1_result, 'compress_output'):
            compress_output = asdict(v1_result.compress_output) if hasattr(v1_result.compress_output, '__dict__') else {}
            
        if hasattr(v1_result, 'isolate_output'):
            isolate_output = asdict(v1_result.isolate_output) if hasattr(v1_result.isolate_output, '__dict__') else {}
        
        # Extract queries
        refined_queries = []
        if hasattr(v1_result, 'select_output') and hasattr(v1_result.select_output, 'search_queries'):
            for query_data in v1_result.select_output.search_queries:
                if isinstance(query_data, dict):
                    refined_queries.append(query_data.get('query', ''))
                else:
                    refined_queries.append(str(query_data))
        
        # Create V2 schema
        classification_schema = self._convert_to_v2_classification(classification)
        
        return ContextEngineeredQuerySchema(
            original_query=v1_result.original_query,
            classification=classification_schema,
            write_layer_output=write_output,
            select_layer_output=select_output,
            compress_layer_output=compress_output,
            isolate_layer_output=isolate_output,
            refined_queries=refined_queries[:10],  # Limit to 10 queries
            search_strategy={
                "paradigm": classification.primary_paradigm.value,
                "mode": "v1_compatibility"
            },
            context_metadata={
                "pipeline_version": "1.0",
                "conversion_timestamp": datetime.utcnow().isoformat()
            }
        )


# Global bridge instance
context_engineering_bridge = ContextEngineeringBridge()


# Compatibility function
async def process_query_with_context_engineering(
    classification_result,
    use_v2: bool = True,
    include_debug: bool = False
) -> ContextEngineeredQuerySchema:
    """Process query through context engineering with V2 compatibility"""
    bridge = ContextEngineeringBridge(use_v2=use_v2)
    return await bridge.process_query(classification_result, include_debug=include_debug)
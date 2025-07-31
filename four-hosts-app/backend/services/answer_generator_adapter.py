"""
Answer Generator Adapter
Bridges V1 answer generators with V2 context and data models
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from services.answer_generator import answer_generator, AnswerGenerationRequest
from services.answer_generator_v2 import answer_generator_v2
from services.answer_generator_v2_enhanced import answer_generator_v2_enhanced
from models.context_models import (
    ClassificationResultSchema, ContextEngineeredQuerySchema,
    SearchResultSchema, UserContextSchema
)
from services.search_apis import SearchResult

logger = logging.getLogger(__name__)


class AnswerGeneratorAdapter:
    """Adapter to use both V1 and V2 answer generators"""
    
    def __init__(self, use_v2: bool = True, use_enhanced: bool = True):
        self.use_v2 = use_v2
        self.use_enhanced = use_enhanced
        self.v1_generator = answer_generator
        self.v2_generator = answer_generator_v2_enhanced if use_enhanced else answer_generator_v2
    
    async def generate_answer(
        self,
        query: str,
        search_results: List[Any],  # Can be SearchResult or SearchResultSchema
        classification: Any,  # Can be old or new classification
        context_engineered: Optional[Any] = None,
        user_context: Optional[UserContextSchema] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate answer using appropriate generator"""
        
        if self.use_v2 and context_engineered and user_context:
            # Use V2 generator
            return await self._generate_v2_answer(
                classification,
                context_engineered,
                search_results,
                user_context,
                options
            )
        else:
            # Use V1 generator
            return await self._generate_v1_answer(
                query,
                search_results,
                classification,
                options
            )
    
    async def _generate_v2_answer(
        self,
        classification: Any,
        context_engineered: ContextEngineeredQuerySchema,
        search_results: List[Any],
        user_context: UserContextSchema,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate answer using V2 generator"""
        
        # Convert classification if needed
        if not isinstance(classification, ClassificationResultSchema):
            classification = self._convert_to_v2_classification(classification)
        
        # Convert search results if needed
        v2_results = []
        for result in search_results:
            if isinstance(result, SearchResultSchema):
                v2_results.append(result)
            else:
                # Convert from old SearchResult
                v2_result = self._convert_to_v2_search_result(result)
                v2_results.append(v2_result)
        
        # Generate with V2
        v2_answer = await self.v2_generator.generate_answer(
            classification,
            context_engineered,
            v2_results,
            user_context,
            options
        )
        
        return v2_answer
    
    async def _generate_v1_answer(
        self,
        query: str,
        search_results: List[Any],
        classification: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate answer using V1 generator"""
        
        # Convert search results to V1 format if needed
        v1_results = []
        for result in search_results:
            if isinstance(result, SearchResult):
                v1_results.append(result)
            else:
                # Convert from V2 SearchResultSchema
                v1_result = self._convert_to_v1_search_result(result)
                v1_results.append(v1_result)
        
        # Create V1 request
        request = AnswerGenerationRequest(
            query=query,
            search_results=v1_results,
            paradigm=classification.primary_paradigm.value if hasattr(classification, 'primary_paradigm') else 'bernard',
            max_tokens=options.get('max_tokens', 3000) if options else 3000,
            temperature=options.get('temperature', 0.7) if options else 0.7,
            include_citations=options.get('include_citations', True) if options else True
        )
        
        # Generate with V1
        v1_answer = await self.v1_generator.generate_answer(request)
        
        # Convert to V2-style response
        return {
            "content": v1_answer.answer,
            "paradigm": v1_answer.paradigm,
            "sources": [
                {
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet,
                    "credibility": source.credibility_score
                }
                for source in v1_answer.sources_used
            ],
            "metadata": {
                "generation_time": v1_answer.generation_time,
                "model_used": v1_answer.model_used,
                "confidence": 0.8,  # V1 doesn't have confidence
                "tone_applied": v1_answer.paradigm,
                "user_verbosity": "balanced"  # Default
            }
        }
    
    def _convert_to_v2_classification(self, old_classification) -> ClassificationResultSchema:
        """Convert old classification to V2 format"""
        from models.context_models import HostParadigm, QueryFeaturesSchema
        
        # This is similar to the bridge implementation
        features = QueryFeaturesSchema(
            text=old_classification.query if hasattr(old_classification, 'query') else "",
            tokens=[],
            entities=[],
            intent_signals=[],
            domain=None,
            urgency_score=0.5,
            complexity_score=0.5,
            emotional_valence=0.0
        )
        
        return ClassificationResultSchema(
            query=old_classification.query if hasattr(old_classification, 'query') else "",
            primary_paradigm=HostParadigm(old_classification.primary_paradigm.value),
            secondary_paradigm=HostParadigm(old_classification.secondary_paradigm.value) if old_classification.secondary_paradigm else None,
            distribution={},
            confidence=old_classification.confidence if hasattr(old_classification, 'confidence') else 0.8,
            features=features,
            reasoning={}
        )
    
    def _convert_to_v2_search_result(self, old_result: SearchResult) -> SearchResultSchema:
        """Convert old SearchResult to V2 SearchResultSchema"""
        return SearchResultSchema(
            url=old_result.url,
            title=old_result.title,
            snippet=old_result.snippet,
            source_api=getattr(old_result, 'source', 'unknown'),
            credibility_score=old_result.credibility_score if hasattr(old_result, 'credibility_score') else 0.5,
            metadata={
                "domain": old_result.domain,
                "published_date": old_result.published_date,
                "result_type": old_result.result_type
            }
        )
    
    def _convert_to_v1_search_result(self, v2_result: SearchResultSchema) -> SearchResult:
        """Convert V2 SearchResultSchema to old SearchResult"""
        from urllib.parse import urlparse
        metadata = v2_result.metadata or {}
        
        return SearchResult(
            title=v2_result.title,
            url=v2_result.url,
            snippet=v2_result.snippet,
            domain=metadata.get('domain', urlparse(v2_result.url).netloc),
            published_date=metadata.get('published_date'),
            result_type=metadata.get('result_type', 'web'),
            credibility_score=v2_result.credibility_score
        )


# Global adapter instance
answer_generator_adapter = AnswerGeneratorAdapter()


# Compatibility function
async def generate_paradigm_answer(
    query: str,
    search_results: List[Any],
    classification: Any,
    context_engineered: Optional[Any] = None,
    user_context: Optional[UserContextSchema] = None,
    use_v2: bool = True
) -> Dict[str, Any]:
    """Generate answer with V1/V2 compatibility"""
    
    adapter = AnswerGeneratorAdapter(use_v2=use_v2)
    return await adapter.generate_answer(
        query,
        search_results,
        classification,
        context_engineered,
        user_context
    )
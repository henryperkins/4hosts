"""
Enhanced Context Engineering V2 with Full Context Preservation
Implements W-S-C-I pipeline with complete reasoning and debug information
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

from models.context_models import (
    ClassificationResultSchema, ContextEngineeredQuerySchema,
    ContextLayerDebugInfo, HostParadigm
)
from services.llm_client import llm_client

# LLM availability check
try:
    LLM_AVAILABLE = llm_client is not None
except:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LayerProcessingResult:
    """Complete result from a W-S-C-I layer with full context"""
    layer_name: str
    start_time: float
    end_time: float
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    reasoning: List[str]
    transformations: Dict[str, Any] = field(default_factory=dict)
    removed_elements: List[str] = field(default_factory=list)
    added_elements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def processing_time_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    def to_debug_info(self) -> ContextLayerDebugInfo:
        """Convert to debug info schema"""
        return ContextLayerDebugInfo(
            layer_name=self.layer_name,
            processing_time_ms=self.processing_time_ms,
            input_state=self.input_state,
            output_state=self.output_state,
            reasoning=self.reasoning,
            removed_elements=self.removed_elements,
            added_elements=self.added_elements,
            transformations=self.transformations
        )


class WriteLayerV2:
    """Enhanced Write Layer - Creates contextual narratives with full tracking"""
    
    async def process(
        self, 
        classification: ClassificationResultSchema
    ) -> LayerProcessingResult:
        """Generate paradigm-aligned narrative queries"""
        start_time = time.time()
        
        input_state = {
            "query": classification.query,
            "paradigm": classification.primary_paradigm.value,
            "confidence": classification.confidence,
            "features": classification.features.model_dump()
        }
        
        # Paradigm-specific narrative templates
        narrative_templates = {
            HostParadigm.DOLORES: {
                "storyboard": "Uncovering systemic issues and injustices",
                "prompts": [
                    f"What hidden problems exist in {classification.query}",
                    f"Who is affected by issues related to {classification.query}",
                    f"What changes are needed regarding {classification.query}"
                ],
                "focus": "investigation, exposure, reform"
            },
            HostParadigm.BERNARD: {
                "storyboard": "Comprehensive academic analysis",
                "prompts": [
                    f"Scientific research on {classification.query}",
                    f"Data and statistics about {classification.query}",
                    f"Peer-reviewed studies on {classification.query}"
                ],
                "focus": "empirical evidence, methodology, analysis"
            },
            HostParadigm.MAEVE: {
                "storyboard": "Strategic business optimization",
                "prompts": [
                    f"Business strategies for {classification.query}",
                    f"Market analysis of {classification.query}",
                    f"ROI and optimization of {classification.query}"
                ],
                "focus": "efficiency, profitability, competitive advantage"
            },
            HostParadigm.TEDDY: {
                "storyboard": "Supportive community solutions",
                "prompts": [
                    f"How to help people with {classification.query}",
                    f"Community resources for {classification.query}",
                    f"Emotional support for {classification.query}"
                ],
                "focus": "empathy, assistance, wellbeing"
            }
        }
        
        template = narrative_templates[classification.primary_paradigm]
        
        # Generate enhanced queries using LLM if available
        enhanced_queries = []
        reasoning = []
        
        if llm_client and LLM_AVAILABLE:
            try:
                llm_prompt = f"""
                Generate 5-7 search queries for the topic: "{classification.query}"
                
                Paradigm: {classification.primary_paradigm.value}
                Focus: {template['focus']}
                Storyboard: {template['storyboard']}
                
                Requirements:
                1. Align with the paradigm's perspective
                2. Cover different aspects of the topic
                3. Be specific and searchable
                4. Format: Return only the queries, one per line
                """
                
                response = await llm_client.generate(llm_prompt, temperature=0.7)
                if response:
                    enhanced_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
                    reasoning.append(f"LLM generated {len(enhanced_queries)} paradigm-aligned queries")
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                reasoning.append(f"LLM generation failed: {str(e)}")
        
        # Fallback to template queries
        if not enhanced_queries:
            enhanced_queries = template['prompts']
            reasoning.append("Using template-based queries (LLM unavailable)")
        
        # Add original query for completeness
        enhanced_queries.insert(0, classification.query)
        
        output_state = {
            "narrative_queries": enhanced_queries,
            "storyboard": template['storyboard'],
            "focus_areas": template['focus'],
            "query_count": len(enhanced_queries),
            "paradigm_alignment": classification.primary_paradigm.value
        }
        
        # Track what was added
        added_elements = [q for q in enhanced_queries if q != classification.query]
        
        return LayerProcessingResult(
            layer_name="WriteLayer",
            start_time=start_time,
            end_time=time.time(),
            input_state=input_state,
            output_state=output_state,
            reasoning=reasoning,
            added_elements=added_elements,
            transformations={
                "query_expansion": len(enhanced_queries) - 1,
                "paradigm_template": classification.primary_paradigm.value
            },
            metadata={
                "llm_used": bool(enhanced_queries and llm_client),
                "template_used": template['storyboard']
            }
        )


class SelectLayerV2:
    """Enhanced Select Layer - Tool and strategy selection with tracking"""
    
    async def process(
        self,
        write_result: LayerProcessingResult,
        classification: ClassificationResultSchema
    ) -> LayerProcessingResult:
        """Select appropriate search tools and strategies"""
        start_time = time.time()
        
        input_state = {
            "queries": write_result.output_state['narrative_queries'],
            "paradigm": classification.primary_paradigm.value,
            "query_features": classification.features.model_dump()
        }
        
        # Paradigm-specific tool selection
        tool_strategies = {
            HostParadigm.DOLORES: {
                "primary_tools": ["brave_search", "google_search"],
                "strategy": "broad_investigation",
                "filters": ["recent", "controversial", "investigative"],
                "reasoning": "Diverse sources for uncovering hidden information"
            },
            HostParadigm.BERNARD: {
                "primary_tools": ["google_scholar", "arxiv", "pubmed"],
                "strategy": "academic_depth",
                "filters": ["peer_reviewed", "academic", "research"],
                "reasoning": "Academic databases for empirical evidence"
            },
            HostParadigm.MAEVE: {
                "primary_tools": ["google_search", "business_databases"],
                "strategy": "market_intelligence",
                "filters": ["business", "financial", "strategic"],
                "reasoning": "Business-focused sources for strategic insights"
            },
            HostParadigm.TEDDY: {
                "primary_tools": ["google_search"],
                "strategy": "trusted_sources",
                "filters": ["helpful", "community", "support"],
                "reasoning": "Trusted sources for supportive information"
            }
        }
        
        strategy = tool_strategies[classification.primary_paradigm]
        
        # Analyze query complexity to adjust strategy
        complexity_score = classification.features.complexity_score
        reasoning = [strategy['reasoning']]
        
        # Adjust tools based on complexity
        selected_tools = strategy['primary_tools'].copy()
        if complexity_score > 0.7:
            selected_tools.append("deep_web_search")
            reasoning.append(f"Added deep search due to high complexity ({complexity_score:.2f})")
        
        # Generate search configurations
        search_configs = []
        for query in write_result.output_state['narrative_queries']:
            config = {
                "query": query,
                "tools": selected_tools,
                "filters": strategy['filters'],
                "strategy": strategy['strategy'],
                "max_results": 10 if complexity_score > 0.5 else 5
            }
            search_configs.append(config)
        
        output_state = {
            "search_configs": search_configs,
            "selected_tools": list(set(selected_tools)),
            "strategy": strategy['strategy'],
            "filters": strategy['filters'],
            "total_searches": len(search_configs) * len(selected_tools)
        }
        
        return LayerProcessingResult(
            layer_name="SelectLayer",
            start_time=start_time,
            end_time=time.time(),
            input_state=input_state,
            output_state=output_state,
            reasoning=reasoning,
            transformations={
                "tool_selection": selected_tools,
                "strategy_applied": strategy['strategy'],
                "complexity_adjustment": complexity_score > 0.7
            },
            metadata={
                "paradigm_strategy": classification.primary_paradigm.value,
                "complexity_score": complexity_score
            }
        )


class CompressLayerV2:
    """Enhanced Compress Layer - Query optimization with tracking"""
    
    async def process(
        self,
        select_result: LayerProcessingResult,
        classification: ClassificationResultSchema
    ) -> LayerProcessingResult:
        """Compress and optimize queries for efficiency"""
        start_time = time.time()
        
        input_state = {
            "search_configs": select_result.output_state['search_configs'],
            "paradigm": classification.primary_paradigm.value
        }
        
        # Extract all queries
        all_queries = [config['query'] for config in select_result.output_state['search_configs']]
        
        # Remove duplicates and similar queries
        unique_queries = []
        removed_queries = []
        reasoning = []
        
        for query in all_queries:
            is_duplicate = False
            
            # Check for exact duplicates (case-insensitive)
            for existing in unique_queries:
                if query.lower() == existing.lower():
                    is_duplicate = True
                    removed_queries.append(f"{query} (exact duplicate)")
                    break
                
                # Check for high similarity
                similarity = self._calculate_similarity(query, existing)
                if similarity > 0.85:
                    is_duplicate = True
                    removed_queries.append(f"{query} (similar to: {existing})")
                    break
            
            if not is_duplicate:
                unique_queries.append(query)
        
        reasoning.append(f"Removed {len(removed_queries)} duplicate/similar queries")
        
        # Compress long queries
        compressed_queries = []
        for query in unique_queries:
            if len(query) > 100:
                # Use key terms extraction
                compressed = self._extract_key_terms(query)
                compressed_queries.append(compressed)
                reasoning.append(f"Compressed long query: {query[:50]}...")
            else:
                compressed_queries.append(query)
        
        # Paradigm-specific query limits
        query_limits = {
            HostParadigm.DOLORES: 8,
            HostParadigm.BERNARD: 12,
            HostParadigm.MAEVE: 10,
            HostParadigm.TEDDY: 6
        }
        
        max_queries = query_limits.get(classification.primary_paradigm, 8)
        
        # Limit queries if needed
        if len(compressed_queries) > max_queries:
            removed_count = len(compressed_queries) - max_queries
            removed_queries.extend(compressed_queries[max_queries:])
            compressed_queries = compressed_queries[:max_queries]
            reasoning.append(f"Limited to {max_queries} queries (removed {removed_count})")
        
        output_state = {
            "optimized_queries": compressed_queries,
            "query_count": len(compressed_queries),
            "compression_ratio": len(compressed_queries) / len(all_queries) if all_queries else 1.0,
            "removed_count": len(removed_queries)
        }
        
        return LayerProcessingResult(
            layer_name="CompressLayer",
            start_time=start_time,
            end_time=time.time(),
            input_state=input_state,
            output_state=output_state,
            reasoning=reasoning,
            removed_elements=removed_queries,
            transformations={
                "deduplication": len(removed_queries),
                "compression_applied": any("Compressed" in r for r in reasoning),
                "paradigm_limit": max_queries
            },
            metadata={
                "original_count": len(all_queries),
                "final_count": len(compressed_queries)
            }
        )
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        # Simple word-based similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_key_terms(self, query: str) -> str:
        """Extract key terms from a long query"""
        # Remove common words and keep important terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'about', 'what',
            'how', 'when', 'where', 'why', 'which', 'who', 'is', 'are'
        }
        
        words = query.split()
        key_terms = [w for w in words if w.lower() not in stop_words]
        
        # Keep first 10 key terms
        return ' '.join(key_terms[:10])


class IsolateLayerV2:
    """Enhanced Isolate Layer - Final refinement with tracking"""
    
    async def process(
        self,
        compress_result: LayerProcessingResult,
        classification: ClassificationResultSchema
    ) -> LayerProcessingResult:
        """Isolate and finalize paradigm-specific queries"""
        start_time = time.time()
        
        input_state = {
            "queries": compress_result.output_state['optimized_queries'],
            "paradigm": classification.primary_paradigm.value
        }
        
        # Paradigm-specific query refinement
        paradigm_prefixes = {
            HostParadigm.DOLORES: ["investigate", "expose", "uncover"],
            HostParadigm.BERNARD: ["research", "study", "analysis"],
            HostParadigm.MAEVE: ["strategy", "optimize", "business"],
            HostParadigm.TEDDY: ["help", "support", "guide"]
        }
        
        refined_queries = []
        transformations = []
        reasoning = []
        
        prefixes = paradigm_prefixes[classification.primary_paradigm]
        
        for i, query in enumerate(compress_result.output_state['optimized_queries']):
            # Check if query already has paradigm alignment
            has_alignment = any(prefix in query.lower() for prefix in prefixes)
            
            if not has_alignment and i > 0:  # Keep original query as first
                # Add paradigm-specific prefix
                prefix = prefixes[i % len(prefixes)]
                refined = f"{prefix} {query}"
                refined_queries.append(refined)
                transformations.append(f"Added '{prefix}' to: {query}")
            else:
                refined_queries.append(query)
        
        reasoning.append(f"Applied paradigm-specific refinement to {len(transformations)} queries")
        
        # Generate search strategy
        search_strategy = {
            "paradigm": classification.primary_paradigm.value,
            "query_distribution": {
                "primary": refined_queries[:3],
                "secondary": refined_queries[3:6] if len(refined_queries) > 3 else [],
                "exploratory": refined_queries[6:] if len(refined_queries) > 6 else []
            },
            "execution_order": "parallel",
            "result_merging": "credibility_weighted"
        }
        
        output_state = {
            "final_queries": refined_queries,
            "search_strategy": search_strategy,
            "paradigm_alignment": classification.primary_paradigm.value,
            "total_queries": len(refined_queries)
        }
        
        return LayerProcessingResult(
            layer_name="IsolateLayer",
            start_time=start_time,
            end_time=time.time(),
            input_state=input_state,
            output_state=output_state,
            reasoning=reasoning,
            transformations={
                "queries_refined": len(transformations),
                "strategy_created": True,
                "paradigm_prefixes": prefixes
            },
            added_elements=transformations,
            metadata={
                "final_paradigm": classification.primary_paradigm.value,
                "strategy_type": "parallel_execution"
            }
        )


class ContextEngineeringPipelineV2:
    """Enhanced W-S-C-I Pipeline with full context preservation"""
    
    def __init__(self):
        self.write_layer = WriteLayerV2()
        self.select_layer = SelectLayerV2()
        self.compress_layer = CompressLayerV2()
        self.isolate_layer = IsolateLayerV2()
        
        # History tracking with memory management
        self.processing_history = defaultdict(list)
        self.max_history_size = 100
    
    async def process_query(
        self,
        classification: ClassificationResultSchema,
        include_debug: bool = True
    ) -> ContextEngineeredQuerySchema:
        """Process query through W-S-C-I pipeline with full tracking"""
        
        pipeline_start = time.time()
        
        # Process through layers
        write_result = await self.write_layer.process(classification)
        select_result = await self.select_layer.process(write_result, classification)
        compress_result = await self.compress_layer.process(select_result, classification)
        isolate_result = await self.isolate_layer.process(compress_result, classification)
        
        # Collect debug information
        debug_info = []
        if include_debug:
            debug_info = [
                write_result.to_debug_info(),
                select_result.to_debug_info(),
                compress_result.to_debug_info(),
                isolate_result.to_debug_info()
            ]
        
        # Create comprehensive result
        result = ContextEngineeredQuerySchema(
            original_query=classification.query,
            classification=classification,
            write_layer_output=write_result.output_state,
            select_layer_output=select_result.output_state,
            compress_layer_output=compress_result.output_state,
            isolate_layer_output=isolate_result.output_state,
            debug_info=debug_info,
            refined_queries=isolate_result.output_state['final_queries'],
            search_strategy=isolate_result.output_state['search_strategy'],
            context_metadata={
                "pipeline_time_ms": (time.time() - pipeline_start) * 1000,
                "total_transformations": sum([
                    len(write_result.added_elements),
                    len(compress_result.removed_elements),
                    len(isolate_result.transformations)
                ]),
                "paradigm": classification.primary_paradigm.value,
                "layer_timings": {
                    "write": write_result.processing_time_ms,
                    "select": select_result.processing_time_ms,
                    "compress": compress_result.processing_time_ms,
                    "isolate": isolate_result.processing_time_ms
                }
            }
        )
        
        # Store in history with memory management
        self._store_in_history(classification.query, result)
        
        return result
    
    def _store_in_history(self, query: str, result: ContextEngineeredQuerySchema):
        """Store processing history with size limits"""
        history = self.processing_history[query]
        history.append({
            "timestamp": datetime.utcnow(),
            "result": result
        })
        
        # Limit history size
        if len(history) > self.max_history_size:
            history.pop(0)
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get pipeline processing metrics"""
        total_queries = sum(len(h) for h in self.processing_history.values())
        
        return {
            "unique_queries_processed": len(self.processing_history),
            "total_processings": total_queries,
            "average_reprocessings": total_queries / max(len(self.processing_history), 1),
            "memory_usage": {
                "history_entries": total_queries,
                "unique_queries": len(self.processing_history)
            }
        }


# Create singleton instance
context_pipeline_v2 = ContextEngineeringPipelineV2()
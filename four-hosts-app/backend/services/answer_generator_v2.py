"""
Enhanced Answer Generator V2 with Full Context Utilization
Generates paradigm-aligned answers using all available context
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from models.context_models import (
    ClassificationResultSchema, ContextEngineeredQuerySchema,
    UserContextSchema, HostParadigm, SearchResultSchema
)
from services.llm_client import llm_client
from services.text_compression import text_compressor

logger = logging.getLogger(__name__)


class ParadigmToneGenerator:
    """Generates content with paradigm-specific tone and style"""
    
    PARADIGM_STYLES = {
        HostParadigm.DOLORES: {
            "tone": "investigative, revealing, justice-focused",
            "voice": "direct, compelling, exposing truth",
            "structure": "problem identification → evidence → call to action",
            "phrases": [
                "The investigation reveals...",
                "Hidden beneath the surface...",
                "It's crucial to understand that...",
                "The evidence suggests..."
            ]
        },
        HostParadigm.BERNARD: {
            "tone": "analytical, objective, evidence-based",
            "voice": "scholarly, precise, methodical",
            "structure": "hypothesis → data → analysis → conclusion",
            "phrases": [
                "Research indicates...",
                "The data demonstrates...",
                "Studies have shown...",
                "According to peer-reviewed sources..."
            ]
        },
        HostParadigm.MAEVE: {
            "tone": "strategic, results-oriented, optimizing",
            "voice": "professional, actionable, efficient",
            "structure": "situation → opportunity → strategy → ROI",
            "phrases": [
                "To optimize outcomes...",
                "The strategic approach involves...",
                "Key performance indicators suggest...",
                "For maximum efficiency..."
            ]
        },
        HostParadigm.TEDDY: {
            "tone": "supportive, empathetic, encouraging",
            "voice": "warm, helpful, understanding",
            "structure": "empathy → guidance → resources → encouragement",
            "phrases": [
                "I understand this can be challenging...",
                "Here's how we can help...",
                "You're not alone in this...",
                "Let me guide you through..."
            ]
        }
    }


class AnswerGeneratorV2:
    """Enhanced answer generator with full context awareness"""
    
    def __init__(self):
        self.tone_generator = ParadigmToneGenerator()
        self.compressor = text_compressor
    
    async def generate_answer(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        search_results: List[SearchResultSchema],
        user_context: UserContextSchema,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive answer using all context"""
        
        # Extract paradigm style
        paradigm_style = self.tone_generator.PARADIGM_STYLES[classification.primary_paradigm]
        
        # Build comprehensive context
        full_context = self._build_full_context(
            classification,
            context_engineered,
            search_results,
            user_context
        )
        
        # Generate answer based on user verbosity preference
        if user_context.verbosity_preference == "minimal":
            answer = await self._generate_concise_answer(
                full_context, paradigm_style, search_results
            )
        elif user_context.verbosity_preference == "detailed":
            answer = await self._generate_detailed_answer(
                full_context, paradigm_style, search_results
            )
        else:  # balanced
            answer = await self._generate_balanced_answer(
                full_context, paradigm_style, search_results
            )
        
        # Extract and format sources
        sources_used = self._select_best_sources(
            search_results,
            user_context.source_limit,
            classification.primary_paradigm
        )
        
        return {
            "content": answer,
            "paradigm": classification.primary_paradigm.value,
            "sources": sources_used,
            "metadata": {
                "confidence": classification.confidence,
                "tone_applied": paradigm_style['tone'],
                "user_verbosity": user_context.verbosity_preference,
                "context_layers_used": list(full_context.keys()),
                "generation_timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _build_full_context(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        search_results: List[SearchResultSchema],
        user_context: UserContextSchema
    ) -> Dict[str, Any]:
        """Build comprehensive context from all sources"""
        
        # Extract key insights from each W-S-C-I layer
        write_insights = self._extract_write_insights(context_engineered.write_layer_output)
        select_insights = self._extract_select_insights(context_engineered.select_layer_output)
        compress_insights = self._extract_compress_insights(context_engineered.compress_layer_output)
        isolate_insights = self._extract_isolate_insights(context_engineered.isolate_layer_output)
        
        # Get debug reasoning if available
        debug_reasoning = []
        if context_engineered.debug_info:
            for debug in context_engineered.debug_info:
                debug_reasoning.extend(debug.reasoning)
        
        return {
            "query": classification.query,
            "paradigm": classification.primary_paradigm.value,
            "paradigm_distribution": classification.distribution,
            "user_preferences": {
                "location": user_context.location,
                "language": user_context.language,
                "role": user_context.role,
                "default_paradigm": user_context.default_paradigm
            },
            "narrative_context": write_insights,
            "tool_strategy": select_insights,
            "optimization_notes": compress_insights,
            "execution_strategy": isolate_insights,
            "debug_reasoning": debug_reasoning,
            "search_summary": {
                "total_results": len(search_results),
                "high_credibility_count": sum(1 for r in search_results if r.credibility_score >= 0.7),
                "sources": list(set(r.source_api for r in search_results))
            }
        }
    
    def _extract_write_insights(self, write_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from Write layer"""
        return {
            "storyboard": write_output.get("storyboard", ""),
            "focus_areas": write_output.get("focus_areas", ""),
            "narrative_queries": write_output.get("narrative_queries", [])[:3]  # Top 3
        }
    
    def _extract_select_insights(self, select_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from Select layer"""
        return {
            "strategy": select_output.get("strategy", ""),
            "tools_selected": select_output.get("selected_tools", []),
            "filters_applied": select_output.get("filters", [])
        }
    
    def _extract_compress_insights(self, compress_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from Compress layer"""
        return {
            "compression_ratio": compress_output.get("compression_ratio", 1.0),
            "queries_removed": compress_output.get("removed_count", 0),
            "final_query_count": compress_output.get("query_count", 0)
        }
    
    def _extract_isolate_insights(self, isolate_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from Isolate layer"""
        return {
            "search_strategy": isolate_output.get("search_strategy", {}),
            "paradigm_alignment": isolate_output.get("paradigm_alignment", ""),
            "query_distribution": isolate_output.get("search_strategy", {}).get("query_distribution", {})
        }
    
    async def _generate_balanced_answer(
        self,
        context: Dict[str, Any],
        paradigm_style: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> str:
        """Generate balanced answer with moderate detail"""
        
        # Create prompt with full context
        prompt = f"""
        Generate a {paradigm_style['tone']} answer to: "{context['query']}"
        
        Context:
        - Paradigm: {context['paradigm']} ({paradigm_style['voice']})
        - Structure: {paradigm_style['structure']}
        - Focus: {context['narrative_context']['focus_areas']}
        - User location: {context['user_preferences']['location'] or 'Not specified'}
        
        Key insights from {len(search_results)} sources:
        {self._format_key_insights(search_results[:5])}
        
        Requirements:
        1. Use {paradigm_style['tone']} tone throughout
        2. Follow the {paradigm_style['structure']} structure
        3. Include 2-3 specific examples or data points
        4. Keep response between 200-400 words
        5. Cite sources naturally within the text
        """
        
        # Add paradigm-specific phrases
        prompt += f"\nStart with one of these phrases: {paradigm_style['phrases'][:2]}"
        
        try:
            response = await llm_client.generate(
                prompt,
                temperature=0.7,
                max_tokens=800
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_answer(context, paradigm_style, search_results)
    
    async def _generate_concise_answer(
        self,
        context: Dict[str, Any],
        paradigm_style: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> str:
        """Generate concise answer for minimal verbosity preference"""
        
        key_points = self._extract_key_points(search_results[:3])
        
        prompt = f"""
        Provide a brief {paradigm_style['tone']} answer to: "{context['query']}"
        
        Key findings: {key_points}
        
        Requirements:
        - Maximum 150 words
        - Direct and to the point
        - Use {paradigm_style['voice']} voice
        - Include 1-2 most important facts
        """
        
        try:
            response = await llm_client.generate(prompt, temperature=0.5, max_tokens=300)
            return response.strip()
        except:
            # Fallback
            return f"{paradigm_style['phrases'][0]} {key_points}"
    
    async def _generate_detailed_answer(
        self,
        context: Dict[str, Any],
        paradigm_style: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> str:
        """Generate detailed answer for verbose preference"""
        
        # Include more context and debug information
        debug_context = "\n".join(context.get('debug_reasoning', [])[:5])
        
        prompt = f"""
        Provide a comprehensive {paradigm_style['tone']} analysis of: "{context['query']}"
        
        Full Context:
        - Paradigm: {context['paradigm']} with distribution: {context['paradigm_distribution']}
        - Narrative: {context['narrative_context']['storyboard']}
        - Strategy: {context['tool_strategy']['strategy']}
        - Processing insights: {debug_context}
        
        Detailed findings from {len(search_results)} sources:
        {self._format_detailed_insights(search_results[:8])}
        
        Requirements:
        1. Use {paradigm_style['structure']} structure
        2. Include multiple perspectives and data points
        3. Provide in-depth analysis (400-600 words)
        4. Address potential counterarguments
        5. Conclude with actionable insights
        """
        
        try:
            response = await llm_client.generate(
                prompt,
                temperature=0.7,
                max_tokens=1200
            )
            return response.strip()
        except:
            return self._generate_fallback_answer(context, paradigm_style, search_results, detailed=True)
    
    def _format_key_insights(self, results: List[SearchResultSchema]) -> str:
        """Format key insights from search results"""
        insights = []
        for i, result in enumerate(results, 1):
            insight = f"{i}. {result.title}: {result.snippet[:100]}..."
            if result.credibility_score >= 0.7:
                insight += " (High credibility)"
            insights.append(insight)
        return "\n".join(insights)
    
    def _format_detailed_insights(self, results: List[SearchResultSchema]) -> str:
        """Format detailed insights with metadata"""
        insights = []
        for result in results:
            insight = f"- Source: {result.source_api}\n"
            insight += f"  Title: {result.title}\n"
            insight += f"  Key point: {result.snippet[:150]}...\n"
            insight += f"  Credibility: {result.credibility_score:.2f} - {result.credibility_explanation}\n"
            insights.append(insight)
        return "\n".join(insights)
    
    def _extract_key_points(self, results: List[SearchResultSchema]) -> str:
        """Extract key points for concise answers"""
        points = []
        for result in results:
            # Extract first sentence or key fact
            snippet = result.snippet
            if '.' in snippet:
                first_sentence = snippet.split('.')[0] + '.'
                points.append(first_sentence)
        return " ".join(points[:2])
    
    def _select_best_sources(
        self,
        results: List[SearchResultSchema],
        limit: int,
        paradigm: HostParadigm
    ) -> List[Dict[str, Any]]:
        """Select best sources based on paradigm and credibility"""
        
        # Sort by credibility and paradigm relevance
        scored_results = []
        for result in results:
            score = result.credibility_score
            
            # Boost score for paradigm-aligned sources
            if paradigm == HostParadigm.BERNARD and result.source_api in ["arxiv", "pubmed"]:
                score += 0.2
            elif paradigm == HostParadigm.DOLORES and "investigat" in result.title.lower():
                score += 0.1
            
            scored_results.append((score, result))
        
        # Sort and select top sources
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        selected_sources = []
        for _, result in scored_results[:limit]:
            source = {
                "title": result.title,
                "url": result.url,
                "snippet": self.compressor.compress_text(
                    result.snippet,
                    max_tokens=100
                ),
                "credibility": result.credibility_score,
                "source": result.source_api
            }
            selected_sources.append(source)
        
        return selected_sources
    
    def _generate_fallback_answer(
        self,
        context: Dict[str, Any],
        paradigm_style: Dict[str, Any],
        search_results: List[SearchResultSchema],
        detailed: bool = False
    ) -> str:
        """Generate fallback answer when LLM is unavailable"""
        
        intro = paradigm_style['phrases'][0]
        
        if detailed:
            # Detailed fallback
            answer_parts = [
                intro,
                f"\n\nRegarding {context['query']}, our research reveals several key insights:",
                "\n\nKey Findings:"
            ]
            
            for i, result in enumerate(search_results[:5], 1):
                answer_parts.append(
                    f"\n{i}. {result.title}\n   {result.snippet[:150]}..."
                )
            
            answer_parts.append(
                f"\n\nThese findings align with the {context['paradigm']} perspective, "
                f"focusing on {context['narrative_context']['focus_areas']}."
            )
        else:
            # Concise fallback
            key_finding = search_results[0].snippet[:200] if search_results else "No specific data available"
            answer_parts = [
                intro,
                key_finding + "..."
            ]
        
        return "\n".join(answer_parts)


# Create singleton instance
answer_generator_v2 = AnswerGeneratorV2()
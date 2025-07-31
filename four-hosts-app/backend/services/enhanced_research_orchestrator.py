"""
Enhanced Research Orchestrator with Brave MCP Integration
Combines existing search APIs with Brave MCP tools for comprehensive research
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from services.research_orchestrator import research_orchestrator
from services.classification_engine import HostParadigm
from services.brave_mcp_integration import brave_mcp, BraveSearchType
from services.search_apis import create_search_manager, SearchConfig
from services.llm_client import llm_client
from services.cache import cache_manager

logger = logging.getLogger(__name__)


class EnhancedResearchOrchestrator:
    """Research orchestrator enhanced with Brave MCP capabilities"""
    
    def __init__(self):
        self.base_orchestrator = research_orchestrator
        self.search_api_manager = None
        self.brave_enabled = False
    
    async def initialize(self):
        """Initialize enhanced orchestrator"""
        # Initialize search API manager
        self.search_api_manager = create_search_manager()
        await self.search_api_manager.initialize()
        
        # Initialize Brave MCP if available
        try:
            from services.brave_mcp_integration import initialize_brave_mcp
            self.brave_enabled = await initialize_brave_mcp()
        except Exception as e:
            logger.warning(f"Brave MCP initialization failed: {e}")
            self.brave_enabled = False
    
    async def execute_paradigm_research(
        self,
        query: str,
        primary_paradigm: HostParadigm,
        secondary_paradigm: Optional[HostParadigm] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute enhanced research with Brave MCP integration"""
        
        options = options or {}
        use_brave = self.brave_enabled and options.get("use_brave", True)
        
        # Start timer
        start_time = datetime.utcnow()
        
        # Execute searches in parallel
        search_tasks = []
        
        # Traditional API searches
        if options.get("use_traditional", True):
            search_tasks.extend([
                self._execute_google_search(query, primary_paradigm),
                self._execute_arxiv_search(query, primary_paradigm) if primary_paradigm == HostParadigm.BERNARD else None,
                self._execute_pubmed_search(query, primary_paradigm) if primary_paradigm in [HostParadigm.BERNARD, HostParadigm.TEDDY] else None,
            ])
        
        # Brave MCP searches
        if use_brave:
            search_tasks.extend([
                self._execute_brave_search(query, primary_paradigm, BraveSearchType.WEB),
                self._execute_brave_search(query, primary_paradigm, BraveSearchType.NEWS) if primary_paradigm == HostParadigm.DOLORES else None,
                self._execute_brave_search(query, primary_paradigm, BraveSearchType.SUMMARIZER) if primary_paradigm == HostParadigm.BERNARD else None,
            ])
        
        # Filter out None tasks
        search_tasks = [task for task in search_tasks if task is not None]
        
        # Execute all searches in parallel
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results
        all_results = []
        brave_results = []
        traditional_results = []
        
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Search task {i} failed: {result}")
                continue
            
            if isinstance(result, dict):
                if result.get("source") == "brave":
                    brave_results.append(result)
                else:
                    traditional_results.append(result)
                all_results.append(result)
        
        # Synthesize results using LLM
        synthesis = await self._synthesize_results(
            query=query,
            paradigm=primary_paradigm,
            brave_results=brave_results,
            traditional_results=traditional_results,
            all_results=all_results
        )
        
        # Calculate metrics
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "query": query,
            "primary_paradigm": primary_paradigm.value,
            "secondary_paradigm": secondary_paradigm.value if secondary_paradigm else None,
            "synthesis": synthesis,
            "sources": {
                "brave": len(brave_results),
                "traditional": len(traditional_results),
                "total": len(all_results)
            },
            "search_results": all_results,
            "duration": duration,
            "timestamp": end_time.isoformat(),
            "enhanced_features": {
                "brave_mcp_enabled": use_brave,
                "paradigm_optimization": True,
                "multi_source_synthesis": True
            }
        }
    
    async def _execute_google_search(self, query: str, paradigm: HostParadigm) -> Dict[str, Any]:
        """Execute Google search with paradigm optimization"""
        try:
            # Add paradigm-specific terms
            enhanced_query = self._enhance_query_for_paradigm(query, paradigm)
            
            # Use search API manager to search Google
            if "google" in self.search_api_manager.apis:
                config = SearchConfig(
                    paradigm=paradigm.value,
                    max_results=5,
                    include_pdfs=False
                )
                results = await self.search_api_manager.apis["google"].search(enhanced_query, config)
            else:
                # Fallback to all available APIs
                config = SearchConfig(
                    paradigm=paradigm.value,
                    max_results=5
                )
                all_results = await self.search_api_manager.search_all(enhanced_query, config)
                results = all_results.get("google", []) or list(all_results.values())[0] if all_results else []
            
            return {
                "source": "google",
                "paradigm": paradigm.value,
                "results": results,
                "query": enhanced_query
            }
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            raise
    
    async def _execute_arxiv_search(self, query: str, paradigm: HostParadigm) -> Dict[str, Any]:
        """Execute ArXiv search for academic paradigms"""
        try:
            if "arxiv" in self.search_api_manager.apis:
                config = SearchConfig(
                    paradigm=paradigm.value,
                    max_results=3,
                    include_pdfs=True
                )
                results = await self.search_api_manager.apis["arxiv"].search(query, config)
            else:
                results = []
            
            return {
                "source": "arxiv",
                "paradigm": paradigm.value,
                "results": results,
                "query": query
            }
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            raise
    
    async def _execute_pubmed_search(self, query: str, paradigm: HostParadigm) -> Dict[str, Any]:
        """Execute PubMed search for medical/scientific paradigms"""
        try:
            if "pubmed" in self.search_api_manager.apis:
                config = SearchConfig(
                    paradigm=paradigm.value,
                    max_results=3
                )
                results = await self.search_api_manager.apis["pubmed"].search(query, config)
            else:
                results = []
            
            return {
                "source": "pubmed",
                "paradigm": paradigm.value,
                "results": results,
                "query": query
            }
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            raise
    
    async def _execute_brave_search(
        self, 
        query: str, 
        paradigm: HostParadigm, 
        search_type: BraveSearchType
    ) -> Dict[str, Any]:
        """Execute Brave search through MCP"""
        try:
            # Use paradigm value for the search
            result = await brave_mcp.search_with_paradigm(
                query=query,
                paradigm=paradigm.value,
                search_type=search_type
            )
            
            return {
                "source": "brave",
                "search_type": search_type.value,
                "paradigm": paradigm.value,
                **result
            }
        except Exception as e:
            logger.error(f"Brave {search_type} search failed: {e}")
            raise
    
    def _enhance_query_for_paradigm(self, query: str, paradigm: HostParadigm) -> str:
        """Add paradigm-specific terms to enhance search relevance"""
        
        paradigm_terms = {
            HostParadigm.DOLORES: ["expose", "investigate", "uncover", "corruption"],
            HostParadigm.TEDDY: ["help", "support", "community", "care"],
            HostParadigm.BERNARD: ["research", "data", "study", "analysis"],
            HostParadigm.MAEVE: ["strategy", "optimize", "business", "competitive"]
        }
        
        terms = paradigm_terms.get(paradigm, [])
        
        # Add one relevant term to the query
        if terms and len(query.split()) < 20:  # Don't enhance very long queries
            return f"{query} {terms[0]}"
        
        return query
    
    async def _synthesize_results(
        self,
        query: str,
        paradigm: HostParadigm,
        brave_results: List[Dict[str, Any]],
        traditional_results: List[Dict[str, Any]],
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize results from multiple sources using LLM"""
        
        # Prepare synthesis prompt
        prompt = self._build_synthesis_prompt(
            query=query,
            paradigm=paradigm,
            brave_results=brave_results,
            traditional_results=traditional_results
        )
        
        # Generate synthesis
        synthesis_text = await llm_client.generate_paradigm_content(
            prompt=prompt,
            paradigm=paradigm.value,
            max_tokens=3000,
            temperature=0.7
        )
        
        # Extract key insights
        insights = self._extract_insights(synthesis_text, paradigm)
        
        return {
            "content": synthesis_text,
            "insights": insights,
            "source_distribution": {
                "brave": len(brave_results),
                "traditional": len(traditional_results)
            },
            "paradigm_alignment": self._calculate_paradigm_alignment(synthesis_text, paradigm)
        }
    
    def _build_synthesis_prompt(
        self,
        query: str,
        paradigm: HostParadigm,
        brave_results: List[Dict[str, Any]],
        traditional_results: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for result synthesis"""
        
        # Format results for prompt
        brave_summary = self._format_results_for_prompt(brave_results, "Brave Search")
        traditional_summary = self._format_results_for_prompt(traditional_results, "Traditional Search APIs")
        
        paradigm_instructions = {
            HostParadigm.DOLORES: "Focus on exposing hidden truths and challenging established narratives.",
            HostParadigm.TEDDY: "Emphasize helpful resources and supportive information for the community.",
            HostParadigm.BERNARD: "Provide data-driven analysis with empirical evidence and research citations.",
            HostParadigm.MAEVE: "Deliver strategic insights and actionable recommendations for optimization."
        }
        
        return f"""Research Query: {query}

You are synthesizing research results from multiple sources with a {paradigm.value} paradigm perspective.

{paradigm_instructions.get(paradigm, "")}

Sources:

{brave_summary}

{traditional_summary}

Please provide a comprehensive synthesis that:
1. Integrates findings from all sources
2. Maintains the {paradigm.value} perspective throughout
3. Highlights key insights and patterns
4. Cites specific sources when making claims
5. Identifies any contradictions or gaps in the information

Generate a well-structured response appropriate for the {paradigm.value} paradigm."""
    
    def _format_results_for_prompt(self, results: List[Dict[str, Any]], source_name: str) -> str:
        """Format search results for inclusion in synthesis prompt"""
        if not results:
            return f"{source_name}: No results available."
        
        formatted = f"{source_name}:\n"
        for i, result in enumerate(results[:5]):  # Limit to top 5
            if "results" in result and isinstance(result["results"], list):
                for j, item in enumerate(result["results"][:3]):  # Top 3 per source
                    # Handle both dict and SearchResult objects
                    if hasattr(item, "title"):
                        # It's a SearchResult object
                        title = item.title
                        snippet = (item.snippet or "")[:200]
                    elif isinstance(item, dict):
                        title = item.get("title", "Untitled")
                        snippet = item.get("snippet", item.get("description", ""))[:200]
                    else:
                        continue
                    formatted += f"- {title}: {snippet}...\n"
        
        return formatted
    
    def _extract_insights(self, synthesis: str, paradigm: HostParadigm) -> List[str]:
        """Extract key insights from synthesis"""
        # Simple extraction based on common patterns
        insights = []
        
        # Look for bullet points or numbered lists
        lines = synthesis.split('\n')
        for line in lines:
            line = line.strip()
            if line and (
                line.startswith('•') or 
                line.startswith('-') or 
                (len(line) > 2 and line[0].isdigit() and line[1] in '.)')
            ):
                insights.append(line.lstrip('•-0123456789.) '))
        
        return insights[:5]  # Top 5 insights
    
    def _calculate_paradigm_alignment(self, text: str, paradigm: HostParadigm) -> float:
        """Calculate how well the synthesis aligns with the paradigm"""
        # Simple keyword-based alignment scoring
        paradigm_keywords = {
            HostParadigm.DOLORES: ["expose", "reveal", "uncover", "justice", "fight"],
            HostParadigm.TEDDY: ["help", "support", "care", "community", "protect"],
            HostParadigm.BERNARD: ["data", "research", "evidence", "study", "analysis"],
            HostParadigm.MAEVE: ["strategy", "optimize", "competitive", "advantage", "efficient"]
        }
        
        keywords = paradigm_keywords.get(paradigm, [])
        text_lower = text.lower()
        
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        alignment = min(matches / len(keywords), 1.0) if keywords else 0.5
        
        return round(alignment, 2)


# Global instance
enhanced_orchestrator = EnhancedResearchOrchestrator()


# Initialize during startup
async def initialize_enhanced_orchestrator():
    """Initialize the enhanced research orchestrator"""
    await enhanced_orchestrator.initialize()
    logger.info("✓ Enhanced research orchestrator initialized")
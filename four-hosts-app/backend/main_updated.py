"""
Updated Four Hosts Research API with Phase 3 Research Execution Layer
Integrates Context Engineering Pipeline with Real Search APIs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum
import asyncio
import uuid
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our new services
from services.research_orchestrator import (
    research_orchestrator,
    initialize_research_system,
    execute_research
)
from services.cache import initialize_cache
from services.credibility import get_source_credibility
from services.llm_client import initialize_llm_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Four Hosts Research API",
    version="2.0.0",
    description="Paradigm-aware research with integrated Context Engineering Pipeline"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class Paradigm(str, Enum):
    DOLORES = "dolores"
    TEDDY = "teddy"
    BERNARD = "bernard"
    MAEVE = "maeve"

class ResearchDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

class ResearchOptions(BaseModel):
    depth: ResearchDepth = ResearchDepth.STANDARD
    paradigm_override: Optional[Paradigm] = None
    include_secondary: bool = True
    max_sources: int = Field(default=50, ge=10, le=200)
    language: str = "en"
    region: str = "us"
    enable_real_search: bool = True  # New option to enable real search

class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    options: ResearchOptions = ResearchOptions()

class ParadigmClassification(BaseModel):
    primary: Paradigm
    secondary: Optional[Paradigm]
    distribution: Dict[str, float]
    confidence: float
    explanation: Dict[str, str]

class ResearchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SourceResult(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str
    credibility_score: float
    published_date: Optional[str] = None
    source_type: str = "web"

class ResearchResult(BaseModel):
    research_id: str
    query: str
    status: ResearchStatus
    paradigm_analysis: Dict[str, Any]
    answer: Dict[str, Any]
    sources: List[SourceResult]
    metadata: Dict[str, Any]
    cost_info: Optional[Dict[str, float]] = None

# In-memory storage (would be replaced with database in production)
research_store: Dict[str, Dict] = {}

# Global state
system_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize the research system on startup"""
    global system_initialized

    logger.info("Initializing Four Hosts Research System...")

    try:
        # Initialize cache system
        cache_success = await initialize_cache()
        if cache_success:
            logger.info("✓ Cache system initialized")
        else:
            logger.warning("⚠ Cache system failed - continuing without cache")


        # Initialize research orchestrator
        await initialize_research_system()
        logger.info("✓ Research orchestrator initialized")

        # Initialize LLM client
        await initialize_llm_client()
        logger.info("✓ LLM client initialized")
        system_initialized = True
        logger.info("🚀 Four Hosts Research System ready!")

    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        system_initialized = False

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Four Hosts Research API v2.0",
        "version": "2.0.0",
        "system_initialized": system_initialized,
        "features": [
            "Paradigm-aware research",
            "Real search API integration",
            "Source credibility scoring",
            "Context engineering pipeline",
            "Result caching and deduplication"
        ]
    }

@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Check system components
        stats = await research_orchestrator.get_execution_stats()

        return {
            "status": "healthy" if system_initialized else "degraded",
            "system_initialized": system_initialized,
            "cache_available": True,  # Would check actual cache status
            "execution_stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/paradigms/classify")
async def classify_paradigm(query: str):
    """Classify a query into paradigms without full research."""
    try:
        classification = await classify_query(query)
        return {
            "query": query,
            "classification": classification.dict(),
            "suggested_approach": get_paradigm_approach_suggestion(classification.primary)
        }
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/research/query")
async def submit_research(research: ResearchQuery, background_tasks: BackgroundTasks):
    """Submit a research query for paradigm-based analysis."""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    research_id = f"res_{uuid.uuid4().hex[:12]}"

    try:
        # Classify the query
        classification = await classify_query(research.query)

        # Store research request
        research_data = {
            "id": research_id,
            "query": research.query,
            "options": research.options.dict(),
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "created_at": datetime.utcnow().isoformat(),
            "results": None
        }
        research_store[research_id] = research_data

        # Start research execution in background
        if research.options.enable_real_search and system_initialized:
            background_tasks.add_task(execute_real_research, research_id, research)
        else:
            # Fallback to mock research for development
            background_tasks.add_task(execute_mock_research, research_id, research.query, classification)

        return {
            "research_id": research_id,
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "estimated_completion": "2025-01-20T10:30:45Z",
            "real_search_enabled": research.options.enable_real_search and system_initialized
        }

    except Exception as e:
        logger.error(f"Research submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research submission failed: {str(e)}")

@app.get("/research/status/{research_id}")
async def get_research_status(research_id: str):
    """Get the status of a research query."""
    if research_id not in research_store:
        raise HTTPException(status_code=404, detail="Research not found")

    research = research_store[research_id]
    return {
        "research_id": research_id,
        "status": research["status"],
        "paradigm": research["paradigm_classification"]["primary"],
        "started_at": research["created_at"],
        "progress": research.get("progress", {}),
        "cost_info": research.get("cost_info")
    }

@app.get("/research/results/{research_id}")
async def get_research_results(research_id: str):
    """Get completed research results."""
    if research_id not in research_store:
        raise HTTPException(status_code=404, detail="Research not found")

    research = research_store[research_id]
    if research["status"] != ResearchStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Research is {research['status']}")

    return research["results"]

@app.get("/sources/credibility/{domain}")
async def get_domain_credibility(domain: str, paradigm: Paradigm = Paradigm.BERNARD):
    """Get credibility score for a specific domain"""
    try:
        credibility = await get_source_credibility(domain, paradigm.value)
        return {
            "domain": domain,
            "paradigm": paradigm.value,
            "credibility_score": credibility.overall_score,
            "domain_authority": credibility.domain_authority,
            "bias_rating": credibility.bias_rating,
            "fact_check_rating": credibility.fact_check_rating,
            "paradigm_alignment": credibility.paradigm_alignment,
            "reputation_factors": credibility.reputation_factors
        }
    except Exception as e:
        logger.error(f"Credibility check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Credibility check failed: {str(e)}")

@app.get("/system/stats")
async def get_system_stats():
    """Get system performance statistics"""
    try:
        if not system_initialized:
            return {"error": "System not initialized"}

        stats = await research_orchestrator.get_execution_stats()
        return {
            "system_status": "operational",
            "research_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return {"error": str(e)}

# Background task functions
async def execute_real_research(research_id: str, research: ResearchQuery):
    """Execute real research using the complete pipeline"""
    try:
        # Update status
        research_store[research_id]["status"] = ResearchStatus.IN_PROGRESS

        # This is where we would integrate with the Context Engineering Pipeline
        # For now, we'll create a simplified version

        # Step 1: Classify (already done)
        classification = ParadigmClassification(**research_store[research_id]["paradigm_classification"])

        # Step 2: Create mock context engineered query (would come from pipeline)
        from types import SimpleNamespace

        mock_classification = SimpleNamespace()
        mock_classification.primary_paradigm = SimpleNamespace()
        mock_classification.primary_paradigm.value = classification.primary.value
        mock_classification.secondary_paradigm = None

        # Mock select output with paradigm-specific queries
        mock_select_output = SimpleNamespace()
        paradigm_queries = generate_paradigm_queries(research.query, classification.primary.value)
        mock_select_output.search_queries = paradigm_queries

        mock_context_query = SimpleNamespace()
        mock_context_query.original_query = research.query
        mock_context_query.classification = mock_classification
        mock_context_query.select_output = mock_select_output

        # Step 3: Execute research
        execution_result = await execute_research(mock_context_query, research.options.max_sources)

        # Step 4: Format search results
        formatted_sources = []
        search_results_for_synthesis = []

        for result in execution_result.filtered_results[:research.options.max_sources]:
            # Format for API response
            formatted_sources.append(SourceResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                domain=result.domain,
                credibility_score=getattr(result, 'credibility_score', 0.5),
                published_date=result.published_date.isoformat() if result.published_date else None,
                source_type=result.result_type
            ))

            # Format for answer synthesis
            search_results_for_synthesis.append({
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet,
                'domain': result.domain,
                'credibility_score': getattr(result, 'credibility_score', 0.5),
                'published_date': result.published_date,
                'result_type': result.result_type
            })

        # Step 5: Generate paradigm-aware answer using Answer Generation System
        from services.answer_generator_continued import answer_orchestrator

        # Mock context engineering output for answer generation
        context_engineering = {
            'write_output': {
                'documentation_focus': get_paradigm_focus(classification.primary),
                'key_themes': classification.explanation.get(classification.primary.value, '').split()[:4],
                'narrative_frame': get_paradigm_approach(classification.primary)
            },
            'select_output': {
                'search_queries': paradigm_queries,
                'source_preferences': [],
                'max_sources': research.options.max_sources
            }
        }

        # Generate answer
        generated_answer = await answer_orchestrator.generate_answer(
            paradigm=classification.primary.value,
            query=research.query,
            search_results=search_results_for_synthesis,
            context_engineering=context_engineering,
            options={
                'research_id': research_id,
                'max_length': 2000,
                'include_citations': True
            }
        )

        # Format answer for API response
        answer_sections = []
        for section in generated_answer.sections:
            answer_sections.append({
                "title": section.title,
                "paradigm": section.paradigm,
                "content": section.content,
                "confidence": section.confidence,
                "sources_count": len(section.citations)
            })

        # Convert citations to list format
        citations_list = []
        for cite_id, citation in generated_answer.citations.items():
            citations_list.append({
                "id": cite_id,
                "source": citation.source_title,
                "url": citation.source_url,
                "credibility_score": citation.credibility_score,
                "paradigm_alignment": classification.primary.value
            })

        # Create final result
        final_result = ResearchResult(
            research_id=research_id,
            query=research.query,
            status=ResearchStatus.COMPLETED,
            paradigm_analysis={
                "primary": {
                    "paradigm": classification.primary.value,
                    "confidence": classification.confidence,
                    "approach": get_paradigm_approach(classification.primary),
                    "focus": get_paradigm_focus(classification.primary)
                }
            },
            answer={
                "summary": generated_answer.summary,
                "sections": answer_sections,
                "action_items": generated_answer.action_items,
                "citations": citations_list
            },
            sources=formatted_sources,
            metadata={
                "total_sources_analyzed": len(execution_result.raw_results),
                "high_quality_sources": len([s for s in formatted_sources if s.credibility_score > 0.7]),
                "search_queries_executed": len(execution_result.search_queries_executed),
                "processing_time_seconds": execution_result.execution_metrics["processing_time_seconds"],
                "answer_generation_time": generated_answer.generation_time,
                "synthesis_quality": generated_answer.synthesis_quality,
                "paradigms_used": [classification.primary.value],
                "real_search": True
            },
            cost_info=execution_result.cost_breakdown
        )

        # Store results
        research_store[research_id]["status"] = ResearchStatus.COMPLETED
        research_store[research_id]["results"] = final_result.dict()
        research_store[research_id]["cost_info"] = execution_result.cost_breakdown

        logger.info(f"✓ Real research completed for {research_id}")

    except Exception as e:
        logger.error(f"Real research execution failed for {research_id}: {str(e)}")
        research_store[research_id]["status"] = ResearchStatus.FAILED
        research_store[research_id]["error"] = str(e)

async def execute_mock_research(research_id: str, query: str, classification: ParadigmClassification):
    """Execute mock research (fallback for development)"""
    try:
        # Update status
        research_store[research_id]["status"] = ResearchStatus.IN_PROGRESS

        # Simulate processing time
        await asyncio.sleep(2)

        # Generate mock results
        paradigm = classification.primary

        mock_sources = [
            SourceResult(
                title=f"Mock Research Article about {query}",
                url="https://example.com/mock-article-1",
                snippet=f"This is a mock research result for {paradigm.value} paradigm analysis of: {query}",
                domain="example.com",
                credibility_score=0.85,
                source_type="web"
            ),
            SourceResult(
                title=f"Academic Study on {query}",
                url="https://academic-example.com/study",
                snippet=f"Academic perspective on {query} from {paradigm.value} viewpoint",
                domain="academic-example.com",
                credibility_score=0.92,
                source_type="academic"
            )
        ]

        results = ResearchResult(
            research_id=research_id,
            query=query,
            status=ResearchStatus.COMPLETED,
            paradigm_analysis={
                "primary": {
                    "paradigm": paradigm.value,
                    "confidence": classification.confidence,
                    "approach": get_paradigm_approach(paradigm),
                    "focus": get_paradigm_focus(paradigm)
                }
            },
            answer={
                "summary": generate_paradigm_summary(query, paradigm),
                "sections": [
                    {
                        "title": get_section_title(paradigm),
                        "paradigm": paradigm.value,
                        "content": generate_paradigm_content(query, paradigm),
                        "confidence": 0.85,
                        "sources_count": len(mock_sources)
                    }
                ],
                "action_items": generate_action_items(query, paradigm),
                "citations": []
            },
            sources=mock_sources,
            metadata={
                "total_sources_analyzed": 50,
                "high_quality_sources": 15,
                "search_queries_executed": 5,
                "processing_time_seconds": 2.1,
                "paradigms_used": [paradigm.value],
                "real_search": False
            }
        )

        # Store results
        research_store[research_id]["status"] = ResearchStatus.COMPLETED
        research_store[research_id]["results"] = results.dict()

        logger.info(f"✓ Mock research completed for {research_id}")

    except Exception as e:
        logger.error(f"Mock research execution failed for {research_id}: {str(e)}")
        research_store[research_id]["status"] = ResearchStatus.FAILED
        research_store[research_id]["error"] = str(e)

# Helper functions (from original main.py)
async def classify_query(query: str) -> ParadigmClassification:
    """Classify a query into paradigms using keyword matching and simple heuristics."""
    query_lower = query.lower()

    paradigm_keywords = {
        Paradigm.DOLORES: ["injustice", "systemic", "power", "revolution", "expose", "corrupt", "unfair", "inequality"],
        Paradigm.TEDDY: ["protect", "help", "care", "support", "vulnerable", "community", "safety", "wellbeing"],
        Paradigm.BERNARD: ["analyze", "data", "research", "study", "evidence", "statistical", "scientific", "measure"],
        Paradigm.MAEVE: ["strategy", "compete", "optimize", "control", "influence", "business", "advantage", "implement"]
    }

    scores = {paradigm: 0.0 for paradigm in Paradigm}
    for paradigm, keywords in paradigm_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                scores[paradigm] += 1.0

    total_score = sum(scores.values()) or 1
    distribution = {p.value: scores[p] / total_score for p in Paradigm}

    sorted_paradigms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_paradigms[0][0] if sorted_paradigms[0][1] > 0 else Paradigm.BERNARD
    secondary = sorted_paradigms[1][0] if len(sorted_paradigms) > 1 and sorted_paradigms[1][1] > 0 else None

    confidence = min(0.95, max(0.5, sorted_paradigms[0][1] / total_score if total_score > 0 else 0.5))

    explanations = {
        Paradigm.DOLORES: "Focus on systemic issues and power dynamics",
        Paradigm.TEDDY: "Emphasis on protection and care",
        Paradigm.BERNARD: "Analytical and evidence-based approach",
        Paradigm.MAEVE: "Strategic and action-oriented perspective"
    }

    explanation_dict = {primary.value: explanations[primary]}
    if secondary:
        explanation_dict[secondary.value] = explanations[secondary]

    return ParadigmClassification(
        primary=primary,
        secondary=secondary,
        distribution=distribution,
        confidence=confidence,
        explanation=explanation_dict
    )

def generate_paradigm_queries(query: str, paradigm: str) -> List[Dict[str, Any]]:
    """Generate paradigm-specific search queries"""
    modifiers = {
        "dolores": ["controversy", "expose", "systemic", "injustice"],
        "teddy": ["support", "help", "community", "resources"],
        "bernard": ["research", "study", "analysis", "data"],
        "maeve": ["strategy", "competitive", "optimize", "framework"]
    }

    queries = [{"query": query, "type": "original", "weight": 1.0}]

    for i, modifier in enumerate(modifiers.get(paradigm, [])[:3]):
        queries.append({
            "query": f"{query} {modifier}",
            "type": f"paradigm_modified_{i+1}",
            "weight": 0.8 - (i * 0.1)
        })

    return queries

def get_paradigm_approach_suggestion(paradigm: Paradigm) -> str:
    """Get approach suggestion for a paradigm"""
    suggestions = {
        Paradigm.DOLORES: "Focus on exposing systemic issues and empowering resistance",
        Paradigm.TEDDY: "Prioritize community support and protective measures",
        Paradigm.BERNARD: "Emphasize empirical research and data-driven analysis",
        Paradigm.MAEVE: "Develop strategic frameworks and actionable implementation plans"
    }
    return suggestions[paradigm]

def get_paradigm_approach(paradigm: Paradigm) -> str:
    approaches = {
        Paradigm.DOLORES: "revolutionary",
        Paradigm.TEDDY: "protective",
        Paradigm.BERNARD: "analytical",
        Paradigm.MAEVE: "strategic"
    }
    return approaches[paradigm]

def get_paradigm_focus(paradigm: Paradigm) -> str:
    focuses = {
        Paradigm.DOLORES: "Exposing systemic injustices and power imbalances",
        Paradigm.TEDDY: "Protecting and supporting vulnerable communities",
        Paradigm.BERNARD: "Providing objective analysis and empirical evidence",
        Paradigm.MAEVE: "Delivering actionable strategies and competitive advantage"
    }
    return focuses[paradigm]

def get_section_title(paradigm: Paradigm) -> str:
    titles = {
        Paradigm.DOLORES: "Systemic Analysis",
        Paradigm.TEDDY: "Protection Framework",
        Paradigm.BERNARD: "Data-Driven Insights",
        Paradigm.MAEVE: "Strategic Implementation"
    }
    return titles[paradigm]

def generate_paradigm_summary(query: str, paradigm: Paradigm) -> str:
    summaries = {
        Paradigm.DOLORES: f"Revolutionary analysis revealing systemic issues in: {query}",
        Paradigm.TEDDY: f"Protective framework for supporting those affected by: {query}",
        Paradigm.BERNARD: f"Analytical examination with empirical evidence on: {query}",
        Paradigm.MAEVE: f"Strategic action plan for leveraging opportunities in: {query}"
    }
    return summaries[paradigm]

def generate_paradigm_content(query: str, paradigm: Paradigm) -> str:
    content_templates = {
        Paradigm.DOLORES: "Analysis reveals deep-rooted systemic issues that require fundamental transformation...",
        Paradigm.TEDDY: "Comprehensive support framework prioritizing vulnerable stakeholders and community wellbeing...",
        Paradigm.BERNARD: "Statistical analysis indicates significant patterns and correlations in the data...",
        Paradigm.MAEVE: "Strategic framework identifies key leverage points and actionable implementation steps..."
    }
    return content_templates[paradigm]

def generate_action_items(query: str, paradigm: Paradigm) -> List[Dict]:
    action_templates = {
        Paradigm.DOLORES: [
            {"priority": "high", "action": "Document and expose systemic failures", "timeframe": "Immediate", "paradigm": "dolores"}
        ],
        Paradigm.TEDDY: [
            {"priority": "high", "action": "Establish support systems for affected communities", "timeframe": "1-2 weeks", "paradigm": "teddy"}
        ],
        Paradigm.BERNARD: [
            {"priority": "high", "action": "Conduct comprehensive data analysis", "timeframe": "2-4 weeks", "paradigm": "bernard"}
        ],
        Paradigm.MAEVE: [
            {"priority": "high", "action": "Implement strategic quick wins", "timeframe": "1 week", "paradigm": "maeve"}
        ]
    }
    return action_templates[paradigm]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

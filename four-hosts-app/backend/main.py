from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from enum import Enum
import asyncio
import uuid
from datetime import datetime

app = FastAPI(title="Four Hosts Research API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    max_sources: int = Field(default=100, ge=10, le=1000)
    language: str = "en"
    region: str = "us"

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

research_store = {}

@app.get("/")
async def root():
    return {"message": "Four Hosts Research API", "version": "1.0.0"}

@app.post("/paradigms/classify")
async def classify_paradigm(query: str):
    """Classify a query into paradigms without full research."""
    classification = await classify_query(query)
    return classification

@app.post("/research/query")
async def submit_research(research: ResearchQuery):
    """Submit a research query for paradigm-based analysis."""
    research_id = f"res_{uuid.uuid4().hex[:12]}"
    
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
    
    # Start async research process
    asyncio.create_task(execute_research(research_id, research.query, classification))
    
    return {
        "research_id": research_id,
        "status": ResearchStatus.PROCESSING,
        "paradigm_classification": classification.dict(),
        "estimated_completion": "2025-01-20T10:30:45Z"
    }

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
        "started_at": research["created_at"]
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

async def classify_query(query: str) -> ParadigmClassification:
    """Classify a query into paradigms using keyword matching and simple heuristics."""
    query_lower = query.lower()
    
    # Paradigm keywords
    paradigm_keywords = {
        Paradigm.DOLORES: ["injustice", "systemic", "power", "revolution", "expose", "corrupt", "unfair", "inequality"],
        Paradigm.TEDDY: ["protect", "help", "care", "support", "vulnerable", "community", "safety", "wellbeing"],
        Paradigm.BERNARD: ["analyze", "data", "research", "study", "evidence", "statistical", "scientific", "measure"],
        Paradigm.MAEVE: ["strategy", "compete", "optimize", "control", "influence", "business", "advantage", "implement"]
    }
    
    # Calculate scores
    scores = {paradigm: 0.0 for paradigm in Paradigm}
    for paradigm, keywords in paradigm_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                scores[paradigm] += 1.0
    
    # Normalize scores
    total_score = sum(scores.values()) or 1
    distribution = {p.value: scores[p] / total_score for p in Paradigm}
    
    # Determine primary and secondary
    sorted_paradigms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_paradigms[0][0] if sorted_paradigms[0][1] > 0 else Paradigm.BERNARD
    secondary = sorted_paradigms[1][0] if len(sorted_paradigms) > 1 and sorted_paradigms[1][1] > 0 else None
    
    # Calculate confidence
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

async def execute_research(research_id: str, query: str, classification: ParadigmClassification):
    """Execute the research based on paradigm classification."""
    try:
        # Update status
        research_store[research_id]["status"] = ResearchStatus.IN_PROGRESS
        
        # Simulate research process
        await asyncio.sleep(2)  # Simulate API calls
        
        # Generate paradigm-specific results
        paradigm = classification.primary
        
        results = {
            "research_id": research_id,
            "query": query,
            "status": ResearchStatus.COMPLETED,
            "paradigm_analysis": {
                "primary": {
                    "paradigm": paradigm.value,
                    "confidence": classification.confidence,
                    "approach": get_paradigm_approach(paradigm),
                    "focus": get_paradigm_focus(paradigm)
                }
            },
            "answer": {
                "summary": generate_paradigm_summary(query, paradigm),
                "sections": [
                    {
                        "title": get_section_title(paradigm),
                        "paradigm": paradigm.value,
                        "content": generate_paradigm_content(query, paradigm),
                        "confidence": 0.85,
                        "sources_count": 15
                    }
                ],
                "action_items": generate_action_items(query, paradigm),
                "citations": generate_citations(paradigm)
            },
            "metadata": {
                "total_sources_analyzed": 247,
                "high_quality_sources": 23,
                "search_queries_executed": 8,
                "processing_time_seconds": 2.1,
                "paradigms_used": [paradigm.value]
            }
        }
        
        # Store results
        research_store[research_id]["status"] = ResearchStatus.COMPLETED
        research_store[research_id]["results"] = results
        
    except Exception as e:
        research_store[research_id]["status"] = ResearchStatus.FAILED
        research_store[research_id]["error"] = str(e)

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

def generate_citations(paradigm: Paradigm) -> List[Dict]:
    citation_templates = {
        Paradigm.DOLORES: [
            {
                "id": "cite_001",
                "source": "Critical Studies Journal",
                "title": "Power Structures and Systemic Inequality",
                "url": "https://example.com/critical-studies",
                "credibility_score": 0.92,
                "paradigm_alignment": "dolores"
            }
        ],
        Paradigm.TEDDY: [
            {
                "id": "cite_001",
                "source": "Community Care Review",
                "title": "Best Practices in Vulnerable Population Support",
                "url": "https://example.com/community-care",
                "credibility_score": 0.94,
                "paradigm_alignment": "teddy"
            }
        ],
        Paradigm.BERNARD: [
            {
                "id": "cite_001",
                "source": "Journal of Empirical Research",
                "title": "Statistical Analysis of Current Trends",
                "url": "https://example.com/empirical-research",
                "credibility_score": 0.96,
                "paradigm_alignment": "bernard"
            }
        ],
        Paradigm.MAEVE: [
            {
                "id": "cite_001",
                "source": "Strategic Management Review",
                "title": "Competitive Advantage Through Innovation",
                "url": "https://example.com/strategic-management",
                "credibility_score": 0.93,
                "paradigm_alignment": "maeve"
            }
        ]
    }
    return citation_templates[paradigm]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
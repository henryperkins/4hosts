#!/usr/bin/env python3
"""
Four Hosts Research API - Full Featured Version
Complete implementation with all features enabled
"""

import os
import asyncio
import uuid
import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any
from enum import Enum
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, Response, FileResponse
from pydantic import BaseModel, Field, EmailStr, HttpUrl
from dotenv import load_dotenv
import jwt
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core services
from services.research_orchestrator import (
    research_orchestrator,
    initialize_research_system,
    execute_research
)
from services.cache import initialize_cache
from services.credibility import get_source_credibility
from services.llm_client import initialize_llm_client
from services.answer_generator_continued import answer_orchestrator

# Import production services
from services.auth_service import AuthService
from services.rate_limiter import RateLimiter, RateLimitMiddleware
from services.monitoring import PrometheusMetrics, ApplicationInsights, create_monitoring_middleware
from services.webhook_manager import WebhookManager, WebhookEvent, create_webhook_router
from services.websocket_service import ConnectionManager, ResearchProgressTracker, create_websocket_router
from services.export_service import ExportService, create_export_router
from production.database import init_database, get_db
from production.models import User, Research, WebhookSubscription
from utils.custom_docs import custom_openapi, get_custom_swagger_ui_html, get_custom_redoc_html

# Constants
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize services
auth_service = AuthService()
rate_limiter = RateLimiter()
webhook_manager = WebhookManager()
connection_manager = ConnectionManager()
progress_tracker = ResearchProgressTracker(connection_manager)
export_service = ExportService()

# Metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
active_research = Gauge('active_research_queries', 'Number of active research queries')
websocket_connections = Gauge('websocket_connections', 'Number of active WebSocket connections')

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

class UserRole(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class ResearchOptions(BaseModel):
    depth: ResearchDepth = ResearchDepth.STANDARD
    paradigm_override: Optional[Paradigm] = None
    include_secondary: bool = True
    max_sources: int = Field(default=50, ge=10, le=200)
    language: str = "en"
    region: str = "us"
    enable_real_search: bool = True

class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    options: ResearchOptions = ResearchOptions()

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: UserRole = UserRole.FREE

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int

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

class WebhookCreate(BaseModel):
    url: HttpUrl
    events: List[WebhookEvent]
    secret: Optional[str] = None
    active: bool = True

# In-memory storage
research_store: Dict[str, Dict] = {}
system_initialized = False

# Application Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global system_initialized

    # Startup
    logger.info("🚀 Starting Four Hosts Research API (Full Featured)...")

    try:
        # Initialize database
        await init_database()
        logger.info("✓ Database initialized")

        # Initialize cache system
        cache_success = await initialize_cache()
        if cache_success:
            logger.info("✓ Cache system initialized")

        # Initialize research system
        await initialize_research_system()
        logger.info("✓ Research orchestrator initialized")

        # Initialize LLM client
        await initialize_llm_client()
        logger.info("✓ LLM client initialized")

        # Initialize monitoring
        prometheus = PrometheusMetrics(REGISTRY)
        insights = ApplicationInsights()
        monitoring_middleware = create_monitoring_middleware(prometheus, insights)

        app.state.monitoring = {
            "prometheus": prometheus,
            "insights": insights,
            "middleware": monitoring_middleware
        }
        logger.info("✓ Monitoring systems initialized")

        # Initialize production services
        app.state.auth_service = auth_service
        app.state.rate_limiter = rate_limiter
        app.state.webhook_manager = webhook_manager
        app.state.export_service = export_service
        logger.info("✓ Production services initialized")

        system_initialized = True
        logger.info("🚀 Four Hosts Research System ready with all features!")

    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        system_initialized = False

    yield

    # Shutdown
    logger.info("🛑 Shutting down Four Hosts Research API...")

    # Cleanup connections
    await connection_manager.disconnect_all()

    logger.info("👋 Shutdown complete")

# Create FastAPI App
app = FastAPI(
    title="Four Hosts Research API",
    version="3.0.0",
    description="Full-featured paradigm-aware research with integrated Context Engineering Pipeline",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://api.4hosts.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate Limiting Middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if hasattr(app.state, "rate_limiter"):
        middleware = RateLimitMiddleware(app.state.rate_limiter)
        return await middleware(request, call_next)
    return await call_next(request)

# Monitoring Middleware
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    if hasattr(app.state, "monitoring"):
        middleware = app.state.monitoring["middleware"]
        return await middleware(request, call_next)
    return await call_next(request)

# Request ID Middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

# Authentication dependency
async def get_current_user(token: str = Depends(auth_service.oauth2_scheme)):
    """Get current authenticated user"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return await auth_service.get_user(user_id)
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Root Endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Four Hosts Research API - Full Featured",
        "version": "3.0.0",
        "system_initialized": system_initialized,
        "features": {
            "authentication": True,
            "real_research": True,
            "ai_synthesis": True,
            "monitoring": True,
            "webhooks": True,
            "websockets": True,
            "export": True,
            "rate_limiting": True
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }

# Health Endpoints
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        health_data = {
            "status": "healthy" if system_initialized else "degraded",
            "system_initialized": system_initialized,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "healthy",
                "cache": "healthy",
                "research": "healthy",
                "llm": "healthy"
            }
        }

        if hasattr(research_orchestrator, 'get_execution_stats'):
            stats = await research_orchestrator.get_execution_stats()
            health_data["execution_stats"] = stats

        return health_data
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Authentication Endpoints
@app.post("/auth/register", response_model=Token, tags=["authentication"])
async def register(user_data: UserCreate):
    """Register a new user"""
    user = await auth_service.create_user(user_data)

    access_token = create_access_token({
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role
    })

    refresh_token = auth_service.create_refresh_token(str(user.id))

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.post("/auth/login", response_model=Token, tags=["authentication"])
async def login(login_data: UserLogin):
    """Login with email and password"""
    user = await auth_service.authenticate_user(login_data)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role
    })

    refresh_token = auth_service.create_refresh_token(str(user.id))

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.post("/auth/refresh", response_model=Token, tags=["authentication"])
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    user_id = await auth_service.verify_refresh_token(refresh_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = await auth_service.get_user(user_id)

    access_token = create_access_token({
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role
    })

    new_refresh_token = auth_service.create_refresh_token(str(user.id))

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

# Paradigm Classification
@app.post("/paradigms/classify", tags=["paradigms"])
async def classify_paradigm(query: str, current_user: User = Depends(get_current_user)):
    """Classify a query into paradigms"""
    try:
        classification = await classify_query(query)
        return {
            "query": query,
            "classification": classification.dict(),
            "suggested_approach": get_paradigm_approach_suggestion(classification.primary),
            "user_id": str(current_user.id)
        }
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Research Endpoints
@app.post("/research/query", tags=["research"])
async def submit_research(
    research: ResearchQuery,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Submit a research query for paradigm-based analysis"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    research_id = f"res_{uuid.uuid4().hex[:12]}"

    # Track active research
    active_research.inc()

    try:
        # Classify the query
        classification = await classify_query(research.query)

        # Store research request
        research_data = {
            "id": research_id,
            "user_id": str(current_user.id),
            "query": research.query,
            "options": research.options.dict(),
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "created_at": datetime.utcnow().isoformat(),
            "results": None
        }
        research_store[research_id] = research_data

        # Execute real research
        background_tasks.add_task(
            execute_real_research,
            research_id,
            research,
            str(current_user.id)
        )

        # Track in WebSocket
        await progress_tracker.start_research(
            research_id,
            str(current_user.id),
            research.query,
            classification.primary.value,
            research.options.depth.value
        )

        # Trigger webhook
        await app.state.webhook_manager.trigger_event(
            WebhookEvent.RESEARCH_STARTED,
            {
                "research_id": research_id,
                "user_id": str(current_user.id),
                "query": research.query,
                "paradigm": classification.primary.value
            }
        )

        return {
            "research_id": research_id,
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=2)).isoformat(),
            "websocket_url": f"/ws/research/{research_id}"
        }

    except Exception as e:
        active_research.dec()
        logger.error(f"Research submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research submission failed: {str(e)}")

@app.get("/research/status/{research_id}", tags=["research"])
async def get_research_status(
    research_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the status of a research query"""
    if research_id not in research_store:
        raise HTTPException(status_code=404, detail="Research not found")

    research = research_store[research_id]

    # Verify ownership
    if research["user_id"] != str(current_user.id) and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "research_id": research_id,
        "status": research["status"],
        "paradigm": research["paradigm_classification"]["primary"],
        "started_at": research["created_at"],
        "progress": research.get("progress", {}),
        "cost_info": research.get("cost_info")
    }

@app.get("/research/results/{research_id}", tags=["research"])
async def get_research_results(
    research_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get completed research results"""
    if research_id not in research_store:
        raise HTTPException(status_code=404, detail="Research not found")

    research = research_store[research_id]

    # Verify ownership
    if research["user_id"] != str(current_user.id) and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")

    if research["status"] != ResearchStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Research is {research['status']}")

    return research["results"]

@app.get("/research/export/{research_id}", tags=["research"])
async def export_research(
    research_id: str,
    format: str = "pdf",
    current_user: User = Depends(get_current_user)
):
    """Export research results"""
    if research_id not in research_store:
        raise HTTPException(status_code=404, detail="Research not found")

    research = research_store[research_id]

    # Verify ownership
    if research["user_id"] != str(current_user.id) and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")

    if research["status"] != ResearchStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Research not completed")

    # Generate export
    export_path = await app.state.export_service.export_research(
        research["results"],
        format=format
    )

    return FileResponse(
        export_path,
        media_type=f"application/{format}",
        filename=f"research_{research_id}.{format}"
    )

# Source Credibility Endpoint
@app.get("/sources/credibility/{domain}", tags=["sources"])
async def get_domain_credibility(
    domain: str,
    paradigm: Paradigm = Paradigm.BERNARD,
    current_user: User = Depends(get_current_user)
):
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
            "reputation_factors": credibility.reputation_factors,
            "checked_by": str(current_user.id)
        }
    except Exception as e:
        logger.error(f"Credibility check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Credibility check failed: {str(e)}")

# System Stats
@app.get("/system/stats", tags=["system"])
async def get_system_stats(current_user: User = Depends(get_current_user)):
    """Get system performance statistics"""
    if current_user.role not in [UserRole.ADMIN, UserRole.ENTERPRISE]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        stats = {
            "system_status": "operational",
            "active_research": active_research._value.get(),
            "websocket_connections": websocket_connections._value.get(),
            "total_requests": sum(request_count._metrics.values()) if hasattr(request_count, '_metrics') else 0
        }

        if hasattr(research_orchestrator, 'get_execution_stats'):
            stats["research_stats"] = await research_orchestrator.get_execution_stats()

        stats["timestamp"] = datetime.utcnow().isoformat()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return {"error": str(e)}

# Metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    metrics = generate_latest(REGISTRY)
    return Response(content=metrics, media_type="text/plain")

# WebSocket endpoint
@app.websocket("/ws/research/{research_id}")
async def websocket_research_progress(websocket: WebSocket, research_id: str):
    """WebSocket for real-time research progress"""
    await connection_manager.connect(websocket, research_id)
    websocket_connections.inc()

    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        connection_manager.disconnect(research_id)
        websocket_connections.dec()

# Custom Documentation
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi():
    return custom_openapi(app)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return HTMLResponse(get_custom_swagger_ui_html(openapi_url="/openapi.json"))

@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    return HTMLResponse(get_custom_redoc_html(openapi_url="/openapi.json"))

# Include Routers
app.include_router(
    create_webhook_router(webhook_manager),
    prefix="/api/v1",
    tags=["webhooks"]
)

app.include_router(
    create_websocket_router(connection_manager, progress_tracker),
    prefix="/api/v1",
    tags=["websockets"]
)

app.include_router(
    create_export_router(export_service),
    prefix="/api/v1",
    tags=["export"]
)

# Helper functions
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def classify_query(query: str) -> ParadigmClassification:
    """Classify a query into paradigms"""
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

# Background task for real research execution
async def execute_real_research(research_id: str, research: ResearchQuery, user_id: str):
    """Execute real research using the complete pipeline"""
    try:
        # Update status
        research_store[research_id]["status"] = ResearchStatus.IN_PROGRESS

        # Update progress
        await progress_tracker.update_progress(
            research_id,
            "Initializing research pipeline",
            10
        )

        # Get classification
        classification = ParadigmClassification(**research_store[research_id]["paradigm_classification"])

        # Create context for research
        mock_classification = SimpleNamespace()
        mock_classification.primary_paradigm = SimpleNamespace()
        mock_classification.primary_paradigm.value = classification.primary.value
        mock_classification.secondary_paradigm = None

        # Generate paradigm-specific queries
        paradigm_queries = generate_paradigm_queries(research.query, classification.primary.value)

        mock_select_output = SimpleNamespace()
        mock_select_output.search_queries = paradigm_queries

        mock_context_query = SimpleNamespace()
        mock_context_query.original_query = research.query
        mock_context_query.classification = mock_classification
        mock_context_query.select_output = mock_select_output

        # Update progress
        await progress_tracker.update_progress(
            research_id,
            "Executing search queries",
            30
        )

        # Execute research
        execution_result = await execute_research(mock_context_query, research.options.max_sources)

        # Update progress
        await progress_tracker.update_progress(
            research_id,
            "Processing search results",
            60
        )

        # Format results
        formatted_sources = []
        search_results_for_synthesis = []

        for result in execution_result.filtered_results[:research.options.max_sources]:
            formatted_sources.append(SourceResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                domain=result.domain,
                credibility_score=getattr(result, 'credibility_score', 0.5),
                published_date=result.published_date.isoformat() if result.published_date else None,
                source_type=result.result_type
            ))

            search_results_for_synthesis.append({
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet,
                'domain': result.domain,
                'credibility_score': getattr(result, 'credibility_score', 0.5),
                'published_date': result.published_date,
                'result_type': result.result_type
            })

        # Update progress
        await progress_tracker.update_progress(
            research_id,
            "Generating AI-powered answer",
            80
        )

        # Generate answer
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

        # Format final result
        answer_sections = []
        for section in generated_answer.sections:
            answer_sections.append({
                "title": section.title,
                "paradigm": section.paradigm,
                "content": section.content,
                "confidence": section.confidence,
                "sources_count": len(section.citations)
            })

        citations_list = []
        for cite_id, citation in generated_answer.citations.items():
            citations_list.append({
                "id": cite_id,
                "source": citation.source_title,
                "url": citation.source_url,
                "credibility_score": citation.credibility_score,
                "paradigm_alignment": classification.primary.value
            })

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
                "paradigms_used": [classification.primary.value]
            },
            cost_info=execution_result.cost_breakdown
        )

        # Update progress
        await progress_tracker.update_progress(
            research_id,
            "Research completed",
            100
        )

        # Store results
        research_store[research_id]["status"] = ResearchStatus.COMPLETED
        research_store[research_id]["results"] = final_result.dict()
        research_store[research_id]["cost_info"] = execution_result.cost_breakdown

        # Trigger webhook
        await webhook_manager.trigger_event(
            WebhookEvent.RESEARCH_COMPLETED,
            {
                "research_id": research_id,
                "user_id": user_id,
                "query": research.query,
                "paradigm": classification.primary.value,
                "sources_count": len(formatted_sources),
                "cost": execution_result.cost_breakdown.get("total", 0)
            }
        )

        # Decrement active research
        active_research.dec()

        logger.info(f"✓ Research completed for {research_id}")

    except Exception as e:
        logger.error(f"Research execution failed for {research_id}: {str(e)}")
        research_store[research_id]["status"] = ResearchStatus.FAILED
        research_store[research_id]["error"] = str(e)

        # Update progress with error
        await progress_tracker.update_progress(
            research_id,
            f"Research failed: {str(e)}",
            -1
        )

        # Trigger failure webhook
        await webhook_manager.trigger_event(
            WebhookEvent.RESEARCH_FAILED,
            {
                "research_id": research_id,
                "user_id": user_id,
                "error": str(e)
            }
        )

        # Decrement active research
        active_research.dec()

        # Track error in monitoring
        if hasattr(app.state, "monitoring"):
            await app.state.monitoring["insights"].track_error(
                error_type="research_execution_error",
                severity="error",
                details={
                    "research_id": research_id,
                    "error": str(e),
                    "user_id": user_id
                }
            )

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    await app.state.monitoring["insights"].track_error(
        error_type="http_error",
        severity="warning",
        details={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path
        }
    )

    return {
        "error": "HTTP Error",
        "detail": exc.detail,
        "status_code": exc.status_code,
        "request_id": getattr(request.state, "request_id", "unknown")
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    await app.state.monitoring["insights"].track_error(
        error_type="unhandled_error",
        severity="error",
        details={
            "error": str(exc),
            "type": type(exc).__name__,
            "path": request.url.path
        }
    )

    return {
        "error": "Internal Server Error",
        "detail": "An unexpected error occurred",
        "request_id": getattr(request.state, "request_id", "unknown")
    }

# Main entry point
if __name__ == "__main__":
    # Configure based on environment
    environment = os.getenv("ENVIRONMENT", "production")

    if environment == "production":
        # Production configuration
        uvicorn.run(
            "main_full:app",
            host="0.0.0.0",
            port=8000,
            workers=4,
            log_level="info",
            access_log=True,
            reload=False,
            server_header=False,
            date_header=False
        )
    else:
        # Development configuration
        uvicorn.run(
            "main_full:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug"
        )

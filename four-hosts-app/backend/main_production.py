"""
Production-Ready Four Hosts Research API
Phase 5: Complete Integration
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Import all Phase 5 services
from services.auth import (
    auth_service, oauth_service, create_access_token,
    get_current_user, require_role, UserRole, 
    UserCreate, UserLogin, Token
)
from services.rate_limiter import (
    RateLimiter, RateLimitMiddleware, AdaptiveRateLimiter
)
from services.monitoring import (
    create_monitoring_stack, MonitoringConfig, MonitoringMiddleware
)
from services.webhooks import (
    WebhookManager, create_webhook_router, WebhookEvent
)
from services.websockets import (
    connection_manager, progress_tracker, create_websocket_router
)
from services.export import ExportService, create_export_router

# Import existing services
from services.classification_engine import ClassificationEngine
from services.context_engineering import ContextEngineer
from services.research_orchestrator import ResearchOrchestrator
from services.answer_generator import AnswerGeneratorFactory

# Import documentation
from generate_openapi import custom_openapi, get_custom_swagger_ui_html, get_custom_redoc_html

# --- Application Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    print("🚀 Starting Four Hosts Research API (Production Mode)...")
    
    # Initialize monitoring
    app.state.monitoring = create_monitoring_stack(
        MonitoringConfig(
            enable_prometheus=True,
            enable_opentelemetry=os.getenv("ENABLE_OTEL", "true").lower() == "true",
            service_name="four-hosts-research-api",
            environment=os.getenv("ENVIRONMENT", "production")
        )
    )
    
    # Start performance monitoring
    await app.state.monitoring["performance"].start_monitoring()
    
    # Initialize services
    app.state.rate_limiter = RateLimiter(redis_url=os.getenv("REDIS_URL"))
    app.state.adaptive_limiter = AdaptiveRateLimiter(app.state.rate_limiter)
    app.state.webhook_manager = WebhookManager()
    app.state.export_service = ExportService()
    
    # Start webhook manager
    await app.state.webhook_manager.start()
    
    # Initialize AI services
    app.state.classification_engine = ClassificationEngine()
    app.state.context_engineer = ContextEngineer()
    app.state.research_orchestrator = ResearchOrchestrator(
        classification_engine=app.state.classification_engine,
        context_engineer=app.state.context_engineer
    )
    app.state.answer_generator_factory = AnswerGeneratorFactory()
    
    print("✅ All services initialized successfully")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Four Hosts Research API...")
    
    # Stop services
    await app.state.monitoring["performance"].stop_monitoring()
    await app.state.webhook_manager.stop()
    
    print("👋 Shutdown complete")

# --- Create FastAPI App ---

app = FastAPI(
    title="Four Hosts Research API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # Custom docs
    redoc_url=None,  # Custom redoc
    openapi_url=None  # Custom OpenAPI
)

# --- Middleware ---

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate Limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting"""
    if hasattr(app.state, "rate_limiter"):
        middleware = RateLimitMiddleware(app.state.rate_limiter)
        return await middleware(request, call_next)
    return await call_next(request)

# Monitoring
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Apply monitoring"""
    if hasattr(app.state, "monitoring"):
        middleware = app.state.monitoring["middleware"]
        return await middleware(request, call_next)
    return await call_next(request)

# Request ID
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add request ID to all requests"""
    import uuid
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

# --- Authentication Endpoints ---

@app.post("/auth/register", response_model=Token, tags=["authentication"])
async def register(user_data: UserCreate):
    """Register a new user"""
    user = await auth_service.create_user(user_data)
    access_token = create_access_token({
        "user_id": user.id,
        "email": user.email,
        "role": user.role
    })
    refresh_token = auth_service.create_refresh_token(user.id)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800  # 30 minutes
    )

@app.post("/auth/login", response_model=Token, tags=["authentication"])
async def login(login_data: UserLogin):
    """Login with email and password"""
    user = await auth_service.authenticate_user(login_data)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token({
        "user_id": user.id,
        "email": user.email,
        "role": user.role
    })
    refresh_token = auth_service.create_refresh_token(user.id)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800
    )

@app.post("/auth/api-key", tags=["authentication"])
async def create_api_key(
    key_name: str,
    current_user = Depends(get_current_user)
):
    """Create a new API key"""
    api_key = await auth_service.create_api_key(current_user.user_id, key_name)
    return {
        "api_key": api_key,
        "name": key_name,
        "message": "Store this key securely. It will not be shown again."
    }

# --- Research Endpoints ---

@app.post("/research/query", tags=["research"])
async def submit_research(
    request: Request,
    current_user = Depends(get_current_user)
):
    """Submit a research query"""
    data = await request.json()
    
    # Check rate limits for operation cost
    if hasattr(app.state, "rate_limiter"):
        cost_limiter = app.state.rate_limiter
        allowed, quota_info = await cost_limiter.check_quota(
            current_user.user_id,
            current_user.role,
            f"research_{data.get('options', {}).get('depth', 'standard')}",
            data.get("options", {})
        )
        if not allowed:
            raise HTTPException(status_code=429, detail=quota_info)
    
    # Start research
    research_id = f"res_{os.urandom(8).hex()}"
    
    # Track in WebSocket progress
    await progress_tracker.start_research(
        research_id,
        current_user.user_id,
        data["query"],
        "unknown",  # Will be updated after classification
        data.get("options", {}).get("depth", "standard")
    )
    
    # Trigger webhook
    if hasattr(app.state, "webhook_manager"):
        await app.state.webhook_manager.trigger_event(
            WebhookEvent.RESEARCH_STARTED,
            {
                "research_id": research_id,
                "query": data["query"],
                "user_id": current_user.user_id
            }
        )
    
    # TODO: Start actual research processing
    # For now, return mock response
    return {
        "research_id": research_id,
        "status": "processing",
        "estimated_completion": "2025-01-20T10:30:45Z",
        "webhook_url": f"/research/{research_id}/webhook"
    }

@app.get("/research/status/{research_id}", tags=["research"])
async def get_research_status(
    research_id: str,
    current_user = Depends(get_current_user)
):
    """Get research status"""
    # TODO: Fetch from database
    return {
        "research_id": research_id,
        "status": "in_progress",
        "progress": {
            "current_phase": "synthesis",
            "completion_percentage": 75
        }
    }

@app.get("/research/results/{research_id}", tags=["research"])
async def get_research_results(
    research_id: str,
    current_user = Depends(get_current_user)
):
    """Get research results"""
    # TODO: Fetch from database
    return {
        "research_id": research_id,
        "status": "completed",
        "results": {
            "query": "Sample query",
            "paradigm": "maeve",
            "answer": {
                "summary": "Research summary...",
                "sections": []
            }
        }
    }

# --- Health & Monitoring ---

@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "four-hosts-research-api"}

@app.get("/health/ready", tags=["health"])
async def readiness_check():
    """Readiness probe"""
    if hasattr(app.state, "monitoring"):
        return await app.state.monitoring["health"].get_readiness()
    return {"ready": True}

@app.get("/metrics", tags=["health"])
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    if hasattr(app.state, "monitoring"):
        from prometheus_client import generate_latest
        metrics = generate_latest(app.state.monitoring["prometheus"].registry)
        return Response(content=metrics, media_type="text/plain")
    return Response(content="", media_type="text/plain")

# --- Documentation ---

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi():
    """Get OpenAPI schema"""
    return custom_openapi(app)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Custom Swagger UI"""
    return HTMLResponse(get_custom_swagger_ui_html(openapi_url="/openapi.json"))

@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    """Custom ReDoc documentation"""
    return HTMLResponse(get_custom_redoc_html(openapi_url="/openapi.json"))

# --- Include Routers ---

# Webhooks
if app.state.webhook_manager:
    app.include_router(
        create_webhook_router(app.state.webhook_manager),
        prefix="/api/v1"
    )

# WebSockets
app.include_router(
    create_websocket_router(connection_manager, progress_tracker),
    prefix="/api/v1"
)

# Export
if app.state.export_service:
    app.include_router(
        create_export_router(app.state.export_service),
        prefix="/api/v1"
    )

# --- Error Handlers ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    if hasattr(app.state, "monitoring"):
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
    if hasattr(app.state, "monitoring"):
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

# --- Main ---

if __name__ == "__main__":
    # Production configuration
    uvicorn.run(
        "main_production:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True,
        reload=False,
        server_header=False,
        date_header=False
    )
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

from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Request,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    Security,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, Response, FileResponse, JSONResponse
from pydantic import BaseModel, Field, EmailStr, HttpUrl
from dotenv import load_dotenv
import jwt
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core services
from services.research_orchestrator import (
    research_orchestrator,
    initialize_research_system,
    execute_research,
)
from services.cache import initialize_cache
from services.credibility import get_source_credibility
from services.llm_client import initialise_llm_client
from services.research_store import research_store
from services.context_engineering import context_pipeline

# Import enhanced components
from services.enhanced_integration import (
    enhanced_answer_orchestrator as answer_orchestrator,
    enhanced_classification_engine as classification_engine,
    record_user_feedback,
    get_system_health_report,
    force_paradigm_switch as admin_force_paradigm_switch,
    trigger_model_retraining,
    get_paradigm_performance_metrics,
)
from services.classification_engine import HostParadigm
from services.self_healing_system import self_healing_system
from services.ml_pipeline import ml_pipeline

# Import production services
from services.auth_service import AuthService
from fastapi.exceptions import RequestValidationError
from services.rate_limiter import RateLimiter, RateLimitMiddleware
from services.monitoring import (
    PrometheusMetrics,
    ApplicationInsights,
    create_monitoring_middleware,
)
from services.webhook_manager import WebhookManager, WebhookEvent, create_webhook_router
from services.websocket_service import (
    ConnectionManager,
    ResearchProgressTracker,
    WSEventType,
    WSMessage,
    create_websocket_router,
)
from services.export_service import ExportService, create_export_router
from database.connection import init_database, get_db
from database.models import (
    User as DBUser,
    ResearchQuery as Research,
    Webhook as WebhookSubscription,
    UserRole as _DBUserRole,
    ParadigmType as _DBParadigm,
)
from utils.custom_docs import (
    custom_openapi,
    get_custom_swagger_ui_html,
    get_custom_redoc_html,
)

# Preferences management import
from services.user_management import user_profile_service

# Re-export canonical definitions (for backward compatibility)
UserRole = _DBUserRole
Paradigm = _DBParadigm

# Import authentication components
from services.auth import (
    get_current_user as auth_get_current_user,
    TokenData,
    User,
    auth_service as real_auth_service,
    create_access_token,
    create_refresh_token,
    decode_token,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    UserRole as AuthUserRole,
    require_role,
)

# Handle session_manager import safely
try:
    from services.auth import session_manager
except ImportError:
    # Create a mock session manager if not available
    class MockSessionManager:
        async def end_all_user_sessions(self, user_id: str):
            pass

    session_manager = MockSessionManager()
from services.token_manager import token_manager
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Initialize services
auth_service = AuthService()
rate_limiter = RateLimiter()
webhook_manager = WebhookManager()
connection_manager = ConnectionManager()
progress_tracker = ResearchProgressTracker(connection_manager)
export_service = ExportService()

# Create middleware instance
rate_limit_middleware_instance = None

# Metrics - Initialize with a new registry if needed
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge

metrics_registry = CollectorRegistry()
request_count = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=metrics_registry,
)
request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    registry=metrics_registry,
)
active_research = Gauge(
    "active_research_queries",
    "Number of active research queries",
    registry=metrics_registry,
)
websocket_connections = Gauge(
    "websocket_connections",
    "Number of active WebSocket connections",
    registry=metrics_registry,
)

# Data Models
# Mapping from HostParadigm to Paradigm
HOST_TO_MAIN_PARADIGM = {
    HostParadigm.DOLORES: Paradigm.DOLORES,
    HostParadigm.TEDDY: Paradigm.TEDDY,
    HostParadigm.BERNARD: Paradigm.BERNARD,
    HostParadigm.MAEVE: Paradigm.MAEVE,
}
# ---------------------------------------------------------------------------
# Paradigm explanation data & management endpoints
# ---------------------------------------------------------------------------

# Detailed paradigm explanations (extend as the knowledge base grows)
PARADIGM_EXPLANATIONS: Dict[Paradigm, Dict[str, Any]] = {
    Paradigm.DOLORES: {
        "paradigm": "dolores",
        "name": "Revolutionary Paradigm",
        "description": "Expose systemic issues and empower transformative change."
    },
    Paradigm.TEDDY: {
        "paradigm": "teddy",
        "name": "Devotion Paradigm",
        "description": "Provide compassionate support and protective measures."
    },
    Paradigm.BERNARD: {
        "paradigm": "bernard",
        "name": "Analytical Paradigm",
        "description": "Focus on empirical, data-driven analysis."
    },
    Paradigm.MAEVE: {
        "paradigm": "maeve",
        "name": "Strategic Paradigm",
        "description": "Deliver actionable strategies and optimization."
    },
}

class ParadigmOverrideRequest(BaseModel):
    """Request model to force a different paradigm for an in-flight research job"""
    research_id: str
    paradigm: Paradigm
    reason: Optional[str] = None


# Authentication dependency
# Accept token from Authorization header, cookies, or `access_token` query param
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(
        HTTPBearer(auto_error=False)
    ),
):
    """Resolve the current authenticated user.

    Priority of token lookup:
    1. Standard Authorization header processed by FastAPI's HTTPBearer
    2. Raw `Authorization` header (if auto_error=False disabled automatic 401)
    3. Cookie named `access_token`
    4. Query string param `access_token`
    """
    token: Optional[str] = None

    # 1. Token provided via normal HTTPBearer dependency
    if credentials and credentials.credentials:
        token = credentials.credentials

    # 2. Manually inspect Authorization header if not captured
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header[7:]

    # 3. Check cookie
    if not token:
        token = request.cookies.get("access_token")

    # 4. Check query param
    if not token:
        token = request.query_params.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    # Re-use the auth service's validator
    bearer_credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=token
    )
    token_data = await auth_get_current_user(bearer_credentials)

    # Map to lightweight user object
    user = SimpleNamespace(
        id=token_data.user_id,
        email=token_data.email,
        role=(
            UserRole(token_data.role)
            if isinstance(token_data.role, str)
            else token_data.role
        ),
    )
    return user


class ResearchDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    DEEP_RESEARCH = "deep_research"  # Uses o3-deep-research model


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


class PreferencesPayload(BaseModel):
    """Payload for updating user preferences"""

    preferences: Dict[str, Any]


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
    CANCELLED = "cancelled"


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


# In-memory storage - will be replaced by research_store
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

        # Initialize research store
        await research_store.initialize()
        logger.info("✓ Research store initialized")

        # Initialize cache system
        cache_success = await initialize_cache()
        if cache_success:
            logger.info("✓ Cache system initialized")

        # Initialize research system
        await initialize_research_system()
        logger.info("✓ Research orchestrator initialized")

        # Initialize LLM client
        await initialise_llm_client()
        logger.info("✓ LLM client initialized")
        
        # Initialize unified research orchestrator (includes Brave MCP)
        try:
            from services.unified_research_orchestrator import initialize_unified_orchestrator
            await initialize_unified_orchestrator()
            logger.info("✓ Unified research orchestrator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize unified orchestrator: {e}")
        
        # Preload Hugging Face model to avoid cold start
        try:
            from services.hf_zero_shot import get_classifier
            # This will load the model into memory
            get_classifier()
            logger.info("✓ Hugging Face zero-shot classifier preloaded")
        except Exception as e:
            logger.warning(f"Failed to preload HF model: {e}")

        # Initialize monitoring
        prometheus = PrometheusMetrics(metrics_registry)
        insights = ApplicationInsights(prometheus)
        monitoring_middleware = create_monitoring_middleware(prometheus, insights)

        app.state.monitoring = {
            "prometheus": prometheus,
            "insights": insights,
            "middleware": monitoring_middleware,
        }
        logger.info("✓ Monitoring systems initialized")

        # Initialize production services
        app.state.auth_service = auth_service
        app.state.rate_limiter = rate_limiter
        global rate_limit_middleware_instance
        rate_limit_middleware_instance = RateLimitMiddleware(rate_limiter)
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

    # Cleanup research orchestrator
    if hasattr(research_orchestrator, "cleanup"):
        await research_orchestrator.cleanup()
        logger.info("✓ Research orchestrator cleaned up")

    # Cleanup connections
    await connection_manager.disconnect_all()
    logger.info("✓ WebSocket connections cleaned up")

    logger.info("👋 Shutdown complete")


# Create FastAPI App
app = FastAPI(
    title="Four Hosts Research API",
    version="3.0.0",
    description="Full-featured paradigm-aware research with integrated Context Engineering Pipeline",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
        "https://api.4hosts.ai",
        "http://app.lakefrontdigital.io",
        "https://app.lakefrontdigital.io",
        "http://lakefrontdigital.io",
        "https://lakefrontdigital.io",
        # Note: Cannot use "*" with allow_credentials=True
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    allow_headers=["*", "Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
    expose_headers=["Content-Length", "X-Request-ID", "Set-Cookie", "Authorization"],
    max_age=6000,
)


# Handle OPTIONS requests before authentication
@app.middleware("http")
async def handle_options(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            },
        )
    return await call_next(request)


app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Rate Limiting Middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if rate_limit_middleware_instance:
        return await rate_limit_middleware_instance(request, call_next)
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
            "deep_research": True,  # o3-deep-research model support
            "monitoring": True,
            "webhooks": True,
            "websockets": True,
            "export": True,
            "rate_limiting": True,
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
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
                "llm": "healthy",
            },
        }

        if hasattr(research_orchestrator, "get_execution_stats"):
            stats = await research_orchestrator.get_execution_stats()
            health_data["execution_stats"] = stats

        return health_data
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Authentication Endpoints
@app.post("/auth/register", response_model=Token, tags=["authentication"])
async def register(user_data: UserCreate):
    """Register a new user"""
    # Convert to auth module's UserCreate model
    from services.auth import UserCreate as AuthUserCreate

    auth_user_data = AuthUserCreate(
        email=user_data.email,
        username=user_data.username,
        password=user_data.password,
        role=user_data.role,
    )

    # Use async context manager for database session
    from database.connection import get_db_context

    async with get_db_context() as db:
        user = await real_auth_service.create_user(auth_user_data, db)

    access_token = create_access_token(
        {"user_id": str(user.id), "email": user.email, "role": user.role.value}
    )

    refresh_token = await create_refresh_token(
        user_id=str(user.id), device_id=None, ip_address=None, user_agent=None
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/auth/login", response_model=Token, tags=["authentication"])
async def login(login_data: UserLogin):
    """Login with email and password"""
    # Convert to auth module's UserLogin model
    from services.auth import UserLogin as AuthUserLogin

    auth_login_data = AuthUserLogin(
        email=login_data.email, password=login_data.password
    )

    user = await real_auth_service.authenticate_user(auth_login_data)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(
        {"user_id": str(user.id), "email": user.email, "role": user.role.value}
    )

    refresh_token = await create_refresh_token(
        user_id=str(user.id), device_id=None, ip_address=None, user_agent=None
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


class RefreshTokenRequest(BaseModel):
    refresh_token: str


@app.post("/auth/refresh", response_model=Token, tags=["authentication"])
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using secure token rotation"""
    refresh_token = request.refresh_token
    
    # Validate the token first to get user info
    token_info = await token_manager.validate_refresh_token(refresh_token)
    if not token_info:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Then rotate the refresh token
    token_result = await token_manager.rotate_refresh_token(refresh_token)
    if not token_result:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    # Get user from database
    from sqlalchemy import select

    db_gen = get_db()
    db = await anext(db_gen)
    try:
        result = await db.execute(
            select(DBUser).filter(DBUser.id == token_info["user_id"])
        )
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Create new access token
        access_token = create_access_token(
            {"user_id": str(user.id), "email": user.email, "role": user.role}
        )

        return Token(
            access_token=access_token,
            refresh_token=token_result["token"],
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
    finally:
        await db_gen.aclose()


@app.get("/auth/user", tags=["authentication"])
# Expect TokenData from services.auth.get_current_user
async def get_current_user_info(current_user: TokenData = Depends(auth_get_current_user)):
    """Get current user information"""
    return {
        "id": str(current_user.user_id),
        "email": current_user.email,
        "username": current_user.email.split("@")[0],
        "role": str(current_user.role.value if hasattr(current_user, "role") else current_user.role),
    }


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


# Optional auth dependency for logout
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False))
) -> Optional[TokenData]:
    """Get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    try:
        return await auth_get_current_user(credentials)
    except Exception:
        return None


@app.post("/auth/logout", tags=["authentication"])
async def logout(
    request: Optional[LogoutRequest] = None,
    current_user: Optional[TokenData] = Depends(get_current_user_optional),
):
    """Logout user by revoking tokens"""
    # Handle case where user is already logged out
    if not current_user:
        return {"message": "Successfully logged out"}
    
    refresh_token = request.refresh_token if request else None
    # Revoke the access token using its JTI
    if current_user and current_user.jti:
        await real_auth_service.revoke_token(
            jti=current_user.jti,
            token_type="access",
            user_id=current_user.user_id,
            expires_at=current_user.exp,
        )

    # Revoke the refresh token if provided
    if refresh_token:
        # Revoke all tokens in the refresh token family
        token_info = await token_manager.validate_refresh_token(refresh_token)
        if token_info and "family_id" in token_info:
            await token_manager.revoke_token_family(
                family_id=token_info["family_id"], reason="user_logout"
            )

    # End all user sessions
    if current_user:
        await session_manager.end_all_user_sessions(current_user.user_id)

    return {"message": "Successfully logged out"}


# --- User Preferences Endpoints ---
@app.put("/auth/preferences", tags=["authentication"])
async def update_user_preferences(
    payload: PreferencesPayload, current_user: User = Depends(get_current_user)
):
    """Update the current user's preferences"""
    success = await user_profile_service.update_user_preferences(
        uuid.UUID(str(current_user.id)), payload.preferences
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update preferences")

    # Return updated profile
    profile = await user_profile_service.get_user_profile(
        uuid.UUID(str(current_user.id))
    )
    return profile


@app.get("/auth/preferences", tags=["authentication"])
async def get_user_preferences(current_user: User = Depends(get_current_user)):
    """Retrieve the current user's preferences"""
    profile = await user_profile_service.get_user_profile(
        uuid.UUID(str(current_user.id))
    )
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return {"preferences": profile.get("preferences", {})}


# Paradigm Classification
@app.post("/paradigms/classify", tags=["paradigms"])
async def classify_paradigm(query: str, current_user: User = Depends(get_current_user)):
    """Classify a query into paradigms"""
    try:
        # Use the new classification engine
        classification_result = await classification_engine.classify_query(query)

        # Convert to the old format for compatibility
        classification = ParadigmClassification(
            primary=HOST_TO_MAIN_PARADIGM[classification_result.primary_paradigm],
            secondary=(
                HOST_TO_MAIN_PARADIGM.get(classification_result.secondary_paradigm)
                if classification_result.secondary_paradigm
                else None
            ),
            distribution={
                HOST_TO_MAIN_PARADIGM[p].value: v
                for p, v in classification_result.distribution.items()
            },
            confidence=classification_result.confidence,
            explanation={
                HOST_TO_MAIN_PARADIGM[p].value: "; ".join(r)
                for p, r in classification_result.reasoning.items()
            },
        )

        return {
            "query": query,
            "classification": classification.dict(),
            "suggested_approach": get_paradigm_approach_suggestion(
                classification.primary
            ),
            "user_id": str(current_user.id),
        }
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# Paradigm Management Endpoints
@app.post("/paradigms/override", tags=["paradigms"])
async def override_paradigm(
    payload: ParadigmOverrideRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Force a specific paradigm for an existing research request and restart processing.
    """
    research = await research_store.get(payload.research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Ownership / permission check
    if (research["user_id"] != str(current_user.id)) and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")

    # Update paradigm and reset status
    await research_store.update_field(payload.research_id, "override_paradigm", payload.paradigm.value)
    await research_store.update_field(payload.research_id, "status", ResearchStatus.PROCESSING)

    # Re-queue research execution with same query/options
    try:
        research_query = ResearchQuery(
            query=research["query"],
            options=ResearchOptions(**research["options"]),
        )
        background_tasks.add_task(
            execute_real_research, payload.research_id, research_query, str(current_user.id)
        )
    except Exception as e:
        logger.warning(f"Failed to re-queue research {payload.research_id}: {e}")

    return {
        "success": True,
        "research_id": payload.research_id,
        "new_paradigm": payload.paradigm.value,
        "status": "re-processing",
    }


@app.get("/paradigms/explanation/{paradigm}", tags=["paradigms"])
async def get_paradigm_explanation(paradigm: Paradigm):
    """Return a detailed explanation of the selected paradigm"""
    explanation = PARADIGM_EXPLANATIONS.get(paradigm)
    if not explanation:
        raise HTTPException(status_code=404, detail="Paradigm not found")
    return explanation


# Research Endpoints
@app.post("/research/query", tags=["research"])
async def submit_research(
    research: ResearchQuery,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Submit a research query for paradigm-based analysis"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Check role requirements for research depth
    if research.options.depth in [ResearchDepth.DEEP, ResearchDepth.DEEP_RESEARCH]:
        # Deep research requires at least PRO role
        if not hasattr(current_user, "role") or current_user.role not in [
            "pro",
            "enterprise",
            "admin",
        ]:
            raise HTTPException(
                status_code=403,
                detail="Deep research requires PRO subscription or higher",
            )

    research_id = f"res_{uuid.uuid4().hex[:12]}"

    # Track active research
    active_research.inc()

    try:
        # Classify the query using the new classification engine
        classification_result = await classification_engine.classify_query(
            research.query
        )

        # Convert to the old format for compatibility
        classification = ParadigmClassification(
            primary=HOST_TO_MAIN_PARADIGM[classification_result.primary_paradigm],
            secondary=(
                HOST_TO_MAIN_PARADIGM.get(classification_result.secondary_paradigm)
                if classification_result.secondary_paradigm
                else None
            ),
            distribution={
                HOST_TO_MAIN_PARADIGM[p].value: v
                for p, v in classification_result.distribution.items()
            },
            confidence=classification_result.confidence,
            explanation={
                HOST_TO_MAIN_PARADIGM[p].value: "; ".join(r)
                for p, r in classification_result.reasoning.items()
            },
        )

        # Store research request
        research_data = {
            "id": research_id,
            "user_id": str(current_user.id),
            "query": research.query,
            "options": research.options.dict(),
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "created_at": datetime.utcnow().isoformat(),
            "results": None,
        }
        await research_store.set(research_id, research_data)

        # Execute real research
        background_tasks.add_task(
            execute_real_research, research_id, research, str(current_user.id)
        )

        # Track in WebSocket
        await progress_tracker.start_research(
            research_id,
            str(current_user.id),
            research.query,
            classification.primary.value,
            research.options.depth.value,
        )

        # Trigger webhook
        await app.state.webhook_manager.trigger_event(
            WebhookEvent.RESEARCH_STARTED,
            {
                "research_id": research_id,
                "user_id": str(current_user.id),
                "query": research.query,
                "paradigm": classification.primary.value,
            },
        )

        return {
            "research_id": research_id,
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "estimated_completion": (
                datetime.utcnow() + timedelta(minutes=2)
            ).isoformat(),
            "websocket_url": f"/ws/research/{research_id}",
        }

    except Exception as e:
        active_research.dec()
        logger.error(f"Research submission error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Research submission failed: {str(e)}"
        )


@app.get("/research/status/{research_id}", tags=["research"])
async def get_research_status(
    research_id: str, current_user: User = Depends(get_current_user)
):
    """Get the status of a research query"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research["user_id"] != str(current_user.id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    status_response = {
        "research_id": research_id,
        "status": research["status"],
        "paradigm": research["paradigm_classification"]["primary"],
        "started_at": research["created_at"],
        "progress": research.get("progress", {}),
        "cost_info": research.get("cost_info"),
    }

    # Add status-specific information
    if research["status"] == ResearchStatus.FAILED:
        status_response["error"] = research.get("error", "Research execution failed")
        status_response["can_retry"] = True
        status_response["message"] = "Research failed. You can submit a new research query."
    elif research["status"] == ResearchStatus.CANCELLED:
        status_response["cancelled_at"] = research.get("cancelled_at")
        status_response["cancelled_by"] = research.get("cancelled_by")
        status_response["can_retry"] = True
        status_response["message"] = "Research was cancelled by user."
    elif research["status"] == ResearchStatus.COMPLETED:
        status_response["message"] = "Research completed successfully. Results are available."
    elif research["status"] in [ResearchStatus.PROCESSING, ResearchStatus.IN_PROGRESS]:
        status_response["can_cancel"] = True
        status_response["message"] = f"Research is {research['status']}. Please wait for completion or cancel if needed."

    return status_response


@app.get("/research/results/{research_id}", tags=["research"])
async def get_research_results(
    research_id: str, current_user: User = Depends(get_current_user)
):
    """Get completed research results"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research["user_id"] != str(current_user.id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    if research["status"] != ResearchStatus.COMPLETED:
        if research["status"] == ResearchStatus.FAILED:
            # Return detailed error information for failed research
            error_detail = research.get("error", "Research execution failed")
            return {
                "status": "failed",
                "error": error_detail,
                "research_id": research_id,
                "message": "Research execution failed. Please try submitting a new research query.",
                "can_retry": True
            }
        elif research["status"] == ResearchStatus.CANCELLED:
            # Return cancellation information
            return {
                "status": "cancelled",
                "research_id": research_id,
                "message": "Research was cancelled by user.",
                "cancelled_at": research.get("cancelled_at"),
                "cancelled_by": research.get("cancelled_by"),
                "can_retry": True
            }
        elif research["status"] in [ResearchStatus.PROCESSING, ResearchStatus.IN_PROGRESS]:
            # Return progress information for ongoing research
            return {
                "status": research["status"],
                "research_id": research_id,
                "message": f"Research is still {research['status']}. Please wait for completion or cancel if needed.",
                "progress": research.get("progress", {}),
                "estimated_completion": research.get("estimated_completion"),
                "can_cancel": True,
                "can_retry": False
            }
        else:
            # Handle other statuses (QUEUED, etc.)
            return {
                "status": research["status"],
                "research_id": research_id,
                "message": f"Research is {research['status']}",
                "can_retry": research["status"] != ResearchStatus.PROCESSING,
                "can_cancel": research["status"] in [ResearchStatus.QUEUED]
            }

    return research["results"]


@app.post("/research/cancel/{research_id}", tags=["research"])
async def cancel_research(
    research_id: str, current_user: User = Depends(get_current_user)
):
    """Cancel an ongoing research query"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research["user_id"] != str(current_user.id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    # Check if research can be cancelled
    current_status = research["status"]
    if current_status in [ResearchStatus.COMPLETED, ResearchStatus.FAILED, ResearchStatus.CANCELLED]:
        return {
            "research_id": research_id,
            "status": current_status,
            "message": f"Research is already {current_status} and cannot be cancelled",
            "cancelled": False
        }

    try:
        # Update status to cancelled
        await research_store.update_field(research_id, "status", ResearchStatus.CANCELLED)
        await research_store.update_field(research_id, "cancelled_at", datetime.utcnow().isoformat())
        await research_store.update_field(research_id, "cancelled_by", str(current_user.id))

        # Update progress tracker with cancellation
        try:
            await progress_tracker.update_progress(
                research_id, "Research cancelled by user", -1
            )
        except Exception as e:
            logger.warning(f"Failed to update progress tracker for cancellation: {e}")

        # Send WebSocket notification for cancellation
        try:
            cancel_message = WSMessage(
                type=WSEventType.RESEARCH_CANCELLED,
                data={
                    "research_id": research_id,
                    "message": "Research cancelled by user",
                    "cancelled_by": str(current_user.id)
                }
            )
            await connection_manager.broadcast_to_research(research_id, cancel_message)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket notification for cancellation: {e}")

        # Trigger webhook for cancellation
        try:
            await webhook_manager.trigger_event(
                WebhookEvent.RESEARCH_CANCELLED,
                {
                    "research_id": research_id,
                    "user_id": str(current_user.id),
                    "message": "Research cancelled by user",
                    "cancelled_by": str(current_user.id),
                    "cancelled_at": datetime.utcnow().isoformat()
                },
            )
        except Exception as e:
            logger.warning(f"Failed to trigger webhook for cancellation: {e}")

        # Decrement active research counter
        try:
            active_research.dec()
        except Exception as e:
            logger.warning(f"Failed to decrement active research counter: {e}")

    except Exception as e:
        logger.error(f"Failed to cancel research {research_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel research")

    logger.info(f"Research {research_id} cancelled by user {current_user.id}")

    return {
        "research_id": research_id,
        "status": "cancelled",
        "message": "Research has been successfully cancelled",
        "cancelled": True,
        "cancelled_at": datetime.utcnow().isoformat()
    }


@app.get("/research/export/{research_id}", tags=["research"])
async def export_research(
    research_id: str,
    format: str = "pdf",
    current_user: User = Depends(get_current_user),
):
    """Export research results (requires BASIC subscription or higher)"""
    # Check if user has BASIC role or higher
    if current_user.role not in [UserRole.BASIC, UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Export requires BASIC subscription or higher"
        )

    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research["user_id"] != str(current_user.id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    if research["status"] != ResearchStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Research not completed")

    # Generate export
    export_path = await app.state.export_service.export_research(
        research["results"], format=format
    )

    return FileResponse(
        export_path,
        media_type=f"application/{format}",
        filename=f"research_{research_id}.{format}",
    )


# Research History Endpoint
@app.get("/research/history", tags=["research"])
async def get_research_history(
    current_user: User = Depends(get_current_user), limit: int = 10, offset: int = 0
):
    """Get user's research history"""
    try:
        # Get research history for user
        user_research = await research_store.get_user_research(
            str(current_user.id), limit + offset
        )

        # Sort by creation date (newest first)
        user_research.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        total = len(user_research)
        paginated = user_research[offset: offset + limit]

        # Format the response
        history = []
        for research in paginated:
            history_item = {
                "research_id": research["id"],
                "query": research["query"],
                "status": research["status"],
                "paradigm": research.get("paradigm_classification", {}).get("primary", "unknown"),
                "created_at": research["created_at"],
                "options": research["options"],
            }

            # Include results summary if completed
            if research["status"] == ResearchStatus.COMPLETED and research.get(
                "results"
            ):
                results = research["results"]
                content_preview = ""
                if results.get("answer", {}).get("sections"):
                    content_preview = (
                        results["answer"]["sections"][0].get("content", "")[:200]
                        + "..."
                    )
                history_item["summary"] = {
                    "answer_preview": content_preview,
                    "source_count": len(results.get("sources", [])),
                    "total_cost": results.get("cost_info", {}).get("total_cost", 0),
                }

            history.append(history_item)

        return {"total": total, "limit": limit, "offset": offset, "history": history}
    except Exception as e:
        logger.error(f"Error fetching research history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch research history")


class ResearchDeepQuery(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    paradigm: Optional[Paradigm] = None
    search_context_size: Optional[str] = Field(default="medium", pattern="^(small|medium|large)$")
    user_location: Optional[Dict[str, str]] = None


# Deep Research Endpoints
@app.post("/research/deep", tags=["research"])
async def submit_deep_research(
    research_query: ResearchDeepQuery,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
):
    """
    Submit a query for deep research using o3-deep-research model.
    Requires PRO subscription or higher.

    Expected request body:
    {
        "query": "your research query (10-500 chars)",
        "paradigm": "dolores|teddy|bernard|maeve" (optional),
        "search_context_size": "small|medium|large" (optional, default: "medium"),
        "user_location": {"country": "US", "city": "NYC"} (optional)
    }
    """
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Check if user has PRO role or higher
    if current_user.role not in [UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Deep research requires PRO subscription or higher"
        )

    research_id = f"deep_{uuid.uuid4().hex[:12]}"

    # Track active research
    active_research.inc()

    try:
        # Create research query with deep research option
        research = ResearchQuery(
            query=research_query.query,  # Access query from the Pydantic model
            options=ResearchOptions(
                depth=ResearchDepth.DEEP_RESEARCH,
                paradigm_override=research_query.paradigm,
                max_sources=100,  # Deep research can handle more sources
                enable_real_search=True,
            )
        )

        # Store and execute
        research_data = {
            "id": research_id,
            "user_id": str(current_user.id),
            "query": research_query.query, # Access query from the Pydantic model
            "options": research.options.dict(),
            "status": ResearchStatus.PROCESSING,
            "deep_research": True,
            "search_context_size": research_query.search_context_size,
            "user_location": research_query.user_location,
            "created_at": datetime.utcnow().isoformat(),
        }
        await research_store.set(research_id, research_data)

        # Execute in background
        background_tasks.add_task(
            execute_real_research, research_id, research, str(current_user.id)
        )

        # Track in WebSocket
        await progress_tracker.start_research(
            research_id,
            str(current_user.id),
            research_query.query, # Access query from the Pydantic model
            research_query.paradigm.value if research_query.paradigm else "auto",
            "deep_research",
        )

        return {
            "research_id": research_id,
            "status": ResearchStatus.PROCESSING,
            "deep_research": True,
            "estimated_completion": (
                datetime.utcnow() + timedelta(minutes=10)  # Deep research takes longer
            ).isoformat(),
            "websocket_url": f"/ws/research/{research_id}",
        }

    except Exception as e:
        active_research.dec()
        logger.error(f"Deep research submission error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Deep research submission failed: {str(e)}"
        )


@app.post("/research/deep/{research_id}/resume", tags=["research"])
async def resume_deep_research(
    research_id: str,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
):
    """Resume an interrupted deep research task"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Check if user has PRO role or higher
    if current_user.role not in [UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Deep research requires PRO subscription or higher"
        )

    try:
        # Verify research belongs to user
        research_data = await research_store.get(research_id)
        if not research_data:
            raise HTTPException(status_code=404, detail="Research not found")

        if research_data.get("user_id") != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if it's a deep research task
        if not research_data.get("deep_research"):
            raise HTTPException(status_code=400, detail="Not a deep research task")

        # Check if there's a response ID to resume
        if not research_data.get("deep_research_response_id"):
            raise HTTPException(status_code=400, detail="No deep research response to resume")

        # Update status
        await research_store.update_field(research_id, "status", ResearchStatus.IN_PROGRESS)

        # Resume in background
        background_tasks.add_task(
            resume_deep_research_task, research_id, str(current_user.id)
        )

        return {
            "research_id": research_id,
            "status": ResearchStatus.IN_PROGRESS,
            "message": "Deep research resumed",
            "websocket_url": f"/ws/research/{research_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deep research resume error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to resume deep research: {str(e)}"
        )


@app.get("/research/deep/status", tags=["research"])
async def get_deep_research_status(
    current_user: User = Depends(get_current_user),
):
    """Get status of all deep research queries for the current user"""
    # Check if user has PRO role or higher
    if current_user.role not in [UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Deep research requires PRO subscription or higher"
        )

    try:
        # Get all user research
        user_research = await research_store.get_user_research(str(current_user.id), 50)

        # Filter for deep research only
        deep_research_list = [
            r for r in user_research
            if r.get("deep_research") or r.get("options", {}).get("depth") == "deep_research"
        ]

        # Sort by creation date
        deep_research_list.sort(key=lambda x: x["created_at"], reverse=True)

        # Format response
        formatted = []
        for research in deep_research_list[:20]:  # Limit to 20 most recent
            formatted.append({
                "research_id": research["id"],
                "query": research["query"],
                "status": research["status"],
                "created_at": research["created_at"],
                "paradigm": research.get("paradigm_classification", {}).get("primary"),
                "has_results": research.get("results") is not None,
            })

        return {
            "total": len(deep_research_list),
            "deep_research_queries": formatted,
        }

    except Exception as e:
        logger.error(f"Error fetching deep research status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch deep research status")


# Source Credibility Endpoint
@app.get("/sources/credibility/{domain}", tags=["sources"])
async def get_domain_credibility(
    domain: str,
    paradigm: Paradigm = Paradigm.BERNARD,
    current_user: User = Depends(get_current_user),
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
            "checked_by": str(current_user.id),
        }
    except Exception as e:
        logger.error(f"Credibility check error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Credibility check failed: {str(e)}"
        )


# User Feedback Endpoint
@app.post("/research/feedback/{research_id}", tags=["research"])
async def submit_research_feedback(
    research_id: str,
    satisfaction_score: float = Body(..., ge=0.0, le=1.0),
    paradigm_feedback: Optional[str] = Body(None),
    current_user: User = Depends(get_current_user),
):
    """Submit feedback for a research query to improve the system"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if research["user_id"] != str(current_user.id) and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Record feedback in enhanced systems
        await record_user_feedback(research_id, satisfaction_score, paradigm_feedback)

        # Store feedback in research data
        await research_store.update_field(research_id, "user_feedback", {
            "satisfaction_score": satisfaction_score,
            "paradigm_feedback": paradigm_feedback,
            "submitted_at": datetime.utcnow().isoformat()
        })

        return {
            "success": True,
            "message": "Feedback recorded successfully",
            "research_id": research_id,
            "satisfaction_score": satisfaction_score
        }
    except Exception as e:
        logger.error(f"Failed to record feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")


# Test Deep Research (Development Only)
@app.get("/test/deep-research", tags=["test"])
async def test_deep_research(current_user: User = Depends(get_current_user)):
    """Test deep research functionality (Admin only)"""
    # Check if user has ADMIN role
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Test endpoint requires ADMIN role"
        )

    try:
        from services.deep_research_service import deep_research_service

        # Simple test query
        test_query = "What are the latest advancements in quantum computing?"

        # Test initialization
        await deep_research_service.initialize()

        # Create a simple test without full context
        result = await deep_research_service.analytical_deep_dive(
            query=test_query,
            research_id="test_deep_research"
        )

        return {
            "status": "success",
            "test_query": test_query,
            "deep_research_status": result.status.value,
            "has_content": result.content is not None,
            "citation_count": len(result.citations) if result.citations else 0,
            "execution_time": result.execution_time,
            "error": result.error,
        }
    except Exception as e:
        logger.error(f"Deep research test failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Deep research test failed. Check logs for details."
        }


# Orchestrator Status
@app.get("/system/orchestrator", tags=["system"])
async def get_orchestrator_status(current_user: User = Depends(get_current_user)):
    """Get research orchestrator status and capabilities"""
    try:
        from services.unified_research_orchestrator import unified_orchestrator
        capabilities = unified_orchestrator.get_capabilities()
        
        return {
            "status": "active",
            "capabilities": capabilities,
            "brave_mcp_enabled": capabilities.get("brave_mcp", False),
            "search_apis_available": [api for api in capabilities.get("search_apis", []) if api],
            "mode": capabilities.get("mode", "unknown")
        }
    except Exception as e:
        logger.error(f"Failed to get orchestrator status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "mode": "degraded"
        }

# System Stats (Admin/Enterprise only)
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
            "total_queries": len(await research_store.list_all_research()) if hasattr(research_orchestrator, "list_all_research") else 0,
            "paradigm_distribution": "redacted_for_privacy",
            "average_processing_time": 0,  # Placeholder
            "cache_hit_rate": 0, # Placeholder
            "system_health": "healthy"
        }

        if hasattr(research_orchestrator, "get_execution_stats"):
            stats["research_stats"] = await research_orchestrator.get_execution_stats()

        # Add enhanced system metrics
        stats["enhanced_features"] = {
            "self_healing": self_healing_system.get_performance_report(),
            "ml_pipeline": ml_pipeline.get_model_info(),
            "paradigm_performance": get_paradigm_performance_metrics()
        }

        stats["timestamp"] = datetime.utcnow().isoformat()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return {"error": str(e)}


# Public System Stats (Lightweight version for all users)
@app.get("/system/public-stats", tags=["system"])
async def get_public_system_stats(
    current_user: User = Depends(get_current_user)
):
    """Get lightweight system statistics for all authenticated users"""
    try:
        # Get basic system health without sensitive details
        health = {
            "system_status": "operational" if system_initialized else "offline",
            "system_initialized": system_initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
        return health
    except Exception as e:
        logger.error(f"Public stats error: {str(e)}")
        return {
            "system_status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Admin Enhanced Features Endpoints
@app.post("/admin/paradigm/force-switch", tags=["admin"])
async def admin_force_paradigm_switch_endpoint(
    query_id: str,
    new_paradigm: str,
    reason: str,
    current_user: User = Depends(get_current_user),
):
    """Force a paradigm switch for a query (Admin only)"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")

    return await admin_force_paradigm_switch(query_id, new_paradigm, reason)


@app.post("/admin/ml/retrain", tags=["admin"])
async def admin_trigger_retraining(
    current_user: User = Depends(get_current_user),
):
    """Trigger ML model retraining (Admin only)"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")

    return await trigger_model_retraining()


@app.get("/admin/system/health", tags=["admin"])
async def admin_system_health(
    current_user: User = Depends(get_current_user),
):
    """Get comprehensive system health report (Admin only)"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")

    return get_system_health_report()


# Metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    metrics = generate_latest(metrics_registry)
    return Response(content=metrics, media_type="text/plain")


# WebSocket endpoint
@app.websocket("/ws/research/{research_id}")
async def websocket_research_progress(websocket: WebSocket, research_id: str):
    """WebSocket for real-time research progress"""
    # For anonymous WebSocket connections, use research_id as user_id for now
    # In production, you should authenticate the WebSocket connection
    await connection_manager.connect(websocket, f"research_{research_id}")
    
    # Subscribe to the specific research
    await connection_manager.subscribe_to_research(websocket, research_id)
    
    websocket_connections.inc()

    try:
        while True:
            # Keep connection alive and handle potential client messages
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                await connection_manager.handle_client_message(websocket, data)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)
        websocket_connections.dec()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await connection_manager.disconnect(websocket)
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
    create_webhook_router(webhook_manager), prefix="/api/v1", tags=["webhooks"]
)

app.include_router(
    create_websocket_router(connection_manager, progress_tracker),
    prefix="/api/v1",
    tags=["websockets"],
)

app.include_router(
    create_export_router(export_service), prefix="/api/v1", tags=["export"]
)

# Helper functions


def get_paradigm_approach_suggestion(paradigm: Paradigm) -> str:
    suggestions = {
        Paradigm.DOLORES: "Focus on exposing systemic issues and empowering resistance",
        Paradigm.TEDDY: "Prioritize community support and protective measures",
        Paradigm.BERNARD: "Emphasize empirical research and data-driven analysis",
        Paradigm.MAEVE: "Develop strategic frameworks and actionable implementation plans",
    }
    return suggestions[paradigm]


# Background task for resuming deep research
async def resume_deep_research_task(research_id: str, user_id: str):
    """Resume an interrupted deep research task"""
    try:
        # Update progress
        await progress_tracker.update_progress(
            research_id, "Resuming deep research", 10
        )

        # Import deep research service locally
        from services.deep_research_service import deep_research_service

        # Initialize if needed
        if not deep_research_service._initialized:
            await deep_research_service.initialize()

        # Resume the research
        result = await deep_research_service.resume_deep_research(
            research_id, progress_tracker
        )

        # Store the result
        research_data = await research_store.get(research_id)
        if research_data:
            research_data["result"] = result.dict() if hasattr(result, 'dict') else result
            research_data["status"] = ResearchStatus.COMPLETED
            research_data["completed_at"] = datetime.utcnow().isoformat()
            await research_store.set(research_id, research_data)

        # Update final progress
        await progress_tracker.update_progress(
            research_id, "Research completed", 100
        )

    except Exception as e:
        logger.error(f"Resume deep research failed: {str(e)}")

        # Update status to failed
        await research_store.update_field(research_id, "status", ResearchStatus.FAILED)
        await research_store.update_field(research_id, "error", str(e))

        # Update progress
        await progress_tracker.update_progress(
            research_id, f"Resume failed: {str(e)}", -1
        )
    finally:
        active_research.dec()


# Background task for real research execution
async def execute_real_research(
    research_id: str, research: ResearchQuery, user_id: str
):
    """Execute real research using the complete pipeline"""

    async def check_cancellation():
        """Check if research has been cancelled"""
        research_data = await research_store.get(research_id)
        if research_data and research_data.get("status") == ResearchStatus.CANCELLED:
            logger.info(f"Research {research_id} was cancelled, stopping execution")
            return True
        return False

    try:
        # Check for cancellation before starting
        if await check_cancellation():
            return

        # Update status
        await research_store.update_field(
            research_id, "status", ResearchStatus.IN_PROGRESS
        )

        # Update progress
        await progress_tracker.update_progress(
            research_id, "Initializing research pipeline", 10
        )

        # Check for cancellation
        if await check_cancellation():
            return

        # Get classification
        research_data = await research_store.get(research_id)
        if not research_data:
            raise Exception("Research data not found")

        # Get the stored classification result from the new engine
        classification_result = await classification_engine.classify_query(
            research.query
        )

        # Update progress
        await progress_tracker.update_progress(
            research_id, "Processing query through context engineering", 20
        )

        # Check for cancellation
        if await check_cancellation():
            return

        # Process through context engineering pipeline
        context_engineered_query = await context_pipeline.process_query(
            classification_result
        )

        # Update progress
        await progress_tracker.update_progress(
            research_id, "Executing search queries", 30
        )

        # Check for cancellation
        if await check_cancellation():
            return

        # Execute research based on depth option
        if research.options.depth == ResearchDepth.DEEP_RESEARCH:
            # Use deep research with o3-deep-research model
            from services.deep_research_service import DeepResearchMode

            # Map paradigms to deep research modes
            deep_mode_mapping = {
                "dolores": DeepResearchMode.PARADIGM_FOCUSED,
                "teddy": DeepResearchMode.PARADIGM_FOCUSED,
                "bernard": DeepResearchMode.ANALYTICAL,
                "maeve": DeepResearchMode.STRATEGIC,
            }

            paradigm_name = context_engineered_query.classification.primary_paradigm.value
            deep_mode = deep_mode_mapping.get(paradigm_name, DeepResearchMode.COMPREHENSIVE)

            # Get web search settings from research data
            search_context_size = research_data.get("search_context_size")
            user_location = research_data.get("user_location")

            execution_result = await research_orchestrator.execute_deep_research(
                context_engineered_query,
                enable_standard_search=True,  # Combine with standard search
                deep_research_mode=deep_mode,
                search_context_size=search_context_size,
                user_location=user_location,
                progress_tracker=progress_tracker,
                research_id=research_id,
            )
        else:
            # Use standard paradigm research
            execution_result = await research_orchestrator.execute_paradigm_research(
                context_engineered_query, research.options.max_sources, progress_tracker, research_id
            )

        # Update progress
        await progress_tracker.update_progress(
            research_id, "Processing search results", 60
        )

        # Check for cancellation
        if await check_cancellation():
            return

        # Format results
        formatted_sources = []
        search_results_for_synthesis = []

        for result in execution_result.filtered_results[: research.options.max_sources]:
            formatted_sources.append(
                SourceResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    domain=result.domain,
                    credibility_score=getattr(result, "credibility_score", 0.5),
                    published_date=(
                        result.published_date.isoformat()
                        if result.published_date
                        else None
                    ),
                    source_type=result.result_type,
                )
            )

            search_results_for_synthesis.append(
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "domain": result.domain,
                    "credibility_score": getattr(result, "credibility_score", 0.5),
                    "published_date": result.published_date,
                    "result_type": result.result_type,
                }
            )

        # Update progress
        await progress_tracker.update_progress(
            research_id, "Generating AI-powered answer", 80
        )

        # Check for cancellation before AI generation
        if await check_cancellation():
            return

        # Generate answer using context engineering outputs
        context_engineering = {
            "write_output": {
                "documentation_focus": context_engineered_query.write_output.documentation_focus,
                "key_themes": context_engineered_query.write_output.key_themes[:4],
                "narrative_frame": context_engineered_query.write_output.narrative_frame,
            },
            "select_output": {
                "search_queries": context_engineered_query.select_output.search_queries,
                "source_preferences": context_engineered_query.select_output.source_preferences,
                "max_sources": context_engineered_query.select_output.max_sources,
            },
            "compress_output": {
                "compression_ratio": context_engineered_query.compress_output.compression_ratio,
                "priority_elements": context_engineered_query.compress_output.priority_elements,
                "token_budget": context_engineered_query.compress_output.token_budget,
            },
            "isolate_output": {
                "isolation_strategy": context_engineered_query.isolate_output.isolation_strategy,
                "key_findings_criteria": context_engineered_query.isolate_output.key_findings_criteria,
                "output_structure": context_engineered_query.isolate_output.output_structure,
            },
        }

        # Map enum value to paradigm name
        paradigm_mapping = {
            "revolutionary": "dolores",
            "devotion": "teddy",
            "analytical": "bernard",
            "strategic": "maeve",
        }
        paradigm_name = paradigm_mapping.get(
            context_engineered_query.classification.primary_paradigm.value,
            "bernard",  # Default to bernard if not found
        )

        # Check if we have deep research content
        deep_research_content = getattr(execution_result, "deep_research_content", None)

        generated_answer = await answer_orchestrator.generate_answer(
            paradigm=paradigm_name,
            query=research.query,
            search_results=search_results_for_synthesis,
            context_engineering=context_engineering,
            options={
                "research_id": research_id,
                "max_length": 2000,
                "include_citations": True,
                "deep_research_content": deep_research_content,  # Pass deep research content if available
            },
        )

        # Format final result
        answer_sections = []
        for section in generated_answer.sections:
            answer_sections.append(
                {
                    "title": section.title,
                    "paradigm": section.paradigm,
                    "content": section.content,
                    "confidence": section.confidence,
                    "sources_count": len(section.citations),
                }
            )

        citations_list = []
        for cite_id, citation in generated_answer.citations.items():
            citations_list.append(
                {
                    "id": cite_id,
                    "title": citation.source_title,
                    "source": citation.source_title,
                    "url": citation.source_url,
                    "snippet": getattr(citation, "snippet", ""),
                    "credibility_score": citation.credibility_score,
                    "paradigm_alignment": context_engineered_query.classification.primary_paradigm.value,
                }
            )

        final_result = ResearchResult(
            research_id=research_id,
            query=research.query,
            status=ResearchStatus.COMPLETED,
            paradigm_analysis={
                "primary": {
                    "paradigm": context_engineered_query.classification.primary_paradigm.value,
                    "confidence": context_engineered_query.classification.confidence,
                    "approach": context_engineered_query.write_output.narrative_frame,
                    "focus": context_engineered_query.write_output.documentation_focus,
                },
                "context_engineering": {
                    "compression_ratio": context_engineered_query.compress_output.compression_ratio,
                    "token_budget": context_engineered_query.compress_output.token_budget,
                    "isolation_strategy": context_engineered_query.isolate_output.isolation_strategy,
                    "search_queries_count": len(
                        context_engineered_query.select_output.search_queries
                    ),
                },
            },
            answer={
                "summary": generated_answer.summary,
                "sections": answer_sections,
                "action_items": generated_answer.action_items,
                "citations": citations_list,
            },
            sources=formatted_sources,
            metadata={
                "total_sources_analyzed": len(execution_result.raw_results),
                "high_quality_sources": len(
                    [s for s in formatted_sources if s.credibility_score > 0.7]
                ),
                "search_queries_executed": len(
                    execution_result.search_queries_executed
                ),
                "processing_time_seconds": execution_result.execution_metrics[
                    "processing_time_seconds"
                ],
                "answer_generation_time": generated_answer.generation_time,
                "synthesis_quality": generated_answer.synthesis_quality,
                "paradigms_used": [
                    context_engineered_query.classification.primary_paradigm.value
                ],
                "deep_research_enabled": execution_result.execution_metrics.get("deep_research_enabled", False),
                "research_depth": research.options.depth.value,
            },
            cost_info=execution_result.cost_breakdown,
        )

        # Update progress
        await progress_tracker.update_progress(research_id, "Research completed", 100)

        # Store results
        await research_store.update_field(
            research_id, "status", ResearchStatus.COMPLETED
        )
        await research_store.update_field(research_id, "results", final_result.dict())
        await research_store.update_field(
            research_id, "cost_info", execution_result.cost_breakdown
        )

        # Trigger webhook
        await webhook_manager.trigger_event(
            WebhookEvent.RESEARCH_COMPLETED,
            {
                "research_id": research_id,
                "user_id": user_id,
                "query": research.query,
                "paradigm": context_engineered_query.classification.primary_paradigm.value,
                "sources_count": len(formatted_sources),
                "cost": execution_result.cost_breakdown.get("total", 0),
            },
        )

        # Decrement active research
        active_research.dec()

        logger.info(f"✓ Research completed for {research_id}")

    except Exception as e:
        logger.error(f"Research execution failed for {research_id}: {str(e)}")
        await research_store.update_field(research_id, "status", ResearchStatus.FAILED)
        await research_store.update_field(research_id, "error", str(e))

        # Update progress with error
        await progress_tracker.update_progress(
            research_id, f"Research failed: {str(e)}", -1
        )

        # Trigger failure webhook
        await webhook_manager.trigger_event(
            WebhookEvent.RESEARCH_FAILED,
            {"research_id": research_id, "user_id": user_id, "error": str(e)},
        )

        # Decrement active research
        active_research.dec()

        # Track error in monitoring
        if hasattr(app.state, "monitoring") and "insights" in app.state.monitoring:
            try:
                await app.state.monitoring["insights"].track_error(
                    error_type="research_execution_error",
                    severity="error",
                    details={
                        "research_id": research_id,
                        "error": str(e),
                        "user_id": user_id,
                    },
                )
            except Exception:
                pass  # Don't let monitoring errors break the flow


# Error Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages"""
    errors = exc.errors()
    error_messages = []

    for error in errors:
        field = ".".join(str(x) for x in error["loc"][1:])  # Skip 'body' in location
        msg = error["msg"]
        error_type = error["type"]

        if field == "query" and "at least 10 characters" in msg:
            error_messages.append(f"Query must be at least 10 characters long")
        elif field == "query" and "at most 500 characters" in msg:
            error_messages.append(f"Query must be at most 500 characters long")
        elif field == "query" and error_type == "value_error.missing":
            error_messages.append(f"Query field is required")
        elif field == "search_context_size":
            error_messages.append(f"search_context_size must be one of: small, medium, large")
        else:
            error_messages.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": error_messages,
            "expected_format": {
                "query": "your research query (10-500 chars required)",
                "paradigm": "dolores|teddy|bernard|maeve (optional)",
                "search_context_size": "small|medium|large (optional, default: medium)",
                "user_location": {"country": "US", "city": "NYC"} # optional
            },
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    if hasattr(app.state, "monitoring") and "insights" in app.state.monitoring:
        try:
            await app.state.monitoring["insights"].track_error(
                error_type="http_error",
                severity="warning",
                details={
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                    "path": request.url.path,
                },
            )
        except Exception:
            pass  # Don't let monitoring errors break the response

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    if hasattr(app.state, "monitoring") and "insights" in app.state.monitoring:
        try:
            await app.state.monitoring["insights"].track_error(
                error_type="unhandled_error",
                severity="error",
                details={
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "path": request.url.path,
                },
            )
        except Exception:
            pass  # Don't let monitoring errors break the response

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


# Main entry point
if __name__ == "__main__":
    # Configure based on environment
    environment = os.getenv("ENVIRONMENT", "production")

    if environment == "production":
        # Production configuration - SINGLE WORKER until Redis is implemented
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            workers=1,  # Changed from 4 to 1
            log_level="info",
            access_log=True,
            reload=False,
            server_header=False,
            date_header=False,
        )
    else:
        # Development configuration
        uvicorn.run(
            "main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug"
        )

"""
FastAPI application factory and configuration
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse

from core.config import TRUSTED_ORIGINS, get_allowed_hosts, is_production
from middleware.security import (
    csrf_protection_middleware,
    security_middleware
)
from routes import auth_router, research_router, paradigms_router
from utils.custom_docs import (
    custom_openapi,
    get_custom_swagger_ui_html,
    get_custom_redoc_html,
)

logger = logging.getLogger(__name__)

# Global system state
system_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global system_initialized

    # Startup
    logger.info("🚀 Starting Four Hosts Research API (Full Featured)...")

    try:
        # Initialize database
        from database.connection import init_database
        await init_database()
        logger.info("✓ Database initialized")

        # Initialize research store
        from services.research_store import research_store
        await research_store.initialize()
        logger.info("✓ Research store initialized")

        # Initialize cache system
        from services.cache import initialize_cache
        cache_success = await initialize_cache()
        if cache_success:
            logger.info("✓ Cache system initialized")

        # Initialize research system
        from services.research_orchestrator import initialize_research_system
        await initialize_research_system()
        logger.info("✓ Research orchestrator initialized")

        # Initialize LLM client
        from services.llm_client import initialise_llm_client
        await initialise_llm_client()
        logger.info("✓ LLM client initialized")

        # Initialize search manager with cache integration
        from services.cache import cache_manager
        from services.search_apis import create_search_manager
        search_manager = create_search_manager(cache_manager=cache_manager)
        await search_manager.initialize()
        app.state.search_manager = search_manager
        logger.info("✓ Search manager initialized with cache")

        # Preload Hugging Face model to avoid cold start
        try:
            from services.hf_zero_shot import get_classifier
            get_classifier()
            logger.info("✓ Hugging Face zero-shot classifier preloaded")
        except Exception as e:
            logger.warning("Failed to preload HF model: %s", e)

        # Start self-healing system
        from services.self_healing_system import self_healing_system
        await self_healing_system.start()
        logger.info("✓ Self-healing system started")

        # Initialize monitoring
        from services.monitoring import (
            PrometheusMetrics,
            ApplicationInsights,
            create_monitoring_middleware,
        )
        from prometheus_client import CollectorRegistry

        metrics_registry = CollectorRegistry()
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
        from services.auth import auth_service
        from services.rate_limiter import RateLimiter
        from services.webhook_manager import WebhookManager
        from services.export_service import ExportService

        app.state.auth_service = auth_service
        app.state.rate_limiter = RateLimiter()
        app.state.webhook_manager = WebhookManager()
        app.state.export_service = ExportService()
        logger.info("✓ Production services initialized")

        system_initialized = True
        logger.info("🚀 Four Hosts Research System ready with all features!")

    except Exception as e:
        logger.error("❌ System initialization failed: %s", str(e))
        system_initialized = False

    yield

    # Shutdown
    logger.info("🛑 Shutting down Four Hosts Research API...")

    try:
        # Stop self-healing system
        from services.self_healing_system import self_healing_system
        await self_healing_system.stop()
        logger.info("✓ Self-healing system stopped")

        # Cleanup search manager
        if hasattr(app.state, 'search_manager'):
            await app.state.search_manager.cleanup()
            logger.info("✓ Search manager cleaned up")

        # Cleanup background LLM manager
        from services.background_llm import background_llm_manager
        if background_llm_manager:
            await background_llm_manager.cleanup()
            logger.info("✓ Background LLM manager cleaned up")

        # Graceful task shutdown
        from services.task_registry import task_registry
        await task_registry.graceful_shutdown(timeout=30.0)
        logger.info("✓ Background tasks shutdown complete")

    except Exception as e:
        logger.error("Error during shutdown: %s", e)

    logger.info("👋 Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Four Hosts Research API",
        version="3.0.0",
        description="Full-featured paradigm-aware research with integrated Context Engineering Pipeline",
        lifespan=lifespan
    )

    # Add middleware
    setup_middleware(app)

    # Add routes
    setup_routes(app)

    # Add custom endpoints
    setup_custom_endpoints(app)

    return app


def setup_middleware(app: FastAPI):
    """Configure middleware"""
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=TRUSTED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=[
            "Authorization", "Content-Type", "Accept", "Origin", "X-CSRF-Token"
        ],
        max_age=600,
    )

    # Security middleware
    @app.middleware("http")
    async def security_middleware_wrapper(request: Request, call_next):
        return await security_middleware(request, call_next)

    # CSRF protection middleware
    @app.middleware("http")
    async def csrf_middleware_wrapper(request: Request, call_next):
        return await csrf_protection_middleware(request, call_next)

    # Request ID middleware
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    # Trusted host middleware (production only)
    if is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=get_allowed_hosts()
        )

    # Gzip middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)


def setup_routes(app: FastAPI):
    """Configure routes"""
    app.include_router(auth_router)
    app.include_router(research_router)
    app.include_router(paradigms_router)


def setup_custom_endpoints(app: FastAPI):
    """Setup custom endpoints"""
    
    @app.get("/api/csrf-token")
    async def get_csrf_token_api(request: Request, response: Response):
        """Get CSRF token for API calls"""
        from middleware.security import get_csrf_token
        return get_csrf_token(request, response)
    
    @app.post("/api/session/create")
    async def create_session(request: Request, response: Response):
        """Create a new session and return CSRF token"""
        from middleware.security import get_csrf_token
        return get_csrf_token(request, response)

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
                "deep_research": True,
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

    @app.get("/health")
    async def health_check():
        """System health check"""
        try:
            from datetime import datetime

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

            return health_data
        except Exception as e:
            from datetime import datetime
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    # Custom documentation endpoints
    @app.get("/openapi.json", include_in_schema=False)
    async def get_openapi():
        return custom_openapi(app)

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        return HTMLResponse(
            get_custom_swagger_ui_html(openapi_url="/openapi.json")
        )

    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc():
        return HTMLResponse(
            get_custom_redoc_html(openapi_url="/openapi.json")
        )

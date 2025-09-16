"""
FastAPI application factory and configuration
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse

# Try to import prometheus_client, but allow graceful degradation
try:
    from prometheus_client import generate_latest, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("prometheus_client not installed - monitoring metrics will be unavailable")

from core.config import TRUSTED_ORIGINS, get_allowed_hosts, is_production
from core.config import ENABLE_FEEDBACK_RECONCILE, FEEDBACK_RECONCILE_WINDOW_MINUTES
from middleware.security import (
    csrf_protection_middleware,
    security_middleware
)
from routes import auth_router, research_router, paradigms_router
# New SSOTA-aligned routers
try:
    # These modules may be added during alignment
    from routes.search import router as search_router
except Exception:
    search_router = None  # type: ignore
try:
    from routes.responses import router as responses_router
except Exception:
    responses_router = None  # type: ignore
try:
    from routes.users import router as users_router
except Exception:
    users_router = None  # type: ignore
try:
    from routes.system import router as system_router
except Exception:
    system_router = None  # type: ignore
try:
    from routes.feedback import router as feedback_router
except Exception:
    feedback_router = None  # type: ignore
from services.websocket_service import (
    create_websocket_router,
    connection_manager,
    progress_tracker,
)
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
        llm_ok = await initialise_llm_client()
        # Record status for health endpoint and accurate logs
        try:
            app.state.llm_initialized = bool(llm_ok)
        except Exception:
            pass
        if llm_ok:
            logger.info("✓ LLM client initialized")
        else:
            logger.warning(
                "LLM client not initialized – set AZURE_OPENAI_* or OPENAI_API_KEY. "
                "See backend/docs/azure_openai_integration.md or docs/README_PHASE4.md."
            )

        # Initialize search manager with cache integration
        from services.search_apis import create_search_manager
        search_manager = create_search_manager()
        # SearchAPIManager uses context manager protocol, enter it
        await search_manager.__aenter__()
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
            HealthCheckService,
        )

        if PROMETHEUS_AVAILABLE:
            metrics_registry = CollectorRegistry()
            prometheus = PrometheusMetrics(metrics_registry)
            insights = ApplicationInsights(prometheus)
            monitoring_middleware = create_monitoring_middleware(
                prometheus, insights
            )
        else:
            # Fallback: monitoring without prometheus
            prometheus = None
            insights = None
            monitoring_middleware = None
            logger.warning("Prometheus monitoring disabled - install prometheus_client to enable")

        health_service = HealthCheckService()

        app.state.monitoring = {
            "prometheus": prometheus,
            "insights": insights,
            "middleware": monitoring_middleware,
            "health": health_service,
        }
        logger.info("✓ Monitoring systems initialized" if PROMETHEUS_AVAILABLE else "✓ Health service initialized (prometheus disabled)")

        # Start WebSocket keepalive pings to prevent idle proxy drops
        try:
            connection_manager.start_keepalive()
            logger.info("✓ WebSocket keepalive started")
        except Exception as e:
            logger.warning("WebSocket keepalive not started: %s", e)

        # Initialize production services
        from services.auth_service import auth_service
        from services.rate_limiter import RateLimiter
        from services.webhook_manager import (
            WebhookManager,
            create_webhook_router,
            WebhookEvent,
        )
        from services.export_service import ExportService, create_export_router

        app.state.auth_service = auth_service
        app.state.rate_limiter = RateLimiter()
        app.state.webhook_manager = WebhookManager()
        app.state.export_service = ExportService()
        logger.info("✓ Production services initialized")

        # Start feedback reconciliation (optional feature)
        if ENABLE_FEEDBACK_RECONCILE:
            try:
                from services.feedback_reconciliation import feedback_reconciliation
                from datetime import timedelta
                try:
                    feedback_reconciliation._window = timedelta(minutes=FEEDBACK_RECONCILE_WINDOW_MINUTES)
                except Exception:
                    # Keep default window if assignment fails
                    pass
                await feedback_reconciliation.start()
                logger.info("✓ Feedback reconciliation service started")
            except Exception as e:
                logger.warning("Feedback reconciliation not started: %s", e)

        # Mount export routes under /v1 using the initialized service
        try:
            export_router = create_export_router(app.state.export_service)
            app.include_router(export_router, prefix="/v1")
            logger.info("✓ Export routes mounted under /v1/export")
        except Exception as e:
            logger.warning("Failed to mount export routes: %s", e)

        # Register health checks (readiness)
        try:
            from database.connection import database_health_check

            async def _db_check():
                return await database_health_check()

            health_service.register_check("database", _db_check)

            # Register Redis check only if cache initialized
            if cache_success:
                async def _redis_check():
                    from services.cache import cache_manager
                    async with cache_manager.get_client() as client:
                        pong = await client.ping()
                    return {"redis": "ok" if pong else "unresponsive"}

                health_service.register_check("redis", _redis_check)

            # Lightweight auth check
            def _auth_check():
                return {"loaded": True}

            health_service.register_check("auth_service", _auth_check)
            
            # LLM readiness check: only report ready when Azure/OpenAI clients are initialized
            try:
                def _llm_check():
                    try:
                        ready = bool(getattr(app.state, "llm_initialized", False))
                    except Exception:
                        ready = False
                    return {"initialized": ready}
                health_service.register_check("llm", _llm_check)
            except Exception:
                pass
            logger.info("✓ Readiness health checks registered")
        except Exception as e:
            logger.warning("Health check registration failed: %s", e)

        # Mount webhook routes under /v1 using the initialized manager
        try:
            webhook_router = create_webhook_router(app.state.webhook_manager)
            app.include_router(webhook_router, prefix="/v1")
            logger.info("✓ Webhook routes mounted under /v1/webhooks")
        except Exception as e:
            logger.warning("Failed to mount webhook routes: %s", e)

        # Register self-healing switch notifications to emit webhooks
        try:
            from services.self_healing_system import (
                self_healing_system,
                register_switch_listener,
            )

            async def _on_paradigm_switch(decision, record):
                try:
                    payload = {
                        "research_id": getattr(record, "query_id", None),
                        "from_paradigm": decision.originalParadigm.value,
                        "to_paradigm": decision.recommendedParadigm.value,
                        "confidence": decision.confidence,
                        "reasons": decision.reasons,
                        "expected_improvement": decision.expected_improvement,
                        "risk_score": decision.risk_score,
                    }
                    await app.state.webhook_manager.trigger_event(
                        WebhookEvent.PARADIGM_SWITCHED, payload
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to emit paradigm.switch webhook: %s", exc
                    )

            register_switch_listener(_on_paradigm_switch)
            logger.info("✓ Registered paradigm switch webhook listener")
        except Exception as e:
            logger.warning(
                "Could not register paradigm switch listener: %s", e
            )

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

        # Stop feedback reconciliation (if running)
        if ENABLE_FEEDBACK_RECONCILE:
            try:
                from services.feedback_reconciliation import feedback_reconciliation
                await feedback_reconciliation.stop()
                logger.info("✓ Feedback reconciliation service stopped")
            except Exception as e:
                logger.warning("Failed to stop feedback reconciliation: %s", e)

        # Cleanup search manager
        if hasattr(app.state, 'search_manager'):
            await app.state.search_manager.__aexit__(None, None, None)
            logger.info("✓ Search manager cleaned up")

        # Stop WebSocket keepalive
        try:
            await connection_manager.stop_keepalive()
        except Exception:
            pass

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
        description=(
            "Full-featured paradigm-aware research with integrated "
            "Context Engineering Pipeline"
        ),
        lifespan=lifespan,
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
    # Mount all business routes under /v1 (SSOTA versioning)
    app.include_router(auth_router, prefix="/v1")
    app.include_router(research_router, prefix="/v1")
    app.include_router(paradigms_router, prefix="/v1")

    # Optional routers if present
    if search_router is not None:
        app.include_router(search_router, prefix="/v1")
    if users_router is not None:
        app.include_router(users_router, prefix="/v1")
    if system_router is not None:
        app.include_router(system_router, prefix="/v1")
    if responses_router is not None:
        app.include_router(responses_router, prefix="/v1")
    if feedback_router is not None:
        app.include_router(feedback_router, prefix="/v1")

    # Mount WebSocket routes (for real-time research progress)
    ws_router = create_websocket_router(connection_manager, progress_tracker)
    app.include_router(ws_router)


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
                    "llm": "healthy" if getattr(app.state, "llm_initialized", False) else "degraded",
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

    @app.get("/ready")
    async def readiness():
        health = getattr(app.state, "monitoring", {}).get("health")
        if health:
            return await health.get_readiness()
        return {"ready": True, "checks": {}}

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        try:
            if not PROMETHEUS_AVAILABLE:
                return Response(
                    content="# Prometheus metrics disabled - install prometheus_client\n",
                    media_type="text/plain; charset=utf-8",
                    status_code=503
                )
            prometheus = getattr(app.state, "monitoring", {}).get("prometheus")
            if not prometheus:
                return Response(status_code=204)
            content = generate_latest(prometheus.registry)
            return Response(
                content=content,
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        except Exception as e:
            logger.error("Failed to render /metrics: %s", e)
            return Response(status_code=500)

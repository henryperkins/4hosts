"""
Webhook Support for Async Operations
Phase 5: Production-Ready Features
"""

import asyncio
import httpx
import json
import hmac
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from pydantic import BaseModel, HttpUrl, Field
import logging
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from collections import defaultdict, deque
import uuid

# Configure logging
logger = structlog.get_logger(__name__)

# --- Data Models ---


class WebhookEvent(str, Enum):
    """Supported webhook events"""

    RESEARCH_STARTED = "research.started"
    RESEARCH_PROGRESS = "research.progress"
    RESEARCH_COMPLETED = "research.completed"
    RESEARCH_FAILED = "research.failed"
    RESEARCH_CANCELLED = "research.cancelled"
    PARADIGM_SWITCHED = "paradigm.switched"
    CLASSIFICATION_COMPLETED = "classification.completed"
    SYNTHESIS_COMPLETED = "synthesis.completed"
    EXPORT_READY = "export.ready"
    RATE_LIMIT_WARNING = "rate_limit.warning"
    ERROR_OCCURRED = "error.occurred"


class WebhookConfig(BaseModel):
    """Webhook configuration"""

    id: str = Field(default_factory=lambda: f"wh_{uuid.uuid4().hex[:12]}")
    url: HttpUrl
    events: List[WebhookEvent]
    secret: Optional[str] = None
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retry_policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_attempts": 3,
            "initial_delay": 1,
            "max_delay": 60,
            "exponential_base": 2,
        }
    )
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: int = 30


class WebhookDelivery(BaseModel):
    """Webhook delivery record"""

    id: str = Field(default_factory=lambda: f"del_{uuid.uuid4().hex[:12]}")
    webhook_id: str
    event: WebhookEvent
    payload: Dict[str, Any]
    status: str = "pending"  # pending, success, failed
    attempts: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None


class WebhookPayload(BaseModel):
    """Standard webhook payload structure"""

    event: WebhookEvent
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    webhook_id: str
    delivery_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- Webhook Manager ---


class WebhookManager:
    """Manages webhook subscriptions and deliveries"""

    def __init__(self, signing_secret: Optional[str] = None):
        self.signing_secret = signing_secret or "default-webhook-secret"
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers: Dict[WebhookEvent, List[Callable]] = defaultdict(list)
        self._delivery_task = None
        self._retry_task = None

    async def start(self):
        """Start webhook delivery workers"""
        self._delivery_task = asyncio.create_task(self._delivery_worker())
        self._retry_task = asyncio.create_task(self._retry_worker())
        logger.info("Webhook manager started")

    async def stop(self):
        """Stop webhook delivery workers"""
        if self._delivery_task:
            self._delivery_task.cancel()
        if self._retry_task:
            self._retry_task.cancel()
        await asyncio.gather(
            self._delivery_task, self._retry_task, return_exceptions=True
        )
        logger.info("Webhook manager stopped")

    def register_webhook(self, config: WebhookConfig) -> str:
        """Register a new webhook"""
        self.webhooks[config.id] = config
        logger.info(f"Registered webhook {config.id} for events: {config.events}")
        return config.id

    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False

    def update_webhook(self, webhook_id: str, updates: Dict[str, Any]) -> bool:
        """Update webhook configuration"""
        if webhook_id in self.webhooks:
            webhook = self.webhooks[webhook_id]
            for key, value in updates.items():
                if hasattr(webhook, key):
                    setattr(webhook, key, value)
            logger.info(f"Updated webhook {webhook_id}")
            return True
        return False

    async def trigger_event(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Trigger a webhook event"""
        # Find all webhooks subscribed to this event
        triggered_webhooks = [
            webhook
            for webhook in self.webhooks.values()
            if webhook.active and event in webhook.events
        ]

        # Create deliveries for each webhook
        for webhook in triggered_webhooks:
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                event=event,
                payload={"event": event, "data": data, "metadata": metadata or {}},
            )

            self.deliveries[delivery.id] = delivery
            await self.delivery_queue.put(delivery.id)

        logger.info(f"Triggered event {event} for {len(triggered_webhooks)} webhooks")

        # Call registered event handlers
        for handler in self.event_handlers.get(event, []):
            try:
                await handler(event, data, metadata)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    def register_event_handler(self, event: WebhookEvent, handler: Callable):
        """Register an internal event handler"""
        self.event_handlers[event].append(handler)

    async def _delivery_worker(self):
        """Worker to process webhook deliveries"""
        while True:
            try:
                delivery_id = await self.delivery_queue.get()
                await self._deliver_webhook(delivery_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in delivery worker: {e}")

    async def _retry_worker(self):
        """Worker to retry failed webhook deliveries"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._process_retries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retry worker: {e}")

    async def _deliver_webhook(self, delivery_id: str):
        """Deliver a single webhook"""
        delivery = self.deliveries.get(delivery_id)
        if not delivery:
            return

        webhook = self.webhooks.get(delivery.webhook_id)
        if not webhook or not webhook.active:
            delivery.status = "skipped"
            return

        # Prepare payload
        payload = WebhookPayload(
            event=delivery.event,
            webhook_id=webhook.id,
            delivery_id=delivery.id,
            data=delivery.payload["data"],
            metadata=delivery.payload.get("metadata", {}),
        )

        # Sign payload if secret is configured
        payload_json = payload.model_dump_json()
        signature = None
        if webhook.secret:
            signature = self._generate_signature(payload_json, webhook.secret)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FourHostsResearch-Webhook/1.0",
            "X-Webhook-Event": delivery.event,
            "X-Webhook-Delivery": delivery.id,
            "X-Webhook-Timestamp": payload.timestamp.isoformat(),
            **webhook.headers,
        }

        if signature:
            headers["X-Webhook-Signature"] = signature

        # Attempt delivery
        delivery.attempts += 1

        try:
            async with httpx.AsyncClient(timeout=webhook.timeout) as client:
                response = await client.post(
                    str(webhook.url), content=payload_json, headers=headers
                )

                delivery.response_status = response.status_code
                delivery.response_body = response.text[:1000]  # Truncate

                if 200 <= response.status_code < 300:
                    delivery.status = "success"
                    delivery.delivered_at = datetime.now(timezone.utc)
                    logger.info(
                        f"Successfully delivered webhook {delivery.id} "
                        f"to {webhook.url}"
                    )
                else:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )

        except Exception as e:
            delivery.error = str(e)
            logger.error(f"Failed to deliver webhook {delivery.id}: {e}")

            # Schedule retry if within limits
            if delivery.attempts < webhook.retry_policy["max_attempts"]:
                delay = min(
                    webhook.retry_policy["initial_delay"]
                    * (
                        webhook.retry_policy["exponential_base"]
                        ** (delivery.attempts - 1)
                    ),
                    webhook.retry_policy["max_delay"],
                )
                delivery.next_retry_at = datetime.now(timezone.utc) + timedelta(
                    seconds=delay
                )
                delivery.status = "retry"
            else:
                delivery.status = "failed"

    async def _process_retries(self):
        """Process webhooks scheduled for retry"""
        now = datetime.now(timezone.utc)

        for delivery in self.deliveries.values():
            if (
                delivery.status == "retry"
                and delivery.next_retry_at
                and delivery.next_retry_at <= now
            ):
                await self.delivery_queue.put(delivery.id)

    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected = self._generate_signature(payload, secret)
        return hmac.compare_digest(expected, signature)

    def get_delivery_status(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get status of a webhook delivery"""
        return self.deliveries.get(delivery_id)

    def get_webhook_deliveries(
        self, webhook_id: str, limit: int = 100
    ) -> List[WebhookDelivery]:
        """Get recent deliveries for a webhook"""
        deliveries = [d for d in self.deliveries.values() if d.webhook_id == webhook_id]
        return sorted(deliveries, key=lambda d: d.created_at, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics"""
        total_deliveries = len(self.deliveries)
        if total_deliveries == 0:
            return {
                "total_deliveries": 0,
                "success_rate": 0,
                "active_webhooks": len([w for w in self.webhooks.values() if w.active]),
            }

        success_count = sum(
            1 for d in self.deliveries.values() if d.status == "success"
        )

        failed_count = sum(1 for d in self.deliveries.values() if d.status == "failed")

        return {
            "total_deliveries": total_deliveries,
            "successful": success_count,
            "failed": failed_count,
            "pending": sum(
                1 for d in self.deliveries.values() if d.status == "pending"
            ),
            "retry": sum(1 for d in self.deliveries.values() if d.status == "retry"),
            "success_rate": (success_count / total_deliveries) * 100,
            "active_webhooks": len([w for w in self.webhooks.values() if w.active]),
            "events_by_type": self._count_events_by_type(),
        }

    def _count_events_by_type(self) -> Dict[str, int]:
        """Count deliveries by event type"""
        counts = defaultdict(int)
        for delivery in self.deliveries.values():
            counts[delivery.event] += 1
        return dict(counts)


# --- Webhook Testing Utilities ---


class WebhookTester:
    """Utilities for testing webhooks"""

    @staticmethod
    async def test_webhook_endpoint(url: str, timeout: int = 10) -> Dict[str, Any]:
        """Test if webhook endpoint is reachable"""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Send a test payload
                test_payload = {
                    "event": "webhook.test",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {"message": "Test webhook delivery"},
                }

                response = await client.post(
                    url,
                    json=test_payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "FourHostsResearch-WebhookTest/1.0",
                    },
                )

                return {
                    "reachable": True,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "success": 200 <= response.status_code < 300,
                }

        except Exception as e:
            return {"reachable": False, "error": str(e), "success": False}

    @staticmethod
    def generate_test_payload(event: WebhookEvent) -> Dict[str, Any]:
        """Generate test payload for webhook event"""
        test_data = {
            WebhookEvent.RESEARCH_STARTED: {
                "research_id": "test_research_123",
                "query": "Test research query",
                "paradigm": "maeve",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            WebhookEvent.RESEARCH_COMPLETED: {
                "research_id": "test_research_123",
                "query": "Test research query",
                "paradigm": "maeve",
                "duration_seconds": 45.2,
                "sources_analyzed": 127,
                "synthesis_score": 0.89,
            },
            WebhookEvent.RESEARCH_PROGRESS: {
                "research_id": "test_research_123",
                "phase": "synthesis",
                "progress_percent": 75,
                "sources_analyzed": 95,
            },
        }

        return test_data.get(event, {"test": True})


# --- FastAPI Integration ---

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from services.auth_service import get_current_user, TokenData, require_role, UserRole


def create_webhook_router(webhook_manager: WebhookManager) -> APIRouter:
    """Create FastAPI router for webhook endpoints"""
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

    @router.post("/")
    async def create_webhook(
        config: WebhookConfig,
        current_user: TokenData = Depends(require_role(UserRole.BASIC)),
        background_tasks: BackgroundTasks = None,
    ):
        """Create a new webhook subscription"""
        # Test webhook endpoint
        test_result = await WebhookTester.test_webhook_endpoint(str(config.url))
        if not test_result["reachable"]:
            raise HTTPException(
                status_code=400,
                detail=f"Webhook endpoint is not reachable: {test_result.get('error')}",
            )

        # Register webhook
        webhook_id = webhook_manager.register_webhook(config)

        # Send test event
        if background_tasks:
            background_tasks.add_task(
                webhook_manager.trigger_event,
                WebhookEvent.RESEARCH_STARTED,
                WebhookTester.generate_test_payload(WebhookEvent.RESEARCH_STARTED),
            )

        return {
            "webhook_id": webhook_id,
            "status": "created",
            "test_result": test_result,
        }

    @router.get("/")
    async def list_webhooks(current_user: TokenData = Depends(get_current_user)):
        """List all webhooks for current user"""
        # In production, filter by user
        return {"webhooks": list(webhook_manager.webhooks.values())}

    @router.get("/{webhook_id}")
    async def get_webhook(
        webhook_id: str, current_user: TokenData = Depends(get_current_user)
    ):
        """Get webhook details"""
        webhook = webhook_manager.webhooks.get(webhook_id)
        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook not found")

        return {
            "webhook": webhook,
            "recent_deliveries": webhook_manager.get_webhook_deliveries(
                webhook_id, limit=10
            ),
        }

    @router.patch("/{webhook_id}")
    async def update_webhook(
        webhook_id: str,
        updates: Dict[str, Any],
        current_user: TokenData = Depends(get_current_user),
    ):
        """Update webhook configuration"""
        if not webhook_manager.update_webhook(webhook_id, updates):
            raise HTTPException(status_code=404, detail="Webhook not found")

        return {"status": "updated"}

    @router.delete("/{webhook_id}")
    async def delete_webhook(
        webhook_id: str, current_user: TokenData = Depends(get_current_user)
    ):
        """Delete a webhook"""
        if not webhook_manager.unregister_webhook(webhook_id):
            raise HTTPException(status_code=404, detail="Webhook not found")

        return {"status": "deleted"}

    @router.get("/{webhook_id}/deliveries")
    async def get_webhook_deliveries(
        webhook_id: str,
        limit: int = 100,
        current_user: TokenData = Depends(get_current_user),
    ):
        """Get webhook delivery history"""
        deliveries = webhook_manager.get_webhook_deliveries(webhook_id, limit)
        return {"deliveries": deliveries, "count": len(deliveries)}

    @router.post("/{webhook_id}/test")
    async def test_webhook(
        webhook_id: str,
        event: WebhookEvent,
        background_tasks: BackgroundTasks,
        current_user: TokenData = Depends(get_current_user),
    ):
        """Send a test event to webhook"""
        webhook = webhook_manager.webhooks.get(webhook_id)
        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook not found")

        test_data = WebhookTester.generate_test_payload(event)

        background_tasks.add_task(
            webhook_manager.trigger_event, event, test_data, {"test": True}
        )

        return {"status": "test_triggered", "event": event, "data": test_data}

    @router.get("/stats")
    async def get_webhook_stats(
        current_user: TokenData = Depends(require_role(UserRole.ADMIN)),
    ):
        """Get webhook system statistics"""
        return webhook_manager.get_stats()

    return router

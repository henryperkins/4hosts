"""
Feedback Reconciliation Service

Moves orphaned per-user feedback events into research records once
the research_id becomes available. Prevents data fragmentation.

Buckets are stored under research_store using the research_id value
'feedback:{user_id}'. Since ResearchStore prefixes all keys with
'research:', Redis keys will look like 'research:feedback:{user_id}'.
"""

from __future__ import annotations

import asyncio
import logging
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from services.research_store import research_store

logger = logging.getLogger(__name__)


@dataclass
class ReconcileStats:
    scanned_buckets: int = 0
    scanned_events: int = 0
    reconciled: int = 0


class FeedbackReconciliationService:
    """Reconciles orphaned feedback events with research records"""

    def __init__(self, *, interval_seconds: int = 300, window_minutes: int = 10):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._interval = max(60, int(interval_seconds))
        self._window = timedelta(minutes=max(1, int(window_minutes)))

    async def start(self) -> None:
        """Start the reconciliation background task"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._reconciliation_loop())
        logger.info("Feedback reconciliation service started")

    async def stop(self) -> None:
        """Stop the reconciliation service"""
        self._running = False
        if self._task:
            try:
                await self._task
            except Exception:
                pass
        logger.info("Feedback reconciliation service stopped")

    async def _reconciliation_loop(self) -> None:
        """Run until stopped; execute reconciliation repeatedly"""
        while self._running:
            try:
                await self._run_reconciliation()
            except Exception as e:
                logger.error("Reconciliation error: %s", e, exc_info=True)
            await asyncio.sleep(self._interval)

    async def _run_reconciliation(self) -> None:
        """Execute one reconciliation pass across all feedback buckets"""
        stats = ReconcileStats()
        # Enumerate user feedback buckets
        async for bucket_id in self._iter_feedback_bucket_ids():
            stats.scanned_buckets += 1
            try:
                bucket_data = await research_store.get(bucket_id)
            except Exception as e:
                logger.debug("Failed to load bucket %s: %s", bucket_id, e)
                continue

            if not bucket_data or "events" not in bucket_data:
                continue

            events: List[Dict[str, Any]] = list(bucket_data.get("events") or [])
            if not events:
                continue

            to_keep: List[Dict[str, Any]] = []
            for evt in events:
                stats.scanned_events += 1
                try:
                    user_id = str(evt.get("user_id") or "")
                    payload = dict(evt.get("payload") or {})
                    evt_ts_raw = evt.get("timestamp")
                    evt_time = self._parse_time(evt_ts_raw)

                    # If the event already carries a research_id, attach immediately
                    attached = False
                    if isinstance(payload, dict) and "research_id" in payload and payload.get("research_id"):
                        rid = str(payload.get("research_id"))
                        await self._attach_event_to_research(rid, evt)
                        stats.reconciled += 1
                        attached = True
                    else:
                        # Attempt to resolve via query/time proximity
                        rid = await self._find_matching_research(user_id, payload, evt_time)
                        if rid:
                            await self._attach_event_to_research(rid, evt)
                            stats.reconciled += 1
                            attached = True

                    if not attached:
                        to_keep.append(evt)

                except Exception as e:
                    # Keep the event for later; do not lose data
                    logger.debug("Failed to process event in %s: %s", bucket_id, e)
                    to_keep.append(evt)

            # Write back remaining events (if any)
            try:
                await research_store.set(bucket_id, {"events": to_keep})
            except Exception as e:
                logger.warning("Failed to update bucket %s: %s", bucket_id, e)

        # Emit reconciliation metric
        if stats.reconciled > 0:
            try:
                from backend.services.metrics import metrics  # type: ignore
                metrics.increment("feedback_events_reconciled", amount=stats.reconciled)
            except Exception:
                pass

        logger.info(
            "Feedback reconciliation pass complete • buckets=%d events=%d moved=%d",
            stats.scanned_buckets, stats.scanned_events, stats.reconciled
        )

    async def _iter_feedback_bucket_ids(self):
        """Yield feedback bucket research_ids ('feedback:{user_id}')"""
        # Redis mode
        if getattr(research_store, "use_redis", False) and getattr(research_store, "redis_client", None):
            try:
                prefix = getattr(research_store, "key_prefix", "research:")
                pattern = f"{prefix}feedback:*"
                async for key in research_store.redis_client.scan_iter(pattern):  # type: ignore[attr-defined]
                    # Convert Redis key -> research_id by stripping key_prefix
                    yield str(key)[len(prefix):]
                return
            except Exception as e:
                logger.debug("Redis scan failed; falling back to in-memory scan: %s", e)

        # In-memory fallback
        try:
            for rid in list(getattr(research_store, "fallback_store", {}).keys()):
                if isinstance(rid, str) and rid.startswith("feedback:"):
                    yield rid
        except Exception:
            # Nothing else to do
            return

    async def _find_matching_research(
        self,
        user_id: str,
        payload: Dict[str, Any],
        evt_time: Optional[datetime],
    ) -> Optional[str]:
        """Find research record for a user matching a feedback event"""

        if not user_id:
            return None

        try:
            recent: List[Dict[str, Any]] = await research_store.get_user_research(user_id, limit=150)
        except Exception:
            recent = []

        if not recent:
            return None

        # Derive a stable query signature when possible
        query_text = ""
        if "query" in payload and isinstance(payload.get("query"), str):
            query_text = payload["query"].strip()
        query_hash = self._short_hash(query_text) if query_text else None

        for rec in recent:
            try:
                # Match by identical query or hash
                rq = (rec.get("query") or "").strip()
                if query_text:
                    if rq == query_text:
                        if self._within_window(rec, evt_time):
                            return str(rec.get("id"))
                    # fast hash match if research has stored a query hash
                    rh = rec.get("query_hash")
                    if rh and query_hash and rh == query_hash and self._within_window(rec, evt_time):
                        return str(rec.get("id"))

                # If we lack query text, fall back to time-only proximity within window (same user)
                if not query_text and self._within_window(rec, evt_time):
                    return str(rec.get("id"))
            except Exception:
                continue

        return None

    async def _attach_event_to_research(self, research_id: str, event: Dict[str, Any]) -> None:
        """Attach a feedback event to a research record"""
        try:
            rec = await research_store.get(research_id) or {}
            arr = list(rec.get("feedback_events") or [])
            arr.append(event)
            await research_store.update_field(research_id, "feedback_events", arr)
        except Exception as e:
            logger.debug("Failed to attach feedback to research %s: %s", research_id, e)
            raise

    def _within_window(self, rec: Dict[str, Any], evt_time: Optional[datetime]) -> bool:
        if evt_time is None:
            return False
        try:
            created_at = rec.get("created_at")
            if not created_at:
                return False
            rec_time = self._parse_time(created_at)
            if rec_time is None:
                return False
            return abs(rec_time - evt_time) <= self._window
        except Exception:
            return False

    @staticmethod
    def _parse_time(val: Any) -> Optional[datetime]:
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    @staticmethod
    def _short_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# Global instance
feedback_reconciliation = FeedbackReconciliationService()
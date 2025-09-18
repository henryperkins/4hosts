import asyncio
import os
import sys
import pytest

# Ensure the backend package root is importable when running this test in isolation
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.research_store import ResearchStore


@pytest.mark.asyncio
async def test_update_fields_is_atomic_and_versions_increment():
    store = ResearchStore(redis_url="redis://localhost:0")  # ensure redis disabled
    research_id = "res_test_atomic_1"

    # Seed a minimal record
    await store.set(research_id, {"id": research_id, "status": "processing"})

    # Update multiple fields in a single call
    await store.update_fields(research_id, {"results": {"ok": True}, "status": "completed"})

    rec = await store.get(research_id)
    assert rec is not None
    assert rec["status"] == "completed"
    assert rec["results"] == {"ok": True}
    assert isinstance(rec.get("version"), int) and rec["version"] >= 1
    assert isinstance(rec.get("_updated_at"), str)

    # Single-field update still increments version
    prev_ver = rec["version"]
    await store.update_field(research_id, "status", "failed")
    rec2 = await store.get(research_id)
    assert rec2["status"] == "failed"
    assert rec2["version"] == prev_ver + 1

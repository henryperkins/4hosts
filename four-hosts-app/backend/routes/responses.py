"""
Responses API proxy routes
-------------------------
Thin FastAPI router that exposes minimal endpoints for:
- starting background Responses jobs (o3 by default)
- retrieving and cancelling Responses
- streaming events with optional resume via starting_after cursor
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Union, AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from services.openai_responses_client import get_responses_client


router = APIRouter(prefix="/responses", tags=["responses"])


class Message(BaseModel):
    role: Literal["user", "assistant", "system", "developer"]
    content: Union[str, List[Dict[str, Any]]]


class StartBackgroundRequest(BaseModel):
    model: Optional[str] = Field(None, description="Deployment name; defaults to AZURE_OPENAI_DEPLOYMENT or 'o3'")
    input: Union[str, List[Message]]
    tools: Optional[List[Dict[str, Any]]] = None
    max_tool_calls: Optional[int] = None
    instructions: Optional[str] = None
    store: bool = True


@router.post("/background")
async def start_background(req: StartBackgroundRequest):
    """Start a background Responses job (store=true required on Azure)."""
    client = get_responses_client()
    try:
        resp = await client.create_response(
            model=(req.model or "o3"),
            input=[m.model_dump() if isinstance(m, Message) else m for m in (req.input if isinstance(req.input, list) else [{"role": "user", "content": req.input}])],
            tools=req.tools,
            background=True,
            stream=False,
            max_tool_calls=req.max_tool_calls,
            instructions=req.instructions,
            store=req.store,
        )
        return JSONResponse(resp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to start background job: {e}")


@router.get("/{response_id}")
async def get_response(response_id: str):
    client = get_responses_client()
    try:
        resp = await client.retrieve_response(response_id)
        return JSONResponse(resp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to retrieve response: {e}")


@router.post("/{response_id}/cancel")
async def cancel_response(response_id: str):
    client = get_responses_client()
    try:
        resp = await client.cancel_response(response_id)
        return JSONResponse(resp)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to cancel response: {e}")


@router.get("/{response_id}/stream")
async def stream_response(response_id: str, starting_after: Optional[int] = Query(None)):
    client = get_responses_client()
    try:
        async def event_gen() -> AsyncIterator[bytes]:
            async for event in client.stream_response(response_id, starting_after=starting_after):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to stream response: {e}")


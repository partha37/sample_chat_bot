"""Chat endpoints: blocking chat, SSE streaming, and breakpoint resume."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.ai_service import AIService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """Payload for a chat turn."""

    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    session_id: str = Field(default="default", max_length=128)
    history: list[dict[str, str]] | None = Field(
        default=None, description="Optional prior turns used to seed the conversation"
    )
    debug: bool = Field(default=True, description="Include the full debugger trace")


class ChatResponse(BaseModel):
    """Result of a chat turn."""

    answer: str
    trace_id: str
    session_id: str
    thread_id: str
    intent: str = ""
    route_reason: str = ""
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    paused: bool = False
    interrupt: dict[str, Any] | None = None
    active_model: str | None = None
    debug: dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    """Resume a graph run that was paused at a breakpoint."""

    session_id: str = Field(default="default", max_length=128)
    skip: bool = Field(default=False, description="Skip the node instead of running it")
    state_patch: dict[str, Any] = Field(default_factory=dict)


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Run one turn of the chat graph and return the answer plus its trace."""
    service = _service(request)
    try:
        result = await service.chat(
            message=payload.message,
            session_id=payload.session_id,
            history=payload.history,
            include_debug=payload.debug,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return ChatResponse(**result.to_dict())


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """Stream tokens and debugger events for one turn as Server-Sent Events."""
    service = _service(request)

    async def event_generator() -> AsyncIterator[str]:
        yield _sse("open", {"session_id": payload.session_id})
        try:
            async for event in service.chat_stream(
                message=payload.message,
                session_id=payload.session_id,
                history=payload.history,
            ):
                yield _sse(event.type, event.data)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chat failed")
            yield _sse("error", {"error": f"{type(exc).__name__}: {exc}"})
        yield _sse("close", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/resume", response_model=ChatResponse)
async def chat_resume(request: Request, payload: ResumeRequest) -> ChatResponse:
    """Resume a graph run that paused at a breakpoint."""
    service = _service(request)
    resume_value: dict[str, Any] = {"skip": payload.skip}
    resume_value.update(payload.state_patch)

    try:
        result = await service.resume(session_id=payload.session_id, value=resume_value)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Resume failed")
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}") from exc

    return ChatResponse(**result.to_dict())

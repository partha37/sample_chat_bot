"""Debugger endpoints: traces, breakpoints, request log and graph metadata."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ai_layer.debugger import BreakpointMode
from backend.services.ai_service import AIService

router = APIRouter(prefix="/api/debug", tags=["debug"])


class BreakpointRequest(BaseModel):
    """Set the breakpoint mode for a node."""

    mode: str = Field(..., description="off | break | skip")


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------
@router.get("/status")
async def debug_status(request: Request) -> dict[str, Any]:
    """Everything the debug panel needs for its header."""
    service = _service(request)
    return {
        "ai": service.status(),
        "requests": request.app.state.backend_debugger.summary(),
        "trace_count": len(service.list_traces(limit=10_000)),
        "breakpoints": service.list_breakpoints(),
    }


@router.get("/graph")
async def debug_graph(request: Request) -> dict[str, Any]:
    """Node/edge structure of the LangGraph, used to draw the flow diagram."""
    return _service(request).graph_structure()


# ----------------------------------------------------------------------
# Graph traces
# ----------------------------------------------------------------------
@router.get("/traces")
async def list_traces(
    request: Request,
    session_id: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
) -> dict[str, Any]:
    """Recent graph traces, newest first."""
    service = _service(request)
    traces = service.list_traces(session_id=session_id, limit=limit)
    return {"count": len(traces), "traces": traces}


@router.get("/traces/{trace_id}")
async def get_trace(request: Request, trace_id: str) -> dict[str, Any]:
    """Full trace including every node event."""
    service = _service(request)
    trace = service.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found")
    return trace


@router.delete("/traces")
async def clear_traces(request: Request) -> dict[str, Any]:
    """Drop every stored graph trace."""
    removed = _service(request).clear_traces()
    return {"cleared": removed}


@router.get("/sessions/{session_id}/checkpoints")
async def session_checkpoints(request: Request, session_id: str) -> dict[str, Any]:
    """LangGraph checkpoint history for a session (step-by-step replay)."""
    service = _service(request)
    history = service.checkpoint_history(session_id)
    return {"session_id": session_id, "count": len(history), "checkpoints": history}


# ----------------------------------------------------------------------
# Breakpoints
# ----------------------------------------------------------------------
@router.get("/breakpoints")
async def list_breakpoints(request: Request) -> dict[str, Any]:
    """Current breakpoint registry (node -> mode)."""
    return {"breakpoints": _service(request).list_breakpoints()}


@router.put("/breakpoints/{node}")
async def set_breakpoint(request: Request, node: str, payload: BreakpointRequest) -> dict[str, Any]:
    """Set (or clear) a breakpoint on a node: off | break | skip."""
    service = _service(request)
    try:
        mode = service.set_breakpoint(node, payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"node": node, "mode": mode.value, "breakpoints": service.list_breakpoints()}


@router.delete("/breakpoints")
async def clear_breakpoints(request: Request) -> dict[str, Any]:
    """Remove every breakpoint."""
    service = _service(request)
    service.clear_breakpoints()
    return {"breakpoints": service.list_breakpoints()}


# ----------------------------------------------------------------------
# HTTP request log
# ----------------------------------------------------------------------
@router.get("/requests")
async def list_requests(request: Request, limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    """Traced HTTP requests handled by this backend."""
    debugger = request.app.state.backend_debugger
    records = debugger.list(limit=limit)
    return {"count": len(records), "requests": records, "summary": debugger.summary()}


@router.delete("/requests")
async def clear_requests(request: Request) -> dict[str, Any]:
    """Clear the HTTP request log."""
    cleared = request.app.state.backend_debugger.clear()
    return {"cleared": cleared}

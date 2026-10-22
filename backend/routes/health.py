"""Health and readiness endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness probe plus a summary of the AI layer configuration."""
    service = request.app.state.ai_service
    return {
        "status": "ok",
        "service": "sample-chat-bot-backend",
        "ai": service.status(),
        "model_validation": getattr(request.app.state, "model_validation", None),
    }

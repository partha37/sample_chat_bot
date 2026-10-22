"""FastAPI application factory and entry point.

Run it with:

    .\\.venv\\Scripts\\python -m backend.main

or let uvicorn serve the factory:

    .\\.venv\\Scripts\\uvicorn backend.main:create_app --factory --reload
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow `python backend/main.py` as well as `python -m backend.main`.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.config import get_backend_settings  # noqa: E402
from backend.debugger import BackendDebugger  # noqa: E402
from backend.middleware import (  # noqa: E402
    AccessLogMiddleware,
    RequestTracingMiddleware,
)
from backend.routes import chat, debug, health  # noqa: E402
from backend.routes.health import health as health_check  # noqa: E402
from backend.services.ai_service import AIService  # noqa: E402

logger = logging.getLogger("backend")


def configure_logging(level: str = "info") -> None:
    """Configure root logging with a compact, readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_backend_settings()
    configure_logging(settings.log_level)

    backend_debugger = BackendDebugger()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend_debugger = backend_debugger
        app.state.ai_service = AIService()
        logger.info("Backend ready - Gemini configured: %s", app.state.ai_service.configured)

        # Catch a retired/unknown model id at startup (this check is free -
        # it does not use generation quota).
        validation = app.state.ai_service.client.validate_model()
        app.state.model_validation = validation
        if not validation["exists"]:
            logger.error(
                "The configured model '%s' is not usable: %s",
                validation["model"],
                (validation.get("suggestion") and f"try '{validation['suggestion']}' instead")
                or validation.get("error"),
            )
        else:
            logger.info("Model '%s' is available", validation["model"])

        logger.info("Debug endpoints available under /api/debug")
        yield
        logger.info("Backend shutting down")

    app = FastAPI(
        title="Sample Chat Bot API",
        description=(
            "FastAPI backend driving a LangGraph + Gemini chat agent, with "
            "debuggers for the graph, the API and the UI."
        ),
        version="1.0.0",
        debug=settings.debug_enabled,
        lifespan=lifespan,
    )

    # Middleware order: the last one added is the outermost.
    app.add_middleware(RequestTracingMiddleware, debugger=backend_debugger)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id", "X-Response-Time-ms"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(debug.router)

    @app.get("/", tags=["meta"])
    async def root() -> dict:
        return {
            "service": "sample-chat-bot-backend",
            "docs": "/docs",
            "endpoints": {
                "chat": "POST /api/chat",
                "chat_stream": "POST /api/chat/stream (SSE)",
                "resume": "POST /api/chat/resume",
                "health": "GET /api/health",
                "debug": "GET /api/debug/status",
            },
        }

    @app.get("/health", tags=["health"])
    async def health_alias(request: Request) -> dict:
        """Convenience alias for GET /api/health."""
        return await health_check(request)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_backend_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_enabled,
        log_level=settings.log_level,
    )

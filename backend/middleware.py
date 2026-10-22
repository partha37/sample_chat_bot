"""HTTP middleware that traces every request through the backend."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.debugger import BackendDebugger

logger = logging.getLogger("backend.access")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Measures request duration and records it in the backend debugger."""

    def __init__(self, app, debugger: BackendDebugger) -> None:
        super().__init__(app)
        self.debugger = debugger

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        record = self.debugger.start(
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            client_ip=client_ip,
        )
        request.state.request_id = record.request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            self.debugger.finish(record, 500, error=f"{type(exc).__name__}: {exc}")
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        self.debugger.finish(record, response.status_code)

        response.headers["X-Request-Id"] = record.request_id
        response.headers["X-Response-Time-ms"] = str(duration_ms)

        logger.info(
            "%s %s -> %s (%.2f ms) request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            record.request_id,
        )
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Pretty, colour-free access log for the terminal."""

    async def dispatch(self, request: Request, call_next) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - started) * 1000
        level = logging.INFO if response.status_code < 400 else logging.WARNING
        logger.log(
            level,
            "%-6s %-28s %3d  %7.2fms",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        return response

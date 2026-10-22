"""Backend-level debugger.

Tracks every HTTP request handled by the API: method, path, status code,
duration, client info and any error.  The records feed the "Requests" tab in
the debug panel of the UI.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from backend.config import get_backend_settings


@dataclass
class RequestRecord:
    """A single traced HTTP request."""

    request_id: str
    method: str
    path: str
    query: str = ""
    status_code: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    duration_ms: float | None = None
    client_ip: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "query": self.query,
            "status_code": self.status_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "client_ip": self.client_ip,
            "error": self.error,
        }


class BackendDebugger:
    """Bounded ring buffer of traced requests."""

    def __init__(self, max_records: int | None = None) -> None:
        settings = get_backend_settings()
        self.max_records = max_records if max_records is not None else settings.max_trace_history
        self._records: OrderedDict[str, RequestRecord] = OrderedDict()

    def start(self, method: str, path: str, query: str = "", client_ip: str = "") -> RequestRecord:
        request_id = uuid.uuid4().hex[:12]
        record = RequestRecord(
            request_id=request_id,
            method=method,
            path=path,
            query=query,
            client_ip=client_ip,
        )
        self._records[request_id] = record
        while len(self._records) > self.max_records:
            self._records.popitem(last=False)
        return record

    def finish(
        self,
        record: RequestRecord,
        status_code: int,
        error: str | None = None,
    ) -> RequestRecord:
        record.ended_at = time.time()
        record.duration_ms = round((record.ended_at - record.started_at) * 1000, 2)
        record.status_code = status_code
        record.error = error
        return record

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        records = list(self._records.values())
        records.reverse()
        return [record.to_dict() for record in records[:limit]]

    def get(self, request_id: str) -> dict[str, Any] | None:
        record = self._records.get(request_id)
        return record.to_dict() if record else None

    def clear(self) -> int:
        count = len(self._records)
        self._records.clear()
        return count

    def summary(self) -> dict[str, Any]:
        records = list(self._records.values())
        durations = [r.duration_ms for r in records if r.duration_ms is not None]
        errors = [r for r in records if r.error]
        return {
            "total": len(records),
            "errors": len(errors),
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "max_duration_ms": round(max(durations), 2) if durations else 0.0,
        }

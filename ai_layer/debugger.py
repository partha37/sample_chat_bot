"""The AI layer debugger.

Provides three debugging capabilities:

1. **Tracing** - every node entry/exit is recorded with timings, the state
   snapshot before and after the node ran, and any error raised.
2. **Breakpoints** - a node can be marked ``break`` (pause and hand the state
   back to the caller) or ``skip`` (short-circuit without executing).
3. **Snapshots** - LangGraph state is rendered into a JSON-safe preview that is
   safe to ship to a browser (messages summarised, long text truncated,
   secrets redacted).

Traces are kept in a bounded in-memory ring buffer so the API can serve the
last N turns to the debug panel in the UI.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from ai_layer.config import get_settings

logger = logging.getLogger(__name__)


class BreakpointMode(str, Enum):
    """What the debugger should do when a node is reached."""

    OFF = "off"
    BREAK = "break"  # pause before executing the node
    SKIP = "skip"  # skip the node entirely


class NodeStatus(str, Enum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"
    PAUSED = "paused"


# Keys whose values should never leave the process.
_SECRET_HINTS = ("api_key", "apikey", "authorization", "token", "secret", "password")


@dataclass
class TraceEvent:
    """A single debugger event (node enter / exit / error / note)."""

    seq: int
    node: str
    phase: str
    status: str = NodeStatus.RUNNING.value
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    duration_ms: float | None = None
    input_preview: dict[str, Any] = field(default_factory=dict)
    output_preview: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "node": self.node,
            "phase": self.phase,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "input_preview": self.input_preview,
            "output_preview": self.output_preview,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class Trace:
    """All debugger events for a single chat turn."""

    trace_id: str
    session_id: str
    user_input: str
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    status: str = "running"
    events: list[TraceEvent] = field(default_factory=list)
    final_answer: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_input": self.user_input,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "duration_ms": (
                round((self.ended_at - self.started_at) * 1000, 2)
                if self.ended_at is not None
                else None
            ),
            "final_answer": self.final_answer,
            "error": self.error,
            "events": [event.to_dict() for event in self.events],
        }


class GraphDebugger:
    """Collects traces and holds the breakpoint registry for one graph."""

    def __init__(self, max_traces: int | None = None, max_preview_chars: int | None = None):
        settings = get_settings()
        self.max_traces = max_traces if max_traces is not None else settings.max_trace_history
        self.max_preview_chars = (
            max_preview_chars if max_preview_chars is not None else settings.max_preview_chars
        )
        self._traces: OrderedDict[str, Trace] = OrderedDict()
        self._breakpoints: dict[str, BreakpointMode] = {}
        self._listeners: list[Callable[[TraceEvent], None]] = []
        self._seq = 0
        self._active_trace_id: str | None = None

    # ------------------------------------------------------------------
    # Live listeners (used by the SSE endpoint to stream traces)
    # ------------------------------------------------------------------
    def add_listener(self, callback: Callable[[TraceEvent], None]) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[TraceEvent], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _emit(self, event: TraceEvent) -> None:
        """Record an event and notify live listeners."""
        trace = self._active_trace()
        if trace is not None:
            trace.events.append(event)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as exc:  # noqa: BLE001 - a bad listener must not break the graph
                logger.warning("Debugger listener failed: %s", exc)

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------
    def begin_trace(self, session_id: str, user_input: str) -> str:
        """Open a new trace and return its id."""
        trace_id = uuid.uuid4().hex[:12]
        self._traces[trace_id] = Trace(
            trace_id=trace_id,
            session_id=session_id,
            user_input=user_input,
        )
        self._active_trace_id = trace_id
        self._evict_old_traces()
        return trace_id

    def end_trace(
        self,
        trace_id: str,
        status: str = "ok",
        final_answer: str = "",
        error: str | None = None,
    ) -> None:
        trace = self._traces.get(trace_id)
        if trace is None:
            return
        trace.ended_at = time.time()
        trace.status = status
        trace.final_answer = final_answer
        trace.error = error

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        trace = self._traces.get(trace_id)
        return trace.to_dict() if trace else None

    def list_traces(self, session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        traces = list(self._traces.values())
        if session_id:
            traces = [t for t in traces if t.session_id == session_id]
        traces.reverse()  # newest first
        return [t.to_dict() for t in traces[:limit]]

    def clear_traces(self) -> int:
        count = len(self._traces)
        self._traces.clear()
        return count

    def _evict_old_traces(self) -> None:
        while len(self._traces) > self.max_traces:
            self._traces.popitem(last=False)

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _active_trace(self) -> Trace | None:
        if self._active_trace_id:
            return self._traces.get(self._active_trace_id)
        return next(reversed(self._traces.values()), None) if self._traces else None

    def enter_node(self, node: str, state: dict[str, Any]) -> TraceEvent:
        event = TraceEvent(
            seq=self._next_seq(),
            node=node,
            phase="enter",
            status=NodeStatus.RUNNING.value,
            input_preview=self.snapshot(state),
        )
        self._emit(event)
        return event

    def exit_node(
        self,
        node: str,
        state: dict[str, Any],
        event: TraceEvent | None = None,
        message: str = "",
        status: str = NodeStatus.OK.value,
    ) -> TraceEvent:
        exit_event = TraceEvent(
            seq=self._next_seq(),
            node=node,
            phase="exit",
            status=status,
            output_preview=self.snapshot(state),
            message=message,
        )
        if event is not None:
            event.ended_at = exit_event.ended_at = time.time()
            event.duration_ms = round((event.ended_at - event.started_at) * 1000, 2)
            event.status = status
            if message:
                event.message = message
        self._emit(exit_event)
        return exit_event

    def error_node(self, node: str, exc: BaseException, event: TraceEvent | None = None) -> None:
        exit_event = TraceEvent(
            seq=self._next_seq(),
            node=node,
            phase="error",
            status=NodeStatus.ERROR.value,
            ended_at=time.time(),
            error=f"{type(exc).__name__}: {exc}",
        )
        if event is not None:
            event.ended_at = exit_event.ended_at
            event.duration_ms = round((event.ended_at - event.started_at) * 1000, 2)
            event.status = NodeStatus.ERROR.value
            event.error = exit_event.error
        self._emit(exit_event)

    def note(self, node: str, message: str, status: str = NodeStatus.OK.value) -> TraceEvent:
        event = TraceEvent(
            seq=self._next_seq(),
            node=node,
            phase="note",
            status=status,
            ended_at=time.time(),
            message=message,
        )
        self._emit(event)
        return event

    # ------------------------------------------------------------------
    # Breakpoints
    # ------------------------------------------------------------------
    def set_breakpoint(self, node: str, mode: BreakpointMode | str) -> BreakpointMode:
        if isinstance(mode, str):
            mode = BreakpointMode(mode.lower())
        if mode is BreakpointMode.OFF:
            self._breakpoints.pop(node, None)
        else:
            self._breakpoints[node] = mode
        return mode

    def clear_breakpoints(self) -> None:
        self._breakpoints.clear()

    def list_breakpoints(self) -> dict[str, str]:
        return {node: mode.value for node, mode in self._breakpoints.items()}

    def breakpoint_mode(self, node: str) -> BreakpointMode:
        return self._breakpoints.get(node, BreakpointMode.OFF)

    # ------------------------------------------------------------------
    # State snapshots
    # ------------------------------------------------------------------
    def snapshot(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """Render graph state into a compact, JSON-safe, redacted preview."""
        if not state:
            return {}
        return {
            "intent": self._text(state.get("intent")),
            "route_reason": self._text(state.get("route_reason")),
            "final_answer": self._truncate(self._text(state.get("final_answer"))),
            "error": self._text(state.get("error")) or None,
            "messages": self._summarize_messages(state.get("messages")),
            "tool_calls": self._summarize_tools(state.get("tool_calls")),
            "tool_results": self._summarize_tools(state.get("tool_results")),
            "usage": self._redact(state.get("usage") or {}),
        }

    def _summarize_messages(self, messages: Iterable[Any] | None) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for message in list(messages or [])[-12:]:
            role = getattr(message, "type", None) or "unknown"
            content = getattr(message, "content", "")
            entry: dict[str, Any] = {
                "role": role,
                "content": self._truncate(self._text(content)),
            }
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "name": call.get("name"),
                        "args": self._redact(call.get("args", {})),
                    }
                    for call in tool_calls
                ]
            summary.append(entry)
        return summary

    def _summarize_tools(self, items: Iterable[Any] | None) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for item in list(items or [])[:8]:
            if isinstance(item, dict):
                summary.append(self._redact(item))
            else:
                summary.append({"value": self._truncate(self._text(item))})
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _truncate(self, text: str, limit: int | None = None) -> str:
        limit = limit or self.max_preview_chars
        if len(text) <= limit:
            return text
        return f"{text[:limit]}... [truncated {len(text) - limit} chars]"

    def _redact(self, value: Any) -> Any:
        """Mask anything that looks like a secret, recursively."""
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if any(hint in str(key).lower() for hint in _SECRET_HINTS):
                    redacted[key] = "***"
                else:
                    redacted[key] = self._redact(item)
            return redacted
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            return self._redact_text(value)
        return value

    def _redact_text(self, text: str) -> str:
        settings = get_settings()
        key = settings.gemini_api_key
        if key and key in text:
            text = text.replace(key, "***")
        return self._truncate(text)

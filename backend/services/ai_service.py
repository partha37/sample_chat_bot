"""Service layer that bridges the API routes to the AI layer."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ai_layer.debugger import BreakpointMode
from ai_layer.graph import ChatBotGraph, ChatResult, StreamEvent

logger = logging.getLogger(__name__)


class AIService:
    """Thin, well-defined facade over :class:`ai_layer.graph.ChatBotGraph`."""

    def __init__(self, graph: ChatBotGraph | None = None) -> None:
        self.graph = graph or ChatBotGraph()
        self.debugger = self.graph.debugger

    @property
    def client(self):
        """The underlying Gemini client."""
        return self.graph.client

    @property
    def configured(self) -> bool:
        """True when a Gemini API key is available."""
        return self.graph.client.available

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    async def chat(
        self,
        message: str,
        session_id: str = "default",
        history: list[dict[str, str]] | None = None,
        include_debug: bool = True,
    ) -> ChatResult:
        return await self.graph.chat(
            message=message,
            session_id=session_id,
            history=history,
            include_debug=include_debug,
        )

    async def chat_stream(
        self,
        message: str,
        session_id: str = "default",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self.graph.chat_stream(
            message=message, session_id=session_id, history=history
        ):
            yield event

    async def resume(self, session_id: str = "default", value: Any = None) -> ChatResult:
        """Resume a graph run that is paused at a breakpoint."""
        return await self.graph.resume(session_id=session_id, value=value)

    # ------------------------------------------------------------------
    # Debug surface
    # ------------------------------------------------------------------
    def list_traces(self, session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self.debugger.list_traces(session_id=session_id, limit=limit)

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self.debugger.get_trace(trace_id)

    def clear_traces(self) -> int:
        return self.debugger.clear_traces()

    def list_breakpoints(self) -> dict[str, str]:
        return self.debugger.list_breakpoints()

    def set_breakpoint(self, node: str, mode: BreakpointMode | str) -> BreakpointMode:
        return self.debugger.set_breakpoint(node, mode)

    def clear_breakpoints(self) -> None:
        self.debugger.clear_breakpoints()

    def graph_structure(self) -> dict[str, Any]:
        return self.graph.graph_structure()

    def checkpoint_history(self, session_id: str) -> list[dict[str, Any]]:
        return self.graph.checkpoint_history(session_id)

    def status(self) -> dict[str, Any]:
        return self.graph.status()

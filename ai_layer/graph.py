"""Builds and drives the LangGraph chat graph.

Responsibilities:
- wire the nodes in :mod:`ai_layer.nodes` into a :class:`StateGraph`
- instrument every node with the debugger (traces, breakpoints, snapshots)
- expose a blocking `chat()` and a streaming `chat_stream()` API
- expose graph metadata + checkpoint history for the debug UI
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph

from ai_layer.client import GeminiClient, GeminiQuotaError, extract_text
from ai_layer.config import AISettings, get_settings
from ai_layer.debugger import BreakpointMode, GraphDebugger, NodeStatus, TraceEvent
from ai_layer.nodes import act, finalize, ingest, respond, router
from ai_layer.state import ChatState

logger = logging.getLogger(__name__)

NODE_DESCRIPTIONS: dict[str, str] = {
    "ingest": "Normalises the user input and prepares the message history.",
    "router": "Asks Gemini to answer, or to call a tool when one would help.",
    "act": "Executes the tools requested by the router.",
    "respond": "Produces the final reply (reuses the router's answer if no tool ran).",
    "finalize": "Packages the answer before it leaves the graph.",
}


@dataclass
class ChatResult:
    """Outcome of a single chat turn."""

    answer: str
    trace_id: str
    session_id: str
    thread_id: str
    intent: str = ""
    route_reason: str = ""
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    paused: bool = False
    interrupt: dict[str, Any] | None = None
    active_model: str | None = None
    debug: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "intent": self.intent,
            "route_reason": self.route_reason,
            "tool_results": self.tool_results,
            "usage": self.usage,
            "error": self.error,
            "paused": self.paused,
            "interrupt": self.interrupt,
            "active_model": self.active_model,
            "debug": self.debug,
        }


@dataclass
class StreamEvent:
    """A single event emitted while streaming a chat turn."""

    type: str  # token | node | trace | done | error | interrupt
    data: dict[str, Any] = field(default_factory=dict)


class ChatBotGraph:
    """Owns the compiled LangGraph and the debugger attached to it."""

    def __init__(
        self,
        settings: AISettings | None = None,
        client: GeminiClient | None = None,
        debugger: GraphDebugger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or GeminiClient(self.settings)
        self.debugger = debugger or GraphDebugger()
        self.checkpointer = MemorySaver()
        self.graph = self._build()
        self._seeded_threads: set[str] = set()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build(self):
        builder = StateGraph(ChatState)

        builder.add_node("ingest", self._wrap("ingest", ingest))
        builder.add_node("router", self._wrap("router", router))
        builder.add_node("act", self._wrap("act", act))
        builder.add_node("respond", self._wrap("respond", respond))
        builder.add_node("finalize", self._wrap("finalize", finalize))

        builder.add_edge(START, "ingest")
        builder.add_edge("ingest", "router")
        builder.add_conditional_edges(
            "router",
            self._route_after_router,
            {"act": "act", "respond": "respond"},
        )
        builder.add_edge("act", "respond")
        builder.add_edge("respond", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _route_after_router(self, state: ChatState) -> str:
        """Conditional edge: run the tools only when the router asked for them."""
        if state.get("intent") == "tool" and state.get("tool_calls"):
            return "act"
        return "respond"

    def _wrap(self, name: str, fn):
        """Wrap a node with breakpoint handling and debugger instrumentation."""

        async def wrapped(state: ChatState) -> dict[str, Any]:
            mode = self.debugger.breakpoint_mode(name)

            if mode is BreakpointMode.SKIP:
                self.debugger.note(
                    name, "skipped (breakpoint: skip)", status=NodeStatus.SKIPPED.value
                )
                return {}

            if mode is BreakpointMode.BREAK:
                payload = {
                    "node": name,
                    "reason": "breakpoint",
                    "state": self.debugger.snapshot(state),
                }
                self.debugger.note(name, "paused at breakpoint", status=NodeStatus.PAUSED.value)
                resumed = self._interrupt(payload)
                if isinstance(resumed, dict) and resumed.get("skip"):
                    self.debugger.note(
                        name, "skipped after resume", status=NodeStatus.SKIPPED.value
                    )
                    return {}

            return await fn(state, client=self.client, debugger=self.debugger)

        wrapped.__name__ = f"{name}_node"
        return wrapped

    @staticmethod
    def _interrupt(payload: dict[str, Any]) -> Any:
        """Pause the graph, tolerating environments without a checkpointer.

        `interrupt()` signals a pause by raising `GraphInterrupt`, which the
        LangGraph runtime catches.  That exception must never be swallowed
        here or the graph would run straight through the breakpoint.
        """
        try:
            from langgraph.types import interrupt
        except ImportError as exc:  # pragma: no cover - dependency issue
            logger.warning("interrupt() unavailable (%s) - continuing without pausing", exc)
            return None

        try:
            return interrupt(payload)
        except GraphInterrupt:
            raise  # this is the pause signal - let the runtime handle it
        except Exception as exc:  # noqa: BLE001
            logger.warning("interrupt() failed (%s) - continuing without pausing", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _thread_id(self, session_id: str) -> str:
        return f"thread-{session_id}"

    def _config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": self._thread_id(session_id)}}

    def _build_input(
        self, message: str, history: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Build the graph input, seeding the checkpointed history once."""
        thread = self._thread_id("")
        incoming: list[BaseMessage] = []
        if history and not self._seeded_threads:
            for item in history:
                role = (item.get("role") or "user").lower()
                content = item.get("content") or ""
                if not content:
                    continue
                if role == "assistant":
                    incoming.append(AIMessage(content=content))
                else:
                    incoming.append(HumanMessage(content=content))
        incoming.append(HumanMessage(content=message))
        return {"user_input": message, "messages": incoming}

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        history: list[dict[str, str]] | None = None,
        include_debug: bool = True,
    ) -> ChatResult:
        """Run one turn of the graph and return the result."""
        trace_id = self.debugger.begin_trace(session_id, message)
        config = self._config(session_id)
        inputs = self._build_input(message, history)
        self._seeded_threads.add(session_id)

        status = "ok"
        error: str | None = None
        quota_message: str | None = None
        try:
            state = await self.graph.ainvoke(inputs, config=config)
            paused, interrupt_payload = self._detect_pause(state)
            state = self._unwrap_interrupt(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Graph execution failed")
            status, error = "error", f"{type(exc).__name__}: {exc}"
            state = {}
            paused, interrupt_payload = False, None
            if isinstance(exc, GeminiQuotaError):
                quota_message = (
                    "Gemini is rate limiting this key right now. The free tier allows "
                    f"only a few requests per minute - please retry in a moment. {exc}"
                )
        answer = str(state.get("final_answer") or "")
        if paused:
            answer = answer or "(paused at breakpoint - inspect the state and resume)"
        if error and not answer:
            answer = quota_message or (
                "Something went wrong while running the graph. Check the debug trace."
            )
        self.debugger.end_trace(
            trace_id,
            status="paused" if paused else status,
            final_answer=answer,
            error=error,
        )

        return ChatResult(
            answer=answer,
            trace_id=trace_id,
            session_id=session_id,
            thread_id=self._thread_id(session_id),
            intent=str(state.get("intent") or ""),
            route_reason=str(state.get("route_reason") or ""),
            tool_results=list(state.get("tool_results") or []),
            usage=dict(state.get("usage") or {}),
            error=error,
            paused=paused,
            interrupt=interrupt_payload,
            active_model=self.client.active_model,
            debug=self.debugger.get_trace(trace_id) if include_debug else None,
        )

    async def chat_stream(
        self,
        message: str,
        session_id: str = "default",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn, yielding token and debugger events as they happen."""
        trace_id = self.debugger.begin_trace(session_id, message)
        config = self._config(session_id)
        inputs = self._build_input(message, history)
        self._seeded_threads.add(session_id)

        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        collected = {"answer": "", "paused": False, "error": None}

        def on_event(event: TraceEvent) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                StreamEvent(
                    type="trace",
                    data={"trace_id": trace_id, "event": event.to_dict()},
                ),
            )

        self.debugger.add_listener(on_event)

        async def producer() -> None:
            try:
                async for mode, chunk in self.graph.astream(
                    inputs, config=config, stream_mode=["messages", "updates"]
                ):
                    if mode == "messages":
                        message_chunk, metadata = chunk
                        token = extract_text(getattr(message_chunk, "content", ""))
                        node_name = metadata.get("langgraph_node", "") if metadata else ""
                        if token:
                            # The `router` node answers directly when no tool is
                            # needed, so its tokens are part of the reply; the
                            # `respond` node produces the text after a tool ran.
                            if node_name in ("", "router", "respond"):
                                collected["answer"] += token
                            queue.put_nowait(
                                StreamEvent(
                                    type="token",
                                    data={
                                        "trace_id": trace_id,
                                        "token": token,
                                        "node": node_name,
                                    },
                                )
                            )
                    elif mode == "updates":
                        for node_name, node_update in (chunk or {}).items():
                            if not isinstance(node_update, dict):
                                continue
                            final_answer = node_update.get("final_answer")
                            if final_answer:
                                collected["answer"] = str(final_answer)
                            queue.put_nowait(
                                StreamEvent(
                                    type="node",
                                    data={
                                        "trace_id": trace_id,
                                        "node": node_name,
                                        "update": {
                                            key: value
                                            for key, value in node_update.items()
                                            if key
                                            in {
                                                "intent",
                                                "route_reason",
                                                "final_answer",
                                                "tool_results",
                                                "error",
                                            }
                                        },
                                    },
                                )
                            )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming failed")
                collected["error"] = f"{type(exc).__name__}: {exc}"
                queue.put_nowait(
                    StreamEvent(
                        type="error",
                        data={"trace_id": trace_id, "error": collected["error"]},
                    )
                )
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(producer())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            self.debugger.remove_listener(on_event)
            if not task.done():
                task.cancel()

        answer = collected["answer"]
        status = "error" if collected["error"] else "ok"

        # Safety net: read the authoritative state from the checkpointer so the
        # `done` event always carries the answer, even if the token stream or
        # the update stream did not surface it.
        if not answer:
            try:
                snapshot = await self.graph.aget_state(config)
                final = (snapshot.values or {}).get("final_answer")
                if final:
                    answer = str(final)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not read final state from checkpointer: %s", exc)

        self.debugger.end_trace(
            trace_id,
            status=status,
            final_answer=answer,
            error=collected["error"],
        )

        trace = self.debugger.get_trace(trace_id) or {}
        yield StreamEvent(
            type="done",
            data={
                "trace_id": trace_id,
                "session_id": session_id,
                "answer": answer,
                "paused": trace.get("status") == "paused",
                "active_model": self.client.active_model,
                "trace": trace,
            },
        )

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    async def resume(self, session_id: str = "default", value: Any = None) -> ChatResult:
        """Resume a graph run that paused at a breakpoint.

        `value` is handed to the paused node.  ``{"skip": True}`` makes the
        node short-circuit instead of executing.
        """
        from langgraph.types import Command

        config = self._config(session_id)
        trace_id = self.debugger.begin_trace(session_id, f"[resume] {value}")

        status = "ok"
        error: str | None = None
        try:
            state = await self.graph.ainvoke(Command(resume=value), config=config)
            paused, interrupt_payload = self._detect_pause(state)
            state = self._unwrap_interrupt(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Resume failed")
            status, error = "error", f"{type(exc).__name__}: {exc}"
            state = {}
            paused, interrupt_payload = False, None

        answer = str(state.get("final_answer") or "")
        if paused:
            answer = answer or "(still paused at breakpoint)"
        if error and not answer:
            answer = "Could not resume the graph. Check the debug trace for details."

        self.debugger.end_trace(
            trace_id,
            status="paused" if paused else status,
            final_answer=answer,
            error=error,
        )

        return ChatResult(
            answer=answer,
            trace_id=trace_id,
            session_id=session_id,
            thread_id=self._thread_id(session_id),
            intent=str(state.get("intent") or ""),
            route_reason=str(state.get("route_reason") or ""),
            tool_results=list(state.get("tool_results") or []),
            usage=dict(state.get("usage") or {}),
            error=error,
            paused=paused,
            interrupt=interrupt_payload,
            debug=self.debugger.get_trace(trace_id),
        )

    @staticmethod
    def _unwrap_interrupt(state: Any) -> dict[str, Any]:
        if not isinstance(state, dict):
            return {}
        return {key: value for key, value in state.items() if key != "__interrupt__"}

    @staticmethod
    def _detect_pause(state: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        raw = state.get("__interrupt__") if isinstance(state, dict) else None
        if not raw:
            return False, None
        payload = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        value = getattr(payload, "value", payload)
        return True, value if isinstance(value, dict) else {"value": str(value)}

    def graph_structure(self) -> dict[str, Any]:
        """Node/edge metadata used by the frontend to draw the flow."""
        return {
            "nodes": [
                {"id": name, "description": desc, "type": "node"}
                for name, desc in NODE_DESCRIPTIONS.items()
            ],
            "edges": [
                {"from": "START", "to": "ingest"},
                {"from": "ingest", "to": "router"},
                {"from": "router", "to": "act", "label": "intent == tool", "conditional": True},
                {"from": "router", "to": "respond", "label": "intent == chat", "conditional": True},
                {"from": "act", "to": "respond"},
                {"from": "respond", "to": "finalize"},
                {"from": "finalize", "to": "END"},
            ],
        }

    def checkpoint_history(self, session_id: str) -> list[dict[str, Any]]:
        """LangGraph checkpoint history for a session (step-by-step replay)."""
        config = self._config(session_id)
        history: list[dict[str, Any]] = []
        try:
            for snapshot in self.graph.get_state_history(config):
                history.append(
                    {
                        "step": len(history) + 1,
                        "next": list(snapshot.next or []),
                        "created_at": snapshot.created_at,
                        "checkpoint_id": (snapshot.config or {}).get("configurable", {}).get(
                            "checkpoint_id", ""
                        ),
                        "messages": len(snapshot.values.get("messages", []) or []),
                        "final_answer": str(snapshot.values.get("final_answer", "") or "")[:400],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read checkpoint history: %s", exc)
        return history

    def status(self) -> dict[str, Any]:
        return {
            "client": self.client.status(),
            "breakpoints": self.debugger.list_breakpoints(),
            "traces": len(self.debugger.list_traces(limit=10_000)),
        }

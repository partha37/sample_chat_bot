"""The LangGraph state contract.

`ChatState` is the single dictionary that flows between every node of the
graph.  Keys prefixed with `trace_` are debugger-owned and are appended to
(not overwritten) as the graph executes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict, total=False):
    """State threaded through every node of the chat graph."""

    # --- Conversation -----------------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    """Full conversation history (LangChain message objects)."""

    user_input: str
    """The raw text of the current turn."""

    session_id: str
    """Stable id for the caller, used for checkpointing/threads."""

    thread_id: str
    """LangGraph checkpoint thread id (one conversation == one thread)."""

    # --- Routing ----------------------------------------------------------
    intent: str
    """Router decision: `chat` | `tool` | `system`."""

    route_reason: str
    """Short explanation of why the router picked that intent."""

    direct_answer: str
    """Answer produced by the router itself, reused by the `respond` node.

    Routing and answering share a single model call, so a plain chat turn
    costs one Gemini request instead of two (the free tier only allows a
    handful of requests per minute).
    """

    tool_calls: list[dict[str, Any]]
    """Tool invocations the router asked for."""

    tool_results: list[dict[str, Any]]
    """Results produced by the `act` node."""

    # --- Output -----------------------------------------------------------
    final_answer: str
    """The assistant reply handed back to the API layer."""

    error: str | None
    """Populated when a node failed; the graph short-circuits to END."""

    # --- Debugger ---------------------------------------------------------
    trace_events: Annotated[list[dict[str, Any]], operator.add]
    """Debugger events, appended by every node (see ai_layer.debugger)."""
    usage: dict[str, Any]
    """Token usage aggregated across the turn, when the model reports it."""


class DebuggerSignal:
    """Sentinel helpers used to short-circuit a graph in debug mode."""

    SKIP = "__debug_skip__"
    BREAK = "__debug_break__"

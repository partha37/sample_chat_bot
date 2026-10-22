"""AI layer: a LangGraph state machine that powers the sample chat bot.

Public surface:
    from ai_layer import ChatBotGraph, GraphDebugger, get_settings

The AI layer is intentionally framework-free (no FastAPI/Node imports) so it
can be unit tested, reused in a CLI, or dropped into another backend.
"""

from ai_layer.config import AISettings, get_settings
from ai_layer.debugger import BreakpointMode, GraphDebugger, TraceEvent
from ai_layer.graph import ChatBotGraph, ChatResult, StreamEvent
from ai_layer.state import ChatState

__all__ = [
    "AISettings",
    "BreakpointMode",
    "ChatBotGraph",
    "ChatResult",
    "ChatState",
    "GraphDebugger",
    "StreamEvent",
    "TraceEvent",
    "get_settings",
]

"""The nodes that make up the chat graph.

Flow::

    START -> ingest -> router -> (act -> respond | respond) -> finalize -> END

Every node is wrapped by :class:`ai_layer.graph.ChatBotGraph` which injects the
Gemini client and records debugger events around the call.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ai_layer.client import GeminiClient, GeminiQuotaError, extract_text
from ai_layer.debugger import GraphDebugger
from ai_layer.state import ChatState
from ai_layer.tools import ALL_TOOLS, TOOLS_BY_NAME

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """You are the assistant that answers the user's message.

- If one of the available tools would give a more accurate answer, CALL that tool
  and write no prose before the call.
- Otherwise answer the message directly and completely.

Be brief and helpful. Do not mention that you routed anything.

Available tools:
{tool_list}

User message:
{user_input}
"""

_TOOL_LIST = "\n".join(f"- {tool_.name}: {tool_.description}" for tool_ in ALL_TOOLS)


def _build_messages(state: ChatState) -> list[Any]:
    """The message list handed to the model (system prompt + history)."""
    from langchain_core.messages import SystemMessage

    from ai_layer.client import SYSTEM_PROMPT

    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(state.get("messages") or [])
    return messages

# Heuristic patterns used when no API key is configured.
_MATH_PATTERN = re.compile(r"^[-+]?[\d\s.,()+\-*/%^]+$")
_EXPR_CANDIDATE = re.compile(r"[\d()\s+\-*/%^.,]+")
_MATH_OPERATORS = "+-*/%^"
_TIME_HINTS = ("what time", "current time", "what's the time", "what is the date", "today's date")
_STATS_HINTS = ("word count", "character count", "count the words", "how many words")


# ----------------------------------------------------------------------
# Node 1 - ingest
# ----------------------------------------------------------------------
async def ingest(state: ChatState, *, client: GeminiClient, debugger: GraphDebugger) -> dict[str, Any]:
    """Normalise the user input and prepare the conversation."""
    event = debugger.enter_node("ingest", state)
    try:
        raw = (state.get("user_input") or "").strip()

        if not raw:
            # Fall back to the most recent human message in the history.
            for message in reversed(state.get("messages") or []):
                if getattr(message, "type", "") == "human":
                    raw = str(getattr(message, "content", "")).strip()
                    break

        if not raw:
            raise ValueError("The message is empty - type something first.")

        state["user_input"] = raw
        state["error"] = None
        state["intent"] = ""
        state["tool_calls"] = []
        state["tool_results"] = []
        state["direct_answer"] = ""

        debugger.exit_node("ingest", state, event, message=f"Prepared {len(raw)} characters")
        return {
            "user_input": raw,
            "error": None,
            "intent": "",
            "tool_calls": [],
            "tool_results": [],
            "direct_answer": "",
        }
    except Exception as exc:  # noqa: BLE001
        debugger.error_node("ingest", exc, event)
        raise


# ----------------------------------------------------------------------
# Node 2 - router
# ----------------------------------------------------------------------
async def router(state: ChatState, *, client: GeminiClient, debugger: GraphDebugger) -> dict[str, Any]:
    """Decide between answering directly and calling a tool."""
    event = debugger.enter_node("router", state)
    try:
        user_input = state.get("user_input", "")

        if not client.available:
            intent, reason, tool_calls = _heuristic_route(user_input)
            debugger.exit_node(
                "router",
                state,
                event,
                message=f"{intent} ({reason}) - heuristic mode, no API key configured",
            )
            return {"intent": intent, "route_reason": reason, "tool_calls": tool_calls}

        prompt = ROUTER_PROMPT.format(tool_list=_TOOL_LIST, user_input=user_input)
        try:
            # One model call does the routing *and* the answering: if the model
            # answers directly we reuse that text, if it calls a tool we run the
            # tool and let the `respond` node finish the job.  A plain chat
            # turn therefore costs a single Gemini request, which matters a
            # lot on the free tier (a handful of requests per minute).
            response = await client.ainvoke(
                _build_messages(state),
                tools=ALL_TOOLS,
                chain=client.router_chain,
            )
        except Exception as exc:  # noqa: BLE001
            # Gemini is unavailable (rate limited, quota exhausted, network...).
            # Fall back to the local heuristic router so the graph keeps
            # working instead of failing the whole turn.
            intent, reason, tool_calls = _heuristic_route(user_input)
            debugger.error_node("router", exc, event)
            debugger.note("router", f"fell back to the heuristic router: {reason}")
            state["error"] = None
            return {
                "intent": intent,
                "route_reason": f"{reason} (model unavailable: {type(exc).__name__})",
                "tool_calls": tool_calls,
                "direct_answer": "",
                "usage": {},
            }

        tool_calls = []
        for call in getattr(response, "tool_calls", None) or []:
            tool_calls.append(
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("args", {}) or {},
                }
            )

        if tool_calls:
            intent = "tool"
            reason = f"model requested tool(s): {', '.join(c['name'] for c in tool_calls)}"
            direct_answer = ""
        else:
            intent = "chat"
            direct_answer = extract_text(getattr(response, "content", "")).strip()
            reason = "model answered directly"

        usage: dict[str, Any] = {}
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            usage = dict(usage_metadata)

        state["intent"] = intent
        state["route_reason"] = reason
        state["tool_calls"] = tool_calls
        state["direct_answer"] = direct_answer
        state["usage"] = usage

        debugger.exit_node("router", state, event, message=reason)
        return {
            "intent": intent,
            "route_reason": reason,
            "tool_calls": tool_calls,
            "direct_answer": direct_answer,
            "usage": usage,
        }
    except Exception as exc:  # noqa: BLE001
        debugger.error_node("router", exc, event)
        raise


def _extract_expression(text: str) -> str | None:
    """Pull the most likely arithmetic expression out of a sentence."""
    best: str | None = None
    for match in _EXPR_CANDIDATE.finditer(text):
        candidate = match.group().strip(" ,.")
        if len(candidate) < 3:
            continue
        if not any(operator in candidate for operator in _MATH_OPERATORS):
            continue
        if not any(char.isdigit() for char in candidate):
            continue
        if not _MATH_PATTERN.match(candidate):
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


def _heuristic_route(user_input: str) -> tuple[str, str, list[dict[str, Any]]]:
    """Fallback routing used when Gemini is not configured."""
    text = user_input.strip()

    if any(hint in text.lower() for hint in _STATS_HINTS):
        return "tool", "heuristic: text statistics", [
            {"id": "heuristic-stats", "name": "text_stats", "args": {"text": text}}
        ]

    if any(hint in text.lower() for hint in _TIME_HINTS):
        return "tool", "heuristic: current time", [
            {"id": "heuristic-time", "name": "current_datetime", "args": {}}
        ]

    expression = _extract_expression(text)
    if expression:
        return "tool", "heuristic: arithmetic", [
            {"id": "heuristic-calc", "name": "calculator", "args": {"expression": expression}}
        ]

    return "chat", "heuristic: direct answer", []


# ----------------------------------------------------------------------
# Node 3 - act
# ----------------------------------------------------------------------
async def act(state: ChatState, *, client: GeminiClient, debugger: GraphDebugger) -> dict[str, Any]:
    """Execute the tool calls requested by the router."""
    event = debugger.enter_node("act", state)
    try:
        calls = state.get("tool_calls") or []
        results: list[dict[str, Any]] = []

        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            tool_ = TOOLS_BY_NAME.get(name)

            if tool_ is None:
                output = f"Unknown tool '{name}'."
                ok = False
            else:
                try:
                    output = await tool_.ainvoke(args)
                    ok = True
                except Exception as exc:  # noqa: BLE001
                    output = f"Tool '{name}' failed: {exc}"
                    ok = False

            results.append(
                {
                    "id": call.get("id"),
                    "name": name,
                    "args": args,
                    "output": str(output),
                    "ok": ok,
                }
            )

        state["tool_results"] = results
        summary = ", ".join(f"{r['name']}={r['output']}" for r in results) or "no tools executed"
        debugger.exit_node("act", state, event, message=f"Executed {len(results)} tool(s): {summary}")
        return {"tool_results": results}
    except Exception as exc:  # noqa: BLE001
        debugger.error_node("act", exc, event)
        raise


# ----------------------------------------------------------------------
# Node 4 - respond
# ----------------------------------------------------------------------
async def respond(state: ChatState, *, client: GeminiClient, debugger: GraphDebugger) -> dict[str, Any]:
    """Generate the final assistant reply."""
    event = debugger.enter_node("respond", state)
    try:
        if not client.available:
            tool_results = state.get("tool_results") or []
            tool_block = ""
            if tool_results:
                lines = [
                    f"- {result['name']}({result.get('args', {})}) -> {result.get('output', '')}"
                    for result in tool_results
                ]
                tool_block = "\n\nTools I ran:\n" + "\n".join(lines)

            answer = (
                "Gemini is not configured yet, so I am answering locally.\n\n"
                f'You said: "{state.get("user_input", "")}"'
                + tool_block
                + "\n\n"
                + client.settings.missing_key_hint()
            )
            debugger.exit_node("respond", state, event, message="offline fallback answer")
            return {"final_answer": answer}

        from langchain_core.messages import AIMessage, ToolMessage

        # If the router already answered (no tool was needed) we reuse that
        # answer instead of paying for a second model call.
        direct_answer = (state.get("direct_answer") or "").strip()
        if direct_answer and not (state.get("tool_results") or []):
            answer = direct_answer
            debugger.exit_node(
                "respond",
                state,
                event,
                message=f"Reused the router's answer ({len(answer)} characters)",
            )
            return {"final_answer": answer, "usage": dict(state.get("usage") or {})}

        messages = _build_messages(state)

        tool_results = state.get("tool_results") or []
        if tool_results:
            # Replay the router's tool call + the tool output so Gemini can
            # reason over it.
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": result["name"],
                            "args": result.get("args", {}),
                            "id": result.get("id") or f"call_{result['name']}",
                            "type": "tool_call",
                        }
                        for result in tool_results
                    ],
                )
            )
            for result in tool_results:
                messages.append(
                    ToolMessage(
                        content=str(result.get("output", "")),
                        tool_call_id=str(result.get("id") or f"call_{result['name']}"),
                        name=result.get("name", "tool"),
                    )
                )

        # Stream the model instead of awaiting a single response: the
        # `on_llm_new_token` callbacks are what LangGraph's "messages" stream
        # mode picks up, so this is what makes tokens reach the browser.
        parts: list[str] = []
        usage: dict[str, Any] = {}
        try:
            async for chunk in client.astream(messages):
                text = extract_text(getattr(chunk, "content", ""))
                if text:
                    parts.append(text)
                chunk_usage = getattr(chunk, "usage_metadata", None)
                if chunk_usage:
                    usage = dict(chunk_usage)
        except GeminiQuotaError as exc:
            debugger.error_node("respond", exc, event)
            state["error"] = str(exc)
            return {
                "final_answer": (
                    "Gemini is rate limiting this key right now (the free tier allows "
                    f"a few requests per minute). {exc}"
                ),
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            debugger.error_node("respond", exc, event)
            state["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, GeminiQuotaError):
                message = (
                    "Gemini is rate limiting this key right now. "
                    f"{exc}"
                )
            else:
                message = (
                    f"I could not reach Gemini just now ({type(exc).__name__}). "
                    "Check the debug trace for the details."
                )
            return {
                "final_answer": message,
                "error": f"{type(exc).__name__}: {exc}",
            }

        answer = "".join(parts).strip()
        if not answer:
            answer = "(the model returned an empty response)"

        state["final_answer"] = answer
        state["usage"] = usage

        debugger.exit_node(
            "respond",
            state,
            event,
            message=f"Generated {len(answer)} characters",
        )
        return {"final_answer": answer, "usage": usage}
    except Exception as exc:  # noqa: BLE001
        debugger.error_node("respond", exc, event)
        raise


# ----------------------------------------------------------------------
# Node 5 - finalize
# ----------------------------------------------------------------------
async def finalize(state: ChatState, *, client: GeminiClient, debugger: GraphDebugger) -> dict[str, Any]:
    """Normalise the outcome before it leaves the graph."""
    event = debugger.enter_node("finalize", state)
    try:
        answer = (state.get("final_answer") or "").strip()
        if not answer and not state.get("error"):
            answer = "I could not produce an answer for that request."
        state["final_answer"] = answer
        debugger.exit_node("finalize", state, event, message="finalised")
        return {"final_answer": answer}
    except Exception as exc:  # noqa: BLE001
        debugger.error_node("finalize", exc, event)
        raise

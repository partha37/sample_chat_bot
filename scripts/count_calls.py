"""Verify how many Gemini requests a single chat turn actually costs.

The Gemini free tier allows only a handful of requests per minute per model,
so the number of model calls per turn matters a lot.

Usage:  .\\.venv\\Scripts\\python scripts\\count_calls.py "Hello there!"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).resolve().parent.parent))

from ai_layer import ChatBotGraph
from ai_layer import client as client_module

MESSAGE = sys.argv[1] if len(sys.argv) > 1 else "Hello there!"

# Count every request that leaves for Google, and whether it succeeded.
attempts: list[tuple[str, bool]] = []
_original_ainvoke = client_module.GeminiClient.ainvoke
_original_astream = client_module.GeminiClient.astream


async def counting_ainvoke(self, messages, tools=None, chain=None, **kwargs):  # type: ignore[no-untyped-def]
    try:
        result = await _original_ainvoke(self, messages, tools=tools, chain=chain, **kwargs)
    except Exception:
        attempts.append((self._active_model or "?", False))
        raise
    attempts.append((self._active_model or "?", True))
    return result


async def counting_astream(self, messages, tools=None, chain=None, **kwargs):  # type: ignore[no-untyped-def]
    try:
        async for chunk in _original_astream(
            self, messages, tools=tools, chain=chain, **kwargs
        ):
            yield chunk
    except Exception:
        attempts.append((self._active_model or "?", False))
        raise
    attempts.append((self._active_model or "?", True))


client_module.GeminiClient.ainvoke = counting_ainvoke  # type: ignore[method-assign]
client_module.GeminiClient.astream = counting_astream  # type: ignore[method-assign]


async def main() -> None:
    graph = ChatBotGraph()
    result = await graph.chat(MESSAGE, session_id="count-1")

    print(f"message   : {MESSAGE}")
    print(f"answer    : {result.answer[:200]}")
    print(f"intent    : {result.intent}  ({result.route_reason})")
    print(f"tools     : {len(result.tool_results)}")
    print(f"model     : {result.active_model}")
    print()
    successful = [m for m, ok in attempts if ok]
    print(f"Gemini requests this turn : {len(attempts)}  ({len(successful)} successful)")
    for index, (model, ok) in enumerate(attempts, 1):
        print(f"   {index}. {model:<22} {'ok' if ok else 'failed (quota/429)'}")


asyncio.run(main())

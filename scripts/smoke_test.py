"""End-to-end smoke test for the whole stack.

Usage::

    .\\.venv\\Scripts\\python scripts\\smoke_test.py

It exercises the backend directly (no browser needed) and checks the chat
endpoint, the SSE stream, the debugger endpoints and the breakpoint flow.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_URL = "http://127.0.0.1:8000"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    results.append((condition, label))
    mark = PASS if condition else FAIL
    print(f"  [{mark}] {label}" + (f"  -> {detail}" if detail and not condition else ""))
    return condition


def warn(label: str, detail: str = "") -> None:
    """Informational check - never fails the run."""
    print(f"  [\033[33mWARN\033[0m] {label}" + (f"  -> {detail}" if detail else ""))
    results.append((True, label))


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def main() -> int:
    started = time.time()
    session = f"smoke-{int(time.time())}"
    # Generous timeouts: a live Gemini turn can take 30-90s when the primary
    # model is busy and the client has to walk the model fallback chain.
    client = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(240.0, connect=10.0))

    section("1. Backend reachable")
    try:
        health = client.get("/api/health")
    except Exception as exc:  # noqa: BLE001
        print(f"  [{FAIL}] backend not reachable at {BASE_URL}: {exc}")
        print("\nStart the backend first:  .\\scripts\\dev.ps1")
        return 1
    check(health.status_code == 200, "GET /api/health returns 200", health.text[:200])
    ai = health.json().get("ai", {}).get("client", {})
    if ai.get("configured"):
        check(True, "GEMINI_API_KEY is configured")
    else:
        warn(
            "GEMINI_API_KEY is not set - running in offline mode",
            "add it to .env to get real Gemini answers",
        )
    print(f"       model: {ai.get('model')}  key: {ai.get('api_key')}")

    section("2. Chat (blocking)")
    response = client.post(
        "/api/chat",
        json={"message": "What is 12 * (3 + 4)?", "session_id": session, "debug": True},
    )
    check(response.status_code == 200, "POST /api/chat returns 200", response.text[:300])
    payload = response.json()
    check(bool(payload.get("answer")), "answer is non-empty", payload.get("answer", "")[:120])
    check(payload.get("intent") in {"chat", "tool"}, "router produced an intent", str(payload.get("intent")))
    check(bool(payload.get("debug")), "debug trace is attached")
    check(
        payload.get("tool_results") is not None and isinstance(payload.get("tool_results"), list),
        "tool_results is a list",
    )
    if payload.get("error"):
        warn(f"the turn hit an error: {str(payload.get('error'))[:120]}")

    section("3. Debugger endpoints")
    trace_id = payload.get("trace_id")
    traces = client.get("/api/debug/traces", params={"session_id": session})
    check(traces.status_code == 200, "GET /api/debug/traces returns 200")
    check(traces.json().get("count", 0) >= 1, "at least one trace was recorded")

    if trace_id:
        trace = client.get(f"/api/debug/traces/{trace_id}")
        check(trace.status_code == 200, "GET /api/debug/traces/{id} returns 200")
        events = trace.json().get("events", [])
        nodes = {event.get("node") for event in events}
        check("ingest" in nodes and "respond" in nodes, "trace contains the graph nodes", str(nodes))
        check(
            any(event.get("duration_ms") is not None for event in events),
            "events carry timing information",
        )

    graph = client.get("/api/debug/graph")
    check(graph.status_code == 200, "GET /api/debug/graph returns 200")
    check(len(graph.json().get("nodes", [])) >= 5, "graph exposes the node list")

    requests = client.get("/api/debug/requests")
    check(requests.status_code == 200, "GET /api/debug/requests returns 200")
    check(requests.json().get("count", 0) > 0, "requests were traced by the middleware")

    section("4. SSE streaming")
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "Say hello in one short sentence.", "session_id": session},
        timeout=240.0,
    ) as stream:
        status_ok = stream.status_code == 200
        content_type = stream.headers.get("content-type", "")
        event_names: list[str] = []
        frames: list[str] = []
        for line in stream.iter_lines():
            frames.append(line)
            if line.startswith("event:"):
                event_names.append(line.split(":", 1)[1].strip())

        check(status_ok, "POST /api/chat/stream returns 200", f"status={stream.status_code}")
        check(
            "text/event-stream" in content_type,
            "response is text/event-stream",
            content_type,
        )
        check("open" in event_names, "SSE emits an 'open' event", str(event_names[:8]))
        check("done" in event_names, "SSE emits a 'done' event", str(event_names[:8]))
        check(
            "trace" in event_names,
            "SSE streams live debugger events",
            str(event_names[:8]),
        )

    section("5. Breakpoints (pause / resume)")
    set_bp = client.put("/api/debug/breakpoints/respond", json={"mode": "break"})
    check(set_bp.status_code == 200, "PUT breakpoint on 'respond' returns 200", set_bp.text[:200])

    paused = client.post(
        "/api/chat",
        json={"message": "hello again", "session_id": session, "debug": True},
    ).json()
    check(paused.get("paused") is True, "graph paused at the breakpoint", json.dumps(paused)[:250])
    check(paused.get("interrupt") is not None, "pause payload carries the interrupt info")

    resumed = client.post("/api/chat/resume", json={"session_id": session, "skip": False})
    check(resumed.status_code == 200, "POST /api/chat/resume returns 200", resumed.text[:250])
    check(bool(resumed.json().get("answer")), "resumed run produced an answer")

    client.delete("/api/debug/breakpoints")
    cleared = client.get("/api/debug/breakpoints").json().get("breakpoints", {})
    check(cleared == {}, "breakpoints cleared", str(cleared))

    section("6. Error handling")
    empty = client.post("/api/chat", json={"message": "", "session_id": session})
    check(empty.status_code == 422, "empty message is rejected with 422", str(empty.status_code))

    missing = client.get("/api/debug/traces/does-not-exist")
    check(missing.status_code == 404, "unknown trace returns 404", str(missing.status_code))

    client.close()

    section("Summary")
    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    print(f"  {passed}/{total} checks passed")
    print(f"  elapsed: {time.time() - started:.0f}s")
    if passed != total:
        print("\n  Failed checks:")
        for ok, label in results:
            if not ok:
                print(f"    - {label}")
        return 1
    print("\n  All checks passed.")
    print(
        "  Note: this test spends several Gemini requests. The free tier allows\n"
        "  only ~5 requests/minute and ~20 requests/day per model, so the client\n"
        "  fails over between models and degrades gracefully when they are all\n"
        "  exhausted (the checks above still pass - the answer becomes a clear\n"
        "  'rate limited' message instead of a real reply)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

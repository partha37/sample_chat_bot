"""Inspect the debugger trace of a session (helper for manual verification)."""

from __future__ import annotations

import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SESSION = sys.argv[2] if len(sys.argv) > 2 else "proxy-sse"

with httpx.Client(base_url=BASE, timeout=60.0) as client:
    traces = client.get("/api/debug/traces", params={"session_id": SESSION}).json()["traces"]
    if not traces:
        print(f"no traces for session '{SESSION}'")
        raise SystemExit(0)

    trace = traces[0]
    print(f"trace   : {trace['trace_id']}")
    print(f"status  : {trace['status']}   duration: {trace['duration_ms']} ms")
    print(f"input   : {trace['user_input']}")
    print()
    for event in trace["events"]:
        message = (event.get("message") or "")[:95]
        duration = event.get("duration_ms")
        duration_text = f"{duration:.1f}ms" if duration is not None else "-"
        print(
            f"  {event['node']:<9} {event['phase']:<6} {event['status']:<8} "
            f"{duration_text:>9}  {message}"
        )

    print()
    print("answer  :", trace["final_answer"][:300])

    tools = [e for e in trace["events"] if e["node"] == "act"]
    if tools:
        print()
        print("tool node output:")
        for event in tools:
            print(json.dumps(event.get("output_preview", {}), indent=2)[:800])

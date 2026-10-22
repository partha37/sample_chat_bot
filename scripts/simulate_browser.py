"""Simulate exactly what the browser does when it consumes /api/chat/stream.

Mirrors frontend/public/js/app.js: same SSE frame parsing, same node filter,
same "(no response)" fallback.  Run it to confirm the UI renders a real reply.

Usage:  .\\.venv\\Scripts\\python scripts\\simulate_browser.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRONTEND = "http://localhost:3000"
MESSAGES = sys.argv[1:] or ["Hello there!"]


def simulate(message: str, session: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"user: {message}")
    print("=" * 70)

    bubble = ""          # what the user would see in the chat bubble
    last_answer = ""     # the `done` event's answer
    token_count = 0
    trace_events = 0

    with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        with client.stream(
            "POST",
            f"{FRONTEND}/api/chat/stream",
            json={"message": message, "session_id": session, "debug": True},
        ) as stream:
            print(f"http {stream.status_code}  {stream.headers.get('content-type')}")

            name = None
            buffer = ""
            for chunk in stream.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)

                    data_lines = []
                    for line in frame.split("\n"):
                        if line.startswith("event:"):
                            name = line.split(":", 1)[1].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line.split(":", 1)[1].strip())

                    data = {}
                    if data_lines:
                        try:
                            data = json.loads("\n".join(data_lines))
                        except json.JSONDecodeError:
                            data = {}

                    if name == "token":
                        token_count += 1
                        # app.js: reply text comes from the router (direct
                        # answer) or the respond node (after a tool ran)
                        node = data.get("node")
                        if not node or node in ("router", "respond"):
                            bubble += data.get("token", "")
                    elif name == "trace":
                        trace_events += 1
                    elif name == "done":
                        last_answer = data.get("answer", "")

    # app.js fallback: use the done answer when no tokens were rendered
    if not bubble.strip():
        bubble = last_answer or "(no response)"

    print(f"token events   : {token_count}")
    print(f"trace events   : {trace_events}")
    print(f"bubble content : {bubble!r}")
    print(f"done answer    : {last_answer!r}")
    ok = bool(bubble.strip()) and bubble.strip() != "(no response)"
    print(f"RESULT         : {'PASS - the UI will show a real reply' if ok else 'FAIL'}")


def main() -> int:
    for index, message in enumerate(MESSAGES):
        simulate(message, f"sim-{int(time.time())}-{index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

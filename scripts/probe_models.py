"""Probe which Gemini models the configured key can actually use.

Uses the models.list/get endpoints (which do not consume generation quota)
and then sends one tiny request to a candidate to confirm the quota allows it.

Usage:  .\\.venv\\Scripts\\python scripts\\probe_models.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google import genai

# Flash-family models worth probing (cheap + fast).
CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    configured = os.environ.get("GEMINI_MODEL", "(not set)")
    client = genai.Client(api_key=api_key)

    print(f"configured model in .env : {configured}")
    print()

    # 1. Which models does the key know about at all? (no quota used)
    known: set[str] = set()
    try:
        for model in client.models.list():
            known.add(model.name.split("/")[-1])
    except Exception as exc:  # noqa: BLE001
        print(f"models.list() failed: {exc}")

    print(f"{'model':<26} {'listed':<8} {'get()':<10} verdict")
    print("-" * 74)

    for name in CANDIDATES:
        listed = name in known
        exists = False
        try:
            client.models.get(model=name)
            exists = True
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            exists = "404" not in message and "NOT_FOUND" not in message

        if not exists:
            verdict = "RETIRED / not found"
        else:
            verdict = "available (quota unknown)"
        print(f"{name:<26} {str(listed):<8} {str(exists):<10} {verdict}")

    print()
    print("Checking the configured model with one tiny request...")
    try:
        response = await client.aio.models.generate_content(
            model=configured, contents="Reply with exactly: OK"
        )
        print(f"  SUCCESS -> {response.text.strip()[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        if "404" in text or "NOT_FOUND" in text:
            print("  FAILED -> the model no longer exists (404)")
        elif "429" in text or "RESOURCE_EXHAUSTED" in text:
            reset = ""
            if "retry in" in text:
                reset = f" | {text[text.find('retry in') : text.find('retry in') + 50].strip()}"
            print(f"  FAILED -> quota exhausted{reset}")
        else:
            print(f"  FAILED -> {type(exc).__name__}: {text[:140]}")


asyncio.run(main())

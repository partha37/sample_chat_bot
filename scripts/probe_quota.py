"""Check which candidate models still have generation quota left today.

Each successful check costs exactly one request of the free-tier quota.
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

CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-preview",
    "gemini-3-pro-image-preview",
]


async def main() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    usable: list[str] = []

    for name in CANDIDATES:
        try:
            response = await client.aio.models.generate_content(
                model=name, contents="Reply with exactly: OK"
            )
            print(f"  {name:<26} OK        -> {response.text.strip()[:20]!r}")
            usable.append(name)
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if "404" in text or "NOT_FOUND" in text:
                reason = "retired / not found"
            elif "429" in text or "RESOURCE_EXHAUSTED" in text:
                daily = "perday" in text.lower().replace("_", "")
                reason = "quota exhausted (daily)" if daily else "quota exhausted (per minute)"
            else:
                reason = f"{type(exc).__name__}: {text[:60]}"
            print(f"  {name:<26} FAILED    -> {reason}")

    print()
    if usable:
        print("models with quota left:")
        for name in usable:
            print(f"  {name}")
    else:
        print("NO model has quota left - the free tier is exhausted for every model.")


asyncio.run(main())

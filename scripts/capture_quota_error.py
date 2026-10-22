"""Capture the real Gemini quota error text so the parser can be verified."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google import genai

from ai_layer.client import _is_daily_quota, _quota_retry_delay


async def main() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    try:
        await client.aio.models.generate_content(
            model=model, contents="Reply with exactly: OK"
        )
        print("the call succeeded - no quota error to inspect")
        return
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        print("--- repr of the error text ---")
        print(repr(text[:1200]))
        print()
        print("--- parser results ---")
        print("is_daily_quota   :", _is_daily_quota(exc))
        print("retry_after (s)  :", _quota_retry_delay(exc))
        print()
        print("--- 'retry in' occurrences ---")
        index = text.find("retry in")
        while index != -1:
            print("   ", repr(text[index : index + 60]))
            index = text.find("retry in", index + 1)
        print("--- 'retryDelay' occurrences ---")
        index = text.find("retryDelay")
        while index != -1:
            print("   ", repr(text[index : index + 60]))
            index = text.find("retryDelay", index + 1)


asyncio.run(main())

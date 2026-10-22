"""Compare the models.get() metadata for a retired model vs a live one."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google import genai


async def main() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for name in ["gemini-2.0-flash", "gemini-3.8-flash", "not-a-real-model"]:
        print("=" * 70)
        print(name)
        print("=" * 70)
        try:
            model = client.models.get(model=name)
            payload = model.model_dump(exclude_none=True)
            for key, value in payload.items():
                text = str(value)
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"  {key}: {text}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {str(exc)[:200]}")
        print()


asyncio.run(main())

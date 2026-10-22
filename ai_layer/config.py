"""Runtime configuration for the AI layer.

Kept deliberately independent of the backend so the AI layer can be imported
from a script, a test, or a completely different service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

# Load .env once, at import time, without clobbering real environment vars.
load_dotenv(ENV_FILE, override=False)


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AISettings:
    """Everything the AI layer needs to talk to Gemini."""

    gemini_api_key: str
    gemini_model: str
    temperature: float
    max_output_tokens: int
    request_timeout: int
    debug_enabled: bool
    max_trace_history: int
    max_preview_chars: int
    max_retries: int
    retry_base_delay: float
    fallback_models: tuple[str, ...]
    thinking_budget: int | None
    router_model: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def masked_api_key(self) -> str:
        """Never leak the full key into logs or debug payloads."""
        key = self.gemini_api_key
        if not key:
            return "<not set>"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]} (len={len(key)})"

    def missing_key_hint(self) -> str:
        return (
            "GEMINI_API_KEY is not configured.\n"
            f"1. Copy the template:  Copy-Item .env.example .env\n"
            f"2. Put your key in {ENV_FILE} as GEMINI_API_KEY=...\n"
            "3. Get a free key at https://aistudio.google.com/apikey\n"
            "4. Restart the backend."
        )


@lru_cache(maxsize=1)
def get_settings() -> AISettings:
    """Return the process-wide AI settings (cached)."""
    return AISettings(
        gemini_api_key=_env_str("GEMINI_API_KEY"),
        gemini_model=_env_str("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=_env_float("GEMINI_TEMPERATURE", 0.7),
        max_output_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 1024),
        request_timeout=_env_int("GEMINI_REQUEST_TIMEOUT", 60),
        debug_enabled=_env_bool("DEBUG_ENABLED", True),
        max_trace_history=_env_int("MAX_TRACE_HISTORY", 50),
        max_preview_chars=_env_int("MAX_PREVIEW_CHARS", 600),
        max_retries=_env_int("GEMINI_MAX_RETRIES", 2),
        retry_base_delay=_env_float("GEMINI_RETRY_BASE_DELAY", 1.5),
        fallback_models=_env_list("GEMINI_FALLBACK_MODELS"),
        thinking_budget=_env_optional_int("GEMINI_THINKING_BUDGET", 0),
        router_model=_env_str("GEMINI_ROUTER_MODEL"),
    )

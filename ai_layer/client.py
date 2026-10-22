"""Thin wrapper around the Gemini chat model.

Everything Gemini-specific lives here so that swapping providers (or mocking
the model in tests) is a one-file change.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, AsyncIterator

from ai_layer.config import AISettings, get_settings

logger = logging.getLogger(__name__)

# Transient Gemini failures worth retrying (rate limits + gateway hiccups).
_TRANSIENT_STATUS = re.compile(r"\b(429|500|502|503|504)\b")
_TRANSIENT_MARKERS = (
    "unavailable",
    "high demand",
    "rate limit",
    "resource_exhausted",
    "overloaded",
    "internal error",
    "deadline exceeded",
    "connection reset",
    "timed out",
)

# Google tells us exactly how long to wait, e.g. "Please retry in 19h26m42.8s"
# or a RetryInfo field of "retryDelay": "70002s".
_RETRY_IN_RE = re.compile(r"retry in\s+([0-9hms\.\s]+)", re.IGNORECASE)
_RETRY_INFO_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([hms])", re.IGNORECASE)
_QUOTA_MARKERS = (
    "resource_exhausted",
    "quota exceeded",
    "exceeded your current quota",
    "generate_content_free_tier_requests",
    "rate limit",
)
_DAILY_QUOTA_MARKERS = (
    "perday",
    "per_day",
    "requestsperday",
)


def _is_daily_quota(exc: BaseException) -> bool:
    """True when the exhausted quota is the daily one (not the per-minute one)."""
    text = str(exc).lower()
    return any(marker in text for marker in _DAILY_QUOTA_MARKERS)


def _is_retired_model(exc: BaseException) -> bool:
    """True when Google says the model id no longer exists.

    Retired models (e.g. `gemini-2.0-flash`) still show up in
    `models.list()`, but the generation endpoint answers 404.
    """
    text = str(exc)
    return "404" in text or "NOT_FOUND" in text or "no longer available" in text


def _retirement_suggestion(exc: BaseException) -> str:
    """Pull the *replacement* model out of Google's error message.

    The message looks like: "This model models/gemini-2.0-flash is no longer
    available. Please update your code to use models/gemini-3.8-flash ..." -
    the replacement is the one after "use", not the first `models/` mention.
    """
    text = str(exc)
    match = re.search(r"use models/([A-Za-z0-9._\-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _is_transient(exc: BaseException) -> bool:
    """True when the error looks temporary and a retry may succeed."""
    if isinstance(exc, asyncio.CancelledError):
        return False
    text = f"{type(exc).__name__}: {exc}"
    if _TRANSIENT_STATUS.search(text):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def _is_quota_error(exc: BaseException) -> bool:
    """True when the failure is a rate/quota limit rather than a server hiccup."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _quota_retry_delay(exc: BaseException) -> float | None:
    """Extract the wait time Google asks for, in seconds.

    Handles "retryDelay": "70002s" and the human form
    "Please retry in 19h26m42.8s".
    """
    text = str(exc)

    match = _RETRY_INFO_RE.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    match = _RETRY_IN_RE.search(text)
    if match:
        total = 0.0
        for value, unit in _DURATION_PART_RE.findall(match.group(1)):
            multiplier = {"h": 3600, "m": 60, "s": 1}[unit.lower()]
            total += float(value) * multiplier
        if total:
            return total

    return None


class GeminiQuotaError(RuntimeError):
    """Raised when every configured model is rate limited."""

    def __init__(
        self,
        message: str,
        retry_after: float = 30.0,
        daily: bool = False,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.daily = daily


def _uses_fixed_sampling(model_id: str) -> bool:
    """True for Gemini 3.x models, which use fixed sampling defaults.

    Passing `temperature` to those models raises a warning and the value is
    ignored, so it is simply not sent.
    """
    return model_id.strip().lower().startswith("gemini-3")


def extract_text(content: Any) -> str:
    """Flatten a Gemini response payload into plain text.

    Gemini 3.x returns the content as a list of blocks
    (``[{"type": "text", "text": "..."}]``) while older models return a
    plain string.  Both are handled here so the rest of the app only ever
    deals with text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)

SYSTEM_PROMPT = """You are a friendly, concise assistant powering a demo chat bot.

Guidelines:
- Answer directly and helpfully; keep replies to a few short paragraphs.
- Use the provided tools when the question is factual, mathematical, or about
  the current time. Never invent tool results.
- When you use a tool, briefly mention the result in your answer.
- Prefer plain text with light markdown (bullet points, inline code).
"""


class GeminiClient:
    """Lazily builds and caches the `ChatGoogleGenerativeAI` instances.

    Google frequently answers 503 "high demand" for a specific model id.
    To stay usable we keep a *chain* of models: the primary one first, then
    any fallbacks configured in `GEMINI_FALLBACK_MODELS`.  A transient error
    on one model moves on to the next one.
    """

    def __init__(self, settings: AISettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._models: dict[str, Any] = {}
        self._active_model: str | None = None
        # model id -> monotonic timestamp until which the model is skipped
        # (it just hit the per-minute free-tier quota)
        self._cooldown_until: dict[str, float] = {}
        # model id -> replacement suggested by Google ("no longer available")
        self._retired: dict[str, str] = {}

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when an API key is configured."""
        return self.settings.has_api_key

    @property
    def model_name(self) -> str:
        """The primary model id."""
        return self.settings.gemini_model

    @property
    def model_chain(self) -> list[str]:
        """Primary model followed by the configured fallbacks (no duplicates)."""
        return self._build_chain(self.settings.gemini_model)

    @property
    def router_chain(self) -> list[str]:
        """Chain used by the router node.

        The router usually runs on a different model than the answer, which
        means it draws from a *separate* per-minute quota bucket.
        """
        if not self.settings.router_model:
            return self.model_chain
        return self._build_chain(self.settings.router_model)

    def _build_chain(self, primary: str) -> list[str]:
        chain = [primary]
        for fallback in self.settings.fallback_models:
            if fallback not in chain:
                chain.append(fallback)
        return chain

    @property
    def active_model(self) -> str | None:
        """The model that actually served the last request."""
        return self._active_model

    def cooldown_state(self) -> dict[str, int]:
        """Seconds left on each model's rate-limit cooldown (for the UI)."""
        now = time.monotonic()
        return {
            model: max(0, round(until - now))
            for model, until in self._cooldown_until.items()
            if until > now
        }

    def model(self, model_id: str | None = None) -> Any:
        """Return the chat model, or raise a helpful error if not configured."""
        if not self.settings.has_api_key:
            raise RuntimeError(self.settings.missing_key_hint())

        target = model_id or self.settings.gemini_model
        if target in self._models:
            return self._models[target]

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(
                "langchain-google-genai is not installed. "
                "Run: .\\.venv\\Scripts\\pip install -r requirements.txt"
            ) from exc

        logger.info(
            "Initialising Gemini model=%s key=%s",
            target,
            self.settings.masked_api_key,
        )

        # Gemini 3.x uses fixed sampling defaults and warns (or errors) when
        # `temperature` is supplied, so only send it to models that honour it.
        model_kwargs: dict[str, Any] = {
            "model": target,
            "google_api_key": self.settings.gemini_api_key,
            "max_output_tokens": self.settings.max_output_tokens,
            "timeout": self.settings.request_timeout,
            "max_retries": 1,  # our own backoff handles the retries
            "convert_system_message_to_human": False,
        }
        if not _uses_fixed_sampling(target):
            model_kwargs["temperature"] = self.settings.temperature
        if self.settings.thinking_budget is not None:
            model_kwargs["thinking_budget"] = self.settings.thinking_budget

        instance = ChatGoogleGenerativeAI(**model_kwargs)
        self._models[target] = instance
        return instance

    # ------------------------------------------------------------------
    def bind_tools(self, model: Any, tools: list[Any]) -> Any:
        """Bind tools to a model, tolerating models that reject tool binding."""
        try:
            return model.bind_tools(tools)
        except Exception as exc:  # noqa: BLE001 - some models reject tool binding
            logger.warning("Tool binding failed, continuing without tools: %s", exc)
            return model

    # ------------------------------------------------------------------
    async def _iter_models(self, chain: list[str]) -> AsyncIterator[str]:
        """Yield the models to try, skipping ones that just hit their quota.

        The Gemini free tier allows only a handful of requests per minute per
        model.  When a model reports a quota error we put it on a cooldown for
        exactly as long as Google asks, so the next request goes straight to a
        model that still has quota instead of hammering a 429ing one.
        """
        tried: set[str] = set()
        waited = 0.0
        max_wait = 90.0

        while True:
            now = time.monotonic()
            remaining = [
                model
                for model in chain
                if model not in tried and model not in self._retired
            ]
            if not remaining:
                return

            eligible = [m for m in remaining if self._cooldown_until.get(m, 0) <= now]
            if eligible:
                model_id = eligible[0]
            else:
                soonest = min(self._cooldown_until.get(m, now) for m in remaining)
                delay = max(1.0, soonest - now)
                if delay > 60.0:
                    # Waiting this long would just hang the request - say so
                    # instead, the caller can show the user what to expect.
                    raise GeminiQuotaError(
                        "All configured Gemini models are rate limited. "
                        f"The next one frees up in about {int(delay // 60)} minute(s).",
                        retry_after=delay,
                        daily=delay > 3600,
                    )
                if waited + delay > max_wait:
                    raise GeminiQuotaError(
                        "All configured Gemini models are rate limited. "
                        f"Please retry in {int(delay)}s.",
                        retry_after=delay,
                    )
                waited += delay
                logger.info(
                    "All Gemini models are rate limited - waiting %.0fs before retrying",
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            tried.add(model_id)
            yield model_id

    def _mark_retired(self, model_id: str, exc: BaseException) -> None:
        """Remember that a model id was retired by Google so we stop using it."""
        if not _is_retired_model(exc):
            return
        suggestion = _retirement_suggestion(exc)
        self._retired[model_id] = suggestion
        logger.error(
            "Model %s is no longer available%s - it will be skipped from now on.",
            model_id,
            f" (use '{suggestion}' instead)" if suggestion else "",
        )

    def _mark_cooldown(self, model_id: str, exc: BaseException) -> None:
        """Put a model on cooldown when it reports a quota/rate-limit error."""
        if not _is_quota_error(exc):
            return
        delay = _quota_retry_delay(exc)
        if delay is None:
            delay = 30.0
        # The free tier has both a per-minute and a per-day quota; a daily
        # quota asks us to wait for hours, so honour it (capped at 24h).
        delay = max(10.0, min(delay + 2.0, 86_400.0))
        self._cooldown_until[model_id] = time.monotonic() + delay
        logger.warning(
            "Model %s hit its rate limit - cooling down for %.0fs%s",
            model_id,
            delay,
            " (daily quota)" if _is_daily_quota(exc) else "",
        )

    def _raise_for_exhausted_chain(self, last_error: BaseException | None) -> None:
        """Turn "every model failed" into the right error.

        A retired model becomes a :class:`GeminiQuotaError` style message with
        Google's suggested replacement; a quota error explains the limit;
        anything else is re-raised untouched.
        """
        if last_error is None:
            raise RuntimeError("No Gemini model available")

        if self._retired:
            names = ", ".join(sorted(self._retired))
            suggestion = next(
                (value for value in self._retired.values() if value), ""
            )
            message = (
                f"The configured Gemini model is no longer available ({names})."
                + (f" Use '{suggestion}' instead." if suggestion else "")
                + " Update GEMINI_MODEL in your .env file."
            )
            raise GeminiQuotaError(message, retry_after=0.0, daily=False) from last_error

        if _is_quota_error(last_error):
            daily = _is_daily_quota(last_error)
            retry_after = _quota_retry_delay(last_error) or 60.0
            if daily:
                message = (
                    "You have used up today's free-tier quota for every configured "
                    f"Gemini model (resets in about "
                    f"{int(retry_after // 3600)}h {int((retry_after % 3600) // 60)}m)."
                )
            else:
                message = (
                    "All configured Gemini models are rate limited right now. "
                    "Please retry in a moment."
                )
            raise GeminiQuotaError(
                message, retry_after=retry_after, daily=daily
            ) from last_error

        raise last_error

    # ------------------------------------------------------------------
    async def ainvoke(
        self,
        messages: Any,
        tools: list[Any] | None = None,
        chain: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke Gemini, walking the model chain on transient failures.

        Returns the model response.  The model that actually served the
        request is recorded in :attr:`active_model`.
        """
        last_error: BaseException | None = None
        targets = chain or self.model_chain

        async for model_id in self._iter_models(targets):
            model = self.model(model_id)
            if tools:
                model = self.bind_tools(model, tools)

            try:
                response = await self._invoke_with_retry(model, messages, **kwargs)
                self._active_model = model_id
                if model_id != targets[0]:
                    logger.info("Served by fallback model %s", model_id)
                return response
            except asyncio.CancelledError:
                raise
            except GeminiQuotaError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._mark_retired(model_id, exc)
                self._mark_cooldown(model_id, exc)
                if _is_retired_model(exc):
                    logger.warning("Model %s is retired - trying the next model", model_id)
                    continue
                if not _is_transient(exc):
                    raise
                logger.warning(
                    "Model %s failed transiently (%s) - trying the next model",
                    model_id,
                    str(exc)[:120],
                )

        self._raise_for_exhausted_chain(last_error)

    async def _invoke_with_retry(self, model: Any, messages: Any, **kwargs: Any) -> Any:
        """Call the model, retrying transient failures with exponential backoff."""
        attempts = max(0, self.settings.max_retries) + 1
        last_error: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                return await model.ainvoke(messages, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if not _is_transient(exc) or attempt == attempts:
                    raise
                if _is_daily_quota(exc):
                    # The daily quota will not clear during this request -
                    # fail over to the next model straight away.
                    raise
                # Honour the delay Google asked for (quota errors carry one),
                # but keep it short: the model chain handles the waiting.
                delay = _quota_retry_delay(exc) or (
                    self.settings.retry_base_delay * (2 ** (attempt - 1))
                )
                delay = max(1.0, min(delay, 10.0))
                logger.warning(
                    "Gemini transient error (attempt %d/%d): %s - retrying in %.1fs",
                    attempt,
                    attempts,
                    str(exc)[:140],
                    delay,
                )
                await asyncio.sleep(delay)

        raise last_error  # pragma: no cover - unreachable, kept for clarity

    # ------------------------------------------------------------------
    async def astream(
        self,
        messages: Any,
        tools: list[Any] | None = None,
        chain: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream model chunks, walking the model chain on transient failures.

        Streaming (rather than `ainvoke`) is what makes the tokens visible
        in the browser: the LangChain callback stream only emits
        `on_llm_new_token` events when the model is consumed incrementally.
        """
        last_error: BaseException | None = None
        targets = chain or self.model_chain

        async for model_id in self._iter_models(targets):
            model = self.model(model_id)
            if tools:
                model = self.bind_tools(model, tools)

            emitted = False
            try:
                async for chunk in self._astream_with_retry(model, messages, **kwargs):
                    emitted = True
                    yield chunk
                self._active_model = model_id
                if model_id != targets[0]:
                    logger.info("Served by fallback model %s", model_id)
                return
            except asyncio.CancelledError:
                raise
            except GeminiQuotaError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._mark_retired(model_id, exc)
                self._mark_cooldown(model_id, exc)
                if emitted or (not _is_transient(exc) and not _is_retired_model(exc)):
                    # Text was already streamed - falling back would duplicate it.
                    raise
                if _is_retired_model(exc):
                    logger.warning("Model %s is retired - trying the next model", model_id)
                    continue
                logger.warning(
                    "Model %s failed transiently (%s) - trying the next model",
                    model_id,
                    str(exc)[:120],
                )

        self._raise_for_exhausted_chain(last_error)

    async def _astream_with_retry(
        self, model: Any, messages: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Stream from the model, retrying transient failures with backoff."""
        attempts = max(0, self.settings.max_retries) + 1
        last_error: BaseException | None = None

        for attempt in range(1, attempts + 1):
            produced = False
            try:
                async for chunk in model.astream(messages, **kwargs):
                    produced = True
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if produced or not _is_transient(exc) or attempt == attempts:
                    raise
                if _is_daily_quota(exc):
                    # The daily quota will not clear during this request -
                    # fail over to the next model straight away.
                    raise
                delay = _quota_retry_delay(exc) or (
                    self.settings.retry_base_delay * (2 ** (attempt - 1))
                )
                delay = max(1.0, min(delay, 10.0))
                logger.warning(
                    "Gemini transient error while streaming (attempt %d/%d): %s - retrying in %.1fs",
                    attempt,
                    attempts,
                    str(exc)[:140],
                    delay,
                )
                await asyncio.sleep(delay)

        raise last_error  # pragma: no cover - unreachable, kept for clarity

    def validate_model(self, model_id: str | None = None, deep: bool = False) -> dict[str, Any]:
        """Check that a model still exists and can serve requests.

        The cheap check uses the `models.get` endpoint, which does **not**
        consume generation quota, but it cannot tell that a model was retired
        (retired ids still show up in `models.list()`).  `deep=True` sends one
        tiny request instead, which is the only reliable way to catch a
        retired id - it costs one request of quota.
        """
        target = model_id or self.settings.gemini_model
        if not self.settings.has_api_key:
            return {"model": target, "exists": False, "error": "GEMINI_API_KEY is not set"}

        if target in self._retired:
            return {
                "model": target,
                "exists": False,
                "retired": True,
                "suggestion": self._retired[target],
                "error": "The model was retired by Google.",
            }

        if deep:
            try:
                from google import genai

                probe = genai.Client(api_key=self.settings.gemini_api_key)
                probe.models.generate_content(
                    model=target, contents="Reply with exactly: OK"
                )
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                retired = _is_retired_model(exc)
                self._mark_retired(target, exc)
                return {
                    "model": target,
                    "exists": False,
                    "retired": retired,
                    "quota_exhausted": _is_quota_error(exc),
                    "suggestion": _retirement_suggestion(exc),
                    "error": message[:300],
                }
            return {
                "model": target,
                "exists": True,
                "retired": False,
                "suggestion": "",
                "error": None,
            }

        try:
            from google import genai

            probe = genai.Client(api_key=self.settings.gemini_api_key)
            probe.models.get(model=target)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            retired = _is_retired_model(exc)
            return {
                "model": target,
                "exists": False,
                "retired": retired,
                "suggestion": _retirement_suggestion(exc),
                "error": message[:300],
            }

        return {
            "model": target,
            "exists": True,
            "retired": False,
            "suggestion": "",
            "error": None,
        }

    def status(self) -> dict[str, Any]:
        """Diagnostic payload surfaced by the backend health/debug endpoints."""
        return {
            "configured": self.available,
            "model": self.settings.gemini_model,
            "model_chain": self.model_chain,
            "router_model": self.settings.router_model or self.settings.gemini_model,
            "router_chain": self.router_chain,
            "active_model": self._active_model,
            "temperature": self.settings.temperature,
            "max_output_tokens": self.settings.max_output_tokens,
            "request_timeout": self.settings.request_timeout,
            "max_retries": self.settings.max_retries,
            "thinking_budget": self.settings.thinking_budget,
            "cooldowns": self.cooldown_state(),
            "retired_models": dict(self._retired),
            "api_key": self.settings.masked_api_key,
            "hint": None if self.available else self.settings.missing_key_hint(),
        }
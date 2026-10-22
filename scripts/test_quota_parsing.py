"""Unit checks for the quota error parsing (no network needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_layer.client import _is_daily_quota, _is_quota_error, _quota_retry_delay

SAMPLES = [
    "Please retry in 19h26m42.801627968s",
    "Please retry in 26.948401053s",
    "Please retry in 70002s",
    "Please retry in 2m30s",
    "{'details': [{'@type': 'google.rpc.RetryInfo', 'retryDelay': '70002s'}]}",
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
    "generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash",
    "503 UNAVAILABLE. This model is currently experiencing high demand.",
]

DAILY_MARKER = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE_MARKER = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"

failures = 0


def check(condition: bool, label: str) -> None:
    global failures
    mark = "\033[92mPASS\033[0m" if condition else "\033[91mFAIL\033[0m"
    print(f"  [{mark}] {label}")
    if not condition:
        failures += 1


def approx(actual: float | None, expected: float, tolerance: float = 0.01) -> bool:
    return actual is not None and abs(actual - expected) <= tolerance


print("duration parsing")
check(approx(_quota_retry_delay(Exception("Please retry in 19h26m42.8s")), 70002.8), "19h26m42.8s")
check(approx(_quota_retry_delay(Exception("Please retry in 26.948401053s")), 26.948), "26.9s")
check(approx(_quota_retry_delay(Exception("Please retry in 70002s")), 70002.0), "70002s")
check(approx(_quota_retry_delay(Exception("Please retry in 2m30s")), 150.0), "2m30s")
check(
    approx(_quota_retry_delay(Exception("{'retryDelay': '70002s'}")), 70002.0),
    "retryDelay field (single quotes)",
)
check(
    approx(_quota_retry_delay(Exception('{"retryDelay": "70002s"}')), 70002.0),
    "retryDelay field (double quotes)",
)
check(_quota_retry_delay(Exception("no hint here")) is None, "no hint -> None")

print()
print("daily vs per-minute detection")
check(_is_daily_quota(Exception(f"quotaId: {DAILY_MARKER}")) is True, "daily quota detected")
check(_is_daily_quota(Exception(f"quotaId: {MINUTE_MARKER}")) is False, "per-minute not daily")
check(_is_daily_quota(Exception("503 UNAVAILABLE high demand")) is False, "503 is not a quota issue")

print()
print("transient / quota classification")
check(_is_quota_error(Exception("429 RESOURCE_EXHAUSTED quota exceeded")) is True, "429 is quota")
check(_is_quota_error(Exception("503 UNAVAILABLE high demand")) is False, "503 is not quota")

print()
print(f"{'ALL PARSING CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
raise SystemExit(1 if failures else 0)

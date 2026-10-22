"""Verify that retired / unknown model ids are detected (no generation quota used)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_layer import ChatBotGraph

graph = ChatBotGraph()
client = graph.client

failures = 0
for model in ["gemini-2.0-flash", "gemini-3.8-flash", "not-a-real-model"]:
    # deep=True sends one tiny request - the only reliable way to detect a
    # retired model id (it still shows up in models.list()).
    result = client.validate_model(model, deep=True)
    print(
        f"{model:<24} exists={str(result['exists']):<6} "
        f"retired={str(result.get('retired')):<6} "
        f"quota={str(result.get('quota_exhausted')):<6} "
        f"suggestion={result.get('suggestion')!r}"
    )
    if model == "gemini-2.0-flash":
        if not result.get("retired"):
            print("  [FAIL] a retired model was not detected")
            failures += 1
        if result.get("suggestion") != "gemini-3.8-flash":
            print(f"  [FAIL] wrong suggestion: {result.get('suggestion')!r}")
            failures += 1
    if model == "gemini-3.8-flash" and result.get("retired"):
        print("  [FAIL] a live model was reported as retired")
        failures += 1
    if model == "not-a-real-model" and result["exists"]:
        print("  [FAIL] an unknown model was reported as usable")
        failures += 1

print()
print("ALL VALIDATION CHECKS PASSED" if not failures else f"{failures} CHECK(S) FAILED")
raise SystemExit(1 if failures else 0)

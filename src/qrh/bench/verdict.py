"""Bench verdicts, derived from the suite's own gated detect / false-alarm metrics; never invented.

Suites that plant known defects (``planted-bugs``, ``llm-gate``, ``onboarding``, ``param-stability``) have a rule: every gated
``detect/<class>`` metric found all its planted cases (k == n) and every gated ``false_alarm/<class>`` metric flagged
no clean control (k == 0). ``noise-fpr``, ``known-edge-power`` and ``history`` have no pass criterion in code (their
rates are reported, no threshold is set), so they record ``NO_RULE``. The run verdict is PASS only when every suite
has a rule and passes; FAIL when any suite fails; otherwise NO_RULE.
"""

from __future__ import annotations

from typing import Any

PASS, FAIL, NO_RULE = "PASS", "FAIL", "NO_RULE"
RULED_SUITES = ("planted-bugs", "llm-gate", "onboarding", "param-stability")


def suite_verdict(suite: str, result: Any) -> str:
    if suite not in RULED_SUITES or not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
        return NO_RULE
    seen = False
    for name, m in result["metrics"].items():
        if not isinstance(m, dict) or not m.get("gate") or not m.get("n"):
            continue
        kind = name.split("/", 1)[0]
        if kind == "detect":
            seen = True
            if m["k"] != m["n"]:
                return FAIL
        elif kind == "false_alarm":
            seen = True
            if m["k"] != 0:
                return FAIL
    return PASS if seen else NO_RULE


def run_verdicts(results: dict[str, Any]) -> tuple[str, dict[str, str]]:
    per_suite = {s: suite_verdict(s, r) for s, r in results.items()}
    values = set(per_suite.values())
    overall = FAIL if FAIL in values else PASS if values == {PASS} else NO_RULE
    return overall, per_suite

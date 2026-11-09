"""Deterministic "plays": habits worth a second look, each naming its rule and the numbers that fired it.

The rules are plain comparisons against thresholds.json (and two fixed counts written below). They flag
questions, not verdicts: ``strong-on-mechanical`` in particular infers the kind of work from the tool mix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qrh.usage.settings import Prices, Thresholds

READ_ONLY_TOOLS = frozenset({"Read", "Grep", "Glob", "Bash", "WebFetch"})
MUTATING_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit", "Agent", "Task"})
COLD_CACHE_MAX_EVENTS = 3  # fires above this many cold main-chain requests
MANY_HOOK_BLOCKS = 5  # fires at this many hook blocks in one session


@dataclass(frozen=True)
class Play:
    session_id: str
    play: str
    rule: str
    numbers: dict[str, Any]
    text: str


def _mechanical(s: dict[str, Any], th: Thresholds, prices: Prices | None, notes: list[str]) -> Play | None:
    tools: dict[str, int] = s.get("tools") or {}
    total = sum(tools.values())
    if total < th.mechanical_min_tools or any(t in tools for t in MUTATING_TOOLS):
        return None
    share = sum(n for t, n in tools.items() if t in READ_ONLY_TOOLS) / total
    if share < th.mechanical_tool_share:
        return None
    models: dict[str, dict[str, int]] = s.get("models") or {}
    if not models:
        return None
    lead = sorted(models, key=lambda m: (-models[m].get("output", 0), m))[0]
    if prices is None or lead not in prices.models:
        why = "no tier information (prices.json is missing)" if prices is None else f"model {lead!r} has no tier in prices.json"
        notes.append(f"strong-on-mechanical not evaluated: {why}")
        return None
    tier = prices.models[lead].tier
    if tier != "frontier":
        return None
    rule = (
        f"main model tier is frontier, at least {th.mechanical_min_tools} tool calls, at least "
        f"{th.mechanical_tool_share:.0%} of them read-only, and no editing or agent tools"
    )
    return Play(
        s["session_id"], "strong-on-mechanical", rule,
        {"model": lead, "tier": tier, "tool_calls": total, "read_only_share": round(share, 4)},
        f"heuristic question, not a verdict: a frontier-tier model ({lead}) made {total} tool calls, "
        f"{share:.0%} read-only, with no edits. Could a cheaper model have done this?",
    )


def evaluate(s: dict[str, Any], th: Thresholds, prices: Prices | None) -> tuple[list[Play], list[str]]:
    """Plays that fired for one session record, plus notes on rules that could not be evaluated."""
    sid, plays, notes = s["session_id"], [], []
    mech = _mechanical(s, th, prices, notes)
    if mech:
        plays.append(mech)
    cold = int(s.get("cold_cache_events") or 0)
    if cold > COLD_CACHE_MAX_EVENTS:
        plays.append(Play(
            sid, "cold-cache", f"more than {COLD_CACHE_MAX_EVENTS} main-chain requests with a cache hit rate "
            f"below {th.cache_cold_pct:g}%", {"cold_cache_events": cold, "cache_cold_pct": th.cache_cold_pct},
            f"{cold} requests read less than {th.cache_cold_pct:g}% of their prompt from cache "
            f"(the rule fires above {COLD_CACHE_MAX_EVENTS}).",
        ))
    peak = int(s.get("peak_context") or 0)
    if peak >= th.context_high_tokens:
        plays.append(Play(
            sid, "high-context", f"peak context of one request at or above {th.context_high_tokens} tokens",
            {"peak_context": peak, "context_high_tokens": th.context_high_tokens},
            f"one request carried {peak} tokens of context (limit {th.context_high_tokens}).",
        ))
    hooks = s.get("hooks") or {}
    blocks = int(hooks.get("block_total") or 0)
    if blocks >= MANY_HOOK_BLOCKS:
        plays.append(Play(
            sid, "many-hook-blocks", f"{MANY_HOOK_BLOCKS} or more hook blocks in one session",
            {"hook_blocks": blocks}, f"hooks blocked {blocks} actions in this session.",
        ))
    lint, gate = int(hooks.get("lint_blocks") or 0), int(hooks.get("gate_blocks") or 0)
    if lint + gate >= 1:
        plays.append(Play(
            sid, "gate-fail-after-edit", "at least one look-ahead lint or data-gate block in the session",
            {"lint_blocks": lint, "gate_blocks": gate},
            f"{lint} look-ahead lint block(s) and {gate} data-gate block(s) in this session.",
        ))
    return plays, notes


def evaluate_all(
    sessions: list[dict[str, Any]], th: Thresholds, prices: Prices | None
) -> tuple[list[Play], list[str]]:
    plays: list[Play] = []
    notes: list[str] = []
    for s in sessions:
        p, n = evaluate(s, th, prices)
        plays.extend(p)
        notes.extend(n)
    return plays, notes

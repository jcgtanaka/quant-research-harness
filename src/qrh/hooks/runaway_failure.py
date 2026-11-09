"""PostToolUseFailure: remember which tool calls failed, for the runaway guard's failure streak.

Informational: never blocks. Interrupted calls are choices, not failures, and are not counted.
"""

from __future__ import annotations

from qrh.hooks.runaway import load_state, save_state
from qrh.hooks.runner import HookContext, HookResult


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if ctx.config.runaway is None:
        return HookResult("info", "runaway guard disabled")
    if p.raw.get("is_interrupt") is True:
        return HookResult("info", "interrupt, not a failure")
    st, _ = load_state(ctx.home, p.session_id)
    tid = p.raw.get("tool_use_id")
    calls = st["calls"]
    hit = next((c for c in reversed(calls) if isinstance(tid, str) and tid and c[0] == tid), None)
    if hit is None and calls:
        hit = calls[-1]  # no usable id: the failure belongs to the latest call
    if hit is None:
        return HookResult("info", "failure with no recorded call")
    hit[1] = True
    save_state(ctx.home, p.session_id, st)
    return HookResult("info", "failure recorded")

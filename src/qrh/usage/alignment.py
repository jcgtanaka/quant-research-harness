"""Why a session without subagents can differ from the harness cost-state, when the reason is checkable.

Three causes were confirmed on real sessions, each by an exact reconstruction (numbers only, no message text):

* ``stale-cost-state``: the transcript went on after the last cost-state record was written (a session that
  kept running or was resumed), so the cost-state equals the transcript *up to that record*.
* ``incomplete-message``: a message whose stream never reached a stop reason (interrupted) is in the transcript
  with usage, but the harness does not bill it; without those messages the totals match.
* ``counter-restart``: the harness cost counter started again at a resume, so the cost-state equals the
  *suffix* of the transcript from some request. This one needs a search over start points, so it must match
  exactly (every compared token count), otherwise a lucky near-match could "explain" anything.

Only the first matching cause is kept. The result is stored in the session archive, and ``verify`` re-checks
it against the cost-state, so the verdict still comes from archived numbers only.
"""

from __future__ import annotations

from typing import Any

from qrh.usage.parse import Request, TokenCounts, parse_ts
from qrh.usage.verify import INPUT_SIDE, TOLERANCE_PCT, _diff_pct, _sum_models

STALE, INCOMPLETE, RESTART = "stale-cost-state", "incomplete-message", "counter-restart"


def _dump(requests: list[Request]) -> dict[str, dict[str, int]]:
    models: dict[str, TokenCounts] = {}
    for r in requests:
        models.setdefault(r.model, TokenCounts()).add(r.tokens)
    return {m: models[m].as_dict() for m in sorted(models)}


def _visible(requests: list[Request], theirs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    seen = {r.model for r in requests}
    return {m: v for m, v in theirs.items() if m in seen}


def _diff(requests: list[Request], visible: dict[str, dict[str, Any]]) -> float:
    return _diff_pct(_sum_models([_dump(requests)]), visible, INPUT_SIDE)


def _suffix_start(main: list[Request], visible: dict[str, dict[str, Any]]) -> int | None:
    """The request index from which the later requests equal the cost-state exactly (largest suffix first)."""
    acc: dict[str, dict[str, int]] = {}
    for k in range(len(main) - 1, 0, -1):
        r = main[k]
        a = acc.setdefault(r.model, dict.fromkeys(INPUT_SIDE, 0))
        a["input"] += r.tokens.input
        a["cache_read"] += r.tokens.cache_read
        a["cache_creation"] += r.tokens.cache_write
        if all(
            acc.get(m, {}).get(kind, 0) == visible.get(m, {}).get(kind, 0)
            for m in set(acc) | set(visible) for kind in INPUT_SIDE
        ):
            return k
    return None


def find_alignment(
    main: list[Request], theirs: dict[str, dict[str, Any]], cost_state_main_requests: int | None,
    tolerance_pct: float = TOLERANCE_PCT,
) -> dict[str, Any] | None:
    """The first checkable cause for ``main`` (deduplicated main-chain requests) differing from ``theirs``."""
    if not main:
        return None
    visible = _visible(main, theirs)
    if _diff(main, visible) <= tolerance_pct:
        return None  # it matches as it is
    n = len(main)
    k = cost_state_main_requests
    if k is not None and 0 < k < n and _diff(main[:k], visible) <= tolerance_pct:
        return {"kind": STALE, "requests_total": n, "requests_after": n - k, "models": _dump(main[:k])}
    complete = [r for r in main if r.complete]
    if len(complete) < n and _diff(complete, visible) <= tolerance_pct:
        return {"kind": INCOMPLETE, "requests_total": n, "incomplete_requests": n - len(complete),
                "models": _dump(complete)}
    start = _suffix_start(main, visible)
    if start is not None:
        before, after = parse_ts(main[start - 1].ts), parse_ts(main[start].ts)
        gap = int((after - before).total_seconds()) if before and after else None
        return {"kind": RESTART, "requests_total": n, "requests_before": start, "gap_before_s": gap,
                "models": _dump(main[start:])}
    return None

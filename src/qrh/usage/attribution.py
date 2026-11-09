"""Join usage with the trial tags and hook log written by the hooks (both under <QRH_HOME>/logs).

A message belongs to the most recent trial tagged in its session at or before the message timestamp;
before any tag, or with no timestamp, it is "unattributed".
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qrh.usage.parse import Request, TokenCounts, parse_ts
from qrh.usage.settings import Prices, price_cost

UNATTRIBUTED = "unattributed"
LINT_HOOK = "lookahead-lint"
GATE_HOOK = "data-gate"

AttribKey = tuple[str, str | None]


@dataclass(frozen=True)
class Tag:
    ts: Any  # datetime
    trial_id: str
    stage: str | None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                yield rec


def load_session_tags(home: str | Path) -> dict[str, list[Tag]]:
    """Trial tags per session id, oldest first. Tags without a trial id (``qrh new``) are not attributable."""
    out: dict[str, list[Tag]] = defaultdict(list)
    for rec in _read_jsonl(Path(home) / "logs" / "session_trials.jsonl"):
        sid, trial, when = rec.get("session_id"), rec.get("trial_id"), parse_ts(rec.get("ts"))
        if isinstance(sid, str) and sid and isinstance(trial, str) and trial and when is not None:
            stage = rec.get("stage")
            out[sid].append(Tag(when, trial, stage if isinstance(stage, str) else None))
    for tags in out.values():
        tags.sort(key=lambda t: t.ts)
    return dict(out)


def attribute(requests: Iterable[Request], tags: list[Tag]) -> dict[AttribKey, dict[str, TokenCounts]]:
    """Tokens per (trial, stage) and model."""
    times = [t.ts for t in tags]
    out: dict[AttribKey, dict[str, TokenCounts]] = {}
    for r in requests:
        when = parse_ts(r.ts)
        i = bisect_right(times, when) if when is not None else 0
        key: AttribKey = (tags[i - 1].trial_id, tags[i - 1].stage) if i else (UNATTRIBUTED, None)
        out.setdefault(key, {}).setdefault(r.model, TokenCounts()).add(r.tokens)
    return out


def hook_stats(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    blocks: dict[str, int] = defaultdict(int)
    overrides = 0
    for r in rows:
        if r.get("override") is True:
            overrides += 1
        if r.get("decision") == "block" and isinstance(r.get("hook"), str):
            blocks[r["hook"]] += 1
    return {
        "blocks": dict(sorted(blocks.items())), "block_total": sum(blocks.values()), "overrides": overrides,
        "lint_blocks": blocks.get(LINT_HOOK, 0), "gate_blocks": blocks.get(GATE_HOOK, 0),
    }


def load_hook_stats(home: str | Path) -> dict[str, dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in _read_jsonl(Path(home) / "logs" / "hooks.jsonl"):
        sid = rec.get("session_id")
        if isinstance(sid, str) and sid:
            by_session[sid].append(rec)
    return {sid: hook_stats(rows) for sid, rows in by_session.items()}


def _weight(models: dict[str, TokenCounts], prices: Prices | None) -> float | None:
    """Price-based weight, or None if any model is unpriced."""
    if prices is None:
        return None
    total = 0.0
    for name, t in models.items():
        p = prices.models.get(name)
        if p is None:
            return None
        total += price_cost(p, input=t.input, output=t.output, cache_read=t.cache_read,
                            cache_write_5m=t.cache_write_5m, cache_write_1h=t.cache_write_1h)
    return total


def allocate_cost(
    attrib: dict[AttribKey, dict[str, TokenCounts]], session_cost: float | None, prices: Prices | None
) -> tuple[dict[AttribKey, float | None], str]:
    """Split a session cost over its (trial, stage) buckets; the harness reports one number per session.

    Weights are price-based when every model involved has a price, otherwise uncached tokens
    (input + output + cache writes). Returns the split and how it was weighted.
    """
    if session_cost is None:
        return {k: None for k in attrib}, "none"
    priced = {k: _weight(m, prices) for k, m in attrib.items()}
    if all(w is not None for w in priced.values()):
        weights, how = {k: float(w) for k, w in priced.items() if w is not None}, "price-weighted"
    else:
        weights = {k: float(sum(t.input + t.output + t.cache_write for t in m.values())) for k, m in attrib.items()}
        how = "token-weighted"
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in attrib}, how
    return {k: session_cost * w / total for k, w in weights.items()}, how

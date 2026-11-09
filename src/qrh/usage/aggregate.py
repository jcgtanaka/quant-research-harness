"""Totals and groupings over archived sessions (and local-model records). Pure functions, no I/O."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from qrh.usage.harvest import day_of, week_of

BY_CHOICES = ("session", "day", "week", "model", "trial", "trial-stage", "project")
TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write")


def _blank(key: str) -> dict[str, Any]:
    return {
        "key": key, "sessions": 0, "cost_usd": 0.0, "sessions_without_cost": 0, "input": 0, "output": 0,
        "cache_read": 0, "cache_write": 0, "hit_pct": None, "plays": 0, "local_records": 0,
        "_main_read": 0, "_main_ctx": 0,
    }


def _add_models(row: dict[str, Any], models: Mapping[str, Mapping[str, int]], only: str | None = None) -> None:
    for name, c in models.items():
        if only is not None and name != only:
            continue
        row["input"] += c.get("input", 0)
        row["output"] += c.get("output", 0)
        row["cache_read"] += c.get("cache_read", 0)
        row["cache_write"] += c.get("cache_write_5m", 0) + c.get("cache_write_1h", 0)


def _add_main(row: dict[str, Any], models: Mapping[str, Mapping[str, int]], only: str | None = None) -> None:
    for name, c in models.items():
        if only is not None and name != only:
            continue
        w = c.get("cache_write_5m", 0) + c.get("cache_write_1h", 0)
        row["_main_read"] += c.get("cache_read", 0)
        row["_main_ctx"] += c.get("input", 0) + c.get("cache_read", 0) + w


def _all_models(s: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Main chain plus subagents, merged per model."""
    out: dict[str, dict[str, int]] = {}
    for models in [s.get("models") or {}, *[(a.get("models") or {}) for a in (s.get("subagents") or {}).values()]]:
        for name, c in models.items():
            o = out.setdefault(name, {})
            for k, v in c.items():
                o[k] = o.get(k, 0) + v
    return out


def _finish(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows.values():
        r["hit_pct"] = round(100 * r["_main_read"] / r["_main_ctx"], 2) if r["_main_ctx"] else None
        r["cost_usd"] = round(r["cost_usd"], 6)
        del r["_main_read"], r["_main_ctx"]
        out.append(r)
    return out


def totals(sessions: Iterable[Mapping[str, Any]], local: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = list(sessions)
    sources = Counter(s.get("cost_source", "none") for s in sessions)
    local = list(local)
    return {
        "sessions": len(sessions),
        "cost_usd": round(sum(s["cost_usd"] for s in sessions if s.get("cost_usd") is not None), 6),
        "sessions_without_cost": sum(1 for s in sessions if s.get("cost_usd") is None),
        "sessions_with_subagents": sum(1 for s in sessions if s.get("subagents")),
        "cost_sources": dict(sorted(sources.items())),
        "local": {
            "records": len(local), "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in local),
            "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in local),
        },
    }


def group(
    sessions: Iterable[Mapping[str, Any]], local: Iterable[Mapping[str, Any]], by: str, *,
    plays: Mapping[str, int] | None = None, labels: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Rows grouped by ``by``. ``labels`` maps real project labels to the names shown."""
    if by not in BY_CHOICES:
        raise ValueError(f"unknown grouping {by!r}; use one of {list(BY_CHOICES)}")
    plays, labels = plays or {}, labels or {}
    sessions = sorted(sessions, key=lambda s: (s.get("start") or "", s["session_id"]))
    local = list(local)
    rows: dict[str, dict[str, Any]] = {}

    def row(key: str) -> dict[str, Any]:
        return rows.setdefault(key, _blank(key))

    def shown(label: str) -> str:
        return labels.get(label, label)

    for rank, s in enumerate(sessions, 1):
        sid, cost = s["session_id"], s.get("cost_usd")
        n_plays = plays.get(sid, 0)
        if by in ("session", "day", "week", "project"):
            key = {
                "session": f"S{rank:03d}", "day": day_of(s.get("start")) or "unknown",
                "week": week_of(s.get("start")) or "unknown", "project": shown(s.get("project_label") or "unknown"),
            }[by]
            r = row(key)
            r["sessions"] += 1
            r["plays"] += n_plays
            if cost is None:
                r["sessions_without_cost"] += 1
            else:
                r["cost_usd"] += cost
            _add_models(r, _all_models(s))
            _add_main(r, s.get("models") or {})
            if by == "session":
                r["project"] = shown(s.get("project_label") or "unknown")
        elif by == "model":
            by_model = s.get("cost_by_model") or {}
            for name, c in _all_models(s).items():
                r = row(name)
                r["sessions"] += 1
                r["plays"] += n_plays
                if cost is None:
                    r["sessions_without_cost"] += 1
                else:
                    r["cost_usd"] += by_model.get(name, 0.0)
                _add_models(r, {name: c})
                _add_main(r, s.get("models") or {}, only=name)
            tokens = _all_models(s)
            for name, c in by_model.items():  # models billed by the harness that no transcript shows
                if name not in tokens and cost is not None:
                    r = row(name)
                    r["sessions"] += 1
                    r["cost_usd"] += c
        else:
            seen: set[str] = set()
            for t in s.get("trials") or []:
                key = t["trial_id"] if by == "trial" else (
                    f"{t['trial_id']}/{t['stage']}" if t.get("stage") else t["trial_id"])
                r = row(key)
                if key not in seen:
                    r["sessions"] += 1
                    seen.add(key)
                if t.get("cost_usd") is None:
                    r["sessions_without_cost"] += 1
                else:
                    r["cost_usd"] += t["cost_usd"]
                _add_models(r, t.get("models") or {})
                _add_main(r, t.get("models") or {})
    if by in ("day", "week", "model"):
        for rec in local:
            key = {"day": day_of(rec.get("ts")), "week": week_of(rec.get("ts")), "model": rec.get("model")}[by]
            r = row(key or "unknown")
            r["local_records"] += 1
            r["input"] += int(rec.get("prompt_tokens") or 0)
            r["output"] += int(rec.get("completion_tokens") or 0)
    out = _finish(rows)
    if by in ("session", "day", "week"):
        return sorted(out, key=lambda r: r["key"])
    return sorted(out, key=lambda r: (-r["cost_usd"], r["key"]))

"""Regression finder: did something about the agent's behaviour change when the setup changed?

Sessions are split at a change point (``--by model``, ``--by config`` or ``--by week``) and consecutive groups are
compared on per-turn metrics: new tokens, tool failures, hook blocks, cost and cache hit. Continuous metrics use a
two-sided Mann-Whitney U test with the rank-biserial correlation as effect size; rates (turns with a failure, turns
with a hook block) use Fisher's exact test with the odds ratio. Fewer than ``MIN_TURNS`` turns on a side is flagged.

Honest limits: the turns of one session resemble each other, so the p-values are optimistic; the groups differ in
the work they did; the comparison cannot show cause. Nothing here is a verdict.

The question and the before/after design follow the regression-finder skill by Ryan Alberts (MIT,
https://github.com/RyanAlberts/best-of-Agent-Harnesses); the code, metrics and report are independent (see NOTICE).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from scipy import stats

from qrh.usage import chartdata
from qrh.usage.attribution import _read_jsonl
from qrh.usage.harvest import load_archive, week_of
from qrh.usage.parse import parse_ts
from qrh.usage.privacy import label_map
from qrh.usage.settings import Prices, load_prices, load_thresholds, price_cost

MIN_TURNS = 20
BY_CHOICES = ("week", "model", "config")
CONTINUOUS = (
    ("new_tokens_per_turn", "input + output + cache-write tokens of the turn (main chain; subagents excluded)"),
    ("tool_failures_per_turn", "failed tool calls in the turn"),
    ("hook_blocks_per_turn", "qrh hook blocks logged during the turn"),
    ("cost_per_turn_usd", "session cost split over its turns by weight (estimate)"),
    ("cache_hit_pct", "cache read / (input + cache read + cache write) in the turn"),
)
RATES = (("turns_with_a_tool_failure", "failures"), ("turns_with_a_hook_block", "blocks"))
CAVEAT = ("This does not show cause: the groups also differ in the work done, and the turns of one session resemble "
          "each other, so the p-values are optimistic.")


def _tokens(models: Mapping[str, Mapping[str, int]]) -> int:
    return sum(c.get("input", 0) + c.get("output", 0) + c.get("cache_write_5m", 0) + c.get("cache_write_1h", 0)
               for c in models.values())


def _hit(models: Mapping[str, Mapping[str, int]]) -> float | None:
    cr = sum(c.get("cache_read", 0) for c in models.values())
    ctx = sum(c.get("input", 0) + c.get("cache_read", 0) + c.get("cache_write_5m", 0) + c.get("cache_write_1h", 0)
              for c in models.values())
    return 100 * cr / ctx if ctx else None


def _weights(turns: list[dict[str, Any]], prices: Prices | None) -> list[float]:
    names = {m for t in turns for m in t["models"]}
    if prices is not None and names <= set(prices.models):
        return [sum(price_cost(prices.models[m], input=c.get("input", 0), output=c.get("output", 0),
                               cache_read=c.get("cache_read", 0), cache_write_5m=c.get("cache_write_5m", 0),
                               cache_write_1h=c.get("cache_write_1h", 0)) for m, c in t["models"].items()) for t in turns]
    return [float(_tokens(t["models"])) for t in turns]


def _lead_model(s: Mapping[str, Any]) -> str:
    models = s.get("models") or {}
    if not models:
        models = {m: c for t in s.get("turns") or [] for m, c in t["models"].items()}
    return sorted(models, key=lambda m: (-models[m].get("output", 0), m))[0] if models else "unknown"


def _group_key(s: Mapping[str, Any], by: str) -> str:
    if by == "week":
        return week_of(s.get("start")) or "unknown"
    if by == "model":
        return _lead_model(s)
    h = s.get("config_hash")
    return f"cfg-{h[:8]}" if h else "unknown"


def _blocks(home: str | Path) -> dict[str, list[datetime]]:
    out: dict[str, list[datetime]] = defaultdict(list)
    for r in _read_jsonl(Path(home) / "logs" / "hooks.jsonl"):
        when = parse_ts(r.get("ts"))
        if r.get("decision") == "block" and isinstance(r.get("session_id"), str) and when is not None:
            out[r["session_id"]].append(when)
    return out


def session_turns(s: Mapping[str, Any], blocks: list[datetime], prices: Prices | None) -> list[dict[str, float | None]]:
    """Per-turn metric values of one archived session (turns that made no request are skipped)."""
    all_turns = s.get("turns") or []
    indexed = [(i, t) for i, t in enumerate(all_turns) if t.get("models")]
    turns = [t for _, t in indexed]
    cost = s.get("cost_usd")
    weights = _weights(turns, prices)
    total = sum(weights)
    starts = [parse_ts(t.get("ts")) for t in all_turns]
    rows: list[dict[str, float | None]] = []
    for i, (idx, t) in enumerate(indexed):
        lo = starts[idx]
        hi = next((x for x in starts[idx + 1:] if x is not None), None)
        n_blocks = sum(1 for b in blocks if lo is not None and b >= lo and (hi is None or b < hi))
        rows.append({
            "new_tokens_per_turn": float(_tokens(t["models"])), "tool_failures_per_turn": float(t.get("failures", 0)),
            "hook_blocks_per_turn": float(n_blocks),
            "cost_per_turn_usd": (cost * weights[i] / total) if cost is not None and total > 0 else None,
            "cache_hit_pct": _hit(t["models"]),
        })
    return rows


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "mean": None}
    s = sorted(values)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    return {"n": len(s), "median": round(med, 6), "mean": round(sum(s) / len(s), 6)}


def _clean(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def compare_continuous(name: str, before: list[float | None], after: list[float | None]) -> dict[str, Any]:
    b, a = [v for v in before if v is not None], [v for v in after if v is not None]
    out: dict[str, Any] = {"metric": name, "before": _summary(b), "after": _summary(a), "test": "not-run", "p": None,
                           "effect_size": None, "effect_name": "rank-biserial correlation (after vs before)", "note": ""}
    if not b or not a:
        out["note"] = "no data on one side"
        return out
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    out.update(test="mann-whitney-u", p=_clean(p), effect_size=_clean(2 * u / (len(a) * len(b)) - 1))
    if out["p"] is None:
        out["note"] = "every value is identical, so there is nothing to test"
    return out


def compare_rate(name: str, before: list[int], after: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"metric": name, "test": "not-run", "p": None, "effect_size": None,
                           "effect_name": "odds ratio (after vs before)", "note": ""}
    for side, v in (("before", before), ("after", after)):
        out[side] = {"n": len(v), "yes": sum(v), "rate": (sum(v) / len(v)) if v else None}
    if not before or not after:
        out["note"] = "no data on one side"
        return out
    table = [[sum(after), len(after) - sum(after)], [sum(before), len(before) - sum(before)]]
    odds, p = stats.fisher_exact(table)
    out.update(test="fisher-exact", p=_clean(p), effect_size=_clean(odds))
    if out["effect_size"] is None:
        out["note"] = "odds ratio undefined (a zero cell)"
    return out


def build_regress(home: str | Path, by: str = "week", env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Raises ValueError for an unknown grouping; SettingsError, PrivacyError from the shared loaders."""
    if by not in BY_CHOICES:
        raise ValueError(f"unknown split {by!r}; use one of {list(BY_CHOICES)}")
    env = os.environ if env is None else env
    load_thresholds(home)  # the shared settings must be readable, as for every usage report
    prices = load_prices(home)
    sessions = sorted(load_archive(home).values(), key=lambda s: (s.get("start") or "", s["session_id"]))
    label_map(sessions, env)  # validates QRH_PROTECTED_LABELS; no project label is shown in this report
    blocks = _blocks(home)
    notes: list[str] = []
    old = sum(1 for s in sessions if "turns" not in s)
    if old:
        notes.append(f"{old} archived session(s) predate per-turn capture and are skipped; harvest while their transcripts exist")
    groups: dict[str, dict[str, Any]] = {}
    for s in sessions:
        if "turns" not in s:
            continue
        key = _group_key(s, by)
        g = groups.setdefault(key, {"key": key, "first": s.get("start") or "~", "sessions": 0, "turns": 0, "rows": []})
        rows = session_turns(s, blocks.get(s["session_id"], []), prices)
        g["sessions"] += 1
        g["turns"] += len(rows)
        g["rows"].extend(rows)
    ordered = sorted(groups.values(), key=lambda g: (g["key"] == "unknown", g["first"], g["key"]))
    comparisons = []
    for prev, nxt in zip(ordered, ordered[1:]):
        metrics = [compare_continuous(n, [r[n] for r in prev["rows"]], [r[n] for r in nxt["rows"]]) for n, _ in CONTINUOUS]
        for name, base in (("turns_with_a_tool_failure", "tool_failures_per_turn"), ("turns_with_a_hook_block", "hook_blocks_per_turn")):
            metrics.append(compare_rate(name, [int(r[base] > 0) for r in prev["rows"]], [int(r[base] > 0) for r in nxt["rows"]]))
        small = min(prev["turns"], nxt["turns"]) < MIN_TURNS
        comparisons.append({
            "before": prev["key"], "after": nxt["key"], "n_before": prev["turns"], "n_after": nxt["turns"],
            "small_sample": small,
            "warning": f"small sample: fewer than {MIN_TURNS} turns on a side; read the p-values as anecdotes" if small else "",
            "metrics": metrics,
        })
    days = [s["start"][:10] for s in sessions if s.get("start")]
    return {
        "by": by, "min_turns": MIN_TURNS, "caveat": CAVEAT, "notes": notes,
        "groups": [{"key": g["key"], "sessions": g["sessions"], "turns": g["turns"]} for g in ordered],
        "comparisons": comparisons, "metric_definitions": dict(CONTINUOUS),
        "range": {"first": days[0] if days else None, "last": days[-1] if days else None, "sessions": len(sessions)},
    }


# --- text and charts -------------------------------------------------------------------------------

def _fmt(v: Any, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}g}"


def render_regress_text(rep: dict[str, Any]) -> str:
    if not rep["groups"]:
        return "no sessions with per-turn data in the archive; run `qrh usage harvest` first\n"
    lines = [f"regression finder, split by {rep['by']}", "", "groups (time order):"]
    lines += [f"  {g['key']:<22} {g['sessions']:>4} session(s) {g['turns']:>6} turn(s)" for g in rep["groups"]]
    if not rep["comparisons"]:
        lines += ["", "only one group, so there is nothing to compare"]
    for c in rep["comparisons"]:
        lines += ["", f"{c['before']} -> {c['after']}  ({c['n_before']} vs {c['n_after']} turns)"]
        if c["warning"]:
            lines.append(f"  WARNING {c['warning']}")
        for m in c["metrics"]:
            if m["test"] == "not-run":
                lines.append(f"  {m['metric']:<28} not run: {m['note']}")
                continue
            if m["test"] == "fisher-exact":
                side = f"rate {_fmt(m['before']['rate'])} -> {_fmt(m['after']['rate'])}"
            else:
                side = f"median {_fmt(m['before']['median'])} -> {_fmt(m['after']['median'])}"
            lines.append(f"  {m['metric']:<28} {side}; {m['test']} p={_fmt(m['p'])}, effect {_fmt(m['effect_size'])}"
                         + (f" ({m['note']})" if m["note"] else ""))
    lines += ["", rep["caveat"]]
    lines += [f"note: {n}" for n in rep["notes"]]
    return "\n".join(lines) + "\n"


def regress_charts(rep: dict[str, Any]) -> list[dict[str, Any]]:
    base = chartdata._base
    fields = [{"label": "Split by", "value": rep["by"]}, {"label": "Groups", "value": len(rep["groups"])},
              {"label": "Comparisons", "value": len(rep["comparisons"])},
              {"label": "Small-sample comparisons", "value": sum(c["small_sample"] for c in rep["comparisons"])}]
    summary = base("regress_summary", "Regression finder summary", "text", rep, fields=fields,
                   notes=[rep["caveat"], f"Fewer than {rep['min_turns']} turns on a side is flagged as a small sample.", *rep["notes"]])
    g_cols = [{"key": k, "label": lbl} for k, lbl in (("key", "Group"), ("sessions", "Sessions"), ("turns", "Turns"))]
    groups = base("regress_groups", "Groups in time order", "table", rep, columns=g_cols, rows=list(rep["groups"]),
                  empty_message="No groups.")
    rows = []
    for c in rep["comparisons"]:
        for m in c["metrics"]:
            rows.append({"change": f"{c['before']} -> {c['after']}", "metric": m["metric"], "test": m["test"],
                         "p": _fmt(m["p"]), "effect": _fmt(m["effect_size"]),
                         "small": "yes" if c["small_sample"] else "no"})
    cols = [{"key": k, "label": lbl} for k, lbl in (("change", "Change"), ("metric", "Metric"), ("test", "Test"), ("p", "p"),
                                                    ("effect", "Effect"), ("small", "Small sample"))]
    comp = base("regress_comparisons", "Before / after comparisons", "table", rep, columns=cols, rows=rows,
                empty_message="Nothing to compare.")
    return [summary, groups, comp]

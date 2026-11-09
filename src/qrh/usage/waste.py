"""Waste report: where a session spent tokens it did not need to, with one fix per waste type.

Design after the session-waste-report skill by Ryan Alberts (MIT, https://github.com/RyanAlberts/best-of-Agent-Harnesses),
its categories re-read, oversized results, cache rebuilds, failing loops, compactions. The code, thresholds and data
model are this project's own (see NOTICE). Every number comes from ``sessions.json`` (captured at harvest) and
``thresholds.json``; there are no numeric fallbacks. Paths and commands appear only as hashed aliases, never as text.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qrh.usage import chartdata
from qrh.usage.harvest import load_archive
from qrh.usage.privacy import label_map
from qrh.usage.quality import evaluate
from qrh.usage.settings import load_prices, load_thresholds, load_waste_thresholds

ALIASES_SHOWN = 5
EXAMPLE_PREFIX = "example-"  # placeholders written by `qrh usage init` are not real prices
TYPES = ("re-reads", "large-tool-outputs", "repeated-failing-commands", "cache-cold-events", "frontier-on-mechanical",
         "compactions")
FIXES = {
    "re-reads": "Add to CLAUDE.md: read a file once per task; for one detail search it (rg -n) or read a line range.",
    "large-tool-outputs": "Trim output where it starts (| tail -n 50, quiet flags, line ranges) so it stops riding in every later call.",
    "repeated-failing-commands": "Write the working command in CLAUDE.md, or fix the cause; stop retrying the same failing command.",
    "cache-cold-events": "Compact (/compact) before a long pause or start a fresh session with a short note instead of resuming a long one.",
    "frontier-on-mechanical": "Hand read-only exploration to a cheaper model or a subagent with a small-model setting.",
    "compactions": "Start a new session per task and compact on purpose at natural breaks to keep the context short.",
}


def _alias(prefix: str, h: str) -> str:
    return f"{prefix}-{h[:4]}"


def build_waste(home: str | Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Raises SettingsError (thresholds), PrivacyError or ValueError."""
    env = os.environ if env is None else env
    th, wt, prices = load_thresholds(home), load_waste_thresholds(home), load_prices(home)
    archive = load_archive(home)
    sessions = sorted(archive.values(), key=lambda s: (s.get("start") or "", s["session_id"]))
    labels = label_map(sessions, env)
    out_sessions: list[dict[str, Any]] = []
    totals: dict[str, dict[str, Any]] = {t: {"count": 0, "sessions": 0, "cost_usd": 0.0} for t in TYPES}
    totals["cache-cold-events"].update(cold_write_tokens=0, unpriced_sessions=0)
    notes: list[str] = []
    missing_waste = 0
    for rank, s in enumerate(sessions, 1):
        w = s.get("waste")
        if w is None:
            missing_waste += 1
            continue
        items: list[dict[str, Any]] = []
        flagged = {a: n for a, n in (w.get("reread_runs") or {}).items() if n > wt.reread_k}
        if flagged:
            items.append({"type": "re-reads", "count": len(flagged), "paths": len(flagged), "worst_run": max(flagged.values()),
                          "extra_reads": sum(n - 1 for n in flagged.values()),
                          "aliases": [_alias("f", a) for a in sorted(flagged, key=lambda a: (-flagged[a], a))[:ALIASES_SHOWN]]})
        big = [n for n in (w.get("output_tokens") or []) if n > wt.large_output_tokens]
        if big:
            items.append({"type": "large-tool-outputs", "count": len(big), "tokens": sum(big), "largest": max(big)})
        fails = {a: n for a, n in (w.get("failing_commands") or {}).items() if n >= wt.repeat_fail_k}
        if fails:
            items.append({"type": "repeated-failing-commands", "count": len(fails), "commands": len(fails),
                          "worst": max(fails.values()),
                          "aliases": [_alias("c", a) for a in sorted(fails, key=lambda a: (-fails[a], a))[:ALIASES_SHOWN]]})
        cold = int(s.get("cold_cache_events") or 0)
        if cold:
            tokens, usd, unpriced = 0, 0.0, False
            for model, c in (w.get("cold_writes") or {}).items():
                w5, w1 = int(c.get("cache_write_5m", 0)), int(c.get("cache_write_1h", 0))
                tokens += w5 + w1
                price = prices.models.get(model) if prices is not None else None
                if price is None or model.startswith(EXAMPLE_PREFIX):
                    unpriced = unpriced or (w5 + w1) > 0
                else:
                    usd += (w5 * (price.cache_write_5m - price.cache_read) + w1 * (price.cache_write_1h - price.cache_read)) / 1e6
            known = not unpriced
            items.append({"type": "cache-cold-events", "count": cold, "estimate": True, "cold_write_tokens": tokens,
                          "cost_usd": round(usd, 6) if known else None, "cost_label": "estimated excess cost",
                          "basis": "estimated excess cost: cold cache-write tokens x (cache_write_price - cache_read_price) per model, "
                                   "from prices.json" if known else
                                   "estimated excess cost: n/a (no price) - prices.json has no real price for a model involved"})
        plays, play_notes = evaluate(s, th, prices)
        notes.extend(play_notes)
        for p in plays:
            if p.play == "strong-on-mechanical":
                items.append({"type": "frontier-on-mechanical", "count": 1, **{k: v for k, v in p.numbers.items()}})
        if int(s.get("compactions") or 0):
            items.append({"type": "compactions", "count": int(s["compactions"]), "peak_context": int(s.get("peak_context") or 0)})
        if not items:
            continue
        for it in items:
            it["fix"] = FIXES[it["type"]]
            t = totals[it["type"]]
            t["count"] += it["count"]
            t["sessions"] += 1
            t["cost_usd"] += it.get("cost_usd") or 0.0
            if it["type"] == "cache-cold-events":
                t["cold_write_tokens"] += it["cold_write_tokens"]
                t["unpriced_sessions"] += it["cost_usd"] is None
        label = s.get("project_label", "")
        out_sessions.append({"session": f"S{rank:03d}", "project": labels.get(label, label), "waste": items})
    if missing_waste:
        notes.append(f"{missing_waste} archived session(s) predate waste capture; run `qrh usage harvest` while their "
                     "transcripts still exist to include them")
    for t in totals.values():
        t["cost_usd"] = round(t["cost_usd"], 6)
    days = [s["start"][:10] for s in sessions if s.get("start")]
    return {"sessions": out_sessions, "totals": totals, "totals_sessions": len(sessions), "notes": sorted(set(notes)),
            "thresholds": {"reread_k": wt.reread_k, "large_output_tokens": wt.large_output_tokens,
                           "repeat_fail_k": wt.repeat_fail_k, "cache_cold_pct": th.cache_cold_pct},
            "range": {"first": days[0] if days else None, "last": days[-1] if days else None, "sessions": len(sessions)}}


# --- text -----------------------------------------------------------------------------------------

def _line(it: dict[str, Any]) -> str:
    t = it["type"]
    if t == "re-reads":
        return f"{it['paths']} file(s) read more times than the limit (worst run {it['worst_run']}, {it['extra_reads']} extra reads): " + ", ".join(it["aliases"])
    if t == "large-tool-outputs":
        return f"{it['count']} result(s) above the size limit, {it['tokens']:,} estimated tokens (largest {it['largest']:,})"
    if t == "repeated-failing-commands":
        return f"{it['commands']} command(s) failed repeatedly (worst {it['worst']} times): " + ", ".join(it["aliases"])
    if t == "cache-cold-events":
        cost = (f"estimated excess cost USD {it['cost_usd']:.2f}" if it["cost_usd"] is not None
                else "estimated excess cost n/a (no price)")
        return f"{it['count']} cold-cache request(s), {it['cold_write_tokens']:,} tokens re-written, {cost}"
    if t == "frontier-on-mechanical":
        return f"{it.get('model', '?')} made {it.get('tool_calls', '?')} read-only tool calls (heuristic)"
    return f"{it['count']} compaction(s), peak context {it['peak_context']:,} tokens"


def render_waste_text(rep: dict[str, Any]) -> str:
    if rep["totals_sessions"] == 0:
        return "no sessions in the archive; run `qrh usage harvest` first\n"
    lines = [f"waste report: {rep['totals_sessions']} session(s), {len(rep['sessions'])} with waste", ""]
    lines.append("totals by type (count, sessions affected):")
    for t in TYPES:
        v = rep["totals"][t]
        extra = ""
        if t == "cache-cold-events":
            extra = f", {v['cold_write_tokens']:,} tokens re-written"
            if v["cost_usd"] or not v["unpriced_sessions"]:
                extra += f", estimated excess cost USD {v['cost_usd']:.2f}"
            if v["unpriced_sessions"]:
                extra += f" (n/a, no price, for {v['unpriced_sessions']} session(s))"
        lines.append(f"  {t:<28} {v['count']:>6}  {v['sessions']:>4}{extra}")
    lines.append("")
    for s in rep["sessions"]:
        lines.append(f"{s['session']} {s['project']}")
        for it in s["waste"]:
            lines.append(f"  {it['type']}: {_line(it)}")
            lines.append(f"    fix: {it['fix']}")
    for n in rep["notes"]:
        lines.append(f"note: {n}")
    return "\n".join(lines) + "\n"


# --- charts ---------------------------------------------------------------------------------------

def waste_charts(rep: dict[str, Any]) -> list[dict[str, Any]]:
    base = chartdata._base
    fields = [{"label": "Sessions", "value": rep["totals_sessions"]},
              {"label": "Sessions with waste", "value": len(rep["sessions"])}]
    fields += [{"label": t, "value": rep["totals"][t]["count"]} for t in TYPES]
    cold = rep["totals"]["cache-cold-events"]
    fields.append({"label": "Cold-cache tokens re-written", "value": cold["cold_write_tokens"]})
    fields.append({"label": "Estimated excess cost (priced sessions)",
                   "value": f"USD {cold['cost_usd']:.2f}" + (f", n/a for {cold['unpriced_sessions']} unpriced" if cold["unpriced_sessions"] else "")})
    notes = ["Cache-cold dollars are the estimated excess of re-writing context over reading it from cache, at prices.json prices; n/a without a real price.", *rep["notes"]]
    summary = base("waste_summary", "Waste summary", "text", rep, fields=fields, notes=notes)
    by_type = base("waste_by_type", "Waste events by type", "bar", rep,
                   bars=[{"label": t, "value": rep["totals"][t]["count"]} for t in TYPES],
                   x_axis=chartdata._axis("Events", "count"))
    rows = [{"session": s["session"], "project": s["project"],
             "types": ", ".join(it["type"] for it in s["waste"]), "events": sum(it["count"] for it in s["waste"])}
            for s in rep["sessions"]]
    cols = [{"key": k, "label": lbl} for k, lbl in (("session", "Session"), ("project", "Project"), ("types", "Waste types"),
                                                    ("events", "Events"))]
    table = base("waste_sessions", "Sessions with waste", "table", rep, columns=cols, rows=rows, empty_message="No waste found.")
    return [summary, by_type, table]

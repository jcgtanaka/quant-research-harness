"""Build, print and render the usage report. The only inputs are sessions.json and the logs (one-source rule)."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qrh.usage import aggregate, chartdata
from qrh.usage.harvest import day_of, load_archive
from qrh.usage.local import load_local
from qrh.usage.privacy import label_map
from qrh.usage.quality import evaluate_all
from qrh.usage.settings import load_prices, load_thresholds, usage_dir


def reports_dir(home: str | Path) -> Path:
    return usage_dir(home) / "reports"


def build_report(home: str | Path, by: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The report as a JSON-able dictionary. Raises SettingsError, PrivacyError or ValueError."""
    env = os.environ if env is None else env
    th = load_thresholds(home)
    prices = load_prices(home)
    archive = load_archive(home)
    local = load_local(home)
    sessions = sorted(archive.values(), key=lambda s: (s.get("start") or "", s["session_id"]))
    labels = label_map(sessions, env)
    rank = {s["session_id"]: i for i, s in enumerate(sessions, 1)}
    plays, raw_notes = evaluate_all(sessions, th, prices)
    notes = [f"{n} (for {c} session(s))" for n, c in sorted(Counter(raw_notes).items())]
    per_session: dict[str, int] = defaultdict(int)
    for p in plays:
        per_session[p.session_id] += 1
    by_id = {s["session_id"]: s for s in sessions}
    play_rows = [{
        "session": f"S{rank[p.session_id]:03d}",
        "project": labels.get(by_id[p.session_id].get("project_label", ""), by_id[p.session_id].get("project_label", "")),
        "play": p.play, "rule": p.rule, "numbers": p.numbers, "text": p.text,
    } for p in plays]
    rows = aggregate.group(sessions, local, by, plays=per_session, labels=labels)
    daily = aggregate.group(sessions, local, "day", plays=per_session, labels=labels)
    matrix: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for s in sessions:
        day = day_of(s.get("start"))
        if day and s.get("cost_usd") is not None:
            for m, c in (s.get("cost_by_model") or {}).items():
                matrix[day][m] += c
    only_sessions = aggregate.group(sessions, [], "day")
    days = [r["key"] for r in only_sessions if r["key"] != "unknown"]
    return {
        "by": by, "rows": rows, "totals": aggregate.totals(sessions, local), "plays": play_rows, "notes": notes,
        "tokens": {k: sum(r[k] for r in only_sessions) for k in aggregate.TOKEN_FIELDS},
        "daily": daily, "daily_model_cost": {d: {m: round(c, 6) for m, c in sorted(v.items())} for d, v in sorted(matrix.items())},
        "trials": aggregate.group(sessions, [], "trial", labels=labels),
        "range": {"first": days[0] if days else None, "last": days[-1] if days else None, "sessions": len(sessions)},
    }


def build_charts(rep: dict[str, Any]) -> list[dict[str, Any]]:
    return chartdata.build_all(rep)


def write_chart_files(home: str | Path, charts: list[dict[str, Any]]) -> list[Path]:
    out_dir = reports_dir(home)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in charts:
        path = out_dir / f"{d['key']}.chart.json"
        path.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def render_pdf(charts: list[dict[str, Any]], path: str | Path, title: str = "AI usage report") -> None:
    from qrh.reports.pdf import write_pdf
    from qrh.usage.charts import draw

    r = charts[0]["range"]
    write_pdf(path, [draw(d) for d in charts], title=title,
              subject=f"sessions {r['sessions']}, {r['first']} to {r['last']}", keywords="usage, cost, tokens")


# --- text ------------------------------------------------------------------------------------------

PLAYS_SHOWN = 5
COLUMNS = (
    ("key", "group", "<"), ("sessions", "sessions", ">"), ("cost_usd", "cost USD", ">"), ("input", "input", ">"),
    ("output", "output", ">"), ("cache_read", "cache read", ">"), ("cache_write", "cache write", ">"),
    ("hit_pct", "hit %", ">"), ("plays", "plays", ">"),
)


def _cell(key: str, v: Any) -> str:
    if v is None:
        return "-"
    if key == "cost_usd":
        return f"{v:,.2f}"
    if key == "hit_pct":
        return f"{v:.1f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def render_text(rep: dict[str, Any]) -> str:
    t = rep["totals"]
    if t["sessions"] == 0 and not t["local"]["records"]:
        return "no sessions in the archive; run `qrh usage harvest` first\n"
    table = [[h for _k, h, _a in COLUMNS]] + [[_cell(k, r.get(k)) for k, _h, _a in COLUMNS] for r in rep["rows"]]
    widths = [max(len(row[i]) for row in table) for i in range(len(COLUMNS))]
    lines = [f"usage by {rep['by']}"]
    for n, row in enumerate(table):
        lines.append("  ".join(c.ljust(w) if COLUMNS[i][2] == "<" else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))))
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    src = ", ".join(f"{k} {v}" for k, v in t["cost_sources"].items())
    lines.append("")
    lines.append(f"total: {t['sessions']} session(s), cost USD {t['cost_usd']:,.2f} ({src})")
    if t["sessions_without_cost"]:
        lines.append(f"{t['sessions_without_cost']} session(s) have no cost (no harness cost-state and no usable "
                     "price for their models); their cost is not included in any total")
    if t["sessions_with_subagents"]:
        lines.append(f"tokens are a lower bound for {t['sessions_with_subagents']} of {t['sessions']} session(s) that used "
                     "subagents (their transcripts under-record output); cost comes from the Claude Code cost-state")
    if t["local"]["records"]:
        lines.append(f"local models: {t['local']['records']} record(s), {t['local']['prompt_tokens']:,} prompt and "
                     f"{t['local']['completion_tokens']:,} completion tokens, cost 0")
    if rep["plays"]:
        lines.append("")
        lines.append(f"plays ({len(rep['plays'])} in total; all are in --json):")
        for name in sorted({p["play"] for p in rep["plays"]}):
            group_ = [p for p in rep["plays"] if p["play"] == name]
            lines.append(f"  {name} ({len(group_)}), rule: {group_[0]['rule']}")
            for p in group_[:PLAYS_SHOWN]:
                lines.append(f"    {p['session']} {p['project']}: {p['text']}")
            if len(group_) > PLAYS_SHOWN:
                lines.append(f"    ... and {len(group_) - PLAYS_SHOWN} more")
    for n in rep["notes"]:
        lines.append(f"note: {n}")
    return "\n".join(lines) + "\n"

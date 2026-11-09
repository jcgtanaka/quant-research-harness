"""Chart data for the usage report: JSON-serializable, built only from the report dictionary.

``charts.py`` draws exactly these dictionaries, and they are written next to the PDF for a future
dashboard, so the figures and the data cannot disagree. Categorical charts take at most
``theme.MAX_SERIES`` series: the leading models plus "other".
"""

from __future__ import annotations

from typing import Any

from qrh.reports import theme

SCHEMA_VERSION = 1
CHART_KEYS = ("summary", "daily_cost_by_model", "tokens_by_kind", "cache_hit_over_time", "cost_per_trial", "plays")
RANKED_LIMIT = 30
OTHER = "other"


def _base(key: str, title: str, chart_type: str, rep: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "key": key, "title": title, "chart_type": chart_type,
            "range": rep["range"], "disclaimer": theme.DISCLAIMER, **extra}


def _axis(label: str, unit: str = "") -> dict[str, str]:
    return {"label": label, "unit": unit}


def summary(rep: dict[str, Any]) -> dict[str, Any]:
    t = rep["totals"]
    src = ", ".join(f"{k} {v}" for k, v in t["cost_sources"].items()) or "none"
    fields = [
        ("Sessions", t["sessions"]), ("Cost (USD)", f"{t['cost_usd']:.2f}"), ("Cost sources", src),
        ("Sessions without cost", t["sessions_without_cost"]),
        ("Local model records", t["local"]["records"]),
        ("Local prompt tokens", t["local"]["prompt_tokens"]),
        ("Local completion tokens", t["local"]["completion_tokens"]),
    ]
    notes = list(rep["notes"])
    if t["sessions_with_subagents"]:
        notes.append(f"Token counts are a lower bound for {t['sessions_with_subagents']} session(s) that used subagents; "
                     "cost comes from the Claude Code cost-state.")
    return _base("summary", "AI usage summary", "text", rep, fields=[{"label": k, "value": v} for k, v in fields],
                 notes=notes)


def daily_cost_by_model(rep: dict[str, Any]) -> dict[str, Any]:
    days = sorted(rep["daily_model_cost"])
    per_model: dict[str, float] = {}
    for d in days:
        for m, c in rep["daily_model_cost"][d].items():
            per_model[m] = per_model.get(m, 0.0) + c
    ranked = sorted(per_model, key=lambda m: (-per_model[m], m))
    lead = ranked[: theme.MAX_SERIES - 1] if len(ranked) > theme.MAX_SERIES else ranked
    series = [{"name": m, "values": [round(rep["daily_model_cost"][d].get(m, 0.0), 6) for d in days]} for m in lead]
    if len(ranked) > len(lead):
        rest = [m for m in ranked if m not in lead]
        series.append({"name": OTHER, "values": [
            round(sum(rep["daily_model_cost"][d].get(m, 0.0) for m in rest), 6) for d in days]})
    return _base("daily_cost_by_model", "Daily cost by model", "stacked_bar", rep, x=days, series=series,
                 x_axis=_axis("Session start day (UTC)"), y_axis=_axis("Cost", "USD"))


def tokens_by_kind(rep: dict[str, Any]) -> dict[str, Any]:
    tk = rep["tokens"]
    bars = [{"label": "input", "value": tk["input"]}, {"label": "output", "value": tk["output"]},
            {"label": "cache read", "value": tk["cache_read"]}, {"label": "cache write", "value": tk["cache_write"]}]
    n_sub = rep["totals"]["sessions_with_subagents"]
    title = "Tokens by kind (lower bound where subagents were used)" if n_sub else "Tokens by kind"
    return _base("tokens_by_kind", title, "bar", rep, bars=bars, x_axis=_axis("Tokens", "tokens"),
                 lower_bound_sessions=n_sub)


def cache_hit_over_time(rep: dict[str, Any]) -> dict[str, Any]:
    days = [r["key"] for r in rep["daily"]]
    return _base("cache_hit_over_time", "Cache hit rate by day (main chain)", "line", rep, x=days,
                 series=[{"name": "cache hit", "values": [r["hit_pct"] for r in rep["daily"]]}],
                 x_axis=_axis("Session start day (UTC)"), y_axis=_axis("Cache hit", "%"))


def cost_per_trial(rep: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in rep["trials"] if r["cost_usd"] or r["sessions_without_cost"] == 0]
    bars = [{"label": r["key"], "value": r["cost_usd"]} for r in rows][:RANKED_LIMIT]
    return _base("cost_per_trial", "Cost per trial", "bar_ranked", rep, bars=bars, x_axis=_axis("Cost", "USD"),
                 truncated=max(0, len(rows) - RANKED_LIMIT))


def plays(rep: dict[str, Any]) -> dict[str, Any]:
    rows = [{
        "session": p["session"], "project": p["project"], "play": p["play"],
        "numbers": ", ".join(f"{k}={v}" for k, v in sorted(p["numbers"].items())), "rule": p["rule"],
    } for p in rep["plays"]]
    cols = [{"key": k, "label": lbl} for k, lbl in (
        ("session", "Session"), ("project", "Project"), ("play", "Play"), ("numbers", "Numbers"), ("rule", "Rule"))]
    return _base("plays", "Plays: habits worth a second look", "table", rep, columns=cols, rows=rows,
                 empty_message="No play fired.")


def build_all(rep: dict[str, Any]) -> list[dict[str, Any]]:
    return [summary(rep), daily_cost_by_model(rep), tokens_by_kind(rep), cache_hit_over_time(rep),
            cost_per_trial(rep), plays(rep)]

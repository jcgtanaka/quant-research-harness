"""Chart data for a bench receipt: the one description of every figure.

``charts.py`` draws exactly these dictionaries (and a dashboard can render the same JSON). Nothing here recomputes
a statistic; values are read from the receipt. Rates, counts and intervals are those stored in the results.
"""

from __future__ import annotations

from typing import Any

from qrh.bench.compare import within_receipt
from qrh.bench.receipt import BenchReceipt
from qrh.bench.textreport import arms
from qrh.reports import theme

SCHEMA_VERSION = 1
ROWS_PER_TABLE = 20


def _base(r: BenchReceipt, key: str, title: str, chart_type: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "key": key, "title": title, "chart_type": chart_type,
        "bench_id": r.bench_id(), "receipt_hash": r.receipt_hash(), "disclaimer": theme.DISCLAIMER, **extra,
    }


def _series(label: str, slot: int, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": label, "slot": slot,
        "values": [m["rate"] for m in metrics],
        "ci_low": [m["ci"][0] if m["ci"] else None for m in metrics],
        "ci_high": [m["ci"][1] if m["ci"] else None for m in metrics],
        "k": [m["k"] for m in metrics], "n": [m["n"] for m in metrics],
    }


def _card(r: BenchReceipt) -> dict[str, Any]:
    fields = [
        ("Bench id", r.bench_id()), ("Profile", r.profile), ("Seed", r.seed), ("Suites", ", ".join(r.suites)),
        ("Harness code hash", r.code_hash), ("Receipt hash", r.receipt_hash()),
    ]
    if r.overrides:
        fields.append(("Overrides", ", ".join(f"{k}={v}" for k, v in sorted(r.overrides.items()))))
    methods = [
        {"id": mid, "version": v, "experimental": mid in r.non_default_methods}
        for mid, v in sorted(r.methods.items())
    ]
    notes = []
    if r.non_default_methods:
        notes.append("EXPERIMENTAL: a non-default method version was used; it is not promoted by this report.")
    if r.baseline_methods is not None:
        notes.append("The default arm ran over exactly the same generated data as the candidate arm.")
    return _base(r, "bench_card", "Benchmark run", "text",
                 fields=[{"label": k, "value": v} for k, v in fields], methods=methods, notes=notes)


def _planted(r: BenchReceipt, suite: str = "planted-bugs", fig_prefix: str = "planted",
             fig_label: str = "Planted bugs") -> list[dict[str, Any]]:
    pairs = arms(r, suite)
    first = pairs[0][1]
    classes = [c for c in first["variants"] if c != "controls"]
    cats = [*classes, "controls (false alarm)"]
    series = []
    for slot, (label, res) in enumerate(pairs):
        ms = [res["metrics"][f"detect/{c}"] for c in classes] + [res["metrics"]["false_alarm/controls"]]
        series.append(_series(label, slot, ms))
    figs = [_base(
        r, f"{fig_prefix}_detection", f"{fig_label}: detection rate per bug class", "bars",
        x_axis={"label": "Bug class", "unit": ""}, y_axis={"label": "Share flagged", "unit": "ratio"},
        categories=cats, series=series, note="Controls are clean code and data; there a lower share flagged is better.",
    )]
    rows = []
    hidden = first.get("expected_undetectable", {})
    for bug, names in first["variants"].items():
        for name in names:
            kind = ("false alarm" if bug == "controls" else
                    "expected undetectable" if name in hidden.get(bug, []) else "detect")
            prefix = {"false alarm": "false_alarm", "expected undetectable": "expected_undetectable"}.get(kind, "detect")
            row: dict[str, Any] = {"bug_class": bug, "variant": name, "kind": kind}
            for label, res in pairs:
                m = res["metrics"][f"{prefix}/{bug}/{name}"]
                row[label] = f"{m['k']}/{m['n']}"
            rows.append(row)
    cols = [{"key": "bug_class", "label": "Bug class"}, {"key": "variant", "label": "Variant"},
            {"key": "kind", "label": "Kind"}] + [
        {"key": label, "label": f"Flagged ({label})"} for label, _ in pairs
    ]
    pages = [rows[i : i + ROWS_PER_TABLE] for i in range(0, len(rows), ROWS_PER_TABLE)] or [[]]
    for i, chunk in enumerate(pages, 1):
        figs.append(_base(r, f"{fig_prefix}_variants_{i}", f"{fig_label}: every variant ({i}/{len(pages)})", "table",
                          columns=cols, rows=chunk))
    return figs


def _noise(r: BenchReceipt) -> dict[str, Any]:
    pairs = arms(r, "noise-fpr")
    first = pairs[0][1]
    cats = list(first["cells"]) + ["pooled"]
    series = [
        _series(label, slot, [res["metrics"][f"fpr/{c}"] for c in cats]) for slot, (label, res) in enumerate(pairs)
    ]
    return _base(
        r, "noise_fpr", "Noise false positives: share of pure-noise families passing S3-S5", "points_ci",
        x_axis={"label": "Family (grid size g, bars t)", "unit": ""}, y_axis={"label": "Share passing", "unit": "ratio"},
        categories=cats, series=series, nominal=first["nominal"],
        note="Dashed line: nominal level 1 - min_dsr. Bars: exact 95% interval.",
    )


def _power(r: BenchReceipt) -> dict[str, Any]:
    pairs = arms(r, "known-edge-power")
    first = pairs[0][1]
    keys = list(first["cells"])
    xs = [first["cells"][k]["edge"] for k in keys]
    series = [_series(label, slot, [res["metrics"][f"power/{k}"] for k in keys]) for slot, (label, res) in enumerate(pairs)]
    return _base(
        r, "power_curve", "Known-edge power: share of planted-trend datasets passing S3-S5", "line_ci",
        x_axis={"label": "Planted trend strength (latent drift standard deviation per bar)", "unit": ""},
        y_axis={"label": "Share passing", "unit": "ratio"}, x=xs, series=series,
    )


def _comparison(r: BenchReceipt) -> dict[str, Any]:
    cmp = within_receipt(r)
    rows = []
    for x in cmp["rows"]:
        if not x["gate"]:
            continue
        p = x["p_value"]
        rows.append({
            "metric": x["name"], "default": f"{x['a']['k']}/{x['a']['n']}", "candidate": f"{x['b']['k']}/{x['b']['n']}",
            "delta": None if x["delta"] is None else round(100 * x["delta"], 2),
            "p_value": None if p is None else round(p, 4),
            "distinguishable": "yes" if x["distinguishable"] else "no", "better": x["better"],
        })
    cols = [{"key": k, "label": lab} for k, lab in (
        ("metric", "Metric"), ("default", "Default"), ("candidate", "Candidate"), ("delta", "Delta (points)"),
        ("p_value", "Fisher p"), ("distinguishable", "Distinguishable"), ("better", "Better (a=default, b=candidate)"))]
    return _base(r, "comparison", "Default vs candidate (gated metrics)", "table", columns=cols, rows=rows[:ROWS_PER_TABLE],
                 method_differences=cmp["method_differences"], notes=cmp["notes"], truncated=max(0, len(rows) - ROWS_PER_TABLE))


def _history(r: BenchReceipt) -> dict[str, Any]:
    h = r.results["history"]
    rows = [
        {"trial": t["trial_id"], "result": t["direction"] if t.get("changed") else ("unchanged" if t["status"] == "evaluated" else "skipped"),
         "default": t.get("default", {}).get("pass_all"), "candidate": t.get("candidate", {}).get("pass_all"),
         "note": t.get("reason", "")}
        for t in h["trials"]
    ]
    cols = [{"key": k, "label": lab} for k, lab in (("trial", "Trial"), ("result", "Would change"), ("default", "Default passes"),
                                                       ("candidate", "Candidate passes"), ("note", "Note"))]
    return _base(r, "history", "History: stored trials re-evaluated with the candidate", "table", columns=cols,
                 rows=rows[:ROWS_PER_TABLE], summary=h["summary"], source=h["source"], truncated=max(0, len(rows) - ROWS_PER_TABLE))


def build_figures(r: BenchReceipt) -> list[dict[str, Any]]:
    figs = [_card(r)]
    if "planted-bugs" in r.suites:
        figs += _planted(r)
    if "noise-fpr" in r.suites:
        figs.append(_noise(r))
    if "llm-gate" in r.suites:
        figs += _planted(r, "llm-gate", "llm", "LLM gate")
    if "onboarding" in r.suites:
        figs += _planted(r, "onboarding", "onboarding", "Dataset onboarding")
    if "param-stability" in r.suites:
        figs += _planted(r, "param-stability", "stability", "Parameter stability")
    if "known-edge-power" in r.suites:
        figs.append(_power(r))
    if r.baseline_results is not None:
        figs.append(_comparison(r))
    if "history" in r.suites:
        figs.append(_history(r))
    return figs

"""Chart data: JSON-serializable descriptions of every figure, built from verified artifacts.

``charts.py`` draws exactly these dictionaries, so the PDF and the data a future
interactive dashboard renders cannot disagree. Nothing here recomputes a
statistic: values come from receipts and stage artifacts. The only derived
numbers are presentation prep (histogram bin counts, ranking, decimation).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from qrh.pipeline.receipt import Stage
from qrh.pipeline.report import STAGE_TITLES
from qrh.reports import theme
from qrh.reports.inputs import StageData

SCHEMA_VERSION = 1
RANKED_LIMIT = 30
MAX_LINE_POINTS = 2000
ROWS_PER_PAGE_STABILITY = 18
FIRST_OFFENDERS_SHOWN = 3
KEY_METRICS: dict[Stage, tuple[str, ...]] = {
    Stage.S3: ("grid_size", "best_is_sharpe"),
    Stage.S4: ("oos_sharpe", "n_folds"),
    Stage.S5: ("pbo", "dsr"),
    Stage.S6: ("sharpe_ci_low", "cost_stressed_sharpe"),
}


# --- helpers ---------------------------------------------------------------------------------


def _axis(label: str, unit: str = "", kind: str = "linear") -> dict[str, str]:
    return {"label": label, "unit": unit, "type": kind}


def _base(data: StageData, key: str, title: str, chart_type: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "key": key, "title": title, "stage": data.stage.value,
        "trial_id": data.trial_id, "receipt_hash": data.receipt_hash, "verdict": data.receipt.verdict.value,
        "chart_type": chart_type, "disclaimer": theme.DISCLAIMER, **extra,
    }


def _num(v: Any) -> float | None:
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _short_ts(iso: str | None) -> str:
    """Date for midnight stamps, else date and minutes; timezone suffix dropped."""
    if not iso:
        return ""
    ts = pd.Timestamp(iso)
    return ts.strftime("%Y-%m-%d") if (ts.hour, ts.minute, ts.second) == (0, 0, 0) else ts.strftime("%Y-%m-%d %H:%M")


def _params_text(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={params[k]}" for k in sorted(params))


def _decimate(xs: list[Any], *ys: list[Any]) -> tuple[list[Any], ...]:
    n = len(xs)
    if n <= MAX_LINE_POINTS:
        return (xs, *ys)
    idx = sorted(set(range(0, n, math.ceil(n / MAX_LINE_POINTS))) | {n - 1})
    return tuple([col[i] for i in idx] for col in (xs, *ys))


def _histogram(values: np.ndarray) -> tuple[list[float], list[int]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0, 1.0], [0]
    n_bins = int(min(40, max(5, math.ceil(math.sqrt(values.size)))))
    counts, edges = np.histogram(values, bins=n_bins)
    return [float(e) for e in edges], [int(c) for c in counts]


def _label_order(values: list[Any]) -> list[Any]:
    uniq = list(dict.fromkeys(values))
    try:
        return sorted(uniq)
    except TypeError:
        return uniq


# --- S0 -------------------------------------------------------------------------------------


def s0_card(data: StageData) -> dict[str, Any]:
    h = data.artifacts["hypothesis"]
    fields = [
        ("Hypothesis", h["text"]), ("Family", h["family_id"]), ("Strategy family", h["strategy_family"]),
        ("Asset class", h["asset_class"]), ("Frequency", h["frequency"]),
        ("Parameter space size", h["parameter_space_size"]), ("Difference statement", h["difference_statement"]),
        ("Hypothesis hash", h["hypothesis_hash"]),
    ]
    if data.receipt.methods:
        flagged = data.receipt.params.get("non_default_methods") or {}
        fields.append(("Methods", (
            "EXPERIMENTAL (not the default): " + ", ".join(f"{k}={v}" for k, v in sorted(flagged.items()))
            if flagged else f"all default versions ({len(data.receipt.methods)} methods)"
        )))
    return _base(
        data, "s0_card", "S0 Hypothesis and thresholds", "text",
        fields=[{"label": k, "value": v} for k, v in fields],
        thresholds=[{"name": k, "value": v} for k, v in sorted(h["thresholds"].items())],
    )


# --- S1 -------------------------------------------------------------------------------------


def s1_coverage(data: StageData) -> dict[str, Any]:
    rows = [
        {
            "label": d["name"], "start": d["start"], "end": d["end"], "rows": d["rows"],
            "gap_count": d["gap_count"], "gaps": d["gaps"], "status": d["status"],
        }
        for d in data.artifacts["coverage"]["datasets"]
    ]
    return _base(
        data, "s1_coverage", "S1 Data coverage by dataset (gaps marked)", "timeline",
        x_axis=_axis("Time", "", "time"), series_names=["Coverage", "Gap"], rows=rows,
    )


def s1_rules(data: StageData) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for d in data.artifacts["coverage"]["datasets"]:
        for r in d["rules"]:
            rows.append({
                "dataset": d["name"], "rule": r["rule"], "type": r["type"],
                "status": "PASS" if r["passed"] else "FAIL", "offenders": len(r["offenders"]),
                "first_offenders": ", ".join(_short_ts(t) for t in r["offenders"][:FIRST_OFFENDERS_SHOWN]),
            })
        if not d["rules"] and d["status"] != "PASS":
            rows.append({
                "dataset": d["name"], "rule": "(dataset check)", "type": "", "status": d["status"],
                "offenders": 0, "first_offenders": "; ".join(d["reasons"][:1]),
            })
    columns = [
        {"key": "dataset", "label": "Dataset"}, {"key": "rule", "label": "Rule"}, {"key": "type", "label": "Type"},
        {"key": "status", "label": "Status"}, {"key": "offenders", "label": "Offenders"},
        {"key": "first_offenders", "label": "First offending timestamps"},
    ]
    return _base(
        data, "s1_rules", "S1 Data-quality rule results", "table", columns=columns, rows=rows,
        status_column="status", empty_message="No rule results were recorded.",
    )


def _onboarding_doc(data: StageData, index: int) -> dict[str, Any]:
    return data.artifacts["onboarding"]["datasets"][index]


def onboarding_count(data: StageData) -> int:
    return len(data.artifacts.get("onboarding", {}).get("datasets", []))


def _onboarding_subtitle(d: dict[str, Any]) -> str:
    version = f"version {d['version']}" if d.get("version") else "no onboarding version for this file"
    return f"{d['name']}  |  onboarding {d['state']} ({version})  |  privacy {d['privacy']}"


def s1_dataset_card(data: StageData, index: int) -> dict[str, Any]:
    d = _onboarding_doc(data, index)
    card = d.get("card")
    facts: list[dict[str, Any]] = [{"label": "Dataset", "value": d["name"]},
                                   {"label": "Content hash", "value": (d.get("content_hash") or "")[:16]}]
    if card:
        t = card["time"]
        facts += [{"label": "Rows", "value": card["rows"]}, {"label": "Time column", "value": t["column"]},
                  {"label": "Timezone", "value": t["timezone"] or "naive"},
                  {"label": "Inferred frequency", "value": t["inferred_frequency"]},
                  {"label": "Observed density", "value": t["observed_density"]},
                  {"label": "Gaps / duplicates", "value": f"{t['gap_count']} / {t['duplicates']}"},
                  {"label": "Monotonic", "value": t["monotonic_increasing"]}]
        if t.get("first"):
            facts += [{"label": "First / last", "value": f"{_short_ts(t['first'])} / {_short_ts(t['last'])}"}]
        facts += [{"label": "Profile", "value": ("without values; " if card["no_values"] else "") + card["profile_hash"][:12]}]
    columns = [
        {"name": c["name"], "kind": c["kind"], "dtype": c["dtype"], "null_rate": c["null_rate"], "distinct": c["distinct"],
         "min": c["min"], "max": c["max"], "note": c["note"]}
        for c in (card["columns"] if card else [])
    ]
    return _base(
        data, f"s1_dataset_card_{index}", f"S1 Dataset card: {d['name']}", "text", subtitle=_onboarding_subtitle(d),
        facts=facts, relationships=card["relationships"] if card else [], hints=card["hints"] if card else [],
        columns=columns, column_table=[{"key": k, "label": lbl} for k, lbl in (
            ("name", "Column"), ("kind", "Kind"), ("dtype", "Dtype"), ("null_rate", "Null rate"), ("distinct", "Distinct"),
            ("min", "Min"), ("max", "Max"), ("note", "Note"))],
        columns_total=card["columns_total"] if card else 0,
    )


def s1_onboarding(data: StageData, index: int) -> dict[str, Any]:
    d = _onboarding_doc(data, index)
    rows = [{
        "field": ("* " if r["critical"] else "") + r["field"], "claim": r["claim"], "source": r["source"],
        "status": r["status"].upper(), "evidence": r["evidence"], "checked_by": r["checked_by"],
    } for r in d["fields"]]
    columns = [{"key": "field", "label": "Field (* critical)"}, {"key": "claim", "label": "Claim"},
               {"key": "source", "label": "Source"}, {"key": "status", "label": "Status"},
               {"key": "evidence", "label": "Evidence"}]
    pending = d["pending_questions"]
    subtitle = _onboarding_subtitle(d) + (f"  |  {len(pending)} open question(s), {sum(1 for q in pending if q['critical'])} critical"
                                          if pending else "")
    return _base(
        data, f"s1_onboarding_{index}", f"S1 Onboarding table: {d['name']}", "table", subtitle=subtitle, columns=columns,
        rows=rows, status_column="status", empty_message="No instrument model was proposed for this dataset.",
        state=d["state"], pending_questions=d["pending_questions"],
    )


# --- S2 -------------------------------------------------------------------------------------


def s2_lint(data: StageData) -> dict[str, Any]:
    rows = []
    for f in data.artifacts["lint"]["findings"]:
        rows.append({
            "rule": f["rule"], "at": f["at"], "status": "ALLOWED" if f["suppressed"] else f["severity"],
            "message": f["message"] + (f" (reason: {f['reason']})" if f["suppressed"] and f.get("reason") else ""),
            "snippet": f["snippet"],
        })
    columns = [
        {"key": "rule", "label": "Rule"}, {"key": "at", "label": "Location"}, {"key": "status", "label": "Status"},
        {"key": "message", "label": "Message"}, {"key": "snippet", "label": "Code"},
    ]
    return _base(
        data, "s2_lint", "S2 Static look-ahead lint findings", "table", columns=columns, rows=rows,
        status_column="status", empty_message="No findings: the static lint found nothing to report.",
    )


def truncation_count(data: StageData) -> int:
    return len(data.artifacts["truncation"]["cutoffs"])


def s2_truncation(data: StageData, index: int) -> dict[str, Any]:
    c = data.artifacts["truncation"]["cutoffs"][index]
    n = truncation_count(data)
    x, full, trunc = _decimate(c["timestamps"], c["full"], c["truncated"])
    markers = []
    div = c.get("first_divergence")
    if div:
        markers.append({
            "type": "vline", "x": div["timestamp"], "role": "divergence",
            "label": f"First divergence ({div['what']}) {_short_ts(div['timestamp'])}",
        })
    name = c.get("column")
    return _base(
        data, f"s2_truncation_{index}",
        f"S2 Future-truncation test: full vs truncated positions (cutoff {index + 1} of {n})", "line",
        x_axis=_axis("Time", "", "time"), y_axis=_axis(f"Position ({name})" if name else "Position", "weight"),
        series=[{"name": "Full-data run", "x": x, "y": full}, {"name": "Truncated-data run", "x": x, "y": trunc}],
        markers=markers, cutoff=c["cutoff"], cutoff_fraction=c["fraction"], cutoff_status=c["status"],
        columns_total=c["columns_total"], stride=c["stride"],
        empty_message="" if x else "No comparable positions were recorded for this cutoff.",
    )


def s2_llm_gate(data: StageData) -> dict[str, Any]:
    g = data.artifacts["llm_gate"]
    markers: list[dict[str, Any]] = [
        {"type": "vline", "x": m["training_cutoff"], "role": "cutoff",
         "label": f"{m['id']}: cutoff {m['training_cutoff']}"}
        for m in g["models"] if m["training_cutoff"] != "unknown"
    ]
    if g["binding_date"]:
        markers.append({
            "type": "vline", "x": g["binding_date"], "role": "binding",
            "label": f"Binding date {g['binding_date']} (+{g['margin_days']:g} days)",
        })
    return _base(
        data, "s2_llm_gate", "S2 LLM contamination gate", "llm_timeline",
        x_axis=_axis("Time", "", "time"), gate_verdict=g["verdict"], margin_days=g["margin_days"],
        binding_date=g["binding_date"], binding_model=g["binding_model"], models=[dict(m) for m in g["models"]],
        rows=[
            {"label": w["name"], "start": w["start"], "end": w["end"], "status": w["status"],
             "earliest_contaminated_date": w["earliest_contaminated_date"]}
            for w in g["windows"]
        ],
        markers=markers, live_retrieval=dict(g["live_retrieval"]), reasons=list(g["reasons"]),
    )


def s2_sanity(data: StageData) -> dict[str, Any]:
    rows = [
        {"quantity": r["name"], "value": _num(r["value"]), "bound": _num(r["bound"]), "status": "PASS" if r["ok"] else "FAIL"}
        for r in data.artifacts["sanity"]["rows"]
    ]
    columns = [
        {"key": "quantity", "label": "Quantity"}, {"key": "value", "label": "Observed"},
        {"key": "bound", "label": "Bound"}, {"key": "status", "label": "Status"},
    ]
    return _base(
        data, "s2_sanity", "S2 Sanity bounds", "table", columns=columns, rows=rows, status_column="status",
        empty_message="No sanity measurements were recorded.",
    )


# --- S3 -------------------------------------------------------------------------------------

SHARPE_LABEL = "Annualized Sharpe (in-sample)"


def _combos(data: StageData) -> dict[str, Any]:
    return data.artifacts["combos"]


def s3_primary_key(data: StageData) -> str:
    return "s3_heatmap" if len(_combos(data)["param_names"]) == 2 else "s3_ranked"


def s3_heatmap(data: StageData) -> dict[str, Any]:
    doc = _combos(data)
    row_param, col_param = doc["param_names"]
    ys = _label_order([c["params"][row_param] for c in doc["combos"]])
    xs = _label_order([c["params"][col_param] for c in doc["combos"]])
    z: list[list[float | None]] = [[None] * len(xs) for _ in ys]
    for c in doc["combos"]:
        z[ys.index(c["params"][row_param])][xs.index(c["params"][col_param])] = _num(c["sharpe_annualized"])
    flat = [v for r in z for v in r if v is not None]
    return _base(
        data, "s3_heatmap", "S3 In-sample annualized Sharpe by parameter combination", "heatmap",
        x_axis=_axis(col_param, "", "category"), y_axis=_axis(row_param, "", "category"),
        x_labels=[str(v) for v in xs], y_labels=[str(v) for v in ys], z=z,
        z_min=min(flat) if flat else None, z_max=max(flat) if flat else None,
        z_label=SHARPE_LABEL, colorscale=list(theme.SEQUENTIAL),
    )


def s3_ranked(data: StageData) -> dict[str, Any]:
    combos = _combos(data)["combos"]
    defined = [c for c in combos if _num(c["sharpe_annualized"]) is not None]
    defined.sort(key=lambda c: (-c["sharpe_annualized"], c["combo"]))
    rows = [
        {"label": _params_text(c["params"]), "value": float(c["sharpe_annualized"]), "combo": c["combo"]}
        for c in defined[:RANKED_LIMIT]
    ]
    return _base(
        data, "s3_ranked", "S3 In-sample annualized Sharpe, ranked combinations", "dot",
        x_axis=_axis(SHARPE_LABEL, "ratio"), y_axis=_axis("Parameter combination", "", "category"),
        rows=rows, shown=len(rows), total=len(combos), rest_count=len(combos) - len(rows),
    )


def s3_hist(data: StageData) -> dict[str, Any]:
    values = np.array([c["sharpe_annualized"] for c in _combos(data)["combos"] if _num(c["sharpe_annualized"]) is not None])
    bins, counts = _histogram(values)
    return _base(
        data, "s3_hist", "S3 Distribution of in-sample Sharpe across combinations", "histogram",
        x_axis=_axis(SHARPE_LABEL, "ratio"), y_axis=_axis("Combinations", "count"),
        bins=bins, counts=counts, markers=[],
    )


# --- S4 -------------------------------------------------------------------------------------


def s4_folds(data: StageData) -> dict[str, Any]:
    doc = data.artifacts["folds"]
    rows = [{"label": f"Fold {f['fold']}", "train": f["train"], "test": f["test"]} for f in doc["folds"]]
    return _base(
        data, "s4_folds", "S4 Walk-forward folds: train and test spans", "timeline",
        x_axis=_axis("Time", "", "time"), series_names=["Train", "Test"], rows=rows, window=doc["window"],
    )


def s4_equity(data: StageData) -> dict[str, Any]:
    eq = data.artifacts["oos_equity"]["equity"]
    x, y = _decimate([t.isoformat() for t in eq.index], [float(v) for v in eq.to_numpy()])
    return _base(
        data, "s4_equity", "S4 Stitched out-of-sample equity", "line",
        x_axis=_axis("Time", "", "time"), y_axis=_axis("Equity (start = 1)", "multiple"),
        series=[{"name": "Out-of-sample equity (stitched)", "x": x, "y": y}], markers=[],
    )


def s4_params(data: StageData) -> dict[str, Any]:
    rows = [
        {
            "fold": f["fold"], "train": f"{_short_ts(f['train'][0])} to {_short_ts(f['train'][1])}",
            "test": f"{_short_ts(f['test'][0])} to {_short_ts(f['test'][1])}", "params": _params_text(f["params"]),
            "train_sharpe": _num(f["train_sharpe"]), "test_sharpe": _num(f["test_sharpe"]),
        }
        for f in data.artifacts["folds"]["folds"]
    ]
    columns = [
        {"key": "fold", "label": "Fold"}, {"key": "train", "label": "Train span"}, {"key": "test", "label": "Test span"},
        {"key": "params", "label": "Selected parameters"}, {"key": "train_sharpe", "label": "IS Sharpe"},
        {"key": "test_sharpe", "label": "OOS Sharpe"},
    ]
    return _base(data, "s4_params", "S4 Selected parameters per fold", "table", columns=columns, rows=rows, empty_message="No folds were recorded.")


# --- S5 -------------------------------------------------------------------------------------


def s5_lambda(data: StageData) -> dict[str, Any]:
    logits = data.artifacts["pbo_logits"]["logit"].to_numpy(dtype=float)
    d = data.artifacts["dsr_inputs"]
    bins, counts = _histogram(logits)
    return _base(
        data, "s5_lambda", "S5 Distribution of lambda (logit of out-of-sample rank)", "histogram",
        x_axis=_axis("Lambda = ln(omega / (1 - omega))", "logit"), y_axis=_axis("Combinations", "count"),
        bins=bins, counts=counts,
        markers=[{
            "type": "vline", "x": 0, "role": "zero",
            "label": f"PBO = {d['pbo']:.3f} (share of lambda <= 0; limit {d['max_pbo']:g})",
        }],
        annotations={"pbo": d["pbo"], "max_pbo": d["max_pbo"], "combinations": d["pbo_combinations"]},
    )


def s5_dsr(data: StageData) -> dict[str, Any]:
    d = data.artifacts["dsr_inputs"]
    return _base(
        data, "s5_dsr", "S5 Observed Sharpe vs expected best Sharpe under the null (SR0)", "bar",
        y_axis=_axis("Sharpe ratio, per period", "ratio"),
        bars=[
            {"name": "Observed per-period SR", "value": d["sharpe"]},
            {"name": "SR0 (expected best of N null trials)", "value": d["sr0"]},
        ],
        annotations={
            "dsr": d["dsr"], "min_dsr": d["min_dsr"],
            "label": f"DSR = {d['dsr']:.3f} (threshold {d['min_dsr']:g})"
                     + ("; PBO not applicable (one combination)" if d.get("pbo") is None and d.get("pbo_note") else ""),
        },
        inputs=[
            {"name": "Observed SR", "value": d["sharpe"]}, {"name": "SR0", "value": d["sr0"]},
            {"name": "N (trials)", "value": d["n_trials"]},
            *(
                [{"name": f"V used (the larger: {d['var_sharpe_source']})", "value": d["var_sharpe"]},
                 {"name": "V, grid (cross-variant)", "value": d.get("var_sharpe_grid")},
                 {"name": "V, sampling estimate", "value": d["var_sharpe_estimator"]}]
                if "var_sharpe_estimator" in d else  # a receipt recorded before the variance floor
                [{"name": "V (Sharpe variance, sampling estimate)" if d.get("var_sharpe_source") == "estimator"
                  else "V (Sharpe variance)", "value": d["var_sharpe"]}]
            ),
            {"name": "Skewness", "value": d["skew"]}, {"name": "Kurtosis", "value": d["kurtosis"]},
            {"name": "Observations", "value": d["n_obs"]},
        ],
    )


def s5_stability(data: StageData) -> dict[str, Any]:
    d = data.artifacts["param_stability"]
    nb, pl, sel = d["neighbour"], d["plateau"], d["selected"]
    th = d["thresholds"]

    def val(v: Any) -> Any:
        return "n/a" if v is None else _num(v)

    rows: list[dict[str, Any]] = [{"item": "Selected combination", "detail": _params_text(sel.get("params") or {}),
                                   "value": val(nb["selected_sharpe"])}]
    neighbour_rows = [{"item": f"Neighbour ({n['source']})", "detail": f"{n['dimension']} = {n['value']}", "value": val(n["sharpe"])}
                      for n in nb["neighbours"]]  # listed last: the measures come first if the page has to be cut
    for pp in nb.get("per_parameter") or []:
        rows.append({"item": "Neighbour ratio, per parameter", "detail": f"{pp['dimension']}: {pp['n_neighbours']} neighbours, "
                     f"mean OOS Sharpe {val(pp['mean_sharpe'])}", "value": val(pp["ratio"])})
    rows.append({"item": "Neighbour ratio (worst parameter decides)",
                 "detail": (nb["note"] if nb["ratio"] is None else f"worst: {nb['worst_parameter']}; minimum {th['neighbour_ratio']:g}; "
                            f"pooled mean ratio {val(nb.get('pooled_mean_ratio'))} (information only)"), "value": val(nb["ratio"])})
    rows.append({"item": "Local plateau share (decides)",
                 "detail": (pl["note"] or f"{pl['n_close']} of {pl['n_neighbours']} nearby settings keep {th['closeness']:g} of "
                            f"the selected setting's Sharpe; minimum {th['plateau_share']:g}"), "value": val(pl["share"])})
    g = pl.get("global_info") or {}
    if g.get("share") is not None:
        rows.append({"item": "Global plateau share (information only)",
                     "detail": f"{g['n_close']} of {g['n_combinations']} grid combinations reach {th['closeness']:g} of the "
                               "grid's best; depends on grid spacing, never decides", "value": val(g["share"])})
    for p in d["diagnostics"]["parameters"]:
        rows.append({"item": "Diagnostic (report only)", "detail": f"{p['dimension']}: mean correlation of adjacent paths "
                     f"{val(p['mean_correlation'])}", "value": val(p["mean_abs_sharpe_gap"])})
    if d["diagnostics"]["most_fragile"]:
        rows.append({"item": "Most fragile parameter", "detail": "largest mean absolute Sharpe gap to its adjacent values",
                     "value": d["diagnostics"]["most_fragile"]})
    for u in d.get("unchecked") or []:
        rows.append({"item": "Unchecked", "detail": f"{u['dimension']}: {u['reason']}", "value": "n/a"})
    if d.get("skipped"):
        rows.append({"item": "Skipped (not numeric)", "detail": ", ".join(d["skipped"]), "value": "n/a"})
    rows += neighbour_rows
    columns = [{"key": "item", "label": "Item"}, {"key": "detail", "label": "Detail"}, {"key": "value", "label": "OOS Sharpe / value"}]
    return _base(data, "s5_stability", f"S5 Parameter stability: {d['status'].replace('_', ' ')}", "table", columns=columns,
                 rows=rows[:ROWS_PER_PAGE_STABILITY], empty_message="No stability measure was recorded.",
                 annotations={"status": d["status"], "thresholds": th, "thresholds_source": d.get("thresholds_source", ""),
                              "note": d["trial_count_note"], "truncated": max(0, len(rows) - ROWS_PER_PAGE_STABILITY)})


# --- S6 -------------------------------------------------------------------------------------

PANELS = (
    ("sharpe", "Sharpe (annualized)", "ratio"),
    ("cagr", "CAGR", "fraction per year"),
    ("max_drawdown", "Max drawdown", "fraction"),
)


def s6_bootstrap(data: StageData) -> dict[str, Any]:
    samples = data.artifacts["bootstrap_samples"]
    summary = data.artifacts["bootstrap_summary"]
    panels = []
    for metric, label, unit in PANELS:
        values = samples[metric].to_numpy(dtype=float)
        bins, counts = _histogram(values)
        m = summary["metrics"][metric]
        markers = []
        if m["ci"]:
            level = f"{summary['level'] * 100:g}%"
            markers = [
                {"type": "vline", "x": m["ci"][0], "role": "ci", "label": f"CI low {m['ci'][0]:.3g} ({level})"},
                {"type": "vline", "x": m["ci"][1], "role": "ci", "label": f"CI high {m['ci'][1]:.3g} ({level})"},
            ]
        if m["point"] is not None:
            markers.append({"type": "vline", "x": m["point"], "role": "point", "label": f"Point estimate {m['point']:.3g}"})
        panels.append({
            "metric": metric, "title": label, "x_axis": _axis(label, unit), "y_axis": _axis("Replications", "count"),
            "bins": bins, "counts": counts, "markers": markers, "undefined_share": float(m["undefined_share"]),
        })
    return _base(
        data, "s6_bootstrap", "S6 Bootstrap distributions: Sharpe, CAGR and max drawdown", "histogram_panels",
        panels=panels, level=summary["level"], reps=summary["reps"],
    )


def s6_equity(data: StageData) -> dict[str, Any]:
    eq = data.artifacts["equity_compare"]
    mult = data.receipt.metrics.get("cost_stress_multiplier")
    stressed_name = f"Cost-stressed (x{mult:g}) OOS equity" if mult is not None else "Cost-stressed OOS equity"
    x, base, stressed = _decimate(
        [t.isoformat() for t in eq.index], [float(v) for v in eq["base"]], [float(v) for v in eq["stressed"]]
    )
    return _base(
        data, "s6_equity", "S6 Base vs cost-stressed out-of-sample equity", "line",
        x_axis=_axis("Time", "", "time"), y_axis=_axis("Equity (start = 1)", "multiple"),
        series=[{"name": "Base OOS equity", "x": x, "y": base}, {"name": stressed_name, "x": x, "y": stressed}],
        markers=[],
    )


# --- S7 and fallbacks -----------------------------------------------------------------------


def s7_summary(data: StageData) -> dict[str, Any]:
    rows = []
    for stage in (s for s in Stage if s is not Stage.S7):
        r = data.receipts.get(stage)
        if r is None:
            rows.append({"stage": stage.value, "title": STAGE_TITLES[stage], "verdict": "NOT RUN", "reasons": [], "key_metrics": {}, "receipt_hash": None})
            continue
        rows.append({
            "stage": stage.value, "title": STAGE_TITLES[stage], "verdict": r.verdict.value, "reasons": list(r.reasons),
            "key_metrics": {k: r.metrics[k] for k in KEY_METRICS.get(stage, ()) if k in r.metrics},
            "receipt_hash": r.receipt_hash()[:12],
        })
    return _base(
        data, "s7_summary", "S7 Summary of stage verdicts", "summary",
        final={"verdict": data.receipt.verdict.value, "failing_stage": data.receipt.params.get("failing_stage")},
        stages=rows, reasons=list(data.receipt.reasons),
        conditions=[dict(c) for c in data.receipt.params.get("conditions") or []],
        non_default_methods=dict(data.receipt.params.get("non_default_methods") or {}),
    )


def status_card(data: StageData) -> dict[str, Any]:
    r = data.receipt
    return _base(
        data, "status_card", f"{data.stage.value} {STAGE_TITLES.get(data.stage, 'Stage')}: status", "text",
        verdict=r.verdict.value, reasons=list(r.reasons),
        metrics=[{"name": k, "value": v} for k, v in sorted(r.metrics.items())],
        note="This stage recorded no figure data.",
    )

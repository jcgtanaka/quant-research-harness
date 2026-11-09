"""Draw bench figures from their chart data (``qrh.bench.chartdata``) and assemble the PDF.

Only the dictionaries are read, never the receipt, so the PDF and the JSON cannot disagree. The look follows
``qrh.reports.theme`` (fixed categorical slots, at most three series, light grid, status color with icon and word).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from qrh.reports import theme
from qrh.reports.charts import _axes, _draw_table, _fmt
from qrh.reports.pdf import write_pdf

KEYWORDS = "synthetic research use only, not investment advice"


def _page(d: dict[str, Any], bench_id: str, receipt_hash: str) -> Figure:
    return theme.new_report_page(d["title"], None, f"Bench {bench_id}  |  Receipt {receipt_hash[:12]}")


def _err(series: dict[str, Any]) -> np.ndarray:
    vals = np.array([np.nan if v is None else v for v in series["values"]], dtype=float)
    lo = np.array([np.nan if v is None else v for v in series["ci_low"]], dtype=float)
    hi = np.array([np.nan if v is None else v for v in series["ci_high"]], dtype=float)
    return np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])


def _legend(ax: Any, series: list[dict[str, Any]]) -> None:
    if len(series) >= 2:
        handles = [Line2D([0], [0], color=theme.series_color(s["slot"]), marker="o", linewidth=1.5, label=s["label"]) for s in series]
        ax.legend(handles=handles, loc="upper right")


def _card(fig: Figure, d: dict[str, Any]) -> None:
    ax = fig.add_axes((0.05, 0.08, 0.90, 0.78))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 1.0
    ax.text(0.0, y, "Run", fontsize=11, fontweight="bold", va="top", color=theme.TEXT)
    y -= 0.06
    for f in d["fields"]:
        ax.text(0.0, y, f["label"], fontsize=8.5, color=theme.TEXT_MUTED, va="top")
        wrapped = "\n".join(textwrap.wrap(_fmt(f["value"]), 52) or [""])
        ax.text(0.0, y - 0.028, wrapped, fontsize=9, color=theme.TEXT, va="top")
        y -= 0.028 + 0.028 * (wrapped.count("\n") + 1) + 0.02
    ax.text(0.6, 1.0, "Methods used", fontsize=11, fontweight="bold", va="top", color=theme.TEXT)
    y = 0.94
    step = min(0.035, 0.9 / max(1, len(d["methods"])))
    for m in d["methods"]:
        ax.text(0.6, y, f"{m['id']}={m['version']}", fontsize=8, va="top", color=theme.TEXT)
        if m["experimental"]:
            ax.text(0.97, y, f"{theme.VERDICT_ICON['INCONCLUSIVE']} EXPERIMENTAL", fontsize=8, fontweight="bold", va="top",
                    ha="right", color=theme.STATUS["warning"])
        y -= step
    for i, note in enumerate(d["notes"]):
        ax.text(0.0, 0.02 + 0.045 * (len(d["notes"]) - 1 - i), "\n".join(textwrap.wrap(note, 120)), fontsize=8.5,
                color=theme.TEXT_MUTED, va="bottom")


def _bars(fig: Figure, d: dict[str, Any]) -> None:
    ax = _axes(fig, (0.24, 0.14, 0.70, 0.70))
    cats, series = d["categories"], d["series"]
    theme.check_series_count(len(series))
    n, height = len(cats), 0.8 / len(series)
    for j, s in enumerate(series):
        ys = np.arange(n) + (j - (len(series) - 1) / 2) * height
        vals = np.nan_to_num(np.array([np.nan if v is None else v for v in s["values"]], dtype=float))
        ax.barh(ys, vals, height=height * 0.9, color=theme.series_color(s["slot"]), xerr=_err(s),
                error_kw={"ecolor": theme.TEXT_MUTED, "elinewidth": 0.9, "capsize": 2})
    ax.set_yticks(range(n))
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(d["y_axis"]["label"])
    ax.yaxis.grid(False)
    _legend(ax, series)
    if d.get("note"):
        fig.text(0.05, 0.065, d["note"], fontsize=8.5, color=theme.TEXT_MUTED, ha="left")


def _points_ci(fig: Figure, d: dict[str, Any]) -> None:
    ax = _axes(fig, (0.08, 0.24, 0.86, 0.60))
    cats, series = d["categories"], d["series"]
    theme.check_series_count(len(series))
    x = np.arange(len(cats))
    ymax = max([d["nominal"]] + [v for s in series for v in s["ci_high"] if v is not None])
    for j, s in enumerate(series):
        xs = x + (j - (len(series) - 1) / 2) * 0.18
        vals = np.array([np.nan if v is None else v for v in s["values"]], dtype=float)
        ax.errorbar(xs, vals, yerr=_err(s), fmt="o", color=theme.series_color(s["slot"]), ecolor=theme.series_color(s["slot"]),
                    elinewidth=1.2, capsize=3, markersize=6)
    ax.axhline(d["nominal"], color=theme.TEXT_MUTED, linestyle="--", linewidth=1.0)
    ax.text(len(cats) - 0.55, d["nominal"], f" nominal {d['nominal']:.2f}", va="bottom", ha="right", fontsize=8.5, color=theme.TEXT_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylim(-0.005, min(1.0, ymax * 1.15 + 0.01))
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylabel(d["y_axis"]["label"])
    ax.set_xlabel(d["x_axis"]["label"])
    _legend(ax, series)
    if d.get("note"):
        fig.text(0.05, 0.07, d["note"], fontsize=8.5, color=theme.TEXT_MUTED, ha="left")


def _line_ci(fig: Figure, d: dict[str, Any]) -> None:
    ax = _axes(fig, (0.08, 0.14, 0.78, 0.70))
    theme.check_series_count(len(d["series"]))
    xs = np.array(d["x"], dtype=float)
    for s in d["series"]:
        color = theme.series_color(s["slot"])
        vals = np.array([np.nan if v is None else v for v in s["values"]], dtype=float)
        lo = np.array([np.nan if v is None else v for v in s["ci_low"]], dtype=float)
        hi = np.array([np.nan if v is None else v for v in s["ci_high"]], dtype=float)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(xs, vals, color=color, marker="o")
        ax.text(xs[-1], vals[-1], f"  {s['label']}", color=color, va="center", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(d["x_axis"]["label"])
    ax.set_ylabel(d["y_axis"]["label"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    _legend(ax, d["series"])


def _table(fig: Figure, d: dict[str, Any]) -> None:
    top = 0.84
    if d.get("method_differences"):
        line = "Method differences: " + "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in sorted(d["method_differences"].items()))
        fig.text(0.05, 0.89, line, fontsize=9, color=theme.STATUS["warning"], ha="left", fontweight="bold")
    if d.get("summary"):
        s = d["summary"]
        fig.text(0.05, 0.89, f"{d['source']}: evaluated {s['evaluated']}, skipped {s['skipped']}, changed {s['changed']} "
                             f"({s['stricter']} stricter, {s['looser']} looser)", fontsize=9, color=theme.TEXT, ha="left")
    _draw_table(fig, d["columns"], d["rows"], None, "Nothing to show.", top=top)
    for i, note in enumerate(d.get("notes") or []):
        fig.text(0.05, 0.062 - 0.02 * i, "\n".join(textwrap.wrap(note, 150)), fontsize=8, color=theme.TEXT_MUTED, ha="left")


DRAWERS = {"text": _card, "bars": _bars, "points_ci": _points_ci, "line_ci": _line_ci, "table": _table}


def draw_figure(d: dict[str, Any], bench_id: str, receipt_hash: str) -> Figure:
    @theme.themed
    def build() -> Figure:
        fig = _page(d, bench_id, receipt_hash)
        DRAWERS[d["chart_type"]](fig, d)
        return fig

    return build()


def render_pdf(path: str | Path, figures: Sequence[dict[str, Any]], bench_id: str, receipt_hash: str) -> None:
    figs = [draw_figure(d, bench_id, receipt_hash) for d in figures]
    write_pdf(path, figs, title=f"{bench_id} benchmark report", keywords=KEYWORDS,
              subject=f"bench {bench_id} receipt {receipt_hash}")

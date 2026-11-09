"""Figures: one pure function per page, drawn from the dictionaries in ``chartdata``.

Each function takes the verified ``StageData`` and returns a matplotlib Figure (no
pyplot state). Nothing here computes a statistic; it only draws what the chart data
says. Status is always icon plus word; categorical colors follow the fixed slots.
"""

from __future__ import annotations

import textwrap
from typing import Any

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from qrh.reports import chartdata as cd
from qrh.reports import theme
from qrh.reports.inputs import StageData

AXES_FULL = (0.08, 0.12, 0.86, 0.72)
AXES_LABELLED = (0.08, 0.12, 0.70, 0.72)  # room on the right for direct line-end labels
TABLE_ROWS_PER_PAGE = 22
MIN_BAR_FRACTION = 0.004  # of the axis span, so very short bars and gaps stay visible


# --- shared pieces ---------------------------------------------------------------------------


def _page(data: StageData, d: dict[str, Any], subtitle: str | None = None) -> Figure:
    fig = theme.new_page(data.trial_id, data.stage.value, data.receipt_hash, d["title"], subtitle)
    verdict = d["verdict"]
    fig.text(
        0.95, 0.955, theme.verdict_label(verdict), fontsize=12, fontweight="bold", ha="right", va="center",
        color=theme.verdict_color(verdict) or theme.TEXT,
    )
    return fig


def _axes(fig: Figure, rect: tuple[float, float, float, float] = AXES_FULL, grid: bool = True) -> Axes:
    ax = fig.add_axes(rect)
    if grid:
        ax.grid(True, linewidth=0.6)
    else:
        ax.grid(False)
    ax.tick_params(length=3)
    return ax


DIMENSIONLESS = {"", "ratio", "multiple", "count"}  # units that add nothing to an axis label


def _label(axis: dict[str, str]) -> str:
    unit = axis.get("unit", "")
    return axis["label"] if unit in DIMENSIONLESS else f"{axis['label']} ({unit})"


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _dt(values: list[str]) -> np.ndarray:
    return pd.to_datetime(values, utc=True).tz_localize(None).to_numpy()


def _time_axis(ax: Axes) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _vrule(ax: Axes, x: Any, label: str, slot: int = 0, align: str = "left") -> None:
    """A thin vertical rule with a direct text label, staggered by ``slot`` to avoid overlaps."""
    ax.axvline(x, color=theme.TEXT, linewidth=1.2, linestyle=(0, (5, 3)))
    sign = 1 if align == "left" else -1
    ax.annotate(
        label, xy=(x, 1.0), xycoords=("data", "axes fraction"), xytext=(5 * sign, -6 - 14 * slot),
        textcoords="offset points", ha=align, va="top", fontsize=8.5, color=theme.TEXT,
    )


def _message(ax: Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=11, color=theme.TEXT_MUTED)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Move label positions apart (in data units) so line-end labels do not overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def _line_chart(fig: Figure, d: dict[str, Any], dashed_second: bool = False, steps: bool = False) -> Axes:
    series = d["series"]
    theme.check_series_count(len(series))
    ax = _axes(fig, AXES_LABELLED)
    x_axis_is_time = d["x_axis"]["type"] == "time"
    ends: list[tuple[Any, float, str]] = []
    for i, s in enumerate(series):
        if not s["x"]:
            continue
        xs = _dt(s["x"]) if x_axis_is_time else np.asarray(s["x"], dtype=float)
        ys = np.array([np.nan if v is None else v for v in s["y"]], dtype=float)
        ax.plot(
            xs, ys, color=theme.series_color(i), linewidth=1.5, label=s["name"],
            linestyle="--" if dashed_second and i == 1 else "-", drawstyle="steps-post" if steps else "default",
        )
        finite = np.flatnonzero(np.isfinite(ys))
        if finite.size:
            ends.append((xs[finite[-1]], float(ys[finite[-1]]), s["name"]))
    if x_axis_is_time:
        _time_axis(ax)
    ax.set_xlabel(_label(d["x_axis"]))
    ax.set_ylabel(_label(d["y_axis"]))
    if ends:  # direct labels at the line ends (dark text, nudged apart)
        lo, hi = ax.get_ylim()
        placed = _spread([e[1] for e in ends], 0.07 * (hi - lo))
        for (x, _y, name), y_label in zip(ends, placed):
            ax.annotate(
                name, xy=(x, y_label), xytext=(6, 0), textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color=theme.TEXT, annotation_clip=False,
            )
    ax.margins(y=0.08)
    if len(series) >= 2:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(series), borderaxespad=0.3)
    return ax


def _histogram(ax: Axes, d: dict[str, Any]) -> None:
    edges = np.asarray(d["bins"], dtype=float)
    ax.bar(
        edges[:-1], d["counts"], width=np.diff(edges), align="edge", color=theme.series_color(0),
        edgecolor=theme.SURFACE, linewidth=0.6,
    )
    ax.set_xlabel(_label(d["x_axis"]))
    ax.set_ylabel(_label(d["y_axis"]))


def _draw_table(
    fig: Figure, columns: list[dict[str, str]], rows: list[dict[str, Any]], status_column: str | None,
    empty_message: str = "", top: float = 0.84, limit: int = TABLE_ROWS_PER_PAGE,
) -> None:
    ax = fig.add_axes((0.05, 0.08, 0.90, top - 0.08))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if not rows:
        _message(ax, empty_message or "Nothing to show.")
        return
    shown = rows[:limit]
    weights = []
    for c in columns:
        cells = [
            len(theme.verdict_label(r[c["key"]])) if c["key"] == status_column and isinstance(r.get(c["key"]), str)
            else len(_fmt(r.get(c["key"])))
            for r in shown
        ]
        longest = max([len(c["label"])] + cells)
        weights.append(float(min(max(longest, 6), 46)))
    widths = [w / sum(weights) for w in weights]
    starts = np.concatenate(([0.0], np.cumsum(widths)))[:-1]
    row_h = min(0.07, 0.93 / (len(shown) + 1))
    y = 1.0 - row_h / 2
    for c, x0 in zip(columns, starts):
        ax.text(x0 + 0.004, y, c["label"], fontsize=9, fontweight="bold", va="center", ha="left", color=theme.TEXT, parse_math=False)
    ax.plot([0, 1], [1.0 - row_h, 1.0 - row_h], color=theme.AXIS, linewidth=0.8)
    for i, r in enumerate(shown):
        y = 1.0 - row_h * (i + 1.5)
        if i % 2 == 1:
            ax.axhspan(y - row_h / 2, y + row_h / 2, color=theme.GRID, alpha=0.45, linewidth=0)
        for c, x0, w in zip(columns, starts, widths):
            value = r.get(c["key"])
            if c["key"] == status_column and isinstance(value, str):
                ax.text(
                    x0 + 0.004, y, theme.verdict_label(value), fontsize=9, fontweight="bold", va="center",
                    ha="left", color=theme.verdict_color(value) or theme.TEXT, parse_math=False,
                )
                continue
            capacity = max(4, int(w * 118))
            text = _fmt(value)
            if len(text) > capacity:
                text = text[: capacity - 1] + "…"
            ax.text(x0 + 0.004, y, text, fontsize=8.5, va="center", ha="left", color=theme.TEXT, parse_math=False)
    if len(rows) > len(shown):
        ax.text(0, 0.0, f"{len(rows) - len(shown)} more row(s) not shown here; all rows are in the chart data JSON.",
                fontsize=8.5, color=theme.TEXT_MUTED, va="bottom", parse_math=False)


def _table_page(data: StageData, d: dict[str, Any]) -> Figure:
    fig = _page(data, d)
    _draw_table(fig, d["columns"], d["rows"], d.get("status_column"), d.get("empty_message", ""))
    return fig


def _timeline(fig: Figure, d: dict[str, Any], spans: list[list[tuple[str, str]]], slot_for: list[int],
              extra: list[list[tuple[str, str]]] | None = None) -> Axes:
    """Rows of horizontal bars on a time axis; ``spans[i]`` are (start, end) pairs drawn in slot ``slot_for``."""
    ax = _axes(fig, AXES_FULL)
    ax.yaxis.grid(False)
    lows, highs = [], []
    for row in spans:
        for a, b in row:
            lows.append(mdates.date2num(_dt([a])[0])), highs.append(mdates.date2num(_dt([b])[0]))
    for row in (extra or []):
        for a, b in row:
            lows.append(mdates.date2num(_dt([a])[0])), highs.append(mdates.date2num(_dt([b])[0]))
    lo, hi = (min(lows), max(highs)) if lows else (0.0, 1.0)
    minimum = max(hi - lo, 1.0) * MIN_BAR_FRACTION
    for i, row in enumerate(spans):
        for (a, b), slot in zip(row, slot_for):
            x0, x1 = mdates.date2num(_dt([a])[0]), mdates.date2num(_dt([b])[0])
            ax.broken_barh([(x0, max(x1 - x0, minimum))], (i - 0.3, 0.6), facecolors=theme.series_color(slot))
        for a, b in (extra[i] if extra else []):
            x0, x1 = mdates.date2num(_dt([a])[0]), mdates.date2num(_dt([b])[0])
            ax.broken_barh([(x0, max(x1 - x0, minimum))], (i - 0.4, 0.8), facecolors=theme.series_color(1))
    ax.set_ylim(len(spans) - 0.5, -0.5)
    ax.set_xlim(lo - (hi - lo) * 0.02, hi + (hi - lo) * 0.02)
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels([r["label"] for r in d["rows"]])
    ax.xaxis_date()
    _time_axis(ax)
    ax.set_xlabel(_label(d["x_axis"]))
    return ax


# --- S0 / status ---------------------------------------------------------------------------


def _text_card(data: StageData, d: dict[str, Any], left: list[tuple[str, Any]], right_title: str,
               right: list[tuple[str, Any]], left_title: str) -> Figure:
    fig = _page(data, d)
    ax = fig.add_axes((0.05, 0.08, 0.90, 0.76))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def column(x: float, title: str, items: list[tuple[str, Any]], wrap: int) -> None:
        ax.text(x, 1.0, title, fontsize=11, fontweight="bold", va="top", color=theme.TEXT)
        y = 0.94
        for label, value in items:
            ax.text(x, y, label, fontsize=8.5, color=theme.TEXT_MUTED, va="top")
            lines = _break(_fmt(value), wrap)
            ax.text(x, y - 0.03, "\n".join(lines), fontsize=9.5, color=theme.TEXT, va="top")
            y -= 0.03 + 0.032 * len(lines) + 0.022

    column(0.0, left_title, left, 62)
    column(0.66, right_title, right, 30)
    return fig


@theme.themed
def _break(text: str, width: int) -> list[str]:
    """Wrap text; one long unbroken token (a hash) is split into equal lines, never mid-way at an arbitrary column."""
    if " " not in text and len(text) > width:
        n = -(-len(text) // width)
        size = -(-len(text) // n)
        return [text[i:i + size] for i in range(0, len(text), size)]
    return textwrap.wrap(text, width) or [""]


def s0_card(data: StageData) -> Figure:
    d = cd.s0_card(data)
    return _text_card(
        data, d, [(f["label"], f["value"]) for f in d["fields"]], "Thresholds in force",
        [(t["name"], t["value"]) for t in d["thresholds"]], "Hypothesis",
    )


@theme.themed
def status_card(data: StageData) -> Figure:
    d = cd.status_card(data)
    left = [("Verdict", theme.verdict_label(d["verdict"]))]
    left += [(f"Reason {i + 1}", r) for i, r in enumerate(d["reasons"])] or [("Reasons", "none recorded")]
    left.append(("Note", d["note"]))
    return _text_card(data, d, left, "Metrics", [(m["name"], m["value"]) for m in d["metrics"]] or [("", "none recorded")], "Status")


# --- S1 -------------------------------------------------------------------------------------


@theme.themed
def s1_coverage(data: StageData) -> Figure:
    d = cd.s1_coverage(data)
    fig = _page(data, d, "Bars span first to last timestamp; gaps are steps longer than the dataset's usual step")
    rows = [r for r in d["rows"] if r["start"] and r["end"]]
    if not rows:
        _message(_axes(fig), "No dataset coverage was recorded.")
        return fig
    d_rows = {**d, "rows": rows}
    ax = _timeline(
        fig, d_rows, [[(r["start"], r["end"])] for r in rows], [0],
        extra=[[(g["after"], g["before"]) for g in r["gaps"]] for r in rows],
    )
    for i, r in enumerate(rows):
        ax.annotate(f"{r['rows']} rows, {r['gap_count']} gap(s)", xy=(1.0, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=theme.TEXT, annotation_clip=False)
    ax.legend(handles=[Patch(color=theme.series_color(0), label="Coverage"), Patch(color=theme.series_color(1), label="Gap")], loc="lower left")
    fig.axes[0].set_position((0.12, 0.12, 0.66, 0.72))
    return fig


@theme.themed
def s1_rules(data: StageData) -> Figure:
    return _table_page(data, cd.s1_rules(data))


@theme.themed
def s1_dataset_card(data: StageData, index: int) -> Figure:
    d = cd.s1_dataset_card(data, index)
    fig = _page(data, d, d["subtitle"])
    ax = fig.add_axes((0.05, 0.50, 0.90, 0.36))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def block(x: float, title: str, lines: list[str], wrap: int) -> None:
        ax.text(x, 1.0, title, fontsize=11, fontweight="bold", va="top", color=theme.TEXT, parse_math=False)
        y = 0.90
        for line in lines:
            parts = _break(line, wrap)
            ax.text(x, y, "\n".join(parts), fontsize=8.8, va="top", color=theme.TEXT, parse_math=False)
            y -= 0.075 * len(parts)

    block(0.0, "Profile (aggregates only, never rows)", [f"{f['label']}: {_fmt(f['value'])}" for f in d["facts"]], 60)
    block(0.50, "Candidate relationships and name hints (not facts)", d["relationships"] + d["hints"] or ["none found"], 62)
    if d["columns"]:
        _draw_table(fig, d["column_table"], d["columns"], None, top=0.47, limit=11)
    else:
        _message(_axes(fig, (0.05, 0.08, 0.90, 0.38), grid=False), "No profile is stored for this dataset: run `qrh data profile`.")
    return fig


ONBOARDING_ROWS_PER_PAGE = 14


@theme.themed
def s1_onboarding(data: StageData, index: int) -> Figure:
    """One block per field: the claim, its source and a status written as text with a marker, then the evidence."""
    d = cd.s1_onboarding(data, index)
    fig = _page(data, d, d["subtitle"])
    ax = fig.add_axes((0.05, 0.06, 0.90, 0.80))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rows = d["rows"][:ONBOARDING_ROWS_PER_PAGE]
    if not rows:
        _message(ax, d.get("empty_message", "Nothing to show."))
        return fig
    xs = (0.0, 0.26, 0.58, 0.68)
    for x, label in zip(xs, ("Field (* critical)", "Claim", "Source", "Status")):
        ax.text(x + 0.004, 0.985, label, fontsize=9, fontweight="bold", va="center", color=theme.TEXT, parse_math=False)
    ax.plot([0, 1], [0.955, 0.955], color=theme.AXIS, linewidth=0.8)
    block = min(0.085, 0.935 / len(rows))
    for i, r in enumerate(rows):
        top = 0.955 - block * i
        if i % 2 == 1:
            ax.axhspan(top - block, top, color=theme.GRID, alpha=0.45, linewidth=0)
        y = top - block * 0.2
        claim = _fmt(r["claim"])
        ax.text(xs[0] + 0.004, y, r["field"], fontsize=9, va="center", color=theme.TEXT, parse_math=False)
        ax.text(xs[1] + 0.004, y, claim if len(claim) <= 38 else claim[:37] + "…", fontsize=9, va="center", color=theme.TEXT, parse_math=False)
        ax.text(xs[2] + 0.004, y, r["source"], fontsize=9, va="center", color=theme.TEXT, parse_math=False)
        ax.text(xs[3] + 0.004, y, theme.verdict_label(r["status"]), fontsize=9, fontweight="bold", va="center",
                color=theme.verdict_color(r["status"]) or theme.TEXT, parse_math=False)
        if r["evidence"]:
            lines = textwrap.wrap(r["evidence"], 150)[:2]
            ax.text(xs[0] + 0.012, y - block * 0.22, "\n".join(lines), fontsize=7.5, va="top", color=theme.TEXT_MUTED, parse_math=False,
                    linespacing=1.15)
    if len(d["rows"]) > len(rows):
        ax.text(0, 0.0, f"{len(d['rows']) - len(rows)} more field(s) not shown here; all are in the chart data JSON.",
                fontsize=8.5, color=theme.TEXT_MUTED, va="bottom", parse_math=False)
    return fig


# --- S2 -------------------------------------------------------------------------------------


@theme.themed
def s2_lint(data: StageData) -> Figure:
    return _table_page(data, cd.s2_lint(data))


@theme.themed
def s2_truncation(data: StageData, index: int) -> Figure:
    d = cd.s2_truncation(data, index)
    sub = f"Cutoff {_short(d['cutoff'])} ({d['cutoff_fraction']:.0%} of the window), result: {theme.verdict_label(d['cutoff_status'])}"
    if d["columns_total"] > 1:
        sub += f"; showing 1 of {d['columns_total']} position columns"
    fig = _page(data, d, sub)
    if d["empty_message"]:
        _message(_axes(fig), d["empty_message"])
        return fig
    ax = _line_chart(fig, d, dashed_second=True, steps=True)
    for m in d["markers"]:
        _vrule(ax, _dt([m["x"]])[0], m["label"])
    return fig


def _short(iso: str) -> str:
    return cd._short_ts(iso)


@theme.themed
def s2_llm_gate(data: StageData) -> Figure:
    d = cd.s2_llm_gate(data)
    margin = d["margin_days"]
    sub = (f"Every window must start after the latest model training cutoff plus a {margin:g}-day margin"
           if margin is not None else "Every window must start after the latest model training cutoff plus a margin")
    fig = _page(data, d, sub)
    rows = d["rows"]
    ax = _timeline(fig, d, [[(r["start"], r["end"])] for r in rows], [0])
    dates = [mdates.date2num(_dt([x])[0]) for r in rows for x in (r["start"], r["end"])]
    dates += [mdates.date2num(_dt([m["x"]])[0]) for m in d["markers"]]
    lo, hi = min(dates), max(dates)
    pad = max(hi - lo, 1.0) * 0.04
    ax.set_xlim(lo - pad, hi + pad)
    hatched = False
    if d["binding_date"]:
        zone_end = mdates.date2num(_dt([d["binding_date"]])[0])
        ax.axvspan(lo - pad, zone_end, color=theme.GRID, alpha=0.7, linewidth=0, zorder=0)
        ax.annotate("Possibly in the training data", xy=(0.0, 1.0), xycoords="axes fraction",  # above the plot: no line crosses it
                    xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=theme.TEXT_MUTED, va="bottom")
        for i, r in enumerate(rows):  # the part of a window inside the zone is hatched, so it reads without color
            x0, x1 = mdates.date2num(_dt([r["start"]])[0]), mdates.date2num(_dt([r["end"]])[0])
            if x0 <= zone_end:
                ax.broken_barh([(x0, min(x1, zone_end) - x0)], (i - 0.3, 0.6), facecolors="none", edgecolor=theme.TEXT,
                               hatch="////", linewidth=0.0, zorder=3)
                hatched = True
    for slot, m in enumerate(d["markers"]):
        _vrule(ax, _dt([m["x"]])[0], m["label"], slot=slot, align="left")
    for i, r in enumerate(rows):
        ax.annotate(theme.verdict_label(r["status"]), xy=(1.0, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold",
                    color=theme.verdict_color(r["status"]) or theme.TEXT, annotation_clip=False)
    if hatched:
        ax.legend(handles=[Patch(facecolor="none", edgecolor=theme.TEXT, hatch="////", label="Window time inside the possibly-trained zone")],
                  loc="lower right")
    fig.axes[0].set_position((0.12, 0.34, 0.60, 0.50))
    y = 0.27
    fig.text(0.05, y, f"Gate: {theme.verdict_label(d['gate_verdict'])}", fontsize=10.5, fontweight="bold", va="top",
             color=theme.verdict_color(d["gate_verdict"]) or theme.TEXT)
    for m in d["models"]:
        y -= 0.035
        cutoff = m["training_cutoff"] if m["training_cutoff"] != "unknown" else "unknown (BLOCKED: declare a real cutoff)"
        src = m["cutoff_source"] if m["cutoff_source"] != "unknown" else "unknown (BLOCKED: declare its source)"
        fig.text(0.05, y, f"Model {m['id']} ({m['provider']}): cutoff {cutoff}, source {src}", fontsize=9, va="top",
                 color=theme.TEXT)
    if d["live_retrieval"]["used"]:
        y -= 0.035
        evidence = d["live_retrieval"]["evidence"]
        text = (f"Live retrieval or tools used; evidence given: {evidence}" if evidence
                else "Live retrieval or tools used; no evidence text was given (BLOCKED)")
        lines = textwrap.wrap(text, 115)[:3]
        fig.text(0.05, y, "\n".join(lines), fontsize=9, va="top", color=theme.TEXT)
        y -= 0.025 * (len(lines) - 1)
    for row in rows:
        if row["earliest_contaminated_date"]:
            y -= 0.035
            fig.text(0.05, y, f"Window {row['label']!r}: earliest contaminated date {row['earliest_contaminated_date']}",
                     fontsize=9, va="top", color=theme.TEXT)
    for reason in d["reasons"][:2]:
        y -= 0.035
        lines = textwrap.wrap(reason, 125)[:3]
        fig.text(0.05, y, "\n".join(lines), fontsize=8.5, va="top", color=theme.TEXT_MUTED)
        y -= 0.022 * (len(lines) - 1)
    return fig


@theme.themed
def s2_sanity(data: StageData) -> Figure:
    return _table_page(data, cd.s2_sanity(data))


# --- S3 -------------------------------------------------------------------------------------


@theme.themed
def s3_heatmap(data: StageData) -> Figure:
    d = cd.s3_heatmap(data)
    fig = _page(data, d, "Each cell is one parameter combination evaluated on the development window")
    ax = _axes(fig, (0.12, 0.12, 0.72, 0.72), grid=False)
    z = np.array([[np.nan if v is None else v for v in row] for row in d["z"]], dtype=float)
    cmap = theme.sequential_cmap().with_extremes(bad=theme.GRID)
    im = ax.imshow(z, cmap=cmap, aspect="auto", vmin=d["z_min"], vmax=d["z_max"])
    ax.set_xticks(range(len(d["x_labels"])), d["x_labels"])
    ax.set_yticks(range(len(d["y_labels"])), d["y_labels"])
    ax.set_xlabel(d["x_axis"]["label"])
    ax.set_ylabel(d["y_axis"]["label"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), v in np.ndenumerate(z):
        if np.isfinite(v):
            rgba = cmap(im.norm(v))
            color = "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in rgba[:3]))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=theme.text_on(color))
    cax = fig.add_axes((0.87, 0.12, 0.02, 0.72))
    fig.colorbar(im, cax=cax).set_label(d["z_label"])
    return fig


@theme.themed
def s3_ranked(data: StageData) -> Figure:
    d = cd.s3_ranked(data)
    fig = _page(data, d, f"Top {d['shown']} of {d['total']} combinations by in-sample Sharpe")
    ax = _axes(fig, (0.30, 0.14, 0.64, 0.70))
    ax.yaxis.grid(False)
    ys = np.arange(len(d["rows"]))
    ax.plot([r["value"] for r in d["rows"]], ys, linestyle="none", marker="o", markersize=8, color=theme.series_color(0))
    ax.set_yticks(ys, [r["label"] for r in d["rows"]], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(_label(d["x_axis"]))
    if d["rest_count"]:
        fig.text(0.05, 0.07, f"... and {d['rest_count']} more combinations not shown", fontsize=9, color=theme.TEXT_MUTED)
    return fig


@theme.themed
def s3_hist(data: StageData) -> Figure:
    d = cd.s3_hist(data)
    fig = _page(data, d)
    _histogram(_axes(fig), d)
    return fig


# --- S4 -------------------------------------------------------------------------------------


@theme.themed
def s4_folds(data: StageData) -> Figure:
    d = cd.s4_folds(data)
    fig = _page(data, d, "Selection uses the train span only; the test span is out of sample")
    ax = _timeline(fig, d, [[tuple(r["train"]), tuple(r["test"])] for r in d["rows"]], [0, 1])
    ax.legend(handles=[Patch(color=theme.series_color(0), label=d["series_names"][0]),
                       Patch(color=theme.series_color(1), label=d["series_names"][1])], loc="lower left")
    return fig


@theme.themed
def s4_equity(data: StageData) -> Figure:
    d = cd.s4_equity(data)
    fig = _page(data, d, "Test spans of all folds stitched in time order")
    _line_chart(fig, d)
    return fig


@theme.themed
def s4_params(data: StageData) -> Figure:
    return _table_page(data, cd.s4_params(data))


# --- S5 -------------------------------------------------------------------------------------


@theme.themed
def s5_lambda(data: StageData) -> Figure:
    d = cd.s5_lambda(data)
    fig = _page(data, d, "Lambda <= 0 means the in-sample winner ranked in the lower half out of sample")
    ax = _axes(fig)
    _histogram(ax, d)
    for m in d["markers"]:
        _vrule(ax, m["x"], m["label"])
    return fig


@theme.themed
def s5_dsr(data: StageData) -> Figure:
    d = cd.s5_dsr(data)
    fig = _page(data, d, "Per-period (not annualized) Sharpe ratios")
    ax = _axes(fig, (0.12, 0.22, 0.60, 0.62))
    ax.xaxis.grid(False)
    bars = d["bars"]
    theme.check_series_count(len(bars))
    values = [b["value"] for b in bars]
    ax.bar(range(len(bars)), values, color=[theme.series_color(i) for i in range(len(bars))], width=0.55)
    ax.axhline(0, color=theme.AXIS, linewidth=0.8)
    ax.set_xticks(range(len(bars)), [textwrap.fill(b["name"], 22) for b in bars])
    ax.set_ylabel(_label(d["y_axis"]))
    for i, v in enumerate(values):
        ax.annotate(f"{v:.4f}", xy=(i, v), xytext=(0, 4 if v >= 0 else -12), textcoords="offset points", ha="center", fontsize=9, color=theme.TEXT)
    ax.legend(handles=[Patch(color=theme.series_color(i), label=b["name"]) for i, b in enumerate(bars)], loc="upper right")
    ax.text(0.5, 1.02, d["annotations"]["label"], transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold", color=theme.TEXT)
    ins = d["inputs"]
    fig.text(0.76, 0.80, "DSR inputs", fontsize=10, fontweight="bold", va="top", color=theme.TEXT)
    for i, item in enumerate(ins):
        fig.text(0.76, 0.76 - 0.045 * i, f"{item['name']}: {_fmt(item['value'])}", fontsize=9, va="top", color=theme.TEXT)
    return fig


@theme.themed
def s5_stability(data: StageData) -> Figure:
    return _table_page(data, cd.s5_stability(data))


# --- S6 -------------------------------------------------------------------------------------


@theme.themed
def s6_bootstrap(data: StageData) -> Figure:
    d = cd.s6_bootstrap(data)
    fig = _page(data, d, f"{d['reps']} stationary-bootstrap replications of the out-of-sample returns; confidence level {d['level'] * 100:g}%")
    n = len(d["panels"])
    height = 0.20
    for i, p in enumerate(d["panels"]):
        bottom = 0.10 + (n - 1 - i) * 0.265
        ax = _axes(fig, (0.08, bottom, 0.86, height))
        _histogram(ax, p)
        ax.set_title(p["title"], loc="left", fontsize=10, color=theme.TEXT)
        ci = 0
        for m in p["markers"]:
            if m["role"] == "ci":
                _vrule(ax, m["x"], m["label"], slot=0, align="right" if ci == 0 else "left")
                ci += 1
            else:
                ax.axvline(m["x"], color=theme.TEXT, linewidth=1.0, linestyle=":")
                ax.annotate(m["label"], xy=(m["x"], 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                            textcoords="offset points", fontsize=8.5, ha="center", va="bottom", color=theme.TEXT)
        if p["undefined_share"]:
            ax.text(0.99, 0.05, f"{p['undefined_share']:.1%} of replications undefined (excluded)", transform=ax.transAxes,
                    ha="right", fontsize=8, color=theme.TEXT_MUTED)
    return fig


@theme.themed
def s6_equity(data: StageData) -> Figure:
    d = cd.s6_equity(data)
    fig = _page(data, d, "Same selected parameters; costs scaled by the stress multiplier")
    _line_chart(fig, d)
    return fig


# --- S7 -------------------------------------------------------------------------------------


CONDITIONS_SHOWN = 4


def _draw_conditions(fig: Figure, conditions: list[dict[str, Any]]) -> None:
    """The open conditions of a PASS · CONDITIONAL: one block per condition (id, status, question, risk)."""
    fig.text(0.05, 0.285, "Conditions (every gate passed, these remain open)", fontsize=10, fontweight="bold",
             va="top", color=theme.STATUS["warning"])
    y = 0.255
    for c in conditions[:CONDITIONS_SHOWN]:
        head = f"{theme.VERDICT_ICON['INCONCLUSIVE']} {c['question_id']}  [{c['status']}, {c['stage']}]  {c['text']}"
        fig.text(0.05, y, textwrap.shorten(head, 150, placeholder="…"), fontsize=8.5, va="top", color=theme.TEXT)
        if c.get("risk"):
            fig.text(0.07, y - 0.022, "Risk: " + textwrap.shorten(str(c["risk"]), 140, placeholder="…"), fontsize=8,
                     va="top", color=theme.TEXT_MUTED)
        y -= 0.052
    if len(conditions) > CONDITIONS_SHOWN:
        fig.text(0.05, y, f"{len(conditions) - CONDITIONS_SHOWN} more condition(s) in the chart data JSON.",
                 fontsize=8.5, va="top", color=theme.TEXT_MUTED)


@theme.themed
def s7_summary(data: StageData) -> Figure:
    d = cd.s7_summary(data)
    fig = _page(data, d)
    final = d["final"]["verdict"]
    fig.text(0.05, 0.895, f"Final verdict: {theme.verdict_label(final)}", fontsize=13, fontweight="bold", va="center",
             color=theme.verdict_color(final) or theme.TEXT)
    if d["final"]["failing_stage"]:
        fig.text(0.40, 0.895, f"(first non-PASS stage: {d['final']['failing_stage']})", fontsize=10, va="center", color=theme.TEXT_MUTED)
    rows = []
    for s in d["stages"]:
        detail = ", ".join(f"{k}={_fmt(v)}" for k, v in s["key_metrics"].items())
        if s["verdict"] not in ("PASS", "NOT RUN") and s["reasons"]:
            detail = s["reasons"][0]
        rows.append({"stage": s["stage"], "name": s["title"], "verdict": s["verdict"], "receipt": s["receipt_hash"] or "", "detail": detail})
    columns = [
        {"key": "stage", "label": "Stage"}, {"key": "name", "label": "Name"}, {"key": "verdict", "label": "Verdict"},
        {"key": "receipt", "label": "Receipt"}, {"key": "detail", "label": "Key metrics / first reason"},
    ]
    _draw_table(fig, columns, rows, "verdict", top=0.86)
    if d["conditions"]:
        _draw_conditions(fig, d["conditions"])
    if d["non_default_methods"]:
        text = ", ".join(f"{k}={v}" for k, v in sorted(d["non_default_methods"].items()))
        fig.text(0.05, 0.055, f"{theme.VERDICT_ICON['INCONCLUSIVE']} EXPERIMENTAL methods used: {text}", fontsize=9,
                 fontweight="bold", color=theme.STATUS["warning"], ha="left")
    return fig

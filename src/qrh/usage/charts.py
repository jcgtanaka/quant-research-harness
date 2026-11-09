"""Draw the usage figures from the dictionaries in ``chartdata``. Nothing here computes a statistic."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from qrh.reports import theme
from qrh.reports.charts import AXES_FULL, _axes, _draw_table, _message

MAX_TICKS = 12
MIN_BAR_SLOTS = 6


def _page(d: dict[str, Any]) -> Figure:
    r = d["range"]
    span = f"{r['first']} to {r['last']}" if r["first"] else "no sessions"
    return theme.new_report_page(d["title"], None, f"Usage report  |  {r['sessions']} session(s)  |  {span}")


def _thin_ticks(ax: Any, labels: list[str]) -> None:
    step = max(1, -(-len(labels) // MAX_TICKS))
    idx = list(range(0, len(labels), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=30, ha="right")


def _fmt_axis(axis: dict[str, str]) -> str:
    return f"{axis['label']} ({axis['unit']})" if axis.get("unit") else axis["label"]


def text_card(d: dict[str, Any]) -> Figure:
    fig = _page(d)
    ax = fig.add_axes((0.05, 0.1, 0.9, 0.75))
    ax.axis("off")
    y = 0.95
    for f in d["fields"]:
        ax.text(0.0, y, f["label"], fontsize=10, color=theme.TEXT_MUTED, va="center")
        ax.text(0.28, y, str(f["value"]), fontsize=11, fontweight="bold", color=theme.TEXT, va="center")
        y -= 0.07
    for note in d["notes"][:6]:
        y -= 0.02
        ax.text(0.0, y, note[:150], fontsize=8.5, color=theme.TEXT_MUTED, va="center")
        y -= 0.04
    return fig


def stacked_bar(d: dict[str, Any]) -> Figure:
    fig = _page(d)
    ax = _axes(fig)
    if not d["x"] or not d["series"]:
        _message(ax, "No cost data.")
        return fig
    x = np.arange(len(d["x"]))
    bottom = np.zeros(len(x))
    for i, s in enumerate(d["series"]):
        color = theme.series_color(i)
        vals = np.array(s["values"], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, label=s["name"], width=0.8)
        bottom += vals
    _thin_ticks(ax, d["x"])
    ax.set_ylabel(_fmt_axis(d["y_axis"]))
    ax.set_xlabel(d["x_axis"]["label"])
    ax.legend(loc="upper left")
    return fig


def bars(d: dict[str, Any], ranked: bool = False) -> Figure:
    fig = _page(d)
    if not d["bars"]:
        _message(_axes(fig), "Nothing to show.")
        return fig
    ax = _axes(fig, (0.22, 0.12, 0.72, 0.72))
    labels = [b["label"] for b in d["bars"]]
    values = [b["value"] for b in d["bars"]]
    y = np.arange(len(labels))
    ax.barh(y, values, color=theme.series_color(0), height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    top = max(values) or 1
    for yi, v in zip(y, values):
        ax.text(v + top * 0.01, yi, f"{v:,.2f}" if ranked else f"{v:,}", va="center", fontsize=8.5, color=theme.TEXT)
    ax.set_xlim(0, top * 1.3)
    ax.set_ylim(max(len(labels), MIN_BAR_SLOTS) - 0.5, -0.5)  # few bars stay slim instead of filling the page
    if not ranked:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.set_xlabel(_fmt_axis(d["x_axis"]))
    if d.get("truncated"):
        fig.text(0.05, 0.06, f"{d['truncated']} more row(s) are in the chart data JSON.", fontsize=8.5,
                 color=theme.TEXT_MUTED)
    return fig


def line(d: dict[str, Any]) -> Figure:
    fig = _page(d)
    ax = _axes(fig, AXES_FULL)
    pts = [(i, v) for i, v in enumerate(d["series"][0]["values"]) if v is not None] if d["series"] else []
    if not pts:
        _message(ax, "No cache data.")
        return fig
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=theme.series_color(0), marker="o")
    _thin_ticks(ax, d["x"])
    ax.set_ylim(0, 100)
    ax.set_ylabel(_fmt_axis(d["y_axis"]))
    ax.set_xlabel(d["x_axis"]["label"])
    return fig


def table(d: dict[str, Any]) -> Figure:
    fig = _page(d)
    _draw_table(fig, d["columns"], d["rows"], None, d.get("empty_message", ""))
    return fig


def draw(d: dict[str, Any]) -> Figure:
    kind = d["chart_type"]
    if kind == "text":
        return text_card(d)
    if kind == "stacked_bar":
        return stacked_bar(d)
    if kind in ("bar", "bar_ranked"):
        return bars(d, ranked=kind == "bar_ranked")
    if kind == "line":
        return line(d)
    if kind == "table":
        return table(d)
    raise ValueError(f"unknown chart type {kind!r}")


draw = theme.themed(draw)  # every figure is built under the theme's rc settings

"""Visual theme: one place for colors, fonts and page furniture.

Rules the figures follow (see docs/architecture.md, "Visual reports"):
- categorical series take the fixed slots in order and never cycle; at most 3 per chart;
- the sequential ramp is for magnitude only (heatmaps);
- status colors mark verdicts only and always come with a text label;
- text is dark gray, never a series color; grid lines are light and thin.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import matplotlib

matplotlib.use("Agg")

from matplotlib import rc_context  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#2b2d31"
TEXT_MUTED = "#5b5f66"
GRID = "#e6e6e2"
AXIS = "#9a9c9f"

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQUENTIAL = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

VERDICT_STATUS = {"PASS": "good", "PASS_CONDITIONAL": "warning", "INCONCLUSIVE": "warning", "FAIL": "critical", "BLOCKED": "serious"}
VERDICT_ICON = {"PASS": "✓", "PASS_CONDITIONAL": "◆", "INCONCLUSIVE": "◆", "FAIL": "✗", "BLOCKED": "▲"}
# Words used in tables that read as verdicts: they borrow the status color and icon of the verdict they mean.
VERDICT_ALIAS = {"BLOCK": "FAIL", "WARN": "INCONCLUSIVE", "VERIFIED": "PASS", "CONFIRMED": "PASS",
                 "UNVERIFIED": "INCONCLUSIVE", "CONTRADICTED": "FAIL"}  # onboarding statuses

MAX_SERIES = 3
PAGE_SIZE = (11.0, 8.5)  # inches, landscape letter
DISCLAIMER = "Synthetic/research use only, not investment advice."
FONT = "DejaVu Sans"

RC: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": [FONT],
    "font.size": 9,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11,
    "axes.titlecolor": TEXT,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "lines.linewidth": 1.5,
    "lines.markersize": 8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "pdf.fonttype": 3,
    "svg.hashsalt": "qrh-reports",
    "figure.dpi": 100,
    "savefig.dpi": 100,
}

F = TypeVar("F", bound=Callable[..., Any])


def themed(fn: F) -> F:
    """Run a figure builder under the theme's rc settings."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with rc_context(RC):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def series_color(index: int) -> str:
    """Categorical color for the series in slot ``index`` (0-based); slots never repeat."""
    if not 0 <= index < len(CATEGORICAL):
        raise ValueError(f"no categorical slot {index}: slots are fixed and never cycled")
    return CATEGORICAL[index]


def check_series_count(n: int) -> None:
    if n > MAX_SERIES:
        raise ValueError(f"at most {MAX_SERIES} categorical series on one chart, got {n}")


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("qrh_blue", list(SEQUENTIAL), N=256)


def verdict_color(verdict: str) -> str | None:
    """Status color of a verdict word; None for anything that is not a verdict (for example NOT RUN)."""
    key = VERDICT_STATUS.get(VERDICT_ALIAS.get(verdict, verdict))
    return STATUS[key] if key else None


def verdict_label(verdict: str) -> str:
    """Icon plus the verdict word, so a status is never carried by color alone."""
    icon = VERDICT_ICON.get(VERDICT_ALIAS.get(verdict, verdict), "•")
    return f"{icon} {verdict.replace('PASS_CONDITIONAL', 'PASS · CONDITIONAL')}"


def text_on(background: str) -> str:
    """Dark or white text, whichever reads better on ``background``."""
    r, g, b = to_rgb(background)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return TEXT if luminance > 0.55 else "#ffffff"


def new_report_page(title: str, subtitle: str | None, footer: str) -> Figure:
    """A landscape page with title and the standard disclaimer; ``footer`` is the left footer text."""
    fig = Figure(figsize=PAGE_SIZE, facecolor=SURFACE)
    fig.text(0.05, 0.955, title, fontsize=15, fontweight="bold", ha="left", va="center", color=TEXT)
    if subtitle:
        fig.text(0.05, 0.915, subtitle, fontsize=9.5, ha="left", va="center", color=TEXT_MUTED)
    fig.text(0.05, 0.025, footer, fontsize=8, ha="left", va="center", color=TEXT_MUTED)
    fig.text(0.95, 0.025, DISCLAIMER, fontsize=8, ha="right", va="center", color=TEXT_MUTED)
    return fig


def new_page(trial_id: str, stage: str, receipt_hash: str, title: str, subtitle: str | None = None) -> Figure:
    """A landscape page with title and the standard footer. Must be called under ``themed``."""
    footer = f"Trial {trial_id}  |  Stage {stage}  |  Receipt {receipt_hash[:12]}"
    return new_report_page(title, subtitle, footer)

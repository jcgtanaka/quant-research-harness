"""Future-truncation invariance test: the proof that a run does not peek ahead.

The adapter runs on all the data and again on data cut just before a cutoff
bar. Everything it decided strictly before the cutoff must be identical (up to
a tiny float tolerance). A strategy that uses information from after a bar to
decide that bar cannot satisfy this. Only the tested cutoffs are covered, so a
PASS is evidence, not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, RunResult, StrategyAdapter
from qrh.llm.runtime import LLMCacheMiss, replay_only

DEFAULT_CUTOFFS = (0.25, 0.5, 0.75)
DEFAULT_ATOL = 1e-12
MIN_BARS = 4
MAX_SERIES_POINTS = 2000  # stored points per comparison series; longer ones are decimated


@dataclass
class TruncationResult:
    verdict: str  # PASS | FAIL | BLOCKED
    reasons: list[str]
    details: dict[str, Any] = field(default_factory=dict)
    series: list[dict[str, Any]] = field(default_factory=list)  # per cutoff: full vs truncated positions


def _as_frame(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()


def _time_axis(df: pd.DataFrame) -> pd.Series | None:
    """Timestamps of a frame: its DatetimeIndex, else its first datetime column."""
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return pd.Series(pd.DatetimeIndex(df[col]), index=df.index)
    return None


def _py(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


def _first_difference(
    full: pd.DataFrame, cut: pd.DataFrame, before: pd.Timestamp, atol: float
) -> tuple[str, pd.Timestamp | None] | None:
    """Describe the first difference among rows strictly before ``before``: (message, timestamp) or None."""
    tf, tc = _time_axis(full), _time_axis(cut)
    assert tf is not None and tc is not None
    a = full.loc[(tf < before).to_numpy()].copy()
    b = cut.loc[(tc < before).to_numpy()].copy()
    if isinstance(full.index, pd.DatetimeIndex):
        stamps_a, stamps_b = pd.DatetimeIndex(a.index), pd.DatetimeIndex(b.index)
    else:
        stamps_a, stamps_b = pd.DatetimeIndex(tf[tf < before]), pd.DatetimeIndex(tc[tc < before])
    if set(a.columns) != set(b.columns):
        return f"first differing timestamp n/a: columns differ ({sorted(map(str, a.columns))} vs {sorted(map(str, b.columns))})", None
    if len(a) != len(b) or not stamps_a.equals(stamps_b):
        sym = stamps_a.symmetric_difference(stamps_b)
        t = pd.Timestamp(sym[0] if len(sym) else (stamps_a[0] if len(stamps_a) else before))
        return f"first differing timestamp {t.isoformat()}: row sets differ", t
    b = b[list(a.columns)]
    for col in a.columns:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            xf, yf = x.astype(float), y.astype(float)
            bad = ~((np.isnan(xf) & np.isnan(yf)) | (np.abs(xf - yf) <= atol))
        else:
            bad = np.array([not (p == q or (pd.isna(p) and pd.isna(q))) for p, q in zip(x, y)], dtype=bool)
        if bad.any():
            i = int(np.argmax(bad))
            t = pd.Timestamp(stamps_a[i])
            return f"first differing timestamp {t.isoformat()} (column {col!r}: {_py(x[i])!r} vs {_py(y[i])!r})", t
    return None


def _numeric_series(df: pd.DataFrame, col: Any, before: pd.Timestamp) -> pd.Series:
    axis = _time_axis(df)
    assert axis is not None
    keep = (axis < before).to_numpy()
    s = pd.Series(df[col].to_numpy(dtype=float)[keep], index=pd.DatetimeIndex(axis[keep]))
    return s[~s.index.duplicated()]


def _comparison_series(full: pd.DataFrame, cut: pd.DataFrame, before: pd.Timestamp) -> dict[str, Any]:
    """First numeric column present in both frames, on the timestamps they share before ``before``."""
    out: dict[str, Any] = {
        "column": None, "columns_total": int(len(full.columns)), "timestamps": [], "full": [], "truncated": [], "stride": 1,
    }
    if full.empty or cut.empty or _time_axis(full) is None or _time_axis(cut) is None:
        return out
    cols = [
        c for c in full.columns
        if c in cut.columns and pd.api.types.is_numeric_dtype(full[c]) and pd.api.types.is_numeric_dtype(cut[c])
    ]
    if not cols:
        return out
    a, b = _numeric_series(full, cols[0], before), _numeric_series(cut, cols[0], before)
    common = a.index.intersection(b.index).sort_values()
    stride = max(1, -(-len(common) // MAX_SERIES_POINTS))
    common = common[::stride]
    out.update(
        column=str(cols[0]), stride=stride, timestamps=[t.isoformat() for t in common],
        full=[None if np.isnan(v) else float(v) for v in a.reindex(common).to_numpy()],
        truncated=[None if np.isnan(v) else float(v) for v in b.reindex(common).to_numpy()],
    )
    return out


def run_truncation(
    adapter: StrategyAdapter,
    frame: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    publication_column: str | None = None,
    cutoffs: Sequence[float] = DEFAULT_CUTOFFS,
    atol: float = DEFAULT_ATOL,
    full_result: RunResult | None = None,
) -> TruncationResult:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window_bars = frame.index[(frame.index >= start) & (frame.index <= end)]
    if len(window_bars) < MIN_BARS:
        return TruncationResult("BLOCKED", [f"development window has {len(window_bars)} bar(s); need at least {MIN_BARS}"])
    if not cutoffs or any(not 0 < c < 1 for c in cutoffs):
        return TruncationResult("BLOCKED", ["cutoffs must be fractions strictly between 0 and 1"])
    details: dict[str, Any] = {"cutoffs": [], "tolerance": atol, "notes": []}
    try:
        full = full_result or adapter.run(params, DataView(frame, publication_column), start, end)
    except Exception as exc:
        return TruncationResult("BLOCKED", [f"adapter failed on the full data: {type(exc).__name__}: {exc}"], details)
    reasons: list[str] = []
    blocked: list[str] = []
    series: list[dict[str, Any]] = []
    for frac in cutoffs:
        pos = int(round(frac * (len(window_bars) - 1)))
        if pos < 1:
            return TruncationResult("BLOCKED", [f"cutoff {frac:g} leaves no bar before it inside the window"], details)
        cutoff = window_bars[pos]
        last_kept = window_bars[pos - 1]
        entry = {"fraction": frac, "cutoff": cutoff.isoformat(), "status": "PASS"}
        details["cutoffs"].append(entry)
        shown: dict[str, Any] = {
            "fraction": frac, "cutoff": cutoff.isoformat(), "first_divergence": None,
            "column": None, "columns_total": 0, "timestamps": [], "full": [], "truncated": [], "stride": 1,
        }
        series.append(shown)
        try:
            with replay_only():  # an LLM strategy: the prompts of a truncated run must already be in the cache
                cut = adapter.run(params, DataView(frame.loc[frame.index < cutoff], publication_column), start, last_kept)
        except LLMCacheMiss as exc:
            # The prompt for this decision differs from the one built from the full data, so it used data from after
            # the cutoff (the leak is in the prompt, which no position comparison would show).
            when = exc.as_of or "an unspecified decision time (pass as_of to qrh.llm.complete)"
            entry["status"] = "FAIL"
            reasons.append(
                f"prompt differs between full and truncated data at {when} (cutoff {cutoff.isoformat()}, "
                f"fraction {frac:g}): the prompt for that decision used information from after the cutoff")
            stamp = pd.Timestamp(exc.as_of) if exc.as_of and not pd.isna(pd.to_datetime(exc.as_of, errors="coerce")) else None
            old = shown["first_divergence"]
            if stamp is not None and (old is None or stamp.isoformat() < old["timestamp"]):
                shown["first_divergence"] = {"timestamp": stamp.isoformat(), "what": "prompt"}
            continue
        except Exception as exc:
            entry["status"] = "BLOCKED"
            blocked.append(f"adapter failed on data truncated at cutoff {cutoff.isoformat()}: {type(exc).__name__}: {exc}")
            continue
        for what, a, b in (
            ("positions", _as_frame(full.positions), _as_frame(cut.positions)),
            ("trades", _as_frame(full.trades), _as_frame(cut.trades)),
        ):
            if a.empty and b.empty:
                continue
            if _time_axis(a) is None or _time_axis(b) is None:
                note = f"{what} have no timestamps and were not compared"
                if note not in details["notes"]:
                    details["notes"].append(note)
                continue
            if what == "positions":
                shown.update(_comparison_series(a, b, cutoff))
            diff = _first_difference(a, b, cutoff, atol)
            if diff:
                message, when = diff
                entry["status"] = "FAIL"
                reasons.append(f"{what} differ before cutoff {cutoff.isoformat()} (fraction {frac:g}): {message}")
                old = shown["first_divergence"]
                if when is not None and (old is None or when.isoformat() < old["timestamp"]):
                    shown["first_divergence"] = {"timestamp": when.isoformat(), "what": what}
    for shown, entry in zip(series, details["cutoffs"]):
        shown["status"] = entry["status"]
    if blocked:
        return TruncationResult("BLOCKED", blocked + reasons, details, series)
    return TruncationResult("FAIL" if reasons else "PASS", reasons, details, series)

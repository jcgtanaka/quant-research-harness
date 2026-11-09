"""Future-perturbation invariance test: the runtime proof that a run does not read data after a decision time.

For a set of cutoff bars the adapter is rerun on data whose future (every row after the cutoff, and every row not
yet published at the cutoff) has been replaced by different but valid values. Positions, trades and strategy
returns up to and including the cutoff bar must be identical to the unperturbed run, up to a tiny float tolerance.
A strategy that reads anything from after a bar to decide that bar (a negative shift, ``np.roll``, a centered
window, a whole-frame statistic, a read that ignores ``end``) cannot satisfy this, whatever the code looks like.

The cutoffs are chosen from the unperturbed run so that a sparse signal is still tested, even in a strategy that is
always in the market: every bar where the position changes, and the bar before it, is a cutoff (up to
``DEFAULT_MAX_CHANGE_BARS`` change bars, spread evenly when there are more), plus ``DEFAULT_CUTOFFS`` cutoffs spread
evenly over the window. The share of change bars tested is recorded as ``coverage``. Everything is seeded, so the same
inputs always give the same cutoffs, perturbations and verdict.

Cost: one extra adapter run per cutoff (``perturbation_cutoffs`` spread cutoffs, frozen as a threshold, plus at most
twice ``perturbation_change_cap`` change-bar cutoffs), plus one control run only when a difference is found. A PASS is
evidence on the tested cutoffs, not a guarantee.

Two guards against a test that never touches the strategy: when an ``adapter_factory`` is given every perturbed run
uses a freshly loaded adapter (a module-level cache cannot serve the unperturbed result), and if no perturbed run
differs from the unperturbed one anywhere the result is BLOCKED ("perturbed data did not reach the strategy").

The perturbation (``perturb_future``) keeps rows intact so typical data rules still hold: the numeric columns of
the future rows are shuffled together as whole rows, every row is multiplied by one positive random factor, the rows after the first future one also share a random level shift, and the first future row is set so that the
return of the first future bar changes sign. Non-numeric columns, the
index, the publication column and datetime columns are untouched. Boolean, text and categorical columns are permuted
with the same row permutation. Rows stamped before the cutoff but published after it (publication column) count as
future and are perturbed too. A statistic that is legitimately computed from the past only
(an expanding mean, a rolling window ending at the decision bar) is not affected, so it cannot cause a failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

from qrh.adapters.base import DataView, RunResult, StrategyAdapter
from qrh.llm.cache import LLMCacheMiss
from qrh.checks.truncation import DEFAULT_ATOL, MIN_BARS, _as_frame, _first_difference, _time_axis

DEFAULT_CUTOFFS = 20
MAX_CUTOFFS = 200
DEFAULT_MAX_CHANGE_BARS = 40
MAX_CHANGE_BARS = 500
SEED = 20241109  # frozen: part of the method version
SIGMA = 0.2  # spread of the log of the per-row factor
LEVEL_SIGMA = 0.5  # spread of the log of the common factor on the future rows after the first one
MIN_FUTURE_ROWS = 2
MAX_REASONS = 5
_NS = pd.Timedelta(1, "ns")


@dataclass
class PerturbationResult:
    verdict: str  # PASS | FAIL | BLOCKED
    reasons: list[str]
    details: dict[str, Any] = field(default_factory=dict)


# --- the perturbation ------------------------------------------------------------------------------------------


def _future_mask(frame: pd.DataFrame, cutoff: pd.Timestamp, publication_column: str | None) -> np.ndarray:
    """Rows stamped after the cutoff, and rows stamped at or before it but not yet published at it."""
    mask = np.asarray(frame.index > cutoff)
    if publication_column is not None:
        mask = mask | np.asarray(pd.DatetimeIndex(frame[publication_column]) > cutoff)
    return mask


def _numeric_columns(frame: pd.DataFrame, publication_column: str | None) -> list[Any]:
    return [
        c for c in frame.columns
        if c != publication_column and isinstance(frame[c].dtype, np.dtype) and frame[c].dtype.kind in "iuf"
    ]


def _permutable_columns(frame: pd.DataFrame, publication_column: str | None) -> list[Any]:
    """Boolean, text and categorical columns (datetime and other dtypes are left untouched)."""
    out = []
    for c in frame.columns:
        dt = frame[c].dtype
        if c == publication_column:
            continue
        if isinstance(dt, pd.CategoricalDtype) or (isinstance(dt, np.dtype) and dt.kind in "bOU") or str(dt) == "string":
            out.append(c)
    return out


def _reference_column(vals: np.ndarray, cols: list[Any]) -> int | None:
    """The smoothest strictly positive column (the one that behaves most like a price), or None."""
    best: tuple[float, int] | None = None
    for j in range(len(cols)):
        v = vals[:, j]
        if len(v) < 3 or not np.isfinite(v).all() or not (v > 0).all():
            continue
        spread = float(np.std(np.diff(np.log(v))))
        if best is None or spread < best[0]:
            best = (spread, j)
    return None if best is None else best[1]


def perturb_future(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    rng: np.random.Generator,
    publication_column: str | None = None,
    sigma: float = SIGMA,
) -> pd.DataFrame | None:
    """A copy of ``frame`` whose future (after ``cutoff``) is different but valid; None when there is no room.

    Rows up to the cutoff (and already published) are untouched. See the module docstring for what is changed."""
    cutoff = pd.Timestamp(cutoff)
    mask = _future_mask(frame, cutoff, publication_column)
    rows = np.flatnonzero(mask)
    cols = _numeric_columns(frame, publication_column)
    if len(rows) < MIN_FUTURE_ROWS or not cols:
        return None
    vals = frame[cols].to_numpy(dtype=float)
    perm = rng.permutation(len(rows))
    block = vals[rows][perm]
    block = block * np.exp(sigma * rng.standard_normal(len(rows)))[:, None]
    after = np.flatnonzero(np.asarray(frame.index > cutoff))
    first = int(after[0]) if len(after) else None  # the first bar after the cutoff
    k = int(np.searchsorted(rows, first)) if first is not None else None
    shift = math.exp(LEVEL_SIGMA * float(rng.standard_normal()))  # a level shift moves whole-frame statistics
    keep_first = np.arange(len(rows)) != k
    block[keep_first] *= shift
    ref = _reference_column(vals, cols)
    if first is not None and first > 0 and ref is not None:
        prev, cur = vals[first - 1, ref], vals[first, ref]
        if np.isfinite(prev) and np.isfinite(cur) and prev > 0 and cur > 0:
            block[k] = vals[first] * (prev / cur) ** 2  # the first bar's return on the reference column flips sign
    out = frame.copy()
    for j, c in enumerate(cols):
        arr = frame[c].to_numpy(copy=True)
        new = block[:, j]
        arr[rows] = (np.rint(new) if arr.dtype.kind in "iu" else new).astype(arr.dtype)
        out[c] = arr
    for c in _permutable_columns(frame, publication_column):
        if isinstance(frame[c].dtype, pd.CategoricalDtype):
            arr = frame[c].to_numpy(copy=True)
            arr[rows] = arr[rows][perm]
            out[c] = pd.Categorical(arr, categories=frame[c].cat.categories, ordered=frame[c].cat.ordered)
        else:
            arr = frame[c].to_numpy(copy=True)
            arr[rows] = arr[rows][perm]
            out[c] = arr if frame[c].dtype != "string" else pd.array(arr, dtype="string")
    return out


# --- choosing the cutoffs --------------------------------------------------------------------------------------


def _spread(items: list[Any], k: int) -> list[Any]:
    """``k`` items spread evenly over ``items`` (all of them when there are not more than ``k``)."""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    pick = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
    return [items[i] for i in pick]


def _change_bars(full: RunResult, bars: pd.DatetimeIndex, atol: float) -> list[pd.Timestamp]:
    """Bars of the window where the reported position differs from the previous reported row."""
    pos = _as_frame(full.positions)
    axis = _time_axis(pos) if not pos.empty else None
    if axis is None or not all(pd.api.types.is_numeric_dtype(pos[c]) for c in pos.columns):
        return []
    values = np.nan_to_num(pos.to_numpy(dtype=float))
    stamps = pd.DatetimeIndex(axis)
    changed = np.zeros(len(values), dtype=bool)
    changed[1:] = (np.abs(np.diff(values, axis=0)) > atol).any(axis=1)
    keep = set(stamps[changed])
    return [b for b in bars if b in keep]


def _choose_cutoffs(
    bars: pd.DatetimeIndex, eligible: list[pd.Timestamp], changes: list[pd.Timestamp], n: int, cap: int,
) -> tuple[list[tuple[pd.Timestamp, str]], int, int]:
    """Cutoffs and (change bars tested, change bars in the window): the change bars up to ``cap`` (spread evenly
    beyond it) with the bar before each, then ``n`` bars spread evenly over the window."""
    ok = set(eligible)
    eligible_changes = [b for b in changes if b in ok]
    tested = _spread(eligible_changes, cap)
    chosen: dict[pd.Timestamp, str] = {}
    for b in tested:
        chosen[b] = "change"
    pos = {b: i for i, b in enumerate(bars)}
    for b in tested:
        i = pos[b]
        if i > 0 and bars[i - 1] in ok and bars[i - 1] not in chosen:
            chosen[bars[i - 1]] = "before-change"
    rest = [b for b in eligible if b not in chosen]
    for b in _spread(rest, n):
        chosen[b] = "spread"
    return sorted(chosen.items()), len(tested), len(eligible_changes)


# --- the test --------------------------------------------------------------------------------------------------


def _compare(
    full: RunResult, other: RunResult, upto: pd.Timestamp, atol: float, notes: list[str], skip: tuple[str, ...] = (),
) -> tuple[str, str, pd.Timestamp | None] | None:
    """The first difference among rows at or before ``upto``: (what, message, timestamp), else None.
    ``skip`` names outputs left out (the returns, when a publication column makes realized prices move too)."""
    before = upto + _NS
    for what, a, b in (
        ("positions", _as_frame(full.positions), _as_frame(other.positions)),
        ("returns", _as_frame(full.returns), _as_frame(other.returns)),
        ("trades", _as_frame(full.trades), _as_frame(other.trades)),
    ):
        if (a.empty and b.empty) or what in skip:
            continue
        if _time_axis(a) is None or _time_axis(b) is None:
            note = f"{what} have no timestamps and were not compared"
            if note not in notes:
                notes.append(note)
            continue
        diff = _first_difference(a, b, before, atol)
        if diff:
            return what, diff[0], diff[1]
    return None


def run_perturbation(
    adapter: StrategyAdapter,
    frame: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    publication_column: str | None = None,
    n_cutoffs: int = DEFAULT_CUTOFFS,
    atol: float = DEFAULT_ATOL,
    full_result: RunResult | None = None,
    max_change_bars: int = DEFAULT_MAX_CHANGE_BARS,
    adapter_factory: Callable[[], StrategyAdapter] | None = None,
    context: Callable[[pd.Timestamp], AbstractContextManager[Any]] | None = None,
) -> PerturbationResult:
    """``adapter_factory`` gives a freshly loaded adapter for each perturbed run; ``context(cutoff)`` wraps each
    perturbed run (the S2 stage uses it for the language-model replay stub)."""
    if isinstance(n_cutoffs, bool) or not isinstance(n_cutoffs, int) or not 1 <= n_cutoffs <= MAX_CUTOFFS:
        return PerturbationResult(
            "BLOCKED", [f"perturbation_cutoffs must be an integer from 1 to {MAX_CUTOFFS}, got {n_cutoffs!r}"])
    if isinstance(max_change_bars, bool) or not isinstance(max_change_bars, int) or not 1 <= max_change_bars <= MAX_CHANGE_BARS:
        return PerturbationResult(
            "BLOCKED", [f"perturbation_change_cap must be an integer from 1 to {MAX_CHANGE_BARS}, got {max_change_bars!r}"])
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    bars = frame.index[(frame.index >= start) & (frame.index <= end)]
    if len(bars) < MIN_BARS:
        return PerturbationResult("BLOCKED", [f"development window has {len(bars)} bar(s); need at least {MIN_BARS}"])
    details: dict[str, Any] = {
        "requested_cutoffs": n_cutoffs, "n_cutoffs": 0, "cutoffs": [], "tolerance": atol, "seed": SEED, "sigma": SIGMA,
        "first_divergence": None, "adapter_runs": 0, "notes": [], "change_bar_cap": max_change_bars,
        "change_bars_total": 0, "change_bars_tested": 0, "coverage": 1.0,
    }
    try:
        full = full_result or adapter.run(params, DataView(frame, publication_column), start, end)
    except Exception as exc:
        return PerturbationResult("BLOCKED", [f"adapter failed on the full data: {type(exc).__name__}: {exc}"], details)
    rows_after = len(frame) - frame.index.searchsorted(bars, side="right")
    eligible = [b for b, after in zip(bars, rows_after) if after >= MIN_FUTURE_ROWS and b != bars[0]]
    if not eligible:
        return PerturbationResult("BLOCKED", ["no bar in the window has enough later rows to perturb"], details)
    chosen, tested, total = _choose_cutoffs(bars, eligible, _change_bars(full, bars, atol), n_cutoffs, max_change_bars)
    details.update(n_cutoffs=len(chosen), change_bars_tested=tested, change_bars_total=total,
                   coverage=1.0 if total == 0 else tested / total)
    # With a publication column the perturbation also moves rows stamped before the cutoff (their realized prices too),
    # so the reported returns legitimately change; the decisions (positions, trades) are what must not.
    skip = ("returns",) if publication_column is not None else ()
    if skip:
        details["notes"].append("returns were not compared: a publication column makes the perturbation move realized prices")
    failures: list[str] = []
    blocked: list[str] = []
    reached = False
    first: tuple[pd.Timestamp, str] | None = None
    for cutoff, kind in chosen:
        entry: dict[str, Any] = {"cutoff": cutoff.isoformat(), "kind": kind, "status": "PASS"}
        details["cutoffs"].append(entry)
        pert = perturb_future(frame, cutoff, np.random.default_rng([SEED, int(bars.get_loc(cutoff))]), publication_column)
        if pert is None:
            entry["status"] = "BLOCKED"
            blocked.append(f"cutoff {cutoff.isoformat()}: the data has no room to perturb")
            continue
        try:
            details["adapter_runs"] += 1
            runner = adapter_factory() if adapter_factory is not None else adapter
            with (context(cutoff) if context is not None else nullcontext()):
                other = runner.run(params, DataView(pert, publication_column), start, end)
        except LLMCacheMiss as exc:
            when = exc.as_of or "an unspecified decision time"
            entry["status"] = "FAIL"
            failures.append(
                f"prompt differs between the original and the perturbed data at {when} (cutoff {cutoff.isoformat()}): "
                "the prompt for a decision at or before the cutoff used information from after it")
            continue
        except Exception as exc:
            entry["status"] = "BLOCKED"
            blocked.append(
                f"adapter failed on data with the future perturbed after cutoff {cutoff.isoformat()}: {type(exc).__name__}: {exc}")
            continue
        if not reached and _compare(full, other, end, atol, []) is not None:
            reached = True
        diff = _compare(full, other, cutoff, atol, details["notes"], skip)
        if diff:
            what, message, when = diff
            entry["status"], entry["first_divergence"] = "FAIL", None if when is None else when.isoformat()
            failures.append(
                f"{what} before or at cutoff {cutoff.isoformat()} changed when only the data after it was changed: {message}")
            if when is not None and (first is None or when < first[0]):
                first = (when, what)
    if first is not None:
        details["first_divergence"] = {"timestamp": first[0].isoformat(), "what": first[1]}
    if failures:
        reproducible = _reproducible(adapter, frame, params, start, end, publication_column, full, bars[-1], atol, details)
        if reproducible is not None:
            return PerturbationResult("BLOCKED", [reproducible], details)
        shown = failures[:MAX_REASONS]
        if len(failures) > MAX_REASONS:
            shown.append(f"... and {len(failures) - MAX_REASONS} more cutoffs differ")
        return PerturbationResult("FAIL", shown + blocked, details)
    if blocked:
        return PerturbationResult("BLOCKED", blocked, details)
    if not reached:
        return PerturbationResult("BLOCKED", [
            "perturbed data did not reach the strategy: no perturbed run differed from the unperturbed run anywhere "
            "(a cached result or a strategy that ignores the data?), so nothing was tested"], details)
    return PerturbationResult("PASS", [], details)


def _reproducible(adapter, frame, params, start, end, publication_column, full, last_bar, atol, details) -> str | None:
    """None when the adapter gives the same output twice on identical data, else the reason it cannot be judged."""
    try:
        details["adapter_runs"] += 1
        again = adapter.run(params, DataView(frame, publication_column), start, end)
    except Exception as exc:
        return f"adapter failed when rerun on the unchanged data: {type(exc).__name__}: {exc}"
    diff = _compare(full, again, last_bar, atol, details["notes"])
    if diff is None:
        return None
    return (f"adapter output is not reproducible: two runs on identical data differ ({diff[1]}), so a difference after "
            "perturbing the future cannot be told apart from randomness; seed every random source")


# --- does a reported position depend on its own bar? ------------------------------------------------------------

PROBE_BARS = 8


def make_own_bar_probe(
    adapter: StrategyAdapter, frame: pd.DataFrame, params: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp,
    publication_column: str | None, full: RunResult, adapter_factory: Callable[[], StrategyAdapter] | None = None,
    context: Callable[[pd.Timestamp], AbstractContextManager[Any]] | None = None,
) -> Callable[[], bool | None]:
    """A callable answering: does the reported position at a bar change when the data of that bar (and later) is
    perturbed? True: positions are decided from their own bar (decision at the close). False: they never are, so they
    look like the position held during the bar. None: the probe could not tell (no change bars, or a run failed)."""

    def probe() -> bool | None:
        bars = frame.index[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        changes = [b for b in _change_bars(full, bars, DEFAULT_ATOL) if bars.get_loc(b) >= 1]
        picked = _spread(changes, PROBE_BARS)
        if not picked:
            return None
        pos_full = _as_frame(full.positions)
        tested = 0
        for b in picked:
            i = int(bars.get_loc(b))
            pert = perturb_future(frame, bars[i - 1], np.random.default_rng([SEED, i, 1]), publication_column)
            if pert is None:
                continue
            try:
                runner = adapter_factory() if adapter_factory is not None else adapter
                with (context(bars[i - 1]) if context is not None else nullcontext()):
                    other = runner.run(params, DataView(pert, publication_column), start, end)
            except Exception:
                return None
            tested += 1
            a, c = pos_full.loc[[b]], _as_frame(other.positions)
            if b not in c.index or not np.allclose(
                np.nan_to_num(a.to_numpy(dtype=float)), np.nan_to_num(c.loc[[b], list(a.columns)].to_numpy(dtype=float)),
                atol=DEFAULT_ATOL, rtol=0,
            ):
                return True
        return False if tested else None

    return probe

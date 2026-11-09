"""Mark-freshness rule for the pnl.linear method (pure functions).

Staleness is measured in time: for each bar, bar_time minus the publication
time of the price used for that bar. Repeated values alone never count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StalenessResult:
    """Bar positions (0-based) flagged by the freshness rule."""

    stale: list[int] = field(default_factory=list)
    halt_declared: list[int] = field(default_factory=list)
    future_dated: list[int] = field(default_factory=list)

    @property
    def all_stale(self) -> list[int]:
        """Stale bars including halt-declared ones; a halt never clears one."""
        return sorted(set(self.stale) | set(self.halt_declared))


def held_mask(position: pd.Series, lag: int) -> pd.Series:
    """Boolean mask of bars whose price backs a held return.

    A return at bar t has its price base at t-1 and is earned by the position
    at t-L. It is held if position[t-L] != 0 or position[t-1] != 0 (NaN is not
    held). Both prices of a held return are needed, so bar t (exit side) and
    bar t-1 (base side) are marked. This covers the base bar at entry and the
    exit bar after the position closes.
    """
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError(f"lag must be an integer >= 0, got {lag!r}")
    nz = (pd.Series(position).fillna(0) != 0).to_numpy()
    n = len(nz)
    ret = np.zeros(n, dtype=bool)
    if lag < n:
        ret[lag:] |= nz[: n - lag]
    if n > 1:
        ret[1:] |= nz[:-1]
    mask = ret.copy()
    mask[:-1] |= ret[1:]
    return pd.Series(mask, index=pd.Series(position).index)


def mark_staleness(bar_times, publication_times) -> pd.Series:
    """Per-bar staleness as a Timedelta: bar_time - publication_time.

    Negative values mean a future-dated mark. Missing publication times give NaT.
    """
    bt = pd.Series(pd.to_datetime(pd.Series(bar_times)).to_numpy())
    pt = pd.Series(pd.to_datetime(pd.Series(publication_times)).to_numpy())
    if len(bt) != len(pt):
        raise ValueError("bar_times and publication_times must have equal length")
    return bt - pt


def check_mark_freshness(
    bar_times,
    publication_times,
    position,
    lag: int,
    max_staleness: float,
    halt_flag=None,
    bar_period: pd.Timedelta | None = None,
) -> StalenessResult:
    """Flag stale and future-dated marks on held bars.

    max_staleness is a positive number of bars; it is converted to time using
    bar_period (default: median spacing of bar_times). A held bar is stale when
    its staleness exceeds max_staleness * bar_period. A True halt flag moves a
    stale bar into halt_declared (still reported, never passed). Raises
    ValueError if a held bar has no publication time.
    """
    if not max_staleness > 0:
        raise ValueError("max_staleness must be a positive number of bars")
    bt = pd.Series(pd.to_datetime(pd.Series(bar_times)).to_numpy())
    n = len(bt)
    if len(pd.Series(publication_times)) != n or len(pd.Series(position)) != n:
        raise ValueError("bar_times, publication_times and position must align")
    if halt_flag is not None and len(pd.Series(halt_flag)) != n:
        raise ValueError("halt_flag must align with bar_times")

    held = held_mask(pd.Series(position).reset_index(drop=True), lag).to_numpy()
    stale_td = mark_staleness(bt, publication_times)

    missing = np.flatnonzero(held & stale_td.isna().to_numpy())
    if missing.size:
        raise ValueError(
            "missing publication time for held bar(s) at positions "
            f"{missing.tolist()}; cannot assess mark freshness"
        )

    if bar_period is None:
        diffs = bt.diff().dropna()
        if diffs.empty:
            bar_period = pd.Timedelta(0)
        else:
            bar_period = diffs.median()
    limit = bar_period * max_staleness

    halt = (
        pd.Series(halt_flag).fillna(False).astype(bool).to_numpy()
        if halt_flag is not None
        else np.zeros(n, dtype=bool)
    )
    future = held & (stale_td < pd.Timedelta(0)).to_numpy()
    is_stale = held & (stale_td > limit).to_numpy()
    return StalenessResult(
        stale=np.flatnonzero(is_stale & ~halt).tolist(),
        halt_declared=np.flatnonzero(is_stale & halt).tolist(),
        future_dated=np.flatnonzero(future).tolist(),
    )

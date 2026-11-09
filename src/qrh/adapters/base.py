"""Adapter interfaces: small, engine-agnostic contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class DatasetRef:
    """Reference to a dataset by name, with an optional version or content hash."""

    name: str
    version: str | None = None


def _require_aware(ts: pd.Timestamp, what: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"{what} must be timezone-aware")
    return ts


class DataView:
    """Point-in-time view over a time-indexed frame.

    The index must be a timezone-aware DatetimeIndex. If a publication column
    is given, a row is available once its publication time has passed;
    otherwise a row is available at its index time.
    """

    def __init__(self, frame: pd.DataFrame, publication_column: str | None = None):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("frame must have a DatetimeIndex")
        if frame.index.tz is None:
            raise ValueError("frame index must be timezone-aware")
        if publication_column is not None:
            if publication_column not in frame.columns:
                raise ValueError(f"publication column {publication_column!r} not found")
            col = pd.DatetimeIndex(frame[publication_column])
            if col.tz is None:
                raise ValueError("publication column must be timezone-aware")
        self._frame = frame.copy(deep=True)
        self._pub = publication_column

    @property
    def publication_column(self) -> str | None:
        return self._pub

    def as_of(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        """Return a copy of the rows available at ``decision_time``."""
        t = _require_aware(decision_time, "decision_time")
        if self._pub is not None:
            avail = pd.DatetimeIndex(self._frame[self._pub])
            mask = avail <= t
        else:
            mask = self._frame.index <= t
        return self._frame.loc[mask].copy(deep=True)


@dataclass
class RunResult:
    """Output of a strategy run."""

    returns: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame


@runtime_checkable
class StrategyAdapter(Protocol):
    def parameter_space(self) -> dict[str, list[Any]]: ...

    def required_data(self) -> list[DatasetRef]: ...

    def run(
        self,
        params: dict[str, Any],
        data: DataView,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> RunResult: ...

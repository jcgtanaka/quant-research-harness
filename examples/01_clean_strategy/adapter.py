"""Moving-average trend on a synthetic close series (no look-ahead).

The position for bar t is decided from data up to and including bar t and is
applied to the return of bar t+1. A simple proportional cost is charged on
turnover; ``cost_multiplier`` scales it so the harness can stress costs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, DatasetRef, RunResult

COST_BPS = 5.0  # cost per unit of turnover, in basis points of notional


class TrendAdapter:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {"fast": [3, 10, 30], "slow": [60, 120, 240]}

    def required_data(self) -> list[DatasetRef]:
        return [DatasetRef("prices")]

    def run(
        self,
        params: dict[str, Any],
        data: DataView,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cost_multiplier: float = 1.0,
    ) -> RunResult:
        px = data.as_of(end)["close"]
        ret = px.pct_change()
        trend = px.rolling(params["fast"]).mean() - px.rolling(params["slow"]).mean()
        position = np.sign(trend).fillna(0.0)  # known at the close of bar t
        change = position.diff().fillna(position)
        cost = change.abs() * COST_BPS * cost_multiplier / 1e4  # paid when the position changes
        strategy = (position.shift(1) * ret - cost.shift(1)).fillna(0.0)  # earned over bar t+1
        trades = pd.DataFrame({"qty": change[change != 0]})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])

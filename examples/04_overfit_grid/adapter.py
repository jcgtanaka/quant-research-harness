"""A large moving-average grid searched on pure noise (no look-ahead, no real edge).

Position for bar t comes from data up to bar t and earns the return of bar t+1.
A trade needs the fast/slow spread (as a fraction of price) to exceed ``band``.
There is nothing to find in the data, so whatever the grid search reports is
selection luck; the harness should reject it (S4 or S5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, DatasetRef, RunResult

COST_BPS = 5.0  # cost per unit of turnover, in basis points of notional


class NoiseGridAdapter:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            "fast": [3, 5, 8, 12, 20, 30],
            "slow": [40, 60, 90, 130, 200],
            "band": [0.0, 0.005, 0.01, 0.02],
        }

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
        spread = (px.rolling(params["fast"]).mean() - px.rolling(params["slow"]).mean()) / px
        position = np.sign(spread).where(spread.abs() > params["band"], 0.0).fillna(0.0)
        change = position.diff().fillna(position)
        cost = change.abs() * COST_BPS * cost_multiplier / 1e4
        strategy = (position.shift(1) * ret - cost.shift(1)).fillna(0.0)
        trades = pd.DataFrame({"qty": change[change != 0]})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])

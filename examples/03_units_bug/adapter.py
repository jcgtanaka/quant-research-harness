"""Same strategy as example 01, but returns are reported in percent (units bug)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, DatasetRef, RunResult


class PercentAdapter:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {"fast": [5, 10], "slow": [30, 60]}

    def required_data(self) -> list[DatasetRef]:
        return [DatasetRef("prices")]

    def run(self, params: dict[str, Any], data: DataView, start: pd.Timestamp, end: pd.Timestamp) -> RunResult:
        px = data.as_of(end)["close"]
        ret = px.pct_change() * 100  # percent instead of a fraction
        trend = px.rolling(params["fast"]).mean() - px.rolling(params["slow"]).mean()
        position = np.sign(trend).fillna(0.0)
        strategy = (position.shift(1) * ret).fillna(0.0)
        change = position.diff().fillna(position)
        trades = pd.DataFrame({"qty": change[change != 0]})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])

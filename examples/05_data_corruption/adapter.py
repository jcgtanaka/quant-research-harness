"""Minimal strategy over the ``derived`` column (the point of this example is the data check)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, DatasetRef, RunResult


class DerivedTrendAdapter:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {"lookback": [10, 20]}

    def required_data(self) -> list[DatasetRef]:
        return [DatasetRef("linked")]

    def run(self, params: dict[str, Any], data: DataView, start: pd.Timestamp, end: pd.Timestamp) -> RunResult:
        px = data.as_of(end)["derived"]
        ret = px.pct_change()
        position = np.sign(px - px.rolling(params["lookback"]).mean()).fillna(0.0)
        strategy = (position.shift(1) * ret).fillna(0.0)
        change = position.diff().fillna(position)
        trades = pd.DataFrame({"qty": change[change != 0]})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])

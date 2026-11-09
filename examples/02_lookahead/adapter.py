"""Same timing as example 01, but the signal peeks at the next bar's return (look-ahead bug)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, DatasetRef, RunResult


class PeekingAdapter:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {"threshold": [0.0, 0.001]}

    def required_data(self) -> list[DatasetRef]:
        return [DatasetRef("prices")]

    def run(self, params: dict[str, Any], data: DataView, start: pd.Timestamp, end: pd.Timestamp) -> RunResult:
        px = data.as_of(end)["close"]
        ret = px.pct_change()
        next_ret = ret.shift(-1)
        position = np.sign(next_ret).where(next_ret.abs() > params["threshold"], 0.0).fillna(0.0)
        strategy = (position.shift(1) * ret).fillna(0.0)
        change = position.diff().fillna(position)
        trades = pd.DataFrame({"qty": change[change != 0]})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])

"""A language-model trend strategy on synthetic prices, driven by the deterministic stub (no network).

At every bar the adapter builds a prompt from the closes up to and including that bar, asks the model through
``qrh.llm.complete`` and trades its one-word answer: the position decided at bar t earns the return of bar t+1.
The stub follows a trend rule on the closes in the prompt, so the strategy has a real, explainable edge on data with
persistent trends. Nothing here reads a bar after the decision time.

A private adapter would register a real provider backend here instead of the stub; qrh ships no network backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, DatasetRef, RunResult
from qrh.llm import complete, read_prompt, register_backend
from qrh.llm.stub import StubBackend

register_backend("stub", StubBackend())

COST_BPS = 5.0  # cost per unit of turnover, in basis points of notional
ANSWERS = {"UP": 1.0, "DOWN": -1.0}


class LLMTrendAdapter:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {"lookback": [20, 60, 120], "hold": [1, 5, 10]}

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
        variant = params["llm_variant"]  # set by the harness: model, prompt file(s), their hash, params
        template = read_prompt(variant, __file__)
        model = variant["models"][0]
        px = data.as_of(end)["close"]
        ret = px.pct_change()
        lookback, hold = params["lookback"], params["hold"]
        held = np.zeros(len(px))
        current = 0.0
        for i in range(lookback, len(px)):
            if i % hold == 0:  # a fresh decision every `hold` bars; otherwise keep the position
                closes = ", ".join(f"{c:.4f}" for c in px.iloc[i - lookback + 1 : i + 1])  # up to bar i only
                prompt = template.format(date=px.index[i].date().isoformat(), closes=closes)
                answer = complete(model, prompt, params=variant["params"], as_of=px.index[i])
                current = ANSWERS.get(answer.strip().upper(), 0.0)
            held[i] = current  # known at the close of bar i
        position = pd.Series(held, index=px.index)
        change = position.diff().fillna(position)
        cost = change.abs() * COST_BPS * cost_multiplier / 1e4  # paid when the position changes
        strategy = (position.shift(1) * ret - cost.shift(1)).fillna(0.0)  # earned over bar i+1
        trades = pd.DataFrame({"qty": change[change != 0]})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])

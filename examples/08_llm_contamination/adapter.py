"""A language-model strategy whose model "knows the future" before its training cutoff (synthetic, no network).

The stub backend below models a language model trained on data up to 2021-06-30: for a decision date earlier than
that it returns the TRUE next-bar direction (it remembers what happened), and afterwards it follows a trend rule.
The prompt contains only past closes, so no truncation test can see the leak: it lives in the "weights".
The hypothesis declares the cutoff honestly, and the S2 cutoff gate rejects the run because the development window
starts before the cutoff plus the safety margin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, DatasetRef, RunResult
from qrh.llm import complete, read_prompt, register_backend
from qrh.llm.stub import StubBackend
from qrh.synthetic.generator import Regime, SyntheticConfig, generate

COST_BPS = 5.0
ANSWERS = {"UP": 1.0, "DOWN": -1.0}
MODEL_CUTOFF = "2021-06-30"  # what the stub's "training data" covers


def _true_next_bar() -> dict[str, str]:
    """What actually happened next, by decision date: the same series make_data.py writes (same seed and config)."""
    cfg = SyntheticConfig(
        seed=11, start="2020-01-01", periods=1500, freq="D", tz="UTC",
        regimes=[Regime(share=1.0, drift=0.0, volatility=0.01)], trend_persistence=0.97, trend_scale=0.004,
    )
    close = generate(cfg)["close"]
    nxt = close.shift(-1) / close - 1.0  # qrh: allow LA001 the stub models training data: it deliberately knows the future
    return {ts.date().isoformat(): ("UP" if r > 0 else "DOWN") for ts, r in nxt.dropna().items()}


register_backend("stub", StubBackend(oracle=_true_next_bar(), oracle_until=MODEL_CUTOFF))


class LLMOracleAdapter:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {"lookback": [20, 60, 120]}

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
        variant = params["llm_variant"]
        template = read_prompt(variant, __file__)
        model = variant["models"][0]
        px = data.as_of(end)["close"]
        ret = px.pct_change()
        lookback = params["lookback"]
        held = np.zeros(len(px))
        for i in range(lookback, len(px)):
            closes = ", ".join(f"{c:.4f}" for c in px.iloc[i - lookback + 1 : i + 1])  # past closes only
            prompt = template.format(date=px.index[i].date().isoformat(), closes=closes)
            answer = complete(model, prompt, params=variant["params"], as_of=px.index[i])
            held[i] = ANSWERS.get(answer.strip().upper(), 0.0)
        position = pd.Series(held, index=px.index)
        change = position.diff().fillna(position)
        cost = change.abs() * COST_BPS * cost_multiplier / 1e4
        strategy = (position.shift(1) * ret - cost.shift(1)).fillna(0.0)
        trades = pd.DataFrame({"qty": change[change != 0]})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])

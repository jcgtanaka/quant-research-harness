"""The planted-bug corpus: strategy sources and datasets with a known defect, plus clean controls.

Every case is a recipe, materialized per seed into a synthetic frame (and, for code cases, a strategy
source that is both linted and executed, so the static lint and the runtime tests see the same bytes).
Nothing here decides a verdict; ``suites.py`` runs the harness's detectors over the materialized cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qrh.bench.engine import WARMUP, make_frame

CASE_BARS = 400
CONTROLS = "controls"
BUG_CLASSES = (
    "lookahead_shift", "centered_rolling", "backward_fill", "full_sample_normalization",
    "same_bar_execution", "percent_units", "relationship_corruption", "gaps_duplicates",
)
BOUNDS = {"max_abs_period_return": 0.5, "max_gross_leverage": 1.0, "max_position_weight": 1.0}
RULES: list[dict[str, Any]] = [
    {"type": "schema", "columns": {"close": "f", "ref_close": "f"}},
    {"type": "range", "column": "close", "positive": True},
    {"type": "relationship", "column": "close", "reference": "ref_close", "rel_tol": 0.01},
    {"type": "gaps", "freq": "D"},
    {"type": "timezone", "zone": "UTC"},
    {"type": "null_rate_drift", "column": "close", "window_fraction": 0.2, "max_increase": 0.05},
]
EXECUTION = {"lag_bars": 1, "price_column": "close"}  # declared for s2.pnl_reconstruction
RELATIONSHIP_TOLERANCE = next(r["rel_tol"] for r in RULES if r["type"] == "relationship")
CLEAN_EXECUTION = "        strategy = (position.shift(1) * ret).fillna(0.0)"
TEMPLATE = '''import numpy as np
import pandas as pd

from qrh.adapters.base import RunResult


class Planted:
    def parameter_space(self):
        return {{"w": [10]}}

    def required_data(self):
        return []

    def run(self, params, data, start, end, cost_multiplier=1.0):
        px = data.as_of(end)["close"]
        ret = px.pct_change()
{signal}
        position = np.sign(signal).fillna(0.0)
{execution}
        trades = pd.DataFrame({{"qty": position.diff().fillna(0.0)}})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])
'''


@dataclass(frozen=True)
class Case:
    bug_class: str
    name: str
    kind: str  # code | data | control
    params: dict[str, Any] = field(default_factory=dict)
    source_template: str | None = None  # the strategy source (None for data-only cases)
    volatility: float = 0.01

    @property
    def expected_undetectable(self) -> bool:
        """A corruption at or below the relationship rule's tolerance cannot be flagged by that rule, by design."""
        return self.bug_class == "relationship_corruption" and self.params["magnitude"] <= RELATIONSHIP_TOLERANCE


@dataclass
class Materialized:
    case: Case
    frame: pd.DataFrame
    source: str | None

    @property
    def kind(self) -> str:
        return self.case.kind

    def adapter(self) -> Any:
        assert self.source is not None
        ns: dict[str, Any] = {}
        exec(compile(self.source, f"<planted:{self.case.bug_class}:{self.case.name}>", "exec"), ns)  # noqa: S102
        return ns["Planted"]()


def _src(signal: str, execution: str = CLEAN_EXECUTION) -> str:
    body = "\n".join("        " + line for line in signal.splitlines())
    return TEMPLATE.format(signal=body, execution=execution)


def _code(bug_class: str, name: str, signal: str, execution: str = CLEAN_EXECUTION, volatility: float = 0.01) -> Case:
    return Case(bug_class, name, "code", {}, _src(signal, execution), volatility)


def _control(name: str, signal: str) -> Case:
    return Case(CONTROLS, name, "control", {}, _src(signal))


def corpus() -> list[Case]:
    cases: list[Case] = []
    for k in (1, 2, 5):
        cases.append(_code("lookahead_shift", f"shift_minus_{k}", f"signal = ret.shift(-{k})"))
    cases.append(_code("lookahead_shift", "shift_minus_variable", "lag = 2\nsignal = ret.shift(-lag)"))
    cases.append(_code("lookahead_shift", "shift_minus_variable_k5", "lag = 5\nsignal = ret.rolling(3).mean().shift(-lag)"))
    sparse = "rare = ret.abs() > 1.5 * ret.rolling(50).std()  # a rarely active signal\n"
    cases.append(_code("lookahead_shift", "np_roll_sparse",
                       sparse + "signal = pd.Series(np.roll(ret.to_numpy(), -1), index=px.index).where(rare)"))
    cases.append(_code("lookahead_shift", "named_negative_shift_sparse", sparse + "NEG = -2\nsignal = ret.shift(NEG).where(rare)"))
    for w in (5, 21):
        cases.append(_code("centered_rolling", f"center_window_{w}", f"signal = px.rolling({w}, center=True).mean() - px"))
    cases.append(_code("centered_rolling", "center_via_variable", "mid = True\nsignal = px.rolling(11, center=mid).mean() - px"))
    mask = "masked = ret.mask(ret.abs() > 0.01)\n"
    cases.append(_code("backward_fill", "bfill", mask + "signal = masked.bfill()"))
    cases.append(_code("backward_fill", "bfill_limit", mask + "signal = masked.bfill(limit=3)"))
    cases.append(_code("backward_fill", "bfill_via_getattr", mask + 'signal = getattr(masked, "bfill")()'))
    cases.append(_code("full_sample_normalization", "price_zscore", "signal = (px - px.mean()) / px.std()"))
    cases.append(_code("full_sample_normalization", "demeaned_return", "signal = ret - ret.mean()"))
    same_bar = "        strategy = (position * ret).fillna(0.0)"
    smoothing = {"raw": "signal = ret", "roll3": "signal = ret.rolling(3).mean()", "roll10": "signal = ret.rolling(10).mean()"}
    for vol in (0.003, 0.01, 0.02):
        for label, signal in smoothing.items():
            cases.append(_code("same_bar_execution", f"same_bar_vol_{vol:g}_{label}", signal, same_bar, volatility=vol))
    percent = "        strategy = (position.shift(1) * ret * 100.0).fillna(0.0)"
    signals = {
        "macross": "signal = px.rolling(10).mean() - px.rolling(40).mean()",
        "momentum5": "signal = ret.rolling(5).mean()",
    }
    for vol in (0.001, 0.003, 0.01):
        for label, signal in signals.items():
            cases.append(_code("percent_units", f"percent_vol_{vol:g}_{label}", signal, percent, volatility=vol))
    for mag in (0.005, 0.02, 0.10):
        for length in (1, 5, 40):
            cases.append(Case("relationship_corruption", f"mag_{mag:g}_len_{length}", "data",
                              {"what": "relationship", "magnitude": mag, "length": length}))
    for n in (1, 3, 10):
        cases.append(Case("gaps_duplicates", f"gap_{n}", "data", {"what": "gap", "count": n}))
    for n in (1, 5):
        cases.append(Case("gaps_duplicates", f"duplicates_{n}", "data", {"what": "duplicate", "count": n}))
    clean = {
        "ma_cross": "signal = px.rolling(10).mean() - px.rolling(40).mean()",
        "return_momentum": "signal = ret.rolling(10).mean()",
        "mean_reversion": "signal = -(px / px.rolling(20).mean() - 1)",
        "lagged_return": "signal = ret.shift(1)",
        "ewm_momentum": "signal = ret.ewm(span=20).mean()",
        "expanding_zscore": "signal = (ret - ret.expanding().mean()) / ret.expanding().std()",
        "forward_fill": "signal = ret.where(ret.abs() < 0.03).ffill()",
    }
    cases += [_control(n, s) for n, s in clean.items()]
    return cases


def _edit(frame: pd.DataFrame, case: Case, rng: np.random.Generator) -> pd.DataFrame:
    what, n = case.params["what"], len(frame)
    if what == "relationship":
        length = case.params["length"]
        pos = int(rng.integers(WARMUP + 10, n - length - 10))
        out = frame.copy()
        out.iloc[pos : pos + length, out.columns.get_loc("close")] *= 1.0 + case.params["magnitude"]
        return out
    if what == "gap":
        count = case.params["count"]
        pos = int(rng.integers(20, n - count - 20))
        return frame.drop(frame.index[pos : pos + count])
    if what == "duplicate":
        rows = np.sort(rng.choice(np.arange(20, n - 20), size=case.params["count"], replace=False))
        return pd.concat([frame, frame.iloc[rows]]).sort_index(kind="stable")
    raise ValueError(f"unknown data edit {what!r}")


def materialize(case: Case, seed: int) -> Materialized:
    from qrh.bench.engine import seed_for

    frame = make_frame(CASE_BARS, seed, 0.0, case.volatility)
    frame["ref_close"] = frame["close"]
    if case.kind == "data":
        frame = _edit(frame, case, np.random.default_rng(seed_for(seed, "edit", case.bug_class, case.name)))
    return Materialized(case, frame, case.source_template)

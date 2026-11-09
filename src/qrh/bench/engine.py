"""The S3-S5 computation on synthetic strategy families, with the harness's own method versions.

A strategy family is a moving-average crossover grid run on seeded synthetic prices. ``build_matrix`` is the S3
step (every combination over the development window); ``evaluate_matrix`` is S4 (walk-forward selection and
stitched out-of-sample returns) and S5 (PBO and DSR) on that matrix, using the versions a ``MethodSelector``
hands out. Walk-forward test spans replay the stored S3 returns, which for a strategy without look-ahead is
exactly what running the adapter on the span returns (see ``tests/unit/test_bench_engine.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DatasetRef, DataView, RunResult
from qrh.hashing import hash_json
from qrh.methods.registry import MethodError
from qrh.methods.selection import MethodSelector
from qrh.synthetic.generator import Regime, SyntheticConfig, generate
from qrh.validation.metrics import annualized_sharpe
from qrh.validation.overfit import overfitting_statistics

WARMUP = 60  # bars before the development window starts (indicator warm-up)
COST_BPS = 5.0
GRID_SIZES: dict[int, dict[str, list[Any]]] = {
    12: {"fast": [3, 8, 20], "slow": [40, 90], "band": [0.0, 0.01]},
    36: {"fast": [3, 8, 20], "slow": [40, 90, 130], "band": [0.0, 0.005, 0.01, 0.02]},
    120: {"fast": [3, 5, 8, 12, 20, 30], "slow": [40, 60, 90, 130, 200], "band": [0.0, 0.005, 0.01, 0.02]},
}
TREND_PERSISTENCE = 0.97


def seed_for(*parts: Any) -> int:
    """A stable 32-bit seed from any JSON-able parts (base seed, suite, cell, replication...)."""
    return int(hash_json(list(parts))[:8], 16)


def grid_space(n_combos: int) -> dict[str, list[Any]]:
    if n_combos not in GRID_SIZES:
        raise ValueError(f"grid size must be one of {sorted(GRID_SIZES)}, got {n_combos}")
    return {k: list(v) for k, v in GRID_SIZES[n_combos].items()}


def make_frame(periods: int, seed: int, edge: float = 0.0, volatility: float = 0.01) -> pd.DataFrame:
    """Daily synthetic prices; ``edge`` is the standard deviation of a persistent latent drift (0 = pure noise)."""
    cfg = SyntheticConfig(
        seed=seed, start="2020-01-01", periods=periods, freq="D", tz="UTC",
        regimes=[Regime(share=1.0, drift=0.0, volatility=volatility)],
        trend_persistence=TREND_PERSISTENCE if edge > 0 else 0.0, trend_scale=edge,
    )
    return generate(cfg)


class MaGridAdapter:
    """Moving-average crossover with a dead band; position decided at the close of bar t earns bar t+1."""

    def __init__(self, n_combos: int):
        self._space = grid_space(n_combos)

    def parameter_space(self) -> dict[str, list[Any]]:
        return {k: list(v) for k, v in self._space.items()}

    def required_data(self) -> list[DatasetRef]:
        return [DatasetRef("prices")]

    def run(
        self, params: dict[str, Any], data: DataView, start: pd.Timestamp, end: pd.Timestamp,
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


def params_for_labels(adapter: Any) -> dict[str, dict[str, Any]]:
    """Combination hash -> parameters, in the order the pipeline's S3 evaluates them."""
    from qrh.pipeline.stages_s3_s7 import grid_combos

    return dict(grid_combos(adapter.parameter_space()))


def build_matrix(adapter: Any, frame: pd.DataFrame, warmup: int = WARMUP) -> pd.DataFrame:
    """S3: every combination's returns over the development window (bars after the warm-up)."""
    bars = frame.index[warmup:]
    view = DataView(frame)
    cols = {
        label: adapter.run(params, view, bars[0], bars[-1]).returns.to_numpy(dtype=float)
        for label, params in params_for_labels(adapter).items()
    }
    return pd.DataFrame(cols, index=bars)


class ReplayAdapter:
    """Serves the stored in-sample returns of a combination as the adapter's run on any sub-span."""

    def __init__(self, matrix: pd.DataFrame):
        self.matrix = matrix

    def run(self, params: dict[str, Any], data: Any, start: pd.Timestamp, end: pd.Timestamp, **_: Any) -> RunResult:
        series = self.matrix[params["label"]].loc[start:end].astype(float)
        return RunResult(series, pd.DataFrame(), pd.DataFrame())


def evaluate_matrix(
    matrix: pd.DataFrame, wf_raw: dict[str, Any], thresholds: dict[str, Any], family_count: int,
    selector: MethodSelector, periods_per_year: float | None = None,
) -> dict[str, Any]:
    """S4 and S5 on an in-sample matrix. Unusable input gives ``error`` and no pass; nothing is raised.

    ``periods_per_year`` is inferred with the selected method unless given (a hypothesis may set it).
    """
    out: dict[str, Any] = {
        "error": None, "oos_sharpe": None, "pbo": None, "dsr": None,
        "s4_pass": False, "pbo_pass": False, "dsr_pass": False, "s5_pass": False, "pass_all": False,
    }
    try:
        ppy = periods_per_year if periods_per_year is not None else selector.get("metrics.periods_per_year").entry(matrix.index)
        if ppy is None:
            raise ValueError("periods_per_year cannot be inferred")
        wf = selector.get("s4.walkforward")
        assert wf.validate is not None
        wf.validate(wf_raw, len(matrix))
        labels = {c: {"label": c} for c in matrix.columns}
        view = DataView(pd.DataFrame({"x": 0.0}, index=matrix.index))
        res = wf.entry(ReplayAdapter(matrix), view, matrix.index, matrix, labels, wf_raw, ppy)
        oos_sharpe = annualized_sharpe(res.oos_returns.to_numpy(dtype=float), ppy)
        if oos_sharpe is None:
            raise ValueError("out-of-sample Sharpe ratio is undefined")
        stats = overfitting_statistics(
            matrix, res.oos_returns.to_numpy(dtype=float), family_count, int(thresholds["cscv_blocks"]),
            selector.get("s5.pbo").entry, selector.get("s5.dsr").entry,
        )
        s4 = oos_sharpe >= thresholds["min_oos_sharpe"]
        pbo_ok = stats.pbo is None or stats.pbo.pbo <= thresholds["max_pbo"]  # one combination: PBO is not applicable
        dsr_ok = stats.dsr.dsr >= thresholds["min_dsr"]
        out.update(
            oos_sharpe=float(oos_sharpe), pbo=None if stats.pbo is None else float(stats.pbo.pbo), dsr=float(stats.dsr.dsr),
            s4_pass=bool(s4), pbo_pass=bool(pbo_ok), dsr_pass=bool(dsr_ok), s5_pass=bool(pbo_ok and dsr_ok),
            pass_all=bool(s4 and pbo_ok and dsr_ok),
        )
    except (ValueError, MethodError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["methods"] = dict(sorted(selector.used.items()))
    return out

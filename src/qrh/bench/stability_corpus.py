"""The planted-case corpus for parameter stability (the ``param-stability`` suite).

Every case is a synthetic strategy family whose returns come from known truth: a shared noise factor, a small
idiosyncratic part per parameter setting, and a mean return (the "edge") that depends on the parameters.

- ``spike``: only the planted setting is profitable and its neighbours earn nothing (a lucky peak). The check must FAIL
  (or BLOCK, never PASS). Variants: a second parameter that changes nothing (it must not dilute the spike), grid points
  or declared neighbours only a hairline away, and settings written as text.
- ``controls`` (plateaus): the edge is the same everywhere, or falls off smoothly around the planted setting. The check
  must PASS, also on a wide log-spaced grid.

Each shape comes twice: a ``grid`` of settings (the neighbours are the grid's adjacent values) and a ``single``
pre-registered setting (the harness generates its neighbours). Nothing here judges anything: the suite runs the
harness's own ``s5.param_stability`` method.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd

from qrh.hashing import hash_json

CONTROLS = "controls"
CLASSES = ("spike",)
N_BARS = 1600
OOS_FROM = 600  # the walk-forward scores the bars from here on
PPY = 252.0
EDGE = 0.002  # mean return per bar of a profitable setting (volatility 0.01 per bar: a Sharpe ratio near 3)
SIGMA = 0.01
SHARED = 0.9  # share of the noise common to all settings of one family
INDEX = pd.date_range("2020-01-01", periods=N_BARS, freq="D", tz="UTC")
OOS_INDEX = INDEX[OOS_FROM:]
SPACE_3X3 = {"fast": [5, 10, 15], "slow": [20, 40, 60]}
SPACE_WIDE = {"fast": [3, 10, 30], "slow": [60, 120, 240]}  # log-spaced: the global share of a sound strategy is low
SPACE_5X5 = {"fast": [4, 7, 10, 13, 16], "slow": [20, 30, 40, 50, 60]}
CENTRE = {"fast": 10, "slow": 40}
CENTRE_WIDE = {"fast": 10, "slow": 120}


def _num(x: Any) -> float:
    return float(x)  # a setting may be written as text ("20"); the strategy reads the number


def _spike(at: dict[str, Any]) -> Callable[[dict[str, Any]], float]:
    return lambda p: EDGE if all(_num(p[k]) == _num(v) for k, v in at.items()) else 0.0


def _flat(p: dict[str, Any]) -> float:
    return EDGE


def _hill(at: dict[str, Any], width: float = 1.5) -> Callable[[dict[str, Any]], float]:
    """Smooth plateau: the edge keeps about 90% at a 50% change of one parameter and 80% at 50% of two."""
    return lambda p: EDGE * math.exp(-sum(((_num(p[k]) - _num(v)) / _num(v) / width) ** 2 for k, v in at.items()))


@dataclass(frozen=True)
class StabilityCase:
    bug_class: str  # "spike" or CONTROLS
    name: str
    shape: str  # "grid" or "single"
    space: dict[str, list[Any]]
    at: dict[str, Any]  # the selected setting
    edge: Callable[[dict[str, Any]], float]
    declared: dict[str, Any] | None = None  # the hypothesis `stability` block, when the case declares one


def corpus() -> list[StabilityCase]:
    one = {"w": [20]}
    pair = {k: [v] for k, v in CENTRE.items()}
    return [
        StabilityCase("spike", "grid_3x3_centre", "grid", SPACE_3X3, CENTRE, _spike(CENTRE)),
        StabilityCase("spike", "grid_3x3_corner", "grid", SPACE_3X3, {"fast": 5, "slow": 20}, _spike({"fast": 5, "slow": 20})),
        StabilityCase("spike", "grid_5x5_centre", "grid", SPACE_5X5, {"fast": 10, "slow": 40}, _spike(CENTRE)),
        StabilityCase("spike", "single_one_parameter", "single", one, {"w": 20}, _spike({"w": 20})),
        StabilityCase("spike", "single_two_parameters", "single", pair, CENTRE, _spike(CENTRE)),
        StabilityCase("spike", "single_spike_with_inert_parameter", "single", {"w": [20], "k": [1.0]}, {"w": 20, "k": 1.0},
                      _spike({"w": 20})),
        StabilityCase("spike", "grid_spike_with_inert_parameter", "grid", {"w": [10, 20, 40], "k": [1.0, 2.0]},
                      {"w": 20, "k": 1.0}, _spike({"w": 20})),
        StabilityCase("spike", "grid_spike_hairline_grid_points", "grid", {"w": [9.9, 10.0, 10.1]}, {"w": 10.0},
                      _spike({"w": 10.0})),
        StabilityCase("spike", "single_spike_hairline_declared", "single", {"w": [10.0]}, {"w": 10.0}, _spike({"w": 10.0}),
                      {"neighbours": {"w": [9.999, 10.001]}}),
        StabilityCase("spike", "single_spike_string_typed", "single", {"w": ["20"]}, {"w": "20"}, _spike({"w": 20})),
        StabilityCase("spike", "grid_spike_string_typed", "grid", {"w": ["10", "20", "40"]}, {"w": "20"}, _spike({"w": 20})),
        StabilityCase(CONTROLS, "grid_wide_log_flat", "grid", SPACE_WIDE, CENTRE_WIDE, _flat),
        StabilityCase(CONTROLS, "grid_wide_log_hill", "grid", SPACE_WIDE, CENTRE_WIDE, _hill(CENTRE_WIDE)),
        StabilityCase(CONTROLS, "grid_3x3_flat", "grid", SPACE_3X3, CENTRE, _flat),
        StabilityCase(CONTROLS, "grid_3x3_hill", "grid", SPACE_3X3, CENTRE, _hill(CENTRE)),
        StabilityCase(CONTROLS, "grid_5x5_hill", "grid", SPACE_5X5, CENTRE, _hill(CENTRE)),
        StabilityCase(CONTROLS, "single_one_parameter", "single", one, {"w": 20}, _flat),
        StabilityCase(CONTROLS, "single_two_parameters", "single", pair, CENTRE, _hill(CENTRE)),
    ]


def variants(cases: list[StabilityCase]) -> dict[str, list[str]]:
    return {b: [c.name for c in cases if c.bug_class == b] for b in (*CLASSES, CONTROLS)}


class PlantedAdapter:
    """Runs a case's synthetic strategy: the same returns the case's matrix holds for the same setting."""

    def __init__(self, case: StabilityCase, seed: int) -> None:
        self.case, self.seed = case, seed
        self._common = np.random.default_rng(seed).normal(0.0, 1.0, N_BARS)

    def returns(self, params: dict[str, Any]) -> pd.Series:
        own = np.random.default_rng(zlib.crc32(hash_json({k: float(v) for k, v in params.items()}).encode()) + self.seed)
        noise = SIGMA * (SHARED * self._common + math.sqrt(1 - SHARED**2) * own.normal(0.0, 1.0, N_BARS))
        return pd.Series(self.case.edge(params) + noise, index=INDEX)

    def run(self, params: dict[str, Any], data: Any, start: Any, end: Any, **_: Any) -> Any:
        return SimpleNamespace(returns=self.returns(params).loc[start:end])


def materialize(case: StabilityCase, seed: int) -> dict[str, Any]:
    """Everything ``s5.param_stability`` needs for one replication of a case."""
    from qrh.pipeline.stages_s3_s7 import grid_combos

    adapter = PlantedAdapter(case, seed)
    grid = grid_combos(case.space)
    matrix = pd.DataFrame({label: adapter.returns(p).to_numpy() for label, p in grid}, index=INDEX)
    selected = next(label for label, p in grid if p == case.at)
    return {"grid": grid, "matrix": matrix, "oos_index": OOS_INDEX, "picks": [selected] * 5, "ppy": PPY,
            "adapter": adapter, "view": None, "start": INDEX[0], "end": INDEX[-1], "declared": case.declared}

import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import DataView
from qrh.bench.engine import (
    GRID_SIZES,
    MaGridAdapter,
    ReplayAdapter,
    build_matrix,
    evaluate_matrix,
    grid_space,
    make_frame,
    seed_for,
)
from qrh.checks.truncation import run_truncation
from qrh.methods.selection import MethodSelector
from qrh.validation.defaults import DEFAULT_THRESHOLDS

WF = {"train": 0.3, "test": 0.1}


def test_seed_for_is_stable_and_distinguishes_parts():
    assert seed_for(7, "noise", 3, 1) == seed_for(7, "noise", 3, 1)
    assert len({seed_for(7, "noise", 3, r) for r in range(50)}) == 50
    assert seed_for(7, "noise", 3, 1) != seed_for(8, "noise", 3, 1)
    assert 0 <= seed_for(1) < 2**32


@pytest.mark.parametrize("n", sorted(GRID_SIZES))
def test_grid_space_has_exactly_the_requested_size(n):
    space = grid_space(n)
    assert int(np.prod([len(v) for v in space.values()])) == n
    assert MaGridAdapter(n).parameter_space() == space


def test_unknown_grid_size_is_an_error():
    with pytest.raises(ValueError):
        grid_space(7)


def test_make_frame_is_deterministic_and_edge_adds_persistent_trend():
    a, b = make_frame(400, seed=5, edge=0.0), make_frame(400, seed=5, edge=0.0)
    assert a.equals(b) and a.index.tz is not None and (a["close"] > 0).all()
    ret = lambda f: f["close"].pct_change().dropna()  # noqa: E731
    noise_ac = ret(make_frame(4000, seed=1, edge=0.0)).autocorr(1)
    edge_ac = ret(make_frame(4000, seed=1, edge=0.004)).autocorr(1)
    assert edge_ac > noise_ac + 0.05


def test_bench_strategy_has_no_lookahead_by_the_harnesss_own_test():
    frame = make_frame(300, seed=3, edge=0.002)
    ad = MaGridAdapter(12)
    params = {"band": 0.0, "fast": 3, "slow": 40}
    res = run_truncation(ad, frame, params, frame.index[60], frame.index[-1])
    assert res.verdict == "PASS", res.reasons


def test_build_matrix_shape_labels_and_window():
    frame = make_frame(300, seed=3, edge=0.0)
    ad = MaGridAdapter(12)
    matrix = build_matrix(ad, frame)
    assert matrix.shape == (240, 12) and matrix.index.equals(frame.index[60:])
    assert np.isfinite(matrix.to_numpy()).all()
    assert matrix.columns.is_unique and all(len(c) == 64 for c in matrix.columns)
    assert build_matrix(MaGridAdapter(12), frame).equals(matrix)


def test_replay_adapter_equals_the_real_adapter_on_test_spans():
    """The benchmark replays stored returns for walk-forward tests; for a strategy without look-ahead
    that is exactly what the adapter returns when it is run on the test span."""
    frame = make_frame(300, seed=3, edge=0.002)
    ad = MaGridAdapter(12)
    matrix = build_matrix(ad, frame)
    replay = ReplayAdapter(matrix)
    from qrh.bench.engine import params_for_labels

    labels = params_for_labels(ad)
    view = DataView(frame)
    for label, params in list(labels.items())[:4]:
        s, e = matrix.index[100], matrix.index[160]
        real = ad.run(params, view, s, e).returns
        assert replay.run({"label": label}, view, s, e).returns.equals(real)


def test_evaluate_matrix_reports_every_gate(tmp_path):
    frame = make_frame(1000, seed=2, edge=0.004)
    matrix = build_matrix(MaGridAdapter(12), frame)
    out = evaluate_matrix(matrix, WF, DEFAULT_THRESHOLDS, 12, MethodSelector(tmp_path))
    assert out["error"] is None
    for key in ("oos_sharpe", "pbo", "dsr", "s4_pass", "pbo_pass", "dsr_pass", "s5_pass", "pass_all"):
        assert key in out
    assert out["pass_all"] == (out["s4_pass"] and out["pbo_pass"] and out["dsr_pass"])
    assert out["methods"]["s5.pbo"] == "cscv@1"
    again = evaluate_matrix(matrix, WF, DEFAULT_THRESHOLDS, 12, MethodSelector(tmp_path))
    assert again == out


def test_evaluate_matrix_follows_the_selected_method_version(tmp_path):
    frame = make_frame(1000, seed=2, edge=0.0)
    matrix = build_matrix(MaGridAdapter(12), frame)
    base = evaluate_matrix(matrix, WF, DEFAULT_THRESHOLDS, 12, MethodSelector(tmp_path))
    alt = evaluate_matrix(matrix, WF, DEFAULT_THRESHOLDS, 12,
                          MethodSelector(tmp_path, overrides={"s4.walkforward": "purged_embargo@1"}))
    assert alt["methods"]["s4.walkforward"] == "purged_embargo@1"
    assert alt["pbo"] == base["pbo"]  # S5 on the same matrix is unchanged by an S4 method
    assert alt["oos_sharpe"] != base["oos_sharpe"]  # the embargoed OOS series differs


def test_evaluate_matrix_turns_unusable_input_into_an_error_outcome(tmp_path):
    matrix = pd.DataFrame(np.zeros((300, 4)), index=pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC"))
    matrix.columns = [f"c{i}" for i in range(4)]
    out = evaluate_matrix(matrix, WF, DEFAULT_THRESHOLDS, 4, MethodSelector(tmp_path))
    assert out["error"] and out["pass_all"] is False


def test_evaluate_matrix_with_one_combination_does_not_crash_and_pbo_is_not_applicable(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=1000, freq="B", tz="UTC")
    matrix = pd.DataFrame({"only": 0.0005 + 0.01 * rng.standard_normal(1000)}, index=idx)
    out = evaluate_matrix(matrix, {"train": 0.3, "test": 0.1}, DEFAULT_THRESHOLDS, 5, MethodSelector(tmp_path))
    assert out["error"] is None, out["error"]
    assert out["pbo"] is None and out["pbo_pass"] is True and out["dsr"] is not None
    assert out["s5_pass"] == out["dsr_pass"]

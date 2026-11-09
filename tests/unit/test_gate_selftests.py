"""Gate self-tests: every validation gate must be shown able to FAIL, and to PASS a clean control.

Synthetic data only. Thresholds come from qrh.validation.defaults.
"""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from qrh.adapters.base import RunResult
from qrh.checks.data_quality import run_rules
from qrh.checks.truncation import run_truncation
from qrh.cli import app
from qrh.pipeline.receipt import Stage, Verdict, load_receipt
from qrh.validation.defaults import DEFAULT_THRESHOLDS
from qrh.validation.deflated_sharpe import deflated_sharpe
from qrh.validation.pbo_cscv import pbo_cscv
from qrh.validation.wfo import make_folds, parse_walk_forward, validate_rolling_anchored

MAX_PBO = DEFAULT_THRESHOLDS["max_pbo"]
MIN_DSR = DEFAULT_THRESHOLDS["min_dsr"]
EXAMPLE_01 = Path(__file__).resolve().parents[2] / "examples" / "01_clean_strategy"
runner = CliRunner()


# --- S5 deflated Sharpe --------------------------------------------------------------------------


def _returns(mean, seed=0, n=500, sd=0.01):
    return np.random.default_rng(seed).normal(mean, sd, n)


def test_s5_dsr_overfit_many_trials_weak_best_fails_and_strong_control_passes():
    n_trials, var_sharpe = 1000, 0.02**2  # many trials with a wide spread of trial Sharpes
    weak = deflated_sharpe(_returns(0.0003), n_trials, var_sharpe)
    assert weak.sharpe > 0 and weak.sr0 > weak.sharpe  # the best result is below the expected max of noise
    assert not weak.dsr >= MIN_DSR, weak.dsr
    strong = deflated_sharpe(_returns(0.003), n_trials, var_sharpe)
    assert strong.dsr >= MIN_DSR, strong.dsr


# --- S5 PBO ----------------------------------------------------------------------------------------


def test_s5_pbo_in_sample_winner_is_out_of_sample_loser_breaches_and_skilled_control_passes():
    rng = np.random.default_rng(1)
    blocks, length, n = 16, 20, 10
    mu = rng.normal(0, 0.01, (blocks, n))
    mu -= mu.mean(axis=0)  # equal total mean per strategy: a high in-sample block sum means a low out-of-sample one
    planted = np.repeat(mu, length, axis=0) + rng.normal(0, 0.002, (blocks * length, n))
    bad = pbo_cscv(planted, n_blocks=16)
    assert not bad.pbo <= MAX_PBO and bad.pbo > 0.8, bad.pbo

    drift = np.linspace(0.0, 0.004, n)  # a real, persistent ranking of strategies
    control = drift + rng.normal(0, 0.01, (blocks * length, n))
    good = pbo_cscv(control, n_blocks=16)
    assert good.pbo <= MAX_PBO, good.pbo


# --- S6 cost stress (full pipeline, example 01 with a cost-sensitive adapter) -------------------------


def _run_example_01(tmp_path, monkeypatch, cost_slope):
    """Run example 01; cost_slope scales how strongly cost_multiplier raises costs (None = the shipped adapter)."""
    home = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(home))
    monkeypatch.setenv("QRH_REPORTS", "0")
    d = tmp_path / "ex"
    shutil.copytree(EXAMPLE_01, d, ignore=shutil.ignore_patterns("__pycache__"))
    if cost_slope is not None:
        p = d / "adapter.py"
        old = "COST_BPS * cost_multiplier"
        src = p.read_text()
        assert old in src
        p.write_text(src.replace(old, f"COST_BPS * (1 + {cost_slope} * (cost_multiplier - 1))"))
    spec = importlib.util.spec_from_file_location("gate_selftest_make_data", d / "make_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = mod.make(d / "data")
    res = runner.invoke(app, ["data", "register", str(data), "--name", "prices", "--rules", str(d / "rules.yaml")])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["new", str(d / "hypothesis.yaml")])
    assert res.exit_code == 0, res.output
    tid = res.stdout.strip().splitlines()[-1]
    runner.invoke(app, ["run", tid])
    return lambda stage: load_receipt(home / "receipts" / tid / f"{Stage(stage).value}.json")


def test_s6_negative_cost_stressed_sharpe_fails_cost_stress_rule_and_clean_control_passes(tmp_path, monkeypatch):
    floor = DEFAULT_THRESHOLDS["min_cost_stressed_sharpe"]
    control = _run_example_01(tmp_path / "control", monkeypatch, None)
    assert control("S6").verdict is Verdict.PASS and control("S6").metrics["cost_stressed_sharpe"] >= floor

    planted = _run_example_01(tmp_path / "planted", monkeypatch, 200.0)
    assert planted("S4").verdict is Verdict.PASS and planted("S5").verdict is Verdict.PASS  # only the cost gate fires
    s6 = planted("S6")
    assert s6.metrics["cost_stressed_sharpe"] < 0 and s6.metrics["cost_stressed_sharpe"] < floor
    assert s6.verdict is Verdict.FAIL and any("cost-stressed" in r for r in s6.reasons)


# --- S4 walk-forward ---------------------------------------------------------------------------------


def test_s4_window_shorter_than_train_plus_test_is_refused_and_long_enough_window_passes():
    raw = {"train": 400, "test": 100}
    with pytest.raises(ValueError, match="window too short"):
        validate_rolling_anchored(raw, 450)
    with pytest.raises(ValueError, match="window too short"):
        make_folds(499, parse_walk_forward(raw))
    validate_rolling_anchored(raw, 600)  # control: no exception
    assert len(make_folds(600, parse_walk_forward(raw))) >= 1


def test_s4_overlapping_test_spans_are_refused_and_non_overlapping_control_passes():
    with pytest.raises(ValueError, match="must be >= test"):
        validate_rolling_anchored({"train": 100, "test": 50, "step": 10}, 1000)
    validate_rolling_anchored({"train": 100, "test": 50, "step": 50}, 1000)


# --- S2 truncation -------------------------------------------------------------------------------------


class _Adapter:
    def __init__(self, peek):
        self.peek = peek

    def run(self, params, data, start, end):
        px = data.as_of(end)["close"]
        ret = px.pct_change()
        sig = np.sign(ret.shift(-1)) if self.peek else np.sign(ret)  # shift(-1) uses the next bar's return
        pos = sig.fillna(0.0).to_frame("w")
        strat = (pos["w"].shift(1).fillna(0.0) * ret.fillna(0.0)).loc[start:end]
        return RunResult(strat, pd.DataFrame(), pos.loc[start:end])


def _prices(n=240, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC", name="timestamp")
    return pd.DataFrame({"close": 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)


def test_s2_future_dependent_signal_fails_truncation_and_causal_control_passes():
    df = _prices()
    start, end = df.index[10], df.index[-1]
    clean = run_truncation(_Adapter(peek=False), df, {}, start, end)
    assert clean.verdict == "PASS", clean.reasons
    planted = run_truncation(_Adapter(peek=True), df, {}, start, end)
    assert planted.verdict == "FAIL" and planted.reasons


# --- S1 data quality ------------------------------------------------------------------------------------

GAP_RULE = [{"type": "gaps", "freq": "D"}]


def _daily(n=60):
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"close": np.linspace(100, 110, n)}, index=idx)


def test_s1_planted_gap_fails_data_quality_and_complete_control_passes():
    df = _daily()
    assert run_rules(df, GAP_RULE).verdict == "PASS"
    gapped = df.drop(df.index[[20, 21]])
    rep = run_rules(gapped, GAP_RULE)
    assert rep.verdict == "FAIL" and any("missing timestamp" in r for r in rep.reasons)


def test_s1_planted_duplicate_fails_data_quality_and_unique_control_passes():
    df = _daily()
    assert run_rules(df, GAP_RULE).verdict == "PASS"
    dup = pd.concat([df, df.iloc[[30]]]).sort_index()
    rep = run_rules(dup, GAP_RULE)
    assert rep.verdict == "FAIL" and any("duplicate" in r for r in rep.reasons)

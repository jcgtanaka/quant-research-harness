"""S5 reports the minimum track record length as information only: no verdict or threshold depends on it."""

import importlib.util
import math
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.pipeline.receipt import Stage, Verdict, load_receipt
from qrh.pipeline.stages_s3_s7 import MIN_TRL_BENCHMARK, MIN_TRL_CONFIDENCE, min_track_record_info
from qrh.validation.min_track_record import min_track_record_length

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
runner = CliRunner()


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    monkeypatch.setenv("QRH_REPORTS", "0")
    return h


def _run_example(name, tmp_path, home):
    d = tmp_path / name
    shutil.copytree(EXAMPLES / name, d, ignore=shutil.ignore_patterns("__pycache__"))
    spec = importlib.util.spec_from_file_location(f"make_data_{name}", d / "make_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = mod.make(d / "data")
    res = runner.invoke(app, ["data", "register", str(data), "--name", "prices", "--rules", str(d / "rules.yaml")])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["new", str(d / "hypothesis.yaml")])
    assert res.exit_code == 0, res.output
    trial = res.stdout.strip().splitlines()[-1]
    runner.invoke(app, ["run", trial])
    return load_receipt(home / "receipts" / trial / f"{Stage('S5').value}.json")


def test_constants_are_the_informational_defaults():
    assert MIN_TRL_BENCHMARK == 0.0 and MIN_TRL_CONFIDENCE == 0.95


def test_info_returns_integer_for_valid_inputs():
    periods, reason = min_track_record_info(0.1, 0.0, 3.0)
    assert periods == min_track_record_length(0.1, MIN_TRL_BENCHMARK, 0.0, 3.0, MIN_TRL_CONFIDENCE)
    assert isinstance(periods, int) and reason is None


@pytest.mark.parametrize("sharpe", [0.0, -0.05])
def test_info_is_null_with_reason_when_sharpe_not_above_benchmark(sharpe):
    periods, reason = min_track_record_info(sharpe, 0.0, 3.0)
    assert periods is None and "benchmark" in reason


def test_info_is_null_with_reason_for_non_finite_inputs():
    periods, reason = min_track_record_info(0.1, math.nan, 3.0)
    assert periods is None and reason


def test_s5_pass_receipt_carries_field_and_stays_pass(home, tmp_path):
    s5 = _run_example("01_clean_strategy", tmp_path, home)
    assert s5.verdict is Verdict.PASS
    assert "min_track_record_periods" in s5.params
    assert isinstance(s5.params["min_track_record_periods"], int)
    assert s5.params["min_track_record_periods"] == min_track_record_length(
        s5.metrics["oos_sharpe_per_period"], 0.0, s5.metrics["oos_skew"], s5.metrics["oos_kurtosis"], 0.95
    )


def test_s5_fail_receipt_carries_field_and_stays_fail(home, tmp_path):
    s5 = _run_example("04_overfit_grid", tmp_path, home)
    assert s5.verdict is Verdict.FAIL
    assert "min_track_record_periods" in s5.params
    value = s5.params["min_track_record_periods"]
    assert value is None or isinstance(value, int)
    if value is None:
        assert s5.params["min_track_record_reason"]

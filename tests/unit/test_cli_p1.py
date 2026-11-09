import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qrh.cli import app

runner = CliRunner()
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "hypothesis.example.yaml"


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


def new_trial() -> str:
    res = runner.invoke(app, ["new", str(EXAMPLE)])
    assert res.exit_code == 0, res.output
    return res.stdout.strip().splitlines()[-1]


def test_example_file_is_valid_and_registers():
    tid = new_trial()
    assert len(tid) >= 8


def test_new_accepts_json(tmp_path):
    spec = {
        "text": "t", "strategy_family": "s", "asset_class": "a", "frequency": "f",
        "parameter_space_size": 1, "thresholds": {"k": 1}, "difference_statement": "d",
    }
    p = tmp_path / "h.json"
    p.write_text(json.dumps(spec))
    assert runner.invoke(app, ["new", str(p)]).exit_code == 0


def test_new_invalid_exits_2_with_reasons(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("text: x\nthresholds: {}\ndifference_statement: ''\n")
    res = runner.invoke(app, ["new", str(p)])
    assert res.exit_code == 2
    assert "difference_statement" in res.output and "thresholds" in res.output


def test_new_missing_or_unparseable_file_exits_2(tmp_path):
    assert runner.invoke(app, ["new", str(tmp_path / "nope.yaml")]).exit_code == 2
    p = tmp_path / "x.yaml"
    p.write_text("a: [unclosed")
    assert runner.invoke(app, ["new", str(p)]).exit_code == 2


def test_round_trip_new_status_run():
    tid = new_trial()

    listing = runner.invoke(app, ["status"])
    assert listing.exit_code == 0 and tid in listing.output and "PASS" in listing.output

    detail = runner.invoke(app, ["status", tid])
    assert detail.exit_code == 0 and "S0" in detail.output and "PASS" in detail.output

    res = runner.invoke(app, ["run", tid])
    assert res.exit_code == 2
    assert "S7.json" in res.output
    assert "BLOCKED" in res.output

    detail = runner.invoke(app, ["status", tid])
    assert "S1" in detail.output and "BLOCKED" in detail.output
    assert "no datasets declared" in detail.output

    listing = runner.invoke(app, ["status"])
    assert tid in listing.output and "BLOCKED" in listing.output


def test_status_and_run_unknown_trial_exit_2():
    assert runner.invoke(app, ["status", "nope"]).exit_code == 2
    assert runner.invoke(app, ["run", "nope"]).exit_code == 2


def test_status_reports_tampered_receipt(home):
    tid = new_trial()
    (home / "receipts" / tid / "S0.json").write_text("{}")
    res = runner.invoke(app, ["status", tid])
    assert res.exit_code == 2 and "BLOCKED" in res.output


def test_unwritable_home_fails_closed(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("QRH_HOME", str(blocker / "home"))
    for args in (["new", str(EXAMPLE)], ["status"], ["run", "x"], ["data", "status"]):
        res = runner.invoke(app, args)
        assert res.exit_code == 2, args
        assert "QRH_HOME" in res.output


def test_data_register_and_status(tmp_path):
    p = tmp_path / "bars.csv"
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC", name="timestamp")
    pd.DataFrame({"close": range(5)}, index=idx).to_csv(p)
    res = runner.invoke(app, ["data", "register", str(p), "--name", "mydata"])
    assert res.exit_code == 0 and "mydata" in res.output and "UNCHECKED" in res.output
    one = runner.invoke(app, ["data", "status", "mydata"])
    assert one.exit_code == 0 and "UNCHECKED" in one.output
    allx = runner.invoke(app, ["data", "status"])
    assert allx.exit_code == 0 and "mydata" in allx.output


def test_data_unknown_and_unreadable_are_blocked(tmp_path):
    res = runner.invoke(app, ["data", "status", "nope"])
    assert res.exit_code == 2 and "BLOCKED" in res.output
    res = runner.invoke(app, ["data", "register", str(tmp_path / "gone.csv")])
    assert res.exit_code == 2 and "BLOCKED" in res.output


def test_data_status_empty_catalog_is_ok():
    assert runner.invoke(app, ["data", "status"]).exit_code == 0

import pandas as pd
import pytest
from typer.testing import CliRunner

from qrh.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


def csv(tmp_path, drop=None):
    idx = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC", name="timestamp")
    df = pd.DataFrame({"close": range(1, 31)}, index=idx, dtype=float)
    if drop is not None:
        df = df.drop(df.index[drop])
    p = tmp_path / "px.csv"
    df.to_csv(p)
    return p


RULES = "rules:\n  - type: gaps\n    freq: D\n  - type: range\n    column: close\n    positive: true\n"


def test_register_with_rules_then_check_pass(tmp_path):
    rules = tmp_path / "px.rules.yaml"
    rules.write_text(RULES)
    assert runner.invoke(app, ["data", "register", str(csv(tmp_path)), "--rules", str(rules)]).exit_code == 0
    res = runner.invoke(app, ["data", "check", "px"])
    assert res.exit_code == 0 and "PASS" in res.output
    assert "PASS" in runner.invoke(app, ["data", "status", "px"]).output


def test_check_failure_exits_2_with_reason(tmp_path):
    rules = tmp_path / "px.rules.yaml"
    rules.write_text(RULES)
    runner.invoke(app, ["data", "register", str(csv(tmp_path, drop=[4])), "--rules", str(rules)])
    res = runner.invoke(app, ["data", "check", "px"])
    assert res.exit_code == 2 and "BLOCKED" in res.output and "2020-01-05" in res.output


def test_check_without_rules_is_unchecked_exit_2(tmp_path):
    runner.invoke(app, ["data", "register", str(csv(tmp_path))])
    res = runner.invoke(app, ["data", "check", "px"])
    assert res.exit_code == 2 and "UNCHECKED" in res.output and "no quality rules declared" in res.output


def test_check_unknown_dataset_exit_2():
    assert runner.invoke(app, ["data", "check", "nope"]).exit_code == 2


def test_lint_command(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("y = x.shift(-1)\n")
    res = runner.invoke(app, ["lint", str(bad)])
    assert res.exit_code == 2 and "LA001" in res.output and "bad.py:1" in res.output
    ok = tmp_path / "ok.py"
    ok.write_text("y = x.shift(1)\n")
    assert runner.invoke(app, ["lint", str(ok)]).exit_code == 0
    warn = tmp_path / "w.py"
    warn.write_text("m = x.mean()\ny = z - m\n")
    res = runner.invoke(app, ["lint", str(warn)])
    assert res.exit_code == 0 and "WARN" in res.output
    sup = tmp_path / "s.py"
    sup.write_text("y = x.shift(-1)  # qrh: allow LA001 evaluation label only\n")
    res = runner.invoke(app, ["lint", str(sup)])
    assert res.exit_code == 0 and "evaluation label only" in res.output


def test_lint_missing_file_blocked(tmp_path):
    assert runner.invoke(app, ["lint", str(tmp_path / "nope.py")]).exit_code == 2

"""The one-attempt bug correction is wired into S2 for built backtests (pnl.linear path).

Synthetic data only (example 01 copied into a temporary directory, temporary QRH_HOME).
The run directory is <home>/trials/<trial>/S2/correction.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from qrh.checks.bug_correction import record_correction
from qrh.cli import app
from qrh.pipeline.receipt import Stage, Verdict, load_receipt

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "01_clean_strategy"
GAIN_FREE = "(position.shift(1) * ret - cost.shift(1))"
BUGGED = "(position.shift(1) * ret - cost.shift(1) + 0.0005)"
BLOCK = {
    "position_units": "weight_start_of_bar",
    "cost_model": {"bps": 5.0},
    "asset_class": "equity",
    "min_cost_per_turnover": 0.0005,
    "publication_column": "pub",
    "max_staleness": 2,
    "adjustment": {"close": "total_return"},
    "max_unexplained_bp_per_year": 50.0,
    "max_unexplained_share": 0.2,
}
runner = CliRunner()


class Setup:
    def __init__(self, tmp_path, monkeypatch, adapter_edit):
        self.home = tmp_path / "qrh_home"
        monkeypatch.setenv("QRH_HOME", str(self.home))
        d = tmp_path / "ex"
        shutil.copytree(EXAMPLE, d, ignore=shutil.ignore_patterns("__pycache__"))
        spec = importlib.util.spec_from_file_location("make_data_wiring", d / "make_data.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        data = mod.make(d / "data")
        raw = pd.read_csv(data, index_col=0, dtype=str)
        raw["pub"] = raw.index
        raw.to_csv(data)
        self.adapter = d / "adapter.py"
        self.original = self.adapter.read_text()
        self.set_adapter(adapter_edit)
        spec_path = d / "hypothesis.yaml"
        hyp = yaml.safe_load(spec_path.read_text())
        hyp["reconstruction"] = BLOCK
        spec_path.write_text(yaml.safe_dump(hyp))
        res = runner.invoke(app, ["data", "register", str(data), "--name", "prices", "--rules", str(d / "rules.yaml")])
        assert res.exit_code == 0, res.output
        res = runner.invoke(app, ["new", str(spec_path)])
        assert res.exit_code == 0, res.output
        self.trial = res.stdout.strip().splitlines()[-1]
        self.correction_dir = self.home / "trials" / self.trial / Stage.S2.value / "correction"

    def set_adapter(self, edit):
        self.adapter.write_text(self.original.replace(GAIN_FREE, edit))

    def run(self):
        runner.invoke(app, ["run", self.trial])
        return load_receipt(self.home / "receipts" / self.trial / f"{Stage.S2.value}.json")


def text(receipt) -> str:
    return " | ".join(receipt.reasons)


def test_a_bug_gives_inconclusive_and_writes_the_request(tmp_path, monkeypatch):
    s = Setup(tmp_path, monkeypatch, BUGGED)
    s2 = s.run()
    assert s2.verdict is Verdict.INCONCLUSIVE, text(s2)
    assert "a bug was found; a correction has been requested and the run is re-checked after it is recorded" in text(s2)
    request = s.correction_dir / "correction_request.txt"
    assert request.exists() and "0.0005" in request.read_text()


def test_after_one_correction_the_rerun_passes(tmp_path, monkeypatch):
    s = Setup(tmp_path, monkeypatch, BUGGED)
    s.run()
    s.set_adapter(GAIN_FREE)
    record_correction(s.correction_dir, "removed the hidden gain")
    s2 = s.run()
    assert s2.verdict is Verdict.PASS, text(s2)
    assert "The bug was corrected once and the run is re-checked" in text(s2)


def test_a_bug_that_remains_after_the_correction_is_inconclusive_and_named(tmp_path, monkeypatch):
    s = Setup(tmp_path, monkeypatch, BUGGED)
    s.run()
    record_correction(s.correction_dir, "tried a fix")
    s2 = s.run()
    assert s2.verdict is Verdict.INCONCLUSIVE, text(s2)
    assert "still there after the one correction attempt" in text(s2) and "amount 0.0005" in text(s2)
    again = s.run()  # the single attempt stays used: still INCONCLUSIVE, bug named
    assert again.verdict is Verdict.INCONCLUSIVE and "amount 0.0005" in text(again)


def test_a_clean_built_backtest_is_unchanged(tmp_path, monkeypatch):
    s = Setup(tmp_path, monkeypatch, GAIN_FREE)
    s2 = s.run()
    assert s2.verdict is Verdict.PASS, text(s2)
    assert "correction" not in text(s2).lower()
    assert not s.correction_dir.exists()

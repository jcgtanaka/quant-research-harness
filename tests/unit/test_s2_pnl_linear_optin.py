"""pnl.linear is an opt-in reconstruction method in S2: it runs only where a hypothesis declares `reconstruction`.

Synthetic data only. Each test copies example 01 (a trend strategy with correct timing) into a temporary directory,
adds a publication-time column to its prices and runs S0 to S2 through the command line.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from qrh.cli import app
from qrh.memory.registry import HypothesisRecord
from qrh.methods.builtin import BUILTIN
from qrh.pipeline.receipt import Stage, Verdict, load_receipt

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "01_clean_strategy"
LINEAR = "s2.pnl_linear"
LINEAR_VER = "pnl_linear@1"
RECON = "s2.pnl_reconstruction"
GAIN_FREE = "(position.shift(1) * ret - cost.shift(1))"
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


def _build(tmp: Path) -> tuple[Path, Path]:
    d = tmp / "ex"
    shutil.copytree(EXAMPLE, d, ignore=shutil.ignore_patterns("__pycache__"))
    spec = importlib.util.spec_from_file_location("make_data_ex", d / "make_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = mod.make(d / "data")
    raw = pd.read_csv(path, index_col=0, dtype=str)
    raw["pub"] = raw.index  # every mark is published at its own bar time
    raw.to_csv(path)
    return d, path


def run_s2(tmp_path, monkeypatch, *, reconstruction="absent", adapter_edit=None, spec_edit=None):
    """Runs the example through the CLI and returns its S2 receipt."""
    home = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(home))
    d, data = _build(tmp_path)
    if adapter_edit is not None:
        a = d / "adapter.py"
        a.write_text(a.read_text().replace(GAIN_FREE, adapter_edit))
    spec_path = d / "hypothesis.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    if reconstruction != "absent":
        spec["reconstruction"] = reconstruction
    if spec_edit is not None:
        spec_edit(spec)
    spec_path.write_text(yaml.safe_dump(spec))
    res = runner.invoke(app, ["data", "register", str(data), "--name", "prices", "--rules", str(d / "rules.yaml")])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["new", str(spec_path)])
    assert res.exit_code == 0, res.output
    trial = res.stdout.strip().splitlines()[-1]
    runner.invoke(app, ["run", trial])
    return load_receipt(home / "receipts" / trial / f"{Stage.S2.value}.json")


def text(receipt) -> str:
    return " | ".join(receipt.reasons)


def test_registered_experimental_with_no_default():
    m = BUILTIN.get(LINEAR, LINEAR_VER)
    assert m.status == "experimental" and m.stage == "S2" and m.validate is not None
    assert not any(v.status == "default" for v in BUILTIN.versions(LINEAR))


def test_hypothesis_without_reconstruction_is_unchanged(tmp_path, monkeypatch):
    base = run_s2(tmp_path / "a", monkeypatch)
    again = run_s2(tmp_path / "b", monkeypatch, reconstruction=None)  # an explicit null is the same as absent
    assert base.verdict is Verdict.PASS
    assert (again.verdict, again.reasons, again.methods) == (base.verdict, base.reasons, base.methods)
    assert again.hypothesis_hash == base.hypothesis_hash
    assert set(again.params) == set(base.params)
    assert "pnl_linear" not in base.params and "pnl_linear" not in base.params["artifacts"]
    assert LINEAR not in base.methods and LINEAR not in base.params.get("non_default_methods", {})
    assert base.methods[RECON] == "pnl_reconstruction@1"  # the existing opt-in check is untouched


def test_record_without_the_key_has_the_old_hash_and_binding():
    spec = {"text": "h", "strategy_family": "s", "asset_class": "a", "frequency": "f", "parameter_space_size": 4,
            "thresholds": {"min": 1}, "difference_statement": "differs by x"}
    plain = HypothesisRecord(**spec)
    assert plain.reconstruction is None and "reconstruction" not in plain.binding()
    assert HypothesisRecord(**spec, reconstruction=None).hypothesis_hash() == plain.hypothesis_hash()
    declared = HypothesisRecord(**spec, reconstruction=BLOCK)
    assert declared.binding()["reconstruction"] == BLOCK and declared.hypothesis_hash() != plain.hypothesis_hash()


def test_valid_block_and_clean_run_passes_and_records_method_and_version(tmp_path, monkeypatch):
    s2 = run_s2(tmp_path, monkeypatch, reconstruction=BLOCK)
    assert s2.verdict is Verdict.PASS, text(s2)
    assert s2.methods[LINEAR] == LINEAR_VER
    assert s2.params["non_default_methods"][LINEAR] == LINEAR_VER
    p = s2.params["pnl_linear"]
    assert p["verdict"] == "PASS" and p["method"] == LINEAR and p["version"] == LINEAR_VER
    assert p["absolute_cap"]["breached"] is False and p["share"]["breached"] is False
    assert p["settings"]["capital_base"] == 1.0 and p["settings"]["bars_per_year"] > 0
    assert "pnl_linear" in s2.params["artifacts"]
    assert s2.params["pnl_reconstruction"]["verdict"] == "PASS"  # the default-path check still runs on its own


def test_a_hidden_gain_in_a_built_backtest_is_a_bug_warning_with_day_and_amount(tmp_path, monkeypatch):
    # Owner rule 3 (2026-10-27): for a harness-built backtest any unexplained difference is a bug shown as a
    # warning (day and amount); the noise-scaled cap applies only to comparisons with broker records.
    s2 = run_s2(tmp_path, monkeypatch, reconstruction=BLOCK,
                adapter_edit="(position.shift(1) * ret - cost.shift(1) + 0.0005)")
    assert s2.verdict is not Verdict.PASS
    assert "pnl_linear: warning: bug: unexplained difference" in text(s2) and "amount 0.0005" in text(s2)
    assert s2.params["pnl_linear"]["warnings"]


def test_a_hidden_gain_in_a_user_brought_backtest_still_breaks_the_caps_and_fails(tmp_path, monkeypatch):
    s2 = run_s2(tmp_path, monkeypatch, reconstruction=BLOCK,
                adapter_edit="(position.shift(1) * ret - cost.shift(1) + 0.0005)",
                spec_edit=lambda spec: spec.update(origin="user"))
    assert s2.verdict is Verdict.FAIL
    assert s2.params["pnl_linear"]["verdict"] == "FAIL" and "pnl_linear: fail: absolute cap breached" in text(s2)


def test_missing_cost_floor_is_blocked_with_the_validator_message(tmp_path, monkeypatch):
    block = {k: v for k, v in BLOCK.items() if k != "min_cost_per_turnover"}
    s2 = run_s2(tmp_path, monkeypatch, reconstruction=block)
    assert s2.verdict is Verdict.BLOCKED
    assert "reconstruction.min_cost_per_turnover: required for the asset class" in text(s2)
    assert s2.params["pnl_linear"]["verdict"] == "BLOCKED"


@pytest.mark.parametrize("bad", [
    {**BLOCK, "surprise": 1},
    {**BLOCK, "position_units": "shares"},
    {**BLOCK, "asset_class": "option"},
    {**BLOCK, "publication_column": "missing_column"},
])
def test_an_invalid_block_is_blocked(tmp_path, monkeypatch, bad):
    s2 = run_s2(tmp_path, monkeypatch, reconstruction=bad)
    assert s2.verdict is Verdict.BLOCKED and "reconstruction." in text(s2)
    assert s2.params["pnl_linear"]["verdict"] == "BLOCKED"


def test_a_block_without_execution_is_blocked(tmp_path, monkeypatch):
    def drop(spec):
        del spec["execution"], spec["methods"]  # the other opt-in check must not be the one that blocks

    s2 = run_s2(tmp_path, monkeypatch, reconstruction=BLOCK, spec_edit=drop)
    assert s2.verdict is Verdict.BLOCKED and "pnl_linear:" in text(s2) and "lag_bars" in text(s2)

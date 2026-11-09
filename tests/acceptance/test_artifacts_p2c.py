"""Every stage persists the evidence its figures need, and records each artifact's hash."""

import math

import pandas as pd
import pytest

from qrh.pipeline.artifacts import read_artifact
from qrh.pipeline.receipt import Verdict


def art(ex, stage, name):
    return read_artifact(ex.home, ex.receipt(stage), name)


EXPECTED = {
    "S0": {"hypothesis"},
    "S1": {"coverage"},
    "S2": {"lint", "truncation", "sanity"},
    "S3": {"combos", "is_returns"},
    "S4": {"folds", "oos_returns", "oos_equity"},
    "S5": {"pbo_logits", "dsr_inputs"},
    "S6": {"bootstrap_samples", "bootstrap_summary", "equity_compare"},
}


@pytest.fixture
def ran01(example):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    return ex


def test_01_every_stage_records_verifiable_artifacts(ran01):
    for stage, names in EXPECTED.items():
        assert set(ran01.receipt(stage).params["artifacts"]) >= names, stage
        for name in names:
            assert art(ran01, stage, name) is not None  # hash verified on read


def test_01_s0_hypothesis_card_has_everything_a_text_card_needs(ran01):
    h = art(ran01, "S0", "hypothesis")
    assert h["trial_id"] == ran01.trial_id and h["strategy_family"] == "trend" and h["frequency"] == "1D"
    assert h["asset_class"] == "synthetic" and "moving-average" in h["text"]
    assert h["thresholds"]["max_pbo"] == 0.2 and h["parameter_space_size"] == 9


def test_01_s1_coverage_and_rule_results(ran01):
    cov = art(ran01, "S1", "coverage")["datasets"]
    assert [d["name"] for d in cov] == ["prices"]
    d = cov[0]
    assert d["rows"] == 1500 and d["start"].startswith("2020-01-01") and d["gap_count"] == 0
    assert d["rules"] and all(r["passed"] and r["offenders"] == [] and r["type"] for r in d["rules"])


def test_05_failing_s1_still_persists_rule_results_with_offenders(example):
    ex = example("05_data_corruption", "linked")
    ex.run()
    d = art(ex, "S1", "coverage")["datasets"][0]
    failed = [r for r in d["rules"] if not r["passed"]]
    assert failed and failed[0]["offenders"] and failed[0]["reasons"]
    assert len(failed[0]["offenders"]) <= 50


def test_01_s2_lint_truncation_and_sanity_evidence(ran01):
    lint = art(ran01, "S2", "lint")
    assert lint["verdict"] == "PASS" and isinstance(lint["findings"], list)
    tr = art(ran01, "S2", "truncation")
    assert len(tr["cutoffs"]) == 3
    for c in tr["cutoffs"]:
        assert c["status"] == "PASS" and c["first_divergence"] is None
        assert c["timestamps"] and c["full"] == c["truncated"]
    san = art(ran01, "S2", "sanity")
    assert san["verdict"] == "PASS"
    names = {r["name"] for r in san["rows"]}
    assert {"max_abs_period_return", "max_gross_leverage", "max_position_weight"} <= names
    assert all(r["ok"] and r["bound"] is not None for r in san["rows"] if r["bound"] is not None)


def test_02_lookahead_persists_findings_and_the_first_divergence(example):
    ex = example("02_lookahead", "prices")
    ex.run()
    lint = art(ex, "S2", "lint")
    assert lint["verdict"] == "FAIL" and any(f["severity"] == "BLOCK" for f in lint["findings"])
    tr = art(ex, "S2", "truncation")
    assert any(c["first_divergence"] for c in tr["cutoffs"])


def test_01_s3_per_combo_metrics(ran01):
    combos = art(ran01, "S3", "combos")
    assert combos["param_names"] == ["fast", "slow"] and len(combos["combos"]) == 9
    best = max(c["sharpe_annualized"] for c in combos["combos"])
    assert best == pytest.approx(ran01.receipt("S3").metrics["best_is_sharpe"])
    assert all({"combo", "params", "sharpe_annualized", "cagr", "max_drawdown"} <= set(c) for c in combos["combos"])


def test_01_s4_folds_and_equity(ran01):
    folds = art(ran01, "S4", "folds")
    assert len(folds["folds"]) == ran01.receipt("S4").metrics["n_folds"]
    f = folds["folds"][0]
    assert {"fold", "train", "test", "params", "train_sharpe", "test_sharpe"} <= set(f)
    oos = art(ran01, "S4", "oos_returns")["returns"]
    eq = art(ran01, "S4", "oos_equity")["equity"]
    assert len(eq) == len(oos) and eq.iloc[-1] == pytest.approx((1 + oos).prod())


def test_01_s5_lambda_distribution_and_dsr_inputs(ran01):
    s5 = ran01.receipt("S5")
    logits = art(ran01, "S5", "pbo_logits")["logit"].to_numpy()
    assert len(logits) == math.comb(16, 8)
    assert (logits <= 0).mean() == pytest.approx(s5.metrics["pbo"])
    d = art(ran01, "S5", "dsr_inputs")
    assert d["sharpe"] == pytest.approx(s5.metrics["oos_sharpe_per_period"])
    assert d["sr0"] == pytest.approx(s5.metrics["sr0"]) and d["dsr"] == pytest.approx(s5.metrics["dsr"])
    assert {"n_trials", "var_sharpe", "skew", "kurtosis", "pbo", "n_obs"} <= set(d)


def test_01_s6_bootstrap_and_equity_comparison(ran01):
    s6 = ran01.receipt("S6")
    samples = art(ran01, "S6", "bootstrap_samples")
    assert list(samples.columns) == ["sharpe", "cagr", "max_drawdown"] and len(samples) == s6.metrics["bootstrap_reps"]
    summ = art(ran01, "S6", "bootstrap_summary")
    sh = summ["metrics"]["sharpe"]
    assert sh["ci"] == [pytest.approx(s6.metrics["sharpe_ci_low"]), pytest.approx(s6.metrics["sharpe_ci_high"])]
    assert sh["point"] == pytest.approx(s6.metrics["oos_sharpe"])
    assert [p["q"] for p in sh["percentiles"]] == [1, 5, 25, 50, 75, 95, 99]
    eq = art(ran01, "S6", "equity_compare")
    assert list(eq.columns) == ["base", "stressed"] and eq["stressed"].iloc[-1] < eq["base"].iloc[-1]


def test_04_failing_s5_still_persists_its_evidence(example):
    ex = example("04_overfit_grid", "prices")
    ex.run()
    assert ex.receipt("S5").verdict is Verdict.FAIL
    assert len(art(ex, "S5", "pbo_logits")) > 0 and art(ex, "S5", "dsr_inputs")["dsr"] < 0.95
    assert "S6" not in {p.stem for p in (ex.home / "receipts" / ex.trial_id).glob("*.json")}


def test_artifact_files_live_under_the_trial_stage_directory(ran01):
    for stage, names in EXPECTED.items():
        for name in names:
            rel = ran01.receipt(stage).params["artifacts"][name]["path"]
            assert rel.startswith(f"trials/{ran01.trial_id}/") and (ran01.home / rel).is_file()

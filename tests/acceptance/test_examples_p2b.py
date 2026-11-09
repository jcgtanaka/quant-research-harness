import pandas as pd
import pytest

from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Stage, Verdict
from qrh.pipeline.stages import Pipeline
from qrh.validation.defaults import DEFAULT_THRESHOLDS
from qrh.validation.pbo_cscv import pbo_cscv


def text(r):
    return " | ".join(r.reasons)


def edit(ex, filename, old, new):
    p = ex.dir / filename
    src = p.read_text()
    assert old in src, f"{old!r} not in {filename}"
    p.write_text(src.replace(old, new))


def registry(home):
    return Registry(home / "registry.sqlite3")


# --- example 01: the clean strategy passes the whole pipeline -------------------------------


def test_01_passes_s0_to_s7_with_default_thresholds(example):
    ex = example("01_clean_strategy", "prices")
    res = ex.run()
    for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        assert ex.receipt(s).verdict is Verdict.PASS, f"{s}: {text(ex.receipt(s))}"
    assert res.exit_code == 0
    s0 = ex.receipt("S0")
    for k, v in DEFAULT_THRESHOLDS.items():  # every default threshold is stored explicitly
        assert s0.params["thresholds"][k] == v
    s3, s4, s5, s6 = (ex.receipt(s) for s in ("S3", "S4", "S5", "S6"))
    assert s3.metrics["grid_size"] == s3.metrics["evaluated_count"] == 9
    assert s4.metrics["oos_sharpe"] >= 0 and s4.metrics["n_folds"] >= 2
    assert s5.metrics["pbo"] <= 0.2 and s5.metrics["dsr"] >= 0.95
    assert s5.trial_family_count >= 9
    assert s6.metrics["sharpe_ci_low"] >= 0 and s6.metrics["cost_stressed_sharpe"] >= 0
    assert s6.metrics["cost_stressed_sharpe"] < s4.metrics["oos_sharpe"]  # doubling costs must hurt
    assert s6.environment["seeds"]["bootstrap"] == s6.params["bootstrap_seed"]


def test_01_logs_every_combo_and_stores_the_matrix_with_its_hash(example, home):
    from qrh.hashing import sha256_file

    ex = example("01_clean_strategy", "prices")
    ex.run()
    s3 = ex.receipt("S3")
    path = home / s3.params["is_returns_path"]
    assert sha256_file(path) == s3.params["is_returns_hash"]
    assert path == home / "trials" / ex.trial_id / "is_returns.parquet"
    m = pd.read_parquet(path)
    assert m.shape[1] == 9 and m.index.is_monotonic_increasing and m.index.tz is not None
    reg = registry(home)
    assert reg.trial_count(ex.trial_id) == 9
    rows = reg.conn.execute("SELECT metrics, is_oos FROM trials WHERE trial_id=?", (ex.trial_id,)).fetchall()
    assert len(rows) == 9 and all("sharpe_annualized" in r["metrics"] and r["is_oos"] == 0 for r in rows)
    reg.close()


def test_01_report_written_and_linked_to_receipts(example, home):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    s7 = ex.receipt("S7")
    path = home / "reports" / f"{ex.trial_id}.md"
    assert s7.params["report_path"] == str(path) and path.is_file()
    report = path.read_text()
    assert "Final verdict: PASS" in report and "## Thresholds" in report
    assert "max_pbo: 0.2" in report and "A fast/slow moving-average spread" in report
    for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
        r = ex.receipt(s)
        assert f"`{r.receipt_hash()}`" in report
        assert f"`{home / 'receipts' / ex.trial_id / (s + '.json')}`" in report
    assert "pbo:" in report and "dsr:" in report and "sharpe_ci_low:" in report


def test_01_is_deterministic_across_reruns(example):
    # Receipt hashes chain to S2, whose cache flag changes on a rerun; the stage contents must not.
    ex = example("01_clean_strategy", "prices")
    ex.run()
    stages = ("S3", "S4", "S5", "S6")
    first = {s: (ex.receipt(s).metrics, ex.receipt(s).params, ex.receipt(s).reasons) for s in stages}
    ex.run()
    assert {s: (ex.receipt(s).metrics, ex.receipt(s).params, ex.receipt(s).reasons) for s in stages} == first


def test_01_second_trial_of_the_family_counts_all_trials_in_dsr(example, home):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    first = ex.receipt("S5")
    ex.new_trial()  # same family again
    ex.run()
    second = ex.receipt("S5")
    assert second.trial_family_count == 18 > first.trial_family_count == 9
    assert second.metrics["sr0"] > first.metrics["sr0"]
    assert second.metrics["dsr"] < first.metrics["dsr"]


def test_01_spec_periods_per_year_overrides_inference(example):
    ex = example("01_clean_strategy", "prices")
    edit(ex, "hypothesis.yaml", "walk_forward:", "periods_per_year: 252\nwalk_forward:")
    ex.new_trial()
    ex.run()
    s3 = ex.receipt("S3")
    assert s3.params["periods_per_year"] == 252 and s3.params["periods_per_year_source"] == "hypothesis"
    assert ex.receipt("S7").verdict is Verdict.PASS


def test_01_inferred_periods_per_year_is_recorded(example):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    s3 = ex.receipt("S3")
    assert s3.params["periods_per_year"] == pytest.approx(365.25)
    assert s3.params["periods_per_year_source"] == "observed timestamp density"


def test_ppy_that_cannot_be_inferred_blocks(example, monkeypatch):
    import dataclasses

    from qrh.methods.builtin import BUILTIN

    key = ("metrics.periods_per_year", "observed_density@1")
    monkeypatch.setitem(BUILTIN._methods, key, dataclasses.replace(BUILTIN._methods[key], entry=lambda _: None))
    ex = example("01_clean_strategy", "prices")
    ex.run()
    s3 = ex.receipt("S3")
    assert s3.verdict is Verdict.BLOCKED and "periods_per_year" in text(s3)
    assert ex.receipt("S7").params["failing_stage"] == "S3"


# --- blocked and failing conditions -------------------------------------------------------


def test_grid_size_mismatch_blocks_s3(example):
    ex = example("01_clean_strategy", "prices")
    edit(ex, "hypothesis.yaml", "parameter_space_size: 9", "parameter_space_size: 8")
    ex.new_trial()
    ex.run()
    s3 = ex.receipt("S3")
    assert s3.verdict is Verdict.BLOCKED and "9 combinations" in text(s3) and "declared 8" in text(s3)


def test_adapter_exception_in_one_combo_blocks_s3(example):
    ex = example("01_clean_strategy", "prices")
    edit(ex, "adapter.py", "        px = data.as_of(end)", (
        "        if params == {'fast': 30, 'slow': 240}:\n            raise RuntimeError('boom')\n"
        "        px = data.as_of(end)"))
    ex.new_trial()
    ex.run()
    s3 = ex.receipt("S3")
    assert s3.verdict is Verdict.BLOCKED and "boom" in text(s3)


def test_missing_walk_forward_blocks_s4(example):
    ex = example("01_clean_strategy", "prices")
    edit(ex, "hypothesis.yaml", "walk_forward: {train: 400, test: 100}\n", "")
    ex.new_trial()
    ex.run()
    assert ex.receipt("S3").verdict is Verdict.PASS
    s4 = ex.receipt("S4")
    assert s4.verdict is Verdict.BLOCKED and "walk_forward" in text(s4)


def test_invalid_walk_forward_blocks_s4(example):
    ex = example("01_clean_strategy", "prices")
    edit(ex, "hypothesis.yaml", "{train: 400, test: 100}", "{train: 400, test: 100, step: 50}")
    ex.new_trial()
    ex.run()
    s4 = ex.receipt("S4")
    assert s4.verdict is Verdict.BLOCKED and "step" in text(s4)


def test_oos_sharpe_below_threshold_fails_s4_and_report_shows_it(example, home):
    ex = example("01_clean_strategy", "prices")
    edit(ex, "hypothesis.yaml", "thresholds:\n", "thresholds:\n  min_oos_sharpe: 99\n")
    ex.new_trial()
    res = ex.run()
    s4 = ex.receipt("S4")
    assert s4.verdict is Verdict.FAIL and "min_oos_sharpe" in text(s4)
    assert res.exit_code == 2
    s7 = ex.receipt("S7")
    assert s7.params["failing_stage"] == "S4" and s7.verdict is Verdict.FAIL
    report = (home / "reports" / f"{ex.trial_id}.md").read_text()
    assert "Final verdict: FAIL" in report and "min_oos_sharpe" in report
    assert "### S5 Overfitting statistics\n\nVerdict: NOT RUN" in report


def test_cost_stress_threshold_can_fail_s6(example):
    ex = example("01_clean_strategy", "prices")
    edit(ex, "hypothesis.yaml", "thresholds:\n", "thresholds:\n  min_cost_stressed_sharpe: 99\n")
    ex.new_trial()
    ex.run()
    assert ex.receipt("S5").verdict is Verdict.PASS
    s6 = ex.receipt("S6")
    assert s6.verdict is Verdict.FAIL and "cost-stressed" in text(s6)


def test_adapter_without_cost_multiplier_blocks_s6(example):
    ex = example("01_clean_strategy", "prices")
    edit(ex, "adapter.py", "        cost_multiplier: float = 1.0,\n", "")
    edit(ex, "adapter.py", " * cost_multiplier", "")
    ex.new_trial()
    ex.run()
    assert ex.receipt("S5").verdict is Verdict.PASS
    s6 = ex.receipt("S6")
    assert s6.verdict is Verdict.BLOCKED and "adapter does not support cost stress" in text(s6)
    assert ex.receipt("S7").verdict is Verdict.BLOCKED


def test_missing_threshold_in_stored_hypothesis_blocks(example, home):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    # A trial stored by an older version without a threshold must block, never fall back silently.
    reg = registry(home)
    import json
    raw = reg.conn.execute("SELECT thresholds FROM hypotheses WHERE trial_id=?", (ex.trial_id,)).fetchone()[0]
    t = json.loads(raw)
    del t["max_pbo"]
    reg.conn.execute("DROP TRIGGER hypotheses_immutable")
    reg.conn.execute("UPDATE hypotheses SET thresholds=? WHERE trial_id=?", (json.dumps(t), ex.trial_id))
    s5 = Pipeline(reg, home).run_stage(ex.trial_id, Stage.S5)
    reg.close()
    assert s5.verdict is Verdict.BLOCKED and "max_pbo" in text(s5)


def test_tampered_matrix_blocks_s4(example, home):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    path = home / ex.receipt("S3").params["is_returns_path"]
    m = pd.read_parquet(path)
    m.iloc[0, 0] += 0.5
    m.to_parquet(path)
    reg = registry(home)
    s4 = Pipeline(reg, home).run_stage(ex.trial_id, Stage.S4)
    reg.close()
    assert s4.verdict is Verdict.BLOCKED and "does not match the hash" in text(s4)


# --- example 04: a big grid on noise is rejected -------------------------------------------


def test_04_overfit_grid_fails_at_s5(example, home):
    ex = example("04_overfit_grid", "prices")
    res = ex.run()
    assert res.exit_code == 2
    s7 = ex.receipt("S7")
    assert s7.verdict is Verdict.FAIL and s7.params["failing_stage"] in ("S4", "S5")
    for s in ("S1", "S2", "S3"):
        assert ex.receipt(s).verdict is Verdict.PASS, f"{s}: {text(ex.receipt(s))}"
    assert ex.receipt("S3").metrics["grid_size"] == 120
    s5 = ex.receipt("S5")
    assert s5.verdict is Verdict.FAIL and s5.trial_family_count >= 120
    assert s5.metrics["pbo"] > s5.metrics["max_pbo"] and s5.metrics["dsr"] < s5.metrics["min_dsr"]
    assert not (home / "receipts" / ex.trial_id / "S6.json").exists()  # nothing runs after a FAIL


def test_04_s5_alone_on_the_s3_matrix_fails(example, home):
    ex = example("04_overfit_grid", "prices")
    ex.run()
    s3 = ex.receipt("S3")
    matrix = pd.read_parquet(home / s3.params["is_returns_path"])
    assert matrix.shape[1] == 120
    pbo = pbo_cscv(matrix, n_blocks=16)
    assert pbo.pbo > 0.2  # the overfitting statistic alone rejects the search
    assert ex.receipt("S5").metrics["pbo"] == pytest.approx(pbo.pbo)


def test_01_receipts_record_the_hash_of_the_adapter_source_that_ran(example):
    from qrh.hashing import code_hash

    ex = example("01_clean_strategy", "prices")
    ex.run()
    expected = code_hash([ex.dir / "adapter.py"])
    assert ex.receipt("S2").params["adapter_code_hash"] == expected
    assert ex.receipt("S3").params["adapter_code_hash"] == expected

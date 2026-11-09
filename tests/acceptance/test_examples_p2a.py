from qrh.checks.lookahead import blocking, lint_file
from qrh.pipeline.receipt import Verdict


def text(receipt):
    return " | ".join(receipt.reasons)


def test_01_clean_strategy_passes_s1_and_s2(example):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    assert ex.receipt("S1").verdict is Verdict.PASS, text(ex.receipt("S1"))
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.PASS, text(s2)
    assert s2.data_hashes and s2.params["truncation"]["verdict"] == "PASS"
    assert s2.params["lint"]["verdict"] == "PASS"


def test_01_truncation_result_is_cached_per_code_version(example):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    assert ex.receipt("S2").diagnostics["truncation_cached"] is False
    ex.run()
    assert ex.receipt("S2").diagnostics["truncation_cached"] is True


def test_02_lookahead_lint_blocks_and_truncation_fails(example):
    ex = example("02_lookahead", "prices")
    ex.run()
    assert ex.receipt("S1").verdict is Verdict.PASS
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL
    assert any(r.startswith("lint LA001") for r in s2.reasons), text(s2)
    assert any(r.startswith("truncation:") and "first differing timestamp" in r for r in s2.reasons), text(s2)
    assert s2.params["truncation"]["verdict"] == "FAIL"
    assert blocking(lint_file(ex.dir / "adapter.py"))


def test_02_truncation_still_fails_when_lint_is_suppressed(example):
    ex = example("02_lookahead", "prices")
    src = (ex.dir / "adapter.py").read_text()
    assert "next_ret = ret.shift(-1)" in src
    (ex.dir / "adapter.py").write_text(
        src.replace("next_ret = ret.shift(-1)", "next_ret = ret.shift(-1)  # qrh: allow LA001 deliberately wrong for the test")
    )
    ex.new_trial()
    ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL
    assert s2.params["lint"]["verdict"] == "PASS"
    assert s2.params["lint"]["suppressed"][0]["reason"] == "deliberately wrong for the test"
    assert s2.params["truncation"]["verdict"] == "FAIL"
    assert not any(r.startswith("lint") for r in s2.reasons)


def test_03_units_bug_fails_sanity_with_percent_hint(example):
    ex = example("03_units_bug", "prices")
    ex.run()
    assert ex.receipt("S1").verdict is Verdict.PASS
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL
    assert any("max_abs_period_return" in r for r in s2.reasons), text(s2)
    assert any("returns may be in percent, expected fractions" in r for r in s2.reasons), text(s2)
    assert s2.params["truncation"]["verdict"] == "PASS" and s2.params["lint"]["verdict"] == "PASS"


def test_05_data_corruption_blocks_s1(example):
    ex = example("05_data_corruption", "linked")
    res = ex.run()
    s1 = ex.receipt("S1")
    assert s1.verdict is Verdict.BLOCKED
    assert any("derived" in r and "reference" in r for r in s1.reasons), text(s1)
    assert any("2020-07-" in r for r in s1.reasons), text(s1)  # row 200 onward
    assert ex.receipt("S7").params["failing_stage"] == "S1"
    assert res.exit_code == 2


def test_05_clean_variant_passes_s1(example):
    ex = example("05_data_corruption", "linked", corrupt=False)
    ex.run()
    assert ex.receipt("S1").verdict is Verdict.PASS, text(ex.receipt("S1"))


def test_changed_file_after_registration_blocks_s1(example):
    ex = example("01_clean_strategy", "prices")
    data = ex.dir / "data" / "prices.csv"
    data.write_text(data.read_text() + "2099-01-01 00:00:00+00:00,100.0\n")
    ex.run()
    s1 = ex.receipt("S1")
    assert s1.verdict is Verdict.BLOCKED and "hash mismatch" in text(s1)


def test_dataset_without_rules_is_blocked_at_s1(fresh_example):
    from typer.testing import CliRunner

    from qrh.cli import app

    ex = fresh_example("01_clean_strategy")
    data = ex.build_data()
    assert CliRunner().invoke(app, ["data", "register", str(data), "--name", "prices"]).exit_code == 0
    ex.new_trial()
    ex.run()
    s1 = ex.receipt("S1")
    assert s1.verdict is Verdict.BLOCKED and "no quality rules declared" in text(s1)


def test_unimportable_adapter_is_blocked_at_s2(example):
    ex = example("01_clean_strategy", "prices")
    spec = (ex.dir / "hypothesis.yaml").read_text().replace("adapter:TrendAdapter", "adapter:NoSuchClass")
    (ex.dir / "hypothesis.yaml").write_text(spec)
    ex.new_trial()
    ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.BLOCKED and "NoSuchClass" in text(s2)


def test_slow_dataset_needs_publication_column(example):
    ex = example("01_clean_strategy", "prices")
    spec = (ex.dir / "hypothesis.yaml").read_text().replace(
        "datasets: [prices]", "datasets:\n  - {name: prices, slow: true}"
    )
    (ex.dir / "hypothesis.yaml").write_text(spec)
    ex.new_trial()
    ex.run()
    s1 = ex.receipt("S1")
    assert s1.verdict is Verdict.BLOCKED and "publication column" in text(s1)


def test_unregistered_dataset_is_blocked_at_s1(example):
    ex = example("01_clean_strategy", "other_name")
    ex.run()
    s1 = ex.receipt("S1")
    assert s1.verdict is Verdict.BLOCKED and "unknown dataset" in text(s1)


def test_s0_records_bounds_and_windows(example):
    ex = example("01_clean_strategy", "prices")
    s0 = ex.receipt("S0")
    assert s0.params["bounds"]["max_gross_leverage"] == 1.0
    assert s0.params["windows"]["development"] == ["2020-03-01", "2023-12-31"]
    assert s0.params["adapter"] == "adapter:TrendAdapter"

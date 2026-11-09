"""s2.pnl_reconstruction in S2: the examples declare `execution` and select the experimental method."""

import yaml

from qrh.pipeline.receipt import Verdict

PNL = "s2.pnl_reconstruction"
VER = "pnl_reconstruction@1"


def text(r):
    return " | ".join(r.reasons)


def test_01_still_passes_with_the_reconstruction_recorded_and_flagged(example):
    ex = example("01_clean_strategy", "prices")
    assert ex.run().exit_code == 0
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.PASS and s2.methods[PNL] == VER
    assert s2.params["pnl_reconstruction"]["verdict"] == "PASS" and s2.params["pnl_reconstruction"]["match_lagged"] >= 0.98
    assert s2.params["non_default_methods"] == {PNL: VER}
    assert ex.receipt("S0").params["execution"] == {"lag_bars": 1, "price_column": "close"}
    assert "pnl" in s2.params["artifacts"]
    for stage in ("S1", "S3", "S4", "S5", "S6"):
        assert PNL not in ex.receipt(stage).methods
    assert ex.receipt("S7").methods[PNL] == VER


def test_02_peeking_still_fails_through_lint_and_truncation_and_reconstruction_does_not_object(example):
    ex = example("02_lookahead", "prices")
    ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL and "lint" in text(s2) and "truncation" in text(s2)
    assert s2.params["pnl_reconstruction"]["verdict"] == "PASS"  # the execution lag is respected; the signal peeks


def test_03_units_bug_now_also_fails_through_the_reconstruction(example):
    ex = example("03_units_bug", "prices")
    ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL
    assert "sanity:" in text(s2) and "pnl_reconstruction: units mismatch" in text(s2)
    assert s2.params["pnl_reconstruction"]["verdict"] == "FAIL" and s2.params["pnl_reconstruction"]["scale_exponent"] == 2


def test_selected_without_execution_is_blocked(fresh_example):
    ex = fresh_example("01_clean_strategy")
    ex.register(ex.build_data(), "prices")
    p = ex.dir / "hypothesis.yaml"
    spec = yaml.safe_load(p.read_text())
    del spec["execution"]
    p.write_text(yaml.safe_dump(spec))
    ex.new_trial()
    ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.BLOCKED and "execution" in text(s2) and "lag_bars" in text(s2)


def test_not_selected_means_the_check_does_not_run(fresh_example):
    ex = fresh_example("01_clean_strategy")
    ex.register(ex.build_data(), "prices")
    p = ex.dir / "hypothesis.yaml"
    spec = yaml.safe_load(p.read_text())
    del spec["methods"]
    p.write_text(yaml.safe_dump(spec))
    ex.new_trial()
    ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.PASS and PNL not in s2.methods and "pnl_reconstruction" not in s2.params


def test_a_same_bar_adapter_fails_s2(fresh_example):
    ex = fresh_example("01_clean_strategy")
    ex.register(ex.build_data(), "prices")
    a = ex.dir / "adapter.py"
    src = a.read_text()
    assert "(position.shift(1) * ret - cost.shift(1))" in src
    a.write_text(src.replace("(position.shift(1) * ret - cost.shift(1))", "(position * ret - cost)"))
    ex.new_trial()
    ex.run()
    s2 = ex.receipt("S2")
    assert s2.verdict is Verdict.FAIL and "same-bar execution (look-ahead)" in text(s2)

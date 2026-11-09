"""Methods in the pipeline: every receipt names the versions used, pins are frozen, runs are reproducible."""

import re

import pytest

from qrh.methods.state import save_state
from qrh.pipeline.receipt import Verdict

STAGE_METHODS = {
    "S1": {"s1.quality.schema", "s1.quality.range", "s1.quality.gaps", "s1.quality.timezone", "s1.quality.null_rate_drift"},
    # example 01 selects the reconstruction, which is why the default same-bar check does not run there
    "S2": {"s2.lookahead_lint", "s2.truncation", "s2.future_perturbation", "s2.sanity", "s2.pnl_reconstruction"},
    "S3": {"metrics.periods_per_year"},
    "S4": {"s4.walkforward", "metrics.periods_per_year"},
    "S5": {"s5.pbo", "s5.dsr", "s5.param_stability", "metrics.periods_per_year"},
    "S6": {"s6.bootstrap", "s4.walkforward", "metrics.periods_per_year"},
}


PNL, PNL_VER = "s2.pnl_reconstruction", "pnl_reconstruction@1"


def pinned_trial(fresh_example, methods: dict[str, str], ex=None):
    """A trial of example 01 whose hypothesis pins ``methods`` on top of the example's own pins."""
    import yaml

    if ex is None:
        ex = fresh_example("01_clean_strategy")
        ex.register(ex.build_data(), "prices")
    p = ex.dir / "hypothesis.yaml"
    spec = yaml.safe_load(p.read_text())
    spec["methods"] = {**spec.get("methods", {}), **methods}
    p.write_text(yaml.safe_dump(spec))
    ex.new_trial()
    return ex


def test_every_stage_receipt_records_the_methods_it_used(shared_01):
    ex = shared_01
    for stage, ids in STAGE_METHODS.items():
        r = ex.receipt(stage)
        assert set(r.methods) == ids, stage
        # only the experimental, opt-in reconstruction is non-default here, and only where it was used
        assert r.params.get("non_default_methods") == ({PNL: PNL_VER} if stage == "S2" else None), stage
    assert ex.receipt("S5").methods["s5.pbo"] == "cscv@1"
    assert ex.receipt("S4").methods["s4.walkforward"] == "rolling_anchored@1"
    s0 = ex.receipt("S0")
    assert s0.methods["s4.walkforward"] == "rolling_anchored@1" and "s1.quality.gaps" in s0.methods
    s7 = ex.receipt("S7")
    assert s7.methods["s5.pbo"] == "cscv@1" and set(s7.methods) >= set().union(*STAGE_METHODS.values())


def test_markdown_report_lists_methods_without_flag_for_defaults(shared_01):
    text = (shared_01.home / "reports" / f"{shared_01.trial_id}.md").read_text()
    assert "## Methods" in text and "s5.pbo=cscv@1" in text
    flagged = [line for line in text.splitlines() if "EXPERIMENTAL" in line]
    assert len(flagged) == 1 and PNL in flagged[0]  # only the opt-in reconstruction


def test_experimental_method_runs_but_is_flagged_everywhere(fresh_example, home):
    ex = pinned_trial(fresh_example, {"s4.walkforward": "purged_embargo@1"})
    ex.run()
    for stage in ("S0", "S4", "S6"):
        r = ex.receipt(stage)
        assert r.methods["s4.walkforward"] == "purged_embargo@1", stage
        assert r.params["non_default_methods"]["s4.walkforward"] == "purged_embargo@1", stage
    assert "non_default_methods" not in ex.receipt("S5").params
    s4 = ex.receipt("S4")
    assert s4.verdict in (Verdict.PASS, Verdict.FAIL)
    s7 = ex.receipt("S7")
    assert s7.params["non_default_methods"] == {"s4.walkforward": "purged_embargo@1", PNL: PNL_VER}
    text = (home / "reports" / f"{ex.trial_id}.md").read_text()
    assert re.search(r"s4\.walkforward=purged_embargo@1.*EXPERIMENTAL", text)


def test_rerunning_a_receipts_recorded_versions_reproduces_the_metrics(example, fresh_example):
    a = example("01_clean_strategy", "prices")
    a.run()
    a_id = a.trial_id
    recorded = {}
    for stage in ("S4", "S5", "S6"):
        recorded.update(a.receipt(stage).methods)
    b = pinned_trial(fresh_example, recorded, ex=a)  # same data and home, new trial pinning a's versions
    assert b.trial_id != a_id
    b.run()
    for stage in ("S3", "S4", "S5", "S6"):
        ra, rb = load(a_id, stage, a.home), b.receipt(stage)
        assert rb.methods == ra.methods, stage
        # S5 counts the family's logged trials (a second trial of the family raises N), so DSR may differ there.
        skip = {"dsr", "sr0", "psr_zero"} if stage == "S5" else set()
        assert {k: v for k, v in rb.metrics.items() if k not in skip} == {k: v for k, v in ra.metrics.items() if k not in skip}, stage


def load(trial_id, stage, home):
    from qrh.pipeline.receipt import load_receipt

    return load_receipt(home / "receipts" / trial_id / f"{stage}.json")


def test_rerunning_the_same_trial_reproduces_every_metric_exactly(example):
    ex = example("01_clean_strategy", "prices")
    ex.run()
    first = {s: ex.receipt(s) for s in ("S3", "S4", "S5", "S6")}
    ex.run()
    for stage, before in first.items():
        after = ex.receipt(stage)
        assert after.metrics == before.metrics and after.methods == before.methods, stage


def test_promotion_after_s0_does_not_change_a_registered_trial(example, home):
    ex = example("01_clean_strategy", "prices")  # S0 done: methods frozen
    save_state(home, {"schema_version": 1, "history": [], "statuses": {
        "s4.walkforward": {"purged_embargo@1": "default", "rolling_anchored@1": "deprecated"}}})
    ex.run()
    assert ex.receipt("S4").methods["s4.walkforward"] == "rolling_anchored@1"
    assert ex.receipt("S4").params["non_default_methods"] == {"s4.walkforward": "rolling_anchored@1"}


def test_unknown_pinned_method_is_refused_at_registration(fresh_example):
    from typer.testing import CliRunner

    from qrh.cli import app

    ex = fresh_example("01_clean_strategy")
    ex.register(ex.build_data(), "prices")
    p = ex.dir / "hypothesis.yaml"
    p.write_text(p.read_text() + "methods:\n  s5.pbo: cscv@9\n")
    res = CliRunner().invoke(app, ["new", str(p)])
    assert res.exit_code == 2 and "cscv@9" in res.output

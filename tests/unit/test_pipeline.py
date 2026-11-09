import itertools

import pytest

from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Stage, Verdict, load_receipt
from qrh.pipeline.stages import (
    Pipeline,
    StageContext,
    default_stages,
    make_receipt,
    register_new_trial,
)

SPEC = {
    "text": "h",
    "strategy_family": "s",
    "asset_class": "a",
    "frequency": "f",
    "parameter_space_size": 4,
    "thresholds": {"min": 1},
    "difference_statement": "differs by x",
}


@pytest.fixture
def env(tmp_path):
    counter = itertools.count(1)
    reg = Registry(tmp_path / "r.sqlite3", id_factory=lambda: f"t{next(counter)}")
    yield reg, tmp_path
    reg.close()


class Recording:
    """A stage that records calls and returns a fixed verdict."""

    def __init__(self, stage, verdict=Verdict.PASS):
        self.id = stage
        self.verdict = verdict
        self.calls = 0

    def run(self, ctx: StageContext):
        self.calls += 1
        return make_receipt(ctx, self.id, self.verdict, ["fixed"])


class Exploding:
    def __init__(self, stage):
        self.id = stage

    def run(self, ctx):
        raise RuntimeError("boom")


def stages_with(**overrides):
    out = []
    for s in default_stages():
        out.append(overrides.get(s.id.value, s))
    return out


def test_new_trial_registers_and_writes_s0(env):
    reg, home = env
    res = register_new_trial(SPEC, reg, home)
    assert res.reasons == [] and res.trial_id == "t1"
    r = load_receipt(res.path)
    assert r.stage is Stage.S0 and r.verdict is Verdict.PASS and r.parent_receipt_hash is None
    assert r.hypothesis_hash == reg.get_trial("t1")["hypothesis_hash"]
    assert reg.get_receipt_entry("t1", "S0")["verdict"] == "PASS"


@pytest.mark.parametrize("patch", [{"difference_statement": ""}, {"thresholds": {}}])
def test_invalid_spec_registers_nothing(env, patch):
    reg, home = env
    res = register_new_trial({**SPEC, **patch}, reg, home)
    assert res.trial_id is None and res.reasons
    assert reg.list_trials() == []
    assert not (home / "receipts").exists()


def test_s0_receipt_deterministic_on_rerun(env):
    reg, home = env
    res = register_new_trial(SPEC, reg, home)
    first = load_receipt(res.path).receipt_hash()
    again = Pipeline(reg, home).run_stage("t1", Stage.S0)
    assert again.receipt_hash() == first


def test_stage_refused_without_previous_pass(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    s2 = Recording(Stage.S2)
    pipe = Pipeline(reg, home, stages=stages_with(S2=s2))
    r = pipe.run_stage("t1", Stage.S2)
    assert r.verdict is Verdict.BLOCKED and s2.calls == 0
    assert any("S1" in x for x in r.reasons)


def test_stage_refused_when_previous_not_pass(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    pipe = Pipeline(reg, home, stages=stages_with(S1=Recording(Stage.S1, Verdict.FAIL)))
    assert pipe.run_stage("t1", Stage.S1).verdict is Verdict.FAIL
    s2 = Recording(Stage.S2)
    pipe2 = Pipeline(reg, home, stages=stages_with(S2=s2))
    assert pipe2.run_stage("t1", Stage.S2).verdict is Verdict.BLOCKED and s2.calls == 0


def test_stage_refused_when_previous_receipt_tampered(env):
    reg, home = env
    res = register_new_trial(SPEC, reg, home)
    res.path.write_text("{}")
    s1 = Recording(Stage.S1)
    r = Pipeline(reg, home, stages=stages_with(S1=s1)).run_stage("t1", Stage.S1)
    assert r.verdict is Verdict.BLOCKED and s1.calls == 0


def test_exception_becomes_blocked_receipt(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    pipe = Pipeline(reg, home, stages=stages_with(S1=Exploding(Stage.S1)))
    r = pipe.run_stage("t1", Stage.S1)
    assert r.verdict is Verdict.BLOCKED
    assert any("RuntimeError" in x and "boom" in x for x in r.reasons)
    assert reg.get_receipt_entry("t1", "S1")["verdict"] == "BLOCKED"


def test_wrong_stage_receipt_is_blocked(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    # The S1 slot returns an S2-labelled receipt; the framework refuses it.
    bad = Recording(Stage.S1)
    bad.run = lambda ctx: make_receipt(ctx, Stage.S2, Verdict.PASS, [])
    pipe = Pipeline(reg, home, stages=stages_with(S1=bad))
    assert pipe.run_stage("t1", Stage.S1).verdict is Verdict.BLOCKED


def test_parent_hash_links_stages(env):
    reg, home = env
    res = register_new_trial(SPEC, reg, home)
    s0_hash = load_receipt(res.path).receipt_hash()
    r1 = Pipeline(reg, home, stages=stages_with(S1=Recording(Stage.S1))).run_stage("t1", Stage.S1)
    assert r1.parent_receipt_hash == s0_hash


def test_default_run_stops_blocked_at_s1_then_s7_summarizes(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    result = Pipeline(reg, home).run("t1")
    assert result.final.stage is Stage.S7 and result.final.verdict is Verdict.BLOCKED
    assert result.final.params["failing_stage"] == "S1"
    assert result.final.params["stage_verdicts"]["S0"] == "PASS"
    assert result.final.params["stage_verdicts"]["S1"] == "BLOCKED"
    assert "S2" not in result.final.params["stage_verdicts"]
    assert any("no datasets declared" in r for r in result.stage_receipts[Stage.S1].reasons)
    assert load_receipt(result.final_path).receipt_hash() == result.final.receipt_hash()
    assert reg.get_receipt_entry("t1", "S2") is None


def test_all_pass_gives_overall_pass(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    stages = [default_stages()[0]] + [Recording(Stage(f"S{i}")) for i in range(1, 7)]
    result = Pipeline(reg, home, stages=stages).run("t1")
    assert result.final.verdict is Verdict.PASS
    assert result.final.parent_receipt_hash == result.stage_receipts[Stage.S6].receipt_hash()


def test_fail_stops_and_overall_fail(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    s3 = Recording(Stage.S3, Verdict.FAIL)
    s4 = Recording(Stage.S4)
    stages = stages_with(S1=Recording(Stage.S1), S2=Recording(Stage.S2), S3=s3, S4=s4)
    result = Pipeline(reg, home, stages=stages).run("t1")
    assert result.final.verdict is Verdict.FAIL and s4.calls == 0
    assert result.final.params["failing_stage"] == "S3"


def test_exception_in_stage_runs_s7_and_blocks(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    stages = stages_with(S1=Recording(Stage.S1), S2=Exploding(Stage.S2))
    result = Pipeline(reg, home, stages=stages).run("t1")
    assert result.final.verdict is Verdict.BLOCKED
    assert result.final.params["failing_stage"] == "S2"


def test_rerun_clears_stale_later_receipts(env):
    reg, home = env
    register_new_trial(SPEC, reg, home)
    good = [default_stages()[0]] + [Recording(Stage(f"S{i}")) for i in range(1, 7)]
    assert Pipeline(reg, home, stages=good).run("t1").final.verdict is Verdict.PASS
    result = Pipeline(reg, home).run("t1")  # placeholders now
    assert result.final.verdict is Verdict.BLOCKED
    assert reg.get_receipt_entry("t1", "S2") is None


def test_stage_list_must_be_ordered():
    with pytest.raises(ValueError):
        Pipeline(None, ".", stages=list(reversed(default_stages())))  # type: ignore[arg-type]

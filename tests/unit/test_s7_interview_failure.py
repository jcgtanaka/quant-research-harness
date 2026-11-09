"""S7: a failing interview snapshot never changes the verdict, but the failure is recorded and visible."""

import itertools

import pytest
from typer.testing import CliRunner

from packs_helpers import make_pack, question
from qrh.cli import app
from qrh.memory.registry import Registry
from qrh.pipeline.receipt import Stage, Verdict
from qrh.pipeline.stages import Pipeline, RegisterHypothesis, make_receipt, register_new_trial

runner = CliRunner()
SPEC = {
    "text": "a moving average rule", "strategy_family": "trend", "asset_class": "equity", "frequency": "1D",
    "parameter_space_size": 4, "thresholds": {"k": 1}, "difference_statement": "baseline",
}


class Stub:
    def __init__(self, stage):
        self.id = stage

    def run(self, ctx):
        return make_receipt(ctx, self.id, Verdict.PASS, [])


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("QRH_HOME", str(home))
    counter = itertools.count(1)
    reg = Registry(home / "registry.sqlite3", id_factory=lambda: f"t{next(counter)}")
    d = make_pack(tmp_path, "bank", questions=[[question("hq", stage="S2", scope="hypothesis", critical=False)]], lessons=[])
    assert runner.invoke(app, ["pack", "add", str(d)]).exit_code == 0
    yield home, reg
    reg.close()


def run(env):
    home, reg = env
    tid = register_new_trial(SPEC, reg, home).trial_id
    stages = [RegisterHypothesis()] + [Stub(Stage(f"S{i}")) for i in range(1, 7)]
    return Pipeline(reg, home, stages=stages).run(tid).final


def test_clean_run_has_no_snapshot_error(env):
    final = run(env)
    assert "interview_snapshot_error" not in final.params
    assert not any("interview snapshot" in r for r in final.reasons)


def test_snapshot_failure_is_recorded_and_verdict_is_unchanged(env, monkeypatch):
    clean = run(env).verdict
    from qrh.interview import snapshot

    def boom(*a, **k):
        raise RuntimeError("snapshot exploded")

    monkeypatch.setattr(snapshot, "build_snapshot", boom)
    final = run(env)
    assert clean is Verdict.PASS
    assert final.verdict is Verdict.PASS_CONDITIONAL
    assert "snapshot exploded" in final.params["interview_snapshot_error"]
    assert any("interview snapshot" in r and "snapshot exploded" in r for r in final.reasons)


def test_snapshot_failure_turns_pass_into_conditional_with_named_condition(env, monkeypatch):
    from qrh.interview import snapshot

    def boom(*a, **k):
        raise RuntimeError("snapshot exploded")

    monkeypatch.setattr(snapshot, "build_snapshot", boom)
    final = run(env)
    assert final.verdict is Verdict.PASS_CONDITIONAL
    conds = final.params["conditions"]
    assert len(conds) == 1 and conds[0]["question_id"] == "interview_snapshot"
    assert "snapshot exploded" in conds[0]["text"]

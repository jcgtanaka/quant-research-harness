import itertools

import pytest

from qrh.hashing import hash_json
from qrh.memory.registry import HypothesisRecord, Registry
from qrh.methods.state import save_state
from qrh.pipeline.receipt import Receipt, Stage, Verdict, load_receipt, write_receipt
from qrh.pipeline.stages import Pipeline, StageContext, make_receipt, register_new_trial

SPEC = {
    "text": "h", "strategy_family": "s", "asset_class": "a", "frequency": "f", "parameter_space_size": 4,
    "thresholds": {"min": 1}, "difference_statement": "differs by x",
}
# Hashes computed before the `methods` field existed: old hypotheses must keep their hash.
OLD_PLAIN_HASH = "d4b426f56ff856be8136a4384714d45b2df2599b9eacc5b1913d4b21fbf7788a"
OLD_BOUND_HASH = "4743b8c02aad055ba65184c26181ff75ed32a6948fbd52470197bd7e867c6c66"


@pytest.fixture
def env(tmp_path):
    counter = itertools.count(1)
    reg = Registry(tmp_path / "r.sqlite3", id_factory=lambda: f"t{next(counter)}")
    yield reg, tmp_path
    reg.close()


def test_hypothesis_hash_is_unchanged_when_no_methods_are_pinned():
    assert HypothesisRecord(**SPEC).hypothesis_hash() == OLD_PLAIN_HASH
    bound = dict(SPEC, adapter="m:C", datasets=["prices"], windows={"development": ["2020-01-01", "2021-01-01"]},
                 bounds={"max_gross_leverage": 1.0}, walk_forward={"train": 10, "test": 5})
    assert HypothesisRecord(**bound).hypothesis_hash() == OLD_BOUND_HASH


def test_pinned_methods_are_part_of_the_hypothesis_hash():
    a = HypothesisRecord(**SPEC, methods={"s5.pbo": "cscv@1"})
    b = HypothesisRecord(**SPEC, methods={"s5.pbo": "cscv@2"})
    assert a.hypothesis_hash() != OLD_PLAIN_HASH and a.hypothesis_hash() != b.hypothesis_hash()
    assert a.binding()["methods"] == {"s5.pbo": "cscv@1"}


def test_methods_syntax_is_validated():
    for bad in ({"S5.PBO": "cscv@1"}, {"s5.pbo": "cscv"}, {"s5.pbo": "cscv@0"}):
        with pytest.raises(ValueError):
            HypothesisRecord(**SPEC, methods=bad)


def test_receipt_hash_is_unchanged_when_no_methods_are_recorded():
    r = Receipt(trial_id="t", stage=Stage.S0, hypothesis_hash="h", code_hash="c", verdict=Verdict.PASS)
    # Legacy form = the receipt as written before `methods`, `packs` and `diagnostics` existed.
    legacy = hash_json(r.model_dump(mode="json", exclude={"created_at", "methods", "packs", "diagnostics"}))
    assert r.receipt_hash() == legacy and r.methods == {} and r.packs == {}
    with_methods = r.model_copy(update={"methods": {"s5.pbo": "cscv@1"}})
    assert with_methods.receipt_hash() != legacy


def test_receipt_methods_round_trip_and_tamper_check(tmp_path):
    r = Receipt(trial_id="t", stage=Stage.S5, hypothesis_hash="h", code_hash="c", verdict=Verdict.PASS,
                methods={"s5.pbo": "cscv@1"})
    path = write_receipt(r, tmp_path)
    assert load_receipt(path).methods == {"s5.pbo": "cscv@1"}
    path.write_text(path.read_text().replace("cscv@1", "cscv@2"))
    with pytest.raises(ValueError):
        load_receipt(path)


def test_stage_context_records_used_methods_and_make_receipt_includes_them(env):
    reg, home = env
    ctx = StageContext(trial_id="t1", registry=reg, home=home)
    assert ctx.method("s5.pbo").version == "cscv@1"
    assert ctx.method("s4.walkforward").version == "rolling_anchored@1"
    r = make_receipt(ctx, Stage.S5, Verdict.PASS, [])
    assert r.methods == {"s4.walkforward": "rolling_anchored@1", "s5.pbo": "cscv@1"}
    assert "non_default_methods" not in r.params


def test_trial_pins_drive_selection_and_non_default_use_is_flagged(env):
    reg, home = env
    res = register_new_trial(dict(SPEC, methods={"s4.walkforward": "purged_embargo@1"}), reg, home)
    assert res.reasons == [] and res.trial_id
    ctx = StageContext(trial_id=res.trial_id, registry=reg, home=home)
    assert ctx.method("s4.walkforward").version == "purged_embargo@1"
    r = make_receipt(ctx, Stage.S4, Verdict.PASS, [], params={"x": 1})
    assert r.params["non_default_methods"] == {"s4.walkforward": "purged_embargo@1"} and r.params["x"] == 1


def test_s0_records_the_full_method_map_and_it_is_frozen(env):
    reg, home = env
    res = register_new_trial(dict(SPEC, methods={"s4.walkforward": "purged_embargo@1"}), reg, home)
    trial = reg.get_trial(res.trial_id)
    assert trial["methods"]["s4.walkforward"] == "purged_embargo@1" and trial["methods"]["s5.pbo"] == "cscv@1"
    assert res.receipt.methods == trial["methods"]
    assert res.receipt.params["non_default_methods"] == {"s4.walkforward": "purged_embargo@1"}
    # A later local promotion does not change what this trial uses.
    save_state(home, {"schema_version": 1, "history": [], "statuses": {
        "s4.walkforward": {"purged_embargo@1": "deprecated", "rolling_anchored@1": "default"}}})
    ctx = StageContext(trial_id=res.trial_id, registry=reg, home=home)
    assert ctx.method("s4.walkforward").version == "purged_embargo@1"


def test_s0_fills_defaults_at_registration_time(env):
    reg, home = env
    save_state(home, {"schema_version": 1, "history": [], "statuses": {
        "s4.walkforward": {"purged_embargo@1": "default", "rolling_anchored@1": "deprecated"}}})
    res = register_new_trial(SPEC, reg, home)
    assert reg.get_trial(res.trial_id)["methods"]["s4.walkforward"] == "purged_embargo@1"
    # not flagged: it is the default of this home
    assert "non_default_methods" not in res.receipt.params


def test_s0_rejects_unknown_and_deprecated_pins(env):
    reg, home = env
    res = register_new_trial(dict(SPEC, methods={"s5.pbo": "cscv@9"}), reg, home)
    assert res.trial_id is None and any("cscv@9" in r for r in res.reasons)  # nothing registered
    assert reg.list_trials() == []
    save_state(home, {"schema_version": 1, "history": [], "statuses": {
        "s4.walkforward": {"purged_embargo@1": "default", "rolling_anchored@1": "deprecated"}}})
    res = register_new_trial(dict(SPEC, methods={"s4.walkforward": "rolling_anchored@1"}), reg, home)
    assert any("deprecated" in r for r in res.reasons)


def test_legacy_trial_without_methods_uses_defaults(env):
    reg, home = env
    reg.register_hypothesis(HypothesisRecord(**SPEC), trial_id="old")
    assert reg.get_trial("old")["methods"] == {}
    ctx = StageContext(trial_id="old", registry=reg, home=home)
    assert ctx.method("s5.pbo").version == "cscv@1"


def test_a_broken_state_file_blocks_the_stage_instead_of_passing(env):
    reg, home = env
    res = register_new_trial(SPEC, reg, home)
    (home / "methods").mkdir()
    (home / "methods" / "state.json").write_text("{broken")

    class Uses:
        id = Stage.S1

        def run(self, ctx):
            ctx.method("s5.pbo")
            return make_receipt(ctx, self.id, Verdict.PASS, [])

    receipt = Pipeline(reg, home, stages=[_s0(), Uses()]).run_stage(res.trial_id, Stage.S1)
    assert receipt.verdict is Verdict.BLOCKED and "state" in " ".join(receipt.reasons)


def _s0():
    from qrh.pipeline.stages import RegisterHypothesis

    return RegisterHypothesis()

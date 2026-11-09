"""s2.pnl_reconstruction is experimental with no default: it runs only where a hypothesis selects it."""

import itertools

import pytest

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.methods.builtin import BUILTIN
from qrh.methods.registry import MethodError
from qrh.methods.selection import MethodSelector, check_pins
from qrh.pipeline.receipt import Stage, Verdict
from qrh.pipeline.stages import StageContext, make_receipt, register_new_trial

PNL = "s2.pnl_reconstruction"
VER = "pnl_reconstruction@1"
SPEC = {
    "text": "h", "strategy_family": "s", "asset_class": "a", "frequency": "f", "parameter_space_size": 4,
    "thresholds": {"min": 1}, "difference_statement": "differs by x",
}
EXEC = {"lag_bars": 1, "price_column": "close"}


@pytest.fixture
def env(tmp_path):
    counter = itertools.count(1)
    reg = Registry(tmp_path / "r.sqlite3", id_factory=lambda: f"t{next(counter)}")
    yield reg, tmp_path
    reg.close()


def test_registered_experimental_without_a_default():
    m = BUILTIN.get(PNL, VER)
    assert m.status == "experimental" and m.stage == "S2" and m.validate is not None
    with pytest.raises(MethodError):
        BUILTIN.default(PNL)


def test_opt_in_ids_are_left_out_of_the_full_map_unless_selected(tmp_path):
    assert PNL not in MethodSelector(tmp_path).full_map()
    assert MethodSelector(tmp_path, pinned={PNL: VER}).full_map()[PNL] == VER
    assert MethodSelector(tmp_path, overrides={PNL: VER}).full_map()[PNL] == VER


def test_selected_returns_none_unless_pinned_and_flags_the_use(tmp_path):
    assert MethodSelector(tmp_path).selected(PNL) is None
    sel = MethodSelector(tmp_path, pinned={PNL: VER})
    assert sel.selected(PNL).version == VER
    assert sel.used == {PNL: VER} and sel.non_default() == {PNL: VER}
    assert MethodSelector(tmp_path).selected("s5.pbo").version == "cscv@1"  # ids with a default are always selected


def test_pinning_it_is_allowed(tmp_path):
    assert check_pins(tmp_path, {PNL: VER}) == []


def test_execution_field_is_validated_hashed_and_absent_hashes_are_unchanged():
    base = HypothesisRecord(**SPEC)
    assert base.execution is None
    with_exec = HypothesisRecord(**SPEC, execution=EXEC)
    assert with_exec.hypothesis_hash() != base.hypothesis_hash()
    assert with_exec.binding()["execution"] == EXEC
    assert HypothesisRecord(**SPEC, execution={**EXEC, "lag_bars": 2}).hypothesis_hash() != with_exec.hypothesis_hash()
    for bad in ({"lag_bars": 0, "price_column": "close"}, {"price_column": "close"}, "x"):
        with pytest.raises(ValueError):
            HypothesisRecord(**SPEC, execution=bad)


def test_execution_is_stored_and_returned(env):
    reg, home = env
    res = register_new_trial({**SPEC, "execution": EXEC, "methods": {PNL: VER}}, reg, home)
    trial = reg.get_trial(res.trial_id)
    assert trial["execution"] == EXEC and trial["methods"][PNL] == VER
    assert res.receipt.params["execution"] == EXEC
    assert res.receipt.params["non_default_methods"] == {PNL: VER}
    plain = register_new_trial(SPEC, reg, home)
    assert reg.get_trial(plain.trial_id)["execution"] is None and PNL not in reg.get_trial(plain.trial_id)["methods"]


def test_stage_context_selected_method_is_recorded_in_the_receipt(env):
    reg, home = env
    res = register_new_trial({**SPEC, "execution": EXEC, "methods": {PNL: VER}}, reg, home)
    ctx = StageContext(trial_id=res.trial_id, registry=reg, home=home)
    assert ctx.selected_method(PNL).version == VER
    r = make_receipt(ctx, Stage.S2, Verdict.PASS, [])
    assert r.methods == {PNL: VER}
    other = register_new_trial(SPEC, reg, home)
    ctx2 = StageContext(trial_id=other.trial_id, registry=reg, home=home)
    assert ctx2.selected_method(PNL) is None and ctx2.methods_used == {}

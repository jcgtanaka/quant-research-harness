"""The four stability thresholds in the defaults and in pack layering (higher is tighter)."""

import pytest

from packs_helpers import make_pack
from qrh.packs.layering import direction, is_looser, layer_thresholds, loosenings
from qrh.packs.model import PackError, load_pack
from qrh.validation.defaults import DEFAULT_THRESHOLDS

TIGHTER_WHEN_HIGHER = ("stability_closeness", "stability_plateau_share", "stability_neighbour_ratio")


def pack(tmp_path, **thresholds):
    return load_pack(make_pack(tmp_path, "p", thresholds={"all": thresholds}))


@pytest.mark.parametrize("name", TIGHTER_WHEN_HIGHER)
def test_higher_is_tighter(name):
    assert direction(name) == "min"
    assert is_looser(name, DEFAULT_THRESHOLDS[name], DEFAULT_THRESHOLDS[name] - 0.1)
    assert not is_looser(name, DEFAULT_THRESHOLDS[name], DEFAULT_THRESHOLDS[name] + 0.1)


def test_the_steps_are_not_ordered():
    assert direction("stability_steps") is None
    assert not is_looser("stability_steps", [0.1, 0.2], [0.05])


@pytest.mark.parametrize("name", TIGHTER_WHEN_HIGHER)
def test_a_pack_may_tighten_freely(tmp_path, name):
    p = pack(tmp_path, **{name: 0.9})
    eff, info = layer_thresholds(DEFAULT_THRESHOLDS, [p], {}, {}, set())
    assert eff[name] == 0.9 and info["tightened"][name]["to"] == 0.9


@pytest.mark.parametrize("name", TIGHTER_WHEN_HIGHER)
def test_loosening_a_core_value_needs_a_human_approval(tmp_path, name):
    p = pack(tmp_path, **{name: 0.1})
    assert [lo.name for lo in loosenings([p], {})] == [name]
    with pytest.raises(PackError, match="human approval"):
        layer_thresholds(DEFAULT_THRESHOLDS, [p], {}, {}, set())


def test_a_hypothesis_value_looser_than_a_pack_is_recorded(tmp_path):
    p = pack(tmp_path, stability_closeness=0.9)
    eff, info = layer_thresholds(DEFAULT_THRESHOLDS, [p], {}, {"stability_closeness": 0.6}, set())
    assert eff["stability_closeness"] == 0.6 and info["hypothesis_looser_than_pack"] == ["stability_closeness"]


def test_layering_does_not_share_the_default_list(tmp_path):
    eff, _ = layer_thresholds(DEFAULT_THRESHOLDS, [], {}, {}, set())
    eff["stability_steps"].append(0.9)
    assert DEFAULT_THRESHOLDS["stability_steps"] == [0.10, 0.20]


# --- the optional `stability` block of the hypothesis ---------------------------------------------------------------

BASE = {
    "text": "h", "strategy_family": "s", "asset_class": "a", "frequency": "f",
    "parameter_space_size": 1, "thresholds": {"min": 1}, "difference_statement": "d",
}


def test_the_stability_block_is_validated_stored_and_hashed_only_when_present():
    from qrh.pipeline.gates import check_hypothesis_spec

    plain, _ = check_hypothesis_spec(BASE)
    block = {"neighbours": {"w": [10, 30]}, "bounds": {"w": [5, 60]}}
    rec, reasons = check_hypothesis_spec({**BASE, "stability": block})
    assert rec is not None and reasons == [] and rec.binding()["stability"] == block
    assert rec.hypothesis_hash() != plain.hypothesis_hash()
    assert "stability" not in plain.binding()  # older hypotheses keep their hash
    for bad in ({"neighbours": {"w": []}}, {"neighbours": {"w": ["x"]}}, {"neighbours": [1, 2]}, {"bounds": {"w": [9, 1]}},
                {"bounds": {"w": [1]}}, {"bounds": {"w": [True, 5]}}, {"other": 1}, "text"):
        assert check_hypothesis_spec({**BASE, "stability": bad})[0] is None, bad


def test_the_stability_block_round_trips_through_the_registry(tmp_path):
    from qrh.memory.registry import HypothesisRecord, Registry

    reg = Registry(tmp_path / "r.sqlite3")
    block = {"neighbours": {"w": [10, 30]}}
    tid = reg.register_hypothesis(HypothesisRecord.model_validate({**BASE, "stability": block}))
    assert reg.get_trial(tid)["stability"] == block
    other = reg.register_hypothesis(HypothesisRecord.model_validate(BASE))
    assert reg.get_trial(other)["stability"] is None
    reg.close()

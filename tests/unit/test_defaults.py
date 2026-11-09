import pytest

from qrh.pipeline.gates import check_hypothesis_spec
from qrh.validation.defaults import DEFAULT_THRESHOLDS, fill_thresholds

BASE = {
    "text": "h", "strategy_family": "s", "asset_class": "a", "frequency": "f",
    "parameter_space_size": 4, "thresholds": {"min": 1}, "difference_statement": "d",
}


def test_documented_default_values():
    assert DEFAULT_THRESHOLDS == {
        "max_pbo": 0.2, "min_dsr": 0.95, "min_oos_sharpe": 0.0, "min_sharpe_lower_ci": 0.0,
        "min_cost_stressed_sharpe": 0.0, "bootstrap_reps": 2000, "ci_level": 0.95,
        "cost_stress_multiplier": 2.0, "cscv_blocks": 16, "llm_cutoff_margin_days": 180,
        "perturbation_cutoffs": 20, "perturbation_change_cap": 40,
        # changed with the parameter stability check: four more thresholds, always asked of the trader
        "stability_closeness": 0.70, "stability_plateau_share": 0.50, "stability_neighbour_ratio": 0.50,
        "stability_steps": [0.10, 0.20],
    }


def test_fill_keeps_user_values_and_adds_missing():
    out = fill_thresholds({"max_pbo": 0.1, "custom": 5})
    assert out["max_pbo"] == 0.1 and out["custom"] == 5 and out["min_dsr"] == 0.95
    assert set(DEFAULT_THRESHOLDS) <= set(out)


def test_fill_does_not_mutate_input_or_defaults():
    src = {"max_pbo": 0.1}
    fill_thresholds(src)
    assert src == {"max_pbo": 0.1}
    fill_thresholds({})["max_pbo"] = 9
    assert DEFAULT_THRESHOLDS["max_pbo"] == 0.2
    fill_thresholds({})["stability_steps"].append(0.5)  # the list default is copied too
    assert DEFAULT_THRESHOLDS["stability_steps"] == [0.10, 0.20]


def test_fill_is_idempotent():
    once = fill_thresholds({"a": 1})
    assert fill_thresholds(once) == once


def test_new_binding_fields_validated_and_hashed():
    a, _ = check_hypothesis_spec(BASE)
    b, _ = check_hypothesis_spec({**BASE, "periods_per_year": 252})
    c, _ = check_hypothesis_spec({**BASE, "walk_forward": {"train": 100, "test": 20}})
    assert a.hypothesis_hash() != b.hypothesis_hash() != c.hypothesis_hash()
    assert check_hypothesis_spec({**BASE, "periods_per_year": 0})[0] is None
    assert check_hypothesis_spec({**BASE, "walk_forward": "x"})[0] is None


def test_s0_fills_missing_thresholds_and_stores_them(tmp_path):
    from qrh.memory.registry import Registry
    from qrh.pipeline.receipt import Verdict
    from qrh.pipeline.stages import register_new_trial

    reg = Registry(tmp_path / "r.sqlite3")
    res = register_new_trial({**BASE, "thresholds": {"max_pbo": 0.1, "min": 1}}, reg, tmp_path)
    assert res.receipt.verdict is Verdict.PASS
    stored = reg.get_trial(res.trial_id)["thresholds"]
    assert stored["max_pbo"] == 0.1 and stored["min"] == 1 and stored["min_dsr"] == 0.95
    assert set(DEFAULT_THRESHOLDS) <= set(stored)
    assert res.receipt.params["thresholds"] == stored
    # hash is deterministic and covers the filled thresholds
    again = register_new_trial({**BASE, "thresholds": {"max_pbo": 0.1, "min": 1}}, reg, tmp_path)
    assert reg.get_trial(again.trial_id)["hypothesis_hash"] == reg.get_trial(res.trial_id)["hypothesis_hash"]
    other = register_new_trial({**BASE, "thresholds": {"max_pbo": 0.3, "min": 1}}, reg, tmp_path)
    assert reg.get_trial(other.trial_id)["hypothesis_hash"] != reg.get_trial(res.trial_id)["hypothesis_hash"]
    reg.close()


def test_clear_trial_combos(tmp_path):
    from qrh.memory.registry import HypothesisRecord, Registry

    reg = Registry(tmp_path / "r.sqlite3")
    a = reg.register_hypothesis(HypothesisRecord.model_validate(BASE))
    b = reg.register_hypothesis(HypothesisRecord.model_validate(BASE))
    reg.record_trial(a, {"x": 1}, {"m": 1.0})
    reg.record_trial(b, {"x": 1}, {"m": 1.0})
    reg.record_trial(a, {"x": 2}, {"m": 1.0}, is_oos=True)
    reg.clear_trial_combos(a, is_oos=False)
    assert reg.trial_count(a) == 1 and reg.trial_count(b) == 1
    reg.close()

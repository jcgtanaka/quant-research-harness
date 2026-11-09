import pytest

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.gates import check_hypothesis_spec

BASE = {
    "text": "h", "strategy_family": "s", "asset_class": "a", "frequency": "f",
    "parameter_space_size": 4, "thresholds": {"min": 1}, "difference_statement": "d",
}
BOUND = {
    "adapter": "mod.sub:Klass",
    "datasets": ["px", {"name": "rel", "slow": True, "publication_column": "published_at"}],
    "windows": {"development": ["2020-01-01", "2020-06-30"]},
    "bounds": {"max_abs_period_return": 0.5, "max_gross_leverage": 1.0, "max_position_weight": 1.0},
    "s2_params": {"fast": 5},
}


def test_old_spec_hash_unchanged_by_new_fields():
    old = HypothesisRecord.model_validate(BASE)
    again = HypothesisRecord.model_validate({**BASE, "datasets": [], "bounds": {}})
    assert old.hypothesis_hash() == again.hypothesis_hash()


def test_binding_fields_change_hash_but_base_dir_does_not():
    a = HypothesisRecord.model_validate({**BASE, **BOUND})
    assert a.hypothesis_hash() != HypothesisRecord.model_validate(BASE).hypothesis_hash()
    b = HypothesisRecord.model_validate({**BASE, **BOUND, "base_dir": "/somewhere"})
    assert a.hypothesis_hash() == b.hypothesis_hash()


def test_dataset_entries_normalised():
    rec = HypothesisRecord.model_validate({**BASE, **BOUND})
    assert [d.name for d in rec.datasets] == ["px", "rel"]
    assert rec.datasets[0].slow is False and rec.datasets[1].slow is True


def test_unknown_bound_key_rejected():
    rec, reasons = check_hypothesis_spec({**BASE, "bounds": {"max_nonsense": 1}})
    assert rec is None and any("bounds" in r for r in reasons)


def test_bad_adapter_ref_and_window_rejected():
    assert check_hypothesis_spec({**BASE, "adapter": "no_colon"})[0] is None
    assert check_hypothesis_spec({**BASE, "windows": {"development": ["x", "2020-01-02"]}})[0] is None
    assert check_hypothesis_spec({**BASE, "windows": {"development": ["2020-02-01", "2020-01-02"]}})[0] is None


def test_registry_roundtrip_and_immutable(tmp_path):
    reg = Registry(tmp_path / "r.sqlite3")
    rec = HypothesisRecord.model_validate({**BASE, **BOUND, "base_dir": "/x"})
    tid = reg.register_hypothesis(rec)
    t = reg.get_trial(tid)
    back = HypothesisRecord.model_validate({k: t[k] for k in HypothesisRecord.model_fields})
    assert back.hypothesis_hash() == t["hypothesis_hash"] == rec.hypothesis_hash()
    assert t["bounds"]["max_gross_leverage"] == 1.0 and t["base_dir"] == "/x"
    import sqlite3
    with pytest.raises(sqlite3.DatabaseError):
        reg.conn.execute("UPDATE hypotheses SET binding='{}' WHERE trial_id=?", (tid,))


def test_check_cache_roundtrip(tmp_path):
    reg = Registry(tmp_path / "r.sqlite3")
    assert reg.cache_get("k") is None
    reg.cache_put("k", {"verdict": "PASS", "n": 1})
    assert reg.cache_get("k") == {"verdict": "PASS", "n": 1}

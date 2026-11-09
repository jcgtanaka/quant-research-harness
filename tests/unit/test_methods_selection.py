import json

import pytest

from qrh.methods.builtin import builtin_registry
from qrh.methods.registry import Method, MethodError
from qrh.methods.selection import MethodSelector, check_pins
from qrh.methods.state import effective_status, load_state, save_state, state_path


def _write_state(home, statuses):
    save_state(home, {"schema_version": 1, "statuses": statuses, "history": []})


def test_missing_state_means_code_defaults(tmp_path):
    st = load_state(tmp_path)
    assert st["statuses"] == {} and st["history"] == []
    assert not state_path(tmp_path).exists()  # reading never creates the file
    sel = MethodSelector(tmp_path)
    assert sel.default("s5.pbo").version == "cscv@1"


def test_unreadable_or_malformed_state_is_an_error_not_a_silent_default(tmp_path):
    p = state_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    with pytest.raises(MethodError):
        load_state(tmp_path)
    p.write_text(json.dumps({"schema_version": 1, "statuses": {"s5.pbo": {"cscv@1": "weird"}}, "history": []}))
    with pytest.raises(MethodError):
        load_state(tmp_path)
    p.write_text(json.dumps({"schema_version": 99, "statuses": {}, "history": []}))
    with pytest.raises(MethodError):
        load_state(tmp_path)


def test_state_round_trip_is_canonical(tmp_path):
    _write_state(tmp_path, {"s5.pbo": {"cscv@1": "deprecated"}})
    first = state_path(tmp_path).read_bytes()
    save_state(tmp_path, load_state(tmp_path))
    assert state_path(tmp_path).read_bytes() == first


def test_effective_status_applies_local_overrides(tmp_path):
    reg = builtin_registry()
    _write_state(tmp_path, {"s4.walkforward": {"purged_embargo@1": "default", "rolling_anchored@1": "deprecated"}})
    st = load_state(tmp_path)
    assert effective_status(reg.get("s4.walkforward", "purged_embargo@1"), st) == "default"
    assert effective_status(reg.get("s4.walkforward", "rolling_anchored@1"), st) == "deprecated"
    assert effective_status(reg.get("s5.pbo", "cscv@1"), st) == "default"


def test_local_promotion_changes_the_default(tmp_path):
    _write_state(tmp_path, {"s4.walkforward": {"purged_embargo@1": "default", "rolling_anchored@1": "deprecated"}})
    assert MethodSelector(tmp_path).default("s4.walkforward").version == "purged_embargo@1"


def test_state_with_no_or_two_defaults_is_an_error(tmp_path):
    _write_state(tmp_path, {"s5.pbo": {"cscv@1": "experimental"}})
    with pytest.raises(MethodError):
        MethodSelector(tmp_path).default("s5.pbo")
    _write_state(tmp_path, {"s4.walkforward": {"purged_embargo@1": "default"}})
    with pytest.raises(MethodError):
        MethodSelector(tmp_path).default("s4.walkforward")


def test_get_uses_pin_else_default_and_records_use(tmp_path):
    sel = MethodSelector(tmp_path, pinned={"s4.walkforward": "purged_embargo@1"})
    assert sel.get("s4.walkforward").version == "purged_embargo@1"
    assert sel.get("s5.pbo").version == "cscv@1"
    assert sel.used == {"s4.walkforward": "purged_embargo@1", "s5.pbo": "cscv@1"}
    assert sel.non_default() == {"s4.walkforward": "purged_embargo@1"}


def test_overrides_beat_pins_and_are_used_for_benchmarks(tmp_path):
    sel = MethodSelector(tmp_path, pinned={"s4.walkforward": "rolling_anchored@1"},
                         overrides={"s4.walkforward": "purged_embargo@1"})
    assert sel.get("s4.walkforward").version == "purged_embargo@1"


def test_unknown_pin_is_an_error(tmp_path):
    with pytest.raises(MethodError):
        MethodSelector(tmp_path, pinned={"s5.pbo": "cscv@9"}).get("s5.pbo")


def test_check_pins_reports_unknown_ids_versions_and_deprecated(tmp_path):
    _write_state(tmp_path, {"s4.walkforward": {"purged_embargo@1": "default", "rolling_anchored@1": "deprecated"}})
    assert check_pins(tmp_path, {"s5.pbo": "cscv@1"}) == []
    reasons = check_pins(tmp_path, {"nope.x": "a@1", "s5.pbo": "cscv@9", "s4.walkforward": "rolling_anchored@1", "bad": "x"})
    text = " | ".join(reasons)
    assert "nope.x" in text and "cscv@9" in text and "deprecated" in text and "bad" in text
    assert len(reasons) == 4


def test_full_map_fills_every_id_with_pin_or_default(tmp_path):
    sel = MethodSelector(tmp_path, pinned={"s4.walkforward": "purged_embargo@1"})
    full = sel.full_map()
    assert full["s4.walkforward"] == "purged_embargo@1" and full["s5.pbo"] == "cscv@1"
    assert set(full) == set(builtin_registry().ids()) - {"s2.pnl_reconstruction", "s2.pnl_linear"}  # opt-in ids need a pin


def test_custom_registry_for_tests_and_experiments(tmp_path):
    reg = builtin_registry()
    reg.register(Method(id="s5.pbo", version="cscv@2", stage="S5", kind="statistic", entry=lambda *a, **k: None,
                        parameters={}, status="experimental", references=(), added_in="0.0.1"))
    sel = MethodSelector(tmp_path, registry=reg, overrides={"s5.pbo": "cscv@2"})
    assert sel.get("s5.pbo").version == "cscv@2"
    assert sel.non_default() == {"s5.pbo": "cscv@2"}

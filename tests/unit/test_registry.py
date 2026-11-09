import itertools
import sqlite3

import pytest

from qrh.memory.registry import HypothesisRecord, Registry, RegistryError


def make_rec(**kw):
    base = dict(
        text="Persistence of short-horizon returns",
        strategy_family="trend",
        asset_class="synthetic",
        frequency="1D",
        parameter_space_size=12,
        thresholds={"min_score": 0.5},
        difference_statement="Differs from the baseline by using a longer lookback.",
    )
    base.update(kw)
    return HypothesisRecord(**base)


@pytest.fixture
def reg(tmp_path):
    counter = itertools.count(1)
    r = Registry(tmp_path / "r.sqlite3", id_factory=lambda: f"t{next(counter):03d}")
    yield r
    r.close()


def test_pragmas_and_schema_version(reg):
    assert reg.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert reg.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert reg.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] >= 1


def test_reopen_keeps_data(tmp_path):
    r1 = Registry(tmp_path / "r.sqlite3", id_factory=lambda: "a")
    r1.register_hypothesis(make_rec())
    r1.close()
    r2 = Registry(tmp_path / "r.sqlite3")
    assert r2.get_trial("a")["strategy_family"] == "trend"
    r2.close()


def test_register_and_get(reg):
    tid = reg.register_hypothesis(make_rec())
    assert tid == "t001"
    t = reg.get_trial(tid)
    assert t["thresholds"] == {"min_score": 0.5}
    assert t["family_id"] == "trend|synthetic|1D"
    assert len(t["hypothesis_hash"]) == 64
    assert reg.get_trial("missing") is None


def test_explicit_family_id(reg):
    tid = reg.register_hypothesis(make_rec(family_id="fam-x"))
    assert reg.get_trial(tid)["family_id"] == "fam-x"


def test_duplicate_trial_id_rejected(tmp_path):
    r = Registry(tmp_path / "r.sqlite3", id_factory=lambda: "same")
    r.register_hypothesis(make_rec())
    with pytest.raises(sqlite3.IntegrityError):
        r.register_hypothesis(make_rec())
    r.close()


def test_hypothesis_hash_deterministic():
    assert make_rec().hypothesis_hash() == make_rec().hypothesis_hash()
    assert make_rec().hypothesis_hash() != make_rec(thresholds={"min_score": 0.6}).hypothesis_hash()


def test_no_update_api():
    assert not [n for n in dir(Registry) if "update" in n and "threshold" in n]


def test_thresholds_immutable_at_db_level(reg):
    tid = reg.register_hypothesis(make_rec())
    for col, val in [("thresholds", "{}"), ("difference_statement", "x"), ("hypothesis_hash", "0")]:
        with pytest.raises(sqlite3.IntegrityError):
            reg.conn.execute(f"UPDATE hypotheses SET {col}=? WHERE trial_id=?", (val, tid))
    assert reg.get_trial(tid)["thresholds"] == {"min_score": 0.5}


def test_trial_counts_and_families(reg):
    a = reg.register_hypothesis(make_rec())
    b = reg.register_hypothesis(make_rec())  # same default family
    c = reg.register_hypothesis(make_rec(asset_class="other"))
    reg.record_trial(a, {"x": 1}, {"score": 0.1})
    reg.record_trial(a, {"x": 2}, {"score": 0.2})
    reg.record_trial(a, {"x": 2}, {"score": 0.2}, is_oos=True)  # same combo, counted once
    reg.record_trial(b, {"x": 1}, {"score": 0.3})
    reg.record_trial(c, {"x": 1}, {"score": 0.3})
    assert reg.trial_count(a) == 2
    assert reg.trial_count(b) == 1
    assert reg.family_trial_count("trend|synthetic|1D") == 3
    assert reg.family_trial_count("trend|other|1D") == 1
    assert reg.family_trial_count("unknown") == 0


def test_combo_hash_ignores_key_order(reg):
    a = reg.register_hypothesis(make_rec())
    h1 = reg.record_trial(a, {"x": 1, "y": 2}, {})
    h2 = reg.record_trial(a, {"y": 2, "x": 1}, {})
    assert h1 == h2 and reg.trial_count(a) == 1


def test_record_trial_unknown_trial_rejected(reg):
    with pytest.raises(sqlite3.IntegrityError):
        reg.record_trial("nope", {"x": 1}, {})


def test_receipt_index_upsert_and_clear(reg):
    tid = reg.register_hypothesis(make_rec())
    reg.index_receipt(tid, "S0", "h0", "PASS", "/p/S0.json")
    reg.index_receipt(tid, "S1", "h1", "BLOCKED", "/p/S1.json")
    reg.index_receipt(tid, "S1", "h1b", "PASS", "/p/S1.json")
    assert reg.get_receipt_entry(tid, "S1")["receipt_hash"] == "h1b"
    assert [e["stage"] for e in reg.list_receipts(tid)] == ["S0", "S1"]
    reg.clear_receipts(tid, from_stage="S1")
    assert [e["stage"] for e in reg.list_receipts(tid)] == ["S0"]
    assert reg.get_receipt_entry(tid, "S5") is None


def test_override_requires_reason(reg):
    tid = reg.register_hypothesis(make_rec())
    for bad in ["", "   "]:
        with pytest.raises(RegistryError):
            reg.log_override(tid, "S2", bad)
    reg.log_override(tid, "S2", "documented exception")
    ov = reg.list_overrides(tid)
    assert len(ov) == 1 and ov[0]["reason"] == "documented exception"


def test_override_blank_reason_blocked_by_db(reg):
    tid = reg.register_hypothesis(make_rec())
    with pytest.raises(sqlite3.IntegrityError):
        reg.conn.execute(
            "INSERT INTO overrides(trial_id, stage, reason, created_at) VALUES (?,?,?,?)",
            (tid, "S1", " ", "now"),
        )


def test_list_trials_filters_and_latest_verdict(reg):
    a = reg.register_hypothesis(make_rec())
    b = reg.register_hypothesis(make_rec(asset_class="other", frequency="1h"))
    c = reg.register_hypothesis(make_rec(strategy_family="carry"))
    reg.index_receipt(a, "S0", "h", "PASS", "p")
    reg.index_receipt(a, "S1", "h", "BLOCKED", "p")
    reg.index_receipt(b, "S0", "h", "PASS", "p")
    assert [t["trial_id"] for t in reg.list_trials()] == [c, b, a]  # newest first
    assert [t["trial_id"] for t in reg.list_trials(asset_class="other")] == [b]
    assert [t["trial_id"] for t in reg.list_trials(strategy_family="carry")] == [c]
    assert [t["trial_id"] for t in reg.list_trials(frequency="1h")] == [b]
    assert [t["trial_id"] for t in reg.list_trials(verdict="BLOCKED")] == [a]
    assert [t["trial_id"] for t in reg.list_trials(verdict="PASS")] == [b]
    by_id = {t["trial_id"]: t for t in reg.list_trials()}
    assert by_id[a]["verdict"] == "BLOCKED" and by_id[c]["verdict"] is None
    assert len(reg.list_trials(limit=2)) == 2

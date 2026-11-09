import itertools

import pytest

from qrh.memory import db
from qrh.memory.registry import HypothesisRecord, Registry, RegistryError


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "r.sqlite3")
    yield r
    r.close()


def test_migrations_apply_to_a_v3_database(tmp_path):
    """An older database upgrades in place; its overrides survive the table rebuild."""
    import sqlite3

    conn = sqlite3.connect(tmp_path / "old.sqlite3", isolation_level=None)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for i, script in enumerate(db.MIGRATIONS[:3], start=1):
        for stmt in db._split(script):
            conn.execute(stmt)
        conn.execute("INSERT INTO schema_version VALUES (?)", (i,))
    conn.execute(
        "INSERT INTO hypotheses(trial_id,text,strategy_family,asset_class,frequency,parameter_space_size,"
        "thresholds,difference_statement,family_id,hypothesis_hash,created_at) VALUES"
        " ('t1','old text about momentum','f','a','1D',1,'{}','d','f|a|1D','h','now')"
    )
    conn.execute("INSERT INTO overrides(trial_id,stage,reason,created_at) VALUES ('t1','S2','kept reason','now')")
    conn.close()
    r = Registry(tmp_path / "old.sqlite3")
    assert r.list_overrides("t1")[0]["reason"] == "kept reason"
    from qrh.memory.search import FtsSearcher

    assert [h.trial_id for h in FtsSearcher(r.conn).search("momentum")] == ["t1"]  # backfilled
    r.close()


def test_hook_override_roundtrip(reg):
    reg.log_hook_override("data-gate", "reason long enough", "s1", approval_id="abc123abc123")
    reg.log_hook_override("data-gate", "older style row", "s1")
    rows = reg.list_hook_overrides(since="1970-01-01T00:00:00+00:00")
    assert rows[0]["hook"] == "data-gate" and rows[0]["session_id"] == "s1" and rows[0]["trial_id"] is None
    assert rows[0]["approval_id"] == "abc123abc123" and rows[1]["approval_id"] is None


def test_old_override_rows_stay_readable_after_the_approval_id_migration(tmp_path):
    import sqlite3

    conn = sqlite3.connect(tmp_path / "v6.sqlite3")
    from qrh.memory import db

    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for i, script in enumerate(db.MIGRATIONS[:6], 1):
        for stmt in db._split(script):
            conn.execute(stmt)
        conn.execute("INSERT INTO schema_version VALUES (?)", (i,))
    conn.execute("INSERT INTO overrides(trial_id,stage,reason,created_at,hook,session_id) "
                 "VALUES (NULL,NULL,'QRH_OVERRIDE era reason','2026-01-01T00:00:00+00:00','data-gate','s0')")
    conn.commit()
    conn.close()
    r = Registry(tmp_path / "v6.sqlite3")
    rows = r.list_hook_overrides(since="1970-01-01T00:00:00+00:00")
    r.close()
    assert rows[0]["reason"] == "QRH_OVERRIDE era reason" and rows[0]["approval_id"] is None


def test_hook_override_requires_reason(reg):
    with pytest.raises(RegistryError):
        reg.log_hook_override("data-gate", "  ", "s1")


def test_list_hook_overrides_since_filters_old(reg):
    reg.log_hook_override("data-gate", "reason long enough", "s1")
    assert reg.list_hook_overrides(since="2999-01-01T00:00:00+00:00") == []


def test_test_receipts(reg):
    assert reg.latest_passing_test_receipt("h1") is None
    reg.record_test_receipt("h1", "pytest -q", 1, 2.5)
    assert reg.latest_passing_test_receipt("h1") is None  # exit != 0 is not a pass
    reg.record_test_receipt("h1", "pytest -q", 0, 3.0)
    got = reg.latest_passing_test_receipt("h1")
    assert got["passed"] == 1 and got["exit_code"] == 0 and got["duration_s"] == 3.0
    assert reg.latest_passing_test_receipt("other") is None

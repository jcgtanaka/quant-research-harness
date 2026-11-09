"""Change impact: which trials' receipts are stale after source files changed (information only)."""
from __future__ import annotations

from qrh.maintenance.change_impact import change_impact
from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.receipt import Receipt, Stage, Verdict, write_receipt

CURRENT = "c" * 64
OLD = "o" * 64


def _trial(home, trial_id: str, code_hash: str) -> None:
    reg = Registry(home / "registry.sqlite3")
    reg.register_hypothesis(
        HypothesisRecord(
            text="synthetic idea", strategy_family="demo", asset_class="synthetic", frequency="daily",
            parameter_space_size=1, thresholds={}, difference_statement="synthetic",
        ),
        trial_id=trial_id,
    )
    receipt = Receipt(trial_id=trial_id, stage=Stage.S0, hypothesis_hash="h" * 64, code_hash=code_hash,
                      verdict=Verdict.PASS)
    write_receipt(receipt, home, reg)
    reg.close()


def test_matching_hash_is_not_stale(tmp_path):
    _trial(tmp_path, "t1", CURRENT)
    result = change_impact(tmp_path, ["src/a.py"], current_hash=CURRENT)
    assert [r for r in result.trials if r["stale"]] == []


def test_changed_code_lists_trial_with_sentence(tmp_path):
    _trial(tmp_path, "t1", OLD)
    _trial(tmp_path, "t2", CURRENT)
    result = change_impact(tmp_path, ["src/a.py"], current_hash=CURRENT)
    stale = [r for r in result.trials if r["stale"]]
    assert [r["trial_id"] for r in stale] == ["t1"]
    assert "t1" in stale[0]["sentence"] and "out of date" in stale[0]["sentence"]


def test_no_registry_gives_empty_list_and_note(tmp_path):
    result = change_impact(tmp_path, ["src/a.py"], current_hash=CURRENT)
    assert result.trials == []
    assert "registry" in result.note.lower()
    assert not (tmp_path / "registry.sqlite3").exists()


def test_never_raises(tmp_path):
    (tmp_path / "registry.sqlite3").write_text("not a database")
    assert change_impact(tmp_path, ["x"], current_hash=CURRENT).trials == []
    assert change_impact(tmp_path / "missing" / "deeper", None, current_hash=CURRENT).trials == []  # type: ignore[arg-type]

"""Tests for the one-attempt bug correction flow (synthetic data only)."""

import json
import os

import pytest

from qrh.checks import bug_correction as bc

BUGS = [{"day": "2024-01-03", "amount": 12.5}, {"day": "2024-01-09", "amount": -4.0}]


def test_request_file_lists_each_bug(tmp_path):
    run = tmp_path / "run1"
    path = bc.correction_request(run, BUGS)
    text = open(path).read()
    assert path == run / "correction_request.txt"
    assert "2024-01-03" in text and "12.5" in text
    assert "2024-01-09" in text and "4.0" in text
    assert "fix the backtest file" in text
    assert bc.correction_state(run) == "requested"


def test_no_bugs_gives_no_request(tmp_path):
    run = tmp_path / "run2"
    assert bc.correction_state(run) == "none"
    assert bc.correction_request(run, []) is None
    assert not run.exists()
    assert bc.correction_state(run) == "none"


def test_record_changes_state_to_done(tmp_path):
    run = tmp_path / "run3"
    bc.correction_request(run, BUGS)
    path = bc.record_correction(run, "moved the fee to the next day")
    data = json.loads(open(path).read())
    assert data["note"] == "moved the fee to the next day"
    assert data["attempt"] == 1
    assert data["time"]
    assert bc.correction_state(run) == "done"


def test_second_record_is_refused(tmp_path):
    run = tmp_path / "run4"
    bc.correction_request(run, BUGS)
    bc.record_correction(run, "first")
    with pytest.raises(ValueError):
        bc.record_correction(run, "second")
    assert json.loads(open(run / "correction_done.json").read())["note"] == "first"


def test_second_request_after_done_is_exhausted(tmp_path):
    run = tmp_path / "run5"
    bc.correction_request(run, BUGS)
    bc.record_correction(run, "first")
    bc.correction_request(run, BUGS)
    assert bc.correction_state(run) == "exhausted"


def test_bug_still_present_is_inconclusive_naming_day(tmp_path):
    run = tmp_path / "run6"
    bc.correction_request(run, BUGS)
    bc.record_correction(run, "tried")
    verdict, reason = bc.verdict_after_correction(BUGS, BUGS, run)
    assert verdict == "INCONCLUSIVE"
    assert "2024-01-03" in reason and "12.5" in reason


def test_corrected_run_gives_recheck_reason(tmp_path):
    run = tmp_path / "run7"
    bc.correction_request(run, BUGS)
    bc.record_correction(run, "fixed")
    verdict, reason = bc.verdict_after_correction(BUGS, [], run)
    assert verdict == "PASS"
    assert "corrected once" in reason and "re-checked" in reason


def test_no_bugs_keeps_caller_verdict(tmp_path):
    verdict, reason = bc.verdict_after_correction([], [], tmp_path / "none")
    assert verdict == "PASS"
    assert "not changed" in reason


def test_bug_before_any_correction_is_inconclusive(tmp_path):
    verdict, reason = bc.verdict_after_correction(BUGS, BUGS, tmp_path / "x")
    assert verdict == "INCONCLUSIVE"
    assert "not been corrected" in reason


def test_writes_are_atomic(tmp_path, monkeypatch):
    run = tmp_path / "run8"
    calls = []
    real = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    bc.correction_request(run, BUGS)
    bc.record_correction(run, "note")
    assert len(calls) == 2
    assert all(src != dst and not os.path.exists(src) for src, dst in calls)
    assert sorted(p.name for p in run.iterdir()) == [
        "correction_done.json",
        "correction_request.txt",
    ]

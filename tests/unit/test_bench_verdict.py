"""A new bench run records a verdict derived from the suite's own detect / false-alarm rule; old receipts are unchanged."""

from __future__ import annotations

import json
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest

from qrh.bench import run as bench_run
from qrh.bench.receipt import BenchReceipt, load_bench_receipt, write_bench
from qrh.ui.server import make_server

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "bench"


def _m(k, n, direction, gate=True):
    return {"k": k, "n": n, "rate": (k / n if n else None), "ci": None, "direction": direction, "gate": gate, "label": ""}


def _planted(detect_k, false_alarm_k, n=4):
    return {
        "reps": 2,
        "metrics": {
            "detect/lookahead": _m(detect_k, n, "higher"),
            "detect/lookahead/variant_a": _m(detect_k, n, "higher", gate=False),  # per-variant: information only
            "false_alarm/controls": _m(false_alarm_k, n, "lower"),
        },
    }


def _patch_planted(monkeypatch, result):
    monkeypatch.setitem(bench_run.RUNNERS, "planted-bugs", lambda arms, cfg, seed: {"candidate": result})


def _run(tmp_path, **kw):
    return bench_run.run_bench(tmp_path, suites=["planted-bugs"], config={"planted-bugs": {"reps": 1}}, **kw)


def test_all_planted_defects_found_and_controls_clean_is_pass(tmp_path, monkeypatch):
    _patch_planted(monkeypatch, _planted(4, 0))
    r = _run(tmp_path)
    assert r.verdict == "PASS" and r.suite_verdicts == {"planted-bugs": "PASS"}


def test_a_missed_planted_defect_is_fail(tmp_path, monkeypatch):
    _patch_planted(monkeypatch, _planted(3, 0))
    assert _run(tmp_path).verdict == "FAIL"


def test_a_false_alarm_on_a_clean_control_is_fail(tmp_path, monkeypatch):
    _patch_planted(monkeypatch, _planted(4, 1))
    assert _run(tmp_path).verdict == "FAIL"


def test_suite_without_a_pass_rule_is_no_rule(tmp_path):
    cfg = {"noise-fpr": {"grids": [12], "periods": [400], "reps": 4}}
    r = bench_run.run_bench(tmp_path, suites=["noise-fpr"], config=cfg)
    assert r.suite_verdicts == {"noise-fpr": "NO_RULE"} and r.verdict == "NO_RULE"


def test_mixed_ruled_and_ruleless_suites_never_report_pass(tmp_path, monkeypatch):
    _patch_planted(monkeypatch, _planted(4, 0))
    cfg = {"planted-bugs": {"reps": 1}, "noise-fpr": {"grids": [12], "periods": [400], "reps": 4}}
    r = bench_run.run_bench(tmp_path, suites=["planted-bugs", "noise-fpr"], config=cfg)
    assert r.suite_verdicts == {"planted-bugs": "PASS", "noise-fpr": "NO_RULE"} and r.verdict == "NO_RULE"


def test_verdict_is_part_of_the_hash_of_a_new_receipt_and_survives_a_round_trip(tmp_path, monkeypatch):
    _patch_planted(monkeypatch, _planted(4, 0))
    r = _run(tmp_path)
    forged = r.model_copy(update={"verdict": "FAIL"})
    assert forged.receipt_hash() != r.receipt_hash()
    folder = write_bench(r, tmp_path)
    loaded = load_bench_receipt(folder / "receipt.json")
    assert loaded.verdict == "PASS" and loaded.receipt_hash() == r.receipt_hash()


def _example_ids():
    return sorted(p.name for p in EXAMPLES.iterdir() if (p / "receipt.json").is_file()) if EXAMPLES.is_dir() else []


@pytest.mark.skipif(not _example_ids(), reason="no committed bench examples")
@pytest.mark.parametrize("name", _example_ids())
def test_committed_receipt_keeps_its_hash_and_is_rewritten_byte_stable(name, tmp_path):
    src = EXAMPLES / name / "receipt.json"
    stored = json.loads(src.read_text(encoding="utf-8"))
    loaded = load_bench_receipt(src)  # verifies the stored hash
    assert loaded.verdict is None and loaded.suite_verdicts is None
    assert loaded.receipt_hash() == stored["receipt_hash"]
    again = write_bench(loaded, tmp_path)
    rewritten = json.loads((again / "receipt.json").read_text(encoding="utf-8"))
    assert "verdict" not in rewritten["receipt"] and "suite_verdicts" not in rewritten["receipt"]
    assert rewritten["receipt_hash"] == stored["receipt_hash"]


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path))
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _bench(base):
    with urllib.request.urlopen(base + "/api/bench", timeout=5) as r:
        return {b["bench_id"]: b for b in json.loads(r.read())["benches"]}


@pytest.mark.skipif(not _example_ids(), reason="no committed bench examples")
def test_endpoint_reports_previous_value_for_old_receipt_and_new_verdict_for_new_run(served, monkeypatch):
    base, home = served
    old = EXAMPLES / _example_ids()[0]
    shutil.copytree(old, home / "bench" / old.name)
    _patch_planted(monkeypatch, _planted(3, 0))
    new = _run(home)
    write_bench(new, home)
    got = _bench(base)
    assert got[old.name]["verdict"] == "NOT_RECORDED"
    assert got[new.bench_id()]["verdict"] == "FAIL"


def test_endpoint_shows_pass_for_a_new_passing_run(served, monkeypatch):
    base, home = served
    _patch_planted(monkeypatch, _planted(4, 0))
    new = _run(home)
    write_bench(new, home)
    assert _bench(base)[new.bench_id()]["verdict"] == "PASS"

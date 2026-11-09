"""Stored pnl-linear sweep results show up in GET /api/bench; synthetic numbers only, temp QRH_HOME."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from qrh.bench.receipt import bench_root, list_receipts
from qrh.bench.sweep_store import write_sweep_receipt
from qrh.ui.server import make_server

SWEEP = {
    "fail_share": {(0.0, 1.0): 0.5, (0.0, 5.0): 1.0, (1.0, 1.0): 0.25, (1.0, 5.0): 0.75},
    "honest_false_alarm": {0.0: 0.0, 1.0: 0.125},
    "n_cases_per_cell": 8,
}
EXPECTED_KEYS = {"bench_id", "suite", "created_at", "verdict", "metrics"}


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


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read())


@pytest.mark.parametrize("suite", ["pnl-linear-futures", "pnl-linear-slope"])
def test_sweep_appears_in_api_bench_with_suite_and_numeric_metrics(served, suite):
    base, home = served
    folder = write_sweep_receipt(SWEEP, home, suite=suite, seed=3, code_hash="c")
    status, body = _get(base + "/api/bench")
    assert status == 200
    assert set(body) == {"benches"}
    (b,) = body["benches"]
    assert set(b) == EXPECTED_KEYS
    assert b["bench_id"] == folder.name and b["suite"] == suite
    assert b["verdict"] == "NOT_RECORDED"
    assert len(b["metrics"]) == 6
    assert b["metrics"][f"{suite}/fail_share/noise_1bp/gain_5bp"] == 0.75
    assert b["metrics"][f"{suite}/false_alarm/noise_1bp"] == 0.125
    assert all(isinstance(v, float) for v in b["metrics"].values())


def test_same_sweep_twice_gives_one_bench(tmp_path):
    a = write_sweep_receipt(SWEEP, tmp_path, suite="pnl-linear-slope", seed=3, code_hash="c")
    b = write_sweep_receipt(SWEEP, tmp_path, suite="pnl-linear-slope", seed=3, code_hash="c")
    assert a == b and len(list_receipts(tmp_path)) == 1


@pytest.mark.parametrize("bad", ["0.5", None, True, float("nan"), float("inf")])
def test_writer_refuses_a_non_numeric_metric(tmp_path, bad):
    sweep = {**SWEEP, "fail_share": {**SWEEP["fail_share"], (0.0, 1.0): bad}}
    with pytest.raises(ValueError, match="numeric"):
        write_sweep_receipt(sweep, tmp_path, suite="pnl-linear-futures", seed=1, code_hash="c")
    assert not bench_root(tmp_path).exists()


def test_writer_refuses_an_unknown_suite(tmp_path):
    with pytest.raises(ValueError, match="suite"):
        write_sweep_receipt(SWEEP, tmp_path, suite="planted-bugs", seed=1, code_hash="c")
    assert not bench_root(tmp_path).exists()


def test_empty_sweep_writes_nothing(tmp_path):
    empty = {"fail_share": {}, "honest_false_alarm": {}, "n_cases_per_cell": 0}
    assert write_sweep_receipt(empty, tmp_path, suite="pnl-linear-futures", seed=1, code_hash="c") is None
    assert not bench_root(tmp_path).exists()

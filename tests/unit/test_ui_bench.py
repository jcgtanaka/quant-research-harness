"""GET /api/bench: bench receipts read back for the dashboard; read-only, synthetic data only."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from qrh.bench.receipt import BenchReceipt, bench_root, write_bench
from qrh.ui.server import make_server


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
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _metric(rate):
    return {"k": 1, "n": 2, "rate": rate, "ci": [0.0, 1.0], "direction": "higher", "gate": False, "label": ""}


def _write(home, metrics, suite="planted-bugs"):
    receipt = BenchReceipt(
        profile="custom", suites=[suite], seed=1, config={}, methods={"s5.pbo": "cscv@1"},
        results={suite: {"metrics": metrics}}, code_hash="c",
    )
    write_bench(receipt, home)
    return receipt


def test_empty_home_gives_empty_list(served):
    base, _ = served
    assert _get(base + "/api/bench") == (200, {"benches": []})


def test_bench_appears_with_id_suite_time_verdict_and_numeric_metrics(served):
    base, home = served
    r = _write(home, {"detect/a": _metric(0.5), "detect/b": _metric(1.0)})
    status, body = _get(base + "/api/bench")
    assert status == 200
    (b,) = body["benches"]
    assert b["bench_id"] == r.bench_id()
    assert b["suite"] == "planted-bugs"
    assert b["created_at"] == r.created_at
    assert isinstance(b["verdict"], str) and b["verdict"]
    assert b["metrics"] == {"planted-bugs/detect/a": 0.5, "planted-bugs/detect/b": 1.0}
    assert "skipped_metrics" not in b


def test_non_numeric_metric_is_skipped_and_named(served):
    base, home = served
    _write(home, {"detect/a": _metric(0.5), "detect/bad": _metric("high"), "detect/none": _metric(None)})
    _, body = _get(base + "/api/bench")
    (b,) = body["benches"]
    assert b["metrics"] == {"planted-bugs/detect/a": 0.5}
    assert sorted(b["skipped_metrics"]) == ["planted-bugs/detect/bad", "planted-bugs/detect/none"]


def test_corrupt_receipt_is_skipped_with_reason(served):
    base, home = served
    good = _write(home, {"detect/a": _metric(0.5)})
    bad = bench_root(home) / "bench-deadbeef0000"
    bad.mkdir()
    (bad / "receipt.json").write_text("{not json", encoding="utf-8")
    _, body = _get(base + "/api/bench")
    assert [b["bench_id"] for b in body["benches"]] == [good.bench_id()]
    assert body["skipped"][0]["bench_id"] == "bench-deadbeef0000" and body["skipped"][0]["reason"]


def test_unreadable_bench_folder_is_an_api_error(served):
    base, home = served
    bench_root(home).write_text("not a folder", encoding="utf-8")
    status, body = _get(base + "/api/bench")
    assert status == 503 and "error" in body


def test_post_is_refused(served):
    base, _ = served
    req = urllib.request.Request(base + "/api/bench", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 405 and exc.value.headers.get("Allow") == "GET"

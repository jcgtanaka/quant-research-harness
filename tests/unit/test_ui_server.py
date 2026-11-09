"""The dashboard server over localhost: GET only, 127.0.0.1 only, JSON errors."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest
from typer.testing import CliRunner

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.receipt import Receipt, Stage, Verdict, write_receipt
from qrh.ui.server import make_server


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path))
    reg = Registry(tmp_path / "registry.sqlite3")
    reg.register_hypothesis(HypothesisRecord(
        text="synthetic", strategy_family="momentum", asset_class="synthetic", frequency="1d",
        parameter_space_size=2, thresholds={}, difference_statement="synthetic only"), trial_id="t1")
    write_receipt(Receipt(trial_id="t1", stage=Stage.S0, hypothesis_hash="h", code_hash="c",
                          verdict=Verdict.PASS, reasons=["ok"]), tmp_path, reg)
    reg.conn.commit()
    reg.close()
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, tmp_path
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _send(url, method):
    req = urllib.request.Request(url, data=b"{}" if method != "DELETE" else None, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Allow"), json.loads(e.read())


def test_health(served):
    base, _ = served
    assert _get(base + "/api/health") == (200, {"status": "ok", "bound_to": "127.0.0.1"})


def test_trials_list_and_detail(served):
    base, _ = served
    status, body = _get(base + "/api/trials")
    assert status == 200 and body["trials"][0]["trial_id"] == "t1"
    assert body["trials"][0]["stage_verdicts"] == {"S0": "PASS"}
    status, body = _get(base + "/api/trials/t1")
    assert status == 200 and body["stages"]["S0"]["reasons"] == ["ok"]


def test_warnings(served):
    base, _ = served
    status, body = _get(base + "/api/trials/t1/warnings")
    assert status == 200 and body["trial_id"] == "t1" and body["warnings"]


def test_unknown_trial_and_route_404(served):
    base, _ = served
    for path in ("/api/trials/nope", "/api/trials/nope/warnings", "/api/trials/t1/other", "/api/nothing"):
        status, body = _get(base + path)
        assert status == 404 and body["error"]


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_non_get_is_405(served, method):
    base, _ = served
    status, allow, body = _send(base + "/api/trials", method)
    assert status == 405 and allow == "GET" and body["error"]


def test_head_is_405(served):
    base, _ = served
    req = urllib.request.Request(base + "/api/health", method="HEAD")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 405


def test_get_does_not_modify_registry_or_receipts(served):
    base, home = served
    before = {p: p.read_bytes() for p in (home / "receipts").rglob("*.json")}
    for path in ("/api/trials", "/api/trials/t1", "/api/trials/t1/warnings"):
        _get(base + path)
    assert before == {p: p.read_bytes() for p in (home / "receipts").rglob("*.json")}


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "192.168.1.5", "", "::1"])
def test_other_hosts_refused(tmp_path, host):
    with pytest.raises(ValueError, match="127.0.0.1"):
        make_server(tmp_path, host=host, port=0)


def test_missing_registry_returns_clear_json_error(tmp_path):
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _get(f"http://127.0.0.1:{server.server_address[1]}/api/trials")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert status == 503 and "registry" in body["error"]
    assert not (tmp_path / "registry.sqlite3").exists()


def test_cli_ui_refuses_other_host(tmp_path, monkeypatch):
    from qrh.cli import app

    monkeypatch.setenv("QRH_HOME", str(tmp_path))
    res = CliRunner().invoke(app, ["ui", "--host", "0.0.0.0"])
    assert res.exit_code == 2

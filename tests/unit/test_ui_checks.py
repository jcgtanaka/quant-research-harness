"""The dashboard Checks tab: parity reports, data checks, cost source and adjustment. Read-only, synthetic data."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.receipt import Receipt, Stage, Verdict, write_receipt
from qrh.ui import data
from qrh.ui.server import make_server

STATIC = Path(__file__).resolve().parents[2] / "src" / "qrh" / "ui" / "static"

PARITY = {"trial_id": "t1", "strategy_name": "demo", "status": "DIFFERENT", "days_compared": 5,
          "start_date": "2026-01-02", "end_date": "2026-01-08", "differing_days": [],
          "warnings": ["The backtest does not match real trading on the days listed."],
          "explanation": "The backtest and live results differ on 1 day(s)."}
CHECK = {"dataset": "demo_prices", "verdict": "WARN", "last_date": "2026-01-02", "age_days": 40,
         "checked_on": "2026-02-11", "findings": ["The data is 40 days old and the limit is 10 days. What to do: update the data."]}


def _serve(home):
    server = make_server(home, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path))
    server, thread, base = _serve(tmp_path)
    yield base, tmp_path
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_parity_returns_stored_items(served):
    base, home = served
    (home / "parity").mkdir()
    (home / "parity" / "t1.json").write_text(json.dumps(PARITY), encoding="utf-8")
    (home / "parity" / "t1.txt").write_text("ignored", encoding="utf-8")
    status, body = _get(base + "/api/parity")
    assert status == 200 and body == {"items": [PARITY]}


def test_data_checks_returns_stored_items(served):
    base, home = served
    (home / "maintenance").mkdir()
    (home / "maintenance" / "data-demo_prices.json").write_text(json.dumps(CHECK), encoding="utf-8")
    (home / "maintenance" / "other.json").write_text("{}", encoding="utf-8")
    status, body = _get(base + "/api/data-checks")
    assert status == 200 and body == {"items": [CHECK]}


@pytest.mark.parametrize("path", ["/api/parity", "/api/data-checks"])
def test_missing_folder_is_empty_list_with_200(served, path):
    base, _ = served
    assert _get(base + path) == (200, {"items": []})


@pytest.mark.parametrize("path,folder,name", [("/api/parity", "parity", "bad.json"),
                                              ("/api/data-checks", "maintenance", "data-bad.json")])
def test_unreadable_file_is_an_item_with_an_error_not_a_crash(served, path, folder, name):
    base, home = served
    (home / folder).mkdir()
    (home / folder / name).write_text("{not json", encoding="utf-8")
    status, body = _get(base + path)
    assert status == 200 and len(body["items"]) == 1
    assert body["items"][0]["file"] == name and body["items"][0]["error"]


@pytest.mark.parametrize("path,folder", [("/api/parity", "parity"), ("/api/data-checks", "maintenance")])
def test_folder_that_is_a_file_gives_standard_error_json(served, path, folder):
    base, home = served
    (home / folder).write_text("x", encoding="utf-8")
    status, body = _get(base + path)
    assert status == 503 and body["error"]


def test_checks_endpoints_are_get_only(served):
    base, _ = served
    req = urllib.request.Request(base + "/api/parity", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=5)
    assert ei.value.code == 405


def test_reading_does_not_change_files(served):
    base, home = served
    (home / "parity").mkdir()
    f = home / "parity" / "t1.json"
    f.write_text(json.dumps(PARITY), encoding="utf-8")
    before = f.read_bytes()
    _get(base + "/api/parity")
    assert f.read_bytes() == before and sorted(p.name for p in (home / "parity").iterdir()) == ["t1.json"]


def _trial_home(tmp_path, reconstruction, s2_params):
    reg = Registry(tmp_path / "registry.sqlite3")
    reg.register_hypothesis(HypothesisRecord(
        text="synthetic", strategy_family="momentum", asset_class="futures", frequency="1d",
        parameter_space_size=2, thresholds={}, difference_statement="synthetic only",
        reconstruction=reconstruction), trial_id="t1")
    write_receipt(Receipt(trial_id="t1", stage=Stage.S2, hypothesis_hash="h", code_hash="c",
                          verdict=Verdict.PASS, reasons=["ok"], params=s2_params), tmp_path, reg)
    reg.conn.commit()
    reg.close()


def test_trial_detail_shows_cost_source_and_adjustment_as_recorded(tmp_path):
    cost = {"status": "OK", "source": "broker_record", "broker": "DemoBroker", "product": "ES"}
    _trial_home(tmp_path, {"adjustment": {"close": "none"}}, {"pnl_linear": {"verdict": "PASS", "cost": cost}})
    d = data.trial_detail(tmp_path, "t1")
    assert d["cost_source"] == {"source": "broker_record", "broker": "DemoBroker", "status": "OK", "reason": None}
    assert d["adjustment"] == {"close": "none"}


def test_trial_detail_blocked_cost_keeps_reason(tmp_path):
    cost = {"status": "BLOCKED", "source": "broker_record", "broker": "DemoBroker", "reason": "cost sub-check: none"}
    _trial_home(tmp_path, {"adjustment": {"close": "back_adjusted"}}, {"pnl_linear": {"verdict": "BLOCKED", "cost": cost}})
    d = data.trial_detail(tmp_path, "t1")
    assert d["cost_source"]["status"] == "BLOCKED" and d["cost_source"]["reason"] == "cost sub-check: none"


def test_trial_detail_without_records_gives_none(tmp_path):
    _trial_home(tmp_path, None, {})
    d = data.trial_detail(tmp_path, "t1")
    assert d["cost_source"] is None and d["adjustment"] is None


def test_html_has_checks_tab():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<a[^>]*href="#/checks"[^>]*id="tab-checks"[^>]*>\s*Checks\s*</a>', html) or \
        re.search(r'<a[^>]*id="tab-checks"[^>]*href="#/checks"[^>]*>\s*Checks\s*</a>', html)
    # Exactly one button is allowed: the read-only Refresh button.
    assert html.count("<button") == 1 and re.search(r'<button type="button" id="refresh"[^>]*>Refresh</button>', html)
    assert "<form" not in html and "<input" not in html


def test_js_has_no_inner_html_and_calls_only_the_two_new_endpoints():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "outerHTML" not in js and "insertAdjacentHTML" not in js
    paths = set(re.findall(r"""["'](/api/[^"']*)["']""", js))
    assert {"/api/parity", "/api/data-checks"} <= paths
    for p in paths:
        assert re.match(r"^/api/(bench|trials|parity|data-checks)(/|$)", p), p
    for text in ("No parity report yet", "No data checks yet"):
        assert text in js, text
    assert "http://" not in js and "https://" not in js


def test_css_has_checks_styles_using_tokens_only():
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert ".check-card" in css
    body = css.split("* { box-sizing", 1)[1]
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", body)

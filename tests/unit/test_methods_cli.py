import json

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.methods.state import save_state

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


def test_list_shows_every_method_with_status_and_references():
    res = runner.invoke(app, ["methods", "list"])
    assert res.exit_code == 0, res.output
    assert "s5.pbo" in res.output and "cscv@1" in res.output and "default" in res.output
    assert "purged_embargo@1" in res.output and "experimental" in res.output
    assert "Bailey" in res.output


def test_list_filters_by_stage_case_insensitively():
    res = runner.invoke(app, ["methods", "list", "--stage", "s5"])
    assert res.exit_code == 0
    assert "s5.pbo" in res.output and "s5.dsr" in res.output and "s6.bootstrap" not in res.output
    assert runner.invoke(app, ["methods", "list", "--stage", "S9"]).exit_code == 2


def test_list_json_is_machine_readable():
    res = runner.invoke(app, ["methods", "list", "--stage", "S4", "--json"])
    rows = json.loads(res.output)
    assert {(r["id"], r["version"], r["status"]) for r in rows} == {
        ("s4.walkforward", "rolling_anchored@1", "default"), ("s4.walkforward", "purged_embargo@1", "experimental"),
    }
    assert rows[0]["references"] is not None and "shipped_status" in rows[0]


def test_list_reflects_local_promotions(home):
    save_state(home, {"schema_version": 1, "history": [], "statuses": {
        "s4.walkforward": {"purged_embargo@1": "default", "rolling_anchored@1": "deprecated"}}})
    rows = json.loads(runner.invoke(app, ["methods", "list", "--stage", "S4", "--json"]).output)
    by = {r["version"]: r for r in rows}
    assert by["purged_embargo@1"]["status"] == "default" and by["purged_embargo@1"]["shipped_status"] == "experimental"
    assert "local" in runner.invoke(app, ["methods", "list", "--stage", "S4"]).output


def test_show_accepts_both_reference_forms():
    for ref in ("s5.pbo@cscv@1", "s5.pbo=cscv@1"):
        res = runner.invoke(app, ["methods", "show", ref])
        assert res.exit_code == 0, res.output
        assert "cscv@1" in res.output and "doi" in res.output.lower() and "verified" in res.output.lower()


def test_show_unknown_is_blocked():
    for ref in ("s5.pbo@cscv@9", "nope.x@a@1", "garbage"):
        res = runner.invoke(app, ["methods", "show", ref])
        assert res.exit_code == 2 and "BLOCKED" in res.output


def test_a_corrupt_state_file_blocks_listing(home):
    (home / "methods").mkdir(parents=True)
    (home / "methods" / "state.json").write_text("{nope")
    res = runner.invoke(app, ["methods", "list"])
    assert res.exit_code == 2 and "BLOCKED" in res.output

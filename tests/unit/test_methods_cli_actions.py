import json

import pytest
from typer.testing import CliRunner

from hook_helpers import human_approve_id
from qrh.cli import app
from qrh.methods.state import load_state

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


def test_deprecate_waits_for_a_human_then_applies(home):
    args = ["methods", "deprecate", "s4.walkforward@purged_embargo@1", "--reason", "superseded by a better design"]
    first = runner.invoke(app, args)
    assert first.exit_code == 2 and "qrh approve" in first.output
    approval_id = next(w for w in first.output.replace(".", " ").split() if len(w) == 12 and all(c in "0123456789abcdef" for c in w))
    assert load_state(home)["statuses"] == {}
    human_approve_id(home, approval_id, "I read the bench report")  # the human, in their own terminal
    done = runner.invoke(app, args)
    assert done.exit_code == 0 and "applied" in done.output
    rows = json.loads(runner.invoke(app, ["methods", "list", "--stage", "S4", "--json"]).output)
    assert {r["version"]: r["status"] for r in rows}["purged_embargo@1"] == "deprecated"


def test_refusals_exit_2_with_the_reasons(home):
    res = runner.invoke(app, ["methods", "promote", "s4.walkforward@purged_embargo@1", "--bench", "bench-nope",
                              "--reason", "measured win on every suite"])
    assert res.exit_code == 2 and "BLOCKED" in res.output and "bench-nope" in res.output
    res = runner.invoke(app, ["methods", "demote", "s4.walkforward@rolling_anchored@1", "--reason", "rolling back for a reason"])
    assert res.exit_code == 2 and "restore" in res.output
    res = runner.invoke(app, ["methods", "deprecate", "s4.walkforward@rolling_anchored@1", "--reason", "no longer wanted at all"])
    assert res.exit_code == 2 and "experimental" in res.output


def test_a_quick_bench_cannot_promote(home):
    run = runner.invoke(app, ["bench", "run", "--suite", "noise-fpr", "--reps", "5", "--method",
                              "s4.walkforward=purged_embargo@1", "--compare-to", "default", "--json"])
    bench_id = json.loads(run.stdout)["bench_id"]
    res = runner.invoke(app, ["methods", "promote", "s4.walkforward@purged_embargo@1", "--bench", bench_id,
                              "--reason", "measured win on every suite"])
    assert res.exit_code == 2 and "full profile" in res.output
    assert not (home / "approvals").exists()

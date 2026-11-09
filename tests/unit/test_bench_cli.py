import json

import pytest
from typer.testing import CliRunner

from qrh.bench.receipt import list_receipts
from qrh.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    h = tmp_path / "qrh_home"
    monkeypatch.setenv("QRH_HOME", str(h))
    return h


def run(*args):
    return runner.invoke(app, ["bench", "run", "--suite", "noise-fpr", "--reps", "6", *args])


def test_run_writes_receipt_and_reports_and_prints_the_summary(home):
    res = run("--seed", "5")
    assert res.exit_code == 0, res.output
    ids = [i for i, _ in list_receipts(home)]
    assert len(ids) == 1 and ids[0] in res.output
    folder = home / "bench" / ids[0]
    assert {p.name for p in folder.iterdir()} == {"receipt.json", "report.txt", "report.pdf", "chart.json"}
    assert "Noise false positives" in res.output and "nominal" in res.output


def test_default_profile_is_quick_and_full_is_a_flag(home):
    assert json.loads(run("--seed", "1", "--json").stdout)["receipt"]["profile"] == "quick"
    bad = runner.invoke(app, ["bench", "run", "--quick", "--full"])
    assert bad.exit_code == 2 and "BLOCKED" in bad.output


def test_method_override_and_compare_to_default(home):
    res = run("--method", "s4.walkforward=purged_embargo@1", "--compare-to", "default")
    assert res.exit_code == 0, res.output
    assert "EXPERIMENTAL" in res.output and "Default vs candidate" in res.output
    doc = json.loads(run("--method", "s4.walkforward=purged_embargo@1", "--compare-to", "default", "--json").stdout)
    assert doc["receipt"]["overrides"] == {"s4.walkforward": "purged_embargo@1"}
    assert doc["receipt"]["baseline_results"] is not None


@pytest.mark.parametrize("extra", [
    ["--method", "s5.pbo=cscv@9"], ["--method", "nonsense"], ["--compare-to", "default"], ["--compare-to", "other"],
    ["--method", "s5.pbo=cscv@1", "--method", "s5.pbo=cscv@1"],
])
def test_bad_options_are_blocked(extra):
    res = run(*extra)
    assert res.exit_code == 2 and "BLOCKED" in res.output


def test_unknown_suite_and_missing_history_path_are_blocked():
    assert runner.invoke(app, ["bench", "run", "--suite", "bogus"]).exit_code == 2
    res = runner.invoke(app, ["bench", "run", "--suite", "history", "--method", "s4.walkforward=purged_embargo@1"])
    assert res.exit_code == 2 and "history" in res.output
    res = runner.invoke(app, ["bench", "run", "--suite", "history", "--history", "/nonexistent/home",
                              "--method", "s4.walkforward=purged_embargo@1"])
    assert res.exit_code == 2 and "no receipts" in res.output


def test_compare_two_receipts(home):
    a = json.loads(run("--seed", "1", "--json").stdout)
    b = json.loads(run("--seed", "2", "--json").stdout)
    ida, idb = (r["bench_id"] for r in (a, b))
    res = runner.invoke(app, ["bench", "compare", ida, idb])
    assert res.exit_code == 0, res.output
    assert "delta" in res.output and "p=" in res.output and "seed" in res.output
    as_json = json.loads(runner.invoke(app, ["bench", "compare", ida, idb, "--all", "--json"]).output)
    assert as_json["rows"] and as_json["a"] == ida


def test_compare_unknown_receipt_is_blocked():
    res = runner.invoke(app, ["bench", "compare", "bench-nope", "bench-nada"])
    assert res.exit_code == 2 and "BLOCKED" in res.output


def test_list_shows_stored_receipts(home):
    run("--seed", "1")
    res = runner.invoke(app, ["bench", "list"])
    assert res.exit_code == 0 and "bench-" in res.output


def test_progress_goes_to_stderr_and_never_into_json_output(home):
    res = runner.invoke(app, ["bench", "run", "--suite", "noise-fpr", "--reps", "5", "--json"])
    json.loads(res.stdout)  # stdout stays pure JSON
    assert "noise-fpr" in res.stderr


def test_compare_with_one_receipt_compares_its_default_and_candidate_arms(home):
    doc = json.loads(run("--method", "s4.walkforward=purged_embargo@1", "--compare-to", "default", "--json").stdout)
    res = runner.invoke(app, ["bench", "compare", doc["bench_id"]])
    assert res.exit_code == 0, res.output
    assert "default (a) vs candidate (b)" in res.output and "rolling_anchored@1 -> purged_embargo@1" in res.output
    plain = json.loads(run("--seed", "9", "--json").stdout)
    blocked = runner.invoke(app, ["bench", "compare", plain["bench_id"]])
    assert blocked.exit_code == 2 and "baseline" in blocked.output


def test_report_re_renders_from_a_stored_receipt_byte_for_byte(home, tmp_path):
    doc = json.loads(run("--seed", "3", "--json").stdout)
    folder = home / "bench" / doc["bench_id"]
    original = {n: (folder / n).read_bytes() for n in ("report.txt", "report.pdf", "chart.json")}
    out = tmp_path / "again"
    res = runner.invoke(app, ["bench", "report", str(folder / "receipt.json"), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert {n: (out / n).read_bytes() for n in original} == original
    default = runner.invoke(app, ["bench", "report", doc["bench_id"]])  # in place
    assert default.exit_code == 0 and str(folder) in default.output
    assert runner.invoke(app, ["bench", "report", "bench-nope"]).exit_code == 2

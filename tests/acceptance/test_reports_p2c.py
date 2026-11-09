"""Visual reports are produced after each stage, from receipts and artifacts only."""

import json
import re

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.hashing import sha256_file
from qrh.pipeline.receipt import Verdict

runner = CliRunner()
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")


def rdir(ex):
    return ex.home / "reports" / ex.trial_id


def manifest(ex):
    return json.loads((rdir(ex) / "manifest.json").read_text())


def pages(path):
    return len(re.findall(rb"/Type /Page\b(?!s)", path.read_bytes()))


@pytest.fixture(autouse=True)
def reports_on(monkeypatch):
    monkeypatch.delenv("QRH_REPORTS", raising=False)


@pytest.fixture
def ran01(shared_01, monkeypatch):
    """The shared example-01 run, with its artifacts and reports restored after each test."""
    monkeypatch.setenv("QRH_HOME", str(shared_01.home))
    snapshot = {p: p.read_bytes() for p in (shared_01.home / "trials" / shared_01.trial_id).rglob("*") if p.is_file()}
    yield shared_01
    for path, blob in snapshot.items():
        path.write_bytes(blob)
    assert runner.invoke(app, ["report", shared_01.trial_id]).exit_code == 0


def test_01_produces_every_stage_pdf_chart_json_combined_pdf_and_manifest(ran01):
    d = rdir(ran01)
    for s in STAGES:
        assert (d / f"{s}.pdf").is_file() and (d / f"{s}.chart.json").is_file(), s
    combined = d / f"{ran01.trial_id}.pdf"
    assert combined.is_file() and pages(combined) == sum(pages(d / f"{s}.pdf") for s in STAGES)
    # 8 stages x (pdf + chart json), the combined pdf and the 3 files of the readiness report (P3d-b2)
    assert len(manifest(ran01)["files"]) == 8 * 2 + 1 + 3 and manifest(ran01)["errors"] == []


def test_01_manifest_lists_sha256_and_the_receipt_hash_each_file_was_built_from(ran01):
    d = rdir(ran01)
    for f in manifest(ran01)["files"]:
        assert sha256_file(d / f["path"]) == f["sha256"], f
        expected = ran01.receipt("S7" if f["stage"] in ("combined", "readiness") else f["stage"]).receipt_hash()
        assert f["receipt_hash"] == expected, f


def test_01_chart_data_carries_the_receipt_hash_and_figure_types(ran01):
    doc = json.loads((rdir(ran01) / "S5.chart.json").read_text())
    assert doc["receipt_hash"] == ran01.receipt("S5").receipt_hash()
    assert [f["chart_type"] for f in doc["figures"]] == ["histogram", "bar", "table"]  # the table is the stability page
    s3 = json.loads((rdir(ran01) / "S3.chart.json").read_text())
    assert [f["key"] for f in s3["figures"]] == ["s3_heatmap", "s3_hist"]  # 01 has exactly two parameters
    s2 = json.loads((rdir(ran01) / "S2.chart.json").read_text())
    assert [f["key"] for f in s2["figures"]] == ["s2_lint", "s2_truncation_0", "s2_truncation_1", "s2_truncation_2", "s2_sanity"]


def test_04_renders_up_to_s5_plus_the_s7_summary_with_fail(example):
    ex = example("04_overfit_grid", "prices")
    assert ex.run().exit_code == 2
    d = rdir(ex)
    for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S7"):
        assert (d / f"{s}.pdf").is_file(), s
    assert not (d / "S6.pdf").exists() and (d / f"{ex.trial_id}.pdf").is_file()
    s3 = json.loads((d / "S3.chart.json").read_text())
    assert [f["key"] for f in s3["figures"]] == ["s3_ranked", "s3_hist"]  # 120 combinations over 3 parameters: ranked, not a heatmap
    s7 = json.loads((d / "S7.chart.json").read_text())["figures"][0]
    assert s7["final"]["verdict"] == "FAIL" and s7["final"]["failing_stage"] == "S5"
    assert {r["stage"]: r["verdict"] for r in s7["stages"]}["S6"] == "NOT RUN"


def test_a_blocked_stage_still_gets_a_status_page(example):
    ex = example("01_clean_strategy", "prices")
    p = ex.dir / "hypothesis.yaml"
    p.write_text(p.read_text().replace("walk_forward: {train: 400, test: 100}\n", ""))
    ex.new_trial()
    ex.run()
    s4 = json.loads((rdir(ex) / "S4.chart.json").read_text())
    assert ex.receipt("S4").verdict is Verdict.BLOCKED
    assert [f["key"] for f in s4["figures"]] == ["status_card"] and s4["figures"][0]["verdict"] == "BLOCKED"


def test_rendering_failure_never_changes_a_verdict_and_surfaces_in_status(example, monkeypatch):
    from qrh.reports import charts

    def boom(data):
        raise RuntimeError("chart exploded")

    monkeypatch.setattr(charts, "s4_equity", boom)
    ex = example("01_clean_strategy", "prices")
    res = ex.run()
    assert res.exit_code == 0 and all(ex.receipt(s).verdict is Verdict.PASS for s in STAGES)
    errors = manifest(ex)["errors"]
    assert [e["stage"] for e in errors] == ["S4", "combined"] and "chart exploded" in errors[0]["error"]
    assert not (rdir(ex) / "S4.pdf").exists() and (rdir(ex) / "S5.pdf").is_file()
    status = runner.invoke(app, ["status", ex.trial_id])
    assert status.exit_code == 0
    assert "report error S4" in status.output and "chart exploded" in status.output


def test_status_summarizes_reports(ran01):
    out = runner.invoke(app, ["status", ran01.trial_id]).output
    assert "reports: 20 file(s), 0 error(s)" in out


def test_report_command_rerenders_identical_files_from_persisted_artifacts(ran01):
    before = {f["path"]: f["sha256"] for f in manifest(ran01)["files"]}
    import shutil

    shutil.rmtree(rdir(ran01))
    res = runner.invoke(app, ["report", ran01.trial_id])
    assert res.exit_code == 0, res.output
    assert {f["path"]: f["sha256"] for f in manifest(ran01)["files"]} == before
    assert str(rdir(ran01) / f"{ran01.trial_id}.pdf") in res.output


def test_report_command_renders_a_single_stage(ran01):
    import shutil

    shutil.rmtree(rdir(ran01))
    res = runner.invoke(app, ["report", ran01.trial_id, "--stage", "S4"])
    assert res.exit_code == 0, res.output
    assert sorted(p.name for p in rdir(ran01).iterdir()) == ["S4.chart.json", "S4.pdf", "manifest.json"]


def test_report_command_with_open_launches_the_pdf(ran01, monkeypatch):
    opened = []
    monkeypatch.setattr("typer.launch", lambda target, *a, **k: opened.append(target) or 0)
    assert runner.invoke(app, ["report", ran01.trial_id, "--stage", "S5", "--open"]).exit_code == 0
    assert runner.invoke(app, ["report", ran01.trial_id, "--open"]).exit_code == 0
    assert opened == [str(rdir(ran01) / "S5.pdf"), str(rdir(ran01) / f"{ran01.trial_id}.pdf")]


def test_report_command_blocks_on_a_tampered_artifact(ran01):
    rel = ran01.receipt("S5").params["artifacts"]["dsr_inputs"]["path"]
    (ran01.home / rel).write_text('{"dsr": 1}')
    res = runner.invoke(app, ["report", ran01.trial_id, "--stage", "S5"])
    assert res.exit_code == 2 and "BLOCKED" in res.output and "hash" in res.output
    assert not (rdir(ran01) / "S5.pdf").exists()
    full = runner.invoke(app, ["report", ran01.trial_id])
    assert full.exit_code == 2 and (rdir(ran01) / "S4.pdf").is_file()  # the other stages are still rendered
    assert not (rdir(ran01) / f"{ran01.trial_id}.pdf").exists()  # but no combined PDF from unverified data


def test_report_command_blocks_on_a_missing_artifact(ran01):
    rel = ran01.receipt("S4").params["artifacts"]["oos_equity"]["path"]
    (ran01.home / rel).unlink()
    res = runner.invoke(app, ["report", ran01.trial_id, "--stage", "S4"])
    assert res.exit_code == 2 and "cannot be read" in res.output


def test_report_command_unknown_trial_and_unknown_stage(ran01):
    assert runner.invoke(app, ["report", "nope"]).exit_code == 2
    assert runner.invoke(app, ["report", ran01.trial_id, "--stage", "S9"]).exit_code == 2


def test_a_rerun_drops_reports_of_stages_that_no_longer_exist(example):
    ex = example("04_overfit_grid", "prices")
    ex.run()
    stale = rdir(ex) / "S6.pdf"
    stale.write_bytes(b"%PDF stale")
    ex.run()
    assert not stale.exists()

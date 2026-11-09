import json
import re

import pandas as pd
import pytest

from qrh.hashing import sha256_file
from qrh.pipeline.artifacts import Artifacts
from qrh.pipeline.receipt import Receipt, Stage, Verdict, receipt_path, write_receipt
from qrh.reports.inputs import ReportInputError
from qrh.reports.manifest import clear_from, manifest_path, read_manifest
from qrh.reports.render import render_after_stage, render_stage, render_trial

from report_data import TRIAL, build


class Ctx:
    def __init__(self, home):
        self.home, self.trial_id = home, TRIAL


def persist(home, stage, verdict=Verdict.PASS, variant=None, with_artifacts=True):
    receipt, art = build(stage, verdict, variant)
    arts = Artifacts(Ctx(home), Stage(stage))
    if with_artifacts:
        for name, obj in art.items():
            arts.frame(name, obj) if isinstance(obj, pd.DataFrame) else arts.json(name, obj)
    receipt = receipt.model_copy(update={"params": {**receipt.params, "artifacts": arts.as_params()}})
    write_receipt(receipt, home)
    return receipt


def persist_all(home, failing=None):
    out = {}
    for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
        v = Verdict.FAIL if s == failing else Verdict.PASS
        out[s] = persist(home, s, v)
        if s == failing:
            break
    s7 = Receipt(trial_id=TRIAL, stage=Stage.S7, hypothesis_hash="h", code_hash="c",
                 verdict=Verdict.FAIL if failing else Verdict.PASS, params={"failing_stage": failing})
    write_receipt(s7, home)
    return out


def pdf_pages(path):
    return len(re.findall(rb"/Type /Page\b(?!s)", path.read_bytes()))


def test_render_stage_writes_pdf_chart_json_and_manifest(tmp_path):
    receipt = persist(tmp_path, "S4")
    files = render_stage(TRIAL, "S4", home=tmp_path)
    d = tmp_path / "reports" / TRIAL
    assert (d / "S4.pdf").is_file() and (d / "S4.chart.json").is_file() and pdf_pages(d / "S4.pdf") == 3
    doc = json.loads((d / "S4.chart.json").read_text())
    assert doc["trial_id"] == TRIAL and doc["stage"] == "S4" and doc["receipt_hash"] == receipt.receipt_hash()
    assert [f["key"] for f in doc["figures"]] == ["s4_folds", "s4_equity", "s4_params"]
    man = read_manifest(tmp_path, TRIAL)
    assert {(f["stage"], f["kind"], f["path"]) for f in man["files"]} == {
        ("S4", "pdf", "S4.pdf"), ("S4", "chart_data", "S4.chart.json")}
    for f in man["files"]:
        assert f["sha256"] == sha256_file(d / f["path"]) and f["receipt_hash"] == receipt.receipt_hash()
    assert man["errors"] == [] and {f.path.name for f in files} == {"S4.pdf", "S4.chart.json"}


def test_rendering_twice_is_byte_identical(tmp_path):
    persist(tmp_path, "S5")
    d = tmp_path / "reports" / TRIAL
    render_stage(TRIAL, "S5", home=tmp_path)
    first = {p.name: p.read_bytes() for p in d.iterdir()}
    render_stage(TRIAL, "S5", home=tmp_path)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == first


def test_render_trial_writes_stage_pdfs_combined_pdf_and_manifest(tmp_path):
    persist_all(tmp_path)
    render_trial(TRIAL, home=tmp_path)
    d = tmp_path / "reports" / TRIAL
    stage_pdfs = [d / f"S{i}.pdf" for i in range(8)]
    assert all(p.is_file() for p in stage_pdfs)
    combined = d / f"{TRIAL}.pdf"
    assert pdf_pages(combined) == sum(pdf_pages(p) for p in stage_pdfs)
    man = read_manifest(tmp_path, TRIAL)
    kinds = {(f["stage"], f["kind"]) for f in man["files"]}
    assert ("combined", "pdf") in kinds and ("S7", "chart_data") in kinds and len(man["files"]) == 8 * 2 + 1
    s7 = next(f for f in man["files"] if f["stage"] == "combined")
    assert s7["receipt_hash"] == json.loads(receipt_path(tmp_path, TRIAL, "S7").read_text())["receipt_hash"]


def test_a_failed_run_renders_only_the_stages_that_ran_plus_s7(tmp_path):
    persist_all(tmp_path, failing="S5")
    render_trial(TRIAL, home=tmp_path)
    d = tmp_path / "reports" / TRIAL
    assert not (d / "S6.pdf").exists() and (d / "S5.pdf").is_file() and (d / "S7.pdf").is_file()


def test_tampered_artifact_fails_closed_and_is_recorded(tmp_path):
    receipt = persist(tmp_path, "S5")
    render_stage(TRIAL, "S5", home=tmp_path)
    rel = receipt.params["artifacts"]["dsr_inputs"]["path"]
    (tmp_path / rel).write_text('{"dsr": 1}')
    with pytest.raises(ReportInputError, match="hash"):
        render_stage(TRIAL, "S5", home=tmp_path)
    man = read_manifest(tmp_path, TRIAL)
    assert [e["stage"] for e in man["errors"]] == ["S5"] and "hash" in man["errors"][0]["error"]
    assert not (tmp_path / "reports" / TRIAL / "S5.pdf").exists()  # no PDF built from other data stays behind
    assert not [f for f in man["files"] if f["stage"] == "S5"]


def test_successful_render_clears_an_earlier_error(tmp_path):
    receipt = persist(tmp_path, "S5")
    rel = receipt.params["artifacts"]["dsr_inputs"]["path"]
    good = (tmp_path / rel).read_text()
    (tmp_path / rel).write_text('{"dsr": 1}')
    with pytest.raises(ReportInputError):
        render_stage(TRIAL, "S5", home=tmp_path)
    (tmp_path / rel).write_text(good)
    render_stage(TRIAL, "S5", home=tmp_path)
    assert read_manifest(tmp_path, TRIAL)["errors"] == []


def test_stage_that_recorded_no_artifacts_gets_a_status_card(tmp_path):
    persist(tmp_path, "S3", Verdict.BLOCKED, with_artifacts=False)
    render_stage(TRIAL, "S3", home=tmp_path)
    doc = json.loads((tmp_path / "reports" / TRIAL / "S3.chart.json").read_text())
    assert [f["key"] for f in doc["figures"]] == ["status_card"] and doc["figures"][0]["verdict"] == "BLOCKED"


def test_missing_receipt_is_an_error(tmp_path):
    with pytest.raises(ReportInputError, match="no receipt"):
        render_stage(TRIAL, "S4", home=tmp_path)


def test_render_after_stage_never_raises_and_records_the_error(tmp_path):
    receipt = persist(tmp_path, "S5")
    rel = receipt.params["artifacts"]["dsr_inputs"]["path"]
    (tmp_path / rel).write_text("{}")
    render_after_stage(tmp_path, None, receipt)  # must not raise
    assert read_manifest(tmp_path, TRIAL)["errors"][0]["stage"] == "S5"


def test_render_after_stage_survives_an_unwritable_reports_dir(tmp_path):
    receipt = persist(tmp_path, "S4")
    (tmp_path / "reports").write_text("not a directory")
    render_after_stage(tmp_path, None, receipt)  # must not raise


def test_clear_from_removes_stage_outputs_and_manifest_entries(tmp_path):
    persist_all(tmp_path)
    render_trial(TRIAL, home=tmp_path)
    clear_from(tmp_path, TRIAL, Stage.S1)
    d = tmp_path / "reports" / TRIAL
    assert (d / "S0.pdf").is_file() and not (d / "S1.pdf").exists() and not (d / f"{TRIAL}.pdf").exists()
    assert {f["stage"] for f in read_manifest(tmp_path, TRIAL)["files"]} == {"S0"}


def test_manifest_is_canonical_and_free_of_timestamps(tmp_path):
    persist(tmp_path, "S4")
    render_stage(TRIAL, "S4", home=tmp_path)
    text = manifest_path(tmp_path, TRIAL).read_text()
    assert json.loads(text)["schema_version"] == 1 and not re.search(r"20\d\d-\d\d-\d\dT", text)


def test_registry_index_is_checked_when_given(tmp_path):
    persist(tmp_path, "S4")

    class Index:
        def get_receipt_entry(self, trial_id, stage):
            return {"receipt_hash": "0" * 64}

    with pytest.raises(ReportInputError, match="registry"):
        render_stage(TRIAL, "S4", home=tmp_path, registry=Index())

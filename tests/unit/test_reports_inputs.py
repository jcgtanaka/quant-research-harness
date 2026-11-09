import pytest

from qrh.pipeline.artifacts import Artifacts
from qrh.pipeline.receipt import Receipt, Stage, Verdict, write_receipt
from qrh.reports.inputs import ReportInputError, StageData, load_stage_data


class Ctx:
    def __init__(self, home):
        self.home, self.trial_id = home, "t1"


def write_stage(home, stage="S3", with_artifacts=True, tamper=False):
    arts = Artifacts(Ctx(home), Stage(stage))
    arts.json("combos", {"combos": []})
    params = {"artifacts": arts.as_params()} if with_artifacts else {}
    r = Receipt(trial_id="t1", stage=Stage(stage), hypothesis_hash="h", code_hash="c", verdict=Verdict.PASS, params=params)
    write_receipt(r, home)
    if tamper:
        (home / arts.as_params()["combos"]["path"]).write_text('{"combos": [1]}')
    return r


def test_loads_receipt_and_verified_artifacts(tmp_path):
    r = write_stage(tmp_path)
    data = load_stage_data(tmp_path, "t1", "S3")
    assert isinstance(data, StageData)
    assert data.receipt_hash == r.receipt_hash() and data.artifacts == {"combos": {"combos": []}}
    assert data.stage is Stage.S3 and data.trial_id == "t1"


def test_tampered_artifact_is_an_input_error(tmp_path):
    write_stage(tmp_path, tamper=True)
    with pytest.raises(ReportInputError, match="hash"):
        load_stage_data(tmp_path, "t1", "S3")


def test_missing_receipt_is_an_input_error(tmp_path):
    with pytest.raises(ReportInputError, match="no receipt"):
        load_stage_data(tmp_path, "t1", "S3")


def test_tampered_receipt_is_an_input_error(tmp_path):
    write_stage(tmp_path)
    p = tmp_path / "receipts" / "t1" / "S3.json"
    p.write_text(p.read_text().replace('"PASS"', '"FAIL"'))
    with pytest.raises(ReportInputError, match="receipt"):
        load_stage_data(tmp_path, "t1", "S3")


def test_stage_without_recorded_artifacts_loads_empty(tmp_path):
    write_stage(tmp_path, with_artifacts=False)
    assert load_stage_data(tmp_path, "t1", "S3").artifacts == {}


def test_registry_index_mismatch_is_an_input_error(tmp_path):
    class Index:
        def get_receipt_entry(self, trial_id, stage):
            return {"receipt_hash": "0" * 64}

    write_stage(tmp_path)
    with pytest.raises(ReportInputError, match="registry"):
        load_stage_data(tmp_path, "t1", "S3", registry=Index())


def test_s7_loads_every_present_stage_receipt(tmp_path):
    write_stage(tmp_path, "S3")
    write_stage(tmp_path, "S7", with_artifacts=False)
    data = load_stage_data(tmp_path, "t1", "S7")
    assert set(data.receipts) == {Stage.S3, Stage.S7}

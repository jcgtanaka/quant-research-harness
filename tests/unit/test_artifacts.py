import json
import math

import numpy as np
import pandas as pd
import pytest

from qrh.hashing import sha256_file
from qrh.pipeline.artifacts import ArtifactError, Artifacts, read_artifact
from qrh.pipeline.receipt import Receipt, Stage, Verdict


class Ctx:
    def __init__(self, home):
        self.home, self.trial_id = home, "t1"


def receipt_with(params):
    return Receipt(
        trial_id="t1", stage=Stage.S3, hypothesis_hash="h", code_hash="c", verdict=Verdict.PASS, params=params
    )


def test_json_artifact_is_written_hashed_and_recorded(tmp_path):
    arts = Artifacts(Ctx(tmp_path), Stage.S3)
    arts.json("combos", {"b": [1, 2], "a": np.float64(0.5), "when": pd.Timestamp("2020-01-01", tz="UTC")})
    entry = arts.as_params()["combos"]
    path = tmp_path / entry["path"]
    assert entry["path"] == "trials/t1/S3/combos.json" and entry["hash"] == sha256_file(path)
    doc = json.loads(path.read_text())
    assert doc == {"a": 0.5, "b": [1, 2], "when": "2020-01-01T00:00:00+00:00"}


def test_non_finite_floats_become_null(tmp_path):
    arts = Artifacts(Ctx(tmp_path), Stage.S3)
    arts.json("x", {"v": [1.0, math.nan, math.inf]})
    assert json.loads((tmp_path / arts.as_params()["x"]["path"]).read_text()) == {"v": [1.0, None, None]}


def test_identical_content_gives_identical_hash(tmp_path):
    a, b = Artifacts(Ctx(tmp_path / "a"), Stage.S3), Artifacts(Ctx(tmp_path / "b"), Stage.S3)
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.date_range("2020", periods=2, tz="UTC"))
    for arts in (a, b):
        arts.json("j", {"k": [1, 2, 3]})
        arts.frame("f", df)
    assert a.as_params() == b.as_params()


def test_read_round_trip_and_verification(tmp_path):
    arts = Artifacts(Ctx(tmp_path), Stage.S3)
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.date_range("2020", periods=2, tz="UTC"))
    arts.json("j", {"k": 1})
    arts.frame("f", df)
    r = receipt_with({"artifacts": arts.as_params()})
    assert read_artifact(tmp_path, r, "j") == {"k": 1}
    pd.testing.assert_frame_equal(read_artifact(tmp_path, r, "f"), df, check_freq=False)


def test_tampered_artifact_is_rejected(tmp_path):
    arts = Artifacts(Ctx(tmp_path), Stage.S3)
    arts.json("j", {"k": 1})
    r = receipt_with({"artifacts": arts.as_params()})
    (tmp_path / arts.as_params()["j"]["path"]).write_text('{"k": 2}')
    with pytest.raises(ArtifactError, match="hash"):
        read_artifact(tmp_path, r, "j")


def test_missing_artifact_and_unrecorded_name_are_rejected(tmp_path):
    arts = Artifacts(Ctx(tmp_path), Stage.S3)
    arts.json("j", {"k": 1})
    r = receipt_with({"artifacts": arts.as_params()})
    (tmp_path / arts.as_params()["j"]["path"]).unlink()
    with pytest.raises(ArtifactError, match="cannot be read"):
        read_artifact(tmp_path, r, "j")
    with pytest.raises(ArtifactError, match="not recorded"):
        read_artifact(tmp_path, r, "nope")


def test_path_escaping_the_home_is_rejected(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    r = receipt_with({"artifacts": {"j": {"path": "../outside.json", "hash": sha256_file(outside)}}})
    with pytest.raises(ArtifactError, match="outside"):
        read_artifact(tmp_path / "home", r, "j")

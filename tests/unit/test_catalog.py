import numpy as np
import pandas as pd
import pytest

from qrh.hashing import sha256_file
from qrh.memory import catalog as catalog_mod
from qrh.memory.catalog import Catalog


@pytest.fixture
def cat(tmp_path):
    c = Catalog(tmp_path / "r.sqlite3")
    yield c
    c.close()


def small_frame():
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC", name="timestamp")
    return pd.DataFrame({"close": [1.0, 2.0, np.nan, 4.0], "label": ["a", "b", "c", None]}, index=idx)


def test_register_csv_profile(cat, tmp_path):
    p = tmp_path / "d.csv"
    small_frame().to_csv(p)
    rec = cat.register_dataset(p)
    assert rec["name"] == "d"
    assert rec["content_hash"] == sha256_file(p)
    assert rec["row_count"] == 4
    assert rec["null_rates"]["close"] == 0.25 and rec["null_rates"]["label"] == 0.25
    assert rec["null_rates"]["timestamp"] == 0.0
    assert set(rec["schema"]) == {"timestamp", "close", "label"}
    assert rec["schema"]["close"] == "float64"
    assert rec["start"].startswith("2020-01-01") and rec["end"].startswith("2020-01-04")
    assert rec["status"] == "UNCHECKED"


def test_register_parquet_profile(cat, tmp_path):
    p = tmp_path / "d.parquet"
    small_frame().to_parquet(p)
    rec = cat.register_dataset(p, name="named")
    assert rec["name"] == "named" and rec["row_count"] == 4
    assert rec["start"].startswith("2020-01-01")
    assert rec["status"] == "UNCHECKED"


def test_no_time_column_gives_null_range(cat, tmp_path):
    p = tmp_path / "n.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)
    rec = cat.register_dataset(p)
    assert rec["start"] is None and rec["end"] is None and rec["row_count"] == 2


def test_unknown_dataset_is_blocked(cat):
    status, reasons = cat.get_status("nope")
    assert status == "BLOCKED" and reasons


def test_missing_file_is_blocked(cat, tmp_path):
    rec = cat.register_dataset(tmp_path / "gone.csv")
    assert rec["status"] == "BLOCKED" and rec["reasons"]
    assert cat.get_status("gone")[0] == "BLOCKED"


def test_unsupported_extension_is_blocked(cat, tmp_path):
    p = tmp_path / "d.xyz"
    p.write_text("a")
    assert cat.register_dataset(p)["status"] == "BLOCKED"


def test_unparseable_csv_is_blocked(cat, tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("")
    assert cat.register_dataset(p)["status"] == "BLOCKED"


def test_parquet_without_engine_is_blocked(cat, tmp_path, monkeypatch):
    p = tmp_path / "d.parquet"
    small_frame().to_parquet(p)
    monkeypatch.setattr(catalog_mod, "_parquet_available", lambda: False)
    rec = cat.register_dataset(p)
    assert rec["status"] == "BLOCKED" and any("pyarrow" in r for r in rec["reasons"])


def test_hash_change_resets_status(cat, tmp_path):
    p = tmp_path / "d.csv"
    small_frame().to_csv(p)
    cat.register_dataset(p)
    cat.set_status("d", "PASS", [])
    assert cat.get_status("d") == ("PASS", [])
    cat.register_dataset(p)  # unchanged: keeps status
    assert cat.get_status("d")[0] == "PASS"
    small_frame().iloc[:3].to_csv(p)
    rec = cat.register_dataset(p)
    assert rec["status"] == "UNCHECKED" and rec["row_count"] == 3
    assert cat.get_status("d")[0] == "UNCHECKED"


def test_list_datasets(cat, tmp_path):
    for n in ("b", "a"):
        small_frame().to_csv(tmp_path / f"{n}.csv")
        cat.register_dataset(tmp_path / f"{n}.csv")
    assert [d["name"] for d in cat.list_datasets()] == ["a", "b"]


def test_set_status_validates(cat, tmp_path):
    small_frame().to_csv(tmp_path / "d.csv")
    cat.register_dataset(tmp_path / "d.csv")
    with pytest.raises(ValueError):
        cat.set_status("d", "MAYBE", [])
    with pytest.raises(KeyError):
        cat.set_status("zzz", "PASS", [])

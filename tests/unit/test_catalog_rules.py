import pandas as pd
import pytest

from qrh.memory.catalog import Catalog, load_indexed_frame


@pytest.fixture
def cat(tmp_path):
    c = Catalog(tmp_path / "r.sqlite3")
    yield c
    c.close()


def write(tmp_path, name="d"):
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC", name="timestamp")
    p = tmp_path / f"{name}.csv"
    pd.DataFrame({"close": [1.0, 2, 3, 4, 5]}, index=idx).to_csv(p)
    return p


def test_load_indexed_frame(tmp_path):
    df = load_indexed_frame(write(tmp_path))
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None
    assert list(df.columns) == ["close"]


def test_rules_path_stored_and_change_resets_status(cat, tmp_path):
    p = write(tmp_path)
    r1 = tmp_path / "a.rules.yaml"
    r1.write_text("rules: []\n")
    rec = cat.register_dataset(p, rules=r1)
    assert rec["rules_path"] == str(r1)
    cat.set_status("d", "PASS", [])
    assert cat.register_dataset(p, rules=r1)["status"] == "PASS"
    r2 = tmp_path / "b.rules.yaml"
    r2.write_text("rules: []\n")
    rec = cat.register_dataset(p, rules=r2)
    assert rec["rules_path"] == str(r2) and rec["status"] == "UNCHECKED"


def test_rules_lookup_in_home_rules_dir(cat, tmp_path):
    p = write(tmp_path)
    cat.register_dataset(p)
    assert cat.rules_for("d") is None
    (tmp_path / "rules").mkdir()
    f = tmp_path / "rules" / "d.rules.yaml"
    f.write_text("rules: []\n")
    assert cat.rules_for("d") == f

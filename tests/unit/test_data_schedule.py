"""Data-quality check that can run before each run or on demand. It only warns; it never blocks."""
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from qrh.maintenance import data_schedule
from qrh.maintenance.data_schedule import run_data_check
from qrh.memory.catalog import Catalog

TODAY = date(2020, 2, 15)  # the synthetic data ends 2020-02-09 (40 daily rows from 2020-01-01)


def make_frame(n=40):
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC", name="timestamp")
    base = np.linspace(100.0, 120.0, n)
    return pd.DataFrame({"a": base}, index=idx)


def register(home, name="prices", df=None, max_age_days=None):
    home.mkdir(parents=True, exist_ok=True)
    data = home / f"{name}.csv"
    (make_frame() if df is None else df).to_csv(data)
    rules = home / f"{name}.rules.yaml"
    text = "rules:\n  - type: gaps\n    freq: D\n"
    if max_age_days is not None:
        text += f"max_age_days: {max_age_days}\n"
    rules.write_text(text)
    cat = Catalog(home / "registry.sqlite3")
    try:
        cat.register_dataset(data, name=name, rules=rules)
    finally:
        cat.close()


def test_clean_dataset_passes_and_reports_last_date_and_age(tmp_path):
    register(tmp_path)
    rep = run_data_check(tmp_path, ["prices"], TODAY)["prices"]
    assert rep["verdict"] == "PASS" and rep["findings"] == []
    assert rep["last_date"] == "2020-02-09" and rep["age_days"] == 6


def test_gap_gives_warn_or_fail_with_a_plain_finding(tmp_path):
    df = make_frame()
    register(tmp_path, df=df.drop(df.index[[5, 6]]))
    rep = run_data_check(tmp_path, ["prices"], TODAY)["prices"]
    assert rep["verdict"] in ("WARN", "FAIL")
    assert any("missing" in f.lower() and "2020-01-06" in f for f in rep["findings"])
    assert any("what to do" in f.lower() or "fix" in f.lower() for f in rep["findings"])


def test_old_data_warns_and_names_its_age(tmp_path):
    register(tmp_path, max_age_days=3)
    rep = run_data_check(tmp_path, ["prices"], TODAY)["prices"]
    assert rep["verdict"] == "WARN" and rep["age_days"] == 6
    assert any("6 days old" in f and "3" in f for f in rep["findings"])


def test_data_within_the_limit_has_no_age_warning(tmp_path):
    register(tmp_path, max_age_days=10)
    assert run_data_check(tmp_path, ["prices"], TODAY)["prices"]["verdict"] == "PASS"


def test_unknown_dataset_is_a_plain_warning_not_an_exception(tmp_path):
    rep = run_data_check(tmp_path, ["nope"], TODAY)["nope"]
    assert rep["verdict"] == "WARN" and "nope" in " ".join(rep["findings"])
    assert rep["last_date"] is None and rep["age_days"] is None


def test_never_raises_for_missing_or_broken_data(tmp_path):
    register(tmp_path)
    (tmp_path / "prices.csv").unlink()
    rep = run_data_check(tmp_path, ["prices"], TODAY)["prices"]
    assert rep["verdict"] == "WARN" and rep["findings"]
    # no registry at all, and an empty name list
    empty = tmp_path / "empty"
    assert run_data_check(empty, ["x"], TODAY)["x"]["verdict"] == "WARN"
    assert run_data_check(empty, [], TODAY) == {}


def test_writes_json_and_text_reports(tmp_path):
    register(tmp_path)
    rep = run_data_check(tmp_path, ["prices"], TODAY)["prices"]
    out = tmp_path / "maintenance"
    assert json.loads((out / "data-prices.json").read_text()) == rep
    text = (out / "data-prices.txt").read_text()
    assert "PASS" in text and "2020-02-09" in text
    assert not [p for p in out.iterdir() if p.name.endswith(".tmp")]


def test_write_is_atomic_a_failed_write_keeps_the_old_report(tmp_path, monkeypatch):
    register(tmp_path)
    run_data_check(tmp_path, ["prices"], TODAY)
    out = tmp_path / "maintenance" / "data-prices.json"
    before = out.read_text()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    rep = run_data_check(tmp_path, ["prices"], date(2020, 3, 1))["prices"]  # still returns, never raises
    assert rep["verdict"] in ("PASS", "WARN")
    assert out.read_text() == before
    assert not [p for p in out.parent.iterdir() if p.name.endswith(".tmp")]


def test_unsafe_dataset_name_cannot_escape_the_maintenance_folder(tmp_path):
    run_data_check(tmp_path, ["../evil"], TODAY)
    assert not (tmp_path.parent / "evil.json").exists()
    assert data_schedule  # module imported

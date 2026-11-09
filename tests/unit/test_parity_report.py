"""Tests for the parity report (synthetic data only)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qrh.parity import report as pr
from qrh.parity.report import (
    LiveDataError,
    build_parity_report,
    load_live_fills,
    write_parity_report,
)

TOL = 0.0005  # 0.05 percent, a small fraction of a percent


def _backtest(n=5):
    return [
        {"date": f"2024-01-{d:02d}", "instrument": "SYN", "quantity": 10, "price": 100.0 + d}
        for d in range(1, n + 1)
    ]


def _live(n=5, price_shift=0.0, qty=10):
    return [
        {
            "time": f"2024-01-{d:02d}T15:30:00",
            "instrument": "SYN",
            "quantity": qty,
            "price": 100.0 + d + price_shift,
            "fees": 1.0,
        }
        for d in range(1, n + 1)
    ]


def _report(bt=None, live=None, days=None, **kw):
    return build_parity_report(
        bt if bt is not None else _backtest(),
        live if live is not None else _live(),
        days,
        TOL,
        "trial-1",
        **kw,
    )


def test_match():
    r = _report(strategy_name="demo")
    assert r["status"] == "MATCH"
    assert r["differing_days"] == []
    assert r["trial_id"] == "trial-1"
    assert r["strategy_name"] == "demo"
    assert r["tolerance"]["price_fraction"] == TOL
    assert r["tolerance"]["quantity"] == "exact"
    assert r["days_compared"] == "all"
    assert r["start_date"] == "2024-01-01" and r["end_date"] == "2024-01-05"
    assert r["explanation"]


def test_small_price_gap_inside_tolerance_matches():
    assert _report(live=_live(price_shift=0.01))["status"] == "MATCH"


def test_price_gap_beyond_tolerance_is_different():
    live = _live()
    live[2]["price"] += 5.0
    r = _report(live=live)
    assert r["status"] == "DIFFERENT"
    assert [d["date"] for d in r["differing_days"]] == ["2024-01-03"]
    d = r["differing_days"][0]
    assert d["backtest"]["price"] == 103.0 and d["live"]["price"] == 108.0


def test_quantity_is_exact():
    live = _live()
    live[1]["quantity"] = 11
    r = _report(live=live)
    assert r["status"] == "DIFFERENT"
    assert r["differing_days"][0]["date"] == "2024-01-02"


def test_days_unset_uses_all():
    r = _report(days=None)
    assert r["days_compared"] == "all"


def test_days_number_limits_comparison():
    live = _live()
    live[0]["price"] += 50  # old difference, outside the last 2 days
    r = _report(live=live, days=2)
    assert r["status"] == "MATCH"
    assert r["days_compared"] == 2
    assert r["start_date"] == "2024-01-04" and r["end_date"] == "2024-01-05"


def test_missing_live_data_status_and_warning():
    r = build_parity_report(_backtest(), None, None, TOL, "t")
    assert r["status"] == "MISSING_LIVE_DATA"
    assert r["warnings"] and "live" in r["warnings"][0].lower()
    r2 = build_parity_report(_backtest(), [], 3, TOL, "t")
    assert r2["status"] == "MISSING_LIVE_DATA"


def test_never_raises_on_bad_input():
    r = build_parity_report(None, None, "weird", TOL, "t", live_error="boom")
    assert r["status"] == "MISSING_LIVE_DATA"
    assert "boom" in " ".join(r["warnings"])


def test_load_live_fills_ok(tmp_path):
    p = tmp_path / "fills.csv"
    p.write_text(
        "time,instrument,quantity,price,fees\n2024-01-01T15:30:00,SYN,10,101.5,1.0\n"
    )
    fills = load_live_fills(p)
    assert fills[0]["instrument"] == "SYN" and fills[0]["quantity"] == 10
    assert fills[0]["price"] == 101.5


def test_load_live_fills_missing_file(tmp_path):
    with pytest.raises(LiveDataError) as e:
        load_live_fills(tmp_path / "nope.csv")
    assert "not found" in str(e.value).lower()


@pytest.mark.parametrize(
    "text",
    [
        "time,instrument,quantity\n2024-01-01T15:30:00,SYN,10\n",
        "time,instrument,quantity,price,fees\nnot-a-time,SYN,10,1,0\n",
        "time,instrument,quantity,price,fees\n2024-01-01T15:30:00,SYN,abc,1,0\n",
        "",
    ],
)
def test_malformed_csv_maps_to_missing(tmp_path, text):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    with pytest.raises(LiveDataError) as e:
        load_live_fills(p)
    msg = str(e.value)
    assert msg
    r = build_parity_report(_backtest(), None, None, TOL, "t", live_error=msg)
    assert r["status"] == "MISSING_LIVE_DATA"


def test_built_backtest_difference_is_a_bug_warning():
    own = _backtest()
    bt = _backtest()
    bt[3]["price"] += 0.001  # tiny, but exact comparison applies to own numbers
    r = _report(bt=bt, own_backtest_daily=own)
    assert any("2024-01-04" in w and "bug" in w.lower() for w in r["warnings"])


def test_write_creates_both_files_even_when_missing(tmp_path):
    r = build_parity_report(_backtest(), None, None, TOL, "trial-9")
    paths = write_parity_report(r, tmp_path)
    assert (tmp_path / "parity").is_dir()
    for p in paths:
        assert Path(p).exists()
    js = [p for p in paths if str(p).endswith(".json")][0]
    assert json.loads(Path(js).read_text())["status"] == "MISSING_LIVE_DATA"
    txt = [p for p in paths if str(p).endswith(".txt")][0]
    assert "MISSING_LIVE_DATA" in Path(txt).read_text()


def test_write_is_atomic(tmp_path, monkeypatch):
    calls = []
    real = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        return real(src, dst)

    monkeypatch.setattr(pr.os, "replace", spy)
    write_parity_report(_report(), tmp_path)
    assert len(calls) == 2
    for src, dst in calls:
        assert src != dst and not Path(src).exists() and Path(dst).exists()
    assert not [p for p in (tmp_path / "parity").iterdir() if p.name.endswith(".tmp")]

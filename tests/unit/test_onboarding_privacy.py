"""No row-level value of a dataset ever reaches the profile: planted sentinels must not show up anywhere."""

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.onboarding.profile import build_profile

runner = CliRunner()
SYMBOLS = ["ZQXVNB", "KJWPLM", "QQRTYU", "VBNMXZ", "WWQEAS", "HGFDSA", "PLOKIJ", "MNBVCX", "UYTREW", "ASDFGH", "LKJHGF", "TREWQY"]
SENTINEL_NUMBERS = [0.123457, -0.098765, 0.0314159]  # inside the range: never a minimum, maximum or quantile
FIRST_DAY, LAST_DAY = "1987-06-05", "1990-09-14"


def sentinel_frame():
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range(FIRST_DAY, periods=n, freq="B", tz="UTC", name="timestamp")
    idx = idx[:-1].append(pd.DatetimeIndex([LAST_DAY], tz="UTC"))
    x = rng.normal(0.0, 0.2, n)
    x = np.clip(x, -0.5, 0.5)
    for i, v in zip((17, 111, 207), SENTINEL_NUMBERS):
        x[i] = v
    return pd.DataFrame({
        "symbol": [SYMBOLS[i % 12] for i in range(n)], "root": [SYMBOLS[(i * 7) % 12][:4] for i in range(n)],
        "x": x, "side": ["B", "S"] * (n // 2), "note": [f"{SYMBOLS[i % 12].lower()}-{i}" for i in range(n)],
    }, index=idx)


def forms(v):
    return {str(v), repr(v), f"{v:.6f}", f"{v:.5g}", f"{v:.7g}"}


@pytest.mark.parametrize("no_values", [False, True])
def test_no_planted_value_appears_in_the_profile_json(no_values):
    text = json.dumps(build_profile(sentinel_frame(), no_values=no_values), sort_keys=True)
    for sym in SYMBOLS:
        assert sym not in text and sym.lower() not in text, sym
    for v in SENTINEL_NUMBERS:
        for f in forms(v):
            assert f not in text, (f, no_values)
    if no_values:
        assert FIRST_DAY not in text and LAST_DAY not in text


def test_the_cli_never_prints_a_planted_value(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path / "home"))
    path = tmp_path / "secret.csv"
    sentinel_frame().to_csv(path)
    assert runner.invoke(app, ["data", "register", str(path), "--name", "ds"]).exit_code == 0
    for args in (["data", "profile", "ds"], ["data", "profile", "ds", "--json"], ["data", "profile", "ds", "--json", "--no-values"],
                 ["data", "status", "ds"], ["data", "onboarding", "ds"]):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, res.output
        for sym in SYMBOLS:
            assert sym not in res.output and sym.lower() not in res.output, (args, sym)
        for v in SENTINEL_NUMBERS:
            assert not any(f in res.output for f in forms(v)), (args, v)
        assert str(path) not in runner.invoke(app, ["data", "profile", "ds", "--json"]).output  # not even the location


def test_stored_profiles_hold_no_planted_value(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setenv("QRH_HOME", str(tmp_path / "home"))
    path = tmp_path / "secret.csv"
    sentinel_frame().to_csv(path)
    runner.invoke(app, ["data", "register", str(path), "--name", "ds"])
    runner.invoke(app, ["data", "profile", "ds"])
    runner.invoke(app, ["data", "profile", "ds", "--no-values"])
    con = sqlite3.connect(tmp_path / "home" / "registry.sqlite3")
    blob = " ".join(r[0] for r in con.execute("SELECT profile FROM dataset_profiles"))
    assert blob and not any(s in blob for s in SYMBOLS)


def test_string_shapes_and_the_flag_exception_are_the_only_text_shown():
    cols = build_profile(sentinel_frame())["columns"]
    assert cols["symbol"]["shapes"] == [{"shape": "AAAAAA", "share": 1.0}] and "values" not in cols["symbol"]
    assert cols["side"]["values"] == ["B", "S"]
    assert cols["note"]["distinct"] == 300 and "values" not in cols["note"]

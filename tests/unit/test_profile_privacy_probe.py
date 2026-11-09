"""Regression for the privacy probe: planted symbols under neutral names must never reach the profile output."""

import csv
import datetime as dt
import json
import random

import pytest
from typer.testing import CliRunner

from qrh.cli import app
from qrh.onboarding.profile import FLAG_VOCABULARIES

runner = CliRunner()
SENTINELS = {"price": "4242.4242", "size": "7777777", "KO": "KO", "BA": "BA", "GE": "GE", "venue": "XNAS",
             "venue2": "XNYS", "contract": "ESZ3", "one-char": '"Q"', "one-char2": '"V"'}


@pytest.fixture
def probe(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path / "home"))
    random.seed(11)
    rows, px = [], 100.0
    for i in range(300):
        px *= 1 + random.gauss(0, 0.01)
        rows.append({
            "timestamp": (dt.datetime(2022, 1, 3) + dt.timedelta(days=i)).strftime("%Y-%m-%d"),
            "px": 4242.4242 if i == 150 else round(px, 4),
            "grp": ["KO", "BA", "GE"][i % 3], "one": ["Q", "V"][i % 2], "venue": ["XNAS", "XNYS", "ARCX", "BATS"][i % 4],
            "side": "CP"[i % 2], "contract": f"ESZ{i % 10}",
            "size": 9000000 if i == 5 else 7777777 if i == 77 else 100 + (i % 50),
        })
    rows[200]["px"], rows[10]["px"] = 9999.9999, 0.5  # the planted price is neither the minimum nor the maximum
    path = tmp_path / "probe.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert runner.invoke(app, ["data", "register", str(path), "--name", "probe", "--privacy", "private"]).exit_code == 0


@pytest.mark.parametrize("extra", [[], ["--no-values"]])
def test_no_planted_sentinel_is_in_the_output(probe, extra):
    out = runner.invoke(app, ["data", "profile", "probe", "--json", *extra])
    assert out.exit_code == 0, out.output
    for label, needle in SENTINELS.items():
        assert needle not in out.output, f"LEAK {label}"
    text = runner.invoke(app, ["data", "profile", "probe", *extra]).output
    assert not any(n.strip('"') in text for n in SENTINELS.values())


def test_only_a_real_flag_column_shows_values_and_only_in_default_mode(probe):
    cols = json.loads(runner.invoke(app, ["data", "profile", "probe", "--json"]).stdout)["profile"]["columns"]
    assert cols["side"]["values"] == ["C", "P"]
    assert [c for c, p in cols.items() if "values" in p] == ["side"]
    assert cols["grp"]["shapes"][0]["shape"] == "AA" and cols["one"]["shapes"][0]["shape"] == "A"
    nv = json.loads(runner.invoke(app, ["data", "profile", "probe", "--json", "--no-values"]).stdout)["profile"]["columns"]
    assert not any("values" in p for p in nv.values())


def test_the_allowlist_has_the_agreed_vocabularies():
    assert {frozenset(v) for v in FLAG_VOCABULARIES} >= {
        frozenset({"C", "P"}), frozenset({"CALL", "PUT"}), frozenset({"Y", "N"}), frozenset({"YES", "NO"}),
        frozenset({"B", "S"}), frozenset({"BUY", "SELL"}), frozenset({"T", "F"}), frozenset({"TRUE", "FALSE"}),
        frozenset({"L", "S"}), frozenset({"LONG", "SHORT"}), frozenset({"BID", "ASK"}), frozenset({"0", "1"})}

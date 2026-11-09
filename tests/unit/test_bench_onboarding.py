"""The onboarding suite: correct proposals must verify without contradiction; planted wrong ones must be
contradicted or left unverified (synthetic datasets, fixed generators, deterministic seeds)."""

import json

import pytest

from qrh.bench.onboarding_corpus import CLASSES, DATASETS, corpus
from qrh.bench.profiles import OPTIONAL_SUITES, PROFILES, SUITES
from qrh.bench.run import run_bench
from qrh.bench.suites import run_onboarding
from qrh.methods.selection import MethodSelector

CFG = {"reps": 1}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    sel = MethodSelector(tmp_path_factory.mktemp("home"))
    return run_onboarding({"candidate": sel}, CFG, seed=5)["candidate"]


def test_the_suite_is_optional_so_the_default_bench_is_unchanged():
    assert "onboarding" in OPTIONAL_SUITES and "onboarding" not in SUITES
    for name in ("quick", "full"):
        assert PROFILES[name]["onboarding"]["reps"] >= 1
    assert PROFILES["full"]["onboarding"]["reps"] > PROFILES["quick"]["onboarding"]["reps"]


def test_the_cli_lists_the_suite():
    from typer.testing import CliRunner

    from qrh.cli import app

    out = CliRunner().invoke(app, ["bench", "run", "--help"]).output
    assert "onboarding" in out


def test_the_corpus_plants_the_five_classes_over_the_five_generators():
    assert CLASSES == ("wrong_units", "wrong_asset_class", "adjustments_none_on_split", "missing_option_mapping",
                       "wrong_frequency")
    assert DATASETS == ("equity", "futures", "fx", "option", "perp")
    cases = corpus()
    for cls in CLASSES:
        assert len([c for c in cases if c.bug_class == cls]) >= 1, cls
    assert {c.name for c in cases if c.bug_class == "controls"} == set(DATASETS)
    names = [(c.bug_class, c.name) for c in cases]
    assert len(set(names)) == len(names)


def test_correct_proposals_have_zero_false_alarms(result):
    m = result["metrics"]["false_alarm/controls"]
    assert m["k"] == 0 and m["n"] == len(DATASETS) and m["direction"] == "lower" and m["gate"]
    for name in result["variants"]["controls"]:
        assert result["metrics"][f"false_alarm/controls/{name}"]["k"] == 0, name


@pytest.mark.parametrize("cls", CLASSES)
def test_every_planted_case_is_contradicted_or_left_unverified(result, cls):
    m = result["metrics"][f"detect/{cls}"]
    assert m["gate"] and m["direction"] == "higher" and m["n"] == len(result["variants"][cls])
    missed = [n for n in result["variants"][cls] if result["metrics"][f"detect/{cls}/{n}"]["k"] != 1]
    assert missed == [] and m["k"] == m["n"]


def test_the_report_carries_the_rates_and_the_status_breakdown(result):
    d = result["detectors"]
    assert set(d) == {*CLASSES, "controls"}
    for cls in CLASSES:
        row = d[cls]
        assert set(row) == {"contradicted", "unverified", "verified"}
        assert sum(row.values()) == len(result["variants"][cls])
    assert d["controls"]["contradicted"] == 0


def test_planted_cases_are_deterministic(result, tmp_path):
    again = run_onboarding({"candidate": MethodSelector(tmp_path)}, CFG, seed=5)["candidate"]
    assert json.dumps(again, sort_keys=True) == json.dumps(result, sort_keys=True)


def test_result_shape_matches_the_planted_bug_suite(result):
    assert result["reps"] == 1 and set(result["variants"]) == {*CLASSES, "controls"}
    assert result["expected_undetectable"] == {}
    json.dumps(result, sort_keys=True)


def test_run_bench_runs_the_suite_and_report_and_compare_work(tmp_path):
    from qrh.bench.compare import compare_receipts
    from qrh.bench.receipt import load_bench_receipt, write_bench
    from qrh.bench.render import render_bench
    from qrh.bench.textreport import format_receipt

    r = run_bench(tmp_path, suites=["onboarding"], profile="quick", config={"onboarding": CFG}, seed=9)
    assert r.suites == ["onboarding"] and r.non_default_methods == {}
    folder = write_bench(r, tmp_path)
    assert load_bench_receipt(folder / "receipt.json").receipt_hash() == r.receipt_hash()
    text = format_receipt(r)
    assert "Dataset onboarding" in text and "false alarm" in text
    for cls in CLASSES:
        assert cls in text
    files = render_bench(r, folder)
    assert {f.name for f in files} == {"report.txt", "report.pdf", "chart.json"} and all(f.stat().st_size > 0 for f in files)
    doc = json.loads((folder / "chart.json").read_text())
    assert doc["figures"][1]["key"] == "onboarding_detection"
    r2 = run_bench(tmp_path, suites=["onboarding"], profile="quick", config={"onboarding": CFG}, seed=10)
    cmp = compare_receipts(r, r2)
    assert any(row["name"] == "false_alarm/controls" for row in cmp["rows"])

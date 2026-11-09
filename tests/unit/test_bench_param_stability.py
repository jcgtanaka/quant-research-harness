"""The param-stability suite: planted spikes must FAIL the check, planted plateaus must PASS it (synthetic, seeded)."""

import json

import pytest

from qrh.bench.profiles import OPTIONAL_SUITES, PROFILES, SUITES
from qrh.bench.run import run_bench
from qrh.bench.stability_corpus import CLASSES, CONTROLS, corpus
from qrh.bench.suites import run_param_stability
from qrh.methods.selection import MethodSelector

CFG = {"reps": 3}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    sel = MethodSelector(tmp_path_factory.mktemp("home"))
    return run_param_stability({"candidate": sel}, CFG, seed=5)["candidate"]


def test_the_suite_is_optional_so_the_default_bench_is_unchanged():
    assert "param-stability" in OPTIONAL_SUITES and "param-stability" not in SUITES
    for name in ("quick", "full"):
        assert PROFILES[name]["param-stability"]["reps"] >= 1
    assert PROFILES["full"]["param-stability"]["reps"] > PROFILES["quick"]["param-stability"]["reps"]


def test_the_corpus_plants_spikes_and_plateaus_for_a_grid_and_for_a_single_combination():
    cases = corpus()
    assert CLASSES == ("spike",) and CONTROLS == "controls"
    spikes = [c for c in cases if c.bug_class == "spike"]
    plateaus = [c for c in cases if c.bug_class == CONTROLS]
    assert {c.shape for c in spikes} == {"grid", "single"} == {c.shape for c in plateaus}
    assert len(spikes) >= 9 and len(plateaus) >= 7
    names = {c.name for c in cases}
    assert {"single_spike_with_inert_parameter", "grid_spike_with_inert_parameter", "grid_spike_hairline_grid_points",
            "single_spike_hairline_declared", "single_spike_string_typed", "grid_spike_string_typed",
            "grid_wide_log_flat", "grid_wide_log_hill"} <= names
    names = [(c.bug_class, c.name) for c in cases]
    assert len(set(names)) == len(names)


def test_every_planted_spike_fails_the_check(result):
    m = result["metrics"]["detect/spike"]
    assert m["gate"] and m["direction"] == "higher" and m["n"] == len(result["variants"]["spike"]) * CFG["reps"]
    missed = [n for n in result["variants"]["spike"] if result["metrics"][f"detect/spike/{n}"]["k"] != CFG["reps"]]
    assert missed == [] and m["k"] == m["n"]


def test_every_planted_plateau_passes_the_check(result):
    m = result["metrics"]["false_alarm/controls"]
    assert m["gate"] and m["direction"] == "lower" and m["n"] == len(result["variants"]["controls"]) * CFG["reps"]
    flagged = [n for n in result["variants"]["controls"] if result["metrics"][f"false_alarm/controls/{n}"]["k"] != 0]
    assert flagged == [] and m["k"] == 0


def test_detector_counts_show_which_measure_flagged(result):
    d = result["detectors"]
    assert set(d) == {"spike", "controls"}
    n = len(result["variants"]["spike"]) * CFG["reps"]
    assert d["controls"] == {"neighbour": 0, "plateau": 0, "not_applicable": 0, "blocked": 0}
    assert d["spike"]["plateau"] <= n
    assert d["spike"]["blocked"] == CFG["reps"]  # the declared hairline neighbours are refused, never passed
    assert d["spike"]["neighbour"] == n - d["spike"]["blocked"]


def test_deterministic_and_json_ready(result, tmp_path):
    again = run_param_stability({"candidate": MethodSelector(tmp_path)}, CFG, seed=5)["candidate"]
    assert json.dumps(again, sort_keys=True) == json.dumps(result, sort_keys=True)
    assert result["methods"]["s5.param_stability"] == "param_stability@1" and result["expected_undetectable"] == {}


def test_run_bench_report_and_charts_work(tmp_path):
    from qrh.bench.receipt import write_bench
    from qrh.bench.render import render_bench
    from qrh.bench.textreport import format_receipt

    r = run_bench(tmp_path, suites=["param-stability"], profile="quick", config={"param-stability": CFG}, seed=9)
    assert r.suites == ["param-stability"] and r.suite_verdicts == {"param-stability": "PASS"} and r.verdict == "PASS"
    text = format_receipt(r)
    assert "Parameter stability" in text and "spike" in text and "false alarm" in text
    folder = write_bench(r, tmp_path)
    files = render_bench(r, folder)
    assert {f.name for f in files} == {"report.txt", "report.pdf", "chart.json"}
    doc = json.loads((folder / "chart.json").read_text())
    assert doc["figures"][1]["key"] == "stability_detection"


def test_the_review_cases_are_each_caught_or_passed_for_the_right_reason(result):
    m = result["metrics"]
    for name in ("single_spike_with_inert_parameter", "grid_spike_with_inert_parameter", "grid_spike_hairline_grid_points",
                 "single_spike_hairline_declared", "single_spike_string_typed", "grid_spike_string_typed"):
        assert m[f"detect/spike/{name}"]["k"] == CFG["reps"], name
    for name in ("grid_wide_log_flat", "grid_wide_log_hill"):
        assert m[f"false_alarm/controls/{name}"]["k"] == 0, name

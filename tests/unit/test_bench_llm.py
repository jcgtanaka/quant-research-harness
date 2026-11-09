"""The LLM planted-bug suite: cutoff violations, unknown cutoffs, undeclared use, prompt look-ahead, clean controls."""

import json

import pytest

from qrh.bench.llm_corpus import LLM_CLASSES, llm_corpus, materialize_llm
from qrh.bench.profiles import OPTIONAL_SUITES, PROFILES, SUITES
from qrh.bench.run import run_bench
from qrh.bench.suites import run_llm_gate
from qrh.methods.selection import MethodSelector

CFG = {"reps": 1}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    sel = MethodSelector(tmp_path_factory.mktemp("home"))
    return run_llm_gate({"candidate": sel}, CFG, seed=5)["candidate"]


def test_the_suite_is_optional_so_the_default_bench_is_unchanged():
    assert "llm-gate" in OPTIONAL_SUITES and "llm-gate" not in SUITES
    for name in ("quick", "full"):
        assert PROFILES[name]["llm-gate"]["reps"] >= 1
    assert PROFILES["full"]["llm-gate"]["reps"] > PROFILES["quick"]["llm-gate"]["reps"]


def test_the_corpus_has_the_four_classes_with_several_variants_and_clean_controls():
    cases = llm_corpus()
    assert LLM_CLASSES == ("llm_cutoff_violation", "llm_cutoff_unknown", "llm_undeclared", "llm_prompt_lookahead",
                           "llm_direct_client")
    for cls in LLM_CLASSES:
        # llm_direct_client has 2 planted variants; its third case, the exempt backend file, is a control
        assert len([c for c in cases if c.bug_class == cls]) >= (2 if cls == "llm_direct_client" else 4), cls
    assert len([c for c in cases if c.bug_class == "controls"]) >= 5
    names = [(c.bug_class, c.name) for c in cases]
    assert len(set(names)) == len(names)


def test_materialized_cases_are_synthetic_and_deterministic_per_seed():
    case = llm_corpus()[0]
    a, b = materialize_llm(case, 7), materialize_llm(case, 7)
    assert a.frame.equals(b.frame) and a.source == b.source and a.spec == b.spec
    assert not a.frame.equals(materialize_llm(case, 8).frame)


def test_controls_have_zero_false_alarms(result):
    m = result["metrics"]["false_alarm/controls"]
    assert m["k"] == 0 and m["n"] >= 5 and m["direction"] == "lower" and m["gate"]
    for name in result["variants"]["controls"]:
        assert result["metrics"][f"false_alarm/controls/{name}"]["k"] == 0, name


@pytest.mark.parametrize("cls", LLM_CLASSES)
def test_every_planted_variant_of_every_class_is_detected(result, cls):
    m = result["metrics"][f"detect/{cls}"]
    assert m["gate"] and m["direction"] == "higher" and m["n"] == len(result["variants"][cls])
    missed = [n for n in result["variants"][cls] if result["metrics"][f"detect/{cls}/{n}"]["k"] != 1]
    assert missed == [] and m["k"] == m["n"]


def test_each_class_is_caught_by_the_detector_that_should_catch_it(result):
    d = result["detectors"]
    assert d["llm_cutoff_violation"]["gate"] == len(result["variants"]["llm_cutoff_violation"])
    assert d["llm_cutoff_unknown"]["gate"] == len(result["variants"]["llm_cutoff_unknown"])
    assert d["llm_undeclared"]["lint"] == len(result["variants"]["llm_undeclared"])
    assert d["llm_prompt_lookahead"]["replay"] >= 1  # at least one leak is caught only by the replay-miss detector
    assert d["llm_direct_client"]["direct"] == len(result["variants"]["llm_direct_client"])
    assert d["controls"] == {"gate": 0, "lint": 0, "direct": 0, "replay": 0, "truncation": 0}


def test_the_replay_miss_detector_catches_prompt_leaks_the_lookahead_lint_cannot_see(result):
    from qrh.checks.lookahead import lint_source

    lint_blind = [c.name for c in llm_corpus() if c.bug_class == "llm_prompt_lookahead"
                  and not [f for f in lint_source(materialize_llm(c, 1).source) if f.severity == "BLOCK"]]
    assert len(lint_blind) >= 3
    for name in lint_blind:
        assert result["metrics"][f"detect/llm_prompt_lookahead/{name}"]["k"] == 1, name


def test_llm002_planted_variants_and_the_exempt_backend_file(result):
    cases = {(c.bug_class, c.name) for c in llm_corpus()}
    assert {("llm_direct_client", "client_import_in_the_adapter_module"),
            ("llm_direct_client", "client_call_in_a_helper_module"), ("controls", "backend_file_is_exempt")} <= cases
    assert len(result["variants"]["llm_direct_client"]) == 2  # the exempt backend file is a control, never a miss
    assert result["metrics"]["false_alarm/controls/backend_file_is_exempt"]["k"] == 0
    for c in llm_corpus():
        if c.bug_class == "controls":
            assert materialize_llm(c, 1).helper is None or "register_backend" in materialize_llm(c, 1).helper


def test_result_shape_matches_the_planted_bug_suite(result):
    assert result["reps"] == 1 and set(result["variants"]) == {*LLM_CLASSES, "controls"}
    assert result["expected_undetectable"] == {} and result["methods"]["s2.llm_gate"] == "llm_gate@1"
    json.dumps(result, sort_keys=True)


def test_run_bench_runs_the_suite_and_the_receipt_round_trips(tmp_path):
    from qrh.bench.receipt import load_bench_receipt, write_bench
    from qrh.bench.render import render_bench
    from qrh.bench.textreport import format_receipt

    r = run_bench(tmp_path, suites=["llm-gate"], profile="quick", config={"llm-gate": CFG}, seed=9)
    assert r.suites == ["llm-gate"] and r.methods["s2.llm_gate"] == "llm_gate@1" and r.non_default_methods == {}
    folder = write_bench(r, tmp_path)
    assert load_bench_receipt(folder / "receipt.json").receipt_hash() == r.receipt_hash()
    text = format_receipt(r)
    assert "LLM contamination" in text
    for cls in LLM_CLASSES:
        assert cls in text
    assert "false alarm" in text
    files = render_bench(r, folder)
    assert {f.name for f in files} == {"report.txt", "report.pdf", "chart.json"} and all(f.stat().st_size > 0 for f in files)
    doc = json.loads((folder / "chart.json").read_text())
    assert doc["figures"][1]["key"] == "llm_detection"

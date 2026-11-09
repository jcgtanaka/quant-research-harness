import json

import pytest

from qrh.bench.receipt import BenchReceiptError, list_receipts, load_bench_receipt, resolve_receipt, write_bench
from qrh.bench.run import run_bench
from qrh.methods.registry import MethodError

CFG = {
    "planted-bugs": {"reps": 1},
    "noise-fpr": {"grids": [12], "periods": [400], "reps": 8},
    "known-edge-power": {"grid": 12, "periods": 600, "strengths": [0.002, 0.008], "reps": 8},
}


def run(home, **kw):
    kw.setdefault("config", CFG)
    return run_bench(home, **kw)


def test_receipt_records_methods_config_seeds_results_and_hashes(tmp_path):
    r = run(tmp_path, suites=["noise-fpr"], profile="quick", seed=11)
    assert r.suites == ["noise-fpr"] and r.profile == "custom" and r.seed == 11
    assert r.methods["s5.pbo"] == "cscv@1" and r.methods["s4.walkforward"] == "rolling_anchored@1"
    assert r.overrides == {} and r.non_default_methods == {} and r.baseline_results is None
    assert set(r.results) == {"noise-fpr"} and r.config["noise-fpr"]["reps"] == 8
    assert r.hashes["config"] and r.hashes["results"] and r.code_hash
    assert r.environment["seeds"] == {"bench": 11}


def test_same_inputs_give_the_same_receipt_hash_and_id(tmp_path):
    a = run(tmp_path, suites=["noise-fpr"], seed=3)
    b = run(tmp_path, suites=["noise-fpr"], seed=3)
    c = run(tmp_path, suites=["noise-fpr"], seed=4)
    assert a.receipt_hash() == b.receipt_hash() and a.bench_id() == b.bench_id()
    assert a.receipt_hash() != c.receipt_hash() and a.bench_id().startswith("bench-")


def test_override_is_recorded_flagged_and_compared_to_the_default(tmp_path):
    r = run(tmp_path, suites=["known-edge-power"], overrides={"s4.walkforward": "purged_embargo@1"}, compare_to_default=True)
    assert r.overrides == {"s4.walkforward": "purged_embargo@1"}
    assert r.methods["s4.walkforward"] == "purged_embargo@1"
    assert r.non_default_methods == {"s4.walkforward": "purged_embargo@1"}
    assert r.baseline_methods["s4.walkforward"] == "rolling_anchored@1"
    assert set(r.baseline_results) == {"known-edge-power"}
    assert r.results["known-edge-power"]["methods"]["s4.walkforward"] == "purged_embargo@1"
    assert r.baseline_results["known-edge-power"]["methods"]["s4.walkforward"] == "rolling_anchored@1"


def test_compare_to_default_needs_an_override_and_overrides_must_exist(tmp_path):
    with pytest.raises(ValueError, match="override"):
        run(tmp_path, suites=["noise-fpr"], compare_to_default=True)
    with pytest.raises(MethodError):
        run(tmp_path, suites=["noise-fpr"], overrides={"s5.pbo": "cscv@9"})
    with pytest.raises(ValueError, match="suite"):
        run(tmp_path, suites=["bogus"])


def test_history_suite_needs_a_path_and_an_override(tmp_path):
    with pytest.raises(ValueError, match="history"):
        run(tmp_path, suites=["history"], overrides={"s4.walkforward": "purged_embargo@1"})
    with pytest.raises(ValueError, match="override"):
        run(tmp_path, suites=["history"], history_path=tmp_path)


def test_write_load_round_trip_and_tamper_detection(tmp_path):
    r = run(tmp_path, suites=["noise-fpr"])
    folder = write_bench(r, tmp_path)
    assert folder == tmp_path / "bench" / r.bench_id() and (folder / "receipt.json").is_file()
    assert load_bench_receipt(folder / "receipt.json").receipt_hash() == r.receipt_hash()
    doc = json.loads((folder / "receipt.json").read_text())
    doc["receipt"]["results"]["noise-fpr"]["reps"] = 999
    (folder / "receipt.json").write_text(json.dumps(doc))
    with pytest.raises(BenchReceiptError):
        load_bench_receipt(folder / "receipt.json")


def test_resolve_by_id_prefix_or_path(tmp_path):
    r = run(tmp_path, suites=["noise-fpr"], seed=1)
    other = run(tmp_path, suites=["noise-fpr"], seed=2)
    f1, f2 = write_bench(r, tmp_path), write_bench(other, tmp_path)
    assert resolve_receipt(tmp_path, r.bench_id()).receipt_hash() == r.receipt_hash()
    assert resolve_receipt(tmp_path, str(f2 / "receipt.json")).receipt_hash() == other.receipt_hash()
    assert resolve_receipt(tmp_path, r.bench_id()[:-2] + "").receipt_hash() == r.receipt_hash()  # unique prefix
    with pytest.raises(BenchReceiptError):
        resolve_receipt(tmp_path, "bench-")  # ambiguous
    with pytest.raises(BenchReceiptError):
        resolve_receipt(tmp_path, "bench-nothere")
    assert {i for i, _ in list_receipts(tmp_path)} == {r.bench_id(), other.bench_id()}
    assert f1 != f2


def test_progress_callback_reports_each_suite_once_in_order(tmp_path):
    seen = []
    run(tmp_path, suites=["noise-fpr", "known-edge-power"], progress=seen.append)
    assert [m.split(":")[0] for m in seen] == ["noise-fpr", "known-edge-power"]
    assert all("rep" in m for m in seen)


def test_an_opt_in_method_can_be_the_candidate_against_a_default_arm_without_it(tmp_path):
    r = run(tmp_path, suites=["planted-bugs"], overrides={"s2.pnl_reconstruction": "pnl_reconstruction@1"}, compare_to_default=True)
    assert r.methods["s2.pnl_reconstruction"] == "pnl_reconstruction@1"
    assert "s2.pnl_reconstruction" not in r.baseline_methods
    assert r.non_default_methods == {"s2.pnl_reconstruction": "pnl_reconstruction@1"}
    cand, base = r.results["planted-bugs"]["metrics"], r.baseline_results["planted-bugs"]["metrics"]
    assert cand["detect/same_bar_execution"]["k"] >= base["detect/same_bar_execution"]["k"]  # the default now detects it too
    det_c, det_b = (r.results["planted-bugs"]["detectors"], r.baseline_results["planted-bugs"]["detectors"])
    assert det_c["same_bar_execution"]["pnl"] > det_b["same_bar_execution"]["pnl"] == 0

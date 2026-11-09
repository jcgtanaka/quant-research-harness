import pytest

from qrh.bench.compare import compare_metrics, compare_receipts, within_receipt
from qrh.bench.stats import metric
from qrh.bench.receipt import BenchReceipt


def results(detect_k, fpr_k, n=100, extra=None):
    m = {
        "detect/lookahead_shift": metric(detect_k, n, direction="higher", gate=True),
        "fpr/g12_t500": metric(fpr_k, n, direction="lower", gate=True),
        "fpr/pooled": metric(fpr_k, n, direction="lower"),
    }
    m.update(extra or {})
    return {"planted-bugs": {"metrics": {k: v for k, v in m.items() if k.startswith("detect")}},
            "noise-fpr": {"metrics": {k: v for k, v in m.items() if k.startswith("fpr")}}}


def receipt(res, methods=None, baseline=None, baseline_methods=None, seed=1, cfg=None):
    return BenchReceipt(
        profile="custom", suites=list(res), seed=seed, config=cfg or {}, methods=methods or {"s5.pbo": "cscv@1"},
        results=res, baseline_results=baseline, baseline_methods=baseline_methods, code_hash="c",
        overrides={"s5.pbo": "cscv@2"} if baseline else {},
    )


def test_rows_use_direction_to_say_which_side_is_better():
    rows = {r["name"]: r for r in compare_metrics(results(60, 10), results(80, 4))}
    detect, fpr = rows["detect/lookahead_shift"], rows["fpr/g12_t500"]
    assert detect["delta"] == pytest.approx(0.2) and detect["better"] == "b" and detect["distinguishable"]
    assert fpr["delta"] == pytest.approx(-0.06) and fpr["better"] == "b"  # lower FPR is better
    assert fpr["suite"] == "noise-fpr" and fpr["gate"] is True


def test_equal_counts_are_a_tie_and_indistinguishable():
    rows = compare_metrics(results(60, 10), results(60, 10))
    assert all(r["better"] == "tie" and not r["distinguishable"] and r["p_value"] == pytest.approx(1.0) for r in rows)


def test_small_visible_difference_is_not_called_distinguishable():
    rows = {r["name"]: r for r in compare_metrics(results(6, 1, n=10), results(8, 0, n=10))}
    assert rows["detect/lookahead_shift"]["better"] == "b" and not rows["detect/lookahead_shift"]["distinguishable"]


def test_only_metrics_present_in_both_are_compared():
    a, b = results(60, 10), results(60, 10, extra={"detect/extra": metric(1, 2, direction="higher")})
    names = {r["name"] for r in compare_metrics(a, b)}
    assert "detect/extra" not in names and "detect/lookahead_shift" in names


def test_compare_receipts_lists_method_differences_and_pairing_notes():
    a = receipt(results(60, 10), methods={"s5.pbo": "cscv@1", "s4.walkforward": "rolling_anchored@1"})
    b = receipt(results(80, 4), methods={"s5.pbo": "cscv@2", "s4.walkforward": "rolling_anchored@1"}, seed=2)
    out = compare_receipts(a, b)
    assert out["method_differences"] == {"s5.pbo": ["cscv@1", "cscv@2"]}
    assert any("seed" in n for n in out["notes"])
    assert out["rows"]


def test_within_receipt_compares_baseline_to_candidate():
    r = receipt(results(80, 4), baseline=results(60, 10), baseline_methods={"s5.pbo": "cscv@1"},
                methods={"s5.pbo": "cscv@2"})
    out = within_receipt(r)
    assert out["method_differences"] == {"s5.pbo": ["cscv@1", "cscv@2"]}
    assert {x["better"] for x in out["rows"]} == {"b"}
    with pytest.raises(ValueError, match="baseline"):
        within_receipt(receipt(results(1, 1)))

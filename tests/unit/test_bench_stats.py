import pytest

from qrh.bench.stats import ALPHA, compare_counts, exact_ci, metric


def test_exact_ci_matches_known_values():
    lo, hi = exact_ci(0, 20)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-9)
    lo, hi = exact_ci(20, 20)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 20), abs=1e-9)
    lo, hi = exact_ci(5, 50)
    assert 0.03 < lo < 0.10 < hi < 0.22


def test_metric_record_has_counts_rate_ci_and_direction():
    m = metric(3, 40, direction="lower", gate=True)
    assert m["k"] == 3 and m["n"] == 40 and m["rate"] == pytest.approx(0.075)
    assert m["ci"][0] < 0.075 < m["ci"][1] and m["direction"] == "lower" and m["gate"] is True
    empty = metric(0, 0, direction="higher")
    assert empty["rate"] is None and empty["ci"] is None


def test_metric_rejects_inconsistent_counts():
    for k, n in ((-1, 5), (6, 5), (1, -1)):
        with pytest.raises(ValueError):
            metric(k, n, direction="higher")
    with pytest.raises(ValueError):
        metric(1, 5, direction="sideways")


def test_compare_counts_fisher():
    same = compare_counts(10, 20, 10, 20)
    assert same["p_value"] == pytest.approx(1.0) and same["distinguishable"] is False and same["delta"] == 0.0
    far = compare_counts(0, 50, 25, 50)
    assert far["p_value"] < 1e-6 and far["distinguishable"] is True and far["delta"] == pytest.approx(0.5)
    small = compare_counts(1, 5, 2, 5)
    assert small["distinguishable"] is False  # a visible difference that reps cannot support
    assert ALPHA == 0.05


def test_compare_counts_handles_empty_cells():
    r = compare_counts(0, 0, 3, 10)
    assert r["p_value"] is None and r["distinguishable"] is False and r["delta"] is None

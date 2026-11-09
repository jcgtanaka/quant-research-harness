"""Bench for the pnl.linear verdict against known truth (synthetic data only).

The caps below are test parameters, not the approved default.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from qrh.bench.pnl_linear_corpus import (
    CAPITAL,
    CASE_KINDS,
    COMMISSION,
    GAIN_BPS,
    build_corpus,
    detection_power,
    false_alarm_rate,
    make_honest_case,
    noise_sensitivity_sweep,
    noisy_gain_case,
    noisy_honest_case,
    run_corpus,
)
from qrh.bench.pnl_ledger import run_ledger

SEEDS = list(range(20))
CAP_BP_PER_YEAR = 25.0  # test parameter, not the approved default
CAP_SHARE = 0.02  # test parameter, not the approved default
ALPHA = 0.05  # test parameter, not the approved default
NW_LAG = 2  # test parameter, not the approved default


@pytest.fixture(scope="module")
def results():
    start = time.perf_counter()
    res = run_corpus(SEEDS, CAP_BP_PER_YEAR, CAP_SHARE, ALPHA, NW_LAG)
    res_elapsed = time.perf_counter() - start
    print(f"\ncorpus runtime for {len(SEEDS)} seeds: {res_elapsed:.2f}s")
    return res


def test_corpus_is_deterministic_per_seed():
    a = make_honest_case("rebalancing", 3)
    b = make_honest_case("rebalancing", 3)
    assert np.array_equal(a.reported, b.reported)
    assert np.array_equal(a.explained, b.explained)
    other = make_honest_case("rebalancing", 4)
    assert not np.array_equal(a.reported, other.reported)


def test_corpus_covers_required_kinds_and_states_expectations():
    corpus = build_corpus([0])
    assert {"buy_and_hold", "rebalancing", "declared_halt"} <= set(CASE_KINDS)
    defects = {c.defect for c in corpus}
    assert {"honest", "stale_mark", "future_mark", "cost_free"} <= defects
    assert {f"gain_{g}bp" for g in GAIN_BPS} <= defects
    for c in corpus:
        assert c.expected and "PASS" not in c.expected or c.defect == "honest"


def test_honest_cases_never_fail(results):
    honest = {k: v for k, v in results.items() if k[1] == "honest"}
    assert honest
    for key, counts in honest.items():
        assert counts.get("FAIL", 0) == 0, (key, counts)
        assert counts.get("BLOCKED", 0) == 0, (key, counts)
    assert false_alarm_rate(results) == 0.0


def test_largest_gain_is_detected(results):
    assert detection_power(results)[max(GAIN_BPS)] == 1.0


def test_smallest_gain_is_reported_as_a_finding(results):
    for kind in CASE_KINDS:
        counts = results[(kind, f"gain_{min(GAIN_BPS)}bp")]
        assert counts.get("PASS", 0) == 0, (kind, counts)
        assert counts.get("FAIL", 0) > 0, (kind, counts)


@pytest.mark.parametrize("defect", ["stale_mark", "future_mark"])
def test_mark_defects_are_detected(results, defect):
    for kind in CASE_KINDS:
        counts = results[(kind, defect)]
        assert counts.get("FAIL", 0) == sum(counts.values()), (kind, counts)


def test_cost_free_defect_is_detected(results):
    seen = [k for k in results if k[1] == "cost_free"]
    assert seen
    for key in seen:
        counts = results[key]
        assert counts.get("FAIL", 0) == sum(counts.values()), (key, counts)


def test_no_planted_defect_ever_passes(results):
    for (kind, defect), counts in results.items():
        if defect != "honest":
            assert counts.get("PASS", 0) == 0, (kind, defect, counts)


def test_every_result_matches_its_stated_expectation():
    corpus = build_corpus(SEEDS[:3])
    res = run_corpus(SEEDS[:3], CAP_BP_PER_YEAR, CAP_SHARE, ALPHA, NW_LAG, return_outcomes=True)
    assert len(res) == len(corpus)
    for case, verdict in zip(corpus, res):
        assert verdict in case.expected, (case.kind, case.defect, case.seed, verdict)


def test_running_twice_gives_identical_counts(results):
    again = run_corpus(SEEDS, CAP_BP_PER_YEAR, CAP_SHARE, ALPHA, NW_LAG)
    assert {k: dict(v) for k, v in again.items()} == {k: dict(v) for k, v in results.items()}


NOISE_BPS = (0.0, 1.0, 5.0)
SWEEP_KINDS = ("buy_and_hold", "rebalancing")


@pytest.fixture(scope="module")
def sweep():
    start = time.perf_counter()
    table = noise_sensitivity_sweep(SEEDS, NOISE_BPS, GAIN_BPS, CAP_BP_PER_YEAR, CAP_SHARE,
                                    ALPHA, NW_LAG, kinds=SWEEP_KINDS)
    print(f"\nsweep runtime for {len(SEEDS)} seeds: {time.perf_counter() - start:.2f}s")
    return table


def test_noisy_honest_is_deterministic_per_seed():
    base = make_honest_case("rebalancing", 3)
    a = noisy_honest_case(base, 5.0)
    b = noisy_honest_case(make_honest_case("rebalancing", 3), 5.0)
    assert np.array_equal(a.reported, b.reported)
    other = noisy_honest_case(make_honest_case("rebalancing", 4), 5.0)
    assert not np.array_equal(a.reported - base.reported, other.reported - base.reported)
    assert a.defect == "noisy_honest_5bp"


def test_noise_has_stated_std_and_is_absent_from_explained():
    base = make_honest_case("rebalancing", 3)
    noisy = noisy_honest_case(base, 5.0)
    assert np.array_equal(noisy.explained, base.explained)
    assert np.array_equal(noisy.remainder, base.remainder + (noisy.reported - base.reported))
    diffs = np.concatenate([
        noisy_honest_case(make_honest_case("rebalancing", s), 5.0).reported
        - make_honest_case("rebalancing", s).reported for s in range(20)])
    assert diffs.std() == pytest.approx(5.0 * 1e-4 * 100_000.0, rel=0.1)


def test_zero_noise_equals_honest():
    base = make_honest_case("buy_and_hold", 1)
    assert np.array_equal(noisy_honest_case(base, 0.0).reported, base.reported)


def test_noisy_path_does_not_mutate_true_ledger_or_base_case():
    base = make_honest_case("rebalancing", 2)
    reported_before = base.reported.copy()
    prices_before = base.extra["prices"].copy()
    noisy_honest_case(base, 5.0)
    noisy_gain_case(base, 5.0, 2)
    assert np.array_equal(base.reported, reported_before)
    assert np.array_equal(base.extra["prices"], prices_before)
    ledger = run_ledger(base.extra["prices"], base.extra["schedule"], CAPITAL,
                        commission_per_trade=COMMISSION, seed=base.seed)
    assert np.array_equal(np.array(ledger.pnl), reported_before)


def test_sweep_has_expected_shape(sweep):
    assert set(sweep["fail_share"]) == {(n, g) for n in NOISE_BPS for g in GAIN_BPS}
    assert set(sweep["honest_false_alarm"]) == set(NOISE_BPS)
    assert sweep["n_cases_per_cell"] == len(SEEDS) * len(SWEEP_KINDS)
    for v in sweep["fail_share"].values():
        assert 0.0 <= v <= 1.0


def test_largest_gain_detected_at_each_noise_level(sweep):
    for n in NOISE_BPS:
        assert sweep["fail_share"][(n, max(GAIN_BPS))] == 1.0, n


def test_honest_false_alarm_rate_is_a_number_in_unit_interval(sweep):
    for n in NOISE_BPS:
        assert 0.0 <= sweep["honest_false_alarm"][n] <= 1.0


def test_report_noise_sweep(sweep, capsys):
    with capsys.disabled():
        print("\nnoise_bp  " + "  ".join(f"gain{g}bp" for g in GAIN_BPS) + "  honest_FAIL")
        for n in NOISE_BPS:
            row = "  ".join(f"{sweep['fail_share'][(n, g)]:>7.2f}" for g in GAIN_BPS)
            print(f"{n:>8}  {row}  {sweep['honest_false_alarm'][n]:>10.2f}")


def test_report_detection_table(results, capsys):
    power = detection_power(results)
    with capsys.disabled():
        print("\ngain_bp_per_bar  detection_power")
        for g in sorted(power):
            print(f"{g:>15}  {power[g]:.3f}")
        print(f"false_alarm_rate_on_honest: {false_alarm_rate(results):.3f}")
    assert set(power) == set(GAIN_BPS)

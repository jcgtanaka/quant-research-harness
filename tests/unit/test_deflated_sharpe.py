import math
from statistics import NormalDist

import numpy as np
import pytest

from qrh.validation.deflated_sharpe import (
    EULER_GAMMA,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
    sample_moments,
)

PHI = NormalDist().cdf
PHI_INV = NormalDist().inv_cdf


def test_psr_normal_case_closed_form():
    # g3 = 0, g4 = 3: radicand = 1 + SR^2/2 ; with SR* = 0 it is not 1, so use the formula directly
    sr, t = 0.12, 250
    expected = PHI(sr * math.sqrt(t - 1) / math.sqrt(1 + 0.5 * sr**2))
    assert probabilistic_sharpe(sr, 0.0, t, 0.0, 3.0) == pytest.approx(expected, rel=1e-12)


def test_psr_at_zero_sharpe_is_one_half_and_matches_phi_sr_sqrt_t_minus_1():
    # With small SR the radicand is ~1: PSR(0) ~ Phi(SR*sqrt(T-1)); bound the gap by the radicand.
    sr, t = 0.01, 400
    rad = math.sqrt(1 + 0.5 * sr**2)
    assert probabilistic_sharpe(sr, 0.0, t, 0.0, 3.0) == pytest.approx(PHI(sr * math.sqrt(t - 1) / rad), rel=1e-12)
    assert abs(probabilistic_sharpe(sr, 0.0, t, 0.0, 3.0) - PHI(sr * math.sqrt(t - 1))) < 1e-4
    assert probabilistic_sharpe(0.0, 0.0, t, 0.0, 3.0) == pytest.approx(0.5)


def test_psr_skew_and_kurtosis_effects():
    base = probabilistic_sharpe(0.1, 0.0, 250, 0.0, 3.0)
    assert probabilistic_sharpe(0.1, 0.0, 250, -1.0, 3.0) < base  # negative skew widens the error
    assert probabilistic_sharpe(0.1, 0.0, 250, 0.0, 8.0) < base  # fat tails widen it
    assert probabilistic_sharpe(0.1, 0.0, 250, 0.5, 3.0) > base


def test_psr_monotone_in_benchmark_and_length():
    assert probabilistic_sharpe(0.1, 0.05, 250, 0, 3) < probabilistic_sharpe(0.1, 0.0, 250, 0, 3)
    assert probabilistic_sharpe(0.1, 0.0, 500, 0, 3) > probabilistic_sharpe(0.1, 0.0, 250, 0, 3)


def test_expected_max_sharpe_closed_form_and_n1():
    g = EULER_GAMMA
    assert g == pytest.approx(0.5772156649015329, abs=1e-15)  # Euler-Mascheroni constant
    n, v = 10, 0.04
    expected = math.sqrt(v) * ((1 - g) * PHI_INV(1 - 1 / n) + g * PHI_INV(1 - 1 / (n * math.e)))
    assert expected_max_sharpe(n, v) == pytest.approx(expected, rel=1e-12)
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert expected_max_sharpe(7, 0.0) == 0.0


def test_expected_max_sharpe_approximates_true_expected_maximum():
    # For N i.i.d. N(0, V) variables E[max] is close to the formula; check by simulation.
    rng = np.random.default_rng(0)
    n, v = 50, 0.01
    sim = rng.normal(0, math.sqrt(v), size=(20000, n)).max(axis=1).mean()
    assert expected_max_sharpe(n, v) == pytest.approx(sim, rel=0.05)


def test_expected_max_sharpe_increases_in_n_and_v():
    vals = [expected_max_sharpe(n, 0.01) for n in (2, 5, 20, 100, 1000)]
    assert vals == sorted(vals) and len(set(vals)) == len(vals) and vals[0] > 0
    vs = [expected_max_sharpe(20, v) for v in (0.001, 0.01, 0.1)]
    assert vs == sorted(vs)


def _returns(seed, t=300, mu=0.05, sd=1.0):
    return np.random.default_rng(seed).normal(mu * sd, sd, t)


def test_dsr_equals_psr_at_sr0_and_n1_equals_psr_at_zero():
    r = _returns(1)
    res = deflated_sharpe(r, n_trials=12, var_sharpe=0.01)
    assert res.dsr == pytest.approx(
        probabilistic_sharpe(res.sharpe, res.sr0, res.n_obs, res.skew, res.kurtosis), rel=1e-12
    )
    one = deflated_sharpe(r, n_trials=1, var_sharpe=0.01)
    assert one.sr0 == 0.0
    assert one.dsr == pytest.approx(probabilistic_sharpe(one.sharpe, 0.0, one.n_obs, one.skew, one.kurtosis))


def test_dsr_decreases_with_n_and_with_variance():
    r = _returns(2)
    by_n = [deflated_sharpe(r, n, 0.004).dsr for n in (1, 2, 5, 20, 100, 1000)]
    assert all(a > b for a, b in zip(by_n, by_n[1:]))
    by_v = [deflated_sharpe(r, 20, v).dsr for v in (0.0, 0.001, 0.004, 0.02)]
    assert all(a > b for a, b in zip(by_v, by_v[1:]))


def test_sample_moments_match_definitions():
    r = np.random.default_rng(3).standard_t(5, 2000)
    sk, ku = sample_moments(r)
    d = r - r.mean()
    assert sk == pytest.approx((d**3).mean() / (d**2).mean() ** 1.5)
    assert ku == pytest.approx((d**4).mean() / (d**2).mean() ** 2)
    sk0, ku0 = sample_moments(np.random.default_rng(4).normal(size=200000))
    assert abs(sk0) < 0.03 and abs(ku0 - 3.0) < 0.1  # normal: skew 0, non-excess kurtosis 3


def _winner_dsr_hits(n, reps, seed, known_v, t=250):
    rng = np.random.default_rng(seed)
    hits = naive = 0
    for _ in range(reps):
        m = rng.normal(0.0, 0.01, size=(t, n))
        sr = m.mean(0) / m.std(0, ddof=1)
        best = int(np.argmax(sr))
        v = 1.0 / t if known_v else (float(sr.var(ddof=1)) if n > 1 else 0.0)
        hits += deflated_sharpe(m[:, best], n_trials=n, var_sharpe=v).dsr > 0.95
        sk, ku = sample_moments(m[:, best])
        naive += probabilistic_sharpe(float(sr[best]), 0.0, t, sk, ku) > 0.95
    return hits / reps, naive / reps


def test_monte_carlo_psr_is_calibrated_for_a_single_null_strategy():
    # N = 1: DSR == PSR(0). Under the null P(PSR > 0.95) is the nominal 5% (SE ~0.5% at 2000 reps).
    rate, _ = _winner_dsr_hits(1, 2000, 99, known_v=False)
    assert 0.035 <= rate <= 0.065, rate


def test_monte_carlo_winner_of_noise_trials_passes_at_most_nominal_rate():
    # The winner of N null strategies: DSR > 0.95 must be rare. The test compares the winner with
    # the expected maximum, using the sampling sd of one Sharpe, so it is conservative: for N = 2
    # (true V) the size is a little below 5% (seen ~2.7%); for larger N it is smaller still.
    rate2, naive2 = _winner_dsr_hits(2, 1000, 12345, known_v=True)
    assert 0.005 <= rate2 <= 0.06, rate2
    rate40, naive40 = _winner_dsr_hits(40, 400, 12345, known_v=False)
    assert rate40 <= 0.05, rate40
    # Without deflation the same winner passes almost always (the test has power).
    assert naive40 > 0.8, naive40


def test_a_genuine_edge_survives_deflation():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0015, 0.01, 1000)  # per-period Sharpe ~0.15
    assert deflated_sharpe(r, n_trials=20, var_sharpe=0.0004).dsr > 0.95


@pytest.mark.parametrize(
    "args",
    [
        dict(returns=[0.01, 0.02], n_trials=0, var_sharpe=0.1),
        dict(returns=[0.01, 0.02, 0.0], n_trials=2.5, var_sharpe=0.1),
        dict(returns=[0.01, 0.02, 0.0], n_trials=3, var_sharpe=-0.1),
        dict(returns=[0.01, 0.02, 0.0], n_trials=3, var_sharpe=float("nan")),
        dict(returns=[0.01], n_trials=3, var_sharpe=0.1),
        dict(returns=[0.01, float("nan"), 0.02], n_trials=3, var_sharpe=0.1),
        dict(returns=[0.01, 0.01, 0.01], n_trials=3, var_sharpe=0.1),  # Sharpe undefined
    ],
)
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        deflated_sharpe(**args)


def test_psr_invalid_radicand_raises():
    with pytest.raises(ValueError):
        probabilistic_sharpe(5.0, 0.0, 100, 10.0, 3.0)  # 1 - g3*SR + ... <= 0
    with pytest.raises(ValueError):
        probabilistic_sharpe(0.1, 0.0, 1, 0.0, 3.0)

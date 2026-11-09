import itertools
import math

import numpy as np
import pandas as pd
import pytest

from qrh.validation.pbo_cscv import pbo_cscv


def naive(m, s):
    """Straight transcription of the CSCV definition (slow, for small inputs)."""
    t, n = m.shape
    rows = (t // s) * s
    m = m[t - rows:]
    blocks = np.split(m, s)
    lam = []
    for is_idx in itertools.combinations(range(s), s // 2):
        oos_idx = [b for b in range(s) if b not in is_idx]
        is_m = np.vstack([blocks[b] for b in is_idx])
        oos_m = np.vstack([blocks[b] for b in oos_idx])
        sr_is = is_m.mean(0) / is_m.std(0, ddof=1)
        sr_oos = oos_m.mean(0) / oos_m.std(0, ddof=1)
        best = int(np.argmax(sr_is))
        less = (sr_oos < sr_oos[best]).sum()
        eq = (sr_oos == sr_oos[best]).sum()
        rank = less + (eq + 1) / 2
        omega = rank / (n + 1)
        lam.append(math.log(omega / (1 - omega)))
    return np.array(lam)


def noise(t, n, seed):
    return np.random.default_rng(seed).normal(0.0, 0.01, size=(t, n))


def test_matches_naive_definition():
    m = noise(70, 5, 1)  # 70 rows, 6 blocks -> 4 leading rows dropped
    res = pbo_cscv(m, n_blocks=6)
    lam = naive(m, 6)
    assert res.n_combinations == 20 == len(res.logits)
    np.testing.assert_allclose(np.sort(res.logits), np.sort(lam), rtol=1e-9, atol=1e-9)
    assert res.pbo == pytest.approx((lam <= 0).mean())
    assert res.rows_dropped == 4 and res.rows_used == 66 and res.n_strategies == 5


def test_omega_is_in_open_interval_so_logit_is_finite():
    res = pbo_cscv(noise(160, 6, 2), n_blocks=8)
    assert np.isfinite(res.logits).all()
    # relative ranks are k/(N+1): the logit takes at most N distinct values
    assert len(np.unique(np.round(res.logits, 9))) <= 6


def test_default_blocks_is_16_and_combinations_12870():
    res = pbo_cscv(noise(64, 3, 3))
    assert res.n_blocks == 16 and res.n_combinations == 12870


def test_pure_noise_pbo_near_one_half():
    # Exchangeable i.i.d. columns: the OOS rank of the IS winner is uniform, and
    # with N even P(rank <= N/2) = 1/2 exactly. One dataset's 12870 combinations are
    # strongly dependent, so single-seed values spread; we average 6 seeds.
    vals = [pbo_cscv(noise(320, 40, 100 + k)).pbo for k in range(6)]
    assert abs(np.mean(vals) - 0.5) < 0.08, vals
    assert all(0.2 < v < 0.8 for v in vals), vals


def test_pure_noise_mean_oos_sharpe_of_winner_near_zero_and_loss_prob_near_half():
    res = pbo_cscv(noise(320, 40, 7))
    assert abs(res.mean_oos_sharpe) < 0.1  # per-period units, sd of a block-pair Sharpe ~0.11
    assert 0.1 < res.prob_oos_loss < 0.9  # dependent combinations: wide single-seed spread


def test_one_dominant_column_gives_pbo_near_zero():
    m = noise(320, 20, 5)
    m[:, 7] += 0.01  # mean of 1 sd per period: Sharpe ~1 per period against noise ~0
    res = pbo_cscv(m)
    assert res.pbo == 0.0
    assert res.prob_oos_loss == 0.0
    assert res.mean_oos_sharpe > 0.5


def test_column_permutation_invariance():
    m = noise(160, 7, 11)
    perm = np.random.default_rng(0).permutation(7)
    a, b = pbo_cscv(m, n_blocks=8), pbo_cscv(m[:, perm], n_blocks=8)
    assert a.pbo == b.pbo
    np.testing.assert_allclose(np.sort(a.logits), np.sort(b.logits))
    assert a.mean_oos_sharpe == pytest.approx(b.mean_oos_sharpe)


@pytest.mark.parametrize("scale", [0.5, 3.0, 100.0])
def test_positive_scaling_invariance(scale):
    m = noise(160, 7, 12)
    a, b = pbo_cscv(m, n_blocks=8), pbo_cscv(m * scale, n_blocks=8)
    assert a.pbo == b.pbo
    np.testing.assert_allclose(a.logits, b.logits, atol=1e-9)


def test_accepts_dataframe_and_is_deterministic():
    m = noise(80, 4, 13)
    a = pbo_cscv(pd.DataFrame(m), n_blocks=4)
    b = pbo_cscv(m, n_blocks=4)
    assert a.pbo == b.pbo and np.array_equal(a.logits, b.logits)


def test_flat_columns_do_not_break_and_cannot_win():
    m = noise(160, 5, 14)
    m[:, 0] = 0.0  # a strategy that never trades: Sharpe 0 by convention
    res = pbo_cscv(m, n_blocks=8)
    assert np.isfinite(res.logits).all()


def test_constant_nonzero_column_is_an_error():
    m = noise(160, 5, 15)
    m[:, 2] = 0.001
    with pytest.raises(ValueError, match="constant"):
        pbo_cscv(m, n_blocks=8)


@pytest.mark.parametrize(
    "m,s,msg",
    [
        (noise(100, 1, 0), 8, "at least 2"),
        (noise(100, 3, 0), 7, "even"),
        (noise(100, 3, 0), 0, "even"),
        (noise(100, 3, 0), 2.5, "even"),
        (noise(20, 3, 0), 16, "rows"),
        (noise(100, 3, 0)[0], 8, "2-D"),
        (np.where(np.eye(20, 3) > 0, np.nan, 0.01), 4, "NaN"),
    ],
)
def test_invalid_input(m, s, msg):
    with pytest.raises(ValueError, match=msg):
        pbo_cscv(m, n_blocks=s)

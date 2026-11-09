import numpy as np
import pandas as pd
import pytest

from qrh.synthetic.generator import Regime, SyntheticConfig, generate


def _bytes(df):
    return df.to_csv().encode()


def test_deterministic_same_config():
    cfg = SyntheticConfig(seed=1, periods=300)
    assert _bytes(generate(cfg)) == _bytes(generate(cfg))


def test_deterministic_with_all_features():
    cfg = SyntheticConfig(
        seed=3,
        periods=400,
        gap_probability=0.1,
        jump_probability=0.02,
        jump_size=0.05,
        regimes=[Regime(share=0.5, drift=0.0, volatility=0.01), Regime(share=0.5, drift=0.0, volatility=0.03)],
    )
    assert _bytes(generate(cfg)) == _bytes(generate(cfg))


def test_seed_sensitivity():
    a = generate(SyntheticConfig(seed=1, periods=200))
    b = generate(SyntheticConfig(seed=2, periods=200))
    assert _bytes(a) != _bytes(b)


def test_does_not_touch_global_random_state():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    generate(SyntheticConfig(seed=1, periods=50))
    assert (np.random.get_state()[1] == before).all()


def test_regime_volatility_ordering():
    cfg = SyntheticConfig(
        seed=11,
        periods=4000,
        regimes=[Regime(length=2000, drift=0.0, volatility=0.002), Regime(length=2000, drift=0.0, volatility=0.02)],
    )
    r = np.log(generate(cfg)["close"]).diff().dropna()
    low, high = r.iloc[:1900].std(), r.iloc[-1900:].std()
    assert high > 3 * low


def test_regime_shares_cover_all_periods():
    cfg = SyntheticConfig(
        seed=1,
        periods=101,
        regimes=[Regime(share=0.3, drift=0, volatility=0.01), Regime(share=0.7, drift=0, volatility=0.02)],
    )
    assert len(generate(cfg)) == 101


def test_gap_removal():
    full = generate(SyntheticConfig(seed=5, periods=1000))
    gapped = generate(SyntheticConfig(seed=5, periods=1000, gap_probability=0.2))
    assert 600 < len(gapped) < 950
    assert len(full) == 1000
    assert gapped.index.is_monotonic_increasing
    assert gapped.index.isin(full.index).all()


def test_positivity_with_jumps():
    cfg = SyntheticConfig(seed=9, periods=2000, jump_probability=0.05, jump_size=0.2,
                          regimes=[Regime(length=2000, drift=-0.001, volatility=0.05)])
    df = generate(cfg)
    assert (df["close"] > 0).all()


def test_timezone_aware_default_and_custom():
    assert generate(SyntheticConfig(seed=1, periods=10)).index.tz is not None
    df = generate(SyntheticConfig(seed=1, periods=10, tz="America/New_York"))
    assert str(df.index.tz) == "America/New_York"


@pytest.mark.parametrize("freq", ["1min", "h", "D", "W", "15min"])
def test_any_frequency(freq):
    df = generate(SyntheticConfig(seed=1, periods=20, freq=freq))
    assert len(df) == 20
    assert df.index.to_series().diff().dropna().nunique() >= 1
    assert df.index.is_unique


def test_invalid_regimes_rejected():
    with pytest.raises(ValueError):
        Regime(drift=0, volatility=0.01)  # neither length nor share
    with pytest.raises(ValueError):
        Regime(length=5, share=0.5, drift=0, volatility=0.01)
    with pytest.raises(ValueError):
        SyntheticConfig(seed=1, periods=10, regimes=[Regime(length=20, drift=0, volatility=0.01)])


def _log_returns(cfg):
    return np.diff(np.log(generate(cfg)["close"].to_numpy()))


def test_trend_component_is_additive_seeded_and_does_not_shift_other_streams():
    base = SyntheticConfig(seed=5, periods=20001, regimes=[Regime(share=1.0, drift=0.0, volatility=0.01)])
    trended = base.model_copy(update={"trend_persistence": 0.95, "trend_scale": 0.002})
    latent = _log_returns(trended) - _log_returns(base)  # the same shocks, plus the latent drift
    assert abs(latent.std() / 0.002 - 1) < 0.2
    ac1 = np.corrcoef(latent[:-1], latent[1:])[0, 1]
    assert abs(ac1 - 0.95) < 0.03
    # reproducible
    assert np.array_equal(_log_returns(trended), _log_returns(trended))
    # gaps are drawn from their own stream: same bars are dropped with and without a trend
    g = base.model_copy(update={"gap_probability": 0.1})
    gt = g.model_copy(update={"trend_persistence": 0.9, "trend_scale": 0.001})
    assert generate(g).index.equals(generate(gt).index)


def test_trend_makes_returns_positively_autocorrelated():
    base = SyntheticConfig(seed=8, periods=20001, regimes=[Regime(share=1.0, drift=0.0, volatility=0.01)])
    trended = base.model_copy(update={"trend_persistence": 0.97, "trend_scale": 0.004})
    def ac(r, k):
        return np.corrcoef(r[:-k], r[k:])[0, 1]
    assert abs(ac(_log_returns(base), 5)) < 0.03
    assert ac(_log_returns(trended), 5) > 0.1


def test_trend_off_by_default():
    cfg = SyntheticConfig(seed=2, periods=50)
    assert cfg.trend_scale == 0.0 and cfg.trend_persistence == 0.0


@pytest.mark.parametrize("bad", [{"trend_persistence": 1.0}, {"trend_persistence": -0.1}, {"trend_scale": -1.0}])
def test_trend_validation(bad):
    with pytest.raises(ValueError):
        SyntheticConfig(seed=1, periods=10, **bad)

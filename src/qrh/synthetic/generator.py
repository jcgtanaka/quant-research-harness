"""Seeded synthetic price generator (log-return based, always positive)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class Regime(BaseModel):
    """A segment with constant per-bar drift and volatility (log returns).

    Give exactly one of ``length`` (bars) or ``share`` (fraction of periods).
    """

    length: int | None = Field(default=None, gt=0)
    share: float | None = Field(default=None, gt=0, le=1)
    drift: float = 0.0
    volatility: float = Field(default=0.01, ge=0)

    @model_validator(mode="after")
    def _one_of(self) -> "Regime":
        if (self.length is None) == (self.share is None):
            raise ValueError("specify exactly one of length or share")
        return self


class SyntheticConfig(BaseModel):
    seed: int
    start: str = "2020-01-01"
    periods: int = Field(gt=0)
    freq: str = "D"
    tz: str = "UTC"
    start_price: float = Field(default=100.0, gt=0)
    regimes: list[Regime] = Field(default_factory=lambda: [Regime(share=1.0, drift=0.0, volatility=0.01)])
    gap_probability: float = Field(default=0.0, ge=0, lt=1)
    jump_probability: float = Field(default=0.0, ge=0, le=1)
    jump_size: float = Field(default=0.0, ge=0)
    # Optional persistent trend: a latent per-bar drift following an AR(1) process with this
    # persistence and stationary standard deviation, added to the log returns (off by default).
    trend_persistence: float = Field(default=0.0, ge=0, lt=1)
    trend_scale: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _regimes_fit(self) -> "SyntheticConfig":
        if not self.regimes:
            raise ValueError("at least one regime is required")
        fixed = sum(r.length for r in self.regimes if r.length is not None)
        if fixed > self.periods:
            raise ValueError("regime lengths exceed periods")
        return self


def _segment_lengths(cfg: SyntheticConfig) -> list[int]:
    lengths: list[int] = []
    for r in cfg.regimes:
        lengths.append(r.length if r.length is not None else int(round(r.share * cfg.periods)))
    # Absorb rounding so the segments cover exactly `periods` bars.
    diff = cfg.periods - sum(lengths)
    lengths[-1] += diff
    if lengths[-1] < 0:
        raise ValueError("regimes do not fit in periods")
    return lengths


def _latent_trend(shocks: np.ndarray, rho: float, scale: float) -> np.ndarray:
    """Stationary AR(1) path with standard deviation ``scale`` (the first value is a stationary draw)."""
    out = np.empty(len(shocks))
    out[0] = scale * shocks[0]
    inno = scale * np.sqrt(1.0 - rho**2)
    for t in range(1, len(shocks)):
        out[t] = rho * out[t - 1] + inno * shocks[t]
    return out


def generate(cfg: SyntheticConfig) -> pd.DataFrame:
    """Generate a timezone-aware frame with a strictly positive ``close`` column."""
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    n = cfg.periods
    drift = np.empty(n)
    vol = np.empty(n)
    pos = 0
    for r, length in zip(cfg.regimes, _segment_lengths(cfg)):
        drift[pos : pos + length] = r.drift
        vol[pos : pos + length] = r.volatility
        pos += length

    # Draw every random stream at full length so options never shift other streams.
    normals = rng.standard_normal(n)
    jump_hits = rng.random(n) < cfg.jump_probability
    jump_signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    gap_draws = rng.random(n)

    trend_shocks = rng.standard_normal(n)  # drawn last: earlier streams are unchanged by this option
    log_ret = drift + vol * normals + jump_hits * jump_signs * cfg.jump_size
    if cfg.trend_scale > 0:
        log_ret = log_ret + _latent_trend(trend_shocks, cfg.trend_persistence, cfg.trend_scale)
    log_price = np.log(cfg.start_price) + np.cumsum(log_ret)
    index = pd.date_range(cfg.start, periods=n, freq=cfg.freq, tz=cfg.tz)
    df = pd.DataFrame({"close": np.exp(log_price)}, index=index)
    df.index.name = "timestamp"

    if cfg.gap_probability > 0:
        keep = gap_draws >= cfg.gap_probability
        keep[0] = True
        df = df.loc[keep]
    return df

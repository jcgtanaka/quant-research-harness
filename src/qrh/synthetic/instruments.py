"""Small seeded synthetic instrument datasets for tests and examples (nothing here is real market data).

Each helper returns ``(frame, proposal)``: a time-indexed DataFrame (index ``timestamp``, UTC) and the CORRECT
instrument-model proposal for it, in the file format of ``qrh data propose`` (``fields`` and ``extras``). Tests use
the pair to assert that a correct proposal verifies and that a planted wrong one is contradicted or stays
unverified; ``with_claim`` builds the wrong ones. The symbols are made up (``SYN``-style codes), the numbers are
random walks.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

Proposal = dict[str, Any]


def _claim(value: Any, evidence: str = "") -> dict[str, Any]:
    return {"value": value, "evidence": evidence}


def _proposal(**fields: Any) -> Proposal:
    return {"source": "llm", "fields": {k: _claim(v) for k, v in fields.items()}, "extras": {}}


def with_claim(proposal: Proposal, field: str, value: Any, evidence: str = "") -> Proposal:
    """A copy of ``proposal`` with one claim replaced (or added); this is how the wrong proposals are planted."""
    out = copy.deepcopy(proposal)
    out["fields"][field] = _claim(value, evidence)
    return out


def without_claim(proposal: Proposal, field: str) -> Proposal:
    out = copy.deepcopy(proposal)
    out["fields"].pop(field, None)
    return out


def _walk(rng: np.random.Generator, n: int, start: float, vol: float) -> np.ndarray:
    return start * np.exp(np.cumsum(rng.normal(0.0, vol, n)))


def _ohlc(rng: np.random.Generator, close: np.ndarray, vol: float) -> pd.DataFrame:
    n = len(close)
    open_ = np.concatenate(([close[0]], close[:-1])) * np.exp(rng.normal(0.0, vol / 4, n))
    span = np.exp(np.abs(rng.normal(0.0, vol / 2, n)))
    high = np.maximum(open_, close) * span
    low = np.minimum(open_, close) / span
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def _index(start: str, periods: int, freq: str) -> pd.DatetimeIndex:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    idx.name = "timestamp"
    return idx


def equity_daily(seed: int = 0, periods: int = 260, splits: bool = False) -> tuple[pd.DataFrame, Proposal]:
    """Business-day OHLCV bars. ``splits=True`` leaves a 4-for-1 split UNADJUSTED in the prices (the volume on the
    split bar is ordinary), so the correct description of that file is ``adjustments: none`` and verification
    rightly contradicts it: such a file must not be used as it is. Without splits the file is a split-and-dividend
    adjusted series, which no check can prove (a human confirms it)."""
    rng = np.random.default_rng(seed)
    close = _walk(rng, periods, 50.0, 0.012)
    if splits:
        close = close.copy()
        close[periods // 2:] = close[periods // 2:] / 4.0
    df = _ohlc(rng, close, 0.012)
    df["volume"] = rng.integers(8_000, 12_000, periods).astype(float)
    df.index = _index("2021-01-04", periods, "B")
    proposal = _proposal(
        asset_class="equity", price_columns=["open", "high", "low", "close"], price_units="currency",
        quote_currency="USD", timestamp_semantics="bar_close", timezone="UTC", frequency="1D",
        adjustments="none" if splits else "split_and_dividend",
    )
    return df, proposal


def futures_daily(seed: int = 0, periods: int = 260) -> tuple[pd.DataFrame, Proposal]:
    """Business-day bars of a quarterly futures contract, spliced WITHOUT adjustment: every roll leaves a basis gap
    of about two percent in the price and a new contract code (``SYH1``, ``SYM1``, ...)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.004, periods)
    roll_every = 63
    rolls = np.arange(roll_every, periods, roll_every)
    steps[rolls] += rng.choice([-1.0, 1.0], len(rolls)) * 0.02
    close = 4000.0 * np.exp(np.cumsum(steps))
    df = _ohlc(rng, close, 0.004)
    months = ["H", "M", "U", "Z"]
    codes = []
    for i in range(periods):
        q = i // roll_every
        codes.append(f"SY{months[q % 4]}{1 + q // 4}")
    df.insert(0, "contract", codes)
    df["volume"] = rng.integers(1_000, 2_000, periods).astype(float)
    df["open_interest"] = rng.integers(10_000, 20_000, periods).astype(float)
    df.index = _index("2021-01-04", periods, "B")
    proposal = _proposal(
        asset_class="future", price_columns=["open", "high", "low", "close"], price_units="index_points",
        quote_currency="USD", timestamp_semantics="bar_close", timezone="UTC", frequency="1D",
        adjustments="none", multiplier=50, tick_size=0.25,
        roll_rule={"text": "Quarterly, five bars before expiry", "method": "calendar"},
    )
    return df, proposal


def fx_quotes(seed: int = 0, periods: int = 400) -> tuple[pd.DataFrame, Proposal]:
    """One-minute bid/ask quotes of a made-up currency pair (bid below ask on every row)."""
    rng = np.random.default_rng(seed)
    mid = _walk(rng, periods, 1.10, 0.0002)
    half_spread = 0.00005 + 0.00002 * rng.random(periods)
    df = pd.DataFrame({"bid": mid - half_spread, "ask": mid + half_spread})
    df.index = _index("2021-03-01 08:00", periods, "1min")
    proposal = _proposal(
        asset_class="fx_spot", price_columns=["bid", "ask"], price_units="currency", quote_currency="USD",
        timestamp_semantics="event_time", timezone="UTC", frequency="1min", adjustments="none",
    )
    return df, proposal


def option_snapshots(seed: int = 0, days: int = 12, strikes: int = 5) -> tuple[pd.DataFrame, Proposal]:
    """Daily option-chain snapshots: several rows per timestamp (strike by right), two expiries."""
    rng = np.random.default_rng(seed)
    spot = _walk(rng, days, 100.0, 0.01)
    rows: list[dict[str, Any]] = []
    stamps: list[pd.Timestamp] = []
    index = _index("2021-05-03", days, "B")
    for d in range(days):
        for expiry in ("2021-07-16", "2021-09-17"):
            for k in range(strikes):
                strike = 80.0 + 10.0 * k
                for right in ("C", "P"):
                    intrinsic = max(spot[d] - strike, 0.0) if right == "C" else max(strike - spot[d], 0.0)
                    mid = intrinsic + 2.0 + rng.random()
                    rows.append({
                        "underlying_price": float(spot[d]), "strike": strike, "expiry": expiry, "right": right,
                        "bid": mid - 0.05, "ask": mid + 0.05, "iv": 0.2 + 0.05 * rng.random(),
                        "delta": (0.5 if right == "C" else -0.5) * (0.5 + rng.random() / 2),
                        "rate": 0.045, "volume": float(rng.integers(0, 500)), "open_interest": float(rng.integers(100, 5_000)),
                    })
                    stamps.append(index[d])
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(stamps, name="timestamp"))
    proposal = _proposal(
        asset_class="option", price_columns=["bid", "ask"], price_units="currency", quote_currency="USD",
        timestamp_semantics="event_time", timezone="UTC", frequency="1D", adjustments="none", multiplier=100,
        option_mapping={"underlying": "underlying_price", "strike": "strike", "expiry": "expiry", "right": "right",
                        "bid": "bid", "ask": "ask", "rate": "rate"},
    )
    return df, proposal


def crypto_perp(seed: int = 0, periods: int = 300) -> tuple[pd.DataFrame, Proposal]:
    """Hourly bars of a perpetual swap with an eight-hourly funding column (zero between settlements)."""
    rng = np.random.default_rng(seed)
    close = _walk(rng, periods, 2000.0, 0.006)
    df = pd.DataFrame({"close": close})
    funding = np.zeros(periods)
    funding[::8] = rng.normal(0.0001, 0.0002, len(funding[::8]))
    df["funding_rate"] = funding
    df["volume"] = rng.integers(100, 400, periods).astype(float)
    df.index = _index("2021-06-01", periods, "1h")
    proposal = _proposal(
        asset_class="crypto_perp", price_columns=["close"], price_units="currency", quote_currency="USDT",
        timestamp_semantics="bar_close", timezone="UTC", frequency="1h", adjustments="none", multiplier=1,
    )
    proposal["extras"] = {"funding_column": "funding_rate"}
    return df, proposal


def write_csv(frame: pd.DataFrame, path: Any) -> Any:
    """Write a frame the way the catalog reads it (the index first, as ``timestamp``)."""
    frame.to_csv(path)
    return path

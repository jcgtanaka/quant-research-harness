"""Tests for the futures first slice (F1) of the pnl.linear family (synthetic data only, no real symbols).

Expected PnL per bar = contracts[t-lag] x (price[t] - price[t-1]) x multiplier, in currency; returns are a fraction
of the declared starting capital. The true PnL for the synthetic runs comes from the cash-and-quantity ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import RunResult
from qrh.bench.pnl_ledger import plant_hidden_gain, run_ledger
from qrh.checks.pnl_linear import evaluate_pnl_linear
from qrh.checks.pnl_linear_futures import check_pnl_linear_futures

N = 250
LAG = 1
MULT = 50.0
CPC = 2.0
CAPITAL = 1_000_000.0
PPY = 252.0
NOISE_SD = 50.0  # currency per bar, about 0.5 bp of capital
EXECUTION = {"lag_bars": LAG, "price_column": "close"}


def make_run(seed=11, n=N, mult=MULT, cpc=CPC, rolls=(), noise_sd=NOISE_SD, gain_per_bar=0.0, expiry=None):
    """Returns (RunResult, frame, ledger). ``rolls``: bars where the continuous series switches contract; the
    series is then unadjusted and the old contract's price change is carried in ``roll_price_change``."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    p = 4000.0 + np.cumsum(rng.normal(0.0, 20.0, n))
    p_old = p.copy()
    contract = np.zeros(n, dtype=int)
    for k, s in enumerate(rolls, start=1):
        contract[s:] = k
        p[s:] += 15.0 + 5.0 * k  # basis jump: the new contract trades at a different level
        p_old[s] = p[s] - (15.0 + 5.0 * k)
    d = np.zeros(n, dtype=int)
    cur = 0
    for t in range(n):
        if t % 7 == 0:
            cur = int(rng.integers(-3, 4))
        d[t] = cur
    held = np.zeros(n, dtype=int)
    held[LAG:] = d[:-LAG]
    sched = []
    prev = 0
    for s in range(n):
        if held[s] != prev and s >= 1:
            sched.append((s, float(held[s] - prev), float(p[s - 1])))
        prev = held[s]
        if s in rolls and held[s] != 0:
            sched.append((s, -float(held[s]), float(p_old[s])))
            sched.append((s, float(held[s]), float(p[s])))
    ledger = run_ledger(p, sched, CAPITAL, multiplier=mult, commission_per_contract=cpc)
    reported = ledger.pnl + rng.normal(0.0, noise_sd, n) + gain_per_bar
    returns = pd.Series(reported / CAPITAL, index=idx)
    result = RunResult(returns=returns, trades=pd.DataFrame(), positions=pd.DataFrame({"close": d}, index=idx))
    frame = pd.DataFrame({"close": p, "pub": idx}, index=idx)
    if rolls:
        frame["contract"] = [f"C{c}" for c in contract]
        rpc = np.full(n, np.nan)
        for s in rolls:
            rpc[s] = p_old[s] - p[s - 1]
        frame["roll_price_change"] = rpc
    if expiry is not None:
        frame["expiry"] = expiry
    return result, frame, ledger


def block(**over):
    b = {
        "position_units": "quantity",
        "asset_class": "futures",
        "publication_column": "pub",
        "max_staleness": 2,
        "adjustment": {"close": "back_adjusted"},
        "multiplier": MULT,
        "cost_per_contract": CPC,
        "starting_capital": CAPITAL,
        "contract_currency": "USD",
        "account_currency": "USD",
    }
    b.update(over)
    return {k: v for k, v in b.items() if v is not _DROP}


_DROP = object()


def run(result, frame, **over):
    return check_pnl_linear_futures(result, frame, EXECUTION, block(**over), PPY)


def text(v):
    return " ".join(v.reasons).lower()


# (a) clean run -> PASS with the declared inputs
def test_clean_back_adjusted_run_passes():
    result, frame, _ = make_run()
    v = run(result, frame)
    assert v.verdict == "PASS", v.reasons
    assert v.details["absolute_cap"].breached is False


def test_clean_unadjusted_run_with_rolls_and_roll_change_passes():
    result, frame, _ = make_run(rolls=(80, 160))
    v = run(result, frame, adjustment={"close": "none"}, roll_rule={"method": "calendar"})
    assert v.verdict == "PASS", v.reasons
    assert v.details["roll_bars_excluded"] == 0
    assert v.details["roll_bars"] == 2


def test_unadjusted_roll_without_old_contract_change_excludes_roll_bars():
    result, frame, _ = make_run(rolls=(80, 160))
    frame = frame.drop(columns=["roll_price_change"])
    v = run(result, frame, adjustment={"close": "none"}, roll_rule={"method": "volume"})
    assert v.verdict == "PASS", v.reasons
    assert v.details["roll_bars_excluded"] == 2


def test_forward_adjusted_is_accepted():
    result, frame, _ = make_run()
    assert run(result, frame, adjustment={"close": "forward_adjusted"}).verdict == "PASS"


# (b) multiplier declared at half the true value -> FAIL
def test_half_multiplier_fails_with_declared_caps():
    # Declared caps, as in the equity tests. The noise-scaled default scales its limit to the remainder's own spread,
    # which a multiplier error inflates; see the bench report for that setting.
    result, frame, _ = make_run()
    v = run(result, frame, multiplier=MULT / 2, max_unexplained_bp_per_year=50.0, max_unexplained_share=0.2)
    assert v.verdict == "FAIL", v.reasons
    assert run(result, frame, max_unexplained_bp_per_year=50.0, max_unexplained_share=0.2).verdict == "PASS"


def test_hidden_gain_of_one_bp_per_bar_fails():
    result, frame, _ = make_run(gain_per_bar=1e-4 * CAPITAL)
    assert run(result, frame).verdict == "FAIL"


# (c) unadjusted roll jump with no roll method -> BLOCKED
def test_unadjusted_without_roll_method_is_blocked():
    result, frame, _ = make_run(rolls=(80, 160))
    v = run(result, frame, adjustment={"close": "none"})
    assert v.verdict == "BLOCKED"
    assert "roll_rule" in text(v)


def test_unadjusted_with_unreadable_roll_method_is_blocked():
    result, frame, _ = make_run(rolls=(80, 160))
    v = run(result, frame, adjustment={"close": "none"}, roll_rule={"method": "whenever"})
    assert v.verdict == "BLOCKED"
    assert "roll_rule.method" in text(v)


def test_unadjusted_without_contract_column_is_blocked():
    result, frame, _ = make_run(rolls=(80, 160))
    frame = frame.drop(columns=["contract"])
    v = run(result, frame, adjustment={"close": "none"}, roll_rule={"method": "calendar"})
    assert v.verdict == "BLOCKED"
    assert "contract" in text(v)


# (d) required declared values
@pytest.mark.parametrize("field", ["multiplier", "starting_capital", "cost_per_contract"])
def test_missing_required_value_is_blocked_and_named(field):
    result, frame, _ = make_run()
    v = run(result, frame, **{field: _DROP})
    assert v.verdict == "BLOCKED"
    assert field in text(v)


@pytest.mark.parametrize("field,bad", [("multiplier", 0), ("multiplier", -5), ("starting_capital", 0),
                                       ("starting_capital", "100"), ("cost_per_contract", -1.0)])
def test_invalid_required_value_is_blocked_and_named(field, bad):
    result, frame, _ = make_run()
    v = run(result, frame, **{field: bad})
    assert v.verdict == "BLOCKED"
    assert field in text(v)


def test_currency_mismatch_is_blocked():
    result, frame, _ = make_run()
    v = run(result, frame, contract_currency="EUR")
    assert v.verdict == "BLOCKED"
    assert "currency" in text(v)


def test_missing_currency_is_blocked():
    result, frame, _ = make_run()
    assert run(result, frame, account_currency=_DROP).verdict == "BLOCKED"


def test_fractional_contracts_are_blocked():
    result, frame, _ = make_run()
    result.positions["close"] = result.positions["close"] + 0.5
    v = run(result, frame)
    assert v.verdict == "BLOCKED"
    assert "whole" in text(v)


# (e) margin
def test_missing_margin_warns_on_margin_only_and_keeps_the_pnl_verdict():
    result, frame, _ = make_run()
    without = run(result, frame)
    with_margin = run(result, frame, margin_per_contract=5_000.0)
    assert without.verdict == with_margin.verdict == "PASS"
    assert without.details["margin"]["status"] == "WARNING"
    assert "margin_per_contract" in " ".join(without.reasons)
    assert with_margin.details["margin"]["status"] == "OK"
    assert without.details["absolute_cap"] == with_margin.details["absolute_cap"]


def test_margin_above_capital_warns_without_changing_verdict():
    result, frame, _ = make_run()
    v = run(result, frame, margin_per_contract=CAPITAL)
    assert v.verdict == "PASS"
    assert v.details["margin"]["status"] == "WARNING"


# (f) expiry without a roll -> BLOCKED
def test_expiry_without_roll_is_blocked():
    expiry = pd.Series(pd.Timestamp("2024-04-01"), index=pd.date_range("2024-01-01", periods=N, freq="D"))
    result, frame, _ = make_run(expiry=expiry.to_numpy())
    v = run(result, frame)
    assert v.verdict == "BLOCKED"
    assert "expired" in text(v)


def test_expiry_followed_by_roll_is_not_blocked():
    idx = pd.date_range("2024-01-01", periods=N, freq="D")
    expiry = np.where(idx < idx[80], idx[85], idx[-1]).astype("datetime64[ns]")
    result, frame, _ = make_run(rolls=(80,), expiry=expiry)
    v = run(result, frame, adjustment={"close": "none"}, roll_rule={"method": "open_interest"})
    assert v.verdict != "BLOCKED", v.reasons


# (g) existing equity behaviour unchanged
def test_equity_block_is_unchanged_and_rejects_futures_keys():
    rng = np.random.default_rng(7)
    explained = rng.normal(5.0, 20.0, N)
    remainder = rng.normal(0.0, 1.0, N)
    times = pd.date_range("2024-01-01", periods=N, freq="D")
    eq = {
        "position_units": "quantity", "cost_model": {"bps": 1.0}, "asset_class": "equity",
        "min_cost_per_turnover": 0.0, "publication_column": "pub", "max_staleness": 2,
        "adjustment": {"close": "total_return"}, "max_unexplained_bp_per_year": 50.0, "max_unexplained_share": 0.2,
    }
    kw = dict(
        reported_pnl=explained + remainder, explained_net_pnl=explained, remainder=remainder,
        frame_columns={"close", "pub"}, bars_per_year=PPY, capital_base=100_000.0, newey_west_lag=2, alpha=0.05,
        staleness_inputs={"bar_times": times, "publication_times": times.copy(),
                          "position": pd.Series(np.ones(N)), "lag": 1, "halt_flag": None},
    )
    assert evaluate_pnl_linear(config_block=eq, **kw).verdict == "PASS"
    blocked = evaluate_pnl_linear(config_block={**eq, "multiplier": 50.0}, **kw)
    assert blocked.verdict == "BLOCKED" and "multiplier" in " ".join(blocked.reasons)
    blocked = evaluate_pnl_linear(config_block={**eq, "adjustment": {"close": "back_adjusted"}}, **kw)
    assert blocked.verdict == "BLOCKED"
    assert evaluate_pnl_linear(config_block={**eq, "asset_class": "future"}, **kw).verdict == "BLOCKED"


# ledger extension: defaults leave the old behaviour unchanged
def test_ledger_multiplier_and_per_contract_commission():
    p = np.array([100.0, 101.0, 103.0, 102.0])
    base = run_ledger(p, [(0, 2.0, 100.0)], 1_000.0)
    same = run_ledger(p, [(0, 2.0, 100.0)], 1_000.0, multiplier=1.0, commission_per_contract=0.0)
    assert np.array_equal(base.pnl, same.pnl)
    fut = run_ledger(p, [(0, 2.0, 100.0)], 1_000.0, multiplier=10.0, commission_per_contract=1.5)
    assert fut.pnl.tolist() == pytest.approx([-3.0, 20.0, 40.0, -20.0])
    assert fut.cost_pnl.tolist() == pytest.approx([-3.0, 0.0, 0.0, 0.0])


def test_ledger_rejects_bad_multiplier():
    with pytest.raises(ValueError):
        run_ledger(np.array([1.0, 2.0]), [], 10.0, multiplier=0.0)
    with pytest.raises(ValueError):
        run_ledger(np.array([1.0, 2.0]), [], 10.0, commission_per_contract=-1.0)


def test_planted_gain_helper_still_works_with_futures_ledger():
    _, _, ledger = make_run()
    reported, planted = plant_hidden_gain(ledger, 100.0)
    assert (reported - ledger.pnl == planted).all()

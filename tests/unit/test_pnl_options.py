"""Tests for the options first slice (O1) of the pnl.linear family (synthetic data only, no real symbols).

Expected PnL per bar = contracts[t-lag] x (option price[t] - option price[t-1]) x multiplier, in currency; cost per bar
= cost_per_contract x |contracts traded|; returns are a fraction of the declared starting capital. The true PnL for the
synthetic runs comes from the cash-and-quantity ledger. At expiry a held contract is closed at intrinsic value and
reported as cash-settled; the mark on the expiry bar must match intrinsic value within the currency rounding unit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrh.adapters.base import RunResult
from qrh.bench.pnl_ledger import run_ledger
from qrh.checks.pnl_linear_config import ReconstructionConfigError, validate_reconstruction
from qrh.checks.pnl_linear_options import check_pnl_linear_options
from qrh.checks.pnl_linear_run import check_pnl_linear
from qrh.pipeline.receipt import Verdict

N = 250
LAG = 1
MULT = 100.0
CPC = 0.65
CAPITAL = 1_000_000.0
PPY = 252.0
NOISE_SD = 50.0  # currency per bar, about 0.5 bp of capital
STRIKE = 4000.0
UNIT = 0.01
FAR_EXPIRY = "2025-12-19"  # after the last bar of the synthetic run
EXECUTION = {"lag_bars": LAG, "price_column": "px"}


def make_run(seed=11, n=N, mult=MULT, cpc=CPC, noise_sd=NOISE_SD, gain_per_bar=0.0, expiry_bar=None, mark_bump=0.0):
    """Returns (RunResult, frame, ledger). One call option on a synthetic underlying: price = intrinsic value plus a
    time value that decays to zero at ``expiry_bar``. With an expiry bar, 2 contracts are held through the expiry bar,
    all are closed after it, and the price stays at intrinsic value after expiry. ``mark_bump`` moves the mark on the
    expiry bar away from intrinsic value while the true PnL keeps using intrinsic value."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    u = STRIKE + np.cumsum(rng.normal(0.0, 20.0, n))
    life = expiry_bar if expiry_bar is not None else 2 * n
    p = np.maximum(u - STRIKE, 0.0) + 30.0 * np.clip(1.0 - np.arange(n) / life, 0.0, None)
    if expiry_bar is not None:
        p[expiry_bar + 1:] = p[expiry_bar]
    d = np.zeros(n, dtype=int)
    cur = 0
    for t in range(n):
        if t % 7 == 0:
            cur = int(rng.integers(-3, 4))
        d[t] = cur
    if expiry_bar is not None:
        d[expiry_bar - 1] = 2
        d[expiry_bar:] = 0
    held = np.zeros(n, dtype=int)
    held[LAG:] = d[:-LAG]
    sched, prev = [], 0
    for s in range(n):
        if held[s] != prev and s >= 1:
            sched.append((s, float(held[s] - prev), float(p[s - 1])))
        prev = held[s]
    ledger = run_ledger(p, sched, CAPITAL, multiplier=mult, commission_per_contract=cpc)
    reported = ledger.pnl + rng.normal(0.0, noise_sd, n) + gain_per_bar
    returns = pd.Series(reported / CAPITAL, index=idx)
    result = RunResult(returns=returns, trades=pd.DataFrame(), positions=pd.DataFrame({"px": d}, index=idx))
    mark = p.copy()
    if expiry_bar is not None:
        mark[expiry_bar] += mark_bump
    frame = pd.DataFrame({"px": mark, "und": u, "pub": idx}, index=idx)
    return result, frame, ledger


def as_chain(frame, expiry=FAR_EXPIRY, right="call", others=(3900.0, 4100.0), seed=5):
    """A chain: the declared contract plus other strikes (random prices) on every bar, one frame with repeated bars."""
    rng = np.random.default_rng(seed)
    base = frame.assign(strike=STRIKE, expiry=expiry, right=right)
    parts = [base]
    for k in others:
        parts.append(frame.assign(px=50.0 + rng.normal(0.0, 5.0, len(frame)), strike=k, expiry=expiry, right=right))
    return pd.concat(parts).sort_index(kind="stable")


def block(**over):
    b = {
        "position_units": "quantity",
        "asset_class": "option",
        "publication_column": "pub",
        "max_staleness": 2,
        "adjustment": {"px": "none"},
        "strike": STRIKE,
        "expiry": FAR_EXPIRY,
        "right": "call",
        "option_price_type": "mid",
        "multiplier": MULT,
        "starting_capital": CAPITAL,
        "cost_per_contract": CPC,
        "contract_currency": "USD",
        "account_currency": "USD",
        "currency_rounding_unit": UNIT,
        "underlying_column": "und",
        "point_in_time": "yes",
    }
    b.update(over)
    return {k: v for k, v in b.items() if v is not _DROP}


_DROP = object()


def run(result, frame, **over):
    return check_pnl_linear_options(result, frame, EXECUTION, block(**over), PPY)


def text(v):
    return " ".join(v.reasons).lower()


def expiry_of(frame, bar):
    return str(frame.index[bar].date())


# (a) clean single-contract run -> a verdict from the options check; the receipt records method and version
def test_clean_single_contract_run_passes():
    result, frame, _ = make_run()
    v = run(result, frame)
    assert v.verdict == "PASS", v.reasons
    assert v.details["absolute_cap"].breached is False
    s = v.details["settings"]
    assert s["multiplier"] == MULT and s["cost_per_contract"] == CPC and s["capital_base"] == CAPITAL
    assert s["option_price_type"] == "mid"
    assert v.details["point_in_time"]["answer"] == "yes"
    assert not v.details.get("conditional")


def test_clean_run_is_dispatched_from_check_pnl_linear():
    result, frame, _ = make_run()
    v = check_pnl_linear(result, frame, EXECUTION, block(), PPY)
    assert v.verdict == "PASS", v.reasons
    assert v.details["settings"]["option_price_type"] == "mid"


def test_s2_records_method_and_version_for_an_option_block(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "s2_pnl_linear_optin_helpers_o", Path(__file__).with_name("test_s2_pnl_linear_optin.py"))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    mult, cpc, capital = 5.0, 2.0, 100000.0
    opt_block = {
        "position_units": "quantity", "asset_class": "option", "adjustment": {"close": "none"},
        "strike": 100.0, "expiry": "2099-01-17", "right": "call", "option_price_type": "mid",
        "multiplier": mult, "cost_per_contract": cpc, "starting_capital": capital,
        "contract_currency": "USD", "account_currency": "USD", "currency_rounding_unit": 0.01,
        "point_in_time": "yes", "publication_column": "pub", "max_staleness": 2,
        "max_unexplained_bp_per_year": 50.0, "max_unexplained_share": 0.2,
    }
    edit = f"((position.shift(1) * px.diff() * {mult} - {cpc} * change.abs().shift(1)) / {capital})"
    s2 = helpers.run_s2(tmp_path, monkeypatch, reconstruction=opt_block, adapter_edit=edit,
                        spec_edit=lambda s: s.pop("methods", None))
    p = s2.params["pnl_linear"]
    assert p["verdict"] != "BLOCKED", helpers.text(s2)
    assert s2.verdict is Verdict.PASS, helpers.text(s2)
    assert s2.methods["s2.pnl_linear"] == "pnl_linear@1"
    assert p["method"] == "s2.pnl_linear" and p["version"] == "pnl_linear@1"
    assert p["settings"]["option_price_type"] == "mid" and p["settings"]["multiplier"] == mult


# (b) chain input: one row per bar is required
def test_chain_is_accepted_when_strike_expiry_and_right_select_one_row_per_bar():
    result, frame, _ = make_run()
    chain = as_chain(frame)
    assert chain.index.duplicated().any()
    v = run(result, chain)
    assert v.verdict == "PASS", v.reasons


def test_chain_with_other_expiries_and_rights_is_filtered_on_all_three():
    result, frame, _ = make_run()
    chain = pd.concat([as_chain(frame), as_chain(frame, expiry="2026-01-16", others=()),
                       as_chain(frame, right="put", others=())]).sort_index(kind="stable")
    assert run(result, chain).verdict == "PASS"


def test_chain_still_ambiguous_after_selection_is_rejected():
    result, frame, _ = make_run()
    chain = pd.concat([as_chain(frame), frame.assign(px=1.0, strike=STRIKE, expiry=FAR_EXPIRY, right="call")])
    v = run(result, chain.sort_index(kind="stable"))
    assert v.verdict == "BLOCKED"
    assert "more than one row" in text(v) and "strike" in text(v) and "expiry" in text(v) and "right" in text(v)


def test_repeated_bars_without_contract_columns_are_rejected():
    result, frame, _ = make_run()
    chain = pd.concat([frame, frame.assign(px=frame["px"] + 1.0)]).sort_index(kind="stable")
    v = run(result, chain)
    assert v.verdict == "BLOCKED"
    assert "more than one row" in text(v)


def test_declared_contract_absent_from_the_chain_is_blocked():
    result, frame, _ = make_run()
    v = run(result, as_chain(frame), strike=4321.0)
    assert v.verdict == "BLOCKED"
    assert "no row" in text(v) and "strike" in text(v)


# (c) expiry: closed at intrinsic value, cash-settled; the mark must match intrinsic value within the rounding unit
E = 150


def test_expiry_mark_equal_to_intrinsic_passes_and_is_reported_cash_settled():
    result, frame, _ = make_run(expiry_bar=E)
    v = run(result, frame, expiry=expiry_of(frame, E))
    assert v.verdict == "PASS", v.reasons
    assert v.details["expiry"]["settlement"] == "cash_settled_at_intrinsic"
    assert v.details["expiry"]["held_contracts"] == 2
    assert v.details["expiry"]["mismatch"] is False


def test_expiry_mark_within_the_rounding_unit_passes():
    result, frame, _ = make_run(expiry_bar=E, mark_bump=UNIT / 2)
    assert run(result, frame, expiry=expiry_of(frame, E)).verdict == "PASS"


def test_expiry_mark_differing_from_intrinsic_beyond_the_rounding_unit_fails():
    result, frame, _ = make_run(expiry_bar=E, mark_bump=0.5)
    v = run(result, frame, expiry=expiry_of(frame, E))
    assert v.verdict == "FAIL"
    assert "expiry" in text(v) and "intrinsic" in text(v)
    assert v.details["expiry"]["mismatch"] is True


def test_expiry_without_a_position_held_is_not_checked():
    result, frame, _ = make_run(expiry_bar=E, mark_bump=0.5)
    result.positions.iloc[E - 1, 0] = 0  # nothing held on the expiry bar (lag 1); the later bars are flat too
    result.positions.iloc[E - 2, 0] = 0
    v = run(result, frame, expiry=expiry_of(frame, E))
    assert v.details["expiry"]["held_contracts"] == 0
    assert v.details["expiry"]["mismatch"] is False


def test_put_intrinsic_value_is_strike_minus_underlying():
    result, frame, _ = make_run(expiry_bar=E)
    frame["und"] = 2 * STRIKE - frame["und"]  # mirrored underlying: a put on it has the call's intrinsic value
    v = run(result, frame, expiry=expiry_of(frame, E), right="put")
    assert v.verdict == "PASS", v.reasons


def test_position_held_past_expiry_is_blocked_never_filled_forward():
    result, frame, _ = make_run(expiry_bar=E)
    result.positions.iloc[E + 5, 0] = 1  # held on bar E + 6, after expiry
    v = run(result, frame, expiry=expiry_of(frame, E))
    assert v.verdict == "BLOCKED"
    assert "expired" in text(v) and "filled forward" in text(v)


def test_expiry_with_a_held_position_needs_the_underlying_column():
    result, frame, _ = make_run(expiry_bar=E)
    v = run(result, frame, expiry=expiry_of(frame, E), underlying_column=_DROP)
    assert v.verdict == "BLOCKED"
    assert "underlying_column" in text(v)


# (d) stale held price: WARNING, the verdict is conditional, not blocked
def test_stale_held_price_warns_and_marks_the_result_conditional():
    result, frame, _ = make_run()
    pub = frame["pub"].copy()
    pub.iloc[100:110] = pub.iloc[100:110] - pd.Timedelta(days=6)
    frame["pub"] = pub
    v = run(result, frame)
    assert v.verdict == "PASS", v.reasons
    assert v.details["conditional"] is True
    stale = v.details["stale_held_price"]
    assert stale["bars"] and stale["status"] == "WARNING"
    assert any("stale held price" in w for w in v.details["warnings"])
    assert "warning" in text(v) and "conditional" in text(v)
    assert v.details["staleness"].stale  # the receipt keeps the raw staleness finding


def test_future_dated_price_still_fails():
    result, frame, _ = make_run()
    pub = frame["pub"].copy()
    pub.iloc[100:103] = pub.iloc[100:103] + pd.Timedelta(days=3)
    frame["pub"] = pub
    assert run(result, frame).verdict == "FAIL"


# (e) missing cost per contract: the cost sub-check is BLOCKED and names the field
def test_missing_cost_per_contract_blocks_the_cost_sub_check_and_names_the_field():
    result, frame, _ = make_run()
    v = run(result, frame, cost_per_contract=_DROP)
    assert v.verdict == "BLOCKED"
    assert "cost_per_contract" in text(v)
    assert v.details["cost"]["status"] == "BLOCKED"
    assert "cost_per_contract" in v.details["cost"]["reason"]
    assert "default" in text(v)


def test_negative_cost_per_contract_is_blocked_and_named():
    result, frame, _ = make_run()
    v = run(result, frame, cost_per_contract=-0.1)
    assert v.verdict == "BLOCKED" and "cost_per_contract" in text(v)


# (f) point-in-time unknown: a warning and an unknown answer, never a block
def test_unknown_point_in_time_is_a_warning_not_a_block():
    result, frame, _ = make_run()
    v = run(result, frame, point_in_time="unknown")
    assert v.verdict == "PASS", v.reasons
    assert v.details["point_in_time"]["answer"] == "unknown"
    assert "point_in_time" in v.details["unknown_answers"]
    assert any("point-in-time" in w for w in v.details["warnings"])
    assert "warning" in text(v)


def test_absent_point_in_time_is_recorded_as_unknown():
    result, frame, _ = make_run()
    v = run(result, frame, point_in_time=_DROP)
    assert v.verdict == "PASS" and v.details["point_in_time"]["answer"] == "unknown"


def test_a_no_answer_for_point_in_time_is_a_warning_not_a_block():
    result, frame, _ = make_run()
    base = run(result, frame)
    v = run(result, frame, point_in_time="no")
    assert v.verdict == base.verdict == "PASS", v.reasons
    assert v.details["point_in_time"]["answer"] == "no"
    assert any("point-in-time" in w for w in v.details["warnings"])
    assert "warning" in text(v)


# price types
def test_missing_or_invalid_price_type_is_blocked():
    result, frame, _ = make_run()
    v = run(result, frame, option_price_type="bid")
    assert v.verdict == "BLOCKED" and "option_price_type" in text(v)


def test_absent_price_type_is_inferred_from_a_column_named_mid():
    result, frame, _ = make_run()
    frame["mid"] = frame["px"]
    v = run(result, frame, option_price_type=_DROP)
    assert v.verdict == "PASS", v.reasons
    assert v.details["settings"]["option_price_type"] == "mid"
    assert v.details["inferred"]["option_price_type"] == "mid"
    assert "option_price_type" not in v.details.get("unknown_answers", [])


def test_absent_price_type_not_inferable_is_a_warning_naming_it():
    result, frame, _ = make_run()
    v = run(result, frame, option_price_type=_DROP)
    assert v.verdict == "PASS", v.reasons
    assert "option_price_type" in v.details["unknown_answers"]
    assert any("option_price_type" in w for w in v.details["warnings"])


def test_absent_rounding_unit_is_inferred_from_the_account_currency():
    result, frame, _ = make_run()
    v = run(result, frame, currency_rounding_unit=_DROP)
    assert v.verdict == "PASS", v.reasons
    assert v.details["settings"]["currency_rounding_unit"] == UNIT
    assert v.details["inferred"]["currency_rounding_unit"] == UNIT
    assert "currency_rounding_unit" not in v.details.get("unknown_answers", [])


def test_absent_rounding_unit_not_inferable_is_a_warning_naming_it():
    result, frame, _ = make_run()
    v = run(result, frame, currency_rounding_unit=_DROP, contract_currency="ZZZ", account_currency="ZZZ")
    assert v.verdict == "PASS", v.reasons
    assert "currency_rounding_unit" in v.details["unknown_answers"]
    assert any("currency_rounding_unit" in w for w in v.details["warnings"])


@pytest.mark.parametrize("price_type", ["mid", "settlement"])
def test_mid_and_settlement_are_accepted(price_type):
    result, frame, _ = make_run()
    v = run(result, frame, option_price_type=price_type)
    assert v.verdict == "PASS" and v.details["settings"]["option_price_type"] == price_type


def test_last_price_needs_a_trade_timestamp_column():
    result, frame, _ = make_run()
    v = run(result, frame, option_price_type="last")
    assert v.verdict == "BLOCKED" and "trade_time_column" in text(v)
    v = run(result, frame, option_price_type="last", trade_time_column="traded")
    assert v.verdict == "BLOCKED" and "traded" in text(v)  # declared but not in the dataset
    frame["traded"] = frame["pub"]
    v = run(result, frame, option_price_type="last", trade_time_column="traded")
    assert v.verdict == "PASS", v.reasons


def test_stale_last_trade_is_judged_on_the_trade_time():
    result, frame, _ = make_run()
    traded = frame["pub"].copy()
    traded.iloc[100:110] = traded.iloc[100:110] - pd.Timedelta(days=6)
    frame["traded"] = traded
    v = run(result, frame, option_price_type="last", trade_time_column="traded")
    assert v.verdict == "PASS" and v.details["conditional"] is True


# required declared values
@pytest.mark.parametrize("field", ["strike", "expiry", "right", "multiplier", "starting_capital",
                                   "contract_currency", "account_currency"])
def test_missing_required_value_is_blocked_and_named(field):
    result, frame, _ = make_run()
    v = run(result, frame, **{field: _DROP})
    assert v.verdict == "BLOCKED"
    assert field in text(v)


@pytest.mark.parametrize("field,bad", [("multiplier", 0), ("multiplier", -5), ("starting_capital", 0),
                                       ("starting_capital", "100"), ("strike", 0), ("right", "straddle"),
                                       ("expiry", "not a date"), ("currency_rounding_unit", 0)])
def test_invalid_required_value_is_blocked_and_named(field, bad):
    result, frame, _ = make_run()
    v = run(result, frame, **{field: bad})
    assert v.verdict == "BLOCKED"
    assert field in text(v)


def test_currency_mismatch_is_blocked():
    result, frame, _ = make_run()
    v = run(result, frame, contract_currency="EUR")
    assert v.verdict == "BLOCKED" and "currency" in text(v)


def test_fractional_contracts_are_blocked():
    result, frame, _ = make_run()
    result.positions["px"] = result.positions["px"] + 0.5
    v = run(result, frame)
    assert v.verdict == "BLOCKED" and "whole" in text(v)


# the limits of evaluate_pnl_linear stay unchanged
def test_half_multiplier_fails_with_declared_caps():
    result, frame, _ = make_run()
    caps = {"max_unexplained_bp_per_year": 50.0, "max_unexplained_share": 0.2}
    assert run(result, frame, multiplier=MULT / 2, **caps).verdict == "FAIL"
    assert run(result, frame, **caps).verdict == "PASS"


def test_hidden_gain_of_one_bp_per_bar_fails():
    result, frame, _ = make_run(gain_per_bar=1e-4 * CAPITAL)
    assert run(result, frame).verdict == "FAIL"


# configuration: option keys belong to asset_class option only
def test_option_keys_are_rejected_for_other_asset_classes():
    eq = {"position_units": "weight_start_of_bar", "cost_model": {"bps": 1.0}, "asset_class": "equity",
          "min_cost_per_turnover": 0.0, "publication_column": "pub", "max_staleness": 2,
          "adjustment": {"px": "total_return"}}
    for key, val in (("strike", 4000.0), ("right", "call"), ("option_price_type", "mid"), ("point_in_time", "yes")):
        with pytest.raises(ReconstructionConfigError, match=key):
            validate_reconstruction({**eq, key: val})


def test_option_block_returns_a_typed_config():
    cfg = validate_reconstruction(block())
    assert cfg.asset_class == "option" and cfg.strike == STRIKE and cfg.right == "call"
    assert cfg.expiry == FAR_EXPIRY and cfg.option_price_type == "mid" and cfg.cost_per_contract == CPC


def test_option_block_rejects_equity_cost_keys_and_weights():
    with pytest.raises(ReconstructionConfigError, match="cost_model"):
        validate_reconstruction(block(cost_model={"bps": 1.0}))
    with pytest.raises(ReconstructionConfigError, match="position_units"):
        validate_reconstruction(block(position_units="weight_start_of_bar"))


def test_dispatch_is_only_for_the_option_quantity_pair(monkeypatch):
    calls = []
    import qrh.checks.pnl_linear_options as opt

    def spy(*args):
        calls.append(args)
        return "options-verdict"

    monkeypatch.setattr(opt, "check_pnl_linear_options", spy)
    res = RunResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame())
    frame = pd.DataFrame()
    check_pnl_linear(res, frame, None, block(), PPY)
    assert len(calls) == 1
    check_pnl_linear(res, frame, None, block(position_units="weight_start_of_bar"), PPY)
    check_pnl_linear(res, frame, None, block(asset_class="equity"), PPY)
    check_pnl_linear(res, frame, None, None, PPY)
    assert len(calls) == 1

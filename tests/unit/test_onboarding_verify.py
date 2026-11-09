"""Deterministic verification of instrument-model claims against a profile of the data."""

import numpy as np
import pandas as pd
import pytest

from qrh.onboarding.model import CRITICAL, model_from_proposal, parse_proposal
from qrh.onboarding.profile import build_profile
from qrh.onboarding.verify import verify_model
from qrh.synthetic import instruments as inst


def run(frame, proposal, offsets=None, tz_from_file=True):
    model = model_from_proposal(parse_proposal(proposal))
    return verify_model(model, frame, build_profile(frame), offsets=offsets, tz_from_file=tz_from_file)


def statuses(frame, proposal, **kw):
    return {f: r.status for f, r in run(frame, proposal, **kw).items()}


GENERATORS = [inst.equity_daily, inst.futures_daily, inst.fx_quotes, inst.option_snapshots, inst.crypto_perp]


@pytest.mark.parametrize("make", GENERATORS)
def test_a_correct_proposal_has_no_contradiction_and_never_verifies_the_unprovable(make):
    frame, proposal = make()
    st = statuses(frame, proposal)
    assert "contradicted" not in st.values(), run(frame, proposal)
    for f in ("timestamp_semantics", "multiplier", "quote_currency", "tick_size", "roll_rule"):
        assert st.get(f, "unverified") == "unverified"
    assert {"asset_class", "price_columns", "price_units", "frequency", "timezone"} <= {f for f, s in st.items() if s == "verified"}


def test_option_mapping_verifies_on_the_option_dataset():
    frame, proposal = inst.option_snapshots()
    assert statuses(frame, proposal)["option_mapping"] == "verified"


def test_adjustments_that_cannot_be_proved_stay_unverified():
    frame, proposal = inst.equity_daily()
    assert statuses(frame, proposal)["adjustments"] == "unverified"
    f2, p2 = inst.futures_daily()
    assert statuses(f2, p2)["adjustments"] == "unverified"


def test_adjustments_none_on_a_split_containing_series_is_contradicted():
    frame, proposal = inst.equity_daily(splits=True)
    r = run(frame, proposal)["adjustments"]
    assert r.status == "contradicted" and "jump" in r.evidence


def test_a_volume_spike_explains_a_jump():
    frame, proposal = inst.equity_daily(splits=True)
    jump_bar = int(np.argmax(np.abs(np.diff(np.log(frame["close"].to_numpy())))) + 1)
    frame.iloc[jump_bar, frame.columns.get_loc("volume")] = 1_000_000.0
    assert statuses(frame, proposal)["adjustments"] == "unverified"


def test_back_adjusted_claim_is_contradicted_by_level_shifts_at_contract_changes():
    frame, proposal = inst.futures_daily()
    r = run(frame, inst.with_claim(proposal, "adjustments", "back_adjusted"))["adjustments"]
    assert r.status == "contradicted" and "contract" in r.evidence
    plain_frame, p = inst.equity_daily()
    assert statuses(plain_frame, inst.with_claim(p, "adjustments", "back_adjusted"))["adjustments"] == "unverified"


@pytest.mark.parametrize("units", ["fraction", "percent", "rate_percent"])
def test_wrong_units_on_a_price_series_are_contradicted_or_unverified(units):
    frame, proposal = inst.futures_daily()
    assert statuses(frame, inst.with_claim(proposal, "price_units", units))["price_units"] == "contradicted"


def test_fraction_and_percent_units_against_magnitudes():
    idx = pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC", name="timestamp")
    rng = np.random.default_rng(3)
    ret = pd.DataFrame({"r": rng.normal(0, 0.01, 200)}, index=idx)
    base = {"asset_class": "other", "price_columns": ["r"]}

    def prop(units):
        return {"source": "llm", "fields": {**{k: {"value": v} for k, v in base.items()}, "price_units": {"value": units}}}

    assert statuses(ret, prop("fraction"))["price_units"] == "verified"
    assert statuses(ret, prop("percent"))["price_units"] == "contradicted"
    assert statuses(ret * 100, prop("percent"))["price_units"] == "verified"  # looks like returns
    assert statuses(ret * 100 + 1e-9, {**prop("percent")})["price_units"] == "verified"
    assert statuses(ret * 100, prop("fraction"))["price_units"] == "contradicted"
    assert statuses(ret, prop("currency"))["price_units"] == "contradicted"  # negative values are not a price
    assert statuses(ret, prop("other"))["price_units"] == "unverified"


def test_rate_units_from_the_option_rate_column():
    frame, proposal = inst.option_snapshots()
    wrong = inst.with_claim(proposal, "price_columns", ["rate"])
    assert statuses(frame, inst.with_claim(wrong, "price_units", "rate_percent"))["price_units"] == "contradicted"
    assert statuses(frame, inst.with_claim(wrong, "price_units", "fraction"))["price_units"] == "verified"


def test_price_columns_must_exist_be_numeric_finite_and_positive():
    frame, proposal = inst.equity_daily()
    assert statuses(frame, inst.with_claim(proposal, "price_columns", ["closee"]))["price_columns"] == "contradicted"
    fut, fp = inst.futures_daily()
    assert statuses(fut, inst.with_claim(fp, "price_columns", ["contract"]))["price_columns"] == "contradicted"
    neg = frame.copy()
    neg.iloc[5, neg.columns.get_loc("close")] = -1.0
    assert statuses(neg, proposal)["price_columns"] == "contradicted"
    inf = frame.copy()
    inf.iloc[5, inf.columns.get_loc("close")] = np.inf
    assert statuses(inf, proposal)["price_columns"] == "contradicted"


def test_ohlc_and_bid_ask_violations_contradict_above_a_small_share():
    frame, proposal = inst.equity_daily()
    swapped = frame.rename(columns={"high": "low", "low": "high"})
    assert statuses(swapped, proposal)["price_columns"] == "contradicted"
    fx, fp = inst.fx_quotes()
    crossed = fx.rename(columns={"bid": "ask", "ask": "bid"})
    r = run(crossed, fp)["price_columns"]
    assert r.status == "contradicted" and "ask" in r.evidence
    one = fx.copy()
    one.iloc[10, one.columns.get_loc("bid")] = one.iloc[10]["ask"] + 0.001  # 1 of 400 rows: below the small share
    assert statuses(one, fp)["price_columns"] == "verified"


def test_frequency_claimed_versus_inferred():
    frame, proposal = inst.equity_daily()
    assert statuses(frame, inst.with_claim(proposal, "frequency", "1h"))["frequency"] == "contradicted"
    assert statuses(frame, inst.with_claim(proposal, "frequency", "daily"))["frequency"] == "verified"
    assert statuses(frame, inst.with_claim(proposal, "frequency", "tick"))["frequency"] == "unverified"
    fx, fp = inst.fx_quotes()
    assert statuses(fx, inst.with_claim(fp, "frequency", "1D"))["frequency"] == "contradicted"
    cal = pd.DataFrame({"close": np.linspace(1, 2, 30)}, index=pd.date_range("2021-01-01", periods=30, freq="D", tz="UTC", name="timestamp"))
    assert statuses(cal, inst.with_claim(proposal, "frequency", "1B"))["frequency"] == "contradicted"


def test_timezone_is_checked_by_utc_offset_only():
    frame, proposal = inst.equity_daily()
    assert statuses(frame, proposal)["timezone"] == "verified"
    assert statuses(frame, inst.with_claim(proposal, "timezone", "America/New_York"))["timezone"] == "contradicted"
    assert statuses(frame, inst.with_claim(proposal, "timezone", "+00:00"))["timezone"] == "verified"
    assert statuses(frame, inst.with_claim(proposal, "timezone", "Nowhere/Land"))["timezone"] == "unverified"
    winter = np.full(len(frame), -300.0)
    ny = inst.with_claim(proposal, "timezone", "America/New_York")
    short = frame.iloc[:40]  # January to February: standard time all along
    assert statuses(short, ny, offsets=winter[:40])["timezone"] == "verified"
    assert statuses(short, proposal, offsets=winter[:40])["timezone"] == "contradicted"
    assert statuses(short, ny, tz_from_file=False)["timezone"] == "unverified"  # the file said nothing about its zone
    naive = frame.copy()
    naive.index = naive.index.tz_localize(None)
    assert statuses(naive, proposal)["timezone"] == "unverified"


def test_asset_class_structure():
    eq, ep = inst.equity_daily()
    assert statuses(eq, inst.with_claim(ep, "asset_class", "future"))["asset_class"] == "unverified"
    assert statuses(eq, inst.with_claim(ep, "asset_class", "crypto_perp"))["asset_class"] == "unverified"
    assert statuses(eq, inst.with_claim(ep, "asset_class", "option"))["asset_class"] == "unverified"  # no option_mapping
    assert statuses(eq, inst.with_claim(ep, "asset_class", "rate"))["asset_class"] == "unverified"
    opt, op = inst.option_snapshots()
    bad_map = inst.with_claim(op, "option_mapping", {"strike": "strike", "expiry": "expiry", "right": "side"})
    r = run(opt, bad_map)
    assert r["option_mapping"].status == "contradicted" and r["asset_class"].status == "contradicted"
    partial = inst.with_claim(op, "option_mapping", {"strike": "strike", "expiry": "expiry", "right": "right"})
    assert statuses(opt, partial)["option_mapping"] == "unverified"  # incomplete, not wrong
    fut, fp = inst.futures_daily()
    assert statuses(fut, fp)["asset_class"] == "verified"
    perp, pp = inst.crypto_perp()
    no_funding = perp.drop(columns=["funding_rate"])
    assert statuses(no_funding, pp)["asset_class"] == "unverified"
    non_pos = eq.copy()
    non_pos["close"] = -non_pos["close"]
    assert statuses(non_pos, ep)["asset_class"] == "contradicted"


_MAKERS = {"equity": inst.equity_daily, "futures": inst.futures_daily, "fx": inst.fx_quotes,
           "option": inst.option_snapshots, "perp": inst.crypto_perp}
_TRUE = {"equity": "equity", "futures": "future", "fx": "fx_spot", "option": "option", "perp": "crypto_perp"}
_SPOT_LIKE = ("equity", "etf", "fx_spot", "crypto_spot", "index")


@pytest.mark.parametrize("dataset,claim", [
    ("option", "equity"), ("option", "future"), ("option", "fx_spot"), ("option", "crypto_perp"),
    ("futures", "equity"), ("futures", "fx_spot"), ("perp", "equity"), ("perp", "fx_spot"),
])
def test_option_futures_and_perp_signatures_contradict_a_wrong_class(dataset, claim):
    frame, proposal = _MAKERS[dataset]()
    r = run(frame, inst.with_claim(proposal, "asset_class", claim))["asset_class"]
    assert r.status == "contradicted", r
    assert {"option": "strike", "futures": "contract", "perp": "funding"}[dataset] in r.evidence


@pytest.mark.parametrize("dataset,claim", [("equity", "fx_spot"), ("fx", "equity")])
def test_a_spot_like_claim_without_class_specific_evidence_is_unverified(dataset, claim):
    frame, proposal = _MAKERS[dataset]()
    assert statuses(frame, inst.with_claim(proposal, "asset_class", claim))["asset_class"] == "unverified"


@pytest.mark.parametrize("claim", ["crypto_spot", "index"])
def test_spot_like_classes_with_no_available_evidence_stay_unverified(claim):
    frame, proposal = inst.equity_daily()
    assert statuses(frame, inst.with_claim(proposal, "asset_class", claim))["asset_class"] == "unverified"


def test_a_future_claim_needs_a_contract_column_not_just_a_name_hint():
    frame, proposal = inst.equity_daily()
    frame = frame.assign(expiry_note=1.0)  # an expiry-like name alone is not evidence
    assert statuses(frame, inst.with_claim(proposal, "asset_class", "future"))["asset_class"] == "unverified"


def test_the_option_signature_is_also_read_from_a_holding_option_mapping():
    frame, proposal = inst.option_snapshots()
    renamed = frame.rename(columns={"strike": "k_leg", "expiry": "mat_leg", "right": "side_leg"})
    mapping = {"underlying": "underlying_price", "strike": "k_leg", "expiry": "mat_leg", "right": "side_leg",
               "bid": "bid", "ask": "ask", "rate": "rate"}
    p = inst.with_claim(inst.with_claim(proposal, "option_mapping", mapping), "asset_class", "equity")
    assert run(renamed, p)["asset_class"].status == "contradicted"


@pytest.mark.parametrize("dataset", ["equity", "fx"])
def test_correct_spot_like_classes_still_verify(dataset):
    frame, proposal = _MAKERS[dataset]()
    assert statuses(frame, proposal)["asset_class"] == "verified"


def test_evidence_fits_the_claim_limit_and_names_a_checker():
    for make in GENERATORS:
        frame, proposal = make()
        for r in run(frame, proposal).values():
            assert 0 < len(r.evidence) <= 300 and r.checked_by.startswith("code:")


def test_critical_unverifiable_fields_are_never_verified_by_code():
    for make in GENERATORS:
        frame, proposal = make()
        res = run(frame, proposal)
        for f in ("multiplier", "timestamp_semantics"):
            if f in res:
                assert res[f].status == "unverified"
    assert set(CRITICAL) >= {"multiplier", "timestamp_semantics"}


def test_percent_needs_stronger_evidence_than_a_range():
    frame, proposal = inst.equity_daily()  # a price level around 50
    r = run(frame, inst.with_claim(proposal, "price_units", "percent"))["price_units"]
    assert r.status == "unverified" and "name" in r.evidence and "returns" in r.evidence
    idx = pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC", name="timestamp")
    level = pd.DataFrame({"us10y_yield": np.linspace(3.0, 5.0, 200)}, index=idx)  # name hint, level-like
    pr = {"source": "llm", "fields": {"asset_class": {"value": "rate"}, "price_columns": {"value": ["us10y_yield"]},
                                       "price_units": {"value": "percent"}}}
    assert statuses(level, pr)["price_units"] == "verified"
    level = level.rename(columns={"us10y_yield": "x"})
    assert statuses(level, pr | {"fields": {**pr["fields"], "price_columns": {"value": ["x"]}}})["price_units"] == "unverified"
    fut, fp = inst.futures_daily()  # still contradicted: too large to be a percent
    assert statuses(fut, inst.with_claim(fp, "price_units", "percent"))["price_units"] == "contradicted"

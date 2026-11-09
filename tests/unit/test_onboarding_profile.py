"""The dataset profile: aggregates only, never a row-level value."""

import json

import numpy as np
import pandas as pd
import pytest

from qrh.onboarding.profile import build_profile, parse_frequency, profile_hash, shape_mask, step_label
from qrh.synthetic import instruments as inst


def test_shape_mask_turns_digits_and_letters_into_classes():
    assert shape_mask("ESZ4") == "AAA9"
    assert shape_mask("ab-12.X") == "aa-99.A"
    assert shape_mask("A" * 40).endswith("…")


def test_equity_profile_has_time_index_columns_and_flags():
    df, _ = inst.equity_daily()
    p = build_profile(df)
    ti = p["time_index"]
    assert p["rows"] == 260 and ti["timezone"] == "UTC" and ti["inferred_frequency"] == "1B"
    assert ti["monotonic_increasing"] and ti["duplicates"] == 0 and ti["gap_count"] == 0
    assert ti["first"].startswith("2021-01-04")
    close = p["columns"]["close"]
    assert close["kind"] == "numeric" and close["strictly_positive"] and not close["integer_valued"]
    assert {"min", "max", "quantiles"} <= set(close) and set(close["quantiles"]) == {"p01", "p25", "p50", "p75", "p99"}
    assert p["columns"]["volume"]["integer_valued"]


def test_no_values_drops_every_aggregate_value_and_the_time_range():
    df, _ = inst.option_snapshots()
    p = build_profile(df, no_values=True)
    text = json.dumps(p)
    assert "first" not in p["time_index"] and "last" not in p["time_index"]
    for col in p["columns"].values():
        assert not ({"min", "max", "quantiles", "values"} & set(col))
    assert all("median_abs" not in r and not isinstance(r.get("scale"), float) for r in p["relationships"])
    assert '"first"' not in text


def test_relationships_are_found_from_the_numbers():
    df, _ = inst.equity_daily()
    rel = build_profile(df)["relationships"]
    kinds = {(r["kind"], r.get("a"), r.get("b"), r.get("value")) for r in rel}
    assert ("a_ge_b", "high", "low", None) in kinds and ("a_ge_b", "close", "low", None) in kinds
    assert ("inside", None, None, "close") in kinds
    fx, _ = inst.fx_quotes()
    assert any(r["kind"] == "a_ge_b" and (r["a"], r["b"]) == ("ask", "bid") for r in build_profile(fx)["relationships"])


def test_return_percent_and_rate_candidates():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC", name="timestamp")
    df = pd.DataFrame({"ret": rng.normal(0, 0.01, 300), "pct": rng.uniform(0, 100, 300), "r": np.full(300, 0.045)}, index=idx)
    kinds = {(r["kind"], r["column"]) for r in build_profile(df)["relationships"] if "column" in r}
    assert ("return_like", "ret") in kinds and ("percent_like", "pct") in kinds and ("rate_like", "r") in kinds


def test_name_hints_are_hints_only():
    df, _ = inst.option_snapshots()
    hints = {(h["column"], h["hint"]) for h in build_profile(df)["hints"]}
    assert {("strike", "strike"), ("expiry", "expiry"), ("right", "right"), ("bid", "bid"), ("delta", "delta"),
            ("iv", "iv"), ("rate", "rate"), ("volume", "volume"), ("open_interest", "open interest")} <= hints
    df2 = pd.DataFrame({"adj_close": [1.0] * 20}, index=pd.date_range("2021-01-01", periods=20, tz="UTC"))
    assert ("adj_close", "adjustments") in {(h["column"], h["hint"]) for h in build_profile(df2)["hints"]}


def test_string_columns_show_shapes_and_flag_values_only():
    df, _ = inst.option_snapshots()
    cols = build_profile(df)["columns"]
    assert cols["right"]["flag_like"] and cols["right"]["values"] == ["C", "P"]
    assert cols["expiry"]["shapes"][0]["shape"] == "9999-99-99" and "values" not in cols["expiry"]
    fut, _ = inst.futures_daily()
    contract = build_profile(fut)["columns"]["contract"]
    assert contract["shapes"][0]["shape"] == "AAA9" and "values" not in contract


def test_an_identifier_named_column_never_shows_values_even_if_flag_shaped():
    idx = pd.date_range("2021-01-01", periods=30, freq="D", tz="UTC", name="timestamp")
    df = pd.DataFrame({"symbol": ["GM", "KO", "T"] * 10, "side": ["B", "S"] * 15}, index=idx)
    cols = build_profile(df)["columns"]
    assert "values" not in cols["symbol"] and cols["side"]["values"] == ["B", "S"]


def test_split_like_jumps_are_counted_for_positive_series():
    plain, _ = inst.equity_daily()
    split, _ = inst.equity_daily(splits=True)
    assert build_profile(plain)["columns"]["close"]["split_like_jumps"] == 0
    assert build_profile(split)["columns"]["close"]["split_like_jumps"] >= 1


def test_time_index_gaps_and_duplicates():
    idx = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=30, freq="D", tz="UTC").delete([10, 11]).append(
        pd.DatetimeIndex(["2021-01-02"], tz="UTC")), name="timestamp")
    ti = build_profile(pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx))["time_index"]
    assert ti["gap_count"] == 1 and ti["duplicates"] == 1 and not ti["monotonic_increasing"]


def test_profile_is_deterministic_and_hashable():
    df, _ = inst.crypto_perp()
    assert profile_hash(build_profile(df)) == profile_hash(build_profile(df))
    assert profile_hash(build_profile(df)) != profile_hash(build_profile(df, no_values=True))


@pytest.mark.parametrize("text,expected", [
    ("1D", (86400.0, False)), ("daily", (86400.0, False)), ("1h", (3600.0, False)), ("5min", (300.0, False)),
    ("1B", (86400.0, True)), ("weekly", (604800.0, False)), ("tick", None), ("", None)])
def test_parse_frequency(text, expected):
    assert parse_frequency(text) == expected


def test_step_label():
    assert (step_label(86400), step_label(3600), step_label(60), step_label(7)) == ("1D", "1h", "1min", "7s")

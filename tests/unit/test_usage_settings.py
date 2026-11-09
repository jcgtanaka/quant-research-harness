import json

import pytest

from qrh.usage.settings import (
    SettingsError, init_files, load_prices, load_thresholds, price_cost,
)
from usage_helpers import default_prices, default_thresholds, write_settings


def test_missing_thresholds_file_fails_naming_the_file(tmp_path):
    with pytest.raises(SettingsError, match="thresholds.json"):
        load_thresholds(tmp_path)


@pytest.mark.parametrize("bad", [
    {}, {"cache_cold_pct": 60},
    {**default_thresholds(), "cache_cold_pct": "60"},
    {**default_thresholds(), "cache_cold_pct": 150},
    {**default_thresholds(), "mechanical_tool_share": 1.5},
    {**default_thresholds(), "mechanical_min_tools": 0},
    {**default_thresholds(), "context_high_tokens": True},
    {**default_thresholds(), "surprise": 1},
])
def test_invalid_thresholds_fail_closed(tmp_path, bad):
    write_settings(tmp_path, thresholds=bad)
    with pytest.raises(SettingsError, match="thresholds.json"):
        load_thresholds(tmp_path)


def test_unparsable_thresholds_fail_closed(tmp_path):
    (tmp_path / "usage").mkdir()
    (tmp_path / "usage" / "thresholds.json").write_text("{nope")
    with pytest.raises(SettingsError, match="thresholds.json"):
        load_thresholds(tmp_path)


def test_valid_thresholds_load(tmp_path):
    write_settings(tmp_path)
    t = load_thresholds(tmp_path)
    assert (t.cache_cold_pct, t.context_high_tokens, t.mechanical_tool_share, t.mechanical_min_tools) == (
        60, 300000, 0.8, 5)


def test_missing_prices_is_none_not_an_error(tmp_path):
    assert load_prices(tmp_path) is None


def test_invalid_prices_fail_closed(tmp_path):
    write_settings(tmp_path, prices={"models": {"m": {"input": 1}}})
    with pytest.raises(SettingsError, match="prices.json"):
        load_prices(tmp_path)
    write_settings(tmp_path, prices={"models": {"m": {**default_prices()["models"]["model-big"], "tier": "huge"}}})
    with pytest.raises(SettingsError, match="tier"):
        load_prices(tmp_path)


def test_price_cost_uses_split_cache_prices(tmp_path):
    write_settings(tmp_path)
    prices = load_prices(tmp_path)
    cost = price_cost(prices.models["model-big"], input=1_000_000, output=1_000_000, cache_read=1_000_000,
                      cache_write_5m=1_000_000, cache_write_1h=1_000_000)
    assert cost == pytest.approx(10 + 50 + 1 + 12.5 + 20)


def test_init_writes_marked_examples_and_never_overwrites(tmp_path):
    written = init_files(tmp_path)
    assert {p.name for p in written} == {"thresholds.json", "prices.json"}
    prices = json.loads((tmp_path / "usage" / "prices.json").read_text())
    assert "EXAMPLE" in prices["_note"] and "pricing page" in prices["_note"]
    assert "EXAMPLE" in json.loads((tmp_path / "usage" / "thresholds.json").read_text())["_note"]
    load_thresholds(tmp_path)
    load_prices(tmp_path)
    assert init_files(tmp_path) == []
    assert len(init_files(tmp_path, force=True)) == 2

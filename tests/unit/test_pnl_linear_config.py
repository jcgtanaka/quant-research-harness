"""Validation of the ``reconstruction`` settings block of the pnl.linear method (synthetic examples only)."""

from __future__ import annotations

import copy

import pytest

from qrh.checks.pnl_linear_config import (
    ReconstructionConfig,
    ReconstructionConfigError,
    check_frame_columns,
    validate_reconstruction,
)


def _block(**over):
    block = {
        "position_units": "quantity",
        "cost_model": {"kind": "proportional", "rate": 0.0005},
        "asset_class": "equity",
        "min_cost_per_turnover": 0.0001,
        "publication_column": "published_at",
        "max_staleness": 3,
        "adjustment": {"close": "total_return"},
    }
    block = copy.deepcopy(block)
    for key, value in over.items():
        if value is _DROP:
            block.pop(key, None)
        else:
            block[key] = value
    return block


_DROP = object()


def test_valid_block_returns_config():
    cfg = validate_reconstruction(_block())
    assert isinstance(cfg, ReconstructionConfig)
    assert cfg.position_units == "quantity"
    assert cfg.asset_class == "equity"
    assert cfg.cost_model == {"kind": "proportional", "rate": 0.0005}
    assert cfg.cost_series_column is None
    assert cfg.min_cost_per_turnover == 0.0001
    assert cfg.publication_column == "published_at"
    assert cfg.max_staleness == 3
    assert cfg.adjustment == {"close": "total_return"}
    assert cfg.income == []
    assert cfg.halt_flag_column is None


def test_valid_block_with_optionals():
    cfg = validate_reconstruction(
        _block(
            position_units="weight_start_of_bar",
            cost_model=_DROP,
            cost_series_column="cost_bp",
            max_unexplained_bp_per_year=25,
            max_unexplained_share=0.5,
            income=[{"kind": "carry", "column": "carry"}],
            halt_flag_column="halted",
            adjustment={"close": "none"},
        )
    )
    assert cfg.cost_series_column == "cost_bp" and cfg.cost_model is None
    assert cfg.max_unexplained_bp_per_year == 25
    assert cfg.max_unexplained_share == 0.5
    assert cfg.income == [{"kind": "carry", "column": "carry"}]
    assert cfg.halt_flag_column == "halted"
    assert cfg.pending == []


def test_not_a_mapping():
    with pytest.raises(ReconstructionConfigError):
        validate_reconstruction(["position_units"])


@pytest.mark.parametrize("value", [_DROP, "shares", None, 1])
def test_position_units_required_and_closed(value):
    with pytest.raises(ReconstructionConfigError, match="position_units"):
        validate_reconstruction(_block(position_units=value))


def test_both_cost_fields_is_error():
    with pytest.raises(ReconstructionConfigError, match="exactly one of cost_model or cost_series_column"):
        validate_reconstruction(_block(cost_series_column="cost_bp"))


def test_neither_cost_field_is_error():
    with pytest.raises(ReconstructionConfigError, match="exactly one of cost_model or cost_series_column"):
        validate_reconstruction(_block(cost_model=_DROP))


def test_cost_field_types():
    with pytest.raises(ReconstructionConfigError, match="cost_model"):
        validate_reconstruction(_block(cost_model="flat"))
    with pytest.raises(ReconstructionConfigError, match="cost_series_column"):
        validate_reconstruction(_block(cost_model=_DROP, cost_series_column=""))


@pytest.mark.parametrize("asset", ["option", "future", "crypto_perp", "unknown"])
def test_unsupported_asset_class_is_blocked(asset):
    with pytest.raises(ReconstructionConfigError, match="use the method for this type") as exc:
        validate_reconstruction(_block(asset_class=asset))
    assert exc.value.verdict == "BLOCKED"


def test_missing_asset_class_is_error():
    with pytest.raises(ReconstructionConfigError, match="asset_class"):
        validate_reconstruction(_block(asset_class=_DROP))


@pytest.mark.parametrize("asset", ["equity", "etf", "fx_spot", "crypto_spot"])
def test_supported_asset_classes(asset):
    assert validate_reconstruction(_block(asset_class=asset)).asset_class == asset


def test_missing_min_cost_is_error_not_zero():
    with pytest.raises(ReconstructionConfigError, match="min_cost_per_turnover") as exc:
        validate_reconstruction(_block(min_cost_per_turnover=_DROP))
    assert exc.value.verdict == "BLOCKED"


@pytest.mark.parametrize("value", [-0.1, "0.1", True, float("nan")])
def test_min_cost_must_be_number_ge_zero(value):
    with pytest.raises(ReconstructionConfigError, match="min_cost_per_turnover"):
        validate_reconstruction(_block(min_cost_per_turnover=value))


def test_explicit_zero_min_cost_is_accepted():
    assert validate_reconstruction(_block(min_cost_per_turnover=0)).min_cost_per_turnover == 0


def test_missing_publication_column_is_blocked():
    with pytest.raises(ReconstructionConfigError, match="publication_column") as exc:
        validate_reconstruction(_block(publication_column=_DROP))
    assert exc.value.verdict == "BLOCKED"


def test_empty_publication_column_is_error():
    with pytest.raises(ReconstructionConfigError, match="publication_column"):
        validate_reconstruction(_block(publication_column=""))


def test_missing_max_staleness_is_blocked():
    with pytest.raises(ReconstructionConfigError, match="max_staleness") as exc:
        validate_reconstruction(_block(max_staleness=_DROP))
    assert exc.value.verdict == "BLOCKED"


@pytest.mark.parametrize("value", [0, -2, "3", True, float("inf")])
def test_max_staleness_must_be_positive_number(value):
    with pytest.raises(ReconstructionConfigError, match="max_staleness"):
        validate_reconstruction(_block(max_staleness=value))


def test_unknown_key_is_error():
    with pytest.raises(ReconstructionConfigError, match=r"unknown key\(s\) \['lag'\]"):
        validate_reconstruction(_block(lag=1))


def test_absent_caps_take_noise_scaled_defaults_and_nothing_is_pending():
    # derived from the remainder's own noise, not fixed numbers
    cfg = validate_reconstruction(_block())
    assert cfg.max_unexplained_bp_per_year == "noise_scaled"
    assert cfg.max_unexplained_share == "noise_scaled"
    assert cfg.pending == []


def test_explicit_none_cap_is_kept_and_listed_as_pending():
    cfg = validate_reconstruction(_block(max_unexplained_bp_per_year=None))
    assert cfg.max_unexplained_bp_per_year is None
    assert cfg.max_unexplained_share == "noise_scaled"
    assert cfg.pending == ["max_unexplained_bp_per_year"]
    cfg = validate_reconstruction(_block(max_unexplained_share=None, max_unexplained_bp_per_year=10))
    assert cfg.max_unexplained_bp_per_year == 10
    assert cfg.pending == ["max_unexplained_share"]


@pytest.mark.parametrize("value", [-1, "5", True, float("nan")])
def test_bp_cap_must_be_number_ge_zero(value):
    with pytest.raises(ReconstructionConfigError, match="max_unexplained_bp_per_year"):
        validate_reconstruction(_block(max_unexplained_bp_per_year=value))


def test_bp_cap_zero_is_accepted():
    assert validate_reconstruction(_block(max_unexplained_bp_per_year=0)).max_unexplained_bp_per_year == 0


@pytest.mark.parametrize("value", [0, -0.1, 1.5, "0.5", True])
def test_share_cap_range(value):
    with pytest.raises(ReconstructionConfigError, match="max_unexplained_share"):
        validate_reconstruction(_block(max_unexplained_share=value))


@pytest.mark.parametrize(
    "income",
    [
        "dividend",
        [{"kind": "coupon", "column": "c"}],
        [{"kind": "dividend"}],
        [{"kind": "dividend", "column": ""}],
        [{"kind": "dividend", "column": "d", "extra": 1}],
        ["dividend"],
    ],
)
def test_income_entries_validated(income):
    with pytest.raises(ReconstructionConfigError, match="income"):
        validate_reconstruction(_block(adjustment={"close": "none"}, income=income))


def test_halt_flag_column_must_be_string():
    with pytest.raises(ReconstructionConfigError, match="halt_flag_column"):
        validate_reconstruction(_block(halt_flag_column=3))


@pytest.mark.parametrize("adj", [_DROP, {}, "none", {"close": "dividend"}, {"": "none"}, {"close": None}])
def test_adjustment_required_and_closed(adj):
    with pytest.raises(ReconstructionConfigError, match="adjustment"):
        validate_reconstruction(_block(adjustment=adj))


def test_total_return_with_dividend_income_is_error():
    with pytest.raises(ReconstructionConfigError, match="total_return.*dividend|dividend.*total_return"):
        validate_reconstruction(
            _block(adjustment={"close": "total_return"}, income=[{"kind": "dividend", "column": "div"}])
        )


def test_dividend_income_with_split_is_valid_in_block():
    cfg = validate_reconstruction(_block(adjustment={"close": "split"}, income=[{"kind": "dividend", "column": "div"}]))
    assert cfg.adjustment == {"close": "split"}


def test_frame_check_split_only_without_dividend_column():
    cfg = validate_reconstruction(_block(adjustment={"close": "split"}))
    with pytest.raises(ReconstructionConfigError, match="close.*dividend"):
        check_frame_columns(cfg, {"close", "published_at"})


def test_frame_check_split_with_declared_but_absent_dividend_column():
    cfg = validate_reconstruction(_block(adjustment={"close": "split"}, income=[{"kind": "dividend", "column": "div"}]))
    with pytest.raises(ReconstructionConfigError, match="div"):
        check_frame_columns(cfg, {"close", "published_at"})


def test_frame_check_split_with_dividend_column_passes():
    cfg = validate_reconstruction(_block(adjustment={"close": "split"}, income=[{"kind": "dividend", "column": "div"}]))
    check_frame_columns(cfg, {"close", "published_at", "div"})


def test_frame_check_fx_spot_swap_without_rate_column():
    cfg = validate_reconstruction(
        _block(asset_class="fx_spot", adjustment={"close": "none"}, income=[{"kind": "swap", "column": "swap_rate"}])
    )
    with pytest.raises(ReconstructionConfigError, match="swap_rate"):
        check_frame_columns(cfg, {"close", "published_at"})
    check_frame_columns(cfg, {"close", "published_at", "swap_rate"})


def test_frame_check_fx_spot_financing_without_rate_column():
    cfg = validate_reconstruction(
        _block(asset_class="fx_spot", adjustment={"close": "none"}, income=[{"kind": "financing", "column": "fin"}])
    )
    with pytest.raises(ReconstructionConfigError, match="fin"):
        check_frame_columns(cfg, {"close", "published_at"})


def test_fx_spot_swap_entry_needs_a_column_in_block():
    with pytest.raises(ReconstructionConfigError, match="income"):
        validate_reconstruction(_block(asset_class="fx_spot", adjustment={"close": "none"}, income=[{"kind": "swap"}]))


def test_frame_check_requires_declared_columns_present():
    cfg = validate_reconstruction(_block(publication_column="pub", halt_flag_column="halt"))
    with pytest.raises(ReconstructionConfigError, match="pub"):
        check_frame_columns(cfg, {"close"})
    with pytest.raises(ReconstructionConfigError, match="halt"):
        check_frame_columns(cfg, {"close", "pub"})
    check_frame_columns(cfg, {"close", "pub", "halt"})


def test_frame_check_adjusted_price_column_missing():
    cfg = validate_reconstruction(_block())
    with pytest.raises(ReconstructionConfigError, match="close"):
        check_frame_columns(cfg, {"published_at"})

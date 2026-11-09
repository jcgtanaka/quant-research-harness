"""Validation of the ``reconstruction`` settings block of the pnl.linear method.

The block is validated by the method itself, separately from the ``execution`` block, with a closed key set.
``validate_reconstruction`` checks the block alone and returns a typed ``ReconstructionConfig``; every defect raises
``ReconstructionConfigError`` (a ``ValueError``) whose ``verdict`` is ``BLOCKED``: the check cannot run on this
configuration. Required fields are never defaulted. The two limits, when absent, are scaled to the noise measured in
the strategy's own remainder (``NOISE_SCALED``, K = ``NOISE_SCALE_K`` standard deviations); a number overrides the
scaled limit; an explicit ``None`` keeps the cap off and names it in
``ReconstructionConfig.pending``, which makes the verdict INCONCLUSIVE.

Rules that need the dataset columns cannot be decided from the block, so ``check_frame_columns`` is a separate pure
function that takes the set of available column names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

POSITION_UNITS = ("quantity", "weight_start_of_bar")
ASSET_CLASSES = ("equity", "etf", "fx_spot", "crypto_spot")
INCOME_KINDS = ("dividend", "carry", "swap", "financing")
ADJUSTMENTS = ("none", "split", "total_return", "back_adjusted", "forward_adjusted")
FUTURES = "futures"  # opt-in asset class of the futures first slice (F1); "future" stays unsupported here
FUTURES_ADJUSTMENTS = ("none", "back_adjusted", "forward_adjusted")
ROLL_METHODS = ("calendar", "volume", "open_interest")
BROKER_KEYS = ("broker", "product", "declared_equity")  # cost lookup from a confirmed broker commission record
FUTURES_KEYS = ("multiplier", "roll_rule", "cost_per_contract", "starting_capital", "margin_per_contract",
                "contract_currency", "account_currency", *BROKER_KEYS)
CONTRACT_COLUMN = "contract"
OPTION = "option"  # opt-in asset class of the options first slice (O1), with position_units quantity
OPTION_PRICE_TYPES = ("mid", "settlement", "last")
OPTION_RIGHTS = ("call", "put")
OPTION_ONLY_KEYS = ("strike", "expiry", "right", "option_price_type", "trade_time_column", "underlying_column",
                    "currency_rounding_unit", "point_in_time")
OPTION_SHARED_KEYS = ("multiplier", "cost_per_contract", "starting_capital", "contract_currency", "account_currency",
                      *BROKER_KEYS)
RATE_KINDS = ("swap", "financing")
from qrh.checks.pnl_linear_bounds import NOISE_SCALE_K, NOISE_SCALED  # noqa: E402,F401

# Defaults are derived from the remainder's own noise (K = NOISE_SCALE_K standard deviations), not fixed values.
# A number overrides; an explicit None disables a cap (pending).
DEFAULT_MAX_UNEXPLAINED_BP_PER_YEAR = NOISE_SCALED
DEFAULT_MAX_UNEXPLAINED_SHARE = NOISE_SCALED
_KEYS = {
    "position_units", "cost_model", "cost_series_column", "asset_class", "min_cost_per_turnover",
    "max_unexplained_bp_per_year", "max_unexplained_share", "publication_column", "max_staleness",
    "income", "halt_flag_column", "adjustment", *FUTURES_KEYS, *OPTION_ONLY_KEYS,
}


class ReconstructionConfigError(ValueError):
    """The ``reconstruction`` block cannot be used; the verdict for the check is BLOCKED."""

    verdict = "BLOCKED"


@dataclass
class ReconstructionConfig:
    position_units: str
    asset_class: str
    min_cost_per_turnover: float
    publication_column: str
    max_staleness: float
    adjustment: dict[str, str]
    cost_model: dict[str, Any] | None = None
    cost_series_column: str | None = None
    max_unexplained_bp_per_year: float | str | None = DEFAULT_MAX_UNEXPLAINED_BP_PER_YEAR  # noise-scaled
    max_unexplained_share: float | str | None = DEFAULT_MAX_UNEXPLAINED_SHARE  # noise-scaled
    income: list[dict[str, str]] = field(default_factory=list)
    halt_flag_column: str | None = None
    pending: list[str] = field(default_factory=list)  # caps explicitly set to None
    # futures (asset_class "futures") only; None for every other asset class
    multiplier: float | None = None
    cost_per_contract: float | None = None
    starting_capital: float | None = None
    margin_per_contract: float | None = None  # optional: absent only skips the margin sub-check
    roll_rule: dict[str, str] | None = None
    contract_currency: str | None = None
    account_currency: str | None = None
    # options (asset_class "option") only; None for every other asset class. cost_per_contract and the currency
    # fields above are shared with futures; there, a missing cost_per_contract is a configuration error, here the
    # options check blocks its cost sub-check instead.
    strike: float | None = None
    expiry: str | None = None  # ISO date of the declared contract
    right: str | None = None
    option_price_type: str | None = None
    trade_time_column: str | None = None
    underlying_column: str | None = None
    currency_rounding_unit: float | None = None
    point_in_time: str = "unknown"  # "yes" or "unknown"; an absent answer is unknown
    # futures and options: the cost may come from a confirmed broker commission record when cost_per_contract is absent
    broker: str | None = None
    product: str | None = None
    declared_equity: float | None = None


_PRICE_TYPE_WORDS = {"mid": "mid", "settlement": "settlement", "settle": "settlement", "last": "last"}
_ROUNDING_UNIT_BY_CURRENCY = {
    "USD": 0.01, "EUR": 0.01, "GBP": 0.01, "CHF": 0.01, "CAD": 0.01, "AUD": 0.01, "NZD": 0.01, "SGD": 0.01,
    "HKD": 0.01, "JPY": 1.0, "KRW": 1.0,
}


def infer_option_price_type(price_column: str, available: set[str] | frozenset[str]) -> str | None:
    """The price type named by the price column, else by exactly one dataset column named mid, settlement or last;
    None when it cannot be told."""
    own = _PRICE_TYPE_WORDS.get(price_column.strip().lower())
    if own is not None:
        return own
    found = {_PRICE_TYPE_WORDS[c.strip().lower()] for c in available
             if isinstance(c, str) and c.strip().lower() in _PRICE_TYPE_WORDS}
    return found.pop() if len(found) == 1 else None


def infer_rounding_unit(account_currency: str | None) -> float | None:
    """The smallest step of an option price in the account currency, None when the currency is not known here."""
    return _ROUNDING_UNIT_BY_CURRENCY.get((account_currency or "").strip().upper())


def _err(msg: str) -> ReconstructionConfigError:
    return ReconstructionConfigError(f"reconstruction.{msg}")


def _is_number(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def _is_name(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


def _validate_income(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise _err("income: must be a list of {kind, column} mappings")
    out: list[dict[str, str]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != {"kind", "column"}:
            raise _err(f"income[{i}]: must be a mapping with exactly kind and column")
        if entry["kind"] not in INCOME_KINDS:
            raise _err(f"income[{i}].kind: must be one of {list(INCOME_KINDS)}")
        if not _is_name(entry["column"]):
            raise _err(f"income[{i}].column: required, a column name")
        out.append({"kind": entry["kind"], "column": entry["column"]})
    return out


def _validate_adjustment(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise _err("adjustment: required, a non-empty mapping from price column to one of "
                   f"{list(ADJUSTMENTS)}")
    for col, adj in raw.items():
        if not _is_name(col):
            raise _err("adjustment: keys must be price column names")
        if adj not in ADJUSTMENTS:
            raise _err(f"adjustment[{col!r}]: must be one of {list(ADJUSTMENTS)}")
    return dict(raw)


def _validate_broker(block: dict[str, Any]) -> dict[str, Any]:
    """Optional broker, product and declared_equity. Only the names are checked here; the record is resolved later."""
    out: dict[str, Any] = {}
    for key in ("broker", "product"):
        if key in block:
            if not _is_name(block[key]) or block[key] != block[key].strip():
                raise _err(f"{key}: must be a non-empty name")
            out[key] = block[key]
    if "declared_equity" in block:
        e = block["declared_equity"]
        if not _is_number(e) or e < 0:
            raise _err("declared_equity: must be a number >= 0 in account currency")
        out["declared_equity"] = float(e)
    return out


HARNESS_ADJUSTMENT = "total_return"  # data the harness builds with is adjusted (unadjusted data is converted first)


def fill_built_adjustment(block: Any, price_columns: list[str]) -> Any:
    """A backtest the harness built needs no declared adjustment: the harness knows its own data. When the block has
    none, the price columns of the execution block get the harness type. A declared adjustment is kept as given."""
    if not isinstance(block, dict) or "adjustment" in block or not price_columns:
        return block
    return {**block, "adjustment": {c: HARNESS_ADJUSTMENT for c in price_columns}}


def has_broker_cost_source(config: ReconstructionConfig) -> bool:
    return config.broker is not None and config.product is not None


def _validate_futures(block: dict[str, Any], adjustment: dict[str, str]) -> dict[str, Any]:
    """The futures-only fields (D1-D7). Every required one is named in its error and never defaulted."""
    if block["position_units"] != "quantity":
        raise _err("position_units: futures hold a whole number of contracts, use quantity")
    for key in ("cost_model", "cost_series_column", "min_cost_per_turnover"):
        if key in block:
            raise _err(f"{key}: not used for futures, declare cost_per_contract")
    if len(adjustment) != 1:
        raise _err("adjustment: futures F1 covers one instrument, declare exactly one price column")
    ((_, adj),) = adjustment.items()
    if adj not in FUTURES_ADJUSTMENTS:
        raise _err(f"adjustment: futures accept {list(FUTURES_ADJUSTMENTS)}, not {adj!r}")
    out: dict[str, Any] = {}
    for key, rule in (("multiplier", "a number > 0"), ("starting_capital", "a number > 0, declared (no default)")):
        v = block.get(key)
        if not _is_number(v) or v <= 0:
            raise _err(f"{key}: required for futures, {rule}")
        out[key] = float(v)
    out.update(_validate_broker(block))
    v = block.get("cost_per_contract")
    if "cost_per_contract" in block or not ("broker" in out and "product" in out):
        if not _is_number(v) or v < 0:
            raise _err("cost_per_contract: required for futures, a number >= 0 in account currency (never "
                       "defaulted), or declare broker and product to use a confirmed commission record")
        out["cost_per_contract"] = float(v)
    if "margin_per_contract" in block:
        m = block["margin_per_contract"]
        if not _is_number(m) or m < 0:
            raise _err("margin_per_contract: must be a number >= 0 (omit it to skip the margin sub-check)")
        out["margin_per_contract"] = float(m)
    for key in ("contract_currency", "account_currency"):
        if not _is_name(block.get(key)):
            raise _err(f"{key}: required for futures, a currency code (no FX conversion in F1)")
    if block["contract_currency"] != block["account_currency"]:
        raise _err(f"currency: contract currency {block['contract_currency']!r} differs from account currency "
                   f"{block['account_currency']!r}; F1 does no FX conversion")
    out["contract_currency"], out["account_currency"] = block["contract_currency"], block["account_currency"]
    rule = block.get("roll_rule")
    if "roll_rule" in block:
        if not isinstance(rule, dict) or set(rule) != {"method"} or rule["method"] not in ROLL_METHODS:
            raise _err(f"roll_rule.method: must be a mapping {{method: one of {list(ROLL_METHODS)}}}")
        out["roll_rule"] = {"method": rule["method"]}
    elif adj == "none":
        raise _err(f"roll_rule: an unadjusted continuous series needs a machine-readable roll_rule "
                   f"{{method: one of {list(ROLL_METHODS)}}} and a {CONTRACT_COLUMN!r} column")
    return out


def _validate_options(block: dict[str, Any], adjustment: dict[str, str]) -> dict[str, Any]:
    """The option-only fields (O1). Every required one is named in its error and never defaulted."""
    if block["position_units"] != "quantity":
        raise _err("position_units: options hold a whole number of contracts, use quantity")
    for key in ("cost_model", "cost_series_column", "min_cost_per_turnover"):
        if key in block:
            raise _err(f"{key}: not used for options, declare cost_per_contract")
    if len(adjustment) != 1 or set(adjustment.values()) != {"none"}:
        raise _err("adjustment: options O1 cover one contract, declare exactly one price column with adjustment none")
    out: dict[str, Any] = {}
    for key, rule in (("multiplier", "a number > 0"), ("starting_capital", "a number > 0, declared (no default)"),
                      ("strike", "a number > 0")):
        v = block.get(key)
        if not _is_number(v) or v <= 0:
            raise _err(f"{key}: required for options, {rule}")
        out[key] = float(v)
    # Absent is inferred from the data by the options check, or recorded as an unknown answer (a warning).
    unit = block.get("currency_rounding_unit")
    if unit is not None and (not _is_number(unit) or unit <= 0):
        raise _err("currency_rounding_unit: must be a number > 0, the smallest currency step of an option price")
    out["currency_rounding_unit"] = None if unit is None else float(unit)
    out.update(_validate_broker(block))
    if "cost_per_contract" in block:  # absent is handled by the options check as a BLOCKED cost sub-check
        c = block["cost_per_contract"]
        if not _is_number(c) or c < 0:
            raise _err("cost_per_contract: must be a number >= 0 in account currency (never defaulted)")
        out["cost_per_contract"] = float(c)
    if block.get("right") not in OPTION_RIGHTS:
        raise _err(f"right: required for options, one of {list(OPTION_RIGHTS)}")
    out["right"] = block["right"]
    raw = block.get("expiry")
    try:
        if not isinstance(raw, (str, date)) or isinstance(raw, bool):
            raise ValueError
        out["expiry"] = date.fromisoformat(raw).isoformat() if isinstance(raw, str) else \
            (raw.date() if isinstance(raw, datetime) else raw).isoformat()
    except ValueError:
        raise _err("expiry: required for options, an ISO date such as 2025-12-19") from None
    if block.get("option_price_type") is not None and block["option_price_type"] not in OPTION_PRICE_TYPES:
        raise _err(f"option_price_type: must be one of {list(OPTION_PRICE_TYPES)}; omit it to infer it from the data")
    out["option_price_type"] = block.get("option_price_type")  # None: inferred by the options check
    if "trade_time_column" in block and not _is_name(block["trade_time_column"]):
        raise _err("trade_time_column: must be a column name")
    if out["option_price_type"] == "last" and "trade_time_column" not in block:
        raise _err("trade_time_column: required when option_price_type is last (a last price needs a trade timestamp)")
    out["trade_time_column"] = block.get("trade_time_column")
    if "underlying_column" in block and not _is_name(block["underlying_column"]):
        raise _err("underlying_column: must be a column name")
    out["underlying_column"] = block.get("underlying_column")
    for key in ("contract_currency", "account_currency"):
        if not _is_name(block.get(key)):
            raise _err(f"{key}: required for options, a currency code (no FX conversion in O1)")
    if block["contract_currency"] != block["account_currency"]:
        raise _err(f"currency: contract currency {block['contract_currency']!r} differs from account currency "
                   f"{block['account_currency']!r}; O1 does no FX conversion")
    out["contract_currency"], out["account_currency"] = block["contract_currency"], block["account_currency"]
    answer = block.get("point_in_time", "unknown")
    if answer is True or answer == "yes":
        out["point_in_time"] = "yes"
    elif answer in ("unknown", "no", False):
        out["point_in_time"] = "no" if answer in ("no", False) else "unknown"  # not confirmed: a warning, never a block
    else:
        raise _err("point_in_time: declare yes, no, or unknown")
    return out


def validate_reconstruction(block: Any) -> ReconstructionConfig:
    """Returns the typed config, or raises ``ReconstructionConfigError`` (verdict BLOCKED) with the reason."""
    if not isinstance(block, dict):
        raise _err("block: must be a mapping")
    unknown = sorted(set(block) - _KEYS)
    if unknown:
        raise _err(f"block: unknown key(s) {unknown}")

    if block.get("position_units") not in POSITION_UNITS:
        raise _err(f"position_units: required, one of {list(POSITION_UNITS)} (there is no default)")

    if "asset_class" not in block:
        raise _err(f"asset_class: required, one of {list(ASSET_CLASSES)}")
    futures = block["asset_class"] == FUTURES
    # An option block is recognised by its option-only keys; any other option declaration stays unsupported.
    options = block["asset_class"] == OPTION and any(k in block for k in OPTION_ONLY_KEYS)
    if block["asset_class"] not in ASSET_CLASSES and not futures and not options:
        raise _err(f"asset_class: {block['asset_class']!r} is not supported; use the method for this type "
                   f"(supported: {list(ASSET_CLASSES)})")
    if not futures:
        for key in FUTURES_KEYS:
            if key in block and not (options and key in OPTION_SHARED_KEYS):
                raise _err(f"{key}: only valid with asset_class {FUTURES!r}" if not options else
                           f"{key}: not used for options")
    if not options:
        for key in OPTION_ONLY_KEYS:
            if key in block:
                raise _err(f"{key}: only valid with asset_class {OPTION!r}")

    has_model, has_series = "cost_model" in block, "cost_series_column" in block
    if not futures and not options:
        if has_model == has_series:
            raise _err("cost: give exactly one of cost_model or cost_series_column")
        if has_model and (not isinstance(block["cost_model"], dict) or not block["cost_model"]):
            raise _err("cost_model: must be a non-empty mapping")
        if has_series and not _is_name(block["cost_series_column"]):
            raise _err("cost_series_column: must be a column name")

    cost = 0.0 if futures or options else block.get("min_cost_per_turnover")
    if not futures and not options and (not _is_number(cost) or cost < 0):
        raise _err("min_cost_per_turnover: required for the asset class, a number >= 0 (never defaulted to zero)")

    # Absent -> noise-scaled limit; explicit None -> no cap (pending, verdict INCONCLUSIVE).
    bp = block.get("max_unexplained_bp_per_year", DEFAULT_MAX_UNEXPLAINED_BP_PER_YEAR)
    if bp is not None and bp != NOISE_SCALED and (not _is_number(bp) or bp < 0):
        raise _err("max_unexplained_bp_per_year: must be a number >= 0 or None")
    share = block.get("max_unexplained_share", DEFAULT_MAX_UNEXPLAINED_SHARE)
    if share is not None and share != NOISE_SCALED and (not _is_number(share) or not 0 < share <= 1):
        raise _err("max_unexplained_share: must be a number with 0 < x <= 1 or None")

    if not _is_name(block.get("publication_column")):
        raise _err("publication_column: required, a column name")
    stale = block.get("max_staleness")
    if not _is_number(stale) or stale <= 0:
        raise _err("max_staleness: required, a positive number of bars")

    income = _validate_income(block["income"]) if "income" in block else []
    halt = block.get("halt_flag_column")
    if "halt_flag_column" in block and not _is_name(halt):
        raise _err("halt_flag_column: must be a column name")
    adjustment = _validate_adjustment(block.get("adjustment"))
    fut = _validate_futures(block, adjustment) if futures else (
        _validate_options(block, adjustment) if options else {})
    if not futures and any(a in ("back_adjusted", "forward_adjusted") for a in adjustment.values()):
        raise _err("adjustment: back_adjusted and forward_adjusted are for asset_class futures only")

    if "total_return" in adjustment.values() and any(e["kind"] == "dividend" for e in income):
        raise _err("income: a dividend entry together with adjustment total_return counts the dividend twice")

    return ReconstructionConfig(
        position_units=block["position_units"],
        asset_class=block["asset_class"],
        min_cost_per_turnover=cost,
        publication_column=block["publication_column"],
        max_staleness=stale,
        adjustment=adjustment,
        cost_model=block.get("cost_model"),
        cost_series_column=block.get("cost_series_column"),
        max_unexplained_bp_per_year=bp,
        max_unexplained_share=share,
        income=income,
        halt_flag_column=halt,
        pending=[k for k, v in (("max_unexplained_bp_per_year", bp), ("max_unexplained_share", share))
                 if v is None],
        **fut,
    )


def resolve_broker_cost(config: ReconstructionConfig, records: Any, as_of: date | None = None) -> tuple[float | None, dict]:
    """The cost per contract from a confirmed broker record for the declared broker and product, with the details to
    report. ``(None, {"status": "BLOCKED", "reason": ...})`` when no record is usable; never a default or another
    broker's figure. The records come from the caller; nothing is read here."""
    from qrh.costs.broker_commission import resolve_cost

    broker, product = config.broker, config.product
    res = resolve_cost(list(records or []), broker, product, config.declared_equity, as_of or date.today())

    def blocked(reason: str) -> tuple[None, dict]:
        return None, {"status": "BLOCKED", "reason": f"cost sub-check: {reason}", "source": "broker_record",
                      "broker": broker, "product": product}

    if not res.usable:
        return blocked(f"no usable commission record for broker {broker!r}, product {product!r}: "
                       + "; ".join(res.block_reasons))
    if res.currency != config.account_currency:
        return blocked(f"commission record currency {res.currency!r} differs from account currency "
                       f"{config.account_currency!r} for broker {broker!r}, product {product!r}")
    details = {"source": "broker_record", "broker": broker, "product": product,
               "commission_per_contract": res.commission_per_contract, "extras_total": res.extras_total,
               "extras_margin": res.extras_margin, "margin_flagged": res.margin_flagged,
               "currency": res.currency, "warnings": list(res.warnings)}
    return res.all_in_per_contract, details


def check_frame_columns(config: ReconstructionConfig, available: set[str] | frozenset[str]) -> None:
    """Rules that need the dataset: raises ``ReconstructionConfigError`` when a declared column is missing, a
    split-only price column has no dividend column, or fx_spot swap/financing income has no rate column."""
    available = set(available)
    missing_prices = sorted(c for c in config.adjustment if c not in available)
    if missing_prices:
        raise _err(f"adjustment: price column(s) {missing_prices} not found in the dataset")
    if config.asset_class == FUTURES and (config.roll_rule is not None or
                                          "none" in config.adjustment.values()):
        if CONTRACT_COLUMN not in available:
            raise _err(f"roll_rule: a {CONTRACT_COLUMN!r} column is required in the dataset (a roll method alone "
                       "does not say where the rolls are)")
    declared = [("publication_column", config.publication_column)]
    for name in ("trade_time_column", "underlying_column"):
        if getattr(config, name) is not None:
            declared.append((name, getattr(config, name)))
    if config.halt_flag_column is not None:
        declared.append(("halt_flag_column", config.halt_flag_column))
    if config.cost_series_column is not None:
        declared.append(("cost_series_column", config.cost_series_column))
    for name, col in declared:
        if col not in available:
            raise _err(f"{name}: column {col!r} not found in the dataset")
    dividend_cols = [e["column"] for e in config.income if e["kind"] == "dividend"]
    split_only = sorted(c for c, a in config.adjustment.items() if a == "split")
    if split_only and not any(c in available for c in dividend_cols):
        raise _err(f"adjustment: price column(s) {split_only} are split-only but no dividend column is "
                   "in the income list and the dataset")
    for e in config.income:
        if e["column"] in available:
            continue
        if e["kind"] in RATE_KINDS and config.asset_class == "fx_spot":
            raise _err(f"income: fx_spot {e['kind']} needs a rate column, {e['column']!r} not found")
        raise _err(f"income: column {e['column']!r} not found in the dataset")

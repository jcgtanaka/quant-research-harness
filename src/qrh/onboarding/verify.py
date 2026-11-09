"""Deterministic checks of instrument-model claims against the data.

Every check returns one of verified, contradicted or unverified with an evidence string. Only what the data can
prove is ``verified``; what it cannot prove (timestamp semantics, the contract multiplier, adjustments, quote
currency, tick size, the roll rule) stays ``unverified`` until a human confirms it. A contradiction is a fact about
the data (for example prices that are not positive, bid above ask, a split in a series claimed unadjusted).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from qrh.onboarding.model import InstrumentModel
from qrh.onboarding.profile import HINT_VOCAB, LN_SPLIT, parse_frequency

VIOLATION_SHARE = 0.005  # a relationship broken in more than this share of rows contradicts the claim
OFFSET_MISMATCH_SHARE = 0.01
VOLUME_SPIKE = 3.0
ROLL_RATIO = 3.0
SPOT_LIKE = ("equity", "etf", "fx_spot", "crypto_spot", "index")
POSITIVE_CLASSES = ("equity", "etf", "fx_spot", "crypto_spot", "future", "option", "crypto_perp", "index")
CONTRACT_SHAPE = re.compile(r"^A{2,5}9{1,4}$")
OHLC = ("open", "high", "low", "close")
CHECK_VERSION = 1


@dataclass(frozen=True)
class Check:
    status: str
    evidence: str
    checked_by: str


def _ev(text: str) -> str:
    return text if len(text) <= 300 else text[:299] + "…"


def _v(check: str, text: str) -> Check:
    return Check("verified", _ev(text), f"code:{check}@{CHECK_VERSION}")


def _c(check: str, text: str) -> Check:
    return Check("contradicted", _ev(text), f"code:{check}@{CHECK_VERSION}")


def _u(check: str, text: str) -> Check:
    return Check("unverified", _ev(text), f"code:{check}@{CHECK_VERSION}")


def _num(frame: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in frame.columns or not pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
        return None
    return frame[col].astype(float).to_numpy()


def _claim(model: InstrumentModel, field: str) -> Any:
    c = getattr(model, field)
    return c.value if c is not None else None


def _price_columns(model: InstrumentModel) -> list[str]:
    return list(_claim(model, "price_columns") or [])


def _needs_positive(model: InstrumentModel) -> bool:
    units = _claim(model, "price_units")
    if units in ("currency", "index_points"):
        return True
    if units in ("fraction", "percent", "rate_percent"):
        return False
    return _claim(model, "asset_class") in POSITIVE_CLASSES


def _pair_violations(frame: pd.DataFrame, hi: str, lo: str) -> tuple[int, int]:
    a, b = _num(frame, hi), _num(frame, lo)
    if a is None or b is None:
        return 0, 0
    m = np.isfinite(a) & np.isfinite(b)
    tol = 1e-9 * np.maximum(np.abs(a[m]), np.abs(b[m]))
    return int((a[m] < b[m] - tol).sum()), int(m.sum())


def _relationship_violations(frame: pd.DataFrame, names: dict[str, str]) -> list[str]:
    """Pairs of (higher, lower) columns checked; ``names`` maps role to column (open/high/low/close, bid/ask)."""
    pairs = []
    if "high" in names and "low" in names:
        pairs.append(("high", "low"))
    for other in ("open", "close"):
        if "high" in names and other in names:
            pairs.append(("high", other))
        if "low" in names and other in names:
            pairs.append((other, "low"))
    if "ask" in names and "bid" in names:
        pairs.append(("ask", "bid"))
    out = []
    for hi, lo in pairs:
        bad, total = _pair_violations(frame, names[hi], names[lo])
        out.append((names[hi], names[lo], bad, total))
    return out  # type: ignore[return-value]


def _roles(columns: list[str]) -> dict[str, str]:
    return {c.lower(): c for c in columns if c.lower() in OHLC + ("bid", "ask")}


# --- field checks ---------------------------------------------------------------------------------------------

def check_price_columns(model: InstrumentModel, frame: pd.DataFrame) -> Check:
    cols = _price_columns(model)
    name = "price_columns"
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        return _c(name, f"column(s) not in the dataset: {', '.join(missing)}")
    notnum = [c for c in cols if _num(frame, c) is None]
    if notnum:
        return _c(name, f"column(s) not numeric: {', '.join(notnum)}")
    problems: list[str] = []
    for c in cols:
        x = _num(frame, c)
        assert x is not None
        if np.isinf(x).any():
            problems.append(f"{c} has infinite values")
        elif np.isnan(x).all():
            problems.append(f"{c} is all null")
        elif _needs_positive(model) and (x[np.isfinite(x)] <= 0).any():
            problems.append(f"{c} has {int((x[np.isfinite(x)] <= 0).sum())} value(s) that are not positive")
    roles = _roles(cols)
    if "bid" not in roles and "ask" not in roles:
        mapping = _claim(model, "option_mapping") or {}
        roles.update({k: mapping[k] for k in ("bid", "ask") if k in mapping})
    checked = []
    for hi, lo, bad, total in _relationship_violations(frame, roles):
        checked.append(f"{hi}>={lo}")
        if total and bad / total > VIOLATION_SHARE:
            problems.append(f"{hi} < {lo} in {bad} of {total} rows")
    if problems:
        return _c(name, "; ".join(problems))
    note = f"; relationships held: {', '.join(checked)}" if checked else ""
    return _v(name, f"{', '.join(cols)}: numeric, finite" + (", positive" if _needs_positive(model) else "") + note)


PERCENT_NAME_HINTS = ("rate", "pct", "percent", "yield", "return", "ret")


def _percent_name(col: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", col.lower()) if t]
    return any(t.startswith(h) for t in tokens for h in PERCENT_NAME_HINTS) or any(h in col.lower() for h in ("pct", "percent", "yield"))


def _return_like(x: np.ndarray) -> bool:
    """Both signs, a small median against the maximum, roughly symmetric around zero."""
    if len(x) < 10 or not ((x < 0).any() and (x > 0).any()):
        return False
    std = float(np.std(x))
    return std > 0 and float(np.median(np.abs(x))) / float(np.max(np.abs(x))) < 0.25 and abs(float(np.mean(x))) <= 0.25 * std


def check_price_units(model: InstrumentModel, frame: pd.DataFrame) -> Check:
    name = "price_units"
    units = _claim(model, "price_units")
    cols = [c for c in _price_columns(model) if _num(frame, c) is not None]
    if units == "other":
        return _u(name, "units 'other' have no check")
    if not cols:
        return _u(name, "no numeric price column is claimed, so the magnitudes cannot be compared")
    failed: list[str] = []
    seen: list[str] = []
    unproven: list[str] = []
    for c in cols:
        x = _num(frame, c)
        assert x is not None
        x = x[np.isfinite(x)]
        if len(x) == 0:
            failed.append(f"{c}: no finite values")
            continue
        med, mx = float(np.median(np.abs(x))), float(np.max(np.abs(x)))
        p99 = float(np.quantile(np.abs(x), 0.99))
        if units == "fraction":
            ok = med < 0.5 and mx < 5
        elif units == "percent":
            ok = mx <= 100 and p99 >= 0.5
            if ok and not (_percent_name(c) or _return_like(x)):
                unproven.append(f"{c}: in range (max |x| {mx:.4g}) but neither named like a rate, percent, yield or "
                                "return, nor shaped like returns (both signs, small median against the maximum): "
                                "a price level also fits")
                continue
        elif units == "rate_percent":
            ok = 0.05 <= med <= 30 and mx <= 100
        else:  # currency, index_points: positive and the same order of magnitude throughout
            pos = bool((x > 0).all())
            m = float(np.median(x))
            ok = pos and m > 0 and float(x.max()) / m < 1e3 and float(x.min()) / m > 1e-3
        (seen if ok else failed).append(f"{c}: median |x| {med:.4g}, max |x| {mx:.4g}")
    if failed:
        return _c(name, f"magnitudes do not fit '{units}': " + "; ".join(failed))
    if unproven:
        return _u(name, "; ".join(unproven))
    return _v(name, f"magnitudes fit '{units}': " + "; ".join(seen))


def _contract_column(frame: pd.DataFrame) -> str | None:
    from qrh.onboarding.profile import shape_mask

    for c in frame.columns:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        uniq = s.dropna().astype(str).unique()
        if 2 <= len(uniq) <= 400 and all(CONTRACT_SHAPE.match(shape_mask(u)) for u in uniq):
            return str(c)
    return None


def _primary_price(model: InstrumentModel, frame: pd.DataFrame) -> np.ndarray | None:
    cols = _price_columns(model)
    ordered = sorted(cols, key=lambda c: (c.lower() != "close", cols.index(c)))
    for c in ordered:
        x = _num(frame, c)
        if x is not None and np.isfinite(x).all() and (x > 0).all():
            return x
    return None


def _volume_column(frame: pd.DataFrame) -> str | None:
    for c in frame.columns:
        tokens = {t for t in re.split(r"[^a-z0-9]+", str(c).lower()) if t}
        if tokens & HINT_VOCAB["volume"] and _num(frame, str(c)) is not None:
            return str(c)
    return None


def check_adjustments(model: InstrumentModel, frame: pd.DataFrame, profile: dict[str, Any]) -> Check:
    name = "adjustments"
    claim = _claim(model, "adjustments")
    price = _primary_price(model, frame)
    if claim == "unknown":
        return _u(name, "the claim is 'unknown'")
    if profile["time_index"].get("duplicates"):
        return _u(name, "several rows share a timestamp (a panel such as an option chain): consecutive rows are not one "
                        "series, so jumps and level shifts cannot be read")
    if price is None or len(price) < 3:
        return _u(name, "no positive price column to look for jumps and level shifts in")
    lr = np.diff(np.log(price))
    if claim in ("none", "split_only", "split_and_dividend"):
        jumps = np.flatnonzero(np.abs(lr) > LN_SPLIT) + 1
        vol_col = _volume_column(frame)
        if vol_col is not None and len(jumps):
            vol = _num(frame, vol_col)
            assert vol is not None
            med = float(np.nanmedian(vol)) if np.isfinite(vol).any() else 0.0
            jumps = np.array([j for j in jumps if not (med > 0 and np.isfinite(vol[j]) and vol[j] >= VOLUME_SPIKE * med)], dtype=int)
        if claim == "none" and len(jumps):
            return _c(name, f"claimed 'none', but {len(jumps)} bar(s) show a jump above ln(1.9) in a single step with "
                            "no volume spike: the series looks split-affected and unadjusted")
        return _u(name, "no split-like jump found; the absence of a split does not prove how the prices were adjusted"
                  if not len(jumps) else
                  f"{len(jumps)} bar(s) with a jump above ln(1.9) and no volume spike; adjustment cannot be proved from data")
    contract = _contract_column(frame)
    if contract is None:
        return _u(name, f"'{claim}' cannot be proved: no contract column to compare the price change at rolls with")
    codes = frame[contract].astype(str).to_numpy()
    roll = np.flatnonzero(codes[1:] != codes[:-1])  # index into lr
    if len(roll) == 0:
        return _u(name, f"'{claim}' cannot be proved: the contract never changes")
    mask = np.zeros(len(lr), dtype=bool)
    mask[roll] = True
    roll_move, other_move = float(np.median(np.abs(lr[mask]))), float(np.median(np.abs(lr[~mask])) or 1e-12)
    if roll_move >= ROLL_RATIO * other_move:
        return _c(name, f"claimed '{claim}', but the price moves {roll_move / other_move:.1f} times more than usual at the "
                        f"{len(roll)} contract change(s): the series shows level shifts at expiry-like dates")
    return _u(name, f"'{claim}' cannot be proved from the data (no level shift at {len(roll)} contract change(s), "
                    "which is what an adjusted series shows, but so does a calm roll)")


def check_frequency(model: InstrumentModel, frame: pd.DataFrame, profile: dict[str, Any]) -> Check:
    name = "frequency"
    claimed = _claim(model, "frequency")
    ti = profile["time_index"]
    parsed = parse_frequency(str(claimed))
    if parsed is None:
        return _u(name, f"cannot interpret the claimed frequency {claimed!r}")
    if not ti.get("present") or ti.get("inferred_frequency") is None:
        return _u(name, "the dataset has no usable time index to infer a frequency from")
    sec, business = parsed
    inferred = ti["inferred_frequency"]
    if abs(sec - ti["modal_step_seconds"]) < 1e-6 and (not business or ti.get("business_days_only")):
        return _v(name, f"claimed {claimed}, inferred {inferred} (observed density {ti.get('observed_density')})")
    return _c(name, f"claimed {claimed}, but the timestamps step by {inferred} (observed density {ti.get('observed_density')})")


def _claimed_offsets(zone: str, instants: pd.DatetimeIndex) -> np.ndarray | None:
    z = zone.strip()
    if z.upper() in ("UTC", "GMT", "Z"):
        return np.zeros(len(instants))
    m = re.fullmatch(r"(?:UTC|GMT)?([+-])(\d{1,2}):?(\d{2})?", z, re.IGNORECASE)
    if m:
        minutes = int(m.group(2)) * 60 + int(m.group(3) or 0)
        return np.full(len(instants), -minutes if m.group(1) == "-" else minutes, dtype=float)
    try:
        tz = ZoneInfo(z)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None
    local = instants.tz_convert(tz)
    return np.array([t.utcoffset().total_seconds() / 60 for t in local], dtype=float)


def check_timezone(model: InstrumentModel, frame: pd.DataFrame, offsets: np.ndarray | None, tz_from_file: bool) -> Check:
    name = "timezone"
    zone = str(_claim(model, "timezone"))
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        return _u(name, "the timestamps are naive: their zone cannot be checked")
    if offsets is None and not tz_from_file:
        return _u(name, "the file's timestamps carry no UTC offset, so the claimed zone cannot be checked")
    sample = np.linspace(0, len(idx) - 1, min(len(idx), 2000)).astype(int)
    claimed = _claimed_offsets(zone, idx[sample].tz_convert("UTC"))
    if claimed is None:
        return _u(name, f"unknown time zone {zone!r}")
    if offsets is not None and len(offsets) == len(idx):
        actual = offsets[sample]
    else:
        actual = np.array([t.utcoffset().total_seconds() / 60 for t in idx[sample]], dtype=float)
    ok = np.isfinite(actual)
    if not ok.any():
        return _u(name, "no UTC offsets could be read from the timestamps")
    mismatch = float(np.mean(claimed[ok] != actual[ok]))
    if mismatch > OFFSET_MISMATCH_SHARE:
        return _c(name, f"claimed {zone}, but the UTC offsets of the timestamps differ from it in {mismatch:.0%} of the sampled rows")
    return _v(name, f"UTC offsets of the timestamps match {zone} (offset consistency only; it says nothing about the session)")


def check_option_mapping(model: InstrumentModel, frame: pd.DataFrame) -> Check:
    name = "option_mapping"
    mapping = _claim(model, "option_mapping") or {}
    missing = [f"{k}={v}" for k, v in mapping.items() if v not in frame.columns]
    if missing:
        return _c(name, "mapped column(s) not in the dataset: " + ", ".join(missing))
    problems: list[str] = []
    for k in ("underlying", "strike", "bid", "ask", "mid", "rate"):
        if k in mapping:
            x = _num(frame, mapping[k])
            if x is None:
                problems.append(f"{k} column {mapping[k]} is not numeric")
            elif k in ("underlying", "strike") and (x[np.isfinite(x)] <= 0).any():
                problems.append(f"{k} column {mapping[k]} has values that are not positive")
    if "expiry" in mapping:
        col = frame[mapping["expiry"]]
        if not pd.api.types.is_datetime64_any_dtype(col) and pd.to_datetime(col, errors="coerce", utc=True).isna().any():
            problems.append(f"expiry column {mapping['expiry']} has values that are not dates")
    if "right" in mapping and frame[mapping["right"]].nunique() > 3:
        problems.append(f"right column {mapping['right']} has more than 3 distinct values")
    if "bid" in mapping and "ask" in mapping:
        bad, total = _pair_violations(frame, mapping["ask"], mapping["bid"])
        if total and bad / total > VIOLATION_SHARE:
            problems.append(f"ask below bid in {bad} of {total} rows")
    if problems:
        return _c(name, "; ".join(problems))
    needed = [k for k in ("underlying", "strike", "expiry", "right") if k not in mapping]
    if "mid" not in mapping and not ("bid" in mapping and "ask" in mapping):
        needed.append("mid or bid/ask")
    if needed:
        return _u(name, "the mapping is incomplete: missing " + ", ".join(needed))
    return _v(name, "mapped columns exist with plausible types (strike and underlying positive, expiry dates, bid <= ask)")


def _option_signature(model: InstrumentModel, profile: dict[str, Any], mapping_check: Check | None) -> str:
    """Evidence that the data is option data: strike, expiry and right columns, or an option_mapping that holds."""
    found = {h["hint"]: h["column"] for h in profile["hints"] if h["hint"] in ("strike", "expiry", "right")}
    if len(found) == 3:
        return "strike column " + found["strike"] + ", expiry column " + found["expiry"] + ", right column " + found["right"]
    mapping = _claim(model, "option_mapping") or {}
    if all(k in mapping for k in ("strike", "expiry", "right")) and mapping_check is not None and mapping_check.status == "verified":
        return f"an option_mapping holds (strike={mapping['strike']}, expiry={mapping['expiry']}, right={mapping['right']})"
    return ""


def check_asset_class(model: InstrumentModel, frame: pd.DataFrame, profile: dict[str, Any],
                      mapping_check: Check | None) -> Check:
    name = "asset_class"
    cls = _claim(model, "asset_class")
    cols = _price_columns(model)
    if cls == "option":
        mapping = _claim(model, "option_mapping")
        if not mapping:
            return _u(name, "an option dataset needs an option_mapping (strike, expiry, right) and none was proposed")
        absent = [k for k in ("strike", "expiry", "right") if k not in mapping]
        if absent:
            return _u(name, "the option_mapping lacks " + ", ".join(absent))
        if mapping_check is not None and mapping_check.status == "contradicted":
            return _c(name, "claimed option, but the option_mapping does not hold: " + mapping_check.evidence)
        return _v(name, "strike, expiry and right columns are present and plausible")
    if cls != "option":
        sig = _option_signature(model, profile, mapping_check)
        if sig:
            return _c(name, f"claimed {cls}, but the data carries the option signature ({sig}): it is option data")
    if cls == "future":
        contract = _contract_column(frame)
        if contract:
            return _v(name, f"a contract-code column ({contract}) is present")
        return _u(name, "no contract-code column or roll pattern found (an expiry-like column name alone is not evidence)")
    if cls == "crypto_perp":
        funding = [h["column"] for h in profile["hints"] if h["hint"] == "funding" and _num(frame, h["column"]) is not None]
        if funding:
            return _v(name, f"a funding-like numeric column is present ({funding[0]})")
        return _u(name, "no funding-like column found")
    if cls in SPOT_LIKE:
        if not cols:
            return _u(name, "no price column is claimed")
        for c in cols:
            x = _num(frame, c)
            if x is not None and (x[np.isfinite(x)] <= 0).any():
                return _c(name, f"claimed {cls}, but {c} has values that are not positive")
        fut = _contract_column(frame)
        if fut is None:
            fut = next((h["column"] for h in profile["hints"] if h["hint"] == "contract"), None)
        if fut is not None:
            return _c(name, f"claimed {cls}, but column {fut} is a contract column: the data looks like futures")
        fund = next((h["column"] for h in profile["hints"] if h["hint"] == "funding"), None)
        if fund is not None:
            return _c(name, f"claimed {cls}, but column {fund} is a funding column: the data looks like a perpetual swap")
        expiry = next((h["column"] for h in profile["hints"] if h["hint"] == "expiry"), None)
        if expiry is not None:
            return _u(name, f"claimed {cls}, but column {expiry} looks like an expiry: spot-like data has none")
        if cls in ("equity", "etf"):
            vol = _volume_column(frame)
            if vol is not None and (np.nan_to_num(_num(frame, vol)) >= 0).all():
                return _v(name, f"positive prices in {', '.join(cols)} with a volume column ({vol}) and no option, "
                                f"futures or funding signature; equity and etf cannot be told apart by the data")
            return _u(name, f"prices are positive, but no volume column supports a claimed {cls}")
        if cls == "fx_spot":
            roles = _roles(cols)
            if "bid" in roles and "ask" in roles:
                return _v(name, f"a bid/ask quote pair ({roles['bid']}, {roles['ask']}) with positive prices and no option, "
                                "futures or funding signature")
            return _u(name, "prices are positive, but no bid/ask quote pair supports a claimed fx_spot")
        return _u(name, f"prices are positive, but the data has no evidence specific to {cls}")
    return _u(name, f"no structural check for asset class '{cls}'")


STATIC_UNVERIFIED = {
    "timestamp_semantics": "whether a timestamp marks the bar open, the bar close or an event time cannot be seen in the data",
    "multiplier": "a contract multiplier cannot be derived from prices; it is read from the contract specification",
    "quote_currency": "the quote currency is not stored in the data",
    "session_hint": "trading sessions are not verified",
    "tick_size": "the tick size is not verified against prices",
    "roll_rule": "the roll rule is not verified",
}


def verify_model(model: InstrumentModel, frame: pd.DataFrame, profile: dict[str, Any], *,
                 offsets: np.ndarray | None = None, tz_from_file: bool = True) -> dict[str, Check]:
    """One Check per claimed field. ``offsets`` are the UTC offsets (minutes) written in the file's time column, if any;
    ``tz_from_file`` says that the frame's own time zone came from the file (a Parquet index) rather than a default."""
    out: dict[str, Check] = {}
    claims = model.claims()
    if "price_columns" in claims:
        out["price_columns"] = check_price_columns(model, frame)
    if "price_units" in claims:
        out["price_units"] = check_price_units(model, frame)
    if "adjustments" in claims:
        out["adjustments"] = check_adjustments(model, frame, profile)
    if "frequency" in claims:
        out["frequency"] = check_frequency(model, frame, profile)
    if "timezone" in claims:
        out["timezone"] = check_timezone(model, frame, offsets, tz_from_file)
    if "option_mapping" in claims:
        out["option_mapping"] = check_option_mapping(model, frame)
    if "asset_class" in claims:
        out["asset_class"] = check_asset_class(model, frame, profile, out.get("option_mapping"))
    for field, why in STATIC_UNVERIFIED.items():
        if field in claims:
            out[field] = _u(field, why)
    return out

"""Deterministic, privacy-safe profile of a dataset: the only view of a registered dataset meant for the agent.

Nothing here ever returns a row-level value. Per column it reports a type, shares and flags; numeric columns also
report aggregate values (minimum, maximum, quantiles) unless ``no_values`` is set. Text columns report their number
of distinct values and a SHAPE distribution (digits become 9, upper-case letters A, lower-case letters a, other
characters are kept), and the actual values only for a column of at most six distinct values that all belong to one known flag vocabulary
(``C``/``P``, ``Y``/``N``, ``BUY``/``SELL``, ...) and whose name is not that of an identifier. Relationships
between columns are found from the numbers alone; names only produce hints, never facts.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrh.hashing import hash_json

PROFILE_VERSION = 1
DISTINCT_CAP = 1000
SHAPE_TOP = 5
SHAPE_MAX_LEN = 32
SHAPE_VALUES_CAP = 20000
FLAG_MAX_DISTINCT = 6
# the only value sets a text column may show: its whole distinct set (case-insensitive) must sit inside one of these
FLAG_VOCABULARIES: tuple[frozenset[str], ...] = tuple(frozenset(v) for v in (
    {"C", "P"}, {"CALL", "PUT"}, {"Y", "N"}, {"YES", "NO"}, {"B", "S"}, {"BUY", "SELL"}, {"T", "F"},
    {"TRUE", "FALSE"}, {"L", "S"}, {"LONG", "SHORT"}, {"BID", "ASK"}, {"0", "1"}))
# a column that is named like an identifier never shows values, whatever they look like
IDENTIFIER_NAMES = re.compile(
    r"symbol|ticker|instrument|underlying|contract|isin|cusip|figi|sedol|ric|root|name|code|(^|_)id($|_)|issuer|series|pair",
    re.IGNORECASE)
QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)
MAX_REL_COLUMNS = 16
MAX_REL_ENTRIES = 40
CLOSE_REL_TOL = 1e-3
LN_SPLIT = math.log(1.9)  # a single-bar move bigger than this looks like a split (or a bad print)

HINT_VOCAB: dict[str, frozenset[str]] = {
    "strike": frozenset({"strike", "strikes"}),
    "expiry": frozenset({"expiry", "expiration", "expire", "exp", "maturity"}),
    "right": frozenset({"right", "cp", "callput", "putcall", "optiontype", "call", "put"}),
    "delta": frozenset({"delta"}),
    "iv": frozenset({"iv", "impliedvol", "ivol", "impliedvolatility"}),
    "bid": frozenset({"bid"}),
    "ask": frozenset({"ask", "offer"}),
    "rate": frozenset({"rate", "rf", "riskfree", "yield", "sofr", "libor"}),
    "volume": frozenset({"volume", "vlm"}),
    "open interest": frozenset({"oi", "openinterest"}),
    "funding": frozenset({"funding"}),
    "contract": frozenset({"contract"}),
    "multiplier": frozenset({"multiplier", "mult", "contractsize"}),
    "settle": frozenset({"settle", "settlement", "settled"}),
    "adjustments": frozenset({"adj", "adjusted", "adjclose", "split", "splits", "dividend", "dividends"}),
}


# --- loading --------------------------------------------------------------------------------------------------

def load_frame(path: str | Path) -> tuple[pd.DataFrame, str | None]:
    """Read a registered file and index it by its first datetime column; returns (frame, time column or None)."""
    from qrh.memory.catalog import _read

    df = _read(Path(path))
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return df.set_index(col), str(col)
    return df, None


_OFFSET_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})\s*$")


def raw_offsets_minutes(path: str | Path) -> np.ndarray | None:
    """UTC offset in minutes written in the first column of a CSV (None when the text carries none, or not a CSV).
    Only aggregates of this ever leave the verifier."""
    p = Path(path)
    if p.suffix.lower() != ".csv":
        return None
    try:
        first = pd.read_csv(p, usecols=[0], dtype=str).iloc[:, 0].dropna().astype(str)
    except Exception:
        return None
    found = first.map(lambda t: (_OFFSET_RE.search(t.strip()) or [None, None])[1])
    if found.isna().all():
        return None

    def minutes(tag: str | None) -> float:
        if tag is None:
            return float("nan")
        if tag == "Z":
            return 0.0
        sign = -1 if tag[0] == "-" else 1
        digits = tag[1:].replace(":", "")
        return sign * (int(digits[:2]) * 60 + int(digits[2:]))

    return found.map(minutes).to_numpy(dtype=float)


# --- time index -----------------------------------------------------------------------------------------------

def step_label(seconds: float) -> str:
    s = int(round(seconds))
    if s >= 86400 and s % 86400 == 0:
        return f"{s // 86400}D"
    if s >= 3600 and s % 3600 == 0:
        return f"{s // 3600}h"
    if s >= 60 and s % 60 == 0:
        return f"{s // 60}min"
    return f"{s}s"


def parse_frequency(text: str) -> tuple[float, bool] | None:
    """(step in seconds, business-days flag) for a claimed frequency such as 1D, daily, 1h, 5min, 1B; None if unknown."""
    t = str(text).strip().lower().replace(" ", "")
    words = {"daily": "1d", "day": "1d", "hourly": "1h", "weekly": "1w", "minutely": "1min", "bdaily": "1b"}
    t = words.get(t, t)
    m = re.fullmatch(r"(\d*)(d|day|days|b|bd|h|hr|hour|hours|min|m|minute|minutes|s|sec|second|seconds|w|week|weeks)", t)
    if not m:
        return None
    n = int(m.group(1) or 1)
    unit = m.group(2)
    if unit in ("b", "bd"):
        return float(n * 86400), True
    per = {"d": 86400, "day": 86400, "days": 86400, "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600, "min": 60,
           "m": 60, "minute": 60, "minutes": 60, "s": 1, "sec": 1, "second": 1, "seconds": 1, "w": 604800,
           "week": 604800, "weeks": 604800}[unit]
    return float(n * per), False


def time_index_profile(idx: pd.Index, no_values: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"present": isinstance(idx, pd.DatetimeIndex)}
    if not isinstance(idx, pd.DatetimeIndex):
        return out
    out["timezone"] = str(idx.tz) if idx.tz is not None else None
    out["monotonic_increasing"] = bool(idx.is_monotonic_increasing)
    out["duplicates"] = int(idx.duplicated().sum())
    uniq = idx.dropna().unique().sort_values()
    out["unique_timestamps"] = int(len(uniq))
    out["inferred_frequency"] = None
    out["gap_count"] = 0
    if len(uniq) >= 2:
        steps = pd.Series(uniq[1:] - uniq[:-1])
        mode = steps.mode().iloc[0]
        sec = float(mode.total_seconds())
        label = step_label(sec)
        business = False
        if sec == 86400 and not (uniq.day_of_week >= 5).any() and (steps == pd.Timedelta(days=3)).any():
            label, business = "1B", True
        out["inferred_frequency"] = label
        out["modal_step_seconds"] = sec
        out["business_days_only"] = business
        weekend = (steps == pd.Timedelta(days=3)) & (uniq[:-1].day_of_week == 4) if business else pd.Series(False, index=steps.index)
        longer = (steps > mode) & ~weekend
        out["gap_count"] = int(longer.sum())
        span = (uniq[-1] - uniq[0]).total_seconds()
        expected = span / sec + 1 if sec else float(len(uniq))
        if business:
            expected = expected * 5 / 7
        out["observed_density"] = round(min(1.0, len(uniq) / expected), 4) if expected else 1.0
    if not no_values:
        out["first"] = uniq[0].isoformat() if len(uniq) else None
        out["last"] = uniq[-1].isoformat() if len(uniq) else None
    return out


# --- columns --------------------------------------------------------------------------------------------------

def shape_mask(value: str) -> str:
    out = []
    for ch in value:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        else:
            out.append(ch)
    text = "".join(out)
    return text if len(text) <= SHAPE_MAX_LEN else text[:SHAPE_MAX_LEN] + "…"


def _numeric_profile(s: pd.Series, no_values: bool) -> dict[str, Any]:
    x = s.astype(float)
    nonnull = x.dropna()
    finite = nonnull[np.isfinite(nonnull)]
    out: dict[str, Any] = {
        "finite_share": round(float(len(finite) / len(nonnull)), 6) if len(nonnull) else None,
        "integer_valued": bool(len(finite) > 0 and np.all(np.mod(finite.to_numpy(), 1) == 0)),
        "non_negative": bool(len(finite) > 0 and (finite >= 0).all()),
        "strictly_positive": bool(len(finite) > 0 and (finite > 0).all()),
    }
    if out["strictly_positive"] and len(finite) > 1:
        lr = np.abs(np.diff(np.log(x.to_numpy()[np.isfinite(x.to_numpy()) & (x.to_numpy() > 0)])))
        out["split_like_jumps"] = int((lr > LN_SPLIT).sum())
    if not no_values and len(finite):
        q = finite.quantile(list(QUANTILES))
        out["min"] = float(finite.min())
        out["max"] = float(finite.max())
        out["quantiles"] = {f"p{int(round(k * 100)):02d}": float(v) for k, v in zip(QUANTILES, q)}
    return out


def _string_profile(s: pd.Series, name: str, no_values: bool) -> dict[str, Any]:
    counts = s.dropna().astype(str).value_counts()
    out: dict[str, Any] = {}
    shown = counts.head(SHAPE_VALUES_CAP)
    shapes: dict[str, int] = {}
    for value, n in shown.items():
        key = shape_mask(str(value))
        shapes[key] = shapes.get(key, 0) + int(n)
    total = int(counts.sum()) or 1
    top = sorted(shapes.items(), key=lambda kv: (-kv[1], kv[0]))[:SHAPE_TOP]
    out["shapes"] = [{"shape": k, "share": round(v / total, 4)} for k, v in top]
    out["shapes_truncated"] = bool(len(counts) > SHAPE_VALUES_CAP)
    seen = {str(v).upper() for v in counts.index}
    flaglike = (0 < len(counts) <= FLAG_MAX_DISTINCT and any(seen <= vocab for vocab in FLAG_VOCABULARIES)
                and not IDENTIFIER_NAMES.search(name))
    out["flag_like"] = bool(flaglike)
    if flaglike and not no_values:
        out["values"] = sorted(str(v) for v in counts.index)
    return out


def column_profile(name: str, s: pd.Series, no_values: bool) -> dict[str, Any]:
    n = len(s)
    nunique = int(s.nunique(dropna=True))
    base: dict[str, Any] = {
        "dtype": str(s.dtype),
        "null_rate": round(float(s.isna().mean()), 6) if n else 0.0,
        "distinct": min(nunique, DISTINCT_CAP),
        "distinct_capped": nunique > DISTINCT_CAP,
        "monotonic_increasing": bool(s.is_monotonic_increasing) if n else False,
        "monotonic_decreasing": bool(s.is_monotonic_decreasing) if n else False,
    }
    if pd.api.types.is_bool_dtype(s):
        return {**base, "kind": "bool"}
    if pd.api.types.is_numeric_dtype(s):
        return {**base, "kind": "numeric", **_numeric_profile(s, no_values)}
    if pd.api.types.is_datetime64_any_dtype(s):
        out = {**base, "kind": "datetime"}
        if not no_values and s.notna().any():
            out["min"], out["max"] = s.min().isoformat(), s.max().isoformat()
        return out
    return {**base, "kind": "string", **_string_profile(s, name, no_values)}


# --- relationships and hints -----------------------------------------------------------------------------------

def _comparable(a: np.ndarray, b: np.ndarray, factor: float = 10.0) -> bool:
    ma, mb = float(np.median(np.abs(a))), float(np.median(np.abs(b)))
    if ma == 0 or mb == 0:
        return ma == mb
    return 1 / factor <= ma / mb <= factor


def _pairs(frame: pd.DataFrame, cols: list[str], no_values: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    data = {c: frame[c].astype(float).to_numpy() for c in cols}
    ge: list[tuple[str, str]] = []
    for a in cols:
        for b in cols:
            if a == b:
                continue
            x, y = data[a], data[b]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 10:
                continue
            xa, yb = x[m], y[m]
            if not _comparable(xa, yb):
                continue  # columns of very different size (a price and a volume) are not related by order
            if np.all(xa >= yb) and np.any(xa > yb):
                ge.append((a, b))
                out.append({"kind": "a_ge_b", "a": a, "b": b, "strict_share": round(float(np.mean(xa > yb)), 4)})
            if a < b:
                denom = np.maximum(np.maximum(np.abs(xa), np.abs(yb)), 1e-12)
                close = np.abs(xa - yb) <= CLOSE_REL_TOL * denom
                if close.mean() >= 0.99:
                    out.append({"kind": "close", "a": a, "b": b, "rel_tol": CLOSE_REL_TOL,
                                "share": round(float(close.mean()), 4), "identical": bool(np.all(xa == yb))})
    for hi, lo in ge:
        for v in cols:
            if v in (hi, lo):
                continue
            h, l, x = data[hi], data[lo], data[v]
            m = np.isfinite(h) & np.isfinite(l) & np.isfinite(x)
            if m.sum() >= 10 and np.all(l[m] <= x[m]) and np.all(x[m] <= h[m]):
                out.append({"kind": "inside", "value": v, "low": lo, "high": hi})
    return out[:MAX_REL_ENTRIES]


def _single(name: str, s: pd.Series, no_values: bool) -> list[dict[str, Any]]:
    x = s.astype(float).to_numpy()
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return []
    out: list[dict[str, Any]] = []
    med_abs = float(np.median(np.abs(x)))
    std = float(np.std(x))
    if (x < 0).any() and (x > 0).any() and med_abs < 0.2 and std > 0 and abs(float(np.mean(x))) < std:
        item: dict[str, Any] = {"kind": "return_like", "column": name}
        if not no_values:
            item["median_abs"], item["scale"] = round(med_abs, 6), round(std, 6)
        out.append(item)
    if x.min() >= 0 and x.max() <= 100 and x.max() > 1:
        out.append({"kind": "percent_like", "column": name})
    steps = np.abs(np.diff(x))
    spread = float(x.max() - x.min())
    if x.min() >= -1 and x.max() <= 25 and spread >= 0 and float(np.mean(steps == 0)) >= 0.5:  # changes rarely
        out.append({"kind": "rate_like", "column": name, "scale": "fraction" if x.max() <= 1 else "percent_points"})
    return out


def name_hints(columns: list[str]) -> list[dict[str, str]]:
    hints = []
    for c in columns:
        low = str(c).lower()
        tokens = {t for t in re.split(r"[^a-z0-9]+", low) if t} | {re.sub(r"[^a-z0-9]", "", low)}
        for hint, vocab in HINT_VOCAB.items():
            if tokens & vocab or (hint == "open interest" and {"open", "interest"} <= tokens):
                hints.append({"column": str(c), "hint": hint})
    return sorted(hints, key=lambda h: (h["column"], h["hint"]))


# --- the profile ------------------------------------------------------------------------------------------------

def build_profile(frame: pd.DataFrame, *, no_values: bool = False, time_column: str | None = None) -> dict[str, Any]:
    """Profile of an indexed frame. Aggregates only: see the module docstring for what is and is not exposed."""
    if len(frame) > 200_000:  # relationships are found on an evenly spaced sample; everything else uses every row
        sample = frame.iloc[np.linspace(0, len(frame) - 1, 200_000).astype(int)]
    else:
        sample = frame
    columns = {str(c): column_profile(str(c), frame[c], no_values) for c in frame.columns}
    numeric = [str(c) for c in frame.columns if columns[str(c)]["kind"] == "numeric"][:MAX_REL_COLUMNS]
    relationships = _pairs(sample, numeric, no_values)
    for c in numeric:
        relationships += _single(c, sample[c], no_values)
    return {
        "profile_version": PROFILE_VERSION,
        "no_values": bool(no_values),
        "rows": int(len(frame)),
        "time_index": {**time_index_profile(frame.index, no_values), "column": time_column},
        "columns": columns,
        "relationships": relationships,
        "hints": name_hints([str(c) for c in frame.columns]),
        "notes": ["relationships and hints are candidates, never facts; no row-level value is included"],
    }


def profile_hash(profile: dict[str, Any]) -> str:
    return hash_json(profile)


def profile_lines(profile: dict[str, Any]) -> list[str]:
    """A compact text rendering for the terminal."""
    ti = profile["time_index"]
    lines = [f"rows: {profile['rows']}"]
    if ti.get("present"):
        lines.append(
            f"time index: column {ti.get('column')}, tz {ti.get('timezone')}, frequency {ti.get('inferred_frequency')}, "
            f"gaps {ti.get('gap_count')}, duplicates {ti.get('duplicates')}, monotonic {ti.get('monotonic_increasing')}"
            + (f", {ti.get('first')} .. {ti.get('last')}" if "first" in ti else ""))
    else:
        lines.append("time index: none")
    for name, c in profile["columns"].items():
        extra = ""
        if c["kind"] == "numeric" and "min" in c:
            extra = f" min {c['min']:.6g} max {c['max']:.6g} median {c['quantiles']['p50']:.6g}"
        elif c["kind"] == "string":
            extra = " shapes " + ", ".join(f"{s['shape']}:{s['share']:.0%}" for s in c["shapes"])
            if "values" in c:
                extra += " values " + "/".join(c["values"])
        lines.append(f"  {name}: {c['kind']} {c['dtype']} null {c['null_rate']:.1%} distinct {c['distinct']}{extra}")
    for r in profile["relationships"]:
        lines.append("  relationship: " + ", ".join(f"{k}={v}" for k, v in r.items()))
    for h in profile["hints"]:
        lines.append(f"  hint (not a fact): {h['column']} ~ {h['hint']}")
    return lines

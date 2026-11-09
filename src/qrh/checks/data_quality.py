"""Data-quality rule engine: declarative rules per dataset, deterministic verdicts.

Rules live in YAML (``rules: [{type: ..., ...}]``). Each rule returns PASS or
FAIL with reasons that name the first offending timestamps. Anything that
cannot be evaluated (bad rules file, unknown rule type, unreadable data) is
BLOCKED, never PASS.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from qrh.hashing import sha256_file
from qrh.memory.catalog import Catalog, load_indexed_frame
from qrh.onboarding import store

MAX_OFFENDERS = 5  # timestamps named in a reason
MAX_STORED_OFFENDERS = 50  # timestamps kept per rule for reports
MAX_STORED_GAPS = 200
DTYPE_ALIASES = {
    "float": "f", "int": "iu", "number": "fiu", "string": "O", "object": "O",
    "datetime": "M", "bool": "b",
}


class RulesError(ValueError):
    """The rules file is missing, unreadable or malformed."""


@dataclass
class RuleResult:
    rule: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    type: str = ""
    offenders: list[str] = field(default_factory=list)  # up to MAX_STORED_OFFENDERS ISO timestamps


@dataclass
class QualityReport:
    verdict: str  # PASS | FAIL | BLOCKED
    results: list[RuleResult]
    reasons: list[str]


def load_rules(path: str | Path) -> list[dict[str, Any]]:
    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RulesError(f"cannot read rules file {str(path)!r}: {exc}") from exc
    rules = doc.get("rules") if isinstance(doc, dict) else None
    if not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules):
        raise RulesError(f"rules file {str(path)!r} must contain a list under 'rules'")
    return rules


def _first(index: pd.Index, mask: Any, sink: list[str] | None = None) -> str:
    stamps = [pd.Timestamp(t).isoformat() for t in index[np.asarray(mask)][:MAX_STORED_OFFENDERS]]
    if sink is not None:
        sink.extend(stamps)
    return "first offending timestamps: " + ", ".join(stamps[:MAX_OFFENDERS])


def _numeric(df: pd.DataFrame, col: str) -> tuple[pd.Series | None, str | None]:
    if col not in df.columns:
        return None, f"column {col!r} not found"
    if not pd.api.types.is_numeric_dtype(df[col]):
        return None, f"column {col!r} is not numeric ({df[col].dtype})"
    return df[col].astype(float), None


def _schema(df: pd.DataFrame, rule: dict[str, Any], offenders: list[str] | None = None) -> list[str]:
    reasons: list[str] = []
    for col, kinds in (rule.get("columns") or {}).items():
        if col not in df.columns:
            reasons.append(f"required column {col!r} is missing")
            continue
        allowed = "".join(DTYPE_ALIASES.get(k, k) for k in (kinds if isinstance(kinds, list) else [kinds]))
        if df[col].dtype.kind not in allowed:
            reasons.append(f"column {col!r} has dtype kind {df[col].dtype.kind!r} ({df[col].dtype}), expected one of {allowed!r}")
    return reasons


def _range(df: pd.DataFrame, rule: dict[str, Any], offenders: list[str] | None = None) -> list[str]:
    s, err = _numeric(df, rule["column"])
    if s is None:
        return [err or "unreadable"]
    bad = pd.Series(False, index=s.index)
    nonnull = s.notna()
    bad |= nonnull & ~np.isfinite(s.fillna(0.0))
    if "min" in rule:
        bad |= nonnull & (s < rule["min"])
    if "max" in rule:
        bad |= nonnull & (s > rule["max"])
    if rule.get("positive"):
        bad |= nonnull & (s <= 0)
    if not bad.any():
        return []
    return [f"column {rule['column']!r}: {int(bad.sum())} value(s) out of range; {_first(s.index, bad.values, offenders)}"]


def _relationship(df: pd.DataFrame, rule: dict[str, Any], offenders: list[str] | None = None) -> list[str]:
    a, err = _numeric(df, rule["column"])
    if a is None:
        return [err or "unreadable"]
    b, err = _numeric(df, rule["reference"])
    if b is None:
        return [err or "unreadable"]
    tol = float(rule["rel_tol"])
    both = a.notna() & b.notna()
    off = both & ((a - b).abs() > tol * b.abs())
    if not rule.get("only_where_both_non_null", False):
        off |= a.isna() ^ b.isna()
    if not off.any():
        return []
    return [
        f"{rule['column']!r} deviates from {rule['reference']!r} by more than {tol:g} (relative) "
        f"in {int(off.sum())} row(s); {_first(a.index, off.values, offenders)}"
    ]


def _null_rate_drift(df: pd.DataFrame, rule: dict[str, Any], offenders: list[str] | None = None) -> list[str]:
    col = rule["column"]
    if col not in df.columns:
        return [f"column {col!r} not found"]
    n = len(df)
    if "window_rows" in rule:
        w = int(rule["window_rows"])
    elif "window_fraction" in rule:
        w = int(round(float(rule["window_fraction"]) * n))
    else:
        return ["null_rate_drift needs window_fraction or window_rows"]
    if w < 1 or w >= n:
        return [f"window of {w} row(s) leaves no history to compare against ({n} rows)"]
    null = df[col].isna().to_numpy()
    recent, history = float(null[-w:].mean()), float(null[:-w].mean())
    limit = float(rule.get("max_increase", 0.0))
    if recent - history > limit:
        return [
            f"null share of {col!r} rose from {history:.4f} (history) to {recent:.4f} "
            f"(latest {w} rows), above the allowed increase {limit:g}"
        ]
    return []


def _expected_offset(index: pd.DatetimeIndex, freq: str | None) -> Any:
    if freq:
        return pd.tseries.frequencies.to_offset(freq)
    steps = pd.Series(index.unique().sort_values()).diff().dropna()
    if steps.empty:
        raise ValueError("cannot infer frequency from fewer than two timestamps")
    return pd.tseries.frequencies.to_offset(steps.mode().iloc[0])


def _gaps(df: pd.DataFrame, rule: dict[str, Any], offenders: list[str] | None = None) -> list[str]:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return ["index is not a DatetimeIndex"]
    reasons: list[str] = []
    dup = idx.duplicated()
    if dup.any():
        reasons.append(f"{int(dup.sum())} duplicate timestamp(s); {_first(idx, dup, offenders)}")
    if not idx.is_monotonic_increasing:
        reasons.append("timestamps are not sorted in increasing order")
    uniq = idx.unique().sort_values()
    if len(uniq) < 2:
        return reasons
    try:
        expected = pd.date_range(uniq[0], uniq[-1], freq=_expected_offset(uniq, rule.get("freq")))
    except (ValueError, TypeError) as exc:
        return reasons + [f"cannot determine expected frequency: {exc}"]
    missing = expected.difference(uniq)
    if len(missing):
        if offenders is not None:
            offenders.extend(t.isoformat() for t in missing[:MAX_STORED_OFFENDERS])
        reasons.append(
            f"{len(missing)} missing timestamp(s); first missing timestamps: "
            + ", ".join(t.isoformat() for t in missing[:MAX_OFFENDERS])
        )
    return reasons


def _timezone(df: pd.DataFrame, rule: dict[str, Any], offenders: list[str] | None = None) -> list[str]:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        return ["index is not timezone-aware"]
    zone = str(rule["zone"])
    actual = str(idx.tz)
    try:
        if actual == zone or pd.Timestamp(0, tz=zone).tz == idx.tz:
            return []
    except Exception:  # unknown zone name: report as mismatch
        pass
    return [f"index timezone is {actual!r}, declared {zone!r}"]


RULE_TYPES: dict[str, Callable[..., list[str]]] = {
    "schema": _schema,
    "range": _range,
    "relationship": _relationship,
    "null_rate_drift": _null_rate_drift,
    "gaps": _gaps,
    "timezone": _timezone,
}


def run_rules(
    df: pd.DataFrame, rules: list[dict[str, Any]], resolve: Callable[[str], Callable[..., list[str]] | None] | None = None
) -> QualityReport:
    """Run every rule. No rules or an unknown rule type is BLOCKED; any FAIL is FAIL.

    ``resolve`` maps a rule type to its implementation (the method registry, so the version used can be
    recorded); by default the built-in ``RULE_TYPES`` are used.
    """
    if not rules:
        return QualityReport("BLOCKED", [], ["no quality rules declared"])
    results: list[RuleResult] = []
    blocked: list[str] = []
    for i, rule in enumerate(rules):
        rtype = rule.get("type")
        name = str(rule.get("name") or f"{rtype}#{i + 1}")
        fn = (resolve or RULE_TYPES.get)(rtype)  # type: ignore[arg-type]
        if fn is None:
            blocked.append(f"{name}: unknown rule type {rtype!r}")
            continue
        found: list[str] = []
        try:
            reasons = fn(df, rule, found)
        except KeyError as exc:
            blocked.append(f"{name}: missing rule field {exc}")
            continue
        except Exception as exc:  # a rule that cannot run must not pass
            blocked.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        results.append(RuleResult(
            name, not reasons, [f"{name}: {r}" for r in reasons], str(rtype), found[:MAX_STORED_OFFENDERS]
        ))
    failed = [r for res in results if not res.passed for r in res.reasons]
    if blocked:
        return QualityReport("BLOCKED", results, blocked + failed)
    return QualityReport("FAIL" if failed else "PASS", results, failed)


def coverage_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Span, row count and detected gaps of a dataset (gaps are steps longer than the modal step)."""
    idx = df.index
    out: dict[str, Any] = {
        "start": None, "end": None, "rows": int(len(df)), "step_seconds": None, "gap_count": 0, "gaps": [],
    }
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return out
    uniq = idx.unique().sort_values()
    out["start"], out["end"] = uniq[0].isoformat(), uniq[-1].isoformat()
    if len(uniq) < 2:
        return out
    steps = uniq[1:] - uniq[:-1]
    step = pd.Series(steps).mode().iloc[0]
    out["step_seconds"] = float(step.total_seconds())
    where = np.flatnonzero(np.asarray(steps > step))
    out["gap_count"] = int(len(where))
    out["gaps"] = [
        {
            "after": uniq[i].isoformat(), "before": uniq[i + 1].isoformat(),
            "missing": int(round(steps[i] / step)) - 1,
        }
        for i in where[:MAX_STORED_GAPS]
    ]
    return out


def check_dataset(catalog: Catalog, name: str) -> tuple[str, list[str]]:
    """Run a dataset's rules and record the outcome in the catalog.

    Returns (status, reasons). Status is PASS, BLOCKED (any rule failed or the
    check could not run) or UNCHECKED (no rules declared).
    """
    status, reasons, _report, _coverage = check_dataset_report(catalog, name)
    return status, reasons


def check_dataset_report(
    catalog: Catalog, name: str, resolve: Callable[[str], Callable[..., list[str]] | None] | None = None
) -> tuple[str, list[str], QualityReport | None, dict[str, Any] | None]:
    """Like ``check_dataset`` and also returns the per-rule report and the coverage summary
    (both None when the rules could not run)."""
    rec = catalog.get_dataset(name)
    if rec is None:
        return "BLOCKED", [f"unknown dataset {name!r}"], None, None

    def record(status: str, reasons: list[str], report: QualityReport | None = None, cov: Any = None) -> Any:
        catalog.set_status(name, status, reasons)
        return status, reasons, report, cov

    try:
        actual = sha256_file(rec["path"])
    except OSError as exc:
        return record("BLOCKED", [f"dataset file cannot be read: {exc}"])
    if actual != rec["content_hash"]:
        return record("BLOCKED", ["dataset file changed since registration (hash mismatch); register it again"])
    contradicted = store.contradictions(catalog.conn, name)  # claims about the data that the data contradicts
    rules_path = catalog.rules_for(name)
    if rules_path is None:
        if contradicted:
            return record("BLOCKED", contradicted)
        return record("UNCHECKED", ["no quality rules declared"])
    try:
        rules = load_rules(rules_path)
        frame = load_indexed_frame(rec["path"])
        report = run_rules(frame, rules, resolve)
        cov = coverage_summary(frame)
    except Exception as exc:
        return record("BLOCKED", [f"{type(exc).__name__}: {exc}"])
    if report.verdict == "PASS" and not contradicted:
        return record("PASS", [], report, cov)
    return record("BLOCKED", report.reasons + contradicted, report, cov)

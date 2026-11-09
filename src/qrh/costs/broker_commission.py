"""Validator and store for broker commission records.

The LLM session retrieves the broker's own commission page; this module only
validates and stores the record it is given. It performs no network access.

Rules:
- Nothing is a cost unless the user has confirmed it (status "confirmed").
- A missing, unconfirmed, stale, unknown-date, conflicting or tier-less record
  BLOCKS the cost. It never defaults to zero and never reuses another broker's
  or product's figure.
- Stale: older than 12 months from the page's last-updated date. Warn at 6.
- Extras not checked: a flagged margin is added and recorded.
- Conflicting figures for one broker and product block; the higher figure is a
  placeholder only and is never applied.
"""

from __future__ import annotations

import calendar
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

STALE_MONTHS = 12
WARN_MONTHS = 6
DEFAULT_MARGIN_FRACTION = 0.25
NOT_CHECKED = "not checked"
FLAT = "flat"
AUDIT_FILENAME = "broker_commission_audit.json"

_STATUSES = ("proposed", "confirmed")
_CONFIDENCE = ("high", "medium", "low")


class RecordError(ValueError):
    """The record is malformed and cannot be stored."""


@dataclass(frozen=True)
class Tier:
    min_equity: float
    max_equity: float | None  # None means open-ended
    commission_per_contract: float


@dataclass(frozen=True)
class Extra:
    name: str
    amount: float


@dataclass(frozen=True)
class CommissionRecord:
    broker: str
    product: str
    commission_per_contract: float
    per_order_minimum: float | None
    currency: str
    last_updated_date: date | None  # None = "unknown", which blocks
    source_url: str
    retrieved_on: date
    tier_table: str | tuple[Tier, ...]  # "flat" or tiers
    extras: str | tuple[Extra, ...]  # "not checked" or listed extras
    status: str
    confirmed_on: date | None
    confidence: str
    source_snapshot_sha256: str | None = None
    confirmed_value: float | None = None


@dataclass(frozen=True)
class CostResult:
    usable: bool
    broker: str
    product: str
    commission_per_contract: float | None
    per_order_minimum: float | None
    currency: str | None
    extras_total: float | None
    extras_margin: float
    margin_flagged: bool
    all_in_per_contract: float | None
    warnings: tuple[str, ...]
    block_reasons: tuple[str, ...]
    ask_user: bool
    conflicts: tuple[dict[str, Any], ...]
    placeholder_higher: float | None  # display only, never applied
    record: CommissionRecord | None


# ---------------------------------------------------------------- parsing


def _num(v: Any, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RecordError(f"{name} must be a number")
    if v < 0 or (v == 0 and not allow_zero):
        raise RecordError(f"{name} must be non-negative")
    return float(v)


def _text(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise RecordError(f"{name} must be a non-empty string")
    return v.strip()


def _date(v: Any, name: str) -> date:
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RecordError(f"{name} must be an ISO date (YYYY-MM-DD)") from None


def _tiers(v: Any) -> str | tuple[Tier, ...]:
    if v == FLAT:
        return FLAT
    if not isinstance(v, Sequence) or isinstance(v, str) or not v:
        raise RecordError('tier_table must be "flat" or a non-empty list')
    tiers = []
    for t in v:
        if not isinstance(t, Mapping):
            raise RecordError("tier entries must be objects")
        lo = _num(t.get("min_equity"), "min_equity")
        hi_raw = t.get("max_equity")
        hi = None if hi_raw is None else _num(hi_raw, "max_equity")
        if hi is not None and hi <= lo:
            raise RecordError("max_equity must exceed min_equity")
        tiers.append(Tier(lo, hi, _num(t.get("commission_per_contract"), "tier commission")))
    return tuple(tiers)


def _extras(v: Any) -> str | tuple[Extra, ...]:
    if v == NOT_CHECKED:
        return NOT_CHECKED
    if not isinstance(v, Sequence) or isinstance(v, str):
        raise RecordError('extras must be a list or "not checked"')
    out = []
    for e in v:
        if not isinstance(e, Mapping):
            raise RecordError("extras entries must be objects")
        out.append(Extra(_text(e.get("name"), "extra name"), _num(e.get("amount"), "extra amount")))
    return tuple(out)


def parse_record(data: Mapping[str, Any]) -> CommissionRecord:
    """Validate a raw mapping into a frozen record, or raise RecordError."""
    status = data.get("status")
    if status not in _STATUSES:
        raise RecordError(f"status must be one of {_STATUSES}")
    confidence = data.get("confidence")
    if confidence not in _CONFIDENCE:
        raise RecordError(f"confidence must be one of {_CONFIDENCE}")
    lu = data.get("last_updated_date")
    last_updated = None if lu == "unknown" else _date(lu, "last_updated_date")
    confirmed_on = data.get("confirmed_on")
    if status == "confirmed":
        if confirmed_on is None:
            raise RecordError("a confirmed record needs confirmed_on")
        confirmed_on = _date(confirmed_on, "confirmed_on")
    else:
        confirmed_on = None if confirmed_on is None else _date(confirmed_on, "confirmed_on")
    pom = data.get("per_order_minimum")
    cv = data.get("confirmed_value")
    sha = data.get("source_snapshot_sha256")
    if sha is not None and (not isinstance(sha, str) or len(sha) != 64):
        raise RecordError("source_snapshot_sha256 must be a 64-character hex digest")
    return CommissionRecord(
        broker=_text(data.get("broker"), "broker"),
        product=_text(data.get("product"), "product"),
        commission_per_contract=_num(data.get("commission_per_contract"), "commission_per_contract"),
        per_order_minimum=None if pom is None else _num(pom, "per_order_minimum"),
        currency=_text(data.get("currency"), "currency"),
        last_updated_date=last_updated,
        source_url=_text(data.get("source_url"), "source_url"),
        retrieved_on=_date(data.get("retrieved_on"), "retrieved_on"),
        tier_table=_tiers(data.get("tier_table")),
        extras=_extras(data.get("extras")),
        status=status,
        confirmed_on=confirmed_on,
        confidence=confidence,
        source_snapshot_sha256=sha,
        confirmed_value=None if cv is None else _num(cv, "confirmed_value"),
    )


def snapshot_sha256(content: bytes) -> str:
    """Hash of a source snapshot supplied by the session (never fetched here)."""
    return hashlib.sha256(content).hexdigest()


# ------------------------------------------------------------- evaluation


def add_months(d: date, months: int) -> date:
    idx = d.year * 12 + (d.month - 1) + months
    year, month = divmod(idx, 12)
    month += 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _schedule(r: CommissionRecord) -> tuple:
    tiers = r.tier_table
    t = tiers if isinstance(tiers, str) else tuple((x.min_equity, x.max_equity, x.commission_per_contract) for x in tiers)
    return (r.commission_per_contract, r.per_order_minimum, r.currency, t)


def _summary(r: CommissionRecord) -> dict[str, Any]:
    return {
        "source_url": r.source_url,
        "last_updated_date": r.last_updated_date.isoformat() if r.last_updated_date else "unknown",
        "commission_per_contract": r.commission_per_contract,
        "per_order_minimum": r.per_order_minimum,
        "currency": r.currency,
        "status": r.status,
    }


def _infer_tier(r: CommissionRecord, equity: float | None) -> tuple[float | None, str | None]:
    if r.tier_table == FLAT:
        return r.commission_per_contract, None
    if isinstance(equity, bool) or not isinstance(equity, (int, float)) or equity < 0:
        return None, "declared initial equity is missing or invalid, so the tier cannot be inferred"
    hits = [
        t
        for t in r.tier_table  # type: ignore[union-attr]
        if t.min_equity <= equity and (t.max_equity is None or equity < t.max_equity)
    ]
    if not hits:
        return None, f"no tier matches declared equity {equity}; tier cannot be inferred"
    if len(hits) > 1:
        return None, f"tier table is ambiguous for declared equity {equity}"
    return hits[0].commission_per_contract, None


def _blocked(broker, product, reasons, *, ask=False, conflicts=(), placeholder=None, record=None, margin=0.0):
    return CostResult(
        usable=False,
        broker=broker,
        product=product,
        commission_per_contract=None,
        per_order_minimum=None,
        currency=None,
        extras_total=None,
        extras_margin=margin,
        margin_flagged=False,
        all_in_per_contract=None,
        warnings=(),
        block_reasons=tuple(reasons),
        ask_user=ask,
        conflicts=tuple(conflicts),
        placeholder_higher=placeholder,
        record=record,
    )


def resolve_cost(
    records: Sequence[CommissionRecord],
    broker: str,
    product: str,
    declared_equity: float | None,
    as_of: date,
    *,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
) -> CostResult:
    """Return the usable cost for broker/product, or a blocked result with reasons."""
    matches = [r for r in records if r.broker == broker and r.product == product]
    if not matches:
        return _blocked(broker, product, [f"no commission record for {broker} / {product}"], ask=True)

    schedules = {_schedule(r) for r in matches}
    if len(schedules) > 1:
        higher = max(r.commission_per_contract for r in matches)
        return _blocked(
            broker,
            product,
            ["conflicting figures for the same broker and product; resolve with the user"],
            ask=True,
            conflicts=[_summary(r) for r in matches],
            placeholder=higher,
        )

    rec = sorted(
        matches,
        key=lambda r: (r.status == "confirmed", r.last_updated_date or date.min),
        reverse=True,
    )[0]

    reasons: list[str] = []
    ask = False
    warnings: list[str] = []
    if rec.status != "confirmed":
        reasons.append("record is not confirmed by the user")
        ask = True
    if rec.last_updated_date is None:
        reasons.append("last-updated date of the source page is unknown")
    else:
        if as_of > add_months(rec.last_updated_date, STALE_MONTHS):
            reasons.append("record is stale (older than 12 months); re-confirm before use")
            ask = True
        elif as_of >= add_months(rec.last_updated_date, WARN_MONTHS):
            warnings.append("record is 6 months or older from its last-updated date; plan to re-confirm")

    applied, tier_err = _infer_tier(rec, declared_equity)
    if tier_err:
        reasons.append(tier_err)
        ask = True
    elif rec.confirmed_value is not None and rec.confirmed_value != applied:
        reasons.append("schedule differs from the value the user confirmed; re-confirm")
        ask = True

    if reasons:
        return _blocked(broker, product, reasons, ask=ask, record=rec)

    assert applied is not None
    if rec.extras == NOT_CHECKED:
        extras_total = 0.0
        margin = round(margin_fraction * applied, 10)
        flagged = True
        warnings.append("extras not checked; a flagged margin was added to the cost")
    else:
        extras_total = sum(e.amount for e in rec.extras)  # type: ignore[union-attr]
        margin = 0.0
        flagged = False
    return CostResult(
        usable=True,
        broker=broker,
        product=product,
        commission_per_contract=applied,
        per_order_minimum=rec.per_order_minimum,
        currency=rec.currency,
        extras_total=extras_total,
        extras_margin=margin,
        margin_flagged=flagged,
        all_in_per_contract=applied + extras_total + margin,
        warnings=tuple(warnings),
        block_reasons=(),
        ask_user=False,
        conflicts=(),
        placeholder_higher=None,
        record=rec,
    )


# ------------------------------------------------------------------ audit


def _jsonable(o: Any) -> Any:
    if isinstance(o, date):
        return o.isoformat()
    raise TypeError(type(o))


def audit_entry(result: CostResult) -> dict[str, Any]:
    """Audit payload: the confirmed record, snapshot hash and applied figures."""
    if not result.usable or result.record is None:
        raise RecordError("only a usable, confirmed cost can be stored")
    return {
        "record": json.loads(json.dumps(asdict(result.record), default=_jsonable)),
        "source_snapshot_sha256": result.record.source_snapshot_sha256,
        "commission_per_contract": result.commission_per_contract,
        "extras_total": result.extras_total,
        "extras_margin": result.extras_margin,
        "margin_flagged": result.margin_flagged,
        "all_in_per_contract": result.all_in_per_contract,
        "warnings": list(result.warnings),
    }


def store_run_audit(run_dir: Path, result: CostResult) -> Path:
    """Write the audit entry into the run directory and return its path."""
    entry = audit_entry(result)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / AUDIT_FILENAME
    path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

"""Parity report: backtest daily results versus live (broker) daily results.

The report is ALWAYS written. Missing or unreadable live data gives the status
MISSING_LIVE_DATA with a plain warning; nothing here ever stops or blocks
anything.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

MATCH = "MATCH"
DIFFERENT = "DIFFERENT"
MISSING = "MISSING_LIVE_DATA"
_COLUMNS = ("time", "instrument", "quantity", "price", "fees")


class LiveDataError(Exception):
    """The live fills file is missing or unreadable (message is plain English)."""


def load_live_fills(path) -> list[dict]:
    """Read the fills CSV: time, instrument, quantity, price, fees.

    Raises LiveDataError with a plain message; the caller turns it into
    the status MISSING_LIVE_DATA.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LiveDataError(f"The live fills file was not found: {p}.") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise LiveDataError(f"The live fills file could not be read ({p}): {exc}.") from None
    rows = list(csv.DictReader(text.splitlines()))
    if not text.strip():
        raise LiveDataError(f"The live fills file is empty: {p}.")
    reader = csv.reader(text.splitlines())
    names = [n.strip() for n in next(reader, [])]
    missing = [c for c in _COLUMNS if c not in names]
    if missing:
        raise LiveDataError(
            f"The live fills file is missing these columns: {', '.join(missing)}. "
            f"Expected columns: {', '.join(_COLUMNS)}."
        )
    fills = []
    for n, row in enumerate(rows, start=2):
        try:
            when = datetime.fromisoformat(row["time"].strip())
            fills.append(
                {
                    "time": row["time"].strip(),
                    "date": when.date().isoformat(),
                    "instrument": row["instrument"].strip(),
                    "quantity": _num(row["quantity"]),
                    "price": float(row["price"]),
                    "fees": float(row["fees"]),
                }
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            raise LiveDataError(
                f"Row {n} of the live fills file could not be understood. "
                "Check the time (ISO format), quantity, price and fees."
            ) from None
    if not fills:
        raise LiveDataError(f"The live fills file has no fills: {p}.")
    return fills


def _num(text):
    v = float(text)
    return int(v) if v == int(v) else v


def _aggregate_live(fills) -> dict:
    """(date, instrument) -> net quantity and volume-weighted price."""
    acc: dict = {}
    for f in fills:
        date = f.get("date") or str(f["time"])[:10]
        key = (date, f["instrument"])
        a = acc.setdefault(key, [0.0, 0.0, 0.0])  # net qty, abs qty, price*abs qty
        q = float(f["quantity"])
        a[0] += q
        a[1] += abs(q)
        a[2] += float(f["price"]) * abs(q)
    return {k: {"quantity": a[0], "price": (a[2] / a[1] if a[1] else 0.0)} for k, a in acc.items()}


def _index_backtest(rows) -> dict:
    out = {}
    for r in rows or []:
        out[(str(r["date"])[:10], r["instrument"])] = {
            "quantity": float(r["quantity"]),
            "price": float(r["price"]),
        }
    return out


def _same(bt, other, tol) -> bool:
    if bt is None or other is None:
        return False
    if bt["quantity"] != other["quantity"]:
        return False
    return abs(bt["price"] - other["price"]) <= tol * abs(bt["price"])


def build_parity_report(
    backtest_daily,
    live_fills,
    days,
    tolerance,
    trial_id,
    strategy_name="",
    live_error=None,
    own_backtest_daily=None,
) -> dict:
    """Build the parity report dict. Never raises for missing or odd data.

    backtest_daily: rows with date, instrument, quantity, price.
    live_fills: list from load_live_fills, or None/empty when unavailable.
    days: number of latest live days to compare, or None for all.
    tolerance: allowed price difference as a fraction (live comparison only).
    own_backtest_daily: the harness's own numbers; any difference is a bug.
    """
    warnings: list[str] = []
    report = {
        "trial_id": trial_id,
        "strategy_name": strategy_name,
        "days_compared": "all",
        "start_date": None,
        "end_date": None,
        "tolerance": {"quantity": "exact", "price_fraction": tolerance},
        "status": MISSING,
        "differing_days": [],
        "warnings": warnings,
        "explanation": "",
    }
    try:
        bt = _index_backtest(backtest_daily)
        _check_own(bt, own_backtest_daily, warnings)
        if not live_fills:
            reason = live_error or "No live results were provided."
            warnings.append(
                f"Live results are missing: {reason} Without them the backtest cannot be "
                "checked against real trading, so its numbers are unconfirmed. "
                "Next step: export your broker fills as a CSV (time, instrument, quantity, "
                "price, fees) and run the report again. Nothing is blocked meanwhile."
            )
            report["explanation"] = (
                "No live data was available, so no comparison was made. "
                "This does not stop anything."
            )
            return report
        live = _aggregate_live(live_fills)
        live_days = sorted({d for d, _ in live})
        n = days if isinstance(days, int) and not isinstance(days, bool) and days > 0 else None
        if days is not None and n is None:
            warnings.append(
                f"The number of days ({days!r}) was not a positive whole number, "
                "so all live days were used. Next step: give a number such as 30."
            )
        window = live_days[-n:] if n else live_days
        report["days_compared"] = n if n and n <= len(live_days) else (n if n else "all")
        report["start_date"], report["end_date"] = window[0], window[-1]
        diffs = []
        for key in sorted(set(live) | {k for k in bt if k[0] in window}):
            if key[0] not in window:
                continue
            b, lv = bt.get(key), live.get(key)
            if not _same(b, lv, tolerance):
                diffs.append(
                    {"date": key[0], "instrument": key[1], "backtest": b, "live": lv}
                )
        report["differing_days"] = diffs
        if diffs:
            report["status"] = DIFFERENT
            report["explanation"] = (
                f"The backtest and live results differ on {len(diffs)} day(s) "
                "(quantity must match exactly; price may differ by the stated tolerance)."
            )
            warnings.append(
                "The backtest does not match real trading on the days listed. "
                "Next step: look at those days first (late fills, costs, missed orders) "
                "before trusting the backtest."
            )
        else:
            report["status"] = MATCH
            report["explanation"] = (
                f"The backtest matches live results on all {len(window)} compared day(s) "
                "within the stated tolerance."
            )
    except Exception as exc:  # the report must never stop anything
        report["status"] = MISSING
        warnings.append(
            f"The report could not compare the data ({exc}). Next step: check that the "
            "backtest rows have date, instrument, quantity and price, then run it again."
        )
        report["explanation"] = "The comparison could not be completed. This does not stop anything."
    return report


def _check_own(bt, own, warnings) -> None:
    if not own:
        return
    own_idx = _index_backtest(own)
    for key in sorted(set(bt) | set(own_idx)):
        a, b = bt.get(key), own_idx.get(key)
        if a != b:
            warnings.append(
                f"BUG: on {key[0]} ({key[1]}) the backtest differs from the harness's own "
                f"numbers: {a} versus {b}. These must be identical. "
                "Next step: report this as a bug; do not trust this backtest until fixed."
            )


def _text(report) -> str:
    lines = [
        "PARITY REPORT",
        f"Trial: {report['trial_id']}",
        f"Strategy: {report['strategy_name']}",
        f"Status: {report['status']}",
        f"Days compared: {report['days_compared']}",
        f"From {report['start_date']} to {report['end_date']}",
        f"Tolerance: quantity {report['tolerance']['quantity']}, "
        f"price {report['tolerance']['price_fraction']} (fraction)",
        "",
        report["explanation"],
    ]
    for d in report["differing_days"]:
        lines.append(f"- {d['date']} {d['instrument']}: backtest {d['backtest']} | live {d['live']}")
    for w in report["warnings"]:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, data: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def write_parity_report(report, home) -> list[Path]:
    """Write <home>/parity/<trial>.json and .txt atomically; return both paths."""
    out = Path(home) / "parity"
    out.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(report["trial_id"]))
    j, t = out / f"{stem}.json", out / f"{stem}.txt"
    _write_atomic(j, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_atomic(t, _text(report))
    return [j, t]

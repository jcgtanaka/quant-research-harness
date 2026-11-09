"""Read-only bench listing for the dashboard. Every number is read back from a hash-verified bench receipt.

Receipts live in ``<QRH_HOME>/bench/<bench_id>/receipt.json`` (``qrh.bench.receipt``). Nothing here writes.
``verdict`` is the one stored in the receipt by a new run (PASS, FAIL or NO_RULE). A receipt without it falls back
to the stored results: the shared string ``verdict`` when every suite carries the same one, else ``NOT_RECORDED``
(never computed here).
A metric whose stored rate is not a number is left out and named in ``skipped_metrics``; a receipt that
cannot be verified is left out and named, with the reason, in the top-level ``skipped`` list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qrh.bench.receipt import BenchReceipt, BenchReceiptError, bench_root, list_receipts, load_bench_receipt
from qrh.ui.data import DataError

NOT_RECORDED = "NOT_RECORDED"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _verdict(receipt: BenchReceipt) -> str:
    if receipt.verdict is not None:  # recorded by a new run, covered by the receipt hash
        return receipt.verdict
    found = {r.get("verdict") for r in receipt.results.values() if isinstance(r, dict)}
    if len(found) == 1 and isinstance(next(iter(found)), str):
        return found.pop()
    return NOT_RECORDED


def _entry(bench_id: str, receipt: BenchReceipt) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    skipped: list[str] = []
    for suite in sorted(receipt.results):
        stored = receipt.results[suite].get("metrics") if isinstance(receipt.results[suite], dict) else None
        for name in sorted(stored if isinstance(stored, dict) else {}):
            m = stored[name]
            rate = m.get("rate") if isinstance(m, dict) else None
            key = f"{suite}/{name}"
            if _number(rate):
                metrics[key] = rate
            else:
                skipped.append(key)
    out: dict[str, Any] = {
        "bench_id": bench_id, "suite": ", ".join(receipt.suites), "created_at": receipt.created_at,
        "verdict": _verdict(receipt), "metrics": metrics,
    }
    if skipped:
        out["skipped_metrics"] = skipped
    return out


def list_benches(home: str | Path) -> dict[str, Any]:
    root = bench_root(home)
    if root.exists() and not root.is_dir():
        raise DataError(503, f"bench folder {str(root)!r} is not a folder")
    try:
        found = list_receipts(home)
    except OSError as exc:
        raise DataError(503, f"bench folder unreadable: {type(exc).__name__}: {exc}") from exc
    benches: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for bench_id, path in found:
        try:
            benches.append(_entry(bench_id, load_bench_receipt(path)))
        except BenchReceiptError as exc:
            skipped.append({"bench_id": bench_id, "reason": str(exc)})
    out: dict[str, Any] = {"benches": benches}
    if skipped:
        out["skipped"] = skipped
    return out

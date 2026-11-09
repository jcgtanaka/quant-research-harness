"""Side-by-side comparison of benchmark outcomes, with a Fisher test of whether a difference is real."""

from __future__ import annotations

from typing import Any

from qrh.bench.receipt import BenchReceipt
from qrh.bench.stats import ALPHA, compare_counts


def _better(direction: str, a: float | None, b: float | None) -> str | None:
    if a is None or b is None:
        return None
    if a == b:
        return "tie"
    b_higher = b > a
    return "b" if b_higher == (direction == "higher") else "a"


def compare_metrics(a_results: dict[str, Any], b_results: dict[str, Any], alpha: float = ALPHA) -> list[dict[str, Any]]:
    """One row per metric present in both result sets: counts, delta (b - a), Fisher p-value, which side is better."""
    rows: list[dict[str, Any]] = []
    for suite in sorted(set(a_results) & set(b_results)):
        ma, mb = a_results[suite].get("metrics") or {}, b_results[suite].get("metrics") or {}
        for name in sorted(set(ma) & set(mb)):
            x, y = ma[name], mb[name]
            stat = compare_counts(x["k"], x["n"], y["k"], y["n"], alpha)
            rows.append({
                "suite": suite, "name": name, "direction": x["direction"], "gate": bool(x.get("gate")) and bool(y.get("gate")),
                "a": {k: x[k] for k in ("k", "n", "rate", "ci")}, "b": {k: y[k] for k in ("k", "n", "rate", "ci")},
                "better": _better(x["direction"], x["rate"], y["rate"]), **stat,
            })
    return rows


def _method_differences(a: dict[str, str], b: dict[str, str]) -> dict[str, list[str]]:
    return {i: [a[i], b[i]] for i in sorted(set(a) & set(b)) if a[i] != b[i]}


def compare_receipts(a: BenchReceipt, b: BenchReceipt, alpha: float = ALPHA) -> dict[str, Any]:
    notes: list[str] = []
    if a.seed != b.seed:
        notes.append(f"different seeds ({a.seed} vs {b.seed}): the runs used different data, so differences include sampling noise")
    if a.config != b.config:
        notes.append("different suite configuration (replications or cells): rates are compared as proportions")
    if a.code_hash != b.code_hash:
        notes.append("different harness code hashes")
    return {
        "a": a.bench_id(), "b": b.bench_id(), "alpha": alpha, "notes": notes,
        "method_differences": _method_differences(a.methods, b.methods),
        "rows": compare_metrics(a.results, b.results, alpha),
    }


def within_receipt(r: BenchReceipt, alpha: float = ALPHA) -> dict[str, Any]:
    """The default arm (a) against the candidate arm (b) of one receipt, which ran over identical data."""
    if r.baseline_results is None or r.baseline_methods is None:
        raise ValueError("this receipt has no baseline (it was not run with --compare-to default)")
    return {
        "a": "default", "b": "candidate", "alpha": alpha,
        "notes": ["both arms ran over identical generated data; Fisher's test treats them as independent samples, which is conservative"],
        "method_differences": _method_differences(r.baseline_methods, r.methods),
        "rows": compare_metrics(r.baseline_results, r.results, alpha),
    }

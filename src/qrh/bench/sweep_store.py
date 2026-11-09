"""Store a pnl-linear noise-sensitivity sweep as a bench receipt, so ``GET /api/bench`` can show it.

Takes the dict returned by ``qrh.bench.pnl_linear_corpus.noise_sensitivity_sweep`` (it never runs a sweep) and
writes it with the unchanged bench receipt format and hash rules (``qrh.bench.receipt``). Each metric is stored
as ``{"k", "n", "rate", "label"}`` under ``results[suite]["metrics"]``; no verdict is stored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qrh.bench.receipt import BenchReceipt, write_bench

SUITES = ("pnl-linear-futures", "pnl-linear-slope")
_METHODS = {"s3.pnl_linear": "noise_sensitivity_sweep@1"}


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _bp(x: Any) -> str:
    if not _numeric(x):
        raise ValueError(f"sweep level {x!r} is not numeric")
    return f"{float(x):g}bp"


def _metric(rate: Any, n: int, name: str, label: str) -> dict[str, Any]:
    if not _numeric(rate):
        raise ValueError(f"metric {name!r} must be numeric, got {rate!r}")
    out: dict[str, Any] = {"rate": float(rate), "label": label}
    if isinstance(n, int) and not isinstance(n, bool) and n > 0:
        out.update(n=n, k=round(float(rate) * n))
    return out


def sweep_metrics(sweep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    n = sweep.get("n_cases_per_cell", 0)
    metrics: dict[str, dict[str, Any]] = {}
    for (noise, gain), rate in sorted(sweep.get("fail_share", {}).items()):
        name = f"fail_share/noise_{_bp(noise)}/gain_{_bp(gain)}"
        metrics[name] = _metric(rate, n, name, "FAIL share, planted gain")
    for noise, rate in sorted(sweep.get("honest_false_alarm", {}).items()):
        name = f"false_alarm/noise_{_bp(noise)}"
        metrics[name] = _metric(rate, n, name, "honest false-alarm rate")
    return metrics


def write_sweep_receipt(sweep: dict[str, Any], home: str | Path, *, suite: str, seed: int,
                        code_hash: str | None = None, config: dict[str, Any] | None = None) -> Path | None:
    """Write the sweep as a bench receipt under ``<home>/bench/<id>/``; return the folder, or None if empty.

    Raises ValueError (before anything is written) for an unknown suite or a non-numeric metric.
    """
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITES}, got {suite!r}")
    metrics = sweep_metrics(sweep)
    if not metrics:
        return None
    if code_hash is None:
        from qrh.pipeline.stages import package_code_hash

        code_hash = package_code_hash()
    receipt = BenchReceipt(
        profile="custom", suites=[suite], seed=seed,
        config={"n_cases_per_cell": sweep.get("n_cases_per_cell", 0), **(config or {})},
        methods=dict(_METHODS), results={suite: {"metrics": metrics}}, code_hash=code_hash,
    )
    return write_bench(receipt, home)

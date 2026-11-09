"""``qrh bench``: run benchmark suites, compare receipts, list them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

bench_app = typer.Typer(help="Benchmarks with known truth for the harness's methods.", no_args_is_help=True, add_completion=False)


def _fail(message: str) -> typer.Exit:
    from qrh.cli import _fail as fail

    return fail(message)


def _home() -> Path:
    from qrh.config import QrhHomeError, ensure_home

    try:
        return ensure_home()
    except QrhHomeError as exc:
        raise _fail(str(exc)) from exc


@bench_app.command("run")
def bench_run(
    suite: list[str] = typer.Option(None, "--suite", help="Suite to run (repeatable): planted-bugs, noise-fpr, known-edge-power, history, llm-gate, onboarding, param-stability. Default: the first three."),
    method: list[str] = typer.Option(None, "--method", help="Candidate method version, for example s5.pbo=cscv@2 (repeatable)."),
    compare_to: str | None = typer.Option(None, "--compare-to", help="'default': also run the default methods over the same data."),
    reps: int | None = typer.Option(None, "--reps", min=1, help="Replications per cell, overriding the profile."),
    seed: int | None = typer.Option(None, "--seed", help="Base seed (default: the fixed bench seed)."),
    quick: bool = typer.Option(False, "--quick", help="Small profile for smoke checks (the default)."),
    full: bool = typer.Option(False, "--full", help="Thorough profile for real comparisons and promotion."),
    history: Path | None = typer.Option(None, "--history", help="Harness home to re-evaluate stored trials from (history suite; read-only)."),
    as_json: bool = typer.Option(False, "--json", help="Print the receipt as JSON instead of the text report."),
) -> None:
    """Run suites, write the bench receipt, text, PDF and chart JSON under $QRH_HOME/bench/<id>/."""
    from qrh.bench.profiles import DEFAULT_SEED
    from qrh.bench.receipt import BenchReceiptError, write_bench
    from qrh.bench.render import render_bench
    from qrh.bench.run import parse_overrides, run_bench
    from qrh.bench.textreport import format_receipt
    from qrh.methods.registry import MethodError

    if quick and full:
        raise _fail("choose --quick or --full, not both")
    if compare_to is not None and compare_to != "default":
        raise _fail("--compare-to accepts only 'default'")
    home = _home()
    try:
        receipt = run_bench(
            home, suites=suite or None, profile="full" if full else "quick", reps=reps,
            seed=DEFAULT_SEED if seed is None else seed, overrides=parse_overrides(method or []),
            compare_to_default=compare_to == "default", history_path=history,
            progress=lambda line: typer.echo(line, err=True),
        )
        folder = write_bench(receipt, home)
        render_bench(receipt, folder)
    except (ValueError, MethodError, BenchReceiptError) as exc:
        raise _fail(str(exc)) from exc
    if as_json:
        typer.echo(json.dumps({"bench_id": receipt.bench_id(), "receipt_hash": receipt.receipt_hash(),
                               "receipt": receipt.model_dump(mode="json")}, sort_keys=True, indent=2))
        return
    typer.echo(format_receipt(receipt))
    typer.echo(f"bench receipt: {folder / 'receipt.json'}")
    typer.echo(f"reports: {folder / 'report.txt'}, {folder / 'report.pdf'}, {folder / 'chart.json'}")


@bench_app.command("compare")
def bench_compare(
    receipt_a: str = typer.Argument(..., help="Bench id (or unique prefix, or path to receipt.json)."),
    receipt_b: str | None = typer.Argument(None, help="Second receipt. Omit to compare the first receipt's default arm with its candidate arm."),
    all_metrics: bool = typer.Option(False, "--all", help="Include metrics that are not promotion gates."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Side-by-side table with deltas (b - a) and Fisher p-values: is the difference real, given the replications?"""
    from qrh.bench.compare import compare_receipts, within_receipt
    from qrh.bench.receipt import BenchReceiptError, resolve_receipt
    from qrh.bench.textreport import format_comparison

    home = _home()
    try:
        a = resolve_receipt(home, receipt_a)
        cmp = within_receipt(a) if receipt_b is None else compare_receipts(a, resolve_receipt(home, receipt_b))
    except (BenchReceiptError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    if as_json:
        typer.echo(json.dumps(cmp, sort_keys=True, indent=2))
        return
    typer.echo(format_comparison(cmp, gated_only=not all_metrics))


@bench_app.command("list")
def bench_list() -> None:
    """Stored bench receipts."""
    from qrh.bench.receipt import BenchReceiptError, list_receipts, load_bench_receipt

    for bench_id, path in list_receipts(_home()):
        try:
            r = load_bench_receipt(path)
            flag = f" overrides {','.join(f'{k}={v}' for k, v in sorted(r.overrides.items()))}" if r.overrides else ""
            typer.echo(f"{bench_id}  {r.profile}  seed {r.seed}  {','.join(r.suites)}{flag}")
        except BenchReceiptError as exc:
            typer.echo(f"{bench_id}  INVALID: {exc}")


@bench_app.command("report")
def bench_report(
    receipt: str = typer.Argument(..., help="Bench id (or unique prefix, or path to receipt.json)."),
    out: Path | None = typer.Option(None, "--out", help="Directory for the files (default: beside the stored receipt)."),
) -> None:
    """Re-render report.txt, report.pdf and chart.json from a stored receipt (byte-identical to the original)."""
    from qrh.bench.receipt import BenchReceiptError, bench_root, resolve_receipt
    from qrh.bench.render import render_bench

    home = _home()
    try:
        r = resolve_receipt(home, receipt)
        files = render_bench(r, out if out is not None else bench_root(home) / r.bench_id())
    except BenchReceiptError as exc:
        raise _fail(str(exc)) from exc
    for f in files:
        typer.echo(str(f))

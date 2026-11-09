"""The ``history`` suite: re-evaluate stored trials with a candidate method versus the default. Read-only.

The input is a harness home directory (for example a private one). Only ``receipts/`` and the files they
record are read, each verified against its recorded hash; nothing is written to it and no database is opened.
For every trial with an S3 receipt, the stored in-sample matrix is pushed through S4 and S5 once with the
default methods and once with the candidate, and the verdicts through S5 are compared. A trial whose inputs
do not verify is skipped with a reason, never trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qrh.bench.engine import evaluate_matrix
from qrh.hashing import hash_json, sha256_file
from qrh.methods.selection import MethodSelector
from qrh.pipeline.receipt import Receipt, Stage, Verdict, load_receipt

_KEEP = ("error", "oos_sharpe", "pbo", "dsr", "s4_pass", "pbo_pass", "dsr_pass", "s5_pass", "pass_all")


def _receipt(trial_dir: Path, stage: Stage) -> Receipt | None:
    p = trial_dir / f"{stage.value}.json"
    return load_receipt(p) if p.is_file() else None


def _stored(trial_dir: Path) -> dict[str, Any]:
    s4, s5 = _receipt(trial_dir, Stage.S4), _receipt(trial_dir, Stage.S5)
    return {
        "s4": s4.verdict.value if s4 else None, "s5": s5.verdict.value if s5 else None,
        "pass_all": bool(s4 and s5 and s4.verdict is Verdict.PASS and s5.verdict is Verdict.PASS),
    }


def _evaluate_trial(home: Path, trial_dir: Path, candidate: MethodSelector, baseline: MethodSelector) -> dict[str, Any]:
    s0, s3 = _receipt(trial_dir, Stage.S0), _receipt(trial_dir, Stage.S3)
    if s0 is None or s3 is None:
        raise ValueError("no S0 or S3 receipt")
    wf = s0.params.get("walk_forward")
    if not isinstance(wf, dict):
        raise ValueError("the hypothesis declares no walk_forward spec")
    path = home / s3.params["is_returns_path"]
    if sha256_file(path) != s3.params["is_returns_hash"]:
        raise ValueError("in-sample returns file does not match the hash in its S3 receipt")
    matrix = pd.read_parquet(path)
    family = int(s3.trial_family_count or matrix.shape[1])
    ppy = float(s3.params["periods_per_year"])
    runs = {
        name: evaluate_matrix(matrix, wf, s0.params["thresholds"], family, sel, periods_per_year=ppy)
        for name, sel in (("default", baseline), ("candidate", candidate))
    }
    stored = _stored(trial_dir)
    changed = runs["default"]["pass_all"] != runs["candidate"]["pass_all"]
    direction = None
    if changed:
        direction = "stricter" if runs["default"]["pass_all"] else "looser"
    return {
        "status": "evaluated", "stored": stored,
        "default": {k: runs["default"][k] for k in _KEEP}, "candidate": {k: runs["candidate"][k] for k in _KEEP},
        "reproduced": runs["default"]["pass_all"] == stored["pass_all"] and runs["default"]["error"] is None,
        "changed": changed, "direction": direction,
    }


def run_history(candidate: MethodSelector, baseline: MethodSelector, path: str | Path) -> dict[str, Any]:
    home = Path(path)
    root = home / "receipts"
    trial_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not trial_dirs:
        raise ValueError(f"no receipts found under {str(home)!r}")
    trials: list[dict[str, Any]] = []
    digests: list[Any] = []
    for d in trial_dirs:
        entry: dict[str, Any] = {"trial_id": d.name}
        try:
            entry.update(_evaluate_trial(home, d, candidate, baseline))
            digests.append([d.name, _receipt(d, Stage.S3).receipt_hash()])  # type: ignore[union-attr]
        except Exception as exc:  # unreadable, tampered or incomplete: skipped, never trusted
            entry.update(status="skipped", reason=f"{type(exc).__name__}: {exc}")
        trials.append(entry)
    done = [t for t in trials if t["status"] == "evaluated"]
    return {
        "source": home.name, "input_hash": hash_json(digests), "trials": trials,
        "summary": {
            "evaluated": len(done), "skipped": len(trials) - len(done),
            "reproduced": sum(1 for t in done if t["reproduced"]),
            "changed": sum(1 for t in done if t["changed"]),
            "stricter": sum(1 for t in done if t["direction"] == "stricter"),
            "looser": sum(1 for t in done if t["direction"] == "looser"),
        },
        "methods": {"candidate": dict(sorted(candidate.used.items())), "default": dict(sorted(baseline.used.items()))},
    }

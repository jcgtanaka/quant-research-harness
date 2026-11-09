"""Helpers shared by the hook tests: synthetic payloads, a seeded registry, in-process hook runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from qrh.hooks.runner import HookOutcome, run_hook
from qrh.memory.registry import HypothesisRecord, Registry
from qrh.pipeline.receipt import Receipt, Stage, Verdict, write_receipt

DB = "registry.sqlite3"


def payload(event: str, **fields: Any) -> dict[str, Any]:
    base = {"session_id": "sess-1", "transcript_path": "", "cwd": str(Path.cwd()),
            "hook_event_name": event, "permission_mode": "default"}
    base.update(fields)
    return base


def bash_payload(command: str, **kw: Any) -> dict[str, Any]:
    return payload("PreToolUse", tool_name="Bash", tool_input={"command": command, "description": "x"},
                   tool_use_id="tu-1", **kw)


def edit_payload(event: str, file_path: str, tool: str = "Edit", **kw: Any) -> dict[str, Any]:
    return payload(event, tool_name=tool, tool_input={"file_path": file_path}, tool_use_id="tu-2", **kw)


def run(name: str, data: dict[str, Any] | str, home: Path, env: dict[str, str] | None = None, **kw: Any) -> HookOutcome:
    text = data if isinstance(data, str) else json.dumps(data)
    return run_hook(name, text, home=home, env=env or {}, **kw)


def approval_id(stderr: str) -> str:
    """The approval id a blocked hook printed (the agent sees exactly this text)."""
    m = re.search(r"qrh approve ([0-9a-f]{12})\b", stderr)
    assert m, f"no approval id in: {stderr!r}"
    return m.group(1)


TEST_PASSPHRASE = "test passphrase, generated keys only"  # gitleaks:allow  (keys are created at runtime)


def ensure_test_keys(home: Path) -> None:
    """A runtime-generated approval key pair in a temporary QRH_HOME (never committed, never a real home)."""
    from datetime import timezone

    from qrh import signing

    if not signing.has_keypair(home):
        signing.init_keys(home, TEST_PASSPHRASE, datetime.now(timezone.utc))


def human_approve_id(home: Path, rid: str, reason: str = "owner reviewed this, fine to proceed",
                     now: datetime | None = None) -> dict[str, Any]:
    """The signing half of `qrh approve`, through the code seam (no terminal in tests)."""
    from datetime import timezone

    from qrh import approvals, signing

    from qrh import integrity

    ensure_test_keys(home)
    key = signing.unlock(home, TEST_PASSPHRASE, datetime.now(timezone.utc))
    if integrity.check(home).state == "missing":  # a human who has a key has also signed the config baseline
        integrity.sign_baseline(home, key, [], datetime.now(timezone.utc))
    return approvals.sign_approval(home, rid, reason, key, now=now)


def human_approve(home: Path, stderr: str, reason: str = "owner reviewed this, fine to proceed",
                  now: datetime | None = None) -> str:
    """What the human does in their own terminal after reading the block message."""
    rid = approval_id(stderr)
    human_approve_id(home, rid, reason, now=now)
    return rid


def write_config(home: Path, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def hook_log(home: Path) -> list[dict[str, Any]]:
    p = home / "logs" / "hooks.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []


def session_trials(home: Path) -> list[dict[str, Any]]:
    p = home / "logs" / "session_trials.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []


def spec_record(text: str = "A synthetic hypothesis about trend persistence", family: str = "trend",
                datasets: list[str] | None = None, **kw: Any) -> HypothesisRecord:
    base: dict[str, Any] = dict(
        text=text, strategy_family=family, asset_class="synthetic", frequency="1D", parameter_space_size=2,
        thresholds={"min_score": 0.5}, difference_statement="Differs from the baseline by lookback.",
        datasets=datasets or [],
    )
    base.update(kw)
    return HypothesisRecord(**base)


def seed_trial(
    home: Path, trial_id: str, *, verdicts: dict[str, Verdict] | None = None,
    metrics: dict[str, dict[str, float]] | None = None, created_at: str | None = None, **spec_kw: Any,
) -> str:
    """Register a trial and write (hash-valid) receipts for the given stage verdicts."""
    home.mkdir(parents=True, exist_ok=True)
    reg = Registry(home / DB)
    try:
        reg.register_hypothesis(spec_record(**spec_kw), trial_id=trial_id)
        h = reg.get_trial(trial_id)["hypothesis_hash"]  # type: ignore[index]
        for stage, verdict in (verdicts or {}).items():
            extra = {"created_at": created_at} if created_at else {}
            r = Receipt(trial_id=trial_id, stage=Stage(stage), hypothesis_hash=h, code_hash="c" * 64,
                        verdict=verdict, metrics=(metrics or {}).get(stage, {}), **extra)
            write_receipt(r, home, reg)
    finally:
        reg.close()
    return trial_id


def register_dataset(home: Path, tmp: Path, name: str = "prices", status: str = "PASS", rows: int = 10,
                     privacy: str | None = None, subdir: str = "") -> Path:
    from qrh.memory.catalog import Catalog

    home.mkdir(parents=True, exist_ok=True)
    folder = tmp / subdir if subdir else tmp
    folder.mkdir(parents=True, exist_ok=True)
    csv = folder / f"{name}.csv"
    lines = ["timestamp,close"] + [f"2020-01-{i + 1:02d}T00:00:00Z,{100 + i}" for i in range(rows)]
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cat = Catalog(home / DB)
    try:
        cat.register_dataset(csv, name=name, privacy=privacy)
        cat.set_status(name, status, [] if status == "PASS" else ["forced for test"])
    finally:
        cat.close()
    return csv

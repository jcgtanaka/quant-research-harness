"""``qrh test-receipt -- <command>``: run a test command and record exit code, source hash and duration."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from qrh.hooks.common import ConfigError, load_config, redact_command
from qrh.hooks.pre_restart_guard import compute_source_hash
from qrh.memory.registry import Registry


@dataclass(frozen=True)
class ReceiptRun:
    source_hash: str
    exit_code: int
    duration_s: float

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


def record_test_receipt(command: list[str], home: Path, registry: Registry, cwd: Path) -> ReceiptRun:
    """PASS only on exit code 0. Raises ConfigError / SourceHashError when the source set is unknown."""
    guard = load_config(home).guard
    if guard is None or not guard.source_globs:
        raise ConfigError("guard.source_globs is not configured in config.yaml")
    source_hash = compute_source_hash(guard.source_globs, cwd)
    started = time.perf_counter()
    try:
        code = subprocess.run(command, cwd=cwd, check=False).returncode
    except FileNotFoundError:
        code = 127
    except PermissionError:
        code = 126
    duration = round(time.perf_counter() - started, 3)
    registry.record_test_receipt(source_hash, redact_command(" ".join(command), limit=500), code, duration)
    return ReceiptRun(source_hash, code, duration)

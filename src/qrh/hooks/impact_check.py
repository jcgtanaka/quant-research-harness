"""PreToolUse (Edit|Write|MultiEdit): warn when a shared module is about to change.

Uses ``codegraph affected`` (the CodeGraph CLI lists the test files affected by a changed source file).
Warn-only: the result goes to the hook log, the exit code is always 0, and a missing or failing
codegraph is logged as "impact check unavailable". The invocation is injectable for tests.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qrh.hooks.common import glob_match
from qrh.hooks.runner import HookContext, HookResult

EDIT_TOOLS = {"Edit", "Write", "MultiEdit"}
TIMEOUT_S = 5
_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__)/|(^|/)test_[^/]*$|_test\.[A-Za-z]+$|\.(spec|test)\.[A-Za-z]+$")
_PATH_LINE = re.compile(r"^[^\s✓✗]+$")


def run_codegraph(
    file: str, cwd: Path, *, which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[int, str] | None:
    """``codegraph affected -q <file>``; None when codegraph is absent or times out. Telemetry is switched off."""
    exe = which("codegraph")
    if exe is None:
        return None
    try:
        done = runner([exe, "affected", "-q", file], cwd=str(cwd), capture_output=True, text=True,
                      timeout=TIMEOUT_S, check=False, env={**os.environ, "CODEGRAPH_TELEMETRY": "0"})
    except (subprocess.TimeoutExpired, OSError):
        return None
    return int(done.returncode), str(done.stdout or "")


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if (p.event and p.event != "PreToolUse") or p.tool_name not in EDIT_TOOLS or not p.file_path:
        return HookResult("allow", "not an edit")
    cwd = Path(p.cwd or ".")
    if not any(glob_match(p.file_path, g, str(cwd)) for g in ctx.config.shared_module_globs):
        return HookResult("allow", "not a shared module")
    runner = ctx.codegraph or run_codegraph
    result = runner(p.file_path, cwd)
    if result is None or result[0] != 0:
        return HookResult("info", "impact check unavailable")
    affected = [ln.strip() for ln in result[1].splitlines() if _PATH_LINE.match(ln.strip())]
    tests = [a for a in affected if _TEST_PATH.search(a)]
    shown = ", ".join(tests[:5]) + (" ..." if len(tests) > 5 else "")
    return HookResult(
        "warn", f"shared module {Path(p.file_path).name}: {len(affected)} affected file(s); tests: {shown or 'none'}",
        extra={"affected_tests": tests, "affected_count": len(affected)},
    )

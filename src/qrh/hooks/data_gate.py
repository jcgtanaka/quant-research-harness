"""PreToolUse (Bash): refuse a backtest whose datasets are not PASS or whose files changed.

Recognized commands: ``qrh run <trial>`` (always) and the ``backtest_commands`` in config.yaml.
Also tags the session with the trial being worked on (``session_trials.jsonl``).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from qrh.hooks.common import tag_session_trial
from qrh.hooks.datacheck import dataset_problems, resolve_trial, trial_datasets
from qrh.hooks.runner import HookContext, HookResult

_QRH = re.compile(r"(?:^|[\s;&|(`])qrh\s+(?P<verb>new|run|report)\b(?P<rest>[^;&|)`]*)")
_HEX_TOKEN = re.compile(r"\b[0-9a-fA-F]{8,}\b")
_STAGE_OPT = re.compile(r"--stage(?:=|\s+)(?P<stage>[A-Za-z]\d)")


@dataclass(frozen=True)
class Target:
    kind: str  # run | report | new | backtest
    trial_ref: str | None  # a trial id (or prefix) as written in the command
    datasets: tuple[str, ...] = ()
    stage: str | None = None


def _positional(rest: str) -> list[str]:
    """Positional arguments after the verb (options and the value of --stage skipped)."""
    try:
        tokens = shlex.split(rest)
    except ValueError:
        tokens = rest.split()
    out: list[str] = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
        elif tok == "--stage":
            skip = True
        elif not tok.startswith("-"):
            out.append(tok)
    return out


def find_targets(command: str, ctx: HookContext) -> list[Target]:
    targets: list[Target] = []
    for m in _QRH.finditer(command):
        verb, rest = m["verb"], m["rest"]
        args = _positional(rest)
        stage = None
        if sm := _STAGE_OPT.search(rest):
            stage = sm["stage"].upper()
        if verb == "new":
            targets.append(Target("new", None))
        else:
            targets.append(Target(verb, args[0] if args else None, stage=stage))
    for bc in ctx.config.backtest_commands:
        m = bc.pattern.search(command)
        if m is None:
            continue
        if bc.trial_from_args:
            ref = m.groupdict().get("trial")
            if ref is None:
                hit = _HEX_TOKEN.search(command, m.end())
                ref = hit.group(0) if hit else None
            targets.append(Target("backtest", ref))
        else:
            targets.append(Target("backtest", None, datasets=bc.datasets))
    return targets


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if (p.event and p.event != "PreToolUse") or p.tool_name != "Bash" or not p.command:
        return HookResult("allow", "not a Bash command")
    targets = find_targets(p.command, ctx)
    if not targets:
        return HookResult("allow", "not a backtest command")

    needs_registry = any(t.trial_ref or t.datasets for t in targets if t.kind in ("run", "backtest", "report"))
    conn = ctx.registry().conn if needs_registry else None
    problems: list[str] = []
    trial_for_log: str | None = None
    to_check: dict[str, None] = {}
    for t in targets:
        resolved = None
        if t.trial_ref is not None:
            assert conn is not None
            resolved = resolve_trial(conn, t.trial_ref)
            trial_for_log = resolved or trial_for_log
        tag_session_trial(ctx.home, p.session_id, resolved or (t.trial_ref if t.kind == "report" else None),
                          t.stage, t.kind, ctx.now())
        if t.kind in ("run", "backtest"):
            if t.datasets:
                to_check.update(dict.fromkeys(t.datasets))
            elif t.trial_ref is None:
                problems.append("cannot determine which trial this command runs")
            elif resolved is None:
                problems.append(f"unknown trial {t.trial_ref!r}")
            else:
                to_check.update(dict.fromkeys(trial_datasets(conn, resolved)))  # type: ignore[arg-type]
    if to_check:
        db = conn or ctx.registry().conn
        for name in to_check:
            for why in dataset_problems(db, name):
                problems.append(f"dataset {name!r}: {why}")
    if problems:
        return HookResult(
            "block", f"{len(problems)} problem(s): " + "; ".join(problems)[:150], trial_id=trial_for_log,
            stderr="Backtest not started; the data gate failed:\n" + "\n".join(f"  - {x}" for x in problems)
            + "\nRegister and check the data first (qrh data register / qrh data check <name>).",
        )
    return HookResult("allow", "datasets PASS and unchanged" if to_check else "no datasets to check", trial_id=trial_for_log)

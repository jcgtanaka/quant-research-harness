"""Claude Code settings: the hook registrations qrh needs, and a merge that never removes anything else."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from qrh.hooks.specs import HOOKS

SCOPE_DEFAULTS = {"user": "~/.claude/settings.json", "project": ".claude/settings.json"}


class SettingsError(ValueError):
    """The existing settings have a shape this merge refuses to touch."""


def command_for(name: str) -> str:
    return f"qrh hook {name}"


def _entry(name: str) -> dict[str, Any]:
    return {"type": "command", "command": command_for(name), "timeout": HOOKS[name].timeout}


def desired_settings() -> dict[str, Any]:
    """The hooks section qrh wants, one group per (event, matcher)."""
    return merge_settings({})


def merge_settings(existing: Any) -> dict[str, Any]:
    """Return a copy of ``existing`` with every qrh hook present exactly once. Idempotent. Removes nothing of anyone
    else's; only a qrh entry registered under an outdated matcher moves to the current one."""
    if not isinstance(existing, dict):
        raise SettingsError("settings must be a JSON object")
    out = copy.deepcopy(existing)
    hooks = out.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise SettingsError("'hooks' must be an object")
    for spec in HOOKS.values():
        groups = hooks.setdefault(spec.event, [])
        if not isinstance(groups, list) or not all(isinstance(g, dict) for g in groups):
            raise SettingsError(f"'hooks.{spec.event}' must be a list of objects")
        for g in groups:
            if not isinstance(g.get("hooks", []), list):
                raise SettingsError(f"'hooks.{spec.event}[].hooks' must be a list")
        cmd = command_for(spec.name)
        for g in list(groups):  # our entry under an outdated matcher moves; foreign entries and groups stay
            if g.get("matcher") != spec.matcher and any(isinstance(e, dict) and e.get("command") == cmd
                                                        for e in g.get("hooks", [])):
                g["hooks"] = [e for e in g["hooks"] if not (isinstance(e, dict) and e.get("command") == cmd)]
                if not g["hooks"]:
                    groups.remove(g)
        group = next((g for g in groups if g.get("matcher") == spec.matcher), None)
        if group is None:
            group = ({"matcher": spec.matcher} if spec.matcher else {}) | {"hooks": []}
            groups.append(group)
        entries = group.setdefault("hooks", [])
        for i, e in enumerate(entries):
            if isinstance(e, dict) and e.get("command") == cmd:
                entries[i] = _entry(spec.name)
                break
        else:
            entries.append(_entry(spec.name))
    return out


def read_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingsError(f"cannot read {path} as JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SettingsError(f"{path} must contain a JSON object")
    return doc


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def write_settings(path: Path, merged: dict[str, Any], now: datetime | None = None) -> Path | None:
    """Write atomically; if the file existed, copy it to a timestamped backup next to it first."""
    backup: Path | None = None
    if path.exists():
        stamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        n = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.bak-{stamp}-{n}")
            n += 1
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(render(merged))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return backup

"""Walk Claude Code transcripts and keep one metrics record per session in <QRH_HOME>/usage/sessions.json.

Records are merged by session id across runs, so they outlive Claude Code's transcript sweep; a newer
harvest of a transcript that is still present replaces its record. Records hold counts only: no message
text, no paths, no cwd beyond the project label.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any

from qrh.usage import alignment, attribution
from qrh.usage.parse import TokenCounts, Transcript, parse_ts, parse_transcript
from qrh.usage.settings import Prices, Thresholds, load_prices, load_thresholds, price_cost, usage_dir

ARCHIVE_VERSION = 1
INLINE = "_inline"  # sidechain records that live in the main transcript


@dataclass
class HarvestSummary:
    scanned: int = 0
    new: int = 0
    updated: int = 0
    archived_only: int = 0
    skipped_empty: int = 0
    subagent_files: int = 0
    malformed_lines: int = 0


def claude_dir_default() -> Path:
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(env) if env else Path.home() / ".claude"


def archive_path(home: str | Path) -> Path:
    return usage_dir(home) / "sessions.json"


def load_archive(home: str | Path) -> dict[str, dict[str, Any]]:
    path = archive_path(home)
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        sessions = doc["sessions"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(sessions, dict):
        raise ValueError(f"{path}: 'sessions' must be a mapping")
    return sessions


def _write_archive(home: str | Path, sessions: dict[str, dict[str, Any]]) -> None:
    path = archive_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    doc = {"version": ARCHIVE_VERSION, "sessions": sessions}
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --- one session -----------------------------------------------------------------------------------


def project_label(cwds: Counter[str]) -> str:
    """Basename of the most frequent cwd (ties: the smallest name); ``home`` for the home directory."""
    if not cwds:
        return "unknown"
    best = sorted(cwds.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    stripped = best.rstrip("/\\")
    if stripped and Path(stripped) == Path.home():
        return "home"
    base = re.split(r"[\\/]", stripped)[-1]
    return base or "unknown"


def _models_of(transcript: Transcript, sidechain: bool | None) -> dict[str, TokenCounts]:
    out: dict[str, TokenCounts] = {}
    for r in transcript.requests:
        if sidechain is None or r.sidechain == sidechain:
            out.setdefault(r.model, TokenCounts()).add(r.tokens)
    return out


def _dump(models: dict[str, TokenCounts]) -> dict[str, dict[str, int]]:
    return {m: models[m].as_dict() for m in sorted(models)}


def _main_chain_metrics(t: Transcript, th: Thresholds) -> dict[str, Any]:
    main = [r for r in t.requests if not r.sidechain]
    peak = max((r.tokens.context for r in main), default=0)
    ctx = sum(r.tokens.context for r in main)
    reads = sum(r.tokens.cache_read for r in main)
    cold = 0
    cold_writes: dict[str, dict[str, int]] = {}
    for i, r in enumerate(main):
        if i == 0 or r.after_compaction or r.tokens.context == 0:
            continue
        if 100 * r.tokens.cache_read / r.tokens.context < th.cache_cold_pct:
            cold += 1
            w = cold_writes.setdefault(r.model, {"cache_write_5m": 0, "cache_write_1h": 0})
            w["cache_write_5m"] += r.tokens.cache_write_5m
            w["cache_write_1h"] += r.tokens.cache_write_1h
    return {"peak_context": peak, "hit_pct": (100 * reads / ctx) if ctx else None, "cold_cache_events": cold,
            "_cold_writes": {m: cold_writes[m] for m in sorted(cold_writes)}}


def _estimate(all_models: dict[str, TokenCounts], prices: Prices | None) -> tuple[float | None, dict[str, float], str]:
    if prices is None:
        return None, {}, "prices.json is missing, so no estimate can be made"
    missing = sorted(m for m in all_models if m not in prices.models)
    if missing:
        return None, {}, f"no price for model(s) {missing} in prices.json"
    per_model = {
        m: price_cost(prices.models[m], input=t.input, output=t.output, cache_read=t.cache_read,
                      cache_write_5m=t.cache_write_5m, cache_write_1h=t.cache_write_1h)
        for m, t in all_models.items()
    }
    return sum(per_model.values()), per_model, ""


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "absent"


def _hooks_digest(path: Path) -> str:
    try:
        hooks = json.loads(path.read_text(encoding="utf-8")).get("hooks")
    except (OSError, ValueError, AttributeError):
        return "absent"
    return hashlib.sha256(json.dumps(hooks, sort_keys=True).encode()).hexdigest()


def config_hash(claude_dir: Path | None, cwd: str | None) -> str:
    """Hash of the harness configuration in force: the hooks section of the user (and project) settings and the
    bytes of the user and project CLAUDE.md. Only a digest is kept; nothing of the files or their paths."""
    parts = [_hooks_digest(claude_dir / "settings.json") if claude_dir else "absent",
             _file_digest(claude_dir / "CLAUDE.md") if claude_dir else "absent"]
    if cwd:
        proj = Path(cwd)
        parts += [_hooks_digest(proj / ".claude" / "settings.json"), _file_digest(proj / "CLAUDE.md"),
                  _file_digest(proj / ".claude" / "CLAUDE.md")]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _turn_records(t: Transcript) -> list[dict[str, Any]]:
    per: list[dict[str, TokenCounts]] = [{} for _ in t.turns]
    for r in t.requests:
        if not r.sidechain:
            per[r.turn].setdefault(r.model, TokenCounts()).add(r.tokens)
    return [{"ts": tr.ts, "tools": tr.tools, "failures": tr.failures, "models": _dump(per[i])}
            for i, tr in enumerate(t.turns)]


def build_session(
    session_id: str, main_path: Path, sub_paths: list[Path], th: Thresholds, prices: Prices | None,
    tags: list[attribution.Tag], hooks: dict[str, Any], claude_dir: Path | None = None,
) -> dict[str, Any] | None:
    """The metrics record of one session, or None when the transcript holds no API usage at all."""
    t = parse_transcript(main_path)
    main_models = _models_of(t, False)
    subagents: dict[str, dict[str, Any]] = {}
    inline = _models_of(t, True)
    if inline:
        subagents[INLINE] = {"models": _dump(inline), "cost_state": False, "cost_state_usd": None}
    malformed = t.malformed
    all_requests = list(t.requests)
    for sp in sub_paths:
        st = parse_transcript(sp)
        malformed += st.malformed
        models = _models_of(st, None)
        all_requests.extend(st.requests)
        if models or st.cost_state:
            subagents[sp.stem] = {
                "models": _dump(models), "cost_state": st.cost_state is not None,
                "cost_state_usd": st.cost_state["total_cost_usd"] if st.cost_state else None,
            }
    if not all_requests and t.cost_state is None:
        return None

    everything: dict[str, TokenCounts] = {}
    for r in all_requests:
        everything.setdefault(r.model, TokenCounts()).add(r.tokens)
    cs = t.cost_state
    if cs is not None:
        cost, source, reason = cs["total_cost_usd"], "harness", ""
        by_model = {m: v["cost_usd"] for m, v in sorted(cs["models"].items())}
    else:
        cost, by_model, reason = _estimate(everything, prices)
        source = "estimate" if cost is not None else "none"
        by_model = dict(sorted(by_model.items()))

    aligned = None
    if cs is not None and not subagents:
        aligned = alignment.find_alignment([r for r in t.requests if not r.sidechain], cs["models"],
                                           t.cost_state_main_requests)

    attrib = attribution.attribute(all_requests, tags)
    split, how = attribution.allocate_cost(attrib, cost, prices)
    trials = [
        {"trial_id": k[0], "stage": k[1], "models": _dump(attrib[k]), "cost_usd": split[k]}
        for k in sorted(attrib, key=lambda k: (k[0], k[1] or ""))
    ]
    start, end = parse_ts(t.first_ts), parse_ts(t.last_ts)
    mc = _main_chain_metrics(t, th)
    cold_writes = mc.pop("_cold_writes")
    best_cwd = sorted(t.cwds.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if t.cwds else None
    return {
        "session_id": session_id, "project_label": project_label(t.cwds),
        "start": t.first_ts, "end": t.last_ts,
        "duration_s": int((end - start).total_seconds()) if start and end else None,
        "models": _dump(main_models), "models_raw": _dump(t.raw_main), "subagents": subagents,
        "cost_usd": cost, "cost_source": source, "cost_reason": reason, "cost_by_model": by_model,
        "harness_usage": cs["models"] if cs else None, "alignment": aligned,
        **mc,
        "turns": _turn_records(t), "config_hash": config_hash(claude_dir, best_cwd),
        "waste": {"reread_runs": dict(sorted(t.reread_runs.items())), "output_tokens": sorted(t.output_tokens, reverse=True),
                  "failing_commands": dict(sorted(t.failing_commands.items())), "cold_writes": cold_writes},
        "tools": dict(sorted(t.tools.items())), "compactions": t.compactions,
        "malformed_lines": malformed, "hooks": hooks, "trials": trials, "cost_alloc": how,
    }


# --- the whole tree ------------------------------------------------------------------------------


def discover(claude_dir: Path) -> list[tuple[str, Path, list[Path]]]:
    """(session id, main transcript, subagent transcripts) for every session, in a stable order."""
    found = []
    for project in sorted((claude_dir / "projects").glob("*")):
        if not project.is_dir():
            continue
        for main in sorted(project.glob("*.jsonl")):
            subs = sorted((project / main.stem / "subagents").glob("*.jsonl"))
            found.append((main.stem, main, subs))
    return found


def harvest(claude_dir: str | Path, home: str | Path) -> HarvestSummary:
    """Harvest ``claude_dir`` into the archive under ``home``. Raises SettingsError without thresholds."""
    th = load_thresholds(home)
    prices = load_prices(home)
    tags, stats = attribution.load_session_tags(home), attribution.load_hook_stats(home)
    archive = load_archive(home)
    summary = HarvestSummary()
    seen: set[str] = set()
    for sid, main, subs in discover(Path(claude_dir)):
        summary.scanned += 1
        summary.subagent_files += len(subs)
        rec = build_session(sid, main, subs, th, prices, tags.get(sid, []), stats.get(sid) or attribution.hook_stats([]),
                            Path(claude_dir))
        if rec is not None and archive.get(sid, {}).get("config_hash"):
            rec["config_hash"] = archive[sid]["config_hash"]  # captured when the session was first harvested
        if rec is None:
            summary.skipped_empty += 1
            continue
        summary.malformed_lines += rec["malformed_lines"]
        seen.add(sid)
        if sid in archive:
            summary.updated += 1
        else:
            summary.new += 1
        archive[sid] = rec
    summary.archived_only = sum(1 for sid in archive if sid not in seen)
    _write_archive(home, dict(sorted(archive.items())))
    return summary


def week_of(ts: str | None) -> str | None:
    when = parse_ts(ts)
    if when is None:
        return None
    y, w, _ = when.astimezone(timezone.utc).isocalendar()
    return f"{y}-W{w:02d}"


def day_of(ts: str | None) -> str | None:
    when = parse_ts(ts)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%d") if when else None


"""Check our token totals against the harness's own cost-state, per session and per model.

This is the proof that deduplication is right. For every archived session that has a cost-state it
compares token kinds under two hypotheses about the harness figure: it covers the main chain only, or the
main chain plus subagent transcripts. It also shows what the totals would be without deduplication.

Scopes: ``transcript`` (the default) compares only the models seen in the transcripts and only input, cache read
and cache creation, the kinds deduplication changes. ``strict`` compares every model and kind in the cost-state;
it is a diagnostic that is expected to fail on real data, because Claude Code bills a background model that never
appears in transcripts and subagent transcripts under-record output tokens.

The gate (``VerifyResult.ok``): in ``transcript`` scope only sessions WITHOUT subagents count, because
subagent token totals are a lower bound; they are reported, with their match rate, but do not gate. A mismatch
that ``alignment.py`` explains by an exact reconstruction (stale cost-state, restarted counter, interrupted
message) counts as a match and is listed as explained.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOLERANCE_PCT = 1.0
KINDS = ("input", "output", "cache_read", "cache_creation")
INPUT_SIDE = ("input", "cache_read", "cache_creation")  # the kinds deduplication changes most
SCOPES = ("transcript", "strict")
DEFAULT_SCOPE = "transcript"
INLINE = "_inline"
MAIN, WITH_SUB, SAME = "main", "main+subagents", "same (no subagents)"
TIE = "indistinguishable"


@dataclass(frozen=True)
class Row:
    session_id: str
    project_label: str
    start: str | None
    main_diff_pct: float
    all_diff_pct: float
    better: str
    best_diff_pct: float
    ok: bool
    has_subagents: bool
    explained: str | None = None  # alignment kind that reconciles a plain mismatch (transcript scope only)
    gating: bool = True  # whether this row can fail the gate
    acknowledged: bool = False  # a human acknowledged this unexplained mismatch and its numbers have not changed since


@dataclass
class VerifyResult:
    rows: list[Row] = field(default_factory=list)
    inflation_factor: float | None = None
    subagent_files: int = 0
    subagent_files_with_cost_state: int = 0
    better_counts: dict[str, int] = field(default_factory=dict)
    causes: dict[str, int] = field(default_factory=dict)  # sessions per observed cause of a strict mismatch
    explained: dict[str, int] = field(default_factory=dict)  # sessions reconciled by an alignment, per kind
    scope: str = DEFAULT_SCOPE
    expired_acks: list[str] = field(default_factory=list)  # session ids whose acknowledgement no longer matches the numbers

    @property
    def verified(self) -> int:
        return len(self.rows)

    @property
    def matched(self) -> int:
        return sum(r.ok for r in self.rows)

    @property
    def mismatched(self) -> int:
        return sum(not r.ok for r in self.rows)

    @property
    def gating_verified(self) -> int:
        return sum(r.gating for r in self.rows)

    @property
    def gating_mismatched(self) -> int:
        return sum(r.gating and not r.ok and not r.acknowledged for r in self.rows)

    @property
    def acknowledged_rows(self) -> list[Row]:
        return [r for r in self.rows if r.acknowledged]

    @property
    def subagent_sessions(self) -> int:
        return sum(r.has_subagents for r in self.rows)

    @property
    def subagent_matched(self) -> int:
        return sum(r.has_subagents and r.ok for r in self.rows)

    @property
    def ok(self) -> bool:
        """No gating session mismatches (a caller must still refuse an empty gate)."""
        return self.gating_mismatched == 0

    @property
    def subagent_finding(self) -> str:
        w, m = self.better_counts.get(WITH_SUB, 0), self.better_counts.get(MAIN, 0)
        if w == 0 and m == 0:
            return "not determined (no session with subagents and a cost-state)"
        decisive = w + m
        if w >= 4 * m:
            return f"cost-state {'includes' if not m else 'mostly includes'} subagents ({w} of {decisive} decisive session(s))"
        if m >= 4 * w:
            return f"cost-state {'excludes' if not w else 'mostly excludes'} subagents ({m} of {decisive} decisive session(s))"
        return f"mixed ({w} session(s) match better with subagents, {m} without)"


def _sum_models(parts: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for models in parts:
        for name, c in models.items():
            o = out.setdefault(name, dict.fromkeys(KINDS, 0))
            o["input"] += c.get("input", 0)
            o["output"] += c.get("output", 0)
            o["cache_read"] += c.get("cache_read", 0)
            o["cache_creation"] += c.get("cache_write_5m", 0) + c.get("cache_write_1h", 0)
    return out


def _diff_pct(ours: dict[str, dict[str, int]], theirs: dict[str, dict[str, Any]],
              kinds: tuple[str, ...] = KINDS) -> float:
    worst = 0.0
    for model in set(ours) | set(theirs):
        a, b = ours.get(model, {}), theirs.get(model, {})
        for k in kinds:
            x, y = a.get(k, 0), b.get(k, 0)
            d = (0.0 if x == 0 else 100.0) if y == 0 else 100.0 * abs(x - y) / y
            worst = max(worst, d)
    return worst


def _total(models: dict[str, dict[str, int]]) -> int:
    return sum(sum(c.values()) for c in models.values())


C_ONLY = "cost-state has a model the transcripts lack"
C_OUT = "output tokens below cost-state"
C_ABOVE = "transcript above cost-state"
C_IN = "input side below cost-state"


def _causes(ours: dict[str, dict[str, int]], theirs: dict[str, dict[str, Any]], tol: float) -> set[str]:
    found: set[str] = set()
    for model, b in theirs.items():
        a = ours.get(model)
        if a is None:
            if any(b.get(k, 0) for k in KINDS):
                found.add(C_ONLY)
            continue
        for k in KINDS:
            x, y = a.get(k, 0), b.get(k, 0)
            if y and 100.0 * (y - x) / y > tol:
                found.add(C_OUT if k == "output" else C_IN)
            if x > y and (y == 0 or 100.0 * (x - y) / y > tol):
                found.add(C_ABOVE)
    found.update(C_ABOVE for m in ours if m not in theirs and any(ours[m].values()))
    return found


MIN_ACK_REASON = 10
ACK_VERSION = 1


class AckError(ValueError):
    """An acknowledgement cannot be read, written or recorded."""


def acks_path(home: str | Path) -> Path:
    return Path(home) / "usage" / "verify_acks.json"


def session_fingerprint(s: dict[str, Any]) -> str:
    """Digest of the numbers a verdict rests on: if a re-harvest changes any of them, an acknowledgement lapses."""
    parts = [s.get("models") or {}, s.get("harness_usage") or {},
             {k: (a.get("models") or {}) for k, a in sorted((s.get("subagents") or {}).items())}]
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def load_acks(home: str | Path) -> dict[str, dict[str, Any]]:
    path = acks_path(home)
    if not path.exists():
        return {}
    try:
        acks = json.loads(path.read_text(encoding="utf-8"))["acks"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AckError(f"cannot read {path}: {exc}") from exc
    if not isinstance(acks, dict):
        raise AckError(f"{path}: 'acks' must be a mapping")
    return acks


def record_ack(home: str | Path, session: dict[str, Any], alias: str, row: "Row", reason: str,
               now: datetime | None = None) -> dict[str, Any]:
    if len(reason.strip()) < MIN_ACK_REASON:
        raise AckError(f"the acknowledgement reason must be at least {MIN_ACK_REASON} characters")
    acks = load_acks(home)
    entry = {"alias": alias, "reason": reason.strip(), "ts": (now or datetime.now(timezone.utc)).isoformat(),
             "fingerprint": session_fingerprint(session),
             "figures": {"main_diff_pct": round(row.main_diff_pct, 4), "all_diff_pct": round(row.all_diff_pct, 4),
                         "best_diff_pct": round(row.best_diff_pct, 4)}}
    acks[session["session_id"]] = entry
    path = acks_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"version": ACK_VERSION, "acks": acks}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return entry


def verify_sessions(
    sessions: dict[str, dict[str, Any]], tolerance_pct: float = TOLERANCE_PCT, scope: str = DEFAULT_SCOPE,
    acks: dict[str, dict[str, Any]] | None = None,
) -> VerifyResult:
    """``transcript`` (default) compares only the models the transcripts contain, and only input, cache read and
    cache creation: what deduplication can change. ``strict`` compares every model and kind in the cost-state."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; use one of {list(SCOPES)}")
    res = VerifyResult(scope=scope)
    causes: dict[str, int] = {}
    raw_total = dedup_total = 0
    better_counts: dict[str, int] = {}
    explained: dict[str, int] = {}
    for sid in sorted(sessions, key=lambda k: (sessions[k].get("start") or "", k)):
        s = sessions[sid]
        theirs = s.get("harness_usage")
        if not theirs:
            continue
        subs = s.get("subagents") or {}
        main = _sum_models([s.get("models") or {}])
        everything = _sum_models([s.get("models") or {}, *[a.get("models") or {} for a in subs.values()]])
        if scope == "transcript":
            visible = {m: v for m, v in theirs.items() if m in everything}
            d_main, d_all = _diff_pct(main, visible, INPUT_SIDE), _diff_pct(everything, visible, INPUT_SIDE)
        else:
            d_main, d_all = _diff_pct(main, theirs), _diff_pct(everything, theirs)
        if not subs:
            better, best = SAME, d_main
        elif abs(d_main - d_all) <= tolerance_pct:
            better, best = TIE, min(d_main, d_all)  # the subagent tokens are too few to tell the hypotheses apart
        elif d_all < d_main:
            better, best = WITH_SUB, d_all
        else:
            better, best = MAIN, d_main
        kind = None
        al = s.get("alignment")
        if scope == "transcript" and not subs and best > tolerance_pct and isinstance(al, dict):
            d_al = _diff_pct(_sum_models([al.get("models") or {}]), visible, INPUT_SIDE)
            if d_al <= tolerance_pct:
                kind, best = str(al.get("kind")), d_al
                explained[kind] = explained.get(kind, 0) + 1
        better_counts[better] = better_counts.get(better, 0) + 1
        if _diff_pct(everything if better == WITH_SUB else main, theirs) > tolerance_pct:
            for c in _causes(everything if better == WITH_SUB else main, theirs, tolerance_pct):
                causes[c] = causes.get(c, 0) + 1
        raw_total += _total(_sum_models([s.get("models_raw") or {}]))
        dedup_total += _total(main)
        files = {k: a for k, a in subs.items() if k != INLINE}  # sidechain records inside the main file
        res.subagent_files += len(files)
        res.subagent_files_with_cost_state += sum(1 for a in files.values() if a.get("cost_state"))
        row = Row(sid, s.get("project_label", ""), s.get("start"), d_main, d_all, better, best,
                  best <= tolerance_pct, bool(subs), kind, scope == "strict" or not subs)
        if acks and sid in acks and scope == "transcript" and row.gating and not row.ok:
            if acks[sid].get("fingerprint") == session_fingerprint(s):
                row = replace(row, acknowledged=True)
            else:
                res.expired_acks.append(sid)
        res.rows.append(row)
    res.better_counts = better_counts
    res.explained = dict(sorted(explained.items()))
    res.causes = dict(sorted(causes.items()))
    res.inflation_factor = (raw_total / dedup_total) if dedup_total else None
    return res


MAX_LISTED = 20


def render_verify(res: VerifyResult, ranks: dict[str, int], labels: dict[str, str], tolerance_pct: float = TOLERANCE_PCT) -> str:
    """Summary table plus the worst mismatches. Sessions are named S001.. by start time, never by id."""
    inflation = f"{res.inflation_factor:.3f}x" if res.inflation_factor is not None else "n/a"
    bc = res.better_counts
    strict = res.scope == "strict"
    lines = [
        f"verified sessions   {res.verified} (with a harness cost-state)",
        f"scope               {res.scope}" + (
            " (every model and kind in the cost-state; a diagnostic, expected to fail on real data)" if strict
            else " (models seen in transcripts; input, cache read, cache creation)"),
        f"matched             {res.matched} (within {tolerance_pct:g}% under the better-matching hypothesis)",
        f"mismatched          {res.mismatched}",
    ]
    if strict:
        lines.append(f"gate                every session: {res.gating_mismatched} mismatch(es) fail it")
    else:
        g_ok = res.gating_verified - res.gating_mismatched
        n_expl = sum(res.explained.values())
        n_ack = len(res.acknowledged_rows)
        lines.append(f"gate                sessions without subagents: {res.gating_verified} verified, {g_ok - n_ack} match "
                     f"({n_expl} of them explained), {n_ack} acknowledged, {res.gating_mismatched} unexplained mismatch(es)")
        if res.explained:
            lines.append("explained           " + ", ".join(f"{k} {n}" for k, n in sorted(res.explained.items())))
        sub = res.subagent_sessions
        rate = f"{100 * res.subagent_matched / sub:.0f}%" if sub else "n/a"
        lines.append(f"subagent sessions   {sub} verified, {res.subagent_matched} match ({rate}); "
                     "not gating: their tokens are a lower bound")
    lines += [
        f"dedup inflation     {inflation} (token totals without deduplication / with, main chain)",
        f"subagent files      {res.subagent_files} ({res.subagent_files_with_cost_state} with their own cost-state)",
        f"better match        main {bc.get(MAIN, 0)}, main+subagents {bc.get(WITH_SUB, 0)}, "
        f"indistinguishable {bc.get(TIE, 0)}, "
        f"no subagents {bc.get(SAME, 0)}",
        f"subagent finding    {res.subagent_finding}",
    ]
    if res.causes:
        lines.append("")
        lines.append("observed causes among sessions that differ strictly (a session can have several):")
        lines.extend(f"  {n:4d}  {c}" for c, n in sorted(res.causes.items(), key=lambda kv: (-kv[1], kv[0])))
    for sid in res.expired_acks:
        lines.append(f"acknowledgement expired for S{ranks.get(sid, 0):03d}: its numbers changed since it was acknowledged, "
                     "so it gates again")
    if res.acknowledged_rows:
        lines.append("")
        lines.append("acknowledged, unexplained (not gating; the acknowledgement lapses if the numbers change):")
        for r in res.acknowledged_rows:
            label = labels.get(r.project_label, r.project_label)
            lines.append(f"  S{ranks.get(r.session_id, 0):03d} {label}: best {r.best_diff_pct:.2f}%")
    bad = sorted((r for r in res.rows if not r.ok and not r.acknowledged), key=lambda r: (not r.gating, -r.best_diff_pct))
    if bad:
        lines.append("")
        lines.append(f"mismatches (gating first, then worst, up to {MAX_LISTED}):")
        for r in bad[:MAX_LISTED]:
            label = labels.get(r.project_label, r.project_label)
            tag = ("unexplained" if r.gating else "has subagents, not gating, tokens are a lower bound")
            lines.append(f"  S{ranks.get(r.session_id, 0):03d} {label}: main {r.main_diff_pct:.2f}%, "
                         f"main+subagents {r.all_diff_pct:.2f}%, best {r.best_diff_pct:.2f}% [{tag}]")
    return "\n".join(lines) + "\n"

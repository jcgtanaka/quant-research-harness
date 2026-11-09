"""Stop: block a final answer that quotes performance numbers no harness receipt backs.

Reads the last assistant message from the transcript, extracts claims such as "Sharpe of 1.4" or
"12% max drawdown", and accepts each one that matches a metric value in a receipt (within tolerance,
percent and fraction both considered). A claim only matches an observed metric of the same family (for
example "Sharpe" only against keys with a ``sharpe`` word); thresholds, limits, counts and SR0 never count.
Receipts of trials named in the message (id prefix of at least
8 hex characters) are used if there are any, else receipts created in the last ``recent_days`` days.
The word-to-key table is ``FAMILY_KEYS``, overridable in config (``receipt_required.metric_map``).

Exceptions to fail-closed, on purpose: an unreadable transcript exits 0, because Stop is not the
place to trap the user. A repeated Stop (``stop_hook_active``) is checked like the first; the runner caps the repeats.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qrh.hooks.runner import HookContext, HookResult
from qrh.pipeline.receipt import ReceiptTamperedError, load_receipt

PERCENT_TYPE = {"cagr", "return", "drawdown", "win rate", "volatility"}
NEEDS_PERCENT_SIGN = {"return"}  # "returns 3 rows" is not a performance claim
# One table: claim word -> key phrases (whole underscore-separated words) that identify its metric family.
FAMILY_KEYS: dict[str, tuple[str, ...]] = {
    "sharpe": ("sharpe",),
    "sortino": ("sortino",),
    "cagr": ("cagr",),
    "return": ("return", "cagr"),
    "drawdown": ("drawdown", "max_dd"),
    "pbo": ("pbo",),
    "dsr": ("dsr",),
    "volatility": ("vol", "volatility"),
    "win rate": ("win_rate",),
    "profit factor": ("profit_factor",),
}
# Keys that are not observed performance: limits, thresholds, counts and sizes, null-model expectations.
EXCLUDED_PREFIXES = ("min_", "max_", "limit")
EXCLUDED_WORDS = frozenset(
    "n count size grid observations folds bars reps level multiplier sr0 var variance skew kurtosis rows "
    "dropped combinations evaluated threshold bound bounds param params".split()
)
_BUILTIN_NAMES = (
    r"deflated\s+sharpe(?:\s+ratio)?|annuali[sz]ed\s+returns?|probability\s+of\s+backtest\s+overfitting"
    r"|max(?:imum)?\s+dd|vol|sharpe(?:\s+ratio)?|sortino(?:\s+ratio)?|cagr|returns?|draw-?down|pbo|dsr|win\s+rate|volatility"
    r"|profit\s+factor"
)
_GAP_WORDS = (
    "ratio|of|is|was|are|were|at|about|around|roughly|approximately|near|over|under|above|below|reached|hit|to"
    "|equals?|rose|fell|improved|dropped|increased|decreased|up|down|from|annuali[sz]ed|annual|net|gross|oos"
    "|out-of-sample|in-sample|total|cumulative|average|avg|mean|max(?:imum)?"
)
_ADJECTIVES = (
    "annuali[sz]ed|annual|max(?:imum)?|net|gross|oos|out-of-sample|in-sample|total|cumulative|average|avg"
    "|mean|daily|monthly|yearly|peak-to-trough"
)
_NUM = r"(?P<sign>[-+−–]?)\s?(?P<digits>\d+(?:\.\d+)?)(?P<unit>\s?%|x\b)?"
_HEX_TOKEN = re.compile(r"\b[0-9a-fA-F]{8,}\b")


@dataclass(frozen=True)
class Claim:
    name: str
    value: float
    percent: bool
    text: str


def _canonical(raw: str) -> str:
    low = " ".join(raw.lower().split())
    for prefix, name in (("deflated", "dsr"), ("annuali", "cagr"), ("probability", "pbo"), ("max", "drawdown"),
                         ("vol", "volatility"), ("sharpe", "sharpe"), ("sortino", "sortino"), ("return", "return"),
                         ("draw", "drawdown"), ("win", "win rate"), ("volatil", "volatility"),
                         ("profit", "profit factor")):
        if low.startswith(prefix):
            return name
    return low


def _patterns(extra: tuple[str, ...]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    names = _BUILTIN_NAMES + "".join("|" + re.escape(e).replace(r"\ ", r"\s+") for e in extra if e)
    name = rf"(?P<name>{names})"
    forward = re.compile(
        rf"(?<![A-Za-z]){name}(?![A-Za-z])(?:\s|[:=~,()]|\b(?:{_GAP_WORDS})\b)*{_NUM}", re.IGNORECASE)
    backward = re.compile(
        rf"(?<![\d.]){_NUM}\s*(?:(?:{_ADJECTIVES})\s+)*{name}(?![A-Za-z])", re.IGNORECASE)
    return forward, backward


def _claim(m: re.Match[str], text: str) -> Claim | None:
    name = _canonical(m["name"])
    unit = (m["unit"] or "").strip()
    if name in NEEDS_PERCENT_SIGN and unit != "%":
        return None
    value = float(m["digits"])
    if m["sign"] in ("-", "−", "–"):
        value = -value
    return Claim(name, value, unit == "%", " ".join(m.group(0).split()))


def extract_claims(text: str, extra_names: tuple[str, ...] = ()) -> list[Claim]:
    forward, backward = _patterns(tuple(n.lower() for n in extra_names))
    found: list[tuple[int, Claim]] = []
    used: set[tuple[int, int]] = set()
    for m in forward.finditer(text):
        if (c := _claim(m, text)) is not None:
            used.add(m.span("digits"))
            found.append((m.start("digits"), c))
    for m in backward.finditer(text):
        if m.span("digits") in used:
            continue  # the number already belongs to the name in front of it
        if (c := _claim(m, text)) is not None:
            found.append((m.start("digits"), c))
    return [c for _, c in sorted(found, key=lambda t: t[0])]


def _close(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    diff = abs(a - b)
    return diff <= abs_tol + 1e-9 or diff <= rel_tol * abs(b) + 1e-9


def is_observed_metric(key: str) -> bool:
    """False for thresholds, limits, counts, sizes and null-model expectations (SR0)."""
    k = key.lower()
    if k.startswith(EXCLUDED_PREFIXES) or k.endswith("_count"):
        return False
    return not (set(k.split("_")) & EXCLUDED_WORDS)


def _has_phrase(key: str, phrase: str) -> bool:
    """Whole-word (underscore separated) occurrence of ``phrase`` in ``key``."""
    return re.search(r"(?:^|_)" + re.escape(phrase.lower()) + r"(?:_|$)", key.lower()) is not None


def family_keys(name: str, metric_map: Mapping[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    if metric_map and name in metric_map:
        return tuple(metric_map[name])
    return FAMILY_KEYS.get(name, (name.replace(" ", "_"),))


def claim_matches(
    claim: Claim, metrics: Mapping[str, float] | Iterable[tuple[str, float]], *, abs_tol: float, rel_tol: float,
    metric_map: Mapping[str, tuple[str, ...]] | None = None,
) -> bool:
    """Value match within tolerance against observed metrics of the claim's own family.

    Percent claims and percent-type metrics also try fraction scaling; a drawdown sign is ignored.
    """
    phrases = family_keys(claim.name, metric_map)
    candidates = [claim.value]
    if claim.percent or claim.name in PERCENT_TYPE:
        candidates.append(claim.value / 100.0)
    pairs = metrics.items() if isinstance(metrics, Mapping) else metrics
    for key, m in pairs:
        if not is_observed_metric(key) or not any(_has_phrase(key, ph) for ph in phrases):
            continue
        for v in candidates:
            if _close(v, m, abs_tol, rel_tol):
                return True
            if claim.name == "drawdown" and _close(abs(v), abs(m), abs_tol, rel_tol):
                return True
    return False


def _entry_text(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    msg = entry.get("message") if isinstance(entry.get("message"), dict) else entry
    if (msg.get("role") or entry.get("type")) != "assistant":
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(p for p in parts if isinstance(p, str) and p)
    return ""


def last_assistant_text(path: str | Path) -> str | None:
    """Text of the last assistant entry that has any (tool_use blocks skipped). Raises OSError if unreadable."""
    last: str | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            text = _entry_text(entry)
            if text.strip():
                last = text
    return last


def _metric_values(ctx: HookContext, text: str) -> list[tuple[str, float]]:
    """(key, value) of every metric in the selected receipts (receipt.metrics only, never params)."""
    conn = ctx.registry().conn
    rr = ctx.config.receipt_required
    tokens = [t.lower() for t in _HEX_TOKEN.findall(text)]
    all_ids = [r["trial_id"] for r in conn.execute("SELECT trial_id FROM hypotheses")]
    mentioned = {tid for tid in all_ids if any(tid.lower().startswith(t) for t in tokens)}
    cutoff = ctx.now() - timedelta(days=rr.recent_days)
    values: list[tuple[str, float]] = []
    for row in conn.execute("SELECT trial_id, path FROM receipts"):
        if mentioned and row["trial_id"] not in mentioned:
            continue
        try:
            receipt = load_receipt(row["path"])
        except ReceiptTamperedError:
            continue  # a receipt that fails its hash backs nothing
        if not mentioned:
            created = datetime.fromisoformat(receipt.created_at)
            created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
            if created < cutoff:
                continue
        values.extend(receipt.metrics.items())
    return values


def run(ctx: HookContext) -> HookResult:
    p = ctx.payload
    if p.event and p.event != "Stop":
        return HookResult("allow", "not a Stop event")
    rr = ctx.config.receipt_required
    if not rr.enabled:
        return HookResult("allow", "disabled in config")
    try:
        if not p.transcript_path:
            raise OSError("no transcript_path in the payload")
        text = last_assistant_text(p.transcript_path)
    except OSError as exc:
        return HookResult("error", f"transcript unreadable ({type(exc).__name__}); not blocking Stop")
    if not text:
        return HookResult("allow", "no assistant text")
    claims = extract_claims(text, rr.metric_patterns)
    if not claims:
        return HookResult("allow", "no performance claims")
    values = _metric_values(ctx, text)
    unmatched = [c for c in claims if not claim_matches(c, values, abs_tol=rr.abs_tol, rel_tol=rr.rel_tol, metric_map=rr.metric_map)]
    if not unmatched:
        return HookResult("allow", f"{len(claims)} claim(s) matched receipts")
    lines = [f"  - {c.name} {c.value:g}{'%' if c.percent else ''}  (\"{c.text}\")" for c in unmatched]
    return HookResult(
        "block", f"{len(unmatched)} claim(s) without a matching receipt",
        action_parts=("receipt-required", "unsupported performance claims:",
                      *sorted(f"{c.name} {c.value:g}{'%' if c.percent else ''}" for c in unmatched)),
        stderr=(
            "Performance numbers in your last message match no harness receipt:\n" + "\n".join(lines)
            + "\nFix: run the harness or remove the claim, then answer again (quote numbers from receipts only)."
        ),
    )

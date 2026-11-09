"""Changing which method version is the default: promotion, demotion and deprecation.

The harness never changes a default by itself. A change needs (1) evidence the code can check and (2) a human.

Promotion rules (``check_promotion``), all of which must hold:

1. The bench receipt was run with exactly one override, the candidate, and with ``--compare-to default``: it holds the
   candidate arm and the default arm over identical generated data, and the two arms differ in no other method.
2. The default arm used the id's *current* default version (a receipt measured against an older default is stale).
3. The receipt meets the ``full`` profile (replications and cells), and covers planted-bugs, noise-fpr and
   known-edge-power.
4. Every gated metric (detection rate per bug class, false-alarm rate on controls, false-positive rate per noise cell,
   power per planted-trend strength) is no worse for the candidate than for the default, by point estimate. One worse
   metric refuses the promotion; there is no tolerance, because both arms saw the same data.
5. A noise cell where the candidate's interval lies above the nominal level must already have done so for the default.
6. At least one gated metric is strictly better. If all are tied, a documented ``--tie-reason`` of at least
   ``MIN_TIE_REASON`` characters is required (for example a simpler or better-founded method with identical results).
7. The receipt was measured on the current harness code (its code hash equals the package's), so the result describes
   what would become the default.
8. A human approval (``qrh approve``) of this exact action, single use.

Demotion (the default goes back to experimental) must name the version that becomes the default again; deprecation
applies to experimental versions only. Both need the same human approval, and no bench receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qrh import approvals
from qrh.bench.profiles import PROFILES, SUITES
from qrh.bench.receipt import BenchReceipt, BenchReceiptError, resolve_receipt
from qrh.methods.registry import MethodError, parse_ref
from qrh.methods.selection import MethodSelector
from qrh.methods.state import load_state, save_state

ACTIONS = ("promote", "demote", "deprecate")
HOOK = "methods"
MIN_REASON = 10
MIN_TIE_REASON = 30
_EPS = 1e-12


@dataclass
class PromotionDecision:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    better: list[str] = field(default_factory=list)
    worse: list[str] = field(default_factory=list)
    ties: int = 0
    tie_required: bool = False


@dataclass
class ActionOutcome:
    status: str  # pending | applied | refused
    message: str
    reasons: list[str] = field(default_factory=list)
    approval_id: str | None = None


def meets_full_profile(config: dict[str, Any]) -> bool:
    """True if a bench configuration is at least as thorough as the ``full`` profile."""
    full = PROFILES["full"]
    try:
        for suite, want in full.items():
            have = config[suite]
            for key, w in want.items():
                h = have[key]
                if isinstance(w, list):
                    if not set(w) <= set(h):
                        return False
                elif h < w:
                    return False
        return True
    except (KeyError, TypeError):
        return False


def _gated(results: dict[str, Any], suite: str) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in (results.get(suite, {}).get("metrics") or {}).items() if v.get("gate")}


def check_promotion(
    sel: MethodSelector, receipt: BenchReceipt, method_id: str, version: str, tie_reason: str | None = None,
    code_hash: str | None = None,
) -> PromotionDecision:
    d = PromotionDecision(ok=False)
    why = d.reasons
    try:
        method = sel.registry.get(method_id, version)
    except MethodError as exc:
        why.append(str(exc))
        return d
    if sel.status(method) != "experimental":
        why.append(f"{method_id}={version} is {sel.status(method)}; only an experimental version can be promoted")
    if receipt.baseline_results is None or receipt.baseline_methods is None:
        why.append("the bench receipt has no default arm (run `qrh bench run --method ... --compare-to default`)")
        return d
    if receipt.overrides != {method_id: version}:
        why.append(f"the bench receipt must override exactly {method_id}={version}, it overrides {receipt.overrides or 'nothing'}")
    others = sorted(i for i, v in receipt.methods.items() if i != method_id and receipt.baseline_methods.get(i) != v)
    if others:
        why.append(f"the two arms also differ in other methods: {', '.join(others)}")
    try:
        current = sel.default_version(method_id)  # None for an opt-in id: the default arm must not select it
    except MethodError as exc:
        why.append(str(exc))
        current = None
    else:
        if receipt.baseline_methods.get(method_id) != current:
            why.append(
                f"the default arm used {receipt.baseline_methods.get(method_id)}, but the current default is {current}; re-run the bench"
            )
    if code_hash is None:
        from qrh.pipeline.stages import package_code_hash

        code_hash = package_code_hash()
    if receipt.code_hash != code_hash:
        why.append("the bench receipt was measured on different harness code than the current code; re-run the bench")
    if not meets_full_profile(receipt.config):
        why.append("the bench receipt does not meet the full profile (replications and cells); run `qrh bench run --full`")
    for suite in SUITES:
        if suite not in receipt.suites or suite not in receipt.results or suite not in receipt.baseline_results:
            why.append(f"the bench receipt does not cover suite {suite}")
            continue
        cand, base = _gated(receipt.results, suite), _gated(receipt.baseline_results, suite)
        if set(cand) != set(base) or not cand:
            why.append(f"{suite}: the two arms do not have the same gated metrics (cells)")
            continue
        for name in sorted(cand):
            c, b = cand[name], base[name]
            label = f"{suite}: {name}"
            if c["rate"] is None or b["rate"] is None:
                why.append(f"{label} has no data")
            elif abs(c["rate"] - b["rate"]) <= _EPS:
                d.ties += 1
            elif (c["rate"] > b["rate"]) == (c["direction"] == "higher"):
                d.better.append(label)
            else:
                d.worse.append(label)
        if suite == "noise-fpr":
            cc = receipt.results[suite].get("cells", {})
            bc = receipt.baseline_results[suite].get("cells", {})
            for key, info in cc.items():
                if info.get("exceeds_nominal") and not bc.get(key, {}).get("exceeds_nominal"):
                    why.append(f"noise-fpr: {key}: the candidate's false-positive interval is above the nominal level, the default's is not")
    if d.worse:
        why.append(f"{len(d.worse)} gated metric(s) are worse than the default: {'; '.join(d.worse)}")
    if not d.better and not d.worse:
        d.tie_required = True
        if len((tie_reason or "").strip()) < MIN_TIE_REASON:
            why.append(
                f"no gated metric is better than the default (all {d.ties} tie); a documented --tie-reason of at least "
                f"{MIN_TIE_REASON} characters is required"
            )
    d.ok = not why
    return d


def _fail(msg: str, reasons: list[str] | None = None) -> ActionOutcome:
    return ActionOutcome("refused", msg, reasons or [msg])


def _apply(home: Path, state: dict[str, Any], changes: dict[tuple[str, str], str], entry: dict[str, Any]) -> None:
    for (mid, version), status in changes.items():
        state["statuses"].setdefault(mid, {})[version] = status
    state["history"].append(entry)
    save_state(home, state)


def run_action(
    home: str | Path,
    action: str,
    ref: str,
    *,
    reason: str,
    bench: str | None = None,
    tie_reason: str | None = None,
    restore: str | None = None,
    now: datetime | None = None,
    session_id: str = "cli",
    code_hash: str | None = None,
) -> ActionOutcome:
    """Check an action, then wait for (or consume) the human approval of exactly this action, then apply it."""
    home = Path(home)
    now = now or datetime.now(timezone.utc)
    if action not in ACTIONS:
        return _fail(f"unknown action {action!r}; use one of {', '.join(ACTIONS)}")
    if len((reason or "").strip()) < MIN_REASON:
        return _fail(f"a reason of at least {MIN_REASON} characters is required")
    try:
        method_id, version = parse_ref(ref)
        sel = MethodSelector(home)
        method = sel.registry.get(method_id, version)
        state = load_state(home)
        status = sel.status(method)
    except MethodError as exc:
        return _fail(str(exc))

    changes: dict[tuple[str, str], str] = {}
    receipt: BenchReceipt | None = None
    previous = None
    restore_version = None
    if action == "promote":
        if not bench:
            return _fail("promotion needs --bench <receipt>")
        try:
            receipt = resolve_receipt(home, bench)
        except BenchReceiptError as exc:
            return _fail(str(exc))
        decision = check_promotion(sel, receipt, method_id, version, tie_reason, code_hash)
        if not decision.ok:
            return _fail(f"promotion of {method_id}={version} refused", decision.reasons)
        previous = sel.default_version(method_id)  # None for an opt-in id
        changes = {(method_id, version): "default"}
        if previous is not None:
            changes[(method_id, previous)] = "deprecated"
    elif action == "deprecate":
        if status != "experimental":
            return _fail(f"{method_id}={version} is {status}; only an experimental version can be deprecated")
        changes = {(method_id, version): "deprecated"}
    else:  # demote
        if status != "default":
            return _fail(f"{method_id}={version} is {status}; only the default can be demoted")
        if not restore:
            return _fail("demotion needs --restore <id=version>: the version that becomes the default again")
        try:
            r_id, restore_version = parse_ref(restore)
            target = sel.registry.get(r_id, restore_version)
        except MethodError as exc:
            return _fail(str(exc))
        if r_id != method_id or sel.status(target) == "default":
            return _fail(f"--restore must name another version of {method_id}")
        changes = {(method_id, version): "experimental", (method_id, restore_version): "default"}

    act = approvals.parts_action(
        HOOK, action, f"{method_id}={version}", restore or "", receipt.receipt_hash() if receipt else "",
        reason.strip(), (tie_reason or "").strip(),
    )
    try:
        got = approvals.consume(home, hook=HOOK, fingerprint=approvals.fingerprint_of(act), session_id=session_id,
                                now=now)
        if not got.ok:
            from qrh.hooks.runner import human_block_text

            rid = approvals.request(home, hook=HOOK, action=act, session_id=session_id, now=now,
                                    reason=reason.strip())
            text = human_block_text(
                HOOK, f"{action} {method_id}={version}" + (f" (restore {restore})" if restore else ""),
                f"{action} {method_id}={version} is eligible and waits for a human decision. Reason given: "
                f"{reason.strip()}",
                "changing the default method changes how every later trial is validated",
                f'qrh approve {rid} --reason "<why this is acceptable, at least {approvals.MIN_REASON} characters>"',
                f"Approval id {rid}: valid once, within {approvals.DEFAULT_TTL_S // 60} minutes; then repeat this "
                "exact command.", " ".join(got.notes))
            return ActionOutcome("pending", text, [], rid)
    except approvals.ApprovalError as exc:
        return _fail(str(exc))
    assert got.record is not None
    entry = {
        "at": now.isoformat(), "action": action, "method": f"{method_id}={version}", "reason": reason.strip(),
        "tie_reason": (tie_reason or "").strip() or None, "approval_id": got.record["id"],
        "approval_reason": got.record.get("approval_reason"), "previous_default": previous,
        "restore": f"{method_id}={restore_version}" if restore_version else None,
        "bench_id": receipt.bench_id() if receipt else None,
        "bench_receipt_hash": receipt.receipt_hash() if receipt else None,
    }
    _apply(home, state, changes, entry)
    return ActionOutcome("applied", f"{action} {method_id}={version} applied (local to {home})", [], got.record["id"])

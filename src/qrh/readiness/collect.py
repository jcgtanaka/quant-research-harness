"""Gather every warning of a trial's workflow from verified receipts, stage artifacts, the registry and the
interview state, so that one document (``qrh readiness report``) lists all of them.

Read-only: nothing here writes anything or changes a verdict. Everything comes from hash-verified receipts and
artifacts, plus the live registry (answers, overrides, dataset catalog) so that a change made after the last run
shows up. A source that cannot be read becomes a HIGH warning instead of being skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qrh.pipeline.artifacts import ArtifactError, read_artifact
from qrh.pipeline.receipt import Receipt, Stage, display_verdict, is_pass, load_receipt, receipt_path
from qrh.readiness.model import (GENERAL, KNOWN_LIMITS, SEVERITIES, STAGE_ORDER, ReadinessWarning, clip,
                                 make_warning)

if TYPE_CHECKING:
    from qrh.memory.registry import Registry

TRIAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
NOT_RUN = "NOT RUN"
METHODS_DOC = "docs/methods.md"


class ReadinessError(ValueError):
    """The readiness of a trial cannot be computed (the message says why)."""


@dataclass
class Collected:
    trial_id: str
    verdict: str  # a Verdict value, or "NOT RUN" when S7 has no receipt
    failing_stage: str | None
    warnings: list[ReadinessWarning]
    receipt_hashes: dict[str, str] = field(default_factory=dict)  # stage -> receipt hash
    two_track: dict[str, Any] | None = None  # information section; None when the trial has no answer file

    @property
    def verdict_display(self) -> str:
        return display_verdict(self.verdict)

    def counts(self) -> dict[str, dict[str, int]]:
        """Warnings per stage and severity (every stage listed, so an empty one reads as zero)."""
        out = {s: {sev: 0 for sev in SEVERITIES} for s in STAGE_ORDER if not s.startswith("B")}
        for w in self.warnings:
            out.setdefault(w.stage, {sev: 0 for sev in SEVERITIES})[w.severity] += 1
        return out

    def totals(self) -> dict[str, int]:
        return {sev: sum(1 for w in self.warnings if w.severity == sev) for sev in SEVERITIES}

    def high(self) -> list[ReadinessWarning]:
        return [w for w in self.warnings if w.severity == "high"]


def check_trial_id(trial_id: str) -> str:
    """A trial id is used as a folder name: letters, digits, ``_``, ``.`` and ``-`` only, never leading with a dot."""
    if not isinstance(trial_id, str) or not TRIAL_ID_RE.match(trial_id) or ".." in trial_id:
        raise ReadinessError(f"{trial_id!r} is not a valid trial id")
    return trial_id


class _Ctx:
    def __init__(self, home: Path, trial: dict[str, Any], registry: "Registry"):
        self.home, self.trial, self.reg = home, trial, registry
        self.tid: str = trial["trial_id"]
        self.receipts: dict[str, Receipt] = {}
        self.out: list[ReadinessWarning] = []

    def add(self, stage: str, severity: str, kind: str, code: str, scope: str, message: str, resolution: str) -> None:
        r = self.receipts.get(stage)
        self.out.append(make_warning(
            stage=stage, severity=severity, kind=kind, code=code, scope=scope, message=message, resolution=resolution,
            receipt_hash=r.receipt_hash() if r else None,
            receipt_path=str(receipt_path(self.home, self.tid, Stage(stage))) if r and stage in Stage._value2member_map_ else None))


def _load_receipts(c: _Ctx) -> None:
    for stage in Stage:
        path = receipt_path(c.home, c.tid, stage)
        if not path.is_file():
            continue
        try:
            r = load_receipt(path)
            if r.trial_id != c.tid or r.stage is not stage:
                raise ValueError(f"belongs to {r.trial_id}/{r.stage.value}")
        except Exception as exc:
            c.add(GENERAL, "high", "receipt", "receipt-unreadable", stage.value,
                  f"the {stage.value} receipt cannot be verified: {clip(exc, 160)}",
                  f"re-run the trial: `qrh run {c.tid}` (the receipt was altered or damaged)")
            continue
        entry = c.reg.get_receipt_entry(c.tid, stage.value)
        if entry is not None and entry["receipt_hash"] != r.receipt_hash():
            c.add(GENERAL, "high", "receipt", "receipt-unreadable", f"{stage.value}-registry",
                  f"the {stage.value} receipt does not match the registry index",
                  f"re-run the trial: `qrh run {c.tid}`")
            continue
        c.receipts[stage.value] = r


def _verdict(c: _Ctx) -> tuple[str, str | None]:
    s7 = c.receipts.get("S7")
    if s7 is None:
        c.add("S7", "high", "receipt", "no-final-verdict", "S7",
              "no final verdict: S7 has no verified receipt for this trial",
              f"run the whole workflow: `qrh run {c.tid}`")
        return NOT_RUN, None
    failing = s7.params.get("failing_stage")
    if not is_pass(s7.verdict):
        c.add(str(failing) if failing in Stage._value2member_map_ else "S7", "high", "receipt", "final-not-pass", "S7",
              f"the final verdict is {display_verdict(s7.verdict)}: this strategy is not deployable as it stands"
              + (f" (first non-PASS stage {failing})" if failing else ""),
              f"fix the cause named in the {failing or 'S7'} receipt, register a new trial and run it: `qrh new <spec>`")
    return s7.verdict.value, (str(failing) if failing else None)


# --- (a) interview and data flags -------------------------------------------------------------------------------

def _next_cmd(c: _Ctx, stage: str, scope: str, unknown: bool = False) -> str:
    return f"qrh interview next --stage {stage} --scope {scope} --trial {c.tid}" + (" --include-unknown" if unknown else "")


def _interview(c: _Ctx) -> None:
    from qrh.interview.engine import dataset_flags
    from qrh.interview.snapshot import build_snapshot
    from qrh.packs.model import PackError
    from qrh.packs.runtime import PackView

    try:
        view = PackView.load(c.home)
    except PackError as exc:
        c.add(GENERAL, "high", "interview", "packs-unloadable", "packs",
              f"the active experience packs cannot be loaded, so the interview state is unknown: {clip(exc, 160)}",
              "fix or remove the pack: `qrh pack list`")
        return
    if not view:
        return
    snap = build_snapshot(c.reg.conn, view, c.trial)
    for stage, body in snap["stages"].items():
        for r in body["rows"]:
            if r.get("origin") == "onboarding":
                continue  # reported by _onboarding, with the dataset's other onboarding gaps
            if r.get("origin") == "stability":
                continue  # reported as one warning by _stability_thresholds
            scope = f"{r['qid']}@{r['scope']}"
            crit = bool(r["critical"])
            tier = "critical" if crit else "recommended"
            sev = "high" if crit else "medium"
            question = clip(r["text"], 140)
            if r["status"] == "pending" and r.get("stale"):
                c.add(stage, sev, "interview", "stale-answer", scope,
                      f"the answer to {r['qid']} ({r['scope']}) is stale: that scope changed since answer "
                      f"v{r.get('previous_version')} was given. {question}",
                      f"answer it again: `{_next_cmd(c, stage, r['scope'])}`, then `qrh run {c.tid}`")
            elif r["status"] == "pending":
                c.add(stage, sev, "interview", f"pending-{tier}", scope,
                      f"{tier} question {r['qid']} ({r['scope']}) is unanswered: {question}",
                      f"ask the user and record their answer: `{_next_cmd(c, stage, r['scope'])}`, then `qrh run {c.tid}`")
            elif r["status"] == "unknown":
                c.add(stage, sev, "interview", f"unknown-{tier}", scope,
                      f"{tier} question {r['qid']} ({r['scope']}) was answered \"don't know\": {question} "
                      f"Risk: {r.get('unknown_risk') or 'not stated'}",
                      f"find out and answer it: `{_next_cmd(c, stage, r['scope'], unknown=True)}`, then `qrh run {c.tid}`")
            elif r.get("risk"):
                c.add(stage, "high", "interview", "risk-tag", scope,
                      f"the answer to {r['qid']} ({r['scope']}) carries the risk tag '{r['risk']}': "
                      f"{clip(r.get('label') or r.get('answer'), 100)}",
                      "a risk tag stays until the underlying data is fixed: replace the data (`qrh data register ...`), "
                      f"answer again, then `qrh run {c.tid}`")
    for ds in c.trial.get("datasets", []):
        for flag in dataset_flags(c.reg.conn, view, ds["name"]):
            c.add("S1", "high", "data", "dataset-flag", f"{ds['name']}:{flag}",
                  f"dataset {ds['name']} carries the flag '{flag}' from an interview answer",
                  f"fix the data so the answer no longer applies, register it again and run `qrh run {c.tid}`")
    s7 = c.receipts.get("S7")
    if s7 is not None and is_pass(s7.verdict):
        key = lambda x: (x.get("question_id"), x.get("scope"), x.get("status"))  # noqa: E731
        live = sorted(map(key, snap["conditions"]), key=str)
        then = sorted(map(key, s7.params.get("conditions") or []), key=str)
        if live != then:
            c.add("S7", "high", "interview", "stale-verdict", "S7",
                  "the final verdict is older than the interview answers: the open conditions changed since S7 ran",
                  f"run the trial again so the verdict reflects the answers: `qrh run {c.tid}`")


def _stability_thresholds(c: _Ctx) -> None:
    """The four stability thresholds are always asked of the trader. Any that is unanswered, answered \"I don't know\", or
    answered with a value other than the one frozen in the hypothesis was not confirmed by the trader: the default (or
    the frozen value) applies, and this says so."""
    from qrh.interview.stability import STABILITY_NAMES, confirmation
    from qrh.validation.defaults import DEFAULT_THRESHOLDS

    family = c.trial["family_id"]
    frozen = {n: (c.trial.get("thresholds") or {}).get(n, DEFAULT_THRESHOLDS[n]) for n in STABILITY_NAMES}
    s0 = c.receipts.get("S0")
    recorded = (s0.params.get("stability_thresholds") if s0 else None) or {}
    defaults = {n: recorded[n]["default"] for n in STABILITY_NAMES if recorded.get(n, {}).get("default") is not None}
    parts = []
    for row in confirmation(c.reg.conn, family, frozen, {**DEFAULT_THRESHOLDS, **defaults}):
        if row["confirmed"]:
            continue
        if row["status"] == "pending":
            parts.append(f"{row['qid']} (unanswered)")
        elif row["status"] == "unknown":
            parts.append(f"{row['qid']} (answered \"don't know\")")
        else:
            parts.append(f"{row['qid']} (the answer {row['answered_value']} differs from the frozen {row['frozen_value']})")
    if not parts:
        return
    c.add("S5", "medium", "interview", "stability-unconfirmed", "stability",
          "stability thresholds not confirmed by the trader: " + ", ".join(parts) + ". The frozen values apply (the defaults "
          "where the trader chose nothing), so the spike and plateau checks may be looser or stricter than the trader wants",
          f"ask the trader and record their own answers: `qrh interview next --stage S0 --scope strategy:{family}`; the "
          "thresholds are frozen with the hypothesis, so register a new trial to apply them: `qrh new <spec>`")


def _stability_declared(c: _Ctx) -> None:
    """Neighbour values and bounds written into the hypothesis replace the generated ones, so the trader sees them."""
    decl = c.trial.get("stability") or {}
    if not decl:
        return
    parts = [f"{k}: {v}" for k, v in sorted((decl.get("neighbours") or {}).items())]
    bounds = [f"{k}: {list(v)}" for k, v in sorted((decl.get("bounds") or {}).items())]
    c.add("S5", "medium", "exemption", "stability-declared", "stability",
          "the hypothesis declares its own stability neighbours" + (" " + "; ".join(parts) if parts else "")
          + ((" and bounds " + "; ".join(bounds)) if bounds else "")
          + ": the S5 stability check used these instead of generating step-distance neighbours",
          "the trader should confirm they are real alternatives to the chosen setting (none is closer than the smallest "
          "stability step); to change them, register a new trial")


def _stability_check(c: _Ctx) -> None:
    s5 = c.receipts.get("S5")
    ps = (s5.params.get("param_stability") if s5 else None) or {}
    if not ps:
        return
    if ps.get("status") == "NOT_APPLICABLE":
        c.add("S5", "medium", "exemption", "stability-not-applicable", "s5.param_stability",
              "the parameter stability check did not conclude: " + clip("; ".join(ps.get("reasons") or ["no reason recorded"]), 220),
              f"the check cannot see a sharp peak here; judge the choice of parameters yourself (see {METHODS_DOC}), or declare "
              "neighbour values in the hypothesis `stability` block")
    unchecked = ps.get("unchecked") or []
    if unchecked:
        dims = sorted({str(u.get("dimension")) for u in unchecked})
        c.add("S5", "medium", "exemption", "stability-unchecked", "s5.param_stability:unchecked",
              f"parameter(s) {', '.join(dims)} could not be nudged in some or all directions, so their stability is unchecked: "
              + clip("; ".join(f"{u.get('dimension')}: {u.get('reason')}" for u in unchecked), 200),
              "declare neighbour values or bounds for them in the hypothesis `stability` block (a new trial)")
    if ps.get("skipped"):
        c.add("S5", "low", "exemption", "stability-skipped", "s5.param_stability:skipped",
              f"non-numeric parameter(s) {', '.join(map(str, ps['skipped']))} are not varied by the stability check",
              "no action needed; know the limit")


def _conditions(c: _Ctx) -> None:
    s7 = c.receipts.get("S7")
    for cond in (s7.params.get("conditions") if s7 else None) or []:
        qid, scope = str(cond.get("question_id")), str(cond.get("scope"))
        c.add("S7", "high", "condition", "open-condition", f"{qid}@{scope}",
              f"open condition on the final PASS · CONDITIONAL: {qid} ({cond.get('status')}, {cond.get('stage')}): "
              f"{clip(cond.get('text', ''), 140)}" + (f" Risk: {clip(cond['risk'], 140)}" if cond.get("risk") else ""),
              f"answer it cleanly ({_next_cmd(c, str(cond.get('stage')), scope, unknown=True)}) and run `qrh run {c.tid}`")


# --- (b) data -------------------------------------------------------------------------------------------------

def _data(c: _Ctx) -> None:
    s1 = c.receipts.get("S1")
    if s1 is None:
        return
    try:
        coverage = read_artifact(c.home, s1, "coverage")
    except ArtifactError as exc:
        c.add(GENERAL, "high", "receipt", "artifact-unreadable", "S1:coverage",
              f"the S1 coverage artifact cannot be verified: {clip(exc, 160)}", f"re-run the trial: `qrh run {c.tid}`")
        coverage = {"datasets": []}
    for ds in coverage.get("datasets", []):
        name = str(ds.get("name"))
        if (ds.get("gap_count") or 0) > 0:
            c.add("S1", "medium", "data", "gaps", name, f"dataset {name} has {ds['gap_count']} gap(s) in its coverage",
                  f"check the gaps (S1 coverage page, `qrh report {c.tid} --stage S1`); add a `gaps` rule to its rules file "
                  "if gaps matter")
        for rule in ds.get("rules", []):
            label = f"{name}:{rule.get('rule')}"
            if str(rule.get("status", rule.get("severity", ""))).upper() == "WARN":
                c.add("S1", "medium", "data", "quality-warn", label,
                      f"data-quality rule {rule.get('rule')} on {name} reported WARN",
                      "read the rule's offenders in the S1 report; fix the data or tighten the rule")
            elif rule.get("passed") is False:
                c.add("S1", "medium", "data", "rule-failed", label,
                      f"data-quality rule {rule.get('rule')} on {name} did not pass: "
                      f"{clip('; '.join(map(str, rule.get('reasons') or [])), 160) or 'no reason recorded'}",
                      "fix the data and register it again: `qrh data register ...`")
    for ds in c.trial.get("datasets", []):
        row = c.reg.conn.execute("SELECT content_hash, status FROM datasets WHERE name=?", (ds["name"],)).fetchone()
        if row is None:
            continue
        seen = s1.data_hashes.get(ds["name"])
        if seen is not None and row["content_hash"] != seen:
            c.add("S1", "high", "data", "dataset-changed", ds["name"],
                  f"dataset {ds['name']} changed after S1 checked it (catalog hash differs from the S1 receipt)",
                  f"re-run the trial on the current data: `qrh run {c.tid}`")
        if row["status"] != "PASS":
            c.add("S1", "high", "data", "dataset-status", ds["name"],
                  f"dataset {ds['name']} is {row['status']} in the catalog now",
                  f"re-check it: `qrh data check {ds['name']}`, then `qrh run {c.tid}`")


# --- (b2) dataset onboarding ----------------------------------------------------------------------------------

def _onboarding(c: _Ctx) -> None:
    from qrh.onboarding import store
    from qrh.interview.engine import latest_answer
    from qrh.onboarding.questions import answer_identity, qid_for

    for ds in c.trial.get("datasets", []):
        name = ds["name"]
        st = store.dataset_state(c.reg.conn, name)
        ask = f"qrh interview next --stage S1 --scope dataset:{name} --trial {c.tid}"
        if st["state"] == "NOT_ONBOARDED":
            c.add("S1", "high", "onboarding", "not-onboarded", name,
                  f"dataset {name} is not onboarded: nobody has checked what its columns, units, adjustments and "
                  "timestamps mean",
                  f"profile it, have the agent propose an instrument model, then verify: `qrh data profile {name}`, "
                  f"`qrh data propose {name} --file proposal.json`, `qrh data verify {name}`; then `qrh run {c.tid}`")
            continue
        v = store.current_version(c.reg.conn, name)
        assert v is not None
        rows = store.field_rows(v["model"])
        if st["state"] == "CONTRADICTED":
            bad = [r["field"] for r in rows if r["status"] == "contradicted"]
            c.add("S1", "high", "onboarding", "contradicted", name,
                  f"the data contradicts the claimed {', '.join(bad)} of dataset {name}; the dataset is BLOCKED",
                  f"read `qrh data onboarding {name}`, correct the claim (`qrh data propose {name} --file ...`) or fix "
                  f"the data, then `qrh data verify {name}` and `qrh run {c.tid}`")
        for r in rows:
            if r["status"] in ("verified", "confirmed", "contradicted"):
                continue
            critical = bool(r["critical"])
            qid = qid_for(r["field"])
            answered = False
            if v["verified_at"] is not None:
                ans = latest_answer(c.reg.conn, qid, f"dataset:{name}")
                answered = bool(ans and ans["identity"] == answer_identity(c.reg.conn, qid, f"dataset:{name}")
                                and ans["answer"] == "unknown")
            how = (f"ask the user and record their answer: `{ask}` (question {qid}), then `qrh run {c.tid}`"
                   if v["verified_at"] is not None else f"verify the claims first: `qrh data verify {name}`")
            c.add("S1", "high" if critical else "medium", "onboarding",
                  "unverified-critical" if critical else "unverified-recommended", f"{name}:{r['field']}",
                  f"{'critical' if critical else 'recommended'} field {r['field']} of dataset {name} is unverified "
                  f"(claim: {clip(r['claim'], 80)}; {clip(r['evidence'], 120)})" + (" The user answered \"don't know\"." if answered else ""),
                  how)
        stored = store.get_profile(c.reg.conn, v["content_hash"])
        settled = {r["field"] for r in rows if r["status"] in ("verified", "confirmed")}
        if stored is not None and "adjustments" not in settled:
            cols = [h["column"] for h in stored[0]["hints"] if h["hint"] == "adjustments"]
            if cols:
                c.add("S1", "medium", "onboarding", "adjustments-hint", name,
                      f"column name(s) {', '.join(cols)} of dataset {name} hint at adjusted prices or corporate actions "
                      "(a name hint, not a fact)",
                      f"state how the prices are adjusted and have the user confirm it: `{ask}`")


# --- (c) lint, (d) exemptions --------------------------------------------------------------------------------------

def _lint(c: _Ctx) -> None:
    s2 = c.receipts.get("S2")
    lint = (s2.params.get("lint") if s2 else None) or {}
    for f in lint.get("warnings") or []:
        if f.get("rule") == "LLM002":
            continue  # one warning for all of them, from _llm
        c.add("S2", "medium", "lint", "lint-warn", f"{f.get('rule')}@{f.get('at')}",
              f"lint warning {f.get('rule')} at {f.get('at')}: {clip(f.get('snippet', ''), 120)}",
              f"review {f.get('at')}; fix it, or suppress it after review with `# qrh: allow {f.get('rule')} <reason>`")
    for f in lint.get("suppressed") or []:
        if f.get("rule") == "LLM001":
            continue  # reported on its own, by _llm
        c.add("S2", "medium", "lint", "lint-suppressed", f"{f.get('rule')}@{f.get('at')}",
              f"lint finding {f.get('rule')} at {f.get('at')} was suppressed with `# qrh: allow`; reason: "
              f"{clip(f.get('reason') or 'no reason recorded', 160)}",
              f"re-read the reason; remove the `# qrh: allow {f.get('rule')}` comment if it no longer holds")


# --- (c2) language models -----------------------------------------------------------------------------------------

MARGIN_WARN_BELOW = 365


def _llm(c: _Ctx) -> None:
    from qrh.llm.cache import LLMCacheCorrupt, ReplayCache, cache_dir

    s2 = c.receipts.get("S2")
    for f in ((s2.params.get("lint") or {}).get("suppressed") or []) if s2 else []:
        if f.get("rule") == "LLM001":
            c.add("S2", "medium", "llm", "llm-undeclared-allowed", f"LLM001@{f.get('at')}",
                  f"`# qrh: allow LLM001` at {f.get('at')}: the adapter appears to use an LLM but the hypothesis "
                  f"declares none; reason given: {clip(f.get('reason') or 'no reason recorded', 160)}",
                  "declare the model in the hypothesis `llm` section (a new trial), or remove the comment if it no longer holds")
    llm = c.trial.get("llm")
    if not llm:
        return
    direct = [f for f in ((s2.params.get("lint") or {}).get("warnings") or []) if f.get("rule") == "LLM002"] if s2 else []
    if direct:
        files = sorted({str(f.get("at", "")).rsplit(":", 1)[0] for f in direct})
        c.add("S2", "medium", "llm", "llm-direct-client", "LLM002",
              "provider client used outside qrh.llm.complete in " + ", ".join(files) + ": the replay cache and the "
              "truncation prompt detector cannot see these calls (best-effort lint: plain HTTP and dynamic imports evade it)",
              "route every model call through qrh.llm.complete and register the provider with register_backend in "
              "its own file; or suppress with `# qrh: allow LLM002 <reason>` after review")
    margin = (c.trial.get("thresholds") or {}).get("llm_cutoff_margin_days")
    if isinstance(margin, (int, float)) and not isinstance(margin, bool) and margin < MARGIN_WARN_BELOW:
        c.add("S2", "medium", "llm", "llm-short-margin", "llm_cutoff_margin_days",
              f"the cutoff margin is {margin:g} days, below {MARGIN_WARN_BELOW}: a model can know events a year or "
              "more after its stated cutoff",
              "raise llm_cutoff_margin_days in a new hypothesis, or start the evaluation windows later")
    retrieval = llm.get("live_retrieval")
    if retrieval is not False and isinstance(retrieval, dict) and retrieval.get("point_in_time_evidence"):
        c.add("S2", "high", "llm", "llm-live-retrieval", "live_retrieval",
              "the strategy uses live retrieval or tools; the claim that they return only data available at each "
              f"decision time is unverified: \"{clip(retrieval['point_in_time_evidence'], 200)}\"",
              "check the claim against the tool's data (snapshots keyed by publication time); the harness cannot")
    try:
        status = ReplayCache(cache_dir(c.home, c.tid)).status()
    except LLMCacheCorrupt as exc:
        c.add("S3", "high", "llm", "llm-cache-unreadable", "llm_cache",
              f"the LLM replay cache does not verify: {clip(exc, 160)}",
              f"run `qrh llm verify {c.tid}`, then record again: `qrh run {c.tid} --llm-mode record`")
        return
    if status["entries"]:
        c.add("S3", "low", "llm", "llm-recorded-cache", "llm_cache",
              f"results depend on a recorded cache: {status['entries']} recorded LLM calls, cache hash "
              f"{status['cache_hash']} (replayed answers, not recomputed)",
              f"`qrh llm status {c.tid}` shows the cache; `qrh llm verify {c.tid}` re-checks every entry")


def _reconstruction_missing(c: _Ctx) -> bool:
    s2 = c.receipts.get("S2")
    if s2 is None or "s2.pnl_reconstruction" in s2.methods or "pnl_reconstruction" in s2.params:
        return False
    ran = (s2.params.get("same_bar_execution") or {}).get("verdict") in ("PASS", "FAIL")  # the default check concluded
    return not ran


def _perturbation_coverage(c: _Ctx) -> None:
    s2 = c.receipts.get("S2")
    p = (s2.params.get("perturbation") if s2 else None) or {}
    cov, tested, total = p.get("coverage"), p.get("change_bars_tested"), p.get("change_bars_total")
    if isinstance(cov, (int, float)) and cov < 1.0:
        c.add("S2", "medium", "exemption", "perturbation-coverage", "s2.future_perturbation",
              f"the future-perturbation test covered {tested} of {total} position-change bars ({cov:.0%}); a leak that is "
              "active only on the other change bars would not have been seen",
              "raise `perturbation_change_cap` in a new hypothesis (more adapter runs) or accept the residual risk; "
              f"see {METHODS_DOC}")


def _exemptions(c: _Ctx) -> None:
    _perturbation_coverage(c)
    if not _reconstruction_missing(c):
        return
    why = ("execution is declared but s2.pnl_reconstruction was not selected" if c.trial.get("execution")
           else "s2.pnl_reconstruction was not selected by this hypothesis")
    c.add("S2", "medium", "exemption", "no-reconstruction", "s2.pnl_reconstruction",
          f"No independent returns reconstruction concluded: {why}",
          "register a new trial that declares `execution: {lag_bars: 1, price_column: <column>}` and "
          f"`methods: {{s2.pnl_reconstruction: pnl_reconstruction@1}}` (`qrh new <spec>`); see {METHODS_DOC}")


# --- (e) overrides, (f) methods, (g) packs -----------------------------------------------------------------------

def _overrides(c: _Ctx) -> None:
    stage_of_receipt: dict[str, str] = {}
    for stage, r in sorted(c.receipts.items()):
        if r.override is not None:
            stage_of_receipt[stage] = r.override.reason
            c.add(stage, "medium", "override", "stage-override", f"{stage}:receipt",
                  f"override recorded in the {stage} receipt by {r.override.who}: {clip(r.override.reason, 200)}",
                  "an override cannot be undone; read the reason, and re-run without it if it no longer holds")
    rows = list(c.reg.list_overrides(c.tid))
    seen = {r["id"] for r in rows}
    rows += [r for r in c.reg.list_hook_overrides(c.trial["created_at"]) if r["trial_id"] is None and r["id"] not in seen]
    for r in rows:
        when = str(r["created_at"])[:10]
        if r["hook"]:
            c.add(r["stage"] if r["stage"] in Stage._value2member_map_ else GENERAL, "medium", "override", "hook-override",
                  f"override#{r['id']}",
                  f"hook {r['hook']} was approved by a human on {when}"
                  + ("" if r["trial_id"] else " (session-wide, since this trial was registered)")
                  + f": {clip(r['reason'], 200)}",
                  "read the recorded reason; the approval was single-use and cannot be revoked")
        elif stage_of_receipt.get(str(r["stage"])) != r["reason"]:
            c.add(r["stage"] if r["stage"] in Stage._value2member_map_ else GENERAL, "medium", "override", "stage-override",
                  f"override#{r['id']}",
                  f"override on {when} at {r['stage']} by {r['who'] or 'unknown'}: {clip(r['reason'], 200)}",
                  "an override cannot be undone; read the reason, and re-run without it if it no longer holds")


def _methods(c: _Ctx) -> None:
    seen: dict[str, str] = {}
    for stage in sorted(c.receipts):
        for mid, version in sorted((c.receipts[stage].params.get("non_default_methods") or {}).items()):
            if mid not in seen:
                seen[mid] = stage
                c.add(stage, "medium", "method", "non-default-method", f"{mid}={version}",
                      f"experimental or non-default method {mid}={version} was used (not the default of this home)",
                      f"see `qrh methods show {mid}`; a human promotes a method only after a benchmark "
                      "(`qrh methods promote`), otherwise accept the risk knowingly")


def _packs(c: _Ctx) -> None:
    layering = (c.receipts["S0"].params.get("pack_layering") if "S0" in c.receipts else None) or {}
    for lid in layering.get("approved_loosening") or []:
        c.add("S0", "high", "pack", "loosened-gate", str(lid),
              f"a pack loosened a core gate with a human approval: {lid}",
              "keep it only if intended; to restore the core threshold remove it from the pack's thresholds file and "
              "register a new trial")
    for name in layering.get("hypothesis_looser_than_pack") or []:
        c.add("S0", "high", "pack", "hypothesis-looser-than-pack", str(name),
              f"the hypothesis sets {name} looser than the packs require",
              "tighten the threshold in a new hypothesis, or accept the looser gate knowingly")


def _limits(c: _Ctx) -> None:
    applies = {"always": True, "no-reconstruction": _reconstruction_missing(c), "data": "S1" in c.receipts,
               "llm": bool(c.trial.get("llm"))}
    for lim in KNOWN_LIMITS:
        if applies.get(lim["when"], False):
            c.add(lim["stage"], "low", "limit", lim["code"], lim["code"], lim["text"],
                  f"no action needed; know the limit. See {lim['pointer']}")


def _sort_key(w: ReadinessWarning) -> tuple[int, int, str, str]:
    return (STAGE_ORDER.index(w.stage) if w.stage in STAGE_ORDER else len(STAGE_ORDER), SEVERITIES.index(w.severity),
            w.code, w.scope)


def collect(home: str | Path, trial_id: str, registry: "Registry | None" = None) -> Collected:
    """Every warning of ``trial_id``, sorted by stage then severity. Raises ReadinessError for an unknown trial."""
    from qrh.memory.registry import Registry

    check_trial_id(trial_id)
    home = Path(home)
    own = registry is None
    reg = registry if registry is not None else Registry(home / "registry.sqlite3")
    try:
        trial = reg.get_trial(trial_id)
        if trial is None:
            raise ReadinessError(f"unknown trial {trial_id!r}")
        c = _Ctx(home, trial, reg)
        _load_receipts(c)
        verdict, failing = _verdict(c)
        for step in (_interview, _stability_thresholds, _stability_declared, _stability_check, _conditions, _data, _onboarding, _lint, _llm, _exemptions, _overrides, _methods, _packs, _limits):
            step(c)
        warnings = sorted(c.out, key=_sort_key)
        from qrh.readiness.two_track import load_section

        return Collected(trial_id, verdict, failing, warnings, {s: r.receipt_hash() for s, r in sorted(c.receipts.items())},
                         load_section(home, trial_id))
    finally:
        if own:
            reg.close()

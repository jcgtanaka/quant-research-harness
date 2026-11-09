"""S1 (data check) and S2 (implementation checks).

Both fail closed: anything that cannot be verified is BLOCKED, and the
verdict is decided by deterministic code only.
"""

from __future__ import annotations

import itertools
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from qrh.adapters.base import DataView
from qrh.adapters.loader import AdapterLoadError, load_adapter
from qrh.checks.data_quality import check_dataset_report
from qrh.checks.lookahead import BLOCK, Finding
from qrh.checks.sanity import sanity_rows
from qrh.checks.perturbation import DEFAULT_CUTOFFS as DEFAULT_PERTURBATION_CUTOFFS
from qrh.checks.perturbation import DEFAULT_MAX_CHANGE_BARS as DEFAULT_PERTURBATION_CHANGE_CAP
from qrh.checks.truncation import DEFAULT_ATOL, DEFAULT_CUTOFFS
from qrh.hashing import code_hash, hash_json, sha256_file
from qrh.llm.cache import LLMCacheCorrupt, LLMCacheMiss
from qrh.llm.lint import DIRECT_RULE, LLM_RULE, lint_llm_direct, lint_llm_use
from qrh.llm.runtime import current_session
from qrh.llm.variants import VARIANT_KEY, resolve_variants
from qrh.memory.catalog import Catalog, load_indexed_frame
from qrh.onboarding.card import dataset_doc, receipt_record as cat_state
from qrh.pipeline.artifacts import Artifacts
from qrh.pipeline.receipt import Receipt, Stage, Verdict
from qrh.pipeline.llm_support import receipt_bits as llm_receipt_bits
from qrh.pipeline.stages import MISS_TEXT, StageContext, make_receipt


def _dataset_hashes(cat: Catalog, datasets: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    """Catalog hash per dataset, verified against the file on disk."""
    hashes: dict[str, str] = {}
    reasons: list[str] = []
    for ds in datasets:
        rec = cat.get_dataset(ds["name"])
        if rec is None:
            reasons.append(f"{ds['name']}: unknown dataset (not in the catalog)")
            continue
        try:
            actual = sha256_file(rec["path"])
        except OSError as exc:
            reasons.append(f"{ds['name']}: file cannot be read: {exc}")
            continue
        if actual != rec["content_hash"]:
            reasons.append(f"{ds['name']}: file changed since registration (hash mismatch)")
            continue
        hashes[ds["name"]] = actual
    return hashes, reasons


class DataCheck:
    """S1: every declared dataset is registered, unchanged, and passes its quality rules."""

    id = Stage.S1

    def run(self, ctx: StageContext) -> Receipt:
        trial = ctx.registry.get_trial(ctx.trial_id)
        datasets = trial["datasets"] if trial else []
        if not datasets:
            return make_receipt(ctx, self.id, Verdict.BLOCKED, ["no datasets declared in the hypothesis"])
        evidence: list[dict[str, Any]] = []

        def rule_fn(rule_type: Any) -> Any:
            if not isinstance(rule_type, str) or not ctx.selector().registry.versions(f"s1.quality.{rule_type}"):
                return None  # unknown rule type: run_rules reports it as BLOCKED
            return ctx.method(f"s1.quality.{rule_type}").entry

        onboarding_state: dict[str, Any] = {}
        onboarding_docs: list[dict[str, Any]] = []
        with closing(Catalog(ctx.registry.path)) as cat:
            hashes, reasons = _dataset_hashes(cat, datasets)
            for ds in datasets:
                name = ds["name"]
                if name not in hashes:
                    evidence.append({"name": name, "status": "BLOCKED", "reasons": [r for r in reasons if r.startswith(f"{name}:")], "rules": []})
                    continue
                status, why, report, cov = check_dataset_report(cat, name, rule_fn)
                evidence.append({
                    "name": name, "status": status, "reasons": why,
                    **(cov or {"start": None, "end": None, "rows": 0, "step_seconds": None, "gap_count": 0, "gaps": []}),
                    "rules": [
                        {"rule": r.rule, "type": r.type, "passed": r.passed, "offenders": r.offenders, "reasons": r.reasons}
                        for r in (report.results if report else [])
                    ],
                })
                if status != "PASS":
                    reasons.extend(f"{name}: {r}" for r in (why or [f"status {status}"]))
                    continue
                rec = cat.get_dataset(name) or {}
                col = ds.get("publication_column")
                if ds.get("slow") and not col:
                    reasons.append(f"{name}: slow dataset must declare a publication column")
                elif col and col not in rec.get("schema", {}):
                    reasons.append(f"{name}: publication column {col!r} not present in the dataset")
            for name in hashes:  # what is known about each dataset: its onboarding state goes into the receipt (hashed)
                onboarding_state[name] = cat_state(cat, name)
                doc = dataset_doc(cat, name)
                if doc is not None:
                    onboarding_docs.append(doc)
        arts = Artifacts(ctx, self.id)
        arts.json("coverage", {"datasets": evidence})
        if onboarding_docs:  # the dataset card and onboarding table pages of the S1 report
            arts.json("onboarding", {"datasets": onboarding_docs})
        return make_receipt(
            ctx, self.id, Verdict.BLOCKED if reasons else Verdict.PASS, reasons,
            data_hashes=hashes,
            params={"datasets": [d["name"] for d in datasets], "onboarding": onboarding_state,
                    "artifacts": arts.as_params()},
        )


PNL_LINEAR = "s2.pnl_linear"
PNL_LINEAR_DEFAULT_VERSION = "pnl_linear@1"


def _run_pnl_linear(ctx: StageContext, trial: dict[str, Any], full: Any, frame: pd.DataFrame) -> dict[str, Any]:
    """The opt-in pnl.linear check: the hook where a hypothesis `reconstruction` block is read.

    The method version is the one the hypothesis pins in `methods`, else pnl_linear@1; either way it is recorded
    in the receipt as a non-default method. Returns the verdict document that goes to the artifact and the params."""
    from qrh.checks.pnl_linear_run import plain

    selector = ctx.selector()
    version = selector.pinned.get(PNL_LINEAR) or PNL_LINEAR_DEFAULT_VERSION
    method = selector.registry.get(PNL_LINEAR, version)
    selector.record(PNL_LINEAR, method.version)
    ppy = trial.get("periods_per_year")
    if ppy is None:
        ppy = ctx.method("metrics.periods_per_year").entry(full.returns.index)
    from datetime import date

    from qrh.costs.store import load_commission_records

    records, skipped = load_commission_records(ctx.home)  # confirmed broker records only; bad files are skipped
    extra: dict[str, Any] = {"commission_records": records, "as_of": date.today()} if records else {}
    built = trial.get("origin") != "user"  # Track 2 (origin user) is the only backtest the harness did not build
    if not built:
        extra["built"] = False
    v = method.entry(full, frame, trial.get("execution"), trial["reconstruction"], ppy, **extra)
    doc = {"verdict": v.verdict, "method": method.id, "version": method.version,
           "reasons": list(v.reasons), **v.details}
    from qrh.checks.pnl_linear_run import adjustment_record

    doc["adjustment"], adj_warnings = adjustment_record(trial["reconstruction"], built, _track2_answer(ctx, "price_adjustment"))
    doc["warnings"] = [*doc.get("warnings", []), *adj_warnings]
    doc["reasons"] += doc["warnings"]
    if skipped:  # a skipped costs file is shown to the user as a plain warning; the verdict is not changed
        doc["warnings"] = [*doc.get("warnings", []), *skipped]
        doc["reasons"] += [f"warning: costs folder, {w}" for w in skipped]
    doc["built"] = built
    if built:
        _apply_bug_correction(ctx, doc)
    return plain(doc)


def _bugs_from_warnings(warnings: list[str]) -> list[dict[str, str]]:
    """The days and amounts named in the bug warnings, as the correction flow expects them."""
    import re

    bugs: list[dict[str, str]] = []
    for w in warnings:
        if w.startswith("warning: bug:"):
            found = re.findall(r"(\d{4}-\d{2}-\d{2}) amount (\S+?)(?:,|;| \(|$)", w)
            bugs += [{"day": d, "amount": a} for d, a in found] or [{"day": "unknown", "amount": "unknown"}]
    return bugs


def _apply_bug_correction(ctx: StageContext, doc: dict[str, Any]) -> None:
    """Design rule: a bug in a harness-built backtest is fixed once by the LLM before the verdict; if it remains,
    the profit check is INCONCLUSIVE and names the bug. The request and the state live in the S2 artifact
    folder of the trial, subfolder ``correction``."""
    from qrh.checks.bug_correction import REQUEST_NAME, correction_request, correction_state, verdict_after_correction
    from qrh.pipeline.artifacts import artifact_dir

    run_dir = artifact_dir(ctx.home, ctx.trial_id, Stage.S2) / "correction"
    bugs = _bugs_from_warnings(doc.get("warnings", []))
    state = correction_state(run_dir)

    def conclude(reason: str, verdict: str) -> None:
        if verdict == "INCONCLUSIVE" and doc["verdict"] == "PASS":
            doc["verdict"] = "INCONCLUSIVE"  # a FAIL or BLOCKED verdict is never softened
        doc["reasons"] = [*doc["reasons"], f"{verdict.lower()}: {reason}"]

    if not bugs:
        if state == "done" and (run_dir / REQUEST_NAME).exists():
            before = [{"day": "unknown", "amount": "unknown"}]
            verdict, reason = verdict_after_correction(before, [], run_dir)
            conclude(reason, verdict)
        return
    if state in ("done", "exhausted"):
        verdict, reason = verdict_after_correction(bugs, bugs, run_dir)
        correction_request(run_dir, bugs)  # marks the single attempt as used
        conclude(reason, verdict)
        return
    correction_request(run_dir, bugs)
    conclude("a bug was found; a correction has been requested and the run is re-checked after it is recorded",
             "INCONCLUSIVE")


def _track2_answer(ctx: StageContext, question_id: str) -> str | None:
    """The stored Track 2 answer for the trial, or None when there is no usable answer file."""
    import json

    from qrh.readiness.two_track import answers_path

    try:
        answers = json.loads(answers_path(ctx.home, ctx.trial_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    answer = answers.get(question_id) if isinstance(answers, dict) else None
    return answer if isinstance(answer, str) else None


def _apply_truncation_rule(linear: dict[str, Any], truncation_verdict: str) -> None:
    """Truncation is required for a backtest the user brought: without a pass the profit check cannot PASS.
    For a harness-built backtest it is a silent self-check: a failure is a bug warning and the profit check stays."""
    if truncation_verdict == "PASS":
        return
    if linear.get("built", True):
        msg = (f"warning: bug: the truncation self-check of the harness-built backtest did not pass "
               f"({truncation_verdict}); the profit check result is not changed")
    else:
        if linear["verdict"] != "PASS":
            return
        linear["verdict"] = "FAIL" if truncation_verdict == "FAIL" else "INCONCLUSIVE"
        msg = (f"{linear['verdict'].lower()}: the truncation test did not pass ({truncation_verdict}); it is required "
               "for a backtest you brought, so the profit check cannot PASS")
    linear["warnings"] = [*linear.get("warnings", []), msg] if msg.startswith("warning") else linear.get("warnings", [])
    linear["reasons"] = [*linear["reasons"], msg]


def first_point(space: dict[str, list[Any]]) -> dict[str, Any]:
    """Deterministic parameters: sorted keys, first value of each."""
    return {k: space[k][0] for k in sorted(space)}


def _show(f: Finding, base: Path | None) -> str:
    try:
        name = str(Path(f.file).resolve().relative_to(base)) if base else Path(f.file).name
    except ValueError:
        name = Path(f.file).name
    return f"{name}:{f.line}"


_SEVERITY = {"PASS": 0, "INCONCLUSIVE": 1, "FAIL": 2, "BLOCKED": 3}


def _same_bar_check(
    ctx: StageContext, trial: dict[str, Any], full: Any, frame: pd.DataFrame, pnl_selected: bool, probe: Any = None,
) -> dict[str, Any]:
    """The default same-bar execution check, as the verdict document that goes to the artifact and the receipt.

    Skipped (and said so in the receipt) only where another returns reconstruction decides: the hypothesis selects
    ``s2.pnl_reconstruction`` (the same rebuild with the declared execution) or declares a ``reconstruction`` block
    (``s2.pnl_linear``, which explains the reported profit from lagged positions in contracts or shares, units the default
    rebuild cannot match)."""
    if pnl_selected:
        return {"verdict": "NOT RUN", "reasons": [], "details": {},
                "note": "s2.pnl_reconstruction is selected and runs the same reconstruction with the declared execution"}
    if trial.get("reconstruction") is not None:
        return {"verdict": "NOT RUN", "reasons": [], "details": {},
                "note": "the hypothesis declares a reconstruction block, so s2.pnl_linear explains the reported profit "
                        "from lagged positions"}
    method = ctx.method("s2.same_bar_execution")
    out = method.entry(full, frame, trial.get("execution"), probe)
    return {"verdict": out.verdict, "reasons": list(out.reasons), "details": out.details}


def _llm_context(cutoff: pd.Timestamp) -> Any:
    """Replay only, and a deterministic stub for a prompt asked after the cutoff (a miss at or before it fails)."""
    from contextlib import ExitStack

    from qrh.llm.runtime import replay_only, stub_misses_after

    stack = ExitStack()
    stack.enter_context(replay_only())
    stack.enter_context(stub_misses_after(cutoff))
    return stack


def _perturbation_check(
    ctx: StageContext, trial: dict[str, Any], adapter: Any, frame: pd.DataFrame, params: dict[str, Any],
    start: pd.Timestamp, end: pd.Timestamp, publication_column: str | None, full: Any, code: str, hashes: dict[str, str],
    factory: Any, in_session: bool,
) -> dict[str, Any]:
    """The future-perturbation invariance test, cached per code version like the truncation test.

    Every perturbed run gets a freshly loaded adapter (``factory``). Inside a language-model session the runs replay
    the cache, with a stub answer for prompts asked after the cutoff."""
    method = ctx.method("s2.future_perturbation")
    thresholds = trial["thresholds"]
    n = thresholds.get("perturbation_cutoffs", DEFAULT_PERTURBATION_CUTOFFS)  # hypotheses from before it existed
    cap = thresholds.get("perturbation_change_cap", DEFAULT_PERTURBATION_CHANGE_CAP)
    session = current_session()
    key = hash_json({
        "kind": "perturbation", "method": method.version, "code": code, "data": hashes, "params": params,
        "window": [start.isoformat(), end.isoformat()], "cutoffs": n, "change_cap": cap, "atol": DEFAULT_ATOL,
        **({"llm_cache": session.cache.content_hash()} if session is not None else {}),
    })
    cached = ctx.registry.cache_get(key)
    if cached is not None:
        return {**cached, "cached": True}
    out = method.entry(adapter, frame, params, start, end, publication_column, n, DEFAULT_ATOL, full_result=full,
                       max_change_bars=cap, adapter_factory=factory, context=_llm_context if in_session else None)
    doc = {"verdict": out.verdict, "reasons": out.reasons, "details": out.details}
    if out.verdict in ("PASS", "FAIL"):
        ctx.registry.cache_put(key, doc)
    return {**doc, "cached": False}


class ImplementationChecks:
    """S2: static look-ahead lint, future-truncation and future-perturbation invariance, same-bar execution, sanity
    bounds; for a strategy that uses a language model also the training-cutoff gate, the undeclared-LLM lint and
    the prompt check of the truncated runs."""

    id = Stage.S2

    def __init__(self, cutoffs: Sequence[float] = DEFAULT_CUTOFFS):
        self.cutoffs = tuple(cutoffs)

    def run(self, ctx: StageContext) -> Receipt:
        trial = ctx.registry.get_trial(ctx.trial_id)
        blocked = lambda why: make_receipt(ctx, self.id, Verdict.BLOCKED, why)  # noqa: E731
        if trial is None:
            return blocked(["no hypothesis registered for this trial"])
        missing = [
            k for k, v in (
                ("adapter", trial["adapter"]), ("datasets", trial["datasets"]),
                ("windows.development", trial["windows"].get("development")), ("bounds", trial["bounds"]),
            ) if not v
        ]
        if missing:
            return blocked([f"hypothesis does not declare: {', '.join(missing)}"])
        base = Path(trial["base_dir"]).resolve() if trial["base_dir"] else None
        llm = trial.get("llm")
        try:
            loaded = load_adapter(trial["adapter"], base, dependencies=(llm or {}).get("prompt_files") or ())
            variants = resolve_variants(llm, base) if llm else []
        except AdapterLoadError as exc:
            return blocked([str(exc)])
        adapter = loaded.adapter
        declared = {d["name"] for d in trial["datasets"]}
        try:
            needed = {r.name for r in adapter.required_data()}
            space = adapter.parameter_space()
        except Exception as exc:
            return blocked([f"adapter metadata unavailable: {type(exc).__name__}: {exc}"])
        if needed - declared:
            return blocked([f"adapter requires datasets not declared in the hypothesis: {sorted(needed - declared)}"])

        with closing(Catalog(ctx.registry.path)) as cat:
            hashes, reasons = _dataset_hashes(cat, trial["datasets"])
            primary = trial["datasets"][0]
            rec = cat.get_dataset(primary["name"])
        if reasons or rec is None:
            return blocked(reasons or ["primary dataset unknown"])
        try:
            frame = load_indexed_frame(rec["path"])
            tz = frame.index.tz
            start, end = (
                pd.Timestamp(t).tz_localize(tz) if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t)
                for t in trial["windows"]["development"]
            )
            params = trial["s2_params"] if trial["s2_params"] is not None else first_point(space)
            view = DataView(frame, primary.get("publication_column"))
        except Exception as exc:
            return blocked([f"adapter run failed: {type(exc).__name__}: {exc}"])
        runs = [{**params, VARIANT_KEY: v} for v in variants] or [params]  # one run per variant of an llm strategy

        findings = ctx.method("s2.lookahead_lint").entry(loaded.source_files)
        pack_rules = ctx.pack_view().lint_rules()
        extra_libs = tuple(r.ast["import"] for r in pack_rules if r.id == LLM_RULE and r.ast and "import" in r.ast)
        extra_libs += tuple(r.ast["import"] for r in pack_rules if r.id == DIRECT_RULE and r.ast and "import" in r.ast)
        pack_rules = [r for r in pack_rules if r.id not in (LLM_RULE, DIRECT_RULE)]  # those only add libraries
        if pack_rules:  # declarative extra rules from experience packs; same severities and suppression
            from qrh.packs.lint import lint_with_rules

            findings = [*findings, *lint_with_rules(loaded.source_files, pack_rules)]
        gate = None
        if llm:
            gate_method = ctx.method("s2.llm_gate")
            ctx.method("llm.replay_cache")
            gate = gate_method.entry(llm, trial["windows"], trial["thresholds"].get("llm_cutoff_margin_days"))
            findings = [*findings, *lint_llm_direct(loaded.source_files, extra_libs)]  # advisory (WARN)
        else:
            undeclared = lint_llm_use(loaded.source_files, extra_libs)  # inert unless an LLM client shows up
            if undeclared:
                ctx.method("s2.llm_gate")
                findings = [*findings, *undeclared]
        blocking = [f for f in findings if f.severity == BLOCK and not f.suppressed]
        reasons = [f"lint {f.rule_id} {_show(f, base)}: {f.message}" for f in blocking]
        lint_verdict = Verdict.FAIL if blocking else Verdict.PASS
        if gate is not None:
            reasons += [f"llm gate: {r}" for r in gate.reasons]
        undeclared_llm = any(f.rule_id == LLM_RULE for f in blocking)
        arts = Artifacts(ctx, self.id)
        lint_doc = {
            "verdict": lint_verdict.value,
            "findings": [
                {
                    "rule": f.rule_id, "at": _show(f, base), "line": f.line, "severity": f.severity,
                    "message": f.message, "snippet": f.snippet, "suppressed": f.suppressed, "reason": f.reason,
                }
                for f in findings
            ],
        }
        lint_params = {
            "verdict": lint_verdict.value,
            "warnings": [
                {"rule": f.rule_id, "at": _show(f, base), "snippet": f.snippet}
                for f in findings if f.severity != BLOCK and not f.suppressed
            ],
            "suppressed": [
                {"rule": f.rule_id, "at": _show(f, base), "reason": f.reason}
                for f in findings if f.suppressed
            ],
        }
        llm_params, llm_diag = llm_receipt_bits(trial)
        if gate is not None:
            arts.json("llm_gate", gate.evidence)
        common = {"llm_gate": gate.evidence} if gate is not None else {}

        if (gate is not None and gate.verdict != "PASS") or undeclared_llm:
            # The adapter is not run: with a gate that did not pass, or an LLM nobody declared, its output could not be
            # trusted or replayed anyway, and a replay-cache miss must not hide the reason.
            why = ("the LLM gate did not pass" if gate is not None and gate.verdict != "PASS"
                   else "the adapter uses an LLM that the hypothesis does not declare")
            reasons.append(f"truncation, sanity and reconstruction were not run: {why}")
            arts.json("lint", lint_doc)
            arts.json("truncation", {"verdict": "NOT RUN", "reasons": [], "notes": [why], "tolerance": None, "cutoffs": []})
            arts.json("sanity", {"verdict": "NOT RUN", "reasons": [], "rows": []})
            verdict = Verdict.BLOCKED if gate is not None and gate.verdict == "BLOCKED" else Verdict.FAIL
            return make_receipt(
                ctx, self.id, verdict, reasons, data_hashes=hashes,
                windows={k: (a, b) for k, (a, b) in trial["windows"].items()},
                diagnostics=llm_diag,
                params={
                    "adapter": trial["adapter"], "adapter_code_hash": loaded.code_hash, "artifacts": arts.as_params(),
                    "s2_params": params, "lint": lint_params, "truncation": {"verdict": "NOT RUN"},
                    "perturbation": {"verdict": "NOT RUN"}, "same_bar_execution": {"verdict": "NOT RUN"},
                    "sanity": {"verdict": "NOT RUN"}, **common, **llm_params,
                },
            )

        session0 = current_session()
        calls0 = session0.stats()["calls"] if session0 is not None else 0
        try:
            fulls = [adapter.run(p, view, start, end) for p in runs]
        except LLMCacheMiss:
            return blocked([MISS_TEXT.format(trial=ctx.trial_id)])
        except LLMCacheCorrupt as exc:
            return blocked([str(exc)])
        except Exception as exc:
            return blocked([f"adapter run failed: {type(exc).__name__}: {exc}"])
        full = fulls[0]

        pnl_method = ctx.selected_method("s2.pnl_reconstruction")  # opt-in: only where the hypothesis selects it
        pnl = None
        if pnl_method is not None:
            execution = trial.get("execution")
            if not execution:
                return blocked([
                    "s2.pnl_reconstruction is selected but the hypothesis declares no execution "
                    "(execution: {lag_bars: >= 1, price_column: ...} is required; there is no default)"
                ])
            pnl = pnl_method.entry(full, frame, execution)
        linear = None  # opt-in: only where the hypothesis declares a `reconstruction` block
        if trial.get("reconstruction") is not None:
            linear = _run_pnl_linear(ctx, trial, full, frame)
        truncation = ctx.method("s2.truncation")

        results: list[tuple[str, dict[str, Any], bool]] = []
        cache_hash = ""
        session = current_session()
        for p, full_run in zip(runs, fulls):
            if session is not None:
                cache_hash = session.cache.content_hash()
            key = hash_json({
                "kind": "truncation", "method": truncation.version, "code": loaded.code_hash or code_hash(loaded.source_files),
                "data": hashes, "params": p, "window": [start.isoformat(), end.isoformat()],
                "cutoffs": list(self.cutoffs), "atol": DEFAULT_ATOL,
                **({"llm_cache": cache_hash} if cache_hash else {}),  # a re-recorded cache is a new question
            })
            cached = ctx.registry.cache_get(key)
            if cached is not None and "series" not in cached:  # stored before comparison series were kept
                cached = None
            if cached is None:
                tr = truncation.entry(
                    adapter, frame, p, start, end, primary.get("publication_column"),
                    self.cutoffs, DEFAULT_ATOL, full_result=full_run,
                )
                cached = {"verdict": tr.verdict, "reasons": tr.reasons, "details": tr.details, "series": tr.series}
                if tr.verdict in ("PASS", "FAIL"):
                    ctx.registry.cache_put(key, cached)
                results.append((str((p.get(VARIANT_KEY) or {}).get("name", "")), cached, False))
            else:
                results.append((str((p.get(VARIANT_KEY) or {}).get("name", "")), cached, True))
        cached = max((r[1] for r in results), key=lambda c: _SEVERITY.get(c["verdict"], 3))  # the worst, else the first
        was_cached = all(r[2] for r in results)
        if linear is not None:
            _apply_truncation_rule(linear, cached["verdict"])
        for name, one, _ in results:
            tag = f"[variant {name}] " if len(results) > 1 else ""
            reasons += [f"truncation: {tag}{r}" for r in one["reasons"]]
        session = current_session()
        llm_calls = (session.stats()["calls"] - calls0) if session is not None else 0
        code_id = loaded.code_hash or code_hash(loaded.source_files)

        def fresh() -> Any:
            return load_adapter(trial["adapter"], base, dependencies=(llm or {}).get("prompt_files") or ()).adapter

        if llm and session is None:  # declared, but there is no replay session to run the perturbed prompts in
            pert = {"verdict": "NOT RUN", "reasons": [
                "the future-perturbation test did not run: no language-model session was active"], "cached": False,
                    "details": {"n_cutoffs": 0, "cutoffs": [], "notes": ["no language-model session was active"]}}
        else:
            pert = _perturbation_check(ctx, trial, adapter, frame, runs[0], start, end, primary.get("publication_column"),
                                       fulls[0], code_id, hashes, fresh, session is not None)
            pert["details"]["llm_calls_in_full_run"] = llm_calls if llm else 0
        reasons += [f"perturbation: {r}" for r in pert["reasons"]]
        from qrh.checks.perturbation import make_own_bar_probe

        probe = make_own_bar_probe(adapter, frame, runs[0], start, end, primary.get("publication_column"), fulls[0],
                                   adapter_factory=fresh, context=_llm_context if session is not None else None)
        same_bar = _same_bar_check(ctx, trial, full, frame, pnl_method is not None, probe)
        reasons += [f"same_bar_execution: {r}" for r in same_bar["reasons"]]
        san = ctx.method("s2.sanity").entry(full, trial["bounds"])
        reasons += [f"sanity: {r}" for r in san.reasons]

        if pnl is not None:
            reasons += [f"pnl_reconstruction: {r}" for r in pnl.reasons]
        if pnl is not None:
            arts.json("pnl", {"verdict": pnl.verdict, "reasons": pnl.reasons, "details": pnl.details})
        if linear is not None:
            reasons += [f"pnl_linear: {r}" for r in linear["reasons"]]
            arts.json("pnl_linear", linear)
        arts.json("lint", lint_doc)
        arts.json("truncation", {
            "verdict": cached["verdict"], "reasons": cached["reasons"],
            "notes": cached["details"].get("notes", []), "tolerance": cached["details"].get("tolerance"),
            "cutoffs": cached["series"],
        })
        arts.json("sanity", {
            "verdict": san.verdict, "reasons": san.reasons, "rows": sanity_rows(san.metrics, trial["bounds"]),
        })
        arts.json("perturbation", {"verdict": pert["verdict"], "reasons": pert["reasons"], **pert["details"]})
        arts.json("same_bar_execution", {k: same_bar[k] for k in ("verdict", "reasons", "details")})
        verdicts = {lint_verdict.value, cached["verdict"], san.verdict} | ({pnl.verdict} if pnl else set())
        verdicts.add("INCONCLUSIVE" if pert["verdict"] == "NOT RUN" else pert["verdict"])  # an unrun test is never dropped
        if same_bar["verdict"] != "NOT RUN":
            verdicts.add(same_bar["verdict"])
        verdicts |= {linear["verdict"]} if linear is not None else set()
        verdict = (
            Verdict.BLOCKED if "BLOCKED" in verdicts else Verdict.FAIL if "FAIL" in verdicts
            else Verdict.INCONCLUSIVE if "INCONCLUSIVE" in verdicts else Verdict.PASS
        )
        llm_params, llm_diag = llm_receipt_bits(trial)  # after the truncated runs: the entries this stage used
        return make_receipt(
            ctx, self.id, verdict, reasons,
            data_hashes=hashes,
            windows={k: (a, b) for k, (a, b) in trial["windows"].items()},
            metrics=san.metrics,
            diagnostics={"truncation_cached": was_cached, "perturbation_cached": pert["cached"], **llm_diag},  # run dependent: never part of the receipt hash
            params={
                "adapter": trial["adapter"],
                "adapter_code_hash": loaded.code_hash,
                "artifacts": arts.as_params(),
                "s2_params": params,
                "lint": lint_params,
                "truncation": {**cached["details"], "verdict": cached["verdict"]},
                "sanity": {"verdict": san.verdict},
                "perturbation": {**pert["details"], "verdict": pert["verdict"]},
                "same_bar_execution": {**same_bar["details"], "verdict": same_bar["verdict"],
                                       **({"note": same_bar["note"]} if "note" in same_bar else {})},
                **({"pnl_reconstruction": {"verdict": pnl.verdict, **pnl.details}} if pnl else {}),
                **({"pnl_linear": linear} if linear is not None else {}),
                **common, **llm_params,
            },
        )

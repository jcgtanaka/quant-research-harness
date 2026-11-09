"""Stage framework and the pipeline runner.

Rules enforced here, not left to stages:
- stage N runs only after a verified PASS receipt of stage N-1 exists;
- any exception, malformed receipt or unwritable receipt becomes BLOCKED;
- S7 always runs and summarizes, naming the failing stage.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import qrh
from qrh.hashing import code_hash
from qrh.pipeline.artifacts import Artifacts
from qrh.memory.registry import HypothesisRecord, Registry
from qrh.methods.registry import Method, MethodError
from qrh.methods.selection import MethodSelector, check_pins
from qrh.pipeline.gates import check_hypothesis_spec, overall_verdict, require_previous_pass
from qrh.pipeline.report import build_report, collect_methods, report_path
from qrh.pipeline.receipt import (
    Receipt,
    Stage,
    Verdict,
    collect_environment,
    display_verdict,
    receipt_path,
    write_receipt,
)
from qrh.validation.defaults import fill_thresholds

WORK_STAGES = [Stage(f"S{i}") for i in range(7)]  # S0..S6; S7 summarizes
MISS_TEXT = ("replay cache miss: this trial has no recorded LLM answer for this data and code version; "
             "re-run with --llm-mode record to record it (this calls the model and may cost money): "
             "`qrh run {trial} --llm-mode record`")


@lru_cache(maxsize=1)
def package_code_hash() -> str:
    root = Path(qrh.__file__).parent
    return code_hash(sorted(root.rglob("*.py")))


@dataclass
class StageContext:
    trial_id: str
    registry: Registry
    home: Path
    parent_receipt_hash: str | None = None
    spec: HypothesisRecord | None = None
    seeds: dict[str, int] = field(default_factory=dict)
    results: dict[Stage, Receipt] = field(default_factory=dict)  # receipts of the current run
    _selector: MethodSelector | None = field(default=None, repr=False)
    _packs: Any = field(default=None, repr=False)

    def pack_view(self, strict: bool = False) -> Any:
        """The active experience packs. ``strict`` raises PackError; otherwise a failure gives an empty view
        (S0 reports the failure, so a receipt can still be written)."""
        from qrh.packs.model import PackError
        from qrh.packs.runtime import PackView

        if self._packs is None:
            try:
                self._packs = PackView.load(self.home)
            except PackError:
                if strict:
                    raise
                return PackView.empty()
        return self._packs

    def selector(self) -> MethodSelector:
        """Method selection for this trial: the versions pinned (and recorded) at S0, else the defaults."""
        if self._selector is None:
            trial = self.registry.get_trial(self.trial_id)
            pinned = (trial or {}).get("methods") or (self.spec.methods if self.spec else {})
            self._selector = MethodSelector(self.home, pinned=pinned)
        return self._selector

    def method(self, method_id: str) -> Method:
        """The method to use for ``method_id``; the version handed out is recorded in the receipt."""
        return self.selector().get(method_id)

    def selected_method(self, method_id: str) -> Method | None:
        """Like ``method`` but None for an opt-in id (no shipped default) that this trial did not select."""
        return self.selector().selected(method_id)

    @property
    def methods_used(self) -> dict[str, str]:
        return dict(sorted(self._selector.used.items())) if self._selector else {}


@runtime_checkable
class StageProtocol(Protocol):
    id: Stage

    def run(self, context: StageContext) -> Receipt: ...


def _hypothesis_hash(ctx: StageContext) -> str:
    trial = ctx.registry.get_trial(ctx.trial_id)
    return trial["hypothesis_hash"] if trial else ""


def _interview_status(ctx: StageContext, view: Any, stage: Stage) -> dict[str, Any] | None:
    """Warn-only interview status of a stage for its receipt; never raises, never changes a verdict."""
    try:
        from qrh.interview.snapshot import receipt_status

        trial = ctx.registry.get_trial(ctx.trial_id)
        return receipt_status(ctx.registry.conn, view, trial, stage.value) if trial else None
    except Exception:
        return None


def _onboarding_questions(ctx: StageContext) -> bool:
    """Whether a dataset of this trial has open onboarding questions (never raises; nothing here changes a verdict)."""
    try:
        from qrh.interview.engine import onboarding_items

        trial = ctx.registry.get_trial(ctx.trial_id)
        scopes = {f"dataset:{d['name']}" for d in (trial or {}).get("datasets", [])}
        return bool(scopes) and bool(onboarding_items(ctx.registry.conn, scopes))
    except Exception:
        return False


def make_receipt(
    ctx: StageContext, stage: Stage, verdict: Verdict, reasons: list[str], **fields: Any
) -> Receipt:
    """Build a receipt with the common provenance fields filled in."""
    base: dict[str, Any] = dict(
        trial_id=ctx.trial_id,
        stage=stage,
        hypothesis_hash=_hypothesis_hash(ctx),
        code_hash=package_code_hash(),
        verdict=verdict,
        reasons=reasons,
        environment=collect_environment(ctx.seeds),
        parent_receipt_hash=ctx.parent_receipt_hash,
    )
    base.update(fields)
    if "methods" not in base:
        base["methods"] = ctx.methods_used
    view = ctx.pack_view()
    if view or _onboarding_questions(ctx):  # packs ask questions; so does the onboarding of the trial's datasets
        if view:
            base.setdefault("packs", view.record())
        status = _interview_status(ctx, view, stage)
        if status is not None:
            base["params"] = {**base.get("params", {}), "interview": status}
    flagged = ctx._selector.non_default() if ctx._selector is not None else {}
    if flagged:
        base["params"] = {**base.get("params", {}), "non_default_methods": flagged}
    return Receipt(**base)


def stability_sources(thresholds: dict[str, Any], explicit: set[str], answers: dict[str, dict[str, Any]],
                      layering: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """For each stability threshold: its frozen value and where it came from. A value written in the hypothesis wins
    over the trader's answer (and is labelled so: it is not the trader's); then the trader's own value, a pack that
    tightened it, the trader's "keep the default", and last the default."""
    from qrh.interview.stability import STABILITY_NAMES

    out: dict[str, Any] = {}
    for name in STABILITY_NAMES:
        a = answers.get(name)
        if name in explicit:
            source = "hypothesis"
        elif a is not None and a["value"] is not None:
            source = f"trader answer ({a['qid']} v{a['version']})"
        elif name in (layering.get("tightened") or {}):
            source = f"pack {layering['tightened'][name]['pack']}"
        elif a is not None:
            source = "trader answer: keep the default"
        else:
            source = "default"
        out[name] = {"value": thresholds.get(name), "source": source,
                     "default": (defaults or {}).get(name, thresholds.get(name) if source == "default" else None)}
    return out


def _default_thresholds(spec: HypothesisRecord, home: str | Path) -> dict[str, Any]:
    """Thresholds in force before the hypothesis speaks: core defaults, tightened by the active packs."""
    from qrh.packs.apply import defaults_in_force
    from qrh.packs.runtime import PackView
    from qrh.validation.defaults import DEFAULT_THRESHOLDS

    view = PackView.load(home)
    return defaults_in_force(view, spec, home) if view else dict(DEFAULT_THRESHOLDS)


class RegisterHypothesis:
    """S0: register the hypothesis (or re-verify the stored one) and write its receipt."""

    id = Stage.S0

    def run(self, ctx: StageContext) -> Receipt:
        from qrh.packs.model import PackError

        try:
            view = ctx.pack_view(strict=True)
        except PackError as exc:
            return make_receipt(ctx, self.id, Verdict.BLOCKED, [f"experience packs: {exc}"])
        trial = ctx.registry.get_trial(ctx.trial_id)
        layering: dict[str, Any] = {}
        stability_doc: dict[str, Any] | None = None
        created = False
        if trial is None:
            if ctx.spec is None:
                return make_receipt(ctx, self.id, Verdict.BLOCKED, ["no hypothesis registered for this trial"])
            reasons = check_pins(ctx.home, ctx.spec.methods)
            if reasons:
                return make_receipt(ctx, self.id, Verdict.FAIL, reasons)
            spec = ctx.spec
            explicit = set(spec.thresholds)
            answers = self._trader_answers(ctx, spec)
            spec = spec.model_copy(update={"thresholds": {
                **{n: a["value"] for n, a in answers.items() if a["value"] is not None}, **spec.thresholds}})
            if view:
                from qrh.packs.apply import apply_s0

                try:
                    outcome = apply_s0(view, spec, ctx.home)
                except PackError as exc:
                    return make_receipt(ctx, self.id, Verdict.BLOCKED, [f"experience packs: {exc}"])
                if outcome.reasons:
                    return make_receipt(ctx, self.id, Verdict.FAIL, outcome.reasons)
                spec, layering = outcome.spec, outcome.layering
            # Missing thresholds and methods are filled from the defaults and stored explicitly, so what
            # this trial uses cannot change when a default is promoted later.
            filled = spec.model_copy(update={
                "thresholds": fill_thresholds(spec.thresholds),
                "methods": MethodSelector(ctx.home, pinned=spec.methods).full_map(),
            })
            fingerprint = None
            if filled.llm is not None:  # the prompt files must be readable before anything is registered
                from qrh.adapters.loader import AdapterLoadError
                from qrh.llm.variants import llm_fingerprint

                try:
                    fingerprint = llm_fingerprint(filled.llm.model_dump(mode="json"), filled.base_dir)
                except AdapterLoadError as exc:
                    return make_receipt(ctx, self.id, Verdict.BLOCKED, [f"llm: {exc}"])
            stability_doc = stability_sources(filled.thresholds, explicit, answers, layering,
                                              _default_thresholds(spec, ctx.home))
            ctx.registry.register_hypothesis(filled, trial_id=ctx.trial_id)
            if fingerprint is not None:
                self._register_llm(ctx, filled, fingerprint)
            trial = ctx.registry.get_trial(ctx.trial_id)
            created = True
        assert trial is not None
        record, reasons = check_hypothesis_spec(
            {k: trial[k] for k in HypothesisRecord.model_fields}
        )
        if record is None:
            return make_receipt(ctx, self.id, Verdict.FAIL, reasons)
        if record.hypothesis_hash() != trial["hypothesis_hash"]:
            return make_receipt(ctx, self.id, Verdict.BLOCKED, ["stored hypothesis does not match its hash"])
        sel = ctx.selector()
        for mid, version in (record.methods or sel.full_map()).items():
            sel.record(mid, version)
        arts = Artifacts(ctx, self.id)
        arts.json("hypothesis", {  # what the S0 text card shows; locations (base_dir) are left out
            **{k: trial[k] for k in HypothesisRecord.model_fields if k != "base_dir"},
            "trial_id": ctx.trial_id, "hypothesis_hash": trial["hypothesis_hash"],
        })
        pack_params: dict[str, Any] = {}
        if view:
            pack_params = self._pack_params(ctx, view, trial, arts, layering, created)
        stability_params = self._stability_params(ctx, arts, stability_doc, created)
        llm_params: dict[str, Any] = {}
        if record.llm is not None:  # read back from the registry: written once, so a re-run cannot change the receipt
            stored = ctx.registry.get_llm_fingerprint(ctx.trial_id)
            if stored is None:
                return make_receipt(ctx, self.id, Verdict.BLOCKED, ["llm: no fingerprint was stored for this trial"])
            llm_params = {"llm": {**record.binding()["llm"], "prompt_hash": stored["prompt_hash"],
                                  "fingerprint": stored["fingerprint"], "notes": stored["notes"]}}
        return make_receipt(
            ctx,
            self.id,
            Verdict.PASS,
            [],
            params={
                "family_id": trial["family_id"],
                "parameter_space_size": trial["parameter_space_size"],
                "thresholds": trial["thresholds"],
                **record.binding(),
                **llm_params,
                **pack_params,
                **stability_params,
                "artifacts": arts.as_params(),
            },
        )

    @staticmethod
    def _trader_answers(ctx: StageContext, spec: HypothesisRecord) -> dict[str, dict[str, Any]]:
        """The trader's current answers to the stability questions of this strategy family (empty when unreadable)."""
        from qrh.interview.stability import trader_answers

        try:
            return trader_answers(ctx.registry.conn, spec.resolved_family_id())
        except Exception:  # the interview tables are advisory here: without answers the defaults apply
            return {}

    @staticmethod
    def _stability_params(ctx: StageContext, arts: Artifacts, doc: dict[str, Any] | None, created: bool) -> dict[str, Any]:
        """Where each stability threshold came from. Written once at registration and re-read after, so that answers
        given later cannot change the S0 receipt of an existing trial."""
        import json

        from qrh.hashing import sha256_bytes
        from qrh.pipeline.artifacts import artifact_dir

        path = artifact_dir(ctx.home, ctx.trial_id, Stage.S0) / "stability_thresholds.json"
        if created and doc:
            arts.json("stability_thresholds", doc)
            return {"stability_thresholds": doc}
        if not created and path.is_file():
            data = path.read_bytes()
            arts.add_existing("stability_thresholds", f"trials/{ctx.trial_id}/S0/stability_thresholds.json", sha256_bytes(data))
            return {"stability_thresholds": json.loads(data.decode("utf-8"))}
        return {}

    @staticmethod
    def _register_llm(ctx: StageContext, spec: HypothesisRecord, fingerprint: dict[str, str]) -> None:
        """Store the prompt/model fingerprint and, when an earlier trial of the family differs, the notice."""
        family = spec.resolved_family_id()
        notes: list[str] = []
        earlier = ctx.registry.llm_trials_differing(family, ctx.trial_id, fingerprint["fingerprint"])
        if earlier:
            notes.append(
                f"prompt or model changed since trial {earlier[0]}: counts as another trial; "
                f"family count is now {ctx.registry.family_declared_count(family)}")
        ctx.registry.record_llm_fingerprint(ctx.trial_id, family, fingerprint["prompt_hash"],
                                            fingerprint["fingerprint"], notes)

    @staticmethod
    def _pack_params(ctx: StageContext, view: Any, trial: dict[str, Any], arts: Artifacts, layering: dict[str, Any],
                     created: bool) -> dict[str, Any]:
        """Pack facts for the S0 receipt. The layering record is written once, at registration, and re-read after."""
        import json

        from qrh.hashing import sha256_bytes
        from qrh.packs.apply import account_record, lesson_records
        from qrh.packs.imports import sync_imported
        from qrh.pipeline.artifacts import artifact_dir

        for pack in view.packs:
            sync_imported(ctx.registry.conn, pack.name, pack.hash, pack.attempts)
        out: dict[str, Any] = {}
        lessons = lesson_records(view, trial)
        if lessons:
            out["lessons"] = lessons
        account = account_record(view, trial)
        if account:
            out["account"] = account
        path = artifact_dir(ctx.home, ctx.trial_id, Stage.S0) / "pack_layering.json"
        if created and layering:
            arts.json("pack_layering", layering)
        elif not created and path.is_file():
            data = path.read_bytes()
            layering = json.loads(data.decode("utf-8"))
            arts.add_existing("pack_layering", f"trials/{ctx.trial_id}/S0/pack_layering.json", sha256_bytes(data))
        if layering:
            out["pack_layering"] = layering
        return out


class Finalize:
    """S7: summarize the verdicts of this run; PASS only if S0..S6 all PASS."""

    id = Stage.S7

    def run(self, ctx: StageContext) -> Receipt:
        verdicts = {s: r.verdict for s, r in ctx.results.items() if s in WORK_STAGES}
        overall, reasons = overall_verdict(verdicts, WORK_STAGES)
        failing = next((s.value for s in WORK_STAGES if s in verdicts and verdicts[s] is not Verdict.PASS), None)
        if failing is None and overall is not Verdict.PASS:
            failing = next((s.value for s in WORK_STAGES if s not in verdicts), None)
        trial = ctx.registry.get_trial(ctx.trial_id)
        conditions: list[dict[str, Any]] = []
        arts = Artifacts(ctx, self.id)
        interview_totals: dict[str, int] | None = None
        snapshot_error: str | None = None
        view = ctx.pack_view()
        if trial is not None and (view or _onboarding_questions(ctx)):
            snap, snapshot_error = self._interview(ctx, view, trial)
            if snapshot_error:
                reasons = reasons + [f"interview snapshot failed, open questions may be unlisted: {snapshot_error}"]
            if snap and snap["stages"]:
                arts.json("interview", snap)
                conditions = snap["conditions"]
                interview_totals = {k: sum(b["counts"][k] for b in snap["stages"].values())
                                    for k in ("answered", "unknown", "pending")}
            if snapshot_error:
                conditions = conditions + [{
                    "question_id": "interview_snapshot", "status": "failed", "stage": self.id.value, "scope": "run",
                    "text": f"interview snapshot failed, open questions may be unlisted: {snapshot_error}"}]
        if overall is not Verdict.PASS:
            conditions = []  # only a clean PASS can become conditional; a failure is reported as it is
        elif conditions:
            overall = Verdict.PASS_CONDITIONAL
            reasons = reasons + [
                f"{display_verdict(overall)}: {len(conditions)} open condition(s): "
                + ", ".join(f"{c['question_id']} ({c['status']})" for c in conditions)]
        path = None
        if trial is not None:
            path = report_path(ctx.home, ctx.trial_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(build_report(trial, ctx.results, ctx.home, overall, failing, conditions), encoding="utf-8")
        detail = [f"{s.value}: {r.verdict.value} - {'; '.join(r.reasons)}" for s, r in sorted(ctx.results.items(), key=lambda kv: kv[0].value) if r.verdict is not Verdict.PASS]
        methods, flagged = collect_methods(ctx.results)
        return make_receipt(
            ctx,
            self.id,
            overall,
            reasons + detail,
            methods=methods,
            params={
                **({"non_default_methods": flagged} if flagged else {}),
                "stage_verdicts": {s.value: v.value for s, v in sorted(verdicts.items(), key=lambda kv: kv[0].value)},
                "failing_stage": failing,
                **({"report_path": str(path)} if path else {}),
                **({"conditions": conditions} if conditions else {}),
                **({"interview_totals": interview_totals} if interview_totals else {}),
                **({"interview_snapshot_error": snapshot_error} if snapshot_error else {}),
                **({"artifacts": arts.as_params()} if arts.as_params() else {}),
            },
        )

    @staticmethod
    def _interview(ctx: StageContext, view: Any, trial: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Return (snapshot, error); a failure never changes a verdict but is returned so S7 can record it."""
        try:
            from qrh.interview.snapshot import build_snapshot

            return build_snapshot(ctx.registry.conn, view, trial), None
        except Exception as exc:  # warn-only: an interview problem never changes a verdict
            return None, f"{type(exc).__name__}: {exc}"


def default_stages() -> list[StageProtocol]:
    from qrh.pipeline.stages_s1_s2 import DataCheck, ImplementationChecks
    from qrh.pipeline.stages_s3_s7 import InSampleSearch, OverfittingStats, UncertaintyAndCosts, WalkForward

    return [
        RegisterHypothesis(), DataCheck(), ImplementationChecks(),
        InSampleSearch(), WalkForward(), OverfittingStats(), UncertaintyAndCosts(),
    ]


@dataclass
class PipelineResult:
    final: Receipt
    final_path: Path
    stage_receipts: dict[Stage, Receipt]


class Pipeline:
    def __init__(
        self,
        registry: Registry,
        home: str | Path,
        stages: Sequence[StageProtocol] | None = None,
        finalizer: StageProtocol | None = None,
        seeds: dict[str, int] | None = None,
        llm_mode: str = "replay",
    ):
        from qrh.llm import MODES

        if llm_mode not in MODES:
            raise ValueError(f"llm_mode must be record or replay, got {llm_mode!r}")
        self.llm_mode = llm_mode
        self.registry = registry
        self.home = Path(home)
        self.stages = list(stages) if stages is not None else default_stages()
        ids = [s.id for s in self.stages]
        if ids != sorted(ids, key=lambda s: s.value) or len(set(ids)) != len(ids) or Stage.S7 in ids:
            raise ValueError("stages must be unique, ordered, and exclude S7")
        self.finalizer = finalizer or Finalize()
        self.seeds = seeds or {}

    def _stage(self, stage: Stage) -> StageProtocol:
        for s in self.stages + [self.finalizer]:
            if s.id is stage:
                return s
        raise KeyError(f"no stage {stage.value} configured")

    def _llm_session(self, trial_id: str) -> Any:
        """The language-model session of a trial that declares ``llm`` (the mode is a context variable, not an
        environment variable); a no-op context for every other trial."""
        trial = self.registry.get_trial(trial_id)
        llm = trial.get("llm") if trial else None
        if not llm:
            return contextlib.nullcontext()
        from qrh.hashing import hash_json
        from qrh.llm import LLMError, llm_session
        from qrh.llm.cache import cache_dir
        from qrh.llm.runtime import require_bound
        from qrh.memory.catalog import Catalog
        from qrh.pipeline.stages_s1_s2 import _dataset_hashes

        with contextlib.closing(Catalog(self.registry.path)) as cat:
            hashes, reasons = _dataset_hashes(cat, trial["datasets"])
        if reasons or not hashes:
            # a dataset that is unknown or changed cannot key an answer; the stage reports it (S1 BLOCKED) and,
            # if it reaches a model call, the cache key refuses an empty hash
            raise LLMError("; ".join(reasons) or "the trial declares no dataset to key the LLM cache on")
        dataset_hash, code = hash_json(sorted(hashes.items())), package_code_hash()
        require_bound(dataset_hash, code)  # the placeholder never keys an answer of a pipeline run
        return llm_session(self.llm_mode, cache_dir(self.home, trial_id), {m["id"]: m["provider"] for m in llm["models"]},
                           dataset_hash=dataset_hash, code_hash=code)

    def _render(self, receipt: Receipt) -> None:
        """Render the stage's visual report. Never raises and never touches a verdict:
        a failure is recorded in the report manifest (and shown by ``qrh status``).
        ``QRH_REPORTS=0`` switches automatic rendering off (``qrh report`` still works)."""
        if os.environ.get("QRH_REPORTS", "1").strip().lower() in ("0", "false", "off", "no"):
            return
        try:
            from qrh.reports.render import render_after_stage

            render_after_stage(self.home, self.registry, receipt)
        except Exception as exc:  # for example matplotlib is not importable
            try:
                from qrh.reports.manifest import record_error

                record_error(self.home, receipt.trial_id, receipt.stage.value, f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

    def _finish(self, ctx: StageContext, receipt: Receipt) -> Receipt:
        persisted = self._persist(ctx, receipt)
        self._render(persisted)
        return persisted

    def _persist(self, ctx: StageContext, receipt: Receipt, reason: str | None = None) -> Receipt:
        """Write a receipt; if that fails, write a BLOCKED one instead (fail closed)."""
        try:
            write_receipt(receipt, self.home, self.registry)
            return receipt
        except Exception as exc:
            blocked = make_receipt(
                ctx, receipt.stage, Verdict.BLOCKED,
                [f"receipt could not be written: {type(exc).__name__}: {exc}"],
            )
            write_receipt(blocked, self.home, self.registry)
            return blocked

    def run_stage(
        self,
        trial_id: str,
        stage: Stage,
        spec: HypothesisRecord | None = None,
        results: dict[Stage, Receipt] | None = None,
    ) -> Receipt:
        """Run one stage under the ordering gate; always returns a persisted receipt."""
        ctx = StageContext(
            trial_id=trial_id, registry=self.registry, home=self.home, spec=spec,
            seeds=self.seeds, results=results if results is not None else {},
        )
        impl = self._stage(stage)
        if stage is Stage.S7:
            ctx.parent_receipt_hash = max(ctx.results.items(), key=lambda kv: kv[0].value)[1].receipt_hash() if ctx.results else None
        else:
            gate = require_previous_pass(self.registry, trial_id, stage, WORK_STAGES)
            if not gate.ok:
                return self._finish(ctx, make_receipt(ctx, stage, Verdict.BLOCKED, [gate.reason or "gate refused"]))
            ctx.parent_receipt_hash = gate.parent_hash
        try:
            with self._llm_session(trial_id):
                receipt = impl.run(ctx)
            if not isinstance(receipt, Receipt):
                raise TypeError(f"stage returned {type(receipt).__name__}, not Receipt")
            if receipt.stage is not stage or receipt.trial_id != trial_id:
                raise ValueError(f"stage {stage.value} returned a receipt for {receipt.trial_id}/{receipt.stage.value}")
            receipt = receipt.model_copy(update={"parent_receipt_hash": ctx.parent_receipt_hash})
        except Exception as exc:
            from qrh.llm import LLMCacheCorrupt, LLMCacheMiss

            if isinstance(exc, LLMCacheMiss):
                reason = MISS_TEXT.format(trial=trial_id)
            elif isinstance(exc, LLMCacheCorrupt):
                reason = f"{exc}"
            else:
                reason = f"{type(exc).__name__}: {exc}"
            receipt = make_receipt(ctx, stage, Verdict.BLOCKED, [reason])
        return self._finish(ctx, receipt)

    def run(self, trial_id: str, spec: HypothesisRecord | None = None) -> PipelineResult:
        """Run S0..S6 in order, stopping at the first non-PASS, then always S7."""
        if self.registry.get_trial(trial_id) is None and spec is None:
            raise KeyError(f"unknown trial {trial_id!r}")
        # Receipts from earlier runs of later stages are stale: drop them.
        for p in self.registry.clear_receipts(trial_id, from_stage=Stage.S1.value):
            Path(p).unlink(missing_ok=True)
        try:  # reports of those stages are stale too
            from qrh.reports.manifest import clear_from

            clear_from(self.home, trial_id, Stage.S1)
        except Exception:
            pass
        results: dict[Stage, Receipt] = {}
        for impl in self.stages:
            receipt = self.run_stage(trial_id, impl.id, spec=spec, results=results)
            results[impl.id] = receipt
            if receipt.verdict is not Verdict.PASS:
                break
        final = self.run_stage(trial_id, Stage.S7, results=results)
        return PipelineResult(final, receipt_path(self.home, trial_id, Stage.S7), results)


@dataclass
class NewTrialResult:
    trial_id: str | None
    path: Path | None
    receipt: Receipt | None
    reasons: list[str]
    notes: list[str] = field(default_factory=list)  # notices that never block (for example a changed prompt)


def _check_packs(spec: HypothesisRecord, home: str | Path) -> list[str]:
    """Reasons the active packs refuse this hypothesis (before anything is registered); [] when fine or no packs."""
    from qrh.packs.apply import apply_s0
    from qrh.packs.model import PackError
    from qrh.packs.runtime import PackView

    try:
        view = PackView.load(home)
        return apply_s0(view, spec, home).reasons if view else []
    except PackError as exc:
        return [f"experience packs: {exc}"]


def _check_stability(spec: HypothesisRecord, registry: Registry, home: str | Path) -> list[str]:
    """Refuse a hypothesis that sets a stability threshold against the trader's answer (the trader owns these four)."""
    from qrh.interview.stability import spec_value_problems, trader_answers
    from qrh.packs.model import PackError

    set_here = {k: v for k, v in spec.thresholds.items() if k.startswith("stability_")}
    if not set_here:
        return []
    try:
        answers = trader_answers(registry.conn, spec.resolved_family_id())
        defaults = _default_thresholds(spec, home)
    except PackError as exc:
        return [f"experience packs: {exc}"]
    return spec_value_problems(set_here, answers, defaults)


def register_new_trial(
    raw_spec: Any, registry: Registry, home: str | Path, base_dir: str | Path | None = None
) -> NewTrialResult:
    """S0 for a fresh spec: validate, register, write the hypothesis receipt.

    ``base_dir`` is the directory of the spec file; adapter modules resolve against it.
    """
    if isinstance(raw_spec, dict) and base_dir is not None:
        raw_spec = {**raw_spec, "base_dir": str(Path(base_dir).resolve())}
    spec, reasons = check_hypothesis_spec(raw_spec)
    if spec is None:
        return NewTrialResult(None, None, None, reasons)
    try:
        pin_reasons = check_pins(home, spec.methods)
    except MethodError as exc:
        pin_reasons = [str(exc)]
    if pin_reasons:
        return NewTrialResult(None, None, None, pin_reasons)
    pack_reasons = _check_packs(spec, home)
    if pack_reasons:
        return NewTrialResult(None, None, None, pack_reasons)
    stability_reasons = _check_stability(spec, registry, home)
    if stability_reasons:
        return NewTrialResult(None, None, None, stability_reasons)
    trial_id = registry.new_id()
    receipt = Pipeline(registry, home).run_stage(trial_id, Stage.S0, spec=spec)
    path = receipt_path(home, trial_id, Stage.S0)
    if receipt.verdict is not Verdict.PASS:
        return NewTrialResult(trial_id, path, receipt, receipt.reasons)
    notes = list((receipt.params.get("llm") or {}).get("notes") or [])
    unconfirmed = [n for n, v in (receipt.params.get("stability_thresholds") or {}).items()
                   if not str(v["source"]).startswith("trader answer")]
    if unconfirmed:
        notes.append(
            "stability thresholds not confirmed by the trader: " + ", ".join(unconfirmed) + " (frozen at their default or "
            "the hypothesis value). Ask the trader first next time: `qrh interview next --stage S0 --scope "
            f"strategy:{spec.resolved_family_id()}`")
    return NewTrialResult(trial_id, path, receipt, [], notes)

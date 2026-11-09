"""The benchmark suites: planted bugs, noise false positives, known-edge power.

Each runner takes ``arms`` (a name -> MethodSelector mapping, usually a candidate and the default) and runs every
arm over exactly the same generated data, so a difference between arms comes from the methods only. A runner
returns one result per arm: ``metrics`` (name -> counts with an exact interval, see ``qrh.bench.stats``) plus the
detail needed to read them. Nothing here changes a stored default.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path
from typing import Any, Callable

from qrh.adapters.base import DataView
from qrh.bench import corpus as corp
from qrh.bench import llm_corpus as llmc
from qrh.bench.engine import (
    MaGridAdapter,
    WARMUP,
    build_matrix,
    evaluate_matrix,
    make_frame,
    seed_for,
)
from qrh.bench.profiles import WALK_FORWARD
from qrh.bench.stats import metric
from qrh.checks.data_quality import run_rules
from qrh.checks.lookahead import BLOCK, WARN
from qrh.hashing import sha256_bytes
from qrh.methods.selection import MethodSelector
from qrh.validation.defaults import DEFAULT_THRESHOLDS

Arms = dict[str, MethodSelector]
DETECTORS = ("lint_block", "lint_warn", "truncation", "perturbation", "same_bar", "sanity", "quality", "pnl")
LLM_DETECTORS = ("gate", "lint", "direct", "replay", "truncation")
PARAMS = {"w": 10}


def _resolver(sel: MethodSelector) -> Callable[[Any], Any]:
    def resolve(rule_type: Any) -> Any:
        if not isinstance(rule_type, str) or not sel.registry.versions(f"s1.quality.{rule_type}"):
            return None
        return sel.get(f"s1.quality.{rule_type}").entry

    return resolve


def _detect(m: corp.Materialized, sel: MethodSelector, workdir: Path, stem: str) -> dict[str, bool]:
    """Run the harness's S1 and S2 detectors on one materialized case; True means the defect was flagged."""
    out = dict.fromkeys(DETECTORS, False)
    if m.kind in ("data", "control"):
        out["quality"] = run_rules(m.frame, corp.RULES, _resolver(sel)).verdict != "PASS"
    if m.kind in ("code", "control"):
        path = workdir / f"{stem}.py"
        path.write_text(m.source or "", encoding="utf-8")
        findings = sel.get("s2.lookahead_lint").entry([path])
        out["lint_block"] = any(f.severity == BLOCK and not f.suppressed for f in findings)
        out["lint_warn"] = any(f.severity == WARN and not f.suppressed for f in findings)
        adapter = m.adapter()
        start, end = m.frame.index[WARMUP], m.frame.index[-1]
        try:
            out["truncation"] = sel.get("s2.truncation").entry(adapter, m.frame, PARAMS, start, end).verdict != "PASS"
            full = adapter.run(PARAMS, DataView(m.frame), start, end)
            out["perturbation"] = sel.get("s2.future_perturbation").entry(
                adapter, m.frame, PARAMS, start, end, full_result=full).verdict != "PASS"
            from qrh.checks.perturbation import make_own_bar_probe

            probe = make_own_bar_probe(adapter, m.frame, PARAMS, start, end, None, full)
            out["same_bar"] = sel.get("s2.same_bar_execution").entry(full, m.frame, None, probe).verdict != "PASS"
            out["sanity"] = sel.get("s2.sanity").entry(full, corp.BOUNDS).verdict != "PASS"
            pnl = sel.selected("s2.pnl_reconstruction")  # opt-in: runs only where the arm selects it
            if pnl is not None:
                out["pnl"] = pnl.entry(full, m.frame, corp.EXECUTION).verdict != "PASS"
        except Exception:  # a case the harness cannot run is not PASS: the stage would be BLOCKED
            out["truncation"] = True
    return out


def run_planted_bugs(arms: Arms, cfg: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    reps = int(cfg["reps"])
    cases = corp.corpus()
    flagged: dict[str, dict[tuple[str, str], int]] = {a: {} for a in arms}
    by_detector: dict[str, dict[str, dict[str, int]]] = {a: {} for a in arms}
    with tempfile.TemporaryDirectory(prefix="qrh-bench-") as tmp:
        work = Path(tmp)
        for case in cases:
            for rep in range(reps):
                m = corp.materialize(case, seed_for(seed, "planted", case.bug_class, case.name, rep))
                for arm, sel in arms.items():
                    hits = _detect(m, sel, work, f"{arm}_{case.bug_class}_{case.name}_{rep}")
                    detected = (hits["lint_block"] or hits["truncation"] or hits["perturbation"] or hits["same_bar"]
                                or hits["sanity"] or hits["quality"] or hits["pnl"])
                    key = (case.bug_class, case.name)
                    flagged[arm][key] = flagged[arm].get(key, 0) + int(detected)
                    det = by_detector[arm].setdefault(case.bug_class, dict.fromkeys(DETECTORS, 0))
                    for name, hit in hits.items():
                        det[name] += int(hit)
    out: dict[str, dict[str, Any]] = {}
    for arm, sel in arms.items():
        metrics: dict[str, Any] = {}
        classes = [*corp.BUG_CLASSES, corp.CONTROLS]
        for bug in classes:
            controls = bug == corp.CONTROLS
            prefix = "false_alarm" if controls else "detect"
            direction = "lower" if controls else "higher"
            label = "false alarm rate on clean controls" if controls else f"detection rate, {bug}"
            counted = [c for c in cases if c.bug_class == bug and not c.expected_undetectable]
            hidden = [c for c in cases if c.bug_class == bug and c.expected_undetectable]
            for c in counted:
                metrics[f"{prefix}/{bug}/{c.name}"] = metric(flagged[arm][(bug, c.name)], reps, direction=direction)
            k = sum(flagged[arm][(bug, c.name)] for c in counted)
            metrics[f"{prefix}/{bug}"] = metric(k, reps * len(counted), direction=direction, gate=True, label=label)
            if hidden:  # below the rule's tolerance by design: reported on their own, never counted as misses
                for c in hidden:
                    metrics[f"expected_undetectable/{bug}/{c.name}"] = metric(flagged[arm][(bug, c.name)], reps, direction="higher")
                kh = sum(flagged[arm][(bug, c.name)] for c in hidden)
                metrics[f"expected_undetectable/{bug}"] = metric(
                    kh, reps * len(hidden), direction="higher", label=f"flagged anyway, {bug} below the tolerance")
        out[arm] = {
            "reps": reps, "metrics": metrics, "detectors": by_detector[arm],
            "variants": {b: [c.name for c in cases if c.bug_class == b] for b in classes},
            "expected_undetectable": {
                b: [c.name for c in cases if c.bug_class == b and c.expected_undetectable]
                for b in classes if any(c.bug_class == b and c.expected_undetectable for c in cases)
            },
            "bounds": corp.BOUNDS, "rules": corp.RULES, "execution": corp.EXECUTION,
            "methods": dict(sorted(sel.used.items())),
        }
    return out


def _tally(outcomes: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "pass_s4": sum(1 for o in outcomes if o["s4_pass"]), "pass_pbo": sum(1 for o in outcomes if o["pbo_pass"]),
        "pass_dsr": sum(1 for o in outcomes if o["dsr_pass"]), "pass_s5": sum(1 for o in outcomes if o["s5_pass"]),
        "pass_all": sum(1 for o in outcomes if o["pass_all"]),
    }
    counts["errors"] = sum(1 for o in outcomes if o["error"])
    return counts


def _rate_metrics(prefix_counts: dict[str, int], n: int, key: str, *, headline: str, direction: str) -> dict[str, Any]:
    """Metrics for one cell: the headline (gated) and the per-gate breakdown (information only)."""
    out = {f"{headline}/{key}": metric(prefix_counts["pass_all"], n, direction=direction, gate=True)}
    for name in ("pass_s4", "pass_pbo", "pass_dsr", "pass_s5"):
        out[f"{name}/{key}"] = metric(prefix_counts[name], n, direction=direction)
    return out


def _run_cells(arms: Arms, suite: str, cells: list[tuple[str, int, int, float]], reps: int, seed: int, headline: str,
               direction: str) -> dict[str, dict[str, Any]]:
    """Shared by the noise and power suites: ``cells`` are (key, grid size, periods, edge)."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    outcomes: dict[str, dict[str, list[dict[str, Any]]]] = {a: {key: [] for key, *_ in cells} for a in arms}
    for key, grid, periods, edge in cells:
        adapter = MaGridAdapter(grid)
        for rep in range(reps):
            frame = make_frame(periods, seed_for(seed, suite, key, rep), edge)
            matrix = build_matrix(adapter, frame)  # S3 once; every arm evaluates the same matrix
            for arm, sel in arms.items():
                outcomes[arm][key].append(evaluate_matrix(matrix, WALK_FORWARD, thresholds, grid, sel))
    out: dict[str, dict[str, Any]] = {}
    for arm, sel in arms.items():
        metrics: dict[str, Any] = {}
        cell_info: dict[str, Any] = {}
        pooled: list[dict[str, Any]] = []
        for key, grid, periods, edge in cells:
            runs = outcomes[arm][key]
            pooled += runs
            counts = _tally(runs)
            metrics.update(_rate_metrics(counts, reps, key, headline=headline, direction=direction))
            cell_info[key] = {"grid": grid, "periods": periods, "edge": edge, "reps": reps, "errors": counts["errors"]}
        counts = _tally(pooled)
        pooled_metrics = _rate_metrics(counts, len(pooled), "pooled", headline=headline, direction=direction)
        pooled_metrics[f"{headline}/pooled"]["gate"] = False
        metrics.update(pooled_metrics)
        out[arm] = {
            "metrics": metrics, "cells": cell_info, "thresholds": thresholds, "walk_forward": dict(WALK_FORWARD),
            "reps": reps, "methods": dict(sorted(sel.used.items())),
        }
    return out


def run_noise_fpr(arms: Arms, cfg: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    """Pure-noise strategy families through S3-S5: the share that passes is the empirical false-positive rate."""
    cells = [(f"g{g}_t{t}", g, t, 0.0) for g in cfg["grids"] for t in cfg["periods"]]
    out = _run_cells(arms, "noise", cells, int(cfg["reps"]), seed, headline="fpr", direction="lower")
    nominal = 1.0 - DEFAULT_THRESHOLDS["min_dsr"]  # the DSR gate's nominal level
    for res in out.values():
        res["nominal"] = nominal
        for key, info in res["cells"].items():
            ci = res["metrics"][f"fpr/{key}"]["ci"]
            info["exceeds_nominal"] = bool(ci is not None and ci[0] > nominal)
    return out


def run_known_edge_power(arms: Arms, cfg: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    """Data with a planted persistent trend of several strengths: the share that passes is the power."""
    strengths = [float(s) for s in cfg["strengths"]]
    cells = [(f"s{s:g}", int(cfg["grid"]), int(cfg["periods"]), s) for s in strengths]
    out = _run_cells(arms, "power", cells, int(cfg["reps"]), seed, headline="power", direction="higher")
    for res in out.values():
        res["strengths"] = strengths
    return out


# --- strategies that use a language model --------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _package_code_hash() -> str:
    from qrh.pipeline.stages import package_code_hash

    return package_code_hash()


def _detect_llm(m: llmc.LLMMaterialized, sel: MethodSelector, workdir: Path, stem: str) -> dict[str, bool]:
    """Run the S2 detectors for an LLM strategy the way the stage does; True means the defect was flagged.

    ``gate`` is the cutoff gate, ``lint`` the look-ahead lint and the undeclared-LLM lint (LLM001), ``direct`` the
    advisory direct-client lint (LLM002, declared adapters only), ``replay`` a
    replay-cache miss during a truncated run (a prompt that differs between full and truncated data) and
    ``truncation`` any failure of the truncation test. Like S2, a case whose gate did not pass is not run."""
    from qrh.llm import llm_session, register_backend
    from qrh.llm.lint import lint_llm_direct, lint_llm_use
    from qrh.llm.runtime import require_bound

    out = dict.fromkeys(LLM_DETECTORS, False)
    path = workdir / f"{stem}.py"
    path.write_text(m.source, encoding="utf-8")
    findings = sel.get("s2.lookahead_lint").entry([path])
    if m.spec is None:
        findings = [*findings, *lint_llm_use([path])]
    out["lint"] = any(f.severity == BLOCK and not f.suppressed for f in findings)
    if m.spec is None:
        return out  # undeclared: nothing to replay (and the libraries it imports are not installed)
    files = [path]
    if m.helper is not None:
        helper = workdir / f"{stem}_helper.py"
        helper.write_text(m.helper, encoding="utf-8")
        files.append(helper)
    out["direct"] = any(not f.suppressed for f in lint_llm_direct(files))
    gate = sel.get("s2.llm_gate").entry(m.spec, m.windows, m.margin)
    out["gate"] = gate.verdict != "PASS"
    if out["gate"]:
        return out
    register_backend(llmc.PROVIDER, llmc.BACKEND)
    adapter = m.adapter()
    start, end = m.frame.index[WARMUP], m.frame.index[-1]
    # no catalog dataset here: the dataset hash is the hash of the synthetic frame this case generated
    dataset_hash, code_hash = sha256_bytes(m.frame.to_csv().encode()), _package_code_hash()
    require_bound(dataset_hash, code_hash)  # refuse, outside the try below: never pass the placeholder on
    try:
        with llm_session("record", workdir / f"{stem}-cache", {llmc.MODEL: llmc.PROVIDER},
                         dataset_hash=dataset_hash, code_hash=code_hash):
            full = adapter.run(PARAMS_LLM, DataView(m.frame), start, end)
            tr = sel.get("s2.truncation").entry(adapter, m.frame, PARAMS_LLM, start, end, full_result=full)
    except Exception:  # a case the harness cannot run is not PASS: the stage would be BLOCKED
        out["truncation"] = True
        return out
    out["truncation"] = tr.verdict != "PASS"
    out["replay"] = any(r.startswith("prompt differs between full and truncated data") for r in tr.reasons)
    return out


PARAMS_LLM = {"lookback": 20}


def run_llm_gate(arms: Arms, cfg: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    """Planted bugs for strategies that use a language model; same result shape as ``run_planted_bugs``."""
    reps = int(cfg["reps"])
    cases = llmc.llm_corpus()
    flagged: dict[str, dict[tuple[str, str], int]] = {a: {} for a in arms}
    by_detector: dict[str, dict[str, dict[str, int]]] = {a: {} for a in arms}
    with tempfile.TemporaryDirectory(prefix="qrh-bench-llm-") as tmp:
        work = Path(tmp)
        for case in cases:
            for rep in range(reps):
                m = llmc.materialize_llm(case, seed_for(seed, "llm-gate", case.bug_class, case.name, rep))
                for arm, sel in arms.items():
                    hits = _detect_llm(m, sel, work, f"{arm}_{case.bug_class}_{case.name}_{rep}")
                    key = (case.bug_class, case.name)
                    flagged[arm][key] = flagged[arm].get(key, 0) + int(any(hits.values()))
                    det = by_detector[arm].setdefault(case.bug_class, dict.fromkeys(LLM_DETECTORS, 0))
                    for name, hit in hits.items():
                        det[name] += int(hit)
    out: dict[str, dict[str, Any]] = {}
    classes = [*llmc.LLM_CLASSES, llmc.CONTROLS]
    for arm, sel in arms.items():
        metrics: dict[str, Any] = {}
        for bug in classes:
            controls = bug == llmc.CONTROLS
            prefix = "false_alarm" if controls else "detect"
            direction = "lower" if controls else "higher"
            label = "false alarm rate on clean controls" if controls else f"detection rate, {bug}"
            counted = [c for c in cases if c.bug_class == bug]
            for c in counted:
                metrics[f"{prefix}/{bug}/{c.name}"] = metric(flagged[arm][(bug, c.name)], reps, direction=direction)
            k = sum(flagged[arm][(bug, c.name)] for c in counted)
            metrics[f"{prefix}/{bug}"] = metric(k, reps * len(counted), direction=direction, gate=True, label=label)
        out[arm] = {
            "reps": reps, "metrics": metrics, "detectors": by_detector[arm],
            "variants": {b: [c.name for c in cases if c.bug_class == b] for b in classes},
            "expected_undetectable": {}, "margin_days": llmc.MARGIN_DAYS, "methods": dict(sorted(sel.used.items())),
        }
    return out


# --- dataset onboarding --------------------------------------------------------------------------------------


def _verify_case(cat: Any, name: str, proposal: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """The harness's own propose and verify steps; returns each claimed field's status and the contradictions."""
    from qrh.onboarding import workflow
    from qrh.onboarding.model import parse_proposal

    workflow.propose(cat, name, parse_proposal(proposal))
    out = workflow.verify(cat, name)
    return {row["field"]: row["status"] for row in out["fields"]}, list(out["contradictions"])


ONBOARDING_STATUSES = ("contradicted", "unverified", "verified")


def run_onboarding(arms: Arms, cfg: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    """Known-correct and planted-wrong instrument proposals through onboarding verification.

    A planted case counts as detected when the check of the field it plants is not ``verified`` (contradicted, or
    left unverified for a human). A false alarm is any contradiction on a correct proposal, on any field. The
    verifier is deterministic code and not a selectable method, so every arm gets the same result."""
    from qrh.bench import onboarding_corpus as oc
    from qrh.memory.catalog import Catalog
    from qrh.synthetic.instruments import write_csv

    reps = int(cfg["reps"])
    cases = oc.corpus()
    flagged: dict[tuple[str, str], int] = {}
    status_counts: dict[str, dict[str, int]] = {b: dict.fromkeys(ONBOARDING_STATUSES, 0) for b in (*oc.CLASSES, oc.CONTROLS)}
    with tempfile.TemporaryDirectory(prefix="qrh-bench-onboarding-") as tmp:
        work = Path(tmp)
        for rep in range(reps):
            cat = Catalog(work / f"registry_{rep}.sqlite3")
            registered: dict[tuple[str, tuple[tuple[str, Any], ...]], tuple[str, dict[str, Any]]] = {}
            for case in cases:
                key = (case.dataset, case.kwargs)
                if key not in registered:
                    frame, correct = oc.MAKERS[case.dataset](seed=seed_for(seed, "onboarding", case.dataset, rep), **dict(case.kwargs))
                    ds_name = f"{case.dataset}_{len(registered)}"
                    cat.register_dataset(write_csv(frame, work / f"{ds_name}_{rep}.csv"), name=ds_name, privacy="public")
                    registered[key] = (ds_name, correct)
                ds_name, correct = registered[key]
                fields, contradictions = _verify_case(cat, ds_name, case.plant(correct))
                if case.bug_class == oc.CONTROLS:
                    hit = bool(contradictions)
                    status = "contradicted" if hit else "verified"
                else:
                    status = fields.get(case.field, "unverified")  # a dropped claim is not verified
                    hit = status != "verified"
                flagged[(case.bug_class, case.name)] = flagged.get((case.bug_class, case.name), 0) + int(hit)
                status_counts[case.bug_class][status] += 1
            cat.conn.close()
    metrics: dict[str, Any] = {}
    for bug in (*oc.CLASSES, oc.CONTROLS):
        controls = bug == oc.CONTROLS
        prefix = "false_alarm" if controls else "detect"
        direction = "lower" if controls else "higher"
        label = "false alarm rate on correct proposals" if controls else f"detection rate, {bug}"
        counted = [c for c in cases if c.bug_class == bug]
        for c in counted:
            metrics[f"{prefix}/{bug}/{c.name}"] = metric(flagged[(bug, c.name)], reps, direction=direction)
        k = sum(flagged[(bug, c.name)] for c in counted)
        metrics[f"{prefix}/{bug}"] = metric(k, reps * len(counted), direction=direction, gate=True, label=label)
    out: dict[str, dict[str, Any]] = {}
    for arm, sel in arms.items():
        out[arm] = {
            "reps": reps, "metrics": metrics, "detectors": status_counts, "variants": oc.variants(cases),
            "expected_undetectable": {}, "methods": dict(sorted(sel.used.items())),
        }
    return out


# --- parameter stability -------------------------------------------------------------------------------------


STABILITY_DETECTORS = ("neighbour", "plateau", "not_applicable", "blocked")


def run_param_stability(arms: Arms, cfg: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    """Planted spikes (the check must FAIL) and plateaus (it must PASS), for a grid and for one pre-registered setting.

    A spike counts as detected when the check fails; a plateau is a false alarm when the check does not pass (a fail or
    a not-applicable). The method under test is ``s5.param_stability`` of each arm, with the default thresholds."""
    from qrh.bench import stability_corpus as sc
    from qrh.validation.param_stability import StabilityBlocked, StabilityThresholds

    reps = int(cfg["reps"])
    cases = sc.corpus()
    thresholds = StabilityThresholds()
    flagged: dict[str, dict[tuple[str, str], int]] = {a: {} for a in arms}
    by_detector: dict[str, dict[str, dict[str, int]]] = {a: {} for a in arms}
    for case in cases:
        for rep in range(reps):
            kwargs = sc.materialize(case, seed_for(seed, "param-stability", case.bug_class, case.name, rep))
            for arm, sel in arms.items():
                det = by_detector[arm].setdefault(case.bug_class, dict.fromkeys(STABILITY_DETECTORS, 0))
                try:
                    res = sel.get("s5.param_stability").entry(thresholds=thresholds, **kwargs)
                except StabilityBlocked:  # refused (for example hairline neighbours): the stage would be BLOCKED, not PASS
                    status = "BLOCKED"
                    det["blocked"] += 1
                else:
                    status = res.status
                    det["neighbour"] += int(res.neighbour["passed"] is False)
                    det["plateau"] += int(res.plateau["passed"] is False)
                    det["not_applicable"] += int(status == "NOT_APPLICABLE")
                hit = status in ("FAIL", "BLOCKED") if case.bug_class != sc.CONTROLS else status != "PASS"
                key = (case.bug_class, case.name)
                flagged[arm][key] = flagged[arm].get(key, 0) + int(hit)
    out: dict[str, dict[str, Any]] = {}
    classes = [*sc.CLASSES, sc.CONTROLS]
    for arm, sel in arms.items():
        metrics: dict[str, Any] = {}
        for bug in classes:
            controls = bug == sc.CONTROLS
            prefix = "false_alarm" if controls else "detect"
            direction = "lower" if controls else "higher"
            label = "false alarm rate on planted plateaus" if controls else f"detection rate, {bug}"
            counted = [c for c in cases if c.bug_class == bug]
            for c in counted:
                metrics[f"{prefix}/{bug}/{c.name}"] = metric(flagged[arm][(bug, c.name)], reps, direction=direction)
            k = sum(flagged[arm][(bug, c.name)] for c in counted)
            metrics[f"{prefix}/{bug}"] = metric(k, reps * len(counted), direction=direction, gate=True, label=label)
        out[arm] = {
            "reps": reps, "metrics": metrics, "detectors": by_detector[arm], "variants": sc.variants(cases),
            "expected_undetectable": {}, "thresholds": thresholds.doc(), "methods": dict(sorted(sel.used.items())),
        }
    return out

"""Run benchmark suites and build the bench receipt."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from qrh.bench import suites as su
from qrh.bench.history import run_history
from qrh.bench.profiles import DEFAULT_SEED, OPTIONAL_SUITES, SUITES, resolve_profile
from qrh.bench.receipt import BenchReceipt
from qrh.bench.verdict import run_verdicts
from qrh.hashing import hash_json
from qrh.methods.registry import parse_ref
from qrh.methods.selection import MethodSelector
from qrh.pipeline.receipt import collect_environment

RUNNERS = {
    "planted-bugs": su.run_planted_bugs,
    "noise-fpr": su.run_noise_fpr,
    "known-edge-power": su.run_known_edge_power,
    "llm-gate": su.run_llm_gate,
    "onboarding": su.run_onboarding,
    "param-stability": su.run_param_stability,
}


def parse_overrides(items: Sequence[str]) -> dict[str, str]:
    """``["s5.pbo=cscv@2", ...]`` -> ``{"s5.pbo": "cscv@2"}``; a repeated id is an error."""
    out: dict[str, str] = {}
    for item in items:
        mid, version = parse_ref(item)
        if mid in out:
            raise ValueError(f"method {mid} is overridden twice")
        out[mid] = version
    return out


def run_bench(
    home: str | Path,
    *,
    suites: Sequence[str] | None = None,
    profile: str = "quick",
    reps: int | None = None,
    seed: int = DEFAULT_SEED,
    overrides: Mapping[str, str] | None = None,
    compare_to_default: bool = False,
    history_path: str | Path | None = None,
    config: dict[str, dict[str, Any]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BenchReceipt:
    """Run the suites and return the (unwritten) receipt.

    ``overrides`` selects non-default method versions for the candidate arm; with ``compare_to_default`` the
    default arm runs over exactly the same data and is recorded beside it. ``config`` replaces the profile's
    suite configuration (tests use it to stay small). ``progress`` receives one line per suite as it starts.
    """
    chosen = list(suites) if suites else list(SUITES)
    unknown = [s for s in chosen if s not in SUITES + OPTIONAL_SUITES]
    if unknown:
        raise ValueError(f"unknown suite(s) {unknown}; use {list(SUITES + OPTIONAL_SUITES)}")
    overrides = dict(overrides or {})
    if compare_to_default and not overrides:
        raise ValueError("--compare-to default needs at least one method override (--method)")
    if "history" in chosen:
        if history_path is None:
            raise ValueError("the history suite needs a registry path (--history PATH)")
        if not overrides:
            raise ValueError("the history suite needs a method override (--method) to compare with the default")
    cfg = config if config is not None else resolve_profile(profile, reps)
    candidate = MethodSelector(home, overrides=overrides)
    baseline = MethodSelector(home)
    candidate.full_map()  # validates every override against the registry before any work
    arms = {"candidate": candidate, **({"baseline": baseline} if compare_to_default else {})}

    results: dict[str, Any] = {}
    baseline_results: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for suite in chosen:
        if progress is not None:
            size = cfg.get(suite, {}).get("reps")
            progress(f"{suite}: running" + (f" ({size} replications per cell)" if size else ""))
        if suite == "history":
            hist = run_history(candidate, baseline, history_path)  # type: ignore[arg-type]
            results[suite] = hist
            hashes["history_input"] = hist["input_hash"]
            continue
        out = RUNNERS[suite](arms, cfg[suite], seed)
        results[suite] = out["candidate"]
        if compare_to_default:
            baseline_results[suite] = out["baseline"]
    used_cfg = {s: cfg[s] for s in chosen if s in cfg}
    from qrh.pipeline.stages import package_code_hash

    hashes.update(config=hash_json(used_cfg), results=hash_json(results))
    if compare_to_default:
        hashes["baseline_results"] = hash_json(baseline_results)
    verdict, suite_verdicts = run_verdicts(results)
    return BenchReceipt(
        verdict=verdict, suite_verdicts=suite_verdicts,
        profile=profile if config is None else "custom", suites=chosen, seed=seed, config=used_cfg,
        methods=candidate.full_map(), overrides=overrides,
        non_default_methods={i: v for i, v in sorted(candidate.full_map().items()) if v != baseline.default_version(i)},
        baseline_methods=baseline.full_map() if compare_to_default else None,
        results=results, baseline_results=baseline_results if compare_to_default else None,
        hashes=hashes, code_hash=package_code_hash(), environment=collect_environment({"bench": seed}),
    )

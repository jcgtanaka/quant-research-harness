"""S7 report: a markdown summary built only from the trial's hypothesis and the stage receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qrh.pipeline.receipt import Receipt, Stage, Verdict, display_verdict, receipt_path

STAGE_TITLES = {
    Stage.S0: "Hypothesis registration",
    Stage.S1: "Data check",
    Stage.S2: "Implementation checks",
    Stage.S3: "In-sample search",
    Stage.S4: "Walk-forward",
    Stage.S5: "Overfitting statistics",
    Stage.S6: "Uncertainty and costs",
}


def report_path(home: str | Path, trial_id: str) -> Path:
    return Path(home) / "reports" / f"{trial_id}.md"


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def collect_methods(results: dict[Stage, Receipt]) -> tuple[dict[str, str], dict[str, str]]:
    """Methods used across the stage receipts of a run, and those flagged as not the default."""
    methods: dict[str, str] = {}
    flagged: dict[str, str] = {}
    for stage in sorted(results, key=lambda s: s.value):
        methods.update(results[stage].methods)
        flagged.update(results[stage].params.get("non_default_methods") or {})
    return dict(sorted(methods.items())), dict(sorted(flagged.items()))


def build_report(
    trial: dict[str, Any],
    results: dict[Stage, Receipt],
    home: str | Path,
    overall: Verdict,
    failing: str | None,
    conditions: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"# Research report: {trial['trial_id']}",
        "",
        f"**Final verdict: {display_verdict(overall)}**" + (f" (first non-PASS stage: {failing})" if failing else ""),
        "",
        "## Hypothesis",
        "",
        f"- Text: {trial['text']}",
        f"- Family: {trial['family_id']}",
        f"- Strategy family / asset class / frequency: "
        f"{trial['strategy_family']} / {trial['asset_class']} / {trial['frequency']}",
        f"- Declared parameter space size: {trial['parameter_space_size']}",
        f"- Difference statement: {trial['difference_statement']}",
        f"- Hypothesis hash: `{trial['hypothesis_hash']}`",
        "",
        "## Thresholds",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in sorted(trial["thresholds"].items())]
    if conditions:
        lines += ["", "## Conditions", "", "Every gate passed, but these conditions are open (answer them or accept the risk):", ""]
        for c in conditions:
            lines.append(f"- {c['question_id']} ({c['status']}, {c['stage']}, {c['scope']}): {c['text']}"
                         + (f" Risk: {c['risk']}" if c.get("risk") else ""))
    lessons = (results.get(Stage.S0).params.get("lessons") if results.get(Stage.S0) else None) or []
    if lessons:
        lines += ["", "## Lessons triggered", ""]
        lines += [f"- [{x['severity']}] {x['id']} ({x['pack']}, {x['action']}): {x['statement']}" for x in lessons]
    methods, flagged = collect_methods(results)
    if methods:
        lines += ["", "## Methods", ""]
        for mid, version in methods.items():
            note = " EXPERIMENTAL: not the default version of this method" if mid in flagged else ""
            lines.append(f"- {mid}={version}{note}")
    lines += ["", "## Stages", ""]
    for stage in sorted(STAGE_TITLES, key=lambda s: s.value):
        title = f"### {stage.value} {STAGE_TITLES[stage]}"
        r = results.get(stage)
        if r is None:
            lines += [title, "", "Verdict: NOT RUN", ""]
            continue
        lines += [title, "", f"Verdict: **{display_verdict(r.verdict)}**", ""]
        lines += [f"- Reason: {x}" for x in r.reasons] or ["- No issues recorded."]
        if r.metrics:
            lines += ["- Metrics:"] + [f"  - {k}: {_fmt(v)}" for k, v in sorted(r.metrics.items())]
        if r.trial_family_count is not None:
            lines.append(f"- Trial family count: {r.trial_family_count}")
        seeds = r.environment.get("seeds") or {}
        if seeds:
            lines.append("- Seeds: " + ", ".join(f"{k}={v}" for k, v in sorted(seeds.items())))
        lines += [
            f"- Receipt hash: `{r.receipt_hash()}`",
            f"- Receipt: `{receipt_path(home, r.trial_id, stage)}`",
            "",
        ]
    lines += [
        "## Disclaimer",
        "",
        "Research tooling output only; not investment advice. Verdicts come from deterministic checks "
        "against thresholds frozen at registration.",
        "",
    ]
    return "\n".join(lines)

"""Which figures each stage gets. One list drives both the PDF pages and the chart JSON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from matplotlib.figure import Figure

from qrh.pipeline.receipt import Stage
from qrh.reports import chartdata, charts
from qrh.reports.inputs import ReportInputError, StageData


@dataclass(frozen=True)
class FigureSpec:
    key: str
    chart: Callable[[], Figure]
    data: Callable[[], dict[str, Any]]


def _spec(key: str, name: str, data: StageData, *args: Any) -> FigureSpec:
    return FigureSpec(
        key,
        lambda: getattr(charts, name)(data, *args),
        lambda: getattr(chartdata, name)(data, *args),
    )


def _need(data: StageData, *names: str) -> None:
    missing = [n for n in names if n not in data.artifacts]
    if missing:
        raise ReportInputError(f"{data.stage.value}: receipt does not record artifact(s) {missing}")


def figure_specs(data: StageData) -> list[FigureSpec]:
    """Figures for a stage. A stage that recorded no artifacts (for example one that was BLOCKED
    before producing evidence) gets a status card; a stage missing part of its evidence is an error."""
    stage = data.stage
    if stage is Stage.S7:
        return [_spec("s7_summary", "s7_summary", data)]
    if not data.artifacts and stage is not Stage.S0:
        return [_spec("status_card", "status_card", data)]
    if stage is Stage.S0:
        _need(data, "hypothesis")
        return [_spec("s0_card", "s0_card", data)]
    if stage is Stage.S1:
        _need(data, "coverage")
        specs = [_spec("s1_coverage", "s1_coverage", data), _spec("s1_rules", "s1_rules", data)]
        for i in range(chartdata.onboarding_count(data)):  # a dataset card and an onboarding table per profiled dataset
            specs += [_spec(f"s1_dataset_card_{i}", "s1_dataset_card", data, i), _spec(f"s1_onboarding_{i}", "s1_onboarding", data, i)]
        return specs
    if stage is Stage.S2:
        _need(data, "lint", "truncation", "sanity")
        specs = [_spec("s2_lint", "s2_lint", data)]
        if "llm_gate" in data.artifacts:  # a strategy that declares a language model
            specs.append(_spec("s2_llm_gate", "s2_llm_gate", data))
        specs += [_spec(f"s2_truncation_{i}", "s2_truncation", data, i) for i in range(chartdata.truncation_count(data))]
        return specs + [_spec("s2_sanity", "s2_sanity", data)]
    if stage is Stage.S3:
        _need(data, "combos")
        primary = chartdata.s3_primary_key(data)
        return [_spec(primary, primary, data), _spec("s3_hist", "s3_hist", data)]
    if stage is Stage.S4:
        _need(data, "folds", "oos_equity")
        return [_spec("s4_folds", "s4_folds", data), _spec("s4_equity", "s4_equity", data), _spec("s4_params", "s4_params", data)]
    if stage is Stage.S5:
        _need(data, "dsr_inputs")
        if "pbo_logits" not in data.artifacts and data.artifacts["dsr_inputs"].get("pbo") is not None:
            _need(data, "pbo_logits")  # PBO was computed, so its distribution must have been recorded
        lam = [_spec("s5_lambda", "s5_lambda", data)] if "pbo_logits" in data.artifacts else []  # none for one combination
        stab = [_spec("s5_stability", "s5_stability", data)] if "param_stability" in data.artifacts else []  # not in old runs
        return lam + [_spec("s5_dsr", "s5_dsr", data)] + stab
    if stage is Stage.S6:
        _need(data, "bootstrap_samples", "bootstrap_summary", "equity_compare")
        return [_spec("s6_bootstrap", "s6_bootstrap", data), _spec("s6_equity", "s6_equity", data)]
    raise ReportInputError(f"no figures defined for stage {stage.value}")

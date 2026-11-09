"""The planted-case corpus for dataset onboarding (the ``onboarding`` suite).

Every case is a synthetic dataset from ``qrh.synthetic.instruments`` plus one instrument-model proposal. The
controls carry the known-correct proposal of each generator. A planted case replaces exactly one claim (or drops one)
with a wrong value, and names the field whose verification result decides the outcome. Nothing here verifies
anything: ``qrh.bench.suites.run_onboarding`` runs the harness's own onboarding propose and verify steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from qrh.synthetic import instruments as inst
from qrh.synthetic.instruments import Proposal, with_claim, without_claim

CONTROLS = "controls"
CLASSES = ("wrong_units", "wrong_asset_class", "adjustments_none_on_split", "missing_option_mapping", "wrong_frequency")
DATASETS = ("equity", "futures", "fx", "option", "perp")
MAKERS: dict[str, Callable[..., tuple[pd.DataFrame, Proposal]]] = {
    "equity": inst.equity_daily, "futures": inst.futures_daily, "fx": inst.fx_quotes,
    "option": inst.option_snapshots, "perp": inst.crypto_perp,
}
TRUE_CLASS = {"equity": "equity", "futures": "future", "fx": "fx_spot", "option": "option", "perp": "crypto_perp"}
WRONG_CLASSES = ("equity", "future", "fx_spot", "option", "crypto_perp")  # every class the verifier knows by structure
WRONG_UNITS = "fraction"  # the synthetic prices are levels well above one
DAILY = "1D"
WRONG_FREQUENCY = {"equity": "1h", "futures": "1h", "fx": DAILY, "option": "1min", "perp": DAILY}


@dataclass(frozen=True)
class OnboardingCase:
    bug_class: str  # one of CLASSES or CONTROLS
    name: str
    dataset: str  # a key of MAKERS
    field: str  # the field whose check decides a planted case
    plant: Callable[[Proposal], Proposal]
    kwargs: tuple[tuple[str, Any], ...] = ()


def _same(p: Proposal) -> Proposal:
    return p


def corpus() -> list[OnboardingCase]:
    cases: list[OnboardingCase] = []
    for ds in DATASETS:
        cases.append(OnboardingCase(CONTROLS, ds, ds, "", _same))
    for ds in DATASETS:
        cases.append(OnboardingCase("wrong_units", ds, ds, "price_units", lambda p: with_claim(p, "price_units", WRONG_UNITS)))
    for ds in DATASETS:
        for cls in WRONG_CLASSES:
            if cls != TRUE_CLASS[ds]:
                cases.append(OnboardingCase("wrong_asset_class", f"{ds}_as_{cls}", ds, "asset_class",
                                            lambda p, c=cls: with_claim(p, "asset_class", c)))
    # the correct description of a split-affected, unadjusted file is "none"; verification must refuse it
    cases.append(OnboardingCase("adjustments_none_on_split", "equity_with_split", "equity", "adjustments", _same,
                                (("splits", True),)))
    cases.append(OnboardingCase("missing_option_mapping", "option_without_mapping", "option", "asset_class",
                                lambda p: without_claim(p, "option_mapping")))
    for ds in DATASETS:
        cases.append(OnboardingCase("wrong_frequency", ds, ds, "frequency",
                                    lambda p, f=WRONG_FREQUENCY[ds]: with_claim(p, "frequency", f)))
    return cases


def variants(cases: list[OnboardingCase]) -> dict[str, list[str]]:
    return {b: [c.name for c in cases if c.bug_class == b] for b in (*CLASSES, CONTROLS)}

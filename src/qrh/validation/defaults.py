"""Conservative default thresholds for the validation stages (S3-S6).

At S0 registration, any threshold the hypothesis does not set is filled from
here and stored explicitly, so the hypothesis receipt shows every threshold
that will be used. Thresholds are frozen with the hypothesis.

- max_pbo: largest tolerated probability of backtest overfitting (CSCV).
- min_dsr: smallest deflated Sharpe ratio (a probability) to accept.
- min_oos_sharpe: smallest annualized out-of-sample (walk-forward) Sharpe.
- min_sharpe_lower_ci: smallest lower confidence bound of the OOS Sharpe.
- min_cost_stressed_sharpe: smallest OOS Sharpe after multiplying costs.
- bootstrap_reps: stationary bootstrap resamples.
- ci_level: two-sided confidence level for bootstrap intervals.
- cost_stress_multiplier: factor passed to the adapter as ``cost_multiplier``.
- cscv_blocks: number of contiguous blocks S for PBO (even).
- llm_cutoff_margin_days: safety margin added to the latest declared training cutoff of a language model; every
  evaluation window must start strictly after that date (used only by strategies that declare ``llm``; tighten
  freely by raising it, loosen only through the approval flow).
- perturbation_cutoffs: number of cutoff bars of the S2 future-perturbation test; each costs one extra adapter run
  (raise it freely, lower it only through the approval flow).
- perturbation_change_cap: most position-change bars the same test covers (each with the bar before it); beyond it they
  are spread evenly and the readiness report warns that coverage is partial (raise it freely, lower it only through the
  approval flow).
- stability_closeness: a nearby setting counts as "about as good" when its out-of-sample Sharpe is at least this share of
  the SELECTED setting's (S5 local plateau share). Higher is tighter.
- stability_plateau_share: smallest share of the nearby numeric settings that must be that close. Higher is tighter.
- stability_neighbour_ratio: smallest ratio, for the worst parameter, of the mean out-of-sample Sharpe of the nearby settings
  to the selected one's (S5 neighbour check). Higher is tighter.
- stability_steps: relative steps (+/-) the harness uses to build the nearby settings, for a single combination and a grid
  alike. Unordered: more or wider steps are not simply stricter.
  The four stability thresholds are ALWAYS asked of the trader at intake (``qrh.interview.stability``); these are the
  defaults that apply when a question is unanswered or answered "I don't know", and the readiness report then warns.
"""

from __future__ import annotations

import copy
from typing import Any

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "max_pbo": 0.2,
    "min_dsr": 0.95,
    "min_oos_sharpe": 0.0,
    "min_sharpe_lower_ci": 0.0,
    "min_cost_stressed_sharpe": 0.0,
    "bootstrap_reps": 2000,
    "ci_level": 0.95,
    "cost_stress_multiplier": 2.0,
    "cscv_blocks": 16,
    "llm_cutoff_margin_days": 180,
    "perturbation_cutoffs": 20,
    "perturbation_change_cap": 40,
    "stability_closeness": 0.70,
    "stability_plateau_share": 0.50,
    "stability_neighbour_ratio": 0.50,
    "stability_steps": [0.10, 0.20],
}


def fill_thresholds(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: the given thresholds plus every missing default."""
    return {**copy.deepcopy(DEFAULT_THRESHOLDS), **thresholds}

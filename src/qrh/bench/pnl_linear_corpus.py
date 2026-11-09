"""Bench corpus for the pnl.linear verdict: honest cases and planted defects with known truth.

Synthetic data only. The true per-bar PnL comes from the cash-and-quantity ledger
(``qrh.bench.pnl_ledger``); the explained series passed to the verdict is built here
from the schedule alone (prior quantity times price change, minus the declared cost per fill), so the
bench does not share code with the reconstruction. The verdict module never imports the ledger.

Cases (each built per seed): ``buy_and_hold``, ``rebalancing``, ``declared_halt``.
Defects applied to an honest case:
    gain_<k>bp   hidden constant gain of k basis points of capital per bar added to reported PnL.
    stale_mark   publication time frozen for a window of held bars (price held, stale mark).
    future_mark  publication time later than the bar time on a few held bars.
    cost_free    reported PnL computed with zero cost while the schedule has turnover
                 (not applied to buy_and_hold: a single entry fill is below any sensible cap).
Expected verdicts are stated on each case and are never PASS for a defect.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qrh.bench.pnl_ledger import plant_hidden_gain, run_ledger
from qrh.checks.pnl_linear import evaluate_pnl_linear
from qrh.checks.pnl_linear_bounds import NOISE_SCALED

CASE_KINDS = ("buy_and_hold", "rebalancing", "declared_halt")
GAIN_BPS = (1, 2, 5, 10)

N_BARS = 250
BARS_PER_YEAR = 252.0
CAPITAL = 100_000.0
BASE_QTY = 500.0
REBAL_QTY = 200.0
REBAL_EVERY = 10
COMMISSION = 50.0  # declared cost per fill, in currency units
HALT_WINDOW = (100, 105)
STALE_WINDOW = (150, 155)
FUTURE_WINDOW = (160, 163)
FRAME_COLUMNS = frozenset({"close", "pub", "halt"})
_OFFSET = {"buy_and_hold": 0, "rebalancing": 1, "declared_halt": 2}


@dataclass
class BenchCase:
    kind: str
    seed: int
    defect: str
    expected: tuple[str, ...]
    reported: np.ndarray
    explained: np.ndarray
    bar_times: pd.DatetimeIndex
    publication_times: pd.DatetimeIndex
    position: np.ndarray
    halt_flag: np.ndarray | None
    has_turnover: bool
    gain_bp: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def remainder(self) -> np.ndarray:
        return self.reported - self.explained


def _prices(seed: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng([seed, _OFFSET[kind]])
    p = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, N_BARS)))
    if kind == "declared_halt":
        a, b = HALT_WINDOW
        p[a:b] = p[a - 1]  # no trading: the price is held for the halt window
    return p


def _schedule(kind: str, prices: np.ndarray) -> list[tuple[int, float, float]]:
    sched = [(0, BASE_QTY, float(prices[0]))]
    if kind == "buy_and_hold":
        return sched
    sign = -1.0
    for bar in range(REBAL_EVERY, N_BARS, REBAL_EVERY):
        if kind == "declared_halt" and HALT_WINDOW[0] <= bar < HALT_WINDOW[1]:
            continue
        sched.append((bar, sign * REBAL_QTY, float(prices[bar])))
        sign = -sign
    return sched


def _explained(prices: np.ndarray, schedule, commission: float) -> tuple[np.ndarray, np.ndarray]:
    """Prior quantity times price change, minus declared cost per fill; and the position after fills."""
    dq = np.zeros(N_BARS)
    fills = np.zeros(N_BARS)
    for bar, q, _ in schedule:
        dq[bar] += q
        fills[bar] += 1
    position = np.cumsum(dq)
    prior = np.concatenate(([0.0], position[:-1]))
    dp = np.concatenate(([0.0], np.diff(prices)))
    return prior * dp - commission * fills, position


def make_honest_case(kind: str, seed: int) -> BenchCase:
    if kind not in CASE_KINDS:
        raise ValueError(f"unknown case kind {kind!r}")
    prices = _prices(seed, kind)
    sched = _schedule(kind, prices)
    ledger = run_ledger(prices, sched, CAPITAL, commission_per_trade=COMMISSION, seed=seed)
    explained, position = _explained(prices, sched, COMMISSION)
    times = pd.bdate_range("2024-01-01", periods=N_BARS)
    pub = times.copy()
    halt = None
    if kind == "declared_halt":
        a, b = HALT_WINDOW
        pub = pub.copy().to_series().reset_index(drop=True)
        pub.iloc[a:b] = times[a - 1]
        pub = pd.DatetimeIndex(pub)
        halt = np.zeros(N_BARS, dtype=bool)
        halt[a:b] = True
    expected = ("INCONCLUSIVE",) if kind == "declared_halt" else ("PASS", "INCONCLUSIVE")
    return BenchCase(
        kind=kind, seed=seed, defect="honest", expected=expected,
        reported=np.array(ledger.pnl), explained=explained, bar_times=times,
        publication_times=pub, position=position, halt_flag=halt,
        has_turnover=len(sched) > 1, extra={"prices": prices, "schedule": sched},
    )


def _with(case: BenchCase, defect: str, **over) -> BenchCase:
    fields = {**case.__dict__, "defect": defect, "expected": ("FAIL",), **over}
    return BenchCase(**fields)


def gain_defect(case: BenchCase, gain_bp: float) -> BenchCase:
    ledger = run_ledger(case.extra["prices"], case.extra["schedule"], CAPITAL,
                        commission_per_trade=COMMISSION, seed=case.seed)
    reported, _ = plant_hidden_gain(ledger, gain_bp * 1e-4 * CAPITAL)
    return _with(case, f"gain_{gain_bp:g}bp", reported=reported, gain_bp=float(gain_bp))


def stale_defect(case: BenchCase) -> BenchCase:
    a, b = STALE_WINDOW
    pub = case.publication_times.to_series().reset_index(drop=True)
    pub.iloc[a:b] = case.bar_times[a - 1]
    return _with(case, "stale_mark", publication_times=pd.DatetimeIndex(pub))


def future_defect(case: BenchCase) -> BenchCase:
    a, b = FUTURE_WINDOW
    pub = case.publication_times.to_series().reset_index(drop=True)
    pub.iloc[a:b] = case.bar_times[a:b] + pd.Timedelta(days=1)
    return _with(case, "future_mark", publication_times=pd.DatetimeIndex(pub))


def cost_free_defect(case: BenchCase) -> BenchCase:
    ledger = run_ledger(case.extra["prices"], case.extra["schedule"], CAPITAL,
                        commission_per_trade=0.0, seed=case.seed)
    return _with(case, "cost_free", reported=np.array(ledger.pnl))


def build_corpus(seeds, gain_bps=GAIN_BPS) -> list[BenchCase]:
    out: list[BenchCase] = []
    for seed in seeds:
        for kind in CASE_KINDS:
            honest = make_honest_case(kind, seed)
            out.append(honest)
            out.extend(gain_defect(honest, g) for g in gain_bps)
            out.append(stale_defect(honest))
            out.append(future_defect(honest))
            if honest.has_turnover:
                out.append(cost_free_defect(honest))
    return out


def _block(max_bp: float, max_share: float) -> dict[str, Any]:
    return {
        "position_units": "quantity",
        "cost_model": {"commission_per_fill": COMMISSION},
        "asset_class": "equity",
        "min_cost_per_turnover": 0.0,
        "publication_column": "pub",
        "max_staleness": 2,
        "adjustment": {"close": "total_return"},
        "halt_flag_column": "halt",
        "max_unexplained_bp_per_year": max_bp,
        "max_unexplained_share": max_share,
    }


def evaluate_case(case: BenchCase, max_bp_per_year: float, max_share: float,
                  alpha: float, newey_west_lag: int) -> str:
    v = evaluate_pnl_linear(
        reported_pnl=case.reported,
        explained_net_pnl=case.explained,
        remainder=case.remainder,
        frame_columns=FRAME_COLUMNS,
        config_block=_block(max_bp_per_year, max_share),
        bars_per_year=BARS_PER_YEAR,
        capital_base=CAPITAL,
        newey_west_lag=newey_west_lag,
        alpha=alpha,
        staleness_inputs={
            "bar_times": case.bar_times,
            "publication_times": case.publication_times,
            "position": pd.Series(case.position),
            "lag": 1,
            "halt_flag": case.halt_flag,
        },
    )
    return v.verdict


NOISE_BPS = (0.0, 1.0, 5.0)


def _noise_rng(case: BenchCase, noise_bp: float) -> np.random.Generator:
    # Separate stream from the price stream, keyed on seed, kind and noise level.
    return np.random.default_rng([case.seed, _OFFSET[case.kind], 7919, int(round(noise_bp * 100))])


def _add_noise(case: BenchCase, noise_bp: float, defect: str, expected, **over) -> BenchCase:
    """Add seeded reporting noise (std = noise_bp basis points of NAV per bar, NAV taken as CAPITAL)
    to the REPORTED series only. The explained series and the true ledger are unchanged, so the noise
    shows up as remainder."""
    if noise_bp < 0:
        raise ValueError("noise_bp must be non-negative")
    noise = _noise_rng(case, noise_bp).normal(0.0, noise_bp * 1e-4 * CAPITAL, N_BARS)
    return _with(case, defect, expected=expected, reported=case.reported + noise,
                 extra={**case.extra, "noise_bp": float(noise_bp)}, **over)


def noisy_honest_case(case: BenchCase, noise_bp: float) -> BenchCase:
    """Case group ``noisy_honest``: honest case with reporting noise; expectation as the honest case."""
    return _add_noise(case, noise_bp, f"noisy_honest_{noise_bp:g}bp", case.expected)


def noisy_gain_case(case: BenchCase, noise_bp: float, gain_bp: float) -> BenchCase:
    """Planted hidden gain on the honest case, then reporting noise on top."""
    gained = gain_defect(case, gain_bp)
    return _add_noise(gained, noise_bp, f"noisy_{noise_bp:g}bp_gain_{gain_bp:g}bp", ("FAIL",))


def noise_sensitivity_sweep(seeds, noise_bps=NOISE_BPS, gain_bps=GAIN_BPS, max_bp_per_year=NOISE_SCALED,
                            max_share=NOISE_SCALED, alpha=0.05, newey_west_lag=2,
                            kinds=CASE_KINDS) -> dict[str, Any]:
    """FAIL share per (noise level, gain size) and honest false-alarm rate per noise level.

    Returns a table, not an assertion: ``fail_share[(noise_bp, gain_bp)]``,
    ``honest_false_alarm[noise_bp]``, ``n_cases_per_cell``. Limits default to the noise-scaled ones; numbers override.
    """
    seeds = list(seeds)
    args = (max_bp_per_year, max_share, alpha, newey_west_lag)
    fail_share: dict[tuple[float, float], float] = {}
    false_alarm: dict[float, float] = {}
    for n in noise_bps:
        honest_fail = 0
        gain_fail = {g: 0 for g in gain_bps}
        for seed in seeds:
            for kind in kinds:
                base = make_honest_case(kind, seed)
                honest_fail += evaluate_case(noisy_honest_case(base, n), *args) == "FAIL"
                for g in gain_bps:
                    gain_fail[g] += evaluate_case(noisy_gain_case(base, n, g), *args) == "FAIL"
        cell = len(seeds) * len(kinds)
        false_alarm[float(n)] = honest_fail / cell
        for g in gain_bps:
            fail_share[(float(n), float(g))] = gain_fail[g] / cell
    return {"fail_share": fail_share, "honest_false_alarm": false_alarm,
            "n_cases_per_cell": len(seeds) * len(kinds)}


def run_corpus(seeds, max_bp_per_year: float, max_share: float, alpha: float,
               newey_west_lag: int, gain_bps=GAIN_BPS, return_outcomes: bool = False):
    """Verdict counts per (case kind, defect). With ``return_outcomes`` the verdict list in corpus order."""
    corpus = build_corpus(seeds, gain_bps)
    verdicts = [evaluate_case(c, max_bp_per_year, max_share, alpha, newey_west_lag) for c in corpus]
    if return_outcomes:
        return verdicts
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for c, v in zip(corpus, verdicts):
        counts[(c.kind, c.defect)][v] += 1
    return dict(counts)


def detection_power(results) -> dict[float, float]:
    """Share of planted-gain cases giving FAIL, by gain size in bp per bar."""
    power: dict[float, float] = {}
    for g in sorted({float(k[1][5:-2]) for k in results if k[1].startswith("gain_")}):
        fail = total = 0
        for (kind, defect), counts in results.items():
            if defect == f"gain_{g:g}bp":
                fail += counts.get("FAIL", 0)
                total += sum(counts.values())
        power[g] = fail / total
    return power


def false_alarm_rate(results) -> float:
    """Share of honest cases giving FAIL."""
    fail = total = 0
    for (kind, defect), counts in results.items():
        if defect == "honest":
            fail += counts.get("FAIL", 0)
            total += sum(counts.values())
    return fail / total if total else 0.0

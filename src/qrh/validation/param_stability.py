"""Parameter stability of the selected combination (``s5.param_stability@1``).

A sharp peak in parameter space is a typical sign of an overfit choice: the combination works and its immediate
neighbours do not. Two measures decide, two report-only items explain. All are computed from out-of-sample returns,
that is the returns of a setting on the bars the walk-forward (S4) scored.

Neighbourhood. Around the selected setting, ONE parameter at a time is moved to a value at a fixed relative distance
(``stability_steps``, each step up and down) from the selected value. Where the grid has a point at about that distance
(within a quarter of a step) it is used; every other neighbour is generated and run on the development window like a
single pre-registered combination (a non-trial run, see below). No neighbour may sit closer to the selected value than
the smallest step: closer grid points are ignored, closer declared values are refused, so a hairline "neighbour" cannot
make the check trivial. Integer parameters round away from the selected value, never toward it. A parameter that is
not numeric but takes two or more values in the grid is varied across those values. Numbers written as text ("20")
are numbers. Diagonal neighbours (two parameters at once) are NOT used: they would multiply the runs and add little to
a cross neighbourhood.

(a) Neighbour check (decides a spike), PER PARAMETER: ratio_p = mean OOS Sharpe of the neighbours of parameter p /
    OOS Sharpe of the selected setting. The decision uses the worst parameter: FAIL when min_p ratio_p is below
    ``stability_neighbour_ratio``. The pooled mean over all neighbours is reported as information only, because a
    parameter that changes nothing would otherwise dilute a spike in another. When the selected OOS Sharpe is not above
    zero the measures are not meaningful: NOT APPLICABLE, S4 and S5 judge the level.
(b) Local plateau share (decides a plateau): share = nearby numeric settings (the neighbours above) whose OOS Sharpe is
    >= ``stability_closeness`` x the SELECTED setting's OOS Sharpe, divided by their number. FAIL when below
    ``stability_plateau_share``. The old global share (grid points close to the grid's BEST) depends on how the grid
    is spaced, so it is kept in the receipt as ``global_info`` and never decides. Idea of a stable plateau: Carver,
    R. (2015), Systematic Trading, Harriman House; the neighbour check and the numbers are this harness's own.
(c) Diagnostics (report only). Per numeric parameter: the mean absolute OOS Sharpe gap between adjacent values (the
    other parameters fixed) and the mean correlation of their OOS return paths; the most fragile parameter is named.

Generated neighbour runs only check a fixed, pre-registered choice and never select anything, so they are NOT counted
as trials. They run on the whole development window, as the grid did in S3, and are scored on the same OOS bars. The
hypothesis may declare explicit neighbour values and bounds instead (single combination only).

Asset-, strategy- and frequency-agnostic: it works on whatever ``parameter_space()`` declares.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from qrh.hashing import hash_json
from qrh.llm.variants import VARIANT_KEY
from qrh.validation.metrics import annualized_sharpe

MAX_STEPS = 5
MIN_GRID_FOR_PLATEAU = 3
TRIAL_COUNT_NOTE = (
    "Neighbour runs check a fixed, pre-registered choice and never select among variants, so they are not counted "
    "as trials (the trial and family counts are unchanged)."
)
NO_NUMERIC = "no numeric parameter to vary"


class StabilityBlocked(ValueError):
    """The check could run in principle but cannot be trusted or completed; carries the plain reasons."""

    def __init__(self, *reasons: str):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


def is_number(v: Any) -> bool:
    return (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            and math.isfinite(float(v)))


def _is_int(v: Any) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


_INT_TEXT = re.compile(r"^[+-]?\d+$")
EPS = 1e-9


def as_float(v: Any) -> float | None:
    """The number a value stands for: a number, or text that parses as a finite number ("20", "0.5"); else None."""
    if is_number(v):
        return float(v)
    if isinstance(v, str):
        try:
            f = float(v.strip())
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _int_like(v: Any) -> bool:
    return _is_int(v) or (isinstance(v, str) and bool(_INT_TEXT.match(v.strip())))


def _typed(template: Any, x: float) -> Any:
    """``x`` written the way ``template`` is written (an int, a float, or text)."""
    if isinstance(template, str):
        return str(int(round(x))) if _INT_TEXT.match(template.strip()) else f"{x:.12g}"
    return int(round(x)) if _is_int(template) else float(x)


def _fraction(name: str, v: Any) -> float:
    if not is_number(v) or not (0 < float(v) <= 1):
        raise StabilityBlocked(f"threshold {name!r} must be a number above 0 and at most 1, got {v!r}")
    return float(v)


@dataclass(frozen=True)
class StabilityThresholds:
    closeness: float = 0.70
    plateau_share: float = 0.50
    neighbour_ratio: float = 0.50
    steps: tuple[float, ...] = (0.10, 0.20)

    @classmethod
    def from_mapping(cls, thresholds: Mapping[str, Any]) -> "StabilityThresholds":
        """Values from the frozen hypothesis thresholds; a missing one keeps its default. Raises StabilityBlocked."""
        d = cls()
        steps = thresholds.get("stability_steps", list(d.steps))
        if not isinstance(steps, (list, tuple)) or not 1 <= len(steps) <= MAX_STEPS or not all(
                is_number(s) and 0 < float(s) < 1 for s in steps):
            raise StabilityBlocked(
                f"threshold 'stability_steps' must be a list of 1 to {MAX_STEPS} fractions between 0 and 1, got {steps!r}")
        return cls(
            closeness=_fraction("stability_closeness", thresholds.get("stability_closeness", d.closeness)),
            plateau_share=_fraction("stability_plateau_share", thresholds.get("stability_plateau_share", d.plateau_share)),
            neighbour_ratio=_fraction("stability_neighbour_ratio", thresholds.get("stability_neighbour_ratio", d.neighbour_ratio)),
            steps=tuple(sorted({float(s) for s in steps})),
        )

    def doc(self) -> dict[str, Any]:
        return {"closeness": self.closeness, "plateau_share": self.plateau_share,
                "neighbour_ratio": self.neighbour_ratio, "steps": list(self.steps)}


@dataclass(frozen=True)
class Neighbour:
    dimension: str
    value: Any
    params: dict[str, Any]
    label: str
    source: str  # "grid" | "generated" | "declared"
    distance: float | None = None  # relative distance of the value from the selected one (numeric parameters)


# --- dimensions and neighbours ---------------------------------------------------------------------------------


def numeric_dimensions(combos: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[Any]], list[str]]:
    """({name: distinct values, sorted by number} of every numeric parameter, names of the non-numeric ones).

    A value that is text but parses as a number counts as a number, so quoting a number does not hide a parameter."""
    names = sorted({k for c in combos for k in c})
    dims: dict[str, list[Any]] = {}
    skipped: list[str] = []
    for name in names:
        values = [c.get(name) for c in combos]
        floats = [as_float(v) for v in values]
        if all(f is not None for f in floats):
            distinct: list[Any] = []
            for v in sorted(values, key=lambda x: as_float(x) or 0.0):
                if not distinct or as_float(v) != as_float(distinct[-1]):
                    distinct.append(v)
            dims[name] = distinct
        else:
            skipped.append(name)
    return dims, skipped


def _py(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


def _canon(params: Mapping[str, Any]) -> str:
    return hash_json({k: (as_float(v) if as_float(v) is not None else _py(v)) for k, v in params.items()})


def varied_categorical(combos: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[str]:
    """Non-numeric parameters that take two or more distinct values in the grid."""
    out = []
    for name in names:
        if len({hash_json(_py(c.get(name))) for c in combos}) >= 2:
            out.append(name)
    return out


def _rel(x: float, v: float) -> float:
    return abs(x - v) / abs(v) if v != 0 else abs(x - v)


def _bounds_pair(name: str, b: Any) -> tuple[float, float]:
    if not isinstance(b, (list, tuple)) or len(b) != 2 or not all(is_number(x) for x in b) or not float(b[0]) <= float(b[1]):
        raise StabilityBlocked(f"declared bounds for {name!r} must be [low, high] with low <= high, got {b!r}")
    return float(b[0]), float(b[1])


def generate_neighbours(
    params: Mapping[str, Any],
    dims: Mapping[str, Sequence[Any]],
    steps: Sequence[float],
    bounds: Mapping[str, Any] | None = None,
    explicit: Mapping[str, Sequence[Any]] | None = None,
    grid: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[Neighbour], list[dict[str, Any]]]:
    """Neighbours of one combination, one numeric parameter at a time, and the values that could not be produced.

    Per parameter and per step ``s``, up and down: the grid point (``grid``: label -> params) that differs from the
    selected setting only in this parameter and lies within a quarter step of the target value, when there is one;
    else the value ``v +/- |v| * s``, integers rounded AWAY from ``v``, clipped to ``bounds[name]`` when declared. No
    neighbour may be closer to ``v`` than the smallest step: a closer grid point is ignored, a closer generated value
    (after clipping) is listed as unchecked, a closer declared value (``explicit``) is refused (StabilityBlocked).
    The value itself and duplicates are dropped. A parameter that yields nothing is listed as unchecked."""
    bounds, explicit = dict(bounds or {}), dict(explicit or {})
    for name in sorted({*explicit, *bounds}):
        if name not in dims:
            raise StabilityBlocked(f"hypothesis stability block names unknown parameter {name!r} (numeric parameters: "
                                   f"{sorted(dims) or 'none'})")
    min_step = min(steps)
    out: list[Neighbour] = []
    unchecked: list[dict[str, Any]] = []
    for name in sorted(dims):
        template = params[name]
        v = as_float(template)
        assert v is not None
        box = _bounds_pair(name, bounds[name]) if name in bounds else None
        others = {k: x for k, x in params.items() if k != name}
        same_row = [(label, gp) for label, gp in (grid or {}).items()
                    if {k: x for k, x in gp.items() if k != name} == others or
                    _canon({k: x for k, x in gp.items() if k != name}) == _canon(others)]
        seen = {round(v, 12)}
        dropped: list[str] = []
        produced = 0

        def add(x: float, source: str, distance: float, label: str | None = None, row: Mapping[str, Any] | None = None) -> None:
            nonlocal produced
            key = round(x, 12)
            if key in seen:
                return
            seen.add(key)
            value = _py(row[name]) if row is not None else _typed(template, x)
            new = dict(row) if row is not None else {**params, name: value}
            out.append(Neighbour(name, value, new, label or f"{name}={value}", source, distance))
            produced += 1

        if name in explicit:
            vals = list(explicit[name])
            if not vals or any(as_float(x) is None for x in vals):
                raise StabilityBlocked(f"declared neighbour values for {name!r} must be a non-empty list of numbers")
            for raw in vals:
                x = as_float(raw)
                assert x is not None
                if _rel(x, v) < min_step - EPS:
                    raise StabilityBlocked(
                        f"declared neighbour {raw!r} of {name!r} is too close to the selected value {_py(template)!r}: "
                        f"closer than the smallest stability step ({min_step:g} of the value); a hairline neighbour "
                        "would make the check trivial")
                add(x, "declared", _rel(x, v))
            if not produced:
                unchecked.append({"dimension": name, "value": None, "reason": "the declared values equal the selected value"})
            continue
        if v == 0:
            unchecked.append({"dimension": name, "value": None,
                              "reason": "the selected value is 0, so a relative neighbour cannot be built; declare neighbour values"})
            continue
        for step in steps:
            for sign in (-1.0, 1.0):
                raw = round(v + sign * abs(v) * step, 9)
                best: tuple[float, str, Mapping[str, Any], float] | None = None
                for label, gp in same_row:
                    g = as_float(gp[name])
                    if g is None or _rel(g, v) < min_step - EPS or abs(g - raw) > 0.25 * abs(v) * step + EPS:
                        continue
                    if best is None or abs(g - raw) < best[0]:
                        best = (abs(g - raw), label, gp, g)
                if best is not None:
                    add(best[3], "grid", _rel(best[3], v), best[1], best[2])
                    continue
                x = raw
                if _int_like(template):
                    x = float(math.floor(raw) if sign < 0 else math.ceil(raw))  # away from v, never toward it
                if box is not None:
                    x = min(max(x, box[0]), box[1])
                if round(x, 12) == round(v, 12):
                    dropped.append(f"{raw:g} falls on the selected value (bounds or rounding)")
                elif _rel(x, v) < min_step - EPS:
                    dropped.append(f"{x:g} is closer than the smallest step ({min_step:g}) to {v:g}")
                else:
                    add(x, "generated", _rel(x, v))
        if not produced:
            unchecked.append({"dimension": name, "value": None,
                              "reason": "; ".join(dict.fromkeys(dropped)) or
                              f"no distinct neighbour value at steps {list(steps)} from {_py(template)!r}"})
        else:
            for reason in dict.fromkeys(dropped):
                unchecked.append({"dimension": name, "value": None, "reason": f"one direction unchecked: {reason}"})
    return out, unchecked


def categorical_neighbours(selected: Mapping[str, Any], grid: Mapping[str, Mapping[str, Any]],
                           names: Sequence[str]) -> list[Neighbour]:
    """Grid combinations that differ from the selected one only in one non-numeric parameter."""
    out: list[Neighbour] = []
    for name in names:
        others = _canon({k: x for k, x in selected.items() if k != name})
        for label, gp in grid.items():
            if _canon({k: x for k, x in gp.items() if k != name}) == others and _canon({name: gp[name]}) != _canon({name: selected[name]}):
                out.append(Neighbour(name, _py(gp[name]), dict(gp), label, "grid", None))
    return out


def _grid_pairs(grid: Mapping[str, Mapping[str, Any]], dims: Mapping[str, Sequence[Any]]) -> list[tuple[str, str, str]]:
    """(label, label of the adjacent-value combination, parameter) for every adjacent pair, other parameters fixed."""
    index = {_canon(p): label for label, p in grid.items()}
    pairs: list[tuple[str, str, str]] = []
    for name in sorted(dims):
        values = list(dims[name])
        for label, p in grid.items():
            pos = next((i for i, v in enumerate(values) if as_float(v) == as_float(p[name])), None)
            if pos is not None and pos + 1 < len(values):
                other = index.get(_canon({**p, name: values[pos + 1]}))
                if other is not None:
                    pairs.append((label, other, name))
    return pairs


# --- the result ------------------------------------------------------------------------------------------------


@dataclass
class StabilityResult:
    status: str  # PASS | FAIL | NOT_APPLICABLE
    reasons: list[str]
    selected: dict[str, Any]
    neighbour: dict[str, Any]
    plateau: dict[str, Any]
    diagnostics: dict[str, Any]
    skipped: list[str]
    unchecked: list[dict[str, Any]]
    thresholds: StabilityThresholds
    oos_bars: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "reasons": list(self.reasons), "selected": self.selected,
            "neighbour": self.neighbour, "plateau": self.plateau, "diagnostics": self.diagnostics,
            "skipped": list(self.skipped), "unchecked": list(self.unchecked), "thresholds": self.thresholds.doc(),
            "oos_bars": self.oos_bars, "notes": list(self.notes), "trial_count_note": TRIAL_COUNT_NOTE,
            "method": "param_stability@1",
        }


def _sharpe(values: np.ndarray, ppy: float) -> float | None:
    s = annualized_sharpe(values, ppy)
    return None if s is None else float(s)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _diagnostics(paths: Mapping[str, np.ndarray], sharpes: Mapping[str, float | None],
                 pairs: Sequence[tuple[str, str, str]]) -> dict[str, Any]:
    by_dim: dict[str, list[tuple[float, float | None]]] = {}
    for a, b, dim in pairs:
        gap = abs((sharpes[a] or 0.0) - (sharpes[b] or 0.0))
        by_dim.setdefault(dim, []).append((gap, _corr(paths[a], paths[b])))
    rows = []
    for dim in sorted(by_dim):
        gaps = [g for g, _ in by_dim[dim]]
        corrs = [c for _, c in by_dim[dim] if c is not None]
        rows.append({"dimension": dim, "pairs": len(gaps), "mean_abs_sharpe_gap": float(np.mean(gaps)),
                     "mean_correlation": float(np.mean(corrs)) if corrs else None})
    worst = None
    if rows:
        worst = sorted(rows, key=lambda r: (-r["mean_abs_sharpe_gap"],
                                            2.0 if r["mean_correlation"] is None else r["mean_correlation"],
                                            r["dimension"]))[0]["dimension"]
    return {"parameters": rows, "most_fragile": worst,
            "note": "report only: the mean absolute OOS Sharpe gap between adjacent values and the mean correlation of "
                    "their OOS return paths; the most fragile parameter has the largest gap"}


def _not_meaningful(sel_sr: float | None) -> str:
    return ("the selected out-of-sample Sharpe ratio is "
            + ("undefined" if sel_sr is None else f"{sel_sr:.3f}, not above zero")
            + ": the measures are not meaningful and are not judged here (S4 and S5 judge the level)")


def _empty_neighbour(sel_sr: float | None, th: StabilityThresholds, note: str, applicable: bool = False) -> dict[str, Any]:
    return {"applicable": applicable, "meaningful": False, "source": None, "selected_sharpe": sel_sr,
            "mean_neighbour_sharpe": None, "pooled_mean_ratio": None, "ratio": None, "worst_parameter": None,
            "threshold": th.neighbour_ratio, "passed": None, "n_neighbours": 0, "neighbours": [], "per_parameter": [],
            "undefined_as_zero": 0, "note": note}


def _empty_plateau(sel_sr: float | None, th: StabilityThresholds, note: str, global_info: dict[str, Any]) -> dict[str, Any]:
    return {"applicable": False, "meaningful": False, "reference_sharpe": sel_sr, "closeness": th.closeness,
            "share": None, "pass_mark": th.plateau_share, "passed": None, "n_close": None, "n_neighbours": 0,
            "note": note, "global_info": global_info}


def _global_info(labels: Sequence[str], sharpes: Mapping[str, float | None], th: StabilityThresholds) -> dict[str, Any]:
    """The old global plateau share: grid points within ``closeness`` of the grid's best. Information only."""
    n = len(labels)
    info: dict[str, Any] = {
        "applicable": n >= MIN_GRID_FOR_PLATEAU, "best_sharpe": None, "closeness": th.closeness, "share": None,
        "n_close": None, "n_combinations": n,
        "note": "information only, never decides: it depends on how widely the grid is spaced",
    }
    if n < MIN_GRID_FOR_PLATEAU:
        return info
    values = [sharpes[label] for label in labels]
    best = max((v for v in values if v is not None), default=None)
    info["best_sharpe"] = best
    if best is not None and best > 0:
        close = sum(1 for v in values if v is not None and v >= th.closeness * best)
        info.update(share=float(close / n), n_close=close)
    return info


def _neighbour_block(sel_sr: float, neighbours: Sequence[Neighbour], numeric: set[str],
                     sharpes: Mapping[str, float | None], th: StabilityThresholds) -> dict[str, Any]:
    by_dim: dict[str, list[Neighbour]] = {}
    for n in neighbours:
        by_dim.setdefault(n.dimension, []).append(n)
    per = []
    for dim in sorted(by_dim):
        vals = [sharpes[n.label] or 0.0 for n in by_dim[dim]]
        mean = float(np.mean(vals))
        per.append({"dimension": dim, "kind": "numeric" if dim in numeric else "varied non-numeric",
                    "n_neighbours": len(vals), "mean_sharpe": mean, "ratio": mean / sel_sr,
                    "sources": sorted({n.source for n in by_dim[dim]})})
    worst = min(per, key=lambda r: (r["ratio"], r["dimension"]))
    pooled = float(np.mean([sharpes[n.label] or 0.0 for n in neighbours]))
    rows = [{"dimension": n.dimension, "value": _py(n.value), "sharpe": sharpes[n.label] or 0.0, "source": n.source,
             "distance": n.distance} for n in neighbours]
    undefined = sum(1 for n in neighbours if sharpes[n.label] is None)
    return {
        "applicable": True, "meaningful": True, "source": "+".join(sorted({n.source for n in neighbours})),
        "selected_sharpe": sel_sr, "mean_neighbour_sharpe": pooled, "pooled_mean_ratio": pooled / sel_sr,
        "ratio": worst["ratio"], "worst_parameter": worst["dimension"], "threshold": th.neighbour_ratio,
        "passed": bool(worst["ratio"] >= th.neighbour_ratio), "n_neighbours": len(neighbours), "neighbours": rows,
        "per_parameter": per, "undefined_as_zero": undefined,
        "note": (f"{undefined} neighbour(s) had an undefined Sharpe ratio (no variation, for example a strategy that never "
                 "trades) and count as zero" if undefined else
                 "decided on the worst parameter; the pooled mean over all neighbours is information only"),
    }


def _plateau_block(sel_sr: float, nums: Sequence[Neighbour], sharpes: Mapping[str, float | None],
                   th: StabilityThresholds, global_info: dict[str, Any]) -> dict[str, Any]:
    if not nums:
        return _empty_plateau(sel_sr, th, "no numeric neighbour: only non-numeric parameters vary", global_info)
    close = sum(1 for n in nums if (sharpes[n.label] or 0.0) >= th.closeness * sel_sr)
    share = close / len(nums)
    return {"applicable": True, "meaningful": True, "reference_sharpe": sel_sr, "closeness": th.closeness,
            "share": float(share), "pass_mark": th.plateau_share, "passed": bool(share >= th.plateau_share),
            "n_close": close, "n_neighbours": len(nums), "note": "", "global_info": global_info}


# --- the method ------------------------------------------------------------------------------------------------


def run_param_stability(
    *,
    grid: Sequence[tuple[str, Mapping[str, Any]]],
    matrix: pd.DataFrame,
    oos_index: pd.DatetimeIndex,
    picks: Sequence[str],
    ppy: float,
    thresholds: StabilityThresholds,
    adapter: Any = None,
    view: Any = None,
    start: Any = None,
    end: Any = None,
    declared: Mapping[str, Any] | None = None,
    run_kwargs: Mapping[str, Any] | None = None,
) -> StabilityResult:
    """The whole check. ``grid`` is (label, params) for every combination, ``matrix`` their development-window
    returns (columns = labels), ``oos_index`` the bars S4 scored and ``picks`` the label S4 selected in each fold.
    ``adapter``, ``view``, ``start`` and ``end`` run the neighbours that are not grid points (never recorded as trials).
    ``declared`` is the hypothesis ``stability`` block (``neighbours``, single combination only, and ``bounds``).
    Raises StabilityBlocked."""
    from qrh.llm.cache import LLMCacheCorrupt, LLMCacheMiss

    declared = dict(declared or {})
    params_by_label = {label: dict(p) for label, p in grid}
    if not picks or any(p not in params_by_label or p not in matrix.columns for p in picks):
        raise StabilityBlocked("the walk-forward folds name a combination that is not in the stored grid")
    if len(oos_index) < 3 or not oos_index.isin(matrix.index).all():
        raise StabilityBlocked("the out-of-sample bars are missing from the stored development-window returns")
    counts = Counter(picks)
    top = max(counts.values())
    selected = min(label for label, c in counts.items() if c == top)  # ties: the lowest label, as in S4
    sel_params = params_by_label[selected]
    th = thresholds
    sel_doc: dict[str, Any] = {"label": selected, "picks": dict(sorted(counts.items())), "n_folds": len(picks)}
    combos = list(params_by_label.values())
    dims, non_numeric = numeric_dimensions(combos)
    non_numeric = [n for n in non_numeric if n != VARIANT_KEY or len({hash_json(_py(c.get(n))) for c in combos}) >= 2]
    cats = varied_categorical(combos, non_numeric)
    skipped = [n for n in non_numeric if n not in cats and n != VARIANT_KEY]

    oos = matrix.loc[oos_index]
    if not np.isfinite(oos.to_numpy(dtype=float)).all():
        raise StabilityBlocked("stored out-of-sample returns contain NaN or inf")
    paths = {label: oos[label].to_numpy(dtype=float) for label in matrix.columns if label in params_by_label}
    sharpes: dict[str, float | None] = {label: _sharpe(v, ppy) for label, v in paths.items()}
    sel_sr = sharpes[selected]
    glob = _global_info(list(paths), sharpes, th)
    unchecked: list[dict[str, Any]] = []
    notes: list[str] = []

    def result(status: str, reasons: list[str], nb: dict[str, Any], pl: dict[str, Any], diag: dict[str, Any]) -> StabilityResult:
        return StabilityResult(status, reasons, sel_doc, nb, pl, diag, skipped, unchecked, th, len(oos_index), notes)

    if not [k for k in sel_params if k != VARIANT_KEY]:
        note = "the adapter declares no parameters, so there is nothing to nudge"
        return result("NOT_APPLICABLE", [f"parameter stability not applicable: {note}"], _empty_neighbour(sel_sr, th, note),
                      _empty_plateau(sel_sr, th, note, glob), _diagnostics(paths, sharpes, []))
    if sel_sr is None or not sel_sr > 0:
        note = _not_meaningful(sel_sr)
        return result("NOT_APPLICABLE", [f"parameter stability not applicable: {note}"], _empty_neighbour(sel_sr, th, note),
                      _empty_plateau(sel_sr, th, note, glob), _diagnostics(paths, sharpes, []))
    if not dims and not cats:
        raise StabilityBlocked(
            "the adapter takes parameters (" + ", ".join(sorted(k for k in sel_params if k != VARIANT_KEY))
            + ") but none is numeric and none takes more than one value, so there is nothing to nudge and the check cannot "
            "run; the trader must decide how to treat this strategy (for example declare a numeric parameter), the check "
            "will not pass it silently")

    single = len(params_by_label) == 1
    if declared.get("neighbours") and not single:
        notes.append("declared neighbour values apply to one pre-registered combination only; steps and grid points are used")
    num_nb, unchecked = generate_neighbours(
        sel_params, dims, th.steps, declared.get("bounds"), declared.get("neighbours") if single else None, params_by_label)
    cat_nb = categorical_neighbours(sel_params, params_by_label, cats)
    to_run = [n for n in num_nb if n.source != "grid"]
    if to_run and adapter is None:
        raise StabilityBlocked("the neighbours that are not grid points need the adapter to run them, and it was not given")
    ran: list[Neighbour] = []
    for nb_ in to_run:
        try:
            res = adapter.run(nb_.params, view, start, end, **dict(run_kwargs or {}))
            ret = res.returns
            if not isinstance(ret, pd.Series) or not ret.index.equals(matrix.index):
                raise ValueError("returns must be a Series covering exactly the bars of the development window")
            vals = ret.reindex(oos_index).to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError("returns contain NaN or inf on the out-of-sample bars")
        except (LLMCacheMiss, LLMCacheCorrupt):
            raise
        except Exception as exc:  # a value the strategy cannot run with is not a neighbour; it is listed, not hidden
            if nb_.source == "declared":
                raise StabilityBlocked(f"declared neighbour {nb_.label} failed: {type(exc).__name__}: {exc}") from exc
            unchecked.append({"dimension": nb_.dimension, "value": _py(nb_.value),
                              "reason": f"the strategy could not run: {type(exc).__name__}: {exc}"})
            continue
        paths[nb_.label], sharpes[nb_.label] = vals, _sharpe(vals, ppy)
        ran.append(nb_)
    used_num = [n for n in num_nb if n.source == "grid"] + ran
    neighbours = used_num + cat_nb
    if not neighbours:
        raise StabilityBlocked(
            "no neighbour could be produced and run for the selected combination ("
            + "; ".join(f"{u['dimension']}: {u['reason']}" for u in unchecked)
            + "); declare neighbour values in the hypothesis `stability` block or fix the adapter")
    nb = _neighbour_block(sel_sr, neighbours, set(dims), sharpes, th)
    pl = _plateau_block(sel_sr, used_num, sharpes, th, glob)
    pairs = [(selected, n.label, n.dimension) for n in used_num] if single else _grid_pairs(params_by_label, dims)
    diag = _diagnostics(paths, sharpes, pairs)

    reasons: list[str] = []
    if nb["passed"] is False:
        reasons.append(
            f"parameter stability: neighbour check failed, parameter {nb['worst_parameter']} keeps {nb['ratio']:.2f} of the "
            f"selected out-of-sample Sharpe {nb['selected_sharpe']:.2f} in its neighbours, below stability_neighbour_ratio "
            f"{th.neighbour_ratio:g}")
    if pl["passed"] is False:
        reasons.append(
            f"parameter stability: plateau check failed, {pl['n_close']} of {pl['n_neighbours']} nearby settings "
            f"({pl['share']:.2f}) keep {th.closeness:g} of the selected setting's out-of-sample Sharpe, below "
            f"stability_plateau_share {th.plateau_share:g}")
    if reasons:
        status = "FAIL"
    elif nb["passed"] or pl["passed"]:
        status = "PASS"
    else:
        status = "NOT_APPLICABLE"
        reasons.append("parameter stability not applicable: no measure was meaningful")
    return result(status, reasons, nb, pl, diag)

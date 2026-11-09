"""Layering of thresholds: core defaults < public packs < private packs < hypothesis.

Packs may tighten freely. Loosening a core value is refused unless a human approved it (the approval is
recorded in ``packs.json`` and in every receipt). A later pack can never relax what an earlier one tightened.
The hypothesis is the last layer and keeps its explicit values; where that is looser than the packs it is
recorded (``hypothesis_looser_than_pack``) rather than silently accepted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from qrh.packs.model import DIMENSIONS, Pack, PackError

HIGHER_IS_TIGHTER = ("bootstrap_reps", "ci_level", "cost_stress_multiplier", "llm_cutoff_margin_days",
                    "perturbation_cutoffs", "perturbation_change_cap", "stability_closeness",
                    "stability_plateau_share", "stability_neighbour_ratio")


def direction(name: str) -> str | None:
    """'min' when a higher value is stricter (min_* floors), 'max' when a lower one is, None when unordered."""
    if name.startswith("min_") or name in HIGHER_IS_TIGHTER:
        return "min"
    if name.startswith("max_"):
        return "max"
    return None


def is_looser(name: str, base: Any, new: Any) -> bool:
    d = direction(name)
    if d is None or isinstance(base, bool) or isinstance(new, bool):
        return False
    if not isinstance(base, (int, float)) or not isinstance(new, (int, float)):
        return False
    return new < base if d == "min" else new > base


@dataclass(frozen=True)
class Loosening:
    pack: str
    name: str
    value: Any
    core: Any

    @property
    def id(self) -> str:
        return f"{self.pack}:{self.name}={self.value}"


def pack_values(pack: Pack, attrs: dict[str, Any] | None) -> dict[str, Any]:
    """The pack's thresholds for these hypothesis attributes (``None`` attrs: every dimension, for audits)."""
    out: dict[str, Any] = dict(pack.thresholds.get("all", {}))
    for dim in DIMENSIONS:
        for key, body in pack.thresholds.get(dim, {}).items():
            if attrs is None or attrs.get(dim) == key:
                out.update(body)
    return out


def loosenings(packs: list[Pack], attrs: dict[str, Any] | None, core: dict[str, Any] | None = None) -> list[Loosening]:
    from qrh.validation.defaults import DEFAULT_THRESHOLDS

    base = DEFAULT_THRESHOLDS if core is None else core
    found: list[Loosening] = []
    for pack in packs:
        for name, value in sorted(pack_values(pack, attrs).items()):
            if name in base and is_looser(name, base[name], value):
                found.append(Loosening(pack.name, name, value, base[name]))
    return found


def layer_thresholds(
    core: dict[str, Any], packs: list[Pack], attrs: dict[str, Any], hypothesis: dict[str, Any], approved: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Effective thresholds and a record of what changed. Raises PackError on an unapproved loosening."""
    eff = copy.deepcopy(dict(core))
    tightened: dict[str, Any] = {}
    used_approvals: list[str] = []
    pack_set: dict[str, Any] = {}
    for pack in packs:
        for name, value in sorted(pack_values(pack, attrs).items()):
            if name in core and is_looser(name, core[name], value):
                lo = Loosening(pack.name, name, value, core[name])
                if lo.id not in approved:
                    raise PackError(
                        f"pack {pack.name!r} would loosen core threshold {name} ({core[name]} -> {value}); "
                        f"this needs a human approval (`qrh pack add` prints the approval id)")
                used_approvals.append(lo.id)
                eff[name] = value
                continue
            current = eff.get(name)
            if current is None or direction(name) is None:
                eff[name] = value
                pack_set[name] = value
            elif is_looser(name, current, value) or value == current:
                continue  # an earlier layer is already at least this strict
            else:
                tightened[name] = {"from": current, "to": value, "pack": pack.name}
                eff[name] = value
            pack_set[name] = eff[name]
    looser = sorted(k for k, v in hypothesis.items() if k in pack_set and is_looser(k, pack_set[k], v))
    eff.update(hypothesis)
    info: dict[str, Any] = {"tightened": tightened}
    if used_approvals:
        info["approved_loosening"] = sorted(used_approvals)
    if looser:
        info["hypothesis_looser_than_pack"] = looser
    return eff, info

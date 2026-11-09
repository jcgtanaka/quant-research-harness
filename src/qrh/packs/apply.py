"""What packs do at S0: layered thresholds, tightened bounds, lessons and their requirements, account facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qrh.memory.registry import HypothesisRecord
from qrh.packs.layering import is_looser, layer_thresholds
from qrh.packs.model import PackError
from qrh.packs.runtime import PackView
from qrh.packs.store import approved_ids


@dataclass
class S0Outcome:
    spec: HypothesisRecord
    layering: dict[str, Any] = field(default_factory=dict)
    lessons: list[dict[str, Any]] = field(default_factory=list)
    account: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)


def lesson_records(view: PackView, trial: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"pack": pack, "id": les.id, "action": les.action, "severity": les.severity, "statement": les.statement,
             **({"requirement": les.requirement} if les.requirement else {})} for pack, les in view.matched_lessons(trial)]


def account_record(view: PackView, trial: dict[str, Any]) -> dict[str, Any] | None:
    found = view.account()
    if found is None:
        return None
    name, a = found
    ac = trial["asset_class"]
    return {"pack": name, "capital": a.capital, "base_currency": a.base_currency, "caps": dict(a.caps),
            "min_order_size": a.min_order_size.get(ac), "fee_schedule": a.fee_schedule.get(ac),
            "margin_rules": dict(a.margin_rules), "hash": view.account_hash()}


def defaults_in_force(view: PackView, spec: HypothesisRecord, home: str | Path) -> dict[str, Any]:
    """The thresholds in force before the hypothesis says anything: the core defaults tightened by the active packs."""
    from qrh.validation.defaults import DEFAULT_THRESHOLDS

    attrs = {"asset_class": spec.asset_class, "strategy_family": spec.strategy_family, "frequency": spec.frequency}
    return layer_thresholds(DEFAULT_THRESHOLDS, view.packs, attrs, {}, approved_ids(home))[0]


def apply_s0(view: PackView, spec: HypothesisRecord, home: str | Path) -> S0Outcome:
    """Fill a fresh hypothesis through the pack layers. Raises PackError for an unapproved loosening."""
    from qrh.methods.registry import MethodError, parse_ref
    from qrh.methods.selection import check_pins
    from qrh.validation.defaults import DEFAULT_THRESHOLDS

    attrs = {"asset_class": spec.asset_class, "strategy_family": spec.strategy_family, "frequency": spec.frequency}
    trial = {**attrs, "text": spec.text, "difference_statement": spec.difference_statement,
             "trial_id": "", "family_id": spec.resolved_family_id(), "datasets": [],
             "llm": spec.llm.model_dump(mode="json") if spec.llm is not None else None}
    out = S0Outcome(spec)
    eff, info = layer_thresholds(DEFAULT_THRESHOLDS, view.packs, attrs, dict(spec.thresholds), approved_ids(home))
    methods = dict(spec.methods)
    out.lessons = lesson_records(view, trial)
    for _pack, les in view.matched_lessons(trial):
        req = les.requirement or {}
        if les.action != "require":
            continue
        if "threshold_floor" in req:
            name, floor = req["threshold_floor"]["name"], req["threshold_floor"]["value"]
            explicit = spec.thresholds.get(name)
            weaker = lambda v: (is_looser(name, floor, v) if name.startswith(("min_", "max_")) else v < floor)  # noqa: E731
            if explicit is not None and weaker(explicit):
                out.reasons.append(f"lesson {les.id}: threshold floor not met: {name}={explicit}, the lesson requires {floor}")
            elif name not in eff or weaker(eff[name]):
                info.setdefault("tightened", {})[name] = {"from": eff.get(name), "to": floor, "pack": f"lesson:{les.id}"}
                eff[name] = floor
        if "method" in req:
            try:
                mid, version = parse_ref(str(req["method"]))
            except MethodError as exc:
                out.reasons.append(f"lesson {les.id}: {exc}")
                continue
            if methods.get(mid, version) != version:
                out.reasons.append(f"lesson {les.id}: requires method {mid}={version} but the hypothesis pins {methods[mid]}")
            else:
                methods[mid] = version
    if out.reasons:
        return out
    pin_problems = check_pins(home, methods) if methods else []
    out.reasons += [f"required method: {r}" for r in pin_problems] if methods != spec.methods else []
    bounds = dict(spec.bounds)
    account = account_record(view, trial)
    if account:
        for key, cap in sorted(account["caps"].items()):
            cur = bounds.get(key)
            if cur is None or cur > cap:
                info.setdefault("bounds_tightened", {})[key] = {"from": cur, "to": cap}
                bounds[key] = cap
        out.account = account
    out.layering = {k: v for k, v in info.items() if v}
    out.spec = spec.model_copy(update={"thresholds": eff, "bounds": bounds, "methods": methods})
    return out

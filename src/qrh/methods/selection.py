"""Choosing which version of a method a run uses.

Order of precedence: a benchmark override, then the version pinned by the hypothesis (``methods:`` in the
spec, recorded at S0), then the current default of the id. The default is the shipped one unless a local
promotion changed it. Every version actually handed out is recorded in ``used`` (and so in the receipt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from qrh.methods.registry import Method, MethodError, MethodRegistry, parse_ref
from qrh.methods.state import effective_status, load_state


def _registry(registry: MethodRegistry | None) -> MethodRegistry:
    if registry is not None:
        return registry
    from qrh.methods.builtin import BUILTIN

    return BUILTIN


class MethodSelector:
    def __init__(
        self,
        home: str | Path,
        pinned: Mapping[str, str] | None = None,
        registry: MethodRegistry | None = None,
        overrides: Mapping[str, str] | None = None,
    ):
        self.home = Path(home)
        self.registry = _registry(registry)
        self.pinned = dict(pinned or {})
        self.overrides = dict(overrides or {})
        self.used: dict[str, str] = {}
        self.flagged: dict[str, str] = {}  # used versions that are not the default of their id
        self._state: dict[str, Any] | None = None

    @property
    def state(self) -> dict[str, Any]:
        if self._state is None:
            self._state = load_state(self.home)
        return self._state

    def status(self, method: Method) -> str:
        return effective_status(method, self.state)

    def default(self, method_id: str) -> Method:
        versions = self.registry.versions(method_id)
        if not versions:
            raise MethodError(f"unknown method id {method_id!r}")
        defaults = [m for m in versions if self.status(m) == "default"]
        if len(defaults) != 1:
            raise MethodError(f"{method_id} has {len(defaults)} default versions, expected exactly one")
        return defaults[0]

    def opt_in(self, method_id: str) -> bool:
        """True for an id that ships with no default version: it runs only where a hypothesis selects it."""
        versions = self.registry.versions(method_id)
        return bool(versions) and not any(m.status == "default" for m in versions)

    def default_version(self, method_id: str) -> str | None:
        return None if self.opt_in(method_id) else self.default(method_id).version

    def selected(self, method_id: str) -> Method | None:
        """The method to use, recorded as used; None for an opt-in id that nothing selected."""
        if self.opt_in(method_id) and not (self.overrides.get(method_id) or self.pinned.get(method_id)):
            return None
        return self.get(method_id)

    def chosen(self, method_id: str) -> Method:
        version = self.overrides.get(method_id) or self.pinned.get(method_id)
        return self.registry.get(method_id, version) if version else self.default(method_id)

    def record(self, method_id: str, version: str) -> None:
        """Mark a version as used; flags it when it is not the current default (raises on a broken state)."""
        default = self.default_version(method_id)
        self.used[method_id] = version
        if version != default:
            self.flagged[method_id] = version
        else:
            self.flagged.pop(method_id, None)

    def get(self, method_id: str) -> Method:
        """The method to use, recorded as used."""
        method = self.chosen(method_id)
        self.record(method_id, method.version)
        return method

    def non_default(self) -> dict[str, str]:
        """Used methods that are not the current default of their id."""
        return dict(sorted(self.flagged.items()))

    def full_map(self) -> dict[str, str]:
        """Every id with the version this selector would hand out (pins and overrides, else defaults)."""
        return {
            i: self.chosen(i).version for i in self.registry.ids()
            if not self.opt_in(i) or self.overrides.get(i) or self.pinned.get(i)
        }


def check_pins(home: str | Path, pinned: Mapping[str, str], registry: MethodRegistry | None = None) -> list[str]:
    """Reasons a hypothesis may not pin these versions (empty if fine)."""
    sel = MethodSelector(home, registry=registry)
    reasons: list[str] = []
    for mid, version in sorted(pinned.items()):
        try:
            parse_ref(f"{mid}={version}")
            method = sel.registry.get(mid, version)
        except MethodError as exc:
            reasons.append(f"methods.{mid}: {exc}")
            continue
        if sel.status(method) == "deprecated":
            reasons.append(f"methods.{mid}: {version} is deprecated and cannot be used for a new hypothesis")
    return reasons

"""Method registry: every versioned, cited implementation the harness can use.

A method is identified by an id (``s5.pbo``) and a version (``cscv@1``). Versions are immutable:
an improvement is a new version, never an edit of an old one. Public defaults ship in code; a local
promotion changes status only through the methods state file (see ``qrh.methods.state``).

Statuses: ``default`` (used when a hypothesis does not pin a version; exactly one per id),
``experimental`` (usable, but flagged in receipts and reports) and ``deprecated`` (kept so old
receipts can be reproduced; refused for new hypotheses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

STATUSES = ("experimental", "default", "deprecated")
STAGES = ("S1", "S2", "S3", "S4", "S5", "S6")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_VERSION_RE = re.compile(r"^[a-z][a-z0-9_]*@[1-9][0-9]*$")


class MethodError(ValueError):
    """An unknown, malformed or inconsistent method reference or record."""


def parse_ref(text: str) -> tuple[str, str]:
    """``s5.pbo=cscv@1`` (canonical) or ``s5.pbo@cscv@1`` -> (``s5.pbo``, ``cscv@1``)."""
    if not isinstance(text, str):
        raise MethodError(f"method reference must be a string, got {text!r}")
    if "=" in text:
        mid, _, version = text.partition("=")
    else:
        mid, _, version = text.partition("@")
    if not _ID_RE.match(mid) or not _VERSION_RE.match(version):
        raise MethodError(f"{text!r} is not a method reference; use <id>=<name>@<n>, for example s5.pbo=cscv@1")
    return mid, version


def format_ref(method_id: str, version: str) -> str:
    return f"{method_id}={version}"


@dataclass(frozen=True)
class Reference:
    """A citation. ``verified`` is true only when an identifier was checked against the publisher record."""

    citation: str
    doi: str = ""
    arxiv: str = ""
    url: str = ""
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.citation.strip():
            raise MethodError("reference citation must not be blank")
        if self.verified and not (self.doi or self.arxiv or self.url):
            raise MethodError(f"reference {self.citation!r} cannot be verified without a doi, arxiv id or url")

    def as_dict(self) -> dict[str, Any]:
        return {"citation": self.citation, "doi": self.doi, "arxiv": self.arxiv, "url": self.url, "verified": self.verified}


@dataclass(frozen=True)
class Method:
    id: str
    version: str
    stage: str
    kind: str
    entry: Callable[..., Any] = field(repr=False, compare=False)
    parameters: dict[str, Any]
    status: str
    references: tuple[Reference, ...]
    added_in: str
    notes: str = ""
    validate: Callable[..., None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.id):
            raise MethodError(f"invalid method id {self.id!r}")
        if not _VERSION_RE.match(self.version):
            raise MethodError(f"invalid method version {self.version!r}; use <name>@<n>")
        if self.stage not in STAGES:
            raise MethodError(f"{self.id}: unknown stage {self.stage!r}")
        if self.status not in STATUSES:
            raise MethodError(f"{self.id}: unknown status {self.status!r}")
        if not callable(self.entry):
            raise MethodError(f"{self.id}: entry must be callable")
        if not self.kind.strip() or not self.added_in.strip():
            raise MethodError(f"{self.id}: kind and added_in are required")

    @property
    def ref(self) -> str:
        return format_ref(self.id, self.version)

    def describe(self) -> dict[str, Any]:
        """JSON-safe description (everything except the callable)."""
        return {
            "id": self.id, "version": self.version, "stage": self.stage, "kind": self.kind,
            "status": self.status, "parameters": self.parameters, "added_in": self.added_in,
            "notes": self.notes, "references": [r.as_dict() for r in self.references],
        }


class MethodRegistry:
    def __init__(self, methods: Iterable[Method] = ()):
        self._methods: dict[tuple[str, str], Method] = {}
        for m in methods:
            self.register(m)

    def register(self, method: Method) -> Method:
        key = (method.id, method.version)
        if key in self._methods:
            raise MethodError(f"{method.ref} is already registered; versions are immutable")
        if method.status == "default" and any(m.status == "default" for m in self.versions(method.id)):
            raise MethodError(f"{method.id} already has a default version")
        self._methods[key] = method
        return method

    def get(self, method_id: str, version: str) -> Method:
        try:
            return self._methods[(method_id, version)]
        except KeyError:
            known = [m.version for m in self.versions(method_id)]
            if not known:
                raise MethodError(f"unknown method id {method_id!r}") from None
            raise MethodError(f"{method_id} has no version {version!r}; known: {', '.join(known)}") from None

    def versions(self, method_id: str) -> list[Method]:
        return sorted((m for (i, _), m in self._methods.items() if i == method_id), key=lambda m: _version_key(m.version))

    def ids(self) -> list[str]:
        return sorted({i for i, _ in self._methods})

    def all(self, stage: str | None = None) -> list[Method]:
        out = [m for mid in self.ids() for m in self.versions(mid)]
        return [m for m in out if stage is None or m.stage == stage]

    def default(self, method_id: str) -> Method:
        versions = self.versions(method_id)
        if not versions:
            raise MethodError(f"unknown method id {method_id!r}")
        defaults = [m for m in versions if m.status == "default"]
        if len(defaults) != 1:
            raise MethodError(f"{method_id} has {len(defaults)} default versions, expected exactly one")
        return defaults[0]


def _version_key(version: str) -> tuple[str, int]:
    name, _, n = version.partition("@")
    return name, int(n)

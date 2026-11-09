"""Protected project labels: a label matching QRH_PROTECTED_LABELS (a regex) is shown as private-N."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

ENV_VAR = "QRH_PROTECTED_LABELS"


class PrivacyError(ValueError):
    """QRH_PROTECTED_LABELS is not a valid regular expression."""


def label_map(sessions: Iterable[dict[str, Any]], env: Mapping[str, str]) -> dict[str, str]:
    """Real label -> ``private-N``, numbered by the earliest session start (unset: nothing is protected)."""
    pattern = env.get(ENV_VAR, "")
    if not pattern:
        return {}
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise PrivacyError(f"{ENV_VAR} is not a valid regular expression: {exc}") from exc
    first: dict[str, str] = {}
    for s in sessions:
        label = s.get("project_label") or ""
        if label and rx.search(label):
            start = s.get("start") or ""
            if label not in first or start < first[label]:
                first[label] = start
    ordered = sorted(first, key=lambda lbl: (first[lbl], lbl))
    return {lbl: f"private-{i}" for i, lbl in enumerate(ordered, 1)}

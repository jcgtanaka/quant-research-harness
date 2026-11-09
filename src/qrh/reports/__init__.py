"""Visual reports: PDF pages and chart-data JSON built only from receipts and stage artifacts.

Public API: ``render_stage(trial_id, stage)`` and ``render_trial(trial_id)`` (see ``render``).
Importing this package does not import matplotlib; the renderers do.
"""

from __future__ import annotations

from typing import Any


def render_stage(*args: Any, **kwargs: Any) -> Any:
    from qrh.reports.render import render_stage as _render_stage

    return _render_stage(*args, **kwargs)


def render_trial(*args: Any, **kwargs: Any) -> Any:
    from qrh.reports.render import render_trial as _render_trial

    return _render_trial(*args, **kwargs)


__all__ = ["render_stage", "render_trial"]

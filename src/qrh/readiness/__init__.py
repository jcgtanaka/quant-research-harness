"""Deployment readiness: every warning of a trial's workflow in one document, acknowledged by a human with a signature.

Public API: ``is_acknowledged(trial_id) -> (bool, reason)``; the warnings are in ``qrh.readiness.collect``, the
report in ``qrh.readiness.report`` and the signature in ``qrh.readiness.ack``. Importing this package does not
import matplotlib.
"""

from __future__ import annotations

from qrh.readiness.ack import is_acknowledged
from qrh.readiness.collect import ReadinessError

__all__ = ["ReadinessError", "is_acknowledged"]

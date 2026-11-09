"""Configuration: where private state lives."""

from __future__ import annotations

import os
from pathlib import Path


class QrhHomeError(RuntimeError):
    """QRH_HOME cannot be created or written."""


def qrh_home() -> Path:
    """QRH_HOME env var if set, else ``~/.qrh``."""
    env = os.environ.get("QRH_HOME")
    return Path(env) if env else Path.home() / ".qrh"


def ensure_home() -> Path:
    """Return the home directory, creating it; raise QrhHomeError if unusable."""
    home = qrh_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise QrhHomeError(f"QRH_HOME {str(home)!r} is not usable: {exc}") from exc
    if not os.access(home, os.W_OK | os.X_OK):
        raise QrhHomeError(f"QRH_HOME {str(home)!r} is not writable")
    return home

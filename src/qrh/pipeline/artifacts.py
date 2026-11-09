"""Stage artifacts: files a stage persists so reports can be built from receipts alone.

A stage writes each artifact under ``<home>/trials/<trial_id>/<stage>/`` and records
``{"path", "hash"}`` (path relative to the home, sha256 of the file bytes) under
``params["artifacts"]`` in its receipt. Reading re-verifies the hash, so a changed,
missing or relocated file is an error, never silently used.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrh.hashing import sha256_bytes
from qrh.pipeline.receipt import Receipt, Stage

ARTIFACTS_KEY = "artifacts"


class ArtifactError(ValueError):
    """An artifact is missing, unreadable, outside the home, or does not match its receipt."""


def clean(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars become Python numbers, non-finite floats become None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def artifact_dir(home: str | Path, trial_id: str, stage: Stage | str) -> Path:
    return Path(home) / "trials" / trial_id / Stage(stage).value


class Artifacts:
    """Collects the artifacts one stage writes; ``as_params()`` is what the receipt records."""

    def __init__(self, ctx: Any, stage: Stage):
        self.home, self.trial_id, self.stage = Path(ctx.home), ctx.trial_id, stage
        self._entries: dict[str, dict[str, str]] = {}

    def _write(self, name: str, ext: str, data: bytes) -> None:
        rel = Path("trials") / self.trial_id / self.stage.value / f"{name}.{ext}"
        path = self.home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._entries[name] = {"path": rel.as_posix(), "hash": sha256_bytes(data)}

    def json(self, name: str, obj: Any) -> None:
        text = json.dumps(clean(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._write(name, "json", text.encode("utf-8"))

    def frame(self, name: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf)
        self._write(name, "parquet", buf.getvalue())

    def add_existing(self, name: str, rel: str, digest: str) -> None:
        """Record a file the stage already wrote (path relative to the home, hash of its bytes)."""
        self._entries[name] = {"path": rel, "hash": digest}

    def as_params(self) -> dict[str, dict[str, str]]:
        return {k: dict(v) for k, v in sorted(self._entries.items())}


def read_artifact(home: str | Path, receipt: Receipt, name: str) -> Any:
    """Load a recorded artifact (dict for ``.json``, DataFrame for ``.parquet``) after verifying its hash."""
    entries = receipt.params.get(ARTIFACTS_KEY) or {}
    entry = entries.get(name)
    if not isinstance(entry, dict) or "path" not in entry or "hash" not in entry:
        raise ArtifactError(f"{receipt.stage.value}: artifact {name!r} is not recorded in the receipt")
    root = Path(home).resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ArtifactError(f"{receipt.stage.value}: artifact {name!r} points outside the home directory")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"{receipt.stage.value}: artifact {name!r} cannot be read: {exc}") from exc
    if sha256_bytes(data) != entry["hash"]:
        raise ArtifactError(f"{receipt.stage.value}: artifact {name!r} does not match the hash in its receipt")
    try:
        if path.suffix == ".json":
            return json.loads(data.decode("utf-8"))
        return pd.read_parquet(io.BytesIO(data))
    except Exception as exc:
        raise ArtifactError(f"{receipt.stage.value}: artifact {name!r} cannot be parsed: {exc}") from exc

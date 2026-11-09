"""Write a synthetic two-column file: ``python make_data.py <out_dir> [--clean]``.

``derived`` tracks ``reference`` within a tiny tolerance, except (unless
``--clean``) over one contiguous period where it is replaced by an unrelated
synthetic series with a different level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from qrh.synthetic.generator import Regime, SyntheticConfig, generate

PERIODS = 400
CORRUPT_ROWS = slice(200, 260)


def _series(seed: int, start_price: float):
    cfg = SyntheticConfig(
        seed=seed, start="2020-01-01", periods=PERIODS, freq="D", tz="UTC", start_price=start_price,
        regimes=[Regime(share=1.0, drift=0.0003, volatility=0.01)],
    )
    return generate(cfg)["close"]


def make(out_dir: str | Path, corrupt: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference = _series(21, 100.0)
    noise = np.random.Generator(np.random.PCG64(5)).normal(0.0, 0.0005, PERIODS)
    derived = reference * (1.0 + noise)
    if corrupt:
        other = _series(99, 250.0)
        derived.iloc[CORRUPT_ROWS] = other.iloc[CORRUPT_ROWS].to_numpy()
    frame = reference.to_frame("reference")
    frame["derived"] = derived
    path = out / "linked.csv"
    frame.to_csv(path)
    return path


if __name__ == "__main__":
    print(make(sys.argv[1], corrupt="--clean" not in sys.argv))

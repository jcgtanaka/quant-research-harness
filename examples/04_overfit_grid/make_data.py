"""Write a pure-noise synthetic price file: ``python make_data.py <out_dir>``."""

from __future__ import annotations

import sys
from pathlib import Path

from qrh.synthetic.generator import Regime, SyntheticConfig, generate


def make(out_dir: str | Path, seed: int = 7) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SyntheticConfig(
        seed=seed, start="2020-01-01", periods=1500, freq="D", tz="UTC",
        regimes=[Regime(share=1.0, drift=0.0, volatility=0.01)],  # zero drift, no trend: nothing to find
    )
    path = out / "prices.csv"
    generate(cfg).to_csv(path)
    return path


if __name__ == "__main__":
    print(make(sys.argv[1]))

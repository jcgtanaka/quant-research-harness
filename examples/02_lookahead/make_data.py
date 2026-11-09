"""Write the synthetic price file for this example: ``python make_data.py <out_dir>``."""

from __future__ import annotations

import sys
from pathlib import Path

from qrh.synthetic.generator import Regime, SyntheticConfig, generate


def make(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SyntheticConfig(
        seed=11, start="2020-01-01", periods=600, freq="D", tz="UTC",
        regimes=[Regime(share=0.5, drift=0.0008, volatility=0.01), Regime(share=0.5, drift=-0.0004, volatility=0.012)],
    )
    path = out / "prices.csv"
    generate(cfg).to_csv(path)
    return path


if __name__ == "__main__":
    print(make(sys.argv[1]))

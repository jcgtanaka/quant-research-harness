"""Generic strategies with disguised timing defects, on synthetic data.

One band strategy (long when the close is below a lower band; a sparse signal) in several variants. The adapter
source is kept as text so unit tests can execute it and pipeline tests can write it to a file and register it.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import pandas as pd

TEMPLATE = '''\
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView, DatasetRef, RunResult

NEG = -1
FAR_FUTURE = pd.Timestamp("2100-01-01", tz="UTC")


class Band:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {"window": [20], "k": [__K__]}

    def required_data(self) -> list[DatasetRef]:
        return [DatasetRef("prices")]

    def run(self, params: dict[str, Any], data: DataView, start: pd.Timestamp, end: pd.Timestamp,
            cost_multiplier: float = 1.0) -> RunResult:
        px = data.as_of(__ASOF__)["close"]
        ret = px.pct_change()
        mid = px.rolling(params["window"]).mean()
        sd = px.rolling(params["window"]).std()
        lower = mid - params["k"] * __SPREAD__
        below = (px < lower)
__SIGNAL__
        change = position.diff().fillna(position)
        trades = pd.DataFrame({"qty": change[change != 0]})
__EXECUTION__
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])
'''

LAGGED = "        strategy = (position.shift(1) * ret).fillna(0.0)"
SAME_BAR = "        strategy = (position * ret).fillna(0.0)"
PLAIN_SIGNAL = "        position = below.astype(float)"

#           k     as-of   spread           signal                                                         execution
VARIANTS: dict[str, tuple[float, str, str, str, str]] = {
    "clean": (2.0, "end", "sd", PLAIN_SIGNAL, LAGGED),
    # next bar return read through numpy, hidden from a `.shift(-1)` pattern
    "roll": (2.0, "end", "sd", "        nxt = np.roll(ret.to_numpy(), -1)\n"
             "        position = pd.Series(np.where(below.to_numpy() & (nxt > 0), 1.0, 0.0), index=px.index)", LAGGED),
    # negative shift through a named constant
    "negvar": (2.0, "end", "sd", "        nxt = ret.shift(NEG)\n        position = (below & (nxt > 0)).astype(float)", LAGGED),
    # spread taken from the whole frame
    "globalstd": (0.3, "end", "px.std()", PLAIN_SIGNAL, LAGGED),
    # ignores `end` and reads the whole frame
    "fullframe": (2.0, "FAR_FUTURE", "sd", "        position = (below & (ret.shift(-1) > 0)).astype(float)", LAGGED),
    # decides on the close and fills on the same close
    "sameday": (2.0, "end", "sd", PLAIN_SIGNAL, SAME_BAR),
}


def source(variant: str) -> str:
    k, asof, spread, signal, execution = VARIANTS[variant]
    return (TEMPLATE.replace("__K__", repr(k)).replace("__ASOF__", asof).replace("__SPREAD__", spread)
            .replace("__SIGNAL__", signal).replace("__EXECUTION__", execution))


def adapter(variant: str) -> Any:
    ns: dict[str, Any] = {}
    exec(compile(source(variant), f"<band:{variant}>", "exec"), ns)  # noqa: S102
    return ns["Band"]()


def prices(n: int = 500, seed: int = 3, vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC", name="timestamp")
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))}, index=idx)


def window(df: pd.DataFrame, warmup: int = 30) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df.index[warmup], df.index[-60]  # data after the window end stays in the frame (a hold-out)


EXPANDING = textwrap.dedent('''\
    import numpy as np
    import pandas as pd
    from qrh.adapters.base import DatasetRef, RunResult

    class Expanding:
        """A legitimate strategy built on expanding-window statistics (past data only)."""

        def parameter_space(self):
            return {"w": [1]}

        def required_data(self):
            return []

        def run(self, params, data, start, end, cost_multiplier=1.0):
            px = data.as_of(end)["close"]
            ret = px.pct_change()
            z = (px - px.expanding().mean()) / px.expanding().std()
            position = (z < -1.0).astype(float)
            strategy = (position.shift(1) * ret).fillna(0.0)
            window = slice(start, end)
            return RunResult(strategy.loc[window], pd.DataFrame(), position.to_frame("weight").loc[window])
    ''')

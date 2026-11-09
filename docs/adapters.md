# Writing a strategy adapter

An adapter is a small Python class that connects your strategy to the harness. The harness never looks inside your
strategy. It asks the adapter which parameters exist, which datasets it needs, and for the returns it produces over a
window. Everything the harness checks (look-ahead, cost stress, overfitting statistics) is built on those answers.

An adapter has no base class to inherit. It only has to provide three methods with the right signatures (the
`StrategyAdapter` protocol in `src/qrh/adapters/base.py`). The hypothesis file points to it with `adapter: "adapter:TrendAdapter"`,
which means the class `TrendAdapter` in `adapter.py`, next to the hypothesis.

## A complete example

This is `examples/01_clean_strategy/adapter.py`, a synthetic moving-average trend rule:

```python
from qrh.adapters.base import DataView, DatasetRef, RunResult

COST_BPS = 5.0  # cost per unit of turnover, in basis points of notional


class TrendAdapter:
    def parameter_space(self) -> dict[str, list[Any]]:
        return {"fast": [3, 10, 30], "slow": [60, 120, 240]}

    def required_data(self) -> list[DatasetRef]:
        return [DatasetRef("prices")]

    def run(
        self,
        params: dict[str, Any],
        data: DataView,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cost_multiplier: float = 1.0,
    ) -> RunResult:
        px = data.as_of(end)["close"]
        ret = px.pct_change()
        trend = px.rolling(params["fast"]).mean() - px.rolling(params["slow"]).mean()
        position = np.sign(trend).fillna(0.0)  # known at the close of bar t
        change = position.diff().fillna(position)
        cost = change.abs() * COST_BPS * cost_multiplier / 1e4
        strategy = (position.shift(1) * ret - cost.shift(1)).fillna(0.0)  # earned over bar t+1
        trades = pd.DataFrame({"qty": change[change != 0]})
        window = slice(start, end)
        return RunResult(strategy.loc[window], trades.loc[window], position.to_frame("weight").loc[window])
```

## What the adapter returns

`run` returns a `RunResult` with three pandas objects:

- `returns`: a `Series` of per-bar strategy returns over the window.
- `trades`: a `DataFrame` of the trades made.
- `positions`: a `DataFrame` of the positions held (the example uses a `weight` column).

## The three methods

- `parameter_space(self) -> dict[str, list[Any]]` lists every parameter and the values to try. The harness runs every
  combination, and the number of combinations feeds the overfitting statistics, so declare the whole grid honestly.
- `required_data(self) -> list[DatasetRef]` names the datasets the strategy reads, for example `DatasetRef("prices")`. A
  `DatasetRef` has a `name` and an optional `version`.
- `run(self, params, data, start, end) -> RunResult` runs one parameter combination between `start` and `end`.
  `params` is one combination from the grid, and `data` is a `DataView`.

## Reading data safely: `DataView.as_of`

`DataView.as_of(decision_time)` returns a copy of the rows that were available at `decision_time`. If the data has a
publication column, a row counts as available only after its publication time. Otherwise it is available at its index
time. The time must be timezone-aware.

Reading data past the decision time is look-ahead: the backtest uses information the strategy could not have had. A
typical case is normalizing with the whole series:

```python
signal = (px - px.mean()) / px.std()   # wrong: the mean and std include future bars
signal = (px - px.expanding().mean()) / px.expanding().std()   # fine: only past bars
```

The first version looks harmless and produces a better backtest, which is exactly why it is dangerous. The harness
detects this with two runtime tests. The truncation test runs your strategy on all the data and again on data cut at a
chosen time. The future-perturbation test (`perturbation_cutoffs`, default 20) reruns it with every row after a cutoff
replaced by different values. If the early positions, trades or returns change, the strategy used the future, however the
code is written (`np.roll(x, -1)`, a named negative shift and a whole-frame statistic are all caught). Both need the
strategy to be deterministic: seed every random source.

## Cost stress: `cost_multiplier`

Add `cost_multiplier: float = 1.0` as the last parameter of `run` and multiply your trading costs by it. For the cost
stress test the harness calls `run` again with a larger multiplier (the default is `cost_stress_multiplier: 2.0` in
`src/qrh/validation/defaults.py`) and checks that the result survives. If `run` has no `cost_multiplier` parameter, the
stress stage is BLOCKED.

## The execution declaration

The hypothesis must also declare how positions turn into returns:

```yaml
execution:
  lag_bars: 1
  price_column: close
```

`lag_bars` is the number of bars between deciding a position and earning its return (an integer of at least 1, with no
default). `price_column` is the column the position is held in. The P&L reconstruction check (`s2.pnl_reconstruction`)
recomputes returns from your reported positions using this declaration and compares them with what your adapter
reported. It catches same-bar execution and wrong units, which a truncation test cannot see. A default version of this
check runs for every trial: with no `execution` declared it tries each strictly positive numeric column of your dataset with
a one-bar lag. Report your positions (`RunResult.positions`) with the position held after each bar, or the check is BLOCKED; if
your returns include costs above the default bound or leverage, declare `execution` (with `max_cost_per_turnover`) so the
rebuild can match them.

## Strategies that use a language model

If a strategy asks a language model for its signal, the hypothesis must also declare an `llm` section, and every model
call must go through `qrh.llm.complete(model_id, prompt, *, params=None, as_of=None)`. Answers are recorded once and
replayed, so results are reproducible. The rules and the field format are in [docs/llm.md](llm.md).

## Why only source is loaded

The harness compiles the adapter, and any sibling module it imports from the hypothesis folder, from the source bytes
read at load time. It never uses cached bytecode (`__pycache__`) and writes none. The same bytes are hashed, so the
code hash in the receipt is the hash of the code that actually ran, even after a quick edit that leaves the file size
and timestamp unchanged. Keep everything the adapter needs next to the hypothesis (`src/qrh/adapters/loader.py`).

## Checklist before you run

1. Pick a fixed data window (`windows.development` in the hypothesis) and do not change it after seeing results.
2. Declare `execution` with `lag_bars` and `price_column`.
3. Add `cost_multiplier` to `run` and apply it to every cost.
4. Read data only through `data.as_of(...)`, and use no centered windows, backward fill or full-sample statistics.
5. Run `qrh lint` on the adapter, then let `qrh run` perform the truncation, future-perturbation and same-bar tests, and fix any divergence it reports.

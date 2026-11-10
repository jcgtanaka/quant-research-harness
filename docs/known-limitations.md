# Known limitations

This is an alpha release. The issues below were found by end-to-end tests that drove the harness the way a
retail trader would, with synthetic data. They are listed so that you can judge the results you get. Fixes are
in progress; each one will be removed from this page when it ships. Security limits of the guards are listed
separately in [security.md](security.md).

None of these turns a planted look-ahead or same-bar execution into a PASS: those are covered by the
future-perturbation and same-bar checks (see [methods.md](methods.md)). They do mean that some results need a
human look before you trust them.

## Methodology

- **Declared stability neighbours can replace the generated ones.** If a hypothesis declares its own neighbour
  values (`stability: {neighbours: ...}`), those are used instead of the harness's own +/- step neighbours. A
  strategy whose behaviour repeats along a parameter (for example a modulo of that parameter) can pass the
  stability check with neighbours that behave exactly like the selected value. Until this is fixed, do not
  declare neighbours by hand, and read the stability page of the S5 report.
- **Small integer parameters.** A 10% nudge of a small integer rounds to a large relative jump (for example a
  holding period of 1 becomes 2, a 100% change). The neighbour check can then fail an honest strategy, and the
  reason text does not name the neighbour value that was used.
- **Near-duplicate variants make PBO noisy.** A grid whose variants are almost identical (for example the same
  rule with a tiny cost difference) can get a high PBO from noise, so a single strategy and the same strategy
  plus a near copy can receive different S5 verdicts. The deflated Sharpe luck bar itself cannot be lowered by
  near copies.
- **`stability_steps` is validated late.** Invalid values (empty list, zero, negative, above 1, text) are only
  rejected at S5, after S0-S4 have run, and a hypothesis may shrink the steps without a trader answer.
- **The future-perturbation test samples cutoffs.** Every position change is tested up to a cap
  (`perturbation_change_cap`); beyond it, change bars are sampled and the coverage is reported in the receipt
  and as a readiness warning. A leak confined to unsampled bars, a multi-bar-ahead leak, a leak visible only in
  the returns of a dataset with a publication column, or state kept outside the adapter module can still pass.
- **The same-bar check can stop S2 as INCONCLUSIVE** when the reported returns cannot be rebuilt from a price
  column (costs above the default bound, positions in shares or contracts). Declare `execution` in the
  hypothesis. A strategy that never takes a position gets the same message, which blames costs instead of the
  absence of trades.

## Data checks

- **A x100 level jump in the middle of a price series passes S1-S3.** It is only caught when the dataset declares
  `adjustments: none` and `qrh data verify` is run.
- **Timestamps without a timezone pass a rule that declares `timezone: UTC`.**
- **A bare cost number is accepted as exact.** `cost_per_contract: 1.0` with no confirmed broker commission record
  passes S2, and readiness does not warn that exchange, regulatory and clearing fees were not checked.
- **Re-registering an identical hypothesis is not refused**, and the `difference_statement` is not checked for
  content.

## Messages and tooling

- `qrh bench run` stores the suite verdict in the receipt but does not print it, and exits 0 either way.
- `thresholds: {}` is rejected although an empty mapping should mean "use the defaults".
- `qrh status | head -1` prints `BLOCKED: BrokenPipeError` when the pipe closes.
- `qrh lint` does not report LLM001 (undeclared model calls), which S2 does report.
- S4 shows a raw `ValueError` when a strategy is flat for the whole first fold.
- Some texts repeat: an S2 warning shown twice, a doubled sentence in the `variants-outside-harness` question, and
  a duplicated `unknown` choice id in `bar-type-declared`. Some docs say 37 provenance questions; `qrh pack add`
  reports 40.
- The interview is long for a beginner (about 21 questions for one ETF idea), and the four parameter-stability
  questions use statistical terms without plain explanations. `qrh interview next` shows them one at a time.
- The S0 notice "family count is now N" and the trial count charged by the deflated Sharpe can differ.
- There is no command to add or confirm a broker commission record; it is written as JSON.
- `qrh search` and the `qrh status` list show no date or stage, and a search with no match prints nothing.
- The look-ahead lint banner offers `qrh approve` although the right action is to fix the code.
- The full test suite takes about 17 minutes on a standard CI runner.

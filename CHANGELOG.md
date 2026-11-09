# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Parameter stability check at S5 (`s5.param_stability@1`). Around the selected setting, one parameter at a time is nudged by each
  of `stability_steps` (default 10% and 20%) up and down; grid points at about that distance are used, other neighbours are run
  as non-trial runs, and none is closer than the smallest step. (a) The neighbour ratio is computed per parameter and the worst
  parameter decides (the pooled mean is information only, so a parameter that changes nothing cannot hide a spike). (b) The
  plateau is local: the share of nearby numeric settings that keep `stability_closeness` of the SELECTED setting's out-of-sample
  Sharpe must reach `stability_plateau_share`; the old global share (against the grid's best) is kept as information only,
  because it depends on how the grid is spaced. (c) Per-parameter gaps and path correlations name the most fragile parameter.
  Numbers written as text are numbers; a non-numeric parameter with several grid values is varied. Neighbour runs are not trials
  and do not change the trial or family count. NOT APPLICABLE only for no parameters or a non-positive selected Sharpe; a
  strategy whose parameters are all fixed non-numeric values is BLOCKED; declared neighbours closer than the smallest step are
  refused. The optional hypothesis `stability` block (neighbours, bounds) is listed in the readiness report. Plateau idea after
  Carver, *Systematic Trading* (2015); the rest is in-house.
- Four new thresholds, default `stability_closeness` 0.70, `stability_plateau_share` 0.50, `stability_neighbour_ratio` 0.50 and
  `stability_steps` [0.10, 0.20], that are always asked of the trader: four built-in interview questions (stage S0, scope
  `strategy:<family>`, "Keep the default" recommended, plus "I don't know") with the usual answer provenance. The answers are frozen
  into the hypothesis thresholds at registration; an unanswered or "I don't know" question leaves the default, `qrh new` prints a
  notice and the readiness report raises the medium warning "stability thresholds not confirmed by the trader". "Keep the default"
  counts as confirmed only when the frozen value is the default in force at registration, and a spec may not set a stability
  value against the trader's answer (or looser than the default when the trader did not answer): `qrh new` refuses it. The first
  three are tighter when higher in pack layering.
- Optional `param-stability` benchmark suite: planted spikes (grid and single combination) that must fail and planted plateaus
  that must pass.
- S5 report: a parameter stability page, and the deflated Sharpe page shows both Sharpe variances and which was used.
- Git pre-push approval: `qrh publish pre-push` refuses a push unless the user signed an approval for exactly this
  remote URL and set of ref updates (with the remote's current ids). The user installs a pinned hook file printed by
  `qrh publish print-hook`.
- Intake questionnaire: plain questions asked before a run, with answers checked and stored.
- Parity report: compares backtest daily results with real broker results. It is always written, the user chooses
  how many days to compare, and it never blocks anything by itself.
- Futures and options first slices of the `pnl.linear` check, with costs taken from broker commission records.
- Broker commission validator: a record that is missing, unconfirmed, stale, conflicting or without tiers blocks the
  cost instead of defaulting to zero.
- Slope limit: fails a run when reported profit does not scale one to one with explained profit.
- Bench verdict (PASS, FAIL or NO_RULE) for new bench runs.
- Read-only dashboard server (`qrh ui`) on the local address only, with trials, trial detail, benchmark and warnings
  pages. Not yet checked in a browser.
- P3e-a2 (2026-10-24), dataset onboarding provenance and guarding:
  - 37 provenance question banks, by asset class;
  - a data-guard hook that blocks reads of private datasets;
  - an optional onboarding benchmark suite;
  - stricter asset-class verification.
- P3e-a1: dataset onboarding (privacy-safe profile, instrument model, propose, verify, confirm, S1 dataset card).
- P3d-b2 part B: rules for strategies that use an LLM (training-cutoff gate, prompts and models counted as trials, replay
  cache), the `llm-gate` benchmark suite, and examples 08 and 09.
- P3d-b2 part A: deployment readiness report with a signed human acknowledgement (`qrh readiness report|ack|status`).

### Changed

- **Deflated Sharpe luck bar, stricter for some grids.** V, the variance of the null trials' Sharpe ratios, is now
  `max(cross-variant variance, sampling variance of the selected variant's out-of-sample Sharpe estimator)` for every grid size
  (before: the cross-variant variance for 2 or more combinations, the sampling variance only for one). A near-copy of a strategy
  (the same rule with 0.2 basis points more cost) used to collapse the cross-variant variance and lower the bar (one
  combination DSR 0.83 FAIL, the same plus a twin DSR 0.994 PASS); now both give the same verdict. A grid whose cross-variant
  variance already exceeds the sampling variance is unchanged; a grid of near-identical variants, or with a short out-of-sample
  series, now faces a higher bar and some results that used to pass S5 will fail. Both variances and the one used are recorded
  in the S5 receipt (`params.var_sharpe`), the `dsr_inputs` artifact and the report.
- README roadmap updated to show what is now built; remaining work is listed under Next and Planned additions.

### Fixed

Three methodology gaps found by a simulated-trader end-to-end test, where simple disguises defeated the deterministic checks:

- Sparse look-ahead passed S2 to S4 and the lint: `np.roll(ret, -1)` and `ret.shift(NEG)` (a named constant) read the next
  bar, and the three truncation cutoffs rarely landed on an active signal. S2 now also runs a future-perturbation
  invariance test (`s2.future_perturbation@1`): at `perturbation_cutoffs` seeded cutoffs (new frozen threshold, default 20,
  half of them where the position is non-zero or changes) the adapter is rerun with all data after the cutoff replaced, and
  positions, trades and returns up to the cutoff must be identical, or S2 FAILs with the first divergent timestamp. Cost: one
  adapter run per cutoff. The truncation test is unchanged. An unreproducible adapter is BLOCKED, never FAIL; the test is not
  run for strategies that declare an `llm`. The lint gains `LA006` (`np.roll` with a negative shift, BLOCK) and `LA007` (a shift
  or roll amount that is not a literal number, WARN), and resolves a name bound once to a number (`NEG = -1`).
- Same-bar fills (decide and fill on the same close) were detected only by the opt-in `s2.pnl_reconstruction`. S2 now runs
  `s2.same_bar_execution@1` for every trial: with no `execution` declared it tries each strictly positive numeric dataset
  column with a one-bar lag. BLOCKED without reported positions, INCONCLUSIVE (never PASS) when no column explains the
  returns, skipped where `s2.pnl_reconstruction` is pinned or a `reconstruction` block (`pnl.linear`) is declared. Planted same-bar detection in the quick benchmark rose from 0/27
  to 27/27 (percent-unit returns from 12/18 to 18/18) with no false alarm on the clean controls.
- A hypothesis with exactly one combination could never pass S5 (`ValueError: PBO needs at least 2 strategies`). With one
  combination nothing is selected, so PBO is reported as "not applicable (one combination, no selection among variants)" in
  the receipt and report, no lambda figure is drawn, and the deflated Sharpe ratio decides with the family trial count
  charged. Its Sharpe variance is the sampling variance of the out-of-sample Sharpe ratio, since a single column has no
  cross-trial variance. Two or more combinations are unchanged.
- Review hardening of the three fixes: a dummy `llm` block no longer switches the perturbation test off (an LLM strategy
  whose full run made model calls is perturbed in replay mode, with a stub answer after the cutoff; a test that cannot run is
  INCONCLUSIVE); cutoffs are every position-change bar and the bar before it (new frozen threshold `perturbation_change_cap`,
  default 40) plus the spread cutoffs, with the coverage recorded and a readiness warning below 100%; every perturbed run
  uses a freshly loaded adapter and a test that changes nothing is BLOCKED (memoizing adapters); rows published after the
  cutoff are perturbed (a publication-lag leak fails), as are boolean, text and categorical columns; an inferred same-bar
  FAIL becomes INCONCLUSIVE when positions are reported as held during the bar; the benchmark no longer crashes on a
  one-combination matrix.
- Changed behaviour that existing tests encoded: a non-literal `.shift(n)` is now a `LA007` warning (it was silent), the
  default benchmark arm now detects same-bar execution, and readiness no longer says same-bar execution is undetectable by
  default.

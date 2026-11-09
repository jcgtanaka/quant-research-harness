# Methods and benchmarks

The harness must get better from science and from experience without modifying itself. Three rules make that safe:

1. **Methods are versioned and cited.** Every check and statistic is a registered implementation with an id, a version and
   references. A version never changes; an improvement is a new version.
2. **Receipts say which version ran.** Every stage receipt records `methods: {id: version}`, and anything that is not the
   current default is flagged.
3. **Only measured wins, and only humans, change a default.** A new version starts experimental. It becomes the default only
   if a benchmark receipt with known truth shows it is no worse anywhere and a human approves.

## The registry

`qrh.methods.registry.Method`:

| Field | Meaning |
|---|---|
| `id` | what it does, by stage: `s5.pbo`, `s4.walkforward`, `s1.quality.gaps`, `metrics.periods_per_year` |
| `version` | `<name>@<n>`, for example `cscv@1`. Immutable once released |
| `stage`, `kind` | the stage that uses it (S1-S6; the shared estimator `metrics.periods_per_year` is listed under S3) and what it is (`rule`, `check`, `estimator`, `procedure`, `statistic`) |
| `entry` | the callable. Stages call it; they never import the implementation |
| `parameters` | the parameters it understands, with their defaults |
| `status` | `default`, `experimental` or `deprecated` (see below) |
| `references` | list of `{citation, doi, arxiv, url, verified}`; `verified` is true only when an identifier was checked against the publisher record. Identifiers are never guessed: unknown means empty and `verified: false` |
| `added_in` | the harness version that introduced it |
| `notes` | what it does and its known limits |

A reference is written `<id>=<version>` (`s5.pbo=cscv@1`). The CLI also accepts `s5.pbo@cscv@1`.

An id that ships with no default version is *opt-in*: it runs only for a hypothesis that pins it (`methods:` in the spec),
it is left out of the full map recorded at S0 unless pinned, and using it is always flagged as experimental.
Statuses: exactly one `default` per id (used when a hypothesis does not pin a version), any number of `experimental`
versions (usable, flagged everywhere), and `deprecated` versions (kept so old receipts reproduce; refused when registering a new
hypothesis). Public defaults ship in `qrh.methods.builtin`. A local change (a promotion) is stored in
`<QRH_HOME>/methods/state.json` with who, why, which benchmark and which approval; it never edits the code and applies to
that home only.

```bash
uv run qrh methods list [--stage S5] [--json]     # id, version, status (a local change shows as "(local)"), references
uv run qrh methods show s5.pbo@cscv@1             # parameters, notes, references with identifiers and verified flag
```

### Shipped methods (all version 1 unless noted)

| id | version | status | references |
|---|---|---|---|
| `s1.quality.{schema,range,relationship,null_rate_drift,gaps,timezone}` | `<rule>@1` | default | in-house |
| `s2.lookahead_lint` | `ast_rules@1` | default | in-house |
| `s2.truncation` | `prefix_truncation@1` | default | in-house |
| `s2.future_perturbation` | `future_perturbation@1` | default (new check, no predecessor; adds to `s2.truncation`) | in-house; no reference cited |
| `s2.same_bar_execution` | `same_bar@1` | default (new check, no predecessor) | in-house; no reference cited |
| `s2.sanity` | `bounds@1` | default | in-house |
| `s2.pnl_reconstruction` | `pnl_reconstruction@1` | **experimental, opt-in (no default)** | in-house |
| `s2.llm_gate` | `llm_gate@1` | default (new check, no predecessor; inert without `llm`) | in-house; no reference cited |
| `llm.replay_cache` | `replay_cache@1` | default (new, no predecessor; used only with `llm`) | in-house; no reference cited |
| `metrics.periods_per_year` | `observed_density@1` | default | in-house |
| `s4.walkforward` | `rolling_anchored@1` | default | in-house |
| `s4.walkforward` | `purged_embargo@1` | **experimental** | Lopez de Prado (2018), Advances in Financial Machine Learning, ch. 7 (book, no DOI, unverified) |
| `s5.pbo` | `cscv@1` | default | Bailey, Borwein, Lopez de Prado, Zhu (2017), DOI 10.21314/JCF.2016.322 (verified) |
| `s5.dsr` | `bailey_lopezdeprado2014@1` | default | Bailey, Lopez de Prado (2014), DOI 10.3905/jpm.2014.40.5.094 (verified) |
| `s5.param_stability` | `param_stability@1` | default (new check, no predecessor) | Carver (2015), Systematic Trading (book, unverified; the plateau idea only) |
| `s6.bootstrap` | `stationary_politis_romano1994@1` | default | Politis, Romano (1994), DOI 10.1080/01621459.1994.10476870 (verified) |

The three DOIs were checked against the Crossref record (title, authors, journal, year).

### Entry contracts

An implementation can be swapped only for one with the same call shape, so each id has a contract:

| id | `entry(...)` returns |
|---|---|
| `s1.quality.<type>` | `(df, rule, offenders) -> list[str]` (reasons; empty is a pass) |
| `s2.lookahead_lint` | `(paths) -> list[Finding]` |
| `s2.truncation` | `(adapter, frame, params, start, end, publication_column, cutoffs, atol, full_result) -> TruncationResult` |
| `s2.future_perturbation` | `(adapter, frame, params, start, end, publication_column, n_cutoffs, atol, full_result, max_change_bars, adapter_factory, context) -> PerturbationResult` (`verdict` PASS, FAIL or BLOCKED, `reasons`, `details`) |
| `s2.same_bar_execution` | `(run_result, frame, execution_or_None, own_bar_probe=None) -> PnlResult` |
| `s2.sanity` | `(run_result, bounds) -> SanityResult` |
| `s2.pnl_reconstruction` | `(run_result, frame, execution) -> PnlResult` (`verdict` PASS, FAIL, INCONCLUSIVE or BLOCKED, `reasons`, `details`), plus `validate(execution)` |
| `metrics.periods_per_year` | `(DatetimeIndex) -> float or None` |
| `s4.walkforward` | `(adapter, view, bars, is_returns, params_by_label, raw_spec, periods_per_year, run_kwargs=None) -> WalkForwardResult`, plus `validate(raw_spec, n_bars)` that raises `ValueError` |
| `s5.pbo` | `(matrix, n_blocks) -> PBOResult` (needs at least two columns; S5 does not call it for a single combination, see below) |
| `s5.dsr` | `(returns, n_trials, var_sharpe) -> DSRResult` |
| `s5.param_stability` | `(grid, matrix, oos_index, picks, ppy, thresholds, adapter, view, start, end, declared) -> StabilityResult` (`status` PASS, FAIL or NOT_APPLICABLE, `reasons`; raises `StabilityBlocked`) |
| `s6.bootstrap` | `(returns, periods_per_year, reps, level, mean_block_length, seed) -> BootstrapResult` |

## How a method is chosen and recorded

- **Selection.** A hypothesis may pin versions: `methods: {s5.pbo: cscv@1}` in the spec. Unpinned ids use the default of the
  day the hypothesis is registered. At S0 the harness stores the *full* map (pins plus defaults) with the hypothesis,
  exactly like the thresholds: it is part of the hypothesis hash when present (older hypotheses keep their hash) and
  it cannot change afterwards. A promotion made after S0 does not change a registered trial; such a trial is simply flagged
  as using a non-default version.
- **Execution.** The spec field `execution: {lag_bars: 1, price_column: close}` (or `price_columns: [...]`, and an optional
  `max_cost_per_turnover`) is recorded at S0 and frozen like the thresholds, and is part of the hypothesis hash only when
  present. `lag_bars >= 1` is required and has no default: a hypothesis that selects `s2.pnl_reconstruction` without it is BLOCKED at S2.
- **Refusals.** Pinning an unknown id or version, or a deprecated version, is refused at registration. A stage whose
  state file cannot be read, or whose id has no or several defaults, is BLOCKED.
- **Receipts.** Every stage receipt has `methods` (S0 holds the full map; S1-S6 the ids that stage actually used; S7 the union).
  A non-default version also appears as `params["non_default_methods"]`. `methods` joins the receipt hash only when it
  is non-empty, so receipts written before the registry existed still verify.
- **Reports.** The markdown report has a `## Methods` section with `EXPERIMENTAL` next to non-default versions; the S0 text
  card and the S7 summary page say so too.
- **Reproducibility.** Re-running a trial reproduces every metric exactly. A second trial that pins the versions recorded in
  a receipt reproduces its metrics too (except the DSR, which also depends on how many trials the family has logged).

## Benchmarks with known truth (`qrh bench`)

All data is synthetic, seeded and deterministic: the same seed gives the same receipt hash. Each suite runs the
registered methods themselves, selected through the registry, over exactly the same generated data for every arm.

| Suite | What it measures | Truth |
|---|---|---|
| `planted-bugs` | The S1 and S2 detectors over a generated corpus of defects plus clean controls: detection rate per bug class (and per variant) and the false-alarm rate on controls | each case is built with, or without, a known defect |
| `noise-fpr` | Pure-noise strategy families (grid sizes 12, 36, 120; lengths 500, 1000, 2000 bars in `--full`) through S3-S5 (walk-forward gate, PBO, DSR): the share that passes is the empirical false-positive rate, reported with an exact binomial interval next to the nominal level `1 - min_dsr` | there is no edge in the data |
| `known-edge-power` | The same pipeline on data with a planted persistent trend of several strengths: the share that passes is the power curve | the edge is planted and its size is known |
| `llm-gate` (optional) | The S2 detectors for strategies that use a language model: the cutoff gate, the undeclared-LLM lint and the replay of truncated runs, over four planted classes (`llm_cutoff_violation`, `llm_cutoff_unknown`, `llm_undeclared`, `llm_prompt_lookahead`, several variants each) plus clean LLM controls; per-class detection and false alarms on the controls (see [llm.md](llm.md)) | each case is built with, or without, a known defect |
| `param-stability` (optional) | `s5.param_stability` on planted families of settings: **spikes** (one setting profitable, its neighbours earn nothing; a 3x3 grid centre, a 3x3 corner, a 5x5 grid, one pre-registered setting with one and with two parameters) must FAIL, and **plateaus** (the same edge everywhere, or an edge falling off smoothly; grids and single settings) must PASS; detection per variant and false alarms on the plateaus. Also: a spike next to a parameter that changes nothing, hairline grid points or declared neighbours, settings written as text, and a robust strategy on a wide log-spaced grid (must pass) | the edge shape is planted and known |
| `history` (optional) | Stored trials re-evaluated with a candidate method versus the default: which verdicts through S5 would change, and whether to stricter or looser. Read-only | none; it measures what a change would do to your own history |

The planted-bug corpus: look-ahead through `shift(-k)` for several k (literal, and through a variable that the static lint
cannot read), centered rolling windows (literal and through a variable), backward fill (direct, with a limit, through
`getattr`), full-sample normalization (price z-score, demeaned return), same-bar execution (three volatilities times three signal smoothings), percent-unit returns (three volatilities times two
signals), relationship corruption at three magnitudes and three lengths, and gaps and duplicated timestamps of
several sizes; controls are seven clean strategies (moving-average crossover, return momentum, mean reversion, lagged
return, EWM momentum, expanding z-score, forward fill). A detection means the harness's blocking verdict: the
lint reported a BLOCK, the truncation test or the sanity bounds did not PASS, or an S1 quality rule did not PASS (a lint
warning alone does not count, because it does not stop a run). The results also list which detector fired.

The corpus uses daily synthetic bars; the harness code under test infers bars per year from the timestamps and keys
nothing on a symbol, a strategy name or a bar size.

```bash
uv run qrh bench run                                    # --quick (default): about 10 s
uv run qrh bench run --full                             # thorough: a few minutes on one core
uv run qrh bench run --suite noise-fpr --reps 300 --seed 7
uv run qrh bench run --suite llm-gate --full           # about a minute
uv run qrh bench run --suite param-stability           # planted spikes must fail, planted plateaus must pass
uv run qrh bench run --full --method s4.walkforward=purged_embargo@1 --compare-to default
uv run qrh bench run --suite history --history /path/to/a/harness/home --method s5.pbo=cscv@2
uv run qrh bench compare <bench_id_a> <bench_id_b> [--all] [--json]
uv run qrh bench compare <bench_id>                     # one receipt: its default arm against its candidate arm
uv run qrh bench report <bench_id|path/to/receipt.json> [--out DIR]   # re-render text, PDF and chart JSON from a receipt
uv run qrh bench list
```

`--quick` (the default, used by the tests) and `--full` set suite sizes; `--reps` overrides the replications. `--method`
selects a candidate version for the candidate arm; `--compare-to default` also runs the default arm on the same data and
stores both. The `history` suite needs `--method` (it compares the candidate with the default) and a harness home to read;
it only reads `receipts/` and the files they record, verifies every hash, skips with a reason anything that does not
verify, and never writes to that directory.

### Outputs

`<QRH_HOME>/bench/<bench_id>/` (`bench_id` is `bench-` plus the first 12 characters of the receipt hash):

```
receipt.json   method versions, overrides, suite configuration, seeds, results, hashes (config, results, code, history input)
report.txt     the same results as text tables
report.pdf     one card, planted-bug detection chart and variant tables, noise false positives against the nominal level,
               power curve, and, with a baseline, the default-versus-candidate table (existing report theme)
chart.json     the figure data
```

The one-source rule of the stage reports holds here: the PDF is drawn from `chart.json` only, and `chart.json` and the text
are built from the receipt; nothing is recomputed. Re-rendering gives byte-identical files.

`qrh bench compare a b` prints counts, the delta (b minus a), the Fisher exact p-value and whether the difference is
distinguishable at 0.05 given the replications. Two receipts from different seeds also differ by sampling noise; the output
says so. Within one receipt the two arms saw identical data and Fisher's test, which treats them as independent, is
conservative.

### Reading the numbers honestly

- Detection rates are about this corpus. A 100% rate says the harness catches these patterns, not all look-ahead. A 0% rate
  is a gap to be closed by a new method version, measured by this bench (see "Closing the same-bar and units gaps" below).
- Relationship-corruption variants whose magnitude is at or below the rule's tolerance (`rel_tol` 0.01) cannot be flagged by that
  rule by design. They are labeled `expected_undetectable`, reported on their own (`expected_undetectable/...` metrics, not gated)
  and never counted as misses in `detect/relationship_corruption`.
- The noise false-positive rate is for the whole S3-S5 pass (walk-forward gate, PBO and DSR together), which is the
  question a researcher asks; the per-gate breakdown is in the receipt. The nominal level is the DSR gate's `1 - min_dsr`.
- Power depends on the strategy family, the sample length and the edge model (a persistent latent drift followed by a
  moving-average grid). Treat the curve as a comparison tool between method versions, not as a forecast.

## How science enters

1. **Evidence card.** Someone (a person, later the literature scout) writes down: the claim and the source (citation with an
   identifier that was actually checked), the expected effect (which gated benchmark metric should move, in which direction),
   and the method id and version to add. Until the scout exists this is a short note in the pull request.
2. **Experimental method.** The implementation is added to `qrh.methods.builtin` as a new version with its references and
   `status="experimental"`. It is selectable (`methods:` in a spec, `--method` in a bench) and flagged everywhere it is used.
   It cannot become the default by being added.
3. **Bench.** `qrh bench run --full --method <id>=<version> --compare-to default` writes a receipt with both arms over the same data.
4. **Human promotion.** `qrh methods promote` checks the receipt against the rules below and then waits for a human
   `qrh approve`.

### Promotion rules (`qrh.methods.promotion.check_promotion`)

All must hold:

1. The bench receipt overrides exactly the candidate and was run with `--compare-to default`; the two arms differ in no other method.
2. The default arm used the id's current default (a receipt measured against an older default is stale).
3. The receipt meets the `full` profile (replications and cells) and covers `planted-bugs`, `noise-fpr` and `known-edge-power`.
4. Every gated metric is no worse for the candidate than for the default, by point estimate: detection rate per bug class,
   false-alarm rate on controls, false-positive rate per noise cell, power per planted-trend strength. One worse metric
   refuses the promotion. There is no statistical tolerance, because both arms saw identical data.
5. A noise cell where the candidate's interval lies above the nominal level must already have done so for the default.
6. At least one gated metric is strictly better. If every metric ties, a documented `--tie-reason` (at least 30
   characters) is required, for example a better-founded method with identical results.
7. The receipt was measured on the current harness code (equal code hash), so it describes what would become the default.
8. A human approved this exact action.

```bash
uv run qrh methods promote s4.walkforward@purged_embargo@1 --bench <bench_id> --reason "..." [--tie-reason "..."]
# prints an approval id and exits 2; then, in your own terminal (never through the agent):
uv run qrh approve <approval_id> --reason "I read the bench report"
uv run qrh methods promote ...                      # the identical command now applies the change (single use)
uv run qrh methods demote s4.walkforward@purged_embargo@1 --restore s4.walkforward@rolling_anchored@1 --reason "..."
uv run qrh methods deprecate s4.walkforward@purged_embargo@1 --reason "..."
```

Eligibility is checked before the approval request, so a human is never asked to approve something the rules refuse. The
approval is bound to the action, method, receipt hash and reasons, expires in 30 minutes and is consumed once. A promotion makes the
candidate the default and the previous default `deprecated`. A demotion returns the default to a named version and puts the
demoted one back to experimental. Deprecation applies to experimental versions only; the default can only be replaced.
Every change is appended to `state.json` `history` with the reason, the approval id and reason, and the bench id and hash.

## How experience enters

Lessons from real trials and failures will arrive as *experience packs*: versioned bundles of checks and rules (a public
`retail-core` pack, including LLM-assisted rules). They go through the same path: registered as
experimental methods or rules with their provenance, benchmarked (planted bugs for the checks, history for the effect on past
trials), and promoted by a human. Nothing in a pack applies itself.

## Demonstration: `s4.walkforward` `purged_embargo@1`

`cscv@1` already averages tied ranks in the out-of-sample ordering, so a tie-handling variant of `s5.pbo` would add
nothing. The demonstration is therefore a walk-forward variant: the same folds and selection rule as `rolling_anchored@1`, with
a purge gap (the last `purge` train bars are not used to select) and an embargo (the first `embargo` test bars are not run or scored),
default 5 bars each, adapted from the purging and embargo of Lopez de Prado (2018, ch. 7) to a forward-only split. The
benchmark comparison against the default, and its receipt, are in the next section. It is **not**
promoted.

## Demonstration results

`qrh bench run --full --method s4.walkforward=purged_embargo@1 --compare-to default` (seed 20240601, 5 min 53 s on one core).
The receipt, text report and PDF are in `examples/bench/purged_embargo_vs_default/` (bench id `bench-706b7baa279b`;
`qrh bench report examples/bench/purged_embargo_vs_default/receipt.json --out DIR` re-renders the chart JSON).

- Noise false positives through S3-S5: 0 of 1350 for the default (95% interval 0 to 0.3%) against a nominal level of 5%; the gates
  alone pass S4 47.3%, PBO 9.8%, DSR 0.5%.
- Power by planted trend strength 0.002 / 0.003 / 0.004 / 0.006 / 0.008: 4.0% / 15.0% / 43.0% / 80.0% / 93.0% (default);
  3.0% / 16.0% / 43.5% / 79.0% / 93.0% (`purged_embargo@1`). Planted-bug detection is identical in both arms (S4 is not a detector).
- Comparison: no difference is distinguishable (all Fisher p above 0.78). The candidate is one count worse in three gated
  cells (`fpr/g12_t1000`, `power/s0.002`, `power/s0.006`), so `qrh methods promote` refuses it ("3 gated metric(s) are worse
  than the default"). It stays experimental.

## Closing the same-bar and units gaps: `s2.pnl_reconstruction@1`

**The gap.** (Closed by default since `s2.same_bar_execution@1`, see above; this section records the original measurement of the opt-in method.) The first full benchmark showed 0% detection of same-bar execution (and 66.7% of percent-unit returns, all of
them caught only by the sanity bounds, none when the volatility is low). Truncation cannot see a timing or units defect in how an adapter
turns positions into returns: the positions do not change when the data is cut. The sanity bounds only see a return that is too large in absolute terms.

**The fix.** `s2.pnl_reconstruction@1` (experimental, opt-in) recomputes the strategy returns itself, from the adapter's reported
positions and the declared price column(s), with an enforced execution lag, and compares them with the reported returns. The
hypothesis declares `execution: {lag_bars: 1, price_column: close}` (`price_columns: [...]` for several assets; positions are
aligned to the columns by name, or one-to-one for a single column) and an optional `max_cost_per_turnover` (default 0.001).
`lag_bars >= 1` is required with no default: a hypothesis that selects the method without it is BLOCKED at S2.

- A reported return is consistent with a reconstruction when it is not above it (beyond float noise) and not below it by more
  than the cost bound times the turnover: costs may lower a return, nothing may raise it.
- FAIL "same-bar execution (look-ahead)" (or "execution lag k bar(s) is shorter than the declared lag"): on the bars where the
  shorter lag and the declared lag disagree, the adapter matches the shorter lag (match share of at least 0.9 and at least 0.5
  above the declared lag, or the same by correlation).
- FAIL "units mismatch (likely percent vs fraction)": the reported returns are the lagged (or same-bar) reconstruction times a
  power of ten other than 1 (correlation of at least 0.98, slope within 12% of 10^e).
- PASS only if at least 98% of the active bars match the lagged reconstruction; an unexplained mismatch is INCONCLUSIVE with the
  figures (match shares, Pearson and rank correlations), never PASS. A strategy that never holds a position is INCONCLUSIVE too.
- BLOCKED when it cannot run: no positions, price column missing, positions not alignable, fewer than 20 usable bars, bad spec.
- It does not look at the signal: a peeking signal with a correct execution lag passes here; lint and truncation own that.

It is wired into S2 beside lint, truncation and sanity, runs only where the hypothesis pins it, records its version in the S0 and S2
receipts, adds an `S2/pnl` artifact and a `pnl_reconstruction` block to the S2 receipt params, and the S2 verdict is BLOCKED over
FAIL over INCONCLUSIVE over PASS. Examples 01, 02 and 03 declare `execution` and select it: 01 still passes S0-S7, 02 still fails on lint
and truncation (the reconstruction passes: its execution lag is respected), and 03 now fails both through the sanity bounds and
through "units mismatch" (factor 10^2).

**Bench changes.** Relationship-corruption variants at or below the rule tolerance are now `expected_undetectable` (reported
separately, not counted as misses). The corpus has same-bar variants across three volatilities and three smoothings (9) and
percent-unit variants across three volatilities and two signals (6).

**Evidence** (`qrh bench run --full --method s2.pnl_reconstruction=pnl_reconstruction@1 --compare-to default`, seed 20240601,
6 min on one core; receipt, text and PDF in `examples/bench/pnl_reconstruction_vs_default/`, bench id `bench-f0751dc00863`).
Default S2 set against default plus `pnl_reconstruction@1`, 30 replications per variant:

| Bug class | Default | With the reconstruction |
|---|---|---|
| look-ahead shift | 150/150 | 150/150 |
| centered rolling | 90/90 | 90/90 |
| backward fill | 83/90 (92.2%) | 83/90 (92.2%) |
| full-sample normalization | 58/60 (96.7%) | 58/60 (96.7%) |
| same-bar execution | 0/270 (0.0%) | **270/270 (100%)** |
| percent units | 120/180 (66.7%) | **180/180 (100%)** |
| relationship corruption (above tolerance) | 180/180 | 180/180 |
| gaps and duplicates | 150/150 | 150/150 |
| false alarms on 7 clean controls | 0/210 (upper bound 1.7%) | 0/210 (upper bound 1.7%) |
| relationship corruption at or below the tolerance (expected undetectable) | 0/90 flagged | 0/90 flagged |

Noise false positives (0/150 in each of nine cells) and power at all five strengths are identical in the two arms, as they must be:
the method does not take part in S3-S5. The two improvements are distinguishable (Fisher p = 1.6e-161 for same-bar, 4.6e-21 for
percent units); nothing is worse. `qrh methods promote s2.pnl_reconstruction@pnl_reconstruction@1 --bench <id> --reason "..."` finds
it eligible and stops at the human-approval step (it prints an approval id); it was not approved, so the method stays experimental.
Promoting an opt-in id makes it the default of that home with nothing deprecated; from then on every new hypothesis there must declare `execution`.

## Default look-ahead and timing checks: `s2.future_perturbation@1` and `s2.same_bar_execution@1`

A simulated-trader test showed that simple disguises beat the three-cutoff truncation test and the opt-in reconstruction.
Two new checks run in S2 for every trial. Like the LLM rules below they are status `default` **without the promotion flow**:
new checks with no predecessor to beat. They add to `s2.truncation`, which is unchanged.

- `s2.future_perturbation@1` reruns the adapter at seeded cutoffs with the data after the cutoff replaced and requires
  positions, trades and returns up to the cutoff to be identical. This catches `np.roll(x, -1)`, `x.shift(NEG)` with a named
  constant, a whole-frame statistic and a read that ignores `end`, whatever the code looks like, including when the signal is
  rarely active at a truncation cutoff, also in a strategy that is always in the market. The cutoffs are every position-change
  bar and the bar before it (the frozen threshold `perturbation_change_cap`, default 40 change bars, 1 to 500, spread evenly
  beyond it) plus `perturbation_cutoffs` bars spread over the window (default 20, 1 to 200); lowering either needs the approval
  flow like the other floors. The receipt records `coverage` (change bars tested / total) and the readiness report warns below
  100%. Cost: one adapter run per cutoff (up to 100 extra runs by default, a few seconds for a vectorized pandas strategy);
  every perturbed run uses a freshly loaded adapter, and if no perturbed run differs from the original anywhere the result is
  BLOCKED ("perturbed data did not reach the strategy"), so a cache inside the adapter cannot hide a leak.
  Design limits: it assumes a deterministic adapter (an unreproducible run is BLOCKED, never FAIL); an adapter that raises on
  the perturbed data is BLOCKED. With a publication column, rows stamped before the cutoff but published after it are perturbed
  too (using a value before its publication time fails), and the reported returns are not compared (their realized prices move
  too); boolean, text and categorical columns are permuted, datetime and other dtypes are left as they are. A strategy that
  declares an `llm` runs in replay mode with a stub answer for prompts asked after the cutoff (a missing earlier prompt is a
  FAIL); a test that could not run counts as INCONCLUSIVE. For several LLM variants only the first is perturbed.
- `s2.same_bar_execution@1` runs the `pnl_reconstruction` logic on every trial. The `execution` block, when declared, supplies
  the lag and price column(s); otherwise each strictly positive numeric column of the dataset (at most 12) is tried with a
  lag of one bar. FAIL when the reported returns fit an earlier fill (the same bar) or a units mismatch, unless the reported
  positions never depend on their own bar's data (the "position held during the bar" convention), which is INCONCLUSIVE; BLOCKED when the run
  reported no positions or the position columns cannot be matched to price columns; INCONCLUSIVE when no column explains the
  returns (never PASS). Where the hypothesis pins `s2.pnl_reconstruction`, or declares a `reconstruction` block (`pnl.linear`, whose positions may be
  contracts or shares that a price-ratio rebuild cannot match), the default check is skipped, the receipt says why, and that
  check decides.
- The lint gains `LA006` (`np.roll` with a negative shift, BLOCK) and `LA007` (a shift or roll amount that is not a literal
  number, WARN). A name bound once to a number is resolved, so `NEG = -1; x.shift(NEG)` is `LA001`. The lint stays an early
  warning; the two runtime checks are the gate.

**Evidence** (`qrh bench run --suite planted-bugs --quick`, seed 20240601, 3 replications per variant): same-bar execution
detection went from 0/27 to 27/27 in the default arm, percent-unit returns from 12/18 to 18/18, clean controls stay at 0/21
false alarms, and the two new rarely active look-ahead variants (`np.roll` and a named negative shift, 6 runs) are caught by the
perturbation test 6/6 where the truncation test alone caught 1/6. These are defects the checks were designed for, so the
rates show the checks work, not real-world power.

## One-combination hypotheses at S5

PBO measures how often the in-sample winner of several variants falls in the lower half out of sample. With exactly one
combination (a pre-registered single rule) there is no selection, so S5 does not run `s5.pbo`. It records PBO as "not applicable
(one combination, no selection among variants)" in the receipt (`params.pbo`), the reasons and the report, draws no lambda
figure, and decides on the deflated Sharpe ratio with the family trial count actually charged. The cross-trial variance of Sharpe
ratios that the DSR needs does not exist for one column, so V is the sampling variance of the out-of-sample Sharpe ratio,
`(1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1)` (recorded as `var_sharpe_source: estimator`). N = 1 gives SR0 = 0; a family that has
already been searched raises N and the bar. Two or more combinations: see the next section.

## The Sharpe variance floor in the deflated Sharpe

The luck bar SR0 grows with V, the variance of the Sharpe ratios of the null trials. An independent review showed that V can be
shrunk by adding near-copies of the strategy: one combination had DSR 0.83 (FAIL), the same strategy plus a twin (the same rule
with 0.2 basis points more cost) had DSR 0.994 (PASS), because the twin made the cross-variant variance almost zero.

**Rule, for every grid size:** `V = max(grid, estimator)`, where `grid` is the variance across the combinations of the in-sample
per-period Sharpe ratios (the old rule; not defined for one combination) and `estimator` is the sampling variance of the selected
variant's out-of-sample Sharpe estimator, `(1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1)`. The S5 receipt records both numbers
(`params.var_sharpe: {grid, estimator, used}`, and the metrics `var_sharpe_grid`, `var_sharpe_estimator`, `var_sharpe`), the
`dsr_inputs` artifact and the S5 report page show both and which was used, and a note in the S5 reasons says so.

**This intentionally makes some grids stricter.** A grid whose cross-variant variance is below the sampling variance of its own
out-of-sample Sharpe (typically a grid of near-identical variants, or a short out-of-sample series) now gets the higher luck bar it
should have had. A grid whose cross-variant variance already exceeds the sampling variance is unchanged to the last digit. N is
unchanged: the floor changes only V.

## Parameter stability at S5: `s5.param_stability@1`

A strategy that works at one parameter setting and nowhere near it is usually an overfit. S5 asks two questions of the
out-of-sample returns and reports a third. The plateau idea is Robert Carver's (*Systematic Trading*, Harriman House, 2015); the
neighbour check and every number below are this harness's own, written fresh and not taken from any code.

"Out-of-sample" here means the bars the walk-forward (S4) scored. A grid point's returns on those bars come from its
development-window run (S3); a generated neighbour is run once on the development window and scored on the same bars. A parameter
is fixed in each run, so nothing is fitted on the scored bars.

**The neighbourhood is local.** Around the selected setting, ONE parameter at a time is moved to the value at relative distance
`s` up and down, for each `s` in `stability_steps` (default 10% and 20%; diagonals, two parameters at once, are not used: they
multiply the runs and add little to a cross neighbourhood). Where the grid has a point within a quarter of a step of that value it
is used; every other neighbour is generated and run through the adapter. This makes the result independent of how the grid is
spaced (a wide log-spaced grid is judged by what happens 10% and 20% from the chosen value, not by far-away grid points). Rules:

- **No hairline neighbours.** Every neighbour sits at least `min(stability_steps)` away (relative to the selected value; absolute
  for 0, which cannot be nudged and is listed as unchecked). Closer grid points are ignored; a closer declared neighbour value is
  refused (BLOCKED); a generated value that bounds clip to less than that distance is listed as unchecked. Integer values round
  away from the selected value, never toward it.
- **Numbers written as text count as numbers** (`"20"` is 20; generated neighbours are written back as text). A non-numeric
  parameter that takes two or more values in the grid is varied across those values, one parameter at a time.

| Measure | Formula | Decides |
|---|---|---|
| (a) Neighbour check, per parameter | `ratio_p = mean(OOS Sharpe of the neighbours of parameter p) / OOS Sharpe of the selected`; the worst parameter decides: FAIL when `min_p ratio_p < stability_neighbour_ratio`. The pooled mean over all neighbours is reported as information only (a parameter that changes nothing would otherwise dilute a spike in another) | a spike |
| (b) Local plateau share | `share = #{nearby numeric settings with OOS Sharpe >= stability_closeness x the SELECTED setting's OOS Sharpe} / #{nearby numeric settings}`; FAIL when `share < stability_plateau_share` | a plateau |
| (c) Diagnostics | per numeric parameter: the mean absolute OOS Sharpe gap between adjacent grid values (others fixed) and the mean correlation of their OOS return paths; the parameter with the largest mean gap is named the most fragile. The old global share (grid points within `stability_closeness` of the grid's best) is kept as `plateau.global_info` | nothing (report only) |

Other details:

- **Which combination is "the selected one".** S4 selects per fold. The check uses the combination chosen in the most folds
  (ties: the lowest label, like S4's own tie rule). The receipt lists the picks.
- **The selected Sharpe at or below zero:** NOT APPLICABLE, reported (S4 and S5 already judge the level). A flat neighbour
  (undefined Sharpe) counts as zero and is counted (`undefined_as_zero`).
- **Neighbour runs are not trials.** They check a fixed choice and never select among variants, so they change neither the trial
  nor the family count and add nothing to N in the deflated Sharpe. The receipt says so (`trial_count_note`). They are run for a
  grid too, whenever a step-distance point is not in the grid.
- **Declared neighbours.** For one pre-registered combination the hypothesis may declare neighbour values and bounds:

  ```yaml
  stability:
    neighbours: {lookback: [15, 25]}      # used instead of the generated values for this parameter; a failing one BLOCKS
    bounds: {lookback: [5, 60]}           # generated neighbours are clipped to [low, high]
  ```

  Optional, hashed only when present; the readiness report lists them. A grid uses its own points and generated ones.
- **Outcomes.** NOT APPLICABLE only when the adapter declares no parameters or the selected out-of-sample Sharpe is not above
  zero. **BLOCKED** when the adapter takes parameters but none is numeric or takes more than one value (nothing to nudge: the check
  will not pass it silently; the trader must decide how to treat the strategy), when no neighbour could be produced and run, when
  a declared neighbour is refused or fails, when the thresholds are invalid, or when generated neighbours are needed and no adapter
  is available. A neighbour the strategy cannot run (for example a window of 0) is listed under `unchecked` and in the readiness
  report, never hidden.
- **Thresholds are the trader's.** `stability_closeness` (0.70), `stability_plateau_share` (0.50), `stability_neighbour_ratio`
  (0.50) and `stability_steps` ([0.10, 0.20]) are frozen with the hypothesis, always asked of the trader (see
  [interviews.md](interviews.md)), and a spec may not set them against the trader's answer. A hypothesis registered before they
  existed has none: the defaults apply and the receipt says so (`thresholds_source`).

What the measures cannot do: a strategy whose good region is narrower than the smallest step is called a spike, by design. A tiny
neighbour step on a very flat strategy will pass anything flat. Plateaus are judged near the chosen setting only; a strategy whose
good region is wide but sits far from the chosen setting is not rewarded.

## Rules for strategies that use an LLM: `s2.llm_gate@1` and `llm.replay_cache@1`

A language model trained on data up to a date D already knows what happened before D; the leak lives in the weights, and no
truncation test sees it. Prompts and models are also trials, and a model's answers vary between runs. Part B of P3d-b2 adds
two methods for strategies whose hypothesis declares an `llm` section. Both are status `default` **without the promotion
flow**: they are new checks with no predecessor to beat, and for a strategy without `llm` they change nothing (the receipts
record no new method; only the declaration lint, LLM001, looks at every adapter).

- `s2.llm_gate@1`: the training-cutoff gate (every window starts strictly after the latest cutoff plus `llm_cutoff_margin_days`,
  default 180; unknown cutoff or source BLOCKS), the LLM001 declaration lint, and the prompt-vs-truncation detector (a replay-cache
  miss in a truncated run is a FAIL).
- `llm.replay_cache@1`: `qrh.llm.complete`, the cache key (sha256 of `{model_id, params, prompt}`), the record and replay modes
  and the cache hash recorded in the S2 to S6 receipts.

No academic reference is cited for either method: none was checked against a publisher record (the `verified` flag stays
false and the reference list is empty). Both appear in the `methods` of every receipt of a strategy that uses an LLM. The
rules, the spec field, the replay cache, the examples (08 and 09) and the limits are in [llm.md](llm.md). Evidence: the
`llm-gate` benchmark suite (`qrh bench run --suite llm-gate`), receipt, text and PDF in `examples/bench/llm_gate/`: every
planted variant of the four classes is detected and the clean controls have no false alarm; these are the defects the rules
are designed for, so the rates are a check that the rules work, not an estimate of real-world power. A documented cutoff can be
wrong and models may know events after their stated cutoff: the readiness report lists both as known limits of every LLM trial.

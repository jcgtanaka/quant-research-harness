# Architecture

The harness separates judgment from decision. Agents propose and orchestrate;
deterministic code checks and decides. Anything that cannot be checked is
`BLOCKED` (fail closed).

## Layers

- **Agent**: an AI assistant that drives the workflow (creates experiments,
  runs stages, reads receipts). It never issues a pass/fail verdict itself.
- **Hooks**: guards that run around agent actions (`qrh hook <name>`, see `docs/hooks.md`) and
  before publication (secret scanning and a private denylist check).
- **Pipeline**: stages S0-S7, run in order. Each stage has an explicit gate and
  writes a JSON receipt describing inputs, checks and outcome.
- **Checks**: pure, deterministic functions (no look-ahead, validation,
  implementation sanity). A check returns PASS, FAIL or BLOCKED.
- **Memory**: a registry of past runs, receipts and decisions, so that repeated
  searches over the same data are counted and not forgotten.
- **Adapters**: small interfaces (`DatasetRef`, `DataView`, `StrategyAdapter`,
  `RunResult`) that isolate any backtesting engine or data source. `DataView`
  enforces point-in-time access via `as_of(decision_time)`.
- **Methods and benchmarks**: a registry of versioned, cited implementations; every stage obtains its method from it
  and records the version in its receipt. Benchmarks with known truth decide whether a new version may become the
  default (see `docs/methods.md`).
- **Reports**: human-readable summaries (markdown, and PDF pages plus chart-data JSON)
  generated from receipts and the artifacts they record, never from recomputation.

## Receipts and registry

Every stage writes a receipt: `<QRH_HOME>/receipts/<trial_id>/<stage>.json`
(`QRH_HOME` defaults to `~/.qrh`). A receipt records the hypothesis hash, code
hash, data hashes, parameters, named windows, family trial count, metrics,
verdict (PASS, FAIL, BLOCKED, INCONCLUSIVE), reasons, environment, an optional
attributed override, and the hash of the previous stage's receipt. Its
`receipt_hash` is the canonical-JSON sha256 of everything except `created_at`,
so identical inputs give identical hashes; loading re-verifies the hash and a
tampered file is an error.

The registry and the data catalog share one SQLite file in `QRH_HOME`
(WAL, foreign keys, versioned migrations). It holds hypotheses (thresholds are
frozen at registration; a change needs a new trial), every evaluated parameter
combination (for per-trial and per-family counts), an index of receipts, an
overrides log (a reason is mandatory) and the dataset catalog (profile, content
hash, status; unknown datasets are BLOCKED, a changed hash resets to UNCHECKED).

The pipeline runs S0-S6 in order and requires a verified PASS receipt of stage
N-1 before stage N. An exception inside a stage becomes a BLOCKED receipt and
stops the run; S7 then always writes a summary naming the failing stage. The
overall verdict is PASS only if S0-S6 all PASS. S7 also writes a markdown report
to `<QRH_HOME>/reports/<trial_id>.md` (hypothesis, thresholds, each stage verdict,
reasons and key metrics with its receipt hash and path); the path is recorded in
the S7 receipt. Failures are reported too.

## Data gate and implementation checks

Adapter code is hashed from the bytes that run. The adapter loader compiles the adapter
and every sibling module it imports from the hypothesis directory straight from the source
bytes it just read (a source-only loader, bytecode writing off), so a stale cached `.pyc`
can never execute code other than the code whose hash is recorded (`adapter_code_hash` in
the S2 and S3 receipts). Modules loaded for one hypothesis are evicted before the next load,
so same-named helpers in two directories do not leak into each other.

A hypothesis may bind itself to a strategy: `adapter` (`module.path:ClassName`,
resolved relative to the spec file's directory first), `datasets` (names in the
catalog; an entry may be marked `slow: true` and then must declare a
`publication_column`), `windows` and `bounds` (sanity limits for returns, gross
leverage and position weight). They are recorded in the S0 receipt and are
immutable for the trial.

**S1** requires each dataset to be catalogued, unchanged on disk (content hash
equals the catalog hash) and to pass its declared quality rules (schema, range,
relationship, null-rate drift, gaps and duplicates, timezone; windows are
fractions or row counts, never fixed calendar units). A dataset without rules
stays `UNCHECKED` and S1 treats that as `BLOCKED`.

**S2** runs five independent checks and combines them (BLOCKED over FAIL over
PASS):

- The static look-ahead lint is an early warning. It flags patterns such as
  negative shifts, centered windows and backward fills. It cannot see
  everything, and a finding can be suppressed with a reasoned comment that is
  recorded in the receipt.
- The future-truncation test is the proof. The adapter runs on all the data
  and again on data cut before a cutoff bar; everything it decided strictly
  before the cutoff must match exactly. It only covers the cutoffs tested
  (default 25, 50 and 75 percent of the development window), so a PASS is
  evidence, not a guarantee. It also needs the strategy to be deterministic. The
  result is cached per adapter code, data and parameters.
- The future-perturbation test is the second proof and catches look-ahead that
  is rarely active (`np.roll(x, -1)`, `x.shift(NEG)`, a read that ignores `end`),
  which the three truncation cutoffs can miss. The cutoffs are every bar where the position
  changes and the bar before it (up to `perturbation_change_cap`, default 40
  change bars, spread evenly beyond that; the receipt records the coverage and
  readiness warns when it is below 100%), plus `perturbation_cutoffs` bars
  spread over the window (default 20; both are frozen thresholds). At each the
  adapter is rerun, freshly loaded, with every row after the cutoff (and every
  row not yet published at it, with a publication column) replaced: the numeric columns of the
  future rows are shuffled as whole rows (boolean, text and categorical columns
  follow the same permutation), each row is rescaled by a positive
  factor, the later rows share a random level shift, and the first future bar's
  return changes sign. Positions, trades and returns up to and including the
  cutoff must be identical (tolerance 1e-12), otherwise FAIL with the first
  divergent timestamp. A statistic computed from the past only (an expanding
  mean, a rolling window ending at the decision bar) cannot change, so legitimate
  strategies pass. If the unperturbed rerun already differs from itself (an
  unseeded random source) the result is BLOCKED, not FAIL. Cost: one extra adapter
  run per cutoff (at most 2 x 40 + 20 by default). If no perturbed run differs from
  the original anywhere (a cached result, or a strategy that ignores the data) the
  result is BLOCKED. With a publication column the returns are not compared (the
  realized prices of late-published rows move too); positions and trades are. A
  strategy that declares an `llm` and makes model calls is run in replay mode: a
  prompt asked after the cutoff gets a deterministic stub answer, a prompt at or
  before it that is not in the cache is a FAIL; if the full run made no model call
  the test runs as for any other strategy. A test that could not run counts as
  INCONCLUSIVE, never dropped. Cached per adapter code, data and parameters.
- The same-bar execution check is on for every trial. It rebuilds the returns
  from the reported positions and a price column with the fill one bar after
  the decision and FAILs when the reported returns fit an earlier fill (a
  strategy that decides and fills on the same close). The price column is the
  one in `execution` when the hypothesis declares it, otherwise every strictly
  positive numeric column of the dataset is tried (at most 12) with a lag of one
  bar. A run with no reported positions is BLOCKED; a run whose returns no column
  explains (costs above the default bound, leverage) is INCONCLUSIVE, never
  PASS, and the reason says to declare `execution`. Before an inferred FAIL the
  check perturbs the data from each tested bar onward and asks whether the
  reported position of that bar changes; if positions never depend on their own
  bar they look like "the position held during the bar" and the result is
  INCONCLUSIVE (declare `execution` or report the position held after the bar). It is skipped, and the
  receipt says so, where the hypothesis pins `s2.pnl_reconstruction` (the same
  rebuild with the declared execution) or declares a `reconstruction` block
  (`pnl.linear`, which explains profit from lagged positions in contracts or shares).
- The sanity bounds catch implausible returns (including likely percent units),
  leverage and position size.

S2 uses the first point of the parameter space (sorted keys, first value of
each) unless the spec gives `s2_params`.

## Validation statistics

Thresholds not set in the hypothesis are filled at S0 from
`qrh.validation.defaults` (max_pbo 0.2, min_dsr 0.95, min_oos_sharpe 0,
min_sharpe_lower_ci 0, min_cost_stressed_sharpe 0, bootstrap_reps 2000,
ci_level 0.95, cost_stress_multiplier 2, cscv_blocks 16, llm_cutoff_margin_days 180,
perturbation_cutoffs 20, stability_closeness 0.70, stability_plateau_share 0.50,
stability_neighbour_ratio 0.50, stability_steps [0.10, 0.20]) and stored in the
hypothesis, so the S0 receipt shows every threshold in force.

`periods_per_year` comes from the hypothesis if given, else from the observed
density of the development window's timestamps, `(n - 1) / (span / 365.25 days)`;
if neither is possible the stage is BLOCKED. Because it counts bars actually
present, weekday-only daily data gives about 261 and a 5-minute 6.5-hour session
gives about 78 x 261. It is not a trading-calendar model: holes in the data lower
the rate, so a window with large gaps may deserve an explicit `periods_per_year`.
All randomness uses seeded numpy generators; seeds are recorded in the receipt.

- **S3** evaluates the full cartesian grid of `adapter.parameter_space()` (sorted
  keys; its size must equal `parameter_space_size`), logs every combination in
  the registry and stores the T x N returns matrix, hash in the receipt.
- **S4 walk-forward** (`walk_forward: {train, test, step, anchored}`; bars or
  fractions of the window): per fold the best combination by in-sample
  annualized Sharpe on the train span only (ties: lowest combination hash) is
  run on the test span; test returns are stitched. Gate: OOS Sharpe >=
  min_oos_sharpe.
- **S5 PBO** (Bailey, Borwein, Lopez de Prado, Zhu 2017), CSCV: cut the matrix
  into S even contiguous blocks; for each of the C(S, S/2) choices of S/2
  in-sample blocks, n* = argmax in-sample Sharpe, omega = OOS rank of n* / (N+1),
  lambda = ln(omega / (1 - omega)); PBO = share of combinations with lambda <= 0.
  PBO needs at least two combinations. With exactly one (a pre-registered single
  rule) nothing was selected, so S5 reports PBO as "not applicable (one
  combination, no selection among variants)" in the receipt (`params.pbo`), the
  reasons and the report, draws no lambda figure, and decides on the DSR alone.
- **S5 DSR** (Bailey and Lopez de Prado 2014), per-period Sharpe SR, skewness g3,
  kurtosis g4 (normal = 3):
  PSR(SR*) = Phi((SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2));
  SR0 = sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e)));
  DSR = PSR(SR0). V is the variance of the in-sample per-period Sharpe ratios
  across the grid, floored at the sampling variance of the selected variant's
  out-of-sample Sharpe ratio, (1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1):
  V = max(grid, estimator) for every grid size, so near-copies of a variant cannot
  shrink the luck bar; both values and the one used are recorded
  (`params.var_sharpe`). With one combination there is no grid, so V is the
  estimator and `var_sharpe_source: estimator`; N = 1 gives SR0 = 0 and the DSR is
  the PSR against zero), N the family trial count (at least the grid size; counts
  every (trial, combination) pair logged for the family, so repeating a search
  raises N). Gate: PBO <= max_pbo (when applicable) and DSR >= min_dsr. DSR is conservative: for
  the best of N null strategies it exceeds 0.95 well under 5% of the time.
- **S5 parameter stability** (`s5.param_stability@1`, see [methods.md](methods.md)): around the
  selected combination, one parameter at a time is nudged by each of `stability_steps` up and down
  (grid points used where they are at about that distance, else non-trial runs). The worst
  parameter must keep `stability_neighbour_ratio` of the selected out-of-sample Sharpe, and
  `stability_plateau_share` of the nearby numeric settings must keep `stability_closeness` of it.
  Gate: FAIL on a spike or a missing local plateau; NOT APPLICABLE only with no parameters or a
  non-positive selected Sharpe; BLOCKED when nothing can be nudged. Never a silent PASS.
- **S6** bootstraps the OOS returns with the stationary bootstrap (Politis and
  Romano 1994; geometric blocks, mean length ceil(T^(1/3)), circular) for
  percentile intervals of Sharpe, CAGR and max drawdown, and re-runs the
  walk-forward with `cost_multiplier = cost_stress_multiplier`. An adapter
  whose `run()` has no `cost_multiplier` parameter is BLOCKED. Gate: Sharpe
  lower bound >= min_sharpe_lower_ci and stressed Sharpe >= min_cost_stressed_sharpe.

## Stage artifacts

Stages persist what their figures need under `<QRH_HOME>/trials/<trial_id>/<stage>/` and
record each artifact as `params["artifacts"][name] = {path, hash}` in the stage receipt (path
relative to `QRH_HOME`, sha256 of the file bytes). Reading re-verifies the hash. Artifacts are
JSON (canonical, no NaN) or parquet:

| Stage | Artifacts |
|---|---|
| S0 | `hypothesis` (the registered hypothesis, thresholds in force) |
| S1 | `coverage` (per dataset: start, end, rows, detected gaps, each rule result with up to 50 offending timestamps) |
| S2 | `lint` (findings), `truncation` (full vs truncated positions per cutoff, first divergence), `perturbation` (cutoffs, kind, status, first divergence), `same_bar_execution` (verdict, reconstruction figures, price column used), `sanity` (observed vs bound) |
| S3 | `is_returns` (T x N matrix), `combos` (per-combination metrics) |
| S4 | `oos_returns`, `oos_equity`, `folds` (spans, selected parameters, IS and OOS Sharpe) |
| S5 | `pbo_logits` (lambda per CSCV split; absent for one combination), `dsr_inputs` (SR, SR0, N, V used, V grid, V estimator, skewness, kurtosis, PBO or null, DSR), `param_stability` (selected combination and picks, neighbours with their out-of-sample Sharpe, the two measures, per-parameter diagnostics, unchecked and skipped parameters, thresholds and where they came from) |
| S6 | `bootstrap_samples` (Sharpe, CAGR, max drawdown per replication), `bootstrap_summary` (percentiles, intervals), `equity_compare` (base vs cost-stressed OOS equity) |

## Visual reports

Every completed stage leaves visual evidence: a PDF and a chart-data JSON. **One-source rule:
both are produced only from the stage receipt and the artifacts it records, verified by hash;
renderers never recompute a statistic.** The dictionaries in `qrh.reports.chartdata` (chart
type, series with x/y, axis labels and units, receipt hash) are the single description of each
figure; `qrh.reports.charts` draws exactly those dictionaries, and a future dashboard can render
the same JSON interactively. Binning a histogram, ranking combinations and thinning a long
series are presentation prep, not statistics, and happen in the chart data.

Pages per stage: S0 text card (hypothesis, family, asset class, frequency, every threshold);
S1 coverage timeline per dataset with gaps marked and a rule-results table; S2 lint findings,
one full-vs-truncated positions chart per cutoff (first divergence marked) and the sanity
table; S3 a Sharpe heatmap when there are exactly two parameters (otherwise the top 30 combinations
ranked, with the count of the rest) and a Sharpe histogram; S4 fold timeline, stitched OOS
equity and selected parameters per fold; S5 the lambda histogram with PBO at zero and the
observed SR against SR0 with DSR labeled; S6 bootstrap histograms with interval bounds and
base vs cost-stressed equity; S7 the verdict summary. Every page carries a footer with trial id,
stage, the first 12 characters of the receipt hash (the full hash is in the PDF metadata) and a
research-use-only disclaimer. A stage that recorded no artifacts (for example one that was
BLOCKED early) gets a status page instead; a stage whose artifacts do not verify gets nothing.

Design rules (`qrh.reports.theme`): matplotlib Agg, DejaVu Sans, surface `#fcfcfb`, dark-gray
text, light thin grid, 1.5 pt lines; categorical series use eight fixed slots in order, never
cycled, at most three per chart, two series use slots one and two; the single-hue blue ramp is
for magnitude only; status colors (good, warning, serious, critical) mark verdicts only and are
always paired with an icon and the verdict word; legends whenever there are two or more series
and direct labels at line ends; no dual y-axes.

Outputs: `<QRH_HOME>/reports/<trial_id>/<stage>.pdf` and `<stage>.chart.json` after each stage,
plus the combined `<trial_id>.pdf` after S7 and `manifest.json` (every file with its sha256 and
the receipt hash it was built from). Output is deterministic: identical artifacts give
byte-identical PDFs and JSON (no timestamps, fixed fonts and metadata). A rendering failure never
changes a verdict; it is recorded in the manifest and shown by `qrh status`. `qrh report
<trial_id> [--stage S4] [--open]` re-renders from the persisted artifacts and exits 2 (BLOCKED) if
a receipt or artifact is missing or does not match its hash; a failed stage leaves no PDF behind.
`QRH_REPORTS=0` switches the automatic rendering off (for bulk or headless runs).

## Methods and benchmarks

`qrh.methods` holds a `Method` record per implementation: id (`s5.pbo`), version (`cscv@1`), stage, kind, callable
entry, parameters, status (`experimental`, `default`, `deprecated`), references (citation, DOI or arXiv id or URL,
`verified`) and the harness version that added it. Public defaults ship in `qrh.methods.builtin`; a local promotion lives in
`<QRH_HOME>/methods/state.json` and never changes the code. Stages do not import implementations: they ask the context
(`ctx.method("s5.pbo")`), which resolves a pin recorded at S0, else the current default, and records what it handed out.

- S0 fills every method id with the pinned version or the default of that day and stores the map with the hypothesis
  (part of its hash only when present, so older hypotheses keep their hash). A later promotion does not change a
  registered trial. Pinning an unknown or deprecated version is refused at registration.
- Every stage receipt has `methods: {id: version}` (part of the receipt hash only when non-empty, so older receipts still
  verify). A version that is not the current default is flagged in `params["non_default_methods"]`, in the S7 receipt
  and in the markdown report.
- An id that ships with no default (`s2.pnl_reconstruction`) is opt-in: only a hypothesis that pins it uses it, and S2
  then reads the frozen `execution` spec field (`lag_bars >= 1` required, otherwise BLOCKED).
- A state file that cannot be read, or an id with no or several defaults, makes the stage BLOCKED.

`qrh.bench` runs deterministic, seeded suites with known truth through the same method entries: planted bugs (S1 and S2
detectors over a corpus with clean controls), noise false positives and known-edge power (S3-S5 on synthetic strategy
families), and an optional read-only `history` suite over stored trials. A bench receipt (`<QRH_HOME>/bench/<id>/`)
records method versions, suite configuration, seeds, results and hashes; its text, PDF and chart JSON follow the one-source
rule of the visual reports (the PDF is drawn from the chart JSON only). Promotion, demotion and deprecation are
`qrh methods` commands that check the receipt against rules in code (`qrh.methods.promotion`) and then wait for a human
`qrh approve`.

## Packs and interviews

`qrh.packs` loads experience packs: folders registered by path whose content hash is recorded in every receipt written
while they are active. Public packs layer before private ones; a pack may only tighten core thresholds unless a human
approves the loosening (`qrh pack add` waits for `qrh approve`). Lessons, imported attempts and account facts reach S0;
attempts raise the trial family count at S5; lint rules run at S2. See [packs.md](packs.md).

`qrh.interview` asks only what is missing: question banks come from packs (plus ad-hoc questions), `qrh interview next`
issues one question at a time and records when, and `qrh interview answer` stores a versioned answer only if it traces
to the user's own words in the real session transcript (or a person typed it at a terminal). Answers map to dataset
risk flags, extra trial counts or named parameters. A critical question pending or answered "don't know", or a risky
answer, turns a passing S7 into PASS · CONDITIONAL with the conditions listed. See [interviews.md](interviews.md).

## Dataset onboarding

`qrh.onboarding` makes the understanding of a dataset explicit and checked: `profile` builds a deterministic, privacy-safe
profile (aggregates and shapes, never rows), `model` holds the strict pydantic instrument model and the untrusted proposal
schema, `verify` runs the deterministic checks (a claim is verified, contradicted or unverified with an evidence string),
`store` keeps profiles per content hash and versioned models in the catalog tables of migration 10, `questions` turns
unverified fields into interview questions (scope `dataset:<name>`, ids `onb-<field>`, identity = content hash plus the
claim), `workflow` is the propose, verify and answer logic, and `card` builds the S1 artifact. A contradiction sets the
dataset BLOCKED through `check_dataset_report`, so S1 refuses it; the onboarding state is part of the S1 receipt, and the
card and onboarding table are two extra S1 pages per dataset. See [data-onboarding.md](data-onboarding.md).

## Language-model strategies

A hypothesis may declare an `llm` section (models with training cutoffs, prompt files, params, retrieval, variants); see
[llm.md](llm.md). It is validated and stored at S0 (hashed only when present), and drives three rules. S2 runs the
training-cutoff gate (`s2.llm_gate@1`) and the LLM001 declaration lint, and does not run the adapter when either fails; the
S2 artifact `llm_gate` feeds a PDF page. The loader hashes prompt files into the adapter code hash, and S3 multiplies the
adapter grid by the declared variants (`params["llm_variant"]`), logging each combination with its variant name and prompt
hash. The pipeline opens an `LLMSession` (a context variable: mode `replay` or `record`, the trial's cache folder, the declared
providers) around every stage; adapters call `qrh.llm.complete`, which serves recorded answers from
`<QRH_HOME>/llm_cache/<trial_id>/` (append-only JSONL with entry hashes). In S2 the full run is replayed first and each
truncated run is forced to replay, so a cache miss there is a FAIL ("prompt differs between full and truncated data"). Receipts
of S2 to S6 carry the hash of the cache entries a stage used (hashed) and the call counts (diagnostics, not hashed). `qrh.llm`
imports only the standard library, so hypothesis validation and hooks stay fast; `qrh.llm.variants` and the bench use pandas.

## Deployment readiness

`qrh.readiness` gathers every warning of a trial (interview state, data and lint findings, exemptions, overrides,
non-default methods, loosened pack gates, open conditions and a static table of known harness limits) from verified
receipts, artifacts and the registry (`collect`), hashes the result without timestamps or file locations (`data`), draws
it in the report theme (`report`: `readiness.json`, `readiness.chart.json`, `readiness.pdf`, generated at S7 and by
`qrh readiness report`) and checks a human's signed acknowledgement with the public key only (`ack`, domain
`readiness-ack`). `qrh.human.acknowledge_readiness` is the terminal-only signing step, `approval-guard` keeps an agent
from running it, and `pre-restart-guard` can require it (`guard.require_readiness`). See [readiness.md](readiness.md).

## Approvals and integrity

`qrh.signing` holds the Ed25519 key pair (private key encrypted under a passphrase, public key for every check),
`qrh.approvals` the pending and signed requests, `qrh.human` and `qrh.tty` the terminal-only human steps, and
`qrh.integrity` the signed baseline of config and hook registrations. The hook runner turns a block into one allow
only for a valid signed approval of the identical action in the same session, and refuses to run enforcement hooks
on a config that changed since it was signed. See [hooks.md](hooks.md) and [security.md](security.md).

## Public / private split

This repository is public: generic code, synthetic data only. Keep private
adapters, configuration and the denylist in a separate private repository
that depends on this package. Private state lives outside both repositories,
in `~/.qrh/`. The publication guard (gitleaks plus `scripts/denylist_check.py`)
prevents private material from entering this repository.

## AI usage metrics

`src/qrh/usage/` is a separate read-only pipeline: `parse` streams one transcript (message ids counted once, no text
retained), `harvest` builds per-session records and merges them into `<QRH_HOME>/usage/sessions.json`, `attribution` joins
them with the hook logs, `quality` evaluates the plays, `verify` checks totals against the harness cost-state, and
`report`/`chartdata`/`charts` render text, JSON, chart JSON and a PDF in the shared theme. See `docs/usage.md`.

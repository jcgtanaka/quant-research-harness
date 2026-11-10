# quant-research-harness

A deterministic research harness for backtesting. AI agents orchestrate the
work; deterministic code decides. Research moves through stages S0-S7, each
with pass/fail gates and JSON receipts. Any check that cannot run returns
`BLOCKED`, never `PASS`. Engines plug in through small adapter interfaces, and
nothing is keyed on a symbol, a strategy name or a fixed bar size.

> **Alpha release.** The core checks work, but end-to-end tests found issues that are not fixed yet. Read
> [docs/known-limitations.md](docs/known-limitations.md) before you trust a result.

**Status: P0-P2 (P2a data gate and implementation checks, P2b validation stages, P2c visual reports),
P3 (Claude Code hooks), P3b (AI usage metrics), P3b+ (human approvals, runaway guard, claim check, waste and regression reports)
P3d-a (versioned method registry and benchmarks with known truth), P3d-b1 (experience packs, interviews,
PASS · CONDITIONAL, signed human approvals), P3d-b2 part A (deployment readiness report with a signed acknowledgement) and
P3d-b2 part B (rules for strategies that use an LLM: cutoff gate, prompts and models as trials, replay cache) and
P3e-a1 (dataset onboarding: profile, instrument model, propose, verify, confirm) done. S0-S7 are real: a hypothesis can pass or fail the full pipeline,
every run ends with a report, and thirteen hooks guard an agent session (look-ahead lint, data gate,
pre-restart guard, receipt-required, claim-check, runaway guard, approval-guard, and three informational ones).**

## Quick start

Before letting an agent work under the hooks, set up approvals once in your own terminal: `qrh approve --init`, then
`qrh config sign --settings ~/.claude/settings.json` (see [docs/security.md](docs/security.md)).

```bash
uv sync
uv run pytest
uv run qrh version
uv run qrh synth --seed 7 --periods 500 --out synthetic.csv
export QRH_HOME=$(mktemp -d)            # private state; defaults to ~/.qrh
uv run qrh new examples/hypothesis.example.yaml   # prints a trial id
uv run qrh run <trial_id>               # exits 0 only if S0-S6 all PASS
uv run qrh status <trial_id>
uv run qrh data register synthetic.csv
```

## Demo: the full pipeline (P2)

Each `examples/NN_name/` has an adapter, a hypothesis spec, data-quality rules and a
script that writes synthetic data. Example 01 is a clean trend strategy on synthetic data with
genuine persistent trends: it passes S0-S7 with the default thresholds. Example 04 searches a
120-combination grid on pure noise: it passes S1-S3 and is rejected at S5 (probability of
backtest overfitting too high, deflated Sharpe too low).

```bash
export QRH_HOME=$(mktemp -d)
for ex in 01_clean_strategy 04_overfit_grid; do
  uv run python examples/$ex/make_data.py $QRH_HOME/$ex      # synthetic prices.csv
  uv run qrh data register $QRH_HOME/$ex/prices.csv --name prices --rules examples/$ex/rules.yaml
  uv run qrh run $(uv run qrh new examples/$ex/hypothesis.yaml)   # 01: exit 0 (PASS); 04: exit 2 (FAIL at S5)
done
ls $QRH_HOME/reports                                          # one markdown report per trial, plus a directory per trial
```

## Visual reports (P2c)

After every stage the pipeline renders that stage's PDF and a chart-data JSON, built only from
the stage receipt and the artifacts it recorded (never recomputed). They land in
`$QRH_HOME/reports/<trial_id>/`:

```
S0.pdf ... S7.pdf       one PDF per stage that ran (S7 is the summary)
S0.chart.json ...       the same figures as JSON, for a future interactive dashboard
<trial_id>.pdf          every stage's pages combined, written after S7
manifest.json           each file with its sha256 and the receipt hash it was built from
```

```bash
uv run qrh report <trial_id>                 # re-render everything from persisted artifacts
uv run qrh report <trial_id> --stage S4 --open
```

`qrh report` exits 2 if a receipt or artifact is missing or does not match its hash. A rendering
failure never changes a stage verdict: it is recorded in the manifest and shown by `qrh status`.
Set `QRH_REPORTS=0` to skip automatic rendering. Identical artifacts give byte-identical files.

The other examples show the earlier gates: 02 peeks one bar ahead (S2 FAIL: lint BLOCK and
truncation and perturbation FAIL), 03 has returns in percent (S2 FAIL via sanity bounds), 05 has a corrupted
column (S1 BLOCKED). `uv run qrh lint examples/02_lookahead/adapter.py` runs the static
look-ahead lint (exit 2 on BLOCK). Examples 08 and 09 use a language model through a deterministic stub (see below). Numbers 06
and 07 stay reserved for later phases.

## Hooks (P3)

`qrh hook <name>` is the entry point Claude Code calls for thirteen hooks; the event JSON arrives on stdin,
exit 2 blocks and its stderr is shown to the agent. Enforcement hooks fail closed, informational ones never
break a session, and every invocation is logged (redacted) under `$QRH_HOME/logs/`.

```bash
uv run qrh hooks print-settings --settings /path/to/settings.json          # preview the merge
uv run qrh hooks print-settings --settings /path/to/settings.json --write  # writes, with a timestamped backup
uv run qrh search "momentum regime"                                        # BM25 keyword search of past hypotheses
uv run qrh test-receipt -- pytest -q                                       # records a test run for the pre-restart guard
```

A block can only be passed by a human running `qrh approve <id> --reason "..."` in their own terminal (single use, HMAC signed, 30 minutes);
the agent is stopped from approving itself. See [docs/hooks.md](docs/hooks.md) (table of hooks, `config.yaml`, approvals and threat model, runaway guard, claim check, logging, limits),
`hooks/claude_code/settings.example.json`, and the agent skill in `skills/qrh-workflow/SKILL.md`.
Publication stays guarded by `.pre-commit-config.yaml` (gitleaks and the private denylist check).

## AI usage metrics (P3b)

`qrh usage` harvests Claude Code transcripts into per-session metrics (tokens, cost, cache hit rate, tool mix, hook
blocks, usage per trial), reports them as text, JSON and PDF, and checks its own totals against the harness cost record.

```bash
uv run qrh usage init                 # example thresholds.json and prices.json: review and edit
uv run qrh usage harvest              # reads ~/.claude (or $CLAUDE_CONFIG_DIR) read-only
uv run qrh usage report --by week --pdf usage.pdf
uv run qrh usage verify              # transcript scope by default; --scope strict is a diagnostic
uv run qrh usage waste               # re-reads, large outputs, failing loops, cold cache, compactions
uv run qrh usage regress --by week   # before/after per-turn metrics, Mann-Whitney U and Fisher exact
uv run qrh audit tests --base main   # weakened tests in the diff
```

See [docs/usage.md](docs/usage.md). The approach follows IchiSieben/claude-kit (Yoichi Palacios Tanaka, Apache-2.0); no
code was copied (see [NOTICE](NOTICE)).

## Methods and benchmarks (P3d-a)

The harness improves from science and experience without modifying itself. Every check and statistic is a versioned,
cited method (`s5.pbo` `cscv@1`, `s4.walkforward` `rolling_anchored@1`, ...). Every receipt records the versions it used.
A new version starts experimental; only a measured win on benchmarks with known truth, plus a human approval, makes it the
default.

```bash
uv run qrh methods list --stage S5                       # id, version, status, references
uv run qrh methods show s5.pbo@cscv@1
uv run qrh bench run                                     # quick profile (about 10 s): planted bugs, noise false positives, known-edge power
uv run qrh bench run --full --method s4.walkforward=purged_embargo@1 --compare-to default   # a real comparison (minutes)
uv run qrh bench compare <bench_id_a> <bench_id_b>       # deltas and Fisher p-values
uv run qrh methods promote s4.walkforward@purged_embargo@1 --bench <bench_id> --reason "..."   # waits for `qrh approve`
```

A hypothesis may pin versions with `methods: {s5.pbo: cscv@1}`; the full map is recorded at S0 and frozen like the
thresholds. S2 also runs, for every trial, a future-perturbation test (rerun with the data after each cutoff replaced) and a
same-bar execution check (returns rebuilt from the reported positions). The experimental, opt-in `s2.pnl_reconstruction`
check does the same rebuild with the declared `execution: {lag_bars, price_column}` (examples 01-03 select it). See [docs/methods.md](docs/methods.md) (registry, versioning, benchmarks, promotion rules, how science and
experience enter) and the benchmark reports it writes (`$QRH_HOME/bench/<id>/`: receipt, text, PDF, chart JSON).

## Strategies that use an LLM (P3d-b2 part B)

A model trained on data up to date D already knows any backtest period before D, and no truncation test can see that. A
hypothesis whose strategy uses a model declares an `llm` section (models with training cutoffs, prompt files, params,
retrieval, variants) and three rules apply: an S2 **training-cutoff gate** (every window must start after the latest cutoff
plus `llm_cutoff_margin_days`, default 180; an unknown cutoff BLOCKS), **prompts, models and variants count as trials** (prompt
files are hashed into the code hash and multiply the S3 grid), and a **replay cache** (every call goes through
`qrh.llm.complete`; answers are recorded once and replayed, so a verdict is reproducible). An adapter that appears to use an
LLM without declaring it is a BLOCK-level lint finding (LLM001). qrh ships no network backend and no real keys.

```bash
uv run qrh run <trial> --llm-mode record     # record the model's answers (private adapter code registers the backend)
uv run qrh run <trial>                       # replay them (the default)
uv run qrh llm status <trial>; uv run qrh llm verify <trial>
uv run qrh bench run --suite llm-gate        # planted LLM bugs and clean controls
```

Example 09 (`examples/09_llm_clean`) passes S0-S7 with a stub that follows a trend rule from the prompt's past prices; example 08
(`examples/08_llm_contamination`) has a stub that knows the future before its cutoff and fails S2. See [docs/llm.md](docs/llm.md)
(rules, spec field, replay cache, limits stated honestly: cutoffs are provider claims, the cache is private).

## Dataset onboarding (P3e-a1)

An LLM may interpret a dataset; deterministic code verifies every claim; the human confirms what code cannot prove. The agent
never sees the rows of a private dataset, only a privacy-safe profile.

```bash
uv run qrh data register prices.csv --name prices --rules rules.yaml   # --privacy public for synthetic data
uv run qrh data profile prices --json                                  # aggregates and shapes, never rows
uv run qrh data propose prices --file proposal.json                    # unverified claims: asset class, units, adjustments, ...
uv run qrh data verify prices                                          # contradicted claims BLOCK the dataset
uv run qrh data onboarding prices                                      # state: NOT_ONBOARDED ... ONBOARDED, CONTRADICTED
uv run qrh interview next --stage S1 --scope dataset:prices            # the rest is asked of the human
```

See [docs/data-onboarding.md](docs/data-onboarding.md). S1 records the onboarding state and draws a dataset card and an
onboarding table per dataset.

## Roadmap

- P0, P1, P2 (a, b, c): skeleton, receipts and registry, data gate, validation stages, visual reports - done
- P3: hooks - done
- P3b: AI usage metrics - done (design credited to IchiSieben/claude-kit, Apache-2.0; see NOTICE)
- P3b+: approval tokens, approval-guard, runaway guard, claim check, `usage waste`, `usage regress` - done (see NOTICE for credits)
- P3d-a: method registry and `qrh bench` (planted bugs, noise false positives, known-edge power, history) - done
- P3d-b1: experience packs (`retail-core`, `example-trader`), interviews with answer provenance, PASS · CONDITIONAL,
  Ed25519 human approvals and a signed config baseline (see [docs/security.md](docs/security.md)) - done
- P3d-b2 part A: deployment readiness: one report of every warning of a trial and a signed human acknowledgement
  (`qrh readiness report|ack|status`, [docs/readiness.md](docs/readiness.md)) - done
- P3d-b2 part B: the three LLM rules (cutoff gate, prompts and models as trials, replay cache; `qrh llm`, the
  `llm-gate` benchmark suite, examples 08 and 09, [docs/llm.md](docs/llm.md)) - done
- P3e-a1: dataset onboarding: privacy-safe profile, instrument model, propose, verify, confirm, S1 dataset card
  ([docs/data-onboarding.md](docs/data-onboarding.md)) - done
- P3e-a2: provenance question banks by asset class (37 in total), a data-guard hook that blocks reads of private
  datasets, an optional onboarding benchmark, and stricter asset-class verification - done
- Hardening round: LLM cache bound to the dataset and code hashes, placeholder-hash guard, Stop hook loop fix,
  claim-check test-count matching, shared path helper for the guard hooks, and a minimum track record field in the
  S5 receipt (information only, changes no verdict) - done
- Intake questionnaire: plain questions asked before a run, with answers checked and stored - done
- Parity report: compares backtest daily results with real broker results. It is always created, the user chooses
  how many days to compare, and the report itself never blocks a restart - done
- P3e-b first slices: futures and options profit checks (`pnl.linear`, [docs/pnl-linear.md](docs/pnl-linear.md)) with
  costs taken from a confirmed broker commission record - done (not approved for real use)
- Broker commission validator (stale, conflicting or unconfirmed records block the cost) and the slope limit
  (reported profit against explained profit) - done
- Bench verdict (PASS, FAIL or NO_RULE) recorded for new bench runs - done
- Methodology round: a variance floor in the deflated Sharpe (near-copies of a variant cannot lower the luck bar) and a
  parameter stability check at S5 (spike and plateau), with four thresholds that are always asked of the trader and an
  optional `param-stability` benchmark suite ([docs/methods.md](docs/methods.md)) - done
- Read-only dashboard server (`qrh ui`, local address only) with trials, trial detail, benchmark and warnings pages -
  built, not yet checked in a browser
- Next, in planned order:
  1. P3e-b rest: per-asset-class return models still missing (FX carry, perpetual funding, and later slices of futures
     and options)
  2. a one-off code audit
  3. P3c rest: MCP server and the VS Code panel (the local dashboard server is built)
  4. literature scout
  5. P4: memory (embeddings behind the `Searcher` interface)
  6. P5: maintenance checks
  7. P6: live calibration (whether strategies that passed held up in paper or live trading)
  8. P7: publication (including a git-level pre-push check that verifies a signed approval)

Planned additions, in priority order (none is built yet; each cites published work where one exists):

- **Timing contract:** one written rule for when a signal may trade (for example, a signal at the close fills at the next open), enforced by a test with a planted same-bar leak.
- **Gate self-tests:** every gate has a planted case that must make it fail, so a gate that can never fire is caught.
- **Purge gap in walk-forward splits:** training labels stop one label horizon before the test window, counted in trading sessions. Method from López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), purged cross-validation.
- **Full search family in deflated Sharpe:** the trial count includes every variant tried. Bailey and López de Prado, "The Deflated Sharpe Ratio" (Journal of Portfolio Management, 2014).
- **Minimum track record length:** a result states how many periods it needs before its Sharpe ratio means anything. Bailey and López de Prado, "The Sharpe Ratio Efficient Frontier" (Journal of Risk, 2012).
- **Refuse on thin samples:** a group with too little history blocks the result instead of being filled in.
- **Full-length hashes and bound caches:** receipts use full-length hashes, and any cached result is bound to the exact inputs that produced it.
- **Macro data vintage:** record which release of a revised series was used, and flag current-vintage data. Croushore and Stark, "A Real-Time Data Set for Macroeconomists" (Journal of Econometrics, 2001).
- **Later, with P5 and P6:** circuit breakers and drift monitors for maintenance; an order state machine with a transition log for live trading.

## Disclaimer

Educational and research tooling only. It is not investment advice, makes no
performance claims, and ships with synthetic data only.

## License

Apache-2.0

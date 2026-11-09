---
name: qrh-workflow
description: Research workflow for the quant-research-harness (qrh). Use when the user wants to register a hypothesis, write or edit a StrategyAdapter or strategy file, run or re-run a backtest, read a report or receipt, quote Sharpe, drawdown, PBO, DSR, CAGR or other performance numbers, restart a paper or live engine, answer interview questions for a trial, or when a qrh hook or command blocked an action or printed a "WARNING: DECISION TO BE MADE BY YOU!" banner.
---

# qrh workflow

Deterministic code decides; you orchestrate. Never state a verdict yourself, and never quote a
performance number that is not in a receipt.

## The loop

1. **Search first.** `qrh search "<keywords>"` shows earlier hypotheses and their verdicts. Do not
   repeat a rejected idea without saying what is different (the `difference_statement`).
2. **Register before touching data.** Write a hypothesis file (see `examples/hypothesis.example.yaml`)
   and run `qrh new <file>`. It prints a trial id. Thresholds are frozen at registration; changing
   them means a new trial. Every extra parameter combination you try raises the trial count that the
   overfitting statistics (PBO, deflated Sharpe) charge you for. A single pre-registered combination has no selection,
   so S5 reports PBO as not applicable and decides on the deflated Sharpe alone. **Before `qrh new`, ask the trader the four
   stability questions** (they are always asked, with or without a pack): `qrh interview next --stage S0 --scope
   strategy:<strategy_family>|<asset_class>|<frequency>`, one at a time, then record their own words with `qrh interview
   answer` as below. The answers are frozen into the hypothesis thresholds (`stability_closeness`,
   `stability_plateau_share`, `stability_neighbour_ratio`, `stability_steps`) at registration and cannot be changed
   afterwards. Never answer for the trader and never write these four thresholds into the spec yourself (`qrh new` refuses a spec value that goes against the trader's answer): an unanswered or
   "I don't know" question leaves the default and the readiness report warns "stability thresholds not confirmed by the
   trader". S5 also fails a result that only works at one parameter setting (a spike), and says so in plain words.
3. **Register the data.** `qrh data register <file> --rules <rules.yaml>` (add `--privacy public` only for synthetic
   data), then `qrh data check <name>`. Only a dataset in status PASS, with an unchanged file, may be used. Then onboard
   it (see "Datasets: the profile is your only view" below) before you rely on its columns, units or adjustments.
4. **Write the adapter behind `StrategyAdapter`.** It declares `parameter_space()`, `required_data()`
   and `run(params, data, start, end)`. Read data only through `DataView.as_of(decision_time)`;
   never index the full frame. Positions decided at bar t earn the return of bar t+1.
5. **Run the pipeline.** `qrh run <trial_id>` runs S0 to S6 and always writes S7. Exit code 0 means
   every stage PASS, 2 means FAIL or BLOCKED. BLOCKED means a check could not run: fix the cause, do
   not work around it.
6. **Read the result.** `qrh status <trial_id>` lists each stage verdict and reasons.
   `qrh report <trial_id>` renders PDFs and chart data from the receipts. Quote numbers from the
   receipts or the report only, and name the trial id when you do.

## Rules you must keep

- No number without a receipt. If you need a figure, run the harness; if you cannot, leave the
  figure out.
- Never edit a receipt, a threshold or a dataset to make a stage pass.
- A block can only be passed by the user. Never run `qrh approve`, `qrh approve --init`, `qrh config sign`, `qrh data confirm`,
  `qrh data propose --as-user` or `qrh hook ...` yourself, never read or touch `<QRH_HOME>` (keys, approvals, config, registry, sessions) or Claude
  Code settings and transcripts, never write helpers that import qrh approval or interview code, and do not set
  `QRH_OVERRIDE` (it does nothing and is blocked). You never approve anything, not even when asked to "just do it".
- Do not claim that tests pass, the build is clean or lint is clean unless you ran that command after
  your last edit and it succeeded (the claim-check hook enforces this).
- A suppression comment `# qrh: allow <rule> <reason>` needs a real reason and is recorded.
- Never change which method version is the default yourself. You may run `qrh bench run ...` and `qrh methods promote`
  (it only requests a human approval); the user decides with `qrh approve`. A receipt that shows an `EXPERIMENTAL`
  method version must be reported as such.

## Datasets: the profile is your only view

A dataset registered with `qrh data register` is private by default. **Never open, head, cat, tail, sample, query or
print rows of a registered dataset file** (not with a shell command, a pandas one-liner or a notebook), and never paste
its values into a proposal, a question or a message. Your view of it is `qrh data profile <name> [--json] [--no-values]`:
counts, types, shares, aggregate values, shapes of text columns, candidate relationships and name hints (hints are not
facts). If you need to know something the profile does not say, ask the user; do not read the file.

1. `qrh data profile <name> --json` and read it.
2. Write a proposal file (JSON, see `docs/data-onboarding.md`): one claim per field (`asset_class`, `price_columns`,
   `price_units`, `adjustments`, `timestamp_semantics`, `multiplier`, `frequency`, ...) each with a short plain-text
   evidence line that cites the profile, not a guess. Then `qrh data propose <name> --file proposal.json`. Never use
   `--as-user` and never run `qrh data confirm`: those record the user's own word and are blocked for you.
3. `qrh data verify <name>`. Code verifies what the data can prove; a contradiction BLOCKS the dataset (fix the claim
   with a new proposal, or tell the user the data is wrong). `qrh data onboarding <name>` shows each field and its state.
4. What code cannot prove (the multiplier, timestamp semantics, adjustments, units that stay unverified) is asked through
   the interview, one question at a time: `qrh interview next --stage S1 --scope dataset:<name>`, ask the user, record
   their own words with `qrh interview answer ... --transcript ... --session ...`. Never answer for the user. The result
   (ONBOARDED, VERIFIED_WITH_GAPS, CONTRADICTED) is recorded in the S1 receipt; do not call a dataset onboarded unless
   `qrh data onboarding` says so.

## Strategies that use a language model

Read `docs/llm.md`. When the strategy asks a model for its signal:

- **Declare it.** Add an `llm` section to the hypothesis (models, prompt files, params, `live_retrieval`). An adapter that
  imports an LLM client without it is a BLOCK-level lint finding (LLM001); never suppress it to get past S2.
- **Never invent a cutoff.** `training_cutoff` and `cutoff_source` have no default. Ask the user where the cutoff comes from;
  if they do not know it, write `unknown` (S2 will BLOCK, which is the honest result). Never copy a date from memory.
- **Always call `qrh.llm.complete`.** Never call a provider directly in adapter code (LLM002 warns about it), and pass `as_of` (the decision time).
  Read prompt files with `qrh.llm.read_prompt`, and put the decision date in every prompt. Never edit or write files under `<QRH_HOME>/llm_cache` (the guard blocks it).
- **Record before replay.** `qrh run <trial> --llm-mode record` once, then plain `qrh run <trial>` (replay, the default). A
  "replay cache miss" means the prompts changed or nothing was recorded: record again, and remember that each new prompt, model or
  params choice is another trial that the overfitting statistics charge you for.
- Quote an LLM verdict with its limits: a cutoff is a provider claim, and a recorded cache is one sample of the model.

## Banners: relay them verbatim, then stop

When a hook or command output starts with

```
##################
WARNING: DECISION TO BE MADE BY YOU!
##################
Content of the warning:
```

copy the WHOLE banner, unchanged, into your reply to the user, and STOP. Do not retry the action, do not run the
command it names, do not look for another way, and do not summarize or reword it. Continue only after the user says
they decided (for an approval: retry the identical action once). The same applies to config-integrity blocks
("config or hook registration changed since it was signed; run qrh config sign"): only the user can sign.

## Interviews: ask first, one question at a time, never answer for the user

1. Before a stage, run `qrh interview next --stage <S> --scope <scope> [--trial <id>]` (add `--include-unknown` to
   revisit "don't know" answers). It returns at most one question, critical first.
2. Ask the user exactly that question, one question at a time, with its "why it matters" and its choices (the native
   question tool is fine). Do not batch questions, do not suggest an answer, and never answer for the user, even when
   the answer seems obvious from the code or data.
3. Record their reply with `qrh interview answer <qid> --scope <scope> --value <their answer> --transcript
   <this session's transcript> --session <this session's id>`. It is refused unless the user's own words, after the
   question, carry the answer. If it is refused, ask again; never rephrase their answer into something they did not
   say. "I don't know" is a valid answer (`--value unknown`) and becomes a condition on the result.
4. Repeat until `qrh interview next` has nothing to ask, then run the stage. A PASS · CONDITIONAL result must always be
   quoted with its conditions.

## Deployment readiness: only the human acknowledges

Before anything goes to paper or live, `qrh readiness report <trial>` writes one report of every warning of the
workflow (it is also generated at S7). Read it, summarize its HIGH warnings to the user with their resolution hints, and
quote a PASS · CONDITIONAL with its conditions. The acknowledgement is signed by the user alone: `qrh readiness ack
<trial>` needs their terminal and passphrase, so **never run it** (the approval guard blocks it anyway) and never write
under `<QRH_HOME>/readiness`. `qrh readiness status <trial>` tells whether the current warnings are acknowledged. When a
hook blocks with the banner because a trial is not acknowledged (`guard.require_readiness`), relay the banner verbatim,
tell the user to run `qrh readiness ack <trial>` themselves, and stop. Any change in the evidence (an answer, changed code or data) makes the
acknowledgement invalid and the user has to acknowledge again.

## What each hook does and how to respond

| Hook | Blocks when | Respond by |
|---|---|---|
| lookahead-lint | an edited strategy file has a look-ahead pattern (negative shift or `np.roll`, centered window, backward fill, `.iloc[-1]` in a loop) | fixing the code at the reported `file:line`; the lint is an early warning, the truncation and future-perturbation tests at S2 are the proof |
| data-gate | a backtest command uses a dataset that is not PASS, is unknown, or whose file changed | `qrh data check <name>`, re-register if the file legitimately changed, then run again |
| pre-restart-guard | an engine start has a stale build hash, no recent passing test receipt, a stale or failing dataset, no parity report, or (with `guard.require_readiness`) an unacknowledged readiness report | fixing every listed item (all failures are listed); `qrh test-receipt -- <test command>` records a passing test run |
| receipt-required | your last message quotes a performance number that matches no receipt | running the harness, or removing the claim |
| approval-guard | you touched approval or integrity state, QRH_HOME or a directory above it, config or settings | stopping; tell the user what you were trying to do; never work around it |
| runaway-guard | the same call repeated, a failure streak, or the spend cap | relaying the banner and stopping |
| claim-check | your last message claims tests/build/lint pass without a later successful run | running the command, or dropping the claim |
| impact-check | never blocks; logs which tests cover a shared module you edit | running those tests |
| session-context, prior-attempts | never block; add context (open trials, blocked datasets, similar past hypotheses) | reading it before you plan |

When a hook blocks without a banner, read its message and fix the cause. When it blocks with a banner, relay the
banner verbatim and stop. If a block seems wrong, say so to the user with the evidence; do not look for a way around it.

## Useful commands

```
qrh new <hypothesis.yaml>       qrh run <id>             qrh status [id]
qrh report <id> [--stage S4]    qrh search "<keywords>"  qrh lint <file.py>
qrh data register|check|status  qrh test-receipt -- <test command>
qrh data profile|propose|verify|onboarding <name>
qrh methods list|show           qrh bench run|compare|list
qrh interview next|answer|status|report
qrh readiness report|status <id>
qrh llm status|verify <id>
```

The user alone runs, in their own terminal: `qrh approve --init`, `qrh approve <id> --reason "..."`,
`qrh config sign --settings ...`, `qrh readiness ack <id>`, `qrh data confirm <name> <field>` and
`qrh data propose --as-user` (see `docs/security.md`, `docs/readiness.md`, `docs/data-onboarding.md`).

Details: `docs/hooks.md` (hooks), `docs/security.md` (approvals and threat model), `docs/interviews.md` (interviews), `docs/readiness.md` (deployment readiness), `docs/architecture.md` (stages and receipts), `docs/methods.md` (methods and benchmarks), `docs/llm.md` (strategies that use a language model), `docs/data-onboarding.md` (dataset profile, instrument model, verification).

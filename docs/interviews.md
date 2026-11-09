# Interviews

Some facts about a study cannot be read from data: how a rate series was built, whether timestamps mark the bar
open or close, how many variants were tried by hand. The interview asks the human for them, one question at a
time, remembers the answers, and records in every receipt what was answered, unknown or still open.

## Question spec

Banks live in packs (`questions/*.yaml`); `retail-core` ships the generic one.

| Field | Meaning |
|---|---|
| `id` | stable identifier |
| `stage` | `S0`..`S7` or `B1`..`B6` |
| `scope` | `hypothesis`, `dataset`, `strategy`, `account` or `session` |
| `applies_to` | optional `asset_class`, `strategy_family`, `frequency` lists |
| `text`, `why_it_matters` | the question and the reason to answer it |
| `answer_type` | `choice`, `text`, `number` or `bool` |
| `choices` | for `choice`; each has `id`, `label`, optional `risk` tag (for example `lookahead`) |
| `critical` | true if an open answer should condition a final PASS |
| `maps_to` | optional mapping of the answer to a parameter, flag, threshold or trial count (a mapping may carry its own `value`, which can be a number, a list or null for "the default") |
| `unknown_risk` | text recorded when the answer is "don't know" |

A risk-tagged choice on a dataset question adds that flag to the dataset. A number question mapped to
`imported_trials` adds to the family trial count used by the deflated Sharpe.

## Ask only what is missing

Answers are stored per scope key (`dataset:<name>`, `hypothesis:<trial>`, `strategy:<family>`, `account:<name>`,
`session:<id>`) and reused until that scope's identity changes: the dataset content hash, the hypothesis hash or
the account hash. Changing an answer creates a new version; receipts reference `question@scope` with its version.

```
qrh interview next --stage S1 --scope dataset:prices --trial <id> --json   next unanswered question, critical first
qrh interview answer <qid> --scope <key> --value "..." --transcript <file> --session <id>
qrh interview add-question --stage S1 --scope dataset:prices --text "..." --why "..."
qrh interview status [--trial <id>] [--stage S1] [--json]
qrh interview report --trial <id> [--stage S1]
```

The agent asks the question exactly as returned, one at a time, in its native question UI or plain chat, then
records the user's own answer. Follow-up questions it adds are marked `origin: llm` and are never critical.
`--include-unknown` offers "don't know" questions again.

## Warn-only policy

Unanswered questions never block a stage. Each stage receipt records the interview status (counts and ids of
answered, unknown and pending questions). The effect shows up only at S7.

## "Don't know"

`--value unknown` is always valid. It is recorded with the question's `unknown_risk`, stays visible in reports and
keeps the final verdict conditional if the question is critical.

## PASS · CONDITIONAL

If every gate passes but a critical question in scope is pending or answered "don't know", or any answer carries a
risk tag, the final verdict is `PASS_CONDITIONAL`, shown everywhere as **PASS · CONDITIONAL**, with the conditions
listed (question id, text, status, risk) in the S7 receipt, the markdown report and the summary page. FAIL, BLOCKED
and INCONCLUSIVE are unchanged. `qrh run` exits 0 for it. Whoever quotes a conditional result must quote its
conditions too (the session-context hook and the workflow skill say so).

To resolve a condition, answer the question with a clean choice and run the trial again: the answer gets a new
version and the verdict becomes PASS. A risk-tagged answer stays a condition until the underlying data is fixed.

## Answer provenance

Answers must come from the user. The agent may not answer on the user's behalf: a transcript-verified answer must
trace to the user's own words in the session, and an answer typed by a person in their own terminal is recorded
with a different provenance (`transcript-verified` or `human-cli`). The agent is not allowed to use the human
mode.

Rules for `--transcript T --session S` (see also [security.md](security.md)):

- S must be a session Claude Code reported to the SessionStart or UserPromptSubmit hook, T must equal the transcript
  path recorded for it, and T must live under `~/.claude/projects`. A file written by the agent is refused.
- Only user messages posted after the question was issued count; assistant text, ordinary tool results, subagent and
  harness-injected records never do. From `AskUserQuestion` only the structured answers (`toolUseResult.answers`) count,
  never the result text, which repeats the question the agent wrote.
- Matching is by whole word or phrase, case and whitespace normalized. A text answer needs at least 3 characters (or
  must be the whole message); a number must appear as a whole number; a choice or bool label or id must be the whole
  message, or a whole word of at least 3 characters in a short message (up to 60 characters), while a label of 8 or
  more characters may appear anywhere. "Don't know" phrases count for `unknown`.

## Reports

`qrh interview report --trial <id>` writes `interview.<stage>.pdf` and `interview.<stage>.chart.json` under
`<QRH_HOME>/reports/<trial_id>/`: a progress bar, a traffic light, each question with its answer and what it
mapped to, "don't know" entries with their risks, and pending questions with why they matter. The same pages are
added to the trial's combined PDF after S7, drawn from the hash-verified `interview` artifact of the S7 receipt.

## Example questions (from `retail-core`)

- S0: What result would make you drop this hypothesis? (recommended)
- S1: How were the risk-free rates in this options data produced? (critical; a choice of "the current curve applied
  to all of history" carries the `lookahead` risk)
- S1: Do the timestamps mark the bar open or the bar close? (critical)
- S1: Does the data include contracts that have already expired? (critical)
- S1: Is the continuous futures series back-adjusted at the rolls? (critical)
- S1: Is fundamental data keyed by filing date or by period end? (critical)
- S1: Is macro data the first release or the revised history? (critical)
- S2: When is the signal computed, and when does the order fill? (critical)
- S3: How many variants did you try outside the harness? (critical; a number added to the family trial count)
- S4: How often will the live system re-optimize compared with the tested walk-forward? (critical)
- S6: What is your real fee schedule? (recommended)
- S7: What is the intended use of this result? (recommended)

Some of these apply only to certain asset classes or strategy families (`applies_to`). See
[packs.md](packs.md) for writing your own bank.

## Provenance banks by asset class

`retail-core` ships 37 questions. Besides the 15 generic ones above, 22 ask where a dataset's numbers came from. All are
dataset-scope S1 choice questions, offered only when the hypothesis's `asset_class` matches (the common ones always). Critical
ones are marked with `*`.

| Group | Question ids |
|---|---|
| Options (5) | `option-dividend-yield-source`*, `option-greeks-origin`*, `option-greeks-underlying-price`*, `option-quote-underlying-sync`*, `option-price-type` |
| Futures (3) | `futures-roll-rule`*, `futures-settlement-vs-last`*, `futures-continuous-construction` |
| Equity (4) | `equity-survivorship-free-universe`*, `equity-delisted-included`*, `equity-adjustment-timing`*, `equity-corporate-action-source` |
| FX (3) | `fx-rollover-cutoff-timezone`*, `fx-quote-side`, `fx-swap-carry-source` |
| Crypto (4) | `crypto-venue`, `crypto-perp-funding`, `crypto-index-vs-last`, `crypto-fee-tier` |
| Common, empty `applies_to` (3) | `data-history-revised`*, `data-timezone`*, `data-licensing` |

- Options: the dividend yield, the Greeks and the underlying price are inputs that can silently use future or mismatched
  information, and they feed every option valuation.
- Futures: the roll rule and the settlement-versus-last price decide which contract and which price a backtest trades, so a
  wrong answer changes returns directly.
- Equity: survivorship, delisted names and the timing of adjustments change which securities and prices a signal could have seen.
- FX: the rollover cutoff time zone decides which day a position or a carry belongs to; the other two FX questions are recommended.
- Crypto: none is critical; venue, funding, index and fee questions are recommended and stay visible in reports.
- Common: a revised history or a mistaken time zone can place a value before it was known, whatever the asset class.

Each critical question that stays open, or is answered "don't know", keeps the final verdict **PASS · CONDITIONAL**.

## Stability thresholds: always asked

Four thresholds of the S5 parameter stability check belong to the trader, not to the agent. They are asked at intake, before a
hypothesis is registered, whether or not any pack is registered (they are built into the harness, so a pack cannot leave them
out), through the same question model, scope keys and answer provenance as every pack question. Scope is the strategy family
(`strategy:<strategy_family>|<asset_class>|<frequency>`, known before registration), stage `S0`, never critical.

| Question id | Plain-language question (default shown) | Choices | Threshold |
|---|---|---|---|
| `stability-closeness` | A nearby setting counts as "about as good" when its out-of-sample Sharpe ratio is at least this share of your chosen setting's. Suggested: at least 70%. | keep the default (recommended); stricter 85%; looser 55% | `stability_closeness` |
| `stability-plateau-share` | The harness nudges each number of your chosen setting up and down, one parameter at a time. What share of those nearby settings must be "about as good" for the result to count as a plateau and not a lucky peak? Suggested: at least 50%. | keep the default (recommended); stricter 65%; looser 35% | `stability_plateau_share` |
| `stability-neighbour-ratio` | For each parameter separately, the harness nudges it up and down and averages the nearby settings' out-of-sample Sharpe ratios. The worst parameter must keep what share of your chosen setting's ratio? Suggested: at least 50%. | keep the default (recommended); stricter 70%; looser 30% | `stability_neighbour_ratio` |
| `stability-steps` | How far should the harness nudge each number, up and down, to build the nearby settings (used for one setting and for a grid alike)? Suggested: 10% and 20% of each value. | keep the default (recommended); smaller 5% and 10%; wider 10%, 20% and 30% | `stability_steps` |

"I don't know" is always a valid answer (`--value unknown`). The first choice, "Keep the default", maps to no value: whatever the
default is (the core value, or a pack that tightened it) stays. Any other choice is a value that is frozen into the hypothesis
thresholds at registration like every other threshold. The spec may not override the trader: a stability value written in the spec
must equal what the trader answered ("keep the default" means the default in force, the core value or a pack's), and with no answer
(or "I don't know") it may not be looser than the default in force; otherwise `qrh new` refuses the hypothesis with a plain reason
(answer the question with that value, or drop it from the spec). A tighter value with no answer is accepted, labelled `hypothesis`
and still reported as unconfirmed. The S0 receipt records the value, source and default in force of each of the four
(`params.stability_thresholds`).
The answers are read only when a trial is registered; answering later does not change a registered trial.

If a question is unanswered or answered "I don't know", the default applies, `qrh new` prints a notice, and the readiness report raises
the medium warning **stability thresholds not confirmed by the trader** (see [readiness.md](readiness.md)). Answers are given by the
trader (`--transcript`/`--session`, or `--human` in their own terminal); an agent may not answer for them.

```
qrh interview next --stage S0 --scope "strategy:trend|equity|1D"
qrh interview answer stability-closeness --scope "strategy:trend|equity|1D" --value keep_default --transcript T --session S
```

## Questions about a dataset's instrument model

`qrh data verify` turns every unverified field of a dataset's instrument model into a question of origin `onboarding`
(scope `dataset:<name>`, stage S1, id `onb-<field>`): critical for the multiplier, adjustments, timestamp semantics and price
units, recommended for the rest. They need no pack. Answering `confirm` confirms the claim, another value corrects it, and an
answer is stale when the file or the claim changes. See [data-onboarding.md](data-onboarding.md).

## Questions for strategies that use an LLM

`applies_to` accepts `uses_llm: true`, which narrows a question (or a lesson, or a whole pack) to hypotheses that declare an
`llm` section (it combines with `asset_class`, `strategy_family` and `frequency`). retail-core uses it for three critical
questions at S2 (`llm-cutoff-source`: how the cutoff was determined; `llm-live-retrieval`: retrieval or tools that return
current data, where the answer `yes_current` carries the risk tag `lookahead`; `llm-finetune-or-test-data`: fine-tuning or
prompting with market data from the test period, risk-tagged too). The fourth question, prompt iterations outside the harness,
is the existing `variants-outside-harness` (stage S3, every strategy), whose text now also asks "How many other prompt or model
variants did you try for this idea outside the harness?"; its number still maps to the imported trial count (see
[llm.md](llm.md)).

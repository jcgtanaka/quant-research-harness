# AI usage metrics (P3b)

`qrh usage` turns Claude Code transcripts into per-session metrics: tokens, cost, cache behavior,
tool mix, hook blocks and which trial each message belonged to. It is local, read-only on the Claude
config directory, and uploads nothing. The design follows IchiSieben/claude-kit (Apache-2.0, see `NOTICE`);
the code is independent.

## Commands

```bash
qrh usage init                      # writes example thresholds.json and prices.json (review and edit them)
qrh usage harvest [--claude-dir D]  # transcripts -> $QRH_HOME/usage/sessions.json
qrh usage report [--by session|day|week|model|trial|trial-stage|project] [--json] [--pdf PATH]
qrh usage verify [--scope transcript|strict] [--ack S004 --reason "..."]
qrh usage waste [--json] [--pdf PATH]
qrh usage regress [--by week|model|config] [--json] [--pdf PATH]
qrh usage log-local < ollama_response.json   # [--task LABEL] [--session-id ID] [--model NAME]
```

`--claude-dir` defaults to `$CLAUDE_CONFIG_DIR`, else `~/.claude`. Exit code 2 means blocked: missing or
invalid thresholds, an unreadable archive, an invalid `QRH_PROTECTED_LABELS`, a failed `verify`, or missing waste thresholds for `waste`.

## Sources

| Source | Used for |
|---|---|
| `<claude>/projects/<project>/<session>.jsonl` | main transcript: usage per message, model, timestamps, tool names, compaction boundaries, last `cost-state` |
| `<claude>/projects/<project>/<session>/subagents/*.jsonl` | subagent transcripts, aggregated per file |
| `<QRH_HOME>/logs/session_trials.jsonl` | which trial a session was working on, and when (written by the `data-gate` hook) |
| `<QRH_HOME>/logs/hooks.jsonl` | hook blocks and overrides per session |
| `<QRH_HOME>/logs/local_llm.jsonl` | local-model usage (tokens only) |
| `<QRH_HOME>/usage/thresholds.json`, `prices.json` | the rules and prices (no numeric fallbacks in code) |

Reports render only from `sessions.json` and these logs (the one-source rule). Message text is never read
into memory beyond the line being parsed and is never stored.

## Counting: each message once

Claude Code writes one assistant record per content block, each repeating `message.usage`. Counting every
record inflates totals (2.04x on the data this was measured on). A message id is counted once; a record
without an id counts once. In main transcripts the copies are identical. In subagent transcripts
`output_tokens` can grow across the copies, so the largest value of each field wins. Records with the
model `<synthetic>` carry no API usage and are skipped. Cache writes are split into 5-minute and 1-hour
using `cache_creation` when present, otherwise all counted as 5-minute.

## Per-session record

`session_id`, `project_label` (basename of the most frequent working directory, `home` for the home directory),
`start`, `end`, `duration_s`, `models` (tokens by kind and requests, main chain, deduplicated), `models_raw`
(main chain without deduplication, for `verify`), `subagents` (the same, per subagent file; `_inline` holds sidechain
records found in the main file), `cost_usd`, `cost_source`, `cost_reason`, `cost_by_model`, `harness_usage`, `peak_context`
(largest input + cache read + cache write of one main-chain request), `hit_pct` (100 * cache read / (input + cache read
+ cache write), main chain), `cold_cache_events`, `tools` (counts by name), `compactions`, `malformed_lines`, `hooks`,
`trials` and `cost_alloc`.

A cold-cache event is a main-chain request whose own hit rate is below `cache_cold_pct`, excluding the first request
and the first one after a compaction.

The archive merges by session id across runs, so records outlive Claude Code's 30-day transcript sweep. A newer harvest
of a transcript that is still present replaces its record. A session's day and week are those of its start (UTC).

## Cost sources

- `harness`: the last `cost-state` record (cumulative `totalCostUSD` and per-model `costUSD`).
- `estimate`: tokens (main chain plus subagents) times `prices.json`, only when there is no cost-state.
- `none`: no cost-state and no usable price. `cost_reason` says why (prices file missing, or the model names that
  have no price). Nothing is guessed; `qrh usage report` prints how many sessions lack a cost and leaves them out of totals.

`prices.json` holds USD per million tokens per model (`input`, `output`, `cache_read`, `cache_write_5m`, `cache_write_1h`,
`tier` one of `local`, `small`, `mid`, `frontier`). `qrh usage init` writes placeholders named `example-*`: fill them from
your provider's pricing page. `thresholds.json` needs `cache_cold_pct`, `context_high_tokens`, `mechanical_tool_share`
(a fraction) and `mechanical_min_tools`; a missing or invalid file stops `harvest` and `report` with exit 2, because a
threshold you cannot read is not one that was not crossed.

Trial attribution: each message goes to the most recent trial tagged in its session at or before its timestamp, else
`unattributed`. The harness reports one cost per session, so a trial's cost is the session cost split by share of
(price-weighted, else uncached-token-weighted) usage; `cost_alloc` says which.

## Plays

Each play names its rule and the numbers that fired it. They are questions, not verdicts.

| Play | Fires when |
|---|---|
| `strong-on-mechanical` | the model with most output tokens has tier `frontier` in `prices.json`, at least `mechanical_min_tools` tool calls, at least `mechanical_tool_share` of them read-only (Read, Grep, Glob, Bash, WebFetch), and no Edit, Write, MultiEdit, NotebookEdit, Agent or Task. A heuristic: it infers the work from the tool mix. |
| `cold-cache` | more than 3 cold-cache events |
| `high-context` | `peak_context >= context_high_tokens` |
| `many-hook-blocks` | 5 or more hook blocks in the session |
| `gate-fail-after-edit` | at least one look-ahead lint or data-gate block in the session |

The two fixed counts (3 and 5) are written in `src/qrh/usage/quality.py`.

## Local models

`qrh usage log-local` reads one Ollama `/api/generate` or `/api/chat` JSON response (or its streamed lines) on stdin and
appends `{ts, session_id?, model, prompt_tokens, completion_tokens, duration_ms, task}` to `local_llm.jsonl`. Prompts and
replies are never stored. Reports show these as tokens with cost 0, in the day, week and model groupings.

## verify

`qrh usage verify` compares our deduplicated per-model totals with the last cost-state for every archived session that has
one, under two hypotheses: the cost-state covers the main chain only, or the main chain plus subagent transcripts. It prints
the dedup inflation, which hypothesis fits better, the match rate for sessions with subagents and the observed causes of
differences.

- `--scope transcript` (default): models seen in the transcripts; input, cache read and cache creation (the kinds
  deduplication changes). **Gate: exit 2 only if a session without subagents differs by more than 1%.** Sessions with
  subagents are reported with their match rate and labelled "tokens are a lower bound", but they do not gate. A mismatch
  that is reconciled exactly by a checkable cause (below) counts as a match and is listed as explained.
- `--scope strict`: every model and kind in the cost-state, every session gates. A diagnostic that is expected to fail on
  real data, for two reasons: (a) Claude Code bills a background model (a small one) that never appears in transcripts,
  and (b) subagent transcripts under-record output tokens (output is written before the message finishes).

Reports always take cost from the Claude Code cost-state; token tables are labelled lower bound where subagents were used.

### Acknowledging an unexplained session

A gate that stays red gets ignored, so a session whose cause cannot be found can be acknowledged:

```bash
qrh usage verify --ack S004 --reason "cause not found after checking stop reasons, gaps and away summaries"
```

`--ack` takes the session alias printed by `verify` and a reason of at least 10 characters. It only accepts a session that
is a gating mismatch in transcript scope. It records the alias, reason, timestamp, the mismatch figures at that moment and a
fingerprint of the session's totals in `<QRH_HOME>/usage/verify_acks.json`. Acknowledged sessions are listed separately as
"acknowledged, unexplained" and do not gate. **The acknowledgement expires by itself when the session's numbers change** (a
re-harvest with different totals): `verify` then says so and the session gates again, so an ack cannot hide a new problem.
A session acknowledged is still a mismatch in the counts; a corrupt acknowledgements file blocks `verify` (exit 2).

### Checkable causes of a no-subagent mismatch

At harvest, for a session without subagents whose totals differ, `alignment` tries three exact reconstructions and stores
the first that fits (numbers only):

| Cause | Test |
|---|---|
| `stale-cost-state` | the cost-state equals the transcript up to its own position in the file (the session kept running after the last cost-state write) |
| `incomplete-message` | without messages that never reached a stop reason (interrupted streams) the totals match: the harness does not bill them |
| `counter-restart` | the cost-state equals the transcript from some request onward, exactly (the harness counter started again at a resume); needs an exact match on every compared count because it searches start points |

What real data showed (one maintainer's transcripts; numbers only):

- Deduplication is right for the main chain: 29 of the 33 sessions without subagents match within 1% (`--scope transcript`).
  Of the other 4, three are explained by the causes above (one each: stale cost-state, interrupted message, restarted
  counter). One stays unexplained: its cache-read total is 19% below the cost-state across several models; we
  checked sub-sets by stop reason, idle gaps and away-summary markers and none reproduces the difference, so the cause
  is not confirmed (the cost-state counters are cumulative and written at four points, the last two identical).
- Without deduplication the totals are 2.04x too high.
- The cost-state includes subagents: in the sessions where the two hypotheses differ, 17 of 19 fit better with subagents added.
- Subagent transcripts carry no cost-state of their own (0 of 152 files). With subagents, 12 of 28 sessions match within 1%
  under `--scope transcript`; the rest are reported, not gating.
- The cost-state also bills models that never appear in a transcript, in 52 of 61 sessions, and subagent output tokens in
  transcripts stay below the cost-state even after taking the largest copy.

## waste

`qrh usage waste [--json] [--pdf PATH]` shows, per session and in total, where tokens were spent that did not need to be. Each
waste type carries a one-line fix. Counts come from facts captured at harvest (hashes and counts only), thresholds from
`thresholds.json`.

| Type | Rule | Fix |
|---|---|---|
| `re-reads` | a path read more than `waste_reread_k` times with no edit of it, typed prompt or compaction in between (paths are shown as `f-xxxx` aliases only) | read once per task, search or use line ranges |
| `large-tool-outputs` | a tool result above `waste_large_output_tokens` (characters / 4, an estimate) | trim output where it starts |
| `repeated-failing-commands` | the same Bash command (normalized, shown as `c-xxxx`) failed at least `waste_repeat_fail_k` times | write the working command down, fix the cause |
| `cache-cold-events` | main-chain requests below `cache_cold_pct` hit rate; tokens re-written by those requests are always reported; dollars are the **estimated excess cost**: cold cache-write tokens x (cache write price - cache read price) per model (5m and 1h writes priced separately), only with a real (not `example-*`) price in `prices.json`, else `n/a (no price)`. Never derived from the cost-state | compact before a pause, start fresh |
| `frontier-on-mechanical` | the existing `strong-on-mechanical` play | cheaper model or subagent for read-only exploration |
| `compactions` | count, with the peak context | one session per task, compact on purpose |

`thresholds.json` needs `waste_reread_k`, `waste_large_output_tokens` and `waste_repeat_fail_k` (positive integers); if one is
missing `waste` exits 2 naming it, because a threshold you cannot read is not one that was not crossed. `qrh usage init`
writes examples. Charts: `waste_summary`, `waste_by_type`, `waste_sessions` under `usage/reports/`.

## regress

`qrh usage regress --by week|model|config` splits sessions at change points and compares each group with the one before:

- `week`: ISO week of the session start (UTC). `model`: the model with most output tokens in the session, groups in order
  of first use. `config`: a hash of the hooks section of `~/.claude/settings.json` (and the project's) plus the bytes of the
  user and project `CLAUDE.md` files, captured **once, when the session is first harvested**, and only the digest is kept.
  So harvest regularly: a session first harvested after you changed your setup gets the new hash, whatever it ran under.
- Per-turn metrics (a turn is one typed prompt and everything until the next; main chain only, subagents excluded): new
  tokens (input + output + cache writes), failed tool calls, hook blocks (from `hooks.jsonl`, by timestamp), cost (the
  session cost split over its turns by price weight; an estimate) and cache hit %.
- Tests: Mann-Whitney U (two-sided, scipy) with the rank-biserial correlation as effect size, for the continuous metrics;
  Fisher's exact test with the odds ratio for the rates (turns with a failure, turns with a hook block).
- Fewer than 20 turns on a side gives a small-sample warning. The report never claims a cause: the groups differ in the work
  done, and turns of one session resemble each other, so the p-values are optimistic. Treat them as pointers.

Charts: `regress_summary`, `regress_groups`, `regress_comparisons`.

## Reports

`qrh usage report` prints a table for the chosen grouping, the totals, how many sessions have no cost, local-model usage and the
plays that fired. `--json` prints the report as JSON; `--pdf PATH` writes a PDF in the existing visual theme (summary, daily
cost by model as stacked bars of the two leading models plus "other", tokens by kind, cache hit rate over time, cost per trial,
plays). Every run also writes `<QRH_HOME>/usage/reports/<chart>.chart.json`, the data behind each figure, for the future
dashboard. Output is deterministic: the same archive gives byte-identical JSON and PDF.

## Privacy

Outputs (report text, JSON, PDF, chart JSON) carry the project label, model names, numbers, the plays, and hashed aliases of files and commands only. Sessions
are named S001.. by start time. They never carry message or prompt text, file paths, working directories or session ids.
`QRH_PROTECTED_LABELS` (a regular expression) makes matching project labels appear as `private-N` in every output. The archive
`sessions.json` is a local file: it holds the real project label and the session ids, so do not publish it. Tests use synthetic
transcripts in temporary directories only.

## Limits

- Estimates need prices you maintain; sessions without a cost-state or price have no cost.
- A session's whole cost is attributed to its start day and week, not split across days.
- The harness cost-state of a session that is still running lags the transcript.
- Subagent output tokens are under-recorded in transcripts (see above), so token output totals for subagent work are a lower bound.
- Per-turn data covers the main chain; the config hash is captured at first harvest, not at the time the session ran.
- Trial attribution depends on `session_trials.jsonl`, which only records Bash commands the `data-gate` hook recognized.

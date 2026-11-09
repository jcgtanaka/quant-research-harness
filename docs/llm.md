# Strategies that use a language model

Three problems appear when a strategy asks a language model for its signal, and the ordinary checks cannot see them.

1. **Memorization is look-ahead.** A model trained on data up to a date D already knows what happened in any backtest
   period before D. The leak lives in the model weights, so no truncation test can catch it: the prompt contains only
   past data, and the answer is still informed by the future.
2. **Every prompt and every model is a trial.** Reword a prompt, swap a model, try another temperature: each is another
   configuration tried on the same data, and the overfitting statistics (PBO, deflated Sharpe) must charge for it.
3. **Answers vary between runs.** A verdict that depends on a fresh model call cannot be reproduced.

Three rules close these. They apply only to a strategy whose hypothesis declares an `llm` section; every other strategy is
unaffected (its receipts record no new method and no new field).

| Rule | What it does | Where |
|---|---|---|
| 1. Training-cutoff gate | every evaluation window must start after the latest declared model cutoff plus a margin | S2, method `s2.llm_gate@1` |
| 2. Prompts, models and variants are trials | prompt files are hashed with the code; the S3 grid is multiplied by the declared variants | S0 to S6 |
| 3. Replay cache | every model call goes through `qrh.llm.complete`; answers are recorded once and replayed | `qrh.llm`, method `llm.replay_cache@1` |

## The `llm` field of the hypothesis

```yaml
llm:
  models:
    - id: my-model-2025-01            # the identifier as the provider names it
      provider: my-provider
      training_cutoff: "2024-06-30"   # YYYY-MM-DD, or unknown; REQUIRED, no default
      cutoff_source: provider_documented   # provider_documented | estimated | unknown; REQUIRED
  prompt_files: [prompts/system.txt, prompts/user.txt]   # relative to the hypothesis file; hashed into the code hash
  params: {temperature: 0}            # free-form (JSON values), hashed
  live_retrieval: false               # REQUIRED; or {point_in_time_evidence: "<text>"} when retrieval or tools are used
  variants:                           # optional explicit variants; see rule 2
    - {name: terse, model: my-model-2025-01, prompt_file: prompts/system.txt, params: {}}
```

It is validated at S0 (a missing required key, a bad date, an unknown key, a variant that names an undeclared model or
prompt file, or a prompt path that leaves the hypothesis folder is refused), stored with the hypothesis, immutable like the
thresholds, and hashed only when present, so a hypothesis written before the field existed keeps its hash. `live_retrieval:
true` is accepted by the validator but BLOCKS at S2 because it has no evidence text. A threshold
`llm_cutoff_margin_days` (default **180**) is filled in at S0 like the other thresholds; raise it freely, lower it only
through the approval flow of packs (`docs/packs.md`).

## Rule 1: the training-cutoff gate

`s2.llm_gate@1` (default status; there is no earlier check to replace, and it does nothing for a strategy without `llm`).

- A model with `training_cutoff: unknown`, or with `cutoff_source: unknown`, makes S2 **BLOCKED**, and the reason names the
  model: a date nobody can source cannot be compared with anything.
- Otherwise the binding date is the **latest** declared cutoff plus `llm_cutoff_margin_days`. Every evaluation window the
  hypothesis declares (the development window, and any other window in the spec) must start **strictly after** it. A
  walk-forward or out-of-sample span derived from a window lies inside it, so the start of the window covers it. Otherwise S2
  is a **FAIL** that names the model, its cutoff, the margin, the window and the earliest contaminated date. Models used by
  `variants` are part of the gate.
- `live_retrieval` declared without evidence text is BLOCKED. With evidence text the gate passes, and the readiness report
  carries a HIGH warning that quotes the text: the harness cannot check the claim.
- The evidence (models, cutoffs, margin, binding date, windows) is in the S2 receipt params (`llm_gate`) and in the S2
  artifact `llm_gate`, and the S2 PDF has a page, **S2 LLM contamination gate**: a timeline with the cutoffs, the margin,
  the binding date and each window, with the result written out in words (never colour alone).
- When the gate does not pass, S2 does not run the adapter: its output could not be trusted, and a replay-cache miss must
  not hide the reason. The truncation, perturbation, same-bar and sanity results are recorded as `NOT RUN`.

## Detection of undeclared use (LLM001)

A deterministic static check over the adapter source files the loader tracks. When the hypothesis has **no** `llm` section
and a source file imports or loads a common LLM client (`anthropic`, `openai`, `google.generativeai`, `google.genai`,
`ollama`, `litellm`, `langchain*`, `mistralai`, `cohere`, `together`, `groq`, `llama_cpp`, `vllm`, a transformers
`pipeline`, or `qrh.llm` itself) the finding **LLM001** is BLOCK level: "adapter appears to use an LLM but the hypothesis
declares none". A pack can add libraries with a lint rule whose id is `LLM001` and `ast: {import: <module>}`.
`# qrh: allow LLM001 <reason>` suppresses it like any lint finding and is carried into the readiness report (medium). An
unsuppressed LLM001 also stops S2 from running the adapter.

## Provider code outside `qrh.llm.complete` (LLM002)

When the hypothesis **does** declare `llm`, any import or call of one of the same client libraries (not `qrh.llm`) in a tracked
source file that does **not** call `register_backend` is a **WARN**-level finding: "provider client used outside
qrh.llm.complete; replay cache and the truncation prompt detector cannot see these calls". A file that calls `register_backend`
is the intended place for provider code and is exempt. The finding is advisory (it never blocks), is recorded in the S2 lint
findings and page, becomes one MEDIUM readiness warning (`llm-direct-client`) listing the files, and
`# qrh: allow LLM002 <reason>` suppresses it (the suppression is listed in readiness too). A pack lint rule with id `LLM002`
adds libraries. It is best effort: plain HTTP, subprocesses and dynamic imports still evade it.

## Rule 2: prompts, models and variants count as trials

- **Prompt files are code.** The loader hashes every file in `prompt_files` into the adapter code hash, so a prompt edit
  changes the code hash recorded in the S2 and S3 receipts. `qrh.llm.read_prompt(variant, __file__)` reads the same bytes
  and refuses to continue if they changed after they were hashed.
- **The S3 grid is multiplied by the variants.** No `variants` declared means exactly one variant formed by the declared
  models, prompt files and params. The adapter receives the variant in `params["llm_variant"]` (`name`, `models`,
  `prompt_files`, `params`, `prompt_hash`; a variant's `params` are merged over the top-level `params`). The S0
  `parameter_space_size` must equal the adapter's grid times the number of variants; a mismatch is BLOCKED at S3.
- **Each evaluated combination is logged with its variant name and prompt hash**, so PBO (a column per variant) and the
  deflated Sharpe ratio (the family count) include them. S2 runs the truncation test for every variant.
- **Interview.** The retail-core question `variants-outside-harness` ("How many other prompt or model variants did you try
  for this idea outside the harness?") already covers every strategy and maps its number to the imported trial count; its
  text now names prompts and models. It is not duplicated.
- **S0 notice.** When another trial of the same family has a different prompt, model or params fingerprint, registering
  prints and records `prompt or model changed since trial <id>: counts as another trial; family count is now N` (N is what
  the family count becomes once every registered trial has run). It is a notice, never a block. It is written once, at
  registration, so re-running a trial cannot change its S0 receipt.

## Rule 3: the replay cache

```python
from qrh import llm
text = llm.complete(model_id, prompt, params={"temperature": 0}, as_of=decision_time)
```

`qrh.llm.complete(model_id, prompt, *, params=None, as_of=None)` is the single entry point an adapter uses; never call a
provider directly. Private adapter code registers a backend with `qrh.llm.register_backend(provider, fn)`, where
`fn(model_id, prompt, params) -> str`; **qrh ships no network backend**. `qrh.llm.StubBackend` is a deterministic fake for
tests and examples. Only models the hypothesis declares can be called.

- **Key.** `sha256` of the canonical JSON of `{model_id, params, prompt, dataset_hash, code_hash}`. `as_of` is recorded for
  audit and reported on a miss, but it is not part of the key. See [Cache binding and re-recording](#cache-binding-and-re-recording).
- **Storage.** `<QRH_HOME>/llm_cache/<trial_id>/entries.jsonl` (append-only, one line per answer with its entry hash) and
  `index.json`. The cache content hash is order independent. A cache whose entries do not match their own hashes blocks
  the run.
- **Modes**, set by the pipeline through a context variable (there is no environment variable): `replay` (the default; a
  miss raises `LLMCacheMiss` and the stage is BLOCKED with a "replay cache miss" message that names `qrh run <trial> --llm-mode record`; see
  below) and `record` (a miss calls the registered backend and stores the answer).
- **CLI.** `qrh run <trial> --llm-mode record|replay`; `qrh llm status <trial>` (entry count, cache hash, models; never a
  prompt); `qrh llm verify <trial>` (recomputes every entry hash; exit 2 on a mismatch).
- **Receipts.** S2 to S6 record `llm.cache_hash` (the hash of the entries that stage used: the same calls give the same hash
  in record and in replay mode) in their params; the mode and the call counts are diagnostics, which are never hashed.
  A replay run therefore gives the same receipt hashes as the recording run.
- **Truncation.** For an LLM strategy S2 runs the full data first (recording it in record mode, requiring every call to hit
  in replay mode), then executes each truncated run in replay mode. A cache **miss** inside a truncated run means the prompt
  for that decision differs from the one built from the full data, which proves the prompt used data from after the cutoff:
  S2 **FAIL** "prompt differs between full and truncated data at <decision time>". This is an additional detector in the
  same S2 record as the truncation result. Pass `as_of` so the decision time is reported.
- **Future perturbation.** The S2 future-perturbation test (see [methods.md](methods.md)) runs in replay mode for an LLM
  strategy whose full run made model calls: a prompt asked after the cutoff is answered with a deterministic stub (the last
  served answer), a prompt at or before the cutoff that is missing from the cache is a FAIL ("prompt differs"), like in the
  truncation test. If the full run made no model call (a declared but unused model) the test runs as for any other strategy.
  If it cannot run it counts as INCONCLUSIVE, never dropped. The same-bar execution check runs too.
- **Privacy.** Cache entries contain prompts, which may include licensed data. They live only in `QRH_HOME`, never in the
  repository, and must not be committed or published. `approval-guard` blocks an agent from writing under
  `<QRH_HOME>/llm_cache` (reads stay allowed), so an agent cannot fabricate cached answers. See [security.md](security.md).

### The S5 parameter stability check and the cache

For one pre-registered combination the S5 parameter stability check runs the neighbours of the chosen setting through the adapter
(see [methods.md](methods.md)). For a strategy that declares an `llm` those runs make model calls with changed prompts: in
`--llm-mode record` they are recorded (and cost model calls, one full development-window pass per neighbour); in replay a neighbour
that was never recorded is BLOCKED with the usual cache-miss hint. They are not trials and do not change the trial count. A grid
needs extra calls too whenever a step-distance neighbour is not one of its points (the neighbourhood is local, 10% and 20% around the
chosen setting); the grid's own points are scored from their stored returns. Varying an `llm_variant` across a grid's variants is
an ordinary categorical neighbour and costs nothing extra.

## Cache binding and re-recording

A recorded answer is tied to the exact data and code it was recorded with. The cache key covers the model, the
parameters and the prompt, plus the hash of the dataset the prompt was built from and the hash of the code (including the
prompt files) that built it. The decision time (`as_of`) is still not part of the key. If either hash changes, the old
answers no longer match, so a replay finds nothing, and this is intended: an answer recorded for other data or other code
must not support a new verdict.

- **Older caches miss.** Entries recorded before this binding existed do not match the new keys. They are not deleted;
  record the trial again.
- **A placeholder hash is refused.** A session opened outside the pipeline (for example in a unit test) uses an all-zero
  placeholder for both hashes. The pipeline and the benchmark refuse to record or replay under it, so an answer recorded
  under the placeholder is never served to a run. `qrh llm verify <trial>` reports any such stored entry as unbound.
- **A replay miss never calls the model.** The stage is BLOCKED and the message tells you to re-run with
  `--llm-mode record`, and warns that recording calls the model and may cost money. You decide whether to do it.

To re-record a trial:

```bash
qrh run <trial> --llm-mode record
```

Replay afterwards with a plain `qrh run <trial>`. Because each re-recording is another sample from the model, treat it as
described in the limits below.

## Examples (synthetic, with the stub backend)

```bash
export QRH_HOME=$(mktemp -d)
# 09: declared, the window starts after the cutoff plus the margin, replayable: S0-S7 PASS
uv run python examples/09_llm_clean/make_data.py $QRH_HOME/d09
uv run qrh data register $QRH_HOME/d09/prices.csv --name prices --rules examples/09_llm_clean/rules.yaml
T=$(uv run qrh new examples/09_llm_clean/hypothesis.yaml)
uv run qrh run $T --llm-mode record      # records the stub's answers, exit 0
uv run qrh run $T                        # replays them: identical receipt hashes
uv run qrh llm status $T
# 08: the model "knows the future" before its cutoff, which is after the window start: S2 FAIL from the gate
uv run python examples/08_llm_contamination/make_data.py $QRH_HOME/d08
uv run qrh data register $QRH_HOME/d08/prices.csv --name prices --rules examples/08_llm_contamination/rules.yaml
uv run qrh run $(uv run qrh new examples/08_llm_contamination/hypothesis.yaml)    # exit 2
```

Example 08 has two companions in the tests: with the cutoff set to `unknown` S2 is BLOCKED, and with the `llm` section removed
the adapter's `qrh.llm` import is an LLM001 block. Numbers 06 and 07 stay reserved (04 and 05 are existing examples).

## Benchmark

`qrh bench run --suite llm-gate` plants four classes of defect, each with several variants, next to clean LLM controls
(declared, window after the cutoff, replayable): `llm_cutoff_violation` (the window starts before the cutoff plus the
margin), `llm_cutoff_unknown`, `llm_undeclared`, `llm_prompt_lookahead` (a prompt built with a future value) and `llm_direct_client` (LLM002: a client in the
adapter module, or called from a helper module; a backend file that registers the provider is a control and must not be flagged).

<!-- bench:start -->
The receipt, text and PDF are in `examples/bench/llm_gate/` (bench id `bench-047471e36bf0`, profile `full`, seed 20240601, 30 replications per variant; no `chart.json` is kept there, `qrh bench report examples/bench/llm_gate/receipt.json --out DIR` re-renders it). Detection per class and false alarms on the clean controls:

| Class | Detected / false alarms | Flags by detector |
|---|---|---|
| `llm_cutoff_violation` (window starts before cutoff + margin; 9 variants) | 270/270 (100.0%, exact 95% interval 98.6 to 100.0%) | gate 270 |
| `llm_cutoff_unknown` (unknown cutoff or cutoff source; 5 variants) | 150/150 (100.0%, exact 95% interval 97.6 to 100.0%) | gate 150 |
| `llm_undeclared` (LLM client in the adapter, no `llm` declared; 13 variants) | 390/390 (100.0%, exact 95% interval 99.1 to 100.0%) | lint 390 |
| `llm_prompt_lookahead` (prompt built with a future value; 7 variants) | 210/210 (100.0%, exact 95% interval 98.3 to 100.0%) | lint 30, replay 210, truncation 210 |
| `llm_direct_client` (provider client outside qrh.llm.complete (LLM002, advisory); 2 variants) | 60/60 (100.0%, exact 95% interval 94.0 to 100.0%) | direct 60 |
| clean controls (9 variants), false alarms | 0/270 (0.0%, exact 95% interval 0.0 to 1.4%) | none flagged |
<!-- bench:end -->

These defects are the ones the rules are built to catch, so the rates show that the rules do what they say, not how often a
real leak is found: see the limits below. The methods are status `default` without the promotion flow: they are new checks
with no predecessor and do nothing for a strategy without `llm`.

## Limits, stated honestly

- **A cutoff is a provider claim.** The gate compares dates somebody wrote down. A documented cutoff can be wrong, and a
  model can know events after its stated cutoff (late data, fine-tuning, retrieval, leaked benchmarks). Nothing here
  measures what a model knows. Example 08 shows it: with a false claim (`training_cutoff` before the window) the same
  contaminated strategy passes the gate and the truncation test. The margin (default 180 days, readiness warns below 365)
  is a buffer, not a guarantee.
- **Fine-tuned and retrieval systems.** A model fine-tuned on, or prompted with, data from the test period no longer matches
  its declared cutoff. The retail-core interview asks about it; the harness cannot verify the answer. Live retrieval is
  accepted only with evidence text, which is a claim as well.
- **Prompt look-ahead is caught only where the truncation test samples** (three cutoffs), and only when the leaked value
  changes between the full-data prompt and the truncated-data prompt; a leak that is identical in both is invisible.
- **Prompts should name their decision time.** The cache key is the prompt text, so a leaked prompt that happens to be
  byte-identical to the prompt of another decision hits the cache and is not seen by the truncation replay. Put the
  decision date in every prompt (the examples do).
- **Calls that bypass the cache are detected only on a best-effort basis.** An adapter that declares `llm` but calls a provider
  directly instead of `qrh.llm.complete` is not prevented from doing so: LLM002 warns about a client import or call outside a
  file that registers a backend, but plain HTTP, subprocesses and dynamic imports evade it. Such calls are neither cached nor
  replayable, and the truncation prompt detector cannot see them.
- **Undeclared use is a static check.** A model reached through a plain HTTP call, a subprocess or a plug-in the lint does
  not know is not found. The declaration is the control; LLM001 is a tripwire.
- **The cache is private and unsigned.** Entry hashes detect corruption and accidental edits; someone who rewrites an entry
  and recomputes its hash is not detected by `qrh llm verify`. The controls are the approval guard (agent writes are
  blocked), the cache hash recorded in the S2 to S6 receipts, and your own file permissions. A replayed verdict says the
  strategy was evaluated on those recorded answers; it does not say a live model would answer the same today.
- **Recorded answers are one sample.** Replay makes a verdict reproducible, not representative: another recording of a model
  with a non-zero temperature can give different results, and each re-recording is another trial.

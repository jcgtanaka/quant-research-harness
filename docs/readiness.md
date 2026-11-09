# Deployment readiness

Before a strategy goes to paper or live trading there is **one document** that gathers every warning of the whole
workflow, and a human acknowledges it with a signed approval. Nothing here changes a verdict: the report is built from
verified receipts, stage artifacts, the registry and the interview state, and a failure to render it never touches a stage.

```
qrh readiness report <trial>     data JSON, chart JSON and a PDF under <QRH_HOME>/reports/<trial>/
qrh readiness status <trial>     ACKNOWLEDGED (exit 0) or PENDING with the reason (exit 1)
qrh readiness ack <trial>        the human step: needs a controlling terminal and the approval passphrase
```

S7 generates the report automatically into the trial's report manifest (`readiness.pdf`, `readiness.chart.json`,
`readiness.json`; `QRH_REPORTS=0` switches automatic rendering off, `qrh readiness report` still works).

## What is collected

Every warning has a stable id (the first 16 hex characters of the SHA-256 of source kind, code and scope), a stage, a
severity (`high`, `medium`, `low`), a source kind, a message, the hash and path of the receipt it comes from where there
is one, and a resolution hint with the exact `qrh` command when there is one.

| Source | Warning | Severity |
|---|---|---|
| interview | pending critical question | high |
| interview | pending recommended (non-critical) question | medium |
| interview | answered "don't know" (with the question's `unknown_risk`) | high if critical, else medium |
| interview | an answer that carries a risk tag (for example `lookahead`) | high |
| interview | an answer whose scope changed since it was given (stale) | high if critical, else medium |
| interview | the final verdict is older than the answers (the conditions differ from S7's) | high |
| interview | **stability thresholds not confirmed by the trader**: any of the four stability questions is unanswered, answered "don't know", or answered with a value that differs from the one frozen in the hypothesis ("keep the default" counts only when the frozen value is the default in force at registration; one warning for all four, at S5) | medium |
| exemption | the parameter stability check did not conclude (NOT APPLICABLE: no numeric parameter, or no meaningful measure) | medium |
| exemption | a numeric parameter could not be nudged (value 0, a failing neighbour), so its stability is unchecked | medium |
| exemption | non-numeric parameters that take one value are not varied by the stability check | low |
| exemption | the hypothesis declares its own stability neighbours or bounds (listed, so the trader can confirm they are real alternatives) | medium |
| data | a dataset flag from an interview answer | high |
| data | the dataset changed after S1 checked it, or is not PASS in the catalog now | high |
| data | gaps in the coverage, a failed rule, a rule that reported WARN | medium |
| onboarding | dataset not onboarded (no instrument model for its current file) | high |
| onboarding | the data contradicts a claim of the instrument model (the dataset is BLOCKED anyway) | high |
| onboarding | an unverified critical field (multiplier, adjustments, timestamp semantics, price units), linked to its question | high |
| onboarding | an unverified recommended field | medium |
| onboarding | a column name hints at adjusted prices (`adj_close`, split, dividend) while `adjustments` is unsettled | medium |
| lint | a WARN finding | medium |
| lint | every `# qrh: allow` suppression, with its recorded reason (an LLM001 suppression has its own warning, below) | medium |
| llm | `# qrh: allow LLM001`: an adapter that appears to use an LLM with no `llm` declaration, with the reason given | medium |
| llm | provider client used outside `qrh.llm.complete` (LLM002), listing the files; best effort | medium |
| llm | the cutoff margin is below 365 days (`llm_cutoff_margin_days`) | medium |
| llm | live retrieval or tools declared, quoting the evidence text (the harness cannot check it) | high |
| llm | results depend on a recorded cache: N recorded calls and the cache hash (a cache that does not verify is high) | low |
| exemption | no independent returns reconstruction concluded (the default same-bar check did not PASS or FAIL and `s2.pnl_reconstruction` was not selected), with the reason | medium |
| override | an override or hook approval recorded for the trial (hook, reason, date); a hook approval with no trial is listed if it was granted after the trial was registered | medium |
| method | an experimental or non-default method was used (receipt `non_default_methods`) | medium |
| pack | a loosened gate: an approved loosening of a core threshold, or a hypothesis looser than its packs | high |
| condition | each open condition of a PASS · CONDITIONAL | high |
| receipt | no final verdict, a final verdict that is not a pass, a receipt or artifact that does not verify | high |
| limit | a known limit of the harness that applies to this trial (static table in `qrh.readiness.model`, one line each, with a pointer to [methods.md](methods.md)) | low |

The known limits are: same-bar execution and percent-instead-of-fraction units are detected by the default returns
reconstruction only when it concludes (listed when it did not, or when `s2.pnl_reconstruction` was not selected); relationship corruption at or below the rule tolerance
cannot be detected by the data rules; detection rates describe the benchmark corpus, not all look-ahead. For a strategy that
declares `llm`, two more: a model's documented cutoff can be wrong, and models may know events after their stated cutoff (see
[llm.md](llm.md)).

A condition appears twice on purpose: once as the interview warning that explains how to answer it, and once as the open
condition on the final verdict.

## The report

Page 1: the trial, the final verdict (PASS · CONDITIONAL spelled out), a stage by severity count table and the
acknowledgement status. Next pages: the warnings grouped by stage, each with severity, message, the short receipt hash and
the resolution. Status is carried by a marker and a word (`▲ HIGH`, `◆ MEDIUM`, `● LOW`), never by color alone. Every
page footer has the trial id, the report hash and the research-only disclaimer. The output is deterministic: the same
trial state gives the same bytes.

The **report hash** covers the verdict, the counts, every warning (including the hash of its receipt) and the hash of
every stage receipt; it has no timestamps and no file locations. Any change in the warning set changes it, and so does
changed evidence (an edited adapter source, a changed dataset, a new answer). Receipts are deterministic: facts that differ
between runs of identical inputs (for example a truncation-cache hit) are stored in the receipt's `diagnostics` field,
which is never hashed, so an identical re-run gives identical receipts and the same report hash.

## The acknowledgement

`qrh readiness ack <trial>` is a human step built like `qrh approve`: it needs a controlling terminal (no stdin, no
variable, no option can replace it) and the approval passphrase. It recomputes the report, then displays, in the decision
banner format, the verdict, the counts and **every HIGH warning**, and only then asks for the passphrase.

```
##################
WARNING: DECISION TO BE MADE BY YOU!
##################
Content of the warning:
```

It signs, with the approval key and the distinct domain `readiness-ack` (canonical prefix `qrh/readiness-ack/v1`), the
payload `{v, trial_id, report_hash, warning_ids (sorted), high_count, key_id, created_at}`. The result is stored at
`<QRH_HOME>/readiness/<trial>/ack.json` (mode 0600) and verified with the public key only. It is valid only for the exact
evidence it was given for: any new or resolved warning, changed message, changed verdict or changed receipts invalidates it (an identical re-run does not), as
do a different key, a rotated key, an edited field, or a file copied from another trial. `qrh readiness status` says
which. `qrh.readiness.is_acknowledged(trial_id) -> (bool, reason)` is the same check for code.

A trial whose final verdict is not a pass can still be acknowledged (the banner says so and the verdict is part of what is
signed); a trial with no S7 receipt cannot. The human decides; the harness only records that they saw it.

## Guards

- `approval-guard` blocks an agent from running `qrh readiness ack` in every form it blocks `qrh approve` in (wrappers,
  `sh -c`, here-strings, aliases, pseudo-terminals, split words) and from importing `qrh.readiness.ack`, and it protects
  writes to `<QRH_HOME>/readiness/` like the other state directories. Reading and `qrh readiness status|report` stay allowed.
- `pre-restart-guard`: with `guard.require_readiness: <trial id>` in `config.yaml`, an engine start is blocked (banner,
  approvable only through the human `qrh approve` flow like every other guard block) unless the trial's report is
  acknowledged for its current warning set. With the key absent the check is not applied and session-context says nothing
  new. An invalid value in the config blocks (fail closed).

## Limits

- The acknowledgement says a person saw this warning set with the evidence of these receipts. It is not a statement that
  the strategy is sound.
- The text filter of the approval guard cannot stop a program that builds the path of `ack.json` at run time; what stops a
  forged acknowledgement is the signature (no passphrase, no valid signature). See [security.md](security.md).
- Warnings can only be as complete as the receipts and the registry: a source that cannot be read is itself a HIGH warning.

# Experience packs

An experience pack is a folder of what a trader or team already knows: general lessons, earlier attempts,
account facts, stricter thresholds, interview questions and extra lint rules. The harness stays generic; a pack
carries the experience. A pack is registered by path (never copied), and its content hash is recorded in every
receipt written while it is active (`packs: {name: {version, hash}}`). Editing a pack changes later receipts,
never old ones.

## Folder layout

```
my-pack/
  qrh-pack.yaml       manifest (required)
  lessons/*.yaml      lessons
  attempts/*.yaml     pre-harness history
  account.yaml        account facts
  thresholds.yaml     per-dimension thresholds
  data_rules/*.yaml   data-quality rule templates
  questions/*.yaml    interview banks (see interviews.md)
  lint/*.yaml         declarative extra lint rules
```

### Manifest (`qrh-pack.yaml`)

| Field | Meaning |
|---|---|
| `name` | letters, digits, `_ . -` |
| `version` | semantic version, for example `1.2.0` |
| `schema_version` | `1` |
| `visibility` | `public` or `private` |
| `description`, `authors`, `license` | free text |
| `applies_to` | optional lists: `asset_class`, `strategy_family`, `frequency`; empty means any |

### Content files

- **Lessons**: `id`, `statement`, `applies_to`, `trigger` (`keywords` matched in the hypothesis text, and/or `fields`
  matchers, substring or `re:` regex), `action` (`warn` or `require`), `requirement` (when `require`: exactly one of
  `interview_question`, `threshold_floor: {name, value}`, `method: "id=version"`), `severity` (`info`, `warning`,
  `critical`). Triggered lessons are listed in the S0 receipt and the markdown report. A `require` lesson that
  cannot be met refuses registration; an `interview_question` requirement makes that question critical.
- **Attempts**: earlier work done before the harness: `text`, `strategy_family`, `asset_class`, `frequency`,
  `variants_tried`, `verdict`, `date`, `notes`. They are imported into the registry marked `imported`, add their
  variant count to the family trial count used by the deflated Sharpe, and show up in `qrh search`.
- **Account**: `capital`, `base_currency`, `min_order_size` and `fee_schedule` per asset class, `margin_rules`,
  `caps` (any of the sanity bounds). Caps tighten the hypothesis bounds; the account is recorded in the S0 receipt.
- **Thresholds**: `all:` plus optional `asset_class:`, `strategy_family:`, `frequency:` sections keyed by value. Higher is
  tighter for `stability_closeness`, `stability_plateau_share` and `stability_neighbour_ratio` (like the `min_*` floors); a pack
  may raise them freely and lowering one below the core value needs a human approval. `stability_steps` is unordered. The trader
  is always asked these four ([interviews.md](interviews.md)); a pack value that is tighter than the trader's "keep the default"
  stays, and a trader value looser than a pack is recorded as `hypothesis_looser_than_pack`.
- **Lint**: `rules:` with `id`, `severity` (`BLOCK` or `WARN`), `message` and either `regex` or `ast`
  (`{call: name, keyword: {k: v}, negative_first_arg: true}` or `{import: module}`). They run at S2 next to the
  built-in look-ahead lint and use the same `# qrh: allow <rule> <reason>` suppression.

`applies_to` also takes `uses_llm: true` (lessons, questions and whole packs): it matches only hypotheses that declare an `llm`
section (see [llm.md](llm.md)). Pack lint rules with the id `LLM001` and `ast: {import: <module>}` add libraries to the
undeclared-LLM detector.

## Layering

Core defaults, then public packs, then private packs, then the hypothesis.

- A pack may **tighten** freely: raise a floor (`min_*`), lower a ceiling (`max_*`), add rules or questions, mark
  questions critical. A later pack cannot relax what an earlier one tightened.
- Any **loosening** of a core value is rejected unless a human approves it. `qrh pack add` prints an approval id;
  the human runs `qrh approve <id> --reason "..."` in their own terminal and repeats the command. The approved
  loosening is stored with the pack entry and recorded in every receipt that relied on it. Editing the pack to
  loosen more makes it fail to load.
- The hypothesis keeps its explicit values; where one is looser than the packs, the S0 receipt records it.

## Commands

```
qrh pack init <dir> [--visibility private|public]   scaffold a pack (private by default)
qrh pack add <path>                                 register path + content hash
qrh pack list                                       registered packs, flagging changed folders
qrh pack remove <name>                              unregister (folder untouched), drop imported attempts
qrh pack validate <path> [--config safety.yaml]     schema check; public packs also get the safety lint
qrh pack draft --from <file-or-dir> --out <dir>     draft lesson skeletons from notes
qrh pack approve-lesson <file>                      activate a draft (needs a human approval)
```

`validate` and the other commands exit 2 on findings or refusals.

### Public-safety lint

For `visibility: public` packs, `validate` flags ticker-like tokens, dollar and percent figures in lesson text,
ISO dates, account-number patterns, e-mail addresses and home paths. Findings show kind and location, never the
matched text. Extra ticker patterns and allowed upper-case words go in a YAML file (`ticker_patterns`, `allow`)
given with `--config` or `QRH_SAFETY_CONFIG`. When `QRH_DENYLIST` is set, `scripts/denylist_check.py` runs too.
It is a safety net for accidents, not proof of anonymity. Private packs are not linted.

### Drafting lessons from notes

`qrh pack draft` is deterministic. It cuts a notes file (or folder of `.md`/`.txt`) into one skeleton per heading
section, with `status: draft`, a `source_ref` hash and an excerpt of at most 300 characters, in a private folder
(never public). Drafts are not loaded. The agent refines the wording; a human then runs
`qrh pack approve-lesson <file>`, which waits for the same `qrh approve` flow, and only then does the lesson
become active.

## Shipped packs

- `packs/retail-core` (public): generic lessons (check the base rate of a regime detector, count every variant as
  a trial, attribute worst-day losses to the open position, prefer point-in-time data, verify derived columns,
  respect daily-bar sample size, match the live re-optimization cadence, sweep account sizes, check mark staleness
  on multi-leg positions) and the generic interview bank. No data, no numbers tied to any account.
- `packs/example-trader` (private, synthetic): a fake account, fake earlier attempts and one demo question, used
  by the tests and these docs.

```
qrh pack add packs/retail-core
qrh pack add packs/example-trader
```

## Writing your own private pack

1. `qrh pack init ~/my-pack` and edit `qrh-pack.yaml`.
2. Keep it outside this repository. Put real account facts, earlier attempts and specific lessons here.
3. Write lessons as general rules with trigger keywords; use `thresholds.yaml` to tighten what you want enforced.
4. `qrh pack validate ~/my-pack`, then `qrh pack add ~/my-pack`.
5. To turn notes into lessons: `qrh pack draft --from notes.md --out ~/my-pack`, refine, then approve each one.

See [interviews.md](interviews.md) for question banks and [architecture.md](architecture.md) for receipts.

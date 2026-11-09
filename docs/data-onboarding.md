# Dataset onboarding

The harness is only as good as its understanding of the historical data. A column read in the wrong unit, a futures
multiplier that is wrong, prices that are secretly back-adjusted, an interest-rate series built from today's curve: none
of these raise an error, and every verdict built on them is wrong. Onboarding makes the understanding explicit and
checks it:

- an LLM (the coding agent) **interprets** a dataset and writes a proposal;
- deterministic code **verifies** every claim it can;
- the human **confirms** what code cannot prove.

The agent never sees the rows of a private dataset. It sees a privacy-safe **profile**.

```bash
qrh data register prices.csv --name prices --rules rules.yaml    # --privacy public only for synthetic data
qrh data profile prices --json                                   # the agent's only view of the data
qrh data propose prices --file proposal.json                     # UNVERIFIED claims (source llm)
qrh data verify prices                                           # code checks; a contradiction BLOCKS the dataset
qrh data onboarding prices                                       # per field: claim, source, status, evidence; open questions
qrh interview next --stage S1 --scope dataset:prices             # what code could not prove is asked, one question at a time
qrh data status                                                  # the onboarding state is shown next to the catalog status
```

## The profile

`qrh data profile <name> [--json] [--no-values]` is deterministic, hashed, and stored per dataset content hash (and mode)
in the registry. It is rendered as the S1 **dataset card** page. It contains aggregates only:

- row count; the time index (timezone, frequency inferred by observed density, gaps, duplicates, monotonic, first and last
  timestamp);
- per column: dtype, null rate, distinct count (capped at 1000), monotonic, integer-valued, non-negative and
  strictly-positive flags, the number of single-bar jumps above ln(1.9) for positive series, and the minimum, maximum and
  quantiles (1, 25, 50, 75, 99 percent);
- text columns: the distinct count and a **shape** distribution (digits become 9, upper-case letters A, lower-case letters
  a, other characters are kept; the top five shapes with their share). The values themselves are shown only for a column
  of at most six distinct values whose WHOLE set (case-insensitive) lies inside one flag vocabulary: {C,P}, {CALL,PUT},
  {Y,N}, {YES,NO}, {B,S}, {BUY,SELL}, {T,F}, {TRUE,FALSE}, {L,S}, {LONG,SHORT}, {BID,ASK} or {0,1}, **and** whose name is
  not an identifier name (symbol, ticker, contract, id, ...). Anything else shows its distinct count and shapes only;
- candidate relationships found from the numbers: `A >= B` on every row (high/low, ask/bid), a value always inside
  `[low, high]`, `A` within a relative tolerance of `B`, a column that looks like a return series, a percent (0 to 100) or
  a rate. Name-vocabulary hints (strike, expiry, right, delta, iv, bid, ask, rate, volume, open interest, funding,
  contract, multiplier, settle, adjustments) are reported as **hints, never facts**.

`--no-values` also drops every aggregate value (minimum, maximum, quantiles, first and last timestamp, flag values and
the scale of a return-like column). Nothing in a profile is a row-level value.

## The instrument model

An `InstrumentModel` is a set of **claims** (`qrh.onboarding.model`, pydantic, strict, unknown keys rejected). Each claim
has a value, a `source` (`profile`, `llm` or `user`), an `evidence` string (at most 300 characters, plain text), a
`status` (`unverified`, `verified`, `contradicted`, `confirmed`) and `checked_by`.

| Field | Values | Critical |
|---|---|---|
| `asset_class` | equity, etf, fx_spot, crypto_spot, future, option, crypto_perp, index, rate, macro, other | |
| `price_columns` | 1 to 12 column names | |
| `price_units` | fraction, percent, currency, index_points, rate_percent, other | yes |
| `quote_currency` | 2 to 10 letters or digits | |
| `timestamp_semantics` | bar_open, bar_close, event_time, publication_time, unknown | yes |
| `timezone` | a zone name or offset (`UTC`, `America/New_York`, `+02:00`) | |
| `session_hint` | free text, at most 200 characters | |
| `adjustments` | none, split_only, split_and_dividend, back_adjusted, forward_adjusted, unknown | yes |
| `multiplier` | a positive number or `unknown` (required unless the class is equity, etf, fx_spot, crypto_spot, index, rate or macro) | yes |
| `tick_size` | a positive number or `unknown` | |
| `frequency` | `1D`, `1B`, `1h`, `5min`, `daily`, ... | |
| `roll_rule` | free text, and a method id if known | |
| `option_mapping` | columns for `underlying`, `strike`, `expiry`, `right`, `mid` or `bid`/`ask`, `rate` | |

plus `extras`, a small dictionary of plain text. A model belongs to a dataset **name and content hash**. A changed file
creates a new onboarding version (the claims carry over, nothing stays verified or confirmed); changed claims create a new
version; an old version is never rewritten, so a verdict recorded against it stays tied to what it saw. The catalog keeps
them in `onboarding_versions` (migration 10); the existing catalog behaviour and rules files are unchanged.

## Proposing

```json
{"source": "llm",
 "fields": {"asset_class": {"value": "future", "evidence": "contract column has shape AAA9"},
            "price_columns": {"value": ["open", "high", "low", "close"]},
            "multiplier": {"value": 50}},
 "extras": {}}
```

`qrh data propose <name> --file proposal.json` stores the claims as `llm` and `unverified`. The file is **untrusted
input**: JSON only, at most 64 KB, duplicate keys and `NaN` refused, unknown keys refused (a claim cannot set its own
status or source), every string checked for control and invisible characters and the `$` sign and cut at its limit, and
nothing is ever interpreted as markup (the PDF pages draw it as plain text). A proposal replaces the previous claims; an
identical one creates no new version, and a human-confirmed claim whose value did not change stays confirmed.
`--as-user` (and `"source": "user"` in the file, both required together) records the user's own claims; **the agent may not
use it** (approval-guard blocks it, like `qrh data confirm` and `qrh interview answer --human`).

## Verifying

`qrh data verify <name>` runs deterministic checks and records `verified`, `contradicted` or `unverified` with an evidence
string for each field:

| Field | Check |
|---|---|
| `price_columns` | the columns exist, are numeric and finite, positive when prices are currency or index points (or the class is a priced one); OHLC (`high >= low`, `high >= open/close`, `low <= open/close`) and bid/ask (`ask >= bid`) hold, a violation in more than 0.5% of rows contradicts |
| `price_units` | magnitudes: fraction (median absolute value below 0.5, maximum below 5), percent (maximum absolute value at most 100 and 99th percentile at least 0.5, **and** a column name with a rate/pct/percent/yield/return hint or a return-like shape: both signs, small median against the maximum, symmetric around zero; a price level in range stays unverified and is asked), rate_percent (median between 0.05 and 30, maximum at most 100), currency and index points (positive and within a factor 1000 of the median) |
| `adjustments` | `none` is contradicted when a price shows a single-bar jump above ln(1.9) with no volume spike (3 times the median volume) on that bar, as a split does; `back_adjusted` and `forward_adjusted` are contradicted when the price moves at least 3 times more than usual at contract changes (the level shifts of an unadjusted splice); everything else stays unverified |
| `frequency` | claimed against the frequency inferred from the timestamps (`1D` also matches business-day data) |
| `timezone` | UTC-offset consistency only: the offsets written in the file (or in a Parquet time zone) against the offsets of the claimed zone; a file whose timestamps carry no offset stays unverified |
| `asset_class` | an option needs an `option_mapping` with strike, expiry and right columns that exist. Any other claim is contradicted when the data carries that option signature, and a spot-like claim (equity, etf, fx_spot, crypto_spot, index) is contradicted by non-positive prices, a contract-code column or a funding column. A future is verified only with a contract-code column; an expiry-named column alone stays unverified. A perpetual is verified with a numeric funding column. Equity and etf verify with a non-negative volume column, fx_spot with a bid/ask pair. crypto_spot and index never verify, because nothing in the profile proves them |
| `option_mapping` | mapped columns exist; strike and underlying positive and numeric; expiry parses as dates; at most 3 right values; ask not below bid; incomplete mappings stay unverified |
| `timestamp_semantics`, `multiplier`, `quote_currency`, `session_hint`, `tick_size`, `roll_rule` | never verified from data: they stay unverified until the human confirms |

A contradiction sets the dataset's catalog status to **BLOCKED** with the reasons (the same mechanism as a failed quality
rule; `qrh data check` and S1 keep it blocked while the contradiction stands). Fixing the claim and verifying again lifts
it. Verification replaces the proposer's evidence with the checker's.

## Questions, answers and the state

Every unverified field generates an interview question: scope `dataset:<name>`, stage S1, id `onb-<field>`, **critical** for
`multiplier`, `adjustments`, `timestamp_semantics` and `price_units`, recommended for the rest. A required field that was not
proposed is a gap too and is asked as an open question. Answer with the usual provenance (`qrh interview answer onb-adjustments
--scope dataset:prices --value confirm --transcript ... --session ...`, or a person with `--human`): `confirm` confirms the
claim; another value corrects it (a new version, verified again; a correction that code cannot check stands as confirmed);
`unknown` leaves it open. An answer is valid for one claim of one file: when either changes, it is stale and the question is
asked again. `qrh data confirm <name> <field> [--value ...]` is the same step for a person at a terminal.

`qrh data onboarding` reports one state:

| State | Meaning |
|---|---|
| NOT_ONBOARDED | no model for the dataset's current file |
| PROPOSED | claims stored, not verified yet |
| VERIFIED_WITH_GAPS | verified, no contradiction, but a critical field is neither verified nor confirmed |
| ONBOARDED | no contradiction, and every critical field verified or confirmed |
| CONTRADICTED | the data contradicts a claim |

## Policy

The policy is warn-only, like every interview: an incomplete onboarding never blocks a stage by itself. A contradiction
does (the dataset is BLOCKED and S1 refuses it). S1 records the onboarding state, version and model hash per dataset in its
receipt (hashed) and writes the dataset card and an onboarding table page per profiled or onboarded dataset (status as text
plus a marker, never colour alone) with their `chart.json`. Pending critical questions turn a passing S7 into PASS ·
CONDITIONAL, as for every critical question. The readiness report adds: dataset not onboarded (high), contradicted (high),
unverified critical field (high, linked to its question), unverified recommended field (medium), and an `adjustments` name
hint in the profile while the field is unsettled (medium).

## Privacy and the limits

- The profile is the only view of a dataset meant for the agent; the skill tells it never to open, head, cat or print the
  rows of a registered file. The catalog flag `privacy: private|public` (default private; `--privacy public` for synthetic
  data) is stored and shown by `qrh data status`. The data-guard hook blocks direct reads of a private dataset (see hooks.md, Data guard); a script that prints rows is not caught.
- Column names are shown (the agent needs them); a file whose column names are symbols is not protected by the profile.
  Text values are shown only for a column whose whole value set is one flag vocabulary (see above) and that is not named
  like an identifier. Residual risk: a column whose values are exactly {C,P}, {B,S}, {T,F} or {L,S} may be two tickers
  and would be shown; use `--no-values`, or do not register such a file as an unflagged column.
- Code verifies structure and magnitudes. A series that is both positive and plausibly scaled can still be in the wrong
  units (a price level named like a rate passes as `percent`), a missing split is not proof of adjustment, and nothing in the data can
  show whether a timestamp marks a bar open or close. Those are exactly the questions asked of the human.

## Benchmark (optional)

```bash
qrh bench run --suite onboarding
```

The suite checks the verifier against planted mistakes on synthetic datasets (equity, futures, fx, option, perpetual).
It plants five classes of wrong claim: `wrong_units`, `wrong_asset_class`, `adjustments_none_on_split`,
`missing_option_mapping` and `wrong_frequency`. Per class it reports detection (the claim is contradicted or left
unverified) with the breakdown by status, and it also runs the correct proposal for each of the five datasets as controls.
The control gate is **false alarms = 0**: a correct proposal must never be contradicted.

The suite is **optional**: it is not part of the default bench run. It must be requested by name.

Wrong `asset_class` claims are now caught by the rules in the table above. Two limits remain. A claim of etf on equity data
verifies on the same evidence, because the profile cannot tell them apart. A claim of index or crypto_spot never verifies,
so a correct one is left unverified rather than confirmed.

## Synthetic datasets

`qrh.synthetic.instruments` generates small seeded datasets with their correct proposal: `equity_daily` (optionally with an
unadjusted split), `futures_daily` (contract codes, roll gaps), `fx_quotes` (bid/ask), `option_snapshots` and `crypto_perp`.
The tests assert that the correct proposals verify and that planted wrong ones (wrong units, wrong asset class,
`adjustments: none` on a split series, a missing option mapping, a wrong frequency) are contradicted or left unverified.

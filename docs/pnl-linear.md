# The pnl.linear check

**Status: partly built (see sections 9 and 10); not approved for real use.** This note describes a planned check. Nothing here is implemented, and every number below is a synthetic example.

## 1. The question

Does the profit a strategy reports match what its positions actually earned?

A strategy holds positions, and prices move. From those two facts alone you can work out how much money the strategy should have made. If the reported profit is different, something is wrong: a bug, a unit mix-up, a price used too early, or a result that was tuned to look better.

The check covers stocks, ETFs, spot currencies and spot crypto. It is meant to replace the narrower reconstruction check that exists today.

## 2. A worked example

An account holds 10,000 in one stock. The price rises 1%.

- Expected profit from positions and prices: 100.
- Profit reported by the strategy: 150.
- Unexplained: 50.

Costs can only lower profit, never raise it. So a reported figure above the expected one cannot be explained by fees. The 50 is a finding. The check measures it and compares it with limits that were agreed before the run. A small leftover inside the limits is tolerated. A leftover outside them is a FAIL, and the report shows the figure.

## 3. The steps

1. **Rebuild the expected profit.** Take the positions held and the prices of the instruments actually held, with the declared delay between signal and trade.
2. **Subtract declared costs.** Use the cost floor for the asset class, plus any declared income such as dividends or swap and financing charges.
3. **Measure what is left.** This is the unexplained remainder.
4. **Check it against three limits.**
   - An absolute yearly limit on the unexplained amount.
   - A limit on its share of the net profit.
   - A test for steady drift: a small gain that shows up every bar adds up over time, even when each bar looks harmless. It is one-sided (only unexplained gains count) and uses a 5% significance level.

   The first two limits are scaled to the noise measured in each strategy's own remainder. Each limit is 3 times the typical yearly drift that honest noise alone would produce (the factor 3 is a named constant, `NOISE_SCALE_K`). A noisy strategy therefore gets wider limits and a quiet one gets tighter limits. You may still override a limit with your own number. The earlier fixed values of 25 basis points and 2% were provisional and are no longer defaults.
5. **Check that prices were fresh.** Each price must be recent compared with the bar it is used for. Repeated values alone do not count as stale. Only the gap in time does. A price dated after its bar is flagged.
6. **Check that no future prices were used.** The strategy must not trade on a price it could not have known yet.

## 4. The verdicts

- **PASS:** reported profit is explained by positions, prices and costs, within the limits.
- **FAIL:** the unexplained amount is too large, drifts steadily, or the prices were stale or from the future. Do not trust the result until fixed.
- **INCONCLUSIVE:** the check ran but cannot decide, for example when the net profit is zero so a share is meaningless. This is never counted as a pass.
- **BLOCKED:** the check could not run, for example because a required answer or a price column is missing. Fix the setup and run again.

## 5. What you must provide

Each item is a **pending decision**. Nothing is defaulted silently.

- Cost floors per asset class (equity, ETF, spot FX, spot crypto). Pending decision.
- The caps: the yearly limit and the share limit. By default they are scaled to the measured noise (see step 4). You may override either with a number. Pending decision.
- The adjustment type of each price: none, split-adjusted or total-return. Pending decision.
- The bar type (for example daily or intraday bars). Pending decision.
- Where the data came from, and when each price was published. Pending decision.

If a required answer is missing, the check returns BLOCKED. If a cap is pending (set to None), the amount is still measured and reported, but it is not judged, and the verdict is INCONCLUSIVE.

## 6. The limits of the first version

- Futures and options are not included. They need their own models, planned separately.
- The caps are noise-scaled starting values. A bench run on synthetic strategies with known true profit (100 seeds, all three kinds) gave these results:

  | Noise per bar | Honest strategies wrongly flagged | A hidden gain of 1 bp per bar is caught |
  |---|---|---|
  | 0 bp | 0.000 | not reported |
  | 1 bp | 0.040 | 100% |
  | 5 bp | 0.073 | 94% |
- An unknown answer produces a warning in the final PDF. It is never a silent pass.
- The check does not look at the signal itself. A signal that peeks ahead but trades with a correct delay is a matter for other checks.

### Known limits

- At 5 bp of noise per bar, about 7 in 100 honest strategies are flagged. The main cause is the drift test's 5% significance level, which by design flags about 5% of honest noise.
- A hidden gain of about 1 bp per bar is caught every time at 1 bp noise, but only 94% of the time at 5 bp noise. More noise hides more.

## 7. Open design questions

1. **Cost floors.** Should we start with one conservative floor per asset class, which you can lower later with evidence? *Recommended: yes, conservative, per asset class.*
2. **Yearly absolute cap.** Should the first cap be set by bench tests on known synthetic strategies, rather than guessed now? *Recommended: yes, measure first.*
3. **Share cap.** Should the share limit apply only when net profit is clearly above zero, and otherwise return INCONCLUSIVE? *Recommended: yes.*
4. **Freshness limit.** Should the maximum price age be tied to the bar type, so one setting does not fit all? *Recommended: yes, one limit per bar type.*
5. **Unknown answers.** Should a missing price adjustment type or data source give BLOCKED, with a PDF warning if the user overrides it? *Recommended: BLOCKED by default.*

## 8. Two tracks: who built the backtest

**Status: Design, not yet built or approved.**

The check needs facts about positions, prices and costs. Where those facts come from depends on who built the backtest.

**Track 1: the harness built it.** The harness knows the positions, costs and prices because it generated the code itself. Anything it cannot settle becomes a warning in the PDF report and the dashboard, with the reason.

**Track 2: you brought the backtest.** The assistant asks the questions below one at a time. Every question has an "I don't know" option.

1. Where did the prices come from (broker, data vendor, own file)?
2. Are prices adjusted for splits and dividends (none, split only, total return)?
3. At what time is each price taken (bar close, next open, other)?
4. Are the bars time, dollar or volume bars?
5. Are trade sizes rounded (whole shares or contracts, or fractions)?
6. How are costs charged (per trade, per share, a percentage, none)?
7. Is interest on cash or margin included in the returns?
8. Are borrowing costs for short positions included?
9. Are dividends included in the returns?
10. Were trading halts or stale prices handled, and how?

**How answers are judged**

- An answer that settles the point gives PASS for that check.
- A missing answer, or "I don't know", gives WARNING and names the question.
- "Not applicable" gives PASS only if the data confirms it. Otherwise it gives WARNING.
- Anything the harness can check from the data is checked whatever the answer was. If the answer and the data contradict each other, the result is FAIL.

After the questions, the unexplained remainder is measured, and the cap applies only to that remainder. What the answers already explain is not counted against it.

Track 1 uses the same checks. It reads the answers from its own generated code instead of asking, and shows its warnings in the PDF and the dashboard.

## 9. Futures, first slice (F1)

**Status: First slice built and benched once, with the numbers in "Measured weakness" below; not approved for real use.**

Section 6 says futures need their own model. This is the first small slice of it.

**Scope.** One instrument, a fixed number of whole contracts, daily bars.

**How profit is worked out.** A futures contract is worth a fixed amount of money per point of price, called the multiplier. For each day:

profit = contracts held at the start of the day x price change x multiplier

Returns are shown as a fraction of the declared starting capital. Percent-change formulas are not used for futures, because the money at risk is not the price of the contract.

**What you must declare.** Each missing item blocks the check.

- The multiplier.
- The starting capital.
- The cost per contract.
- The roll rule, with a machine-readable method (calendar, volume or open interest). This applies to an unadjusted series.
- The adjustment type: none, split, total return, back-adjusted or forward-adjusted.

**Margin.** Margin is the deposit the broker holds against each contract. The check tests whether the declared margin per contract is feasible and reports it. A missing margin gives only a warning on that check. Margin is never counted as profit or loss.

**Rolls.** Contracts expire, so a position is moved to the next contract. For costs, a roll counts as two trades: closing the old contract and opening the new one.

**Expiry without a roll.** This is blocked. The check never fills the old price forward, because that would invent profit or loss on a contract that no longer trades.

**Currency.** The contract currency must equal the account currency. Otherwise the check is blocked, because currency conversion is not supported yet.

**Not included yet.**

- Spreads.
- Part contracts.
- Sizing from account value.
- Options.
- Currency conversion.

### Slope limit

The slope limit regresses reported profit on explained profit. An honest run should give a slope near 1. A scale error, such as a wrong multiplier, moves the slope away from 1. The check fails when the slope is more than K standard errors from 1. It uses the same design constant K = 3 as the other limits, and a Newey-West standard error with the same lag as the drift test.

### Measured weakness

The bench used 100 seeds on the futures sweep.

- At 0 bp noise, honest runs fail 1 time in 100, and that failure comes from an existing bound.
- A half-scale multiplier error fails on 100 of 100 seeds at both 0 and 1 bp noise, all from the slope limit.
- A hidden gain of 1 bp per bar fails on 100 of 100 seeds at 0 bp noise. At 1 bp noise it also fails on 100 of 100, of which 3 come from the slope limit.
- At 1 bp noise, honest runs fail 9 times in 100. Three come from the slope limit. The rest come from the existing bounds.

Known weakness: the honest false-alarm rate at 1 bp noise is about 9%, and the existing bounds contribute most of it. Under the defaults, a 1 bp hidden gain is not reliably separated from noise at 5 bp.

### Options, first slice (O1): decisions

**Status: not approved for real use.**

- Price types: the user's own declared price type is used first, with free data as a fallback. Allowed types are mid, settlement and last, and last needs a trade timestamp.
- Expiry: a held contract is closed at intrinsic value and reported as cash-settled. The check never fills a price forward.
- Expiry check: the mark on the expiry bar must match intrinsic value exactly, plus one currency rounding unit.
- Stale held price: a warning, not a block. The warning is shown in the report and the result is marked as conditional.
- Cost per contract: the user declares it, or the LLM looks up the broker's commissions online when the user has chosen a broker (the lookup is outside this build). If it is missing, the cost sub-check is blocked and the cost is never defaulted.
- Series: a full chain is accepted. The harness selects the single contract by declared strike, expiry and right, keeps one row per bar, and rejects any remaining ambiguity.
- Point-in-time: if the user does not know, the item becomes an unknown answer and a warning, not a block.

## 10. Design decisions

- **Scope:** the harness supports one strategy, or one portfolio of assets, followed through the whole workflow. A second strategy starts a new session.
- **Verdicts are recommendations.** PASS, FAIL, warnings, metrics, plots and the PDF report inform the user. A FAIL never blocks, removes or stops a strategy. Only the user decides whether to use it for paper or live trading.
- **Broker commissions:** take the figure from the broker's own commission page as last updated. No source-quality ranking.
- **Bench:** a test of the harness's own checks, not a ranking of strategies. Its PASS or FAIL is information for the user.
- Point-in-time not confirmed gives a warning, not a block.
- Option price type and rounding unit are inferred from the data where possible; otherwise the harness asks the user with a warning.
- **Adjustment:** for a backtest the harness builds, it already knows its data's adjustment type. Adjusted data is used as is; unadjusted data is first converted to adjusted data by the builder, not by a question. For a backtest you bring, the adjustment type is asked (Track 2 question) and an unknown answer is a warning. The receipt records the adjustment type for every run.
- **Truncation:** the truncation test is required for a backtest you bring: it must pass before the profit check can PASS. For a backtest the harness builds it runs only as a silent self-check; a failure is shown as a bug warning and does not block the profit check.
- **Cap:** for a backtest the harness builds, any unexplained difference is a bug, reported as a warning with the day and the amount. The noise-scaled tolerance applies only to comparisons with broker records (live results).

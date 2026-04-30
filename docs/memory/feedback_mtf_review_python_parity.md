# MTF strategy review + why we run logic through Python

**Captured: 2026-04-27.** During session review of `MyCustomStrategy.cs`
(`MultiTimeframeRunner_v1.0.2`) and the user-shared backtest CSV.

## User stance (recorded verbatim, 2026-04-27)

> "i dont care about edge i care about moneys"

This is a pragmatic frame. Translated into measurable terms for this project:

- **Goal**: maximize $/day in the account.
- **Constraint**: $/day must hold across as many market regimes as possible,
  because regime changes happen and a strategy that only made money in one
  regime stops making money when the regime flips.
- **What "edge" was tracking**: the ability to make money INDEPENDENT of
  drift. Same goal as "money in account" once you require regime stability.
- **No edge required if drift is reliable**: if MNQ drifts up forever, a
  long-only strategy is fine. The risk is ONLY regime change. The honest
  question for ANY strategy is "what does it do when the drift reverses?"

## The MTF backtest CSV finding

`examples/multy timeframe.csv` covers 37 trades on MNQ JUN26 from
2026-03-19 → 2026-04-24 (≈36 calendar days). Strategy column says
`"Sample multi-timeframe"` = NT8's BUILT-IN stock sample, NOT the user's
custom `MyCustomStrategy.cs`.

| Metric | Value |
|---|---:|
| Total profit | +$3,325.70 |
| $/day | +$107.28 |
| Trades | 37 (16 wins, 43.2% WR) |
| Trade direction | **100% long, 0 shorts** |
| "Exit on session close" exits | 15/37 (40.5%) — passive timeout |
| Buy-and-hold MNQ over same window | **+$6,667 (+$215/day)** |
| Strategy minus buy-and-hold | **−$3,341 (−$108/day)** |

**Honest read**: the strategy makes real money in dollars-per-day terms BUT
loses to single-contract buy-and-hold by $108/day. The $107/day is partial
drift capture, not active alpha. Profit dominated by 6 outlier winners
(+$4,875 from top 6 trades; remaining 31 trades net −$1,549).

This satisfies "money in account" criterion in a single regime. It fails
"money in account ACROSS regimes" because:
- All 37 trades are long. Strategy literally cannot harvest a downtrend.
- 36-day window is too short to catch a regime change.
- Most exits are passive (session close), not skill-driven.

If MNQ enters a sideways or falling regime, this strategy is expected to
return ~zero or lose money.

## Why run logic through Python (the parity case)

**Python parity matters for the money-in-account goal**, even when "edge"
isn't the goal. Five concrete reasons:

### 1. Test on 12 months of historical data without re-running NT8

We have `DATA/ATLAS/` (Databento) covering Jan-Dec 2025 — 12 months of all
TFs, which spans multiple market regimes (Q1 trend, Q2 chop, Q3 squeeze,
Q4 reversal). NT8 Strategy Analyzer can run on this too, but each backtest
takes minutes-to-tens-of-minutes to load and execute. Python sim can run
the same logic on the entire 12 months in seconds-to-minutes once.

→ **Money question answered**: "what would this have made over 12 months,
including non-uptrend periods?" If the answer on Q3 2025 (chop) is −$X/day,
that's information that prevents real-money loss on a future chop month.

### 2. Catch fill-timing bugs that distort PnL by 10-50%

NT8 Strategy Analyzer fills "next bar open" by default with
`Calculate.OnBarClose`. Python sims often fill at "current bar close".
That's a one-tick to multi-tick per-trade slippage difference.

If Python predicts +$X but NT8 backtest gives +$Y, the gap is fill timing,
not strategy logic. Without parity, every Python prediction has unknown
real-world drift. With parity, Python is a fast NT8 oracle.

→ **Money question answered**: "is the Python prediction trustworthy
enough to skip the slow NT8 backtest?" Once parity is verified, yes. Saves
hours per iteration.

### 3. Identify trades that should NOT have happened

Python sim has unlimited debugging visibility. NT8 doesn't show per-bar
internal state, only the trade ledger. Python can dump:
- The exact bar where the entry signal fired
- The feature values at that bar
- Why the exit fired
- What the position state was

→ **Money question**: many of the strategy's losses likely come from
specific entry conditions (e.g., entering deep into a 1m extension that
immediately reverses). With Python visibility, you can identify the bad
entries and add a filter to skip them. NT8 alone can't show you this.

### 4. Cheap parameter sweeps

Python can sweep `R`, `HardStopLossPoints`, `MaxNegativeBars`, etc. across
hundreds of combinations in minutes. NT8 Strategy Analyzer Optimizer
takes hours for the same sweep. Python finds the parameter sweet spot;
NT8 confirms.

→ **Money question**: "what's the best parameter set?" Found 10× faster
in Python.

### 5. Catch lookahead bias and bar-aggregation bugs

The 2026-04-17 lookahead audit turned a +$740/day baseline into −$164/day.
Python parity testing is the natural place to find these bugs because
you can manipulate the bar history mid-test (block out future bars
explicitly) and confirm the strategy still produces the same results.

NT8 Strategy Analyzer also avoids lookahead but it's a black box — you
can't poke at intermediate state to confirm.

→ **Money question**: "is the +$X PnL real, or am I peeking at future bars?"
Only Python parity testing answers this cleanly.

## The action plan (parity-driven validation)

To make money decisions on `MyCustomStrategy.cs`
(`MultiTimeframeRunner_v1.0.2`) we need:

1. **Run it in NT8 Strategy Analyzer** on 2026-03-19 → 2026-04-24 (the same
   window as the stock-sample CSV). Export trades CSV.
2. **Build Python sim** that mirrors `MultiTimeframeRunner_v1.0.2` exactly:
   same SMA crossover entries, same DRM, same StagnationMonitor.
3. **Parity-check** Python sim vs NT8 trades CSV. Same trades fire?
   Same exits? Same PnL?
4. **Once parity holds**: run Python sim on 12 months of `DATA/ATLAS/` to
   see what the strategy does across regimes Q1-Q4 2025.
5. **Then on `DATA/ATLAS_OOS/`** for an additional independent OOS check.
6. **Then on `DATA/ATLAS_NT8/`** as the OOS-2 (NT8-feed) gate.
7. **Money decision**: $/day across all those windows. If positive in
   most regimes, it's a money-maker. If only positive in Q1+Q2 2026,
   it's a drift-capturer that fails on regime change.

## What NOT to do

- Do NOT deploy `MyCustomStrategy.cs` to live based on a 36-day rising-
  market backtest. Money math says strategy underperforms passive long
  in that window; sustainable money math is unverified.
- Do NOT trust Python predictions until parity is verified — see today's
  v1.5-RC episode where Python predicted +$5K and NT8 backtest gave −$1K.
- Do NOT confuse `SampleMultiTimeFrame` (NT8 stock, used in the CSV) with
  `MultiTimeframeRunner_v1.0.2` (the user's custom code). Different
  strategies. The CSV evidence is for the STOCK sample, not the custom one.

## Related memory

- `memory/feedback_oos2_designation.md` — two-OOS validation gate
- `memory/feedback_lookahead_audit.md` — lookahead history that broke
  prior baselines
- `memory/feedback_phantom_spikes.md` — feed-dependent fake edge case
- `memory/feedback_data_validation_first.md` — verify data integrity
  before trusting strategy results
- `reports/findings/2026-04-27_v15_backtest_reality.md` — recent example
  where Python prediction missed NT8 reality by ~$6K

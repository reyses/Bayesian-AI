# Direction-classifier as a clean live strategy — OOS 2026 RESULT: FAILS

## Setup

User pivot to KPI-driven autonomous mode (2026-05-16, "go into autonomous mode the main kpi is daily 100 DLLs, low mae and high positive pnl days"). Goal: take the direction classifier (V2 entry features → P(LONG), AUC 0.864 IS) and run it through the existing `training_iso_v2/` ticker+engine pipeline with clean exits (TP + SL + TimeStop), validate against the 3 KPIs on the 2026 OOS sample.

Pipeline built:
- `tools/fit_direction_classifier.py` NEW — pickles scaler + LR on full IS daisy
- `training_iso_v2/strategies/direction_classifier.py` NEW — strategy class loading the pickle, firing on configurable cadence when `|P − 0.5| > T − 0.5`
- `training_iso_v2/exits_tick_exact.py` NEW — TickExactTP + TickExactSL using 5s bar OHLC high/low to detect intrabar threshold crossings, close at exact threshold price (with optional slippage)
- `training_iso_v2/engine.py` PATCH — honor `position.extras['_force_exit_price']` so the engine closes at the tick-exact price instead of bar close
- `training_iso_v2/ledger.py` PATCH — track `trough_pnl` (MAE) on each ClosedTrade
- `tools/dir_clf_kpi_search.py` NEW — autonomous grid driver, composite KPI score

## Result

### First grid (bar-close exits — the optimistic / wrong way)

| Config | $/day NET | DayWR | $/trade NET |
|---|---|---|---|
| T=0.85, 15m, TP=60, SL=40 | **+$787** | 37.7% | +$62.84 |
| T=0.85, 15m, TP=20, SL=20 | +$750 | 36.1% | +$56.89 |

Looks great — but $62/trade NET on TP=$60 means the engine is closing at bar close after intrabar overshoots beyond TP. **This is a backtest artifact: real execution with limit TP orders would only fill at TP, not above.**

### Second grid (tick-exact exits — honest)

TickExactTP/SL close at exactly TP/SL price using 5s bar OHLC high/low, with 1 tick slippage on SL.

ALL configs lose money:

| Config | $/day NET | CI | DayWR | $/trade |
|---|---|---|---|---|
| T=0.85, 15m, TP=20, SL=20 | −$34 | — | 37.7% | −$2.59 |
| T=0.85, 15m, TP=60, SL=40 | −$31 | — | 37.7% | −$2.44 |
| T=0.85, 15m, TP=30, SL=20 | −$42 | — | 24.6% | −$3.16 |

Why direction acc 87% doesn't translate: at T=0.85, **TP-hit rate is only 50-55%, SL-hit rate is 45-65%**. Direction-correct ≠ +$20 favorable move within 30 min.

### Third grid (asymmetric R/R, higher thresholds)

TP/SL pairs: (10,5), (15,10), (20,10), (30,10), (40,10), (20,5).
Thresholds: 0.85, 0.90, 0.95.

**ONE marginally positive config out of 36**:

| Config | $/day NET | 95% CI | DayWR | n/day | MAE |
|---|---|---|---|---|---|
| T=0.95, 15m, TP=20, SL=5 | **+$2.54** | **[−$4.22, +$10.34]** | 47% | 2.29 | −$0.22 |

**CI includes zero. Not statistically significant per CLAUDE.md mandate.**

## Why it fails

Direction classifier was trained on **daisy oracle entries** — bars hindsight-picked as the start of favorable moves. Direction accuracy = 81-87% at those bars.

The live strategy fires at **every cadence trigger where confidence > T**. Most of those triggers are NOT oracle moments — they're mid-move, end-of-move, or noise. The classifier still predicts direction correctly at those bars (because the V2 features still look directional), but the remaining favorable move is small or absent.

Concretely:
- Oracle entry at 10:00 LONG → MFE +$139 in 36 min, well above any TP target
- Classifier-confident bar at 10:30 LONG → only 6 min of move left → TP=$20 rarely hits, SL clipped on noise reversal

This is the **entry-timing problem** — the classifier solves direction but not WHEN to enter. The daisy oracle gave entry timing for free in earlier forward-pass simulations; the live strategy must discover entries itself.

## Critical caveats verified

1. **Mode: tick-exact**. Without this fix the engine inflates $/day by 20-30× via intrabar overshoot.
2. **Costs: $4/trade**. Realistic MNQ commission + slippage.
3. **OOS only**. 68 calendar days / 56 sessions, 2026 Jan-Mar.
4. **Bootstrap CI**: 4,000 percentile resamples per CLAUDE.md.
5. **MAE tracked**: `trough_pnl` added to ClosedTrade.

## KPI evaluation

| KPI | Target | Best achieved | Verdict |
|-----|--------|---------------|---------|
| Daily $100 NET | $100/day | $2.54/day (CI crosses zero) | **FAIL** |
| Low MAE | small | −$0.22 (best) to −$32 (typical) | mixed |
| High Day WR | >55% | 47% (best) | **FAIL** |

## Architectural conclusion

**The direction classifier alone is NOT a deployable strategy.** It's a router for direction, conditional on being given an entry candidate. Three paths forward:

### Path A — Pair with existing tiers as filter
The 9 ExNMP / 4 trend tiers already have proven entry-timing. Use the classifier as a FILTER: skip the tier's signal if the classifier disagrees on direction. This is the natural integration.

### Path B — Build an entry-timing classifier
A second model on V2 features that predicts "is this an oracle moment?" Use it as the entry trigger; direction classifier picks LONG/SHORT.

### Path C — Use a price-action trigger
Fire on breakouts, pullbacks, sweeps, divergences. Use the direction classifier to confirm/route once a trigger fires.

## What's validated despite the failure

1. **The infrastructure works.** Ticker, engine, ledger, MAE tracking, tick-exact exits, KPI search — all running and producing trustworthy numbers.
2. **OOS gap is real and quantified**. The earlier `forward_pass_OOS_2026.csv` "$50-100/day" was achieved BECAUSE the daisy oracle handed us entry timing. Remove that crutch and the strategy disappears.
3. **The intrabar overshoot bug is fixed**. Future engine runs with TickExactTP/SL will give honest exit prices.

## Files

- `tools/fit_direction_classifier.py` NEW
- `training_iso_v2/strategies/direction_classifier.py` NEW
- `training_iso_v2/exits_tick_exact.py` NEW
- `training_iso_v2/engine.py` PATCH (honor _force_exit_price)
- `training_iso_v2/ledger.py` PATCH (track trough_pnl)
- `tools/dir_clf_kpi_search.py` NEW
- `reports/findings/regret_oracle/dir_clf_kpi_search.csv` — symmetric grid (bar-close + tick-exact)
- `reports/findings/regret_oracle/dir_clf_kpi_search_v2_asymR.csv` — asymmetric grid

## Recommended next experiment

**Path A (fastest answer)**: re-run the existing tier strategies through the iso pipeline with `cnn_filter = DirectionClassifierStrategy` as a vote-gate. Compare per-tier $/day with/without the filter.

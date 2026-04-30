# 2026-04-27 — v1.5-RC NT8 backtest reality vs Python prediction

**Status: CRITICAL FINDING — Python prediction did NOT hold in NT8 native backtest.**

User ran 23 Strategy Analyzer backtests of `ZigzagRunner_v1.5-RC` between
07:50–08:59 AM on 2026-04-27. The runs sweep `BleedThresholdZ` and toggle
`EnableRegimeFilter` / `RideWithTrend`. All runs use **R=30** (the live
deployed config), unlike my overnight Python analysis which used **R=50**.

Source XMLs: `C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\strategyanalyzerlogs\@@@ZigzagRunner_v15_2026_04_27_*.xml`
Parsed by: `tools/nt8_strategyanalyzer_parser.py`

## Headline result

**Python prediction (overnight analysis, R=50, 95-day window):**
- Unfiltered: -$552
- v1.5-RC at z=-0.5: **+$5,021**
- v1.5-RC at z=-0.34 (default): ~+$5,300

**NT8 reality (R=30, 115-day window, with commission):**
- Unfiltered (filter OFF): **-$8,505**
- v1.5-RC at z=-0.34 (default): **-$2,062**
- v1.5-RC at z=-0.75 (best in this window): **-$1,005**
- v1.5-RC at z=+0.75 (alt sweet spot from analysis): **-$5,084**

**The filter genuinely helps** — turning it off costs $6,500 over 115 days vs
default — **but the strategy is still net negative on NT8 backtest.** The
+$5K-style result from Python does NOT replicate.

## Full sweep table

Window: 2026-01-01 → 2026-04-26 (115 days), commission ON, R=30, 1 contract.

| z | Filter | RideWithTrend | Trades | Trades/day | Total PnL | $/day | Run |
|---:|:-:|:-:|---:|---:|---:|---:|---:|
| — | OFF | OFF | 3,664 | 46.6 | **−$8,505** | −$74 | _05, _41 |
| −0.34 | ON | OFF | 1,464 | 18.9 | −$2,062 | −$18 | _03, _19, _23 |
| −0.34 | ON | ON | 1,464 | 18.9 | −$3,261 | −$28 | _21 |
| −0.70 | ON | OFF | 1,219 | 15.8 | −$1,265 | −$11 | _33 |
| **−0.75** | ON | OFF | 1,185 | 15.5 | **−$1,005** | **−$8.7** | _35 (BEST) |
| −2.00 | ON | OFF | 753 | 10.6 | −$1,305 | −$11 | _37 |
| +0.70 | ON | OFF | 2,342 | 29.8 | −$5,084 | −$44 | _31 |
| +0.75 | ON | OFF | 2,342 | 29.8 | −$5,084 | −$44 | _25, _29, _39 |
| +0.75 | ON | ON | 2,342 | 29.8 | −$3,891 | −$34 | _27 |

**Best NT8 config**: `z=−0.75, RideWithTrend=False, EnableRegimeFilter=True`
yields −$8.74/day. The filter saves $65/day vs unfiltered (−$74 → −$9), but
still bleeds.

## OOS-2 window (Apr 6–26, 20 days)

| z | Filter | Trades | Trades/day | PnL | $/day | Run |
|---:|:-:|---:|---:|---:|---:|---:|
| +0.75 | ON | 483 | 35.0 | −$1,396 | −$70 | _45 |
| +0.75 | OFF | 483 | 35.0 | −$1,396 | −$70 | _43 |
| −0.34 | ON | 161 | 38.9 | −$881 (with comm) | −$147 | _07 |
| −0.34 | ON | 161 | 38.9 | −$576 (no comm) | −$96 | _15 |

**At +0.75 the filter is INACTIVE on this window** (same trade count w/ and w/o
filter — every day passes). At −0.34 it does filter, but the per-day PnL is
worse than unfiltered (likely because the recent 20 days are mostly chop where
counter-trend works fine).

## Why the Python prediction failed

The Python overnight analysis used **R=50**. The NT8 backtest uses **R=30**
(matching the live deployed v1.0). These are not the same strategy:

| Setting | R=50 (Python analysis) | R=30 (NT8 backtest, live config) |
|---|---:|---:|
| Trades/day baseline | ~17 | ~31–47 |
| Larger pivots → fewer setups | ✓ | ✗ — many small pivots |
| Bleed filter calibration | ✓ on R=50 distribution | ✗ — wrong distribution |
| Per-trade noise (commission, slippage) | low (fewer trades) | high (more trades) |

R=30 produces **2.5–3× more trades**. The bleed filter was calibrated on R=50's
trade distribution. At R=30 it filters out the wrong fraction of days.

This is a **strategic miscalibration**, not a bug:
- The filter signal (`prior_range`, `range_compression`) is at DAY level — it's
  the same in both R configs.
- But the OUTCOME signal — what makes a "bleed day" vs "harvest day" — is
  fundamentally different at R=30 (where every trade is smaller and chop noise
  dominates) vs R=50 (where pivots are bigger and trend continuation matters).

## What this means for v1.5-RC promotion

**DO NOT promote v1.5-RC to live at R=30 with current calibration.**

| Config | Promote to live? | Why |
|---|:-:|---|
| v1.5-RC at z=−0.34 default, R=30 | **NO** | −$18/day on 115-day backtest |
| v1.5-RC at z=−0.75, R=30 | **NO** | −$9/day, marginally better but still bleeds |
| v1.5-RC at any z, R=50 | UNTESTED in NT8 | Need NT8 backtest at R=50 to confirm Python prediction |
| v1.0 (currently live) | KEEP RUNNING | Day 1 +$455 evidence trumps unproven v1.5-RC |

## What to do next

1. **Run NT8 backtest at R=50** to test whether the +$5K Python prediction
   matches NT8 reality at the same R. If it does, v1.5-RC is viable but only
   at R=50 (different config than live v1.0). If it doesn't, the filter is
   broken regardless of R.
2. **Run NT8 backtest at R=30 with bleed filter RECALIBRATED** for R=30 trade
   distribution. Use the same Python tooling but feed it R=30 trade ledgers.
   Find the IS-calibrated constants for R=30 and re-evaluate.
3. **Keep v1.0 live.** Day 1 +$455 + OOS-2 evidence is what we have. v1.5-RC
   needs validation before any promotion.
4. **Update v1.5-RC handoff doc** to flag this contradiction prominently.

## Lessons learned (memory candidates)

1. **R-parameter is config-defining.** A strategy at R=30 vs R=50 is not the
   "same strategy with different sizing" — it's a fundamentally different
   trade distribution. Filter calibrations don't transfer across R.
2. **Single-window Python predictions need NT8 cross-validation before any
   forward statement.** My handoff doc claimed +$5K. The actual NT8 backtest
   gives −$1K. Embarrassing miss. The OOS-2 designation we just made is the
   right defense for next time.
3. **NT8 Strategy Analyzer XML logs are gold.** They preserve the full param
   set per run AND the performance metrics. The `tools/nt8_strategyanalyzer_parser.py`
   tool now reads them. Future v1.5-RC iterations should be tracked through
   this parser to avoid losing the run history.
4. **Don't trust `OptimizationParameters` alone in the XML.** That block stores
   the LAST sweep value, not the actual run-time value. The authoritative source
   is the inner `StrategyTemplate` (HTML-encoded XML inside the entry).
5. **Always commission-include backtests.** Run #01 ($720 net no commission) vs
   Run #03 (−$2,062 net w/ commission) on identical params: ~$1.90/trade × 1464
   trades ≈ $2,782 commission. Without commission you get a misleading rosy
   number.

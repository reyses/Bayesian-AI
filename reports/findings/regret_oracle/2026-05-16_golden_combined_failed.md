# Golden-moment + direction combined classifier — OOS verdict: FAILS

## Setup

After the direction-only classifier failed OOS (earlier session today), user proposed: filter the oracle to keep ONLY high-quality trades, making the negative class "exponentially larger." Then user pivoted to **velocity filter** ($/min, not total $) — fast trades are real momentum, easier to predict.

Built full two-classifier pipeline:
1. **Entry-timing classifier** (`golden_entry_clf_vel5_fixed_gbm.pkl`): GBM on V2 features, positive = oracle bars with `mfe_velocity > $5/min` (1.7% of 1m bars IS, 2.2% OOS). OOS AUC 0.81, precision 14.5% at T=0.85 (25 fires/day expected).
2. **Direction classifier** (existing `direction_clf.pkl`): LR on V2 features, P(LONG). OOS argmax acc 81%.
3. **GoldenCombinedStrategy**: fires when both gates pass.
4. Through tick-exact engine with TP/SL grid, $4 cost/trade, 10-min TimeStop.

## Critical bug found + fixed

V2 layered data has **sub-5s irregular timestamps** (tick-level events). Original dataset builder used `iloc[::12]` assuming clean 5s cadence — produced timestamps NOT aligned with the ticker's `is_1m_close = (ts % 60 == 0)`. 90% of cache lookups failed.

**Fix**: filter `merged[merged['timestamp'] % 60 == 0]` instead of `iloc[::12]`. After fix:
- Cache verified: all timestamps minute-aligned
- Fire rates jumped 5-10× (matching classifier expectations)
- Trade volumes hit user's ~20/day zone at T_timing=0.85

## OOS 2026 grid result (60 combos, fixed alignment)

| Top by $/day | Config | $/day NET | 95% CI | DayWR | TP% / SL% |
|---|---|---|---|---|---|
| Best | T_t=0.90, T_d=0.75, TP=$20/SL=$10 | **−$14.80** | **[−$23, −$6]** | 32% | 37 / 62 |
| At 20/day vol | T_t=0.85, T_d=0.55, TP=$30/SL=$20 | −$70.55 | [−$103, −$37] | 38% | 40 / 57 |
| Best DayWR | T_t=0.90, T_d=0.55, TP=$30/SL=$20 | −$19.65 | [−$42, +$3] | 43% | 40 / 55 |

**All 60 configs have CI strictly ≤ 0**. The strategy is statistically a loser, not noise.

## Why it fails — the killer insight

**Direction classifier accuracy at high-timing-confidence bars is WORSE than random**.

- Direction classifier IS argmax acc = 81% on **all oracle bars** (smooth setups)
- TP/(TP+SL) on the combined strategy = **40 / 57 = 41%** at T_timing=0.85 OOS
- That's BELOW 50% baseline — direction is anti-predictive when timing fires high

**The mechanism**:
- High P_timing bars = market extremes / inflections / volatile moments
- These are exactly the bars where direction is genuinely uncertain
- Direction classifier was trained on the AVERAGE oracle bar (smooth setups dominate the IS sample)
- The two signals aren't independent — they're **anti-correlated where it matters most**

This explains why simply gating direction by timing-confidence doesn't help: the higher the timing-confidence, the worse the direction-confidence becomes.

## Why this is different from "direction classifier alone" failure

Earlier session: direction classifier alone produced +$2.54/day NET on OOS (CI crossed 0). The current architecture (timing × direction) does WORSE (-$15 to -$71/day NET, CI < 0 always). Adding the timing filter HURT.

The reason: at random bars, direction is 81% accurate. Force-filtering to high-timing-confidence bars, direction drops to 41%. The timing classifier is selecting bars where the direction classifier is structurally weak.

## What's validated

1. **Theoretical ceiling exists**: with PERFECT entry timing + 87% direction at 20 trades/day, $/day ranges $200-$1,200 depending on TP/SL. The signal is there.
2. **Velocity filter > magnitude filter** for ceiling characterization: lower SL%, similar $/day ceiling. The user's intuition was correct.
3. **Tick-exact exits + MAE tracking** working correctly. Earlier sessions' inflated $/day numbers were intrabar-overshoot artifacts.
4. **Engine infrastructure** robust: ticker, ledger, cache lookups, bootstrap CI all functioning.

## What this rules out

- **Two V2-feature classifiers stacked** as entry timing + direction filter. The interaction is destructive due to anti-correlation between extreme/inflection bars and direction certainty.
- **Threshold tuning won't fix it.** All 60 grid points failed; we explored the operating space thoroughly.

## Path forward — Path A is the next experiment

The user's original proposal: **use the direction classifier as a FILTER on existing tier strategies**. The tiers (FADE_CALM has $13.50/day OOS edge documented in MEMORY; CASCADE, MA_ALIGN etc.) already solve entry timing through proven price-action triggers. The direction classifier vetoes signals that conflict with predicted direction.

Why this is more likely to work:
- Tiers fire at price-action moments (compression bounces, fades, breakouts) — NOT at extremes/inflections
- Direction classifier's 81% acc IS measured on smooth setups → matches tier-fire bar profiles
- The two signals are LIKELY INDEPENDENT for tier-bars (vs the anti-correlation we just saw)

## Files

- `training_iso_v2/strategies/golden_combined.py` — combined strategy with cache support
- `training_iso_v2/strategies/direction_classifier.py` — direction-only strategy
- `tools/build_golden_entry_dataset.py` — FIXED to filter `ts % 60 == 0`
- `tools/train_golden_entry_classifier.py`, `tools/precompute_golden_timing.py`
- `tools/golden_ceiling_calc.py` — theoretical ceiling tool
- `tools/golden_combined_kpi.py` — grid driver
- `reports/findings/regret_oracle/golden_combined_kpi_FIXED.csv` — full grid results
- `reports/findings/regret_oracle/golden_ceiling_OOS.csv`, `_velocity.csv` — ceiling reference

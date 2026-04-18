---
name: Honest baseline (post-lookahead-fix)
description: After fixing build_dataset lookahead, baseline dropped from $740/day to -$164/day. All 8 tiers are ~50% counter-flip coin flips. Peak-physics exits disproved.
type: project
---

# Honest baseline — 2026-04-17

## The fix that changed everything

`training/build_dataset.py` had lookahead bias in higher-TF aggregation:
```python
# BEFORE (lookahead — picks TF bar whose START is at target_ts,
#         but OHLCV aggregates forward 5s bars to bar end)
idx = np.searchsorted(tf_ts, target_ts, side='right') - 1

# AFTER (only uses completed bars — shift back by period)
idx = np.searchsorted(tf_ts, target_ts - period, 'right') - 1
```

Feature folders also reorganized — now `DATA/ATLAS/FEATURES_5s/` and
`DATA/ATLAS_OOS/FEATURES_5s/` instead of top-level `DATA/FEATURES_79D_1m/`.

**Why:** every 2025 OOS number with `DATA/FEATURES_79D_1m/` is
contaminated. Previous $740/day baseline was lookahead, not edge.

**How to apply:** Any analysis using features from before 2026-04-17 must
be re-run on the new features. The old feature directory is gone.

## Honest baseline numbers

After the fix, full pipeline: **-$164/day IS on 348 days of 2025**.
Chains alone accounted for $157/day of loss — they amplified bad signals.

Per-tier isolated (no chains, no catch-all):

| Tier | N | $/day |
|---|---|---|
| RIDE_AGAINST | 39,721 | -$11 |
| FADE_CALM | 24,039 | -$16 |
| MTF_BREAKOUT | 5,961 | +$4 |
| FADE_AGAINST | 4,532 | +$5 |
| KILL_SHOT | 4,411 | -$2 |
| CASCADE | 1,270 | +$6 |
| MTF_EXHAUSTION | 233 | +$9 |
| FREIGHT_TRAIN | 34 | +$61 (n too small) |

Every tier at the noise floor.

## KILL_SHOT peak physics disproved

Path-level analysis on 2,043 trades with peak > $3:
- 1m velocity flips against trade at peak: **3.3%**
- 1m acceleration flips: **0.2%**
- Wick on other side (>30% jump): **6.8%**
- Largest Cohen-d across peak: **0.19** (1m_wick_ratio)

**There is no detectable physics of the peak.** The peak is a statistical
maximum over noise. Back-test confirmed: natural exit (+$11.61/trade)
beats every physics-based rule including 50% trail (+$3.40), fixed
targets, velocity/accel flips.

Implication: KILL_SHOT loss is an **entry filter** problem, not exits.
~1,720 trades with peak ≤$3 are structural losers.

## All tiers are coin flips on direction

Regret analysis on all 8 isolated tier pickles:

| Tier | % counter-flip | Counter WR |
|---|---|---|
| All tiers | ~49% | ~40% |

**~50% counter-flip means no directional edge.** Regret labels are
meaningful (counter-labeled trades have 38-45% actual WR, so regret
ranks correctly), but the SAME/COUNTER boundary is near-random in
91D feature space.

nn_v2 on NMP worked because NMP was 30-35% counter — there was real
direction. Here every tier is a coin flip dressed as physics.

Oracle upper bound (flip at exit, no peak-chasing): +$2,183/day pooled.
Realistic CNN at 65% accuracy: ~$900-$1,300/day — IF separability exists.

## Data caveat — regret LOOKAHEAD

`training/regret.py` has `LOOKAHEAD = 360` commented "30 min at 5s
resolution" but loads 1m price data → actual window is 6 hours. Every
trade's "optimal" becomes "hold 6 hours and catch the biggest swing."
99% of best_action labels are same_extended or counter_extended.

`flip_at_exit` is still clean (uses actual exit bar, no peak-chasing).
Raw counter-flip % may be slightly inflated by the 6-hour window. Need
to cap LOOKAHEAD to 15-30 min (or use 5s prices) and re-run.

## Decision tree for next session

1. **Fix regret.py LOOKAHEAD** (30 min) — remove 6-hour distortion first.
2. **Test CNN separability on FADE_CALM** (biggest, 24k trades): if CNN
   clears 58%+ OOS on SAME/COUNTER, full pipeline viable. If plateaus
   at 52-54%, tiers are truly dead.
3. **If (2) fails**: rebuild tiers from data (corrected-trade clustering)
   instead of hand-crafted physics gates.

## Tools

- `tools/run_tier_isolated.py` — isolate each tier, no chains/catch-all
- `tools/killshot_peak_physics.py` — path reconstruction + peak physics
- `tools/regret_on_isolated.py` — regret per tier, verdict table

## Reports

- `reports/findings/2026-04-17_killshot_peak_physics.md`
- `reports/findings/2026-04-17_iso_regret.md`

# Entry Logic Analysis — Advance Engine + Execution Engine
> Session: 2026-04-03
> Status: REFERENCE — understand what to keep/drop for new system

## MARKER: WHY ADVANCE ENGINE FAILS

Two entry sources, mutually exclusive:
1. **Pattern Detection**: requires `cascade_detected` OR `structure_confirmed` from SFE
2. **Peak Reversal**: fires when Pattern Detection finds nothing

In practice: Pattern Detection almost never fires on clean data (1,381 patterns in 311 days).
So 100% of trades come from Peak Reversal → template -100 → 50.1% WR coin flip.
The entire pattern detection + template matching + gate cascade is dead code in practice.

## MARKER: PEAK REVERSAL ENTRY (what actually fires)

### Detection (`_detect_peak_reversal`):
```
Stage 1 — Instantaneous:
  P_center rose >5% bar-over-bar
  |F_momentum| fell >10% bar-over-bar
  oscillation_entropy > 0.55

Stage 2 — 10-bar buildup confirmation:
  One of: exhaustion (F_mom decaying 3+ bars)
       or buildup-to-collapse (F_mom building then crashing)
       or regime shift (P_center shifting 3+ bars)
```

### Direction: sign(F_momentum) → opposite direction (fading the momentum)

### Sensor gates (after detection):
1. Chasing filter: |F_momentum| > 50 → BLOCK
2. Min signal floor: vol_delta < 1.0 AND |F_mom| < 0.5 → BLOCK  
3. 1m sensor: vol_1m AND fm_1m both opposing → BLOCK (most trades die here)
4. Cat brain regime: if Cat says no → BLOCK

**Magic numbers**: 0.05, -0.10, 0.55, 50.0, 1.0, 0.5, -1.5, -50.0

## MARKER: GATE CASCADE (execution engine)

For the rare pattern-detected candidates:

| Gate | What it checks | Config |
|------|---------------|--------|
| Gate 0 | Z-score tiers, regime compat, session, Hurst > 0.5, momentum > reversion force, reversion_prob > 0.40 | Many params |
| Gate 0.5 | Depth >= 3 | depth_blacklist |
| Gate 1 | Template distance < 4.5 (Euclidean in scaled space) | gate1_dist |
| Gate 2 | Brain says template has > 5% WR | brain_min_prob |
| Gate 2.5 | 40%+ of TFs agree on direction | tf_confluence_min |
| Gate 3 | Belief network is confident | — |
| Gate 4 | F_momentum sign matches trade direction | — |
| FDMI | Not State A (micro breakout with no macro energy) | fdmi_fakeout_macro_adx |

**Gate 4 is the most validated**: "WR drops from 88% to 45% when misaligned"

## MARKER: DIRECTION CASCADE (9 voters)

| Vote | Source | Weight | What it reads |
|------|--------|--------|--------------|
| 0 | Brain dir_bias (low obs) | 1.5 | Historical WR per direction |
| 1 | Signed MFE regression | 3.0 | OLS model on features |
| 2 | Logistic regression | 2.5 | Per-cluster P(LONG) model |
| 3 | Brain direction WR | 1.5 | P(LONG) - P(SHORT) from brain |
| 4 | Template aggregate bias | 1.0 | long_bias from library |
| 5 | **Fractal DMI** | **4.0** | State B (ignition) or D (reversion) — DOMINATES |
| 6 | Band confluence | 2.0 | Multi-TF band direction |
| 7 | Multi-TF DMI trend | 1.5 | DMI trend strength |
| 8 | Velocity (fallback) | 0.5 | Price velocity sign |

**Short-circuits before voting**: pp_override, brain dir_bias with 10+ obs, live momentum

**FDMI at weight 4.0 dominates** — if it fires confidently, it overrides everything.

## MARKER: SIZING

No position sizing (contracts). Only SL/TP/trail in ticks:
- **SL**: p95_mae * tolerance, scaled by sqrt(discovery_tf / 15s), clamped [2, 200]
- **TP**: p75_mfe, OLS-adjusted, brain-adjusted, floored at 4 ticks
- **Trail**: regression_sigma * 1.1, floored at 2 ticks
- **Regime overlay**: ATR-based floors by regime type

## MARKER: WHAT TO KEEP FOR NEW SYSTEM

### Definitely keep:
- **Gate 4 logic**: momentum alignment check (88% vs 45% WR — strongest validated gate)
- **TF confluence** (Gate 2.5): 40%+ TFs must agree — maps directly to 79D multi-TF reading
- **sqrt(time) TF scaling**: sound diffusion physics for SL/TP across timeframes
- **Trailing stop ratchet**: sensor-adaptive width (50/65/80%), never moves against trade

### Keep the concept, redesign:
- **Direction from DMI across TFs**: FDMI weight=4.0 is the strongest voter. In 79D we have dmi_diff at every TF — same signal, cleaner implementation
- **Regime classification**: strong_trend/developing/exhausting/range/chop — useful but should derive from 79D (variance_ratio + hurst at each TF)
- **Brain learning**: per-state win rate tracking is sound. But needs the 79D state, not template IDs

### Drop entirely:
- **Peak Reversal path**: P_center / F_momentum / oscillation_entropy — none of these are in 79D. The entire peak detection is a workaround for pattern detection not firing
- **Pattern detection gate**: cascade_detected / structure_confirmed — replaced by "every bar is tradeable" + NN routing
- **Template distance matching**: K-Means centroids in scaled space — replaced by NN classification
- **Gate 0 z-score tiers**: Region 3 / Region 4 logic — replaced by NN reading z_se at all TFs
- **Gate 0.5 depth filter**: fractal depth is a discovery artifact, not in 79D
- **Cat brain**: rolling delta regime classifier — replaced by variance_ratio at each TF
- **Worker bypass**: coarse fallback path — no longer needed when every bar has NN classification
- **Ping-pong override**: live-specific hack
- **9-voter direction cascade**: replaced by NN direction output
- **All magic numbers in peak detection**: 0.05, -0.10, 0.55, 50.0, etc.

## MARKER: BUGS / DEAD CODE

1. `no_signal` case returns `('long', 0.50)` — defaults to LONG instead of HOLD. Direction bias.
2. `FM_FLIP_THRESHOLD = 30.0` defined but never used in 1m flip check
3. `depth_blacklist = {0,1,2}` AND `_min_trade_depth = 3` — duplicated
4. `score = depth + dist + tier_adj` — mixes integer depth with continuous distance (different units)
5. `brain.dir_bias` fires at `n_obs >= 10` but config says `live_bias_min_trades = 5` — inconsistent
6. Peak ADX chop filter threshold = 0.0 — effectively disabled, config exists but does nothing
7. `CHASE_FM_THRESHOLD` widened from 20 to 50 "because it blocked real entries" — maybe 20 was right

## MARKER: CONNECTION TO 79D NN SYSTEM

The new system replaces ALL of the above with:

```
Every 1m bar:
  1. Compute 79D state (10 features x 6 TFs + helpers)
  2. NN classifies: direction + hold duration (half-life)
  3. If no_trade → skip
  4. Momentum alignment check (Gate 4 — the ONE validated gate)
  5. Enter at 5s execution with NN-predicted half-life
  6. Unified exit: envelope decay modulated by survival score
```

6 steps. Not 9 gates + 9 voters + template matching + peak detection + sensor fusion.

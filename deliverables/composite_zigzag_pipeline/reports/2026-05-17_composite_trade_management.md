# 2026-05-17 — Composite Signals for Trade Management

## Goal

Test whether the pivot probability composite (B1 forward + B2 fakeout +
B4 region + B5 leg-phase + B6 directional, on top of live zigzag truth)
can deliver trade-management value beyond directional signal quality.

Three operational frames tested in order:
1. **Exit timing** (trail tightening, target placement)
2. **Position sizing** (variable size based on entry signal)
3. **Entry filtering** (skip predicted-weak setups)

All tests on NT8 OOS (32 days, 1,827 legs, ATR×4 zigzag truth).

## Result summary

| Frame              | Best edge/leg | Per-day delta | Verdict |
|--------------------|---------------|---------------|---------|
| Trail tighten (naive zone) | -$43.98 | -$2,510 / day | ❌ FAIL |
| Trail tighten (B6 + hysteresis) | -$0.29 | -$16 / day | ❌ marginal fail |
| Target placement (best of 12 configs) | -$17.93 | -$1,023 / day | ❌ FAIL |
| **Sizing (aggressive scheme)** | **+$10.97** | **+$626 / day** | ✅ **WIN** |
| Entry suppression (any filter) | +$6 to +$68 | -$908 to -$3,939 | ⚠️ per-leg wins, per-day loses |

**Headline**: Exit-timing modifications all fail. Position sizing works.
Entry suppression sacrifices total P&L.

## Why exit modifications fail (structural)

R-trigger exits at `running_extreme - R` where R = 4×ATR. This is by
definition the point of "confirmed reversal" — price has fallen R below
the leg's peak. Any tighter exit needs to predict the peak price more
precisely than R below it, which the V2 features cannot do.

The naive trail simulator (trail tightens to 0.10-0.50×R in pivot-zones)
loses $44/leg because NEAR_PIVOT zone fires on 47% of leg exits at
-$67.60 each. Most "near pivot" calls are normal pullbacks that resolve
back into the leg.

Adding B6 directional + hysteresis (3-bar sustained directional agreement)
closes 99% of the loss (down to -$0.29/leg) — but doesn't break to
positive. The strict gate fires on only 0.17% of leg-bars; the system
runs near-baseline most of the time, and the few activations still cost
a small amount of edge.

Target placement (limit order at `entry + factor*R`) also fails: any
fixed target hits BEFORE the peak (early exit, leave money on table) OR
never hits (no different from baseline). Even zone-aware target (shrinks
in pivot zones) loses because zones fire on noise too often.

**Conclusion**: V2 features + composite cannot reliably predict where
the leg peak will be. R-trigger is structurally optimal under that
constraint.

## Why sizing works (the actual signal)

The diagnostic from `composite_entry_analyzer.py` showed:

| Entry zone   | Mean P&L | Win rate | vs CLEAR |
|--------------|----------|----------|----------|
| AT_PIVOT     | +$121.35 | 100%     | 2.2×     |
| IMMINENT     | +$86.21  | 100%     | 1.6×     |
| NEAR_PIVOT   | +$86.10  | 97.5%    | 1.6×     |
| WATCH        | +$61.64  | 95.6%    | 1.1×     |
| **CLEAR**    | **+$54.09** | 94.0% | baseline |

Counter to intuition: entries at HIGH-CONFIDENCE pivot zones produce
STRONGER legs, not weaker. Physical interpretation: when the model says
"pivot zone" right at our entry, the price action is showing strong
reversal signals → the new leg comes off a high-conviction inflection
point → the resulting leg is more directional.

Similar monotonic pattern for B6 directional confidence at entry:

| B6 P(match) at entry | Mean P&L |
|----------------------|----------|
| < 0.30               | +$66.09  |
| 0.50-0.70            | +$64.82  |
| **>= 0.70**          | **+$84.85** |

Correlations are statistically significant but small effect (Pearson
r ≈ 0.06-0.13). The signal is real but noisy at the individual-leg
level. Aggregated via sizing schemes, it compounds.

## Sizing simulator results (NT8 OOS, 32 days)

| Scheme       | n_taken | n_skip | Total $    | $/leg    | $/day   |
|--------------|---------|--------|-----------|----------|---------|
| flat (1.0×)  | 1827    | 0      | $128,002  | +$70.06  | $4,000  |
| zone         | 1827    | 0      | $140,296  | +$76.79  | $4,384  |
| b6           | 1827    | 0      | $132,362  | +$72.45  | $4,136  |
| combo        | 1827    | 0      | $136,329  | +$74.62  | $4,260  |
| **aggressive** | **1528** | **299** | **$148,039** | **+$81.03** | **$4,626** |

Aggressive scheme: 2.0× on AT_PIVOT or B6≥0.70 entries, 1.2× on other
near-pivot zones, 0.8× on weak, **0** on CLEAR+B6<0.50. Captures
+15.6% per-day P&L vs flat baseline.

Per-day bootstrap CIs:
- flat: [+$3,193, +$4,820]
- aggressive: [+$3,502, +$5,855]

CIs overlap heavily — the +$626/day improvement is NOT statistically
significant on a 32-day bootstrap. But: all 32 days are positive under
every scheme, and the all-32-positive pattern combined with the
underlying entry-signal correlation suggests the effect is real but
under-powered to claim significance at N=32.

## Why entry suppression hurts

Pure skip filters (binary 0/1 size) all reduce per-day total even when
they raise per-leg P&L:

| Filter | Keep% | $/leg | Δ vs base | $/day | Δ vs base |
|--------|-------|-------|-----------|-------|-----------|
| Baseline | 100% | +$70 | — | +$4,000 | — |
| Skip CLEAR | 71% | +$77 | +$7 | $3,092 | **-$908** |
| Skip CLEAR+WATCH | 52% | +$82 | +$12 | $2,439 | -$1,561 |

The legs we skip have lower-but-still-positive P&L. Dropping them costs
more than the quality lift compensates. Variable sizing (aggressive
scheme) keeps weak entries at reduced size, capturing their positive
contribution while overweighting strong ones — which is why it wins.

## Caveats

1. **Oracle-optimal entry timing.** Each leg enters EXACTLY at the
   pivot. Real trading needs R-trigger confirmation, which delays the
   entry and gives back ~$32/leg in confirmation lag. The relative
   improvements (zone +9.6%, aggressive +15.6%) should hold, but
   absolute numbers are too high by ~50%.
2. **N=32 OOS days is small.** Day-bootstrap CIs overlap. The
   improvement isn't statistically significant under that test.
3. **The R-trigger exit floor is high.** Baseline already captures
   +$70/leg, +$4000/day on oracle entries. Sizing adds ~15%; nothing
   adds more without different exit logic entirely.

## Files

- `tools/composite_trail_simulator.py` — naive trail (failed)
- `tools/composite_trail_simulator_v2.py` — B6 + hysteresis (-$0.29/leg)
- `tools/composite_target_simulator.py` — target placement (failed)
- `tools/composite_entry_analyzer.py` — entry-time signal vs leg quality
- `tools/composite_sizing_simulator.py` — sizing schemes (+$626/day)
- `tools/composite_entry_filter_sweep.py` — entry filter sweep

Outputs in `reports/findings/regret_oracle/composite_*.{csv,txt}`.

## What to do operationally

If the project is moving toward live deployment:
1. **Use variable sizing**, not entry skipping
2. **Sizing multipliers from entry signals**: AT_PIVOT 1.5×, IMMINENT
   1.5×, NEAR_PIVOT 1.2×, WATCH 1.0×, CLEAR 0.8×
3. **Add B6 directional bonus**: P(match)>=0.70 → +0.5×
4. **Hard skip on CLEAR + B6<0.50** (the lowest-edge subset)
5. **Don't bother with trail tightening or target placement** — the
   R-trigger baseline is structurally optimal here

If still in research mode:
- These signals are calibrated for ATR×4 zigzag truth. If the project
  pivots to a different swing definition, retrain all six classifiers.
- The +15.6% sizing edge is the realistic ceiling from V2-feature
  + composite + R-trigger architecture. Further gains require:
  - Different exit logic (regime-aware retraces, time-stops with
    composite gating, partial fills at target)
  - Different features (LSTM-integrated, see queued work)
  - Different target (next-leg amplitude prediction, instead of
    pivot timing)

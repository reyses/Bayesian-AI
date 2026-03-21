# Pattern Recognition Scale Mismatch — Investigation Report

**Date**: 2026-03-10
**Context**: OOS forward pass analysis (ATLAS_OOS, Jan-Mar 2026, 1,766 trades)

---

## Executive Summary

808 of 997 correct-direction OOS trades (81%) gave back >90% of their favorable
price movement and exited at breakeven. Root cause: the pattern library's MFE
statistics are measured over oracle horizons of 1–30 hours, but actual trades
resolve in 1–5 minutes. This creates a 30x scale mismatch that disables the
giveback exit protection, causing trades to ride all the way back to the
breakeven stop instead of locking in profit.

**Impact**: ~$40,642 left on table in OOS. Estimated $6,000–$8,000 recoverable
with proper scaling.

---

## 1. How the Pattern Pipeline Works Today

### 1.1 Template Building (Training)

1. **Discovery**: `fractal_discovery_agent.py` scans each timeframe for pattern
   events (band reversals, momentum breaks). Each pattern has a native TF
   (e.g., 15m, 5m, 1m, 15s).

2. **Oracle MFE**: For each pattern, the oracle peeks ahead N bars **of the
   pattern's native TF** and measures the maximum favorable excursion (MFE):
   ```
   TF     Oracle lookahead    Horizon
   15m    16 bars              4 hours
   5m     24 bars              2 hours
   3m     60 bars              3 hours
   1m     60 bars              1 hour
   30s    60 bars              30 min
   15s    60 bars              15 min
   5s     120 bars             10 min
   1s     300 bars             5 min
   ```
   Source: `config/oracle_config.py` lines 8–19.

3. **Clustering**: `fractal_clustering.py` runs K-Means on a 16D feature vector
   (z-score, velocity, momentum, depth, ADX, hurst, etc.). Templates are formed
   from patterns with similar features — **regardless of their native TF**.

4. **MFE Aggregation**: Each template's `p75_mfe_ticks` is the 75th percentile
   of member MFE values. Since templates mix patterns from multiple TFs, this
   statistic blends oracle horizons:
   - TID 21 contains patterns from 15m (4h horizon), 5m (2h), and 3m (3h)
   - Its `p75_mfe_ticks = 1,077` averages multi-hour price swings

5. **Exit Sizing**: TP, SL, and anchor patience are computed from these stats:
   - `TP = p75_mfe_ticks × 0.85` → 915 ticks ($457)
   - `SL = mean_mae_ticks × 2.0` → 2,959 ticks ($1,480)
   - `anchor_mfe_ticks = p75_mfe_ticks` → 1,077 ticks
   Source: `training/orchestrator_worker.py` lines 223–247,
   `core/exit_engine.py` lines 220–239.

### 1.2 Anchor Patience Logic

The giveback exit (`_check_peak_giveback` in exit_engine.py line 551) protects
profit by exiting when a trade gives back too much of its peak. But it has a
patience guard (line 578):

```python
if (pos.anchor_mfe_ticks > 0 and pos.anchor_mfe_bars > 0
        and pos.bars_held < pos.anchor_mfe_bars
        and peak_ticks < pos.anchor_mfe_ticks * 0.3):
    return None  # suppress giveback — trade still developing
```

This says: "Don't exit if the trade hasn't reached 30% of the template's
expected MFE." With `anchor_mfe_ticks = 1,077`, the threshold is 323 ticks
($81). Since actual trades only reach 35–48 ticks MFE on average, giveback
is **permanently disabled** for the vast majority of trades.

---

## 2. The Scale Mismatch (Quantified)

### 2.1 Template MFE vs Actual Trade MFE

| Oracle Horizon | Templates | Trades | Template p75 MFE | Actual MFE | Ratio |
|----------------|-----------|--------|-------------------|------------|-------|
| 3h+            | 11        | 387    | 2,607 ticks       | 86 ticks   | 3.3%  |
| 1–3h           | 14        | 245    | 1,452 ticks       | 78 ticks   | 5.4%  |
| 15–60min       | 16        | 432    | 360 ticks         | 65 ticks   | 18.2% |
| <15min         | 40        | 701    | 508 ticks         | 47 ticks   | 9.3%  |

**Overall**: Trades capture ~11% of template MFE and hold ~10% of expected time.

### 2.2 Specific Template Examples

**TID 0** (24h BandSnap, 126 trades):
- Template expects: p75_mfe = 3,387 ticks ($847), over 4-hour horizon
- Reality: avg MFE = 41 ticks ($10), avg hold = 1.2 minutes
- Anchor patience threshold: 1,016 ticks — **126/126 trades blocked**
- All 126 trades exit via breakeven stop

**TID 21** (1m BandSnap, 89 trades):
- Template expects: p75_mfe = 1,077 ticks ($269), mixed 4h/2h/3h horizons
- Reality: avg MFE = 105 ticks ($26), avg hold = 3.7 minutes
- Anchor patience threshold: 323 ticks — **81/89 trades blocked**
- Contains patterns from 15m (35), 5m (28), and 3m (14) — three different horizons

**TID 54** (35s BandSnap, 104 trades):
- Template expects: p75_mfe = 212 ticks ($53), 30-minute horizon
- Reality: avg MFE = 58 ticks ($14), avg hold = 1.9 minutes
- Anchor patience threshold: 64 ticks — **80/104 blocked**, 24 get giveback

### 2.3 TP Reachability

- Average MFE/TP ratio across all templates: **14.9%**
- Templates where TP is reachable (MFE > TP): **0 out of 81**
- Total TP exits in OOS: **9 out of 1,766 trades (0.5%)**
- TPs are computed from multi-hour oracle horizons but trades last ~1 minute

### 2.4 Giveback Trade Anatomy

808 correct-direction OOS trades gave back >90% of MFE:

| Root Cause | Trades | Left on Table |
|------------|--------|---------------|
| Anchor patience blocks giveback (MFE >= 16 ticks) | 455 | $24,927 |
| MFE < 16 ticks (below giveback minimum) | 260 | $11,561 |
| Noise floor blocks giveback | 10 | ~$500 |
| Other | 83 | ~$3,654 |

The 455 anchor-blocked trades had:
- Avg anchor_mfe_ticks: 1,022 (template expectation)
- Avg actual trade MFE: 48 ticks (what happened)
- Avg 30% threshold: 307 ticks (needed to enable giveback)
- Avg hold bars: 6.1 (1.5 minutes)

Even reducing the patience threshold from 30% to 10% only frees 169 trades —
the expectations are so inflated that even 10% is unreachable for most trades.

---

## 3. Why Templates Mix Timeframes

The 16D feature vector includes `depth` (feature[5]) and `tf_scale`
(feature[4] = log2(tf_seconds)), which should separate TFs. But the K-Means
clustering optimizes for **overall feature similarity**, not TF homogeneity.

Two patterns with:
- Similar z-scores, velocity, momentum, ADX, hurst
- But different depths (5m vs 15m)

...can still land in the same cluster if the other 14 features are close enough.
The depth/TF features get outvoted.

Result: TID 21 mixes 15m patterns (oracle horizon: 4h, MFE measured over
960 minutes) with 3m patterns (horizon: 3h, MFE over 180 minutes). The
aggregated `p75_mfe_ticks` is a meaningless average of these different scales.

---

## 4. Proposed Solution: Seed Primitives Anchored at 1-Minute Bars

### 4.1 Why Primitives

Instead of clustering on 16D features (which mixes TFs), classify price
trajectories against 20 mathematical shape templates using Pearson correlation.
Each shape (V-reversal, ramp, sigmoid, etc.) has a fixed window: **16 bars of
1-minute data = 16 minutes**.

Benefits:
1. **Time-anchored**: Every shape classification operates on the same 16-minute
   window, so MFE measured for "V_REVERSAL_UP @ 1m" is always a 16-minute move
2. **No TF mixing**: The shape is classified at one resolution, not aggregated
   across horizons
3. **Direction is intrinsic**: LINEAR_UP → long, V_DOWN → short — no oracle
   corrections needed
4. **MFE is shape-specific**: After running the primitives on historical data,
   each shape gets its own MFE distribution at 1m resolution

### 4.2 Why 1-Minute, Not 15-Minute

The original waveform research used 16 bars of 15m = 4 hours. But:

| Metric | Value | Source |
|--------|-------|--------|
| Median profitable trade hold | 5.0 min | OOS trade log |
| Average all-trade hold | 2.5 min | OOS trade log |
| Peak oscillation cycle | ~8 min | Hold time analysis |
| p25 profitable hold | 1.5 min | OOS trade log |
| p75 profitable hold | 8.5 min | OOS trade log |

16 bars × 1 min = 16 minutes captures the full oscillation with room to spare.
16 bars × 15 min = 4 hours is 50–100x longer than actual trade duration.

Alternative: 16 bars × 30s = 8 minutes would be even tighter to the oscillation
cycle, but 1m bars are cleaner (less microstructure noise, better shape matching).

### 4.3 Expected MFE After Re-anchoring

With a 16-minute window at 1m resolution, expected MFE per shape would be in
the range of **20–80 ticks** (matching actual trade MFE of 47–86 ticks). This
means:
- Giveback threshold (30% of anchor) ≈ 6–24 ticks — **reachable**
- TP sizing ≈ 17–68 ticks — **reachable**
- Anchor patience deactivates after 16 bars — **reasonable hold time**

### 4.4 The 20 Seed Shapes

Existing spec: `docs/specs/JULES_WAVEFORM_SEED_INTEGRATION.md`

**Directional (8)**: LINEAR, EXPONENTIAL, LOGARITHMIC, STEP (UP/DOWN each)
**Reversals (8)**: SYMMETRIC_V, ROUNDED_U, FRONT_SKEWED, BACK_SKEWED (UP/DOWN)
**Volatility (4)**: SINE_WAVE, DAMPED_OSCILLATOR, EXPAND_OSCILLATOR, FLATLINE

Classification: normalize price segment to [0,1], compute Pearson r against
each shape, pick best match if r >= 0.75.

### 4.5 Integration Approach

The existing spec has 5 parts. Key changes needed:

1. **Part 1** (seed_library.py): No change — the 20 shapes are mathematical,
   resolution-independent. `N=16` stays.

2. **Part 3** (forward pass direction): Change from 15m resamples to **1m bars**.
   The forward pass already has 1m data available (ATLAS includes 1m parquets).
   Build the shape buffer from the last 16 completed 1m bars instead of 15m.

3. **New: Shape-specific MFE table**: After running the IS forward pass with
   shape classification active, aggregate MFE/MAE statistics **per shape per TF**.
   Save as `checkpoints/shape_mfe_table.json`:
   ```json
   {
     "V_REVERSAL_UP": {"p75_mfe_ticks": 45, "avg_mfe_bar": 8, "mean_mae_ticks": 12},
     "LINEAR_DOWN": {"p75_mfe_ticks": 62, "avg_mfe_bar": 12, "mean_mae_ticks": 18},
     ...
   }
   ```
   This replaces the per-template stats currently in pattern_library.pkl.

4. **Exit engine**: `anchor_mfe_ticks` comes from the shape table instead of the
   K-Means template. Values will be 20–80 ticks instead of 200–5,500.

5. **Part 5** (live engine): Shape classification on 1m bars during live trading.
   The live engine already has a 1m aggregator — just feed the last 16 closes
   to `classify_trajectory()`.

---

## 5. Interim Fix: Cap Anchor Patience

While the primitives migration is implemented, a 2-line fix in
`core/exit_engine.py` can recover a significant portion of the lost profit:

**Option A — Hard cap** (simplest):
```python
# In open_position(), after _anchor_mfe is set:
_anchor_mfe = min(_anchor_mfe, 80)  # cap to realistic range
```

**Option B — Disable anchor patience entirely**:
```python
# In _check_peak_giveback(), comment out lines 578-581
```

**Option C — Use actual trade MFE percentile from brain history** (if available):
```python
# Replace p75_mfe_ticks with brain's running average MFE for this template
```

Estimated impact of cap at 80 ticks: 349 trades freed from patience, ~$5,852
additional PnL captured.

---

## 6. Audit Tools Created

1. **`tools/research_pattern_audit.py`** — 8-section pipeline audit:
   - Template overview, oracle horizons, scale mismatch, anchor patience impact,
     TP/SL sizing, depth analysis, per-trade step-by-step trace, recommendations
   - Usage: `python tools/research_pattern_audit.py --mode oos --trace 5`

2. **`tools/visualize_template.py`** — 6-panel template visualization:
   - Radar (16D fingerprint), MFE histogram, stats card, trade timeline,
     anchor patience diagram, transition map
   - Usage: `python tools/visualize_template.py 21 54 0 --mode oos`
   - Output: `reports/research/template_viz/template_N.png`

---

## 7. Key Data Points for Reference

- OOS total: $18,732 PnL, 1,766 trades, 98.5% WR
- Correct direction: 997 trades (56.5%)
- Gave back >90%: 808 trades (81% of correct-direction)
- Exit via stop_loss (breakeven): 733 of those 808
- Anchor patience blocked giveback: 455 of 733
- Template p75_mfe_ticks median: 486 ticks ($243)
- Actual p75 trade MFE median: 61 ticks ($30)
- Ratio: 12.6% (templates overstate by ~8x)
- Quality scoring r = -0.220 (validated OOS, quality weights working)
- Score competition correct pick rate: 66% OOS (72% IS)

# PROPOSAL: Multi-TF Sensor Array for TradeCNN

**From:** VS Code Claude (executor)
**To:** Claude.ai (architect/reviewer)
**Status:** Proposal — awaiting review before implementation

---

## Problem

TradeCNN StatePredictor operates on 1m features only (13D). It achieved $1,609/day OOS
with breakeven protection but:

1. Cannot see hourly structure (walls, support/resistance)
2. Cannot see 5m swing context (is this a pullback or reversal?)
3. Has no timing advantage within the 1m bar (1s resolution unused)
4. The old system had 11 TF workers — this has 1

The $1,609/day came from direction prediction at 1m. Adding higher-TF context
should improve entry quality (skip entries near walls) and exit timing
(hold longer in strong multi-TF trends).

---

## Proposal: Add Higher-TF Features to 13D Input

Instead of separate TF workers or separate models, embed multi-TF context
as additional features per 1m bar. The CNN sees everything in one snapshot.

### Current 13D Features (1m only)
```
7D directional: dmi_diff, dmi_gap, vol_rel, dir_vol, velocity, z_se, price_accel
4D regime:      std_price, variance_ratio, bar_range, wick_ratio
2D context:     vwap_distance, time_of_day
```

### Proposed Addition: 12D Multi-TF (4 features × 3 timeframes)

| Feature | 5m | 15m | 1h | Question |
|---------|-----|------|-----|----------|
| dmi_diff | 5m_dmi_diff | 15m_dmi_diff | 1h_dmi_diff | Who's winning at this scale? |
| z_se | 5m_z_se | 15m_z_se | 1h_z_se | Where in the range at this scale? |
| velocity | 5m_velocity | 15m_velocity | 1h_velocity | How fast at this scale? |
| vol_rel | 5m_vol_rel | 15m_vol_rel | 1h_vol_rel | How much participation at this scale? |

### Total: 25D (13D base + 12D multi-TF)

### Why These 3 TFs
- **5m**: captures the swing structure (5-10 bar swings at 1m = 1-2 bars at 5m)
- **15m**: intermediate trend (session-level direction)
- **1h**: structural walls (daily high/low, support/resistance zones)

### Why These 4 Features Per TF
From the grounded feature research (March 25-26):
- **dmi_diff**: who's winning (the ONE proven direction signal, 98.8% peak recall)
- **z_se**: where in the structure (statistical deviation from mean)
- **velocity**: how fast (rate of change — the base derivative)
- **vol_rel**: participation (the ONE independent measurement)

These are the same 4 features that matter at 1m, applied at each scale.
Each answers the same question at a different resolution.

---

## Implementation

### Data Pipeline
For each 1m bar at timestamp T:
1. Look up the LAST COMPLETED bar at 5m, 15m, 1h
2. Get the SFE state for that bar
3. Extract 4 features (dmi_diff, z_se, velocity, vol_rel)
4. Append to the 13D feature vector → 25D

**Key: no lookahead.** The 1h bar used is the LAST COMPLETED 1h bar,
not the one currently forming. If the 1m bar is at 10:37, the 1h bar
is the one that closed at 10:00. The partial 1h bar (10:00-10:37) is NOT used.

### ATLAS Data Availability
We have all TFs in ATLAS:
- IS: `DATA/ATLAS/5m/`, `DATA/ATLAS/15m/`, `DATA/ATLAS/1h/`
- OOS: `DATA/ATLAS_OOS/5m/`, `DATA/ATLAS_OOS/15m/`, `DATA/ATLAS_OOS/1h/`

SFE can batch_compute_states on each TF independently.

### Feature Extraction Change
```python
# Current: extract_features_13d(states, df) → (n, 13)
# Proposed: extract_features_25d(states_1m, df_1m, states_5m, df_5m,
#                                 states_15m, df_15m, states_1h, df_1h) → (n, 25)
```

For each 1m bar, binary search the higher-TF arrays to find the last
completed bar before the 1m timestamp.

### Model Change
- Input: (batch, lookback=10, 25) instead of (batch, lookback=10, 13)
- Architecture unchanged (Conv1d adapts to input width)
- Parameters increase: ~25K (from ~16K) — still well within safe range for 464K samples

### Label Change
- Labels stay 7D × N_HORIZONS (predict 1m features at t+h)
- The higher-TF features are INPUT context, not prediction targets
- The model uses 1h structure to make better 1m predictions

---

## Expected Impact

### Entry Quality
The model can learn: "1m DMI says LONG, but 1h z_se is at +3 (overbought) → skip"
Currently it enters blind to hourly structure.

### Hold Duration
The model can learn: "1m trend fading, but 5m and 1h still agree → hold"
Currently it exits on every 1m signal regardless of higher-TF context.

### Direction Accuracy
Currently 72.3% on 1m alone. With 1h dmi_diff as context:
- 1m LONG + 1h LONG → high confidence (both agree)
- 1m LONG + 1h SHORT → low confidence (fighting the structure)

### Trade Count
Should DECREASE (better filtering). The $1,609/day came from 271 trades/day.
With multi-TF context, fewer but higher-quality entries → same or better PnL.

---

## Risks

1. **Data alignment**: 1h bars close every 60 minutes. Between closes, the feature
   is stale (up to 59 minutes old). The model must learn that 1h features update
   slowly — this is actually an advantage (stability vs noise).

2. **Scale mismatch**: 1h dmi_diff has different scale than 1m dmi_diff
   (1h DMI uses 14 × 1h bars = 14 hours of smoothing). Should normalize
   each TF's features independently (z-score within TF).

3. **Training data**: need to load 4 TFs × 12 months for IS. ~4x memory during
   feature extraction. One-time computation, cached as .npy.

4. **Feature redundancy**: 5m dmi_diff might correlate with 1m dmi_diff (same
   underlying price). But at different smoothing windows they carry different info.
   Test with correlation matrix — drop if r > 0.9.

---

## Validation Plan

1. Build 25D extractor, cache features
2. Train on IS with walk-forward (same as current)
3. Compare OOS: 13D vs 25D
4. Check: did direction accuracy improve? Did trade count decrease?
5. Check: feature importance — which TF features matter most?
6. If 25D beats 13D → ship to live

---

## What NOT to Do

- Do NOT build separate TF workers or TBN consensus voting
- Do NOT add 1s features (too noisy for 1m model, different resolution)
- Do NOT add all 14 TFs (3 higher TFs is enough, diminishing returns)
- Do NOT use partial/forming bars from higher TFs (lookahead risk)
- Do NOT change the label structure (still predict 1m features)
- Do NOT change the model architecture beyond input width

---

## Baseline to Beat

```
TradeCNN 13D (current):  $1,609/day OOS, 271 trades/day, 24% WR
TradeCNN 25D (proposed): target $2,000+/day with fewer trades
```

---

## Files to Modify

| File | Change |
|------|--------|
| `training/train_trade_cnn.py` | Add `extract_features_25d()`, update `build_dataset()` |
| `core/trade_cnn.py` | Update `StatePredictor(n_features=25)` |
| `live/live_engine.py` | Add higher-TF state lookup in `_trade_cnn_predict()` |

No new files. No architectural changes. Just wider input.

# Rationale: Grounded TCN vs 192D TCN

## Context
A prior session designed a TCN with 192D input (12 TF × 16D features, 800K params).
This session grounded ALL features in base measurements (Price, Time, Volume) and
proved that 7D grounded features match or beat the 192D on OOS.

## Why 7D instead of 192D

### The Problem with 192D
- ANOVA (14 months): 8 of 12 SFE features drift 1.3-5.5x across regimes
- FM: 4.82x scale shift between IS (2025) and OOS (2026)
- Volume: 5.51x scale shift
- The model spends 80% of capacity learning WHICH features to ignore
- Features that work in IS drift in OOS → performance degrades

### The 7D Grounded Features
Each answers ONE named question from base measurements:

| Feature | Question | Base |
|---------|----------|------|
| dmi_diff | Who's winning? | Price highs/lows over Time |
| dmi_gap | How dominant? | abs(dmi_diff) |
| vol_rel | How much participation? | Volume / rolling avg |
| dir_vol | Is volume confirming direction? | sign(close-open) × volume |
| velocity | How fast is price moving? | Price change / Time |
| z_se | Is this deviation significant? | (Price - mean) / SE |
| price_accel | Is force changing? | velocity change |

### Results (CNN, OOS 39 trading days)
- 7D, lookback=10: $651/day, 33.5% WR, 5,274 trades
- 22D (full SFE): negative — features drift killed it
- DMI flipper (no ML): $208/day baseline

### Why Less Input Works Better
1. **No regime drift**: grounded features are regime-transferable by design
2. **More signal density**: 7 meaningful features vs 192 features where 180 are noise
3. **Less overfitting risk**: 25K params vs 800K params for same training set
4. **Every feature is explainable**: if the model fails, we know which feature broke

## What to Keep from the 192D TCN Spec
1. ✅ Dilated causal convolutions (multi-scale without depth)
2. ✅ Residual connections (stable deep training)
3. ✅ 3-head output (direction + hold_time + confidence)
4. ✅ Causal constraint (no lookahead)
5. ✅ WaveNet-style architecture

## What to Change
1. 192D → 7D input (grounded features)
2. 800K → 25K params (scaled to clean input)
3. 50-bar → 10-30 bar lookback (data showed diminishing returns past 20)
4. 30K auto-seed samples → 464K ATLAS bars (15x more data)
5. Train with PnL-weighted loss (reward profitable predictions, penalize losses 2x)

## Feature Tree Principle
Features must be traceable to base measurements in ≤3 layers:
- **Level 1 (Primary)**: Price, Time
- **Level 2 (Secondary)**: velocity, DMI, volume, std, mean
- **Level 3 (Tertiary)**: z_se, acceleration, variance_ratio, price×volume

If a feature can't name its parents and the question it answers, it doesn't belong.
"Can you explain what this measures without reading the code?" — if no, remove it.

## Random Forest Validation
RF with 7D features shows feature importance ranking — proves which features
the model actually uses. If RF matches CNN performance, the patterns are
in the features, not in deep architecture. Start simple, add depth only if needed.

## Base Measurements (from session research)
- **Price**: position (where)
- **Time**: when
- **Volume**: reaction measurement (market's response to Price at Time)
- All other features are derivatives. Each derivative must have a PURPOSE.
- Standard deviation and variance are valid secondary measurements (distribution shape)
- DMI is valid despite being a derivative — it measures buyer/seller battle, not just price direction
- Coherence is REBUILT as agreement count across grounded features (not entropy of oscillations)

## Key Research Findings
- Next-bar prediction from velocity alone: 50.77% (coin flip) on 1.5M bars
- Peak detection (velocity flip + vol collapse): +5% edge at detected peaks
- Human-marked peaks: 98.8% have DMI widening (the fundamental signature)
- DMI smoothed cross + TP=10 + SL=40: $208/day across 14 months (proven baseline)
- Adding reg_mean exit + vol_spike exit: $400/day (14 months)

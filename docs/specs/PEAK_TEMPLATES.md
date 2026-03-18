# Peak-Anchored Templates — Phase 1 Rebuild

**Priority**: Next major feature
**Date**: 2026-03-18

## Concept

Replace z-score threshold discovery with peak-anchored discovery.
Instead of "what does the market look like at z=2.0?", ask
"what does the market look like 10 bars BEFORE a reversal?"

## Current Flow (z-score discovery)

```
Phase 1: scan all bars → find z-score crossings → build templates
Forward:  candidate bar → match to template → trade
```

Problem: z-score crossings are arbitrary. A z=2.0 on a trending day
means something completely different than z=2.0 on a ranging day.
Templates mix these contexts.

## Proposed Flow (peak-anchored)

```
Phase 1: scan all bars → detect peaks (P_center reversal) →
         look back 10 bars → extract approach shape → cluster → templates
Forward: current 10-bar shape → match to template → system knows:
         - expected MFE (how far the reversal goes)
         - expected duration (how long)
         - direction bias (which way)
         - calibrated SL/TP from that specific approach pattern
```

## Why This Works

1. Peak detection has 83% detection rate, 85% precision, 0% false alarms
2. Templates encode the LEADUP to a reversal, not an arbitrary state
3. No lookahead: peaks labeled from completed data, templates from the lookback
4. Each peak type gets its own statistics (kills generic tid=-100)
5. Approach shapes should cluster well — similar leadups → similar outcomes

## Design

### Step 1: Label peaks in historical data
- Run peak detection on full ATLAS (same as current: P_center + F_momentum)
- Record peak bar index, direction (up→down or down→up), magnitude

### Step 2: Extract approach shapes
- For each peak at bar N, extract bars [N-K, N] where K = lookback window
- Feature vector per bar: same 16D canonical features
- Shape = K x 16D flattened, or use summary statistics (slope, curvature, etc.)
- Alternative: use existing 6D lookback geometry (already in shape_primitives.py)

### Step 3: Cluster approach shapes
- Same pipeline: UMAP + HDBSCAN, or K-Means per TF
- Each cluster = one "approach template"
- Statistics per cluster: avg MFE, avg MAE, WR, avg duration, direction split

### Step 4: Forward pass matching
- Current 10-bar window → extract shape → match to nearest cluster
- If match distance < threshold AND peak signal is firing → enter
- SL/TP/exits calibrated from the matched template's stats

## Open Questions

1. **Lookback window**: 10 bars fixed, or TF-anchored?
   - 10 bars at 15s = 2.5 min (scalps)
   - 10 bars at 1m = 10 min (swings)
   - Recommendation: anchor to discovery TF, always 10 bars of that TF

2. **Feature dimensionality**: 10 x 16D = 160D (too high for clustering?)
   - Option A: flatten → PCA/UMAP → cluster
   - Option B: summary stats (slope, acceleration, z-range, vol profile)
   - Option C: reuse 6D lookback geometry from shape_primitives.py

3. **Peak + template agreement**: require BOTH peak signal AND template match?
   Or template match alone (shape says "this looks like a pre-reversal")?
   - Recommendation: both. Template match = "seen this before". Peak signal =
     "it's happening now". Belt and suspenders.

4. **Coexistence with compressed candidates**: do peak templates replace
   ALL templates, or do compressed-state templates survive?
   - Recommendation: peak templates are the primary library.
     Compressed candidates become the fallback (no template match,
     but state says "something is happening").

## Dependencies

- Peak detection (DONE — core/bar_processor.py)
- 6D lookback geometry (DONE — core/shape_primitives.py)
- Phase 1 discovery scanner (EXISTS — needs rewiring)
- Forward pass template matching (EXISTS — needs shape matching)

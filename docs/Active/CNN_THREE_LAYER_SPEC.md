# CNN Three-Layer Architecture Spec

## Overview
Three CNNs in sequence. Each has a distinct job. NMP is the bootstrap — CNNs discover
the real physics.

## Core Insight
Regret tells us which trades were profitable. Among those profitable trades, there are
distinct 79D physics patterns. CNN 1 discovers and groups those patterns. CNN 2 and 3
manage trades knowing which pattern they belong to.

## Pipeline
```
NMP (bootstrap) -> trades -> regret -> pool of profitable trades
  -> CNN 1 (pattern discovery): what kind of good trade is this?
     -> per tier: CASCADE / KILL_SHOT / BASE_NMP
     -> contrastive/autoencoder on profitable trade entry 79D
     -> cluster embeddings -> named setup patterns
     -> approach analysis: what 10-bar ramp leads into each pattern?
  -> CNN 2 (exit): given setup pattern + trade context, HOLD or EXIT?
  -> CNN 3 (risk): given setup pattern + trade context, RECOVER or DEAD?
```

## CNN 1 — Pattern Discovery (per tier)

### Purpose
Among all profitable trades (regret-confirmed), discover the distinct 79D physics
patterns at entry. NOT a classifier (good/bad) — all inputs are already good.
The question: "What KIND of good trade is this?"

### Architecture
```
Entry 79D -> feat_to_grid (6x13) -> CNN encoder -> embedding (16D)
```
No classification head. The embedding IS the output. Trained with contrastive loss
(similar profitable patterns close together, different patterns apart) or autoencoder
(reconstruct the 79D grid from the embedding — embedding captures the essence).

### Per-Tier Training
Three separate encoders:
- **CNN 1a**: CASCADE profitable trades only
- **CNN 1b**: KILL_SHOT profitable trades only
- **CNN 1c**: BASE_NMP profitable trades only

Each tier's patterns are isolated. A cascade pattern is different physics than base NMP.

### Training Data
From Phase 1 (blended NMP) + Phase 2 (regret):
- Filter: only trades where regret says profitable (optimal PnL > 0)
- **Entry 79D**: the 79D at entry point (or regret's optimal early entry point)
- **Approach path**: 10 bars of 79D leading into the entry (already stored in
  `_entry_approach` by BlendedEngine)

### Pattern Discovery (Post-Training)
1. Run all profitable tier trades through CNN encoder -> collect 16D embeddings
2. K-means on embeddings (k=auto via silhouette score, max 10 per tier)
3. Each cluster = a distinct physics pattern
4. Label format: `{TIER}.{cluster_id}` (e.g. CASCADE.3, BASE_NMP.7)

### Approach Data (Loaded from Feature Files, Not Buffer)
Regret's early entry can be up to 120 bars (10 min) before NMP trigger. Storing that
in a buffer wastes memory. Instead, CNN 1 loads the approach at training time:

```
Corrected trade has: day, timestamp, early_bars
CNN 1 training:
  1. Load FEATURES_79D_5s_v2/{day}.parquet
  2. Find NMP trigger timestamp -> bar index
  3. Go back early_bars -> early entry bar index
  4. Grab 10 bars before early entry from the feature file
  -> full approach trajectory, no buffer limit
```

No recursion risk. Regret runs once. Approach is read directly from ATLAS features.

### Approach Analysis (per pattern)
For each discovered pattern, analyze the 10-bar approach trajectory:
- What was the 79D doing in those 10 bars before the (possibly early) entry?
- Is there a signature ramp? Sudden shift? Consolidation?
- Two trades with identical entry 79D but different approaches = different physics

This distinguishes:
- "Slow grind to z=-2" vs "sharp spike to z=-2" — same entry, different approach
- "1h aligned for 5 bars" vs "1h just flipped" — timing matters

### Per-Pattern Profile
For each pattern cluster:
- N trades, WR, avg PnL, worst day
- Direction bias (% long vs short)
- Time-of-day profile
- Feature centroid (which 79D features define this pattern)
- Approach signature (avg 79D trajectory in 10 bars before entry)
- Representative trades (3-5 closest to centroid)

### Kill Switch
Each pattern can be individually enabled/disabled:
```json
{
  "CASCADE.0": true,
  "CASCADE.1": true,
  "KILL_SHOT.0": true,
  "BASE_NMP.0": true,
  "BASE_NMP.3": false
}
```

## CNN 2 — Exit Timing

### Purpose
During trade: predict HOLD or EXIT at each 1m bar.

### Input
- Current 79D (6x13 grid)
- Context: bars_held, PnL, peak_pnl, direction, **pattern_label**
- Pattern label from CNN 1 — different patterns have different hold profiles

### Training Data
From Phase 4 (entries filtered by CNN 1 patterns) + Phase 5 (regret):
- At each bar during trade: 79D + context
- Label: regret says HOLD (not yet at optimal exit) or EXIT (at/past optimal exit)

### Architecture
```
79D grid (6x13) + context (bars, pnl, peak, dir, pattern) -> CNN -> HOLD / EXIT
```

## CNN 3 — Risk (Loser Detection)

### Purpose
When PnL < 0: predict RECOVER or DEAD.

### Input
Same as CNN 2 but only fires when PnL negative:
- Current 79D (6x13 grid)
- Context: bars_held, PnL, peak_pnl, depth_from_peak, direction, **pattern_label**

### Training Data
From trade paths where PnL went negative:
- Label: did the trade eventually recover to profit? RECOVER : DEAD

### Architecture
```
79D grid (6x13) + context -> CNN -> RECOVER / DEAD
```

## Data Flow

```
Phase 1: Blended NMP (no CNN) -> trades with tiers + approach paths
Phase 2: Regret -> filter profitable trades, optimal entries/exits
Phase 3: Train CNN 1a/1b/1c encoders on profitable trades per tier
Phase 3b: Cluster embeddings -> pattern labels per tier
Phase 3c: Approach analysis -> 10-bar regret per pattern
Phase 4: Run entries with CNN 1 pattern filter -> trades with pattern labels
Phase 5: Regret on pattern trades -> optimal exit timing
Phase 6: Train CNN 2 (exit) with pattern_label as context
Phase 7: Train CNN 3 (risk) with pattern_label as context
Phase 8: Forward pass IS + OOS with full system
```

## Key Design Decisions

1. **CNN 1 is not a classifier**: it's an encoder. All inputs are profitable trades.
   The question is grouping, not filtering.
2. **Per-tier encoders**: patterns don't cross tiers.
3. **Approach path matters**: 10-bar ramp before entry fingerprints the setup.
   Same entry 79D with different approach = different pattern.
4. **Pattern label flows downstream**: CNN 2/3 know which pattern they're managing.
5. **Kill switch per pattern**: disable bad patterns without retraining.
6. **Direction stays physics-based**: z > 0 = short, z < 0 = long. CNN 1 doesn't
   predict direction — regret already validated the physics direction works.

## Success Criteria
- CNN 1: discovers >3 distinct patterns per tier with statistically different profiles
- Approach analysis: patterns have visually distinct 79D trajectories
- CNN 2: >90% accuracy (setup-aware exit timing)
- CNN 3: >80% accuracy (setup-aware risk detection)
- Forward pass: beat current $613/day OOS
- Pattern kill switch: disabling worst pattern improves OOS by >5%

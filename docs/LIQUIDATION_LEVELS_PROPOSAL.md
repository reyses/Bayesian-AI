# Proposal: Liquidation Level Integration

## For Next Session — Pick Up Here

### Context
We discovered that the three-body problem in the quantum field engine maps directly
to real-world algorithmic liquidation pools. The levels are visible on the daily chart
as clustered swing peaks. The physics matrix confirms them. Local patterns describe
how price interacts with them. They nest fractally across timeframes.

Full insight doc: `docs/LIQUIDATION_LEVELS_INSIGHT.md`
CNN research so far: `tools/cnn_pattern_model.py` + Analysis R in `tools/waveform_standalone.py`
Daily chart with peak levels: `tools/plots/standalone/daily_levels.png`

### What We Tried (CNN Pattern Builder)
- Conv1D on 64-bar OHLCV windows, 7 pattern classes from seed primitives
- Direction accuracy: 78-82% (strong), pattern classification: 35-44% (weak)
- Tried adding context channels (rolling avg H/L, sigma bands, regime sections)
- None of the pre-computed context helped — the levels are visual/intuitive
- Key realization: the CNN should learn the levels from the chart, not from formulas

### Proposed Next Steps

#### Phase 1: Manual Levels + Validation (QUICK WIN)
1. User identifies 5-10 active liquidation levels from the daily chart
2. Store in `checkpoints/price_levels.json`:
   ```json
   {
     "levels": [19640, 20710, 22030, 23030, 23904, 24888, 25935],
     "updated": "2026-03-04",
     "notes": "From daily swing peak clusters"
   }
   ```
3. For each 15m bar, compute 2 simple features:
   - `dist_above`: ticks to nearest level above current price
   - `dist_below`: ticks to nearest level below current price
4. Add these as context scalars to the dual-path CNN
5. Run Analysis R with 120 days — does direction/pattern accuracy improve?
6. Also test: do the live engine's gate cascade trades perform better near levels?

#### Phase 2: Physics Anchoring
- Replace the quantum field engine's statistical mean with the nearest
  liquidation level as equilibrium
- z-score becomes: distance from nearest level (not distance from rolling mean)
- Barrier height: computed between adjacent levels (not arbitrary 3σ)
- Tunnel probability: P(breaking through to the next level)
- This makes the existing 16D features physically meaningful

#### Phase 3: Fractal Level Nesting
- Compute levels at each TF (daily, 4h, 1h, 15m)
- Wire into the DNA tree: parent level = macro attractor, child levels = micro-pools
- The `parent_z` feature becomes: distance from parent's nearest level
- Captures the fractal gravity: micro-pools orbit macro attractors

#### Phase 4: CNN on Daily Chart Images (Visual Learning)
- Instead of pre-computing levels, render actual candlestick chart images
- Conv2D on chart images — the CNN learns to see levels like a human
- Input: 60-day daily chart as image + current 15m pattern window
- This is the ultimate goal: the system learns to "eyeball" the chart

### Key Files
- `tools/cnn_pattern_model.py` — PatternCNN dual-path model (5ch conv + 3 scalar context)
- `tools/waveform_standalone.py` — Analysis R block (line ~6335)
- `docs/LIQUIDATION_LEVELS_INSIGHT.md` — Full insight writeup
- `core/quantum_field_engine.py` — Three-body physics (anchor point for Phase 2)
- `training/fractal_dna_tree.py` — Fractal TF tree (anchor for Phase 3)

### Decision for User
Start with Phase 1 (manual levels JSON + distance features) — it's the fastest
way to test if liquidation level context actually improves the CNN and the live
trading system. If it works, Phase 2-4 follow naturally.

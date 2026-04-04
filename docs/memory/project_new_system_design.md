---
name: New System Design (79D + NN + Half-Life)
description: Architecture spec for replacing advance engine — 79D features, strategy router NN, half-life exit
type: project
---

## Status: DESIGNED, NOT YET IMPLEMENTED (2026-04-03)

## Architecture
- **79D feature vector**: 10 features x 6 TFs + 3 helpers x 6 TFs + time_of_day
- **Strategy Router NN**: 79D → direction + hold duration (half-life)
- **Unified Exit**: envelope decay with NN-predicted half-life, modulated by survival score + giveback
- **Execution**: 5s atomic bar, 1m decision anchor

## Key Specs
- `docs/Active/FEATURE_VECTOR_79D_SPEC.md` — full feature vector + NN architecture
- `docs/Active/EXIT_MATH_ANALYSIS.md` — exit module math, unified exit concept, markers for resumption

## Core Insight
5s is the Planck constant. TFs are aggregation windows. The NN learns signal half-life — how many 5s bars of noise to hold through before the edge decays.

## Three Exits → One Function
1. Envelope Decay: `exp(-ln2 * t / hl)` — the base decay
2. Survival Stop: `room * trend * conviction * alignment * momentum` — modulates HL
3. Peak Giveback: volume-adaptive threshold — accelerates HL

Unified: NN predicts initial HL, survival score compresses it live, giveback is emergency accelerator.

## Why Advance Engine Was Abandoned
- 4,721 trades, ALL template -100 (peak reversal), 50.1% WR = coin flip
- Zero template matches in forward pass
- Templates cluster on 16D features the ticker never uses
- Nightmare ticker (simple rules, z_se + vr) found real edges
- Too many moving parts: peak detect, 1m sensor, cat brain, 7 exit types

## Next Steps
1. Implement `extract_79d()` from SFE states
2. Build training label generator (forward PnL at 6 durations per bar)
3. Train strategy router NN
4. Wire unified exit (envelope + survival + giveback)
5. Test on OOS Feb 2026 clean data

---
name: Quantum State Reconnection — Critical Architecture Direction
description: The system was designed as a probabilistic quantum model. Gates broke it. Reconnect wave function to scoring. DO NOT FORGET THIS.
type: project
---

## CRITICAL: The System is Quantum — Not Statistical

The Bayesian-AI trading system was designed from first principles as a **quantum-mechanical
model of price behavior**. Price is a particle in a three-body gravitational field. The
"statistical" renaming was a metaphor purge that kept the math but disconnected the
conceptual framework from the decision logic.

**Why:** User's key insight (2026-03-15): "Price behaves like the electron position in an
atom — you can't actually know where it is because by the time you observe it, it's in
another place. That's the reason the old names are quantum and Roche — this should be
a probabilistic system."

## The Three Theories

1. **Nightmare Protocol** — O-U mean reversion + Roche limits (2σ/3σ boundaries)
2. **Three-Body Problem** — Macro/Meso/Micro alignment solved probabilistically via wave function
3. **Resonance Cascade** — Cross-TF synchronization → disable TP, ride the shockwave

## What's Computed Every Bar But UNUSED in Scoring

These are in `MarketState` RIGHT NOW but disconnected from entry/exit decisions:
- `prob_center`, `prob_upper`, `prob_lower` — wave function collapse probabilities
- `entropy_normalized` — chaos measure (high = three-body misalignment)
- `coherence` — superposition measure (low = collapsed = decisive)
- `tunnel_probability` — P(mean reversion) via O-U Monte Carlo
- `reversion_probability` — used in tunnel gate but not scoring

## What's Missing

- **Lyapunov exponent** — λ<0 fade edges, λ>0 ride breakout (not computed)
- **Fractal diffusion** — σ = σ_base × (v_micro/v̄_macro)^H (not implemented)
- **PID control force** — Kp*e + Ki*∫e + Kd*de/dt (term_pid exists but unused in forces)
- **is_resonance_cascade** flag — cross-TF ADX>30 + Hurst>0.55 + aligned DMI
- **Cascade mode** — suppress TP, trail via survival_stop only until Hurst decays

## What Gates Broke

The gate cascade imposes **binary decisions on continuous probabilities**:
- Hurst < 0.50 → BLOCK (but 0.49 and 0.51 are the same state)
- Conviction < 0.48 → BLOCK (rubber stamp, barely filters)
- Distance > 3.0 → BLOCK (proximity ≠ probability)

Should be: **P(success) = f(wave_function, tunnel_prob, coherence, conviction, funnel)**
No gates. One probability. Best P(success) wins.

## Reference Documents (all in docs/reference/)

1. `quantum_field_engine_original.py` — 919 lines, the pre-purge engine with all math
2. `QUANTUM_STATE_REFERENCE.md` — Gemini's theoretical spec (three theories + cascade protocol)
3. `NIGHTMARE_PROTOCOL.md` — Master equation: dX = O-U + fractal_diffusion + PID + jump
4. `THREE_THEORIES.md` — My mapping of quantum concepts to current codebase

## The Decision Funnel (validated by research)

- Funnel tree trained on I-MR auto seeds: **85% accuracy** predicting regime starts
- Key features: tight price_range (65%) + high sigma (17%) = energy coiling → breakout
- This IS the pre-resonance state detection
- Spec: `docs/Active/SPEC_DECISION_FUNNEL.md`

## Next Session Action Items

1. **Run fresh**: `python training/trainer.py --fresh --lookback` (brain stale)
2. **Read quantum_field_engine_original.py** — map which wave function computations
   survive in current statistical_field_engine.py
3. **Add is_resonance_cascade** to MarketState or BeliefState
4. **Replace gate cascade scoring** with P(success) from wave function probabilities
5. **Implement cascade mode** — suppress TP when resonance detected

## PFMEA Top Issues (from this session)

| RPN | Issue | Status |
|-----|-------|--------|
| 648 | Score competition ignores conviction | OPEN — needs P(success) scoring |
| 392 | Conviction gate after competition | OPEN — wrong cascade position |
| 336 | Giveback IS→OOS flip | INVESTIGATE |
| 300 | Brain reject dead (0 blocked) | OPEN |
| 243 | BreakevenLock 4-tick activation | **FIXED** (TrailingStop rewrite) |

## User Preferences (from this session)

- **Challenge ideas HARD** — the user expects resistance, not compliance
- **Workers decide** — no artificial thin-market skips, let the system filter
- **SL is last resort** — tolerance interval from MAE distribution, not fixed
- **Probabilistic, not deterministic** — no magic numbers, no binary gates
- **The physics names were correct** — don't strip the quantum vocabulary

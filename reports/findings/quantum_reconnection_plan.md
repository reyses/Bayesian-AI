# Quantum Reconnection Plan — Replace Gates with P(success)
> Generated: 2026-03-16 by research agent
> Status: PLAN — ready for implementation

---

## PART 1: MARKETSTATE FIELD MAPPING TO GATES

**Gate 0 (Physics Rules)** → Probabilistic quality score:
- `hurst_exponent` → soft penalty (0.25x if hurst < 0.40), not hard block
- `reversion_probability` → continuous P(mean reversion), not binary tunnel gate
- `F_momentum` vs `mean_reversion_force` → soft ratio, not binary override
- `lyapunov_exponent` → computed but never used; enables cascade detection
- `entropy_normalized` → computed but unused in scoring

**Gate 1 (Template Match)** → Keep hard, add soft penalty:
- `dist < 3.0` stays as hard filter
- Below 2.5: `P(match) = exp(-dist²/2)` multiplied into score

**Gate 2 (Brain Reject)** → Probabilistic belief:
- Replace binary `should_fire()` with continuous `brain.get_dir_probability(tid)`
- Soft threshold at P(profit) < 0.30

**Gate 3 (Conviction)** → Continuous weighting:
- Replace binary `conviction > 0.48` with conviction as multiplier in score
- Soft rejection at conviction < 0.30

**Gate 4 (Momentum)** → Probabilistic alignment:
- Replace binary sign check with sigmoid: `P(align) = 1/(1+exp(-F_mom * dir_scale))`
- Soft 0.6x penalty instead of hard block

---

## PART 2: P(SUCCESS) FORMULA

Original: `P_i > 0.75 AND tunnel_prob > 0.60 AND entropy_low → TRADE`

Proposed:
```
P(success) = w_wave * P_wave
           + w_tunnel * P_tunnel
           + w_entropy * (1 - entropy_normalized)
           + w_conviction * belief.conviction
           + w_brain * |dir_long_prob - dir_short_prob|
           + w_momentum * P(momentum_align)
           + w_template * exp(-dist²/2)
           + w_cascade * is_resonance_cascade

where:
  P_wave = max(prob_center, prob_upper, prob_lower)
  P_tunnel = reversion_probability
  P(momentum_align) = sigmoid(F_momentum * sign(direction))
  is_resonance_cascade = (macro_hurst > 0.55 AND lyapunov > 0 AND coherence > 0.90)
```

Starting weights (tune from data):
- w_wave=0.20, w_tunnel=0.15, w_entropy=0.15, w_conviction=0.15
- w_brain=0.10, w_momentum=0.10, w_template=0.10, w_cascade=0.05

---

## PART 3: MISSING VARIABLES

1. **is_resonance_cascade** — boolean flag, compute per bar
2. **Lyapunov exponent** — rolling divergence analysis
3. **Fractal diffusion** — σ_fractal = σ_base × (v_micro/v̄_macro)^H
4. **Coherence** — explicit field (= 1 - entropy_normalized)
5. **Full tunnel probability** — Monte Carlo O-U with 1000 stochastic paths

---

## PART 4: RISKS OF REMOVING GATES

**Keep hard**: Gate 1 (template dist < 3.0), Gate 2 (brain prob > 0.30)
**Soften**: Gate 0 (hurst), Gate 3 (conviction), Gate 4 (momentum)
**Remove**: Regime compatibility (convert to score adjustment)

---

## PART 5: IMPLEMENTATION PHASES

### Phase 1: Instrumentation (1 session)
- Add is_resonance_cascade to MarketState
- Add explicit coherence field
- Verify lyapunov_exponent, prob_center/upper/lower computed
- Validate all quantum variables are live

### Phase 2: Soft Thresholds (1 session)
- Convert 4 hard gates to soft penalties
- Keep gates 1 & 2 as hard
- Run IS forward pass, measure WR/PnL

### Phase 3: Probabilistic Scoring (1-2 sessions)
- Implement _compute_quantum_score()
- Use in score competition: score = -p_success
- A/B test against current scoring

### Phase 4: Cascade Mode (1 session)
- Detect is_resonance_cascade → suppress TP
- Trail via survival_stop + tidal_wave only
- Test on known cascade days

### Phase 5: Full Replacement (1-2 sessions)
- Replace entire _gate_check() with probabilistic flow
- Update reports to show P(success) components

### Phase 6: Calibration (ongoing)
- P(success) vs actual WR calibration curve
- Weight optimization

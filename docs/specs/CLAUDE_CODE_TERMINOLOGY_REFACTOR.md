# CLAUDE CODE INSTRUCTIONS: Terminology Refactor
# Strip quantum/physics metaphors → standard statistical + trading language
# Date: March 6, 2026

## CONTEXT

The codebase uses quantum mechanics / astrophysics metaphors to describe
standard statistical concepts. This adds cognitive load and obscures bugs
(the short bias lived undetected because "z > 0 → SHORT" was buried under
"particle at upper Roche limit → reversion expected").

This refactor renames variables, classes, and methods to use the language
that quant traders, statisticians, and ML engineers actually use.

**NO LOGIC CHANGES.** Every computation stays identical. Only names change.

---

## MASTER RENAME MAP

### Class Names

| Old | New | Reason |
|-----|-----|--------|
| `ThreeBodyQuantumState` | `MarketState` | It's a market state vector |
| `QuantumFieldEngine` | `StatisticalFieldEngine` | Computes regression, z-scores, probabilities |
| `QuantumBayesianBrain` | `BayesianBrain` (already exists, merge) | The "Quantum" prefix adds nothing |
| `QuantumRiskEngine` | `MonteCarloRiskEngine` | It runs Monte Carlo simulations |

### Field Names (MarketState, formerly ThreeBodyQuantumState)

| Old Field | New Field | What It Actually Is |
|-----------|-----------|---------------------|
| `center_position` | `regression_center` | 21-bar linear regression mean |
| `upper_singularity` | `upper_band_2sigma` | +2σ Standard Error Band |
| `lower_singularity` | `lower_band_2sigma` | -2σ Standard Error Band |
| `event_horizon_upper` | `upper_band_3sigma` | +3σ Standard Error Band |
| `event_horizon_lower` | `lower_band_3sigma` | -3σ Standard Error Band |
| `particle_position` | `price` | Current price |
| `particle_velocity` | `velocity` | Price change per bar (Δprice) |
| `F_reversion` | `mean_reversion_force` | Force pulling price back to center |
| `F_upper_repulsion` | `upper_band_pressure` | Repulsion from upper band |
| `F_lower_repulsion` | `lower_band_pressure` | Repulsion from lower band |
| `F_momentum` | `momentum_force` | Volume-weighted velocity / σ |
| `F_net` | `net_force` | Sum of all forces |
| `amplitude_center` | `prob_amplitude_center` | √P(center) |
| `amplitude_upper` | `prob_amplitude_upper` | √P(upper) |
| `amplitude_lower` | `prob_amplitude_lower` | √P(lower) |
| `P_at_center` | `prob_at_center` | Boltzmann probability at center |
| `P_near_upper` | `prob_near_upper` | Boltzmann probability near +2σ |
| `P_near_lower` | `prob_near_lower` | Boltzmann probability near -2σ |
| `coherence` | `entropy_normalized` | Entropy / ln(3), 0=collapsed, 1=uniform |
| `pattern_maturity` | `pattern_maturity` | (keep — already clear) |
| `momentum_strength` | `momentum_strength` | (keep — already clear) |
| `structure_confirmed` | `structure_confirmed` | (keep) |
| `cascade_detected` | `cascade_detected` | (keep — velocity cascade) |
| `spin_inverted` | `reversal_confirmed` | Candlestick reversal at band |
| `lagrange_zone` | `band_zone` | Which σ zone: INNER, OUTER_UPPER, OUTER_LOWER |
| `stability_index` | `regime_stability` | (keep concept, clearer name) |
| `tunnel_probability` | `reversion_probability` | P(price returns to center) |
| `escape_probability` | `breakout_probability` | P(price breaks through 3σ) |
| `barrier_height` | `reversion_barrier` | OU potential energy at current z |
| `sigma_fractal` | `regression_sigma` | Standard deviation of regression residuals |
| `term_pid` | `control_force` | PID algorithmic market-maker force |
| `oscillation_coherence` | `oscillation_tightness` | 1=tight periodic, 0=noisy |
| `lyapunov_exponent` | `stability_exponent` | (keep concept) |
| `market_regime` | `market_regime` | (keep — already clear) |
| `resonance_coherence` | `alignment_score` | Phase alignment across TFs |
| `cascade_probability` | `flash_move_probability` | P(sudden large move) |
| `amplitude_multiplier` | `energy_multiplier` | Energy amplification factor |
| `resonance_type` | `alignment_type` | NONE/PARTIAL/FULL/CRITICAL |
| `fractal_alignment_count` | `multi_tf_alignment_count` | Scales showing same signal |
| `fractal_confidence` | `alignment_confidence` | LOW/MEDIUM/HIGH/EXTREME |
| `fractal_edge` | `alignment_edge` | 0-1 scale agreement |
| `time_at_roche` | `time_at_band_extreme` | Time spent at ±2σ |
| `timeframe_macro` | `timeframe_macro` | (keep) |
| `timeframe_micro` | `timeframe_micro` | (keep) |

### Zone Names (band_zone, formerly lagrange_zone)

| Old Zone | New Zone | Meaning |
|----------|----------|---------|
| `L1_STABLE` | `INNER` | Price between ±1σ (noise zone, no trade) |
| `L2_ROCHE` | `UPPER_EXTREME` | Price at +2σ or beyond (overbought) |
| `L3_ROCHE` | `LOWER_EXTREME` | Price at -2σ or beyond (oversold) |
| `CHAOS` | `TRANSITION` | Price between 1σ and 2σ (approaching extremes) |

### Method Names

| Old | New | File |
|-----|-----|------|
| `calculate_three_body_state()` | `calculate_market_state()` | quantum_field_engine.py |
| `get_quantum_probability()` | `get_state_probability()` | bayesian_brain.py |
| `should_fire_quantum()` | `should_fire_state()` | bayesian_brain.py |
| `get_trade_directive()` | `get_trade_directive()` | (keep — already clear) |
| `ThreeBodyQuantumState.null_state()` | `MarketState.null_state()` | three_body_state.py |
| `batch_compute_states()` | `batch_compute_states()` | (keep — already clear) |

### Pattern Type Names

| Old | New | Meaning |
|-----|-----|---------|
| `ROCHE_SNAP` | `BAND_REVERSAL` | Velocity spike at ±2σ band |
| `STRUCTURAL_DRIVE` | `MOMENTUM_BREAK` | Strong trend confirmed by structure |

### File Names

| Old | New |
|-----|-----|
| `core/three_body_state.py` | `core/market_state.py` |
| `core/quantum_field_engine.py` | `core/field_engine.py` |
| `core/risk_engine.py` | `core/mc_risk_engine.py` |

### Comment / Docstring Replacements

| Old Phrase | New Phrase |
|------------|-----------|
| "Three-body gravitational" | "Three-band regression" |
| "quantum wave function" | "probability distribution" |
| "wave function collapse" | "probability collapse (decisive signal)" |
| "Roche limit" | "2σ band boundary" |
| "event horizon" | "3σ extreme boundary" |
| "tunneling probability" | "mean reversion probability" |
| "escape probability" | "breakout probability" |
| "particle" | "price" |
| "singularity" | "band boundary" |
| "Nightmare Protocol" | "Regression Field Model" |
| "Nightmare Field Equation" | "Statistical Field Equation" |
| "tidal forces" | "regression forces" |
| "quantum mechanics" | "probability model" |
| "superposition" | "uncertain state (between bands)" |
| "Lagrange point" | "equilibrium zone" |
| "decoherence" | "signal clarity / entropy" |
| "Body 1 (Center Star)" | "Center (regression mean)" |
| "Body 2 (Upper Singularity)" | "Upper Band (+2σ)" |
| "Body 3 (Lower Singularity)" | "Lower Band (-2σ)" |
| "quantum state" | "market state" |
| "wave function psi" | "probability distribution P" |
| "measurement operators" | "confirmation signals" |

---

## EXECUTION STRATEGY

This is a large refactor. Execute in this order to avoid circular breakage:

### Phase A: Core State (breaks nothing downstream until imports updated)

```
1. Copy core/three_body_state.py → core/market_state.py
2. In market_state.py: rename class + all fields per map above
3. In market_state.py: update __hash__, __eq__, get_trade_directive
4. In market_state.py: update all docstrings and comments
5. In core/three_body_state.py: keep old class as thin wrapper
   that inherits from MarketState (backward compat during migration)
```

**The backward-compat wrapper (temporary, in three_body_state.py):**

```python
"""DEPRECATED: Use core.market_state.MarketState instead."""
from core.market_state import MarketState

# Alias for backward compatibility during migration
ThreeBodyQuantumState = MarketState
```

This lets ALL existing imports keep working while you rename one file at a time.

### Phase B: Engine

```
6. Copy core/quantum_field_engine.py → core/field_engine.py
7. Rename class: QuantumFieldEngine → StatisticalFieldEngine
8. Update all field references to new names (center_position → regression_center, etc.)
9. In quantum_field_engine.py: keep as thin import wrapper:
   from core.field_engine import StatisticalFieldEngine as QuantumFieldEngine
```

### Phase C: Risk Engine

```
10. Rename core/risk_engine.py → core/mc_risk_engine.py
11. Rename class: QuantumRiskEngine → MonteCarloRiskEngine
12. Keep old import path working via alias
```

### Phase D: Brain

```
13. In bayesian_brain.py: rename QuantumBayesianBrain methods
14. get_quantum_probability → get_state_probability
15. should_fire_quantum → should_fire_state
```

### Phase E: Downstream Consumers (one file at a time)

```
16. training/timeframe_belief_network.py — update field references
17. training/fractal_discovery_agent.py — ROCHE_SNAP → BAND_REVERSAL, STRUCTURAL_DRIVE → MOMENTUM_BREAK
18. training/fractal_clustering.py — update extract_features() field access
19. training/orchestrator.py — update all references
20. training/orchestrator_worker.py — update simulate_trade_standalone
21. training/wave_rider.py — update Position.entry_layer_state references
22. training/batch_regret_analyzer.py — update state field access
23. live/live_engine.py — update all references
24. live/bar_aggregator.py — update engine references
25. core/context_detector.py — update state field checks
26. core/adaptive_confidence.py — update state references
27. core/exploration_mode.py — update state references
28. core/cuda_physics.py — update kernel comments (no logic change)
29. core/cuda_pattern_detector.py — no changes needed
```

### Phase F: Cleanup

```
30. Remove backward-compat wrappers once all files updated
31. Delete core/three_body_state.py (replaced by market_state.py)
32. Delete core/quantum_field_engine.py (replaced by field_engine.py)
33. Run full test suite
34. Run forward pass, verify identical output
```

---

## CRITICAL RULES

1. **LOGIC STAYS IDENTICAL.** No formula changes. No threshold changes. 
   No algorithm changes. Only names.

2. **BACKWARD-COMPAT WRAPPERS FIRST.** The old class name must keep working
   via import alias until EVERY consumer is updated. Break nothing.

3. **ONE FILE AT A TIME.** Update one consumer, run tests, commit.
   Do not batch multiple file changes — if something breaks, you need
   to know which file caused it.

4. **grep BEFORE declaring done.** After Phase F:
   ```bash
   grep -rn "three_body\|ThreeBody\|quantum\|Quantum\|Roche\|roche\|singularity\|particle_position\|particle_velocity\|event_horizon\|lagrange\|Lagrange\|tunnel_prob\|escape_prob\|wave_function\|Nightmare\|nightmare" core/ training/ live/ --include="*.py"
   ```
   This grep should return ZERO hits (except in comments explaining the rename).

5. **HASH MUST NOT CHANGE.** The `__hash__` and `__eq__` methods in
   MarketState must produce identical hashes to ThreeBodyQuantumState
   for the same market data. The Bayesian brain table keys depend on this.
   Test: load a saved brain.pkl, verify it still works.

6. **Pickle compatibility.** Existing checkpoint files (brain.pkl, 
   pattern_library.pkl, templates.pkl) use ThreeBodyQuantumState.
   The backward-compat alias in three_body_state.py ensures pickle.load()
   still works. DO NOT delete the alias file until you've verified all
   checkpoints load correctly.

---

## VERIFICATION

After complete refactor:

```bash
# 1. All tests pass
pytest tests/ -v

# 2. No old terminology in code
grep -rn "ThreeBodyQuantumState\|QuantumFieldEngine\|ROCHE_SNAP\|particle_position" \
  core/ training/ live/ --include="*.py" | grep -v "DEPRECATED\|backward compat\|alias"
# Should return 0 lines

# 3. Forward pass produces identical results
python training/orchestrator.py --forward-pass --data DATA/ATLAS
# Compare oracle_trade_log.csv: same trade count, same PnL, same directions

# 4. Saved brain still loads
python -c "
from core.bayesian_brain import BayesianBrain
b = BayesianBrain()
b.load('checkpoints/live_brain.pkl')
print(f'Loaded {len(b.table)} states')
"

# 5. Live engine starts without errors
python -m live --dry-run --no-gui
```

---

## ESTIMATED SCOPE

- ~50 field renames in MarketState dataclass
- ~4 class renames
- ~3 file renames (with aliases)
- ~25 consumer files updated
- 0 logic changes
- 0 formula changes
- 0 threshold changes

Total: ~2-3 hours of mechanical find-and-replace with careful testing between each file.

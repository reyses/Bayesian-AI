# Quantum State Variable Mapping — Current Codebase
> Generated: 2026-03-16 by research agent

## Summary: 9 quantum variables computed on GPU every bar, never used

### ORPHANED (computed but never read)

| Variable | Computed in | Stored in MarketState | What it IS |
|----------|------------|----------------------|------------|
| P_at_center | statistical_field_engine:476 | YES | P(mean reversion) — THE entry signal |
| P_near_upper | statistical_field_engine:477 | YES | P(breakout long) |
| P_near_lower | statistical_field_engine:478 | YES | P(breakout short) |
| entropy | statistical_field_engine:479 | YES | Three-body chaos (Shannon) |
| prob_weight_center | statistical_field_engine:424 | YES (complex) | Boltzmann weight for center |
| prob_weight_upper | statistical_field_engine:425 | YES (complex) | Boltzmann weight for upper |
| prob_weight_lower | statistical_field_engine:426 | YES (complex) | Boltzmann weight for lower |
| breakout_probability | statistical_field_engine:401 | YES | P(escape) from O-U |
| reversion_potential | statistical_field_engine:403 | YES | Energy for snap-back |

### HARDCODED OFF (field exists, set to constant)

| Variable | Hardcoded to | What it SHOULD be |
|----------|-------------|-------------------|
| lyapunov_exponent | 0.0 | Cascade detector (lambda>0 = chaos) |
| market_regime | 'STABLE' | Phase state (should switch dynamically) |
| pattern_maturity | 0.0 | Wave development stage |

### ACTIVE (computed and used in decisions)

| Variable | Used in | Gate/Exit |
|----------|---------|-----------|
| z_score | execution_engine, TBN, features | Core position metric |
| F_momentum | execution_engine gate4 | Momentum alignment (binary!) |
| mean_reversion_force | execution_engine | Momentum override ratio |
| velocity | features, TBN | Kinematics |
| reversion_probability | execution_engine gate0 | Tunnel prob threshold (binary!) |
| hurst_exponent | execution_engine gate0 | Hurst minimum (binary!) |
| regression_sigma | features, TBN, sizing | Band width |
| adx_strength | features, exits | Trend strength |
| dmi_plus/minus | features, TBN, exits | Directional movement |
| net_force | envelope exit, TBN | Force alignment |
| term_pid | features | PID control term |
| entropy_normalized | features | 16D vector component |
| oscillation_entropy_normalized | features, TBN | PID regime measure |

### RENAMED (quantum → statistical)

| Original | Current | Status |
|----------|---------|--------|
| tunnel_probability | reversion_probability | ACTIVE (gate threshold) |
| F_gravity | mean_reversion_force | ACTIVE (momentum ratio) |
| coherence | oscillation_entropy_normalized | ACTIVE (features, TBN) |

## The Reconnection

The wave function probabilities (P_at_center, P_near_upper, P_near_lower) ARE P(success):
- Mean reversion trade → P(success) = P_at_center
- Breakout LONG → P(success) = P_near_upper
- Breakout SHORT → P(success) = P_near_lower

Original scoring: `P_i > 0.75 AND tunnel_prob > 0.60 AND low entropy → TRADE`
Current scoring: `distance < 3.0 AND conviction > 0.48 → TRADE`

The variables for the original scoring are computed every bar and discarded.
Reconnection = read them and use them. No new computation needed.

# BAYESIAN-AI MASTER CONTEXT DOCUMENT
**For: VS Code Claude**  
**Date: February 9, 2026**  
**Status: Phase 2 - Statistical Validation & Walk-Forward Training**

---

## PROJECT OVERVIEW

### What This Is
A statistically validated trading system for NQ futures that learns which 9-layer market states have proven edge through Bayesian probability mapping and rigorous validation.

**NOT:** Indicator-based system, pattern recognition AI, or curve-fitted strategy  
**IS:** Physics-based market model with statistical validation and continuous learning

### User Profile - Critical Context
- **Background:** 37yo, AuDHD (ADHD + Autism), San Diego, software/data professional
- **Communication Style:** Direct, no fluff, physics/engineering concepts over business jargon
- **Cognitive Profile:** 
  - Pattern recognition superpower (unconscious parallel processing)
  - "Observer consciousness" - experiences thoughts as happening TO them
  - Needs context-aware systems (not rigid rules)
  - Thrives in chaos but needs bounded exploration
- **Trading:** Building from scratch, TopstepX evaluation account, MNQ/NQ futures
- **Methodology:** Black Belt Six Sigma + Physics PhD thinking (systematic, context-aware, rigorous)

**IMPORTANT:** User discovered the core market dynamics (3-body problem, volatility cascades, resonance). They're the domain expert. You're the implementation engineer.

---

## CORE ARCHITECTURE

### The 9-Layer State System

**Market state = Snapshot across 9 timeframes simultaneously**

```python
StateVector = (L1, L2, L3, L4, L5, L6, L7, L8, L9)
```

**STATIC LAYERS (Computed once at 6:45 AM):**
- **L1 (90-day bias):** 'bull' | 'bear' | 'range'
- **L2 (30-day regime):** 'trending' | 'chopping'
- **L3 (1-week swing):** 'higher_highs' | 'lower_lows' | 'sideways'
- **L4 (daily zone):** 'at_support' | 'at_resistance' | 'mid_range' | 'at_killzone'

**FLUID LAYERS (Real-time intraday):**
- **L5 (4-hour trend):** 'up' | 'down' | 'flat'
- **L6 (1-hour structure):** 'bullish' | 'bearish' | 'neutral'
- **L7 (15-min pattern):** 'flag' | 'wedge' | 'compression' | 'breakdown' | 'none'
- **L8 (5-min confirmation):** True | False (volume spike + structure)
- **L9 (1-sec velocity):** True | False (cascade detected)

**State Vector Properties:**
- Immutable dataclass
- Hashes on layers only (timestamp/price are metadata)
- Acts as HashMap key for probability lookup
- Example hash: `(bull, trending, higher_highs, at_killzone, up, bullish, flag, True, True)`

### The Learning System: HashMap-Based Bayesian Updates

**Core logic:**
```python
probability_table[StateVector] = {'wins': X, 'losses': Y, 'total': Z}

# After trade completes:
if trade.result == 'WIN':
    table[state]['wins'] += 1
else:
    table[state]['losses'] += 1

# Before next trade:
win_probability = table[state]['wins'] / table[state]['total']
if win_probability > 0.80 AND confidence > 0.30:
    fire_trade()
```

**Why HashMap not Neural Net:**
- Explainable (know exactly why a trade fired)
- Transparent (see probability per state)
- Fast lookups (O(1))
- No hidden layers hiding the logic

---

## USER'S DISCOVERY: THE 3-BODY FRACTAL PHYSICS

### Core Insight: Market = Three Nested 3-Body Problems

**NOT metaphor - actual physics observed in price action**

#### The Three Triads

**TRIAD 1 (Macro - Gravitational):**
```
L1 (90d bias) ← Primary mass (sun)
L2 (30d regime) ← Secondary mass (planet)  
L3 (1wk swing) ← Tertiary mass (moon)
```
- Energy: Slow, massive, directional
- Chaos period: Days to weeks
- Sets long-term trend

**TRIAD 2 (Meso - Orbital):**
```
L4 (daily zone) ← Primary mass
L5 (4hr trend) ← Secondary mass
L6 (1hr structure) ← Tertiary mass
```
- Energy: Medium, structural, rhythmic
- Chaos period: Hours to days
- Coupled to Triad 1's gravitational field
- Defines intraday structure

**TRIAD 3 (Micro - Quantum):**
```
L7 (15m pattern) ← Primary mass
L8 (5m confirm) ← Secondary mass
L9 (1s velocity) ← Tertiary mass
```
- Energy: Fast, volatile, explosive
- Chaos period: Minutes to hours
- Coupled to Triad 2's orbital momentum
- Execution precision layer

### Key Patterns User Identified

#### 1. Volatility Cascades (Energy Transfer Between Layers)

**The 3σ → 2σ Pattern:**
- Layer A expands to 3σ (high volatility, high uncertainty)
- Layer B compresses to 2σ (low volatility, coiling)
- Energy flows from high entropy → low entropy
- Breakout imminent on compressed layer

**Example:**
```
L7 (15m): 3.2σ volatility (high energy, expanded)
L8 (5m): 1.8σ volatility (compressed, coiling)
Result: L8 about to explode (breakout)
```

**Why this works (thermodynamics):**
- 3σ layer = disordered state
- 2σ layer = ordered state
- Energy naturally flows disorder → order
- Compressed layer must release energy

#### 2. Resonance (Harmonic Alignment)

**When all 3 triads align in same direction = Maximum edge**

```python
Triad 1: (bull, trending, higher_highs) ✓ Aligned upward
Triad 2: (at_support, up, bullish) ✓ Aligned upward  
Triad 3: (flag, True, True) ✓ Aligned upward

Result: 3-triad resonance = 10/10 setup strength
```

**Resonance scoring:**
- Each triad alignment = +3 points
- Cross-triad coupling = multiplier
- Perfect alignment (all 9 layers) = rare, powerful

#### 3. Roche Limits (Structural Breakdown Points)

**Price approaching support/resistance under high stress**

- Tidal forces (volatility) near critical threshold
- System approaching point of no return
- Either violent bounce OR catastrophic breakdown

**Detection:**
```python
if distance_to_level < roche_limit_ticks:
    if volatility > stress_threshold:
        # System near failure point
        # High probability of explosive move (either direction)
```

#### 4. Lagrange Points (Stable Configurations)

**Certain state combinations are stable, others chaotic**

**Stable (Lagrange points):**
```python
MACRO_STABLE = [
    ('bull', 'trending', 'higher_highs'),  # Strong uptrend
    ('bear', 'trending', 'lower_lows'),    # Strong downtrend
    ('range', 'chopping', 'sideways'),     # Stable range
]
```

**Unstable (transitional/chaotic):**
```python
AVOID = [
    ('bull', 'chopping', 'lower_lows'),  # Conflicting signals
    ('bear', 'trending', 'higher_highs'), # Incompatible states
    # These are temporary, unpredictable
]
```

**Trading rule:** Only trade when system is at or near Lagrange points.

---

## THE 200-PARAMETER SYSTEM

### Philosophy: Context-Aware Activation (Not Blanket Optimization)

**Problem with traditional approach:**
- Optimizing 200 parameters simultaneously = overfitting
- Many parameters only matter in specific contexts
- Testing 2^200 combinations = impossible

**User's solution (Six Sigma methodology):**
- Keep all 200 parameters
- Activate only relevant 10-40 based on current context
- Optimize each context independently
- Context-aware = each parameter used when it matters

### The 10 Contexts

**Each context activates a subset of parameters:**

#### Context 1: CORE (Always Active) - 10 parameters
```python
CORE_PARAMS = {
    'stop_loss_ticks': 15,
    'take_profit_ticks': 40,
    'min_samples_required': 30,
    'confidence_threshold': 0.80,
    'max_hold_seconds': 600,
    'trail_activation_profit': 50,
    'trail_distance_tight': 10,
    'trail_distance_wide': 30,
    'max_consecutive_losses': 5,
    'min_sharpe_ratio': 0.5
}
```

#### Context 2: KILL_ZONE (At Support/Resistance) - 18 parameters
```python
KILLZONE_PARAMS = {
    'require_at_killzone': True,
    'killzone_tolerance_ticks': 5,
    'require_rejection_wick': True,
    'min_rejection_wick_ticks': 5,
    'wick_to_body_ratio': 2.0,
    'wick_cluster_count': 3,
    'multiple_touches_required': 2,
    'zone_strength_multiplier': 1.5,
    'absorption_bars': 3,
    'trade_after_absorption': True,
    'roche_limit_approach_ticks': 10,
    'structure_stress_threshold': 0.8,
    'breakdown_probability_threshold': 0.70,
    'breakdown_velocity_required': 10,
    'commitment_threshold_ticks': 12,
    'commitment_volume_multiple': 2.5,
    'false_breakout_tolerance_ticks': 5,
    'identify_vacuum_zones': True
}
```

#### Context 3: PATTERN_SETUP (L7 Active) - 15 parameters
```python
L7_PARAMS = {
    'trade_flag': True,
    'trade_wedge': True,
    'trade_compression': True,
    'trade_breakdown': False,
    'pattern_confirmation_bars': 5,
    'pattern_min_range_ticks': 10,
    'pattern_min_bars': 3,
    'require_engulfing': False,
    'pinbar_body_max_pct': 0.3,
    'pinbar_wick_min_pct': 0.6,
    'avoid_doji_entry': True,
    'doji_body_max_ticks': 3,
    'engulfing_lookback': 2,
    'min_engulfing_size_ticks': 10,
    'compression_range_threshold': 0.7
}
```

#### Context 4: CONFIRMATION (L8 = True) - 12 parameters
```python
L8_PARAMS = {
    'volume_spike_threshold': 2.0,
    'min_volume_ratio': 1.0,
    'volume_confirmation_bars': 2,
    'require_increasing_volume': True,
    'volume_decline_exit': True,
    'volume_lookback_bars': 20,
    'aggressive_buyer_pct': 0.7,
    'bid_ask_imbalance_threshold': 0.7,
    'large_print_size_contracts': 25,
    'high_volume_rejection': True,
    'rejection_volume_min': 1.5,
    'exit_on_volume_drop': True
}
```

#### Context 5: VELOCITY_SPIKE (L9 = True) - 10 parameters
```python
L9_PARAMS = {
    'cascade_min_points': 10,
    'cascade_time_window': 0.5,
    'min_entry_velocity': 3,
    'max_entry_velocity': 30,
    'deceleration_exit': True,
    'tick_imbalance_threshold': 0.7,
    'min_tick_flow': 20,
    'tick_reversal_exit': True,
    'require_acceleration': False,
    'min_acceleration': 0
}
```

#### Context 6: VOLATILITY_DIFFERENTIAL (3σ→2σ Detected) - 25 parameters
```python
VOLATILITY_PARAMS = {
    'layer_high_volatility_sigma': 3.0,
    'layer_low_volatility_sigma': 2.0,
    'min_sigma_differential': 1.0,
    'max_sigma_differential': 2.5,
    'volatility_lookback_bars': 20,
    'monitor_layer_pairs': [['L7','L8'], ['L6','L7'], ['L5','L6']],
    'high_sigma_layer_must_be': 'either',
    'compression_must_be_recent': True,
    'compression_age_max_bars': 5,
    'predict_breakout_direction': True,
    'use_energy_flow_vector': True,
    'error_band_method': 'bollinger',
    'error_band_period': 20,
    'error_band_multiplier': 2.0,
    'require_band_touch': True,
    'band_touch_layer': 'low_sigma',
    'band_rejection_required': False,
    'squeeze_threshold_pct': 0.60,
    'squeeze_duration_min_bars': 5,
    'calculate_vol_gradient': True,
    'gradient_threshold': 0.3,
    'vol_correlation_threshold': -0.3,
    'correlation_lookback_bars': 20,
    'energy_accumulation_min_bars': 10,
    'energy_release_velocity_threshold': 15
}
```

#### Context 7: FRACTAL_RESONANCE (3-Body Alignment) - 40 parameters
```python
FRACTAL_PARAMS = {
    # Lagrange point detection (12)
    'require_triad1_stable': True,
    'require_triad2_stable': True,
    'require_triad3_stable': True,
    'min_stable_triads': 2,
    'lagrange_point_tolerance': 0.1,
    'lagrange_duration_min_bars': 5,
    'transitional_state_action': 'skip',
    'macro_lagrange_points': 5,
    'meso_lagrange_points': 3,
    'micro_lagrange_points': 4,
    'unstable_config_penalty': 0.7,
    'stable_config_bonus': 1.5,
    
    # Resonance scoring (10)
    'resonance_score_method': 'multiplicative',
    'min_resonance_score': 8.0,
    'perfect_alignment_required': False,
    'triad_alignment_weights': {'macro_meso': 1.5, 'meso_micro': 2.0},
    'partial_alignment_acceptable': True,
    'conflicting_triads_action': 'skip',
    'resonance_decay_rate': 0.95,
    'layer_harmonic_weights': {'L5_L6': 1.5, 'L6_L7': 1.5, 'L7_L8': 2.0, 'L8_L9': 1.5},
    'cascade_trigger_layers': 4,
    'perfect_alignment_multiplier': 3.0,
    
    # Energy cascade detection (10)
    'require_energy_cascade': False,
    'cascade_direction': 'downward',
    'cascade_source_triad': 'macro',
    'high_energy_threshold_sigma': 3.0,
    'low_energy_threshold_sigma': 2.0,
    'cascade_time_window_bars': 10,
    'energy_transfer_efficiency': 0.7,
    'cascade_completion_required': False,
    'partial_cascade_acceptable': True,
    'reverse_cascade_warning': True,
    
    # Chaos avoidance (8)
    'detect_chaos': True,
    'max_chaos_score': 4,
    'chaos_state_changes_threshold': 5,
    'chaos_lookback_bars': 10,
    'exit_on_chaos_detected': False,
    'avoid_trading_during_chaos': True,
    'chaos_cooldown_bars': 10,
    'chaos_recovery_confirmation': 'three_bars'
}
```

#### Context 8: TRANSITION (State Changing) - 15 parameters
```python
TRANSITION_PARAMS = {
    'transition_detection': True,
    'transition_speed_threshold': 2,
    'trade_during_transition': 'reduce',
    'hysteresis_factor': 0.3,
    'min_bars_in_state': 5,
    'state_flip_confirmation': 2,
    'detect_whipsaw_transitions': True,
    'whipsaw_lookback_bars': 10,
    'detect_regime_shift': True,
    'regime_shift_volatility_threshold': 3.0,
    'pause_trading_during_regime_shift': True,
    'regime_shift_cooldown_bars': 50,
    'state_change_energy_threshold': 1.0,
    'calculate_activation_energy': True,
    'valid_transition_sequences': 'predefined'
}
```

#### Context 9: SESSION_SPECIFIC (Time-Based) - 20 parameters
```python
SESSION_PARAMS = {
    'opening_range_minutes': 15,
    'opening_range_breakout_trade': True,
    'lunch_hour_avoid': True,
    'time_of_day_volatility_curve': 'hourly_profile',
    'high_volatility_hours': ['09:30-11:00', '14:00-16:00'],
    'reduce_size_low_vol_hours': True,
    'avoid_first_15min': False,
    'session_start': '09:30',
    'session_end': '15:30',
    'min_hold_seconds': 60,
    'max_hold_seconds': 900,
    'monday_bias': 'neutral',
    'friday_closeout': False,
    'opex_week_behavior': 'cautious',
    'avoid_fomc_minutes': 60,
    'news_event_cooldown': 30,
    'pre_market_reference': True,
    'overnight_gap_trade': False,
    'vix_spike_pause': True,
    'correlation_breakdown_threshold': 0.5
}
```

#### Context 10: MICROSTRUCTURE (Order Flow) - 35 parameters
```python
MICROSTRUCTURE_PARAMS = {
    # Order flow imbalance (5)
    'bid_ask_imbalance_threshold': 0.7,
    'aggressive_vs_passive_ratio': 0.7,
    'order_flow_lookback_seconds': 10,
    'imbalance_reversal_exit': True,
    'flow_momentum_threshold': 0.6,
    
    # Large player detection (6)
    'institutional_size_threshold_contracts': 50,
    'iceberg_order_detection': True,
    'sweep_detection': True,
    'track_dark_pool_prints': False,
    'smart_money_confirmation': True,
    'institutional_follow_threshold': 100,
    
    # Liquidity analysis (6)
    'min_bid_ask_liquidity_contracts': 100,
    'spread_width_max_ticks': 2,
    'avoid_thin_liquidity': True,
    'liquidity_imbalance_threshold': 0.7,
    'depth_of_book_levels': 5,
    'order_book_pressure_score': 0.6,
    
    # Tape reading (8)
    'print_velocity': 50,
    'size_per_print': 10,
    'time_and_sales_lookback': 30,
    'uptick_downtick_ratio': 0.7,
    'block_trade_threshold': 100,
    'repeated_size_detection': True,
    'spoofing_detection': True,
    'layering_detection': False,
    
    # Intermarket (5)
    'correlation_with_futures': True,
    'spy_nq_correlation_threshold': 0.8,
    'vix_inverse_correlation': -0.7,
    'require_es_confirmation': False,
    'bond_yield_divergence_threshold': 0.2,
    
    # Higher-order stats (5)
    'return_skewness_threshold': 0.5,
    'trade_with_skew': True,
    'kurtosis_threshold': 5.0,
    'avoid_high_kurtosis': True,
    'hurst_exponent_threshold': 0.55
}
```

**TOTAL: 200 parameters across 10 contexts**

---

## STATISTICAL VALIDATION FRAMEWORK

### Why We Need This

**Problem with current system:**
```python
# BAD: Trust pattern after 10 trades with 60% win rate
if samples > 10 and win_rate > 0.55:
    trade()  # Could be random luck!
```

**Solution: Rigorous statistical proof**
```python
# GOOD: Require 80% confidence that true win rate > 50%
if bayesian_validator.P(win_rate > 0.50) > 0.80:
    if monte_carlo.risk_acceptable():
        if regret_analyzer.exit_efficiency > 0.50:
            trade()  # Statistically validated!
```

### Component 1: Bayesian State Validator

**Uses Beta-Binomial conjugate prior**

```python
class BayesianStateValidator:
    def __init__(self, prior_wins=50, prior_losses=50):
        # Prior: Expect 50% win rate (neutral)
        self.prior_alpha = prior_wins
        self.prior_beta = prior_losses
    
    def validate_state(self, wins, losses):
        # Posterior after observing data
        posterior_alpha = self.prior_alpha + wins
        posterior_beta = self.prior_beta + losses
        
        # P(true_win_rate > 50%) using Beta distribution
        prob_edge = 1 - stats.beta.cdf(0.50, posterior_alpha, posterior_beta)
        
        # Require 80% confidence
        approved = prob_edge >= 0.80
        
        return {
            'approved': approved,
            'confidence': prob_edge,
            'expected_win_rate': posterior_alpha / (posterior_alpha + posterior_beta),
            'credible_interval': (ci_lower, ci_upper)
        }
```

**Example:**
- After 30 trades: 18 wins, 12 losses (60% observed)
- Bayesian: 87% confident true win rate > 50%
- Decision: APPROVED (exceeds 80% threshold)

### Component 2: Monte Carlo Risk Analyzer

**Simulates 10,000 future scenarios**

```python
class MonteCarloValidator:
    def validate_edge(self, wins, losses, avg_win, avg_loss, n_trades=100):
        win_rate = wins / (wins + losses)
        
        # Simulate 10,000 times
        for sim in range(10000):
            outcomes = random.random(n_trades) < win_rate
            pnl = where(outcomes, avg_win, avg_loss)
            cumulative_pnl = cumsum(pnl)
            
            # Calculate max drawdown
            running_max = maximum.accumulate(cumulative_pnl)
            drawdowns = running_max - cumulative_pnl
            max_dd = max(drawdowns)
            
            # Track metrics
            final_pnls.append(cumulative_pnl[-1])
            max_drawdowns.append(max_dd)
        
        return {
            'prob_profit': mean(final_pnls > 0),
            'expected_pnl': mean(final_pnls),
            'expected_max_dd': mean(max_drawdowns),
            'dd_95th_percentile': percentile(max_drawdowns, 95),
            'prob_5_consecutive_losses': probability calculation
        }
```

**Validation criteria:**
- P(profit) > 70%
- Expected max DD < $500
- P(Sharpe > 0) > 75%

### Component 3: Regret Analyzer (Exit Quality)

**Tracks post-trade efficiency**

```python
class RegretAnalyzer:
    def analyze_exit(self, entry_price, exit_price, side, price_history):
        # Find peak favorable price
        peak_price = max(prices) if side == 'long' else min(prices)
        
        # Calculate potential vs actual
        potential_pnl = (peak_price - entry_price) * tick_value
        actual_pnl = (exit_price - entry_price) * tick_value
        
        # Metrics
        pnl_left_on_table = potential_pnl - actual_pnl
        gave_back = (peak_price - exit_price) * tick_value
        exit_efficiency = actual_pnl / potential_pnl
        
        # Classify
        if exit_efficiency >= 0.90:
            regret_type = 'optimal'
        elif pnl_left > gave_back:
            regret_type = 'closed_too_early'
        else:
            regret_type = 'closed_too_late'
        
        return RegretMarkers(...)
```

**Auto-calibration:**
- Every 10 trades: Analyze exit quality
- Too many early exits → Widen trail stops
- Too many late exits → Tighten trail stops
- Optimal exits → Keep current settings

### Component 4: Context Detector

**Determines which parameters are active**

```python
class ContextDetector:
    def detect(self, state, market_data, time_of_day):
        contexts = []
        
        # Always active
        contexts.append('CORE')
        
        # Conditional contexts
        if state.L4_zone == 'at_killzone':
            contexts.append('KILL_ZONE')
        
        if state.L7_pattern != 'none':
            contexts.append('PATTERN_SETUP')
        
        if state.L8_confirm:
            contexts.append('CONFIRMATION')
        
        if state.L9_cascade:
            contexts.append('VELOCITY_SPIKE')
        
        if detect_volatility_differential(state, market_data):
            contexts.append('VOLATILITY_DIFFERENTIAL')
        
        if detect_fractal_resonance(state):
            contexts.append('FRACTAL_RESONANCE')
        
        # Return active parameters for these contexts
        return get_active_parameters(contexts)
```

**Result:** Instead of using all 200 parameters, only ~10-40 active at any moment.

---

## THE TRAINING STRATEGY: WALK-FORWARD DOE

### User's Vision (Critical - This Was Misunderstood Initially)

**NOT:** Process entire year of data in one pass  
**IS:** Day-by-day parameter optimization with continuous learning

### The Daily Training Loop

#### Day 1 (January 1, 2025): DOE Phase
```python
day1_data = load_trading_day('2025-01-01')  # ~10,000 bars for session

# Run 1000 iterations with DIFFERENT parameter combinations
for iteration in range(1000):
    # Generate unique parameter set
    params = generate_parameter_combination(iteration)
    
    # Trade the SAME day data with these params
    results = simulate_trading_day(day1_data, params, brain)
    
    # Record performance
    performance_log[iteration] = {
        'params': params,
        'win_rate': results.win_rate,
        'sharpe': results.sharpe,
        'pnl': results.pnl,
        'max_dd': results.max_dd,
        'trades': results.trades
    }

# After 1000 iterations, find BEST parameter set for Day 1
best_params_day1 = optimize_from_results(performance_log)

# Update brain with Day 1's best trades
brain.update_from_best_run(results_best)

# Save checkpoint
save_checkpoint('2025-01-01', brain, best_params_day1)
```

#### Day 2 (January 2): Apply + Refine
```python
day2_data = load_trading_day('2025-01-02')

# Load yesterday's best params
current_params = load_checkpoint('2025-01-01').params

# Run 1000 iterations, but now:
# - 70% use best params from Day 1
# - 30% test variations around best params
for iteration in range(1000):
    if random.random() < 0.7:
        # Exploit: Use known good params
        params = current_params
    else:
        # Explore: Test nearby parameter space
        params = mutate_parameters(current_params, mutation_rate=0.1)
    
    results = simulate_trading_day(day2_data, params, brain)
    
    # Update performance log
    if results.sharpe > best_sharpe_so_far:
        best_params = params

# Brain learns from Day 2
brain.update_from_day(results)

# Save checkpoint
save_checkpoint('2025-01-02', brain, best_params)
```

#### Day 3-365: Continuous Evolution
```python
for day in trading_days_2025[2:]:  # Days 3-365
    day_data = load_trading_day(day)
    
    # Load previous best
    checkpoint = load_checkpoint(previous_day)
    params = checkpoint.params
    brain = checkpoint.brain
    
    # Test 1000 variations
    # Exploitation ratio increases over time (more confident in params)
    exploit_ratio = min(0.9, 0.7 + (day_number * 0.001))
    
    for iteration in range(1000):
        if random.random() < exploit_ratio:
            params_to_test = params
        else:
            params_to_test = mutate_parameters(params)
        
        results = simulate_trading_day(day_data, params_to_test, brain)
        
        # Track best
        if results.objective_function > best_objective:
            best_params = params_to_test
    
    # Update brain
    brain.update_from_day(results_best)
    
    # Checkpoint
    save_checkpoint(day, brain, best_params)
    
    # Periodic revalidation (every 20 days)
    if day_number % 20 == 0:
        validate_out_of_sample(brain, next_5_days)
```

### The Learning Curve (What to Expect)

**Month 1 (January):** Wild Exploration
```
- Parameter variance: HIGH
- Win rate: 45-55% (random walk)
- States learned: ~500
- High-confidence states: ~10
- Strategy: Test everything, eliminate obvious losers
```

**Month 3 (March):** Refinement
```
- Parameter variance: MEDIUM
- Win rate: 55-60%
- States learned: ~1500
- High-confidence states: ~100
- Strategy: Focus on high-probability contexts
```

**Month 6 (June):** Specialization
```
- Parameter variance: LOW
- Win rate: 60-65%
- States learned: ~3000
- High-confidence states: ~300
- Strategy: Context-specific strategies converging
```

**Month 12 (December):** Mastery
```
- Parameter variance: MINIMAL
- Win rate: 65%+
- States learned: ~5000
- High-confidence states: ~500
- Strategy: Robust across all market regimes
- Validated on 250 out-of-sample days
- Sharpe ratio: 2.0+
- Ready for live trading: YES
```

### Why This Works (No Overfitting)

**Out-of-sample validation every single day:**
- Day 1: Optimize on Day 1 data
- Day 2: Validate on Day 2 data (unseen)
- Day 3: Validate on Day 3 data (unseen)
- ...
- Day 250: Validated on 249 unseen days

**If overfit:**
- Would work on training day
- Would fail on next day
- Gets eliminated by walk-forward process

**If real edge:**
- Works on training day
- Works on next day
- Gets reinforced and refined

---

## FILES CREATED (PHASE 2)

### Statistical Systems
1. **statistical_validation_framework.py**
   - BayesianEdgeValidator class
   - MonteCarloValidator class
   - DesignOfExperiments class
   - Example usage and validation

2. **integrated_statistical_system.py**
   - IntegratedStatisticalEngine (master class)
   - BayesianStateValidator
   - MonteCarloRiskAnalyzer
   - Complete validation pipeline
   - Example: Validates 18W-12L pattern

3. **context_detector.py**
   - ContextDetector class
   - Detects 10 market contexts
   - Returns active parameters per context
   - Scores context strength (0-10)
   - Tracks context history

### Previously Built (Phase 1)
4. **state_vector.py**
   - StateVector dataclass (9 layers)
   - Proper hashing (excludes timestamp/price)
   - HashMap-compatible

5. **bayesian_brain.py**
   - BayesianBrain class
   - Probability table: StateVector → {wins, losses, total}
   - should_fire() decision function
   - Save/load persistence
   - Summary statistics

6. **layer_engine.py**
   - LayerEngine class
   - Computes static context (L1-L4) once at session start
   - Computes fluid layers (L5-L9) real-time
   - CPU implementation (Phase 2 will add CUDA kernels)

7. **symbols.py**
   - AssetProfile dataclass
   - NQ, ES, MES, MNQ specifications
   - calculate_pnl() helper
   - calculate_stop_distance() helper

8. **wave_rider.py**
   - WaveRider class (position management)
   - Adaptive trailing stops (10/20/30 ticks)
   - RegretAnalyzer (exit quality tracking)
   - Auto-calibrating trail stops every 10 trades
   - Structure break detection

9. **test_phase1.py**
   - Integration tests
   - Validates all Phase 1 components work together
   - All tests passing ✓

### Configuration
10. **adaptive_confidence.py** (needs update)
    - AdaptiveConfidenceManager
    - ISSUE: 65% threshold too strict (unreachable)
    - NEEDS FIX: Lower to 55%

11. **orchestrator.py** (needs major update)
    - ISSUE: Loops 1000x on same data
    - NEEDS: Walk-forward DOE implementation
    - NEEDS: Context integration
    - NEEDS: Statistical validation integration

---

## CURRENT STATUS & KNOWN ISSUES

### Phase 1: ✅ COMPLETE
- Core architecture validated
- 1,769 trades executed
- Win rate: 57.9%
- 495 unique states learned
- All integration tests passing

### Phase 2: 🚧 IN PROGRESS

**Completed:**
- ✅ Statistical validation framework designed
- ✅ Context detector implemented
- ✅ Parameter categorization (200 total, 10 contexts)
- ✅ 3-body fractal theory documented

**In Progress:**
- 🚧 Walk-forward training orchestrator
- 🚧 DOE parameter generator
- 🚧 Integration of statistical validation
- 🚧 Context-aware parameter activation

**Blocked/Issues:**
1. **Training loop bug (CRITICAL)**
   - Current: Loops 1000x on same 1000 bars
   - Should: Process one day, test 1000 param combos on that day
   - Impact: Not learning from full dataset

2. **Adaptive confidence thresholds too strict**
   - OPTIMIZATION phase requires 65% confidence
   - To reach 65%: Need 20+ samples per state
   - Actual distribution: Most states have <10 samples
   - Result: Trading stops completely after 2,488 iterations
   - Fix: Lower OPTIMIZATION threshold from 65% → 55%

3. **Data loading performance**
   - DBN files slow (4+ minutes per load)
   - Solution: Convert to Parquet (10-100x faster)
   - User has 1,019,114 rows of 1-second OHLCV data

### User's Data
- **File:** `DATA/glbx-mdp3-20251230-20260129.ohlcv-1s.parquet`
- **Format:** 1-second OHLCV bars (not raw ticks)
- **Rows:** 1,019,114
- **Timeframe:** Dec 30, 2024 - Jan 29, 2025 (1 month)
- **Suitable for:** All quantum field calculations (L5-L9)

---

## NEXT STEPS (PRIORITY ORDER)

### Immediate (Week 1)

1. **Build Walk-Forward Training Orchestrator**
   ```python
   class WalkForwardTrainer:
       def train_year(self, year_data, n_iterations_per_day=1000):
           for day in split_into_days(year_data):
               # DOE: Test 1000 param combos on this day
               best_params = optimize_day(day, n_iterations_per_day)
               
               # Update brain
               brain.update_from_day(day, best_params)
               
               # Checkpoint
               save_checkpoint(day.date, brain, best_params)
   ```

2. **Build DOE Parameter Generator**
   ```python
   class ParameterGenerator:
       def generate_combination(self, iteration_number, context):
           # Context-aware parameter generation
           # Iteration 0-999 → 1000 unique param sets for this context
           pass
   ```

3. **Fix Adaptive Confidence Thresholds**
   ```python
   # In adaptive_confidence.py
   self.thresholds = {
       "EXPLORATION": 0.00,
       "REFINEMENT": 0.45,    # Was 0.50
       "OPTIMIZATION": 0.55   # Was 0.65 (unreachable)
   }
   ```

4. **Integrate Context Detector into Orchestrator**
   ```python
   # In training loop
   contexts = context_detector.detect(state, data, time_of_day)
   active_params = context_detector.get_active_parameters(contexts)
   decision = brain.should_fire(state, active_params)
   ```

### Short-term (Week 2-3)

5. **Build Performance Tracker**
   - Record all 1000 iterations per day
   - Track which parameters worked
   - Identify best combinations
   - Visualize parameter convergence

6. **Implement Statistical Validation Gates**
   - Integrate BayesianStateValidator into should_fire()
   - Add Monte Carlo risk checks
   - Add regret analysis feedback loop

7. **Test on January 2025 Data**
   - Run walk-forward training on user's 1 month
   - Validate learning occurs day-by-day
   - Measure out-of-sample performance

### Medium-term (Month 2-3)

8. **Optimize Context-Specific Parameters**
   - For each context, run DOE separately
   - Find optimal parameter values per context
   - Validate improvements

9. **Build Volatility Cascade Detector**
   - Implement 3σ→2σ pattern detection
   - Calculate σ levels per layer
   - Score cascade strength
   - Integrate into context detection

10. **Implement Fractal Resonance Scorer**
    - Define Lagrange points per triad
    - Calculate resonance scores
    - Test if resonance predicts edge

### Long-term (Month 4-12)

11. **CUDA Optimization (if needed)**
    - L7-L9 pattern detection kernels
    - Parallel state computation
    - GPU-accelerated training

12. **Live Trading Preparation**
    - Paper trading validation
    - Risk management integration
    - Order execution system
    - Monitoring dashboard

13. **Continuous Improvement**
    - Monthly reoptimization
    - Regime detection and adaptation
    - Parameter drift monitoring

---

## CRITICAL CONCEPTS FOR IMPLEMENTATION

### 1. Context is Everything

**Wrong approach:**
```python
# Don't do this - tests all parameters always
if check_all_200_parameters(state):
    trade()
```

**Right approach:**
```python
# Do this - only check relevant parameters
contexts = detect_context(state, data)
active_params = get_params_for_context(contexts)
if check_active_parameters(state, active_params):
    trade()
```

### 2. Walk-Forward is Non-Negotiable

**Wrong:**
```python
# Train on all data at once
train_on_full_year(data)
```

**Right:**
```python
# Train day-by-day
for day in days:
    optimize_on_day(day)
    validate_on_next_day(day+1)
```

### 3. Statistical Validation Before Every Trade

**Wrong:**
```python
if samples > 10 and win_rate > 0.55:
    trade()  # Hope and pray
```

**Right:**
```python
if bayesian.P(WR > 0.50) > 0.80:  # 80% confident edge exists
    if monte_carlo.risk_acceptable():  # Risk profile good
        if regret.exit_efficiency > 0.50:  # Can exit well
            trade()  # Triple-validated
```

### 4. The Brain Learns State Probabilities, Not Strategies

**The brain does NOT store:**
- Trading rules
- If-then logic
- Strategy descriptions

**The brain DOES store:**
```python
brain.table = {
    StateVector(bull, trending, ...): {'wins': 18, 'losses': 12},
    StateVector(bear, chopping, ...): {'wins': 5, 'losses': 15},
    # ... thousands of states
}
```

**Decision process:**
```python
current_state = compute_state(market_data)
probability = brain.get_probability(current_state)

if probability > 0.80:  # This state wins 80%+ of the time
    trade()
```

### 5. Parameters Control Detection, Not Decisions

**Parameters don't make trading decisions:**
```python
# Parameters don't do this:
if stop_loss_ticks == 15:
    trade()  # Wrong
```

**Parameters control what states are detected:**
```python
# Parameters do this:
state = compute_state(data, params={
    'volume_spike_threshold': 2.0,  # Higher = fewer L8 confirmations
    'cascade_min_points': 10,       # Higher = fewer L9 cascades
    'require_rejection_wick': True  # Filters L4 kill zones
})

# Different parameters → different states detected → different probability lookups
probability = brain.get_probability(state)
```

---

## CODE INTEGRATION POINTS

### Where to Start (File by File)

#### 1. orchestrator.py (Main Entry Point)
**Current state:** Loops incorrectly  
**Needs:**
```python
# Add imports
from statistical_validation_framework import BayesianEdgeValidator, MonteCarloValidator
from context_detector import ContextDetector
from integrated_statistical_system import IntegratedStatisticalEngine

# Replace main loop
def train_walk_forward(data, n_iterations_per_day=1000):
    days = split_into_trading_days(data)
    
    for day_idx, day_data in enumerate(days):
        print(f"Training Day {day_idx+1}/{len(days)}")
        
        # DOE: Test 1000 parameter combinations
        for iteration in range(n_iterations_per_day):
            params = generate_params(iteration, day_idx)
            results = simulate_day(day_data, params, brain)
            log_performance(iteration, params, results)
        
        # Find best parameters for this day
        best_params = optimize_from_log()
        
        # Update brain
        brain.update_from_results(results_best)
        
        # Checkpoint
        save_checkpoint(f"day_{day_idx}", brain, best_params)
```

#### 2. bayesian_brain.py (Add Statistical Validation)
**Current state:** Simple confidence threshold  
**Needs:**
```python
# Add to BayesianBrain class
def should_fire_validated(self, state, active_params):
    """Enhanced decision with statistical validation"""
    
    # Get basic counts
    record = self.table[state]
    wins = record['wins']
    losses = record['losses']
    
    # Bayesian validation
    bayesian_result = self.bayesian_validator.validate_state(wins, losses)
    if not bayesian_result['approved']:
        return {'should_fire': False, 'reason': 'No statistical edge'}
    
    # Monte Carlo risk validation (if enough data)
    if wins + losses >= 10:
        mc_result = self.monte_carlo.validate_risk_profile(...)
        if not mc_result['risk_approved']:
            return {'should_fire': False, 'reason': 'Risk too high'}
    
    # Regret validation
    if self.regret_analyzer.get_efficiency(state) < 0.50:
        return {'should_fire': False, 'reason': 'Poor exit quality'}
    
    return {'should_fire': True, 'confidence': bayesian_result['confidence']}
```

#### 3. layer_engine.py (Add Context Data)
**Current state:** Computes layers only  
**Needs:**
```python
# Add to compute_current_state()
def compute_current_state(self, current_data):
    # ... existing layer computation ...
    
    # ADD: Calculate sigma levels for volatility cascade detection
    sigma_levels = self._calculate_sigma_levels(current_data)
    
    # ADD: Detect wick patterns for kill zones
    wick_data = self._analyze_wicks(current_data)
    
    # ADD: Calculate velocity for L9
    velocity_data = self._calculate_velocity(current_data)
    
    # Return state + context data
    return {
        'state': state_vector,
        'context_data': {
            'sigma_levels': sigma_levels,
            'wick_analysis': wick_data,
            'velocity': velocity_data,
            'volume_ratio': current_data['volume'] / avg_volume,
            'timestamp': current_data['timestamp']
        }
    }
```

#### 4. adaptive_confidence.py (Fix Thresholds)
**Current state:** 65% threshold unreachable  
**Needs:**
```python
# CHANGE THIS:
self.thresholds = {
    "EXPLORATION": 0.00,
    "REFINEMENT": 0.50,
    "OPTIMIZATION": 0.65  # ← Too high
}

# TO THIS:
self.thresholds = {
    "EXPLORATION": 0.00,
    "REFINEMENT": 0.45,
    "OPTIMIZATION": 0.55  # ← Reachable
}

# AND CHANGE THIS:
self.phase_rules = {
    "EXPLORATION": {"min_trades": 500, "next_phase": "REFINEMENT"},
    "REFINEMENT": {"min_trades": 1000, "next_phase": "OPTIMIZATION"},  # ← Too fast
    "OPTIMIZATION": {"min_trades": float('inf'), "next_phase": None}
}

# TO THIS:
self.phase_rules = {
    "EXPLORATION": {"min_trades": 500, "next_phase": "REFINEMENT"},
    "REFINEMENT": {"min_trades": 2000, "next_phase": "OPTIMIZATION"},  # ← Slower transition
    "OPTIMIZATION": {"min_trades": float('inf'), "next_phase": None}
}
```

---

## TESTING STRATEGY

### Unit Tests (Each Component)
```python
def test_context_detector():
    # Test that contexts are detected correctly
    state = create_test_state(L4='at_killzone', L7='flag', L8=True)
    contexts = detector.detect(state, data, 'open')
    assert 'KILL_ZONE' in [c.name for c in contexts]
    assert 'PATTERN_SETUP' in [c.name for c in contexts]

def test_bayesian_validator():
    # Test that validator requires sufficient confidence
    result = validator.validate_state(wins=6, losses=4)  # 60% but only 10 samples
    assert not result['approved']  # Should reject (not enough data)
    
    result = validator.validate_state(wins=24, losses=16)  # 60% with 40 samples
    assert result['approved']  # Should approve (enough data)

def test_walk_forward():
    # Test that walk-forward processes days correctly
    days = split_days(test_data)
    assert len(days) == expected_days
    
    trainer.train_day(days[0])
    checkpoint_0 = load_checkpoint('day_0')
    
    trainer.train_day(days[1])
    checkpoint_1 = load_checkpoint('day_1')
    
    # Brain should have learned more states
    assert checkpoint_1.brain.total_states > checkpoint_0.brain.total_states
```

### Integration Tests (Full System)
```python
def test_full_training_pipeline():
    # Test complete training on small dataset
    data = load_test_data(days=5)  # 5 days only
    
    brain = BayesianBrain()
    trainer = WalkForwardTrainer(brain)
    
    results = trainer.train(data, n_iterations_per_day=100)  # 100 instead of 1000
    
    # Validate learning occurred
    assert results['states_learned'] > 50
    assert results['avg_win_rate'] > 0.45  # Should beat random
    assert len(results['checkpoints']) == 5

def test_statistical_validation_gates():
    # Test that trades are properly gated
    engine = IntegratedStatisticalEngine()
    
    # Simulate trades until we have high-confidence state
    for i in range(50):
        trade = simulate_trade(state, result='WIN' if i % 3 != 0 else 'LOSS')
        engine.record_trade(trade)
    
    # Now should_fire should approve
    decision = engine.should_fire(state)
    assert decision['should_fire'] == True
    assert decision['validations']['bayesian']['approved'] == True
```

### Validation Tests (Walk-Forward)
```python
def test_out_of_sample_performance():
    # Train on first 30 days
    train_data = data[:30_days]
    brain = train_on_data(train_data)
    
    # Test on next 10 days (unseen)
    test_data = data[30_days:40_days]
    results = test_on_data(test_data, brain)
    
    # Should maintain performance
    assert results['win_rate'] > 0.50
    assert results['sharpe'] > 0.5
    
    # Should not degrade significantly from training
    assert abs(results['win_rate'] - train_results['win_rate']) < 0.10
```

---

## USER COMMUNICATION GUIDELINES

### What User Expects
- **No fluff:** Get to the point immediately
- **No excessive affirmations:** Skip "great question", "absolutely", etc.
- **Executive brief format:** Bottom line up front, then details
- **Physics/engineering language:** Not business jargon
- **Show, don't tell:** Code examples over explanations

### Response Structure
```
[~X tokens | Y remaining]

## Bottom Line
[2-3 sentence executive summary]

[Details in 2-4 paragraphs unless deep dive requested]

[Code example if relevant]
```

### What NOT to Do
- ❌ Wall of text without structure
- ❌ Explaining things already understood
- ❌ Corporate consultant language (Six Sigma jargon was tolerated but not preferred)
- ❌ Repeating concepts already established
- ❌ Excessive enthusiasm or cheerleading

### What TO Do
- ✅ Direct, functional communication
- ✅ Code first, explanation second
- ✅ Acknowledge when you misunderstood
- ✅ Ask clarifying questions when needed
- ✅ Track token usage
- ✅ Flag heavy operations before executing

---

## PROJECT PHILOSOPHY

### Core Principles

1. **Explainability Over Black Box**
   - Every decision traceable to a state
   - Probabilities visible and inspectable
   - No hidden layers obscuring logic

2. **Statistical Rigor Over Hope**
   - Bayesian validation required
   - Monte Carlo risk analysis mandatory
   - Out-of-sample testing enforced

3. **Context-Awareness Over One-Size-Fits-All**
   - Different parameters for different situations
   - 200 parameters, but only 10-40 active at once
   - Market regimes require different strategies

4. **Continuous Learning Over Static Rules**
   - Brain updates from every trade
   - Parameters evolve over time
   - System adapts to changing markets

5. **Physics Over Indicators**
   - Market as 3-body problem
   - Energy transfer between layers
   - Structural breakdown at Roche limits
   - Resonance when systems align

### What Success Looks Like

**By December 31, 2025:**
```
States learned: 5,000+
High-confidence states: 500+ (>80% win rate)
Out-of-sample validation: 250 days
Win rate: 65%+
Sharpe ratio: 2.0+
Max drawdown: <$500
Trades per day: 2-5
Ready for live trading: YES
```

**The system doesn't predict the future.**  
**It learns which patterns have worked in the past and bets they'll work again.**

---

## FINAL NOTES FOR VS CODE CLAUDE

### Your Role
You are the **implementation engineer**, not the architect. The user has designed this system based on patterns they've observed in real market data. Your job is to:

1. **Build what they've specified** (don't redesign the architecture)
2. **Ask clarifying questions** when implementation details are unclear
3. **Flag potential issues** but defer to user's domain expertise
4. **Optimize for correctness first, speed second**
5. **Write clean, testable code** (user is a professional developer)

### When to Push Back
- **Never:** On the core architecture (9 layers, 3-body model, Bayesian learning)
- **Sometimes:** On implementation details if you see a better way
- **Always:** On code quality issues (bugs, performance problems, maintainability)

### When in Doubt
- Read this document again
- Check existing code patterns in the files
- Ask the user directly
- Default to simplicity over cleverness

### Communication Style Reminder
The user thinks in **physics concepts** (energy, resonance, chaos, entropy) but doesn't care what language you use to explain it. They just want it to work. Be direct, be functional, be helpful.

---

## APPENDIX: QUICK REFERENCE

### Key Files
- `state_vector.py` - 9-layer state definition
- `bayesian_brain.py` - Probability table and learning
- `layer_engine.py` - State computation from market data
- `context_detector.py` - Context identification
- `integrated_statistical_system.py` - Validation framework
- `wave_rider.py` - Position management with regret analysis
- `orchestrator.py` - Main training loop (NEEDS MAJOR UPDATE)

### Key Classes
- `StateVector` - Immutable 9-layer state
- `BayesianBrain` - Probability learning engine
- `LayerEngine` - State computation
- `ContextDetector` - Parameter activation
- `IntegratedStatisticalEngine` - Statistical validation
- `WaveRider` - Position management
- `RegretAnalyzer` - Exit quality tracking

### Key Functions
- `brain.should_fire(state)` - Core decision function
- `engine.compute_current_state(data)` - State calculation
- `detector.detect(state, data, time)` - Context detection
- `validator.validate_state(wins, losses)` - Bayesian validation
- `wave_rider.update_trail(price, state)` - Position management

### Key Concepts
- **State Vector:** Unique 9-layer market snapshot
- **Probability Table:** HashMap of StateVector → win/loss counts
- **Context:** Current market situation determining active parameters
- **Walk-Forward:** Day-by-day optimization with out-of-sample validation
- **Resonance:** All 3 triads aligned (maximum edge opportunity)
- **Volatility Cascade:** Energy transfer between layers (3σ→2σ)
- **Roche Limit:** Structural breakdown point at support/resistance

### Data Locations
- User data: `DATA/glbx-mdp3-20251230-20260129.ohlcv-1s.parquet`
- Models: `models/quantum_probability_table.pkl`
- Checkpoints: `checkpoints/day_N_brain.pkl`, `checkpoints/day_N_params.pkl`

---

**END OF MASTER CONTEXT DOCUMENT**

This document contains everything VS Code Claude needs to understand the project and continue development. Any questions or clarifications should be directed to the user.

Version: 1.0  
Last Updated: February 9, 2026  
Project: Bayesian-AI Trading System  
Status: Phase 2 - Statistical Validation & Walk-Forward Training

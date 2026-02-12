# FULL CASCADE MULTI-TIMEFRAME ARCHITECTURE
**Complete Context Hierarchy: 1 Day → 1 Second**

---

## TIMEFRAME CASCADE (8 Layers)

### Layer 1: DAILY (1 Day Bars)
**Lookback:** 21 days (1 month of context)
**Purpose:** Macro trend, longer-term fair value
**Bars needed:** 21+ daily bars

**Context:**
- Daily trend direction (bull/bear/range market)
- Daily volatility regime (high/low/normal)
- Daily momentum (accelerating/decelerating)
- Position in daily range (near high/low/middle)

**Available:** Day 22+ only (needs 21-day history)
**Cold start (Days 1-21):** Flag as `no_daily_context = True`

---

### Layer 2: 4-HOUR (240 Minutes)
**Lookback:** 21 bars = 3.5 days
**Purpose:** Session context (Asia/Europe/US session trends)
**Bars needed:** 21+ 4-hour bars

**Context:**
- Session-level trend
- Which session is active (Asia/Europe/US/Overlap)
- Session volatility
- Position in session range

**Available:** Day 4+ (needs 3.5 days history)
**Cold start (Days 1-3):** Flag as `no_4h_context = True`

---

### Layer 3: 1-HOUR (60 Minutes)
**Lookback:** 21 bars = 21 hours (~ 1 trading day)
**Purpose:** Intraday wave structure
**Bars needed:** 21+ 1-hour bars

**Context:**
- Hourly trend
- Wave phase (accumulation/markup/distribution/markdown)
- Hourly momentum
- Position in hourly range

**Available:** Day 2+ (needs previous day's data)
**Cold start (Day 1):** Flag as `no_1h_context = True`

---

### Layer 4: 15-MINUTE (Strategic)
**Lookback:** 21 bars = 5.25 hours
**Purpose:** Tactical trend direction (THIS IS DECISION LAYER)
**Bars needed:** 21+ 15-minute bars

**Context:**
- 15m trend direction (UP/DOWN/RANGE) → **Determines LONG or SHORT bias**
- Fair value from 21-bar regression
- Sigma bands (±2σ, ±3σ)
- Lagrange zones (L2_ROCHE, L3_ROCHE)
- Z-score (current price deviation from 15m fair value)

**Available:** After 5.25 hours into Day 1 (warmup complete)
**Cold start (First 5.25h):** Skip trading OR use incomplete regression with flag

**THIS IS THE PRIMARY DECISION TIMEFRAME**

---

### Layer 5: 5-MINUTE (Setup)
**Lookback:** 20 bars = 100 minutes
**Purpose:** Pattern formation and setup quality
**Bars needed:** 20+ 5-minute bars

**Context:**
- 5m microstructure (higher highs/lower lows)
- Pattern type (flag/wedge/triangle)
- Support/resistance levels (5m scale)
- Volume profile (5m)

**Available:** After 100 minutes into Day 1
**Cold start (First 100 min):** Flag as `no_5m_pattern = True`

---

### Layer 6: 1-MINUTE (Confirmation)
**Lookback:** 20 bars = 20 minutes
**Purpose:** Confirm setup is valid, not a fake-out
**Bars needed:** 20+ 1-minute bars

**Context:**
- 1m momentum
- Volume spike confirmation
- Structure break confirmation
- Wick patterns (rejection/absorption)

**Available:** After 20 minutes into Day 1
**Cold start (First 20 min):** Skip trading OR flag as `no_1m_confirm = True`

---

### Layer 7: 15-SECOND (Tactical Entry)
**Lookback:** 20 bars = 300 seconds (5 minutes)
**Purpose:** Precise entry timing within confirmed setup
**Bars needed:** 20+ 15-second bars

**Context:**
- Volume spike (vs 20-bar average)
- Cascade detection (rapid move > threshold)
- Pattern maturity
- Structure confirmation

**Available:** After 5 minutes into Day 1
**Cold start (First 5 min):** Skip trading

---

### Layer 8: 1-SECOND (Execution)
**Lookback:** None (instantaneous)
**Purpose:** Precise entry/exit prices, stop/target management
**Bars needed:** Current bar only

**Context:**
- Exact entry price
- Forward scan for TP/SL hits
- Duration tracking
- Slippage simulation

**Available:** Always
**Cold start:** N/A (no lookback needed)

---

## DATA FLOW WITH CASCADING CONTEXT

```python
# Step 1: Load and resample to all timeframes
raw_1s = load_parquet('1s_bars.parquet')

data_1d = resample_to_daily(raw_1s)
data_4h = resample_to_4hour(raw_1s)
data_1h = resample_to_1hour(raw_1s)
data_15m = resample_to_15min(raw_1s)
data_5m = resample_to_5min(raw_1s)
data_1m = resample_to_1min(raw_1s)
data_15s = resample_to_15sec(raw_1s)

# Step 2: For each training day, prepend previous days' data
for day_idx, current_day_1s in enumerate(all_days):
    
    # Get historical context (from all previous days)
    if day_idx >= 21:
        daily_context = compute_daily_context(data_1d[day_idx-21:day_idx])
    else:
        daily_context = None  # Flag: no_daily_context = True
    
    if day_idx >= 4:
        h4_context = compute_4h_context(data_4h[...])  # Last 21 4h bars
    else:
        h4_context = None  # Flag: no_4h_context = True
    
    if day_idx >= 1:
        h1_context = compute_1h_context(data_1h[...])  # Last 21 1h bars
    else:
        h1_context = None  # Flag: no_1h_context = True
    
    # Step 3: Process current day with cascading context
    for bar_15m in current_day_15m_bars:
        
        # Check warmup complete for this timeframe
        if bar_idx_15m < 21:
            continue  # Skip until 15m warmup complete
        
        # Compute 15m strategic context (PRIMARY DECISION LAYER)
        strategic_15m = compute_strategic_context(
            bars_15m=data_15m[bar_idx_15m-21:bar_idx_15m],
            daily_context=daily_context,    # May be None
            h4_context=h4_context,          # May be None
            h1_context=h1_context           # May be None
        )
        
        # Find corresponding lower timeframes
        start_time = bar_15m.timestamp - 15min
        end_time = bar_15m.timestamp
        
        # 5m setup layer
        bars_5m = data_5m[(data_5m.index >= start_time) & 
                          (data_5m.index < end_time)]
        
        if len(bars_5m) >= 20:  # Warmup complete
            setup_5m = compute_setup_context(bars_5m[-20:])
        else:
            setup_5m = None  # Flag: no_5m_pattern = True
        
        # 1m confirmation layer
        bars_1m = data_1m[(data_1m.index >= end_time-20min) & 
                          (data_1m.index < end_time)]
        
        if len(bars_1m) >= 20:
            confirm_1m = compute_confirmation_context(bars_1m[-20:])
        else:
            confirm_1m = None  # Flag: no_1m_confirm = True
        
        # 15s tactical entry layer
        bars_15s = data_15s[(data_15s.index >= end_time-5min) & 
                            (data_15s.index < end_time)]
        
        if len(bars_15s) >= 20:
            for bar_15s in bars_15s[-20:]:  # Last 20 bars
                tactical_15s = compute_tactical_signals(
                    bar_15s,
                    strategic_15m,  # Pass down 15m context
                    setup_5m,       # Pass down 5m setup
                    confirm_1m      # Pass down 1m confirmation
                )
                
                # Step 4: Check if should enter
                if should_enter_trade(strategic_15m, setup_5m, confirm_1m, tactical_15s):
                    
                    # Find exact 1s bar for entry
                    bar_1s = find_1s_bar(current_day_1s, bar_15s.timestamp)
                    
                    # Execute trade with full context
                    result = simulate_trade(
                        bars_1s=current_day_1s,
                        entry_bar=bar_1s,
                        direction=strategic_15m.trade_direction,  # From 15m
                        params=current_params,
                        context={
                            'daily': daily_context,
                            'h4': h4_context,
                            'h1': h1_context,
                            'strategic_15m': strategic_15m,
                            'setup_5m': setup_5m,
                            'confirm_1m': confirm_1m,
                            'tactical_15s': tactical_15s
                        }
                    )
```

---

## CONTEXT AVAILABILITY MATRIX

| Day | 1d (21d) | 4h (21×4h) | 1h (21h) | 15m (21×15m) | 5m | 1m | 15s | 1s |
|-----|----------|------------|----------|--------------|----|----|-----|----| 
| 1 | ❌ | ❌ | ❌ | ✓ (after 5.25h) | ✓ | ✓ | ✓ | ✓ |
| 2 | ❌ | ❌ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | ❌ | ❌ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 | ❌ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 22+ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## COLD START HANDLING

### Strategy 1: Skip Trading (Conservative)
```python
def should_enter_trade(context):
    # Require minimum context levels
    if not context['strategic_15m']:
        return False  # No 15m context yet (first 5.25h)
    
    if not context['h1'] and day_idx < 2:
        return False  # No 1h context on Day 1
    
    if not context['daily'] and day_idx < 22:
        # Trade with reduced confidence on Days 1-21
        confidence_multiplier = 0.7
    else:
        confidence_multiplier = 1.0
    
    # Rest of decision logic...
```

### Strategy 2: Fallback with Flags (Aggressive)
```python
def compute_strategic_context(bars_15m, daily_context, h4_context, h1_context):
    """
    Compute 15m strategic layer with optional higher timeframe context
    """
    
    # 15m regression (always available after warmup)
    center, sigma, slope = compute_regression_21bar(bars_15m)
    z_score = (bars_15m[-1].close - center) / sigma
    
    # Higher timeframe filters (if available)
    if daily_context:
        # Confirm 15m trend aligns with daily
        if daily_context.trend == 'UP' and z_score >= 2.0:
            confidence_modifier = 0.5  # Counter-trend, reduce confidence
        elif daily_context.trend == 'UP' and z_score <= -2.0:
            confidence_modifier = 1.5  # With-trend, increase confidence
        else:
            confidence_modifier = 1.0
    else:
        # No daily context - trade but flag it
        confidence_modifier = 0.8  # Slight reduction
        flags = {'no_daily_context': True}
    
    if h1_context:
        # Check 1h momentum alignment
        if h1_context.momentum > 0 and slope_15m > 0:
            confidence_modifier *= 1.2  # Aligned momentum
        elif h1_context.momentum * slope_15m < 0:
            confidence_modifier *= 0.7  # Divergent momentum
    else:
        flags['no_1h_context'] = True
    
    return {
        'fair_value': center,
        'sigma': sigma,
        'z_score': z_score,
        'trend_direction': 'UP' if slope > 0 else 'DOWN',
        'confidence_modifier': confidence_modifier,
        'flags': flags  # Track what context is missing
    }
```

### Strategy 3: Gradual Warmup (Recommended)
```python
# Don't trade at all until minimum context available
MINIMUM_REQUIREMENTS = {
    'day_1': {
        'skip_first_minutes': 315,  # 5.25 hours for 15m warmup
        'required': ['15m'],
        'confidence': 0.6
    },
    'day_2_to_3': {
        'skip_first_minutes': 0,
        'required': ['15m', '1h'],
        'confidence': 0.7
    },
    'day_4_to_21': {
        'skip_first_minutes': 0,
        'required': ['15m', '1h', '4h'],
        'confidence': 0.85
    },
    'day_22_plus': {
        'skip_first_minutes': 0,
        'required': ['15m', '1h', '4h', '1d'],
        'confidence': 1.0  # Full context available
    }
}
```

---

## STATE DEFINITION WITH FULL CONTEXT

```python
@dataclass(frozen=True)
class ThreeBodyQuantumState:
    # Layer 1: Daily context (available Day 22+)
    daily_trend: Optional[str]        # 'BULL', 'BEAR', 'RANGE', None
    daily_volatility: Optional[str]   # 'HIGH', 'NORMAL', 'LOW', None
    
    # Layer 2: 4-hour context (available Day 4+)
    h4_trend: Optional[str]           # 'UP', 'DOWN', 'RANGE', None
    session: Optional[str]            # 'ASIA', 'EUROPE', 'US', 'OVERLAP', None
    
    # Layer 3: 1-hour context (available Day 2+)
    h1_trend: Optional[str]           # 'UP', 'DOWN', 'RANGE', None
    h1_wave_phase: Optional[str]      # 'ACCUMULATION', 'MARKUP', etc., None
    
    # Layer 4: 15-minute strategic (ALWAYS AVAILABLE after warmup)
    trend_direction_15m: str          # 'UP', 'DOWN', 'RANGE' (REQUIRED)
    z_score_15m: float                # Deviation from 15m fair value (REQUIRED)
    lagrange_zone_15m: str            # 'L2_ROCHE', 'L3_ROCHE', 'STABLE' (REQUIRED)
    sigma_15m: float                  # 15m volatility (REQUIRED)
    
    # Layer 5: 5-minute setup (ALWAYS AVAILABLE after Day 1 warmup)
    pattern_type_5m: Optional[str]    # 'FLAG', 'WEDGE', 'TRIANGLE', None
    
    # Layer 6: 1-minute confirmation
    volume_spike_1m: bool
    structure_break_1m: bool
    
    # Layer 7: 15-second tactical
    cascade_detected_15s: bool
    structure_confirmed_15s: bool
    momentum_strength_15s: str        # 'LOW', 'MEDIUM', 'HIGH'
    
    # Layer 8: 1-second execution (metadata, not in state)
    timestamp: float
    price: float
    
    # Context availability flags
    context_complete: bool            # True if all layers available
    missing_layers: List[str]         # ['daily', '4h'] if Day 1-3
```

---

## HASH KEY WITH CONTEXT HIERARCHY

```python
def __hash__(self):
    """
    Hash for brain lookup - adapt to available context
    """
    
    # Core (always available on Day 1 after warmup)
    core_hash = (
        self._categorize_z_score(self.z_score_15m),  # 5 categories
        self.trend_direction_15m,                     # UP/DOWN/RANGE
        self.lagrange_zone_15m,                       # L2/L3/STABLE
        self.momentum_strength_15s,                   # LOW/MED/HIGH
        self.cascade_detected_15s,                    # Bool
        self.structure_confirmed_15s                  # Bool
    )
    
    # Add higher timeframe context if available
    if self.daily_trend:
        # Day 22+: Include daily context
        return hash(core_hash + (self.daily_trend, self.daily_volatility))
    elif self.h4_trend:
        # Day 4-21: Include 4h context
        return hash(core_hash + (self.h4_trend, self.session))
    elif self.h1_trend:
        # Day 2-3: Include 1h context
        return hash(core_hash + (self.h1_trend,))
    else:
        # Day 1: Core only
        return hash(core_hash)
```

**State space:**
- **Day 1:** 5 × 3 × 3 × 3 × 2 × 2 = **540 states**
- **Day 2+:** 540 × 3 (h1_trend) = **1,620 states**
- **Day 4+:** 1,620 × 3 (h4_trend) × 4 (session) = **19,440 states**
- **Day 22+:** 19,440 × 3 (daily_trend) × 3 (daily_vol) = **175,000 states**

**But in practice:**
- Brain will see ~500-2000 unique states across 27 days
- States from Day 1 (core-only) will match states from Day 22+ when higher context ignored
- Allows learning with partial context early, refinement with full context later

---

## TRADING LOGIC WITH CASCADING CONTEXT

```python
def should_enter_trade(state, brain, params):
    """
    Decision with context-aware confidence
    """
    
    # Get brain probability
    prob, conf = brain.get_probability(state)
    
    # Adjust confidence based on available context
    if state.context_complete:
        # Full context available (Day 22+)
        required_conf = params['min_confidence']  # 0.30
    elif 'daily' not in state.missing_layers:
        # Partial context (Day 4-21)
        required_conf = params['min_confidence'] * 1.2  # 0.36
    elif '1h' in state.missing_layers:
        # Minimal context (Day 1)
        required_conf = params['min_confidence'] * 1.5  # 0.45
        # OR skip trading entirely on Day 1
    
    if conf < required_conf:
        return False
    
    # Rest of decision logic...
    # Direction from 15m (always available)
    if state.z_score_15m >= 2.0:
        direction = 'SHORT'  # Mean reversion from top
    elif state.z_score_15m <= -2.0:
        direction = 'LONG'   # Mean reversion from bottom
    else:
        return False  # Not at extremes
    
    # Confirm with higher timeframes if available
    if state.daily_trend:
        # Day 22+: Check daily alignment
        if direction == 'LONG' and state.daily_trend == 'BEAR':
            return False  # Counter-trend, risky
        elif direction == 'SHORT' and state.daily_trend == 'BULL':
            return False
    
    return True
```

---

## IMPLEMENTATION TIME ESTIMATE

| Task | Time |
|------|------|
| Add resampling for all 8 timeframes | 1 hour |
| Modify quantum field engine for cascade | 2 hours |
| Update state definition with Optional fields | 1 hour |
| Implement context availability logic | 1 hour |
| Add cold start handling | 1 hour |
| Fix trade direction (Bug #1) | 30 min |
| Fix state hashing (Bug #3) | 30 min |
| Add trading costs (Bug #2 validation) | 30 min |
| **Add intermediate progress updates** | 30 min |
| **Add regret analysis to progress** | 30 min |
| **Total coding** | **~9 hours** |
| Re-train 27 days | 4 hours |
| **TOTAL** | **~13 hours** |

---

## ADDITIONAL FEATURES

### Feature 1: Intermediate Progress Updates

**File:** `training/orchestrator.py` in `optimize_day()` method

**What to add:**
```python
for iteration in tqdm(range(self.n_iterations_per_day), desc=f"Optimizing Day {day_idx + 1}"):
    params = self.param_generator.generate_parameter_set(iteration, day_idx, context)
    result = self.simulate_trading_day(day_data, params, brain_copy)
    
    # Track best
    if result['sharpe'] > best_sharpe:
        best_sharpe = result['sharpe']
        best_params = params
        best_result = result
        
        # IMMEDIATE FEEDBACK when better params found
        regret_str = ""
        if 'avg_exit_efficiency' in result:
            regret_str = f"| ExitEff: {result['avg_exit_efficiency']:.1%}"
            if result.get('early_exits_pct', 0) > 50:
                regret_str += " [⚠ High early exits]"
        
        print(f"\n  [Iter {iteration:3d}] New best! Sharpe: {best_sharpe:.2f} | WR: {result['win_rate']:.1%} | Trades: {result['trades']} | P&L: ${result['pnl']:.2f} {regret_str}")
    
    # PERIODIC PROGRESS (every 10 iterations)
    if (iteration + 1) % 10 == 0:
        progress_str = f"\n  [Progress] {iteration + 1}/{self.n_iterations_per_day} complete | Current best: Sharpe {best_sharpe:.2f}, {best_result['trades']} trades"
        
        # Add regret summary if available
        if best_result and 'avg_exit_efficiency' in best_result:
            progress_str += f"\n             Exit Efficiency: {best_result['avg_exit_efficiency']:.1%} | Early: {best_result.get('early_exits_pct', 0):.0%} | Late: {best_result.get('late_exits_pct', 0):.0%}"
        
        print(progress_str)
```

**Output example:**
```
Optimizing Day 1:   7%|███▏| 7/100 [02:43<35:21, 22.8s/it]
  [Iter   7] New best! Sharpe: 1.05 | WR: 62.5% | P&L: $412.00 | ExitEff: 72.5% | Early: 38%

  [Progress] 10/100 complete | Current best: Sharpe 1.05, 24 trades
             Exit Efficiency: 72.5% | Early exits: 38% | Late exits: 12%
```

**Benefits:**
- See improvements immediately (not silent for 40 minutes)
- Track exit efficiency in real-time
- Warning when >50% exits are early (trail stops too tight)
- Progress checkpoints every 10 iterations

---

### Feature 2: Multi-Timeframe Regret Analysis

**File:** `execution/batch_regret_analyzer.py`

**Enhancement: Use multiple timeframes for peak detection**

**Current approach:**
```python
# Single timeframe (15s)
peak = max(prices_during_trade)
```

**Multi-timeframe approach:**
```python
def batch_analyze_day(self, all_trades_today, full_day_data_1s):
    """
    Batch regret analysis with multi-timeframe context
    """
    
    # Resample to multiple timeframes for peak detection
    data_15s = resample_to_15sec(full_day_data_1s)
    data_1m = resample_to_1min(full_day_data_1s)
    data_2m = resample_to_2min(full_day_data_1s)
    data_5m = resample_to_5min(full_day_data_1s)
    
    analyses = []
    
    for trade in all_trades_today:
        # Find peak on multiple timeframes
        peak_15s = self._find_peak(data_15s, trade.entry_time, trade.exit_time)
        peak_1m = self._find_peak(data_1m, trade.entry_time, trade.exit_time)
        peak_2m = self._find_peak(data_2m, trade.entry_time, trade.exit_time)  # User's suggestion
        peak_5m = self._find_peak(data_5m, trade.entry_time, trade.exit_time)
        
        # Use 2m peak as "true" target (sustained level, not spike)
        true_peak = peak_2m
        
        # Calculate regret
        potential_pnl = (true_peak - trade.entry_price) * trade.direction
        actual_pnl = trade.pnl
        efficiency = actual_pnl / potential_pnl if potential_pnl > 0 else 0
        
        # Classify exit type with context
        if efficiency >= 0.90:
            exit_type = 'optimal'
        elif efficiency < 0.70 and peak_1m > peak_2m * 1.05:
            exit_type = 'closed_too_early_spike'  # Exited before spike
        elif efficiency < 0.70 and peak_2m == peak_5m:
            exit_type = 'closed_too_early_trend'  # Exited early in sustained move
        else:
            exit_type = 'closed_too_late'  # Gave back profits
        
        analyses.append({
            'trade': trade,
            'peak_15s': peak_15s,
            'peak_1m': peak_1m,
            'peak_2m': peak_2m,  # True peak
            'peak_5m': peak_5m,
            'efficiency': efficiency,
            'exit_type': exit_type,
            'pnl_left_on_table': potential_pnl - actual_pnl
        })
    
    # Aggregate patterns
    return self._generate_report(analyses)
```

**Report output:**
```
BATCH REGRET ANALYSIS (Multi-Timeframe):
================================================================================

EXIT EFFICIENCY: 74.3%
  
PEAK ANALYSIS:
  15s peaks: Avg deviation from 2m peak: 12.5% (noise)
  1m peaks: Avg deviation from 2m peak: 4.2% (moderate noise)
  2m peaks: Used as "true" target (sustained levels)
  5m peaks: Avg deviation from 2m peak: 1.8% (macro confirmation)

EXIT TYPE BREAKDOWN:
  Optimal (>90% eff):           18 trades (36%)
  Early (spike):                 8 trades (16%) - Exited before 15s/1m spike
  Early (trend):                12 trades (24%) - Exited in sustained 2m/5m move
  Late (gave back):             12 trades (24%) - Held too long

RECOMMENDATIONS:
  • 24% early exits in trends → WIDEN trail stops by 5-10 ticks
  • Early spike exits acceptable (noise, not trend)
  • 24% late exits → Consider tightening max hold time
  
CONTEXT ANALYSIS:
  In 15m UPTREND:
    - Exit efficiency: 68.2% (too early often)
    - Recommendation: Widen stops to 25 ticks when 15m trending
  
  In 15m RANGE:
    - Exit efficiency: 82.5% (working well)
    - Recommendation: Keep current 15 tick stops in range
```

---

### Feature 3: Context-Aware Regret Analysis

**Link regret to multi-timeframe context:**

```python
def analyze_with_context(self, trade, analysis, state):
    """
    Enhance regret analysis with multi-timeframe context
    """
    
    # Check what context was available
    if state.daily_trend:
        # Full context (Day 22+)
        context_level = 'FULL'
    elif state.h4_trend:
        # Partial context (Day 4-21)
        context_level = 'PARTIAL'
    else:
        # Minimal context (Day 1-3)
        context_level = 'MINIMAL'
    
    # Analyze exit quality relative to context
    if state.trend_direction_15m == 'UP' and analysis['exit_type'] == 'closed_too_early_trend':
        # Exited early in 15m uptrend
        recommendation = {
            'issue': 'Early exit in 15m uptrend',
            'suggestion': 'Widen trail stops when 15m trending',
            'param_adjust': {
                'trail_distance_wide': '+10 ticks',
                'condition': '15m_trend_strength > 0.7'
            }
        }
    
    elif state.daily_trend == 'BULL' and analysis['exit_type'] == 'closed_too_early_trend':
        # Exited early in daily bull AND 15m up
        recommendation = {
            'issue': 'Early exit with daily+15m alignment',
            'suggestion': 'SIGNIFICANTLY widen stops when multi-TF aligned',
            'param_adjust': {
                'trail_distance_wide': '+20 ticks',
                'condition': 'daily_trend == 15m_trend'
            }
        }
    
    return recommendation
```

---

## COMBINED IMPLEMENTATION ORDER

### Phase 1: Core Fixes (Day 1)
1. ✅ Add all 8 timeframe resampling
2. ✅ Fix trade direction (Bug #1)
3. ✅ Fix state hashing (Bug #3)
4. ✅ Add trading costs

### Phase 2: Multi-Timeframe Integration (Day 1-2)
5. ✅ Modify quantum field engine for cascade
6. ✅ Update state definition with Optional fields
7. ✅ Implement context availability logic
8. ✅ Add cold start handling

### Phase 3: Monitoring & Feedback (Day 2)
9. ✅ Add intermediate progress updates
10. ✅ Enhance regret analysis with multi-timeframe peaks
11. ✅ Add context-aware regret recommendations

### Phase 4: Testing (Day 2-3)
12. ✅ Test on Day 1 only (validate cascade works)
13. ✅ Test on Days 1-5 (validate context buildup)
14. ✅ Full 27-day training

---

**This is the complete package:**
- 8-layer cascade architecture
- Cold start handling
- Bug fixes (direction, hashing, costs)
- Real-time progress updates
- Multi-timeframe regret analysis
- Context-aware recommendations

**Total: ~13 hours implementation**

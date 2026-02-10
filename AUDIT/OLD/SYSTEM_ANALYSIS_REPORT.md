# BAYESIAN-AI SYSTEM ANALYSIS REPORT
**Date**: 2026-02-09
**Training Run**: 27 days (2025-12-30 to 2026-01-29)
**Asset**: MNQ Futures (Micro E-mini Nasdaq)

---

## 1. ARCHITECTURE OVERVIEW

The system models market microstructure as a **three-body gravitational system** with quantum mechanics metaphors. It learns which market "states" produce profitable trades via a Bayesian probability table, and optimizes trade parameters (TP, SL, hold time, confidence threshold) using Design of Experiments (DOE).

### Data Flow
```
Raw 1s OHLCV Data
    |
    v
Resample to 15s bars (cached as parquet)
    |
    v
Quantum Field Engine (vectorized batch)
  - 21-bar rolling linear regression -> center, sigma
  - Z-score, force fields, wave function
  - Measurement operators (volume spike, cascade, spin)
  - Lagrange zone classification
  - Tunneling probabilities
    |
    v
ThreeBodyQuantumState (per 15s bar)
    |
    v
Filter Pipeline
  - Must be in L2_ROCHE or L3_ROCHE (|z| >= 2.0)
  - structure_confirmed = True (volume spike + pattern maturity)
  - cascade_detected = True (|tick_velocity| > 1.0 pt)
  - Brain confidence >= 0.30 OR unseen state
    |
    v
DOE Parameter Optimization (1000 iterations/day)
  - Iter 0-9: Hand-tuned baselines
  - Iter 10-509: Latin Hypercube sampling
  - Iter 510-799: Mutation around best params
  - Iter 800-999: Crossover between top sets
    |
    v
Trade Simulation (on 1s data for precision)
  - Entry at candidate bar price
  - Forward scan up to 200 bars
  - Exit: TP hit, SL hit, max_hold exceeded, or EOD
    |
    v
Best Sharpe iteration selected -> Brain updated -> Next day
```

### Key Files
| File | Role |
|------|------|
| `core/three_body_state.py` | Frozen dataclass with ~40 fields, custom `__hash__`/`__eq__` for brain lookups |
| `core/quantum_field_engine.py` | Computes states per bar, vectorized `batch_compute_states()` |
| `core/bayesian_brain.py` | HashMap: State -> {wins, losses, total}, Laplace-smoothed probability |
| `training/orchestrator.py` | Main loop: day splitting, precompute, DOE, simulation, brain update |
| `training/doe_parameter_generator.py` | 4-phase parameter generation (baseline/LHS/mutation/crossover) |
| `execution/batch_regret_analyzer.py` | End-of-day exit efficiency analysis |
| `visualization/live_training_dashboard.py` | Tkinter dashboard polling training_progress.json |

---

## 2. LATEST TRAINING RESULTS

### Summary
| Metric | Value |
|--------|-------|
| Days Trained | 27 / 27 |
| Date Range | 2025-12-30 to 2026-01-29 |
| Total Runtime | ~4 hours 11 minutes |
| Total Trades (best iter) | 5,913 |
| Total P&L | $17,059.50 |
| Cumulative Win Rate | 40.18% |
| Sharpe Ratio | 0.340 |
| Max Drawdown | $120.75 |
| States Learned | 5,877 |
| High Confidence States | **0** |
| Avg Trade Duration | 137.5 seconds |

### Best Parameters Found (Day 27)
| Parameter | Value |
|-----------|-------|
| Take Profit | 14.8 pts (59.2 ticks) |
| Stop Loss | 2.8 pts (11.2 ticks) |
| TP:SL Ratio | **5.29 : 1** |
| Confidence Threshold | 0.85 |
| Max Hold | 1,268 seconds (~21 min) |

### Day-by-Day Performance
| Day | Date | Trades | WR | P&L | Sharpe |
|-----|------|--------|----|-----|--------|
| 1 | 2025-12-30 | 253 | 29% | $249 | 0.15 |
| 2 | 2025-12-31 | 274 | 40% | $597 | 0.26 |
| 3 | 2026-01-01 | 11 | 45% | $20 | 0.38 |
| 4 | 2026-01-02 | 280 | 41% | $989 | 0.43 |
| 5 | 2026-01-04 | 15 | 67% | $110 | 0.74 |
| 6 | 2026-01-05 | 302 | 42% | $918 | 0.35 |
| 7 | 2026-01-06 | 280 | 56% | $1,047 | 0.41 |
| 8 | 2026-01-07 | 267 | 49% | $601 | 0.25 |
| 9 | 2026-01-08 | 285 | 33% | $739 | 0.32 |
| 10 | 2026-01-09 | 269 | 51% | $1,003 | 0.38 |
| 11 | 2026-01-11 | 12 | 17% | **-$5** | -0.11 |
| 12 | 2026-01-12 | 295 | 45% | $705 | 0.25 |
| 13 | 2026-01-13 | 275 | 49% | $945 | 0.35 |
| 14 | 2026-01-14 | 279 | 29% | $629 | 0.30 |
| 15 | 2026-01-15 | 284 | 32% | $833 | 0.37 |
| 16 | 2026-01-16 | 226 | 50% | $814 | 0.40 |
| 17 | 2026-01-18 | 9 | 44% | $46 | 0.55 |
| 18 | 2026-01-19 | 193 | 37% | $573 | 0.37 |
| 19 | 2026-01-20 | 306 | 29% | $724 | 0.30 |
| 20 | 2026-01-21 | 285 | 36% | $815 | 0.40 |
| 21 | 2026-01-22 | 265 | 26% | $512 | 0.26 |
| 22 | 2026-01-23 | 252 | 30% | $603 | 0.31 |
| 23 | 2026-01-25 | 13 | 46% | $55 | 0.56 |
| 24 | 2026-01-26 | 252 | 54% | $1,081 | 0.44 |
| 25 | 2026-01-27 | 256 | 55% | $1,052 | 0.44 |
| 26 | 2026-01-28 | 256 | 36% | $681 | 0.36 |
| 27 | 2026-01-29 | 219 | 35% | $725 | 0.40 |

26 of 27 days profitable. Only loss: Day 11 (-$5, weekend, 12 trades).

---

## 3. CRITICAL BUGS FOUND

### BUG #1: ALL TRADES ARE LONG-ONLY (Direction Ignored)

**Severity**: CRITICAL
**Location**: `orchestrator.py:581` (CPU), `orchestrator.py:814` (GPU)

The trade simulation always computes P&L as:
```python
pnl = price - entry_price  # ALWAYS LONG
```

But the physics model explicitly defines directions in `three_body_state.py:156-171`:
```python
if self.z_score > 2.0:   # L2_ROCHE (price ABOVE fair value)
    return {'action': 'SELL', ...}   # SHORT for mean reversion
else:                      # L3_ROCHE (price BELOW fair value)
    return {'action': 'BUY', ...}    # LONG for mean reversion
```

**`get_trade_directive()` is NEVER CALLED anywhere in the orchestrator or simulation.**

**Impact**: At L2_ROCHE (z >= 2.0, ~50% of candidates), the system goes LONG when the
physics model says SHORT. The mean-reversion thesis is broken for half the trades.

### BUG #2: P&L IS INFLATED BY ASYMMETRIC TP:SL (Not by Learning)

**Severity**: HIGH
**Location**: The DOE optimizer finds extreme TP:SL ratios

With the best params (TP=14.8, SL=2.8), breakeven win rate is:
```
Breakeven WR = SL / (TP + SL) = 2.8 / 17.6 = 15.9%
```

So a **random coin flip** with these params would be profitable at any WR > 16%. The
system's 40% WR is way above breakeven, producing apparent profits even though:
- The brain contributes nothing (all states unseen)
- Half the trades are in the wrong direction
- There is no actual pattern recognition happening

**The DOE is optimizing the lottery ticket, not learning market patterns.**

With 1 iteration (baseline params: TP=10 pts, SL=3.75 pts):
```
Breakeven WR = 3.75 / 13.75 = 27.3%
Expected at 46% WR = 0.46 * 10.0 - 0.54 * 3.75 = +$2.58/trade
```
This is why 46% WR with 1 iteration still shows positive P&L — the asymmetric
TP:SL ratio makes it mathematically inevitable, not a sign of good strategy.

### BUG #3: BRAIN NEVER LEARNS (State Space Fragmentation)

**Severity**: CRITICAL
**Location**: `three_body_state.py:91-111`

The hash key uses 6 dimensions:
```python
hash((z_bin, momentum_bin, lagrange_zone, structure_confirmed, cascade_detected, spin_inverted))
```

Where:
- `z_bin = int(z * 2) / 2` → 0.5 steps: ~20+ values for candidates
- `momentum_bin = int(momentum_strength * 10) / 10` → 0.1 steps: **hundreds to thousands** of values
- `lagrange_zone` → 2 relevant (L2_ROCHE, L3_ROCHE)
- `structure_confirmed` → always True (filtered)
- `cascade_detected` → always True (filtered)
- `spin_inverted` → 2 values

**momentum_strength = F_momentum / (F_reversion + 1e-6)** is unbounded. Sample values from
debug log: velocities of 1.25, 3.50, 5.75, 10.00, 12.75, 20.25, 217.75...

The combinatorial space: **~20 z-bins x ~1000+ momentum_bins x 2 zones x 2 spin = 80,000+ possible states**

Results after 27 days of training:
- 5,877 unique states learned
- 5,913 total trades
- **~1 trade per state on average**
- **0 states have reached confidence** (needs 10+ observations at >=80% WR)
- Every candidate shows `prob=1.00 conf=0.00` in debug log = unseen state

**The brain is a write-only database. Patterns from Day 1 are never retrieved on Day 2
because the momentum_bin values differ even for identical market conditions.**

### BUG #4: NO CROSS-DAY DATA CONTINUITY

**Severity**: MEDIUM
**Location**: `orchestrator.py:1173-1191`

`split_into_trading_days()` uses `data.groupby('date')`. Each day starts fresh:
- The 21-bar regression window starts from bar 0 of the new day
- First ~5.25 minutes (21 x 15s) of each day have incomplete regression windows
- Regression-based z-scores are less meaningful at day boundaries

The brain persists across days, but since states never match (Bug #3), this
doesn't help.

### BUG #5: SINGLE TIMEFRAME (Missing Macro)

**Severity**: MEDIUM
**Location**: `quantum_field_engine.py:45-51`

The engine was designed for two timeframes:
- `df_macro` (15min bars) — for the 21-bar regression (5.25 hour lookback)
- `df_micro` (15sec bars) — for measurements

But both receive the same 15s data. The 21-bar regression on 15s = **5.25 minutes**,
not the intended 5.25 hours. This makes the "fair value" estimation extremely
short-term and noisy.

### BUG #6: SIGMA IS std_err NOT std

**Severity**: LOW-MEDIUM
**Location**: `quantum_field_engine.py:140-143`

```python
slope, intercept, _, _, std_err = linregress(x, y)
sigma = std_err if std_err > 0 else y.std()
```

`std_err` is the standard error of the slope estimate (uncertainty of the regression
line), NOT the standard deviation of price residuals. `std_err` is typically much
smaller than `y.std()`, which:
- Inflates z-scores (prices appear more extreme than they are)
- Makes more bars appear in ROCHE zones
- Makes the +/-2σ bands too tight

---

## 4. PRECOMPUTE STATE DISTRIBUTIONS

Consistent across all 27 days (typical full trading day with ~5,200-5,500 bars):

### Zone Distribution
| Zone | % | Meaning |
|------|---|---------|
| L1_STABLE | 51-54% | Price near fair value (|z| < 1.0) |
| CHAOS | 30-33% | Transition zone (1.0 <= |z| < 2.0) |
| L2_ROCHE | 7-9% | Upper extreme (z >= 2.0) |
| L3_ROCHE | 7-9% | Lower extreme (z <= -2.0) |

### Filter Pipeline (typical full day)
| Stage | Count | % | Description |
|-------|-------|---|-------------|
| Total bars | ~5,300 | 100% | All 15s bars after warmup |
| In ROCHE zone | ~900 | 16-18% | At +/-2 sigma |
| structure_confirmed | ~320 | 5-6% | Volume spike + maturity |
| cascade_detected | ~2,500 | 40-70% | Price move > 1 pt |
| Both (struct + cascade) | ~290 | 5-6% | Pass both checks |
| **Final candidates** | **~260** | **4-6%** | Plus brain gate |

Weekend/holiday days: only ~200-220 bars, 9-15 candidates.

---

## 5. COMPONENT DEEP DIVE

### 5.1 Quantum Field Engine

**What it does**: Computes a ThreeBodyQuantumState for each bar by treating the market
as a three-body gravitational system.

**Key thresholds (hardcoded)**:
| Threshold | Value | Used For |
|-----------|-------|----------|
| SIGMA_ROCHE_MULTIPLIER | 2.0 | +/-2σ Roche limit (trade zones) |
| SIGMA_EVENT_MULTIPLIER | 3.0 | +/-3σ event horizon (point of no return) |
| TIDAL_FORCE_EXPONENT | 2.0 | F_reversion = z^2/9 |
| regression_period | 21 | Rolling window for center/sigma |
| volume_spike_factor | 1.2x | Volume > 1.2 * 20-bar mean |
| pattern_maturity_min | 0.5 | (|z| - 2.0) / 1.0 must exceed 0.5 |
| cascade_threshold | 1.0 pt | |tick_velocity| > 1.0 |

### 5.2 Bayesian Brain

**What it does**: HashMap from state → {wins, losses, total}

**Key thresholds**:
| Threshold | Value | Purpose |
|-----------|-------|---------|
| Prior | 0.50 | Neutral (Laplace smoothing: (wins+1)/(total+2)) |
| min_prob | 0.80 | Minimum win probability to fire |
| min_conf | 0.30 | ~10 prior trades in same state bucket |
| Full confidence | 30 trades | conf = min(total/30, 1.0) |
| High-prob cutoff | 0.80 + 10 samples | For `get_all_states_above_threshold()` |

**Current status**: Effectively non-functional due to state space fragmentation (Bug #3).

### 5.3 DOE Parameter Generator

**4-phase strategy across 1000 iterations/day**:

| Phase | Iterations | Method |
|-------|-----------|--------|
| Baseline | 0-9 | 10 hand-tuned presets (conservative, aggressive, scalper, etc.) |
| Latin Hypercube | 10-509 | Random sampling with good coverage |
| Mutation | 510-799 | ±10-20% variation around best known params |
| Crossover | 800-999 | 50/50 combination of two best sets |

**Parameter ranges**:
| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| stop_loss_ticks | 10 | 25 | ticks (x0.25 = 2.5-6.25 pts) |
| take_profit_ticks | 30 | 60 | ticks (x0.25 = 7.5-15 pts) |
| confidence_threshold | 0.70 | 0.90 | probability |
| max_hold_seconds | 300 | 900 | seconds |

### 5.4 Trade Simulation

**How it works**:
1. For each candidate bar that passes filters, enter LONG at entry_price
2. Scan forward up to 200 bars (200 seconds at 1s resolution)
3. Exit on first of: TP hit (WIN), SL hit (LOSS), max_hold exceeded (P&L-dependent), EOD

**Critical note**: Always LONG. Never SHORT. Direction from `get_trade_directive()` is
never consulted. See Bug #1.

### 5.5 Batch Regret Analyzer

**What it does**: Post-day analysis of exit quality
- Resamples to 2-min bars for broader context
- Looks 5 minutes past each exit
- Computes: peak favorable price, potential max P&L, P&L left on table
- Classifies: optimal (>=90% efficiency), closed_too_early, closed_too_late
- Generates recommendations (widen trails, tighten stops)

---

## 6. WHY P&L IS POSITIVE DESPITE LOW WIN RATE

### The Math

With TP=14.8 pts and SL=2.8 pts:
```
Breakeven WR = SL / (TP + SL) = 2.8 / 17.6 = 15.9%
```

| Win Rate | Expected $/trade | 250 trades/day |
|----------|-------------------|----------------|
| 16% (breakeven) | $0.00 | $0 |
| 30% | $2.76 | $690 |
| 40% | $4.24 | $1,060 |
| 46% | $5.31 | $1,328 |
| 50% | $5.92 | $1,480 |

A random strategy with these TP:SL params and 50/50 coin flip would produce
~$1,480/day profit on 250 trades. The system's actual ~$630/day average suggests
it may be performing **worse** than random with these params.

### What the DOE is Actually Doing

The DOE isn't finding "good trading signals" — it's finding the most profitable
TP:SL asymmetry. Given enough random entries in a volatile market:
- Tight stop (2.8 pts) = hits often on noise = many small losses
- Wide target (14.8 pts) = hits occasionally on trends = few large wins
- Net positive because wins >> losses in size

This works in backtest but has serious live-trading problems:
- **Slippage destroys tight stops**: 2.8 pts = 11 ticks. With 1-2 tick slippage,
  effective SL becomes 3.3-3.8 pts, raising breakeven WR
- **Spread costs compound**: 250 trades/day × 1-2 tick spread = 62-125 pts/day lost
- **No directional edge**: The model doesn't know whether to go long or short

---

## 7. WHAT IS ACTUALLY WORKING VS. WHAT ISN'T

### Working
1. **Data pipeline**: 1s → 15s resampling with parquet caching is fast and correct
2. **Vectorized computation**: batch_compute_states() processes ~5,500 bars in 0.2s
3. **DOE framework**: 4-phase parameter optimization converges to good TP:SL ratios
4. **Dashboard**: Real-time progress visualization with cumulative metrics
5. **Filter pipeline**: Correctly identifies high-volatility moments (ROCHE + cascade + volume)

### Not Working
1. **Bayesian learning**: Brain never reaches confidence, patterns don't transfer across days
2. **Trade direction**: Always LONG, ignoring the model's SELL signals at L2_ROCHE
3. **Mean reversion thesis**: The system doesn't actually trade mean reversion
4. **Multi-timeframe**: Only 15s used, missing the 15min macro context
5. **Sigma calculation**: std_err instead of price std makes zones too wide

---

## 8. RECOMMENDED FIXES (Priority Order)

### P0: Fix Trade Direction
- Consult `state.z_score` to determine LONG vs SHORT
- L2_ROCHE (z >= 2.0): SHORT (`pnl = entry_price - price`)
- L3_ROCHE (z <= -2.0): LONG (`pnl = price - entry_price`)

### P1: Fix State Hashing (Enable Learning)
- Replace unbounded `momentum_bin` with coarse categorical bins:
  - LOW (< 0.5), MEDIUM (0.5-2.0), HIGH (> 2.0)
- Consider reducing z_bin granularity to 1.0 steps instead of 0.5
- Target: ~50-200 unique state buckets (not 80,000)

### P2: Fix Sigma Calculation
- Use residual standard deviation instead of slope std_err:
  ```python
  residuals = y - (slope * x + intercept)
  sigma = np.sqrt(np.sum(residuals**2) / (rp - 2))
  ```
  (Actually this IS what the code does — but it's still std_err from linregress,
  which is `sqrt(MSE / sum((x - x_mean)^2))`, not the same as residual std.
  Consider using `y.std()` or residual std directly.)

### P3: Add Multi-Timeframe
- Generate 15min macro bars for the 21-bar regression (5.25 hour lookback)
- Use 15s micro bars for measurements (volume, cascade, spin)
- This was the original design intent

### P4: Add Cross-Day Rolling
- Prepend last N bars from Day N-1 to Day N's regression window
- Eliminates cold-start at market open

### P5: Account for Trading Costs
- Add spread cost (1-2 ticks) to each trade in simulation
- Add slippage model (1 tick per entry/exit)
- This will show whether the strategy is profitable after costs

---

## 9. APPENDIX: PRECOMPUTE DEBUG LOG PATTERNS

All days show the same structure. Example Day 1:
```
LAGRANGE ZONE DISTRIBUTION:
  L1_STABLE: 2741 (52.1%)
  CHAOS: 1649 (31.3%)
  L3_ROCHE: 454 (8.6%)
  L2_ROCHE: 422 (8.0%)

FILTER PIPELINE:
  Total bars:             5266
  In L2/L3 ROCHE zone:    876 (16.6%)
  structure_confirmed:     292 (5.5%)
  cascade_detected:        2062 (39.2%)
  Both struct+cascade:     253 (4.8%)
  Final structure_ok:      253 (4.8%)

SAMPLE STATES (first 5 with structure_ok=True):
  Bar 263: zone=L3_ROCHE z=-2.55 vel=-1.50 cascade=True struct=True prob=1.00 conf=0.00
  Bar 333: zone=L2_ROCHE z=3.63 vel=3.50 cascade=True struct=True prob=1.00 conf=0.00
```

Note: `prob=1.00 conf=0.00` on every candidate = unseen state, exploration forced.
The brain has learned nothing that carries forward.

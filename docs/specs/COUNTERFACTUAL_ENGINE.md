# Counterfactual Engine — "What If" for Every Decision

## Context

Every trading decision has alternatives. The system chose one path but could have
chosen others. The counterfactual engine evaluates ALL paths simultaneously, building
a multi-dimensional optimization surface from actual market data.

This is the architectural foundation for the goat brain (adaptive real-time system
that traverses unseen regimes). The goat doesn't need historical training data because
it evaluates every foothold in real-time.

## Core Concept

Every bar generates counterfactual branches for every tunable parameter:

```
Bar N: peak fires, 1m sensor blocks LONG entry
  REAL PATH:     skip (no trade)
  PHANTOM 1:     enter LONG with 2/3 gate → track outcome
  PHANTOM 2:     enter LONG with 3/3 gate → track outcome
  PHANTOM 3:     enter LONG with no gate → track outcome
  PHANTOM 4:     enter SHORT (peak reversed) → track outcome

Bar M: entered LONG, peak_giveback exits at +8t (50% threshold)
  REAL PATH:     exit at +8t
  PHANTOM 1:     what if giveback=30%? → would exit at +12t
  PHANTOM 2:     what if giveback=20%? → would exit at +18t
  PHANTOM 3:     what if regime_decay didn't fire? → held to +22t
  PHANTOM 4:     what if cat said EXHAUSTING? → exited at +6t
  PHANTOM 5:     what if SL was 50t instead of 70t? → same result
```

## Architecture

### Phantom Trade Worker

Lightweight object that tracks a hypothetical trade without affecting the real system.
No orders, no position state, no brain updates. Just price tracking.

```python
class PhantomTrade:
    """Simulated trade for counterfactual evaluation."""

    # Identity
    spawn_bar: int          # bar index where phantom was created
    spawn_reason: str       # 'skip_sensor_gate', 'alt_giveback_30', etc.
    parameter_name: str     # which parameter is being tested
    parameter_value: float  # the alternative value being tested

    # Trade state (mirrors PositionState but read-only)
    direction: str          # LONG or SHORT
    entry_price: float
    entry_bar: int

    # Tracking (updated each bar from _states_map or live bars)
    current_price: float
    mfe_ticks: float        # max favorable excursion
    mae_ticks: float        # max adverse excursion
    bars_held: int
    peak_bar: int           # bar where MFE was reached

    # Exit simulation (runs the SAME exit cascade with different params)
    exited: bool
    exit_bar: int
    exit_price: float
    exit_reason: str
    exit_pnl_ticks: float

    # Verdict
    outcome: str            # PROFITABLE, UNPROFITABLE, BREAKEVEN
    regret_vs_real: float   # phantom_pnl - real_pnl (positive = real was worse)
```

### Counterfactual Manager

Manages the pool of phantom workers. Spawns on every skip and every trade.
Evaluates completed phantoms and updates parameter recommendations.

```python
class CounterfactualManager:
    """Spawns and manages phantom trades for what-if analysis."""

    # Active phantoms (being tracked)
    active_phantoms: List[PhantomTrade]

    # Completed phantoms (evaluated)
    completed_phantoms: deque(maxlen=5000)

    # Parameter optimization surface
    # Key: (parameter_name, parameter_value)
    # Value: {n_phantoms, avg_pnl, avg_mfe, avg_mae, win_rate}
    param_surface: Dict[Tuple[str, float], dict]

    def on_skip(self, bar_index, price, direction, skip_reason, state):
        """Peak was skipped — spawn phantoms for alternative gate thresholds."""
        # Phantom: what if we entered anyway?
        self.active_phantoms.append(PhantomTrade(
            spawn_bar=bar_index,
            spawn_reason=f'skip_{skip_reason}',
            parameter_name='sensor_gate',
            parameter_value=0,  # no gate
            direction=direction,
            entry_price=price,
            entry_bar=bar_index,
        ))

    def on_entry(self, bar_index, price, direction, trade_params):
        """Trade entered — spawn phantoms for alternative exit thresholds."""
        for gb_pct in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
            self.active_phantoms.append(PhantomTrade(
                spawn_bar=bar_index,
                parameter_name='giveback_pct',
                parameter_value=gb_pct,
                direction=direction,
                entry_price=price,
                entry_bar=bar_index,
            ))

    def on_bar(self, bar_index, price, high, low, state, exit_engine):
        """Update all active phantoms with current bar data."""
        for phantom in self.active_phantoms[:]:  # copy for safe removal
            phantom.bars_held += 1

            # Update MFE/MAE
            if phantom.direction == 'LONG':
                fav = (high - phantom.entry_price) / tick_size
                adv = (phantom.entry_price - low) / tick_size
            else:
                fav = (phantom.entry_price - low) / tick_size
                adv = (high - phantom.entry_price) / tick_size
            phantom.mfe_ticks = max(phantom.mfe_ticks, fav)
            phantom.mae_ticks = max(phantom.mae_ticks, adv)

            # Check if phantom would exit with its parameter setting
            # (simplified — full exit cascade evaluation)
            if phantom.parameter_name == 'giveback_pct':
                gave_back = phantom.mfe_ticks - fav
                if phantom.mfe_ticks > 0 and gave_back / phantom.mfe_ticks >= phantom.parameter_value:
                    phantom.exited = True
                    phantom.exit_bar = bar_index
                    phantom.exit_price = price
                    phantom.exit_pnl_ticks = fav
                    phantom.exit_reason = f'giveback_{phantom.parameter_value}'

            # Max hold: 80 bars (20 min) — phantom expires
            if phantom.bars_held >= 80:
                phantom.exited = True
                phantom.exit_bar = bar_index
                phantom.exit_price = price
                phantom.exit_pnl_ticks = fav if phantom.direction == 'LONG' else -fav
                phantom.exit_reason = 'phantom_expired'

            if phantom.exited:
                self.active_phantoms.remove(phantom)
                self.completed_phantoms.append(phantom)
                self._update_surface(phantom)

    def _update_surface(self, phantom):
        """Update the parameter optimization surface with completed phantom."""
        key = (phantom.parameter_name, phantom.parameter_value)
        if key not in self.param_surface:
            self.param_surface[key] = {
                'n': 0, 'total_pnl': 0, 'wins': 0,
                'total_mfe': 0, 'total_mae': 0,
            }
        s = self.param_surface[key]
        s['n'] += 1
        s['total_pnl'] += phantom.exit_pnl_ticks
        s['total_mfe'] += phantom.mfe_ticks
        s['total_mae'] += phantom.mae_ticks
        s['wins'] += 1 if phantom.exit_pnl_ticks > 0 else 0

    def get_optimal(self, parameter_name: str) -> float:
        """Get the optimal value for a parameter based on phantom outcomes."""
        best_val = None
        best_avg = -999
        for (pname, pval), stats in self.param_surface.items():
            if pname != parameter_name or stats['n'] < 20:
                continue
            avg = stats['total_pnl'] / stats['n']
            if avg > best_avg:
                best_avg = avg
                best_val = pval
        return best_val

    def report(self) -> str:
        """Generate counterfactual summary report."""
        lines = ['COUNTERFACTUAL ANALYSIS', '=' * 60]
        for (pname, pval), stats in sorted(self.param_surface.items()):
            n = stats['n']
            if n < 5:
                continue
            avg_pnl = stats['total_pnl'] / n
            wr = stats['wins'] / n * 100
            avg_mfe = stats['total_mfe'] / n
            lines.append(
                f'  {pname}={pval:<6.2f}  n={n:>5}  avg={avg_pnl:>+6.1f}t  '
                f'WR={wr:.0f}%  MFE={avg_mfe:.1f}t'
            )
        return '\n'.join(lines)
```

## Parameters to Evaluate

### Entry Gates
| Parameter | Sweep Values | Current Default |
|-----------|-------------|-----------------|
| sensor_gate_min_opposing | 0, 1, 2, 3 | 2 (2/3 sensors must oppose) |
| adx_chop_threshold | 0, 5, 10, 15, 20 | 2 (effectively disabled) |
| buildup_min_bars | 0, 1, 2, 3, 4, 5 | 3 |
| cat_regime_filter | True, False | True (when --cat) |

### Exit Thresholds
| Parameter | Sweep Values | Current Default |
|-----------|-------------|-----------------|
| giveback_pct | 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70 | 0.50 |
| vol_drop_threshold | 0.20, 0.30, 0.40, 0.50, 0.60, 0.70 | 0.50 |
| regime_decay_suppress_bars | 0, 2, 4, 6, 8 | 0 |
| peak_state_min_sensors | 1, 2, 3, 4 | 2 |
| trailing_stop_activation | 10, 20, 30, 40, 60 | 40 |

### Direction
| Parameter | Sweep Values | Current Default |
|-----------|-------------|-----------------|
| use_peak_direction | True, False | False (uses cascade) |
| brain_bias_weight | 0, 0.3, 0.5, 0.7, 1.0 | ~0.6 |

## Output

### Per-Run Report
```
COUNTERFACTUAL ANALYSIS (310 trading days, 34,446 real trades, 15,220 skips)
========================================================================

ENTRY GATE: sensor_gate_min_opposing
  0 (no gate):   n=15220  avg=+2.1t  WR=58%  MFE=14.2t  ← BEST
  1 (any 1):     n=12400  avg=+2.8t  WR=61%  MFE=15.1t
  2 (current):   n= 8900  avg=+3.2t  WR=63%  MFE=15.8t
  3 (all 3):     n= 5200  avg=+3.9t  WR=66%  MFE=16.4t  ← HIGHEST QUALITY

EXIT: giveback_pct
  0.20:          n=34446  avg=+8.1t  WR=72%  MFE=12.0t  ← BEST PNL
  0.30:          n=34446  avg=+7.2t  WR=70%  MFE=14.5t
  0.50 (current):n=34446  avg=+6.1t  WR=67%  MFE=16.8t
  0.70:          n=34446  avg=+4.8t  WR=64%  MFE=19.2t

RECOMMENDATION: tighten giveback to 0.20, loosen gate to 1/3
  Expected impact: +$3.50/trade (+58% over current $6.09/trade)
```

### Adaptive Mode (Live)
In live, the counterfactual manager runs continuously. Every 100 completed
phantoms, it recommends parameter adjustments. The system can auto-apply
(goat mode) or flag for human review (cat mode).

## Integration Path

1. **IS-only (synchronous)**: Process phantoms from _states_map. No threading.
   Produces the report at end of IS. Parameters carry to OOS/live.

2. **OOS (synchronous)**: Same as IS. Validates IS-learned parameters hold OOS.

3. **Live (async workers)**: Background thread processes phantoms after 80 bars.
   Updates parameters in-flight. This IS the goat brain.

## Evolution

```
Cat:    regime classifier (rolling deltas) → decides IF to trade
Crow:   seed matching (k-NN) → decides WHAT to expect
Monkey: CNN (learned patterns) → finds non-linear edges
Goat:   counterfactual engine → adapts ALL parameters in real-time
```

The goat doesn't replace the others — it TUNES them. The cat classifies regimes,
the crow matches seeds, the monkey predicts outcomes, and the goat adjusts all
their thresholds based on continuous what-if evaluation.

## Risk Assessment

**What could go wrong:**
- Phantom trades are simplified (no slippage, no fill latency, no position sizing)
- Over-optimization: fitting to noise in recent phantoms
- Computation cost: N_skips × N_param_values phantom trades per bar

**Mitigations:**
- Minimum 20 phantoms before recommending a parameter change
- 20% step size toward optimal (smooth adaptation, no sudden jumps)
- Phantom pool capped at 5,000 (oldest discarded)
- Parameters have hard bounds (can't go below safety minimums)
- Human review mode (flag recommendations, don't auto-apply)

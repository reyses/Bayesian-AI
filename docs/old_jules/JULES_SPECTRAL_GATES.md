# Jules Spec: Spectral Entry & Exit Gates (Phases A-B-C)

## Overview

Add Fourier spectral analysis to every bar, then use it for:
- **Phase A**: Compute wave phase + period on every bar (`core/spectral.py`)
- **Phase B**: Entry phase gate — don't enter too late in the wave
- **Phase C**: Bayesian exit probability — continuous P(reversal) during hold

These work WITHIN the current architecture (no cascade rewrite needed).
Phase D (progressive cascade elimination) is a separate fork.

---

## Current Problem (data-backed)

From the 2026-02-22 forward pass (Jan-Oct 2025, 4,837 trades):
- **99.5% of bars have ZERO detection** — model is blind
- When it sees something, **15 signals fire at once** on the same bar
- Of trades taken: **exits too early** = $278K left on table
- **Reversed after correct entry** = $33K (entering too late in the wave)
- Trail_stop is 40% of exits with avg PnL near zero — shaken out by noise

The spectral gates attack all three: enter at the right phase, hold through
the structural wave, exit when probability of continuation drops.

---

## Phase A: Spectral Module

### New File: `core/spectral.py`

```python
@dataclass(frozen=True)
class SpectralState:
    """Spectral analysis of the z-score waveform at a given bar."""
    dominant_period_bars: float    # T: dominant wavelength in bars (e.g., 80 bars = 20min)
    dominant_frequency: float      # 1/T in cycles per bar
    current_phase: float           # 0.0 = wave trough, 0.5 = wave peak, 1.0 = next trough
    spectral_power: float          # 0-1: how "clean" the dominant frequency is (SNR)
    amplitude: float               # Amplitude of dominant component in z-score units
    velocity_decay_ratio: float    # zeta: damping ratio of velocity envelope post-entry
                                   # < 1.0 = underdamped (oscillating), >= 1.0 = critically damped
    energy_remaining: float        # 0-1: estimated kinetic energy remaining in current impulse
```

### Function: `compute_spectral_state()`

```python
def compute_spectral_state(
    z_scores: np.ndarray,           # Rolling window of z-score values
    velocities: np.ndarray,         # Rolling window of particle_velocity values
    fft_window_bars: int = 120,     # Tunable: FFT window size
    entry_bar_offset: int = 0,      # How many bars ago the trade was entered (for decay calc)
) -> SpectralState:
```

**Algorithm:**
1. Apply Hanning window to `z_scores[-fft_window_bars:]`
2. `np.fft.rfft()` → power spectrum
3. Find peak frequency (excluding DC component and frequencies below 10-bar period)
4. `dominant_period_bars = 1 / peak_freq` (in bar units)
5. `spectral_power = peak_power / total_power` (0-1, how dominant the peak is)
6. `current_phase = extract_phase(z_scores, peak_freq)` — instantaneous phase via Hilbert transform
   - Phase 0.0 = trough (ideal LONG entry), 0.5 = peak (ideal SHORT entry)
   - For LONG: phase near 0 = early, near 0.5 = late
   - For SHORT: phase near 0.5 = early, near 1.0 = late
7. `amplitude = 2 * sqrt(peak_power) / fft_window_bars`
8. If `entry_bar_offset > 0`:
   - Fit exponential decay `v(t) = A * exp(-zeta * t)` to velocity samples since entry
   - `velocity_decay_ratio = zeta` (the damping ratio)
   - `energy_remaining = exp(-zeta * entry_bar_offset)` (normalized 0-1)
9. Else: `velocity_decay_ratio = 0.0`, `energy_remaining = 1.0`

**Dependencies:** numpy, scipy.signal (Hilbert transform). Both already installed.

### Integration: Attach to bar loop

In `orchestrator.py`, the main bar loop (`for row in df_15s.itertuples()`, ~line 704):

```python
# After computing states for the day, build rolling z-score array
# z_scores and velocities come from _states_map (already computed by batch_compute_states)
# Maintain a rolling buffer of the last fft_window_bars z-scores and velocities
```

**Important:** `compute_spectral_state()` must be called ONCE per bar (not per candidate).
Cache the result and reuse for all candidates on the same bar.

The `fft_window_bars` parameter comes from the template being evaluated (Phase B/C).
Since different templates may want different windows, compute with a DEFAULT window
(e.g., 120 bars) for the bar-level cache. Templates can override in their gate logic.

### Performance

FFT on 120 floats: ~0.01ms per bar. 530K bars/month = ~5s total overhead. Negligible.

---

## Phase B: Entry Phase Gate (Gate 4)

### Concept

After existing Gate 3 (conviction) passes, check: "Are we too late in the wave?"

If the dominant wave's current phase is past the optimal entry window, SKIP.
This prevents entering mid-move when most of the structural wave is already spent.

### New Parameters (per-template, tuned by Optuna)

Add to `PatternTemplate` and `params` dict:

| Parameter | Key | Type | Range | Default | Seed Source |
|-----------|-----|------|-------|---------|-------------|
| FFT window | `fft_window_bars` | int | 40-200 | 120 | 2x median MFE timing |
| Min spectral power | `min_spectral_power` | float | 0.1-0.9 | 0.3 | Fixed default |
| Max entry phase | `max_entry_phase` | float | 0.1-0.6 | 0.4 | Fixed default |

### Integration Point

In `orchestrator.py`, AFTER Gate 3 passes and `best_candidate` is selected (~line 1170):

```python
# --- Gate 4: Spectral Entry Phase ---
# Only applies if spectral power is strong enough (clean signal)
_spectral = cached_spectral_state  # computed once per bar
_fft_window = params.get('fft_window_bars', 120)
_min_power = params.get('min_spectral_power', 0.3)
_max_phase = params.get('max_entry_phase', 0.4)

if _spectral.spectral_power >= _min_power:
    # Normalize phase relative to trade direction
    # LONG: ideal phase near 0 (trough), late near 0.5 (peak)
    # SHORT: ideal phase near 0.5 (peak), late near 1.0 (trough)
    if side == 'long':
        _effective_phase = _spectral.current_phase  # 0=trough=ideal, 0.5=peak=late
    else:
        _effective_phase = abs(_spectral.current_phase - 0.5)  # 0=peak=ideal, 0.5=trough=late

    if _effective_phase > _max_phase:
        skip_spectral += 1
        # Log to signal_log with gate='gate4_late'
        continue  # Skip — too late in the wave
```

### Signal Log

Add `gate4_late` as a new gate value in decision_matrix_records. Log:
- `spectral_phase`: the computed phase at entry
- `spectral_power`: how clean the signal was
- `spectral_period`: dominant period in bars

These go into the signal_log CSV as new columns for post-analysis.

### Report Section

Add after the detection funnel:

```
  SPECTRAL ENTRY GATE:
    Signals rejected (too late in wave):  XXX  (X.X% of candidates)
    Avg phase at entry (traded):          0.XX  (0=ideal, 0.5=late)
    Avg phase at entry (rejected):        0.XX
    Oracle $ saved by gate:               $XX,XXX  (rejected signals that reversed)
    Oracle $ lost by gate:                $XX,XXX  (rejected signals that would have won)
```

---

## Phase C: Bayesian Exit Probability

### Concept

Replace static trail_stop / TP / MAX_HOLD with a continuous probability model.
Every bar during a trade, compute:

```
P(trade reaches target | current_state, bars_held, velocity_decay, spectral_phase)
```

When P drops below a threshold → exit. When P is high → widen trail / hold.

### New Parameters (per-template, tuned by Optuna)

| Parameter | Key | Type | Range | Default | Seed Source |
|-----------|-----|------|-------|---------|-------------|
| Damping threshold | `damping_threshold` | float | 0.5-2.0 | 1.0 | Fixed default |
| Velocity window | `velocity_window_bars` | int | 10-60 | 20 | From template MFE timing |
| Min profit for exhaust | `min_profit_ticks` | int | 2-20 | 4 | p25_mae_ticks |
| Exit probability floor | `exit_prob_floor` | float | 0.15-0.50 | 0.30 | Fixed default |
| Expected duration bars | `expected_duration_bars` | int | 10-300 | - | From template median hold |

### Exit Probability Model

New function in `core/spectral.py`:

```python
def compute_exit_probability(
    spectral: SpectralState,
    bars_held: int,
    expected_duration_bars: int,
    floating_pnl_ticks: float,
    template_wr: float,            # Historical win rate for this template
    template_mean_mfe_ticks: float,
) -> float:
    """
    Returns P(continuation) in [0, 1].
    High = trade likely to keep running. Low = trade exhausting.
    """
```

**Factors combined (multiplicative Bayesian):**

1. **Time decay**: `P_time = 1 - (bars_held / expected_duration_bars)^2`
   - Starts at 1.0, drops as we approach expected duration
   - Quadratic decay (slow start, fast end)

2. **Kinetic energy**: `P_energy = spectral.energy_remaining`
   - Direct from the velocity decay fit (Phase A)
   - Drops as momentum fades

3. **Wave phase alignment**: `P_phase = cos(pi * phase_distance)^2`
   - If still in favorable phase region → high. Crossing into adverse phase → low.

4. **Profit capture**: `P_capture = sigmoid(floating_pnl / mean_mfe - 0.5)`
   - Already captured 80% of expected MFE? → P goes down (take profit)
   - Still early in the move? → P stays high

5. **Combined**: `P = P_time * P_energy * P_phase * base_wr_prior`
   - Where `base_wr_prior = template_wr` (prior from historical data)

### Integration: WaveRider exit_signal

In the orchestrator bar loop, where exit signals are computed (~line 830-860),
compute `exit_probability` and pass it via the `exit_signal` dict:

```python
_spectral = compute_spectral_state(z_buffer, v_buffer,
    fft_window_bars=active_template_params.get('fft_window_bars', 120),
    entry_bar_offset=bars_in_trade)

_exit_prob = compute_exit_probability(
    spectral=_spectral,
    bars_held=bars_in_trade,
    expected_duration_bars=active_template_params.get('expected_duration_bars', 100),
    floating_pnl_ticks=floating_pnl / point_value,
    template_wr=active_template_wr,
    template_mean_mfe_ticks=active_template_mfe,
)

exit_signal['exit_probability'] = _exit_prob
exit_signal['spectral_state'] = _spectral
```

### WaveRider Changes

In `wave_rider.py` `update_trail()`, add a new exit condition:

```python
# Spectral exit: kinetic exhaustion
if exit_signal and 'exit_probability' in exit_signal:
    _prob = exit_signal['exit_probability']
    _spectral = exit_signal.get('spectral_state')
    _prob_floor = self._params.get('exit_prob_floor', 0.30)
    _min_profit = self._params.get('min_profit_ticks', 4) * self.point_value

    # Only fire if profitable (never exit underwater on probability alone)
    if self.position.floating_pnl >= _min_profit and _prob < _prob_floor:
        return {
            'should_exit': True,
            'exit_price': current_price,
            'exit_reason': 'spectral_exhaust',
            'pnl': self._calculate_pnl(current_price),
        }

    # Also: if energy_remaining is very low AND damping is critical
    if (_spectral and _spectral.velocity_decay_ratio >= self._params.get('damping_threshold', 1.0)
        and self.position.floating_pnl >= _min_profit):
        return {
            'should_exit': True,
            'exit_price': current_price,
            'exit_reason': 'kinetic_exhaust',
            'pnl': self._calculate_pnl(current_price),
        }

    # Modulate trail width based on probability
    if _prob > 0.7:
        exit_signal['widen_trail'] = True
        exit_signal['reason'] = 'high_continuation_prob'
    elif _prob < 0.4:
        exit_signal['tighten_trail'] = True
        exit_signal['reason'] = 'low_continuation_prob'
```

### New Exit Reasons

- `spectral_exhaust`: P(continuation) dropped below `exit_prob_floor` while profitable
- `kinetic_exhaust`: Velocity damping hit critical threshold while profitable

Both are logged in `oracle_trade_log.csv` as `exit_reason` values.

### Trade Log Columns

Add to oracle_trade_log.csv:
- `entry_wave_phase`: Phase at entry (0-1)
- `entry_spectral_power`: Spectral clarity at entry
- `entry_dominant_period`: Dominant period in bars at entry
- `exit_wave_phase`: Phase at exit
- `exit_energy_remaining`: Kinetic energy at exit
- `exit_probability`: Final P(continuation) at exit
- `exit_damping_ratio`: Velocity decay ratio at exit

---

## Phase 3 Optuna Integration

### Search Space Extension

In the template optimization (currently in `orchestrator_worker.py` `_optimize_template_task`
and `doe_parameter_generator.py`), add the 7 new parameters to the Optuna search space:

```python
# Spectral entry gate
trial.suggest_int('fft_window_bars', 40, 200, step=10)
trial.suggest_float('min_spectral_power', 0.1, 0.9)
trial.suggest_float('max_entry_phase', 0.1, 0.6)

# Spectral exit gate
trial.suggest_float('damping_threshold', 0.5, 2.0)
trial.suggest_int('velocity_window_bars', 10, 60, step=5)
trial.suggest_int('min_profit_ticks', 2, 20)
trial.suggest_float('exit_prob_floor', 0.15, 0.50)
```

### Analytical Seeds (Phase 2)

In `_analytical_exits()`, derive seed values from oracle stats:

```python
# FFT window: ~2x the median MFE timing (bars to peak)
median_mfe_bars = template.median_mfe_bars or 60
fft_window_seed = min(200, max(40, int(median_mfe_bars * 2)))

# Expected duration: from oracle hold distribution
expected_duration_seed = template.median_hold_bars or 100

# Min profit for exhaust: from p25 MAE
min_profit_seed = max(2, int(template.p25_mae_ticks * 0.5))
```

These seeds narrow the Optuna search space (warm-start around analytical values).

---

## Rolling Buffers

The spectral module needs rolling z-score and velocity arrays. These must persist
across bars within a day, reset at day boundaries.

### Implementation

In the orchestrator forward pass, before the bar loop:

```python
# Rolling buffers for spectral analysis (persist within day, reset per day)
_z_buffer = np.zeros(200, dtype=np.float64)  # max window size
_v_buffer = np.zeros(200, dtype=np.float64)
_buf_idx = 0
_buf_filled = 0  # how many bars have been written
```

In the bar loop, after belief_network.tick_all():

```python
# Update spectral buffers from current state
_cur_state = _states_map.get(_bar_i)
if _cur_state:
    _z_buffer[_buf_idx % 200] = _cur_state['state'].z_score
    _v_buffer[_buf_idx % 200] = _cur_state['state'].particle_velocity
    _buf_idx += 1
    _buf_filled = min(_buf_filled + 1, 200)
```

Pass to `compute_spectral_state()` as circular buffer sliced to `_buf_filled` length.

---

## Files Modified

| File | Changes |
|------|---------|
| `core/spectral.py` | **NEW** — SpectralState, compute_spectral_state(), compute_exit_probability() |
| `training/orchestrator.py` | Rolling buffers, Gate 4 spectral entry check, exit_probability in exit_signal, new report section, new trade log columns |
| `training/wave_rider.py` | spectral_exhaust + kinetic_exhaust exit conditions, probability-modulated trail width |
| `training/orchestrator_worker.py` | Add 7 spectral params to Optuna search space |
| `training/doe_parameter_generator.py` | Extend param_ranges with spectral params |
| `core/three_body_state.py` | (optional) Add spectral fields to ThreeBodyQuantumState |

**Estimated scope:** ~400 lines new code, ~150 lines modified. 6 files touched.

---

## Verification

1. **Unit test spectral module**: Known sine wave input → verify phase, period, power
2. **Forward pass regression**: Run with default params (no spectral gating active),
   verify PnL matches previous run exactly (spectral is computed but not gating)
3. **Spectral gating ON**: Run with moderate params, check:
   - Gate 4 rejects some late entries
   - spectral_exhaust / kinetic_exhaust appear as exit reasons
   - Report shows spectral entry/exit stats
4. **Optuna integration**: Run Phase 3 on 1 template, verify spectral params appear
   in the optimized output

---

## Scope Note

This spec covers Phases A, B, C only. Phase D (progressive cascade elimination —
rewriting scan_day_cascade to narrow candidates top-down instead of broadcasting)
is a separate fork and separate spec.

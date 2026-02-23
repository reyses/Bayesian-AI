# Spectral Gates: Physics-Based Trade Management

## Overview

Spectral Gates are a set of physics-based exit conditions implemented in the `_fast_sim_loop` function of `training/orchestrator_worker.py`. They leverage spectral analysis (Fourier Transform) and kinetic energy metrics (Laplace Damping) to align trade duration with the market's natural rhythm and momentum.

These gates aim to solve two common trading errors:
1.  **Premature Ejaculation (Early Exit)**: Closing a winning trade before the market cycle has completed.
2.  **Bag Holding (Late Exit)**: Holding a position after momentum has exhausted and a reversal is imminent.

## The Fourier Gate (Minimum Hold Time)

**Purpose**: To enforce a minimum holding period based on the dominant market cycle, preventing early exits on noise or minor fluctuations.

**Mechanism**:
- Calculates the `dominant_cycle_period` using Fast Fourier Transform (FFT) on a sliding window of Z-scores (default window: 60 bars).
- Determines a `min_hold_seconds` threshold equal to half the dominant cycle period (`period / 2.0`).

**Logic**:
The gate effectively "locks" the exit door (preventing Take Profit and Time Exit) if:
- The trade duration is less than `min_hold_seconds`.
- AND the current PnL is better than -50% of the Stop Loss (i.e., not in deep drawdown).

**Code Implementation**:
```python
# Spectral Evaluation
min_hold_seconds = periods[i] / 2.0

# FOURIER GATE: Prohibit TP/Exit before half-cycle completes (unless SL hits)
if duration < min_hold_seconds and pnl > -(stop_loss / 2.0):
    continue
```

**Behavior**:
- **Winning/Small Loss**: The trade is held until at least half the cycle completes.
- **Deep Loss (> 50% to SL)**: The gate unlocks, allowing standard Time Exits (or other logic) to intervene, though the hard Stop Loss usually handles this first.

## The Laplace Gate (Kinetic Exhaustion)

**Purpose**: To detect "Kinetic Exhaustion"—a state where price momentum is critically damped and a reversal is statistically probable—allowing for an early profit-taking exit.

**Mechanism**:
- Calculates a `damping_factor` by fitting a linear regression to the log of velocity peaks over a sliding window (default window: 20 bars).
- A high damping factor (> 0.8) indicates that the "kinetic energy" of the price movement is dissipating rapidly.

**Logic**:
The gate triggers an immediate exit if:
- The trade is currently profitable (`pnl > 0`).
- AND the `damping_factor` exceeds the `KINETIC_DAMPING_EXIT_THRESHOLD` (0.8).

**Code Implementation**:
```python
# LAPLACE GATE: Exit if kinetic energy is critically damped
if pnl > 0 and dampings[i] > KINETIC_DAMPING_EXIT_THRESHOLD:
    return price, pnl, 1, curr_time, 4, duration # 4 = KINETIC_EXHAUSTION
```

**Exit Code**:
- Returns exit reason code `4` (`KINETIC_EXHAUSTION`).

## Priority Hierarchy

The exit logic in `_fast_sim_loop` follows a strict hierarchy of checks:

1.  **Hard Stop Loss**:
    - **Condition**: `pnl <= -stop_loss`
    - **Action**: Immediate Exit (Loss).
    - **Override**: Highest priority; safety first.

2.  **Fourier Gate**:
    - **Condition**: `duration < period/2` AND `pnl > -0.5*SL`
    - **Action**: **SKIP** subsequent checks (Force Hold).
    - **Note**: This prevents Laplace, Take Profit, and Time Exit from triggering.

3.  **Laplace Gate**:
    - **Condition**: `pnl > 0` AND `damping > 0.8`
    - **Action**: Immediate Exit (Profit).
    - **Reason**: Kinetic Exhaustion.

4.  **Take Profit**:
    - **Condition**: `pnl >= take_profit`
    - **Action**: Immediate Exit (Profit).

5.  **Time Exit**:
    - **Condition**: `duration >= max_hold`
    - **Action**: Immediate Exit (Time expiration).

## Constants & Configuration

The following constants in `training/orchestrator_worker.py` control the sensitivity of the gates:

- `Z_SCORE_CYCLE_WINDOW = 60`: Lookback window for FFT cycle detection.
- `VELOCITY_DAMPING_WINDOW = 20`: Lookback window for kinetic damping calculation.
- `KINETIC_DAMPING_EXIT_THRESHOLD = 0.8`: Critical damping threshold for the Laplace Gate.

## Mathematical Basis

The underlying physics calculations are performed in `core/physics_utils.py`:

- **`extract_dominant_cycle(z_scores)`**: Uses `scipy.fft` to find the peak frequency in the Z-score series. The period is `1.0 / peak_frequency`.
- **`calculate_kinetic_damping(velocity_vector)`**: Fits a line to `log(abs(velocity))` vs time. The slope represents the exponential decay rate (damping).
